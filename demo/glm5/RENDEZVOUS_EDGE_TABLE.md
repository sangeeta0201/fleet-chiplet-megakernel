# The ten rendezvous as EDGES — and why "ten vs TileRT's two" is a category error

Date 2026-08-23, branch `merge-rocshmem`. **Measured baseline attached** (n=3,
mean 10.571 ms/iter, §5); the structural claim in §1 is a fact read off
`~/TileRT`, not a price.

The board's premise was: *the same 9 ops run on the reference with TWO fused
all-reduces; we run TEN rendezvous; that difference is the last 2 ms on this
machine.* This file asks the joint question the two failed re-shards never asked
— what does each edge actually require — and the first thing that falls out is
that **the premise compares two different quantities.**

---

## 1. The category error: TileRT pays ELEVEN device-wide sync points, not two

TileRT is **not** a megakernel. Every registered op lowers to one or more
`torch.ops.tilert.*_op(...)` calls — separate kernel launches. Read
`tilert/models/deepseek_v3_2/ops/rmsnorm_projx_wqkva.py:473`: the single
registered op `rmsnorm_projx_wqkva` issues **two** launches, `_rmsnorm_quant`
then `_projx_wqkva`. A kernel-launch boundary *is* a device-wide dependency
barrier; it is the exact thing a persistent megakernel exists to remove.

Per GLM-5 MoE decode layer, per rank, on a `PureMlaV2` device (GPU 1–7):

Verified launch-by-launch with `python3 demo/glm5/count_tilert_launches.py`,
which greps `~/TileRT` read-only and fails loudly if a symbol moves. All eleven
resolve inside the **GLM-5-specific** `tilert/models/glm_5/_dsa_v32/ops/` tree
(not the DeepSeek fallbacks), except the attention itself:

| # | TileRT launch | source (rel. to `~/TileRT`) | scope |
|---|---|---|---|
| 1 | `rmsnorm_quant_op` | `.../glm_5/_dsa_v32/ops/rmsnorm_quant.py:43` | device |
| 2 | `projx_wqkva_op` | `.../glm_5/_dsa_v32/ops/projx_wqkva.py:37` | device |
| 3 | `rmsnorm_proj_qb_op` | `.../glm_5/_dsa_v32/ops/rmsnorm_projq_wqb.py:32` | device |
| 4 | `rmsnorm_kv_op` | `.../glm_5/_dsa_v32/ops/rmsnorm_kv.py:29` | device |
| 5 | `projq_wqb_op` | `.../glm_5/_dsa_v32/ops/projq_wqb.py:35` | device |
| 6 | `flash_sparse_mla_op` | `.../deepseek_v3_2/ops/flash_sparse_mla.py:76` | device |
| 7 | `projo_wkvb_op` | `.../glm_5/_dsa_v32/ops/projo_wkvb.py:33` | device |
| 8 | **`unproj_o_allreduce_op`** | `.../glm_5/_dsa_v32/ops/unproj_o_allreduce.py:34` | device **+ cross-rank** |
| 9 | `rmsnorm_expert_proj_op` | `.../glm_5/_dsa_v32/ops/rmsnorm_expert_proj.py:159` | device |
| 10 | `expert_select_up_gate_silu_op` | `.../glm_5/_dsa_v32/ops/expert_sel_up_gate_silu.py:42` | device |
| 11 | **`expert_down_allreduce_op`** | `.../glm_5/_dsa_v32/ops/expert_down_allreduce.py:35` | device **+ cross-rank** |

**11 device-wide sync points, of which 2 are also cross-rank.**

Ours, from `price_xcd_narrowing.py`'s `BARRIERS`: **10 rendezvous, of which 2
are cross-rank** (`glm-layer-pays-two-cross-rank-rendezvous`: the EP fold and
the QB_TP peer wait).

| | ours | TileRT |
|---|---|---|
| device-wide sync points / layer | **10** | **11** |
| cross-rank collectives / layer | **2** | **2** |

The "two" in the original diff is TileRT's **cross-rank** count. We match it
exactly. On the intra-device count we are already one *ahead*. **The rendezvous
count is not where the 5.3× lives, and no re-sharding of ours can find it
there** — there is nothing to close.

## 2. The edge table — producer → consumer, the element, and the minimum scope

Shapes from `/home/claudeuser/models/glm5-mxfp4/config.json`: hidden 6144,
`q_lora_rank` 2048, `kv_lora_rank` 512, `qk_rope_head_dim` 64,
`qk_nope_head_dim` 192, `v_head_dim` 256, 64 heads, `moe_intermediate_size`
2048, 8 routed + 1 shared expert.

| # | rendezvous | edge | the element worker *w* reads, and who *v* is | min scope **required** | set by |
|---|---|---|---|---|---|
| E1 | `entry_bar` (9) | MoE(L−1) → qkv_a(L) | `x[0..6143]`, the whole residual row; *w*'s qkv_a tile contracts over all 6144. *v* = every W2 tile of L−1, atomically accumulating, on all 232 workers + 8 EP peer planes | **device + cross-rank** | **contraction**: RMSNorm and the GEMV both span the full hidden row |
| E2 | `qkv_barrier` (17) | qkv_a → q_b | `q_a[0..2047]`; *w*'s q_b tile computes `W_qb[h] @ q_a` over **all 2048**. *v* = XCD *x* owns `q_a[256x, 256x+256)` | **device** | **low-rank bottleneck**: q_a is a rank-2048 factor — every q_b output depends on every q_a element |
| E3 | `qb_barrier` (19) | q_b/W_UK → decode | `q_absorbed[h][0..511]` for all `h` in *w*'s q_group. *v* = rank *p* produced only heads `[8p, 8p+8)` (`QB_TP_HEADS = 64/8`) | **cross-rank** | **task graph**: QB_TP head shard (8) vs the decode's 16-head q_group |
| E4 | `decode_barrier` (21) | decode → merge | partial `(o, m, l)` for `(q_group g, kv_chunk c)`; merge for *g* contracts over all 16 chunks. *v* = the worker owning `(g, c)` | **XCD-pair** | **task graph**: the kv-chunk split is ours, chosen for parallelism |
| E5 | `attn_release` (23) | merge → W_UV | merged `o[h][0..511]`. W_UV is `o[h] @ W_UV[h]`, **contracting over 512 latent — head-local**. *v* = the merge worker for q_group `h/16` | **XCD-pair** | **task graph**: W_UV is *replicated* (4 tiles/XCD reading all heads) rather than dealt by head |
| E6 | `rel_tree` (25) | — | **not a data edge**; release/self-heal fan-out | mechanism | — |
| E7 | `wuv_barrier` (28) | W_UV → o_proj | `v[h][0..255]` for **all 64 heads**; o_proj contracts over 64×256 = 16384. *v* = every W_UV tile | **device** | **contraction**: o_proj's reduction axis is the full head-concatenated v |
| E8 | `hier_barrier` (30) | o_proj → router | `hidden[0..6143]`; router contracts over all 6144. *v* = rank *p* produced columns `[768p, +768)` | **device + cross-rank** | **contraction** — and this is TileRT's own `unproj_o_allreduce` (#8 above) |
| E9 | `routing poll` (32) | router TopK → MoE dispatch | the 8 selected expert ids + weights; every W13 tile must know whether its expert is live. *v* = the single TopK worker | **device broadcast** | **data dependence** — TileRT pays it as launch #10 |
| E10 | `w13_barrier` (6) | W13/SwiGLU → W2 | `intermediate[e][0..2047]`; *w*'s W2 tile contracts over all 2048. *v* = W13 tiles for *e*, dealt 8/XCD across all 8 XCDs (`gang_moe_linear_mxfp8_mi300.cuh:327` — "an expert's tiles partition its INTERMEDIATE dimension") | **device** | **task graph**: the intermediate deal |

**Four of the ten (E1, E2, E7, E8) are forced by a full contraction** over an
axis that no partition of the hidden dimension survives. Two of those four are
comm points TileRT pays too. One (E6) is not a data edge. **Three (E3, E4, E5,
E10 — four counting E10) are set by the tile deal and are therefore the only
ones the joint question can move.** All four have already been ablated.

## 3. The partitioning question, both halves

### 3a. Attention: can a contiguous hidden slice stay worker-local through q_a → q_b → W_UK → decode → W_UV?

**No, and the reason is MLA's low-rank structure, not the task graph.**

`q_b` is `[64 heads × 256] ← W_qb @ q_a[2048]`. A *rank reduction* is a dense
contraction: every one of q_b's 16384 outputs depends on **every one** of q_a's
2048 inputs. There is no partition of the hidden dimension — or of anything else
— under which a q_b tile reads only what its own worker produced. The same holds
at E7: the 512-wide `kv_lora` latent is shared by all 64 heads, so the decode
cannot be made head-local in the latent.

Locality here is **purchasable but not free**: replicate `q_a` per XCD and E2
becomes XCD-local. Price by the makespan rule (adding work is regime C, 1:1):
qkv_a busy 11.844 → ~95 µs/layer, **+5.4 ms**
(`REGIME_B_SEGMENT_LEGALITY.md` §2a). Replicating only the 512-wide latent is
2.4× qkv_a's weight bytes, ≈ **+1.0 ms**, and it does not touch E2 at all.

### 3b. MoE: router → W13 → W2 with one reduction at the end?

This is exactly TP's column-parallel-up / row-parallel-down, and it is the right
question — W2's output is *already* atomically accumulated into the hidden row,
so an XCD-local K-split would need **no new reduction at all**. It is
nonetheless closed twice, both times by a measurement:

1. **It does not compile.** W2 has `MFMA_ITERS = REDUCTION_SIZE / K_PER_MFMA =
   16` and `static_assert(MFMA_ITERS >= 4 && MFMA_ITERS % 4 == 0)`
   (`gang_moe_linear_mxfp8_mi300.cuh:532`). An 8-way split leaves 2. Same
   blocker that killed the MoE EP→TP shard (`9338988`).
2. **The intra-rank version was BUILT and measured: `MPK_W2_KSPLIT=2` is
   +4.12 ms** (`glm-w2-splitk-is-a-large-negative`). Not a roofline — a build.

The other MoE-local form — one whole expert per XCD, which makes E10 XCD-local
with no K-split — costs work instead: the busiest rank holds E[max] ≈ 3 routed +
1 shared = **4.0 experts**, so 4 of 8 XCDs would idle and MoE busy goes
31.07 → ~62 µs/layer, **+2.0 ms at 1:1**.

## 4. Pricing the whole class with the makespan rule — the prize is 0.095 ms

Every decomposition in §3 buys the same thing: it converts a **device**-scope
rendezvous into an **XCD**-scope one. That conversion has been measured
end-to-end, for all ten at once, legality ignored:

> **GPU-wide max arrival − worst-XCD max arrival = 0.000 µs at all ten
> rendezvous.** Narrowing every one of them simultaneously is worth **0.095 ms**
> (`glm-per-xcd-barrier-narrowing-is-zero-by-measurement`).

Every chiplet holds a worker at the global max, so an XCD-scope release lets
seven XCDs out early and the eighth not at all. The corroborating direct
measurement: `MPK_WUV_IN_MERGE` **deleted a whole rendezvous** — ten became
nine, gated 4/4 — for **−0.003 ms** (`e5d1ff5`).

| decomposition | what it buys | what it costs | net |
|---|---|---|---|
| replicate q_a per XCD → E2 XCD-local | ≤ 0.095 | **+5.4 ms** | −5.3 |
| replicate the latent per XCD | 0 (E2 survives) | **+1.0 ms** | −1.0 |
| one expert per XCD → E10 XCD-local | ≤ 0.095 | **+2.0 ms** | −1.9 |
| XCD K-split W13→W2 → E10 vanishes | ≤ 0.095 | **does not compile**; built at 2-way = **+4.12 ms** | −4.0 |
| E3/E4/E5 re-deals | 0.228 / −0.070 / 0.003 | already built | ≤ +0.23 |

**Nothing in the class reaches 1 ms, so nothing here is authorized to build.**
The stop is auditable against the table above: the ceiling is a *measurement*
(0.095 ms), not an estimate, and it caps every row.

## 5. The measured number this board owes

`bench_repeat.sh v7base 3`, MODEL_PATH pinned to `/home/claudeuser/models/glm5-mxfp4`,
NP=8, bs=1, one decode row, shipping build, no flags:

| run | ms/iter |
|---|---|
| r1 | 10.535 |
| r2 | 10.543 |
| r3 | 10.634 |
| **n=3** | **mean 10.571, min 10.535, max 10.634** |

Text checked, not just the number: rank 0 emits *"The capital of France is …
Fact: Paris."* — coherent, on-topic, non-degenerate. Consistent with the
standing 10.619 baseline (Δ 0.048 ms, well inside the 0.26 ms floor).

## 6. What this leaves

The rendezvous class is closed **on its premise**, not merely on its price. With
sync-point counts equal (10 vs 11) and cross-rank counts equal (2 vs 2), the
2.02 vs 10.571 ms gap has to live in the tiles: our tile class is 5.292 ms of a
10.619 ms wall — ~69.6 µs/layer against TileRT's ~26 µs for the **whole** layer.
We are 3.09× off our own byte roof.

The arithmetic that bounds the goal, from the closed budget (`close_the_wall.py`,
tile 5.292 / coll 1.587 / rdv 2.371 / bound 0.751 / resid 0.617):

> Delete **every tile** and **every rendezvous** — all 7.663 ms of the two
> largest classes, at 1:1, which is more generous than the measured 0.27
> conversion — and the wall is **2.956 ms**. Still above 2.

Reaching 2 ms therefore requires the collective (1.587) and the boundary (0.751)
to go as well, i.e. essentially every measured line at once. That is a statement
about the budget, not about any one lever.
