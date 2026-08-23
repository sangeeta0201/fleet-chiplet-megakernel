# Regime B on the two largest segments: paper legality pass + priced prediction

Date 2026-08-23. Branch `merge-rocshmem`. **No GPU run** — every number here
comes from the existing `MPK_BAR_SKEW=3` bs=1 NP=8 log (`/tmp/item11_bs1.log`)
and from reading the kernel sources. Reproduce with
`python3 demo/glm5/price_regime_b_segments.py`.

Regime B is the only makespan operation that has ever moved this wall: delete a
whole phase **together with its rendezvous**, and the wall moves by the segment
`max(b) - max(a)` (`glm-makespan-predictor-three-regimes`). The two largest
segments in the layer are `ref -> qkv` at **31.98 us/layer** and
`w13 -> entry_bar` at **27.13 us/layer** — 2.059 and 1.747 ms/token at face
value. That face value is what makes the class look like the 5.3x lever.

**It is not. Both segments have an EMPTY legal-deletion set.**

---

## 1. What a segment must contain to be regime-B deletable

The operation needs two things at once:

1. the **work** in the segment must be removable (moved, made redundant, or
   never needed), and
2. the **rendezvous** that terminates it must be droppable — i.e. the
   producer -> consumer edge it enforces must be satisfiable *without* a
   GPU-wide rendezvous, by one of: per-XCD scope, per-worker-local scope, or
   recompute-instead-of-communicate.

Failing (2) alone is fatal even if (1) holds: the barrier stays, the segment
stays, and you are back in regime A, whose cap is
`max_all - max_outside` and which measured 0.000-1.199 us/layer for all five
non-MoE phases (`glm-cutting-work-in-a-phase-is-absorbed`).

---

## 2. Segment `ref -> qkv_barrier`, 31.977 us/layer (2.059 ms/token)

Decomposition, MAX-over-232-workers of per-worker mean arrival, rank 0,
38322 layer samples/worker:

| component | slot | us/lyr | share |
|---|---|---|---|
| entry_bar release fan-out observed | 0 | 2.259 | 7.1% |
| Phase 0-EP: the EP collective + peer poll | 1 | 15.617 | **48.8%** |
| dispatch into the attn half | 16 | 2.257 | 7.1% |
| qkv_a tiles | 17 | 11.844 | **37.0%** |

Slot 4 (`all peer signals observed`) is EXCLUDED: cnt ranges 4..13726 over only
8 workers, the `MPK_BAR_SKEW_DROP_NS` guard eating most samples, and its max
(39.843) exceeds the strictly-later slot 1 (17.876), which is impossible. Read
a guard before dividing by it.

### Edge 2a. `qkv_a` -> `q_b` (the edge the qkv_barrier enforces)

**Producer.** `gang_mla_attn_fused_mi300.cuh:677`, qkv_a's output is sharded by
**output column across XCDs**:

```c
unsigned short *xcd_out =
    static_cast<unsigned short *>(qkv_a_out_ptr) +
    static_cast<size_t>(xcd_id) * qkv_n_wgs_per_xcd * QKV_OUTPUT_PER_WG;
```

XCD *x* writes columns `[x*S, (x+1)*S)` of the `[q_a | latent]` row and nothing
else.

**Consumer.** Same file, line 848: `q_b`'s GEMM is handed `qkv_a_out_ptr`
*whole* (declared at line 178 as `// [0] [q_a | latent], declared whole`) as
its **reduction operand**, contracting over `QB_REDUCTION_SIZE` — the full q_a
latent — and again at line 853 as `kv_latent` at `KV_INPUT_OFFSET`.

**Verdict: ILLEGAL.** Every q_b tile reads all 8 XCDs' column slices. The edge
is all-to-all across XCDs by construction.

* per-XCD scope — no, the consumer's reduction axis *is* the sharding axis.
* per-worker-local — no, same reason, one step worse.
* recompute-instead-of-communicate — each XCD would have to recompute the other
  seven slices: qkv_a goes 11.844 -> ~95 us/layer, i.e. +5.4 ms/token to save
  a 2.26 us fan-out. Strictly negative.

### Edge 2b. previous layer's MoE -> this layer's Phase 0-EP (the entry_bar edge)

Covered in §3; it is the same edge the entry_bar segment terminates on.

### Edge 2c. the EP collective itself (15.617 us, 48.8% of the segment)

This is not a barrier, it is a **cross-rank reduction**. Rank *p* holds only its
own slice of the routed experts; the other seven ranks' contributions to the
hidden row physically do not exist in this GPU's memory. That is the definition
of expert parallelism.

**Verdict: ILLEGAL, and not even a communication-scope question.** There is no
per-XCD or per-worker-local version of data that is on another *node*.
Recompute-instead-of-communicate would mean replicating all 744B of expert
weights on every rank — the model does not fit.

Already priced from three independent directions and all neutral: hoisting the
fold (`glm-qkv-ep-fold-hoist-is-neutral`), widening it
(`glm-widening-the-ep-fold-is-neutral`), and the once-only skew model
(`glm-inter-rank-skew-is-paid-once` — it is a TAX, not a lever).

### Segment verdict

Legal removals: **none**. Predicted wall delta **0.000 ms**.

The ILLEGAL ceiling — pretending the whole segment were deletable — is
`2.259 x 1.00 + 29.718 x 0.27 = 10.28 us/lyr = 0.662 ms`, and that price buys
deleting the entire EP collective *and* the whole qkv_a GEMM.

---

## 3. Segment `w13_barrier -> entry_bar`, 27.131 us/layer (1.747 ms/token)

| component | slot | us/lyr | share |
|---|---|---|---|
| w13_barrier release fan-out observed | 7 | 5.310 | 19.6% |
| W2 tiles | 8 | 12.407 | **45.7%** |
| layer boundary: return, fence, pointer refresh, dispatch | 11 | 6.751 | 24.9% |
| next entry_bar arrival | 9 | 2.663 | 9.8% |

The kernel already says so at `gang_oproj_router_fused_mi300.cuh:1720`:
*"S8 is where the layer's work ends; everything between S8 and the next
layer-entry barrier is arrival and rendezvous."*

### Edge 3a. `W13` -> `W2` (the edge the w13_barrier enforces)

**Producer.** `gang_oproj_router_fused_mi300.cuh:1456`, W13 tiles are dealt by a
**global** tile index strided across all 232 workers, not per-XCD:

```c
for (int t = xcd_rank; t < moe_w13_live; t += tiles_per_xcd) { ... moe_swiglu_out_ptr, t }
```

**Consumer.** Line 1692, W2 tile `t` is handed `moe_swiglu_out_ptr` and
contracts over `MOE_INTERMEDIATE` — the *whole* SwiGLU intermediate for its
expert. That intermediate is produced by `MOE_W13_TILES_PER_EXPERT` = 8 tiles
per XCD (`glm-moe-phase-is-round-quantized-and-w2-is-latency-bound`), dealt
round-robin, hence spread over workers on **all 8 XCDs**.

**Verdict: ILLEGAL.** All-to-all across XCDs, same shape as edge 2a. And even if
a tile-deal rewrite made it per-XCD, the per-XCD narrowing class is already
measured at **0.095 ms for ALL TEN rendezvous narrowed simultaneously**
(`glm-per-xcd-barrier-narrowing-is-zero-by-measurement`) — one barrier's share
of that is under the 0.26 ms noise floor by a factor of 20.

### Edge 3b. `W2` -> next layer's Phase 0-EP (the edge the entry_bar enforces)

This one is the strongest of the four, and it is the general fact that kills the
whole MoE side of the class.

`gang_oproj_router_fused_mi300.cuh:325-338`: the MoE workspace is a
**device-scope atomicAdd accumulator over the full hidden row**, and the code
comments say exactly why the write must be write-through:

```c
// Write-through: the consumer is a device-scope atomicAdd, performed past the
// XCD's L2, so a dirty local line holding the zero could later be written
// back over the accumulated result.
```

Every W2 tile, on every XCD, for every routed expert, atomically accumulates
into the same `[BATCH_SIZE * HIDDEN_SIZE]` f32 row. The consumer — layer L+1's
Phase 0-EP fold, then the qkv_a RMSNorm — reads that row whole.

**Verdict: ILLEGAL, maximally.** This is not "sharded and needs a gather"; it is
a **reduction whose every contributor touches every output element region**.
There is no scope narrower than device-wide that makes the sum correct.
Recompute-instead-of-communicate means running the whole MoE on every XCD.

### Edge 3c. the layer boundary (6.751 + 2.663 = 9.414 us)

Not a data edge — it is return path, `threadfence_gpu`, the 34+13 pointer-table
copy, two `__syncthreads`, and the dispatch. This is the ONE component in either
segment that is not essential math and not an all-to-all rendezvous, and it is
**already measured**: the ablation is worth **0.051 ms**
(`glm-layer-boundary-ablation-is-zero`), against a 0.26 ms floor, despite
padding at 1.22:1 (`glm-layer-boundary-pad-slope-is-1-22`) — uniform time there
is critical-path but there is only 0.05 ms of it to take.

### Segment verdict

Legal removals: **none new**; the only non-essential component is the layer
boundary at an already-measured 0.051 ms. Predicted wall delta **0.000 ms** for
anything not already on the ledger.

ILLEGAL ceiling: `5.310 x 1.00 + 21.821 x 0.27 = 11.20 us/lyr = 0.721 ms`, and
that price buys deleting the MoE down-projection.

---

## 4. Priced prediction, written down before any build

Formula from the ruling, using the two coefficients measured in `73c0afb`:

    predicted wall delta = (rendezvous us/lyr removed) x 1.00
                         + (work      us/lyr removed) x 0.27
    us/lyr -> ms/token:  x 76 layers x K,  K = 139.72 / 164.947 = 0.847

| segment | face value | legal removals | **predicted** | illegal ceiling |
|---|---|---|---|---|
| `ref -> qkv` | 2.059 ms | none | **0.000 ms** | 0.662 ms |
| `w13 -> entry_bar` | 1.747 ms | none (boundary already = 0.051) | **0.000 ms** | 0.721 ms |

Build threshold was 0.5 ms. **Neither segment prices through, and neither one
reaches 0.5 ms even with an illegal 100%-deletable assumption.**

Per the ruling: *"If both segments predict under 0.5 ms, commit the pricing and
say the class is closed on its own arithmetic — do not build a sub-noise-floor
change."*

**REGIME B IS CLOSED ON THE TWO SEGMENTS THAT WERE SUPPOSED TO CARRY IT.**
No GPU run was spent, so there is no measured-vs-predicted pair to report: the
prediction is 0.000 and the build was correctly not attempted.

---

## 5. What this actually establishes (the generalizable part)

The four edges above are not four accidents. Every rendezvous in the GLM-5 layer
separates a producer whose output shard is **misaligned with how the consumer's
tiles are dealt** — usually because the producer shards the very axis the
consumer contracts over:

| rendezvous | producer shards by | consumer needs | scope needed |
|---|---|---|---|
| qkv_barrier | qkv_a output column, `xcd_id * S` | contracts over the full q_a latent | device |
| w13_barrier | W13 tile, global stride over 232 | contracts over the expert's full intermediate | device |
| entry_bar | W2 tile, device-scope atomicAdd into one row | reads the whole hidden row | device |
| qb_barrier | one whole HEAD per XCD (`qb_head_base + xcd_id`), and under QB_TP one head-group per RANK | decode tiles are dealt by KV chunk, so any worker may hold any head | device **and cross-rank** |

The qb_barrier is the one that is not a contraction-axis conflict: XCD *x*
writes head *x* complete. It still needs device scope because the *consumer's*
deal is by KV chunk, not by head — and cross-rank scope because under QB_TP the
heads live on eight different ranks. That peer wait is already priced at
0.228 ms (`glm-qb-peer-wait-ceiling-is-0.335ms`), and re-dealing decode by head
is the head-sharding proposal that memory already records as a NO-GO.

This is the structural reason the barrier-mechanism levers have all measured out
(`glm-barrier-narrowing-is-measured-out`, `glm-deleting-a-whole-rendezvous-is-neutral`,
`glm-per-xcd-barrier-narrowing-is-zero-by-measurement`) and why
`glm-no-legal-independent-round-fusion-exists` holds. **A GEMM chain whose every
stage is column-sharded and row-contracted admits no communication-scope
reduction at all.** The only way to remove one of these rendezvous is to change
the *sharding*, i.e. shard by the reduction axis instead and pay a split-K
reduction — and that is `glm-w2-splitk-is-a-large-negative`, +4.12 ms.

So the residual after this pass is not "find a third segment". The layer's 10
rendezvous are load-bearing and the work between them is essential math.

### Correction: the residual is NOT "arithmetic volume"

The first draft of this section said the remaining gap to 2 ms is a
numerics/arithmetic-volume problem. **That is wrong and
`demo/glm5/close_the_wall.py` (same commit) shows why.** The measured byte
volume already permits 2.319 ms (`roofline.py`: 1.936 HBM at busiest-rank
routing + 0.383 measured EP latency). Volume is not the constraint.

Closing the whole wall against labelled components — the 7-phase busy/spin table
plus exactly the pieces the two segment decompositions above supply — accounts
**10.002 of 10.619 ms**, residual 0.617 (5.8%):

| class | ms/token | share | its own roof | headroom |
|---|---|---|---|---|
| tile math+bytes | **5.816** | 54.8% | 1.712 (same 8 phases, own widths) | **4.104** |
| cross-rank collective | 1.587 | 14.9% | 0.383 (EP) | 1.204 |
| rendezvous | 1.848 | 17.4% | 0 | 1.848 |
| task/layer boundary | 0.751 | 7.1% | 0 | 0.751 |
| residual | 0.617 | 5.8% | 0 | 0.617 |

**The tiles run at 29% of their own byte roof — 3.40x off — and that 4.104 ms is
the largest single item in the budget by a factor of two.** It is an efficiency
problem *inside* the tiles, consistent with
`glm-attention-tiles-are-latency-bound-not-valu-bound` (qkv_a 68% vmcnt), not a
FLOP or byte count.

Denominator warning: the tile roof is **1.712 ms**, from
`demo/glm5/tile_roof_by_phase.py` — the same eight phases priced at their own
live widths. It is *not* `roofline.py`'s 1.936, which is a whole-*iteration*
floor including the dense prologue layers, embeddings and lm_head. A first pass
here divided by 1.936 and got 3.00x; the apples-to-apples figure is 3.40x.

The caveat that keeps this honest: busy converts to wall at ~0.27 near this
operating point, so 4.104 x 0.27 = **1.108 ms** even if every tile in the layer
were simultaneously taken to the byte roof. That is still 4x the noise floor and
the largest predicted lever left on the board — but it is 1.1 ms, not 4.1.

### ...and it is not a BANDWIDTH story either

`tile_roof_by_phase.py` resolves the 4.104 ms per phase, ranked by headroom:

| phase | headroom ms | % of roof | workers | MB |
|---|---|---|---|---|
| router | **0.944** | 2.7% | 128 | 1.62 |
| q_b (+W_UK, W_UV) | **0.695** | 12.5% | 128 | 6.22 |
| decode | **0.597** | 1.3% | 16 | 0.07 |
| qkv_a | 0.524 | 32.6% | 168 | 16.63 |
| W13 | 0.445 | 63.0% | 232 | 53.48 |
| W2 | 0.421 | 47.4% | 232 | 26.74 |
| o_proj | 0.285 | 40.3% | 192 | 12.98 |
| merge / Ph8 | 0.195 | 0.0% | 128 | 0.00 |

**The top three headroom rows are the three phases carrying the least bytes.**
The three byte-heavy phases hold 97 of the layer's 118 MB and only 1.390 ms of
headroom between them — and W13's and W2's shares are already closed by
measurement (`MPK_MOE_PF_GROUPS` null twice; OPW=16 −1.34, KSPLIT=2 −4.12,
OPW=128 neutral). So the 3.40x is concentrated exactly where bandwidth is
irrelevant. It is latency and fixed per-tile cost.

Two guards on that table:
* every "busy" row is a stage-stamp span bracketing the **block**, not the GEMM,
  so it is a superset of the phase's tile time. W13 reads 63% here but 76% in
  `width_corrected_roofline.py`, which uses the measured subphase counter
  SP3[4] = 13.06 µs and is the right number. Every % of roof here is a lower
  bound; every headroom an upper bound. Prefer a subphase counter where one
  exists.
* correcting for width moved the tile roof 1.731 → 1.712 ms, i.e. the whole
  width correction on this phase set is 0.018 ms. Byte-heavy phases already run
  at 232 and gain nothing; narrow phases lose but carry no bytes.
