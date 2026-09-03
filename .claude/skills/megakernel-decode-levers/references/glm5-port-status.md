# Porting the gpt-oss levers to GLM-5 744B — status and blockers

Assessed 2026-09-03 against GLM-5 744B, 4 GPU ranks, EP, 8 XCDs/GPU, 78 MoE
layers + 3 dense prologue layers, decode ~8.73 ms/token.

**Summary: none of the nine levers is expected to move GLM-5's wall, and the
reason is structural rather than a porting difficulty.** Two independent
measurements in this tree cap the two mechanisms the gpt-oss series relies on.
Read those two first — they falsify most of the table below before any code is
written.

This is consistent with, and now explains, the earlier conclusion in
`demo/glm5/GPT_OSS_LEVER_DIFF.md`: of 22 levers in the earlier 3.339 → 1.936
series, 15 were already present and none of the rest was a catch-up item.

---

## The two measurements that cap the class

### 1. Tile-level wins do not convert to wall on GLM-5

`MPK_MOE_PF_GROUPS` (deep k-loop prefetch) is a **28% standalone tile win**
that measured **three nulls in three paired A/Bs**: W13 guarded 10.460 → 10.491
(+0.031), guard-free 10.503 → 10.517 (+0.014), W2 n=6 pooled 10.526 → 10.521
(−0.005). The ledger's own words: *"Whatever the standalone is measuring, the
megakernel's W13/W2 tiles are not paying it."*
(`demo/glm5/CLOSING_LEDGER.md` §5d, `8123795` / `f3ef505` / `7e26c88`.)

**Most of the gpt-oss series is tile-level latency hiding** — W13 fragment
recycling, the counted handoff, bias-behind-SwiGLU, LM-head LDS pipelining, the
MFMA-unrolled arms. GLM-5 has measured that channel closed three times.

The GEMV load-depth work on this branch is a fourth instance: load depth 8 → 32
with drains 2 → 1 and zero spills measured **neutral** on the wall
(`MPK_ATTN_GEMV_PF`, `039a35e`).

### 2. Narrowing rendezvous scope is capped at 0.095 ms

> GPU-wide max arrival − worst-XCD max arrival = **0.000 µs at all ten
> rendezvous**. Narrowing every one of them simultaneously is worth **0.095 ms**
> (`glm-per-xcd-barrier-narrowing-is-zero-by-measurement`,
> `demo/glm5/RENDEZVOUS_EDGE_TABLE.md` §4).

Every chiplet holds a worker at the global max, so an XCD-scope release lets
seven XCDs out early and the eighth not at all. Corroborated directly:
`MPK_WUV_IN_MERGE` **deleted a whole rendezvous** — ten became nine, gated 4/4
— for **−0.003 ms** (`e5d1ff5`).

This caps pattern 1 (per-slice release), which is `MPK_ATTN_SLICE_RELEASE`.

---

## Per-lever status

| lever | gpt-oss delta | GLM-5 status |
|---|---|---|
| default-on flag audit | 1.902 → 1.877 | **EMPTY.** All 125 off-by-default `MPK_*` flags audited; none has a recorded win. Everything validated-positive already ships on. |
| `MPK_ATTN_SLICE_RELEASE` | 1.880 → 1.864 | **CAPPED at 0.095 ms** for all ten rendezvous at once; a whole rendezvous deleted bought −0.003 ms. |
| `MPK_W13_T0_COUNTED_HANDOFF` | 1.865 → 1.840 | **Channel closed** (tile-level, PF_GROUPS ×3). Also `MPK_W13_EARLY_REL` measured NULL: 0.51 ms counted → 0.10 ms wall. |
| RMSNorm ssq off LDS | 1.851 → 1.835 | **PORTABLE, predicted ~0.035 ms.** 59 `__shfl_xor` sites exist. Scaling gpt-oss's per-layer saving by 78/36 layers lands at/below GLM-5's 0.26 ms noise floor and at the size of its baseline drift. Only lever with no structural blocker. |
| `MPK_MOE_XCD_PAIR` | 1.706 → 1.642 | **STRUCTURALLY INAPPLICABLE** under EP — see below. |
| W13 fragment recycle/pipeline | shipped | Channel closed (tile-level). |
| W13 bias behind tile-0 SwiGLU | shipped | Channel closed. Related: bias prefetch in GLM-5's router TopK tail is a **205 ns** cold fetch. |
| routing release epochs derived | shipped | Routing poll is **0.684 µs/layer, bare spin, idle bus**; ceiling 1.199 µs/layer. |
| LM-head groups via recycled LDS | shipped | Channel closed; LM head is not in GLM-5's per-layer loop. |

---

## Why `MPK_MOE_XCD_PAIR` cannot port

This is the largest gpt-oss lever (−0.064 ms) and the one that looks most
attractive, because GLM-5 genuinely still does what it removed. GLM-5's W13
tile prologue walks `d_mask[i]` to find its expert and then loads
`d_routing[e * BATCH_SIZE + tok]` — a dependent chain, and no weight address
can be formed until it finishes
(`gang_moe_linear_mxfp8_mi300.cuh:1821-1891`, `:2015-2045`).

The blocker is arithmetic, not effort:

| | gpt-oss 120B | GLM-5 744B |
|---|---|---|
| routed experts | 128 | **256** |
| picks per token (top-k) | **4** | **8** |
| GPUs / EP ranks | **1** (`world_size = 1`) | **4** |
| experts owned per rank | all 128 | 64 |
| **live experts per rank per token** | **4** | **~2** |
| XCDs per GPU | 8 (4 pairs) | 8 |

gpt-oss's map is static because **#picks == #XCD-pairs**: pair *i* runs pick
*i*, so ownership needs no routing read at all. GLM-5 has 8 picks spread over
4 EP ranks, so a rank holds only ~2 live experts for 8 XCDs. Pinning an expert
to an XCD pair would idle roughly six of eight XCDs.

GLM-5's round-robin stripe is *load balancing*, not laziness, and the source
says so: *"Tiles interleave round-robin across the XCDs — `global_tile =
tile_idx*8 + xcd_id` — rather than blocking, so that all 8 XCDs stay busy when
fewer than 8 experts are active, which at top-4-plus-shared is always"*
(`gang_moe_linear_mxfp8_mi300.cuh:1571-1574`). And independently:
*"expert → XCD is a routing outcome, not a static map"*
(`demo/glm5/price_xcd_narrowing.py:136`).

The overlap half (weight DMA under the router's later picks) is separately
blocked: you cannot prefetch through a barrier whose output **is** the address.
Expert identity selects which bytes to fetch, so the only escape is
*speculative* expert prefetch (`CLOSING_LEDGER.md:205-214`). gpt-oss did not
speculate — its static map made the id knowable without a routing read, which
is the step GLM-5's EP arithmetic forecloses.

Do not attempt `MPK_EARLY_ROUTING` either: it failed on gpt-oss with three
different text hashes in three runs, because an ascending/compacted mask's slot
0 is the lowest expert id, not the first-selected one. GLM-5's `d_mask` is
compacted ascending, so it has exactly the same defect.

---

## Where gpt-oss offers nothing, and it is the biggest class

GLM-5's closed budget (`close_the_wall.py`, at the 10.619 ms baseline):

| class | ms | gpt-oss analogue |
|---|---:|---|
| tile | 5.292 | levers exist, **channel measured closed ×3** |
| rendezvous | 2.371 | scope narrowing **capped 0.095** |
| **cross-rank collective** | **1.587** | **none — gpt-oss is single-GPU** |
| layer boundary | 0.751 | whole-boundary ablation is 0.051 ms |
| residual | 0.617 | — |

**gpt-oss runs at `world_size = 1`** (`~/mirage/demo/gpt_oss/demo.py:374`, and
both bench scripts pin one device). It pays **zero** cross-rank collective and
**zero** cross-rank rendezvous. GLM-5 pays a measured **1.587 ms** — the EP
fold plus the QB_TP cross-rank gather.

So the single largest remaining GLM-5 class is one the gpt-oss branch has never
had to solve, and its 1.64 ms is not a template for a 4-rank 744B EP decode.
Catch-up levers for that class have to come from a multi-GPU reference
(TileRT's two fused all-reduces), not from this branch.

---

## What to do with this file

Before porting anything else from `megakernel-decode-levers`, ask which of the
two capped mechanisms it relies on:

- Does it reduce **tile time**? Then PF_GROUPS's three nulls apply. Price it
  against the wall, never against a standalone tile bench.
- Does it narrow a **barrier's scope**? Then the 0.095 ms ceiling applies.
- Does it reduce **cross-rank bytes or collective count**? Then it is in the
  one class that is still open, and gpt-oss has no example of it.
