# Multi-GPU (2-GPU) gpt-oss-120b: where the time goes

Decode, batch size 1, MI355X, 36 layers, rocSHMEM IPC backend.
Single-GPU reference: **2.138 ms/token**. Two GPUs: **4.333 ms/token**.

This file records the measurements behind that gap so the next change is aimed
at the right thing. Everything here was measured on branch `multi-gpu-rocshmem`
with `FUSE_FULL_LAYER=0 PRECOMPUTED_DISPATCH=0` (the 2-GPU task decomposition;
the fused-layer 1-GPU build collapses the four stages into one gang task and
emits no breakdown, which is why an early span run came back empty).

## 1. TP works. Communication is the whole deficit.

`MPK_SPAN_TIMING=1`, per layer, microseconds:

| | compute | gap | total |
|---|---|---|---|
| 1 GPU | 57.11 | 8.16 | 65.27 |
| 2 GPU (rank 0) | 33.60 | 59.93 | 93.53 |

Compute nearly halves — tensor parallelism is doing its job. The deficit is
entirely gap: **+51.8 us/layer x 36 layers = 1.86 ms**, which is the whole
2.2 ms regression. Two allreduces per layer x 36 layers = **72 cross-GPU
syncs per token**, and at bs=1 there is no other work in flight to hide them.

## 2. The gap is fixed per-collective cost, not a drain proportional to compute

Sweeping the router's top-k (`MOE_TOPK`, a latency probe — output is wrong by
construction) shrinks MoE compute on both configurations. If the gap were
compute-proportional the two deltas would scale together. They do not:

```
  +-------+-------+-------+
  | top-k | 1-GPU | 2-GPU |
  +-------+-------+-------+
  | 4     | 2.138 | 4.333 |
  +-------+-------+-------+
  | 1     | 1.991 | 4.133 |
  +-------+-------+-------+
  | delta | 0.147 | 0.200 |
  +-------+-------+-------+
```

Cutting MoE work by 4x moves 1-GPU by 0.147 ms and 2-GPU by 0.200 ms — only
0.053 ms more. So roughly **1.8 ms of the 2-GPU cost is fixed sync latency**
that does not care how much compute surrounds it.

## 3. Perfect overlap is not enough

The user's framing was: if communication were perfectly overlapped, 2 GPUs
should land at the latency of one GPU doing half the expert work. That number
was measured directly — single GPU at `MOE_TOPK=2` is **2.036 ms** (topk=4:
2.138, topk=1: 1.991). Expert parallelism's entire prize is 0.102 ms (4.8%).

Meanwhile the overlap ceiling is `max(compute, gap) = max(33.60, 59.93) =
59.93 us/layer x 36 = **2.16 ms**` — still above single-GPU's 2.138 ms.
Communication is 1.8x compute, so it cannot be hidden behind compute no matter
how well the two are interleaved. **The number of syncs has to come down, not
just be overlapped.** That is what ATTN_DP (deletes allreduce #1, 72 -> 36
syncs) and the fused o_proj+put (deletes the identity + copy tasks around the
remaining one) are for.

## 3b. ATTN_DP: deleting allreduce #1 is worth 0.774 ms

`ATTN_DP=1` (`run_mp2_dp.sh`) replicates attention on both ranks instead of
splitting the heads, so o_proj's output is already complete per-rank and the
attention allreduce disappears. 72 syncs/token -> 36.

| config | ms/token | output |
|---|---|---|
| 1 GPU | 2.138 | correct |
| 2 GPU, TP attn + replicated MoE | 4.329 | correct |
| 2 GPU, DP attn + no allreduce | 3.555 | correct |
| 2 GPU, DP attn + EP MoE | 4.453 | **garbage** |
| 2 GPU, TP attn + EP MoE | 5.224 | **garbage** |
| 2 GPU, DP attn + EP_NOSLICE | 4.531 | doubled-MoE (expected) |

Correctness: DP passes **4/4** prompts at 256 new tokens against the 1-GPU
reference (`compare_tokens.py`, prefix agreement 21-66 tokens).

Two things to read off this. First, the 0.774 ms saved by removing one of the
two collectives is close to half the ~1.8 ms fixed sync cost from section 2 —
consistent with the collectives being the cost and roughly equal to each other.
Second, 3.555 ms is *still* well above single-GPU's 2.138 ms with only 36 syncs
left, so removing the remaining allreduce is necessary but probably not
sufficient either.

Note the DP-no-allreduce row shards nothing: both ranks run the whole model
redundantly. It is a correctness reference and a latency floor for "DP
attention + one collective", not a shippable configuration.

## 3c. Expert parallelism: FIXED. It was a missing exit barrier.

**Resolved.** DP+EP with the collective inlined into the monolith passes 4/4
prompts at 256 tokens. The cause is below; the original investigation notes
are kept after it because their two isolations were correct and are what
narrowed it down.

The last line of the original entry guessed right: "a race the single-layer
snapshot does not catch." Specifically, the combine had **no exit fence**.
The cross-rank exchange runs on 8 of 240 workers (workgroup 0 of each XCD);
the other 232 fell out of the MoE barrier straight into the next layer, and
because the multi-layer replay loop re-enters the same function with swapped
pointer tables, "the next layer" starts immediately. Three overlapping
corruptions followed:

  * the next layer's QKV prologue read the combined output before the reduce
    wrote it,
  * its Phase 7 overwrote `attn_proj_out` while the fold was still reading it
    as the residual,
  * its Phase 8 `atomicAdd`ed into `moe_workspace_f32` around the fold's
    zeroing of that same buffer.

All three are timing-dependent and none is visible in a single-layer
`--verify` snapshot -- which is exactly why the two isolations below both came
back clean. It also explains the "error compounding" reading: the per-layer
diffs of 32 and 96 on ~1000 were not accumulated rounding, they were a race
landing partially.

Fix: phase 9e, a per-XCD release flag written by the combiner and polled by
every other worker before it leaves the layer
(`gang_full_layer_fused_mi300.cuh`). Two smaller defects were fixed alongside
it and either alone was fatal: the per-layer signal array (a signal counter
compared against a run-monotonic threshold must be shared across layers, or it
only ever reaches +1 per token and hangs on layer 1), and the counter buffer
being 832 ints while the EP barriers needed 1216.

Cost, measured back-to-back, `MAX_SEQ_LENGTH=128`, decode steady-state:

| config                                 | ms/token | tokens    |
|----------------------------------------|----------|-----------|
| 2 GPU DP, monolith + precomputed       | 2.122    | 4/4 @ 256 |
| 2 GPU DP+EP, monolith + inline combine | 2.892    | 4/4 @ 256 |

So +0.77 ms for the one remaining collective, even fully inlined with zero
scheduler round-trips. That is the honest number: the fusion removed the
deadlock and the 72 round-trips per token, but 36 cross-GPU syncs on the
critical path still cost more than the halved expert work saves at bs=1. EP's
win here is memory capacity, not decode latency. Worth noting that
0.77 ms / 36 layers = 21 us per sync, which is close to bare put+signal+wait
latency -- there is not much fusion left to extract, so a faster EP needs
FEWER syncs (batching layers between collectives), not a cheaper one.

### Original investigation (kept -- the isolations were sound)

DP+EP is the configuration that actually halves the weights while keeping one
collective, and it emits garbage. So does TP+EP, so this is independent of
ATTN_DP.

Two isolations have been run, and they do not point where you would expect:

* `EP_NOSLICE=1` (keep the MoE allreduce, do NOT slice expert weights, so both
  ranks compute the full sum and the allreduce yields 2*W + x) produces the
  *predicted* doubled-MoE degradation: "The capital capital? The user capital?".
  So **the MoE allreduce sums correctly.**
* `--verify` on DP+EP shows routing is correctly global and disjoint (rank0
  owns active expert [36], rank1 owns [98, 111, 116]), each rank's partial
  matches its EP-aware PyTorch reference, and post-allreduce `mlp_final`
  matches the FULL all-experts reference (top-8 elements agree; max abs diff
  256 against values of order 1000). So **the last-layer EP arithmetic is
  approximately right too.**

Which means the obvious suspect — the slicing / local_eid mapping — is not
obviously guilty, and the failure is something that only shows up over 36
layers: per-layer error compounding, or a race the single-layer snapshot does
not catch. The per-layer diffs above (32 and 96 on ~1000) are larger than the
replicated path's, which would fit compounding. Unresolved; this is the next
thing to chase.

Worth noting how it got missed originally: commit 428b74b verified EP by
checking that rank0's partial + rank1's partial equalled the post-allreduce
`mlp_final`. That is self-consistent even when both partials are wrong — it
tests the allreduce, not the experts. Reading the generated text catches it
immediately.

## 3d. Fusion under DP: enabled, but the payload fusion is broken on 1 GPU too

Under `ATTN_DP=1 MOE_EP=0` there is not one collective left in the step: each
rank runs the complete model on the same token, so per rank the task graph is
*identical* to the single-GPU one and every single-GPU fusion is legal. The
seven `world_size == 1` fusion guards were gating on the wrong thing — they
exist because TENSOR parallelism breaks fusion (sharded KV heads break the
`kv_head == xcd_id` mapping; a mid-layer allreduce splits o_proj from the
router), and neither applies here. They now gate on
`dp_local = attn_dp and not moe_ep`.

Measured, 2 GPU, DP + replicated MoE, decode ms/token, **dynamic dispatch**
(`PRECOMPUTED_DISPATCH=0`):

| FUSE_QKV_ATTN | FUSE_OPROJ_TOPK | FUSE_FULL_LAYER | ms | output |
|---|---|---|---|---|
| 0 | 0 | 0 | 4.013 | correct |
| 1 | 0 | 0 | 4.015 | correct |
| 0 | 1 | 0 | 3.774 | **0 tokens** |
| 1 | 1 | 0 | 3.777 | **0 tokens** |
| any | any | 1 | — | **hangs** (spins forever in the megakernel) |

Read off: QKV+attn fusion is free (4.015 vs 4.013 — no gain, no harm), and the
entire 0.24 ms is in o_proj+TopK fusion, which does not work.

**FUSE_OPROJ_TOPK does not work on ONE GPU either.** `FUSE_OPROJ_TOPK=1
FUSE_FULL_LAYER=0` on `run_1gpu.sh` also generates 0 tokens (with or without
QKV fusion). Pre-existing broken sub-configuration, not a DP regression: the
standalone `gang_linear_mxfp4_res_bias_rmsnorm_topk` task is only exercised in
production inside the full-layer monolith, where it is reached through a
different counter/barrier setup. The DP gating change just made it reachable
from a 2-GPU launcher.

### The monolith hang is a dynamic-dispatch bug, not a multi-GPU bug

The `FUSE_FULL_LAYER=1` hang above reproduces on **one** GPU with
`PRECOMPUTED_DISPATCH=0` — same signature, stops at
`[HOST_DBG] launch_persistent_kernel ENTER` and spins. Nothing to do with the
second worker queue at `persistent_kernel.cuh:1162` (an earlier note here said
that; it was wrong).

Cause is a gang-rank / dispatch mismatch. Under **precomputed** dispatch a gang
worker takes its XCD-local rank straight from the template
(`persistent_kernel.cuh:2022-2047`, `block_xcd_local_rank = pc_xcd_rank`), so
ranks are dense `0..dispatch_count-1` and every tile is claimed. Under
**dynamic** dispatch the scheduler broadcasts to `my_workers[widx]` for
`widx < dispatch_count` (`:3228-3240`) but never transmits `widx`; the worker
re-derives its rank by scanning `worker_xcd_map` for all workers with a lower
global id on the same XCD. The two orderings do not agree, so the dispatched
set gets sparse/duplicated ranks, some tiles are never computed, and the
hierarchical barrier (which waits for `total_oproj_tiles` arrivals) never
completes.

Fix, if the dynamic path is ever needed: have the scheduler transmit `widx` as
the gang rank, mirroring what precomputed already does. Not needed for the
configuration below, which is faster anyway.

### DP + precomputed + monolith: 2.135 ms on 2 GPUs

With `PRECOMPUTED_DISPATCH=1` the monolith runs under 2-GPU DP and matches
single GPU exactly:

| config | decode ms/iter | output |
|---|---|---|
| 1 GPU, monolith + precomputed | 2.138 | correct |
| **2 GPU DP, monolith + precomputed** | **2.135** | correct ("Paris") |
| 2 GPU DP, unfused + dynamic | 3.777 | correct |

Why it works where everything else deadlocked: `ATTN_DP=1 MOE_EP=0` emits
**zero cross-GPU events**, so precomputed dispatch never reaches the cross-GPU
signaling path (`is_nvshmem_event`, `:544/:1556/:2643`) that deadlocks it under
TP, and precomputed supplies the gang rank directly so the dynamic-dispatch bug
above never fires. The per-rank task graph is byte-for-byte the single-GPU one.

**This config shards nothing.** Both ranks run the whole model on the same
token, so 2.135 ms is single-GPU latency bought with two GPUs — zero speedup,
and zero multi-GPU overhead. Its value is as a floor: it proves the fused +
precomputed fast path survives multi-GPU intact, so any real sharding
(EP MoE, §3c) starts from 2.135 rather than from 3.777.

## 4. Tile geometry: no staircase to pipeline against

o_proj emits `2880/16 = 184` tiles (`o_output_per_wg = 16`) against 240
workers, so every tile gets its own workgroup and they all finish inside one
short window. This mirrors the W2 finding upstream (mirage eb6b2f6): 89% of W2
tiles finish inside a 2 us window. A chunked "put chunk j while computing
chunk j+1" pipeline therefore has almost nothing to overlap — the chunks do not
complete in sequence, they complete together. Fusing the put into the o_proj
epilogue is still worth it, but for **task/dispatch elimination**, not for
pipelining.

The epilogue writes disjoint column slices per workgroup
(`gang_linear_mxfp4_res_bias_mi300.cuh:~358`), so a fused put there is legal.

**But under ATTN_DP that put no longer exists.** Fusing o_proj+put was the plan
for reclaiming the identity + copy dispatches around allreduce #1; deleting the
allreduce outright removed all three tasks instead, which strictly dominates
fusing them. The o_proj epilogue now writes a plain local result.

The fusion idea still applies to the *surviving* collective, the MoE AR#2
combine, which is still emitted as three tasks (`moe_residual_add_f32` ->
`identity` -> `allreduce`, demo.py:2452-2470). Upstream mirage already has this
as `MPK_INLINE_AR2`: fold + grid-bridge collapsed into one grid-partitioned
fold-copy task at ar_grid, measured at -0.217 ms/token on 2x MI350. Porting it
is the natural next task-elimination step -- but it is gated on EP being
correct, since `inline_ar2` requires `moe_ep`.

## 5. AR granularity coarsening is broken on this branch

`_ar_elems_per_block()` in `demo.py` defaults to **no coarsening** (64
elems/block, grid 46). Coarsening is opt-in via `AR{1,2}_ELEMS_PER_BLOCK` and
currently produces wrong output:

* AR#1 coarsened: repetitive degeneration ("The The This is a bit of. The 1.0.")
* AR#2 coarsened (`EP_NOSLICE=1`, grid 4): 4.760 ms but word salad
  ("isos-history Pers diagramHEL Ve Esc"); control at grid 46 gave 5.286 ms and
  coherent text.

Both looked like clean speedups until output was checked. Upstream mirage
coarsens only AR#2. Do not re-enable without running the correctness suite.

## 6. Correctness harness

Exact token match is **not** the pass criterion — TP reorders the o_proj
reduction, bf16 low bits change, and greedy decode flips on a near-tie logit.
Observed prefix agreement on *correct* 2-GPU runs: 3 to 66 tokens.

```bash
./run_correctness_suite.sh run_1gpu.sh 1gpu     # 4 prompts x 256 new tokens
./run_correctness_suite.sh run_mp2.sh  mp2
python compare_tokens.py /tmp/gptoss_correctness 1gpu mp2
```

`compare_tokens.py` gates on a per-prompt content keyword. A distinct-token
ratio was tried and rejected: the broken AR#2 word salad scored 0.89 distinct
against 0.78 for correct text, i.e. it ranked the garbage above the good
output.

Status: **2-GPU TP-attention passes 4/4 prompts at 256 new tokens.**
