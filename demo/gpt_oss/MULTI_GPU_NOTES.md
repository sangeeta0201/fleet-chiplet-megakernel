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
