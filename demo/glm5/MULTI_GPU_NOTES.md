# GLM on 8 GPUs — expert parallelism

Companion to `demo/gpt_oss/MULTI_GPU_NOTES.md`, which is the authoritative
record for the mechanism. This file covers only what is *different* for GLM.

## Status

| piece | state |
|---|---|
| `fork/multi-gpu-rocshmem` merged onto the GLM branch | done (`f99a319`) |
| single-GPU GLM unchanged by the merge | done — 3.684 ms, "Paris", 46 fused layers |
| peer transport works on this box | done — all 8 GPUs mutually peer-accessible |
| `MAX_INPUTS_PER_TASK` raised 28 -> 32 | done |
| EP counter slots + template params in the monolith | done (inert at `EP_WORLD_SIZE == 1`) |
| EP fold/exchange body in the monolith | **not started** |
| `ep_prev_gather` sum inside GLM's Phase 1 | done — `test_ep_prologue_sum` passes |
| demo.py EP plumbing + expert slicing | **not started** |
| `run_mp8_dp_ep_fused.sh` | **not started** |
| GLM-5.2 weights on this box | needs a streaming loader, see below |

## The transport floor, measured here

`tests/standalone/test_ep_collective`, world 2, 8 producers, 5888 B/slot,
2000 iters, MI355X:

| variant | us/exchange |
|---|---|
| bare peer store | 1.600 |
| + signal | 3.037 |
| aggregated | 4.481 |
| per-XCD, 8 lines | 4.075 |
| full phase 9 | 8.106 |
| full phase 9, no release fence | 10.637 |

The benchmark is hardcoded to two ranks (`Rank r[2]`). It was not generalized,
because `run_mp4_dp_ep_fused.sh` already records Phase 9 at 0.229 ms (world 2)
vs 0.231 ms (world 4) — flat, which is what the mechanism predicts: N-1
peer stores issued back to back from one thread, sharing one drain.

## What GLM gets for free, and what it does not

**Free: phase 9a.** gpt-oss has to introduce a GPU-wide MoE barrier before the
fold, because a W2 tile for any expert can execute on any XCD. GLM's
multi-layer path *already* pays exactly that barrier — the layer-entry barrier
at `FULL_LAYER_ENTRY_SLOT`, added for task #14 because Phase 1 zeroes
`moe_ws_f32` as it consumes it. Same guarantee, same arrival count
(`tiles_per_xcd * 8`). The EP fold can sit directly behind it instead of
introducing a second all-XCD sync.

**Not free: the residual fold.** This is the real work.

gpt-oss threads `ep_prev_gather` into its QKV prologue: the prologue already
makes one pass over the previous layer's output vector, so summing
`world_size` slots instead of reading one costs the extra loads and nothing
else. GLM's equivalent pass is `_rnlm8_resadd_norm_rcp`, which fuses
residual-add + RMSNorm + FP8 quant into a single pass and reads
`(moe_ws_f32, residual)`. Under EP that pair has to become
`sum over p of ep_gather[p]`, with the residual folded in by exactly one rank
(`EP_FOLD_PE`) so it survives the cross-rank sum exactly once.

So the change is inside a fused three-way helper, not at a call site. That is
the one place where "use as much code as possible from gpt-oss" does not
directly apply — the surrounding mechanism ports verbatim, this pass does not.

**Done.** `_rnlm8_resadd_norm_rcp` now takes `EP_PEER_SLOTS` / `EP_SLOT_ELEMS`,
plumbed up through `gang_rmsnorm_linear_mxfp8_bias_kernel`,
`gang_mla_attn_fused_kernel_mi300` and the monolith, which selects
`input_ptrs[29]` (`ep_prev_gather`) over `input_ptrs[0]` when
`EP_WORLD_SIZE > 1`. At world size 1 every template argument defaults away and
the emitted code is unchanged.

One divergence from gpt-oss, deliberate: GLM does **not** read the f32
workspace under EP. gpt-oss can, because its fold zeroes the workspace as it
folds, so the term is a known zero. GLM zeroes it a phase later, in
`gang_oproj_router_fused_mi300.cuh`, so at Phase 1 it still holds this rank's
partial — which the fold has already published into this rank's gather slot.
Adding it would double-count. Skipping it also drops 8 KB of loads per layer.

`tests/standalone/test_ep_prologue_sum` measures both halves, world 8,
hidden 2048:

| check | result |
|---|---|
| EP sum vs a CPU replay of the same bf16 sum | bit-identical, 2048/2048 |
| EP vs the single-GPU f32-workspace path | rel L2 3.0e-3, max abs 7.8e-3 |
| `rms_rcp`, EP vs single-GPU | rel 9.7e-5 |

The second row is not a bug and cannot be driven to zero: EP rounds each
rank's partial to bf16 *before* the sum, the single-GPU path rounds once
*after*. Eight bf16 roundings in quadrature is ~1.1e-2 of the partial
magnitude, and the partials are ~1/8 of the row. This is the numerical price
of the design, and it is charged once per layer.

## Counter slot map (GLM monolith)

`FULL_LAYER_COUNTER_SLOTS = 96`, each slot a 64-byte line:

```
[ 0.. 9] attention qkv_a -> q_b        [40..49] o_proj -> router
[10..19] q_b -> decode                 [50..59] routing-ready epoch
[20..29] decode -> merge               [60..69] MoE W13 -> W2
[30..39] attention -> o_proj           [70]     router TopK arrivals
[71..78] layer-entry release flags     [79]     layer-entry arrivals
[80..87] EP exit release flags         [88]     EP fold arrivals
```

The EP slots are allocated unconditionally. Undersizing a counter buffer does
not fail loudly, it corrupts whatever torch allocated next — gpt-oss lost time
to exactly that (832 -> 1216 ints, §3c of its notes).

## Parallelism choice

`ATTN_DP=1` + `MOE_EP=1` with `ep_slice`, mirroring gpt-oss.

`EP_SLOT` is not available to GLM-4.7-Flash: it partitions the *activated*
top-k list by slot, and top-4 cannot span 8 ranks. `ep_slice` partitions the
expert-id space instead — 64 experts / 8 ranks = 8 each, divides cleanly.

TP8 attention was considered and rejected. It is better on bytes
(4.54 GB/GPU/token vs 15.59 for DP+EP8, so a 0.94 ms roof vs 3.24 ms) but it
adds a *second* collective per layer, and at the measured ~21 us/sync that is
78 extra syncs for GLM-5.2 — about 4.2 ms for TP8+EP against 4.8 ms for DP+EP.
Both are dominated by sync count, not by bytes, which is the same conclusion
gpt-oss reached: "a faster EP needs FEWER syncs, not a cheaper one."

## GLM-5.2 weights: a staging problem, not a capacity one

**GPU memory is not the constraint.** 8 x 270.6 GB = 2.16 TB of HBM, ~298 MB
used per GPU. GLM-5.2 FP8 at ~692 GB is ~86 GB/GPU under EP8, against 252 GiB
each. That matches the "resident 99 GB" row in the byte budget below.

The one real constraint is **disk**: 658 GB free on `/` against ~692 GB of
shards. 3.5 TB total, 2.7 TB used, most of it not visible from inside the
container. `models--zai-org--GLM-5-FP8` in the HF cache is 11 MB -- config and
shard index only.

That constraint is soft, because nothing requires the checkpoint to be
resident on disk all at once. Download shard -> load to GPU -> delete shard
makes peak disk one shard, ~5 GB against 142 of them. The costs are one extra
network pass and a load that cannot resume without re-downloading. This is the
plan of record; stream-requantizing FP8 -> MXFP4 was considered and is not
needed, since it gives up FP8 fidelity to solve a problem that goes away.

gpt-oss-120b is likewise absent (32 KB), which is why the transport was
validated with the standalone benchmark rather than end to end. What is local:
GLM-4.7-Flash, 59 GB, real weights -- so EP comes up on Flash and on
`--random-weights` GLM-5.2 geometry first either way.

## Byte budget, GLM-5.2, fp8, per token

| item | value |
|---|---|
| attention/layer, un-absorbed | 157.4 MB |
| attention/layer, absorbed | 279.4 MB |
| one routed expert | 36.0 MB |
| dense MLP | 216.0 MB |
| total weights | 692 GB (experts alone 675 GB) |
| active/token, 1 GPU | 36.35 GB |
| DP attn + EP8 | 15.59 GB/GPU -> 3.24 ms roof, 99 GB resident |
| TP8 attn + EP8 | 4.54 GB/GPU -> 0.94 ms roof, 88 GB resident |

Absorption is a **78% penalty** at GLM-5.2 dims, because `kv_lora_rank` 512
exceeds `v_head_dim` 256. Task #27 (un-absorb `kv_b_v`/`kv_b_k`) is therefore
mandatory for GLM-5.2, not the optional tuning it is for GLM-4.7-Flash.

## Gotcha that cost time here

A stale editable install at `~/.local/.../__editable__.mirage_project*.pth`
points `import mirage` at `/home/claudeuser/mirage/python` — a *different*
checkout, on branch `amd-multi-gpu-rocshmem`, with no GLM code. Runs from
`demo/glm5` silently exercise the wrong tree and fail with
`'PersistentKernel' object has no attribute 'gang_rmsnorm_linear_mxfp8_bias_layer'`.
Both `PYTHONPATH` and `MIRAGE_HOME` must be pinned to this tree, exactly as
`demo/gpt_oss/env_common.sh` already documents. glm5 has no such script yet.
