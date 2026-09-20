# SPX+NPS2 was silently computing the wrong answer

## Result

`MPK_AID_SPLIT_OUT` makes SPX+NPS2 produce output bit-identical to SPX+NPS1.
Without it, SPX+NPS2 produces a different continuation on essentially every run.

Gate: `grep -a assistantanalysis <log> | tail -1 | tr -d '\r\n' | md5sum`.
SPX+NPS1 reference = `3d54adb71e19`, deterministic over 3 runs including one
with `MPK_PHASE_SLOTS` at a completely different timing profile.

| configuration | runs | distinct outputs | matches NPS1 |
| --- | --- | --- | --- |
| SPX+NPS1 | 3 | 1 | -- |
| SPX+NPS2 + `MPK_AID_SPLIT_OUT` | 8+ | 1 | yes, bit-identical |
| SPX+NPS2 baseline | 16 | ~11 | never |

Decode ends in a cross-XCD argmax reduce, so this is greedy sampling: there is
no randomness to explain the variation.

`MPK_PHASE_SLOTS` HIDES the race -- both arms matched NPS1 under it. Judge
correctness on uninstrumented runs only.

## Why

AMD's own coherence documentation (Greathouse, "MI300 Coherence between
Partition and NPS Modes"; "Cache coherence summary", MI350 Plan) states that
NPS2 requires at least two compute partitions, and that a partition is coherent
only with the memory region it is "close" to. SPX+NPS2 is one partition over two
regions, so one region is always far, and far memory is `MTYPE_NC`. That is not
a driver policy we can tune: there is no configuration in which one partition
gets `MTYPE_RW` on both regions.

An `MTYPE_NC` line "cannot be invalidated by probes coming from other GPUs". The
O-proj consumer's acquire for `attn_proj_out` is a plain vL1-only `buffer_inv`,
justified in-tree as sufficient given the Phase 9 layer barrier. That argument
rests on eviction pressure, not on a guarantee, so a stale L2 copy of a hidden
-state column another XCD has since republished can survive into the RMSNorm.
The failure signature is the one the code comment itself predicts: "rows of
rmsnorm_out differing between two runs whose attn_proj_out is bit-identical".

On an AID-local `MTYPE_RW` replica the producer's write-through store generates
a probe that does invalidate the reader's line, because the line is homed in the
reader's own memory region -- RW's coherence domain. The acquire becomes sound in
hardware rather than by luck.

## Shape

Worker (xcd, wg) owns 368 of the 2880 bf16 columns; all 184 workers then read
the whole 5760 B row. So each producer publishes its slice into both AID
replicas (two extra write-through stores next to the existing one) and every
consumer reads the copy homed in its own range.

Perf, uninstrumented, interleaved 3x3, paired: **-1.32%, t = -7.54**. Small
because the row is only ~1 MB of reads per layer -- the O-proj phase cost is
synchronisation, not data movement. The correctness fix is the point.

## Rejected, measured

- `MPK_AID_SPLIT_ATTNOUT`, two shapes: post-merge slice copy **+3.79%**
  (t=+3.71), and publish inside `merge_splitkv_ck_fmha` next to its own store,
  which needs no new barrier, **+3.25%** (t=+3.55). Both lose by the same
  margin, so the cost is the extra write traffic on the merge's critical path.
  The attn_out consumer reads only the two slices its wave needs, so the
  publish:read ratio is far worse than the O-proj row's.
- `MPK_AID_SPLIT_ROUTING`: -1.17%, t=-1.00, i.e. noise.
- Finer driver-level interleave: per-workgroup skew improves only from a 1.19 to
  a 1.12 cluster ratio, and at 256 KiB the model never finishes loading
  (min_block_size pins ~600k buddy blocks for 155 GB of weights).

All three are macro-gated and default off, so the default build is unchanged.
