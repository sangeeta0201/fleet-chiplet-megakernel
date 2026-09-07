# Captured measurements

Node `mi355x-thor-2`, container `fleet_v1`, GPU 6, tree `9d478ea` plus
`slicewait-probe.patch`. All timings in microseconds per layer; ticks converted
at 10 ns (`s_memrealtime` at 100 MHz).

## Five-bucket comparison, NPS1 vs NPS2

`./drive_phase7 --layers=300`, medians over 1680 samples per arm after
discarding the first 30%.

```
bucket                           np1      np2   ratio   |   model1   model2   ratio
----------------------------------------------------------------------------------
slicewait (cross-XCD attn)      0.36     9.68   26.9x   |     0.84    23.04   27.4x
mfma (O-proj GEMM)              4.00     4.66    1.2x   |     1.56     5.92    3.8x
bar (hier. barrier)             1.72     6.28    3.7x   |     2.96    11.12    3.8x
rmsnorm+router                  1.52     1.48    1.0x   |     1.64     2.56    1.6x
topk (+wait)                    2.00     2.20    1.1x   |     0.80     1.28    1.6x
TOTAL Phase 7                   9.66    27.52    2.8x   |     7.96    57.88    7.3x
----------------------------------------------------------------------------------

slicewait spread  nps1 p10 0.36 p50 0.36 p90 0.52 max 0.84
                  nps2 p10 6.12 p50 9.68 p90 15.48 max 23.96
```

The `model*` columns are the full 36-layer instrumented run, included so the
standalone numbers can be checked against the workload they are standing in for.

## Poller sweep, NPS2

`./drive_phase7 --layers=200 --tiles=N`. Polling waves = 8 XCDs x tiles x 4
waves per block.

```
tiles=23  blocks=184  pollingwaves=736   slicewait_p50=7.76    bar_p50=6.28
tiles=12  blocks=96   pollingwaves=384   slicewait_p50=4.36    bar_p50=3.84
tiles=4   blocks=32   pollingwaves=128   slicewait_p50=1.48    bar_p50=2.44
tiles=1   blocks=8    pollingwaves=32    slicewait_p50=0.64    bar_p50=2.44
```

Per-wave cost: 0.0105, 0.0114, 0.0116, 0.0200 us. Flat within ~10% over the
first three points, i.e. the wait is proportional to the number of pollers. At
32 pollers NPS2 sits at 0.64 us against NPS1's 0.36 us at 736 pollers, so most
of the mode gap disappears once the queue is short.

## Raw sample lines

NPS1, steady state:

```
[OPROJ_INNER] slicewait=0.36 mfma=1.44 bar=2.32 rmsnorm_router=1.68 topk=0.76 total=6.56
[OPROJ_INNER] slicewait=0.40 mfma=1.64 bar=3.00 rmsnorm_router=1.56 topk=0.96 total=7.56
[OPROJ_INNER] slicewait=0.36 mfma=1.48 bar=1.80 rmsnorm_router=1.52 topk=0.88 total=6.04
```

NPS2, steady state:

```
[OPROJ_INNER] slicewait=49.84 mfma=1.24 bar=9.12 rmsnorm_router=1.76 topk=1.60 total=63.56
[OPROJ_INNER] slicewait=72.08 mfma=1.24 bar=9.44 rmsnorm_router=3.24 topk=0.52 total=86.52
[OPROJ_INNER] slicewait=53.44 mfma=1.28 bar=9.40 rmsnorm_router=2.36 topk=5.44 total=71.92
```

Those NPS2 lines are from the full model run rather than the standalone harness,
which is why the absolute wait is larger than the 9.68 us median above: the
model has real producers whose completion times vary, adding arrival skew on top
of the queueing the standalone run isolates.

## Prior narrowing

| step | result |
| --- | --- |
| Decode baseline, same tree | NPS1 1.532 ms/iter, NPS2 10.378 ms/iter |
| Weight placement sweep | AID0 9.322, allocator-scattered 10.430, AID1 12.823 ms/iter; both AID arms served 114.04 GiB with 0 passed through |
| 12-slot phase profiler | slot 6 18.33 -> 144.01 us (7.9x, 68% of regression); slot 7 35.77 -> 238.74 us (6.7x, 16%); QKV GEMM 0.97x |
| Phase profiler self-check | 1.000 ns/tick NPS1, 0.999 ns/tick NPS2; reconstructed iteration time within 3.5% of measured |

## Real kernel, the fix applied (drive_phase7, SPX+NPS2)

`drive_phase7` calls the production
`gang_linear_mxfp4_res_bias_rmsnorm_topk_kernel`, which takes
`attn_slice_release` as a parameter while the harness performs the producer
store -- so the release-flag fix lands entirely in the harness, with no change
to the fleet headers. One AID-local COHERENT replica per AID, producer publishes
into both, each XCD polls the replica homed in its own AID.

`./drive_phase7 --layers=200 --tiles=23 --aid=N --coherent=N --split=N`,
`aid_local_xcp_nc=1 aid_local_flag_mtype=2`, 1600 [OPROJ_INNER] samples per cell.

```
bucket            NPS2 base   NPS2 +fix   NPS1 ref
slicewait             14.16        0.36       0.36
bar                    6.56        3.80       2.32
mfma                   3.40        4.08       1.44
rmsnorm_router         1.52        1.56       1.68
topk                   2.16        1.80       0.76
total                 29.52       11.56       6.56
```

slicewait matches the NPS1 reference exactly. Two second-order effects:

- `bar` improved 6.56 -> 3.80 without being touched. Its lines live in
  `counters`, still a plain hipMalloc and still MTYPE_NC. Removing ~1472
  uncached polls per layer freed the channels that the barrier traffic shares.
  The same collateral shows up in bench_phase7's `read` bucket, 0.40 -> 0.08.
- `mfma` went 3.40 -> 4.08. The bottleneck moved: the slow flag wait used to
  stagger the waves, and once it is gone they arrive together and contend for
  the still-NC weight buffer.

Remaining gap to NPS1 is `bar` (3.80 vs 2.32) and `mfma` (4.08 vs 1.44). The
`bar` half needs the arrival/release sites inside the fleet headers to publish
into both replicas, since those addresses are formed from `counters` rather than
passed in.

## bench_phase7, NPS1 reference in the same session

`run_phase7_nps1.sh`, LAYERS=3600 TILES=23, one module for both modes so only
the partition mode differs.

```
                       wall us/layer   slicewait p50   read p50
SPX+NPS1, hipMalloc             4.26            1.04       0.08
SPX+NPS1, AID-split             4.05            1.00       0.08
SPX+NPS2, hipMalloc            30.x            14.36       0.40
SPX+NPS2, AID-split coherent    3.83            0.80       0.08
```

`read` is 0.08 in every cell that is not flag-starved, which is why replicating
`attn_out` itself buys nothing -- see the dupdata rows below.

## Replicating the slice data buys nothing (bench_phase7 --dupdata)

`attn_out` is genuinely cross-AID: every consumer reads all eight slices, so
four of them are remote whatever the placement. Replicating it into both AIDs
and pointing each XCD at its own copy was measured and rejected.

```
                                 wall us/layer   slicewait p50   read p50
--dupdata=0 (single, NC)                  3.81            0.84       0.08
--dupdata=1 (per-AID pair, NC)            3.87            0.88       0.08
--dupdata=2 (per-AID pair, coherent)      3.87            0.92       0.08
```

The read is 2 KiB per wave issued as eight independent coalesced loads retired
under one s_waitcnt, i.e. a single pipelined memory latency (~80 ns) whether
local or remote -- there is no queue to shorten. Duplication only adds a second
producer store, which is why wall goes slightly up.
