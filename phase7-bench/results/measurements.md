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
