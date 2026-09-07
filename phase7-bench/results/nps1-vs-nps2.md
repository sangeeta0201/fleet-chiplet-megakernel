# Phase 7 breakdown: NPS1 vs NPS2, before and after the sync fixes

All three runs come from **one binary**, measured on mi355x-thor-2, SPX, patched
driver, `--tiles=23` (184 blocks), 400 layers, medians over 2240 samples after
discarding the first 30% as warmup. NPS1 was measured by switching the node's
memory partition and switching back, not by reusing older numbers  the gate
changes shifted NPS2's total from 10.18 to 10.68 us, which is the same order as
the gap being argued about, so a mixed-build comparison would not have settled
anything.

Configurations: NPS1 and NPS2 base are both `--aid=0`, the stock path with one
copy of the flags in normal HBM. NPS2 fixed is
`--aid=1 --coherent=1 --split=1`: flags AID-local under a coherent MTYPE, with
one replica per memory partition and each half of the XCDs polling its own.

## All three compute the same thing

Every run, in both modes, in all three configurations:

```
attn_proj_out = d5f0c50124e2cb83    rmsnorm_out = ff1bdbe7ad2c4ed3
```

This is the gate from `correctness-gate.md`, which has two negative controls
that do change these hashes, so the agreement is evidence rather than a
tautology. It means the NPS2 fixed path is not fast because it is skipping
something.

## The breakdown

Microseconds, p50 (p10, p90):

| bucket | NPS1 | NPS2 base | NPS2 fixed | fixed vs NPS1 |
| --- | --- | --- | --- | --- |
| slicewait | 0.40 (0.40, 0.44) | 4.28 (2.24, 9.48) | 0.40 (0.36, 0.44) | +0.00 |
| mfma | 3.96 (3.92, 4.00) | 5.96 (3.80, 13.64) | 3.96 (3.92, 4.00) | +0.00 |
| bar | 1.76 (1.60, 1.96) | 6.90 (3.32, 10.32) | 2.48 (2.08, 2.72) | **+0.72** |
| rmsnorm_router | 1.52 (1.44, 1.60) | 1.44 (1.40, 1.60) | 1.64 (1.48, 1.76) | +0.12 |
| topk | 2.04 (0.48, 3.76) | 1.92 (0.44, 3.52) | 2.32 (0.48, 6.00) | +0.28 |
| **total** | **9.76** (7.96, 11.52) | **23.50** (14.20, 30.32) | **10.68** (9.04, 14.64) | **+0.92** |

NPS2 base is 2.41x NPS1. NPS2 fixed is 1.09x on this run, recovering 12.82 us
of the 13.74 us gap  93%.

### Repeatability

NPS2 fixed was measured again after the NPS1 round trip, i.e. across two driver
reloads and two partition changes:

| | slicewait | mfma | bar | rmsnorm_router | topk | total |
| --- | --- | --- | --- | --- | --- | --- |
| first | 0.40 | 3.96 | 2.48 | 1.64 | 2.32 | 10.68 |
| repeat | 0.40 | 3.92 | 2.36 | 1.60 | 1.76 | 10.04 |

`total` therefore carries about +/-0.5 us of run-to-run spread, nearly all of it
from `topk`, so the honest form of the headline is that NPS2 fixed is
**1.03x to 1.09x** NPS1 rather than any single ratio. `bar` is the stable part:
2.36 to 2.48 against NPS1's 1.76, so the +0.6 to +0.7 us floor reproduces across
driver reloads while the total does not to better than half a microsecond.

## Reading it

**`slicewait` and `mfma` are at exact parity**, p10 and p90 included, not just
the median. Both were sync-bound, not bandwidth-bound: the 184 pollers on one
cache line were the whole of `slicewait`, and `mfma`'s 5.96 in base is the
cross-AID acquire traffic on the remaining slices, not weight traffic (see the
caveat below).

**`bar` is the only real residual: +0.6 to +0.7 us.** The p10/p90 ranges do not
overlap  NPS1 (1.60, 1.96) against fixed (2.08, 2.72)  so this is a floor,
not noise. It is the part of the hierarchical barrier that cannot be
AID-split: level 1 is a per-XCD local atomic and level 2 is a single global
atomic across all eight XCDs on one shared line. Splitting the *release* flags
per partition is what removed the queueing slope (+0.056 us/poller in the
earlier sweep, -0.013 after), but the global arrival counter is inherently one
line and half the XCDs are always remote to it.

**`rmsnorm_router` +0.12 and `topk` +0.28 are noise.** Their ranges overlap
heavily, and `topk` spans (0.48, 6.00) because it folds in the TopK barrier
wait.

**Base is not just slower, it is erratic.** `mfma` spans 3.80 to 13.64 and
`slicewait` 2.24 to 9.48, against ranges of 0.08 us for both after the fix.
That variance is the signature of cross-AID contention rather than a fixed
added latency.

## The caveat that still applies

`mfma` reading 3.96 in both modes does **not** mean the O-proj weight traffic
is fine. The harness stages its weight tile into LDS from the *harness* side,
outside the kernel's `t0..t4` window, so this bucket measures the LDS read, the
FP8 quantize and the MFMA issue  never the global weight fetch that Phase 6
does in the real model. Staging real weights instead of leaving the tile empty
moved this bucket by -0.12 us, which is how little it depends on the weights
being there at all.

The model's `mfma` going 1.56 -> 5.92 is a different quantity from this one and
remains un-reproduced and open. Nothing here speaks to it.

## Reproducing

```bash
bash buckets.sh nps2_base --aid=0
bash buckets.sh nps2_fix  --aid=1 --coherent=1 --split=1
sudo -n python3 run_root.py set_mode2.sh NPS1     # ~2 min, reloads amdgpu
bash buckets.sh nps1      --aid=0
sudo -n python3 run_root.py set_mode2.sh NPS2     # restore
bash combine.sh
```
