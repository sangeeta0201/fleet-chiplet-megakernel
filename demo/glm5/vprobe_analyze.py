#!/usr/bin/env python3
"""Turn [VPROBE] expert lists into V, the MoE row fold's ceiling.

The probe emits one line per router invocation on rank 0:

    [VPROBE] e=<routing epoch> U=<n_act> ids <9 expert ids, -1 padded>

Epochs are contiguous and advance once per router invocation, so invocation
i and invocation i+L are the SAME LAYER in CONSECUTIVE TOKENS, where L is the
number of MoE layers per decode iteration.  L is not assumed: the script scans
every lag and reports the one that minimises the mean union size.  Same-layer
pairs are the only ones with any reason to correlate, so the dip at lag = L is
the measurement AND its own validation -- if no lag dips below the
independence line, the answer is "no overlap" and the fold is worth nothing.

    V = 2*TOPK - |topk(t) u topk(t+1)|,   saving = (V/TOPK) * MoE_delta
"""
import sys, re, collections

TOPK = 8
N_ROUTED = 256
MOE_DELTA = 1.269   # ms, the second live row's MoE TILE share:
                    # (9.400 W13 + 7.520 W2) us/layer x 75 MoE layers.
                    # CORRECTED from 0.830 -- the region map mislabels
                    # S5->S6 as "routing" (it is the Phase 5 W13 tiles) and
                    # S6->S7 as "W13" (it is the W13->W2 barrier).  See
                    # price_moe_row_fold.py for the stamp-site proof.
ROW = 5.458         # ms, a second live row end to end
BAR = 0.5           # ms, the build/no-build bar

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vprobe/bs1.log"
rx = re.compile(r"\[VPROBE\] e=(-?\d+) U=(\d+) ids ((?:-?\d+\s*)+)")

samples = []   # (epoch, frozenset of ROUTED expert ids)
for line in open(path, errors="ignore"):
    if "[VPROBE]" not in line:
        continue
    m = rx.search(line)
    if not m:
        continue
    epoch = int(m.group(1))
    ids = [int(x) for x in m.group(3).split()]
    # drop padding and the shared expert (id >= N_ROUTED): it is replicated,
    # always present, and a fold cannot save a fetch that happens once anyway.
    routed = frozenset(i for i in ids if 0 <= i < N_ROUTED)
    samples.append((epoch, routed))

if len(samples) < 200:
    print(f"  INSUFFICIENT SAMPLES: {len(samples)}")
    sys.exit(1)

# Epochs are contiguous apart from a few step-of-2 gaps, so index BY EPOCH
# rather than by position: pairing e with e+lag then needs no assumption that
# the sample list has no holes.  (An earlier version demanded a strictly
# contiguous run and kept 75 of 38325 samples, which capped the lag scan below
# the layer count and made the answer noise.)
by_epoch = {e: s for e, s in samples}
seq = [s for _, s in sorted(samples)]

sizes = collections.Counter(len(s) for s in seq)
print(f"  samples {len(samples)}, epochs {min(by_epoch)}..{max(by_epoch)}")
print(f"  routed top-k size distribution: {dict(sorted(sizes.items()))}"
      f"   (expected {{{TOPK}: n}})")

indep = TOPK * TOPK / N_ROUTED
print(f"  independence line: E|A n B| = {TOPK}*{TOPK}/{N_ROUTED} = {indep:.3f}")


def mean_overlap(lag):
    tot = n = 0
    for e, a in by_epoch.items():
        b = by_epoch.get(e + lag)
        if b is None:
            continue
        tot += len(a & b)
        n += 1
    return (tot / n, n) if n else None


MAXLAG = 200
_raw = [(lag, mean_overlap(lag)) for lag in range(1, MAXLAG)]
lags = [(l, v[0]) for l, v in _raw if v is not None and v[1] > 500]
best_lag, best_ov = max(lags, key=lambda t: t[1])

print(f"\n  lag scan (mean |A n B| = V), lags 1..{lags[-1][0]},"
      f" >=500 pairs each:")
top = sorted(lags, key=lambda t: -t[1])[:8]
for l, v in top:
    print(f"    lag {l:4d}  V = {v:.3f}"
          f"   {'<-- same-layer, consecutive tokens' if l == best_lag else ''}")
worst = min(lags, key=lambda t: t[1])
print(f"    ...")
print(f"    lag {worst[0]:4d}  V = {worst[1]:.3f}   <-- least-overlapping lag")
allmean = sum(v for _, v in lags) / len(lags)
print(f"    mean over all lags: {allmean:.3f}  (the cross-layer / independence"
      f" floor)")

V = best_ov
save = V / TOPK * MOE_DELTA
print(f"\n  V = {V:.3f} of a possible {TOPK}"
      f"   ({100*V/TOPK:.1f}% expert reuse between consecutive tokens)")
print(f"  priced fold saving = (V/{TOPK}) * {MOE_DELTA} ms = {save:.4f} ms")
print(f"                     = {100*save/ROW:.2f}% of the {ROW} ms second live row")
print(f"  bar = {BAR} ms (needs V >= {BAR*TOPK/MOE_DELTA:.2f})"
      f"  ->  {'BUILD IT' if save >= BAR else 'NO-GO'}")
print(f"\n  and this is a BYTES ceiling: EP sharding, in-phase absorption and"
      f"\n  round quantization all discount it further, none inflate it.")
