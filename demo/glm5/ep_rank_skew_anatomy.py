#!/usr/bin/env python3
"""Where does the EP collective's 16-24 us/layer of inter-rank skew come from?

Offline. Reads the two Phase-B counter logs left by
probe_ep_collective_ceiling.sh. No GPU run.

THE QUESTION.  glm-inter-rank-skew-is-paid-once established that the EP
collective is a rank-alignment TAX: delete it and 83% reappears at the q_b
gather.  So the only way to move it is to reduce the skew itself.  That made
the skew's SOURCE the next question, and it was posed as a load-balance
question, because the obvious candidate is EP routed-expert imbalance -- at
bs=1 with k=8 experts over 8 ranks the per-rank count is ~1 with a max near 3.
But glm-ep-routed-imbalance-is-floor-not-wall prices that at 1.05 us of
makespan spread against a 16-24 us wait.  Those do not close.

WHAT THIS SCRIPT SHOWS: they do not close because THE PREMISE IS WRONG.  There
is no static rank imbalance to find.

SPAN CHOICE, and why it is legal.  glm-ep-peer-poll-is-not-the-cost forbids
reading cross-rank ARRIVAL ORDER out of per-rank stamps, because every stamp is
measured from that rank's own layer-entry barrier and the eight barriers are
not synchronized.  That kills differencing the WAIT (S3->S4) across ranks to
ask who published last.  It does not kill comparing a rank's own WORK spans to
another rank's own work spans: S0->S3 is "how long this rank took to fold and
publish, on its own clock", and S0->S14 is "how long this rank's layer took, on
its own clock".  Those are durations, not timestamps, so no shared clock is
needed.  This script only ever compares durations.

===================== RESULT, 2026-08-22 =====================
Per-rank, us/layer, median over each slot's writers, base arm:

  span                          r0     r1     r2     r3     r4     r5     r6     r7  spread
  S0->S3  local fold          4.68   4.60   4.38   4.54   4.48   4.54   4.49   4.68    0.30
  S0->S14 layer span        153.86 154.90 155.18 154.48 155.59 154.87 155.01 155.06    1.73
  S3->S4  peer wait          10.02  15.14  20.35  21.94  23.94  20.54  20.19  20.77   13.92

  ablated arm, same two work spans:
  S0->S3  local fold          5.19   4.95   4.98   4.84   4.91   4.95   4.92   4.76    0.43
  S0->S14 layer span        150.85 152.24 152.32 152.06 152.42 151.76 152.05 152.51    1.67

THE FINDING.  Every rank finishes its local EP fold within 0.30 us of every
other, and every rank's MEAN layer takes within 1.73 us of every other.  Yet
the mean peer wait is ~20 us.  A static imbalance model cannot produce that: if
rank R were simply slow, its slowness would show up in its own S0->S14, and the
largest such difference in the machine is 1.73 us.

So the skew is not BIAS, it is VARIANCE.  Each of these numbers is a mean over
~38300 samples (cnt/sum counters; the instrument cannot report a distribution).
The ranks have equal MEANS and the wait is driven by the per-layer MAX:

    E[max over 8 ranks] - E[own]  is large when per-layer variance is large,
    even when all eight means are identical.

Every layer, some rank happens to be the slow one, and all eight pay it.  It is
a different rank each layer, which is exactly why it averages out of S0->S14
and cannot be seen in any per-rank mean.

CONSERVATION CHECK, which supports the same reading.  Total cross-rank wait per
layer, summing the layer's two cross-rank rendezvous:

    base      S0->S1 24.14 + S19->S20  9.35 = 33.49 us/layer
    ablated   S0->S1  7.51 + S19->S20 23.23 = 30.74 us/layer

Deleting one of the two rendezvous outright left the TOTAL cross-rank wait
essentially unchanged (-8%).  A fixed per-layer jitter tax, collected once, is
the model that predicts this; "two independent barrier costs" is not.

CONSEQUENCE, and it revises the recommendation this run published an hour ago.
glm-inter-rank-skew-is-paid-once closed by saying the next EP work is a
load-balance question.  THAT IS WRONG and this supersedes it.  You cannot
balance away a variance term -- the means are already balanced to 1.1%.  Any
lever that equalizes per-rank WORK is dead on arrival.  What would move it is
reducing per-layer variance, or decoupling the ranks so one slow layer on one
rank does not stall the other seven.  The second is the structural direction;
it is also what MPK_ML_REPLAY-style pipelining across layers would buy.

WHAT THIS SCRIPT CANNOT DO.  The BARSTAGEWS counters are cnt+sum only, so they
report a mean and no distribution.  The variance claim above is an INFERENCE
from means -- it is the only model consistent with (spread of means 1.73) and
(mean wait 20), but it is not a direct measurement of the variance.  Measuring
it directly needs a per-layer stamp dump, not an accumulator.  Do not quote a
variance number from this file; quote the two spreads and the wait.
"""
import re
import collections
import sys

ARMS = ("base", "epablate")
LOGS = {a: f"/tmp/epabl_ctr_{a}.log" for a in ARMS}
PAT = re.compile(r"\[1,(\d+)\].*BARSTAGEWS (\d+) (\d+) (\d+) (\d+)")


def load(path):
    d = collections.defaultdict(lambda: collections.defaultdict(dict))
    for ln in open(path, errors="ignore"):
        m = PAT.search(ln)
        if m:
            r, s, w, c, t = (int(x) for x in m.groups())
            if c:
                d[r][s][w] = t / c / 1000.0
    return d


def med(vals):
    v = sorted(vals)
    return v[len(v) // 2]


# Work spans only -- durations on each rank's own clock, never timestamps.
# S3->S4 is a WAIT and is printed for contrast, not compared as an arrival.
SPANS = [
    (0, 3, "S0->S3  local fold, own clock"),
    (0, 14, "S0->S14 layer span, own clock"),
    (3, 4, "S3->S4  peer wait (contrast; NOT an arrival order)"),
]

R = {}
for a in ARMS:
    try:
        R[a] = load(LOGS[a])
    except FileNotFoundError:
        sys.exit(f"missing {LOGS[a]} -- run probe_ep_collective_ceiling.sh first")
    if not R[a]:
        sys.exit(f"{LOGS[a]} has no BARSTAGEWS rows")

for a in ARMS:
    print("=" * 96)
    print(f"{a}: per-rank spans, us/layer (median over each slot's writers)")
    print("=" * 96)
    print(f"{'span':<52}" + "".join(f"{r:>7}" for r in range(8)) + f"{'spread':>9}")
    for x, y, lab in SPANS:
        row = []
        for r in range(8):
            rows = R[a].get(r, {})
            row.append(med(rows[y].values()) - med(rows[x].values())
                       if x in rows and y in rows else None)
        vals = [v for v in row if v is not None]
        if not vals:
            continue
        print(f"{lab:<52}"
              + "".join(f"{v:7.2f}" if v is not None else "      -" for v in row)
              + f"{max(vals) - min(vals):9.2f}")
    print()

# Conservation of the total cross-rank wait across the two rendezvous.
print("=" * 96)
print("TOTAL CROSS-RANK WAIT PER LAYER (the two rendezvous summed), us/layer")
print("=" * 96)
for a in ARMS:
    tot = []
    for r in range(8):
        rows = R[a].get(r, {})
        ep = (med(rows[1].values()) - med(rows[0].values())
              if 0 in rows and 1 in rows else None)
        d = set(rows.get(20, {}))
        qb = None
        if 19 in rows and 20 in rows and d:
            qb = (med([v for w, v in rows[20].items() if w in d])
                  - med([v for w, v in rows[19].items() if w in d]))
        if ep is not None and qb is not None:
            tot.append((ep, qb))
    if tot:
        e = sum(t[0] for t in tot) / len(tot)
        q = sum(t[1] for t in tot) / len(tot)
        print(f"  {a:<10} S0->S1 {e:6.2f} + S19->S20 {q:6.2f} = {e + q:6.2f}")
print("""
  Deleting one of the two rendezvous outright leaves the TOTAL essentially
  unchanged. A fixed per-layer jitter tax collected once predicts this; two
  independent barrier costs does not.""")
