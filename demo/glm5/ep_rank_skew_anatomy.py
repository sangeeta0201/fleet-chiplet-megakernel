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

WHAT THIS SCRIPT SHOWS: they do not close because the imbalance is NOT in the
routed experts.  Half the wait is a single static straggler -- rank 0's SHARED
expert -- and half is unattributed.

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
Per-rank, us/layer, median over each slot's writers, base arm.  WORK spans are
bracketed by stamps with no cross-rank wait between them; WAIT spans contain
one.  The distinction is the whole point -- see THE TRAP below.

  span                                   r0     r1     r2     r3     r4     r5     r6     r7  spread
  WORK  S0->S3   local EP fold         4.67   4.60   4.38   4.52   4.47   4.51   4.47   4.68    0.30
  WORK  S1->S19  attention            27.70  28.03  28.20  27.82  28.28  27.95  28.78  27.86    1.07
  WORK  S20->S14 decode..MoE+boundary 102.22  92.69  91.94  92.81  90.89  92.19  92.96  90.86   11.36
  WAIT  S0->S1   EP bracket            15.82  25.30  25.48  24.68  26.80  25.20  23.78  26.46   10.98
  WAIT  S19->S20 q_b gather            10.52  10.41  10.84  10.79  10.70  11.07  10.77  10.95    0.66
  WAIT  S3->S4   peer wait (leaders)   10.02  15.14  20.35  21.94  23.94  20.54  20.19  20.77   13.92
  TOTAL S0->S14  layer span           153.86 154.90 155.18 154.47 155.59 154.87 155.01 155.07    1.73

THE TRAP, and this file first fell into it.  An earlier version of this header
argued "every rank's mean layer span agrees to 1.73 us, so there is no static
imbalance; the skew must be variance."  THAT ARGUMENT IS INVALID.  The ranks
are locked together by the per-layer rendezvous, so their layer PERIODS are
forced equal -- a systematically slow rank does not show up as a longer layer,
it shows up as its PEERS waiting longer.  Equal S0->S14 across ranks is what a
working rendezvous looks like; it carries no information about imbalance.
Discriminating needs WORK spans between cross-rank sync points.  It does, and
the answer flips.

THE FINDING: the ~20.4 us peer wait is HALF static bias, half residual.

  rank 0 work excess, S20->S14 vs peer mean    +10.17 us/layer
  rank 0 wait deficit, S3->S4 vs peer mean     -10.39 us/layer

Those close to 0.21 us.  Rank 0 does ~10 us/layer more work in the MoE half and
waits ~10 us less at the rendezvous; its seven peers pay that as idle.  Rank 0
is the SHARED-EXPERT rank, and glm-shared-expert-is-the-ep-straggler
independently priced that excess at 9.09 us/layer -- this is a clean
replication at 10.17 from a different instrument.  Decomposition of the peer
mean wait of 20.41 us:

  rank-0 shared-expert static bias   10.17 us/layer   0.773 ms/token
  peer-to-peer static spread          2.10 us/layer   (inside the above)
  residual, not explained by bias    10.24 us/layer   0.778 ms/token

So it is neither "all bias" nor "all variance" -- it is a 50/50 split, and each
half is worth ~0.78 ms/token.  The two halves need completely different work.

THE ABLATED ARM REPLICATES THE WHOLE DECOMPOSITION AT A DIFFERENT RENDEZVOUS,
which is the strongest single piece of evidence here.  Compile out the EP peer
wait and the layer's first cross-rank sync becomes the q_b gather.  The bias
follows it:

  arm        first sync   r0 WORK excess   r0 WAIT deficit   closes to
  base       S3->S4            +10.17           -10.39         0.21 us
  ablated    S19->S20          +10.42           -10.19         0.23 us

Rank 0's MoE-half work is unchanged at 102.28 -- so the excess is real work, not
an echo of the wait it happens to be paid at.  And S19->S20 goes from a 0.66
spread with r0 mid-pack to an 11.46 spread with r0 lowest.  Same rank, same
~10 us, different rendezvous.  That is exactly what
glm-inter-rank-skew-is-paid-once predicts, now with the straggler NAMED.

(The ablated arm's residual rises to 15.50; do not read that as a finding. That
arm has one fewer sync point, so misalignment accumulates further before it is
collected, and it is wrong-output by construction.)

CONSERVATION CHECK.  Total cross-rank wait per layer, both rendezvous summed:

    base      S0->S1 24.19 + S19->S20 10.76 = 34.95 us/layer
    ablated   S0->S1  7.54 + S19->S20 24.64 = 32.19 us/layer

Deleting one of the two rendezvous outright left the TOTAL essentially
unchanged (-8%) -- consistent with glm-inter-rank-skew-is-paid-once: a single
per-layer alignment tax, collected once, wherever the first cross-rank sync is.

WHAT TO DO WITH EACH HALF.
  * The BIAS half is the shared expert and is already well-trodden: hoisting it
    is zero makespan (glm-shared-expert-hoist-is-zero-makespan), row-sharding
    it is wrong output (glm-shared-expert-cannot-be-row-sharded), and its SP
    imbalance reads as absorbed (glm-shared-expert-imbalance-is-absorbed).
    What is NEW here is the price: measured at the rendezvous rather than in
    the phase, it is 0.773 ms/token of peer idle, not "absorbed".  Those two
    readings need reconciling before anything is built.
  * The RESIDUAL half is not attributed. It is jitter, link/mechanism floor, or
    a second bias this decomposition does not separate.

WHAT THIS SCRIPT CANNOT DO.  The BARSTAGEWS counters are cnt+sum only, so every
number here is a MEAN.  The bias half is solid -- it is a difference of means
and closes to 0.22 us.  The residual half is a RESIDUAL and is not evidence for
any particular mechanism; calling it "variance" would be naming something this
instrument cannot see.  Splitting it needs a per-layer distribution, which
needs a histogram, not an accumulator.  The aggregate BARSTAGE min/max is no
help: its max sits at ~890 us against the 1 ms drop guard, an extreme value
over 9M samples.
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
    return v[len(v) // 2] if v else None


def span(rows, x, y):
    """Median span x->y over the workers that cross BOTH stamps.

    Slot populations differ (3/4 have 8 XCD leaders, 20 has the 64-worker
    decode set, 14 has all 232), so an unrestricted median differences two
    different populations.  Intersect first.
    """
    if x not in rows or y not in rows:
        return None
    w = set(rows[x]) & set(rows[y])
    if not w:
        return None
    return med([rows[y][i] for i in w]) - med([rows[x][i] for i in w])


# Durations on each rank's own clock, never timestamps.
#
# WORK spans contain no cross-rank wait, so their per-rank spread IS the static
# imbalance.  WAIT spans contain one, and their spread is the consequence.
# Do NOT read the total layer span as evidence either way -- the rendezvous
# forces it equal across ranks by construction.  That mistake is what this
# script's first version made.
SPANS = [
    ("WORK", 0, 3, "S0->S3   local EP fold"),
    ("WORK", 1, 19, "S1->S19  attention body"),
    ("WORK", 20, 14, "S20->S14 decode..MoE+boundary [shared expert lives here]"),
    ("WAIT", 0, 1, "S0->S1   EP bracket"),
    ("WAIT", 19, 20, "S19->S20 q_b gather"),
    ("WAIT", 3, 4, "S3->S4   peer wait (8 XCD leaders)"),
    ("TOTAL", 0, 14, "S0->S14  layer span [FORCED equal -- proves nothing]"),
]

# The span carrying rank 0's shared expert, and the wait it is paid at.  Which
# rendezvous that is DIFFERS BY ARM: with the EP peer wait compiled out the
# layer's first cross-rank sync becomes the q_b gather, and the bias follows it
# there.  Stamp 4 does not exist in the ablated arm anyway.
BIAS_WORK = (20, 14)
BIAS_WAIT = {"base": (3, 4), "epablate": (19, 20)}

R = {}
for a in ARMS:
    try:
        R[a] = load(LOGS[a])
    except FileNotFoundError:
        sys.exit(f"missing {LOGS[a]} -- run probe_ep_collective_ceiling.sh first")
    if not R[a]:
        sys.exit(f"{LOGS[a]} has no BARSTAGEWS rows")

for a in ARMS:
    print("=" * 100)
    print(f"{a}: per-rank spans, us/layer (median over the workers crossing BOTH stamps)")
    print("=" * 100)
    print(f"{'kind  span':<58}" + "".join(f"{r:>7}" for r in range(8))
          + f"{'spread':>9}")
    for kind, x, y, lab in SPANS:
        row = [span(R[a].get(r, {}), x, y) for r in range(8)]
        vals = [v for v in row if v is not None]
        if not vals:
            continue
        print(f"{kind:<6}{lab:<52}"
              + "".join(f"{v:7.2f}" if v is not None else "      -" for v in row)
              + f"{max(vals) - min(vals):9.2f}")
    print()

# ---------------------------------------------------------------- the split
# Does a static bias explain the wait?  Rank 0 carries the shared expert.  If
# the bias model is right, rank 0's WORK excess over the peer mean must equal
# its WAIT deficit -- it spends at the tiles what the others spend idle.
print("=" * 100)
print("BIAS vs RESIDUAL: is the peer wait static imbalance, or not?")
print("=" * 100)
for a in ARMS:
    bw = BIAS_WAIT[a]
    work = [span(R[a].get(r, {}), *BIAS_WORK) for r in range(8)]
    wait = [span(R[a].get(r, {}), *bw) for r in range(8)]
    if any(v is None for v in work) or any(v is None for v in wait):
        print(f"  {a:<10} S{bw[0]}->S{bw[1]} absent")
        continue
    pw, pt = sum(work[1:]) / 7, sum(wait[1:]) / 7
    excess, deficit = work[0] - pw, wait[0] - pt
    spread = max(work[1:]) - min(work[1:])
    resid = pt - excess
    print(f"  {a}  (work S{BIAS_WORK[0]}->S{BIAS_WORK[1]}, "
          f"wait S{bw[0]}->S{bw[1]} = this arm's FIRST cross-rank sync)")
    print(f"    rank0 WORK excess vs peer mean   {excess:+7.2f} us/layer")
    print(f"    rank0 WAIT deficit vs peer mean  {deficit:+7.2f} us/layer")
    print(f"    the two close to                 {abs(excess + deficit):7.2f} us"
          "   <- a static bias predicts 0")
    print(f"    peer mean wait                   {pt:7.2f} us/layer")
    print(f"      of which rank0 bias            {excess:7.2f} us/layer"
          f"   {excess * 76 / 1e3:6.3f} ms/token")
    print(f"      peer-to-peer work spread       {spread:7.2f} us/layer")
    print(f"      RESIDUAL, unattributed         {resid:7.2f} us/layer"
          f"   {resid * 76 / 1e3:6.3f} ms/token")
print("""
  Rank 0 is the shared-expert rank. glm-shared-expert-is-the-ep-straggler
  priced its excess at 9.09 us/layer from the subphase counters; this is an
  independent replication from the stage stamps.
  The RESIDUAL is a residual -- it is NOT evidence for jitter, or for any other
  named mechanism. These counters are cnt+sum and report only a mean.""")
print()

# Conservation of the total cross-rank wait across the two rendezvous.
print("=" * 96)
print("TOTAL CROSS-RANK WAIT PER LAYER (the two rendezvous summed), us/layer")
print("=" * 96)
for a in ARMS:
    tot = []
    for r in range(8):
        rows = R[a].get(r, {})
        ep, qb = span(rows, 0, 1), span(rows, 19, 20)
        if ep is not None and qb is not None:
            tot.append((ep, qb))
    if tot:
        e = sum(t[0] for t in tot) / len(tot)
        q = sum(t[1] for t in tot) / len(tot)
        print(f"  {a:<10} S0->S1 {e:6.2f} + S19->S20 {q:6.2f} = {e + q:6.2f}")
print("""
  Deleting one of the two rendezvous outright leaves the TOTAL essentially
  unchanged (-8%). A single per-layer alignment tax, collected once at whatever
  the first cross-rank sync is, predicts this; two independent barrier costs
  does not. See glm-inter-rank-skew-is-paid-once.""")
