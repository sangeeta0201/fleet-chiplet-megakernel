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
Per-rank, us/layer, COUNT-WEIGHTED mean over each slot's writers, base arm.
WORK spans are bracketed by stamps with no cross-rank wait between them; WAIT
spans contain one.  The distinction is the whole point -- see THE TRAP below.

  span                                   r0     r1     r2     r3     r4     r5     r6     r7  spread
  WORK  S0->S3   local EP fold         4.54   4.49   4.42   4.53   4.54   4.42   4.46   4.51    0.12
  WORK  S1->S19  attention            28.09  28.37  28.69  28.15  28.68  28.41  29.22  28.35    1.13
  WORK  S20->S14 decode..MoE+boundary  98.24  89.76  89.22  89.91  88.22  89.37  90.17  88.13   10.11
  WAIT  S0->S1   EP bracket            15.78  25.26  25.46  24.71  26.84  25.05  23.75  26.35   11.06
  WAIT  S19->S20 q_b gather            12.60  12.49  12.82  12.93  12.96  13.10  12.83  13.34    0.85
  WAIT  S3->S4   peer wait (leaders)    9.71  19.24  19.50  18.65  20.81  19.24  17.76  20.47   11.10
  TOTAL S0->S14  layer span           154.71 155.88 156.19 155.69 156.70 155.92 155.96 156.17    1.99

WHY COUNT-WEIGHTED, and the first version of this file got it wrong.  Slot 4
(the EP peer wait) is stamped by exactly ONE worker per layer -- whichever XCD
leader actually waited -- and WHICH worker varies layer to layer.  Its 8
writers hold 108 to 11433 samples on rank 0, totalling 38835, exactly 1/8 of
slot 3's 310688 (ratio 8.00, checked).  An unweighted median over 8 per-worker
means weights the 108-sample worker like the 11433-sample one.  Weighting by
count is the unbiased per-sample mean and is the only correct estimator when a
slot's writers are not interchangeable.  It moved the headline bias from 10.17
to 8.98 -- CLOSER to the 9.09 measured independently by the subphase counters.

THE TRAP, and this file first fell into it.  An earlier version of this header
argued "every rank's mean layer span agrees to 1.73 us, so there is no static
imbalance; the skew must be variance."  THAT ARGUMENT IS INVALID.  The ranks
are locked together by the per-layer rendezvous, so their layer PERIODS are
forced equal -- a systematically slow rank does not show up as a longer layer,
it shows up as its PEERS waiting longer.  Equal S0->S14 across ranks is what a
working rendezvous looks like; it carries no information about imbalance.
Discriminating needs WORK spans between cross-rank sync points.  It does, and
the answer flips.

THE FINDING: the ~19.4 us peer wait is HALF static bias, half residual.

  rank 0 work excess, S20->S14 vs peer mean     +8.98 us/layer
  rank 0 wait deficit, S3->S4 vs peer mean      -9.67 us/layer

Those close to 0.69 us.  Rank 0 does ~9 us/layer more work in the MoE half and
waits ~9 us less at the rendezvous; its seven peers pay that as idle.  Rank 0
is the SHARED-EXPERT rank, and glm-shared-expert-is-the-ep-straggler
independently priced that excess at 9.09 us/layer -- 8.98 here is a 1.2%
replication from a different instrument.  Decomposition of the 19.38 us peer
mean wait:

  rank-0 shared-expert static bias    8.98 us/layer   0.683 ms/token
  peer-to-peer static spread          2.04 us/layer   (inside the above)
  residual, not explained by bias    10.40 us/layer   0.790 ms/token

So it is neither "all bias" nor "all variance" -- it is a ~50/50 split, and each
half is worth ~0.7-0.8 ms/token.  The halves need completely different work.

LOCALIZED, and this is the part that reopens a closed lever.  See
ep_bias_localize.py: splitting S20->S14 at every stamp puts the entire excess
in the two MoE GEMMs and nowhere else --

  S5->S6  W13 tiles   r0 11.82  peers  5.80   +6.02   (peer spread 0.64)
  S7->S8  W2  tiles   r0  7.54  peers  4.12   +3.41   (peer spread 0.36)
  the other 16 regions of the MoE half        -0.37 .. +0.54

+9.43 of the excess, at ~9x the peer spread, in exactly the two phases the
shared expert's FFN runs in, split 1.77:1 -- which is the gate+up : down work
ratio.  That is as clean an attribution as this instrument can give.

THE ABLATED ARM REPLICATES THE WHOLE DECOMPOSITION AT A DIFFERENT RENDEZVOUS,
which is the strongest single piece of evidence here.  Compile out the EP peer
wait and the layer's first cross-rank sync becomes the q_b gather.  The bias
follows it:

  arm        first sync   r0 WORK excess   r0 WAIT deficit   closes to
  base       S3->S4             +8.98            -9.67         0.69 us
  ablated    S19->S20           +9.14           -10.43         1.29 us

Rank 0's MoE-half work is unchanged (98.24 -> 98.26) -- so the excess is real
work, not an echo of the wait it happens to be collected at.  And S19->S20 goes
from a 0.85 spread with r0 mid-pack to an 11.93 spread with r0 lowest.  Same
rank, same ~9 us, different rendezvous.  That is exactly what
glm-inter-rank-skew-is-paid-once predicts, now with the straggler NAMED.

(The ablated arm's residual rises to 18.93; do not read that as a finding. That
arm has one fewer sync point, so misalignment accumulates further before it is
collected, and it is wrong-output by construction.)

CONSERVATION CHECK.  Total cross-rank wait per layer, both rendezvous summed:

    base      S0->S1 24.19 + S19->S20 10.76 = 34.95 us/layer
    ablated   S0->S1  7.54 + S19->S20 24.64 = 32.19 us/layer

Deleting one of the two rendezvous outright left the TOTAL essentially
unchanged (-8%) -- consistent with glm-inter-rank-skew-is-paid-once: a single
per-layer alignment tax, collected once, wherever the first cross-rank sync is.

WHAT TO DO WITH EACH HALF.
  * The BIAS half REOPENS glm-shared-expert-hoist-is-zero-makespan, whose
    evidence does not reproduce.  That note concluded "rank 0's W13 makespan is
    the SHORTEST of 8" (8.55 vs ~12.5) using max_w(S6) - max_w(S5).  Run the
    SAME statistic on these logs and rank 0 is the LONGEST of 8 -- 18.11 vs
    12.27, +5.84.  It is not an order-statistic artifact: median and makespan
    agree here (+6.02 and +5.84), and they agree on W2 and on the whole MoE
    half too.  The peers barely moved (12.56 -> 12.27); rank 0 went 8.55 ->
    18.11.  The likely cause is the instrument: that measurement was recorded
    2026-08-21 11:44, and glm-stage-stamp-drop-guard-inflated-the-boundary was
    discovered at 12:00 the SAME DAY -- SIXTEEN MINUTES LATER.  It ran with the
    default 10 ms guard, under which one ml=0 sample per iteration survives
    with ~10000x weight, and its conclusion was never re-derived after the fix.
    Treat the hoist as OPEN, not closed.  (Its sibling
    glm-shared-expert-imbalance-is-absorbed is untouched by this -- that note
    is about SP3[k] being aggregate worker-seconds, a separate and still valid
    warning.  glm-shared-expert-cannot-be-row-sharded is also untouched: row
    sharding is wrong output regardless of the price.)
  * The RESIDUAL half is not attributed. It is jitter, link/mechanism floor, or
    a second bias this decomposition does not separate.

WHAT THIS SCRIPT CANNOT DO.  The BARSTAGEWS counters are cnt+sum only, so every
number here is a MEAN.  The bias half is solid -- it is a difference of means,
it closes to 0.69 us, it replicates at a second rendezvous, it localizes to the
two phases the shared expert runs in, and it agrees with an independent
instrument to 1.2%.  The residual half is a RESIDUAL and is not evidence for
any particular mechanism; calling it "variance" would be naming something this
instrument cannot see.  Splitting it needs a per-layer distribution, which
needs a histogram, not an accumulator.  The aggregate BARSTAGE min/max is no
help: its max sits at ~890 us against the 1 ms drop guard, an extreme value
over 9M samples.

NOTE ON ADDITIVITY.  ep_bias_localize.py's sub-regions do NOT sum to S20->S14
(116.45 vs 102.22 on rank 0 under the median estimator).  Order statistics of
different worker subsets are not additive.  Read the r0-peers COLUMN, which is
a difference of like for like; do not read the column sum as a budget.
"""
import re
import collections
import sys

ARMS = ("base", "epablate")
LOGS = {a: f"/tmp/epabl_ctr_{a}.log" for a in ARMS}
PAT = re.compile(r"\[1,(\d+)\].*BARSTAGEWS (\d+) (\d+) (\d+) (\d+)")


def load(path):
    """rank -> slot -> worker -> (count, total_ns).  Keep the COUNT."""
    d = collections.defaultdict(lambda: collections.defaultdict(dict))
    for ln in open(path, errors="ignore"):
        m = PAT.search(ln)
        if m:
            r, s, w, c, t = (int(x) for x in m.groups())
            if c:
                d[r][s][w] = (c, t)
    return d


def wmean(rows, s):
    """COUNT-WEIGHTED mean stamp over a slot's writers, us.

    Why weighted, and this bit me.  Slot 4 (the EP peer wait) is stamped by
    exactly ONE worker per layer -- whichever XCD leader actually waited -- and
    WHICH worker that is varies layer to layer.  Its 8 writers therefore hold
    wildly unequal sample counts (108 to 11433 on rank 0) even though the total
    is a clean 38835, exactly 1/8 of slot 3's 310688.  An unweighted median
    over those 8 per-worker means weights a 108-sample worker the same as an
    11433-sample one.  Weighting by count is the unbiased per-sample mean and
    is the only correct estimator when a slot's writers are not interchangeable.
    """
    if s not in rows:
        return None
    c = sum(x[0] for x in rows[s].values())
    t = sum(x[1] for x in rows[s].values())
    return t / c / 1000.0 if c else None


def span(rows, x, y):
    a, b = wmean(rows, x), wmean(rows, y)
    return None if a is None or b is None else b - a


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
    print(f"{a}: per-rank spans, us/layer (count-weighted mean over each slot's writers)")
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
