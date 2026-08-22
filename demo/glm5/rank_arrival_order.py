#!/usr/bin/env python3
"""Is the cross-rank arrival order STATIC, or does the last arriver rotate?

Offline.  Reads BARSTAGEWS counter logs already on disk.  No GPU run.

WHY THIS EXISTS.  glm-ep-skew-is-half-shared-expert-bias splits the EP skew
into an 8.98 us/layer shared-expert bias (attributed, and closed at the wall by
MPK_SHARED_DUP) and a 10.40 us/layer RESIDUAL that is NOT attributed to
anything.  At 76 layers that residual is 0.790 ms of peer idle.  The note says
splitting it "needs a per-layer histogram".  It does not -- not for the FIRST
question, which is cheaper:

    Is the residual a STATIC per-rank ordering (some rank is reliably late,
    which a work rebalance could fix), or is it JITTER (a different rank is
    late each layer, which nothing fixes)?

THE ESTIMATOR, AND WHY IT IS LEGAL.  GPU clocks on 8 separate devices are not
comparable, so you cannot subtract rank r's timestamp from rank s's and call it
an arrival delta (glm-ep-peer-poll-is-not-the-cost).  But a DURATION is
comparable across ranks, and the peer wait S3->S4 is a duration.  At a
rendezvous everyone leaves together, so:

    longer wait  <=>  arrived earlier
    SHORTEST wait <=> LAST arriver, i.e. the rank that set the barrier

So rank-ordering the per-rank S3->S4 wait ASCENDING ranks the ranks by lateness
descending.  This needs no cross-rank clock comparison at all.

WHAT MAKES IT A TEST AND NOT A DESCRIPTION.  One log shows an ordering whether
or not the ordering is real -- 8 noisy numbers always sort.  The test is
REPRODUCIBILITY across independent runs.  Three logs are available:

    /tmp/shdup1/ctr_base.log   MPK_SHARED_DUP=0
    /tmp/shdup1/ctr_dup.log    MPK_SHARED_DUP=1   } ranks 1-7 untouched in both
    /tmp/shdup2/ctr_dup.log    MPK_SHARED_DUP=2   } -- independent sessions

MPK_SHARED_DUP only ever adds W13 copies to RANK 0's shared expert.  Ranks 1-7
run byte-identical code in all three.  So the peer subpopulation gives three
INDEPENDENT samples of its own internal ordering.  If the same peer is last in
all three, the ordering is static.  If it rotates, it is jitter.

(shdup2/ctr_base.log is a COPY of shdup1's -- that arm timed out and DUP=0 is a
byte-for-byte no-op -- so it is deliberately not listed as a fourth sample.)

READ THE WORK SPANS TOO.  S5->S6, S7->S8 and S20->S14 contain no cross-rank
sync, so their per-rank spread is real work imbalance and can be compared with
the wait ordering directly.  If a rank waits least AND works most, that is one
consistent story.  If lateness does not track work, the residual is not a work
imbalance and no rebalance addresses it.
"""
import collections
import itertools
import re
import sys

PAT = re.compile(r"\[1,(\d+)\].*BARSTAGEWS (\d+) (\d+) (\d+) (\d+)")
N_LAYERS = 76

SAMPLES = [
    ("base   DUP=0", "/tmp/shdup1/ctr_base.log"),
    ("dup1   DUP=1", "/tmp/shdup1/ctr_dup.log"),
    ("dup2   DUP=2", "/tmp/shdup2/ctr_dup.log"),
]

SPANS = [
    ("WAIT", 3, 4, "S3->S4  EP peer wait   (short = LATE arriver)"),
    ("WAIT", 0, 1, "S0->S1  EP bracket"),
    ("WORK", 5, 6, "S5->S6  W13 tiles"),
    ("WORK", 7, 8, "S7->S8  W2 tiles"),
    ("WORK", 20, 14, "S20->S14 whole MoE half"),
]


def load(path):
    d = collections.defaultdict(lambda: collections.defaultdict(dict))
    for ln in open(path, errors="ignore"):
        m = PAT.search(ln)
        if m:
            r, s, w, c, t = (int(x) for x in m.groups())
            if c:
                d[r][s][w] = (c, t)
    return d


def wmean(rows, s):
    if s not in rows:
        return None
    c = sum(x[0] for x in rows[s].values())
    t = sum(x[1] for x in rows[s].values())
    return t / c / 1000.0 if c else None


def wspan(rows, x, y):
    a, b = wmean(rows, x), wmean(rows, y)
    return None if a is None or b is None else b - a


def spearman(a, b):
    """Rank correlation over the 7 peers.  n=7, so |rho| >= 0.786 is p<0.05."""
    n = len(a)
    ra = {v: i for i, v in enumerate(sorted(a))}
    rb = {v: i for i, v in enumerate(sorted(b))}
    d2 = sum((ra[x] - rb[y]) ** 2 for x, y in zip(a, b))
    return 1 - 6 * d2 / (n * (n * n - 1))


def main():
    R = {}
    for lab, path in SAMPLES:
        try:
            R[lab] = load(path)
        except FileNotFoundError:
            print(f"missing {path} -- skipping {lab}", file=sys.stderr)
    if not R:
        sys.exit("no counter logs found")

    for kind, x, y, lab in SPANS:
        print("=" * 92)
        print(f"{kind}  {lab}      count-weighted mean, us/layer")
        print("=" * 92)
        print(f"{'sample':<16}" + "".join(f"{'r%d' % r:>8}" for r in range(8))
              + f"{'peer sd':>10}{'late':>8}")
        peer_vecs = {}
        for slab in R:
            v = [wspan(R[slab].get(r, {}), x, y) for r in range(8)]
            if any(u is None for u in v):
                print(f"{slab:<16}  (slot missing)")
                continue
            peers = v[1:]
            mu = sum(peers) / 7
            sd = (sum((p - mu) ** 2 for p in peers) / 7) ** 0.5
            # last arriver among the PEERS only: shortest wait
            late = 1 + min(range(7), key=lambda i: peers[i]) if kind == "WAIT" \
                else 1 + max(range(7), key=lambda i: peers[i])
            print(f"{slab:<16}" + "".join(f"{u:8.2f}" for u in v)
                  + f"{sd:10.2f}" + f"{'r%d' % late:>8}")
            peer_vecs[slab] = peers
        # reproducibility of the PEER ordering across independent samples
        if len(peer_vecs) >= 2:
            print(f"{'':<16}peer rank-order reproducibility (Spearman, n=7, "
                  f"|rho|>=0.786 is p<0.05):")
            for a, b in itertools.combinations(sorted(peer_vecs), 2):
                rho = spearman(peer_vecs[a], peer_vecs[b])
                mark = "  STATIC" if abs(rho) >= 0.786 else ""
                print(f"{'':<16}  {a.split()[0]:>6} vs {b.split()[0]:<6}"
                      f"  rho = {rho:+.3f}{mark}")
        print()

    print("=" * 92)
    print("HOW TO DECIDE")
    print("=" * 92)
    print("""  STATIC   the same peer is 'late' in all three samples AND the pairwise
           Spearman rho on the peer wait vector is high.  Then the residual is
           a fixed per-rank imbalance and a rebalance addresses it -- priced at
           the CONVEX transfer coefficient (~0.3 near the operating point,
           glm-peer-idle-to-wall-response-is-convex), NOT at face value.
  JITTER   the late peer rotates and rho is near 0.  Then the 0.790 ms residual
           is per-layer noise, no static rebalance touches it, and the EP
           residual closes as not-a-lever without building anything.
  Cross-check: if the WAIT ordering does not anti-correlate with the WORK
  ordering (S20->S14), lateness is not caused by work volume, so a work
  rebalance is the wrong instrument even if the ordering IS static.""")


if __name__ == "__main__":
    main()

# ============================================================================
# RESULT, 2026-08-22.  THE 0.790 ms "UNATTRIBUTED EP RESIDUAL" IS NOT SKEW.
# It is a UNIFORM FLOOR that the LAST ARRIVER ITSELF PAYS.  Closed, no-go.
# No GPU run -- three counter logs already on disk.
# ============================================================================
#
# S3->S4 EP peer wait, count-weighted mean, us/layer:
#
#   sample        r0     r1     r2     r3     r4     r5     r6     r7   peer sd
#   base DUP=0   9.94  19.07  19.58  19.14  19.87  19.12  19.18  20.45    0.48
#   dup1 DUP=1   8.98  22.04  22.70  21.72  23.25  22.07  22.00  22.36    0.48
#   dup2 DUP=2   7.30  25.93  27.61  26.40  26.61  26.99  25.30  27.50    0.77
#
# FINDING 1 -- RANK 0 IS THE LAST ARRIVER, AND IT STILL WAITS ~10 us/layer.
# Shortest wait = last arriver.  Rank 0 is shortest by ~9.5 us in every sample,
# which is the shared-expert bias (8.98) re-derived from the other side.  But a
# rank that arrives LAST cannot be waiting on any peer's lateness -- and rank 0
# still sits in S3->S4 for 9.94 us/layer.  That 9.94 IS the 10.40 "residual".
# So the residual was never cross-rank skew.  It is a floor every rank pays,
# the critical one included: mechanism/transport of the collective plus
# whatever intra-rank wait the span brackets.  Same shape as the 8.28 us/layer
# of UNIFORM spin whose whole rendezvous was deleted for 0.003 ms of wall
# (glm-deleting-a-whole-rendezvous-is-neutral).  Already priced, and priced at
# zero.  Do not re-open it as "0.790 ms unattributed".
#
# Corroboration from the DUP sweep: as rank 0 is loaded it gets later and its
# own wait FALLS 9.94 -> 8.98 -> 7.30 while the peers' rises 19.49 -> 22.31 ->
# 26.62.  Rank 0's wait is decaying toward a floor, not toward zero: it shed
# only 2.64 us for 8.73 us of added lateness.
#
# FINDING 2 -- THE 7 PEERS ARE UNIFORM TO 0.5%.  peer sd 0.48 us/layer on a
# 19.5 us wait; full range 20.45 - 19.07 = 1.38 us/layer.  The peer ordering
# does NOT reproduce: pairwise Spearman over the three independent samples is
# +0.607 / +0.500 / +0.679, all under the n=7 significance bar of 0.786.  All
# three are positive, so a weak static component may exist -- but it is capped
# by the range regardless of significance:
#
#     1.38 us/layer x 76 layers   = 0.105 ms of PEER IDLE
#     x 0.32 transfer coefficient = 0.034 ms of wall
#
# (0.32 is the coefficient at the operating point, and it FALLS going left --
# glm-peer-idle-to-wall-response-is-convex.)  A perfect static rebalance across
# ranks 1-7 is worth ~0.03 ms.  Dead an order of magnitude under the 0.26 floor.
#
# FINDING 3 -- THE WORK SPANS AGREE.  Peer sd, us/layer: W13 0.18 on 6.2,
# W2 0.24 on 5.2, whole MoE half 0.45 on 88.4 (0.5%).  Routed-expert work is
# balanced across the 7 peers to half a percent, independently confirming
# glm-ep-routed-imbalance-is-floor-not-wall from per-rank stamps rather than
# from byte counts.  And the WORK ordering does not reproduce either (rho
# +0.036 to +0.679), so even the sign of a rebalance is not established.
#
# WHAT THIS RETIRES.  glm-ep-skew-is-half-shared-expert-bias said splitting the
# 10.40 us residual "needs a per-layer histogram".  It does not, and the
# histogram is not worth building: the split that mattered was
# last-arriver-floor vs peer-ordering, and a duration comparison answers it
# with logs that already existed.  Both halves of the 19.38 us/layer peer wait
# are now closed -- 8.98 by MPK_SHARED_DUP at the wall, 10.40 by this.
#
# THE REUSABLE METHOD.  Cross-rank arrival ORDER is unobtainable from per-rank
# stamps (unsynchronized clocks, glm-ep-peer-poll-is-not-the-cost) but arrival
# LATENESS is: at a rendezvous everyone leaves together, so the wait DURATION
# ranks the ranks and durations are clock-free.  And the last arriver's own
# residual wait separates "skew" from "floor" in one number.
