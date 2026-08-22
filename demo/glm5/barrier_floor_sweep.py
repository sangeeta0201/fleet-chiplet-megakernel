#!/usr/bin/env python3
"""FLOOR vs SKEW for every region in the layer, in one sweep.

Offline.  Reads BARSTAGEWS per-worker rows already on disk.  No GPU run.

THE METHOD, generalized from glm-ep-residual-is-the-last-arrivers-own-floor.
That note split ONE cross-rank wait into "skew" and "floor" with a single
observation: at a rendezvous everyone leaves together, so the participant with
the SHORTEST wait is the LAST ARRIVER, and a last arriver cannot be waiting on
anyone else -- whatever it still waits is a floor the mechanism charges to
every participant.  Nothing in that argument is specific to ranks.  Applied to
the 232 workers of one GPU it splits every intra-GPU region the same way:

    wait_w = b_w - a_w                     (paired, per worker, ONE clock)
    FLOOR  = min_w wait_w                  the last arriver's own wait.  Every
                                           worker pays it, the critical one
                                           included, so it IS on the wall.
    SKEW   = max_w wait_w - min_w wait_w   idle worker time.  What a perfect
                                           arrival balance would remove -- and
                                           NOT automatically wall time.

The deletable ceiling of a barrier is SKEW, never SKEW + FLOOR: the floor is
what the mechanism costs when arrival is already perfect, and deleting a whole
rendezvous outright measured 0.003 ms
(glm-deleting-a-whole-rendezvous-is-neutral).

WHY PAIRED, NOT max(b) - max(a).  The board's estimator telescopes to the layer
span, which is what a BUDGET needs, but it is a difference of two different
workers' clocks and cannot be decomposed into a per-worker wait at all.
Floor/skew needs the same worker on both ends (paired_span.py).  The two answer
different questions and both are right for theirs.

THE STAMP IS AN ARRIVAL TIME, which is what makes any of this legal:
mpk_stage_stamp records "ns since the layer-entry barrier completed"
(mpk_atoms.cuh), so t/c is a mean arrival WITHIN the layer and a per-worker
difference of two slots is a duration.  Four stamps are marked
"NESTED -- check cnt" in the source; the sweep prints the write-count ratio so
you can see whether any slot actually fires more than once per layer.  At this
geometry none does (all 0.99-1.00x slot 0), so the durations hold.

SHAPE, read from the data rather than the label:
    RELEASE  sd(b) << sd(a)   workers arrive spread, leave together = a barrier
    SPREAD   sd(a) << sd(b)   arrive together, leave spread = a tile phase
    UNIFORM  both small       serial work or a uniform wait
A tile phase is included as a NEGATIVE CONTROL: if W13 tiles ever classifies
RELEASE, the test is broken.
"""
import collections
import re
import statistics
import sys

PAT = re.compile(r"\[1,(\d+)\].*BARSTAGEWS (\d+) (\d+) (\d+) (\d+)")
N_LAYERS = 76
XFER = 0.32  # peer-idle -> wall, glm-peer-idle-to-wall-response-is-convex

# Verified against the `// Stage stamp N:` comments at every call site,
# 2026-08-22.  These agree with board_budget.py.  They do NOT agree with
# paired_span.py's older map, which was shifted by one across 16..19 and has
# been corrected there.
ORDER = [0, 1, 2, 16, 17, 18, 19, 21, 23, 24, 25, 26, 27, 28,
         29, 30, 31, 32, 5, 6, 7, 8, 12, 13, 14]
LABEL = {
    (0, 1): "EP collective", (1, 2): "post-EP -> qkv_a",
    (2, 16): "-> attn call boundary", (16, 17): "qkv_a tiles",
    (17, 18): "qkv_a -> q_b barrier", (18, 19): "q_b tiles + KV append",
    (19, 21): "q_b->decode bar + MLA decode",
    (21, 23): "decode->merge bar + merge",
    (23, 24): "Ph8 attn -> o_proj barrier", (24, 25): "o_proj wt prefetch",
    (25, 26): "per-XCD attention release", (26, 27): "prefetch DMA retired",
    (27, 28): "entering the MoE half", (28, 29): "W_UK/W_UV + Mech-C bar",
    (29, 30): "o_proj GEMV + residual", (30, 31): "o_proj -> router barrier",
    (31, 32): "router GEMV + TopK", (32, 5): "routing-ready poll",
    (5, 6): "W13 tiles  [CONTROL]", (6, 7): "W13 -> W2 barrier",
    (7, 8): "W2 tiles", (8, 12): "MoE exit", (12, 13): "(12->13)",
    (13, 14): "layer boundary",
}


def load(path, want_rank=None):
    """-> {rank: {slot: {worker: (mean_arrival_us, cnt)}}}"""
    d = collections.defaultdict(lambda: collections.defaultdict(dict))
    for ln in open(path, errors="ignore"):
        m = PAT.search(ln)
        if not m:
            continue
        r, s, w, c, t = (int(x) for x in m.groups())
        if c and (want_rank is None or r == want_rank):
            d[r][s][w] = (t / c / 1000.0, c)
    return d


def shape_of(sda, sdb):
    if sdb < 0.5 * sda:
        return "RELEASE"
    if sda < 0.5 * sdb:
        return "SPREAD"
    if max(sda, sdb) < 0.5:
        return "UNIFORM"
    return "mixed"


def report(path, rank):
    R = load(path, rank)
    if rank not in R:
        print(f"rank {rank} not in {path}")
        return
    rows = R[rank]
    base = statistics.mean(c for _, c in rows[0].values())
    print(f"\n{path}   rank {rank}   ({N_LAYERS} layers)")
    print("=" * 106)
    print(f"{'region':<34}{'n':>5}{'x/lyr':>7}{'sd(a)':>7}{'sd(b)':>7}"
          f"{'shape':>9}{'floor':>8}{'skew':>8}{'floor':>9}{'skew':>8}")
    print(f"{'':<34}{'':>5}{'':>7}{'us':>7}{'us':>7}{'':>9}"
          f"{'us/lyr':>8}{'us/lyr':>8}{'ms':>9}{'ms':>8}")
    print("=" * 106)
    tot_f = tot_s = rel_s = 0.0
    for a, b in zip(ORDER, ORDER[1:]):
        if a not in rows or b not in rows:
            continue
        A, B = rows[a], rows[b]
        both = sorted(set(A) & set(B))
        if len(both) < 8:
            continue
        sda = statistics.pstdev([A[w][0] for w in both])
        sdb = statistics.pstdev([B[w][0] for w in both])
        waits = sorted(B[w][0] - A[w][0] for w in both)
        floor, skew = waits[0], waits[-1] - waits[0]
        sh = shape_of(sda, sdb)
        fm, sm = floor * N_LAYERS / 1e3, skew * N_LAYERS / 1e3
        tot_f += fm
        tot_s += sm
        if sh == "RELEASE":
            rel_s += sm
        nest = statistics.mean(c for _, c in B.values()) / base
        print(f"{('S%d->S%d ' % (a, b)) + LABEL.get((a, b), ''):<34}"
              f"{len(both):>5}{nest:>7.2f}{sda:>7.2f}{sdb:>7.2f}{sh:>9}"
              f"{floor:>8.2f}{skew:>8.2f}{fm:>9.3f}{sm:>8.3f}")
    print("=" * 106)
    print(f"{'TOTAL':<34}{'':>5}{'':>7}{'':>7}{'':>7}{'':>9}{'':>8}{'':>8}"
          f"{tot_f:>9.3f}{tot_s:>8.3f}")
    print(f"\n  'x/lyr' = this slot's writes per writer / slot 0's.  Any value "
          f"far from 1.00 means the\n  stamp fires more than once a layer and "
          f"t/c is a blend, not an arrival.")
    print(f"  FLOOR total {tot_f:.3f} ms is critical-path time no rebalance "
          f"can touch.")
    print(f"  SKEW at RELEASE points only: {rel_s:.3f} ms of idle -> "
          f"{rel_s * XFER:.3f} ms at the 0.32 coefficient,\n  and that "
          f"coefficient was measured for CROSS-RANK peer idle, not for this. "
          f"Upper bound only.")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/shdup1/ctr_base.log"
    for rank in (int(x) for x in (sys.argv[2:] or ["1", "4"])):
        report(path, rank)


if __name__ == "__main__":
    main()

# ============================================================================
# RESULT, 2026-08-22.  /tmp/shdup1/ctr_base.log, ranks 1 and 4 (agree to 2%).
# No GPU run.  A COMPLETE FLOOR/SKEW DECOMPOSITION OF THE LAYER.
# ============================================================================
#
#   FLOOR total  6.498 ms      SKEW total  13.441 ms      (layer budget 12.098)
#
# (0) THE CONTROL PASSED.  W13 tiles classifies SPREAD (sd 0.10 in, 4.00 out) --
# the exact inverse of a barrier.  Every tile phase does.  The shape column is
# reading the mechanism, not the label.
#
# (1) 54% OF THE LAYER IS FLOOR.  6.498 of 12.098 ms is time the LAST ARRIVER
# spends -- the critical worker, by construction.  No arrival rebalancing
# anywhere touches any of it.  The ranked floor list is the honest target list:
#
#     S0->S1   EP collective              24.92 us/lyr   1.894 ms   UNIFORM
#     S28->S29 W_UK/W_UV + Mech-C bar      9.84          0.747      UNIFORM
#     S18->S19 q_b tiles + KV append       8.20          0.624      SPREAD
#     S29->S30 o_proj GEMV + residual      6.92          0.526      UNIFORM
#     S6->S7   W13 -> W2 barrier           5.67          0.431      RELEASE
#     S17->S18 qkv_a -> q_b barrier        4.87          0.370      RELEASE
#     S32->S5  routing-ready poll          3.80          0.289      RELEASE
#     S25->S26 per-XCD attention release   2.76          0.210      RELEASE
#     ...16 more, none above 0.17 ms       -----         5.091 of 6.498
#
# The EP collective alone is 29% of the whole floor and it is UNIFORM at BOTH
# ends (sd 0.16 in, 0.09 out): every worker on every rank sits in it 24.92
# us/layer.  It is the largest single item in the layer by this estimator, and
# it is already closed as the rank-alignment TAX
# (glm-inter-rank-skew-is-paid-once).
#
# (2) SKEW IS NOT WALL TIME, AND THE TOTAL PROVES IT.  13.441 ms of skew
# against a 12.098 ms layer.  A quantity larger than the thing it is inside
# cannot be additive wall time; it is idle worker-time, most of it concurrent.
#
# (3) ONLY 4 OF 24 REGIONS ARE TRUE RELEASE POINTS -- S17->S18, S25->S26,
# S32->S5, S6->S7.  Their last-arriver floors are 4.87 / 2.76 / 3.80 / 5.67,
# mean 4.28 us.  glm-one-rendezvous-costs-3.77us measured the same quantity a
# completely different way and got 3.77.  Two estimators, one number.
#
# (4) EVERY RELEASE'S SKEW IS THE PRECEDING REGION'S SPREAD, ONE FOR ONE:
#     S16->S17 SPREAD 11.54  ->  S17->S18 RELEASE skew 12.07
#     S5->S6   SPREAD 10.99  ->  S6->S7   RELEASE skew 11.18
#     S31->S32 SPREAD 14.07  ->  S32->S5  RELEASE skew 24.36 (+104 idle workers)
#     S21->S23 mixed  24.66  ->  S25->S26 RELEASE skew 26.32
# This is mechanical, not a coincidence: a barrier's skew IS the arrival spread
# handed to it.  So "attack this barrier" is always really "attack the tile
# imbalance in front of it", which is 3 for 3 negative
# (glm-counted-region-time-before-a-barrier-is-not-a-lever,
# glm-barrier-narrowing-is-measured-out).  The board's own advice, re-derived
# from a different direction.
#
# (5) THE BOARD HIDES SKEW BY CONSTRUCTION, AND THAT IS CORRECT.  S25->S26 is
# priced 0.193 ms by max(b)-max(a) but carries 26.32 us/layer -- 2.000 ms -- of
# idle.  S32->S5 is priced 0.317 against 1.851 ms of idle.  max(b)-max(a) is a
# CRITICAL-PATH estimator and telescopes to the layer span, which is exactly
# right for a budget.  But "small on the board" does not mean "small hole", and
# a hole is where a lever would have to put work.  Read both columns.
#
# (6) NEGATIVE FOR THE RECORD: the 5.618 ms of skew at the four release points
# would be 1.798 ms at the 0.32 peer-idle-to-wall coefficient, which is the
# most optimistic number this sweep can produce.  Do NOT quote it.  That
# coefficient was measured for CROSS-RANK peer idle
# (glm-peer-idle-to-wall-response-is-convex); intra-GPU idle at a barrier
# whose skew is the preceding phase's tile spread has been measured at
# ~0 three separate times.  The sweep prints it only so nobody has to
# recompute it to reach the same conclusion.
#
# (7) LABEL BUG FOUND AND FIXED IN A SIBLING.  paired_span.py's LABEL map was
# shifted by one across S2..S19 -- it called S16->S17 "q_b / W_UK" when the
# `// Stage stamp 17: qkv_a tiles done.` comment says it is qkv_a tiles.
# board_budget.py was right.  Verified every entry against the call-site
# comments before publishing this table; paired_span.py is corrected.  Its
# RESULT block quoted only S0->S1, S28->S29 and S31->S32, all outside the
# shifted range, so its findings stand.

# ============================================================================
# CORRECTION, same day.  THE FLOOR METHOD CANNOT SEE A CROSS-RANK WAIT, and
# for the largest line in the table that is 82% of it.
# ============================================================================
#
# min_w is taken over the 232 workers of ONE rank.  A wait that every worker on
# the rank pays equally is therefore indistinguishable from uniform local work
# -- it lands in FLOOR and reads UNIFORM.  That is exactly what S0->S1 is.
# Stamp 3 splits it and has the SAME write count as stamps 0 and 1 (38836), so
# the per-worker difference is legal:
#
#   rank              0     1     2     3     4     5     6     7    mean
#   S3-S0 local fold+publish                                          4.47 us
#   S1-S3 peer wait + release fan-out                                19.81 us
#   S0->S1, the 8 EP folders                                         24.30 us
#   S0->S1, all 232 workers                                          24.33 us  <- 0.03 apart
#
#   local fold + publish        4.47 us/lyr   0.340 ms   18.4%
#   PEER WAIT + release        19.81 us/lyr   1.506 ms   81.6%
#
# So the 1.894 ms headline is 0.34 ms of local work and 1.51 ms of the
# rank-alignment TAX, which glm-inter-rank-skew-is-paid-once already closed --
# it does not disappear if you delete the collective, it relocates 83% of
# itself to the q_b gather.  THE HONEST LOCAL FLOOR IS 6.498 - 1.506 = 4.99 ms,
# and subtracting the layer's other cross-rank rendezvous (the QB_TP peer wait,
# ceiling 0.312, glm-qb-peer-wait-ceiling-is-0.335ms) puts it near 4.68 ms.
# Quote 6.498 only as "floor by the intra-rank estimator".
#
# RANK 0 IS THE MINIMUM PEER WAIT AGAIN -- 11.52 vs the peers' ~20.7 -- so rank
# 0 is the LAST ARRIVER at the EP collective.  Third independent sighting of
# the pattern, and it is the same rank the shared-expert work localizes to
# (glm-ep-skew-is-half-shared-expert-bias): more work in layer L-1 means later
# arrival in layer L.  The two findings agree.
#
# AND A NEW INSTANCE OF AN OLD TRAP.  The first attempt used stamp 4 (the
# published split is S3->S4 peer wait, S4->S1 fan-out) and got a NEGATIVE
# 3.21 us for S1-S4.  Stamp 4 is CONDITIONAL: its write count per worker ranges
# 32 to 12972 against 38836 for stamps 0/1/3, so t/c is a mean over a biased
# subset of layers and differencing it against a full-count stamp is illegal.
# The negative interval is the tell.  glm-subphase-slots-have-per-slot-
# populations says read the guard before dividing by it; this extends it to
# STAGE STAMPS, where the count is per-worker and printed right there in the
# BARSTAGEWS row.  Check it even when the neighbouring slots are full-count.
