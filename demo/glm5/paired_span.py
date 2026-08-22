#!/usr/bin/env python3
"""PAIRED per-worker region spans -- the honest estimator when two stamps have
different worker populations.

Offline.  Reads BARSTAGEWS logs already on disk.  No GPU run.

THE BUG THIS EXISTS TO FIND.  `board_budget.py` prices a region [a->b] as
stat(b) - stat(a), where each stat is taken over whichever workers wrote that
slot.  That is only a duration when the two slots have the SAME population.
Several stamps are deliberately SUBSET stamps -- e.g. stamp 31 sits inside the
`if` that only router-participating workers enter, while stamp 32 is written by
every worker.  Subtracting an all-worker statistic from a subset statistic
mixes two different paths through the layer and the difference is not a span at
all.  The board marks these `*` and says "can be understated"; it can equally
be OVERSTATED, and nobody had checked which.

THE FIX.  Restrict to workers that wrote BOTH slots and difference PER WORKER
first:

    span_w = sum_b[w]/cnt_b[w] - sum_a[w]/cnt_a[w]

Same worker, same clock, same path.  Then take order statistics over w.  This
is clock-legal (per-rank stamps only, never across ranks --
glm-ep-peer-poll-is-not-the-cost) and population-legal.

Prints, per region: |A|, |B|, |A n B|, and the paired median / max.  When
|A| != |B| the cross-population number the board uses is printed alongside so
the size of the error is visible.
"""
import collections
import re
import statistics
import sys

PAT = re.compile(r"\[1,(\d+)\].*BARSTAGEWS (\d+) (\d+) (\d+) (\d+)")
N_LAYERS = 76

# Verified against the mpk_stage_stamp() call sites, 2026-08-22.
ORDER = [0, 1, 2, 16, 17, 18, 19, 21, 23, 24, 25, 26, 27, 28,
         29, 30, 31, 32, 5, 6, 7, 8, 12, 13, 14]
# CORRECTED 2026-08-22 against the `// Stage stamp N:` comment at every call
# site.  The first version of this map was SHIFTED BY ONE across S2..S19 --
# it called S16->S17 "q_b / W_UK" when stamp 17 is "qkv_a tiles done".
# board_budget.py had it right all along.  The RESULT block below quotes only
# S0->S1, S28->S29 and S31->S32, all outside the shifted range, so its findings
# are unaffected.  See barrier_floor_sweep.py note (7).
LABEL = {
    (0, 1): "EP collective", (1, 2): "post-EP -> qkv_a",
    (2, 16): "-> attn call boundary", (16, 17): "qkv_a tiles",
    (17, 18): "qkv_a -> q_b barrier", (18, 19): "q_b tiles + KV append",
    (19, 21): "q_b->decode bar + MLA decode",
    (21, 23): "decode->merge bar + merge",
    (23, 24): "Phase 8 attn -> o_proj bar", (24, 25): "o_proj wt prefetch",
    (25, 26): "per-XCD attn release", (26, 27): "prefetch DMA retired",
    (27, 28): "entering the MoE half", (28, 29): "W_UK/W_UV + Mech-C bar",
    (29, 30): "o_proj GEMV + residual", (30, 31): "o_proj -> router bar",
    (31, 32): "router GEMV + TopK", (32, 5): "routing-ready poll",
    (5, 6): "W13 tiles", (6, 7): "W13 -> W2 barrier", (7, 8): "W2 tiles",
    (8, 12): "MoE exit", (12, 13): "", (13, 14): "layer boundary",
}


def load(path):
    d = collections.defaultdict(lambda: collections.defaultdict(dict))
    for ln in open(path, errors="ignore"):
        m = PAT.search(ln)
        if m:
            r, s, w, c, t = (int(x) for x in m.groups())
            if c:
                d[r][s][w] = (c, t)
    return d


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/shdup1/ctr_base.log"
    rank = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    R = load(path)
    if rank not in R:
        sys.exit(f"rank {rank} not in {path}")
    rows = R[rank]

    print(f"{path}   rank {rank}   {N_LAYERS} layers")
    print("=" * 108)
    print(f"{'region':<34}{'|A|':>5}{'|B|':>5}{'|AnB|':>6}"
          f"{'paired med':>12}{'paired max':>12}{'board xpop':>12}{'err':>10}")
    print(f"{'':<34}{'':>5}{'':>5}{'':>6}"
          f"{'us/layer':>12}{'us/layer':>12}{'us/layer':>12}{'ms':>10}")
    print("=" * 108)
    total_err = 0.0
    flagged = []
    for a, b in zip(ORDER, ORDER[1:]):
        if a not in rows or b not in rows:
            continue
        A, B = rows[a], rows[b]
        both = sorted(set(A) & set(B))
        if not both:
            print(f"{('S%d->S%d ' % (a, b)) + LABEL.get((a, b), ''):<34}"
                  f"{len(A):>5}{len(B):>5}{0:>6}   DISJOINT -- no paired span")
            continue
        spans = [(B[w][1] / B[w][0] - A[w][1] / A[w][0]) / 1000.0 for w in both]
        med, mx = statistics.median(spans), max(spans)
        # the cross-population statistic the board uses: max over each slot
        xa = max(t / c for c, t in A.values()) / 1000.0
        xb = max(t / c for c, t in B.values()) / 1000.0
        xpop = xb - xa
        err = (xpop - mx) * N_LAYERS / 1e3
        mark = " *" if len(A) != len(B) else "  "
        if len(A) != len(B):
            total_err += err
            flagged.append((a, b, LABEL.get((a, b), ""), mx, xpop, err))
        print(f"{('S%d->S%d ' % (a, b)) + LABEL.get((a, b), ''):<34}"
              f"{len(A):>5}{len(B):>5}{len(both):>6}"
              f"{med:>12.2f}{mx:>12.2f}{xpop:>12.2f}{err:>+9.3f}{mark}")

    print("=" * 108)
    print("* = slot populations differ; the board's cross-population number is "
          "not a span.\n  'err' is board-minus-paired, in ms over the run: "
          "positive = the board OVERSTATES.")
    if flagged:
        print(f"\n  total signed error on the {len(flagged)} starred regions: "
              f"{total_err:+.3f} ms")
        for a, b, lab, mx, xpop, err in sorted(flagged, key=lambda r: -abs(r[5])):
            print(f"    S{a}->S{b:<3} {lab:<28} paired {mx*N_LAYERS/1e3:6.3f} ms"
                  f"   board {xpop*N_LAYERS/1e3:6.3f} ms   {err:+.3f}")


if __name__ == "__main__":
    main()

# ============================================================================
# RESULT, 2026-08-22.  Base arm /tmp/shdup1/ctr_base.log, ranks 0/1/4.
# No GPU run.  ONE NEGATIVE, ONE POSITIVE, ONE INSTRUMENT NOTE.
# ============================================================================
#
# (1) NEGATIVE -- THE POPULATION-MISMATCH HYPOTHESIS IS REFUTED.  The worry
# that drove this script was that board_budget.py's `*` regions might be badly
# wrong because they difference statistics over different worker sets.  They
# are not.  Paired-vs-board, ms over the run:
#
#     S31->S32 router GEMV + TopK    paired 1.126   board 1.102   -0.024
#     S28->S29 W_UK/W_UV + Mech-C    paired 0.798   board 0.769   -0.030
#     S0->S1   EP collective         paired 2.050   board 1.899   -0.151
#
# All three signs are NEGATIVE: the board UNDERSTATES, never overstates, which
# is exactly what its own footnote claims ("never a credit, only a floor").
# Two of the three are inside 0.03 ms.  The board's estimator also telescopes
# -- sum of max(b)-max(a) over the regions IS the layer span -- which the
# paired max does not, so max(b)-max(a) stays the right statistic for a budget
# and paired is the right one for "how long does a worker spend in here".
# Do not re-open the starred regions as a measurement-error story.
#
# (2) POSITIVE -- S31->S32 IS A NARROW PHASE, AND SHARPLY BIMODAL.  Per-worker
# span over the 192 that write both stamps, us/layer:
#
#     rank   n    min    p25    med    p75    max     sd
#     r0    189   0.74   0.80  12.98  13.83  16.14   6.12
#     r1    192   0.75   0.81  12.95  13.53  14.81   5.93
#     r4    192   0.73   0.80  12.74  13.07  14.49   5.75
#
# Two modes 16x apart, and the geometry at the source says exactly why:
#   * `if (xcd_rank < oproj_topk_tiles_per_xcd)` admits 24 of 29 workers per
#     XCD to the whole o_proj/router block -> 192 of 232.  The other 40 never
#     write stamp 31 at all.
#   * inside it, `for (t = xcd_rank; t < router_tile_n; t += tiles_per_xcd)`
#     with router_tile_n = 16 admits 16 per XCD -> 128.  The remaining 8 per
#     XCD (64 total) fall through the loop in 0.80 us.
#
# So the router phase runs at 128/232 = 55% occupancy and 104 workers idle
# through ~13 us/layer.  The board prices this region at 1.076 ms and its
# verdict covers only the TopK TAIL (k-loop 72%, rank-select +33%, fold 96 ns,
# bias prefetch dead) -- all of which are nanosecond-scale items
# (glm-router-topk-tail-breakdown).  The 1.076 ms itself has never been
# attributed.  It is the largest such block left on the board.
#
# (3) WHAT THE 13 us LOOKS LIKE, AND THE OPEN QUESTION.  The slow mode is
# TIGHT: p25 12.98 -> max 14.81 is ~1.8 us of spread, against an S31 arrival sd
# of 0.40 us.  Workers that enter together leave together, which is the
# signature of a COMMON RELEASE, not of per-worker work.  And the code says
# what the release is: the o_proj hier_barrier wait is deliberately NOT taken
# before stamp 31 -- "the *wait* deliberately does not happen here ... Both are
# handed to the router kernel, which issues its gamma and gate-weight loads
# before polling so they are in flight while the barrier spins"
# (gpt-oss-prefetch-across-barrier-idiom).  So S31->S32 brackets a barrier POLL
# plus the GEMV, and the board's label "router GEMV + TopK" mislocates it: the
# separately-listed S30->S31 "o_proj -> router barrier" (0.204 ms) is only the
# ARRIVAL, not the wait.
#
# Sizing the work half from the other direction: at ROUTER_EXPERTS_PER_TILE=2
# each participating worker does 2 dot products of length 5120 -- ~20 KB, well
# under 1 us even at a narrow phase's 41.6 GB/s/CU
# (glm-per-cu-roofline-denominator-is-wrong).  So the GEMV cannot be more than
# a small fraction of 13 us and the poll is most of it.  NOT YET MEASURED --
# these counters cannot split poll from work inside one kernel.  That split is
# the next probe, and it decides the lever:
#   poll-dominated -> widening the phase to 232 workers buys nothing (everyone
#                     would just poll), and the target is whatever the pollers
#                     are waiting FOR.
#   work-dominated -> 55% occupancy is the lever and the ceiling is
#                     13.0 x (1 - 128/232) x 76 = 0.44 ms.
#
# ---- ANSWERED SAME DAY, and BOTH BRANCHES ABOVE ARE WRONG.  See
# router_poll_vs_work.py: the split is 54/46, so NEITHER dominates, and the
# 0.44 ms ceiling in the second branch is RETRACTED -- there are exactly
# NUM_EXPERTS/EPT = 128 router TILES and each of the 128 workers already takes
# one, so the 104 idle workers have nothing to take and the occupancy is not
# free to change.  An occupancy figure is only a lever when the work is
# divisible.  Region CLOSED, no component above 0.17 ms.  Also: the probe this
# block scoped -- a new stage stamp, header edit, two-step rebuild, GPU run --
# was unnecessary; SP6[0..4] had been compiled in the whole time.  Grep the
# existing counter banks before adding one.
#
# (4) INSTRUMENT NOTE -- rank 0 reads sd 7.13 us at S30 against peers' 0.20.
# That is ONE outlier row, not an anomaly: r0 has n=191 where peers have 192,
# and its minimum is worker 208 at mean arrival 0.0 us -- a row that was never
# really written.  Drop it before computing any spread on rank 0.  Do not
# chase the "rank 0 o_proj barrier is skewed" story it invents.
