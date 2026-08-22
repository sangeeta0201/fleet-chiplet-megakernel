#!/usr/bin/env python3
"""Split the router region S31->S32 into BARRIER POLL and REAL WORK.

Offline.  Reads MPK_SUBPHASE_TIMING logs already on disk.  No GPU run.

WHY THIS EXISTS.  paired_span.py found that S31->S32 ("router GEMV + TopK",
1.076 ms, board #4) is the largest unattributed block left, that only 128 of
232 workers enter it, and that the slow mode is TIGHT against a 0.40 us arrival
sd -- the signature of a common release rather than per-worker work.  It could
not separate the two, and the split decides the lever:

    poll-dominated -> widening the phase to 232 workers buys nothing.
    work-dominated -> 55% occupancy is the lever, ceiling ~0.44 ms.

The answer is NEITHER -- 54/46 -- and the ceiling was wrong anyway.  Read the
RESULT block at the bottom before using this; the region is now closed.

IT TURNS OUT NO NEW INSTRUMENT WAS NEEDED.  gang_rmsnorm_linear_bias_mi300.cuh
already brackets the o_proj spin as SP6[0] and splits what is left into
SP6[1..4], all guarded `tid == 0 && g_subphase_active`, exactly as the caller's
SP3[2] "Router" is.  Nobody had ever read bank 6 against the region.

DIVISORS -- read the guard, not 232 (glm-subphase-slots-have-per-slot-
populations).  SP6[0..3] are one add per KERNEL CALL from thread 0, and
g_subphase_cnt[6] counts exactly those calls.  SP6[4] is the TopK tail, which
fires once per layer per rank, and its own count is parked in SP6[5].  The
ratio cnt[6]/ns[6][5] is therefore calls-per-tail-firing = the router worker
population, and it comes out 128.0 -- an independent confirmation of the
128/232 that paired_span.py derived from the source gates.

CONTAMINATION, and why the headline number survives it.  Each region closes
with a GPU-wide atomicAdd that lands INSIDE the next region's bracket, so
SP6[2] carries SP6[1]'s atomic and SP6[3] carries SP6[2]'s.  SP6[0] does not:
its bracket opens immediately before the spin loop and closes immediately
after, with nothing but the spin inside, and the next bracket (_rt_t0) opens
after SP6[0]'s own atomic has retired.  So the POLL number is clean and the
work breakdown is an upper bound on each of its parts.  That is the right
direction for this question -- the conclusion is "no part of the work half
reaches the 0.26 ms floor", and an upper bound is what makes that safe.
"""
import collections
import re
import sys

PAT = re.compile(r"\[1,(\d+)\].*SP (\d+) (?:cnt (\d+)|(\d+) (\d+))")
N_LAYERS = 76

LOGS = [
    ("spsplit  08-21 12:17", "/tmp/glm5_spsplit.log"),
    ("sp_qkv   08-21 13:27", "/tmp/glm5_sp_qkv.log"),
    ("spu_1    08-21 13:55", "/tmp/glm5_spu_1.log"),
    ("spu_6    08-21 13:56", "/tmp/glm5_spu_6.log"),
    ("spb_1    08-21 14:07", "/tmp/glm5_spb_1.log"),
]

# SP6 slot -> label.  [5] is a COUNT, not a time.
SP6 = {0: "o_proj barrier POLL", 1: "Step1 RMSNorm",
       2: "Step2 gate dot + normed write", 3: "Step3 logit store + arrival",
       4: "Step4 TopK tail"}


def load(path):
    """-> {rank: {bank: {slot: ns}}}, {rank: {bank: cnt}}"""
    ns = collections.defaultdict(lambda: collections.defaultdict(dict))
    cnt = collections.defaultdict(dict)
    for ln in open(path, errors="ignore"):
        m = PAT.search(ln)
        if not m:
            continue
        r, bank = int(m.group(1)), int(m.group(2))
        if m.group(3) is not None:
            cnt[r][bank] = int(m.group(3))
        else:
            ns[r][bank][int(m.group(4))] = int(m.group(5))
    return ns, cnt


def report(tag, path):
    try:
        ns, cnt = load(path)
    except OSError as e:
        print(f"{tag}: {e}")
        return
    ranks = sorted(r for r in ns if 6 in ns[r] and 0 in ns[r][6])
    if not ranks:
        print(f"{tag}: no SP6 rows")
        return
    print(f"\n{tag}   {path}")
    print("-" * 92)
    print(f"{'rank':<6}{'calls':>10}{'pop':>6}", end="")
    for s in range(5):
        print(f"{('SP6[%d]' % s):>11}", end="")
    print(f"{'sum':>10}{'poll%':>8}")
    for r in ranks:
        b6, c = ns[r][6], cnt[r].get(6, 0)
        tail = b6.get(5, 0)
        if not c or not tail:
            continue
        per = []
        for s in range(5):
            d = tail if s == 4 else c
            per.append(b6.get(s, 0) / d / 1000.0)  # us per event
        # amortize the tail over the calls so the column sums to a per-call span
        amort = list(per)
        amort[4] = b6.get(4, 0) / c / 1000.0
        tot = sum(amort)
        print(f"{r:<6}{c:>10}{c/tail:>6.1f}", end="")
        for v in amort:
            print(f"{v:>11.3f}", end="")
        print(f"{tot:>10.3f}{100*amort[0]/tot:>7.1f}%")
    # region cost from the caller's own charge, for cross-check
    for r in ranks[:1]:
        b3, c3 = ns[r].get(3, {}), cnt[r].get(3, 0)
        if c3 and 2 in b3:
            print(f"       cross-check SP3[2] 'Router' = "
                  f"{b3[2]/c3/1000.0:.3f} us per its own guard (cnt {c3})")


def main():
    for tag, path in (LOGS if len(sys.argv) < 2
                      else [(p, p) for p in sys.argv[1:]]):
        report(tag, path)
    print("\nSP6[k] are us PER KERNEL CALL (divisor cnt[6]); the TopK tail is "
          "shown amortized\nover the calls so the row sums to one worker's "
          "span through the region.\n'pop' = cnt[6] / cnt_tail = router "
          "workers per rank.")


if __name__ == "__main__":
    main()

# ============================================================================
# RESULT, 2026-08-22.  5 subphase logs x 8 ranks = 40 samples.  No GPU run.
# BOARD ITEM #4 (S31->S32, 1.076 ms) IS NOW ATTRIBUTED, AND IT IS DEAD.
# ============================================================================
#
#   quantity                     mean    sd    range          share
#   POLL  SP6[0]  o_proj spin   6.999  1.099  4.518- 9.069    54%
#   WORK  SP6[1..4]             5.877  0.075  5.723- 6.041    46%
#   (us per kernel call; 76 layers -> 0.532 ms poll + 0.447 ms work)
#
# NEITHER HALF DOMINATES, so the either/or the previous probe was set up to
# decide does not apply.  Both halves were then priced separately and BOTH ARE
# BELOW THE 0.26 ms FLOOR.
#
# (0) POPULATION CONFIRMED INDEPENDENTLY.  cnt[6] / ns[6][5] = 1219200 / 9525
# = 128.0 exactly, in all 40 samples.  paired_span.py derived 128 of 232 from
# reading the two nested source gates; this derives it from a counter ratio.
# Two methods, same number.
#
# (1) THE WORK HALF IS COMPUTE, and the sd proves it.  0.075 us on 5.877 --
# 1.3% -- across five different runs and eight ranks.  A wait does not do that;
# the poll in the same table has sd 1.099 (16%).  Breakdown, us/call:
#     Step1 RMSNorm                 2.23   0.169 ms over the run
#     Step2 gate dot + normed write 2.18   0.166
#     Step3 logit store + arrival   1.42   0.108
#     Step4 TopK tail (amortized)   0.056  0.004
# Every line is under the floor on its own and the whole half is 0.447 ms.
#
# (2) *** MY OWN 0.44 ms WIDENING CEILING IS RETRACTED. ***  paired_span.py
# said "work-dominated -> 55% occupancy is the lever, ceiling 13.0 x
# (1 - 128/232) x 76 = 0.44 ms".  The work half IS large enough to matter, and
# the ceiling is still WRONG, because the occupancy is not free to change:
# there are exactly NUM_EXPERTS / ROUTER_EXPERTS_PER_TILE = 256/2 = 128 router
# TILES, and each of the 128 workers already takes exactly one.  The 104 idle
# workers have nothing to take.  The only way to hand them work is EPT=1, which
# makes 256 tiles over 232 workers -- a second round for 24 of them -- and the
# source already records that as measured worse ("that second call re-paid the
# barrier spin, the redundant RMSNorm, two block-wide reductions and the
# arrival atomic to add one dot product").  An occupancy figure is only a lever
# when the work is divisible; here it is quantized at one tile.
#
# (3) THE POLL IS THE LAST-ARRIVER PATTERN AGAIN, and prices the same way.
# Rank 0 has the MINIMUM poll in 5 of 5 logs (mean 4.942 vs peers' 7.293).
# Cross-rank skew = 2.351 us/layer -> 2.351 x 76 x 0.32 = 0.057 ms of wall for
# a PERFECT rebalance (glm-ep-residual-is-the-last-arrivers-own-floor,
# glm-peer-idle-to-wall-response-is-convex).  What is left is rank 0's own
# 4.942 us floor = 0.376 ms, which is counted region time in front of a
# barrier: 3 for 3 negative
# (glm-counted-region-time-before-a-barrier-is-not-a-lever).
#
# (4) THE CONTAMINATION RUNS THE SAFE WAY.  Each region closes with a GPU-wide
# atomicAdd that lands inside the NEXT region's bracket, so SP6[2] carries
# SP6[1]'s atomic and SP6[3] carries SP6[2]'s.  SP6[0] does not: its bracket
# holds nothing but the spin loop, and _rt_t0 opens after SP6[0]'s own atomic
# retires.  So the poll is clean and the work half is an UPPER bound -- which
# only makes "no part of it reaches 0.26 ms" safer.
#
# (5) A NEGATIVE ABOUT INSTRUMENTS, WORTH MORE THAN THE FINDING.  The probe
# scoped for this question was a NEW stage stamp inside the router kernel
# (MPK_STAGE_SLOTS is 36; 33-35 free), which is a header edit, a two-step
# rebuild, a permanent_output_dir wipe and a GPU run.  The split had been
# compiled into the tree the whole time, with a comment saying exactly what it
# was for -- "SP3[2] minus this is the router's real compute" -- and had never
# been read against the region.  Grep the existing counter banks before adding
# one.
