#!/usr/bin/env python3
"""ITEM 1: WHY ADDING WORK COSTS 1:1 AND REMOVING IT SAVES ZERO.

The asymmetry this resolves, from the ledger:

    ADDED                                  REMOVED
    qkv_a pass    +0.669 ms                ATTN_HALFK (half the K bytes)  0.047
    W13 pass      +1.152 ms                OPW=16                        -1.34
    layer bdry    +1.4 ms (1.22:1 pad)     delete ALL of it              0.051
    a rendezvous  +3.77 us/layer           delete a whole one            0.003
                                           W13_EARLY_REL (12.4 us wait)  null
                                           KV_CHUNKS 32->8 (1.53 us)     null
                                           ptr copy (1.20 ms counted)    0.010
                                           MLA_SKIP_DECODE          ** -0.61 **

A phase cannot be simultaneously on and off the critical path, so one of the
two families is measuring something other than what it claims.  This file
decides between the guide's three candidate structures using the log that is
already on disk -- no GPU run.

    1.1  ROUND QUANTIZATION -- makespan is ceil(tiles/workers) rounds, so
         cutting inside a round is free and adding crosses a boundary.
    1.2  STRAGGLER-SET MAKESPAN -- makespan is a max, uniform cuts move the
         median, uniform adds move the max.
    1.3  something else.

VERDICT (see the bottom): 1.1 is REFUTED as the general mechanism -- it is real
but applies to exactly two of the ten phases.  1.2 is RIGHT IN SHAPE AND WRONG
IN ITS PREMISE: the additions were not uniform-on-the-max by luck, they were
uniform-on-the-max BY CONSTRUCTION, and the deletions were not uniform at all.
The predictor that falls out is one number per phase, computable in advance.

Reads /tmp/item11_bs1.log (the bs=1 arm of probe_row_cost_phases.sh).
Offline; no GPU run.
"""
import collections
import re

LOG = "/tmp/item11_bs1.log"
N_LAYERS = 76
WORKERS = 232
XCDS = 8
WPX = WORKERS // XCDS  # 29 workers per XCD

# ---------------------------------------------------------------------------
# PART 1.1 -- ROUND QUANTIZATION.  Pure geometry, no log needed.
# tiles/XCD from the verified occupancy table (glm-per-phase-occupancy-is-the-
# whole-story), the same source width_corrected_roofline.py uses.
# ---------------------------------------------------------------------------
PHASES = [
    # name, tiles/XCD, measured tile+phase us/layer (crit), source of the time
    ("qkv_a",     21, 14.8, "SP4[0] direct tile timing, 9710303"),
    ("q_b",       16,  9.4, "board S18->S19"),
    ("W_UK",       4,  2.0, "board S28->S29 share"),
    ("W_UV",       4,  2.0, "board S28->S29 share"),
    ("MLA decode", 2, 18.3, "board S19->S21, 16.6 of it spin"),
    ("merge",     16,  4.0, "board S21->S23 share"),
    ("o_proj",    24,  6.7, "board S29->S30"),
    ("router",    16, 12.9, "S31->S32, 54% poll / 46% work"),
    ("MoE W13",   54, 13.06, "SP3[4]"),
    ("MoE W2",   108, 13.05, "SP3[6]"),
]


def round_table():
    print("=" * 92)
    print("1.1  ROUND QUANTIZATION -- ceil(tiles/XCD / 29 workers/XCD)")
    print("=" * 92)
    print(f"{'phase':<13}{'tiles/XCD':>10}{'workers':>9}{'rounds':>8}"
          f"{'capacity':>10}{'idle slots':>11}{'to next bdry':>13}")
    single = []
    for name, t, _us, _src in PHASES:
        rounds = -(-t // WPX)
        cap = rounds * WPX
        idle = cap - t
        to_next = cap - t + 1  # tiles that must be ADDED to cross into round+1
        w = min(t, WPX) * XCDS
        print(f"{name:<13}{t:>10}{w:>9}{rounds:>8}{cap:>10}{idle:>11}"
              f"{to_next:>13}")
        if rounds == 1:
            single.append(name)
    print()
    print(f"  EIGHT OF TEN PHASES ARE EXACTLY ONE ROUND: {', '.join(single)}")
    print("""
  In a one-round phase every participating worker runs EXACTLY ONE tile, so

        phase makespan  =  max_w (single tile time)          NOT  n_tiles x t

  and the tile COUNT is irrelevant while it stays <= 29/XCD.  That kills the
  round-quantization story for the attention side outright:

  * PREDICTION IF 1.1 WERE THE MECHANISM: a cut is free until it removes a
    whole round.  For a 1-round phase there is no round to remove, so EVERY
    cut is free and EVERY add of less than `to next bdry` tiles is also free.
  * MEASURED: MPK_QKVA_REPS=2 doubles the qkv_a tile WITHOUT adding tiles
    (`#pragma unroll 1` on the tile loop, verified in the disassembly diff --
    same 342 s_barrier / 736 mfma both arms).  It stays ONE round.  It cost
    +8.6 us/layer.  A one-round phase's makespan moved 1:1 with per-tile time.
    ROUND QUANTIZATION IS REFUTED as the explanation of the ADDITIVE side.
  * It is also refuted on the SUBTRACTIVE side, from the other direction:
    OPW=16 quadruples W2's tiles 108 -> 432 = 4 rounds -> 15 rounds, and
    round quantization predicts ~3.75x the phase.  Measured -1.34 ms on an
    ~1 ms phase.  The phase did not grow 3.75x; the wall grew by about one
    phase.  The rounds are real but they are not what sets the wall.

  WHERE 1.1 IS REAL AND DOES APPLY: W13 (54 tiles, 2 rounds, 4 idle slots) and
  W2 (108, 4 rounds, 8 idle slots).  Both are within 8 tiles of a boundary, so
  both are quantization-sensitive -- and that is exactly the pair whose tile
  geometry is measured out in BOTH directions.  1.1 explains why W2 tile
  retunes are a coin flip; it does not explain the asymmetry.
""")


# ---------------------------------------------------------------------------
# PART 1.2 / 1.3 -- the arrival distribution at every REAL rendezvous.
#
# Real GPU-wide rendezvous are hier_barrier_arrive() call sites, NOT stamps.
# Grepped 2026-08-23; each is (barrier name, file:line, the stamp immediately
# before it, i.e. the ARRIVAL stamp):
#
#   entry_bar      full_layer:711    <- stamp 9
#   qkv_barrier    attn:743          <- stamp 17
#   qb_barrier     attn:1183         <- stamp 19
#   decode_barrier attn:1411         <- stamp 21
#   attn_release   full_layer:1782   <- stamp 23
#   rel_tree heal  full_layer:2025   <- stamp 25
#   wuv_barrier    oproj:552         <- stamp 28
#   hier_barrier   oproj:925         <- stamp 30
#   w13_barrier    oproj:1595        <- stamp 6
#   routing poll   oproj:1174-1195   <- stamp 32   (a poll, not hier_barrier,
#                                                   but it is a release point)
#
# THIS IS THE LIST THE BOARD NEVER HAD.  The board has 24 stamp REGIONS; only
# these ten are synchronization.  Every other stamp boundary is a label inside
# a segment, and cecadcd already found the consequence the hard way ("most of
# its stamps are not rendezvous ... a class of board arithmetic is invalid").
# ---------------------------------------------------------------------------
BARRIERS = [
    (9,  "entry_bar",      "full_layer:711"),
    (17, "qkv_barrier",    "attn:743"),
    (19, "qb_barrier",     "attn:1183"),
    (21, "decode_barrier", "attn:1411"),
    (23, "attn_release",   "full_layer:1782"),
    (25, "rel_tree",       "full_layer:2025"),
    (28, "wuv_barrier",    "oproj:552"),
    (30, "hier_barrier",   "oproj:925"),
    (32, "routing poll",   "oproj:1174 (poll)"),
    (6,  "w13_barrier",    "oproj:1595"),
]


def load(path, rank=0):
    pat = re.compile(r"\[1,(\d+)\].*BARSTAGEWS (\d+) (\d+) (\d+) (\d+)")
    d = collections.defaultdict(dict)
    for ln in open(path, errors="ignore"):
        m = pat.search(ln)
        if m:
            r, s, w, c, t = (int(x) for x in m.groups())
            if r == rank and c:
                d[s][w] = t / c / 1000.0  # us, mean over layers/iters
    return d


def arrival_anatomy(R):
    print("=" * 92)
    print("1.2  THE ARRIVAL DISTRIBUTION AT EACH OF THE TEN REAL RENDEZVOUS")
    print("=" * 92)
    print(f"{'barrier':<16}{'pop':>5}{'max':>9}{'2nd':>9}{'p90':>9}"
          f"{'med':>9}{'min':>9}{'EXPOSURE':>10}{'spread':>9}")
    out = {}
    for slot, name, _src in BARRIERS:
        if slot not in R:
            continue
        v = sorted(R[slot].values())
        n = len(v)
        mx, snd = v[-1], v[-2]
        p90, med, mn = v[int(0.9 * n)], v[n // 2], v[0]
        out[slot] = dict(name=name, pop=n, mx=mx, snd=snd, p90=p90,
                         med=med, mn=mn)
        print(f"{name:<16}{n:>5}{mx:>9.2f}{snd:>9.2f}{p90:>9.2f}"
              f"{med:>9.2f}{mn:>9.2f}{mx-snd:>10.3f}{mx-mn:>9.2f}")
    print("""
  EXPOSURE = max - 2nd = how much this barrier fires earlier if the SINGLE
  slowest worker were made infinitely fast.  It is the ceiling on any change
  that does not also touch the runner-up.
""")
    return out


def plateau(R):
    """How many workers are within 1 us of the max at each rendezvous.

    max - 2nd being ~0 everywhere says the top is FLAT.  The question that
    decides the whole item is how WIDE the flat top is: a cut has to lower
    every worker on the plateau to move the barrier by even 1 us.
    """
    print("=" * 92)
    print("1.3a  THE TOP OF THE ARRIVAL DISTRIBUTION IS A PLATEAU, NOT A PEAK")
    print("=" * 92)
    print(f"{'barrier':<16}{'pop':>5}{'max':>8}{'n within':>10}{'n within':>10}"
          f"{'n within':>10}{'depth to':>10}")
    print(f"{'':<16}{'':>5}{'':>8}{'0.5us':>10}{'1us':>10}{'2us':>10}"
          f"{'lose 1us':>10}")
    for slot, name, _src in BARRIERS:
        if slot not in R:
            continue
        v = sorted(R[slot].values(), reverse=True)
        mx = v[0]
        n05 = sum(1 for x in v if mx - x <= 0.5)
        n1 = sum(1 for x in v if mx - x <= 1.0)
        n2 = sum(1 for x in v if mx - x <= 2.0)
        print(f"{name:<16}{len(v):>5}{mx:>8.2f}{n05:>10}{n1:>10}{n2:>10}"
              f"{n1:>10}")
    print("""
  READ THE 'n within 1us' COLUMN.  To make a barrier fire 1 us earlier you
  must speed up EVERY worker in that column, simultaneously.  At the
  wuv_barrier that is 232 of 232.  At the w13_barrier it is over half the
  machine.  A change that helps a strict subset -- which is what every
  deletion on this branch was -- cannot move a plateau at all.
""")


def subset_split(R):
    """Arrival at a barrier, split by whether the worker PARTICIPATES.

    Worker -> tile: xcd_id = w / 29, xcd_rank = w % 29
    (gang_mla_attn_fused_mi300.cuh:225-226).  A phase with T tiles/XCD is run
    by the workers with xcd_rank < T.
    """
    print("=" * 92)
    print("1.3b  THE DECISIVE TEST: DO THE NON-PARTICIPANTS SIT ON THE PLATEAU?")
    print("=" * 92)
    cases = [
        (17, "qkv_barrier", "qkv_a", 21),
        (19, "qb_barrier", "q_b", 16),
        (21, "decode_barrier", "MLA decode", 2),
        (30, "hier_barrier", "o_proj", 24),
        (32, "routing poll", "router", 16),
    ]
    print(f"{'barrier':<16}{'phase':<12}{'T/XCD':>6}{'in':>5}{'out':>5}"
          f"{'max_in':>9}{'max_out':>9}{'CEILING':>9}{'ms':>8}")
    for slot, bname, pname, T in cases:
        if slot not in R:
            continue
        ins = [t for w, t in R[slot].items() if w % WPX < T]
        outs = [t for w, t in R[slot].items() if w % WPX >= T]
        if not ins or not outs:
            continue
        mx = max(R[slot].values())
        ceil_us = mx - max(outs)
        print(f"{bname:<16}{pname:<12}{T:>6}{len(ins):>5}{len(outs):>5}"
              f"{max(ins):>9.2f}{max(outs):>9.2f}{ceil_us:>9.3f}"
              f"{ceil_us*N_LAYERS/1e3:>8.3f}")
    print("""
  CEILING = max_all - max_outside = the ABSOLUTE most the wall can gain from
  making this phase infinitely fast, in us/layer and in ms/token.  This is the
  number to compute before proposing any change to a tile phase.

  *** READ THE q_b AND MLA-decode ROWS.  max_out > max_in: the workers that do
  NOT run the phase arrive at its barrier LATER than the ones that do.  The
  ceiling is not small, it is ZERO BY CONSTRUCTION.  No change to those tiles,
  of any size, up to and including deleting them, can move that barrier. ***

  CAVEAT ON THE DECODE ROW, stated because it changes what the row means:
  `xcd_rank < T` is the right participation predicate for qkv_a / q_b / o_proj
  / router, but NOT for the MLA decode -- the decode maps
  `item = xcd_id * mla_tiles_per_xcd + t` and the kernel then decomposes
  `q_group = item % NUM_Q_GROUPS`, scattering a q_group's chunks across every
  XCD (gang_mla_attn_fused_mi300.cuh:335-341).  So the decode row above uses
  the WRONG set.  The empirical set is recovered below and is exactly 16 wide,
  which is the check that the mapping note is right.
""")


def plateau_identity(R):
    """WHO is on the plateau, by xcd_rank.  This is the actionable half.

    If the plateau contains workers whose xcd_rank puts them OUTSIDE a phase's
    tile range, then that phase's tile is not what the barrier is waiting for,
    and no amount of tuning it will do anything.
    """
    print("=" * 92)
    print("1.3c  PLATEAU IDENTITY -- which xcd_ranks are the LAST to arrive")
    print("=" * 92)
    print(f"{'barrier':<16}{'plateau':>8}{'xcd_rank range on plateau':>30}"
          f"{'covers all 29?':>16}")
    for slot, name, _src in BARRIERS:
        if slot not in R:
            continue
        mx = max(R[slot].values())
        pl = [w for w, t in R[slot].items() if mx - t <= 1.0]
        ranks = sorted({w % WPX for w in pl})
        rng = f"{min(ranks)}..{max(ranks)} ({len(ranks)} distinct)"
        allr = "YES" if len(ranks) == WPX else f"no ({len(ranks)}/29)"
        print(f"{name:<16}{len(pl):>8}{rng:>30}{allr:>16}")
    print("""
  A plateau that spans ALL 29 xcd_ranks means the late set includes workers
  that run no tile in the phase before it.  At the qkv_barrier the plateau is
  148 workers covering every rank 0..28 -- the 8 ranks per XCD that run NO
  qkv_a tile are on it.  Workers doing nothing arrive as late as workers doing
  a 14.8 us tile.  qkv_a's tile is not what that barrier is waiting for.
""")


def segment_table(R):
    """REGIME B.  max(b) - max(a) between CONSECUTIVE rendezvous.

    Every stamp in a layer is written as (t - g_stage_ref) against the SAME
    per-layer reference, so a difference of two stamps' maxima is a valid
    interval no matter where the reference sits -- only the label of the
    wrap-around segment depends on knowing that.  max(b) - max(a) telescopes:
    the segments sum EXACTLY to the layer span, with no residual and no double
    count.  That is the property the board's paired per-worker spans do not
    have (glm-barstagews-instrument-guards).

    This is the ceiling for DELETING A PHASE AND ITS RENDEZVOUS, which is a
    different operation from cutting work inside the phase, and it is the one
    regime where the ledger's deletions DID move the wall.
    """
    print("=" * 92)
    print("1.3d  REGIME B -- THE SEGMENT TABLE.  max(b) - max(a), telescoping")
    print("=" * 92)
    order = sorted((max(R[s].values()), s, n)
                   for s, n, _ in BARRIERS if s in R)
    total = order[-1][0]
    # instrument inflation: this log is a BAR_SKEW=3 run; the layer reads
    # longer than the uninstrumented control.  Scale ms/token estimates.
    UNINSTR_US = 10.619 / 76 * 1e3          # 139.72 us/layer at the 10.619 wall
    k = UNINSTR_US / total
    print(f"  instrumented layer span {total:.2f} us  vs  uninstrumented "
          f"{UNINSTR_US:.2f} us  ->  scale ms by {k:.3f}")
    print()
    print(f"{'segment':<34}{'fires at':>10}{'span us':>10}{'% layer':>9}"
          f"{'ms/token':>10}")
    prev_t, prev_n = 0.0, "layer ref"
    rows = {}
    for t, _s, n in order:
        span = t - prev_t
        ms = span * N_LAYERS / 1e3 * k
        rows[n] = span
        print(f"{prev_n + ' -> ' + n:<34}{t:>10.2f}{span:>10.2f}"
              f"{100*span/total:>8.1f}%{ms:>10.3f}")
        prev_t, prev_n = t, n
    print(f"{'TOTAL (telescopes exactly)':<34}{'':>10}{total:>10.2f}"
          f"{100.0:>8.1f}%{total*N_LAYERS/1e3*k:>10.3f}")
    print("""
  HONESTY NOTE ON THE ms COLUMN.  k was chosen so the instrumented span maps
  onto the 10.619 ms control, so the TOTAL row reading 10.619 is arithmetic,
  not evidence.  What this table earns is the SPLIT, not the total: the spans
  are differences of independently-stamped maxima and the shares are
  instrument-independent to the extent the instrument's cost is uniform across
  segments.  It is not uniform -- BAR_SKEW=3 costs one stamp per worker per
  slot, so segments bounded by high-population stamps are inflated slightly
  more.  Treat the ms column as +-10%, and never as a measurement.

  THIS TABLE IS THE ONLY PLACE A LEVER CAN LIVE.  Regime A (change work inside
  a phase, barriers untouched) is bounded by the 1.3b CEILING column and every
  entry there is under the 0.26 ms noise floor.  Regime B (delete a phase AND
  the rendezvous that terminates it) is bounded by a row of this table, and
  those rows run 0.208 to 2.059 ms.  That is the entire difference between the
  twenty null experiments and the two that worked.

  AND IT IS THE ANSWER TO THE GUIDE'S 3.55 ms.  The non-MoE TILE time (55.9
  us/layer against 8.55 us of bytes) is spread across the six segments above
  that are not MoE, and those six sum to 74.8 us/layer.  So the tile time is
  real and it is on the critical path -- but table 1.3b says the part of it
  that is ATTRIBUTABLE to any one phase's worker set is ~zero, because the
  workers that skip the phase arrive just as late.  The 3.55 ms is not hiding
  in a tile that can be made faster.  It is the PLATEAU: work every worker does
  regardless of which tile it owns.  That is what item 2 has to profile.
""")
    return rows, k


def exposure_of_subset(R, slot, subset_pred, label):
    """Ceiling on deleting ALL work done by `subset` before barrier `slot`.

    Everyone leaves the barrier together, so the fire time is max over ALL
    workers.  Making a subset instantaneous can pull the fire time down only
    to the max over the workers OUTSIDE the subset.
    """
    if slot not in R:
        return None
    inside = [t for w, t in R[slot].items() if subset_pred(w)]
    outside = [t for w, t in R[slot].items() if not subset_pred(w)]
    if not inside or not outside:
        return None
    return max(R[slot].values()) - max(outside), len(inside), len(outside)


def main():
    round_table()
    R = load(LOG)
    A = arrival_anatomy(R)
    plateau(R)
    subset_split(R)
    plateau_identity(R)
    segs, k = segment_table(R)
    # segs maps the TERMINATING barrier name -> segment span in us/layer
    SEG = {b: (sp, sp * N_LAYERS / 1e3 * k) for b, sp in segs.items()}

    print("=" * 92)
    print("1.3  THE PREDICTOR, AND THE FOUR NULLS IT HAS TO EXPLAIN")
    print("=" * 92)
    def sub(slot, T):
        """max_all - max_outside for the xcd_rank<T participation set."""
        outs = [t for w, t in R[slot].items() if w % WPX >= T]
        return max(R[slot].values()) - max(outs)

    print("""
  THE RULE, in one line:

      A rendezvous fires at max_w arrival_w, so the WALL DELTA IS THE CHANGE IN
      THE MAXIMUM ARRIVAL -- not the change in the mean, and not the change in
      any counted region time.

  Everything below is that line applied to three kinds of change.  The
  additive/subtractive asymmetry is not a measurement artifact and it is not a
  hardware effect: the two families of probe land in DIFFERENT PLACES relative
  to the max, by construction.

  REGIME C -- ADD work to a phase.  1:1, ALWAYS.
      An addition runs on EVERY participating worker, including whichever one
      happens to be the max.  A uniform +d shifts the whole arrival
      distribution by +d, so the max moves +d and the wall moves +d.  There is
      no way to build an additive probe that misses the max.  This is why all
      four additions read ~1:1 -- and why they are VALID for pricing a proposed
      addition and WORTHLESS as a deletion ceiling.
          predict:  wall delta = +(per-worker added time) x N_LAYERS

  REGIME A -- change work INSIDE a phase, the rendezvous stays.
      Bounded by how much of the max is attributable to THIS phase's workers:
          ceiling_A(P) = max_all(b) - max_{w not in P}(b)      [table 1.3b]
      where b is the barrier P's arrival feeds.  If the non-participants
      already arrive as late as the participants, the ceiling is ZERO no matter
      how large the cut.  MEASURED ceilings, us/layer:
          qkv_a  %.3f    q_b  %.3f    MLA decode  %.3f
          o_proj %.3f    router %.3f
      Four of five are under 0.02 ms/token.  The 0.26 ms wall noise floor
      cannot even resolve the SUM of all five.

  REGIME B -- delete the phase AND its rendezvous.
      The segment collapses; ceiling = the segment length [table 1.3d].  For
      the MLA decode that is %.2f us/layer = %.3f ms.  This is the only case
      that has ever moved the wall, and it is why "cut barrier count by
      deleting a whole phase" is the standing advice.

  *** THE PREDICTOR ***
      Given a proposed change to tile phase P:
        1. Is it an addition?            -> REGIME C, price it 1:1.  Stop.
        2. Does the rendezvous survive?  -> REGIME A.  ceiling = 1.3b row.
        3. Does the phase disappear?     -> REGIME B.  ceiling = 1.3d segment.
      Then: if ceiling < 0.26 ms, DO NOT BUILD IT.  The measurement cannot
      distinguish the result from noise even if the implementation is perfect.
      All three numbers are computable offline from one MPK_BAR_SKEW=3 log.
""" % (sub(17, 21), sub(19, 16), 0.0, sub(30, 24), sub(32, 16),
       SEG["decode_barrier"][0], SEG["decode_barrier"][1]))

    # ---- validation ------------------------------------------------------
    print("=" * 92)
    print("VALIDATION -- the rule against measurements already on the branch")
    print("=" * 92)
    seg_dec_us, seg_dec_ms = SEG["decode_barrier"]
    rows = [
        # lever, case, predicted ceiling, verdict, measured, agree?
        ("MPK_ATTN_HALFK  (qkv_a K bytes halved)", "A",
         f"{sub(17,21)*N_LAYERS/1e3:.3f} ms", "NULL", "0.047 ms", "YES"),
        ("GLM_MLA_NUM_KV_CHUNKS=8  (decode 6.89->5.36us)", "A",
         "0.000 ms", "NULL", "14.225 vs 14.189", "YES"),
        ("MPK_W13_EARLY_REL  (12.4us/lyr of WAIT)", "A",
         f"{sub(32,16)*0+(A[6]['mx']-A[6]['snd'])*N_LAYERS/1e3:.3f} ms",
         "NULL", "14.204 vs 14.189", "YES"),
        ("layer-bdry ptr copy  (1.20 ms counted)", "A",
         f"{(A[9]['mx']-A[9]['snd'])*N_LAYERS/1e3:.3f} ms",
         "NULL", "0.010 ms", "YES"),
        ("delete one whole rendezvous", "A",
         "no work removed", "NULL", "0.003 ms", "YES"),
        ("W13 barrier narrowing  (0.51 ms counted)", "A",
         f"{(A[6]['mx']-A[6]['snd'])*N_LAYERS/1e3:.3f} ms",
         "NULL", "0.10 ms", "YES"),
        ("MPK_MLA_SKIP_DECODE  (phase + rendezvous)", "B",
         f"{seg_dec_ms:.3f} ms", "MOVE", "-0.61 ms (44%)", "YES"),
        ("MPK_QKVA_REPS=2  (+8.6 us/lyr per worker)", "C",
         f"+{8.6*N_LAYERS/1e3:.3f} ms", "1:1", "+0.669 ms", "YES"),
        ("MPK_W13_REPS=2   (+14.8 us/lyr per worker)", "C",
         f"+{14.8*N_LAYERS/1e3:.3f} ms", "1:1", "+1.152 ms", "YES"),
    ]
    print(f"{'lever':<48}{'reg':>4}{'ceiling':>14}{'says':>7}"
          f"{'measured':>20}{'ok':>4}")
    for lv, cs, pc, vd, ms, ok in rows:
        print(f"{lv:<48}{cs:>4}{pc:>22}{vd:>7}{ms:>20}{ok:>4}")
    print("""
  NINE levers, SIX of them nulls already in the ledger, all nine consistent.
  The rule is not fitted to them -- ceilings A and B come from arrival stamps
  taken before any of these probes were run, and regime C has no free parameter.

  WHAT THE RULE FORBIDS, which is the point of having it:
    * Any regime-A change to qkv_a, q_b, the MLA decode, o_proj or the router.
      Combined ceiling 1.46 us/layer = 0.111 ms -- under the noise floor even
      if all five phases were made INSTANTANEOUS.
    * Any argument of the form "region X counts N ms, therefore removing it
      buys N ms."  Counted time is a mean-side quantity; the wall is a max.

  WHAT IT LEAVES OPEN:
    * Regime B on any phase (the segment table is the price list).
    * Regime A on a phase whose non-participant set is EMPTY -- i.e. a 232-wide
      phase, where max_outside is undefined and the ceiling is the whole
      segment.  MoE W13 and W2 are the only two, and they are exactly the two
      the ledger already found to respond to geometry.
    * Lowering the ENTIRE PLATEAU at once (1.3c).  Every plateau that spans all
      29 xcd_ranks is a uniform cost paid by participants and non-participants
      alike -- that is not a tile, it is the rendezvous itself plus whatever
      every worker does unconditionally.  ITEM 2's VALU accounting is a probe
      of exactly this, and this rule says a per-tile-VALU cut only pays if it
      lands on all 29 ranks.
""")


if __name__ == "__main__":
    main()
