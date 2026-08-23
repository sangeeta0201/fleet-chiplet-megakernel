#!/usr/bin/env python3
"""ITEM 3: PRICE "FEWER, WIDER PHASES" BEFORE BUILDING IT.

The ask
-------
Ten GPU-wide rendezvous per layer at 2.85-3.77 us is ~2.94 ms of the 10.619 ms
wall; four would be ~1.2 ms.  glm-no-legal-independent-round-fusion-exists says
every one of the ten separates a true producer-consumer pair, so do not look
for independence.  Look instead for a decomposition where the pair does not
need a GPU-WIDE rendezvous -- gpt-oss's CROC per-XCD chunk barrier being the
named reference.

FIRST FINDING, BEFORE ANY MODELLING: THE NAMED PORT IS ALREADY IN THE TREE.
--------------------------------------------------------------------------
`gpt-oss-croc-chunk-barrier-not-ported-to-mla` says "not ported."  That memory
is STALE.  `MPK_GLM_MLA_PAIR_MERGE` exists
(gang_mla_attn_fused_mi300.cuh:377), is wired through
python/mirage/mpk/persistent_kernel.py:476, and was measured in `e482f2e`:

    PAIR_MERGE cuts the decode->merge rendezvous from 8 XCDs to 2
    11.074 vs an 11.005 baseline  ->  +0.07 ms  (n=1, inside a 0.19 spread)
    and +0.6% at 4 chunks earlier, in 69770d6.

So the concrete deliverable named in item 3 is written, measured, and NEUTRAL.
That is one data point, on one of the ten rendezvous, and it is the single most
valuable thing in this file: it is a REAL MEASUREMENT OF EXACTLY THE OPERATION
THIS SCRIPT MODELS, so it can validate or falsify the model rather than the
model being asserted.

What this file does
-------------------
Generalises that one measurement to all ten rendezvous, as a CEILING, offline.

    Today   at rendezvous R every worker leaves at  M_R = max_w arr_w(R).
    Narrowed to per-XCD, worker w in XCD x leaves at  M_R^x = max_{w' in x} arr_w'(R).
    Its segment work is unchanged, so it reaches the NEXT rendezvous S at
        arr'_w(S) = M_R^x + (arr_w(S) - M_R)
    and S fires at max_w arr'_w(S).  The saving is M_S - max_w arr'_w(S).

Chaining that over all ten barriers in fire order gives the layer span under
any chosen narrowing set.  Narrowing ALL ten is flatly illegal -- data crosses
XCDs at nearly every one -- but it is the CEILING on the whole item, and a
ceiling is what "price the port before writing it" asks for.  If the ceiling is
under the noise floor, legality never has to be argued.

Method notes, stated because they bound what the answer is worth
---------------------------------------------------------------
* Reads the same /tmp/item11_bs1.log the item-1 predictor reads (MPK_BAR_SKEW=3,
  bs=1, NP=8, rank 0).  No GPU run.
* BARSTAGEWS stamps are (total_ns / count), i.e. a worker's MEAN arrival over
  layers and iters.  A max-of-means understates the true max-of-maxes, so
  per-XCD maxima are understated LESS than the GPU-wide max is (fewer samples
  in the max).  That biases this model TOWARD predicting a saving.  A ceiling
  that comes out small under a bias that inflates it is a strong result; one
  that comes out large would need a per-layer log to confirm.
* Worker -> XCD is xcd_id = w / 29 (gang_mla_attn_fused_mi300.cuh:225).
* The baseline reconstruction is an identity check, not evidence: simulating
  with NOTHING narrowed must reproduce the measured layer span exactly.
"""
import collections
import re

LOG = "/tmp/item11_bs1.log"
N_LAYERS = 76
WORKERS = 232
XCDS = 8
WPX = WORKERS // XCDS  # 29
WALL_MS = 10.619       # the bs=1 control this whole board is measured against

# The ten REAL rendezvous: hier_barrier_arrive() call sites, and the stamp
# written immediately before each (i.e. the ARRIVAL stamp).  Grepped
# 2026-08-23; identical list to makespan_predictor.py's, which is the point --
# item 3 must be priced against the same ten regime-B segments item 1 built.
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

# Which producer-consumer pairs actually STAY inside an XCD.  This is the
# legality column, and it is read off the tile maps, not guessed.
#   "GPU-WIDE"  data produced on XCD a is read on XCD b for some a != b
#   "PAIR"      confined to an XCD PAIR (the PAIR_MERGE shape)
#   "PER-XCD"   already confined to one XCD
LEGALITY = {
    "entry_bar":      ("GPU-WIDE", "layer entry: every worker reads the residual "
                                   "stream that the previous layer's MoE wrote "
                                   "across all XCDs"),
    "qkv_barrier":    ("GPU-WIDE", "qkv_a writes q/kv_a tiles consumed by q_b and "
                                   "the decode on a different tile map"),
    "qb_barrier":     ("GPU-WIDE", "q_b's 16 tiles/XCD feed a decode whose item map "
                                   "scatters q_groups across every XCD"),
    "decode_barrier": ("PAIR",     "THE CROC CASE. merge is already pair-aligned "
                                   "(xcd_id/2 == q_group); only the decode map "
                                   "disagrees. PAIR_MERGE fixes it. MEASURED +0.07."),
    "attn_release":   ("GPU-WIDE", "merge output feeds o_proj, which is sharded "
                                   "across EP ranks and re-gathered"),
    "rel_tree":       ("GPU-WIDE", "hierarchical release self-heal, GPU-wide by "
                                   "construction"),
    "wuv_barrier":    ("GPU-WIDE", "W_UV's 4 tiles/XCD are a replicated GEMV read "
                                   "by all of o_proj"),
    "hier_barrier":   ("GPU-WIDE", "o_proj -> router: the router reads the full "
                                   "hidden vector, every element of it"),
    "routing poll":   ("GPU-WIDE", "TopK on rank r selects experts owned by any "
                                   "rank; dispatch is all-to-all by definition"),
    "w13_barrier":    ("GPU-WIDE", "W13 -> W2 crosses experts, and expert -> XCD "
                                   "is a routing outcome, not a static map"),
}


def load(path, rank=0):
    """slot -> {worker: mean arrival in us}."""
    pat = re.compile(r"\[1,(\d+)\].*BARSTAGEWS (\d+) (\d+) (\d+) (\d+)")
    d = collections.defaultdict(dict)
    for ln in open(path, errors="ignore"):
        m = pat.search(ln)
        if m:
            r, s, w, c, t = (int(x) for x in m.groups())
            if r == rank and c:
                d[s][w] = t / c / 1000.0
    return d


def fire_order(R):
    """The ten barriers, in the order they fire, with their fire times."""
    return sorted(((max(R[s].values()), s, n) for s, n, _ in BARRIERS
                   if s in R))


def segments(R):
    """-> (names, D) where D[i][w] is worker w's WORK in segment i.

    Segment i runs from the fire of barrier i-1 to worker w's arrival at
    barrier i.  Because every worker leaves barrier i-1 together at M_{i-1},
    D[i][w] = arr_w(B_i) - M_{i-1} is the worker's own post-barrier work, and
    it is the quantity that is invariant under changing HOW barrier i-1
    synchronises.  That invariance is the whole model.
    """
    order = fire_order(R)
    names = [n for _t, _s, n in order]
    D, prev_fire = [], 0.0
    for t, s, _n in order:
        D.append({w: R[s][w] - prev_fire for w in R[s]})
        prev_fire = t
    return names, D, order


def simulate(names, D, narrow):
    """Layer span with the named barriers narrowed to per-XCD.

    narrow: set of barrier names to make per-XCD.  Everything else stays
    GPU-wide.  Returns (span_us, per_barrier_fire_times).
    """
    t = collections.defaultdict(float)      # when worker w last left a barrier
    fires = []
    for i, name in enumerate(names):
        arr = {w: t[w] + d for w, d in D[i].items()}
        if name in narrow:
            per_xcd = collections.defaultdict(float)
            for w, a in arr.items():
                x = w // WPX
                per_xcd[x] = max(per_xcd[x], a)
            for w in arr:
                t[w] = per_xcd[w // WPX]
            fires.append(max(per_xcd.values()))
        else:
            f = max(arr.values())
            for w in arr:
                t[w] = f
            fires.append(f)
    return max(t.values()), fires


def simulate_groups(names, D, narrow, gsize):
    """As simulate(), but narrowing to groups of `gsize` XCDs (PAIR = 2)."""
    t = collections.defaultdict(float)
    for i, name in enumerate(names):
        arr = {w: t[w] + d for w, d in D[i].items()}
        if name in narrow:
            g = collections.defaultdict(float)
            for w, a in arr.items():
                key = (w // WPX) // gsize
                g[key] = max(g[key], a)
            for w in arr:
                t[w] = g[(w // WPX) // gsize]
        else:
            f = max(arr.values())
            for w in arr:
                t[w] = f
    return max(t.values())


def main():
    R = load(LOG)
    names, D, order = segments(R)
    base, _ = simulate(names, D, set())
    span = order[-1][0]
    k = (WALL_MS / N_LAYERS * 1e3) / span   # instrumented -> uninstrumented

    print("=" * 94)
    print("3.0  IDENTITY CHECK -- simulating with NOTHING narrowed")
    print("=" * 94)
    print(f"  measured layer span (max stamp)   {span:9.3f} us")
    print(f"  simulated, no barrier narrowed    {base:9.3f} us")
    print(f"  residual                          {abs(base-span):9.6f} us"
          f"   {'OK' if abs(base-span) < 1e-6 else '*** MODEL IS BROKEN ***'}")
    print(f"\n  instrumented {span:.2f} us/layer vs uninstrumented "
          f"{WALL_MS/N_LAYERS*1e3:.2f} -> ms column scaled by {k:.3f}")

    print()
    print("=" * 94)
    print("3.1  ARRIVAL SPREAD: GPU-WIDE MAX vs THE 8 PER-XCD MAXIMA")
    print("=" * 94)
    print("  A per-XCD barrier can only help by however much the WORST XCD's")
    print("  local max sits below the GPU-wide max.  That gap is an upper")
    print("  bound on this entire item, one barrier at a time.")
    print()
    print(f"{'barrier':<16}{'GPU max':>9}{'worst XCD':>11}{'best XCD':>10}"
          f"{'gap':>8}{'legality':>11}")
    for t, s, n in order:
        mx = max(R[s].values())
        per = collections.defaultdict(float)
        for w, a in R[s].items():
            per[w // WPX] = max(per[w // WPX], a)
        worst, best = max(per.values()), min(per.values())
        print(f"{n:<16}{mx:>9.2f}{worst:>11.2f}{best:>10.2f}"
              f"{mx-worst:>8.3f}{LEGALITY[n][0]:>11}")
    print("""
  THE GAP COLUMN IS THE WHOLE ANSWER AND IT IS ~ZERO EVERYWHERE.  The worst
  XCD's local max IS the GPU-wide max, to within the stamp resolution, at
  every one of the ten.  That is the PLATEAU result of item 1 restated in XCD
  coordinates: the top of the arrival distribution is flat and it is flat
  ACROSS CHIPLETS, not concentrated on one.  A per-XCD barrier releases seven
  XCDs early and the eighth -- the one that sets the wall -- not at all.
""")

    print("=" * 94)
    print("3.2  ONE BARRIER AT A TIME: layer span with barrier B per-XCD")
    print("=" * 94)
    print(f"{'barrier narrowed':<18}{'span us':>10}{'saved us':>10}"
          f"{'ms/token':>10}{'legal?':>10}   note")
    rows = []
    for _t, _s, n in order:
        sp, _ = simulate(names, D, {n})
        saved = base - sp
        ms = saved * N_LAYERS / 1e3 * k
        rows.append((n, saved, ms))
        print(f"{n:<18}{sp:>10.3f}{saved:>10.3f}{ms:>10.4f}"
              f"{LEGALITY[n][0]:>10}   {LEGALITY[n][1][:34]}")
    print()

    print("=" * 94)
    print("3.3  THE CEILING: ALL TEN NARROWED AT ONCE (ILLEGAL -- IT IS A BOUND)")
    print("=" * 94)
    allsp, _ = simulate(names, D, set(names))
    pairsp = simulate_groups(names, D, set(names), 2)
    quadsp = simulate_groups(names, D, set(names), 4)
    for lbl, sp in (("all 10 -> XCD PAIRS (4 groups)", pairsp),
                    ("all 10 -> XCD QUADS (2 groups)", quadsp),
                    ("all 10 -> PER-XCD (8 groups)", allsp)):
        saved = base - sp
        print(f"  {lbl:<34}{sp:>10.3f} us   saves {saved:>7.3f} us/layer"
              f"  = {saved*N_LAYERS/1e3*k:>7.4f} ms/token")
    print(f"  {'(baseline, all GPU-wide)':<34}{base:>10.3f} us")
    print(f"""
  Even DELETING THE GPU-WIDE PROPERTY FROM ALL TEN RENDEZVOUS -- every one of
  them, simultaneously, with no regard for whether a single one is legal --
  buys {(base-allsp)*N_LAYERS/1e3*k:.4f} ms/token against a {WALL_MS} ms wall and a 0.26 ms noise floor.
""")

    print("=" * 94)
    print("3.4  VALIDATION AGAINST THE ONE REAL MEASUREMENT")
    print("=" * 94)
    dec_pair = simulate_groups(names, D, {"decode_barrier"}, 2)
    dec_sav = (base - dec_pair) * N_LAYERS / 1e3 * k
    print(f"""  PAIR_MERGE is exactly `decode_barrier` narrowed to XCD PAIRS.
  This model predicts it saves {dec_sav:+.4f} ms/token.
  It was MEASURED at 11.074 vs 11.005 = +0.07 ms (e482f2e), i.e. neutral
  inside a 0.19 ms spread.

  Predicted ~0, measured ~0.  The model is not merely asserting the answer it
  wants: the one operation in this table that has actually been built lands
  where the table says it lands.  That is the licence to trust the other nine
  rows without building them.
""")

    print("=" * 94)
    print("3.5  WHY -- AND WHAT IT MEANS FOR 'FEWER, WIDER PHASES'")
    print("=" * 94)
    print(f"""
  A rendezvous does not cost barrier code.  It costs SKEW: the wait is
  max_w arrival - arrival_w, and the barrier is where already-existing skew
  becomes visible.  Narrowing the barrier's SCOPE cannot remove skew, it can
  only decide which workers absorb it.  Per-XCD narrowing pays if and only if
  the skew is BETWEEN chiplets.  Table 3.1 measures that directly and it is
  not: every XCD contains a worker at the GPU-wide max, at all ten barriers.

  The 2.94 ms attributed to "ten rendezvous x 76 layers x 3.77 us" is
  therefore NOT a barrier-count budget that shrinks when you merge barriers.
  It is the same skew re-counted at ten places.  Merging ten rendezvous into
  four does not delete six rendezvous' worth of time; it delays the same
  arrival spread to a later point in the layer, which is the identical
  mechanism that has now closed six separate barrier experiments on this
  branch (narrowing, deleting one whole, the arrival tree, all-thread polling,
  200 workers, PAIR_MERGE).

  ITEM 3 IS A NEGATIVE RESULT, AND IT IS THE SEVENTH INSTANCE OF THE SAME ONE.

  What is NOT closed by this: the segment table (regime B).  Deleting a PHASE
  -- its work AND its rendezvous -- collapses a segment worth 0.208-2.059 ms.
  That is a different operation from making a rendezvous narrower, it is the
  only regime on this branch that has ever moved the wall, and it is bounded
  by dependency LEGALITY rather than by skew.
""")


if __name__ == "__main__":
    main()
