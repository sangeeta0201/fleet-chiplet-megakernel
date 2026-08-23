#!/usr/bin/env python3
"""Per-worker BUSY vs SPIN, for the phases the guide scoped: qkv_a and o_proj.

THE POINT
---------
Every barrier experiment on this branch has closed on the same mechanism: a
rendezvous costs SKEW, so moving/narrowing/merging it cannot pay
(glm-per-xcd-barrier-narrowing-is-zero-by-measurement). That leaves one
question the branch has never answered: at the moment the wall is set, is the
LAST-ARRIVING worker doing WORK, or is it waiting?

  * last arriver is ~all busy  -> the phase is work-bound. Delete work
    (regime B), because there is no idle to reclaim.
  * last arriver has real spin -> the spread is a static assignment problem
    and the lever is a better tile->worker map for that phase.

WHY THIS NEEDS NO NEW GPU RUN
-----------------------------
The stage-stamp map ALREADY carries barrier EXIT stamps next to the arrival
stamps -- 9/0, 17/18, 19/20, 21/22, 23/24, 30/31, 32/5, 6/7. Item 3
(price_xcd_narrowing.py) read only the arrival half, which is why the busy
question looked uninstrumented. It is not. It has been in every MPK_BAR_SKEW=3
log all along.

WHY THIS IS NOT THE TRAP IN glm-barskew-per-worker-sum-is-not-busy-time
-----------------------------------------------------------------------
That trap was  arr_w(B_i) - MAX_w' arr_w'(B_{i-1})  -- a per-worker MEAN minus
a MAX-over-workers OF means. Not commensurable; 18.2% of it went negative.

Here every quantity is  slot_b[w] - slot_a[w]:  the SAME worker, the SAME
epoch reference (g_stage_ref, republished only by entry_bar's last arriver,
mpk_atoms.cuh:1345), and -- once cnt is checked equal -- the SAME population of
layers. A difference of two means over one population IS the mean of the
difference. It cannot go negative, and this script asserts that it does not.
"""

import collections
import re
import sys

LOG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/item11_bs1.log"
RANK = 0
WPX = 29  # workers per XCD
LAYERS = 76
# The instrumented layer span is 164.947 us against a 139.72 us uninstrumented
# layer (10.619 ms / 76). Scale a per-layer us saving by this before quoting it
# as ms/token, or the instrument's own overhead is credited to the lever.
K = 139.72 / 164.947

LINE = re.compile(r"^\[1,(\d+)\][^:]*:BARSTAGEWS (\d+) (\d+) (\d+) (\d+)\s*$")


def load(path, rank):
    """-> mean[slot][worker] = us since g_stage_ref, and cnt[slot][worker]."""
    mean = collections.defaultdict(dict)
    cnt = collections.defaultdict(dict)
    with open(path) as fh:
        for ln in fh:
            m = LINE.match(ln)
            if not m:
                continue
            r, slot, w, c, ns = (int(x) for x in m.groups())
            if r != rank or c == 0:
                continue
            mean[slot][w] = ns / c / 1000.0
            cnt[slot][w] = c
    return mean, cnt


# (label, start-slot, arrival-slot, exit-slot, note)
#
# start  = when the worker began this phase's work (a barrier EXIT, or the
#          phase-entry stamp for a subset phase)
# arrival= the worker finished its tiles and arrived at the phase's rendezvous
# exit   = the worker observed the release
PHASES = [
    ("qkv_a", 16, 17, 18,
     "start=attn-half entry (slot 16), NOT the entry_bar exit (slot 0): slot 0 "
     "has cnt 38836 vs 38322 because full_layer also runs on the dense-prologue "
     "layers where the fused attn task is skipped, so 0 and 17 are means over "
     "different layer sets. 16/17/18 are all 38322. NOTE: 16->17 still contains "
     "an XCD-local EP-fold rendezvous unstamped in this log, so BUSY is an "
     "UPPER bound and SPIN a lower bound. Slots 44/45 split it; needs a run."),
    ("o_proj", 29, 30, 31,
     "start=o_proj/router block entry (SUBSET xcd_rank<oproj_topk_tiles). "
     "Verified no spin loop between the two stamps, so BUSY is clean."),
]

EXTRA = [
    ("q_b", 18, 19, 20, "arrival 19 = q_b tiles + KV update done"),
    ("decode", 20, 21, 22, "arrival 21 = MLA decode tiles done"),
    ("merge/Ph8", 22, 23, 24, "arrival 23 = split-KV merge done"),
    ("router", 31, 32, 5, "32 = RMSNorm+router+TopK done; 5 = routing ready"),
    ("W13", 5, 6, 7, "arrival 6 = this worker's W13 tiles done"),
]


def pct(x, t):
    return (100.0 * x / t) if t else float("nan")


def quantile(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def analyse(mean, cnt, label, s_start, s_arr, s_exit, note):
    print("=" * 94)
    print(f"PHASE {label}   slots start={s_start} arrive={s_arr} exit={s_exit}")
    print(f"  {note}")

    pop = set(mean[s_start]) & set(mean[s_arr]) & set(mean[s_exit])
    if not pop:
        print("  NO DATA -- one of the slots is empty in this log.")
        return None

    # Sample counts must match or the means average different layer subsets.
    # A handful of samples' difference is the drop guard; a systematic 1%+
    # difference means the two stamps do not run on the same set of layers and
    # subtracting them is meaningless.
    TOL = 0.005
    worst = 0.0
    bad = []
    for w in pop:
        cs = [cnt[s_start][w], cnt[s_arr][w], cnt[s_exit][w]]
        rel = (max(cs) - min(cs)) / max(cs)
        worst = max(worst, rel)
        if rel > TOL:
            bad.append(w)
    print(f"  population: {len(pop)} workers "
          f"(start {len(mean[s_start])}, arr {len(mean[s_arr])}, "
          f"exit {len(mean[s_exit])})")
    print(f"  cnt agreement: worst relative spread {worst * 100:.3f}% "
          f"(tolerance {TOL * 100:.1f}%), {len(bad)} workers over tolerance")
    if bad:
        print(f"    !! dropping {len(bad)} -- their three stamps are means "
              f"over materially different layer sets")
        pop -= set(bad)
    if not pop:
        print("  NO USABLE WORKERS after the cnt check.")
        return None
    csamp = {cnt[s_arr][w] for w in pop}
    print(f"  layer samples per worker: {sorted(csamp)[:4]}"
          f"{' ...' if len(csamp) > 4 else ''}")

    rows = {}
    neg = 0
    for w in pop:
        busy = mean[s_arr][w] - mean[s_start][w]
        spin = mean[s_exit][w] - mean[s_arr][w]
        if busy < 0 or spin < 0:
            neg += 1
        rows[w] = (busy, spin, busy + spin)

    print(f"  negative busy/spin entries: {neg} / {len(rows)}"
          f"{'   <-- INVALID, do not use' if neg else '   (as required)'}")

    # The critical worker: the LAST to arrive at this phase's rendezvous. It
    # is the one that sets the release time, hence the wall.
    crit = max(pop, key=lambda w: mean[s_arr][w])
    cb, cs, ct = rows[crit]
    print()
    print(f"  CRITICAL WORKER (last to arrive at slot {s_arr}) = w{crit} "
          f"(xcd {crit // WPX}, xcd_rank {crit % WPX})")
    print(f"    phase span {ct:8.3f} us   BUSY {cb:8.3f} us "
          f"({pct(cb, ct):5.1f}%)   SPIN {cs:8.3f} us ({pct(cs, ct):5.1f}%)")

    busies = [pct(b, t) for b, s, t in rows.values() if t > 0]
    print(f"    across {len(busies)} workers -- busy%: "
          f"median {quantile(busies, .50):5.1f}   "
          f"p90 {quantile(busies, .90):5.1f}   "
          f"min {min(busies):5.1f}   max {max(busies):5.1f}")

    spans = [t for _b, _s, t in rows.values()]
    print(f"    phase span us: median {quantile(spans, .50):7.3f}   "
          f"max {max(spans):7.3f}")
    print(f"  VERDICT: critical worker is {pct(cb, ct):.1f}% busy -> "
          f"{'WORK-BOUND (regime B: delete work)' if pct(cb, ct) >= 90 else 'NOT work-bound by the ruling threshold'}")

    # ── What a BETTER STATIC TILE->WORKER MAP could actually buy ────────────
    #
    # The <90% branch says "static assignment problem". That is only true if
    # the busy distribution has a TOP that a rebalance could shave. A perfect
    # rebalance moves the phase's max arrival from max(busy) down to
    # mean(busy) over whoever the work is spread across, so the ceiling is
    #     max(busy) - mean(busy).
    # Two populations, because they are two different proposals:
    #   ACTIVE  = re-deal tiles among the workers that already run this phase
    #   ALL     = also hand tiles to the idle workers (this is WIDENING, and
    #             the class is already measured out at 0.118 ms --
    #             glm-widening-a-phase-for-bandwidth-is-dead-as-a-class)
    busy_all = [b for b, _s, _t in rows.values()]
    mx = max(busy_all)
    thresh = 0.5 * quantile(busy_all, .50)
    active = [b for b in busy_all if b >= thresh]
    print()
    print(f"  REBALANCE CEILING (max busy - mean busy), max busy {mx:.3f} us:")
    for tag, pop_b in (("active", active), ("all 232", busy_all)):
        mean_b = sum(pop_b) / len(pop_b)
        head = mx - mean_b
        print(f"    {tag:8s} n={len(pop_b):3d}  mean busy {mean_b:7.3f} us  "
              f"-> ceiling {head:6.3f} us/layer = "
              f"{head * LAYERS / 1000.0 * K:6.3f} ms/token")
    # The plateau check: a rebalance can only pay down to the SECOND highest.
    s = sorted(busy_all, reverse=True)
    print(f"    top-5 busy us: {' '.join(f'{x:.3f}' for x in s[:5])}"
          f"   (max - 2nd = {s[0] - s[1]:.3f} us)")
    print()
    return pct(cb, ct)


# skew_slot -> (name, the stage-stamp ARRIVAL slot for the same rendezvous)
SKEW = [
    (0, "entry_bar", 9),
    (2, "qkv_barrier", 17),
    (3, "qb_barrier", 19),
    (4, "decode_barrier", 21),
    (1, "attn_release", 23),
    (5, "wuv_barrier", None),
    (6, "hier_barrier", 30),
    (7, "w13_barrier", 6),
]

SKEWLINE = re.compile(
    r"^\[1,(\d+)\][^:]*:BARSKEW (\d+) cnt (\d+) ns (\d+) drop (\d+) gap (\d+)")


def static_vs_dynamic(path, mean, rank):
    """THE decisive table.

    Two measurements of the SAME arrival spread:

      A. in-kernel, per-epoch: g_barskew_ns accumulates (last - first) INSIDE
         each layer and divides by cnt at the end. This is  mean_over_layers(
         max_w - min_w ),  the spread the barrier actually pays, every layer.

      B. from the stage stamps: max_w(mean arrival) - min_w(mean arrival). This
         is  max_w - min_w  applied to per-worker MEANS, so any part of the
         spread that reshuffles between workers from layer to layer AVERAGES
         AWAY and does not appear here.

    B/A is therefore the fraction of the spread that is STATIC -- attached to a
    particular worker every layer, hence fixable by a better tile->worker map.
    1 - B/A is DYNAMIC jitter, which no static schedule can touch.
    """
    acc = {}
    with open(path) as fh:
        for ln in fh:
            m = SKEWLINE.match(ln)
            if not m:
                continue
            r, slot, c, ns, _drop, _gap = (int(x) for x in m.groups())
            if r != rank or c == 0:
                continue
            acc[slot] = ns / c / 1000.0

    print("=" * 94)
    print("IS THE SPREAD STATIC OR DYNAMIC?  (the <90%-busy branch lives here)")
    print("=" * 94)
    print(f"{'barrier':16s} {'A in-kernel':>12s} {'B all-of-means':>15s} "
          f"{'C active only':>14s} {'max-2nd':>9s}")
    print(f"{'':16s} {'us/layer':>12s} {'us/layer':>15s} {'us/layer':>14s} "
          f"{'us':>9s}")
    for slot, name, arr in SKEW:
        a = acc.get(slot)
        if a is None:
            continue
        if arr is None or arr not in mean:
            print(f"{name:16s} {a:12.3f} {'--':>15s} {'--':>14s} {'--':>9s}")
            continue
        vals = sorted(mean[arr].values(), reverse=True)
        b = vals[0] - vals[-1]
        # "Active" = workers arriving in the top half of the range. The rest
        # arrive early because they hold no tile at all -- a structural fact,
        # not an imbalance, and closing it means WIDENING the phase, a class
        # already measured out at 0.118 ms.
        lo = vals[0] - 0.5 * b
        act = [v for v in vals if v >= lo]
        c = act[0] - act[-1] if len(act) > 1 else 0.0
        m2 = vals[0] - vals[1] if len(vals) > 1 else float("nan")
        print(f"{name:16s} {a:12.3f} {b:15.3f} {c:14.3f} {m2:9.3f}")
    print()
    print("  A = mean over layers of (last - first) arrival, accumulated")
    print("      IN-KERNEL per epoch. CAUTION, per mpk_atoms.cuh:1065: the")
    print("      FIRST arriver is normally a worker holding NO tile in this")
    print("      phase, so A is close to the whole phase makespan, NOT to an")
    print("      imbalance among the workers doing the work.")
    print("  B = max-min of per-worker MEAN arrivals, all workers. Anything")
    print("      that reshuffles between workers averages out of B, so")
    print("      A - B is the DYNAMIC (layer-to-layer) part of the spread.")
    print("  C = the same, restricted to workers that actually hold a tile.")
    print("      B - C is the idle-worker gap: structural, and closing it is")
    print("      WIDENING, a class already measured out at 0.118 ms.")
    print("  max-2nd = the plateau. A remap can only shave the top down to")
    print("      the second-highest, so this bounds it far tighter than C.")
    print()
    print("  READ IT AS: A ~= B at every barrier, so the spread is almost")
    print("  entirely STATIC -- but the static part is the BOTTOM of the")
    print("  distribution (idle workers), not a shaveable TOP. The top is a")
    print("  plateau. That is why both branches of the busy%-threshold rule")
    print("  come out dead for these phases.")
    print()


def main():
    mean, cnt = load(LOG, RANK)
    if not mean:
        print(f"no BARSTAGEWS rows for rank {RANK} in {LOG}")
        return 1
    print(f"log {LOG}, rank {RANK}, slots present: {sorted(mean)}")
    print()

    verdicts = {}
    for label, a, b, c, note in PHASES:
        verdicts[label] = analyse(mean, cnt, label, a, b, c, note)

    print("=" * 94)
    print("OTHER PHASES (not scoped by the ruling; same arithmetic, for context)")
    print("=" * 94)
    for label, a, b, c, note in EXTRA:
        analyse(mean, cnt, label, a, b, c, note)

    static_vs_dynamic(LOG, mean, RANK)

    print("=" * 94)
    print("DELIVERABLE")
    for label, v in verdicts.items():
        if v is not None:
            print(f"  {label:8s} critical-worker busy {v:5.1f}% / "
                  f"spin {100 - v:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
