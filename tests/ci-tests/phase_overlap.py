#!/usr/bin/env python3
"""Measure how much of a layer two different ops are actually running at once.

Reads the same [PTRACEW] log the Perfetto exporter reads and answers, per XCD
per layer: for each pair of ops, how much wall time has at least one worker in
op A while at least one other worker is in op B. That is the number a claim like
"attention overlaps the QKV GEMM" has to rest on.

The subtlety this script exists for: a phase mark is taken by every worker that
REACHES the phase, not only by the ones that run it. The marks sit outside their
guards -- they have to, since a mark inside would leave non-participants with no
timestamp and break the ascending-marks invariant -- so a worker that branches
past QKV still stamps slot 1 and still looks, from ticks alone, like a worker
that ran a very fast GEMM. Counting those intervals as "in QKV" turns a
dependency stall into a concurrency finding. Overlap computed that way is
phantom: it reports parallelism that consists of workers doing nothing.

So every interval on a gated slot is admitted only if its p= participation bit
is set. Runs with --naive to reproduce the wrong number for comparison.

  Usage: phase_overlap.py <run.log> [--naive] [-o pairs.json]

Requires a log with p= masks (MPK_PHASE_PART). Without them the gated slots
cannot be filtered and the script refuses rather than silently reporting the
phantom number.
"""
import argparse
import json
import re
import sys
from collections import defaultdict

ROW_RE = re.compile(
    r"^\[PTRACEW\] w=(\d+) x=(-?\d+) l=(\d+)(?: p=(\d+))?(?: a=(\d+))?"
    r"((?: \d+)+)\s*$")
HDR_RE = re.compile(r"^\[PTRACE\] slots=(\d+) layers=(\d+) tick_ns=(\d+)\s*$")
SUB_RE = re.compile(r"^\[PTRACES\] w=(\d+) l=(\d+) n=(\d+)((?: \d+:\d+)+)\s*$")

# Per-tile marks bounding each MoE arm. With these the arms stop being a
# property of the whole phase and become intervals: a worker that runs W13 then
# W2 is busy in W13 only for its W13 tile, not for the entire slot-8 span. That
# distinction is the whole intra-phase number -- scored phase-wide, a both-arms
# worker reads as simultaneously in both arms for 20 us, which is not a fact
# about the machine but about the resolution of the measurement.
ARM_MARKS = {"moe W13+SwiGLU": (10, 11), "moe W2": (12, 13)}

# Slots that are real compute. Barriers and spin-waits are excluded: two workers
# both sitting in layer_gate_poll is not two ops overlapping, and counting it
# would make every layer look ~100% concurrent.
COMPUTE_SLOTS = {
    1: "qkv_fused",
    3: "attention",
    6: "oproj+router",
    8: "moe",
}
# Of those, the ones whose span only means the op when the bit is set. Here that
# is all of them -- each sits behind an xcd_rank guard -- but they are listed
# separately because the two facts are independent and the next slot added may
# be ungated.
GATED_SLOTS = {1, 3, 6, 8}

# Fused phases whose workers split into arms. Slot 8 is one phase in the tick
# stream but two different ops in the machine: each tile is a W13 tile or a W2
# tile, some workers get no tile at all, and the strided loop hands the low
# ranks two tiles so they run one arm and then the other. Counting all of that
# as "moe" is what made the phase look 100% occupied when workers were idle
# inside it. The code is a set of arms -- 0 none, 1 W13, 2 W2, 3 both -- and
# each distinct set is scored as its own op, because the span is one phase wide
# and cannot be cut at the tile boundary inside it.
ARM_BITS, ARM_MASK = 2, 3
ARM_OPS = {
    8: {0: None,   # entered the loop, got no tile -- not compute
        1: "moe W13+SwiGLU",
        2: "moe W2",
        3: "moe W13+W2"},
}


def parse(path):
    tick_ns, rows, ungated = None, [], 0
    subs = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            if line.startswith("[PTRACE] "):
                m = HDR_RE.match(line)
                if m:
                    tick_ns = int(m.group(3))
            elif line.startswith("[PTRACES] "):
                m = SUB_RE.match(line)
                if m:
                    subs[(int(m.group(1)), int(m.group(2)))] = [
                        (int(c), int(t)) for c, t in
                        (p.split(":") for p in m.group(4).split())]
            elif line.startswith("[PTRACEW] "):
                m = ROW_RE.match(line)
                if not m:
                    continue
                ts = [int(x) for x in m.group(6).split()]
                if m.group(4) is None:
                    ungated += 1
                    continue
                rows.append((int(m.group(1)), int(m.group(2)), int(m.group(3)),
                             ts, int(m.group(4)),
                             int(m.group(5)) if m.group(5) else None))
    return tick_ns, rows, ungated, subs


def op_of(s, pt, am, naive):
    """Which op this worker ran in slot s, or None if it ran nothing.

    Two independent ways to occupy a phase without doing its work, and both
    have bitten this analysis: falling through a rank guard (the bit), and
    entering a fused phase but taking the other arm or no tile (the arm code).
    """
    if naive:
        return COMPUTE_SLOTS[s]
    if s in GATED_SLOTS and not (pt >> s) & 1:
        return None
    if s in ARM_OPS and am is not None:
        return ARM_OPS[s].get((am >> (s * ARM_BITS)) & ARM_MASK,
                              COMPUTE_SLOTS[s])
    return COMPUTE_SLOTS[s]


def union_len(ivals):
    """Total length covered by a set of [a,b) intervals, overlaps counted once."""
    if not ivals:
        return 0
    out, (cs, ce) = 0, ivals[0]
    for a, b in ivals[1:]:
        if a > ce:
            out += ce - cs
            cs, ce = a, b
        else:
            ce = max(ce, b)
    return out + ce - cs


def intersect(xs, ys):
    """Intervals present in both unions. Both inputs must be sorted."""
    out, i, j = [], 0, 0
    while i < len(xs) and j < len(ys):
        a = max(xs[i][0], ys[j][0])
        b = min(xs[i][1], ys[j][1])
        if b > a:
            out.append((a, b))
        if xs[i][1] < ys[j][1]:
            i += 1
        else:
            j += 1
    return out


def merge(ivals):
    ivals = sorted(ivals)
    out = []
    for a, b in ivals:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--naive", action="store_true",
                    help="ignore p= and count every worker that reached the "
                         "phase (reproduces the phantom number)")
    ap.add_argument("-o", "--out", help="write per-pair JSON here")
    args = ap.parse_args()

    tick_ns, rows, ungated, subs = parse(args.log)
    if tick_ns is None:
        sys.exit(f"no [PTRACE] header in {args.log}")
    if not rows:
        sys.exit(f"{args.log} has no rows with p= masks ({ungated} without). "
                 "Rebuild with MPK_PHASE_PART and recapture; without the masks "
                 "the gated slots cannot be separated from fall-through.")
    if ungated:
        print(f"note: skipped {ungated} rows with no p= mask")

    # Per (xcd, layer, slot): the set of intervals during which SOME worker was
    # in that op. Per-XCD because the eight XCDs run the layer independently and
    # unioning them would let XCD 0's QKV "overlap" XCD 5's MoE.
    by = defaultdict(list)
    admitted, rejected = defaultdict(int), defaultdict(int)
    for w, x, l, ts, pt, am in rows:
        for s in COMPUTE_SLOTS:
            a, b = ts[s - 1], ts[s]
            if a == 0 or b == 0 or b <= a:
                continue
            op = op_of(s, pt, am, args.naive)
            if op is None:
                rejected[COMPUTE_SLOTS[s]] += 1
                continue
            admitted[op] += 1
            by[(x, l, op)].append((a, b))

    # Ops are named, not slot-numbered, because slot 8 yields two of them.
    ops = sorted({k[2] for k in by} | set(admitted))

    for k in by:
        by[k] = merge(by[k])

    mode = "NAIVE (marks only)" if args.naive else "participation-filtered"
    print(f"log {args.log}")
    print(f"mode: {mode}\n")
    print(f"{'op':<18} {'intervals':>10} {'rejected':>10}")
    for name in ops:
        print(f"{name:<18} {admitted[name]:>10} {rejected.get(name, 0):>10}")

    # Denominator: the wall time of the layer on that XCD, summed over all
    # (xcd, layer). Overlap is reported as a fraction of that, so "3 us" becomes
    # a share of the thing it is supposed to be hiding under.
    layer_wall = defaultdict(lambda: [None, None])
    for w, x, l, ts, pt, am in rows:
        lo, hi = ts[0], ts[len(ts) - 1]
        if not lo or hi <= lo:
            continue
        cur = layer_wall[(x, l)]
        cur[0] = lo if cur[0] is None else min(cur[0], lo)
        cur[1] = hi if cur[1] is None else max(cur[1], hi)
    total_wall = sum((hi - lo) for lo, hi in layer_wall.values()
                     if lo is not None)
    total_us = total_wall * tick_ns / 1000.0

    # Which slot each op came from. Two ops from the SAME slot are two arms of
    # one fused phase, and their "overlap" is not the thing this script is
    # measuring: the question is whether the megakernel runs different *phases*
    # at once, and two MoE workers are both in MoE however their tiles fell.
    # Worse, the arm split manufactures such pairs -- a W13-only worker against
    # a both-arms worker is 16% of wall, which would land in the headline as
    # concurrency and read as an improvement on the pre-split number. So the
    # two kinds are summed separately and only cross-phase feeds the headline.
    op_slot = {}
    for s in COMPUTE_SLOTS:
        for r in rows:
            o = op_of(s, r[4], r[5], args.naive)
            if o:
                op_slot.setdefault(o, s)

    results = {}
    cross, intra = [], []
    for i, o1 in enumerate(ops):
        for o2 in ops[i + 1:]:
            same = op_slot.get(o1) is not None and op_slot.get(o1) == \
                op_slot.get(o2)
            (intra if same else cross).append((o1, o2))

    def score(pair_list, header, note=None):
        print(f"\n{header}")
        if note:
            print(f"  {note}")
        if not pair_list:
            print("  (none)")
            return []
        print(f"{'pair':<34} {'overlap us':>12} {'% of wall':>10} "
              f"{'us/layer-xcd':>14}")
        got = []
        for s1, s2 in pair_list:
            tot = 0
            for (x, l) in layer_wall:
                a = by.get((x, l, s1))
                b = by.get((x, l, s2))
                if not a or not b:
                    continue
                tot += union_len(intersect(a, b))
            us = tot * tick_ns / 1000.0
            pct = 100.0 * us / total_us if total_us else 0.0
            nm = f"{s1} || {s2}"
            results[nm] = {"overlap_us": round(us, 3),
                           "pct_of_wall": round(pct, 4),
                           "kind": "intra_phase" if (s1, s2) in intra
                                   else "cross_phase",
                           "us_per_layer_xcd": round(
                               us / max(len(layer_wall), 1), 4)}
            print(f"{nm:<34} {us:>12.2f} {pct:>9.3f}% "
                  f"{us / max(len(layer_wall), 1):>14.4f}")
            got.append((s1, s2))
        return got

    print(f"\npairwise concurrency, per XCD per layer, summed over "
          f"{len(layer_wall)} (xcd,layer) units")
    print(f"total layer wall time across those units: {total_us:.1f} us")
    score(cross, "cross-phase (different phases running at the same time)")
    score(intra, "intra-phase (same phase, workers on different arms)",
          "still not cross-phase concurrency -- two MoE workers are both in\n"
          "  MoE however their tiles fell. Reported, not summed into the\n"
          "  headline. Scored phase-wide; see the tile-resolution figure below.")

    # ── the same question at tile resolution ─────────────────────────────
    # The phase-wide intra number above is mostly an artifact of the interval
    # being one phase wide: a worker that runs W13 then W2 is counted as busy
    # in BOTH for its entire span, so it overlaps itself. Marks 10-13 bound the
    # tiles, so the arms can be scored as what they are -- disjoint intervals
    # inside one worker's span. The gap between the two is real: it is loop
    # overhead and barrier wait belonging to neither arm.
    if any(subs.get((r[0], r[2])) for r in rows):
        arm_iv = defaultdict(list)
        n_tiles = defaultdict(int)
        for w, x, l, ts, pt, am in rows:
            a, b = ts[7], ts[8]
            if not a or not b or b <= a:
                continue
            byc = defaultdict(list)
            for code, tick in (subs.get((w, l)) or []):
                byc[code].append(tick)
            for nm, (c0, c1) in ARM_MARKS.items():
                for beg, end in zip(byc.get(c0, []), byc.get(c1, [])):
                    if end > beg and beg >= a and end <= b:
                        arm_iv[(x, l, nm)].append((beg, end))
                        n_tiles[nm] += 1
        for k in arm_iv:
            arm_iv[k] = merge(arm_iv[k])
        names = sorted({k[2] for k in arm_iv})
        print("\nintra-phase at TILE resolution (marks 10-13, not the phase "
              "span)")
        for nm in names:
            print(f"  {nm:<22} {n_tiles[nm]:>6} tiles")
        if len(names) > 1:
            print(f"{'pair':<34} {'overlap us':>12} {'% of wall':>10}")
            for i, o1 in enumerate(names):
                for o2 in names[i + 1:]:
                    tot = 0
                    for (x, l) in layer_wall:
                        p = arm_iv.get((x, l, o1))
                        q = arm_iv.get((x, l, o2))
                        if p and q:
                            tot += union_len(intersect(p, q))
                    us = tot * tick_ns / 1000.0
                    pct = 100.0 * us / total_us if total_us else 0.0
                    results[f"{o1} || {o2} (tile)"] = {
                        "overlap_us": round(us, 3),
                        "pct_of_wall": round(pct, 4),
                        "kind": "intra_phase_tile"}
                    print(f"{o1 + ' || ' + o2:<34} {us:>12.2f} {pct:>9.3f}%")

    # ── occupancy ────────────────────────────────────────────────────────
    # How many workers are in a compute op at once. Same participation rule:
    # a worker that branched past QKV is not "in QKV", it is idle, and counting
    # it inflates occupancy precisely where the machine is emptiest.
    print(f"\ncompute occupancy (workers simultaneously in a compute op)")
    print(f"{'op':<18} {'peak/XCD':>9} {'peak die-wide':>14}")
    by_layer = defaultdict(list)
    for w, xx, ll, ts, pt, am in rows:
        by_layer[ll].append((w, xx, ts, pt, am))
    peaks = {}
    slot_of = {}
    for s in COMPUTE_SLOTS:
        for r in rows:
            o = op_of(s, r[4], r[5], args.naive)
            if o:
                slot_of.setdefault(o, s)
    for name in ops:
        s = slot_of.get(name)
        if s is None:
            continue
        pk_xcd, pk_die = 0, 0
        for ll, members in by_layer.items():
            per_xcd, die = defaultdict(list), []
            for w, xx, ts, pt, am in members:
                a, b = ts[s - 1], ts[s]
                if a == 0 or b == 0 or b <= a:
                    continue
                if op_of(s, pt, am, args.naive) != name:
                    continue
                per_xcd[xx] += [(a, 1), (b, -1)]
                die += [(a, 1), (b, -1)]
            for evs in list(per_xcd.values()) + [die]:
                evs.sort()
                cur = 0
                for _, d in evs:
                    cur += d
                    if evs is die:
                        pk_die = max(pk_die, cur)
                    else:
                        pk_xcd = max(pk_xcd, cur)
        peaks[name] = {"per_xcd": pk_xcd, "die_wide": pk_die}
        print(f"{name:<18} {pk_xcd:>9} {pk_die:>14}")
    results["_peaks"] = peaks

    # Mean compute occupancy over the layer: time-weighted average of "how many
    # workers are in a compute op", die-wide. This is the number the post quotes
    # as "N of 248 workers"; it is the one most distorted by fall-through,
    # because a non-participant's interval lands exactly in the stretch where
    # the machine is emptiest and papers over it.
    tot_area, tot_time = 0.0, 0.0
    for ll, members in by_layer.items():
        evs = []
        for w, xx, ts, pt, am in members:
            for s in COMPUTE_SLOTS:
                a, b = ts[s - 1], ts[s]
                if a == 0 or b == 0 or b <= a:
                    continue
                if op_of(s, pt, am, args.naive) is None:
                    continue
                evs += [(a, 1), (b, -1)]
        if not evs:
            continue
        evs.sort()
        cur, prev = 0, evs[0][0]
        for t, d in evs:
            if t > prev:
                tot_area += cur * (t - prev)
                prev = t
            cur += d
        # Denominator is the layer's own wall time, not the extent of the
        # compute events. Using the events would shrink the denominator exactly
        # when filtering removes intervals, so occupancy would RISE as the
        # machine was shown to be emptier -- the two modes have to divide by the
        # same thing to be comparable.
        spans = [layer_wall[(x, l)] for (x, l) in layer_wall if l == ll]
        if spans:
            tot_time += max(h for _, h in spans) - min(o for o, _ in spans)
    nworkers = len({r[0] for r in rows})
    mean_occ = tot_area / tot_time if tot_time else 0.0
    print(f"\nmean compute occupancy: {mean_occ:.1f} of {nworkers} workers "
          f"({100.0 * mean_occ / max(nworkers, 1):.0f}%)")
    results["_occupancy"] = {"mean_workers": round(mean_occ, 2),
                             "total_workers": nworkers,
                             "pct": round(100.0 * mean_occ / max(nworkers, 1),
                                          2)}

    # The headline. Cross-phase only, for the reason given above the pair
    # tables: two arms of MoE are still MoE, and folding them in here would
    # raise this number by an order of magnitude while measuring less.
    def any_two_of(pair_list):
        tot = 0
        for (x, l) in layer_wall:
            segs = []
            for s1, s2 in pair_list:
                a, b = by.get((x, l, s1)), by.get((x, l, s2))
                if a and b:
                    segs.extend(intersect(a, b))
            tot += union_len(merge(segs))
        return tot * tick_ns / 1000.0

    any_us = any_two_of(cross)
    intra_us = any_two_of(intra)
    print()
    for lbl, v in (("any two PHASES concurrent", any_us),
                   ("(same-phase, different arms)", intra_us)):
        print(f"{lbl:<34} {v:>12.2f} "
              f"{100.0 * v / total_us if total_us else 0:>9.3f}%")
    results["_any_two"] = {"overlap_us": round(any_us, 3),
                           "kind": "cross_phase",
                           "pct_of_wall": round(
                               100.0 * any_us / total_us if total_us else 0, 4)}
    results["_any_two_intra"] = {
        "overlap_us": round(intra_us, 3), "kind": "intra_phase",
        "pct_of_wall": round(
            100.0 * intra_us / total_us if total_us else 0, 4)}
    results["_meta"] = {"mode": mode, "tick_ns": tick_ns,
                        "layer_xcd_units": len(layer_wall),
                        "total_wall_us": round(total_us, 3),
                        "rows": len(rows)}

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
