#!/usr/bin/env python3
"""Whole-trace span table: how much wall time each named span accounts for.

Consumes the JSON written by phase_slots_to_perfetto.py. The Perfetto UI shows
where time goes for one worker at a time; this answers the aggregate question
-- across all 248 workers and all 36 layers, which named span is the biggest
line, and is it big because it is slow or because it is drawn many times.

Those two causes are different problems and the mean hides both, so the table
reports n, median, and total separately: a span with n=20228 and a 3.8 us
median is a per-tile cost repeated 20k times, and a span with n=8928 and an
18.8 us median is one expensive wait per worker-layer. Optimising them means
opposite things.

  Usage: span_table.py <trace.json> [--cat phase|subphase|all]
                       [--filter SUBSTR] [--by-arm]

--by-arm additionally breaks each span down by the args set on it
(participated / running_tile / split_of), which is how a "skipped" span is
told apart from one that ran.
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict


def load(path):
    with open(path) as f:
        d = json.load(f)
    return [e for e in d["traceEvents"] if e.get("ph") == "X"]


def table(rows, title, top=None, floor=None):
    """rows: {name: [durations]} -> print sorted by total descending.

    `floor` is the instrument's own measured cost. A whole-table "30.8% of
    spans are under the floor" was not enough to read a row by: it left every
    small median ambiguous, and a row whose every sample is instrument noise
    looks identical to a row that is genuinely 0.44 us of kernel. The %flr
    column resolves that per row -- 100% means the span measured nothing but
    the mark.
    """
    if not rows:
        return
    items = sorted(rows.items(), key=lambda kv: -sum(kv[1]))
    if top:
        items = items[:top]
    wide = max(len(n) for n, _ in items)
    grand = sum(sum(v) for v in rows.values())
    fcol = f"  {'%flr':>5}" if floor else ""
    print(f"\n{title}")
    print(f"  {'span':<{wide}}  {'n':>7}  {'median':>9}  {'mean':>9}  "
          f"{'p95':>9}  {'total':>11}  {'share':>7}{fcol}")
    print("  " + "-" * (wide + 62 + (7 if floor else 0)))
    for name, ds in items:
        ds_s = sorted(ds)
        p95 = ds_s[min(len(ds_s) - 1, int(0.95 * len(ds_s)))]
        tot = sum(ds)
        f = (f"  {100.0*sum(1 for d in ds if d <= floor)/len(ds):>4.0f}%"
             if floor else "")
        print(f"  {name:<{wide}}  {len(ds):>7}  {statistics.median(ds):>8.2f}u  "
              f"{tot/len(ds):>8.2f}u  {p95:>8.2f}u  {tot/1000.0:>10.2f}m  "
              f"{100.0*tot/grand:>6.1f}%{f}")
    print(f"  {'TOTAL':<{wide}}  {sum(len(v) for v in rows.values()):>7}  "
          f"{'':>9}  {'':>9}  {'':>9}  {grand/1000.0:>10.2f}m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--cat", default="phase",
                    choices=["phase", "subphase", "all"],
                    help="phase = worker rows, subphase = detail rows")
    ap.add_argument("--filter", default=None,
                    help="only spans whose name contains this substring")
    ap.add_argument("--by-arm", action="store_true",
                    help="split each span by its participated/running_tile args")
    # A barrier wait is bimodal by construction: the workers on the critical
    # path leave immediately and the rest pay the difference. Reporting one
    # median over all of them therefore describes nobody, which is how a
    # 9.88 us tail on one rank once hid inside a 0.56 us median.
    #
    # Rank is derived from the trace (workers grouped by their own xcd arg,
    # then position within that group) rather than from worker % 31, so no
    # assumption about the dispatch formula is baked in. Note that a rank row
    # aggregates the 8 workers holding that rank, one per XCD: its median is a
    # per-worker figure but its TOTAL is 8 workers' time, not one worker's.
    ap.add_argument("--by-rank", action="store_true",
                    help="break each span down by rank within its XCD")
    ap.add_argument("--top", type=int, default=None)
    args = ap.parse_args()

    ev = load(args.trace)
    if args.cat != "all":
        ev = [e for e in ev if e.get("cat") == args.cat]
    if args.filter:
        ev = [e for e in ev if args.filter in e["name"]]
    if not ev:
        print("no slices matched", file=sys.stderr)
        return 1

    rank = {}
    if args.by_rank:
        by_xcd = defaultdict(set)
        for e in ev:
            a = e.get("args", {})
            if "xcd" in a and "worker" in a:
                by_xcd[a["xcd"]].add(a["worker"])
        for ws in by_xcd.values():
            for i, w in enumerate(sorted(ws)):
                rank[w] = i

    rows = defaultdict(list)
    for e in ev:
        key = e["name"]
        if args.by_rank:
            r = rank.get(e.get("args", {}).get("worker"))
            key += f"  rank {r:>2}" if r is not None else "  rank ?"
        if args.by_arm:
            a = e.get("args", {})
            bits = []
            if a.get("participated") is False:
                bits.append("skipped")
            if a.get("running_tile") is False:
                bits.append("gap")
            if bits:
                key += "  [" + ",".join(bits) + "]"
        rows[key].append(e["dur"])

    # The trace measures itself: slot 4 spans two phase marks on adjacent
    # source lines with no code between them, so its duration IS one mark's
    # cost. Report it as the floor so a median can be read against something,
    # rather than leaving every reader to wonder whether a 0.5 us span is a
    # fast op or no op at all.
    floor = [e["dur"] for e in load(args.trace)
             if e["name"].startswith("instrument floor")]

    # Makespan, so "share" above can be read against wall time rather than
    # against the sum of all worker-seconds (which counts 248 workers at once).
    lo = min(e["ts"] for e in ev)
    hi = max(e["ts"] + e["dur"] for e in ev)
    table(rows, f"spans (cat={args.cat}"
                f"{', filter=' + args.filter if args.filter else ''})",
          top=args.top,
          floor=statistics.median(floor) if floor else None)
    print(f"\n  makespan {hi - lo:.1f} us; worker-time totals above sum across "
          f"all workers, so they exceed it by design.")
    if floor:
        f = statistics.median(floor)
        under = sum(1 for e in ev if e["dur"] <= f)
        print(f"  instrument floor {f:.2f} us (median of {len(floor)} "
              f"self-measurements); {under}/{len(ev)} "
              f"({100.0*under/len(ev):.1f}%) of the spans above are at or "
              f"under it and report the instrument, not the kernel.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
