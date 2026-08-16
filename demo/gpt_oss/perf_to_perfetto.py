#!/usr/bin/env python3
"""Convert MPK_PERFETTO [PERF] lines into a Perfetto/chrome://tracing JSON.

Usage:
    MPK_PERFETTO=100 ./run_mp4_dp_ep_fused.sh > run.log
    python perf_to_perfetto.py run.log trace.json
    # open trace.json at https://ui.perfetto.dev

Input lines look like (mpirun prefixes them with the rank tag):
    [1,0]<stdout>:[PERF] <worker> <layer> <phase> <t_entry> <t_exit>

Ticks are s_memrealtime at 100 MHz, i.e. 10 ns each, relative to a per-rank
origin. Perfetto wants microseconds, so ticks/100.

Each rank becomes a process and each worker a thread, so the timeline shows
all 240 workers x N GPUs side by side. Worker w is XCD w//30, lane w%30.
"""
import json
import re
import sys
from collections import defaultdict

# Phase ids must match the mpk_perf_span() calls in
# gang_full_layer_fused_mi300.cuh.
PHASES = {
    0: ("qkv_gemm", "thread_state_running"),
    1: ("qkv_barrier", "thread_state_iowait"),
    2: ("attn", "thread_state_running"),
    3: ("xcd_barrier", "thread_state_iowait"),
    4: ("oproj_topk", "thread_state_running"),
    5: ("moe", "thread_state_running"),
    6: ("ep9a_barrier", "thread_state_iowait"),
    7: ("ep9_combine", "thread_state_unknown"),
}

LINE = re.compile(
    r"(?:\[1,(\d+)\]<stdout>:)?\[PERF\]\s+"
    r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    src, dst = sys.argv[1], sys.argv[2]

    events = []
    # Per (rank, phase) totals, so the script also prints the summary the
    # timeline is meant to explain.
    totals = defaultdict(float)
    counts = defaultdict(int)
    # Per (rank, worker) span of the whole captured iteration.
    lo = {}
    hi = {}
    ranks = set()
    # (rank, phase) -> layer -> [(entry_us, exit_us)], for the wall/skew stats.
    by_layer_phase = defaultdict(lambda: defaultdict(list))

    with open(src) as fh:
        for line in fh:
            m = LINE.search(line)
            if not m:
                continue
            rank = int(m.group(1) or 0)
            worker = int(m.group(2))
            layer = int(m.group(3))
            phase = int(m.group(4))
            t_in = int(m.group(5))
            t_out = int(m.group(6))
            if t_out <= t_in:
                continue
            ranks.add(rank)

            name, color = PHASES.get(phase, (f"phase{phase}", "grey"))
            # 10 ns/tick -> us
            ts = t_in / 100.0
            dur = (t_out - t_in) / 100.0

            events.append({
                "name": f"{name} L{layer}",
                "cat": name,
                "ph": "X",
                "ts": ts,
                "dur": dur,
                "pid": rank,
                "tid": worker,
                "cname": color,
                "args": {"layer": layer, "xcd": worker // 30,
                         "lane": worker % 30, "phase": name},
            })
            totals[(rank, name)] += dur
            counts[(rank, name)] += 1
            by_layer_phase[(rank, name)][layer].append((ts, ts + dur))
            key = (rank, worker)
            lo[key] = min(lo.get(key, ts), ts)
            hi[key] = max(hi.get(key, ts + dur), ts + dur)

    if not events:
        print("no [PERF] lines found -- was MPK_PERFETTO set?", file=sys.stderr)
        return 1

    # Name the processes/threads so the Perfetto UI is readable.
    meta = []
    for r in sorted(ranks):
        meta.append({"name": "process_name", "ph": "M", "pid": r, "tid": 0,
                     "args": {"name": f"GPU {r}"}})
        for w in range(240):
            meta.append({
                "name": "thread_name", "ph": "M", "pid": r, "tid": w,
                "args": {"name": f"xcd{w // 30} w{w % 30:02d}"},
            })
            # sort_index keeps workers in numeric order rather than by first
            # event time, so XCD blocks stay contiguous in the UI.
            meta.append({
                "name": "thread_sort_index", "ph": "M", "pid": r, "tid": w,
                "args": {"sort_index": w},
            })

    with open(dst, "w") as fh:
        json.dump({"traceEvents": meta + events,
                   "displayTimeUnit": "ns"}, fh)

    print(f"wrote {dst}: {len(events)} spans, {len(ranks)} rank(s)")

    # Summing a phase over 240 workers is not wall time -- they run in
    # parallel. What the layer actually costs is the WALL span of each
    # (layer, phase) across workers, and what it wastes is the gap between
    # the fastest and slowest worker in that phase.
    for r in sorted(ranks):
        print(f"\nGPU {r}")
        print(f"  {'phase':<14} {'wall us':>9} {'mean us':>9} "
              f"{'slowest':>9} {'skew us':>9}")
        rows = []
        for (rr, name) in sorted({k for k in totals if k[0] == r},
                                 key=lambda k: k[1]):
            per_layer_wall = []
            per_layer_skew = []
            per_layer_max = []
            for L, w2s in by_layer_phase.get((r, name), {}).items():
                entries = [a for a, _ in w2s]
                exits = [b for _, b in w2s]
                durs = [b - a for a, b in w2s]
                per_layer_wall.append(max(exits) - min(entries))
                per_layer_max.append(max(durs))
                per_layer_skew.append(max(durs) - min(durs))
            if not per_layer_wall:
                continue
            n = counts[(r, name)]
            rows.append((
                sum(per_layer_wall) / len(per_layer_wall),
                name,
                totals[(r, name)] / n,
                sum(per_layer_max) / len(per_layer_max),
                sum(per_layer_skew) / len(per_layer_skew),
            ))
        for wall, name, mean, mx, skew in sorted(rows, reverse=True):
            print(f"  {name:<14} {wall:>9.2f} {mean:>9.2f} "
                  f"{mx:>9.2f} {skew:>9.2f}")
        print("  (wall = max exit - min entry over 240 workers, per layer;"
              " skew = slowest - fastest worker)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
