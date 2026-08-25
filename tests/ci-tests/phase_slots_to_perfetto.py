#!/usr/bin/env python3
"""Convert Fleet's [PTRACEW] dump into a Perfetto trace.

Consumes a run built with -DMPK_PHASE_SLOTS -DMPK_PHASE_TRACE. Each [PTRACEW]
line is one (worker, layer) with 12 absolute s_memrealtime ticks; this turns
them into Chrome Trace Format complete events, which ui.perfetto.dev opens
directly.

Why this exists alongside summarize_phase_slots.py: that script reports means,
and a mean cannot show overlap. Two workers each averaging 5 us in MoE tell you
nothing about whether they ran concurrently or back to back -- and concurrency
is the entire claim a megakernel makes. Only absolute timestamps on a shared
clock can show it, which is what s_memrealtime is: a constant-rate counter,
common to every XCD on the die, so timestamps from different workers are
directly comparable without any cross-calibration.

  Usage: phase_slots_to_perfetto.py <run.log> -o trace.json
         then open trace.json at https://ui.perfetto.dev

Layout in the UI: one process per XCD, one thread per worker, one slice per
phase. Threads sort by worker id, so a barrier appears as a wall of slices
ending at the same x and the stragglers are the ones that overhang it.
"""
import argparse
import json
import re
import sys
from collections import defaultdict

# Names match summarize_phase_slots.py's SLOT_NAMES so the two instruments can
# be read side by side. Slot i's span runs from timestamp i-1 to timestamp i,
# which is why slot 0 -- the inter-layer span -- has no slice of its own here:
# its start is the *previous* layer's slot 11, and it is emitted below by
# looking back rather than by pairing within a row.
SPAN_NAMES = [
    None,                     # 0: layer entry (start marker; see above)
    "qkv_gemm",               # 1: ts[0] -> ts[1]
    "qkv_epoch_barrier",      # 2
    "attention+merge",        # 3
    "(marker)",               # 4
    "attn_release_wait",      # 5
    "oproj+rmsnorm+router",   # 6
    "topk_wait",              # 7
    "moe(w13+swiglu+w2)",     # 8
    "layer_arrive+fanout+pf",  # 9
    "layer_gate_poll",        # 10
    "layer_exit",             # 11
]

# Colour by kind, so a screenshot reads without a legend. Perfetto accepts the
# Catapult reserved colour names.
COLOR = {
    "qkv_gemm": "thread_state_running",
    "attention+merge": "thread_state_running",
    "oproj+rmsnorm+router": "thread_state_running",
    "moe(w13+swiglu+w2)": "thread_state_running",
    "qkv_epoch_barrier": "thread_state_iowait",
    "attn_release_wait": "thread_state_iowait",
    "topk_wait": "thread_state_iowait",
    "layer_gate_poll": "thread_state_iowait",
    "inter_layer": "grey",
}

HDR_RE = re.compile(r"^\[PTRACE\] slots=(\d+) layers=(\d+) tick_ns=(\d+)\s*$")
# Strict for the same reason summarize_phase_slots.py is: the host's stdout
# interleaves with the device printf stream, and a spliced line must be
# dropped and counted rather than parsed with missing fields defaulted to 0.
ROW_RE = re.compile(r"^\[PTRACEW\] w=(\d+) x=(-?\d+) l=(\d+)((?: \d+)+)\s*$")


def parse(path):
    hdr = None
    rows = []
    corrupt = 0
    saw_end = False
    with open(path, errors="replace") as fh:
        for line in fh:
            if line.startswith("[PTRACE] "):
                m = HDR_RE.match(line)
                if m:
                    hdr = {
                        "slots": int(m.group(1)),
                        "layers": int(m.group(2)),
                        "tick_ns": int(m.group(3)),
                    }
            elif line.startswith("[PTRACEW] "):
                m = ROW_RE.match(line)
                if not m:
                    corrupt += 1
                    continue
                ts = [int(x) for x in m.group(4).split()]
                # A row that parsed but has the wrong slot count is the same
                # failure as one that did not parse: the printf stream was
                # spliced mid-line. Drop it. Aborting the whole conversion
                # here would mean one interleaved line costs the entire trace.
                if hdr is not None and len(ts) != hdr["slots"]:
                    corrupt += 1
                    continue
                rows.append((int(m.group(1)), int(m.group(2)),
                             int(m.group(3)), ts))
            elif line.startswith("[PTRACE_END]"):
                saw_end = True
    return hdr, rows, corrupt, saw_end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("-o", "--out", default="trace.json")
    ap.add_argument("--layers", type=int, default=0,
                    help="emit only the first N layers (0 = all)")
    args = ap.parse_args()

    hdr, rows, corrupt, saw_end = parse(args.log)
    if hdr is None:
        sys.exit(f"no [PTRACE] header in {args.log} -- was this built with "
                 f"-DMPK_PHASE_TRACE and run with MPK_PHASE_SLOTS=1?")
    if not rows:
        sys.exit(f"[PTRACE] header but no [PTRACEW] rows in {args.log}")
    if corrupt:
        print(f"WARNING: dropped {corrupt} interleaved/truncated rows")
    # The device printf buffer is finite and this is the largest thing Fleet
    # prints. Without the trailing marker the tail is missing, and a truncated
    # trace looks exactly like a real load imbalance -- the last workers appear
    # to stop early. Warn loudly rather than render a plausible lie.
    if not saw_end:
        print("WARNING: no [PTRACE_END] -- the device printf buffer truncated "
              "the dump. Later workers/layers are MISSING, not idle. Raise "
              "HIP_PRINTF_BUFFER_SIZE or lower MPK_PHASE_TRACE_LAYERS.")

    nslots = hdr["slots"]
    if nslots != len(SPAN_NAMES):
        sys.exit(f"{nslots} slots but {len(SPAN_NAMES)} span names -- the slot "
                 f"map in persistent_kernel.cuh changed; update SPAN_NAMES")

    if args.layers:
        rows = [r for r in rows if r[2] < args.layers]
        if not rows:
            sys.exit(f"--layers {args.layers} selected no rows")

    tick_ns = hdr["tick_ns"]
    # Perfetto's unit is microseconds (float ok). Rebase on the earliest mark so
    # the trace starts near zero instead of at a raw 64-bit counter value.
    t0 = min(ts[0] for _, _, _, ts in rows if ts[0])

    def us(tick):
        return (tick - t0) * tick_ns / 1000.0

    events = []
    xcds = set()
    # Previous layer's exit per worker, for the inter-layer span. Rows arrive
    # grouped by worker and ascending in layer (the device loop emits them that
    # way), but sort explicitly rather than depend on it.
    rows.sort(key=lambda r: (r[0], r[2]))
    prev_exit = {}
    n_skipped = 0

    for w, xcd, layer, ts in rows:
        xcds.add(xcd)
        for s in range(1, nslots):
            a, b = ts[s - 1], ts[s]
            # A zero mark is a slot the worker never reached (role-dependent:
            # MoE-only ranks skip QKV and attention entirely). Non-monotonic
            # would mean the shared-clock assumption failed; count it.
            if a == 0 or b == 0:
                continue
            if b < a:
                n_skipped += 1
                continue
            name = SPAN_NAMES[s]
            ev = {
                "name": name,
                "cat": "phase",
                "ph": "X",
                "pid": xcd,
                "tid": w,
                "ts": us(a),
                "dur": us(b) - us(a),
                "args": {"layer": layer, "slot": s, "worker": w, "xcd": xcd},
            }
            if name in COLOR:
                ev["cname"] = COLOR[name]
            events.append(ev)
        # The inter-layer span, drawn from the previous layer's exit. This is
        # the gap the [PSLOTW] table folds into slot 0; showing it as a slice
        # is what makes the row of layers look continuous instead of leaving
        # unexplained whitespace between them.
        if w in prev_exit and ts[0] and prev_exit[w] <= ts[0]:
            events.append({
                "name": "inter_layer",
                "cat": "phase",
                "ph": "X",
                "pid": xcd,
                "tid": w,
                "ts": us(prev_exit[w]),
                "dur": us(ts[0]) - us(prev_exit[w]),
                "cname": COLOR["inter_layer"],
                "args": {"layer": layer, "slot": 0, "worker": w, "xcd": xcd},
            })
        if ts[nslots - 1]:
            prev_exit[w] = ts[nslots - 1]

    if n_skipped:
        print(f"WARNING: {n_skipped} spans had end < start and were dropped. "
              f"s_memrealtime is meant to be die-wide and monotonic; this "
              f"many is a real anomaly, not rounding.")

    # Process/thread names. Without these Perfetto shows bare numeric ids.
    for x in sorted(xcds):
        events.append({"name": "process_name", "ph": "M", "pid": x, "tid": 0,
                       "args": {"name": f"XCD {x}"}})
        events.append({"name": "process_sort_index", "ph": "M", "pid": x,
                       "tid": 0, "args": {"sort_index": x}})
    seen = {}
    for w, xcd, _, _ in rows:
        if (xcd, w) in seen:
            continue
        seen[(xcd, w)] = True
        events.append({"name": "thread_name", "ph": "M", "pid": xcd, "tid": w,
                       "args": {"name": f"worker {w}"}})
        events.append({"name": "thread_sort_index", "ph": "M", "pid": xcd,
                       "tid": w, "args": {"sort_index": w}})

    with open(args.out, "w") as fh:
        json.dump({"traceEvents": events,
                   "displayTimeUnit": "ns"}, fh)

    span_evs = [e for e in events if e["ph"] == "X"]
    nlayers = len({r[2] for r in rows})
    makespan = max(e["ts"] + e["dur"] for e in span_evs) - \
        min(e["ts"] for e in span_evs)
    print(f"wrote {args.out}: {len(span_evs)} slices, "
          f"{len(seen)} workers, {len(xcds)} XCDs, {nlayers} layers")
    print(f"makespan {makespan:.1f} us over {nlayers} layers "
          f"= {makespan / max(nlayers, 1):.2f} us/layer")
    print(f"open at https://ui.perfetto.dev (Open trace file)")


if __name__ == "__main__":
    main()
