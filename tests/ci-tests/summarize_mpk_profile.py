#!/usr/bin/env python3
"""Aggregate an MPK profiler dump into a per-task-type latency breakdown.

`demo/*/demo.py --profiling` saves the raw profiler buffer (`profile_output.pt`
for glm5). Each task contributes three entries -- FETCHED, BEGIN, END -- so the
buffer splits cleanly into the two numbers that matter when chasing decode
latency:

    dep_wait = BEGIN - FETCHED   worker had the task but was blocked on inputs
    compute  = END   - BEGIN     the kernel itself

A task graph that is a linear chain hides most of its cost in `dep_wait`: the
workers are awake and spinning, and the critical path is the *sum* of the
per-task compute times rather than their max. So a breakdown where dep_wait
dominates points at task count / fusion, not at any one kernel being slow.

Note the wall-clock the demo prints under --profiling is meaningless (the
profiler writes serialise the workers); only the relative numbers below are.

Usage:
    python3 tests/ci-tests/summarize_mpk_profile.py demo/glm5/profile_output.pt
    python3 tests/ci-tests/summarize_mpk_profile.py <dump> --iters 8 --top 25
"""
import argparse
import os
import re
import sys

import numpy as np
import torch

# Matches the shifts in include/mirage/persistent_kernel/profiler.h.
EVENT_NO_SHIFT = 19
BLOCK_GROUP_IDX_SHIFT = 11
EVENT_IDX_SHIFT = 2
EVENT_BEGIN, EVENT_END, EVENT_INSTANT, EVENT_FETCHED = 0x0, 0x1, 0x2, 0x3

# get_timestamp() already scales s_memrealtime by 10, so the unit is ns.
DEFAULT_TICKS_PER_US = 1000.0
WRAP = 1 << 32

HEADER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "include", "mirage", "persistent_kernel", "runtime_header.h",
)


def load_task_names(header_path=HEADER):
    """Parse `TASK_* = <int>` out of the runtime header so this never goes stale."""
    names = {}
    try:
        with open(header_path) as f:
            for line in f:
                m = re.match(r"\s*(TASK_[A-Z0-9_]+)\s*=\s*(\d+)\s*,", line)
                if m:
                    names[int(m.group(2))] = m.group(1)
    except OSError as e:
        print(f"warning: could not read {header_path}: {e}", file=sys.stderr)
    return names


def paired_deltas(key_a, ts_a, key_b, ts_b):
    """Match two event streams on their (block_group, task_type, event_no) key.

    Returns the matched keys and (ts_b - ts_a), unwrapped across the 32-bit
    timestamp rollover.
    """
    common, ia, ib = np.intersect1d(key_a, key_b, assume_unique=False,
                                    return_indices=True)
    delta = (ts_b[ib].astype(np.int64) - ts_a[ia].astype(np.int64)) % WRAP
    return common, delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", help="profile_output.pt saved by demo.py --profiling")
    ap.add_argument("--iters", type=int, default=1,
                    help="decode steps in the dump (MPK_PROFILING_NUM_ITERS)")
    ap.add_argument("--top", type=int, default=30, help="rows to print")
    ap.add_argument("--ticks-per-us", type=float, default=DEFAULT_TICKS_PER_US)
    args = ap.parse_args()

    buf = torch.load(args.dump, map_location="cpu")
    num_blocks, num_groups = (int(x) for x in buf[:1].view(dtype=torch.int32))
    words = buf[1:].view(dtype=torch.uint32).numpy()
    tags, times = words[0::2], words[1::2]

    nz = tags != 0
    tags, times = tags[nz].astype(np.uint32), times[nz].astype(np.uint32)
    print(f"blocks={num_blocks} groups={num_groups} events={len(tags)}")

    event_no = (tags >> EVENT_NO_SHIFT).astype(np.int64)
    block_group = ((tags >> BLOCK_GROUP_IDX_SHIFT) & 0xFF).astype(np.int64)
    task_type = ((tags >> EVENT_IDX_SHIFT) & 0x1FF).astype(np.int64)
    kind = (tags & 0x3).astype(np.int64)

    # Unique per (worker slot, task type, occurrence) -- enough to pair events.
    key = (block_group << 22) | (task_type << 13) | event_no

    sel = {k: (key[kind == k], times[kind == k], task_type[kind == k])
           for k in (EVENT_FETCHED, EVENT_BEGIN, EVENT_END)}

    ntypes = 512
    k_b, t_b, _ = sel[EVENT_BEGIN]
    k_e, t_e, _ = sel[EVENT_END]
    k_f, t_f, _ = sel[EVENT_FETCHED]

    common_be, d_compute = paired_deltas(k_b, t_b, k_e, t_e)
    common_fb, d_depwait = paired_deltas(k_f, t_f, k_b, t_b)

    tt_be = (common_be >> 13) & 0x1FF
    tt_fb = (common_fb >> 13) & 0x1FF

    compute = np.bincount(tt_be, weights=d_compute, minlength=ntypes)
    counts = np.bincount(tt_be, minlength=ntypes)
    depwait = np.bincount(tt_fb, weights=d_depwait, minlength=ntypes)

    if counts.sum() == 0:
        print("no complete BEGIN/END pairs -- was the build made with "
              "--profiling and the buffer large enough?")
        return 1

    names = load_task_names()
    tpu, it = args.ticks_per_us, max(args.iters, 1)

    order = np.argsort(-compute)
    total_compute = compute.sum() / tpu / it
    total_depwait = depwait.sum() / tpu / it

    print(f"\n{'task type':<52}{'n/iter':>8}{'compute us':>12}"
          f"{'us/call':>10}{'depwait us':>12}")
    print("-" * 94)
    shown = 0
    for t in order:
        if counts[t] == 0:
            continue
        n = counts[t] / it
        c = compute[t] / tpu / it
        print(f"{names.get(int(t), f'<{t}>'):<52}{n:>8.0f}{c:>12.1f}"
              f"{c / max(n, 1):>10.2f}{depwait[t] / tpu / it:>12.1f}")
        shown += 1
        if shown >= args.top:
            break
    print("-" * 94)
    print(f"{'TOTAL (summed over workers, per iter)':<52}"
          f"{counts.sum() / it:>8.0f}{total_compute:>12.1f}"
          f"{'':>10}{total_depwait:>12.1f}")

    # Critical path: the megakernel is one long linear chain, so the useful
    # wall-clock proxy is the span from first BEGIN to last END.
    lo, hi = times.min(), times.max()
    wall = ((int(hi) - int(lo)) % WRAP) / tpu / it
    print(f"\nwall-clock span in dump : {wall:.1f} us/iter")
    print(f"summed compute / wall   : {total_compute / max(wall, 1e-9):.2f}x "
          f"(mean busy workers; {num_blocks} exist)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
