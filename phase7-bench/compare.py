#!/usr/bin/env python3
"""Compare Phase 7 sub-phase timings between two runs.

Parses the [OPROJ_INNER] lines emitted under MPK_OPROJ_INNER_TIMING (with the
slicewait probe applied) and prints per-bucket medians and ratios.

usage: compare.py nps1.log nps2.log
"""
import re
import statistics as st
import sys

PAT = re.compile(
    r"slicewait=([\d.]+) mfma=([\d.]+) bar=([\d.]+) "
    r"rmsnorm_router=([\d.]+) topk=([\d.]+) total=([\d.]+)"
)

BUCKETS = [
    "slicewait (cross-XCD attn)",
    "mfma (O-proj GEMM)",
    "bar (hier. barrier)",
    "rmsnorm+router",
    "topk",
    "TOTAL Phase 7",
]

# Measured on the instrumented 36-layer model run, same buckets and units.
MODEL_REF = [
    (0.84, 23.04),
    (1.56, 5.92),
    (2.96, 11.12),
    (1.64, 2.56),
    (0.80, 1.28),
    (7.96, 57.88),
]

WARMUP_FRACTION = 0.3


def load(path):
    rows = []
    for line in open(path, errors="replace"):
        m = PAT.search(line)
        if m:
            rows.append([float(x) for x in m.groups()])
    if not rows:
        sys.exit(
            "no [OPROJ_INNER] samples in %s.\n"
            "Either the build lacks -DMPK_OPROJ_INNER_TIMING, or the slicewait "
            "probe was not applied (then the line has no slicewait= field)." % path
        )
    return rows[int(len(rows) * WARMUP_FRACTION):]


def pct(sorted_vals, p):
    return sorted_vals[min(int(p * len(sorted_vals)), len(sorted_vals) - 1)]


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a, b = load(sys.argv[1]), load(sys.argv[2])

    print("Phase 7 sub-phases, us per layer, medians over %d / %d samples"
          % (len(a), len(b)))
    print("(warmup: first %d%% of samples discarded)\n"
          % int(WARMUP_FRACTION * 100))
    print("%-27s %8s %8s %8s   | %8s %8s %8s"
          % ("bucket", "run1", "run2", "ratio", "model1", "model2", "ratio"))
    print("-" * 84)

    for i, name in enumerate(BUCKETS):
        m1 = st.median([r[i] for r in a])
        m2 = st.median([r[i] for r in b])
        r1, r2 = MODEL_REF[i]
        print("%-27s %8.2f %8.2f %7.1fx   | %8.2f %8.2f %7.1fx"
              % (name, m1, m2, (m2 / m1 if m1 else 0), r1, r2, r2 / r1))
    print("-" * 84)

    for label, rows in (("run1", a), ("run2", b)):
        w = sorted(r[0] for r in rows)
        print("\n%s slicewait spread  p10 %.2f  p50 %.2f  p90 %.2f  max %.2f"
              % (label, pct(w, 0.10), pct(w, 0.50), pct(w, 0.90), w[-1]))

    # A failed mode switch is the most common way to get a meaningless result:
    # set_mode.sh aborts, the run proceeds in the previous mode, and every
    # ratio lands at 1.0x.
    slice_ratio = (st.median([r[0] for r in b])
                   / max(st.median([r[0] for r in a]), 1e-9))
    if 0.8 < slice_ratio < 1.25:
        print("\nWARNING: slicewait ratio is %.2fx. If these logs were meant to "
              "be different\nmemory modes, the switch probably failed -- check "
              "for 'ABORT: amdgpu held'\nand confirm current_memory_partition "
              "for each arm." % slice_ratio)


if __name__ == "__main__":
    main()
