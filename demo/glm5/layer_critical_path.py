#!/usr/bin/env python3
"""The whole layer as one critical path: every segment, sized, from one log.

Offline. Reads a `MPK_BAR_SKEW=3` log. No GPU run.

WHY THIS EXISTS.  price_busy_vs_spin.py answers "is the last arriver busy or
waiting" for seven phases and is the right tool for that question, but it
stops at W13 and covers 100.084 us of a 164.947 us layer. The 64.9 us it does
not reach is where the ledger's open item 2 lives, and the two biggest pieces
of it -- the W2 phase and the layer boundary -- had never been sized. This
script walks the ENTIRE layer so the segments sum to the layer, which is the
only way to know that a region is missing rather than merely unlisted.

THE POPULATION TRAP, WHICH THIS SCRIPT ENFORCES RATHER THAN DOCUMENTS.
Stamps do not all fire on the same set of layers. The counts in a current log:

    slots 5,6,7,8,12,13,14, 20,22,26..32   cnt 38317   MoE (fused) layers only
    slots 9,10,11                          cnt 38827   every layer, prologue too
    slots 0,3                              cnt 38836
    slots 2,16,17                          cnt 38322
    slots 18,19,21,23,24,25                cnt 38321

A difference of two means over DIFFERENT populations is not a duration, and
the error is not small or random: the extra 510 samples in the 38827 group are
the dense prologue layers, where the fused MoE task does not run at all, so
they enter the mean with a completely different value. Subtracting S14 (38317)
from S11 (38827) yields 7.04 us of "layer boundary" that is largely that
artifact. Every difference below asserts equal counts and REFUSES otherwise,
which is what turns the unmeasurable regions into an explicit list instead of
a plausible number.

WHAT IT FINDS, on /tmp/item11_bs1.log (rank 0, 10.619 ms baseline):

  * The two 232-wide MoE phases -- W13 and W2, the ONLY phases every worker
    runs -- carry 30.8 us/layer of critical-worker busy time = 1.98 ms/token.
    That is the ceiling on any tile-interior lever that converts 1:1, because
    a cut on a narrower phase leaves the workers that skip it arriving exactly
    when they always did (see MAKESPAN_RULE.md, table A).

  * The layer boundary, on matched populations, is 4.474 us/layer =
    0.288 ms/token, and it splits into three near-equal thirds: the return
    path and fence (1.96), the pointer-table refresh issue (1.73), and the
    block join (0.78). It is not the tail's bulk.

  * 31.70 us/layer -- 2.040 ms/token, 19% of the layer -- is NOT MEASURABLE
    from these stamps. It sits between W2's arrival and the next layer's
    entry, and the only stamps in it (9, 10, 11) are on the every-layer
    population, so nothing in this log can be differenced against them.
    That is the single largest unattributed block left on the board, it is
    bigger than either MoE phase, and closing it needs one stamp that fires
    on the fused layers only -- not a cleverer reading of this log.

The two together are the whole strategy: 1.98 ms is the most any tile-interior
work can return, and 2.04 ms is sitting in a region nobody has instrumented.
"""

import collections
import re
import sys

LOG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/item11_bs1.log"
RANK = 0
LAYERS = 76
# Instrumented span 164.947 us vs a 139.72 us uninstrumented layer. Scale any
# per-layer figure by this before quoting ms/token, or the instrument's own
# overhead gets credited to the lever.
LAYER_US = 164.947
K = 139.72 / LAYER_US

LINE = re.compile(r"^\[1,(\d+)\][^:]*:BARSTAGEWS (\d+) (\d+) (\d+) (\d+)\s*$")


def load(path, rank):
    mean = collections.defaultdict(dict)
    cnt = collections.defaultdict(dict)
    with open(path, errors="ignore") as fh:
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


def main():
    mean, cnt = load(LOG, RANK)
    if not mean:
        sys.exit(f"no BARSTAGEWS lines for rank {RANK} in {LOG}")

    def pop(s):
        return sorted(cnt[s].values())[0]

    def avg(s):
        return sum(mean[s].values()) / len(mean[s])

    def mx(s):
        return max(mean[s].values())

    print(f"log {LOG}   rank {RANK}   layer {LAYER_US:.3f} us   k={K:.4f}\n")
    print("SLOT INVENTORY  (spread < 1 us => a barrier EXIT; else an ARRIVAL)")
    print(f"  {'slot':>4} {'kind':>6} {'pop':>7} {'mean us':>9} {'spread':>7} {'nw':>4}")
    for s in sorted(mean):
        v = list(mean[s].values())
        sp = max(v) - min(v)
        print(f"  {s:>4} {'EXIT' if sp < 1.0 else 'arriv':>6} {pop(s):>7} "
              f"{avg(s):>9.3f} {sp:>7.3f} {len(v):>4}")

    # Populations must match, but "match" cannot mean bit-equal: adjacent
    # stamps routinely differ by one or two samples because a run is cut mid
    # layer, and refusing on that hides nine real segments to guard against
    # nothing. The trap being guarded is structural -- the 38317 (fused-layer)
    # group against the 38827 (every-layer) group, a gap of 510 -- so the
    # tolerance sits two orders of magnitude below it and one above the noise.
    TOL = 0.001

    def same_pop(a, b):
        pa, pb = pop(a), pop(b)
        return abs(pa - pb) <= TOL * max(pa, pb)

    # (label, from, to, arrival-slot or None)
    segs = [
        ("qkv_a",            16, 18, 17),
        ("q_b",              18, 20, 19),
        ("MLA decode",       20, 22, 21),
        ("merge / Phase 8",  22, 26, 23),
        ("pre-o_proj",       26, 29, 28),
        ("o_proj",           29, 31, 30),
        ("router",           31,  5, 32),
        ("W13",               5,  7,  6),
        # W2's release is not stamped, so `b` IS the arrival. The segment then
        # ends when the LAST worker arrives, not at the mean -- using the mean
        # for the span while the max sets busy is what makes spin come out
        # negative, which is the sibling tool's assert firing for a real
        # reason. Spin here is unmeasured, not zero.
        ("W2 (no exit stamp)", 7,  8,  8),
        ("bdry: ret+fence",   8, 12, None),
        ("bdry: ptr refresh",12, 13, None),
        ("bdry: block join", 13, 14, None),
    ]

    print("\nCRITICAL PATH  (a segment is skipped when its two stamps have "
          "different populations)")
    print(f"  {'segment':22s} {'span us':>8} {'busy':>8} {'spin':>7} "
          f"{'busy%':>6} {'ms/token':>9}")
    skipped = []
    covered = 0.0
    for name, a, b, arr in segs:
        if not same_pop(a, b):
            skipped.append((name, a, b, pop(a), pop(b)))
            continue
        no_exit = arr is not None and arr == b
        span = (mx(b) if no_exit else avg(b)) - avg(a)
        covered += span
        if arr is not None and same_pop(arr, a):
            busy = mx(arr) - avg(a)
            bt = f"{busy:8.3f}"
            if no_exit:
                st, bp = f"{'n/s':>7}", f"{'-':>6}"
            else:
                spin = span - busy
                assert spin >= -1e-9, f"{name}: negative spin {spin:.3f}"
                st, bp = f"{spin:7.3f}", f"{100.0 * busy / span:5.1f}%"
        else:
            bt, st, bp = f"{'-':>8}", f"{'-':>7}", f"{'-':>6}"
        print(f"  {name:22s} {span:8.3f} {bt} {st} {bp} "
              f"{span * K * LAYERS / 1000:9.3f}")

    print(f"  {'-' * 64}")
    print(f"  {'COVERED':22s} {covered:8.3f} {'':8s} {'':7s} {'':6s} "
          f"{covered * K * LAYERS / 1000:9.3f}")
    gap = LAYER_US - covered
    print(f"  {'NOT MEASURABLE':22s} {gap:8.3f} {'':8s} {'':7s} {'':6s} "
          f"{gap * K * LAYERS / 1000:9.3f}   <-- the open block")

    if skipped:
        print("\n  segments refused for population mismatch:")
        for name, a, b, pa, pb in skipped:
            print(f"    {name:22s} slot {a} pop {pa} vs slot {b} pop {pb}")
        print("  These need a stamp that fires on the fused layers only.")

    print("\nTHE 1:1 CEILING -- W13 and W2 are the only 232-wide phases, so "
          "they are the\nonly ones where a tile cut reaches every worker "
          "including whichever sets the max.")
    tot = 0.0
    for nm, a, arr in (("W13", 5, 6), ("W2", 7, 8)):
        busy = mx(arr) - avg(a)
        tot += busy
        print(f"  {nm:4s} critical-worker busy {busy:6.3f} us/layer = "
              f"{busy * K * LAYERS / 1000:.3f} ms/token")
    print(f"  {'both':4s} {' ' * 21}{tot:6.3f} us/layer = "
          f"{tot * K * LAYERS / 1000:.3f} ms/token")
    print("  Making BOTH instantaneous is worth that and no more. Every other\n"
          "  phase is narrower than 232, so its non-participants set the max\n"
          "  and a cut there converts at ~0.3 (MPK_ATTN_HALFK: 0.175 ms of\n"
          "  uniform cut across the active set moved the wall 0.047).")


if __name__ == "__main__":
    main()
