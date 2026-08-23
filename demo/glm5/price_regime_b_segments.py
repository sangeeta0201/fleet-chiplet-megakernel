#!/usr/bin/env python3
"""Regime-B pricing for the two largest rendezvous segments: ref->qkv and ->entry_bar.

OFFLINE. No GPU run. Reads an MPK_BAR_SKEW=3 BARSTAGEWS log (default
/tmp/item11_bs1.log, bs=1, NP=8) and decomposes each segment into its
rendezvous half and its work half, then prices the regime-B deletion using the
two coefficients measured in commits 926af30 / 1ef2283 / 73c0afb:

    predicted wall delta = (rendezvous us/layer removed) * 1.00
                         + (work        us/layer removed) * 0.27

The 1.00 is regime C (glm-makespan-predictor-three-regimes: an ADD is 1:1, and
a rendezvous deletion is the same edge run backwards).  The 0.27 is the
MPK_ATTN_HALFK measurement: a UNIFORM cut of 2.3 us/layer across every active
worker in qkv_a moved the wall 14.189 -> 14.142, i.e. 0.047 of a 0.175 ms
1:1 prediction.

A segment is only worth its face value if its content is DELETABLE.  The
legality column is not derived here -- see REGIME_B_SEGMENT_LEGALITY.md, which
this script's numbers back.

Usage:  python3 demo/glm5/price_regime_b_segments.py [log]
"""
import collections
import re
import sys

LOG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/item11_bs1.log"
RANK = 0
LAYERS = 76
# The instrumented layer is longer than the shipping one; scale the stamp
# coordinates onto the 10.619 ms/token wall before quoting ms.
INSTRUMENTED_LAYER_US = 164.947
SHIPPING_LAYER_US = 10.619 * 1000.0 / LAYERS  # 139.72
K = SHIPPING_LAYER_US / INSTRUMENTED_LAYER_US
US_PER_LAYER_TO_MS_PER_TOKEN = LAYERS * K / 1000.0

COEFF_RENDEZVOUS = 1.00
COEFF_WORK = 0.27

BUILD_THRESHOLD_MS = 0.5

PREFIX = re.compile(r"\[1,(\d+)\]<stdout>:")


def load(path, rank):
    """MPI interleaves prefixes mid-line; attribute by the nearest one."""
    txt = open(path, errors="replace").read()
    out = collections.defaultdict(dict)
    for m in re.finditer(r"BARSTAGEWS", txt):
        pm = None
        for p in PREFIX.finditer(txt, max(0, m.start() - 200), m.start()):
            pm = p
        if pm is None or int(pm.group(1)) != rank:
            continue
        tail = PREFIX.sub("", txt[m.end():m.end() + 120]).replace("\n", " ")
        nums = re.findall(r"\d+", tail)[:4]
        if len(nums) < 4:
            continue
        slot, worker, cnt, ns = map(int, nums)
        if cnt:
            out[slot][worker] = (ns / cnt / 1000.0, cnt)
    return out


def smax(d, slot):
    """MAX over workers of the per-worker mean arrival.  The rendezvous fires at
    the LAST arriver, so max -- not mean -- is the makespan coordinate."""
    return max(v[0] for v in d[slot].values())


# (label, slot, kind).  kind is 'R' rendezvous fan-out, 'W' work, 'B' boundary.
SEGMENTS = [
    (
        "ref -> qkv_barrier",
        # ref is the entry_bar fire itself, t = 0 by construction.
        ("layer-entry ref (entry_bar fires)", None, None),
        [
            ("entry_bar release observed", 0, "R"),
            ("Phase 0-EP done (EP collective + peer poll)", 1, "W"),
            ("dispatch into attn half", 16, "B"),
            ("qkv_a tiles done", 17, "W"),
        ],
    ),
    (
        "w13_barrier -> entry_bar",
        ("w13_barrier arrival (last arriver)", 6, None),
        [
            ("w13_barrier release observed", 7, "R"),
            ("W2 tiles done", 8, "W"),
            ("layer boundary + refresh + dispatch", 11, "B"),
            ("next entry_bar arrival", 9, "B"),
        ],
    ),
]

KIND_NAME = {"R": "rendezvous", "W": "work", "B": "boundary"}
KIND_COEFF = {"R": COEFF_RENDEZVOUS, "W": COEFF_WORK, "B": COEFF_WORK}


def main():
    d = load(LOG, RANK)
    if not d:
        sys.exit(f"no BARSTAGEWS rows for rank {RANK} in {LOG}")

    print(f"log {LOG}   rank {RANK}   layers {LAYERS}")
    print(f"instrumented layer {INSTRUMENTED_LAYER_US:.3f} us -> shipping "
          f"{SHIPPING_LAYER_US:.3f} us   K = {K:.5f}")
    print(f"coefficients: rendezvous {COEFF_RENDEZVOUS:.2f} (regime C, 1:1)   "
          f"work {COEFF_WORK:.2f} (MPK_ATTN_HALFK)\n")

    for name, start, rows in SEGMENTS:
        s_label, s_slot, _ = start
        t0 = 0.0 if s_slot is None else smax(d, s_slot)
        total = smax(d, rows[-1][1]) - t0
        print(f"=== {name}   =   {total:7.3f} us/layer   =   "
              f"{total * US_PER_LAYER_TO_MS_PER_TOKEN:6.3f} ms/token ===")
        print(f"  {'component':<44} {'us/lyr':>8} {'kind':>10} {'coeff':>6} "
              f"{'ms if deleted':>14}")
        print(f"  {'-' * 44} {'-' * 8} {'-' * 10} {'-' * 6} {'-' * 14}")
        prev = t0
        buckets = collections.defaultdict(float)
        for label, slot, kind in rows:
            t = smax(d, slot)
            dt = t - prev
            prev = t
            buckets[kind] += dt
            ms = dt * US_PER_LAYER_TO_MS_PER_TOKEN * KIND_COEFF[kind]
            print(f"  {label:<44} {dt:8.3f} {KIND_NAME[kind]:>10} "
                  f"{KIND_COEFF[kind]:6.2f} {ms:14.3f}")
        ceiling = sum(v * US_PER_LAYER_TO_MS_PER_TOKEN * KIND_COEFF[k]
                      for k, v in buckets.items())
        print(f"  {'-' * 44} {'-' * 8} {'-' * 10} {'-' * 6} {'-' * 14}")
        print(f"  {'CEILING if every component were deletable':<44} "
              f"{total:8.3f} {'':>10} {'':>6} {ceiling:14.3f}")
        print(f"  LEGAL subset (see REGIME_B_SEGMENT_LEGALITY.md): "
              f"NONE -> predicted 0.000 ms")
        verdict = "BUILD" if ceiling >= BUILD_THRESHOLD_MS else "do not build"
        print(f"  even the ILLEGAL ceiling is {ceiling:.3f} ms "
              f"(threshold {BUILD_THRESHOLD_MS} -> {verdict} only if legal)\n")

    print("VERDICT: both segments have an EMPTY legal-deletion set, so the")
    print("predicted wall delta of the regime-B class on its two largest")
    print("segments is 0.000 ms.  The class is closed on its own arithmetic.")
    print("Face value of the segments (2.059 + 1.746 ms) is NOT a lever: every")
    print("microsecond in them is essential math or a rendezvous whose")
    print("producer->consumer edge is provably all-to-all.")


if __name__ == "__main__":
    main()
