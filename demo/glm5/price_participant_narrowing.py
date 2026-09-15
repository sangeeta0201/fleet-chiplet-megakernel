#!/usr/bin/env python3
"""CEILING ON GPT-OSS-STYLE PARTICIPANT NARROWING AT GLM'S attn_release.

OFFLINE. No GPU run, no build. Reads the same MPK_BAR_SKEW=3 arrival log
price_xcd_narrowing.py and makespan_predictor.py read.

THE CHANGE BEING PRICED
-----------------------
gpt-oss's fused layer joins its QKV barrier only with the workers that produce
or consume that layer's QKV output; the rest skip it and run ahead
(gang_full_layer_fused_mi300.cuh:1141-1195). The GLM analogue is to join
`attn_release` only with the workers that produce or consume attn_out, and let
the others fall straight through into the MoE half.

This is NOT the per-XCD narrowing price_xcd_narrowing.py computes. That one
keeps every worker in the barrier and shrinks its SCOPE; this one keeps the
scope GPU-wide and shrinks its POPULATION. Different ceilings, priced apart.

WHY THE BOUND IS COMPUTED WITHOUT A WORKER -> TILE MAP
-----------------------------------------------------
`w` in a BARSTAGEWS row is `blockIdx.x` (mpk_atoms.cuh:1415), not `tile_idx`,
so the `xcd_rank = w % 29` identity MAKESPAN_RULE.md uses is an assumption
about the dispatch, not something the log proves. The check that it is not
proven: the 48 workers absent from `hier_barrier` are scattered (0..7, then
10, 12, 21, 27, 30 ...) rather than the clean `w % 29 >= 24` residue class the
identity predicts.

So the bound here is deliberately map-free and generous: drop the K workers
that arrive LATEST. No real participant set can do better than that, because
any other set of size K leaves a later worker behind for the barrier to wait
on. If the oracle bound is under the noise floor, no map can rescue the lever
and the map never has to be settled. The residue-class reading is printed too,
as a cross-check, for both plausible dispatch widths.

WHICH BARRIERS A SKIPPER COULD SKIP
-----------------------------------
A worker that owns no o_proj tile owns no W_UV tile and no router tile either
(4/16/16 tiles per XCD against a 29-wide dispatch), so its first genuine data
dependency after the merge is the routing poll -- the MoE W13 weights it needs
are chosen by a TopK it does not compute. The maximal legal skip-ahead is
therefore attn_release + rel_tree + wuv_barrier + hier_barrier, and that whole
set is priced below alongside attn_release alone.
"""
import collections
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from price_xcd_narrowing import (BARRIERS, LOG, N_LAYERS, WALL_MS,  # noqa
                                 fire_order, load, segments)

TARGETS_ONE = ["attn_release"]
TARGETS_ALL = ["attn_release", "rel_tree", "wuv_barrier", "hier_barrier"]
NOISE_FLOOR_MS = 0.26
# MPK_PRINT_GEOMETRY=1 at the shipped config. o_proj is the widest consumer of
# attn_out, so it sets the participant set.
OPROJ_TILES_PER_XCD = 24


def simulate(names, D, skip_at):
    """Layer span when barrier `b` is joined only by the workers skip_at[b]
    does NOT name. Skippers never block and carry their own arrival forward.
    """
    t = collections.defaultdict(float)
    fires = {}
    for i, name in enumerate(names):
        arr = {w: t[w] + d for w, d in D[i].items()}
        skip = skip_at.get(name, frozenset())
        part = [a for w, a in arr.items() if w not in skip]
        f = max(part) if part else max(arr.values())
        for w, a in arr.items():
            t[w] = a if w in skip else f
        fires[name] = f
    return max(t.values()), fires


def main():
    R = load(LOG)
    names, D, order = segments(R)
    span = order[-1][0]
    k = (WALL_MS / N_LAYERS * 1e3) / span
    slot = {n: s for s, n, _ in BARRIERS}[TARGETS_ONE[0]]
    arr = R[slot]
    pop = len(arr)

    def ms(us):
        return us * N_LAYERS * k / 1000.0

    print("=" * 88)
    print("  PARTICIPANT NARROWING AT attn_release -- offline ceiling")
    print("=" * 88)
    print(f"  log {LOG}   {pop} workers   scale {k:.3f} (instrumented -> wall)")
    print()

    mx = max(arr.values())
    ranked = sorted(arr.values(), reverse=True)
    print("--- the top of the arrival distribution at attn_release ---")
    print(f"  max {mx:.3f} us; the barrier fires here.")
    for n in (1, 8, 16, 32, 40, 48, 64, 96, 128):
        if n < pop:
            print(f"  drop the {n:>3} latest -> new max {ranked[n]:8.3f} us"
                  f"   gains {mx - ranked[n]:6.3f} us/layer"
                  f"   = {ms(mx - ranked[n]):.4f} ms/token")
    print()

    base, base_fires = simulate(names, D, {})
    print("--- identity check ---")
    print(f"  measured layer span {span:.3f} us; simulated {base:.3f} us; "
          f"residual {abs(base - span):.6f} "
          f"{'OK' if abs(base - span) < 1e-6 else '*** MODEL BROKEN ***'}")
    print()

    late = [w for w, _a in sorted(arr.items(), key=lambda kv: -kv[1])]
    print("--- ORACLE BOUND: the skippers are the K LATEST arrivers ---")
    print("  No real participant set beats this. Two skip sets:")
    print("    [1] attn_release only")
    print("    [2] attn_release + rel_tree + wuv_barrier + hier_barrier,")
    print("        i.e. a skipper runs to the routing poll, its first real")
    print("        data dependency.")
    print()
    print(f"{'K skipped':>10}{'span [1]':>11}{'saved':>9}{'ms [1]':>9}"
          f"{'span [2]':>11}{'saved':>9}{'ms [2]':>9}")
    for K in (8, 16, 32, 40, 48, 64, 96, 128):
        if K >= pop:
            continue
        s = frozenset(late[:K])
        s1, _ = simulate(names, D, {n: s for n in TARGETS_ONE})
        s2, _ = simulate(names, D, {n: s for n in TARGETS_ALL})
        print(f"{K:>10}{s1:>11.3f}{base - s1:>9.3f}{ms(base - s1):>9.4f}"
              f"{s2:>11.3f}{base - s2:>9.3f}{ms(base - s2):>9.4f}")
    print()

    print("--- CROSS-CHECK: the residue-class reading, both dispatch widths ---")
    print("  (skippers = xcd_rank >= oproj_tiles_per_xcd, which is what the")
    print("   kernel would actually compute; shown for completeness because")
    print("   the log cannot confirm which width the blockIdx maps through)")
    print(f"{'width':>7}{'skippers':>10}{'span [1]':>11}{'saved':>9}"
          f"{'span [2]':>11}{'saved':>9}")
    for wpx in (29, 30):
        s = frozenset(w for w in arr if (w % wpx) >= OPROJ_TILES_PER_XCD)
        s1, _ = simulate(names, D, {n: s for n in TARGETS_ONE})
        s2, _ = simulate(names, D, {n: s for n in TARGETS_ALL})
        print(f"{wpx:>7}{len(s):>10}{s1:>11.3f}{base - s1:>9.3f}"
              f"{s2:>11.3f}{base - s2:>9.3f}")
    print()

    best_us = max(base - simulate(names, D,
                                  {n: frozenset(late[:K]) for n in TARGETS_ALL})[0]
                  for K in (8, 16, 32, 40, 48))
    print("=" * 88)
    print(f"  Largest saving at any realistic skipper count (K <= 48), with the")
    print(f"  most favourable possible skipper set and all four barriers")
    print(f"  skipped: {best_us:.3f} us/layer = {ms(best_us):.4f} ms/token.")
    if ms(best_us) >= NOISE_FLOOR_MS:
        print(f"  >= the {NOISE_FLOOR_MS} ms noise floor. Worth building.")
    else:
        print(f"  Under the {NOISE_FLOOR_MS} ms wall noise floor: the A/B could")
        print("  not distinguish a perfect implementation from nothing.")
    print("=" * 88)


if __name__ == "__main__":
    main()
