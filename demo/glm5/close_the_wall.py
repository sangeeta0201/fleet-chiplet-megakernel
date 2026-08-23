#!/usr/bin/env python3
"""Close the whole 10.619 ms/token wall against labelled components, then check
the residual against the measured HBM byte floor.

OFFLINE. No GPU run. Sources:
  * the 7-phase busy/spin table from price_busy_vs_spin.py (commit 1ef2283),
  * the two segment decompositions from price_regime_b_segments.py (2ddf4dc),
    which cover exactly the parts the 7-phase table does NOT reach,
  * roofline.py's MEASURED HBM floor (do not edit its constants).

The point of running both together: the 7-phase table accounts 6.959 of 10.619
and it is tempting to call the missing 3.66 ms "overhead". It is not. Every
microsecond of it is named below, and once it is named the conclusion that
regime B pointed at -- "the residual is arithmetic volume" -- is FALSIFIED.
"""

LAYERS = 76
INSTRUMENTED_LAYER_US = 164.947
WALL_MS = 10.619
SHIPPING_LAYER_US = WALL_MS * 1000.0 / LAYERS
K = SHIPPING_LAYER_US / INSTRUMENTED_LAYER_US
F = LAYERS * K / 1000.0  # us/layer (instrumented) -> ms/token (shipping)

# HBM roof, from demo/glm5/roofline.py.  MEASURED constants, not re-derived.
# NOTE: HBM_FLOOR_BUSIEST_MS is a whole-ITERATION floor -- all 10.01 GB/iter/rank
# including the dense prologue layers, embeddings and lm_head.  It is the right
# denominator for the WALL, and the wrong one for the tile line alone.  For the
# tile line use TILE_ROOF_MS, which is the same eight phases priced at their own
# live widths by demo/glm5/tile_roof_by_phase.py.
HBM_FLOOR_BUSIEST_MS = 1.936
EP_COLLECTIVE_FLOOR_MS = 0.383
ACHIEVABLE_FLOOR_MS = HBM_FLOOR_BUSIEST_MS + EP_COLLECTIVE_FLOOR_MS
TILE_ROOF_MS = 1.712

# (label, us/layer, class)
# class: 'tile'  = a GEMM/attention tile doing real math+bytes
#        'coll'  = a cross-rank collective (bytes over the fabric)
#        'rdv'   = rendezvous fan-out / poll period
#        'bound' = task-graph / layer-boundary bookkeeping
BUDGET = [
    # --- the 7-phase busy/spin table, split into its two halves (1ef2283) ---
    ("qkv_a tiles (busy)", 12.058, "tile"),
    ("q_b tiles (busy)", 12.336, "tile"),
    ("decode (busy)", 9.389, "tile"),
    ("merge / Phase 8 (busy)", 3.026, "tile"),
    ("o_proj tiles (busy)", 7.404, "tile"),
    ("router (busy)", 15.062, "tile"),
    ("W13 tiles (busy)", 18.660, "tile"),
    ("spin at qkv_barrier", 4.916, "rdv"),
    ("spin at qb_barrier (QB_TP cross-rank gather)", 9.034, "coll"),
    ("spin at decode_barrier", 2.331, "rdv"),
    ("spin at attn_release", 2.102, "rdv"),
    ("spin at hier_barrier", 2.394, "rdv"),
    ("spin at routing poll", 4.168, "rdv"),
    ("spin at w13_barrier", 5.219, "rdv"),
    # --- what the 7-phase table does NOT reach, from the two segments (2ddf4dc) ---
    ("EP collective (Phase 0-EP)", 15.617, "coll"),
    ("W2 tiles", 12.407, "tile"),
    ("layer boundary + refresh + dispatch", 9.414, "bound"),
    ("dispatch into attn half", 2.257, "bound"),
    ("entry_bar release fan-out", 2.259, "rdv"),
    ("w13_barrier release fan-out", 5.310, "rdv"),
]

CLASS_LABEL = {
    "tile": "tile math+bytes",
    "coll": "cross-rank collective",
    "rdv": "rendezvous",
    "bound": "task/layer boundary",
}


def main():
    print(f"wall {WALL_MS} ms/token over {LAYERS} layers = "
          f"{SHIPPING_LAYER_US:.3f} us/layer shipping")
    print(f"instrumented layer {INSTRUMENTED_LAYER_US:.3f} us   K = {K:.5f}\n")

    print(f"{'component':<46} {'us/lyr':>8} {'ms/token':>9}  class")
    print("-" * 46 + " " + "-" * 8 + " " + "-" * 9 + "  " + "-" * 22)
    tot_us = 0.0
    by_class = {}
    for label, us, cls in BUDGET:
        tot_us += us
        by_class[cls] = by_class.get(cls, 0.0) + us
        print(f"{label:<46} {us:8.3f} {us * F:9.3f}  {CLASS_LABEL[cls]}")
    print("-" * 46 + " " + "-" * 8 + " " + "-" * 9)
    print(f"{'ACCOUNTED':<46} {tot_us:8.3f} {tot_us * F:9.3f}")
    resid = WALL_MS - tot_us * F
    print(f"{'RESIDUAL (dense prologue layers, sampling, harness)':<46} "
          f"{'':>8} {resid:9.3f}\n")

    print("By class:")
    print(f"  {'class':<24} {'ms/token':>9} {'share':>7}")
    for cls in ("tile", "coll", "rdv", "bound"):
        ms = by_class[cls] * F
        print(f"  {CLASS_LABEL[cls]:<24} {ms:9.3f} {ms / WALL_MS * 100:6.1f}%")
    print(f"  {'residual':<24} {resid:9.3f} {resid / WALL_MS * 100:6.1f}%")

    tile_ms = by_class["tile"] * F
    coll_ms = by_class["coll"] * F
    rdv_ms = by_class["rdv"] * F
    bound_ms = by_class["bound"] * F

    print("\n" + "=" * 74)
    print("  AGAINST THE MEASURED ROOF (roofline.py constants, not re-derived)")
    print("=" * 74)
    print(f"  HBM floor, busiest-rank routing            {HBM_FLOOR_BUSIEST_MS:7.3f} ms")
    print(f"  + EP collective latency (measured)         {EP_COLLECTIVE_FLOOR_MS:7.3f} ms")
    print(f"  = achievable floor                         {ACHIEVABLE_FLOOR_MS:7.3f} ms")
    print()
    print(f"  tile math+bytes, MEASURED                  {tile_ms:7.3f} ms")
    print(f"  its own roof: THE SAME 8 PHASES at their")
    print(f"    own live widths (tile_roof_by_phase.py)  {TILE_ROOF_MS:7.3f} ms")
    print(f"  -> tiles run at "
          f"{TILE_ROOF_MS / tile_ms * 100:.0f}% of the byte roof, "
          f"i.e. {tile_ms / TILE_ROOF_MS:.2f}x off")
    print(f"     headroom INSIDE the tiles              "
          f"{tile_ms - TILE_ROOF_MS:7.3f} ms")
    print(f"     (do NOT divide tile_ms by {HBM_FLOOR_BUSIEST_MS} -- that is a")
    print(f"      whole-ITERATION floor and a denominator mismatch)")
    print()
    print(f"  cross-rank collective, MEASURED            {coll_ms:7.3f} ms")
    print(f"  its own roof (measured EP latency)         {EP_COLLECTIVE_FLOOR_MS:7.3f} ms")
    print(f"     headroom                               "
          f"{coll_ms - EP_COLLECTIVE_FLOOR_MS:7.3f} ms")
    print()
    print(f"  rendezvous (roof = 0)                      {rdv_ms:7.3f} ms")
    print(f"  task/layer boundary (roof = 0)             {bound_ms:7.3f} ms")
    print(f"  residual (roof = 0)                        {resid:7.3f} ms")
    print("=" * 74)

    print("""
READING IT.

1.  "The residual is arithmetic VOLUME" is FALSE.  The byte volume already
    permits {floor:.3f} ms.  Volume is not the problem; the tiles are {x:.2f}x off
    their OWN byte roof, which is an efficiency problem inside the tiles, and
    that agrees with glm-attention-tiles-are-latency-bound-not-valu-bound
    (qkv_a is 68% vmcnt) rather than with any FLOP or byte count.

2.  But tile efficiency is ALSO not directly a lever, because busy converts to
    wall at 0.27 near this operating point (MPK_ATTN_HALFK, commit 73c0afb).
    {head:.3f} ms of tile headroom x 0.27 = {conv:.3f} ms of wall -- and only if
    every tile in the layer were simultaneously taken to the byte roof.

2b. AND IT IS NOT A BANDWIDTH STORY.  tile_roof_by_phase.py resolves this
    headroom per phase: the top three rows are router 0.944, q_b 0.695 and
    decode 0.597 -- the three phases carrying the LEAST bytes (1.62, 6.22 and
    0.07 MB).  The three byte-heavy phases holding 97 of the layer's 118 MB
    (W13, W2, qkv_a) have only 1.390 ms between them, and W13's and W2's shares
    are already closed by measurement.  So the {x:.2f}x is concentrated exactly
    where bandwidth is irrelevant: it is latency and fixed per-tile cost.

3.  The three zero-roof classes -- rendezvous {rdv:.3f}, boundary {bound:.3f},
    residual {res:.3f} = {zsum:.3f} ms -- are the only components whose floor is
    actually 0.  Every one of them has been attacked and measured out
    individually (glm-regime-b-is-closed-every-edge-is-all-to-all,
    glm-layer-boundary-ablation-is-zero, and the seven barrier-mechanism nulls).

So the honest statement is neither "arithmetic volume" nor "schedule": the wall
is {wall} ms, its floor is {floor:.3f} ms, and the {gap:.3f} ms in between is
distributed across four classes with NO single class above {mx:.3f} ms and no
class that converts to wall at better than ~0.27 without deleting a whole
rendezvous -- which the legality pass just proved is impossible for all ten.
""".format(floor=ACHIEVABLE_FLOOR_MS,
           x=tile_ms / TILE_ROOF_MS,
           head=tile_ms - TILE_ROOF_MS,
           conv=(tile_ms - TILE_ROOF_MS) * 0.27,
           rdv=rdv_ms, bound=bound_ms, res=resid,
           zsum=rdv_ms + bound_ms + resid,
           wall=WALL_MS, gap=WALL_MS - ACHIEVABLE_FLOOR_MS,
           mx=max(tile_ms - TILE_ROOF_MS,
                  coll_ms - EP_COLLECTIVE_FLOOR_MS,
                  rdv_ms, bound_ms, resid)))


if __name__ == "__main__":
    main()
