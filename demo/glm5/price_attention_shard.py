#!/usr/bin/env python3
"""ATTENTION-SIDE SHARD DIFF: which of the ten rendezvous exist because of OUR
attention shard layout, and what does deleting each one cost -- BY ABLATION.

OFFLINE. No GPU run, no build.

The ruling that produced this file:
  1. name, per rendezvous, which one exists because of our attention shard
     layout (head-parallel q_b/W_UK, the QB_TP gather, the W_UK/W_UV split) and
     which would survive ANY layout.  A rendezvous that survives every layout is
     not a candidate -- drop it immediately.
  2. price each SURVIVOR from a DIRECT MEASUREMENT already on the ledger.  Not a
     roofline, not a byte scale.  If no direct measurement of that variable
     exists, mark the row UNPRICED rather than estimating it.
  3. check compile-legality FIRST, not after the price.
  4. build only if a single survivor prices >= 0.5 ms on a direct measurement
     AND compiles.

Rule 2 exists because this run priced the EP->TP MoE at ~1.4 ms by scaling
bytes against a roofline when a direct measurement of that exact variable was
already on the ledger saying 15x less (demo/glm5/MOE_TP_FALSIFIED.md).  Every
price below names its ablation and its n.

The ten rendezvous are the hier_barrier_arrive() call sites, in fire order,
identical to the list in price_xcd_narrowing.py and makespan_predictor.py.
"""

WALL_MS = 10.619
NOISE_FLOOR_MS = 0.26
BUILD_BAR_MS = 0.5

# us/layer (instrumented) -> ms/token (shipping): x 76 layers x K
F = 76 * (10.619 * 1000.0 / 76) / 164.947 / 1000.0

# ---------------------------------------------------------------------------
# THE ATTENTION SHARD LAYOUT, read off the kernel (not inferred from op names).
#
#   qkv_a   gang_mla_attn_fused_mi300.cuh:677
#           output sharded by COLUMN across XCDs: XCD x owns columns
#           [x*S, (x+1)*S) of the [q_a | latent] row.  q_a is the 2048-wide
#           q_lora rank and latent is the 512-wide kv_lora rank.  NEITHER AXIS
#           IS HEADS -- there is no head structure in qkv_a's output at all.
#
#   q_b     :430  QB_TP_HEADS = NUM_Q_HEADS / EP_WORLD_SIZE = 64/8 = 8
#   W_UK    :439  qb_head_base = EP_MY_PE * QB_TP_HEADS
#           ALREADY HEAD-PARALLEL.  Rank p owns heads [8p, 8p+8); inside a rank
#           that is one head per XCD.  The kernel comment at :413 says why:
#           "a head is the unit both stages are already blocked on, and ...
#           that keeps Phase 3b's barrier XCD-local".
#           The cross-rank all-gather of the 8 head slices is the QB_TP gather.
#
#   decode  :374  NUM_Q_GROUPS = NUM_Q_HEADS / 16 = 4;  items are
#           (q_group, kv_chunk) = 4 x 16 = 64.  The unit is a 16-head GROUP.
#
#   merge   :344  q_group = xcd_id * NUM_Q_GROUPS / 8 = xcd_id / 2 -- pair
#           aligned already.
#
#   W_UV    replicated GEMV, 4 tiles/XCD, read by all of o_proj.
#
#   o_proj  gang_oproj_router_fused_mi300.cuh:677  OPROJ_TP_COLS
#           COLUMN-sharded (N-split) across ranks + all-gather on the ep_signal
#           line.  Rank p emits hidden columns [p*768, +768) and therefore
#           CONTRACTS OVER ALL 64 HEADS.  This is the fact that forces the
#           attention tail to be all-to-all no matter how the heads are dealt.
# ---------------------------------------------------------------------------

# (name, arrival stamp, site, us/layer face value, layout_dependent, why,
#  ablation, price_ms, compiles)
#   layout_dependent: True  = exists because of OUR attention shard layout
#                     False = survives ANY layout -> dropped, not priced
#   price_ms: None = UNPRICED (no direct measurement of this variable)
RENDEZVOUS = [
    (
        "entry_bar", 9, "full_layer:711", 2.259, False,
        "MoE(L-1) -> layer L. The producer is a device-scope atomicAdd over the "
        "whole hidden row; nothing on the attention side deals it. Survives "
        "every attention layout, including TileRT's.",
        "layer-boundary ablation = 0.051 ms (glm-layer-boundary-ablation-is-zero)",
        None, None,
    ),
    (
        "qkv_barrier", 17, "attn:743", 4.916, False,
        "qkv_a's output axis is a LORA RANK (q_a 2048 | kv latent 512), not "
        "heads. No dealing of heads makes this edge local, because the consumer "
        "contracts over the whole latent. TileRT does not shard it either -- "
        "rmsnorm_projx_wqkva is REPLICATED, which removes the rendezvous by "
        "recompute, not by re-sharding.",
        "recompute priced in REGIME_B_SEGMENT_LEGALITY.md 2a: qkv_a 11.844 -> "
        "~95 us/lyr = +5.4 ms. Strictly negative.",
        None, None,
    ),
    (
        "qb_barrier", 19, "attn:1183", 9.034, True,
        "THE ONE THE RULING WAS AIMED AT. q_b/W_UK are head-parallel (8 heads "
        "per rank, 1 per XCD); the consumer decode is dealt by (q_group, "
        "kv_chunk), a 16-head unit. Producer unit != consumer unit, so the "
        "slices must be gathered -- across XCDs and, under QB_TP, across ranks.",
        "MPK_QB_SKIP_PEER_WAIT deletes the 7 peer stores + 7-peer poll outright. "
        "n=6 pooled over two independent triples, 10.498 -> 10.271",
        0.228, False,
    ),
    (
        "decode_barrier", 21, "attn:1411", 2.331, True,
        "decode is dealt by (q_group, kv_chunk) and scatters a q_group's chunks "
        "across every XCD; merge is already pair-aligned (xcd_id/2 == q_group). "
        "Only the decode map disagrees -- a pure layout artifact.",
        "MPK_GLM_MLA_PAIR_MERGE re-deals decode to match. BUILT and measured "
        "(e482f2e): +0.07 ms at 16 chunks, +0.6% at 4",
        -0.07, True,
    ),
    (
        "attn_release", 23, "full_layer:1782", 2.102, True,
        "merge output is q_group-sharded; o_proj is COLUMN-sharded and so "
        "contracts over all 64 heads. Head-local under an o_proj K-split, "
        "all-to-all under the N-split we ship.",
        "per-XCD narrowing replay of the measured arrival log: GPU-wide max "
        "minus worst-XCD max = 0.000 us at this barrier "
        "(glm-per-xcd-barrier-narrowing-is-zero-by-measurement)",
        0.000, True,
    ),
    (
        "rel_tree", 25, "full_layer:2025", 0.0, False,
        "Not a data edge -- the hierarchical release/self-heal tree of the "
        "barrier above it. A mechanism, and mechanism levers are measured out.",
        "per-XCD narrowing 0.000; seven barrier-mechanism nulls",
        None, None,
    ),
    (
        "wuv_barrier", 28, "oproj:552", 0.0, True,
        "W_UV is PER HEAD; o_proj's N-split contracts over all of them, so the "
        "replicated GEMV has to be complete GPU-wide before o_proj starts. "
        "Head-local under an o_proj K-split.",
        "MPK_WUV_IN_MERGE (e5d1ff5) hoists W_UV behind the pair-local barrier "
        "and DELETES this rendezvous outright -- ten become nine. Correctness "
        "gated 4/4. Measured -0.003 ms (n=2 paired), -0.019 on min-iter n=5",
        0.003, True,
    ),
    (
        "hier_barrier", 30, "oproj:925", 2.394, False,
        "o_proj -> router: the router reads the FULL hidden row. This is "
        "TileRT's own communication point (unproj_o_allreduce). It survives "
        "every layout -- TileRT pays it too, fused into the epilogue, and "
        "epilogue-vs-separate is measured neutral twice on the EP fold.",
        "per-XCD narrowing 0.013 ms; fold hoist + widen both neutral",
        None, None,
    ),
    (
        "routing poll", 32, "oproj:1174", 4.168, False,
        "MoE side. TopK on rank r selects experts owned by any rank; dispatch "
        "is all-to-all by definition of EP. Nothing attention deals it.",
        "MoE EP->TP falsified separately (MOE_TP_FALSIFIED.md)",
        None, None,
    ),
    (
        "w13_barrier", 6, "oproj:1595", 5.219, False,
        "MoE side. W13 -> W2 crosses experts and expert -> XCD is a routing "
        "outcome, not a static map.",
        "regime-B segment w13 -> entry_bar predicts 0.000",
        None, None,
    ),
]

# ---------------------------------------------------------------------------
# STEP 3 OF THE RULING: COMPILE LEGALITY, CHECKED BEFORE THE PRICE.
# ---------------------------------------------------------------------------
LEGALITY = [
    ("qb_barrier",
     "DOES NOT INSTANTIATE",
     "Making the q_b -> decode edge local means rank p decodes only the heads "
     "it owns. QB_TP_HEADS = 64/8 = 8 (attn:430). The decode's unit is a "
     "q_group of 16 (NUM_Q_GROUPS = NUM_Q_HEADS / 16, attn:374, a constexpr; "
     "16 is the MFMA M-tile). 8 heads is HALF a q_group -- a rank cannot form "
     "one. NUM_Q_GROUPS would have to become 8, i.e. NUM_Q_HEADS/8, which is "
     "not the tile shape the decode kernel is written against. PAIR_MERGE's "
     "own guards (attn:382) also require 8 % NUM_Q_GROUPS == 0 and "
     "XCDS_PER_GROUP | NUM_KV_CHUNKS."),
    ("attn_release + wuv_barrier",
     "COMPILES",
     "Both go XCD-local only if o_proj flips from N-split (column, all-gather "
     "on the ep_signal line) to K-split (row, all-reduce) -- TileRT's "
     "unproj_o_allreduce. o_proj has NO MFMA at all "
     "(glm-wuk-wuv-oproj-have-no-mfma), so there is no MFMA_ITERS % 4 "
     "static_assert to fail here -- unlike the W2 K-split that blocked the MoE "
     "TP shard. This one is legal to build."),
    ("decode_barrier",
     "COMPILES -- AND IS ALREADY BUILT",
     "MPK_GLM_MLA_PAIR_MERGE, attn:377, wired through "
     "persistent_kernel.py:476."),
]


def main():
    print("=" * 78)
    print("  ATTENTION-SIDE SHARD DIFF -- the ten rendezvous, attributed")
    print("=" * 78)
    print("  wall %.3f ms/token   noise floor %.2f   build bar %.2f\n"
          % (WALL_MS, NOISE_FLOOR_MS, BUILD_BAR_MS))

    print("STEP 1 -- WHICH EXIST BECAUSE OF *OUR* ATTENTION SHARD LAYOUT\n")
    print(f"  {'rendezvous':<15} {'stamp':>5} {'us/lyr':>7} {'face ms':>8}  layout-dependent?")
    print("  " + "-" * 15 + " " + "-" * 5 + " " + "-" * 7 + " " + "-" * 8 + "  " + "-" * 18)
    survivors, dropped = [], []
    for r in RENDEZVOUS:
        name, stamp, _site, us, dep, _why, _abl, _p, _c = r
        tag = "YES -- survivor" if dep else "no -- DROP"
        print(f"  {name:<15} {stamp:>5} {us:7.3f} {us * F:8.3f}  {tag}")
        (survivors if dep else dropped).append(r)
    print(f"\n  {len(survivors)} survivors, {len(dropped)} dropped on layout alone.")
    print("  NOTE: rel_tree and wuv_barrier show 0.000 us/lyr because they have no")
    print("  separate line in close_the_wall.py's BUDGET -- their spin is folded")
    print("  into the neighbouring attention-tail spans. Both are still real")
    print("  hier_barrier_arrive() sites and both are priced below by ablation, so")
    print("  the missing face value changes no verdict.\n")

    print("  Why each dropped one survives ANY layout:")
    for name, _s, _si, _us, _d, why, abl, _p, _c in dropped:
        print(f"    * {name}: {why}")
        print(f"      [ledger] {abl}")
    print()

    print("=" * 78)
    print("STEP 3 (run BEFORE the price, per the ruling) -- COMPILE LEGALITY")
    print("=" * 78)
    for what, verdict, why in LEGALITY:
        print(f"  {what}: {verdict}")
        print(f"    {why}\n")

    print("=" * 78)
    print("STEP 2 -- PRICE EACH SURVIVOR FROM A DIRECT MEASUREMENT ONLY")
    print("=" * 78)
    print(f"  {'rendezvous':<15} {'face ms':>8} {'ABLATED ms':>11} {'compiles':>9}  ablation")
    print("  " + "-" * 15 + " " + "-" * 8 + " " + "-" * 11 + " " + "-" * 9 + "  " + "-" * 20)
    best = 0.0
    unpriced = 0
    for name, _s, _si, us, _d, _why, abl, price, comp in survivors:
        if price is None:
            unpriced += 1
            pstr, cstr = "UNPRICED", "-"
        else:
            pstr = f"{price:.3f}"
            cstr = "yes" if comp else "NO"
            best = max(best, price)
        print(f"  {name:<15} {us * F:8.3f} {pstr:>11} {cstr:>9}  {abl.split('.')[0][:44]}")
    print()
    face = sum(r[3] for r in survivors) * F
    print(f"  face value of all four survivors      {face:7.3f} ms")
    print(f"  sum of their ABLATED prices           "
          f"{sum(p for *_x, p, _c in [(r[0], r[6], r[7], r[8]) for r in survivors] if p):7.3f} ms")
    print(f"  largest single ablated price          {best:7.3f} ms  (qb_barrier)")
    print(f"  UNPRICED rows                         {unpriced:7d}")
    print()
    print("  And the four CANNOT be summed anyway: glm-inter-rank-skew-is-paid-once")
    print("  measured that deleting the EP collective moves 83% of its time onto")
    print("  exactly the qb_barrier region (S19->S20: 9.35 -> 23.23 us/layer).")
    print("  The 0.228 is the SECOND cross-rank rendezvous' marginal mechanism,")
    print("  and it is only that cheap because the first one already paid the skew.")
    print()

    print("=" * 78)
    print("  VERDICT")
    print("=" * 78)
    if best >= BUILD_BAR_MS:
        print(f"  BUILD: a survivor prices {best:.3f} >= {BUILD_BAR_MS}.")
    else:
        print(f"  NO BUILD. Largest directly-measured survivor is {best:.3f} ms,")
        print(f"  under the {BUILD_BAR_MS} ms build bar AND under the "
              f"{NOISE_FLOOR_MS} ms noise floor.")
        print()
        print("  The single largest one, qb_barrier at 0.228, was ALREADY closed by")
        print("  the ablation that measured it: glm-qb-peer-wait-ceiling-is-0.335ms")
        print("  ends 'Head-sharding attention end-to-end is CLOSED', because the")
        print("  rewrite MOVES the transfer onto o_proj's existing gather instead of")
        print("  deleting it -- the realized gain is strictly LESS than 0.228 -- and")
        print("  it does not instantiate at 8 heads against a 16-head q_group.")
        print()
        print("  Per the ruling: that is the answer. Go to (a).")
    print("=" * 78)


if __name__ == "__main__":
    main()
