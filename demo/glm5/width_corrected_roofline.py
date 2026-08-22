#!/usr/bin/env python3
"""Does GLM's PHASE WIDTH change the roofline?  Measured answer: almost not at all.

roofline.py divides every byte by a flat HBM_TBS = 5.17 TB/s.  That is only the
achievable rate when the whole GPU fetches at once, and GLM never does: the
layer is a bulk-synchronous ladder of phases running at 16, 32, 128, 168, 192
and 232 workers.  test_narrow_grid_bandwidth.hip MEASURED the achievable
aggregate at each of those widths and it is strongly sublinear at the bottom --
a 16-wide phase gets 707 GB/s, not 5170.

So the obvious worry is that roofline.py's 2.319 ms "achievable floor" is a
fantasy, and the real floor -- the one that says whether 2 ms/token is even
arithmetically possible -- is much higher.

THIS SCRIPT PRICES THAT WORRY AND IT DOES NOT SURVIVE.  The correction is worth
about 0.1 ms across the whole model, because THE BYTE-HEAVY PHASES ARE THE WIDE
ONES.  MoE W13+W2 is 68% of the layer's bytes and runs at the full 232.  The
genuinely narrow phases -- MLA decode at 16 workers, W_UK at 32 -- are
byte-trivial (74 KB and 811 KB against a 118 MB layer).  Narrowness and byte
weight are anti-correlated in this layer, which is lucky and worth knowing.

CONSEQUENCE, which is why this file exists: "widen the narrow phases to get more
bandwidth" is DEAD as a class, before anyone writes one.  The bandwidth left on
the table by every narrow phase in the layer, summed, is under 0.11 ms.  The
9.1 ms gap to the roofline is schedule, exactly as roofline.py's closing line
says -- this script is the check that closes off the bandwidth escape hatch.

No GPU run: both inputs are already measured.  Widths come from the verified
tiles/XCD table (memory: glm-per-phase-occupancy-is-the-whole-story, derived
from the [CFG] lines at NP=8 and cross-checked against the router's exact
cnt[6]/cnt_tail = 128.0).  Bandwidths come from test_narrow_grid_bandwidth.hip.
"""
from __future__ import annotations

import json
from pathlib import Path

# --------------------------------------------------------------------------- #
# MEASURED: achievable AGGREGATE GB/s vs grid width, depth 4 (the optimum at
# every width -- depth 8 is 15% worse at narrow grids).  From
# tests/standalone/test_narrow_grid_bandwidth.hip, quoted in
# glm-per-cu-roofline-denominator-is-wrong.md.  Working set is past MALL.
# --------------------------------------------------------------------------- #
BW_CURVE = [(16, 707.0), (32, 1399.0), (64, 2660.0), (96, 3800.0),
            (128, 4765.0), (232, 5373.0), (256, 5365.0)]

HBM_GBS = 5170.0          # roofline.py's flat constant, for the comparison
EP_COLLECTIVE_US = 5.1    # MEASURED, unchanged from roofline.py
WORKERS = 232
XCDS = 8


def bw_at(width: int) -> float:
    """Linear interpolation into the measured width->aggregate curve."""
    width = max(1, min(width, BW_CURVE[-1][0]))
    if width <= BW_CURVE[0][0]:
        # Below the smallest measured point, scale down linearly: at 16 wide the
        # curve is still in its linear regime (707/16 = 44.2 GB/s per CU, and
        # 32 gives 43.7), so this is interpolation, not extrapolation into fog.
        return BW_CURVE[0][1] * width / BW_CURVE[0][0]
    for (w0, b0), (w1, b1) in zip(BW_CURVE, BW_CURVE[1:]):
        if width <= w1:
            return b0 + (b1 - b0) * (width - w0) / (w1 - w0)
    return BW_CURVE[-1][1]


def main() -> int:
    c = json.loads(Path("/home/claudeuser/models/glm5-mxfp4/config.json").read_text())
    H = c["hidden_size"]
    L, dense = c["num_hidden_layers"], c["first_k_dense_replace"]
    moe_layers = L - dense
    n_heads = c["num_attention_heads"]
    q_lora, kv_lora = c["q_lora_rank"], c["kv_lora_rank"]
    qk_nope, qk_rope, v_head = c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]
    n_routed, topk = c["n_routed_experts"], c["num_experts_per_tok"]
    n_shared, moe_inter = c["n_shared_experts"], c["moe_intermediate_size"]
    R, seq = 8, 128
    aq, mq = 1.0 + 1.0 / 32, 0.5 + 1.0 / 32

    per_expert = 3 * H * moe_inter
    moe_max_b = 3.0 * per_expert * mq          # busiest rank, E[max] of 8-in-8
    shared_b = n_shared * per_expert * mq
    moe_b = moe_max_b + shared_b
    kv_b = (kv_lora + qk_rope) * 1.0 * seq

    # phase -> (bytes/layer/rank, tiles per XCD or None, note)
    # tiles/XCD from the verified table; workers = min(tiles, 29) * 8 XCDs.
    phases = [
        ("qkv_a  (q_a + kv_a)", (H * q_lora + H * (kv_lora + qk_rope)) * aq, 21, ""),
        ("q_b",                 q_lora * n_heads * (qk_nope + qk_rope) / R * aq, 16, ""),
        ("W_UK",                kv_lora * n_heads * qk_nope / R * aq, 4, "narrowest real phase"),
        ("W_UV",                kv_lora * n_heads * v_head / R * aq, 4, "assumed = W_UK width"),
        ("MLA decode (KV)",     kv_b, 2, "16 workers, 74 KB"),
        ("merge",               0.0, 16, "no weight bytes"),
        ("o_proj",              n_heads * v_head * H / R * aq, 24, ""),
        ("router",              H * n_routed * aq, 16, "128 workers, exact"),
        ("MoE W13",             moe_b * 2 / 3, 54, "oversubscribed -> 232"),
        ("MoE W2",              moe_b * 1 / 3, 108, "oversubscribed -> 232"),
    ]

    print("=" * 84)
    print("WIDTH-CORRECTED HBM FLOOR, GLM-5 744B, bs=1, NP=8, per MoE layer per rank")
    print("=" * 84)
    print(f"  {'phase':<22} {'MB':>7} {'tiles/XCD':>10} {'workers':>8} "
          f"{'GB/s':>7} {'us flat':>8} {'us wide':>8} {'penalty':>8}")

    tot_flat = tot_wide = tot_bytes = 0.0
    for name, b, tiles, note in phases:
        workers = min(tiles, 29) * XCDS
        bw = bw_at(workers)
        # bytes / (GB/s) -> us:  b / (bw * 1e9) * 1e6  ==  b / (bw * 1e3)
        t_flat = b / (HBM_GBS * 1e3)
        t_wide = b / (bw * 1e3)
        tot_flat += t_flat
        tot_wide += t_wide
        tot_bytes += b
        print(f"  {name:<22} {b/1e6:>7.2f} {tiles:>10} {workers:>8} "
              f"{bw:>7.0f} {t_flat:>8.2f} {t_wide:>8.2f} {t_wide-t_flat:>8.2f}"
              + (f"   {note}" if note else ""))

    print("-" * 84)
    print(f"  {'LAYER':<22} {tot_bytes/1e6:>7.2f} {'':>10} {'':>8} {'':>7} "
          f"{tot_flat:>8.2f} {tot_wide:>8.2f} {tot_wide-tot_flat:>8.2f}")

    ep_ms = EP_COLLECTIVE_US * moe_layers / 1e3
    flat_ms = tot_flat * moe_layers / 1e3
    wide_ms = tot_wide * moe_layers / 1e3
    print()
    print(f"  MoE layers                                  {moe_layers}")
    print(f"  HBM floor, FLAT 5.17 TB/s                   {flat_ms:.3f} ms")
    print(f"  HBM floor, WIDTH-CORRECTED                  {wide_ms:.3f} ms")
    print(f"  the width penalty                           {wide_ms-flat_ms:.3f} ms")
    print(f"  + EP collective (MEASURED 5.1 us x {moe_layers})       {ep_ms:.3f} ms")
    print(f"  = width-corrected achievable floor          {wide_ms+ep_ms:.3f} ms")
    print()

    # Where could widening possibly help?  Only where a phase is BOTH narrow and
    # carries bytes.  Price the whole class at once: give every phase the
    # 232-wide rate and see what the layer saves.
    best_us = sum(b / (bw_at(WORKERS) * 1e3) for _, b, _, _ in phases)
    best_ms = best_us * moe_layers / 1e3
    print("  THE WIDENING CLASS, priced in full:")
    print(f"    every phase magically at the 232-wide rate  {best_ms:.3f} ms")
    print(f"    i.e. the ENTIRE bandwidth gain from widening "
          f"EVERY narrow phase = {wide_ms - best_ms:.3f} ms")
    print()
    # ---------------------------------------------------------------- part 2
    # The same corrected roof, pointed at the two phases that actually carry
    # the bytes.  This is where the width correction earns its keep: it turns
    # "% of roof" from a number nobody could act on into an explanation of
    # three already-measured nulls.
    print("=" * 84)
    print("  ACHIEVED vs the WIDTH-CORRECTED roof, the two byte-heavy phases")
    print("  (times MEASURED: SP3[4] MoeW13 13.06 us/layer, SP3[6] MoeW2 13.05,")
    print("   both pop 232 -- glm-per-phase-occupancy-is-the-whole-story)")
    print()
    print(f"  {'phase':<6} {'MB':>6} {'us':>6} {'GB/s':>7} {'% roof':>7} "
          f"{'KB/tile':>8} {'rounds':>7} {'headroom ms':>12}")
    for name, mb, us, tiles in (("W13", 53.48, 13.06, 54 * XCDS),
                                ("W2", 26.74, 13.05, 108 * XCDS)):
        ach = mb * 1e6 / (us * 1e-6) / 1e9
        head = (us - mb * 1e6 / (BW_CURVE[-2][1] * 1e3)) * moe_layers / 1e3
        print(f"  {name:<6} {mb:>6.2f} {us:>6.2f} {ach:>7.0f} "
              f"{100*ach/BW_CURVE[-2][1]:>6.1f}% {mb*1e6/tiles/1e3:>8.1f} "
              f"{tiles/XCDS/29:>7.2f} {head:>12.3f}")
    print("""
  BOTH ROWS ARE ALREADY EXPLAINED, and that is the point of computing them:

  W13 at 76% of its real roof is NOT "36% of roof" and NOT "2.5x unexplained".
  Those two readings came from dividing by 41.6 GB/s/CU at an assumed live
  width of 64.  W13's verified geometry is 54 tiles/XCD over 29 workers -- it
  is a 232-WIDE phase, so its per-CU share is 23.2, not 41.6.  At 76% of roof
  there is only 0.233 ms in the whole phase, which is why MPK_MOE_PF_GROUPS
  measured null on W13 twice: a 28% standalone tile win had nothing to win.

  W2 at 38% of roof has a real 0.605 ms of BANDWIDTH gap, and the source
  already names the mechanism -- 864 tiles of 30.9 KB, each re-staging the
  whole activation into LDS and running a full 64-row atomicAdd epilogue,
  fixed cost that neither splitting nor widening divides.  IT IS STILL NOT A
  LEVER.  Three measured attempts: OPW=16 -1.34 ms, W2_KSPLIT=2 -4.12 ms,
  OPW=128 NEUTRAL.  The MoE half is barrier-bound, so shrinking the phase is
  absorbed (glm-cutting-work-in-a-phase-is-absorbed).  W2's spread being the
  one unabsorbed spread in the layer is NECESSARY for a lever, not sufficient.

  So: a bandwidth gap and a lever are different things, and this table is the
  clean worked example.  0.605 ms is genuinely on the floor of that phase and
  genuinely unreachable by any in-place change to it.""")
    print("=" * 84)
    print("  Read: narrowness and byte weight are ANTI-CORRELATED in this layer.")
    print("  MoE is 68% of the bytes and already runs at 232.  The 16- and")
    print("  32-wide phases are byte-trivial.  'Widen a phase for bandwidth' is")
    print("  dead as a CLASS -- not one instance of it can pay.  Any remaining")
    print("  win from widening a phase has to come from LATENCY/makespan, which")
    print("  is a schedule argument and must be measured on the wall.")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
