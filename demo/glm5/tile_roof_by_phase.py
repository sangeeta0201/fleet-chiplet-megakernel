#!/usr/bin/env python3
"""Resolve the 3.00x tile-roof gap (close_the_wall.py, 452b186) into PER-PHASE rows.

OFFLINE. No GPU run.

close_the_wall.py found that the layer's tile math+bytes is 5.816 ms against a
1.936 ms whole-layer HBM roof -- 3.00x off, 3.880 ms of headroom, the largest
item in the budget.  That is a CLASS-level statement and it must not be
attributed to any single phase, because the flat 5.17 TB/s denominator is wrong
for a phase that does not run at 232 workers
(glm-per-cu-roofline-denominator-is-wrong) and because W13 is already known to
sit at 76% of its real roof.

This script pairs, phase by phase:
  * MEASURED busy us/layer -- the critical worker's own non-spin time, from the
    stage-stamp exit/arrival pairs (price_busy_vs_spin.py, 1ef2283), plus W2
    from the segment decomposition (price_regime_b_segments.py, 2ddf4dc);
  * the WIDTH-CORRECTED byte roof at that phase's verified live width, using the
    same measured width->bandwidth curve as width_corrected_roofline.py.

Two things it deliberately does NOT do:
  * it does not price an MFMA/FLOP roof.  Every phase here is byte- or
    latency-bound at bs=1 (glm-attention-tiles-are-latency-bound-not-valu-bound:
    qkv_a is 21% VALU / 5% MFMA / 68% vmcnt), so the byte roof is the binding
    one and a FLOP roof would be vacuous.
  * it does not convert headroom to wall.  Busy converts at ~0.27
    (MPK_ATTN_HALFK, 73c0afb).  The x0.27 column is shown but it is an upper
    bound that assumes the cut is UNIFORM across the phase's active set.
"""
from __future__ import annotations

import json
from pathlib import Path

# Measured aggregate GB/s vs grid width, depth 4.
# tests/standalone/test_narrow_grid_bandwidth.hip, same curve as
# width_corrected_roofline.py.
BW_CURVE = [(16, 707.0), (32, 1399.0), (64, 2660.0), (96, 3800.0),
            (128, 4765.0), (232, 5373.0), (256, 5365.0)]
XCDS = 8
LAYERS = 76
WALL_MS = 10.619
INSTRUMENTED_LAYER_US = 164.947
K = (WALL_MS * 1000.0 / LAYERS) / INSTRUMENTED_LAYER_US
BUSY_TO_WALL = 0.27


def bw_at(width: int) -> float:
    width = max(1, min(width, BW_CURVE[-1][0]))
    if width <= BW_CURVE[0][0]:
        return BW_CURVE[0][1] * width / BW_CURVE[0][0]
    for (w0, b0), (w1, b1) in zip(BW_CURVE, BW_CURVE[1:]):
        if width <= w1:
            return b0 + (b1 - b0) * (width - w0) / (w1 - w0)
    return BW_CURVE[-1][1]


def main() -> int:
    c = json.loads(
        Path("/home/claudeuser/models/glm5-mxfp4/config.json").read_text())
    H = c["hidden_size"]
    n_heads = c["num_attention_heads"]
    q_lora, kv_lora = c["q_lora_rank"], c["kv_lora_rank"]
    qk_nope, qk_rope = c["qk_nope_head_dim"], c["qk_rope_head_dim"]
    v_head = c["v_head_dim"]
    n_routed, moe_inter = c["n_routed_experts"], c["moe_intermediate_size"]
    n_shared = c["n_shared_experts"]
    R, seq = 8, 128
    aq, mq = 1.0 + 1.0 / 32, 0.5 + 1.0 / 32   # bf16+scale / mxfp4+scale

    per_expert = 3 * H * moe_inter
    moe_b = 3.0 * per_expert * mq + n_shared * per_expert * mq  # busiest rank
    kv_b = (kv_lora + qk_rope) * 1.0 * seq

    # INTERNAL-SPIN AUDIT.  A stage-stamp span bills everything between the two
    # stamps as "busy", including any rendezvous that lives INSIDE the phase and
    # therefore has no barrier slot of its own.  Grepped every span and its
    # callees for `while (`, `s_sleep` and MPK_WS_WAIT_BEGIN:
    #
    #   qkv_a   CONTAMINATED  attn_fused:641   XCD-local EP-fold rendezvous
    #                         (this is exactly what stamps 44/45 straddle;
    #                          they are committed but NOT YET COMPILED, so the
    #                          magnitude is unmeasured)
    #   q_b     CONTAMINATED  attn_fused:954   XCD-local W_UK rendezvous,
    #                         also unstamped and unmeasured
    #   router  CONTAMINATED  gang_rmsnorm_linear_bias_mi300.cuh:852, the
    #                         OPROJ_BARRIER spin inside the router kernel.
    #                         THIS ONE HAS A MEASUREMENT: the router region is
    #                         54% poll / 46% compute
    #                         (glm-router-region-is-a-55pct-narrow-phase).
    #   decode, merge, o_proj, W13, W2   CLEAN -- no spin in the span or in any
    #                         callee (gang_mla_decode / mla_kv_cache_update,
    #                          gang_linear_mxfp8, gang_moe_linear_mxfp4).
    #
    # compute_frac below applies ONLY where a measurement exists.  Where it does
    # not, the row keeps 1.0 and is flagged UPPER BOUND -- do not invent a
    # fraction for it.
    COMPUTE_FRAC = {"router": 0.46}
    CONTAMINATED = {"qkv_a", "q_b (+W_UK,W_UV)", "router"}

    # (label, measured busy us/lyr INSTRUMENTED, bytes/lyr/rank, tiles/XCD, note)
    #
    # busy: price_busy_vs_spin.py's critical-worker busy column, except W2 which
    # has no exit stamp pair and comes from stamp 7->8 in the segment table.
    #
    # bytes / tiles-per-XCD: width_corrected_roofline.py's verified table.  The
    # q_b row absorbs W_UK+W_UV bytes (811 KB + 1.08 MB) because the stage-stamp
    # span 18->19 does not separate them; that makes q_b's roof GENEROUS.
    PHASES = [
        ("qkv_a", 12.058,
         (H * q_lora + H * (kv_lora + qk_rope)) * aq, 21, ""),
        ("q_b (+W_UK,W_UV)", 12.336,
         q_lora * n_heads * (qk_nope + qk_rope) / R * aq
         + kv_lora * n_heads * qk_nope / R * aq
         + kv_lora * n_heads * v_head / R * aq, 16, "roof is generous"),
        ("decode", 9.389, kv_b, 2, "16 workers, 74 KB of KV"),
        ("merge / Ph8", 3.026, 0.0, 16, "NO weight bytes -> roof is 0"),
        ("o_proj", 7.404, n_heads * v_head * H / R * aq, 24, ""),
        ("router", 15.062, H * n_routed * aq, 16, "128 workers, exact"),
        ("W13", 18.660, moe_b * 2 / 3, 54, "oversubscribed -> 232"),
        ("W2", 12.407, moe_b * 1 / 3, 108, "oversubscribed -> 232"),
    ]

    print("=" * 100)
    print("  TILE ROOF BY PHASE -- measured busy vs the WIDTH-CORRECTED byte roof")
    print(f"  bs=1 NP=8, per layer per rank.  K = {K:.5f} "
          f"(instrumented {INSTRUMENTED_LAYER_US} -> shipping "
          f"{WALL_MS*1000/LAYERS:.3f} us/layer)")
    print("=" * 100)
    print(f"  {'phase':<18} {'MB':>6} {'wrk':>4} {'GB/s':>6} "
          f"{'roof us':>8} {'meas us':>8} {'x off':>6} {'% roof':>7} "
          f"{'head ms':>8} {'x0.27':>7}  spin?")
    print("  " + "-" * 104)

    tot_meas = tot_roof = tot_head = 0.0
    rows = []
    for name, busy_us, b, tiles, note in PHASES:
        workers = min(tiles, 29) * XCDS
        bw = bw_at(workers)
        roof_us = b / (bw * 1e3)
        frac = COMPUTE_FRAC.get(name, 1.0)
        meas_us = busy_us * K * frac     # shipping coordinates, spin removed
        head_ms = max(0.0, meas_us - roof_us) * LAYERS / 1e3
        tot_meas += meas_us
        tot_roof += roof_us
        tot_head += head_ms
        xoff = meas_us / roof_us if roof_us > 0 else float("inf")
        pct = 100.0 * roof_us / meas_us
        xs = "inf" if roof_us == 0 else f"{xoff:.2f}"
        if frac < 1.0:
            tag = f"x{frac:.2f} measured"
        elif name in CONTAMINATED:
            tag = "YES, UPPER BOUND"
        else:
            tag = "clean"
        print(f"  {name:<18} {b/1e6:>6.2f} {workers:>4} {bw:>6.0f} "
              f"{roof_us:>8.2f} {meas_us:>8.2f} {xs:>6} {pct:>6.1f}% "
              f"{head_ms:>8.3f} {head_ms*BUSY_TO_WALL:>7.3f}  {tag}"
              + (f"   {note}" if note else ""))
        rows.append((name, head_ms, pct, workers, b, tag))

    print("  " + "-" * 96)
    print(f"  {'LAYER (tiles)':<18} {'':>6} {'':>4} {'':>6} "
          f"{tot_roof:>8.2f} {tot_meas:>8.2f} "
          f"{tot_meas/tot_roof:>6.2f} {100*tot_roof/tot_meas:>6.1f}% "
          f"{tot_head:>8.3f} {tot_head*BUSY_TO_WALL:>7.3f}")

    # Apples-to-apples: the SAME phase set priced at the flat 5.17 TB/s.
    flat_us = sum(b / (5170.0 * 1e3) for _, _, b, _, _ in PHASES)
    print()
    print(f"  measured tile time                {tot_meas*LAYERS/1e3:.3f} ms   "
          f"(close_the_wall.py: 5.816)")
    print(f"  same phases at FLAT 5.17 TB/s     {flat_us*LAYERS/1e3:.3f} ms "
          f"-> {tot_meas/flat_us:.2f}x off")
    print(f"  same phases WIDTH-CORRECTED       {tot_roof*LAYERS/1e3:.3f} ms "
          f"-> {tot_meas/tot_roof:.2f}x off")
    print(f"  headroom                          {tot_head:.3f} ms   x0.27 = "
          f"{tot_head*BUSY_TO_WALL:.3f} ms of wall")
    print(f"""
  NOTE the direction: correcting for width makes the tile roof LOWER
  ({flat_us*LAYERS/1e3:.3f} -> {tot_roof*LAYERS/1e3:.3f} ms) and the gap BIGGER
  ({tot_meas/flat_us:.2f}x -> {tot_meas/tot_roof:.2f}x), the opposite of the
  worry width_corrected_roofline.py was written to test.  The reason is that the
  byte-heavy phases here already run at 232 and gain nothing, while the narrow
  phases lose -- but the narrow phases carry almost no bytes, so the whole
  correction is only {(flat_us-tot_roof)*LAYERS/1e3:.3f} ms.  Neither number
  changes the verdict.

  close_the_wall.py's 1.936 ms is NOT this 1.712: it is roofline.py's
  whole-ITERATION byte floor (all 10.01 GB/iter/rank, including the dense
  prologue layers, embeddings and lm_head), while this row is the eight tile
  phases only.  Comparing them directly is the mistake this line exists to
  prevent.\n""")

    rows.sort(key=lambda r: -r[1])
    print("  RANKED BY HEADROOM:")
    for name, head_ms, pct, workers, b, tag in rows:
        print(f"    {name:<18} {head_ms:>7.3f} ms  at {pct:>5.1f}% of roof, "
              f"{workers:>3} workers, {b/1e6:>6.2f} MB   [{tag}]")
    clean = [r for r in rows if r[5] == "clean"]
    print(f"\n    largest CLEAN row: {clean[0][0]} at {clean[0][1]:.3f} ms "
          f"on {clean[0][4]/1e6:.2f} MB -- pure latency, no bytes to blame")

    print("""
  READING IT.

  1. THE TABLE IS FLAT.  No row holds more than 0.695 ms of headroom, and the
     two rows above 0.5 are decode (0.597, CLEAN) and q_b (0.695, an UPPER
     BOUND).  At the 0.27 busy->wall coefficient that is 0.161 and 0.188 ms --
     at or under the 0.26 ms noise floor.  The whole tile class is 0.967 ms of
     wall and only if EVERY phase reaches its byte roof simultaneously.  There
     is no big single item inside the biggest class in the budget.

  2. IT IS NOT A BANDWIDTH STORY.  decode carries 0.07 MB and merge/Ph8 carries
     none, yet they burn 7.95 and 2.56 us; their byte roof is ~0 so "% of roof"
     is meaningless for them.  Their headroom is latency and fixed per-tile
     cost, matching glm-attention-tiles-are-latency-bound-not-valu-bound.
     Meanwhile the three byte-heavy phases holding 97 of the layer's 118 MB
     (W13, W2, qkv_a) hold only 1.390 ms between them, and W13's and W2's
     shares are already closed by measurement (MPK_MOE_PF_GROUPS null twice;
     OPW=16 -1.34, KSPLIT=2 -4.12, OPW=128 neutral).

  3. THE ROUTER ROW MOVED, AND THAT IS THE METHOD POINT.  Uncorrected it was
     the #1 row at 0.944 ms.  Its span contains the OPROJ_BARRIER spin inside
     the router kernel (gang_rmsnorm_linear_bias_mi300.cuh:852), which the
     stage-stamp span bills as "busy" because that rendezvous has no barrier
     slot of its own.  Applying the MEASURED 46% compute fraction drops it to
     0.420 ms and sixth place.  qkv_a and q_b have the same defect
     (attn_fused:641 and :954) with NO measurement to correct by, so both are
     UPPER BOUNDS -- and q_b, the nominal #1, is one of them.

  4. W13 READS 63% HERE BUT 76% IN width_corrected_roofline.py, AND THAT SCRIPT
     IS RIGHT: it uses the MEASURED subphase counter SP3[4] = 13.06 us, while
     this table's span 5->6 also contains the MoE dispatch and expert-table
     prologue.  Every row here brackets the BLOCK, not the GEMM.  Prefer a
     subphase counter wherever one exists.""")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
