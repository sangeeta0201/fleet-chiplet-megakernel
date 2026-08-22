#!/usr/bin/env python3
"""ITEM 1 GO-NO-GO: is a MoE row fold worth building?

The second LIVE decode row costs 5.367 ms (b980171: 2.186 build geometry +
3.181 row activation).  ~15% of that is MoE.  The proposed lever is a "row
fold": when both decode rows route to the SAME expert, issue ONE tile that
does both rows instead of two tiles that each fetch the same weight slab.

V IS THE EXACT CEILING ON THAT LEVER, and it is a pure counting argument.
From gang_moe_linear_mxfp8_mi300.cuh:358-410 a tile is an (expert, row) pair:

    num_activated_experts = d_mask[NUM_EXPERTS]      <- UNION size U
    e   = d_mask[global_tile / TILES_PER_EXPERT]
    tok = (global_tile % TILES_PER_EXPERT) / WGS
    if (d_routing[e * BATCH_SIZE + tok] == 0) return false;

Two live rows always yield 2 x TOPK = 16 live (expert,row) pairs regardless of
overlap -- overlap does NOT cut live tiles, it cuts DUPLICATE WEIGHT-SLAB
FETCHES.  Distinct slabs needed = U.  Slabs fetched today = 16.  So

    V = 16 - U  = the number of redundant slab fetches, 0 <= V <= TOPK

and since bs=1 fetches TOPK=8 slabs, the second row's MoE cost IS those 8
extra fetches.  A perfect fold recovers V of them:

    saving = (V / TOPK) * MoE_delta

Run:  python3 demo/glm5/price_moe_row_fold.py
"""

# ---- model geometry (config.json, /home/claudeuser/models/glm5-mxfp4) ------
N_ROUTED   = 256   # n_routed_experts
TOPK       = 8     # num_experts_per_tok
N_LAYERS   = 78    # config.json num_hidden_layers (the MTP draft layer is
                   # num_nextn_predict_layers=1 ON TOP of these, and runs as
                   # its own replay run -- it is not one of the 78)
DENSE      = 3     # first_k_dense_replace
MOE_LAYERS = N_LAYERS - DENSE   # 75
# CORRECTED 2026-08-22 from 76-3=73.  The V probe settles it independently of
# the config: the router epoch counter advances in a period-76 pattern (75
# printing invocations + 1 silent epoch per decode iteration), measured as
# exactly 76 across all 510 iteration boundaries in /tmp/vprobe/bs1.log.
# This raises the fold's ceiling by 2.7%, i.e. it corrects AGAINST the verdict.

# ---- measured (glm-mtp-chain-is-cheap-the-row-is-expensive, region map) ----
#
# CORRECTED 2026-08-22.  The region map labels S5->S6 "routing" and S6->S7
# "W13".  BOTH LABELS ARE WRONG, and I priced the fold off them.  Every
# mpk_stage_stamp id has exactly ONE call site, so the mapping is not
# ambiguous -- in gang_oproj_router_fused_mi300.cuh:
#     stamp 5 @1220  routing wait done, W13 not started
#     stamp 6 @1521  this worker's W13 TILES are done   -> S5->S6 = Phase 5 W13
#     stamp 7 @1614  the W13->W2 release is observed    -> S6->S7 = Phase 6 BARRIER
#     stamp 8 @1669  W2 tiles done                      -> S7->S8 = Phase 7 W2
# So the second row's W13 TILE delta is the +9.400 line, not the +3.550 line;
# +3.550 is a barrier, which a row fold cannot delete.  The MoE group total is
# unchanged (9.400+3.550+7.520 = 20.470); only the split inside it moves.  The
# tile delta a fold can attack is 53% larger than I first priced -- again a
# correction AGAINST the verdict.
ROW_TOTAL  = 5.367   # ms, a second LIVE row end to end
W13_DELTA  = 9.400   # us/layer, Phase 5 TILES   (was 3.550 = the barrier)
W2_DELTA   = 7.520   # us/layer, Phase 7 TILES
BAR_DELTA  = 3.550   # us/layer, Phase 6 W13->W2 BARRIER -- NOT foldable
W13_BASE   = 10.110  # us/layer at bs=1, Phase 5 tiles
W2_BASE    = 8.117   # us/layer at bs=1, Phase 7 tiles

MOE_DELTA_MS = (W13_DELTA + W2_DELTA) * MOE_LAYERS / 1000.0
THRESHOLD    = 0.5   # ms -- the guide's build/no-build bar


def saving_ms(v):
    return (v / TOPK) * MOE_DELTA_MS


print("=" * 76)
print("WHAT THE SECOND LIVE ROW COSTS, AND HOW MUCH OF IT THE FOLD CAN REACH")
print("=" * 76)
print(f"  second live row, end to end        {ROW_TOTAL:7.3f} ms")
print(f"  of which MoE ({W13_DELTA}+{W2_DELTA} us/layer x {MOE_LAYERS})"
      f"  {MOE_DELTA_MS:7.3f} ms  = {100*MOE_DELTA_MS/ROW_TOTAL:.1f}%")
print(f"  everything else (EP, o_proj, attn) {ROW_TOTAL-MOE_DELTA_MS:7.3f} ms"
      f"  <- the fold CANNOT touch this")

print("\n" + "=" * 76)
print("CEILING vs V")
print("=" * 76)
print(f"  {'V':>4} {'U=16-V':>7} {'saving ms':>10} {'% of row':>9}   verdict")
for v in (0, 1, 2, 4, 6, 8):
    s = saving_ms(v)
    print(f"  {v:4d} {16-v:7d} {s:10.3f} {100*s/ROW_TOTAL:8.1f}%   "
          f"{'BUILD' if s >= THRESHOLD else 'below the ' + str(THRESHOLD) + ' ms bar'}")

v_needed = THRESHOLD * TOPK / MOE_DELTA_MS
print(f"\n  V needed to clear the {THRESHOLD} ms bar: {v_needed:.2f}"
      f"  ({'IMPOSSIBLE -- V <= ' + str(TOPK) if v_needed > TOPK else 'possible'})")
print(f"  ABSOLUTE CEILING (V = TOPK = {TOPK}, i.e. the two rows route"
      f" IDENTICALLY):\n    {saving_ms(TOPK):.3f} ms ="
      f" {100*saving_ms(TOPK)/ROW_TOTAL:.1f}% of the second row.")

print("\n" + "=" * 76)
print("PRIOR ON V")
print("=" * 76)
ev = TOPK * TOPK / N_ROUTED
print(f"  Independent uniform routing, top-{TOPK} of {N_ROUTED}:")
print(f"    E[|A n B|] = {TOPK}*{TOPK}/{N_ROUTED} = {ev:.3f}"
      f"  ->  saving {saving_ms(ev):.3f} ms")
print(f"  Consecutive-token routing IS correlated, but it would take")
print(f"  {100*v_needed/TOPK:.0f}% expert reuse between adjacent tokens to reach"
      f" even {THRESHOLD} ms.")

print("\n" + "=" * 76)
print("THREE DISCOUNTS THE CEILING ABOVE DOES NOT APPLY")
print("=" * 76)
print(f"""  The {saving_ms(TOPK):.3f} ms figure is an upper bound on BYTES, not on wall.  All
  three known discounts push the realized number down, none push it up:

  1. EP SHARDING.  Each rank owns {N_ROUTED//8} experts, ~1-2 activated at bs=1.  A
     duplicate (expert, row0)/(expert, row1) pair sits on ONE rank.  Folding
     removes a tile from that rank only, and the layer's makespan is the
     SLOWEST rank -- so it pays only when the duplicate is on the straggler.
     glm-ep-routed-imbalance-is-floor-not-wall: 3x bytes of routed-expert
     imbalance produced 1.05 us of makespan spread.

  2. IN-PHASE ABSORPTION.  glm-cutting-work-in-a-phase-is-absorbed is 3 for 3:
     only deleting a WHOLE SERIAL PHASE moved the wall.  A row fold does not
     delete the MoE phase, it thins it.

  3. ROUND QUANTIZATION.  glm-moe-phase-is-round-quantized-and-w2-is-latency-
     bound: W13 is 8 tiles/XCD/expert and W2 is 12, and ceil(tiles/29) rounds.
     Removing V tiles that do not cross a round boundary buys exactly zero.

  Against these, glm-batch-row-mfma-fold-is-a-negative already measured the
  ANALOGOUS fold on qkv_a at +0.348 us/layer -- a negative, because an MFMA
  tile is M=16/32 so the second row already rides free in the instruction.
  The MoE fold differs only in that it also saves the slab FETCH.""")

print("\n" + "=" * 76)
print("VERDICT")
print("=" * 76)
print(f"""  The fold's ABSOLUTE ceiling -- both rows routing to an identical
  expert set, every duplicate fetch eliminated, zero absorption -- is
  {saving_ms(TOPK):.3f} ms of a {ROW_TOTAL:.3f} ms second row ({100*saving_ms(TOPK)/ROW_TOTAL:.1f}%), and it needs
  V = {TOPK}.  The independence prior puts V at {ev:.2f}, worth {saving_ms(ev):.3f} ms.
  The bar is {THRESHOLD} ms, which needs V >= {v_needed:.2f}, i.e. {100*v_needed/TOPK:.0f}% of the two
  rows' top-8 sets to coincide.

  This is a NO-GO unless the measured V is startlingly high.  MEASURE V FIRST
  (one decimated printf of d_mask[NUM_EXPERTS] on the MTP arm); do not build
  the fold on the strength of the arithmetic in either direction.""")


# ============================================================================
# CROSS-CHECK: is the second row's MoE cost really DISTINCT-EXPERT FETCHES?
# ============================================================================
# If W13/W2 tile time is set by how many expert weight slabs get fetched, then
# going from 8 fetches (bs=1) to 16 (bs=2, no fold) should scale the TILE time
# by 16/8 = 2.00.  This is a prediction the region map can falsify, and it is
# independent of everything above.
V_MEAS = 2.801          # measured, demo/glm5/vprobe_analyze.py
print()
print("=" * 76)
print("CROSS-CHECK: fetch-count model of the MoE tile phases")
print("=" * 76)
print("  fetch-bound prediction for bs=2 with NO fold: 16/8 = 2.000x")
print()
print(f"  {'phase':<6} {'bs=1':>8} {'bs=2':>8} {'ratio':>7}   {'vs 2.000':>9}")
tot_fold = 0.0
for name, base, d in (("W13", W13_BASE, W13_DELTA), ("W2", W2_BASE, W2_DELTA)):
    ratio = (base + d) / base
    # With a perfect fold only |union| = 16 - V slabs are fetched.
    folded_ratio = (2 * TOPK - V_MEAS) / TOPK
    folded_us = base * folded_ratio
    save_us = (base + d) - folded_us
    tot_fold += save_us
    print(f"  {name:<6} {base:8.3f} {base+d:8.3f} {ratio:7.3f}   {ratio-2.0:+9.3f}")
    print(f"         perfect fold -> {folded_ratio:.3f}x = {folded_us:.3f} us,"
          f" saving {save_us:.3f} us/layer")
print()
print(f"  Both phases land within 4% of the 2.000x fetch-bound prediction, and")
print(f"  they do so INDEPENDENTLY.  The second row's MoE cost is distinct-")
print(f"  expert weight traffic -- the model behind V is confirmed, not assumed.")
print()
print(f"  Fold saving from the MEASURED ratios: {tot_fold:.3f} us/layer"
      f" x {MOE_LAYERS} = {tot_fold*MOE_LAYERS/1000:.3f} ms")
print(f"  (the (V/TOPK)*MoE_delta form gives {V_MEAS/TOPK*MOE_DELTA_MS:.3f} ms;")
print(f"   the ratio form is tighter because the bs=1 base also contains the")
print(f"   shared expert and non-fetch work, which no fold touches)")
print(f"  Either way the ceiling is under the {THRESHOLD} ms bar -> NO-GO STANDS.")
