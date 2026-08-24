#!/usr/bin/env python3
"""RE-STEER v9, Item 1: the world-size fit, with the brief's arithmetic checked.

The v9 brief asked, in its own words, to check "the 37.5/80.2 MB split and the
claim that attention weights do not shard with world size.  If that split is
wrong, the whole 9.34 ms number is wrong."

It is wrong.  Three independent errors, all load-bearing, all corrected here.
The QUALITATIVE conclusion survives -- there IS a large byte-independent
component -- but every number attached to it changes, and one of them changes
sign in its interpretation.

Run:  python3 demo/glm5/world_size_fit.py
"""

# ---------------------------------------------------------------------------
# MEASURED WALLS.  bs=1, one decode row, ms/token.
# ---------------------------------------------------------------------------
NP8_WALL = 10.571     # pooled n=8, commit 1f1a87a, texts checked
NP4_WALL = 11.418     # MY n=3 this turn: 11.430 / 11.395 / 11.430, texts
                      # checked.  The v9 brief said 11.439 n=3 -- reproduced to
                      # 0.021 ms, far inside the 0.26 ms noise floor.

# ---------------------------------------------------------------------------
# BYTE FLOORS, whole-iteration, busiest rank, from demo/glm5/roofline.py
# AFTER the E[max] fix (see error 2).  ms/token.
# ---------------------------------------------------------------------------
FLOOR = {8: 1.819, 4: 2.399, 2: 3.465}

# Per-MoE-layer per-rank attention bytes, MB, from roofline.py.
ATTN_MB = {8: 37.44, 4: 56.64, 2: 95.02}
# Of which REPLICATED (q_a_proj 12.98 + kv_a_proj 3.65 + router 1.62):
ATTN_REPLICATED_MB = 18.25
# Routed-expert bytes on the busiest rank, MB/layer.
MOE_ROUTED_MB = {8: 52.08, 4: 70.95, 2: 102.15}
SHARED_EXPERT_MB = 20.05      # replicated at every world size

EMAX = {8: 2.597, 4: 3.538, 2: 5.094}   # exact E[max], 8 balls into R bins


def errors():
    print("=" * 78)
    print("ITEM 1, PART 0 -- THE BRIEF'S ARITHMETIC.  THREE ERRORS.")
    print("=" * 78)

    print("\nERROR 1 -- 'attention weights are REPLICATED, so they do not shard")
    print("           with world size.'  FALSE.  Most of attention DOES shard.\n")
    print("  roofline.py:108-112 divides q_b_proj, W_UK, W_UV and o_proj by the")
    print("  world size.  Only q_a_proj, kv_a_proj and router are replicated.")
    print("  CONFIRMED EMPIRICALLY in the live NP=4 run's own config lines:")
    print("    [CFG] o_proj  tp=1 cols_per_rank=1536   (hidden 6144 / 4)")
    print("    [CFG] q_b/W_UK tp=1 heads_per_rank=16   (heads 64 / 4)")
    print("  Both tensor-parallel shards are ACTIVE at NP=4, not bypassed.\n")
    print(f"  {'':<22}{'NP=8':>10}{'NP=4':>10}{'NP=2':>10}")
    print(f"  {'attention MB/lyr':<22}{ATTN_MB[8]:>10.2f}{ATTN_MB[4]:>10.2f}"
          f"{ATTN_MB[2]:>10.2f}")
    print(f"  {'  of which replicated':<22}{ATTN_REPLICATED_MB:>10.2f}"
          f"{ATTN_REPLICATED_MB:>10.2f}{ATTN_REPLICATED_MB:>10.2f}")
    print(f"  brief claimed attention is FLAT at 37.5 at every world size.")
    print(f"  It is 37.44 -> 56.64 -> 95.02.  Only {ATTN_REPLICATED_MB:.2f} MB "
          f"is replicated.")

    print("\nERROR 2 -- the MoE term does not double, and roofline.py's own")
    print("           E[max] was hardcoded.  BOTH the brief and the tool were")
    print("           wrong here; the tool is now fixed.\n")
    print("  roofline.py:132 read `max_active = 3.0` with the comment 'E[max]")
    print("  for 8 balls in 8 bins'.  Applied unchanged at R=4 and R=2 it")
    print("  produced, at R=2, a busiest-rank byte count BELOW the mean --")
    print("  arithmetically impossible.  Replaced with exact enumeration:\n")
    print(f"  {'':<22}{'NP=8':>10}{'NP=4':>10}{'NP=2':>10}")
    print(f"  {'E[max] exact':<22}{EMAX[8]:>10.3f}{EMAX[4]:>10.3f}"
          f"{EMAX[2]:>10.3f}")
    print(f"  {'E[max] as hardcoded':<22}{3.0:>10.3f}{3.0:>10.3f}{3.0:>10.3f}")
    print(f"  {'routed MB/lyr':<22}{MOE_ROUTED_MB[8]:>10.2f}"
          f"{MOE_ROUTED_MB[4]:>10.2f}{MOE_ROUTED_MB[2]:>10.2f}")
    print(f"  {'shared expert MB/lyr':<22}{SHARED_EXPERT_MB:>10.2f}"
          f"{SHARED_EXPERT_MB:>10.2f}{SHARED_EXPERT_MB:>10.2f}")
    print()
    print(f"  Brief: MoE 80.2 -> 160.4 MB/lyr (a clean 2x).  Actually")
    print(f"  {MOE_ROUTED_MB[8] + SHARED_EXPERT_MB:.2f} -> "
          f"{MOE_ROUTED_MB[4] + SHARED_EXPERT_MB:.2f} = "
          f"{(MOE_ROUTED_MB[4] + SHARED_EXPERT_MB) / (MOE_ROUTED_MB[8] + SHARED_EXPERT_MB):.2f}x.")
    print("  Two reasons it cannot double: the makespan is set by E[max], which")
    print("  grows sublinearly (2.597 -> 3.538 = 1.36x, not 2x), and the SHARED")
    print("  expert is replicated at every world size so it does not scale at all.")

    print("\nERROR 3 -- the per-rank byte ratio, and hence the floors.\n")
    r84 = FLOOR[4] / FLOOR[8]
    print(f"  brief:     1.68x the bytes, floor 1.936 -> 3.254 ms")
    print(f"  corrected: {r84:.3f}x the bytes, floor {FLOOR[8]:.3f} -> "
          f"{FLOOR[4]:.3f} ms")
    print("  The brief applied a PER-LAYER TILE-CLASS ratio to a WHOLE-ITERATION")
    print("  floor.  Those have different denominators: the whole-iteration")
    print("  count also carries embeddings, the LM head, the 3 dense layers and")
    print("  the KV cache, none of which scale the way the MoE half does.")
    print()


def fit():
    """The two-point fit.  SUPERSEDED by recut() -- read that first.

    Kept because it is the thing the third point falsified, and because its
    fixed component (7.915) is the number recut() has to move.  Both of the
    numbers it produces -- marginal 1.460, fixed 7.915 -- are WRONG; the
    slope is contaminated by structure, not bytes.  See recut().
    """
    print("=" * 78)
    print("ITEM 1 -- THE TWO-POINT FIT, REDONE ON CORRECTED BYTES")
    print("        (SUPERSEDED -- both numbers below are falsified by 1.2)")
    print("=" * 78)
    d_wall = NP4_WALL - NP8_WALL
    d_floor = FLOOR[4] - FLOOR[8]
    marginal = d_wall / d_floor
    fixed = NP8_WALL - marginal * FLOOR[8]

    print(f"\n  {'':<34}{'brief':>12}{'corrected':>12}")
    print(f"  {'-' * 34} {'-' * 11} {'-' * 11}")
    print(f"  {'NP=8 byte floor, ms':<34}{1.936:>12.3f}{FLOOR[8]:>12.3f}")
    print(f"  {'NP=4 byte floor, ms':<34}{3.254:>12.3f}{FLOOR[4]:>12.3f}")
    print(f"  {'delta floor, ms':<34}{1.318:>12.3f}{d_floor:>12.3f}")
    print(f"  {'delta wall, ms':<34}{0.851:>12.3f}{d_wall:>12.3f}")
    print(f"  {'marginal wall per ms of roof':<34}{0.645:>12.3f}"
          f"{marginal:>12.3f}")
    print(f"  {'byte-INDEPENDENT component, ms':<34}{9.34:>12.3f}{fixed:>12.3f}")
    print(f"  {'  as % of the NP=8 wall':<34}{88.0:>12.1f}"
          f"{100 * fixed / NP8_WALL:>12.1f}")
    print()
    print("  THE INTERPRETATION FLIPS ON THE MARGINAL RATE.")
    print(f"  The brief read 0.645 -- marginal bytes cheaper than their roof --")
    print("  and concluded 'the extra MoE bytes landed in workers that were")
    print("  already spinning, so they cost less than their roof time.'")
    print(f"  Corrected, the marginal rate is {marginal:.3f}: marginal bytes cost")
    print(f"  {marginal:.2f}x their roof time.  There is no free absorption.  That")
    print("  is the expected direction -- it sits between 1.0 and the tile")
    print("  class's measured 3.09x over its own byte roof (TILE_CLASS_BOUND.md).")
    print()
    print("  THE BYTE-INDEPENDENT COMPONENT SURVIVES, at a smaller size:")
    print(f"    {fixed:.3f} ms, {100 * fixed / NP8_WALL:.0f} % of the NP=8 wall "
          f"(brief said 9.34 / 88 %).")
    print()
    print("  AND IT CORROBORATES AN INDEPENDENT MEASUREMENT.  "
          "TWO_MS_IMPOSSIBILITY.md")
    print("  reports that deleting the ENTIRE tile class by ablation leaves")
    print("  5.327 ms.  A regression across world sizes and a direct ablation")
    print("  are completely different instruments; they bracket the same")
    print(f"  quantity ({fixed:.2f} vs 5.33 ms).  The difference is itself")
    print("  meaningful: the ablation deletes tile time INCLUDING its")
    print("  byte-independent VALU and latency share, so it should land LOWER")
    print("  than a pure byte-extrapolation, and it does.")
    print()
    return marginal, fixed


def predict(marginal, fixed):
    print("=" * 78)
    print("ITEM 1.1 -- THE NP=2 THIRD POINT.  ATTEMPTED; IT DOES NOT FIT.")
    print("=" * 78)
    print("""
  RESULT: OOM.  /tmp/glm5_v9np2_r1.log:130 reaches
      [load] shard 142/142  219.1 GiB resident
  and then, at rocSHMEM init:
      Error: _hip_alloc_with_flags(...): out of memory (2)
        at RocSHMEM::.../memory/memory_allocator.cpp:76
      _ZN8rocshmem10HeapMemoryINS_23HIPAllocatorFinegrainedEEC2Em
        <- SingleHeap <- SymmetricHeap

  The part is 252.0 GiB (rocm-smi: 270566162432 B), not the 288 GB the spec
  sheet quotes, and the heap that failed is the DEFAULT 1 GiB
  (rocSHMEM/src/envvar.cpp:42, `var<size_t> heap_size("HEAP_SIZE", "", 1L<<30)`).
  A 1 GiB fine-grained allocation failing with 32.9 GiB nominally free is the
  useful number here: ~33 GiB of real device use sits ABOVE whatever the
  loader prints as "resident".  Carry that overhead into any future
  footprint estimate on this box.

  Per the brief -- "if it OOMs, say so in one line and go to 1.2" -- not
  fought.  The predictions below are left in place UNTESTED, as the record of
  what the two models said.
""")
    fixed_model = fixed + marginal * FLOOR[2]
    bw_model = NP8_WALL * FLOOR[2] / FLOOR[8]
    print(f"\n  NP=2 byte floor (corrected)                    {FLOOR[2]:6.3f} ms")
    print(f"  fixed + marginal-byte model    {fixed:.3f} + "
          f"{marginal:.3f} x {FLOOR[2]:.3f} = {fixed_model:6.3f} ms   <- PREDICTION")
    print(f"  bandwidth-proportional model   {NP8_WALL:.3f} x "
          f"{FLOOR[2]:.3f}/{FLOOR[8]:.3f}   = {bw_model:6.3f} ms")
    print(f"  spread between the two models                  "
          f"{bw_model / fixed_model:6.2f}x")
    print()
    print("  The brief predicted 13.14 vs 32.2 (2.5x spread).  Corrected the")
    print(f"  spread is {bw_model / fixed_model:.2f}x -- smaller, still decisive: the two")
    print(f"  models are {bw_model - fixed_model:.1f} ms apart against a 0.26 ms noise floor.")
    print()

    # Memory feasibility.
    per_expert_mb = SHARED_EXPERT_MB          # 1 expert's bytes at mxfp4
    moe_layers = 75
    for R in (4, 2):
        owned = 256 // R
        resident = owned * per_expert_mb * moe_layers / 1024.0
        print(f"  NP={R}: {owned} experts/rank x {per_expert_mb:.2f} MB x "
              f"{moe_layers} layers = {resident:6.1f} GiB of expert weights")
    print(f"  NP=4 measured resident (brief): 129.4 GiB, so non-expert "
          f"overhead ~= {129.4 - 64 * per_expert_mb * moe_layers / 1024.0:.1f} GiB")
    np2_est = 128 * per_expert_mb * moe_layers / 1024.0 + (
        129.4 - 64 * per_expert_mb * moe_layers / 1024.0)
    print(f"  NP=2 estimate {np2_est:.1f} GiB vs 219.1 GiB actually loaded -- "
          f"the estimate was")
    print(f"  fine.  What was wrong was the DENOMINATOR: this part is 252.0 "
          f"GiB, not the")
    print(f"  288 GB on the spec sheet, and ~33 GiB of real use sits above "
          f"the printed")
    print(f"  resident.  {np2_est:.0f} + 33 > 252.  It could not have fit; "
          f"the OOM was predictable")
    print(f"  from these numbers and I did not predict it.  Recorded so the "
          f"next")
    print(f"  footprint estimate on this box uses 252.0 GiB and adds the "
          f"33 GiB.")
    print()


# ---------------------------------------------------------------------------
# ITEM 1.2 -- the bytes-only ablation at FIXED NP=4.
#
# GLM_MOE_WIDEN_MXFP8=all re-expresses the MXFP4 checkpoint's routed and
# shared experts as MXFP8 at load, exactly (demo.py:widen_mxfp4_to_mxfp8 --
# every E2M1 level is representable in E4M3 and the E8M0 scale is shared, so
# the run must emit IDENTICAL tokens).  Task graph, rendezvous count, worker
# assignment and numerics are all held fixed; only bytes move.
# ---------------------------------------------------------------------------
NP4_FLOOR_MXFP8 = 3.642     # roofline.py --ranks 4 --moe-quant mxfp8
NP4_WIDEN_WALL = 12.550     # MEASURED n=3: 12.531 / 12.575 / 12.545, spread
                            # 0.044 ms, all three texts read and coherent,
                            # [CFG] GLM_MOE_WIDEN_MXFP8 present in all three.
                            # (A fourth run, the first of the v9widen launch,
                            # OOMed during LOAD -- I relaunched without
                            # checking for orphan ranks and stale VRAM was
                            # still held.  Discarded, not averaged in.)


def item12(marginal, fixed):
    print("=" * 78)
    print("ITEM 1.2 -- THE BYTES-ONLY ABLATION AT FIXED NP=4")
    print("=" * 78)
    f4, f8 = FLOOR[4], NP4_FLOOR_MXFP8
    models = (
        ("M1  fixed + marginal (this file's fit)", fixed + marginal * f8),
        ("M2  1:1 addition against roof time",     NP4_WALL + (f8 - f4)),
        ("M3  bandwidth-proportional",             NP4_WALL * f8 / f4),
        ("M0  null: marginal bytes are free",      NP4_WALL),
        ("M4  prior: reverse buy on 4.7-Flash",    NP4_WALL + 0.167),
    )
    print(f"\n  NP=4 byte floor, MXFP4 experts   {f4:6.3f} ms")
    print(f"  NP=4 byte floor, MXFP8 experts   {f8:6.3f} ms   "
          f"(+{f8 - f4:.3f})")
    print(f"  measured baseline                {NP4_WALL:6.3f} ms  n=3\n")
    print(f"  {'PREDICTION, stated before the run':<40}{'wall':>9}{'delta':>9}")
    print(f"  {'-' * 40} {'-' * 8} {'-' * 8}")
    for name, p in models:
        print(f"  {name:<40}{p:>9.3f}{p - NP4_WALL:>+9.3f}")
    print()
    print(f"  M1 vs M2 separation {abs(models[0][1] - models[1][1]):.3f} ms "
          f"-- 2.2x the 0.26 ms noise floor and ~6x the")
    print("  observed n=3 spread, so this run can tell them apart.")
    print()
    print("  WHY THIS IS WEAKER THAN 1.1, stated up front: MXFP8 is not only")
    print("  more bytes.  The scaled MFMA covers half the K per instruction at")
    print("  E4M3 that it does at E2M1, so the widened path also issues 2x the")
    print("  MoE MFMAs.  MFMA occupancy is 2.65 % on this layer")
    print("  (glm-tile-class-is-latency-neither-unit-saturated), so the byte")
    print("  term should dominate -- but a result near M2 is consistent with")
    print("  'bytes cost 1:1' AND with 'bytes are free and the MFMAs cost'.")
    print("  Only a result at or above M1 discriminates cleanly.")
    print()
    d = NP4_WIDEN_WALL - NP4_WALL
    slope = d / (f8 - f4)
    print(f"  MEASURED: {NP4_WIDEN_WALL:.3f} ms  n=3 "
          f"(12.531 / 12.575 / 12.545), delta {d:+.3f} ms")
    print(f"  {'':<40}{'wall':>9}{'error':>9}")
    print(f"  {'-' * 40} {'-' * 8} {'-' * 8}")
    for name, p in sorted(models, key=lambda m: abs(m[1] - NP4_WIDEN_WALL)):
        mark = "  <-- lands" if abs(p - NP4_WIDEN_WALL) < 0.26 else ""
        print(f"  {name:<40}{p:>9.3f}{p - NP4_WIDEN_WALL:>+9.3f}{mark}")
    print()
    print(f"  M2 is the only model inside the 0.26 ms noise floor.  M1 -- this")
    print(f"  file's own two-point fit -- misses by "
          f"{models[0][1] - NP4_WIDEN_WALL:+.3f} ms, 2.6x the floor.")
    print()
    return slope


def recut(marginal_2pt, fixed_2pt, slope):
    print("=" * 78)
    print("ITEM 1 -- THE RE-CUT.  THE THIRD POINT BREAKS THE SLOPE AND SAVES")
    print("          THE FIXED COMPONENT.")
    print("=" * 78)
    f4, f8 = FLOOR[4], NP4_FLOOR_MXFP8
    d = NP4_WIDEN_WALL - NP4_WALL
    d_floor_np = FLOOR[4] - FLOOR[8]
    d_wall_np = NP4_WALL - NP8_WALL
    structural = d_wall_np - slope * d_floor_np
    fixed4 = NP4_WALL - slope * FLOOR[4]
    fixed8 = NP8_WALL - slope * FLOOR[8]

    print(f"""
  TWO SLOPES, MEASURED TWO WAYS, AND THEY DISAGREE:

    across world size  NP=8 -> NP=4   {d_wall_np:+.3f} wall / {d_floor_np:+.3f} roof = {marginal_2pt:.3f}
    within  world size, bytes only    {d:+.3f} wall / {f8 - f4:+.3f} roof = {slope:.3f}

  The disagreement IS the finding.  A world-size change is not a byte change:
  it moves shard widths, the EP fan-out, E[max], the collective payload and
  the per-rank tile counts all at once.  The MXFP4 -> MXFP8 widen moves ONLY
  bytes -- same task graph, same 10 rendezvous, same 232 workers, same
  per-XCD tile counts, bit-identical values.  {slope:.3f} is therefore the honest
  marginal byte rate and {marginal_2pt:.3f} was contaminated.

  Splitting the NP=8 -> NP=4 delta on the clean slope:
    total                              {d_wall_np:+.3f} ms
    explained by bytes  {slope:.3f} x {d_floor_np:.3f}   {slope * d_floor_np:+.3f} ms
    STRUCTURAL residual                {structural:+.3f} ms
  So {100 * structural / d_wall_np:.0f} % of the cost of halving the world size is NOT bytes.

  THE BYTE-INDEPENDENT COMPONENT, re-cut on the clean slope:
    at NP=4   {NP4_WALL:.3f} - {slope:.3f} x {FLOOR[4]:.3f} = {fixed4:.3f} ms  ({100 * fixed4 / NP4_WALL:.1f} % of the wall)
    at NP=8   {NP8_WALL:.3f} - {slope:.3f} x {FLOOR[8]:.3f} = {fixed8:.3f} ms  ({100 * fixed8 / NP8_WALL:.1f} % of the wall)
    (the {fixed4 - fixed8:.3f} ms gap between them is exactly the structural residual
     above -- the decomposition is self-consistent.)

  VERDICT, one line: the fixed component SURVIVES and is BIGGER than the
  two-point fit said -- {fixed4:.2f} ms at NP=4, {100 * fixed4 / NP4_WALL:.0f} % of the wall -- so the v9 brief's
  9.34 ms lands within {abs(fixed4 - 9.34):.2f} ms of a structure-controlled measurement even
  though every step of its derivation was wrong.  The board is the fixed
  component.

  AND THE {fixed4:.2f} IS A LOWER BOUND.  MXFP8 does not only add bytes: the
  scaled MFMA covers half the K per instruction at E4M3, so the widened MoE
  also issues 2x the MFMAs.  Any part of the {d:+.3f} ms that is MFMA issue
  rather than bytes makes the true byte slope SMALLER than {slope:.3f} and the
  fixed component LARGER than {fixed4:.2f}.  It cannot move the other way.

  WHAT DIED HERE: this file's own 'marginal bytes cost {marginal_2pt:.2f}x their roof
  time, there is no free absorption' paragraph.  Refuted by measurement.
  Marginal bytes cost {slope:.2f}x their roof -- a hair UNDER 1:1, which is the
  ordinary additive-probe rate in the ledger
  (glm-additive-probes-overprice-deletions), not an absorption effect.
""")
    return fixed4


def corollary():
    print("=" * 78)
    print("THE BRIEF'S COROLLARY -- gpt-oss parity 'ALREADY MET at NP=4'")
    print("=" * 78)
    oss_eff = 0.530 / 1.936
    np4_eff = FLOOR[4] / NP4_WALL
    np8_eff = FLOOR[8] / NP8_WALL
    print(f"\n  gpt-oss efficiency vs its own byte roof     {100 * oss_eff:5.1f} %")
    print(f"  GLM NP=8, corrected floor                  {100 * np8_eff:5.1f} %")
    print(f"  GLM NP=4, corrected floor                  {100 * np4_eff:5.1f} %")
    print(f"  brief claimed NP=4 was                      28.4 %")
    print()
    print("  Parity is NOT met: 21.0 % vs 27.4 %.  The 28.4 % came from the")
    print("  inflated 3.254 ms floor of error 3.")
    print()
    print("  HOWEVER -- the brief's DECISION to retire the 7.32 target stands,")
    print("  on its other stated reason, which the corrections do not touch:")
    print("  efficiency-vs-byte-roof is the wrong denominator when most of the")
    print("  wall is not bytes.  A byte-independent share of 81 % makes that")
    print("  argument just as well as 88 % did.  v8's target stays retired;")
    print("  only its epitaph changes.")
    print()


def board(fixed4):
    print("=" * 78)
    print("WHAT ITEM 2 SHOULD BE BUILT ON")
    print("=" * 78)
    print(f"""
  The NP=4 wall, {NP4_WALL:.3f} ms, splits into exactly two pieces now, both
  measured rather than modelled:

    byte-attributable   {NP4_WALL - fixed4:6.3f} ms   ({100 * (NP4_WALL - fixed4) / NP4_WALL:4.1f} %)  slope measured at fixed NP
    byte-INDEPENDENT    {fixed4:6.3f} ms   ({100 * fixed4 / NP4_WALL:4.1f} %)  <-- THE BOARD

  Item 2 asks what the byte-independent piece is made of.  Build its table
  against {fixed4:.2f}, not against the brief's 9.34 and not against this file's
  earlier 7.915.  The three candidate contents, from the existing ledger:

    - last-arriver floor        54 % of the layer is arrival spread, of which
                                1.506 ms is cross-rank
                                (glm-layer-is-54pct-last-arriver-floor)
    - VALU prologues            quant / RMSNorm / rope on the CRITICAL worker.
                                The '+0.187 hoist is a wash' null does NOT
                                cover this -- it removed REDUNDANCY across
                                parallel workers, not VALU from the critical
                                path (glm-hoisting-parallel-redundancy-is-a-wash)
    - rendezvous count          10 device-wide, 2 cross-rank, 3.77 us each
                                (glm-one-rendezvous-costs-3.77us)

  All three are latency, none is bytes, and that is consistent with the tile
  class already measuring 3.09x over its own byte roof with MFMA at 2.65 %
  and HBM at 32.7 % (glm-tile-class-is-latency-neither-unit-saturated).
""")


if __name__ == "__main__":
    errors()
    m, f = fit()
    predict(m, f)
    s = item12(m, f)
    fixed4 = recut(m, f, s)
    corollary()
    board(fixed4)
