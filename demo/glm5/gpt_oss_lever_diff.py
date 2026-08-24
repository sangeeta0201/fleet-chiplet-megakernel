#!/usr/bin/env python3
"""RE-STEER v8, Item 1: the gpt-oss lever walk, and the per-LAYER normalization.

Item 1.1/1.2 -- every lever in ~/mirage's git log from ROOFLINE_ANALYSIS.md's
commit to c156900 (3.339 -> 1.936 ms), each classified PRESENT / (a) not ported
/ (b) ported-and-null / (c) structurally inapplicable.

Item 1.3 -- the per-phase efficiency diff.  It cannot be done from
~/mirage/demo/gpt_oss/{ROOFLINE_ANALYSIS,PHASE_BREAKDOWN}.md: both are stale
(3.339 and 3.25 ms) and no current per-phase table at 1.936 exists there.  What
CAN be done, and is more decisive, is to normalize both walls PER LAYER and per
byte.  Doing so relocates the entire 3.27 ms gap.

Read-only against ~/mirage.  No GPU run; every input is a number already
measured on this branch and cited to its commit.
"""

# --------------------------------------------------------------------------
# Measured inputs.  Each is cited.
# --------------------------------------------------------------------------
GLM_WALL_MS = 10.571     # pooled n=8: 10.588 (n=5, a678637) + 10.543 (n=3, this turn)
GLM_LAYERS = 76
GLM_GB_TOK_GPU = 10.01   # the guide's v8 table
OSS_WALL_MS = 1.936      # ~/mirage c156900
OSS_LAYERS = 36          # gpt-oss 120B
OSS_GB_TOK_GPU = 2.74    # the guide's v8 table
HBM_TBS = 5.17           # measured, demo/glm5/roofline.py

# GLM class budget, us/layer -> ms/token via F (close_the_wall.py, 1ef2283 +
# 2ddf4dc).  F = 76 * K / 1000, K = 139.72/164.947.
F = 0.064377
GLM_CLASS_US = {          # us/layer, instrumented
    # 82.209 is close_the_wall.py's OWN tile sum (its 8 "tile" lines, W2
    # included).  Do NOT substitute TILE_CLASS_BOUND.md's 69.64 here: that is a
    # modelled 10-phase figure that excludes W2, and mixing it with this file's
    # coll/rdv/bound would not sum to the ACCOUNTED 10.002 ms.
    "tile":  82.209,
    "coll":  24.651,      # EP collective 15.617 + QB_TP cross-rank gather 9.034
    "rdv":   36.833,
    "bound": 11.671,
}
GLM_RESIDUAL_MS = 0.617   # dense prologue layers, sampling, harness

# gpt-oss is SINGLE-GPU.  Verified this turn:
#   ~/mirage/demo/gpt_oss/demo.py:374            world_size = 1
#   ~/mirage/demo/gpt_oss/bench_1k1k.sh          HIP_VISIBLE_DEVICES=<one>
#   ~/mirage/demo/gpt_oss/bench_vllm_v017.sh     HIP_VISIBLE_DEVICES=<one>
OSS_CROSS_RANK_MS = 0.0

# --------------------------------------------------------------------------
# Item 1.1 / 1.2 -- the lever table.
#   status: P  = present in GLM (grep-verified or cited)
#           b  = ported to GLM and measured null/negative (memory cite)
#           c  = structurally inapplicable to MLA / EP / 8-rank
#           a  = NOT PORTED -- this is the only category that is work
# --------------------------------------------------------------------------
LEVERS = [
    # commit,     lever,                                  delta,  status, evidence
    ("36fb412", "Monotonic barriers",                     -6.7, "P",
     "grep: monoton|epoch in 3 GLM headers vs 2 oss"),
    ("74173f1", "Fused-layer batching + MoE scale overlap", -7.5, "P",
     "gang_mla_full_layer_fused: MoE is Phases 5+7 of the one gang task"),
    ("c156900", "OProj prefetch + FP8 reciprocal",         -5.6, "P",
     "gang_moe_linear_mxfp4:124 amax*(1/448); :646 amax*(1/6) 'not v_div'"),
    ("5f51a90", "O-proj LDS prefetch",                     -4.5, "P",
     "prefetch-across-barrier idiom ported; gpt-oss-prefetch-across-barrier-idiom"),
    ("1b8787f", "O-proj LDS prefetch correctness fix",     -4.2, "P",
     "same header"),
    ("bda8607", "global_load_lds_dwordx4 in W13 + W2",     -3.4, "P",
     "grep: global_load_lds present in GLM MoE header"),
    ("e8adbc0", "Fused LM head + argmax",                  -3.1, "c",
     "gang_full_layer_with_lmhead_fused exists; GLM has no MLA+lmhead variant. "
     "One task on the LAST layer only -- 1/76 of the wall, not per-layer"),
    ("dff01a2", "W13 K=2944 tile shape",                   -2.7, "b",
     "GLM's MoE tile sweep: OPW=16 -1.34 ms, OPW=128 neutral "
     "(glm-moe-tiles-are-at-the-hbm-roof-narrowing-loses)"),
    ("24b8d4e", "W2 prefetch overlap",                     -2.3, "b",
     "W2 L2 prefetch measured 0.13% = neutral (glm-w2-l2-prefetch-measured-neutral)"),
    ("2e0f83c", "Collapse 36 per-layer tasks to 1",        -2.1, "P",
     "GLM runs the multi-layer loop; MPK_ABL_ML_BOUNDARY is its ablation probe "
     "(glm-layer-boundary-ablation-is-zero)"),
    ("85f77bf", "CROC chunk barrier",                      -1.8, "P",
     "ported as PAIR_MERGE, +0.07 = closed (gpt-oss-croc-chunk-barrier-...)"),
    ("ec9d969", "merge TPT 16 -> 32",                      -1.4, "b",
     "PAIR_MERGE + KV-chunk retunes: 4 knobs, all inside noise"),
    ("9ea3ebd", "vmcnt serialization fix",                 -1.4, "P",
     "grep: inline-asm vmcnt in both closures"),
    ("79d55aa", "Fuse write-through store",                -1.2, "P",
     "grep: st_wt in 5 GLM headers vs 3 oss"),
    ("a188c46", "exp2 TopK",                               -0.9, "P",
     "grep: exp2 in 2 GLM headers vs 2 oss"),
    ("c7bf746", "Per-wave W13 decomposition",              -0.9, "b",
     "GLM W13 is 8 tiles/XCD/expert, round-quantized "
     "(glm-moe-phase-is-round-quantized-and-w2-is-latency-bound)"),
    ("92735b1", "Mechanism C",                             -0.6, "P",
     "S28->S29 in the stage-stamp table is 'W_UK/W_UV + Mech-C'"),
    ("78dedf0", "All-thread barrier polling",              -0.6, "b",
     "+0.73 ms on GLM, MEASURED TWICE (glm-all-thread-barrier-polling-is-a-negative)"),
    ("681a57e", "301 -> 21 tasks, 42 -> 7 events",          0.0, "P",
     "no measured delta in its own subject; GLM's boundary class ablates to "
     "0.051 ms total (glm-layer-boundary-ablation-is-zero)"),
    ("df77f37", "Multi-layer fusion",                       0.0, "P",
     "same probe; GLM's per-layer loop is in-kernel"),
    ("0352a70", "XOR swizzle (standalone bench, 7.9x)",     0.0, "a",
     "ABSENT from BOTH closures -- never landed in gpt-oss's shipping path "
     "either, so it is not part of the 27.9%"),
    ("b94a54e", "Branchless TopK",                          0.0, "P",
     "grep: branchless TopK asm in both"),
]


def item12():
    print("=" * 78)
    print("ITEM 1.1 / 1.2 -- the gpt-oss lever walk, 3.339 -> 1.936 ms")
    print("=" * 78)
    print(f"{'commit':<9} {'lever':<40} {'delta%':>7}  st")
    print("-" * 9 + " " + "-" * 40 + " " + "-" * 7 + "  --")
    for c, name, d, st, _ev in LEVERS:
        print(f"{c:<9} {name:<40} {d:7.1f}  {st}")
    counts = {}
    for _c, _n, _d, st, _e in LEVERS:
        counts[st] = counts.get(st, 0) + 1
    print()
    print(f"  PRESENT in GLM                        {counts.get('P', 0):2d}")
    print(f"  (b) ported and measured null/negative {counts.get('b', 0):2d}")
    print(f"  (c) structurally inapplicable         {counts.get('c', 0):2d}")
    print(f"  (a) NOT PORTED -- the only work       {counts.get('a', 0):2d}")
    print()
    for c, name, _d, st, ev in LEVERS:
        if st == "a":
            print(f"  (a) {c} {name}")
            print(f"      {ev}")
    print()


def item13():
    print("=" * 78)
    print("ITEM 1.3 -- the per-phase diff, done as a PER-LAYER normalization")
    print("=" * 78)
    print("~/mirage's per-phase tables are stale (3.339 / 3.25 ms) and no table")
    print("at 1.936 exists.  Normalizing per layer is both possible and sharper.\n")

    glm_local_ms = GLM_WALL_MS - GLM_CLASS_US["coll"] * F
    oss_local_ms = OSS_WALL_MS - OSS_CROSS_RANK_MS

    rows = [
        ("layers", GLM_LAYERS, OSS_LAYERS),
        ("GB/token/GPU", GLM_GB_TOK_GPU, OSS_GB_TOK_GPU),
        ("MB/layer/GPU", GLM_GB_TOK_GPU * 1000 / GLM_LAYERS,
         OSS_GB_TOK_GPU * 1000 / OSS_LAYERS),
        ("byte floor, ms/token", GLM_GB_TOK_GPU / HBM_TBS,
         OSS_GB_TOK_GPU / HBM_TBS),
        ("MEASURED wall, ms/token", GLM_WALL_MS, OSS_WALL_MS),
        ("cross-rank, ms/token", GLM_CLASS_US["coll"] * F, OSS_CROSS_RANK_MS),
        ("LOCAL wall, ms/token", glm_local_ms, oss_local_ms),
        ("us/layer, wall", GLM_WALL_MS * 1000 / GLM_LAYERS,
         OSS_WALL_MS * 1000 / OSS_LAYERS),
        ("us/layer, LOCAL", glm_local_ms * 1000 / GLM_LAYERS,
         oss_local_ms * 1000 / OSS_LAYERS),
        ("MB per us, LOCAL", GLM_GB_TOK_GPU * 1000 / (glm_local_ms * 1000),
         OSS_GB_TOK_GPU * 1000 / (oss_local_ms * 1000)),
    ]
    print(f"{'':<28} {'GLM-5 744B':>12} {'gpt-oss 120B':>14}   ratio")
    print("-" * 28 + " " + "-" * 12 + " " + "-" * 14 + "   -----")
    for label, g, o in rows:
        r = g / o if o else float("inf")
        rs = f"{r:5.2f}x" if o else "  n/a"
        print(f"{label:<28} {g:12.3f} {o:14.3f}   {rs}")
    print()

    eff_glm = (GLM_GB_TOK_GPU / HBM_TBS) / GLM_WALL_MS
    eff_oss = (OSS_GB_TOK_GPU / HBM_TBS) / OSS_WALL_MS
    eff_glm_local = (GLM_GB_TOK_GPU / HBM_TBS) / glm_local_ms
    print("Efficiency vs own byte roof:")
    print(f"  gpt-oss, as measured (1 GPU)          {eff_oss * 100:5.1f} %")
    print(f"  GLM, as measured (8 GPUs)             {eff_glm * 100:5.1f} %")
    print(f"  GLM, cross-rank removed (like-for-like) {eff_glm_local * 100:5.1f} %")
    print()

    # The tile class alone, against gpt-oss's WHOLE layer.
    glm_tile_mb_per_us = (GLM_GB_TOK_GPU * 1000 / GLM_LAYERS) / GLM_CLASS_US["tile"]
    oss_layer_mb_per_us = ((OSS_GB_TOK_GPU * 1000 / OSS_LAYERS)
                           / (OSS_WALL_MS * 1000 / OSS_LAYERS))
    print("THE LOAD-BEARING ROW -- GLM's tile class vs gpt-oss's WHOLE layer:")
    print(f"  GLM tile class only     {GLM_CLASS_US['tile']:6.2f} us/layer for "
          f"{GLM_GB_TOK_GPU * 1000 / GLM_LAYERS:6.2f} MB = "
          f"{glm_tile_mb_per_us:5.2f} MB/us")
    print(f"  gpt-oss ENTIRE layer    "
          f"{OSS_WALL_MS * 1000 / OSS_LAYERS:6.2f} us/layer for "
          f"{OSS_GB_TOK_GPU * 1000 / OSS_LAYERS:6.2f} MB = "
          f"{oss_layer_mb_per_us:5.2f} MB/us")
    print(f"  -> GLM's tiles are {glm_tile_mb_per_us / oss_layer_mb_per_us:.2f}x "
          f"MORE byte-efficient than gpt-oss's entire layer.")
    print()

    # Where the gap actually is.
    target_ms = (GLM_GB_TOK_GPU / HBM_TBS) / eff_oss + GLM_CLASS_US["coll"] * F
    gap = GLM_WALL_MS - target_ms
    nontile_us = (GLM_CLASS_US["rdv"] + GLM_CLASS_US["bound"])
    nontile_ms = nontile_us * F
    print("WHERE THE GAP IS:")
    print(f"  v8 target (GLM at gpt-oss's {eff_oss * 100:.1f} % + measured "
          f"cross-rank)   {target_ms:6.3f} ms")
    print(f"  measured                                                "
          f"{GLM_WALL_MS:6.3f} ms")
    print(f"  GAP                                                     "
          f"{gap:6.3f} ms")
    print()
    print(f"  cross-rank (gpt-oss pays 0 of this; it is world_size=1) "
          f"{GLM_CLASS_US['coll'] * F:6.3f} ms")
    print(f"  GLM non-tile local: rendezvous + boundary               "
          f"{nontile_ms:6.3f} ms  ({nontile_us:.1f} us/layer x 76)")
    print(f"  GLM residual (dense prologue, harness)                  "
          f"{GLM_RESIDUAL_MS:6.3f} ms")
    print(f"  GLM tile class                                          "
          f"{GLM_CLASS_US['tile'] * F:6.3f} ms")
    print()
    print("  The tile class is NOT the gap: at gpt-oss's own MB/us it would")
    print(f"  cost {(GLM_GB_TOK_GPU * 1000 / GLM_LAYERS) / oss_layer_mb_per_us:.2f}"
          f" us/layer, MORE than the {GLM_CLASS_US['tile']:.2f} it actually costs.")
    print()
    print(f"  76 layers vs 36: GLM pays every per-layer fixed cost "
          f"{GLM_LAYERS / OSS_LAYERS:.2f}x as often")
    print(f"  for only {(GLM_GB_TOK_GPU / GLM_LAYERS) / (OSS_GB_TOK_GPU / OSS_LAYERS):.2f}x"
          f" the bytes per layer.  Fixed cost does not scale with bytes.")
    print(f"  GLM non-tile local per layer: {nontile_us:.1f} us of a "
          f"{(glm_local_ms * 1000 / GLM_LAYERS):.1f} us local layer = "
          f"{100 * nontile_us / (glm_local_ms * 1000 / GLM_LAYERS):.0f} %.")
    print()

    # ------------------------------------------------------------------
    # Reconciliation with the guide's 7.32.  Its construction is
    #   1.936 / 0.279 + 0.383  where 0.383 is the EP BYTE FLOOR, i.e. it
    # charges GLM only the IDEAL cross-rank cost.  The measured cross-rank is
    # 1.587.  The whole difference between 7.32 and this file's 8.660 is that
    # one substitution, and it names the first board line exactly.
    # ------------------------------------------------------------------
    EP_FLOOR_MS = 0.383       # demo/glm5/roofline.py, measured
    GUIDE_TARGET = 7.32
    coll_ms = GLM_CLASS_US["coll"] * F
    print("=" * 78)
    print("RECONCILIATION -- the guide's 7.32 vs this file's 8.660")
    print("=" * 78)
    print(f"  guide: 1.936 / 0.279 + {EP_FLOOR_MS} (EP BYTE FLOOR)      "
          f"{GUIDE_TARGET:6.3f} ms")
    print(f"  here:  1.936 / {eff_oss:.3f} + {coll_ms:.3f} (MEASURED cross-rank) "
          f"{target_ms:6.3f} ms")
    print(f"  the difference is one substitution: measured cross-rank minus its")
    print(f"  own floor = {coll_ms:.3f} - {EP_FLOOR_MS:.3f} = "
          f"{coll_ms - EP_FLOOR_MS:.3f} ms\n")
    total_gap = GLM_WALL_MS - GUIDE_TARGET
    print(f"  TOTAL GAP to the v8 target      {GLM_WALL_MS:6.3f} - "
          f"{GUIDE_TARGET:.2f} = {total_gap:6.3f} ms")
    print(f"    line 1  cross-rank headroom above its own floor   "
          f"{coll_ms - EP_FLOOR_MS:6.3f} ms   class 'coll'")
    print(f"    line 2  local gap at gpt-oss's own efficiency     "
          f"{gap:6.3f} ms   NOT in 'tile'")
    print(f"    line 3  rounding (0.279 vs {eff_oss:.3f}; pooled n=8)  "
          f"{total_gap - (coll_ms - EP_FLOOR_MS) - gap:6.3f} ms")
    print()
    print("  Line 2 is NOT in the tile class -- GLM's tiles already beat")
    print("  gpt-oss's whole-layer MB/us.  It is in rendezvous + boundary,")
    oss_layer_us = OSS_WALL_MS * 1000 / OSS_LAYERS
    print(f"  {nontile_ms:.3f} ms at {nontile_us:.1f} us/layer -- which is "
          f"{100 * nontile_us / oss_layer_us:.0f} % of")
    print(f"  gpt-oss's ENTIRE layer ({oss_layer_us:.1f} us), spent on no bytes "
          f"at all.")
    print()


if __name__ == "__main__":
    item12()
    item13()
