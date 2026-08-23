#!/usr/bin/env python3
"""What KIND of bound is the tile class's 3.09x?  ONE answer for the whole class.

THE QUESTION (guide ruling, 2026-08-23).  close_the_wall.py's largest budget
line is the tile class: 69.6 us/layer shipping against a 22.5 us width-corrected
byte roof, 3.09x.  The board has been treating that as nine separate phases and
finding no phase above 0.188 ms.  Treat it as ONE item and ask what resource is
actually binding:

  (A) latency / occupancy-bound  -> HBM near roof, MFMA density low
  (B) bandwidth-bound below peak -> HBM far below roof, and then the
                                    dequant/LDS mechanism has to be NAMED
  (C) neither unit saturated     -> the class is waiting on dependent-issue
                                    latency and outstanding-request depth

The wall cannot resolve this (0.26 ms floor), so nothing here is priced against
the wall.  Two numbers decide it, and both are ratios or absolute counts:

  MFMA ISSUE OCCUPANCY = MFMA busy SIMD-cycles / available SIMD-cycles
  ACHIEVED HBM RATE    = bytes moved / measured tile time

MFMA occupancy is computed here as an UPPER BOUND from exact geometry, not
estimated: at bs=1 every GEMM is M=1, one MFMA covers a 16-row output tile, so
the instruction count is exactly weight_elements / (M_TILE_N * K_TILE) and the
cycle cost per instruction is the repo's own MFMA_CYC=32 (isa_accounting.py).
It is an UPPER bound because it charges an MFMA to phases that provably have
none (glm-wuk-wuv-oproj-have-no-mfma) unless they are listed with mfma=False.

Achieved HBM rate is computed on MODELLED bytes.  That is a LOWER bound on the
real traffic, so it can only be wrong in the direction of "we actually move
more than this" -- which is exactly what the rocprofv3 FetchSize/WriteSize
reduction below tests.  Point it at a counter directory to close that gap:

    python3 demo/glm5/tile_class_bound.py [/tmp/glm5_pmc_bytes]

Byte model and phase table are taken verbatim from tile_roof_by_phase.py so the
3.09x reproduces; the only additions are element counts, MFMA flags and the
alternate-numerics column.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

CONFIG = "/home/claudeuser/models/glm5-mxfp4/config.json"

LAYERS = 76
WALL_MS = 10.619
SHIP_LAYER_US = WALL_MS * 1000.0 / LAYERS          # 139.72
INSTRUMENTED_LAYER_US = 164.947
K = SHIP_LAYER_US / INSTRUMENTED_LAYER_US

# Hardware.  MI355X, gfx950.  CLOCK and MFMA_CYC are isa_accounting.py's own
# constants; HBM_TBS is roofline.py's MEASURED constant -- not re-derived here.
CLOCK_GHZ = 2.4
CUS = 256
SIMDS = CUS * 4
HBM_TBS = 5.17

# Cycles per MFMA instruction and elements covered per instruction.
#   f8f6f4  v_mfma_scale_f32_16x16x128_f8f6f4 : 16x128 B-tile, 32 cycles
#   bf16    v_mfma_f32_16x16x32_bf16          : 16x32  B-tile, 16 cycles
# Both land on the same cycles-per-weight-element at peak (bf16 is half fp8's
# rate, and covers a quarter of the elements), which is the only property the
# occupancy number depends on.
CYC_PER_ELT = {"fp8": 32.0 / (16 * 128), "bf16": 16.0 / (16 * 32)}

# Width-corrected bandwidth, tests/standalone/test_narrow_grid_bandwidth.hip.
BW_CURVE = [(16, 707.0), (32, 1399.0), (64, 2660.0), (96, 3800.0),
            (128, 4765.0), (232, 5373.0), (256, 5365.0)]
XCDS = 8

# MEASURED HBM controller busy, from sample_hbm_activity.sh on a shipping
# NP=8 bs=1 run, keeping only samples with gpu_busy > 50 (the kept samples
# average 97% gpu_busy, so they are on the decode plateau).  Two host-side
# instruments, whole-wall granularity.  They disagree by ~2.5x -- expected,
# since mem_busy_percent is a decimated duty-cycle register read
# asynchronously to the phase structure, and the 1 Hz sampler has n=4.
# Neither is calibrated better than a factor of ~3, and nothing here depends
# on one: the falsification threshold is 3x the prediction, not 1.2x.
#   (label, mean %, max %, samples per GPU)
HBM_SAMPLED = [
    ("sysfs 10 ms  n=8 GPUs", 5.2, 18, 149),   # 5.1-5.7 across the eight
    ("rocm-smi ~1 Hz", 13.0, 16, 4),           # 11.6-14.5 across the eight
]


def bw_at(width: int) -> float:
    width = max(1, min(width, BW_CURVE[-1][0]))
    if width <= BW_CURVE[0][0]:
        return BW_CURVE[0][1] * width / BW_CURVE[0][0]
    for (w0, b0), (w1, b1) in zip(BW_CURVE, BW_CURVE[1:]):
        if width <= w1:
            return b0 + (b1 - b0) * (width - w0) / (w1 - w0)
    return BW_CURVE[-1][1]


def phases():
    c = json.loads(Path(CONFIG).read_text())
    H = c["hidden_size"]
    nh = c["num_attention_heads"]
    q_lora, kv_lora = c["q_lora_rank"], c["kv_lora_rank"]
    qk_nope, qk_rope = c["qk_nope_head_dim"], c["qk_rope_head_dim"]
    v_head = c["v_head_dim"]
    n_routed, moe_inter = c["n_routed_experts"], c["moe_intermediate_size"]
    n_shared = c["n_shared_experts"]
    R, seq = 8, 128

    aq, mq = 1.0 + 1.0 / 32, 0.5 + 1.0 / 32     # mxfp8+scale / mxfp4+scale
    w8 = 1.0                                     # TileRT: int8/fp8 weight, no
                                                 # microscale block
    per_expert = 3 * H * moe_inter
    moe_e = 3.0 * per_expert + n_shared * per_expert   # busiest rank, ELEMENTS
    kv_e = (kv_lora + qk_rope) * seq                   # KV cache elements

    # (label, busy us/lyr instrumented, elements, bytes/elt, tiles/XCD,
    #  mfma format or None, compute_frac, note)
    #
    # busy / tiles / compute_frac: tile_roof_by_phase.py, unchanged.
    # mfma=None where the phase provably issues no MFMA
    # (glm-wuk-wuv-oproj-have-no-mfma); those rows still carry their bytes.
    P = [
        ("qkv_a", 12.058, H * q_lora + H * (kv_lora + qk_rope), aq, 21,
         "fp8", 1.0, ""),
        ("q_b", 12.336 * 0.70, q_lora * nh * (qk_nope + qk_rope) / R, aq, 16,
         "fp8", 1.0, "span 18->19 also holds W_UK/W_UV"),
        ("W_UK", 12.336 * 0.15, kv_lora * nh * qk_nope / R, aq, 16,
         None, 1.0, "NO MFMA"),
        ("W_UV", 12.336 * 0.15, kv_lora * nh * v_head / R, aq, 16,
         None, 1.0, "NO MFMA"),
        ("decode", 9.389, 0.0, 0.0, 2, "bf16", 1.0, "KV only, no weights"),
        ("merge / Ph8", 3.026, 0.0, 0.0, 16, None, 1.0, "no bytes at all"),
        ("o_proj", 7.404, nh * v_head * H / R, aq, 24, None, 1.0, "NO MFMA"),
        ("router", 15.062, H * n_routed, aq, 16, "bf16", 0.46,
         "54% poll measured"),
        ("W13", 18.660, moe_e * 2 / 3, mq, 54, "fp8", 1.0, ""),
        ("W2", 12.407, moe_e * 1 / 3, mq, 108, "fp8", 1.0, ""),
    ]
    # decode's MFMA work is QK^T + PV, not a weight GEMM: account it directly.
    decode_macs = nh * seq * (kv_lora + qk_rope) + nh * seq * v_head
    return P, kv_e, decode_macs, aq, mq, w8


def read_pmc(outdir):
    """Sum rocprofv3 counter values over every profiled worker_kernel dispatch."""
    p = Path(outdir)
    hits = sorted(p.rglob("*counter_collection.csv"))
    if not hits:
        return None
    totals, ndisp = {}, set()
    for f in hits:
        with f.open() as fh:
            for row in csv.DictReader(fh):
                name = row.get("Counter_Name") or row.get("Counter Name")
                val = row.get("Counter_Value") or row.get("Counter Value")
                did = row.get("Dispatch_Id") or row.get("Dispatch Id")
                if not name or val in (None, ""):
                    continue
                totals[name] = totals.get(name, 0.0) + float(val)
                ndisp.add((str(f), did))
    return {"file": str(hits[0]), "n_dispatch": len(ndisp), "totals": totals}


def main() -> int:
    P, kv_e, decode_macs, aq, mq, w8 = phases()

    print("=" * 92)
    print("  TILE CLASS AS ONE ITEM -- what kind of bound is the 3.09x?")
    print(f"  bs=1 NP=8, per layer per rank.  Wall {WALL_MS} ms/token, "
          f"{SHIP_LAYER_US:.2f} us/layer shipping.")
    print("=" * 92)
    print(f"  {'phase':<14} {'Melem':>8} {'MB':>7} {'wrk':>4} {'roof us':>8} "
          f"{'meas us':>8} {'MFMA kcyc':>10}  note")
    print("  " + "-" * 90)

    tot_meas = tot_roof = tot_bytes = tot_elem = 0.0
    mfma_cyc = 0.0
    tot_bytes_w8 = 0.0
    live_simd_cyc = []
    for name, busy, elem, bpe, tiles, fmt, frac, note in P:
        b = elem * bpe
        if name == "decode":
            b = kv_e * 1.0                       # fp8 latent cache
        workers = min(tiles, 29) * XCDS
        roof_us = b / (bw_at(workers) * 1e3)
        meas_us = busy * K * frac
        cyc = (decode_macs if name == "decode" else elem) * \
            CYC_PER_ELT[fmt] if fmt else 0.0
        tot_meas += meas_us
        tot_roof += roof_us
        tot_bytes += b
        tot_elem += elem
        mfma_cyc += cyc
        # LIVE denominator: only this phase's workers hold SIMDs, and a worker
        # is 4 wave64s = one per SIMD (glm-per-cu-roofline-denominator-is-wrong
        # -- dividing every phase by all 1024 SIMDs is the known error).
        live_simd_cyc.append(meas_us * 1e-6 * CLOCK_GHZ * 1e9 * workers * 4)
        # TileRT numerics: 8-bit weights everywhere, bf16 KV.
        tot_bytes_w8 += (kv_e * 2.0) if name == "decode" else elem * w8
        print(f"  {name:<14} {elem/1e6:>8.2f} {b/1e6:>7.2f} {workers:>4} "
              f"{roof_us:>8.2f} {meas_us:>8.2f} {cyc/1e3:>10.1f}  {note}")

    print("  " + "-" * 90)
    print(f"  {'CLASS':<14} {tot_elem/1e6:>8.2f} {tot_bytes/1e6:>7.2f} "
          f"{'':>4} {tot_roof:>8.2f} {tot_meas:>8.2f} {mfma_cyc/1e3:>10.1f}")
    print(f"  ratio measured/roof = {tot_meas/tot_roof:.2f}x"
          f"   ({tot_meas*LAYERS/1e3:.3f} ms vs {tot_roof*LAYERS/1e3:.3f} ms "
          f"over {LAYERS} layers)")

    # ---- the two numbers the ruling asks for -------------------------------
    avail_simd_cyc = tot_meas * 1e-6 * CLOCK_GHZ * 1e9 * SIMDS
    occ_class = 100.0 * mfma_cyc / avail_simd_cyc
    occ_wall = occ_class * tot_meas / SHIP_LAYER_US
    ach_tbs = tot_bytes / (tot_meas * 1e-6) / 1e12
    roof_tbs_flat = HBM_TBS
    roof_tbs_width = tot_bytes / (tot_roof * 1e-6) / 1e12

    print()
    print("=" * 92)
    print("  THE TWO CLASS-LEVEL NUMBERS")
    print("=" * 92)
    occ_live = 100.0 * mfma_cyc / sum(live_simd_cyc)
    print(f"  MFMA issue occupancy over the TILE CLASS   {occ_class:8.2f} %"
          "   (all 1024 SIMDs)")
    print(f"    ... on the LIVE SIMDs of each phase      {occ_live:8.2f} %"
          "   (per-phase width; the fair one)")
    print(f"  MFMA issue occupancy over the WHOLE WALL   {occ_wall:8.2f} %")
    print(f"    MFMA busy      {mfma_cyc/1e6:7.3f} M SIMD-cycles/layer/rank"
          "   (REQUIRED work -- see docstring)")
    print(f"    available      {avail_simd_cyc/1e6:7.1f} M SIMD-cycles "
          f"({SIMDS} SIMDs x {tot_meas:.1f} us x {CLOCK_GHZ} GHz)")
    print(f"    live           {sum(live_simd_cyc)/1e6:7.1f} M SIMD-cycles")
    print()
    print(f"  achieved HBM rate, MODELLED bytes         {ach_tbs:8.3f} TB/s")
    print(f"    vs flat peak            {roof_tbs_flat:6.2f} TB/s -> "
          f"{100*ach_tbs/roof_tbs_flat:5.1f} % of peak")
    print(f"    vs width-corrected roof {roof_tbs_width:6.2f} TB/s -> "
          f"{100*ach_tbs/roof_tbs_width:5.1f} % of achievable")

    # ---- measured HBM controller busy vs what the byte model predicts ------
    # The achieved rate above divides MODELLED bytes by MEASURED time, so it is
    # only a lower bound on real traffic: if the kernel re-read ~3x the model,
    # the class would be AT the roof and the verdict would invert.  These are
    # sysfs mem_busy_percent samples from sample_hbm_activity.sh -- host-side,
    # zero device perturbation, whole-wall granularity.
    pred_busy = 100.0 * (tot_bytes * LAYERS) / (WALL_MS * 1e-3) / (HBM_TBS * 1e12)
    print()
    print(f"  HBM controller busy over the WHOLE WALL, predicted"
          f"  {pred_busy:5.1f} %")
    print(f"    ({tot_bytes*LAYERS/1e9:.2f} GB of tile-class weights per iter per rank")
    print(f"     / {WALL_MS} ms / {HBM_TBS} TB/s -- a LOWER bound on wall traffic:")
    print("     KV, activations and the collectives are not in this byte model)")
    for label, mean, mx, n in HBM_SAMPLED:
        print(f"    measured, {label:<22} {mean:5.1f} %   max {mx:.0f} %   n={n}/GPU")
    print("    the escape being tested -- 3x hidden re-reads, i.e. the class at")
    print(f"    the roof -- needs ~{3*pred_busy:.0f} %.  Not approached, even instantaneously.")

    # ---- rocprofv3 reduction, if a counter dir was given -------------------
    outdir = sys.argv[1] if len(sys.argv) > 1 else None
    pmc = read_pmc(outdir) if outdir else None
    print()
    print("=" * 92)
    print("  MEASURED COUNTERS")
    print("=" * 92)
    if not pmc:
        print(f"  no counter CSV under {outdir!r} -- the two numbers above are")
        print("  the analytic bound only.  Run profile_tile_class.sh first.")
    else:
        print(f"  {pmc['file']}   ({pmc['n_dispatch']} profiled dispatches)")
        t = pmc["totals"]
        for k in sorted(t):
            print(f"    {k:<34} {t[k]:>20,.0f}")
        gui = t.get("GRBM_GUI_ACTIVE")
        mf = t.get("SQ_VALU_MFMA_BUSY_CYCLES")
        if gui and mf:
            print(f"\n    MfmaUtil = MFMA_BUSY / (GUI_ACTIVE * {SIMDS}) = "
                  f"{100.0*mf/(gui*SIMDS):.3f} %  over the whole worker kernel")
            print(f"    over the tile class (tile share "
                  f"{tot_meas/SHIP_LAYER_US:.3f}) = "
                  f"{100.0*mf/(gui*SIMDS)*SHIP_LAYER_US/tot_meas:.3f} %")
        fetch = t.get("FetchSize")
        write = t.get("WriteSize")
        if fetch is not None:
            actual = (fetch + (write or 0.0)) * 1024.0   # KB -> bytes
            print(f"\n    HBM traffic, whole kernel {actual/1e9:.2f} GB")
            print(f"    modelled tile bytes       "
                  f"{tot_bytes*LAYERS/1e9:.2f} GB/iter/rank over {LAYERS} layers")
            print(f"    ACTUAL / MODELLED = "
                  f"{actual/(tot_bytes*LAYERS):.2f}x  "
                  "(>2.5x would mean we ARE near the HBM roof and the class is "
                  "latency-bound on real traffic; ~1x means the bytes are not "
                  "the story)")

    # ---- step 3: the numerics asymmetry, one line --------------------------
    print()
    print("=" * 92)
    print("  NUMERICS ASYMMETRY -- our byte roof vs TileRT's (W8A16 / BF16)")
    print("=" * 92)
    ours_ms = tot_bytes * LAYERS / (HBM_TBS * 1e12) * 1e3
    theirs_ms = tot_bytes_w8 * LAYERS / (HBM_TBS * 1e12) * 1e3
    print(f"  ours   mxfp8 attn ({aq:.5f} B/elt) + mxfp4 MoE ({mq:.5f} B/elt)"
          f"   {tot_bytes/1e6:7.2f} MB/lyr -> {ours_ms:.3f} ms/token at "
          f"{HBM_TBS} TB/s")
    print(f"  theirs 8-bit weights everywhere ({w8:.5f} B/elt), bf16 KV      "
          f"   {tot_bytes_w8/1e6:7.2f} MB/lyr -> {theirs_ms:.3f} ms/token at "
          f"{HBM_TBS} TB/s")
    print(f"  ratio  theirs / ours = {tot_bytes_w8/tot_bytes:.2f}x")
    print()
    print(f"  ONE LINE: our numerics make the tile-class byte roof "
          f"{tot_bytes_w8/tot_bytes:.2f}x SMALLER than")
    print(f"  TileRT's, not larger.  At {ours_ms:.3f} ms/token the byte roof "
          "sits UNDER the 2 ms goal,")
    print("  so MXFP4-weights + FP8-activations is not what makes 2 ms "
          "unreachable -- it is")
    print("  the only reason 2 ms is arithmetically on the table at all.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
