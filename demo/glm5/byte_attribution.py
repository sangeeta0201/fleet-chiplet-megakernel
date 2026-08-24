#!/usr/bin/env python3
"""RE-STEER v9, Item 2.1: WHAT IS THE 9.233 ms MADE OF?

Crosses two MEASUREMENTS that already exist.  Nothing here is a new run.

  (a) the NP=4 per-phase table -- 23 regions, closing to 99.5 % of the
      11.418 ms wall (memory: glm-np4-per-phase-breakdown, produced by
      demo/glm5/ws_phase.py off a MPK_BAR_SKEW=3 log, crit_us de-inflated
      by 0.872 for the probe's own cost);
  (b) the marginal byte rate, 0.911 ms of wall per ms of HBM roof, measured
      at FIXED NP=4 by the lossless MXFP4 -> MXFP8 expert widen
      (demo/glm5/world_size_fit.py, 11.418 -> 12.550 ms n=3).

Region byte counts come from roofline.py --ranks 4 at the same config.

WHAT THIS IS AND IS NOT.  It is a decomposition of two measurements, so every
row inherits their error bars and nothing below is independently measured.
It is NOT a price list: a region's byte-independent residue is what would
remain if its bytes were free, which is not the same as what deleting it
buys.  Price by ablation before building anything
(memory: price-by-ablation-never-by-roofline-or-op-name).

Run:  python3 demo/glm5/byte_attribution.py
"""

WALL = 11.418            # NP=4, bs=1, n=3, texts checked
BYTE_SLOPE = 0.911       # ms wall per ms of HBM roof, measured at fixed NP=4
HBM_TBS = 5.17           # MEASURED, roofline.py
MOE_LAYERS = 75          # the byte model's layer count
TABLE_LAYERS = 76        # the phase table's layer count (11.418 / 150.2 us)


def roof_ms(mb_per_layer: float) -> float:
    """HBM roof, ms/iteration, for X MB read once per MoE layer per rank."""
    return mb_per_layer * 1e6 * MOE_LAYERS / (HBM_TBS * 1e12) * 1e3


# ---------------------------------------------------------------------------
# The 23 regions.  (name, measured ms/76L, MB/layer/rank, byte source)
#
# Byte attribution rules, stated so they can be argued with:
#   * a GEMM region carries its own weight bytes and nothing else;
#   * routed-expert bytes are the BUSIEST-rank figure (70.95 MB, E[max]=3.538)
#     split 2:1 W13:W2, because per expert w1+w3 = 2*H*inter and w2 = H*inter;
#   * a BARRIER region carries ZERO bytes.  That is the whole point of the
#     exercise -- it is what makes them 100 % byte-independent by construction;
#   * the EP collective's payload is 12 KB (roofline.py's own comment) and it
#     is priced as a LATENCY floor, not bandwidth, so it carries zero;
#   * W_UV (2.16 MB) and the shared expert (20.05 MB) have NO row of their own
#     in the phase table -- W_UV lives inside the S24..S27 nested stamps and
#     the shared expert inside the MoE half.  They are listed as UNPLACED
#     below rather than smeared into a neighbour.
# ---------------------------------------------------------------------------
REGIONS = [
    ("W13 tiles (S5->S6)",                    1.514, 47.30, "routed x 2/3"),
    ("EP collective (fold+peer wait+release)", 1.197, 0.00, "12 KB payload; latency"),
    ("W2 tiles (S7->S8)",                     1.131, 23.65, "routed x 1/3"),
    ("P4+5 q_b->decode BARRIER + MLA decode", 1.040,  0.07, "KV latent, seq 128"),
    ("P3 q_b: RMSNorm + absorbed q_b + KV app", 1.031, 10.27, "q_b 8.65 + W_UK 1.62"),
    ("P11 post-attn RMSNorm + router + TopK",  0.862,  1.62, "router"),
    ("P9 o_proj GEMV + residual",              0.771, 25.95, "o_proj"),
    ("P1 qkv_a: resid+RMSNorm+[q_a|kv_a]",     0.666, 16.63, "q_a 12.98 + kv_a 3.65"),
    ("MoE entry -> o_proj worker set entered", 0.616,  0.00, "dispatch"),
    ("W13->W2 BARRIER wait (S6->S7)",          0.425,  0.00, "barrier"),
    ("P6+7 decode->merge BARRIER + merge",     0.367,  0.00, "barrier + LDS merge"),
    ("P2 qkv_a->q_b BARRIER",                  0.363,  0.00, "barrier"),
    ("routing poll -> MoE entry",              0.256,  0.00, "poll"),
    ("MoE exit -> bookkeeping",                0.229,  0.00, "bookkeeping"),
    ("P8 attn->o_proj GPU-wide BARRIER",       0.192,  0.00, "barrier"),
    ("prefetch issue / DMA retire / half exit", 0.176, 0.00, "nested S25-27"),
    ("per-XCD attention release wait",         0.142,  0.00, "barrier"),
    ("P10 o_proj->router BARRIER arrival",     0.117,  0.00, "barrier"),
    ("attn-half entry (dispatch)",             0.115,  0.00, "dispatch"),
    ("loop top",                               0.103,  0.00, "loop"),
    ("pointer refresh",                        0.051,  0.00, "pointer fetch"),
]

UNPLACED_MB = {"W_UV (inside S24..S27)": 2.16, "shared expert (inside MoE half)": 20.05}

# Bytes inside the measured wall but OUTSIDE the 76-layer loop the phase table
# walks: the 3 dense layers and the LM head.  roofline.py --ranks 4 puts the
# whole-iteration roof at 2.399 ms; the 75-MoE-layer part is 2.143.
OUT_OF_LOOP_ROOF = 0.256


def main() -> None:
    print("=" * 92)
    print("ITEM 2.1 -- THE 11.418 ms NP=4 WALL, SPLIT INTO BYTES AND NOT-BYTES")
    print("=" * 92)
    print(f"\n  byte rate {BYTE_SLOPE} ms wall per ms of HBM roof "
          f"(MEASURED at fixed NP=4, world_size_fit.py)")
    print(f"  1 MB/layer/rank = {roof_ms(1.0):.4f} ms of roof over "
          f"{MOE_LAYERS} MoE layers\n")

    hdr = (f"  {'region':<42}{'meas':>7}{'MB/l':>7}{'roof':>7}"
           f"{'bytes':>7}{'FIXED':>8}{'x roof':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    tot_meas = tot_roof = tot_byte = tot_fixed = 0.0
    rows = []
    for name, meas, mb, _src in REGIONS:
        rf = roof_ms(mb)
        byte = BYTE_SLOPE * rf
        fixed = meas - byte
        rows.append((name, meas, mb, rf, byte, fixed))
        tot_meas += meas
        tot_roof += rf
        tot_byte += byte
        tot_fixed += fixed

    for name, meas, mb, rf, byte, fixed in rows:
        ratio = f"{meas / rf:>7.2f}x" if rf > 0.002 else f"{'inf':>8}"
        print(f"  {name:<42}{meas:>7.3f}{mb:>7.2f}{rf:>7.3f}"
              f"{byte:>7.3f}{fixed:>8.3f}{ratio}")

    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'TOTAL (23 regions)':<42}{tot_meas:>7.3f}"
          f"{sum(r[2] for r in rows):>7.2f}{tot_roof:>7.3f}"
          f"{tot_byte:>7.3f}{tot_fixed:>8.3f}"
          f"{tot_meas / tot_roof:>7.2f}x")
    print()

    unplaced_mb = sum(UNPLACED_MB.values())
    print(f"  UNPLACED BYTES -- real reads with no region of their own:")
    for k, v in UNPLACED_MB.items():
        print(f"    {k:<40}{v:>7.2f} MB/l  = {roof_ms(v):.3f} ms of roof")
    print(f"    {'total':<40}{unplaced_mb:>7.2f} MB/l  = "
          f"{roof_ms(unplaced_mb):.3f} ms of roof")
    print(f"  They are inside SOME region's measured time, so the per-region")
    print(f"  FIXED column above is an OVERSTATEMENT by up to "
          f"{BYTE_SLOPE * roof_ms(unplaced_mb):.3f} ms in total.")
    print()
    print(f"  OUT-OF-LOOP BYTES -- 3 dense layers + LM head: "
          f"{OUT_OF_LOOP_ROOF:.3f} ms of roof.")
    print(f"  Inside the measured 11.418 but outside the 76-layer table.")
    print()
    full_roof = tot_roof + roof_ms(unplaced_mb) + OUT_OF_LOOP_ROOF
    print(f"  BYTE ACCOUNTING CLOSES: {tot_roof:.3f} placed + "
          f"{roof_ms(unplaced_mb):.3f} unplaced + {OUT_OF_LOOP_ROOF:.3f} out-of-loop")
    print(f"  = {full_roof:.3f} ms, vs roofline.py --ranks 4's whole-iteration "
          f"2.399 ms.  Every byte is on the table.")
    print()

    # ---------------------------------------------------------------- the split
    print("=" * 92)
    print("THE SPLIT")
    print("=" * 92)
    table_fixed = tot_meas - BYTE_SLOPE * full_roof
    print(f"""
  from the 23-region table   {tot_meas:.3f} meas - {BYTE_SLOPE} x {full_roof:.3f} roof
                             byte-attributable {BYTE_SLOPE * full_roof:.3f},
                             byte-INDEPENDENT  {table_fixed:.3f} ms
  from the widen regression  {WALL:.3f} meas - {BYTE_SLOPE} x 2.399 roof
                             byte-attributable {WALL - 9.233:.3f},
                             byte-INDEPENDENT  {9.233:.3f} ms

  THESE TWO ARE NOT INDEPENDENT AND I AM NOT GOING TO CLAIM THEY ARE.  Both
  subtract the same {BYTE_SLOPE} x 2.399; the {abs(9.233 - table_fixed):.3f} ms between them is nothing but
  the phase table's own 99.5 % closure ({WALL:.3f} - {tot_meas:.3f}).  The table's
  contribution is the RANKING below, not a second estimate of the total.

  (Charge only the PLACED bytes and the table reads {tot_fixed:.3f}, which is why
  the per-region FIXED column sums high.  The 9.2 figure is the one to
  quote; the per-region column is for RANKING, not for totalling.)
""")

    # -------------------------------------------------- the slope-free bound
    zero_byte = [r for r in rows if r[2] == 0.0]
    zb = sum(r[5] for r in zero_byte)
    print("=" * 92)
    print("A LOWER BOUND THAT OWES THE 0.911 SLOPE NOTHING")
    print("=" * 92)
    print(f"""
  {len(zero_byte)} of the 23 regions read ZERO weight bytes -- they are barriers,
  polls, dispatch and bookkeeping.  Their measured time is {zb:.3f} ms, and
  that figure is independent of the byte model entirely: it does not use
  0.911, it does not use HBM_TBS, it does not use roofline.py.  Whatever the
  marginal byte rate turns out to be, at least {zb:.3f} ms of the {WALL:.3f} ms wall
  ({100 * zb / WALL:.0f} %) cannot be bought with bandwidth.

  The widen regression says {9.233:.3f}.  The slope-free floor says {zb:.3f}.  The
  {9.233 - zb:.3f} ms between them is the byte-independent residue sitting INSIDE the
  seven GEMM regions -- load latency and VALU, not bytes and not MFMA.  That
  gap is what Item 2.2 has to explain.
""")

    # ------------------------------------------------------------- the ranking
    print("=" * 92)
    print("THE BYTE-INDEPENDENT 9.2 ms, RANKED.  THIS IS THE ITEM 2 BOARD.")
    print("=" * 92)
    BUCKET = {
        "BARRIER / rendezvous / poll": [
            "EP collective (fold+peer wait+release)",
            "W13->W2 BARRIER wait (S6->S7)",
            "P6+7 decode->merge BARRIER + merge",
            "P2 qkv_a->q_b BARRIER",
            "routing poll -> MoE entry",
            "P8 attn->o_proj GPU-wide BARRIER",
            "per-XCD attention release wait",
            "P10 o_proj->router BARRIER arrival",
        ],
        "DISPATCH / bookkeeping / loop": [
            "MoE entry -> o_proj worker set entered",
            "MoE exit -> bookkeeping",
            "prefetch issue / DMA retire / half exit",
            "attn-half entry (dispatch)",
            "loop top",
            "pointer refresh",
        ],
        "TILE residue (latency + VALU inside a GEMM)": [
            "W13 tiles (S5->S6)",
            "W2 tiles (S7->S8)",
            "P4+5 q_b->decode BARRIER + MLA decode",
            "P3 q_b: RMSNorm + absorbed q_b + KV app",
            "P11 post-attn RMSNorm + router + TopK",
            "P9 o_proj GEMV + residual",
            "P1 qkv_a: resid+RMSNorm+[q_a|kv_a]",
        ],
    }
    by_name = {r[0]: r for r in rows}
    print()
    for bucket, names in BUCKET.items():
        sub = sum(by_name[n][5] for n in names)
        print(f"  {bucket:<48}{sub:>8.3f} ms  "
              f"({100 * sub / tot_fixed:>4.1f} % of fixed)")
        for n in sorted(names, key=lambda x: -by_name[x][5]):
            print(f"      {n:<44}{by_name[n][5]:>8.3f}")
    print()
    print(f"""  READING IT.

  1. NO SINGLE BARRIER IS THE BOARD.  Eight barrier/poll regions sum to
     {sum(by_name[n][5] for n in BUCKET['BARRIER / rendezvous / poll']):.3f} ms, and the largest -- the EP collective at 1.197 -- is
     already measured out as a mechanism at every angle tried
     (glm-barrier-narrowing-is-measured-out, ...-deleting-a-whole-rendezvous-
     is-neutral, ...-per-xcd-barrier-narrowing-is-zero-by-measurement,
     ...-all-thread-barrier-polling-is-a-negative).  This bucket is BIG and
     it is CLOSED as a mechanism; what is left in it is arrival spread, i.e.
     the last-arriver floor, which is a property of the OTHER regions.

  2. THE TILE RESIDUE IS THE LARGEST BUCKET at {sum(by_name[n][5] for n in BUCKET['TILE residue (latency + VALU inside a GEMM)']):.3f} ms.  It is not bytes and
     it is not MFMA (2.65 % occupancy).  It is load LATENCY plus the VALU
     prologues -- exactly the class already measured at 3.09x over its own
     byte roof.  This is where Item 2.2 goes.

  3. DISPATCH is {sum(by_name[n][5] for n in BUCKET['DISPATCH / bookkeeping / loop']):.3f} ms and is the only bucket nobody has attacked.
     Six regions, none above 0.616, all pure overhead.

  4. THE ONE STRUCTURAL ANOMALY survives this cut: P4+5 is {by_name['P4+5 q_b->decode BARRIER + MLA decode'][5]:.3f} ms of
     which ~0 is bytes, and its crit/typ is 9.0x -- ~16 workers do 15.7 us
     while ~216 spin.  Every other region is inside 1.6x.  That is the
     highest fixed-ms-per-unit-of-imbalance in the layer.
""")


if __name__ == "__main__":
    main()
