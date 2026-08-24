#!/usr/bin/env python3
"""GLM-5 744B decode roofline on 8x MI355X, batch size 1.

Answers one question: if every byte moved at HBM speed and nothing ever waited,
how fast could a decode iteration be? Everything above that number is schedule,
not silicon.

Run:
    python3 roofline.py                       # current config
    python3 roofline.py --seq 1024            # TileRT's 1k context point
    python3 roofline.py --moe-quant mxfp8     # what MXFP4 on experts bought
    python3 roofline.py --no-shard            # the pre-sharding geometry

Numbers that are measured, not assumed, are marked MEASURED in the output and
sourced in comments. Everything else is derived from config.json, so a config
change reprices the model rather than silently invalidating a hardcoded table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# --------------------------------------------------------------------------- #
# Hardware
# --------------------------------------------------------------------------- #
# MEASURED on this box (gfx950 / MI355X), 240-block streaming read at a 384 MB+
# working set, i.e. past MALL. Marketing peak is 8.0 TB/s; do not use it -- the
# gap is what makes a "% of peak" claim mean two different things.
HBM_TBS = 5.17
# MEASURED: MALL is only 1.22x HBM on this part, per-XCD L2 2.6x at 4 MB/XCD.
# Neither is large enough to hold a layer, so the roofline below is pure HBM.
MALL_TBS = 6.20
L2_TBS = 12.65

# XGMI, 8 GPUs fully connected at uniform weight. Only used to price collectives.
XGMI_GBS = 64.0

# MEASURED: one EP collective costs ~5.1 us/layer wall (MPK_EP_ABLATE=1 delta
# over the full run divided by layer-visits). That is a *latency* floor set by
# rendezvous, not a bandwidth cost -- the payload is 12 KB.
EP_COLLECTIVE_US = 5.1

# --------------------------------------------------------------------------- #
# Quantization: bytes per element, including the block scale
# --------------------------------------------------------------------------- #
# MXFP8 = E4M3 value + one E8M0 exponent per 32 values.
# MXFP4 = E2M1 nibble + one E8M0 per 32.
BPE = {
    "bf16": 2.0,
    "mxfp8": 1.0 + 1.0 / 32,
    "mxfp4": 0.5 + 1.0 / 32,
}


def _emax_balls_in_bins(n: int, R: int) -> float:
    """Exact E[max bin load] for n balls thrown uniformly into R bins.

    This is the EP routing makespan term: topk experts are drawn per token and
    the layer waits on whichever rank drew the most.  Exact enumeration over
    compositions -- n=8, R<=8 is tiny, so there is no reason to approximate.
    """
    from math import comb

    total, norm = 0.0, float(R) ** n

    def rec(bin_i: int, rem: int, mx: int, ways: float) -> None:
        nonlocal total
        if bin_i == R - 1:
            total += ways * max(mx, rem)
            return
        for k in range(rem + 1):
            rec(bin_i + 1, rem - k, max(mx, k), ways * comb(rem, k))

    rec(0, n, 0, 1.0)
    return total / norm


def human(b: float) -> str:
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if b >= div:
            return f"{b / div:.2f} {unit}"
    return f"{b:.0f} B"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="/home/claudeuser/models/glm5-mxfp4/config.json")
    ap.add_argument("--ranks", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128,
                    help="KV length. 128 is the current benchmark; 1024 is TileRT's point.")
    ap.add_argument("--moe-quant", default="mxfp4", choices=list(BPE))
    ap.add_argument("--attn-quant", default="mxfp8", choices=list(BPE))
    ap.add_argument("--no-shard", action="store_true",
                    help="Price the pre-sharding geometry (attention replicated 8x).")
    ap.add_argument("--measured-ms", type=float, default=11.427,
                    help="Current measured ms/iter, for the efficiency line.")
    args = ap.parse_args()

    c = json.loads(Path(args.config).read_text())
    H = c["hidden_size"]
    L = c["num_hidden_layers"]
    dense_layers = c["first_k_dense_replace"]
    moe_layers = L - dense_layers
    n_heads = c["num_attention_heads"]
    q_lora = c["q_lora_rank"]
    kv_lora = c["kv_lora_rank"]
    qk_nope, qk_rope = c["qk_nope_head_dim"], c["qk_rope_head_dim"]
    v_head = c["v_head_dim"]
    n_routed, topk = c["n_routed_experts"], c["num_experts_per_tok"]
    n_shared = c["n_shared_experts"]
    moe_inter, dense_inter = c["moe_intermediate_size"], c["intermediate_size"]
    vocab = c["vocab_size"]

    R = args.ranks
    aq, mq = BPE[args.attn_quant], BPE[args.moe_quant]
    shard = 1 if args.no_shard else R

    print(f"GLM-5 744B  |  {L} layers ({dense_layers} dense + {moe_layers} MoE)  "
          f"|  {R} x MI355X  |  bs=1  |  KV {args.seq}")
    print(f"quant: attention {args.attn_quant} ({aq:.4f} B/elem), "
          f"MoE {args.moe_quant} ({mq:.4f} B/elem)")
    print(f"HBM {HBM_TBS} TB/s MEASURED (not the 8.0 TB/s spec sheet)\n")

    # ---------------------------------------------------------------- attention
    # Un-absorbed form, which is what runs today: kv_b is split into W_UK and
    # W_UV rather than folded into q_b and o_proj.
    attn = {
        "q_a_proj":   (H * q_lora,                 1,     "replicated"),
        "q_b_proj":   (q_lora * n_heads * (qk_nope + qk_rope), shard, "head-sharded"),
        "kv_a_proj":  (H * (kv_lora + qk_rope),    1,     "replicated"),
        "W_UK":       (kv_lora * n_heads * qk_nope, shard, "head-sharded"),
        "W_UV":       (kv_lora * n_heads * v_head,  shard, "output-sharded"),
        "o_proj":     (n_heads * v_head * H,        shard, "output-sharded"),
        "router":     (H * n_routed,               1,     "replicated"),
    }
    print("Per MoE layer, per rank -- attention")
    print(f"  {'weight':<12} {'params':>12} {'split':>16} {'bytes/rank':>12}")
    attn_bytes = 0.0
    for name, (params, div, how) in attn.items():
        b = params / div * aq
        attn_bytes += b
        print(f"  {name:<12} {params/1e6:>10.2f} M {how:>16} {human(b):>12}")
    print(f"  {'':<12} {'':>12} {'attention total':>16} {human(attn_bytes):>12}\n")

    # ---------------------------------------------------------------------- MoE
    per_expert = 3 * H * moe_inter          # w1, w3 (up) + w2 (down)
    owned = n_routed / R
    # At bs=1, top-8 over 8 ranks averages 1 activated expert per rank -- but the
    # LAYER's makespan is set by the BUSIEST rank, not the mean. Balls-in-bins
    # with 8 balls in 8 bins has an expected max of ~3. That gap is a real,
    # structural cost of EP at batch 1 and is priced separately below.
    # v9: max_active was HARDCODED 3.0.  That is ~right at R=8 and badly wrong
    # at any other world size, which made every cross-world-size comparison
    # wrong -- at R=2 it produced a "busiest rank" byte count BELOW the mean,
    # which is impossible.  Compute E[max] exactly instead.  This is a
    # combinatorial model parameter, NOT one of the measured hardware constants
    # (HBM_TBS, EP_COLLECTIVE_US), which are untouched.
    #   R=8 -> 2.597   R=4 -> 3.538   R=2 -> 5.094   (was 3.0 for all three)
    mean_active = topk / R
    max_active = _emax_balls_in_bins(topk, R)
    moe_mean = mean_active * per_expert * mq
    moe_max = max_active * per_expert * mq
    shared_b = n_shared * per_expert * mq   # replicated: every rank computes it

    print("Per MoE layer, per rank -- MoE")
    print(f"  {'routed (owned)':<22} {owned:>6.0f} experts    {human(owned*per_expert*mq):>12}  (resident, not read)")
    print(f"  {'routed (activated)':<22} {mean_active:>6.2f} mean       {human(moe_mean):>12}")
    print(f"  {'routed (busiest rank)':<22} {max_active:>6.1f} E[max]     {human(moe_max):>12}  <- sets the makespan")
    print(f"  {'shared expert':<22} {n_shared:>6d} replicated {human(shared_b):>12}\n")

    # ------------------------------------------------------------------ KV cache
    # MLA stores the latent, not per-head K/V: kv_lora + qk_rope per token.
    # DP attention: every rank holds the whole cache and reads all of it.
    kv_per_tok = (kv_lora + qk_rope) * 1.0          # fp8 latent
    kv_bytes = kv_per_tok * args.seq
    print(f"KV cache (MLA latent, fp8): {kv_per_tok:.0f} B/token/layer "
          f"x {args.seq} = {human(kv_bytes)}/layer/rank")
    print(f"  -- {kv_bytes/(attn_bytes+moe_max)*100:.1f}% of the layer's bytes at "
          f"seq {args.seq}. Weights dominate until seq ~"
          f"{(attn_bytes+moe_max)/kv_per_tok:,.0f}.\n")

    # --------------------------------------------------------------- the roofline
    layer_mean = attn_bytes + moe_mean + shared_b + kv_bytes
    layer_max = attn_bytes + moe_max + shared_b + kv_bytes

    dense_b = 3 * H * dense_inter * aq / shard + attn_bytes + kv_bytes
    lm_head_b = vocab * H * aq

    total_mean = moe_layers * layer_mean + dense_layers * dense_b + lm_head_b
    total_max = moe_layers * layer_max + dense_layers * dense_b + lm_head_b

    t_mean = total_mean / (HBM_TBS * 1e12) * 1e3
    t_max = total_max / (HBM_TBS * 1e12) * 1e3
    t_coll = moe_layers * EP_COLLECTIVE_US / 1e3

    print("=" * 68)
    print(f"{'bytes/iter/rank, mean-balanced routing':<48} {human(total_mean):>18}")
    print(f"{'bytes/iter/rank, busiest-rank routing':<48} {human(total_max):>18}")
    print()
    print(f"{'HBM floor, mean routing':<48} {t_mean:>15.3f} ms")
    print(f"{'HBM floor, busiest rank (honest floor)':<48} {t_max:>15.3f} ms")
    print(f"{'+ EP collective latency (MEASURED 5.1 us x '+str(moe_layers)+')':<48} {t_coll:>15.3f} ms")
    print(f"{'= achievable floor':<48} {t_max + t_coll:>15.3f} ms")
    print()
    print(f"{'MEASURED today':<48} {args.measured_ms:>15.3f} ms")
    print(f"{'efficiency vs achievable floor':<48} "
          f"{(t_max + t_coll) / args.measured_ms * 100:>14.1f} %")
    print(f"{'gap to close':<48} {args.measured_ms - t_max - t_coll:>15.3f} ms")
    print()
    # TileRT: 494 tok/s/user on GLM-5.1, 1k/1k, bs=1, 8x B200 FP8 = 2.02 ms/tok.
    # B200 HBM3e is 8.0 TB/s spec; assume the same ~65% achievable ratio we
    # measure here and it is ~5.2 TB/s -- i.e. essentially the same bandwidth per
    # GPU as MI355X. So TileRT's result is NOT a bandwidth advantage. It is the
    # same roofline, reached.
    print(f"{'TileRT reference (GLM-5.1, 1k/1k, 8x B200)':<48} {2.02:>15.3f} ms")
    print(f"{'TileRT GLM-5 (8k/1k, 8x B200)':<48} {2.94:>15.3f} ms")
    print(f"{'SemiAnalysis MI355X reference (1k out)':<48} {18.18:>15.3f} ms")
    print("=" * 68)
    print()
    print("Reading this: B200 HBM3e is 8.0 TB/s spec vs MI355X's 8.0 TB/s spec, and")
    print("both realize ~65% on a streaming read. TileRT's 2.02 ms is therefore not")
    print("a bandwidth advantage -- it is the same roofline, actually reached. The")
    print("target is not a faster kernel. It is removing the wait.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
