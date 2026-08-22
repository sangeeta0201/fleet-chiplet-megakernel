#!/usr/bin/env python3
"""GLM-5 / GLM-4.7-Flash MLA + MoE inference demo on MI350/MI355 using the
Mirage persistent kernel.

Usage:
    # PyTorch reference (no Mirage):
    HIP_VISIBLE_DEVICES=0 python3 demo/glm5/demo.py --model-path <path>

    # Mirage persistent kernel:
    HIP_VISIBLE_DEVICES=0 python3 demo/glm5/demo.py --model-path <path> --use-mirage

Both checkpoints share one architecture (absorbed MLA + a ``noaux_tc`` sigmoid
router), so the same task graph builds for either; only the shapes differ.
``--max-layers`` truncates both the Torch reference and the task graph, which
is how a 744B checkpoint is brought up on one GPU.
"""

from models.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM, GlmMoeDsaConfig
from transformers import AutoTokenizer
import torch
import torch.distributed as dist
import argparse
import os
import math
import json

# Local model directory or HF repo id. Override with --model-path or the
# GLM_MODEL_PATH env var.
DEFAULT_MODEL_PATH = os.environ.get("GLM_MODEL_PATH", "zai-org/GLM-4.7-Flash")

# CI correctness-dump defaults. Torch vs Mirage token dumps land here for
# tests/ci-tests/test_glm5_inference_output.py.
DEFAULT_SAVE_DIR = os.path.join("outputs", "glm5")
# The dump is truncated to this many generated tokens. Overridable because the
# multi-prompt sweep generates 256: leave it at 100 and a divergence past token
# 100 never reaches the JSON, so the comparison silently passes.
MAX_SAVE_TOKENS = int(os.environ.get("MAX_SAVE_TOKENS", "100"))


# ── Shape helpers ────────────────────────────────────────────────────────────
# The gang GEMM path has two hard divisibility rules, both inherited from
# linear_kernel_ck's tile shape (see gang_linear_layer in persistent_kernel.py):
#
#   * NPerBlock = 64, and the layer splits the output into 8 XCD chunks before
#     tiling, so  output_size % (8 * 64) == 0  i.e. a multiple of 512.
#   * KPerBlock = 256 at batch <= 16, so the reduction dim must be % 256.
#
# Everything padded below is padded to satisfy one of those two.
GANG_OUT_ALIGN = 512
GANG_RED_ALIGN = 256


def align_up(x: int, a: int) -> int:
    return ((x + a - 1) // a) * a


def max_factor_leq_n(m: int, n: int) -> int:
    """Largest factor of m that is <= n. Used for argmax grid sizing, where
    the partial task count must divide the (padded) vocab exactly."""
    max_factor = 1
    i = 1
    while i * i <= m:
        if m % i == 0:
            if i <= n:
                max_factor = max(max_factor, i)
            if m // i <= n:
                max_factor = max(max_factor, m // i)
        i += 1
    return max_factor


def pad_rows(w: torch.Tensor, target_rows: int) -> torch.Tensor:
    """Zero-pad a [out, in] weight along `out`. Padded output columns are then
    exactly zero, which is what makes a padded RMSNorm reduction safe: the
    extra columns contribute nothing to the sum of squares."""
    if w.shape[0] == target_rows:
        return w.contiguous()
    assert w.shape[0] < target_rows
    pad = torch.zeros(target_rows - w.shape[0], w.shape[1],
                      dtype=w.dtype, device=w.device)
    return torch.cat([w, pad], dim=0).contiguous()


def pad_cols(w: torch.Tensor, target_cols: int) -> torch.Tensor:
    """Zero-pad a [out, in] weight along `in` (the reduction dim)."""
    if w.shape[1] == target_cols:
        return w.contiguous()
    assert w.shape[1] < target_cols
    pad = torch.zeros(w.shape[0], target_cols - w.shape[1],
                      dtype=w.dtype, device=w.device)
    return torch.cat([w, pad], dim=1).contiguous()


def quantize_mxfp8(w: torch.Tensor) -> tuple:
    """Quantize a [..., out, K] bf16 weight to MXFP8: E4M3 values with one
    E8M0 exponent per 32 contiguous K elements.

    Contiguous-32 is not a free choice. The scaled MFMA takes one scale per
    lane, and it addresses that operand by matrix position rather than by lane
    contents -- lane 16*g+m carries row m's exponent for K block
    [g*32, g*32+32), whatever bytes the lane's data register happens to hold.
    Grouping any other way silently mixes exponents between element groups and
    reads as elevated quantization noise. See
    tests/standalone/test_mxfp8_mfma_layout.hip.

    Returns (data uint8 [..., out, K], scales uint8 [..., out, K/32]).
    """
    K = w.shape[-1]
    assert K % 32 == 0, f"K {K} must be a multiple of 32"
    wf = w.float().reshape(*w.shape[:-1], K // 32, 32)
    amax = wf.abs().amax(dim=-1)

    # E8M0 exponent, matching _gang_compute_e8m0_fp8 bit for bit: take the raw
    # exponent of amax/448 and round up whenever the mantissa is non-zero, so
    # the block's largest value lands at or below E4M3's 448 ceiling.
    target = (amax / 448.0).contiguous()
    u = target.view(torch.int32)
    raw_exp = ((u >> 23) & 0xFF) + ((u & 0x7FFFFF) != 0).to(torch.int32)
    se = torch.where(amax == 0, torch.zeros_like(raw_exp),
                     raw_exp.clamp(0, 255))

    # se == 0 means an all-zero block; the kernel decodes that exponent as 1.0,
    # so divide by 1.0 here too rather than by 2^-127.
    scale = torch.where(se == 0, torch.ones_like(target),
                        (se.to(torch.int32) << 23).view(torch.float32))
    data = (wf / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)

    return (data.view(torch.uint8).reshape(*w.shape[:-1], K),
            se.to(torch.uint8))


def pack_mxfp8_workgroup(data: torch.Tensor, scales: torch.Tensor,
                         output_per_wg: int = 64) -> torch.Tensor:
    """Repack quantize_mxfp8 or quantize_mxfp4 output into the per-workgroup
    layout the MXFP8/MXFP4 kernels read, mirroring gpt-oss's
    pack_mxfp4_workgroup:

        [E, wgs, OPW*row_bytes data | OPW*(K/32) scale bytes]

    row_bytes is K at MXFP8 and K/2 at MXFP4; it comes from the data tensor's
    own last dimension, and the scale count fixes K, so one packer serves both
    widths and the kernel reads the width back off this stride.

    Rows are K-major within the data half; scales are [row][k/32]. A 2-D
    weight is treated as E == 1 and returned without the leading axis.
    """
    squeeze = (data.dim() == 2)
    if squeeze:
        data = data.unsqueeze(0)
        scales = scales.unsqueeze(0)
    E, out_dim, row_bytes = data.shape
    K = scales.shape[2] * 32
    assert scales.shape == (E, out_dim, K // 32)
    assert row_bytes in (K, K // 2), \
        f"{row_bytes} bytes per row is neither MXFP8 ({K}) nor MXFP4 {K // 2}"
    assert out_dim % output_per_wg == 0, \
        f"out_dim {out_dim} must be divisible by output_per_wg {output_per_wg}"

    wgs = out_dim // output_per_wg
    packed = torch.cat(
        [data.reshape(E, wgs, output_per_wg * row_bytes),
         scales.reshape(E, wgs, output_per_wg * (K // 32))],
        dim=2).contiguous()
    return packed.squeeze(0) if squeeze else packed


# E2M1, the MXFP4 element format: sign + 2 exponent + 1 mantissa bits, so
# eight magnitudes and a 6.0 ceiling against E4M3's 448.
_E2M1_LEVELS = torch.tensor([0., 0.5, 1., 1.5, 2., 3., 4., 6.])


def fake_quantize_mxfp4(w: torch.Tensor) -> torch.Tensor:
    """Round a bf16 weight through MXFP4 and back, in place of quantizing to it.

    This is a measurement tool, not a code path we ship. MXFP4 would cut the
    MoE from 2.24 to 1.22 GB per token -- by far the largest single byte lever
    left -- but GLM-4.7-Flash ships bf16, so unlike gpt-oss, whose weights were
    trained for the format, this is a post-hoc 4-bit quantization. On layer
    20's experts it measures 1.18e-01 relative RMS against MXFP8's 2.65e-02.

    Routing the values through MXFP4 while still *packing* them as MXFP8 gives
    the end-to-end quality answer for zero kernel work: the megakernel runs
    unchanged and only the numbers it reads are 4-bit-accurate. If agreement
    survives, the kernel is worth writing; if it does not, no kernel would have
    saved it. Same block structure as quantize_mxfp8 -- one E8M0 exponent per
    32 contiguous K -- so the comparison isolates the element format.
    """
    K = w.shape[-1]
    assert K % 32 == 0, f"K {K} must be a multiple of 32"
    wf = w.float().reshape(*w.shape[:-1], K // 32, 32)
    amax = wf.abs().amax(dim=-1)
    target = (amax / 6.0).contiguous()
    u = target.view(torch.int32)
    raw_exp = ((u >> 23) & 0xFF) + ((u & 0x7FFFFF) != 0).to(torch.int32)
    se = torch.where(amax == 0, torch.zeros_like(raw_exp), raw_exp.clamp(0, 255))
    scale = torch.where(se == 0, torch.ones_like(target),
                        (se.to(torch.int32) << 23).view(torch.float32))
    n = wf / scale.unsqueeze(-1)
    levels = _E2M1_LEVELS.to(n.device)
    q = levels[(n.abs().unsqueeze(-1) - levels).abs().argmin(-1)] * n.sign()
    return (q * scale.unsqueeze(-1)).reshape(w.shape).to(w.dtype)


# Narrow the routed experts from MXFP8 to MXFP4. The MoE weights are 2.238 GB
# of the 4.537 GB this model streams per token, so this is the single largest
# byte lever left and, together with de-padding and un-absorbing attention, the
# arithmetic path to 2 ms -- 4.54 -> 2.67 GB at 1.44 TB/s is 1.86.
#
# GLM-4.7-Flash ships bf16, so unlike gpt-oss this is post-hoc 4-bit
# quantization of weights never trained for it, and the format error is 4x
# MXFP8's: 1.18e-01 relative RMS against 2.65e-02 on layer 20's experts. What
# licences it is the end-to-end probe -- fake_quantize_mxfp4 below, which routes
# expert values through E2M1 while still packing MXFP8, so the megakernel runs
# untouched -- which reproduced the MXFP8 baseline's token agreement exactly.
# A WikiText-2 perplexity sweep is still owed, as it is for MXFP8.
#
# Measured back to back: 4.634 -> 4.467 ms/token, with the 50-token CI check
# giving the same 22-token longest common block and 18-token exact prefix as
# MXFP8. Note how small that is against the byte arithmetic -- 1.02 GB removed
# would be ~1.0 ms if the MoE were bandwidth-bound, and it bought 0.17. The
# stage runs at roughly 1 TB/s of the 5.4 the part can stream, so bytes are no
# longer what binds it; see task #30. MXFP4 still earns its place, both for the
# 0.17 and because the byte budget has to come down anyway before utilization
# work can cash out. Set GLM_MOE_MXFP4=0 for MXFP8.
MOE_MXFP4 = os.environ.get("GLM_MOE_MXFP4", "1") == "1"


# The same question for the attention weights (task #66): qkv_a, q_b, W_UK,
# W_UV and o_proj are all MXFP8 today, and narrowing them to MXFP4 halves
# their bytes. The performance ceiling is already priced and it is small --
# MPK_ATTN_HALFK, which halves qkv_a's bytes AND its FLOPs, bought 4.91
# us/layer, so the whole program is under ~0.9 ms -- but the ruling is to
# build the queue largest-first and this is the largest item on it.
#
# Quality is the gate, and it is a much harder gate here than on the MoE.
# GLM ships bf16, so this is post-hoc 4-bit quantization of weights never
# trained for it (1.18e-01 relative RMS against MXFP8's 2.65e-02), and unlike
# a routed expert -- one of 256, contributing one of eight top-k terms -- every
# attention weight is on the path of every token. So run the numerics before
# writing any kernel, exactly as GLM_FAKE_MXFP4_EXPERTS did for the MoE:
# round the values through E2M1 but keep packing MXFP8, so the megakernel is
# byte-for-byte the shipped one and only the numbers it reads are 4-bit.
# If the tokens survive, the kernel is worth writing; if they do not, no
# kernel would have saved it.
#
# Deliberately NOT applied to the LM head, which pack_dense_mxfp8 also serves:
# the head is read once per token rather than 78 times, so it is not part of
# the lever, and including it would confound the quality answer.
FAKE_MXFP4_ATTN = os.environ.get("GLM_FAKE_MXFP4_ATTN", "0") == "1"


# The real thing, for o_proj only. o_proj is 100.7 MB of the ~166 MB of
# attention weight a layer reads per GPU -- six times qkv_a, which is what the
# ~0.9 ms ceiling was extrapolated from -- and it is the only member of the
# set measured byte-bound (76% of HBM peak), so it is the one where halving
# bytes should convert to time rather than disappear into a barrier.
#
# MEASURED, n=6 alternated in one batch, NP=8, device-clock decode average:
#   MXFP8  10.768 10.865 10.626 10.791 10.641 10.914  mean 10.768
#   MXFP4  10.572 10.559 10.680 10.845 10.573 10.541  mean 10.628
# i.e. -0.139 ms, ~2 sigma and right at the 0.26 ms single-run noise floor.
# The prediction was -0.76 to -1.0 ms (50 MB/layer/GPU at 76% of 5.17 TB/s),
# so about 85% of the byte saving does not convert: the o_proj weight block is
# already DMA'd up the hierarchy during the Phase 6 barrier spin
# (gang_mla_full_layer_fused_mi300.cuh, the buffer_load_lds ladder), so
# halving it shortens a transfer that was already hidden. Default on anyway --
# it is a strict byte reduction, it is correctness-gated, and it takes 7.9 GB
# of o_proj weight per GPU down to 4.0.
#
# The kernel side is MPK_OPROJ_MXFP4, set from this same variable in
# python/mirage/mpk/persistent_kernel.py. The two MUST agree: the packed
# layout is identical apart from the row width, so a mismatch mis-addresses
# every weight row and produces garbage rather than failing to build.
OPROJ_MXFP4 = os.environ.get("GLM_OPROJ_MXFP4", "1") == "1"


def quantize_mxfp4(w: torch.Tensor) -> tuple:
    """Quantize a [..., out, K] bf16 weight to MXFP4: E2M1 nibbles with one
    E8M0 exponent per 32 contiguous K elements.

    Same block structure and the same scale arithmetic as quantize_mxfp8 -- the
    MFMA addresses its scale operand by matrix position either way, so the
    contiguous-32 grouping is no more optional here than there -- with 6.0 in
    place of 448.0 as the ceiling the exponent has to bring the block under.

    The rounding is gpt-oss's quantize_bf16_to_mxfp4, thresholds and all: the
    midpoints between the eight E2M1 magnitudes, applied as a ladder of
    comparisons rather than a nearest-level search, with the sign in bit 3.
    Nibble order within a byte is element 2b low, 2b+1 high, matching what
    __builtin_amdgcn_cvt_scalef32_pk_fp4_f32 writes on the activation side.

    Returns (data uint8 [..., out, K/2], scales uint8 [..., out, K/32]).
    """
    K = w.shape[-1]
    assert K % 32 == 0, f"K {K} must be a multiple of 32"
    wf = w.float().reshape(*w.shape[:-1], K // 32, 32)
    amax = wf.abs().amax(dim=-1)

    target = (amax / 6.0).contiguous()
    u = target.view(torch.int32)
    raw_exp = ((u >> 23) & 0xFF) + ((u & 0x7FFFFF) != 0).to(torch.int32)
    se = torch.where(amax == 0, torch.zeros_like(raw_exp),
                     raw_exp.clamp(0, 255))
    scale = torch.where(se == 0, torch.ones_like(target),
                        (se.to(torch.int32) << 23).view(torch.float32))

    n = wf / scale.unsqueeze(-1)
    a = n.abs()
    nib = torch.zeros_like(a, dtype=torch.uint8)
    for thr, code in ((0.25, 1), (0.75, 2), (1.25, 3), (1.75, 4),
                      (2.50, 5), (3.50, 6), (5.00, 7)):
        nib[a >= thr] = code
    nib[n < 0] |= 8

    packed = nib[..., 0::2] | (nib[..., 1::2] << 4)
    return (packed.reshape(*w.shape[:-1], K // 2), se.to(torch.uint8))


def pack_moe_mxfp8(stacked: torch.Tensor,
                   output_per_wg: int = 64) -> torch.Tensor:
    """Quantize + pack a stacked [E, out, K] bf16 expert weight into the MXFP8
    per-workgroup layout, one expert at a time.

    Quantizing all E at once would materialise an [E, out, K] fp32 intermediate
    inside quantize_mxfp8 -- 1.6 GB for GLM's 65 x 3072 x 2048 gate/up stack,
    on top of the bf16 stack it is being built from. Per-expert bounds that
    transient at 1/E of it, at no cost to the result.
    """
    assert stacked.dim() == 3, stacked.shape
    if MOE_MXFP4:
        quant = quantize_mxfp4
    elif os.environ.get("GLM_FAKE_MXFP4_EXPERTS", "0") == "1":
        quant = lambda w: quantize_mxfp8(fake_quantize_mxfp4(w))
    else:
        quant = quantize_mxfp8
    packed = [pack_mxfp8_workgroup(*quant(stacked[e]), output_per_wg)
              for e in range(stacked.shape[0])]
    return torch.stack(packed).contiguous()


def pack_dense_mxfp8(w: torch.Tensor, output_per_wg: int = 64,
                     rows_per_chunk: int = 8192,
                     fake_fp4: bool = False,
                     fp4: bool = False) -> torch.Tensor:
    """Quantize + pack a 2-D [out, K] bf16 weight into the MXFP8 per-workgroup
    layout, in row chunks.

    Same reason as pack_moe_mxfp8: quantize_mxfp8 materialises an fp32 copy of
    whatever it is handed, which for GLM's 155136 x 2048 LM head is 1.27 GB on
    top of the bf16 original. Rows are independent and a workgroup is a
    contiguous run of `output_per_wg` rows, so chunking on a multiple of that
    and concatenating along the workgroup axis is exact.

    fake_fp4 routes the values through E2M1 and back while still packing MXFP8,
    the same measurement trick GLM_FAKE_MXFP4_EXPERTS uses on the MoE side. It
    answers the quality half of "MXFP4 for the attention weights" for zero
    kernel work -- see FAKE_MXFP4_ATTN.

    fp4 is the real thing: E2M1 nibbles, half the data bytes. The workgroup
    layout is unchanged -- pack_mxfp8_workgroup takes its row width from the
    data tensor -- so only the consuming kernel has to know, and it is
    gang_gemv_mxfp4_kernel rather than gang_gemv_mxfp8_kernel. The two flags
    are exclusive: fake_fp4 exists to answer the question fp4 acts on.
    """
    assert w.dim() == 2, w.shape
    assert not (fp4 and fake_fp4), \
        "fake_fp4 is the probe for fp4; packing both at once measures nothing"
    rows = w.shape[0]
    assert rows % output_per_wg == 0, (rows, output_per_wg)
    step = max(output_per_wg,
               (rows_per_chunk // output_per_wg) * output_per_wg)
    quant = (quantize_mxfp4 if fp4
             else (lambda x: quantize_mxfp8(fake_quantize_mxfp4(x))) if fake_fp4
             else quantize_mxfp8)
    parts = [pack_mxfp8_workgroup(*quant(w[r:r + step]), output_per_wg)
             for r in range(0, rows, step)]
    return torch.cat(parts, dim=0).contiguous()


def interleave_gate_up(w_gate: torch.Tensor, w_up: torch.Tensor,
                       num_groups: int) -> torch.Tensor:
    """Lay out [gate | up] as `num_groups` consecutive [gate_chunk; up_chunk]
    slabs.

    silu_mul_task_impl reads its multiplicand at ``input_ptr + OUTPUT_SIZE``,
    where OUTPUT_SIZE is the *per-block* width after grid.x partitioning. A
    flat [gate(I) | up(I)] weight is therefore only read correctly when
    grid.x == 1. Grouping the rows the same way mpk.shuffle_tensors does (see
    the llama3 QKV shuffle) keeps grid.x == 8, one silu_mul task per XCD.

    The routed experts use this too, but at ``num_groups == intermediate``,
    i.e. one row per chunk: gang_moe_w13_linear_kernel's fused SwiGLU
    epilogue needs gate_j and up_j to land in the same thread's four
    consecutive accumulators, which only happens when they are adjacent
    columns.
    """
    inter, hidden = w_gate.shape
    assert inter % num_groups == 0, (inter, num_groups)
    chunk = inter // num_groups
    return torch.stack([w_gate.reshape(num_groups, chunk, hidden),
                        w_up.reshape(num_groups, chunk, hidden)],
                       dim=1).reshape(2 * inter, hidden).contiguous()


def absorb_q_b(q_b_weight: torch.Tensor, w_uk: torch.Tensor,
               num_heads: int, qk_nope: int, qk_rope: int,
               q_lora: int) -> torch.Tensor:
    """Fold W_UK into q_b_proj.

    Per head, ``q_absorbed[c] = sum_n q_nope[n] * W_UK[h, n, c]`` and
    ``q_nope = q_b_nope[h] @ q_a_norm``, so the whole thing is one GEMM against

        q_b_absorbed[h] = [ W_UK[h]^T @ q_b_nope[h] ; q_b_rope[h] ]   [576, q_lora]

    stacked over heads into [H * 576, q_lora]. This is what makes the cached
    576-wide latent row the only K the decode kernel ever needs.
    """
    qb = q_b_weight.view(num_heads, qk_nope + qk_rope, q_lora)
    qb_nope = qb[:, :qk_nope, :].float()          # [H, 192, q_lora]
    qb_rope = qb[:, qk_nope:, :].float()          # [H,  64, q_lora]
    absorbed = torch.einsum("hnc,hnq->hcq", w_uk.float(), qb_nope)  # [H,512,q]
    out = torch.cat([absorbed, qb_rope], dim=1)   # [H, 576, q_lora]
    return out.reshape(num_heads * (w_uk.shape[2] + qk_rope), q_lora)


def absorb_o_proj(o_weight: torch.Tensor, w_uv: torch.Tensor,
                  num_heads: int, v_head: int, kv_lora: int) -> torch.Tensor:
    """Fold W_UV into o_proj: [hidden, H*v_head] -> [hidden, H*kv_lora].

    The decode kernel's PV accumulates over the leading ``kv_lora`` dims of the
    latent row, so its output is per-head [512] rather than [256]. Absorbing
    W_UV here removes the second per-head GEMM entirely.
    """
    hidden = o_weight.shape[0]
    out = torch.empty(hidden, num_heads * kv_lora,
                      dtype=torch.float32, device=o_weight.device)
    ow = o_weight.float()
    uv = w_uv.float()
    for h in range(num_heads):
        out[:, h * kv_lora:(h + 1) * kv_lora] = (
            ow[:, h * v_head:(h + 1) * v_head] @ uv[h])
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-mirage", action="store_true",
                        help="Use Mirage persistent-kernel decode")
    parser.add_argument("--max-num-batched-tokens", default=1, type=int)
    parser.add_argument("--max-num-batched-requests", default=1, type=int)
    parser.add_argument("--page-size", default=4096, type=int)
    parser.add_argument("--max-num-pages", default=16, type=int)
    parser.add_argument("--output-dir", help="Output files directory")
    parser.add_argument("--trace-name", default="",
                        help="Perfetto trace output name")
    parser.add_argument("--profiling", action="store_true")
    parser.add_argument("--max-seq-length", default=512, type=int)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--tokenizer-path", type=str, default=None,
        help=("Load the tokenizer from here instead of --model-path. Needed "
              "for a config-only GLM-5 checkpoint, which ships no tokenizer."))
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--prompt", type=str,
                        default="The capital of France is")
    parser.add_argument(
        "--save-tokens", nargs="?", const="auto", default=None,
        help=("Dump generated token_ids to JSON for the correctness test. If "
              "the path is omitted, saves to outputs/glm5/{torch_output.json|"
              "mpk_output.json}."),
    )
    parser.add_argument("--max-layers", type=int, default=None,
                        help="Only use the first N layers (bring-up on a "
                             "memory-constrained GPU)")
    parser.add_argument("--random-weights", action="store_true",
                        help="Skip the checkpoint; exercise shapes only")
    parser.add_argument(
        "--mxfp4-checkpoint", dest="mxfp4_checkpoint", action="store_true",
        default=None,
        help="Read routed experts as pre-quantised MXFP4 blocks/scales "
             "(the stream_quant_glm5.py output) instead of bf16. Default is "
             "auto: on iff the checkpoint index advertises .mxfp4_blocks.")
    parser.add_argument("--no-mxfp4-checkpoint", dest="mxfp4_checkpoint",
                        action="store_false")
    parser.add_argument(
        "--rope-interleave", dest="rope_interleave", action="store_true",
        default=None,
        help="Force interleaved RoPE pairs (2j, 2j+1). GLM-5 sets this in "
             "config.json; GLM-4.7-Flash omits the key and inherits the "
             "family default, which is interleaved.")
    parser.add_argument("--no-rope-interleave", dest="rope_interleave",
                        action="store_false")
    parser.add_argument(
        "--mtp", type=int, default=0,
        help="Load N multi-token-prediction draft layers (GLM-5 ships 1) and "
             "run speculative decode in the torch reference path. 0 (default) "
             "drops the MTP weights exactly as before.")
    args = parser.parse_args()
    # The megakernel builds its task graph from model.main_layers, so the MTP
    # layer is loaded and then never dispatched. Without this guard
    # `--use-mirage --mtp 1` runs plain greedy decode, pays the draft layer's
    # weights in HBM, and reports a per-token latency that looks like MTP's and
    # is not -- exactly the kind of number this repo does not quote.
    assert not (args.use_mirage and args.mtp), (
        "--mtp is torch-reference only for now: the draft layer, its eh_proj "
        "front end and the accept/reject step are not in the megakernel task "
        "graph yet, so --use-mirage would silently decode one token per "
        "iteration and mis-report ms/token")

    # Resolve where to dump generated tokens for the correctness test.
    if args.save_tokens:
        if args.save_tokens == "auto":
            fn = "mpk_output.json" if args.use_mirage else "torch_output.json"
            save_path = os.path.join(DEFAULT_SAVE_DIR, fn)
        else:
            save_path = args.save_tokens
    else:
        save_path = None

    # Force cutlass off on ROCm/MI300X.
    use_cutlass_kernel = False

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        world_size = comm.Get_size()
        rank = comm.Get_rank()
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        # Overridable: a killed run leaves the listener in TIME_WAIT (or, if
        # it hung, still bound), and the next launch dies with EADDRINUSE
        # before it has even built. Debugging a livelock means killing runs.
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")
    except ImportError:
        world_size = 1
        rank = 0

    # Rank-suffix the token dump. Under DP attention every rank decodes the
    # same prompt and must produce the same answer, so the per-rank dumps are
    # the correctness gate for the EP fold -- a fold that drops one peer's
    # experts corrupts that rank's residual stream and nobody else's. Without
    # the suffix all eight ranks race on one path and the file that survives
    # is whichever rank closed last, which is exactly the rank you do not get
    # to choose.
    if save_path is not None:
        if world_size > 1:
            _stem, _ext = os.path.splitext(save_path)
            save_path = f"{_stem}_rank{rank}{_ext}"
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
    global print
    # Silencing the other ranks keeps a normal run readable, but it also hides
    # the one thing an 8-rank latency investigation needs: the per-rank decode
    # summary. [FWD_PASS_TOTAL] is printed by every rank and is not a
    # substitute -- it averages prefill in, and prefill is where startup skew
    # lands, so a rank that is 8x on FWD_PASS_TOTAL may be identical in decode.
    if rank != 0 and os.environ.get("MPK_PRINT_ALL_RANKS", "0") != "1":
        print = lambda *_, **__: None
    elif rank != 0:
        _rank_tag = f"[r{rank}] "
        _real_print = print
        print = lambda *a, **k: _real_print(_rank_tag, *a, **k)

    # ── data-parallel attention ──────────────────────────────────────────
    # MLA keeps ONE shared latent head, so the gpt-oss `kv_head == xcd_id`
    # split has no analogue and slicing the q heads eight ways leaves 8 heads
    # per rank -- a shape the decode kernel's 16-head MFMA group does not
    # have. At batch 1 replicating attention is nearly free: every rank keeps
    # all heads and the whole latent cache, runs the identical attention on
    # the identical token, and the result is already complete per rank. That
    # deletes the attention all-reduce outright and leaves the MoE fold as the
    # only collective in the step.
    #
    # attn_ws is what from_pretrained is handed, and it is the whole switch:
    # GlmMLA divides num_heads by it and inserts a dist.all_reduce in o_proj,
    # neither of which DP may have.
    attn_dp = world_size > 1 and os.environ.get("ATTN_DP", "1") == "1"
    attn_ws = 1 if attn_dp else world_size

    # Expert parallelism has to be decided BEFORE the weights are loaded, not
    # just before they are packed. At GLM-4.7-Flash's 64 experts every rank can
    # afford to load all of them and throw away the seven eighths it does not
    # own; at GLM-5's 256 experts x 76 layers it cannot -- the full bf16 MoE is
    # 1488 GB against 288 GB of HBM. So the rank's slice is pushed down into
    # model construction, and `ep_base` below becomes an offset into a list
    # that already starts at this rank's first expert.
    moe_ep = world_size > 1 and os.environ.get("MOE_EP", "1") == "1"

    print("Input arguments:", args)
    print(f"world_size({world_size}) rank({rank})")
    if attn_dp:
        print("[ATTN_DP] attention replicated per rank; "
              "no attention all-reduce")
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(rank)

    with torch.device("cuda"):
        model = GlmMoeDsaForCausalLM.from_pretrained(
            args.model_path, world_size=attn_ws,
            max_num_pages=args.max_num_pages, page_size=args.page_size,
            num_layers=args.max_layers, random_weights=args.random_weights,
            ep_rank=(rank if moe_ep else 0),
            ep_world=(world_size if moe_ep else 1),
            mxfp4_experts=args.mxfp4_checkpoint,
            num_mtp_layers=args.mtp,
        ).to(dtype=torch.bfloat16, device="cuda")
        # A partially-fetched checkpoint (config + shard index only, which is
        # how the 744B geometry is brought up on one GPU) carries no tokenizer
        # files. --tokenizer-path borrows one; with --random-weights the text
        # is meaningless anyway, only the shapes matter.
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path or args.model_path, trust_remote_code=True)

    config = model.config
    if args.rope_interleave is not None:
        config.rope_interleave = args.rope_interleave
    # GLM configs list several stop ids ([154820, 154827, 154829] for Flash).
    # The megakernel only carries one, so hand it the first and keep the full
    # set for the torch reference loop.
    eos_token_ids = (list(config.eos_token_id)
                     if isinstance(config.eos_token_id, (list, tuple))
                     else [config.eos_token_id])
    # Main stack only. An MTP draft layer, when loaded, lives past the end of
    # this list and is not part of the fused per-layer graph.
    num_layers = model.model.num_main_layers
    print(f"{config.hf_model_type}: {num_layers} layers "
          f"(of {config.num_hidden_layers}), hidden {config.hidden_size}, "
          f"{config.num_attention_heads} q heads, {config.n_routed_experts} "
          f"experts top-{config.num_experts_per_tok}, "
          f"rope_interleave={config.rope_interleave}")

    total_num_requests = 1 if not args.use_mirage else args.max_num_batched_requests

    tokens = torch.full((total_num_requests, args.max_seq_length), 0,
                        dtype=torch.long, device="cuda")

    text = args.prompt
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": text}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([formatted], return_tensors="pt",
                                 add_special_tokens=False).to("cuda")
        print(f"Chat template applied: {len(model_inputs.input_ids[0])} tokens")
    else:
        model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
    for r in range(total_num_requests):
        for i in range(model_inputs.input_ids.shape[-1]):
            tokens[r, i] = model_inputs.input_ids[0, i]
    prompt_lengths = torch.full(
        (total_num_requests,), model_inputs.input_ids.shape[-1],
        dtype=torch.int, device="cuda")

    # Position embeddings: [1, seq, qk_rope_head_dim], HF layout
    # (emb = cat(freqs, freqs)). Only the first half of each row is ever read
    # by the megakernel's partial interleaved RoPE.
    positions = torch.arange(args.max_seq_length).unsqueeze(0).to("cuda")
    position_embeddings = model.model.rotary_emb(positions)

    input_tokens = torch.full((args.max_num_batched_tokens, 1), 0,
                              dtype=torch.long, device="cuda")
    output_tokens = torch.full((args.max_num_batched_tokens, 1), 0,
                               dtype=torch.long, device="cuda")
    prev_pos = 0

    starter, ender = (torch.cuda.Event(enable_timing=True),
                      torch.cuda.Event(enable_timing=True))
    step = torch.full((total_num_requests,), 0, dtype=torch.int32,
                      device="cuda")
    num_new_tokens = torch.full((total_num_requests,), 1, dtype=torch.int32,
                                device="cuda")

    profiler_tensor = None

    if args.use_mirage:
        import mirage as mi

        # Gang dispatch is what the MI350/MI355 GEMM and MoE kernels are
        # written for; the non-gang fallbacks exist but are not the tuned path.
        os.environ.setdefault("USE_GANG", "1")

        # ── Geometry ─────────────────────────────────────────────────────────
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        # gang_mla_decode requires num_q_heads % 16 == 0 (one MFMA 16x16 tile
        # per query group). GLM-4.7-Flash's 20 heads pad to 32; the padded
        # heads get zero q_b_absorbed rows and zero o_absorbed columns, so they
        # attend uniformly and contribute nothing.
        num_heads_pad = align_up(num_heads, 16)
        # q_b's output does not need all of num_heads_pad.
        #
        # num_heads_pad is 16-aligned because MLA decode reduces 16 q heads per
        # MFMA group. q_b has no such rule: its only constraints are that each
        # XCD's column chunk hold whole heads (so the fused rope slice is a
        # question about wg_idx alone) and that the row stay GANG_OUT_ALIGN
        # aligned. Both follow from an 8-aligned slot count, because qk_dim is
        # an odd multiple of the rope width.
        #
        # This matters because tile count is quantised by the worker pool, not
        # by bytes. At 32 slots q_b is 18432/64/8 = 36 GEMM tiles + 1 latent
        # tile = 37 per XCD against 30 workers, so it dispatches twice and the
        # second round runs 7 tiles on 30 workers -- the stage costs two full
        # rounds to do 1.23 rounds of work. 24 slots is 28 tiles: one round,
        # and 25% less q_b weight traffic as a side effect.
        #
        # The slots past num_heads are zero either way; the only change is that
        # slots [qb_head_slots, num_heads_pad) stop being *written* with the
        # zeros they already hold from allocation, so MLA reads the same q.
        qb_head_slots = align_up(num_heads, 8)
        kv_lora = config.kv_lora_rank
        qk_rope = config.qk_rope_head_dim
        qk_nope = config.qk_nope_head_dim
        qk_head_dim = config.qk_head_dim
        v_head = config.v_head_dim
        qk_dim = kv_lora + qk_rope                      # 576
        q_lora = config.q_lora_rank
        q_lora_pad = align_up(q_lora, GANG_OUT_ALIGN)
        kv_a_out_pad = align_up(qk_dim, GANG_OUT_ALIGN)  # 576 -> 1024
        # q_a_proj and kv_a_proj_with_mqa are one GEMM emitting
        # [q_a (q_lora_pad) | latent (kv_a_out_pad)]; see the layer loop.
        qkv_a_pad = q_lora_pad + kv_a_out_pad
        num_experts = config.n_routed_experts
        topk = config.num_experts_per_tok
        moe_inter = config.moe_intermediate_size
        shared_inter = moe_inter * config.n_shared_experts
        dense_inter = config.intermediate_size
        first_k_dense = min(config.first_k_dense_replace, num_layers)
        # The shared expert rides along as one extra routed expert (id
        # `num_experts`, slot `topk`, weight 1) instead of a second pair of
        # GEMMs -- see the router kernel's header comment. It has exactly a
        # routed expert's shape, which is what makes that legal.
        num_shared = config.n_shared_experts
        assert num_shared == 1, f"expected one shared expert, got {num_shared}"
        assert shared_inter == moe_inter, (shared_inter, moe_inter)
        num_experts_total = num_experts + num_shared
        topk_total = topk + num_shared

        assert hidden_size % GANG_OUT_ALIGN == 0, hidden_size
        assert hidden_size % GANG_RED_ALIGN == 0, hidden_size
        assert (num_heads_pad * qk_dim) % GANG_OUT_ALIGN == 0
        assert (2 * moe_inter) % 64 == 0 and moe_inter % 128 == 0

        # Fold the routed experts' SiLU-mul into W13's epilogue. Deletes 5
        # tasks and one dispatch barrier per MoE layer (230 of the 238
        # silu_mul tasks per token) and keeps the 2*moe_inter-wide gate/up
        # intermediate in registers instead of round-tripping it through HBM.
        # Needs the pairwise-interleaved gate/up weight layout below.
        FUSE_MOE_SWIGLU = os.environ.get("GLM_FUSE_MOE_SWIGLU", "1") == "1"

        # Fold the topk weighting and the cross-expert sum into W2's
        # epilogue, as gang_moe_fused_mxfp4 does for gpt-oss: scale by
        # routing_weight and f32-atomicAdd into a [bs, hidden] workspace,
        # instead of writing a [bs, topk_total, hidden] slab that
        # moe_mul_sum_add then re-reads. Leaves a one-task residual add that
        # also re-zeroes the workspace.
        FUSE_MOE_MULSUMADD = (
            os.environ.get("GLM_FUSE_MOE_MULSUMADD", "1") == "1")

        # Store the routed experts' weights as MXFP8 (E4M3 + one E8M0 per 32
        # contiguous K) instead of bf16. The MoE GEMMs are 47.4% of the 9.17
        # GB/token this model streams per token, and at ~1.25 TB/s effective
        # bandwidth the run is already bandwidth-bound, so halving the bytes
        # is the only lever left on them. Both fused epilogues carry over
        # unchanged. Worth 7.02 -> 6.35 ms/token, with the 50-token CI check
        # still matching the bf16 Torch reference exactly; a WikiText-2
        # perplexity sweep is still owed. Set GLM_MOE_MXFP8=0 for bf16.
        MOE_MXFP8 = os.environ.get("GLM_MOE_MXFP8", "1") == "1"
        # Fuse the absorbed o_proj into the router task, replacing one of the
        # nine per-layer dispatch boundaries with a cross-XCD barrier. Applies
        # to MoE layers on the MXFP8 GEMV path only; the dense layer keeps the
        # standalone o_proj.
        #
        #
        # The whole tail of a MoE layer in one task: o_proj, post-attention
        # RMSNorm, router, TopK, MoE W13+SwiGLU, MoE W2+MulSumAdd. Five events
        # and five dispatches become two in-kernel barriers.
        #
        # Fusing only o_proj+router measured at parity (5.381 vs 5.362): the
        # barrier costs what the dispatch cost, which is the expected result
        # for a pairwise fusion and is not where gpt-oss's speed comes from.
        # The profile says the cost is occupancy, not dispatch -- 85.7 of 240
        # workers busy on average, and 3.2x more depwait than compute on MoE
        # W13. A narrow stage only stops paying for a 240-worker barrier if it
        # rides inside a wide stage's workgroup, and that needs one task.
        FUSE_OPROJ_ROUTER = (
            os.environ.get("GLM_FUSE_OPROJ_ROUTER", "1") == "1")
        # The same treatment for the attention half: input RMSNorm + qkv_a,
        # q_a RMSNorm + q_b + latent append, MLA decode and the split-KV merge
        # in one gang task. Four dispatches and four events become three
        # in-kernel barriers, and the two narrow phases -- the decode is 16 of
        # 240 workers, the merge 32 -- stop paying for a 240-worker boundary
        # by riding inside the q_b workgroup instead.
        #
        # Off by default, because measured back to back it loses: 4.766 ms
        # against 4.655 for the four separate tasks. Trading four scheduler
        # boundaries for three in-kernel gang barriers is not the win it looks
        # like, because the barriers are *wider* than the work they separate --
        # the decode and merge phases hold every dispatched worker at a 224-way
        # barrier to do 32 workgroups of work, whereas the task graph lets the
        # narrow stages be narrow. See task #30; this is worth revisiting once
        # the intra-task imbalance is fixed, since the fusion itself is sound.
        # The residual fold is deliberately independent of this knob.
        FUSE_ATTN = os.environ.get("GLM_FUSE_ATTN", "0") == "1"
        # And both halves together: the whole decoder layer as one gang task.
        # Six dispatches per layer become one, and the attention->o_proj event
        # becomes the fifteen-phase task's Phase 8 in-kernel barrier.
        #
        # This subsumes GLM_FUSE_ATTN and GLM_FUSE_OPROJ_ROUTER on MoE layers:
        # it needs everything they need (MXFP8 on both dense GEMMs and on the
        # experts, both MoE epilogues folded, more than one KV chunk so there
        # is a merge phase) plus a write-through merge, which it forces below
        # because attn_out now crosses a barrier instead of an event.
        FUSE_FULL_LAYER = (
            os.environ.get("GLM_FUSE_FULL_LAYER", "0") == "1")

        # ── expert parallelism ───────────────────────────────────────────
        # The routed experts split by ID across the ranks -- gpt-oss's
        # `ep_slice`. Its other split, EP_SLOT, partitions the ACTIVATED list
        # instead and is the better-balanced of the two, but a top-4 list
        # cannot span 8 ranks, so GLM does not get to use it.
        #
        # What is local is only the weight tensors. The router, the routing
        # table, the activated list, both RMSNorms and o_proj stay REPLICATED
        # and keyed by the global expert id -- every rank has to derive the
        # same activated list or they would disagree about who owns what.
        #
        # The shared expert has no id range to fall in: it is id
        # `num_experts`, every token routes to it, and it must be computed by
        # exactly ONE rank or the cross-rank sum would count it world_size
        # times. The kernel gives it to EP_SHARED_PE == EP_FOLD_PE; every rank
        # still stores a replica it never reads, which is cheaper than a
        # ragged weight shape.
        # moe_ep was decided before the load, above.
        if moe_ep:
            assert num_experts % world_size == 0, (
                f"ep_slice needs {num_experts} routed experts to divide by "
                f"world_size {world_size}")
            ep_local = num_experts // world_size
            ep_base = rank * ep_local
        else:
            ep_local = num_experts
            ep_base = 0

        # Where this rank's slice starts *in the list the model holds*. When
        # construction was already EP-aware the list begins at ep_base, so the
        # offset is zero; `ep_base` itself stays the global id and is what the
        # log and the kernel's expert-id arithmetic mean.
        model_ep_sharded = any(
            getattr(l.mlp, "ep_local", num_experts) != num_experts
            for l in model.model.main_layers if l.is_moe)
        ep_slice_base = 0 if model_ep_sharded else ep_base
        # The single rank that folds the real residual into its MoE partial,
        # so the residual survives the cross-rank sum exactly once. The kernel
        # gives the same rank the shared expert, for the same reason.
        ep_fold_rank = int(os.environ.get("EP_FOLD_RANK", "0"))
        assert 0 <= ep_fold_rank < world_size
        if moe_ep:
            # The fold is Phase 0 of the fused layer, not a task of its own,
            # so EP and whole-layer fusion are one switch.
            assert FUSE_FULL_LAYER, (
                "MOE_EP needs GLM_FUSE_FULL_LAYER=1: the fold, the exchange "
                "and the peer wait are the head of the fused layer and there "
                "is no unfused path for them")
            assert attn_dp, (
                "MOE_EP assumes DP attention; TP attention would need an "
                "all-reduce of its own, which this branch does not have")
            # Every EP threshold is a function of task_layer_idx, which only
            # exists inside the multi-layer batched loop. persistent_kernel.py
            # defaults PRECOMPUTED_DISPATCH to 0 whenever rocSHMEM is on, so
            # this has to be set explicitly -- and refused rather than run as
            # a hang.
            assert os.environ.get("PRECOMPUTED_DISPATCH") == "1", (
                "MOE_EP needs PRECOMPUTED_DISPATCH=1, whose default is 0 "
                "under rocSHMEM: the fold's signal thresholds ride the "
                "multi-layer layer counter")
            assert os.environ.get("MPK_ML_REPLAY", "1") == "1", (
                "MOE_EP needs the multi-layer replay path (MPK_ML_REPLAY=1)")
            print(f"[MOE_EP] rank {rank} owns routed experts "
                  f"[{ep_base}, {ep_base + ep_local}) of {num_experts}; "
                  f"shared expert and residual fold on rank {ep_fold_rank}")

        # Output rows per MoE workgroup, and so the tile count. The default is
        # the N-parallel width; 16 selects the kernel's K-parallel branch, all
        # four waves on the same 16 rows with the reduction split between them.
        #
        # This is the same starvation qkv_a had (see GLM_QKV_MXFP8_OPW below)
        # and expert parallelism makes it much worse. _gang_moe_mxfp8_tile
        # builds the tile space over the OWNED activated experts, and at EP=8 a
        # rank owns about one of the eight routed experts, so at OPW=64 a whole
        # layer's W13 is 2*2048/64 = 64 tiles and W2 is 6144/64 = 96 -- 8 and
        # 12 per XCD against 29 workers. At 16 both are 4x that.
        # Separate knobs so the two stages can move one at a time.
        #
        # W2's tile width is now measured out in BOTH directions and the knob
        # should be left at 64 (2026-08-21, n=3 each, one control batch at
        # 11.020 ms/iter mean):
        #
        #   GLM_MOE_W2_OPW=16   (K-parallel, 4x tiles)   -1.34 ms  [earlier]
        #   MPK_W2_KSPLIT=2     (2x tiles, half K each)  -4.12 ms  -> 15.136
        #   GLM_MOE_W2_OPW=128  (half the tiles)          10.972,  neutral
        #
        # Splitting loses because W2's per-tile cost is mostly fixed -- the
        # whole-activation LDS stage and quantize, and the full atomicAdd
        # epilogue -- and neither divides. Widening does not win because the
        # fixed cost it amortises was never on the critical path: the MoE half
        # of the layer is barrier-bound, so shrinking the phase is absorbed,
        # exactly as in the ceiling probes. Both directions, one conclusion --
        # W2 tile geometry is saturated, stop tuning it.
        #
        # 2026-08-21 addendum: per-worker stage stamps (MPK_BAR_SKEW=3) later
        # showed W2 is the ONE phase whose tile imbalance is not absorbed --
        # 9.66 us of spread at S8 against 1.63 us of layer boundary after it --
        # which revives the round-quantization argument: 12 tiles/XCD/expert
        # over 29 workers means a rank owning 3 experts needs 36 tiles = 2
        # rounds, so widen until 3 experts fit in one round. It does not
        # survive the arithmetic. tiles_per_XCD = (hidden_size/OPW)/8 and
        # task_register.cc asserts hidden_size % OPW == 0, so 9 tiles/XCD needs
        # OPW = 6144/72 = 85.33 and is unreachable; the clean neighbours are
        # OPW=96 (8/XCD) and OPW=128 (6/XCD). OPW=128 is the row above, and at
        # 6 tiles/XCD even FOUR owned experts fit in one round (24 <= 29) --
        # strictly more aggressive than 8 or 9, and it measured neutral. So the
        # unabsorbed spread is real but its geometry fix is already tested.
        # An unabsorbed spread is necessary for a lever, not sufficient.
        MOE_W13_OPW = int(os.environ.get("GLM_MOE_W13_OPW", "64"))
        MOE_W2_OPW = int(os.environ.get("GLM_MOE_W2_OPW", "64"))
        # Experts per router call. The router is one worker per expert, so
        # GLM-5's 256 experts are 32 tiles per XCD against 29 workers: two
        # grid-stride rounds for a mean of 1.10 calls, and the second round
        # re-pays the o_proj barrier spin, the redundant RMSNorm, two block
        # reductions and the arrival atomic just to add one dot product.
        # Measured at the post-deferred-rope operating point (subphase
        # counters, 15.106 ms/iter instrumented): SP3[2] Router 1.415 +
        # SP3[3] RoutingWait 1.010 = 16% of wall, and SP6/SP3 call counts
        # gave the 1.1035 ratio directly. 2 makes it 16 tiles and one round.
        #
        # MEASURED 2026-08-19, same build, one variable: 11.721 ms/iter at 1
        # against 10.990 at 2, -6.2%. (The 1 here is the refactored kernel and
        # is 0.22 ms slower than the 11.500 recorded before it -- the dp[] and
        # red[] indexing is not free at one expert. It goes away at 2.)
        # 4 is legal -- 8 tiles per XCD -- but doubles the prefetch register
        # array again on top of a kernel already at ~332 VGPRs.
        ROUTER_EPT = int(os.environ.get("GLM_ROUTER_EPT", "2"))
        # The router fold. Both of the router's contractions -- the gate GEMV
        # and the RMSNorm's sum-of-squares -- run over the hidden row that the
        # sharded o_proj has just finished producing a 1/world-th slice of, so
        # both decompose over exactly that shard and can be reduced across the
        # ranks on the all-gather rendezvous instead of after it. Needs the
        # shard, so it needs EP and the whole-layer fusion that carries it.
        #
        # MEASURED NEUTRAL, off by default (2026-08-20, four runs, the last two
        # interleaved in one session so the baseline is not a quoted number):
        # decode 14.402 ms/iter folded against 14.477 unfolded, a 0.075 ms
        # difference inside a 0.26 ms noise floor. Both generate coherent text.
        #
        #   ns/call            fold=0   fold=1
        #   SP6[1] ssq           2326     1981   -345
        #   SP6[2] gate GEMV     2186     2090    -96
        #   SP6[0] prefetch +    6377     7375   +998
        #          o_proj spin
        #   SP3[0] o_proj      0.481ms  0.575ms  +0.094 ms/iter
        #
        # The premise was wrong, and the counters say exactly how. Deleting the
        # gate GEMV bought 96 ns, not the ~2.2 us the step appears to cost,
        # because #35's prefetch across the o_proj barrier had already hidden
        # the whole GEMV behind the spin -- the weight was in registers before
        # the row it multiplies existed. So the fold has nothing to reclaim on
        # the router side, while the accumulate-and-push it adds to the o_proj
        # epilogue is ~1 us of genuinely new work on the critical path, which
        # is what SP6[0] and SP3[0] are reporting. Moving work off a stage that
        # already had free slack onto one that has none is a wash by
        # construction; no amount of tuning the fold flips that.
        #
        # Kept behind the flag rather than deleted: the plumbing is correct and
        # exercised, and it is the only worked example in the tree of reducing
        # a downstream contraction on an existing rendezvous.
        ROUTER_FOLD = (moe_ep and FUSE_FULL_LAYER
                       and os.environ.get("GLM_ROUTER_FOLD", "0") == "1")
        assert not MOE_MXFP4 or MOE_MXFP8, \
            "GLM_MOE_MXFP4 narrows the MXFP8 expert path; it is not a bf16 mode"
        # Only the fused tail carries the width through to the kernel. The
        # standalone gang_moe_w{13,2}_linear_mxfp8 layers still hard-code the
        # MXFP8 stride and would mis-address a nibble-packed weight, so they
        # are refused rather than silently wrong.
        assert not MOE_MXFP4 or FUSE_OPROJ_ROUTER, \
            "GLM_MOE_MXFP4 needs GLM_FUSE_OPROJ_ROUTER=1"
        if MOE_MXFP8:
            # Depth-4 MFMA pipeline: only the last of the four slots carries a
            # tail guard, so a partial final group would compute k-tiles that
            # do not exist.
            assert hidden_size % 512 == 0, hidden_size
            assert moe_inter % 512 == 0, moe_inter
            assert (2 * moe_inter) % MOE_W13_OPW == 0
            assert hidden_size % MOE_W2_OPW == 0
            # The K-parallel branch emits exactly 16 rows per workgroup and
            # splits MFMA_ITERS four ways, one group of four k-tiles minimum.
            for _nm, _opw, _k in (("W13", MOE_W13_OPW, hidden_size),
                                  ("W2", MOE_W2_OPW, moe_inter)):
                assert _opw % 64 == 0 or _opw == 16, \
                    f"MoE {_nm} OPW {_opw} is neither N- nor K-parallel"
                if _opw == 16:
                    assert (_k // 128) % 16 == 0, \
                        (f"MoE {_nm} K={_k} gives {_k // 128} MFMA iters, "
                         f"which does not split into 4 waves x a multiple of 4")

        # Same treatment for the two dense GEMMs that hang off an RMSNorm and
        # share one kernel: qkv_a (K=2048, N=2048, 394 MB/token) and the LM
        # head (K=2048, N=155136, 635 MB once). 1.03 GB of the 9.17 GB budget
        # between them. Both are already fused-RMSNorm tasks, so this is the
        # MXFP8 twin of gang_rmsnorm_linear_mxfp4_bias, not a call to the
        # plain dense kernel. Set GLM_DENSE_MXFP8=0 for bf16.
        DENSE_MXFP8 = os.environ.get("GLM_DENSE_MXFP8", "1") == "1"
        # Output rows per workgroup, and so the tile count. The LM head is
        # 155136 rows wide and wants the widest tile it can get; qkv_a is 2048,
        # which at 64 rows a workgroup is 4 tiles per XCD -- 32 of 240 workers.
        # The kernel's OPW<64 branch splits K across the 4 waves instead of N,
        # which is what un-starves it, the same trade #19 made for o_proj.
        DENSE_MXFP8_OPW = int(os.environ.get("GLM_DENSE_MXFP8_OPW", "64"))
        # 16 rather than 64: worth 6.30 -> 6.13 ms/token. It is the only
        # value the K-parallel branch is correct for.
        QKV_MXFP8_OPW = int(os.environ.get("GLM_QKV_MXFP8_OPW", "16"))
        # q_b gets its own switch: it is the one dense GEMM whose MXFP8 form
        # needs a fused-kvupd kernel of its own, so being able to A/B it
        # against the bf16 twin on a single build is worth a knob.
        QB_MXFP8 = DENSE_MXFP8 and os.environ.get("GLM_QB_MXFP8", "1") == "1"
        # o_proj likewise: it is the narrow-tile GEMV rather than an MFMA
        # kernel, so its MXFP8 form is a separate body too, and the knob keeps
        # the A/B on one build. Only meaningful while the GEMV is in use --
        # there is no MXFP8 CK tile.
        OPROJ_MXFP8 = os.environ.get("GLM_OPROJ_MXFP8", "1") == "1"
        if DENSE_MXFP8:
            assert hidden_size % 512 == 0, hidden_size
            # One workgroup per 64 output rows, partitioned 8 ways across the
            # XCDs. vocab_size is aligned to GANG_OUT_ALIGN (512) further down,
            # which is the same constraint, so only qkv_a needs checking here.
            assert qkv_a_pad % (8 * QKV_MXFP8_OPW) == 0, qkv_a_pad
        # q_b's OPW is forced to qk_rope by the fused rope slice, so it gets
        # no knob -- only a check that the per-XCD chunk holds whole heads.
        qb_out_width = qb_head_slots * qk_dim
        if not (qb_out_width % (8 * qk_rope) == 0
                and (qb_out_width // 8) % qk_dim == 0
                and qb_out_width % GANG_OUT_ALIGN == 0):
            qb_out_width = num_heads_pad * qk_dim
            qb_head_slots = num_heads_pad
        assert (num_heads_pad * qk_dim) % (8 * qk_rope) == 0
        assert ((num_heads_pad * qk_dim) // (8 * qk_rope) * qk_rope) % qk_dim == 0

        # ── un-absorbing W_UK from q_b, the Q-side mirror of W_UV ────────
        # Absorption widens q_b's per-head output from qk_nope + qk_rope to
        # kv_lora + qk_rope -- 256 to 576 on GLM-5 -- so [H*576, q_lora_pad] is
        # 77.9 MB of MXFP8 a layer against 34.6 for the plain weight plus 6.5
        # for the W_UK stack. Exact for the same reason the V side is: MLA's
        # QK product is linear in the cached latent.
        #
        # Only the whole-layer fused task has a Phase 3b to apply it in, so the
        # dense prologue layers keep the absorbed q_b. num_heads must be
        # unpadded and 8-aligned: q_b's per-XCD column chunk and W_UK's per-XCD
        # tile run have to land on the same eight heads, which is what lets
        # Phase 3b's barrier stay XCD-local.
        UNABSORB_K = (int(os.environ.get("GLM_UNABSORB_QB", "1")) == 1
                      and QB_MXFP8
                      and qk_nope < kv_lora
                      and num_heads == num_heads_pad
                      and num_heads % 8 == 0
                      and (num_heads * (qk_nope + qk_rope)) % GANG_OUT_ALIGN == 0)
        # Same lane arithmetic as W_UV: 256/ROWS lanes per row at 16 fp8 each,
        # so qk_nope % (256/ROWS * 16) == 0. At qk_nope = 192 that allows 64
        # (4 lanes, 64 elements a lane) and 128 (2 lanes, 32); 64 gives 512
        # tiles, 64 per XCD, eight per head.
        # 128 measured better than 64: 512 tiles -> 256, so each XCD's 29
        # workers make 2 grid-stride rounds instead of 3, and the GEMV inner
        # loop unrolls 2x. Worth 4.3 s of aggregate worker time in SP4[7].
        WUK_GEMV_ROWS = int(os.environ.get("GLM_WUK_GEMV_ROWS", "128"))
        # q_b's output row when W_UK is left out of it: the [nope | rope]
        # scratch Phase 3b reduces over, not the query row.
        qb_nope_span = qk_nope + qk_rope
        qb_nope_width = num_heads * qb_nope_span
        # q_b's GEMM tile width. It used to be pinned to qk_rope because a
        # head's rope slice had to be exactly one workgroup; the kernel now only
        # needs the slice to be the *tail* of one, so any divisor of the head
        # span that is at least the rope width works.
        #
        # 64 is measured best, and NOT for the reason the makespan model
        # predicts. 8 heads/XCD * 256 / OPW tiles (plus the latent tile) clear
        # 29 workers in ceil(tiles/29) grid-stride rounds, and 128 is the only
        # legal width that fits in one round -- 17 tiles instead of 33. Costing
        # a tile as (per-tile RMSNorm+quant prologue) + (bytes/14.5 GB/s per WG)
        # made that look like 17.3 us against 21.8.
        #
        # It measured the other way: 12.884 ms/iter at 128 against 12.261 at 64,
        # NP=8, same build, output correct in both. So a tile's cost is
        # markedly superlinear in OPW -- more than the ~2 us of lost overlap
        # from HOIST_PREFILL switching off at TILES_PER_WAVE == 2, so most
        # likely accumulator VGPR pressure on the N-parallel MFMA path. The
        # extra round is cheaper than the wider tile, and the makespan argument
        # does not survive contact.
        #
        # Absorbed q_b has a 576-wide head span, which only 64 divides, so this
        # knob only reaches the un-absorbed path either way.
        #
        # MEASURED 2026-08-19, after the head shard and the deferred rope: 16 is
        # best, 11.500 ms/iter against 11.805 at 64, same build, 4/4 on the
        # correctness suite. Sharded, an XCD owns one head, so 64 is four tiles
        # against 29 workers and the makespan is one whole 135 KB tile -- the
        # superlinearity that beat the makespan model above only bites when the
        # tiles outnumber the workers. 32 is not legal: the MXFP8 GEMM's
        # K-parallel branch (OPW < 64) emits exactly 16 rows per workgroup, so
        # the ladder is 16, then 64.
        QB_GEMM_OPW = int(os.environ.get("GLM_QB_OPW", "16"))
        if UNABSORB_K:
            assert kv_lora % WUK_GEMV_ROWS == 0
            assert qk_nope % ((256 // WUK_GEMV_ROWS) * 16) == 0
            assert (qb_nope_width // 8) % qb_nope_span == 0
            # The >= qk_rope floor was the rope's, not the GEMM's: the rotation
            # reads (2j, 2j+1) and writes (j, j + rope/2) across a head's whole
            # rope slice, so the slice could not straddle two workgroups with
            # no barrier between them. Un-absorbed, it no longer runs in this
            # GEMM at all -- the kernel defers it past Phase 3b's XCD-local
            # W_UK release -- so the only remaining constraint is the tile
            # width the MFMA path wants.
            assert QB_GEMM_OPW % 16 == 0
            assert qb_nope_span % QB_GEMM_OPW == 0, (
                f"GLM_QB_OPW={QB_GEMM_OPW} must divide the {qb_nope_span}-wide "
                "head span")

        assert (2 * dense_inter) % GANG_OUT_ALIGN == 0
        assert dense_inter % GANG_RED_ALIGN == 0

        # Split KV along the sequence and spread the (q_group, kv_chunk) work
        # items over the workers; the merge task recombines them. MLA has a
        # single shared latent head, so the gpt-oss `kv_head == xcd_id` mapping
        # does not carry over.
        #
        # This used to target exactly 8 work items, one per XCD, which left the
        # decode running 8-way parallel on a 240-worker machine -- 656 us of
        # exclusive critical path at peak parallelism 8. An XCD hosts 30
        # workers, not one, so the useful target is q_groups * chunks large
        # enough to fill them, bounded by how finely the KV actually splits.
        # Measured (ms/iter): 4 -> 7.127, 8 -> 7.038, 16 -> 6.995, 32 -> 7.512.
        # Past 16 the merge's own fan-out and the per-chunk prologue cost more
        # than the added parallelism returns.
        #
        # Re-measured (ms/token decode) once the decode and the merge moved
        # inside the fused attention task, as (chunks x merge_dim_splits):
        # 8x16 -> 4.987, 16x16 -> 4.902, 16x32 -> 4.912, 32x32 -> 5.457. The
        # fused task's dispatch width is pinned at 28 tiles/XCD by the q_b
        # phase, so the decode's ranks are resident either way and 8 chunks
        # left 26 of 28 of them parked at the barrier; 16 is still the knee
        # for the same reason it was before, and 32 falls off a cliff because
        # the merge reads NUM_KV_CHUNKS dependent LSE/o_acc columns per chunk.
        #
        # The 64-token tile below used to floor this at 8 for the 512-token
        # default sequence, i.e. it never reached the 16 the sweep above picked
        # both times. 32 tokens per chunk hits 16.
        #
        # 32 tokens per chunk is the wrong granularity at short sequences: at
        # max_seq_length=128 it yields 4 chunks, so decode runs on
        # q_groups*chunks = 16 tiles = 2/XCD, 7% of the 29 workers, and the
        # other 216 spin through it twice (decode->merge, then the Phase 8
        # barrier). MPK_MLA_SKIP_DECODE prices that whole complex at 3.40 ms of
        # the 14.189 baseline, and only ~0.5 ms of it is decode's own work.
        # Measured at max_seq_length=128, one variable, output verified:
        #   chunks=4  (32 tok/chunk, old default) -> 14.189 ms
        #   chunks=16 ( 8 tok/chunk)              -> 11.464 ms  (-2.73, -19.2%)
        # So occupancy dominates chunk granularity -- 8-token chunks are fine.
        # Cutting to 8 tokens per chunk reaches the 16 cap at any sequence >=
        # 128, which is where the older 16x32 sweep landed anyway.
        num_q_groups = num_heads_pad // 16
        _env_chunks = os.environ.get("GLM_MLA_NUM_KV_CHUNKS")
        if _env_chunks is not None:
            num_kv_chunks = int(_env_chunks)
        else:
            _kv_tiles = max(1, (args.max_seq_length + 7) // 8)
            num_kv_chunks = max(1, min(16, _kv_tiles))
        assert num_kv_chunks >= 1
        # The merge is otherwise one task per q group -- 2 CUs of 256, each
        # thread carrying kv_lora/16 = 32 unrolled softmax chains. Slice the
        # 512-wide latent so the merge fans out instead. 8 slices puts
        # kv_lora/8/16 = 4 dims on each thread across 16 tasks.
        # 32 under whole-layer fusion, 16 otherwise. Fusion puts the merge on
        # the Phase 8 barrier's critical path with 26 of 30 workers per XCD
        # parked behind it, so doubling merge_tiles_per_xcd from 4 to 8 pays
        # here in a way it never did when the merge was its own task: measured
        # 4.486 -> 4.115 ms fused, against 4.465 -> 4.493 unfused.
        #
        # 32 is the finest split the merge supports -- kv_lora / 32 / 16 = 1
        # dim per thread -- and getting there needed two fixes in
        # merge_splitkv_ck_fmha's write-through epilogue, see the comments
        # there. Without them this configuration silently corrupted half of
        # attn_out.
        MLA_MERGE_DIM_SPLITS = int(
            os.environ.get("GLM_MLA_MERGE_DIM_SPLITS",
                           "32" if FUSE_FULL_LAYER else "16"))
        assert kv_lora % (MLA_MERGE_DIM_SPLITS * 16) == 0
        # Off by default: measured a dead heat (8.025 vs 8.026 ms). The win it
        # buys gpt-oss is deleting a separate readback+flush pass inside the
        # fused layer task, and there is no such pass here while the merge is
        # still its own task. At dim_splits=16 each thread writes 2 bf16 and
        # the store path is not the bottleneck either way. Kept plumbed for
        # when the merge moves inside a fused MLA layer.
        # Forced on under whole-layer fusion: the merge writes only this
        # XCD's tiles of attn_out and o_proj reduces over the whole row, and
        # an in-kernel barrier -- unlike an event -- does not flush the
        # producing XCD's L2. The kernel static_asserts it.
        MLA_MERGE_WRITE_THROUGH = (
            os.environ.get("GLM_MLA_MERGE_WT", "0") == "1" or FUSE_FULL_LAYER)
        print(f"[CFG] q_heads={num_heads}->{num_heads_pad} "
              f"qb_slots={qb_head_slots} qb_opw={QB_GEMM_OPW} "
              f"q_groups={num_q_groups} kv_chunks={num_kv_chunks} "
              f"latent_row={qk_dim} q_lora={q_lora}->{q_lora_pad} "
              f"merge_dim_splits={MLA_MERGE_DIM_SPLITS} "
              f"fuse_moe_swiglu={int(FUSE_MOE_SWIGLU)} "
              f"fuse_moe_mulsumadd={int(FUSE_MOE_MULSUMADD)} "
              f"moe_mxfp8={int(MOE_MXFP8)} dense_mxfp8={int(DENSE_MXFP8)} "
              f"qb_mxfp8={int(QB_MXFP8)} "
              f"qkv_opw={QKV_MXFP8_OPW} dense_opw={DENSE_MXFP8_OPW}")

        num_workers, num_schedulers = mi.get_configurations_from_gpu(rank)

        # LM head. Pad the vocab to the gang GEMM's 512-column granularity,
        # then pick an argmax task count that divides the padded vocab.
        vocab_size = align_up(config.vocab_size, GANG_OUT_ALIGN)
        lm_head_weight = pad_rows(model.lm_head.weight.data, vocab_size)
        argmax_num_tasks = max_factor_leq_n(vocab_size, num_workers)
        print(f"[CFG] vocab {config.vocab_size} -> {vocab_size}, "
              f"argmax tasks={argmax_num_tasks} (workers={num_workers})")

        if args.profiling:
            profiler_tensor = torch.zeros(
                30000 * 1280, dtype=torch.uint64, device="cuda").contiguous()

        qo_indptr_buffer = torch.empty(
            args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
        paged_kv_indptr_buffer = torch.empty(
            args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
        paged_kv_indices_buffer = torch.zeros(
            args.max_num_pages, dtype=torch.int32, device="cuda")
        paged_kv_last_page_len_buffer = torch.empty(
            args.max_num_batched_requests, dtype=torch.int32, device="cuda")

        mpk = mi.PersistentKernel(
            mode="offline",
            world_size=world_size,
            mpi_rank=rank,
            num_workers=num_workers,
            num_local_schedulers=num_schedulers,
            num_remote_schedulers=0,
            max_seq_length=args.max_seq_length,
            max_num_batched_requests=args.max_num_batched_requests,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_pages=args.max_num_pages,
            page_size=args.page_size,
            eos_token_id=(eos_token_ids[0] if not args.ignore_eos
                          else 0x7FFFFFFF),
            meta_tensors={
                "step": step,
                "tokens": tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "num_new_tokens": num_new_tokens,
                "prompt_lengths": prompt_lengths,
                "qo_indptr_buffer": qo_indptr_buffer,
                "paged_kv_indptr_buffer": paged_kv_indptr_buffer,
                "paged_kv_indices_buffer": paged_kv_indices_buffer,
                "paged_kv_last_page_len_buffer": paged_kv_last_page_len_buffer,
            },
            profiler_tensor=profiler_tensor,
            trace_name=args.trace_name,
            spec_decode_config=None,
            use_cutlass_kernel=use_cutlass_kernel,
        )

        bs = args.max_num_batched_tokens

        # ── Intermediate tensors ─────────────────────────────────────────────
        x = mpk.attach_input(torch_tensor=input_tokens, name="input_token")
        cos_table = position_embeddings[0][0, :args.max_seq_length, :].contiguous()
        sin_table = position_embeddings[1][0, :args.max_seq_length, :].contiguous()
        cos_pos_embed = mpk.attach_input(torch_tensor=cos_table,
                                         name="cos_position_embedding")
        sin_pos_embed = mpk.attach_input(torch_tensor=sin_table,
                                         name="sin_position_embedding")

        # Intermediate tensors: always use attach_input (PyTorch-backed) to
        # avoid megakernel internal memory aliasing that causes wrong output.
        _tensor_refs = {}  # prevent GC of backing tensors

        def make_tensor(name, dims, torch_dtype=torch.bfloat16):
            t = torch.zeros(dims, dtype=torch_dtype, device="cuda")
            _tensor_refs[name] = t
            return mpk.attach_input(torch_tensor=t, name=name)

        GANG_TILE_N = int(os.environ.get("GANG_TILE_N", "64"))
        GANG_WGM = int(os.environ.get("GANG_WGM", "0"))
        # o_proj split-K. The split-K kernel tiles K at KPerBlock = 256, so a
        # split is only legal when it divides the reduction into whole 256-
        # element steps; otherwise fall back to the output-parallel GEMM.
        # Default off: measured 10.70 ms/token at k_splits=4 and 10.86 at 8,
        # against 10.16 with the plain output-parallel GEMM. Decode latency
        # here tracks the *number of ops in the task chain*, not tasks per op,
        # so spending more workers on one op only adds merge traffic.
        #
        # Re-measured at the 8.025 ms baseline, after reduction_override let
        # split-K compose with the de-padded weight: 9.289 / 8.525 / 8.620 ms
        # at k = 2 / 4 / 8. Still a clear loss, and the starvation it is aimed
        # at is real -- o_proj holds 23% of the token's bytes and 21% of the
        # critical path on 32 of 240 workers, moving 1.13 TB/s where the
        # 240-worker MoE W13 manages 2.6. The partials are the problem, not
        # the parallelism: every split round-trips fp32 through a global
        # workspace and then spins on a done counter. Fixing this properly
        # means merging the splits in LDS inside a fused layer task, the way
        # gpt-oss's gang_full_layer_fused does, not through HBM.
        GANG_K_SPLITS = int(os.environ.get("GANG_K_SPLITS", "1"))
        # The absorbed o_proj reduces over the attention output, one kv_lora
        # row per q head. Padded heads carry zero o_absorbed columns, so on a
        # checkpoint whose head count is not a multiple of 16 the GEMM spends
        # (num_heads_pad - num_heads) / num_heads_pad of its weight traffic --
        # 37.5% on GLM-4.7-Flash's 20 -> 32 -- multiplying by zeros. It is the
        # largest single op in the decode chain, so that waste is worth
        # removing. The only rule this GEMM puts on its reduction is
        # KPerBlock = 256, and the real heads occupy the leading columns, so
        # when num_heads * kv_lora is already 256-aligned we can hand it the
        # unpadded weight and stop the reduction short of the padding.
        # The q side gets the same treatment through qb_head_slots above, but
        # by narrowing the output row rather than the reduction: q_b's output
        # is what MLA decode indexes into, so it can only shrink in whole
        # heads and only down to an 8-aligned slot count.
        o_proj_red = num_heads * kv_lora
        if o_proj_red % 256 != 0:
            o_proj_red = num_heads_pad * kv_lora
        # ── or: don't absorb W_UV at all ──────────────────────────────────
        # Absorption trades bytes for ops, and at GLM-5 on 8 ranks the layer
        # is bytes. Per layer per GPU, MXFP8: qkv_a 16.1 M, q_b 75.5 M,
        # o_proj 201.3 M -- 302 MB of attention weight against ~40 MB of
        # EP-sliced expert weight, so o_proj alone is 60% of the token. Its
        # reduction is num_heads * kv_lora only *because* W_UV is folded in;
        # un-folded it is num_heads * v_head, and GLM-5's v_head (256) is half
        # its kv_lora (512). 201.3 M -> 100.7 M, plus 8.4 M for the W_UV stack
        # itself: -95 MB a layer, -7.4 GB a token over 78 layers.
        #
        # What it costs is a block-diagonal GEMV
        #   v[h * v_head + j] = sum_c attn_out[h * kv_lora + c] * W_UV[h][j][c]
        # ahead of o_proj, and one cross-XCD barrier between the two. Only the
        # whole-layer fused task implements it (Phase 8b); the dense prologue
        # layers keep the absorbed weight, which is why both widths are built.
        #
        # Exact, not an approximation: MLA's PV accumulation is linear in the
        # cached latent, so applying W_UV before o_proj and folding it into
        # o_proj differ only in rounding.
        o_proj_red_absorbed = o_proj_red
        o_proj_red_unabsorbed = num_heads * v_head
        UNABSORB_V = (int(os.environ.get("GLM_UNABSORB_OPROJ", "1")) == 1
                      and v_head < kv_lora
                      and o_proj_red_unabsorbed % 256 == 0
                      # Phase 8b walks attn_out head by head at a stride of
                      # kv_lora, so a padded head count would need the V row
                      # padded to match. GLM-5's 64 heads need neither.
                      and num_heads == num_heads_pad)
        # 256/ROWS lanes per row and 16 fp8 each, so ROWS >= 4 and
        # kv_lora % (256/ROWS * 16) == 0. Retuned after o_proj was sharded
        # (ms/iter, everything else at the tuned point): 64 -> 12.533,
        # 128 -> 12.231, 256 -> 12.881. 128 is 16384/128/8 = 16 tiles per XCD,
        # half a round of 29 workers; 256 halves that again and the phase goes
        # latency-bound.
        WUV_GEMV_ROWS = int(os.environ.get("GLM_WUV_GEMV_ROWS", "128"))
        # WUV_MFMA=1 takes W_UV off the bf16-dot GEMV and onto
        # v_mfma_scale_f32_16x16x128_f8f6f4 with an FP8 activation, which on
        # gfx950 is the only fp8-activation instruction there is (the VALU fp8
        # dots need dot11-insts, which gfx942 has and gfx950 dropped).
        #
        # Off by default: it is a measured regression, and the standalone probe
        # that predicted a win was run in the wrong regime.
        #
        # A/B at a fixed 64 rows (so the tile count is not the variable), rank
        # 0, MPK_SUBPHASE_TIMING=1, ms/iter:
        #
        #   counter          GEMV     MFMA     delta
        #   SP0[7] loop      0.367    0.463    +26.3%
        #   SP2[7] barrier   0.480    0.601    +25.4%
        #   SP3[7] both      0.870    1.087    +24.9%
        #   decode wall     14.255   14.524    +1.9%
        #
        # tests/standalone/test_fp8_act_mfma_bw.hip measured +12% for the same
        # kernel swap -- at K=10240, i.e. MFMA_ITERS=80. W_UV reduces over
        # K=512, which is MFMA_ITERS=4, and the kernel's software pipeline is
        # depth 4 (static_assert MFMA_ITERS >= 4 && % 4 == 0). So W_UV sits at
        # the exact minimum: every iteration is pipeline fill, none is steady
        # state, and the FP8 activation's amax reduction is a fixed prologue
        # cost with only 33.8 KB of weight per WG to amortize it over. The
        # streaming rate the probe measured is a property of the steady state
        # this shape never reaches.
        #
        # It also costs the tuned row width. gang_linear_mxfp8_kernel's
        # OUTPUT_PER_WG is a hardcoded 4 waves x 16 rows, so this pins 64, and
        # 64 is the width the sweep above rejected (32 tiles per XCD against 29
        # workers is two grid-stride rounds where 128's 16 tiles is one) -- a
        # further -0.30 ms on top of the numbers above, which were taken at 64
        # on both sides.
        #
        # Turning this back on needs a shallow-K pipeline variant, not a knob.
        if int(os.environ.get("WUV_MFMA", "0")) == 1:
            if os.environ.get("GLM_WUV_GEMV_ROWS") is None:
                WUV_GEMV_ROWS = 64
            assert WUV_GEMV_ROWS == 64, (
                "WUV_MFMA needs GLM_WUV_GEMV_ROWS=64; got "
                f"{WUV_GEMV_ROWS}")
        # The other way to un-starve o_proj, and the one that works: keep the
        # op output-parallel but make the tiles narrower. The 64-column floor
        # came from the 16x64x256 MFMA tile, and at batch 1 that tile was
        # discarding 15 of its 16 rows to begin with, so dropping MFMA for a
        # plain GEMV costs nothing and lets tile_n go to 8. That is 256 tiles
        # instead of 32, and the partial sums never leave registers.
        # Measured sweep at K=10240, out=2048 (ms/iter): 4 -> 7.185, 8 -> 7.354,
        # 16 -> 7.135, 32 -> 7.788, 64 -> 9.141 (CK MFMA at 64: 7.896). Not
        # monotonic, because tile count has to land well on 30 workers/XCD: 16
        # gives 16 tiles and one clean round, 8 gives 32 tiles and spends a
        # whole second round on two of them. 0 restores the CK path.
        _oproj_rows_env = os.environ.get("GLM_OPROJ_GEMV_ROWS")
        OPROJ_GEMV_ROWS = int(_oproj_rows_env) if _oproj_rows_env else 16
        use_gemv_oproj = OPROJ_GEMV_ROWS > 0 and not (
            GANG_K_SPLITS > 1 and o_proj_red % (GANG_K_SPLITS * 256) == 0)
        # Everything the tp predicate below tests except the divisibility,
        # which needs the tile width that this decides. Sharding cuts the
        # columns by world_size, so at an unchanged tile width it cuts the tile
        # count by the same factor and leaves most of the XCD idle: at GLM-5
        # tile_n=16 gives 6 tiles against 29 workers. Give the factor back to
        # the tile, down to the 4-row floor (256/ROWS lanes at 16 fp8 each).
        # Measured, sharded, ms/iter: 16 -> 13.057, 8 -> 12.598, 4 -> 12.533.
        oproj_tp_eligible = (int(os.environ.get("GLM_OPROJ_TP", "1")) == 1
                             and moe_ep and world_size > 1
                             and FUSE_FULL_LAYER and OPROJ_MXFP8
                             and use_gemv_oproj)
        if oproj_tp_eligible and _oproj_rows_env is None:
            OPROJ_GEMV_ROWS = max(4, OPROJ_GEMV_ROWS // world_size)
        oproj_tile_n = OPROJ_GEMV_ROWS if use_gemv_oproj else GANG_TILE_N
        # 1.97 GB/token of bf16 weight, the last big one left. The packing is
        # the same pack_dense_mxfp8 the MFMA kernels use: its data half is
        # plain row-major, and only the MFMA *gather* wanted the split layout,
        # so the GEMV reads it as-is. See gang_gemv_mxfp8_mi300.cuh.
        use_mxfp8_oproj = OPROJ_MXFP8 and use_gemv_oproj
        if OPROJ_MXFP4 and not (use_mxfp8_oproj and FUSE_FULL_LAYER):
            # Only the fused whole-layer task calls the MXFP4 GEMV; the dense
            # prologue's o_proj builders still template on the MXFP8 body and
            # would mis-address a nibble-packed row rather than fail to build.
            # Same restriction, and the same reason, as GLM_MOE_MXFP4's.
            #
            # Silently fall back rather than assert, because the default is on:
            # a bf16 or unfused configuration is a legal way to run this demo
            # and should not be broken by a default. Write the decision back
            # into the environment, because persistent_kernel.py reads the SAME
            # variable to set -DMPK_OPROJ_MXFP4 and the two must agree -- a
            # host packer and a kernel that disagree on the row width produce
            # garbage rather than a build error.
            os.environ["GLM_OPROJ_MXFP4"] = "0"
            OPROJ_MXFP4 = False
            print("[CFG] GLM_OPROJ_MXFP4 off: needs the MXFP8 GEMV o_proj "
                  "inside the fused whole-layer task "
                  f"(mxfp8={use_mxfp8_oproj} fused={FUSE_FULL_LAYER})")
        if use_mxfp8_oproj:
            # Elements per lane per iteration over 256/rows lanes: 16 E4M3
            # bytes, or 32 E2M1 nibbles, out of the same 16-byte load. Only
            # o_proj's own reduction changes format -- W_UV stays MXFP8, so
            # its checks keep the 16.
            oproj_per_lane = 32 if OPROJ_MXFP4 else 16
            assert o_proj_red % (
                (256 // oproj_tile_n) * oproj_per_lane) == 0, o_proj_red
            if UNABSORB_V:
                assert o_proj_red_unabsorbed % (
                    (256 // oproj_tile_n) * oproj_per_lane) == 0, \
                    o_proj_red_unabsorbed
                assert v_head % WUV_GEMV_ROWS == 0
                assert kv_lora % ((256 // WUV_GEMV_ROWS) * 16) == 0
        if UNABSORB_V and not use_mxfp8_oproj:
            # Phase 8b only exists inside the MXFP8 GEMV o_proj.
            UNABSORB_V = False
        # ── rank-sharded o_proj ───────────────────────────────────────────
        # o_proj is the largest replicated weight in the layer: 103.8 MB per
        # rank per layer at GLM-5's shape, read identically on all 8 ranks
        # because attention is data-parallel at batch 1. Output-wise sharding
        # gives rank p rows [p * hidden/EP, +that) and therefore exactly those
        # columns of the hidden row; the columns are disjoint, so the row is
        # completed by an all-gather rather than a reduction, and the residual
        # add stays inside the one rank that owns each column.
        #
        # There is no kernel flag: the task detects the shard from its own tile
        # count (see oproj_tp in gang_oproj_router_fused_mi300.cuh), so this
        # slice is the entire switch. It needs the fused whole-layer task,
        # which is the only path with the symmetric signal array to rendezvous
        # on, and per-XCD tiles that stay whole.
        oproj_tp = (oproj_tp_eligible and use_mxfp8_oproj
                    and hidden_size % (world_size * 8 * oproj_tile_n) == 0)
        oproj_tp_cols = hidden_size // world_size if oproj_tp else hidden_size
        print(f"[CFG] o_proj tp={int(oproj_tp)} cols_per_rank={oproj_tp_cols} "
              f"tiles_per_xcd={oproj_tp_cols // 8 // oproj_tile_n}")
        # ── rank-sharded q_b + W_UK ───────────────────────────────────────
        # The same trade as o_proj, on the second-largest replicated weight:
        # q_b is 34.6 MB and W_UK 6.6 MB per rank per layer, both read
        # identically on all 8 ranks because attention is data-parallel.
        # Here the motive is makespan rather than bytes. q_b dispatches 33
        # tiles per XCD (32 GEMM at 10.28 us plus tile 0's latent append)
        # against 29 workers, so 25 of 29 idle a whole tile -- 8.9 of the
        # 12.4 us mean spin at the W_UK barrier. The tile count is locked by
        # shape: OPW must divide the 256-wide head span and be >= qk_rope, so
        # the legal per-XCD counts are 32/16/8/4 and never 29, and narrowing
        # loses because the tile is already at 67% of the per-CU HBM roof.
        # Only fewer heads per XCD removes the second round.
        #
        # Rank p owns heads [p * H/EP, +that), one head per (rank, XCD). Both
        # scratch rows stay declared at the full head count on every rank;
        # only the weights are sliced, and the kernel pushes each W_UK output
        # tile into the peers' copy of q_workspace as it lands, then folds the
        # all-gather rendezvous into the existing Phase-4 barrier. As with
        # o_proj there is no kernel flag -- the task reads the shard off its
        # own tile ratio -- so this slice plus the symmetric q_workspace is
        # the entire switch.
        #
        # Not extended to the MLA decode: NUM_Q_GROUPS = heads/16 would go to
        # 0 at 8 heads. Phase 5 onwards still sees all 64.
        qb_tp = (int(os.environ.get("GLM_QB_TP", "1")) == 1
                 and moe_ep and world_size > 1
                 and FUSE_FULL_LAYER and UNABSORB_K
                 # One whole head per (rank, XCD), at minimum.
                 and num_heads % (world_size * 8) == 0
                 and (num_heads // world_size) * qb_nope_span
                 % GANG_OUT_ALIGN == 0
                 # W_UK's tiles have to stay a whole number per XCD too.
                 and ((num_heads // world_size) * kv_lora
                      // WUK_GEMV_ROWS) % 8 == 0)
        qb_tp_heads = (num_heads // world_size) if qb_tp else num_heads
        print(f"[CFG] q_b/W_UK tp={int(qb_tp)} heads_per_rank={qb_tp_heads} "
              f"qb_tiles_per_xcd={qb_tp_heads * qb_nope_span // 8 // QB_GEMM_OPW}"
              f" wuk_tiles_per_xcd={qb_tp_heads * kv_lora // 8 // WUK_GEMV_ROWS}")
        n_tiles_xcd = hidden_size // 8 // oproj_tile_n
        # The split-K task takes a reduction_override now, so de-padding and
        # split-K compose: it reduces over the leading o_proj_red columns and
        # splits *those* k_splits ways.
        use_splitk_oproj = (GANG_K_SPLITS > 1
                            and o_proj_red % (GANG_K_SPLITS * 256) == 0)
        print(f"[CFG] o_proj unabsorb_v={int(UNABSORB_V)} "
              f"K_absorbed={o_proj_red_absorbed} "
              f"K_unabsorbed={o_proj_red_unabsorbed} "
              f"wuv_rows={WUV_GEMV_ROWS}")
        print(f"[CFG] o_proj K={o_proj_red} out={hidden_size} "
              f"tile_n={oproj_tile_n} gemv={int(use_gemv_oproj)} "
              f"mxfp8={int(use_mxfp8_oproj)} "
              f"n_tiles/XCD={n_tiles_xcd} split-K="
              f"{GANG_K_SPLITS if use_splitk_oproj else 1} -> "
              f"{n_tiles_xcd * (GANG_K_SPLITS if use_splitk_oproj else 1) * 8}"
              f" tasks (of {num_workers} workers)")

        y = make_tensor("embed_out", (bs, hidden_size))
        # MPK_QKV_PRO_HOIST publishes the quantized qkv_a row into the tail of
        # this buffer, past the (bs, hidden_size) bf16 it nominally holds. Per
        # XCD: bs (E4M3 row, one E8M0 per 128) pairs, then one 256 B line per
        # token for the 6 slice partials and the 6 epoch stamps. The layout is
        # duplicated in gang_mla_attn_fused_mi300.cuh -- keep the two in step.
        # Always allocated: it is 53 KB against a 744B model, and sizing the
        # buffer off a compile-time knob the Python side does not see is how a
        # silent out-of-bounds store gets written.
        # The room is taken as extra ROWS, not a wider row: the kernel finds the
        # tail at bs * hidden_size * 2 bytes, so the row stride has to stay
        # hidden_size or that offset lands inside token 1's activations.
        _pub_tok = hidden_size + hidden_size // 128
        _pub_part_off = ((_pub_tok * bs + 127) // 128) * 128
        _pub_stride = ((_pub_part_off + bs * 256 + 255) // 256) * 256
        _pub_rows = -(-8 * _pub_stride // (hidden_size * 2))
        rmsnorm_out = make_tensor(
            "rmsnorm_out", (bs + _pub_rows, hidden_size))
        qkv_a_out = make_tensor("qkv_a_out", (bs, qkv_a_pad))
        q_a_norm_out = make_tensor("q_a_norm_out", (bs, qkv_a_pad))
        # The absorbed q_b_proj writes the roped Q straight into this, so the
        # separate q_absorbed staging buffer the KV update used to read is gone.
        #
        # The row is declared qb_out_width wide but allocated at the full
        # num_heads_pad, and the two widths do different jobs.
        #
        # The declared width is what the gang partitions: it comes from the
        # output tensor's own dim(1), not from output_stride, so it has to be
        # qb_out_width for q_b's live heads to land contiguously from slot 0.
        # Partitioning a 32-slot row instead would give XCD x three heads at
        # slot 4x -- heads at {0,1,2, 4,5,6, ...}, a hole every fourth slot --
        # and o_proj's de-padded reduction reads the leading num_heads *
        # kv_lora columns of the attention output assuming there is no hole.
        #
        # The allocation stays 32 slots because MLA decode reduces 16 q heads
        # per MFMA group and its second group loads slots 16..31 whatever the
        # declared width says. Those tail slots read the zeros torch.zeros put
        # there and contribute nothing, exactly as they do today; the only
        # change is that q_b stops rewriting them with the zeros they already
        # hold. MLA maps q_workspace replicated rather than partitioned, so
        # the declared width never reaches its addressing.
        #
        # Both tasks are handed this one DTensor rather than two aliases over
        # the same storage: the runtime derives task-graph edges from shared
        # tensor ids (src/kernel/runtime.cc:594), so an alias drops the
        # q_b -> MLA edge. Going through .view() rather than a 2-D slice keeps
        # the row stride equal to the row width, which attach_input requires.
        if qb_tp:
            # Under the head shard each rank fills only its own 1/EP-th of
            # this row and pushes those columns into the peers' copies, so it
            # has to sit on the symmetric heap for the peer store to land at
            # the same address. One buffer for all layers, safe for the same
            # reason attn_proj_out's is: a peer cannot reach layer L+1's
            # Phase 3b without passing the layer-L MoE fold, which needs this
            # rank's layer-L MoE output, which needs this rank's layer-L
            # decode to have read the row.
            #
            # Declared at the full width -- UNABSORB_K already forces
            # num_heads == num_heads_pad, so qb_out_width is the whole row and
            # the padded/unpadded distinction below does not arise.
            assert qb_out_width == num_heads_pad * qk_dim, (
                qb_out_width, num_heads_pad * qk_dim)
            mla_q_ws = mpk.new_tensor(
                dims=(bs, qb_out_width),
                dtype=mi.bfloat16,
                name="mla_q_workspace",
                io_category="nvshmem_tensor",
            )
        else:
            _q_ws_full = torch.zeros((bs, num_heads_pad * qk_dim),
                                     dtype=torch.bfloat16, device="cuda")
            _tensor_refs["mla_q_workspace_alloc"] = _q_ws_full
            _q_ws_row = _q_ws_full.view(-1)[:bs * qb_out_width].view(
                bs, qb_out_width)
            _tensor_refs["mla_q_workspace"] = _q_ws_row
            mla_q_ws = mpk.attach_input(torch_tensor=_q_ws_row,
                                        name="mla_q_workspace")
        # LSE is written unconditionally by the decode kernel; its stride is
        # num_q_groups * num_kv_chunks * 16 == num_heads_pad * num_kv_chunks.
        mla_lse = make_tensor("mla_lse", (bs, num_heads_pad * num_kv_chunks),
                              torch_dtype=torch.float32)
        attn_out = make_tensor("attn_out", (bs, num_heads_pad * kv_lora))
        if num_kv_chunks > 1:
            # Per-chunk float partials; the merge task combines them.
            mla_o_acc = make_tensor(
                "mla_o_acc", (bs, num_heads_pad * num_kv_chunks * kv_lora),
                torch_dtype=torch.float32)
        else:
            mla_o_acc = None
        # Phase 8b's output and o_proj's input when W_UV is un-absorbed: the
        # per-head V slice, num_heads * v_head wide against attn_out's
        # num_heads_pad * kv_lora.
        mla_v_out = (make_tensor("mla_v_out", (bs, o_proj_red_unabsorbed))
                     if UNABSORB_V else None)
        # Phase 3's output and Phase 3b's input when W_UK is un-absorbed: the
        # per-head [nope | rope] row. The rope columns are written (that is
        # where the GEMM puts them) and never read -- the rotation lands in
        # the query row instead -- but they keep the head span a whole number
        # of q_b workgroups, which is what makes the rope slice one workgroup.
        mla_q_nope = (make_tensor("mla_q_nope", (bs, qb_nope_width))
                      if UNABSORB_K else None)
        if oproj_tp:
            # Under rank-sharded o_proj every rank computes a disjoint 1/EP-th
            # of this row and pushes it straight into the peers' copies, so it
            # has to live on the symmetric heap -- a plain cudaMalloc would put
            # the peer store at an unrelated address on the remote rank.
            #
            # One buffer for all layers, not one per layer as ep_gather needs.
            # The hazard ep_gather guards against -- a peer running ahead and
            # overwriting layer L's row while this rank still reads it -- is
            # closed here by the MoE fold: a peer cannot reach layer L+1's
            # o_proj until it has seen this rank's layer-L partial, which this
            # rank publishes at the head of layer L+1, i.e. after its own
            # layer-L router and MoE have finished reading the row.
            attn_proj_out = mpk.new_tensor(
                dims=(bs, hidden_size),
                dtype=mi.bfloat16,
                name="attn_proj_out",
                io_category="nvshmem_tensor",
            )
        else:
            attn_proj_out = make_tensor("attn_proj_out", (bs, hidden_size))
        # Split-K accumulator for the absorbed o_proj. Absorption widens that
        # GEMM's reduction to num_heads_pad * kv_lora (16384 on Flash) while
        # its output stays at hidden_size, and gang GEMMs only split the
        # output: hidden/8 XCDs/tile_n = 4 tiles per XCD, i.e. 32 of 240
        # workers. Splitting K as well brings the rest of the machine in; the
        # partials merge through this f32 buffer.
        #
        # The buffer is *wider than hidden_size on purpose*. Each XCD gets
        # dim 1 / 8, and `register_gang_splitk_linear_res_mi300_task` puts that
        # XCD's per-n_tile done counters immediately after its float data, at
        # `input_ptrs[3] + bs * n_tiles * tile_n`. Sized at exactly
        # hidden_size, that lands inside the *next* XCD's accumulator (and off
        # the end for XCD 7), which silently corrupts o_proj -- the decode
        # degenerates to a single repeated token. So give every chunk
        # n_tiles_xcd extra floats to hold its own counters.
        #
        # The kernel's row stride is n_tiles*tile_n and its rows are
        # contiguous *inside the XCD chunk*, so at bs > 1 the chunk is
        # bs*n_tiles_xcd*tile_n floats then n_tiles_xcd counters -- which a
        # (bs, W) tensor does NOT lay out, because dim-1 partitioning would
        # interleave the XCDs between rows. Declare it as one row and let the
        # (1,-1,-1) input map cut it into 8 contiguous chunks; at bs == 1 this
        # is byte-identical to the old shape.
        splitk_ws = make_tensor("splitk_workspace",
                                (1, bs * hidden_size + 8 * n_tiles_xcd),
                                torch_dtype=torch.float32)
        rmsnorm_out_moe = make_tensor("rmsnorm_out_moe", (bs, hidden_size))
        layer_out = make_tensor("layer_out", (bs, hidden_size))
        # The dense prologue layer has one more residual add than a MoE layer,
        # so it lands on the wrong side of the layer_out / attn_proj_out
        # ping-pong that the fused residual resolve sets up. One extra row
        # buys the parity back and keeps every MoE layer identical; see the
        # FUSE_RESADD wiring below.
        dense_resid = make_tensor("dense_resid", (bs, hidden_size))

        dense_mid = make_tensor("dense_mid", (bs, 2 * dense_inter))
        dense_act = make_tensor("dense_act", (bs, dense_inter))

        # Routing tables are sized for the total expert / slot count; the
        # router derives `num_shared` from the gap between these and the
        # logits width.
        moe_gate_out = make_tensor("moe_gate_out", (bs, num_experts))
        moe_routing_indices = make_tensor(
            "moe_routing_indices", (num_experts_total, bs),
            torch_dtype=torch.int32)
        moe_mask = make_tensor("moe_mask", (num_experts_total + 1,),
                               torch_dtype=torch.int32)
        moe_topk_weight = make_tensor("moe_topk_weight", (bs, topk_total),
                                      torch_dtype=torch.float32)
        # Cross-XCD arrival counter for the fused router. The last of the
        # num_experts workers to land runs the TopK tail, then resets this to
        # 0 for the next layer, so one buffer serves all of them.
        router_topk_counter = make_tensor("router_topk_counter", (1,),
                                          torch_dtype=torch.int32)
        # Barrier slots for the fused o_proj+router+MoE task, each on its own
        # 64-byte line: the o_proj->router barrier's 8 release flags plus its
        # arrival counter at [0..8], the routing-ready epoch and its 8 per-XCD
        # flags at [10..18], and the W13->W2 barrier at [20..28]. All monotonic
        # and never reset, so all 47 layers share one buffer.
        oproj_router_counter = make_tensor("oproj_router_counter", (29 * 16,),
                                           torch_dtype=torch.int32)
        # Same 29-slot layout for the fused attention task: qkv_a -> q_b at
        # [0..8], q_b -> decode at [10..18], decode -> merge at [20..28]. A
        # separate buffer from the MoE half's, since the two tasks are in
        # flight against each other across the layer boundary.
        attn_fused_counter = make_tensor("attn_fused_counter", (29 * 16,),
                                         torch_dtype=torch.int32)
        # Whole-layer fusion collapses those three buffers into one, not for
        # tidiness but because the merged input list is 27 slots against a
        # MAX_INPUTS_PER_TASK of 34 (28 before EP needed three more slots).
        # Slot map (each on its own 64-byte line):
        # attention qkv_a->q_b [0..9], q_b->decode [10..19],
        # decode->merge [20..29], attention->o_proj [30..39],
        # o_proj->router [40..49], routing-ready epoch [50..59],
        # W13->W2 [60..69], the router's own TopK arrival counter at [70],
        # and the multi-layer layer-entry barrier at [71..79] (per-XCD release
        # flags at [71..78], arrival counter at [79]).
        #
        # Expert parallelism adds the EP exit barrier: per-XCD release flags at
        # [80..87] and the folding work-groups' arrival counter at [88]. Those
        # slots are allocated unconditionally -- 256 bytes of int32 is not worth
        # a conditional, and undersizing a counter buffer corrupts whatever
        # torch allocated next rather than failing loudly (gpt-oss lost time to
        # exactly that; see MULTI_GPU_NOTES.md §3c, counter buffer 832 -> 1216).
        #
        # Must match FULL_LAYER_COUNTER_SLOTS in
        # gang_mla_full_layer_fused_mi300.cuh.
        # Un-absorbed kv_b_v adds Phase 8b's barrier: per-XCD release flags at
        # [96..103] and the arrival counter at [104]. Un-absorbed kv_b_k adds
        # Phase 3b's, which is XCD-local: for XCD x the flag at [106 + x] and
        # its arrival counter eight lines further on, i.e. [106..113].
        full_layer_counter = make_tensor(
            "full_layer_counter",
            # 314, not 114. Two compile-time-optional regions are reserved
            # unconditionally, so the host allocation never depends on a flag
            # -- the ranks would otherwise disagree about the buffer length:
            #   [114 .. 217]  MPK_BAR_TREE's per-XCD arrival counters, eight
            #                 per barrier at `barrier base + 114`
            #   [218 .. 313]  MPK_NULL_PHASES, four rendezvous x 24 lines
            # Keep in step with FULL_LAYER_COUNTER_SLOTS in
            # gang_mla_full_layer_fused_mi300.cuh.
            (((218 + 4 * 24) if UNABSORB_K else 106 if UNABSORB_V else 96) * 16,),
            torch_dtype=torch.int32)
        # ── the EP exchange buffers ──────────────────────────────────────
        # One gather buffer PER FUSED LAYER, plus one for the tail. The fold
        # is at the head of layer L+1 and Phase 1 reads the same buffer in
        # the same layer, which reads oddly until you notice the ordering: a
        # peer that has cleared the wait is free to run ahead and fold layer
        # L+1 while this rank is still reading layer L's row, and nothing
        # orders those two. One buffer per layer is what removes the hazard.
        # 8 x 2048 bf16 is 32 KB a layer on Flash.
        #
        # The signal array is SHARED by every layer and must be: its
        # thresholds ride the run-monotonic layer counter, so it accumulates
        # exactly one store per (peer, layer) and there is nothing to reset. A
        # per-layer array would only ever reach 1 while layer 1 waited for 2.
        #
        # Both live in the symmetric heap: the fold addresses a peer by heap
        # delta, so a plain torch allocation would land the write somewhere
        # unrelated on the remote rank.
        ep_gather_list = []
        ep_signal = None
        router_partials = None
        if moe_ep:
            ep_gather_list = [
                mpk.new_tensor(
                    dims=(world_size, bs, hidden_size),
                    dtype=mi.bfloat16,
                    name=f"ep_gather_{li}",
                    io_category="nvshmem_tensor",
                )
                # num_layers buffers for the fused layers (the dense prologue
                # layers never use theirs) and one more for the tail.
                for li in range(num_layers + 1)
            ]
            # uint64 counters, one 64-byte line per PE so two peers' stores
            # never share a line. Declared int32 because that is what
            # get_datatype_size() supports; only the byte count matters and
            # the kernel reinterprets the base pointer as uint64*.
            ep_signal = mpk.new_tensor(
                dims=(world_size * 16,),
                dtype=mi.int32,
                name="ep_signal",
                io_category="nvshmem_tensor",
            )
            if ROUTER_FOLD:
                # The router fold's reduction scratch. Lines [0..7] are this
                # rank's per-XCD accumulators, cleared by whichever block wins
                # the o_proj barrier election; lines [8..] are the per-rank
                # sums, pushed peer-to-peer on that same rendezvous and
                # double-buffered by layer parity. Element num_experts of a
                # line is the RMSNorm's sum-of-squares, which rides along
                # because it reduces over the identical partition.
                #
                # One buffer for all layers, not one per layer like ep_gather:
                # the whole thing is produced and consumed inside a single
                # layer's Phase 1-3, and the parity pair already covers the
                # only overlap (a peer running ahead into its next layer).
                router_partials = mpk.new_tensor(
                    dims=(8 + 2 * world_size, num_experts + 1),
                    dtype=mi.float32,
                    name="router_partials",
                    io_category="nvshmem_tensor",
                )
        moe_mid = make_tensor("moe_mid", (bs, topk_total, 2 * moe_inter))
        moe_act = make_tensor("moe_act", (bs, topk_total, moe_inter))
        moe_out = make_tensor("moe_out", (bs, topk_total, hidden_size))
        # atomicAdd target for the fused W2 epilogue. Zero-initialised here
        # and re-zeroed by moe_residual_add_f32 as it consumes each layer.
        moe_ws_f32 = make_tensor("moe_ws_f32", (bs, hidden_size),
                                 torch_dtype=torch.float32)

        argmax_in = make_tensor("argmax_in", (bs, vocab_size))
        argmax_part_value = make_tensor("argmax_part_value",
                                        (bs, argmax_num_tasks))
        argmax_part_index = make_tensor("argmax_part_index",
                                        (bs, argmax_num_tasks),
                                        torch_dtype=torch.int64)
        argmax_out = mpk.attach_input(torch_tensor=output_tokens,
                                      name="output_token")

        # Zero biases. gang_rmsnorm_linear_bias_layer partitions `bias` with
        # map (1, -1, -1), so it has to be 2-D [1, output_size]; the MoE gang
        # GEMMs want [num_experts, output_stride]. GLM has no GEMM biases, so
        # every one of these is a constant zero -- one shared tensor per shape.
        _bias_cache = {}

        def zero_bias(size):
            if size not in _bias_cache:
                t = torch.zeros(1, size, dtype=torch.bfloat16, device="cuda")
                _tensor_refs[f"zero_bias_{size}"] = t
                _bias_cache[size] = mpk.attach_input(
                    torch_tensor=t, name=f"zero_bias_{size}")
            return _bias_cache[size]

        _moe_bias_cache = {}

        def zero_moe_bias(size):
            if size not in _moe_bias_cache:
                # Rows are LOCAL experts, not global: the bias is a weight
                # tensor, indexed by the same local_eid as gate_up/down, and
                # the registrar asserts dim[0] == the weight tensor's dim[0].
                # Off EP the two counts coincide.
                t = torch.zeros(ep_local + num_shared, size,
                                dtype=torch.bfloat16, device="cuda")
                _tensor_refs[f"zero_moe_bias_{size}"] = t
                _moe_bias_cache[size] = mpk.attach_input(
                    torch_tensor=t, name=f"zero_moe_bias_{size}")
            return _moe_bias_cache[size]

        _layer_weight_refs = []

        def _attach_input_keep(torch_tensor, name):
            """attach_input + keep tensor alive to prevent pointer reuse."""
            _layer_weight_refs.append(torch_tensor)
            return mpk.attach_input(torch_tensor=torch_tensor, name=name)

        def _release(*params):
            """Drop a Parameter's storage once an absorbed/stacked copy of it
            has been attached.

            Only ever called on weights whose megakernel copy is a genuinely
            new tensor (absorption products, gate/up interleaves, expert
            stacks) -- never on ones where the ``.contiguous()`` in the attach
            may have returned the same storage. The Torch reference path is
            not run under --use-mirage, so these are dead after this point,
            and for GLM-5 the expert stacks alone would otherwise double MoE
            residency."""
            for p in params:
                p.data = torch.empty(0, dtype=p.data.dtype,
                                     device=p.data.device)


        # ── Graph ────────────────────────────────────────────────────────────
        w_embed = _attach_input_keep(model.model.embed_tokens.weight.data,
                                     "embed_tokens")
        mpk.embed_layer(
            input=x, weight=w_embed, output=y,
            grid_dim=(1, 1, 1), block_dim=(256, 1, 1), input_source=1,
        )
        x = y

        # The last fused layer's argument bundle, reused verbatim by the EP
        # tail task below.
        last_fl_kwargs = None
        for i, layer in enumerate(model.model.main_layers):
            attn = layer.self_attn
            attn._absorb()

            w_norm = _attach_input_keep(layer.input_layernorm.weight.data,
                                        f"layer_{i}_input_layernorm")
            # q_a_proj and kv_a_proj_with_mqa read the *same* normalised
            # hidden. The task chain is linear -- every op has to consume
            # something the op before it produced (runtime.cc:572) -- so two
            # siblings hanging off one norm is not expressible. They become one
            # GEMM emitting [q_a (q_lora_pad) | latent (kv_a_out_pad)], which
            # q_b then follows and mla_kv_cache_update reads at kv_offset.
            # What kept them apart is handled by `norm_span`: q_a_layernorm
            # sums only the leading q_lora_pad columns, leaving the latent out
            # of its RMS denominator.
            qkv_a_stack = torch.cat(
                [pad_rows(attn.q_a_proj.weight.data, q_lora_pad),
                 pad_rows(attn.kv_a_proj_with_mqa.weight.data,
                          kv_a_out_pad)], dim=0).contiguous()
            if DENSE_MXFP8:
                # The packer preserves row order, so the [q_a | latent] split
                # the rest of the layer indexes by still lands where it did.
                # The padded rows are exactly zero, which quantizes to an
                # all-zero block with an E8M0 of 0 -- decoded as 1.0.
                qkv_a_stack = pack_dense_mxfp8(qkv_a_stack, QKV_MXFP8_OPW,
                                              fake_fp4=FAKE_MXFP4_ATTN)
            w_qkv_a = _attach_input_keep(qkv_a_stack, f"layer_{i}_qkv_a_proj")
            # q_a_layernorm weight, zero past q_lora: the padded q_a columns
            # are exactly zero (w_qkv_a's extra rows are zero) so they cost the
            # denominator nothing, and the zeros over the latent columns keep
            # the normalised row's tail at zero -- which is what lets q_b read
            # the whole qkv_a_pad-wide row as its reduction.
            w_q_a_norm = _attach_input_keep(
                torch.nn.functional.pad(attn.q_a_layernorm.weight.data,
                                        (0, qkv_a_pad - q_lora)).contiguous(),
                f"layer_{i}_q_a_layernorm")
            w_kv_a_norm = _attach_input_keep(
                attn.kv_a_layernorm.weight.data.contiguous(),
                f"layer_{i}_kv_a_layernorm")

            # The whole attention half in one gang task. Needs both GEMMs
            # on the MXFP8 path -- the fused kernel only wraps those bodies --
            # and a real merge phase, which one KV chunk does not have.
            fuse_attn = (FUSE_ATTN and DENSE_MXFP8 and QB_MXFP8
                         and num_kv_chunks > 1)
            # The bs > 1 break lived here and it is fixed, but not by forcing
            # this flag -- read the history before re-adding that force.
            #
            # Measured, dense prologue layer 0, prompt "The capital of France
            # is", the o_proj's view of attn_out:
            #
            #             one live row   two live rows
            #   row 0     -2.458224      -0.984846
            #   row 1     --              0.000000
            #
            # while that layer's INPUT (x, and q_b's 2048-wide norm input) was
            # bit-identical between the two arms. GLM_FUSE_ATTN defaults to 0,
            # so the three dense layers are the only ones that take the
            # unfused chain -- every MoE layer is whole-layer fused, which sets
            # fuse_attn itself -- and that chain was the last uninstrumented
            # window in the bs=2 hunt.
            #
            # Forcing fuse_attn = True here does produce correct text, and it
            # was the first fix. It is also expensive: GLM_FUSE_ATTN=1 measured
            # 11.465 ms against 10.469 for the unfused prologue, n=3 each,
            # serialized (+0.996 ms, ~0.33 ms for each of the three dense
            # layers, against a 0.26 ms noise floor). The fused attention task
            # holds every dispatched worker at wide in-kernel barriers around
            # phases a dense layer barely uses.
            #
            # The cheaper fix is that the standalone chain was never actually
            # single-row in the kernels -- gang_mla_decode_kernel has carried
            # token_idx and tok_back since the fused path got them; it was
            # gang_mla_decode_layer's total_work_items that had no batch_size
            # factor and its registration that instantiated BATCH_SIZE = 1, so
            # the dispatch was one token wide and row 1 was never written. The
            # split-KV merge needed nothing: merge_splitkv_ck_fmha already
            # loops tok over num_tokens * heads off qo_indptr, which is why the
            # fused caller's merge_total carries no batch factor either.
            if bs > 1:
                assert num_kv_chunks > 1 or fuse_attn, (
                    "batch > 1 with a single KV chunk skips the split-KV "
                    "merge, and the decode's direct-to-attn_out path is not "
                    "row-indexed")
            # Whole-layer fusion is the union of both halves' preconditions,
            # and it is emitted at the MoE call site further down because the
            # expert weights it needs are not built until then. Here it only
            # has to switch every attention stage off.
            fuse_full_layer = (FUSE_FULL_LAYER and layer.is_moe
                               and DENSE_MXFP8 and QB_MXFP8
                               and num_kv_chunks > 1
                               and use_mxfp8_oproj and MOE_MXFP8
                               and FUSE_MOE_SWIGLU and FUSE_MOE_MULSUMADD)
            if fuse_full_layer:
                fuse_attn = True

            # ── q_b, absorbed or not ─────────────────────────────────────
            # Only the whole-layer fused task has a Phase 3b to apply W_UK in,
            # so the dense prologue layers keep the absorbed weight. q_b only
            # wants the q_a half of the fused [q_a | latent] row either way, so
            # it reduces over q_lora_pad rather than the full qkv_a_pad: both
            # are legal reductions (256-aligned) and the q_a half leads the
            # row, but stopping at q_lora_pad halves the widest weight in the
            # model -- [num_heads_pad * qk_dim, qkv_a_pad] is 75.5 MB/layer on
            # Flash, and every column past q_lora_pad is zero.
            unabsorb_k_this = UNABSORB_K and fuse_full_layer
            if unabsorb_k_this:
                # The checkpoint's own q_b, [H * (nope + rope), q_lora]; no
                # row padding, because UNABSORB_K already requires the head
                # count to be unpadded.
                q_b_w = attn.q_b_proj.weight.data.to(torch.bfloat16)
                assert q_b_w.shape[0] == qb_nope_width, (
                    q_b_w.shape, qb_nope_width)
                q_b_w = pad_cols(q_b_w, q_lora_pad)
                # W_UK is [H, qk_nope, kv_lora]: for head h,
                #   q[h][c] = sum_j q_nope[h][j] * W_UK[h][j][c]
                # so the GEMV's [rows, reduction] is that transposed, with the
                # head axis flattened into the rows.
                w_uk_rows = attn._w_uk.transpose(1, 2).reshape(
                    num_heads * kv_lora, qk_nope).to(torch.bfloat16
                                                     ).contiguous()
                if qb_tp:
                    # Keep only this rank's heads. Both weights are row-major
                    # in the head axis -- q_b's rows are head-major over the
                    # [nope | rope] span, W_UK's over the latent columns -- so
                    # the slice is one contiguous block in each, and the tile
                    # counts drop by world_size with nothing else changed.
                    # That ratio is what the kernel reads the shard off.
                    q_b_w = q_b_w[rank * qb_tp_heads * qb_nope_span:
                                  (rank + 1) * qb_tp_heads * qb_nope_span,
                                  :].contiguous()
                    w_uk_rows = w_uk_rows[rank * qb_tp_heads * kv_lora:
                                          (rank + 1) * qb_tp_heads * kv_lora,
                                          :].contiguous()
                w_wuk = _attach_input_keep(
                    pack_dense_mxfp8(w_uk_rows, WUK_GEMV_ROWS,
                                     fake_fp4=FAKE_MXFP4_ATTN),
                    f"layer_{i}_w_uk")
            else:
                q_b_w = absorb_q_b(
                    attn.q_b_proj.weight.data, attn._w_uk,
                    num_heads, qk_nope, qk_rope, q_lora).to(torch.bfloat16)
                q_b_w = pad_cols(q_b_w, q_lora_pad)
                q_b_w = pad_rows(q_b_w, qb_out_width)
                w_wuk = None
            # Absorbed q_b's 576-wide head span is only divisible by the rope
            # width, so only the un-absorbed layers get the wider tile. See
            # QB_GEMM_OPW.
            qb_opw_this = QB_GEMM_OPW if unabsorb_k_this else qk_rope
            if QB_MXFP8:
                # The weight is packed per workgroup of qb_opw_this columns, so
                # the pack width and the kernel's OUTPUT_PER_WG are the same
                # number and have to be chosen together.
                q_b_w = pack_dense_mxfp8(q_b_w, qb_opw_this,
                                         fake_fp4=FAKE_MXFP4_ATTN)
            w_q_b = _attach_input_keep(q_b_w, f"layer_{i}_q_b_absorbed")

            # ── o_proj, absorbed or not ──────────────────────────────────
            # Only the whole-layer fused task has a Phase 8b to apply W_UV in,
            # so the dense prologue layers keep the absorbed weight and the
            # MoE layers get the narrow one plus the W_UV stack. Both are
            # legal for the same checkpoint; see the byte argument above.
            unabsorb_this = UNABSORB_V and fuse_full_layer
            if unabsorb_this:
                layer_o_proj_red = o_proj_red_unabsorbed
                o_w = pad_cols(attn.o_proj.weight.data.to(torch.bfloat16),
                               layer_o_proj_red)
                # W_UV is [H, v_head, kv_lora] -- exactly the GEMV's
                # [rows, reduction] once the head axis is flattened into the
                # rows, since row h * v_head + j reduces over that head's
                # kv_lora slice of attn_out.
                w_uv_rows = attn._w_uv.reshape(
                    num_heads * v_head, kv_lora).to(torch.bfloat16).contiguous()
                w_wuv = _attach_input_keep(
                    pack_dense_mxfp8(w_uv_rows, WUV_GEMV_ROWS,
                                     fake_fp4=FAKE_MXFP4_ATTN),
                    f"layer_{i}_w_uv")
            else:
                layer_o_proj_red = o_proj_red
                w_wuv = None
                o_w = absorb_o_proj(
                    attn.o_proj.weight.data, attn._w_uv,
                    num_heads, v_head, kv_lora).to(torch.bfloat16)
                o_w = pad_cols(o_w, layer_o_proj_red)
            if oproj_tp and fuse_full_layer:
                # Keep only the output rows this rank owns. Everything
                # downstream follows from dim 0: oproj_tiles_per_xcd is
                # oproj_weight.dim(0) // 8, so the tile count drops from 48 to
                # 6 per XCD with no other change, and that ratio is what the
                # kernel reads the shard off. Only the fused whole-layer task
                # -- the dense prologue keeps the replicated weight.
                o_w = o_w[rank * oproj_tp_cols:(rank + 1) * oproj_tp_cols, :]
                o_w = o_w.contiguous()
            if use_mxfp8_oproj:
                # One workgroup per GEMV tile, so the packing's workgroup axis
                # *is* the tile axis: 2048 / 16 = 128 workgroups, 16 per XCD,
                # exactly the tile count the bf16 GEMV had. At MXFP4 the
                # workgroup count is identical and only the row width halves,
                # so the tile geometry -- and every count derived from
                # dim(0) -- is untouched.
                # fuse_full_layer, not FUSE_FULL_LAYER: the flag is global but
                # the fused task is per layer, and GLM-5's first three layers
                # are dense. Those keep the standalone MXFP8 GEMV, which
                # templates on the fp8 body and would mis-address a
                # nibble-packed row -- it asserts on the row width first, which
                # is how this was caught. Three of 78 layers, so leaving them
                # MXFP8 costs nothing measurable.
                o_w = pack_dense_mxfp8(o_w, oproj_tile_n,
                                       fake_fp4=FAKE_MXFP4_ATTN,
                                       fp4=OPROJ_MXFP4 and fuse_full_layer)
            w_o = _attach_input_keep(o_w, f"layer_{i}_o_absorbed")

            attn._w_uk = None
            attn._w_uv = None
            _release(attn.q_b_proj.weight, attn.kv_b_proj.weight,
                     attn.o_proj.weight)

            # latent_cache[i] is [pages, page_size, 576]; the kernels want a
            # 4-D [pages, page_size, 1, 576]. unsqueeze is a contiguous view.
            kv_cache = _attach_input_keep(
                model.model.latent_cache[i].unsqueeze(2),
                f"layer_{i}_latent_cache")

            # The residual fold is independent of whether the attention half is
            # one task or four: it only needs the layer's *first* task to be an
            # MXFP8 rmsnorm+linear, which both paths have, and the MoE W2
            # epilogue to be accumulating into moe_ws_f32 in the first place.
            fold_resadd = FUSE_MOE_MULSUMADD and (fuse_attn or DENSE_MXFP8)
            if fuse_attn and not fuse_full_layer:
                mpk.gang_mla_attn_fused_layer(
                    x=x,
                    pre_norm_weight=w_norm,
                    pre_norm_scratch=rmsnorm_out,
                    qkv_mxfp8_weight=w_qkv_a,
                    qkv_bias=zero_bias(qkv_a_pad),
                    q_a_norm_weight=w_q_a_norm,
                    q_a_norm_scratch=q_a_norm_out,
                    qb_mxfp8_weight=w_q_b,
                    qb_bias=zero_bias(qb_out_width),
                    kv_norm_weight=w_kv_a_norm,
                    cos_pos_embed=cos_pos_embed,
                    sin_pos_embed=sin_pos_embed,
                    kv_cache=kv_cache,
                    attn_counters=attn_fused_counter,
                    moe_workspace_f32=moe_ws_f32,
                    qkv_a_out=qkv_a_out,
                    q_workspace=mla_q_ws,
                    lse=mla_lse,
                    o_acc=mla_o_acc,
                    attn_out=attn_out,
                    x_out=layer_out,
                    qkv_output_per_wg=QKV_MXFP8_OPW,
                    qkv_actual_hidden_dim=hidden_size,
                    qb_output_per_wg=qb_opw_this,
                    qb_reduction_size=q_lora_pad,
                    qb_actual_hidden_dim=q_lora,
                    kv_offset=q_lora_pad,
                    mla_params=(num_heads_pad, kv_lora, qk_rope, qk_head_dim,
                                num_kv_chunks),
                    q_workspace_slots=qb_head_slots,
                    merge_dim_splits=MLA_MERGE_DIM_SPLITS,
                    merge_write_through=MLA_MERGE_WRITE_THROUGH,
                    block_dim=(256, 1, 1),
                )
            # 1. input_layernorm + [q_a_proj | kv_a_proj_with_mqa]
            if fuse_attn:
                pass  # Phase 1 of the fused task above.
            elif DENSE_MXFP8:
                mpk.gang_rmsnorm_linear_mxfp8_bias_layer(
                    # Under the fold `x` is the unresolved residual: the
                    # prologue adds the previous layer's MoE accumulator to it,
                    # normalizes the sum, and publishes the resolved row to
                    # layer_out for this layer's o_proj to add back.
                    norm_input=x,
                    norm_weight=w_norm,
                    norm_output=rmsnorm_out,
                    mxfp8_weight=w_qkv_a,
                    bias=zero_bias(qkv_a_pad),
                    output=qkv_a_out,
                    actual_hidden_dim=hidden_size,
                    output_per_wg=QKV_MXFP8_OPW,
                    output_stride=qkv_a_pad,
                    resadd_workspace_f32=moe_ws_f32 if fold_resadd else None,
                    resadd_x_out=layer_out if fold_resadd else None,
                    block_dim=(256, 1, 1),
                )
            else:
                mpk.gang_rmsnorm_linear_bias_layer(
                    norm_input=x,
                    norm_weight=w_norm,
                    norm_output=rmsnorm_out,
                    linear_weight=w_qkv_a,
                    bias=zero_bias(qkv_a_pad),
                    output=qkv_a_out,
                    actual_hidden_dim=hidden_size,
                    tile_n=GANG_TILE_N,
                    output_stride=qkv_a_pad,
                    wgm=GANG_WGM,
                    block_dim=(256, 1, 1),
                )
            # 2. q_a_layernorm (leading q_lora_pad columns only) + absorbed
            #    q_b_proj, with the KV cache update fused into its epilogue:
            #    the roped Q lands in mla_q_ws directly, and the latent row
            #    (kv_a_layernorm + RoPE + paged append) rides on one worker.
            if fuse_attn:
                pass  # Phase 3 of the fused task above.
            elif QB_MXFP8:
                mpk.gang_rmsnorm_linear_mxfp8_bias_mla_kvupd_layer(
                    norm_input=qkv_a_out,
                    norm_weight=w_q_a_norm,
                    norm_output=q_a_norm_out,
                    mxfp8_weight=w_q_b,
                    bias=zero_bias(qb_out_width),
                    kv_norm=w_kv_a_norm,
                    cos_pos_embed=cos_pos_embed,
                    sin_pos_embed=sin_pos_embed,
                    kv_cache=kv_cache,
                    q_workspace=mla_q_ws,
                    actual_hidden_dim=q_lora,
                    output_per_wg=qk_rope,
                    output_stride=qb_out_width,
                    reduction_size=q_lora_pad,
                    kv_offset=q_lora_pad,
                    block_dim=(256, 1, 1),
                )
            else:
                mpk.gang_rmsnorm_linear_bias_mla_kvupd_layer(
                    norm_input=qkv_a_out,
                    norm_weight=w_q_a_norm,
                    norm_output=q_a_norm_out,
                    linear_weight=w_q_b,
                    bias=zero_bias(qb_out_width),
                    kv_norm=w_kv_a_norm,
                    cos_pos_embed=cos_pos_embed,
                    sin_pos_embed=sin_pos_embed,
                    kv_cache=kv_cache,
                    q_workspace=mla_q_ws,
                    actual_hidden_dim=q_lora,
                    norm_span=q_lora_pad,
                    reduction_size=q_lora_pad,
                    kv_offset=q_lora_pad,
                    tile_n=GANG_TILE_N,
                    output_stride=qb_out_width,
                    wgm=GANG_WGM,
                    block_dim=(256, 1, 1),
                )
            # 4. absorbed MLA decode, split over (q_group, kv_chunk),
            #    then the split-KV merge. Phases 5 and 7 of the fused task.
            if not fuse_attn:
                mpk.gang_mla_decode_layer(
                    q_workspace=mla_q_ws,
                    kv_cache=kv_cache,
                    lse=mla_lse,
                    output=(mla_o_acc if num_kv_chunks > 1 else attn_out),
                    mla_params=(num_heads_pad, kv_lora, qk_rope, qk_head_dim,
                                num_kv_chunks),
                    q_workspace_slots=qb_head_slots,
                    block_dim=(256, 1, 1),
                )
                if num_kv_chunks > 1:
                    # merge_splitkv_ck_fmha treats the q groups as kv heads:
                    # merge_task_offset = bid.y (runtime.cc), so grid.y indexes
                    # the group and each task merges 16 q heads' chunks.
                    mpk.paged_attention_ck_fmha_merge_layer(
                        lse=mla_lse,
                        output_tmp=mla_o_acc,
                        output=attn_out,
                        attention_params=(num_heads_pad, kv_lora,
                                          num_kv_chunks, num_q_groups),
                        grid_dim=(args.max_num_batched_requests,
                                  num_q_groups * MLA_MERGE_DIM_SPLITS, 1),
                        block_dim=(256, 1, 1),
                        dim_splits=MLA_MERGE_DIM_SPLITS,
                        write_through=MLA_MERGE_WRITE_THROUGH,
                    )
            # 5. absorbed o_proj + residual
            #
            # On the fused path the residual stream for this layer does not
            # exist until the attention task's Phase 1 resolves
            # `moe_ws_f32 + x` into layer_out, so that -- not the previous
            # layer's `x` -- is what o_proj adds back. Off it, the standalone
            # moe_residual_add_f32 at the end of each layer still produces it
            # and `x` is already the resolved row.
            oproj_resid = layer_out if fold_resadd else x
            # The fused task calls the MXFP8 MoE kernels with both epilogues
            # on -- SwiGLU folded into W13, mul-sum-add folded into W2 -- so
            # the standalone silu and mul_sum_add stages have no place to run
            # inside it.
            fuse_oproj_router = (FUSE_OPROJ_ROUTER and layer.is_moe
                                 and use_mxfp8_oproj and MOE_MXFP8
                                 and FUSE_MOE_SWIGLU and FUSE_MOE_MULSUMADD)
            if fuse_full_layer:
                fuse_oproj_router = True
            if fuse_oproj_router:
                # Runs as Phase 1 of the fused router task below.
                pass
            elif use_splitk_oproj:
                mpk.gang_splitk_linear_with_residual_layer(
                    input=attn_out,
                    weight=w_o,
                    residual=oproj_resid,
                    workspace=splitk_ws,
                    output=attn_proj_out,
                    tile_n=GANG_TILE_N,
                    output_stride=hidden_size,
                    k_splits=GANG_K_SPLITS,
                    reduction_size=layer_o_proj_red,
                    block_dim=(256, 1, 1),
                )
            elif use_mxfp8_oproj:
                mpk.gang_gemv_mxfp8_with_residual_layer(
                    input=attn_out,
                    mxfp8_weight=w_o,
                    residual=oproj_resid,
                    output=attn_proj_out,
                    rows_per_wg=oproj_tile_n,
                    output_stride=hidden_size,
                    reduction_size=layer_o_proj_red,
                    wgm=GANG_WGM,
                    block_dim=(256, 1, 1),
                )
            else:
                mpk.gang_linear_with_residual_layer(
                    input=attn_out,
                    weight=w_o,
                    residual=oproj_resid,
                    output=attn_proj_out,
                    tile_n=oproj_tile_n,
                    output_stride=hidden_size,
                    wgm=GANG_WGM,
                    reduction_size=layer_o_proj_red,
                    gemv=use_gemv_oproj,
                    block_dim=(256, 1, 1),
                )

            w_norm_moe = _attach_input_keep(
                layer.post_attention_layernorm.weight.data,
                f"layer_{i}_post_attention_layernorm")

            if not layer.is_moe:
                # 6a. Dense layer (the first `first_k_dense_replace` of them).
                w_dense_gu = _attach_input_keep(
                    interleave_gate_up(layer.mlp.gate_proj.weight.data,
                                       layer.mlp.up_proj.weight.data, 8),
                    f"layer_{i}_dense_gate_up")
                w_dense_down = _attach_input_keep(
                    layer.mlp.down_proj.weight.data.contiguous(),
                    f"layer_{i}_dense_down")
                _release(layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight)
                mpk.gang_rmsnorm_linear_bias_layer(
                    norm_input=attn_proj_out,
                    norm_weight=w_norm_moe,
                    norm_output=rmsnorm_out_moe,
                    linear_weight=w_dense_gu,
                    bias=zero_bias(2 * dense_inter),
                    output=dense_mid,
                    actual_hidden_dim=hidden_size,
                    tile_n=GANG_TILE_N,
                    output_stride=2 * dense_inter,
                    wgm=GANG_WGM,
                    block_dim=(256, 1, 1),
                )
                mpk.silu_mul_layer(
                    input=dense_mid, output=dense_act,
                    grid_dim=(8, 1, 1), block_dim=(256, 1, 1),
                )
                dense_out = dense_resid if fold_resadd else layer_out
                mpk.gang_linear_with_residual_layer(
                    input=dense_act,
                    weight=w_dense_down,
                    residual=attn_proj_out,
                    output=dense_out,
                    tile_n=GANG_TILE_N,
                    output_stride=hidden_size,
                    wgm=GANG_WGM,
                    block_dim=(256, 1, 1),
                )
                x = dense_out
                continue

            # 6b. MoE layer. The shared expert is stacked as expert
            # `num_experts` and routed into slot `topk` with weight 1, so
            # moe_mul_sum_add computes
            #   attn_proj_out + shared(norm) + sum_k w_k * expert_k(norm)
            # in one pass, with no second gate/up/down triple and no extra add.
            shared = layer.mlp.shared_experts
            w_router = _attach_input_keep(
                layer.mlp.gate.weight.data.contiguous(),
                f"layer_{i}_router")
            w_router_bias = _attach_input_keep(
                layer.mlp.gate.e_score_correction_bias.data.to(
                    torch.bfloat16).contiguous(),
                f"layer_{i}_router_bias")
            # The fold's copy of the same weight: transposed, and sliced to
            # the hidden columns this rank's o_proj shard owns. Transposed
            # because an o_proj tile holds four hidden columns and contributes
            # to all num_experts logits, so the row-major [E, H] form would
            # have it read num_experts lines at a 2*hidden-byte stride to use
            # eight bytes of each; [H, E] makes a column's contribution one
            # contiguous 2*num_experts-byte line.
            #
            # 393 KB per rank per layer against the 3.1 MB the un-folded
            # router reads -- every one of a rank's 128 router workers reads
            # its own 2-expert pair of full 12 KB rows today, and under the
            # fold nobody reads the gate weight after the barrier at all.
            w_router_t = None
            if ROUTER_FOLD:
                _rt_cols = hidden_size // world_size
                w_router_t = _attach_input_keep(
                    layer.mlp.gate.weight.data.t()[
                        rank * _rt_cols:(rank + 1) * _rt_cols, :
                    ].contiguous(),
                    f"layer_{i}_router_wt")

            # Expert stacks, shared expert last. With the SwiGLU fused into
            # W13's epilogue the gate and up rows are interleaved pairwise so
            # the pair meets in one thread's accumulators; unfused, the plain
            # [gate | up] concat is what moe_silu_mul expects.
            # Under EP this rank stores only its id slice of the routed
            # experts, packed down to [0, ep_local), then the shared expert
            # at ep_local -- which is exactly the local_eid the MoE tile
            # helper computes. Every rank stores the shared replica; only
            # ep_fold_rank ever reads it.
            if layer.mlp.mxfp4_experts:
                # The routed experts arrived from disk already in MXFP4, so the
                # only work left is the row permutation and the per-workgroup
                # pack -- both of which are pure reshuffles of the quantised
                # rows. interleave_gate_up is width-agnostic (it only ever
                # reshapes dim 0), so the same call serves the K/2-wide blocks
                # and the K/32-wide scales, and it commutes with quantisation
                # exactly because MXFP4 blocks never cross a row.
                #
                # The shared expert is not pre-quantised: it is 1/256th of the
                # bytes, the streamer leaves it bf16, and quantising it here
                # keeps it on the identical code path as GLM-4.7-Flash.
                assert MOE_MXFP8 and MOE_MXFP4, (
                    "an MXFP4 checkpoint needs the MXFP4 expert kernel "
                    "(GLM_MOE_MXFP4=1); there is no bf16 path for it")

                def _pair(d, key):
                    b, s = d[key]
                    return b, s

                def _pack_one(gb, gs, ub, us, db, ds):
                    if FUSE_MOE_SWIGLU:
                        gu_b = interleave_gate_up(gb, ub, moe_inter)
                        gu_s = interleave_gate_up(gs, us, moe_inter)
                    else:
                        gu_b = torch.cat([gb, ub], dim=0)
                        gu_s = torch.cat([gs, us], dim=0)
                    return (pack_mxfp8_workgroup(gu_b, gu_s, MOE_W13_OPW),
                            pack_mxfp8_workgroup(db, ds, MOE_W2_OPW))

                gu_parts, down_parts = [], []
                for d in layer.mlp.expert_mxfp4:
                    gb, gs = _pair(d, "gate_proj")
                    ub, us = _pair(d, "up_proj")
                    db, ds = _pair(d, "down_proj")
                    gu, dn = _pack_one(gb, gs, ub, us, db, ds)
                    gu_parts.append(gu)
                    down_parts.append(dn)
                sg, sgs = quantize_mxfp4(shared.gate_proj.weight.data)
                su, sus = quantize_mxfp4(shared.up_proj.weight.data)
                sd, sds = quantize_mxfp4(shared.down_proj.weight.data)
                gu, dn = _pack_one(sg, sgs, su, sus, sd, sds)
                gu_parts.append(gu)
                down_parts.append(dn)
                gu_stack = torch.stack(gu_parts).contiguous()
                down_stack = torch.stack(down_parts).contiguous()
                layer.mlp.expert_mxfp4 = None   # drop the unpacked references
                _release(shared.gate_proj.weight, shared.up_proj.weight,
                         shared.down_proj.weight)
            else:
                routed = list(layer.mlp.experts)
                experts = (routed[ep_slice_base:ep_slice_base + ep_local]
                           + [shared])
                gu_stack = torch.stack([
                    interleave_gate_up(e.gate_proj.weight.data,
                                       e.up_proj.weight.data, moe_inter)
                    if FUSE_MOE_SWIGLU else
                    torch.cat([e.gate_proj.weight.data,
                               e.up_proj.weight.data], dim=0)
                    for e in experts
                ]).contiguous()
                down_stack = torch.stack([e.down_proj.weight.data
                                          for e in experts]).contiguous()
                if MOE_MXFP8:
                    # The packer preserves row order, so the pairwise gate/up
                    # interleave above carries through untouched.
                    gu_stack = pack_moe_mxfp8(gu_stack, MOE_W13_OPW)
                    down_stack = pack_moe_mxfp8(down_stack, MOE_W2_OPW)
                # Release the FULL list, not the owned slice: the non-owned
                # experts were loaded and are now dead weight.
                for e in routed + [shared]:
                    _release(e.gate_proj.weight, e.up_proj.weight,
                             e.down_proj.weight)
            w_moe_gu = _attach_input_keep(gu_stack, f"layer_{i}_moe_gate_up")
            w_moe_down = _attach_input_keep(down_stack, f"layer_{i}_moe_down")

            # post_attention_layernorm + router GEMV + routing, fused.
            #
            # Kept apart these were three consecutive single-task ops -- the
            # rmsnorm ran on one worker per token, the router GEMM was too
            # narrow for the gang tile (num_experts/8 falls under the 64-wide
            # N tile), and the routing task is inherently serial. Three
            # dispatch barriers per layer with 239 of 240 workers idle.
            #
            # The Croc-style fusion instead gives one worker to each expert:
            # every worker recomputes the (identical) RMSNorm, does its own
            # row of the gate GEMV, and the last one across all 8 XCDs runs
            # the TopK tail behind an atomic-counter barrier. The redundant
            # norm is far cheaper than the barriers it replaces.
            if fuse_full_layer:
                # The whole layer. Everything from the input RMSNorm to the
                # MoE W2 accumulation, in one dispatch.
                #
                # `oproj_resid` and `x_out` are the same tensor by
                # construction: Phase 1 resolves the residual stream into
                # layer_out and Phase 9 adds it back. That is the same
                # producer/consumer pair the two-task form had, only now
                # inside one task -- and the resolve already writes through
                # (st_wt_u64 in _rnlm8_resadd_norm_rcp), so it survives the
                # Phase 8 barrier without an event.
                # Built as a dict rather than passed inline so the EP tail
                # task below can rebind the last layer's weights verbatim --
                # it is the same task type on the same layer, differing only
                # in its gather buffer and the ep_tail_only variant flag.
                fl_kwargs = dict(
                    x=x,
                    pre_norm_weight=w_norm,
                    pre_norm_scratch=rmsnorm_out,
                    qkv_mxfp8_weight=w_qkv_a,
                    qkv_bias=zero_bias(qkv_a_pad),
                    q_a_norm_weight=w_q_a_norm,
                    q_a_norm_scratch=q_a_norm_out,
                    qb_mxfp8_weight=w_q_b,
                    qb_bias=zero_bias(
                        qb_nope_width if unabsorb_k_this else qb_out_width),
                    kv_norm_weight=w_kv_a_norm,
                    cos_pos_embed=cos_pos_embed,
                    sin_pos_embed=sin_pos_embed,
                    kv_cache=kv_cache,
                    moe_workspace_f32=moe_ws_f32,
                    counters=full_layer_counter,
                    oproj_mxfp8_weight=w_o,
                    residual=oproj_resid,
                    post_norm_weight=w_norm_moe,
                    post_norm_output=rmsnorm_out_moe,
                    router_weight=w_router,
                    router_bias=w_router_bias,
                    logits_scratch=moe_gate_out,
                    moe_gate_up_weight=w_moe_gu,
                    moe_down_weight=w_moe_down,
                    moe_w13_bias=zero_moe_bias(2 * moe_inter),
                    moe_w2_bias=zero_moe_bias(hidden_size),
                    moe_swiglu_out=moe_act,
                    qkv_a_out=qkv_a_out,
                    q_workspace=mla_q_ws,
                    lse=mla_lse,
                    o_acc=mla_o_acc,
                    attn_out=attn_out,
                    x_out=layer_out,
                    hidden=attn_proj_out,
                    topk_weight=moe_topk_weight,
                    routing_indices=moe_routing_indices,
                    active_expert_ids=moe_mask,
                    qkv_output_per_wg=QKV_MXFP8_OPW,
                    qkv_actual_hidden_dim=hidden_size,
                    qb_output_per_wg=qb_opw_this,
                    qb_reduction_size=q_lora_pad,
                    qb_actual_hidden_dim=q_lora,
                    kv_offset=q_lora_pad,
                    mla_params=(num_heads_pad, kv_lora, qk_rope, qk_head_dim,
                                num_kv_chunks),
                    # Un-absorbed, q_b's output row is q_nope and Phase 3b
                    # writes the whole query row, so there is no slot subset
                    # to take -- the "slots" here are q_nope's, all of them.
                    q_workspace_slots=(num_heads if unabsorb_k_this
                                       else qb_head_slots),
                    merge_dim_splits=MLA_MERGE_DIM_SPLITS,
                    oproj_rows_per_wg=oproj_tile_n,
                    oproj_reduction_size=layer_o_proj_red,
                    wuv_mxfp8_weight=w_wuv,
                    v_out=mla_v_out if unabsorb_this else None,
                    wuv_rows_per_wg=WUV_GEMV_ROWS if unabsorb_this else 0,
                    wuv_v_head_dim=v_head if unabsorb_this else 0,
                    wuk_mxfp8_weight=w_wuk,
                    q_nope=mla_q_nope if unabsorb_k_this else None,
                    wuk_rows_per_wg=WUK_GEMV_ROWS if unabsorb_k_this else 0,
                    qk_nope_head_dim=qk_nope if unabsorb_k_this else 0,
                    actual_hidden_dim=hidden_size,
                    num_experts_per_tok=topk,
                    routed_scaling_factor=config.routed_scaling_factor,
                    norm_topk_prob=config.norm_topk_prob,
                    moe_w13_output_per_wg=MOE_W13_OPW,
                    moe_w2_output_per_wg=MOE_W2_OPW,
                    router_experts_per_tile=ROUTER_EPT,
                    block_dim=(256, 1, 1),
                )
                if moe_ep:
                    # The fold at the head of this layer publishes the
                    # PREVIOUS layer's partial, so the buffer is keyed by the
                    # layer doing the reading -- one per fused layer.
                    fl_kwargs.update(ep_gather=ep_gather_list[i],
                                     ep_signal=ep_signal,
                                     ep_fold_rank=ep_fold_rank)
                    if ROUTER_FOLD:
                        fl_kwargs.update(router_weight_t=w_router_t,
                                         router_partials=router_partials)
                mpk.gang_mla_full_layer_fused_layer(**fl_kwargs)
                last_fl_kwargs = fl_kwargs
            elif fuse_oproj_router:
                mpk.gang_oproj_router_fused_layer(
                    input=attn_out,
                    oproj_mxfp8_weight=w_o,
                    residual=oproj_resid,
                    norm_weight=w_norm_moe,
                    norm_output=rmsnorm_out_moe,
                    router_weight=w_router,
                    router_bias=w_router_bias,
                    logits_scratch=moe_gate_out,
                    router_counter=router_topk_counter,
                    oproj_counters=oproj_router_counter,
                    moe_gate_up_weight=w_moe_gu,
                    moe_down_weight=w_moe_down,
                    moe_w13_bias=zero_moe_bias(2 * moe_inter),
                    moe_w2_bias=zero_moe_bias(hidden_size),
                    moe_swiglu_out=moe_act,
                    hidden=attn_proj_out,
                    topk_weight=moe_topk_weight,
                    routing_indices=moe_routing_indices,
                    active_expert_ids=moe_mask,
                    moe_workspace_f32=moe_ws_f32,
                    rows_per_wg=oproj_tile_n,
                    reduction_size=layer_o_proj_red,
                    actual_hidden_dim=hidden_size,
                    num_experts_per_tok=topk,
                    routed_scaling_factor=config.routed_scaling_factor,
                    norm_topk_prob=config.norm_topk_prob,
                    moe_w13_output_per_wg=MOE_W13_OPW,
                    moe_w2_output_per_wg=MOE_W2_OPW,
                    block_dim=(256, 1, 1),
                )
            else:
                mpk.gang_rmsnorm_linear_bias_topk_sigmoid_layer(
                    norm_input=attn_proj_out,
                    norm_weight=w_norm_moe,
                    norm_output=rmsnorm_out_moe,
                    linear_weight=w_router,
                    bias=w_router_bias,
                    logits_scratch=moe_gate_out,
                    gang_counter=router_topk_counter,
                    topk_weight=moe_topk_weight,
                    routing_indices=moe_routing_indices,
                    active_expert_ids=moe_mask,
                    actual_hidden_dim=hidden_size,
                    tile_n=1,
                    output_stride=num_experts,
                    num_experts_per_tok=topk,
                    routed_scaling_factor=config.routed_scaling_factor,
                    norm_topk_prob=config.norm_topk_prob,
                    block_dim=(256, 1, 1),
                )
            # W13+SwiGLU and W2+MulSumAdd ran as Phases 5 and 7 of the fused
            # task; only the residual add is left.
            if not fuse_oproj_router:
                moe_w13_layer = (mpk.gang_moe_w13_linear_mxfp8_layer
                                 if MOE_MXFP8
                                 else mpk.gang_moe_w13_linear_layer)
                moe_w13_layer(
                    input=rmsnorm_out_moe,
                    weight=w_moe_gu,
                    moe_routing_indices=moe_routing_indices,
                    moe_mask=moe_mask,
                    bias=zero_moe_bias(2 * moe_inter),
                    output=moe_act if FUSE_MOE_SWIGLU else moe_mid,
                    fuse_swiglu=FUSE_MOE_SWIGLU,
                    block_dim=(256, 1, 1),
                    **({"output_per_wg": MOE_W13_OPW} if MOE_MXFP8 else {}),
                )
                if not FUSE_MOE_SWIGLU:
                    mpk.moe_silu_mul_layer(
                        input=moe_mid,
                        output=moe_act,
                        grid_dim=(args.max_num_batched_tokens, topk_total, 1),
                        block_dim=(256, 1, 1),
                    )
                moe_w2_layer = (mpk.gang_moe_w2_linear_mxfp8_layer if MOE_MXFP8
                                else mpk.gang_moe_w2_linear_layer)
                moe_w2_layer(
                    input=moe_act,
                    weight=w_moe_down,
                    moe_routing_indices=moe_routing_indices,
                    moe_mask=moe_mask,
                    bias=zero_moe_bias(hidden_size),
                    output=moe_ws_f32 if FUSE_MOE_MULSUMADD else moe_out,
                    routing_weight=(moe_topk_weight if FUSE_MOE_MULSUMADD
                                    else None),
                    block_dim=(256, 1, 1),
                    **({"output_per_wg": MOE_W2_OPW} if MOE_MXFP8 else {}),
                )
            # The residual add is Phase 1 of the *next* layer's attention
            # task, so it is emitted here only for the layer that has no next
            # one -- which also re-zeroes moe_ws_f32 for the next token, the
            # zero that the first layer's resolve relies on.
            # Under EP nobody resolves here at all: the last layer's MoE
            # partial is folded and published by the tail task below, which
            # re-zeroes moe_ws_f32 as it reads it, and the LM head's prologue
            # does the cross-rank sum. The standalone moe_residual_add_f32
            # would double-count this rank's contribution.
            resolve_here = not (fold_resadd
                                and (moe_ep or i < num_layers - 1))
            if not resolve_here:
                x = attn_proj_out
            elif FUSE_MOE_MULSUMADD:
                mpk.moe_residual_add_f32_layer(
                    workspace_f32=moe_ws_f32,
                    residual=attn_proj_out,
                    output=layer_out,
                    grid_dim=(1, 1, 1),
                    block_dim=(256, 1, 1),
                )
                x = layer_out
            else:
                mpk.moe_mul_sum_add_layer(
                    input=moe_out,
                    weight=moe_topk_weight,
                    residual=attn_proj_out,
                    output=layer_out,
                    grid_dim=(args.max_num_batched_tokens,
                              hidden_size // 256, 1),
                    block_dim=(256, 1, 1),
                )
                x = layer_out

        # ── The EP tail ──────────────────────────────────────────────────
        # GLM folds at the HEAD of a layer, which buys it one rendezvous per
        # layer where gpt-oss pays two -- and costs it this: the LAST fused
        # layer's MoE output is never folded, because there is no layer L+1
        # to do it. So emit one more of the same task, in the variant that
        # runs the layer-entry barrier, the fold, the exchange and the peer
        # wait and then returns before Phase 1.
        #
        # It must be IMMEDIATELY after the last real layer with nothing in
        # between. The multi-layer scan in persistent_kernel.cuh groups
        # *consecutive* runs of this task type and swaps variant_id per layer
        # out of ml_variant_ids, so a different instantiation for the tail is
        # exactly what that table is for -- but anything wedged between the
        # two would break the run and disable replay for the whole model.
        # Joining the batch is not cosmetic either: task_layer_idx has to
        # keep counting, or the tail's signal threshold would not agree
        # across ranks.
        if moe_ep:
            assert last_fl_kwargs is not None, \
                "MOE_EP found no fused layer to tail"
            mpk.gang_mla_full_layer_fused_layer(**dict(
                last_fl_kwargs,
                # The residual stream the fold adds to moe_ws_f32. `x` is the
                # last layer's `hidden`, which is what the next layer's fold
                # would have read.
                x=x,
                ep_gather=ep_gather_list[num_layers],
                ep_tail_only=True,
            ))

        # ── Tail: final norm + LM head + argmax ──────────────────────────────
        w_final_norm = _attach_input_keep(model.model.norm.weight.data,
                                          "model_norm_weight")
        w_lm_head = _attach_input_keep(
            pack_dense_mxfp8(lm_head_weight, DENSE_MXFP8_OPW) if DENSE_MXFP8
            else lm_head_weight, "lm_head")
        assert not moe_ep or DENSE_MXFP8, \
            "MOE_EP needs the MXFP8 LM head: it is the cross-rank combine"
        if DENSE_MXFP8:
            # Under EP the LM head IS the combine. Its residual fold already
            # walks the whole row, so summing world_size gather slots rides
            # inside a pass it was making anyway -- which is what lets the
            # tail exchange have no exit barrier behind it: there is no window
            # between "combined" and "consumed" for one to protect.
            #
            # resadd_workspace_f32 is handed moe_ws_f32 but never read: the
            # fold already zeroed it, and this rank's partial is in its own
            # gather slot. It is passed for the task-graph edge -- moe_ws_f32
            # is an output of the tail task, and it is the only tensor that
            # orders the LM head after it (ep_gather is an input to both).
            ep_lm_kwargs = dict(
                norm_input=ep_gather_list[num_layers],
                resadd_workspace_f32=moe_ws_f32,
                resadd_x_out=layer_out,
                ep_peer_slots=world_size,
            ) if moe_ep else dict(norm_input=x)
            mpk.gang_rmsnorm_linear_mxfp8_bias_layer(
                norm_weight=w_final_norm,
                norm_output=rmsnorm_out,
                mxfp8_weight=w_lm_head,
                bias=zero_bias(vocab_size),
                output=argmax_in,
                actual_hidden_dim=hidden_size,
                output_per_wg=DENSE_MXFP8_OPW,
                output_stride=vocab_size,
                block_dim=(256, 1, 1),
                **ep_lm_kwargs,
            )
        else:
            mpk.gang_rmsnorm_linear_bias_layer(
                norm_input=x,
                norm_weight=w_final_norm,
                norm_output=rmsnorm_out,
                linear_weight=w_lm_head,
                bias=zero_bias(vocab_size),
                output=argmax_in,
                actual_hidden_dim=hidden_size,
                tile_n=GANG_TILE_N,
                output_stride=vocab_size,
                wgm=GANG_WGM,
                block_dim=(256, 1, 1),
            )
        mpk.argmax_partial_layer(
            input=argmax_in,
            output=(argmax_part_value, argmax_part_index),
            grid_dim=(argmax_num_tasks, 1, 1),
            block_dim=(256, 1, 1),
        )
        mpk.argmax_reduce_layer(
            input=(argmax_part_value, argmax_part_index),
            output=argmax_out,
            grid_dim=(1, 1, 1),
            block_dim=(256, 1, 1),
        )

        num_ops = len(mpk.kn_graph.cygraph.get_graph_structure())
        print(f"DEBUG: kn_graph has {num_ops} operators before "
              f"generate_task_graph")
        results = mpk.kn_graph.generate_task_graph(num_gpus=world_size,
                                                   my_gpu_id=rank)
        with open(f"task_graph_{rank}.json", "w") as f:
            f.write(results["json_file"])
        with open(f"kernel_{rank}.cu", "w") as f:
            f.write(results["cuda_code"])

        mpk.compile(output_dir=args.output_dir)

    # ── Execution ────────────────────────────────────────────────────────────
    output_len = args.max_new_tokens if args.max_new_tokens is not None else (
        tokens.size(1) - prompt_lengths[0].item())
    output_len = max(0, min(output_len,
                            tokens.size(1) - prompt_lengths[0].item()))

    mtp_stats = None
    if not args.use_mirage:
        prompt_len = prompt_lengths[0].item()
        decode_limit = prompt_len + output_len
        cur_pos = prompt_len

        if args.mtp:
            # ── MTP speculative decode (torch reference) ─────────────────
            # One draft layer predicts token p+1 from (token p, hidden p-1); the
            # main stack then scores both in a single 2-token pass. If its own
            # argmax at position p agrees with the draft, both tokens commit and
            # the iteration produced two tokens for one main-model pass. The
            # emitted text is argmax-identical to the non-MTP loop either way --
            # rejection falls back to exactly the token greedy decode would emit.
            step.fill_(prompt_len - 1)
            logits, prenorm = model.forward(
                input_ids=tokens[:, :prompt_len],
                position_embeddings=(position_embeddings[0][:, :prompt_len],
                                     position_embeddings[1][:, :prompt_len]),
                step=step, return_prenorm=True)
            pos = prompt_len
            tokens[0, pos] = logits.argmax(dim=-1)[0, -1]
            # Tokens the draft layer has not consumed yet, and the main-stack
            # hidden state each one is conditioned on (position - 1).
            new_ids, new_prev = tokens[:, pos:pos + 1], prenorm[:, -1:, :]
            n_iters = n_gen = n_accept = n_draft = 0
            hit_eos = int(tokens[0, pos]) in eos_token_ids and not args.ignore_eos

            torch.cuda.synchronize()
            starter.record()
            while (not hit_eos and pos + 1 < decode_limit
                   and pos + 2 < tokens.size(1)):
                k = new_ids.shape[1]
                step.fill_(pos)
                draft_logits = model.mtp_draft(
                    input_ids=new_ids, prev_hidden=new_prev,
                    position_embeddings=(
                        position_embeddings[0][:, pos - k + 1:pos + 1],
                        position_embeddings[1][:, pos - k + 1:pos + 1]),
                    step=step)
                draft = draft_logits.argmax(dim=-1)[0, -1]
                tokens[0, pos + 1] = draft

                step.fill_(pos + 1)
                logits, prenorm = model.forward(
                    input_ids=tokens[:, pos:pos + 2],
                    position_embeddings=(
                        position_embeddings[0][:, pos:pos + 2],
                        position_embeddings[1][:, pos:pos + 2]),
                    step=step, all_positions=True, return_prenorm=True)
                verified = logits.argmax(dim=-1)[0]        # [2]
                n_iters += 1
                n_draft += 1
                if int(verified[0]) == int(draft):
                    n_accept += 1
                    tokens[0, pos + 2] = verified[1]
                    n_acc = 2
                else:
                    # The draft was wrong; position p+1 takes the main model's own
                    # token. Its stale latent row is rewritten by the next pass
                    # before anything reads it.
                    tokens[0, pos + 1] = verified[0]
                    n_acc = 1
                # An accepted pair may overrun --max-new-tokens by one; the
                # token is computed either way, it just is not committed, so
                # the MTP and non-MTP legs emit the same count.
                n_acc = min(n_acc, decode_limit - 1 - pos)
                new_ids = tokens[:, pos + 1:pos + 1 + n_acc]
                new_prev = prenorm[:, :n_acc, :]
                for j in range(n_acc):
                    n_gen += 1
                    if (int(tokens[0, pos + 1 + j]) in eos_token_ids
                            and not args.ignore_eos):
                        hit_eos = True
                        pos = pos + 1 + j
                        break
                else:
                    pos += n_acc
            ender.record()
            torch.cuda.synchronize()
            run_time = starter.elapsed_time(ender)
            prev_pos, cur_pos = pos, pos
            mtp_stats = dict(iters=n_iters, gen=n_gen,
                             accept=(n_accept / n_draft) if n_draft else 0.0,
                             ms_per_token=run_time / max(1, n_gen),
                             ms_per_iter=run_time / max(1, n_iters))
        else:
            for cur_pos in range(prompt_len, decode_limit):
                step.fill_(cur_pos - 1)
                input_ids = tokens[:, prev_pos:cur_pos]
                cos_embeddings = position_embeddings[0][:, prev_pos:cur_pos]
                sin_embeddings = position_embeddings[1][:, prev_pos:cur_pos]
                logits = model.forward(
                    input_ids=input_ids,
                    position_embeddings=(cos_embeddings, sin_embeddings),
                    step=step,
                )
                next_token = logits.argmax(dim=-1)[0, -1]
                tokens[0, cur_pos] = next_token
                prev_pos = cur_pos
                if int(next_token) in eos_token_ids and not args.ignore_eos:
                    break
                if cur_pos == prompt_len:
                    torch.cuda.synchronize()
                    starter.record()

            ender.record()
            torch.cuda.synchronize()
            run_time = starter.elapsed_time(ender)


        end_idx = prev_pos + 1
        generated_ids = tokens[:, :end_idx]
        response = tokenizer.batch_decode(generated_ids,
                                          skip_special_tokens=True)[0]
        print(response)
        print("Prompt length {}, generate length {}, per-token latency {} ms"
              .format(prompt_len, cur_pos - prompt_len,
                      run_time / max(1, cur_pos - prompt_len)))
        if mtp_stats:
            print("[MTP] {gen} tokens in {iters} iterations, acceptance "
                  "{acc:.3f}, {mspt:.3f} ms/token, {mspi:.3f} ms/iteration"
                  .format(gen=mtp_stats["gen"], iters=mtp_stats["iters"],
                          acc=mtp_stats["accept"],
                          mspt=mtp_stats["ms_per_token"],
                          mspi=mtp_stats["ms_per_iter"]))
        # Every rank writes (to its own suffixed path) -- see the save_path
        # resolution above.
        if save_path:
            slice_end = min(end_idx, prompt_len + MAX_SAVE_TOKENS)
            out = {
                "token_ids": tokens[0, prompt_len:slice_end].tolist(),
                "text": tokenizer.decode(tokens[0, :end_idx],
                                         skip_special_tokens=True),
                "generate_length": max(0, end_idx - prompt_len),
                "mode": "torch",
                "rank": rank,
            }
            with open(save_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Saved tokens to {save_path}")
    else:
        # Capture device printf (which writes to fd 1, bypassing sys.stdout)
        # so we can parse [FWD_PASS] iter=N time_ms=X lines and split prefill
        # vs decode totals after mpk() returns.
        import sys
        import tempfile
        import re
        sys.stdout.flush()
        sys.stderr.flush()
        _fwd_pass_log = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".fwdlog", delete=False)
        _saved_stdout_fd = os.dup(1)
        os.dup2(_fwd_pass_log.fileno(), 1)

        starter.record()
        mpk()
        ender.record()
        torch.cuda.synchronize()
        run_time = starter.elapsed_time(ender)

        if profiler_tensor is not None:
            torch.save(profiler_tensor.cpu(), "profile_output.pt")

        sys.stdout.flush()
        os.dup2(_saved_stdout_fd, 1)
        os.close(_saved_stdout_fd)
        _fwd_pass_log.flush()
        _fwd_pass_log.seek(0)
        _captured = _fwd_pass_log.read()
        _fwd_pass_log.close()
        for _line in _captured.splitlines(keepends=True):
            if "[FWD_PASS]" not in _line:
                sys.stdout.write(_line)
        sys.stdout.flush()

        _fwd_times = {}
        for _m in re.finditer(r"\[FWD_PASS\] iter=(\d+) time_ms=([\d.]+)",
                              _captured):
            _fwd_times[int(_m.group(1))] = float(_m.group(2))
        _fwd_dropped = 0
        _fwd_total_avg = None
        _fwd_total_iters = 0
        _m_tot = re.search(
            r"\[FWD_PASS_TOTAL\] iters=(\d+) total_ms=[\d.]+ "
            r"avg_ms=([\d.]+) dropped=(\d+)", _captured)
        if _m_tot:
            _fwd_total_iters = int(_m_tot.group(1))
            _fwd_total_avg = float(_m_tot.group(2))
            _fwd_dropped = int(_m_tot.group(3))

        for r in range(total_num_requests):
            generated_ids = tokens[r, : step[r] + 1]
            valid_ids = generated_ids[generated_ids >= 0]
            print(tokenizer.decode(valid_ids, skip_special_tokens=True))

        if save_path:
            gen0 = tokens[0, : step[0].item() + 1]
            gen0 = gen0[gen0 >= 0]
            pl0 = prompt_lengths[0].item()
            end0 = gen0.numel()
            slice_end = min(end0, pl0 + MAX_SAVE_TOKENS)
            out = {
                "token_ids": gen0[pl0:slice_end].tolist(),
                "text": tokenizer.decode(gen0[:end0],
                                         skip_special_tokens=True),
                "generate_length": max(0, end0 - pl0),
                "mode": "mpk",
                "rank": rank,
            }
            with open(save_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Saved tokens to {save_path}")

        prompt_len = prompt_lengths[0].item()
        total_tokens = step.max().item() + 1
        generated_tokens = total_tokens - prompt_len
        prefill_iterations = math.ceil(prompt_len / args.max_num_batched_tokens)
        decode_iterations = generated_tokens
        total_iterations = prefill_iterations + decode_iterations
        avg_time_per_iter = (run_time / total_iterations
                             if total_iterations > 0 else 0)

        # Iter numbering: iter=1 is the first prepare step (no compute), and
        # FWD_PASS for iter=N reports the time between END_OF_TASK_GRAPH N-1
        # and N, so logical iter = kernel iter - 1.
        _prefill_total = _decode_total = 0.0
        _prefill_count = _decode_count = 0
        for _it, _t in _fwd_times.items():
            _logical = _it - 1
            if _logical < 1:
                continue
            if _logical <= prefill_iterations:
                _prefill_total += _t
                _prefill_count += 1
            elif _logical <= total_iterations:
                _decode_total += _t
                _decode_count += 1

        print("=" * 80)
        print("[Wall-time average]")
        print(f"  avg_per_iter (run_time / {total_iterations}): "
              f"{avg_time_per_iter:.3f} ms")
        print(f"  Combined: {total_tokens} tokens, per-token latency: "
              f"{run_time / max(1, total_tokens):.3f} ms")
        print("-" * 80)
        print("[Steady-state per-iter (device clock, prefill vs decode)]")
        if _prefill_count > 0:
            print(f"  Prefill: {prompt_len} tokens in "
                  f"{_prefill_count}/{prefill_iterations} iters ~= "
                  f"{_prefill_total:.1f}ms total "
                  f"(avg {_prefill_total / _prefill_count:.3f}ms/iter)")
        if _decode_count > 0:
            _decode_samples = [_t for _it, _t in _fwd_times.items()
                               if prefill_iterations < _it - 1 <= total_iterations]
            print(f"  Decode:  {generated_tokens} tokens in "
                  f"{_decode_count}/{decode_iterations} iters ~= "
                  f"{_decode_total:.1f}ms total "
                  f"(avg {_decode_total / _decode_count:.3f}ms/iter)")
            print(f"  Decode per-iter range: min={min(_decode_samples):.3f}ms "
                  f"max={max(_decode_samples):.3f}ms")
            # Which iterations stalled, and where in the decode run. A single
            # multi-second outlier is a liveness bug, not a latency number, and
            # its position (first decode iter vs. random) says which.
            _outliers = sorted(
                ((_t, _it - 1) for _it, _t in _fwd_times.items()
                 if prefill_iterations < _it - 1 <= total_iterations
                 and _t > 10.0 * min(_decode_samples)),
                reverse=True)
            if _outliers:
                print("  Decode outliers (>10x min): " + ", ".join(
                    f"iter {_i} = {_t:.1f}ms" for _t, _i in _outliers[:8]))
        if _fwd_dropped > 0:
            print(f"  NOTE: device per-iter ring overflowed -- {_fwd_dropped} "
                  f"of {_fwd_total_iters} samples dropped; all-iteration "
                  f"device average {_fwd_total_avg:.3f}ms/iter")
        print("=" * 80)
