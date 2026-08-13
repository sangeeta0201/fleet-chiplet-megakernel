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
MAX_SAVE_TOKENS = 100


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
    """Repack quantize_mxfp8 output into the per-workgroup layout the MXFP8
    kernels read, mirroring gpt-oss's pack_mxfp4_workgroup:

        [E, wgs, OPW*K data bytes | OPW*(K/32) scale bytes]

    Rows are K-major within the data half; scales are [row][k/32]. A 2-D
    weight is treated as E == 1 and returned without the leading axis.
    """
    squeeze = (data.dim() == 2)
    if squeeze:
        data = data.unsqueeze(0)
        scales = scales.unsqueeze(0)
    E, out_dim, K = data.shape
    assert scales.shape == (E, out_dim, K // 32)
    assert out_dim % output_per_wg == 0, \
        f"out_dim {out_dim} must be divisible by output_per_wg {output_per_wg}"

    wgs = out_dim // output_per_wg
    packed = torch.cat(
        [data.reshape(E, wgs, output_per_wg * K),
         scales.reshape(E, wgs, output_per_wg * (K // 32))],
        dim=2).contiguous()
    return packed.squeeze(0) if squeeze else packed


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
    packed = [pack_mxfp8_workgroup(*quantize_mxfp8(stacked[e]), output_per_wg)
              for e in range(stacked.shape[0])]
    return torch.stack(packed).contiguous()


def pack_dense_mxfp8(w: torch.Tensor, output_per_wg: int = 64,
                     rows_per_chunk: int = 8192) -> torch.Tensor:
    """Quantize + pack a 2-D [out, K] bf16 weight into the MXFP8 per-workgroup
    layout, in row chunks.

    Same reason as pack_moe_mxfp8: quantize_mxfp8 materialises an fp32 copy of
    whatever it is handed, which for GLM's 155136 x 2048 LM head is 1.27 GB on
    top of the bf16 original. Rows are independent and a workgroup is a
    contiguous run of `output_per_wg` rows, so chunking on a multiple of that
    and concatenating along the workgroup axis is exact.
    """
    assert w.dim() == 2, w.shape
    rows = w.shape[0]
    assert rows % output_per_wg == 0, (rows, output_per_wg)
    step = max(output_per_wg,
               (rows_per_chunk // output_per_wg) * output_per_wg)
    parts = [pack_mxfp8_workgroup(*quantize_mxfp8(w[r:r + step]), output_per_wg)
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
        "--rope-interleave", dest="rope_interleave", action="store_true",
        default=None,
        help="Force interleaved RoPE pairs (2j, 2j+1). GLM-5 sets this in "
             "config.json; GLM-4.7-Flash omits the key and inherits the "
             "family default, which is interleaved.")
    parser.add_argument("--no-rope-interleave", dest="rope_interleave",
                        action="store_false")
    args = parser.parse_args()

    # Resolve where to dump generated tokens for the correctness test.
    if args.save_tokens:
        if args.save_tokens == "auto":
            fn = "mpk_output.json" if args.use_mirage else "torch_output.json"
            save_path = os.path.join(DEFAULT_SAVE_DIR, fn)
        else:
            save_path = args.save_tokens
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
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
        os.environ["MASTER_PORT"] = "12355"
    except ImportError:
        world_size = 1
        rank = 0

    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
    global print
    if rank != 0:
        print = lambda *_, **__: None

    print("Input arguments:", args)
    print(f"world_size({world_size}) rank({rank})")
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(rank)

    with torch.device("cuda"):
        model = GlmMoeDsaForCausalLM.from_pretrained(
            args.model_path, world_size=world_size,
            max_num_pages=args.max_num_pages, page_size=args.page_size,
            num_layers=args.max_layers, random_weights=args.random_weights,
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
    num_layers = len(model.model.layers)
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
        MOE_MXFP8_OPW = 64
        if MOE_MXFP8:
            # Depth-4 MFMA pipeline: only the last of the four slots carries a
            # tail guard, so a partial final group would compute k-tiles that
            # do not exist.
            assert hidden_size % 512 == 0, hidden_size
            assert moe_inter % 512 == 0, moe_inter
            assert (2 * moe_inter) % MOE_MXFP8_OPW == 0
            assert hidden_size % MOE_MXFP8_OPW == 0

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
        assert (num_heads_pad * qk_dim) % (8 * qk_rope) == 0
        assert ((num_heads_pad * qk_dim) // (8 * qk_rope) * qk_rope) % qk_dim == 0

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
        num_q_groups = num_heads_pad // 16
        _env_chunks = os.environ.get("GLM_MLA_NUM_KV_CHUNKS")
        if _env_chunks is not None:
            num_kv_chunks = int(_env_chunks)
        else:
            _kv_tiles = max(1, (args.max_seq_length + 63) // 64)
            num_kv_chunks = max(1, min(16, _kv_tiles))
        assert num_kv_chunks >= 1
        # The merge is otherwise one task per q group -- 2 CUs of 256, each
        # thread carrying kv_lora/16 = 32 unrolled softmax chains. Slice the
        # 512-wide latent so the merge fans out instead. 8 slices puts
        # kv_lora/8/16 = 4 dims on each thread across 16 tasks.
        MLA_MERGE_DIM_SPLITS = int(
            os.environ.get("GLM_MLA_MERGE_DIM_SPLITS", "16"))
        assert kv_lora % (MLA_MERGE_DIM_SPLITS * 16) == 0
        # Off by default: measured a dead heat (8.025 vs 8.026 ms). The win it
        # buys gpt-oss is deleting a separate readback+flush pass inside the
        # fused layer task, and there is no such pass here while the merge is
        # still its own task. At dim_splits=16 each thread writes 2 bf16 and
        # the store path is not the bottleneck either way. Kept plumbed for
        # when the merge moves inside a fused MLA layer.
        MLA_MERGE_WRITE_THROUGH = (
            os.environ.get("GLM_MLA_MERGE_WT", "0") == "1")
        print(f"[CFG] q_heads={num_heads}->{num_heads_pad} "
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
        # The q side cannot do this: q_b's *output* must stay 512-aligned and
        # keep the num_heads_pad * qk_dim stride that MLA decode indexes with.
        o_proj_red = num_heads * kv_lora
        if o_proj_red % 256 != 0:
            o_proj_red = num_heads_pad * kv_lora
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
        OPROJ_GEMV_ROWS = int(os.environ.get("GLM_OPROJ_GEMV_ROWS", "16"))
        use_gemv_oproj = OPROJ_GEMV_ROWS > 0 and not (
            GANG_K_SPLITS > 1 and o_proj_red % (GANG_K_SPLITS * 256) == 0)
        oproj_tile_n = OPROJ_GEMV_ROWS if use_gemv_oproj else GANG_TILE_N
        # 1.97 GB/token of bf16 weight, the last big one left. The packing is
        # the same pack_dense_mxfp8 the MFMA kernels use: its data half is
        # plain row-major, and only the MFMA *gather* wanted the split layout,
        # so the GEMV reads it as-is. See gang_gemv_mxfp8_mi300.cuh.
        use_mxfp8_oproj = OPROJ_MXFP8 and use_gemv_oproj
        if use_mxfp8_oproj:
            # 16 fp8 per lane per iteration over 256/rows lanes.
            assert o_proj_red % ((256 // oproj_tile_n) * 16) == 0, o_proj_red
        n_tiles_xcd = hidden_size // 8 // oproj_tile_n
        # The split-K task takes a reduction_override now, so de-padding and
        # split-K compose: it reduces over the leading o_proj_red columns and
        # splits *those* k_splits ways.
        use_splitk_oproj = (GANG_K_SPLITS > 1
                            and o_proj_red % (GANG_K_SPLITS * 256) == 0)
        print(f"[CFG] o_proj K={o_proj_red} out={hidden_size} "
              f"tile_n={oproj_tile_n} gemv={int(use_gemv_oproj)} "
              f"mxfp8={int(use_mxfp8_oproj)} "
              f"n_tiles/XCD={n_tiles_xcd} split-K="
              f"{GANG_K_SPLITS if use_splitk_oproj else 1} -> "
              f"{n_tiles_xcd * (GANG_K_SPLITS if use_splitk_oproj else 1) * 8}"
              f" tasks (of {num_workers} workers)")

        y = make_tensor("embed_out", (bs, hidden_size))
        rmsnorm_out = make_tensor("rmsnorm_out", (bs, hidden_size))
        qkv_a_out = make_tensor("qkv_a_out", (bs, qkv_a_pad))
        q_a_norm_out = make_tensor("q_a_norm_out", (bs, qkv_a_pad))
        # The absorbed q_b_proj writes the roped Q straight into this, so the
        # separate q_absorbed staging buffer the KV update used to read is gone.
        mla_q_ws = make_tensor("mla_q_workspace", (bs, num_heads_pad * qk_dim))
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
        # n_tiles_xcd extra floats to hold its own counters. The kernel's row
        # stride stays n_tiles*tile_n, so this only holds at bs == 1.
        assert bs == 1, "split-K o_proj workspace layout assumes bs == 1"
        splitk_ws = make_tensor("splitk_workspace",
                                (bs, hidden_size + 8 * n_tiles_xcd),
                                torch_dtype=torch.float32)
        rmsnorm_out_moe = make_tensor("rmsnorm_out_moe", (bs, hidden_size))
        layer_out = make_tensor("layer_out", (bs, hidden_size))

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
                t = torch.zeros(num_experts_total, size,
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

        for i, layer in enumerate(model.model.layers):
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
                qkv_a_stack = pack_dense_mxfp8(qkv_a_stack, QKV_MXFP8_OPW)
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

            q_b_absorbed = absorb_q_b(
                attn.q_b_proj.weight.data, attn._w_uk,
                num_heads, qk_nope, qk_rope, q_lora).to(torch.bfloat16)
            # q_b only wants the q_a half of the fused [q_a | latent] row, so
            # it reduces over q_lora_pad rather than the full qkv_a_pad. Both
            # are legal reductions (256-aligned) and the q_a half leads the
            # row, but stopping at q_lora_pad halves the widest weight in the
            # model -- [num_heads_pad * qk_dim, qkv_a_pad] is 75.5 MB/layer on
            # Flash, and every column past q_lora_pad is zero.
            q_b_absorbed = pad_cols(q_b_absorbed, q_lora_pad)
            q_b_absorbed = pad_rows(q_b_absorbed, num_heads_pad * qk_dim)
            if QB_MXFP8:
                # OPW is pinned to qk_rope_head_dim so a head's rope slice is
                # exactly one workgroup; 18432/64/8 = 36 tiles per XCD, which
                # is plenty of work, so there is no narrow-tile question here.
                q_b_absorbed = pack_dense_mxfp8(q_b_absorbed, qk_rope)
            w_q_b = _attach_input_keep(q_b_absorbed,
                                       f"layer_{i}_q_b_absorbed")

            o_absorbed = absorb_o_proj(
                attn.o_proj.weight.data, attn._w_uv,
                num_heads, v_head, kv_lora).to(torch.bfloat16)
            o_absorbed = pad_cols(o_absorbed, o_proj_red)
            if use_mxfp8_oproj:
                # One workgroup per GEMV tile, so the packing's workgroup axis
                # *is* the tile axis: 2048 / 16 = 128 workgroups, 16 per XCD,
                # exactly the tile count the bf16 GEMV had.
                o_absorbed = pack_dense_mxfp8(o_absorbed, oproj_tile_n)
            w_o = _attach_input_keep(o_absorbed, f"layer_{i}_o_absorbed")

            attn._w_uk = None
            attn._w_uv = None
            _release(attn.q_b_proj.weight, attn.kv_b_proj.weight,
                     attn.o_proj.weight)

            # latent_cache[i] is [pages, page_size, 576]; the kernels want a
            # 4-D [pages, page_size, 1, 576]. unsqueeze is a contiguous view.
            kv_cache = _attach_input_keep(
                model.model.latent_cache[i].unsqueeze(2),
                f"layer_{i}_latent_cache")

            # 1. input_layernorm + [q_a_proj | kv_a_proj_with_mqa]
            if DENSE_MXFP8:
                mpk.gang_rmsnorm_linear_mxfp8_bias_layer(
                    norm_input=x,
                    norm_weight=w_norm,
                    norm_output=rmsnorm_out,
                    mxfp8_weight=w_qkv_a,
                    bias=zero_bias(qkv_a_pad),
                    output=qkv_a_out,
                    actual_hidden_dim=hidden_size,
                    output_per_wg=QKV_MXFP8_OPW,
                    output_stride=qkv_a_pad,
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
            if QB_MXFP8:
                mpk.gang_rmsnorm_linear_mxfp8_bias_mla_kvupd_layer(
                    norm_input=qkv_a_out,
                    norm_weight=w_q_a_norm,
                    norm_output=q_a_norm_out,
                    mxfp8_weight=w_q_b,
                    bias=zero_bias(num_heads_pad * qk_dim),
                    kv_norm=w_kv_a_norm,
                    cos_pos_embed=cos_pos_embed,
                    sin_pos_embed=sin_pos_embed,
                    kv_cache=kv_cache,
                    q_workspace=mla_q_ws,
                    actual_hidden_dim=q_lora,
                    output_per_wg=qk_rope,
                    output_stride=num_heads_pad * qk_dim,
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
                    bias=zero_bias(num_heads_pad * qk_dim),
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
                    output_stride=num_heads_pad * qk_dim,
                    wgm=GANG_WGM,
                    block_dim=(256, 1, 1),
                )
            # 4. absorbed MLA decode, split over (q_group, kv_chunk)
            mpk.gang_mla_decode_layer(
                q_workspace=mla_q_ws,
                kv_cache=kv_cache,
                lse=mla_lse,
                output=(mla_o_acc if num_kv_chunks > 1 else attn_out),
                mla_params=(num_heads_pad, kv_lora, qk_rope, qk_head_dim,
                            num_kv_chunks),
                block_dim=(256, 1, 1),
            )
            if num_kv_chunks > 1:
                # merge_splitkv_ck_fmha treats the q groups as kv heads:
                # merge_task_offset = bid.y (runtime.cc), so grid.y indexes the
                # group and each task merges 16 q heads' chunks.
                mpk.paged_attention_ck_fmha_merge_layer(
                    lse=mla_lse,
                    output_tmp=mla_o_acc,
                    output=attn_out,
                    attention_params=(num_heads_pad, kv_lora, num_kv_chunks,
                                      num_q_groups),
                    grid_dim=(args.max_num_batched_requests,
                              num_q_groups * MLA_MERGE_DIM_SPLITS, 1),
                    block_dim=(256, 1, 1),
                    dim_splits=MLA_MERGE_DIM_SPLITS,
                    write_through=MLA_MERGE_WRITE_THROUGH,
                )
            # 5. absorbed o_proj + residual
            if use_splitk_oproj:
                mpk.gang_splitk_linear_with_residual_layer(
                    input=attn_out,
                    weight=w_o,
                    residual=x,
                    workspace=splitk_ws,
                    output=attn_proj_out,
                    tile_n=GANG_TILE_N,
                    output_stride=hidden_size,
                    k_splits=GANG_K_SPLITS,
                    reduction_size=o_proj_red,
                    block_dim=(256, 1, 1),
                )
            elif use_mxfp8_oproj:
                mpk.gang_gemv_mxfp8_with_residual_layer(
                    input=attn_out,
                    mxfp8_weight=w_o,
                    residual=x,
                    output=attn_proj_out,
                    rows_per_wg=oproj_tile_n,
                    output_stride=hidden_size,
                    reduction_size=o_proj_red,
                    wgm=GANG_WGM,
                    block_dim=(256, 1, 1),
                )
            else:
                mpk.gang_linear_with_residual_layer(
                    input=attn_out,
                    weight=w_o,
                    residual=x,
                    output=attn_proj_out,
                    tile_n=oproj_tile_n,
                    output_stride=hidden_size,
                    wgm=GANG_WGM,
                    reduction_size=o_proj_red,
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
                mpk.gang_linear_with_residual_layer(
                    input=dense_act,
                    weight=w_dense_down,
                    residual=attn_proj_out,
                    output=layer_out,
                    tile_n=GANG_TILE_N,
                    output_stride=hidden_size,
                    wgm=GANG_WGM,
                    block_dim=(256, 1, 1),
                )
                x = layer_out
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

            # Expert stacks, shared expert last. With the SwiGLU fused into
            # W13's epilogue the gate and up rows are interleaved pairwise so
            # the pair meets in one thread's accumulators; unfused, the plain
            # [gate | up] concat is what moe_silu_mul expects.
            experts = list(layer.mlp.experts) + [shared]
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
                gu_stack = pack_moe_mxfp8(gu_stack, MOE_MXFP8_OPW)
                down_stack = pack_moe_mxfp8(down_stack, MOE_MXFP8_OPW)
            w_moe_gu = _attach_input_keep(gu_stack, f"layer_{i}_moe_gate_up")
            w_moe_down = _attach_input_keep(down_stack, f"layer_{i}_moe_down")
            for e in experts:
                _release(e.gate_proj.weight, e.up_proj.weight,
                         e.down_proj.weight)

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
            moe_w13_layer = (mpk.gang_moe_w13_linear_mxfp8_layer if MOE_MXFP8
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
                **({"output_per_wg": MOE_MXFP8_OPW} if MOE_MXFP8 else {}),
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
                routing_weight=moe_topk_weight if FUSE_MOE_MULSUMADD else None,
                block_dim=(256, 1, 1),
                **({"output_per_wg": MOE_MXFP8_OPW} if MOE_MXFP8 else {}),
            )
            if FUSE_MOE_MULSUMADD:
                mpk.moe_residual_add_f32_layer(
                    workspace_f32=moe_ws_f32,
                    residual=attn_proj_out,
                    output=layer_out,
                    grid_dim=(1, 1, 1),
                    block_dim=(256, 1, 1),
                )
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

        # ── Tail: final norm + LM head + argmax ──────────────────────────────
        w_final_norm = _attach_input_keep(model.model.norm.weight.data,
                                          "model_norm_weight")
        w_lm_head = _attach_input_keep(
            pack_dense_mxfp8(lm_head_weight, DENSE_MXFP8_OPW) if DENSE_MXFP8
            else lm_head_weight, "lm_head")
        if DENSE_MXFP8:
            mpk.gang_rmsnorm_linear_mxfp8_bias_layer(
                norm_input=x,
                norm_weight=w_final_norm,
                norm_output=rmsnorm_out,
                mxfp8_weight=w_lm_head,
                bias=zero_bias(vocab_size),
                output=argmax_in,
                actual_hidden_dim=hidden_size,
                output_per_wg=DENSE_MXFP8_OPW,
                output_stride=vocab_size,
                block_dim=(256, 1, 1),
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

    if not args.use_mirage:
        prompt_len = prompt_lengths[0].item()
        decode_limit = prompt_len + output_len
        cur_pos = prompt_len
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
        if save_path and rank == 0:
            slice_end = min(end_idx, prompt_len + MAX_SAVE_TOKENS)
            out = {
                "token_ids": tokens[0, prompt_len:slice_end].tolist(),
                "text": tokenizer.decode(tokens[0, :end_idx],
                                         skip_special_tokens=True),
                "generate_length": max(0, end_idx - prompt_len),
                "mode": "torch",
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

        if save_path and rank == 0:
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
        if _fwd_dropped > 0:
            print(f"  NOTE: device per-iter ring overflowed -- {_fwd_dropped} "
                  f"of {_fwd_total_iters} samples dropped; all-iteration "
                  f"device average {_fwd_total_avg:.3f}ms/iter")
        print("=" * 80)
