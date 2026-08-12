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


def interleave_gate_up(w_gate: torch.Tensor, w_up: torch.Tensor,
                       num_groups: int) -> torch.Tensor:
    """Lay out [gate | up] as `num_groups` consecutive [gate_chunk; up_chunk]
    slabs.

    silu_mul_task_impl reads its multiplicand at ``input_ptr + OUTPUT_SIZE``,
    where OUTPUT_SIZE is the *per-block* width after grid.x partitioning. A
    flat [gate(I) | up(I)] weight is therefore only read correctly when
    grid.x == 1. Grouping the rows the same way mpk.shuffle_tensors does (see
    the llama3 QKV shuffle) keeps grid.x == 8, one silu_mul task per XCD.

    moe_silu_mul_layer does *not* need this: its input is 3-D and dim 2 is
    unpartitioned, so the routed experts keep the plain concat layout.
    """
    inter = w_gate.shape[0]
    assert inter % num_groups == 0, (inter, num_groups)
    chunk = inter // num_groups
    parts = []
    for g in range(num_groups):
        parts.append(w_gate[g * chunk:(g + 1) * chunk])
        parts.append(w_up[g * chunk:(g + 1) * chunk])
    return torch.cat(parts, dim=0).contiguous()


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
        assert (2 * dense_inter) % GANG_OUT_ALIGN == 0
        assert dense_inter % GANG_RED_ALIGN == 0

        # Split KV along the sequence across the 8 XCDs. MLA has a single
        # shared latent head, so the gpt-oss `kv_head == xcd_id` mapping does
        # not carry over -- instead each XCD claims one (q_group, kv_chunk)
        # work item and the merge task recombines them. Aim for exactly 8 work
        # items so every XCD gets one.
        num_q_groups = num_heads_pad // 16
        _env_chunks = os.environ.get("GLM_MLA_NUM_KV_CHUNKS")
        if _env_chunks is not None:
            num_kv_chunks = int(_env_chunks)
        else:
            _kv_tiles = max(1, (args.max_seq_length + 63) // 64)
            num_kv_chunks = max(1, min(8 // max(1, num_q_groups), _kv_tiles))
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
              f"merge_dim_splits={MLA_MERGE_DIM_SPLITS}")

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
        n_tiles_xcd = hidden_size // 8 // GANG_TILE_N
        # The split-K task takes a reduction_override now, so de-padding and
        # split-K compose: it reduces over the leading o_proj_red columns and
        # splits *those* k_splits ways.
        use_splitk_oproj = (GANG_K_SPLITS > 1
                            and o_proj_red % (GANG_K_SPLITS * 256) == 0)
        print(f"[CFG] o_proj K={o_proj_red} out={hidden_size} "
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
            w_qkv_a = _attach_input_keep(
                torch.cat([pad_rows(attn.q_a_proj.weight.data, q_lora_pad),
                           pad_rows(attn.kv_a_proj_with_mqa.weight.data,
                                    kv_a_out_pad)], dim=0).contiguous(),
                f"layer_{i}_qkv_a_proj")
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
            w_q_b = _attach_input_keep(q_b_absorbed,
                                       f"layer_{i}_q_b_absorbed")

            o_absorbed = absorb_o_proj(
                attn.o_proj.weight.data, attn._w_uv,
                num_heads, v_head, kv_lora).to(torch.bfloat16)
            o_absorbed = pad_cols(o_absorbed, o_proj_red)
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
            else:
                mpk.gang_linear_with_residual_layer(
                    input=attn_out,
                    weight=w_o,
                    residual=x,
                    output=attn_proj_out,
                    tile_n=GANG_TILE_N,
                    output_stride=hidden_size,
                    wgm=GANG_WGM,
                    reduction_size=o_proj_red,
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

            # Expert stacks, shared expert last. moe_silu_mul reads its
            # multiplicand at a 3-D offset, so unlike the dense MLP these keep
            # the plain [gate | up] concat layout.
            experts = list(layer.mlp.experts) + [shared]
            w_moe_gu = _attach_input_keep(
                torch.stack([
                    torch.cat([e.gate_proj.weight.data,
                               e.up_proj.weight.data], dim=0)
                    for e in experts
                ]).contiguous(),
                f"layer_{i}_moe_gate_up")
            w_moe_down = _attach_input_keep(
                torch.stack([e.down_proj.weight.data
                             for e in experts]).contiguous(),
                f"layer_{i}_moe_down")
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
            mpk.gang_moe_w13_linear_layer(
                input=rmsnorm_out_moe,
                weight=w_moe_gu,
                moe_routing_indices=moe_routing_indices,
                moe_mask=moe_mask,
                bias=zero_moe_bias(2 * moe_inter),
                output=moe_mid,
                block_dim=(256, 1, 1),
            )
            mpk.moe_silu_mul_layer(
                input=moe_mid,
                output=moe_act,
                grid_dim=(args.max_num_batched_tokens, topk_total, 1),
                block_dim=(256, 1, 1),
            )
            mpk.gang_moe_w2_linear_layer(
                input=moe_act,
                weight=w_moe_down,
                moe_routing_indices=moe_routing_indices,
                moe_mask=moe_mask,
                bias=zero_moe_bias(hidden_size),
                output=moe_out,
                block_dim=(256, 1, 1),
            )
            mpk.moe_mul_sum_add_layer(
                input=moe_out,
                weight=moe_topk_weight,
                residual=attn_proj_out,
                output=layer_out,
                grid_dim=(args.max_num_batched_tokens, hidden_size // 256, 1),
                block_dim=(256, 1, 1),
            )
            x = layer_out

        # ── Tail: final norm + LM head + argmax ──────────────────────────────
        w_final_norm = _attach_input_keep(model.model.norm.weight.data,
                                          "model_norm_weight")
        w_lm_head = _attach_input_keep(lm_head_weight, "lm_head")
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
