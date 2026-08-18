# Copyright 2025 Mirage Team. Adapted from HuggingFace DeepSeek-V3 / GLM MoE
# implementations.
# Licensed under the Apache License, Version 2.0.
"""GLM MoE + MLA models for Mirage inference.

Covers two checkpoints that share one architecture:

  * ``zai-org/GLM-5-FP8``      -- ``GlmMoeDsaForCausalLM`` (``glm_moe_dsa``),
    78 layers, 64 heads, 256 routed experts, FP8 128x128 blockscale weights,
    plus a DSA sparse-attention indexer.
  * ``zai-org/GLM-4.7-Flash``  -- ``Glm4MoeLiteForCausalLM`` (``glm4_moe_lite``),
    47 layers, 20 heads, 64 routed experts, BF16 weights, no indexer.

Both use MLA with identical head geometry (qk_nope 192 + qk_rope 64,
v_head 256, kv_lora 512) and the same ``noaux_tc`` sigmoid router with an
``e_score_correction_bias``, ``norm_topk_prob`` and a ``routed_scaling_factor``.
That makes GLM-4.7-Flash the natural bring-up vehicle for the GLM-5 megakernel
path -- same tasks, one eighth the weights.

Two deliberate simplifications relative to the HF reference, both documented
where they happen:

  * **Absorbed MLA.** ``W_UK`` is folded into the query projection and ``W_UV``
    into ``o_proj``, so the KV cache holds one 576-wide latent row per token
    (512 ``c_kv`` + 64 ``k_rope``) instead of materialised per-head K and V.
    This is algebraically exact, and it is the same formulation
    ``gang_mla_decode_kernel`` implements -- keeping the reference in the same
    formulation is what makes a Torch-vs-Mirage comparison meaningful.
  * **Dense attention, no DSA indexer.** GLM-5's indexer selects the top
    ``index_topk = 2048`` positions. Below 2048 tokens of context that
    selection is the identity, so at Stage-1 sequence lengths dense attention
    *is* sparse attention. The indexer weights are loaded and ignored;
    ``ForCausalLM.forward`` raises if the context ever exceeds ``index_topk``
    so this can never silently become an approximation.
"""

import json
import math
import os
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel


class GlmMoeDsaConfig(PretrainedConfig):
    """Config for ``glm_moe_dsa`` and ``glm4_moe_lite``.

    Neither model type exists in transformers 4.x (both land in 5.x), so
    ``AutoConfig.from_pretrained`` cannot parse their ``config.json``. Rather
    than pin an unreleased transformers, read the JSON directly -- the two
    configs differ only in which optional keys are present.
    """

    model_type = "glm_moe_dsa"

    def __init__(self, **kw):
        self.vocab_size = kw.get("vocab_size", 154880)
        self.hidden_size = kw.get("hidden_size", 6144)
        self.intermediate_size = kw.get("intermediate_size", 12288)
        self.moe_intermediate_size = kw.get("moe_intermediate_size", 2048)
        self.num_hidden_layers = kw.get("num_hidden_layers", 78)
        self.num_attention_heads = kw.get("num_attention_heads", 64)
        self.num_key_value_heads = kw.get("num_key_value_heads",
                                          self.num_attention_heads)
        self.rms_norm_eps = kw.get("rms_norm_eps", 1e-5)
        self.max_position_embeddings = kw.get("max_position_embeddings", 202752)
        self.attention_bias = kw.get("attention_bias", False)
        self.hidden_act = kw.get("hidden_act", "silu")
        self.initializer_range = kw.get("initializer_range", 0.02)
        self.tie_word_embeddings = kw.get("tie_word_embeddings", False)

        # MLA
        self.q_lora_rank = kw.get("q_lora_rank", 2048)
        self.kv_lora_rank = kw.get("kv_lora_rank", 512)
        self.qk_nope_head_dim = kw.get("qk_nope_head_dim", 192)
        self.qk_rope_head_dim = kw.get("qk_rope_head_dim", 64)
        self.v_head_dim = kw.get("v_head_dim", 256)
        self.qk_head_dim = kw.get(
            "qk_head_dim", self.qk_nope_head_dim + self.qk_rope_head_dim)

        # RoPE. GLM-5 nests theta under `rope_parameters`; GLM-4.7-Flash has it
        # flat. `rope_interleave` is explicit on GLM-5 and absent on
        # GLM-4.7-Flash, where the family default (interleaved) applies.
        rope_params = kw.get("rope_parameters") or {}
        self.rope_theta = kw.get("rope_theta", rope_params.get("rope_theta", 1e6))
        self.rope_interleave = kw.get("rope_interleave", True)

        # MoE
        self.n_routed_experts = kw.get("n_routed_experts", 256)
        self.n_shared_experts = kw.get("n_shared_experts", 1)
        self.num_experts_per_tok = kw.get("num_experts_per_tok", 8)
        self.first_k_dense_replace = kw.get("first_k_dense_replace", 3)
        self.n_group = kw.get("n_group", 1)
        self.topk_group = kw.get("topk_group", 1)
        self.norm_topk_prob = kw.get("norm_topk_prob", True)
        self.routed_scaling_factor = kw.get("routed_scaling_factor", 2.5)
        self.scoring_func = kw.get("scoring_func", "sigmoid")
        self.topk_method = kw.get("topk_method", "noaux_tc")
        self.moe_layer_freq = kw.get("moe_layer_freq", 1)

        # DSA. Absent on GLM-4.7-Flash, which has no indexer.
        self.index_n_heads = kw.get("index_n_heads", 0)
        self.index_head_dim = kw.get("index_head_dim", 0)
        self.index_topk = kw.get("index_topk", 0)
        self.indexer_rope_interleave = kw.get("indexer_rope_interleave", True)

        # MTP. Not executed here -- layer `num_hidden_layers` is skipped.
        self.num_nextn_predict_layers = kw.get("num_nextn_predict_layers", 0)

        self.architectures = kw.get("architectures", ["GlmMoeDsaForCausalLM"])
        self.hf_model_type = kw.get("model_type", self.model_type)
        super().__init__(
            pad_token_id=kw.get("pad_token_id"),
            eos_token_id=kw.get("eos_token_id"),
            tie_word_embeddings=self.tie_word_embeddings,
        )

    @property
    def has_indexer(self) -> bool:
        return self.index_n_heads > 0

    @classmethod
    def from_model_path(cls, model_path: str) -> "GlmMoeDsaConfig":
        with open(os.path.join(model_path, "config.json")) as f:
            raw = json.load(f)
        raw.pop("quantization_config", None)
        return cls(**raw)


class GlmRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class GlmRotaryEmbedding(nn.Module):
    """RoPE over the ``qk_rope_head_dim`` slice only.

    Returns ``cos``/``sin`` of width ``qk_rope_head_dim`` in HF layout
    (``emb = cat(freqs, freqs)``), which is what both the Torch path here and
    ``rope_interleave_partial`` in the megakernel consume.
    """

    def __init__(self, config: GlmMoeDsaConfig):
        super().__init__()
        self.config = config
        dim = config.qk_rope_head_dim
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0

    @torch.no_grad()
    def forward(self, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        ).to(position_ids.device)
        position_ids_expanded = position_ids[:, None, :].float()
        with torch.autocast(device_type=position_ids.device.type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=torch.bfloat16), sin.to(dtype=torch.bfloat16)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin, interleave: bool):
    """Rotate the trailing rope slice of ``x`` -- ``[..., rope_dim]``.

    With ``interleave`` the rotary pair for angle j is (2j, 2j+1); the result is
    still emitted in half-split order, exactly as transformers'
    ``apply_rotary_pos_emb_interleave`` does (it de-interleaves with a
    view/transpose, then applies the ordinary half-split rotation). The output
    permutation is identical for Q and K, so the QK dot product is unchanged.
    ``rope_interleave_partial`` in the megakernel matches this convention.
    """
    if interleave:
        shape = x.shape
        x = x.view(*shape[:-1], shape[-1] // 2, 2).transpose(-1, -2).reshape(shape)
    return (x * cos) + (_rotate_half(x) * sin)


class GlmMLA(nn.Module):
    """Absorbed multi-head latent attention with a paged latent KV cache.

    The cache row is ``[c_kv (kv_lora_rank) | k_rope (qk_rope_head_dim)]`` --
    576 elements for both checkpoints -- and is shared by all query heads. QK
    reduces over the full 576; PV accumulates over the leading
    ``kv_lora_rank`` dims of the *same* row, since after absorption V is not a
    separate tensor. ``gang_mla_decode_kernel`` computes exactly this.
    """

    def __init__(self, config: GlmMoeDsaConfig, latent_cache, layer_idx: int,
                 world_size: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.world_size = world_size
        self.num_heads = config.num_attention_heads
        self.num_local_heads = self.num_heads // world_size
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.latent_cache = latent_cache
        # MLA scales by the unabsorbed qk head dim (192 + 64), not the 576-wide
        # absorbed reduction. The megakernel takes qk_head_dim as its own
        # registration param for the same reason.
        self.scaling = self.qk_head_dim ** -0.5

        bias = config.attention_bias
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                config.hidden_size, self.num_local_heads * self.qk_head_dim,
                bias=False)
        else:
            self.q_a_proj = nn.Linear(config.hidden_size, self.q_lora_rank,
                                      bias=bias)
            self.q_a_layernorm = GlmRMSNorm(self.q_lora_rank,
                                            eps=config.rms_norm_eps)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.num_local_heads * self.qk_head_dim,
                bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim,
            bias=bias)
        self.kv_a_layernorm = GlmRMSNorm(self.kv_lora_rank,
                                         eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_local_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False)
        self.o_proj = nn.Linear(self.num_local_heads * self.v_head_dim,
                                config.hidden_size, bias=bias)

        # Absorption operands, materialised lazily on first use so that
        # from_pretrained can overwrite kv_b_proj.weight beforehand.
        self._w_uk = None  # [H, qk_nope, kv_lora]
        self._w_uv = None  # [H, v_head, kv_lora]

    def _absorb(self):
        if self._w_uk is not None:
            return
        # kv_b_proj.weight is [H * (qk_nope + v_head), kv_lora].
        w = self.kv_b_proj.weight.view(
            self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank)
        self._w_uk = w[:, : self.qk_nope_head_dim, :].contiguous().float()
        self._w_uv = w[:, self.qk_nope_head_dim:, :].contiguous().float()

    def forward(self, hidden_states, position_embeddings=None, step=None):
        self._absorb()
        bsz, q_len, _ = hidden_states.size()
        assert bsz == 1, "reference path is batch-1, matching the demo loop"

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(q_len, self.num_local_heads, self.qk_head_dim)
        q_nope, q_rope = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed = self.kv_a_proj_with_mqa(hidden_states).view(
            q_len, self.kv_lora_rank + self.qk_rope_head_dim)
        c_kv, k_rope = torch.split(
            compressed, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # The latent that gets cached is the *post-layernorm* one: kv_b_proj is
        # linear in it, which is what makes the absorption exact.
        c_kv = self.kv_a_layernorm(c_kv)

        # The caller passes the slice for the current tokens only (same
        # convention as demo/gpt_oss), so index it positionally, not by `step`.
        cos, sin = position_embeddings
        interleave = self.config.rope_interleave
        cos_q, sin_q = cos[0, :q_len], sin[0, :q_len]
        q_rope = apply_rope(q_rope, cos_q.unsqueeze(1), sin_q.unsqueeze(1),
                            interleave)
        k_rope = apply_rope(k_rope, cos_q, sin_q, interleave)

        # Append to the paged latent cache. Page 0 only: the reference path
        # runs one request with page_size >= max_seq_length, matching the demo.
        row = torch.cat([c_kv, k_rope], dim=-1)  # [q_len, 576]
        if q_len > 1:
            self.latent_cache[self.layer_idx, 0, :q_len] = row
            kv_len = q_len
        else:
            self.latent_cache[self.layer_idx, 0, step] = row
            kv_len = int(step.item()) + 1

        kv = self.latent_cache[self.layer_idx, 0, :kv_len].float()  # [S, 576]
        c_cache = kv[:, : self.kv_lora_rank]                        # [S, 512]

        # q_absorbed = [q_nope @ W_UK | q_rope] -- one 576-wide query per head.
        q_absorbed = torch.einsum(
            "qhn,hnc->qhc", q_nope.float(), self._w_uk)
        q_full = torch.cat([q_absorbed, q_rope.float()], dim=-1)  # [q, H, 576]

        scores = torch.einsum("qhd,sd->hqs", q_full, kv) * self.scaling
        if q_len > 1:
            causal = torch.triu(
                torch.full((q_len, kv_len), float("-inf"),
                           device=scores.device, dtype=scores.dtype),
                diagonal=kv_len - q_len + 1)
            scores = scores + causal.unsqueeze(0)
        probs = F.softmax(scores, dim=-1, dtype=torch.float32)

        # PV accumulates over the leading kv_lora dims of the same rows.
        ctx = torch.einsum("hqs,sc->qhc", probs, c_cache)     # [q, H, 512]
        # W_UV folds out here rather than into o_proj's weight, which keeps
        # o_proj loadable straight from the checkpoint.
        out = torch.einsum("qhc,hvc->qhv", ctx, self._w_uv)   # [q, H, 256]
        out = out.reshape(1, q_len, self.num_local_heads * self.v_head_dim)
        out = self.o_proj(out.to(hidden_states.dtype))

        if self.world_size > 1:
            dist.all_reduce(out)
        return out


class GlmMLP(nn.Module):
    """SwiGLU MLP -- the dense layers, the shared expert, and each routed
    expert are all this, only the intermediate size differs."""

    def __init__(self, hidden_size, intermediate_size, bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# The eight E2M1 magnitudes, sign in bit 3 -- the exact inverse of the ladder
# of thresholds demo.py's quantize_mxfp4 encodes with.
_E2M1_LUT = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                          -0., -.5, -1., -1.5, -2., -3., -4., -6.])


def dequantize_mxfp4(blocks: torch.Tensor, scales: torch.Tensor,
                     out_dtype=torch.bfloat16) -> torch.Tensor:
    """MXFP4 [out, K/2] nibbles + [out, K/32] E8M0 exponents -> [out, K].

    The inverse of demo.py's ``quantize_mxfp4``, and it has to agree with it
    on both conventions the packing chose: nibble order within a byte is
    element 2b low / 2b+1 high, and a zero exponent means a scale of 1.0
    rather than 2^-127 (quantize_mxfp4 emits se == 0 for an all-zero block).

    This exists so GLM-5 has a Torch reference at all. The checkpoint only
    ever lands on disk as MXFP4 -- the bf16 form is 1488 GB -- so without a
    dequantiser the megakernel's output has nothing to be compared against.
    """
    lut = _E2M1_LUT.to(blocks.device)
    lo, hi = (blocks & 0x0F).long(), (blocks >> 4).long()
    vals = torch.stack([lut[lo], lut[hi]], dim=-1).reshape(
        *blocks.shape[:-1], blocks.shape[-1] * 2)
    K = vals.shape[-1]
    sf = torch.where(scales == 0,
                     torch.ones_like(scales, dtype=torch.float32),
                     (scales.to(torch.int32) << 23).view(torch.float32))
    return (vals.reshape(*vals.shape[:-1], K // 32, 32)
            * sf.unsqueeze(-1)).reshape(*blocks.shape[:-1], K).to(out_dtype)


class GlmTopkRouter(nn.Module):
    """``noaux_tc``: sigmoid gating, bias-corrected selection, renormalised
    weights, then a constant rescale.

    The bias shifts *which* experts win but not the weight they carry -- the
    gathered weights come from the unbiased sigmoid. ``moe_topk_sigmoid_bias``
    in the megakernel implements the same split.
    """

    def __init__(self, config: GlmMoeDsaConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty(self.n_routed_experts, config.hidden_size))
        self.register_buffer("e_score_correction_bias",
                             torch.zeros(self.n_routed_experts))

    @torch.no_grad()
    def get_topk_indices(self, scores):
        scores_for_choice = (scores.view(-1, self.n_routed_experts)
                             + self.e_score_correction_bias.unsqueeze(0))
        if self.n_group > 1:
            group_scores = (
                scores_for_choice.view(-1, self.n_group,
                                       self.n_routed_experts // self.n_group)
                .topk(2, dim=-1)[0].sum(dim=-1))
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1,
                                   sorted=False)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (group_mask.unsqueeze(-1)
                          .expand(-1, self.n_group,
                                  self.n_routed_experts // self.n_group)
                          .reshape(-1, self.n_routed_experts))
            scores_for_choice = scores_for_choice.masked_fill(
                ~score_mask.bool(), 0.0)
        return torch.topk(scores_for_choice, k=self.top_k, dim=-1,
                          sorted=False)[1]

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.hidden_size)
        router_logits = F.linear(hidden_states.float(), self.weight.float())
        scores = router_logits.sigmoid()
        topk_indices = self.get_topk_indices(scores)
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (
                topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights, router_logits


class GlmMoE(nn.Module):
    """Routed experts + one shared expert, optionally expert-parallel.

    ``ep_world > 1`` allocates only this rank's ``n_routed_experts //
    ep_world`` experts. That is not an optimisation at GLM-5's scale, it is
    what makes the model constructible: 256 experts x 76 layers in bf16 is
    1488 GB, against 288 GB of HBM per GPU.

    ``mxfp4_experts`` goes further and never materialises the bf16 form at
    all. One rank's 32-expert slice is 183 GB as bf16 and 46 GB as MXFP4, and
    the megakernel only ever reads the MXFP4 packing, so the bf16 is pure
    transient. The quantised experts are deliberately NOT nn.Modules --
    load_state_dict has no notion of a blocks/scales pair -- and are filled by
    ``GlmMoeDsaForCausalLM.from_pretrained`` into ``expert_mxfp4``.
    """

    def __init__(self, config: GlmMoeDsaConfig, ep_rank: int = 0,
                 ep_world: int = 1, mxfp4_experts: bool = False):
        super().__init__()
        self.config = config
        assert config.n_routed_experts % ep_world == 0, (
            f"{config.n_routed_experts} routed experts must divide by "
            f"ep_world {ep_world}")
        self.ep_local = config.n_routed_experts // ep_world
        self.ep_base = ep_rank * self.ep_local
        self.mxfp4_experts = mxfp4_experts
        if mxfp4_experts:
            self.experts = None
            # [{"gate_proj": (blocks, scales), "up_proj": ..., "down_proj": ...}]
            self.expert_mxfp4 = [dict() for _ in range(self.ep_local)]
        else:
            # Sharding the bf16 path too would need load_state_dict to remap
            # global expert index to local, which nothing needs: bf16 is the
            # GLM-4.7-Flash reference path and it runs unsharded.
            assert ep_world == 1, "bf16 experts do not support ep_world > 1"
            self.experts = nn.ModuleList([
                GlmMLP(config.hidden_size, config.moe_intermediate_size)
                for _ in range(self.ep_local)
            ])
        self.gate = GlmTopkRouter(config)
        self.shared_experts = GlmMLP(
            config.hidden_size,
            config.moe_intermediate_size * config.n_shared_experts)

    def _mxfp4_expert(self, e, x):
        """SwiGLU for one MXFP4 expert, dequantised on the fly.

        Not cached: the full bf16 form of one layer's 256 experts is 9.6 GB,
        and top-8 of 256 means a decode step touches 8. Dequantising the three
        projections costs ~38M elements, which is nothing next to loading them
        would be. This is the reference leg -- it is allowed to be slow, it is
        not allowed to hold a second copy of the model.
        """
        w = self.expert_mxfp4[e - self.ep_base]
        g, u, d = (dequantize_mxfp4(*w[p], out_dtype=x.dtype)
                   for p in ("gate_proj", "up_proj", "down_proj"))
        return F.linear(F.silu(F.linear(x, g)) * F.linear(x, u), d)

    def forward(self, hidden_states):
        if self.ep_local != self.config.n_routed_experts:
            raise NotImplementedError(
                "the Torch reference MoE needs every expert resident; this "
                "module holds only rank-local "
                f"{'MXFP4' if self.mxfp4_experts else 'bf16'} experts "
                f"[{self.ep_base}, {self.ep_base + self.ep_local}). Use the "
                "megakernel path (--use-mirage) for the sharded model.")
        expert = (self._mxfp4_expert if self.experts is None
                  else lambda e, x: self.experts[e](x))
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights, router_logits = self.gate(hidden_states)
        flat = hidden_states.view(-1, self.config.hidden_size)

        out = torch.zeros_like(flat, dtype=torch.float32)
        for tok in range(flat.shape[0]):
            for slot in range(topk_indices.shape[1]):
                e = int(topk_indices[tok, slot])
                out[tok] += (expert(e, flat[tok:tok + 1]).float()
                             * float(topk_weights[tok, slot]))[0]
        out = out.to(hidden_states.dtype).view(*orig_shape)
        # The shared expert runs on every token and is added unweighted.
        return out + self.shared_experts(residuals), router_logits


class GlmDecoderLayer(nn.Module):
    def __init__(self, config: GlmMoeDsaConfig, latent_cache, layer_idx: int,
                 world_size: int, ep_rank: int = 0, ep_world: int = 1,
                 mxfp4_experts: bool = False):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = GlmMLA(config, latent_cache, layer_idx, world_size)
        self.input_layernorm = GlmRMSNorm(config.hidden_size,
                                          eps=config.rms_norm_eps)
        self.post_attention_layernorm = GlmRMSNorm(config.hidden_size,
                                                   eps=config.rms_norm_eps)
        # first_k_dense_replace leading layers are plain dense MLPs at the full
        # `intermediate_size`; everything after is MoE.
        self.is_moe = layer_idx >= config.first_k_dense_replace
        if self.is_moe:
            self.mlp = GlmMoE(config, ep_rank=ep_rank, ep_world=ep_world,
                              mxfp4_experts=mxfp4_experts)
        else:
            self.mlp = GlmMLP(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states, position_embeddings=None, step=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states,
                                       position_embeddings=position_embeddings,
                                       step=step)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.is_moe:
            hidden_states, router_logits = self.mlp(hidden_states)
        else:
            hidden_states, router_logits = self.mlp(hidden_states), None
        hidden_states = residual + hidden_states
        return hidden_states, router_logits


class GlmPreTrainedModel(PreTrainedModel):
    config_class = GlmMoeDsaConfig
    _supports_sdpa = False

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, GlmRMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, GlmTopkRouter):
            module.weight.data.normal_(mean=0.0, std=std)
            module.e_score_correction_bias.data.zero_()


class GlmMoeDsaModel(GlmPreTrainedModel):
    def __init__(self, config: GlmMoeDsaConfig, world_size: int,
                 max_num_pages: int, page_size: int, num_layers: int = None,
                 ep_rank: int = 0, ep_world: int = 1,
                 mxfp4_experts: bool = False):
        super().__init__(config)
        self.config = config
        n = num_layers if num_layers is not None else config.num_hidden_layers

        # One 576-wide latent row per token, shared across all heads: 1.15 KB
        # per token per layer against 64 heads x 256 dims x 2 tensors for
        # unabsorbed MLA.
        self.latent_cache = torch.empty(
            (n, max_num_pages, page_size,
             config.kv_lora_rank + config.qk_rope_head_dim),
            dtype=torch.bfloat16, device="cuda")

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            GlmDecoderLayer(config, self.latent_cache, i, world_size,
                            ep_rank=ep_rank, ep_world=ep_world,
                            mxfp4_experts=mxfp4_experts)
            for i in range(n)
        ])
        self.norm = GlmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = GlmRotaryEmbedding(config)
        self.post_init()

    def forward(self, input_ids, position_embeddings=None, step=None):
        hidden_states = self.embed_tokens(input_ids)
        all_router_logits = []
        for layer in self.layers:
            hidden_states, router_logits = layer(
                hidden_states, position_embeddings=position_embeddings,
                step=step)
            all_router_logits.append(router_logits)
        return self.norm(hidden_states), all_router_logits


# ── FP8 128x128 blockscale dequantisation ────────────────────────────────────
# GLM-5-FP8 stores every GEMM weight as float8_e4m3fn `weight` plus an fp32
# `weight_scale_inv` holding one scale per 128x128 tile. Stage 2 consumes the
# quantised form directly; the Torch reference here dequantises so it can serve
# as the numerical oracle for that work.

FP8_BLOCK = 128


def dequantize_fp8_blockscale(weight: torch.Tensor, scale_inv: torch.Tensor,
                              dtype=torch.bfloat16) -> torch.Tensor:
    """``weight`` [out, in] float8_e4m3fn, ``scale_inv`` [ceil(out/128),
    ceil(in/128)] float32 -> dequantised [out, in]."""
    out_dim, in_dim = weight.shape
    w = weight.to(torch.float32)
    s = scale_inv.to(torch.float32)
    # repeat_interleave rather than a reshape: the last block along each axis
    # is partial whenever the dim is not a multiple of 128.
    s = s.repeat_interleave(FP8_BLOCK, dim=0).repeat_interleave(FP8_BLOCK, dim=1)
    return (w * s[:out_dim, :in_dim]).to(dtype)


class GlmMoeDsaForCausalLM(GlmPreTrainedModel):
    def __init__(self, config: GlmMoeDsaConfig, world_size: int = 1,
                 max_num_pages: int = 16, page_size: int = 4096,
                 num_layers: int = None, ep_rank: int = 0, ep_world: int = 1,
                 mxfp4_experts: bool = False):
        super().__init__(config)
        self.model = GlmMoeDsaModel(config, world_size, max_num_pages,
                                    page_size, num_layers=num_layers,
                                    ep_rank=ep_rank, ep_world=ep_world,
                                    mxfp4_experts=mxfp4_experts)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size,
                                 bias=False)
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, world_size=1,
                        max_num_pages=16, page_size=4096, num_layers=None,
                        random_weights=False, ep_rank=0, ep_world=1,
                        mxfp4_experts=False, verbose=True, **kwargs):
        import glob

        from safetensors import safe_open

        if os.path.isdir(pretrained_model_name_or_path):
            model_path = pretrained_model_name_or_path
        else:
            from huggingface_hub import snapshot_download
            allow = None
            if random_weights:
                # Config and tokenizer only -- enough to build the graph and
                # smoke-test shapes without pulling hundreds of GB.
                allow = ["*.json", "*.jinja", "*.txt"]
            model_path = snapshot_download(pretrained_model_name_or_path,
                                           allow_patterns=allow)

        config = GlmMoeDsaConfig.from_model_path(model_path)

        # Auto-detect the expert format from the index rather than making the
        # caller assert it: getting this wrong is not a clean failure. Claiming
        # MXFP4 over a bf16 checkpoint leaves every expert slot empty, and the
        # reverse loads uint8 nibbles as if they were weights.
        if mxfp4_experts is None and not random_weights:
            idx_f = os.path.join(model_path, "model.safetensors.index.json")
            mxfp4_experts = False
            if os.path.exists(idx_f):
                with open(idx_f) as fh:
                    mxfp4_experts = any(
                        k.endswith(".mxfp4_blocks")
                        for k in json.load(fh)["weight_map"])
            if verbose:
                print(f"[load] expert format: "
                      f"{'MXFP4 (pre-quantised)' if mxfp4_experts else 'bf16'}",
                      flush=True)
        mxfp4_experts = bool(mxfp4_experts)

        model = cls(config, world_size=world_size, max_num_pages=max_num_pages,
                    page_size=page_size, num_layers=num_layers,
                    ep_rank=ep_rank, ep_world=ep_world,
                    mxfp4_experts=mxfp4_experts)
        if random_weights:
            print("Using randomly initialised weights (--random-weights): "
                  "shapes and kernels are exercised, output text is not "
                  "meaningful.")
            return model

        n_layers = len(model.model.layers)
        files = sorted(glob.glob(os.path.join(model_path,
                                              "model-*.safetensors")))
        if not files:
            raise FileNotFoundError(f"No safetensors found in {model_path}")

        ep_local = config.n_routed_experts // ep_world
        ep_base = ep_rank * ep_local

        # Pair `weight` with `weight_scale_inv` before assigning; they can land
        # in the same shard but nothing guarantees it.
        fp8_pending = {}
        loaded = set()
        n_expert_t = 0

        def commit(mapped):
            model.load_state_dict(mapped, strict=False)
            loaded.update(mapped.keys())

        def route_expert(name, fh):
            """Place one `...mlp.experts.<e>.<proj>.weight.mxfp4_{blocks,scales}`
            into its layer's expert_mxfp4 slot, or skip it.

            Returns True if the name was an expert tensor -- consumed or
            skipped -- so the caller does not also hand it to load_state_dict.
            Skipping is the point: at ep_world=8 seven of every eight experts
            are never read off disk at all, which is what keeps the load to
            46 GB per rank instead of 390.
            """
            for suffix, slot in ((".mxfp4_blocks", 0), (".mxfp4_scales", 1)):
                if not name.endswith(suffix):
                    continue
                base = name[:-len(suffix)]
                parts = base.split(".")          # model layers L mlp experts E proj weight
                if len(parts) < 7 or parts[4] != "experts":
                    return False
                layer_i, glob_e, proj = int(parts[2]), int(parts[5]), parts[6]
                if layer_i >= n_layers:
                    return True
                if not (ep_base <= glob_e < ep_base + ep_local):
                    return True                  # another rank owns it
                store = model.model.layers[layer_i].mlp.expert_mxfp4
                pair = store[glob_e - ep_base].setdefault(proj, [None, None])
                pair[slot] = fh.get_tensor(name).cuda(non_blocking=True)
                return True
            return False

        for fi, f in enumerate(files):
            with safe_open(f, framework="pt", device="cpu") as fh:
                mapped = {}
                for name in fh.keys():
                    # Drop the MTP layer, the DSA indexer, and anything past
                    # --max-layers. `model.layers.N.` is the only place a layer
                    # index appears.
                    if name.startswith("model.layers."):
                        idx = int(name.split(".")[2])
                        if idx >= n_layers:
                            continue
                    if (".indexer." in name
                            or name.endswith("indexers_proj.weight")):
                        continue
                    if mxfp4_experts and route_expert(name, fh):
                        n_expert_t += 1
                        continue
                    if name.endswith(".weight_scale_inv"):
                        fp8_pending.setdefault(name[:-len("_scale_inv")], {})[
                            "scale"] = fh.get_tensor(name)
                        continue
                    tensor = fh.get_tensor(name)
                    if tensor.dtype == torch.float8_e4m3fn:
                        fp8_pending.setdefault(name, {})["weight"] = tensor
                        continue
                    mapped[name] = tensor
                commit(mapped)
            # Dequantise whatever is now complete and release it.
            ready = {k: v for k, v in fp8_pending.items()
                     if "weight" in v and "scale" in v}
            if ready:
                commit({k: dequantize_fp8_blockscale(v["weight"], v["scale"])
                        for k, v in ready.items()})
                for k in ready:
                    del fp8_pending[k]
            if verbose and (fi % 10 == 0 or fi == len(files) - 1):
                print(f"[load] shard {fi + 1}/{len(files)} "
                      f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB "
                      f"resident", flush=True)

        leftover = [k for k, v in fp8_pending.items()
                    if "weight" in v and "scale" not in v]
        if leftover:
            raise RuntimeError(
                f"{len(leftover)} FP8 weights had no weight_scale_inv, "
                f"e.g. {leftover[:3]}")

        if mxfp4_experts:
            # Every local expert must have all three projections, each with
            # both halves of the pair. A silently half-loaded expert would show
            # up much later as garbage text from one rank's slice only.
            n_moe = sum(1 for l in model.model.layers if l.is_moe)
            want = {"gate_proj", "up_proj", "down_proj"}
            for l in model.model.layers:
                if not l.is_moe:
                    continue
                for e, d in enumerate(l.mlp.expert_mxfp4):
                    assert set(d) == want and all(
                        p[0] is not None and p[1] is not None
                        for p in d.values()), (
                        f"layer {l.layer_idx} local expert {e} "
                        f"(global {ep_base + e}) incomplete: "
                        f"{ {k: [x is not None for x in v] for k, v in d.items()} }")
            if verbose:
                print(f"[load] {n_expert_t} MXFP4 expert tensors -> "
                      f"{n_moe} MoE layers x {ep_local} experts, "
                      f"global [{ep_base}, {ep_base + ep_local})", flush=True)

        expected = set(model.state_dict().keys())
        never = sorted(expected - loaded)
        if never:
            print(f"WARNING: {len(never)} parameters were not found in the "
                  f"checkpoint, e.g. {never[:5]}")
        return model

    @torch.inference_mode()
    def forward(self, input_ids, position_embeddings=None, step=None):
        cfg = self.config
        if cfg.has_indexer:
            kv_len = (input_ids.shape[-1] if input_ids.shape[-1] > 1
                      else int(step.item()) + 1)
            if kv_len > cfg.index_topk:
                raise NotImplementedError(
                    f"context {kv_len} exceeds index_topk={cfg.index_topk}; "
                    "the DSA indexer is a Stage-4 item and dense attention is "
                    "only equivalent below that threshold.")
        hidden_states, _ = self.model(input_ids=input_ids,
                                      position_embeddings=position_embeddings,
                                      step=step)
        return self.lm_head(hidden_states[:, -1:, :])
