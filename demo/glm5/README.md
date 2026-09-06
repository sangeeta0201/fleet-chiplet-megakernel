# GLM-5 / GLM-4.7-Flash on the Fleet megakernel (MI350 / MI355, gfx950)

Absorbed-MLA + `noaux_tc`-MoE decode as a single persistent kernel, using the
same chiplet/XCD gang scheduling the GPT-OSS 120B demo uses.

Two checkpoints share one architecture and therefore one task graph:

| | `zai-org/GLM-4.7-Flash` | `zai-org/GLM-5-FP8` |
|---|---|---|
| `model_type` | `glm4_moe_lite` | `glm_moe_dsa` |
| layers | 47 | 78 (+1 MTP, skipped) |
| hidden | 2048 | 6144 |
| q heads | 20 | 64 |
| q_lora / kv_lora | 768 / 512 | 2048 / 512 |
| qk_nope + qk_rope | 192 + 64 | 192 + 64 |
| v_head | 256 | 256 |
| experts | 64, top-4, +1 shared | 256, top-8, +1 shared |
| dense prefix | 1 layer | 3 layers |
| `routed_scaling_factor` | 1.8 | 2.5 |
| weights | bf16 | FP8 E4M3, 128×128 blockscale |
| DSA indexer | — | 32 heads × 128, top-2048 |

GLM-4.7-Flash is the plumbing vehicle; layer-truncated GLM-5 (`--max-layers`)
is the one with the real kernel geometry.

## Running

```bash
# PyTorch reference
HIP_VISIBLE_DEVICES=0 python3 demo.py --model-path /models/GLM-4.7-Flash \
    --max-new-tokens 32 --save-tokens

# Megakernel
HIP_VISIBLE_DEVICES=0 python3 demo.py --model-path /models/GLM-4.7-Flash \
    --max-new-tokens 32 --save-tokens --use-mirage
```

`--save-tokens` writes `outputs/glm5/{torch_output.json,mpk_output.json}` for
`tests/ci-tests/test_glm5_inference_output.py`. The whole comparison is
`tests/ci-tests/run_ci_tests_glm5.sh`, which runs both paths and applies the
tolerant longest-common-block check.

To exercise the real 744B kernel geometry (64 q heads, `q_lora` 2048, 6144
hidden, 256 experts) on one GPU, truncate the layers. `--max-layers 4` is the
smallest truncation that covers both sides of `first_k_dense_replace = 3`.
A config-only checkpoint ships no tokenizer, so borrow one:

```bash
HIP_VISIBLE_DEVICES=0 python3 demo.py --model-path zai-org/GLM-5-FP8 \
    --tokenizer-path zai-org/GLM-4.7-Flash \
    --random-weights --max-layers 4 --use-mirage
```

That shape puts 4 q-head groups x 2 sequence chunks on exactly 8 XCDs.

Useful flags: `--max-layers N` (truncates the Torch reference *and* the task
graph, so the comparison stays meaningful), `--random-weights` (config +
tokenizer only — shapes and kernels without the checkpoint),
`--max-seq-length`, `--profiling`, `--no-rope-interleave`,
`--tokenizer-path`.

Env knobs: `GLM_MLA_NUM_KV_CHUNKS` (sequence split across XCDs),
`GANG_TILE_N`, `GANG_WGM`, `GLM_MODEL_PATH`.

**After editing any `.cuh` or task-registration source, `rm -rf
permanent_output_dir`** — header edits do not invalidate the cached build.

## 744B GLM-5.2 (NP=4, devices 4–7)

Checkpoint: `/mnt/nvme1/GLM-5.2-MXFP4` (same files as
`/home/schowdha/models/GLM-5.2-MXFP4`). Chat thinking is **off** unless
`--enable-thinking`. Exact token match is illegal (`correctness_gate.py`).

Run inside `fleet_v2` as `claudeuser`. Do not widen `HIP_VISIBLE_DEVICES`
off `4,5,6,7`.

```bash
cd demo/glm5

# Keywords + G1 cross-rank. 4 prompts, seq=512, gen=256.
./run_correctness_suite.sh run_mp8_dp_ep_fused.sh mp8ep
python3 compare_tokens.py /tmp/glm5_correctness mp8ep mp8ep

# Teacher-forced PPL (WikiText-2, 128 tokens, [gMASK]<sop>, GLM_LMHEAD_TP=0)
PPL_MODE=1 GLM_LMHEAD_TP=0 ./run_ppl.sh
pytest ../../tests/ci-tests/test_glm5_perplexity.py

# GSM8K, Redline 3-shot flexible-extract. Default LIMIT=1 until a second
# in-process mpk() launch stops wedging at SCHED_XCD.
./run_gsm8k.sh
pytest ../../tests/ci-tests/test_glm5_gsm8k.py

# Latency: ALWAYS 1024 ISL / 1024 OSL. Short-prompt TPOT is not the number.
./run_latency_1k1k.sh
```

### Measured 2026-09-03, NP=4, `HIP_VISIBLE_DEVICES=4,5,6,7`

| Gate | Result |
|---|---|
| `compare_tokens.py` keywords + 4-rank identity | **4/4 PASS** (paris / scatter / prime / stack+queue) |
| G1 cross-rank | **PASS** all 4 prompts |
| G2 coherence | **PASS** p0 distinct 0.497 / top-bigram 4, n=183; p1 0.366 / 8, n=216; p2 0.386 / 12, n=207; p3 0.348 / 7, n=201 |
| PPL 128 WikiText-2 | **PASS** ppl=**19.704**, 4 ranks identical. Ceiling 60. GLM-5 on this slice was ~5.7 |
| GSM8K pytest | **2 passed**, accuracy 0/1 (floor 0). Gold=18, extracted=0.5, `LIMIT=1` |

G2 n < 256 is the n-gram / extra-EOS halt: GLM-5.2 greedy often never emits
`<|endoftext|>`, `<|user|>`, or `<|observation|>`, then loops. The kernel
stops on all three ids, on an AABB 8–32-gram, and when any generated bigram
hits 25.

### Latency, 1024 / 1024 ISL / OSL

Always this shape. `--ignore-eos` + `--prompt-tokens 1024` so OSL is a full
1024 and ISL is an exact WikiText-2 prefix (`[gMASK]<sop>` included). Compiled
`max_seq_length=2048` (DSA identity holds at `kv_len <= 2048`).

| | Prefill avg | Decode avg | Decode min | OSL dumped | G1 | text |
|---|---|---|---|---|---|---|
| NP=4, devices 4–7, bs=1 | *pending 1k/1k run* | | | | | |

Do not quote the short-prompt G2 ~9 ms/iter as the latency number.

## What the megakernel actually computes

### Absorbed MLA

MLA is stored as two low-rank factors, `W_UK` (`kv_lora → qk_nope`) and `W_UV`
(`kv_lora → v_head`), both slices of `kv_b_proj`. Rather than materialise
per-head K and V, both fold into their neighbours:

```
q_b_absorbed[h] = [ W_UK[h]ᵀ @ q_b_nope[h] ; q_b_rope[h] ]   [576, q_lora]
o_absorbed[:, h] = o_proj[:, h] @ W_UV[h]                    [hidden, 512]
```

What is left is MQA with asymmetric head dims: one cached row per token,

```
kv_row = [ c_kv (512) | k_rope (64) ]      576 elements, 1.15 KB
```

QK reduces over all 576; PV accumulates over the leading 512 dims of the *same*
row. The attention scale stays `qk_head_dim ** -0.5` (256) — the unabsorbed
head dim, not the 576-wide reduction. The cached latent is the **post-
`kv_a_layernorm`** `c_kv`: `kv_b_proj` is linear in the normalised latent, which
is exactly what makes the absorption exact.

Against unabsorbed MLA this is 64 × 256 × 2 → 576 elements per token per layer,
a 57× smaller KV cache, and it removes a per-head GEMM from the decode path.

### Chiplet mapping

GPT-OSS's gang attention maps `kv_head == xcd_id`. MLA has a single shared
latent head, so that mapping does not carry over. Instead the work item is
`(request, q_head_group, kv_chunk)` — a q_head_group being one MFMA M tile of
16 query heads — and the demo picks `num_kv_chunks` so
`num_q_groups × num_kv_chunks == 8`, one item per XCD (GLM-5: 4 × 2;
GLM-4.7-Flash padded to 32 heads: 2 × 4). With more than one chunk the decode
writes float partials plus natural-log LSE in the layout
`merge_splitkv_ck_fmha` already consumes, with the q_head_group standing in for
`kv_head`.

The XCD index reaches the kernel through `tile_idx`, which the runtime forms as
`n_tile_start + t` with `n_tile_start = bid.x * tiles_per_xcd`. That offset is
only applied to task types listed explicitly in `src/kernel/runtime.cc`; a gang
task missing from the list silently gets `n_tile_start = 0`, so every XCD
recomputes work item 0 and every other item keeps whatever the scratch buffer
last held. `TASK_GANG_MLA_DECODE_MI300` is therefore registered in all four of
the hardcoded task-type lists: the `n_tile_count` metadata switch, the
`n_tile_start` switch, the `_execute_task` skip list, and the
`_execute_gang_task` inclusion list.

### Router

`noaux_tc`: sigmoid over the router logits; `e_score_correction_bias` shifts
*which* experts win but not the weight they carry (weights are gathered from
the unbiased sigmoid); `norm_topk_prob` renormalises with a `+1e-20`
denominator; then `× routed_scaling_factor`. `n_group == topk_group == 1` on
both checkpoints, so group-limited routing is a no-op.

### Layer schedule

```
gang_rmsnorm_linear_bias   input_layernorm + [q_a_proj | kv_a_proj_with_mqa]
gang_rmsnorm_linear_bias   q_a_layernorm (leading q_lora_pad cols) + q_b_absorbed
mla_kv_cache_update        kv_a_layernorm + partial interleaved RoPE,
                           latent row -> paged cache, roped Q -> workspace
gang_mla_decode            absorbed MLA, split over (q_group, kv_chunk)
[paged_attention_ck_fmha_merge]                        (num_kv_chunks > 1)
gang_linear_with_residual  o_absorbed + residual        -> attn_proj_out

dense layers (i < first_k_dense_replace):
  gang_rmsnorm_linear_bias post_attention_layernorm + gate_up
  silu_mul
  gang_linear_with_residual down + residual

MoE layers:
  rmsnorm                  post_attention_layernorm
  linear                   router
  moe_topk_sigmoid_bias    noaux_tc selection (+ shared expert in slot k)
  gang_moe_w13 / moe_silu_mul / gang_moe_w2
  moe_mul_sum_add          weighted expert sum + attn_proj_out

tail: gang_rmsnorm_linear_bias (final norm + LM head), argmax_partial/reduce
```

Two ordering constraints come from the runtime rather than the math. The task
graph is a **linear chain**, and every op must consume at least one tensor the
op immediately before it produced (`runtime.cc:572`) — weights don't count,
they are graph inputs. `gang_rmsnorm_linear_bias` makes this sharper than it
sounds: it declares its `norm_output` scratch as an *input*, so only the linear
`output` counts as produced. A second GEMM reading that scratch shares nothing
with the op that wrote it, which means **two sibling projections off one fused
RMSNorm are not expressible**.

So `q_a_proj` and `kv_a_proj_with_mqa` are concatenated into a single
`[q_lora_pad | kv_lora + qk_rope]` GEMM fused with `input_layernorm`. The
numerical objection — `q_a_layernorm` must normalise only the `q_lora` columns,
and a fused output would drag the latent into that RMS denominator — is handled
by a `NORM_SPAN` template parameter that bounds the sum of squares to the
leading `q_lora_pad` columns (`if constexpr`, so the gpt-oss path is
unchanged). `mla_kv_cache_update` then takes a `KV_INPUT_OFFSET` to find the
latent slice inside that same row.

The same rule rules out running the shared expert as its own GEMM pair — that
makes the MoE block a diamond (shared and routed branches both hang off the
post-attention norm and rejoin at the weighted sum) with no valid
linearisation. Instead the shared expert rides along as **one extra routed
expert**: it is stacked at id `num_experts`, the router writes it into slot
`topk` of every token with weight 1, and `gang_moe_w13 / moe_silu_mul /
gang_moe_w2 / moe_mul_sum_add` process it with the routed ones. This is
AITER's `shared_expert_id` trick (`aiter/fused_moe.py`), and it is exact here
because GLM's shared expert has precisely a routed expert's shape
(`moe_intermediate_size`). It also removes three tasks per layer. The residual
that `moe_mul_sum_add` adds is then simply `attn_proj_out`.

## Padding

The gang GEMM path has two hard divisibility rules, both from
`linear_kernel_ck`'s tile shape: `NPerBlock = 64` with an 8-way XCD split of
the output (so **output size % 512**), and `KPerBlock = 256` at batch ≤ 16 (so
**reduction size % 256**). `gang_mla_decode` adds **q heads % 16**.

| tensor | GLM-5 | GLM-4.7-Flash |
|---|---|---|
| q heads | 64 ✓ | 20 → **32** |
| `q_a_proj` out | 2048 ✓ | 768 → **1024** |
| `kv_a_proj` out | 576 → **1024** | 576 → **1024** |
| fused `[q_a \| latent]` | **3072** | **2048** |
| `q_b_absorbed` out | 64·576 = 36864 ✓ | 32·576 = 18432 ✓ |
| `o_absorbed` out | 6144 ✓ | 2048 ✓ |
| vocab | 154880 → **155136** | idem |

Padded q heads get zero `q_b_absorbed` rows and zero `o_absorbed` columns: they
attend uniformly and contribute nothing to the output. Padded projection
outputs are exactly zero (the padded weight rows are zero), so a padded RMSNorm
reduction is safe and `actual_hidden_dim` keeps the denominator right.

## `silu_mul` layout

`silu_mul_task_impl` reads its multiplicand at `input_ptr + OUTPUT_SIZE`, where
`OUTPUT_SIZE` is the *per-block* width after `grid.x` partitioning — so a flat
`[gate(I) | up(I)]` weight is only read correctly at `grid.x == 1`. The dense
gate/up weights of the dense prefix layers are therefore laid out as 8
consecutive `[gate_chunk; up_chunk]` slabs (`interleave_gate_up`, the same
grouping `mpk.shuffle_tensors` produces), keeping one `silu_mul` task per XCD.
`moe_silu_mul_layer` is unaffected — its input is 3-D and dim 2 is
unpartitioned — so the expert stacks (shared expert included) keep the plain
concat layout.

## Known deviations

* **`rms_norm_eps`.** `task_register.cc` hardcodes `1e-6f` at all its norm
  sites (house convention); both GLM configs specify `1e-5`.
* **No DSA indexer.** GLM-5's indexer selects the top `index_topk = 2048`
  positions; below 2048 tokens of context that selection is the identity, so at
  Stage-1 sequence lengths dense attention *is* sparse attention. The Torch
  reference raises rather than silently approximating above the threshold.
* **FP8 weights are dequantised on load.** Stage 2 consumes the 128×128
  blockscale form directly.
* **`world_size == 1`.** ROCm in-megakernel collectives are Stage 3.
* **MTP layer skipped** (`num_nextn_predict_layers = 1`).

## Files

* `demo.py` — task graph + Torch reference + generation loop
* `models/modeling_glm_moe_dsa.py` — config, absorbed-MLA reference,
  `noaux_tc` router, FP8 blockscale dequantisation, checkpoint loader
