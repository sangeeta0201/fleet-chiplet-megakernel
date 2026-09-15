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

| when | Prefill avg | Decode avg | Decode min | OSL dumped | G1 | text |
|---|---|---|---|---|---|---|
| 09-05 05:02 | 21.6 | 13.970 | 12.731 | 1024 | PASS | Boulter bio |
| 09-06 05:01 | 21.6 | 13.968 | 12.734 | 1024 | PASS | Boulter bio, on-topic at tail |
| 09-06 06:44 | 185.9 | **13.979** | 12.642 | 1024 | PASS | Boulter bio, ``` loop from ~72% |
| 09-06 18:42 | — | 14.030 | 12.788 | 1024 | PASS | Boulter bio |
| 09-06 18:45 | — | 13.999 | 12.722 | 1024 | PASS | Boulter bio |

**14.0 ms/token, n=5** (avg 13.968–14.030, spread 0.062; min 12.642–12.788).
NP=4, devices 4–7, bs=1, no MTP, GLM-5.2-MXFP4. Flat across two days and
every A/B on this branch — none of those changed the shipping build.

### External reference: vLLM + HIP graphs, same shape and same 4 GPUs. 09-06.

| stack | quant | TPOT ms/token | decomposition |
|---|---|---|---|
| **mpk megakernel** | **MXFP4** | **14.0** (n=5) | NP=4, ATTN_DP + MOE_EP |
| **ATOM** | **MXFP4** | **10.55** (n=3+10) | TP=4 |
| ATOM | FP8 | 11.75 (n=3+10) | TP=4 |
| vLLM, HIP graphs ON | FP8 | 20.44 (n=3, 20.43–20.53) | TP=4 |
| vLLM, `--enforce-eager` | FP8 | 103.63 (n=3) | TP=4 |
| *ATOM + MTP-3 (speculative)* | *MXFP4* | *3.77* | *TP=4, accept 0.84* |

**mpk IS NOT AT THE FLOOR. ATOM beats it by 1.33x at identical weights on the
same four GPUs.** Retract the earlier "14.0 ms is the floor for this
decomposition" — that conclusion was drawn from levers *internal* to this
decomposition and does not survive an external engine on the same hardware.
ATOM: `rocm/atom-dev:latest`, `amd/GLM-5.2-MXFP4`, devices 4-7, graphs on
(`CUDAGraphMode.FULL`), no speculation (`speculative_config: None`), no DP-attn,
no EP. Decode attention is a hand-written AITER asm kernel
(`mla_a8w8_qh16_qseqlen1_gqaratio16_ps`). Four independent TPOT derivations per
arm agree to 0.03 ms.

#### Where the 3.45 ms goes — per-phase, us/layer, 78 layers. 09-08.

The gap is **not arithmetic**. mpk beats ATOM on four of twelve phases. It loses
on synchronisation and per-layer control flow that ATOM simply does not have.

| ATOM | mpk | Δ | phase |
|---:|---:|---:|---|
| 0.00 | 33.93 | **+33.93** | attention→o_proj rendezvous (32.50 measured spin) |
| 2.71 | 26.02 | **+23.30** | layer-boundary control flow + dense layers + lm_head |
| 5.54 | 13.33 | +7.80 | MoE W2 + `w13_barrier` wait (W2 tiles alone: 5.91 vs 5.54) |
| 7.52 | 14.49 | +6.97 | MLA decode core (incl. barrier wait) |
| 10.11 | 13.29 | +3.19 | collective (post-MoE all-reduce / EP fold) |
| 8.82 | 11.44 | +2.61 | MoE dispatch — *uncertain, mpk side mostly poll* |
| 11.26 | 12.59 | +1.32 | qkv_a + q_a/kv_a norm + quant |
| 19.27 | 19.52 | +0.25 | MLA split-KV merge |
| 10.12 | 5.86 | **−4.25** | MoE W13 + SwiGLU — *mpk wins on EP* |
| 13.72 | 8.91 | **−4.81** | q_b / W_UK + rope + KV write — *mpk wins* |
| 7.28 | 0.00 | **−7.28** | DSA indexer — *mpk has none* |
| 39.36 | 20.11 | **−19.25** | W_UV+o_proj+AR+RMSNorm+router — *mpk wins, coarse map* |
| **135.71** | **179.49** | **+43.78** | = 3.415 ms/token vs the 3.45 target |

**1. The rendezvous — BUT SEE THE STATISTIC WARNING BELOW.** o_proj contracts
the full 16384-wide v row, so it needs every XCD's W_UV output; mpk spends
**32.50 us/layer** spinning on the `attn_release` flag. ATOM expresses the
identical dependency by putting W_UV and o_proj on one stream and letting the
command processor order them — ~3.6 us, already inside its kernel time.

> **WRONG STATISTIC. Measured null 09-08, see `OPROJ_KSPLIT_NULL.md`.**
> The 33.93 us/layer above is **mean-worker occupancy**. The wall responds to
> **max arrival**, which `price_attention_shard.py` already had at **2.102
> us/layer** — a ~16x difference. **128 of 232 workers arrive within 1 us of the
> max**: the barrier's top is a *plateau*, not a straggler. The 104 early
> workers do burn ~32 us spinning, but releasing them cannot make the plateau
> arrive sooner — they just spin at `hier_barrier` instead. The spin RELOCATES.
> ATOM reads 0.00 us because it has no idle workers to count, not because it
> removed critical path.
>
> Priced by ceiling probe (delete the waits, measure the upper bound, build only
> if it clears the floor): ISA gate passed (`s_sleep` 34→32 for one wait, 34→30
> for both; flag-off md5-identical to pristine), latency **−0.169 ms against a
> 0.26 ms noise floor**, and that arm emits token garbage — so it is an upper
> bound a correct K-split cannot beat. Prediction was pre-registered before the
> run and held. `MAKESPAN_RULE.md` is now **4-for-4**: deleting a rendezvous
> WITHOUT deleting the work in front of it is regime A, not regime B.

**2. ATOM's per-layer control flow is free at runtime.** `cudagraph_mode=FULL`
bakes all **1614 kernel launches** into one replayable graph with pre-resolved
pointers — 58.4 CUDA API calls per step, ~1 host call per 28 kernels. mpk pays
13.00 us/layer of TaskDesc pointer refresh + block join. (The other 13.02 is an
honest unattributed residual: the stamps cover 76 MoE epochs, so dense prologue
layers, lm_head and sampling fall outside.)

**3. ATOM fuses the collective five ways.** `allreduce_fusion_kernel_1stage`
does one-stage all-reduce + residual add + RMSNorm + fp8 activation quant +
bf16 mirror in a single kernel. That is why it has **no separate input-RMSNorm
kernel at all**.

**DENSE-VS-SPARSE: REFUTED, SIGN REVERSED.** The obvious hypothesis was that
ATOM runs sparse DSA and mpk runs dense. Wrong. `index_topk=2048` and kv only
sweeps 1024→2048 here, so top-2048 selects *every* position — the selection is
the identity on both sides. ATOM runs the real indexer on the 21 `full` layers
and **pays 7.28 us/layer for sparsity it cannot use**. mpk is ahead here. At
kv > 2048 the sign flips hard, but not at 1k/1k.

Also corrected: `AITER_QUICK_REDUCE_QUANTIZATION=INT4` is **inert** at
concurrency 1 — its dispatch table needs ≥8 MB and a bs=1 message is 12 KB.

**DO NOT read this table as a list of prizes.** It is a *mean-worker*
decomposition: "phase X costs Y" means the mean worker spends Y there, not that
deleting X saves Y. `MAKESPAN_RULE.md` is 4-for-4 predicting nulls on exactly
that inference — the o_proj K-split (09-08) was the fourth, and it was proposed
off *this table*.

**The rule, stated as a statistic:** for any barrier-bounded phase, the wall
responds to **max arrival**, never to mean-worker occupancy. On `attn_release`
those differ 16x (2.102 vs 33.93 us/layer). Before quoting any row here as a
lever, get its max-arrival number from `price_xcd_narrowing.py` /
`price_attention_shard.py`. The two phases that look largest in this table —
the rendezvous and the layer boundary — have max-arrival ceilings of **0.000 ms**
and **0.051 ms** respectively, both already measured, both under the floor.

What remains genuinely actionable is the *structural* difference — ATOM has no
idle-worker population to release because the hardware command processor orders
its dependencies — not shaving, moving, or deleting any single barrier of ours.

Two statistics caveats: mpk's 14.0 is a per-iteration *minimum*, ATOM's 10.55 a
streaming *mean*, which mildly favours mpk, so the true gap is if anything a
little wider. Instrumented mpk ran 14.718 (+5.1%); the *split* is the result,
not the absolutes.

**Format was never the confounder.** FP8 -> MXFP4 buys ATOM only 1.11x
(11.75 -> 10.55), so it cannot explain the 20.44 -> 14.0 vLLM/mpk gap. That gap
was engine quality, and ATOM beats both. The whole `GLM_MOE_WIDEN_MXFP8`
exercise below was aimed at a confound that turns out to be second-order.

1024 ISL / 1024 OSL, concurrency 1, `HIP_VISIBLE_DEVICES=4,5,6,7`, both arms
coherent text. vLLM `0.9.2rc2.dev13074+g5887cf8ac` from
`rocm/vllm-dev:nightly_cdna4`, run in a **separate container** so `fleet_v2`'s
torch/vllm were untouched. Backend `ROCM_AITER_MLA_SPARSE`, capture mode
`FULL_AND_PIECEWISE` — the FULL capture at size 1 means the whole batch-1 decode
step replays as one graph. TPOT derived by subtraction,
`(T[max_tokens=1024] - T[max_tokens=1]) / 1023`, so prefill cancels.

**Format-matched arm is BLOCKED BY VRAM at this shape. 09-08.**
`GLM_MOE_WIDEN_MXFP8=all` (lossless MXFP4→MXFP8, identical tokens) is the way to
close the format gap from our side, and it works at short prompt
(`world_size_fit.py`: 11.418 → 12.550, **+1.132 ms**, n=3 each). At 1024/1024 it
**reproducibly wedges**: kernel enqueues (no error), GPUs pin at 100%, and it
never returns. 2/2 attempts at NP=4, and again at NP=8.

**It is NOT memory** — that was the first hypothesis and it is wrong. NP=4 made
it look like memory (these GPUs are **252.0 GB total**, MXFP4 at 1024/1024
already sits at 242 GB, and widen pushes to **251.2 GB = 99.6% full**). But NP=8
runs the same widen at **186 GB of 252 GB, 66 GB of headroom**, and wedges
identically. So the widened path has a sequence-length-dependent hang: it
completes at short prompt (n=3) and never returns at 1024/1024, at both world
sizes. That is a real bug, independent of the format question, and it is what
blocks the format-matched measurement.

So the format-corrected estimate is **14.0 + 1.13 ≈ 15.1 ms**, which puts mpk at
**~1.35×** vs vLLM's FP8 20.44 rather than 1.46×. That is a projection carrying
a delta measured at a *different shape* — mark it as such, do not table it as
measured.

**Read the 1.46× with care: the arms are not the same numeric format.** mpk runs
MXFP4 (399 GB checkpoint); vLLM ran FP8 (704 GiB). At bs=1 decode, weight bytes
are the first-order cost, so part of that 1.46× is the format, not kernel
quality. A pure kernel comparison needs vLLM on MXFP4 or mpk on FP8; neither
exists yet. Do not quote 1.46× as a kernel-quality result.

What *is* format-independent: **graphs remove 83.2 ms/token (5.07×)** from the
same vLLM build. That prices per-kernel launch+dispatch overhead across 76
layers, and it is the overhead a persistent megakernel deletes by construction.
Both mpk and graphs-on vLLM have already removed it, so the 20.44 → 14.0 gap is
*not* launch overhead — it is schedule quality plus the format difference.

### MTP speculative decode is a REGRESSION on GLM-5.2. Measured 09-06.

`MPK_SPEC_DECODE=1 --mtp 1 --max-num-batched-tokens 2`, ISL 1024, seq 2048:

```
[SPEC] proposed=794 accepted=228 accept_rate=0.287 committed=1023
       tokens_per_iter=1.288 decode_ms_per_iter=18.765 ms_per_token=14.565
```

**14.565 vs 14.0 = +4%.** Break-even needs `tokens_per_iter >= 18.765/14.0 =
1.34`; the draft delivers 1.288, just under. Do not re-derive this from the
GLM-5 744B figure (accept 0.861, 8.884 ms/token): that was a *different model*
on a short repetitive prompt. On GLM-5.2 against a WikiText prefix the same
draft head accepts **0.287**, and one extra row costs +4.8 ms/iter.

**RETRACTED IN PART, 09-08: reason 1 below is an OUR-BUG, not a model fact.**
ATOM runs MTP on this same GLM-5.2 with `--num-speculative-tokens 3 --method
mtp` and measures **0.84 acceptance, 3.52 accept length, 3.77 ms/token**. The
draft head is fine; 0.287 is our implementation. Speculation is therefore the
largest OPEN lever on this model, not a closed one. Reason 2 (the prefill
micro-chunking) still stands and is still ours to fix.

**Our extra row also costs ~5x what ATOM's does — a SECOND, independent bug.**
ATOM: 10.55 -> 13.3 ms/iter for **3** extra rows = +0.92 ms/row. mpk: 14.0 ->
18.765 for **1** extra row = +4.77 ms/row. At bs=1 an extra row should be nearly
free (weight streaming dominates and is shared), which is why ATOM's 4-row
iteration is only 26% over its 1-row base. Fixing acceptance alone would still
lose; the row cost has to come down too. Verified same model both sides:
`amd/GLM-5.2-MXFP4`, `GlmMoeDsaForCausalLM`, 78 layers, 256 experts,
index_topk 2048, 1 nextn layer.

Two reasons this path looked dead:

1. Acceptance 0.287 is far below the ~0.34 break-even ratio. Width-2 makes it
   worse, not better — it adds a third row's cost against `0.287^2` odds.
   **← refuted: ATOM gets 0.84 on the same weights.**
2. `--max-num-batched-tokens 2` is *required* by the harness and caps the
   prefill chunk at 2 tokens, so a 1024-token prompt prefills in **512
   iterations at 542 ms each = 277 s** (baseline prefill avg is 21.6 ms). That
   is the real cause of the two earlier "wedges" at OSL=1024 — the 400 s
   `STALL_SECS` was firing during micro-chunked prefill, not in decode.

Speculation is the only lever whose shape transfers 1:1 past the ten barriers
(`MAKESPAN_RULE.md`), and ATOM's 0.84 acceptance proves it is available on this
model. Fixing our accept/reject path is now the highest-value work on the
branch — ATOM's MTP row is **3.77 ms**, against our 14.0.

The old closing line here — "14.0 ms is the floor for this decomposition" — is
**withdrawn**. It was inferred from internal levers all capped at ~0.1 ms, but
an external engine reaches 10.55 ms on the same weights and GPUs, so the cap was
a property of our search, not of the hardware.

A sixth run (09-06 06:37) printed avg 39.306 from one 25.8 s hitch at decode
token ~792. That is a liveness spike, not TPOT; excluding it lands on 14.0
with the rest. Prefill avg varies with chunking, not with decode.

Do not quote the short-prompt G2 ~9 ms/iter as the latency number. It is the
same kernel at a ~128-token KV, so it is ~4 ms lower for a reason that has
nothing to do with any optimization; quoting it next to a 1k/1k number makes
a flat wall look like a regression. If a probe must run short-prompt for
turnaround, score it as a paired delta only and say so at the call site.

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
