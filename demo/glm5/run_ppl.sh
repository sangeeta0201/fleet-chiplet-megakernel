#!/bin/bash
# GLM-5 perplexity: score a WikiText-2 slice with the megakernel, then gate on
# it (tests/ci-tests/test_glm5_perplexity.py).
#
# This is the GLM-5 counterpart of
# tests/ci-tests/run_ci_tests_gpt_oss_perplexity.sh, and it exists because
# token equality is not a legal gate on this model: correctness_gate.py
# measured two attractor continuations from ONE build on ONE prompt, differing
# at token 0, so "arm B emits arm A's tokens" fails about half the time
# comparing a build to itself. Perplexity is a continuous number taken under
# teacher forcing, which is what removes the argmax cascade.
#
# PPL_MODE=1 loads the corpus as one long prompt and runs prefill only. The
# megakernel does not overwrite tokens[] inside the prompt, so every position
# conditions on the reference prefix -- prefill IS teacher forcing. The LM head
# writes one logit row per iteration, hence --max-num-batched-tokens 1.
#
# TWO THINGS THIS RUN DOES NOT DO, both deliberate:
#
#   * No Torch reference arm. GLM-5 744B at 4 bits is ~372 GB against 288 GB of
#     HBM on one MI350, so the single-GPU Torch leg that the GPT-OSS script
#     compares against does not fit. The gate is therefore an absolute band
#     plus cross-rank agreement. Drop a torch_ppl.json into the output dir and
#     the test picks up the ratio arm on its own.
#
#   * GLM_LMHEAD_TP=0, which is NOT the shipping decode config. With the vocab
#     sharded four ways no single rank holds a normalizable logit row, and a
#     cross-rank logsumexp is not implemented. Untiled, every rank scores the
#     full vocabulary independently, which is what makes the cross-rank
#     agreement check meaningful. What this gate covers is therefore the 78
#     layers, the attention, the MoE and the EP fold -- everything except the
#     LM head's vocab tiling.
#
# Env (override as needed):
#   MODEL_PATH      GLM-5 MXFP4 checkpoint (defaults to the pinned local dir)
#   PPL_MAX_TOKENS  corpus tokens to score (default 512)
#   NP              ranks (default 4, devices 4-7, per standing instruction)
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

# Pin the model, exactly as run_correctness_suite.sh does. env_common.sh
# defaults to GLM-4.7-Flash, whose MLA shape does not satisfy the absorbed
# o_proj assert, so an unpinned run dies on every rank before it scores a
# single position -- i.e. the gate silently does not run.
export MODEL_PATH="${MODEL_PATH:-/home/claudeuser/models/glm5-mxfp4}"

export PPL_MODE=1
# Not optional: see the header. demo.py raises rather than silently scoring an
# unnormalizable row.
export GLM_LMHEAD_TP=0

PPL_MAX_TOKENS="${PPL_MAX_TOKENS:-512}"
OUT_DIR="${GLM5_OUTPUT_DIR:-$(cd ../.. && pwd)/outputs/glm5}"
mkdir -p "$OUT_DIR"
# Stale dumps from an earlier slice would be picked up by the test's glob and
# fail the slice-identity check with a confusing message. Clear them.
rm -f "$OUT_DIR"/mpk_ppl*.json

# The sink resizes the megakernel and PPL_MODE changes the emitted LM head
# argument list, so the decode build cannot be reused.
unset KEEP_BUILD

# Bound the run on LOG SILENCE, not wall clock: the build is minutes of quiet
# hipcc and the megakernel has an intermittent cold-start wedge that pins all
# four GPUs indefinitely. Same mechanism bench_repeat.sh uses.
# shellcheck source=stall_watchdog.sh
. ./stall_watchdog.sh
STALL_SECS="${STALL_SECS:-300}"

echo "=== GLM-5 perplexity: $PPL_MAX_TOKENS tokens of ${PPL_CORPUS:-wikitext2}"
echo "    MODEL_PATH=$MODEL_PATH  OUT_DIR=$OUT_DIR  NP=${NP:-4}"

# Retry the cold-start wedge, exactly as run_correctness_suite.sh does. It is
# not specific to this script -- the suite hit it on prompt 0 of a clean run --
# and it gets likelier with sequence length: 128 tokens has never wedged here,
# 256 wedged once in two attempts, 512 twice in two. Without a retry a 512-token
# gate is a coin flip, and a wedge reads as "no dump" rather than "try again".
#
# Fresh rendezvous port per ATTEMPT: a killed run leaves the listener bound or
# in TIME_WAIT, and the retry would then die with EADDRINUSE before it ever
# reaches the kernel.
RETRIES="${RETRIES:-2}"
for attempt in $(seq 0 "$RETRIES"); do
  export MASTER_PORT=$(( ${MASTER_PORT_BASE:-29990} + attempt ))
  run_with_watchdog "$OUT_DIR/ppl_run.log" "$STALL_SECS" \
    ./run_mp8_dp_ep_fused.sh \
      --max-num-batched-tokens 1 \
      --ppl-corpus "${PPL_CORPUS:-wikitext2}" \
      --ppl-max-tokens "$PPL_MAX_TOKENS" \
      --ppl-out "$OUT_DIR/mpk_ppl.json"
  rc=$?
  [ "$rc" -ne 124 ] && break
  echo "--- STALLED at launch (attempt $((attempt + 1))), retrying"
done

n=$(ls "$OUT_DIR"/mpk_ppl*.json 2>/dev/null | wc -l)
echo "--- run rc=$rc, $n dump(s) in $OUT_DIR"
if [ "$n" -eq 0 ]; then
  echo "FAILED: no perplexity dump -- see $OUT_DIR/ppl_run.log" >&2
  exit 1
fi

GLM5_OUTPUT_DIR="$OUT_DIR" python3 -m pytest -q -s \
  "$(cd ../.. && pwd)/tests/ci-tests/test_glm5_perplexity.py"
