#!/bin/bash
# Multi-prompt correctness sweep for GLM, mirroring
# demo/gpt_oss/run_correctness_suite.sh.
#
# Runs a prompt set through a given launcher and dumps each prompt's generated
# token_ids to JSON; compare_tokens.py then diffs a candidate run against the
# reference.
#
#   USE_MIRAGE=0 ./run_correctness_suite.sh run_1gpu.sh            torch
#   ./run_correctness_suite.sh run_1gpu.sh                         1gpu
#   ./run_correctness_suite.sh run_mp8_dp_ep_fused.sh              mp8ep
#   python3 compare_tokens.py /tmp/glm5_correctness torch mp8ep
#
# Every prompt costs a full megakernel recompile, so this is minutes per
# prompt. Results are written incrementally -- a sweep that dies at prompt 3
# still leaves the first three comparable.
#
# Under a multi-rank launcher demo.py suffixes the dump per rank, so prompt i
# yields <tag>_p<i>_rank<r>.json for every r. That is deliberate: with DP
# attention every rank decodes the same prompt and owes the same answer, so
# cross-rank disagreement is a direct read on the EP fold.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

LAUNCHER="${1:?usage: run_correctness_suite.sh <launcher.sh> <tag>}"
TAG="${2:?usage: run_correctness_suite.sh <launcher.sh> <tag>}"

OUT_DIR="${OUT_DIR:-/tmp/glm5_correctness}"
mkdir -p "$OUT_DIR"

# Pin the model, exactly as bench_repeat.sh does. env_common.sh defaults to
# GLM-4.7-Flash, whose MLA shape does not satisfy the absorbed-o_proj assert,
# so an unpinned sweep dies on every rank at
# "absorbed o_proj reduces over num_q_heads * kv_lora_rank = 16384, got 10240"
# before it generates a single token -- i.e. the gate silently does not run.
export MODEL_PATH="${MODEL_PATH:-/home/claudeuser/models/glm5-mxfp4}"

# Long enough that a subtly wrong reduction shows as divergence rather than as
# a lucky matching prefix, and short enough to stay inside the page budget.
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
# demo.py truncates the dump at MAX_SAVE_TOKENS; leave it at its default of 100
# and a divergence past token 100 never reaches the JSON.
export MAX_SAVE_TOKENS="${MAX_SAVE_TOKENS:-$((MAX_NEW_TOKENS + 8))}"

PROMPTS=(
  "The capital of France is"
  "Write a short paragraph explaining why the sky appears blue."
  "List the first ten prime numbers and explain what makes a number prime."
  "Explain the difference between a stack and a queue in computer science."
)

# The megakernel does not depend on the prompt, so build once and reuse:
# 4 prompts x a full rebuild is ~an hour, against ~3 min per run on a warm
# build. Clear it here rather than trusting the caller's KEEP_BUILD, so the
# first run of the sweep is always against freshly generated code.
rm -rf permanent_output_dir permanent_output_dir_rank*
export KEEP_BUILD=1

# Bound each run. The megakernel has an intermittent cold-start wedge that
# pins all four GPUs indefinitely; without a bound one bad prompt eats the
# sweep and the remaining prompts never run.
#
# A wall-clock bound is the wrong instrument for it, and this sweep is where
# that showed: RUN_TIMEOUT=1200 means a wedged prompt burns twenty minutes
# before the next one starts, so in practice the wedge got killed by hand and
# the kill took the FOLLOWING prompt's mpirun with it -- two prompts lost to
# one wedge, twice now. The watchdog trips on LOG SILENCE instead, which
# separates "quiet because hipcc is running" from "quiet because hung", and
# arms only after every rank prints ENTER. Same mechanism bench_repeat.sh
# uses. See demo/glm5/stall_watchdog.sh for the full rationale.
# shellcheck source=stall_watchdog.sh
. ./stall_watchdog.sh
STALL_SECS="${STALL_SECS:-120}"
RETRIES="${RETRIES:-2}"

for i in "${!PROMPTS[@]}"; do
  p="${PROMPTS[$i]}"
  dst="$OUT_DIR/${TAG}_p${i}.json"
  log="$OUT_DIR/${TAG}_p${i}.log"
  # Fresh rendezvous port per prompt: a killed run leaves the listener bound
  # or in TIME_WAIT, and the next launch dies with EADDRINUSE before it builds.
  echo "=== [$TAG] prompt $i: $p"
  for attempt in $(seq 0 "$RETRIES"); do
    # Fresh rendezvous port per ATTEMPT, not just per prompt: a killed run
    # leaves the listener bound or in TIME_WAIT, and the retry would then die
    # with EADDRINUSE before it ever reaches the kernel.
    export MASTER_PORT=$(( ${MASTER_PORT_BASE:-29950} + i * 8 + attempt ))
    run_with_watchdog "$log" "$STALL_SECS" \
      ./"$LAUNCHER" --prompt "$p" --save-tokens "$dst"
    rc=$?
    [ "$rc" -ne 124 ] && break
    echo "    STALLED (attempt $((attempt + 1))), retrying"
  done
  # The multi-rank case writes <stem>_rank<r>.json and never <stem>.json, so
  # test for either shape rather than for the literal path handed to demo.py.
  n=$(ls "$OUT_DIR/${TAG}_p${i}".json "$OUT_DIR/${TAG}_p${i}"_rank*.json \
        2>/dev/null | wc -l)
  if [ "$rc" -ne 0 ] || [ "$n" -eq 0 ]; then
    echo "    FAILED (rc=$rc, files=$n) -- see $log"
  else
    echo "    ok -> $n file(s)"
  fi
done
