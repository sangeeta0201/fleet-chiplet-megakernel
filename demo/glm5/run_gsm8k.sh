#!/bin/bash
# Redline/ATOM GSM8K accuracy against the GLM megakernel (GPUs 4-7).
#
# Redline scores GPT-OSS via utils/gsm8k_lm_eval.py talking to an OpenAI
# endpoint (lm-eval gsm8k, 3-shot, chat template, flexible-extract). Fleet has
# no endpoint, so demo.py --gsm8k runs the same protocol in-process: compile
# once, reset the persistent kernel, decode each sample.
#
#   ./run_gsm8k.sh                  # 8 samples, 3-shot, devices 4-7
#   GSM8K_LIMIT=100 ./run_gsm8k.sh  # Redline CI sample count
#
# Token equality is still not a legal gate on this model (correctness_gate.py).
# This scores extracted numeric answers, plus G1 cross-rank identity on those
# answers.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

OUT_DIR="${GLM5_OUTPUT_DIR:-/tmp/glm5_gsm8k}"
mkdir -p "$OUT_DIR"

export MODEL_PATH="${MODEL_PATH:-${GLM_MODEL_PATH:-/mnt/nvme1/GLM-5.2-MXFP4}}"
# 3-shot chat is ~591 tokens. seq=1536/gen=512 compiled and then
# systematically wedged after SCHED_XCD (ENTER, grids up, no FWD_PASS).
# 768 fits prompt+128 new tokens, matches the 512-seq kernel that G2
# just ran live, and the ngram/EOS stop cuts collapse before 128 anyway.
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-768}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
# 3-shot max prompt is 591 tokens. A second in-process mpk() launch
# wedges at SCHED_XCD (idx=0 lives; idx=1 hangs 100% of measured
# attempts), so the default is one sample. Raise GSM8K_LIMIT once
# the persistent-kernel re-launch is fixed.
export GSM8K_LIMIT="${GSM8K_LIMIT:-1}"

# shellcheck source=stall_watchdog.sh
. ./stall_watchdog.sh
# After ENTER a healthy decode at seq=768 is well under this. The
# cold-start wedge prints ENTER then SCHED_XCD and freezes; 180s is
# enough to retry without burning 400s per dead launch.
STALL_SECS="${STALL_SECS:-180}"
RETRIES="${RETRIES:-8}"

LAUNCHER="${LAUNCHER:-./run_mp8_dp_ep_fused.sh}"

echo "=== GLM GSM8K (Redline protocol): limit=$GSM8K_LIMIT fewshot=3 seq=$MAX_SEQ_LENGTH gen=$MAX_NEW_TOKENS"
if [ "${GSM8K_KEEP_BUILD:-0}" != "1" ]; then
  rm -rf permanent_output_dir permanent_output_dir_rank*
fi
export KEEP_BUILD=1

rc=1
for attempt in $(seq 0 "$RETRIES"); do
  export MASTER_PORT=$(( ${MASTER_PORT_BASE:-30010} + attempt ))
  run_with_watchdog "$OUT_DIR/gsm8k_run.log" "$STALL_SECS" \
    "$LAUNCHER" \
      --gsm8k \
      --gsm8k-limit "$GSM8K_LIMIT" \
      --gsm8k-fewshot "${GSM8K_FEWSHOT:-3}" \
      --gsm8k-split "${GSM8K_SPLIT:-test}" \
      --gsm8k-seed "${GSM8K_SEED:-1234}" \
      --gsm8k-out "$OUT_DIR/gsm8k.json"
  rc=$?
  n=$(ls "$OUT_DIR"/gsm8k.json "$OUT_DIR"/gsm8k_rank*.json 2>/dev/null | wc -l)
  if [ "$rc" -eq 0 ] && [ "$n" -gt 0 ]; then
    break
  fi
  if [ "$rc" -eq 124 ]; then
    echo "    STALLED (attempt $((attempt + 1))), retrying"
  else
    echo "    FAIL rc=$rc files=$n (attempt $((attempt + 1))), retrying"
  fi
done

n=$(ls "$OUT_DIR"/gsm8k.json "$OUT_DIR"/gsm8k_rank*.json 2>/dev/null | wc -l)
echo "--- run rc=$rc, $n dump(s) in $OUT_DIR"
if [ "$n" -eq 0 ]; then
  echo "FAILED: no GSM8K dump -- see $OUT_DIR/gsm8k_run.log" >&2
  exit 1
fi

GLM5_OUTPUT_DIR="$OUT_DIR" GLM5_GSM8K_REQUIRE_DUMPS=1 python3 -m pytest -q -s \
  "$(cd ../.. && pwd)/tests/ci-tests/test_glm5_gsm8k.py"
