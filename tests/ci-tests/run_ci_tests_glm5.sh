#!/usr/bin/env bash
# GLM-5 / GLM-4.7-Flash bs=1 end-to-end correctness: run the Torch reference
# and the Mirage (MPK) path, dump their generated tokens, then compare with the
# vLLM-style tolerant check (tests/ci-tests/test_glm5_inference_output.py).
#
# Env (override as needed):
#   MODEL_PATH            local GLM dir or HF repo id (default zai-org/GLM-4.7-Flash)
#   HIP_VISIBLE_DEVICES   target GPU (default 0)
#   GLM5_PROMPT           prompt (default "The capital of France is")
#   GLM5_MAX_SEQ_LEN      sequence length (default 512)
#   GLM5_MAX_LAYERS       truncate to N layers (default: all; use for GLM-5 744B)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MIRAGE_HOME="${MIRAGE_HOME:-$ROOT}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
MODEL_PATH="${MODEL_PATH:-${GLM_MODEL_PATH:-zai-org/GLM-4.7-Flash}}"
PROMPT="${GLM5_PROMPT:-The capital of France is}"
MAX_SEQ_LEN="${GLM5_MAX_SEQ_LEN:-512}"

DEMO="$ROOT/demo/glm5/demo.py"
COMMON=(--model-path "$MODEL_PATH" --prompt "$PROMPT" --max-seq-length "$MAX_SEQ_LEN" --save-tokens)
if [[ -n "${GLM5_MAX_LAYERS:-}" ]]; then
  COMMON+=(--max-layers "$GLM5_MAX_LAYERS")
fi

echo "MIRAGE_HOME=$MIRAGE_HOME  HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES  MODEL_PATH=$MODEL_PATH"
echo "Running Torch reference..."
python3 "$DEMO" "${COMMON[@]}"
echo "Running Mirage (MPK)..."
python3 "$DEMO" --use-mirage "${COMMON[@]}"
echo "Comparing outputs..."
pytest -q -s "$ROOT/tests/ci-tests/test_glm5_inference_output.py"
