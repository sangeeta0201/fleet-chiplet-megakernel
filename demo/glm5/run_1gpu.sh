#!/bin/bash
# Single-GPU GLM, whole-layer fusion. The reference leg of the correctness
# suite, and the launcher every single-GPU latency number on this branch comes
# from.
#
# USE_MIRAGE=0 runs the same prompt through the HF/torch model instead of the
# megakernel. That is the token reference: it shares the tokenizer, the
# checkpoint and the sampling policy with the mirage leg, so a disagreement is
# the megakernel and not the harness.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
NP=1
source ./env_common.sh

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export GLM_FUSE_FULL_LAYER="${GLM_FUSE_FULL_LAYER:-1}"
export PRECOMPUTED_DISPATCH="${PRECOMPUTED_DISPATCH:-1}"
export MPK_ML_REPLAY="${MPK_ML_REPLAY:-1}"

USE_MIRAGE="${USE_MIRAGE:-1}"
if [ "$USE_MIRAGE" = "1" ]; then
  MIRAGE_FLAG="--use-mirage"
  if [ "${KEEP_BUILD:-0}" != "1" ]; then
    rm -rf permanent_output_dir permanent_output_dir_rank*
  fi
else
  # The torch leg builds no kernel, so KEEP_BUILD is irrelevant to it -- and it
  # must NOT delete the build dir, or it invalidates the mirage leg that the
  # suite may have already run.
  MIRAGE_FLAG=""
fi

python3 demo.py $MIRAGE_FLAG \
  --max-seq-length "${MAX_SEQ_LENGTH:-128}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-16}" \
  --model-path "$MODEL_PATH" "$@"
_run_rc=$?
echo "RUN_EXIT=$_run_rc"
exit "$_run_rc"
