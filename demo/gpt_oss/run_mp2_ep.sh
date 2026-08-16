#!/bin/bash
# 2-GPU gpt-oss-120b: tensor-parallel attention + expert-parallel MoE.
#
# On top of run_mp2.sh's TP attention, MOE_EP=1 slices the 128 experts
# [rank*64, (rank+1)*64) so each rank loads half the expert weights too. The
# router stays replicated and the fp32 weighted partial sums are
# SUM-allreduced, with EP_FOLD_RANK deciding which rank folds the residual in.
#
# This is the configuration where the weight-load halving is complete -- both
# attention and MoE -- and so the one that has a shot at beating the ~2.1 ms
# single-GPU baseline.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env_common.sh

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-6,7}"
export ROCSHMEM_MAX_NUM_CONTEXTS="${ROCSHMEM_MAX_NUM_CONTEXTS:-2}"
export PRECOMPUTED_DISPATCH="${PRECOMPUTED_DISPATCH:-0}"
export MOE_EP=1

if [ "${KEEP_BUILD:-0}" != "1" ]; then
  rm -rf permanent_output_dir permanent_output_dir_rank*
fi

mpirun -np 2 --tag-output --allow-run-as-root \
  $(mpk_x_args) \
  python demo.py --use-mirage \
    --max-seq-length "${MAX_SEQ_LENGTH:-128}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-16}" \
    --ignore-eos \
    --model-path "$MODEL_PATH" "$@"
echo "MPIRUN_EXIT=$?"
