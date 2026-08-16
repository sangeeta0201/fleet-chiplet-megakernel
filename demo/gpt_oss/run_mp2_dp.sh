#!/bin/bash
# 2-GPU gpt-oss-120b: DATA-parallel attention + expert-parallel MoE.
#
# The difference from run_mp2_ep.sh is ATTN_DP=1. Under TP the attention heads
# are split and o_proj's row-parallel output must be SUM-allreduced before the
# router runs; that allreduce plus the MoE combine is 2 syncs x 36 layers = 72
# cross-GPU syncs per token. Measured per-layer gap was 59.93us against 33.60us
# of compute (MULTI_GPU_NOTES.md) -- at bs=1 the syncs, not the math, are the
# latency, and no amount of overlap fixes that because comm is 1.8x compute.
#
# ATTN_DP replicates attention on both ranks instead of splitting it: same
# heads, same KV cache, same token, so o_proj's output is already complete and
# allreduce #1 is deleted outright. 72 syncs -> 36. What it costs is duplicated
# attention compute and a duplicated KV cache, which at bs=1 / short context is
# a small share of the step; at large batch or long context the trade flips.
#
# MOE_EP=1 is still on, so the expert weights remain sharded and the one
# surviving allreduce is the MoE combine.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env_common.sh

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-6,7}"
export ROCSHMEM_MAX_NUM_CONTEXTS="${ROCSHMEM_MAX_NUM_CONTEXTS:-2}"
export PRECOMPUTED_DISPATCH="${PRECOMPUTED_DISPATCH:-0}"
export ATTN_DP=1
# MOE_EP=0 leaves zero allreduces in the step -- both ranks then run the whole
# model redundantly. Useless for latency, but the cleanest correctness check of
# the DP attention path itself, since it should track single-GPU closely.
export MOE_EP="${MOE_EP:-1}"

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
