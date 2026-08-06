#!/bin/bash
# 2-GPU gpt-oss-120b: DATA-parallel attention, replicated MoE, FULL fusion.
#
# This is the only multi-GPU configuration that reaches single-GPU latency
# (2.135 ms/token decode vs 2.135 on run_1gpu.sh, MULTI_GPU_NOTES.md 3d).
#
# It gets there by having no collectives at all. ATTN_DP=1 deletes allreduce #1
# and MOE_EP=0 deletes #2, so the step emits zero cross-GPU events and each
# rank's task graph is byte-for-byte the single-GPU one. That is what makes the
# two single-GPU fast paths legal here, and both are load-bearing:
#
#   FUSE_FULL_LAYER=1     collapses QKV+attn+o_proj+topk+MoE into one gang
#                         dispatch (task type 217). Worth ~1.6 ms/token.
#   PRECOMPUTED_DISPATCH=1  is NOT optional. The monolith deadlocks under
#                         dynamic dispatch on any GPU count -- the scheduler
#                         broadcasts to my_workers[widx] but never transmits
#                         widx, so the worker re-derives its gang rank by
#                         scanning worker_xcd_map and gets a different answer;
#                         tiles go unclaimed and the hierarchical barrier never
#                         completes. Precomputed hands the rank over directly
#                         (persistent_kernel.cuh:2022-2047).
#
# What this configuration does NOT do is shard anything: both ranks run the
# whole model on the same token, so at bs=1 the second GPU buys no latency. Its
# value is (a) 2x throughput when each rank is fed a different sequence, which
# is what data parallelism is for, and (b) proof that the fused + precomputed
# fast path survives multi-GPU intact, so sharded configs start their budget
# from 2.135 rather than from the 3.777 ms unfused/dynamic path.
#
# Turning MOE_EP=1 back on re-introduces the combine allreduce and currently
# emits garbage (MULTI_GPU_NOTES.md 3c) -- that is the open work, not this.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env_common.sh

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-6,7}"
export ROCSHMEM_MAX_NUM_CONTEXTS="${ROCSHMEM_MAX_NUM_CONTEXTS:-2}"
export ATTN_DP=1
export MOE_EP=0
export FUSE_FULL_LAYER="${FUSE_FULL_LAYER:-1}"
export PRECOMPUTED_DISPATCH="${PRECOMPUTED_DISPATCH:-1}"

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
