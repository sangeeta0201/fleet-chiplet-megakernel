#!/bin/bash
# 2-GPU gpt-oss-120b: tensor-parallel attention, replicated MoE.
#
# Q/KV heads are split across the two ranks, so each rank loads half the
# attention weights; the O-projection partials are SUM-allreduced. MoE is
# left replicated here -- run_mp2_ep.sh adds expert parallelism on top.
#
# Pick two GPUs on the same node; rocSHMEM uses the IPC (single-node)
# backend, so no fabric setup is needed.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env_common.sh

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-6,7}"
export ROCSHMEM_MAX_NUM_CONTEXTS="${ROCSHMEM_MAX_NUM_CONTEXTS:-2}"

# The precomputed-dispatch template is timed against the single-GPU schedule.
# Under multi-GPU the cross-GPU waits do not fit it and the megakernel
# deadlocks, so the dynamic scheduler is required. persistent_kernel.py
# already defaults this off when rocSHMEM is in use; set it explicitly so a
# stale value in the caller's shell cannot re-enable it.
export PRECOMPUTED_DISPATCH="${PRECOMPUTED_DISPATCH:-0}"

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
