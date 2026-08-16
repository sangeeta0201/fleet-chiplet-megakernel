#!/bin/bash
# 4-GPU gpt-oss-120b: DP attention + EP MoE, full fusion, ONE expert per rank.
#
# Same configuration as run_mp2_dp_ep_fused.sh at world size 4. With top_k=4
# and EP_SLOT=1 the activated list splits 1/1/1/1, so each rank runs exactly
# one expert -- the finest slice this model admits at bs=1.
#
# HISTORY: this measured 3.103 ms against 2-GPU's 2.12, and the reason was NOT
# that four ranks are inherently worse. Two things in Phase 9 were hardcoded to
# world 2, so a 4-GPU run silently took the PRE-OPTIMIZATION path on all 36
# layers:
#   1. The direct peer store was gated `if constexpr (EP_WORLD_SIZE == 2)`, so
#      world 4 fell back to staged rocSHMEM putmem_signal, once per peer --
#      3 sequential work-group collectives instead of one fused store. The
#      direct-vs-staged difference at ONE peer is itself 2.229 vs 2.378.
#   2. The 9d wait looped mpk_shmem_signal_wait_ge per peer, so peer 1 was not
#      even polled until peer 0 landed: the sum of detection latencies, not the
#      max.
# Both are now general (see _full_layer_ep_fold_partial's NPEER and
# _full_layer_ep_wait_peers' concurrent poll). The fold's epilogue IS the
# transfer, so N-1 peers cost N-1 write-through stores issued back to back and
# drained once. The store COUNT is linear (N-1), but the measured cost is flat:
# Phase 9 is 0.229 ms at world 2 and 0.231 ms at world 4, because the stores
# issue back to back and share one drain.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env_common.sh

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-4,5,6,7}"
export ROCSHMEM_MAX_NUM_CONTEXTS="${ROCSHMEM_MAX_NUM_CONTEXTS:-4}"
# DP attention is not a tuning knob here: dp_fusable in demo.py is
# `attn_dp and moe_ep`, and it gates the full-layer monolith. ATTN_DP=0 (tensor
# parallel) sharded KV heads break the kv_head==xcd_id mapping in the fused QKV
# task, so the whole run falls back to the unfused per-task path. Overridable
# only to measure that fallback.
export ATTN_DP="${ATTN_DP:-1}"
export MOE_EP=1
export EP_SLOT="${EP_SLOT:-1}"
export W13_OPW="${W13_OPW:-64}"
export FUSE_FULL_LAYER="${FUSE_FULL_LAYER:-1}"
export PRECOMPUTED_DISPATCH="${PRECOMPUTED_DISPATCH:-1}"

if [ "${KEEP_BUILD:-0}" != "1" ]; then
  rm -rf permanent_output_dir permanent_output_dir_rank*
fi

mpirun -np 4 --tag-output --allow-run-as-root \
  $(mpk_x_args) \
  python demo.py --use-mirage \
    --max-seq-length "${MAX_SEQ_LENGTH:-128}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-16}" \
    --ignore-eos \
    --model-path "$MODEL_PATH" "$@"
echo "MPIRUN_EXIT=$?"
