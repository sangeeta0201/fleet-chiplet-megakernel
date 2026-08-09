#!/bin/bash
# 2-GPU gpt-oss-120b: DATA-parallel attention + EXPERT-parallel MoE, FULL fusion.
#
# This is the configuration that actually shards: the 128 experts split 64/64
# across ranks, halving MoE weight memory and MoE compute per rank, while
# attention stays replicated so allreduce #1 is gone. One collective per layer
# (the MoE combine) instead of two.
#
# The MoE combine is INLINED into the monolith (Phase 9 of
# gang_full_layer_fused_mi300.cuh), not dispatched as separate tasks. That
# matters for two reasons:
#
#   1. Correctness. A dispatched collective needs its own gang group, so each
#      layer had two interleaved gang groups and they deadlocked against each
#      other (workers parked on the collective's deps while the monolith's
#      XCD barrier waited for those same workers). Inlining removes the second
#      gang group, so the deadlock is gone by construction.
#   2. Latency. The dispatched version cost 3 tasks/layer (fold, identity,
#      allreduce) = 2 extra scheduler round-trips per layer, 72 per token.
#      Inlined, the workers never talk to the scheduler for the collective:
#      the MoE tile loop falls straight through into a GPU-wide barrier, a
#      put+signal, a signal-wait and a local reduce.
#
# The ~0.9 ms sync figure in MULTI_GPU_NOTES.md 3 measured DISPATCHED
# collectives, where the round-trips dominate, so it does not apply here.
# Measured on this path (MAX_SEQ_LENGTH=128, decode steady-state, back-to-back
# against run_mp2_dp_fused.sh):
#
#   2 GPU DP, monolith + precomputed        2.122 ms/token   4/4 @ 256 tokens
#   2 GPU DP+EP, monolith + inline combine  2.892 ms/token   4/4 @ 256 tokens
#
# +0.77 ms, i.e. ~21 us per layer for the one remaining sync -- about bare
# put+signal+wait latency, so there is little fusion left to extract. At bs=1
# that still outweighs the halved expert work. EP's win here is memory
# capacity and per-rank weight traffic; a faster EP needs FEWER syncs
# (batching layers between collectives), not a cheaper one.
#
# Fusion applies here for the same reason it does under dp_local: everything
# upstream of the MoE combine -- QKV, attention, o_proj, router, top-k -- is
# per-rank identical to single GPU. The monolith carries an expert-ownership
# window (EXPERT_BASE / NUM_LOCAL_EXPERTS); tiles for non-owned experts decode
# against the replicated global mask and early-return.
#
# PRECOMPUTED_DISPATCH=1 is required for the monolith (see run_mp2_dp_fused.sh
# for why) and was verified not to deadlock with cross-GPU events present.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env_common.sh

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-6,7}"
export ROCSHMEM_MAX_NUM_CONTEXTS="${ROCSHMEM_MAX_NUM_CONTEXTS:-2}"
export ATTN_DP=1
export MOE_EP=1
# Slot-parallel expert ownership (activated-list position, weights replicated)
# rather than an expert-id window. Splits 2/2 on every token instead of 3-1 half
# the time, and lets the tile space be built over owned experts only.
# Measured 2.484 vs 2.520 ms for the id split.
export EP_SLOT="${EP_SLOT:-1}"
# W13 tile size, EP-specific. The 1-GPU default of 128 was tuned where no
# worker is spare; under EP only half the experts run here, so 64 spends the
# freed workers on shortening the W13 -> per-expert barrier -> W2 chain that
# actually sets MoE latency at bs=1. Measured 2.352 vs 2.484 under EP, and
# 2.251 vs 2.133 on ONE GPU -- it is a win only when EP has freed the workers.
export W13_OPW="${W13_OPW:-64}"
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
