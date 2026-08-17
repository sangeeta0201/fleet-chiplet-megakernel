#!/bin/bash
# 8-GPU GLM: DP attention + EP MoE, whole-layer fusion.
#
# Counterpart of demo/gpt_oss/run_mp4_dp_ep_fused.sh. Two differences, both
# forced by the model rather than chosen:
#
#   * No EP_SLOT. gpt-oss splits the ACTIVATED top-k list by slot, which is the
#     better-balanced split. GLM-4.7-Flash is top-4 and a 4-entry list cannot
#     span 8 ranks, so GLM uses ep_slice -- the expert-ID split. 64 routed
#     experts / 8 ranks = 8 each.
#
#   * PRECOMPUTED_DISPATCH is mandatory, not a default. Every EP threshold in
#     the GLM monolith is a function of task_layer_idx, which only exists
#     inside the multi-layer batched loop, and persistent_kernel.py defaults
#     the knob to 0 whenever rocSHMEM is on. demo.py asserts on it rather than
#     letting the run hang.
#
# GLM's fold sits at the HEAD of layer L+1, behind the layer-entry barrier the
# multi-layer path already pays, so EP costs one rendezvous per layer where it
# costs gpt-oss two. The last fused layer's MoE output has no layer L+1 to fold
# it, hence the extra EP_TAIL_ONLY task demo.py appends before the LM head.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
# Before env_common.sh: it divides the core count by NP to size OMP_NUM_THREADS.
NP="${NP:-8}"
source ./env_common.sh

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export ROCSHMEM_MAX_NUM_CONTEXTS="${ROCSHMEM_MAX_NUM_CONTEXTS:-8}"
# DP attention is not a tuning knob. MLA keeps one shared latent head, so the
# kv_head==xcd_id mapping the fused attention task assumes has no TP analogue,
# and slicing 20 q heads eight ways is not a shape the decode kernel has.
# demo.py asserts attn_dp under MOE_EP.
export ATTN_DP="${ATTN_DP:-1}"
# Overridable only as an ablation: MOE_EP=0 at world > 1 replicates the whole
# model per rank and leaves no collective in the step at all, which is the
# isolation test for "is this the EP fold or is this multi-GPU launch".
export MOE_EP="${MOE_EP:-1}"
export EP_FOLD_RANK="${EP_FOLD_RANK:-0}"
export GLM_FUSE_FULL_LAYER="${GLM_FUSE_FULL_LAYER:-1}"
export PRECOMPUTED_DISPATCH="${PRECOMPUTED_DISPATCH:-1}"
export MPK_ML_REPLAY="${MPK_ML_REPLAY:-1}"

if [ "${KEEP_BUILD:-0}" != "1" ]; then
  rm -rf permanent_output_dir permanent_output_dir_rank*
fi

mpirun -np "$NP" --tag-output --allow-run-as-root \
  $(mpk_x_args) \
  python3 demo.py --use-mirage \
    --max-seq-length "${MAX_SEQ_LENGTH:-128}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-16}" \
    --model-path "$MODEL_PATH" "$@"
echo "MPIRUN_EXIT=$?"
