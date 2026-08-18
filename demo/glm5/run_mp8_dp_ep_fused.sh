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

# Leave two CUs per XCD idle. This is a liveness fix, not a tuning knob, and
# it is what makes the full 78-layer NP=8 run complete at all.
#
# The default (utils.py) is 240 workers + 8 schedulers = 248 blocks on 256 CUs,
# i.e. 31 of the 32 CUs on every XCD. The megakernel requires every block to be
# co-resident -- a worker that never gets a CU never reports into
# worker_xcd_ready_count, and all eight schedulers then spin in the bootstrap
# wait forever without dispatching a single task. Workers and schedulers are
# two separate kernels on two separate streams, so their blocks are round-robined
# onto XCDs independently; nothing guarantees the 8 scheduler blocks land one
# per XCD, and any XCD that draws two of them needs 33 slots for 32 CUs.
#
# Measured, full 78 layers, NP=8, both runs with the NUMA binding below:
#
#   240 workers : hang at layer 0. Rank 3's worker 237 -- XCD 237%8 = 5,
#                 xcd_rank 237/8 = 29, i.e. the last worker on its XCD -- sat at
#                 MPK_WS_UNWRITTEN, never resident. Its 8 schedulers were all at
#                 MPK_WS_SCHED_ENTERED with worker_xcd_ready_count = 239 of 240.
#                 The other 7 ranks were healthy and merely blocked behind it:
#                 0,1,2,4 waiting on peer 3 at exp=1 with rank 3's own signal
#                 copy reading 0 (never published, not a lost push), and 5,6,7
#                 one layer further waiting on rank 1.
#   232 workers : completes. 18.110 ms/iter decode, no [EPTMO]/[EPREL], correct
#                 text ("...Paris. ...Berlin. ...Warsaw. ...Rome.").
#
# Scoped to this script on purpose. gpt-oss and single-GPU GLM have run at 240
# workers for a long time; the co-residency margin is the same there in
# principle, but the failure has only ever been observed at NP=8, and changing
# their worker count would silently move every latency number already recorded
# against them.
export MPK_NUM_WORKERS="${MPK_NUM_WORKERS:-232}"

# Pin each rank to the NUMA node its GPU hangs off.
#
# OpenMPI 4.1.2 with no binding flags maps round-robin BY SOCKET, so ranks
# 0,2,4,6 land on socket 0 and 1,3,5,7 on socket 1, while HIP_VISIBLE_DEVICES
# hands rank i GPU i -- and GPUs 0-3 sit on NUMA 0, GPUs 4-7 on NUMA 1
# (/sys/.../numa_node: 05,15,65,75 -> 0; 85,95,e5,f5 -> 1). Six of the eight
# ranks therefore drive a GPU across the socket link. `ppr:4:numa` fills node 0
# with ranks 0-3 and node 1 with ranks 4-7, the rank->GPU->NUMA identity this
# demo wants.
#
# This was added while chasing the hang above, on the theory that the straggler
# set tracked socket 1. It did not: the laggards moved from {1,3,5,7} to {4,6,7}
# when the binding changed, the collapse point moved with them (fold 6454 vs
# 347), and the actual cause turned out to be the co-residency race. XGMI here
# is fully connected at uniform weight, so there is no GPU-side asymmetry for a
# socket to explain. Keep it because rank->GPU->NUMA identity is right on its
# own terms; do NOT credit it with fixing anything.
#
# Override with MPK_MPI_BIND="--bind-to none" to isolate a binding regression.
MPK_MPI_BIND="${MPK_MPI_BIND:---map-by ppr:$((NP / 2)):numa --bind-to numa}"

mpirun -np "$NP" --tag-output --allow-run-as-root \
  $MPK_MPI_BIND \
  $(mpk_x_args) \
  stdbuf -oL -eL python3 demo.py --use-mirage \
    --max-seq-length "${MAX_SEQ_LENGTH:-128}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-16}" \
    --model-path "$MODEL_PATH" "$@"
echo "MPIRUN_EXIT=$?"
