#!/bin/bash
# Shared environment for the GLM megakernel runs.
# Sourced by run_mp8_dp_ep_fused.sh -- not run directly.
#
# Deliberately parallel to demo/gpt_oss/env_common.sh; the only GLM-specific
# parts are MODEL_PATH and the GLM_* knob names in MPK_FORWARD_VARS.

FLEET_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# MIRAGE_HOME decides which tree the megakernel compiles against, and
# PYTHONPATH decides which one `import mirage` resolves to. A stale editable
# install (~/.local/.../__editable__.mirage_project*.pth) points at a
# *different* checkout -- /home/claudeuser/mirage, branch
# amd-multi-gpu-rocshmem, which has no GLM code -- so both must be pinned or
# the run silently exercises the wrong codebase.
export MIRAGE_HOME="$FLEET_HOME"
export PYTHONPATH="$FLEET_HOME/python:${PYTHONPATH:-}"

export MODEL_PATH="${MODEL_PATH:-zai-org/GLM-4.7-Flash}"

# rocSHMEM + the MPI it was built against. rocSHMEM's IPC backend only needs
# MPI for bootstrap (rank exchange), not for the data path.
#
# The path is probed rather than hardcoded: gpt_oss/env_common.sh names
# /home/claudeuser/rocshmem, which does not exist on this box -- the install
# is /home/claudeuser/rocshmem_install (the source tree is /home/claudeuser/
# rocSHMEM and has headers but no librocshmem.a). Getting this wrong fails
# late and unhelpfully, at codegen, on every rank at once.
if [ -z "${ROCSHMEM_INC_PATH:-}" ]; then
  for d in /home/claudeuser/rocshmem_install /home/claudeuser/rocshmem \
           /opt/rocshmem /usr/local/rocshmem; do
    if [ -f "$d/include/rocshmem/rocshmem.hpp" ] && \
       [ -f "$d/lib/librocshmem.a" ]; then
      export ROCSHMEM_INC_PATH="$d/include"
      export ROCSHMEM_LIB_PATH="${ROCSHMEM_LIB_PATH:-$d/lib}"
      break
    fi
  done
fi
if [ -z "${ROCSHMEM_INC_PATH:-}" ]; then
  echo "env_common.sh: no rocSHMEM install found; set ROCSHMEM_INC_PATH" >&2
fi
export ROCSHMEM_INC_PATH="${ROCSHMEM_INC_PATH:-}"
export ROCSHMEM_LIB_PATH="${ROCSHMEM_LIB_PATH:-}"
if [ -d /home/claudeuser/ompi/lib ]; then
  export MPI_INC_PATH="${MPI_INC_PATH:-/home/claudeuser/ompi/include}"
  export MPI_LIB_PATH="${MPI_LIB_PATH:-/home/claudeuser/ompi/lib}"
  export PATH="/home/claudeuser/ompi/bin:$PATH"
  export LD_LIBRARY_PATH="/home/claudeuser/ompi/lib:/home/claudeuser/ucx/lib:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
else
  export MPI_INC_PATH="${MPI_INC_PATH:-/usr/lib/x86_64-linux-gnu/openmpi/include}"
  export MPI_LIB_PATH="${MPI_LIB_PATH:-/usr/lib/x86_64-linux-gnu/openmpi/lib}"
  export LD_LIBRARY_PATH="$MPI_LIB_PATH:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
fi

# Host thread oversubscription. torch and OpenMP both default to one thread per
# core -- 256 on this box -- and mpirun starts NP copies of that, so an 8-rank
# run puts ~2048 runnable threads on 256 cores. The megakernel is persistent and
# the host thread's only job between iterations is to bump the step counter, so
# losing that thread to the scheduler idles the whole GPU. Divide the cores.
if [ -z "${OMP_NUM_THREADS:-}" ]; then
  _mpk_cores=$(nproc 2>/dev/null || echo 8)
  _mpk_np="${NP:-1}"
  export OMP_NUM_THREADS=$(( _mpk_cores / _mpk_np ))
  [ "$OMP_NUM_THREADS" -lt 1 ] && export OMP_NUM_THREADS=1
  unset _mpk_cores _mpk_np
fi
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

# Every knob the GLM path reads, forwarded to all ranks by mpirun. Listing them
# unconditionally is deliberate: -x on an unset variable is a no-op, so a knob
# set in the caller's shell reaches every rank without editing this list.
MPK_FORWARD_VARS=(
  MIRAGE_HOME PYTHONPATH MODEL_PATH GLM_MODEL_PATH
  HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
  ROCSHMEM_INC_PATH ROCSHMEM_LIB_PATH MPI_INC_PATH MPI_LIB_PATH
  LD_LIBRARY_PATH PATH
  OMP_NUM_THREADS MKL_NUM_THREADS
  ROCSHMEM_MAX_NUM_CONTEXTS MASTER_PORT
  ATTN_DP MOE_EP EP_FOLD_RANK
  PRECOMPUTED_DISPATCH MPK_ML_REPLAY
  GLM_FUSE_FULL_LAYER GLM_FUSE_ATTN GLM_FUSE_OPROJ_ROUTER
  GLM_FUSE_MOE_SWIGLU GLM_FUSE_MOE_MULSUMADD
  GLM_MOE_MXFP4 GLM_MOE_MXFP8 GLM_FAKE_MXFP4_EXPERTS GLM_FAKE_MXFP4_ATTN
  GLM_DENSE_MXFP8 GLM_DENSE_MXFP8_OPW GLM_OPROJ_MXFP8 GLM_QB_MXFP8
  GLM_OPROJ_MXFP4
  GLM_QKV_MXFP8_OPW GLM_MOE_W13_OPW GLM_MOE_W2_OPW
  GLM_OPROJ_GEMV_ROWS GLM_OPROJ_PREFETCH GLM_OPROJ_TP
  GLM_PROLOGUE_PREFETCH GLM_OPROJ_RESTAGE
  GLM_UNABSORB_OPROJ GLM_WUV_GEMV_ROWS WUV_MFMA
  GLM_UNABSORB_QB GLM_WUK_GEMV_ROWS GLM_QB_OPW GLM_QB_TP
  GLM_MLA_NUM_KV_CHUNKS GLM_MLA_MERGE_DIM_SPLITS GLM_MLA_MERGE_WT
  # Compile-time and read per-rank via os.environ in persistent_kernel.py, so
  # without -x only rank 0 builds the pair-local decode barrier and the ranks
  # deadlock at Phase 6. Same bug the ceiling probes below had.
  GLM_MLA_PAIR_MERGE
  GANG_TILE_N GANG_WGM GANG_K_SPLITS
  MPK_SPAN_TIMING MPK_SUBPHASE_TIMING MPK_DEVICE_TIMING MPK_WORKER_STATE
  MPK_EP_SIG_DBG MPK_EP_FORCE_STAGED MPK_EP_ABLATE MPK_EP_WAIT_TIMEOUT
  MPK_EP_TMO_PRINT_LAYERS
  # Ceiling probes. All are WRONG OUTPUT by construction and all are
  # compile-time, so every rank has to see them or the ranks build different
  # megakernels and the layer barriers deadlock.
  MPK_MLA_SKIP_DECODE MPK_ATTN_HALFK MPK_W13_EARLY_REL
  MPK_ABL_QKV MPK_ABL_QKV_PRO
  # Correct-output pricing probe, but compile-time all the same.
  MPK_NULL_PHASES MPK_NULL_TREE MPK_NULL_TILES
  # Real changes, not probes, but compile-time: every rank must build them or
  # the arrival counters (BAR_TREE) / the W2 tile space (W2_KSPLIT) disagree
  # with the host loop bound and the layer barriers wedge.
  MPK_BAR_TREE MPK_W2_KSPLIT MPK_MOE_LIVE_BOUND MPK_W2_STAGE_FULL
  MPK_MOE_ACT_FP8
  MPK_QKV_EP_FOLD MPK_QKV_PRO_HOIST MPK_QUANT_V16
  MPK_BAR_SKEW MPK_EP_FOLD_WGS MPK_EP_POLL_BATCH MPK_ML_PTR_PREFETCH
  MPK_ML_BOUNDARY_PAD MPK_BAR_SKEW_DROP_NS MPK_WUV_IN_MERGE GLM_RESADD_UNROLL
  MPK_PRINT_GEOMETRY
  MPK_PRINT_ALL_RANKS
  MPK_HOST_DBG_POLL
  MAX_SAVE_TOKENS
  MPK_NUM_WORKERS
)

mpk_x_args() {
  local v
  for v in "${MPK_FORWARD_VARS[@]}"; do printf -- '-x %s ' "$v"; done
}
