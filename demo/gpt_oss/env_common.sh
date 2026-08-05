#!/bin/bash
# Shared environment for the gpt-oss-120b megakernel runs.
# Sourced by run_1gpu.sh / run_mp2.sh / run_mp2_ep.sh -- not run directly.

FLEET_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# MIRAGE_HOME decides which tree the megakernel compiles against, and
# PYTHONPATH decides which one `import mirage` resolves to. A stale editable
# install (~/.local/.../__editable__.mirage_project*.pth) points at a
# *different* checkout, so both must be pinned or the run silently exercises
# the wrong codebase.
export MIRAGE_HOME="$FLEET_HOME"
export PYTHONPATH="$FLEET_HOME/python:${PYTHONPATH:-}"

export USE_FP8_ACT=1

export MODEL_PATH="${MODEL_PATH:-/root/schowdha/models/gpt-oss-120b}"

# rocSHMEM + the MPI it was built against. rocSHMEM's IPC backend only needs
# MPI for bootstrap (rank exchange), not for the data path.
export ROCSHMEM_INC_PATH="${ROCSHMEM_INC_PATH:-/home/claudeuser/rocshmem/include}"
export ROCSHMEM_LIB_PATH="${ROCSHMEM_LIB_PATH:-/home/claudeuser/rocshmem/lib}"
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

# Every knob the multi-GPU path reads, forwarded to both ranks by mpirun.
# Listing them unconditionally is deliberate: -x on an unset variable is a
# no-op, so a knob set in the caller's shell reaches rank 1 without editing
# this list.
MPK_FORWARD_VARS=(
  MIRAGE_HOME PYTHONPATH USE_FP8_ACT MODEL_PATH
  HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
  ROCSHMEM_INC_PATH ROCSHMEM_LIB_PATH MPI_INC_PATH MPI_LIB_PATH
  LD_LIBRARY_PATH PATH
  ROCSHMEM_MAX_NUM_CONTEXTS
  MOE_EP EP_NOAR EP_NOSLICE EP_FOLD_RANK
  ATTN_DP MPK_INLINE_AR2 INLINE_AR2 FUSED_MGPU_NOAR
  AR2_ELEMS_PER_BLOCK AR2_TARGET_GRID AR2_PREFETCH AR2_PREFETCH_BLOCKS
  MPK_SPAN_TIMING MPK_SUBPHASE_TIMING MPK_MOE_SUBPHASE
  MPK_OVERLAP_XGPU MPK_DEVICE_ACCUM
  FUSE_FULL_LAYER FUSE_FULL_LAYER_MGPU FUSE_QKV_ATTN FUSE_OPROJ_MOE
  PRECOMPUTED_DISPATCH USE_GANG PPL_MODE PPL_MXFP4_MATCH
  MLP_DBG MLP_FINAL_IDX
)

mpk_x_args() {
  local v
  for v in "${MPK_FORWARD_VARS[@]}"; do printf -- '-x %s ' "$v"; done
}
