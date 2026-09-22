#!/bin/bash
# Build a gfx1250 translation unit for the FFM-Lite / AM models.
#
# This is scripts/ffm_build_mi450.sh with the paths taken from env.sh instead of
# hardcoded. -O3 is load-bearing for CORRECTNESS, not just speed: at -O0 the
# gfx1250 backend keeps __shared__ accesses in the generic address space and
# FFM-Lite's flat sub-dword path then returns the whole containing dword for a
# 1-byte read of a __shared__ array.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
R=$FLEET
exec $TOOLCHAIN/llvm/bin/clang++ -x hip --offload-arch=gfx1250 \
  --rocm-path=$TOOLCHAIN \
  -I$R/include -I$R/include/mirage/hip_compat \
  -I$R/include/mirage/persistent_kernel \
  -I$R/deps/composable_kernel/include -I$R/deps/rocblas/include \
  -I$R/deps/cutlass/include -I$R/deps/json/include \
  -D__HIP_PLATFORM_AMD__=1 -DMIRAGE_AMD_MI300 -DMIRAGE_AMD_MI450 -DMIRAGE_BACKEND_USE_ROCM \
  -DMPK_TARGET_CC=125 -DMODE_OFFLINE -DMAX_WORKER_PER_SCHEDULER=128 \
  -DMPK_MAX_NUM_BATCHED_REQUESTS=8 -DMPK_MAX_NUM_BATCHED_TOKENS=512 \
  -DMPK_MAX_NUM_PAGES=1024 -DMPK_PAGE_SIZE=64 -DMPK_MAX_SEQ_LENGTH=4096 \
  -DMPK_PROFILING_NUM_ITERS=0 -DMIRAGE_USE_CUTLASS_KERNEL=0 \
  -DCK_TILE_FMHA_FWD_FAST_EXP2=1 -DMPK_ENABLE_GANG_TASKS \
  -DMPK_FUSED_LAYER_BATCHING \
  -DMIRAGE_XCD_ID_FALLBACK=1 \
  -O3 -std=c++17 -w "$@"
