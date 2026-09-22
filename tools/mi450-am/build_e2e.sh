#!/bin/bash
# Build the megakernel e2e harness.
#
# DO NOT ADD FLAGS. This reproduces the one flag set measured to emit the
# correct token (55). The megakernel is schedule-sensitive around the clock read
# (see PORTING_MI450.md, task #21) and the sensitivity is sharp enough that
# merely adding -DMPK_HOST_BREADCRUMB -DMPK_MAX_RESIDENT_BLOCKS=9 flips the
# emitted token to 29, as does -DMIRAGE_REALTIME_FALLBACK=1.
#
# The difference from build.sh is -DMIRAGE_XCD_ID_FROM_BLOCKIDX_Y=1.
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
  -DMIRAGE_XCD_ID_FALLBACK=1 -DMIRAGE_XCD_ID_FROM_BLOCKIDX_Y=1 \
  -O3 -std=c++17 -w "$@"
