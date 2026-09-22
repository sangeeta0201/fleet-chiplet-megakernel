#!/bin/bash
# compile a TU for gfx1250 with the full MPK flag set
R=/home/claudeuser/fleet-mi450
/home/claudeuser/mi450-toolchain/llvm/bin/clang++ -x hip --offload-arch=gfx1250 \
  --rocm-path=/home/claudeuser/mi450-toolchain \
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
  -DMIRAGE_XCD_ID_FALLBACK=1 -DMPK_HOST_BREADCRUMB -DMPK_MAX_RESIDENT_BLOCKS=9 \
  -O3 -std=c++17 -w "$@"

# -O3 is load-bearing for *correctness* here, not just speed. At -O0 the
# gfx1250 backend keeps __shared__ accesses in the generic address space
# (the ISA contains no ds_* instructions at all), and FFM-Lite's flat
# sub-dword path mishandles the LDS aperture: a 1-byte read of a
# __shared__ unsigned char array returns the whole containing dword, so
# `arr[i] != 0` silently becomes "any of the 4 bytes at i is nonzero".
# Verified with /tmp/p1/ldsbyte.hip -- gfx1250 -O0 FAILs, gfx1250 -O3
# passes, gfx950 passes at both. This cost a long debugging detour on
# moe_topk_softmax, whose s_hit[] hit mask is exactly that pattern.
