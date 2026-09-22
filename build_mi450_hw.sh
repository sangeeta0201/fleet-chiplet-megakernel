#!/usr/bin/env bash
# Build a gfx1250 (MI450) translation unit for REAL HARDWARE.
#
# This is deliberately a separate script from the FFM build. Three of the flags
# the FFM builds carry are not merely unnecessary on silicon -- they are wrong,
# and each fails silently rather than loudly:
#
#   -DMIRAGE_XCD_ID_FALLBACK=1        hardcodes xcd_id() to 0. On hardware every
#                                     workgroup then believes it is on XCD 0, so
#                                     the MoE tile decode `tile_idx * 8 + xcd_id`
#                                     collapses: 7 of every 8 tiles never run and
#                                     8 XCDs redundantly compute the same tile.
#                                     Wrong results, no diagnostic.
#   -DMIRAGE_XCD_ID_FROM_BLOCKIDX_Y=1 test-only seam (substitutes blockIdx.y).
#                                     arch_traits.cuh says "never define this in
#                                     a shipping build" -- correct only when the
#                                     launch actually sets gridDim.y == 8.
#   -DMPK_MAX_RESIDENT_BLOCKS=<n>     clamps the worker count down to whatever
#                                     FFM could keep co-resident (9 for this
#                                     megakernel). Silicon has no such ceiling;
#                                     leaving it defined caps you at a handful of
#                                     workers and looks like a performance
#                                     mystery, not a build error.
#
# None of those three appear below. That is the entire point of the file.
#
# Usage:  ./build_mi450_hw.sh <source.hip> -o <binary> [extra flags]
#
# Optional environment:
#   ROCM        path to a gfx1250-capable ROCm/LLVM (default: the port toolchain)
#   TICK_NS     nanoseconds per realtime tick, if you have measured it
#               (see calibrate_tick_ns below). Unset leaves the unverified
#               placeholder of 10 and profiler microseconds stay untrustworthy.
set -euo pipefail

R="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM="${ROCM:-/home/claudeuser/mi450-toolchain}"

EXTRA=()
# MIRAGE_TICK_NS is a scale factor on profiler timestamps only; it cannot change
# numerical results. Passing it is optional precisely so a correctness bring-up
# is not blocked on having measured the clock yet.
[ -n "${TICK_NS:-}" ] && EXTRA+=("-DMIRAGE_TICK_NS=${TICK_NS}")

exec "$ROCM/llvm/bin/clang++" -x hip --offload-arch=gfx1250 \
  --rocm-path="$ROCM" \
  -I"$R/include" -I"$R/include/mirage/hip_compat" \
  -I"$R/include/mirage/persistent_kernel" \
  -I"$R/deps/composable_kernel/include" -I"$R/deps/rocblas/include" \
  -I"$R/deps/cutlass/include" -I"$R/deps/json/include" \
  -D__HIP_PLATFORM_AMD__=1 \
  -DMIRAGE_AMD_MI300 -DMIRAGE_AMD_MI450 -DMIRAGE_BACKEND_USE_ROCM \
  -DMPK_TARGET_CC=125 -DMODE_OFFLINE -DMAX_WORKER_PER_SCHEDULER=128 \
  -DMPK_MAX_NUM_BATCHED_REQUESTS=8 -DMPK_MAX_NUM_BATCHED_TOKENS=512 \
  -DMPK_MAX_NUM_PAGES=1024 -DMPK_PAGE_SIZE=64 -DMPK_MAX_SEQ_LENGTH=4096 \
  -DMPK_PROFILING_NUM_ITERS=0 -DMIRAGE_USE_CUTLASS_KERNEL=0 \
  -DCK_TILE_FMHA_FWD_FAST_EXP2=1 -DMPK_ENABLE_GANG_TASKS \
  -DMPK_FUSED_LAYER_BATCHING \
  "${EXTRA[@]}" \
  -O3 -std=c++17 -w "$@"

# -----------------------------------------------------------------------------
# Why -O3 is not a preference here
#
# gfx1250 at -O0 miscompiles sub-dword LDS access: a byte read of __shared__
# returns the containing dword. Every mi450 kernel that packs bf16 or fp8 into
# LDS depends on this working. -O3 is a correctness flag on this target, not a
# tuning flag. Do not "build a debug version" by dropping to -O0 and then trust
# the numbers.
#
# MIRAGE_AMD_MI300 stays defined on an MI450 build. It is not a typo. Roughly 30
# sites use it to mean "AMD, not NVIDIA"; MIRAGE_AMD_MI450 is tested first at
# persistent_kernel.cuh:332 to select the mi450 task headers. Removing MI300
# drops the whole build off the AMD path.
# -----------------------------------------------------------------------------
