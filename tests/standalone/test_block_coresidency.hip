// Is 2 blocks/CU reachable on this part at all?
//
// WHY THIS TEST EXISTS. The GLM megakernel runs at 1 wave/SIMD. Two facts sit
// on top of each other and it matters which one binds:
//
//   (a) `worker_kernel` is 284 unified VGPRs (set by mla_decode_absorbed, a
//       phase that uses 16 of 232 workers). 512/SIMD unified file, granule 8:
//       284 -> 288 -> floor(512/288) = 1 wave/SIMD. Need <= 256 for 2.
//   (b) The launch is 232 worker blocks + 8 scheduler blocks on a GPU with
//       ~256 CUs, so the grid cannot put 2 blocks on a CU no matter what the
//       register file allows.
//
// (b) binds TODAY, which is why cutting registers alone would measure a null.
// But (a) is a hard PRECONDITION for ever fixing (b): raising MPK_NUM_WORKERS
// past the CU count only helps if a CU can host two of these blocks. So the
// question that gates the whole occupancy ladder -- worth several turns of
// register surgery if the answer is yes, and worth zero if the answer is no --
// is simply:
//
//   how many 256-thread blocks does this GPU actually co-schedule per CU,
//   as a function of the kernel's unified VGPR count?
//
// hipOccupancyMaxActiveBlocksPerMultiprocessor answers it analytically. This
// test also answers it EMPIRICALLY, because the analytic number is a model and
// the model has been wrong before on this box. The empirical method is the
// standard one: launch a grid far larger than the machine, have every resident
// block bump a live counter, hold them all resident for a fixed spin, and take
// the high-water mark. A block that is not co-resident cannot contribute,
// so the maximum observed live count IS the co-residency.
//
// Register pressure is forced by holding NACC real loaded floats live ACROSS
// the residency window, with an impossible-condition sink so nothing is dead.
// Range clobbers ("a[0:31]") are rejected by this clang, and __launch_bounds__
// would beg the question by asking the driver for the occupancy under test.
//
// The arm labels are nominal. The number that matters is the ACTUAL
// .vgpr_count and .vgpr_spill_count per arm, which the companion
// check_coresidency_regs.sh reads out of the object metadata -- an arm that
// spilled is measuring scratch traffic, not register pressure, and must not
// be read as a register data point.
//
// Run: ./test_block_coresidency
#include <hip/hip_runtime.h>
#include <cstdio>

#define HIP_CHECK(x)                                                           \
  do {                                                                         \
    hipError_t _e = (x);                                                       \
    if (_e != hipSuccess) {                                                    \
      printf("HIP error %s at line %d\n", hipGetErrorString(_e), __LINE__);     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

// Hold every resident block alive for the same window so the high-water mark
// is taken while the machine is full. ~2 ms at 100 MHz s_memrealtime.
__device__ __forceinline__ void hold(unsigned long long ticks) {
  unsigned long long t0 = __builtin_amdgcn_s_memrealtime();
  while (__builtin_amdgcn_s_memrealtime() - t0 < ticks) {
    __builtin_amdgcn_s_sleep(8);
  }
}

// live  : how many blocks are resident right now
// hiwat : the maximum `live` ever observed
template <int NACC>
__global__ void __launch_bounds__(256)
    coresident(int *live, int *hiwat, unsigned long long ticks,
               float const *seed, float *sink) {
  // Real loads, so the values cannot be constant-folded away.
  float acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; i++) {
    acc[i] = seed[(threadIdx.x + i) & 255];
  }

  if (threadIdx.x == 0) {
    int now = atomicAdd(live, 1) + 1;
    atomicMax(hiwat, now);
  }
  __syncthreads();

  // acc[] is live across this, which is what pins the register allocation.
  hold(ticks);

  __syncthreads();
  if (threadIdx.x == 0) {
    atomicSub(live, 1);
  }

  float s = 0.f;
#pragma unroll
  for (int i = 0; i < NACC; i++) {
    s += acc[i];
  }
  // Sink. Never taken, but the compiler cannot prove it.
  if (s == 1.2345678e-31f) {
    sink[0] = s;
  }
}

template <int NACC>
static int measure(char const *label, int cus, int grid,
                   unsigned long long ticks, float const *seed, float *sink) {
  int *d;
  HIP_CHECK(hipMalloc(&d, 2 * sizeof(int)));
  HIP_CHECK(hipMemset(d, 0, 2 * sizeof(int)));

  int occ = 0;
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &occ, (void const *)coresident<NACC>, 256, 0));

  hipLaunchKernelGGL((coresident<NACC>), dim3(grid), dim3(256), 0, 0, d, d + 1,
                     ticks, seed, sink);
  HIP_CHECK(hipDeviceSynchronize());

  int h[2];
  HIP_CHECK(hipMemcpy(h, d, 2 * sizeof(int), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d));

  printf("  %-28s  analytic=%d blk/CU   empirical hi-water=%4d blocks "
         "= %.2f blk/CU\n",
         label, occ, h[1], (double)h[1] / (double)cus);
  return h[1];
}

int main() {
  hipDeviceProp_t p;
  HIP_CHECK(hipGetDeviceProperties(&p, 0));
  int const cus = p.multiProcessorCount;
  printf("device                : %s\n", p.gcnArchName);
  printf("multiProcessorCount   : %d\n", cus);
  printf("maxThreadsPerMultiProc: %d\n", p.maxThreadsPerMultiProcessor);
  printf("regsPerBlock          : %d\n", p.regsPerBlock);
  printf("sharedMemPerBlock     : %zu\n", (size_t)p.sharedMemPerBlock);
  printf("warpSize              : %d\n\n", p.warpSize);

  // 8x the machine, so the grid can never be the limit.
  int const grid = cus * 8;
  unsigned long long const ticks = 200000ull; // ~2 ms at 100 MHz

  float *seed, *sink;
  HIP_CHECK(hipMalloc(&seed, 256 * sizeof(float)));
  HIP_CHECK(hipMalloc(&sink, sizeof(float)));
  HIP_CHECK(hipMemset(seed, 0, 256 * sizeof(float)));
  HIP_CHECK(hipMemset(sink, 0, sizeof(float)));

  printf("256-thread blocks, grid=%d (8x the machine):\n", grid);
  // NACC -> .vgpr_count is NACC+3 for all of these (checked, 0 spill).
  measure<8>("vgpr 11   minimal", cus, grid, ticks, seed, sink);
  measure<120>("vgpr 123", cus, grid, ticks, seed, sink);
  measure<200>("vgpr 203", cus, grid, ticks, seed, sink);
  measure<253>("vgpr 256  <- the target", cus, grid, ticks, seed, sink);
  measure<261>("vgpr 264", cus, grid, ticks, seed, sink);
  measure<281>("vgpr 284  <- GLM today", cus, grid, ticks, seed, sink);

  printf("\nRead: if nacc=8 lands at >= 2 blk/CU, a CU CAN host two of these\n"
         "blocks, so GLM's 284 unified VGPRs are a real precondition worth\n"
         "register surgery. If nacc=8 lands at ~1 blk/CU, blocks/CU is capped\n"
         "by something the register file does not control, and the whole\n"
         "occupancy ladder is dead no matter how many registers come off.\n");
  return 0;
}
