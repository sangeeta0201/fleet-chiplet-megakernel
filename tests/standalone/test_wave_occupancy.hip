// Task #148 step 1: does 2 waves/SIMD help the MoE k-loop AT ALL?
//
// Why this test exists. `/tmp/regmeta.sh` on the shipping image says
// `worker_kernel` is 325 vgpr + 69 agpr unified, so 512/325 = ONE wave per
// SIMD. The MoE tile loop is 2.2 ms of the 9.08 ms wall (W13 1.185 + W2
// 1.048) and needs only 103 / 117 registers of its own -- it inherits 1
// wave/SIMD because AMDGPU gives the whole noinline call tree one flat
// allocation.
//
// Every prior occupancy attempt (#74/#118/#119) tried to reach 2 waves/SIMD
// through 2 BLOCKS per CU, which needs 480+ worker blocks; 464 hangs, 232
// faults, 240 is a hard floor, and the wpe=3 register squeeze costs +2.24 ms.
// So only the COST arm was ever measured. The benefit arm -- "if the tile did
// run at 2 waves/SIMD, would it be faster?" -- has never been run, and it is
// the premise under a much larger build (widening WORKER_NUM_THREADS 256->512
// reaches 2 waves/SIMD with the SAME 240 blocks, but it touches 63 task
// headers that hardcode 256 threads).
//
// The experiment. Hold total work and total bytes EXACTLY fixed and vary only
// blocks/CU, using dynamic LDS as the only knob:
//
//   arm A  dyn 64 KB -> 2*(6.4+64) = 141 KB <= 160 KB/CU  -> 2 blocks/CU
//                       3*(6.4+64) = 211 KB               -> not 3
//          512 blocks land in one round of 512 slots        = 2 waves/SIMD
//   arm B  dyn 80 KB -> 2*(6.4+80) = 173 KB >  160 KB/CU  -> 1 block/CU
//          512 blocks land in two clean rounds of 256       = 1 wave/SIMD
//
// Both arms: 512 blocks x 256 threads x REPS k-loops, identical addresses,
// identical bytes. The LDS is REQUESTED and never touched, so it costs
// nothing but residency. 512 (not 480) so arm B's second round is full and
// the ragged-tail confound is out.
//
// Bytes are made real: each (block, rep) reads its own 4x W_BYTES slice out
// of a 315 MB pool, which is larger than the 256 MB MALL, so the loads are
// not served out of cache. A cache-resident probe would answer a different
// question than the one the MoE tile asks.
//
// The k-loop under test is the REAL `_gang_moe_kloop` from the shipping
// header, at the shipping default width, not a re-implementation.
//
//   hipcc -o test_wave_occupancy test_wave_occupancy.hip \
//     -I../../include -I../../include/mirage/persistent_kernel \
//     -I../../deps/json/include -DMIRAGE_BACKEND_USE_ROCM \
//     --offload-arch=gfx950 -O3
//   HIP_VISIBLE_DEVICES=4 ./test_wave_occupancy
//
// Takes a whole GPU for ~10 s and DOES read GPU timing, so unlike
// test_moe_kloop_width.hip it must NOT run alongside a benchmark.

#include <cstdio>
#include <cstdlib>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <vector>

// Same two shims test_moe_kloop_width.hip uses.
#define MPK_WS_WAVE_SYNC(wave) ((void)0)
namespace kernel {
__device__ __forceinline__ int _gang_moe_get_xcd_id() {
  int xcd_id;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcd_id));
  return xcd_id;
}
} // namespace kernel

#include "mirage/persistent_kernel/mpk_atoms.cuh"
#include "mirage/persistent_kernel/tasks/mi300/gang_moe_linear_mxfp8_mi300.cuh"

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t _e = (call);                                                    \
    if (_e != hipSuccess) {                                                    \
      fprintf(stderr, "HIP %s:%d: %s\n", __FILE__, __LINE__,                   \
              hipGetErrorString(_e));                                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

using namespace kernel;

static constexpr int MFMA_ITERS = 48; // GLM-5 W13: hidden 6144 / 128
static constexpr bool WEIGHT_FP4 = false;
static constexpr int GROUPS_REQ = 4; // the shipping default

static constexpr int W_KS = 16 * 128; // _gang_moe_w_kstride<false>()
static constexpr int SC_KS = 64;      // _gang_moe_sc_kstride()
static constexpr size_t W_BYTES = (size_t)MFMA_ITERS * W_KS + 4096;
static constexpr size_t SC_BYTES = (size_t)MFMA_ITERS * SC_KS + 4096;
static constexpr int TOK_BYTES = MFMA_ITERS * 128 + 256;
static constexpr int TOKSC_BYTES = MFMA_ITERS + 64;

static constexpr int LANES = 64;
static constexpr int BLOCK_THREADS = 256; // == WORKER_NUM_THREADS
static constexpr int WAVES = BLOCK_THREADS / LANES;
static constexpr int GRID = 512;
static constexpr int REPS = 32;

// One slice per (block, rep) visit, with a private sub-slice per wave so the
// four waves of a block do not simply hit each other's L1 lines.
static constexpr size_t SLICE_W = (size_t)WAVES * W_BYTES;
static constexpr size_t SLICE_SC = (size_t)WAVES * SC_BYTES;
// OCC_COLD=1 gives every (block, rep) visit its own slice -- GRID*REPS of
// them, a 6.7 GB pool -- so no line is ever re-read and the arms are compared
// in the HBM-bound regime the real MoE tile runs in. The default 768-slice
// pool is 315 MB, nominally larger than the 256 MB MALL, but 21 concurrent
// blocks share each slice, so it measures ~28 TB/s and is a LATENCY test.
// Both regimes are worth having: the tile is 45% bytes / 55% not.
#ifndef OCC_COLD
#define OCC_COLD 0
#endif
static constexpr int NSLICES = OCC_COLD ? (GRID * REPS) : 768;

__host__ __device__ static inline uint32_t mix(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}
__host__ static inline uint8_t rand_fp8(uint32_t i) {
  uint32_t r = mix(i);
  uint32_t sign = (r >> 20) & 1u;
  uint32_t exp = 6u + ((r >> 8) & 3u);
  uint32_t man = (r >> 3) & 7u;
  return (uint8_t)((sign << 7) | (exp << 3) | man);
}
__host__ static inline uint8_t rand_e8m0(uint32_t i) {
  return (uint8_t)(126u + (mix(i ^ 0x9e37u) & 3u));
}

// DYN_KB is a launch-time argument, not a template parameter: both arms must
// be the SAME machine code, or a register or scheduling difference between
// them would be indistinguishable from the occupancy difference under test.
__global__ void __launch_bounds__(BLOCK_THREADS)
    occ_arm(uint8_t const *__restrict__ W,
            uint8_t const *__restrict__ WS,
            uint8_t const *__restrict__ TOK,
            uint8_t const *__restrict__ TOKSC,
            float *__restrict__ out) {
  extern __shared__ uint8_t s_dyn[]; // requested for residency, never touched
  __shared__ uint8_t s_tok[TOK_BYTES];
  __shared__ uint8_t s_tok_sc[TOKSC_BYTES];
  for (int i = threadIdx.x; i < TOK_BYTES; i += BLOCK_THREADS) {
    s_tok[i] = TOK[i];
  }
  for (int i = threadIdx.x; i < TOKSC_BYTES; i += BLOCK_THREADS) {
    s_tok_sc[i] = TOKSC[i];
  }
  __syncthreads();

  int const lane_id = threadIdx.x & (LANES - 1);
  int const wave = threadIdx.x >> 6;
  int const g = lane_id >> 4; // same derivation as the W13 call site

  f32x4_t sum = {0.f, 0.f, 0.f, 0.f};
  for (int r = 0; r < REPS; r++) {
    int const slice = (blockIdx.x * REPS + r) % NSLICES;
    uint8_t const *w = W + (size_t)slice * SLICE_W + (size_t)wave * W_BYTES;
    uint8_t const *ws = WS + (size_t)slice * SLICE_SC + (size_t)wave * SC_BYTES;
    f32x4_t acc = _gang_moe_kloop<WEIGHT_FP4, false, GROUPS_REQ, MFMA_ITERS,
                                  false>(w, ws, /*row_scale_base=*/0, s_tok,
                                         s_tok_sc, g);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      sum[i] += acc[i];
    }
  }

  // Sink the accumulator so nothing above is dead, without a store per rep.
  if (sum[0] == 12345.0f) {
    out[(blockIdx.x * BLOCK_THREADS + threadIdx.x) & 1023] = sum[1] + s_dyn[0];
  }
}

static float time_arm(char const *name, int dyn_kb, uint8_t const *W,
                      uint8_t const *WS, uint8_t const *TOK,
                      uint8_t const *TOKSC, float *out) {
  size_t const dyn = (size_t)dyn_kb * 1024;
  int occ = 0;
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &occ, (void const *)occ_arm, BLOCK_THREADS, dyn));

  // Warm up, then time five launches and take the minimum: the pool is
  // larger than MALL either way, so run-to-run spread here is scheduler
  // noise, not a cache-warmth artifact.
  hipLaunchKernelGGL(occ_arm, dim3(GRID), dim3(BLOCK_THREADS), dyn, 0, W, WS,
                     TOK, TOKSC, out);
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t t0, t1;
  HIP_CHECK(hipEventCreate(&t0));
  HIP_CHECK(hipEventCreate(&t1));
  float best = 1e30f;
  for (int it = 0; it < 5; it++) {
    HIP_CHECK(hipEventRecord(t0, 0));
    hipLaunchKernelGGL(occ_arm, dim3(GRID), dim3(BLOCK_THREADS), dyn, 0, W, WS,
                       TOK, TOKSC, out);
    HIP_CHECK(hipEventRecord(t1, 0));
    HIP_CHECK(hipEventSynchronize(t1));
    float ms = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
    if (ms < best) {
      best = ms;
    }
  }
  HIP_CHECK(hipEventDestroy(t0));
  HIP_CHECK(hipEventDestroy(t1));

  double const kloops = (double)GRID * WAVES * REPS;
  double const bytes = (double)GRID * REPS * SLICE_W;
  printf("  %-22s dyn=%3d KB  occ=%d blk/CU (%d wave/SIMD)  %8.3f ms  "
         "%7.3f us/kloop-wave  %6.2f TB/s\n",
         name, dyn_kb, occ, occ, best, best * 1e3 / kloops * GRID * WAVES,
         bytes / (best * 1e-3) / 1e12);
  return best;
}

int main() {
  printf("MoE k-loop occupancy probe: does 2 waves/SIMD help the tile?\n");
  printf("  grid=%d x %d thr, REPS=%d, MFMA_ITERS=%d, GROUPS=%d\n", GRID,
         BLOCK_THREADS, REPS, MFMA_ITERS, GROUPS_REQ);
  printf("  weight pool %.0f MB over %d slices (MALL is 256 MB)\n",
         (double)NSLICES * SLICE_W / 1e6, NSLICES);

  std::vector<uint8_t> hTOK(TOK_BYTES), hTOKSC(TOKSC_BYTES);
  for (int i = 0; i < TOK_BYTES; i++) {
    hTOK[i] = rand_fp8((uint32_t)i + 0x5000u);
  }
  for (int i = 0; i < TOKSC_BYTES; i++) {
    hTOKSC[i] = rand_e8m0((uint32_t)i + 0x600u);
  }

  size_t const wpool = (size_t)NSLICES * SLICE_W;
  size_t const scpool = (size_t)NSLICES * SLICE_SC;
  uint8_t *dW, *dWS, *dTOK, *dTOKSC;
  float *dOut;
  HIP_CHECK(hipMalloc(&dW, wpool));
  HIP_CHECK(hipMalloc(&dWS, scpool));
  HIP_CHECK(hipMalloc(&dTOK, TOK_BYTES));
  HIP_CHECK(hipMalloc(&dTOKSC, TOKSC_BYTES));
  HIP_CHECK(hipMalloc(&dOut, 1024 * sizeof(float)));

  // Fill one slice on the host and broadcast it: the VALUES do not matter
  // (nothing is checked here -- test_moe_kloop_width.hip owns correctness),
  // only that no byte is a NaN or a denormal that could change MFMA timing.
  {
    std::vector<uint8_t> hW(SLICE_W), hWS(SLICE_SC);
    for (size_t i = 0; i < SLICE_W; i++) {
      hW[i] = rand_fp8((uint32_t)i);
    }
    for (size_t i = 0; i < SLICE_SC; i++) {
      hWS[i] = rand_e8m0((uint32_t)i);
    }
    for (int s = 0; s < NSLICES; s++) {
      HIP_CHECK(hipMemcpy(dW + (size_t)s * SLICE_W, hW.data(), SLICE_W,
                          hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dWS + (size_t)s * SLICE_SC, hWS.data(), SLICE_SC,
                          hipMemcpyHostToDevice));
    }
  }
  HIP_CHECK(hipMemcpy(dTOK, hTOK.data(), TOK_BYTES, hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(dTOKSC, hTOKSC.data(), TOKSC_BYTES, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(dOut, 0, 1024 * sizeof(float)));

  printf("\n");
  float const a = time_arm("2 waves/SIMD (2 blk)", 64, dW, dWS, dTOK, dTOKSC,
                           dOut);
  float const b = time_arm("1 wave/SIMD  (1 blk)", 80, dW, dWS, dTOK, dTOKSC,
                           dOut);
  // Re-run A after B: if the two orderings disagree, the difference is drift
  // or thermal, not occupancy.
  float const a2 = time_arm("2 waves/SIMD (repeat)", 64, dW, dWS, dTOK, dTOKSC,
                            dOut);

  printf("\n  1-wave / 2-wave = %.3fx  (repeat %.3fx)\n", b / a, b / a2);
  printf("  A speedup above ~1.05x means 2 waves/SIMD is a real lever for the\n"
         "  MoE tile and the wide-block build is worth its 63-file cost.\n"
         "  Near 1.00x kills the whole axis for 10 s of GPU.\n");

  HIP_CHECK(hipFree(dW));
  HIP_CHECK(hipFree(dWS));
  HIP_CHECK(hipFree(dTOK));
  HIP_CHECK(hipFree(dTOKSC));
  HIP_CHECK(hipFree(dOut));
  return 0;
}
