/* Copyright 2025 CMU
 *
 * Does a second co-resident block per CU actually help a GLM-shaped tile?
 *
 * Memory says the whole occupancy ladder is gated on an unpriced assumption:
 * that going from 1 block/CU to 2 blocks/CU buys latency hiding worth ~1.8 ms
 * on the wall. That was never measured -- it was inferred from round counts.
 * Every attempt to get there in the megakernel has run into register or
 * scratch gates, so the payoff has to be priced somewhere those gates do not
 * apply.
 *
 * The trick here is to use *dynamic LDS as the co-residency knob*. The kernel
 * body is byte-identical across arms -- same code object, same VGPR count,
 * same instruction schedule -- and only the launch's dynamic shared-memory
 * request changes. A large request starves the CU down to one block; a small
 * one lets the hardware pack as many as the registers allow. Total work, grid
 * size, and memory traffic are held fixed. So any wall-clock difference is
 * co-residency and nothing else.
 *
 * The body is the real gang_moe_w13_linear_mxfp8_kernel, which is the shape
 * memory characterizes as HBM 32.7% of roof / MFMA 2.65% / 68% vmcnt -- i.e.
 * latency-bound with neither unit saturated. That is exactly the regime where
 * a second wave should help if it is ever going to.
 *
 * Numerics are deliberately not checked: the weights are a memset byte
 * pattern. This is a throughput probe, and MFMA/HBM timing on gfx950 does not
 * depend on operand values. test_mxfp8_moe.hip owns correctness for this
 * kernel.
 *
 * Build: ./build.sh
 * Run:   ./test_occupancy_payoff
 */

#include <cstdio>
#include <cstdlib>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <vector>

#define MPK_WS_WAVE_SYNC(wave) ((void)0)
namespace kernel {
__device__ __forceinline__ int _gang_moe_get_xcd_id() {
  return (int)blockIdx.y;
}
} // namespace kernel

#include "mirage/persistent_kernel/mpk_atoms.cuh"
#include "mirage/persistent_kernel/tasks/mi300/gang_moe_linear_mxfp8_mi300.cuh"

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t err = call;                                                     \
    if (err != hipSuccess) {                                                   \
      fprintf(stderr,                                                          \
              "HIP error %s:%d: %s\n",                                         \
              __FILE__,                                                        \
              __LINE__,                                                        \
              hipGetErrorString(err));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

using namespace kernel;

// Same tile geometry as test_mxfp8_moe.hip -- OPW=64 output rows per block,
// K=HIDDEN -- but with enough activated experts to fill the device several
// times over, so wave count is the thing being varied.
static constexpr int HIDDEN = 2048;
static constexpr int INTER = 1536;
static constexpr int NUM_EXPERTS = 64;
static constexpr int NUM_TOPK = 33;
static constexpr int BATCH = 1;
static constexpr int OPW = 64;

static constexpr int W13_N = 2 * INTER;     // 3072
static constexpr int W13_K = HIDDEN;        // 2048
static constexpr int W13_WGS = W13_N / OPW; // 48
static constexpr int W13_TPE = BATCH * W13_WGS;

static constexpr int SMEM_BYTES = W13_K + W13_K / 128 + 256;

static constexpr size_t wg_bytes(int k) {
  return (size_t)OPW * k + (size_t)OPW * (k / 32);
}

__global__ void run_w13(void const *input,
                        void const *weight,
                        void const *routing,
                        void const *mask,
                        void const *bias,
                        void *output) {
  gang_moe_w13_linear_mxfp8_kernel<BATCH,
                                   W13_N,
                                   W13_N,
                                   W13_K,
                                   NUM_EXPERTS,
                                   NUM_TOPK,
                                   W13_TPE,
                                   OPW,
                                   true /*FUSE_SWIGLU*/>(
      input, weight, routing, mask, bias, output, (int)blockIdx.x);
}

// The megakernel does NOT launch one block per tile: it runs a fixed pool of
// persistent workers, each grid-striding over its XCD's tile list. That loop
// is a second way to hide the same latency -- tile t+1's loads can in
// principle issue under tile t's tail without any co-residency at all. So the
// flat one-block-per-tile arm above overstates what occupancy is worth to us.
// This wrapper reproduces the real shape: gridDim.x blocks, each looping.
__global__ void run_w13_stride(void const *input,
                               void const *weight,
                               void const *routing,
                               void const *mask,
                               void const *bias,
                               void *output,
                               int tiles_per_xcd) {
  for (int t = (int)blockIdx.x; t < tiles_per_xcd; t += (int)gridDim.x) {
    gang_moe_w13_linear_mxfp8_kernel<BATCH,
                                     W13_N,
                                     W13_N,
                                     W13_K,
                                     NUM_EXPERTS,
                                     NUM_TOPK,
                                     W13_TPE,
                                     OPW,
                                     true /*FUSE_SWIGLU*/>(
        input, weight, routing, mask, bias, output, t);
  }
}

int main() {
  int const num_act = NUM_TOPK;
  int const total_tiles = num_act * W13_TPE;
  int const tiles_per_xcd = (total_tiles + 7) / 8;
  int const nblocks = tiles_per_xcd * 8;

  // Routing tables are the only thing that has to be structurally valid: a
  // block whose expert is not activated takes an early exit and stops being a
  // memory-traffic sample.
  std::vector<int> h_mask(NUM_EXPERTS + 1, 0),
      h_routing((size_t)NUM_EXPERTS * BATCH, 0);
  for (int s = 0; s < num_act; s++) {
    h_mask[s] = s;
    h_routing[(size_t)s * BATCH + 0] = s + 1;
  }
  h_mask[NUM_EXPERTS] = num_act;

  size_t const w13_packed = (size_t)NUM_EXPERTS * W13_WGS * wg_bytes(W13_K);

  unsigned short *d_x, *d_b13, *d_act;
  uint8_t *d_w13;
  int *d_mask, *d_routing;
  HIP_CHECK(hipMalloc(&d_x, W13_K * 2));
  HIP_CHECK(hipMalloc(&d_b13, (size_t)NUM_EXPERTS * W13_N * 2));
  HIP_CHECK(hipMalloc(&d_w13, w13_packed));
  HIP_CHECK(hipMalloc(&d_act, (size_t)BATCH * NUM_TOPK * INTER * 2));
  HIP_CHECK(hipMalloc(&d_mask, h_mask.size() * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_routing, h_routing.size() * sizeof(int)));

  // 0x3C is a finite E4M3 mantissa byte and a finite E8M0 exponent byte, so
  // neither denormal nor NaN paths are exercised.
  HIP_CHECK(hipMemset(d_w13, 0x3C, w13_packed));
  HIP_CHECK(hipMemset(d_x, 0x3C, W13_K * 2));
  HIP_CHECK(hipMemset(d_b13, 0, (size_t)NUM_EXPERTS * W13_N * 2));
  HIP_CHECK(hipMemset(d_act, 0, (size_t)BATCH * NUM_TOPK * INTER * 2));
  HIP_CHECK(hipMemcpy(d_mask,
                      h_mask.data(),
                      h_mask.size() * sizeof(int),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_routing,
                      h_routing.data(),
                      h_routing.size() * sizeof(int),
                      hipMemcpyHostToDevice));

  hipFuncAttributes attr;
  HIP_CHECK(hipFuncGetAttributes(&attr, (void const *)run_w13));

  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));

  printf("device            : %s, %d CUs\n", prop.name, prop.multiProcessorCount);
  printf("kernel            : run_w13 (gang_moe_w13_linear_mxfp8_kernel)\n");
  printf("  static LDS      : %zu B\n", (size_t)attr.sharedSizeBytes);
  printf("  VGPRs           : %d   spills: %zu B\n",
         attr.numRegs,
         (size_t)attr.localSizeBytes);
  printf("  LDS/CU limit    : %zu B\n", (size_t)prop.maxSharedMemoryPerMultiProcessor);
  printf("grid              : %d blocks x 256 thr  (%d tiles, %d per XCD)\n",
         nblocks,
         total_tiles,
         tiles_per_xcd);
  printf("  blocks per CU   : %.2f\n", (double)nblocks / prop.multiProcessorCount);

  // Weight bytes streamed by one pass: every activated (expert, tile) pair
  // reads its own OPW x K slab exactly once.
  double const pass_bytes = (double)total_tiles * (double)wg_bytes(W13_K);
  printf("  weight bytes    : %.1f MB per pass\n", pass_bytes / 1e6);

  dim3 const grid(tiles_per_xcd, 8);

  int const ITERS = 200;
  int const WARMUP = 20;

  // Sweep the dynamic-LDS request. Everything else is pinned. Values are
  // chosen to straddle the co-residency cutovers rather than to be round.
  int const lds_arms[] = {
      SMEM_BYTES, 12000, 18000, 26000, 40000, 56000, 84000, 160000};
  int const n_arms = sizeof(lds_arms) / sizeof(lds_arms[0]);

  printf("\n  dyn LDS     blk/CU     ms/pass     GB/s     vs 1blk/CU\n");
  printf("  ---------------------------------------------------------\n");

  double baseline_ms = 0.0; // time of the most-starved arm that actually ran
  int baseline_occ = 0;
  double best_ms = 1e30;
  int best_occ = 0;

  // Walk from the most-starved arm backwards so the 1 blk/CU baseline is
  // established before anything is compared against it.
  for (int i = n_arms - 1; i >= 0; i--) {
    int const dyn = lds_arms[i];

    int occ = 0;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ, (void const *)run_w13, 256, (size_t)dyn));

    hipEvent_t ev0, ev1;
    HIP_CHECK(hipEventCreate(&ev0));
    HIP_CHECK(hipEventCreate(&ev1));

    for (int it = 0; it < WARMUP; it++) {
      run_w13<<<grid, 256, dyn>>>(
          d_x, d_w13, d_routing, d_mask, d_b13, d_act);
    }
    // A dynamic-LDS request the driver rejects makes every launch a silent
    // no-op, which would time as a spectacular speedup. Catch it here.
    hipError_t const launch_err = hipGetLastError();
    if (launch_err != hipSuccess) {
      printf("  %7d   REJECTED (%s)\n", dyn, hipGetErrorString(launch_err));
      HIP_CHECK(hipEventDestroy(ev0));
      HIP_CHECK(hipEventDestroy(ev1));
      continue;
    }
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipEventRecord(ev0));
    for (int it = 0; it < ITERS; it++) {
      run_w13<<<grid, 256, dyn>>>(
          d_x, d_w13, d_routing, d_mask, d_b13, d_act);
    }
    HIP_CHECK(hipEventRecord(ev1));
    HIP_CHECK(hipEventSynchronize(ev1));

    float ms_total = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms_total, ev0, ev1));
    HIP_CHECK(hipEventDestroy(ev0));
    HIP_CHECK(hipEventDestroy(ev1));

    double const ms = ms_total / ITERS;
    double const gbs = pass_bytes / (ms * 1e-3) / 1e9;

    if (baseline_ms == 0.0) {
      baseline_ms = ms;
      baseline_occ = occ;
    }
    if (ms < best_ms) {
      best_ms = ms;
      best_occ = occ;
    }

    if (baseline_ms > 0.0) {
      printf("  %7d   %6d   %9.4f   %7.1f     %+6.1f%%\n",
             dyn,
             occ,
             ms,
             gbs,
             100.0 * (baseline_ms - ms) / baseline_ms);
    } else {
      printf("  %7d   %6d   %9.4f   %7.1f          --\n", dyn, occ, ms, gbs);
    }
  }

  // ── the megakernel-shaped arm ────────────────────────────────────────────
  // Same total tile work, but delivered by a persistent-style pool that
  // grid-strides. Sweeping the pool width against the LDS knob separates
  // "more blocks resident" from "more tiles per block".
  printf("\n  persistent-pool arm (each block grid-strides over its tiles)\n");
  printf("  blocks   dyn LDS   blk/CU   tiles/blk    ms/pass     GB/s\n");
  printf("  ----------------------------------------------------------\n");

  struct PoolArm {
    int bx;  // blocks per XCD
    int dyn;
  };
  PoolArm const pool_arms[] = {
      {32, 160000},  // 256 blocks, forced 1/CU  <- the shipping shape
      {32, SMEM_BYTES},
      {64, 160000},  // 512 blocks, forced 1/CU (2 sequential waves)
      {64, SMEM_BYTES},   // 512 blocks, 2/CU
      {128, SMEM_BYTES},  // 1024 blocks, 4/CU
  };
  double pool_best = 1e30;
  for (auto const &a : pool_arms) {
    int occ = 0;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ, (void const *)run_w13_stride, 256, (size_t)a.dyn));
    dim3 const g(a.bx, 8);

    for (int it = 0; it < WARMUP; it++) {
      run_w13_stride<<<g, 256, a.dyn>>>(
          d_x, d_w13, d_routing, d_mask, d_b13, d_act, tiles_per_xcd);
    }
    hipError_t const e = hipGetLastError();
    if (e != hipSuccess) {
      printf("  %6d   REJECTED (%s)\n", a.bx * 8, hipGetErrorString(e));
      continue;
    }
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t p0, p1;
    HIP_CHECK(hipEventCreate(&p0));
    HIP_CHECK(hipEventCreate(&p1));
    HIP_CHECK(hipEventRecord(p0));
    for (int it = 0; it < ITERS; it++) {
      run_w13_stride<<<g, 256, a.dyn>>>(
          d_x, d_w13, d_routing, d_mask, d_b13, d_act, tiles_per_xcd);
    }
    HIP_CHECK(hipEventRecord(p1));
    HIP_CHECK(hipEventSynchronize(p1));
    float mt = 0.f;
    HIP_CHECK(hipEventElapsedTime(&mt, p0, p1));
    HIP_CHECK(hipEventDestroy(p0));
    HIP_CHECK(hipEventDestroy(p1));
    double const ms = mt / ITERS;
    if (ms < pool_best) {
      pool_best = ms;
    }
    printf("  %6d   %7d   %6d   %9.2f   %8.4f   %7.1f\n",
           a.bx * 8,
           a.dyn,
           occ,
           (double)tiles_per_xcd / a.bx,
           ms,
           pass_bytes / (ms * 1e-3) / 1e9);
  }

  printf("\n");
  printf("baseline arm: %d block(s)/CU at %.4f ms\n", baseline_occ, baseline_ms);
  if (baseline_ms == 0.0 || baseline_occ == best_occ) {
    printf("VOID: the LDS knob never separated the arms -- every launch got "
           "%d blocks/CU. Raise the largest dyn LDS value.\n", best_occ);
  } else {
    double const gain = 100.0 * (baseline_ms - best_ms) / baseline_ms;
    printf("best arm: %d blocks/CU at %.4f ms, %+.1f%% vs 1 block/CU\n",
           best_occ,
           best_ms,
           gain);
    if (gain < 3.0) {
      printf("VERDICT: %d blocks/CU is worth %.1f%% -- inside noise. The "
             "occupancy ladder is not worth register surgery.\n",
             best_occ,
             gain);
    } else {
      printf("VERDICT: %d blocks/CU is worth %.1f%%. The ladder is ALIVE; the "
             "VGPR hunt is justified.\n",
             best_occ,
             gain);
    }
  }
  return 0;
}
