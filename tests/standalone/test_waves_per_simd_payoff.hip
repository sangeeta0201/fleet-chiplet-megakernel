// Does 2 waves/SIMD actually buy anything for GLM-shaped work?
//
// WHY. test_block_coresidency.hip established that gfx950 gives 2 blocks/CU at
// <= 256 unified VGPRs and 1 at >= 262, and that GLM's worker_kernel sits at
// 284. Getting under the cliff means register surgery on mla_decode_absorbed,
// AND raising the worker count past 256 blocks, and neither pays without the
// other -- a big-bang restructure. Before spending turns on it, price the
// payoff.
//
// THE DESIGN AVOIDS THE REGISTER QUESTION ENTIRELY. Occupancy is set by the
// GRID, not by register pressure: the same binary, the same total work, run at
// gridDim = 1 x CUs and 2 x CUs. Both kernels here are low-register, so the
// hardware will happily place 2 blocks/CU when asked. That makes this a strict
// one-variable test of "1 block/CU vs 2 blocks/CU at fixed total work", with
// no ballast array perturbing the inner loop's scheduling.
//
// TWO SHAPES, because the layer has two kinds of phase:
//
//   stream : HBM-bound, dwordx4 loads, no dependent chain. This is the MoE
//            W13/W2 tile, measured at 74% of the per-CU byte roof. If the
//            phase is genuinely at the byte roof, a second block per CU cannot
//            help -- the bytes are already the limit.
//   chain  : dependent-FMA bound, tiny footprint. This stands in for the
//            depwait half of the layer -- "50% of the machine is depwait" --
//            where a second wave per SIMD is exactly the mechanism that would
//            fill the stall slots.
//
// A win on `chain` and a null on `stream` would say the occupancy ladder pays
// only on the attention phases; a null on both kills it outright.
//
// Run: ./test_waves_per_simd_payoff
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

// ── HBM-bound: grid-stride over fixed-size tiles, UNROLL x 16 B per lane per
// trip. UNROLL is the CONFOUND CONTROL. A single-load grid-stride at 1 block
// per CU may simply not keep enough loads in flight to saturate HBM -- the
// house rule is >= 8 -- in which case a "2 blocks/CU wins" result would really
// be "more loads in flight wins", and the fix would be unrolling the existing
// tile loops rather than the whole occupancy restructure. Sweeping UNROLL at
// FIXED grid separates the two: if UNROLL=8 at 1 blk/CU reaches the same
// bandwidth as UNROLL=1 at 2 blk/CU, the occupancy ladder is not needed for
// the byte-bound phases. ──
template <int UNROLL>
__global__ void __launch_bounds__(256)
    stream_tiles(uint4 const *__restrict__ w, float *sink, int tiles_total,
                 int tile_u4) {
  for (int t = blockIdx.x; t < tiles_total; t += gridDim.x) {
    uint4 const *base = w + (size_t)t * (size_t)tile_u4;
    uint4 acc = make_uint4(0u, 0u, 0u, 0u);
    int const step = 256 * UNROLL;
    for (int i = threadIdx.x; i < tile_u4; i += step) {
      uint4 v[UNROLL];
      // All UNROLL addresses are known before any use, so these issue
      // back-to-back and sit in flight together.
#pragma unroll
      for (int u = 0; u < UNROLL; u++) {
        int idx = i + u * 256;
        v[u] = base[idx < tile_u4 ? idx : tile_u4 - 1];
      }
#pragma unroll
      for (int u = 0; u < UNROLL; u++) {
        acc.x += v[u].x;
        acc.y += v[u].y;
        acc.z += v[u].z;
        acc.w += v[u].w;
      }
    }
    if (acc.x == 0xdeadbeefu && acc.y == 0xdeadbeefu && acc.z == 0xdeadbeefu &&
        acc.w == 0xdeadbeefu) {
      sink[0] = (float)acc.x;
    }
  }
}

// ── Dependent-FMA bound: one serial chain per lane, nothing to overlap
// within the wave, so the only way to fill the SIMD is another wave. ──
__global__ void __launch_bounds__(256)
    chain_fma(float const *__restrict__ seed, float *sink, int units,
              int depth) {
  for (int u = blockIdx.x; u < units; u += gridDim.x) {
    float a = seed[threadIdx.x & 255] + (float)u;
    float b = 1.0000001f;
#pragma unroll 8
    for (int i = 0; i < depth; i++) {
      a = __builtin_fmaf(a, b, 1.0f); // strictly serial
    }
    if (a == 1.2345678e-31f) {
      sink[0] = a;
    }
  }
}

struct Res {
  float ms;
};

template <typename F> static Res timeit(F launch, int iters) {
  hipEvent_t e0, e1;
  hipEventCreate(&e0);
  hipEventCreate(&e1);
  launch(); // warm
  hipDeviceSynchronize();
  hipEventRecord(e0);
  for (int i = 0; i < iters; i++) {
    launch();
  }
  hipEventRecord(e1);
  hipEventSynchronize(e1);
  float ms = 0.f;
  hipEventElapsedTime(&ms, e0, e1);
  hipEventDestroy(e0);
  hipEventDestroy(e1);
  return Res{ms / iters};
}

int main() {
  hipDeviceProp_t p;
  HIP_CHECK(hipGetDeviceProperties(&p, 0));
  int const cus = p.multiProcessorCount;
  printf("device %s, %d CUs\n\n", p.gcnArchName, cus);

  // ── stream ──────────────────────────────────────────────────────────────
  // 192 KB per tile, close to a GLM MoE W13 tile's weight slab.
  int const tile_u4 = 192 * 1024 / 16;
  int const tiles_total = 8 * cus; // fixed total work, divisible by both grids
  size_t const bytes = (size_t)tiles_total * (size_t)tile_u4 * 16ull;

  uint4 *w;
  float *sink;
  HIP_CHECK(hipMalloc(&w, bytes));
  HIP_CHECK(hipMalloc(&sink, sizeof(float)));
  HIP_CHECK(hipMemset(w, 1, bytes));
  HIP_CHECK(hipMemset(sink, 0, sizeof(float)));

  printf("stream: %d tiles x 192 KB = %.1f MB total, fixed across arms\n",
         tiles_total, bytes / 1048576.0);
  auto run_stream = [&](auto kern, int unroll, int mult) {
    int grid = cus * mult;
    Res r = timeit(
        [&] {
          hipLaunchKernelGGL(kern, dim3(grid), dim3(256), 0, 0, w, sink,
                             tiles_total, tile_u4);
        },
        20);
    printf("  unroll=%d  grid=%4d (%d blk/CU)  %8.3f ms   %7.2f GB/s\n", unroll,
           grid, mult, r.ms, bytes / (r.ms * 1e-3) / 1e9);
  };
  run_stream(stream_tiles<1>, 1, 1);
  run_stream(stream_tiles<1>, 1, 2);
  run_stream(stream_tiles<2>, 2, 1);
  run_stream(stream_tiles<4>, 4, 1);
  run_stream(stream_tiles<8>, 8, 1);
  run_stream(stream_tiles<8>, 8, 2);

  // ── chain ───────────────────────────────────────────────────────────────
  float *seed;
  HIP_CHECK(hipMalloc(&seed, 256 * sizeof(float)));
  HIP_CHECK(hipMemset(seed, 0, 256 * sizeof(float)));
  int const units = 8 * cus;
  int const depth = 4096;

  printf("\nchain: %d units x %d serial FMAs, fixed across arms\n", units,
         depth);
  for (int mult = 1; mult <= 2; mult++) {
    int grid = cus * mult;
    Res r = timeit(
        [&] {
          hipLaunchKernelGGL(chain_fma, dim3(grid), dim3(256), 0, 0, seed, sink,
                             units, depth);
        },
        20);
    printf("  grid=%4d (%d blk/CU)  %8.3f ms\n", grid, mult, r.ms);
  }

  printf("\nRead: the second row of each block is what the whole occupancy\n"
         "ladder (28 registers off mla_decode_absorbed, plus doubling the\n"
         "worker count) would buy for that shape. A null on both shapes kills\n"
         "the ladder outright.\n");
  return 0;
}
