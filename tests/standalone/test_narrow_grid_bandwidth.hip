// What can a FEW CUs pull, and how deep must the pipeline be to get it?
//
// WHY THIS TEST EXISTS. Every roofline number on this branch divides HBM by
// the full CU count: "the 20.2 GB/s per-CU share", "the W13 tile is at 74% of
// the per-CU byte roof", "qkv_a's MFMA K-loop is at 90% of the per-CU byte
// roof". 20.2 = 5.17 TB/s / 256 CUs. That denominator is only correct when all
// 256 CUs are fetching at once.
//
// GLM never does that. Per-phase occupancy is the whole story: W13 runs ~8 of
// 29 workers per XCD, W2 ~12, mla_decode 16 of 232, merge 16 of 232. When 64
// of 256 CUs are the only ones touching HBM, each of them is entitled to far
// more than a 1/256 share, and the ceiling is not the divided share at all --
// it is that CU's own outstanding-request throughput. Little's law: delivered
// bytes = outstanding bytes / latency. Cut the number of fetching CUs by 4x
// and you cut outstanding bytes by 4x, so the per-CU rate should hold and the
// AGGREGATE should collapse -- unless the loop is deep enough to keep the
// missing requests in flight itself.
//
// Which means the depth question inverts with grid width. The companion
// test_waves_per_simd_payoff.hip found depth stops mattering past ~4 loads in
// flight -- but that ran at grid = 256, where 256 CUs x 4 in flight is already
// far more than HBM needs. At grid = 64 the same depth-4 loop has a quarter of
// the requests outstanding and may be nowhere near the roof.
//
// THIS MATTERS BECAUSE OF WHAT IT WOULD MAKE TRUE OF THE REAL KERNEL. GLM's
// gang_moe_w13_linear_mxfp8_kernel runs a depth-4 software pipeline: a0..a3
// prefetched one k-group ahead, 32 B each. If per-CU bandwidth at a narrow
// grid keeps climbing past depth 4, then W13's "74% of roof" was measured
// against a roof that does not apply, the tile is much further from its real
// ceiling than believed, and the lever is deepening the existing pipeline --
// a correct-output, in-place tile change, which the ledger says transfers at
// about 1:1 ([[glm-resadd-unroll-is-marginal]]).
//
// THE MEASUREMENT. One kernel, one binary. Sweep grid (how many CUs fetch) x
// UNROLL (how many loads each keeps in flight) with the TOTAL BYTES HELD
// FIXED, so every cell moves the same data and the only variables are width
// and depth. Report per-CU GB/s, because that is the number the roofline
// arithmetic actually wants.
//
// The footprint is deliberately larger than the 256 MB MALL so that repeat
// iterations cannot be served from cache and every arm is measuring HBM.
//
// Run: ./test_narrow_grid_bandwidth
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

// Grid-stride over fixed-size tiles, UNROLL x 16 B per lane per trip. All
// UNROLL addresses are computed before any value is used, so they issue
// back-to-back and sit in flight together; the adds below are what forces the
// s_waitcnt, and they come after every load has been issued.
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
    // Impossible sink: nothing above is dead, nothing is hoisted.
    if (acc.x == 0xdeadbeefu && acc.y == 0xdeadbeefu && acc.z == 0xdeadbeefu &&
        acc.w == 0xdeadbeefu) {
      sink[0] = (float)acc.x;
    }
  }
}

template <typename F> static float timeit(F launch, int iters) {
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
  return ms / iters;
}

static uint4 *g_w;
static float *g_sink;
static int g_tiles, g_tile_u4;
static size_t g_bytes;

template <int UNROLL> static float cell(int grid, int iters) {
  return timeit(
      [&] {
        hipLaunchKernelGGL((stream_tiles<UNROLL>), dim3(grid), dim3(256), 0, 0,
                           g_w, g_sink, g_tiles, g_tile_u4);
      },
      iters);
}

int main() {
  hipDeviceProp_t p;
  HIP_CHECK(hipGetDeviceProperties(&p, 0));
  int const cus = p.multiProcessorCount;
  printf("device %s, %d CUs\n", p.gcnArchName, cus);

  // 192 KB per tile, close to a GLM MoE W13 tile's weight slab (208 KB).
  g_tile_u4 = 192 * 1024 / 16;
  g_tiles = 6144; // 1.125 GB, comfortably past the MALL
  g_bytes = (size_t)g_tiles * (size_t)g_tile_u4 * 16ull;

  HIP_CHECK(hipMalloc(&g_w, g_bytes));
  HIP_CHECK(hipMalloc(&g_sink, sizeof(float)));
  HIP_CHECK(hipMemset(g_w, 1, g_bytes));
  HIP_CHECK(hipMemset(g_sink, 0, sizeof(float)));

  printf("%d tiles x 192 KB = %.2f GB, FIXED across every cell\n", g_tiles,
         g_bytes / 1073741824.0);
  printf("(footprint > MALL, so repeats cannot be cache-served)\n\n");

  int const grids[] = {16, 32, 64, 96, 128, 232, 256};
  int const ngrids = sizeof(grids) / sizeof(grids[0]);

  printf("aggregate GB/s (and per-fetching-CU GB/s in brackets)\n");
  printf("  %-22s", "grid (blocks=CUs)");
  for (int d = 0; d < 4; d++) {
    printf("%18s", d == 0   ? "depth 1"
                   : d == 1 ? "depth 2"
                   : d == 2 ? "depth 4"
                            : "depth 8");
  }
  printf("\n");

  for (int gi = 0; gi < ngrids; gi++) {
    int const grid = grids[gi];
    // Narrow arms move the same bytes on fewer CUs, so they take longer per
    // iteration; fewer iterations keeps the sweep to a few seconds without
    // changing what is measured.
    int const iters = grid <= 32 ? 3 : (grid <= 96 ? 5 : 10);
    char const *tag = grid == 64    ? " <- W13 live"
                      : grid == 96  ? " <- W2 live"
                      : grid == 232 ? " <- all workers"
                                    : "";
    printf("  %4d%-18s", grid, tag);
    float ms[4];
    ms[0] = cell<1>(grid, iters);
    ms[1] = cell<2>(grid, iters);
    ms[2] = cell<4>(grid, iters);
    ms[3] = cell<8>(grid, iters);
    for (int d = 0; d < 4; d++) {
      double agg = g_bytes / (ms[d] * 1e-3) / 1e9;
      printf("%11.0f [%5.1f]", agg, agg / grid);
    }
    printf("\n");
  }

  printf("\nRead the DEPTH-4 column down. That is the depth GLM's W13 k-loop\n"
         "actually runs. If its per-CU number at grid=64 is well under the\n"
         "same row's depth-8 number, the real MoE tiles are latency-bound at\n"
         "their current pipeline depth and deepening them is a correct-output\n"
         "in-place win. If depth 4 and depth 8 agree at grid=64, the tiles are\n"
         "at their true per-CU ceiling and this whole direction is closed.\n");
  return 0;
}
