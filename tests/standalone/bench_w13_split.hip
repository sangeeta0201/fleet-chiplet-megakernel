// Does W13 under MPK_MOE_TP get faster with more, shorter tiles?
//
// Runs the shipping deep k-loop (_gang_moe_kloop at depth 6) over W13-shaped
// MXFP4 tiles with the per-layer byte volume fixed at TP's 9 experts x 8 row
// blocks x 64 rows x K=6144 (15 MB/GPU), split three ways:
//   T=72  tiles x 48 k-steps per wave   (today)
//   T=144 tiles x 24 k-steps            (a two-way K split)
//   T=288 tiles x 12 k-steps            (four-way)
// Each block is one tile, 4 waves, one 16-row group per wave, exactly as the
// N-parallel W13 branch. Layers rotate through distinct buffers (1.1 GB) so
// nothing is cache-resident. Timing only; the accumulators are stored so the
// loop is not dead.
//
//   hipcc -O3 --offload-arch=gfx950 -I../../include -I../../include/mirage/persistent_kernel \
//     -DMIRAGE_BACKEND_USE_ROCM bench_w13_split.hip -o bench_w13_split
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <vector>

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
using namespace kernel;

#define HIP_CHECK(c)                                                           \
  do {                                                                         \
    hipError_t e_ = (c);                                                       \
    if (e_ != hipSuccess) {                                                    \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static constexpr int ROWS = 64;              // rows per tile (4 waves x 16)
static constexpr int K_FULL = 6144;
static constexpr int KT_FULL = K_FULL / 128;  // 48 k-tiles
static constexpr size_t LAYER_BYTES =
    (size_t)72 * ROWS * (K_FULL / 2 + K_FULL / 32);  // 15.0 MB
static constexpr int NLAYERS = 75;

// One tile = ROWS rows x KT k-tiles of K-major MXFP4 (data then scales), i.e.
// the same bytes per tile-k-step as the real W13 at any split.
template <int KT>
__global__ void __launch_bounds__(256, 1)
    w13_tiles(uint8_t const *__restrict__ layer0, float *__restrict__ out,
              int nlayers) {
 for (int l = 0; l < nlayers; l++) {
  uint8_t const *layer = layer0 + (size_t)l * LAYER_BYTES;
  constexpr int W_ROW_BYTES = KT * 64;
  constexpr int NB32 = KT * 4;
  constexpr int TILE_DATA = ROWS * W_ROW_BYTES;
  constexpr int TILE_BYTES = TILE_DATA + ROWS * NB32;
  __shared__ uint8_t s_tok[KT * 128 + 64];
  __shared__ uint8_t s_sc[KT + 16];
  for (int i = threadIdx.x; i < KT * 128; i += 256) s_tok[i] = 0x38;
  for (int i = threadIdx.x; i < KT; i += 256) s_sc[i] = 127;
  __syncthreads();
  int const tid = threadIdx.x, warp = tid >> 6, lane = tid & 63;
  int const col = lane & 15, g = lane >> 4;
  int const w_row = warp * 16 + col;
  uint8_t const *tile = layer + (size_t)blockIdx.x * TILE_BYTES;
  uint8_t const *w_data_row =
      tile + (w_row / 16) * (16 * W_ROW_BYTES) + (w_row % 16) * 64;
  uint8_t const *wg_scales = tile + TILE_DATA;
  int const row_scale_base = (w_row / 16) * (16 * NB32) + (w_row % 16) * 4;
  f32x4_t acc = _gang_moe_kloop<true, false, MPK_MOE_PF_GROUPS_W13, KT, false>(
      w_data_row, wg_scales, row_scale_base, s_tok, s_sc, g);
  if (col == 0) {
    out[(blockIdx.x * 4 + warp) * 4 + g] += acc[0] + acc[1] + acc[2] + acc[3];
  }
  __syncthreads();
 }
}

template <int KT>
static void run(uint8_t *buf, float *out, char const *tag) {
  constexpr int T = 72 * KT_FULL / KT;
  hipEvent_t a, b;
  HIP_CHECK(hipEventCreate(&a));
  HIP_CHECK(hipEventCreate(&b));
  for (int warm = 0; warm < 2; warm++) {
    w13_tiles<KT><<<T, 256>>>(buf, out, NLAYERS);
  }
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> ms;
  for (int rep = 0; rep < 5; rep++) {
    HIP_CHECK(hipEventRecord(a));
    w13_tiles<KT><<<T, 256>>>(buf, out, NLAYERS);
    HIP_CHECK(hipEventRecord(b));
    HIP_CHECK(hipEventSynchronize(b));
    float t;
    HIP_CHECK(hipEventElapsedTime(&t, a, b));
    ms.push_back(t);
  }
  float best = ms[0];
  for (float t : ms) best = t < best ? t : best;
  double us = best * 1000.0 / NLAYERS;
  printf("%-22s tiles=%3d k-steps/wave=%2d  %.2f us/layer  %.2f TB/s\n", tag,
         T, KT, us, LAYER_BYTES / (us * 1e-6) / 1e12);
}

int main() {
  uint8_t *buf;
  float *out;
  size_t const total = LAYER_BYTES * NLAYERS;
  HIP_CHECK(hipMalloc(&buf, total));
  HIP_CHECK(hipMalloc(&out, 288 * 4 * 4 * sizeof(float)));
  HIP_CHECK(hipMemset(buf, 0x22, total));  // FP4 1.0 nibbles, finite
  // Scales must be finite E8M0: every byte 0x22 = 34 is fine (2^-93).
  printf("W13 split benchmark, %.1f MB per layer, %d layers, depth %d\n",
         LAYER_BYTES / 1e6, NLAYERS, MPK_MOE_PF_GROUPS_W13);
  run<48>(buf, out, "72 x 48 (today)");
  run<24>(buf, out, "144 x 24 (2-way)");
  run<12>(buf, out, "288 x 12 (4-way)");
  run<48>(buf, out, "72 x 48 (again)");
  return 0;
}
