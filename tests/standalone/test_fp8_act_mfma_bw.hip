/* Copyright 2025 CMU
 *
 * Does an FP8 activation buy anything over the dequant-to-bf16 VALU chain?
 *
 * The three GLM stages still on gang_gemv_mxfp8_kernel -- o_proj, W_UV, W_UK --
 * read MXFP8 weights but keep the activation in bf16, expanding the weight one
 * pair at a time with v_cvt_scalef32_pk_bf16_fp8 and accumulating with
 * v_dot2c_f32_bf16. Counting the shipped gfx950 ISA at the o_proj tile:
 *
 *     42  v_cvt_scalef32_pk_bf16_fp8
 *     40  v_dot2c_f32_bf16
 *     12  ds_read_b128            (activation, staged)
 *      8  global_load_dwordx4     (128 B/lane of weight)
 *
 * ~82 VALU ops per 128 bytes of weight per lane. The alternative is to quantize
 * the activation to FP8 too and let the MFMA do the dequant in hardware.
 *
 * On gfx950 that is the ONLY way to get an FP8 activation. v_dot4_f32_fp8_fp8
 * and its siblings need target feature dot11-insts, which gfx942 has and
 * gfx950 does not:
 *
 *     error: '__builtin_amdgcn_dot4_f32_fp8_fp8' needs target feature dot11-insts
 *
 * so there is no VALU fp8 dot to fall back on -- the scaled MFMA is the whole
 * menu. That is also what gpt-oss does on every GEMM it owns.
 *
 * ── the two arguments, which point opposite ways ───────────────────────────
 *
 * For: a CDNA SIMD is 16 lanes, so a wave64 VALU op holds it for 4 cycles.
 * v_dot2c_f32_bf16 is 64 lanes x 2 MACs / 4 cycles, halved again by the paired
 * cvt, so 16 useful MACs/cycle/SIMD. v_mfma_scale_f32_16x16x128_f8f6f4 retires
 * 32768 MACs in ~32 cycles; at batch 1 fifteen of its sixteen N columns are
 * waste, but the M dimension carries 16 real output rows, so 16*128/32 = 64
 * useful MACs/cycle/SIMD survives. 4x on the arithmetic.
 *
 * Against: per-CU that 4x is spare capacity nobody is asking for. The current
 * loop sustains ~154 GB/s of weight per CU on VALU alone against a 20.2 GB/s
 * per-CU share of a 5.17 TB/s HBM, so the dequant has ~7.6x of headroom and
 * deleting it should be invisible.
 *
 * What makes it worth measuring anyway is the gap test_gemv_mxfp8_bw already
 * documented on this box: kernel A streams at 37.8 GB/s/WG while the best GEMV
 * variant reaches 18.7, and deeper unrolling (its kernel H) did not move it, so
 * it is not memory-level parallelism. "Not MLP-limited but still half speed" is
 * consistent with VALU issue slots crowding out load issue, and if that is what
 * the gap is then cutting VALU instructions 4x is exactly what shows up.
 *
 * ── why this is a separate file from test_gemv_mxfp8_bw ────────────────────
 *
 * That harness no longer compiles, for reasons unrelated to any of this:
 * gang_gemv_mxfp8_mi300.cuh -> gang_gemv_mi300.cuh -> gang_linear_mi300.cuh
 * pulls in linear_ck_mi300.cuh, and ck_tile's BlockUniversalGemmAsBsCr has
 * since gained a non-const operator() that its own GemmPipelineAGmemBGmemCRegV2
 * still calls through a const object -- 10 errors, all inside ROCm 7.0.0's
 * headers, present at HEAD before this file existed. gang_linear_mxfp8_mi300.cuh
 * is CK-free, so the MFMA side builds standalone; the bf16-dot comparand is the
 * number that harness already recorded on this same box.
 *
 * Kernel A is reproduced here verbatim so the roofline is re-measured rather
 * than quoted -- if it does not come back at ~37.8 GB/s/WG, the box has moved
 * and the recorded GEMV numbers are not comparable either.
 *
 * Build: hipcc -o test_fp8_act_mfma_bw test_fp8_act_mfma_bw.hip \
 *          -D__HIP_PLATFORM_AMD__ -DMIRAGE_BACKEND_USE_ROCM \
 *          --offload-arch=gfx950 -munsafe-fp-atomics -O3 -std=c++17 \
 *          -I ../../include -I ../../include/mirage/persistent_kernel
 * Run:   ./test_fp8_act_mfma_bw
 */

#include <cstdio>
#include <cstdlib>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

// Same two stubs test_mxfp8_linear.hip supplies, for the same reason: inside
// the megakernel they come from headers included ahead of the task bodies.
#define MPK_WS_WAVE_SYNC(wave) ((void)0)
namespace kernel {
__device__ __forceinline__ int _gang_moe_get_xcd_id() {
  int xcd_id;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcd_id));
  return xcd_id;
}
} // namespace kernel

#include "mirage/persistent_kernel/mpk_atoms.cuh"
#include "mirage/persistent_kernel/tasks/mi300/gang_linear_mxfp8_mi300.cuh"

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

// ───────────────────────────────────────────────────────────────────────────
// Kernel A: pure streaming read, UNROLL dwordx4 loads in flight per thread.
// Verbatim from test_gemv_mxfp8_bw.hip -- this is the roofline the MFMA number
// gets read against, and re-measuring it is what makes the comparison legal.
// ───────────────────────────────────────────────────────────────────────────
template <int UNROLL>
__global__ __launch_bounds__(256) void stream_read_kernel(
    uint4 const *__restrict__ src, size_t n_vec, unsigned *sink) {
  size_t const stride = (size_t)gridDim.x * 256 * UNROLL;
  size_t const base0 = (size_t)blockIdx.x * 256 * UNROLL + threadIdx.x;
  unsigned acc = 0;
  for (size_t b = base0; b < n_vec; b += stride) {
    uint4 v[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; u++) {
      size_t const i = b + (size_t)u * 256;
      v[u] = src[i < n_vec ? i : n_vec - 1];
    }
#pragma unroll
    for (int u = 0; u < UNROLL; u++) {
      acc += v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    }
  }
  if (acc == 0xdeadbeefu) {
    sink[0] = acc;
  }
}

// ───────────────────────────────────────────────────────────────────────────
// Kernel I: the same reduction, FP8 activation, FP8xFP8 scaled MFMA.
//
// gang_linear_mxfp8_kernel already IS this kernel -- same pack_dense_mxfp8
// weight layout the GEMV reads, activation quantized into LDS by
// _gang_wave_parallel_fp8_quant, v_mfma_scale_f32_16x16x128_f8f6f4 -- so
// nothing is written here but the driver, and the thing measured is the
// shipped code rather than a probe's imitation of it.
//
// The one shape difference is forced: OUTPUT_PER_WG is hardcoded to 64 (4
// waves x 16 rows) where the GEMV runs a 16-row tile. That is not incidental
// -- it is exactly the starvation task #19 introduced the narrow GEMV to fix,
// and it is the real obstacle to porting this into o_proj. It does not bias
// this measurement, which is per-workgroup bandwidth over equal bytes, but it
// does mean a win here is not yet a win in the megakernel.
// ───────────────────────────────────────────────────────────────────────────
template <int K>
__global__ __launch_bounds__(256) void linear_mfma_driver(void const *input,
                                                          void const *weight,
                                                          void *output,
                                                          int n_tiles,
                                                          int o_stride) {
  for (int t = blockIdx.x; t < n_tiles; t += gridDim.x) {
    kernel::gang_linear_mxfp8_kernel<1, K>(input,
                                           weight,
                                           output,
                                           /*num_active_tokens=*/1,
                                           /*tile_n=*/64,
                                           /*output_stride=*/o_stride,
                                           /*output_size=*/o_stride,
                                           /*m_tiles=*/1,
                                           n_tiles,
                                           /*wgm=*/0,
                                           t,
                                           /*bias_ptr=*/nullptr);
  }
}

static double time_ms(void (*launch)(void *), void *ctx, int iters) {
  hipEvent_t a, b;
  HIP_CHECK(hipEventCreate(&a));
  HIP_CHECK(hipEventCreate(&b));
  launch(ctx); // warm up
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipEventRecord(a));
  for (int i = 0; i < iters; i++) {
    launch(ctx);
  }
  HIP_CHECK(hipEventRecord(b));
  HIP_CHECK(hipEventSynchronize(b));
  float ms = 0.f;
  HIP_CHECK(hipEventElapsedTime(&ms, a, b));
  HIP_CHECK(hipEventDestroy(a));
  HIP_CHECK(hipEventDestroy(b));
  return ms / iters;
}

struct StreamCtx {
  uint4 const *src;
  size_t n_vec;
  unsigned *sink;
  int blocks;
  int unroll;
};

template <int UNROLL>
static void launch_stream(void *p) {
  StreamCtx *c = static_cast<StreamCtx *>(p);
  stream_read_kernel<UNROLL><<<c->blocks, 256>>>(c->src, c->n_vec, c->sink);
}

static void run_stream(StreamCtx *c, size_t bytes) {
  void (*fn)(void *) = nullptr;
  switch (c->unroll) {
    case 1: fn = launch_stream<1>; break;
    case 2: fn = launch_stream<2>; break;
    case 4: fn = launch_stream<4>; break;
    case 8: fn = launch_stream<8>; break;
    case 16: fn = launch_stream<16>; break;
    default: return;
  }
  double ms = time_ms(fn, c, 5);
  printf("  unroll=%2d  blocks=%4d  %7.3f ms  %7.0f GB/s  %6.2f GB/s/WG\n",
         c->unroll, c->blocks, ms, bytes / (ms * 1e6),
         bytes / (ms * 1e6) / c->blocks);
}

struct MfmaCtx {
  void const *input;
  void const *weight;
  void *output;
  int n_tiles;
  int o_stride;
  int blocks;
};

template <int K>
static void launch_linear_mfma(void *p) {
  MfmaCtx *c = static_cast<MfmaCtx *>(p);
  // s_tok_fp8[K] + s_tok_scales[K/32], the layout gang_linear_mxfp8_kernel
  // carves out of its extern __shared__ block.
  linear_mfma_driver<K><<<c->blocks, 256, K + K / 32>>>(
      c->input, c->weight, c->output, c->n_tiles, c->o_stride);
}

template <int K>
static void run_linear_mfma(void const *input, void const *weight,
                            void *output, int n_tiles, int blocks) {
  constexpr int OPW = 64; // fixed inside gang_linear_mxfp8_kernel
  MfmaCtx c{input, weight, output, n_tiles, n_tiles * OPW, blocks};
  size_t const bytes = (size_t)OPW * (K + K / 32) * n_tiles;
  double ms = time_ms(launch_linear_mfma<K>, &c, 5);
  printf("  rows/wg=%2d  tiles=%6d  blocks=%4d  %7.3f ms  %7.0f GB/s  "
         "%6.2f GB/s/WG\n",
         OPW, n_tiles, blocks, ms, bytes / (ms * 1e6),
         bytes / (ms * 1e6) / blocks);
}

int main() {
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  printf("device: %s, %d CUs\n\n", props.gcnArchName, props.multiProcessorCount);

  // 4 GiB, comfortably past the 256 MB MALL, so every read is HBM.
  size_t const BUF = 4ull << 30;
  void *buf = nullptr;
  HIP_CHECK(hipMalloc(&buf, BUF));
  HIP_CHECK(hipMemset(buf, 0x3c, BUF));
  unsigned *sink = nullptr;
  HIP_CHECK(hipMalloc(&sink, sizeof(unsigned)));

  int const blocks[] = {64, 128, 192, 240, 256, 480, 1024};

  printf("A. pure streaming read (roofline re-measure), %zu MB\n", BUF >> 20);
  for (int bi = 0; bi < 7; bi++) {
    StreamCtx c{static_cast<uint4 const *>(buf), BUF / sizeof(uint4), sink,
                blocks[bi], 8};
    run_stream(&c, BUF);
  }
  printf("\n");

  // The absorbed-o_proj reduction size, same as test_gemv_mxfp8_bw's kernel B.
  constexpr int K = 10240;
  void *input = nullptr;
  HIP_CHECK(hipMalloc(&input, (size_t)K * 2));
  HIP_CHECK(hipMemset(input, 0x3c, (size_t)K * 2));

  // 64 * 10560 * 5875 = 3.97 GB, the same sweep size kernel B ran at rows=8.
  int const mfma_tiles = 5875;
  void *out = nullptr;
  HIP_CHECK(hipMalloc(&out, (size_t)mfma_tiles * 64 * 2));

  printf("I. gang_linear_mxfp8_kernel (FP8 act, FP8xFP8 MFMA), K=%d\n", K);
  for (int bi = 0; bi < 7; bi++) {
    run_linear_mfma<K>(input, buf, out, mfma_tiles, blocks[bi]);
  }
  printf("\n");

  printf("Recorded on this box for the bf16-dot GEMV (test_gemv_mxfp8_bw):\n");
  printf("  kernel A streaming roof      37.8 GB/s/WG @ 128 blocks\n");
  printf("  kernel B GEMV, global act    14.5 GB/s/WG @ 128 blocks\n");
  printf("  kernel C GEMV, LDS act       18.7-20.7 GB/s/WG @ 128 blocks\n");

  HIP_CHECK(hipFree(out));
  HIP_CHECK(hipFree(input));
  HIP_CHECK(hipFree(sink));
  HIP_CHECK(hipFree(buf));
  return 0;
}
