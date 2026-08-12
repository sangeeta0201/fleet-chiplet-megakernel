/* Copyright 2025 CMU
 *
 * Numerical test for the MXFP8 dense gang linear kernels
 * (gang_linear_mxfp8_kernel / gang_linear_res_mxfp8_kernel in
 * tasks/mi300/gang_linear_mxfp8_mi300.cuh).
 *
 * Two references, because two different things can be wrong:
 *
 *   exact   A scalar dequantize-and-sum over the very same packed bytes and
 *           exponents the MFMA consumes, with the activation quantized by the
 *           same _gang_wave_parallel_fp8_quant call. Quantization error is
 *           common to both sides, so this isolates the operand layout and the
 *           pipeline -- it has to match to ~1e-6 or the kernel is wrong.
 *
 *   bf16    The unquantized fp32 dot product on the host. This one is expected
 *           to differ; it is reported so the format's error is a measured
 *           number rather than an assumption. It is not an assertion, since
 *           the honest test of quantization quality is perplexity, not a
 *           random-matrix GEMM.
 *
 * Build: ./build.sh
 * Run:   ./test_mxfp8_linear
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <vector>

// Two symbols the MoE kernels in gang_moe_linear_mxfp4_mi300.cuh reach for,
// which inside the megakernel come from headers task_header.cuh included
// earlier. Neither is used by the MXFP8 kernels under test; supplying them
// here beats pulling in gang_moe_linear_mi300.cuh, which would drag in CK.
//
// MPK_WS_WAVE_SYNC is barrier-watchdog telemetry that writes to a host buffer
// existing only inside the megakernel, and is already a no-op when that buffer
// is null. _gang_moe_get_xcd_id is copied verbatim.
#define MPK_WS_WAVE_SYNC(wave) ((void)0)
namespace kernel {
__device__ __forceinline__ int _gang_moe_get_xcd_id() {
  int xcd_id;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcd_id));
  return xcd_id;
}
} // namespace kernel

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

// GLM-4.7-Flash's qkv_a shape: K = hidden 2048, N = q_lora 768 + kv_lora 512 +
// rope 64 padded to 2048. 2048/64 = 32 workgroups, enough tiles to exercise
// the windowed traversal.
static constexpr int K = 2048;
static constexpr int N = 2048;
static constexpr int BATCH = 1;
static constexpr int OUTPUT_PER_WG = 64;
static constexpr int N_WGS = N / OUTPUT_PER_WG;
static constexpr int BLOCKS_32 = K / 32;
static constexpr size_t WG_DATA_BYTES = (size_t)OUTPUT_PER_WG * K;
static constexpr size_t WG_SCALE_BYTES = (size_t)OUTPUT_PER_WG * BLOCKS_32;
static constexpr size_t WG_BYTES = WG_DATA_BYTES + WG_SCALE_BYTES;

// Enough for the FP8 token staging the kernels declare: K data bytes plus one
// exponent per 128-element tile.
static constexpr int SMEM_BYTES = K + K / 128 + 256;

using namespace kernel;

// ── weight packing ─────────────────────────────────────────────────────────
// One block per output row, one thread per 32-element scale block. Packs into
// the [wg][row-major FP8 data][row-major E8M0 scales] layout the kernel reads.
// Exponents cover contiguous K blocks -- see test_mxfp8_mfma_layout.hip.
__global__ void pack_weights(unsigned short const *src, uint8_t *out) {
  int n = blockIdx.x;
  int b = threadIdx.x;
  if (b >= BLOCKS_32) {
    return;
  }
  int wg = n / OUTPUT_PER_WG;
  int row = n % OUTPUT_PER_WG;
  uint8_t *data = out + (size_t)wg * WG_BYTES + (size_t)row * K;
  uint8_t *scales = out + (size_t)wg * WG_BYTES + WG_DATA_BYTES + row * BLOCKS_32;

  unsigned short const *srow = src + (size_t)n * K + b * 32;
  float vals[32];
  float amax = 0.0f;
  for (int j = 0; j < 32; j++) {
    vals[j] = _gang_bf16_to_float(srow[j]);
    amax = fmaxf(amax, fabsf(vals[j]));
  }
  uint8_t se = _gang_compute_e8m0_fp8(amax);
  scales[b] = se;
  float scale_f = 1.0f;
  if (se != 0) {
    union {
      float f;
      uint32_t u;
    } sv;
    sv.u = (uint32_t)se << 23;
    scale_f = sv.f;
  }
  for (int j = 0; j < 32; j += 4) {
    fp8x4_t pk =
        _gang_quant_4xfp8(vals[j], vals[j + 1], vals[j + 2], vals[j + 3], scale_f);
    *(int *)(data + b * 32 + j) = *(int const *)&pk;
  }
}

// ── references and kernel wrappers ─────────────────────────────────────────

__device__ __forceinline__ float fp8_to_f32(uint8_t b) {
  return __builtin_amdgcn_cvt_f32_fp8((int)b, 0);
}

__device__ __forceinline__ float e8m0_to_f32(uint8_t se) {
  if (se == 0) {
    return 1.0f;
  }
  union {
    float f;
    uint32_t u;
  } sv;
  sv.u = (uint32_t)se << 23;
  return sv.f;
}

// Scalar sum over exactly the bytes the MFMA sees, in plain K order.
__global__ void reference_kernel(unsigned short const *input,
                                 uint8_t const *w,
                                 float *out) {
  extern __shared__ char smem[];
  uint8_t *s_tok = (uint8_t *)smem;
  uint8_t *s_scales = s_tok + K;
  _gang_wave_parallel_fp8_quant<K>(input, s_tok, s_scales);

  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    int wg = n / OUTPUT_PER_WG;
    int row = n % OUTPUT_PER_WG;
    uint8_t const *data = w + (size_t)wg * WG_BYTES + (size_t)row * K;
    uint8_t const *scales =
        w + (size_t)wg * WG_BYTES + WG_DATA_BYTES + row * BLOCKS_32;
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
      sum += (fp8_to_f32(data[k]) * e8m0_to_f32(scales[k >> 5])) *
             (fp8_to_f32(s_tok[k]) * e8m0_to_f32(s_scales[k >> 7]));
    }
    out[n] = sum;
  }
}

__global__ void run_linear(void const *input,
                           void const *weight,
                           void *output,
                           void const *bias,
                           int m_tiles,
                           int n_tiles,
                           int wgm) {
  gang_linear_mxfp8_kernel<BATCH, K>(input,
                                     weight,
                                     output,
                                     BATCH,
                                     OUTPUT_PER_WG,
                                     N,
                                     N,
                                     m_tiles,
                                     n_tiles,
                                     wgm,
                                     (int)blockIdx.x,
                                     bias);
}

__global__ void run_linear_res(void const *input,
                               void const *weight,
                               void const *residual,
                               void *output,
                               void const *bias,
                               int m_tiles,
                               int n_tiles,
                               int wgm) {
  gang_linear_res_mxfp8_kernel<BATCH, K>(input,
                                         weight,
                                         residual,
                                         output,
                                         BATCH,
                                         OUTPUT_PER_WG,
                                         N,
                                         N,
                                         m_tiles,
                                         n_tiles,
                                         wgm,
                                         (int)blockIdx.x,
                                         bias);
}

// ── host helpers ───────────────────────────────────────────────────────────

static unsigned short f2bf(float f) {
  union {
    float f;
    unsigned u;
  } v;
  v.f = f;
  unsigned bias = ((v.u >> 16) & 1) + 0x7FFF;
  return (unsigned short)((v.u + bias) >> 16);
}

static float bf2f(unsigned short b) {
  union {
    float f;
    unsigned u;
  } v;
  v.u = (unsigned)b << 16;
  return v.f;
}

static float frand() {
  return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
}

int main() {
  srand(20260812);

  std::vector<unsigned short> h_a(K), h_w((size_t)N * K), h_bias(N), h_res(N);
  for (int k = 0; k < K; k++) {
    h_a[k] = f2bf(frand());
  }
  // Give the rows different dynamic ranges so a single global exponent could
  // not accidentally pass for per-block ones.
  for (int n = 0; n < N; n++) {
    float row_mag = exp2f((float)(n % 7) - 3.0f);
    for (int k = 0; k < K; k++) {
      h_w[(size_t)n * K + k] = f2bf(frand() * row_mag * (k % 64 < 8 ? 8.0f : 1.0f));
    }
    h_bias[n] = f2bf(frand() * 0.1f);
    h_res[n] = f2bf(frand());
  }

  unsigned short *d_a, *d_w_bf16, *d_bias, *d_res, *d_out, *d_out_res;
  uint8_t *d_w;
  float *d_ref;
  HIP_CHECK(hipMalloc(&d_a, K * 2));
  HIP_CHECK(hipMalloc(&d_w_bf16, (size_t)N * K * 2));
  HIP_CHECK(hipMalloc(&d_bias, N * 2));
  HIP_CHECK(hipMalloc(&d_res, N * 2));
  HIP_CHECK(hipMalloc(&d_out, N * 2));
  HIP_CHECK(hipMalloc(&d_out_res, N * 2));
  HIP_CHECK(hipMalloc(&d_w, (size_t)N_WGS * WG_BYTES));
  HIP_CHECK(hipMalloc(&d_ref, N * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_a, h_a.data(), K * 2, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(
      d_w_bf16, h_w.data(), (size_t)N * K * 2, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_bias, h_bias.data(), N * 2, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_res, h_res.data(), N * 2, hipMemcpyHostToDevice));

  pack_weights<<<N, BLOCKS_32>>>(d_w_bf16, d_w);
  HIP_CHECK(hipDeviceSynchronize());

  reference_kernel<<<1, 256, SMEM_BYTES>>>(d_a, d_w, d_ref);
  HIP_CHECK(hipDeviceSynchronize());

  int const m_tiles = 1, n_tiles = N_WGS, wgm = 1;
  run_linear<<<m_tiles * n_tiles, 256, SMEM_BYTES>>>(
      d_a, d_w, d_out, d_bias, m_tiles, n_tiles, wgm);
  run_linear_res<<<m_tiles * n_tiles, 256, SMEM_BYTES>>>(
      d_a, d_w, d_res, d_out_res, d_bias, m_tiles, n_tiles, wgm);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> ref(N);
  std::vector<unsigned short> got(N), got_res(N);
  HIP_CHECK(
      hipMemcpy(ref.data(), d_ref, N * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(got.data(), d_out, N * 2, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(got_res.data(), d_out_res, N * 2, hipMemcpyDeviceToHost));

  // Unquantized reference, for the informational error figure.
  std::vector<float> exact(N);
  for (int n = 0; n < N; n++) {
    float s = 0.0f;
    for (int k = 0; k < K; k++) {
      s += bf2f(h_w[(size_t)n * K + k]) * bf2f(h_a[k]);
    }
    exact[n] = s;
  }

  // Mixed absolute/relative bound, the usual form for a dot product that can
  // cancel. The relative term is set by the bf16 store: 8 mantissa bits, so
  // 2^-9 per element, budgeted at 2^-8 to leave a bit for the MFMA and the
  // scalar reference summing the same 2048 products in different orders. The
  // absolute term covers outputs that cancel to near zero, where fp32
  // accumulation noise of order mag*2^-24*sqrt(K) swamps any relative
  // measure; it is small enough that a mislaid operand still shows.
  //
  // Deliberately not normalized by the tile max alone: a wrong scale grouping
  // damages the small outputs worst, which is precisely what that would hide.
  double mag = 0.0;
  for (int n = 0; n < N; n++) {
    mag = fmax(mag, fabs(ref[n]));
  }
  double const rel_tol = 1.0 / 256.0;
  double const abs_tol = mag * 1e-5;

  double worst = 0.0, worst_res = 0.0;
  int arg = -1;
  for (int n = 0; n < N; n++) {
    double want = ref[n] + bf2f(h_bias[n]);
    double d = fabs(bf2f(got[n]) - want) / (rel_tol * fabs(want) + abs_tol);
    if (d > worst) {
      worst = d;
      arg = n;
    }
    double want_res = want + bf2f(h_res[n]);
    worst_res = fmax(worst_res,
                     fabs(bf2f(got_res[n]) - want_res) /
                         (rel_tol * fabs(want_res) + abs_tol));
  }

  // Quantization quality as relative RMS, which is the figure that predicts
  // perplexity. A max over near-cancelling elements would not.
  double num = 0.0, den = 0.0;
  for (int n = 0; n < N; n++) {
    num += (ref[n] - exact[n]) * (ref[n] - exact[n]);
    den += (double)exact[n] * exact[n];
  }
  double rms_q = sqrt(num / den);

  // worst/worst_res are in units of the budget, so < 1 passes.
  bool ok = worst < 1.0 && worst_res < 1.0;

  printf("K=%d N=%d  (%d workgroups)\n", K, N, N_WGS);
  printf("  budget = %.3e*|want| + %.3e\n", rel_tol, abs_tol);
  printf("  linear      vs exact-dequant : %.2fx budget %s (worst n=%d)\n",
         worst,
         worst < 1.0 ? "OK" : "FAIL",
         arg);
  printf("  linear+res  vs exact-dequant : %.2fx budget %s\n",
         worst_res,
         worst_res < 1.0 ? "OK" : "FAIL");
  printf("  MXFP8 format error vs bf16   : %.3e rel RMS (informational)\n",
         rms_q);

  // With MXFP8_DUMP_DIR set, write the bf16 source and the device-packed
  // bytes so check_mxfp8_pack.py can diff the demo's torch packer against
  // them. The two have to agree byte for byte: a packer that lays rows or
  // exponents out differently does not fault, it just quantizes the model
  // wrong, which is indistinguishable from the format being bad.
  if (char const *dir = getenv("MXFP8_DUMP_DIR")) {
    std::vector<uint8_t> packed((size_t)N_WGS * WG_BYTES);
    HIP_CHECK(hipMemcpy(
        packed.data(), d_w, packed.size(), hipMemcpyDeviceToHost));
    char path[1024];
    snprintf(path, sizeof(path), "%s/w_bf16.bin", dir);
    FILE *f = fopen(path, "wb");
    fwrite(h_w.data(), 2, h_w.size(), f);
    fclose(f);
    snprintf(path, sizeof(path), "%s/w_packed.bin", dir);
    f = fopen(path, "wb");
    fwrite(packed.data(), 1, packed.size(), f);
    fclose(f);
    printf("  dumped w_bf16.bin / w_packed.bin to %s\n", dir);
  }
  printf("\n%s\n", ok ? "[PASS]" : "[FAIL]");
  return ok ? 0 : 1;
}
