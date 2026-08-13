/* Copyright 2025 CMU
 *
 * Accuracy of the narrow-tile MXFP8 GEMV's inner product, at the o_proj shape.
 *
 * Why this exists. gang_gemv_detail::fma_pair used to widen both halves of a
 * packed bf16 pair to f32 with a shift and a mask and then issue two fmaf; it
 * now issues one v_dot2c_f32_bf16 instead. That is five instructions down to
 * one, and the ISA dump said those widenings were 48% of the o_proj GEMV's
 * VALU, but it also changes the arithmetic: two roundings become one fused
 * two-term dot. End-to-end that showed up as the GLM-4.7-Flash greedy decode
 * agreeing with the Torch reference for 22 tokens instead of 50 -- a shift big
 * enough that "it should be more accurate" is not good enough. So: measure it.
 *
 * The reference is the same reduction in double precision from the same fp8
 * bytes and E8M0 exponents, so the only thing being compared is how the
 * K-length sum is accumulated. K = 10240 and each lane owns 640 of those
 * terms, which is where any difference would show.
 *
 * The answer, on gfx950: 2.38e-07 relative RMS for the dot product against
 * 2.67e-07 for the widening version, so the new one is 11% *more* accurate,
 * as one rounding per pair instead of two should be. Both sit at f32 epsilon
 * and five orders of magnitude below MXFP8's own 3.7e-02 format error
 * (test_mxfp8_linear), so neither is what limits the megakernel's agreement
 * with Torch -- 47 layers of greedy decode simply amplify a last-ulp change.
 *
 * Build: hipcc -o test_gemv_mxfp8_accuracy test_gemv_mxfp8_accuracy.hip \
 *          -D__HIP_PLATFORM_AMD__ -DMIRAGE_BACKEND_USE_ROCM \
 *          -DCK_TILE_FMHA_FWD_FAST_EXP2=1 --offload-arch=gfx950 \
 *          -munsafe-fp-atomics -O3 -std=c++17 -I ../../include \
 *          -I ../../include/mirage/persistent_kernel \
 *          -I ../../deps/composable_kernel/include
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <vector>

#define HIP_CHECK(x)                                                           \
  do {                                                                         \
    hipError_t e = (x);                                                        \
    if (e != hipSuccess) {                                                     \
      printf("HIP error %s at %d\n", hipGetErrorString(e), __LINE__);          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// mpk_atoms first: the GEMV's WRITE_THROUGH epilogue needs st_wt_u16.
#include "mpk_atoms.cuh"
#include "tasks/mi300/gang_gemv_mxfp8_mi300.cuh"

constexpr int K = 10240;
constexpr int ROWS = 16;
constexpr int SCALE_BLOCK = 32;
constexpr int SCALES_PER_ROW = K / SCALE_BLOCK;
constexpr int WG_DATA_BYTES = ROWS * K;
constexpr int WG_BYTES = WG_DATA_BYTES + ROWS * SCALES_PER_ROW;

// ── the pre-dot2 inner product, kept here so both can be run side by side ───
__device__ __forceinline__ float fma_pair_widen(unsigned w, unsigned a,
                                                float acc) {
  unsigned wl = w << 16, wh = w & 0xFFFF0000u;
  unsigned al = a << 16, ah = a & 0xFFFF0000u;
  float wlf, whf, alf, ahf;
  __builtin_memcpy(&wlf, &wl, 4);
  __builtin_memcpy(&whf, &wh, 4);
  __builtin_memcpy(&alf, &al, 4);
  __builtin_memcpy(&ahf, &ah, 4);
  acc = __builtin_fmaf(wlf, alf, acc);
  acc = __builtin_fmaf(whf, ahf, acc);
  return acc;
}

// One workgroup, one tile, the same lane mapping the shipped kernel uses.
// USE_DOT2 picks which inner product runs; everything else is identical, so
// the difference between the two outputs is the rounding and nothing else.
template <bool USE_DOT2>
__global__ __launch_bounds__(256) void gemv_tile(unsigned short const *A,
                                                 unsigned char const *wg,
                                                 float *out) {
  using kernel::gang_gemv_mxfp8_detail::cvt_fp8_pair;
  using kernel::gang_gemv_mxfp8_detail::e8m0_to_f32;

  constexpr int LANES_PER_ROW = 256 / ROWS;
  constexpr int VEC = 16;
  constexpr int ITERS = K / (LANES_PER_ROW * VEC);

  int const tid = threadIdx.x;
  int const row = tid / LANES_PER_ROW;
  int const lane = tid % LANES_PER_ROW;
  unsigned char const *w_row = wg + (size_t)row * K;
  unsigned char const *s_row = wg + WG_DATA_BYTES + (size_t)row * SCALES_PER_ROW;

  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int i = 0; i < ITERS; i++) {
    int const k = (i * LANES_PER_ROW + lane) * VEC;
    uint4 const wv = *reinterpret_cast<uint4 const *>(w_row + k);
    uint4 const a0 = *reinterpret_cast<uint4 const *>(A + k);
    uint4 const a1 = *reinterpret_cast<uint4 const *>(A + k + 8);
    float const sc = e8m0_to_f32(s_row[k / SCALE_BLOCK]);
    unsigned const wd[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
    for (int h = 0; h < 4; h++) {
      uint4 const a = (h >> 1) ? a1 : a0;
      unsigned const ax = (h & 1) ? a.z : a.x;
      unsigned const ay = (h & 1) ? a.w : a.y;
      unsigned const w0 = cvt_fp8_pair<false>(wd[h], sc);
      unsigned const w1 = cvt_fp8_pair<true>(wd[h], sc);
      if constexpr (USE_DOT2) {
        acc[2 * (h & 1)] =
            kernel::gang_gemv_detail::fma_pair(w0, ax, acc[2 * (h & 1)]);
        acc[2 * (h & 1) + 1] =
            kernel::gang_gemv_detail::fma_pair(w1, ay, acc[2 * (h & 1) + 1]);
      } else {
        acc[2 * (h & 1)] = fma_pair_widen(w0, ax, acc[2 * (h & 1)]);
        acc[2 * (h & 1) + 1] = fma_pair_widen(w1, ay, acc[2 * (h & 1) + 1]);
      }
    }
  }

  float sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
#pragma unroll
  for (int off = LANES_PER_ROW >> 1; off > 0; off >>= 1) {
    sum += __shfl_xor(sum, off);
  }
  if (lane == 0) {
    out[row] = sum;
  }
}

static float b2f_h(unsigned short v) {
  unsigned u = (unsigned)v << 16;
  float f;
  __builtin_memcpy(&f, &u, 4);
  return f;
}

// E4M3 (OCP, no infinities, S.1111.111 == NaN) decode, on the host.
static double e4m3_h(unsigned char b) {
  int const s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 0x7;
  double v;
  if (e == 0) {
    v = std::ldexp((double)m / 8.0, -6);
  } else {
    v = std::ldexp(1.0 + (double)m / 8.0, e - 7);
  }
  return s ? -v : v;
}

int main() {
  srand(1234);
  std::vector<unsigned short> hA(K);
  std::vector<unsigned char> hW(WG_BYTES);
  for (int i = 0; i < K; i++) {
    // bf16 in roughly the range a post-attention row occupies.
    float f = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    unsigned u;
    __builtin_memcpy(&u, &f, 4);
    hA[i] = (unsigned short)(u >> 16);
  }
  for (int i = 0; i < WG_DATA_BYTES; i++) {
    unsigned char b = (unsigned char)(rand() & 0xFF);
    if ((b & 0x7F) == 0x7F) {
      b &= 0xFE; // avoid the E4M3 NaN encoding
    }
    hW[i] = b;
  }
  for (int i = 0; i < ROWS * SCALES_PER_ROW; i++) {
    hW[WG_DATA_BYTES + i] = (unsigned char)(120 + (rand() % 9)); // E8M0 ~2^-7..2^1
  }

  unsigned short *dA = nullptr;
  unsigned char *dW = nullptr;
  float *dO = nullptr;
  HIP_CHECK(hipMalloc(&dA, K * 2));
  HIP_CHECK(hipMalloc(&dW, WG_BYTES));
  HIP_CHECK(hipMalloc(&dO, ROWS * sizeof(float)));
  HIP_CHECK(hipMemcpy(dA, hA.data(), K * 2, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dW, hW.data(), WG_BYTES, hipMemcpyHostToDevice));

  std::vector<float> dot2(ROWS), widen(ROWS);
  gemv_tile<true><<<1, 256>>>(dA, dW, dO);
  HIP_CHECK(hipMemcpy(dot2.data(), dO, ROWS * 4, hipMemcpyDeviceToHost));
  gemv_tile<false><<<1, 256>>>(dA, dW, dO);
  HIP_CHECK(hipMemcpy(widen.data(), dO, ROWS * 4, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  // Reference: the same terms, summed in double. Note the weights are read
  // back through bf16, because that is what cvt_scalef32_pk_bf16_fp8 produces
  // and it is not the rounding under test.
  double err_d = 0.0, err_w = 0.0, mag = 0.0;
  for (int r = 0; r < ROWS; r++) {
    double ref = 0.0;
    for (int k = 0; k < K; k++) {
      double sc = std::ldexp(1.0, (int)hW[WG_DATA_BYTES + r * SCALES_PER_ROW +
                                         k / SCALE_BLOCK] -
                                      127);
      double w = e4m3_h(hW[(size_t)r * K + k]) * sc;
      // round the scaled weight to bf16, as the hardware convert does
      float wf = (float)w;
      unsigned u;
      __builtin_memcpy(&u, &wf, 4);
      unsigned bias = ((u >> 16) & 1) + 0x7FFF;
      unsigned short wb = (unsigned short)((u + bias) >> 16);
      ref += (double)b2f_h(wb) * (double)b2f_h(hA[k]);
    }
    err_d += (dot2[r] - ref) * (dot2[r] - ref);
    err_w += (widen[r] - ref) * (widen[r] - ref);
    mag += ref * ref;
  }
  double const rms_d = std::sqrt(err_d / mag);
  double const rms_w = std::sqrt(err_w / mag);
  printf("K=%d rows=%d, %d terms per output\n", K, ROWS, K);
  printf("  v_dot2c_f32_bf16 (shipped)  rel RMS vs f64 = %.3e\n", rms_d);
  printf("  widen + 2x fmaf  (previous) rel RMS vs f64 = %.3e\n", rms_w);
  printf("  ratio dot2/widen = %.3fx  (<1 means the new one is more accurate)\n",
         rms_d / rms_w);
  printf("\n%s\n", rms_d <= rms_w * 1.05 ? "[PASS]" : "[FAIL]");
  return rms_d <= rms_w * 1.05 ? 0 : 1;
}
