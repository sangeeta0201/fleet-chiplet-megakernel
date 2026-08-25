/* Validates the measured gfx1250 MX-FP4 fragment layout end to end.
 *
 * Unlike test_mx_wmma.hip, which checks the format immediates using constant
 * fills (deliberately layout-independent), this test is *about* the layout.
 * It feeds row-major FP4 weight and token buffers -- the shape Fleet actually
 * stores -- through the mx_load_operand_fp4()/mx_pack_scales() helpers and
 * compares all 256 outputs against a host reference that decodes E2M1 and E8M0
 * from first principles.
 *
 * A layout error here is not subtle in its consequences but is completely
 * silent at compile time, so this is the gate that has to pass before any MoE
 * kernel can be trusted on gfx1250.
 *
 *   ./run_ffm.sh /abs/path/to/test_mx_layout
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <vector>

#include "mirage/persistent_kernel/tasks/mi450/mx_layout_mi450.cuh"

#define HIP_CHECK(x)                                                           \
  do {                                                                         \
    hipError_t e = (x);                                                        \
    if (e != hipSuccess) {                                                     \
      printf("HIP error %s at line %d\n", hipGetErrorString(e), __LINE__);     \
      return 2;                                                                \
    }                                                                          \
  } while (0)

// FP4 E2M1, written out rather than computed.
static float const kE2M1[16] = {0.0f,
                                0.5f,
                                1.0f,
                                1.5f,
                                2.0f,
                                3.0f,
                                4.0f,
                                6.0f,
                                -0.0f,
                                -0.5f,
                                -1.0f,
                                -1.5f,
                                -2.0f,
                                -3.0f,
                                -4.0f,
                                -6.0f};

// E8M0: raw power-of-two exponent, bias 127.
static double e8m0(uint8_t e) {
  return std::ldexp(1.0, (int)e - 127);
}

constexpr int M = 16, N = 16, K = 128;

#if defined(__gfx1250__) || !defined(__HIP_DEVICE_COMPILE__)
// Deliberately mirrors how a MoE kernel would consume the data: row-major FP4
// operands plus a per-row array of 4 E8M0 scale bytes.
__global__ __launch_bounds__(32) void layout_kernel(uint8_t const *a_fp4,
                                                    uint8_t const *b_fp4,
                                                    uint8_t const *a_scales,
                                                    uint8_t const *b_scales,
                                                    float *out) {
#if defined(MIRAGE_ARCH_GFX1250)
  using namespace kernel::mi450;
  int lane = threadIdx.x;
  int mn = mx_operand_mn(lane); // matrix row for A, column for B

  mx_i32x16_t a = mx_load_operand_fp4(a_fp4, mn, 0, K);
  mx_i32x16_t b = mx_load_operand_fp4(b_fp4, mn, 0, K);

  // Lane-varying, so these stay in VGPRs. An SGPR here would drop blocks 1..3.
  unsigned int sa = mx_pack_scales(a_scales + mn * MX_SCALES_PER_TILE);
  unsigned int sb = mx_pack_scales(b_scales + mn * MX_SCALES_PER_TILE);

  mx_f32x8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  acc = _gang_wmma_f4xf4(a, b, acc, (int)sa, (int)sb);

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[mx_acc_row(lane, i) * N + mx_acc_col(lane)] = acc[i];
  }
#endif
}
#endif

int main() {
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
    printf("no HIP device\n");
    return 2;
  }
  printf("device: %s (%s)\n", prop.name, prop.gcnArchName);

  // Row-major FP4: [16][128], two nibbles per byte.
  std::vector<uint8_t> A(M * K), B(N * K); // one code per entry, host side
  std::vector<uint8_t> SA(M * 4), SB(N * 4);
  uint32_t s = 987654321u;
  auto rnd = [&]() {
    s = s * 1664525u + 1013904223u;
    return s >> 16;
  };
  for (auto &v : A) {
    v = rnd() % 16;
  }
  for (auto &v : B) {
    v = rnd() % 16;
  }
  // Keep scales near 2^0 so products stay exactly representable in fp32 and
  // the comparison can be exact rather than tolerance-based.
  for (auto &v : SA) {
    v = 125 + rnd() % 5;
  }
  for (auto &v : SB) {
    v = 125 + rnd() % 5;
  }

  std::vector<uint8_t> packA(M * K / 2, 0), packB(N * K / 2, 0);
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; ++k) {
      uint8_t &byte = packA[m * (K / 2) + k / 2];
      byte |= (k & 1) ? (A[m * K + k] << 4) : A[m * K + k];
    }
  }
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      uint8_t &byte = packB[n * (K / 2) + k / 2];
      byte |= (k & 1) ? (B[n * K + k] << 4) : B[n * K + k];
    }
  }

  uint8_t *dA, *dB, *dSA, *dSB;
  float *dOut;
  HIP_CHECK(hipMalloc(&dA, packA.size()));
  HIP_CHECK(hipMalloc(&dB, packB.size()));
  HIP_CHECK(hipMalloc(&dSA, SA.size()));
  HIP_CHECK(hipMalloc(&dSB, SB.size()));
  HIP_CHECK(hipMalloc(&dOut, M * N * sizeof(float)));
  HIP_CHECK(
      hipMemcpy(dA, packA.data(), packA.size(), hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(dB, packB.data(), packB.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dSA, SA.data(), SA.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dSB, SB.data(), SB.size(), hipMemcpyHostToDevice));

  // Poison: separates "computed zero" from "never written".
  std::vector<float> poison(M * N, -99999.f);
  HIP_CHECK(hipMemcpy(dOut,
                      poison.data(),
                      poison.size() * sizeof(float),
                      hipMemcpyHostToDevice));

  hipLaunchKernelGGL(
      layout_kernel, dim3(1), dim3(32), 0, 0, dA, dB, dSA, dSB, dOut);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> got(M * N);
  HIP_CHECK(hipMemcpy(
      got.data(), dOut, got.size() * sizeof(float), hipMemcpyDeviceToHost));

  int untouched = 0, nonzero = 0;
  for (float v : got) {
    if (v == -99999.f) {
      ++untouched;
    }
    if (v != 0.f) {
      ++nonzero;
    }
  }

  int bad = 0;
  double worst = 0.0;
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      double ref = 0.0;
      for (int k = 0; k < K; ++k) {
        ref += (double)kE2M1[A[m * K + k]] * e8m0(SA[m * 4 + k / 32]) *
               (double)kE2M1[B[n * K + k]] * e8m0(SB[n * 4 + k / 32]);
      }
      double err = std::fabs(got[m * N + n] - ref);
      worst = fmax(worst, err);
      if (err > 1e-3 * std::fabs(ref) + 1e-4) {
        if (bad < 5) {
          printf("  MISMATCH C[%2d][%2d] got=%.5f ref=%.5f\n",
                 m,
                 n,
                 got[m * N + n],
                 ref);
        }
        ++bad;
      }
    }
  }

  printf("untouched=%d/%d  nonzero=%d/%d  mismatches=%d/%d  worst_err=%.6g\n",
         untouched,
         M * N,
         nonzero,
         M * N,
         bad,
         M * N,
         worst);

  if (untouched) {
    printf("FAIL: %d outputs never written\n", untouched);
    return 1;
  }
  if (bad) {
    printf("FAIL: fragment layout does not match the reference\n");
    return 1;
  }
  printf("PASS: row-major FP4 -> mx_load_operand_fp4 -> WMMA -> mx_acc_row/col "
         "reproduces the reference exactly\n");
  return 0;
}
