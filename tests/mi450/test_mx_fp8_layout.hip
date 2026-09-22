/* Validates the gfx1250 MX FP4(weights) x FP8(tokens) fragment layout.
 *
 * test_mx_layout.hip does this for the FP4xFP4 form. This is the mixed form,
 * which is what the MX linear kernels actually use: FP4 weights against FP8
 * E4M3 quantized tokens.
 *
 * THIS TEST HAS ALREADY EARNED ITS KEEP. It was written against an inference --
 * that the FP8 byte index is just the FP4 nibble index reread at byte width,
 * since both formats carry 64 operand elements per lane -- and it falsified it
 * immediately: 256/256 mismatches, worst_rel 1.6e+02. probe_fp8_pairing.hip
 * then measured the real pairing, and the B rule turned out to have a
 * different shape from A's (see mx_layout_mi450.cuh). mx_load_operand_fp8 now
 * implements the measured rule and this test passes.
 *
 * Keep it as the regression pin. It drives row-major FP8 token data through
 * the loader and checks all 256 outputs against a host reference that decodes
 * E4M3 and E8M0 from first principles. If the byte order within a lane's
 * record is wrong, K terms pair with the wrong weights and nearly every output
 * is wrong -- this is a mapping that fails loudly or not at all, which is
 * exactly why the original inference could not survive contact with it.
 *
 * Data choice follows the lesson from the attention port: exactly
 * representable values, but NOT affine in the index. Affine data makes
 * transposes and half-swaps invisible because the values they disagree about
 * happen to be equal. Values here come from a mixing hash, and the test
 * asserts its own discriminating power before trusting a pass.
 *
 *   ./run_ffm.sh /abs/path/to/test_mx_fp8_layout
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <vector>

#include "mirage/persistent_kernel/tasks/mi450/mx_layout_mi450.cuh"
#include "mirage/persistent_kernel/tasks/mi450/mx_wmma_mi450.cuh"

#define HIP_CHECK(x)                                                           \
  do {                                                                         \
    hipError_t e = (x);                                                        \
    if (e != hipSuccess) {                                                     \
      printf("HIP error %s at line %d\n", hipGetErrorString(e), __LINE__);     \
      return 2;                                                                \
    }                                                                          \
  } while (0)

// FP4 E2M1 code -> value.
static double const kE2M1[16] = {0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
                                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};

// FP8 E4M3 (OCP): 1 sign, 4 exponent, 3 mantissa, bias 7. Subnormals when the
// exponent field is 0. 0xFF/0x7F are NaN and are never generated here.
static double e4m3(uint8_t b) {
  int sign = (b >> 7) & 1;
  int exp = (b >> 3) & 0xF;
  int man = b & 0x7;
  double mag;
  if (exp == 0) {
    mag = std::ldexp((double)man / 8.0, -6);
  } else {
    mag = std::ldexp(1.0 + (double)man / 8.0, exp - 7);
  }
  return sign ? -mag : mag;
}

// E8M0: raw power-of-two exponent, bias 127.
static double e8m0(uint8_t e) {
  return std::ldexp(1.0, (int)e - 127);
}

constexpr int M = 16, N = 16, K = 128;
constexpr int NBLK = K / 32; // 4 scale blocks

#if defined(__gfx1250__) || !defined(__HIP_DEVICE_COMPILE__)
// Consumes the buffers in the shape a real MX kernel stores them:
//   weights: [M][K/2] row-major FP4 nibbles
//   tokens:  [N][K]   row-major FP8 bytes
__global__ __launch_bounds__(32) void fp8_layout_kernel(uint8_t const *w,
                                                        uint8_t const *t,
                                                        uint8_t const *ws,
                                                        uint8_t const *ts,
                                                        float *out) {
#if defined(MIRAGE_ARCH_GFX1250)
  using namespace kernel::mi450;
  int lane = threadIdx.x;
  int row = mx_operand_mn(lane); // lane & 15

  mx_i32x16_t a = mx_load_operand_fp4(w, row, 0, K);
  mx_i32x16_t b = mx_load_operand_fp8(t, row, 0, K);

  // Scales must be lane-varying to land in a VGPR (trap #1 in
  // mx_layout_mi450.cuh); indexing by `row` is what guarantees that.
  unsigned int sa = mx_pack_scales(ws + row * NBLK);
  unsigned int sb = mx_pack_scales(ts + row * NBLK);

  mx_f32x8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  acc = _gang_wmma_f4xf8(a, b, acc, (int)sa, (int)sb);

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int m = mx_acc_row(lane, i); // 8*(lane/16) + i
    int n = mx_acc_col(lane);    // lane % 16
    out[m * N + n] = acc[i];
  }
#endif
}
#endif

static unsigned mix(unsigned x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

int main() {
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
    printf("no HIP device\n");
    return 2;
  }
  printf("device: %s  warpSize=%d\n", prop.gcnArchName, prop.warpSize);

  std::vector<uint8_t> hW((size_t)M * K / 2), hT((size_t)N * K);
  std::vector<uint8_t> hWs((size_t)M * NBLK), hTs((size_t)N * NBLK);

  // Weight nibbles: full E2M1 code range, hashed so neighbouring K differ.
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; k += 2) {
      uint8_t lo = (uint8_t)(mix((unsigned)(m * 977 + k * 31 + 1)) & 0xF);
      uint8_t hi = (uint8_t)(mix((unsigned)(m * 613 + k * 17 + 5)) & 0xF);
      hW[(size_t)m * (K / 2) + k / 2] = (uint8_t)(lo | (hi << 4));
    }
  }
  // Token bytes: exponent field 4..10 (well inside range, never NaN),
  // full 3-bit mantissa, hashed sign.
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      unsigned h = mix((unsigned)(n * 1409 + k * 53 + 11));
      uint8_t sign = (uint8_t)((h >> 20) & 1);
      uint8_t exp = (uint8_t)(4 + (h % 7));
      uint8_t man = (uint8_t)((h >> 8) & 0x7);
      hT[(size_t)n * K + k] = (uint8_t)((sign << 7) | (exp << 3) | man);
    }
  }
  // E8M0 scales near 2^0 so products stay in a comfortable range, but varying
  // per (row, block) so a scale-block mix-up is visible.
  for (int m = 0; m < M; ++m) {
    for (int b = 0; b < NBLK; ++b) {
      hWs[(size_t)m * NBLK + b] =
          (uint8_t)(125 + (mix((unsigned)(m * 71 + b * 13 + 3)) % 5));
    }
  }
  for (int n = 0; n < N; ++n) {
    for (int b = 0; b < NBLK; ++b) {
      hTs[(size_t)n * NBLK + b] =
          (uint8_t)(125 + (mix((unsigned)(n * 37 + b * 91 + 7)) % 5));
    }
  }

  // ── Host reference, from first principles ───────────────────────────────
  // D[m][n] = sum_k  W[m][k]*wscale[m][k/32] * T[n][k]*tscale[n][k/32]
  std::vector<double> ref((size_t)M * N, 0.0);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      double s = 0.0;
      for (int k = 0; k < K; ++k) {
        uint8_t packed = hW[(size_t)m * (K / 2) + k / 2];
        uint8_t code = (k & 1) ? (uint8_t)(packed >> 4) : (uint8_t)(packed & 0xF);
        double w = kE2M1[code] * e8m0(hWs[(size_t)m * NBLK + k / 32]);
        double t = e4m3(hT[(size_t)n * K + k]) * e8m0(hTs[(size_t)n * NBLK + k / 32]);
        s += w * t;
      }
      ref[(size_t)m * N + n] = s;
    }
  }

  // ── Discriminating-power guard ──────────────────────────────────────────
  // A layout test whose reference is symmetric cannot detect a transpose. Fail
  // before running rather than report a meaningless pass.
  {
    int asym = 0, distinct_rows = 0;
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        if (m != n && std::fabs(ref[(size_t)m * N + n] - ref[(size_t)n * N + m]) >
                          1e-9) {
          ++asym;
        }
      }
    }
    for (int m = 1; m < M; ++m) {
      if (std::fabs(ref[(size_t)m * N] - ref[0]) > 1e-9) {
        ++distinct_rows;
      }
    }
    if (asym == 0 || distinct_rows == 0) {
      printf("FAIL(test): reference is degenerate (asym=%d distinct_rows=%d). "
             "A transpose or row-swap would be invisible; fix the input data.\n",
             asym,
             distinct_rows);
      return 1;
    }
    printf("discriminating: asymmetric pairs=%d  distinct rows=%d\n",
           asym,
           distinct_rows);
  }

  uint8_t *dW, *dT, *dWs, *dTs;
  float *dOut;
  HIP_CHECK(hipMalloc(&dW, hW.size()));
  HIP_CHECK(hipMalloc(&dT, hT.size()));
  HIP_CHECK(hipMalloc(&dWs, hWs.size()));
  HIP_CHECK(hipMalloc(&dTs, hTs.size()));
  HIP_CHECK(hipMalloc(&dOut, (size_t)M * N * sizeof(float)));
  HIP_CHECK(hipMemcpy(dW, hW.data(), hW.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dT, hT.data(), hT.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dWs, hWs.data(), hWs.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dTs, hTs.data(), hTs.size(), hipMemcpyHostToDevice));

  // Poison, so "never written" cannot pass as "computed zero".
  std::vector<float> poison((size_t)M * N, -98765.f);
  HIP_CHECK(hipMemcpy(dOut,
                      poison.data(),
                      poison.size() * sizeof(float),
                      hipMemcpyHostToDevice));

  hipLaunchKernelGGL(
      fp8_layout_kernel, dim3(1), dim3(32), 0, 0, dW, dT, dWs, dTs, dOut);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> got((size_t)M * N);
  HIP_CHECK(hipMemcpy(
      got.data(), dOut, got.size() * sizeof(float), hipMemcpyDeviceToHost));

  int fails = 0, untouched = 0;
  double worst = 0.0;
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float g = got[(size_t)m * N + n];
      if (g == -98765.f) {
        ++untouched;
        continue;
      }
      double r = ref[(size_t)m * N + n];
      double denom = std::fabs(r) > 1e-6 ? std::fabs(r) : 1.0;
      double rel = std::fabs((double)g - r) / denom;
      if (rel > worst) {
        worst = rel;
      }
      if (rel > 1e-5) {
        if (fails < 6) {
          printf("FAIL m=%d n=%d got=%.6f want=%.6f rel=%.3e\n", m, n, g, r, rel);
        }
        ++fails;
      }
    }
  }

  printf("untouched=%d  mismatches=%d  worst_rel=%.3e\n",
         untouched,
         fails,
         worst);
  if (untouched) {
    printf("FAIL: %d outputs never written\n", untouched);
    return 1;
  }
  if (fails) {
    printf("FAIL: FP8 operand layout no longer matches the measured mapping "
           "(k = 32*chunk + 16*(lane>>4) + i). Re-run "
           "probe_fp8_pairing.hip -- and pass all four E8M0 scale bytes "
           "(0x7F7F7F7F), not a bare 127, or K>=32 reads as 2^-127.\n");
    return 1;
  }
  printf("PASS: FP4xFP8 layout matches host reference (worst_rel=%.3e)\n",
         worst);
  return 0;
}
