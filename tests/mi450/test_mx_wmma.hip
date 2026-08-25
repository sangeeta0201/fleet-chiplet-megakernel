/* Correctness harness for the MI450 MXFP4 primitives.
 *
 * What this is actually screening for. The gfx1250 wmma_scale builtin takes
 * its operand-format selectors in different argument positions than the gfx950
 * mfma_scale it replaces, and every one of those arguments is a bare integer
 * immediate. Passing FP8's code where FP4 belongs -- or getting the scale
 * format wrong -- compiles cleanly, emits the right instruction mnemonic, and
 * silently computes against a different numeric interpretation of the same
 * bits. Reading ISA cannot catch that. Only executing against a reference that
 * decodes E2M1 and E4M3 independently can.
 *
 *   ./run_ffm.sh /abs/path/to/test_mx_wmma
 *
 * The reference below decodes the formats from first principles (explicit
 * value tables, not hardware conversion instructions) so that a wrong format
 * immediate in the kernel cannot be mirrored by the same mistake in the
 * reference.
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <vector>

#include "mirage/persistent_kernel/tasks/mi450/mx_wmma_mi450.cuh"

#define HIP_CHECK(x)                                                           \
  do {                                                                         \
    hipError_t e = (x);                                                        \
    if (e != hipSuccess) {                                                     \
      printf("HIP error %s at line %d\n", hipGetErrorString(e), __LINE__);     \
      return 2;                                                                \
    }                                                                          \
  } while (0)

// FP4 E2M1: 1 sign, 2 exponent, 1 mantissa. The full set of 16 representable
// values, written out rather than computed, so this cannot drift toward
// whatever the hardware happens to do.
static float const kE2M1[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

// FP8 E4M3 (OCP): 1 sign, 4 exponent, 3 mantissa, bias 7.
static float e4m3_to_f32(uint8_t v) {
  int sign = (v >> 7) & 1;
  int exp = (v >> 3) & 0xF;
  int man = v & 0x7;
  float mag;
  if (exp == 0) {
    mag = std::ldexp((float)man, -9); // subnormal: 2^-6 * man/8
  } else if (exp == 0xF && man == 0x7) {
    mag = NAN;
  } else {
    mag = std::ldexp(1.0f + (float)man / 8.0f, exp - 7);
  }
  return sign ? -mag : mag;
}

// E8M0 scale: the byte is a raw power-of-two exponent, bias 127.
static float e8m0_to_f32(uint8_t e) {
  if (e == 0xFF) {
    return NAN;
  }
  return std::ldexp(1.0f, (int)e - 127);
}

constexpr int TILE_M = 16, TILE_N = 16, TILE_K = 128;

#if defined(__gfx1250__) || !defined(__HIP_DEVICE_COMPILE__)
__global__ __launch_bounds__(32) void mx_kernel(void const *a_ptr,
                                                void const *b_ptr,
                                                float *out,
                                                unsigned int scale_a,
                                                unsigned int scale_b) {
#if defined(MIRAGE_ARCH_GFX1250)
  using namespace kernel::mi450;
  int lane = threadIdx.x;

  // 64 bytes per lane for each operand, contiguous by lane.
  mx_i32x16_t a = ((mx_i32x16_t const *)a_ptr)[lane];
  mx_i32x16_t b = ((mx_i32x16_t const *)b_ptr)[lane];

  mx_f32x8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  acc = _gang_wmma_f4xf8(a, b, acc, (int)scale_a, (int)scale_b);

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[lane * 8 + i] = acc[i];
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

  // A: FP4 weights, 16 rows x 128 K = 2048 nibbles = 1024 bytes.
  // B: FP8 tokens, 16 cols x 128 K = 2048 bytes.
  // Both are laid out as 32 lanes x 64 bytes, which is what the kernel reads.
  std::vector<uint8_t> hA(32 * 64), hB(32 * 64);
  for (size_t i = 0; i < hA.size(); ++i) {
    // Two independent nibbles per byte; keep them different so a nibble-order
    // error is visible.
    hA[i] = (uint8_t)((((i * 7) % 15) & 0xF) | ((((i * 11) % 15) & 0xF) << 4));
  }
  for (size_t i = 0; i < hB.size(); ++i) {
    // Restrict to exponents near the bias so products stay well inside fp32
    // and away from E4M3's NaN encoding.
    hB[i] = (uint8_t)(((i * 13) % 7 + 4) << 3 | ((i * 5) % 8));
  }

  uint8_t const sa_byte = 127; // 2^0
  uint8_t const sb_byte = 127; // 2^0
  unsigned int scale_a = sa_byte;
  unsigned int scale_b = sb_byte;

  uint8_t *dA, *dB;
  float *dOut;
  HIP_CHECK(hipMalloc(&dA, hA.size()));
  HIP_CHECK(hipMalloc(&dB, hB.size()));
  HIP_CHECK(hipMalloc(&dOut, TILE_M * TILE_N * sizeof(float)));
  HIP_CHECK(hipMemcpy(dA, hA.data(), hA.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB, hB.data(), hB.size(), hipMemcpyHostToDevice));

  // Poison: distinguishes "computed zero" from "never written".
  std::vector<float> poison(TILE_M * TILE_N, -99999.f);
  HIP_CHECK(hipMemcpy(
      dOut, poison.data(), poison.size() * sizeof(float), hipMemcpyHostToDevice));

  hipLaunchKernelGGL(
      mx_kernel, dim3(1), dim3(32), 0, 0, dA, dB, dOut, scale_a, scale_b);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> got(TILE_M * TILE_N);
  HIP_CHECK(hipMemcpy(got.data(),
                      dOut,
                      got.size() * sizeof(float),
                      hipMemcpyDeviceToHost));

  int untouched = 0;
  for (float v : got) {
    if (v == -99999.f) {
      ++untouched;
    }
  }

  // Decode what the hardware was actually handed, so the report says something
  // useful even before the lane->element mapping is pinned down.
  double a_sum = 0.0, b_sum = 0.0;
  for (size_t i = 0; i < hA.size(); ++i) {
    a_sum += kE2M1[hA[i] & 0xF] + kE2M1[(hA[i] >> 4) & 0xF];
  }
  for (size_t i = 0; i < hB.size(); ++i) {
    b_sum += e4m3_to_f32(hB[i]);
  }

  double out_sum = 0.0, out_absmax = 0.0;
  int nan_count = 0;
  for (float v : got) {
    if (std::isnan(v)) {
      ++nan_count;
    } else {
      out_sum += v;
      out_absmax = fmax(out_absmax, fabs((double)v));
    }
  }

  printf("scale_a=2^%d  scale_b=2^%d\n",
         (int)sa_byte - 127,
         (int)sb_byte - 127);
  printf("operand check: sum(A as E2M1)=%.3f  sum(B as E4M3)=%.3f\n",
         a_sum,
         b_sum);
  printf("output: untouched=%d/%d  nan=%d  sum=%.4f  absmax=%.4f\n",
         untouched,
         (int)got.size(),
         nan_count,
         out_sum,
         out_absmax);
  printf("first 8: ");
  for (int i = 0; i < 8; ++i) {
    printf("%.3f ", got[i]);
  }
  printf("\n");

  if (untouched) {
    printf("FAIL: %d outputs never written -- kernel did not run\n", untouched);
    return 1;
  }
  if (nan_count) {
    printf("FAIL: %d NaN outputs -- likely a bad format immediate\n",
           nan_count);
    return 1;
  }
  if (out_absmax == 0.0) {
    printf("FAIL: all-zero output. With nonzero A and B this means the format "
           "immediates are being decoded as something that zeroes out.\n");
    return 1;
  }
  printf("PASS: MX WMMA executed and produced finite nonzero output\n");
  return 0;
}
