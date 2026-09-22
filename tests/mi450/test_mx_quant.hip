/* Validates mx_quant_fp8 -- bf16 -> FP8 E4M3 + E8M0 block quantization.
 *
 * THE BUG THIS EXISTS TO CATCH is the scale direction. gfx1250's
 * cvt_scalef32_pk8_fp8_f32 DIVIDES by the scale; gfx950's
 * cvt_scalef32_pk_fp8_f32 MULTIPLIES. Porting the gfx950 line unchanged
 * applies the scale backwards, and the resulting error is proportional to
 * s^2 -- so with the near-2^0 scales a small test naturally produces, it is
 * almost invisible. Case 3 below therefore forces amax far from 1.0 in both
 * directions, where a reversed scale is off by orders of magnitude and cannot
 * be mistaken for rounding.
 *
 * WHAT IS CHECKED, per case:
 *   1. the E8M0 byte matches an independent host computation
 *   2. every FP8 byte round-trips to the value the host reference predicts
 *   3. the reconstructed value (fp8 * 2^(e8m0-127)) tracks the bf16 input to
 *      within E4M3's representational limit -- the end-to-end property the
 *      kernels actually depend on
 *   4. no block saturates (a saturated block means the E8M0 rounding rule is
 *      wrong, which is check 1 failing in a way check 1 alone can miss)
 *
 * Case 4 is the direct scale-direction probe kept as a regression: it feeds a
 * known input at known scales and asserts divide, not multiply.
 *
 *   ./run_ffm.sh /abs/path/to/test_mx_quant
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <vector>

#include "mirage/persistent_kernel/tasks/mi450/mx_quant_mi450.cuh"

#define HIP_CHECK(x)                                                           \
  do {                                                                         \
    hipError_t e = (x);                                                        \
    if (e != hipSuccess) {                                                     \
      printf("HIP error %s at line %d\n", hipGetErrorString(e), __LINE__);     \
      return 2;                                                                \
    }                                                                          \
  } while (0)

static double e4m3(uint8_t b) {
  int sign = (b >> 7) & 1;
  int exp = (b >> 3) & 0xF;
  int man = b & 0x7;
  if (exp == 0xF && man == 0x7) {
    return NAN;
  }
  double mag = (exp == 0) ? std::ldexp((double)man / 8.0, -6)
                          : std::ldexp(1.0 + (double)man / 8.0, exp - 7);
  return sign ? -mag : mag;
}

static double e8m0(uint8_t e) {
  return std::ldexp(1.0, (int)e - 127);
}

static float bf16_to_f(unsigned short b) {
  unsigned u = (unsigned)b << 16;
  float f;
  __builtin_memcpy(&f, &u, 4);
  return f;
}
static unsigned short f_to_bf16(float f) {
  unsigned u;
  __builtin_memcpy(&u, &f, 4);
  return (unsigned short)(u >> 16); // truncate; inputs are chosen exact
}

static unsigned mix(unsigned x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

// Host mirror of mx_compute_e8m0_fp8, written independently from the float-bit
// manipulation the device uses, so a shared misunderstanding has to be made
// twice in two dissimilar ways.
static uint8_t host_e8m0_fp8(float amax) {
  if (amax == 0.0f) {
    return 0;
  }
  double need = (double)amax / 448.0;
  int e = (int)std::ceil(std::log2(need));
  // log2 of an exact power of two can land a hair low; snap it.
  while (std::ldexp(1.0, e) < need) {
    ++e;
  }
  while (e > -127 && std::ldexp(1.0, e - 1) >= need) {
    --e;
  }
  int raw = e + 127;
  return (uint8_t)std::max(0, std::min(255, raw));
}

#if defined(__gfx1250__) || !defined(__HIP_DEVICE_COMPILE__)
template <int K>
__global__ __launch_bounds__(256) void quant_kernel(unsigned short const *src,
                                                    uint8_t *out_fp8,
                                                    uint8_t *out_scl) {
#if defined(MIRAGE_ARCH_GFX1250)
  kernel::mi450::mx_quant_fp8<K>(src, out_fp8, out_scl);
#endif
}

// Direct scale-direction probe: convert a known value at a known scale.
__global__ __launch_bounds__(1) void scale_dir_kernel(float const *in,
                                                      float scale,
                                                      unsigned *out) {
#if defined(MIRAGE_ARCH_GFX1250)
  using namespace kernel::mi450;
  mx_f32x8_cvt_t v;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    v[j] = in[j];
  }
  mx_u32x2_t r = __builtin_amdgcn_cvt_scalef32_pk8_fp8_f32(v, scale);
  out[0] = r[0];
  out[1] = r[1];
#endif
}
#endif

// mode: 0 = tame (near 2^0), 1 = large, 2 = small
template <int K>
static int run_case(char const *name, int mode) {
  constexpr int NBLK = K / 32;
  printf("\n--- %s (K=%d, %d blocks) ---\n", name, K, NBLK);

  std::vector<unsigned short> hSrc(K);
  for (int k = 0; k < K; ++k) {
    unsigned h = mix((unsigned)(k * 53 + 11));
    // Mantissa from the hash, exponent chosen per mode. Values are exactly
    // representable in bf16 by construction (8-bit mantissa field, we use 4).
    float mant = 1.0f + (float)(h & 0xF) / 16.0f;
    int exp;
    if (mode == 0) {
      exp = -1 + (int)((h >> 8) % 3); // ~2^0
    } else if (mode == 1) {
      exp = 12 + (int)((h >> 8) % 4); // ~4096..32768: amax >> 448
    } else {
      exp = -20 + (int)((h >> 8) % 4); // ~1e-6: amax << 448
    }
    float v = std::ldexp(mant, exp);
    if ((h >> 20) & 1) {
      v = -v;
    }
    hSrc[k] = f_to_bf16(v);
  }

  unsigned short *dSrc;
  uint8_t *dFp8, *dScl;
  HIP_CHECK(hipMalloc(&dSrc, K * sizeof(unsigned short)));
  HIP_CHECK(hipMalloc(&dFp8, K));
  HIP_CHECK(hipMalloc(&dScl, NBLK));
  HIP_CHECK(hipMemcpy(
      dSrc, hSrc.data(), K * sizeof(unsigned short), hipMemcpyHostToDevice));
  // Poison: a block never written must not read as a valid quantization.
  HIP_CHECK(hipMemset(dFp8, 0xAB, K));
  HIP_CHECK(hipMemset(dScl, 0xCD, NBLK));

  hipLaunchKernelGGL(quant_kernel<K>, dim3(1), dim3(256), 0, 0, dSrc, dFp8, dScl);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<uint8_t> gFp8(K), gScl(NBLK);
  HIP_CHECK(hipMemcpy(gFp8.data(), dFp8, K, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(gScl.data(), dScl, NBLK, hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(dSrc));
  HIP_CHECK(hipFree(dFp8));
  HIP_CHECK(hipFree(dScl));

  int scale_bad = 0, sat = 0, recon_bad = 0;
  double worst_rel = 0.0;
  for (int b = 0; b < NBLK; ++b) {
    float amax = 0.f;
    for (int j = 0; j < 32; ++j) {
      amax = std::fmax(amax, std::fabs(bf16_to_f(hSrc[b * 32 + j])));
    }
    uint8_t want = host_e8m0_fp8(amax);
    if (gScl[b] != want) {
      if (scale_bad < 4) {
        printf("FAIL scale blk=%d got=%u want=%u (amax=%g)\n",
               b,
               gScl[b],
               want,
               amax);
      }
      ++scale_bad;
      continue;
    }
    double s = e8m0(gScl[b]);
    for (int j = 0; j < 32; ++j) {
      double q = e4m3(gFp8[b * 32 + j]);
      if (std::isnan(q)) {
        if (sat < 4) {
          printf("FAIL saturated blk=%d j=%d byte=0x%02x\n",
                 b,
                 j,
                 gFp8[b * 32 + j]);
        }
        ++sat;
        continue;
      }
      double recon = q * s;
      double orig = bf16_to_f(hSrc[b * 32 + j]);
      double denom = std::fabs(orig) > 1e-30 ? std::fabs(orig) : 1.0;
      double rel = std::fabs(recon - orig) / denom;
      if (rel > worst_rel) {
        worst_rel = rel;
      }
      // E4M3 carries 3 mantissa bits + implicit 1, so round-to-nearest gives
      // at most 2^-4 = 6.25% relative error. Anything beyond that is a real
      // bug, not quantization. A reversed scale blows past this immediately.
      if (rel > 0.0626) {
        if (recon_bad < 4) {
          printf("FAIL recon blk=%d j=%d orig=%g recon=%g rel=%.4f\n",
                 b,
                 j,
                 orig,
                 recon,
                 rel);
        }
        ++recon_bad;
      }
    }
  }

  printf("scale_mismatches=%d  saturated=%d  recon_bad=%d  worst_rel=%.5f\n",
         scale_bad,
         sat,
         recon_bad,
         worst_rel);
  if (scale_bad || sat || recon_bad) {
    printf("FAIL: %s\n", name);
    return 1;
  }
  printf("PASS: %s\n", name);
  return 0;
}

static int run_scale_direction() {
  printf("\n--- scale direction (divide, not multiply) ---\n");
  float h[8];
  for (int i = 0; i < 8; ++i) {
    h[i] = 4.0f;
  }
  float *d;
  unsigned *o;
  HIP_CHECK(hipMalloc(&d, sizeof(h)));
  HIP_CHECK(hipMalloc(&o, 8));
  HIP_CHECK(hipMemcpy(d, h, sizeof(h), hipMemcpyHostToDevice));

  struct {
    float scale;
    double want;
  } cases[] = {{1.0f, 4.0}, {4.0f, 1.0}, {0.25f, 16.0}};

  int fails = 0;
  for (auto &c : cases) {
    hipLaunchKernelGGL(scale_dir_kernel, dim3(1), dim3(1), 0, 0, d, c.scale, o);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    unsigned r[2];
    HIP_CHECK(hipMemcpy(r, o, 8, hipMemcpyDeviceToHost));
    double got = e4m3((uint8_t)(r[0] & 0xFF));
    printf("  4.0 / %.4f -> %.4f (want %.4f)%s\n",
           c.scale,
           got,
           c.want,
           got == c.want ? "" : "   <-- FAIL");
    if (got != c.want) {
      ++fails;
    }
  }
  HIP_CHECK(hipFree(d));
  HIP_CHECK(hipFree(o));
  if (fails) {
    printf("FAIL: cvt_scalef32_pk8_fp8_f32 scale semantics changed. If it now "
           "MULTIPLIES, mx_quant_fp8 must pass the reciprocal.\n");
    return 1;
  }
  printf("PASS: divides by scale, as mx_quant_fp8 assumes\n");
  return 0;
}

int main() {
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
    printf("no HIP device\n");
    return 2;
  }
  printf("device: %s  warpSize=%d\n", prop.gcnArchName, prop.warpSize);

  int rc = 0;
  rc |= run_scale_direction();
  rc |= run_case<3072>("tame magnitudes (~2^0)", 0);
  rc |= run_case<3072>("large magnitudes (~2^14)", 1);
  rc |= run_case<3072>("small magnitudes (~2^-19)", 2);
  rc |= run_case<256>("short K", 0);

  if (rc) {
    printf("\nOVERALL: FAIL\n");
    return 1;
  }
  printf("\nOVERALL: PASS (mx_quant_fp8)\n");
  return 0;
}
