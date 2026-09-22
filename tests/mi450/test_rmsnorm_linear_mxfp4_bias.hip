/* Validates gang_rmsnorm_linear_mxfp4_bias_kernel_mi450 -- the first full MX
 * kernel of the port, and the first place the 8-wave32 geometry forces a
 * genuinely different decomposition rather than a rescaled one.
 *
 * ---------------------------------------------------------------------------
 * WHAT IS ACTUALLY NEW HERE
 * ---------------------------------------------------------------------------
 * The pieces this kernel is built from are already pinned by their own tests:
 * mx_k_loop_f4xf8 (test_mx_k_loop, bit-exact at K=256/384/3072), mx_quant_fp8
 * (test_mx_quant, four magnitude regimes), the WMMA fragment layout
 * (test_mx_fp8_layout), and rmsnorm_inline_amd (test_gang_rmsnorm_linear_bias).
 * Re-testing those here would pad the suite without adding information.
 *
 * What is untested until this file exists is the 2D wave split and everything
 * around it:
 *
 *   1. THE TILE/WAVE MAPPING. N_GROUPS waves take distinct output tiles and
 *      WAVES_PER_TILE waves split K within a tile. gfx950 had only the two
 *      corners; OUTPUT_PER_WG=64 -- the real QKV shape -- lands at 4x2, which
 *      neither corner covers, and transcribing gfx950's `OUTPUT_PER_WG >= 64`
 *      branch condition would have routed it into a K-parallel path that
 *      assumes a single output tile.
 *   2. THE LDS CROSS-WAVE REDUCTION. Indexed by absolute warp_id, so it must be
 *      NUM_WAVES*16 floats. Sized from the gfx950 literal 4, waves 4..7 would
 *      overwrite waves 0..3 and the sum would quietly be missing half its
 *      terms -- a plausible-looking wrong answer, not a crash.
 *   3. THE EPILOGUE INDEX MATH. out_n = wg_idx*OUTPUT_PER_WG + tile*16 +
 *      mx_acc_row(lane,i), bias lookup, and the output_stride row pitch.
 *
 * So the four shapes below are chosen to cover the split's corners and middle,
 * not to sample the space randomly:
 *
 *   <1, 64,3072,2880>  N_TILES=4  -> 4 groups x 2 K-waves, 1 tile each  [QKV]
 *   <1,128,3072,2880>  N_TILES=8  -> 8 groups x 1, pure N-parallel corner
 *   <1,256,1024,1024>  N_TILES=16 -> 8 groups x 1, TILES_PER_GROUP=2
 *   <1, 16,1024,1024>  N_TILES=1  -> 1 group x 8 K-waves, pure K-parallel
 *
 * Every shape runs n_wgs_per_xcd=3, so a dropped `wg_idx * OUTPUT_PER_WG`
 * offset or a wrong WG_BYTES stride is visible; with a single workgroup both
 * are identity and invisible. output_stride is deliberately larger than the
 * total output width for the same reason.
 *
 * ---------------------------------------------------------------------------
 * WHY THE REFERENCE IS BUILT FROM THE DEVICE'S OWN norm_output
 * ---------------------------------------------------------------------------
 * RMSNorm ends in rsqrtf, which is a hardware approximation and is not
 * bit-reproducible on the host. Deriving the GEMM reference from a host
 * RMSNorm would therefore fold an irreducible ~1ulp disagreement into every
 * one of 3072 accumulated terms, and the resulting tolerance would be loose
 * enough to hide real bugs.
 *
 * Instead the kernel's norm_output buffer is read back and used as the GEMM
 * reference's input. RMSNorm is then checked separately against a host double
 * reference on its own terms. This splits one loose test into two tight ones.
 *
 * The host then mirrors mx_quant_fp8 exactly -- including E4M3
 * round-to-nearest-even, which was MEASURED, not assumed
 * (tests/mi450/probe_e4m3_tie: all 24 midpoints in binades [1,2) and [4,8),
 * both signs, chose the even-mantissa neighbour). Getting that rule wrong
 * would put a full E4M3 ulp, ~6% relative, on roughly 1 element in 16, which
 * is four orders of magnitude past the bound below -- so the bound itself
 * proves the mirror is right.
 *
 * ---------------------------------------------------------------------------
 * THE TOLERANCE IS DERIVED PER OUTPUT, NOT PICKED
 * ---------------------------------------------------------------------------
 * Two exact sources of disagreement remain, and each is bounded analytically:
 *
 *   accumulation order -- the device sums K products in float in WMMA order,
 *     the host sums in double. Bounded by K * FLT_EPSILON * sum|product|,
 *     which the host computes per output row rather than assuming.
 *   the bf16 store -- the device rounds the final float to bf16 (8-bit
 *     mantissa). Bounded by half a bf16 ulp of the result.
 *
 * The test also counts how many outputs match the bf16-rounded reference
 * BIT-EXACTLY, and MEASUREMENT says that is 100% at all four shapes: bf16's
 * 8-bit significand is coarse enough that float-vs-double accumulation over
 * 3072 terms never reaches a rounding boundary. So the test asserts a 90%
 * floor on that rate as well. It is much tighter than the analytic bound and
 * catches error growth the bound alone would absorb.
 *
 * A structural bug -- wrong tile, dropped K half, clobbered reduction slot --
 * moves outputs by order unity, thousands of times past either check.
 *
 * ---------------------------------------------------------------------------
 * MUTATION RESULTS -- INCLUDING ONE SURVIVOR THIS TEST CANNOT KILL
 * ---------------------------------------------------------------------------
 * Five mutants, run with /tmp/p1/ccmut450.sh (mutant -I PREPENDED; appending it
 * silently produces binaries identical to the clean build):
 *
 *   A  s_reduce indexed [warp_id % 4] -- the gfx950 literal-4 sizing bug
 *      KILLED at QKV (143 over bound, bit-exact 48/192) and at the K-parallel
 *      corner (0/48 bit-exact). Survives the two WAVES_PER_TILE==1 shapes,
 *      correctly: they never touch s_reduce. This is why the shape list spans
 *      the split's corners instead of testing one representative shape.
 *   B  grp/krank swapped (warp_id % N_GROUPS instead of / WAVES_PER_TILE)
 *      KILLED at QKV, 192/192 wrong. Survives shapes where WAVES_PER_TILE==1,
 *      where the two expressions coincide -- again, only the 4x2 middle case
 *      distinguishes them, and gfx950 had no middle case.
 *   C  epilogue uses the lane's supplied row instead of mx_acc_row
 *      KILLED at two shapes, and notably by the POISON check (192 and 384
 *      untouched outputs) as much as by the value check -- a wrong accumulator
 *      row mapping does not just misplace values, it leaves half the output
 *      never written.
 *   E  bias add dropped
 *      KILLED, but only just, and this is the informative one: at the QKV
 *      shape it PASSES the analytic bound outright, because the bias values are
 *      ~2^-3 while the accumulated results are ~2^6, so dropping them moves
 *      results by less than the float-accumulation slack. What kills it is the
 *      bit-exact rate collapsing (119/384, 104/768). Without that check this
 *      mutant would have survived at every shape whose bias is small relative
 *      to its output. The 90% bit-exact floor is not belt-and-braces.
 *
 *   D  the __syncthreads() between writing s_reduce and reading it, REMOVED
 *      *** SURVIVED. All four shapes pass. ***
 *      This is not a gap this test can close. The K-parallel corner has
 *      TILES_PER_GROUP==1, so there is exactly one write-then-read of s_reduce
 *      with no barrier of any kind between them, across 8 waves -- and it still
 *      produced bit-exact results. That means FFM is executing the waves
 *      closely enough in lockstep that the race never resolves the wrong way.
 *      Real hardware gives no such guarantee.
 *
 *      Do not read MUT-D's survival as evidence the barrier is unnecessary. It
 *      is evidence that FFM cannot test barriers, which is a documented
 *      property of the model, not of this kernel. Removing that __syncthreads()
 *      is a real bug that this suite -- and any suite runnable under FFM --
 *      will report as green. It needs silicon.
 *
 * DISCRIMINATING POWER, asserted before any pass is reported: output rows must
 * differ from each other, workgroups must differ from each other, and for the
 * K-split shapes the two K halves must contribute differently -- otherwise
 * "dropped half the K extent" would pass by coincidence. All data comes from a
 * mixing hash of the indices, never an affine function of them.
 *
 *   ./run_ffm.sh /abs/path/to/test_rmsnorm_linear_mxfp4_bias
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <vector>

#include "mirage/persistent_kernel/tasks/mi300/gang_rmsnorm_linear_bias_mi300.cuh"
#include "mirage/persistent_kernel/tasks/mi450/gang_rmsnorm_linear_mxfp4_bias_mi450.cuh"

#define HIP_CHECK(x)                                                           \
  do {                                                                         \
    hipError_t e = (x);                                                        \
    if (e != hipSuccess) {                                                     \
      printf("HIP error %s at line %d\n", hipGetErrorString(e), __LINE__);     \
      return 2;                                                                \
    }                                                                          \
  } while (0)

// ── Independent decode tables ───────────────────────────────────────────────
static double const kE2M1[16] = {0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
                                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};

static double e4m3(uint8_t b) {
  int s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 0x7;
  double mag = (e == 0) ? std::ldexp((double)m / 8.0, -6)
                        : std::ldexp(1.0 + (double)m / 8.0, e - 7);
  return s ? -mag : mag;
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
static unsigned short f_to_bf16_rne(float f) {
  unsigned u;
  __builtin_memcpy(&u, &f, 4);
  return (unsigned short)((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

static unsigned mix(unsigned x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

// ── Host mirror of mx_quant_fp8 ─────────────────────────────────────────────
// E8M0 exponent: written from log2/ceil, deliberately unlike the device's
// float-bit manipulation, so a shared misunderstanding must be made twice.
static uint8_t host_e8m0(float amax) {
  if (amax == 0.0f) {
    return 0;
  }
  double need = (double)amax / 448.0;
  int e = (int)std::ceil(std::log2(need));
  while (std::ldexp(1.0, e) < need) {
    ++e;
  }
  while (e > -127 && std::ldexp(1.0, e - 1) >= need) {
    --e;
  }
  return (uint8_t)std::max(0, std::min(255, e + 127));
}

// float -> E4M3 with round-to-nearest-EVEN, the measured hardware rule.
// Unified across normals and subnormals: with e = max(floor(log2|x|), -6), the
// quantum is 2^(e-3) and n = RNE(|x| / quantum) lands in 0..7 for subnormals
// and 8..15 for normals, which is exactly the encoding boundary.
static uint8_t host_f2e4m3(double x, int *saturated) {
  uint8_t sign = (x < 0 || (x == 0 && std::signbit(x))) ? 0x80 : 0x00;
  double a = std::fabs(x);
  if (a == 0.0) {
    return sign;
  }
  int e = (int)std::floor(std::log2(a));
  if (e < -6) {
    e = -6;
  }
  double n = std::nearbyint(a / std::ldexp(1.0, e - 3)); // FE_TONEAREST = RNE
  if (n >= 16.0) { // mantissa carried into the next binade
    ++e;
    n = 8.0;
  }
  if (e > 8 || (e == 8 && n > 14.0)) { // past 448, E4M3's largest finite
    *saturated += 1;
    return (uint8_t)(sign | 0x7E);
  }
  if (e == -6 && n < 8.0) {
    return (uint8_t)(sign | (int)n); // subnormal: exp field 0
  }
  return (uint8_t)(sign | ((e + 7) << 3) | ((int)n - 8));
}

// Mirror of mx_quant_fp8 over one row: 32-element blocks, scale DIVIDES.
static void host_quant_fp8(unsigned short const *row,
                           int K,
                           std::vector<uint8_t> &fp8,
                           std::vector<uint8_t> &scl,
                           int *saturated) {
  int const nblk = K / 32;
  fp8.assign(K, 0);
  scl.assign(nblk, 0);
  for (int b = 0; b < nblk; ++b) {
    float amax = 0.f;
    for (int j = 0; j < 32; ++j) {
      amax = std::fmax(amax, std::fabs(bf16_to_f(row[b * 32 + j])));
    }
    uint8_t se = host_e8m0(amax);
    double s = (se == 0) ? 1.0 : std::ldexp(1.0, (int)se - 127);
    for (int j = 0; j < 32; ++j) {
      fp8[b * 32 + j] =
          host_f2e4m3((double)bf16_to_f(row[b * 32 + j]) / s, saturated);
    }
    scl[b] = se;
  }
}

// ── Launcher ────────────────────────────────────────────────────────────────
#if defined(__gfx1250__) || !defined(__HIP_DEVICE_COMPILE__)
template <int BATCH, int OPW, int RED, int AHD>
__global__ __launch_bounds__(256) void rnlm_kernel(void const *nin,
                                                   void const *nw,
                                                   void *nout,
                                                   void const *w,
                                                   void const *bias,
                                                   void *out,
                                                   int ntok,
                                                   int nwg,
                                                   int ostride) {
#if defined(MIRAGE_ARCH_GFX1250)
  kernel::gang_rmsnorm_linear_mxfp4_bias_kernel_mi450<BATCH, OPW, RED, AHD>(
      nin, nw, nout, w, bias, out, ntok, nwg, ostride, (int)blockIdx.x);
#endif
}
#endif

// ── One shape ───────────────────────────────────────────────────────────────
template <int OPW, int RED, int AHD>
static int run_case(char const *name) {
  constexpr int NWG = 3;
  constexpr int NBLK = RED / 32;
  constexpr int WG_DATA = OPW * (RED / 2);
  constexpr int WG_SCL = OPW * NBLK;
  constexpr int WG_BYTES = WG_DATA + WG_SCL;
  constexpr int NOUT = NWG * OPW;
  constexpr int OSTRIDE = NOUT + 7; // deliberately not NOUT

  constexpr int N_TILES = OPW / 16;
  constexpr int N_GROUPS = N_TILES < 8 ? N_TILES : 8;
  constexpr int WPT = 8 / N_GROUPS;
  printf("\n--- %s  OPW=%d K=%d hidden=%d -> %d groups x %d K-waves, %d "
         "tiles/group ---\n",
         name,
         OPW,
         RED,
         AHD,
         N_GROUPS,
         WPT,
         N_TILES / N_GROUPS);

  // ── Inputs ────────────────────────────────────────────────────────────────
  // Hidden activations and norm weight, zero-padded past AHD (the kernel's
  // RMSNorm divides by AHD but sums over RED, and relies on the pad being 0).
  std::vector<unsigned short> hIn(RED, 0), hNw(RED, 0);
  for (int k = 0; k < AHD; ++k) {
    unsigned h = mix((unsigned)(k * 131 + 17));
    float v = std::ldexp(1.0f + (float)(h & 0xFF) / 256.0f,
                         -2 + (int)((h >> 12) % 4));
    if ((h >> 20) & 1) {
      v = -v;
    }
    hIn[k] = f_to_bf16_rne(v);
    unsigned g = mix((unsigned)(k * 977 + 5));
    hNw[k] = f_to_bf16_rne(std::ldexp(1.0f + (float)(g & 0xFF) / 256.0f, -1));
  }

  // MXFP4 weights: nibble magnitudes restricted to codes 0..4 (0..2.0) so the
  // accumulated magnitude stays modest and the analytic bound stays tight.
  std::vector<uint8_t> hW((size_t)NWG * WG_BYTES);
  for (int wg = 0; wg < NWG; ++wg) {
    uint8_t *d = hW.data() + (size_t)wg * WG_BYTES;
    uint8_t *s = d + WG_DATA;
    for (int r = 0; r < OPW; ++r) {
      for (int k = 0; k < RED; k += 2) {
        unsigned h0 = mix((unsigned)(wg * 7919 + r * 977 + k * 31 + 1));
        unsigned h1 = mix((unsigned)(wg * 6151 + r * 613 + k * 17 + 5));
        uint8_t lo = (uint8_t)((h0 % 5) | ((h0 >> 8 & 1) << 3));
        uint8_t hi = (uint8_t)((h1 % 5) | ((h1 >> 8 & 1) << 3));
        d[(size_t)r * (RED / 2) + k / 2] = (uint8_t)(lo | (hi << 4));
      }
      for (int b = 0; b < NBLK; ++b) {
        s[(size_t)r * NBLK + b] =
            (uint8_t)(126 + (mix((unsigned)(wg * 31 + r * 71 + b * 13 + 3)) % 3));
      }
    }
  }

  std::vector<unsigned short> hBias(NOUT);
  for (int n = 0; n < NOUT; ++n) {
    unsigned h = mix((unsigned)(n * 2311 + 29));
    float v = std::ldexp(1.0f + (float)(h & 0xFF) / 256.0f, -3);
    hBias[n] = f_to_bf16_rne(((h >> 20) & 1) ? -v : v);
  }

  // ── Device ────────────────────────────────────────────────────────────────
  unsigned short *dIn, *dNw, *dNout, *dBias, *dOut;
  uint8_t *dW;
  HIP_CHECK(hipMalloc(&dIn, RED * 2));
  HIP_CHECK(hipMalloc(&dNw, RED * 2));
  HIP_CHECK(hipMalloc(&dNout, RED * 2));
  HIP_CHECK(hipMalloc(&dW, hW.size()));
  HIP_CHECK(hipMalloc(&dBias, NOUT * 2));
  HIP_CHECK(hipMalloc(&dOut, OSTRIDE * 2));
  HIP_CHECK(hipMemcpy(dIn, hIn.data(), RED * 2, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dNw, hNw.data(), RED * 2, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dW, hW.data(), hW.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dBias, hBias.data(), NOUT * 2, hipMemcpyHostToDevice));
  // Poison both outputs: a pass must not come from untouched memory, and the
  // padded tail of the output row must stay untouched.
  HIP_CHECK(hipMemset(dOut, 0xAB, OSTRIDE * 2));
  HIP_CHECK(hipMemset(dNout, 0xAB, RED * 2));

  int const smem = kernel::mi450::mx_rnlm_smem_bytes<RED>();
  printf("dynamic LDS = %d bytes\n", smem);
  hipLaunchKernelGGL((rnlm_kernel<1, OPW, RED, AHD>),
                     dim3(NWG),
                     dim3(256),
                     smem,
                     0,
                     dIn,
                     dNw,
                     dNout,
                     dW,
                     dBias,
                     dOut,
                     1,
                     NWG,
                     OSTRIDE);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<unsigned short> gNout(RED), gOut(OSTRIDE);
  HIP_CHECK(hipMemcpy(gNout.data(), dNout, RED * 2, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(gOut.data(), dOut, OSTRIDE * 2, hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(dIn));
  HIP_CHECK(hipFree(dNw));
  HIP_CHECK(hipFree(dNout));
  HIP_CHECK(hipFree(dW));
  HIP_CHECK(hipFree(dBias));
  HIP_CHECK(hipFree(dOut));

  // ── Check 1: RMSNorm on its own terms ─────────────────────────────────────
  {
    double ssq = 0.0;
    for (int k = 0; k < RED; ++k) {
      double v = bf16_to_f(hIn[k]);
      ssq += v * v;
    }
    double rcp = 1.0 / std::sqrt(ssq / (double)AHD + 1e-5);
    int bad = 0;
    double worst = 0.0;
    for (int k = 0; k < RED; ++k) {
      double want = bf16_to_f(hIn[k]) * rcp * bf16_to_f(hNw[k]);
      double got = bf16_to_f(gNout[k]);
      // Finiteness before tolerance. `d > tol` is false when d is NaN, so a
      // kernel emitting NaN for the whole row would report mismatches=0 and
      // worst_rel=0.000e+00 and pass. The poison check below does not cover it
      // either -- arithmetic NaN has different bits from 0xABAB, so the slot
      // reads as legitimately written. This kernel is specifically exposed: a
      // wrong cross-wave sum of squares can go negative, and rsqrt of a
      // negative is NaN.
      if (!std::isfinite(got) || !std::isfinite(want)) {
        if (bad < 4) {
          printf("FAIL rmsnorm k=%d: NON-FINITE got=%f want=%f -- a tolerance "
                 "check cannot detect this\n", k, got, want);
        }
        ++bad;
        continue;
      }
      // bf16 half-ulp, plus slack for rsqrtf's ~1ulp float approximation.
      double tol = std::fabs(want) * (1.0 / 256.0 + 1e-6) + 1e-30;
      double d = std::fabs(got - want);
      if (std::fabs(want) > 0 && d / std::fabs(want) > worst) {
        worst = d / std::fabs(want);
      }
      if (d > tol) {
        if (bad < 4) {
          printf("FAIL rmsnorm k=%d got=%.9g want=%.9g\n", k, got, want);
        }
        ++bad;
      }
    }
    if (gNout[RED - 1] == (unsigned short)0xABAB) {
      printf("FAIL: norm_output untouched (poison survived)\n");
      return 1;
    }
    printf("rmsnorm: mismatches=%d worst_rel=%.3e\n", bad, worst);
    if (bad) {
      return 1;
    }
  }

  // ── Host mirror of the device's quantization of that same row ─────────────
  std::vector<uint8_t> qFp8, qScl;
  int saturated = 0;
  host_quant_fp8(gNout.data(), RED, qFp8, qScl, &saturated);
  if (saturated) {
    printf("FAIL(test): host mirror saturated %d elements -- the E8M0 rule "
           "must keep |x|/scale <= 448\n",
           saturated);
    return 1;
  }

  std::vector<double> tok(RED);
  for (int k = 0; k < RED; ++k) {
    tok[k] = e4m3(qFp8[k]) * e8m0(qScl[k / 32]);
  }

  // ── Reference GEMM, in double, with a per-row error bound ─────────────────
  std::vector<double> ref(NOUT), bound(NOUT);
  for (int wg = 0; wg < NWG; ++wg) {
    uint8_t const *d = hW.data() + (size_t)wg * WG_BYTES;
    uint8_t const *s = d + WG_DATA;
    for (int r = 0; r < OPW; ++r) {
      double acc = 0.0, absacc = 0.0;
      for (int k = 0; k < RED; ++k) {
        uint8_t p = d[(size_t)r * (RED / 2) + k / 2];
        uint8_t code = (k & 1) ? (uint8_t)(p >> 4) : (uint8_t)(p & 0xF);
        double w = kE2M1[code] * e8m0(s[(size_t)r * NBLK + k / 32]);
        double prod = w * tok[k];
        acc += prod;
        absacc += std::fabs(prod);
      }
      int n = wg * OPW + r;
      double v = acc + (double)bf16_to_f(hBias[n]);
      ref[n] = v;
      // float accumulation order over RED terms, then the bf16 store.
      //
      // bf16 keeps 8 significand bits (7 stored + implicit), so for
      // 2^e <= |v| < 2^(e+1) the ulp is 2^(e-7) and a half-ulp is 2^(e-8),
      // which is at most |v| * 2^-8. Writing 2^-9 here -- one bit tighter,
      // which is the natural slip if you count only the stored mantissa --
      // makes the bound half the real rounding error, and outputs the device
      // rounded perfectly then report as failures. That is how this line was
      // first written; the tell was over_bound > 0 while bit-exact was 100%.
      double half_ulp = std::ldexp(std::fabs(v), -8);
      bound[n] = (double)RED * 1.1920929e-7 * absacc + half_ulp * 1.001 + 1e-30;
    }
  }

  // ── Discriminating power ──────────────────────────────────────────────────
  {
    int distinct_rows = 0, distinct_wgs = 0, distinct_khalves = 0;
    for (int r = 1; r < OPW; ++r) {
      if (std::fabs(ref[r] - ref[0]) > 1e-9) {
        ++distinct_rows;
      }
    }
    for (int wg = 1; wg < NWG; ++wg) {
      if (std::fabs(ref[wg * OPW] - ref[0]) > 1e-9) {
        ++distinct_wgs;
      }
    }
    // Would dropping one wave's K slice be visible? Compare per-slice partials
    // of output row 0 -- if they matched, a lost slice could pass.
    if (WPT > 1) {
      double first = 0.0;
      for (int sl = 0; sl < WPT; ++sl) {
        double p = 0.0;
        for (int k = sl * (RED / WPT); k < (sl + 1) * (RED / WPT); ++k) {
          uint8_t pk = hW[(size_t)k / 2];
          uint8_t code =
              (k & 1) ? (uint8_t)(pk >> 4) : (uint8_t)(pk & 0xF);
          p += kE2M1[code] * e8m0(hW[WG_DATA + k / 32]) * tok[k];
        }
        if (sl == 0) {
          first = p;
        } else if (std::fabs(p - first) > 1e-9) {
          ++distinct_khalves;
        }
      }
      if (distinct_khalves == 0) {
        printf("FAIL(test): K slices contribute identically -- a dropped slice "
               "would be invisible\n");
        return 1;
      }
    }
    if (distinct_rows == 0 || distinct_wgs == 0) {
      printf("FAIL(test): degenerate reference (rows=%d wgs=%d)\n",
             distinct_rows,
             distinct_wgs);
      return 1;
    }
    printf("discriminating: distinct rows=%d/%d  wgs=%d/%d  K-slices=%d\n",
           distinct_rows,
           OPW - 1,
           distinct_wgs,
           NWG - 1,
           distinct_khalves);
  }

  // ── Compare ───────────────────────────────────────────────────────────────
  int fails = 0, untouched = 0, bitexact = 0;
  double worst_ratio = 0.0;
  for (int n = 0; n < NOUT; ++n) {
    if (gOut[n] == (unsigned short)0xABAB) {
      if (untouched < 4) {
        printf("FAIL untouched out_n=%d (poison survived)\n", n);
      }
      ++untouched;
      continue;
    }
    double got = bf16_to_f(gOut[n]);
    // Finiteness before the bound check -- `d > bound[n]` is false for NaN, so
    // an all-NaN output would report over_bound=0 and worst=0.000 x bound. The
    // bit-exact counter above is NaN-safe but is a quality signal, not the
    // failure gate.
    if (!std::isfinite(got) || !std::isfinite(ref[n])) {
      if (fails < 6) {
        printf("FAIL out_n=%d: NON-FINITE got=%f want=%f -- a bound check "
               "cannot detect this\n", n, got, ref[n]);
      }
      ++fails;
      continue;
    }
    double d = std::fabs(got - ref[n]);
    double ratio = d / bound[n];
    if (ratio > worst_ratio) {
      worst_ratio = ratio;
    }
    if (gOut[n] == f_to_bf16_rne((float)ref[n])) {
      ++bitexact;
    }
    if (d > bound[n]) {
      if (fails < 6) {
        printf("FAIL out_n=%d got=%.9g want=%.9g |d|=%.4g bound=%.4g\n",
               n,
               got,
               ref[n],
               d,
               bound[n]);
      }
      ++fails;
    }
  }
  // The padded tail of the output row must be untouched: writing past NOUT
  // means out_n or output_stride is wrong in a way NOUT-only checks miss.
  int clobbered = 0;
  for (int n = NOUT; n < OSTRIDE; ++n) {
    if (gOut[n] != (unsigned short)0xABAB) {
      ++clobbered;
    }
  }

  printf("untouched=%d  clobbered_pad=%d  over_bound=%d  bit-exact=%d/%d  "
         "worst=%.3f x bound\n",
         untouched,
         clobbered,
         fails,
         bitexact,
         NOUT,
         worst_ratio);

  // Bit-exactness is the tight signal, and measurement says it is 100% at
  // every shape here: bf16's 8-bit significand is coarse enough that the
  // float-vs-double accumulation difference over 3072 terms never reaches the
  // rounding boundary. So hold it to that. A drop below 90% would mean the
  // accumulated error grew by orders of magnitude -- the loose analytic bound
  // above might still pass, and this is what would notice.
  if (bitexact < (NOUT * 9) / 10) {
    printf("FAIL: bit-exact rate collapsed to %d/%d -- accumulation error grew "
           "far beyond bf16 rounding even though the analytic bound held\n",
           bitexact,
           NOUT);
    return 1;
  }

  if (untouched || clobbered || fails) {
    printf("FAIL: %s\n", name);
    return 1;
  }
  printf("PASS: %s\n", name);
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
  rc |= run_case<64, 3072, 2880>("QKV shape (4x2, the middle case)");
  rc |= run_case<128, 3072, 2880>("pure N-parallel corner");
  rc |= run_case<256, 1024, 1024>("N-parallel, 2 tiles per group");
  rc |= run_case<16, 1024, 1024>("pure K-parallel corner");

  if (rc) {
    printf("\nOVERALL: FAIL\n");
    return 1;
  }
  printf("\nOVERALL: PASS (gang_rmsnorm_linear_mxfp4_bias_mi450)\n");
  return 0;
}
