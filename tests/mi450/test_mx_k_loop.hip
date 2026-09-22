/* Validates mx_k_loop_f4xf8 -- the factored MXFP4 inner GEMM loop that all
 * eight gfx950 MX pipelines collapse onto.
 *
 * WHAT THIS COVERS THAT test_mx_fp8_layout.hip DOES NOT
 *
 * That test is a single 16x16x128 tile: one K-tile, one scale block per
 * operand, no loop. It pins the fragment layout and nothing else. This test
 * runs a real multi-tile K extent, which is where the loop's own bugs live:
 *
 *   - the two-buffer ping-pong (does tile k1 get consumed with k1's scales,
 *     or with k0's?)
 *   - scale-block advance (k/32, four blocks per K-tile -- an off-by-one here
 *     scales an entire 32-wide slab by the wrong power of two)
 *   - the odd-K_ITERS tail (silently dropping the last tile is invisible to
 *     any test whose K_ITERS is even)
 *   - the k2 < REDUCTION_SIZE prefetch bound
 *
 * Three K extents are run deliberately:
 *   K=256  (2 tiles)  minimum legal; loop body runs exactly once, no tail
 *   K=384  (3 tiles)  ODD -- exercises the tail path
 *   K=3072 (24 tiles) the real GPT-OSS padded hidden size
 *
 * DO NOT DROP K=384 AS REDUNDANT. It looks redundant -- Fleet registers no odd
 * shape, and K=256 and K=3072 both cover the loop -- but mutation testing says
 * otherwise. Deleting the `if constexpr (K_ITERS % 2 == 1)` tail from
 * mx_k_loop_f4xf8 is caught at K=384 (256 mismatches) and by NOTHING ELSE:
 * that mutant passes K=256 and K=3072 exactly, because with an even tile count
 * the tail never runs. K=384 is the only thing standing between a silently
 * dropped final K-tile and a green suite.
 *
 * Two more mutants, both killed at every extent: consuming tile k1 with tile
 * kt's token scale block (worst_abs 3.8e+01 at K=256, 1.6e+02 at K=3072), and
 * reverting the token load to the falsified FP4 K rule (9.9e+01 throughout).
 *
 * A note on running mutants: prepend the shadowing -I, do not append it.
 * cctest450.sh puts "$@" after its own -I flags, so an appended mutant path
 * never wins and all three of these "survived" on the first attempt -- the
 * binaries were identical to the clean build. /tmp/p1/ccmut450.sh exists to
 * get the order right.
 *
 * EXACTNESS. Every product here is exactly representable and every partial
 * sum stays well inside float's 24-bit significand, so the comparison against
 * a host `double` reference is bit-exact and the tolerance is literally zero.
 * That is deliberate: an approximate tolerance on a 3072-long reduction hides
 * exactly the kind of one-tile-dropped error this test exists to catch. E2M1
 * codes are restricted to small magnitudes and E8M0 scales to a narrow band
 * around 2^0 so the accumulated magnitude cannot reach the point where float
 * addition rounds.
 *
 * DISCRIMINATING POWER. Values come from a mixing hash of the index, not an
 * affine function of it -- affine data makes half-swaps and transposes
 * invisible because the values they disagree about happen to be equal (the
 * lesson from the attention port). The test asserts its reference is
 * asymmetric with distinct rows before it will report a pass, and it asserts
 * that the K-tiles actually differ from one another, so "dropped the last
 * tile" cannot coincidentally match.
 *
 *   ./run_ffm.sh /abs/path/to/test_mx_k_loop
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <vector>

#include "mirage/persistent_kernel/tasks/mi450/mx_pipeline_mi450.cuh"

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

static double e4m3(uint8_t b) {
  int sign = (b >> 7) & 1;
  int exp = (b >> 3) & 0xF;
  int man = b & 0x7;
  double mag = (exp == 0) ? std::ldexp((double)man / 8.0, -6)
                          : std::ldexp(1.0 + (double)man / 8.0, exp - 7);
  return sign ? -mag : mag;
}

static double e8m0(uint8_t e) {
  return std::ldexp(1.0, (int)e - 127);
}

static unsigned mix(unsigned x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

constexpr int M = 16, N = 16;

#if defined(__gfx1250__) || !defined(__HIP_DEVICE_COMPILE__)
// One wave, one 16x16 output tile, full K extent through the factored loop.
// Weights come from global (as in the real kernels); tokens are staged into
// LDS first, again as in the real kernels.
template <int K>
__global__ __launch_bounds__(32) void k_loop_kernel(uint8_t const *w,
                                                    uint8_t const *ws,
                                                    uint8_t const *t,
                                                    uint8_t const *ts,
                                                    float *out) {
#if defined(MIRAGE_ARCH_GFX1250)
  using namespace kernel::mi450;
  constexpr int NBLK = K / 32;

  extern __shared__ char smem[];
  uint8_t *s_tok = (uint8_t *)smem;
  uint8_t *s_scl = s_tok + K;

  // Stage token row 0 + its scales into LDS.
  for (int i = threadIdx.x; i < K; i += 32) {
    s_tok[i] = t[i];
  }
  for (int i = threadIdx.x; i < NBLK; i += 32) {
    s_scl[i] = ts[i];
  }
  __syncthreads();

  int lane = threadIdx.x & 31;
  int w_row = mx_operand_mn(lane); // the row THIS LANE SUPPLIES

  mx_f32x8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  acc = mx_k_loop_f4xf8<K>(
      mx_to_global(w), mx_to_global(ws), w_row, s_tok, s_scl, acc);

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    // mx_acc_row, NOT w_row -- the accumulator holds a different row than the
    // lane supplied.
    out[mx_acc_row(lane, i) * N + mx_acc_col(lane)] = acc[i];
  }
#endif
}
#endif

template <int K>
static int run_case() {
  constexpr int NBLK = K / 32;
  printf("\n--- K=%d (%d tiles, %s) ---\n",
         K,
         K / 128,
         (K / 128) % 2 ? "ODD: exercises tail" : "even");

  std::vector<uint8_t> hW((size_t)M * K / 2), hWs((size_t)M * NBLK);
  std::vector<uint8_t> hT((size_t)K), hTs((size_t)NBLK);

  // Weight nibbles: magnitudes 0..2.0 only (codes 0..4 and their negatives),
  // keeping every product small enough that a 3072-term sum stays exact.
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; k += 2) {
      unsigned h0 = mix((unsigned)(m * 977 + k * 31 + 1));
      unsigned h1 = mix((unsigned)(m * 613 + k * 17 + 5));
      uint8_t lo = (uint8_t)((h0 % 5) | ((h0 >> 8 & 1) << 3));
      uint8_t hi = (uint8_t)((h1 % 5) | ((h1 >> 8 & 1) << 3));
      hW[(size_t)m * (K / 2) + k / 2] = (uint8_t)(lo | (hi << 4));
    }
  }
  // Token bytes: E4M3 with exponent field 7 (2^0 scale) and full mantissa, so
  // values are 1.0 + n/8 -- exact in 3 bits, and products with the weights
  // above are exact in 6.
  for (int k = 0; k < K; ++k) {
    unsigned h = mix((unsigned)(k * 53 + 11));
    uint8_t sign = (uint8_t)((h >> 20) & 1);
    uint8_t man = (uint8_t)(h & 0x7);
    hT[k] = (uint8_t)((sign << 7) | (7 << 3) | man);
  }
  // E8M0 scales: 2^-1 .. 2^1. Varying per block so a scale-block off-by-one
  // is visible, but bounded so magnitudes stay tame.
  for (int m = 0; m < M; ++m) {
    for (int b = 0; b < NBLK; ++b) {
      hWs[(size_t)m * NBLK + b] =
          (uint8_t)(126 + (mix((unsigned)(m * 71 + b * 13 + 3)) % 3));
    }
  }
  for (int b = 0; b < NBLK; ++b) {
    hTs[b] = (uint8_t)(126 + (mix((unsigned)(b * 91 + 7)) % 3));
  }

  // ── Host reference in double ────────────────────────────────────────────
  // Only token row 0 is real, so every output column n holds the same value;
  // the interesting axis is m. Compute the full 16x16 anyway so a transpose
  // shows up as a row/column mismatch rather than silently passing.
  std::vector<double> ref((size_t)M * N, 0.0);
  for (int m = 0; m < M; ++m) {
    double s = 0.0;
    for (int k = 0; k < K; ++k) {
      uint8_t packed = hW[(size_t)m * (K / 2) + k / 2];
      uint8_t code = (k & 1) ? (uint8_t)(packed >> 4) : (uint8_t)(packed & 0xF);
      double w = kE2M1[code] * e8m0(hWs[(size_t)m * NBLK + k / 32]);
      double t = e4m3(hT[k]) * e8m0(hTs[k / 32]);
      s += w * t;
    }
    for (int n = 0; n < N; ++n) {
      ref[(size_t)m * N + n] = s;
    }
  }

  // ── Guards: would this data notice a bug? ───────────────────────────────
  {
    int distinct_rows = 0;
    for (int m = 1; m < M; ++m) {
      if (std::fabs(ref[(size_t)m * N] - ref[0]) > 1e-12) {
        ++distinct_rows;
      }
    }
    // Per-tile partial sums must differ, or "dropped the last K-tile" could
    // pass by coincidence.
    int distinct_tiles = 0;
    double first_tile = 0.0;
    for (int tile = 0; tile < K / 128; ++tile) {
      double s = 0.0;
      for (int k = tile * 128; k < (tile + 1) * 128; ++k) {
        uint8_t packed = hW[k / 2];
        uint8_t code =
            (k & 1) ? (uint8_t)(packed >> 4) : (uint8_t)(packed & 0xF);
        s += kE2M1[code] * e8m0(hWs[k / 32]) * e4m3(hT[k]) * e8m0(hTs[k / 32]);
      }
      if (tile == 0) {
        first_tile = s;
      } else if (std::fabs(s - first_tile) > 1e-12) {
        ++distinct_tiles;
      }
    }
    if (distinct_rows == 0 || (K / 128 > 1 && distinct_tiles == 0)) {
      printf("FAIL(test): degenerate data (distinct_rows=%d distinct_tiles=%d)"
             " -- a dropped tile or swapped row would be invisible\n",
             distinct_rows,
             distinct_tiles);
      return 1;
    }
    printf("discriminating: distinct rows=%d  distinct K-tiles=%d\n",
           distinct_rows,
           distinct_tiles);
  }

  uint8_t *dW, *dWs, *dT, *dTs;
  float *dOut;
  HIP_CHECK(hipMalloc(&dW, hW.size()));
  HIP_CHECK(hipMalloc(&dWs, hWs.size()));
  HIP_CHECK(hipMalloc(&dT, hT.size()));
  HIP_CHECK(hipMalloc(&dTs, hTs.size()));
  HIP_CHECK(hipMalloc(&dOut, (size_t)M * N * sizeof(float)));
  HIP_CHECK(hipMemcpy(dW, hW.data(), hW.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dWs, hWs.data(), hWs.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dT, hT.data(), hT.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dTs, hTs.data(), hTs.size(), hipMemcpyHostToDevice));

  std::vector<float> poison((size_t)M * N, -98765.f);
  HIP_CHECK(hipMemcpy(dOut,
                      poison.data(),
                      poison.size() * sizeof(float),
                      hipMemcpyHostToDevice));

  hipLaunchKernelGGL(
      k_loop_kernel<K>, dim3(1), dim3(32), K + NBLK, 0, dW, dWs, dT, dTs, dOut);
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
      double d = std::fabs((double)g - r);
      if (d > worst) {
        worst = d;
      }
      // Exact: see the header note on why the tolerance is zero.
      if (d != 0.0) {
        if (fails < 6) {
          printf("FAIL m=%d n=%d got=%.9g want=%.9g\n", m, n, g, r);
        }
        ++fails;
      }
    }
  }

  HIP_CHECK(hipFree(dW));
  HIP_CHECK(hipFree(dWs));
  HIP_CHECK(hipFree(dT));
  HIP_CHECK(hipFree(dTs));
  HIP_CHECK(hipFree(dOut));

  printf("untouched=%d  mismatches=%d  worst_abs=%.3e\n",
         untouched,
         fails,
         worst);
  if (untouched || fails) {
    printf("FAIL: K=%d\n", K);
    return 1;
  }
  printf("PASS: K=%d exact\n", K);
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
  rc |= run_case<256>();  // 2 tiles: minimum legal, no tail
  rc |= run_case<384>();  // 3 tiles: ODD, exercises the tail
  rc |= run_case<3072>(); // 24 tiles: real GPT-OSS padded hidden size
  if (rc) {
    printf("\nOVERALL: FAIL\n");
    return 1;
  }
  printf("\nOVERALL: PASS (mx_k_loop_f4xf8 exact at K=256/384/3072)\n");
  return 0;
}
