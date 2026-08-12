/* Copyright 2025 CMU
 *
 * Determines the per-lane operand layout for FP8 E4M3 as src0 (cbsz=0) of
 * v_mfma_scale_f32_16x16x128_f8f6f4 on gfx950.
 *
 * Why this test exists: the MXFP4 path feeds FP4 weights as src0 and FP8
 * tokens as src1, and the two use visibly different addressing. src1 is
 * gathered as two 16-byte chunks 64 bytes apart (_gang_load_fp8_mfma_b);
 * src0 is a single contiguous 16-byte read at g*16. Both are consistent with
 * "lane g owns 32 consecutive K elements" only because FP4 packs two elements
 * per byte -- at 4 bits, g*16 bytes IS element g*32. Moving weights to FP8
 * doubles the byte stride, and the FP4 code cannot tell us which grouping the
 * hardware wants once the two stop coinciding.
 *
 * Two independent unknowns, so the test sweeps them separately:
 *
 *   LOADER  -- which buffer bytes go into which register slot, i.e. which
 *              A element gets paired with which B element. Getting this
 *              wrong scrambles the dot product outright.
 *   SCALES  -- which 32 elements share one E8M0 exponent. The MFMA takes one
 *              scale per lane per operand, so the quantizer has to group
 *              elements exactly the way the hardware does. Getting this wrong
 *              is subtle: scales are powers of two and neighbouring blocks
 *              often share an exponent, so most elements survive and the
 *              error looks like ordinary quantization noise.
 *
 * A K-major FP8 buffer is packed once per (loader, scales) pair, and the same
 * bytes and exponents are then summed two ways -- through the MFMA, and by a
 * scalar dequantize-and-sum. Quantization error is common to both sides, so
 * only a layout mismatch can separate them.
 *
 * Measured on gfx950 (MI355X):
 *
 *   loader   scales      max_rel
 *   split    split32     4.186e-01
 *   split    contig32    8.270e-06   <- correct
 *   split    uniform128  8.270e-06      (degenerate, see below)
 *   contig   split32     1.071e+00
 *   contig   contig32    9.423e-01
 *   contig   uniform128  9.273e-01
 *
 * So the two mappings are independent of each other:
 *
 *   data   lane g holds the split gather, [kt+g*16, +16) and [kt+g*16+64, +16)
 *          -- byte for byte the same as src1. One loader serves both operands.
 *   scale  lane 16*g+m carries the exponent for row m over the CONTIGUOUS
 *          K block [g*32, g*32+32), which is not the set of elements that lane
 *          holds. The scale operand is addressed by matrix position (row,
 *          k-block), not by whatever data happens to sit in the lane: a wave's
 *          64 lanes cover exactly the 16 rows x 4 k-blocks of an A tile.
 *
 * That independence is invisible at 4 bits, which is why the MXFP4 path never
 * had to make the distinction -- with two elements per byte, a lane's 16-byte
 * read IS elements [g*32, g*32+32), so the data and scale groupings coincide.
 * At 8 bits they come apart, and grouping the quantizer the way the lane reads
 * (split32) costs 42% relative error.
 *
 * uniform128 passing proves nothing: it gives all four lanes one shared
 * exponent, which is consistent with every grouping. It is in the sweep only
 * to keep a known-good row in the table.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

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

typedef int __attribute__((ext_vector_type(4))) i32x4_t;
typedef int __attribute__((ext_vector_type(8))) i32x8_t;
typedef float __attribute__((ext_vector_type(4))) f32x4_t;
typedef short __attribute__((ext_vector_type(2))) fp8x4_t;

constexpr int K = 512; // 4 MFMA k-tiles
constexpr int N = 64;  // one workgroup: 4 waves x 16 rows
constexpr int K_PER_MFMA = 128;
constexpr int MFMA_ITERS = K / K_PER_MFMA;
constexpr int BLOCKS_PER_ROW = K / 32;

enum Loader { LOAD_SPLIT = 0, LOAD_CONTIG = 1 };
enum Grouping { GRP_SPLIT32 = 0, GRP_CONTIG32 = 1, GRP_UNIFORM128 = 2 };

// ── helpers, copied from gang_moe_linear_mxfp4_mi300.cuh ───────────────────

__device__ __forceinline__ float _bf16_to_float(unsigned short v) {
  unsigned u = (unsigned)v << 16;
  float f;
  __builtin_memcpy(&f, &u, 4);
  return f;
}

__device__ __forceinline__ uint8_t _compute_e8m0_fp8(float amax) {
  if (amax == 0.0f) {
    return 0;
  }
  float target = amax * (1.0f / 448.0f);
  union {
    float f;
    uint32_t u;
  } v;
  v.f = target;
  int raw_exp = (int)((v.u >> 23) & 0xFF);
  if (v.u & 0x7FFFFF) {
    raw_exp++;
  }
  return (uint8_t)max(0, min(255, raw_exp));
}

__device__ __forceinline__ float _e8m0_to_float(uint8_t se) {
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

__device__ __forceinline__ float _fp8_to_float(uint8_t b) {
  return __builtin_amdgcn_cvt_f32_fp8((int)b, 0);
}

__device__ __forceinline__ i32x8_t
    _load_split(uint8_t const *data, int kt, int g) {
  i32x4_t lo = *(i32x4_t const *)(data + kt + g * 16);
  i32x4_t hi = *(i32x4_t const *)(data + kt + g * 16 + 64);
  i32x8_t r;
  r[0] = lo[0];
  r[1] = lo[1];
  r[2] = lo[2];
  r[3] = lo[3];
  r[4] = hi[0];
  r[5] = hi[1];
  r[6] = hi[2];
  r[7] = hi[3];
  return r;
}

__device__ __forceinline__ i32x8_t
    _load_contig(uint8_t const *data, int kt, int g) {
  return *(i32x8_t const *)(data + kt + g * 32);
}

__device__ __forceinline__ f32x4_t
    _mfma_f8xf8(i32x8_t a, i32x8_t b, f32x4_t c, int scale_a, int scale_b) {
  return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
      a, b, c, 0, 0, 0, scale_a, 0, scale_b);
}

// Which lane-group's exponent covers element kl of a 128-tile.
__device__ __host__ __forceinline__ int owner_g(int kl, int grouping) {
  if (grouping == GRP_CONTIG32) {
    return kl >> 5;
  }
  if (grouping == GRP_UNIFORM128) {
    return 0;
  }
  return (kl < 64) ? (kl >> 4) : ((kl - 64) >> 4);
}

// ── packing ────────────────────────────────────────────────────────────────

// Two passes so the exponent is settled before any value is encoded.
__global__ void pack_kernel(unsigned short const *src,
                            uint8_t *data,
                            uint8_t *scales,
                            int rows,
                            int grouping) {
  int row = blockIdx.x;
  int blk = threadIdx.x;
  if (row >= rows || blk >= BLOCKS_PER_ROW) {
    return;
  }
  int ki = blk >> 2;
  int g = blk & 3;

  float amax = 0.0f;
  for (int kl = 0; kl < K_PER_MFMA; kl++) {
    if (owner_g(kl, grouping) != g) {
      continue;
    }
    float v = _bf16_to_float(src[(size_t)row * K + ki * K_PER_MFMA + kl]);
    amax = fmaxf(amax, fabsf(v));
  }
  uint8_t se = _compute_e8m0_fp8(amax);
  scales[(size_t)row * BLOCKS_PER_ROW + blk] = se;
  __syncthreads();

  float scale_f = _e8m0_to_float(se);
  for (int kl = 0; kl < K_PER_MFMA; kl += 4) {
    if (owner_g(kl, grouping) != g) {
      continue;
    }
    int k = ki * K_PER_MFMA + kl;
    fp8x4_t pk = {};
    pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(
        pk,
        _bf16_to_float(src[(size_t)row * K + k + 0]),
        _bf16_to_float(src[(size_t)row * K + k + 1]),
        scale_f,
        false);
    pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(
        pk,
        _bf16_to_float(src[(size_t)row * K + k + 2]),
        _bf16_to_float(src[(size_t)row * K + k + 3]),
        scale_f,
        true);
    *(fp8x4_t *)(data + (size_t)row * K + k) = pk;
  }
}

__global__ void pack_token_kernel(unsigned short const *src,
                                  uint8_t *data,
                                  uint8_t *scales) {
  int ki = threadIdx.x;
  if (ki >= MFMA_ITERS) {
    return;
  }
  float amax = 0.0f;
  for (int j = 0; j < K_PER_MFMA; j++) {
    amax = fmaxf(amax, fabsf(_bf16_to_float(src[ki * K_PER_MFMA + j])));
  }
  uint8_t se = _compute_e8m0_fp8(amax);
  scales[ki] = se;
  float scale_f = _e8m0_to_float(se);
  for (int j = 0; j < K_PER_MFMA; j += 4) {
    int k = ki * K_PER_MFMA + j;
    fp8x4_t pk = {};
    pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(
        pk, _bf16_to_float(src[k]), _bf16_to_float(src[k + 1]), scale_f, false);
    pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(
        pk,
        _bf16_to_float(src[k + 2]),
        _bf16_to_float(src[k + 3]),
        scale_f,
        true);
    *(fp8x4_t *)(data + k) = pk;
  }
}

// Scalar sum over the packed bytes in plain K order. Independent of the
// loader: it pairs A[k] with B[k], which is what a correct loader must do.
__global__ void reference_kernel(uint8_t const *w_data,
                                 uint8_t const *w_scales,
                                 uint8_t const *a_data,
                                 uint8_t const *a_scales,
                                 float *out,
                                 int grouping) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) {
    return;
  }
  float sum = 0.0f;
  for (int ki = 0; ki < MFMA_ITERS; ki++) {
    float as = _e8m0_to_float(a_scales[ki]);
    for (int kl = 0; kl < K_PER_MFMA; kl++) {
      int k = ki * K_PER_MFMA + kl;
      int blk = ki * 4 + owner_g(kl, grouping);
      float ws = _e8m0_to_float(w_scales[(size_t)n * BLOCKS_PER_ROW + blk]);
      sum += (_fp8_to_float(w_data[(size_t)n * K + k]) * ws) *
             (_fp8_to_float(a_data[k]) * as);
    }
  }
  out[n] = sum;
}

template <int LOADER, int GROUPING>
__global__ void mfma_kernel(uint8_t const *w_data,
                            uint8_t const *w_scales,
                            uint8_t const *a_data,
                            uint8_t const *a_scales,
                            float *out) {
  int const tid = threadIdx.x;
  int const warp_id = tid >> 6;
  int const lane_id = tid & 63;
  int const col = lane_id & 15;
  int const g = lane_id >> 4;

  int const w_row = warp_id * 16 + col;
  size_t const row_base = (size_t)w_row * K;
  int const row_scale_base = w_row * BLOCKS_PER_ROW;

  f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int ki = 0; ki < MFMA_ITERS; ki++) {
    int kt = ki * K_PER_MFMA;
    i32x8_t a = (LOADER == LOAD_SPLIT) ? _load_split(w_data + row_base, kt, g)
                                       : _load_contig(w_data + row_base, kt, g);
    // Under UNIFORM128 every lane of the tile shares group 0's exponent.
    int sg = (GROUPING == GRP_UNIFORM128) ? 0 : g;
    int sa = (int)w_scales[row_scale_base + ki * 4 + sg];
    i32x8_t b = _load_split(a_data, kt, g);
    int sb = (int)a_scales[ki];
    acc = _mfma_f8xf8(a, b, acc, sa, sb);
  }

  if (col == 0) {
    for (int i = 0; i < 4; i++) {
      out[warp_id * 16 + g * 4 + i] = acc[i];
    }
  }
}

static unsigned short f2bf(float f) {
  union {
    float f;
    unsigned u;
  } v;
  v.f = f;
  unsigned bias = ((v.u >> 16) & 1) + 0x7FFF;
  return (unsigned short)((v.u + bias) >> 16);
}

static float frand() {
  return (float)rand() / (float)(RAND_MAX) * 2.0f - 1.0f;
}

int main() {
  srand(1234);
  std::vector<unsigned short> h_w((size_t)N * K), h_a(K);
  // Spread magnitudes across k so neighbouring 32-blocks land on different
  // E8M0 exponents -- otherwise a wrong grouping would pass by luck.
  for (int n = 0; n < N; n++) {
    for (int k = 0; k < K; k++) {
      float mag = exp2f((float)((k / 16) % 9) - 4.0f);
      h_w[(size_t)n * K + k] = f2bf(frand() * mag);
    }
  }
  for (int k = 0; k < K; k++) {
    h_a[k] = f2bf(frand());
  }

  unsigned short *d_w, *d_a;
  uint8_t *d_wd, *d_ws, *d_ad, *d_as;
  float *d_ref, *d_out;
  HIP_CHECK(hipMalloc(&d_w, h_w.size() * 2));
  HIP_CHECK(hipMalloc(&d_a, h_a.size() * 2));
  HIP_CHECK(hipMalloc(&d_wd, (size_t)N * K));
  HIP_CHECK(hipMalloc(&d_ws, (size_t)N * BLOCKS_PER_ROW));
  HIP_CHECK(hipMalloc(&d_ad, K));
  HIP_CHECK(hipMalloc(&d_as, MFMA_ITERS));
  HIP_CHECK(hipMalloc(&d_ref, N * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, N * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_w, h_w.data(), h_w.size() * 2, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_a, h_a.data(), h_a.size() * 2, hipMemcpyHostToDevice));

  pack_token_kernel<<<1, 64>>>(d_a, d_ad, d_as);
  HIP_CHECK(hipDeviceSynchronize());

  char const *lname[] = {"split", "contig"};
  char const *gname[] = {"split32", "contig32", "uniform128"};
  int best_l = -1, best_g = -1;
  double best = 1e30;

  printf("%-8s %-11s %-12s %s\n", "loader", "scales", "max_rel", "verdict");
  for (int l = 0; l < 2; l++) {
    for (int gr = 0; gr < 3; gr++) {
      pack_kernel<<<N, BLOCKS_PER_ROW>>>(d_w, d_wd, d_ws, N, gr);
      HIP_CHECK(hipDeviceSynchronize());
      reference_kernel<<<1, N>>>(d_wd, d_ws, d_ad, d_as, d_ref, gr);
      HIP_CHECK(hipDeviceSynchronize());

#define LAUNCH(L, G)                                                           \
  mfma_kernel<L, G><<<1, 256>>>(d_wd, d_ws, d_ad, d_as, d_out)
      if (l == 0) {
        if (gr == 0) {
          LAUNCH(LOAD_SPLIT, GRP_SPLIT32);
        } else if (gr == 1) {
          LAUNCH(LOAD_SPLIT, GRP_CONTIG32);
        } else {
          LAUNCH(LOAD_SPLIT, GRP_UNIFORM128);
        }
      } else {
        if (gr == 0) {
          LAUNCH(LOAD_CONTIG, GRP_SPLIT32);
        } else if (gr == 1) {
          LAUNCH(LOAD_CONTIG, GRP_CONTIG32);
        } else {
          LAUNCH(LOAD_CONTIG, GRP_UNIFORM128);
        }
      }
#undef LAUNCH
      HIP_CHECK(hipDeviceSynchronize());

      std::vector<float> ref(N), got(N);
      HIP_CHECK(hipMemcpy(
          ref.data(), d_ref, N * sizeof(float), hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(
          got.data(), d_out, N * sizeof(float), hipMemcpyDeviceToHost));

      double mag = 0.0;
      for (int i = 0; i < N; i++) {
        mag = fmax(mag, fabs(ref[i]));
      }
      double max_rel = 0.0;
      for (int i = 0; i < N; i++) {
        max_rel = fmax(max_rel, fabs(got[i] - ref[i]) / fmax(mag, 1e-6));
      }
      bool pass = max_rel < 1e-3;
      printf("%-8s %-11s %-12.3e %s\n",
             lname[l],
             gname[gr],
             max_rel,
             pass ? "MATCH" : "");
      if (max_rel < best) {
        best = max_rel;
        best_l = l;
        best_g = gr;
      }
    }
  }

  bool ok = best < 1e-3;
  printf("\n%s best = loader:%s scales:%s (max_rel=%.3e)\n",
         ok ? "[PASS]" : "[FAIL]",
         lname[best_l],
         gname[best_g],
         best);
  return ok ? 0 : 1;
}
