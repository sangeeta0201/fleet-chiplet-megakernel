/* Copyright 2025 CMU
 *
 * MPK_OPROJ_MFMA check for the row-parallel MXFP4 o_proj GEMV
 * (gang_gemv_mxfp4_kernel at GLM-5's NP=8 tile: 32 rows x 4096 K).
 *
 * 192 workgroups (24 an XCD, all concurrent) each run their tile twice: pass 0
 * stages the activation, pass 1 reuses it with the weights L2-warm, as the
 * Phase 8 prefetch leaves them, so it times the GEMV body alone. Every row is
 * checked against two host references: the exact product (weights dequantized,
 * bf16 activation), and -- built with -DMPK_OPROJ_MFMA=1 -- the product with
 * the activation quantized the way the kernel does (E4M3, one E8M0 per 128,
 * round to nearest even), which the MFMA tile must match to f32 accumulation
 * error. Built with -DMPK_OPROJ_MFMA=0 it times the VALU tile in the same
 * harness.
 *
 * Build/run: see the command in test_mla_decode.hip's header; same flags plus
 * -DMPK_OPROJ_MFMA=1 (or 0).
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <hip/hip_runtime.h>

#define MPK_WS_WAVE_SYNC(wave) ((void)0)
#define MPK_WS_MARK(code, aux) ((void)0)
#define MPK_WS_WAIT_BEGIN(b, e) ((void)0)
#define MPK_WS_WAIT_TICK(o, s) ((void)0)
#define MPK_WS_WAIT_AUX(...) ((void)0)
#ifndef MPK_TARGET_CC
#define MPK_TARGET_CC 95
#endif
namespace kernel {
__device__ __forceinline__ int _gang_moe_get_xcd_id() {
  int xcd_id;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcd_id));
  return xcd_id;
}
} // namespace kernel
#include "mirage/persistent_kernel/mpk_atoms.cuh"
#include "mirage/persistent_kernel/tasks/mi300/gang_gemv_mxfp4_mi300.cuh"
#include "mirage/persistent_kernel/tasks/mi300/gang_moe_linear_mxfp4_mi300.cuh"


#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t err = call;                                                     \
    if (err != hipSuccess) {                                                   \
      fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,            \
              hipGetErrorString(err));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

using namespace kernel;
static constexpr int K = 4096, ROWS = 32, TILES = 24, NWG = 8 * TILES;
static constexpr int WG_BYTES = ROWS * (K / 2) + ROWS * (K / 32);
static constexpr int OUT_PER_XCD = TILES * ROWS;
#if MPK_OPROJ_PF_LDS
static constexpr int LDS_BYTES = MPK_OPROJ_W_LDS_OFF + WG_BYTES;
static constexpr bool WL = true;
#else
static constexpr int LDS_BYTES = 68 * 1024;
static constexpr bool WL = false;
#endif

__device__ __forceinline__ void run_tile(unsigned short const *act,
                                         unsigned char const *w, float *out,
                                         int t, bool stage) {
  gang_gemv_mxfp4_kernel<1, K, ROWS, /*HAS_RESIDUAL=*/false,
                         /*WRITE_THROUGH=*/false, /*F32_OUT=*/true,
                         /*LL_OUT=*/false, /*LDS_OUT=*/true
#if MPK_OPROJ_PF_LDS
                         , /*W_LDS=*/WL
#endif
                         >(
      act, w, nullptr, out, 1, ROWS, OUT_PER_XCD, 1, TILES, 0, t, nullptr,
      stage);
}

// Block b: XCD b % 8, tile b / 8 of that XCD's slab. Writes each pass's
// LDS_OUT rows to rows_out[2][NWG][ROWS] and its time to stamps[NWG][2].
__global__ void __launch_bounds__(256)
    bench(unsigned short const *act, unsigned char const *w, float *out,
          float *rows_out, unsigned long long *stamps, int *arrive) {
  extern __shared__ char _fused_smem[];
  int const x = blockIdx.x % 8, t = blockIdx.x / 8, tid = threadIdx.x;
  unsigned char const *wx = w + (size_t)x * TILES * WG_BYTES;
  float *ox = out + (size_t)x * OUT_PER_XCD;
  i32x4_t const rsrc = make_w_buffer_rsrc(wx, (uint32_t)TILES * WG_BYTES);
  // Warm L2 with the tile as the default's Phase 8 prefetch does (DMA into a
  // never-read 4 KB window).
  {
    auto *dst = (__attribute__((address_space(3))) uint32_t *)(
        _fused_smem + 64 * 1024 + (tid >> 6) * 1024);
    int voff = t * WG_BYTES + tid * 16;
#pragma unroll
    for (int j = 0; j < WG_BYTES / 4096; j++) {
      __llvm_amdgcn_raw_buffer_load_lds(rsrc, dst, 16, voff, 0, 0, 1);
      voff += 4096;
      asm volatile("" : "+v"(voff) : : "memory");
    }
  }
#if MPK_OPROJ_PF_LDS
  // The tile into LDS as the Phase 8 prefetch leaves it under PF_LDS.
  {
    unsigned const m0 =
        __builtin_amdgcn_readfirstlane(
            (unsigned)reinterpret_cast<uintptr_t>(_fused_smem)) +
        MPK_OPROJ_W_LDS_OFF +
        __builtin_amdgcn_readfirstlane((unsigned)tid >> 6) * 1024u;
#if MPK_OPROJ_MFMA
    gang_gemv_mxfp4_detail::w_lds_dma_mfma<K, ROWS>(rsrc, (unsigned)t * WG_BYTES, m0);
#else
    gang_gemv_mxfp4_detail::w_lds_dma<K, ROWS>(rsrc, (unsigned)t * WG_BYTES, m0);
#endif
  }
#endif
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  __syncthreads();
  asm volatile("buffer_inv" ::: "memory");
  if (tid == 0) {
    atomicAdd(arrive, 1);
    while (__hip_atomic_load(arrive, __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_AGENT) < NWG) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
  __syncthreads();
  for (int pass = 0; pass < 2; pass++) {
    unsigned long long const t0 = __builtin_amdgcn_s_memrealtime();
    run_tile(act, wx, ox, t, pass == 0);
    __syncthreads();
    unsigned long long const t1 = __builtin_amdgcn_s_memrealtime();
    if (tid < ROWS) {
      rows_out[((size_t)pass * NWG + blockIdx.x) * ROWS + tid] =
          reinterpret_cast<float const *>(
              _fused_smem + gang_gemv_mxfp4_detail::lds_out_off<1, K>())[tid];
    }
    if (tid == 0) {
      stamps[blockIdx.x * 2 + pass] = t1 - t0;
    }
    __syncthreads();
  }
}

static float bf16f(unsigned short h) {
  unsigned u = (unsigned)h << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
// E4M3 (max 448), round to nearest even, of an already-scaled value.
static float e4m3_rne(float x) {
  if (x == 0.0f) {
    return 0.0f;
  }
  float const ax = fabsf(x);
  int e;
  frexpf(ax, &e);
  int ex = e - 1;
  if (ex < -6) {
    ex = -6;
  }
  float const step = ldexpf(1.0f, ex - 3);
  float q = rintf(ax / step) * step;
  if (q > 448.0f) {
    q = 448.0f;
  }
  return copysignf(q, x);
}
static unsigned e8m0_fp8(float amax) {
  if (amax == 0.0f) {
    return 0u;
  }
  float const tgt = amax * (1.0f / 448.0f);
  unsigned u;
  memcpy(&u, &tgt, 4);
  int ex = (int)((u >> 23) & 0xFF);
  if (u & 0x7FFFFF) {
    ex++;
  }
  return (unsigned)std::max(0, std::min(255, ex));
}

int main() {
  srand(20261018);
  size_t const wbytes = (size_t)8 * TILES * WG_BYTES;
  std::vector<unsigned char> h_w(wbytes);
  for (int x = 0; x < 8 * TILES; x++) {
    unsigned char *p = h_w.data() + (size_t)x * WG_BYTES;
    for (int i = 0; i < ROWS * (K / 2); i++) {
      p[i] = (unsigned char)(rand() & 0xFF);
    }
    for (int i = 0; i < ROWS * (K / 32); i++) {
      p[ROWS * (K / 2) + i] = (unsigned char)(120 + rand() % 12);
    }
  }
  // Magnitudes spread over ~2^10 so the per-128 scales differ block to block.
  std::vector<unsigned short> h_a(K);
  for (int i = 0; i < K; i++) {
    float const f = ((rand() % 2001) - 1000) / 500.0f *
                    ldexpf(1.0f, (i / 128) % 11 - 5);
    unsigned u;
    memcpy(&u, &f, 4);
    h_a[i] = (unsigned short)(u >> 16);
  }
  std::vector<float> a_ex(K), a_q(K);
  for (int i = 0; i < K; i++) {
    a_ex[i] = bf16f(h_a[i]);
  }
  for (int b = 0; b < K / 128; b++) {
    float amax = 0.0f;
    for (int i = 0; i < 128; i++) {
      amax = std::max(amax, fabsf(a_ex[b * 128 + i]));
    }
    unsigned const se = e8m0_fp8(amax);
    float const sc = se == 0u ? 1.0f : ldexpf(1.0f, (int)se - 127);
    float const dq = ldexpf(1.0f, (int)se - 127);
    for (int i = 0; i < 128; i++) {
      a_q[b * 128 + i] = e4m3_rne(a_ex[b * 128 + i] / sc) * dq;
    }
  }
  static const float lut[16] = {0.f,  .5f,  1.f,  1.5f,  2.f,  3.f,  4.f,  6.f,
                                -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
  std::vector<double> ref_ex((size_t)NWG * ROWS), ref_q((size_t)NWG * ROWS);
  for (int blk = 0; blk < NWG; blk++) {
    int const x = blk % 8, t = blk / 8;
    unsigned char const *tw = h_w.data() + ((size_t)x * TILES + t) * WG_BYTES;
    for (int r = 0; r < ROWS; r++) {
      unsigned char const *wd = tw + (size_t)r * (K / 2);
      unsigned char const *ws = tw + ROWS * (K / 2) + (size_t)r * (K / 32);
      double se = 0.0, sq = 0.0;
      for (int k = 0; k < K; k++) {
        unsigned const nib = (wd[k / 2] >> ((k & 1) * 4)) & 0xF;
        double const wv = lut[nib] * ldexp(1.0, (int)ws[k / 32] - 127);
        se += wv * a_ex[k];
        sq += wv * a_q[k];
      }
      ref_ex[(size_t)blk * ROWS + r] = se;
      ref_q[(size_t)blk * ROWS + r] = sq;
    }
  }
  unsigned char *d_w;
  unsigned short *d_a;
  float *d_out, *d_rows;
  unsigned long long *d_st;
  int *d_arr;
  HIP_CHECK(hipMalloc(&d_w, wbytes));
  HIP_CHECK(hipMalloc(&d_a, K * 2));
  HIP_CHECK(hipMalloc(&d_out, 8 * OUT_PER_XCD * 4));
  HIP_CHECK(hipMalloc(&d_rows, 2 * NWG * ROWS * 4));
  HIP_CHECK(hipMalloc(&d_st, NWG * 2 * 8));
  HIP_CHECK(hipMalloc(&d_arr, 4));
  HIP_CHECK(hipMemcpy(d_w, h_w.data(), wbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_a, h_a.data(), K * 2, hipMemcpyHostToDevice));
  HIP_CHECK(hipFuncSetAttribute((const void *)bench,
                                hipFuncAttributeMaxDynamicSharedMemorySize,
                                LDS_BYTES));
  int rc = 0;
  std::vector<double> ts, tg;
  for (int rep = 0; rep < 5; rep++) {
    HIP_CHECK(hipMemset(d_arr, 0, 4));
    HIP_CHECK(hipMemset(d_rows, 0xFF, 2 * NWG * ROWS * 4));
    hipLaunchKernelGGL(bench, dim3(NWG), dim3(256), LDS_BYTES, 0, d_a, d_w,
                       d_out, d_rows, d_st, d_arr);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> h_rows(2 * NWG * ROWS);
    std::vector<unsigned long long> h_st(NWG * 2);
    HIP_CHECK(hipMemcpy(h_rows.data(), d_rows, h_rows.size() * 4,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_st.data(), d_st, h_st.size() * 8,
                        hipMemcpyDeviceToHost));
    // Pass 1 reuses pass 0's staging: the two must agree bit for bit.
    int diff = 0;
    double err_ex = 0.0, err_q = 0.0, mag = 0.0;
    for (int i = 0; i < NWG * ROWS; i++) {
      diff += memcmp(&h_rows[i], &h_rows[NWG * ROWS + i], 4) != 0;
      mag = std::max(mag, fabs(ref_ex[i]));
    }
    for (int i = 0; i < NWG * ROWS; i++) {
      err_ex = std::max(err_ex, fabs(h_rows[i] - ref_ex[i]) / mag);
      err_q = std::max(err_q, fabs(h_rows[i] - ref_q[i]) / mag);
    }
    for (int b = 0; b < NWG; b++) {
      ts.push_back(h_st[b * 2] * 10.0 / 1000.0);
      tg.push_back(h_st[b * 2 + 1] * 10.0 / 1000.0);
    }
    // The MFMA tile rounds the row to E4M3: it must match the quantized
    // reference to f32 accumulation error; against the exact product it
    // carries that rounding.
    bool const ok = diff == 0 &&
                    (MPK_OPROJ_MFMA ? (err_q < 1e-5 && err_ex < 5e-2)
                                    : err_ex < 2e-2);
    printf("%s rep %d (MPK_OPROJ_MFMA=%d): %d of %d rows differ pass 0 vs 1, "
           "max err / max|ref| vs exact %.2e, vs E4M3-quantized row %.2e\n",
           ok ? "PASS" : "FAIL", rep, MPK_OPROJ_MFMA, diff, NWG * ROWS,
           err_ex, err_q);
    rc |= ok ? 0 : 1;
  }
  std::sort(ts.begin(), ts.end());
  std::sort(tg.begin(), tg.end());
  printf("GEMV us/tile, 192 concurrent, 5 reps (MPK_OPROJ_MFMA=%d, weights %s): "
         "staged+GEMV med %.2f max %.2f | GEMV med %.2f p90 %.2f max %.2f\n",
         MPK_OPROJ_MFMA, WL ? "in LDS" : "L2-warm", ts[ts.size() / 2], ts.back(), tg[tg.size() / 2],
         tg[tg.size() * 9 / 10], tg.back());
  return rc;
}
