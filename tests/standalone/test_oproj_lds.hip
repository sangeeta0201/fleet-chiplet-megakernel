/* Copyright 2025 CMU
 *
 * MPK_OPROJ_PF_LDS check for the row-parallel MXFP4 o_proj GEMV
 * (gang_gemv_mxfp4_kernel at GLM-5's NP=8 tile: 32 rows x 4096 K).
 *
 * Every workgroup runs one tile twice from the same staged activation: once
 * with its weights read from global (the default, warmed into L2 first, as the
 * Phase 8 prefetch does), once from the copy w_lds_dma leaves in LDS. The
 * LDS_OUT rows must be bit-identical, and the two GEMV durations are stamped
 * per workgroup with s_memrealtime (24 workgroups an XCD, all concurrent).
 *
 * Build/run: see the command in test_mla_decode.hip's header; same flags plus
 * -DMPK_OPROJ_PF_LDS=1.
 */
#include <algorithm>
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

#if !MPK_OPROJ_PF_LDS
#error "build with -DMPK_OPROJ_PF_LDS=1"
#endif

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
static constexpr int LDS_BYTES = MPK_OPROJ_W_LDS_OFF + WG_BYTES;

template <bool W_LDS>
__device__ __forceinline__ void run_tile(unsigned short const *act,
                                         unsigned char const *w, float *out,
                                         int t, bool stage) {
  gang_gemv_mxfp4_kernel<1, K, ROWS, /*HAS_RESIDUAL=*/false,
                         /*WRITE_THROUGH=*/false, /*F32_OUT=*/true,
                         /*LL_OUT=*/false, /*LDS_OUT=*/true, W_LDS>(
      act, w, nullptr, out, 1, ROWS, OUT_PER_XCD, 1, TILES, 0, t, nullptr,
      stage);
}

// Block b: XCD b % 8, tile b / 8 of that XCD's slab. Writes the two LDS_OUT
// row sets to rows_out[2][NWG][ROWS] and the GEMV times to stamps[NWG][2].
__global__ void __launch_bounds__(256)
    bench(unsigned short const *act, unsigned char const *w, float *out,
          float *rows_out, unsigned long long *stamps, int *arrive) {
  extern __shared__ char _fused_smem[];
  int const x = blockIdx.x % 8, t = blockIdx.x / 8, tid = threadIdx.x;
  unsigned char const *wx = w + (size_t)x * TILES * WG_BYTES;
  float *ox = out + (size_t)x * OUT_PER_XCD;
  i32x4_t const rsrc = make_w_buffer_rsrc(wx, (uint32_t)TILES * WG_BYTES);
  // Warm L2 with the tile as the default's Phase 8 prefetch does (sc0 DMA into
  // a never-read 4 KB window), then DMA it into LDS.
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
  gang_gemv_mxfp4_detail::w_lds_dma<K, ROWS>(
      rsrc, (unsigned)t * WG_BYTES,
      __builtin_amdgcn_readfirstlane(
          (unsigned)reinterpret_cast<uintptr_t>(_fused_smem)) +
          MPK_OPROJ_W_LDS_OFF +
          __builtin_amdgcn_readfirstlane((unsigned)tid >> 6) * 1024u);
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  __syncthreads();
  asm volatile("buffer_inv" ::: "memory");
  // All blocks start together.
  if (tid == 0) {
    atomicAdd(arrive, 1);
    while (__hip_atomic_load(arrive, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < NWG) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
  __syncthreads();
  // Pass 0 stages the activation; passes 1 (global weights, L2-warm) and 2
  // (LDS weights) reuse it, so they time the GEMV body alone.
  for (int pass = 0; pass < 3; pass++) {
    unsigned long long const t0 = __builtin_amdgcn_s_memrealtime();
    if (pass < 2) {
      run_tile<false>(act, wx, ox, t, pass == 0);
    } else {
      run_tile<true>(act, wx, ox, t, false);
    }
    __syncthreads();
    unsigned long long const t1 = __builtin_amdgcn_s_memrealtime();
    if (tid < ROWS) {
      rows_out[((size_t)(pass == 2) * NWG + blockIdx.x) * ROWS + tid] =
          reinterpret_cast<float const *>(
              _fused_smem + gang_gemv_mxfp4_detail::lds_out_off<1, K>())[tid];
    }
    if (tid == 0) {
      stamps[blockIdx.x * 3 + pass] = t1 - t0;
    }
    __syncthreads();
  }
}

int main() {
  srand(20261014);
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
  std::vector<unsigned short> h_a(K);
  for (int i = 0; i < K; i++) {
    float const f = ((rand() % 2001) - 1000) / 500.0f;
    unsigned u;
    memcpy(&u, &f, 4);
    h_a[i] = (unsigned short)(u >> 16);
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
  HIP_CHECK(hipMalloc(&d_st, NWG * 3 * 8));
  HIP_CHECK(hipMalloc(&d_arr, 4));
  HIP_CHECK(hipMemcpy(d_w, h_w.data(), wbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_a, h_a.data(), K * 2, hipMemcpyHostToDevice));
  HIP_CHECK(hipFuncSetAttribute((const void *)bench,
                                hipFuncAttributeMaxDynamicSharedMemorySize,
                                LDS_BYTES));
  int rc = 0;
  std::vector<double> ts, tg, tl;
  for (int rep = 0; rep < 5; rep++) {
    HIP_CHECK(hipMemset(d_arr, 0, 4));
    HIP_CHECK(hipMemset(d_rows, 0xFF, 2 * NWG * ROWS * 4));
    hipLaunchKernelGGL(bench, dim3(NWG), dim3(256), LDS_BYTES, 0, d_a, d_w,
                       d_out, d_rows, d_st, d_arr);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> h_rows(2 * NWG * ROWS);
    std::vector<unsigned long long> h_st(NWG * 3);
    HIP_CHECK(hipMemcpy(h_rows.data(), d_rows, h_rows.size() * 4,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_st.data(), d_st, h_st.size() * 8,
                        hipMemcpyDeviceToHost));
    int diff = 0, nz = 0;
    for (int i = 0; i < NWG * ROWS; i++) {
      diff += memcmp(&h_rows[i], &h_rows[NWG * ROWS + i], 4) != 0;
      nz += h_rows[i] != 0.0f;
    }
    // Host reference for block 0's rows (tile 0 of XCD 0), loose tolerance.
    double maxerr = 0.0;
    for (int r = 0; r < ROWS; r++) {
      double acc = 0.0;
      unsigned char const *wd = h_w.data() + (size_t)r * (K / 2);
      unsigned char const *ws = h_w.data() + ROWS * (K / 2) + (size_t)r * (K / 32);
      static const float lut[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                    -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
      for (int k = 0; k < K; k++) {
        unsigned const nib = (wd[k / 2] >> ((k & 1) * 4)) & 0xF;
        double const sc = ldexp(1.0, (int)ws[k / 32] - 127);
        unsigned u = (unsigned)h_a[k] << 16;
        float af;
        memcpy(&af, &u, 4);
        acc += lut[nib] * sc * af;
      }
      maxerr = std::max(maxerr, fabs(acc - h_rows[r]) / std::max(1.0, fabs(acc)));
    }
    for (int b = 0; b < NWG; b++) {
      ts.push_back(h_st[b * 3] * 10.0 / 1000.0);
      tg.push_back(h_st[b * 3 + 1] * 10.0 / 1000.0);
      tl.push_back(h_st[b * 3 + 2] * 10.0 / 1000.0);
    }
    bool const ok = diff == 0 && nz > NWG * ROWS / 2 && maxerr < 2e-2;
    printf("%s rep %d: %d of %d rows differ (global vs LDS), %d nonzero, "
           "block 0 vs host max rel err %.2e\n",
           ok ? "PASS" : "FAIL", rep, diff, NWG * ROWS, nz, maxerr);
    rc |= ok ? 0 : 1;
  }
  for (auto *v : {&ts, &tg, &tl}) {
    std::sort(v->begin(), v->end());
  }
  printf("GEMV us/tile, 192 concurrent, 5 reps: staged+global med %.2f max %.2f | "
         "global(L2) med %.2f p90 %.2f max %.2f | LDS med %.2f p90 %.2f max %.2f\n",
         ts[ts.size() / 2], ts.back(), tg[tg.size() / 2], tg[tg.size() * 9 / 10],
         tg.back(), tl[tl.size() / 2], tl[tl.size() * 9 / 10], tl.back());
  return rc;
}
