// Standalone check + timing of the GLM-5 LM head instance of
// gang_rmsnorm_linear_mxfp8_bias_kernel<1, 64, 6144, 6144, false, true, 8>
// with and without MPK_LMH_BAL (mpk_atoms.cuh). Build twice: -DMPK_LMH_BAL=0,
// and -DMPK_LMH_BAL=1 -DMPK_WORKERS_PER_XCD=29 (add -DMPK_LMH_ARGMAX=1
// -DMPK_LMH_ARGMAX_CHUNK=128 -DMPK_LMH_ARGMAX_SLOTS=232 for the key check).
// Each run writes its logits to argv[1] and, given argv[2], compares against
// the other build's. The launch mimics the megakernel's dispatch: block b is
// XCD slice b % 8, rank b / 8, and walks tiles rank, rank + 29, ... of its
// slice; four copies of the weight rotate so the MALL cannot serve them.
//   ./test_lmh_bal0 /tmp/lmh0.bin && ./test_lmh_bal1 /tmp/lmh1.bin /tmp/lmh0.bin
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#define MPK_WS_WAVE_SYNC(wave) ((void)0)
#define MPK_WS_MARK(code, aux) ((void)0)
#define MPK_WS_WAIT_BEGIN(b, e) ((void)0)
#define MPK_WS_WAIT_TICK(o, s) ((void)0)
#define MPK_WS_WAIT_AUX(...) ((void)0)
namespace kernel {
__device__ __forceinline__ int _gang_moe_get_xcd_id() {
  int xcd_id;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcd_id));
  return xcd_id;
}
} // namespace kernel
#include "mirage/persistent_kernel/mpk_atoms.cuh"
#include "mirage/persistent_kernel/tasks/mi300/gang_rmsnorm_linear_mxfp8_bias_mi300.cuh"

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
static constexpr int K = 6144, OPW = 64, NWG = 38, SLICES = 8, WPX = 29;
static constexpr int NB32 = K / 32;
static constexpr size_t WGB = (size_t)OPW * K + OPW * NB32;
static constexpr int SHARD = SLICES * NWG * OPW; // 19456
static constexpr int NCOPY = 4;

__global__ void __launch_bounds__(256)
    lmh(unsigned short const *gather, unsigned short const *nw,
        uint8_t const *w, unsigned short const *bias, unsigned short *out,
        float const *ws, unsigned short *xout, int ntiles,
        unsigned long long *stamps) {
  int const slice = blockIdx.x % SLICES;
  int const rank = blockIdx.x / SLICES;
  if (threadIdx.x == 0) stamps[blockIdx.x * 2] = __builtin_amdgcn_s_memrealtime();
  for (int t = rank; t < ntiles; t += WPX) {
    gang_rmsnorm_linear_mxfp8_bias_kernel<1, OPW, K, K, false, true, 8>(
        gather, nw, nullptr, w + (size_t)slice * NWG * WGB,
        bias + slice * NWG * OPW, out + slice * NWG * OPW, 1, NWG, SHARD, t,
        ws, xout);
  }
  if (threadIdx.x == 0) stamps[blockIdx.x * 2 + 1] = __builtin_amdgcn_s_memrealtime();
}

// The LM head's resolve alone (EP slot sum + RMSNorm reciprocal + LDS stage).
__global__ void __launch_bounds__(256)
    resolve_only(unsigned short const *gather, unsigned short const *nw,
                 float const *ws, unsigned short *xout, float *sink,
                 unsigned long long *stamps) {
  extern __shared__ char smem[];
  unsigned short *s_x = (unsigned short *)smem;
  unsigned short *s_nw = s_x + K;
  if (threadIdx.x == 0) stamps[blockIdx.x * 2] = __builtin_amdgcn_s_memrealtime();
  float r = _rnlm8_resadd_norm_rcp<K, 8, K, true, false>(
      ws, gather, xout, s_x, blockIdx.x == 0, 1e-5f, nw, s_nw);
  if (threadIdx.x == 0) {
    stamps[blockIdx.x * 2 + 1] = __builtin_amdgcn_s_memrealtime();
    if (r == 12345.0f) sink[0] = r + s_x[7];
  }
}

#if MPK_LMH_ARGMAX
// Keys over a synthetic logit row, max-reduced: the winner MPK_LMH_ARGMAX's
// xrank reduce would pick.
__global__ void keys_max(unsigned short const *row, int n,
                         unsigned long long *out) {
  unsigned long long best = 0ull;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    unsigned long long k = lmh_argmax_key(row[i], (unsigned)i);
    best = k > best ? k : best;
  }
  for (int off = 32; off > 0; off >>= 1) {
    unsigned lo = __shfl_xor((unsigned)best, off, 64);
    unsigned hi = __shfl_xor((unsigned)(best >> 32), off, 64);
    unsigned long long o = ((unsigned long long)hi << 32) | lo;
    best = o > best ? o : best;
  }
  __shared__ unsigned long long s[4];
  if ((threadIdx.x & 63) == 0) s[threadIdx.x >> 6] = best;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < 4; w++) best = s[w] > best ? s[w] : best;
    *out = best;
  }
}
#endif

static unsigned short h_f2b(float f) {
  unsigned u;
  memcpy(&u, &f, 4);
  unsigned b = ((u >> 16) & 1u) + 0x7FFFu;
  return (unsigned short)((u + b) >> 16);
}
static float h_b2f(unsigned short b) {
  unsigned u = (unsigned)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

int main(int argc, char **argv) {
  char const *out_path = argc > 1 ? argv[1] : "/tmp/c_lmh_out.bin";
  char const *ref_path = argc > 2 ? argv[2] : nullptr;
  int const ntiles = MPK_LMH_BAL ? WPX : NWG;
  srand(4321);
  std::vector<unsigned short> gather(8 * K), nw(K), bias(SHARD, 0);
  for (int i = 0; i < 8 * K; i++) {
    gather[i] = h_f2b((rand() / (float)RAND_MAX - 0.5f) * (i < K ? 2.0f : 0.3f));
  }
  for (int i = 0; i < K; i++) {
    nw[i] = h_f2b(0.5f + rand() / (float)RAND_MAX);
  }
  std::vector<uint8_t> w(WGB * NWG * SLICES);
  for (size_t wg = 0; wg < (size_t)NWG * SLICES; wg++) {
    uint8_t *p = w.data() + wg * WGB;
    for (size_t i = 0; i < (size_t)OPW * K; i++) {
      // E4M3 with |x| <= 2^3 and no NaN encodings.
      uint8_t b = (uint8_t)(rand() & 0xFF);
      if ((b & 0x7F) > 0x5F) b = (b & 0x80) | (b & 0x3F);
      p[i] = b;
    }
    for (int i = 0; i < OPW * NB32; i++) {
      p[(size_t)OPW * K + i] = (uint8_t)(118 + rand() % 4);
    }
  }
#if MPK_LMH_BAL && !MPK_LMH_BAL_ROWMAJ
  {
    // [wg][64 rows][K] -> [wg][4][96 half-k-tiles][16 rows][64 B]; scales
    // [wg][64][192] -> [wg][4][48 k-tiles][16 rows][4 B].
    std::vector<uint8_t> km(w.size());
    for (size_t wg = 0; wg < (size_t)NWG * SLICES; wg++) {
      uint8_t const *src = w.data() + wg * WGB;
      uint8_t *dst = km.data() + wg * WGB;
      for (int r = 0; r < OPW; r++) {
        int const sub = r / 16, rr = r % 16;
        for (int h = 0; h < K / 64; h++) {
          memcpy(dst + (size_t)sub * 16 * K + (size_t)h * 1024 + rr * 64,
                 src + (size_t)r * K + h * 64, 64);
        }
        for (int kt = 0; kt < K / 128; kt++) {
          memcpy(dst + (size_t)OPW * K + sub * 16 * NB32 + kt * 64 + rr * 4,
                 src + (size_t)OPW * K + r * NB32 + kt * 4, 4);
        }
      }
    }
    w.swap(km);
  }
#endif
  unsigned short *d_gather, *d_nw, *d_bias, *d_out, *d_xout;
  float *d_ws;
  uint8_t *d_w;
  HIP_CHECK(hipMalloc(&d_gather, gather.size() * 2));
  HIP_CHECK(hipMalloc(&d_nw, K * 2));
  HIP_CHECK(hipMalloc(&d_bias, SHARD * 2));
  HIP_CHECK(hipMalloc(&d_out, SHARD * 2));
  HIP_CHECK(hipMalloc(&d_xout, K * 2));
  HIP_CHECK(hipMalloc(&d_ws, K * 4));
  HIP_CHECK(hipMalloc(&d_w, w.size() * NCOPY));
  HIP_CHECK(hipMemcpy(d_gather, gather.data(), gather.size() * 2, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_nw, nw.data(), K * 2, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_bias, bias.data(), SHARD * 2, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_ws, 0, K * 4));
  for (int c = 0; c < NCOPY; c++) {
    HIP_CHECK(hipMemcpy(d_w + c * w.size(), w.data(), w.size(), hipMemcpyHostToDevice));
  }
  int const lds = mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE;
  HIP_CHECK(hipFuncSetAttribute((void const *)lmh,
                                hipFuncAttributeMaxDynamicSharedMemorySize, lds));
  hipEvent_t e0, e1;
  HIP_CHECK(hipEventCreate(&e0));
  HIP_CHECK(hipEventCreate(&e1));
  std::vector<float> ms;
  unsigned long long *d_st;
  HIP_CHECK(hipMalloc(&d_st, WPX * SLICES * 2 * 8));
  std::vector<double> dur_sum(WPX, 0.0), start_sum(WPX, 0.0);
  int nrec = 0;
  for (int it = 0; it < 40; it++) {
    HIP_CHECK(hipMemset(d_out, 0, SHARD * 2));
    HIP_CHECK(hipEventRecord(e0));
    hipLaunchKernelGGL(lmh, dim3(WPX * SLICES), dim3(256), lds, 0, d_gather,
                       d_nw, d_w + (it % NCOPY) * w.size(), d_bias, d_out,
                       d_ws, d_xout, ntiles, d_st);
    HIP_CHECK(hipEventRecord(e1));
    HIP_CHECK(hipEventSynchronize(e1));
    float t;
    HIP_CHECK(hipEventElapsedTime(&t, e0, e1));
    if (it >= 8) {
      ms.push_back(t * 1000.0f);
      std::vector<unsigned long long> st(WPX * SLICES * 2);
      HIP_CHECK(hipMemcpy(st.data(), d_st, st.size() * 8, hipMemcpyDeviceToHost));
      unsigned long long t0 = ~0ull;
      for (int b = 0; b < WPX * SLICES; b++) t0 = std::min(t0, st[b * 2]);
      for (int b = 0; b < WPX * SLICES; b++) {
        dur_sum[b / SLICES] += (st[b * 2 + 1] - st[b * 2]) / 100.0;  // 100 MHz
        start_sum[b / SLICES] += (st[b * 2] - t0) / 100.0;
      }
      nrec++;
    }
  }
  printf("per-rank mean (over slices and iters): start / duration us\n");
  for (int r = 0; r < WPX; r++) {
    printf("  r%02d %5.2f %6.2f%s", r, start_sum[r] / (nrec * SLICES),
           dur_sum[r] / (nrec * SLICES), (r % 4 == 3) ? "\n" : "");
  }
  printf("\n");
  {
    std::vector<double> rs;
    for (int it = 0; it < 20; it++) {
      hipLaunchKernelGGL(resolve_only, dim3(WPX * SLICES), dim3(256), lds, 0,
                         d_gather, d_nw, d_ws, d_xout, (float *)d_bias, d_st);
      HIP_CHECK(hipDeviceSynchronize());
      std::vector<unsigned long long> st(WPX * SLICES * 2);
      HIP_CHECK(hipMemcpy(st.data(), d_st, st.size() * 8, hipMemcpyDeviceToHost));
      double sum = 0;
      for (int b = 0; b < WPX * SLICES; b++) sum += (st[b * 2 + 1] - st[b * 2]) / 100.0;
      if (it >= 4) rs.push_back(sum / (WPX * SLICES));
    }
    std::sort(rs.begin(), rs.end());
    printf("resolve_only: mean per-block %.2f us (median over iters)\n", rs[rs.size() / 2]);
  }
#if MPK_LMH_ARGMAX
  {
    // block_reduce_max_idx on the host: per wave a shfl_down tournament that
    // keeps the lower lane on a tie, then the same over the wave results.
    auto tourney = [](std::vector<float> &val, std::vector<long long> &idx) {
      int const n = (int)val.size();  // multiple of 64, <= 256
      std::vector<float> wv(4, -INFINITY);
      std::vector<long long> wi(4, -1);
      for (int w = 0; w < n / 64; w++) {
        std::vector<float> v(val.begin() + w * 64, val.begin() + w * 64 + 64);
        std::vector<long long> x(idx.begin() + w * 64, idx.begin() + w * 64 + 64);
        for (int off = 32; off > 0; off >>= 1) {
          std::vector<float> nv = v;
          std::vector<long long> nx = x;
          for (int l = 0; l + off < 64; l++) {
            if (v[l + off] > v[l]) { nv[l] = v[l + off]; nx[l] = x[l + off]; }
          }
          v = nv; x = nx;
        }
        wv[w] = v[0]; wi[w] = x[0];
      }
      std::vector<float> v(64, -INFINITY);
      std::vector<long long> x(64, -1);
      for (int w = 0; w < 4; w++) { v[w] = wv[w]; x[w] = wi[w]; }
      for (int off = 32; off > 0; off >>= 1) {
        std::vector<float> nv = v;
        std::vector<long long> nx = x;
        for (int l = 0; l + off < 64; l++) {
          if (v[l + off] > v[l]) { nv[l] = v[l + off]; nx[l] = x[l + off]; }
        }
        v = nv; x = nx;
      }
      val.assign(1, v[0]); idx.assign(1, x[0]);
    };
    int const C = MPK_LMH_ARGMAX_CHUNK, NC = SHARD / C;
    unsigned short *d_row;
    unsigned long long *d_k;
    HIP_CHECK(hipMalloc(&d_row, SHARD * 2));
    HIP_CHECK(hipMalloc(&d_k, 8));
    int bad = 0;
    for (int trial = 0; trial < 200; trial++) {
      std::vector<unsigned short> row(SHARD);
      int const nlev = 1 + trial % 7;
      for (int i = 0; i < SHARD; i++) {
        int r = rand() % 100;
        unsigned short b = h_f2b((float)(rand() % nlev) * 0.5f - 1.0f);
        if (r == 0) b = 0x8000;          // -0.0
        else if (r == 1) b = 0x0000;     // +0.0
        else if (r == 2) b = 0xFF80;     // -inf
        else if (r == 3 && trial % 3 == 0) b = 0x7FC0;  // NaN
        if (trial % 11 == 0) b = (r < 50) ? 0x8000 : 0x0000;  // all zeros
        if (trial % 13 == 0) b = 0xFF80;                      // all -inf
        row[i] = b;
      }
      // host two-level reduce
      std::vector<float> pv(256, -INFINITY);
      std::vector<long long> pi(256, -1);
      for (int c = 0; c < NC; c++) {
        std::vector<float> v(256, -INFINITY);
        std::vector<long long> x(256, -1);
        for (int e = 0; e < C; e++) {
          float f = h_b2f(row[c * C + e]);
          if (f > -INFINITY) { v[e] = f; x[e] = e; }
        }
        tourney(v, x);
        pv[c] = v[0];
        pi[c] = (x[0] == -1) ? -1 : (((long long)c << 32) | x[0]);
      }
      // second level: thread c takes partial c only if it beats -inf
      std::vector<float> v2(256, -INFINITY);
      std::vector<long long> x2(256, -1);
      for (int c = 0; c < NC; c++) {
        if (pv[c] > -INFINITY) { v2[c] = pv[c]; x2[c] = pi[c]; }
      }
      tourney(v2, x2);
      long long want = x2[0] == -1 ? -1 : ((x2[0] >> 32) * C + (x2[0] & 0xFFFFFFFF));
      HIP_CHECK(hipMemcpy(d_row, row.data(), SHARD * 2, hipMemcpyHostToDevice));
      hipLaunchKernelGGL(keys_max, dim3(1), dim3(256), 0, 0, d_row, SHARD, d_k);
      unsigned long long k;
      HIP_CHECK(hipMemcpy(&k, d_k, 8, hipMemcpyDeviceToHost));
      long long got = k == 0ull ? -1 : (long long)(k & 0x7FFF);
      bool sign_ok = true;
      if (got >= 0) {
        bool negz = (k & 0x8000ull) != 0;
        sign_ok = negz == (row[got] == 0x8000);
      }
      if (got != want || !sign_ok) {
        if (bad < 5) printf("trial %d: keys %lld vs two-level %lld (sign ok %d)\n", trial, got, want, (int)sign_ok);
        bad++;
      }
    }
    printf("argmax keys vs two-level reduce: %d of 200 trials differ\n", bad);
    // the kernel's own keys for this run's logits
    std::vector<unsigned long long> kk(8 * 64);
    HIP_CHECK(hipMemcpyFromSymbol(kk.data(), HIP_SYMBOL(g_lmh_key), kk.size() * 8));
    unsigned long long best = 0;
    for (int i = 0; i < 8 * WPX; i++) best = std::max(best, kk[i]);
    printf("LM head tile keys: argmax column %llu\n", best & 0x7FFF);
  }
#endif
  std::sort(ms.begin(), ms.end());
  printf("MPK_LMH_BAL=%d tiles/slice=%d  median %.2f us  min %.2f us\n",
         MPK_LMH_BAL, ntiles, ms[ms.size() / 2], ms[0]);
  std::vector<unsigned short> out(SHARD);
  HIP_CHECK(hipMemcpy(out.data(), d_out, SHARD * 2, hipMemcpyDeviceToHost));
  FILE *f = fopen(out_path, "wb");
  fwrite(out.data(), 2, SHARD, f);
  fclose(f);
  int zeros = 0;
  for (int i = 0; i < SHARD; i++) zeros += out[i] == 0;
  printf("zero logits: %d of %d\n", zeros, SHARD);
  if (ref_path) {
    std::vector<unsigned short> ref(SHARD);
    FILE *g = fopen(ref_path, "rb");
    if (!g || fread(ref.data(), 2, SHARD, g) != (size_t)SHARD) {
      printf("no ref\n");
      return 1;
    }
    fclose(g);
    int diff = 0, big = 0;
    float maxrel = 0.0f;
    int amax_o = 0, amax_r = 0;
    for (int i = 0; i < SHARD; i++) {
      float a = h_b2f(out[i]), b = h_b2f(ref[i]);
      if (out[i] != ref[i]) diff++;
      float rel = fabsf(a - b) / std::max(fabsf(b), 1e-3f);
      maxrel = std::max(maxrel, rel);
      if (rel > 0.02f) big++;
      if (a > h_b2f(out[amax_o])) amax_o = i;
      if (b > h_b2f(ref[amax_r])) amax_r = i;
    }
    printf("vs ref: %d of %d bf16 differ, %d by >2%%, max rel %.4g; argmax %d vs %d\n",
           diff, SHARD, big, maxrel, amax_o, amax_r);
    return big ? 2 : 0;
  }
  return 0;
}
