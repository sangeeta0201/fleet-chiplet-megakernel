// Does a W13-shaped read (NB blocks x SLICE bytes, contiguous per block, dwordx4) run faster when the
// bytes are in the Infinity Cache (MALL) than from HBM? Cold (after a 2 GB flush), MALL-hot (read once, then
// EVICT_MB of unrelated traffic to push it out of L2), L2-hot (read twice back to back). Duration from in-kernel
// s_memrealtime (10 ns ticks): last block's end minus first block's start, so launch overhead is excluded.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP %s at %d\n", hipGetErrorString(e), __LINE__); exit(1); } } while (0)

typedef unsigned int u4 __attribute__((ext_vector_type(4)));

__global__ void __launch_bounds__(256) rd(u4 const *__restrict__ buf, size_t slice16, unsigned long long *t,
                                           unsigned *sink) {
  unsigned long long const t0 = __builtin_amdgcn_s_memrealtime();
  u4 const *p = buf + (size_t)blockIdx.x * slice16;
  u4 acc = {0u, 0u, 0u, 0u};
  for (size_t i = threadIdx.x; i < slice16; i += 256 * 8) {
    u4 v[8];
#pragma unroll
    for (int k = 0; k < 8; k++) {
      size_t const j = i + (size_t)k * 256;
      v[k] = j < slice16 ? p[j] : (u4){0u, 0u, 0u, 0u};
    }
#pragma unroll
    for (int k = 0; k < 8; k++) acc ^= v[k];
  }
  unsigned const x = acc.x ^ acc.y ^ acc.z ^ acc.w;
  __syncthreads();
  unsigned long long const t1 = __builtin_amdgcn_s_memrealtime();
  if (threadIdx.x == 0) {
    t[2 * blockIdx.x] = t0;
    t[2 * blockIdx.x + 1] = t1;
    if (x == 0x12345678u) sink[blockIdx.x] = x;
  }
}

__global__ void __launch_bounds__(256) touch(u4 const *__restrict__ buf, size_t n16, unsigned *sink) {
  u4 acc = {0u, 0u, 0u, 0u};
  for (size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; i < n16; i += (size_t)gridDim.x * 256) acc ^= buf[i];
  unsigned const x = acc.x ^ acc.y ^ acc.z ^ acc.w;
  if (x == 0x12345678u) sink[blockIdx.x] = x;
}

static double span_us(std::vector<unsigned long long> const &h, int nb) {
  unsigned long long a = ~0ull, b = 0;
  for (int i = 0; i < nb; i++) { a = std::min(a, h[2 * i]); b = std::max(b, h[2 * i + 1]); }
  return (b - a) * 0.01;
}
static double med_block_us(std::vector<unsigned long long> const &h, int nb) {
  std::vector<double> d(nb);
  for (int i = 0; i < nb; i++) d[i] = (h[2 * i + 1] - h[2 * i]) * 0.01;
  std::sort(d.begin(), d.end());
  return d[nb / 2];
}

int main(int argc, char **argv) {
  int const NB = argc > 1 ? atoi(argv[1]) : 144;
  size_t const SLICE = (size_t)(argc > 2 ? atoi(argv[2]) : 100352);  // 98 KB
  size_t const slice16 = SLICE / 16;
  size_t const TB = (size_t)NB * SLICE;
  size_t const FLUSH = (size_t)2 << 30;
  size_t const EV = (size_t)512 << 20;
  u4 *tgt, *flush, *ev;
  unsigned long long *dt;
  unsigned *sink;
  CK(hipMalloc(&tgt, TB));
  CK(hipMalloc(&flush, FLUSH));
  CK(hipMalloc(&ev, EV));
  CK(hipMalloc(&dt, 2 * NB * sizeof(unsigned long long)));
  CK(hipMalloc(&sink, 4096 * sizeof(unsigned)));
  CK(hipMemset(tgt, 1, TB));
  CK(hipMemset(flush, 2, FLUSH));
  CK(hipMemset(ev, 3, EV));
  CK(hipDeviceSynchronize());
  std::vector<unsigned long long> h(2 * NB);
  auto timed = [&]() {
    rd<<<NB, 256>>>(tgt, slice16, dt, sink);
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(h.data(), dt, h.size() * sizeof(unsigned long long), hipMemcpyDeviceToHost));
  };
  auto do_flush = [&]() { touch<<<1024, 256>>>(flush, FLUSH / 16, sink); CK(hipDeviceSynchronize()); };
  auto do_evict = [&](size_t mb) {
    if (mb) { touch<<<1024, 256>>>(ev, (mb << 20) / 16, sink); CK(hipDeviceSynchronize()); }
  };
  printf("NB=%d SLICE=%zu total=%.1f MB\n", NB, SLICE, TB / 1048576.0);
  for (int rep = 0; rep < 5; rep++) {
    do_flush();
    timed();
    double const cold = span_us(h, NB), cold_b = med_block_us(h, NB);
    timed();
    double const l2 = span_us(h, NB), l2_b = med_block_us(h, NB);
    double mall[4], mall_b[4];
    size_t const evs[4] = {32, 64, 128, 256};
    for (int e = 0; e < 4; e++) {
      do_flush();
      touch<<<1024, 256>>>(tgt, TB / 16, sink);  // bring it in
      CK(hipDeviceSynchronize());
      do_evict(evs[e]);
      timed();
      mall[e] = span_us(h, NB);
      mall_b[e] = med_block_us(h, NB);
    }
    printf("rep %d  cold %.2f us (block med %.2f)  L2-hot %.2f (%.2f)  after-evict 32MB %.2f (%.2f)  64MB %.2f (%.2f)"
           "  128MB %.2f (%.2f)  256MB %.2f (%.2f)\n",
           rep, cold, cold_b, l2, l2_b, mall[0], mall_b[0], mall[1], mall_b[1], mall[2], mall_b[2], mall[3],
           mall_b[3]);
  }
  return 0;
}
