/* Copyright 2025 CMU
 *
 * Unit test for the write-through store primitives in mpk_atoms.cuh.
 *
 * WHY. Instantiating the MLA decode with WRITE_THROUGH=true (the fused
 * configuration, previously never exercised by any test) failed every
 * split-KV case, and swapping the one st_wt_u128 in its epilogue for two
 * st_wt_u64 made all 19 cases pass. That points at st_wt_u128, but the MLA
 * epilogue is a poor place to characterize a store: it runs under MFMA
 * register pressure with a computed scatter address. st_wt_u128 is also used
 * to ZERO the MoE accumulator workspace (gang_oproj_router_fused) and to
 * publish EP partials (xrank_sum_add), so whether it is actually broken
 * decides whether those are silently corrupt too.
 *
 * This writes a known pattern through each primitive and reads it back with
 * ordinary loads. No MFMA, no pressure, nothing else in flight.
 */

#include <cstdio>
#include <hip/hip_runtime.h>
#include <vector>

#include "mirage/persistent_kernel/mpk_atoms.cuh"

static constexpr int N = 4096; // dwords

__global__ void k_u128(unsigned *buf) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned base = (unsigned)(i * 4);
  st_wt_u128(
      (void *)(buf + i * 4), base + 0u, base + 1u, base + 2u, base + 3u);
}

__global__ void k_u64(unsigned *buf) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned base = (unsigned)(i * 4);
  unsigned long long lo = (unsigned long long)base | ((unsigned long long)(base + 1u) << 32);
  unsigned long long hi = (unsigned long long)(base + 2u) | ((unsigned long long)(base + 3u) << 32);
  st_wt_u64((void *)(buf + i * 4), lo);
  st_wt_u64((void *)(buf + i * 4 + 2), hi);
}

__global__ void k_u32(unsigned *buf) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned base = (unsigned)(i * 4);
#pragma unroll
  for (int h = 0; h < 4; h++) {
    st_wt_u32((void *)(buf + i * 4 + h), base + (unsigned)h);
  }
}

__global__ void k_zero128(unsigned *buf) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  st_wt_zero128((void *)(buf + i * 4));
}

template <typename F>
static int check(char const *tag, F launch, bool expect_zero) {
  unsigned *d;
  hipMalloc(&d, N * 4);
  // Poison so a dropped dword is visible rather than accidentally correct.
  hipMemset(d, 0xAB, N * 4);
  launch(d);
  hipError_t e = hipDeviceSynchronize();
  if (e != hipSuccess) {
    printf("FAIL %-12s launch: %s\n", tag, hipGetErrorString(e));
    hipFree(d);
    return 1;
  }
  std::vector<unsigned> h(N);
  hipMemcpy(h.data(), d, N * 4, hipMemcpyDeviceToHost);
  hipFree(d);

  int bad = 0;
  int bad_by_lane[4] = {0, 0, 0, 0};
  for (int i = 0; i < N; i++) {
    unsigned want = expect_zero ? 0u : (unsigned)i;
    if (h[i] != want) {
      if (bad < 6) {
        printf("  [%4d] got 0x%08x want 0x%08x\n", i, h[i], want);
      }
      bad++;
      bad_by_lane[i & 3]++;
    }
  }
  if (bad) {
    printf("FAIL %-12s %d/%d wrong  (by position mod 4: %d %d %d %d)\n",
           tag, bad, N, bad_by_lane[0], bad_by_lane[1], bad_by_lane[2],
           bad_by_lane[3]);
    return 1;
  }
  printf("PASS %-12s all %d dwords correct\n", tag, N);
  return 0;
}

int main() {
  int rc = 0;
  int const blocks = N / 4 / 64;
  rc |= check("st_wt_u128", [&](unsigned *d) {
    hipLaunchKernelGGL(k_u128, dim3(blocks), dim3(64), 0, 0, d);
  }, false);
  rc |= check("st_wt_u64x2", [&](unsigned *d) {
    hipLaunchKernelGGL(k_u64, dim3(blocks), dim3(64), 0, 0, d);
  }, false);
  rc |= check("st_wt_u32x4", [&](unsigned *d) {
    hipLaunchKernelGGL(k_u32, dim3(blocks), dim3(64), 0, 0, d);
  }, false);
  rc |= check("st_wt_zero128", [&](unsigned *d) {
    hipLaunchKernelGGL(k_zero128, dim3(blocks), dim3(64), 0, 0, d);
  }, true);
  return rc;
}
