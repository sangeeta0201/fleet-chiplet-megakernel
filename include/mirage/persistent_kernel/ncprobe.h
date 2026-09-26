// NC scratch placement probe (opt-in: MPK_NC_PROBE). One lane per physical
// XCC (claimed via HW_REG_XCC_ID) times `sc0 sc1` loads, each drained before
// the next, at a list of addresses. On MTYPE_NC these never hit L2, so the
// faster XCC group is the home of the address; on MTYPE_RW they hit L2 and
// read flat on every XCC.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

namespace mpk_ncprobe {

__global__ void ncprobe_kernel(unsigned long long const *__restrict addrs,
                               int n,
                               unsigned *__restrict claim,
                               unsigned *__restrict out_ns) {
  if (threadIdx.x != 0) {
    return;
  }
  unsigned xcc;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcc));
  xcc &= 7u;
  if (atomicCAS(&claim[xcc], 0u, 1u) != 0u) {
    return;
  }
  int const r0 = (int)((xcc * (unsigned)n) / 8u);
  for (int i = 0; i < n; i++) {
    int const r = (r0 + i) % n;
    unsigned long long const a = addrs[r];
    unsigned best = 0xffffffffu;
    for (int rep = 0; rep < 5; rep++) {
      unsigned v;
      asm volatile("global_load_dword %0, %1, off sc0 sc1\n\t"
                   "s_waitcnt vmcnt(0)"
                   : "=v"(v)
                   : "v"(a)
                   : "memory");
      unsigned long long const t0 = __builtin_amdgcn_s_memrealtime();
      for (int k = 0; k < 8; k++) {
        asm volatile("global_load_dword %0, %1, off sc0 sc1\n\t"
                     "s_waitcnt vmcnt(0)"
                     : "=v"(v)
                     : "v"(a)
                     : "memory");
      }
      unsigned long long const t1 = __builtin_amdgcn_s_memrealtime();
      unsigned const ns = (unsigned)((t1 - t0) * 10ull / 8);
      best = ns < best ? ns : best;
    }
    out_ns[r * 8 + (int)xcc] = best;
  }
}

inline void probe_list(char const *tag,
                       std::vector<unsigned long long> const &addrs,
                       unsigned long long base) {
  int const n = (int)addrs.size();
  unsigned long long *d_addrs = nullptr;
  unsigned *d_claim = nullptr, *d_out = nullptr;
  hipMalloc((void **)&d_addrs, n * sizeof(unsigned long long));
  hipMalloc((void **)&d_claim, 8 * sizeof(unsigned));
  hipMalloc((void **)&d_out, (size_t)n * 8 * sizeof(unsigned));
  hipMemcpy(d_addrs, addrs.data(), n * sizeof(unsigned long long),
            hipMemcpyHostToDevice);
  hipMemset(d_claim, 0, 8 * sizeof(unsigned));
  hipMemset(d_out, 0xff, (size_t)n * 8 * sizeof(unsigned));
  hipLaunchKernelGGL(ncprobe_kernel, dim3(128), dim3(64), 0, 0, d_addrs, n,
                     d_claim, d_out);
  hipError_t const err = hipDeviceSynchronize();
  std::vector<unsigned> out((size_t)n * 8, 0xffffffffu);
  unsigned claim[8] = {0};
  if (err == hipSuccess) {
    hipMemcpy(out.data(), d_out, out.size() * sizeof(unsigned),
              hipMemcpyDeviceToHost);
    hipMemcpy(claim, d_claim, sizeof claim, hipMemcpyDeviceToHost);
  }
  hipFree(d_addrs);
  hipFree(d_claim);
  hipFree(d_out);
  if (err != hipSuccess) {
    printf("[NCPROBE] %s: kernel failed: %s\n", tag, hipGetErrorString(err));
    fflush(stdout);
    return;
  }
  printf("[NCPROBE] %s: XCCs claimed %u%u%u%u%u%u%u%u\n", tag, claim[0],
         claim[1], claim[2], claim[3], claim[4], claim[5], claim[6], claim[7]);
  for (int r = 0; r < n; r++) {
    unsigned lo = 0, hi = 0;
    for (int x = 0; x < 8; x++) {
      (x < 4 ? lo : hi) += out[r * 8 + x];
    }
    lo /= 4;
    hi /= 4;
    char const *home = (lo < 160 && hi < 160) ? "L2"
                       : (lo + 40 < hi)       ? "AID0"
                       : (hi + 40 < lo)       ? "AID1"
                                              : "?";
    printf("[NCPROBE] %s +%6.2f MiB lo=%u hi=%u %s | %u %u %u %u %u %u %u %u\n",
           tag, (double)(addrs[r] - base) / (1 << 20), lo, hi, home,
           out[r * 8 + 0], out[r * 8 + 1], out[r * 8 + 2], out[r * 8 + 3],
           out[r * 8 + 4], out[r * 8 + 5], out[r * 8 + 6], out[r * 8 + 7]);
  }
  fflush(stdout);
}

} // namespace mpk_ncprobe
