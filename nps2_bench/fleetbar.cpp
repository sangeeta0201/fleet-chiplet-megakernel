// Fleet's ACTUAL O-proj barrier, standalone, in either mode, with a knob for
// arrival skew.
//
// Why this and not policy.cpp: policy.cpp's HIER/DUAL policies are the
// NPS2-aware designs. Comparing those against NPS1 is meaningless because
// NPS1 would never use them. What matters is whether the barrier fleet
// actually runs is mode-neutral, and if so, what makes it slow in fleet.
//
// Fleet's shape, copied from
// gang_linear_mxfp4_res_bias_rmsnorm_topk_mi300.cuh:1649-1704:
//   1. every tile      atom_add on its own XCD's counter        (per-XCD)
//   2. last per XCD    atom_add on ONE shared global counter    (8 total)
//   3. last of 8       lanes 0..7 publish 8 release flags       (dual-publish)
//   4. every tile      polls ITS OWN XCD's release flag         (1 line each)
//
// The skew knob is the point. Fleet's workers do not arrive together: they do
// different amounts of real work on data at different distances. A barrier
// bills the LAST arriver, so if NPS2 widens the arrival spread, the barrier
// pays that spread on top of its protocol cost -- and an isolated benchmark
// where every worker does identical trivial work cannot show it.
//
//   skew_ns = 0    all tiles arrive together  -> pure protocol cost
//   skew_ns > 0    tile t on XCD x waits (x * skew_ns) before arriving
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define HIP_OK(x)                                                              \
  do {                                                                         \
    hipError_t e = (x);                                                        \
    if (e != hipSuccess) {                                                     \
      printf("HIP FAIL %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
      return 2;                                                                \
    }                                                                          \
  } while (0)

static constexpr int kXcds = 8;
static constexpr int kStride = 16;   // HIER_STRIDE
static constexpr int kTiles = 23;    // tiles per XCD, as fleet runs it
static constexpr int kBlocks = kXcds * kTiles; // 184

__device__ __forceinline__ int my_xcd_hw() {
  unsigned int id;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(id));
  return (int)(id & (kXcds - 1));
}
__device__ __forceinline__ void st_wt(void *a, unsigned v) {
  asm volatile("global_store_dword %0, %1, off sc0 sc1" : : "v"(a), "v"(v) : "memory");
}
__device__ __forceinline__ unsigned ld_sc(void const *a) {
  unsigned v;
  asm volatile("global_load_dword %0, %1, off sc0 sc1\n s_waitcnt vmcnt(0)"
               : "=v"(v) : "v"(a) : "memory");
  return v;
}

// bar[0..7]   per-XCD release flags   (what every tile polls)
// bar[8]      global counter          (8 atomics, level 2)
// bar[28..35] per-XCD local counters  (level 1)
__global__ __launch_bounds__(256) void k_fleetbar(int *bar, int iters,
                                                  unsigned long long skew_ticks,
                                                  unsigned long long *out) {
  int const xcd = my_xcd_hw();
  int const tid = threadIdx.x;
  unsigned long long acc = 0;

  for (int it = 1; it <= iters; it++) {
    // Artificial arrival skew: XCD x arrives x*skew later. This is the term
    // that exists in fleet and not in an isolated collective benchmark.
    if (skew_ticks != 0) {
      unsigned long long const t0 = __builtin_amdgcn_s_memrealtime();
      unsigned long long const w = skew_ticks * (unsigned long long)xcd;
      while (__builtin_amdgcn_s_memrealtime() - t0 < w) {
      }
    }

    unsigned long long const tb = __builtin_amdgcn_s_memrealtime();
    __syncthreads();
    if (tid == 0) {
      // level 1: this XCD's own counter
      int const lp = atomicAdd(&bar[(28 + xcd) * kStride], 1);
      if ((lp % kTiles) == kTiles - 1) {
        // level 2: ONE shared counter, 8 atomics per barrier, 4 of which
        // cross the AID boundary in NPS2
        int const gp = atomicAdd(&bar[8 * kStride], 1);
        if ((gp % kXcds) == kXcds - 1) {
          // level 3: publish all eight release flags
          for (int x = 0; x < kXcds; x++) {
            st_wt(&bar[x * kStride], (unsigned)it);
          }
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        }
      }
    }
    // every tile polls its own XCD's flag
    if (tid == 0) {
      while (ld_sc(&bar[xcd * kStride]) < (unsigned)it) {
      }
    }
    __syncthreads();
    acc += __builtin_amdgcn_s_memrealtime() - tb;
  }
  if (tid == 0) {
    out[blockIdx.x] = acc;
  }
}

int main(int argc, char **argv) {
  int const iters = argc > 1 ? atoi(argv[1]) : 200;

  hipDeviceProp_t prop;
  HIP_OK(hipGetDeviceProperties(&prop, 0));
  printf("=== fleet's O-proj barrier, standalone: %s, %d CUs, %d blocks ===\n",
         prop.gcnArchName, prop.multiProcessorCount, kBlocks);

  int *bar;
  HIP_OK(hipMalloc(&bar, 64 * kStride * sizeof(int)));
  unsigned long long *out;
  HIP_OK(hipMalloc(&out, kBlocks * sizeof(unsigned long long)));

  printf("\n  skew_ns   barrier_ns  (median over %d blocks)   p95    max\n",
         kBlocks);
  int const skews[] = {0, 100, 250, 500, 1000, 2000};
  for (int si = 0; si < 6; si++) {
    unsigned long long const skew_ticks = (unsigned long long)skews[si] / 10;
    HIP_OK(hipMemset(bar, 0, 64 * kStride * sizeof(int)));
    HIP_OK(hipMemset(out, 0, kBlocks * sizeof(unsigned long long)));
    hipLaunchKernelGGL(k_fleetbar, dim3(kBlocks), dim3(256), 0, 0,
                       bar, iters, skew_ticks, out);
    hipError_t const e = hipDeviceSynchronize();
    if (e != hipSuccess) {
      printf("  %7d   launch failed: %s\n", skews[si], hipGetErrorString(e));
      continue;
    }
    std::vector<unsigned long long> h(kBlocks);
    HIP_OK(hipMemcpy(h.data(), out, kBlocks * sizeof(unsigned long long),
                     hipMemcpyDeviceToHost));
    std::vector<double> ns;
    for (auto v : h) {
      ns.push_back((double)v * 10.0 / (double)iters);
    }
    std::sort(ns.begin(), ns.end());
    // subtract the injected skew so the number is barrier cost, not the wait
    double const inj = (double)skews[si] * 3.5; // mean over xcd 0..7
    printf("  %7d   %10.0f   %10.0f %10.0f     (injected mean %.0f)\n",
           skews[si], ns[ns.size() / 2], ns[(size_t)(ns.size() * 0.95)],
           ns.back(), inj);
  }

  printf("\n  Read the skew=0 row as the protocol cost, and the slope as what\n"
         "  fleet actually pays. If skew=0 is mode-neutral but the slope is\n"
         "  steeper in NPS2, the barrier is fine and the arrival spread is the\n"
         "  whole story.\n");
  return 0;
}

