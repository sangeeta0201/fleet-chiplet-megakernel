// Fleet's barrier, made AID-local everywhere except two crossings.
//
// Baseline (fleetbar.cpp): all state on one plain hipMalloc, so in NPS2 about
// half of every access is remote. 1747 ns in NPS1, 3343 in NPS2, and the
// delta is skew-independent, so it is the accesses themselves.
//
// This version:
//   L1  per-XCD arrival counter   AID-homed          -> always local
//   L1b per-AID aggregate         AID-homed          -> always local
//   L2  shared global counter     ONE atomic per AID -> 2 crossings total
//   L3  release publish           dual-published     -> 8 far stores
//   L4  poll                      own AID's replica  -> always local
//
// So cross-AID traffic per barrier is 2 atomics + 8 write-through stores,
// instead of roughly half of everything.
//
// Also sweeps the store drain: fleet's `bar` spans t1->t2, which includes
// draining each worker's O-proj output into the SHARED row. Neither
// policy.cpp nor fleetbar has stores, which is the other reason they
// disagreed with fleet.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#if !defined(__HIP_DEVICE_COMPILE__)
#include "aid_local.h"
static void *rt_alloc_aid(size_t b, int a) {
  return mirage::aid::alloc_in_aid(b, a, 0ull);
}
#else
static void *rt_alloc_aid(size_t, int) { return nullptr; }
#endif

#define HIP_OK(x)                                                              \
  do {                                                                         \
    hipError_t e = (x);                                                        \
    if (e != hipSuccess) {                                                     \
      printf("HIP FAIL %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
      return 2;                                                                \
    }                                                                          \
  } while (0)

static constexpr int kXcds = 8;
static constexpr int kStride = 16;
static constexpr int kTiles = 23;
static constexpr int kBlocks = kXcds * kTiles;
static constexpr int kMaxStoreB = 4096;

__device__ __forceinline__ int my_xcd_hw() {
  unsigned int id;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(id));
  return (int)(id & (kXcds - 1));
}
__device__ __forceinline__ void st_wt(void *a, unsigned v) {
  asm volatile("global_store_dword %0, %1, off sc0 sc1" : : "v"(a), "v"(v) : "memory");
}
__device__ __forceinline__ void st_wt64(void *a, unsigned long long v) {
  asm volatile("global_store_dwordx2 %0, %1, off sc0 sc1" : : "v"(a), "v"(v) : "memory");
}
__device__ __forceinline__ unsigned ld_sc(void const *a) {
  unsigned v;
  asm volatile("global_load_dword %0, %1, off sc0 sc1\n s_waitcnt vmcnt(0)"
               : "=v"(v) : "v"(a) : "memory");
  return v;
}

struct Bar {
  int *glob;                  // always shared: the one all-to-all word
  int *cnt[2];                // per-XCD counters
  int *agg[2];                // per-AID aggregate
  int *rel[2];                // release flags
  unsigned long long *outbuf; // shared, stands in for attn_proj_out
  int local_state;            // 1 = AID-homed state + per-AID aggregate
  int store_b;
};

__global__ __launch_bounds__(256) void k_bar(Bar b, int iters,
                                             unsigned long long *out) {
  int const xcd = my_xcd_hw();
  int const aid = xcd >> 2;
  int const tid = threadIdx.x;
  int *const cnt = b.cnt[aid];
  int *const rel = b.rel[aid];
  unsigned long long acc = 0;

  for (int it = 1; it <= iters; it++) {
    unsigned long long const tb = __builtin_amdgcn_s_memrealtime();

    if (b.store_b > 0) {
      int const n64 = b.store_b / 8;
      unsigned long long *dst = b.outbuf + (size_t)blockIdx.x * (kMaxStoreB / 8);
      for (int i = tid; i < n64; i += (int)blockDim.x) {
        st_wt64(&dst[i], (unsigned long long)it);
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }

    __syncthreads();
    if (tid == 0) {
      // Distinct regions inside the arena. In the all-shared arm cnt, agg and
      // rel all point at the same allocation, so without separate offsets the
      // arrival counter and the release flag would be the SAME word -- which
      // silently corrupted the barrier and made it deadlock at 200 iters
      // while appearing to work at 50.
      //   rel  -> [xcd]          (0..7)
      //   agg  -> [26 + aid]     (26..27)
      //   cnt  -> [28 + xcd]     (28..35)
      int const lp = atomicAdd(&cnt[(28 + xcd) * kStride], 1);
      if ((lp % kTiles) == kTiles - 1) {
        bool release = false;
        if (b.local_state) {
          // One extra local level so only ONE worker per AID crosses.
          int const ap = atomicAdd(&b.agg[aid][(26 + aid) * kStride], 1);
          if ((ap % (kXcds / 2)) == (kXcds / 2) - 1) {
            int const gp = atomicAdd(&b.glob[8 * kStride], 1);
            release = ((gp % 2) == 1);
          }
        } else {
          int const gp = atomicAdd(&b.glob[8 * kStride], 1);
          release = ((gp % kXcds) == kXcds - 1);
        }
        if (release) {
          for (int x = 0; x < kXcds; x++) {
            st_wt(&b.rel[0][x * kStride], (unsigned)it);
            if (b.rel[1] != b.rel[0]) {
              st_wt(&b.rel[1][x * kStride], (unsigned)it);
            }
          }
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        }
      }
      while (ld_sc(&rel[xcd * kStride]) < (unsigned)it) {
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
  printf("=== fleet barrier: AID-local state + 2 crossings, %s, %d blocks ===\n",
         prop.gcnArchName, kBlocks);

  size_t const sz = (size_t)64 * kStride * sizeof(int);
  int *sh;
  HIP_OK(hipMalloc(&sh, sz));
  unsigned long long *outbuf;
  HIP_OK(hipMalloc(&outbuf, (size_t)kBlocks * kMaxStoreB));
  unsigned long long *out;
  HIP_OK(hipMalloc(&out, kBlocks * sizeof(unsigned long long)));

  int *ac[2] = {}, *ar[2] = {}, *ag[2] = {};
  for (int a = 0; a < 2; a++) {
    ac[a] = static_cast<int *>(rt_alloc_aid(sz, a));
    ar[a] = static_cast<int *>(rt_alloc_aid(sz, a));
    ag[a] = static_cast<int *>(rt_alloc_aid(sz, a));
  }
  bool const ok = ac[0] && ac[1] && ar[0] && ar[1] && ag[0] && ag[1];
  printf("  AID arenas: %s\n", ok ? "allocated" : "UNAVAILABLE (NPS1/stock)");

  auto run = [&](char const *label, bool loc, int sb) {
    Bar b{};
    b.glob = sh;
    b.cnt[0] = loc ? ac[0] : sh;  b.cnt[1] = loc ? ac[1] : sh;
    b.rel[0] = loc ? ar[0] : sh;  b.rel[1] = loc ? ar[1] : sh;
    b.agg[0] = loc ? ag[0] : sh;  b.agg[1] = loc ? ag[1] : sh;
    b.outbuf = outbuf; b.local_state = loc ? 1 : 0; b.store_b = sb;
    (void)hipMemset(sh, 0, sz);
    if (loc) {
      for (int a = 0; a < 2; a++) {
        (void)hipMemset(ac[a], 0, sz);
        (void)hipMemset(ar[a], 0, sz);
        (void)hipMemset(ag[a], 0, sz);
      }
    }
    (void)hipMemset(out, 0, kBlocks * sizeof(unsigned long long));
    hipLaunchKernelGGL(k_bar, dim3(kBlocks), dim3(256), 0, 0, b, iters, out);
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) {
      printf("  %-30s FAILED %s\n", label, hipGetErrorString(e)); return;
    }
    std::vector<unsigned long long> h(kBlocks);
    (void)hipMemcpy(h.data(), out, kBlocks * sizeof(unsigned long long),
                    hipMemcpyDeviceToHost);
    std::vector<double> ns;
    for (auto v : h) { ns.push_back((double)v * 10.0 / (double)iters); }
    std::sort(ns.begin(), ns.end());
    printf("  %-30s median %6.0f  p95 %6.0f ns\n",
           label, ns[ns.size()/2], ns[(size_t)(ns.size()*0.95)]);
  };

  printf("\n--- barrier state placement (no stores) ---\n");
  printf("  NPS1 reference 1747 | NPS2 all-shared reference 3343\n");
  run("all shared", false, 0);
  if (ok) { run("AID-local + 2 crossings", true, 0); }

  printf("\n--- store drain into the SHARED row ---\n");
  int const sbs[] = {64, 256, 1024, 4096};
  for (int i = 0; i < 4; i++) {
    char l[64];
    snprintf(l, sizeof l, "shared state, %5d B", sbs[i]);
    run(l, false, sbs[i]);
  }
  if (ok) {
    for (int i = 0; i < 4; i++) {
      char l[64];
      snprintf(l, sizeof l, "AID state,    %5d B", sbs[i]);
      run(l, true, sbs[i]);
    }
  }
  return 0;
}

