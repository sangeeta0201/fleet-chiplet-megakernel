// Can the event-counter mirror be made MONOTONE, so the expensive fallback
// read can be dropped entirely?
//
// What the first benchmark established (SPX/NPS2, 184 blocks):
//   poll shared counter          10,591 ns      <- the read IS the cost
//   poll mirror, fallback 1/64    3,117 ns
//   poll mirror, fallback 1/1024 16,771 ns      <- mirror goes STALE
//   poll mirror, fallback 1/65536 1,536,807 ns
// NPS1 shared is 1,586 ns, so NPS2's shared read is 6.7x worse.
//
// The mirror is stale because EIGHT XCDs publish to ONE slot with unordered
// write-through stores, so a late store from an earlier XCD moves it
// backwards. Progress then depends on the periodic truth read -- the
// expensive one -- which is why the net saving in fleet is zero.
//
// Design under test (arm "subtotal"): give each AID its own slot with
// EXACTLY ONE writer -- the last XCD of that AID -- and keep both slots in
// both replicas. Every slot then has a single publisher, so it is monotone; a
// waiter reads two u64s both homed in its own AID and sums them, and never
// reads the shared counter at all.
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
static constexpr int kPerAid = kXcds / 2;

__device__ __forceinline__ int my_xcd_hw() {
  unsigned int id;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(id));
  return (int)(id & (kXcds - 1));
}
__device__ __forceinline__ void st_wt64(void *a, unsigned long long v) {
  asm volatile("global_store_dwordx2 %0, %1, off sc0 sc1"
               : : "v"(a), "v"(v) : "memory");
}
__device__ __forceinline__ unsigned long long ld_ev(void const *a) {
  unsigned long long v;
  asm volatile("global_load_dwordx2 %0, %1, off sc0 sc1\n s_waitcnt vmcnt(0)"
               : "=v"(v) : "v"(a) : "memory");
  return v;
}

// mode 0 = shared only, 1 = mirror + 1/64 fallback (fleet today),
// mode 2 = per-AID subtotal slots, no fallback
struct Args {
  unsigned long long *glob;
  unsigned long long *loc;      // per-XCD arrival counters
  unsigned long long *agg[2];   // per-AID aggregate, own allocation per AID
  unsigned long long *mir[2];
  int mode;
  unsigned long long *out_wait;
  unsigned long long *out_spin;
  int *out_bad;
};

__global__ __launch_bounds__(64) void k_ev(Args a, int iters) {
  int const xcd = my_xcd_hw();
  int const aid = xcd >> 2;
  int const tid = threadIdx.x;
  unsigned long long wait_acc = 0, spin_acc = 0;

  for (int it = 1; it <= iters; it++) {
    unsigned long long const need = (unsigned long long)kXcds *
                                    (unsigned long long)it;
    if (tid == 0) {
      unsigned long long const lc =
          atomicAdd(&a.loc[xcd * kStride], 1ULL) + 1ULL;
      if (lc == (unsigned long long)kTiles * (unsigned long long)it) {
        if (a.mode == 2) {
          // One aggregate per AID, on a line in that AID: a same-AID atomic,
          // which IS coherent (unlike a cross-AID one).
          unsigned long long const ac =
              atomicAdd(&a.agg[aid][0], 1ULL) + 1ULL;
          if (ac == (unsigned long long)kPerAid * (unsigned long long)it) {
            // Last XCD of this AID: sole writer of this AID's subtotal, so
            // the value only ever increases -- monotone by construction.
            st_wt64(&a.mir[0][aid * kStride], ac);
            st_wt64(&a.mir[1][aid * kStride], ac);
          }
        } else {
          unsigned long long const old = atomicAdd(a.glob, 1ULL);
          if (a.mode == 1) {
            st_wt64(&a.mir[0][0], old + 1ULL);
            st_wt64(&a.mir[1][0], old + 1ULL);
          }
        }
      }
    }

    if (tid == 0) {
      unsigned long long const tb = __builtin_amdgcn_s_memrealtime();
      unsigned spin = 0;
      for (;;) {
        unsigned long long v;
        if (a.mode == 2) {
          // two slots, BOTH homed in this tile's own AID
          v = ld_ev(&a.mir[aid][0]) + ld_ev(&a.mir[aid][kStride]);
        } else if (a.mode == 1 && (spin & 63u) != 63u) {
          v = ld_ev(&a.mir[aid][0]);
        } else {
          v = ld_ev(a.glob);
        }
        if (v >= need) {
          break;
        }
        spin++;
        if (spin > 4000000u) {
          atomicAdd(a.out_bad, 1);
          break;
        }
        __builtin_amdgcn_s_sleep(1);
      }
      wait_acc += __builtin_amdgcn_s_memrealtime() - tb;
      spin_acc += spin;
    }
    __syncthreads();
  }
  if (tid == 0) {
    a.out_wait[blockIdx.x] = wait_acc;
    a.out_spin[blockIdx.x] = spin_acc;
  }
}

int main(int argc, char **argv) {
  int const iters = argc > 1 ? atoi(argv[1]) : 200;
  hipDeviceProp_t prop;
  HIP_OK(hipGetDeviceProperties(&prop, 0));
  printf("=== monotone event mirror: %s, %d blocks, %d iters ===\n",
         prop.gcnArchName, kBlocks, iters);

  size_t const sz = (size_t)kXcds * kStride * sizeof(unsigned long long);
  unsigned long long *glob, *loc, *ow, *osp;
  unsigned long long *ag[2] = {nullptr, nullptr};
  HIP_OK(hipMalloc(&glob, sz));
  HIP_OK(hipMalloc(&loc, sz));
  HIP_OK(hipMalloc(&ow, kBlocks * sizeof(unsigned long long)));
  HIP_OK(hipMalloc(&osp, kBlocks * sizeof(unsigned long long)));
  int *bad;
  HIP_OK(hipMalloc(&bad, sizeof(int)));

  unsigned long long *m[2] = {nullptr, nullptr};
  for (int i = 0; i < 2; i++) {
    m[i] = static_cast<unsigned long long *>(rt_alloc_aid(sz, i));
  }
  bool const ok = m[0] && m[1];
  printf("  AID mirrors: %s\n", ok ? "allocated" : "UNAVAILABLE (NPS1/stock)");
  if (!ok) { HIP_OK(hipMalloc(&m[0], sz)); HIP_OK(hipMalloc(&m[1], sz)); }
  for (int i = 0; i < 2; i++) {
    ag[i] = static_cast<unsigned long long *>(rt_alloc_aid(sz, i));
    if (ag[i] == nullptr) { HIP_OK(hipMalloc(&ag[i], sz)); }
  }


  auto run = [&](char const *label, int mode) {
    Args a{};
    a.glob = glob; a.loc = loc; a.agg[0] = ag[0]; a.agg[1] = ag[1];
    a.mir[0] = m[0]; a.mir[1] = m[1]; a.mode = mode;
    a.out_wait = ow; a.out_spin = osp; a.out_bad = bad;
    (void)hipMemset(glob, 0, sz);
    (void)hipMemset(loc, 0, sz);
    (void)hipMemset(m[0], 0, sz);
    (void)hipMemset(m[1], 0, sz);
    (void)hipMemset(ag[0], 0, sz);
    (void)hipMemset(ag[1], 0, sz);
    (void)hipMemset(bad, 0, sizeof(int));
    (void)hipMemset(ow, 0, kBlocks * sizeof(unsigned long long));
    (void)hipMemset(osp, 0, kBlocks * sizeof(unsigned long long));
    hipLaunchKernelGGL(k_ev, dim3(kBlocks), dim3(64), 0, 0, a, iters);
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) {
      printf("  %-30s FAILED %s\n", label, hipGetErrorString(e)); return;
    }
    std::vector<unsigned long long> w(kBlocks), s(kBlocks);
    (void)hipMemcpy(w.data(), ow, kBlocks*sizeof(unsigned long long), hipMemcpyDeviceToHost);
    (void)hipMemcpy(s.data(), osp, kBlocks*sizeof(unsigned long long), hipMemcpyDeviceToHost);
    int hb = 0;
    (void)hipMemcpy(&hb, bad, sizeof(int), hipMemcpyDeviceToHost);
    std::vector<double> ns; double sp = 0;
    for (int i = 0; i < kBlocks; i++) {
      ns.push_back((double)w[i] * 10.0 / iters);
      sp += (double)s[i] / iters;
    }
    std::sort(ns.begin(), ns.end());
    printf("  %-30s wait_med %8.0f ns  p95 %8.0f  spins %7.1f%s\n",
           label, ns[ns.size()/2], ns[(size_t)(ns.size()*0.95)],
           sp / kBlocks, hb ? "  GAVE UP" : "");
  };

  printf("\n");
  run("shared counter only", 0);
  run("mirror + 1/64 fallback (today)", 1);
  run("per-AID subtotals, no fallback", 2);
  printf("\n  NPS1 reference for the shared arm: 1,586 ns\n");
  return 0;
}

