// The event-dependency wait, standalone, in either mode.
//
// This is fleet's two-level event counting, copied from
// persistent_kernel.cuh:3274-3318 and :1919-1921:
//
//   every worker    atom_add on its XCD's local counter
//   last per XCD    atomicAdd(flush) on ONE global u64 counter
//                   + st_wt_u64 publish into both AID mirrors
//   every worker    spin until counter >= num_triggers * iteration
//
// ATT says this wait is 66% of Phase 7's NPS2 stall increase, at 15,522
// cyc/hit against NPS1's 188, while the load issues in 8 cycles in BOTH modes
// and NPS2 executes FEWER iterations. So the cost is not the read. This
// isolates the wait so the hypothesis can be tested in seconds instead of
// 5-minute fleet runs.
//
// Arms:
//   shared   poll the global counter directly (spanning NC in NPS2)
//   mirror   poll the AID-local mirror, shared every 64th spin (fleet today)
//   skew     add per-XCD producer delay, to see whether arrival spread or
//            read latency sets the wait
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
static constexpr int kStride = 16;   // u64 slots, keeps XCDs off one line
static constexpr int kTiles = 23;    // workers per XCD, as fleet runs Phase 7
static constexpr int kBlocks = kXcds * kTiles;

__device__ __forceinline__ int my_xcd_hw() {
  unsigned int id;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(id));
  return (int)(id & (kXcds - 1));
}
__device__ __forceinline__ void st_wt64(void *a, unsigned long long v) {
  asm volatile("global_store_dwordx2 %0, %1, off sc0 sc1"
               : : "v"(a), "v"(v) : "memory");
}
// MPK_LD_EVENT's shape: a system-scope u64 load, exactly what ATT traced
__device__ __forceinline__ unsigned long long ld_ev(void const *a) {
  unsigned long long v;
  asm volatile("global_load_dwordx2 %0, %1, off sc0 sc1\n s_waitcnt vmcnt(0)"
               : "=v"(v) : "v"(a) : "memory");
  return v;
}

struct Args {
  unsigned long long *glob;   // the authoritative counter
  unsigned long long *loc;    // per-XCD local counters
  unsigned long long *mir[2]; // AID mirrors
  int use_mirror;             // 1 = poll the mirror (fleet today)
  unsigned fallback_mask;     // read the shared counter when (spin & mask) == mask
  unsigned long long skew;    // per-XCD producer delay, ticks
  unsigned long long *out_wait;
  unsigned long long *out_spin;
};

__global__ __launch_bounds__(64) void k_evwait(Args a, int iters) {
  int const xcd = my_xcd_hw();
  int const aid = xcd >> 2;
  int const tid = threadIdx.x;
  unsigned long long wait_acc = 0, spin_acc = 0;

  for (int it = 1; it <= iters; it++) {
    // ---- producer: optional skew, then the two-level arrival ----
    if (a.skew != 0) {
      unsigned long long const t0 = __builtin_amdgcn_s_memrealtime();
      unsigned long long const w = a.skew * (unsigned long long)xcd;
      while (__builtin_amdgcn_s_memrealtime() - t0 < w) {
      }
    }
    if (tid == 0) {
      unsigned long long const lc =
          atomicAdd(&a.loc[xcd * kStride], 1ULL) + 1ULL;
      if (lc == (unsigned long long)kTiles * (unsigned long long)it) {
        // last on this XCD: flush 1 to the global counter, publish mirrors
        unsigned long long const old = atomicAdd(a.glob, 1ULL);
        st_wt64(&a.mir[0][0], old + 1ULL);
        st_wt64(&a.mir[1][0], old + 1ULL);
      }
    }

    // ---- consumer: spin until the counter reaches this iteration ----
    unsigned long long const need = (unsigned long long)kXcds *
                                    (unsigned long long)it;
    if (tid == 0) {
      unsigned long long const tb = __builtin_amdgcn_s_memrealtime();
      unsigned spin = 0;
      for (;;) {
        unsigned long long v;
        if (a.use_mirror && (spin & a.fallback_mask) != a.fallback_mask) {
          v = ld_ev(&a.mir[aid][0]);
        } else {
          v = ld_ev(a.glob);
        }
        if (v >= need) {
          break;
        }
        spin++;
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
  printf("=== event-dependency wait, standalone: %s, %d blocks, %d iters ===\n",
         prop.gcnArchName, kBlocks, iters);

  size_t const sz = (size_t)kXcds * kStride * sizeof(unsigned long long);
  unsigned long long *glob, *loc, *ow, *os_;
  HIP_OK(hipMalloc(&glob, sz));
  HIP_OK(hipMalloc(&loc, sz));
  HIP_OK(hipMalloc(&ow, kBlocks * sizeof(unsigned long long)));
  HIP_OK(hipMalloc(&os_, kBlocks * sizeof(unsigned long long)));

  unsigned long long *m[2] = {nullptr, nullptr};
  for (int i = 0; i < 2; i++) {
    m[i] = static_cast<unsigned long long *>(rt_alloc_aid(sz, i));
  }
  bool const aid_ok = m[0] && m[1];
  printf("  AID mirrors: %s\n", aid_ok ? "allocated" : "UNAVAILABLE (NPS1/stock)");
  if (!aid_ok) {
    HIP_OK(hipMalloc(&m[0], sz));
    HIP_OK(hipMalloc(&m[1], sz));
  }

  auto run = [&](char const *label, int mirror, unsigned long long skew, unsigned fbm = 63u) {
    Args a{};
    a.glob = glob; a.loc = loc; a.mir[0] = m[0]; a.mir[1] = m[1];
    a.use_mirror = mirror; a.skew = skew / 10; a.fallback_mask = fbm;
    a.out_wait = ow; a.out_spin = os_;
    (void)hipMemset(glob, 0, sz);
    (void)hipMemset(loc, 0, sz);
    (void)hipMemset(m[0], 0, sz);
    (void)hipMemset(m[1], 0, sz);
    (void)hipMemset(ow, 0, kBlocks * sizeof(unsigned long long));
    (void)hipMemset(os_, 0, kBlocks * sizeof(unsigned long long));
    hipLaunchKernelGGL(k_evwait, dim3(kBlocks), dim3(64), 0, 0, a, iters);
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) {
      printf("  %-26s FAILED %s\n", label, hipGetErrorString(e)); return;
    }
    std::vector<unsigned long long> w(kBlocks), s(kBlocks);
    (void)hipMemcpy(w.data(), ow, kBlocks * sizeof(unsigned long long), hipMemcpyDeviceToHost);
    (void)hipMemcpy(s.data(), os_, kBlocks * sizeof(unsigned long long), hipMemcpyDeviceToHost);
    std::vector<double> ns;
    double sp = 0;
    for (int i = 0; i < kBlocks; i++) {
      ns.push_back((double)w[i] * 10.0 / iters);
      sp += (double)s[i] / iters;
    }
    std::sort(ns.begin(), ns.end());
    printf("  %-26s wait_med %8.0f ns  p95 %8.0f  spins/iter %6.1f\n",
           label, ns[ns.size()/2], ns[(size_t)(ns.size()*0.95)], sp / kBlocks);
  };

  printf("\n--- no injected skew: is the READ the cost? ---\n");
  run("poll shared counter", 0, 0);
  run("poll AID mirror", 1, 0);

  printf("\n--- injected producer skew: is ARRIVAL the cost? ---\n");
  for (unsigned long long sk : {250ull, 1000ull, 4000ull}) {
    char l[64];
    snprintf(l, sizeof l, "shared, skew %4llu ns", sk);
    run(l, 0, sk);
    snprintf(l, sizeof l, "mirror, skew %4llu ns", sk);
    run(l, 1, sk);
  }
  printf("\n--- fallback interval sweep, no skew (how often to read the truth) ---\n");
  run("mirror, fallback 1/64", 1, 0, 63u);
  run("mirror, fallback 1/256", 1, 0, 255u);
  run("mirror, fallback 1/1024", 1, 0, 1023u);
  run("mirror, fallback 1/65536", 1, 0, 65535u);

  printf("\n  If the two no-skew rows match, the read is not the cost and the\n"
         "  mirror cannot help -- the wait is producer arrival, and only the\n"
         "  skew rows will move.\n");
  return 0;
}

