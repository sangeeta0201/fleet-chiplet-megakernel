// Is the NPS2 penalty the AID BOUNDARY, or NPS2 itself?
//
// Established: in NPS2 every memory wait costs 3.6x more to COMPLETE
// (s_waitcnt 156 -> 559 cyc/hit) while the loads issue at normal cost, and
// that accounts for 86% of the stall increase. Placement and access-count
// changes have all failed to move it.
//
// This isolates the boundary. Confine the whole workload to FOUR XCDs -- all
// inside one AID -- and put the polled line either in that same AID or in the
// far one:
//
//   8 XCDs, spanning line     the configuration fleet runs
//   4 XCDs (0-3), line in AID0   all traffic local, no boundary crossed
//   4 XCDs (0-3), line in AID1   same 4 XCDs, every access crosses
//   4 XCDs (4-7), line in AID1   the mirror image, to rule out XCD identity
//
// If "4 XCDs, local" matches NPS1 while "4 XCDs, far" is slow, the boundary
// is the whole story. If "4 XCDs, local" is STILL slow, then NPS2 costs
// something even without crossing, and no placement work can ever fix it.
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

static constexpr int kStride = 16;
static constexpr int kTiles = 23;
static constexpr int kBlocks = 8 * kTiles; // always launch 184

__device__ __forceinline__ int my_xcd_hw() {
  unsigned int id;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(id));
  return (int)(id & 7u);
}
__device__ __forceinline__ unsigned long long ld_ev(void const *a) {
  unsigned long long v;
  asm volatile("global_load_dwordx2 %0, %1, off sc0 sc1\n s_waitcnt vmcnt(0)"
               : "=v"(v) : "v"(a) : "memory");
  return v;
}

struct Args {
  unsigned long long *cnt;    // the polled counter (single-group mode)
  unsigned long long *loc;    // per-XCD arrival counters
  unsigned xcd_mask;          // which XCDs participate
  int n_active;               // how many XCDs that is
  unsigned long long *out_wait;
  unsigned long long *out_hits;
  // SPLIT mode: all 8 XCDs run, but each AID is its own independent group
  // with its own AID-local counter, so no wait ever crosses the boundary.
  // This is the configuration that should give one-AID latency at full
  // occupancy.
  int split;
  unsigned long long *cnt_aid[2];
  unsigned long long *loc_aid[2];
};

__global__ __launch_bounds__(64) void k_one(Args a, int iters) {
  int const xcd = my_xcd_hw();
  int const tid = threadIdx.x;
  bool const active = (a.xcd_mask >> xcd) & 1u;
  unsigned long long wait_acc = 0, hits = 0;

  if (!active) {
    return; // inactive XCDs take no part at all
  }

  // In split mode every structure this XCD touches is homed in its own AID,
  // and the group is just the four XCDs of that AID.
  int const aid = xcd >> 2;
  unsigned long long *const cnt = a.split ? a.cnt_aid[aid] : a.cnt;
  unsigned long long *const loc = a.split ? a.loc_aid[aid] : a.loc;
  int const group = a.split ? 4 : a.n_active;

  for (int it = 1; it <= iters; it++) {
    if (tid == 0) {
      unsigned long long const lc =
          atomicAdd(&loc[xcd * kStride], 1ULL) + 1ULL;
      if (lc == (unsigned long long)kTiles * (unsigned long long)it) {
        atomicAdd(cnt, 1ULL);
      }
    }
    unsigned long long const need =
        (unsigned long long)group * (unsigned long long)it;
    if (tid == 0) {
      unsigned long long const tb = __builtin_amdgcn_s_memrealtime();
      unsigned spin = 0;
      while (ld_ev(cnt) < need) {
        spin++;
        if (spin > 2000000u) {
          break;
        }
        __builtin_amdgcn_s_sleep(1);
      }
      wait_acc += __builtin_amdgcn_s_memrealtime() - tb;
      hits += spin + 1;
    }
    __syncthreads();
  }
  if (tid == 0) {
    a.out_wait[blockIdx.x] = wait_acc;
    a.out_hits[blockIdx.x] = hits;
  }
}

int main(int argc, char **argv) {
  int const iters = argc > 1 ? atoi(argv[1]) : 200;
  hipDeviceProp_t prop;
  HIP_OK(hipGetDeviceProperties(&prop, 0));
  printf("=== boundary vs mode: %s, launching %d blocks ===\n",
         prop.gcnArchName, kBlocks);

  // The counter sits at [0] and the eight per-XCD arrival slots start at
  // +kStride, so the highest slot touched is [kStride + 7*kStride] = [128].
  // An 8*kStride (128-slot) buffer put XCD 7's arrival one past the end, so
  // that XCD's arrival never counted and its group never reached its
  // threshold -- the split-arm hang, in both partition modes.
  size_t const sz = (size_t)16 * kStride * sizeof(unsigned long long);
  unsigned long long *span, *ow, *oh;
  HIP_OK(hipMalloc(&span, sz));          // spanning: what fleet uses
  HIP_OK(hipMalloc(&ow, kBlocks * sizeof(unsigned long long)));
  HIP_OK(hipMalloc(&oh, kBlocks * sizeof(unsigned long long)));
  unsigned long long *aid[2] = {nullptr, nullptr};
  for (int i = 0; i < 2; i++) {
    aid[i] = static_cast<unsigned long long *>(rt_alloc_aid(sz, i));
  }
  bool const ok = aid[0] && aid[1];
  printf("  AID buffers: %s\n", ok ? "allocated" : "UNAVAILABLE (NPS1/stock)");
  if (!ok) { HIP_OK(hipMalloc(&aid[0], sz)); HIP_OK(hipMalloc(&aid[1], sz)); }

  auto run = [&](char const *label, unsigned long long *buf, unsigned mask,
                 int split = 0) {
    int n = 0;
    for (int i = 0; i < 8; i++) { n += (mask >> i) & 1u; }
    Args a{};
    a.cnt = buf; a.loc = buf + kStride; // counter at [0], arrivals after it
    a.xcd_mask = mask; a.n_active = n;
    a.out_wait = ow; a.out_hits = oh;
    a.split = split;
    for (int i = 0; i < 2; i++) {
      a.cnt_aid[i] = aid[i];
      a.loc_aid[i] = aid[i] + kStride;
    }
    if (split) {
      (void)hipMemset(aid[0], 0, sz);
      (void)hipMemset(aid[1], 0, sz);
    }
    (void)hipMemset(buf, 0, sz);
    (void)hipMemset(ow, 0, kBlocks * sizeof(unsigned long long));
    (void)hipMemset(oh, 0, kBlocks * sizeof(unsigned long long));
    hipLaunchKernelGGL(k_one, dim3(kBlocks), dim3(64), 0, 0, a, iters);
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) {
      printf("  %-32s FAILED %s\n", label, hipGetErrorString(e)); return;
    }
    std::vector<unsigned long long> w(kBlocks), h(kBlocks);
    (void)hipMemcpy(w.data(), ow, kBlocks*sizeof(unsigned long long), hipMemcpyDeviceToHost);
    (void)hipMemcpy(h.data(), oh, kBlocks*sizeof(unsigned long long), hipMemcpyDeviceToHost);
    std::vector<double> ns; double hh = 0; int cnt = 0;
    for (int i = 0; i < kBlocks; i++) {
      if (w[i] == 0) { continue; }
      ns.push_back((double)w[i] * 10.0 / iters);
      hh += (double)h[i] / iters; cnt++;
    }
    if (ns.empty()) { printf("  %-32s no active blocks\n", label); return; }
    std::sort(ns.begin(), ns.end());
    printf("  %-32s wait_med %8.0f ns  polls/iter %6.1f  (%d blocks, %d XCDs)\n",
           label, ns[ns.size()/2], hh / cnt, cnt, n);
    if (split) {
      // A stuck level names itself: the group counter should reach 4*iters and
      // each XCD's arrival slot 23*iters.
      std::vector<unsigned long long> d(16 * kStride);
      for (int g = 0; g < 2; g++) {
        (void)hipMemcpy(d.data(), aid[g], sz, hipMemcpyDeviceToHost);
        printf("      AID%d cnt=%llu (want %d)  arrivals:", g,
               (unsigned long long)d[0], 4 * iters);
        for (int x = 0; x < 8; x++) {
          printf(" %llu", (unsigned long long)d[kStride + x * kStride]);
        }
        printf("  (want %d each for its own 4 XCDs)\n", kTiles * iters);
      }
    }
  };

  printf("\n");
  run("8 XCDs, spanning line", span, 0xFFu);
  if (ok) {
    run("4 XCDs 0-3, line in AID0 (local)", aid[0], 0x0Fu);
    run("4 XCDs 0-3, line in AID1 (far)",   aid[1], 0x0Fu);
    run("4 XCDs 4-7, line in AID1 (local)", aid[1], 0xF0u);
    run("4 XCDs 4-7, line in AID0 (far)",   aid[0], 0xF0u);
    printf("\n  --- SPLIT: all 8 XCDs, two independent per-AID groups ---\n");
    run("8 XCDs, split per AID (local)", aid[0], 0xFFu, 1);
  }
  printf("\n  If the two local rows match NPS1 and the far rows are slow, the\n"
         "  AID boundary is the whole story. If the local rows are STILL slow,\n"
         "  NPS2 costs something without any crossing at all.\n");
  return 0;
}
