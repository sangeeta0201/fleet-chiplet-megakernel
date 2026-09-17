// Stage 1 gate A: does aid_rt.h reproduce the Stage 0 primitive numbers?
//
// policy.cpp measured the raw policies in SPX/NPS2 at 256 concurrent workers:
//   barrier HIER  2904 ns   (FLAT 15515, null-policy baseline 3767)
// If the runtime's wrapped barrier is materially worse than that, the
// abstraction is costing something and must be fixed before any op is built
// on it. Timed on one block's clock only -- s_memrealtime is not assumed to
// be phase-aligned across XCDs.
#include "aid_rt.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

// aid_local.h's namespace is host-only, but HIP's device pass still parses
// main(), so a bare mirage::aid::... call in host code fails to resolve when
// compiling for gfx950. Wrap it in a symbol that exists in both passes.
#if !defined(__HIP_DEVICE_COMPILE__)
#include "aid_local.h"
static void *rt_alloc_aid(size_t bytes, int aid) {
  return mirage::aid::alloc_in_aid(bytes, aid, 0ull);
}
#else
static void *rt_alloc_aid(size_t, int) {
  return nullptr;
}
#endif

#define HIP_OK(x)                                                              \
  do {                                                                         \
    hipError_t e = (x);                                                        \
    if (e != hipSuccess) {                                                     \
      printf("HIP FAIL %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
      return 2;                                                                \
    }                                                                          \
  } while (0)

using namespace nps2;

// 100 MHz tick, so this is 0.2 s. Short on purpose: a deadlocked barrier
// should surface as a fast TIMED OUT, not as a sweep that appears to hang.
static constexpr unsigned long long kDeadline = 20ULL * 1000ULL * 1000ULL;

__global__ void k_barrier(BarrierSet b, int iters, unsigned long long *out) {
  unsigned long long t_acc = 0;
  int timed_out = 0;
  for (int it = 1; it <= iters; it++) {
    unsigned long long const t0 = rt_clock();
    bool const ok = b.arrive_and_wait(0, (unsigned)it, kDeadline);
    t_acc += rt_clock() - t0;
    if (!ok) {
      timed_out = 1;
      break; // a timed-out barrier leaves the epoch skewed; do not pile on
    }
    __syncthreads();
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    out[0] = t_acc;
    out[1] = (unsigned long long)timed_out;
  }
}

int main(int argc, char **argv) {
  int const iters = argc > 1 ? atoi(argv[1]) : 200;

  hipDeviceProp_t prop;
  HIP_OK(hipGetDeviceProperties(&prop, 0));
  printf("=== aid_rt self-test on %s, %d CUs ===\n", prop.gcnArchName,
         prop.multiProcessorCount);
  if (prop.multiProcessorCount < 200) {
    printf("  WARNING: %d CUs -- not an SPX partition (SPX is 256)\n",
           prop.multiProcessorCount);
  }

  // Each replica MUST be homed in its own AID. Dual-publish has no benefit
  // otherwise: on two spanning-NC buffers you pay two stores and every poller
  // still hits the same far line, which measured strictly WORSE than the flat
  // shared line (22385 ns at 256 blocks vs FLAT's 15515). Placement is not a
  // refinement of this protocol, it is the protocol.
  bool aid_backed = true;
  GateSet g{};
  g.n_gates = 64;
  size_t const gbytes = (size_t)g.n_gates * kGateStrideInts * sizeof(unsigned);
  for (int a = 0; a < kNumAids; a++) {
    g.rep[a] = static_cast<unsigned int *>(
        rt_alloc_aid(gbytes, a));
    if (g.rep[a] == nullptr) {
      aid_backed = false;
      HIP_OK(hipMalloc(&g.rep[a], gbytes));
    }
  }

  BarrierSet b{};
  b.release = g;
  size_t const cbytes = (size_t)kNumXcds * kGateStrideInts * sizeof(unsigned);
  for (int a = 0; a < kNumAids; a++) {
    b.xcd_count[a] = static_cast<unsigned int *>(
        rt_alloc_aid(cbytes, a));
    if (b.xcd_count[a] == nullptr) {
      aid_backed = false;
      HIP_OK(hipMalloc(&b.xcd_count[a], cbytes));
    }
  }
  // Genuinely shared: exactly 8 atomics per barrier land here, one per XCD,
  // which is far below the contention cliff. Spanning NC is fine for it.
  HIP_OK(hipMalloc(&b.global_count, kGateStrideInts * sizeof(unsigned)));
  printf("  replicas: %s\n", aid_backed
                                 ? "AID-local (alloc_in_aid)"
                                 : "FELL BACK to hipMalloc -- gate is invalid");

  unsigned long long *d_out;
  HIP_OK(hipMalloc(&d_out, 4 * sizeof(unsigned long long)));

  int const widths[] = {8, 16, 32, 64, 128, 192, 256};
  int const hier_ref[] = {1496, 1443, 1513, 1698, 2108, 2488, 2904};
  int const flat_ref[] = {1059, 1335, 2183, 4033, 7646, 12562, 15515};

  printf("\n  blocks   aid_rt ns/op   HIER ref   FLAT ref\n");
  for (int i = 0; i < 7; i++) {
    int const nblk = widths[i];
    // Probe the real distribution for this grid rather than assuming nblk/8.
    {
      unsigned int *d_cnt;
      HIP_OK(hipMalloc(&d_cnt, kNumXcds * sizeof(unsigned)));
      HIP_OK(hipMemset(d_cnt, 0, kNumXcds * sizeof(unsigned)));
      hipLaunchKernelGGL(k_probe_xcd, dim3(nblk), dim3(64), 0, 0, d_cnt);
      HIP_OK(hipDeviceSynchronize());
      unsigned int h_cnt[kNumXcds] = {};
      HIP_OK(hipMemcpy(h_cnt, d_cnt, sizeof h_cnt, hipMemcpyDeviceToHost));
      for (int x = 0; x < kNumXcds; x++) {
        b.n_expect[x] = (int)h_cnt[x];
      }
      HIP_OK(hipFree(d_cnt));
    }
    for (int a = 0; a < kNumAids; a++) {
      HIP_OK(hipMemset(g.rep[a], 0, gbytes));
      HIP_OK(hipMemset(b.xcd_count[a], 0, cbytes));
    }
    HIP_OK(hipMemset(b.global_count, 0, kGateStrideInts * sizeof(unsigned)));
    HIP_OK(hipMemset(d_out, 0, 4 * sizeof(unsigned long long)));

    hipLaunchKernelGGL(k_barrier, dim3(nblk), dim3(64), 0, 0, b, iters, d_out);
    hipError_t const le = hipDeviceSynchronize();
    if (le != hipSuccess) {
      printf("  %5d     launch/sync failed: %s\n", nblk,
             hipGetErrorString(le));
      continue;
    }

    unsigned long long h[4] = {};
    HIP_OK(hipMemcpy(h, d_out, sizeof h, hipMemcpyDeviceToHost));
    double const ns = (double)h[0] * 10.0 / (double)iters; // 100 MHz tick
    printf("  %5d     %9.0f      %5d      %5d%s\n", nblk, ns, hier_ref[i],
           flat_ref[i], h[1] ? "   TIMED OUT" : "");
  }

  printf("\n  PASS if the measured column tracks HIER (within ~20%%) and is\n"
         "  far below FLAT at 128+ blocks. A measured column that grows like\n"
         "  FLAT means the wrapper serialises and the design is not yet real.\n");
  return 0;
}

