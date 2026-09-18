// Cross-XCD flag visibility under SPX+NPS2.
//
// The megakernel deadlocks in NPS2 but not NPS1, always short by exactly one
// XCD's arrival, with every gate already polling `sc0 sc1` (MPK_SYS_POLL_LOAD
// defaults to 2). That rules out a stale poll and points at the flag's
// backing memory having no single coherence point across the two AIDs.
//
// Block 0 publishes a value; every block polls it and reports per XCD.
// Repeated per allocation type, so the answer is "which allocator is coherent
// across AIDs", not merely "is it broken".
//
// Enough blocks are launched to cover all 8 XCDs, and coverage is reported
// explicitly: a "no block landed here" XCD must not be read as a failure.
//
// hipcc -O3 --offload-arch=gfx950 nps2_xcd_visibility.cpp -o nps2_xcd_visibility
#include <hip/hip_runtime.h>
#include <cstdio>

#define HIP_TRY(x)                                                             \
  do {                                                                         \
    hipError_t e = (x);                                                        \
    if (e != hipSuccess) {                                                     \
      return -1;                                                               \
    }                                                                          \
  } while (0)

__device__ __forceinline__ int xcc_id() {
  int x;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 4)" : "=s"(x));
  return x;
}

// global_load_dword ... sc0 sc1 -> "Coherent Cache Bypass (always)" per the
// CDNA4 load cache-control table. Same instruction MPK_LD_GATE emits.
__device__ __forceinline__ int ld_sys(int const *p) {
  int v;
  asm volatile("global_load_dword %0, %1, off sc0 sc1\n\t"
               "s_waitcnt vmcnt(0)"
               : "=v"(v)
               : "v"(p)
               : "memory");
  return v;
}

// nt only: L1 Miss Evict but L2 "Hit Stream" -- still hits L2.
__device__ __forceinline__ int ld_nt(int const *p) {
  int v;
  asm volatile("global_load_dword %0, %1, off nt\n\t"
               "s_waitcnt vmcnt(0)"
               : "=v"(v)
               : "v"(p)
               : "memory");
  return v;
}

__device__ __forceinline__ void st_wt(int *p, int v) {
  asm volatile("global_store_dword %0, %1, off sc0 sc1\n\t"
               "s_waitcnt vmcnt(0)" ::"v"(p),
               "v"(v)
               : "memory");
}

// seen[x]   : a block ran on XCD x (coverage)
// stuck[x]  : blocks on XCD x that timed out
// lastval[x]: last value observed on XCD x
__global__ void publish_observe(int *flag,
                                int *seen,
                                int *stuck,
                                int *lastval,
                                int spin_limit,
                                int expect,
                                int use_nt) {
  if (threadIdx.x != 0) {
    return;
  }
  int x = xcc_id();
  if (x < 0 || x >= 8) {
    return;
  }
  atomicExch(&seen[x], 1);

  if (blockIdx.x == 0) {
    // Let every other XCD fault the line into its own L2 first, so a stale
    // line has a chance to manifest instead of the publisher winning the race.
    for (int i = 0; i < 50000; i++) {
      __builtin_amdgcn_s_sleep(1);
    }
    st_wt(flag, expect);
    __threadfence();
    atomicExch(&lastval[x], expect);
    return; // publisher does not poll
  }

  int n = 0;
  int v = use_nt ? ld_nt(flag) : ld_sys(flag);
  while (v != expect && n < spin_limit) {
    n++;
    __builtin_amdgcn_s_sleep(1);
    v = use_nt ? ld_nt(flag) : ld_sys(flag);
  }
  atomicExch(&lastval[x], v);
  if (v != expect) {
    atomicAdd(&stuck[x], 1);
  }
}

struct Alloc {
  char const *name;
  int kind; // 0 hipMalloc, 1 uncached, 2 finegrained, 3 hostMalloc, 4 managed
};

static int run_case(Alloc const &a, int use_nt, int *out_bad) {
  int *flag = nullptr;
  void *host_backing = nullptr;
  size_t const bytes = 4096;

  switch (a.kind) {
  case 0:
    HIP_TRY(hipMalloc(&flag, bytes));
    break;
  case 1:
    HIP_TRY(hipExtMallocWithFlags(
        (void **)&flag, bytes, hipDeviceMallocUncached));
    break;
  case 2:
    HIP_TRY(hipExtMallocWithFlags(
        (void **)&flag, bytes, hipDeviceMallocFinegrained));
    break;
  case 3:
    HIP_TRY(hipHostMalloc(&host_backing, bytes, hipHostMallocMapped));
    HIP_TRY(hipHostGetDevicePointer((void **)&flag, host_backing, 0));
    break;
  case 4:
    HIP_TRY(hipMallocManaged((void **)&flag, bytes));
    break;
  }

  int *d_seen, *d_stuck, *d_last;
  HIP_TRY(hipMalloc(&d_seen, 8 * sizeof(int)));
  HIP_TRY(hipMalloc(&d_stuck, 8 * sizeof(int)));
  HIP_TRY(hipMalloc(&d_last, 8 * sizeof(int)));
  HIP_TRY(hipMemset(flag, 0, bytes));
  HIP_TRY(hipMemset(d_seen, 0, 8 * sizeof(int)));
  HIP_TRY(hipMemset(d_stuck, 0, 8 * sizeof(int)));
  HIP_TRY(hipMemset(d_last, 0, 8 * sizeof(int)));
  HIP_TRY(hipDeviceSynchronize());

  int const expect = 42;
  // 256 blocks so every XCD is covered many times over.
  publish_observe<<<256, 64>>>(
      flag, d_seen, d_stuck, d_last, 400000, expect, use_nt);
  HIP_TRY(hipDeviceSynchronize());

  int seen[8], stuck[8], last[8];
  HIP_TRY(hipMemcpy(seen, d_seen, sizeof(seen), hipMemcpyDeviceToHost));
  HIP_TRY(hipMemcpy(stuck, d_stuck, sizeof(stuck), hipMemcpyDeviceToHost));
  HIP_TRY(hipMemcpy(last, d_last, sizeof(last), hipMemcpyDeviceToHost));

  int bad = 0, uncovered = 0;
  printf("  %-12s poll=%-6s ", a.name, use_nt ? "nt" : "sc0sc1");
  for (int i = 0; i < 8; i++) {
    if (!seen[i]) {
      printf("x%d:-      ", i);
      uncovered++;
    } else if (stuck[i]) {
      printf("x%d:STUCK(%d,saw=%d) ", i, stuck[i], last[i]);
      bad++;
    } else {
      printf("x%d:ok    ", i);
    }
  }
  printf(" %s%s\n",
         bad ? "<-- NOT COHERENT" : "COHERENT",
         uncovered ? " (some XCDs uncovered)" : "");

  (void)hipFree(d_seen);
  (void)hipFree(d_stuck);
  (void)hipFree(d_last);
  if (a.kind == 3) {
    (void)hipHostFree(host_backing);
  } else {
    (void)hipFree(flag);
  }
  *out_bad = bad;
  return 0;
}

int main() {
  hipDeviceProp_t p;
  if (hipGetDeviceProperties(&p, 0) != hipSuccess) {
    printf("cannot query device\n");
    return 1;
  }
  printf("device CUs=%d arch=%s\n", p.multiProcessorCount, p.gcnArchName);
  printf("publisher: block 0 (st_wt sc0 sc1). pollers: all other blocks.\n\n");

  Alloc cases[] = {{"hipMalloc", 0},
                   {"uncached", 1},
                   {"finegrained", 2},
                   {"hostMalloc", 3},
                   {"managed", 4}};

  int total_bad = 0;
  for (auto &a : cases) {
    for (int use_nt = 0; use_nt <= 1; use_nt++) {
      int bad = 0;
      if (run_case(a, use_nt, &bad) != 0) {
        printf("  %-12s poll=%-6s SKIPPED (alloc/API unsupported)\n",
               a.name,
               use_nt ? "nt" : "sc0sc1");
        continue;
      }
      total_bad += bad;
    }
  }
  printf("\nXCDs that never observed the publish: %d\n", total_bad);
  return 0;
}
