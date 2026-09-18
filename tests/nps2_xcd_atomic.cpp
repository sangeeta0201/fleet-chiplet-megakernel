// Cross-XCD *atomic* rendezvous under SPX+NPS2.
//
// Forcing the gates to `nt` (MPK_SYS_POLL_LOAD=0) did not fix the megakernel
// hang, so the broken primitive is not the flag load. Fleet's barrier is an
// atomic RMW rendezvous: every XCD does an atomic increment and then waits for
// the total. If an atomic on coarse-grained hipMalloc memory resolves to the
// *local* AID's coherence point, arrivals from XCDs on opposite AIDs land in
// two separate copies and the total never reaches the threshold -- which is
// exactly the observed "observed=5 expected=6 short_by=1".
//
// This reproduces that rendezvous per allocation type and reports, per XCD,
// the highest total that XCD ever observed.
//
// hipcc -O3 --offload-arch=gfx950 nps2_xcd_atomic.cpp -o nps2_xcd_atomic
#include <hip/hip_runtime.h>
#include <cstdio>

#define HIP_TRY(x)                                                             \
  do {                                                                         \
    if ((x) != hipSuccess) {                                                    \
      return -1;                                                                \
    }                                                                          \
  } while (0)

__device__ __forceinline__ int xcc_id() {
  int x;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 4)" : "=s"(x));
  return x;
}

// What atom_add_release_gpu_s32 emits: device/system-scope atomic add.
// sc0 on an atomic requests the pre-op value, so this is the returning form.
__device__ __forceinline__ int atom_add_sys(int *p, int v) {
  int old;
  asm volatile("global_atomic_add %0, %1, %2, off sc0 sc1\n\t"
               "s_waitcnt vmcnt(0)"
               : "=v"(old)
               : "v"(p), "v"(v)
               : "memory");
  return old;
}

__device__ __forceinline__ int ld_sys(int const *p) {
  int v;
  asm volatile("global_load_dword %0, %1, off sc0 sc1\n\t"
               "s_waitcnt vmcnt(0)"
               : "=v"(v)
               : "v"(p)
               : "memory");
  return v;
}

__device__ __forceinline__ int ld_nt(int const *p) {
  int v;
  asm volatile("global_load_dword %0, %1, off nt\n\t"
               "s_waitcnt vmcnt(0)"
               : "=v"(v)
               : "v"(p)
               : "memory");
  return v;
}

// atomic_mode: 0 = sc0 sc1 inline asm, 1 = plain HIP atomicAdd
// read_mode:   0 = sc0 sc1, 1 = nt
__global__ void rendezvous(int *counter,
                           int *seen,
                           int *maxobs,
                           int *stuck,
                           int expect,
                           int spin_limit,
                           int atomic_mode,
                           int read_mode) {
  if (threadIdx.x != 0) {
    return;
  }
  int x = xcc_id();
  if (x < 0 || x >= 8) {
    return;
  }
  atomicExch(&seen[x], 1);

  if (atomic_mode == 0) {
    (void)atom_add_sys(counter, 1);
  } else {
    atomicAdd(counter, 1);
  }
  __threadfence();

  int n = 0;
  int v = read_mode ? ld_nt(counter) : ld_sys(counter);
  while (v < expect && n < spin_limit) {
    n++;
    __builtin_amdgcn_s_sleep(1);
    v = read_mode ? ld_nt(counter) : ld_sys(counter);
  }
  atomicMax(&maxobs[x], v);
  if (v < expect) {
    atomicAdd(&stuck[x], 1);
  }
}

struct Alloc {
  char const *name;
  int kind; // 0 hipMalloc, 1 uncached, 2 finegrained
};

static int run_case(Alloc const &a,
                    int atomic_mode,
                    int read_mode,
                    int nblocks,
                    int *out_bad) {
  int *counter = nullptr;
  size_t const bytes = 4096;
  switch (a.kind) {
  case 0:
    HIP_TRY(hipMalloc(&counter, bytes));
    break;
  case 1:
    HIP_TRY(hipExtMallocWithFlags(
        (void **)&counter, bytes, hipDeviceMallocUncached));
    break;
  case 2:
    HIP_TRY(hipExtMallocWithFlags(
        (void **)&counter, bytes, hipDeviceMallocFinegrained));
    break;
  }

  int *d_seen, *d_max, *d_stuck;
  HIP_TRY(hipMalloc(&d_seen, 8 * sizeof(int)));
  HIP_TRY(hipMalloc(&d_max, 8 * sizeof(int)));
  HIP_TRY(hipMalloc(&d_stuck, 8 * sizeof(int)));
  HIP_TRY(hipMemset(counter, 0, bytes));
  HIP_TRY(hipMemset(d_seen, 0, 8 * sizeof(int)));
  HIP_TRY(hipMemset(d_max, 0, 8 * sizeof(int)));
  HIP_TRY(hipMemset(d_stuck, 0, 8 * sizeof(int)));
  HIP_TRY(hipDeviceSynchronize());

  rendezvous<<<nblocks, 64>>>(counter,
                              d_seen,
                              d_max,
                              d_stuck,
                              nblocks,
                              300000,
                              atomic_mode,
                              read_mode);
  HIP_TRY(hipDeviceSynchronize());

  int seen[8], mx[8], stuck[8], host_total = 0;
  HIP_TRY(hipMemcpy(seen, d_seen, sizeof(seen), hipMemcpyDeviceToHost));
  HIP_TRY(hipMemcpy(mx, d_max, sizeof(mx), hipMemcpyDeviceToHost));
  HIP_TRY(hipMemcpy(stuck, d_stuck, sizeof(stuck), hipMemcpyDeviceToHost));
  HIP_TRY(hipMemcpy(&host_total, counter, sizeof(int), hipMemcpyDeviceToHost));

  int bad = 0;
  printf("  %-12s atom=%-6s read=%-6s host_total=%3d/%3d  ",
         a.name,
         atomic_mode ? "hip" : "sc0sc1",
         read_mode ? "nt" : "sc0sc1",
         host_total,
         nblocks);
  for (int i = 0; i < 8; i++) {
    if (!seen[i]) {
      printf("x%d:- ", i);
    } else if (stuck[i]) {
      printf("x%d:%d ", i, mx[i]);
      bad++;
    } else {
      printf("x%d:ok ", i);
    }
  }
  printf(" %s\n", bad ? "<-- RENDEZVOUS FAILED" : "OK");

  (void)hipFree(d_seen);
  (void)hipFree(d_max);
  (void)hipFree(d_stuck);
  (void)hipFree(counter);
  *out_bad = bad;
  return 0;
}

int main() {
  hipDeviceProp_t p;
  if (hipGetDeviceProperties(&p, 0) != hipSuccess) {
    printf("cannot query device\n");
    return 1;
  }
  printf("CUs=%d arch=%s\n", p.multiProcessorCount, p.gcnArchName);
  printf("each block: 1 atomic increment, then wait for the full total.\n");
  printf("per-XCD column shows the highest total that XCD ever observed.\n\n");

  Alloc cases[] = {{"hipMalloc", 0}, {"uncached", 1}, {"finegrained", 2}};
  // 8 blocks = one per XCD, the shape of fleet's cross-XCD barrier.
  int const nblocks = 8;
  int total_bad = 0;
  for (auto &a : cases) {
    for (int am = 0; am <= 1; am++) {
      for (int rm = 0; rm <= 1; rm++) {
        int bad = 0;
        if (run_case(a, am, rm, nblocks, &bad) != 0) {
          printf("  %-12s SKIPPED (alloc unsupported)\n", a.name);
          continue;
        }
        total_bad += bad;
      }
    }
  }
  printf("\nfailed XCD rendezvous count: %d\n", total_bad);
  return 0;
}
