// Minimal cross-XCD atomic test under SPX+NPS2.
//
// No device-side instrumentation: N blocks each perform exactly one atomic
// increment, the host reads the total. Anything less than N means increments
// are being lost -- i.e. the same address is not one coherent location across
// XCDs. Deliberately avoids recording results with the primitive under test,
// which is what made the previous version's numbers self-contradictory.
//
// hipcc -O3 --offload-arch=gfx950 nps2_atomic_min.cpp -o nps2_atomic_min
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void inc_once(int *c) {
  if (threadIdx.x == 0) {
    atomicAdd(c, 1);
  }
}

// Same op fleet's release path uses: device/system-scope atomic add.
__global__ void inc_once_sys(int *c) {
  if (threadIdx.x == 0) {
    int old;
    asm volatile("global_atomic_add %0, %1, %2, off sc0 sc1\n\t"
                 "s_waitcnt vmcnt(0)"
                 : "=v"(old)
                 : "v"(c), "v"(1)
                 : "memory");
  }
}

static char const *kindname(int k) {
  return k == 0 ? "hipMalloc" : (k == 1 ? "uncached" : "finegrained");
}

static int alloc_of(int kind, int **p, size_t bytes) {
  if (kind == 0) {
    return hipMalloc(p, bytes) == hipSuccess ? 0 : -1;
  }
  unsigned f = (kind == 1) ? hipDeviceMallocUncached : hipDeviceMallocFinegrained;
  return hipExtMallocWithFlags((void **)p, bytes, f) == hipSuccess ? 0 : -1;
}

int main() {
  hipDeviceProp_t p;
  (void)hipGetDeviceProperties(&p, 0);
  printf("CUs=%d arch=%s\n\n", p.multiProcessorCount, p.gcnArchName);
  printf("%-12s %-8s %6s %8s %8s\n", "alloc", "atomic", "blocks", "expect", "got");

  int bad = 0;
  for (int kind = 0; kind <= 2; kind++) {
    for (int sys = 0; sys <= 1; sys++) {
      for (int nb : {8, 64, 256}) {
        int *c = nullptr;
        if (alloc_of(kind, &c, 4096) != 0) {
          printf("%-12s %-8s   (alloc unsupported)\n", kindname(kind), sys ? "sc0sc1" : "hip");
          continue;
        }
        (void)hipMemset(c, 0, 4096);
        (void)hipDeviceSynchronize();
        if (sys) {
          inc_once_sys<<<nb, 64>>>(c);
        } else {
          inc_once<<<nb, 64>>>(c);
        }
        (void)hipDeviceSynchronize();
        int got = -1;
        (void)hipMemcpy(&got, c, sizeof(int), hipMemcpyDeviceToHost);
        printf("%-12s %-8s %6d %8d %8d %s\n",
               kindname(kind),
               sys ? "sc0sc1" : "hip",
               nb,
               nb,
               got,
               got == nb ? "" : "<-- LOST INCREMENTS");
        if (got != nb) {
          bad++;
        }
        (void)hipFree(c);
      }
    }
  }
  printf("\nfailing configurations: %d\n", bad);
  return 0;
}
