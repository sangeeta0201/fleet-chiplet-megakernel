// XCD-local flag rendezvous: does a plain global store from one CU become
// visible to an `nt` (or `sc0`) load from another CU on the same XCD?
//
// MPK_ROUTER_XSPLIT and MPK_ROUTER_LL exchange data between the router tiles
// of one XCD through that XCD's L2: plain stores, polled with loads that must
// miss the vL1. This checks the mechanism with nothing else around it.
//
// One block per CU (the dynamic LDS request forbids a second), every block
// resident at once. Each block reads its XCC_ID and takes a slot on that XCD
// with a device-scope atomic; the first GROUP blocks of every XCD then run
// EPOCHS rounds of: store flag[xcd][slot] = epoch, wait until all GROUP flags
// of the XCD reach epoch. A wait that exceeds SPIN_CAP is counted as a failure
// and abandoned, so a broken mechanism reports instead of hanging.
//
// Build: hipcc -O3 --offload-arch=gfx950 test_xcd_flag.hip -o xcd_flag
//        -DPOLL_SC0=1 polls with sc0 instead of nt (expected to FAIL: sc0
//        alone is Hit-LRU in the vL1).
#include <hip/hip_runtime.h>
#include <cstdio>

#ifndef POLL_SC0
#define POLL_SC0 0
#endif

constexpr int NXCD = 8;
constexpr int GROUP = 16;
constexpr int EPOCHS = 2000;
constexpr int SPIN_CAP = 1 << 20;

__device__ __forceinline__ int xcc_id() {
  int x;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(x));
  return x & 7;
}

__global__ void __launch_bounds__(256, 1)
    xcd_flag(int *slots, unsigned *flags, int *fails, int *maxspin) {
  extern __shared__ char lds_hog[];
  (void)lds_hog;
  __shared__ int s_slot;
  int const xcd = xcc_id();
  if (threadIdx.x == 0) {
    s_slot = atomicAdd(&slots[xcd], 1);
  }
  __syncthreads();
  int const slot = s_slot;
  if (slot >= GROUP) {
    return;
  }
  unsigned *const f = flags + xcd * 64;
  for (unsigned e = 1; e <= EPOCHS; e++) {
    if (threadIdx.x == 0) {
      asm volatile("global_store_dword %0, %1, off\n"
                   "s_waitcnt vmcnt(0)"
                   :
                   : "v"(f + slot), "v"(e)
                   : "memory");
    }
    if (threadIdx.x < 64) {
      int spins = 0;
      while (true) {
        unsigned seen = e;
        if (threadIdx.x < GROUP) {
#if POLL_SC0
          asm volatile("global_load_dword %0, %1, off sc0\n"
                       "s_waitcnt vmcnt(0)"
                       : "=v"(seen)
                       : "v"(f + threadIdx.x)
                       : "memory");
#else
          asm volatile("global_load_dword %0, %1, off nt\n"
                       "s_waitcnt vmcnt(0)"
                       : "=v"(seen)
                       : "v"(f + threadIdx.x)
                       : "memory");
#endif
        }
        if (__ballot(seen >= e) == ~0ull) {
          break;
        }
        if (++spins == SPIN_CAP) {
          if (threadIdx.x == 0) {
            atomicAdd(&fails[xcd], 1);
          }
          break;
        }
        __builtin_amdgcn_s_sleep(1);
      }
      if (threadIdx.x == 0) {
        atomicMax(&maxspin[xcd], spins);
      }
    }
    __syncthreads();
  }
}

int main() {
  int dev = 0;
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, dev);
  int const blocks = prop.multiProcessorCount;
  int *slots, *fails, *maxspin;
  unsigned *flags;
  hipMalloc(&slots, NXCD * sizeof(int));
  hipMalloc(&fails, NXCD * sizeof(int));
  hipMalloc(&maxspin, NXCD * sizeof(int));
  hipMalloc(&flags, NXCD * 64 * sizeof(unsigned));
  hipMemset(slots, 0, NXCD * sizeof(int));
  hipMemset(fails, 0, NXCD * sizeof(int));
  hipMemset(maxspin, 0, NXCD * sizeof(int));
  hipMemset(flags, 0, NXCD * 64 * sizeof(unsigned));
  size_t const lds = 100 * 1024;
  hipFuncSetAttribute((void const *)xcd_flag,
                      hipFuncAttributeMaxDynamicSharedMemorySize, (int)lds);
  hipLaunchKernelGGL(xcd_flag, dim3(blocks), dim3(256), lds, 0, slots, flags,
                     fails, maxspin);
  hipError_t err = hipDeviceSynchronize();
  int h_slots[NXCD], h_fails[NXCD], h_max[NXCD];
  hipMemcpy(h_slots, slots, sizeof(h_slots), hipMemcpyDeviceToHost);
  hipMemcpy(h_fails, fails, sizeof(h_fails), hipMemcpyDeviceToHost);
  hipMemcpy(h_max, maxspin, sizeof(h_max), hipMemcpyDeviceToHost);
  int total = 0;
  for (int x = 0; x < NXCD; x++) {
    printf("xcd %d: blocks=%d fails=%d max_spins=%d\n", x, h_slots[x],
           h_fails[x], h_max[x]);
    total += h_fails[x];
  }
  printf("%s (%s polls, %d epochs, err=%s)\n", total == 0 ? "PASS" : "FAIL",
         POLL_SC0 ? "sc0" : "nt", EPOCHS, hipGetErrorString(err));
  return total == 0 ? 0 : 1;
}
