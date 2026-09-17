// Fleet's barrier with the release path switchable: shared-NC flags vs
// AID-replicated flags.
//
// fleetbar.cpp established that fleet's barrier shape costs 1747 ns in NPS1
// and 3343 ns in NPS2 at ZERO skew, with the +1600 ns delta constant at every
// skew level -- protocol cost, not arrival spread.
//
// That harness used plain hipMalloc flags, i.e. fleet WITHOUT
// MPK_AID_SPLIT_FLAGS. Fleet has it on, so: how much of the 1600 ns does
// replication recover, and what is left?
//
//   shared   every flag on one spanning (NC) allocation
//   aid      release flags replicated per AID, published with two
//            write-through stores, each tile polling the copy homed in its own
//            range -- the DUAL policy that measured 918 ns flat to 256 pollers
//
// The level-2 global counter stays shared in both arms: eight XCDs must agree
// on one value, so it is the genuinely all-to-all word and cannot be
// replicated.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#if !defined(__HIP_DEVICE_COMPILE__)
#include "aid_local.h"
static void *rt_alloc_aid(size_t bytes, int aid) {
  return mirage::aid::alloc_in_aid(bytes, aid, 0ull);
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

__device__ __forceinline__ int my_xcd_hw() {
  unsigned int id;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(id));
  return (int)(id & (kXcds - 1));
}
__device__ __forceinline__ void st_wt(void *a, unsigned v) {
  asm volatile("global_store_dword %0, %1, off sc0 sc1"
               : : "v"(a), "v"(v) : "memory");
}
__device__ __forceinline__ unsigned ld_sc(void const *a) {
  unsigned v;
  asm volatile("global_load_dword %0, %1, off sc0 sc1\n s_waitcnt vmcnt(0)"
               : "=v"(v) : "v"(a) : "memory");
  return v;
}

struct Bar {
  int *shared;
  int *rel0;
  int *rel1;
  int dual; // 1 = publish into both replicas, 0 = single shared line
};

__global__ __launch_bounds__(256) void k_bar(Bar b, int iters,
                                             unsigned long long *out) {
  int const xcd = my_xcd_hw();
  int const aid = xcd >> 2;
  int const tid = threadIdx.x;
  int *const mine = (aid == 0) ? b.rel0 : b.rel1;
  unsigned long long acc = 0;

  for (int it = 1; it <= iters; it++) {
    unsigned long long const tb = __builtin_amdgcn_s_memrealtime();
    __syncthreads();
    if (tid == 0) {
      int const lp = atomicAdd(&b.shared[(28 + xcd) * kStride], 1);
      if ((lp % kTiles) == kTiles - 1) {
        int const gp = atomicAdd(&b.shared[8 * kStride], 1);
        if ((gp % kXcds) == kXcds - 1) {
          // Write-through stores cross the AID boundary; an atomic would not.
          for (int x = 0; x < kXcds; x++) {
            st_wt(&b.rel0[x * kStride], (unsigned)it);
            if (b.dual) {
              st_wt(&b.rel1[x * kStride], (unsigned)it);
            }
          }
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        }
      }
      while (ld_sc(&mine[xcd * kStride]) < (unsigned)it) {
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
  char const *mode = "?";
  printf("=== fleet barrier, release-path A/B: %s, %d blocks, %d iters ===\n",
         prop.gcnArchName, kBlocks, iters);

  size_t const sz = (size_t)64 * kStride * sizeof(int);
  int *shared;
  HIP_OK(hipMalloc(&shared, sz));
  unsigned long long *out;
  HIP_OK(hipMalloc(&out, kBlocks * sizeof(unsigned long long)));

  int *a0 = static_cast<int *>(rt_alloc_aid(sz, 0));
  int *a1 = static_cast<int *>(rt_alloc_aid(sz, 1));
  bool const aid_ok = (a0 != nullptr && a1 != nullptr);
  printf("  AID replicas: %s\n",
         aid_ok ? "allocated (AID0 + AID1)"
                : "UNAVAILABLE -- NPS1, or stock driver");
  (void)mode;

  auto run = [&](char const *label, int *r0, int *r1, int dual) {
    Bar b{};
    b.shared = shared; b.rel0 = r0; b.rel1 = r1; b.dual = dual;
    (void)hipMemset(shared, 0, sz);
    (void)hipMemset(r0, 0, sz);
    if (r1 != r0) { (void)hipMemset(r1, 0, sz); }
    (void)hipMemset(out, 0, kBlocks * sizeof(unsigned long long));
    hipLaunchKernelGGL(k_bar, dim3(kBlocks), dim3(256), 0, 0, b, iters, out);
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) {
      printf("  %-24s FAILED: %s\n", label, hipGetErrorString(e));
      return;
    }
    std::vector<unsigned long long> h(kBlocks);
    (void)hipMemcpy(h.data(), out, kBlocks * sizeof(unsigned long long),
                    hipMemcpyDeviceToHost);
    std::vector<double> ns;
    for (auto v : h) { ns.push_back((double)v * 10.0 / (double)iters); }
    std::sort(ns.begin(), ns.end());
    printf("  %-24s median %6.0f   p95 %6.0f   max %6.0f ns\n",
           label, ns[ns.size()/2], ns[(size_t)(ns.size()*0.95)], ns.back());
  };

  printf("\n  reference: NPS1 1747 ns, NPS2 3343 ns (both shared-NC flags)\n\n");
  run("shared NC flags", shared, shared, 0);
  if (aid_ok) {
    run("AID-replicated (dual)", a0, a1, 1);
  }
  printf("\n  A large drop => fleet's MPK_AID_SPLIT_FLAGS already captures\n"
         "  most of the +1600 ns and the residue is the shared level-2\n"
         "  counter. Little drop => the release path still has headroom.\n");
  return 0;
}

