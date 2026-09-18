// Standalone reproducer of fleet's MoE W13->W2 barrier at a reduced XCD count.
//
// Fleet iterates in ~10 minutes per attempt; this runs in seconds. It copies
// the exact structure:
//
//   * per-expert barrier region, MOE_BAR_LINE = 16 ints between slots,
//     slots 0..7 = per-XCD release flags, slot 8 = the arrival counter
//   * one expert per participating XCD (the 4-XCD map), 46 groups per expert
//   * 31 ranks per XCD sweeping 2 groups each (31 + 15 = 46 arrivals)
//   * arrival = atom_add on slot 8; the last arrival publishes release_val to
//     the flag slots; consumers then wait on a flag
//
// and reports, per expert: the arrival count, whether the release ran, and the
// value in every flag slot -- the same quantities fleet's instrumentation
// reported, but observable in one second instead of ten minutes.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define HIP_OK(x)                                                              \
  do {                                                                         \
    hipError_t _e = (x);                                                       \
    if (_e != hipSuccess) {                                                    \
      printf("HIP FAIL %s:%d %s\n", __FILE__, __LINE__,                        \
             hipGetErrorString(_e));                                           \
      return 2;                                                                \
    }                                                                          \
  } while (0)

static constexpr int MOE_BAR_LINE = 16;
static constexpr int MOE_BAR_COUNTER_SLOT = 8;
static constexpr int MOE_BAR_SLOTS = 16;
static constexpr int MOE_BAR_STRIDE = MOE_BAR_SLOTS * MOE_BAR_LINE;
static constexpr int W13_TILES = 46;
static constexpr int RANKS = 31;   // workers per XCD that fleet actually has
static constexpr int TILES = 23;   // blocks per XCD in the probe grid
static constexpr int NBLOCKS = 8 * RANKS;

__device__ __forceinline__ int my_xcd() {
  unsigned id;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(id));
  return (int)(id & 7u);
}
__device__ __forceinline__ int ld_sys(int const *p) {
  int v;
  asm volatile("global_load_dword %0, %1, off sc0 sc1\n s_waitcnt vmcnt(0)"
               : "=v"(v) : "v"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_wt(int *p, int v) {
  asm volatile("global_store_dword %0, %1, off sc0 sc1" ::"v"(p), "v"(v)
               : "memory");
}

struct Args {
  int *bar;
  int nxcd;        // participating XCDs
  int pub_mode;    // 0 = st_wt, 1 = atomicMax
  int slot_mode;   // 0 = consumer reads slot xcd_id, 1 = reads slot 0
  int layers;
  int *out_obs;    // per block: what the consumer last observed
  int *rank_ctr;   // per-XCD rank counter, claimed atomically as fleet does
};

__global__ __launch_bounds__(64) void k_moebar(Args a) {
  int const xcd = my_xcd();
  if (xcd >= a.nxcd) {
    return; // outside the subset: takes no part, as in fleet
  }
  // Claim a dense rank within this XCD, exactly as fleet does. Deriving it
  // from blockIdx is wrong: block i lands on XCD i%8, so blockIdx/nxcd skips
  // ranks once nxcd < 8.
  __shared__ int s_rank;
  if (threadIdx.x == 0) {
    s_rank = atomicAdd(&a.rank_ctr[xcd], 1);
  }
  __syncthreads();
  int const rank = s_rank;
  if (rank >= RANKS) {
    return;
  }
  int const expert = xcd;               // one expert per XCD at 4 XCDs
  int const base = expert * MOE_BAR_STRIDE;
  int const sweeps = (W13_TILES + RANKS - 1) / RANKS; // 2

  for (int layer = 1; layer <= a.layers; layer++) {
    int const expected = layer;
    // ---- W13 phase: every group of this expert arrives ----
    for (int s = 0; s < sweeps; s++) {
      int const grp = rank + s * RANKS;
      if (grp >= W13_TILES) {
        continue;
      }
      if (threadIdx.x == 0) {
        int prev = atomicAdd(&a.bar[base + MOE_BAR_COUNTER_SLOT * MOE_BAR_LINE],
                             1);
        if ((prev % W13_TILES) == W13_TILES - 1) {
          atomicAdd(&a.bar[base + 10 * MOE_BAR_LINE], 1); // witness
          // Epoch from the arrival count, not from this block's layer.
          int const epoch = (prev + 1) / W13_TILES;
          for (int x = 0; x < 8; x++) {
            if (a.pub_mode == 0) {
              st_wt(&a.bar[base + x * MOE_BAR_LINE], epoch);
            } else {
              atomicMax(&a.bar[base + x * MOE_BAR_LINE], epoch);
            }
          }
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        }
      }
    }
    // ---- W2 phase: wait for the release ----
    if (threadIdx.x == 0) {
      int const slot = (a.slot_mode == 0) ? xcd : 0;
      int obs = 0;
      unsigned spin = 0;
      while ((obs = ld_sys(&a.bar[base + slot * MOE_BAR_LINE])) < expected) {
        if (++spin > 8000000u) {
          break; // report instead of hanging
        }
        __builtin_amdgcn_s_sleep(1);
      }
      a.out_obs[blockIdx.x] = obs;
    }
    __syncthreads();
  }
}

int main(int argc, char **argv) {
  int const nxcd = argc > 1 ? atoi(argv[1]) : 4;
  int const layers = argc > 2 ? atoi(argv[2]) : 3;
  hipDeviceProp_t p{};
  (void)hipGetDeviceProperties(&p, 0);
  printf("CUs=%d  nxcd=%d  layers=%d  (46 groups/expert, %d ranks/XCD)\n",
         p.multiProcessorCount, nxcd, layers, RANKS);

  size_t const bar_ints = (size_t)8 * MOE_BAR_STRIDE;
  int *bar = nullptr, *obs = nullptr, *rctr = nullptr;
  HIP_OK(hipMalloc(&bar, bar_ints * sizeof(int)));
  HIP_OK(hipMalloc(&obs, NBLOCKS * sizeof(int)));
  HIP_OK(hipMalloc(&rctr, 8 * sizeof(int)));

  for (int pub = 0; pub < 2; pub++) {
    for (int sm = 0; sm < 2; sm++) {
      (void)hipMemset(bar, 0, bar_ints * sizeof(int));
      (void)hipMemset(obs, 0, NBLOCKS * sizeof(int));
      (void)hipMemset(rctr, 0, 8 * sizeof(int));
      Args a{};
      a.bar = bar; a.nxcd = nxcd; a.pub_mode = pub; a.slot_mode = sm;
      a.layers = layers; a.out_obs = obs; a.rank_ctr = rctr;
      hipLaunchKernelGGL(k_moebar, dim3(NBLOCKS), dim3(64), 0, 0, a);
      hipError_t e = hipDeviceSynchronize();
      printf("\n--- publish=%-9s consumer reads=%-9s -> %s\n",
             pub ? "atomicMax" : "st_wt", sm ? "slot 0" : "slot xcd",
             e == hipSuccess ? "completed" : hipGetErrorString(e));
      std::vector<int> h(bar_ints);
      (void)hipMemcpy(h.data(), bar, bar_ints * sizeof(int),
                      hipMemcpyDeviceToHost);
      for (int x = 0; x < nxcd; x++) {
        int b = x * MOE_BAR_STRIDE;
        printf("    expert %d: counter=%-4d witness=%-3d slots=", x,
               h[b + MOE_BAR_COUNTER_SLOT * MOE_BAR_LINE],
               h[b + 10 * MOE_BAR_LINE]);
        for (int s = 0; s < 8; s++) { printf("%d,", h[b + s * MOE_BAR_LINE]); }
        printf("   (want counter=%d, slots=%d)\n", W13_TILES * layers, layers);
      }
      std::vector<int> ho(NBLOCKS);
      (void)hipMemcpy(ho.data(), obs, NBLOCKS * sizeof(int),
                      hipMemcpyDeviceToHost);
      int stuck = 0;
      for (int i = 0; i < NBLOCKS; i++) {
        if (ho[i] != 0 && ho[i] < layers) { stuck++; }
      }
      printf("    blocks that never saw the last release: %d\n", stuck);
    }
  }
  return 0;
}
