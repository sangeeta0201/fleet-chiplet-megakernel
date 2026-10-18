// The TP W13 round in isolation (MPK_W13_HELP_PF, MPK_W13_SH_PRE in mpk_atoms.cuh), with every participating
// workgroup released together by a global start barrier: the W13 tiles in the megakernel start within ~0.2 us
// of each other, and a plain launch spreads them over ~2 us. NT tiles (NT / 8 per XCC by atomic ticket) stream B bytes each, D dwordx4 per thread in
// flight, as four contiguous per-wave runs (the K_HALF tile's shape). HELP > 0: H helpers per XCC pull the last
// HELP KB of every wave run of their XCC's tiles into L2 (sc0 loads into registers, summed into a sink) at the
// same release. Reports per-tile median / p90 and the round (last tile end - release), median over reps.
// usage: bench_w13_sync <NT> <tile_bytes> <D> [HELP_KB] [H]
// build: hipcc -O3 --offload-arch=gfx950 bench_w13_sync.hip -o bench_w13_sync
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

typedef int i32x4 __attribute__((ext_vector_type(4)));

template <int D>
__global__ void __launch_bounds__(256) wsync(char const *base, size_t expert_bytes, size_t tile_bytes, int per_xcc,
                                             int tiles_per_expert, int help_kb, int helpers, int const *experts,
                                             unsigned long long *t, int *sink, int *bar, int total) {
  unsigned x;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(x));
  int const xcd = (int)(x & 7);
  __shared__ int s_rank;
  if (threadIdx.x == 0) {
    s_rank = atomicAdd(&sink[1 + xcd], 1);
  }
  __syncthreads();
  int const rank = s_rank;
  bool const is_tile = rank < per_xcc;
  bool const is_help = !is_tile && help_kb > 0 && rank < per_xcc + helpers;
  if (!is_tile && !is_help) {
    return;
  }
  // global release: every participant arrives, then all spin until the count is complete
  if (threadIdx.x == 0) {
    __hip_atomic_fetch_add(bar, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    int spins = 0;
    while (__hip_atomic_load(bar, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < total && ++spins < (1 << 26)) {
    }
  }
  __syncthreads();
  unsigned long long const t0 = __builtin_amdgcn_s_memrealtime();
  int const wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
  size_t const run = tile_bytes / 4;  // one wave's contiguous run
  i32x4 acc = {0, 0, 0, 0};
  if (is_tile) {
    int const g = rank * 8 + xcd;
    char const *tb = base + (size_t)experts[g / tiles_per_expert] * expert_bytes +
                     (size_t)(g % tiles_per_expert) * tile_bytes;
    i32x4 const *p = reinterpret_cast<i32x4 const *>(tb + (size_t)wave * run);
    size_t const n = run / 16;
    for (size_t i = lane; i < n; i += 64 * D) {
      i32x4 v[D];
#pragma unroll
      for (int j = 0; j < D; j++) {
        size_t const k = i + (size_t)j * 64;
        v[j] = k < n ? __builtin_nontemporal_load(p + k) : i32x4{0, 0, 0, 0};
      }
#pragma unroll
      for (int j = 0; j < D; j++) {
        acc ^= v[j];
      }
    }
  } else {
    // helper h of this XCC: chunks (tile, wave run) dealt round-robin over helpers and their waves
    int const h = rank - per_xcc;
    size_t const tail = (size_t)help_kb * 1024;
    int const nchunks = per_xcc * 4;
    for (int c = h + wave * helpers; c < nchunks; c += 4 * helpers) {
      int const tt = c / 4, w = c % 4;
      int const g = tt * 8 + xcd;
      char const *tb = base + (size_t)experts[g / tiles_per_expert] * expert_bytes +
                       (size_t)(g % tiles_per_expert) * tile_bytes;
      i32x4 const *p = reinterpret_cast<i32x4 const *>(tb + (size_t)w * run + run - tail);
      size_t const n = tail / 16;
      for (size_t i = lane; i < n; i += 64 * 8) {
        i32x4 v[8];
#pragma unroll
        for (int j = 0; j < 8; j++) {
          size_t const k = i + (size_t)j * 64;
          v[j] = k < n ? p[k] : i32x4{0, 0, 0, 0};
        }
#pragma unroll
        for (int j = 0; j < 8; j++) {
          acc ^= v[j];
        }
      }
    }
  }
  __syncthreads();
  unsigned long long const t1 = __builtin_amdgcn_s_memrealtime();
  if (threadIdx.x == 0 && is_tile) {
    int const g = rank * 8 + xcd;
    t[g * 2] = t0;
    t[g * 2 + 1] = t1;
  }
  if (acc[0] == 0x12345678 && acc[1] == 0x9abcdef) {
    sink[0] = acc[2];
  }
}

int main(int argc, char **argv) {
  int const NT = argc > 1 ? atoi(argv[1]) : 144;
  size_t const tile_bytes = argc > 2 ? atol(argv[2]) : 32 * 3072;
  int const D = argc > 3 ? atoi(argv[3]) : 8;
  int const help_kb = argc > 4 ? atoi(argv[4]) : 0;
  int const helpers = argc > 5 ? atoi(argv[5]) : 0;
  int const per_xcc = NT / 8;
  int const NTT = per_xcc * 8;
  int const total = 8 * (per_xcc + (help_kb > 0 ? helpers : 0));
  int const tiles_per_expert = (NTT + 8) / 9;
  size_t const expert_bytes = (size_t)tiles_per_expert * tile_bytes;
  int const NE = 257;
  char *base;
  if (hipMalloc(&base, expert_bytes * NE) != hipSuccess) return 1;
  (void)hipMemset(base, 1, expert_bytes * NE);
  int *experts, *sink, *bar;
  unsigned long long *t;
  (void)hipMalloc(&experts, 16 * 4);
  (void)hipMalloc(&sink, 16 * 4);
  (void)hipMalloc(&bar, 4);
  (void)hipMalloc(&t, NTT * 16);
  // a 512 MB scrub between reps so nothing of the previous rep is cached
  char *scrub;
  size_t const scrub_bytes = (size_t)512 << 20;
  (void)hipMalloc(&scrub, scrub_bytes);
  std::vector<double> per, round;
  srand(7);
  for (int rep = 0; rep < 200; rep++) {
    int he[16];
    for (int s = 0; s < 16; s++) {
      he[s] = s == 8 ? 256 : rand() % 256;
    }
    (void)hipMemcpy(experts, he, 16 * 4, hipMemcpyHostToDevice);
    (void)hipMemset(sink, 0, 16 * 4);
    (void)hipMemset(bar, 0, 4);
    (void)hipMemset(scrub, rep & 0xff, scrub_bytes);
    (void)hipDeviceSynchronize();
    dim3 grid(256), blk(256);
    if (D == 4)
      hipLaunchKernelGGL(wsync<4>, grid, blk, 0, 0, base, expert_bytes, tile_bytes, per_xcc, tiles_per_expert,
                         help_kb, helpers, experts, t, sink, bar, total);
    else if (D == 8)
      hipLaunchKernelGGL(wsync<8>, grid, blk, 0, 0, base, expert_bytes, tile_bytes, per_xcc, tiles_per_expert,
                         help_kb, helpers, experts, t, sink, bar, total);
    else
      hipLaunchKernelGGL(wsync<16>, grid, blk, 0, 0, base, expert_bytes, tile_bytes, per_xcc, tiles_per_expert,
                         help_kb, helpers, experts, t, sink, bar, total);
    (void)hipDeviceSynchronize();
    if (rep < 10) continue;
    std::vector<unsigned long long> ht(NTT * 2);
    (void)hipMemcpy(ht.data(), t, NTT * 16, hipMemcpyDeviceToHost);
    unsigned long long lo = ~0ull, hi = 0;
    for (int g = 0; g < NTT; g++) {
      per.push_back((ht[g * 2 + 1] - ht[g * 2]) * 10.0 / 1000.0);
      lo = std::min(lo, ht[g * 2]);
      hi = std::max(hi, ht[g * 2 + 1]);
    }
    round.push_back((hi - lo) * 10.0 / 1000.0);
  }
  std::sort(per.begin(), per.end());
  std::sort(round.begin(), round.end());
  double const mb = (double)NTT * tile_bytes / 1e6;
  printf("NT %3d tile %6zu B D %2d help %2d KB x %2d/XCC: tile med %5.2f p90 %5.2f us, round med %5.2f us (%.2f TB/s)\n",
         NTT, tile_bytes, D, help_kb, helpers, per[per.size() / 2], per[per.size() * 9 / 10],
         round[round.size() / 2], mb / round[round.size() / 2]);
  return 0;
}
