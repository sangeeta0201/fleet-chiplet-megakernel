// Stage 3: a decode MoE layer on the nps2 runtime.
//
// This is the piece fleet structurally cannot do. Fleet's MoE weights are
// SHARED buffers -- all eight XCD pointers are identical, because experts are
// selected dynamically per token -- so relocation skips them and every
// workgroup ends up half-remote. Measured consequence: fleet's MoE weight
// read costs 386 ns against this box's 154 ns AID-local, i.e. WORSE than the
// benchmark's fully-remote arm, on the op holding 29% of the NPS2 gap.
//
// The fix has two halves and needs both:
//   1. expert-parity PLACEMENT -- expert e's slabs live in AID (e & 1)
//   2. data-directed DISPATCH  -- the tiles for expert e are executed by
//      workgroups on XCDs inside AID (e & 1)
// Placement without dispatch is `split50`, which measured no better than
// all-remote (24.84 vs 24.37) because the remote half of the workgroups gates
// the kernel.
//
// Shape follows gpt-oss-120b bs=1 decode: 128 experts, top-4, hidden 2880,
// intermediate 2880. W13 is gate+up (2x), W2 projects back. FP32 stands in
// for the MXFP4 MFMA inner kernel -- this validates the memory and dispatch
// architecture, and swapping in the real MFMA kernel is mechanical because
// neither the placement nor the barrier depends on the arithmetic.
//
// Correctness gate: the checksum must be IDENTICAL across every arm. A fast
// wrong answer is not a result.
#include "aid_rt.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

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

static constexpr int kExperts = 128;
static constexpr int kTopK = 4;
static constexpr int kHidden = 2880;
static constexpr int kInter = 1024; // trimmed so one layer fits comfortably
static constexpr int kTilesPerExpert = 8;

// Per-expert slab sizes, in floats.
static constexpr size_t kW13Elems = (size_t)kInter * 2 * kHidden / 64; // packed
static constexpr size_t kW2Elems = (size_t)kHidden * kInter / 64;

struct ExpertWeights {
  // One pointer per expert. Under parity placement, expert e's pointer is a
  // slab inside the AID-(e&1) arena, so it is genuinely local to its
  // consumers rather than merely "in an AID".
  float const *w13[kExperts];
  float const *w2[kExperts];
};

struct MoeArgs {
  ExpertWeights w;
  int const *route;         // [kTopK] selected experts for this token
  float const *act[kNumAids]; // class B: identical bytes, one copy per AID
  float *inter;             // [kTopK][kInter] W13 output, per-expert private
  float *out;               // [kHidden] final, the one class C handoff
  BarrierSet bar;
  unsigned int *timeouts;   // barrier deadline misses, so they are not silent
  int parity_dispatch; // 1 = tiles follow data's AID, 0 = naive blockIdx order
};

// Which (slot, tile) does this workgroup own?
//
// BOTH arms must cover the 32 work items exactly once, or the arms compute
// different things and the checksum comparison is meaningless. The first
// version's parity branch double-counted and skipped, which is exactly what
// the correctness gate caught.
//
// parity_dispatch=1 is the point of the design: a workgroup on an XCD in AID
// a takes only work whose expert has parity a, so every weight byte it reads
// is local. parity_dispatch=0 assigns item i to workgroup i regardless of
// where the weight lives -- half the workgroups then read across the
// boundary, which is what fleet does today.
__device__ __forceinline__ bool my_work(MoeArgs const &A, int wg, int &slot,
                                        int &tile) {
  int const total = kTopK * kTilesPerExpert; // 32
  if (!A.parity_dispatch) {
    if (wg >= total) {
      return false;
    }
    slot = wg / kTilesPerExpert;
    tile = wg % kTilesPerExpert;
    return true;
  }

  int const xcd = my_xcd();
  int const my_a = aid_of_xcd(xcd);
  // Rank of this workgroup among all workgroups sitting on the same AID.
  // WG -> XCD is wg % 8, and XCDs 0-3 are AID0, so within an AID the rank is
  // (wg / 8) * 4 + (xcd % 4).
  int const rank_in_aid = (wg / kNumXcds) * (kNumXcds / 2) + (xcd % (kNumXcds / 2));

  // Enumerate this AID's work items: every tile of every routed expert whose
  // parity matches. Take the rank-th one.
  int seen = 0;
  for (int s = 0; s < kTopK; s++) {
    if ((A.route[s] & 1) != my_a) {
      continue;
    }
    for (int t = 0; t < kTilesPerExpert; t++) {
      if (seen == rank_in_aid) {
        slot = s;
        tile = t;
        return true;
      }
      seen++;
    }
  }
  return false;
}

__global__ __launch_bounds__(64) void k_moe(MoeArgs A, unsigned int epoch) {
  int const wg = blockIdx.x;
  int slot = 0, tile = 0;
  bool const active = my_work(A, wg, slot, tile);

  // ---- W13: read expert weights (class A, local under parity) ----
  if (active) {
    int const e = A.route[slot];
    float const *w = A.w.w13[e];
    int const per_tile = kInter / kTilesPerExpert;
    int const base = tile * per_tile;
    // Class B: read the replica homed in this workgroup's own AID.
    float const *a = A.act[aid_of_xcd(my_xcd())];
    for (int i = threadIdx.x; i < per_tile; i += blockDim.x) {
      // Stand-in for the MXFP4 MFMA: a strided dot product over hidden.
      float acc = 0.0f;
      size_t const woff = ((size_t)(base + i) * kHidden) / 64;
      for (int h = 0; h < kHidden / 64; h++) {
        acc += w[(woff + h) % kW13Elems] * a[h * 64];
      }
      // SwiGLU stand-in, monotone and cheap so the checksum stays stable.
      float const g = acc;
      A.inter[(size_t)slot * kInter + base + i] = g / (1.0f + __expf(-g));
    }
  }

  // ---- barrier: W13 must be complete before W2 reduces it ----
  // This is the gate that measured 3344 ns at 256 blocks vs FLAT's 15515.
  // 0.02 s deadline: long enough that a healthy barrier never trips it, short
  // enough that a broken one shows up as a fast counted failure instead of
  // looking like a slow kernel.
  if (!A.bar.arrive_and_wait(0, epoch, 2ULL * 1000ULL * 1000ULL)) {
    if (threadIdx.x == 0) {
      atomicAdd(A.timeouts, 1u);
    }
  }

  // ---- W2: reduce into the one class C buffer ----
  if (active && tile == 0) {
    int const e = A.route[slot];
    float const *w = A.w.w2[e];
    for (int h = threadIdx.x; h < kHidden; h += blockDim.x) {
      float acc = 0.0f;
      for (int i = 0; i < kInter; i += 64) {
        acc += w[((size_t)h * kInter / 64 + i / 64) % kW2Elems] *
               A.inter[(size_t)slot * kInter + i];
      }
      // Genuine all-to-all: every slot contributes to every output element.
      atomicAdd(&A.out[h], acc);
    }
  }
}

// ----------------------------------------------------------------- host ----
struct Arena {
  char *base[kNumAids] = {};
  size_t used[kNumAids] = {};
  size_t cap = 0;

  bool init(size_t bytes_per_aid) {
    cap = bytes_per_aid;
    for (int a = 0; a < kNumAids; a++) {
      base[a] = static_cast<char *>(rt_alloc_aid(bytes_per_aid, a));
      if (base[a] == nullptr) {
        return false;
      }
    }
    return true;
  }
  // Hand out a slab from the arena homed in `aid`.
  float *take(int aid, size_t elems) {
    size_t const b = elems * sizeof(float);
    if (used[aid] + b > cap) {
      return nullptr;
    }
    float *p = reinterpret_cast<float *>(base[aid] + used[aid]);
    used[aid] += (b + 255) & ~(size_t)255;
    return p;
  }
};

int main(int argc, char **argv) {
  // Unbuffered: if a barrier deadlocks and the process is killed by timeout,
  // block-buffered stdout is discarded and the run looks like it produced
  // nothing at all rather than showing how far it got.
  setvbuf(stdout, nullptr, _IONBF, 0);

  int const n_wgs = argc > 1 ? atoi(argv[1]) : 184;
  int const iters = argc > 2 ? atoi(argv[2]) : 200;

  hipDeviceProp_t prop;
  HIP_OK(hipGetDeviceProperties(&prop, 0));
  printf("=== MoE layer on nps2 runtime: %s, %d CUs, grid %d ===\n",
         prop.gcnArchName, prop.multiProcessorCount, n_wgs);
  printf("  %d experts, top-%d, hidden %d, inter %d\n", kExperts, kTopK,
         kHidden, kInter);

  // Expert weights, parity-placed: expert e in AID (e & 1).
  size_t const per_expert = (kW13Elems + kW2Elems) * sizeof(float);
  size_t const per_aid = per_expert * (kExperts / 2) + (64u << 20);
  Arena arena;
  if (!arena.init(per_aid)) {
    printf("  arena alloc failed (%.1f GiB per AID) -- is this SPX+NPS2?\n",
           per_aid / 1073741824.0);
    return 1;
  }
  printf("  arena: %.2f GiB per AID\n", per_aid / 1073741824.0);

  ExpertWeights w{};
  std::vector<float> host_slab(kW13Elems > kW2Elems ? kW13Elems : kW2Elems);
  for (size_t i = 0; i < host_slab.size(); i++) {
    host_slab[i] = 0.001f * (float)((i * 37) % 251 - 125);
  }
  for (int e = 0; e < kExperts; e++) {
    int const a = e & 1; // parity placement
    float *p13 = arena.take(a, kW13Elems);
    float *p2 = arena.take(a, kW2Elems);
    if (p13 == nullptr || p2 == nullptr) {
      printf("  arena exhausted at expert %d\n", e);
      return 1;
    }
    HIP_OK(hipMemcpy(p13, host_slab.data(), kW13Elems * sizeof(float),
                     hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(p2, host_slab.data(), kW2Elems * sizeof(float),
                     hipMemcpyHostToDevice));
    w.w13[e] = p13;
    w.w2[e] = p2;
  }

  // Class B: activation, one replica per AID.
  std::vector<float> host_act(kHidden);
  for (int i = 0; i < kHidden; i++) {
    host_act[i] = 0.01f * (float)(i % 97 - 48);
  }
  float *act[kNumAids];
  for (int a = 0; a < kNumAids; a++) {
    act[a] = arena.take(a, kHidden);
    HIP_OK(hipMemcpy(act[a], host_act.data(), kHidden * sizeof(float),
                     hipMemcpyHostToDevice));
  }

  // Routing: top-4 experts. Deliberately two of each parity so the
  // parity-dispatch arm has balanced work.
  int const h_route[kTopK] = {6, 17, 40, 83};
  int *d_route;
  HIP_OK(hipMalloc(&d_route, kTopK * sizeof(int)));
  HIP_OK(hipMemcpy(d_route, h_route, sizeof h_route, hipMemcpyHostToDevice));

  float *d_inter, *d_out;
  HIP_OK(hipMalloc(&d_inter, (size_t)kTopK * kInter * sizeof(float)));
  HIP_OK(hipMalloc(&d_out, kHidden * sizeof(float)));

  // Sync state: gates and per-XCD counters AID-local, the 8-per-barrier
  // global counter shared.
  GateSet g{};
  g.n_gates = 8;
  size_t const gb = (size_t)g.n_gates * kGateStrideInts * sizeof(unsigned);
  for (int a = 0; a < kNumAids; a++) {
    g.rep[a] = reinterpret_cast<unsigned int *>(arena.take(a, gb / 4));
  }
  BarrierSet bar{};
  bar.release = g;
  for (int a = 0; a < kNumAids; a++) {
    bar.xcd_count[a] = reinterpret_cast<unsigned int *>(
        arena.take(a, (size_t)kNumXcds * kGateStrideInts));
  }
  HIP_OK(hipMalloc(&bar.global_count, kGateStrideInts * sizeof(unsigned)));

  // Measure the real workgroup -> XCD distribution for this exact grid.
  {
    size_t const nslots = 2 * kNumXcds + 1;
    unsigned int *d_cnt;
    HIP_OK(hipMalloc(&d_cnt, nslots * sizeof(unsigned)));
    HIP_OK(hipMemset(d_cnt, 0, nslots * sizeof(unsigned)));
    hipLaunchKernelGGL(k_probe_xcd, dim3(n_wgs), dim3(64), 0, 0, d_cnt);
    HIP_OK(hipDeviceSynchronize());
    std::vector<unsigned int> h_cnt(nslots);
    HIP_OK(hipMemcpy(h_cnt.data(), d_cnt, nslots * sizeof(unsigned),
                     hipMemcpyDeviceToHost));
    printf("  WG->XCD real (HW_REG_XCC_ID):");
    int tot = 0;
    for (int x = 0; x < kNumXcds; x++) {
      bar.n_expect[x] = (int)h_cnt[x];
      tot += (int)h_cnt[x];
      printf(" %u", h_cnt[x]);
    }
    printf("  (total %d of %d)\n", tot, n_wgs);
    printf("  WG->XCD assumed (blockIdx%%8):");
    for (int x = 0; x < kNumXcds; x++) {
      printf(" %u", h_cnt[kNumXcds + x]);
    }
    printf("\n  workgroups where the assumption is WRONG: %u of %d\n",
           h_cnt[2 * kNumXcds], n_wgs);
    if (tot != n_wgs) {
      printf("  WARNING: probe saw %d of %d workgroups; barrier would hang\n",
             tot, n_wgs);
      return 1;
    }
    HIP_OK(hipFree(d_cnt));
  }

  auto reset_sync = [&]() {
    for (int a = 0; a < kNumAids; a++) {
      (void)hipMemset(g.rep[a], 0, gb);
      (void)hipMemset(bar.xcd_count[a], 0,
                      (size_t)kNumXcds * kGateStrideInts * 4);
    }
    (void)hipMemset(bar.global_count, 0, kGateStrideInts * 4);
  };

  unsigned int *d_to;
  HIP_OK(hipMalloc(&d_to, sizeof(unsigned)));

  printf("\n  arm                      us/layer   bar_timeouts   checksum\n");
  double t_par = 0, t_naive = 0;
  double sum_par = 0, sum_naive = 0;

  for (int arm = 0; arm < 2; arm++) {
    MoeArgs A{};
    A.w = w;
    A.route = d_route;
    A.act[0] = act[0];
    A.act[1] = act[1];
    A.inter = d_inter;
    A.out = d_out;
    A.bar = bar;
    A.timeouts = d_to;
    A.parity_dispatch = (arm == 0) ? 1 : 0;

    // --- correctness: ONE launch into a freshly zeroed output. The output is
    // an atomicAdd accumulator, so timing over N launches deliberately piles
    // N layers into it -- that is fine for timing and useless for checking,
    // which is why the two are measured separately.
    reset_sync();
    HIP_OK(hipMemset(d_out, 0, kHidden * sizeof(float)));
    HIP_OK(hipMemset(d_to, 0, sizeof(unsigned)));
    hipLaunchKernelGGL(k_moe, dim3(n_wgs), dim3(64), 0, 0, A, 1u);
    HIP_OK(hipDeviceSynchronize());
    unsigned int h_to = 0;
    HIP_OK(hipMemcpy(&h_to, d_to, sizeof h_to, hipMemcpyDeviceToHost));

    // A timeout means one of the three levels did not complete. Read all
    // three back and say WHICH, instead of inferring it: per-XCD arrivals
    // (level 1), the 8-wide global count (level 2), and the release gate
    // (level 3).
    if (h_to != 0) {
      unsigned int hx[kNumAids][kNumXcds * kGateStrideInts] = {};
      unsigned int hg[kGateStrideInts] = {};
      unsigned int hr[kNumAids][kGateStrideInts] = {};
      for (int a = 0; a < kNumAids; a++) {
        HIP_OK(hipMemcpy(hx[a], bar.xcd_count[a],
                         (size_t)kNumXcds * kGateStrideInts * 4,
                         hipMemcpyDeviceToHost));
        HIP_OK(hipMemcpy(hr[a], g.rep[a], kGateStrideInts * 4,
                         hipMemcpyDeviceToHost));
      }
      HIP_OK(hipMemcpy(hg, bar.global_count, kGateStrideInts * 4,
                       hipMemcpyDeviceToHost));
      printf("    barrier stalled: %u blocks timed out\n", h_to);
      printf("      L1 per-XCD arrivals:");
      for (int x = 0; x < kNumXcds; x++) {
        printf(" %u/%d", hx[aid_of_xcd(x)][x * kGateStrideInts],
               bar.n_expect[x]);
      }
      printf("\n      L2 global count: %u (need %d)\n", hg[0], kNumXcds);
      printf("      L3 release gate: AID0=%u AID1=%u (need >=1)\n", hr[0][0],
             hr[1][0]);
    }
    std::vector<float> h_out(kHidden);
    HIP_OK(hipMemcpy(h_out.data(), d_out, kHidden * sizeof(float),
                     hipMemcpyDeviceToHost));
    double chk = 0;
    for (int i = 0; i < kHidden; i++) {
      chk += (double)h_out[i];
    }

    // --- timing: best of 3, each a fresh epoch sequence
    hipEvent_t ea, eb;
    hipEventCreate(&ea);
    hipEventCreate(&eb);
    double best = 1e30;
    for (int rep = 0; rep < 3; rep++) {
      reset_sync();
      HIP_OK(hipMemset(d_out, 0, kHidden * sizeof(float)));
      hipEventRecord(ea);
      for (int it = 1; it <= iters; it++) {
        hipLaunchKernelGGL(k_moe, dim3(n_wgs), dim3(64), 0, 0, A,
                           (unsigned)it);
      }
      hipEventRecord(eb);
      hipError_t const se = hipDeviceSynchronize();
      if (se != hipSuccess) {
        printf("  arm %d: %s\n", arm, hipGetErrorString(se));
        break;
      }
      float ms = 0.0f;
      hipEventElapsedTime(&ms, ea, eb);
      double const us = (double)ms * 1000.0 / (double)iters;
      if (us < best) {
        best = us;
      }
    }
    hipEventDestroy(ea);
    hipEventDestroy(eb);

    printf("  %-24s %8.2f   %12u   %.9e\n",
           arm == 0 ? "parity place+dispatch" : "naive dispatch (fleet)", best,
           h_to, chk);
    if (arm == 0) {
      t_par = best;
      sum_par = chk;
    } else {
      t_naive = best;
      sum_naive = chk;
    }
  }

  printf("\n");
  if (t_par > 0 && t_naive > 0) {
    printf("  parity dispatch is %.2fx %s than naive\n",
           t_naive > t_par ? t_naive / t_par : t_par / t_naive,
           t_naive > t_par ? "FASTER" : "SLOWER");
  }
  bool const same = fabs(sum_par - sum_naive) <=
                    1e-6 * (fabs(sum_par) + fabs(sum_naive) + 1e-12);
  printf("  checksums %s (%.6e vs %.6e)\n", same ? "MATCH" : "DIFFER -- INVALID",
         sum_par, sum_naive);
  printf("  %s\n", same ? "correctness gate: pass"
                        : "correctness gate: FAIL, timing is meaningless");
  return 0;
}

