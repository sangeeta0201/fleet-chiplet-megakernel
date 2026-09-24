// Physical placement probe for the MoE's buffers (opt-in: MPK_PPROBE=1).
//
// Answers, per layer and per replica half, whether the XCDs that consume an
// address are the ones it is homed next to -- by measurement, independent of
// the driver's placement rules. One lane on each physical XCC (claimed via
// HW_REG_XCC_ID, never inferred from blockIdx, which fleet rotates by one)
// issues `sc0 sc1` loads to the same address, each drained before the next, so
// exactly one request is in flight. On MTYPE_NC a coherent-scope load never
// hits L2, so every load pays the trip to the home stack: ~420 cycles near,
// ~700 far. On MTYPE_RW it hits L2 after the first touch, which shows up as a
// flat ~110-130 ns on all XCCs -- the probe then reports the MTYPE, not a home.
//
// No canary: the home of an address is whichever half of the XCCs reads it
// faster, so the verdict cannot inherit an assumption about which XCC belongs
// to which AID.
#pragma once
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace mpk_pprobe {

constexpr int kSamples = 8; // addresses per region, spread over it
constexpr int kSteps = 8;   // timed loads per address

__global__ void pprobe_kernel(unsigned long long const *__restrict addrs,
                              int nreg,
                              int reps,
                              unsigned *__restrict claim,
                              unsigned *__restrict out_ns) {
  if (threadIdx.x != 0) {
    return;
  }
  unsigned xcc;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcc));
  xcc &= 7u;
  if (atomicCAS(&claim[xcc], 0u, 1u) != 0u) {
    return;
  }
  // Stagger the start so the eight XCCs are never on the same line at once.
  int const r0 = (int)((xcc * (unsigned)nreg) / 8u);
  for (int i = 0; i < nreg; i++) {
    int const r = (r0 + i) % nreg;
    for (int s = 0; s < kSamples; s++) {
      unsigned long long const a = addrs[r * kSamples + s];
      unsigned best = 0xffffffffu;
      for (int rep = 0; rep < reps; rep++) {
        unsigned v;
        asm volatile("global_load_dword %0, %1, off sc0 sc1\n\t"
                     "s_waitcnt vmcnt(0)"
                     : "=v"(v)
                     : "v"(a)
                     : "memory");
        unsigned long long const t0 = __builtin_amdgcn_s_memrealtime();
        for (int k = 0; k < kSteps; k++) {
          asm volatile("global_load_dword %0, %1, off sc0 sc1\n\t"
                       "s_waitcnt vmcnt(0)"
                       : "=v"(v)
                       : "v"(a)
                       : "memory");
        }
        unsigned long long const t1 = __builtin_amdgcn_s_memrealtime();
        unsigned const ns = (unsigned)((t1 - t0) * 10ull / kSteps);
        best = ns < best ? ns : best;
      }
      out_ns[(r * 8 + (int)xcc) * kSamples + s] = best;
    }
  }
}

static inline long long env_ll(char const *k, long long d) {
  char const *e = std::getenv(k);
  return (e != nullptr && *e != '\0') ? atoll(e) : d;
}

struct Region {
  int layer;
  int half; // 0/1 = replica half consumed by XCCs (x>>2)==half; -1 = home only
  char const *name;
};

// all_tasks is still pre-compaction: positions[L] + xcd is XCD xcd's
// descriptor for layer L. The MoE slots are shared, so XCD 0's copy suffices.
template <typename TaskDescT>
inline void run(std::vector<TaskDescT> &all_tasks,
                std::vector<size_t> const &positions) {
  // gpt-oss-120b geometry, as the MoE kernel computes it: [2E, 46, 200192]
  // gate_up, [2E, 46, 100096] down, [2E, 5888] / [2E, 2944] bf16 biases.
  long long const E = env_ll("MPK_PPROBE_E", 128);
  struct Slot {
    int slot;
    char const *name;
    long long expert_bytes;
  } const halves[] = {
      {17, "gate_up", env_ll("MPK_PPROBE_W13_EB", 46LL * 200192)},
      {18, "down", env_ll("MPK_PPROBE_W2_EB", 46LL * 100096)},
      {19, "w13_bias", env_ll("MPK_PPROBE_B13_EB", 5888LL * 2)},
      {20, "w2_bias", env_ll("MPK_PPROBE_B2_EB", 2944LL * 2)},
  };
  struct Home {
    int slot;
    char const *name;
    long long bytes;
  } const homes[] = {
      {0, "workspace", 4LL * 2944 * 4},
      {21, "moe_barrier", 4096},
      {22, "swiglu_torch", 4LL * 3072 * 2},
      {12, "moe_norm", 2944LL * 2},
  };
  std::vector<Region> regs;
  std::vector<unsigned long long> addrs;
  for (size_t L = 0; L < positions.size(); L++) {
    TaskDescT const &td = all_tasks[positions[L]];
    for (Slot const &q : halves) {
      char *p = (char *)td.input_ptrs[q.slot];
      if (p == nullptr) {
        continue;
      }
      long long const half = E * q.expert_bytes;
      long long const step = half / kSamples;
      for (int h = 0; h < 2; h++) {
        regs.push_back({(int)L, h, q.name});
        for (int s = 0; s < kSamples; s++) {
          long long const off = (h * half + step * s + step / 2) & ~127LL;
          addrs.push_back((unsigned long long)(p + off));
        }
      }
    }
    for (Home const &q : homes) {
      char *p = (char *)td.input_ptrs[q.slot];
      if (p == nullptr) {
        continue;
      }
      regs.push_back({(int)L, -1, q.name});
      for (int s = 0; s < kSamples; s++) {
        long long const off = ((q.bytes / kSamples) * s) & ~127LL;
        addrs.push_back((unsigned long long)(p + off));
      }
    }
  }
  int const nreg = (int)regs.size();
  if (nreg == 0) {
    printf("[PPROBE] no MoE regions found\n");
    fflush(stdout);
    return;
  }
  int const reps = (int)env_ll("MPK_PPROBE_REPS", 5);
  unsigned long long *d_addrs = nullptr;
  unsigned *d_claim = nullptr, *d_out = nullptr;
  size_t const n_out = (size_t)nreg * 8 * kSamples;
  hipMalloc((void **)&d_addrs, addrs.size() * sizeof(unsigned long long));
  hipMalloc((void **)&d_claim, 8 * sizeof(unsigned));
  hipMalloc((void **)&d_out, n_out * sizeof(unsigned));
  hipMemcpy(d_addrs,
            addrs.data(),
            addrs.size() * sizeof(unsigned long long),
            hipMemcpyHostToDevice);
  hipMemset(d_claim, 0, 8 * sizeof(unsigned));
  hipMemset(d_out, 0xff, n_out * sizeof(unsigned));
  hipLaunchKernelGGL(pprobe_kernel,
                     dim3(128),
                     dim3(64),
                     0,
                     0,
                     d_addrs,
                     nreg,
                     reps,
                     d_claim,
                     d_out);
  hipError_t const err = hipDeviceSynchronize();
  std::vector<unsigned> out(n_out, 0xffffffffu);
  unsigned claim[8] = {0};
  if (err == hipSuccess) {
    hipMemcpy(out.data(), d_out, n_out * sizeof(unsigned),
              hipMemcpyDeviceToHost);
    hipMemcpy(claim, d_claim, sizeof claim, hipMemcpyDeviceToHost);
  }
  hipFree(d_addrs);
  hipFree(d_claim);
  hipFree(d_out);
  if (err != hipSuccess) {
    printf("[PPROBE] kernel failed: %s\n", hipGetErrorString(err));
    fflush(stdout);
    return;
  }
  printf("[PPROBE] ==== %d regions, %d samples x %d loads, min of %d reps; "
         "XCCs claimed %u%u%u%u%u%u%u%u ====\n",
         nreg, kSamples, kSteps, reps, claim[0], claim[1], claim[2], claim[3],
         claim[4], claim[5], claim[6], claim[7]);
  // A sample is homed on the faster XCC group only if the groups are clearly
  // apart; the near/far hop is ~130 ns, so 40 ns is well clear of noise.
  unsigned const kSep = (unsigned)env_ll("MPK_PPROBE_SEP_NS", 40);
  unsigned const kHit = (unsigned)env_ll("MPK_PPROBE_HIT_NS", 160);
  struct Tot {
    char const *name;
    int local, remote, unclear, cached;
  };
  std::vector<Tot> tot;
  auto tot_of = [&](char const *nm) -> Tot & {
    for (Tot &t : tot) {
      if (t.name == nm) {
        return t;
      }
    }
    tot.push_back({nm, 0, 0, 0, 0});
    return tot.back();
  };
  for (int r = 0; r < nreg; r++) {
    Region const &g = regs[r];
    unsigned mean_x[8];
    int loc = 0, rem = 0, unc = 0, hit = 0, home0 = 0, home1 = 0;
    for (int x = 0; x < 8; x++) {
      unsigned long long acc = 0;
      for (int s = 0; s < kSamples; s++) {
        acc += out[(r * 8 + x) * kSamples + s];
      }
      mean_x[x] = (unsigned)(acc / kSamples);
    }
    for (int s = 0; s < kSamples; s++) {
      unsigned long long a0 = 0, a1 = 0;
      for (int x = 0; x < 8; x++) {
        unsigned const v = out[(r * 8 + x) * kSamples + s];
        (x < 4 ? a0 : a1) += v;
      }
      unsigned const g0 = (unsigned)(a0 / 4), g1 = (unsigned)(a1 / 4);
      if (g0 < kHit && g1 < kHit) {
        hit++;
        continue;
      }
      int home = -1;
      if (g0 + kSep < g1) {
        home = 0;
      } else if (g1 + kSep < g0) {
        home = 1;
      }
      if (home == 0) {
        home0++;
      } else if (home == 1) {
        home1++;
      }
      if (g.half < 0) {
        continue;
      }
      if (home < 0) {
        unc++;
      } else if (home == g.half) {
        loc++;
      } else {
        rem++;
      }
    }
    Tot &t = tot_of(g.name);
    t.local += loc;
    t.remote += rem;
    t.unclear += unc;
    t.cached += hit;
    char verdict[64];
    if (hit == kSamples) {
      snprintf(verdict, sizeof verdict, "L2-HIT (RW?)");
    } else if (g.half < 0) {
      snprintf(verdict, sizeof verdict, "home AID0=%d AID1=%d unclear=%d",
               home0, home1, kSamples - home0 - home1 - hit);
    } else {
      snprintf(verdict, sizeof verdict, "cons=AID%d local=%d remote=%d%s",
               g.half, loc, rem, rem > 0 ? "  <-- REMOTE" : "");
    }
    printf("[PPROBE] L=%02d %-12s h=%2d ns %4u %4u %4u %4u | %4u %4u %4u %4u  "
           "%s\n",
           g.layer, g.name, g.half, mean_x[0], mean_x[1], mean_x[2],
           mean_x[3], mean_x[4], mean_x[5], mean_x[6], mean_x[7], verdict);
  }
  for (Tot const &t : tot) {
    printf("[PPROBE] SUMMARY %-12s local=%d remote=%d unclear=%d l2hit=%d\n",
           t.name, t.local, t.remote, t.unclear, t.cached);
  }
  fflush(stdout);
}

} // namespace mpk_pprobe

