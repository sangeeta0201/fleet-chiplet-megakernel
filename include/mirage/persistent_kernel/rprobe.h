// Home-die probe for the per-AID flag replicas (opt-in: MPK_RPROBE=1).
//
// XCDs 0-3 poll replica 0 and XCDs 4-7 replica 1, so a replica line is
// "favorable" when it is homed on the die of its own pollers. In NPS2 the
// allocator guarantees that; in NPS1 memory is interleaved and each line lands
// on either die by address, per process.
//
// Replica lines are MTYPE_RW, where a device-scope load hits L2, so each timed
// load is preceded by an L2 eviction: the probing wave streams a private 16 MiB
// buffer (4x its XCD's L2), then lane 0 times one sc1 load of the line, which
// is served by the home die's MALL or HBM. sc1 is the scope the megakernel
// uses on these lines; a system-scope (sc0 sc1) load of an alloc_in_aid BO
// takes a PERMISSION_FAULTS 0x3 retry fault. Min over reps per XCC; the home
// is the XCC half that reads it clearly faster.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace mpk_rprobe {

// Set where the replicas are published; probed after the first launch.
inline void *g_rep[2] = {nullptr, nullptr};

__global__ void rprobe_kernel(unsigned long long const *__restrict addrs,
                              int n,
                              int reps,
                              char const *__restrict evict,
                              unsigned long long evict_bytes,
                              unsigned *__restrict claim,
                              unsigned *__restrict out_ns) {
  __shared__ int s_xcc;
  __shared__ int s_ok;
  if (threadIdx.x == 0) {
    unsigned xcc;
    MPK_GET_XCC_ID16(xcc);
    xcc &= 7u;
    s_xcc = (int)xcc;
    s_ok = atomicCAS(&claim[xcc], 0u, 1u) == 0u;
  }
  __syncthreads();
  if (!s_ok) {
    return;
  }
  int const xcc = s_xcc;
  // Plain C++ loads for the eviction stream so the compiler tracks their
  // waits: an asm load without its wait lets the compiler reuse the
  // destination VGPR while the load is in flight.
  unsigned const *ev = reinterpret_cast<unsigned const *>(
      evict + (unsigned long long)xcc * evict_bytes);
  unsigned long long const nw = evict_bytes / 4;
  unsigned acc = 0;
  for (int i = 0; i < n; i++) {
    unsigned best = 0xffffffffu;
    unsigned long long a = addrs[i];
    for (int r = 0; r < reps; r++) {
#pragma unroll 8
      for (unsigned long long k = threadIdx.x * 4ull; k < nw; k += 64ull * 4ull) {
        acc += ev[k];
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      __builtin_amdgcn_wave_barrier();
      if (threadIdx.x == 0) {
        unsigned v;
        unsigned long long t0, t1;
        asm volatile("s_memrealtime %1\n\t"
                     "s_waitcnt lgkmcnt(0)\n\t"
                     "global_load_dword %0, %3, off sc1\n\t"
                     "s_waitcnt vmcnt(0)\n\t"
                     "s_memrealtime %2\n\t"
                     "s_waitcnt lgkmcnt(0)"
                     : "=&v"(v), "=&s"(t0), "=&s"(t1)
                     : "v"(a)
                     : "memory");
        unsigned const ns = (unsigned)((t1 - t0) * 10ull);
        best = ns < best ? ns : best;
        acc += v;
      }
      __builtin_amdgcn_wave_barrier();
    }
    if (threadIdx.x == 0) {
      out_ns[i * 8 + xcc] = best;
    }
  }
  if (acc == 0x9e3779b9u) {
    out_ns[0] = acc;  // keeps the eviction loads alive
  }
}

struct Line {
  char const *fam;
  int rep, idx;
};

// Probes the hot sync lines of both replicas and prints one [RPROBE] line per
// family plus a summary. rep[r] = replica r's base (ints).
inline void run(void *const rep_v[2]) {
  if (rep_v[0] == nullptr || rep_v[1] == nullptr) {
    printf("[RPROBE] no replicas (stock driver): nothing to probe\n");
    fflush(stdout);
    return;
  }
  // (family, int offset of slot 0, slot stride in ints, slots)
  struct Fam {
    char const *name;
    int off, stride, slots;
  };
  Fam const fams[] = {
      {"attn_release", 0 * 128, 16, 8},  {"layer_release", 1 * 128, 16, 8},
      {"hier_release", 3 * 128, 16, 8},  {"qkv_epoch", 4 * 128, 16, 8},
      {"qkv_arrive", 24576, 16, 8},      {"chunk_bar", 24832, 16, 8},
      {"ltk_tags", 32768 + 1024, 32, 4}, {"ltk_norm", 32768 + 1152, 32, 8},
      {"iter_gate", 491520, 16, 8},
  };
  std::vector<unsigned long long> addrs;
  std::vector<Line> lines;
  for (int r = 0; r < 2; r++) {
    int *base = static_cast<int *>(rep_v[r]);
    for (Fam const &f : fams) {
      for (int s = 0; s < f.slots; s++) {
        addrs.push_back((unsigned long long)(base + f.off + s * f.stride));
        lines.push_back({f.name, r, s});
      }
    }
  }
  int const n = (int)addrs.size();
  char const *e = std::getenv("MPK_RPROBE_REPS");
  int const reps = e ? atoi(e) : 3;
  unsigned long long const evict_bytes = 16ull << 20;
  unsigned long long *d_addrs = nullptr;
  unsigned *d_claim = nullptr, *d_out = nullptr;
  char *d_evict = nullptr;
  bool ok = hipMalloc((void **)&d_addrs, n * sizeof(unsigned long long)) == hipSuccess &&
            hipMalloc((void **)&d_claim, 8 * sizeof(unsigned)) == hipSuccess &&
            hipMalloc((void **)&d_out, (size_t)n * 8 * sizeof(unsigned)) == hipSuccess &&
            hipMalloc((void **)&d_evict, 8 * evict_bytes) == hipSuccess;
  fprintf(stderr,
          "[RPROBE] rep0 %p rep1 %p (last line +%llu B) evict %p..%p addrs %p out %p ok=%d\n",
          rep_v[0], rep_v[1], (unsigned long long)(addrs.back() - (unsigned long long)rep_v[1]),
          (void *)d_evict, (void *)(d_evict + 8 * evict_bytes), (void *)d_addrs, (void *)d_out,
          (int)ok);
  if (!ok) {
    return;
  }
  ok = hipMemcpy(d_addrs, addrs.data(), n * sizeof(unsigned long long),
                 hipMemcpyHostToDevice) == hipSuccess &&
       hipMemset(d_claim, 0, 8 * sizeof(unsigned)) == hipSuccess &&
       hipMemset(d_out, 0xff, (size_t)n * 8 * sizeof(unsigned)) == hipSuccess &&
       hipMemset(d_evict, 0, 8 * evict_bytes) == hipSuccess &&
       hipDeviceSynchronize() == hipSuccess;
  if (!ok) {
    fprintf(stderr, "[RPROBE] setup failed\n");
    return;
  }
  hipLaunchKernelGGL(rprobe_kernel, dim3(256), dim3(64), 0, 0, d_addrs, n, reps,
                     d_evict, evict_bytes, d_claim, d_out);
  hipError_t const err = hipDeviceSynchronize();
  std::vector<unsigned> out((size_t)n * 8, 0xffffffffu);
  unsigned claim[8] = {0};
  if (err == hipSuccess) {
    (void)hipMemcpy(out.data(), d_out, out.size() * sizeof(unsigned), hipMemcpyDeviceToHost);
    (void)hipMemcpy(claim, d_claim, sizeof claim, hipMemcpyDeviceToHost);
  }
  (void)hipFree(d_addrs);
  (void)hipFree(d_claim);
  (void)hipFree(d_out);
  (void)hipFree(d_evict);
  if (err != hipSuccess) {
    printf("[RPROBE] kernel failed: %s\n", hipGetErrorString(err));
    fflush(stdout);
    return;
  }
  printf("[RPROBE] ==== %d lines x %d reps, XCCs claimed %u%u%u%u%u%u%u%u ====\n",
         n, reps, claim[0], claim[1], claim[2], claim[3], claim[4], claim[5],
         claim[6], claim[7]);
  int fav_all = 0, unf_all = 0, unc_all = 0;
  for (int r = 0; r < 2; r++) {
    for (Fam const &f : fams) {
      int fav = 0, unf = 0, unc = 0;
      unsigned long long s0 = 0, s1 = 0;
      char map[16] = {0};
      int k = 0;
      for (int i = 0; i < n; i++) {
        if (lines[i].rep != r || lines[i].fam != f.name) {
          continue;
        }
        unsigned long long a0 = 0, a1 = 0;
        for (int x = 0; x < 8; x++) {
          (x < 4 ? a0 : a1) += out[(size_t)i * 8 + x];
        }
        a0 /= 4;
        a1 /= 4;
        s0 += a0;
        s1 += a1;
        int home = (a0 + 40 < a1) ? 0 : (a1 + 40 < a0) ? 1 : -1;
        map[k++] = home < 0 ? '?' : (char)('0' + home);
        if (home < 0) {
          unc++;
        } else if (home == r) {
          fav++;
        } else {
          unf++;
        }
      }
      fav_all += fav;
      unf_all += unf;
      unc_all += unc;
      printf("[RPROBE] rep%d %-13s home %s  near %d far %d unclear %d  "
             "mean ns xcc0-3 %llu xcc4-7 %llu\n",
             r, f.name, map, fav, unf, unc, s0 / (k ? k : 1), s1 / (k ? k : 1));
    }
  }
  printf("[RPROBE] SUMMARY near-own-pollers %d far %d unclear %d\n", fav_all, unf_all,
         unc_all);
  fflush(stdout);
}

}  // namespace mpk_rprobe
