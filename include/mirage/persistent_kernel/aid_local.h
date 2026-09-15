/* Copyright 2025 CMU / AMD
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

// AID-local placement for XCD-sliced gang-task inputs (SPX + NPS2, gfx950).
//
// In SPX+NPS2 the hardware reports one memory range per AID (144 GiB each on
// an 8-stack MI355X, boundary pfn 0x2400000). A plain hipMalloc lands in the
// XCP's home range, so half the XCDs read every byte of it across the AID
// boundary. Fleet already binds work to chiplets -- a gang task is launched
// with grid_dim.x == 8 and `imap.x` names the tensor dimension that the XCD
// axis partitions -- so for any input whose imap.x >= 0, XCD x streams a
// *disjoint* 1/8 slice. Those slices can be placed in the AID that owns XCD x
// and every one of their reads becomes local.
//
// This header does exactly that relocation and nothing else. The contract from
// aid-local-hbm/GEMV_3PLUS4.md is preserved by construction:
//
//   "Every large tensor the WG streams must be four-map + XCD bind. Barriers /
//    atomics / KV that every XCD must see stay on a shared map-0 hipMalloc."
//
// The safety property is structural rather than a hand-maintained list. A
// tensor is relocated only if the eight per-XCD TaskDescs hold eight pointers
// in a uniform non-zero arithmetic progression. That is true precisely when
// imap.x >= 0 (see src/kernel/runtime.cc, where the per-task base becomes
// `base + (dim[imap.x]/8) * bid.x * stride[imap.x]`). Every barrier, counter,
// KV cache and workspace in the fused layer task is declared with imap
// (-1,-1,-1) or (-1,k,-1), so all eight tasks carry the *same* pointer, the
// stride is 0, and the buffer is skipped. A shared buffer therefore cannot be
// relocated even by mistake.
//
// Enable per op, one at a time:
//
//   MPK_AID_LOCAL_SLOTS=9        # O-proj weight only
//   MPK_AID_LOCAL_SLOTS=4,9,13   # QKV weight, O-proj weight, router weight
//
// Slot numbers are input_ptrs indices; see the layout comment at the top of
// gang_full_layer_fused_mi300.cuh. Unset (the default) is a no-op, so a build
// carrying this header behaves exactly as today until a slot is named.
//
// Requires the patched amdgpu carrying AMDGPU_GEM_CREATE_AID_LOCAL
// (aid-local-hbm/0001) and the SPX+NPS2 mode table (0002). Every failure path
// leaves the original pointer in place and prints why, so a stock driver or an
// NPS1 machine degrades to today's behaviour instead of faulting.

#pragma once

#if defined(MIRAGE_BACKEND_USE_ROCM) &&                                        \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <drm/amdgpu_drm.h>
#include <drm/drm.h>

namespace mirage {
namespace aid {

// amdgpu-aid-local-vram GEM_CREATE domain flags. Bit 17 asks for AID-local
// placement; bits 18/19 are the range index. An 8-stack part has exactly two
// ranges, so only bit 18 is ever needed here (MI355X_8STACK.md).
static constexpr uint64_t GEM_CREATE_AID_LOCAL = 1ULL << 17;
static constexpr uint64_t GEM_CREATE_AID_SELECT = 1ULL << 18;
// Stock AMDGPU_GEM_CREATE_COHERENT. Without it an AID_LOCAL BO takes
// MTYPE_RW under aid_local_xcp_nc=1, and a non-temporal poll of an RW line
// never observes another die's write-through store -- that is a hang, not a
// slowdown. With it the driver applies aid_local_flag_mtype (CC), which is
// what makes a replicated flag line safe to spin on.
static constexpr uint64_t GEM_CREATE_COHERENT = 1ULL << 13;

static constexpr int kNumXcds = 8;
static constexpr int kMaxRanges = 4;

// SPX round-robins workgroup i to XCD i%8, and XCDs 0-3 sit on AID0 while
// 4-7 sit on AID1 (GEMV_3PLUS4.md: WG i -> XCD i%8 -> AID xcd>>2).
static inline int aid_of_xcd(int xcd) {
  return xcd < (kNumXcds / 2) ? 0 : 1;
}

struct Ctx {
  int drm_fd = -1;
  // Range index to use for AID0 and AID1. Index order is per-GPU, so these are
  // resolved by first-pfn against the midpoint of the covered span, never by
  // index (aid-local-hbm README: "group by pfn, never by index").
  unsigned range_for_aid[2] = {~0u, ~0u};
  bool ok = false;
  size_t bytes_placed = 0;
  int buffers_placed = 0;
};

// Open the render node whose PCI link matches `pci` (e.g. "0000:e5:00.0").
static inline int open_render_node(char const *pci) {
  for (int m = 128; m < 256; m++) {
    char path[128], link[256];
    snprintf(path, sizeof path, "/sys/class/drm/renderD%d/device", m);
    ssize_t n = readlink(path, link, sizeof link - 1);
    if (n <= 0) {
      continue;
    }
    link[n] = 0;
    if (strstr(link, pci) == nullptr) {
      continue;
    }
    snprintf(path, sizeof path, "/dev/dri/renderD%d", m);
    int fd = open(path, O_RDWR | O_CLOEXEC);
    if (fd >= 0) {
      return fd;
    }
  }
  return -1;
}

// Parse `aid_aperture` and pick one range per AID, exactly as
// aid-local-hbm/tools/gemv_spx.cpp does.
static inline bool resolve_ranges(char const *pci, unsigned out[2]) {
  char sp[256];
  snprintf(sp, sizeof sp, "/sys/bus/pci/devices/%s/aid_aperture", pci);
  FILE *f = fopen(sp, "r");
  if (f == nullptr) {
    printf("[AID] %s unreadable -- stock driver or not NPS2; AID-local off\n",
           sp);
    return false;
  }
  uint64_t sizes[kMaxRanges] = {0, 0, 0, 0};
  uint64_t fpfns[kMaxRanges] = {0, 0, 0, 0};
  char line[256];
  while (fgets(line, sizeof line, f) != nullptr) {
    unsigned idx;
    uint64_t fp, lp, sz;
    if (sscanf(line,
               "aid%u fpfn 0x%lx lpfn 0x%lx size 0x%lx",
               &idx,
               &fp,
               &lp,
               &sz) == 4 &&
        idx < kMaxRanges) {
      sizes[idx] = sz;
      fpfns[idx] = fp;
    }
  }
  fclose(f);

  uint64_t max_end = 0;
  for (unsigned r = 0; r < kMaxRanges; r++) {
    if (sizes[r] != 0) {
      uint64_t end = fpfns[r] + (sizes[r] >> 12);
      max_end = end > max_end ? end : max_end;
    }
  }
  if (max_end == 0) {
    printf("[AID] %s has no ranges; AID-local off\n", sp);
    return false;
  }
  uint64_t boundary = max_end / 2;

  uint64_t best_lo = 0, best_hi = 0;
  out[0] = ~0u;
  out[1] = ~0u;
  for (unsigned r = 0; r < kMaxRanges; r++) {
    if (sizes[r] == 0) {
      continue;
    }
    if (fpfns[r] < boundary) {
      if (sizes[r] > best_lo) {
        best_lo = sizes[r];
        out[0] = r;
      }
    } else {
      if (sizes[r] > best_hi) {
        best_hi = sizes[r];
        out[1] = r;
      }
    }
  }
  if (out[0] == ~0u || out[1] == ~0u) {
    printf("[AID] could not pick one range per AID from %s "
           "(boundary pfn 0x%lx); AID-local off\n",
           sp,
           boundary);
    return false;
  }
  printf("[AID] boundary pfn 0x%lx: AID0 -> range %u (%.0f GiB), "
         "AID1 -> range %u (%.0f GiB)\n",
         boundary,
         out[0],
         sizes[out[0]] / 1073741824.0,
         out[1],
         sizes[out[1]] / 1073741824.0);
  return true;
}

static inline Ctx &ctx() {
  static Ctx c;
  static bool tried = false;
  if (tried) {
    return c;
  }
  tried = true;

  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess) {
    return c;
  }
  char bus[64] = {0};
  if (hipDeviceGetPCIBusId(bus, sizeof bus, dev) != hipSuccess) {
    return c;
  }
  // HIP reports "0000:E5:00.0"; sysfs paths are lowercase.
  for (char *p = bus; *p != 0; p++) {
    if (*p >= 'A' && *p <= 'F') {
      *p = (char)(*p - 'A' + 'a');
    }
  }
  if (!resolve_ranges(bus, c.range_for_aid)) {
    return c;
  }
  c.drm_fd = open_render_node(bus);
  if (c.drm_fd < 0) {
    printf("[AID] no render node for %s; AID-local off\n", bus);
    return c;
  }
  c.ok = true;
  printf("[AID] %s ready for AID-local placement\n", bus);
  return c;
}

// GEM_CREATE a VRAM BO pinned to `aid`, export it, and map it into HIP.
// Returns nullptr on any failure (caller keeps the original pointer).
static inline void *alloc_in_aid(size_t bytes, int aid, bool coherent = false) {
  Ctx &c = ctx();
  if (!c.ok) {
    return nullptr;
  }
  uint64_t flags = GEM_CREATE_AID_LOCAL |
                   (c.range_for_aid[aid] & 1u ? GEM_CREATE_AID_SELECT : 0) |
                   (coherent ? GEM_CREATE_COHERENT : 0) |
                   AMDGPU_GEM_CREATE_NO_CPU_ACCESS;

  union drm_amdgpu_gem_create req;
  memset(&req, 0, sizeof req);
  req.in.bo_size = bytes;
  req.in.alignment = 2ULL << 20; // 2 MiB, so the BO maps with 2 MiB PTEs
  req.in.domains = AMDGPU_GEM_DOMAIN_VRAM;
  req.in.domain_flags = flags;
  if (ioctl(c.drm_fd, DRM_IOCTL_AMDGPU_GEM_CREATE, &req) != 0) {
    printf("[AID] GEM_CREATE %zu B flags 0x%llx failed: %s\n",
           bytes,
           (unsigned long long)flags,
           strerror(errno));
    return nullptr;
  }

  struct drm_prime_handle prime;
  memset(&prime, 0, sizeof prime);
  prime.handle = req.out.handle;
  if (ioctl(c.drm_fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &prime) != 0) {
    printf("[AID] PRIME export failed: %s\n", strerror(errno));
    return nullptr;
  }

  hipExternalMemoryHandleDesc hd = {};
  hd.type = hipExternalMemoryHandleTypeOpaqueFd;
  hd.handle.fd = prime.fd;
  hd.size = bytes;
  hipExternalMemory_t ext;
  if (hipImportExternalMemory(&ext, &hd) != hipSuccess) {
    printf("[AID] hipImportExternalMemory failed\n");
    return nullptr;
  }
  hipExternalMemoryBufferDesc bd = {};
  bd.offset = 0;
  bd.size = bytes;
  void *ptr = nullptr;
  if (hipExternalMemoryGetMappedBuffer(&ptr, ext, &bd) != hipSuccess) {
    printf("[AID] hipExternalMemoryGetMappedBuffer failed\n");
    return nullptr;
  }
  return ptr;
}

// ---------------------------------------------------------------------------
// Placement probe (opt-in via MPK_AID_LOCAL_PROBE): prove, per XCD, whether the
// slice it reads physically lives in its own AID -- BEFORE and AFTER relocation.
//
// It reuses the fleet's chiplet bind: a grid of kNumXcds blocks round-robins
// one block per XCD in SPX, and each block reads HW_REG_XCC_ID to learn its XCD
// (hence its home AID = xcc>>2). It times a single-outstanding (MLP=1)
// non-temporal read of (a) its actual weight slice, (b) a canary known to sit
// in AID0, (c) a canary known to sit in AID1. Whichever canary the slice's
// latency matches names the AID the slice really lives in. It is read-only and
// never writes the weight, so "work does not change" holds; only which AID
// backs the pages differs between the baseline and post-relocation calls.
//
// Requires aid_local_xcp_nc=1 (non-temporal loads fault on AID_LOCAL BOs
// otherwise), exactly like tools/aid_latency.cpp.

__device__ __forceinline__ unsigned aidp_xcc() {
  unsigned v;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(v));
  return v & 0xf;
}

// One active lane; each load's address depends on the previous load's value, so
// exactly one request is ever outstanding (measures latency, not bandwidth).
// The value only perturbs the stride by 0..63 elements and every index is taken
// mod nelem, so the walk stays in bounds and never writes. Identical pattern
// for the slice and both canaries makes their times directly comparable.
__device__ __forceinline__ unsigned long long
aidp_walk(const uint32_t *__restrict p, uint64_t nelem, int steps) {
  uint64_t idx = 0;
  for (int i = 0; i < 256; i++) { // warm the TLB; not timed
    uint32_t v = __builtin_nontemporal_load(&p[idx]);
    idx = (idx + 1024 + (v & 63u)) % nelem;
  }
  unsigned long long t0 = __builtin_amdgcn_s_memrealtime();
  for (int i = 0; i < steps; i++) {
    uint32_t v = __builtin_nontemporal_load(&p[idx]);
    idx = (idx + 1024 + (v & 63u)) % nelem;
  }
  unsigned long long t1 = __builtin_amdgcn_s_memrealtime();
  return (t1 - t0) + (idx & 0ull); // keep idx live
}

__global__ void aidp_probe_kernel(const uint32_t *const *__restrict slices,
                                  const uint32_t *__restrict canA,
                                  const uint32_t *__restrict canB,
                                  uint64_t nelem, int steps,
                                  unsigned long long *__restrict t_slice,
                                  unsigned long long *__restrict t_home,
                                  unsigned long long *__restrict t_far,
                                  unsigned *__restrict meta) {
  if (threadIdx.x != 0) {
    return; // MLP = 1: one active lane per block
  }
  int b = blockIdx.x;
  unsigned xcc = aidp_xcc() & 7u;
  unsigned home = xcc >> 2;                    // AID this XCD lives on
  const uint32_t *phome = home ? canB : canA;  // canary in the home AID
  const uint32_t *pfar = home ? canA : canB;   // canary in the far AID
  t_home[b] = aidp_walk(phome, nelem, steps);
  t_far[b] = aidp_walk(pfar, nelem, steps);
  t_slice[b] = aidp_walk(slices[b], nelem, steps);
  meta[b] = (xcc << 8) | home;
}

// Print the per-XCD table. slice_ptrs[x] is XCD x's slice base; slice_bytes is
// the per-XCD stride. Canaries are allocated once and reused across calls.
static inline void probe_xcd_locality(const char *tag,
                                      void *const slice_ptrs[kNumXcds],
                                      size_t slice_bytes) {
  Ctx &c = ctx();
  if (!c.ok) {
    return;
  }
  const size_t kCap = 64ull << 20; // cap the probe window; canaries this size
  size_t window = (slice_bytes < kCap ? slice_bytes : kCap) & ~((size_t)4095);
  if (window < (1u << 16)) {
    printf("[AID probe] %s: slice %zu B too small to probe\n", tag, slice_bytes);
    return;
  }
  uint64_t nelem = window / 4;
  const int steps = 4000;

  static void *canA = nullptr, *canB = nullptr; // AID0 / AID1, allocated once
  if (canA == nullptr) {
    canA = alloc_in_aid(kCap, 0);
    canB = alloc_in_aid(kCap, 1);
    if (canA == nullptr || canB == nullptr) {
      printf("[AID probe] canary alloc failed; skipping probe\n");
      return;
    }
    hipMemset(canA, 0x3c, kCap);
    hipMemset(canB, 0x3c, kCap);
    hipDeviceSynchronize();
  }

  const uint32_t *h_slices[kNumXcds];
  for (int i = 0; i < kNumXcds; i++) {
    h_slices[i] = static_cast<const uint32_t *>(slice_ptrs[i]);
  }
  const uint32_t **d_slices = nullptr;
  unsigned long long *d_ts = nullptr, *d_th = nullptr, *d_tf = nullptr;
  unsigned *d_meta = nullptr;
  hipMalloc((void **)&d_slices, sizeof(h_slices));
  hipMalloc(&d_ts, kNumXcds * sizeof(unsigned long long));
  hipMalloc(&d_th, kNumXcds * sizeof(unsigned long long));
  hipMalloc(&d_tf, kNumXcds * sizeof(unsigned long long));
  hipMalloc(&d_meta, kNumXcds * sizeof(unsigned));
  hipMemcpy(d_slices, h_slices, sizeof(h_slices), hipMemcpyHostToDevice);

  hipLaunchKernelGGL(aidp_probe_kernel, dim3(kNumXcds), dim3(64), 0, 0, d_slices,
                     static_cast<const uint32_t *>(canA),
                     static_cast<const uint32_t *>(canB), nelem, steps, d_ts,
                     d_th, d_tf, d_meta);
  if (hipDeviceSynchronize() != hipSuccess) {
    printf("[AID probe] %s: kernel failed (is aid_local_xcp_nc=1 set?)\n", tag);
    hipFree(d_slices);
    hipFree(d_ts);
    hipFree(d_th);
    hipFree(d_tf);
    hipFree(d_meta);
    return;
  }

  unsigned long long ts[kNumXcds], th[kNumXcds], tf[kNumXcds];
  unsigned meta[kNumXcds];
  hipMemcpy(ts, d_ts, sizeof ts, hipMemcpyDeviceToHost);
  hipMemcpy(th, d_th, sizeof th, hipMemcpyDeviceToHost);
  hipMemcpy(tf, d_tf, sizeof tf, hipMemcpyDeviceToHost);
  hipMemcpy(meta, d_meta, sizeof meta, hipMemcpyDeviceToHost);

  printf("[AID probe] %s: %llu KiB/XCD window, %d dependent NT reads (MLP=1)\n",
         tag, (unsigned long long)(window >> 10), steps);
  int local_cnt = 0;
  for (int b = 0; b < kNumXcds; b++) {
    unsigned xcc = meta[b] >> 8, home = meta[b] & 0xffu;
    double ns_s = ts[b] * 10.0 / steps;
    double ns_h = th[b] * 10.0 / steps;
    double ns_f = tf[b] * 10.0 / steps;
    long long dh = (long long)ts[b] - (long long)th[b];
    long long df = (long long)ts[b] - (long long)tf[b];
    if (dh < 0) {
      dh = -dh;
    }
    if (df < 0) {
      df = -df;
    }
    bool local = dh <= df; // slice latency matches the home-AID canary
    unsigned data_aid = local ? home : (home ^ 1u);
    local_cnt += local ? 1 : 0;
    printf("      WG %d  XCD %u  lives on AID%u  reads AID%u  %6.1f ns  [%s]"
           "  (home %.1f / far %.1f ns)\n",
           b, xcc, home, data_aid, ns_s, local ? "LOCAL " : "REMOTE", ns_h,
           ns_f);
  }
  printf("      => %d/%d XCDs read locally\n", local_cnt, kNumXcds);

  hipFree(d_slices);
  hipFree(d_ts);
  hipFree(d_th);
  hipFree(d_tf);
  hipFree(d_meta);
}

// Parse MPK_AID_LOCAL_SLOTS ("9" or "4,9,13") into input_ptrs indices.
static inline std::vector<int> slots_from_env(int max_slot) {
  std::vector<int> out;
  char const *e = getenv("MPK_AID_LOCAL_SLOTS");
  if (e == nullptr || *e == 0) {
    return out;
  }
  std::string s(e);
  size_t i = 0;
  while (i < s.size()) {
    size_t j = s.find(',', i);
    if (j == std::string::npos) {
      j = s.size();
    }
    if (j > i) {
      int v = atoi(s.substr(i, j - i).c_str());
      if (v >= 0 && v < max_slot) {
        out.push_back(v);
      } else {
        printf("[AID] ignoring out-of-range slot %d\n", v);
      }
    }
    i = j + 1;
  }
  return out;
}

// ---------------------------------------------------------------------------
// AID-split flag replicas (MPK_AID_SPLIT_FLAGS).
//
// The megakernel's per-XCD release flags live in a plain hipMalloc buffer. In
// SPX+NPS2 that buffer spans both memory ranges, is local to neither, and the
// driver demotes it to MTYPE_NC -- so every poll of it goes to the far
// coherency point: 182 ns from XCD 0-3 and 279 ns from 4-7, against 97 ns for
// an AID-local line. The O-proj slice wait alone polls these lines ~184 times
// a layer, which is the 14.16 us `slicewait` measured in phase7-bench.
//
// The fix is one COHERENT AID-local replica per AID, each holding all eight
// flags one 64 B line apart. A publisher writes its own flag into *both*
// replicas; every XCD polls only the replica homed in its own AID, so no
// reader is ever outside the coherence domain of the line it spins on. The
// far-AID half of each publish is sound because a remote writer still
// invalidates the sharers co-located with the line -- it is a remote *reader*
// that never gets probed. Measured in phase7-bench: slicewait 14.16 -> 0.36
// us, exact SPX+NPS1 parity (fork-fcm dd6ef9a).
//
// Replicating *data* does not pay and is deliberately not done here: the same
// session measured attn_out duplication at 0.08 us whether the slice is local,
// remote or replicated. Only polled flag lines are worth splitting.
static constexpr int kFlagStrideInts = 16; // 64 B, one line per XCD
static constexpr size_t kFlagRepBytes = 2ull << 20;

// Allocate the two replicas and zero them. Returns false if AID-local
// allocation is unavailable (stock driver, or not NPS2), leaving `out`
// untouched so the caller can fall back to the shared buffer.
static inline bool alloc_flag_replicas(void *out[2]) {
  Ctx &c = ctx();
  if (!c.ok) {
    printf("[AID] split flags unavailable: no AID-local allocator "
           "(stock driver, or not SPX+NPS2)\n");
    return false;
  }
  void *rep[2] = {nullptr, nullptr};
  for (int aid = 0; aid < 2; aid++) {
    rep[aid] = alloc_in_aid(kFlagRepBytes, aid, /*coherent=*/true);
    if (rep[aid] == nullptr) {
      printf("[AID] split flags: could not place a replica in AID%d\n", aid);
      return false;
    }
  }
  for (int aid = 0; aid < 2; aid++) {
    if (hipMemset(rep[aid], 0, kFlagRepBytes) != hipSuccess) {
      printf("[AID] split flags: hipMemset of the AID%d replica failed\n", aid);
      return false;
    }
  }
  out[0] = rep[0];
  out[1] = rep[1];
  printf("[AID] split flags: AID0 replica %p, AID1 replica %p (%zu B each, "
         "COHERENT)\n",
         rep[0],
         rep[1],
         kFlagRepBytes);
  return true;
}

// Relocate the named input slots of every fused-layer gang task into the AID
// that owns the XCD which reads them.
//
// `all_tasks` must still be pre-compaction, so that positions[L] + xcd is the
// TaskDesc for XCD `xcd` of layer L. Rewriting all_tasks (rather than only the
// ml pointer tables) is deliberate: layer 0 runs from the precomputed dispatch
// copy of all_tasks and layers 1+ from ml_input_table, and the table is built
// out of all_tasks a few lines below -- so one rewrite covers both.
template <typename TaskDescT>
inline void relocate_xcd_sliced_inputs(std::vector<TaskDescT> &all_tasks,
                                       std::vector<size_t> const &positions,
                                       int n_in) {
  std::vector<int> slots = slots_from_env(n_in);
  if (slots.empty()) {
    return;
  }
  if (!ctx().ok) {
    printf("[AID] MPK_AID_LOCAL_SLOTS set but AID-local is unavailable; "
           "leaving every pointer as-is\n");
    return;
  }
  const bool aid_probe = std::getenv("MPK_AID_LOCAL_PROBE") != nullptr;

  for (size_t L = 0; L < positions.size(); L++) {
    size_t pos = positions[L];
    for (size_t si = 0; si < slots.size(); si++) {
      int slot = slots[si];

      // The eight per-XCD pointers must form a uniform, non-zero progression.
      // Non-zero proves the tensor is XCD-partitioned (imap.x >= 0); uniform
      // proves each XCD's slice is the same size, which is what makes
      // `stride` the slice length. Anything else -- above all a *shared*
      // buffer, whose eight pointers are identical and whose stride is 0 --
      // is skipped. This is the check that keeps barriers, atomics, task
      // queues and KV out of AID-local memory.
      char *p0 = static_cast<char *>(all_tasks[pos + 0].input_ptrs[slot]);
      char *p1 = static_cast<char *>(all_tasks[pos + 1].input_ptrs[slot]);
      if (p0 == nullptr || p1 == nullptr) {
        continue;
      }
      ptrdiff_t stride = p1 - p0;
      if (stride <= 0) {
        if (L == 0) {
          printf("[AID] slot %d is not XCD-partitioned (stride %td) -- "
                 "shared buffer, skipped\n",
                 slot,
                 stride);
        }
        continue;
      }
      bool uniform = true;
      for (int xcd = 2; xcd < kNumXcds; xcd++) {
        char *px = static_cast<char *>(all_tasks[pos + xcd].input_ptrs[slot]);
        if (px == nullptr || px - p0 != stride * xcd) {
          uniform = false;
          break;
        }
      }
      if (!uniform) {
        if (L == 0) {
          printf("[AID] slot %d has a non-uniform XCD stride; skipped\n", slot);
        }
        continue;
      }

      // Baseline: where does each XCD's slice physically live BEFORE we move
      // anything? (Read-only; the weight bytes are untouched.)
      if (aid_probe && L == 0) {
        void *pre[kNumXcds];
        for (int xcd = 0; xcd < kNumXcds; xcd++) {
          pre[xcd] = p0 + stride * xcd;
        }
        printf("[AID] slot %d:\n", slot);
        probe_xcd_locality("baseline (pre-relocation)", pre, (size_t)stride);
      }

      // One AID-local BO per XCD, seeded from the slice it replaces.
      int moved = 0;
      for (int xcd = 0; xcd < kNumXcds; xcd++) {
        char *src = p0 + stride * xcd;
        void *dst = alloc_in_aid((size_t)stride, aid_of_xcd(xcd));
        if (dst == nullptr) {
          break;
        }
        if (hipMemcpy(dst, src, (size_t)stride, hipMemcpyDeviceToDevice) !=
            hipSuccess) {
          printf("[AID] seed copy failed for slot %d xcd %d\n", slot, xcd);
          break;
        }
        all_tasks[pos + xcd].input_ptrs[slot] = dst;
        ctx().bytes_placed += (size_t)stride;
        ctx().buffers_placed++;
        moved++;
      }
      if (moved != kNumXcds) {
        printf("[AID] slot %d layer %zu only %d/%d XCDs placed -- run is now "
               "mixed; abort and investigate\n",
               slot,
               L,
               moved,
               kNumXcds);
      } else if (L == 0) {
        printf("[AID] slot %d: %td B per XCD, XCD 0-3 -> AID0, 4-7 -> AID1\n",
               slot,
               stride);
      }

      // Verify: re-probe the SAME XCDs now that the slices have moved. Same
      // code, same work -- only which AID backs each slice has changed.
      if (aid_probe && L == 0 && moved == kNumXcds) {
        void *post[kNumXcds];
        for (int xcd = 0; xcd < kNumXcds; xcd++) {
          post[xcd] = all_tasks[pos + xcd].input_ptrs[slot];
        }
        probe_xcd_locality("aid-local (post-relocation)", post, (size_t)stride);
      }
    }
  }
  if (hipDeviceSynchronize() != hipSuccess) {
    printf("[AID] sync after seed copies failed\n");
  }
  printf("[AID] placed %d buffers, %.2f GiB total, across %zu fused layers\n",
         ctx().buffers_placed,
         ctx().bytes_placed / 1073741824.0,
         positions.size());
  fflush(stdout);
}

} // namespace aid
} // namespace mirage

#endif // ROCm
