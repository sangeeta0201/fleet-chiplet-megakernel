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
// Stock AMDGPU_GEM_CREATE_COHERENT / _UNCACHED. On gfx950 the PTE encoding is
// MTYPE_NC=0, RW=1, CC=2, UC=3, and the loaded per-BO-flag driver reports its
// choice in dmesg ("PERBO: AID_LOCAL BO is local -> mtype_local=1",
// "FLAGMTYPE fired: knob=2 -> mtype=2"). In SPX+NPS2:
//
//   spanning hipMalloc     -> NC: cacheable, NOT coherent across XCDs.
//   AID_LOCAL alone        -> RW: coherent within one memory partition only.
//   AID_LOCAL | COHERENT   -> CC: coherent everywhere, but backed by the DF-CS
//                             shadow tags (~8 L2 lines per channel). Overflow
//                             loses probes and hangs, so CC is only safe for a
//                             handful of lines -- never for bulk data.
//   AID_LOCAL | UNCACHED   -> UC: always correct, always slowest.
static constexpr uint64_t GEM_CREATE_COHERENT = 1ULL << 13;
static constexpr uint64_t GEM_CREATE_UNCACHED = 1ULL << 14;

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
  // Capacity of each AID's range, and how much of it this process has taken.
  // Overrunning a range does not fail: GEM_CREATE still returns a BO, it just
  // stops being AID-local, which shows up only as unexplained latency. These
  // counters exist so that shows up as a log line instead.
  uint64_t range_bytes[2] = {0, 0};
  size_t bytes_by_aid[2] = {0, 0};
  int bufs_by_aid[2] = {0, 0};
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
static inline bool resolve_ranges(char const *pci, unsigned out[2],
                                  uint64_t out_bytes[2]) {
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
      // Print every range as parsed. Which range is AID0 is decided below by
      // fpfn order, and that decision is an assumption about the hardware, so
      // the inputs to it belong in the log where they can be checked against
      // the probe's measured near/far canary timings.
      printf("[AID] aid_aperture range %u: fpfn 0x%lx lpfn 0x%lx size %.1f GiB\n",
             idx, fp, lp, sz / 1073741824.0);
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
  out_bytes[0] = sizes[out[0]];
  out_bytes[1] = sizes[out[1]];
  printf("[AID] boundary pfn 0x%lx: AID0 -> range %u (%.0f GiB, fpfn 0x%lx), "
         "AID1 -> range %u (%.0f GiB, fpfn 0x%lx)\n",
         boundary,
         out[0],
         sizes[out[0]] / 1073741824.0,
         fpfns[out[0]],
         out[1],
         sizes[out[1]] / 1073741824.0,
         fpfns[out[1]]);
  printf("[AID] AID_SELECT bit will be %s for AID0 and %s for AID1\n",
         (out[0] & 1u) ? "SET" : "clear",
         (out[1] & 1u) ? "SET" : "clear");
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
  if (!resolve_ranges(bus, c.range_for_aid, c.range_bytes)) {
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
static inline void *
    alloc_in_aid(size_t bytes, int aid, uint64_t extra_flags = 0) {
  Ctx &c = ctx();
  if (!c.ok) {
    return nullptr;
  }
  uint64_t flags = GEM_CREATE_AID_LOCAL |
                   (c.range_for_aid[aid] & 1u ? GEM_CREATE_AID_SELECT : 0) |
                   extra_flags | AMDGPU_GEM_CREATE_NO_CPU_ACCESS;

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

  c.bytes_by_aid[aid] += bytes;
  c.bufs_by_aid[aid]++;
  // Report each allocation against the range it is supposed to fit inside.
  // Filling a range is the failure mode that looks like success: GEM_CREATE
  // keeps returning BOs, they just stop being AID-local, and the only symptom
  // is latency. Note `fill` counts only this process's AID-local BOs -- weights
  // the model allocated through hipMalloc already occupy one of these ranges
  // and are invisible here, so real occupancy is higher than shown.
  double cap = c.range_bytes[aid] / 1073741824.0;
  double used = c.bytes_by_aid[aid] / 1073741824.0;
  printf("[AID]   alloc #%d in AID%d: %.2f GiB (range %u, AID_SELECT %s) -> %p"
         "; AID%d fill %.2f/%.0f GiB (%.0f%%)\n",
         c.bufs_by_aid[aid],
         aid,
         bytes / 1073741824.0,
         c.range_for_aid[aid],
         (flags & GEM_CREATE_AID_SELECT) ? "set" : "clear",
         ptr,
         aid,
         used,
         cap,
         cap > 0.0 ? 100.0 * used / cap : 0.0);
  if (cap > 0.0 && used > 0.75 * cap) {
    size_t free_b = 0, total_b = 0;
    if (hipMemGetInfo(&free_b, &total_b) != hipSuccess) {
      free_b = 0;
    }
    printf("[AID]   WARNING: AID%d is %.0f%% full from this process alone "
           "(device free %.1f GiB). Further allocations here may land outside "
           "the AID and read remote with no error.\n",
           aid,
           100.0 * used / cap,
           free_b / 1073741824.0);
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
//
// aidp_sink observes the final walk index so the dependent-load chain has a
// visible side effect. Without it the loads feed nothing (the old `idx & 0`
// discarded them), the compiler eliminates the whole timed loop, and t1 == t0
// so every latency prints 0.0 ns.
static __device__ volatile unsigned long long aidp_sink;

__device__ __forceinline__ unsigned long long
aidp_walk(const uint32_t *__restrict p, uint64_t mask, int steps) {
  // `mask` is nelem-1 for a power-of-two element count, so the wrap is an AND.
  // A `% nelem` here would put a 64-bit modulo by a runtime value inside the
  // dependent chain and charge its ALU latency to memory on every step, which
  // inflated every reading by enough to push local reads (~155 ns on this part)
  // up into the remote band (~259 ns).
  uint64_t idx = 0;
  // Warm every page of the window, not a prefix of it: the timed loop wanders
  // over the whole window, so a short warm-up leaves it taking TLB misses that
  // are also charged to memory. Step by 2 MiB / 4 B to touch each large page.
  uint64_t warm = 0;
  for (uint64_t w = 0; w <= mask; w += (2ull << 20) / sizeof(uint32_t)) {
    warm += __builtin_nontemporal_load(&p[w]);
  }
  aidp_sink = warm; // one store, before the timed window
  for (int i = 0; i < 512; i++) { // warm the chase itself; not timed
    uint32_t v = __builtin_nontemporal_load(&p[idx]);
    idx = (idx + 1024 + (v & 63u)) & mask;
  }
  unsigned long long t0 = __builtin_amdgcn_s_memrealtime();
  asm volatile("" ::: "memory"); // pin the timed loads inside the t0..t1 window
  for (int i = 0; i < steps; i++) {
    uint32_t v = __builtin_nontemporal_load(&p[idx]);
    idx = (idx + 1024 + (v & 63u)) & mask;
  }
  asm volatile("" ::: "memory");
  unsigned long long t1 = __builtin_amdgcn_s_memrealtime();
  aidp_sink = idx; // observe the chain so the dependent loads survive DCE
  return t1 - t0;
}

__global__ void aidp_probe_kernel(const uint32_t *const *__restrict slices,
                                  const uint32_t *__restrict canA,
                                  const uint32_t *__restrict canB,
                                  uint64_t mask, int steps,
                                  unsigned long long *__restrict t_slice,
                                  unsigned long long *__restrict t_a,
                                  unsigned long long *__restrict t_b,
                                  unsigned *__restrict meta) {
  if (threadIdx.x != 0) {
    return; // MLP = 1: one active lane per block
  }
  int b = blockIdx.x;
  unsigned xcc = aidp_xcc() & 7u;
  // Time both canaries unconditionally and let the host decide which is near.
  // Selecting home/far here with `xcc >> 2` would make the probe assume the
  // very XCD-to-AID mapping it exists to check: if that mapping and the
  // range-to-AID mapping were both inverted, the two errors would cancel and
  // every XCD would be reported LOCAL while reading entirely remote memory.
  t_a[b] = aidp_walk(canA, mask, steps);
  t_b[b] = aidp_walk(canB, mask, steps);
  t_slice[b] = aidp_walk(slices[b], mask, steps);
  meta[b] = (xcc << 8) | (xcc >> 2); // measured id, and the assumed home AID
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
  // Round the element count down to a power of two so the walk can wrap with an
  // AND instead of a modulo; a `%` by a runtime value would sit on the dependent
  // chain and be charged to memory.
  uint64_t nelem = window / 4;
  uint64_t pow2 = 1;
  while (pow2 * 2 <= nelem) {
    pow2 *= 2;
  }
  uint64_t mask = pow2 - 1;
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
  unsigned long long *d_ts = nullptr, *d_ta = nullptr, *d_tb = nullptr;
  unsigned *d_meta = nullptr;
  hipMalloc((void **)&d_slices, sizeof(h_slices));
  hipMalloc(&d_ts, kNumXcds * sizeof(unsigned long long));
  hipMalloc(&d_ta, kNumXcds * sizeof(unsigned long long));
  hipMalloc(&d_tb, kNumXcds * sizeof(unsigned long long));
  hipMalloc(&d_meta, kNumXcds * sizeof(unsigned));
  hipMemcpy(d_slices, h_slices, sizeof(h_slices), hipMemcpyHostToDevice);

  // Repeat and keep the per-XCD MINIMUM. Latency noise is one-sided -- nothing
  // makes an access faster than the hardware allows, while interference makes it
  // arbitrarily slower -- so the fastest pass is the best estimate of the true
  // cost. tools/aid_latency.cpp uses 40 reps for the same reason; a single pass
  // (what this did before) reports whatever interference happened to occur.
  int reps = 10;
  {
    char const *e = std::getenv("MPK_AID_LOCAL_PROBE_REPS");
    if (e != nullptr) {
      int v = atoi(e);
      if (v > 0 && v <= 200) {
        reps = v;
      }
    }
  }
  unsigned long long ts[kNumXcds], ta[kNumXcds], tb[kNumXcds];
  unsigned meta[kNumXcds] = {0};
  for (int i = 0; i < kNumXcds; i++) {
    ts[i] = ta[i] = tb[i] = ~0ull;
  }
  for (int r = 0; r < reps; r++) {
    hipLaunchKernelGGL(aidp_probe_kernel, dim3(kNumXcds), dim3(64), 0, 0,
                       d_slices, static_cast<const uint32_t *>(canA),
                       static_cast<const uint32_t *>(canB), mask, steps, d_ts,
                       d_ta, d_tb, d_meta);
    if (hipDeviceSynchronize() != hipSuccess) {
      printf("[AID probe] %s: kernel failed (is aid_local_xcp_nc=1 set?)\n",
             tag);
      hipFree(d_slices);
      hipFree(d_ts);
      hipFree(d_ta);
      hipFree(d_tb);
      hipFree(d_meta);
      return;
    }
    unsigned long long s[kNumXcds], a[kNumXcds], b[kNumXcds];
    hipMemcpy(s, d_ts, sizeof s, hipMemcpyDeviceToHost);
    hipMemcpy(a, d_ta, sizeof a, hipMemcpyDeviceToHost);
    hipMemcpy(b, d_tb, sizeof b, hipMemcpyDeviceToHost);
    hipMemcpy(meta, d_meta, sizeof meta, hipMemcpyDeviceToHost);
    for (int i = 0; i < kNumXcds; i++) {
      ts[i] = s[i] < ts[i] ? s[i] : ts[i];
      ta[i] = a[i] < ta[i] ? a[i] : ta[i];
      tb[i] = b[i] < tb[i] ? b[i] : tb[i];
    }
  }

  // Below this the two canaries are indistinguishable, so no verdict derived
  // from comparing against them means anything. The AID hop measures ~60-100 ns
  // on this part, so a separation in the single digits is noise.
  const double kMinSepNs = 20.0;

  // Where each buffer lives is decided by comparing XCD groups reading the SAME
  // buffer, never by comparing a buffer against a canary. The walk stride is
  // `idx + 1024 + (v & 63)`, so it depends on the value loaded: the canaries are
  // memset to a uniform 0x3c and walk at a constant stride, while a weight
  // buffer holds real MXFP4 and walks irregularly. That makes canary-vs-slice
  // times incomparable -- in practice slices measure 400-530 ns against
  // canaries at 236-390 ns, so the slice is slower than both and "nearest
  // canary" is decided by noise. Comparing XCD 0-3 against XCD 4-7 on one
  // buffer holds the data and the pattern fixed and varies only the reader,
  // which is the one comparison the walk supports.
  double lo_sum = 0.0, hi_sum = 0.0;
  for (int b = 0; b < kNumXcds; b++) {
    double ns = ts[b] * 10.0 / steps;
    if ((meta[b] >> 8) < (unsigned)(kNumXcds / 2)) {
      lo_sum += ns;
    } else {
      hi_sum += ns;
    }
  }
  double lo_avg = lo_sum / (kNumXcds / 2), hi_avg = hi_sum / (kNumXcds / 2);
  double skew = lo_avg - hi_avg;

  printf("[AID probe] %s: %llu KiB/XCD window, %d dependent NT reads (MLP=1)\n",
         tag, (unsigned long long)(window >> 10), steps);
  printf("      WG XCD | canA_ns canB_ns  sep | near assumed | slice_ns  in  |"
         " verdict\n");
  int local_cnt = 0, mismap = 0, weak = 0;
  for (int b = 0; b < kNumXcds; b++) {
    unsigned xcc = meta[b] >> 8, assumed = meta[b] & 0xffu;
    double ns_s = ts[b] * 10.0 / steps;
    double ns_a = ta[b] * 10.0 / steps;
    double ns_b = tb[b] * 10.0 / steps;
    // Which range this XCD sits next to, measured rather than assumed.
    unsigned near = (ns_a <= ns_b) ? 0u : 1u;
    double sep = (ns_a > ns_b) ? ns_a - ns_b : ns_b - ns_a;
    // Which side of the machine this buffer sits on, from the group skew rather
    // than from canary matching: whichever XCD half reads it faster is the half
    // whose AID backs it.
    unsigned lives = (skew <= 0.0) ? 0u : 1u;
    bool local = (lives == near);
    local_cnt += local ? 1 : 0;
    mismap += (near != assumed) ? 1 : 0;
    weak += (sep < kMinSepNs) ? 1 : 0;
    printf("      %2d %3u | %7.1f %7.1f %4.1f%s | AID%u AID%u%s | %7.1f AID%u |"
           " %s\n",
           b, xcc, ns_a, ns_b, sep, sep < kMinSepNs ? "?" : " ", near, assumed,
           near != assumed ? "!" : " ", ns_s, lives,
           local ? "LOCAL " : "REMOTE");
  }
  // Machine-parseable duplicate of the summary, so a run's placement map can be
  // scraped out of the log without re-deriving it from the table above.
  printf("[MAP] %s lo_ns=%.1f hi_ns=%.1f skew_ns=%+.1f aid=%u placeable=%d "
         "local=%d/%d\n",
         tag,
         lo_avg,
         hi_avg,
         skew,
         (skew <= 0.0) ? 0u : 1u,
         (skew <= -kMinSepNs || skew >= kMinSepNs) ? 1 : 0,
         local_cnt,
         kNumXcds);
  printf("      => XCD0-3 avg %.1f ns, XCD4-7 avg %.1f ns, skew %+.1f ns "
         "=> buffer looks resident in AID%u; %d/%d XCDs read locally\n",
         lo_avg,
         hi_avg,
         skew,
         (skew <= 0.0) ? 0u : 1u,
         local_cnt,
         kNumXcds);
  if (skew > -kMinSepNs && skew < kMinSepNs) {
    printf("      !! group skew is only %+.1f ns: this buffer is either "
           "interleaved across both AIDs or the measurement is too noisy to "
           "place it -- do not read a home AID out of this row\n",
           skew);
  }
  // The two ways this table can be confidently wrong, called out by name rather
  // than left for a human to spot in the columns.
  if (mismap != 0) {
    printf("      !! %d XCD(s) measured NEARER the other range than aid_of_xcd()"
           " assumes: the XCD->AID grouping or the range->AID order is"
           " inverted, and placement is sending them to the far AID\n",
           mismap);
  }
  if (weak != 0) {
    printf("      !! %d XCD(s) saw < %.0f ns between the two canaries: for those"
           " rows LOCAL/REMOTE is noise, not measurement\n",
           weak, kMinSepNs);
  }

  hipFree(d_slices);
  hipFree(d_ts);
  hipFree(d_ta);
  hipFree(d_tb);
  hipFree(d_meta);
}

// Time ONE buffer from all eight XCDs. Locality is then a property of the
// buffer: whichever XCD half reads it faster is the half whose AID backs it.
// Handing the probe a split pointer set instead (XCD 0-3 on one replica, 4-7 on
// the other) cannot answer "is each half local" at all, because the two halves
// are then reading different buffers and the skew between them conflates the
// two placements. Each replica has to be timed from both halves.
static inline void probe_buffer_home(char const *tag, void *buf, size_t bytes) {
  void *same[kNumXcds];
  for (int i = 0; i < kNumXcds; i++) {
    same[i] = buf;
  }
  probe_xcd_locality(tag, same, bytes);
}

// Fused-layer input slot names, in the order gang_full_layer_fused_layer()
// declares them in persistent_kernel.py (24 inputs). Only used for the map.
static char const *kSlotNames[24] = {
    "workspace_f32",     "residual",       "norm_weight_pre",
    "norm_scratch_pre",  "qkv_weight",     "qkv_bias",
    "sinks",             "qkv_barrier",    "lse_acc",
    "oproj_weight",      "oproj_bias",     "norm_weight_post",
    "norm_scratch_post", "router_weight",  "router_bias",
    "logits_scratch",    "oproj_counters", "gate_up_weight",
    "down_weight",       "w13_bias",       "w2_bias",
    "moe_barrier",       "swiglu_out",     "o_acc_f32"};

// Dump, for every input the fused layer reads, what it is, how large its backing
// allocation is, whether the eight XCDs share one pointer or take a slice each,
// and which AID the pages sit in. This is the baseline characterisation: without
// it, "move the weights closer" is guesswork about which allocations are even on
// the far side and how many bytes per token cross the interconnect.
//
// Placement is reported from the XCD-group skew on the same buffer, which is the
// only sound timing comparison available on a read-only weight (see the note in
// probe_xcd_locality). Allocations whose skew is under the noise floor are marked
// unplaceable rather than assigned an AID.
template <typename TaskDescT>
static inline void dump_baseline_map(std::vector<TaskDescT> &all_tasks,
                                     std::vector<size_t> const &positions,
                                     int n_in) {
  if (!ctx().ok) {
    printf("[MAP] AID-local unavailable; cannot map placement\n");
    return;
  }
  int const n = n_in < 24 ? n_in : 24;
  printf("[MAP] ==== NPS2 baseline placement map: %d input slots, layer 0 of "
         "%zu fused layers ====\n",
         n,
         positions.size());
  for (int a = 0; a < 2; a++) {
    printf("[MAP] range AID%d = index %u, %.0f GiB\n",
           a,
           ctx().range_for_aid[a],
           ctx().range_bytes[a] / 1073741824.0);
  }
  size_t const pos = positions[0];
  for (int slot = 0; slot < n; slot++) {
    void *p0 = all_tasks[pos].input_ptrs[slot];
    if (p0 == nullptr) {
      printf("[MAP] slot=%d name=%s null\n", slot, kSlotNames[slot]);
      continue;
    }
    bool shared = true;
    for (int xcd = 1; xcd < kNumXcds; xcd++) {
      if (all_tasks[pos + xcd].input_ptrs[slot] != p0) {
        shared = false;
        break;
      }
    }
    // Size of the whole allocation behind this pointer, not of the slice: what
    // decides placement is where the allocation went.
    void *abase = nullptr;
    size_t abytes = 0;
    if (hipMemGetAddressRange(reinterpret_cast<hipDeviceptr_t *>(&abase),
                              &abytes, p0) != hipSuccess) {
      abase = nullptr;
      abytes = 0;
    }
    // Offset of this layer's pointer inside the allocation. Nonzero means the
    // allocation is a pool shared with other layers (the MoE packer does this),
    // which is what makes per-layer replication wasteful and per-segment
    // replication correct.
    size_t aoff = (abase != nullptr)
                      ? static_cast<size_t>(static_cast<char *>(p0) -
                                            static_cast<char *>(abase))
                      : 0;
    char tag[192];
    snprintf(tag,
             sizeof tag,
             "slot=%d name=%s kind=%s alloc_mib=%.2f off_mib=%.2f",
             slot,
             kSlotNames[slot],
             shared ? "shared" : "sliced",
             abytes / 1048576.0,
             aoff / 1048576.0);
    // Probe the pointers the kernel actually uses: one shared base, or the eight
    // per-XCD slices. Either way all eight reads land in the same allocation, so
    // the group skew still locates it.
    void *ptrs[kNumXcds];
    for (int xcd = 0; xcd < kNumXcds; xcd++) {
      ptrs[xcd] = all_tasks[pos + xcd].input_ptrs[slot];
    }
    // Size the window from each pointer to the END of the allocation it sits in,
    // never from the allocation's total size. A slot pointer is generally an
    // offset *into* its allocation -- the MoE packer pools several layers into
    // one segment, and a sliced tensor gives XCD x a base 
    // `alloc + x*stride` -- so walking `abytes` from there runs off the end and
    // faults the GPU ("Memory access fault ... Reason: Unknown", which is what
    // the first map run hit). The probe walks one window length from every
    // pointer, so the usable window is the SMALLEST headroom of the eight.
    size_t probe_bytes = ~(size_t)0;
    bool bounded = true;
    for (int xcd = 0; xcd < kNumXcds && bounded; xcd++) {
      void *xbase = nullptr;
      size_t xbytes = 0;
      if (ptrs[xcd] == nullptr ||
          hipMemGetAddressRange(reinterpret_cast<hipDeviceptr_t *>(&xbase),
                                &xbytes, ptrs[xcd]) != hipSuccess ||
          xbase == nullptr || xbytes == 0) {
        bounded = false;
        break;
      }
      char *cur = static_cast<char *>(ptrs[xcd]);
      char *lo = static_cast<char *>(xbase);
      char *hi = lo + xbytes;
      if (cur < lo || cur >= hi) {
        bounded = false;
        break;
      }
      size_t head = static_cast<size_t>(hi - cur);
      probe_bytes = head < probe_bytes ? head : probe_bytes;
    }
    if (!bounded) {
      printf("[MAP] %s unbounded=1 not_probed (no usable address range)\n", tag);
      continue;
    }
    probe_xcd_locality(tag, ptrs, probe_bytes);
  }
  printf("[MAP] ==== end map ====\n");
  fflush(stdout);
}

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
  // MPK_AID_SPLIT_FLAGS_MTYPE selects the PTE memory type of the replicas.
  //
  // RW is the default despite being slower than CC (4.34 vs 4.11 ms/token),
  // because CC is backed by the DF-CS shadow tags -- ~8 L2 lines per channel
  // -- and overflowing that directory loses probes and hangs. Measured: the
  // same CC build ran clean once and then deadlocked before a single task
  // retired on a repeat, and produced a third distinct output text on another
  // run. CC is only safe for a handful of lines, which does not survive
  // adding more flag families.
  //
  // RW needs no directory, so it scales. It is sound here for the same reason
  // the split works at all: each replica is read only by XCDs inside its own
  // memory partition, which is exactly RW's coherence domain, and a remote
  // *writer* still invalidates the sharers co-located with the line.
  char const *m = getenv("MPK_AID_SPLIT_FLAGS_MTYPE");
  char const *mname = "rw";
  uint64_t extra = 0;
  if (m != nullptr && strcmp(m, "uc") == 0) {
    mname = "uc";
    extra = GEM_CREATE_UNCACHED;
  } else if (m != nullptr && strcmp(m, "cc") == 0) {
    mname = "cc";
    extra = GEM_CREATE_COHERENT;
  }
  void *rep[2] = {nullptr, nullptr};
  for (int aid = 0; aid < 2; aid++) {
    rep[aid] = alloc_in_aid(kFlagRepBytes, aid, extra);
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
         "mtype=%s)\n",
         rep[0],
         rep[1],
         kFlagRepBytes,
         mname);
  return true;
}
// ---------------------------------------------------------------------------
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
  // MPK_AID_LOCAL_MAP=1 characterises the run before anything is moved: every
  // input slot, its size, and which AID it sits in. MPK_AID_LOCAL_DRY_RUN=1 then
  // returns without allocating or repointing, so the map describes the untouched
  // baseline. Both run independently of MPK_AID_LOCAL_SLOTS, because the point is
  // to see all 24 allocations, not only the ones already chosen for relocation.
  if (std::getenv("MPK_AID_LOCAL_MAP") != nullptr && !positions.empty()) {
    dump_baseline_map(all_tasks, positions, n_in);
  }
  if (std::getenv("MPK_AID_LOCAL_DRY_RUN") != nullptr) {
    printf("[AID] DRY_RUN: nothing allocated or repointed\n");
    return;
  }
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
  // Diagnostic modes used to separate *why* AID-local placement changes speed:
  //   NO_REPOINT -- allocate and fill the replicas but keep the ORIGINAL
  //                 pointers, so the only thing that changes is the extra
  //                 resident BO footprint (isolates allocation/TLB pressure).
  //   SAME_COPY  -- point all eight XCDs at the AID0 copy: identical buffer
  //                 type to the replicated run, but no locality win (isolates
  //                 buffer memory type from placement).
  // Neither set = normal home-AID placement.
  const bool aid_no_repoint =
      std::getenv("MPK_AID_LOCAL_NO_REPOINT") != nullptr;
  const bool aid_same_copy = std::getenv("MPK_AID_LOCAL_SAME_COPY") != nullptr;
  // Which copy SAME_COPY sends every XCD to: 0 = AID0 copy (repA, allocated
  // without the AID_SELECT bit), 1 = AID1 copy (repB, which sets it). Running
  // SAME_COPY against each copy in turn tests the two buffers against the same
  // access pattern, so a difference between them is a property of the buffer
  // rather than of locality.
  int aid_same_copy_which = 0;
  {
    char const *e = std::getenv("MPK_AID_LOCAL_SAME_COPY_AID");
    if (e != nullptr && e[0] == '1') {
      aid_same_copy_which = 1;
    }
  }
  // SINGLE_REPLICA -- the original bank already lives in one AID, so only the
  // other AID needs a copy. Replicating into both means the AID that already
  // holds the ~62 GiB original also takes ~79 GiB of replica, ~141 GiB of a
  // 144 GiB range, while the other AID sits at ~79 GiB; the full side's reads
  // then get slower even though they are nominally local. Keeping the original
  // as the near copy for its own AID halves the footprint and leaves both
  // ranges with headroom. ORIG_AID says which AID the original is in, measured
  // from the baseline probe (XCD 4-7 read it ~100 ns faster => AID1).
  const bool aid_single_replica =
      std::getenv("MPK_AID_LOCAL_SINGLE_REPLICA") != nullptr;
  int aid_orig_aid = 1;
  {
    char const *e = std::getenv("MPK_AID_LOCAL_ORIG_AID");
    if (e != nullptr && e[0] == '0') {
      aid_orig_aid = 0;
    }
  }
  // Which AID each XCD is treated as belonging to, as eight '0'/'1' characters
  // indexed by XCD ("00001111" = aid_of_xcd, the default). Two independent
  // assumptions feed this and either being backwards would send every XCD to
  // the far copy instead of the near one: the fpfn ordering that decides which
  // aid_aperture range is AID0, and the grouping of *logical* XCD indices --
  // dispatch round-robins over logical indices, which need not match the
  // physical XCC_ID the probe reads. Overriding from the environment lets one
  // build test every mapping instead of guessing.
  int xcd_aid[kNumXcds];
  for (int x = 0; x < kNumXcds; x++) {
    xcd_aid[x] = aid_of_xcd(x);
  }
  {
    char const *e = std::getenv("MPK_AID_LOCAL_XCD_AID");
    size_t n = 0;
    while (e != nullptr && e[n] != 0) {
      n++;
    }
    if (n >= (size_t)kNumXcds) {
      for (int x = 0; x < kNumXcds; x++) {
        xcd_aid[x] = (e[x] == '1') ? 1 : 0;
      }
      printf("[AID] XCD->AID map overridden to %c%c%c%c%c%c%c%c\n",
             e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7]);
    }
  }

  // The MoE packer pools several layers' expert weights into one backing
  // allocation (observed: one ~2.2 GiB segment holds ~2 layers of gate_up, so
  // consecutive layers share a seg_base and differ only by `off`). Replicate
  // each unique segment into each AID exactly ONCE and repoint every layer that
  // lives in it; otherwise a 2-layers-per-segment pool is copied once per layer
  // -- 2x the replica memory it needs. Keyed by segment base.
  struct SegRep {
    void *base;
    void *repA;
    void *repB;
  };
  std::vector<SegRep> seg_reps;

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
        // Shared tensor: every XCD holds the same base, so there is no slice to
        // move, and there is no static per-XCD slice to place either -- which
        // expert an XCD reads is chosen at runtime by TopK. So for a shared MoE
        // weight we (1) always MEASURE its per-XCD locality (the baseline
        // remote-access log), and (2) optionally REPLICATE the whole bank into
        // both AIDs and point each XCD at its home-AID copy, so every expert
        // read is local regardless of routing -- at 2x the bank's memory.
        // Replication is the ceiling measurement (max achievable MoE win) and
        // is gated behind MPK_AID_LOCAL_REPLICATE, so the default stays
        // measure-only. Anything not uniformly shared is left alone (barriers,
        // atomics, queues, KV).
        bool shared = true;
        for (int xcd = 1; xcd < kNumXcds; xcd++) {
          if (static_cast<char *>(all_tasks[pos + xcd].input_ptrs[slot]) !=
              p0) {
            shared = false;
            break;
          }
        }
        if (!shared) {
          if (L == 0) {
            printf("[AID] slot %d stride %td but not uniformly shared; "
                   "skipped\n",
                   slot,
                   stride);
          }
          continue;
        }

        void *seg_base = nullptr;
        size_t seg_size = 0;
        if (hipMemGetAddressRange(reinterpret_cast<hipDeviceptr_t *>(&seg_base),
                                  &seg_size,
                                  reinterpret_cast<hipDeviceptr_t>(p0)) !=
                hipSuccess ||
            seg_base == nullptr || seg_size == 0) {
          if (L == 0) {
            printf("[AID] slot %d: hipMemGetAddressRange failed; left shared\n",
                   slot);
          }
          continue;
        }
        size_t off = static_cast<size_t>(p0 - static_cast<char *>(seg_base));

        // (1) Baseline remote-access log: probe the real buffer per XCD. A
        // shared bank sits in ONE AID, so the four XCDs on the other AID read
        // every byte of it remotely.
        if (aid_probe && L == 0) {
          printf("[AID] slot %d shared MoE weight, %.2f MiB backing:\n",
                 slot,
                 seg_size / 1048576.0);
          // Which AID this lands in is not fixed: it is wherever hipMalloc put
          // it, and it has been observed on either side across runs of the same
          // script. That is why the tag does not name an AID -- read the skew.
          probe_buffer_home("moe baseline (original bank)", p0,
                             seg_size - off);
        }

        // (2) Optional replication (ceiling). Off unless MPK_AID_LOCAL_REPLICATE
        // is set. Places a full copy of the backing segment in AID0 and AID1
        // and points each XCD at its home-AID copy, so every read is local no
        // matter which expert TopK selects. Segments are deduplicated via
        // seg_reps, so the cost is 2x the *unique* MoE weight bytes (~2x62 GiB
        // for all 36 layers, split evenly across the two 144 GiB AIDs) plus the
        // now-unused original -- NOT 2x per layer. A production build would
        // instead keep the original as the near-AID copy and add one far-AID
        // copy; replication is the simplest all-local ceiling measurement.
        if (std::getenv("MPK_AID_LOCAL_REPLICATE") == nullptr) {
          continue;
        }
        void *repA = nullptr;
        void *repB = nullptr;
        bool cached = false;
        for (SegRep const &sr : seg_reps) {
          if (sr.base == seg_base) {
            repA = sr.repA;
            repB = sr.repB;
            cached = true;
            break;
          }
        }
        if (!cached) {
          if (aid_single_replica) {
            // One copy, in the AID the original is NOT in. The original stays
            // in place and serves its own AID's XCDs.
            int const far = 1 - aid_orig_aid;
            repA = alloc_in_aid(seg_size, far);
            if (repA == nullptr) {
              printf("[AID] slot %d: replica alloc (%.2f MiB in AID%d) failed; "
                     "left shared\n",
                     slot,
                     seg_size / 1048576.0,
                     far);
              continue;
            }
            if (hipMemcpy(repA, seg_base, seg_size, hipMemcpyDeviceToDevice) !=
                hipSuccess) {
              printf("[AID] slot %d: replica copy failed\n", slot);
              continue;
            }
            repB = nullptr;
            seg_reps.push_back(SegRep{seg_base, repA, nullptr});
            ctx().bytes_placed += seg_size;
            ctx().buffers_placed += 1;
            printf("[AID] slot %d: SINGLE_REPLICA %.2f MiB segment -> AID%d "
                   "only; AID%d keeps the original (segment #%zu, first seen at "
                   "layer %zu, off %zu)\n",
                   slot,
                   seg_size / 1048576.0,
                   far,
                   aid_orig_aid,
                   seg_reps.size(),
                   L,
                   off);
          } else {
            repA = alloc_in_aid(seg_size, 0);
            repB = alloc_in_aid(seg_size, 1);
            if (repA == nullptr || repB == nullptr) {
              printf("[AID] slot %d: replica alloc (2 x %.2f MiB) failed; left "
                     "shared\n",
                     slot,
                     seg_size / 1048576.0);
              continue;
            }
            if (hipMemcpy(repA, seg_base, seg_size, hipMemcpyDeviceToDevice) !=
                    hipSuccess ||
                hipMemcpy(repB, seg_base, seg_size, hipMemcpyDeviceToDevice) !=
                    hipSuccess) {
              printf("[AID] slot %d: replica copy failed\n", slot);
              continue;
            }
            seg_reps.push_back(SegRep{seg_base, repA, repB});
            ctx().bytes_placed += 2 * seg_size;
            ctx().buffers_placed += 2;
            printf("[AID] slot %d: replicated %.2f MiB segment to AID0+AID1 "
                   "(new segment #%zu, first seen at layer %zu, off %zu)\n",
                   slot,
                   seg_size / 1048576.0,
                   seg_reps.size(),
                   L,
                   off);
          }
        }
        // Repoint this layer's XCDs. Normally each XCD goes to its home-AID
        // copy. In SAME_COPY mode every XCD is pointed at the AID0 copy
        // instead: the buffers, their memory type and the total footprint are
        // all identical to the replicated run, and the ONLY difference is that
        // XCD 4-7 now read across the AID boundary. So SAME_COPY vs replicated
        // is a pure locality A/B. `off` is this layer's offset in the segment.
        if (aid_no_repoint) {
          if (L == 0) {
            printf("[AID] slot %d: NO_REPOINT -- replicas allocated but "
                   "original pointers kept\n",
                   slot);
          }
          continue;
        }
        if (aid_same_copy && L == 0) {
          printf("[AID] slot %d: SAME_COPY -- all 8 XCDs -> AID%d copy "
                 "(locality removed, buffer type unchanged)\n",
                 slot,
                 aid_same_copy_which);
        }
        void *const same_rep = (aid_same_copy_which == 0) ? repA : repB;
        for (int xcd = 0; xcd < kNumXcds; xcd++) {
          int const want = aid_same_copy ? aid_same_copy_which : xcd_aid[xcd];
          if (aid_single_replica && want == aid_orig_aid) {
            // Already reading its own AID via the original pointer.
            if (L == 0) {
              printf("[AID]   slot %d XCD %d -> original (AID%d), unchanged\n",
                     slot,
                     xcd,
                     aid_orig_aid);
            }
            continue;
          }
          void *rep = aid_same_copy      ? same_rep
                      : aid_single_replica ? repA
                                         : (want == 0 ? repA : repB);
          all_tasks[pos + xcd].input_ptrs[slot] =
              static_cast<char *>(rep) + off;
          // Spell out the resulting assignment. "Placed 130 buffers" says the
          // allocation worked; it does not say each XCD ended up on the copy
          // intended for it, which is the part that was wrong.
          if (L == 0) {
            printf("[AID]   slot %d XCD %d -> copy %c (AID%d) at %p\n",
                   slot,
                   xcd,
                   aid_single_replica ? 'R' : (want == 0 ? 'A' : 'B'),
                   want,
                   all_tasks[pos + xcd].input_ptrs[slot]);
          }
        }
        if (aid_probe && L == 0) {
          // Time each replica from all eight XCDs, separately. Probing the split
          // pointer set would have XCD 0-3 on repA and XCD 4-7 on repB, and the
          // group skew would then mix the two placements instead of locating
          // either. Read together, these two rows prove the assignment: repA
          // should favour XCD 0-3 and repB should favour XCD 4-7.
          probe_buffer_home("replica A (intended AID0)", repA, seg_size - off);
          if (repB != nullptr) {
            probe_buffer_home("replica B (intended AID1)", repB, seg_size - off);
          }
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
        void *dst = alloc_in_aid((size_t)stride, xcd_aid[xcd]);
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
  Ctx &cc = ctx();
  printf("[AID] placed %d buffers, %.2f GiB total, across %zu fused layers\n",
         cc.buffers_placed,
         cc.bytes_placed / 1073741824.0,
         positions.size());
  // The split between the two AIDs matters more than the total. The model's own
  // hipMalloc weights already occupy one of the two ranges and are not counted
  // here, so whichever AID also receives the replicas can be near full while
  // the other is half empty -- and only the full one's reads get slower, which
  // reads as "locality made things worse" rather than "one range ran out".
  for (int a = 0; a < 2; a++) {
    double cap = cc.range_bytes[a] / 1073741824.0;
    double used = cc.bytes_by_aid[a] / 1073741824.0;
    printf("[AID] AID%d footprint: %d BOs, %.2f of %.0f GiB (%.0f%%)\n",
           a,
           cc.bufs_by_aid[a],
           used,
           cap,
           cap > 0.0 ? 100.0 * used / cap : 0.0);
  }
  {
    double u0 = cc.bytes_by_aid[0] / 1073741824.0;
    double u1 = cc.bytes_by_aid[1] / 1073741824.0;
    double hi = u0 > u1 ? u0 : u1, lo = u0 > u1 ? u1 : u0;
    if (hi > 0.0 && lo < 0.8 * hi) {
      printf("[AID] !! AID footprint is lopsided (%.2f vs %.2f GiB) -- the "
             "fuller range's reads are the ones to suspect\n",
             u0,
             u1);
    }
    size_t free_b = 0, total_b = 0;
    if (hipMemGetInfo(&free_b, &total_b) == hipSuccess) {
      printf("[AID] device memory after placement: %.1f GiB free of %.1f GiB\n",
             free_b / 1073741824.0,
             total_b / 1073741824.0);
    }
  }
  fflush(stdout);
}

} // namespace aid
} // namespace mirage

#endif // ROCm
