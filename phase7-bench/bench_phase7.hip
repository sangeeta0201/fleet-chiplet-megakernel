// Standalone reproducer for the Phase 7 cross-XCD attention-slice wait.
//
// Mirrors the structure of gang_linear_mxfp4_res_bias_rmsnorm_topk's first op:
//   - attn_out is [1, 4096] bf16 = 8 KiB, split into 8 slices of 512 bf16.
//   - XCD x produces slice x, then publishes attn_release[x*16] with a
//     write-through store (st_wt_u32, sc0 sc1).
//   - Every XCD consumes all 8 slices: wave w spins on flags 2w and 2w+1 with
//     a system-scope load (ld_sys_s32, sc0 sc1), acquires, then reads 2 KiB.
//   - A per-layer global rendezvous keeps the 8 XCDs in lockstep, as the
//     fused kernel's phase-9 barrier does.
//
// The point of the delay knobs is to separate the two things that can make the
// spin long: flag visibility latency (delay 0 -> everyone arrives together) and
// producer arrival skew (delay_skew > 0 -> XCD x releases later than XCD x-1).
//
// s_memrealtime ticks at 100 MHz, so one tick is 10 ns. That is the same
// conversion the in-kernel [OPROJ_INNER] printf uses, so the numbers here are
// directly comparable to it.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <drm/amdgpu_drm.h>
#include <drm/drm.h>

#define NXCD 8
#define NWAVES 4
#define NTHREADS (NWAVES * 64)
#define ATTN_SLICE 512 // bf16 elements produced by one XCD
#define REDUCTION (NXCD * ATTN_SLICE)
#define FLAG_STRIDE 16 // ints; 64 B apart, matching attn_release[x * 16]

#define HIP_OK(call)                                                           \
  do {                                                                         \
    hipError_t _e = (call);                                                    \
    if (_e != hipSuccess) {                                                     \
      fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #call,           \
              hipGetErrorString(_e));                                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// --------------------------------------------------------------------------
// AID-local allocator, same GEM_CREATE path as bench_aidsplit.hip.
//
// Only the *synchronization* objects are placed and promoted here: the eight
// release flags and the per-layer rendezvous counter. attn_out deliberately
// stays on a plain hipMalloc, because every wave reads slices produced by XCDs
// in both AIDs -- it is genuinely cross-AID data, so a coherent MTYPE on it
// would be the unsafe case (see ROOT-CAUSE.md, "Where this stops working").
// --------------------------------------------------------------------------

static constexpr uint64_t GEM_CREATE_AID_LOCAL = 1ULL << 17;
static constexpr uint64_t GEM_CREATE_AID_SELECT = 1ULL << 18;
static constexpr uint64_t GEM_CREATE_COHERENT = 1ULL << 13;

static int g_drm_fd = -1;
static unsigned g_range_for_aid[2] = {~0u, ~0u};
static bool g_coherent = false;

static int open_render_node(char const *pci) {
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

// Group the aid_aperture ranges by first pfn against the midpoint of the
// covered span, never by index; the index order is per-GPU.
static bool resolve_ranges(char const *pci) {
  char sp[256];
  snprintf(sp, sizeof sp, "/sys/bus/pci/devices/%s/aid_aperture", pci);
  FILE *f = fopen(sp, "r");
  if (f == nullptr) {
    printf("[AID] %s unreadable -- stock driver or not NPS2\n", sp);
    return false;
  }
  uint64_t sizes[4] = {0, 0, 0, 0}, fpfns[4] = {0, 0, 0, 0};
  char line[256];
  while (fgets(line, sizeof line, f) != nullptr) {
    unsigned idx;
    uint64_t fp, lp, sz;
    if (sscanf(line, "aid%u fpfn 0x%lx lpfn 0x%lx size 0x%lx", &idx, &fp, &lp,
               &sz) == 4 &&
        idx < 4) {
      sizes[idx] = sz;
      fpfns[idx] = fp;
    }
  }
  fclose(f);
  uint64_t max_end = 0;
  for (int r = 0; r < 4; r++) {
    if (sizes[r]) {
      uint64_t e = fpfns[r] + (sizes[r] >> 12);
      max_end = e > max_end ? e : max_end;
    }
  }
  if (!max_end) {
    return false;
  }
  uint64_t const boundary = max_end / 2;
  uint64_t best_lo = 0, best_hi = 0;
  for (int r = 0; r < 4; r++) {
    if (!sizes[r]) {
      continue;
    }
    if (fpfns[r] < boundary) {
      if (sizes[r] > best_lo) {
        best_lo = sizes[r];
        g_range_for_aid[0] = (unsigned)r;
      }
    } else {
      if (sizes[r] > best_hi) {
        best_hi = sizes[r];
        g_range_for_aid[1] = (unsigned)r;
      }
    }
  }
  if (g_range_for_aid[0] == ~0u || g_range_for_aid[1] == ~0u) {
    printf("[AID] could not pick one range per AID\n");
    return false;
  }
  printf("[AID] AID0 -> range %u, AID1 -> range %u\n", g_range_for_aid[0],
         g_range_for_aid[1]);
  return true;
}

static bool aid_init() {
  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess) {
    return false;
  }
  char bus[64] = {0};
  if (hipDeviceGetPCIBusId(bus, sizeof bus, dev) != hipSuccess) {
    return false;
  }
  for (char *p = bus; *p; p++) {
    if (*p >= 'A' && *p <= 'F') {
      *p = (char)(*p - 'A' + 'a');
    }
  }
  if (!resolve_ranges(bus)) {
    return false;
  }
  g_drm_fd = open_render_node(bus);
  if (g_drm_fd < 0) {
    printf("[AID] no render node for %s\n", bus);
    return false;
  }
  return true;
}

static void *alloc_in_aid(size_t bytes, int aid) {
  uint64_t const flags = GEM_CREATE_AID_LOCAL |
                         (g_range_for_aid[aid] & 1u ? GEM_CREATE_AID_SELECT : 0) |
                         (g_coherent ? GEM_CREATE_COHERENT : 0);
  union drm_amdgpu_gem_create req;
  memset(&req, 0, sizeof req);
  req.in.bo_size = bytes;
  req.in.alignment = 2ULL << 20;
  req.in.domains = AMDGPU_GEM_DOMAIN_VRAM;
  req.in.domain_flags = flags;
  if (ioctl(g_drm_fd, DRM_IOCTL_AMDGPU_GEM_CREATE, &req) != 0) {
    printf("[AID] GEM_CREATE %zu B flags 0x%llx failed: %s\n", bytes,
           (unsigned long long)flags, strerror(errno));
    return nullptr;
  }
  struct drm_prime_handle prime;
  memset(&prime, 0, sizeof prime);
  prime.handle = req.out.handle;
  if (ioctl(g_drm_fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &prime) != 0) {
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
  printf("[AID] %.2f MiB in AID%d (range %u, COHERENT %s) -> %p\n",
         bytes / 1048576.0, aid, g_range_for_aid[aid],
         (flags & GEM_CREATE_COHERENT) ? "set" : "clear", ptr);
  return ptr;
}

// The rendezvous counter shares the AID-local buffer with the flags rather
// than getting its own, at a 1 MiB offset so it cannot land on a flag line.
#define BAR_OFF_INTS (1 << 18) // 1 MiB / 4

__device__ __forceinline__ int ld_sys_s32(int *addr) {
  int val;
  asm volatile("global_load_dword %0, %1, off sc0 sc1\n"
               "s_waitcnt vmcnt(0)"
               : "=v"(val)
               : "v"(addr)
               : "memory");
  return val;
}

__device__ __forceinline__ void st_wt_u32(void *addr, unsigned int val) {
  asm volatile("global_store_dword %0, %1, off sc0 sc1"
               :
               : "v"(addr), "v"(val)
               : "memory");
}

__device__ __forceinline__ int get_xcd() {
  int x;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(x));
  return x;
}

// One block per XCD; report where each block actually landed so the harness can
// refuse to measure a run whose blocks collided onto fewer than 8 XCDs.
__global__ void probe_xcd(int *out) {
  if (threadIdx.x == 0) {
    out[blockIdx.x] = get_xcd();
  }
}

__global__ __launch_bounds__(NTHREADS) void phase7(
    unsigned short *attn_out, int *flags_a, int *flags_b, unsigned int *bar_a,
    unsigned int *bar_b, unsigned long long *samp_wait,
    unsigned long long *samp_read, unsigned int *sink, int n_layers,
    int delay_ticks, int delay_skew, int dual, int split_on, int tiles,
    int *seen) {
  int const xcd = get_xcd();
  int const tid = threadIdx.x;
  int const warp = tid >> 6;
  int const lane = tid & 63;

  // Production Phase 7 runs `tiles` O-proj workgroups per XCD and every one of
  // them polls the same eight flags, which is where the fan-out that drives the
  // regression comes from. SPX dispatch round-robins block i to XCD i%8, so the
  // blocks sharing a `local` are one per XCD.
  int const local = (int)blockIdx.x / NXCD;
  if (tid == 0) {
    seen[blockIdx.x] = xcd;
  }

  // Which copy of the synchronization state this XCD reads. With split_on the
  // upper half of the XCDs polls the copy homed in AID1 and the lower half the
  // one in AID0, so every reader is co-located with the line it polls, which
  // is the precondition that makes a coherent MTYPE sound here.
  bool const upper = split_on && xcd >= NXCD / 2;
  int *flags = upper ? flags_b : flags_a;
  unsigned int *bar = upper ? bar_b : bar_a;

  // Wave w covers K elements [1024w, 1024w+1024), which is exactly slices
  // 2w and 2w+1 -- the same forced mapping as the real kernel.
  int const first_xcd = warp * 2;
  int *rel0 = &flags[first_xcd * FLAG_STRIDE];
  int *rel1 = &flags[(first_xcd + 1) * FLAG_STRIDE];
  unsigned int const *src =
      (unsigned int const *)(attn_out + first_xcd * ATTN_SLICE);
  unsigned int *my_slice = (unsigned int *)(attn_out + xcd * ATTN_SLICE);

  int const row = ((xcd * tiles + local) * NWAVES + warp) * n_layers;

  unsigned int checksum = 0;

  for (int layer = 1; layer <= n_layers; ++layer) {
    // ?? producer: model the attention/merge that precedes the release ??
    if (delay_ticks || delay_skew) {
      unsigned long long const until =
          __builtin_amdgcn_s_memrealtime() +
          (unsigned long long)(delay_ticks + delay_skew * xcd);
      while (__builtin_amdgcn_s_memrealtime() < until) {
        __builtin_amdgcn_s_sleep(1);
      }
    }
    // One producer per XCD, as in the real kernel where the Phase 6 merge
    // publishes the slice once. The other `tiles - 1` workgroups are pure
    // consumers.
    if (local == 0) {
      // 256 threads x 1 dword = 256 dwords = 512 bf16 = this XCD's slice.
      st_wt_u32(&my_slice[tid], (unsigned)(layer * 2654435761u + xcd));
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }
    __syncthreads();
    if (local == 0 && tid == 0) {
      // Publish into every copy. The value is the same in both, so the
      // information a consumer sees is identical no matter which it polls; only
      // the set of readers per line changes. A store from a far-AID XCD still
      // invalidates the sharers co-located with the line, which is what makes
      // one producer able to release both halves.
      st_wt_u32(&flags_a[xcd * FLAG_STRIDE], (unsigned)layer);
      if (dual) {
        st_wt_u32(&flags_b[xcd * FLAG_STRIDE], (unsigned)layer);
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }

    // ?? consumer: the span the real kernel calls `slicewait` ??
    unsigned long long const t0 = __builtin_amdgcn_s_memrealtime();
    while (ld_sys_s32(rel0) < layer || ld_sys_s32(rel1) < layer) {
      __builtin_amdgcn_s_sleep(1);
    }
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    asm volatile("buffer_inv" ::: "memory");
    unsigned long long const t0b = __builtin_amdgcn_s_memrealtime();

    // ?? the 2 KiB this wave then actually reads ??
    unsigned int acc = 0;
#pragma unroll
    for (int i = 0; i < (2 * ATTN_SLICE) / 2 / 64; ++i) {
      acc += src[lane + i * 64];
    }
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    unsigned long long const t1 = __builtin_amdgcn_s_memrealtime();
    checksum += acc;

    if (lane == 0) {
      samp_wait[row + layer - 1] = t0b - t0;
      samp_read[row + layer - 1] = t1 - t0b;
    }

    // ?? per-layer global rendezvous across all 8 XCDs ??
    __syncthreads();
    if (tid == 0) {
      // Same treatment as the flags: every XCD increments every copy, so each
      // copy independently reaches layer * NXCD and a consumer can wait on the
      // one in its own AID. Without `dual` this must increment once only, or
      // the count would run ahead of the arrivals it is supposed to represent.
      atomicAdd(bar_a, 1u);
      if (dual) {
        atomicAdd(bar_b, 1u);
      }
      while (ld_sys_s32((int *)bar) < layer * NXCD * tiles) {
        __builtin_amdgcn_s_sleep(1);
      }
    }
    __syncthreads();
  }

  // Keep the loads from being folded away without perturbing the timed spans.
  if (checksum == 0xdeadbeefu && lane == 0) {
    sink[0] = checksum;
  }
}

static double pct(std::vector<unsigned long long> &v, double p) {
  if (v.empty()) {
    return 0.0;
  }
  size_t i = (size_t)(p * (double)(v.size() - 1));
  return (double)v[i] * 10.0 / 1000.0; // ticks -> us
}

int main(int argc, char **argv) {
  int n_layers = 3600;
  int delay_ticks = 0;
  int delay_skew = 0;
  int use_aid = 0;
  int split_on = 1;
  int tiles = 23; // O-proj workgroups per XCD in production
  char const *tag = "run";
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--tiles=", 8)) {
      tiles = atoi(argv[i] + 8);
    } else if (!strncmp(argv[i], "--layers=", 9)) {
      n_layers = atoi(argv[i] + 9);
    } else if (!strncmp(argv[i], "--delay=", 8)) {
      delay_ticks = atoi(argv[i] + 8);
    } else if (!strncmp(argv[i], "--skew=", 7)) {
      delay_skew = atoi(argv[i] + 7);
    } else if (!strncmp(argv[i], "--aid=", 6)) {
      use_aid = atoi(argv[i] + 6);
    } else if (!strncmp(argv[i], "--coherent=", 11)) {
      g_coherent = atoi(argv[i] + 11) != 0;
    } else if (!strncmp(argv[i], "--split=", 8)) {
      split_on = atoi(argv[i] + 8);
    } else if (!strncmp(argv[i], "--tag=", 6)) {
      tag = argv[i] + 6;
    }
  }

  int *d_probe = nullptr;
  HIP_OK(hipMalloc(&d_probe, NXCD * sizeof(int)));
  HIP_OK(hipMemset(d_probe, 0xff, NXCD * sizeof(int)));
  hipLaunchKernelGGL(probe_xcd, dim3(NXCD), dim3(64), 0, 0, d_probe);
  HIP_OK(hipDeviceSynchronize());
  int probe[NXCD];
  HIP_OK(hipMemcpy(probe, d_probe, sizeof(probe), hipMemcpyDeviceToHost));
  bool seen[NXCD] = {false};
  printf("block -> xcd:");
  for (int i = 0; i < NXCD; ++i) {
    printf(" %d", probe[i]);
    if (probe[i] >= 0 && probe[i] < NXCD) {
      seen[probe[i]] = true;
    }
  }
  printf("\n");
  for (int i = 0; i < NXCD; ++i) {
    if (!seen[i]) {
      fprintf(stderr,
              "ABORT: no block landed on XCD %d; the 8 blocks did not spread "
              "one per XCD, so the producer set is incomplete and the spin "
              "would never retire.\n",
              i);
      return 2;
    }
  }

  unsigned short *d_attn = nullptr;
  int *d_flags_a = nullptr, *d_flags_b = nullptr;
  unsigned int *d_bar_a = nullptr, *d_bar_b = nullptr, *d_sink = nullptr;
  unsigned long long *d_wait = nullptr, *d_read = nullptr;
  int const nblocks = NXCD * tiles;
  size_t const nsamp = (size_t)nblocks * NWAVES * n_layers;
  int *d_seen = nullptr;
  HIP_OK(hipMalloc(&d_seen, nblocks * sizeof(int)));
  HIP_OK(hipMemset(d_seen, 0xff, nblocks * sizeof(int)));
  // attn_out stays a plain hipMalloc on purpose: it is read across the AID
  // boundary by construction, so MTYPE_NC is the only correct mapping for it.
  HIP_OK(hipMalloc(&d_attn, REDUCTION * sizeof(unsigned short)));
  HIP_OK(hipMalloc(&d_sink, sizeof(unsigned int)));
  HIP_OK(hipMalloc(&d_wait, nsamp * sizeof(unsigned long long)));
  HIP_OK(hipMalloc(&d_read, nsamp * sizeof(unsigned long long)));
  HIP_OK(hipMemset(d_attn, 0, REDUCTION * sizeof(unsigned short)));
  HIP_OK(hipMemset(d_sink, 0, sizeof(unsigned int)));

  int dual = 0;
  size_t const sync_bytes = 2ull << 20;
  if (use_aid) {
    if (!aid_init()) {
      fprintf(stderr, "ABORT: --aid=1 but AID-local allocation is "
                      "unavailable (stock driver, or not NPS2)\n");
      return 2;
    }
    d_flags_a = (int *)alloc_in_aid(sync_bytes, 0);
    d_flags_b = (int *)alloc_in_aid(sync_bytes, 1);
    if (d_flags_a == nullptr || d_flags_b == nullptr) {
      fprintf(stderr, "ABORT: could not place a sync buffer in each AID\n");
      return 2;
    }
    d_bar_a = (unsigned int *)(d_flags_a + BAR_OFF_INTS);
    d_bar_b = (unsigned int *)(d_flags_b + BAR_OFF_INTS);
    HIP_OK(hipMemset(d_flags_a, 0, sync_bytes));
    HIP_OK(hipMemset(d_flags_b, 0, sync_bytes));
    dual = 1;
  } else {
    // Baseline: exactly what the kernel does today, one hipMalloc'd flag page
    // and one counter, both MTYPE_NC and both read from both AIDs.
    HIP_OK(hipMalloc(&d_flags_a, 4096));
    HIP_OK(hipMalloc(&d_bar_a, sizeof(unsigned int)));
    HIP_OK(hipMemset(d_flags_a, 0, 4096));
    HIP_OK(hipMemset(d_bar_a, 0, sizeof(unsigned int)));
    d_flags_b = d_flags_a;
    d_bar_b = d_bar_a;
    split_on = 0;
  }

  printf("[%s] aid=%d coherent=%d split=%d  %d tiles/XCD = %d blocks, %d "
         "polling waves, %d layers, delay=%d skew=%d ticks\n",
         tag, use_aid, (int)g_coherent, split_on, tiles, nblocks,
         nblocks * NWAVES, n_layers, delay_ticks, delay_skew);
  printf("      attn_out %p (NC, cross-AID by design)  flags %p / %p\n",
         (void *)d_attn, (void *)d_flags_a, (void *)d_flags_b);

  hipEvent_t e0, e1;
  HIP_OK(hipEventCreate(&e0));
  HIP_OK(hipEventCreate(&e1));
  HIP_OK(hipEventRecord(e0));
  hipLaunchKernelGGL(phase7, dim3(nblocks), dim3(NTHREADS), 0, 0, d_attn,
                     d_flags_a, d_flags_b, d_bar_a, d_bar_b, d_wait, d_read,
                     d_sink, n_layers, delay_ticks, delay_skew, dual, split_on,
                     tiles, d_seen);
  HIP_OK(hipEventRecord(e1));
  HIP_OK(hipDeviceSynchronize());
  HIP_OK(hipGetLastError());
  float ms = 0.0f;
  HIP_OK(hipEventElapsedTime(&ms, e0, e1));

  std::vector<unsigned long long> h_wait(nsamp), h_read(nsamp);
  HIP_OK(hipMemcpy(h_wait.data(), d_wait, nsamp * sizeof(unsigned long long),
                   hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(h_read.data(), d_read, nsamp * sizeof(unsigned long long),
                   hipMemcpyDeviceToHost));

  // Discard the first 10% of layers as warmup.
  int const skip = n_layers / 10;
  printf("\n[%s] wall %.2f ms for %d layers = %.2f us/layer\n", tag, ms,
         n_layers, ms * 1000.0 / n_layers);

  std::vector<int> h_seen(nblocks);
  HIP_OK(hipMemcpy(h_seen.data(), d_seen, nblocks * sizeof(int),
                   hipMemcpyDeviceToHost));
  int rr_ok = 0;
  for (int b = 0; b < nblocks; ++b) {
    if (h_seen[b] == b % NXCD) {
      ++rr_ok;
    }
  }
  printf("[%s] block->XCD round-robin holds for %d of %d blocks\n", tag, rr_ok,
         nblocks);

  std::vector<unsigned long long> all_w, all_r;
  printf("\n%-14s", "consumer XCD");
  for (int w = 0; w < NWAVES; ++w) {
    printf("  wave%d(xcd%d,%d)", w, w * 2, w * 2 + 1);
  }
  printf("   (median over %d tiles)\n", tiles);
  for (int x = 0; x < NXCD; ++x) {
    printf("  xcd%d wait us ", x);
    for (int w = 0; w < NWAVES; ++w) {
      std::vector<unsigned long long> v;
      for (int t = 0; t < tiles; ++t) {
        for (int l = skip; l < n_layers; ++l) {
          v.push_back(
              h_wait[(size_t)((x * tiles + t) * NWAVES + w) * n_layers + l]);
        }
      }
      std::sort(v.begin(), v.end());
      printf("  %13.2f", pct(v, 0.5));
      all_w.insert(all_w.end(), v.begin(), v.end());
    }
    printf("\n");
  }
  for (int x = 0; x < NXCD; ++x) {
    printf("  xcd%d read us ", x);
    for (int w = 0; w < NWAVES; ++w) {
      std::vector<unsigned long long> v;
      for (int t = 0; t < tiles; ++t) {
        for (int l = skip; l < n_layers; ++l) {
          v.push_back(
              h_read[(size_t)((x * tiles + t) * NWAVES + w) * n_layers + l]);
        }
      }
      std::sort(v.begin(), v.end());
      printf("  %13.2f", pct(v, 0.5));
      all_r.insert(all_r.end(), v.begin(), v.end());
    }
    printf("\n");
  }

  std::sort(all_w.begin(), all_w.end());
  std::sort(all_r.begin(), all_r.end());
  printf("\n[%s] slicewait us  p10 %.2f  p50 %.2f  p90 %.2f  max %.2f\n", tag,
         pct(all_w, 0.10), pct(all_w, 0.50), pct(all_w, 0.90),
         pct(all_w, 1.00));
  printf("[%s] read us       p10 %.2f  p50 %.2f  p90 %.2f  max %.2f\n", tag,
         pct(all_r, 0.10), pct(all_r, 0.50), pct(all_r, 0.90),
         pct(all_r, 1.00));
  return 0;
}
