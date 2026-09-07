// Can SPX+NPS2 reach NPS1 flag latency by giving each half of the XCDs its own
// AID-local copy of the release flags?
//
// This builds on three things measured by bench_hangdiag:
//
//   1. Coherence works exactly among the XCDs co-located with a line's home
//      AID -- XCD0-3 with AID0, XCD4-7 with AID1 -- and nowhere else. Which
//      half is coherent follows the page, not the silicon: it flips when the
//      XCP claims the other range.
//   2. A store from a far-AID XCD still correctly invalidates the sharers that
//      ARE co-located with the line. xcd0's store was observed by xcd4-7 even
//      though they held the line cached from before it.
//   3. The driver's is_local test includes
//      (is_vram && (bo->flags & AMDGPU_GEM_CREATE_AID_LOCAL)), so an AID-local
//      BO is stamped with mtype_local -- a coherent MTYPE -- while ordinary
//      hipMalloc pages in the same process stay MTYPE_NC.
//
// Together those say the earlier per-BO coherent-MTYPE attempt failed for a
// fixable reason: it made one flag BO coherent but still had all eight XCDs
// polling it, so half were outside the domain. Put a copy in each AID, have
// every producer write both, and let each consumer poll only its own AID's
// copy, and every reader is in-domain. The L2 may then legitimately answer the
// poll, and the queue of 184 pollers at one home channel disappears.
//
// --split=0 runs the same kernel with every XCD polling the AID0 copy, which is
// the out-of-domain case and should hang for XCD4-7. That is the control: it
// shows the win comes from placement and not from the allocator.

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <algorithm>
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

#define HIP_OK(call)                                                           \
  do {                                                                         \
    hipError_t _e = (call);                                                    \
    if (_e != hipSuccess) {                                                    \
      fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #call,           \
              hipGetErrorString(_e));                                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// --------------------------------------------------------------------------
// Minimal AID-local allocator, the same GEM_CREATE path as
// nps2_work/aid_local.h, reduced to what this benchmark needs.
// --------------------------------------------------------------------------

static constexpr uint64_t GEM_CREATE_AID_LOCAL = 1ULL << 17;
static constexpr uint64_t GEM_CREATE_AID_SELECT = 1ULL << 18;
// The driver's coherence test is bo->flags & (COHERENT | EXT_COHERENT), and
// COHERENT is in AMDGPU_GEM_CREATE_SETTABLE_MASK, so a BO can ask for both a
// home AID and a coherent MTYPE in one GEM_CREATE. With aid_local_flag_mtype=2
// that is per-BO coherence: these flags become MTYPE_CC while every ordinary
// hipMalloc page in the same process stays MTYPE_NC.
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
      printf("[AID] range %u: fpfn 0x%lx size %.1f GiB\n", idx, fp,
             sz / 1073741824.0);
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
  printf("[AID] %s ready\n", bus);
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
  printf("[AID] %.2f MiB in AID%d (range %u, SELECT %s, COHERENT %s) -> %p\n",
         bytes / 1048576.0, aid, g_range_for_aid[aid],
         (flags & GEM_CREATE_AID_SELECT) ? "set" : "clear",
         (flags & GEM_CREATE_COHERENT) ? "set" : "clear", ptr);
  return ptr;
}

// --------------------------------------------------------------------------

// Everything a consumer reads has to live in its own AID, or the experiment
// measures the wrong thing. The release timestamps are read cross-XCD exactly
// like the flags, so they are carried inside the same AID-local buffers rather
// than a plain hipMalloc allocation: flags at offset 0, timestamps at 1 MiB.
#define RELTS_OFF_INTS (1 << 18) // 1 MiB / 4
#define RELTS_STRIDE 64

__device__ __forceinline__ int get_xcd() {
  int x;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(x));
  return x;
}

__device__ __forceinline__ void st_wt_u32(void *addr, unsigned int val) {
  asm volatile("global_store_dword %0, %1, off sc0 sc1"
               :
               : "v"(addr), "v"(val)
               : "memory");
}

__device__ __forceinline__ int ld_sys_s32(int *addr) {
  int v;
  asm volatile("global_load_dword %0, %1, off sc0 sc1\n"
               "s_waitcnt vmcnt(0)"
               : "=v"(v)
               : "v"(addr)
               : "memory");
  return v;
}

// Layers are paced by the wall clock rather than by a cross-block barrier. A
// shared barrier line would be exactly the cross-AID spin this is trying to
// avoid, and would hang the run under a coherent MTYPE for reasons that have
// nothing to do with the flags. Every block reads the same 100 MHz reference,
// so absolute deadlines keep them in step without any shared memory at all.
__global__ __launch_bounds__(NTHREADS) void aidsplit(
    int *fa, int *fb, unsigned int *s_vis, int *seen, int n_layers, int tiles,
    int stride_ints, int split_on, unsigned int period, unsigned int pdelay,
    unsigned long long bail) {
  int const xcd = get_xcd();
  int const tid = threadIdx.x;
  int const warp = tid >> 6;
  int const lane = tid & 63;

  // Rank comes from blockIdx, not an atomic claim. In SPX dispatch round-robins
  // block i to XCD i%8, and a cross-XCD atomic is itself subject to the
  // coherence failure under test -- if it mis-assigned ranks there would be two
  // producers per XCD, or none, and the run would fail for the wrong reason.
  int const local = (int)blockIdx.x / NXCD;
  if (local >= tiles) {
    return;
  }
  if (tid == 0) {
    seen[blockIdx.x] = xcd; // verify the round-robin actually holds
  }

  // The whole point: which copy this XCD reads. split_on routes each half to
  // the copy that lives in its own AID; split_on=0 sends everyone to AID0's.
  int *mine = (split_on && xcd >= NXCD / 2) ? fb : fa;
  unsigned int *my_ts = (unsigned int *)(mine + RELTS_OFF_INTS);
  int const first_xcd = warp * 2;
  int const row = ((xcd * tiles + local) * NWAVES + warp) * n_layers;

  unsigned long long const t_base = __builtin_amdgcn_s_memrealtime();

  for (int layer = 1; layer <= n_layers; ++layer) {
    unsigned long long const start = t_base + (unsigned long long)layer * period;
    while (__builtin_amdgcn_s_memrealtime() < start) {
      __builtin_amdgcn_s_sleep(2);
    }

    if (local == 0 && tid == 0) {
      while (__builtin_amdgcn_s_memrealtime() < start + pdelay) {
        __builtin_amdgcn_s_sleep(1);
      }
      unsigned int const now = (unsigned)__builtin_amdgcn_s_memrealtime();
      st_wt_u32(&fa[RELTS_OFF_INTS + xcd * RELTS_STRIDE], now);
      st_wt_u32(&fb[RELTS_OFF_INTS + xcd * RELTS_STRIDE], now);
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      // Both copies always carry the same value; only which one a consumer
      // reads changes.
      st_wt_u32(&fa[xcd * stride_ints], (unsigned)layer);
      st_wt_u32(&fb[xcd * stride_ints], (unsigned)layer);
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }

    unsigned long long const deadline =
        __builtin_amdgcn_s_memrealtime() + bail;
    int timed_out = 0;
    for (int k = 0; k < 2; ++k) {
      int *p = &mine[(first_xcd + k) * stride_ints];
      while (ld_sys_s32(p) < layer) {
        if (__builtin_amdgcn_s_memrealtime() > deadline) {
          timed_out = 1;
          break;
        }
        __builtin_amdgcn_s_sleep(1);
      }
      if (timed_out) {
        break;
      }
    }
    unsigned long long const t1 = __builtin_amdgcn_s_memrealtime();

    if (lane == 0) {
      if (timed_out) {
        s_vis[row + layer - 1] = 0xffffffffu;
      } else {
        unsigned int const r0 =
            (unsigned int)ld_sys_s32((int *)&my_ts[first_xcd * RELTS_STRIDE]);
        unsigned int const r1 = (unsigned int)ld_sys_s32(
            (int *)&my_ts[(first_xcd + 1) * RELTS_STRIDE]);
        unsigned int const rmax = ((int)(r1 - r0) > 0) ? r1 : r0;
        s_vis[row + layer - 1] = (unsigned int)t1 - rmax;
      }
    }
    if (timed_out) {
      break;
    }
  }
}

int main(int argc, char **argv) {
  int tiles = 23, n_layers = 120, split_on = 1;
  long stride_bytes = 64;
  double bail_ms = 50.0;
  unsigned int period = 20000; // 200 us between layers
  unsigned int pdelay = 2000;  // producer publishes 20 us in
  char const *tag = "aidsplit";
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--tiles=", 8)) {
      tiles = atoi(argv[i] + 8);
    } else if (!strncmp(argv[i], "--layers=", 9)) {
      n_layers = atoi(argv[i] + 9);
    } else if (!strncmp(argv[i], "--flagstride=", 13)) {
      stride_bytes = atol(argv[i] + 13);
    } else if (!strncmp(argv[i], "--split=", 8)) {
      split_on = atoi(argv[i] + 8);
    } else if (!strncmp(argv[i], "--bailms=", 9)) {
      bail_ms = atof(argv[i] + 9);
    } else if (!strncmp(argv[i], "--coherent=", 11)) {
      g_coherent = atoi(argv[i] + 11) != 0;
    } else if (!strncmp(argv[i], "--tag=", 6)) {
      tag = argv[i] + 6;
    }
  }
  int const stride_ints = (int)(stride_bytes / 4);
  int const nblocks = NXCD * tiles;
  size_t const nsamp = (size_t)NXCD * tiles * NWAVES * n_layers;

  HIP_OK(hipSetDevice(0));
  if (!aid_init()) {
    printf("[%s] AID-local allocation unavailable; cannot run\n", tag);
    return 2;
  }

  size_t const flag_bytes = 2ull << 20;
  int *fa = (int *)alloc_in_aid(flag_bytes, 0);
  int *fb = (int *)alloc_in_aid(flag_bytes, 1);
  if (fa == nullptr || fb == nullptr) {
    printf("[%s] could not place a flag array in each AID\n", tag);
    return 2;
  }
  HIP_OK(hipMemset(fa, 0, flag_bytes));
  HIP_OK(hipMemset(fb, 0, flag_bytes));

  unsigned int *d_vis = nullptr;
  int *d_seen = nullptr;
  HIP_OK(hipMalloc(&d_vis, nsamp * sizeof(unsigned int)));
  HIP_OK(hipMalloc(&d_seen, nblocks * sizeof(int)));
  HIP_OK(hipMemset(d_vis, 0, nsamp * sizeof(unsigned int)));
  HIP_OK(hipMemset(d_seen, 0xff, nblocks * sizeof(int)));

  printf("[%s] split=%d  coherent=%d  stride %ld B  %d tiles/XCD  %d waves  "
         "%d layers\n",
         tag, split_on, (int)g_coherent, stride_bytes, tiles,
         nblocks * NWAVES, n_layers);

  hipLaunchKernelGGL(aidsplit, dim3(nblocks), dim3(NTHREADS), 0, 0, fa, fb,
                     d_vis, d_seen, n_layers, tiles, stride_ints, split_on,
                     period, pdelay,
                     (unsigned long long)(bail_ms * 1e-3 * 100e6));
  HIP_OK(hipDeviceSynchronize());
  HIP_OK(hipGetLastError());

  // The rank scheme assumes block i runs on XCD i%8. Check it rather than
  // trust it, because everything downstream depends on one producer per XCD.
  std::vector<int> seen(nblocks);
  HIP_OK(hipMemcpy(seen.data(), d_seen, nblocks * sizeof(int),
                   hipMemcpyDeviceToHost));
  int bad_rr = 0;
  for (int b = 0; b < nblocks; ++b) {
    if (seen[b] != b % NXCD) {
      ++bad_rr;
    }
  }
  printf("[%s] block->XCD round-robin holds for %d of %d blocks\n", tag,
         nblocks - bad_rr, nblocks);

  std::vector<unsigned int> vis(nsamp);
  HIP_OK(hipMemcpy(vis.data(), d_vis, nsamp * sizeof(unsigned int),
                   hipMemcpyDeviceToHost));

  // Count timeouts over every layer, not just past the warmup skip. A wave that
  // never makes progress times out on layer 1 and then breaks out, so its one
  // failure record sits inside the warmup window; skipping it reported the run
  // as a small sample count rather than as the failure it is.
  int const skip = n_layers / 10;
  std::vector<unsigned int> ok;
  int stuck_by_xcd[NXCD] = {0}, done_by_xcd[NXCD] = {0};
  for (int x = 0; x < NXCD; ++x) {
    for (int lo = 0; lo < tiles; ++lo) {
      for (int w = 0; w < NWAVES; ++w) {
        int const row = ((x * tiles + lo) * NWAVES + w) * n_layers;
        for (int l = 0; l < n_layers; ++l) {
          unsigned int const v = vis[row + l];
          if (v == 0xffffffffu) {
            ++stuck_by_xcd[x];
          } else if (v != 0 && v < 100000000u) {
            ++done_by_xcd[x];
            if (l >= skip) {
              ok.push_back(v);
            }
          }
        }
      }
    }
  }
  int const per_xcd = tiles * NWAVES * n_layers;
  printf("[%s] per XCD, layers observed / timed out (of %d):\n", tag, per_xcd);
  for (int x = 0; x < NXCD; ++x) {
    printf("        xcd%d  %6d observed  %4d timed out%s\n", x, done_by_xcd[x],
           stuck_by_xcd[x],
           (done_by_xcd[x] < per_xcd / 2) ? "   <-- made no real progress" : "");
  }
  if (ok.empty()) {
    printf("[%s] no successful observations\n", tag);
    return 1;
  }
  std::sort(ok.begin(), ok.end());
  printf("[%s] vis_p50=%.2f vis_p90=%.2f us over %zu samples\n", tag,
         ok[ok.size() / 2] * 10.0 / 1000.0,
         ok[(size_t)(0.9 * (ok.size() - 1))] * 10.0 / 1000.0, ok.size());
  return 0;
}
