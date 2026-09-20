// Standalone Phase 7 benchmark.
//
// Calls the *real* gang_linear_mxfp4_res_bias_rmsnorm_topk_kernel with the same
// instantiation the model uses, so the five [OPROJ_INNER] buckets (slicewait,
// mfma, bar, rmsnorm+router, topk) come out of the production code path rather
// than a reimplementation that could miss the effect.
//
// Instantiation and scalars are lifted from the generated test.cu:
//   gang_full_layer_fused_kernel_mi300<1, 64, 2944, 2880, 64, 8, 4096, 512, 8,
//                                      4096, 512, 8, 128, 1, 16, 4096, 128, 4,
//                                      2944, 2944, 128, 64, true, 1>
//   -> Phase 7 is <BATCH=1, OUTPUT_PER_WG=16, REDUCTION=4096,
//                  HIDDEN=2880, NUM_EXPERTS=128, TOPK_K=4>
//   -> n_wgs_per_xcd=23, output_stride=2944, router_tile_n=16,
//      total_oproj_tiles=184, total_topk_tiles=128, tiles_per_xcd=23
//
// The attention producer that Phase 7 waits on is replaced by a stub: the rank-0
// block on each XCD writes that XCD's 1 KiB slice and publishes
// attn_release[xcd*16]. That is the whole point of the harness -- with
// --skew=0 every producer releases at the same instant, so slicewait measures
// flag visibility alone; with --skew>0 producer x releases later than x-1, so
// slicewait picks up arrival skew. Running both in NPS1 and NPS2 separates the
// two explanations for the 27x.

#define MPK_USE_CK_FMHA 1
#include "persistent_kernel.cuh"

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

using namespace mirage::runtime;

// persistent_kernel.cuh declares these for the generated test.cu to define, and
// its own worker_kernel/persistent_kernel reference them. This harness never
// launches those kernels, but the symbols still have to resolve.
__device__ __forceinline__ void
    _execute_task(TaskDesc const *task_desc,
                  RuntimeConfig const &runtime_config) {
  (void)task_desc;
  (void)runtime_config;
}

__device__ __forceinline__ void
    _execute_gang_task(TaskDesc const *task_desc,
                       RuntimeConfig const &runtime_config, int tile_idx) {
  (void)task_desc;
  (void)runtime_config;
  (void)tile_idx;
}

// Likewise the transpiler-generated task-graph builder. This harness launches
// Phase 7 directly and never builds a task graph.
static void _init_persistent_kernel(std::vector<FullTaskDesc> &all_tasks,
                                    std::vector<EventDesc> &all_events,
                                    std::vector<TaskId> &first_tasks,
                                    int num_gpus, int my_gpu_id) {
  (void)all_tasks;
  (void)all_events;
  (void)first_tasks;
  (void)num_gpus;
  (void)my_gpu_id;
}

#define NXCD 8
#define TILES_PER_XCD 23
#define TOTAL_TILES (NXCD * TILES_PER_XCD) // 184
#define NTHREADS 256

#define OPROJ_BATCH 1
#define OPROJ_OUTPUT_PER_WG 16
#define OPROJ_REDUCTION 4096
#define ACTUAL_HIDDEN 2880
#define NUM_EXPERTS 128
#define TOPK_K 4
#define OUTPUT_STRIDE 2944
#define ROUTER_TILE_N 16
#define TOTAL_TOPK_TILES 128

// MXFP4 weight bytes per workgroup, same arithmetic the kernel does.
#define WG_DATA_BYTES (OPROJ_OUTPUT_PER_WG * (OPROJ_REDUCTION / 2))
#define WG_SCALE_BYTES (OPROJ_OUTPUT_PER_WG * (OPROJ_REDUCTION / 32))
#define WG_BYTES (WG_DATA_BYTES + WG_SCALE_BYTES)

#define ATTN_SLICE (OPROJ_REDUCTION / NXCD) // 512 bf16 = 1 KiB
#define FLAG_STRIDE 16                      // ints, matches attn_release[x*16]

#define HIP_OK(call)                                                           \
  do {                                                                         \
    hipError_t _e = (call);                                                    \
    if (_e != hipSuccess) {                                                    \
      fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #call,           \
              hipGetErrorString(_e));                                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

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

static void *alloc_in_aid(size_t bytes, int aid, bool coherent) {
  uint64_t const flags = GEM_CREATE_AID_LOCAL |
                         (g_range_for_aid[aid] & 1u ? GEM_CREATE_AID_SELECT : 0) |
                         (coherent ? GEM_CREATE_COHERENT : 0);
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

// The rendezvous counter shares the AID-local buffer with the flags, at a 1 MiB
// offset so it cannot land on a flag line.
#define BAR_OFF_INTS (1 << 18)

__device__ unsigned int g_local_claim[NXCD];

__device__ __forceinline__ int drv_get_xcd() {
  int x;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(x));
  return x;
}

__device__ __forceinline__ void drv_st_wt_u32(void *addr, unsigned int val) {
  asm volatile("global_store_dword %0, %1, off sc0 sc1"
               :
               : "v"(addr), "v"(val)
               : "memory");
}

__device__ __forceinline__ int drv_ld_sys_s32(int *addr) {
  int val;
  asm volatile("global_load_dword %0, %1, off sc0 sc1\n"
               "s_waitcnt vmcnt(0)"
               : "=v"(val)
               : "v"(addr)
               : "memory");
  return val;
}

__global__ __launch_bounds__(NTHREADS) void drive_phase7(
    void *attn_out, void *weight, void *residual, void *bias,
    void *norm_weight, void *norm_output, void *router_weight,
    void *router_bias, void *logits_scratch, void *counters, void *output,
    void *topk_weight, void *routing_indices, void *active_expert_ids,
    int *flags, int *flags_b, unsigned int *bar, unsigned int *bar_b,
    int *xcd_seen, int n_layers, int delay_ticks, int delay_skew, int tiles,
    int dual, int split_on) {
  int const xcd = drv_get_xcd();
  int const tid = threadIdx.x;

  // Claim a rank within this XCD rather than trusting a blockIdx->XCD mapping.
  __shared__ int s_local;
  if (tid == 0) {
    s_local = (int)atomicAdd(&g_local_claim[xcd], 1u);
    atomicAdd(&xcd_seen[xcd], 1);
  }
  __syncthreads();
  int const local = s_local;
  if (local >= tiles) {
    return; // more blocks than tiles on this XCD; do not corrupt the barrier
  }
  int const tile_idx = xcd * tiles + local;

  // Which replica this XCD touches. The upper half of the XCDs uses the copy
  // homed in AID1 and the lower half the copy in AID0, so every poll is served
  // in the reader's own coherence domain -- the precondition that makes a
  // coherent MTYPE sound on these lines.
  bool const upper = split_on && xcd >= NXCD / 2;
  int *const rel = upper ? flags_b : flags;
  unsigned int *const my_bar = upper ? bar_b : bar;

  unsigned short *my_slice = (unsigned short *)attn_out + xcd * ATTN_SLICE;
  unsigned char *my_weight =
      (unsigned char *)weight + (size_t)xcd * tiles * WG_BYTES;
  unsigned short *my_residual =
      (unsigned short *)residual + (size_t)xcd * tiles * OPROJ_OUTPUT_PER_WG;

  for (int layer = 1; layer <= n_layers; ++layer) {
    // ?? stub for the attention merge Phase 7 waits on ??
    if (local == 0) {
      if (delay_ticks || delay_skew) {
        unsigned long long const until =
            __builtin_amdgcn_s_memrealtime() +
            (unsigned long long)(delay_ticks + delay_skew * xcd);
        while (__builtin_amdgcn_s_memrealtime() < until) {
          __builtin_amdgcn_s_sleep(1);
        }
      }
      // 256 threads x 1 dword = 512 bf16 = this XCD's slice.
      drv_st_wt_u32((unsigned int *)my_slice + tid,
                    (unsigned)(layer * 2654435761u + xcd));
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      __syncthreads();
      if (tid == 0) {
        // Same value into both replicas, so a consumer sees the identical fact
        // whichever one it polls. The far-AID store still invalidates the
        // sharers co-located with that line, which is what lets one producer
        // release both halves.
        drv_st_wt_u32(&flags[xcd * FLAG_STRIDE], (unsigned)layer);
        if (dual) {
          drv_st_wt_u32(&flags_b[xcd * FLAG_STRIDE], (unsigned)layer);
        }
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      }
    }

    kernel::gang_linear_mxfp4_res_bias_rmsnorm_topk_kernel<
        OPROJ_BATCH, OPROJ_OUTPUT_PER_WG, OPROJ_REDUCTION, ACTUAL_HIDDEN,
        NUM_EXPERTS, TOPK_K>(
        attn_out, my_weight, my_residual, bias, norm_weight, norm_output,
        router_weight, router_bias, logits_scratch, counters, output,
        topk_weight, routing_indices, active_expert_ids,
        /*num_active_tokens=*/1,
        /*n_wgs_per_xcd=*/tiles,
        /*output_stride=*/OUTPUT_STRIDE,
        /*router_tile_n=*/ROUTER_TILE_N,
        /*total_oproj_tiles=*/NXCD * tiles,
        /*total_topk_tiles=*/TOTAL_TOPK_TILES,
        /*tiles_per_xcd=*/tiles,
        /*tile_idx=*/tile_idx,
        /*routing_ready_ptr=*/nullptr,
        /*layer_epoch=*/layer,
        /*ts_base=*/nullptr,
        /*attn_slice_release=*/rel);

    // ?? per-layer rendezvous across all 184 tiles, as phase 9 does ??
    __syncthreads();
    if (tid == 0) {
      atomicAdd(bar, 1u);
      if (dual) {
        atomicAdd(bar_b, 1u);
      }
      while (drv_ld_sys_s32((int *)my_bar) < layer * NXCD * tiles) {
        __builtin_amdgcn_s_sleep(1);
      }
    }
    __syncthreads();
  }
}

int main(int argc, char **argv) {
  int n_layers = 200;
  int delay_ticks = 0, delay_skew = 0;
  int tiles = TILES_PER_XCD;
  int use_aid = 0, split_on = 1;
  char const *tag = "run";
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--layers=", 9)) {
      n_layers = atoi(argv[i] + 9);
    } else if (!strncmp(argv[i], "--delay=", 8)) {
      delay_ticks = atoi(argv[i] + 8);
    } else if (!strncmp(argv[i], "--skew=", 7)) {
      delay_skew = atoi(argv[i] + 7);
    } else if (!strncmp(argv[i], "--tiles=", 8)) {
      tiles = atoi(argv[i] + 8);
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
  if (tiles < 1 || tiles > TILES_PER_XCD) {
    fprintf(stderr, "--tiles must be in [1, %d]\n", TILES_PER_XCD);
    return 2;
  }

  void *attn = nullptr, *weight = nullptr, *residual = nullptr, *bias = nullptr;
  void *nw = nullptr, *no = nullptr, *rw = nullptr, *rb = nullptr;
  void *logits = nullptr, *counters = nullptr, *out = nullptr, *tkw = nullptr;
  void *ridx = nullptr, *aeid = nullptr;
  int *flags = nullptr, *flags_b = nullptr, *xcd_seen = nullptr;
  unsigned int *bar = nullptr, *bar_b = nullptr;
  int dual = 0;

  size_t const weight_bytes = (size_t)NXCD * tiles * WG_BYTES;
  HIP_OK(hipMalloc(&attn, OPROJ_REDUCTION * 2));
  HIP_OK(hipMalloc(&weight, weight_bytes));
  HIP_OK(hipMalloc(&residual, OUTPUT_STRIDE * 2));
  HIP_OK(hipMalloc(&bias, OUTPUT_STRIDE * 2));
  HIP_OK(hipMalloc(&nw, ACTUAL_HIDDEN * 2));
  HIP_OK(hipMalloc(&no, ACTUAL_HIDDEN * 2));
  HIP_OK(hipMalloc(&rw, (size_t)NUM_EXPERTS * ACTUAL_HIDDEN * 2));
  HIP_OK(hipMalloc(&rb, NUM_EXPERTS * 2));
  HIP_OK(hipMalloc(&logits, 64 * 1024));
  HIP_OK(hipMalloc(&counters, 64 * 1024));
  HIP_OK(hipMalloc(&out, OUTPUT_STRIDE * 2));
  HIP_OK(hipMalloc(&tkw, 64 * 1024));
  HIP_OK(hipMalloc(&ridx, 64 * 1024));
  HIP_OK(hipMalloc(&aeid, 64 * 1024));
  HIP_OK(hipMalloc(&flags, 4096));
  HIP_OK(hipMalloc(&bar, sizeof(unsigned int)));
  HIP_OK(hipMalloc(&xcd_seen, NXCD * sizeof(int)));

  HIP_OK(hipMemset(attn, 0, OPROJ_REDUCTION * 2));
  HIP_OK(hipMemset(weight, 0x11, weight_bytes));
  HIP_OK(hipMemset(residual, 0, OUTPUT_STRIDE * 2));
  HIP_OK(hipMemset(bias, 0, OUTPUT_STRIDE * 2));
  HIP_OK(hipMemset(nw, 0x3c, ACTUAL_HIDDEN * 2)); // ~1.0 bf16
  HIP_OK(hipMemset(no, 0, ACTUAL_HIDDEN * 2));
  HIP_OK(hipMemset(rw, 0x3c, (size_t)NUM_EXPERTS * ACTUAL_HIDDEN * 2));
  HIP_OK(hipMemset(rb, 0, NUM_EXPERTS * 2));
  HIP_OK(hipMemset(logits, 0, 64 * 1024));
  HIP_OK(hipMemset(counters, 0, 64 * 1024));
  HIP_OK(hipMemset(out, 0, OUTPUT_STRIDE * 2));
  HIP_OK(hipMemset(tkw, 0, 64 * 1024));
  HIP_OK(hipMemset(ridx, 0, 64 * 1024));
  HIP_OK(hipMemset(aeid, 0, 64 * 1024));
  HIP_OK(hipMemset(flags, 0, 4096));
  HIP_OK(hipMemset(bar, 0, sizeof(unsigned int)));
  HIP_OK(hipMemset(xcd_seen, 0, NXCD * sizeof(int)));

  if (use_aid) {
    if (!aid_init()) {
      fprintf(stderr, "ABORT: --aid=1 but AID-local allocation is unavailable "
                      "(stock driver, or not NPS2)\n");
      return 2;
    }
    size_t const sync_bytes = 2ull << 20;
    void *fa = alloc_in_aid(sync_bytes, 0, g_coherent);
    void *fb = alloc_in_aid(sync_bytes, 1, g_coherent);
    if (fa == nullptr || fb == nullptr) {
      fprintf(stderr, "ABORT: could not place a sync buffer in each AID\n");
      return 2;
    }
    flags = (int *)fa;
    flags_b = (int *)fb;
    bar = (unsigned int *)(flags + BAR_OFF_INTS);
    bar_b = (unsigned int *)(flags_b + BAR_OFF_INTS);
    HIP_OK(hipMemset(flags, 0, sync_bytes));
    HIP_OK(hipMemset(flags_b, 0, sync_bytes));
    dual = 1;
  } else {
    // Baseline: one hipMalloc'd flag page and one counter, both MTYPE_NC and
    // both read from either AID -- exactly what the model does today.
    flags_b = flags;
    bar_b = bar;
    split_on = 0;
  }

  printf("[%s] attn_out %p  flags %p / %p  weight %p (%.2f MiB)\n", tag, attn,
         (void *)flags, (void *)flags_b, weight, weight_bytes / 1048576.0);
  printf("[%s] aid=%d coherent=%d split=%d\n", tag, use_aid, (int)g_coherent,
         split_on);
  printf("[%s] %d tiles (%d/XCD), %d threads, %d layers, delay=%d skew=%d, "
         "%d polling waves\n",
         tag, NXCD * tiles, tiles, NTHREADS, n_layers, delay_ticks, delay_skew,
         NXCD * tiles * (NTHREADS / 64));

  HIP_OK(hipFuncSetAttribute((void const *)drive_phase7,
                             hipFuncAttributeMaxDynamicSharedMemorySize,
                             MAX_DYNAMIC_SHARED_MEMORY_SIZE));

  hipEvent_t e0, e1;
  HIP_OK(hipEventCreate(&e0));
  HIP_OK(hipEventCreate(&e1));
  HIP_OK(hipEventRecord(e0));
  hipLaunchKernelGGL(drive_phase7, dim3(NXCD * tiles), dim3(NTHREADS),
                     MAX_DYNAMIC_SHARED_MEMORY_SIZE, 0, attn, weight, residual,
                     bias, nw, no, rw, rb, logits, counters, out, tkw, ridx,
                     aeid, flags, flags_b, bar, bar_b, xcd_seen, n_layers,
                     delay_ticks, delay_skew, tiles, dual, split_on);
  HIP_OK(hipEventRecord(e1));
  HIP_OK(hipDeviceSynchronize());
  HIP_OK(hipGetLastError());
  float ms = 0.0f;
  HIP_OK(hipEventElapsedTime(&ms, e0, e1));

  int seen[NXCD];
  HIP_OK(hipMemcpy(seen, xcd_seen, sizeof(seen), hipMemcpyDeviceToHost));
  printf("[%s] blocks per XCD:", tag);
  for (int i = 0; i < NXCD; ++i) {
    printf(" %d", seen[i]);
  }
  printf("\n[%s] wall %.2f ms / %d layers = %.2f us per layer\n", tag, ms,
         n_layers, ms * 1000.0 / n_layers);
  return 0;
}
