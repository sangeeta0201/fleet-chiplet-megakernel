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

// Byte offset of the scale half within the staged LDS tile. Must match the
// kernel's OPROJ_LDS_DATA_PAD exactly, or the MFMA reads its scales from the
// wrong place; both round WG_DATA_BYTES up to a whole 256-thread x 16-byte
// pass the same way.
#define LDS_W_DATA_PAD (((WG_DATA_BYTES / 16 + 255) / 256) * 256 * 16)

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
static bool g_dcoh = false;

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

// The hierarchical-barrier release replicas share the same AID-local buffer, at
// a 1.5 MiB offset so they land on neither a flag line nor the rendezvous
// counter. Eight lines of 16 ints, the same layout the kernel uses inside
// `counters`.
#define HIER_OFF_INTS (3 << 17)

// The eight per-XCD level-1 arrival lines, at 1.75 MiB so they clear the flags,
// the rendezvous counter and the release replicas. These are *partitioned*, not
// replicated: lines 0..3 are only ever touched in AID0's buffer and 4..7 only in
// AID1's, so each buffer's other half stays untouched and the kernel can keep
// indexing both by xcd_id.
#define HIER_LOCAL_OFF_INTS (7 << 16)

// The hierarchical inter-layer rendezvous, 4 KiB past the level-1 lines, which
// only need 8 lines of 16 ints. 768 bytes, so it clears the end of the 2 MiB
// sync buffer with room to spare.
//
// The flat rendezvous this replaces has every one of the 184 blocks increment
// *both* replicas -- 368 atomics on two lines per layer, and for 92 of the
// blocks each layer one of the two is a read-modify-write across the partition
// boundary. Measured, that is what staggers the two halves by 0.84 us before
// Phase 7 even starts, which the Phase 7 barrier then faithfully reports as
// `wait`. The tree below keeps every step in the arriving block's own AID and
// puts exactly one store on the critical path that crosses the boundary.
//
// Every line below is indexed by the *half* as well, so the two halves never
// share one. That is what lets the same code run in NPS1, where there is one
// buffer and both pointers alias it: the topology stays a genuine 4/4 split and
// only the placement collapses. Indexing solely by placement would have both
// halves increment one level-2 counter, and the release would fire on the fourth
// arrival, before the other four XCDs had shown up.
//
// Region layout, in ints, relative to the base of one copy:
//   [x * 16]        level-1 arrival, one line per XCD. Partitioned, not
//                   replicated: line x is only ever touched by XCD x's own 23
//                   blocks, which are co-located by construction, so there is no
//                   misroutable reader here at all.
//   [128 + h * 16]  level-2 arrival for half h, its own 4 level-1 closers. Local.
//   [160 + h * 16]  the slot half h *observes*, written by the other half's
//                   closer. Single-writer, single-reader, and homed with the
//                   reader: the signal is a remote write-through store, which is
//                   off the critical path, while the poll is served locally.
//   [192 + h * 16]  half h's release epoch, polled by its own 92 blocks locally.
#define RDV_OFF_INTS (HIER_LOCAL_OFF_INTS + 1024)
#define RDV_L1_INTS(x) ((x) * 16)
#define RDV_L2_INTS(h) (128 + (h) * 16)
#define RDV_PEER_INTS(h) (160 + (h) * 16)
#define RDV_REL_INTS(h) (192 + (h) * 16)
#define RDV_REGION_BYTES 1024

// The barrier's level-2 arrival, 12 KiB past the level-1 lines so it clears the
// rendezvous region above. Four lines of 16 ints: two arrival counters and two
// handshake slots, indexed by half exactly as the rendezvous is. This is the last
// piece of barrier state that was still in the plain `hipMalloc`'d `counters`
// buffer, and therefore the last one the SPX+NPS2 MTYPE_NC demotion applied to.
#define HIER_L2_OFF_INTS (HIER_LOCAL_OFF_INTS + 3072)

// --srdv. One slot per participant. Nothing here is ever the target of a
// read-modify-write, and no slot has more than one writer for the whole run,
// so the coherency point has nothing to order and the poll never contends with
// the publish. Tier 1 is NXCD arrays of `tiles` slots; tier 2 is NXCD slots on
// a single line, which is why it is safe to make that one line device-coherent
// even though the 2 MiB flag page is not.
#define SENT_LOC_OFF_INTS (HIER_LOCAL_OFF_INTS + 5120)
#define SENT_LOC_STRIDE 32
#define SENT_GLB_OFF_INTS (HIER_LOCAL_OFF_INTS + 6144)
#ifndef MPK_DRV_SRDV_SLEEP
#define MPK_DRV_SRDV_SLEEP 1
#endif

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

// ---- placement confirmation, lifted from mtype_micro ----------------------
//
// The allocator reporting "range 1 requested" is not evidence that the pages
// are in AID1. The only evidence is latency: a dependent chase with one lane
// per XCD, so there is exactly one outstanding load and the number is latency
// rather than bandwidth. On these MTYPE_NC buffers nothing is cacheable, so
// every step goes to HBM and the local/remote split shows up directly -- ~148 ns
// when the reading XCD owns the stacks and ~242 ns when it does not.
//
// Destructive: it overwrites the buffer with a pointer chain, so it has to run
// before the weight fill, never between layers.
#define PROBE_ELEMS_PER_PAGE 1024 // 4 KiB apart, so no two steps share a page

// A device-coherency read. Unlike a non-temporal load -- which is only an
// eviction-policy hint and will happily hit a line the L2 already holds -- this
// has to reach the coherency point on every step. That is what makes placement
// observable in a buffer small enough to be L2-resident, and it is the same
// instruction class the barrier polls with, so it reports the latency that
// actually matters for sync traffic.
__device__ __forceinline__ uint32_t probe_ld_sys_u32(uint32_t const *p) {
  uint32_t v;
  asm volatile("global_load_dword %0, %1, off sc0 sc1\n"
               "s_waitcnt vmcnt(0)"
               : "=v"(v)
               : "v"(p)
               : "memory");
  return v;
}

// mode 0: non-temporal, i.e. the latency the workload actually sees, cache and
//         all. mode 1: device-scope, i.e. where the pages really are.
__global__ void aid_probe_chase(uint32_t const *__restrict__ p, int steps,
                                unsigned long long *__restrict__ ticks,
                                uint32_t *__restrict__ sink,
                                unsigned *__restrict__ xcd_of, int mode) {
  if (threadIdx.x != 0) {
    return;
  }
  xcd_of[blockIdx.x] = (unsigned)drv_get_xcd();
  uint32_t idx = 0;
  // Warm the TLB first, so the timed loop measures the data path and not a
  // page-table walk.
  for (int i = 0; i < 8192; ++i) {
    idx = mode ? probe_ld_sys_u32(&p[idx])
               : __builtin_nontemporal_load(&p[idx]);
  }
  unsigned long long t0 = __builtin_amdgcn_s_memrealtime();
  if (mode) {
    for (int i = 0; i < steps; ++i) {
      idx = probe_ld_sys_u32(&p[idx]);
    }
  } else {
    for (int i = 0; i < steps; ++i) {
      idx = __builtin_nontemporal_load(&p[idx]);
    }
  }
  unsigned long long t1 = __builtin_amdgcn_s_memrealtime();
  ticks[blockIdx.x] = t1 - t0;
  sink[blockIdx.x] = idx;
}

static void probe_placement(char const *name, void *buf, size_t bytes,
                            int mode) {
  size_t const nelem = bytes / sizeof(uint32_t);
  // Space the chain a page apart when there are pages to spare, and a cache
  // line apart otherwise. The requirement is only that consecutive steps land
  // on distinct lines; `out` is 5888 B and has no pages to walk, but it is
  // exactly the buffer whose home matters most, since every access to it is
  // write-through and reaches HBM.
  size_t step_elems = PROBE_ELEMS_PER_PAGE;
  if (nelem / PROBE_ELEMS_PER_PAGE < 64) {
    step_elems = 16; // 64 B = one line
  }
  size_t const nslots = nelem / step_elems;
  if (nslots < 8) {
    printf("[PROBE] %-14s too small to chase (%zu slots)\n", name, nslots);
    return;
  }
  std::vector<uint32_t> perm(nslots);
  for (size_t i = 0; i < nslots; ++i) {
    perm[i] = (uint32_t)i;
  }
  // A fixed shuffle, so the walk order is unpredictable to any prefetcher but
  // identical between the buffers being compared.
  unsigned long long s = 88172645463325252ull;
  for (size_t i = nslots - 1; i > 0; --i) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    std::swap(perm[i], perm[s % (i + 1)]);
  }
  std::vector<uint32_t> host(nelem, 0u);
  for (size_t i = 0; i < nslots; ++i) {
    host[(size_t)perm[i] * step_elems] =
        (uint32_t)((size_t)perm[(i + 1) % nslots] * step_elems);
  }
  HIP_OK(hipMemcpy(buf, host.data(), bytes, hipMemcpyHostToDevice));

  int const steps = 4096;
  unsigned long long *d_ticks = nullptr;
  uint32_t *d_sink = nullptr;
  unsigned *d_xcd = nullptr;
  HIP_OK(hipMalloc(&d_ticks, NXCD * sizeof *d_ticks));
  HIP_OK(hipMalloc(&d_sink, NXCD * sizeof *d_sink));
  HIP_OK(hipMalloc(&d_xcd, NXCD * sizeof *d_xcd));
  unsigned long long best[NXCD];
  for (int b = 0; b < NXCD; ++b) {
    best[b] = ~0ull;
  }
  unsigned long long t[NXCD];
  unsigned x[NXCD];
  // Best of 20: the minimum is the uncontended path, which is what placement
  // determines. A mean would fold in scheduling noise.
  for (int r = 0; r < 20; ++r) {
    hipLaunchKernelGGL(aid_probe_chase, dim3(NXCD), dim3(64), 0, 0,
                       (uint32_t const *)buf, steps, d_ticks, d_sink, d_xcd,
                       mode);
    HIP_OK(hipDeviceSynchronize());
    HIP_OK(hipMemcpy(t, d_ticks, sizeof t, hipMemcpyDeviceToHost));
    for (int b = 0; b < NXCD; ++b) {
      best[b] = std::min(best[b], t[b]);
    }
  }
  HIP_OK(hipMemcpy(x, d_xcd, sizeof x, hipMemcpyDeviceToHost));
  double lo = 0, hi = 0;
  int nlo = 0, nhi = 0;
  printf("[PROBE] %-14s (%7zu B, %4zu lines):", name, bytes, nslots);
  for (int b = 0; b < NXCD; ++b) {
    double ns = best[b] * 10.0 / (double)steps; // 100 MHz -> 10 ns per tick
    printf(" xcd%u=%.0f", x[b], ns);
    if (x[b] < NXCD / 2) {
      lo += ns;
      ++nlo;
    } else {
      hi += ns;
      ++nhi;
    }
  }
  if (nlo && nhi) {
    printf("  | XCD0-3 %.1f  XCD4-7 %.1f  skew %+.1f ns", lo / nlo, hi / nhi,
           hi / nhi - lo / nlo);
  }
  printf("\n");
  HIP_OK(hipFree(d_ticks));
  HIP_OK(hipFree(d_sink));
  HIP_OK(hipFree(d_xcd));
}

__global__ __launch_bounds__(NTHREADS) void drive_phase7(
    void *attn_out, void *weight, void *residual, void *attn_out_b,
    void *weight_b, void *residual_b, void *bias,
    void *norm_weight, void *norm_output, void *router_weight,
    void *router_bias, void *logits_scratch, void *counters, void *output,
    void *topk_weight, void *routing_indices, void *active_expert_ids,
    int *flags, int *flags_b, unsigned int *bar, unsigned int *bar_b,
    int *hier_lo, int *hier_hi, int *hloc_lo, int *hloc_hi, int *rdv_lo,
    int *rdv_hi, int *hl2_lo, int *hl2_hi, int *xcd_seen, int n_layers,
    int delay_ticks, int delay_skew, int tiles, int dual, int split_on,
    int l2_fold, int *srdv_la, int *srdv_lb, int *srdv_ga, int *srdv_gb) {
  int const xcd = drv_get_xcd();
  int const tid = threadIdx.x;

  // The same dynamic LDS the callee declares, so the tile staged below is the
  // tile its MFMA reads. The offset comes from the kernel's own constexpr
  // rather than a copied literal.
  extern __shared__ char _lm_smem[];
  constexpr int LDS_W_OFF =
      kernel::oproj_lds_w_off(OPROJ_BATCH, OPROJ_REDUCTION);

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

  // Which AID this XCD's stacks are in. Independent of split_on: that flag says
  // whether the *sync* state is replicated, and data placement is a separate
  // question -- a private slab needs no replica, only the right home.
  bool const hi_aid = xcd >= NXCD / 2;

  // --dsplit. `weight` and `residual` are already partitioned per XCD by the
  // index arithmetic below, and every access is inside the XCD's own slab, so
  // there is exactly one toucher per slab and no replica is needed: the upper
  // half's slabs simply move to AID1. The index becomes within-half, since each
  // buffer now holds NXCD/2 slabs rather than NXCD.
  int const wx = weight_b ? (hi_aid ? xcd - NXCD / 2 : xcd) : xcd;
  unsigned char *my_weight =
      (unsigned char *)((weight_b && hi_aid) ? weight_b : weight) +
      (size_t)wx * tiles * WG_BYTES;
  unsigned short *my_residual =
      (unsigned short *)((residual_b && hi_aid) ? residual_b : residual) +
      (size_t)wx * tiles * OPROJ_OUTPUT_PER_WG;

  // attn_out is the exception: the O-proj reduction reads all NXCD slices, so
  // partitioning cannot make it local and it has to be replicated instead. Each
  // producer publishes its own 1 KiB slice into both copies -- one local store
  // and one remote -- so that every consumer can read all 4096 values from the
  // copy in its own AID. That trades 8 remote KiB per layer for the four XCDs
  // that currently read the entire buffer across the boundary.
  unsigned short *my_slice = (unsigned short *)attn_out + xcd * ATTN_SLICE;
  unsigned short *my_slice_b =
      attn_out_b ? (unsigned short *)attn_out_b + xcd * ATTN_SLICE : nullptr;
  void *const attn_base = (attn_out_b && hi_aid) ? attn_out_b : attn_out;

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
      //
      // A well-conditioned bf16 rather than a hash reinterpreted as two
      // arbitrary exponents. The value depends on both the layer and the XCD,
      // which is what lets a slice read too early -- still carrying the
      // previous layer's value -- change the O-proj result. Every term is a
      // negative power of two, so the stored bf16 is the intended value rather
      // than a rounding of it.
      float const fv = 1.0f + 0.5f * (float)(layer & 3) + 0.0625f * (float)xcd;
      unsigned int const bf = __float_as_uint(fv) >> 16;
      drv_st_wt_u32((unsigned int *)my_slice + tid, bf | (bf << 16));
      // Identical bytes into the second replica, so a consumer sees the same
      // value whichever copy it reads. Placed before the vmcnt below so the one
      // wait covers both stores and the flag still publishes after both land.
      if (my_slice_b) {
        drv_st_wt_u32((unsigned int *)my_slice_b + tid, bf | (bf << 16));
      }
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

    // ---- stand-in for Phase 6's buffer_load_lds weight DMA --------------
    //
    // Without this the MFMA's B operand is an unpopulated LDS tile and the
    // whole GEMM is zero, whatever the barriers do.
    //
    // Re-staged every layer rather than once before the loop: the kernel's own
    // scratch lives below oproj_lds_w_off() and should not reach this region,
    // but a harness meant to catch ordering bugs should not rest on that.
#ifdef MPK_DRV_STAGE_ONCE
    // The weights are the same every layer, so one copy is enough -- as long as
    // nothing in the kernel writes into this LDS region between layers. Compare
    // the hash against the per-layer build to find out.
    if (layer == 1)
#endif
    {
      // 16 bytes per thread per pass, as one dwordx4. Every base is at least
      // 16-byte aligned -- hipMalloc returns 256, WG_BYTES is a multiple of 16,
      // and oproj_lds_w_off() rounds to 16 -- but saying so through uint4 is
      // what lets the compiler emit the wide access. Spelled as a byte memcpy
      // it has to assume alignment 1 and copies a byte at a time.
      uint4 const *w_src =
          (uint4 const *)(my_weight + (size_t)local * WG_BYTES);
      uint4 *w_dst = (uint4 *)((unsigned char *)_lm_smem + LDS_W_OFF);
      for (int i = tid; i < WG_DATA_BYTES / 16; i += NTHREADS) {
        w_dst[i] = w_src[i];
      }
      uint4 const *s_src = w_src + WG_DATA_BYTES / 16;
      uint4 *s_dst =
          (uint4 *)((unsigned char *)_lm_smem + LDS_W_OFF + LDS_W_DATA_PAD);
      for (int i = tid; i < WG_SCALE_BYTES / 16; i += NTHREADS) {
        s_dst[i] = s_src[i];
      }
    }

    // ---- give RMSNorm a row that is not flat ----------------------------
    //
    // Every weight byte is identical, so all 16 of a block's output columns
    // come out equal, and so does every other block's. RMSNorm divides a flat
    // row by its own RMS and returns exactly 1.0 in every column whatever the
    // magnitude, which would erase the layer dependence the check needs.
    // A per-column residual makes the row non-flat and layer-dependent, so a
    // column read before its XCD stored it shows up in rmsnorm_out.
    //
    // Each block writes only the 16 entries it goes on to read itself, so the
    // __syncthreads below is the whole of the ordering required.
    if (tid < OPROJ_OUTPUT_PER_WG) {
      int const gcol = tile_idx * OPROJ_OUTPUT_PER_WG + tid;
      float const rv = 0.5f * (float)((gcol + layer) & 15);
      my_residual[local * OPROJ_OUTPUT_PER_WG + tid] =
          (unsigned short)(__float_as_uint(rv) >> 16);
    }
    __syncthreads();

    kernel::gang_linear_mxfp4_res_bias_rmsnorm_topk_kernel<
        OPROJ_BATCH, OPROJ_OUTPUT_PER_WG, OPROJ_REDUCTION, ACTUAL_HIDDEN,
        NUM_EXPERTS, TOPK_K>(
        attn_base, my_weight, my_residual, bias, norm_weight, norm_output,
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
        /*attn_slice_release=*/rel,
        /*hier_release_lo=*/hier_lo,
        /*hier_release_hi=*/hier_hi,
        /*hier_local_lo=*/hloc_lo,
        /*hier_local_hi=*/hloc_hi,
        /*hier_l2_lo=*/hl2_lo,
        /*hier_l2_hi=*/hl2_hi,
        /*l2_fold=*/l2_fold);

    // ?? per-layer rendezvous across all 184 tiles, as phase 9 does ??
    __syncthreads();
    if (srdv_la != nullptr && srdv_ga == nullptr) {
      // --srdv=2, the control: no rendezvous at all. Wrong answer by
      // construction; the point is what the rendezvous costs.
    } else if (srdv_la != nullptr) {
      // Publish. A write-through store, so there is no pre-op value to return
      // and the wave has nothing to wait on afterwards; the vmcnt(0) below is
      // ordering this block's data ahead of its own arrival, not the arrival.
      int *const loc =
          ((upper && srdv_lb != nullptr) ? srdv_lb : srdv_la) +
          xcd * SENT_LOC_STRIDE;
      if (tid == 0) {
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        drv_st_wt_u32((unsigned int *)(loc + local), (unsigned)layer);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      }
      if (tid < 64) {
        // One lane per slot, so each poll iteration is one coalesced load
        // instead of `tiles` scalar ones. Lanes with no slot vote ready so the
        // __all() is still a whole-wave vote.
        if (local == 0) {
          for (;;) {
            int const v = ((int)tid < tiles) ? drv_ld_sys_s32(loc + (int)tid)
                                             : layer;
            if (__all(v >= layer)) {
              break;
            }
            __builtin_amdgcn_s_sleep(MPK_DRV_SRDV_SLEEP);
          }
          if (tid == 0) {
            // The whole boundary crossing: two stores, and nobody waits on
            // either. Consumers in each AID read the copy homed in that AID.
            drv_st_wt_u32((unsigned int *)(srdv_ga + xcd), (unsigned)layer);
            if (srdv_gb != nullptr) {
              drv_st_wt_u32((unsigned int *)(srdv_gb + xcd), (unsigned)layer);
            }
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          }
        }
        int *const glb = (upper && srdv_gb != nullptr) ? srdv_gb : srdv_ga;
        for (;;) {
          int const v =
              ((int)tid < NXCD) ? drv_ld_sys_s32(glb + (int)tid) : layer;
          if (__all(v >= layer)) {
            break;
          }
          __builtin_amdgcn_s_sleep(MPK_DRV_SRDV_SLEEP);
        }
      }
    } else if (tid == 0) {
      if (rdv_lo != nullptr) {
        // Three levels, and no counter is shared across the partition boundary.
        // Every counter here is monotonic rather than reset, so the arrival
        // tests are on the residue and the release test is `<`: a block that is
        // slow to leave layer L cannot then mistake layer L+1's release for its
        // own, which an equality test on a recycled counter would allow.
        // `half` is the tree's shape and is always a 4/4 split; `split_on` only
        // decides whether the two halves' lines are placed in different AIDs.
        int const half = (xcd >= NXCD / 2) ? 1 : 0;
        int *const my_rdv = (split_on && half) ? rdv_hi : rdv_lo;
        int *const peer_rdv = (split_on && !half) ? rdv_hi : rdv_lo;

        // Level 1: this XCD's 23 blocks, one line, all co-located.
        unsigned int const a1 =
            atomicAdd((unsigned int *)(my_rdv + RDV_L1_INTS(xcd)), 1u);
        if ((int)(a1 % (unsigned)tiles) == tiles - 1) {
          // Level 2: this half's 4 level-1 closers, one line, still all local.
          unsigned int const a2 =
              atomicAdd((unsigned int *)(my_rdv + RDV_L2_INTS(half)), 1u);
          if ((int)(a2 % (NXCD / 2u)) == NXCD / 2 - 1) {
            // Level 3: the only boundary crossing in the whole rendezvous, and
            // a store rather than a read-modify-write.
            drv_st_wt_u32(peer_rdv + RDV_PEER_INTS(1 - half), (unsigned)layer);
            while (drv_ld_sys_s32(my_rdv + RDV_PEER_INTS(half)) < layer) {
              __builtin_amdgcn_s_sleep(1);
            }
            drv_st_wt_u32(my_rdv + RDV_REL_INTS(half), (unsigned)layer);
          }
        }
        while (drv_ld_sys_s32(my_rdv + RDV_REL_INTS(half)) < layer) {
          __builtin_amdgcn_s_sleep(1);
        }
      } else {
        atomicAdd(bar, 1u);
        if (dual) {
          atomicAdd(bar_b, 1u);
        }
        while (drv_ld_sys_s32((int *)my_bar) < layer * NXCD * tiles) {
          __builtin_amdgcn_s_sleep(1);
        }
      }
    }
    __syncthreads();
  }
}

int main(int argc, char **argv) {
  int n_layers = 200;
  int delay_ticks = 0, delay_skew = 0;
  int tiles = TILES_PER_XCD;
  int use_aid = 0, split_on = 1, lsplit_on = 0, hrdv_on = 0, bsplit_on = 0;
  int srdv_on = 0;
  int dsplit_on = 0, probe_on = 0, catomic = 0;
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
    } else if (!strncmp(argv[i], "--dcoh=", 7)) {
      g_dcoh = atoi(argv[i] + 7) != 0;
    } else if (!strncmp(argv[i], "--split=", 8)) {
      split_on = atoi(argv[i] + 8);
    } else if (!strncmp(argv[i], "--lsplit=", 9)) {
      lsplit_on = atoi(argv[i] + 9);
    } else if (!strncmp(argv[i], "--hrdv=", 7)) {
      hrdv_on = atoi(argv[i] + 7);
    } else if (!strncmp(argv[i], "--srdv=", 7)) {
      srdv_on = atoi(argv[i] + 7);
    } else if (!strncmp(argv[i], "--bsplit=", 9)) {
      bsplit_on = atoi(argv[i] + 9);
    } else if (!strncmp(argv[i], "--dsplit=", 9)) {
      dsplit_on = atoi(argv[i] + 9);
    } else if (!strncmp(argv[i], "--probe=", 8)) {
      probe_on = atoi(argv[i] + 8);
    } else if (!strncmp(argv[i], "--catomic=", 10)) {
      catomic = atoi(argv[i] + 10);
    } else if (!strncmp(argv[i], "--tag=", 6)) {
      tag = argv[i] + 6;
    }
  }
  if (tiles < 1 || tiles > TILES_PER_XCD) {
    fprintf(stderr, "--tiles must be in [1, %d]\n", TILES_PER_XCD);
    return 2;
  }
  // The chase destroys what it walks, so the probe has to run before every fill
  // -- which is before --catomic replaces `counters`. Its "counters" row would
  // then describe the allocation that was thrown away, and it would look exactly
  // like a real placement reading. Refuse instead.
  if (probe_on && catomic) {
    fprintf(stderr, "ABORT: --probe and --catomic cannot be combined; the probe "
                    "runs before the counters buffer is replaced, so its row "
                    "would describe the discarded allocation\n");
    return 2;
  }

  void *attn = nullptr, *weight = nullptr, *residual = nullptr, *bias = nullptr;
  // Second homes for the three per-XCD data buffers, non-null only with
  // --dsplit. The kernel treats null as "one allocation, global index".
  void *attn_b = nullptr, *weight_b = nullptr, *residual_b = nullptr;
  void *nw = nullptr, *no = nullptr, *rw = nullptr, *rb = nullptr;
  void *logits = nullptr, *counters = nullptr, *out = nullptr, *tkw = nullptr;
  void *ridx = nullptr, *aeid = nullptr;
  int *flags = nullptr, *flags_b = nullptr, *xcd_seen = nullptr;
  int *hier_lo = nullptr, *hier_hi = nullptr;
  int *hloc_lo = nullptr, *hloc_hi = nullptr;
  int *rdv_lo = nullptr, *rdv_hi = nullptr;
  int *srdv_la = nullptr, *srdv_lb = nullptr;
  int *srdv_ga = nullptr, *srdv_gb = nullptr;
  int *hl2_lo = nullptr, *hl2_hi = nullptr;
  unsigned int *bar = nullptr, *bar_b = nullptr;
  int dual = 0;

  size_t const weight_bytes = (size_t)NXCD * tiles * WG_BYTES;
  // Each half holds only its own XCDs' slabs, so the pair costs the same VRAM as
  // the single allocation it replaces. This is a partition, not a replica.
  size_t const weight_half_bytes = (size_t)(NXCD / 2) * tiles * WG_BYTES;
  // A bitmask, because the two halves of this change are different mechanisms
  // and have to be attributed separately. Bit 1 homes the private per-XCD slabs
  // (weight, residual): one toucher each, so it is pure placement and cannot
  // cost anything. Bit 2 adds the attn_out replica, which buys a local read for
  // the all-to-all reduction but pays a remote store on the publish path. Only
  // measuring 3 conflates a free change with one that has a price.
  bool const d_wr = (dsplit_on & 1) != 0;
  bool const d_attn = (dsplit_on & 2) != 0;
  if (dsplit_on) {
    if (!use_aid) {
      fprintf(stderr, "ABORT: --dsplit needs --aid=1 (homing the data needs "
                      "the AID-local allocator)\n");
      return 2;
    }
    if (!aid_init()) {
      fprintf(stderr, "ABORT: --dsplit but AID-local allocation is "
                      "unavailable (stock driver, or not NPS2)\n");
      return 2;
    }
  }
  // coherent=false deliberately: these are data, not markers. A coherent MTYPE
  // on a 34 KiB-per-block working set would overflow the DF-CS shadow-tag
  // directory and hang, and nothing here needs coherence -- the weight and
  // residual slabs have a single toucher each, and attn_out's two replicas are
  // published by write-through stores ahead of the flag that gates them.
  //
  // These BOs therefore land on MTYPE_NC, the same type the single hipMalloc'd
  // allocation already gets: with aid_local_xcp_nc=Y the driver forces
  // is_local=false for all VRAM in a spanning XCP, and !coherent skips the
  // FLAGMTYPE override, so the chain falls through to NC. That is what makes
  // this an apples-to-apples test of homing rather than of caching.
  if (d_attn) {
    attn = alloc_in_aid(OPROJ_REDUCTION * 2, 0, g_dcoh);
    attn_b = alloc_in_aid(OPROJ_REDUCTION * 2, 1, g_dcoh);
  } else {
    HIP_OK(hipMalloc(&attn, OPROJ_REDUCTION * 2));
  }
  if (d_wr) {
    weight = alloc_in_aid(weight_half_bytes, 0, g_dcoh);
    weight_b = alloc_in_aid(weight_half_bytes, 1, g_dcoh);
    residual = alloc_in_aid(OUTPUT_STRIDE * 2, 0, g_dcoh);
    residual_b = alloc_in_aid(OUTPUT_STRIDE * 2, 1, g_dcoh);
  } else {
    HIP_OK(hipMalloc(&weight, weight_bytes));
    HIP_OK(hipMalloc(&residual, OUTPUT_STRIDE * 2));
  }
  if (attn == nullptr || weight == nullptr || residual == nullptr ||
      (d_attn && attn_b == nullptr) ||
      (d_wr && (weight_b == nullptr || residual_b == nullptr))) {
    fprintf(stderr, "ABORT: --dsplit could not place a data buffer in each "
                    "AID\n");
    return 2;
  }
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

  // Ahead of every memset and fill, because the chase overwrites what it walks.
  // Two modes on the same buffer, because they answer different questions and
  // only together are they evidence. sc0/sc1 bypasses the L2 and reports where
  // the pages really are, which is the only way to confirm a home for a buffer
  // small enough to be L2-resident -- and it is also the latency that the
  // write-through and atomic traffic below genuinely pays. nt reports what a
  // cached reader sees.
  //
  // What nt does NOT establish, and the first reading of this probe got wrong:
  // a flat nt row does not mean the buffer cannot be a source of skew. This is a
  // warm best-of-20 dependent chase, so it measures residency under ideal
  // conditions rather than in the staging loop. `weight` reads flat here at
  // ~82 ns from all eight XCDs, and homing it per AID still moved the entry
  // spread 0.585 -> 0.190 us. Flat nt plus skewed sc0/sc1 means "cached when
  // warm, placed where I asked" -- not "immune to placement".
  if (probe_on) {
    for (int mode = 0; mode < 2; ++mode) {
      printf("[PROBE] --- %s loads ---\n",
             mode ? "device-scope sc0 sc1" : "non-temporal");
      if (weight_b) {
        probe_placement("weight(AID0)", weight, weight_half_bytes, mode);
        probe_placement("weight(AID1)", weight_b, weight_half_bytes, mode);
      } else {
        probe_placement("weight(single)", weight, weight_bytes, mode);
      }
      // The buffers whose accesses cannot be cached, and which are therefore the
      // only data that can contribute to arrival skew:
      //   out       published with st_wt_u64 (sc0 sc1) and acquired cross-XCD
      //             with sc1 -- the callee says it bypasses L2 and lands in HBM
      //   counters  topk_counter, hit with atomicAdd from every block
      //   logits    written with st_wt_u16, also write-through
      probe_placement("out", out, OUTPUT_STRIDE * 2, mode);
      probe_placement("counters", counters, 64 * 1024, mode);
      probe_placement("logits", logits, 64 * 1024, mode);
      probe_placement("attn_out", attn, OPROJ_REDUCTION * 2, mode);
    }
  }

  HIP_OK(hipMemset(attn, 0, OPROJ_REDUCTION * 2));
  if (attn_b) {
    HIP_OK(hipMemset(attn_b, 0, OPROJ_REDUCTION * 2));
  }
  // Two fills, because a group's data and scale halves are different formats.
  // e2m1 0x2 is 1.0, so 0x22 is a pair of ones. The scale is E8M0 biased at
  // 127, and 0x77 is 2^-8, chosen so a 4096-long reduction of ones lands near
  // 20 instead of 4096: bf16 resolves 0.125 there, and one XCD's slice going
  // stale moves the result by 1.0, so it survives the store instead of
  // rounding away.
  //
  // The single 0x11 fill this replaces left every scale at 2^-110, which
  // underflowed all 4096 products and made the harness compute zeros.
  // Before the fill, because the chase overwrites what it walks. This is the
  // only statement in the harness about where the data actually is; everything
  // else is a request, not a confirmation.
  // Two modes on the same buffer, because they answer different questions and
  // only together are they evidence. nt reports the latency the staging read
  // actually sees, cache included -- if that is flat across XCDs the data cannot
  // be a source of skew whatever its home. sc0/sc1 bypasses the L2 and reports
  // where the pages really are, which is the only way to confirm the partition
  // landed in the AID it asked for when the buffer is small enough to be
  // L2-resident.
  // Every slab gets the same two fills whichever buffer it lives in, so the
  // partition is byte-for-byte the allocation it replaces and the result hash is
  // invariant to --dsplit by construction.
  void *const wbuf[2] = {weight, weight_b};
  int const wg_per_buf = weight_b ? (NXCD / 2) * tiles : NXCD * tiles;
  for (int b = 0; b < (weight_b ? 2 : 1); ++b) {
    for (int g = 0; g < wg_per_buf; ++g) {
      char *wg = (char *)wbuf[b] + (size_t)g * WG_BYTES;
      HIP_OK(hipMemset(wg, 0x22, WG_DATA_BYTES));
      HIP_OK(hipMemset(wg + WG_DATA_BYTES, 0x77, WG_SCALE_BYTES));
    }
  }
  HIP_OK(hipMemset(residual, 0, OUTPUT_STRIDE * 2));
  if (residual_b) {
    HIP_OK(hipMemset(residual_b, 0, OUTPUT_STRIDE * 2));
  }
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

    // --catomic: move the level-2 arrival counter out of plain hipMalloc, which
    // is the one piece of barrier state whose MTYPE nobody has been able to
    // choose. This is the direct test of the NPS1-versus-NPS2 question.
    //
    // The state of the argument it settles. Physically the counter line has ONE
    // home in both partition modes, and in both modes four of the eight dies sit
    // on the other AID, so "half the dies are remote to it" is as true of NPS1 as
    // of NPS2 -- topology cannot be the difference between 0.30 and 0.96 us. What
    // does differ is the MTYPE: an AID_LOCAL BO is local to its range, so
    // `is_local` holds and it takes `mtype_local` (MTYPE_RW, NPS1's default),
    // while plain hipMalloc VRAM in an XCP that spans both ranges is local to
    // neither and gets demoted to MTYPE_NC. `--coherent=1` never tested this: it
    // routes through `aid_local_flag_mtype`, which is 2, so it measured CC.
    //
    //   1  AID0, non-coherent  -> MTYPE_RW, the MTYPE NPS1 gives this line
    //   2  AID0, coherent      -> whatever aid_local_flag_mtype says (CC today)
    //   3  AID1, non-coherent  -> RW again, but homed in the other partition.
    //      The placement control: this file has been claiming that homing cannot
    //      reach an `sc1` atomic because it serialises at the device coherency
    //      point regardless. With MTYPE held fixed, this is that claim's test.
    if (catomic) {
      // Safe only because --split and --lsplit have emptied this buffer of
      // everything that is *polled*. What is left in it is the level-2 counter
      // and topk_counter, both touched exclusively by device-scope atomics,
      // which reach the coherency point and never read a cached copy. The
      // release flags and the level-1 lines would not be safe here: a
      // non-temporal poll of an MTYPE_RW line never observes another die's
      // write-through store, even a die in the same AID, and that is the
      // recorded hang rather than a slowdown.
      if (!split_on || !lsplit_on) {
        fprintf(stderr, "ABORT: --catomic needs --split=1 --lsplit=1, or the "
                        "release flags and level-1 lines stay in this buffer "
                        "and an nt poll of an RW line hangs\n");
        return 2;
      }
      void *ca = alloc_in_aid(64 * 1024, catomic == 3 ? 1 : 0, catomic == 2);
      if (ca == nullptr) {
        fprintf(stderr, "ABORT: could not place the counters buffer\n");
        return 2;
      }
      HIP_OK(hipFree(counters));
      counters = ca;
      HIP_OK(hipMemset(counters, 0, 64 * 1024));
    }
    bar = (unsigned int *)(flags + BAR_OFF_INTS);
    bar_b = (unsigned int *)(flags_b + BAR_OFF_INTS);
    // Only with --split=1. Handing both halves the same replica would be the
    // misrouted case: coherent lines whose readers span both partitions, which
    // hangs rather than running slowly.
    if (split_on) {
      hier_lo = flags + HIER_OFF_INTS;
      hier_hi = flags_b + HIER_OFF_INTS;
    }
    // Level 1 is independent of --split: it needs no replication, only homing.
    // Each line's 23 writers are one XCD's workers, already co-located with
    // each other, so there is no misroutable reader here at all.
    if (lsplit_on) {
      hloc_lo = flags + HIER_LOCAL_OFF_INTS;
      hloc_hi = flags_b + HIER_LOCAL_OFF_INTS;
    }
    // Also independent of --split, and for the same reason: every line in the
    // region is indexed by the half that owns it, so placing the two halves in
    // different AIDs partitions the state rather than replicating it.
    if (hrdv_on) {
      rdv_lo = flags + RDV_OFF_INTS;
      rdv_hi = flags_b + RDV_OFF_INTS;
    }
    // Unlike the two above, this one does depend on --split: splitting the
    // aggregation gives each half its own releaser, and a releaser can only
    // publish locally into a replica that is in its own partition.
    if (bsplit_on && split_on) {
      // --bsplit=2 is the negative control for the claim that placement does
      // nothing for an `sc0 sc1` atomic. It swaps the two regions, so every
      // half's arrival counter is deliberately homed in the *other* partition
      // and every level-2 atomic is remote. If the claim is right this costs
      // nothing measurable; if placement matters after all, this is where it
      // shows. Safe despite the coherent MTYPE because nothing here is a cached
      // reader: `sc1` atomics bypass the L2s by definition and the handshake
      // polls are `ld_nt`. The handshake's signal store is already remote in the
      // un-swapped form, so remote access to these buffers is nothing new.
      //
      // --bsplit=3 is the fold: the same two per-AID regions, but each half's
      // closer then does one atomic on the global line instead of the
      // handshake, so that line takes 2 arrivals rather than 8. It exists to
      // separate the two things --bsplit=1 changed at once -- fewer arrivals on
      // the shared line, and a handshake that cost 0.36 us -- and so to say
      // whether the level-2 rise from 0.66 to 0.96 under symmetric arrivals is
      // really serialization on that line.
      bool const swap = bsplit_on == 2;
      hl2_lo = (swap ? flags_b : flags) + HIER_L2_OFF_INTS;
      hl2_hi = (swap ? flags : flags_b) + HIER_L2_OFF_INTS;
    } else if (bsplit_on) {
      fprintf(stderr, "ABORT: --bsplit needs --split=1 (the per-half arrival "
                      "counters have to be homed in two partitions)\n");
      return 2;
    }
    if (srdv_on) {
      if (srdv_on == 2) {
        // tier 1 non-null, tier 2 null: the kernel reads that as "skip".
        srdv_la = flags + SENT_LOC_OFF_INTS;
      }
      if (srdv_on == 1 && tiles > SENT_LOC_STRIDE) {
        fprintf(stderr, "ABORT: --srdv needs tiles <= %d\n", SENT_LOC_STRIDE);
        return 2;
      }
      // Tier 1 goes to the replica homed in each XCD's own AID, so the array a
      // block writes and the array its aggregator polls are both local. Tier 2
      // is written to both replicas and read from whichever is local.
      srdv_la = flags + SENT_LOC_OFF_INTS;
      if (srdv_on == 1) {
        srdv_ga = flags + SENT_GLB_OFF_INTS;
      }
      if (split_on && srdv_on == 1) {
        srdv_lb = flags_b + SENT_LOC_OFF_INTS;
        srdv_gb = flags_b + SENT_GLB_OFF_INTS;
      }
    }
    HIP_OK(hipMemset(flags, 0, sync_bytes));
    HIP_OK(hipMemset(flags_b, 0, sync_bytes));
    dual = 1;
  } else {
    // Baseline: one hipMalloc'd flag page and one counter, both MTYPE_NC and
    // both read from either AID -- exactly what the model does today.
    flags_b = flags;
    bar_b = bar;
    split_on = 0;
    lsplit_on = 0;
    // No release replicas without AID placement, so there is nothing for a
    // second releaser to publish into. The shared counter stays.
    bsplit_on = 0;
    // And no way to ask for an MTYPE either: choosing one means allocating the
    // BO ourselves. In NPS1 this is moot -- the line is MTYPE_RW already, which
    // is the whole point of the comparison.
    catomic = 0;
    // The tree is still worth running with one coherence domain: it is the
    // control that says how much of its benefit is the shape and how much is the
    // placement. Both pointers alias, and the per-half line indices keep it a
    // correct 4/4 barrier anyway.
    if (hrdv_on) {
      void *r = nullptr;
      HIP_OK(hipMalloc(&r, RDV_REGION_BYTES));
      HIP_OK(hipMemset(r, 0, RDV_REGION_BYTES));
      rdv_lo = (int *)r;
      rdv_hi = (int *)r;
    }
  }

  printf("[%s] attn_out %p  flags %p / %p  weight %p (%.2f MiB)\n", tag, attn,
         (void *)flags, (void *)flags_b, weight, weight_bytes / 1048576.0);
  if (dsplit_on) {
    printf("[%s] dsplit: attn_out %p / %p  weight %p / %p  residual %p / %p\n",
           tag, attn, attn_b, weight, weight_b, residual, residual_b);
  }
  printf("[%s] srdv=%d\n", tag, srdv_on);
  printf("[%s] aid=%d coherent=%d split=%d hier_split=%d local_split=%d "
         "hier_rdv=%d l2_split=%d l2_fold=%d catomic=%d\n",
         tag, use_aid, (int)g_coherent, split_on, hier_lo != nullptr,
         hloc_lo != nullptr, rdv_lo != nullptr, hl2_lo != nullptr,
         bsplit_on == 3, catomic);
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
                     attn_b, weight_b, residual_b,
                     bias, nw, no, rw, rb, logits, counters, out, tkw, ridx,
                     aeid, flags, flags_b, bar, bar_b, hier_lo, hier_hi,
                     hloc_lo, hloc_hi, rdv_lo, rdv_hi, hl2_lo, hl2_hi, xcd_seen,
                     n_layers, delay_ticks, delay_skew, tiles, dual, split_on,
                     bsplit_on == 3, srdv_la, srdv_lb, srdv_ga, srdv_gb);
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

  // Correctness gate for the barrier. Inputs are constant memsets and the
  // per-layer producer value is a pure function of (layer, xcd), so both
  // buffers below are deterministic when the barrier holds. Two runs that
  // disagree on rmsnorm_out while agreeing on attn_proj_out mean the norm read
  // columns that were not yet published -- which is exactly what this barrier
  // exists to prevent, and what a routing mistake in the release flags would
  // cause.
  {
    size_t const out_bytes = OUTPUT_STRIDE * 2;
    size_t const norm_bytes = ACTUAL_HIDDEN * 2;
    size_t const big = out_bytes > norm_bytes ? out_bytes : norm_bytes;
    unsigned char *hb = (unsigned char *)malloc(big);
    if (hb != nullptr) {
      unsigned long long h[2];
      void *src[2] = {out, no};
      size_t len[2] = {out_bytes, norm_bytes};
      for (int k = 0; k < 2; ++k) {
        HIP_OK(hipMemcpy(hb, src[k], len[k], hipMemcpyDeviceToHost));
        unsigned long long v = 1469598103934665603ull;
        for (size_t i = 0; i < len[k]; ++i) {
          v ^= hb[i];
          v *= 1099511628211ull;
        }
        h[k] = v;
      }
      printf("[%s] hash attn_proj_out=%016llx rmsnorm_out=%016llx\n", tag, h[0],
             h[1]);
      free(hb);
    }
    // Which buffers the kernel actually touched. A zero nonzero-count means
    // the phase never published anything there, which is a different problem
    // from publishing the wrong thing.
    struct {
      char const *name;
      void *dev;
      size_t bytes;
    } probes[] = {
        // inputs first: this is the half of the dataflow still unexplained
        {"attn_out(in)", attn, OPROJ_REDUCTION * 2},
        {"weight_data(in)", weight, 256},
        {"weight_scale(in)", (char *)weight + WG_DATA_BYTES, 256},
        {"norm_weight(in)", nw, 256},
        {"attn_proj_out", out, OUTPUT_STRIDE * 2},
        {"rmsnorm_out", no, ACTUAL_HIDDEN * 2},
        {"topk_weight", tkw, 4096},
        {"routing_indices", ridx, 4096},
        {"logits_scratch", logits, 4096},
        {"counters", counters, 4096},
    };
    for (unsigned p = 0; p < sizeof(probes) / sizeof(probes[0]); ++p) {
      unsigned char *b = (unsigned char *)malloc(probes[p].bytes);
      if (b == nullptr) {
        continue;
      }
      HIP_OK(hipMemcpy(b, probes[p].dev, probes[p].bytes, hipMemcpyDeviceToHost));
      size_t nz = 0;
      for (size_t i = 0; i < probes[p].bytes; ++i) {
        nz += (b[i] != 0);
      }
      unsigned int const *w = (unsigned int const *)b;
      printf("[%s] %-16s %6zu/%zu nonzero bytes  head %08x %08x %08x %08x\n",
             tag, probes[p].name, nz, probes[p].bytes, w[0], w[1], w[2], w[3]);
      free(b);
    }
    {
    }
  }
  return 0;
}
