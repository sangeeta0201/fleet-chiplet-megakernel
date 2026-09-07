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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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
    int *flags, unsigned int *bar, int *xcd_seen, int n_layers,
    int delay_ticks, int delay_skew, int tiles) {
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
        drv_st_wt_u32(&flags[xcd * FLAG_STRIDE], (unsigned)layer);
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
        /*attn_slice_release=*/flags);

    // ?? per-layer rendezvous across all 184 tiles, as phase 9 does ??
    __syncthreads();
    if (tid == 0) {
      atomicAdd(bar, 1u);
      while (drv_ld_sys_s32((int *)bar) < layer * NXCD * tiles) {
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
  int *flags = nullptr, *xcd_seen = nullptr;
  unsigned int *bar = nullptr;

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

  printf("[%s] attn_out %p  flags %p  weight %p (%.2f MiB)\n", tag, attn,
         (void *)flags, weight, weight_bytes / 1048576.0);
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
                     aeid, flags, bar, xcd_seen, n_layers, delay_ticks,
                     delay_skew, tiles);
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
