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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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
    unsigned short *attn_out, int *flags, unsigned int *bar,
    unsigned long long *samp_wait, unsigned long long *samp_read,
    unsigned int *sink, int n_layers, int delay_ticks, int delay_skew) {
  int const xcd = get_xcd();
  int const tid = threadIdx.x;
  int const warp = tid >> 6;
  int const lane = tid & 63;

  // Wave w covers K elements [1024w, 1024w+1024), which is exactly slices
  // 2w and 2w+1 -- the same forced mapping as the real kernel.
  int const first_xcd = warp * 2;
  int *rel0 = &flags[first_xcd * FLAG_STRIDE];
  int *rel1 = &flags[(first_xcd + 1) * FLAG_STRIDE];
  unsigned int const *src =
      (unsigned int const *)(attn_out + first_xcd * ATTN_SLICE);
  unsigned int *my_slice = (unsigned int *)(attn_out + xcd * ATTN_SLICE);

  unsigned long long const *sample_base = nullptr;
  (void)sample_base;
  int const row = (xcd * NWAVES + warp) * n_layers;

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
    // 256 threads x 1 dword = 256 dwords = 512 bf16 = this XCD's slice.
    st_wt_u32(&my_slice[tid], (unsigned)(layer * 2654435761u + xcd));
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __syncthreads();
    if (tid == 0) {
      st_wt_u32(&flags[xcd * FLAG_STRIDE], (unsigned)layer);
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
      atomicAdd(bar, 1u);
      while (ld_sys_s32((int *)bar) < layer * NXCD) {
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
  char const *tag = "run";
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--layers=", 9)) {
      n_layers = atoi(argv[i] + 9);
    } else if (!strncmp(argv[i], "--delay=", 8)) {
      delay_ticks = atoi(argv[i] + 8);
    } else if (!strncmp(argv[i], "--skew=", 7)) {
      delay_skew = atoi(argv[i] + 7);
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
  int *d_flags = nullptr;
  unsigned int *d_bar = nullptr, *d_sink = nullptr;
  unsigned long long *d_wait = nullptr, *d_read = nullptr;
  size_t const nsamp = (size_t)NXCD * NWAVES * n_layers;
  HIP_OK(hipMalloc(&d_attn, REDUCTION * sizeof(unsigned short)));
  HIP_OK(hipMalloc(&d_flags, 4096));
  HIP_OK(hipMalloc(&d_bar, sizeof(unsigned int)));
  HIP_OK(hipMalloc(&d_sink, sizeof(unsigned int)));
  HIP_OK(hipMalloc(&d_wait, nsamp * sizeof(unsigned long long)));
  HIP_OK(hipMalloc(&d_read, nsamp * sizeof(unsigned long long)));
  HIP_OK(hipMemset(d_attn, 0, REDUCTION * sizeof(unsigned short)));
  HIP_OK(hipMemset(d_flags, 0, 4096));
  HIP_OK(hipMemset(d_bar, 0, sizeof(unsigned int)));
  HIP_OK(hipMemset(d_sink, 0, sizeof(unsigned int)));

  printf("attn_out %p  flags %p  (%d layers, delay=%d skew=%d ticks)\n",
         (void *)d_attn, (void *)d_flags, n_layers, delay_ticks, delay_skew);

  hipEvent_t e0, e1;
  HIP_OK(hipEventCreate(&e0));
  HIP_OK(hipEventCreate(&e1));
  HIP_OK(hipEventRecord(e0));
  hipLaunchKernelGGL(phase7, dim3(NXCD), dim3(NTHREADS), 0, 0, d_attn, d_flags,
                     d_bar, d_wait, d_read, d_sink, n_layers, delay_ticks,
                     delay_skew);
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

  std::vector<unsigned long long> all_w, all_r;
  printf("\n%-14s", "consumer XCD");
  for (int w = 0; w < NWAVES; ++w) {
    printf("  wave%d(xcd%d,%d)", w, w * 2, w * 2 + 1);
  }
  printf("\n");
  for (int x = 0; x < NXCD; ++x) {
    printf("  xcd%d wait us ", x);
    for (int w = 0; w < NWAVES; ++w) {
      std::vector<unsigned long long> v;
      for (int l = skip; l < n_layers; ++l) {
        v.push_back(h_wait[(size_t)(x * NWAVES + w) * n_layers + l]);
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
      for (int l = skip; l < n_layers; ++l) {
        v.push_back(h_read[(size_t)(x * NWAVES + w) * n_layers + l]);
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
