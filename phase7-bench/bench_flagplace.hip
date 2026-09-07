// Flag-placement probe for the Phase 7 cross-XCD attention-slice wait.
//
// The production kernel puts the eight release flags at attn_release[x * 16],
// so they are 64 B apart and all eight land inside one 4 KiB page. This asks
// whether that address packing is what costs 27x in NPS2, by making the stride
// a runtime knob and sweeping it.
//
// It also splits the wait into terms the in-kernel [OPROJ_INNER] slicewait
// bucket lumps together, which matters because the per-layer rendezvous is
// itself a cross-AID barrier and can release a producer late:
//
//   poll = t1 - t0           time this wave spent in the spin loop
//   vis  = t1 - rel_ts_max   flag store -> observation, i.e. real visibility
//   skew = rel_ts_max - t0   how late the producer published vs. poll start
//   inv  = t2 - t1           the buffer_inv that follows the spin
//
// poll == skew + vis by construction, so a large poll with small vis is the
// harness waiting on a late producer, while a large vis is the flag itself
// being slow to become visible. Only the latter is a placement problem.
//
// s_memrealtime is a device-wide 100 MHz reference on gfx950, so one tick is
// 10 ns and timestamps taken on different XCDs are directly comparable. That
// is what makes the cross-XCD vis term meaningful.

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
#define RELTS_STRIDE 64 // ints; give each producer's timestamp its own line

#define HIP_OK(call)                                                           \
  do {                                                                         \
    hipError_t _e = (call);                                                    \
    if (_e != hipSuccess) {                                                    \
      fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #call,           \
              hipGetErrorString(_e));                                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

__device__ unsigned int g_claim[NXCD];

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
  int val;
  asm volatile("global_load_dword %0, %1, off sc0 sc1\n"
               "s_waitcnt vmcnt(0)"
               : "=v"(val)
               : "v"(addr)
               : "memory");
  return val;
}

// A weaker scope was tried here and does not work: a load that may hit vL1
// keeps re-reading the line it already holds, so the spin only retires when
// that line happens to be evicted. Every gate below is sc0 sc1 for that
// reason; the hierarchical arm wins by issuing fewer of them, not weaker ones.

__global__ __launch_bounds__(NTHREADS) void flagplace(
    unsigned short *attn_out, int *flags, int *local_flags, unsigned int *rel_ts,
    unsigned int *bar, int *xcd_seen, unsigned int *s_poll, unsigned int *s_inv,
    unsigned int *s_vis, int *s_skew, int n_layers, int tiles, int stride_ints,
    int delay_skew, int hier, int sleep_n, int n_copies) {
  int const xcd = get_xcd();
  int const tid = threadIdx.x;
  int const warp = tid >> 6;
  int const lane = tid & 63;

  // Claim a rank within this XCD rather than trusting a blockIdx -> XCD map.
  __shared__ int s_local;
  if (tid == 0) {
    s_local = (int)atomicAdd(&g_claim[xcd], 1u);
    atomicAdd(&xcd_seen[xcd], 1);
  }
  __syncthreads();
  int const local = s_local;
  if (local >= tiles) {
    return; // never join the barrier, or it can never retire
  }

  // Wave w covers K elements [1024w, 1024w+1024) = slices 2w and 2w+1, the
  // same forced mapping the real kernel has.
  int const first_xcd = warp * 2;
  // With --copies=K the eight-flag array is replicated K times and block rank
  // `local` reads copy `local % K`. Producers write every copy, so the
  // information is identical and only the *sharing* of each line changes: K=1
  // is 92 waves per line, K=23 is 4. Total poll count is untouched. That
  // separates per-line serialization from a global request-rate limit, and it
  // is the general form of "give each AID its own flag page".
  int const copy = local % n_copies;
  int *rel0 = flags + (copy * NXCD + first_xcd) * stride_ints;
  int *rel1 = flags + (copy * NXCD + first_xcd + 1) * stride_ints;
  unsigned int *my_slice = (unsigned int *)(attn_out + xcd * ATTN_SLICE);
  unsigned int const *src =
      (unsigned int const *)(attn_out + first_xcd * ATTN_SLICE);
  int const row = ((xcd * tiles + local) * NWAVES + warp) * n_layers;

  unsigned int checksum = 0;

  for (int layer = 1; layer <= n_layers; ++layer) {
    // ---- producer: rank 0 on each XCD stands in for the attention merge ----
    if (local == 0) {
      if (delay_skew) {
        unsigned long long const until = __builtin_amdgcn_s_memrealtime() +
                                         (unsigned long long)(delay_skew * xcd);
        while (__builtin_amdgcn_s_memrealtime() < until) {
          __builtin_amdgcn_s_sleep(1);
        }
      }
      // 256 threads x 1 dword = 512 bf16 = this XCD's slice.
      st_wt_u32(&my_slice[tid], (unsigned)(layer * 2654435761u + xcd));
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      __syncthreads();
      if (tid == 0) {
        // Publish the release time before the flag, and drain in between, so a
        // consumer that has seen the flag is guaranteed to see the timestamp.
        st_wt_u32(&rel_ts[xcd * RELTS_STRIDE],
                  (unsigned)__builtin_amdgcn_s_memrealtime());
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        for (int c = 0; c < n_copies; ++c) {
          st_wt_u32(&flags[(c * NXCD + xcd) * stride_ints], (unsigned)layer);
        }
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      }
    }

    // ---- consumer: the span the real kernel calls slicewait ----
    unsigned long long const t0 = __builtin_amdgcn_s_memrealtime();
    if (hier) {
      // Two-level gate, all of it still system-scope so it is correct under
      // MTYPE_NC. Per XCD, the rank-0 block's wave 0 polls all eight remote
      // flags and republishes one XCD-local flag; every other block's wave 0
      // polls only that. Waves 1-3 of every block do not poll at all and pick
      // the result up through __syncthreads().
      //
      // Flag-polls per layer drop from 736 waves x 2 = 1472 to
      // 8 x 8 + 176 x 1 = 240. hier=3 additionally replicates the XCD-local
      // flag over eight lines so the 22 waiting blocks per XCD do not all
      // queue on one line.
      if (warp == 0) {
        if (local == 0) {
          for (int x = 0; x < NXCD; ++x) {
            while (ld_sys_s32(flags + x * stride_ints) < layer) {
              for (int s = 0; s < sleep_n; ++s) {
                __builtin_amdgcn_s_sleep(1);
              }
            }
          }
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          if (lane == 0) {
            int const ncopy = (hier >= 3) ? 8 : 1;
            for (int c = 0; c < ncopy; ++c) {
              st_wt_u32(&local_flags[(xcd * 8 + c) * 16], (unsigned)layer);
            }
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          }
        } else {
          int const copy = (hier >= 3) ? (local & 7) : 0;
          while (ld_sys_s32(&local_flags[(xcd * 8 + copy) * 16]) < layer) {
            for (int s = 0; s < sleep_n; ++s) {
              __builtin_amdgcn_s_sleep(1);
            }
          }
        }
      }
      __syncthreads();
    } else {
      while (ld_sys_s32(rel0) < layer || ld_sys_s32(rel1) < layer) {
        for (int s = 0; s < sleep_n; ++s) {
          __builtin_amdgcn_s_sleep(1);
        }
      }
    }
    unsigned long long const t1 = __builtin_amdgcn_s_memrealtime();
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    asm volatile("buffer_inv" ::: "memory");
    unsigned long long const t2 = __builtin_amdgcn_s_memrealtime();

    // Read the 2 KiB this wave's K interval covers, so the acquire is real.
    unsigned int acc = 0;
#pragma unroll
    for (int i = 0; i < (2 * ATTN_SLICE) / 2 / 64; ++i) {
      acc += src[lane + i * 64];
    }
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    checksum += acc;

    if (lane == 0) {
      s_poll[row + layer - 1] = (unsigned int)(t1 - t0);
      s_inv[row + layer - 1] = (unsigned int)(t2 - t1);
      // Reading rel_ts costs two more system-scope loads, so only the rank-0
      // blocks do it. 32 waves per layer is plenty for a median and keeps the
      // extra traffic from perturbing everyone else's spin.
      if (local == 0) {
        unsigned int const r0 =
            (unsigned int)ld_sys_s32((int *)&rel_ts[first_xcd * RELTS_STRIDE]);
        unsigned int const r1 = (unsigned int)ld_sys_s32(
            (int *)&rel_ts[(first_xcd + 1) * RELTS_STRIDE]);
        unsigned int const rmax = ((int)(r1 - r0) > 0) ? r1 : r0;
        s_vis[row + layer - 1] = (unsigned int)t1 - rmax;
        s_skew[row + layer - 1] = (int)(rmax - (unsigned int)t0);
      }
    }

    // ---- per-layer rendezvous across every block, as phase 9 does ----
    __syncthreads();
    if (tid == 0) {
      atomicAdd(bar, 1u);
      while (ld_sys_s32((int *)bar) < layer * NXCD * tiles) {
        __builtin_amdgcn_s_sleep(1);
      }
    }
    __syncthreads();
  }

  if (checksum == 0xdeadbeefu && lane == 0) {
    flags[1] = (int)checksum;
  }
}

static double pct(std::vector<unsigned int> &v, double p) {
  if (v.empty()) {
    return 0.0;
  }
  size_t i = (size_t)(p * (double)(v.size() - 1));
  return (double)v[i] * 10.0 / 1000.0; // ticks -> us
}

int main(int argc, char **argv) {
  int n_layers = 200;
  int tiles = 23;
  int delay_skew = 0;
  long stride_bytes = 64;
  int hier = 0, sleep_n = 1, n_copies = 1;
  char const *tag = "run";
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--layers=", 9)) {
      n_layers = atoi(argv[i] + 9);
    } else if (!strncmp(argv[i], "--tiles=", 8)) {
      tiles = atoi(argv[i] + 8);
    } else if (!strncmp(argv[i], "--flagstride=", 13)) {
      stride_bytes = atol(argv[i] + 13);
    } else if (!strncmp(argv[i], "--skew=", 7)) {
      delay_skew = atoi(argv[i] + 7);
    } else if (!strncmp(argv[i], "--hier=", 7)) {
      hier = atoi(argv[i] + 7);
    } else if (!strncmp(argv[i], "--sleep=", 8)) {
      sleep_n = atoi(argv[i] + 8);
    } else if (!strncmp(argv[i], "--copies=", 9)) {
      n_copies = atoi(argv[i] + 9);
    } else if (!strncmp(argv[i], "--tag=", 6)) {
      tag = argv[i] + 6;
    }
  }
  if (sleep_n < 1) {
    sleep_n = 1;
  }
  if (n_copies < 1) {
    n_copies = 1;
  }
  if (stride_bytes < 4 || (stride_bytes & 3) != 0) {
    fprintf(stderr, "--flagstride must be a multiple of 4 and >= 4\n");
    return 2;
  }
  int const stride_ints = (int)(stride_bytes / 4);

  unsigned short *d_attn = nullptr;
  int *d_flags = nullptr, *d_local = nullptr, *d_seen = nullptr,
      *d_skew = nullptr;
  unsigned int *d_relts = nullptr, *d_bar = nullptr;
  unsigned int *d_poll = nullptr, *d_inv = nullptr, *d_vis = nullptr;

  size_t const nsamp = (size_t)NXCD * tiles * NWAVES * n_layers;
  size_t const flag_bytes = (size_t)n_copies * NXCD * stride_bytes + 4096;
  HIP_OK(hipMalloc(&d_attn, REDUCTION * sizeof(unsigned short)));
  HIP_OK(hipMalloc(&d_flags, flag_bytes));
  HIP_OK(hipMalloc(&d_local, 8192));
  HIP_OK(hipMemset(d_local, 0, 8192));
  HIP_OK(hipMalloc(&d_relts, NXCD * RELTS_STRIDE * sizeof(unsigned int)));
  HIP_OK(hipMalloc(&d_bar, sizeof(unsigned int)));
  HIP_OK(hipMalloc(&d_seen, NXCD * sizeof(int)));
  HIP_OK(hipMalloc(&d_poll, nsamp * sizeof(unsigned int)));
  HIP_OK(hipMalloc(&d_inv, nsamp * sizeof(unsigned int)));
  HIP_OK(hipMalloc(&d_vis, nsamp * sizeof(unsigned int)));
  HIP_OK(hipMalloc(&d_skew, nsamp * sizeof(int)));
  HIP_OK(hipMemset(d_attn, 0, REDUCTION * sizeof(unsigned short)));
  HIP_OK(hipMemset(d_flags, 0, flag_bytes));
  HIP_OK(hipMemset(d_relts, 0, NXCD * RELTS_STRIDE * sizeof(unsigned int)));
  HIP_OK(hipMemset(d_bar, 0, sizeof(unsigned int)));
  HIP_OK(hipMemset(d_seen, 0, NXCD * sizeof(int)));
  HIP_OK(hipMemset(d_poll, 0, nsamp * sizeof(unsigned int)));
  HIP_OK(hipMemset(d_inv, 0, nsamp * sizeof(unsigned int)));
  HIP_OK(hipMemset(d_vis, 0, nsamp * sizeof(unsigned int)));
  HIP_OK(hipMemset(d_skew, 0, nsamp * sizeof(int)));

  printf("[%s] flags %p  stride %ld B (%zu B span)  %d tiles/XCD  %d waves "
         "polling  hier=%d sleep=%d  %d layers\n",
         tag, (void *)d_flags, stride_bytes, (size_t)(NXCD - 1) * stride_bytes,
         tiles, NXCD * tiles * NWAVES, hier, sleep_n, n_layers);

  hipEvent_t e0, e1;
  HIP_OK(hipEventCreate(&e0));
  HIP_OK(hipEventCreate(&e1));
  HIP_OK(hipEventRecord(e0));
  hipLaunchKernelGGL(flagplace, dim3(NXCD * tiles), dim3(NTHREADS), 0, 0,
                     d_attn, d_flags, d_local, d_relts, d_bar, d_seen, d_poll,
                     d_inv, d_vis, d_skew, n_layers, tiles, stride_ints,
                     delay_skew, hier, sleep_n, n_copies);
  HIP_OK(hipEventRecord(e1));
  HIP_OK(hipDeviceSynchronize());
  HIP_OK(hipGetLastError());
  float ms = 0.0f;
  HIP_OK(hipEventElapsedTime(&ms, e0, e1));

  int seen[NXCD];
  HIP_OK(hipMemcpy(seen, d_seen, sizeof(seen), hipMemcpyDeviceToHost));
  printf("[%s] blocks per XCD:", tag);
  for (int i = 0; i < NXCD; ++i) {
    printf(" %d", seen[i]);
  }
  printf("\n");
  for (int i = 0; i < NXCD; ++i) {
    if (seen[i] < tiles) {
      fprintf(stderr,
              "ABORT: XCD %d got %d blocks, wanted %d; the producer set is "
              "incomplete so the spin could never retire.\n",
              i, seen[i], tiles);
      return 2;
    }
  }

  std::vector<unsigned int> h_poll(nsamp), h_inv(nsamp), h_vis(nsamp);
  std::vector<int> h_skew(nsamp);
  HIP_OK(hipMemcpy(h_poll.data(), d_poll, nsamp * 4, hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(h_inv.data(), d_inv, nsamp * 4, hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(h_vis.data(), d_vis, nsamp * 4, hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(h_skew.data(), d_skew, nsamp * 4, hipMemcpyDeviceToHost));

  int const skip = n_layers / 10; // discard warmup
  std::vector<unsigned int> poll, inv, vis;
  std::vector<int> skew;
  for (int x = 0; x < NXCD; ++x) {
    for (int lo = 0; lo < tiles; ++lo) {
      for (int w = 0; w < NWAVES; ++w) {
        size_t const base = (size_t)((x * tiles + lo) * NWAVES + w) * n_layers;
        for (int l = skip; l < n_layers; ++l) {
          poll.push_back(h_poll[base + l]);
          inv.push_back(h_inv[base + l]);
          if (lo == 0) { // only rank-0 blocks recorded vis/skew
            vis.push_back(h_vis[base + l]);
            skew.push_back(h_skew[base + l]);
          }
        }
      }
    }
  }
  std::sort(poll.begin(), poll.end());
  std::sort(inv.begin(), inv.end());
  std::sort(vis.begin(), vis.end());
  std::sort(skew.begin(), skew.end());

  printf("[%s] wall %.2f ms / %d layers = %.2f us per layer\n", tag, ms,
         n_layers, ms * 1000.0 / n_layers);
  printf("[%s] RESULT stride=%ld copies=%d (%d waves/line) hier=%d sleep=%d "
         "tiles=%d poll_p50=%.2f vis_p50=%.2f skew_p50=%.2f inv_p50=%.2f "
         "poll_p90=%.2f vis_p90=%.2f\n",
         tag, stride_bytes, n_copies,
         NXCD * ((tiles + n_copies - 1) / n_copies), hier, sleep_n, tiles,
         pct(poll, 0.50), pct(vis, 0.50),
         (double)skew[skew.size() / 2] * 10.0 / 1000.0, pct(inv, 0.50),
         pct(poll, 0.90), pct(vis, 0.90));
  return 0;
}
