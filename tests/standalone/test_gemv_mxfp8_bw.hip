/* Copyright 2025 CMU
 *
 * Bandwidth probe for the narrow-tile MXFP8 GEMV
 * (gang_gemv_mxfp8_kernel in tasks/mi300/gang_gemv_mxfp8_mi300.cuh).
 *
 * Why this exists. The GLM-4.7-Flash megakernel's subphase timers say the
 * absorbed o_proj moves 21.6 MB in 14.8 us on 128 workgroups -- 1.46 TB/s, or
 * 11.4 GB/s per workgroup -- while the MoE W13 phase on 240 workgroups gets
 * 3.15 TB/s, 13.1 GB/s per workgroup. Both are far under what MI355X's HBM3E
 * can do. Two very different fixes follow from the two possible causes:
 *
 *   workgroup-count limited   Each WG is already pulling all it can, and the
 *                             o_proj is slow only because 2048 / 16 rows gives
 *                             128 WGs out of 240. Fix = re-tile so every phase
 *                             is 240 WGs wide.
 *
 *   per-WG MLP limited        A WG cannot keep enough loads in flight, so more
 *                             WGs would keep scaling. Fix = deeper unroll /
 *                             more outstanding loads, which helps every phase
 *                             at once and is worth far more.
 *
 * Kernel A settles it in the abstract: a pure streaming read, sweeping the
 * block count and the number of dwordx4 loads a thread keeps in flight. If
 * bandwidth saturates at few blocks, the machine is MLP-limited per WG; if it
 * keeps climbing with blocks, it is WG-limited.
 *
 * Kernel B is the real GEMV over a buffer far larger than the 256 MB MALL, so
 * the traffic is genuinely HBM and comparable to the in-megakernel number. The
 * megakernel's 21.6 MB of o_proj weight would otherwise sit in cache here and
 * report a bandwidth that has nothing to do with the measured one.
 *
 * ── what the sweep concluded ───────────────────────────────────────────────
 *
 * Neither of the two causes above. Both fixes were tried and neither closed
 * the gap: staging the activation row in LDS and forcing addrspace(1) took
 * kernel B from 14.5 to 20.7 GB/s/WG at 128 blocks, and yet the same change in
 * the megakernel moved the o_proj subphase by 2.9% and the decode step not at
 * all. Deeper unrolling (kernel H, up to the full 40 chunks a thread owns) and
 * software-pipelining the LDS fill against the first weight batch both did
 * nothing, so it is not memory-level parallelism either.
 *
 * What actually costs the factor of two is the *shape*, and kernel G is what
 * showed it. Kernels A..F all run a grid-stride loop, so a block chews through
 * ~184 tiles back to back and one tile's drain overlaps the next tile's ramp.
 * gang_oproj_router_fused gives each worker exactly ONE tile per layer
 * (oproj_tiles_per_xcd == 16, tile_idx == xcd_rank) and then a cross-XCD
 * barrier re-synchronises all 128 of them, so every block ramps and drains in
 * lockstep with every other and neither tail overlaps anything. Same kernel,
 * same access pattern, same 128 blocks:
 *
 *     184 tiles back to back      8.21 us/tile   20.7 GB/s/WG
 *     one tile + global barrier  11.98 us/tile   14.1 GB/s/WG
 *     megakernel, measured       14.43 us/tile   11.7 GB/s/WG
 *
 * So the kernel was never the thing to fix, and the lever is to give a worker
 * more consecutive work between barriers -- deeper fusion -- or more workers:
 * kernel G at 240 blocks moves 1741 GB/s against 128 blocks' 1244, because
 * per-WG efficiency falls more slowly than the block count rises.
 *
 * Build: ./build.sh
 * Run:   ./test_gemv_mxfp8_bw
 */

#include <cstdio>
#include <cstdlib>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <vector>

// Same two stubs test_mxfp8_linear.hip supplies, for the same reason: inside
// the megakernel they come from headers included ahead of the task bodies.
#define MPK_WS_WAVE_SYNC(wave) ((void)0)
namespace kernel {
__device__ __forceinline__ int _gang_moe_get_xcd_id() {
  int xcd_id;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcd_id));
  return xcd_id;
}
} // namespace kernel

#include "mirage/persistent_kernel/mpk_atoms.cuh"
#include "mirage/persistent_kernel/tasks/mi300/gang_gemv_mxfp8_mi300.cuh"

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t err = call;                                                     \
    if (err != hipSuccess) {                                                   \
      fprintf(stderr,                                                          \
              "HIP error %s:%d: %s\n",                                         \
              __FILE__,                                                        \
              __LINE__,                                                        \
              hipGetErrorString(err));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// ───────────────────────────────────────────────────────────────────────────
// Kernel A: pure streaming read, UNROLL dwordx4 loads in flight per thread.
//
// The grid strides over the whole buffer, so every configuration moves exactly
// the same bytes and the only variables are the block count and the depth.
// The accumulator is written out under a condition the compiler cannot fold,
// so nothing is dead-code eliminated.
// ───────────────────────────────────────────────────────────────────────────
template <int UNROLL>
__global__ __launch_bounds__(256) void stream_read_kernel(uint4 const *__restrict__ src,
                                                          size_t n_vec,
                                                          unsigned *sink) {
  size_t const stride = (size_t)gridDim.x * 256 * UNROLL;
  size_t const base0 = (size_t)blockIdx.x * 256 * UNROLL + threadIdx.x;
  unsigned acc = 0;
  for (size_t b = base0; b < n_vec; b += stride) {
    uint4 v[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; u++) {
      size_t const i = b + (size_t)u * 256;
      v[u] = src[i < n_vec ? i : n_vec - 1];
    }
#pragma unroll
    for (int u = 0; u < UNROLL; u++) {
      acc += v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    }
  }
  if (acc == 0xdeadbeefu) {
    sink[0] = acc;
  }
}

// ───────────────────────────────────────────────────────────────────────────
// Kernel B: the real GEMV, grid-strided over n_tiles workgroup tiles.
//
// This is exactly how the megakernel drives it -- a fixed pool of persistent
// workers each looping over the tiles it owns -- so `blocks` here plays the
// role of the worker count and `n_tiles` the role of the output width.
// ───────────────────────────────────────────────────────────────────────────
template <int K, int ROWS_PER_WG>
__global__ __launch_bounds__(256) void gemv_driver(void const *input,
                                                   void const *weight,
                                                   void *output,
                                                   int n_tiles,
                                                   int o_stride) {
  for (int t = blockIdx.x; t < n_tiles; t += gridDim.x) {
    kernel::gang_gemv_mxfp8_kernel<1, K, ROWS_PER_WG, false, false>(
        input,
        weight,
        nullptr,
        output,
        /*num_active_tokens=*/1,
        /*tile_n=*/ROWS_PER_WG,
        o_stride,
        /*m_tiles=*/1,
        n_tiles,
        /*wgm=*/0,
        t);
  }
}

// ───────────────────────────────────────────────────────────────────────────
// Kernel C/D: the same GEMV with the activation row staged in LDS.
//
// The hypothesis kernel B's numbers point at. A workgroup's ROWS_PER_WG rows
// all walk the *same* activation row, so the current inner loop issues the
// identical global load ROWS_PER_WG times: for the absorbed o_proj that is
// 16 x 20 KB = 320 KB of activation traffic against 164 KB of weight. Even
// served entirely from L2 those loads occupy issue slots and, worse, vmcnt
// slots -- the weight load a lane is actually waiting on cannot retire ahead
// of them, so the depth-8 unroll buys a third of the memory-level parallelism
// it appears to.
//
// Staged once into LDS the inner loop issues nothing but the weight
// dwordx4 (STAGE_SCALES also lifts the E8M0 byte out), which is exactly the
// shape of kernel A's 38 GB/s/WG streaming read.
// ───────────────────────────────────────────────────────────────────────────
template <int K, int ROWS_PER_WG, bool STAGE_SCALES, int UNROLL_IN = 0,
          bool PER_TILE = false>
__global__ __launch_bounds__(256) void gemv_lds_driver(void const *input,
                                                       void const *weight,
                                                       void *output,
                                                       int n_tiles,
                                                       int o_stride) {
  using kernel::gang_gemv_detail::f2b;
  using kernel::gang_gemv_mxfp8_detail::cvt_fp8_pair;
  using kernel::gang_gemv_mxfp8_detail::e8m0_to_f32;

  constexpr int NTHREADS = 256;
  constexpr int LANES_PER_ROW = NTHREADS / ROWS_PER_WG;
  constexpr int VEC = 16;
  constexpr int SCALE_BLOCK = 32;
  constexpr int ITERS = K / (LANES_PER_ROW * VEC);
  constexpr int WG_DATA_BYTES = ROWS_PER_WG * K;
  constexpr int WG_BYTES = WG_DATA_BYTES + ROWS_PER_WG * (K / SCALE_BLOCK);
  constexpr int SCALES_PER_ROW = K / SCALE_BLOCK;

  __shared__ unsigned short s_a[K];
  __shared__ unsigned char s_s[STAGE_SCALES ? ROWS_PER_WG * SCALES_PER_ROW : 1];

  int const tid = threadIdx.x;
  int const row = tid / LANES_PER_ROW;
  int const lane = tid % LANES_PER_ROW;

  // One cooperative pass over the activation row, 8 bf16 per thread per step.
  // PER_TILE moves the fill inside the tile loop, which is what the megakernel
  // really pays: a worker enters gang_gemv_mxfp8_kernel once per tile, so the
  // fill amortizes over one tile's weight, not over the block's whole share.
  if constexpr (!PER_TILE) {
    uint4 const *src = static_cast<uint4 const *>(input);
    uint4 *dst = reinterpret_cast<uint4 *>(s_a);
#pragma unroll
    for (int i = tid; i < K / 8; i += NTHREADS) {
      dst[i] = src[i];
    }
  }

  unsigned char const *W = static_cast<unsigned char const *>(weight);
  unsigned short *O = static_cast<unsigned short *>(output);

  for (int t = blockIdx.x; t < n_tiles; t += gridDim.x) {
    if constexpr (PER_TILE) {
      uint4 const *src = static_cast<uint4 const *>(input);
      uint4 *dst = reinterpret_cast<uint4 *>(s_a);
#pragma unroll
      for (int i = tid; i < K / 8; i += NTHREADS) {
        dst[i] = src[i];
      }
    }
    unsigned char const *wg = W + (size_t)t * WG_BYTES;
    unsigned char const *w_row = wg + (size_t)row * K;
    unsigned char const *s_row = wg + WG_DATA_BYTES + (size_t)row * SCALES_PER_ROW;

    if constexpr (STAGE_SCALES) {
      // The scale half is ROWS_PER_WG * K/32 bytes -- 5 KB for the o_proj
      // tile -- and 16-byte aligned by the packer, so it comes in as dwordx4.
      uint4 const *ssrc =
          reinterpret_cast<uint4 const *>(wg + WG_DATA_BYTES);
      uint4 *sdst = reinterpret_cast<uint4 *>(s_s);
#pragma unroll
      for (int i = tid; i < (ROWS_PER_WG * SCALES_PER_ROW) / 16; i += NTHREADS) {
        sdst[i] = ssrc[i];
      }
    }
    __syncthreads();

    constexpr int UNROLL = UNROLL_IN ? UNROLL_IN
                           : (ITERS % 8 == 0) ? 8
                           : (ITERS % 4 == 0) ? 4
                                              : 1;
    static_assert(ITERS % UNROLL == 0, "row must tile evenly over the unroll");
    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    for (int i0 = 0; i0 < ITERS; i0 += UNROLL) {
      uint4 wv[UNROLL];
      unsigned char sv[UNROLL];
#pragma unroll
      for (int u = 0; u < UNROLL; u++) {
        int const k = ((i0 + u) * LANES_PER_ROW + lane) * VEC;
        wv[u] = *reinterpret_cast<uint4 const *>(w_row + k);
        sv[u] = STAGE_SCALES ? s_s[row * SCALES_PER_ROW + k / SCALE_BLOCK]
                             : s_row[k / SCALE_BLOCK];
      }
#pragma unroll
      for (int u = 0; u < UNROLL; u++) {
        int const k = ((i0 + u) * LANES_PER_ROW + lane) * VEC;
        uint4 const a0 = *reinterpret_cast<uint4 const *>(s_a + k);
        uint4 const a1 = *reinterpret_cast<uint4 const *>(s_a + k + 8);
        float const sc = e8m0_to_f32(sv[u]);
        unsigned const wd[4] = {wv[u].x, wv[u].y, wv[u].z, wv[u].w};
#pragma unroll
        for (int h = 0; h < 4; h++) {
          uint4 const a = (h >> 1) ? a1 : a0;
          unsigned const ax = (h & 1) ? a.z : a.x;
          unsigned const ay = (h & 1) ? a.w : a.y;
          acc[2 * (h & 1)] = kernel::gang_gemv_detail::fma_pair(
              cvt_fp8_pair<false>(wd[h], sc), ax, acc[2 * (h & 1)]);
          acc[2 * (h & 1) + 1] = kernel::gang_gemv_detail::fma_pair(
              cvt_fp8_pair<true>(wd[h], sc), ay, acc[2 * (h & 1) + 1]);
        }
      }
    }

    float sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
#pragma unroll
    for (int off = LANES_PER_ROW >> 1; off > 0; off >>= 1) {
      sum += __shfl_xor(sum, off);
    }
    if (lane == 0) {
      O[(size_t)t * ROWS_PER_WG + row] = f2b(sum);
    }
    __syncthreads();
  }
}

// ───────────────────────────────────────────────────────────────────────────
// Kernel E: LDS activation + lane-contiguous chunk blocking.
//
// The remaining gap to kernel A is the E8M0 fetch. In the shipped mapping a
// lane's UNROLL chunks are LANES_PER_ROW apart, so their scale bytes are
// strided and cost eight separate global_load_ubyte per batch -- eight vmem
// instructions, each touching 8 useful bytes in a wave, sitting in the same
// vmcnt queue as the weight loads that actually matter.
//
// Give a lane UNROLL *consecutive* chunks instead and two things fall out at
// once: the eight scale bytes become four (two chunks share a 32-element scale
// block) and land contiguously, so they are one dword; and the lane's eight
// weight dwordx4 become 128 contiguous bytes, 2 KB contiguous per row across
// the wave's 16 lanes.
//
// The packed weight is untouched -- this changes the order a row is walked in,
// not how it is stored -- so it is a drop-in for the shipped packer.
// ───────────────────────────────────────────────────────────────────────────
template <int K, int ROWS_PER_WG, int UNROLL>
__global__ __launch_bounds__(256) void gemv_lds_contig_driver(void const *input,
                                                              void const *weight,
                                                              void *output,
                                                              int n_tiles,
                                                              int o_stride) {
  using kernel::gang_gemv_detail::f2b;
  using kernel::gang_gemv_mxfp8_detail::cvt_fp8_pair;
  using kernel::gang_gemv_mxfp8_detail::e8m0_to_f32;

  constexpr int NTHREADS = 256;
  constexpr int LANES_PER_ROW = NTHREADS / ROWS_PER_WG;
  constexpr int VEC = 16;
  constexpr int SCALE_BLOCK = 32;
  constexpr int CHUNKS = K / VEC;                     // per row
  constexpr int CHUNKS_PER_BATCH = LANES_PER_ROW * UNROLL;
  constexpr int BATCHES = CHUNKS / CHUNKS_PER_BATCH;
  constexpr int WG_DATA_BYTES = ROWS_PER_WG * K;
  constexpr int WG_BYTES = WG_DATA_BYTES + ROWS_PER_WG * (K / SCALE_BLOCK);
  constexpr int SCALES_PER_ROW = K / SCALE_BLOCK;
  static_assert(CHUNKS % CHUNKS_PER_BATCH == 0, "row must tile evenly");
  static_assert(UNROLL % 2 == 0, "a lane's run must cover whole scale blocks");

  __shared__ unsigned short s_a[K];

  int const tid = threadIdx.x;
  int const row = tid / LANES_PER_ROW;
  int const lane = tid % LANES_PER_ROW;

  {
    uint4 const *src = static_cast<uint4 const *>(input);
    uint4 *dst = reinterpret_cast<uint4 *>(s_a);
#pragma unroll
    for (int i = tid; i < K / 8; i += NTHREADS) {
      dst[i] = src[i];
    }
  }
  __syncthreads();

  unsigned char const *W = static_cast<unsigned char const *>(weight);
  unsigned short *O = static_cast<unsigned short *>(output);

  for (int t = blockIdx.x; t < n_tiles; t += gridDim.x) {
    unsigned char const *wg = W + (size_t)t * WG_BYTES;
    unsigned char const *w_row = wg + (size_t)row * K;
    unsigned char const *s_row =
        wg + WG_DATA_BYTES + (size_t)row * SCALES_PER_ROW;

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int b = 0; b < BATCHES; b++) {
      int const base = b * CHUNKS_PER_BATCH + lane * UNROLL;
      // UNROLL consecutive chunks = UNROLL/2 scale bytes, contiguous.
      uint4 wv[UNROLL];
      unsigned char sbuf[UNROLL / 2];
#pragma unroll
      for (int u = 0; u < UNROLL; u++) {
        __builtin_memcpy(&wv[u], w_row + (size_t)(base + u) * VEC, 16);
      }
      __builtin_memcpy(sbuf, s_row + base / 2, UNROLL / 2);

#pragma unroll
      for (int u = 0; u < UNROLL; u++) {
        int const k = (base + u) * VEC;
        uint4 const a0 = *reinterpret_cast<uint4 const *>(s_a + k);
        uint4 const a1 = *reinterpret_cast<uint4 const *>(s_a + k + 8);
        float const sc = e8m0_to_f32(sbuf[u >> 1]);
        unsigned const wd[4] = {wv[u].x, wv[u].y, wv[u].z, wv[u].w};
#pragma unroll
        for (int h = 0; h < 4; h++) {
          uint4 const a = (h >> 1) ? a1 : a0;
          unsigned const ax = (h & 1) ? a.z : a.x;
          unsigned const ay = (h & 1) ? a.w : a.y;
          acc[2 * (h & 1)] = kernel::gang_gemv_detail::fma_pair(
              cvt_fp8_pair<false>(wd[h], sc), ax, acc[2 * (h & 1)]);
          acc[2 * (h & 1) + 1] = kernel::gang_gemv_detail::fma_pair(
              cvt_fp8_pair<true>(wd[h], sc), ay, acc[2 * (h & 1) + 1]);
        }
      }
    }

    float sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
#pragma unroll
    for (int off = LANES_PER_ROW >> 1; off > 0; off >>= 1) {
      sum += __shfl_xor(sum, off);
    }
    if (lane == 0) {
      O[(size_t)t * ROWS_PER_WG + row] = f2b(sum);
    }
  }
}

static double time_ms(void (*launch)(void *), void *ctx, int iters) {
  hipEvent_t a, b;
  HIP_CHECK(hipEventCreate(&a));
  HIP_CHECK(hipEventCreate(&b));
  launch(ctx); // warm up
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipEventRecord(a));
  for (int i = 0; i < iters; i++) {
    launch(ctx);
  }
  HIP_CHECK(hipEventRecord(b));
  HIP_CHECK(hipEventSynchronize(b));
  float ms = 0.f;
  HIP_CHECK(hipEventElapsedTime(&ms, a, b));
  HIP_CHECK(hipEventDestroy(a));
  HIP_CHECK(hipEventDestroy(b));
  return ms / iters;
}

// ── kernel A driver ────────────────────────────────────────────────────────
struct StreamCtx {
  uint4 const *src;
  size_t n_vec;
  unsigned *sink;
  int blocks;
  int unroll;
};

template <int UNROLL>
static void launch_stream(void *p) {
  StreamCtx *c = static_cast<StreamCtx *>(p);
  stream_read_kernel<UNROLL><<<c->blocks, 256>>>(c->src, c->n_vec, c->sink);
}

static void run_stream(StreamCtx *c, size_t bytes) {
  void (*fn)(void *) = nullptr;
  switch (c->unroll) {
    case 1: fn = launch_stream<1>; break;
    case 2: fn = launch_stream<2>; break;
    case 4: fn = launch_stream<4>; break;
    case 8: fn = launch_stream<8>; break;
    case 16: fn = launch_stream<16>; break;
    default: return;
  }
  double ms = time_ms(fn, c, 5);
  printf("  blocks=%4d  unroll=%2d  %7.3f ms  %7.0f GB/s  %6.2f GB/s/WG\n",
         c->blocks,
         c->unroll,
         ms,
         bytes / (ms * 1e6),
         bytes / (ms * 1e6) / c->blocks);
}

// ── kernel B driver ────────────────────────────────────────────────────────
struct GemvCtx {
  void const *input;
  void const *weight;
  void *output;
  int n_tiles;
  int o_stride;
  int blocks;
};

// ── F. the megakernel's real shape: workers alongside barrier spinners ──────
//
// In gang_oproj_router_fused only 16 of the 30 workers on an XCD have an
// o_proj tile; the other 14 fall straight through to a poll on the routing
// barrier and spin there for the whole phase. So the 128 workgroups doing the
// GEMV share the machine with 112 that are issuing dependent nontemporal loads
// as fast as s_sleep lets them. This driver reproduces that: WORK_BLOCKS do
// the GEMV, the rest poll a counter the workers bump on the way out.
//
// `target` is monotonic across launches so nothing has to be reset between
// timed iterations.
template <int K, int ROWS_PER_WG, int WORK_BLOCKS>
__global__ __launch_bounds__(256) void gemv_spin_driver(void const *input,
                                                        void const *weight,
                                                        void *output,
                                                        int n_tiles,
                                                        int o_stride,
                                                        int *done,
                                                        int target) {
  if (blockIdx.x < WORK_BLOCKS) {
    for (int t = blockIdx.x; t < n_tiles; t += WORK_BLOCKS) {
      kernel::gang_gemv_mxfp8_kernel<1, K, ROWS_PER_WG, false, false>(
          input, weight, nullptr, output, 1, ROWS_PER_WG, o_stride, 1, n_tiles,
          0, t);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      __hip_atomic_fetch_add(done, 1, __ATOMIC_RELEASE,
                             __HIP_MEMORY_SCOPE_AGENT);
    }
  } else {
    if (threadIdx.x == 0) {
      while (__builtin_nontemporal_load(done) < target) {
        __builtin_amdgcn_s_sleep(1);
      }
    }
    __syncthreads();
  }
}

struct SpinCtx {
  void const *input;
  void const *weight;
  void *output;
  int n_tiles;
  int o_stride;
  int blocks;
  int *done;
  int launches;
};

template <int K, int ROWS_PER_WG, int WORK_BLOCKS>
static void launch_gemv_spin(void *p) {
  SpinCtx *c = static_cast<SpinCtx *>(p);
  c->launches++;
  gemv_spin_driver<K, ROWS_PER_WG, WORK_BLOCKS>
      <<<c->blocks, 256, K * sizeof(unsigned short)>>>(
          c->input, c->weight, c->output, c->n_tiles, c->o_stride, c->done,
          c->launches * WORK_BLOCKS);
}

template <int K, int ROWS_PER_WG, int WORK_BLOCKS>
static void run_gemv_spin(void const *input,
                          void const *weight,
                          void *output,
                          int n_tiles,
                          int blocks,
                          int *done) {
  SpinCtx c{input, weight,  output, n_tiles,
            n_tiles * ROWS_PER_WG, blocks, done, 0};
  size_t const bytes = (size_t)ROWS_PER_WG * (K + K / 32) * n_tiles;
  double ms = time_ms(launch_gemv_spin<K, ROWS_PER_WG, WORK_BLOCKS>, &c, 5);
  printf("  rows/wg=%2d  work=%4d  spin=%4d  tiles=%6d  %7.3f ms  %7.0f GB/s"
         "  %6.2f GB/s/WG\n",
         ROWS_PER_WG,
         WORK_BLOCKS,
         blocks - WORK_BLOCKS,
         n_tiles,
         ms,
         bytes / (ms * 1e6),
         bytes / (ms * 1e6) / WORK_BLOCKS);
}

// ── G. one tile per block per layer, with a global barrier between layers ───
//
// Kernels B..F all run a grid-stride loop: a block chews through ~184 tiles
// back to back, so the activation row it stages into LDS is hot in L1 after
// the first tile and the blocks drift out of phase with each other. The
// megakernel is the opposite shape. gang_oproj_router_fused gives each worker
// exactly ONE o_proj tile per layer (oproj_tiles_per_xcd == 16, tile_idx ==
// xcd_rank), and the layer's cross-XCD barrier re-synchronises all 128 of them
// before the next one. So every layer pays a cold 20 KB LDS fill, all 128
// workgroups demand the same 320 cache lines at the same instant, and there is
// no next tile to overlap the fill against.
//
// This driver reproduces that exactly, including the three nontemporal barrier
// reads the real task issues before its __syncthreads, and reports the same
// quantity the megakernel's subphase timer does: mean ns per worker per layer
// between task entry and the post-GEMV s_waitcnt.
//
// The tile index walks forward with the layer so no weight is ever re-read,
// matching a decode step's zero reuse across the 46 layers.
template <int K, int ROWS_PER_WG, int WORK_BLOCKS>
__global__ __launch_bounds__(256) void gemv_layer_driver(
    void const *input, void const *weight, void *output, int n_layers,
    int o_stride, int *bar, int base, unsigned long long *acc) {
  int const tid = threadIdx.x;
  __shared__ int s_e[3];
  for (int l = 0; l < n_layers; l++) {
    unsigned long long t0 = __builtin_amdgcn_s_memrealtime();
    // The real task's three ld_nt_s32 reads of the barrier slots. They are
    // nontemporal on lines other XCDs write, so they miss to memory, and the
    // __syncthreads below puts that latency on the critical path.
    if (tid == 0) {
      s_e[0] = __builtin_nontemporal_load(&bar[0]);
      s_e[1] = __builtin_nontemporal_load(&bar[16]);
      s_e[2] = __builtin_nontemporal_load(&bar[32]);
    }
    __syncthreads();
    if (blockIdx.x < WORK_BLOCKS) {
      int const t = l * WORK_BLOCKS + blockIdx.x;
      kernel::gang_gemv_mxfp8_kernel<1, K, ROWS_PER_WG, false, false>(
          input, weight, nullptr, output, 1, ROWS_PER_WG, o_stride, 1,
          n_layers * WORK_BLOCKS, 0, t);
      __syncthreads();
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      if (tid == 0) {
        unsigned long long t1 = __builtin_amdgcn_s_memrealtime();
        __hip_atomic_fetch_add(acc, t1 - t0, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
      }
    } else {
      __syncthreads();
    }
    // Global barrier over every block. Monotonic like the megakernel's, so
    // nothing needs resetting between timed launches; `base` carries the
    // counter forward across them.
    __syncthreads();
    if (tid == 0) {
      __hip_atomic_fetch_add(&bar[64], 1, __ATOMIC_RELEASE,
                             __HIP_MEMORY_SCOPE_AGENT);
      int const need = base + (l + 1) * (int)gridDim.x;
      // Nontemporal load, not an acquire load: an acquire at agent scope emits
      // a buffer_inv every iteration, and 112 blocks invalidating L1 in a tight
      // loop is a poll the megakernel does not do -- it spins on ld_nt_s32 and
      // invalidates once, after the flag is seen.
      while (__builtin_nontemporal_load(&bar[64]) < need) {
        __builtin_amdgcn_s_sleep(1);
      }
    }
    __syncthreads();
    // s_e is rewritten next layer; keep the compiler from hoisting the reads
    // out of the loop by making the barrier value depend on nothing it can see.
    if (s_e[0] + s_e[1] + s_e[2] == 0x7fffffff && tid == 0) {
      *acc += 1;
    }
  }
}

// ── H. the same layer shape, but sweeping memory-level parallelism ─────────
//
// G shows the one-tile-per-layer shape costs ~2x, and the reason has to be
// MLP: a tile is 169 KB over 256 threads = 41 dwordx4 per thread, and at
// UNROLL 8 that is five s_waitcnt-separated batches, i.e. five serialized HBM
// round trips with no next tile to overlap them against. In the grid-stride
// drivers the next tile's loads are already in flight when this one waits, so
// the same five batches cost nothing extra.
//
// Two knobs, measured independently:
//   UNROLL   how many weight loads a thread keeps outstanding at once. 40 is
//            the whole row -- one round trip -- at 160 VGPRs for wv alone,
//            which only fits because the persistent kernel runs one 256-thread
//            workgroup per CU, so a wave owns all 512 VGPRs of its SIMD.
//   OVERLAP  software-pipeline the activation fill against the first weight
//            batch. The fill's global loads are issued *first* so the ds_write
//            can retire on a partial vmcnt without waiting for the weights.
template <int K, int ROWS_PER_WG, int WORK_BLOCKS, int UNROLL, bool OVERLAP>
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(1, 1)))
void gemv_layer_deep_driver(void const *input, void const *weight,
                            void *output, int n_layers, int *bar, int base,
                            unsigned long long *acc) {
  using kernel::gang_gemv_detail::f2b;
  using kernel::gang_gemv_mxfp8_detail::cvt_fp8_pair;
  using kernel::gang_gemv_mxfp8_detail::e8m0_to_f32;
  using kernel::gang_gemv_mxfp8_detail::ld_g;
  using kernel::gang_gemv_mxfp8_detail::u32x4_t;

  constexpr int NTHREADS = 256;
  constexpr int LANES_PER_ROW = NTHREADS / ROWS_PER_WG;
  constexpr int VEC = 16;
  constexpr int SCALE_BLOCK = 32;
  constexpr int ITERS = K / (LANES_PER_ROW * VEC);
  constexpr int WG_DATA_BYTES = ROWS_PER_WG * K;
  constexpr int WG_BYTES = WG_DATA_BYTES + ROWS_PER_WG * (K / SCALE_BLOCK);
  constexpr int SCALES_PER_ROW = K / SCALE_BLOCK;
  constexpr int FILL = K / 8 / NTHREADS; // uint4s of activation per thread
  static_assert(ITERS % UNROLL == 0, "row must tile evenly over the unroll");
  static_assert(K / 8 % NTHREADS == 0, "activation must tile evenly");

  extern __shared__ char _fused_smem[];
  unsigned short *s_a = reinterpret_cast<unsigned short *>(_fused_smem);

  int const tid = threadIdx.x;
  int const row = tid / LANES_PER_ROW;
  int const lane = tid % LANES_PER_ROW;
  unsigned char const *W = static_cast<unsigned char const *>(weight);
  unsigned short *O = static_cast<unsigned short *>(output);
  u32x4_t const *fsrc = static_cast<u32x4_t const *>(input);
  u32x4_t *fdst = reinterpret_cast<u32x4_t *>(s_a);
  __shared__ int s_e[3];

  for (int l = 0; l < n_layers; l++) {
    unsigned long long t0 = __builtin_amdgcn_s_memrealtime();
    if (tid == 0) {
      s_e[0] = __builtin_nontemporal_load(&bar[0]);
      s_e[1] = __builtin_nontemporal_load(&bar[16]);
      s_e[2] = __builtin_nontemporal_load(&bar[32]);
    }
    if (blockIdx.x < WORK_BLOCKS) {
      int const t = l * WORK_BLOCKS + blockIdx.x;
      unsigned char const *wg = W + (size_t)t * WG_BYTES;
      unsigned char const *w_row = wg + (size_t)row * K;
      unsigned char const *s_row =
          wg + WG_DATA_BYTES + (size_t)row * SCALES_PER_ROW;
      float acc4[4] = {0.f, 0.f, 0.f, 0.f};
      u32x4_t wv0[OVERLAP ? UNROLL : 1];
      unsigned char sv0[OVERLAP ? UNROLL : 1];

      if constexpr (OVERLAP) {
        u32x4_t fb[FILL];
#pragma unroll
        for (int j = 0; j < FILL; j++) {
          fb[j] = ld_g<u32x4_t>(fsrc + tid + j * NTHREADS);
        }
#pragma unroll
        for (int u = 0; u < UNROLL; u++) {
          int const k = (u * LANES_PER_ROW + lane) * VEC;
          wv0[u] = ld_g<u32x4_t>(w_row + k);
          sv0[u] = ld_g<unsigned char>(s_row + k / SCALE_BLOCK);
        }
#pragma unroll
        for (int j = 0; j < FILL; j++) {
          fdst[tid + j * NTHREADS] = fb[j];
        }
      } else {
#pragma unroll
        for (int j = 0; j < FILL; j++) {
          fdst[tid + j * NTHREADS] = ld_g<u32x4_t>(fsrc + tid + j * NTHREADS);
        }
      }
      __syncthreads();

      for (int i0 = 0; i0 < ITERS; i0 += UNROLL) {
        u32x4_t wv[UNROLL];
        unsigned char sv[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; u++) {
          int const k = ((i0 + u) * LANES_PER_ROW + lane) * VEC;
          if (OVERLAP && i0 == 0) {
            wv[u] = wv0[u];
            sv[u] = sv0[u];
          } else {
            wv[u] = ld_g<u32x4_t>(w_row + k);
            sv[u] = ld_g<unsigned char>(s_row + k / SCALE_BLOCK);
          }
        }
#pragma unroll
        for (int u = 0; u < UNROLL; u++) {
          int const k = ((i0 + u) * LANES_PER_ROW + lane) * VEC;
          u32x4_t const a0 = *reinterpret_cast<u32x4_t const *>(s_a + k);
          u32x4_t const a1 = *reinterpret_cast<u32x4_t const *>(s_a + k + 8);
          float const sc = e8m0_to_f32(sv[u]);
          unsigned const wd[4] = {wv[u].x, wv[u].y, wv[u].z, wv[u].w};
#pragma unroll
          for (int h = 0; h < 4; h++) {
            u32x4_t const a = (h >> 1) ? a1 : a0;
            unsigned const ax = (h & 1) ? a.z : a.x;
            unsigned const ay = (h & 1) ? a.w : a.y;
            acc4[2 * (h & 1)] = kernel::gang_gemv_detail::fma_pair(
                cvt_fp8_pair<false>(wd[h], sc), ax, acc4[2 * (h & 1)]);
            acc4[2 * (h & 1) + 1] = kernel::gang_gemv_detail::fma_pair(
                cvt_fp8_pair<true>(wd[h], sc), ay, acc4[2 * (h & 1) + 1]);
          }
        }
      }

      float sum = (acc4[0] + acc4[1]) + (acc4[2] + acc4[3]);
#pragma unroll
      for (int off = LANES_PER_ROW >> 1; off > 0; off >>= 1) {
        sum += __shfl_xor(sum, off);
      }
      if (lane == 0) {
        O[(size_t)t * ROWS_PER_WG + row] = f2b(sum);
      }
      __syncthreads();
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      if (tid == 0) {
        unsigned long long t1 = __builtin_amdgcn_s_memrealtime();
        __hip_atomic_fetch_add(acc, t1 - t0, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
      }
    } else {
      __syncthreads();
    }

    __syncthreads();
    if (tid == 0) {
      __hip_atomic_fetch_add(&bar[64], 1, __ATOMIC_RELEASE,
                             __HIP_MEMORY_SCOPE_AGENT);
      int const need = base + (l + 1) * (int)gridDim.x;
      while (__builtin_nontemporal_load(&bar[64]) < need) {
        __builtin_amdgcn_s_sleep(1);
      }
    }
    __syncthreads();
    if (s_e[0] + s_e[1] + s_e[2] == 0x7fffffff && tid == 0) {
      *acc += 1;
    }
  }
}

struct LayerCtx {
  void const *input;
  void const *weight;
  void *output;
  int n_layers;
  int o_stride;
  int blocks;
  int *bar;
  unsigned long long *acc;
  int launches;
};

template <int K, int ROWS_PER_WG, int WORK_BLOCKS>
static void launch_gemv_layer(void *p) {
  LayerCtx *c = static_cast<LayerCtx *>(p);
  int const base = c->launches * c->n_layers * c->blocks;
  c->launches++;
  gemv_layer_driver<K, ROWS_PER_WG, WORK_BLOCKS>
      <<<c->blocks, 256, K * sizeof(unsigned short)>>>(
          c->input, c->weight, c->output, c->n_layers, c->o_stride, c->bar,
          base, c->acc);
}

template <int K, int ROWS_PER_WG, int WORK_BLOCKS>
static void run_gemv_layer(void const *input, void const *weight, void *output,
                           int n_layers, int blocks, int *bar,
                           unsigned long long *acc) {
  HIP_CHECK(hipMemset(acc, 0, sizeof(unsigned long long)));
  LayerCtx c{input, weight, output, n_layers,
             n_layers * WORK_BLOCKS * ROWS_PER_WG, blocks, bar, acc, 0};
  size_t const bytes =
      (size_t)ROWS_PER_WG * (K + K / 32) * n_layers * WORK_BLOCKS;
  int const iters = 5;
  double ms = time_ms(launch_gemv_layer<K, ROWS_PER_WG, WORK_BLOCKS>, &c, iters);
  unsigned long long h_acc = 0;
  HIP_CHECK(hipMemcpy(&h_acc, acc, sizeof(h_acc), hipMemcpyDeviceToHost));
  // s_memrealtime ticks at 100 MHz, and the accumulator spans every timed
  // launch plus the warm-up ones time_ms does, so normalise by c.launches.
  double const ns_per_worker_layer =
      (double)h_acc * 10.0 / ((double)c.launches * n_layers * WORK_BLOCKS);
  printf("  rows/wg=%2d  work=%4d  spin=%4d  layers=%3d  %7.3f ms  "
         "%7.0f GB/s  %6.2f GB/s/WG   phase0=%8.0f ns/worker/layer\n",
         ROWS_PER_WG, WORK_BLOCKS, blocks - WORK_BLOCKS, n_layers, ms,
         bytes / (ms * 1e6), bytes / (ms * 1e6) / WORK_BLOCKS,
         ns_per_worker_layer);
}

template <int K, int ROWS_PER_WG, int WORK_BLOCKS, int UNROLL, bool OVERLAP>
static void launch_gemv_deep(void *p) {
  LayerCtx *c = static_cast<LayerCtx *>(p);
  int const base = c->launches * c->n_layers * c->blocks;
  c->launches++;
  gemv_layer_deep_driver<K, ROWS_PER_WG, WORK_BLOCKS, UNROLL, OVERLAP>
      <<<c->blocks, 256, K * sizeof(unsigned short)>>>(
          c->input, c->weight, c->output, c->n_layers, c->bar, base, c->acc);
}

template <int K, int ROWS_PER_WG, int WORK_BLOCKS, int UNROLL, bool OVERLAP>
static void run_gemv_deep(void const *input, void const *weight, void *output,
                          int n_layers, int blocks, int *bar,
                          unsigned long long *acc) {
  HIP_CHECK(hipMemset(acc, 0, sizeof(unsigned long long)));
  LayerCtx c{input, weight, output, n_layers,
             n_layers * WORK_BLOCKS * ROWS_PER_WG, blocks, bar, acc, 0};
  size_t const bytes =
      (size_t)ROWS_PER_WG * (K + K / 32) * n_layers * WORK_BLOCKS;
  double ms =
      time_ms(launch_gemv_deep<K, ROWS_PER_WG, WORK_BLOCKS, UNROLL, OVERLAP>,
              &c, 5);
  unsigned long long h_acc = 0;
  HIP_CHECK(hipMemcpy(&h_acc, acc, sizeof(h_acc), hipMemcpyDeviceToHost));
  double const ns_per_worker_layer =
      (double)h_acc * 10.0 / ((double)c.launches * n_layers * WORK_BLOCKS);
  printf("  work=%4d  spin=%4d  unroll=%2d  overlap=%d  %7.3f ms  %7.0f GB/s"
         "  %6.2f GB/s/WG   phase0=%8.0f ns/worker/layer\n",
         WORK_BLOCKS, blocks - WORK_BLOCKS, UNROLL, (int)OVERLAP, ms,
         bytes / (ms * 1e6), bytes / (ms * 1e6) / WORK_BLOCKS,
         ns_per_worker_layer);
}

template <int K, int ROWS_PER_WG>
static void launch_gemv(void *p) {
  GemvCtx *c = static_cast<GemvCtx *>(p);
  // The kernel now stages the activation row through dynamic LDS, so the
  // launch has to hand it the space the megakernel's workers always have.
  gemv_driver<K, ROWS_PER_WG><<<c->blocks, 256, K * sizeof(unsigned short)>>>(
      c->input, c->weight, c->output, c->n_tiles, c->o_stride);
}

template <int K, int ROWS_PER_WG>
static void run_gemv(void const *input,
                     void const *weight,
                     void *output,
                     int n_tiles,
                     int blocks) {
  GemvCtx c{input, weight, output, n_tiles, n_tiles * ROWS_PER_WG, blocks};
  size_t const wg_bytes = (size_t)ROWS_PER_WG * (K + K / 32);
  size_t const bytes = wg_bytes * n_tiles;
  double ms = time_ms(launch_gemv<K, ROWS_PER_WG>, &c, 5);
  printf("  rows/wg=%2d  tiles=%6d  blocks=%4d  %7.3f ms  %7.0f GB/s  "
         "%6.2f GB/s/WG\n",
         ROWS_PER_WG,
         n_tiles,
         blocks,
         ms,
         bytes / (ms * 1e6),
         bytes / (ms * 1e6) / blocks);
}

template <int K, int ROWS_PER_WG, bool STAGE_SCALES, int UNROLL_IN,
          bool PER_TILE>
static void launch_gemv_lds(void *p) {
  GemvCtx *c = static_cast<GemvCtx *>(p);
  gemv_lds_driver<K, ROWS_PER_WG, STAGE_SCALES, UNROLL_IN, PER_TILE>
      <<<c->blocks, 256>>>(
          c->input, c->weight, c->output, c->n_tiles, c->o_stride);
}

template <int K, int ROWS_PER_WG, bool STAGE_SCALES, int UNROLL_IN = 0,
          bool PER_TILE = false>
static void run_gemv_lds(void const *input,
                         void const *weight,
                         void *output,
                         int n_tiles,
                         int blocks) {
  GemvCtx c{input, weight, output, n_tiles, n_tiles * ROWS_PER_WG, blocks};
  size_t const bytes = (size_t)ROWS_PER_WG * (K + K / 32) * n_tiles;
  double ms = time_ms(
      launch_gemv_lds<K, ROWS_PER_WG, STAGE_SCALES, UNROLL_IN, PER_TILE>, &c, 5);
  printf("  rows/wg=%2d  fill=%s  unroll=%2d  tiles=%6d  blocks=%4d  %7.3f ms"
         "  %7.0f GB/s  %6.2f GB/s/WG\n",
         ROWS_PER_WG,
         PER_TILE ? "per-tile" : "once    ",
         UNROLL_IN,
         n_tiles,
         blocks,
         ms,
         bytes / (ms * 1e6),
         bytes / (ms * 1e6) / blocks);
}

template <int K, int ROWS_PER_WG, int UNROLL>
static void launch_gemv_contig(void *p) {
  GemvCtx *c = static_cast<GemvCtx *>(p);
  gemv_lds_contig_driver<K, ROWS_PER_WG, UNROLL><<<c->blocks, 256>>>(
      c->input, c->weight, c->output, c->n_tiles, c->o_stride);
}

template <int K, int ROWS_PER_WG, int UNROLL>
static void run_gemv_contig(void const *input,
                            void const *weight,
                            void *output,
                            int n_tiles,
                            int blocks) {
  GemvCtx c{input, weight, output, n_tiles, n_tiles * ROWS_PER_WG, blocks};
  size_t const bytes = (size_t)ROWS_PER_WG * (K + K / 32) * n_tiles;
  double ms = time_ms(launch_gemv_contig<K, ROWS_PER_WG, UNROLL>, &c, 5);
  printf("  rows/wg=%2d  unroll=%2d  tiles=%6d  blocks=%4d  %7.3f ms  %7.0f "
         "GB/s  %6.2f GB/s/WG\n",
         ROWS_PER_WG, UNROLL, n_tiles, blocks, ms,
         bytes / (ms * 1e6), bytes / (ms * 1e6) / blocks);
}

int main(int argc, char **argv) {
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  printf("device: %s, %d CUs\n\n", props.gcnArchName, props.multiProcessorCount);

  // 4 GiB, comfortably past the 256 MB MALL, so every read is HBM.
  size_t const BUF = 4ull << 30;
  void *buf = nullptr;
  HIP_CHECK(hipMalloc(&buf, BUF));
  HIP_CHECK(hipMemset(buf, 0x3c, BUF));
  unsigned *sink = nullptr;
  HIP_CHECK(hipMalloc(&sink, sizeof(unsigned)));

  printf("A. pure streaming read, %zu MB\n", BUF >> 20);
  int const blocks[] = {64, 128, 192, 240, 256, 480, 512, 1024, 2048};
  int const unrolls[] = {1, 2, 4, 8, 16};
  for (int ui = 0; ui < 5; ui++) {
    for (int bi = 0; bi < 9; bi++) {
      StreamCtx c{static_cast<uint4 const *>(buf),
                  BUF / sizeof(uint4),
                  sink,
                  blocks[bi],
                  unrolls[ui]};
      run_stream(&c, BUF);
    }
    printf("\n");
  }

  // B. the real GEMV at the absorbed-o_proj reduction size. n_tiles is sized
  // so the weight sweep is ~3.9 GB regardless of the row tile, i.e. the same
  // bytes move in every row.
  constexpr int K = 10240;
  void *input = nullptr;
  HIP_CHECK(hipMalloc(&input, (size_t)K * 2));
  HIP_CHECK(hipMemset(input, 0x3c, (size_t)K * 2));

  printf("B. gang_gemv_mxfp8_kernel, K=%d, weights streamed from HBM\n", K);
  struct {
    int rows;
    int tiles;
  } cfg[] = {{8, 47000}, {16, 23500}, {32, 11750}};
  int const gemv_blocks[] = {64, 128, 192, 240, 256, 480, 1024};
  for (int ci = 0; ci < 3; ci++) {
    void *out = nullptr;
    HIP_CHECK(hipMalloc(&out, (size_t)cfg[ci].tiles * cfg[ci].rows * 2));
    for (int bi = 0; bi < 7; bi++) {
      switch (cfg[ci].rows) {
        case 8:
          run_gemv<K, 8>(input, buf, out, cfg[ci].tiles, gemv_blocks[bi]);
          break;
        case 16:
          run_gemv<K, 16>(input, buf, out, cfg[ci].tiles, gemv_blocks[bi]);
          break;
        case 32:
          run_gemv<K, 32>(input, buf, out, cfg[ci].tiles, gemv_blocks[bi]);
          break;
      }
    }
    HIP_CHECK(hipFree(out));
    printf("\n");
  }

  printf("C. LDS activation: once-per-block (optimistic) vs per-tile (real)\n");
  {
    void *out = nullptr;
    HIP_CHECK(hipMalloc(&out, (size_t)23500 * 16 * 2));
    for (int bi = 0; bi < 5; bi++) {
      run_gemv_lds<K, 16, false, 8, false>(input, buf, out, 23500,
                                           gemv_blocks[bi]);
      run_gemv_lds<K, 16, false, 8, true>(input, buf, out, 23500,
                                          gemv_blocks[bi]);
      run_gemv_lds<K, 16, false, 20, true>(input, buf, out, 23500,
                                           gemv_blocks[bi]);
      run_gemv_lds<K, 16, false, 40, true>(input, buf, out, 23500,
                                           gemv_blocks[bi]);
    }
    HIP_CHECK(hipFree(out));
    printf("\n");
  }

  printf("F. GEMV on WORK blocks with the rest spinning on a barrier\n");
  {
    void *out = nullptr;
    HIP_CHECK(hipMalloc(&out, (size_t)23500 * 16 * 2));
    int *done = nullptr;
    HIP_CHECK(hipMalloc(&done, sizeof(int)));
    HIP_CHECK(hipMemset(done, 0, sizeof(int)));
    run_gemv_spin<K, 16, 128>(input, buf, out, 23500, 128, done);
    run_gemv_spin<K, 16, 128>(input, buf, out, 23500, 240, done);
    run_gemv_spin<K, 16, 128>(input, buf, out, 23500, 248, done);
    run_gemv_spin<K, 16, 240>(input, buf, out, 23500, 240, done);
    HIP_CHECK(hipFree(done));
    HIP_CHECK(hipFree(out));
    printf("\n");
  }

  printf("G. one tile per block per layer, global barrier between layers\n");
  {
    int const NL = 46;
    void *out = nullptr;
    HIP_CHECK(hipMalloc(&out, (size_t)NL * 240 * 16 * 2));
    int *bar = nullptr;
    HIP_CHECK(hipMalloc(&bar, 128 * sizeof(int)));
    HIP_CHECK(hipMemset(bar, 0, 128 * sizeof(int)));
    unsigned long long *acc = nullptr;
    HIP_CHECK(hipMalloc(&acc, sizeof(unsigned long long)));
    run_gemv_layer<K, 16, 128>(input, buf, out, NL, 128, bar, acc);
    run_gemv_layer<K, 16, 128>(input, buf, out, NL, 240, bar, acc);
    run_gemv_layer<K, 16, 240>(input, buf, out, NL, 240, bar, acc);
    printf("H. same shape, sweeping unroll depth and fill/weight overlap\n");
    run_gemv_deep<K, 16, 128, 8, false>(input, buf, out, NL, 240, bar, acc);
    run_gemv_deep<K, 16, 128, 8, true>(input, buf, out, NL, 240, bar, acc);
    run_gemv_deep<K, 16, 128, 10, false>(input, buf, out, NL, 240, bar, acc);
    run_gemv_deep<K, 16, 128, 10, true>(input, buf, out, NL, 240, bar, acc);
    run_gemv_deep<K, 16, 128, 20, false>(input, buf, out, NL, 240, bar, acc);
    run_gemv_deep<K, 16, 128, 20, true>(input, buf, out, NL, 240, bar, acc);
    run_gemv_deep<K, 16, 128, 40, false>(input, buf, out, NL, 240, bar, acc);
    run_gemv_deep<K, 16, 128, 40, true>(input, buf, out, NL, 240, bar, acc);
    HIP_CHECK(hipFree(acc));
    HIP_CHECK(hipFree(bar));
    HIP_CHECK(hipFree(out));
    printf("\n");
  }

  printf("E. LDS activation + lane-contiguous chunks + packed scale dword\n");
  for (int ci = 0; ci < 3; ci++) {
    void *out = nullptr;
    HIP_CHECK(hipMalloc(&out, (size_t)cfg[ci].tiles * cfg[ci].rows * 2));
    for (int bi = 0; bi < 7; bi++) {
      switch (cfg[ci].rows) {
        case 8:
          run_gemv_contig<K, 8, 4>(input, buf, out, cfg[ci].tiles, gemv_blocks[bi]);
          run_gemv_contig<K, 8, 10>(input, buf, out, cfg[ci].tiles, gemv_blocks[bi]);
          break;
        case 16:
          run_gemv_contig<K, 16, 4>(input, buf, out, cfg[ci].tiles, gemv_blocks[bi]);
          run_gemv_contig<K, 16, 8>(input, buf, out, cfg[ci].tiles, gemv_blocks[bi]);
          run_gemv_contig<K, 16, 10>(input, buf, out, cfg[ci].tiles, gemv_blocks[bi]);
          break;
        case 32:
          run_gemv_contig<K, 32, 4>(input, buf, out, cfg[ci].tiles, gemv_blocks[bi]);
          run_gemv_contig<K, 32, 8>(input, buf, out, cfg[ci].tiles, gemv_blocks[bi]);
          break;
      }
    }
    HIP_CHECK(hipFree(out));
    printf("\n");
  }

  HIP_CHECK(hipFree(input));
  HIP_CHECK(hipFree(sink));
  HIP_CHECK(hipFree(buf));
  return 0;
}
