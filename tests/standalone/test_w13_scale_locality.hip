// Where does a GLM W13 MoE tile's time actually go?
//
// THE GAP. A W13 tile is 204 KB (192 KB MXFP4 weight + 12 KB E8M0 scales) and
// measures 13.9 us of body. Accounting against the CORRECTED per-CU roof from
// test_narrow_grid_bandwidth.hip -- 41.6 GB/s for a CU in a 64-wide phase, not
// the 20.2 GB/s "share" the old roofline used -- gives:
//
//   weight bytes   192 KB / 41.6 GB/s        =  4.9 us
//   MFMA           48 iters x ~32 cycles     =  0.73 us
//   token quant    measured, subphase slot 1 =  0.86 us
//                                             --------
//                                               ~6.5 us of 13.9
//
// Something costs ~2x the bytes. The OPW=16 arm agrees independently: cutting
// weight bytes 4x only moved the tile 13.9 -> 12.2 us, so most of the tile
// does not scale with weight bytes at all.
//
// THE HYPOTHESIS: THE E8M0 SCALE LOAD'S CACHE-LINE UTILIZATION.
//
//   int sa0 = (int)wg_scales[row_scale_base + kt/32 + g];
//
// with row_scale_base = w_row * NUM_BLOCKS_32 = w_row * 192. Across a wave,
// col (= lane & 15) picks the weight row and g (= lane >> 4) picks the K
// group, so one instruction touches 16 rows x 4 consecutive bytes, strided
// 192 B. That is 16 cache lines requested to deliver 64 useful bytes -- 3%
// line utilization.
//
// The whole tile's scale set is only 64 rows x 192 B = 12 KB (128 lines,
// 16 KB), which fits the 32 KB vector L1, so in a quiet cache the 48
// iterations would all hit. But the same loop streams 192 KB of weight through
// that same L1, which evicts the scale lines between uses. Worst case the
// scale loads re-fetch 48 iters x 4 waves x 16 lines x 128 B = 393 KB, twice
// the tile's weight traffic, to deliver 12 KB.
//
// THE FIX IF TRUE is cheap and correct-output: stage the workgroup's 12 KB of
// scales into LDS once at tile start with fully coalesced dwordx4 loads, then
// read them with ds_read_u8 in the k-loop. The kernel uses 7296 B of a 160 KB
// LDS budget, so the space is free.
//
// THE ARMS. Same weight stream, same MFMA, same depth-4 pipeline, only the
// scale source changes:
//
//   global : exactly what ships today.
//   lds    : the proposed fix -- one coalesced 12 KB stage, then ds_read_u8.
//   none   : scales replaced by a compile-time constant. Not shippable; it is
//            the floor, and it bounds what `lds` could ever recover.
//
// A null between `global` and `none` kills the hypothesis outright and says
// the residual is somewhere else. `lds` landing near `none` says build it.
//
// Run: ./test_w13_scale_locality
#include <hip/hip_runtime.h>
#include <cstdio>

#define HIP_CHECK(x)                                                           \
  do {                                                                         \
    hipError_t _e = (x);                                                       \
    if (_e != hipSuccess) {                                                    \
      printf("HIP error %s at line %d\n", hipGetErrorString(_e), __LINE__);     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

typedef int i32x4_t __attribute__((ext_vector_type(4)));
typedef int i32x8_t __attribute__((ext_vector_type(8)));
typedef float f32x4_t __attribute__((ext_vector_type(4)));

// ── GLM-5 W13 geometry, verbatim ───────────────────────────────────────────
constexpr int REDUCTION_SIZE = 6144; // hidden
constexpr int OUTPUT_PER_WG = 64;    // GLM_MOE_W13_OPW default
constexpr int NUM_WAVES = 4;
constexpr int K_PER_MFMA = 128;
constexpr int MFMA_ITERS = REDUCTION_SIZE / K_PER_MFMA;      // 48
constexpr int NUM_BLOCKS_32 = REDUCTION_SIZE / 32;           // 192
constexpr int W_ROW_BYTES = REDUCTION_SIZE / 2;              // 3072, FP4
constexpr int WG_DATA_BYTES = OUTPUT_PER_WG * W_ROW_BYTES;   // 192 KB
constexpr int WG_SCALE_BYTES = OUTPUT_PER_WG * NUM_BLOCKS_32; // 12 KB
constexpr int WG_BYTES = WG_DATA_BYTES + WG_SCALE_BYTES;     // 204 KB
constexpr int FP8_TOK_DATA = REDUCTION_SIZE;

__device__ __forceinline__ i32x8_t load_fp4_a(uint8_t const *row, int kt,
                                              int g) {
  i32x4_t lo = *(i32x4_t const *)(row + kt / 2 + g * 16);
  i32x8_t r = {};
  r[0] = lo[0];
  r[1] = lo[1];
  r[2] = lo[2];
  r[3] = lo[3];
  return r;
}

__device__ __forceinline__ i32x8_t load_fp8_b(uint8_t const *data, int kt,
                                              int g) {
  i32x4_t lo = *(i32x4_t const *)(data + kt + g * 16);
  i32x4_t hi = *(i32x4_t const *)(data + kt + g * 16 + 64);
  i32x8_t r;
  r[0] = lo[0];
  r[1] = lo[1];
  r[2] = lo[2];
  r[3] = lo[3];
  r[4] = hi[0];
  r[5] = hi[1];
  r[6] = hi[2];
  r[7] = hi[3];
  return r;
}

__device__ __forceinline__ f32x4_t mfma_f4xf8(i32x8_t a, i32x8_t b, f32x4_t c,
                                              int sa, int sb) {
  return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c, 4, 0, 0, sa,
                                                          0, sb);
}

// SCALE_SRC: 0 = global (ships today), 1 = LDS-staged, 2 = constant (floor),
//            3 = stage to LDS but STILL READ GLOBAL, 4 = bare __syncthreads
//
// ── ARMS 3 AND 4 EXIST BECAUSE ARM 1 WAS CONFOUNDED. ──
// First pass measured global 16.29, LDS 11.89, none 15.96 us/tile. Deleting
// the scale load outright (`none`) buys 0.33 us, so the scattered E8M0 load is
// FREE -- fully hidden under the weight stream, and the hypothesis this file
// was written to test is refuted. But arm 1 beat even `none` by 4.1 us, which
// no change of scale source can explain. What arm 1 also introduced was two
// __syncthreads() per tile plus a coalesced 12 KB burst.
//
// The suspected mechanism is wave desynchronization. The four waves stream
// four different 48 KB row-blocks of the tile, and nothing in the shipping
// loop keeps them together: in a grid-stride loop wave 0 can run ahead into
// tile t+1 while wave 3 is still on tile t, so the block has eight streams
// spanning ~400 KB in flight instead of four spanning 204 KB. A per-tile
// barrier would pin them to one tile.
//
//   arm 3 : same barriers AND same coalesced burst as arm 1, k-loop still
//           reads scales from global. Isolates "LDS as a scale source".
//   arm 4 : barrier only, no burst, scales from global. Isolates the barrier
//           from the burst.
//
// Second pass: 0=16.26, 4=16.29, 2=15.94, 3=13.92, 1=11.90. The barrier is a
// null and the scale load is nearly free, yet the coalesced burst is worth
// -2.34 and reading LDS on top of it another -2.02. A burst that merely
// supplies scales cannot beat an arm that needs no scales at all (arm 2), so
// the burst is doing something to the memory system in its own right.
//
//   arm 5 : coalesced burst of the SAME 12 KB shape from an unrelated scratch
//           buffer, scales still read from global. If arm 5 lands at arm 3,
//           the burst's LOCATION is irrelevant and this is generic memory
//           pacing -- which would apply to every tile kernel in the layer,
//           not just to scales. If arm 5 lands at arm 0, the burst matters
//           only because it prefetches the scale lines.
//
// Third pass: arm 5 = 12.87, BETTER than arm 3's 13.96. An unrelated source
// beats the scales themselves, so the burst's location is irrelevant and the
// scale hypothesis is dead in every form. What every arm shares is the number
// that actually matters: 12.1 GB/s/CU of weight against the 41.6 GB/s/CU a CU
// can pull at this grid width. THE TILE IS AT 29% OF ITS REAL ROOF.
//
// The prime suspect is now the weight access SHAPE. Per instruction a wave
// reads 16 rows x 64 contiguous bytes (g = 0..3 x 16 B), strided W_ROW_BYTES =
// 3072 apart -- sixteen half-cache-line requests, where the streaming harness
// that reached 41.6 GB/s/CU issued one fully coalesced 4 KB burst.
//
//   arm 6 : identical in every respect except that the 64 lanes of a wave
//           read 1024 CONTIGUOUS bytes per k-group, i.e. what a K-major
//           repack of the weight would give. Same total bytes, same MFMA,
//           same depth-4 pipeline, same scale loads. This prices "repack the
//           MXFP4 weight offline and change nothing else", which is real:
//           the packer already controls row order. Values are meaningless,
//           the impossible sink keeps them live; this is a timing probe.
//
// Fourth pass: arm 6 = 14.48, so the K-major shape is worth -1.80 us (11%) --
// real, and it composes with nothing else in the file. It does NOT explain the
// burst, which is still worth twice as much from an unrelated address. Arms
// 7-12 take the burst apart along its three independent axes, because a
// mechanism that generalizes would apply to W2, o_proj and qkv_a as well and
// is worth far more than one tile's 11%:
//
//   arm 7  : TWO bare __syncthreads per tile, no burst. Arm 4 had only one,
//            so "barrier count" was never actually held fixed against arms
//            1/3/5. If arm 7 lands at arm 5, the burst is irrelevant.
//   arm 8  : barriers + a 4 KB burst (one pass).   } burst SIZE sweep, same
//   arm 9  : barriers + a 48 KB burst (12 passes). } unrelated source
//   arm 10 : barriers + s_sleep only, ~0.1 us.  } is it just a DELAY between
//   arm 12 : barriers + s_sleep only, ~0.4 us.  } the previous tile and this
//   arm 11 : the shippable stack -- LDS-staged scales AND K-major weight.
//            Only meaningful if 1 and 6 compose.
template <int SCALE_SRC>
__global__ void __launch_bounds__(256)
    w13_tile(uint8_t const *__restrict__ W, float *sink, int tiles_total,
             uint8_t const *__restrict__ scratch) {
  extern __shared__ char smem[];
  uint8_t *s_tok_fp8 = (uint8_t *)smem;
  uint8_t *s_tok_scales = s_tok_fp8 + FP8_TOK_DATA;
  // 16-byte aligned, past the token scratch, exactly as the real kernel places
  // its auxiliary LDS.
  uint8_t *s_scales =
      (uint8_t *)(((uintptr_t)(s_tok_scales + NUM_BLOCKS_32) + 15u) &
                  ~(uintptr_t)15);

  int const tid = threadIdx.x;
  int const warp_id = tid >> 6;
  int const lane_id = tid & 63;
  int const col = lane_id & 15;
  int const g = lane_id >> 4;

  // Stand in for the token-quant prologue. Identical in every arm, so it
  // cannot bias the comparison; the real one is a measured 0.86 us.
  for (int i = tid; i < FP8_TOK_DATA / 4; i += 256) {
    ((unsigned *)s_tok_fp8)[i] = 0x3c3c3c3cu;
  }
  for (int i = tid; i < NUM_BLOCKS_32; i += 256) {
    s_tok_scales[i] = 127;
  }
  __syncthreads();

  for (int t = blockIdx.x; t < tiles_total; t += gridDim.x) {
    uint8_t const *wg_data = W + (size_t)t * (size_t)WG_BYTES;
    uint8_t const *wg_scales = wg_data + WG_DATA_BYTES;

    if constexpr (SCALE_SRC == 4) {
      // Barrier only. No burst, no LDS use.
      __syncthreads();
    }
    if constexpr (SCALE_SRC == 5) {
      // Same burst shape and same barriers, unrelated source address.
      __syncthreads();
      for (int i = tid; i < WG_SCALE_BYTES / 16; i += 256) {
        ((i32x4_t *)s_scales)[i] = ((i32x4_t const *)scratch)[i];
      }
      __syncthreads();
    }
    if constexpr (SCALE_SRC == 7) {
      // Barrier COUNT held equal to arms 1/3/5, nothing else.
      __syncthreads();
      __syncthreads();
    }
    if constexpr (SCALE_SRC == 8 || SCALE_SRC == 9) {
      // Burst size sweep: 4 KB (one pass) vs 48 KB (twelve passes).
      constexpr int NB = (SCALE_SRC == 8) ? 4096 : 49152;
      __syncthreads();
      for (int i = tid; i < NB / 16; i += 256) {
        ((i32x4_t *)s_scales)[i] = ((i32x4_t const *)scratch)[i];
      }
      __syncthreads();
    }
    if constexpr (SCALE_SRC == 10 || SCALE_SRC == 12) {
      // Pure delay in the same slot the burst occupies. s_sleep(N) is ~64N
      // clocks; at ~2.4 GHz, 4 -> ~0.11 us and 16 -> ~0.43 us.
      constexpr int NS = (SCALE_SRC == 10) ? 4 : 16;
      __syncthreads();
      __builtin_amdgcn_s_sleep(NS);
      __syncthreads();
    }
    if constexpr (SCALE_SRC == 13) {
      // LDS traffic only, no global read. Isolates the ds_write half.
      __syncthreads();
      for (int i = tid; i < 4096 / 16; i += 256) {
        ((i32x4_t *)s_scales)[i] = i32x4_t{0, 0, 0, 0};
      }
      __syncthreads();
    }
    if constexpr (SCALE_SRC == 14) {
      // Global read only, no LDS write. Isolates the global-read half.
      __syncthreads();
      for (int i = tid; i < 4096 / 16; i += 256) {
        i32x4_t v = ((i32x4_t const *)scratch)[i];
        asm volatile("" ::"v"(v));
      }
      __syncthreads();
    }
    if constexpr (SCALE_SRC == 16) {
      // PER-BLOCK VARIABLE delay. Arms 10/12 slept the SAME amount in every
      // block, so they could not de-correlate the 64 streams; a memory op's
      // latency varies block to block and can. If this reproduces the burst,
      // the mechanism is inter-block skew -- and the real megakernel's W13
      // blocks are already skewed by dispatch, so it would NOT transfer.
      __syncthreads();
      for (int i = 0; i < (int)(blockIdx.x & 7); i++) {
        __builtin_amdgcn_s_sleep(2);
      }
      __syncthreads();
    }
    if constexpr (SCALE_SRC == 11) {
      // The shippable stack: staged scales, and the weight read K-major.
      __syncthreads();
      for (int i = tid; i < WG_SCALE_BYTES / 16; i += 256) {
        ((i32x4_t *)s_scales)[i] = ((i32x4_t const *)wg_scales)[i];
      }
      __syncthreads();
    }
    if constexpr (SCALE_SRC == 1 || SCALE_SRC == 3) {
      // THE PROPOSED FIX. 12 KB in fully coalesced 16-byte loads: 256 threads
      // x 16 B = 4 KB per pass, 3 passes. Every line fetched is fully used,
      // against 3% utilization for the scattered per-iteration form.
      __syncthreads(); // previous tile's readers are done with s_scales
      for (int i = tid; i < WG_SCALE_BYTES / 16; i += 256) {
        ((i32x4_t *)s_scales)[i] = ((i32x4_t const *)wg_scales)[i];
      }
      __syncthreads();
    }

    // K-major repack: wave w owns a contiguous 48 KB slab, and within it each
    // k-group is 64 lanes x 16 B = 1024 contiguous bytes.
    constexpr int WAVE_SLAB = MFMA_ITERS * 1024; // 48 KB, x4 waves = 192 KB
    uint8_t const *w_packed = wg_data + (size_t)warp_id * WAVE_SLAB;
    auto load_packed = [&](int ki_) {
      return *(i32x4_t const *)(w_packed + ki_ * 1024 + lane_id * 16);
    };

    int const wave_tile = warp_id; // TILES_PER_WAVE == 1 at OPW=64
    int const w_row = wave_tile * 16 + col;
    uint8_t const *w_data_row = wg_data + (size_t)w_row * W_ROW_BYTES;
    int const row_scale_base = w_row * NUM_BLOCKS_32;

    // One accessor, so the three arms differ in exactly one expression.
    auto load_a = [&](int ki_) -> i32x8_t {
      if constexpr (SCALE_SRC == 6 || SCALE_SRC == 11) {
        i32x4_t lo = load_packed(ki_);
        i32x8_t r = {};
        r[0] = lo[0]; r[1] = lo[1]; r[2] = lo[2]; r[3] = lo[3];
        return r;
      } else {
        return load_fp4_a(w_data_row, ki_ * K_PER_MFMA, g);
      }
    };

    auto scale_at = [&](int idx) -> int {
      if constexpr (SCALE_SRC == 2) {
        return 127; // the floor arm, no scale traffic at all
      } else if constexpr (SCALE_SRC == 1 || SCALE_SRC == 11) {
        return (int)s_scales[idx];
      } else {
        return (int)wg_scales[idx]; // global, same as arm 0
      }
    };

    f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

    i32x8_t a0 = load_a(0);
    int sa0 = scale_at(row_scale_base + 0 * 4 + g);
    i32x8_t a1 = load_a(1);
    int sa1 = scale_at(row_scale_base + 1 * 4 + g);
    i32x8_t a2 = load_a(2);
    int sa2 = scale_at(row_scale_base + 2 * 4 + g);
    i32x8_t a3 = load_a(3);
    int sa3 = scale_at(row_scale_base + 3 * 4 + g);

// IMPORTANT: #pragma unroll 1 prevents ROCm miscompilation (and keeps the
// three arms structurally identical, so an ISA diff is meaningful).
#pragma unroll 1
    for (int ki = 0; ki < MFMA_ITERS; ki += 4) {
      {
        i32x8_t b = load_fp8_b(s_tok_fp8, ki * K_PER_MFMA, g);
        acc = mfma_f4xf8(a0, b, acc, sa0, (int)s_tok_scales[ki]);
      }
      if (ki + 4 < MFMA_ITERS) {
        int kt4 = (ki + 4) * K_PER_MFMA;
        a0 = load_a(ki + 4);
        sa0 = scale_at(row_scale_base + kt4 / 32 + g);
      }
      {
        i32x8_t b = load_fp8_b(s_tok_fp8, (ki + 1) * K_PER_MFMA, g);
        acc = mfma_f4xf8(a1, b, acc, sa1, (int)s_tok_scales[ki + 1]);
      }
      if (ki + 5 < MFMA_ITERS) {
        int kt5 = (ki + 5) * K_PER_MFMA;
        a1 = load_a(ki + 5);
        sa1 = scale_at(row_scale_base + kt5 / 32 + g);
      }
      {
        i32x8_t b = load_fp8_b(s_tok_fp8, (ki + 2) * K_PER_MFMA, g);
        acc = mfma_f4xf8(a2, b, acc, sa2, (int)s_tok_scales[ki + 2]);
      }
      if (ki + 6 < MFMA_ITERS) {
        int kt6 = (ki + 6) * K_PER_MFMA;
        a2 = load_a(ki + 6);
        sa2 = scale_at(row_scale_base + kt6 / 32 + g);
      }
      {
        i32x8_t b = load_fp8_b(s_tok_fp8, (ki + 3) * K_PER_MFMA, g);
        acc = mfma_f4xf8(a3, b, acc, sa3, (int)s_tok_scales[ki + 3]);
      }
      if (ki + 7 < MFMA_ITERS) {
        int kt7 = (ki + 7) * K_PER_MFMA;
        a3 = load_a(ki + 7);
        sa3 = scale_at(row_scale_base + kt7 / 32 + g);
      }
    }

    // Impossible sink: keeps the whole chain live without a real write-out.
    if (acc[0] == 1.2345678e-31f && acc[1] == 1.2345678e-31f) {
      sink[0] = acc[0] + acc[1] + acc[2] + acc[3];
    }
  }
}

// ── THE DEEP-PREFETCH KERNEL ───────────────────────────────────────────────
//
// Fifth pass looked at the ISA instead of guessing. Every arm above, including
// the ones that ship, compiles the k-loop to:
//
//   .LBB_11:                      ; k-loop header
//     global_load_dwordx4 x4      ; a0..a3 for the NEXT k-group
//     global_load_ubyte   x4      ; its scales
//     s_waitcnt vmcnt(1)          ; <-- waits for 7 of the 8, immediately
//     ...
//     4x v_mfma_scale_f32_16x16x128_f8f6f4
//     s_waitcnt vmcnt(0)
//
// **There is no load/MFMA overlap.** The source prefetches one k-group ahead,
// which puts only 3 MFMAs -- about 96 cycles -- between a load and its use,
// against an HBM latency near 800. So the tile runs at roughly one group in
// flight, which is exactly what 12.1 GB/s/CU against the 41.6 GB/s/CU a CU
// pulls at this grid width ([[glm-per-cu-roofline-denominator-is-wrong]]) says.
//
// This kernel is the same arithmetic with the prefetch distance as a template
// parameter: GROUPS k-groups are held in registers while GROUPS more are in
// flight. GROUPS=1 reproduces the shipping depth and is the control -- it must
// land on arm 0 or the rewrite changed something else.
template <int GROUPS, int LDS_SCALES, int KMAJOR>
__global__ void __launch_bounds__(256)
    w13_tile_deep(uint8_t const *__restrict__ W, float *sink, int tiles_total,
                  uint8_t const *__restrict__ scratch) {
  extern __shared__ char smem[];
  uint8_t *s_tok_fp8 = (uint8_t *)smem;
  uint8_t *s_tok_scales = s_tok_fp8 + FP8_TOK_DATA;
  uint8_t *s_scales =
      (uint8_t *)(((uintptr_t)(s_tok_scales + NUM_BLOCKS_32) + 15u) &
                  ~(uintptr_t)15);

  int const tid = threadIdx.x;
  int const warp_id = tid >> 6;
  int const lane_id = tid & 63;
  int const col = lane_id & 15;
  int const g = lane_id >> 4;

  for (int i = tid; i < FP8_TOK_DATA / 4; i += 256) {
    ((unsigned *)s_tok_fp8)[i] = 0x3c3c3c3cu;
  }
  for (int i = tid; i < NUM_BLOCKS_32; i += 256) {
    s_tok_scales[i] = 127;
  }
  __syncthreads();

  static_assert(MFMA_ITERS % GROUPS == 0, "GROUPS must divide the k-loop");

  for (int t = blockIdx.x; t < tiles_total; t += gridDim.x) {
    uint8_t const *wg_data = W + (size_t)t * (size_t)WG_BYTES;
    uint8_t const *wg_scales = wg_data + WG_DATA_BYTES;

    if constexpr (LDS_SCALES) {
      __syncthreads();
      for (int i = tid; i < WG_SCALE_BYTES / 16; i += 256) {
        ((i32x4_t *)s_scales)[i] = ((i32x4_t const *)wg_scales)[i];
      }
      __syncthreads();
    }

    constexpr int WAVE_SLAB = MFMA_ITERS * 1024;
    uint8_t const *w_packed = wg_data + (size_t)warp_id * WAVE_SLAB;
    int const w_row = warp_id * 16 + col;
    uint8_t const *w_data_row = wg_data + (size_t)w_row * W_ROW_BYTES;
    int const row_scale_base = w_row * NUM_BLOCKS_32;

    auto load_a = [&](int ki_) -> i32x4_t {
      if constexpr (KMAJOR) {
        return *(i32x4_t const *)(w_packed + ki_ * 1024 + lane_id * 16);
      } else {
        return *(i32x4_t const *)(w_data_row + ki_ * (K_PER_MFMA / 2) + g * 16);
      }
    };
    auto scale_at = [&](int idx) -> int {
      if constexpr (LDS_SCALES) {
        return (int)s_scales[idx];
      } else {
        return (int)wg_scales[idx];
      }
    };

    f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};
    i32x4_t A[GROUPS];
    int S[GROUPS];
#pragma unroll
    for (int j = 0; j < GROUPS; j++) {
      A[j] = load_a(j);
      S[j] = scale_at(row_scale_base + j * 4 + g);
    }

#pragma unroll 1
    for (int ki = 0; ki < MFMA_ITERS; ki += GROUPS) {
      i32x4_t N[GROUPS];
      int NS[GROUPS];
      // Issue the whole NEXT block of loads before consuming the current one,
      // so GROUPS x 4 dwordx4 are outstanding across GROUPS x 4 MFMAs.
#pragma unroll
      for (int j = 0; j < GROUPS; j++) {
        int kk = ki + GROUPS + j;
        if (kk < MFMA_ITERS) {
          N[j] = load_a(kk);
          NS[j] = scale_at(row_scale_base + kk * 4 + g);
        }
      }
#pragma unroll
      for (int j = 0; j < GROUPS; j++) {
        i32x8_t b = load_fp8_b(s_tok_fp8, (ki + j) * K_PER_MFMA, g);
        i32x8_t a = {};
        a[0] = A[j][0];
        a[1] = A[j][1];
        a[2] = A[j][2];
        a[3] = A[j][3];
        acc = mfma_f4xf8(a, b, acc, S[j], (int)s_tok_scales[ki + j]);
      }
#pragma unroll
      for (int j = 0; j < GROUPS; j++) {
        A[j] = N[j];
        S[j] = NS[j];
      }
    }

    if (acc[0] == 1.2345678e-31f && acc[1] == 1.2345678e-31f) {
      sink[0] = acc[0] + acc[1] + acc[2] + acc[3];
    }
  }
}

static uint8_t *g_W;
static float *g_sink;
static int g_tiles;
static size_t g_smem;
static uint8_t *g_scratch;

template <int SRC> static float run(int grid, int iters) {
  hipEvent_t e0, e1;
  hipEventCreate(&e0);
  hipEventCreate(&e1);
  auto go = [&] {
    hipLaunchKernelGGL((w13_tile<SRC>), dim3(grid), dim3(256), g_smem, 0, g_W,
                       g_sink, g_tiles, g_scratch);
  };
  go();
  hipDeviceSynchronize();
  hipEventRecord(e0);
  for (int i = 0; i < iters; i++) {
    go();
  }
  hipEventRecord(e1);
  hipEventSynchronize(e1);
  float ms = 0.f;
  hipEventElapsedTime(&ms, e0, e1);
  hipEventDestroy(e0);
  hipEventDestroy(e1);
  return ms / iters;
}

template <int G, int L, int K> static float run_deep(int grid, int iters) {
  hipEvent_t e0, e1;
  hipEventCreate(&e0);
  hipEventCreate(&e1);
  auto go = [&] {
    hipLaunchKernelGGL((w13_tile_deep<G, L, K>), dim3(grid), dim3(256), g_smem,
                       0, g_W, g_sink, g_tiles, g_scratch);
  };
  go();
  hipDeviceSynchronize();
  hipEventRecord(e0);
  for (int i = 0; i < iters; i++) {
    go();
  }
  hipEventRecord(e1);
  hipEventSynchronize(e1);
  float ms = 0.f;
  hipEventElapsedTime(&ms, e0, e1);
  hipEventDestroy(e0);
  hipEventDestroy(e1);
  return ms / iters;
}

int main() {
  hipDeviceProp_t p;
  HIP_CHECK(hipGetDeviceProperties(&p, 0));
  printf("device %s, %d CUs\n", p.gcnArchName, p.multiProcessorCount);

  int const grid = 64; // W13's live worker count, 8 tiles/XCD x 8 XCDs
  g_tiles = grid * 24; // 1536 tiles x 204 KB = 313 MB, past the MALL
  size_t const bytes = (size_t)g_tiles * (size_t)WG_BYTES;
  // Sized for the largest burst arm and held IDENTICAL across every arm, so
  // the LDS request is not a second variable. At grid = 64 on 256 CUs every
  // block owns a CU regardless, so this cannot move occupancy.
  g_smem = FP8_TOK_DATA + NUM_BLOCKS_32 + 16 + 49152;

  HIP_CHECK(hipMalloc(&g_W, bytes));
  HIP_CHECK(hipMalloc(&g_sink, sizeof(float)));
  HIP_CHECK(hipMalloc(&g_scratch, 49152));
  HIP_CHECK(hipMemset(g_scratch, 0x11, 49152));
  HIP_CHECK(hipMemset(g_W, 0x22, bytes));
  HIP_CHECK(hipMemset(g_sink, 0, sizeof(float)));

  printf("W13 geometry: OPW=%d, K=%d, %d MFMA iters, tile = %d KB "
         "(%d KB weight + %d KB scales)\n",
         OUTPUT_PER_WG, REDUCTION_SIZE, MFMA_ITERS, WG_BYTES / 1024,
         WG_DATA_BYTES / 1024, WG_SCALE_BYTES / 1024);
  printf("grid = %d blocks (W13 live width), %d tiles = %.0f MB, LDS = %zu B\n",
         grid, g_tiles, bytes / 1048576.0, g_smem);
  printf("shipping tile body, for reference: 13.9 us\n\n");

  struct {
    char const *name;
    float ms;
  } r[17];
  r[0] = {"global  (ships today)", run<0>(grid, 10)};
  r[1] = {"LDS-staged, read LDS", run<1>(grid, 10)};
  r[2] = {"none    (floor, not shippable)", run<2>(grid, 10)};
  r[3] = {"LDS-staged, read GLOBAL", run<3>(grid, 10)};
  r[4] = {"bare __syncthreads/tile", run<4>(grid, 10)};
  r[5] = {"burst UNRELATED 12 KB, read GLOBAL", run<5>(grid, 10)};
  r[6] = {"K-major repacked weight", run<6>(grid, 10)};
  r[7] = {"TWO bare __syncthreads/tile", run<7>(grid, 10)};
  r[8] = {"burst UNRELATED 4 KB", run<8>(grid, 10)};
  r[9] = {"burst UNRELATED 48 KB", run<9>(grid, 10)};
  r[10] = {"s_sleep(4) only, ~0.11 us", run<10>(grid, 10)};
  r[12] = {"s_sleep(16) only, ~0.43 us", run<12>(grid, 10)};
  r[11] = {"STACK: LDS scales + K-major", run<11>(grid, 10)};
  r[13] = {"4 KB LDS write only, no global", run<13>(grid, 10)};
  r[14] = {"4 KB global read only, no LDS", run<14>(grid, 10)};
  r[15] = {"(unused)", 0.f};
  r[16] = {"PER-BLOCK variable sleep", run<16>(grid, 10)};

  printf("  %-32s%10s%12s%14s\n", "scale source", "ms", "us/tile",
         "weight GB/s/CU");
  for (int i = 0; i < 17; i++) {
    if (r[i].ms == 0.f) {
      continue;
    }
    double us_tile = r[i].ms * 1000.0 / (g_tiles / (double)grid);
    double gbs = (double)WG_DATA_BYTES / (us_tile * 1e-6) / 1e9;
    printf("  %-32s%10.3f%12.2f%14.1f\n", r[i].name, r[i].ms, us_tile, gbs);
  }
  printf("\n  --- prefetch depth (separate kernel; GROUPS=1 is the control) "
         "---\n");
  struct {
    char const *name;
    float ms;
  } d[8];
  d[0] = {"GROUPS=1  (control = arm 0)", run_deep<1, 0, 0>(grid, 10)};
  d[1] = {"GROUPS=2", run_deep<2, 0, 0>(grid, 10)};
  d[2] = {"GROUPS=4", run_deep<4, 0, 0>(grid, 10)};
  d[3] = {"GROUPS=8", run_deep<8, 0, 0>(grid, 10)};
  d[4] = {"GROUPS=16", run_deep<16, 0, 0>(grid, 10)};
  d[5] = {"GROUPS=8  + LDS scales", run_deep<8, 1, 0>(grid, 10)};
  d[6] = {"GROUPS=8  + LDS scales + K-major", run_deep<8, 1, 1>(grid, 10)};
  d[7] = {"GROUPS=16 + LDS scales + K-major", run_deep<16, 1, 1>(grid, 10)};
  for (int i = 0; i < 8; i++) {
    double us_tile = d[i].ms * 1000.0 / (g_tiles / (double)grid);
    double gbs = (double)WG_DATA_BYTES / (us_tile * 1e-6) / 1e9;
    printf("  %-32s%10.3f%12.2f%14.1f\n", d[i].name, d[i].ms, us_tile, gbs);
  }

  // GROUPS=4 IS WHAT SHIPS. The knee here is 8, but 8 measured +0.26 ms at the
  // NP=4 wall while 4 measured -0.142, so every stacked multiplier has to be
  // priced on top of 4, not on top of the standalone's own optimum.
  printf("\n  --- multipliers ON TOP OF THE SHIPPING GROUPS=4 ---\n");
  struct {
    char const *name;
    float ms;
  } e[4];
  e[0] = {"GROUPS=4  (shipping default)", run_deep<4, 0, 0>(grid, 10)};
  e[1] = {"GROUPS=4  + LDS scales", run_deep<4, 1, 0>(grid, 10)};
  e[2] = {"GROUPS=4  + K-major", run_deep<4, 0, 1>(grid, 10)};
  e[3] = {"GROUPS=4  + LDS scales + K-major", run_deep<4, 1, 1>(grid, 10)};
  for (int i = 0; i < 4; i++) {
    double us_tile = e[i].ms * 1000.0 / (g_tiles / (double)grid);
    double gbs = (double)WG_DATA_BYTES / (us_tile * 1e-6) / 1e9;
    printf("  %-32s%10.3f%12.2f%14.1f\n", e[i].name, e[i].ms, us_tile, gbs);
  }


  printf("\nRead: arm0 vs arm2 prices the scattered E8M0 scale load itself.\n"
         "arm1 vs arm3 prices LDS as a scale SOURCE holding the barrier and\n"
         "burst fixed. arm4 vs arm0 prices a bare per-tile barrier alone. If\n"
         "arm3 and arm4 both land near arm1, the lever is keeping the four\n"
         "waves on the same tile, and it has nothing to do with scales.\n");
  return 0;
}
