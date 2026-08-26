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

// ── THE buffer_load_lds RING ───────────────────────────────────────────────
//
// Sixth pass. The depth curve above says the tile wants depth 8-16, and the
// two best cells both need LDS-staged scales on top:
//
//   GROUPS=4  + K-major            9.67 us   <- SHIPS
//   GROUPS=8  + LDS scales + Kmaj  8.56 us
//   GROUPS=16 + LDS scales + Kmaj  7.86 us
//
// but depth in REGISTERS is not free, and at the wall it has cost every time
// it was tried: MPK_MOE_PF_GROUPS=8 measured +0.260 ms, MPK_MOE_PF_DBUF=1
// +0.226 ms (+31 unified VGPR), MPK_ATTN_PF_DBUF=1 +0.226 ms (284 -> 310).
// GROUPS=16 holds 16 x i32x4 A plus 16 x i32x4 N = 128 VGPRs of pipeline
// alone. Every standalone win in this file is therefore unshippable as
// written, and the depth axis looks closed.
//
// It is not, because `buffer_load_dwordx4 ... lds:1` has NO DESTINATION VGPR.
// The data lands straight in LDS and completion is tracked by vmcnt, so a
// DEPTH-deep pipeline costs DEPTH x 4 KB of LDS -- which is free, 77 KB of the
// worker's 155 KB request is not load-bearing
// ([[glm-agpr-is-spill-slack-lds-half-is-free]]) -- and zero registers. This
// arm prices that, and it is the only untested point on the depth axis.
//
// Mechanics, per gfx9 MUBUF LDS mode: the lane's global address is the usual
// base + voffset(lane), and the returned 16 bytes go to LDS at
// M0 + lds_ptr + lane_id * 16. One wave64 therefore fills 1024 contiguous LDS
// bytes -- exactly one K-major k-group -- and the four waves need distinct
// 1 KB slices. The ring is DEPTH slots x NUM_WAVES x 1 KB.
//
// Two things the compiler cannot see and must be told:
//   * it does not know buffer_load_lds writes s_ring, so the ds_read needs a
//     hand-written `s_waitcnt vmcnt(DEPTH-1)` (in-order retirement makes that
//     exactly "the oldest outstanding load has landed"), and
//   * it must not hoist the next issue above the ds_read of the slot that
//     issue overwrites -- hence sched_barrier(0) on both sides.
// The steady-state block always issues, so its wait is a constant DEPTH-1; the
// final block issues nothing and is peeled so its waits can count down.
extern "C" __device__ void __llvm_amdgcn_raw_buffer_load_lds(
    i32x4_t rsrc,
    __attribute__((address_space(3))) uint32_t *lds_ptr,
    int32_t size,
    int32_t voffset,
    int32_t soffset,
    int32_t imm_offset,
    int32_t aux) __asm("llvm.amdgcn.raw.buffer.load.lds");

__device__ __forceinline__ i32x4_t make_w_buffer_rsrc(void const *base,
                                                      uint32_t range_bytes) {
  i32x4_t r;
  uint64_t addr = reinterpret_cast<uint64_t>(base);
  r[0] = static_cast<int>(
      __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(addr & 0xFFFFFFFFu)));
  r[1] = static_cast<int>(
      __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(addr >> 32)));
  r[2] = static_cast<int>(__builtin_amdgcn_readfirstlane(range_bytes));
  r[3] = static_cast<int>(0x00020000u);
  return r;
}

template <int N> __device__ __forceinline__ void wait_vmcnt() {
  asm volatile("s_waitcnt vmcnt(%0)" ::"n"(N) : "memory");
}

// aux = 1 is sc0, deliberately NOT gpt-oss's 3 (sc0|nt): nt marks the line
// evict-first and the tile is re-read by nobody but is large, so nt measured
// worse in the o_proj prefetch (4.674 vs 4.486).
constexpr int DMA_AUX = 1;

// The peeled final block. `#pragma unroll` does NOT make its induction
// variable a constant expression, and the countdown wait needs one, so the
// tail is unrolled by recursion instead.
template <int DEPTH, int S, typename F>
__device__ __forceinline__ void dma_tail(F &step) {
  if constexpr (S < DEPTH) {
    wait_vmcnt<DEPTH - 1 - S>();
    step(S);
    dma_tail<DEPTH, S + 1>(step);
  }
}

template <int DEPTH>
__global__ void __launch_bounds__(256)
    w13_tile_dma(uint8_t const *__restrict__ W, float *sink, int tiles_total,
                 uint8_t const *__restrict__ scratch) {
  extern __shared__ char smem[];
  uint8_t *s_tok_fp8 = (uint8_t *)smem;
  uint8_t *s_tok_scales = s_tok_fp8 + FP8_TOK_DATA;
  uint8_t *s_scales =
      (uint8_t *)(((uintptr_t)(s_tok_scales + NUM_BLOCKS_32) + 15u) &
                  ~(uintptr_t)15);
  uint8_t *s_ring = s_scales + WG_SCALE_BYTES; // 12 KB, already 16B-aligned

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

  static_assert(MFMA_ITERS % DEPTH == 0, "DEPTH must divide the k-loop");
  static_assert(MFMA_ITERS / DEPTH >= 2, "need one steady block plus the tail");
  constexpr int NBLK = MFMA_ITERS / DEPTH;
  constexpr int WAVE_SLAB = MFMA_ITERS * 1024;

  i32x4_t const rsrc =
      make_w_buffer_rsrc(W, (uint32_t)((size_t)tiles_total * (size_t)WG_BYTES));

  for (int t = blockIdx.x; t < tiles_total; t += gridDim.x) {
    int const tile_base = t * WG_BYTES;
    uint8_t const *wg_scales = W + (size_t)tile_base + WG_DATA_BYTES;

    // Scales staged once, coalesced -- required, not optional: the depth sweep
    // shows LDS scales HURT at depth 4 (10.90 vs 9.67) and only pay at 8+.
    __syncthreads();
    for (int i = tid; i < WG_SCALE_BYTES / 16; i += 256) {
      ((i32x4_t *)s_scales)[i] = ((i32x4_t const *)wg_scales)[i];
    }
    __syncthreads();

    int const wave_voff = tile_base + warp_id * WAVE_SLAB + lane_id * 16;
    int const row_scale_base = (warp_id * 16 + col) * NUM_BLOCKS_32;

    auto issue = [&](int kk, int slot) {
      auto *dst = (__attribute__((address_space(3))) uint32_t
                       *)(s_ring + ((size_t)slot * NUM_WAVES + warp_id) * 1024);
      __llvm_amdgcn_raw_buffer_load_lds(rsrc, dst, 16, wave_voff + kk * 1024, 0,
                                        0, DMA_AUX);
    };
    auto consume = [&](int kk, int slot, f32x4_t acc) -> f32x4_t {
      i32x4_t lo = *(i32x4_t const *)(s_ring +
                                      ((size_t)slot * NUM_WAVES + warp_id) *
                                          1024 +
                                      lane_id * 16);
      i32x8_t a = {};
      a[0] = lo[0];
      a[1] = lo[1];
      a[2] = lo[2];
      a[3] = lo[3];
      i32x8_t b = load_fp8_b(s_tok_fp8, kk * K_PER_MFMA, g);
      return mfma_f4xf8(a, b, acc, (int)s_scales[row_scale_base + kk * 4 + g],
                        (int)s_tok_scales[kk]);
    };

#pragma unroll
    for (int j = 0; j < DEPTH; j++) {
      issue(j, j);
    }

    f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll 1
    for (int blk = 0; blk < NBLK - 1; blk++) {
      int const base = blk * DEPTH;
#pragma unroll
      for (int s = 0; s < DEPTH; s++) {
        wait_vmcnt<DEPTH - 1>();
        acc = consume(base + s, s, acc);
        __builtin_amdgcn_sched_barrier(0);
        issue(base + DEPTH + s, s);
        __builtin_amdgcn_sched_barrier(0);
      }
    }
    {
      constexpr int base = (NBLK - 1) * DEPTH;
      auto step = [&](int s) { acc = consume(base + s, s, acc); };
      dma_tail<DEPTH, 0>(step);
    }

    if (acc[0] == 1.2345678e-31f && acc[1] == 1.2345678e-31f) {
      sink[0] = acc[0] + acc[1] + acc[2] + acc[3];
    }
  }
}

// ══ SIXTH PASS VERDICT: THE DEPTH AXIS IS CLOSED. ═════════════════════════
//
//   arm                                   us/tile   standalone VGPR   LDS ring
//   deep G=4  + K-major   (SHIPS)            9.62         72             --
//   deep G=8  + LDS sc + Kmaj                8.57        116             --
//   deep G=16 + LDS sc + Kmaj                7.95        198             --
//   DMA one-slot ring, DEPTH=16             20.69         57            64 KB
//   DMA2 half-buffered G=8                   9.91         47            64 KB
//   DMA2 half-buffered G=12                  9.67         50            96 KB
//   DMA2 half-buffered G=16                  9.17         57           128 KB
//   DMA BW-ONLY (no ds_read, no MFMA)        4.29          -            64 KB
//
// Three things fall out, and together they shut the axis.
//
// 1. THE BYTES ARE NOT THE PROBLEM. buffer_load_lds moves the tile's whole
//    192 KB in 4.29 us = 45.9 GB/s/CU, at or slightly past the 41.6 GB/s/CU
//    that test_narrow_grid_bandwidth measured for a CU in a 64-wide phase. So
//    the shipping 9.62 us tile is 55% NON-BYTE, and the older "29% of its real
//    roof" reading above is about the k-loop's structure, not about a
//    bandwidth the hardware will not give.
//
// 2. THE DMA PAYS A STRUCTURAL LDS TAX THE REGISTER PATH DOES NOT. A global
//    load lands in a VGPR; a DMA load lands in LDS and must then be ds_read
//    out. That is 192 KB of extra LDS write plus 192 KB of extra LDS read per
//    tile, and it is exactly the ~1.3 us that separates DMA2 G=8 (9.91) from
//    deep G=8 + LDS scales + K-major (8.57) at identical depth. Depth is free
//    in registers with the DMA, but the delivery is not free in LDS.
//
// 3. THE REGISTERS IT SAVES ARE NOT SCARCE. Measured on the real image with
//    /tmp/vgpr.sh: worker_kernel .vgpr_count is 325 at MPK_MOE_PF_GROUPS = 2
//    AND at 4, and 362 at 8. The MoE k-loop has ZERO register headroom value
//    below depth 8 -- something else sets the 325 -- so cutting it 72 -> 47
//    buys nothing at the wall, and the whole +0.260 ms of PF_GROUPS=8 is the
//    single 325 -> 362 step.
//
// What would ship is DMA2 at G=8, the deepest ring that fits the ~77 KB of
// free worker LDS ([[glm-agpr-is-spill-slack-lds-half-is-free]]), and it is
// +0.29 us/tile WORSE than what ships for zero compensating register win.
// G=16 does beat shipping by 0.45 us/tile but wants a 128 KB ring. Even taken
// at the optimistic 0.129 ms-per-us/tile conversion that is 0.06 ms, under the
// 0.26 ms noise floor, and it does not fit.
//
// ── DMA, HALF-BUFFERED ─────────────────────────────────────────────────────
//
// The one-slot-at-a-time ring above is 20.7-24.8 us against 9.62 shipping, but
// the BW-ONLY probe below says the DMA delivers the whole tile in 4.3 us at
// 45.9 GB/s/CU. So the instruction path is not the problem: the ring serialized
// `wait -> ds_read -> lgkmcnt(0) -> MFMA -> issue` with one wave per SIMD, so
// every k-group paid a full exposed LDS round trip 48 times.
//
// This form separates the two pipelines. The HBM pipeline is a double buffer of
// G-group halves -- G to 2G KB per wave in flight, zero destination VGPRs. The
// LDS pipeline is the G independent ds_reads inside a half, which the compiler
// batches, so LDS latency is paid once per half instead of once per group.
// Register cost is only the ds_read results (4 VGPRs x G), against the deep
// kernel's A[G] + N[G] = 8 VGPRs x G.
template <int G>
__global__ void __launch_bounds__(256)
    w13_tile_dma2(uint8_t const *__restrict__ W, float *sink, int tiles_total,
                  uint8_t const *__restrict__ scratch) {
  extern __shared__ char smem[];
  uint8_t *s_tok_fp8 = (uint8_t *)smem;
  uint8_t *s_tok_scales = s_tok_fp8 + FP8_TOK_DATA;
  uint8_t *s_scales =
      (uint8_t *)(((uintptr_t)(s_tok_scales + NUM_BLOCKS_32) + 15u) &
                  ~(uintptr_t)15);
  uint8_t *s_ring = s_scales + WG_SCALE_BYTES;

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

  static_assert(MFMA_ITERS % G == 0, "G must divide the k-loop");
  constexpr int NB = MFMA_ITERS / G;
  constexpr int WAVE_SLAB = MFMA_ITERS * 1024;

  i32x4_t const rsrc =
      make_w_buffer_rsrc(W, (uint32_t)((size_t)tiles_total * (size_t)WG_BYTES));

  for (int t = blockIdx.x; t < tiles_total; t += gridDim.x) {
    int const tile_base = t * WG_BYTES;
    uint8_t const *wg_scales = W + (size_t)tile_base + WG_DATA_BYTES;

    __syncthreads();
    for (int i = tid; i < WG_SCALE_BYTES / 16; i += 256) {
      ((i32x4_t *)s_scales)[i] = ((i32x4_t const *)wg_scales)[i];
    }
    __syncthreads();

    int const wave_voff = tile_base + warp_id * WAVE_SLAB + lane_id * 16;
    int const row_scale_base = (warp_id * 16 + col) * NUM_BLOCKS_32;
    // Half h occupies slots [h*G, h*G+G); wave w owns 1 KB inside each slot.
    auto half_lds = [&](int h) {
      return s_ring + ((size_t)h * G * NUM_WAVES + warp_id) * 1024;
    };
    auto issue_half = [&](int blk, int h) {
      uint8_t *base = half_lds(h);
#pragma unroll
      for (int s = 0; s < G; s++) {
        auto *dst = (__attribute__((address_space(3))) uint32_t
                         *)(base + (size_t)s * NUM_WAVES * 1024);
        __llvm_amdgcn_raw_buffer_load_lds(
            rsrc, dst, 16, wave_voff + (blk * G + s) * 1024, 0, 0, DMA_AUX);
      }
    };

    f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};
    issue_half(0, 0);

#pragma unroll 1
    for (int b = 0; b < NB; b++) {
      int const h = b & 1;
      __builtin_amdgcn_sched_barrier(0);
      if (b + 1 < NB) {
        issue_half(b + 1, 1 - h);
        wait_vmcnt<G>(); // half h has landed; half 1-h stays in flight
      } else {
        wait_vmcnt<0>();
      }
      __builtin_amdgcn_sched_barrier(0);
      uint8_t const *base = half_lds(h);
      int const kbase = b * G;
#pragma unroll
      for (int s = 0; s < G; s++) {
        i32x4_t lo = *(i32x4_t const *)(base + (size_t)s * NUM_WAVES * 1024 +
                                        lane_id * 16);
        i32x8_t a = {};
        a[0] = lo[0];
        a[1] = lo[1];
        a[2] = lo[2];
        a[3] = lo[3];
        i32x8_t bv = load_fp8_b(s_tok_fp8, (kbase + s) * K_PER_MFMA, g);
        acc = mfma_f4xf8(a, bv, acc,
                         (int)s_scales[row_scale_base + (kbase + s) * 4 + g],
                         (int)s_tok_scales[kbase + s]);
      }
    }

    if (acc[0] == 1.2345678e-31f && acc[1] == 1.2345678e-31f) {
      sink[0] = acc[0] + acc[1] + acc[2] + acc[3];
    }
  }
}

// Pure HBM->LDS throughput at this shape: the same 48 x 1 KB per wave per
// tile, the same ring, the same waits, and NO ds_read and NO MFMA. If this
// alone is slower than the full global-load k-loop, the DMA instruction path
// is the bottleneck and no amount of restructuring above it can help.
template <int DEPTH>
__global__ void __launch_bounds__(256)
    w13_dma_bw(uint8_t const *__restrict__ W, float *sink, int tiles_total,
               uint8_t const *__restrict__ scratch) {
  extern __shared__ char smem[];
  uint8_t *s_ring = (uint8_t *)smem;
  int const tid = threadIdx.x;
  int const warp_id = tid >> 6;
  int const lane_id = tid & 63;

  static_assert(MFMA_ITERS % DEPTH == 0, "DEPTH must divide the k-loop");
  constexpr int NBLK = MFMA_ITERS / DEPTH;
  constexpr int WAVE_SLAB = MFMA_ITERS * 1024;

  i32x4_t const rsrc =
      make_w_buffer_rsrc(W, (uint32_t)((size_t)tiles_total * (size_t)WG_BYTES));

  for (int t = blockIdx.x; t < tiles_total; t += gridDim.x) {
    int const wave_voff =
        t * WG_BYTES + warp_id * WAVE_SLAB + lane_id * 16;
    auto issue = [&](int kk, int slot) {
      auto *dst = (__attribute__((address_space(3))) uint32_t
                       *)(s_ring + ((size_t)slot * NUM_WAVES + warp_id) * 1024);
      __llvm_amdgcn_raw_buffer_load_lds(rsrc, dst, 16, wave_voff + kk * 1024, 0,
                                        0, DMA_AUX);
    };
#pragma unroll
    for (int j = 0; j < DEPTH; j++) {
      issue(j, j);
    }
#pragma unroll 1
    for (int blk = 1; blk < NBLK; blk++) {
#pragma unroll
      for (int s = 0; s < DEPTH; s++) {
        wait_vmcnt<DEPTH - 1>();
        __builtin_amdgcn_sched_barrier(0);
        issue(blk * DEPTH + s, s);
        __builtin_amdgcn_sched_barrier(0);
      }
    }
    wait_vmcnt<0>();
    __syncthreads();
    if (s_ring[lane_id] == 0xABu && s_ring[lane_id + 1] == 0xCDu) {
      sink[0] = 1.0f;
    }
  }
}

static uint8_t *g_W;
static float *g_sink;
static int g_tiles;
static size_t g_smem;
static size_t g_smem_dma;
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

template <int D> static float run_dma(int grid, int iters) {
  hipEvent_t e0, e1;
  hipEventCreate(&e0);
  hipEventCreate(&e1);
  auto go = [&] {
    hipLaunchKernelGGL((w13_tile_dma<D>), dim3(grid), dim3(256), g_smem_dma, 0,
                       g_W, g_sink, g_tiles, g_scratch);
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

template <int D> static float run_dma2(int grid, int iters) {
  hipEvent_t e0, e1;
  hipEventCreate(&e0);
  hipEventCreate(&e1);
  auto go = [&] {
    hipLaunchKernelGGL((w13_tile_dma2<D>), dim3(grid), dim3(256), g_smem_dma, 0,
                       g_W, g_sink, g_tiles, g_scratch);
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

template <int D> static float run_dma_bw(int grid, int iters) {
  hipEvent_t e0, e1;
  hipEventCreate(&e0);
  hipEventCreate(&e1);
  auto go = [&] {
    hipLaunchKernelGGL((w13_dma_bw<D>), dim3(grid), dim3(256), g_smem_dma, 0,
                       g_W, g_sink, g_tiles, g_scratch);
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
  // The DMA arms need the ring on top of the staged scales. Sized for the
  // deepest arm (DEPTH=16 -> 64 KB) and held identical across the DMA arms.
  // Still 1 block/CU at grid = 64 on 256 CUs, so this cannot move occupancy.
  g_smem_dma = FP8_TOK_DATA + NUM_BLOCKS_32 + 16 + WG_SCALE_BYTES +
               2 * 16 * NUM_WAVES * 1024;

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


  // The point of the whole file: depth WITHOUT registers.
  printf("\n  --- buffer_load_lds ring (zero destination VGPRs) ---\n"
         "  reference cells: shipping G=4+Kmaj 9.67, reg-best G=16+LDS+Kmaj "
         "7.86, byte floor 4.9\n");
  struct {
    char const *name;
    float ms;
  } f[4];
  f[0] = {"DMA DEPTH=4   (16 KB ring)", run_dma<4>(grid, 10)};
  f[1] = {"DMA DEPTH=8   (32 KB ring)", run_dma<8>(grid, 10)};
  f[2] = {"DMA DEPTH=12  (48 KB ring)", run_dma<12>(grid, 10)};
  f[3] = {"DMA DEPTH=16  (64 KB ring)", run_dma<16>(grid, 10)};
  struct {
    char const *name;
    float ms;
  } fb[2];
  struct {
    char const *name;
    float ms;
  } f2[4];
  f2[0] = {"DMA2 half-buffered G=6", run_dma2<6>(grid, 10)};
  f2[1] = {"DMA2 half-buffered G=8", run_dma2<8>(grid, 10)};
  f2[2] = {"DMA2 half-buffered G=12", run_dma2<12>(grid, 10)};
  f2[3] = {"DMA2 half-buffered G=16", run_dma2<16>(grid, 10)};
  fb[0] = {"DMA BW-ONLY DEPTH=8 (no MFMA)", run_dma_bw<8>(grid, 10)};
  fb[1] = {"DMA BW-ONLY DEPTH=16 (no MFMA)", run_dma_bw<16>(grid, 10)};
  for (int i = 0; i < 4; i++) {
    double us_tile = f[i].ms * 1000.0 / (g_tiles / (double)grid);
    double gbs = (double)WG_DATA_BYTES / (us_tile * 1e-6) / 1e9;
    printf("  %-32s%10.3f%12.2f%14.1f\n", f[i].name, f[i].ms, us_tile, gbs);
  }
  for (int i = 0; i < 4; i++) {
    double us_tile = f2[i].ms * 1000.0 / (g_tiles / (double)grid);
    double gbs = (double)WG_DATA_BYTES / (us_tile * 1e-6) / 1e9;
    printf("  %-32s%10.3f%12.2f%14.1f\n", f2[i].name, f2[i].ms, us_tile, gbs);
  }
  for (int i = 0; i < 2; i++) {
    double us_tile = fb[i].ms * 1000.0 / (g_tiles / (double)grid);
    double gbs = (double)WG_DATA_BYTES / (us_tile * 1e-6) / 1e9;
    printf("  %-32s%10.3f%12.2f%14.1f\n", fb[i].name, fb[i].ms, us_tile, gbs);
  }

  printf("\nRead: arm0 vs arm2 prices the scattered E8M0 scale load itself.\n"
         "arm1 vs arm3 prices LDS as a scale SOURCE holding the barrier and\n"
         "burst fixed. arm4 vs arm0 prices a bare per-tile barrier alone. If\n"
         "arm3 and arm4 both land near arm1, the lever is keeping the four\n"
         "waves on the same tile, and it has nothing to do with scales.\n");
  return 0;
}
