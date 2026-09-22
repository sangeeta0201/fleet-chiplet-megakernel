/* gfx1250 FP8 (E4M3) block quantization of bf16 activations.
 *
 * The MX linear kernels multiply MXFP4 weights by FP8 E4M3 *tokens*, and the
 * tokens are quantized on the fly from bf16 into LDS. This is the gfx1250
 * counterpart of _gang_wave_parallel_fp8_quant in
 * tasks/mi300/gang_moe_linear_mxfp4_mi300.cuh.
 *
 * ---------------------------------------------------------------------------
 * TWO MEASURED FACTS, both of which silently corrupt every token if assumed
 * ---------------------------------------------------------------------------
 *
 * 1. THE gfx950 CONVERT BUILTIN DOES NOT EXIST HERE.
 *    __builtin_amdgcn_cvt_scalef32_pk_fp8_f32 requires target feature
 *    `fp8-cvt-scale-insts`, which gfx1250 does not have -- verified, the
 *    compiler rejects it outright. gfx1250 instead has a pk8 form,
 *    __builtin_amdgcn_cvt_scalef32_pk8_fp8_f32, which takes eight floats at
 *    once and returns two packed dwords. That is a better fit than the gfx950
 *    pairwise form (one instruction instead of four), so SUB_BLOCK work is
 *    done in groups of 8 rather than 4.
 *
 * 2. THE pk8 FORM DIVIDES BY THE SCALE. THE gfx950 FORM MULTIPLIES.
 *    Measured (tests/mi450/probe_fp8_quant_scale, retained as the
 *    scale-direction case in test_mx_quant.hip):
 *
 *        input 4.0, scale 1.00  -> 4.0
 *        input 4.0, scale 4.00  -> 1.0     <-- divided
 *        input 4.0, scale 0.25  -> 16.0
 *
 *    The gfx950 code computes `scale_f` as the E8M0 block scale and passes it
 *    straight in, relying on multiplication. Porting that line unchanged would
 *    apply the scale in the wrong direction: every token would come out scaled
 *    by 1/s^2 relative to the E8M0 byte the WMMA then multiplies back in.
 *
 *    With E8M0 scales near 2^0 -- which is exactly what a small test uses to
 *    keep values in range -- s^2 is also near 1, so this error is quietly
 *    small in a toy case and catastrophic on real activations. It is the kind
 *    of bug that passes a layout test and fails a model. Hence the reciprocal
 *    below, and hence test_mx_quant.hip asserts on a scale far from 1.0.
 */
#pragma once

#include "mirage/persistent_kernel/tasks/mi450/mx_layout_mi450.cuh"

namespace kernel {
namespace mi450 {

#if defined(MIRAGE_ARCH_GFX1250)

typedef float __attribute__((ext_vector_type(8))) mx_f32x8_cvt_t;
typedef unsigned int __attribute__((ext_vector_type(2))) mx_u32x2_t;

__device__ __forceinline__ float mx_bf16_to_float(unsigned short b) {
  unsigned u = (unsigned)b << 16;
  float f;
  __builtin_memcpy(&f, &u, 4);
  return f;
}

// E8M0 exponent for an FP8 E4M3 block. E4M3's largest finite magnitude is 448.
//
// Rounding UP on any nonzero mantissa is deliberate and is not a precision
// nicety: rounding down would put amax/scale slightly above 448 and saturate
// the block's largest element. Overshooting by at most one binade costs one
// bit of mantissa; undershooting clips.
__device__ __forceinline__ uint8_t mx_compute_e8m0_fp8(float amax) {
  if (amax == 0.0f) {
    return 0;
  }
  union {
    float f;
    uint32_t u;
  } v;
  v.f = amax * (1.0f / 448.0f);
  int raw_exp = (int)((v.u >> 23) & 0xFF);
  if (v.u & 0x7FFFFF) {
    raw_exp++;
  }
  return (uint8_t)max(0, min(255, raw_exp));
}

// Reduce `amax` across each aligned run of GROUP consecutive lanes.
//
// wave32, so __shfl_xor's width argument must not be left at the gfx950
// default of 64. The butterfly itself is width-agnostic as long as GROUP
// divides the wave, which the caller's static_assert enforces.
//
// Only valid when whole GROUPs are active: a lane whose loop condition failed
// never wrote `amax`, and reading it back yields an uninitialized register.
template <int GROUP>
__device__ __forceinline__ float mx_amax_group(float amax) {
#pragma unroll
  for (int off = 1; off < GROUP; off <<= 1) {
    amax = fmaxf(amax, __shfl_xor(amax, off, GROUP));
  }
  return amax;
}

// Quantize REDUCTION_SIZE bf16 values into FP8 E4M3 + E8M0 block scales.
//
//   s_tok_fp8    <- [REDUCTION_SIZE] bytes
//   s_tok_scales <- [REDUCTION_SIZE/32] bytes, one E8M0 per 32-element block
//
// Block size is fixed at 32 to match MX_K_PER_SCALE_BLOCK: the WMMA applies
// one E8M0 byte per 32 K elements, so the quantizer must agree exactly or the
// scales land on the wrong K range. This is simpler than the gfx950 version,
// which derived a variable SUB_BLOCK and then reduced amax across lanes with a
// butterfly; here one thread owns one whole 32-element block, so there is no
// cross-lane reduction and no ragged-tail special case at all.
//
// Requires blockDim.x threads to cooperate; caller must __syncthreads() before
// reading the results.
template <int REDUCTION_SIZE>
__device__ __forceinline__ void
    mx_quant_fp8(unsigned short const *__restrict__ src_bf16,
                 uint8_t *__restrict__ s_tok_fp8,
                 uint8_t *__restrict__ s_tok_scales) {
  static_assert(REDUCTION_SIZE % MX_K_PER_SCALE_BLOCK == 0,
                "REDUCTION_SIZE must be a multiple of the 32-element MX scale "
                "block");
  constexpr int NBLOCKS = REDUCTION_SIZE / MX_K_PER_SCALE_BLOCK;

  for (int blk = threadIdx.x; blk < NBLOCKS; blk += blockDim.x) {
    int const base = blk * MX_K_PER_SCALE_BLOCK;

    float vals[MX_K_PER_SCALE_BLOCK];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < MX_K_PER_SCALE_BLOCK; ++j) {
      vals[j] = mx_bf16_to_float(src_bf16[base + j]);
      amax = fmaxf(amax, fabsf(vals[j]));
    }

    uint8_t se = mx_compute_e8m0_fp8(amax);
    // Reconstruct 2^(se-127) by hand rather than calling ldexpf: se is already
    // the biased exponent field, so this is a shift, not a computation.
    float scale_f;
    if (se == 0) {
      scale_f = 1.0f;
    } else {
      union {
        float f;
        uint32_t u;
      } sv;
      sv.u = (uint32_t)se << 23;
      scale_f = sv.f;
    }

    // THE DIVIDE, NOT MULTIPLY, POINT. See fact 2 in the header. The builtin
    // computes value/scale, so handing it scale_f directly is correct here and
    // would be wrong on gfx950, where the same-named-looking builtin
    // multiplies.
#pragma unroll
    for (int j = 0; j < MX_K_PER_SCALE_BLOCK; j += 8) {
      mx_f32x8_cvt_t v8;
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        v8[t] = vals[j + t];
      }
      mx_u32x2_t packed =
          __builtin_amdgcn_cvt_scalef32_pk8_fp8_f32(v8, scale_f);
      *(mx_u32x2_t *)(s_tok_fp8 + base + j) = packed;
    }

    s_tok_scales[blk] = se;
  }
}

// ── Multi-row gather form ───────────────────────────────────────────────────
//
// The MoE kernel packs an expert's whole routed token set into the WMMA's N
// dimension: token column `c` lives at LDS row `c`, and the rows are gathered
// from scattered element offsets (W13 reads the norm buffer at tok*K; W2 reads
// the SwiGLU output at tok*(NUM_TOPK*I) + slot*I). This is the gfx1250
// counterpart of _gang_multirow_fp8_quant_gather in
// tasks/mi300/gang_moe_linear_mxfp4_mi300.cuh.
//
// THREE THINGS THAT ARE NOT THE mi300 VERSION, each of which would be a silent
// wrong answer if transcribed:
//
//  1. SCALE GRANULARITY IS 32, NOT 128. gfx950's MFMA takes ONE E8M0 byte per
//     128-element K tile, so its quantizer emits K/128 scales per row and
//     derives a variable SUB_BLOCK to feed a cross-lane amax butterfly.
//     gfx1250's WMMA takes FOUR bytes per 128-element tile, one per 32
//     elements. So SC_STRIDE here is K/32, one thread owns one whole 32-element
//     block, and there is no butterfly and no ragged-tail case at all. Passing
//     an mi300-shaped SC_STRIDE would make every scale after the first land on
//     the wrong K range.
//  2. THE CONVERT DIVIDES. See fact 2 in the file header.
//  3. NT_LOAD is a real requirement on the W2 side, not a hint -- see the note
//     at the call site.
//
// TOK_ROW_STRIDE is the LDS row pitch in bytes and is a caller parameter rather
// than K, because the MoE pads each row (K+16) to spread LDS banks. Both it and
// SC_STRIDE must be multiples of 16 and 4 respectively for the WMMA operand
// loads that read this buffer back; asserted below rather than assumed.
//
// `row_elem_off` is an LDS table of n_rows source element offsets, published by
// the caller's own __syncthreads before this is called. Rows [n_rows, ROWS) are
// left untouched: their columns are masked off by tok_active in the epilogue,
// and the WMMA reads them as whatever the previous phase left, which
// contributes to no stored output. (Zeroing them would cost a full pass over
// LDS to make no observable difference.)
template <int REDUCTION_SIZE,
          int ROWS,
          int TOK_ROW_STRIDE,
          int SC_STRIDE,
          bool NT_LOAD = false>
__device__ __forceinline__ void
    mx_quant_fp8_gather(unsigned short const *__restrict__ src_bf16,
                        int const *__restrict__ row_elem_off,
                        int n_rows,
                        uint8_t *__restrict__ s_tok_fp8,
                        uint8_t *__restrict__ s_tok_scales) {
  static_assert(REDUCTION_SIZE % MX_K_PER_SCALE_BLOCK == 0,
                "REDUCTION_SIZE must be a multiple of the 32-element MX scale "
                "block");
  static_assert(TOK_ROW_STRIDE % 16 == 0,
                "LDS token row pitch must be 16B aligned: the WMMA B operand "
                "reads it back with int4 loads");
  static_assert(SC_STRIDE >= REDUCTION_SIZE / MX_K_PER_SCALE_BLOCK,
                "scale row pitch is too small for 32-element scale blocks -- "
                "an mi300-shaped K/128 stride would alias adjacent rows");
  constexpr int NBLOCKS = REDUCTION_SIZE / MX_K_PER_SCALE_BLOCK;

  // ROWS is the number of rows the caller allocated in LDS. Clamping here
  // rather than trusting n_rows is deliberate: n_rows comes from a ballot
  // popcount over the routing table, so a routing buffer that has been
  // corrupted or mis-sized turns into an LDS write past the end of the token
  // region -- which lands in the weight tiles and produces a plausible-looking
  // wrong answer rather than a fault.
  int const rows = n_rows < ROWS ? n_rows : ROWS;

  // Flattened (row, block) work list. Flattened rather than nested so the
  // 256 threads stay busy when n_rows is small, which is the common case
  // (top-4 routing at batch 1 puts a single token on each expert).
  int const total = rows * NBLOCKS;
  for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
    int const r = idx / NBLOCKS;
    int const blk = idx - r * NBLOCKS;
    int const base = blk * MX_K_PER_SCALE_BLOCK;
    unsigned short const *src = src_bf16 + row_elem_off[r] + base;

    float vals[MX_K_PER_SCALE_BLOCK];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < MX_K_PER_SCALE_BLOCK; ++j) {
      unsigned short raw;
      if constexpr (NT_LOAD) {
        raw = __builtin_nontemporal_load(src + j);
      } else {
        raw = src[j];
      }
      vals[j] = mx_bf16_to_float(raw);
      amax = fmaxf(amax, fabsf(vals[j]));
    }

    uint8_t se = mx_compute_e8m0_fp8(amax);
    float scale_f;
    if (se == 0) {
      scale_f = 1.0f;
    } else {
      union {
        float f;
        uint32_t u;
      } sv;
      sv.u = (uint32_t)se << 23;
      scale_f = sv.f;
    }

    uint8_t *dst = s_tok_fp8 + (size_t)r * TOK_ROW_STRIDE + base;
#pragma unroll
    for (int j = 0; j < MX_K_PER_SCALE_BLOCK; j += 8) {
      mx_f32x8_cvt_t v8;
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        v8[t] = vals[j + t];
      }
      mx_u32x2_t packed =
          __builtin_amdgcn_cvt_scalef32_pk8_fp8_f32(v8, scale_f);
      *(mx_u32x2_t *)(dst + j) = packed;
    }

    s_tok_scales[(size_t)r * SC_STRIDE + blk] = se;
  }
}

// NT-load form of the single-row quantizer. Same body as mx_quant_fp8 with the
// loads marked non-temporal; used by the MoE W2 phase, whose input was written
// by a W13 tile on another XCD and is read exactly once.
template <int REDUCTION_SIZE>
__device__ __forceinline__ void
    mx_quant_fp8_nt(unsigned short const *__restrict__ src_bf16,
                    uint8_t *__restrict__ s_tok_fp8,
                    uint8_t *__restrict__ s_tok_scales) {
  static_assert(REDUCTION_SIZE % MX_K_PER_SCALE_BLOCK == 0,
                "REDUCTION_SIZE must be a multiple of the 32-element MX scale "
                "block");
  constexpr int NBLOCKS = REDUCTION_SIZE / MX_K_PER_SCALE_BLOCK;

  for (int blk = threadIdx.x; blk < NBLOCKS; blk += blockDim.x) {
    int const base = blk * MX_K_PER_SCALE_BLOCK;

    float vals[MX_K_PER_SCALE_BLOCK];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < MX_K_PER_SCALE_BLOCK; ++j) {
      vals[j] = mx_bf16_to_float(__builtin_nontemporal_load(src_bf16 + base + j));
      amax = fmaxf(amax, fabsf(vals[j]));
    }

    uint8_t se = mx_compute_e8m0_fp8(amax);
    float scale_f;
    if (se == 0) {
      scale_f = 1.0f;
    } else {
      union {
        float f;
        uint32_t u;
      } sv;
      sv.u = (uint32_t)se << 23;
      scale_f = sv.f;
    }

#pragma unroll
    for (int j = 0; j < MX_K_PER_SCALE_BLOCK; j += 8) {
      mx_f32x8_cvt_t v8;
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        v8[t] = vals[j + t];
      }
      mx_u32x2_t packed =
          __builtin_amdgcn_cvt_scalef32_pk8_fp8_f32(v8, scale_f);
      *(mx_u32x2_t *)(s_tok_fp8 + base + j) = packed;
    }

    s_tok_scales[blk] = se;
  }
}

#endif // MIRAGE_ARCH_GFX1250

} // namespace mi450
} // namespace kernel
