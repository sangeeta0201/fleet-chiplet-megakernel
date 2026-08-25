/* MXFP4 matrix primitives for MI450 (gfx1250).
 *
 * This is the datapath the README credits with the 10.7 -> 5.2 ms MoE win on
 * gfx950, and the good news is that the *tile shape survives the port*: both
 * architectures do 16x16x128 f8f6f4 with hardware dequant. What changes is the
 * instruction and the per-lane operand width.
 *
 *   gfx950   v_mfma_scale_f32_16x16x128_f8f6f4   wave64
 *              A = i32x8  (32 B/lane)
 *              B = i32x8  (32 B/lane)
 *              C = f32x4  (4 f32/lane)
 *
 *   gfx1250  v_wmma_scale_f32_16x16x128_f8f6f4   wave32
 *              A = i32x16 (64 B/lane declared, only 32 B live for FP4)
 *              B = i32x16 (64 B/lane declared, only 32 B live for FP4)
 *              C = f32x8  (8 f32/lane)
 *
 * The tile holds the same 16x16x128 of data either way. For FP4 the operand
 * type is oversized: 16 rows x 128 k x 4 bit = 1024 B, which over 32 lanes is
 * 32 B/lane, and a live-byte scan on FFM confirms the hardware consumes only
 * bytes 0..31 of each i32x16. So the MX datapath costs the same 8 live VGPRs
 * per operand as gfx950 -- there is no register-pressure penalty here. (The
 * i32x16 width is there for the FP8 case, where all 64 B are live.)
 *
 * The per-lane element mapping is NOT a rescaling of the wave64 one, and is
 * documented and derived in mx_layout_mi450.cuh. Do not hand-index these
 * operands without reading that file: the K dimension is interleaved between
 * the two lane-halves, and the scale operand has two separate traps.
 *
 * The builtin signature also changes shape, and not in a way that can be
 * guessed -- decoded from BuiltinsAMDGPU.def
 * ("V8fIiV16iIiV16iIsV8fIiIiiIiIiiIbIb") and confirmed by reading emitted ISA:
 *
 *   wmma_scale(A_fmt, A, B_fmt, B, C_mod, C,
 *              scaleA_sel, scaleA_fmt, scaleA,
 *              scaleB_sel, scaleB_fmt, scaleB,
 *              reuse_a, reuse_b)
 *
 * versus the gfx950 form, where the format selectors are the cbsz/blgp
 * immediates sitting *after* the accumulator:
 *
 *   mfma_scale(A, B, C, cbsz, blgp, scaleA_sel, scaleA, scaleB_sel, scaleB)
 *
 * So a mechanical translation of the argument list produces something that
 * compiles and is wrong. Hence the wrappers below, which keep the mi300 call
 * sites' argument order and do the remapping in one place.
 *
 * STATUS: validated under FFM-Lite -- see tests/mi450/test_mx_wmma.hip. Format
 * codes and the FP4 K-tile layout are checked against a host reference that
 * decodes E2M1/E4M3 independently.
 */
#pragma once

#include "mirage/persistent_kernel/arch_traits.cuh"

namespace kernel {
namespace mi450 {

#if defined(MIRAGE_ARCH_GFX1250)

typedef int __attribute__((ext_vector_type(4))) mx_i32x4_t;
typedef int __attribute__((ext_vector_type(8))) mx_i32x8_t;
typedef int __attribute__((ext_vector_type(16))) mx_i32x16_t;
typedef float __attribute__((ext_vector_type(8))) mx_f32x8_t;

// Operand format codes for the A_fmt/B_fmt immediates. Same encoding the
// gfx950 cbsz/blgp immediates used, just in a different argument position.
constexpr int MX_FMT_FP8_E4M3 = 0;
constexpr int MX_FMT_FP4_E2M1 = 4;

// Scale format: 0 = E8M0, the MX standard exponent-only scale.
//
// The scale_sel immediate is NOT a block selector. It is SCL_OPSEL: 0 = read
// scale bytes from lanes 0..15, 1 = read them from lanes 16..31 (unused for
// this shape). Which K block a scale byte covers is fixed by its *byte
// position* in the 32-bit operand -- byte b covers k in [32b, 32b+32).
// Measured on FFM and stated in the MI400 guide S4.6.12.6.2. Always pass 0.
//
// Related trap: the scale operand must be lane-varying so it lands in a VGPR.
// From an SGPR only bits [7:0] are read, so blocks 1..3 silently inherit
// block 0's scale. See mx_layout_mi450.cuh.
constexpr int MX_SCALE_FMT_E8M0 = 0;

// FP4xFP8 scaled WMMA: 16x16x128 with hardware dequant.
// A = weights (FP4 E2M1), B = tokens (FP8 E4M3).
//
// Argument order deliberately mirrors _gang_mfma_f4xf8 in the mi300 tree so
// call sites port without reordering.
__device__ __forceinline__ mx_f32x8_t _gang_wmma_f4xf8(mx_i32x16_t a,
                                                       mx_i32x16_t b,
                                                       mx_f32x8_t c,
                                                       int scale_a,
                                                       int scale_b) {
  return __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
      MX_FMT_FP4_E2M1, // A is FP4 E2M1 (weights)
      a,
      MX_FMT_FP8_E4M3, // B is FP8 E4M3 (tokens)
      b,
      0, // C modifier: none
      c,
      0,                 // scaleA byte select
      MX_SCALE_FMT_E8M0, // scaleA format
      scale_a,
      0,                 // scaleB byte select
      MX_SCALE_FMT_E8M0, // scaleB format
      scale_b,
      false, // reuse_a -- power hint, tuning knob once correct
      false); // reuse_b
}

// FP4xFP4 scaled WMMA: both operands FP4 E2M1. Half the K-depth cost of the
// mixed form on gfx950; whether that ratio holds on gfx1250 is a performance
// question FFM cannot answer.
__device__ __forceinline__ mx_f32x8_t _gang_wmma_f4xf4(mx_i32x16_t a,
                                                       mx_i32x16_t b,
                                                       mx_f32x8_t c,
                                                       int scale_a,
                                                       int scale_b) {
  return __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(MX_FMT_FP4_E2M1,
                                                          a,
                                                          MX_FMT_FP4_E2M1,
                                                          b,
                                                          0,
                                                          c,
                                                          0,
                                                          MX_SCALE_FMT_E8M0,
                                                          scale_a,
                                                          0,
                                                          MX_SCALE_FMT_E8M0,
                                                          scale_b,
                                                          false,
                                                          false);
}

// ---------------------------------------------------------------------------
// FP4 dequant
// ---------------------------------------------------------------------------
//
// gfx950 converts 2 FP4 values per call (v_cvt_scalef32_pk_bf16_fp4); gfx1250
// converts 8 (v_cvt_scale_pk8_bf16_fp4). Callers that looped in steps of 2
// should step by 8 -- leaving the old stride merely wastes 4x the
// instructions, but a caller that assumed a 2-wide *write* will now overrun
// its destination by 6 elements, which is silent corruption rather than a
// compile error.
typedef __bf16 __attribute__((ext_vector_type(8))) mx_bf16x8_t;

// Dequantize 8 packed FP4 E2M1 values (one uint = 8 nibbles) with an E8M0
// scale, producing 8 bf16.
__device__ __forceinline__ mx_bf16x8_t mx_cvt_pk8_bf16_fp4(unsigned int packed,
                                                           unsigned int scale) {
  return __builtin_amdgcn_cvt_scale_pk8_bf16_fp4(packed, scale, 0);
}

#endif // MIRAGE_ARCH_GFX1250

} // namespace mi450
} // namespace kernel
