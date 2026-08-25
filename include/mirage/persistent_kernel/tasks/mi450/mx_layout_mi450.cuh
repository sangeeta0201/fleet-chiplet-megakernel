/* gfx1250 MX (block-scaled FP4) WMMA fragment layout.
 *
 * WHY THIS FILE EXISTS
 *
 * The gfx950 MoE kernels hardcode a wave64 MFMA fragment layout:
 *
 *     col = lane & 15;   // output row within the 16x16 tile
 *     g   = lane >> 4;   // K-group 0..3, 32 FP4 nibbles each
 *
 * That mapping cannot be rescaled to wave32 by inspection. gfx1250 halves the
 * lane count while keeping the same 16x16x128 tile, so each lane must carry
 * twice the data -- but the operand register is i32x16 (64 B/lane) and the A
 * tile is only 1024 B, which over 32 lanes is 32 B/lane. Something had to be
 * either unused or interleaved, and guessing wrong compiles cleanly and
 * silently computes garbage.
 *
 * So it was measured on FFM-Lite with one-hot probes (tests/mi450/, and
 * /tmp probes probe_layout/probe2/probe3/probe5/probe6), then checked against
 * the MI400 Shader Programming Guide S4.6.12.6.2. The two agree exactly.
 *
 * ---------------------------------------------------------------------------
 * MEASURED LAYOUT -- V_WMMA_SCALE_F32_16X16X128_F8F6F4, both operands FP4
 * ---------------------------------------------------------------------------
 *
 * Accumulator (f32x8 per lane, D = A x B, D is 16x16):
 *
 *     C[m][n]  lives in  lane = 16*(m/8) + n,  slot = m % 8
 *
 *   i.e. lanes 0..15 hold output rows 0..7, lanes 16..31 hold rows 8..15, and
 *   the lane index within each half is the output column. This is NOT the
 *   gfx950 arrangement (there, acc[i] = C[g*4+i][col] with g = lane>>4 over
 *   four groups).
 *
 * A operand (16 rows x 128 K, FP4) -- and B identically, with n for m:
 *
 *     lane   = 16*h + m           where h in {0,1}
 *     nibble = 32*(k/64) + (k%32) where h = (k%64)/32
 *
 *   Read the other way: lane L holds matrix row (L%16). Its 64 nibbles are
 *
 *       nibbles  0..31  ->  k =  0..31   (h=0)  or  k = 32..63  (h=1)
 *       nibbles 32..63  ->  k = 64..95   (h=0)  or  k = 96..127 (h=1)
 *
 *   So the two lane-halves split K into interleaved 32-wide blocks: lanes
 *   0..15 supply k in [0,32)u[64,96), lanes 16..31 supply k in [32,64)u[96,128).
 *   Confirmed by probe: A in lane 0 crossed with B in lane 16 produces exactly
 *   zero output, because their K ranges are disjoint.
 *
 *   *** ONLY 32 OF THE 64 BYTES PER LANE ARE LIVE FOR FP4. ***
 *   The live-byte scan found bytes 0..31 consumed and 32..63 ignored, which
 *   makes sense: 16 rows x 128 k x 4 bit = 1024 B, over 32 lanes = 32 B/lane.
 *   The i32x16 operand type is sized for the FP8 case. This matters because
 *   it means the MX path costs the SAME 8 VGPRs per operand as gfx950, not the
 *   16 the type suggests -- the register-pressure worry in mx_wmma_mi450.cuh's
 *   header was wrong, and is corrected there.
 *
 * Scales (one E8M0 byte per 32-element K block):
 *
 *     scale_a for row m  ->  lane m (lanes 0..15 ONLY), byte (k/32)
 *     scale_b for col n  ->  lane n (lanes 0..15 ONLY), byte (k/32)
 *
 *   Lanes 16..31 of the scale operands are unused, which the guide states
 *   outright ("SCL_OPSEL[0]: 0 = use lanes 0..15; 1 = use lanes 16..31") and
 *   the probe confirmed: scaling lane 16 changed nothing.
 *
 * ---------------------------------------------------------------------------
 * TWO TRAPS, both of which a reasonable port would fall into
 * ---------------------------------------------------------------------------
 *
 * 1. THE SCALE MUST BE IN A VGPR, NOT AN SGPR.
 *    If the scale operand is wave-uniform the compiler puts it in an SGPR, and
 *    per the guide "when a source is an SGPR, only bits [7:0] are used". The
 *    upper three scale bytes are then silently dropped and every K block gets
 *    block 0's scale. The first probe hit exactly this: bytes 1..3 appeared
 *    to have no effect at all, which looked like a hardware quirk and was
 *    actually a register-class artifact. Index the scale by lane (as the
 *    helpers below do) so it stays a VGPR.
 *
 * 2. scale_sel IS *NOT* A BLOCK SELECTOR.
 *    It is SCL_OPSEL: 0 = take scales from lanes 0..15, 1 = from lanes 16..31.
 *    Fleet's mi300 code passes 0 for "block 0" and that happens to be right
 *    for the wrong reason. Passing 1 does not select the second K block -- it
 *    reads scales out of the unused upper lanes, which is measurably wrong
 *    (probe: sel=1 dropped the scaling entirely). Always 0 here.
 *
 * The K interleave in particular means an FP4 weight tile CANNOT be copied
 * straight from a row-major [row][k/2] buffer into the operand register the
 * way the gfx950 code does with a single i32x8 load at `g * 16`. Use
 * mx_pack_a_offsets() below, or repack in LDS.
 */
#pragma once

#include "mirage/persistent_kernel/arch_traits.cuh"
#include "mirage/persistent_kernel/tasks/mi450/mx_wmma_mi450.cuh"

namespace kernel {
namespace mi450 {

#if defined(MIRAGE_ARCH_GFX1250)

// --- Accumulator addressing -------------------------------------------------

// Which lane holds output element C[m][n].
__device__ __host__ __forceinline__ constexpr int mx_acc_lane(int m, int n) {
  return 16 * (m / 8) + n;
}

// Which of the 8 accumulator slots in that lane holds C[m][n].
__device__ __host__ __forceinline__ constexpr int mx_acc_slot(int m) {
  return m % 8;
}

// Inverse: the output element this lane's slot `i` holds. Every lane owns 8
// elements of a single column, so the epilogue writes a column strip.
__device__ __forceinline__ int mx_acc_row(int lane, int slot) {
  return 8 * (lane / 16) + slot;
}

__device__ __forceinline__ int mx_acc_col(int lane) {
  return lane & 15;
}

// --- Operand addressing -----------------------------------------------------

// The matrix row (for A) or column (for B) this lane supplies.
__device__ __forceinline__ int mx_operand_mn(int lane) {
  return lane & 15;
}

// Which of the two interleaved K half-sets this lane supplies: h=0 covers
// k in [0,32) u [64,96); h=1 covers k in [32,64) u [96,128).
__device__ __forceinline__ int mx_operand_half(int lane) {
  return lane >> 4;
}

// Does lane `lane` carry K element k at all?
__device__ __host__ __forceinline__ constexpr bool mx_lane_has_k(int lane,
                                                                 int k) {
  return (lane >> 4) == ((k % 64) / 32);
}

// Position of K element k within this lane's 64-nibble operand record.
// Valid only when mx_lane_has_k(lane, k).
__device__ __host__ __forceinline__ constexpr int mx_nibble_of_k(int k) {
  return 32 * (k / 64) + (k % 32);
}

// Only the low 32 bytes (= 64 nibbles) of the i32x16 operand are consumed for
// FP4, so a packer only ever needs to fill 8 dwords.
constexpr int MX_FP4_LIVE_DWORDS = 8;
constexpr int MX_FP4_LIVE_BYTES = 32;
constexpr int MX_K_PER_TILE = 128;
constexpr int MX_K_PER_SCALE_BLOCK = 32;
constexpr int MX_SCALES_PER_TILE = MX_K_PER_TILE / MX_K_PER_SCALE_BLOCK; // 4

// --- Packing ----------------------------------------------------------------
//
// Byte offsets, within a row-major FP4 buffer of shape [rows][K/2], of the 4
// 8-byte chunks this lane needs. The lane's operand record is built from four
// 64-bit loads rather than one contiguous 32-byte load, because of the K
// interleave described above.
//
// Chunk c (0..3) covers 16 consecutive K elements:
//     c=0 -> k base = h*32 +  0      lands at nibbles  0..15
//     c=1 -> k base = h*32 + 16      lands at nibbles 16..31
//     c=2 -> k base = h*32 + 64      lands at nibbles 32..47
//     c=3 -> k base = h*32 + 80      lands at nibbles 48..63
__device__ __forceinline__ int mx_src_k_base(int lane, int chunk) {
  int h = lane >> 4;
  return h * 32 + (chunk & 1) * 16 + (chunk >> 1) * 64;
}

// Load this lane's A/B operand from a row-major FP4 buffer `base` of shape
// [rows][k_stride_elems / 2] bytes, for K-tile starting at k0.
//
// Fills only the live low 32 bytes; the upper 32 are left zero, which the
// hardware ignores for FP4.
__device__ __forceinline__ mx_i32x16_t
    mx_load_operand_fp4(uint8_t const *base, int row, int k0, int k_stride) {
  mx_i32x16_t out;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    out[i] = 0;
  }
  int lane = threadIdx.x & 31;
#pragma unroll
  for (int chunk = 0; chunk < 4; ++chunk) {
    // 16 FP4 values = 8 bytes = 2 dwords.
    int k = k0 + mx_src_k_base(lane, chunk);
    uint8_t const *src = base + (size_t)row * (k_stride / 2) + k / 2;
    int2 v = *(int2 const *)src;
    out[chunk * 2 + 0] = v.x;
    out[chunk * 2 + 1] = v.y;
  }
  return out;
}

// --- Global-address-space variants ------------------------------------------
//
// mx_load_operand_fp4() above takes a generic pointer, which is correct but
// costs performance when the source really is global memory. Fleet passes
// weights in as a `void const *` kernel argument, so the compiler cannot prove
// the address space and emits FLAT_LOAD. Per MI400 guide S4.9.5:
//
//   "Flat instructions are simultaneously issued as a VMEM and LDS
//    instruction. This means that both VMEM and LDS must be available before
//    the instruction can be launched. [...] Global and Scratch instructions
//    are generally faster because they do not require access to the LDS."
//
// Confirmed in the emitted ISA: the MoE K-loop had `flat_load_b128` for the
// weight operand and `ds_load_b128` for the token operand -- the token side
// was inferred correctly because it comes from __shared__, the weight side was
// not. Casting to address_space(1) turns it into `global_load_b128`.
//
// ONLY call this with a pointer that is genuinely in global memory. The cast
// is unchecked; handing it an LDS or scratch address is undefined behaviour,
// not a slow path.
using mx_global_u8 = __attribute__((address_space(1))) uint8_t const;

__device__ __forceinline__ mx_global_u8 *mx_to_global(uint8_t const *p) {
  return (mx_global_u8 *)p;
}

__device__ __forceinline__ mx_i32x16_t mx_load_operand_fp4_global(
    mx_global_u8 *base, int row, int k0, int k_stride) {
  using g_int2 = __attribute__((address_space(1))) int2 const;
  mx_i32x16_t out;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    out[i] = 0;
  }
  int lane = threadIdx.x & 31;
#pragma unroll
  for (int chunk = 0; chunk < 4; ++chunk) {
    int k = k0 + mx_src_k_base(lane, chunk);
    mx_global_u8 *src = base + (size_t)row * (k_stride / 2) + k / 2;
    int2 v = *(g_int2 *)src;
    out[chunk * 2 + 0] = v.x;
    out[chunk * 2 + 1] = v.y;
  }
  return out;
}

// Prefetch the K-tile this lane will need `k_ahead` elements from now.
//
// GLOBAL_PREFETCH_B8, guide S4.9.6: one byte per lane, but it fetches the
// entire containing cacheline and returns nothing to the wave. Scope 0 (WGP)
// "pulls in at all cache levels on miss", which is what we want for weights
// that are about to be consumed by this same wave.
//
// It does not touch LOADcnt, so it needs no wait and cannot be waited on --
// it is a pure hint. A prefetch past the end of the weight buffer would still
// issue a real address translation (the guide notes a UTC fault would be
// reported to the host), so callers must bound-check `k0 + k_ahead`.
__device__ __forceinline__ void
    mx_prefetch_operand_fp4_global(mx_global_u8 *base, int row, int k0,
                                   int k_stride) {
  int lane = threadIdx.x & 31;
  // Chunks 0 and 2 sit 32 B apart in the row; one 128 B cacheline covers the
  // whole 32 B live record and then some, so a single prefetch per lane is
  // enough for the tile.
  int k = k0 + mx_src_k_base(lane, 0);
  mx_global_u8 *src = base + (size_t)row * (k_stride / 2) + k / 2;
  __builtin_amdgcn_global_prefetch(
      (__attribute__((address_space(1))) void const *)src, 0);
}

// Build the per-lane scale operand from 4 consecutive E8M0 bytes.
//
// MUST be called with a lane-varying `scales` pointer so the result lands in a
// VGPR -- see trap #1 in the header comment. Lanes 16..31 are ignored by the
// hardware but are harmless to compute.
__device__ __forceinline__ unsigned int mx_pack_scales(uint8_t const *scales) {
  return (unsigned int)scales[0] | ((unsigned int)scales[1] << 8) |
         ((unsigned int)scales[2] << 16) | ((unsigned int)scales[3] << 24);
}

// Global-address-space form of the above. The four byte loads get merged by
// the backend into a single 32-bit load (verified: the MoE K-loop emits one
// `global_load_b32` for the scale operand, not four `global_load_u8`), so this
// is written as bytes for clarity without costing anything. Keeping it as
// separate bytes also sidesteps an alignment assumption: the scale array is
// indexed by `row * NUM_BLOCKS_32 + kt/32`, whose 4-byte alignment depends on
// REDUCTION_SIZE and is not guaranteed for every shape.
__device__ __forceinline__ unsigned int
    mx_pack_scales_global(mx_global_u8 *scales) {
  return (unsigned int)scales[0] | ((unsigned int)scales[1] << 8) |
         ((unsigned int)scales[2] << 16) | ((unsigned int)scales[3] << 24);
}

// The scale_sel immediate. Always 0: it selects which lane-half supplies
// scales, and the upper half is unused. See trap #2.
constexpr int MX_SCALE_SEL = 0;

// --- FP4 encode -------------------------------------------------------------
//
// gfx950 packs FP4 with v_cvt_scalef32_pk_fp4_f32, which gfx1250 does not
// have (it has the pk8 *decode*, v_cvt_scale_pk8_bf16_fp4, but no matching
// scaled encode). So the encode is done explicitly.
//
// E2M1 codes and magnitudes: 0, 0.5, 1, 1.5, 2, 3, 4, 6; sign in bit 3.
//
// Rounding is round-half-to-even on the *code*, which for this format is the
// same thing as even-mantissa. The seven midpoints and where each one lands:
//
//     0.25 -> 0    0.75 -> 2    1.25 -> 2    1.75 -> 4
//     2.5  -> 4    3.5  -> 6    5.0  -> 6
//
// so each code's interval is closed on the side facing an even code and open
// on the side facing an odd one. Codes 0, 2, 4, 6 therefore own both of their
// adjacent midpoints and codes 1, 3, 5 own neither.
//
// E2M1 has no infinity, so magnitudes above 6.0 saturate rather than
// overflow. Callers scale by the block's E8M0 reciprocal first, which is what
// keeps values in range.
__device__ __forceinline__ uint8_t mx_encode_e2m1(float v) {
  uint8_t sign = __builtin_signbitf(v) ? 0x8 : 0x0;
  float a = fabsf(v);

  uint8_t mag;
  if (a <= 0.25f) {
    mag = 0; // 0.0
  } else if (a < 0.75f) {
    mag = 1; // 0.5
  } else if (a <= 1.25f) {
    mag = 2; // 1.0
  } else if (a < 1.75f) {
    mag = 3; // 1.5
  } else if (a <= 2.5f) {
    mag = 4; // 2.0
  } else if (a < 3.5f) {
    mag = 5; // 3.0
  } else if (a <= 5.0f) {
    mag = 6; // 4.0
  } else {
    mag = 7; // 6.0, saturating
  }
  return (uint8_t)(sign | mag);
}

#endif // MIRAGE_ARCH_GFX1250

} // namespace mi450
} // namespace kernel
