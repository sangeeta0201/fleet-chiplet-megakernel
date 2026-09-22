/* gfx1250 MXFP4 inner GEMM loop, factored out of the MX kernels.
 *
 * WHY THIS FILE EXISTS
 *
 * gang_rmsnorm_linear_mxfp4_bias_mi300.cuh (2962 lines) and
 * gang_moe_fused_mxfp4_mi300.cuh (2372 lines) are large mostly because they
 * contain EIGHT near-duplicate hand-scheduled MFMA pipelines -- 41
 * _gang_mfma_f4xf8 call sites between them. The variants differ in software
 * pipeline depth (1, 4, or 8 slots), in whether tokens come from LDS or
 * global, and in their epilogues. They do NOT differ in the contraction.
 *
 * Porting all eight line-for-line would mean re-deriving the same wave64->
 * wave32 argument eight times and getting eight chances to get it wrong. So
 * the contraction is extracted here once, and each pipeline becomes a call
 * plus its own epilogue. The epilogues genuinely differ (bias, SwiGLU, top-k
 * scaling, residual) and are left in place.
 *
 * WHAT THE gfx950 PIPELINE DEPTHS WERE FOR, AND WHY THEY DO NOT CARRY OVER
 *
 * The depth-4 and depth-8 variants exist because gfx950 MFMA writes AGPRs and
 * the compiler would otherwise serialize load -> MFMA. Unrolling by hand into
 * N register slots let independent MFMAs issue back to back.
 *
 * gfx1250 has no AGPRs, so the accumulator lives in ordinary VGPRs, and the
 * register budget that funded a depth-8 rotation is not there. The measured
 * result on this target (documented at length in
 * gang_moe_linear_mxfp4_mi450.cuh) is that depth-2 ping-pong is the right
 * shape: deeper rotations make the compiler emit ~26 v_dual_mov_b32 per
 * iteration shuffling buffers, which costs more VALU than the latency it
 * hides. So all eight call sites collapse onto ONE depth-2 loop here. That is
 * a deliberate behavioural change from the gfx950 code, not an oversight.
 *
 * TWO-BUFFER PING-PONG, NOT A ROTATION: each buffer's role is constant within
 * the loop body (buffer 0 is always consumed first, buffer 1 second), which is
 * what keeps the compiler from emitting copies. Stepping two K-tiles per trip
 * is what makes that possible.
 */
#pragma once

#include "mirage/persistent_kernel/tasks/mi450/mx_layout_mi450.cuh"
#include "mirage/persistent_kernel/tasks/mi450/mx_wmma_mi450.cuh"

namespace kernel {
namespace mi450 {

#if defined(MIRAGE_ARCH_GFX1250)

// FP4 weights (global) x FP8 E4M3 tokens (LDS), accumulating over the full
// REDUCTION_SIZE K extent for one 16x16 output tile.
//
//   g_data    : MXFP4 weight nibbles, row-major [rows][REDUCTION_SIZE/2]
//   g_scales  : E8M0 weight scales,   row-major [rows][REDUCTION_SIZE/32]
//   w_row     : the weight row THIS LANE SUPPLIES. Not the row whose result
//               this lane's accumulator ends up holding -- WMMA is
//               cooperative. Callers must use mx_acc_row() in the epilogue.
//               Conflating the two yields a plausible-looking transpose.
//   s_tok_fp8 : FP8 tokens in LDS, [REDUCTION_SIZE] bytes for the tile's token
//   s_tok_scl : E8M0 token scales in LDS, [REDUCTION_SIZE/32] bytes
//   acc       : accumulator, added to (not overwritten), so a caller can chain
//               K-splits across calls.
//
// The token operand uses mx_load_operand_fp8, whose K order is NOT the FP4
// order -- that was measured, and the natural assumption was wrong. See the
// FP8 section of mx_layout_mi450.cuh.
template <int REDUCTION_SIZE>
__device__ __forceinline__ mx_f32x8_t
    mx_k_loop_f4xf8(mx_global_u8 *g_data,
                    mx_global_u8 *g_scales,
                    int w_row,
                    uint8_t const *s_tok_fp8,
                    uint8_t const *s_tok_scl,
                    mx_f32x8_t acc) {
  static_assert(REDUCTION_SIZE % MX_K_PER_TILE == 0,
                "REDUCTION_SIZE must be a multiple of 128 for FP4 WMMA");
  constexpr int NUM_BLOCKS_32 = REDUCTION_SIZE / MX_K_PER_SCALE_BLOCK;
  constexpr int K_ITERS = REDUCTION_SIZE / MX_K_PER_TILE;
  static_assert(K_ITERS >= 2, "pipelined K-loop assumes at least 2 K-tiles");

  mx_global_u8 *row_scales = g_scales + (size_t)w_row * NUM_BLOCKS_32;

  mx_i32x16_t a0, a1, b0, b1;
  unsigned int sa0, sa1, sb0, sb1;

  // Prologue: tile 0 in flight.
  a0 = mx_load_operand_fp4_global(g_data, w_row, 0, REDUCTION_SIZE);
  b0 = mx_load_operand_fp8(s_tok_fp8, 0, 0, REDUCTION_SIZE);
  sa0 = mx_pack_scales_global(row_scales);
  sb0 = mx_pack_scales(s_tok_scl);

  // The token B operand is a single row (row 0): these kernels contract one
  // token against many weight rows, so every lane reads the same token but a
  // different weight row. mx_load_operand_fp8 still varies by lane, because
  // the lane half selects which 16 K of each 32-wide chunk it carries.

#pragma unroll 1
  for (int kt = 0; kt + 2 * MX_K_PER_TILE <= REDUCTION_SIZE;
       kt += 2 * MX_K_PER_TILE) {
    int const k1 = kt + MX_K_PER_TILE;
    int const k2 = kt + 2 * MX_K_PER_TILE;

    // Issue tile k1 while tile kt is still only in flight.
    a1 = mx_load_operand_fp4_global(g_data, w_row, k1, REDUCTION_SIZE);
    b1 = mx_load_operand_fp8(s_tok_fp8, 0, k1, REDUCTION_SIZE);
    sa1 = mx_pack_scales_global(row_scales + k1 / MX_K_PER_SCALE_BLOCK);
    sb1 = mx_pack_scales(s_tok_scl + k1 / MX_K_PER_SCALE_BLOCK);

    // Pull tile k2 toward the WGP cache. GLOBAL_PREFETCH_B8 returns no data
    // and touches no counter, so it needs no wait. Bounded: a prefetch past
    // the end of the weight buffer still issues a real address translation,
    // and a UTC fault on it is reported to the host (guide S4.9.6).
    if (k2 < REDUCTION_SIZE) {
      mx_prefetch_operand_fp4_global(g_data, w_row, k2, REDUCTION_SIZE);
    }

    acc = _gang_wmma_f4xf8(a0, b0, acc, (int)sa0, (int)sb0);

    // Refill buffer 0 with tile k2, consumed on the next trip.
    if (k2 < REDUCTION_SIZE) {
      a0 = mx_load_operand_fp4_global(g_data, w_row, k2, REDUCTION_SIZE);
      b0 = mx_load_operand_fp8(s_tok_fp8, 0, k2, REDUCTION_SIZE);
      sa0 = mx_pack_scales_global(row_scales + k2 / MX_K_PER_SCALE_BLOCK);
      sb0 = mx_pack_scales(s_tok_scl + k2 / MX_K_PER_SCALE_BLOCK);
    }

    acc = _gang_wmma_f4xf8(a1, b1, acc, (int)sa1, (int)sb1);
  }

  // Odd tail. K_ITERS is even for every shape Fleet currently registers
  // (GPT-OSS pads hidden 2880 -> 3072 = 24 tiles), but the kernel only
  // requires REDUCTION_SIZE % 128 == 0, so an odd count is legal and must not
  // silently drop its last tile.
  if constexpr (K_ITERS % 2 == 1) {
    acc = _gang_wmma_f4xf8(a0, b0, acc, (int)sa0, (int)sb0);
  }
  return acc;
}

#endif // MIRAGE_ARCH_GFX1250

} // namespace mi450
} // namespace kernel
