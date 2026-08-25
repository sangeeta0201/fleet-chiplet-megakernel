/* Hand-tuned WMMA GEMM for MI450 (gfx1250).
 *
 * Port of tasks/mi300/gemm_handtuned_mi300.cuh, which targets gfx950 and is
 * built on v_mfma_f32_16x16x16bf16_1k over a 64-lane wave. gfx1250 is a
 * wave32 part with no MFMA at all -- the CDNA builtin is rejected outright
 * ("needs target feature mai-insts") -- so the inner loop moves to
 * v_wmma_f32_16x16x32_bf16 and the data distribution changes with it.
 *
 * What changes, and why the loop body is not a line-for-line translation:
 *
 *   wave width      64 -> 32. The gfx950 kernel assigns one 16x16 output tile
 *                   per wave and relies on lanes 0-63 covering (row, k-quad)
 *                   pairs. At 32 lanes that mapping no longer tiles the
 *                   operand, so the per-lane fragment layout is rebuilt below.
 *
 *   K per matrix    16 -> 32. One WMMA consumes twice the K depth of one MFMA,
 *   instruction     so the K loop takes half as many steps for the same
 *                   REDUCTION_SIZE. K_STEP is derived rather than hardcoded to
 *                   keep the two backends' loop bounds in sync.
 *
 *   fragments       MFMA: A/B are v4s (4 bf16/lane), C is v4f.
 *                   WMMA: A/B are 16 bf16/lane, D is 8 f32/lane. Per lane that
 *                   is 4x the operand data and 2x the accumulator, which is
 *                   the main register-pressure difference to watch.
 *
 *   operands        The gfx1250 builtin takes explicit modifier immediates
 *                   (A_mod, B_mod, C_mod) and two reuse flags around the
 *                   matrices:
 *                     wmma(A_neg, A, B_neg, B, C_mod, C, reuse_a, reuse_b)
 *                   The reuse hints are power optimizations; they are left
 *                   false here and are a tuning knob once this is correct.
 *
 * STATUS: validated numerically under FFM-Lite (mi450 topology, gfx1250) via
 * tests/mi450/test_gemm_wmma.hip -- all 1024 outputs match a host fp32
 * reference exactly, with the output buffer pre-poisoned to prove every
 * element was actually written. FFM's own counters corroborate the dispatch:
 * insts_waves=4 and insts_valu_xdlmacc=64, i.e. 16 K-steps x 4 waves of WMMA.
 * The per-lane fragment mapping below is therefore correct, not merely
 * plausible. Run it with ./run_ffm.sh from the repo root.
 *
 * What this does NOT establish: FFM is a functional model. It models no
 * cycles, bandwidth, or contention, so nothing here is evidence about
 * performance -- only about correctness.
 */
#pragma once

#include "mirage/persistent_kernel/arch_traits.cuh"

namespace kernel {
namespace mi450 {

#if defined(MIRAGE_ARCH_GFX1250)

typedef __bf16 wmma_ab_t __attribute__((ext_vector_type(16))); // A/B fragment
typedef float wmma_acc_t __attribute__((ext_vector_type(8)));  // D/C fragment

// WMMA 16x16x32 on a 32-lane wave.
//
// Operand fragments (A and B are symmetric; the instruction computes
// D[m][n] = sum_k A[m][k] * B[n][k], i.e. B is consumed K-major exactly as the
// MFMA path already stores it, so the weight layout in memory does not change):
//
//   lane l in [0,32) holds row (l % 16) of the tile.
//   The half selector (l / 16) picks which 16 of the 32 K values:
//     k_base = (l / 16) * 16, and the lane's 16 elements are
//     A[l % 16][k_base + 0 .. k_base + 15].
//
// Accumulator fragment (16x16 f32 = 256 values over 32 lanes = 8 per lane):
//   lane l holds column (l % 16), rows (l / 16) * 8 + [0..7].
//
// Both mappings are the documented wave32 layout, restated here because the
// store loop below depends on the accumulator half of it being exact.
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 32;

// M=BATCH_SIZE (1-16), N<=64, K=REDUCTION_SIZE. 128 threads = 4 waves of 32,
// matching the gfx950 kernel's 4 output tiles per block while halving the
// block's thread count -- the tile decomposition is preserved, the thread
// budget is not, because a wave is now half as wide.
template <int BATCH_SIZE, int REDUCTION_SIZE>
__device__ __forceinline__ void
    gemm_handtuned_wmma(void const *__restrict__ input_ptr,
                        void const *__restrict__ weight_ptr,
                        void const *__restrict__ residual_ptr,
                        void *__restrict__ output_ptr,
                        int num_active_tokens,
                        bool residual_add,
                        int output_size,
                        int n_offset,
                        int o_stride) {
  static_assert(REDUCTION_SIZE % WMMA_K == 0,
                "REDUCTION_SIZE must be a multiple of the WMMA K depth (32); "
                "the gfx950 path only required a multiple of 16");
  static_assert(BATCH_SIZE <= WMMA_M, "BATCH_SIZE must fit the 16-row tile");

  __bf16 const *A = (__bf16 const *)input_ptr;
  __bf16 const *B = (__bf16 const *)weight_ptr;
  __bf16 const *R = (__bf16 const *)residual_ptr;
  __bf16 *O = (__bf16 *)output_ptr;

  int const tid = threadIdx.x;
  int const wave_id = tid / mirage::arch::WAVE_SIZE; // 0..3
  int const lane = tid % mirage::arch::WAVE_SIZE;    // 0..31

  int const frag_row = lane % 16;   // operand row this lane supplies
  int const frag_half = lane / 16;  // which 16 of the 32 K values
  int const k_base = frag_half * 16;

  // This wave owns output columns [wave_id*16, wave_id*16+16).
  int const n_tile_base = n_offset + wave_id * WMMA_N;

  wmma_acc_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  constexpr int K_STEPS = REDUCTION_SIZE / WMMA_K;
  for (int ks = 0; ks < K_STEPS; ++ks) {
    int const k_offset = ks * WMMA_K + k_base;

    // A fragment: row frag_row of the activation tile, 16 contiguous K values.
    // Rows past the live batch contribute zero rather than reading out of
    // bounds -- same guard as the gfx950 kernel, just at fragment width.
    wmma_ab_t a_frag;
    if (frag_row < BATCH_SIZE && frag_row < num_active_tokens) {
      __bf16 const *a_src = A + frag_row * REDUCTION_SIZE + k_offset;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        a_frag[i] = a_src[i];
      }
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        a_frag[i] = (__bf16)0.f;
      }
    }

    // B fragment: weight row (output column) frag_row of this wave's tile.
    wmma_ab_t b_frag;
    int const b_row = n_tile_base + frag_row;
    if (b_row < output_size) {
      __bf16 const *b_src = B + b_row * REDUCTION_SIZE + k_offset;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        b_frag[i] = b_src[i];
      }
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        b_frag[i] = (__bf16)0.f;
      }
    }

    // acc += A x B^T. Modifier immediates are all "none"; the trailing pair is
    // (reuse_a, reuse_b), left off until the layout is validated.
    acc = __builtin_amdgcn_wmma_f32_16x16x32_bf16(
        false, a_frag, false, b_frag, 0, acc, false, false);
  }

  // Store: lane l holds column (l % 16), rows (l / 16) * 8 + [0..7].
  int const out_col = n_tile_base + (lane % 16);
  int const row_base = (lane / 16) * 8;

  if (out_col < output_size) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int const out_row = row_base + i;
      if (out_row < BATCH_SIZE && out_row < num_active_tokens) {
        int const idx = out_row * o_stride + out_col;
        float v = acc[i];
        if (residual_add && R != nullptr) {
          v += (float)R[idx];
        }
        O[idx] = (__bf16)v;
      }
    }
  }
}

#endif // MIRAGE_ARCH_GFX1250

} // namespace mi450
} // namespace kernel
