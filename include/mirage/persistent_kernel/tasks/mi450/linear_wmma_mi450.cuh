/* Linear (GEMM) for gfx1250 (MI450), written on WMMA.
 *
 * This replaces linear_kernel_ck from tasks/mi300/linear_ck_mi300.cuh, which
 * routes through CK's BlockUniversalGemmAsBsCr pipeline. CK cannot target
 * gfx1250 at all: its WMMA path goes through __builtin_amdgcn_wmma_*_w32_gfx12,
 * which requires the wmma-128b-insts target feature, and gfx1250 does not have
 * it. gfx1250's WMMA builtins are a disjoint set with different shapes. So this
 * is a rewrite, not a port of CK's tile configuration.
 *
 * WHY THIS FILE MATTERS BEYOND "linear": it is on the critical path of three
 * separate phases. gang_rmsnorm_linear_bias -> gang_linear_kernel ->
 * linear_kernel_ck, and both MX orchestrators end in a dense projection. Every
 * one of those was blocked on CK. Matching linear_kernel_ck's signature exactly
 * means the callers need a one-line alias rather than a rewrite.
 *
 * ── Signature compatibility ──
 *
 * The parameter list is identical to linear_kernel_ck, including the unused
 * FORCE_SMALL_TILE (CK used it to pick among four MFMA tile tiers; there is
 * one tiling here, so it is accepted and ignored rather than removed, so the
 * call sites stay byte-identical).
 *
 * ── Tiling ──
 *
 * A task block is 256 threads = 8 waves of 32 (WORKER_NUM_THREADS, which is
 * 4 wave64s on gfx950 -- the wave count doubles and the tiling is re-derived,
 * not rescaled). Each wave owns one 16-wide column strip of the output, so a
 * block covers N=128 columns per pass and strides by 128 until output_size is
 * consumed. M is tiled by 16 (the WMMA M), so a BATCH_SIZE above 16 loops.
 *
 * Fragment layout is the one validated in gemm_handtuned_mi450.cuh and reused
 * by the attention kernel:
 *   operand: lane l holds row (l%16), K elements (l/16)*16 + [0..15]
 *   accum:   lane l holds column (l%16), rows (l/16)*8 + [0..7]
 * and the instruction computes D[m][n] = sum_k A[m][k] * B[n][k], i.e. B is
 * consumed K-major -- which is exactly how the weight is already stored
 * ([N][K] row-major), so no weight relayout is needed anywhere in the port.
 */
#pragma once

#include "mirage/persistent_kernel/arch_traits.cuh"

namespace kernel {
namespace mi450 {

#if defined(MIRAGE_ARCH_GFX1250)

typedef __bf16 linear_ab_t __attribute__((ext_vector_type(16)));
typedef float linear_acc_t __attribute__((ext_vector_type(8)));

static constexpr int LIN_WMMA_M = 16;
static constexpr int LIN_WMMA_N = 16;
static constexpr int LIN_WMMA_K = 32;

// Dense linear layer: output[m][n] = sum_k input[m][k] * weight[n][k] + bias[n]
//
//   input   [BATCH_SIZE, REDUCTION_SIZE]  row-major
//   weight  [output_size, REDUCTION_SIZE] row-major (N-major, K contiguous)
//   output  [BATCH_SIZE, o_stride]        row-major, only output_size columns
//   bias    [output_size]                 optional, broadcast over rows
//
// FORCE_SMALL_TILE is accepted for signature parity with linear_kernel_ck and
// deliberately unused.
template <typename T,
          int BATCH_SIZE,
          int REDUCTION_SIZE,
          bool FORCE_SMALL_TILE = false>
__device__ __forceinline__ void
    linear_kernel_wmma(void const *input_ptr,
                       void const *weight_ptr,
                       void const *residual_ptr,
                       void *output_ptr,
                       int num_active_tokens,
                       bool residual_add,
                       int output_size,
                       int o_stride,
                       void const *bias_ptr = nullptr) {
#ifdef MPK_DISABLE_LINEAR
  return;
#endif
  static_assert(REDUCTION_SIZE % LIN_WMMA_K == 0,
                "REDUCTION_SIZE must be a multiple of the WMMA K depth (32). "
                "The gfx950 path only needed a multiple of 16, so a shape that "
                "worked there can fail here -- GPT-OSS keeps K % 128 == 0, so "
                "this holds, but a new shape must be checked.");

  __bf16 const *A = static_cast<__bf16 const *>(input_ptr);
  __bf16 const *B = static_cast<__bf16 const *>(weight_ptr);
  __bf16 const *R = static_cast<__bf16 const *>(residual_ptr);
  __bf16 const *BIAS = static_cast<__bf16 const *>(bias_ptr);
  __bf16 *O = static_cast<__bf16 *>(output_ptr);

  int const tid = threadIdx.x;
  int const wave_id = tid / mirage::arch::WAVE_SIZE;
  int const lane = tid % mirage::arch::WAVE_SIZE;
  int const num_waves = mirage::arch::waves_in((int)blockDim.x);

  int const frag_row = lane % 16;  // operand row / accumulator column
  int const frag_half = lane / 16; // which 16 of the 32 K values
  int const k_base = frag_half * 16;

  constexpr int K_STEPS = REDUCTION_SIZE / LIN_WMMA_K;
  constexpr int M_TILES = (BATCH_SIZE + LIN_WMMA_M - 1) / LIN_WMMA_M;

  // Rows this lane writes back (accumulator distributes by row).
  int const out_row_base = (lane / 16) * 8;

  for (int m_tile = 0; m_tile < M_TILES; ++m_tile) {
    int const m_base = m_tile * LIN_WMMA_M;

    // Each wave takes one 16-column strip; the block strides by
    // num_waves * 16 columns until output_size is covered.
    //
    // num_waves must not be hardcoded, but note the failure mode is a *silent
    // 2x* rather than wrong output: with the gfx950 literal 4 here, waves 0-3
    // still cover every strip (stride 64 over starts 0/16/32/48) and waves 4-7
    // merely rewrite the same values. Mutation-tested -- that mutant passes all
    // 8 shapes, and no correctness test can catch it, under FFM or on silicon.
    // Hardcoding wave_id's divisor instead (tid/64) *is* caught: strips 4-7 go
    // unclaimed and the poison check reports "never written".
    for (int n_base = wave_id * LIN_WMMA_N; n_base < output_size;
         n_base += num_waves * LIN_WMMA_N) {

      linear_acc_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

      // A row and B row this lane supplies. Rows past the live batch or past
      // the real output width contribute zero rather than reading out of
      // bounds -- the gfx950 path got this clamp from the buffer descriptor's
      // num_records, which gfx1250 has no equivalent of, so it is explicit.
      int const a_row = m_base + frag_row;
      bool const a_live = (a_row < BATCH_SIZE) && (a_row < num_active_tokens);
      int const b_row = n_base + frag_row;
      bool const b_live = b_row < output_size;

      __bf16 const *a_src =
          a_live ? (A + (size_t)a_row * REDUCTION_SIZE + k_base) : nullptr;
      __bf16 const *b_src =
          b_live ? (B + (size_t)b_row * REDUCTION_SIZE + k_base) : nullptr;

      for (int ks = 0; ks < K_STEPS; ++ks) {
        int const k_off = ks * LIN_WMMA_K;

        linear_ab_t a_frag;
        if (a_live) {
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            a_frag[i] = a_src[k_off + i];
          }
        } else {
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            a_frag[i] = (__bf16)0.f;
          }
        }

        linear_ab_t b_frag;
        if (b_live) {
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            b_frag[i] = b_src[k_off + i];
          }
        } else {
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            b_frag[i] = (__bf16)0.f;
          }
        }

        acc = __builtin_amdgcn_wmma_f32_16x16x32_bf16(
            false, a_frag, false, b_frag, 0, acc, false, false);
      }

      // Store. Lane l holds column (l%16), rows (l/16)*8 + [0..7].
      int const out_col = n_base + frag_row;
      if (out_col < output_size) {
        float const bias_v = (BIAS != nullptr) ? (float)BIAS[out_col] : 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          int const out_row = m_base + out_row_base + i;
          if (out_row < BATCH_SIZE && out_row < num_active_tokens) {
            size_t const idx = (size_t)out_row * o_stride + out_col;
            float v = acc[i] + bias_v;
            if (residual_add && R != nullptr) {
              v += (float)R[idx];
            }
            O[idx] = (__bf16)v;
          }
        }
      }
    }
  }
}

#else // !MIRAGE_ARCH_GFX1250

// Host-pass / non-gfx1250 declaration.
//
// MIRAGE_ARCH_GFX1250 comes from __gfx1250__, which is only defined in the
// device compilation pass. Callers that name this template from a shared header
// (gang_linear_mi300.cuh does) are still parsed in the host pass, so without
// this the host pass fails with "no member named linear_kernel_wmma". Providing
// an empty body rather than making every caller wrap its call in the arch macro
// keeps the call sites readable -- HIP does not codegen __device__ bodies in the
// host pass, so this never executes.
template <typename T,
          int BATCH_SIZE,
          int REDUCTION_SIZE,
          bool FORCE_SMALL_TILE = false>
__device__ __forceinline__ void linear_kernel_wmma(void const *,
                                                   void const *,
                                                   void const *,
                                                   void *,
                                                   int,
                                                   bool,
                                                   int,
                                                   int,
                                                   void const * = nullptr) {
}

#endif // MIRAGE_ARCH_GFX1250

} // namespace mi450
} // namespace kernel
