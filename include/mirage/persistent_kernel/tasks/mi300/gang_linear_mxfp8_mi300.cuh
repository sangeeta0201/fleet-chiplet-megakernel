/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Gang dense linear kernel using MXFP8 weights with FP8xFP8 MFMA (gfx950).
//
// This is gang_linear_mxfp4_mi300.cuh with the weight operand widened from 4
// bits to 8. It exists because GLM-4.7-Flash ships bf16 and we quantize at
// load time, which leaves the format free -- and at 47 layers of MoE, bf16 is
// the thing that binds. The token moves ~9.2 GB of weights at an effective
// ~1.22 TB/s, which is already the bandwidth gpt-oss achieves; the remaining
// gap to its 2.0 ms is data volume, not scheduling. MXFP8 halves the volume to
// ~1.03 bytes per weight (one byte plus one E8M0 exponent per 32) while
// keeping a full E4M3 mantissa, which is a far softer accuracy landing than
// E2M1 for a model that was never quantization-aware trained.
//
// Three differences from the FP4 kernel, all of them mechanical:
//
//   row stride   OPW*K bytes of data instead of OPW*(K/2)
//   weight load  the split gather (_gang_load_fp8_mfma_b) instead of one
//                contiguous 16-byte read, because at 8 bits a lane's 32
//                elements are no longer 16 consecutive bytes
//   cbsz         0 (FP8 E4M3 src0) instead of 4 (FP4 E2M1 src0)
//
// The scale indexing is byte-for-byte the same as FP4's, and that is not a
// coincidence worth trusting on inspection -- see
// tests/standalone/test_mxfp8_mfma_layout.hip. The MFMA's scale operand is
// addressed by matrix position, so lane 16*g+m carries row m's exponent for
// the *contiguous* K block [g*32, g*32+32), regardless of which bytes that
// lane's data register happens to hold. At 4 bits those two groupings coincide
// and the distinction is invisible; at 8 bits they diverge, and grouping the
// quantizer the way the lane reads costs 42% relative error.
//
// Weight format, per workgroup of OPW=64 output rows:
//   [FP8 E4M3 data: OPW * K bytes][E8M0 scales: OPW * (K/32) bytes]
// Data is K-major within a row. Scales are indexed [row][k/32].

#pragma once
// Same single include as gang_linear_mxfp4_mi300.cuh. Note that header is not
// self-contained: its MoE kernels call _gang_moe_get_xcd_id and
// MPK_WS_WAVE_SYNC, which task_header.cuh supplies from earlier includes.
// Naming those dependencies here would drag CK in through
// gang_moe_linear_mi300.cuh, so the ordering is inherited instead, and the
// standalone test stubs the two symbols.
#include "tasks/mi300/gang_moe_linear_mxfp4_mi300.cuh"

namespace kernel {

// FP8xFP8 scaled MFMA: 16x16x128, hardware dequant + multiply.
// A = weights (FP8 E4M3), 32 bytes/lane. B = tokens (FP8 E4M3), 32 bytes/lane.
// Both operands fill the whole i32x8, unlike the FP4 case where src0 used only
// the lower 128 bits.
__device__ __forceinline__ f32x4_t _gang_mfma_f8xf8(
    i32x8_t a, i32x8_t b, f32x4_t c, int scale_a, int scale_b) {
  return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
      a, b, c, 0 /*cbsz: FP8 E4M3 src0*/, 0 /*blgp: FP8 E4M3 src1*/,
      0, scale_a, 0, scale_b);
}

// Shared prologue for both variants: windowed traversal (HipKittens Alg. 1)
// resolving a flat tile index to (m_tile, n_tile). Returns false when the tile
// falls outside a short final window.
__device__ __forceinline__ bool _gang_mxfp8_tile_coords(
    int tile_idx, int m_tiles, int n_tiles, int wgm, int *m_tile, int *n_tile) {
  int W = (wgm > 0 && wgm < m_tiles) ? wgm : m_tiles;
  int tid_per_group = W * n_tiles;
  int group_id = tile_idx / tid_per_group;
  int first_row = group_id * W;
  int win_h = m_tiles - first_row;
  if (win_h > W) {
    win_h = W;
  }
  int local = tile_idx % tid_per_group;
  if (local >= win_h * n_tiles) {
    return false;
  }
  *m_tile = first_row + (local % win_h);
  *n_tile = local / win_h;
  return true;
}

// Dense gang linear with MXFP8 weights.
//
// 256 threads / 4 waves. Each wave handles 16 output rows (N-parallel).
// K=128 per MFMA via __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4.
// Input is quantized on the fly from bf16 to FP8 E4M3 in shared memory.
//
// WRITE_THROUGH sends the epilogue store past the XCD's L2 (sc0 sc1), for the
// same reason gang_gemv_mxfp8_kernel takes the flag: a *fused* caller that
// follows this GEMM with nothing more than an in-kernel barrier needs the
// result visible to the other seven XCDs, and on MI300/MI350 the L2 is per-XCD
// and not coherent, so a plain store is not. A standalone task does not need it
// -- the scheduler's end-of-task fence covers it -- hence the default.
template <int BATCH_SIZE,      // = m_per_tile (rows per M-tile)
          int REDUCTION_SIZE,  // K dimension
          bool WRITE_THROUGH = false,
          // Row stride of input_ptr when the caller hands over a slice of a
          // WIDER row than this GEMM reduces. W_UV is the caller that needs
          // it: input_ptr is one head of attn_out, so the reduction is
          // KV_LORA_RANK but the row is NUM_Q_HEADS * KV_LORA_RANK. At one
          // row the two are indistinguishable, which is why this went
          // unnoticed until BATCH_SIZE > 1.
          int INPUT_ROW_STRIDE = REDUCTION_SIZE>
__device__ __noinline__ void gang_linear_mxfp8_kernel(
    void const *input_ptr,  // [batch, REDUCTION_SIZE] bf16
    void const *weight_ptr, // [1, n_wgs, wg_bytes] MXFP8 packed
    void *output_ptr,       // [batch, output_stride] bf16
    int num_active_tokens,
    int tile_n,        // = output_per_wg (64)
    int output_stride, // full output row stride
    int output_size,   // actual output dim (may be < output_stride)
    int m_tiles,       // total M-tiles
    int n_tiles,       // N-tiles per XCD
    int wgm,           // window height W (Algorithm 1)
    int tile_idx,
    void const *bias_ptr) // [1, output_stride] bf16, optional
{
  static_assert(REDUCTION_SIZE % 128 == 0,
                "K must be a multiple of 128 for FP8 MFMA");

  assert(tile_idx >= 0);

  int m_tile, n_tile;
  if (!_gang_mxfp8_tile_coords(
          tile_idx, m_tiles, n_tiles, wgm, &m_tile, &n_tile)) {
    return;
  }

  constexpr int OUTPUT_PER_WG = 64; // 4 waves x 16 rows
  constexpr int NUM_BLOCKS_32 = REDUCTION_SIZE / 32;
  constexpr int WG_DATA_BYTES = OUTPUT_PER_WG * REDUCTION_SIZE;
  constexpr int WG_SCALE_BYTES = OUTPUT_PER_WG * NUM_BLOCKS_32;
  constexpr int WG_BYTES = WG_DATA_BYTES + WG_SCALE_BYTES;

  constexpr int K_PER_MFMA = 128;
  // MPK_ATTN_HALFK: run half the K-loop and keep everything else identical --
  // same tile map, same WG stride, same MFMA shape, half the weight bytes off
  // HBM. WRONG OUTPUT by construction. This prices MXFP4 for the attention /
  // dense weights (task #66) before any quantization plumbing is written:
  // MXFP4 halves bytes at constant FLOPs, HALFK halves both, so HALFK is an
  // UPPER BOUND on the MXFP4 gain. If the bound is small the lever is dead.
  // Clamped to the depth-4 pipeline's minimum, so W_UV's K=512 (MFMA_ITERS 4)
  // is left at full width rather than dropped below the static_assert.
  constexpr int MFMA_ITERS_FULL = REDUCTION_SIZE / K_PER_MFMA;
#ifdef MPK_ATTN_HALFK
  // Floor of 16 after halving, not 4: the K_PARALLEL branch splits MFMA_ITERS
  // across NUM_WAVES=4 and its own static_assert needs ITERS_PER_WAVE >= 4.
  // So only stages with MFMA_ITERS >= 32 are halved -- at GLM's shapes that is
  // qkv_a (K=6144, 48 iters -> 24). q_b (16), W_UK and W_UV (K=512, 4) stay at
  // full width, which makes this a LOWER bound on the byte lever as well as an
  // upper bound on the MXFP4 gain for the stage it does reach.
  constexpr int MFMA_ITERS =
      (MFMA_ITERS_FULL / 2 >= 16 && (MFMA_ITERS_FULL / 2) % 4 == 0)
          ? MFMA_ITERS_FULL / 2
          : MFMA_ITERS_FULL;
#else
  constexpr int MFMA_ITERS = MFMA_ITERS_FULL;
#endif
  static_assert(MFMA_ITERS >= 4,
                "Depth-4 pipeline requires REDUCTION_SIZE >= 512");
  // Only slot 3 carries a tail guard, so a partial final group would let
  // slots 1 and 2 compute k-tiles that do not exist. Inherited from the FP4
  // kernel, where every shape is a multiple of 4 as well; made explicit here
  // rather than left to be discovered. GLM's K values -- 1024, 1536, 2048,
  // 10240 -- all satisfy it.
  static_assert(MFMA_ITERS % 4 == 0,
                "Depth-4 pipeline requires REDUCTION_SIZE % 512 == 0");

  constexpr int FP8_TOK_DATA = REDUCTION_SIZE;

  unsigned short const *A = (unsigned short const *)input_ptr;
  uint8_t const *W_data = (uint8_t const *)weight_ptr;
  unsigned short *d_output = (unsigned short *)output_ptr;
  unsigned short const *d_bias = (unsigned short const *)bias_ptr;

  extern __shared__ char _gang_lin_mxfp8_smem[];
  uint8_t *s_tok_fp8 = (uint8_t *)_gang_lin_mxfp8_smem;
  uint8_t *s_tok_scales = s_tok_fp8 + FP8_TOK_DATA;

  int const tid = threadIdx.x;
  int const warp_id = tid >> 6;
  int const lane_id = tid & 63;
  int const col = lane_id & 15; // output row within the 16x16 MFMA tile
  int const g = lane_id >> 4;   // K-group (0..3)

  uint8_t const *wg_data = W_data + static_cast<size_t>(n_tile) * WG_BYTES;
  uint8_t const *wg_scales = wg_data + WG_DATA_BYTES;

  // Rows this M-tile actually carries. BATCH_SIZE is the tile height; at one
  // row the loop below folds to the old straight-line body, which is what
  // keeps the bs == 1 codegen identical.
  int tile_rows = 1;
  if constexpr (BATCH_SIZE > 1) {
    tile_rows = num_active_tokens - m_tile * BATCH_SIZE;
    tile_rows = tile_rows > BATCH_SIZE ? BATCH_SIZE : tile_rows;
    tile_rows = tile_rows < 1 ? 1 : tile_rows;
  }

#pragma unroll 1
  for (int _row = 0; _row < tile_rows; ++_row) {
    unsigned short const *input_base =
        A +
        (static_cast<size_t>(m_tile) * BATCH_SIZE + _row) * INPUT_ROW_STRIDE;

    unsigned short *out_base =
        d_output +
        (static_cast<size_t>(m_tile) * BATCH_SIZE + _row) * output_stride +
        static_cast<size_t>(n_tile) * OUTPUT_PER_WG;

    // Phase 1: quantize bf16 input to FP8 E4M3 in shared memory. Ends in a
    // __syncthreads, so the previous row's readers are fenced off from the
    // rewrite by the one at the bottom of this loop body.
    _gang_wave_parallel_fp8_quant<REDUCTION_SIZE>(
        input_base, s_tok_fp8, s_tok_scales);

    // Phase 2: depth-4 pipelined FP8(weight) x FP8(token) MFMA.
    {
    int wave_tile = warp_id;
    int w_row = wave_tile * 16 + col;
    uint8_t const *w_data_row =
        wg_data + static_cast<size_t>(w_row) * REDUCTION_SIZE;
    int const row_scale_base = w_row * NUM_BLOCKS_32;

    f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

    // Pre-fill: load k-tiles 0..3 into pipeline slots.
    i32x8_t a0 = _gang_load_fp8_mfma_b(w_data_row, 0 * K_PER_MFMA, g);
    int sa0 = (int)wg_scales[row_scale_base + 0 * 4 + g];
    i32x8_t a1 = _gang_load_fp8_mfma_b(w_data_row, 1 * K_PER_MFMA, g);
    int sa1 = (int)wg_scales[row_scale_base + 1 * 4 + g];
    i32x8_t a2 = _gang_load_fp8_mfma_b(w_data_row, 2 * K_PER_MFMA, g);
    int sa2 = (int)wg_scales[row_scale_base + 2 * 4 + g];
    i32x8_t a3 = _gang_load_fp8_mfma_b(w_data_row, 3 * K_PER_MFMA, g);
    int sa3 = (int)wg_scales[row_scale_base + 3 * 4 + g];

// IMPORTANT: #pragma unroll 1 prevents ROCm miscompilation.
#pragma unroll 1
    for (int ki = 0; ki < MFMA_ITERS; ki += 4) {
      // Slot 0: compute k-tile ki, prefetch ki+4
      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, ki * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki];
        acc = _gang_mfma_f8xf8(a0, b, acc, sa0, sb);
      }
      if (ki + 4 < MFMA_ITERS) {
        int kt4 = (ki + 4) * K_PER_MFMA;
        a0 = _gang_load_fp8_mfma_b(w_data_row, kt4, g);
        sa0 = (int)wg_scales[row_scale_base + kt4 / 32 + g];
      }

      // Slot 1: compute k-tile ki+1, prefetch ki+5
      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 1) * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki + 1];
        acc = _gang_mfma_f8xf8(a1, b, acc, sa1, sb);
      }
      if (ki + 5 < MFMA_ITERS) {
        int kt5 = (ki + 5) * K_PER_MFMA;
        a1 = _gang_load_fp8_mfma_b(w_data_row, kt5, g);
        sa1 = (int)wg_scales[row_scale_base + kt5 / 32 + g];
      }

      // Slot 2: compute k-tile ki+2, prefetch ki+6
      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 2) * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki + 2];
        acc = _gang_mfma_f8xf8(a2, b, acc, sa2, sb);
      }
      if (ki + 6 < MFMA_ITERS) {
        int kt6 = (ki + 6) * K_PER_MFMA;
        a2 = _gang_load_fp8_mfma_b(w_data_row, kt6, g);
        sa2 = (int)wg_scales[row_scale_base + kt6 / 32 + g];
      }

      // Slot 3: compute k-tile ki+3, prefetch ki+7
      if (ki + 3 < MFMA_ITERS) {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 3) * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki + 3];
        acc = _gang_mfma_f8xf8(a3, b, acc, sa3, sb);
      }
      if (ki + 7 < MFMA_ITERS) {
        int kt7 = (ki + 7) * K_PER_MFMA;
        a3 = _gang_load_fp8_mfma_b(w_data_row, kt7, g);
        sa3 = (int)wg_scales[row_scale_base + kt7 / 32 + g];
      }
    }

    // Epilogue: acc + optional bias -> output.
    if (col == 0) {
      for (int i = 0; i < 4; i++) {
        int out_n = n_tile * OUTPUT_PER_WG + wave_tile * 16 + g * 4 + i;
        if (out_n < output_size) {
          float sum = acc[i];
          if (d_bias) {
            sum += _gang_bf16_to_float(d_bias[out_n]);
          }
          int out_idx = wave_tile * 16 + g * 4 + i;
          if constexpr (WRITE_THROUGH) {
            st_wt_u16(&out_base[out_idx], _gang_float_to_bf16(sum));
          } else {
            out_base[out_idx] = _gang_float_to_bf16(sum);
          }
        }
      }
    }
    }

    __syncthreads();
  }
}

// Dense gang linear with MXFP8 weights + residual add.
// Same as gang_linear_mxfp8_kernel but the epilogue adds a residual.
template <int BATCH_SIZE, int REDUCTION_SIZE>
__device__ __noinline__ void gang_linear_res_mxfp8_kernel(
    void const *input_ptr,    // [batch, REDUCTION_SIZE] bf16
    void const *weight_ptr,   // [1, n_wgs, wg_bytes] MXFP8 packed
    void const *residual_ptr, // [batch, output_stride] bf16
    void *output_ptr,         // [batch, output_stride] bf16
    int num_active_tokens,
    int tile_n,
    int output_stride,
    int output_size,
    int m_tiles,
    int n_tiles,
    int wgm,
    int tile_idx,
    void const *bias_ptr) {
  static_assert(REDUCTION_SIZE % 128 == 0,
                "K must be a multiple of 128 for FP8 MFMA");

  assert(tile_idx >= 0);

  int m_tile, n_tile;
  if (!_gang_mxfp8_tile_coords(
          tile_idx, m_tiles, n_tiles, wgm, &m_tile, &n_tile)) {
    return;
  }

  constexpr int OUTPUT_PER_WG = 64;
  constexpr int NUM_BLOCKS_32 = REDUCTION_SIZE / 32;
  constexpr int WG_DATA_BYTES = OUTPUT_PER_WG * REDUCTION_SIZE;
  constexpr int WG_SCALE_BYTES = OUTPUT_PER_WG * NUM_BLOCKS_32;
  constexpr int WG_BYTES = WG_DATA_BYTES + WG_SCALE_BYTES;

  constexpr int K_PER_MFMA = 128;
  // MPK_ATTN_HALFK: run half the K-loop and keep everything else identical --
  // same tile map, same WG stride, same MFMA shape, half the weight bytes off
  // HBM. WRONG OUTPUT by construction. This prices MXFP4 for the attention /
  // dense weights (task #66) before any quantization plumbing is written:
  // MXFP4 halves bytes at constant FLOPs, HALFK halves both, so HALFK is an
  // UPPER BOUND on the MXFP4 gain. If the bound is small the lever is dead.
  // Clamped to the depth-4 pipeline's minimum, so W_UV's K=512 (MFMA_ITERS 4)
  // is left at full width rather than dropped below the static_assert.
  constexpr int MFMA_ITERS_FULL = REDUCTION_SIZE / K_PER_MFMA;
#ifdef MPK_ATTN_HALFK
  // Floor of 16 after halving, not 4: the K_PARALLEL branch splits MFMA_ITERS
  // across NUM_WAVES=4 and its own static_assert needs ITERS_PER_WAVE >= 4.
  // So only stages with MFMA_ITERS >= 32 are halved -- at GLM's shapes that is
  // qkv_a (K=6144, 48 iters -> 24). q_b (16), W_UK and W_UV (K=512, 4) stay at
  // full width, which makes this a LOWER bound on the byte lever as well as an
  // upper bound on the MXFP4 gain for the stage it does reach.
  constexpr int MFMA_ITERS =
      (MFMA_ITERS_FULL / 2 >= 16 && (MFMA_ITERS_FULL / 2) % 4 == 0)
          ? MFMA_ITERS_FULL / 2
          : MFMA_ITERS_FULL;
#else
  constexpr int MFMA_ITERS = MFMA_ITERS_FULL;
#endif
  static_assert(MFMA_ITERS >= 4,
                "Depth-4 pipeline requires REDUCTION_SIZE >= 512");
  // Only slot 3 carries a tail guard, so a partial final group would let
  // slots 1 and 2 compute k-tiles that do not exist. Inherited from the FP4
  // kernel, where every shape is a multiple of 4 as well; made explicit here
  // rather than left to be discovered. GLM's K values -- 1024, 1536, 2048,
  // 10240 -- all satisfy it.
  static_assert(MFMA_ITERS % 4 == 0,
                "Depth-4 pipeline requires REDUCTION_SIZE % 512 == 0");

  constexpr int FP8_TOK_DATA = REDUCTION_SIZE;

  unsigned short const *A = (unsigned short const *)input_ptr;
  uint8_t const *W_data = (uint8_t const *)weight_ptr;
  unsigned short const *d_residual = (unsigned short const *)residual_ptr;
  unsigned short *d_output = (unsigned short *)output_ptr;
  unsigned short const *d_bias = (unsigned short const *)bias_ptr;

  extern __shared__ char _gang_lin_mxfp8_smem[];
  uint8_t *s_tok_fp8 = (uint8_t *)_gang_lin_mxfp8_smem;
  uint8_t *s_tok_scales = s_tok_fp8 + FP8_TOK_DATA;

  int const tid = threadIdx.x;
  int const warp_id = tid >> 6;
  int const lane_id = tid & 63;
  int const col = lane_id & 15;
  int const g = lane_id >> 4;

  unsigned short const *input_base =
      A + static_cast<size_t>(m_tile) * BATCH_SIZE * REDUCTION_SIZE;

  uint8_t const *wg_data = W_data + static_cast<size_t>(n_tile) * WG_BYTES;
  uint8_t const *wg_scales = wg_data + WG_DATA_BYTES;

  size_t out_row_off =
      static_cast<size_t>(m_tile) * BATCH_SIZE * output_stride +
      static_cast<size_t>(n_tile) * OUTPUT_PER_WG;
  unsigned short *out_base = d_output + out_row_off;
  unsigned short const *res_base = d_residual + out_row_off;

  _gang_wave_parallel_fp8_quant<REDUCTION_SIZE>(
      input_base, s_tok_fp8, s_tok_scales);

  {
    int wave_tile = warp_id;
    int w_row = wave_tile * 16 + col;
    uint8_t const *w_data_row =
        wg_data + static_cast<size_t>(w_row) * REDUCTION_SIZE;
    int const row_scale_base = w_row * NUM_BLOCKS_32;

    f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

    i32x8_t a0 = _gang_load_fp8_mfma_b(w_data_row, 0 * K_PER_MFMA, g);
    int sa0 = (int)wg_scales[row_scale_base + 0 * 4 + g];
    i32x8_t a1 = _gang_load_fp8_mfma_b(w_data_row, 1 * K_PER_MFMA, g);
    int sa1 = (int)wg_scales[row_scale_base + 1 * 4 + g];
    i32x8_t a2 = _gang_load_fp8_mfma_b(w_data_row, 2 * K_PER_MFMA, g);
    int sa2 = (int)wg_scales[row_scale_base + 2 * 4 + g];
    i32x8_t a3 = _gang_load_fp8_mfma_b(w_data_row, 3 * K_PER_MFMA, g);
    int sa3 = (int)wg_scales[row_scale_base + 3 * 4 + g];

// IMPORTANT: #pragma unroll 1 prevents ROCm miscompilation.
#pragma unroll 1
    for (int ki = 0; ki < MFMA_ITERS; ki += 4) {
      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, ki * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki];
        acc = _gang_mfma_f8xf8(a0, b, acc, sa0, sb);
      }
      if (ki + 4 < MFMA_ITERS) {
        int kt4 = (ki + 4) * K_PER_MFMA;
        a0 = _gang_load_fp8_mfma_b(w_data_row, kt4, g);
        sa0 = (int)wg_scales[row_scale_base + kt4 / 32 + g];
      }

      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 1) * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki + 1];
        acc = _gang_mfma_f8xf8(a1, b, acc, sa1, sb);
      }
      if (ki + 5 < MFMA_ITERS) {
        int kt5 = (ki + 5) * K_PER_MFMA;
        a1 = _gang_load_fp8_mfma_b(w_data_row, kt5, g);
        sa1 = (int)wg_scales[row_scale_base + kt5 / 32 + g];
      }

      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 2) * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki + 2];
        acc = _gang_mfma_f8xf8(a2, b, acc, sa2, sb);
      }
      if (ki + 6 < MFMA_ITERS) {
        int kt6 = (ki + 6) * K_PER_MFMA;
        a2 = _gang_load_fp8_mfma_b(w_data_row, kt6, g);
        sa2 = (int)wg_scales[row_scale_base + kt6 / 32 + g];
      }

      if (ki + 3 < MFMA_ITERS) {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 3) * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki + 3];
        acc = _gang_mfma_f8xf8(a3, b, acc, sa3, sb);
      }
      if (ki + 7 < MFMA_ITERS) {
        int kt7 = (ki + 7) * K_PER_MFMA;
        a3 = _gang_load_fp8_mfma_b(w_data_row, kt7, g);
        sa3 = (int)wg_scales[row_scale_base + kt7 / 32 + g];
      }
    }

    // Epilogue: acc + bias + residual -> output.
    if (col == 0) {
      for (int i = 0; i < 4; i++) {
        int out_n = n_tile * OUTPUT_PER_WG + wave_tile * 16 + g * 4 + i;
        if (out_n < output_size) {
          float sum = acc[i];
          if (d_bias) {
            sum += _gang_bf16_to_float(d_bias[out_n]);
          }
          int out_idx = wave_tile * 16 + g * 4 + i;
          sum += _gang_bf16_to_float(res_base[out_idx]);
          out_base[out_idx] = _gang_float_to_bf16(sum);
        }
      }
    }
  }

  __syncthreads();
}

} // namespace kernel
