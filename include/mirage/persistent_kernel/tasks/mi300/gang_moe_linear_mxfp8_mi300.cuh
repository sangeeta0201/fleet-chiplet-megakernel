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

// Gang MoE W13/W2 with MXFP8 weights and FP8xFP8 MFMA (gfx950).
//
// This is the largest single item in the token's weight budget: five active
// experts per layer move 94.4 MB of the 180.9 MB a MoE layer touches, which is
// 47.4% of the whole 9.17 GB token. Halving it is worth more than every
// scheduling change on this branch put together.
//
// Two lineages meet here. The GEMM body is gpt-oss's
// gang_moe_linear_mxfp4_kernel_mi300 -- flat round-robin tile distribution over
// the 8 XCDs, per-token tiles, on-the-fly FP8 quantization of the activation
// into LDS -- with the weight operand widened 4 bits to 8 exactly as
// gang_linear_mxfp8_mi300.cuh does it, and with that file's depth-4 software
// pipeline over the k-loop. The epilogues are GLM's, transplanted unchanged
// from gang_moe_linear_mi300.cuh: the fused SiLU-mul on W13 and the fused
// topk-weight + f32 atomicAdd on W2. Those two fusions are worth ~500 us
// between them and are independent of the weight format, so the port keeps
// them rather than reverting to gpt-oss's swigluoai and its separate reduce.
//
// Structural difference from GLM's CK kernels worth naming: those tile W13's M
// dimension in blocks of 16 tokens, while this decomposes per token like the
// gpt-oss kernel does. At batch 1 the two produce the same 48 tiles per expert,
// so tiles_per_expert does not move; above batch 1 the tile count grows
// linearly here instead of in steps of 16.
//
// See gang_linear_mxfp8_mi300.cuh for why the scale indexing is what it is --
// the MFMA addresses its scale operand by matrix position, not by which bytes
// the lane loaded, and the two only coincide at 4 bits.
//
// Weight format, per workgroup of OUTPUT_PER_WG rows of one expert:
//   [FP8 E4M3 data: OPW * K bytes][E8M0 scales: OPW * (K/32) bytes]
// Experts are contiguous: expert e starts at e * EXPERT_WGS * WG_BYTES.

#pragma once
// Inherits the same include ordering as gang_linear_mxfp8_mi300.cuh; see the
// note there. silu_mul_mi300.cuh is named explicitly rather than inherited
// because unlike gang_moe_linear_mi300.cuh -- the other header that supplies
// fast_silu -- it is CK-free, so naming it costs nothing.
#include "tasks/mi300/gang_linear_mxfp8_mi300.cuh"
#include "tasks/mi300/silu_mul_mi300.cuh"

namespace kernel {

// Resolve a gang tile index to (expert_id, token, workgroup).
//
// Tiles interleave round-robin across the XCDs -- global_tile = tile_idx*8 +
// xcd_id -- rather than blocking, so that all 8 XCDs stay busy when fewer than
// 8 experts are active, which at top-4-plus-shared is always.
//
// Returns false when this worker has no tile, has run past the activated
// experts, or the token does not route to this expert.
template <int BATCH_SIZE, int NUM_EXPERTS, int TILES_PER_EXPERT, int WGS>
__device__ __forceinline__ bool _gang_moe_mxfp8_tile(int tile_idx,
                                                     int const *d_mask,
                                                     int const *d_routing,
                                                     int *expert_id,
                                                     int *tok_idx,
                                                     int *wg_idx,
                                                     int *topk_slot) {
  int const num_activated_experts = d_mask[NUM_EXPERTS];
  int global_tile = tile_idx * 8 + _gang_moe_get_xcd_id();
  if (global_tile >= num_activated_experts * TILES_PER_EXPERT) {
    return false;
  }
  int const e = d_mask[global_tile / TILES_PER_EXPERT];
  int const within = global_tile % TILES_PER_EXPERT;
  int const tok = within / WGS;
  if (tok >= BATCH_SIZE) {
    return false;
  }
  int const route_val = d_routing[e * BATCH_SIZE + tok];
  if (route_val == 0) {
    return false;
  }
  *expert_id = e;
  *tok_idx = tok;
  *wg_idx = within % WGS;
  *topk_slot = route_val - 1;
  return true;
}

// Gang MoE W13 (gate+up) with MXFP8 weights.
//
// Input:  [batch, K = hidden]                          bf16
// Weight: [num_experts, WGS, WG_BYTES]                 MXFP8 packed
// Output: [batch, topk, OUTPUT_STRIDE]                 bf16, or
//         [batch, topk, OUTPUT_STRIDE/2] when FUSE_SWIGLU
//
// FUSE_SWIGLU requires the weight rows pairwise interleaved -- row 2j is
// gate_j, row 2j+1 is up_j -- because a lane owns acc[0..3] at four
// *consecutive* N, so gate and its up partner only meet in registers when they
// are adjacent columns. The packer preserves row order, so the same
// interleave the bf16 path already applies carries over untouched.
template <int BATCH_SIZE,
          int OUTPUT_SIZE,
          int OUTPUT_STRIDE,
          int REDUCTION_SIZE,
          int NUM_EXPERTS,
          int NUM_TOPK,
          int TILES_PER_EXPERT,
          int OUTPUT_PER_WG,
          bool FUSE_SWIGLU = false,
          // Set by a fused caller that runs W2 in the same task, with only an
          // in-kernel barrier between them. W2 for one expert reduces over
          // that expert's whole intermediate, whose workgroups are spread
          // across all 8 XCDs, so it reads activations this XCD produced.
          // Per-XCD L2 is not coherent on MI300/MI350: as separate tasks the
          // event boundary's buffer_wbl2 is what makes them visible, and a
          // fused caller has no such boundary. Writing through costs the
          // store's L2 residency, which nothing on this XCD wants back.
          bool WRITE_THROUGH = false>
__device__ __noinline__ void
    gang_moe_w13_linear_mxfp8_kernel(void const *input_ptr,
                                     void const *weight_ptr,
                                     void const *routing_ptr,
                                     void const *mask_ptr,
                                     void const *bias_ptr,
                                     void *output_ptr,
                                     int tile_idx) {
  static_assert(OUTPUT_PER_WG % 64 == 0,
                "OUTPUT_PER_WG must be a multiple of 64 (4 waves x 16 rows)");
  static_assert(REDUCTION_SIZE % 128 == 0,
                "K must be a multiple of 128 for FP8 MFMA");

  constexpr int NUM_BLOCKS_32 = REDUCTION_SIZE / 32;
  constexpr int WG_DATA_BYTES = OUTPUT_PER_WG * REDUCTION_SIZE;
  constexpr int WG_SCALE_BYTES = OUTPUT_PER_WG * NUM_BLOCKS_32;
  constexpr int WG_BYTES = WG_DATA_BYTES + WG_SCALE_BYTES;
  constexpr int EXPERT_WGS = OUTPUT_STRIDE / OUTPUT_PER_WG;
  constexpr int64_t EXPERT_BYTES = static_cast<int64_t>(EXPERT_WGS) * WG_BYTES;

  constexpr int K_PER_MFMA = 128;
  constexpr int MFMA_ITERS = REDUCTION_SIZE / K_PER_MFMA;
  // Same depth-4 pipeline as the dense MXFP8 kernel, and the same constraint:
  // only slot 3 carries a tail guard, so a partial final group would let slots
  // 1 and 2 compute k-tiles that do not exist.
  static_assert(MFMA_ITERS >= 4 && MFMA_ITERS % 4 == 0,
                "Depth-4 pipeline requires REDUCTION_SIZE % 512 == 0");

  constexpr int NUM_WAVES = 4;
  constexpr int TILES_PER_WAVE = OUTPUT_PER_WG / 16 / NUM_WAVES;
  constexpr int FP8_TOK_DATA = REDUCTION_SIZE;

  unsigned short const *A = (unsigned short const *)input_ptr;
  uint8_t const *W = (uint8_t const *)weight_ptr;
  unsigned short const *d_bias = (unsigned short const *)bias_ptr;
  unsigned short *d_output = (unsigned short *)output_ptr;

  int expert_id, tok_idx, wg_idx, topk_slot;
  if (!_gang_moe_mxfp8_tile<BATCH_SIZE,
                            NUM_EXPERTS,
                            TILES_PER_EXPERT,
                            EXPERT_WGS>(tile_idx,
                                        (int const *)mask_ptr,
                                        (int const *)routing_ptr,
                                        &expert_id,
                                        &tok_idx,
                                        &wg_idx,
                                        &topk_slot)) {
    return;
  }

  uint8_t const *wg_data = W + static_cast<int64_t>(expert_id) * EXPERT_BYTES +
                           static_cast<int64_t>(wg_idx) * WG_BYTES;
  uint8_t const *wg_scales = wg_data + WG_DATA_BYTES;

  extern __shared__ char _gang_moe_mxfp8_smem[];
  uint8_t *s_tok_fp8 = (uint8_t *)_gang_moe_mxfp8_smem;
  uint8_t *s_tok_scales = s_tok_fp8 + FP8_TOK_DATA;

  int const tid = threadIdx.x;
  int const warp_id = tid >> 6;
  int const lane_id = tid & 63;
  int const col = lane_id & 15; // weight row within the 16x16 MFMA tile
  int const g = lane_id >> 4;   // K-group (0..3)

  // Phase 1: quantize this token's activation to FP8 E4M3 in LDS.
  _gang_wave_parallel_fp8_quant<REDUCTION_SIZE>(
      A + static_cast<size_t>(tok_idx) * REDUCTION_SIZE,
      s_tok_fp8,
      s_tok_scales);

  // Phase 2: depth-4 pipelined FP8(weight) x FP8(token) MFMA.
  for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; tile_iter++) {
    int const wave_tile = warp_id + tile_iter * NUM_WAVES;
    int const w_row = wave_tile * 16 + col;
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
        acc = _gang_mfma_f8xf8(a0, b, acc, sa0, (int)s_tok_scales[ki]);
      }
      if (ki + 4 < MFMA_ITERS) {
        int kt4 = (ki + 4) * K_PER_MFMA;
        a0 = _gang_load_fp8_mfma_b(w_data_row, kt4, g);
        sa0 = (int)wg_scales[row_scale_base + kt4 / 32 + g];
      }

      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 1) * K_PER_MFMA, g);
        acc = _gang_mfma_f8xf8(a1, b, acc, sa1, (int)s_tok_scales[ki + 1]);
      }
      if (ki + 5 < MFMA_ITERS) {
        int kt5 = (ki + 5) * K_PER_MFMA;
        a1 = _gang_load_fp8_mfma_b(w_data_row, kt5, g);
        sa1 = (int)wg_scales[row_scale_base + kt5 / 32 + g];
      }

      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 2) * K_PER_MFMA, g);
        acc = _gang_mfma_f8xf8(a2, b, acc, sa2, (int)s_tok_scales[ki + 2]);
      }
      if (ki + 6 < MFMA_ITERS) {
        int kt6 = (ki + 6) * K_PER_MFMA;
        a2 = _gang_load_fp8_mfma_b(w_data_row, kt6, g);
        sa2 = (int)wg_scales[row_scale_base + kt6 / 32 + g];
      }

      if (ki + 3 < MFMA_ITERS) {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 3) * K_PER_MFMA, g);
        acc = _gang_mfma_f8xf8(a3, b, acc, sa3, (int)s_tok_scales[ki + 3]);
      }
      if (ki + 7 < MFMA_ITERS) {
        int kt7 = (ki + 7) * K_PER_MFMA;
        a3 = _gang_load_fp8_mfma_b(w_data_row, kt7, g);
        sa3 = (int)wg_scales[row_scale_base + kt7 / 32 + g];
      }
    }

    // Epilogue. acc[i] = C[g*4+i][col]; at one token per tile only col 0 holds
    // a result.
    if (col == 0) {
      int const out_base = wg_idx * OUTPUT_PER_WG + wave_tile * 16 + g * 4;
      unsigned short const *bias_row = d_bias + expert_id * OUTPUT_STRIDE;

      if constexpr (FUSE_SWIGLU) {
        constexpr int ACT_STRIDE = OUTPUT_STRIDE / 2;
        static_assert(2 * ACT_STRIDE == OUTPUT_STRIDE,
                      "FUSE_SWIGLU expects a half-width activation output");
        // out_base is a multiple of 4, so the four accumulators are always
        // two whole gate/up pairs starting on a pair boundary.
        unsigned short *act_addr = d_output +
                                   tok_idx * (NUM_TOPK * ACT_STRIDE) +
                                   topk_slot * ACT_STRIDE + (out_base >> 1);
        unsigned short act[2];
        bool act_ok[2];
#pragma unroll
        for (int p = 0; p < 2; p++) {
          int const out_n = out_base + 2 * p;
          act_ok[p] = (out_n + 1 < OUTPUT_SIZE);
          if (act_ok[p]) {
            float const gate =
                acc[2 * p] + _gang_bf16_to_float(bias_row[out_n]);
            float const up =
                acc[2 * p + 1] + _gang_bf16_to_float(bias_row[out_n + 1]);
            act[p] = _gang_float_to_bf16(fast_silu(gate) * up);
          }
        }
        if constexpr (WRITE_THROUGH) {
          // The two activations are adjacent columns of the same row, and
          // out_base is a multiple of 4, so act_addr is 4-byte aligned and the
          // pair is one dword -- half as many write-through stores as doing
          // them separately.
          if (act_ok[0] && act_ok[1]) {
            unsigned packed = (unsigned)act[0] | ((unsigned)act[1] << 16);
            st_wt_u32((void *)act_addr, packed);
          } else {
#pragma unroll
            for (int p = 0; p < 2; p++) {
              if (act_ok[p]) {
                st_wt_u16((void *)&act_addr[p], act[p]);
              }
            }
          }
        } else {
#pragma unroll
          for (int p = 0; p < 2; p++) {
            if (act_ok[p]) {
              act_addr[p] = act[p];
            }
          }
        }
      } else {
        unsigned short *out_addr = d_output +
                                   tok_idx * (NUM_TOPK * OUTPUT_STRIDE) +
                                   topk_slot * OUTPUT_STRIDE + out_base;
#pragma unroll
        for (int i = 0; i < 4; i++) {
          int const out_n = out_base + i;
          if (out_n < OUTPUT_SIZE) {
            out_addr[i] = _gang_float_to_bf16(
                acc[i] + _gang_bf16_to_float(bias_row[out_n]));
          }
        }
      }
    }
  }

  __syncthreads();
}

// Gang MoE W2 (down projection) with MXFP8 weights.
//
// Input:  [batch, topk, K = intermediate]              bf16
// Weight: [num_experts, WGS, WG_BYTES]                 MXFP8 packed
// Output: [batch, topk, OUTPUT_STRIDE] bf16, or the [batch, hidden] f32
//         workspace when FUSE_MULSUMADD
//
// FUSE_MULSUMADD folds the topk weighting and the cross-expert sum into this
// epilogue: scale by routing_weight[tok, slot] and f32-atomicAdd into a
// workspace, instead of writing a [batch, topk, hidden] slab that a later task
// re-reads and reduces. The accumulation stays fp32 all the way to the residual
// add. Atomics make the summation order non-deterministic, which is the trade
// gpt-oss already takes.
//
// The activation this reads is produced by another XCD's W13 tile, so it is a
// cross-XCD read and uses the NT-load quantizer to avoid polluting L2 with a
// line nobody will read again.
template <int BATCH_SIZE,
          int OUTPUT_SIZE,
          int OUTPUT_STRIDE,
          int REDUCTION_SIZE,
          int NUM_EXPERTS,
          int NUM_TOPK,
          int TILES_PER_EXPERT,
          int OUTPUT_PER_WG,
          bool FUSE_MULSUMADD = false>
__device__ __noinline__ void
    gang_moe_w2_linear_mxfp8_kernel(void const *input_ptr,
                                    void const *weight_ptr,
                                    void const *routing_ptr,
                                    void const *mask_ptr,
                                    void const *bias_ptr,
                                    void *output_ptr,
                                    int tile_idx,
                                    void const *routing_weight_ptr = nullptr) {
  static_assert(OUTPUT_PER_WG % 64 == 0,
                "OUTPUT_PER_WG must be a multiple of 64 (4 waves x 16 rows)");
  static_assert(REDUCTION_SIZE % 128 == 0,
                "K must be a multiple of 128 for FP8 MFMA");

  constexpr int NUM_BLOCKS_32 = REDUCTION_SIZE / 32;
  constexpr int WG_DATA_BYTES = OUTPUT_PER_WG * REDUCTION_SIZE;
  constexpr int WG_SCALE_BYTES = OUTPUT_PER_WG * NUM_BLOCKS_32;
  constexpr int WG_BYTES = WG_DATA_BYTES + WG_SCALE_BYTES;
  constexpr int EXPERT_WGS = OUTPUT_STRIDE / OUTPUT_PER_WG;
  constexpr int64_t EXPERT_BYTES = static_cast<int64_t>(EXPERT_WGS) * WG_BYTES;

  constexpr int K_PER_MFMA = 128;
  constexpr int MFMA_ITERS = REDUCTION_SIZE / K_PER_MFMA;
  static_assert(MFMA_ITERS >= 4 && MFMA_ITERS % 4 == 0,
                "Depth-4 pipeline requires REDUCTION_SIZE % 512 == 0");

  constexpr int NUM_WAVES = 4;
  constexpr int TILES_PER_WAVE = OUTPUT_PER_WG / 16 / NUM_WAVES;
  constexpr int FP8_TOK_DATA = REDUCTION_SIZE;

  unsigned short const *A = (unsigned short const *)input_ptr;
  uint8_t const *W = (uint8_t const *)weight_ptr;
  unsigned short const *d_bias = (unsigned short const *)bias_ptr;
  unsigned short *d_output = (unsigned short *)output_ptr;
  float *d_workspace = (float *)output_ptr;
  float const *d_routing_weight = (float const *)routing_weight_ptr;

  int expert_id, tok_idx, wg_idx, topk_slot;
  if (!_gang_moe_mxfp8_tile<BATCH_SIZE,
                            NUM_EXPERTS,
                            TILES_PER_EXPERT,
                            EXPERT_WGS>(tile_idx,
                                        (int const *)mask_ptr,
                                        (int const *)routing_ptr,
                                        &expert_id,
                                        &tok_idx,
                                        &wg_idx,
                                        &topk_slot)) {
    return;
  }

  uint8_t const *wg_data = W + static_cast<int64_t>(expert_id) * EXPERT_BYTES +
                           static_cast<int64_t>(wg_idx) * WG_BYTES;
  uint8_t const *wg_scales = wg_data + WG_DATA_BYTES;

  extern __shared__ char _gang_moe_mxfp8_smem[];
  uint8_t *s_tok_fp8 = (uint8_t *)_gang_moe_mxfp8_smem;
  uint8_t *s_tok_scales = s_tok_fp8 + FP8_TOK_DATA;

  int const tid = threadIdx.x;
  int const warp_id = tid >> 6;
  int const lane_id = tid & 63;
  int const col = lane_id & 15;
  int const g = lane_id >> 4;

  _gang_wave_parallel_fp8_quant_nt<REDUCTION_SIZE>(
      A + static_cast<size_t>(tok_idx) * (NUM_TOPK * REDUCTION_SIZE) +
          static_cast<size_t>(topk_slot) * REDUCTION_SIZE,
      s_tok_fp8,
      s_tok_scales);

  float rw = 0.0f;
  if constexpr (FUSE_MULSUMADD) {
    rw = d_routing_weight[tok_idx * NUM_TOPK + topk_slot];
  }

  for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; tile_iter++) {
    int const wave_tile = warp_id + tile_iter * NUM_WAVES;
    int const w_row = wave_tile * 16 + col;
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
        acc = _gang_mfma_f8xf8(a0, b, acc, sa0, (int)s_tok_scales[ki]);
      }
      if (ki + 4 < MFMA_ITERS) {
        int kt4 = (ki + 4) * K_PER_MFMA;
        a0 = _gang_load_fp8_mfma_b(w_data_row, kt4, g);
        sa0 = (int)wg_scales[row_scale_base + kt4 / 32 + g];
      }

      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 1) * K_PER_MFMA, g);
        acc = _gang_mfma_f8xf8(a1, b, acc, sa1, (int)s_tok_scales[ki + 1]);
      }
      if (ki + 5 < MFMA_ITERS) {
        int kt5 = (ki + 5) * K_PER_MFMA;
        a1 = _gang_load_fp8_mfma_b(w_data_row, kt5, g);
        sa1 = (int)wg_scales[row_scale_base + kt5 / 32 + g];
      }

      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 2) * K_PER_MFMA, g);
        acc = _gang_mfma_f8xf8(a2, b, acc, sa2, (int)s_tok_scales[ki + 2]);
      }
      if (ki + 6 < MFMA_ITERS) {
        int kt6 = (ki + 6) * K_PER_MFMA;
        a2 = _gang_load_fp8_mfma_b(w_data_row, kt6, g);
        sa2 = (int)wg_scales[row_scale_base + kt6 / 32 + g];
      }

      if (ki + 3 < MFMA_ITERS) {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 3) * K_PER_MFMA, g);
        acc = _gang_mfma_f8xf8(a3, b, acc, sa3, (int)s_tok_scales[ki + 3]);
      }
      if (ki + 7 < MFMA_ITERS) {
        int kt7 = (ki + 7) * K_PER_MFMA;
        a3 = _gang_load_fp8_mfma_b(w_data_row, kt7, g);
        sa3 = (int)wg_scales[row_scale_base + kt7 / 32 + g];
      }
    }

    if (col == 0) {
      int const out_base = wg_idx * OUTPUT_PER_WG + wave_tile * 16 + g * 4;
      unsigned short const *bias_row = d_bias + expert_id * OUTPUT_STRIDE;

      if constexpr (FUSE_MULSUMADD) {
        // The shared expert rides in routing slot NUM_TOPK-1 with weight 1.0,
        // so it needs no special case here.
        float *ws_addr =
            d_workspace + static_cast<size_t>(tok_idx) * OUTPUT_STRIDE +
            out_base;
#pragma unroll
        for (int i = 0; i < 4; i++) {
          if (out_base + i < OUTPUT_SIZE) {
            atomicAdd(&ws_addr[i],
                      (acc[i] + _gang_bf16_to_float(bias_row[out_base + i])) *
                          rw);
          }
        }
      } else {
        unsigned short *out_addr =
            d_output +
            static_cast<size_t>(tok_idx) * (NUM_TOPK * OUTPUT_STRIDE) +
            static_cast<size_t>(topk_slot) * OUTPUT_STRIDE + out_base;
#pragma unroll
        for (int i = 0; i < 4; i++) {
          if (out_base + i < OUTPUT_SIZE) {
            out_addr[i] = _gang_float_to_bf16(
                acc[i] + _gang_bf16_to_float(bias_row[out_base + i]));
          }
        }
      }
    }
  }

  __syncthreads();
}

} // namespace kernel
