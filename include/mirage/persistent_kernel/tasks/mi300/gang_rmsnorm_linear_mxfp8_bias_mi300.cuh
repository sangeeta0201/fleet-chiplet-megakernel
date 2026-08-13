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

// Fused RMSNorm + MXFP8 Gang Linear + Bias for MI350 (gfx950).
//
// This is gang_rmsnorm_linear_mxfp4_bias_mi300.cuh with the weight operand
// widened from 4 bits to 8, exactly as gang_linear_mxfp8_mi300.cuh is
// gang_linear_mxfp4_mi300.cuh widened. GLM-4.7-Flash ships bf16 and we
// quantize at load time, so the format is ours to choose, and E4M3 keeps a
// full mantissa where E2M1 does not -- the model was never quantization-aware
// trained.
//
// Three differences from the FP4 kernel, all mechanical:
//
//   row stride   OPW*K bytes of data instead of OPW*(K/2)
//   weight load  the split gather (_gang_load_fp8_mfma_b) instead of one
//                contiguous 16-byte read, because at 8 bits a lane's 32
//                elements are no longer 16 consecutive bytes
//   cbsz         0 (FP8 E4M3 src0) instead of 4 (FP4 E2M1 src0), via
//                _gang_mfma_f8xf8
//
// Scale indexing is unchanged from FP4. That is load-bearing and not obvious:
// the MFMA addresses its scale operand by matrix position, so lane 16*g+m
// carries row m's exponent for the *contiguous* K block [g*32, g*32+32)
// regardless of which bytes that lane's data register holds. See the note in
// gang_linear_mxfp8_mi300.cuh and tests/standalone/test_mxfp8_mfma_layout.hip.
//
// Weight format, per workgroup of OPW output rows:
//   [n_wgs_per_xcd, wg_bytes], wg_bytes = OPW*K + OPW*(K/32)
// Data is K-major within a row. Scales are indexed [row][k/32].
//
// Dispatch: 8 gang tasks (1 per XCD), tiles assigned by tile_idx.
//   tok_idx = tile_idx / n_wgs_per_xcd
//   wg_idx  = tile_idx % n_wgs_per_xcd
//
// On GLM this serves qkv_a (K=2048, N=2048) and the LM head (K=2048,
// N=155136): 1.03 GB/token of bf16 weight traffic between them, halved.

#pragma once
// The FP8xFP8 MFMA wrapper and the packed-weight helpers both live in the
// dense MXFP8 header; it in turn inherits _gang_moe_get_xcd_id and
// MPK_WS_WAVE_SYNC from earlier task_header.cuh includes rather than naming
// them (see the note there).
#include "tasks/mi300/gang_linear_mxfp8_mi300.cuh"
#include "tasks/mi300/gang_rmsnorm_linear_bias_mi300.cuh" // RMSNorm prologue

namespace kernel {

// _gang_wave_parallel_fp8_quant with the RMSNorm's Phase 4 folded in.
//
// The stock quantizer reads an already-normalized bf16 row. Producing that row
// costs a 4 KB global store and a 4 KB global load per block, and the value
// never leaves the workgroup -- see rmsnorm_rcp_amd. This variant reads the
// *raw* row instead and applies `* rms_rcp * norm_weight[i]` on the way into
// the E4M3 pack, so the bf16 intermediate never exists in any memory space.
//
// The re-read is not new traffic. Phase 1 of the norm just walked this row on
// this block, so it is L1-resident, and the norm weight was going to be read
// by Phase 4 anyway -- the same 4 KB, moved rather than added. What changes is
// the width of the thread mapping: the norm scales 8 elements per thread over
// 256 threads, the quantizer 32 over the 64 threads that own a sub-block, so
// the same multiplies land on a quarter of the lanes. Thirty-two v_fma's
// against a global round trip is not a close trade.
//
// Everything else -- the sub-block split, the clamped __shfl partner index,
// the E8M0 derivation, the packing order -- is _gang_wave_parallel_fp8_quant's
// and is kept identical on purpose.
template <int REDUCTION_SIZE>
__device__ __forceinline__ void _gang_wave_parallel_fp8_quant_rmsnorm(
    unsigned short const *__restrict__ src_bf16,
    unsigned short const *__restrict__ norm_weight,
    float rms_rcp,
    uint8_t *__restrict__ s_tok_fp8,
    uint8_t *__restrict__ s_tok_scales) {

  constexpr int SUB_BLOCK = 32;
  constexpr int NSUBBLOCKS = REDUCTION_SIZE / SUB_BLOCK;
  int const tid = threadIdx.x;
  int const lane_id = tid & 63;

  for (int sb = tid; sb < NSUBBLOCKS; sb += blockDim.x) {
    int const base = sb * SUB_BLOCK;
    int const super_blk = sb / 4;
    int const sub_idx = sb & 3;

    float vals[32];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 32; j++) {
      float v = _gang_bf16_to_float(src_bf16[base + j]) * rms_rcp *
                _gang_bf16_to_float(norm_weight[base + j]);
      vals[j] = v;
      amax = fmaxf(amax, fabsf(v));
    }

    int base_lane = lane_id & ~3;
    int const sb_first = sb - sub_idx;
    int const n_valid = min(4, (NSUBBLOCKS - 1) - sb_first + 1);
    float a0 = __shfl(amax, base_lane);
    float a1 = __shfl(amax, base_lane + min(1, n_valid - 1));
    float a2 = __shfl(amax, base_lane + min(2, n_valid - 1));
    float a3 = __shfl(amax, base_lane + min(3, n_valid - 1));
    float block_amax = fmaxf(fmaxf(a0, a1), fmaxf(a2, a3));

    uint8_t se = _gang_compute_e8m0_fp8(block_amax);
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
    for (int j = 0; j < 32; j += 4) {
      fp8x4_t pk = {};
      pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(
          pk, vals[j], vals[j + 1], scale_f, false);
      pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(
          pk, vals[j + 2], vals[j + 3], scale_f, true);
      *(int *)(s_tok_fp8 + base + j) = *(int const *)&pk;
    }

    if (sub_idx == 0) {
      s_tok_scales[super_blk] = se;
    }
  }
  __syncthreads();
}

// Fused RMSNorm + MXFP8 Gang Linear + Bias.
//
// Template params:
//   BATCH_SIZE        - max batch size (usually 1 for decode)
//   OUTPUT_PER_WG     - output rows per workgroup (e.g. 64)
//   REDUCTION_SIZE    - input/reduction dimension (e.g. 2048)
//   ACTUAL_HIDDEN_DIM - unpadded hidden size for the RMS denominator
//
// Runtime params:
//   tile_idx          - gang task tile index (0..total_tiles_per_xcd-1)
//   n_wgs_per_xcd     - number of workgroups per XCD
//   output_stride     - full output stride (for row indexing)
//   num_active_tokens - actual number of active tokens
template <int BATCH_SIZE,
          int OUTPUT_PER_WG,
          int REDUCTION_SIZE,
          int ACTUAL_HIDDEN_DIM = REDUCTION_SIZE>
__device__ __noinline__ void gang_rmsnorm_linear_mxfp8_bias_kernel(
    void const *norm_input_ptr,  // [batch, REDUCTION_SIZE] bf16
    void const *norm_weight_ptr, // [REDUCTION_SIZE] bf16
    void *norm_output_ptr,       // unused: see the note at Step 1+2
    void const *weight_ptr,      // [n_wgs_per_xcd, wg_bytes] packed MXFP8
    void const *bias_ptr,        // [1, output_size_per_xcd] bf16 (partitioned)
    void *output_ptr,            // [batch, output_stride] bf16 (partitioned)
    int num_active_tokens,
    int n_wgs_per_xcd,
    int output_stride,
    int tile_idx) {

  static_assert(OUTPUT_PER_WG % 16 == 0,
                "OUTPUT_PER_WG must be multiple of 16");
  static_assert(REDUCTION_SIZE % 128 == 0,
                "REDUCTION_SIZE must be multiple of 128 for FP8 MFMA");

  // ── Weight layout constants ─────────────────────────────────────────────
  constexpr int NUM_BLOCKS_32 = REDUCTION_SIZE / 32;
  constexpr int WG_DATA_BYTES = OUTPUT_PER_WG * REDUCTION_SIZE;
  constexpr int WG_SCALE_BYTES = OUTPUT_PER_WG * NUM_BLOCKS_32;
  constexpr int WG_BYTES = WG_DATA_BYTES + WG_SCALE_BYTES;

  // ── MFMA constants ─────────────────────────────────────────────────────
  constexpr int K_PER_MFMA = 128;
  constexpr int MFMA_ITERS = REDUCTION_SIZE / K_PER_MFMA;
  static_assert(MFMA_ITERS >= 4,
                "Depth-4 pipeline requires REDUCTION_SIZE >= 512");
  // Only slot 3 carries a tail guard, so a partial final group would let
  // slots 1 and 2 compute k-tiles that do not exist.
  static_assert(MFMA_ITERS % 4 == 0,
                "Depth-4 pipeline requires REDUCTION_SIZE % 512 == 0");

  // ── Wave tiling ─────────────────────────────────────────────────────────
  constexpr int NUM_WAVES = 4;
  constexpr int TILES_PER_WAVE = OUTPUT_PER_WG / 16 / NUM_WAVES;

  // ── Token activation in shared memory ────────────────────────────────────
  constexpr int FP8_TOK_DATA = REDUCTION_SIZE;

  uint8_t const *W = (uint8_t const *)weight_ptr;
  unsigned short const *d_bias = (unsigned short const *)bias_ptr;
  unsigned short *d_output = (unsigned short *)output_ptr;

  extern __shared__ char _rnlm8_smem[];
  uint8_t *s_tok_fp8 = (uint8_t *)_rnlm8_smem;
  uint8_t *s_tok_scales = s_tok_fp8 + FP8_TOK_DATA;

  int const tid = threadIdx.x;
  int const warp_id = tid >> 6;
  int const lane_id = tid & 63;
  int const col = lane_id & 15;
  int const g = lane_id >> 4;

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  unsigned long long _sp_t0 = 0, _sp_t1 = 0, _sp_t2 = 0, _sp_t3 = 0;
  bool _sp_rec = (tile_idx == 0 && tid == 0 && g_subphase_active);
  if (_sp_rec) {
    _sp_t0 = __builtin_amdgcn_s_memrealtime();
  }
#endif
  // ── Tile dispatch ──────────────────────────────────────────────
  // tile_idx is block-uniform, so this early exit is too, and it is safe to
  // take it in front of the __syncthreads() inside the norm below.
  int batch_count =
      (num_active_tokens < BATCH_SIZE) ? num_active_tokens : BATCH_SIZE;
  int tok_idx = tile_idx / n_wgs_per_xcd;
  int wg_idx = tile_idx % n_wgs_per_xcd;

  if (tok_idx >= batch_count) {
    return;
  }

  // Workgroup weight pointers
  uint8_t const *wg_data = W + static_cast<int64_t>(wg_idx) * WG_BYTES;
  uint8_t const *wg_scales = wg_data + WG_DATA_BYTES;

  // ── Step 1+2: RMSNorm and quantize, with no bf16 round trip ─────────────
  //
  // norm_output_ptr is not written. It used to carry the normalized row from
  // the norm to the quantizer -- 4 KB out to global and 4 KB back, per block,
  // on the dependency path, for a value no other task reads. (In demo/glm5
  // this task's norm_output is `rmsnorm_out`, shared scratch for qkv_a and the
  // LM head, and nothing takes it as an input.) The row now goes norm ->
  // quantizer in registers; the parameter stays only because the task
  // signature is generated. A consumer that actually wants the normalized row
  // needs a store added back here, not a silent read.
  //
  // The loop over batch rows goes too: it normalized every row of the batch on
  // every block and then used one. rms_rcp is this block's own token's.
  (void)norm_output_ptr;
  unsigned short const *input_row =
      (unsigned short const *)norm_input_ptr + tok_idx * REDUCTION_SIZE;
  float const rms_rcp =
      gang_rmsnorm_detail::rmsnorm_rcp_amd<REDUCTION_SIZE, ACTUAL_HIDDEN_DIM>(
          input_row);

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  if (_sp_rec) {
    _sp_t1 = __builtin_amdgcn_s_memrealtime();
  }
#endif
  _gang_wave_parallel_fp8_quant_rmsnorm<REDUCTION_SIZE>(
      input_row,
      (unsigned short const *)norm_weight_ptr,
      rms_rcp,
      s_tok_fp8,
      s_tok_scales);

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  if (_sp_rec) {
    _sp_t2 = __builtin_amdgcn_s_memrealtime();
  }
#endif
  // ── Step 3: MFMA FP8(weights) x FP8(tokens) ────────────────────────────
  if constexpr (OUTPUT_PER_WG >= 64) {
    // N-parallel: 4 waves handle different output rows (depth-4 pipeline)
    for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; tile_iter++) {
      int wave_tile = warp_id + tile_iter * NUM_WAVES;
      int w_row = wave_tile * 16 + col;

      uint8_t const *w_data_row =
          wg_data + static_cast<int64_t>(w_row) * REDUCTION_SIZE;
      int const row_scale_base = w_row * NUM_BLOCKS_32;

      f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

      // Pre-fill: load k-tiles 0..3 into pipeline slots
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
          i32x8_t b =
              _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 1) * K_PER_MFMA, g);
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
          i32x8_t b =
              _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 2) * K_PER_MFMA, g);
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
          i32x8_t b =
              _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 3) * K_PER_MFMA, g);
          int sb = (int)s_tok_scales[ki + 3];
          acc = _gang_mfma_f8xf8(a3, b, acc, sa3, sb);
        }
        if (ki + 7 < MFMA_ITERS) {
          int kt7 = (ki + 7) * K_PER_MFMA;
          a3 = _gang_load_fp8_mfma_b(w_data_row, kt7, g);
          sa3 = (int)wg_scales[row_scale_base + kt7 / 32 + g];
        }
      }

      // ── Step 4: Bias epilogue, write BF16 output ─────────────────────────
      if (col == 0) {
        for (int i = 0; i < 4; i++) {
          int out_n = wg_idx * OUTPUT_PER_WG + wave_tile * 16 + g * 4 + i;
          float sum = acc[i];

          // Add bias (partitioned per XCD)
          unsigned bt = (unsigned)d_bias[out_n] << 16;
          float bv;
          __builtin_memcpy(&bv, &bt, 4);

          int out_idx = tok_idx * output_stride + out_n;
          d_output[out_idx] = _gang_float_to_bf16(sum + bv);
        }
      }
    }
  } else {
    // K-parallel: 4 waves all process same 16 rows, split K across waves.
    // Note this branch emits exactly 16 rows per workgroup regardless of
    // OUTPUT_PER_WG -- w_row is `col` and the epilogue writes g*4+i -- so 16
    // is the only value it is correct for. The FP4 original leaves that
    // implicit; make it a hard error here.
    static_assert(OUTPUT_PER_WG == 16,
                  "the K-parallel branch covers 16 output rows per workgroup");
    constexpr int TOTAL_K_ITERS = MFMA_ITERS;
    constexpr int ITERS_PER_WAVE = TOTAL_K_ITERS / NUM_WAVES;
    static_assert(TOTAL_K_ITERS % NUM_WAVES == 0,
                  "MFMA_ITERS must be divisible by NUM_WAVES for K-parallel");
    static_assert(ITERS_PER_WAVE >= 4,
                  "Depth-4 K-parallel requires ITERS_PER_WAVE >= 4");

    int const ki_start = warp_id * ITERS_PER_WAVE;
    int const ki_end = ki_start + ITERS_PER_WAVE;

    int w_row = col; // All 4 waves process same 16 output rows
    uint8_t const *w_data_row =
        wg_data + static_cast<int64_t>(w_row) * REDUCTION_SIZE;
    int const row_scale_base = w_row * NUM_BLOCKS_32;

    f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

    // Pre-fill: load k-tiles 0..3 into pipeline slots
    i32x8_t a0 = _gang_load_fp8_mfma_b(w_data_row, ki_start * K_PER_MFMA, g);
    int sa0 = (int)wg_scales[row_scale_base + ki_start * 4 + g];
    i32x8_t a1 =
        _gang_load_fp8_mfma_b(w_data_row, (ki_start + 1) * K_PER_MFMA, g);
    int sa1 = (int)wg_scales[row_scale_base + (ki_start + 1) * 4 + g];
    i32x8_t a2 =
        _gang_load_fp8_mfma_b(w_data_row, (ki_start + 2) * K_PER_MFMA, g);
    int sa2 = (int)wg_scales[row_scale_base + (ki_start + 2) * 4 + g];
    i32x8_t a3 =
        _gang_load_fp8_mfma_b(w_data_row, (ki_start + 3) * K_PER_MFMA, g);
    int sa3 = (int)wg_scales[row_scale_base + (ki_start + 3) * 4 + g];

// IMPORTANT: #pragma unroll 1 prevents ROCm miscompilation.
#pragma unroll 1
    for (int ki = ki_start; ki < ki_end; ki += 4) {
      // Slot 0: compute k-tile ki, prefetch ki+4
      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, ki * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki];
        acc = _gang_mfma_f8xf8(a0, b, acc, sa0, sb);
      }
      if (ki + 4 < ki_end) {
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
      if (ki + 5 < ki_end) {
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
      if (ki + 6 < ki_end) {
        int kt6 = (ki + 6) * K_PER_MFMA;
        a2 = _gang_load_fp8_mfma_b(w_data_row, kt6, g);
        sa2 = (int)wg_scales[row_scale_base + kt6 / 32 + g];
      }

      // Slot 3: compute k-tile ki+3, prefetch ki+7
      if (ki + 3 < ki_end) {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 3) * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki + 3];
        acc = _gang_mfma_f8xf8(a3, b, acc, sa3, sb);
      }
      if (ki + 7 < ki_end) {
        int kt7 = (ki + 7) * K_PER_MFMA;
        a3 = _gang_load_fp8_mfma_b(w_data_row, kt7, g);
        sa3 = (int)wg_scales[row_scale_base + kt7 / 32 + g];
      }
    }

    // Cross-wave LDS reduction (reuse token scratch area, dead after MFMA)
    float *lds_reduce = (float *)_rnlm8_smem;
    if (col == 0) {
      for (int i = 0; i < 4; i++) {
        lds_reduce[warp_id * OUTPUT_PER_WG + g * 4 + i] = acc[i];
      }
    }
    __syncthreads();

    // Wave 0 reduces across waves and writes output with bias
    if (warp_id == 0 && col == 0) {
      for (int i = 0; i < 4; i++) {
        float v = 0.0f;
        for (int w = 0; w < NUM_WAVES; w++) {
          v += lds_reduce[w * OUTPUT_PER_WG + g * 4 + i];
        }

        int out_n = wg_idx * OUTPUT_PER_WG + g * 4 + i;

        unsigned bt = (unsigned)d_bias[out_n] << 16;
        float bv;
        __builtin_memcpy(&bv, &bt, 4);

        int out_idx = tok_idx * output_stride + out_n;
        d_output[out_idx] = _gang_float_to_bf16(v + bv);
      }
    }
  }

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  if (_sp_rec) {
    _sp_t3 = __builtin_amdgcn_s_memrealtime();
    // Slot 0: QKV. [1]=RMSNorm [2]=FP8Quant [3]=MFMA+Epi
    atomicAdd(&g_subphase_ns[0][1], (_sp_t1 - _sp_t0) * 10);
    atomicAdd(&g_subphase_ns[0][2], (_sp_t2 - _sp_t1) * 10);
    atomicAdd(&g_subphase_ns[0][3], (_sp_t3 - _sp_t2) * 10);
    atomicAdd(&g_subphase_cnt[0], 1ULL);
  }
#endif
  __syncthreads();
}

} // namespace kernel
