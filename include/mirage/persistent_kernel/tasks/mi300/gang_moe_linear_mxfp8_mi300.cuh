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
//
// WEIGHT_FP4 narrows that data half to OPW * (K/2) bytes of E2M1 nibbles and
// nothing else. The scales are already E8M0-per-32 in both formats, the tile
// decode, the LDS activation and both epilogues are format-blind, and the MFMA
// is the same v_mfma_scale_f32_16x16x128_f8f6f4 with cbsz switched from FP8 to
// FP4 -- so the two paths differ in exactly three expressions (the row stride,
// the A-operand load, and cbsz) and are one template rather than two files.
// The activation stays FP8: this is W4A8, and quantizing the token to 4 bits
// as gpt-oss's f4xf4 path does would spend the quality budget twice over for
// no bandwidth, since the token is 2 KB against the weight's 48 MB.

#pragma once
// Inherits the same include ordering as gang_linear_mxfp8_mi300.cuh; see the
// note there. silu_mul_mi300.cuh is named explicitly rather than inherited
// because unlike gang_moe_linear_mi300.cuh -- the other header that supplies
// fast_silu -- it is CK-free, so naming it costs nothing.
#include "tasks/mi300/gang_linear_mxfp8_mi300.cuh"
#include "tasks/mi300/silu_mul_mi300.cuh"

namespace kernel {

// The weight operand at whichever width it is packed. FP4 fills only the lower
// 128 bits of the i32x8 and is contiguous there, so it is one 16-byte load
// against FP8's split gather of two.
template <bool FP4>
__device__ __forceinline__ i32x8_t _gang_load_w_mfma_a(uint8_t const *row,
                                                       int kt,
                                                       int g) {
  if constexpr (FP4) {
    return _gang_load_fp4_mfma_b(row, kt, g);
  } else {
    return _gang_load_fp8_mfma_b(row, kt, g);
  }
}

template <bool FP4>
__device__ __forceinline__ f32x4_t _gang_mfma_w_x_f8(
    i32x8_t a, i32x8_t b, f32x4_t c, int scale_a, int scale_b) {
  if constexpr (FP4) {
    return _gang_mfma_f4xf8(a, b, c, scale_a, scale_b);
  } else {
    return _gang_mfma_f8xf8(a, b, c, scale_a, scale_b);
  }
}

// Resolve a gang tile index to (expert_id, token, workgroup).
//
// Tiles interleave round-robin across the XCDs -- global_tile = tile_idx*8 +
// xcd_id -- rather than blocking, so that all 8 XCDs stay busy when fewer than
// 8 experts are active, which at top-4-plus-shared is always.
//
// Returns false when this worker has no tile, has run past the activated
// experts, or the token does not route to this expert.
//
// ── expert parallelism ────────────────────────────────────────────────────
// Under EP each rank stores and computes only EP_NUM_ROUTED / EP_WORLD_SIZE of
// the routed experts (the id split gpt-oss calls `ep_slice`; GLM cannot use its
// slot split, because top-4 does not span 8 ranks). Routing, the mask and the
// activated list stay REPLICATED and keyed by the global expert id -- only the
// weight and bias tensors are local, so only their index is remapped, to
// *local_eid.
//
// The shared expert has no id range to fall in: it is id EP_NUM_ROUTED, every
// token routes to it, and it must be computed exactly once across the whole
// world or the cross-rank sum would count it EP_WORLD_SIZE times. One rank
// (EP_SHARED_PE) owns it; the others store a copy they never read, which is
// 36 MB of dead weight per rank at GLM-5.2 dims and not worth a ragged tensor
// shape to avoid.
//
// The tile space is built over the OWNED subsequence of the activated list,
// not over the whole list with the non-owned tiles early-returning. gpt-oss
// measured why: owned tiles come in runs of TILES_PER_EXPERT, and a run
// against the stride-N worker map aliases into 2 real tiles on one worker
// where the tile count should never exceed 1 -- and that straggler is exactly
// what the next barrier waits out. Compacting costs two scans of the activated
// list (at most TOPK_K + 1 loads, hoisted uniformly across the workgroup) per
// tile, against a tile that is a whole GEMV.
template <int BATCH_SIZE,
          int NUM_EXPERTS,
          int TILES_PER_EXPERT,
          int WGS,
          int EP_WORLD_SIZE = 1,
          int EP_MY_PE = 0,
          // Routed experts only; the shared expert is id EP_NUM_ROUTED.
          int EP_NUM_ROUTED = NUM_EXPERTS,
          int EP_SHARED_PE = 0>
__device__ __forceinline__ bool _gang_moe_mxfp8_tile(int tile_idx,
                                                     int const *d_mask,
                                                     int const *d_routing,
                                                     int *expert_id,
                                                     int *local_eid,
                                                     int *tok_idx,
                                                     int *wg_idx,
                                                     int *topk_slot) {
  int const num_activated_experts = d_mask[NUM_EXPERTS];
  int global_tile = tile_idx * 8 + _gang_moe_get_xcd_id();
  int e, leid;
  if constexpr (EP_WORLD_SIZE > 1) {
    static_assert(EP_NUM_ROUTED % EP_WORLD_SIZE == 0,
                  "ep_slice needs the routed expert count to divide by the "
                  "world size");
    constexpr int EP_LOCAL_ROUTED = EP_NUM_ROUTED / EP_WORLD_SIZE;
    constexpr int EP_BASE = EP_MY_PE * EP_LOCAL_ROUTED;
    int const owned_rank = global_tile / TILES_PER_EXPERT;
    int seen = -1;
    e = -1;
    for (int i = 0; i < num_activated_experts; i++) {
      int const cand = d_mask[i];
      bool const owned =
          (cand >= EP_NUM_ROUTED)
              ? (EP_MY_PE == EP_SHARED_PE)
              : (cand >= EP_BASE && cand < EP_BASE + EP_LOCAL_ROUTED);
      if (owned && ++seen == owned_rank) {
        e = cand;
        break;
      }
    }
    if (e < 0) {
      return false;
    }
    // Local weight layout: the owned routed range packed down to
    // [0, EP_LOCAL_ROUTED), then the shared expert.
    leid = (e >= EP_NUM_ROUTED) ? (EP_LOCAL_ROUTED + (e - EP_NUM_ROUTED))
                                : (e - EP_BASE);
  } else {
    if (global_tile >= num_activated_experts * TILES_PER_EXPERT) {
      return false;
    }
    e = d_mask[global_tile / TILES_PER_EXPERT];
    leid = e;
  }
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
  *local_eid = leid;
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
          bool WRITE_THROUGH = false,
          // E2M1 nibbles instead of E4M3 bytes on the weight side only; see
          // the note at the top of the file.
          bool WEIGHT_FP4 = false,
          // Expert parallelism; see the note on _gang_moe_mxfp8_tile. 1/0
          // compiles the remap out and local_eid == expert_id.
          int EP_WORLD_SIZE = 1,
          int EP_MY_PE = 0,
          int EP_NUM_ROUTED = NUM_EXPERTS,
          int EP_SHARED_PE = 0>
__device__ __noinline__ void
    gang_moe_w13_linear_mxfp8_kernel(void const *input_ptr,
                                     void const *weight_ptr,
                                     void const *routing_ptr,
                                     void const *mask_ptr,
                                     void const *bias_ptr,
                                     void *output_ptr,
                                     int tile_idx) {
  static_assert(OUTPUT_PER_WG % 64 == 0 || OUTPUT_PER_WG == 16,
                "OUTPUT_PER_WG is either N-parallel (a multiple of 64 = 4 "
                "waves x 16 rows) or the K-parallel width, 16");
  static_assert(REDUCTION_SIZE % 128 == 0,
                "K must be a multiple of 128 for FP8 MFMA");

  constexpr int NUM_BLOCKS_32 = REDUCTION_SIZE / 32;
  constexpr int W_ROW_BYTES = WEIGHT_FP4 ? REDUCTION_SIZE / 2 : REDUCTION_SIZE;
  constexpr int WG_DATA_BYTES = OUTPUT_PER_WG * W_ROW_BYTES;
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
  // N-parallel splits the workgroup's rows across the waves; K-parallel gives
  // every wave the same 16 rows and splits the reduction, so it has exactly
  // one tile. See the branch below.
  constexpr bool K_PARALLEL = (OUTPUT_PER_WG < 64);
  constexpr int TILES_PER_WAVE =
      K_PARALLEL ? 1 : (OUTPUT_PER_WG / 16 / NUM_WAVES);
  constexpr int FP8_TOK_DATA = REDUCTION_SIZE;

  unsigned short const *A = (unsigned short const *)input_ptr;
  uint8_t const *W = (uint8_t const *)weight_ptr;
  unsigned short const *d_bias = (unsigned short const *)bias_ptr;
  unsigned short *d_output = (unsigned short *)output_ptr;

  int expert_id, local_eid, tok_idx, wg_idx, topk_slot;
  if (!_gang_moe_mxfp8_tile<BATCH_SIZE,
                            NUM_EXPERTS,
                            TILES_PER_EXPERT,
                            EXPERT_WGS,
                            EP_WORLD_SIZE,
                            EP_MY_PE,
                            EP_NUM_ROUTED,
                            EP_SHARED_PE>(tile_idx,
                                          (int const *)mask_ptr,
                                          (int const *)routing_ptr,
                                          &expert_id,
                                          &local_eid,
                                          &tok_idx,
                                          &wg_idx,
                                          &topk_slot)) {
    return;
  }

  uint8_t const *wg_data = W + static_cast<int64_t>(local_eid) * EXPERT_BYTES +
                           static_cast<int64_t>(wg_idx) * WG_BYTES;
  uint8_t const *wg_scales = wg_data + WG_DATA_BYTES;

  extern __shared__ char _gang_moe_mxfp8_smem[];
  uint8_t *s_tok_fp8 = (uint8_t *)_gang_moe_mxfp8_smem;
  uint8_t *s_tok_scales = s_tok_fp8 + FP8_TOK_DATA;

  // K-parallel reduces four waves' partial sums through LDS. Placed AFTER the
  // token scratch rather than aliased over it, which is what the dense MXFP8
  // kernel does: a wave that reaches the reduction while wave 0 is still
  // consuming the low end of s_tok_fp8 would otherwise overwrite k-tiles wave
  // 0 has not read. 256 bytes against a 6.3 KB allocation is not worth the
  // race.
  float *s_reduce =
      (float *)(((uintptr_t)(s_tok_scales + NUM_BLOCKS_32) + 15u) &
                ~(uintptr_t)15);

  int const tid = threadIdx.x;
  int const warp_id = tid >> 6;
  int const lane_id = tid & 63;
  int const col = lane_id & 15; // weight row within the 16x16 MFMA tile
  int const g = lane_id >> 4;   // K-group (0..3)

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  // Slot 1 is otherwise unused. [0]/[1] split this tile into the token-quant
  // prologue and everything after it, [6] counts tiles, so ns/count is a
  // per-tile microsecond figure without needing a worker/layer divisor.
  unsigned long long _sp_q0 = __builtin_amdgcn_s_memrealtime();
#endif
  // Phase 1: quantize this token's activation to FP8 E4M3 in LDS.
  _gang_wave_parallel_fp8_quant<REDUCTION_SIZE>(
      A + static_cast<size_t>(tok_idx) * REDUCTION_SIZE,
      s_tok_fp8,
      s_tok_scales);
#ifdef MPK_ENABLE_SUBPHASE_TIMING
  unsigned long long _sp_q1 = __builtin_amdgcn_s_memrealtime();
  if (tid == 0 && g_subphase_active) {
    atomicAdd(&g_subphase_ns[1][0], (_sp_q1 - _sp_q0) * 10);
    atomicAdd(&g_subphase_ns[1][6], 1ULL);
    atomicAdd(&g_subphase_cnt[1], 1ULL);
  }
#endif

  // Phase 2 epilogue, hoisted out of both parallelization branches. `acc`
  // holds one lane's four output columns, starting at out_base; at one token
  // per tile only col 0 carries a result, so every caller guards on that.
  auto emit = [&](f32x4_t const &acc, int const out_base) {
    unsigned short const *bias_row = d_bias + local_eid * OUTPUT_STRIDE;

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
          float const gate = acc[2 * p] + _gang_bf16_to_float(bias_row[out_n]);
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
  };

  // Phase 2: depth-4 pipelined FP8(weight) x FP8(token) MFMA.
  //
  // Two parallelizations, chosen by OUTPUT_PER_WG. N-parallel is the wide
  // form: the workgroup's rows are dealt out across the four waves. It wants
  // OUTPUT_PER_WG >= 64 and gives the largest tile, which is right when there
  // are more tiles than workers.
  //
  // Under expert parallelism there are not. _gang_moe_mxfp8_tile builds the
  // tile space over the OWNED activated experts, and at EP=8 a rank owns ~2
  // activated experts (one routed of eight, plus the shared expert on
  // EP_SHARED_PE), so a GLM-5 layer has 127 live W13 tiles at
  // OUTPUT_PER_WG=64 -- 15.9 per XCD against 29 workers. K-parallel is the
  // narrow form that was supposed to fix it: all four waves take the same 16
  // rows and split the reduction, so the tile count goes up 4x and each tile
  // was expected to cost a quarter as much. Same trade the dense MXFP8 kernel
  // makes for qkv_a at QKV_MXFP8_OPW=16, and #19 for o_proj.
  //
  // MEASURED NEGATIVE, keep the branch but do not default to it. GLM-5,
  // NP=8 EP, GLM_MOE_W13_OPW=16 vs 64: 13.539 vs 12.196 ms/iter, and
  // SP3[4] MoeW13 2.008 vs 0.923 ms/iter. Per-tile subphase slot 1 says why.
  // The N-parallel OPW=64 tile is 0.86 us of token quant + 13.9 us of body
  // for 208 KB of weight, i.e. 15.0 GB/s against the 20.2 GB/s per-CU share
  // of HBM -- already 74% of roof. The K-parallel tile moves a quarter of
  // those bytes and still costs 12.2 us, 4.0 GB/s. A 12-iteration k-loop
  // cannot keep enough loads in flight to reach the roof, so the split buys
  // 4x the tiles at 4x the cost per byte and the makespan gets worse, not
  // better. The prologue is NOT the fixed cost that would have made
  // subdivision pay: at 0.86 us it is 6% of the tile.
  if constexpr (!K_PARALLEL) {
  for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; tile_iter++) {
    int const wave_tile = warp_id + tile_iter * NUM_WAVES;
    int const w_row = wave_tile * 16 + col;
    uint8_t const *w_data_row =
        wg_data + static_cast<size_t>(w_row) * W_ROW_BYTES;
    int const row_scale_base = w_row * NUM_BLOCKS_32;

    f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

    i32x8_t a0 = _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, 0 * K_PER_MFMA, g);
    int sa0 = (int)wg_scales[row_scale_base + 0 * 4 + g];
    i32x8_t a1 = _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, 1 * K_PER_MFMA, g);
    int sa1 = (int)wg_scales[row_scale_base + 1 * 4 + g];
    i32x8_t a2 = _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, 2 * K_PER_MFMA, g);
    int sa2 = (int)wg_scales[row_scale_base + 2 * 4 + g];
    i32x8_t a3 = _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, 3 * K_PER_MFMA, g);
    int sa3 = (int)wg_scales[row_scale_base + 3 * 4 + g];

// IMPORTANT: #pragma unroll 1 prevents ROCm miscompilation.
#pragma unroll 1
    for (int ki = 0; ki < MFMA_ITERS; ki += 4) {
      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, ki * K_PER_MFMA, g);
        acc = _gang_mfma_w_x_f8<WEIGHT_FP4>(a0, b, acc, sa0,
                                            (int)s_tok_scales[ki]);
      }
      if (ki + 4 < MFMA_ITERS) {
        int kt4 = (ki + 4) * K_PER_MFMA;
        a0 = _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, kt4, g);
        sa0 = (int)wg_scales[row_scale_base + kt4 / 32 + g];
      }

      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 1) * K_PER_MFMA, g);
        acc = _gang_mfma_w_x_f8<WEIGHT_FP4>(a1, b, acc, sa1,
                                            (int)s_tok_scales[ki + 1]);
      }
      if (ki + 5 < MFMA_ITERS) {
        int kt5 = (ki + 5) * K_PER_MFMA;
        a1 = _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, kt5, g);
        sa1 = (int)wg_scales[row_scale_base + kt5 / 32 + g];
      }

      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 2) * K_PER_MFMA, g);
        acc = _gang_mfma_w_x_f8<WEIGHT_FP4>(a2, b, acc, sa2,
                                            (int)s_tok_scales[ki + 2]);
      }
      if (ki + 6 < MFMA_ITERS) {
        int kt6 = (ki + 6) * K_PER_MFMA;
        a2 = _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, kt6, g);
        sa2 = (int)wg_scales[row_scale_base + kt6 / 32 + g];
      }

      if (ki + 3 < MFMA_ITERS) {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 3) * K_PER_MFMA, g);
        acc = _gang_mfma_w_x_f8<WEIGHT_FP4>(a3, b, acc, sa3,
                                            (int)s_tok_scales[ki + 3]);
      }
      if (ki + 7 < MFMA_ITERS) {
        int kt7 = (ki + 7) * K_PER_MFMA;
        a3 = _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, kt7, g);
        sa3 = (int)wg_scales[row_scale_base + kt7 / 32 + g];
      }
    }

    // Epilogue. acc[i] = C[g*4+i][col]; at one token per tile only col 0 holds
    // a result.
    if (col == 0) {
      emit(acc, wg_idx * OUTPUT_PER_WG + wave_tile * 16 + g * 4);
    }
  }
  } else {
    // K-parallel: all four waves take the same 16 output rows and split the
    // reduction between them, then reduce through LDS.
    static_assert(OUTPUT_PER_WG == 16,
                  "the K-parallel branch covers 16 output rows per workgroup");
    constexpr int ITERS_PER_WAVE = MFMA_ITERS / NUM_WAVES;
    static_assert(MFMA_ITERS % NUM_WAVES == 0,
                  "MFMA_ITERS must be divisible by NUM_WAVES for K-parallel");
    // Same reason the N-parallel loop needs MFMA_ITERS % 4 == 0: only slot 3
    // carries a tail guard, so a partial final group would let slots 1 and 2
    // compute k-tiles outside this wave's range.
    static_assert(ITERS_PER_WAVE >= 4 && ITERS_PER_WAVE % 4 == 0,
                  "Depth-4 K-parallel requires ITERS_PER_WAVE a multiple of 4");

    int const ki_start = warp_id * ITERS_PER_WAVE;
    int const ki_end = ki_start + ITERS_PER_WAVE;
    int const w_row = col; // all four waves, same 16 rows
    uint8_t const *w_data_row =
        wg_data + static_cast<size_t>(w_row) * W_ROW_BYTES;
    int const row_scale_base = w_row * NUM_BLOCKS_32;

    f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

    i32x8_t a0 =
        _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, ki_start * K_PER_MFMA, g);
    int sa0 = (int)wg_scales[row_scale_base + ki_start * 4 + g];
    i32x8_t a1 = _gang_load_w_mfma_a<WEIGHT_FP4>(
        w_data_row, (ki_start + 1) * K_PER_MFMA, g);
    int sa1 = (int)wg_scales[row_scale_base + (ki_start + 1) * 4 + g];
    i32x8_t a2 = _gang_load_w_mfma_a<WEIGHT_FP4>(
        w_data_row, (ki_start + 2) * K_PER_MFMA, g);
    int sa2 = (int)wg_scales[row_scale_base + (ki_start + 2) * 4 + g];
    i32x8_t a3 = _gang_load_w_mfma_a<WEIGHT_FP4>(
        w_data_row, (ki_start + 3) * K_PER_MFMA, g);
    int sa3 = (int)wg_scales[row_scale_base + (ki_start + 3) * 4 + g];

// IMPORTANT: #pragma unroll 1 prevents ROCm miscompilation.
#pragma unroll 1
    for (int ki = ki_start; ki < ki_end; ki += 4) {
      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, ki * K_PER_MFMA, g);
        acc = _gang_mfma_w_x_f8<WEIGHT_FP4>(a0, b, acc, sa0,
                                            (int)s_tok_scales[ki]);
      }
      if (ki + 4 < ki_end) {
        int kt4 = (ki + 4) * K_PER_MFMA;
        a0 = _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, kt4, g);
        sa0 = (int)wg_scales[row_scale_base + kt4 / 32 + g];
      }

      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 1) * K_PER_MFMA, g);
        acc = _gang_mfma_w_x_f8<WEIGHT_FP4>(a1, b, acc, sa1,
                                            (int)s_tok_scales[ki + 1]);
      }
      if (ki + 5 < ki_end) {
        int kt5 = (ki + 5) * K_PER_MFMA;
        a1 = _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, kt5, g);
        sa1 = (int)wg_scales[row_scale_base + kt5 / 32 + g];
      }

      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 2) * K_PER_MFMA, g);
        acc = _gang_mfma_w_x_f8<WEIGHT_FP4>(a2, b, acc, sa2,
                                            (int)s_tok_scales[ki + 2]);
      }
      if (ki + 6 < ki_end) {
        int kt6 = (ki + 6) * K_PER_MFMA;
        a2 = _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, kt6, g);
        sa2 = (int)wg_scales[row_scale_base + kt6 / 32 + g];
      }

      if (ki + 3 < ki_end) {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 3) * K_PER_MFMA, g);
        acc = _gang_mfma_w_x_f8<WEIGHT_FP4>(a3, b, acc, sa3,
                                            (int)s_tok_scales[ki + 3]);
      }
      if (ki + 7 < ki_end) {
        int kt7 = (ki + 7) * K_PER_MFMA;
        a3 = _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, kt7, g);
        sa3 = (int)wg_scales[row_scale_base + kt7 / 32 + g];
      }
    }

    if (col == 0) {
#pragma unroll
      for (int i = 0; i < 4; i++) {
        s_reduce[warp_id * OUTPUT_PER_WG + g * 4 + i] = acc[i];
      }
    }
    __syncthreads();

    if (warp_id == 0 && col == 0) {
      f32x4_t sum = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int w = 0; w < NUM_WAVES; w++) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
          sum[i] += s_reduce[w * OUTPUT_PER_WG + g * 4 + i];
        }
      }
      emit(sum, wg_idx * OUTPUT_PER_WG + g * 4);
    }
  }

  __syncthreads();
#ifdef MPK_ENABLE_SUBPHASE_TIMING
  if (tid == 0 && g_subphase_active) {
    atomicAdd(&g_subphase_ns[1][1],
              (__builtin_amdgcn_s_memrealtime() - _sp_q1) * 10);
  }
#endif
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
          bool FUSE_MULSUMADD = false,
          bool WEIGHT_FP4 = false,
          // Expert parallelism; see the note on _gang_moe_mxfp8_tile. 1/0
          // compiles the remap out and local_eid == expert_id.
          int EP_WORLD_SIZE = 1,
          int EP_MY_PE = 0,
          int EP_NUM_ROUTED = NUM_EXPERTS,
          int EP_SHARED_PE = 0>
__device__ __noinline__ void
    gang_moe_w2_linear_mxfp8_kernel(void const *input_ptr,
                                    void const *weight_ptr,
                                    void const *routing_ptr,
                                    void const *mask_ptr,
                                    void const *bias_ptr,
                                    void *output_ptr,
                                    int tile_idx,
                                    void const *routing_weight_ptr = nullptr) {
  static_assert(OUTPUT_PER_WG % 64 == 0 || OUTPUT_PER_WG == 16,
                "OUTPUT_PER_WG is either N-parallel (a multiple of 64 = 4 "
                "waves x 16 rows) or the K-parallel width, 16");
  static_assert(REDUCTION_SIZE % 128 == 0,
                "K must be a multiple of 128 for FP8 MFMA");

  constexpr int NUM_BLOCKS_32 = REDUCTION_SIZE / 32;
  constexpr int W_ROW_BYTES = WEIGHT_FP4 ? REDUCTION_SIZE / 2 : REDUCTION_SIZE;
  constexpr int WG_DATA_BYTES = OUTPUT_PER_WG * W_ROW_BYTES;
  constexpr int WG_SCALE_BYTES = OUTPUT_PER_WG * NUM_BLOCKS_32;
  constexpr int WG_BYTES = WG_DATA_BYTES + WG_SCALE_BYTES;
  constexpr int EXPERT_WGS = OUTPUT_STRIDE / OUTPUT_PER_WG;
  constexpr int64_t EXPERT_BYTES = static_cast<int64_t>(EXPERT_WGS) * WG_BYTES;

  constexpr int K_PER_MFMA = 128;
  constexpr int MFMA_ITERS = REDUCTION_SIZE / K_PER_MFMA;
  static_assert(MFMA_ITERS >= 4 && MFMA_ITERS % 4 == 0,
                "Depth-4 pipeline requires REDUCTION_SIZE % 512 == 0");

  constexpr int NUM_WAVES = 4;
  // See the note on the W13 kernel's branch: K_PARALLEL is the narrow-tile
  // form that keeps all 29 workers per XCD fed when expert parallelism has
  // left the layer with only one owned expert's worth of tiles.
  constexpr bool K_PARALLEL = (OUTPUT_PER_WG < 64);
  constexpr int TILES_PER_WAVE =
      K_PARALLEL ? 1 : (OUTPUT_PER_WG / 16 / NUM_WAVES);
  constexpr int FP8_TOK_DATA = REDUCTION_SIZE;

  unsigned short const *A = (unsigned short const *)input_ptr;
  uint8_t const *W = (uint8_t const *)weight_ptr;
  unsigned short const *d_bias = (unsigned short const *)bias_ptr;
  unsigned short *d_output = (unsigned short *)output_ptr;
  float *d_workspace = (float *)output_ptr;
  float const *d_routing_weight = (float const *)routing_weight_ptr;

  int expert_id, local_eid, tok_idx, wg_idx, topk_slot;
  if (!_gang_moe_mxfp8_tile<BATCH_SIZE,
                            NUM_EXPERTS,
                            TILES_PER_EXPERT,
                            EXPERT_WGS,
                            EP_WORLD_SIZE,
                            EP_MY_PE,
                            EP_NUM_ROUTED,
                            EP_SHARED_PE>(tile_idx,
                                          (int const *)mask_ptr,
                                          (int const *)routing_ptr,
                                          &expert_id,
                                          &local_eid,
                                          &tok_idx,
                                          &wg_idx,
                                          &topk_slot)) {
    return;
  }

  uint8_t const *wg_data = W + static_cast<int64_t>(local_eid) * EXPERT_BYTES +
                           static_cast<int64_t>(wg_idx) * WG_BYTES;
  uint8_t const *wg_scales = wg_data + WG_DATA_BYTES;

  extern __shared__ char _gang_moe_mxfp8_smem[];
  uint8_t *s_tok_fp8 = (uint8_t *)_gang_moe_mxfp8_smem;
  uint8_t *s_tok_scales = s_tok_fp8 + FP8_TOK_DATA;

  // See the W13 kernel: after the token scratch, not aliased over it.
  float *s_reduce =
      (float *)(((uintptr_t)(s_tok_scales + NUM_BLOCKS_32) + 15u) &
                ~(uintptr_t)15);

  int const tid = threadIdx.x;
  int const warp_id = tid >> 6;
  int const lane_id = tid & 63;
  int const col = lane_id & 15;
  int const g = lane_id >> 4;

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  unsigned long long _sp_q0 = __builtin_amdgcn_s_memrealtime();
#endif
  _gang_wave_parallel_fp8_quant_nt<REDUCTION_SIZE>(
      A + static_cast<size_t>(tok_idx) * (NUM_TOPK * REDUCTION_SIZE) +
          static_cast<size_t>(topk_slot) * REDUCTION_SIZE,
      s_tok_fp8,
      s_tok_scales);
#ifdef MPK_ENABLE_SUBPHASE_TIMING
  unsigned long long _sp_q1 = __builtin_amdgcn_s_memrealtime();
  if (tid == 0 && g_subphase_active) {
    atomicAdd(&g_subphase_ns[1][2], (_sp_q1 - _sp_q0) * 10);
    atomicAdd(&g_subphase_ns[1][7], 1ULL);
  }
#endif

  float rw = 0.0f;
  if constexpr (FUSE_MULSUMADD) {
    rw = d_routing_weight[tok_idx * NUM_TOPK + topk_slot];
  }

  // Epilogue, hoisted out of both parallelization branches; see W13.
  auto emit = [&](f32x4_t const &acc, int const out_base) {
    unsigned short const *bias_row = d_bias + local_eid * OUTPUT_STRIDE;

    if constexpr (FUSE_MULSUMADD) {
      // The shared expert rides in routing slot NUM_TOPK-1 with weight 1.0,
      // so it needs no special case here.
      float *ws_addr = d_workspace +
                       static_cast<size_t>(tok_idx) * OUTPUT_STRIDE + out_base;
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
          d_output + static_cast<size_t>(tok_idx) * (NUM_TOPK * OUTPUT_STRIDE) +
          static_cast<size_t>(topk_slot) * OUTPUT_STRIDE + out_base;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        if (out_base + i < OUTPUT_SIZE) {
          out_addr[i] = _gang_float_to_bf16(
              acc[i] + _gang_bf16_to_float(bias_row[out_base + i]));
        }
      }
    }
  };

  if constexpr (!K_PARALLEL) {
  for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; tile_iter++) {
    int const wave_tile = warp_id + tile_iter * NUM_WAVES;
    int const w_row = wave_tile * 16 + col;
    uint8_t const *w_data_row =
        wg_data + static_cast<size_t>(w_row) * W_ROW_BYTES;
    int const row_scale_base = w_row * NUM_BLOCKS_32;

    f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

    i32x8_t a0 = _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, 0 * K_PER_MFMA, g);
    int sa0 = (int)wg_scales[row_scale_base + 0 * 4 + g];
    i32x8_t a1 = _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, 1 * K_PER_MFMA, g);
    int sa1 = (int)wg_scales[row_scale_base + 1 * 4 + g];
    i32x8_t a2 = _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, 2 * K_PER_MFMA, g);
    int sa2 = (int)wg_scales[row_scale_base + 2 * 4 + g];
    i32x8_t a3 = _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, 3 * K_PER_MFMA, g);
    int sa3 = (int)wg_scales[row_scale_base + 3 * 4 + g];

// IMPORTANT: #pragma unroll 1 prevents ROCm miscompilation.
#pragma unroll 1
    for (int ki = 0; ki < MFMA_ITERS; ki += 4) {
      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, ki * K_PER_MFMA, g);
        acc = _gang_mfma_w_x_f8<WEIGHT_FP4>(a0, b, acc, sa0,
                                            (int)s_tok_scales[ki]);
      }
      if (ki + 4 < MFMA_ITERS) {
        int kt4 = (ki + 4) * K_PER_MFMA;
        a0 = _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, kt4, g);
        sa0 = (int)wg_scales[row_scale_base + kt4 / 32 + g];
      }

      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 1) * K_PER_MFMA, g);
        acc = _gang_mfma_w_x_f8<WEIGHT_FP4>(a1, b, acc, sa1,
                                            (int)s_tok_scales[ki + 1]);
      }
      if (ki + 5 < MFMA_ITERS) {
        int kt5 = (ki + 5) * K_PER_MFMA;
        a1 = _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, kt5, g);
        sa1 = (int)wg_scales[row_scale_base + kt5 / 32 + g];
      }

      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 2) * K_PER_MFMA, g);
        acc = _gang_mfma_w_x_f8<WEIGHT_FP4>(a2, b, acc, sa2,
                                            (int)s_tok_scales[ki + 2]);
      }
      if (ki + 6 < MFMA_ITERS) {
        int kt6 = (ki + 6) * K_PER_MFMA;
        a2 = _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, kt6, g);
        sa2 = (int)wg_scales[row_scale_base + kt6 / 32 + g];
      }

      if (ki + 3 < MFMA_ITERS) {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 3) * K_PER_MFMA, g);
        acc = _gang_mfma_w_x_f8<WEIGHT_FP4>(a3, b, acc, sa3,
                                            (int)s_tok_scales[ki + 3]);
      }
      if (ki + 7 < MFMA_ITERS) {
        int kt7 = (ki + 7) * K_PER_MFMA;
        a3 = _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, kt7, g);
        sa3 = (int)wg_scales[row_scale_base + kt7 / 32 + g];
      }
    }

    if (col == 0) {
      emit(acc, wg_idx * OUTPUT_PER_WG + wave_tile * 16 + g * 4);
    }
  }
  } else {
    // K-parallel; see the W13 kernel for the shape argument.
    static_assert(OUTPUT_PER_WG == 16,
                  "the K-parallel branch covers 16 output rows per workgroup");
    constexpr int ITERS_PER_WAVE = MFMA_ITERS / NUM_WAVES;
    static_assert(MFMA_ITERS % NUM_WAVES == 0,
                  "MFMA_ITERS must be divisible by NUM_WAVES for K-parallel");
    static_assert(ITERS_PER_WAVE >= 4 && ITERS_PER_WAVE % 4 == 0,
                  "Depth-4 K-parallel requires ITERS_PER_WAVE a multiple of 4");

    int const ki_start = warp_id * ITERS_PER_WAVE;
    int const ki_end = ki_start + ITERS_PER_WAVE;
    int const w_row = col;
    uint8_t const *w_data_row =
        wg_data + static_cast<size_t>(w_row) * W_ROW_BYTES;
    int const row_scale_base = w_row * NUM_BLOCKS_32;

    f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

    i32x8_t a0 =
        _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, ki_start * K_PER_MFMA, g);
    int sa0 = (int)wg_scales[row_scale_base + ki_start * 4 + g];
    i32x8_t a1 = _gang_load_w_mfma_a<WEIGHT_FP4>(
        w_data_row, (ki_start + 1) * K_PER_MFMA, g);
    int sa1 = (int)wg_scales[row_scale_base + (ki_start + 1) * 4 + g];
    i32x8_t a2 = _gang_load_w_mfma_a<WEIGHT_FP4>(
        w_data_row, (ki_start + 2) * K_PER_MFMA, g);
    int sa2 = (int)wg_scales[row_scale_base + (ki_start + 2) * 4 + g];
    i32x8_t a3 = _gang_load_w_mfma_a<WEIGHT_FP4>(
        w_data_row, (ki_start + 3) * K_PER_MFMA, g);
    int sa3 = (int)wg_scales[row_scale_base + (ki_start + 3) * 4 + g];

// IMPORTANT: #pragma unroll 1 prevents ROCm miscompilation.
#pragma unroll 1
    for (int ki = ki_start; ki < ki_end; ki += 4) {
      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, ki * K_PER_MFMA, g);
        acc = _gang_mfma_w_x_f8<WEIGHT_FP4>(a0, b, acc, sa0,
                                            (int)s_tok_scales[ki]);
      }
      if (ki + 4 < ki_end) {
        int kt4 = (ki + 4) * K_PER_MFMA;
        a0 = _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, kt4, g);
        sa0 = (int)wg_scales[row_scale_base + kt4 / 32 + g];
      }

      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 1) * K_PER_MFMA, g);
        acc = _gang_mfma_w_x_f8<WEIGHT_FP4>(a1, b, acc, sa1,
                                            (int)s_tok_scales[ki + 1]);
      }
      if (ki + 5 < ki_end) {
        int kt5 = (ki + 5) * K_PER_MFMA;
        a1 = _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, kt5, g);
        sa1 = (int)wg_scales[row_scale_base + kt5 / 32 + g];
      }

      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 2) * K_PER_MFMA, g);
        acc = _gang_mfma_w_x_f8<WEIGHT_FP4>(a2, b, acc, sa2,
                                            (int)s_tok_scales[ki + 2]);
      }
      if (ki + 6 < ki_end) {
        int kt6 = (ki + 6) * K_PER_MFMA;
        a2 = _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, kt6, g);
        sa2 = (int)wg_scales[row_scale_base + kt6 / 32 + g];
      }

      if (ki + 3 < ki_end) {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 3) * K_PER_MFMA, g);
        acc = _gang_mfma_w_x_f8<WEIGHT_FP4>(a3, b, acc, sa3,
                                            (int)s_tok_scales[ki + 3]);
      }
      if (ki + 7 < ki_end) {
        int kt7 = (ki + 7) * K_PER_MFMA;
        a3 = _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, kt7, g);
        sa3 = (int)wg_scales[row_scale_base + kt7 / 32 + g];
      }
    }

    if (col == 0) {
#pragma unroll
      for (int i = 0; i < 4; i++) {
        s_reduce[warp_id * OUTPUT_PER_WG + g * 4 + i] = acc[i];
      }
    }
    __syncthreads();

    if (warp_id == 0 && col == 0) {
      f32x4_t sum = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int w = 0; w < NUM_WAVES; w++) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
          sum[i] += s_reduce[w * OUTPUT_PER_WG + g * 4 + i];
        }
      }
      emit(sum, wg_idx * OUTPUT_PER_WG + g * 4);
    }
  }

  __syncthreads();
#ifdef MPK_ENABLE_SUBPHASE_TIMING
  if (tid == 0 && g_subphase_active) {
    atomicAdd(&g_subphase_ns[1][3],
              (__builtin_amdgcn_s_memrealtime() - _sp_q1) * 10);
  }
#endif
}

} // namespace kernel
