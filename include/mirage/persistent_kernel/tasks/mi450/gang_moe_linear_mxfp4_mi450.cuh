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

// Gang MoE MXFP4 linear kernel for MI450 (gfx1250).
//
// Port of tasks/mi300/gang_moe_linear_mxfp4_mi300.cuh. The datapath survives
// intact -- both parts do 16x16x128 f8f6f4 with hardware dequant -- but the
// wave32/WMMA fragment layout is not a rescaling of the wave64/MFMA one, so
// every index in the inner loop and the epilogue changed. The layout was
// measured on FFM-Lite and is documented in mx_layout_mi450.cuh; this file
// only consumes it.
//
// WHAT CHANGED FROM THE gfx950 VERSION
//
// 1. 8 waves, not 4. A 256-thread workgroup is 8 wave32s. NUM_WAVES=8 halves
//    TILES_PER_WAVE, so each wave now owns one 16-row output tile at the
//    OUTPUT_PER_WG=128 shape instead of two.
//
// 2. K is interleaved across the lane halves, so the single contiguous
//    16-byte weight load
//
//        i32x8_t a_reg = *(i32x8_t const *)(wg_data + w_row*(K/2) + kt/2 + g*16);
//
//    becomes four 8-byte loads (mx_load_operand_fp4). gfx950's four K-groups
//    of 32 (g = lane>>4) are replaced by two K-halves of 2x32 each.
//
// 3. Scales are packed, not per-lane-scalar. gfx950 passes one E8M0 byte
//    chosen by the lane's own K-group. gfx1250 takes all four blocks' bytes
//    in one 32-bit operand, and reads it only from lanes 0..15.
//
//    This operand must stay lane-varying. If it becomes wave-uniform the
//    compiler puts it in an SGPR and the hardware then reads only bits [7:0],
//    silently applying block 0's scale to all 128 K elements. Indexing by
//    w_row (which depends on the lane) is what keeps it in a VGPR.
//
// 4. The accumulator is 8 wide and holds a column strip:
//        gfx950   acc[i] = C[g*4 + i][col]    4 lanes x 4 rows
//        gfx1250  acc[i] = C[8*(lane/16) + i][lane%16]
//    With BATCH_SIZE=1 only column 0 is live either way, but on gfx1250 that
//    is lanes 0 and 16 covering rows 0..7 and 8..15, rather than four lanes.
//
// OPTIMIZATIONS, and why each one is here
//
// All four were driven by reading the emitted gfx1250 ISA, not by assumption,
// and each was re-validated on FFM afterward. MI400 Shader Programming Guide
// sections cited inline at each site.
//
// a. GLOBAL rather than FLAT addressing (S4.9.5) for the weights, scales and
//    activation. These arrive as `void const *` kernel arguments, so the
//    compiler cannot prove the address space and defaulted to FLAT -- which
//    issues down the VMEM *and* LDS paths and stalls until both are free, for
//    data that is never in LDS. First build: flat_load_b128 for A. Now:
//    global_load_b128. Free; no register cost.
//
// b. Software pipelining by one K-tile, with two register buffers ping-ponged.
//    The original loop was strictly load -> s_wait_loadcnt 0 -> WMMA -> branch,
//    i.e. fully exposed memory latency on every tile. `#pragma unroll 2` did
//    not help -- the compiler just emitted two serial copies -- because it
//    cannot sink the wait past a WMMA that consumes the loaded registers.
//
// c. GLOBAL_PREFETCH_B8 one tile further ahead (S4.9.6). Returns no data and
//    touches no counter, so it needs no wait; bounded so it never runs off the
//    end of the weight buffer.
//
// d. Dword LDS stores in the quantizer instead of byte stores. Note this one
//    is deliberately NOT the widest available store: buffering all 4 dwords for
//    a single b128 cost 44 extra VGPRs (86 -> 130), which at 16-VGPR
//    granularity is 10 waves/SIMD down to 7. The dword form keeps 88 VGPRs and
//    the original occupancy. Details at the site.
//
// Net: 88 VGPRs (was 84 before any of this), same 96-VGPR/10-wave occupancy
// tier, with the loop latency overlapped rather than exposed.
//
// WHAT FFM CAN AND CANNOT TELL US HERE
//
// Validated on FFM-Lite by tests/mi450/test_moe_linear_mxfp4.hip: the naive
// and pipelined loops both reproduce an independent host reference exactly,
// and agree with each other bit for bit, at K=512 (4 K-tiles, even) and K=384
// (3 K-tiles, odd -- the ping-pong tail). Bit-exactness is the right bar
// because the optimizations reorder memory, never arithmetic.
//
// NOT VALIDATED, and not validatable here:
//   - Every performance claim above. FFM-Lite is functional only: no cycles,
//     no bandwidth, no contention. The ISA-level facts (which instruction was
//     emitted, how many VGPRs) are real and checked; that they are *faster* is
//     inference from the guide and needs silicon to confirm.
//   - The 240-way XCD barrier and cross-XCD tile distribution. FFM runs
//     workgroups serially, so it will happily pass a barrier that would
//     deadlock on hardware.
//   - mirage::arch::xcd_id(). gfx1250 has no HW_REG_XCC_ID; the s_sendmsg
//     replacement assembles but FFM cannot decode it. It is also far more
//     expensive than gfx950's s_getreg, so it is read once here and hoisted.
//     See arch_traits.cuh.

#pragma once
#include "mirage/persistent_kernel/arch_traits.cuh"
#include "mirage/persistent_kernel/tasks/mi450/mx_layout_mi450.cuh"
#include "tasks/common/common_header.cuh"
#include "tasks/mi300/swigluoai_mi300.cuh"
#include <hip/hip_bf16.h>

namespace kernel {

#if defined(MIRAGE_ARCH_GFX1250)

using kernel::mi450::mx_encode_e2m1;

// ── bf16 <-> float ────────────────────────────────────────────────────────

__device__ __forceinline__ float _gang_bf16_to_float_mi450(unsigned short b) {
  union {
    float f;
    unsigned u;
  } v;
  v.u = ((unsigned)(unsigned short)b) << 16;
  return v.f;
}

__device__ __forceinline__ unsigned short
    _gang_float_to_bf16_mi450(float f) {
  union {
    float f;
    unsigned u;
  } v;
  v.f = f;
  unsigned rounding_bias = ((v.u >> 16) & 1) + 0x7FFF;
  return (unsigned short)((v.u + rounding_bias) >> 16);
}

// E8M0 block scale for FP4 (max representable magnitude 6.0). Multiply by
// reciprocal rather than divide, same as gfx950.
__device__ __forceinline__ uint8_t _gang_compute_e8m0_fp4_mi450(float amax) {
  if (amax == 0.0f) {
    return 0;
  }
  union {
    float f;
    uint32_t u;
  } v;
  v.f = amax * (1.0f / 6.0f);
  int raw_exp = (int)((v.u >> 23) & 0xFF);
  if (v.u & 0x7FFFFF) {
    raw_exp++; // round up if mantissa non-zero
  }
  return (uint8_t)max(0, min(255, raw_exp));
}

// ── FP4 token quantization ────────────────────────────────────────────────
//
// Writes the token in plain row-major nibble-packed form -- the same layout
// gfx950 produces. The WMMA-specific interleave is applied at load time by
// mx_load_operand_fp4(), not here, so the LDS image stays a straightforward
// [K/2 bytes data | K/32 bytes scales] that other kernels can also consume.
//
// gfx1250 has no v_cvt_scalef32_pk_fp4_f32; the packing is done with the
// pk8 converter's inverse, i.e. explicit encode. Kept scalar and simple: this
// is not the hot loop, and correctness of the encode is what the WMMA path
// depends on.
template <int REDUCTION_SIZE>
__device__ __forceinline__ void
    _gang_fp4_quant_mi450(unsigned short const *__restrict__ src_bf16,
                          uint8_t *__restrict__ s_tok_fp4,
                          uint8_t *__restrict__ s_tok_scales) {
  constexpr int BLOCK_SIZE = 32;
  constexpr int NBLOCKS = REDUCTION_SIZE / BLOCK_SIZE;
  int const tid = threadIdx.x;

  // The activation is a kernel argument, so the compiler defaults to FLAT for
  // it exactly as it did for the weights. Flat issues down the VMEM and LDS
  // paths at once and stalls until both are free (guide S4.9.5), for data
  // never in LDS. Each thread reads 64 contiguous bytes here, so this is the
  // one place outside the K-loop where it is worth the cast.
  using g_ushort = __attribute__((address_space(1))) unsigned short const;
  g_ushort *g_src = (g_ushort *)src_bf16;

  for (int blk = tid; blk < NBLOCKS; blk += blockDim.x) {
    int base = blk * BLOCK_SIZE;

    float vals[32];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 32; j++) {
      vals[j] = _gang_bf16_to_float_mi450(g_src[base + j]);
      amax = fmaxf(amax, fabsf(vals[j]));
    }

    uint8_t se = _gang_compute_e8m0_fp4_mi450(amax);
    s_tok_scales[blk] = se;

    float inv_scale;
    if (se == 0) {
      inv_scale = 1.0f;
    } else {
      union {
        float f;
        uint32_t u;
      } sv;
      // 2^-(se-127): reciprocal of the block scale, built by exponent
      // arithmetic so no division is emitted.
      sv.u = (uint32_t)(254 - se) << 23;
      inv_scale = sv.f;
    }

    // Pack 8 nibbles at a time and store a dword, rather than a byte at a
    // time. The original wrote s_tok_fp4[] byte by byte and the compiler
    // faithfully emitted 16 ds_store_b8 per block -- it cannot merge them
    // itself, since it must assume the byte writes may alias.
    //
    // Dwords, not one b128, and the reason is measured. Buffering all four
    // dwords for a single 16-byte store keeps them live simultaneously and
    // pushed the kernel from 86 to 130 VGPRs. VGPRs are allocated in blocks of
    // 16 for wave32 out of 1024 per SIMD (guide S3.3.2.1), so that is 96 -> 144
    // allocated, i.e. 10 waves/SIMD down to 7 -- a 30% occupancy loss to save
    // three LDS instructions in code that runs once per token, not per K-tile.
    // Storing each dword as it is built costs 88 VGPRs, which still rounds into
    // the same 96-VGPR/10-wave tier as the original.
    //
    // Bank behaviour (guide S4.7.1: 64 banks x 4 B): thread `blk` writes 16
    // bytes at byte offset blk*16, so its four dwords hit banks 4*blk..4*blk+3
    // (mod 64) and each group of 16 consecutive threads covers all 64 banks
    // exactly once. base/2 = blk*16 is 4-byte aligned for every blk, so the
    // dword store is always legal.
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      uint32_t acc_w = 0;
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        int j = w * 8 + b * 2;
        uint8_t lo = mx_encode_e2m1(vals[j] * inv_scale);
        uint8_t hi = mx_encode_e2m1(vals[j + 1] * inv_scale);
        acc_w |= (uint32_t)(uint8_t)(lo | (hi << 4)) << (8 * b);
      }
      *(uint32_t *)(s_tok_fp4 + base / 2 + w * 4) = acc_w;
    }
  }
  __syncthreads();
}

// ── Gang MoE MXFP4 kernel (gfx1250 hardware FP4xFP4 WMMA path) ────────────
//
// FUSE_SWIGLU: when true (only valid with W13_LINEAR=true), fuses SwiGLU into
// the epilogue. Output rows alternate gate/up, so consecutive accumulator
// slots pair up -- which is still true on gfx1250, because a lane's 8 slots
// are 8 *consecutive* output rows.
template <int BATCH_SIZE,
          int OUTPUT_SIZE,
          int OUTPUT_STRIDE,
          int REDUCTION_SIZE,
          int NUM_EXPERTS,
          int NUM_TOPK,
          int TILES_PER_EXPERT,
          int OUTPUT_PER_WG,
          bool W13_LINEAR,
          bool FUSE_SWIGLU = false>
__device__ __noinline__ void
    gang_moe_linear_mxfp4_kernel_mi450(void const *input_ptr,
                                       void const *weight_ptr,
                                       void const *routing_ptr,
                                       void const *mask_ptr,
                                       void const *bias_ptr,
                                       void *output_ptr,
                                       int tile_idx) {
  using namespace kernel::mi450;

  static_assert(OUTPUT_PER_WG % 16 == 0,
                "OUTPUT_PER_WG must be multiple of 16");
  static_assert(REDUCTION_SIZE % 128 == 0,
                "REDUCTION_SIZE must be multiple of 128 for FP4 WMMA");

  // MXFP4 weight layout constants (byte-identical to gfx950 -- the weight
  // files do not need repacking for this port).
  constexpr int NUM_BLOCKS_32 = REDUCTION_SIZE / 32;
  constexpr int WG_DATA_BYTES = OUTPUT_PER_WG * (REDUCTION_SIZE / 2);
  constexpr int WG_SCALE_BYTES = OUTPUT_PER_WG * NUM_BLOCKS_32;
  constexpr int WG_BYTES = WG_DATA_BYTES + WG_SCALE_BYTES;
  constexpr int EXPERT_WGS = OUTPUT_STRIDE / OUTPUT_PER_WG;
  constexpr int64_t EXPERT_BYTES = static_cast<int64_t>(EXPERT_WGS) * WG_BYTES;

  constexpr int K_PER_WMMA = MX_K_PER_TILE; // 128

  // 8 wave32s per 256-thread workgroup, versus 4 wave64s on gfx950.
  constexpr int NUM_WAVES = 8;
  constexpr int N_TILES = OUTPUT_PER_WG / 16;
  static_assert(N_TILES % NUM_WAVES == 0,
                "OUTPUT_PER_WG/16 must divide evenly across 8 wave32s; "
                "gfx950 assumed 4 waves and some shapes only satisfy that");
  constexpr int TILES_PER_WAVE = N_TILES / NUM_WAVES;
  constexpr int N_TILES_PER_WG = EXPERT_WGS;

  constexpr int FP4_TOK_DATA = REDUCTION_SIZE / 2;
  constexpr int FP4_TOK_SCALES = REDUCTION_SIZE / 32;

  unsigned short const *A = (unsigned short const *)input_ptr;
  uint8_t const *W = (uint8_t const *)weight_ptr;
  int const *d_routing = (int const *)routing_ptr;
  int const *d_mask = (int const *)mask_ptr;
  unsigned short const *d_bias = (unsigned short const *)bias_ptr;
  unsigned short *d_output = (unsigned short *)output_ptr;

  extern __shared__ char _gang_mxfp4_smem[];
  uint8_t *s_tok_fp4 = (uint8_t *)_gang_mxfp4_smem;
  uint8_t *s_tok_scales = s_tok_fp4 + FP4_TOK_DATA;

  int const tid = threadIdx.x;
  int const warp_id = tid >> 5;  // wave ID (0..7)   -- was tid >> 6
  int const lane_id = tid & 31;  // lane within wave -- was tid & 63
  int const col = mx_operand_mn(lane_id);   // lane_id & 15
  int const half = mx_operand_half(lane_id); // lane_id >> 4, K-half not K-group

  // gfx1250 has no HW_REG_XCC_ID; this goes through sendmsg instead. It is
  // also far more expensive than gfx950's s_getreg and is UNVALIDATED (FFM
  // cannot decode the instruction). See arch_traits.cuh.
  int const my_xcd = mirage::arch::xcd_id();
  int const num_activated_experts = d_mask[NUM_EXPERTS];

  int global_tile = tile_idx * 8 + my_xcd;
  int total_tiles = num_activated_experts * TILES_PER_EXPERT;
  if (global_tile >= total_tiles) {
    return;
  }
  int expert_idx = global_tile / TILES_PER_EXPERT;
  int tile_within_expert = global_tile % TILES_PER_EXPERT;
  int expert_id = d_mask[expert_idx];
  int const *expert_routing = d_routing + expert_id * BATCH_SIZE;

  uint8_t const *expert_weight =
      W + static_cast<int64_t>(expert_id) * EXPERT_BYTES;

  int tok_idx = tile_within_expert / N_TILES_PER_WG;
  int wg_idx = tile_within_expert % N_TILES_PER_WG;

  if (tok_idx >= BATCH_SIZE) {
    return;
  }

  int route_val = expert_routing[tok_idx];
  if (route_val == 0) {
    return;
  }
  int topk_slot = route_val - 1;

  uint8_t const *wg_data =
      expert_weight + static_cast<int64_t>(wg_idx) * WG_BYTES;
  uint8_t const *wg_scales = wg_data + WG_DATA_BYTES;

  // ── Phase 1: load BF16 input, quantize to FP4 in shared memory ─────────
  unsigned short const *input_base;
  if constexpr (W13_LINEAR) {
    input_base = A + tok_idx * REDUCTION_SIZE;
  } else {
    input_base =
        A + tok_idx * (NUM_TOPK * REDUCTION_SIZE) + topk_slot * REDUCTION_SIZE;
  }

  _gang_fp4_quant_mi450<REDUCTION_SIZE>(input_base, s_tok_fp4, s_tok_scales);

  // ── Phase 2: WMMA FP4(weights) x FP4(tokens) ──────────────────────────
  for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; tile_iter++) {
    int wave_tile = warp_id + tile_iter * NUM_WAVES;

    mx_f32x8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    // The weight row this lane *supplies*. Note it is not the row this lane's
    // accumulator ends up holding -- WMMA is cooperative, and the epilogue
    // below uses mx_acc_row() for that. Conflating the two is the classic way
    // to get a plausible-looking but transposed result.
    int const w_row = wave_tile * 16 + col;

    // Weights are a kernel argument (`void const *`), so the compiler cannot
    // prove the address space and defaults to FLAT. That was visible in the
    // first build of this file: `flat_load_b128` for A, `ds_load_b128` for B.
    // Flat issues down both the VMEM and LDS paths and stalls until BOTH are
    // free (guide S4.9.5), for data that is never in LDS. The cast makes it
    // GLOBAL_LOAD. Safe here because wg_data/wg_scales are derived from
    // weight_ptr, which is device global memory.
    mx_global_u8 *g_data = mx_to_global(wg_data);
    mx_global_u8 *g_scales = mx_to_global(wg_scales);

    // Software-pipelined by one K-tile: the loads for tile i+1 are issued
    // before the WMMA for tile i, so the load latency overlaps the matrix op
    // instead of sitting in front of it.
    //
    // This is written out by hand rather than left to `#pragma unroll`. An
    // earlier version used `#pragma unroll 2` and the compiler produced two
    // copies of a strictly serial `load -> s_wait_loadcnt 0 -> WMMA` body in
    // separate basic blocks -- twice the code for none of the overlap. The
    // dependence that stops the scheduler from doing this itself is the
    // s_wait: it cannot sink a wait past a WMMA that consumes the loaded
    // registers unless the *next* tile's registers are distinct, which only
    // an explicit second buffer provides.
    //
    // Register cost is one extra A (8 live VGPRs), one extra B (8), and two
    // scale VGPRs.
    //
    // NOTE ON WMMA HAZARDS (guide S4.6.12.1): back-to-back WMMAs here need no
    // V_NOP padding. The hazard table's RAW case is "Matrix A/B/Index same as
    // previous instruction's Matrix D". This chain feeds D into the *C*
    // operand of the next WMMA, which is the ordinary accumulate path and is
    // not a listed hazard. The epilogue's VALU reads of acc do hit the
    // 1-V_NOP case for `F8F6F4 with SRCA & SRCB == F4`, and the compiler's
    // hazard recognizer handles it -- conservatively, in fact: it emits 4
    // V_NOPs, the figure for the `SRCA | SRCB != F4` row, apparently without
    // tracking the matrix_a_fmt/matrix_b_fmt modifiers. Over-padding is safe,
    // so it is left alone rather than worked around.
    // The two register buffers are ping-ponged rather than rotated. A single
    // `cur`/`nxt` pair expressed the pipeline correctly but made the compiler
    // emit ~26 v_dual_mov_b32 per iteration copying nxt into cur -- more VALU
    // traffic than the WMMA it was meant to be hiding. Stepping two K-tiles
    // per trip makes each buffer's role constant within the body, and the
    // copies vanish.
    {
      constexpr int K_ITERS = REDUCTION_SIZE / K_PER_WMMA;
      static_assert(K_ITERS >= 2,
                    "pipelined K-loop assumes at least 2 K-tiles");

      mx_i32x16_t a0, a1, b0, b1;
      unsigned int sa0, sa1, sb0, sb1;

      // Prologue: tile 0 in flight.
      a0 = mx_load_operand_fp4_global(g_data, w_row, 0, REDUCTION_SIZE);
      b0 = mx_load_operand_fp4(s_tok_fp4, 0, 0, REDUCTION_SIZE);
      sa0 = mx_pack_scales_global(g_scales + w_row * NUM_BLOCKS_32);
      sb0 = mx_pack_scales(s_tok_scales);

#pragma unroll 1
      for (int kt = 0; kt + 2 * K_PER_WMMA <= REDUCTION_SIZE;
           kt += 2 * K_PER_WMMA) {
        int const k1 = kt + K_PER_WMMA;
        int const k2 = kt + 2 * K_PER_WMMA;

        // Issue tile k1 while tile kt is still only in flight.
        a1 = mx_load_operand_fp4_global(g_data, w_row, k1, REDUCTION_SIZE);
        b1 = mx_load_operand_fp4(s_tok_fp4, 0, k1, REDUCTION_SIZE);
        sa1 = mx_pack_scales_global(g_scales + w_row * NUM_BLOCKS_32 + k1 / 32);
        sb1 = mx_pack_scales(s_tok_scales + k1 / 32);

        // Pull tile k2 toward the WGP cache. GLOBAL_PREFETCH_B8 returns no
        // data and touches no counter, so it needs no wait. Bounded, because
        // a prefetch past the end of the weight buffer still issues a real
        // address translation and a UTC fault on it is reported to the host
        // (guide S4.9.6).
        if (k2 < REDUCTION_SIZE) {
          mx_prefetch_operand_fp4_global(g_data, w_row, k2, REDUCTION_SIZE);
        }

        acc = _gang_wmma_f4xf4(a0, b0, acc, (int)sa0, (int)sb0);

        // Refill buffer 0 with tile k2, consumed on the next trip.
        if (k2 < REDUCTION_SIZE) {
          a0 = mx_load_operand_fp4_global(g_data, w_row, k2, REDUCTION_SIZE);
          b0 = mx_load_operand_fp4(s_tok_fp4, 0, k2, REDUCTION_SIZE);
          sa0 =
              mx_pack_scales_global(g_scales + w_row * NUM_BLOCKS_32 + k2 / 32);
          sb0 = mx_pack_scales(s_tok_scales + k2 / 32);
        }

        acc = _gang_wmma_f4xf4(a1, b1, acc, (int)sa1, (int)sb1);
      }

      // Odd tail. K_ITERS is even for every shape Fleet currently registers
      // (GPT-OSS pads hidden 2880 -> 3072, giving 24 tiles), but the kernel
      // only requires REDUCTION_SIZE % 128 == 0, so an odd count is legal and
      // must not silently drop its last tile.
      if constexpr (K_ITERS % 2 == 1) {
        acc = _gang_wmma_f4xf4(a0, b0, acc, (int)sa0, (int)sb0);
      }
    }

    // ── Epilogue ────────────────────────────────────────────────────────
    // acc[i] = C[8*(lane/16) + i][col]. With BATCH_SIZE=1 only col 0 is a
    // real token, so lanes 0 and 16 carry all 16 output rows between them.
    if (col == 0) {
      if constexpr (FUSE_SWIGLU) {
        constexpr int ACT_STRIDE = OUTPUT_STRIDE / 2;
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
          int row = mx_acc_row(lane_id, i); // 8*half + i
          int out_n = wg_idx * OUTPUT_PER_WG + wave_tile * 16 + row;
          if (out_n + 1 < OUTPUT_SIZE) {
            unsigned bt_g = (unsigned)d_bias[expert_id * OUTPUT_STRIDE + out_n]
                            << 16;
            unsigned bt_u =
                (unsigned)d_bias[expert_id * OUTPUT_STRIDE + out_n + 1] << 16;
            float bias_g;
            __builtin_memcpy(&bias_g, &bt_g, 4);
            float bias_u;
            __builtin_memcpy(&bias_u, &bt_u, 4);

            float activated =
                fast_swigluoai(acc[i] + bias_g, acc[i + 1] + bias_u);

            int act_n = out_n / 2;
            int out_idx = tok_idx * (NUM_TOPK * ACT_STRIDE) +
                          topk_slot * ACT_STRIDE + act_n;
            d_output[out_idx] = _gang_float_to_bf16_mi450(activated);
          }
        }
      } else {
#pragma unroll
        for (int i = 0; i < 8; i++) {
          int row = mx_acc_row(lane_id, i);
          int out_n = wg_idx * OUTPUT_PER_WG + wave_tile * 16 + row;
          if (out_n < OUTPUT_SIZE) {
            float bias_val;
            unsigned bt = (unsigned)d_bias[expert_id * OUTPUT_STRIDE + out_n]
                          << 16;
            __builtin_memcpy(&bias_val, &bt, 4);

            float val = acc[i] + bias_val;
            int out_idx = tok_idx * (NUM_TOPK * OUTPUT_STRIDE) +
                          topk_slot * OUTPUT_STRIDE + out_n;
            d_output[out_idx] = _gang_float_to_bf16_mi450(val);
          }
        }
      }
    }
  }

  __syncthreads();
}

#endif // MIRAGE_ARCH_GFX1250

} // namespace kernel
