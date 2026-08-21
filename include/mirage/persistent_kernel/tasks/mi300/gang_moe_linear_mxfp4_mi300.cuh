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

// Gang MoE MXFP4 linear kernel for MI350 (gfx950).
//
// MPK_MOE_KBATCH: how many K-trips of the MFMA reduction load their operands
// before any of them is consumed. 1 keeps the original one-load-in-flight
// loop. See the block comment at the K-reduction loop.
#ifndef MPK_MOE_KBATCH
#define MPK_MOE_KBATCH 1
#endif
//
// Uses hardware-accelerated FP4×FP8 MFMA:
// __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4
// - Weights: FP4 E2M1 with E8M0 block scales (pre-packed, read from HBM)
// - Tokens:  BF16 input, quantized on-the-fly to FP8 E4M3 in shared memory
// - K=128 elements per MFMA instruction (vs 16 in the old bf16 path)
// - Hardware handles dequant + scale application + multiply-accumulate
//
// Weight format: MXFP4 packed per workgroup (same layout as before):
//   [E, expert_wgs, wg_bytes] where wg_bytes = OPW*(K/2) + OPW*(K/32)
//
// Dispatch: 8 gang tasks (1 per XCD), tiles spread across all XCDs:
//   global_tile = tile_idx * 8 + xcd_id  (interleaved round-robin)
//   expert_idx = global_tile / TILES_PER_EXPERT
//   tile_within_expert = global_tile % TILES_PER_EXPERT

#pragma once
#include "tasks/common/common_header.cuh"
#include "tasks/mi300/swigluoai_mi300.cuh"
#include <hip/hip_bf16.h>

namespace kernel {

// ── FP8 E4M3 quantization types and helpers (gfx950) ──────────────────────
typedef int __attribute__((ext_vector_type(4))) i32x4_t;
typedef int __attribute__((ext_vector_type(8))) i32x8_t;
typedef float __attribute__((ext_vector_type(4))) f32x4_t;
typedef short __attribute__((ext_vector_type(2))) fp8x4_t;

// Direct HBM→LDS via MUBUF buffer_load_dwordx4 lds:1
extern "C" __device__ void __llvm_amdgcn_raw_buffer_load_lds(
    i32x4_t rsrc,
    __attribute__((address_space(3))) uint32_t *lds_ptr,
    int32_t size,
    int32_t voffset,
    int32_t soffset,
    int32_t imm_offset,
    int32_t aux) __asm("llvm.amdgcn.raw.buffer.load.lds");

// VGPR-target raw buffer load (16 bytes / dwordx4)
extern "C" __device__ i32x4_t __llvm_amdgcn_raw_buffer_load_v4i32(
    i32x4_t rsrc,
    int32_t voffset,
    int32_t soffset,
    int32_t aux) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

// Wave-uniform buffer resource descriptor (V#)
__device__ __forceinline__ i32x4_t make_w_buffer_rsrc(void const *base,
                                                      uint32_t range_bytes) {
  i32x4_t r;
  uint64_t addr = reinterpret_cast<uint64_t>(base);
  r[0] = static_cast<int>(__builtin_amdgcn_readfirstlane(
      static_cast<uint32_t>(addr & 0xFFFFFFFFu)));
  r[1] = static_cast<int>(
      __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(addr >> 32)));
  r[2] = static_cast<int>(__builtin_amdgcn_readfirstlane(range_bytes));
  r[3] = static_cast<int>(0x00020000u);
  return r;
}

// Convert bf16 (as unsigned short) to float
__device__ __forceinline__ float _gang_bf16_to_float(unsigned short b) {
  union {
    float f;
    unsigned u;
  } v;
  v.u = ((unsigned)(unsigned short)b) << 16;
  return v.f;
}

// Convert float to bf16 (as unsigned short) with rounding
__device__ __forceinline__ unsigned short _gang_float_to_bf16(float f) {
  union {
    float f;
    unsigned u;
  } v;
  v.f = f;
  unsigned rounding_bias = ((v.u >> 16) & 1) + 0x7FFF;
  return (unsigned short)((v.u + rounding_bias) >> 16);
}

// Compute E8M0 block scale for FP8 E4M3 (max value 448)
__device__ __forceinline__ uint8_t _gang_compute_e8m0_fp8(float amax) {
  if (amax == 0.0f) {
    return 0;
  }
  float target = amax * (1.0f / 448.0f);
  union {
    float f;
    uint32_t u;
  } v;
  v.f = target;
  int raw_exp = (int)((v.u >> 23) & 0xFF);
  if (v.u & 0x7FFFFF) {
    raw_exp++; // round up if mantissa non-zero
  }
  return (uint8_t)max(0, min(255, raw_exp));
}

// Quantize 4 floats to packed FP8 E4M3 using hardware cvt instruction
__device__ __forceinline__ fp8x4_t
    _gang_quant_4xfp8(float v0, float v1, float v2, float v3, float scale_f) {
  fp8x4_t pk = {};
  pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(pk, v0, v1, scale_f, false);
  pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(pk, v2, v3, scale_f, true);
  return pk;
}

// Quantize a block of BF16 values to FP8 E4M3 with E8M0 block scale
// n must be 128 (K_PER_MFMA)
__device__ __forceinline__ void
    _gang_quant_bf16_block_fp8(unsigned short const *__restrict__ src_bf16,
                               uint8_t *__restrict__ data_out,
                               uint8_t *__restrict__ scale_out,
                               int n) {
  float amax = 0.0f;
  for (int j = 0; j < n; j++) {
    float v = _gang_bf16_to_float(src_bf16[j]);
    float av = v < 0.0f ? -v : v;
    amax = amax > av ? amax : av;
  }

  uint8_t se = _gang_compute_e8m0_fp8(amax);
  *scale_out = se;
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

  for (int j = 0; j < n; j += 4) {
    float v0 = _gang_bf16_to_float(src_bf16[j]);
    float v1 = _gang_bf16_to_float(src_bf16[j + 1]);
    float v2 = _gang_bf16_to_float(src_bf16[j + 2]);
    float v3 = _gang_bf16_to_float(src_bf16[j + 3]);
    fp8x4_t pk = _gang_quant_4xfp8(v0, v1, v2, v3, scale_f);
    *(fp8x4_t *)(data_out + j) = pk;
  }
}

// Per-thread FP8 quantization mirroring FP4 structure.
// Splits each 128-element MFMA block into 4 × 32-element sub-blocks.
// Each thread handles one sub-block: load 32 bf16, find local amax,
// shuffle with 3 neighbors to get 128-element block amax, compute scale,
// pack 32 values to FP8. Single iteration for K=3072 (96 sub-blocks < 256
// threads).
template <int REDUCTION_SIZE>
__device__ __forceinline__ void
    _gang_wave_parallel_fp8_quant(unsigned short const *__restrict__ src_bf16,
                                  uint8_t *__restrict__ s_tok_fp8,
                                  uint8_t *__restrict__ s_tok_scales) {

  constexpr int SUB_BLOCK = 32;
  constexpr int NSUBBLOCKS = REDUCTION_SIZE / SUB_BLOCK; // 96 for K=3072
  int const tid = threadIdx.x;
  int const lane_id = tid & 63;

  for (int sb = tid; sb < NSUBBLOCKS; sb += blockDim.x) {
    int const base = sb * SUB_BLOCK;
    int const super_blk = sb / 4; // which 128-element block
    int const sub_idx = sb & 3;   // which sub-block within super-block

    // Load 32 bf16 values and find local amax (identical to FP4 quant)
    float vals[32];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 32; j++) {
      vals[j] = _gang_bf16_to_float(src_bf16[base + j]);
      amax = fmaxf(amax, fabsf(vals[j]));
    }

    // Combine amaxes from 4 sub-blocks sharing the same 128-element
    // super-block. Threads sb, sb+1, sb+2, sb+3 are consecutive lanes in the
    // same wave. Use __shfl to read each neighbor's amax (4 reads, 3 fmaxf).
    //
    // The partner index is clamped for the same reason as in the NT variant
    // below: NSUBBLOCKS = REDUCTION_SIZE/32 need not be a multiple of 4, so
    // the tail super-block would otherwise reduce against lanes whose loop
    // condition failed and whose `amax` register was never written.
    int base_lane = lane_id & ~3; // round down to group of 4
    int const sb_first = sb - sub_idx;
    int const n_valid = min(4, (NSUBBLOCKS - 1) - sb_first + 1);
    float a0 = __shfl(amax, base_lane);
    float a1 = __shfl(amax, base_lane + min(1, n_valid - 1));
    float a2 = __shfl(amax, base_lane + min(2, n_valid - 1));
    float a3 = __shfl(amax, base_lane + min(3, n_valid - 1));
    float block_amax = fmaxf(fmaxf(a0, a1), fmaxf(a2, a3));

    // Compute E8M0 scale
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

// Pack 32 values to FP8 (4 values → 4 bytes per cvt pair)
#pragma unroll
    for (int j = 0; j < 32; j += 4) {
      fp8x4_t pk = {};
      pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(
          pk, vals[j], vals[j + 1], scale_f, false);
      pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(
          pk, vals[j + 2], vals[j + 3], scale_f, true);
      *(int *)(s_tok_fp8 + base + j) = *(int const *)&pk;
    }

    // First sub-block writes scale byte for the super-block
    if (sub_idx == 0) {
      s_tok_scales[super_blk] = se;
    }
  }
  __syncthreads();
}

// ── Split form of the quant above, for interleaving with weight-load issue ──
//
// Issuing a buffer_load is not free on gfx950: a wave cannot push loads into
// the memory system faster than it retires them, so the 24-load W13 prefetch
// spends ~1.5 us stalled at the issue point (measured; and NOT the s_mov m0
// hazard -- 24 s_mov vs 1 s_mov is 0.873 vs 0.871 us). That stall is fillable
// with VALU, which is what the split buys: front() ends with the token in
// registers and nothing outstanding, then the caller alternates issue blocks
// with back_range() calls that touch registers and LDS only.
//
// The order matters for a second reason. LLVM's waitcnt pass cannot see inside
// an inline asm block, so it cannot wait on the token load alone -- consuming
// any pre-asm VMEM value emits s_waitcnt vmcnt(0), which covers every
// in-flight weight load. front() must therefore run before the first issue
// block, or the quant blocks on the whole 98 KB weight stream.
template <int REDUCTION_SIZE>
struct _gang_fp8_quant_state {
  float vals[32];
  float scale_f;
  int base;
  int super_blk;
  int sub_idx;
  bool active;
};

template <int REDUCTION_SIZE>
__device__ __forceinline__ _gang_fp8_quant_state<REDUCTION_SIZE>
    _gang_fp8_quant_front(unsigned short const *__restrict__ src_bf16,
                          uint8_t *__restrict__ s_tok_scales) {
  constexpr int SUB_BLOCK = 32;
  constexpr int NSUBBLOCKS = REDUCTION_SIZE / SUB_BLOCK;
  static_assert(NSUBBLOCKS <= 256,
                "split quant assumes one sub-block per thread (no loop)");

  int const tid = threadIdx.x;
  int const lane_id = tid & 63;
  _gang_fp8_quant_state<REDUCTION_SIZE> st;
  st.active = tid < NSUBBLOCKS;

  int const sb = st.active ? tid : NSUBBLOCKS - 1;
  st.base = sb * SUB_BLOCK;
  st.super_blk = sb / 4;
  st.sub_idx = sb & 3;

  float amax = 0.0f;
#pragma unroll
  for (int j = 0; j < 32; j++) {
    st.vals[j] = _gang_bf16_to_float(src_bf16[st.base + j]);
    amax = fmaxf(amax, fabsf(st.vals[j]));
  }

  // Same clamped partner indexing as the unsplit version: NSUBBLOCKS need not
  // be a multiple of 4, so the tail super-block would otherwise reduce against
  // lanes whose `amax` was never written.
  int base_lane = lane_id & ~3;
  int const sb_first = sb - st.sub_idx;
  int const n_valid = min(4, (NSUBBLOCKS - 1) - sb_first + 1);
  float a0 = __shfl(amax, base_lane);
  float a1 = __shfl(amax, base_lane + min(1, n_valid - 1));
  float a2 = __shfl(amax, base_lane + min(2, n_valid - 1));
  float a3 = __shfl(amax, base_lane + min(3, n_valid - 1));
  float block_amax = fmaxf(fmaxf(a0, a1), fmaxf(a2, a3));

  uint8_t se = _gang_compute_e8m0_fp8(block_amax);
  if (se == 0) {
    st.scale_f = 1.0f;
  } else {
    union {
      float f;
      uint32_t u;
    } sv;
    sv.u = (uint32_t)se << 23;
    st.scale_f = sv.f;
  }
  if (st.active && st.sub_idx == 0) {
    s_tok_scales[st.super_blk] = se;
  }
  return st;
}

// Packs elements [LO, HI) of this thread's sub-block. Register + LDS only, so
// it can sit between weight-load issue blocks without forcing a waitcnt.
template <int REDUCTION_SIZE, int LO, int HI>
__device__ __forceinline__ void
    _gang_fp8_quant_back_range(_gang_fp8_quant_state<REDUCTION_SIZE> const &st,
                               uint8_t *__restrict__ s_tok_fp8) {
  if (!st.active) {
    return;
  }
#pragma unroll
  for (int j = LO; j < HI; j += 4) {
    fp8x4_t pk = {};
    pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(
        pk, st.vals[j], st.vals[j + 1], st.scale_f, false);
    pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(
        pk, st.vals[j + 2], st.vals[j + 3], st.scale_f, true);
    *(int *)(s_tok_fp8 + st.base + j) = *(int const *)&pk;
  }
}

// NT-load variant of per-thread FP8 quantization for W2 cross-XCD reads.
// Same sub-block structure as non-NT variant but uses dwordx4 NT loads.
template <int REDUCTION_SIZE>
__device__ __forceinline__ void _gang_wave_parallel_fp8_quant_nt(
    unsigned short const *__restrict__ src_bf16,
    uint8_t *__restrict__ s_tok_fp8,
    uint8_t *__restrict__ s_tok_scales) {

  constexpr int SUB_BLOCK = 32;
  constexpr int NSUBBLOCKS = REDUCTION_SIZE / SUB_BLOCK; // 96 for K=3072
  int const tid = threadIdx.x;
  int const lane_id = tid & 63;
  uint32_t const *src32 = (uint32_t const *)src_bf16;

  for (int sb = tid; sb < NSUBBLOCKS; sb += blockDim.x) {
    int const base = sb * SUB_BLOCK;
    int const super_blk = sb / 4;
    int const sub_idx = sb & 3;
    uint32_t const *base_ptr = src32 + base / 2;

    // 4 wide NT loads (64 bytes = 32 bf16)
    //
    // The outputs MUST be early-clobber ("=&v"). This is one asm block with
    // four separate instructions, so the compiler is free to allocate an
    // output register on top of an input it believes is dead after the
    // block -- and it does: without the '&' it emits
    //     global_load_dwordx4 v[4:7],   v[4:5],  ...
    //     global_load_dwordx4 v[8:11],  v[6:7],  ...
    //     global_load_dwordx4 v[12:15], v[8:9],  ...
    //     global_load_dwordx4 v[16:19], v[10:11],...
    // where the first load's destination overwrites the address operands of
    // the next three before they issue. Those loads then use whatever the
    // returned data happened to be as an address. A wild address that never
    // completes leaves the wave parked on the s_waitcnt vmcnt(0) below
    // forever, which hangs the __syncthreads at the end of this function and
    // through it the whole block -- the captured deadlock is exactly that:
    // wave 0 missing from the quant sync mask (0xe) while every wave had
    // already cleared the W13->W2 barrier (0xf).
    uint32_t dw[16];
    asm volatile("global_load_dwordx4 %0, %4, off sc0 sc1 nt\n"
                 "global_load_dwordx4 %1, %5, off sc0 sc1 nt\n"
                 "global_load_dwordx4 %2, %6, off sc0 sc1 nt\n"
                 "global_load_dwordx4 %3, %7, off sc0 sc1 nt"
                 : "=&v"(*(i32x4_t *)&dw[0]),
                   "=&v"(*(i32x4_t *)&dw[4]),
                   "=&v"(*(i32x4_t *)&dw[8]),
                   "=&v"(*(i32x4_t *)&dw[12])
                 : "v"(base_ptr),
                   "v"(base_ptr + 4),
                   "v"(base_ptr + 8),
                   "v"(base_ptr + 12)
                 : "memory");
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

    // Convert to float and find amax
    float vals[32];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; j++) {
      float lo = _gang_bf16_to_float((unsigned short)(dw[j] & 0xFFFF));
      float hi = _gang_bf16_to_float((unsigned short)(dw[j] >> 16));
      vals[j * 2] = lo;
      vals[j * 2 + 1] = hi;
      amax = fmaxf(amax, fmaxf(fabsf(lo), fabsf(hi)));
    }

    // Combine amaxes across the 4 sub-blocks of this 128-element super-block.
    //
    // NSUBBLOCKS is REDUCTION_SIZE/32 and is NOT guaranteed to be a multiple
    // of 4: for the W2 path REDUCTION_SIZE = INTERMEDIATE_SIZE = 2880, giving
    // NSUBBLOCKS = 90. The last super-block therefore has only 2 real
    // sub-blocks (sb 88, 89), but the shuffles below still read lanes for
    // sb 90 and 91 -- threads whose loop condition failed, so their `amax`
    // is an uninitialized register. Clamping the partner index to the last
    // valid sub-block makes the reduction read only lanes that ran.
    int base_lane = lane_id & ~3;
    int const sb_first = sb - sub_idx;  // first sb of this super-block
    int const sb_last = NSUBBLOCKS - 1; // last sb that actually runs
    int const n_valid = min(4, sb_last - sb_first + 1);
    float a0 = __shfl(amax, base_lane);
    float a1 = __shfl(amax, base_lane + min(1, n_valid - 1));
    float a2 = __shfl(amax, base_lane + min(2, n_valid - 1));
    float a3 = __shfl(amax, base_lane + min(3, n_valid - 1));
    float block_amax = fmaxf(fmaxf(a0, a1), fmaxf(a2, a3));

    // Compute E8M0 scale
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

// Pack 32 values to FP8
#pragma unroll
    for (int j = 0; j < 32; j += 4) {
      fp8x4_t pk = {};
      pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(
          pk, vals[j], vals[j + 1], scale_f, false);
      pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(
          pk, vals[j + 2], vals[j + 3], scale_f, true);
      *(int *)(s_tok_fp8 + base + j) = *(int const *)&pk;
    }

    // First sub-block writes scale byte
    if (sub_idx == 0) {
      s_tok_scales[super_blk] = se;
    }
  }
  // Record that this wave finished its strided share and is entering the
  // block-wide sync. If a stall shows a mask below 0xf here, the missing
  // wave never got out of the loop above -- almost certainly stuck on the
  // vmcnt(0) that drains its NT loads of d_swiglu_out.
  MPK_WS_WAVE_SYNC(tid >> 6);
  __syncthreads();
}

// ── Staging copy for an activation that is ALREADY MXFP8 ────────────────────
//
// The quantizer above exists because W13 hands W2 a bf16 activation, so every
// one of W2's tiles re-derives the same E4M3 bytes and the same E8M0 scales
// from the same 2048-element vector. When the producer emits MXFP8 directly
// (gang_moe_w13_linear_mxfp8_kernel's EMIT_FP8 epilogue) there is nothing to
// derive: the bytes W2 wants are the bytes in memory, and this degenerates to
// a global->LDS copy of half the bytes with no VALU at all.
//
// Same coherence rules as the quantizer it replaces. The producer is a W13
// tile on another XCD and per-XCD L2 is not coherent, so the loads keep
// `sc0 sc1 nt`: sc0 sc1 to miss this XCD's L2 rather than read a stale line,
// nt because nothing on this XCD wants the line back.
//
// Scale granularity is one E8M0 per 32 elements here, not the quantizer's one
// per 128. A W13 workgroup owns OUTPUT_PER_WG/2 = 32 consecutive activation
// columns, so 32 is the widest block whose amax it can compute without a
// cross-workgroup reduction -- and it is also what the weight side already
// uses, so the MFMA's B scale just gains the `* 4 + g` the A scale has.
template <int LEN>
__device__ __forceinline__ void
    _gang_stage_mxfp8_nt(uint8_t const *__restrict__ src_data,
                         uint8_t const *__restrict__ src_scale,
                         uint8_t *__restrict__ dst_data,
                         uint8_t *__restrict__ dst_scale) {
  static_assert(LEN % 128 == 0,
                "staging moves whole dwordx4s of data and whole dwords of "
                "per-32 scale");
  constexpr int NVEC = LEN / 16;   // dwordx4 of E4M3 data
  constexpr int NSC4 = LEN / 128;  // dword of E8M0 scale (4 blocks of 32)
  int const tid = threadIdx.x;

  for (int v = tid; v < NVEC; v += blockDim.x) {
    // Early-clobber for the same reason the quantizer documents: the compiler
    // will otherwise allocate the destination on top of the live address.
    i32x4_t d;
    asm volatile("global_load_dwordx4 %0, %1, off sc0 sc1 nt"
                 : "=&v"(d)
                 : "v"((uint32_t const *)src_data + v * 4)
                 : "memory");
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    *(i32x4_t *)(dst_data + v * 16) = d;
  }
  for (int s = tid; s < NSC4; s += blockDim.x) {
    uint32_t d;
    asm volatile("global_load_dword %0, %1, off sc0 sc1 nt"
                 : "=&v"(d)
                 : "v"((uint32_t const *)src_scale + s)
                 : "memory");
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    *((uint32_t *)dst_scale + s) = d;
  }
  MPK_WS_WAVE_SYNC(tid >> 6);
  __syncthreads();
}

// FP4×FP8 scaled MFMA: 16x16x128, hardware dequant + multiply
// A = weights (FP4 E2M1), 16 bytes/lane in lower 128 bits of i32x8
// B = tokens  (FP8 E4M3), 32 bytes/lane split across i32x8
__device__ __forceinline__ f32x4_t _gang_mfma_f4xf8(
    i32x8_t a, i32x8_t b, f32x4_t c, int scale_a, int scale_b) {
  return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
      a,
      b,
      c,
      4, // cbsz: FP4 E2M1 for src0 (weights)
      0, // blgp: FP8 E4M3 for src1 (tokens)
      0,
      scale_a,
      0,
      scale_b);
}

// A load that is *known* to come from device *global* memory.
//
// Clang infers address spaces intraprocedurally, so a pointer that arrives as
// an argument to a __noinline__ task kernel stays generic and every deref of
// it is emitted as flat_load rather than global_load. On gfx9 that is not
// merely a slower addressing mode: a flat instruction increments **both**
// vmcnt and lgkmcnt, so the `s_waitcnt lgkmcnt(0)` that retires an LDS read
// also waits on every outstanding weight load. In an MFMA loop that reads its
// A operand from global and its B operand from LDS -- which is every GEMM in
// this directory -- that single waitcnt sits directly in front of the MFMAs
// and drains the software pipeline, so the prefetch buys nothing at all.
//
// Casting at the load restores global_load and decouples the two counters.
// Safe only because every pointer handed to these kernels is device global:
// weights, scales, activations, residual, bias and output all come from the
// megakernel workspace or from a task descriptor. Never point this at LDS.
//
// The cast is two-step because clang rejects a reinterpret_cast that changes
// both the pointee type and the address space at once. T must be a POD or an
// ext_vector_type -- a HIP_vector_type class (uint2, uint4, ...) has a copy
// constructor taking a generic reference, which silently undoes the cast.
//
// Same idiom as gang_gemv_mxfp8_detail::ld_g; kept here so every consumer of
// _gang_load_fp8_mfma_b can reach it.
template <typename T>
__device__ __forceinline__ T _gang_ld_g(void const *p) {
  T const *q = static_cast<T const *>(p);
  return *(__attribute__((address_space(1))) T const *)q;
}

// Load FP8 B-operand for 16x16x128 MFMA.
// FP8 layout requires two 16-byte chunks at K offsets [g*16..g*16+15]
// and [g*16+64..g*16+79] within the 128-element tile.
__device__ __forceinline__ i32x8_t _gang_load_fp8_mfma_b(uint8_t const *data,
                                                         int kt,
                                                         int g) {
  i32x4_t lo = *(i32x4_t const *)(data + kt + g * 16);
  i32x4_t hi = *(i32x4_t const *)(data + kt + g * 16 + 64);
  i32x8_t r;
  r[0] = lo[0];
  r[1] = lo[1];
  r[2] = lo[2];
  r[3] = lo[3];
  r[4] = hi[0];
  r[5] = hi[1];
  r[6] = hi[2];
  r[7] = hi[3];
  return r;
}

// Global-memory twin of _gang_load_fp8_mfma_b, for the weight (A) operand.
// Identical addressing; the only difference is that the two 16-byte gathers
// go through _gang_ld_g so they emit global_load_dwordx4 and stay out of
// lgkmcnt. Use this one for anything read out of the weight buffer and the
// plain one for anything read out of LDS.
__device__ __forceinline__ i32x8_t _gang_load_fp8_mfma_b_g(uint8_t const *data,
                                                           int kt,
                                                           int g) {
  i32x4_t lo = _gang_ld_g<i32x4_t>(data + kt + g * 16);
  i32x4_t hi = _gang_ld_g<i32x4_t>(data + kt + g * 16 + 64);
  i32x8_t r;
  r[0] = lo[0];
  r[1] = lo[1];
  r[2] = lo[2];
  r[3] = lo[3];
  r[4] = hi[0];
  r[5] = hi[1];
  r[6] = hi[2];
  r[7] = hi[3];
  return r;
}

// ── FP4×FP4 helpers (gfx950) ─────────────────────────────────────────────

// Compute E8M0 block scale for FP4 (max representable value = 6.0)
// Uses multiply by reciprocal (1 v_mul_f32) instead of division
// (Newton-Raphson).
__device__ __forceinline__ uint8_t _gang_compute_e8m0_fp4(float amax) {
  if (amax == 0.0f) {
    return 0;
  }
  union {
    float f;
    uint32_t u;
  } v;
  v.f = amax * (1.0f / 6.0f); // v_mul_f32, not v_div (Newton-Raphson)
  int raw_exp = (int)((v.u >> 23) & 0xFF);
  if (v.u & 0x7FFFFF) {
    raw_exp++; // round up if mantissa non-zero
  }
  return (uint8_t)max(0, min(255, raw_exp));
}

// Per-thread FP4 quantization of bf16 input into LDS.
// 256 threads, K/32 blocks (96 for K=3072). Each thread handles one 32-element
// block per iteration. Writes nibble-packed FP4 data + E8M0 scales.
// LDS layout: [K/2 bytes data | K/32 bytes scales]
template <int REDUCTION_SIZE>
__device__ __forceinline__ void
    _gang_fp4_quant(unsigned short const *__restrict__ src_bf16,
                    uint8_t *__restrict__ s_tok_fp4,
                    uint8_t *__restrict__ s_tok_scales) {

  constexpr int BLOCK_SIZE = 32;
  constexpr int NBLOCKS = REDUCTION_SIZE / BLOCK_SIZE;
  int const tid = threadIdx.x;

  for (int blk = tid; blk < NBLOCKS; blk += blockDim.x) {
    int base = blk * BLOCK_SIZE;

    // Load 32 bf16 values and find amax using fmaxf/fabsf (v_max_f32, no VCC
    // stalls)
    float vals[32];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 32; j++) {
      vals[j] = _gang_bf16_to_float(src_bf16[base + j]);
      amax = fmaxf(amax, fabsf(vals[j]));
    }

    // Compute E8M0 scale via multiply + bit-shift (no division or exp2f)
    uint8_t se = _gang_compute_e8m0_fp4(amax);
    s_tok_scales[blk] = se;
    float scale_f;
    if (se == 0) {
      scale_f = 1.0f;
    } else {
      union {
        float f;
        uint32_t u;
      } sv;
      sv.u = (uint32_t)se << 23; // 2^(se-127) via bit construction
      scale_f = sv.f;
    }

// Pack to FP4 using hardware cvt instruction (4 values → 2 bytes)
#pragma unroll
    for (int j = 0; j < 32; j += 4) {
      unsigned pk = 0;
      pk = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
          pk, vals[j], vals[j + 1], scale_f, false);
      pk = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
          pk, vals[j + 2], vals[j + 3], scale_f, true);
      *(uint16_t *)(s_tok_fp4 + base / 2 + j / 2) = (uint16_t)pk;
    }
  }
  __syncthreads();
}

// NT-load variant of FP4 quant for W2 cross-XCD reads.
// Per-thread: each thread handles one 32-element block per iteration.
// Optimized: 4×dwordx4 NT loads, fmaxf/fabsf amax, multiply+bit-shift E8M0.
template <int REDUCTION_SIZE>
__device__ __forceinline__ void
    _gang_fp4_quant_nt(unsigned short const *__restrict__ src_bf16,
                       uint8_t *__restrict__ s_tok_fp4,
                       uint8_t *__restrict__ s_tok_scales) {

  constexpr int BLOCK_SIZE = 32;
  constexpr int NBLOCKS = REDUCTION_SIZE / BLOCK_SIZE;
  int const tid = threadIdx.x;
  uint32_t const *src32 = (uint32_t const *)src_bf16;

  for (int blk = tid; blk < NBLOCKS; blk += blockDim.x) {
    int base = blk * BLOCK_SIZE;
    uint32_t const *base_ptr = src32 + base / 2;

    // 4 wide NT loads (64 bytes = 32 bf16) instead of 16 individual dword loads
    // Early-clobber outputs are required here for the same reason as in
    // _gang_wave_parallel_fp8_quant_nt: without '&' the allocator puts the
    // first load's destination on top of the later loads' address registers.
    uint32_t dw[16];
    asm volatile("global_load_dwordx4 %0, %4, off sc0 sc1 nt\n"
                 "global_load_dwordx4 %1, %5, off sc0 sc1 nt\n"
                 "global_load_dwordx4 %2, %6, off sc0 sc1 nt\n"
                 "global_load_dwordx4 %3, %7, off sc0 sc1 nt"
                 : "=&v"(*(i32x4_t *)&dw[0]),
                   "=&v"(*(i32x4_t *)&dw[4]),
                   "=&v"(*(i32x4_t *)&dw[8]),
                   "=&v"(*(i32x4_t *)&dw[12])
                 : "v"(base_ptr),
                   "v"(base_ptr + 4),
                   "v"(base_ptr + 8),
                   "v"(base_ptr + 12)
                 : "memory");
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

    // Convert to float and find amax using fmaxf/fabsf (v_max_f32, no VCC
    // stalls)
    float vals[32];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; j++) {
      float lo = _gang_bf16_to_float((unsigned short)(dw[j] & 0xFFFF));
      float hi = _gang_bf16_to_float((unsigned short)(dw[j] >> 16));
      vals[j * 2] = lo;
      vals[j * 2 + 1] = hi;
      amax = fmaxf(amax, fmaxf(fabsf(lo), fabsf(hi)));
    }

    // Compute E8M0 scale via multiply + bit-shift (no division or exp2f)
    uint8_t se = _gang_compute_e8m0_fp4(amax);
    s_tok_scales[blk] = se;
    float scale_f;
    if (se == 0) {
      scale_f = 1.0f;
    } else {
      union {
        float f;
        uint32_t u;
      } sv;
      sv.u = (uint32_t)se << 23; // 2^(se-127) via bit construction
      scale_f = sv.f;
    }

// Pack to FP4
#pragma unroll
    for (int j = 0; j < 32; j += 4) {
      unsigned pk = 0;
      pk = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
          pk, vals[j], vals[j + 1], scale_f, false);
      pk = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
          pk, vals[j + 2], vals[j + 3], scale_f, true);
      *(uint16_t *)(s_tok_fp4 + base / 2 + j / 2) = (uint16_t)pk;
    }
  }
  __syncthreads();
}

// Load FP4 B-operand for 16x16x128 MFMA.
// FP4 uses only the lower 128 bits (16 bytes) of i32x8 — contiguous load.
// Data layout: [K/2 bytes], offset = kt/2 + g*16
__device__ __forceinline__ i32x8_t _gang_load_fp4_mfma_b(uint8_t const *data,
                                                         int kt,
                                                         int g) {
  i32x4_t lo = *(i32x4_t const *)(data + kt / 2 + g * 16);
  i32x8_t r = {};
  r[0] = lo[0];
  r[1] = lo[1];
  r[2] = lo[2];
  r[3] = lo[3];
  return r;
}

// FP4×FP4 scaled MFMA: 16x16x128, 16 cycles (half of FP4×FP8)
// Both A and B are FP4 in lower 128 bits of i32x8
__device__ __forceinline__ f32x4_t _gang_mfma_f4xf4(
    i32x8_t a, i32x8_t b, f32x4_t c, int scale_a, int scale_b) {
  return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
      a,
      b,
      c,
      4, // cbsz: FP4 for src0 (weights)
      4, // blgp: FP4 for src1 (tokens)
      0,
      scale_a,
      0,
      scale_b);
}

// ── BF16 MFMA helpers (W4A16 path: FP4 weights dequanted to BF16) ─────────

// BF16×BF16 MFMA: v_mfma_f32_16x16x32_bf16
// A = weights (BF16, dequanted from FP4), 8 BF16 values per lane
// B = tokens  (BF16, native), 8 BF16 values per lane
// K=32 per instruction (vs K=128 for FP4/FP8)
typedef __bf16 __attribute__((ext_vector_type(2))) bf16x2_t;
typedef __bf16 __attribute__((ext_vector_type(8))) bf16x8_t;

__device__ __forceinline__ f32x4_t _gang_mfma_bf16(bf16x8_t a,
                                                   bf16x8_t b,
                                                   f32x4_t c) {
  return __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c, 0, 0, 0);
}

// Hardware FP4→BF16 dequant using v_cvt_scalef32_pk_bf16_fp4 (gfx950).
// 4 instructions produce bf16x8_t (vs ~40 ALU instructions with software LUT).
// Each instruction converts 1 byte (2 FP4 nibbles) → 2 BF16 values, scaled by
// E8M0.
//
// fp4_data: 4 bytes (8 nibbles) for this lane's K-group
// scale_e8m0: E8M0 block scale byte (1 per 32 K-elements)
// Returns: bf16x8_t with 8 BF16 values
__device__ __forceinline__ bf16x8_t _gang_dequant_fp4_to_bf16_8(
    uint8_t const *__restrict__ fp4_data, uint8_t scale_e8m0) {

  uint32_t raw_fp4;
  __builtin_memcpy(&raw_fp4, fp4_data, 4);
  // Scale: reinterpret E8M0 byte as float exponent (2^(e8m0-127))
  float scale;
  uint32_t su = (uint32_t)scale_e8m0 << 23;
  __builtin_memcpy(&scale, &su, 4);

  // v_cvt_scalef32_pk_bf16_fp4: converts 2 FP4 nibbles (1 byte) → 2 BF16 values
  // word_sel 0..3 selects byte 0..3 of the 32-bit source
  union {
    bf16x2_t h[4];
    bf16x8_t v;
  } r;
  r.h[0] = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(raw_fp4, scale, 0);
  r.h[1] = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(raw_fp4, scale, 1);
  r.h[2] = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(raw_fp4, scale, 2);
  r.h[3] = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(raw_fp4, scale, 3);
  return r.v;
}

// Hardware FP4→BF16 dequant from raw pre-loaded FP4 data (no memory access).
// raw_fp4: 4 bytes (8 nibbles = 8 E2M1 values), already loaded
// scale_e8m0: E8M0 block scale byte (already loaded)
__device__ __forceinline__ bf16x8_t
    _gang_dequant_raw_fp4_to_bf16_8(uint32_t raw_fp4, uint8_t scale_e8m0) {

  float scale;
  uint32_t su = (uint32_t)scale_e8m0 << 23;
  __builtin_memcpy(&scale, &su, 4);

  union {
    bf16x2_t h[4];
    bf16x8_t v;
  } r;
  r.h[0] = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(raw_fp4, scale, 0);
  r.h[1] = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(raw_fp4, scale, 1);
  r.h[2] = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(raw_fp4, scale, 2);
  r.h[3] = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(raw_fp4, scale, 3);
  return r.v;
}

// Load BF16 token B-operand for v_mfma_f32_16x16x32_bf16.
// Each lane loads 8 consecutive BF16 values for its K-group.
// Data layout: [K * 2 bytes] (BF16 native, 2 bytes per element)
// g = lane_id >> 4 (0..3), each group covers K positions [g*8..g*8+7]
__device__ __forceinline__ bf16x8_t
    _gang_load_bf16_mfma_b(unsigned short const *data, int kt, int g) {
  bf16x8_t r;
  __builtin_memcpy(&r, data + kt + g * 8, sizeof(bf16x8_t));
  return r;
}

// ── Gang MoE MXFP4 kernel (gfx950 hardware FP4×FP8 MFMA path) ───────────
// Handles both W13 and W2 projections.
// Each gang task processes tiles assigned to its XCD.
//
// FUSE_SWIGLU: When true (only valid with W13_LINEAR=true), fuses SwiGLU
// activation into the epilogue. The output has interleaved gate/up pairs
// (acc[0]=gate, acc[1]=up, acc[2]=gate, acc[3]=up). The epilogue applies
// fast_swigluoai(gate+bias, up+bias) and writes half the output (2 values
// per lane instead of 4). Output tensor is [bs, topk, OUTPUT_SIZE/2].
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
    gang_moe_linear_mxfp4_kernel_mi300(void const *input_ptr,
                                       void const *weight_ptr,
                                       void const *routing_ptr,
                                       void const *mask_ptr,
                                       void const *bias_ptr,
                                       void *output_ptr,
                                       int tile_idx) {
  static_assert(OUTPUT_PER_WG % 16 == 0,
                "OUTPUT_PER_WG must be multiple of 16");
  static_assert(REDUCTION_SIZE % 128 == 0,
                "REDUCTION_SIZE must be multiple of 128 for FP4 MFMA");

  // MXFP4 weight layout constants (unchanged from original)
  constexpr int NUM_BLOCKS_32 =
      REDUCTION_SIZE / 32; // number of 32-element scale blocks
  constexpr int WG_DATA_BYTES =
      OUTPUT_PER_WG * (REDUCTION_SIZE / 2); // FP4 nibble-packed
  constexpr int WG_SCALE_BYTES = OUTPUT_PER_WG * NUM_BLOCKS_32; // E8M0 scales
  constexpr int WG_BYTES = WG_DATA_BYTES + WG_SCALE_BYTES;
  constexpr int EXPERT_WGS = OUTPUT_STRIDE / OUTPUT_PER_WG;
  constexpr int64_t EXPERT_BYTES = static_cast<int64_t>(EXPERT_WGS) * WG_BYTES;

  // Hardware MFMA constants
  constexpr int K_PER_MFMA =
      128; // FP4 MFMA processes 128 K-elements per instruction
  constexpr int MFMA_ITERS = REDUCTION_SIZE / K_PER_MFMA; // e.g. 3072/128 = 24

  // N-parallel: 4 waves, each handles 16 output rows
  constexpr int NUM_WAVES = 4;
  constexpr int N_TILES = OUTPUT_PER_WG / 16;
  constexpr int TILES_PER_WAVE = N_TILES / NUM_WAVES;
  constexpr int N_TILES_PER_WG = EXPERT_WGS;

  // FP4 token storage in shared memory: data + scales
  constexpr int FP4_TOK_DATA =
      REDUCTION_SIZE / 2; // 0.5 byte per element (nibble-packed)
  constexpr int FP4_TOK_SCALES =
      REDUCTION_SIZE / 32; // 1 E8M0 scale per 32-element block
  constexpr int FP4_TOK_BYTES = FP4_TOK_DATA + FP4_TOK_SCALES;

  unsigned short const *A = (unsigned short const *)input_ptr;
  uint8_t const *W = (uint8_t const *)weight_ptr;
  int const *d_routing = (int const *)routing_ptr;
  int const *d_mask = (int const *)mask_ptr;
  unsigned short const *d_bias = (unsigned short const *)bias_ptr;
  unsigned short *d_output = (unsigned short *)output_ptr;

  extern __shared__ char _gang_mxfp4_smem[];
  // Shared memory layout: FP4 quantized tokens + scales
  uint8_t *s_tok_fp4 = (uint8_t *)_gang_mxfp4_smem;
  uint8_t *s_tok_scales = s_tok_fp4 + FP4_TOK_DATA;

  int const tid = threadIdx.x;
  int const warp_id = tid >> 6; // wave ID (0..3)
  int const lane_id = tid & 63;
  int const col = lane_id & 15; // output row within 16x16 MFMA tile (N-dim)
  int const g = lane_id >> 4;   // K-group (0..3), each handles 32 FP4 nibbles

  // Get XCD ID for flat global tile distribution
  int xcd_id = _gang_moe_get_xcd_id();

  int const num_activated_experts = d_mask[NUM_EXPERTS];

  // Flat global tile distribution across all 8 XCDs.
  // Interleave tiles round-robin so all XCDs stay active even when
  // num_activated_experts < 8 (e.g. top-4 at bs=1).
  int global_tile = tile_idx * 8 + xcd_id;
  int total_tiles = num_activated_experts * TILES_PER_EXPERT;
  if (global_tile >= total_tiles) {
    return;
  }
  int expert_idx = global_tile / TILES_PER_EXPERT;
  int tile_within_expert = global_tile % TILES_PER_EXPERT;
  int expert_id = d_mask[expert_idx];
  int const *expert_routing = d_routing + expert_id * BATCH_SIZE;

  // Expert weight base
  uint8_t const *expert_weight =
      W + static_cast<int64_t>(expert_id) * EXPERT_BYTES;

  // Decode tile position: (token_idx, wg_idx)
  int tok_idx = tile_within_expert / N_TILES_PER_WG;
  int wg_idx = tile_within_expert % N_TILES_PER_WG;

  if (tok_idx >= BATCH_SIZE) {
    return;
  }

  // Check routing
  int route_val = expert_routing[tok_idx];
  if (route_val == 0) {
    return;
  }
  int topk_slot = route_val - 1;

  // Workgroup weight pointers (FP4 data followed by E8M0 scales)
  uint8_t const *wg_data =
      expert_weight + static_cast<int64_t>(wg_idx) * WG_BYTES;
  uint8_t const *wg_scales = wg_data + WG_DATA_BYTES;

  // ── Phase 1: Load BF16 input and quantize to FP4 in shared memory ──────
  unsigned short const *input_base;
  if constexpr (W13_LINEAR) {
    input_base = A + tok_idx * REDUCTION_SIZE;
  } else {
    input_base =
        A + tok_idx * (NUM_TOPK * REDUCTION_SIZE) + topk_slot * REDUCTION_SIZE;
  }

  _gang_fp4_quant<REDUCTION_SIZE>(input_base, s_tok_fp4, s_tok_scales);

  // ── Phase 2: MFMA FP4(weights) × FP4(tokens) ─────────────────────────
  // Distribute MFMA K-iterations across waves evenly
  constexpr int base_iters = MFMA_ITERS / NUM_WAVES;
  constexpr int extra_iters = MFMA_ITERS % NUM_WAVES;

  for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; tile_iter++) {
    int wave_tile = warp_id + tile_iter * NUM_WAVES;
    // MFMA 16x16: col (lane%16) = batch dim, g (lane/16) = output group (4 rows
    // each)

    f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

    int const w_row = wave_tile * 16 + col;

#if MPK_MOE_KBATCH > 1
    // ── MPK_MOE_KBATCH: issue KB trips' weight loads before consuming any ──
    //
    // The loop below this one is `#pragma unroll 1` with the A-operand load
    // feeding the MFMA in the same trip, so the wave runs
    //   global_load a -> s_waitcnt vmcnt(0) lgkmcnt(0) -> v_mfma -> repeat
    // with exactly ONE load in flight for all MFMA_ITERS trips (48 at W13's
    // K=6144, 16 at W2's K=2048). That makes the tile a chain of HBM
    // round-trip latencies, not a bandwidth stream: the expert weights are
    // read once per token and never reused, so every one of those loads is a
    // cold miss.
    //
    // Same transform as qkv_a's resolve loop (b73b45f): batch the loads into
    // registers first, then consume. Here the fence is the MFMA's own
    // s_waitcnt rather than a ds_write, but the effect is identical -- the
    // compiler cannot hoist trip k+1's load above trip k's wait because the
    // accumulator chain orders the MFMAs and the wait sits inside it.
    //
    // b_reg/sb come from LDS and are batched too: ds_read raises lgkmcnt, and
    // the MFMA waits on lgkmcnt(0) as well, so leaving them in the consume
    // loop would just move the serialization from vmcnt to lgkmcnt.
    //
    // Cost is KB * 16 VGPR (a_reg 8 + b_reg 8) plus two scalars. Registers
    // are free here -- see the occupancy table; the block count pins 1
    // wave/SIMD, not the register file.
    //
    // -- MEASURED NEUTRAL. THE MoE K-LOOP IS NOT LOAD-ISSUE-BOUND. ----------
    // Alternated KB=1/KB=4 in one batch, n=3 each, min-of-115 ms/iter:
    //   KB=1  10.262 / 10.232 / 10.279   mean 10.258
    //   KB=4  10.282 / 10.229 / 10.258   mean 10.256
    // -0.002 ms, ranks fully interleaved, nowhere near the 0.26 ms noise
    // floor. The build is real: -DMPK_MOE_KBATCH=4 appears 8x in the log and
    // the two .so images differ by md5. It is not a register problem either:
    // worker_kernel stays at 284 VGPR / private_seg 596 / vgpr_spill_count 0
    // on both arms, so the batched operands cost nothing.
    //
    // Why it does not pay: the identical transform on qkv_a's resolve loop
    // (b73b45f) bought -29% because that loop was at ~1.5 loads in flight
    // against an L2 share it was using 46% of. This loop is already near its
    // roof -- the W13 tile runs at ~74% of the per-CU HBM share -- so the
    // wave is waiting on delivered bytes, not on issue slots, and adding
    // outstanding requests cannot shorten a bandwidth queue. Same verdict,
    // same reason, as MPK_QUANT_V16 on the qkv_a prologue.
    //
    // Default 1. Kept documented rather than deleted so the next person
    // sizing "one load in flight" against this loop reads the number first:
    // `#pragma unroll 1` here is not a bug.
    constexpr int KB = MPK_MOE_KBATCH;
    for (int k0 = 0; k0 < MFMA_ITERS; k0 += KB) {
      i32x8_t a_bat[KB];
      i32x8_t b_bat[KB];
      int sa_bat[KB];
      int sb_bat[KB];
#pragma unroll
      for (int u = 0; u < KB; u++) {
        int const kt = (k0 + u) * K_PER_MFMA;
        if (k0 + u < MFMA_ITERS) {
          a_bat[u] = *(i32x8_t const *)(wg_data + w_row * (REDUCTION_SIZE / 2) +
                                        kt / 2 + g * 16);
          sa_bat[u] = (int)wg_scales[w_row * NUM_BLOCKS_32 + kt / 32 + g];
          b_bat[u] = _gang_load_fp4_mfma_b(s_tok_fp4, kt, g);
          sb_bat[u] = (int)s_tok_scales[kt / 32 + g];
        }
      }
#pragma unroll
      for (int u = 0; u < KB; u++) {
        if (k0 + u < MFMA_ITERS) {
          acc = _gang_mfma_f4xf4(a_bat[u], b_bat[u], acc, sa_bat[u], sb_bat[u]);
        }
      }
    }
#else
// K-reduction loop: each MFMA processes 128 K-elements
#pragma unroll 1
    for (int kt = 0; kt < REDUCTION_SIZE; kt += K_PER_MFMA) {
      // Load FP4 weight A-operand: 16 bytes per lane (32 FP4 nibbles)
      int a_off = w_row * (REDUCTION_SIZE / 2) + kt / 2 + g * 16;
      i32x8_t a_reg = *(i32x8_t const *)(wg_data + a_off);

      // Load FP4 token B-operand: contiguous 16B per lane
      i32x8_t b_reg = _gang_load_fp4_mfma_b(s_tok_fp4, kt, g);

      // Weight scale: E8M0 per 32-element block
      int sa = (int)wg_scales[w_row * NUM_BLOCKS_32 + kt / 32 + g];

      // Token scale: E8M0 per 32-element block (4 per 128 K-elements)
      int sb = (int)s_tok_scales[kt / 32 + g];

      // Hardware MFMA: FP4×FP4, 16 cycles
      acc = _gang_mfma_f4xf4(a_reg, b_reg, acc, sa, sb);
    }
#endif

    // ── Epilogue: write result with bias ────────────────────────────────
    // MFMA C layout: col (lane_id & 15) = batch dimension, g (lane_id >> 4) =
    // output group acc[i] = C[g*4+i][col], so with BS=1 only col==0 is valid.
    if (col == 0) {
      if constexpr (FUSE_SWIGLU) {
        // Fused SwiGLU epilogue: acc[0]=gate, acc[1]=up, acc[2]=gate, acc[3]=up
        // Apply SwiGLU(gate+bias, up+bias) and write 2 activated values.
        constexpr int ACT_STRIDE = OUTPUT_STRIDE / 2;
        for (int i = 0; i < 4; i += 2) {
          int out_n = wg_idx * OUTPUT_PER_WG + wave_tile * 16 + g * 4 + i;
          if (out_n + 1 < OUTPUT_SIZE) {
            // Add biases (interleaved in [E, OUTPUT_STRIDE])
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
            d_output[out_idx] = _gang_float_to_bf16(activated);
          }
        }
      } else {
        // Standard epilogue: write 4 output elements per lane.
        for (int i = 0; i < 4; i++) {
          int out_n = wg_idx * OUTPUT_PER_WG + wave_tile * 16 + g * 4 + i;
          if (out_n < OUTPUT_SIZE) {
            float sum = acc[i];

            // Add bias (2D: [E, OUTPUT_STRIDE])
            float bias_val;
            unsigned bt = (unsigned)d_bias[expert_id * OUTPUT_STRIDE + out_n]
                          << 16;
            __builtin_memcpy(&bias_val, &bt, 4);

            float val = sum + bias_val;
            int out_idx = tok_idx * (NUM_TOPK * OUTPUT_STRIDE) +
                          topk_slot * OUTPUT_STRIDE + out_n;
            d_output[out_idx] = _gang_float_to_bf16(val);
          }
        }
      }
    }
  }

  __syncthreads();
}

} // namespace kernel
