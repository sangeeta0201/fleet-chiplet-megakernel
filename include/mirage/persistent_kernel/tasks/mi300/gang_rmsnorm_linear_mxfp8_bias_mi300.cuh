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
//
// SRC_IS_GLOBAL and NW_IS_LDS say which address space each input lives in.
// The callers differ: the plain path used to hand over the raw row in device
// global, the FUSE_RESADD path hands over s_x_bf16, which is LDS. It matters
// because the global case is read through an addrspace(1) cast -- otherwise
// clang, which cannot see through the __noinline__ task boundary, emits
// flat_load, and a flat instruction bumps lgkmcnt as well as vmcnt on gfx9.
// This function ends in ds_write, so that lgkmcnt would be waited on with the
// row still in flight. Pointing a global read at LDS would be UB; the template
// arguments are the only thing keeping the cases apart.
//
// NW_IS_LDS exists for the same reason the row staging does, one level up:
// with both operands in LDS this function issues no vmem at all, so the
// caller's hoisted A-tile prefetch can stay in flight across it. See the
// LDS_PROLOGUE block in the kernel.
//
// BARRIER_LDS_ONLY is the other half of that. The trailing __syncthreads()
// publishes s_tok_fp8 and s_tok_scales, which is LDS traffic, but HIP lowers
// it to `s_waitcnt vmcnt(0) lgkmcnt(0)` -- and vmcnt(0) would retire exactly
// the prefetch we are trying to keep outstanding. The barrier is split into
// its LDS-only parts when the caller has vmem in flight it wants to keep.
#ifndef MPK_QUANT_V16
#define MPK_QUANT_V16 0
#endif
template <int REDUCTION_SIZE,
          bool SRC_IS_GLOBAL,
          bool NW_IS_LDS = false,
          bool BARRIER_LDS_ONLY = false>
__device__ __forceinline__ void _gang_wave_parallel_fp8_quant_rmsnorm(
    unsigned short const *__restrict__ src_bf16,
    unsigned short const *__restrict__ norm_weight,
    float rms_rcp,
    uint8_t *__restrict__ s_tok_fp8,
    uint8_t *__restrict__ s_tok_scales) {

  using gu16 = __attribute__((address_space(1))) unsigned short const *;
  auto ld_src = [&](int i) -> unsigned short {
    if constexpr (SRC_IS_GLOBAL) {
      return ((gu16)src_bf16)[i];
    } else {
      return src_bf16[i];
    }
  };
  auto ld_nw = [&](int i) -> unsigned short {
    if constexpr (NW_IS_LDS) {
      return norm_weight[i];
    } else {
      return ((gu16)norm_weight)[i];
    }
  };

  // ── MPK_QUANT_V16: 16-byte loads for the two bf16 operands ──────────────
  // The scalar loop below reads 32 contiguous bf16 from each of the row and
  // the norm weight. LLVM's load-store vectorizer merges them, but only into
  // `global_load_dwordx2` -- through the addrspace(1) cast it cannot prove
  // better than 8-byte alignment on an `unsigned short const *`. That is 8
  // loads per operand per sub-block where 4 would do, in what the qkv_a
  // region split (commit 48fea7f) measured as 40% of the tile.
  //
  // Both are 16-byte aligned in fact: every caller's row starts at a token
  // boundary of a torch tensor or at an LDS array base, and `base` is a
  // multiple of SUB_BLOCK = 32 elements = 64 bytes. Stating it lets the
  // dwordx4 out. Bit-identical arithmetic -- only the load width changes.
//
// -- MEASURED NEUTRAL. THE PROLOGUE IS NOT LOAD-ISSUE-BOUND. ---------------
// Alternated OFF/ON in one batch, n=3 each, decode ms/iter:
//   OFF  10.895 / 10.784 / 10.648   mean 10.776
//   ON   10.996 / 10.904 / 10.614   mean 10.838
// +0.062 ms, inside the 0.26 ms noise floor. The ISA change is real -- the
// image goes from 0 to 657 global_load_dwordx4 -- so this is not a build
// that did not take. Generated text is correct on the ON arm.
//
// Why it does not pay: the 32-element loop is 32 bf16->f32 converts, 32
// multiplies and 32 fmax per thread against 8 loads, and 192 of 256 threads
// run exactly one sub-block. Halving the load *count* leaves the same bytes
// and the same VALU, and the row is in L2 already (qkv_a's 24 tiles per XCD
// all read it). Load issue was never the queue this prologue waits in.
// Default OFF; kept because the ISA-histogram recipe that found it is the
// reusable part.
  // HIP's int4 is a HIP_vector_type class, which has no constructor reachable
  // from a raw reinterpreting load; a POD of 8 bf16 with alignas(16) is the
  // same 16 bytes and lowers to the same dwordx4.
  struct alignas(16) i4 {
    unsigned short h[8];
  };
  auto ld_src16 = [&](int i) -> i4 {
    if constexpr (SRC_IS_GLOBAL) {
      using gi4 = __attribute__((address_space(1))) i4 const *;
      return *(gi4)(void const *)(src_bf16 + i);
    } else {
      return *(i4 const *)(void const *)(src_bf16 + i);
    }
  };
  auto ld_nw16 = [&](int i) -> i4 {
    if constexpr (NW_IS_LDS) {
      return *(i4 const *)(void const *)(norm_weight + i);
    } else {
      using gi4 = __attribute__((address_space(1))) i4 const *;
      return *(gi4)(void const *)(norm_weight + i);
    }
  };

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
#if MPK_QUANT_V16
    {
      i4 sv[4], nv[4];
#pragma unroll
      for (int c = 0; c < 4; c++) {
        sv[c] = ld_src16(base + c * 8);
        nv[c] = ld_nw16(base + c * 8);
      }
#pragma unroll
      for (int c = 0; c < 4; c++) {
#pragma unroll
        for (int e = 0; e < 8; e++) {
          float v = _gang_bf16_to_float(sv[c].h[e]) * rms_rcp *
                    _gang_bf16_to_float(nv[c].h[e]);
          vals[c * 8 + e] = v;
          amax = fmaxf(amax, fabsf(v));
        }
      }
      (void)ld_src;
      (void)ld_nw;
    }
#else
#pragma unroll
    for (int j = 0; j < 32; j++) {
      float v = _gang_bf16_to_float(ld_src(base + j)) * rms_rcp *
                _gang_bf16_to_float(ld_nw(base + j));
      vals[j] = v;
      amax = fmaxf(amax, fabsf(v));
    }
    (void)ld_src16;
    (void)ld_nw16;
#endif

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
  if constexpr (BARRIER_LDS_ONLY) {
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    __builtin_amdgcn_s_barrier();
  } else {
    __syncthreads();
  }
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
// Publish four adjacent bf16 outputs.
//
// The default leaves them in this XCD's L2, which is all a same-XCD consumer
// needs and is what every standalone caller wants: the task graph's event
// boundary does the buffer_wbl2 that makes them visible elsewhere. A fused
// caller has no event boundary, so when the consumer sits on another XCD it
// asks for WRITE_THROUGH and the store goes past L2 to memory.
//
// The four columns are contiguous, so one global_store_dwordx2 carries them --
// the same packing gang_moe_linear_mxfp8's WRITE_THROUGH epilogue does on its
// SwiGLU pair. That needs an 8-byte-aligned destination: OUTPUT_PER_WG % 16 ==
// 0 makes every term of the column index a multiple of 4, so the requirement
// reduces to a 4-aligned output_stride, which the registrar asserts.
template <bool WRITE_THROUGH>
__device__ __forceinline__ void _rnlm8_store4(unsigned short *dst,
                                              unsigned short v0,
                                              unsigned short v1,
                                              unsigned short v2,
                                              unsigned short v3) {
  if constexpr (WRITE_THROUGH) {
    unsigned long long packed = (unsigned long long)v0 |
                                ((unsigned long long)v1 << 16) |
                                ((unsigned long long)v2 << 32) |
                                ((unsigned long long)v3 << 48);
    st_wt_u64((void *)dst, packed);
  } else {
    dst[0] = v0;
    dst[1] = v1;
    dst[2] = v2;
    dst[3] = v3;
  }
}

// Resolve the residual stream out of the MoE's f32 workspace, and reduce it,
// in one pass -- the prologue that replaces the MOE_RESIDUAL_ADD_F32 task.
//
// GLM's MoE W2 epilogue scales by the routing weight and atomicAdds into an
// f32 workspace, so a layer's output is `workspace + the pre-MoE residual`.
// That add used to be a task of its own: grid_dim (1,1,1), 47 dispatches per
// token, one workgroup awake behind a full event boundary and 239 idle. gpt-oss
// never had it -- gang_resaddf32_rmsnorm_linear_mxfp4_bias_kernel folds the
// same add into the next stage's RMSNorm prologue. This is that fold, for the
// MXFP8 path.
//
// Three things come out of the single pass over the row:
//   d_x_out[h] = bf16(workspace[h] + residual[h])  -- the residual stream, which
//                the o_proj/router task still reads as its own residual
//   s_x[h]     = the same value, staged in LDS for the quantizer
//   the sum of squares that Phase 4 of the norm needs
//
// The LDS copy is what makes this a win rather than a wash. gpt-oss keeps the
// summed row in registers because its quantizer walks the row with the same
// thread mapping the prologue does; GLM's maps 32 contiguous elements to one
// thread against this pass's four strided, so a register handoff would need one
// of the two mappings rewritten. Reading the row back from global instead would
// race this workgroup's own stores through a write-through vL1. LDS costs two
// bytes per element and no occupancy -- the persistent kernel already runs a
// single workgroup per CU.
//
// The square is taken on the *rounded* bf16, not on the f32 sum, so the value
// reduced here is exactly the value the quantizer will scale. That makes the
// fold bit-identical to the two-task path it replaces.
//
// Every workgroup on the gang recomputes the whole row -- it has to, each needs
// the result in its own LDS -- but only `store_x` ones publish it, since all
// 128 of them would otherwise write the same 4 KB to the same addresses. The
// caller sets it on one workgroup per XCD and the store is write-through, so
// the row lands in memory once per XCD rather than as eight dirty copies in
// eight L2s.
//
// Note the workspace is not zeroed here: every gang worker reads the whole row,
// so a worker that zeroed its slice would race the ones still reading. The
// o_proj task does it up front instead, a whole task later in the event chain,
// where its own W13->W2 barrier orders the zero against the next accumulate.
//
// The residual is `norm_input` itself -- under the fold that tensor is the
// residual stream rather than an already-resolved row, so there is no second
// pointer and nothing appears twice in the task's input list.
//
// ── expert parallelism ───────────────────────────────────────────────────
// EP_PEER_SLOTS > 1 makes this pass the cross-rank reduction as well.
// `d_res` then points at the symmetric gather buffer -- EP_PEER_SLOTS
// consecutive [batch, REDUCTION_SIZE] bf16 planes, EP_SLOT_ELEMS apart, one
// per rank -- and the row is the sum of all of them.
//
// This is gpt-oss's EP_PEER_SLOTS in gang_rmsnorm_linear_mxfp4_bias_mi300.cuh,
// and the reason is the same one: the previous owner of these adds was a
// separate reduce pass at the end of the layer, and that pass needed an exit
// barrier behind it so nobody entered the next layer before the combined row
// was whole. Making the consumer BE the reduction leaves no window between
// "combined" and "consumed" for a barrier to protect.
//
// One deliberate divergence from gpt-oss: `d_ws` is not read at all under EP.
// gpt-oss still adds its f32 workspace because its fold zeroes the workspace
// as it folds, so the term is a known zero. GLM does not zero there -- the
// o_proj stage does it, a phase later (see the note above, and the zeroing
// block in gang_oproj_router_fused_mi300.cuh) -- so at this point the
// workspace still holds this rank's partial, which the fold has already
// written into this rank's gather slot. Adding it would count the local
// contribution twice. Skipping it also drops 8 KB of loads per layer.
//
// STAGE_NW additionally copies the norm weight into LDS on the same pass. It
// is a separate row, so the two loads are independent and cost latency only
// once; doing it in a loop of its own would need its own s_waitcnt in front of
// its own ds_write. The point of having it in LDS at all is that the quantizer
// then reads no global memory, which is what lets the caller keep an A-tile
// prefetch in flight across it.
//
// ── the one-shot fold ────────────────────────────────────────────────────
// Sum the EP_PEER_SLOTS planes of one row once and publish the resolved bf16
// row, so PRE_FOLDED consumers read 12 KB instead of 108 KB. Sliced across
// NSLICE workgroups: slice `s` walks iterations [s*ITERS/NSLICE, ...) of the
// same 256 x float4 walk the prologue uses, so the addresses and the rounding
// are the prologue's, element for element.
//
// Summation order matters for bit-identity and is preserved: the prologue
// starts from slot 0 and adds slots 1..7 in order; starting from 0.0f and
// adding slot 0 first is the same sequence, since 0.0f + x == x exactly.
//
// The store is write-through, and every XCD runs its own copy of the fold
// writing the same bytes to the same addresses -- the same
// identical-value-per-L2 argument `store_x` already relies on.
template <int REDUCTION_SIZE, int EP_PEER_SLOTS, int EP_SLOT_ELEMS, int NSLICE>
__device__ __forceinline__ void
_rnlm8_ep_fold_slice(unsigned short const *__restrict__ d_res,
                     unsigned short *__restrict__ d_x_out,
                     int slice) {
  constexpr int VEC = 4;
  constexpr int NTHREADS = 256;
  constexpr int ITERS = REDUCTION_SIZE / (NTHREADS * VEC);
  static_assert(ITERS % NSLICE == 0,
                "the fold slice count must divide the row's float4 walk");
  constexpr int PER = ITERS / NSLICE;
  int const tid = threadIdx.x;
#pragma unroll 1
  for (int v = slice * PER; v < (slice + 1) * PER; v++) {
    int const off = (v * NTHREADS + tid) * VEC;
    uint2 pk[EP_PEER_SLOTS];
#pragma unroll
    for (int p = 0; p < EP_PEER_SLOTS; p++) {
      pk[p] = *reinterpret_cast<uint2 const *>(
          d_res + (size_t)p * EP_SLOT_ELEMS + off);
    }
    float f[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int p = 0; p < EP_PEER_SLOTS; p++) {
      f[0] += _gang_bf16_to_float((unsigned short)pk[p].x);
      f[1] += _gang_bf16_to_float((unsigned short)(pk[p].x >> 16));
      f[2] += _gang_bf16_to_float((unsigned short)pk[p].y);
      f[3] += _gang_bf16_to_float((unsigned short)(pk[p].y >> 16));
    }
    unsigned short const b[4] = {
        _gang_float_to_bf16(f[0]), _gang_float_to_bf16(f[1]),
        _gang_float_to_bf16(f[2]), _gang_float_to_bf16(f[3])};
    st_wt_u64((void *)(d_x_out + off),
              (unsigned long long)((unsigned)b[0] | ((unsigned)b[1] << 16)) |
                  ((unsigned long long)((unsigned)b[2] |
                                        ((unsigned)b[3] << 16))
                   << 32));
  }
}

// ── The hoisted prologue ─────────────────────────────────────────────────
// _rnlm8_ep_fold_slice hoists the EP reduction and stops there, so every tile
// still stages the row and the norm weight into LDS, still block-reduces the
// sum of squares, and still quantizes -- 6.98 us of a 17.20 us qkv_a tile,
// derived 192 times per layer per rank from one 6144-element row (48fea7f).
// The fold was the cheap third of that and hoisting it alone measured neutral
// (c72b559); this hoists the whole thing.
//
// PRO_WGS workgroups per XCD each take one slice. A slice folds its columns,
// publishes the resolved bf16 to d_x_out exactly as the fold did, and squares
// what it stored. The sum of squares is the only cross-slice term, so the
// slices exchange it and nothing else: each writes its partial and then an
// epoch stamp, and each spins until all PRO_WGS stamps carry this layer's
// epoch. Stamping instead of counting is what keeps the buffer from needing to
// be zeroed between layers -- a counter would, and there is no phase left in
// the layer that could do it.
//
// Then each slice normalizes and quantizes its own columns into the publish
// buffer, in the layout the consumer's LDS wants: E4M3 bytes at [base] and one
// E8M0 per 128 at [base/128], which is _gang_wave_parallel_fp8_quant_rmsnorm's
// own layout, called here on the slice. That is legal only because a slice is
// a whole number of 128-element scale groups; the static_assert below is the
// load-bearing one.
//
// The exchange is XCD-local and PRO_WGS wide, not a rendezvous: every XCD runs
// its own copy writing the same bytes to its own L2, the same
// identical-value-per-L2 argument store_x and the fold already rely on. The 24
// tiles are released by the XCD-local flag the fold already built, so the
// layer gains no barrier.
//
// The squares are taken on the ROUNDED bf16, not on the f32 accumulator. The
// un-hoisted prologue reads the row back out of memory as bf16, so squaring
// the wider value here would give a different rms_rcp and the hoist would not
// be a refactor.
//
// ── MEASURED NEUTRAL. Defaulted off; MPK_QKV_PRO_HOIST=1 turns it on. ──
// Correct output: 4/4 correctness prompts, generated text read, all 8 ranks
// identical. Alternated OFF/ON/OFF/ON/OFF/ON in one batch, one -D the variable:
//
//   OFF  10.620 / 10.880 / 10.882   mean 10.794
//   ON   11.036 / 10.882 / 11.026   mean 10.981   (+0.187 ms)
//
// Inside the 0.26 ms noise floor and the wrong sign. The 6.98 us this deletes
// from each of 24 tiles is real, and it does not show up, because the 24 tiles
// were deriving it CONCURRENTLY -- 24 workgroups each spending 6.98 us at the
// same time is 6.98 us of makespan, not 168. What replaces it is serial: six
// workgroups fold, exchange six partials through L2 (a spin, not a barrier, but
// still a round trip), quantize, and only then does the 29-arrival release let
// the tiles start. Publish + release costs about what the deleted prologue
// cost, so the phase is a wash.
//
// The general form, and this is the seventh instance: deleting redundant work
// inside a GLM phase is absorbed unless the redundancy was SERIAL. Hoisting
// converts parallel redundancy into a serial producer plus a rendezvous, and at
// GLM's per-phase occupancy that trade is at best even. Price the hoisted
// producer's own makespan against the per-tile saving before building one.
template <int REDUCTION_SIZE,
          int ACTUAL_HIDDEN_DIM,
          int EP_PEER_SLOTS,
          int EP_SLOT_ELEMS,
          int PRO_WGS>
__device__ __forceinline__ void
_rnlm8_pro_publish(unsigned short const *__restrict__ d_res,
                   unsigned short *__restrict__ d_x_out,
                   unsigned short const *__restrict__ d_nw,
                   uint8_t *__restrict__ pub_fp8,
                   uint8_t *__restrict__ pub_scales,
                   int *__restrict__ pub_part,
                   int *__restrict__ pub_flag,
                   int slice,
                   int epoch,
                   float eps) {
  constexpr int SLICE_ELEMS = REDUCTION_SIZE / PRO_WGS;
  constexpr int VEC = 4;
  constexpr int NTHREADS = 256;
  constexpr int ITERS = SLICE_ELEMS / (NTHREADS * VEC);
  static_assert(REDUCTION_SIZE % (PRO_WGS * 128) == 0,
                "a prologue slice must be a whole number of 128-element E8M0 "
                "groups, or the published scales do not line up with the "
                "consumer's");
  static_assert(SLICE_ELEMS % (NTHREADS * VEC) == 0 && ITERS >= 1,
                "a prologue slice must be a whole number of 256 x float4 "
                "passes, the walk _rnlm8_ep_fold_slice uses");

  int const tid = threadIdx.x;
  int const base = slice * SLICE_ELEMS;
  float ssq = 0.0f;

#pragma unroll 1
  for (int v = 0; v < ITERS; v++) {
    int const off = base + (v * NTHREADS + tid) * VEC;
    float f[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int p = 0; p < EP_PEER_SLOTS; p++) {
      uint2 const pk = *reinterpret_cast<uint2 const *>(
          d_res + (size_t)p * EP_SLOT_ELEMS + off);
      f[0] += _gang_bf16_to_float((unsigned short)pk.x);
      f[1] += _gang_bf16_to_float((unsigned short)(pk.x >> 16));
      f[2] += _gang_bf16_to_float((unsigned short)pk.y);
      f[3] += _gang_bf16_to_float((unsigned short)(pk.y >> 16));
    }
    unsigned short const b[4] = {
        _gang_float_to_bf16(f[0]), _gang_float_to_bf16(f[1]),
        _gang_float_to_bf16(f[2]), _gang_float_to_bf16(f[3])};
    st_wt_u64((void *)(d_x_out + off),
              (unsigned long long)((unsigned)b[0] | ((unsigned)b[1] << 16)) |
                  ((unsigned long long)((unsigned)b[2] |
                                        ((unsigned)b[3] << 16))
                   << 32));
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float const q = _gang_bf16_to_float(b[i]);
      ssq += q * q;
    }
  }

  // Block-reduce this slice's partial. rmsnorm_rcp_amd's Phase 3, verbatim.
#pragma unroll
  for (int offset = 32; offset > 0; offset >>= 1) {
    ssq += __shfl_xor(ssq, offset);
  }
  __shared__ float pro_red[16];
  int const wave_id = tid >> 6;
  int const lane_id = tid & 63;
  int const num_waves = blockDim.x >> 6;
  if (lane_id == 0) {
    pro_red[wave_id] = ssq;
  }
  __syncthreads();
  if (wave_id == 0) {
    ssq = (lane_id < num_waves) ? pro_red[lane_id] : 0.0f;
    for (int offset = num_waves >> 1; offset > 0; offset >>= 1) {
      ssq += __shfl_xor(ssq, offset);
    }
    if (lane_id == 0) {
      pro_red[0] = ssq;
    }
  }
  __syncthreads();

  // Exchange the PRO_WGS partials. Write-through past L1 and read back nt, the
  // same pairing every XCD-local flag in this file uses; the partial has to be
  // ordered before the stamp or a peer can read a stale float under a fresh
  // epoch.
  if (tid == 0) {
    union {
      float f;
      unsigned u;
    } pv;
    pv.f = pro_red[0];
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    st_wt_u32((void *)(pub_part + slice), pv.u);
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    st_wt_u32((void *)(pub_flag + slice), (unsigned)epoch);
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

    float tot = 0.0f;
#pragma unroll 1
    for (int w = 0; w < PRO_WGS; w++) {
      int _spins = 0;
      while (ld_nt_s32(pub_flag + w) < epoch) {
        ++_spins;
        __builtin_amdgcn_s_sleep(1);
      }
    }
#pragma unroll 1
    for (int w = 0; w < PRO_WGS; w++) {
      union {
        float f;
        unsigned u;
      } rv;
      rv.u = (unsigned)ld_nt_s32(pub_part + w);
      tot += rv.f;
    }
    pro_red[0] = rsqrtf(tot / float(ACTUAL_HIDDEN_DIM) + eps);
  }
  __syncthreads();
  float const rms_rcp = pro_red[0];

  // d_x_out was written through this WG's own L1; drop the stale lines before
  // the quantizer reads them back.
  asm volatile("buffer_inv" ::: "memory");
  _gang_wave_parallel_fp8_quant_rmsnorm<SLICE_ELEMS,
                                        /*SRC_IS_GLOBAL=*/true>(
      d_x_out + base,
      d_nw + base,
      rms_rcp,
      pub_fp8 + base,
      pub_scales + base / 128);
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
}

//
// ── PRE_FOLDED ───────────────────────────────────────────────────────────
// The argument above -- "making the consumer BE the reduction leaves no
// window for a barrier to protect" -- is right about the barrier and wrong
// about the arithmetic once EP_PEER_SLOTS is 8 and the gang is 24 wide. Every
// one of qkv_a's 24 workgroups per XCD reads all 8 slots of the 6144-wide row:
// 8 x 12 KB + 12 KB of norm weight = 108 KB per tile, 192 tiles per layer per
// rank, 20.7 MB/layer/rank of identical traffic -- more than qkv_a's own
// 16.1 MB of weight. Measured, the prologue is ~60% of a 13.50 us tile and the
// resadd+fold half of it alone is ~50%.
// PRE_FOLDED says a step ahead of this one has already summed the slots and
// published the resolved bf16 row, so this pass reads ONE plane, stages it and
// squares it. It does not add d_ws either: the folder did that (under EP,
// d_ws is not a term at all -- see the divergence note above).
template <int REDUCTION_SIZE,
          int EP_PEER_SLOTS = 0,
          int EP_SLOT_ELEMS = 0,
          bool STAGE_NW = false,
          bool PRE_FOLDED = false>
__device__ __forceinline__ float
_rnlm8_resadd_norm_rcp(float const *__restrict__ d_ws,
                       unsigned short const *__restrict__ d_res, // == norm_input
                       unsigned short *__restrict__ d_x_out,
                       unsigned short *__restrict__ s_x,
                       bool store_x,
                       float eps = 1e-5f,
                       unsigned short const *__restrict__ d_nw = nullptr,
                       unsigned short *__restrict__ s_nw = nullptr) {
  constexpr int VEC = 4;
  constexpr int NTHREADS = 256;
  static_assert(REDUCTION_SIZE % (NTHREADS * VEC) == 0,
                "FUSE_RESADD wants the row to divide evenly over 256x float4");
  constexpr int ITERS = REDUCTION_SIZE / (NTHREADS * VEC);

  constexpr bool EP = (EP_PEER_SLOTS > 1) && !PRE_FOLDED;
  static_assert(!EP || EP_SLOT_ELEMS >= REDUCTION_SIZE,
                "the EP gather slot stride must span at least one row");
  // The peers other than slot 0, which is read as the plain `d_res` above.
  // Sized to 1 rather than 0 off the EP path so the array is never zero-length.
  constexpr int NEXTRA = EP ? (EP_PEER_SLOTS - 1) : 1;

  int const tid = threadIdx.x;
  float ssq = 0.0f;

  // MPK_ABL_QKV_PRO: run one of the ITERS passes and zero-fill the rest of the
  // staged row instead. WRONG OUTPUT by construction. This is the only caller
  // of the resadd prologue (qkv_a), so it isolates "prologue" from "GEMM"
  // inside SP4[0] without touching the tile map, the WG stride or the MFMA
  // count -- MPK_ATTN_HALFK already priced the GEMM half at ~0.
#ifdef MPK_ABL_QKV_PRO
  constexpr int ABL_ITERS = 1;
#else
  constexpr int ABL_ITERS = ITERS;
#endif
#pragma unroll 1
  for (int v = ABL_ITERS; v < ITERS; v++) {
    *reinterpret_cast<uint2 *>(s_x + (v * NTHREADS + tid) * VEC) = uint2{0, 0};
  }
  // MEASURED: this loop is 46.2% of the qkv_a tile (6119 of 13246 ns, SP bank
  // 0 [1][4] over [1][5]) and it moves 96 KB of L2-resident bytes, i.e. 15.7
  // GB/s per CU against a ~52 GB/s per-CU L2 share. It is at 30% of a roof,
  // not at one, and the reason is here: at REDUCTION_SIZE 6144 with 256
  // threads x float4, ITERS is 6, and `unroll 1` makes those six trips strictly
  // sequential. Each trip issues 9 loads (residual + norm weight + 7 peer
  // slots) and then consumes them immediately, so the loop pays SIX serialized
  // memory latencies with only ~9 loads in flight.
  //
  // Unrolling puts ITERS/N latencies on the critical path instead of ITERS.
  // The registers are available: worker_kernel is 284 VGPR + 36 AGPR of 512
  // and occupancy is pinned at 1 wave/SIMD by block count, so 320..480 are
  // free (see the depth-8 note in memory). One extra trip in flight costs
  // ~9 x 2 = 18 VGPRs.
  //
  // Unlike the MFMA loop further down, this `unroll 1` never carried a reason
  // -- the ROCm MAI miscompile that pins that one needs a v_mfma, and there is
  // none here.
  //
  // MEASURED, n=3 paired, GLM_RESADD_UNROLL 1 vs 6 (6 == full at ITERS=6):
  //   min-of-115-iters   10.466 10.422 10.407  ->  10.375 10.398 10.306
  //   mean of those          10.432          ->      10.360   (-0.072 ms)
  //   device avg_ms          10.689          ->      10.619   (-0.070 ms)
  // Five of six reps carried a box-level 24-27 ms outlier, so the per-rep mean
  // is unusable and this is read on `min`. -0.072 is BELOW the 0.26 ms wall
  // noise floor; what makes it believable at all is that all three unroll=6
  // mins sit below all three unroll=1 mins (3v3 rank separation, permutation
  // p=0.05) and the device clock agrees in sign. Treat it as ~-0.07 +/- 0.05,
  // not as a resolved number, and do not stack conclusions on it.
  //
  // A follow-up SP A/B (MPK_SUBPHASE_TIMING=1, unroll 1 vs 6, divided by this
  // call site's own tile count [1][5] = 1828800, identical in both arms) says
  // the wall number is NOT absorption -- it is the full transfer of a small
  // tile win:
  //
  //   [1][4] resolve loop        6094 -> 5646 ns   -448  (-7.4%)
  //   prefill fill ([0][1]-[1][4]) 842 ->  828     noise
  //   [0][2] quantizer            855 ->  875      noise
  //   [0][3] MFMA K-loop         5432 -> 5401      noise
  //   tile total                13224 -> 12750     -474  (-3.6%)
  //
  // Only the targeted line moved. qkv_a is UNDER-FILLED -- 23 tiles/XCD on 29
  // workers -- so the phase makespan IS one tile duration, and 78 layers x
  // 474 ns predicts -0.037 ms at 1:1. Measured -0.072, ratio 1.95. The extra
  // is almost certainly qkv_a's other 17.57 us, which is the EP peer wait:
  // all 8 peers ran the same shortened tile, so the wait shrank too.
  //
  // So: an IN-PLACE tile speedup in an under-filled phase transfers at 1:1 or
  // better. That is a different operation from the two qkv_a HOISTS (c72b559
  // +0.090, 7a5c559 +0.187), which moved work behind an extra rendezvous and
  // paid back in barrier what they saved in tile. Do not quote "absorbed" for
  // an in-place tile speedup. At ~2:1 this whole tile is worth ~2.0 ms.
  //
  // The unroll captured only 7.4% of the resolve loop because it did not
  // actually get the loads in flight: 54 loads/thread/tile over 5646 ns at
  // 2.4 GHz is ~251 cycles per load, i.e. ~1.5 outstanding even at full
  // unroll. The compiler unrolled but could not hoist -- the in-loop LDS store
  // to s_x and the non-__restrict__ peer pointers block the reorder. The loop
  // is at 17 GB/s per CU against a ~52 GB/s per-CU L2 share, so there is 3x
  // left and it is an issue-order problem, not a byte problem. Next move is
  // explicit batching: load every trip into registers first, then combine.
  //
  // Default is 6 because the change is free, correctness-gated 4/4 with all 8
  // ranks identical, and the sign is consistent across two independent clocks.
#ifndef MPK_RESADD_UNROLL
#define MPK_RESADD_UNROLL 6
#endif
#define MPK_RESADD_STR_(x) #x
#define MPK_RESADD_STR(x) MPK_RESADD_STR_(x)

  // MPK_RESADD_BATCH: issue every global load for every trip BEFORE consuming
  // any of them, instead of relying on the unroll to interleave them.
  //
  // The unroll above does put all six trips in one basic block, but it only
  // bought 7.4% and the loads still complete at ~251 cycles apiece (~1.5 in
  // flight). The blocker is the `ds_write` to s_x at the bottom of each trip:
  // it raises lgkmcnt, and the compiler will not sink a wait for trip v's LDS
  // store past trip v+1's global loads, so each trip's fetch group is fenced
  // off from the next. Splitting the loop hoists all 48 (EP) global loads
  // above the first ds_write, where nothing orders them against each other.
  //
  // Cost is registers: ITERS(6) x (1 + NEXTRA(7)) uint2 = 96 VGPRs, plus 12
  // more under STAGE_NW. worker_kernel is 284 VGPR + 36 AGPR = 320 of 512 and
  // occupancy is pinned at 1 wave/SIMD by block count, so ~480 is the real
  // ceiling and this lands near 428. That is inside the budget but not by
  // much -- if the compiler spills to scratch this is strictly worse, so the
  // VGPR count has to be read off the built image before believing an A/B.
  //
  // MEASURED. It does not spill: worker_kernel stays at exactly 284 VGPR /
  // 36 AGPR / private_seg 596 with vgpr_spill_count 0, unchanged from the
  // baseline -- this region was never the register peak. SP A/B against the
  // same tile-count guard [1][5] = 1828800:
  //
  //   [1][4] resolve loop   5646 -> 4009 ns   -1637  (-29.0%)
  //   [0][2] quantizer       875 ->  852      noise
  //   [0][3] MFMA K-loop    5401 -> 5396      noise
  //   tile total           12750 -> 11101     -1650  (-12.9%)
  //
  // and at the wall, n=3 paired with its control in the same batch:
  //
  //   min-of-115   10.345 10.400 10.385  ->  10.282 10.226 10.274
  //   mean of min      10.377           ->      10.261   (-0.116 ms)
  //   device avg_ms    10.650           ->      10.491   (-0.159 ms)
  //
  // 3v3 rank separation again (worst batched min 10.282 beats best unbatched
  // 10.345). 78 x 1650 ns predicts -0.129 at 1:1, so this transferred at
  // ~1:1 -- which also revises the 1.95:1 claimed for the unroll above: that
  // was a -0.037 ms signal read at the edge of resolution and it was
  // flattered by noise. **Use 1:1 for an in-place tile speedup in an
  // under-filled phase.** Still the important part: not absorbed.
  //
  // NOT bit-identical to the unbatched path (agree@142/264 vs the resunroll
  // reference) even though the add order is unchanged -- splitting the loop
  // lets the compiler contract FMAs differently, and that drifts a token
  // eventually. Correctness gate is 4/4 PASS with all 8 ranks identical and
  // the text read: Paris, Rayleigh scattering, a correct prime definition.
  //
  // Remaining headroom here: 96 KB in 4009 ns is 24 GB/s per CU against the
  // ~52 GB/s per-CU L2 share, so the loop went 33% -> 46% of roof.
#ifndef MPK_RESADD_BATCH
#define MPK_RESADD_BATCH 1
#endif
#if MPK_RESADD_BATCH
  uint2 bat_r[ITERS];
  uint2 bat_p[ITERS][NEXTRA];
  float4 bat_w[(!EP && !PRE_FOLDED) ? ITERS : 1];
  uint64_t bat_nw[STAGE_NW ? ITERS : 1];
#pragma unroll
  for (int v = 0; v < ABL_ITERS; v++) {
    int const off = (v * NTHREADS + tid) * VEC;
    bat_r[v] = *reinterpret_cast<uint2 const *>(d_res + off);
    if constexpr (EP) {
#pragma unroll
      for (int p = 0; p < NEXTRA; p++) {
        bat_p[v][p] = *reinterpret_cast<uint2 const *>(
            d_res + (size_t)(p + 1) * EP_SLOT_ELEMS + off);
      }
    } else if constexpr (!PRE_FOLDED) {
      bat_w[v] = *reinterpret_cast<float4 const *>(d_ws + off);
    }
    if constexpr (STAGE_NW) {
      uint64_t const *nwp = reinterpret_cast<uint64_t const *>(d_nw + off);
      bat_nw[v] = *(__attribute__((address_space(1))) uint64_t const *)nwp;
    }
  }
#endif

  _Pragma(MPK_RESADD_STR(unroll MPK_RESADD_UNROLL))
  for (int v = 0; v < ABL_ITERS; v++) {
    int const off = (v * NTHREADS + tid) * VEC;
#if MPK_RESADD_BATCH
    uint2 const r = bat_r[v];
#else
    uint2 const r = *reinterpret_cast<uint2 const *>(d_res + off);
#endif
    if constexpr (STAGE_NW) {
      // Four bf16 is eight bytes, and `off` is a multiple of four elements, so
      // this is one aligned dwordx2 each way. Two-step addrspace(1) cast for
      // the same reason rmsnorm_rcp_amd uses one: a flat_load would bump
      // lgkmcnt and be waited on by the ds_write right below it.
#if MPK_RESADD_BATCH
      *reinterpret_cast<uint64_t *>(s_nw + off) = bat_nw[v];
#else
      uint64_t const *nwp = reinterpret_cast<uint64_t const *>(d_nw + off);
      *reinterpret_cast<uint64_t *>(s_nw + off) =
          *(__attribute__((address_space(1))) uint64_t const *)nwp;
#endif
    }

    float f[4];
    if constexpr (EP) {
      // All NEXTRA slot loads issued before any is consumed: they are
      // independent addresses on distinct pages, so the extra ranks cost
      // bandwidth (2 KB per peer at REDUCTION_SIZE 2048) and not latency.
      uint2 pk[NEXTRA];
#pragma unroll
      for (int p = 0; p < NEXTRA; p++) {
#if MPK_RESADD_BATCH
        pk[p] = bat_p[v][p];
#else
        pk[p] = *reinterpret_cast<uint2 const *>(
            d_res + (size_t)(p + 1) * EP_SLOT_ELEMS + off);
#endif
      }
      f[0] = _gang_bf16_to_float((unsigned short)r.x);
      f[1] = _gang_bf16_to_float((unsigned short)(r.x >> 16));
      f[2] = _gang_bf16_to_float((unsigned short)r.y);
      f[3] = _gang_bf16_to_float((unsigned short)(r.y >> 16));
#pragma unroll
      for (int p = 0; p < NEXTRA; p++) {
        f[0] += _gang_bf16_to_float((unsigned short)pk[p].x);
        f[1] += _gang_bf16_to_float((unsigned short)(pk[p].x >> 16));
        f[2] += _gang_bf16_to_float((unsigned short)pk[p].y);
        f[3] += _gang_bf16_to_float((unsigned short)(pk[p].y >> 16));
      }
    } else if constexpr (PRE_FOLDED) {
      // The row is already resolved. Nothing to add -- but it still has to go
      // through bf16 on the way to s_x and ssq, and it already IS bf16, so the
      // round-trip is exact and this stays bit-identical to the folded path.
      f[0] = _gang_bf16_to_float((unsigned short)r.x);
      f[1] = _gang_bf16_to_float((unsigned short)(r.x >> 16));
      f[2] = _gang_bf16_to_float((unsigned short)r.y);
      f[3] = _gang_bf16_to_float((unsigned short)(r.y >> 16));
    } else {
#if MPK_RESADD_BATCH
      float4 const w = bat_w[v];
#else
      float4 const w = *reinterpret_cast<float4 const *>(d_ws + off);
#endif
      f[0] = w.x + _gang_bf16_to_float((unsigned short)r.x);
      f[1] = w.y + _gang_bf16_to_float((unsigned short)(r.x >> 16));
      f[2] = w.z + _gang_bf16_to_float((unsigned short)r.y);
      f[3] = w.w + _gang_bf16_to_float((unsigned short)(r.y >> 16));
    }

    unsigned short const b[4] = {
        _gang_float_to_bf16(f[0]), _gang_float_to_bf16(f[1]),
        _gang_float_to_bf16(f[2]), _gang_float_to_bf16(f[3])};
    uint2 const packed = {(unsigned)b[0] | ((unsigned)b[1] << 16),
                          (unsigned)b[2] | ((unsigned)b[3] << 16)};
    *reinterpret_cast<uint2 *>(s_x + off) = packed;
    if (store_x) {
      st_wt_u64((void *)(d_x_out + off),
                (unsigned long long)packed.x |
                    ((unsigned long long)packed.y << 32));
    }

#pragma unroll
    for (int i = 0; i < 4; i++) {
      float const q = _gang_bf16_to_float(b[i]);
      ssq += q * q;
    }
  }

  // Phase 3 of rmsnorm_rcp_amd, verbatim: wave reduce, then one cross-wave
  // pass through LDS. The trailing __syncthreads() publishes red[0] and, here,
  // also publishes s_x to the quantizer's different thread mapping.
#pragma unroll
  for (int offset = 32; offset > 0; offset >>= 1) {
    ssq += __shfl_xor(ssq, offset);
  }

  __shared__ float red[16];
  int const wave_id = tid >> 6;
  int const lane_id = tid & 63;
  int const num_waves = blockDim.x >> 6;
  if (lane_id == 0) {
    red[wave_id] = ssq;
  }
  __syncthreads();
  if (wave_id == 0) {
    ssq = (lane_id < num_waves) ? red[lane_id] : 0.0f;
    for (int offset = num_waves >> 1; offset > 0; offset >>= 1) {
      ssq += __shfl_xor(ssq, offset);
    }
    if (lane_id == 0) {
      red[0] = ssq;
    }
  }
  __syncthreads();

  return rsqrtf(red[0] / float(REDUCTION_SIZE) + eps);
}

// _rnlm8_resadd_norm_rcp without the residual: the plain path's
// rmsnorm_rcp_amd, plus the LDS staging of the row and the norm weight.
//
// rmsnorm_rcp_amd deliberately does *not* cache the row -- "nothing here
// revisits the input, so caching it would only hold VGPRs" -- which was true
// when the quantizer re-read the row from a still-L1-hot global address. It is
// the re-read that is the problem now, not its cost: it is vmem issued between
// the caller's A-tile prefetch and the MFMAs that consume it, and vmcnt is
// in-order on gfx9, so waiting for the re-read also waits for the prefetch.
// Staging both operands here turns the entire quantizer into LDS traffic.
//
// The row is walked once instead of twice, so this is not extra work; the LDS
// costs 2 bytes per element and no occupancy, since the persistent kernel
// already runs one workgroup per CU.
//
// ACTUAL_HIDDEN_DIM is the RMS divisor, as in rmsnorm_rcp_amd. There is no
// NORM_SPAN parameter because this kernel's only caller never used one: q_a,
// the shape that needs a narrowed span, gets it from a narrowed REDUCTION_SIZE
// instead (see gang_rmsnorm_linear_mxfp8_bias_mla_kvupd_mi300.cuh).
template <int REDUCTION_SIZE, int ACTUAL_HIDDEN_DIM>
__device__ __forceinline__ float
_rnlm8_stage_norm_rcp(unsigned short const *__restrict__ d_in,
                      unsigned short const *__restrict__ d_nw,
                      unsigned short *__restrict__ s_x,
                      unsigned short *__restrict__ s_nw,
                      float eps = 1e-5f) {
  // The traversal is rmsnorm_rcp_amd's, element for element and in the same
  // order, so `ssq` accumulates the same partial sums into the same lanes and
  // the returned rms_rcp is bit-identical to the function this replaces. Only
  // the two ds_writes are new.
  constexpr int NTHREADS = 256;
  constexpr int VEC_SIZE = (REDUCTION_SIZE % (NTHREADS * 8) == 0) ? 8 : 4;
  constexpr int VEC_ITERS = REDUCTION_SIZE / (NTHREADS * VEC_SIZE);
  constexpr int VEC_END = VEC_ITERS * NTHREADS * VEC_SIZE;
  using gu16 = __attribute__((address_space(1))) unsigned short const *;
  using gu64 = __attribute__((address_space(1))) uint64_t const *;

  int const tid = threadIdx.x;
  int const nthreads = blockDim.x;
  float ssq = 0.0f;

  auto stage4 = [&](int off) -> uint64_t {
    uint64_t const x = *(gu64)reinterpret_cast<uint64_t const *>(d_in + off);
    uint64_t const w = *(gu64)reinterpret_cast<uint64_t const *>(d_nw + off);
    *reinterpret_cast<uint64_t *>(s_x + off) = x;
    *reinterpret_cast<uint64_t *>(s_nw + off) = w;
    return x;
  };

#pragma unroll 1
  for (int v = 0; v < VEC_ITERS; v++) {
    int const off = (v * nthreads + tid) * VEC_SIZE;
    uint64_t const lo = stage4(off);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float const q =
          _gang_bf16_to_float((unsigned short)((lo >> (16 * i)) & 0xFFFFu));
      ssq += q * q;
    }
    if constexpr (VEC_SIZE == 8) {
      uint64_t const hi = stage4(off + 4);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float const q =
            _gang_bf16_to_float((unsigned short)((hi >> (16 * i)) & 0xFFFFu));
        ssq += q * q;
      }
    }
  }
  // Dead at every shape this kernel is instantiated at (REDUCTION_SIZE is 2048
  // or 6144, both a whole number of 256x8 passes); kept because
  // REDUCTION_SIZE % 128 == 0 is the only thing the kernel actually asserts.
  for (int i = VEC_END + tid; i < REDUCTION_SIZE; i += nthreads) {
    unsigned short const x = ((gu16)d_in)[i];
    s_x[i] = x;
    s_nw[i] = ((gu16)d_nw)[i];
    float const q = _gang_bf16_to_float(x);
    ssq += q * q;
  }

  // Phase 3 of rmsnorm_rcp_amd, verbatim.
#pragma unroll
  for (int offset = 32; offset > 0; offset >>= 1) {
    ssq += __shfl_xor(ssq, offset);
  }

  __shared__ float red[16];
  int const wave_id = tid >> 6;
  int const lane_id = tid & 63;
  int const num_waves = blockDim.x >> 6;
  if (lane_id == 0) {
    red[wave_id] = ssq;
  }
  __syncthreads();
  if (wave_id == 0) {
    ssq = (lane_id < num_waves) ? red[lane_id] : 0.0f;
    for (int offset = num_waves >> 1; offset > 0; offset >>= 1) {
      ssq += __shfl_xor(ssq, offset);
    }
    if (lane_id == 0) {
      red[0] = ssq;
    }
  }
  __syncthreads();

  return rsqrtf(red[0] / float(ACTUAL_HIDDEN_DIM) + eps);
}

template <int BATCH_SIZE,
          int OUTPUT_PER_WG,
          int REDUCTION_SIZE,
          int ACTUAL_HIDDEN_DIM = REDUCTION_SIZE,
          bool WRITE_THROUGH = false,
          bool FUSE_RESADD = false,
          // Expert parallelism: > 1 makes the FUSE_RESADD prologue the
          // cross-rank reduction, reading `norm_input_ptr` as the symmetric
          // gather buffer. See _rnlm8_resadd_norm_rcp.
          int EP_PEER_SLOTS = 0,
          // The EP slots were already summed into `norm_input_ptr` by a fold
          // step ahead of this one -- read one plane, not EP_PEER_SLOTS of
          // them. See _rnlm8_ep_fold_slice.
          bool EP_PRE_FOLDED = false,
          // Opt in to the bank-0 [1][2][3] region split. Off everywhere but
          // qkv_a: ~20 call sites reach this kernel, and a shared `tile_idx
          // == 0` guard made those slots somebody else's tile (see 9710303).
          // With the split owned by one call site, [1][2][3] are qkv_a's
          // prologue / quantizer / MFMA and [1][5] is its own tile count, so
          // the three divide out against the [0][0] whole-tile timer.
          bool SP_QKV = false,
          // The prologue was hoisted: a group ahead of this phase published
          // the quantized row and `pro_pub_ptr` points at it, so Step 1+2 and
          // the quantizer are both deleted and the tile copies E4M3 + E8M0
          // into LDS instead of deriving them. See _rnlm8_pro_publish.
          bool PRO_PUB = false>
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
    int tile_idx,
    // FUSE_RESADD only. Defaulted so the twenty-odd existing call sites, none
    // of which resolve a residual, stay as they are. The residual itself is
    // norm_input_ptr, which under the fold holds the unresolved residual
    // stream rather than a finished row.
    void const *resadd_workspace_f32_ptr = nullptr, // [batch, REDUCTION_SIZE] f32
    void *resadd_x_out_ptr = nullptr,               // [batch, REDUCTION_SIZE] bf16
    // PRO_PUB only: this XCD's published prologue. REDUCTION_SIZE bytes of
    // E4M3 followed by REDUCTION_SIZE/128 bytes of E8M0.
    void const *pro_pub_ptr = nullptr) {

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
  // MPK_ATTN_HALFK: run half the K-loop and keep everything else identical --
  // same tile map, same WG stride, same MFMA shape, half the weight bytes off
  // HBM. WRONG OUTPUT by construction. This prices MXFP4 for the attention /
  // dense weights (task #66) before any quantization plumbing is written:
  // MXFP4 halves bytes at constant FLOPs, HALFK halves both, so HALFK is an
  // UPPER BOUND on the MXFP4 gain. If the bound is small the lever is dead.
  // Clamped to the depth-4 pipeline's minimum, so W_UV's K=512 (MFMA_ITERS 4)
  // is left at full width rather than dropped below the static_assert.
  //
  // ── MEASURED. The bound is small; task #66 is not worth building. ──
  // One instrumented NP=8 pair, MPK_SUBPHASE_TIMING=1, same batch, us/layer:
  //
  //   bank 4 (cnt 1219200)   base   halfk   delta
  //     qkv_a               31.32   26.41   -4.91
  //     qkv barrier         13.37   12.19   -1.18
  //     q_b + kvupd         10.09    9.93   -0.16
  //     W_UK + barrier      23.26   22.33   -0.93
  //     bank 4 total       103.82   97.12   -6.70
  //   bank 3 total          88.46   89.09   +0.63
  //   instrumented wall     14.779  14.192  -0.587 ms
  //
  // Two things fall out. First, the MoE half does not move (+0.63, noise), so
  // HALFK did not shift the TopK or the EP balance -- this is the one
  // wrong-output probe on GLM whose wall reading is not confounded the way
  // ABL_QKV_PRO's was. Second, the reachable prize is tiny: halving qkv_a's K
  // halves its bytes AND its FLOPs and buys 15.7% of the slot, because the
  // tile is prologue-heavy (48fea7f) and the phase is one grid-stride round
  // whose makespan is one tile. MXFP4 halves bytes only, so 4.91 us/layer is a
  // hard ceiling on qkv_a, ~0.38 ms of wall. Scaling that across every dense
  // stage -- q_b, the already-8-way-sharded o_proj, W_UK/W_UV -- puts the whole
  // "MXFP4 for the attention weights" program under ~0.9 ms, upper bound,
  // before any dequant VALU. Do not build it as a route to 4 ms.
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
  // slots 1 and 2 compute k-tiles that do not exist.
  static_assert(MFMA_ITERS % 4 == 0,
                "Depth-4 pipeline requires REDUCTION_SIZE % 512 == 0");

  // ── Wave tiling ─────────────────────────────────────────────────────────
  // Above this many k-tiles the A operand no longer fits in registers
  // (8 VGPRs each), so the N-parallel branch falls back to the rotating
  // depth-4 pipeline. See the note at the branch.
  //
  // Currently 0, i.e. the straight-line branch is off. It is correct in
  // principle and its asm is clean, but at MFMA_ITERS=8 it puts the function
  // at 248 VGPRs, and the allocator responds by spilling the last B tile into
  // the accumulator file and then coalescing the final MFMA's destination
  // onto it -- `v_mfma a[0:3], v[128:135], a[0:7], a[8:11]`, a destination
  // overlapping srcB, which MAI forbids. It is 2 of the 664 v_mfma in the
  // whole translation unit and both are this kernel; nothing else in the
  // megakernel comes close enough to the register ceiling to trip it.
  //
  // More to the point it bought nothing: 3.853 and 3.854 ms against a
  // 3.855 ms baseline. The timing is valid even though those runs decoded
  // garbage -- the same loads and MFMAs issue either way. q_b is 240 workers
  // wide and the barrier after it is 32 wide, so finishing early only makes
  // the 240 wait longer. See the note in memory: the layer is barrier-bound,
  // not bandwidth-bound. Re-enable only alongside a fix to the barrier shape.
  constexpr int FULL_PRELOAD_ITERS = 0;
  constexpr int NUM_WAVES = 4;
  constexpr int TILES_PER_WAVE = OUTPUT_PER_WG / 16 / NUM_WAVES;

  // ── Token activation in shared memory ────────────────────────────────────
  constexpr int FP8_TOK_DATA = REDUCTION_SIZE;

  uint8_t const *W = (uint8_t const *)weight_ptr;
  unsigned short const *d_bias = (unsigned short const *)bias_ptr;
  unsigned short *d_output = (unsigned short *)output_ptr;

  // ── LDS prologue ────────────────────────────────────────────────────────
  // Stage the row and the norm weight in LDS so the quantizer issues no vmem,
  // and hoist the first four A-tiles above it so their latency runs underneath.
  //
  // The two halves are one change, not two. vmcnt is in-order on gfx9: hoisting
  // the prefetch on its own just moves the exposed weight latency from after
  // the quantizer to before it, because the quantizer's own global reads are
  // younger and waiting on them drains the prefetch with them. The prologue has
  // to touch no global memory at all for the hoist to buy anything.
  //
  // GLM_PROLOGUE_PREFETCH=0 is the ablation; it restores the previous code
  // exactly, including the LDS footprint. It is an ablation and not a
  // correctness fallback: rms_rcp is bit-identical either way (the staging
  // traversal is rmsnorm_rcp_amd's, element for element), and the quantizer
  // reads the same values, only out of LDS instead of global.
  //
  // Measured, GLM-5 744B, NP=8 EP, 78 layers, MPK_SUBPHASE_TIMING=1, SP4
  // (gang_mla_attn_fused), aggregate worker-seconds over cnt 1219200:
  //
  //   SP4 slot           off      on      delta
  //   [0] qkv_a        26.657  24.859   -1.80  (-6.7%)
  //   [1] qkv barrier  12.532  11.570   -0.96
  //   [2] q_b + kvupd  63.827  51.678  -12.15  (-19.0%)
  //   [3] q_b barrier   1.399   1.284   -0.12
  //   [4] MLA decode    1.109   1.119   +0.01
  //   [5] dec barrier  16.466  17.575   +1.11
  //   [6] merge         2.214   2.191   -0.02
  //   SP4 total       124.20  110.28   -13.93 (-11.2%)
  //
  // Uninstrumented end-to-end decode, three runs each:
  //   off 17.993 / 18.220 / 18.227 ms   (mean 18.147)
  //   on  17.284 / 17.550 / 17.412 ms   (mean 17.415, -0.73 ms, -4.0%)
  //
  // The two GEMM stages give up 14 worker-s and the decode barrier takes back
  // 1.1 of it -- these workers now arrive early and wait -- so most of it
  // translates. Zero spills at all four instantiations.
#ifdef MPK_GLM_PROLOGUE_PREFETCH_OFF
  constexpr bool LDS_PROLOGUE = false;
#else
  constexpr bool LDS_PROLOGUE = true;
#endif

  extern __shared__ char _rnlm8_smem[];
  uint8_t *s_tok_fp8 = (uint8_t *)_rnlm8_smem;
  uint8_t *s_tok_scales = s_tok_fp8 + FP8_TOK_DATA;
  // The resadd staging buffer sits past the quantizer's, rounded up to 128 B so
  // its uint2 traffic stays aligned. Only the K-parallel branch reuses
  // _rnlm8_smem (as lds_reduce, from offset 0) and that is after the MFMA, by
  // which point s_x_bf16 is dead.
  constexpr int RESADD_SMEM_OFF =
      ((FP8_TOK_DATA + NUM_BLOCKS_32 + 127) / 128) * 128;
  unsigned short *s_x_bf16 =
      (unsigned short *)(_rnlm8_smem + RESADD_SMEM_OFF);
  // The norm weight follows it. REDUCTION_SIZE is a multiple of 128, so the
  // row is a multiple of 256 B and this stays 128 B-aligned without padding.
  // Worst case here is REDUCTION_SIZE 6144 (qkv_a): 6 KB quantizer + 12 KB row
  // + 12 KB weight is 30 KB of the 155 KB the gfx950 worker is launched with,
  // and the persistent kernel runs one workgroup per CU regardless, so none of
  // it costs occupancy.
  constexpr int NW_SMEM_OFF =
      RESADD_SMEM_OFF + (LDS_PROLOGUE ? REDUCTION_SIZE * 2 : 0);
  unsigned short *s_nw_bf16 =
      (unsigned short *)(_rnlm8_smem + NW_SMEM_OFF);
  static_assert(!LDS_PROLOGUE ||
                    NW_SMEM_OFF + REDUCTION_SIZE * 2 <=
                        mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE,
                "the LDS prologue does not fit in the worker's dynamic LDS");

  int const tid = threadIdx.x;
  int const warp_id = tid >> 6;
  int const lane_id = tid & 63;
  int const col = lane_id & 15;
  int const g = lane_id >> 4;

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  unsigned long long _sp_t0 = 0, _sp_t1 = 0, _sp_t2 = 0, _sp_t3 = 0;
  // Every tile of the one opted-in call site, not tile 0 of all of them.
  bool _sp_rec = (SP_QKV && tid == 0 && g_subphase_active);
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
  //
  // PRO_PUB deletes this whole step. A group of workgroups ahead of the phase
  // has already resolved the row, taken the RMSNorm reciprocal and quantized,
  // and published E4M3 + one E8M0 per 128 -- exactly what the quantizer below
  // would have written into LDS. See _rnlm8_pro_publish.
  (void)norm_output_ptr;
  unsigned short const *input_row = nullptr;
  float rms_rcp = 0.0f;
  if constexpr (PRO_PUB) {
    (void)resadd_workspace_f32_ptr;
    (void)resadd_x_out_ptr;
  } else if constexpr (FUSE_RESADD) {
    // norm_input_ptr is the *residual*, not an already-resolved row: the row
    // this pass normalizes does not exist yet, it is workspace + residual, and
    // producing it is what this pass does.
    static_assert(ACTUAL_HIDDEN_DIM == REDUCTION_SIZE,
                  "FUSE_RESADD has no padded-row variant: the workspace and "
                  "residual buffers are exactly REDUCTION_SIZE wide");
    // Under EP the slot stride is the whole [batch, REDUCTION_SIZE] plane, so
    // slot p of token `tok_idx` sits at p * BATCH_SIZE * REDUCTION_SIZE +
    // tok_idx * REDUCTION_SIZE -- the base pointer below is unchanged and the
    // helper adds the slot term itself.
    rms_rcp = _rnlm8_resadd_norm_rcp<REDUCTION_SIZE,
                                     EP_PEER_SLOTS,
                                     BATCH_SIZE * REDUCTION_SIZE,
                                     /*STAGE_NW=*/LDS_PROLOGUE,
                                     /*PRE_FOLDED=*/EP_PRE_FOLDED>(
        (float const *)resadd_workspace_f32_ptr + tok_idx * REDUCTION_SIZE,
        (unsigned short const *)norm_input_ptr + tok_idx * REDUCTION_SIZE,
        (unsigned short *)resadd_x_out_ptr + tok_idx * REDUCTION_SIZE,
        s_x_bf16,
        // Pre-folded, the folder already published the resolved row -- which
        // is the row this pass would be re-storing, byte for byte.
        /*store_x=*/!EP_PRE_FOLDED && wg_idx == 0,
        /*eps=*/1e-5f,
        (unsigned short const *)norm_weight_ptr,
        s_nw_bf16);
    input_row = s_x_bf16;
  } else {
    (void)resadd_workspace_f32_ptr;
    (void)resadd_x_out_ptr;
    unsigned short const *raw_row =
        (unsigned short const *)norm_input_ptr + tok_idx * REDUCTION_SIZE;
    if constexpr (LDS_PROLOGUE) {
      rms_rcp = _rnlm8_stage_norm_rcp<REDUCTION_SIZE, ACTUAL_HIDDEN_DIM>(
          raw_row, (unsigned short const *)norm_weight_ptr, s_x_bf16,
          s_nw_bf16);
      input_row = s_x_bf16;
    } else {
      input_row = raw_row;
      rms_rcp = gang_rmsnorm_detail::rmsnorm_rcp_amd<REDUCTION_SIZE,
                                                     ACTUAL_HIDDEN_DIM>(
          input_row);
    }
  }

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  // Split the prologue: everything above is the residual resolve, the LDS
  // staging of the row and the norm weight, and the RMSNorm reciprocal --
  // the part a hoist could delete. Everything below up to _sp_t1 is the
  // depth-4 weight prefetch fill, which is this tile's own bytes and could
  // not be hoisted. [0][1] is still the whole prologue, so the fill is
  // [0][1] - [1][4].
  //
  // MEASURED at GLM-5 744B / MXFP4 attention weights, MPK_SUBPHASE_TIMING=1,
  // divided by [1][5] = 1828800 (this call site's own tile count, 186
  // tiles/layer/rank over 126 iters x 78 layers):
  //
  //   [1][4]          resolve + EP fold + RMSNorm rcp + LDS stage  6119 ns  46.2%
  //   [0][1]-[1][4]   depth-4 prefetch fill issue                   841 ns   6.3%
  //   [0][2]          FP8 quantizer                                 851 ns   6.4%
  //   [0][3]          MFMA K-loop + epilogue                       5435 ns  41.0%
  //                                                        tile  13246 ns
  //
  // The tile total agrees with the 13.5 us in the barrier-skew decomposition,
  // where qkv_a's phase is 32.40 us/layer = 17.57 us EP peer wait + 14.83 us
  // of tile. So the phase makespan carries exactly ONE tile.
  //
  // Two things follow, and both correct earlier readings:
  //
  // 1. qkv_a is NOT MFMA-K-loop bound. The K-loop is 41%. Deepening the
  //    depth-4 pipeline here (the FULL_PRELOAD_ITERS branch above, and the
  //    gpt-oss depth-8 port) is aimed at 41% of the tile at best, which is
  //    consistent with both measuring neutral.
  // 2. The one load-width experiment run against this stage, MPK_QUANT_V16
  //    (0 -> 657 global_load_dwordx4, +0.062 ms), widened the QUANTIZER --
  //    [0][2], 6.4% of the tile. It was aimed at 6%, so its null result says
  //    nothing about the 46% above it.
  //
  // The resolve reads EP_PEER_SLOTS=8 x REDUCTION_SIZE bf16 = 96 KB per tile
  // and takes 6119 ns = 15.7 GB/s per CU. Every one of the 186 tiles reads the
  // same 96 KB, so it is L2-resident, and the per-CU L2 share is ~52 GB/s --
  // this is running at ~30% of it, i.e. it is latency/MLP-bound, not at a
  // roof. Note this is a DIFFERENT claim from the two hoist experiments
  // (c72b559 +0.090, 7a5c559 +0.187): those MOVED the work to a smaller
  // worker set behind an extra rendezvous, and paid back in barrier what they
  // saved in tile. Making the resolve faster in place has no such offset.
  if (_sp_rec) {
    atomicAdd(&g_subphase_ns[1][4],
              (__builtin_amdgcn_s_memrealtime() - _sp_t0) * 10);
  }
#endif
  // ── The hoisted A-tile prefetch ─────────────────────────────────────────
  // Both MFMA branches below open by filling the depth-4 pipeline's four
  // slots, and both do it after the quantizer. Issue that fill here instead:
  // it depends only on wg_idx, warp_id and the lane, all of which are known on
  // entry, and the quantizer that follows is now pure LDS, so the loads stay
  // outstanding across it and the first MFMA waits on a counter that has
  // already had the whole quantizer to drain.
  //
  // Only the first group is hoisted, and only when it is the *only* group of
  // its wave -- the two GLM shapes that matter are QB_OUTPUT_PER_WG 64
  // (N-parallel, TILES_PER_WAVE 1) and QKV_OUTPUT_PER_WG 16 (K-parallel, one
  // fill per wave). A wider OUTPUT_PER_WG would hold these 32 VGPRs live
  // across every later tile_iter for no benefit, so it keeps the old form.
  constexpr bool HOIST_PREFILL =
      LDS_PROLOGUE && (OUTPUT_PER_WG >= 64 ? (TILES_PER_WAVE == 1) : true);
  i32x8_t ph_a[HOIST_PREFILL ? 4 : 1];
  int ph_sa[HOIST_PREFILL ? 4 : 1];
  if constexpr (HOIST_PREFILL) {
    // N-parallel gives each wave its own 16 rows starting at k-tile 0;
    // K-parallel gives all four waves the same 16 rows and splits K.
    int const h_row = (OUTPUT_PER_WG >= 64) ? (warp_id * 16 + col) : col;
    int const h_ki0 =
        (OUTPUT_PER_WG >= 64) ? 0 : (warp_id * (MFMA_ITERS / NUM_WAVES));
    uint8_t const *h_data = wg_data + static_cast<int64_t>(h_row) * REDUCTION_SIZE;
    int const h_scale_base = h_row * NUM_BLOCKS_32;
#pragma unroll
    for (int ki = 0; ki < 4; ki++) {
      ph_a[ki] =
          _gang_load_fp8_mfma_b_g(h_data, (h_ki0 + ki) * K_PER_MFMA, g);
      ph_sa[ki] = (int)_gang_ld_g<uint8_t>(wg_scales + h_scale_base +
                                           (h_ki0 + ki) * 4 + g);
    }
  }

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  if (_sp_rec) {
    _sp_t1 = __builtin_amdgcn_s_memrealtime();
  }
#endif
  if constexpr (PRO_PUB) {
    // Straight copy of the published row into the LDS the MFMA loop reads.
    // FP8_TOK_DATA bytes of E4M3 plus REDUCTION_SIZE/128 E8M0 -- 6144 + 48 at
    // qkv_a's shape, against the 24 KB of bf16 the deleted prologue staged.
    // dwordx4 over the data, bytes over the tail of the scales.
    constexpr int PUB_SCALES = REDUCTION_SIZE / 128;
    // The publisher lays the tokens out back to back, row then scales, so this
    // stride is derivable here without knowing the rest of its block.
    uint8_t const *pub = (uint8_t const *)pro_pub_ptr +
                         (size_t)tok_idx * (FP8_TOK_DATA + PUB_SCALES);
    using gu32 = __attribute__((address_space(1))) unsigned const *;
    int4 *s4 = (int4 *)s_tok_fp8;
    gu32 g4 = (gu32)pub;
    for (int i = tid; i < FP8_TOK_DATA / 16; i += blockDim.x) {
      int4 v;
      v.x = (int)g4[i * 4 + 0];
      v.y = (int)g4[i * 4 + 1];
      v.z = (int)g4[i * 4 + 2];
      v.w = (int)g4[i * 4 + 3];
      s4[i] = v;
    }
    using gu8 = __attribute__((address_space(1))) uint8_t const *;
    gu8 gs = (gu8)(pub + FP8_TOK_DATA);
    for (int i = tid; i < PUB_SCALES; i += blockDim.x) {
      s_tok_scales[i] = gs[i];
    }
    if constexpr (HOIST_PREFILL) {
      asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
      __builtin_amdgcn_s_barrier();
    } else {
      __syncthreads();
    }
  } else {
    _gang_wave_parallel_fp8_quant_rmsnorm<REDUCTION_SIZE,
                                          /*SRC_IS_GLOBAL=*/!FUSE_RESADD &&
                                              !LDS_PROLOGUE,
                                          /*NW_IS_LDS=*/LDS_PROLOGUE,
                                          /*BARRIER_LDS_ONLY=*/HOIST_PREFILL>(
        input_row,
        LDS_PROLOGUE ? s_nw_bf16 : (unsigned short const *)norm_weight_ptr,
        rms_rcp,
        s_tok_fp8,
        s_tok_scales);
  }

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

      if constexpr (MFMA_ITERS <= FULL_PRELOAD_ITERS) {
        // ── Straight-line: the whole K reduction fits in registers ──────────
        //
        // At K=1024 (q_b) MFMA_ITERS is 8, so the rotating depth-4 pipeline
        // below runs exactly two iterations -- and pays for the privilege
        // twice over. `#pragma unroll 1` forbids the unroll that would let the
        // compiler rotate registers, so instead it emits ~35 v_mov at the loop
        // bottom to shuffle the refilled tiles into the slot registers, and a
        // `s_waitcnt vmcnt(0)` in front of them to make the copies legal. That
        // drains every prefetch at the end of each iteration.
        //
        // Two iterations of a pipeline is not a pipeline. Issue all
        // MFMA_ITERS tiles up front instead: 2*MFMA_ITERS global_load_dwordx4
        // in flight before the first MFMA, no rotation, no copies, and each
        // MFMA waits on a *decreasing* vmcnt rather than zero. 8 tiles is 64
        // VGPRs of A operand, which is what the depth-4 version already peaked
        // at (4 held + 4 in flight), so this costs no occupancy.
        //
        // The bound is register pressure: MFMA_ITERS=16 (K=2048, the dense
        // MLP) would need 128 VGPRs of A and spill, so that shape keeps the
        // rotating pipeline.
        i32x8_t a[MFMA_ITERS];
        int sa[MFMA_ITERS];
#pragma unroll
        for (int ki = 0; ki < MFMA_ITERS; ki++) {
          a[ki] = _gang_load_fp8_mfma_b_g(w_data_row, ki * K_PER_MFMA, g);
          sa[ki] =
              (int)_gang_ld_g<uint8_t>(wg_scales + row_scale_base + ki * 4 + g);
        }
        // The B operands come out of LDS and have to be preloaded too, into
        // their own registers -- this is hazard 1 from
        // tests/standalone/test_mfma_pipeline_hazards.hip.
        //
        // Left to itself the allocator hands every b the *same* bank, so the
        // asm reads
        //     v_mfma  a[0:3], v[14:21], v[144:151], ...
        //     ds_read_b128 v[14:17], v2 offset:128
        // -- an LDS write into v[14:21] while the MFMA immediately above is
        // still sampling it. lgkmcnt tracks when LDS data reaches the VGPR,
        // not when the MFMA has finished reading its sources, so the next
        // MFMA sees mixed operands. The hazard test calls this intermittent
        // at ~20% of launches; here it fires on seven of the eight MFMAs
        // because a single bank is recycled for all of them, and the model
        // decodes fluent nonsense.
        //
        // Holding all MFMA_ITERS b values live forces disjoint banks, which
        // is the ping-pong fix taken to its limit: no register is ever both a
        // live MFMA source and an in-flight LDS destination.
        i32x8_t b[MFMA_ITERS];
#pragma unroll
        for (int ki = 0; ki < MFMA_ITERS; ki++) {
          b[ki] = _gang_load_fp8_mfma_b(s_tok_fp8, ki * K_PER_MFMA, g);
        }
        // Nail every load down on this side of the MFMAs.
        //
        // Without this the machine scheduler sinks all but two tiles back
        // below the first MFMA -- its register-pressure heuristic targets an
        // occupancy this kernel does not have (the megakernel runs one wave
        // per SIMD at 248 VGPRs) and it happily trades 16 loads in flight for
        // a smaller live set. The result is `s_waitcnt vmcnt(0)` in front of
        // six of the eight MFMAs, i.e. the same fully-exposed latency the
        // rotating pipeline had, just spelled differently -- and it is also
        // what re-introduces the WAR hazard above.
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (int ki = 0; ki < MFMA_ITERS; ki++) {
          acc = _gang_mfma_f8xf8(a[ki], b[ki], acc, sa[ki],
                                 (int)s_tok_scales[ki]);
        }
        // This is what "prevents ROCm miscompilation" below actually means,
        // and it is worth naming because it is not a codegen bug.
        //
        // `acc` is read only under `if (col == 0)` in the epilogue. Given
        // straight-line MFMAs, LLVM happily sinks the whole chain into that
        // block -- the asm shows `s_and_b64 exec, exec, vcc` landing *above*
        // the first v_mfma. But MFMA gathers its A and B operands across all
        // 64 lanes of the wave, so running it under a 1-in-16 EXEC mask
        // silently computes on 4 lanes' worth of data. The model still decodes;
        // it just emits garbage.
        //
        // The `#pragma unroll 1` loop in the branch below blocks the sink as a
        // side effect (LLVM will not sink a loop into a conditional), which is
        // why nobody had to name the hazard. Straight-line code has no such
        // accident, so pin `acc` here with a volatile asm the sinker cannot
        // move: same "+v" optimization-barrier idiom as
        // gang_mla_full_layer_fused_mi300.cuh:467.
        asm volatile("" : "+v"(acc));
      } else {
        // Pre-fill: load k-tiles 0..3 into pipeline slots. Under HOIST_PREFILL
        // this wave's only fill was already issued above the quantizer, so the
        // slots start from those registers instead; w_row there is
        // warp_id * 16 + col, which is this loop's w_row at tile_iter 0, and
        // TILES_PER_WAVE == 1 is what makes that the only iteration.
        i32x8_t a0, a1, a2, a3;
        int sa0, sa1, sa2, sa3;
        if constexpr (HOIST_PREFILL) {
          a0 = ph_a[0]; sa0 = ph_sa[0];
          a1 = ph_a[1]; sa1 = ph_sa[1];
          a2 = ph_a[2]; sa2 = ph_sa[2];
          a3 = ph_a[3]; sa3 = ph_sa[3];
        } else {
          a0 = _gang_load_fp8_mfma_b_g(w_data_row, 0 * K_PER_MFMA, g);
          sa0 = (int)_gang_ld_g<uint8_t>(wg_scales + row_scale_base + 0 * 4 + g);
          a1 = _gang_load_fp8_mfma_b_g(w_data_row, 1 * K_PER_MFMA, g);
          sa1 = (int)_gang_ld_g<uint8_t>(wg_scales + row_scale_base + 1 * 4 + g);
          a2 = _gang_load_fp8_mfma_b_g(w_data_row, 2 * K_PER_MFMA, g);
          sa2 = (int)_gang_ld_g<uint8_t>(wg_scales + row_scale_base + 2 * 4 + g);
          a3 = _gang_load_fp8_mfma_b_g(w_data_row, 3 * K_PER_MFMA, g);
          sa3 = (int)_gang_ld_g<uint8_t>(wg_scales + row_scale_base + 3 * 4 + g);
        }

        // IMPORTANT: #pragma unroll 1 prevents ROCm miscompilation -- see the
        // EXEC-mask note in the branch above for what that actually is.
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
            a0 = _gang_load_fp8_mfma_b_g(w_data_row, kt4, g);
            sa0 = (int)_gang_ld_g<uint8_t>(wg_scales + row_scale_base + kt4 / 32 + g);
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
            a1 = _gang_load_fp8_mfma_b_g(w_data_row, kt5, g);
            sa1 = (int)_gang_ld_g<uint8_t>(wg_scales + row_scale_base + kt5 / 32 + g);
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
            a2 = _gang_load_fp8_mfma_b_g(w_data_row, kt6, g);
            sa2 = (int)_gang_ld_g<uint8_t>(wg_scales + row_scale_base + kt6 / 32 + g);
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
            a3 = _gang_load_fp8_mfma_b_g(w_data_row, kt7, g);
            sa3 = (int)_gang_ld_g<uint8_t>(wg_scales + row_scale_base + kt7 / 32 + g);
          }
        }
      } // MFMA_ITERS > FULL_PRELOAD_ITERS

      // ── Step 4: Bias epilogue, write BF16 output ─────────────────────────
      if (col == 0) {
        unsigned short packed[4];
        for (int i = 0; i < 4; i++) {
          int out_n = wg_idx * OUTPUT_PER_WG + wave_tile * 16 + g * 4 + i;
          float sum = acc[i];

          // Add bias (partitioned per XCD)
          unsigned bt = (unsigned)d_bias[out_n] << 16;
          float bv;
          __builtin_memcpy(&bv, &bt, 4);

          packed[i] = _gang_float_to_bf16(sum + bv);
        }
        int out_idx = tok_idx * output_stride + wg_idx * OUTPUT_PER_WG +
                      wave_tile * 16 + g * 4;
        _rnlm8_store4<WRITE_THROUGH>(
            d_output + out_idx, packed[0], packed[1], packed[2], packed[3]);
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

    // Pre-fill: load k-tiles 0..3 into pipeline slots. Under HOIST_PREFILL
    // these were issued above the quantizer; the hoist used w_row = col and
    // ki0 = warp_id * ITERS_PER_WAVE, which is exactly (w_row, ki_start) here.
    i32x8_t a0, a1, a2, a3;
    int sa0, sa1, sa2, sa3;
    if constexpr (HOIST_PREFILL) {
      a0 = ph_a[0]; sa0 = ph_sa[0];
      a1 = ph_a[1]; sa1 = ph_sa[1];
      a2 = ph_a[2]; sa2 = ph_sa[2];
      a3 = ph_a[3]; sa3 = ph_sa[3];
    } else {
      a0 = _gang_load_fp8_mfma_b_g(w_data_row, ki_start * K_PER_MFMA, g);
      sa0 = (int)_gang_ld_g<uint8_t>(wg_scales + row_scale_base + ki_start * 4 + g);
      a1 = _gang_load_fp8_mfma_b_g(w_data_row, (ki_start + 1) * K_PER_MFMA, g);
      sa1 = (int)_gang_ld_g<uint8_t>(wg_scales + row_scale_base + (ki_start + 1) * 4 + g);
      a2 = _gang_load_fp8_mfma_b_g(w_data_row, (ki_start + 2) * K_PER_MFMA, g);
      sa2 = (int)_gang_ld_g<uint8_t>(wg_scales + row_scale_base + (ki_start + 2) * 4 + g);
      a3 = _gang_load_fp8_mfma_b_g(w_data_row, (ki_start + 3) * K_PER_MFMA, g);
      sa3 = (int)_gang_ld_g<uint8_t>(wg_scales + row_scale_base + (ki_start + 3) * 4 + g);
    }

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
        a0 = _gang_load_fp8_mfma_b_g(w_data_row, kt4, g);
        sa0 = (int)_gang_ld_g<uint8_t>(wg_scales + row_scale_base + kt4 / 32 + g);
      }

      // Slot 1: compute k-tile ki+1, prefetch ki+5
      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 1) * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki + 1];
        acc = _gang_mfma_f8xf8(a1, b, acc, sa1, sb);
      }
      if (ki + 5 < ki_end) {
        int kt5 = (ki + 5) * K_PER_MFMA;
        a1 = _gang_load_fp8_mfma_b_g(w_data_row, kt5, g);
        sa1 = (int)_gang_ld_g<uint8_t>(wg_scales + row_scale_base + kt5 / 32 + g);
      }

      // Slot 2: compute k-tile ki+2, prefetch ki+6
      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 2) * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki + 2];
        acc = _gang_mfma_f8xf8(a2, b, acc, sa2, sb);
      }
      if (ki + 6 < ki_end) {
        int kt6 = (ki + 6) * K_PER_MFMA;
        a2 = _gang_load_fp8_mfma_b_g(w_data_row, kt6, g);
        sa2 = (int)_gang_ld_g<uint8_t>(wg_scales + row_scale_base + kt6 / 32 + g);
      }

      // Slot 3: compute k-tile ki+3, prefetch ki+7
      if (ki + 3 < ki_end) {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 3) * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki + 3];
        acc = _gang_mfma_f8xf8(a3, b, acc, sa3, sb);
      }
      if (ki + 7 < ki_end) {
        int kt7 = (ki + 7) * K_PER_MFMA;
        a3 = _gang_load_fp8_mfma_b_g(w_data_row, kt7, g);
        sa3 = (int)_gang_ld_g<uint8_t>(wg_scales + row_scale_base + kt7 / 32 + g);
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
      unsigned short packed[4];
      for (int i = 0; i < 4; i++) {
        float v = 0.0f;
        for (int w = 0; w < NUM_WAVES; w++) {
          v += lds_reduce[w * OUTPUT_PER_WG + g * 4 + i];
        }

        int out_n = wg_idx * OUTPUT_PER_WG + g * 4 + i;

        unsigned bt = (unsigned)d_bias[out_n] << 16;
        float bv;
        __builtin_memcpy(&bv, &bt, 4);

        packed[i] = _gang_float_to_bf16(v + bv);
      }
      int out_idx = tok_idx * output_stride + wg_idx * OUTPUT_PER_WG + g * 4;
      _rnlm8_store4<WRITE_THROUGH>(
          d_output + out_idx, packed[0], packed[1], packed[2], packed[3]);
    }
  }

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  if (_sp_rec) {
    _sp_t3 = __builtin_amdgcn_s_memrealtime();
    // Slot 0, qkv_a only. [1] = residual resolve + RMSNorm rcp + the depth-4
    // prefetch fill, [2] = the FP8 quantizer, [3] = the MFMA K-loop and the
    // epilogue store. [1][5] is this call site's own tile count -- do NOT
    // divide these by g_subphase_cnt[0], which the [0][0] whole-tile timer
    // and four other banks also feed.
    atomicAdd(&g_subphase_ns[0][1], (_sp_t1 - _sp_t0) * 10);
    atomicAdd(&g_subphase_ns[0][2], (_sp_t2 - _sp_t1) * 10);
    atomicAdd(&g_subphase_ns[0][3], (_sp_t3 - _sp_t2) * 10);
    atomicAdd(&g_subphase_ns[1][5], 1ULL);
  }
#endif
  __syncthreads();
}

} // namespace kernel
