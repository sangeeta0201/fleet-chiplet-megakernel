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
#include "tasks/mi300/mpk_bsdbg.cuh"

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

  for (int sb = tid; sb < NSUBBLOCKS; sb += MPK_NT) {
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
  int const num_waves = MPK_NT >> 6;
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
// addrspace(1) load helpers for the resolve batch. They go through a scalar
// uint64_t rather than casting the vector type directly: dereferencing an
// `address_space(1)` pointer to a class type (uint2/float4 are classes in HIP)
// yields an address-space-qualified prvalue, and the implicit copy-assignment
// operator does not accept one -- "no viable overloaded '='". uint64_t has no
// user-declared operator=, so the same cast on it compiles, which is exactly
// why the STAGE_NW load below could already do this and the others could not.
// One global_load_dwordx2 per call; the float4 form is two of them and is only
// reached off the EP path.
__device__ __forceinline__ uint2 _rnlm8_ld_g_u2(void const *p) {
  uint64_t const v =
      *(__attribute__((address_space(1))) uint64_t const *)(uint64_t const *)p;
  uint2 r;
  r.x = (unsigned)v;
  r.y = (unsigned)(v >> 32);
  return r;
}
__device__ __forceinline__ float4 _rnlm8_ld_g_f4(void const *p) {
  auto const *q =
      (__attribute__((address_space(1))) uint64_t const *)(uint64_t const *)p;
  uint64_t const a = q[0];
  uint64_t const b = q[1];
  float4 r;
  r.x = __uint_as_float((unsigned)a);
  r.y = __uint_as_float((unsigned)(a >> 32));
  r.z = __uint_as_float((unsigned)b);
  r.w = __uint_as_float((unsigned)(b >> 32));
  return r;
}

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

  // MPK_RESADD_GLOBAL: address the batched loads through addrspace(1).
  //
  // The batch above works -- the built image issues all ITERS x (1 + NEXTRA)
  // + ITERS loads back to back and drains them at ONE `s_waitcnt vmcnt(0)`,
  // so `carry` is 30 at EP_PEER_SLOTS=4, not the ~1.5 the note above inferred
  // from bandwidth. But 24 of those 30 are `flat_load_dwordx2` and only the 6
  // norm-weight loads are `global_load_dwordx2`, because `bat_nw` is the only
  // one of the four that carries the addrspace(1) cast. Two costs follow:
  //
  //   1. flat has no SGPR-base form on gfx9, so every one of the 24 needs a
  //      64-bit `v_add_co_u32`/`v_addc_co_u32` pair to materialize its
  //      address -- ~90 VALU ops of pure addressing in the region.
  //   2. flat raises lgkmcnt as well as vmcnt, which is why the drain reads
  //      `s_waitcnt vmcnt(0) lgkmcnt(0)`: the batch is fenced against LDS
  //      traffic it has no dependence on.
  //
  // Under addrspace(1) the peer slots become `global_load_dwordx2 v, voff,
  // s[base] offset:imm` -- one shared 32-bit voffset, an SGPR base per slot
  // and immediate trip offsets. This is the same edit 3c1fa83 made in the MLA
  // decode kernel, and the opposite of the MoE result
  // (glm-moe-addrspace1-is-a-negative...), which regressed because it changed
  // the s_waitcnt COUNT in a k-loop. It cannot do that here: the region has
  // exactly one drain before and after.
#ifndef MPK_RESADD_GLOBAL
#define MPK_RESADD_GLOBAL 1
#endif
#if MPK_RESADD_BATCH
  uint2 bat_r[ITERS];
  uint2 bat_p[ITERS][NEXTRA];
  float4 bat_w[(!EP && !PRE_FOLDED) ? ITERS : 1];
  uint64_t bat_nw[STAGE_NW ? ITERS : 1];
#if MPK_RESADD_GLOBAL
#define MPK_RA_LD_U2(p) _rnlm8_ld_g_u2((void const *)(p))
#define MPK_RA_LD_F4(p) _rnlm8_ld_g_f4((void const *)(p))
#else
#define MPK_RA_LD_U2(p) (*reinterpret_cast<uint2 const *>(p))
#define MPK_RA_LD_F4(p) (*reinterpret_cast<float4 const *>(p))
#endif
#pragma unroll
  for (int v = 0; v < ABL_ITERS; v++) {
    int const off = (v * NTHREADS + tid) * VEC;
    bat_r[v] = MPK_RA_LD_U2(d_res + off);
    if constexpr (EP) {
#pragma unroll
      for (int p = 0; p < NEXTRA; p++) {
        bat_p[v][p] =
            MPK_RA_LD_U2(d_res + (size_t)(p + 1) * EP_SLOT_ELEMS + off);
      }
    } else if constexpr (!PRE_FOLDED) {
      bat_w[v] = MPK_RA_LD_F4(d_ws + off);
    }
    if constexpr (STAGE_NW) {
      uint64_t const *nwp = reinterpret_cast<uint64_t const *>(d_nw + off);
      bat_nw[v] = *(__attribute__((address_space(1))) uint64_t const *)nwp;
    }
  }
#undef MPK_RA_LD_U2
#undef MPK_RA_LD_F4
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
  int const num_waves = MPK_NT >> 6;
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
  int const nthreads = MPK_NT;
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
  int const num_waves = MPK_NT >> 6;
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

// ── K-major weight layout for the attention-half GEMM ──────────────────────
//
// The MoE twin of this (MPK_MOE_KMAJOR, gang_moe_linear_mxfp8_mi300.cuh)
// measured -0.278 ms at NP=4 by turning sixteen scattered L2 requests per
// instruction into eight fully-used ones. The same pathology is here, in the
// same shape:
//
//   w_data_row = wg_data + w_row * REDUCTION_SIZE,   w_row = wave_tile*16 + col
//   _gang_load_fp8_mfma_b_g reads  data + kt + g*16  and  + 64
//
// A wave's 64 lanes are (col, g) with col in [0,16) and g in [0,4), so one
// global_load_dwordx4 covers sixteen rows x sixty-four bytes at REDUCTION_SIZE
// stride: sixteen distinct 128-byte requests, each half-used. The `hi` load at
// +64 is another sixteen.
//
// K-major stores the workgroup's data half as [row/16][k][row%16][64B] --
// exactly the order the wave reads it -- so the same instruction reads
// 16*64 = 1024 CONTIGUOUS bytes: eight fully-used requests.
//
// The granule is 64 bytes, not the 128 an FP8 k-tile occupies, and that is the
// one place this differs from the MoE version. The MoE ships MXFP4, whose
// k-tile IS 64 bytes per row, so grouping at the k-tile gave contiguity for
// free. At FP8 the k-tile is 128 bytes and one instruction only reaches half
// of it, so grouping at 128 would leave the wave reading sixteen 64-byte
// pieces 128 bytes apart -- the same request count as row-major, just with
// better locality. Splitting each k-tile into its lo and hi half and grouping
// sixteen rows within EACH half is what actually collapses the count.
//
// Both layouts then reduce to `base + kt*KMUL + g*16` for the lo half and
// `+ HI_OFF` for the hi, so the k-loops carry two compile-time constants
// instead of a layout branch:
//
//              base                                   KMUL  HI_OFF
//   row-major  wg_data + w_row*RS                       1      64
//   K-major    wg_data + (w_row/16)*RS*16               16   1024
//                      + (w_row%16)*64
//
// Level 2 does the same to the E8M0 scales: four bytes per row per k-tile, so
// a wave reads sixteen 4-byte pieces NUM_BLOCKS_32 apart today and one
// contiguous 64-byte run under [row/16][k][row%16][4B].
//
// Scope: this applies ONLY to the OUTPUT_PER_WG == 16 (K-parallel) branch,
// which is qkv_a and the unabsorbed q_b and nothing else. The N-parallel
// branch serves the LM head and the dense MLP out of the same packer, and
// those call sites are not repacked, so they must stay row-major. demo.py's
// pack_dense_mxfp8 enforces the identical rule -- a mismatch here is silent
// garbage, not a build error.
#ifndef MPK_DENSE_KMAJOR
// Measured at NP=4 bs=1 on devices 4-7, paired, one variable per step.
//
//   level          clean avg          per-iter min (the stall-immune stat)
//   0 row-major    10.681 (n=3)       10.531 (n=5, 10.509-10.551)
//   1 data K-maj   10.582 (n=5)       10.428 (n=5, 10.398-10.447)   -0.103
//   2 + scales     10.504 (n=6)       10.343 (n=6, 10.262-10.409)   -0.086
//
// 0->1 and 1->2 each have both statistics agreeing, and at each step five or
// six of the arm's runs sit below the control's BEST run. Level 2 is the
// cheaper of the two to believe: the scale half is only ~3% of the bytes but
// a whole 16-request instruction of its own, and request count -- not bytes
// -- is what this class pays for. Together they are -0.19 ms.
//
// Level 2 is NOT free once OUTPUT_PER_WG != 16: _rnlm8_sck gates it the same
// way _rnlm8_wk gates the data half, so the N-parallel call sites stay
// row-major and pack_dense_mxfp8 must agree. A mismatch is silent garbage.
#define MPK_DENSE_KMAJOR 2
#endif

// Whether THIS instantiation reads a K-major weight. See the scope note.
template <int OUTPUT_PER_WG>
__device__ __forceinline__ constexpr bool _rnlm8_wk() {
  return MPK_DENSE_KMAJOR >= 1 && OUTPUT_PER_WG == 16;
}
template <int OUTPUT_PER_WG>
__device__ __forceinline__ constexpr bool _rnlm8_sck() {
  return MPK_DENSE_KMAJOR >= 2 && OUTPUT_PER_WG == 16;
}

// The A-operand load, addressed by the k-tile BYTE offset the row-major
// callers already pass (kt = ki * 128) and scaled here by the layout's stride
// multiplier. KMUL/HI_OFF are template parameters rather than reads of the
// macro so that a single translation unit could hold both layouts.
template <int KMUL, int HI_OFF>
__device__ __forceinline__ i32x8_t
    _rnlm8_load_w(uint8_t const *base, int kt, int g) {
  uint8_t const *p = base + kt * KMUL + g * 16;
  i32x4_t lo = _gang_ld_g<i32x4_t>(p);
  i32x4_t hi = _gang_ld_g<i32x4_t>(p + HI_OFF);
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

// The matching scale byte. `off4` is the row-major offset the callers already
// compute -- ki*4, or equivalently kt/32 -- and is scaled the same way.
//
// MEASURED NO-GO (2026-08-25): folding the GROUPS-wide batch of these onto ONE
// base pointer with compile-time immediate offsets -- the change that bought
// -0.123 ms on the MoE twin as MPK_MOE_SCBASE -- is a small LOSS here.
//
//   arm (folded)  n=5 mean 9.766  (9.797 9.784 9.750 9.729 9.771)
//   control       n=5 mean 9.718  (9.735 9.704 9.733 9.695 9.725)
//
// and that is despite worker_kernel .vgpr_count 333 -> 327 with zero spills,
// and despite the fold landing (the qkv_a OPW=16/K=6144 body goes from 0 of 24
// `global_load_ubyte` in immediate-offset form to 12 of 24).
//
// The issue-order hypothesis -- that batching four one-byte loads ahead of the
// weight tiles shortens the window the weight loads have to themselves, vmcnt
// being in-order -- is REFUTED by the image: an interleaved variant that keeps
// the fold but restores `W W SC` order emits a byte-identical hot-body load
// sequence, because the scheduler was already interleaving them. It only costs
// 4 VGPRs back (331) for the base pointer it keeps live.
//
// So the dense k-loop is not address-register-bound the way the MoE one was.
// Do not re-try this without a new mechanism; the arm has been built and run.
template <int KMUL>
__device__ __forceinline__ int
    _rnlm8_load_sc(uint8_t const *base, int off4, int g) {
  return (int)_gang_ld_g<uint8_t>(base + off4 * KMUL + g);
}

// ── The GROUPS-deep, guard-free k-loop ─────────────────────────────────────
//
// Both branches of the GEMM below shipped the same rotating depth-4 pipeline:
// four slot registers a0..a3, each reloaded right after the MFMA that consumes
// it, under `#pragma unroll 1` with an `if (ki + N < end)` tail guard on every
// refill. It reads as depth 4. Disassembling the shipping image
// (llvm-objdump of permanent_output_dir_rank0's gfx950 bundle, the
// REDUCTION_SIZE=6144 / OUTPUT_PER_WG=16 instantiation, i.e. qkv_a) says it is
// not:
//
//   0002B6FC  s_or_b64 exec, exec, s[8:9]      ; <- loop back-edge target
//   0002B714  s_waitcnt vmcnt(0)               ; <- FULL DRAIN, every trip
//   0002B718  v_mov_b32_e32 v178, v167         ; 28 more of these: the slot
//   ...                                        ;  rotation the unroll-1 forbids
//   0002B7A8  ds_read_b128 ...                 ; body proper starts here
//
// Not one `s_waitcnt vmcnt` appears anywhere inside the body after that. The
// entire prefetch is cashed in at the top of each trip, so the loop runs one
// k-group of latency exposed per four MFMAs -- which is what
// memory:glm-attention-tiles-are-latency-bound-not-valu-bound measured as
// qkv_a at 68% vmcnt.
//
// Two things cause it, and they are the same two the MoE k-loop had
// (gang_moe_linear_mxfp8_mi300.cuh, MPK_MOE_PF_GROUPS):
//
//   1. The tail guards put each refill in its own basic block, so the number
//      of outstanding vm ops at the join differs per path and SIInsertWaitcnts
//      gives up and emits vmcnt(0). A prefetch that the wait drains is not a
//      prefetch.
//   2. `#pragma unroll 1` (mandatory -- it is what stops LLVM sinking the MFMA
//      chain under the `col == 0` EXEC mask) denies the compiler the unroll it
//      would need to rotate registers by renaming, so it rotates them with 28
//      v_mov instead, and those movs need the vmcnt(0) to be legal.
//
// The fix is the MoE one: make the steady-state body straight-line by fixing
// the trip count at compile time and peeling the last block. GROUPS loads
// issue for block b+1 before any MFMA of block b runs, the copies become a
// compile-time array shuffle the register allocator resolves for free, and the
// wait in front of the MFMAs can be a partial vmcnt.
//
// KI_LEN is the number of k-tiles THIS WAVE walks -- MFMA_ITERS on the
// N-parallel branch, ITERS_PER_WAVE on the K-parallel one -- and ki_start is
// where it starts. GROUPS must divide KI_LEN; the caller's `_rnlm8_pf_groups`
// picks the largest divisor <= the requested depth and returns 0 when there is
// none, which selects the rotating loop instead.
//
// SEEDED takes the four A tiles the prologue already hoisted above the
// quantizer (HOIST_PREFILL). That hoist issues exactly 4 tiles, so it only
// composes with GROUPS == 4.
//
// W_KMUL/W_HIOFF/SC_KMUL carry the weight layout (see MPK_DENSE_KMAJOR). The
// loop is layout-blind: both layouts are `base + kt*KMUL + g*16`, so the only
// thing that changes is the constant, and `w_data_row`/`w_scale_row` are the
// caller's already-offset bases for this wave's sixteen rows.
template <int GROUPS,
          int KI_LEN,
          bool SEEDED,
          int W_KMUL,
          int W_HIOFF,
          int SC_KMUL>
__device__ __forceinline__ f32x4_t
    _rnlm8_kloop_deep(uint8_t const *w_data_row,
                      uint8_t const *w_scale_row,
                      uint8_t const *s_tok_fp8,
                      uint8_t const *s_tok_scales,
                      int ki_start,
                      int g,
                      i32x8_t const *seed_a,
                      int const *seed_sa) {
  constexpr int K_PER_MFMA = 128;
  static_assert(KI_LEN % GROUPS == 0,
                "the deep k-loop needs GROUPS to divide the trip count");
  static_assert(!SEEDED || GROUPS == 4,
                "the hoisted prefill issues exactly four A tiles");
  constexpr int NBLK = KI_LEN / GROUPS;

  f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};
  i32x8_t A[GROUPS];
  int S[GROUPS];
#pragma unroll
  for (int j = 0; j < GROUPS; j++) {
    if constexpr (SEEDED) {
      A[j] = seed_a[j];
      S[j] = seed_sa[j];
    } else {
      int kk = ki_start + j;
      A[j] = _rnlm8_load_w<W_KMUL, W_HIOFF>(w_data_row, kk * K_PER_MFMA, g);
      S[j] = _rnlm8_load_sc<SC_KMUL>(w_scale_row, kk * 4, g);
    }
  }

// IMPORTANT: #pragma unroll 1 prevents ROCm miscompilation -- it is what keeps
// LLVM from sinking the MFMA chain into the `col == 0` epilogue block, where it
// would run under a 1-in-16 EXEC mask and silently compute on a quarter of the
// wave. The GROUPS-wide bodies inside are fully unrolled, so the steady state
// is still straight-line code.
#pragma unroll 1
  for (int blk = 0; blk < NBLK - 1; blk++) {
    int const base = ki_start + blk * GROUPS;
    i32x8_t N[GROUPS];
    int NS[GROUPS];
    // Issue the whole NEXT block before consuming the current one.
#pragma unroll
    for (int j = 0; j < GROUPS; j++) {
      int kk = base + GROUPS + j;
      N[j] = _rnlm8_load_w<W_KMUL, W_HIOFF>(w_data_row, kk * K_PER_MFMA, g);
      NS[j] = _rnlm8_load_sc<SC_KMUL>(w_scale_row, kk * 4, g);
    }
#pragma unroll
    for (int j = 0; j < GROUPS; j++) {
      i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (base + j) * K_PER_MFMA, g);
      acc = _gang_mfma_f8xf8(A[j], b, acc, S[j], (int)s_tok_scales[base + j]);
    }
#pragma unroll
    for (int j = 0; j < GROUPS; j++) {
      A[j] = N[j];
      S[j] = NS[j];
    }
  }

  // Peeled last block: consume what the previous trip prefetched, issue
  // nothing past the end of this wave's range.
  int const tail = ki_start + (NBLK - 1) * GROUPS;
#pragma unroll
  for (int j = 0; j < GROUPS; j++) {
    i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (tail + j) * K_PER_MFMA, g);
    acc = _gang_mfma_f8xf8(A[j], b, acc, S[j], (int)s_tok_scales[tail + j]);
  }
  // Pin the accumulator. At NBLK == 1 -- ITERS_PER_WAVE == GROUPS, which is
  // the K=2048 / OUTPUT_PER_WG=16 instantiation -- the block loop above has a
  // zero trip count and vanishes, leaving a straight-line MFMA chain. The
  // caller reads `acc` only under `col == 0` (or `col < rows_here`), and given
  // straight-line code LLVM sinks the whole chain into that block: the asm
  // shows `s_and_b64 exec, exec, vcc` landing above the first v_mfma. MFMA
  // gathers its operands across all 64 lanes, so under a 1-in-16 EXEC mask it
  // silently computes on a quarter of the data. Same hazard, same fix, as the
  // FULL_PRELOAD_ITERS branch in the caller.
  asm volatile("" : "+v"(acc));
  return acc;
}

// ── DOUBLE-BUFFERED FORM OF THE LOOP ABOVE ─────────────────────────────────
//
// Same defect, same fix, as the MoE twin (gang_moe_linear_mxfp8_mi300.cuh,
// _gang_moe_kloop_dbuf, which carries the full ISA dump). `A[j] = N[j]` at the
// backedge is a USE of every load destination, so SIInsertWaitcnts puts an
// s_waitcnt vmcnt(0) in front of it -- a full drain -- and the outstanding
// load count returns to zero every k-block. That is the unroll=1 point on the
// curve tests/standalone/test_waves_per_simd_payoff.hip measured on this part
// (3446 GB/s at 1 in flight, 5334 at 4), and it is exactly what the deep loop
// was written to avoid. Swapping two buffers instead of copying one into the
// other keeps every value in the register its load wrote, so the wait in front
// of the MFMAs can name the just-issued loads and let them ride the branch.
//
// SEEDED is supported: the hoisted prefill fills the A buffer, and the first
// thing the loop does is issue into N, which is what the unseeded path does
// too.
//
// ANY NBLK >= 1, including odd. The earlier form required an even NBLK >= 4
// and so excluded the one site that needs it most: the K-parallel qkv_a branch
// walks ITERS_PER_WAVE = 48/4 = 12 with GROUPS=4, i.e. NBLK=3, and fell back to
// the copying loop. Counted on the built image, that loop
// ([0x2f600..0x2f834], the hottest tile in the model) carries ~40
// `v_mov_b32_e32` rotation copies against four `v_mfma_scale_f32_16x16x128_
// f8f6f4` per trip -- which is the "qkv_a is 74% VALU *with* the MFMA" line in
// the profile.
//
// The odd case needs no peel and no extra basic block. Both parities share one
// steady loop `for (blk = 0; blk + 2 < NBLK; blk += 2)`; on exit A holds block
// `blk`, which is NBLK-1 when NBLK is odd and NBLK-2 when it is even. So the
// epilogue differs only by an `if constexpr` on the parity -- a compile-time
// branch, not a runtime one, which is what the MoE header's warning about a
// prefetch in its own basic block was about.
//
//   NBLK  loop trips  A holds on exit  epilogue
//   1     0           0 = NBLK-1       consume A
//   2     0           0 = NBLK-2       issue N(1); consume A(0); consume N(1)
//   3     1           2 = NBLK-1       consume A
//   4     1           2 = NBLK-2       issue N(3); consume A(2); consume N(3)
//   5     2           4 = NBLK-1       consume A
//
// For even NBLK this is instruction-identical to the previous form (its bound
// `blk <= NBLK-4` and this one's `blk <= NBLK-3` select the same even trips).
template <int GROUPS,
          int KI_LEN,
          bool SEEDED,
          int W_KMUL,
          int W_HIOFF,
          int SC_KMUL>
__device__ __forceinline__ f32x4_t
    _rnlm8_kloop_dbuf(uint8_t const *w_data_row,
                      uint8_t const *w_scale_row,
                      uint8_t const *s_tok_fp8,
                      uint8_t const *s_tok_scales,
                      int ki_start,
                      int g,
                      i32x8_t const *seed_a,
                      int const *seed_sa) {
  constexpr int K_PER_MFMA = 128;
  static_assert(KI_LEN % GROUPS == 0,
                "the deep k-loop needs GROUPS to divide the trip count");
  static_assert(!SEEDED || GROUPS == 4,
                "the hoisted prefill issues exactly four A tiles");
  constexpr int NBLK = KI_LEN / GROUPS;
  static_assert(NBLK >= 1, "dbuf needs at least one k-block");

  f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};
  i32x8_t A[GROUPS], N[GROUPS];
  int S[GROUPS], NS[GROUPS];

  // By reference: passing the arrays by value would reintroduce the copy this
  // variant exists to delete.
  auto issue = [&](i32x8_t (&dst)[GROUPS], int (&dsc)[GROUPS], int b) {
#pragma unroll
    for (int j = 0; j < GROUPS; j++) {
      int const kk = ki_start + b * GROUPS + j;
      dst[j] = _rnlm8_load_w<W_KMUL, W_HIOFF>(w_data_row, kk * K_PER_MFMA, g);
      dsc[j] = _rnlm8_load_sc<SC_KMUL>(w_scale_row, kk * 4, g);
    }
  };
  auto consume = [&](i32x8_t const (&src)[GROUPS], int const (&ssc)[GROUPS],
                     int b) {
#pragma unroll
    for (int j = 0; j < GROUPS; j++) {
      int const kk = ki_start + b * GROUPS + j;
      i32x8_t bb = _gang_load_fp8_mfma_b(s_tok_fp8, kk * K_PER_MFMA, g);
      acc = _gang_mfma_f8xf8(src[j], bb, acc, ssc[j], (int)s_tok_scales[kk]);
    }
  };

  if constexpr (SEEDED) {
#pragma unroll
    for (int j = 0; j < GROUPS; j++) {
      A[j] = seed_a[j];
      S[j] = seed_sa[j];
    }
  } else {
    issue(A, S, 0);
  }

// IMPORTANT: #pragma unroll 1 for the same reason as the loop above -- it is
// what keeps LLVM from sinking the MFMA chain into the `col == 0` epilogue
// block, where it would run under a 1-in-16 EXEC mask.
#pragma unroll 1
  for (int blk = 0; blk + 2 < NBLK; blk += 2) {
    issue(N, NS, blk + 1);
    consume(A, S, blk);
    issue(A, S, blk + 2);
    consume(N, NS, blk + 1);
  }

  // On exit A holds block NBLK-1 (odd NBLK) or NBLK-2 (even NBLK).
  if constexpr (NBLK % 2 == 0) {
    issue(N, NS, NBLK - 1);
    consume(A, S, NBLK - 2);
    consume(N, NS, NBLK - 1);
  } else {
    consume(A, S, NBLK - 1);
  }
  // Pin the accumulator: same EXEC-mask sinking hazard as the loop above.
  asm volatile("" : "+v"(acc));
  return acc;
}

// MPK_ATTN_PF_DBUF: which sites take the double-buffered form.
//
//   0  copying (_rnlm8_kloop_deep) everywhere
//   1  double-buffered everywhere
//   2  double-buffered only where NBLK is ODD -- i.e. exactly the sites the
//      previous even-NBLK-only guard excluded. At GLM-5's shapes that is the
//      K-parallel qkv_a branch (ITERS_PER_WAVE=12, GROUPS=4, NBLK=3) and
//      nothing else.
//
// =1 WAS MEASURED NEGATIVE, but on the even sites only, and the reason was
// register pressure, not the copies: the MoE twin at NP=4 was 0.226 ms slower
// (10.492 vs 10.266 per-iter min, n=5 each, disjoint) and raised
// worker_kernel's .vgpr_count from 284 to 310 with zero spills, an allocation
// shared by every tile kernel in the binary.
//
// =2 is the arm that verdict does NOT cover, and it is what ships. It touches
// one loop, the hottest one, and that loop already has both A[] and N[] live
// across the copy block -- `A[j] = N[j]` is a use of every load destination --
// so deleting the copies does not widen the live range the way adding a whole
// extra prefetch block to the even sites did.
//
// MEASURED, GLM-5, NP=4, devices 4-7, bs=1, 2026-08-25, same session:
//
//   =0  9.995 9.941 9.963 9.996                    n=4  mean 9.974  sd 0.023
//   =2  9.955 9.925 9.964 9.955 9.950 9.972 9.924  n=7  mean 9.949  sd 0.017
//
//   delta -0.024 ms (median -0.024, same sign).  Welch t = 1.86, and the
//   ranges OVERLAP (=0 min 9.941 < =2 max 9.972).  So this is INSIDE the
//   0.26 ms wall noise floor and is NOT a resolved win -- it is shipped on the
//   ISA argument with a consistent sign, not on the wall.
//
// What changed in the image (`llvm-objdump -d --mcpu=gfx950`):
//   - gang_rmsnorm_linear_mxfp8_bias_kernel's K-parallel k-loop stops being a
//     loop at all. At NBLK=3 the single steady trip folds flat, so the ~40
//     `v_mov_b32_e32` rotation copies and the backedge both disappear and the
//     12 MFMAs run straight-line. isa_loads_in_flight.py no longer reports a
//     hot loop in that kernel at all (it was `97 insn / 12 loads / issued 0`).
//   - worker_kernel .vgpr_count 325 -> 333, .agpr_count 69 -> 77, spills 0
//     both sides. Both allocations are already far past the 256 that a second
//     wave/SIMD would need, and the occupancy gate here is LDS (155/160 KB/CU,
//     memory: glm-occupancy-lock-is-lds-not-registers), so the +8/+8 costs no
//     occupancy. Re-check that if the LDS lock ever moves.
//   - Static image grows (110604 -> 114645 disassembly lines) because the
//     unrolled body is duplicated. Dynamic instruction count per k-loop falls.
#ifndef MPK_ATTN_PF_DBUF
#define MPK_ATTN_PF_DBUF 2
#endif

// Depth dispatch for the two forms above.
template <int GROUPS,
          int KI_LEN,
          bool SEEDED,
          int W_KMUL,
          int W_HIOFF,
          int SC_KMUL>
__device__ __forceinline__ f32x4_t
    _rnlm8_kloop_pick(uint8_t const *w_data_row,
                      uint8_t const *w_scale_row,
                      uint8_t const *s_tok_fp8,
                      uint8_t const *s_tok_scales,
                      int ki_start,
                      int g,
                      i32x8_t const *seed_a,
                      int const *seed_sa) {
  constexpr int NB = KI_LEN / GROUPS;
  constexpr bool TAKE_DBUF =
      (MPK_ATTN_PF_DBUF == 1) || (MPK_ATTN_PF_DBUF == 2 && NB % 2 == 1);
  if constexpr (TAKE_DBUF) {
    return _rnlm8_kloop_dbuf<GROUPS, KI_LEN, SEEDED, W_KMUL, W_HIOFF, SC_KMUL>(
        w_data_row, w_scale_row, s_tok_fp8, s_tok_scales, ki_start, g, seed_a,
        seed_sa);
  } else {
    return _rnlm8_kloop_deep<GROUPS, KI_LEN, SEEDED, W_KMUL, W_HIOFF, SC_KMUL>(
        w_data_row, w_scale_row, s_tok_fp8, s_tok_scales, ki_start, g, seed_a,
        seed_sa);
  }
}

// Largest divisor of `ki` that is <= `req` and >= 2, or 0 if there is none.
// GLM-5's attention stages give MFMA_ITERS of 48 (qkv_a), 16 (q_b) and 4
// (W_UK/W_UV); the K-parallel branch divides those by NUM_WAVES=4 first. Every
// one of those is a power of two times 3, so a requested depth of 4 is taken
// exactly wherever there are at least 4 tiles to walk.
__device__ __host__ constexpr int _rnlm8_pf_groups(int ki, int req) {
  for (int gr = (req < ki ? req : ki); gr >= 2; gr--) {
    if (ki % gr == 0) {
      return gr;
    }
  }
  return 0;
}

// Prefetch depth for the attention-half GEMM's k-loop. 0 restores the rotating
// depth-4 loop above, which is what every number before this knob was measured
// against.
//
// SWEPT BOTH WAYS AT NP=4, 2026-08-25. Four was inherited from the MoE twin's
// MPK_MOE_PF_GROUPS when this loop was written and had no measurement of its
// own, which is the constants-set-elsewhere bug class that has paid three
// times on this branch (o_proj -0.148, workers 232->240 -0.108, both K-major
// scale levels). It is not one of them: 4 is a real optimum.
//
// _rnlm8_pf_groups takes the largest divisor of the trip count that is <= the
// request, and qkv_a's K-parallel branch walks ITERS_PER_WAVE = 48/4 = 12, so
// the request maps 2->2, 4->4, 8->6.
//
//   request  depth   per-iter min                  clean avg
//   2        2       10.523  n=2, 10.517..10.529   10.673  n=2   +0.243
//   4        4       10.280  n=5, 10.248..10.335   10.431  n=4   (shipping)
//   8        6       10.411  n=4, 10.383..10.436   10.543  n=2   +0.131
//
// Concave, with neither neighbour's range touching 4's, and both statistics
// agreeing at both ends. Shallower exposes load latency in a loop already
// measured at 68% vmcnt; deeper costs 2 * GROUPS * 8 VGPRs for A[] and N[]
// together, and at depth 6 that is 96 registers of live prefetch against 64,
// which buys back less than the occupancy it spends. AGPRs being spill slack
// (memory: glm-agpr-is-spill-slack-lds-half-is-free) makes depth 6 legal, not
// free.
//
// Depth 12 (request 16) is untested and not worth testing: it puts 192 VGPRs
// of A[]+N[] live, and the trend from 6 already points the wrong way.
#ifndef MPK_ATTN_PF_GROUPS
#define MPK_ATTN_PF_GROUPS 4
#endif

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
          bool PRO_PUB = false,
          // Fold the batch rows into the MFMA's N dimension instead of into
          // the tile index. The 16x16x128 scaled MFMA computes 16 output
          // columns and at BATCH_SIZE 1 fifteen of them are wasted: the B
          // operand gather (_gang_load_fp8_mfma_b) addresses by k-block only,
          // so every lane feeds the same token and the epilogue reads acc
          // under `col == 0`. With FOLD_ROWS the caller passes
          // n_wgs_per_xcd tiles instead of BATCH_SIZE * n_wgs_per_xcd, each
          // tile quantizes every row into its own LDS plane, lane column
          // `col` feeds row `col`, and the epilogue writes columns
          // 0..batch_count-1. The weight slab is then fetched once per tile
          // instead of once per (tile, row) -- which is what the bs=2 cost
          // decomposition priced at +16 us/layer of pure redundancy across
          // qkv_a, q_b and o_proj.
          bool FOLD_ROWS = false,
          // Row stride of norm_input_ptr when it is WIDER than the reduction
          // this GEMM takes. Defaulting it to REDUCTION_SIZE is what pinned
          // every narrowed-reduction caller to BATCH_SIZE 1: the token row
          // offset below is the only place a stride is used at all, so at one
          // row the two could not be told apart. q_b is the caller that needs
          // it -- it reduces over the q_a prefix of a wider [q_a | latent]
          // row -- and it passes KV_INPUT_STRIDE, which is that row's width.
          int INPUT_ROW_STRIDE = REDUCTION_SIZE>
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
  // ── bs=2 bisection probe ────────────────────────────────────────────────
  // GLM_FUSE_ATTN defaults to 0, so the three dense prologue layers run this
  // UNFUSED chain and the whole-layer task covers every MoE layer -- which is
  // why the first attempt, a probe inside gang_mla_attn_fused_kernel_mi300,
  // saw fused layers 0-2 and not a single dense one. The dense layers are
  // scheduled first, so seq 0..23 (8 XCD tasks x 3 layers) is exactly the
  // prologue no matter who else calls this kernel later.
  // Read at task entry, i.e. behind the previous task's event boundary, and
  // only on the reduction-dim input, which every XCD sees whole -- an output
  // or residual pointer is column-sliced 8 ways and reading a full row off one
  // slice runs into the next row.
  if (tile_idx == 0 && threadIdx.x == 0) {
    MPK_BSDBG_SEQ(23, norm_input_ptr, REDUCTION_SIZE, "d_x_in",
                  BATCH_SIZE, INPUT_ROW_STRIDE, 24);
  }

  static_assert(REDUCTION_SIZE % 128 == 0,
                "REDUCTION_SIZE must be multiple of 128 for FP8 MFMA");

  // ── Weight layout constants ─────────────────────────────────────────────
  constexpr int NUM_BLOCKS_32 = REDUCTION_SIZE / 32;
  constexpr int WG_DATA_BYTES = OUTPUT_PER_WG * REDUCTION_SIZE;
  constexpr int WG_SCALE_BYTES = OUTPUT_PER_WG * NUM_BLOCKS_32;
  constexpr int WG_BYTES = WG_DATA_BYTES + WG_SCALE_BYTES;

  // The layout of the data and scale halves inside the workgroup. Row-major
  // and K-major differ only in these three constants and in the per-wave base
  // each branch computes -- WG_BYTES and the tile map are identical, because
  // K-major is a permutation of a workgroup's bytes and not a resize. See
  // MPK_DENSE_KMAJOR.
  constexpr bool W_KMAJOR = _rnlm8_wk<OUTPUT_PER_WG>();
  constexpr bool SC_KMAJOR = _rnlm8_sck<OUTPUT_PER_WG>();
  constexpr int W_KMUL = W_KMAJOR ? 16 : 1;
  constexpr int W_HIOFF = W_KMAJOR ? 1024 : 64;
  constexpr int SC_KMUL = SC_KMAJOR ? 16 : 1;
  static_assert(!W_KMAJOR || OUTPUT_PER_WG == 16,
                "the K-major weight layout is only packed for the K-parallel "
                "(OUTPUT_PER_WG == 16) call sites; see pack_dense_mxfp8");
  static_assert(!SC_KMAJOR || NUM_BLOCKS_32 % 4 == 0,
                "K-major scales group four E8M0 bytes per row per k-tile");

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
  // One LDS plane per folded row. FOLD_ROWS off is TOK_ROWS == 1, i.e. the
  // layout below is byte-for-byte what it always was.
  constexpr int TOK_ROWS = FOLD_ROWS ? BATCH_SIZE : 1;
  static_assert(!FOLD_ROWS || BATCH_SIZE <= 16,
                "FOLD_ROWS puts the row on the MFMA's 16 output columns");

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
  uint8_t *s_tok_fp8_all = (uint8_t *)_rnlm8_smem;
  uint8_t *s_tok_scales_all = s_tok_fp8_all + TOK_ROWS * FP8_TOK_DATA;
  // The resadd staging buffer sits past the quantizer's, rounded up to 128 B so
  // its uint2 traffic stays aligned. Only the K-parallel branch reuses
  // _rnlm8_smem (as lds_reduce, from offset 0) and that is after the MFMA, by
  // which point s_x_bf16 is dead.
  constexpr int RESADD_SMEM_OFF =
      ((TOK_ROWS * (FP8_TOK_DATA + NUM_BLOCKS_32) + 127) / 128) * 128;
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
  // Folded, the tile index IS the workgroup index: the row dimension has left
  // the tile space for the MFMA's output columns.
  int tok_base = FOLD_ROWS ? 0 : (tile_idx / n_wgs_per_xcd);
  int wg_idx = FOLD_ROWS ? tile_idx : (tile_idx % n_wgs_per_xcd);
  int const rows_here = FOLD_ROWS ? batch_count : 1;

  if constexpr (!FOLD_ROWS) {
    if (tok_base >= batch_count) {
      return;
    }
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
  // The depth-4 prefetch fill (below, inside the row loop's first trip) must
  // outlive the loop -- it is the tile's weights, which no longer depend on
  // the row.
  constexpr bool HOIST_PREFILL =
      LDS_PROLOGUE && (OUTPUT_PER_WG >= 64 ? (TILES_PER_WAVE == 1) : true);
  i32x8_t ph_a[HOIST_PREFILL ? 4 : 1];
  int ph_sa[HOIST_PREFILL ? 4 : 1];

  // Steps 1 and 2 run once per folded row, each into its own LDS plane.
  // rows_here is 1 unless FOLD_ROWS, so this is a no-op loop everywhere else.
  for (int _row = 0; _row < rows_here; ++_row) {
  int const tok_idx = tok_base + _row;
  uint8_t *const s_tok_fp8 = s_tok_fp8_all + _row * FP8_TOK_DATA;
  uint8_t *const s_tok_scales = s_tok_scales_all + _row * NUM_BLOCKS_32;
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
        (unsigned short const *)norm_input_ptr +
        (size_t)tok_idx * INPUT_ROW_STRIDE;
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
  if constexpr (HOIST_PREFILL) if (_row == 0) {
    // N-parallel gives each wave its own 16 rows starting at k-tile 0;
    // K-parallel gives all four waves the same 16 rows and splits K.
    int const h_row = (OUTPUT_PER_WG >= 64) ? (warp_id * 16 + col) : col;
    int const h_ki0 =
        (OUTPUT_PER_WG >= 64) ? 0 : (warp_id * (MFMA_ITERS / NUM_WAVES));
    // Under K-major the wave's sixteen rows are one contiguous block and the
    // row index moves inside it rather than scaling REDUCTION_SIZE; the k
    // stride then lives in W_KMUL, not in this base.
    uint8_t const *h_data =
        wg_data + (W_KMAJOR
                       ? static_cast<int64_t>(h_row / 16) * REDUCTION_SIZE * 16 +
                             static_cast<int64_t>(h_row % 16) * 64
                       : static_cast<int64_t>(h_row) * REDUCTION_SIZE);
    uint8_t const *h_scale =
        wg_scales + (SC_KMAJOR ? (h_row / 16) * NUM_BLOCKS_32 * 16 +
                                     (h_row % 16) * 4
                               : h_row * NUM_BLOCKS_32);
#pragma unroll
    for (int ki = 0; ki < 4; ki++) {
      ph_a[ki] = _rnlm8_load_w<W_KMUL, W_HIOFF>(
          h_data, (h_ki0 + ki) * K_PER_MFMA, g);
      ph_sa[ki] = _rnlm8_load_sc<SC_KMUL>(h_scale, (h_ki0 + ki) * 4, g);
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
    for (int i = tid; i < FP8_TOK_DATA / 16; i += MPK_NT) {
      int4 v;
      v.x = (int)g4[i * 4 + 0];
      v.y = (int)g4[i * 4 + 1];
      v.z = (int)g4[i * 4 + 2];
      v.w = (int)g4[i * 4 + 3];
      s4[i] = v;
    }
    using gu8 = __attribute__((address_space(1))) uint8_t const *;
    gu8 gs = (gu8)(pub + FP8_TOK_DATA);
    for (int i = tid; i < PUB_SCALES; i += MPK_NT) {
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
  } // _row: end of the per-row RMSNorm + quantize

  // Lane column `col` feeds output column `col`, which under FOLD_ROWS is
  // batch row `col`. Columns past the live batch replay row 0 -- the MFMA
  // computes them either way and the epilogue drops them.
  int const b_row = FOLD_ROWS ? ((col < rows_here) ? col : 0) : 0;
  uint8_t const *const s_tok_fp8 = s_tok_fp8_all + b_row * FP8_TOK_DATA;
  uint8_t const *const s_tok_scales = s_tok_scales_all + b_row * NUM_BLOCKS_32;

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

      // W_KMAJOR is false on this branch by construction (see _rnlm8_wk), so
      // these are the row-major bases; the form is shared with the K-parallel
      // branch only so the two read alike.
      uint8_t const *w_data_row =
          wg_data + (W_KMAJOR
                         ? static_cast<int64_t>(w_row / 16) * REDUCTION_SIZE * 16 +
                               static_cast<int64_t>(w_row % 16) * 64
                         : static_cast<int64_t>(w_row) * REDUCTION_SIZE);
      uint8_t const *w_scale_row =
          wg_scales + (SC_KMAJOR ? (w_row / 16) * NUM_BLOCKS_32 * 16 +
                                       (w_row % 16) * 4
                                 : w_row * NUM_BLOCKS_32);

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
          a[ki] = _rnlm8_load_w<W_KMUL, W_HIOFF>(w_data_row, ki * K_PER_MFMA, g);
          sa[ki] =
              _rnlm8_load_sc<SC_KMUL>(w_scale_row, ki * 4, g);
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
      } else if constexpr (_rnlm8_pf_groups(MFMA_ITERS, MPK_ATTN_PF_GROUPS) >=
                           2) {
        // ── Guard-free GROUPS-deep pipeline ────────────────────────────────
        // See _rnlm8_kloop_deep for the ISA the rotating loop below actually
        // compiles to. HOIST_PREFILL implies TILES_PER_WAVE == 1 here
        // (OUTPUT_PER_WG >= 64 on this branch), so ph_a is this tile's fill.
        constexpr int GR = _rnlm8_pf_groups(MFMA_ITERS, MPK_ATTN_PF_GROUPS);
        acc = _rnlm8_kloop_pick<GR, MFMA_ITERS, HOIST_PREFILL && GR == 4,
                                      W_KMUL, W_HIOFF, SC_KMUL>(
            w_data_row, w_scale_row, s_tok_fp8, s_tok_scales,
            /*ki_start=*/0, g, ph_a, ph_sa);
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
          a0 = _rnlm8_load_w<W_KMUL, W_HIOFF>(w_data_row, 0 * K_PER_MFMA, g);
          sa0 = _rnlm8_load_sc<SC_KMUL>(w_scale_row, 0 * 4, g);
          a1 = _rnlm8_load_w<W_KMUL, W_HIOFF>(w_data_row, 1 * K_PER_MFMA, g);
          sa1 = _rnlm8_load_sc<SC_KMUL>(w_scale_row, 1 * 4, g);
          a2 = _rnlm8_load_w<W_KMUL, W_HIOFF>(w_data_row, 2 * K_PER_MFMA, g);
          sa2 = _rnlm8_load_sc<SC_KMUL>(w_scale_row, 2 * 4, g);
          a3 = _rnlm8_load_w<W_KMUL, W_HIOFF>(w_data_row, 3 * K_PER_MFMA, g);
          sa3 = _rnlm8_load_sc<SC_KMUL>(w_scale_row, 3 * 4, g);
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
            a0 = _rnlm8_load_w<W_KMUL, W_HIOFF>(w_data_row, kt4, g);
            sa0 = _rnlm8_load_sc<SC_KMUL>(w_scale_row, kt4 / 32, g);
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
            a1 = _rnlm8_load_w<W_KMUL, W_HIOFF>(w_data_row, kt5, g);
            sa1 = _rnlm8_load_sc<SC_KMUL>(w_scale_row, kt5 / 32, g);
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
            a2 = _rnlm8_load_w<W_KMUL, W_HIOFF>(w_data_row, kt6, g);
            sa2 = _rnlm8_load_sc<SC_KMUL>(w_scale_row, kt6 / 32, g);
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
            a3 = _rnlm8_load_w<W_KMUL, W_HIOFF>(w_data_row, kt7, g);
            sa3 = _rnlm8_load_sc<SC_KMUL>(w_scale_row, kt7 / 32, g);
          }
        }
      } // MFMA_ITERS > FULL_PRELOAD_ITERS

      // ── Step 4: Bias epilogue, write BF16 output ─────────────────────────
      if (FOLD_ROWS ? (col < rows_here) : (col == 0)) {
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
        int out_idx = (FOLD_ROWS ? col : tok_base) * output_stride +
                      wg_idx * OUTPUT_PER_WG + wave_tile * 16 + g * 4;
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
    // w_row is `col` in [0,16) here, so under K-major w_row/16 is 0 and the
    // base is just this lane's 64-byte slot inside the wave's block.
    uint8_t const *w_data_row =
        wg_data + (W_KMAJOR
                       ? static_cast<int64_t>(w_row / 16) * REDUCTION_SIZE * 16 +
                             static_cast<int64_t>(w_row % 16) * 64
                       : static_cast<int64_t>(w_row) * REDUCTION_SIZE);
    uint8_t const *w_scale_row =
        wg_scales + (SC_KMAJOR ? (w_row / 16) * NUM_BLOCKS_32 * 16 +
                                     (w_row % 16) * 4
                               : w_row * NUM_BLOCKS_32);

    f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

    // ── Guard-free GROUPS-deep pipeline ──────────────────────────────────
    // Same transform as the N-parallel branch; here the trip count is this
    // wave's slice of K, not all of it. The hoisted prefill used w_row = col
    // and ki0 = warp_id * ITERS_PER_WAVE, which is exactly (w_row, ki_start),
    // so it seeds the first block directly.
    constexpr int KGR = _rnlm8_pf_groups(ITERS_PER_WAVE, MPK_ATTN_PF_GROUPS);
    if constexpr (KGR >= 2) {
      acc = _rnlm8_kloop_pick<KGR, ITERS_PER_WAVE, HOIST_PREFILL && KGR == 4,
                                  W_KMUL, W_HIOFF, SC_KMUL>(
          w_data_row, w_scale_row, s_tok_fp8, s_tok_scales,
          ki_start, g, ph_a, ph_sa);
    } else {

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
      a0 = _rnlm8_load_w<W_KMUL, W_HIOFF>(w_data_row, ki_start * K_PER_MFMA, g);
      sa0 = _rnlm8_load_sc<SC_KMUL>(w_scale_row, ki_start * 4, g);
      a1 = _rnlm8_load_w<W_KMUL, W_HIOFF>(w_data_row, (ki_start + 1) * K_PER_MFMA, g);
      sa1 = _rnlm8_load_sc<SC_KMUL>(w_scale_row, (ki_start + 1) * 4, g);
      a2 = _rnlm8_load_w<W_KMUL, W_HIOFF>(w_data_row, (ki_start + 2) * K_PER_MFMA, g);
      sa2 = _rnlm8_load_sc<SC_KMUL>(w_scale_row, (ki_start + 2) * 4, g);
      a3 = _rnlm8_load_w<W_KMUL, W_HIOFF>(w_data_row, (ki_start + 3) * K_PER_MFMA, g);
      sa3 = _rnlm8_load_sc<SC_KMUL>(w_scale_row, (ki_start + 3) * 4, g);
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
        a0 = _rnlm8_load_w<W_KMUL, W_HIOFF>(w_data_row, kt4, g);
        sa0 = _rnlm8_load_sc<SC_KMUL>(w_scale_row, kt4 / 32, g);
      }

      // Slot 1: compute k-tile ki+1, prefetch ki+5
      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 1) * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki + 1];
        acc = _gang_mfma_f8xf8(a1, b, acc, sa1, sb);
      }
      if (ki + 5 < ki_end) {
        int kt5 = (ki + 5) * K_PER_MFMA;
        a1 = _rnlm8_load_w<W_KMUL, W_HIOFF>(w_data_row, kt5, g);
        sa1 = _rnlm8_load_sc<SC_KMUL>(w_scale_row, kt5 / 32, g);
      }

      // Slot 2: compute k-tile ki+2, prefetch ki+6
      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 2) * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki + 2];
        acc = _gang_mfma_f8xf8(a2, b, acc, sa2, sb);
      }
      if (ki + 6 < ki_end) {
        int kt6 = (ki + 6) * K_PER_MFMA;
        a2 = _rnlm8_load_w<W_KMUL, W_HIOFF>(w_data_row, kt6, g);
        sa2 = _rnlm8_load_sc<SC_KMUL>(w_scale_row, kt6 / 32, g);
      }

      // Slot 3: compute k-tile ki+3, prefetch ki+7
      if (ki + 3 < ki_end) {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 3) * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki + 3];
        acc = _gang_mfma_f8xf8(a3, b, acc, sa3, sb);
      }
      if (ki + 7 < ki_end) {
        int kt7 = (ki + 7) * K_PER_MFMA;
        a3 = _rnlm8_load_w<W_KMUL, W_HIOFF>(w_data_row, kt7, g);
        sa3 = _rnlm8_load_sc<SC_KMUL>(w_scale_row, kt7 / 32, g);
      }
    }

    } // KGR < 2: the rotating depth-4 loop

    // Cross-wave LDS reduction (reuse token scratch area, dead after MFMA).
    // Folded, lane column `col` carries batch row `col`, so the scratch grows
    // a row axis: [wave][row][n]. At TOK_ROWS 2 and OUTPUT_PER_WG 16 that is
    // 4 * 2 * 16 floats = 512 B, against the ~12.7 KB of quantized token the
    // MFMA has just finished with.
    constexpr int RED_ROWS = TOK_ROWS;
    constexpr int RED_STRIDE = RED_ROWS * OUTPUT_PER_WG;
    static_assert(NUM_WAVES * RED_STRIDE * (int)sizeof(float) <=
                      TOK_ROWS * (FP8_TOK_DATA + NUM_BLOCKS_32),
                  "the cross-wave reduction no longer fits the token scratch");
    float *lds_reduce = (float *)_rnlm8_smem;
    // "dead after MFMA" is true per wave, not per workgroup. Wave w reads
    // s_tok_fp8[w*REDUCTION_SIZE/4 ...], but every wave's reduction slot lands
    // in the first NUM_WAVES*RED_STRIDE*4 bytes -- inside wave 0's read range.
    // Nothing ordered wave 1's store against wave 0's last k-block load, so a
    // stalled wave 0 read back another wave's accumulator as token bytes. The
    // window is 256 B unfolded and 512 B folded; the barrier is one per tile.
    __syncthreads();
    if (FOLD_ROWS ? (col < rows_here) : (col == 0)) {
      int const r = FOLD_ROWS ? col : 0;
      for (int i = 0; i < 4; i++) {
        lds_reduce[warp_id * RED_STRIDE + r * OUTPUT_PER_WG + g * 4 + i] =
            acc[i];
      }
    }
    __syncthreads();

    // Wave 0 reduces across waves and writes output with bias. Folded, wave 0
    // has 16 lane columns and needs only rows_here of them, so the row it
    // reduces is again `col` -- the same lane that produced it.
    if (warp_id == 0 && (FOLD_ROWS ? (col < rows_here) : (col == 0))) {
      int const r = FOLD_ROWS ? col : 0;
      unsigned short packed[4];
      for (int i = 0; i < 4; i++) {
        float v = 0.0f;
        for (int w = 0; w < NUM_WAVES; w++) {
          v += lds_reduce[w * RED_STRIDE + r * OUTPUT_PER_WG + g * 4 + i];
        }

        int out_n = wg_idx * OUTPUT_PER_WG + g * 4 + i;

        unsigned bt = (unsigned)d_bias[out_n] << 16;
        float bv;
        __builtin_memcpy(&bv, &bt, 4);

        packed[i] = _gang_float_to_bf16(v + bv);
      }
      int out_idx = (FOLD_ROWS ? (tok_base + r) : tok_base) * output_stride +
                    wg_idx * OUTPUT_PER_WG + g * 4;
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
