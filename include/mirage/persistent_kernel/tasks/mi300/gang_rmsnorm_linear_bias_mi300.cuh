/* Fused RMSNorm + Gang Linear + Bias for MI300/MI350.
 *
 * Eliminates the dispatch barrier between RMSNorm and the QKV/router gang
 * linear by having every gang-linear worker compute the RMSNorm prologue
 * locally before its MFMA. All workers compute the same value and write to
 * the same output buffer; concurrent writes are safe (idempotent).
 *
 * Saves ~4-5us per fused op × 2 RMSNorm tasks per layer × 36 layers per
 * iteration (~300us total) by removing 72 task dispatch barriers per token.
 *
 * NOTE: this differs from gang_rmsnorm_linear_mi300.cuh in two ways:
 *   1. Correct AMD wavefront=64 cross-wave reduction (the existing file uses
 *      >>5 which is wrong on AMD and would over-count by 2x).
 *   2. Supports ACTUAL_HIDDEN_DIM != STORAGE_DIM for GPT-OSS (which pads
 *      hidden=2880 to STORAGE_DIM=3072 for tile alignment).
 */
#pragma once
#include "gang_linear_mi300.cuh"
#include "moe_topk_sigmoid_bias_mi300.cuh"
#include "moe_topk_softmax_mi300.cuh"
#include <hip/hip_bf16.h>
#include "mpk_bsdbg.cuh"

namespace kernel {

// Self-heal gate for the Mechanism-C flag polls. Same value and same
// reasoning as the copy in gang_mla_full_layer_fused_mi300.cuh, which carries
// the full note; duplicated under #ifndef because the monoliths land in this
// translation unit in include order and any one of them may be first.
#ifndef MPK_FL_REPUBLISH_SPINS
#define MPK_FL_REPUBLISH_SPINS 1024
#endif

namespace gang_rmsnorm_detail {
using bf16 = __hip_bfloat16;

// AMD-correct in-place RMSNorm prologue. Computed redundantly by all workers
// in the workgroup; result written to global memory at output_ptr.
//
// STORAGE_DIM: number of bf16 elements stored along the hidden axis (padded)
// ACTUAL_HIDDEN_DIM: divisor for the RMS mean (unpadded hidden size)
// NORM_SPAN: number of leading elements the sum-of-squares runs over
//
// The norm weight is assumed zero-padded past ACTUAL_HIDDEN_DIM, so the
// padded tail multiplies by zero. Sum-of-squares is taken over STORAGE_DIM
// (the padded tail is also zero in input, since the producer pads with 0),
// but the divisor uses ACTUAL_HIDDEN_DIM.
//
// NORM_SPAN < STORAGE_DIM is for GLM's q_a_layernorm, whose input row is the
// fused `[q_a | kv_latent]` projection: only the leading q_a columns belong in
// the RMS denominator, and the latent columns are decidedly not zero. The
// whole row is still normalized-and-scaled on the way out, so the trailing
// columns come out zero (the norm weight is zero there) and the downstream
// GEMM's matching weight columns are zero too.
// The sum-of-squares half of rmsnorm_inline_amd, returning 1/rms and writing
// nothing.
//
// It exists for the consumers that do not want the normalized row in memory.
// A fused RMSNorm + quantized GEMM writes the row to global only so that its
// own quantizer can read it straight back -- a 4 KB store and a 4 KB load per
// block, on the dependency path, for a value that never leaves the workgroup.
// Handing the caller `rms_rcp` instead lets it apply `* rms_rcp * weight[i]`
// at the point of use, where the input is still L1-hot from Phase 1 below.
//
// Phases 1-3 are rmsnorm_inline_amd's, minus the register cache: nothing here
// revisits the input, so caching it would only hold VGPRs. Phase 4 is the
// caller's. The trailing __syncthreads() is the one that publishes red[0], so
// the returned value is uniform and the LDS is free on return.
// ── MPK_RMSNORM_DPP: the ssq butterfly off LDS ───────────────────────────
// `__shfl_xor` lowers to `ds_bpermute` -- an LDS round trip and an
// `s_waitcnt lgkmcnt` to move a value that never left the register file. The
// six-step butterfly below therefore pays six of them on the dependency path
// of every RMSNorm. gpt-oss measured exactly this swap at 1.851 -> 1.835 ms
// ("RMSNorm ssq off LDS", 2ee5278); the census on today's GLM image says it is
// unported here -- 514 ds_bpermute against 8 cross-lane VALU ops in the whole
// code object.
//
// COVERAGE. The first cut only swapped rmsnorm_rcp_amd. That is the
// LDS_PROLOGUE=false fallback. GLM-5.2 ships LDS_PROLOGUE=true, so the live
// ssq is _rnlm8_stage_norm_rcp / _rnlm8_resadd_norm_rcp (mxfp8 qkv/o_proj)
// plus rmsnorm_inline_amd and the fused-gate ssq in this file. All of those
// store only lane 0, so the same helper is legal at each.
//
// 2026-09-16, 1024/1024, chunks=32, GPUs 4-7. Complete coverage compiled
// (ds_bpermute 514 -> 430, permlane32 4 -> 18, mov_dpp 0 -> 56) and is
// NOT bit-identical on this path: decode_min 11.318 inside the 11.25-11.36
// band, G1 PASS, but G2_distinct 0.502 and TEXT_TAIL is garbage against the
// Boulter attractor. Keep OFF. Do not default on until an isolated site
// reproduces the gpt-oss 2ee5278 hash.
//
// BIT-IDENTICAL, and that is checkable rather than hopeful: the only lane this
// reduction's caller reads is lane 0 (`if (lane_id == 0) red[wave_id] = sum`),
// and for lane 0 `xor-N` and `shr-N` name the same source lane, since
// `0 ^ N == 0 + N`. Lane 0 therefore walks the same summation tree in the same
// order; the other 63 lanes hold different partials and are discarded.
//
// The two permlane swaps are the form already validated in this tree at
// gang_mla_decode_mi300.cuh:868. `bound_ctrl:1` makes an out-of-range DPP lane
// read 0, the identity for the add. `s_nop 1` before each cross-lane op is
// required, not defensive: CDNA4 ISA Table 11 gives two wait states for a VALU
// write followed by a DPP/PERMLANE read of the same register, and every stage
// here reads what the previous add just wrote.
#ifndef MPK_RMSNORM_DPP
#define MPK_RMSNORM_DPP 0
#endif
#if MPK_RMSNORM_DPP
__device__ __forceinline__ float _mpk_wave_sum_to_lane0(float v) {
  // xor-32, then xor-16, via the register-swap form.
  {
    float a = v, b = v;
    asm volatile("s_nop 1\n\tv_permlane32_swap_b32_e32 %0, %1"
                 : "+v"(a), "+v"(b));
    v = a + b;
  }
  {
    float a = v, b = v;
    asm volatile("s_nop 1\n\tv_permlane16_swap_b32_e32 %0, %1"
                 : "+v"(a), "+v"(b));
    v = a + b;
  }
  // Fold each 16-lane row into its lane 0. One asm block: separate blocks let
  // the compiler slip spurious s_nop between the stages.
  float t;
  asm volatile(
      "s_nop 1\n\t"
      "v_mov_b32_dpp %1, %0 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
      "v_add_f32 %0, %0, %1\n\t"
      "s_nop 1\n\t"
      "v_mov_b32_dpp %1, %0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
      "v_add_f32 %0, %0, %1\n\t"
      "s_nop 1\n\t"
      "v_mov_b32_dpp %1, %0 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
      "v_add_f32 %0, %0, %1\n\t"
      "s_nop 1\n\t"
      "v_mov_b32_dpp %1, %0 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
      "v_add_f32 %0, %0, %1"
      : "+v"(v), "=&v"(t));
  return v;
}
#endif

template <int STORAGE_DIM, int ACTUAL_HIDDEN_DIM, int NORM_SPAN = STORAGE_DIM>
__device__ __forceinline__ float rmsnorm_rcp_amd(void const *input_ptr,
                                                 float eps = 1e-5f) {
  bf16 const *__restrict__ d_input = static_cast<bf16 const *>(input_ptr);

  // 8 bf16 per thread needs STORAGE_DIM to be a multiple of 256*8. At
  // STORAGE_DIM=1024 -- q_b, whose input is the 768-wide q_lora row padded to
  // 1024 -- it is not, so VEC_ITERS came out 0, the whole vectorized loop was
  // dead, and the row was read by the scalar tail below: four separate 2-byte
  // global loads per thread where one 8-byte load would do. Drop to 4 bf16 per
  // thread for those shapes. STORAGE_DIM=2048 is unaffected and still takes
  // the 8-wide path.
  constexpr int NTHREADS = 256;
  constexpr int VEC_SIZE = (STORAGE_DIM % (NTHREADS * 8) == 0) ? 8 : 4;
  int const tid = threadIdx.x;
  int const nthreads = MPK_NT;
  constexpr int VEC_ITERS = STORAGE_DIM / (NTHREADS * VEC_SIZE);
  constexpr int VEC_END = VEC_ITERS * NTHREADS * VEC_SIZE;
  float sum = 0.0f;

#pragma unroll 1
  for (int v = 0; v < VEC_ITERS; v++) {
    int offset = (v * nthreads + tid) * VEC_SIZE;
    // addrspace(1) so this is global_load, not flat_load. A flat instruction
    // increments both vmcnt and lgkmcnt on gfx9, so the lgkmcnt(0) that
    // retires the cross-wave reduction's LDS traffic below would also wait on
    // this row. Same two-step cast as gang_gemv_mxfp8_detail::ld_g; d_input is
    // always device global here (the LDS-staged caller uses
    // _rnlm8_resadd_norm_rcp instead, which never reaches this function).
    uint64_t const *in_lo_p =
        reinterpret_cast<uint64_t const *>(&d_input[offset]);
    uint64_t in_lo =
        *(__attribute__((address_space(1))) uint64_t const *)in_lo_p;
    bf16 const *lo = reinterpret_cast<bf16 const *>(&in_lo);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float vlo = __bfloat162float(lo[i]);
      if constexpr (NORM_SPAN == STORAGE_DIM) {
        sum += vlo * vlo;
      } else {
        sum += (offset + i < NORM_SPAN) ? vlo * vlo : 0.0f;
      }
    }
    if constexpr (VEC_SIZE == 8) {
      uint64_t const *in_hi_p =
          reinterpret_cast<uint64_t const *>(&d_input[offset + 4]);
      uint64_t in_hi =
          *(__attribute__((address_space(1))) uint64_t const *)in_hi_p;
      bf16 const *hi = reinterpret_cast<bf16 const *>(&in_hi);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float vhi = __bfloat162float(hi[i]);
        if constexpr (NORM_SPAN == STORAGE_DIM) {
          sum += vhi * vhi;
        } else {
          sum += (offset + 4 + i < NORM_SPAN) ? vhi * vhi : 0.0f;
        }
      }
    }
  }
  for (int i = VEC_END + tid; i < STORAGE_DIM; i += nthreads) {
    float val = __bfloat162float(d_input[i]);
    if constexpr (NORM_SPAN == STORAGE_DIM) {
      sum += val * val;
    } else {
      sum += (i < NORM_SPAN) ? val * val : 0.0f;
    }
  }

#if MPK_RMSNORM_DPP
  sum = _mpk_wave_sum_to_lane0(sum);
#else
#pragma unroll
  for (int offset = 32; offset > 0; offset >>= 1) {
    sum += __shfl_xor(sum, offset);
  }
#endif

  __shared__ float red[16];
  int wave_id = tid >> 6;
  int lane_id = tid & 63;
  int num_waves = nthreads >> 6;
  if (lane_id == 0) {
    red[wave_id] = sum;
  }
  __syncthreads();
  if (wave_id == 0) {
    sum = (lane_id < num_waves) ? red[lane_id] : 0.0f;
    for (int offset = num_waves >> 1; offset > 0; offset >>= 1) {
      sum += __shfl_xor(sum, offset);
    }
    if (lane_id == 0) {
      red[0] = sum;
    }
  }
  __syncthreads();

  return rsqrtf(red[0] / float(ACTUAL_HIDDEN_DIM) + eps);
}

// NUM_ROWS is the token count, not a tile height: this prologue is the
// redundant one every gang-linear worker runs before its own tile, so it has
// to produce the WHOLE normalized activation, all rows of it, or the GEMM
// reads a row nobody wrote. At NUM_ROWS 1 the loop below is the straight-line
// code it has always been.
//
// The row loop wraps phases 1-4 rather than being pushed inside them: the
// register cache (in_cache_lo/hi, tail_cache) is sized for one row and is
// reused per trip, so a second token costs zero extra VGPRs and one more pass
// over a row that is already in L2 from the first.
template <int STORAGE_DIM,
          int ACTUAL_HIDDEN_DIM,
          int NORM_SPAN = STORAGE_DIM,
          int NUM_ROWS = 1,
          int ROW_STRIDE = STORAGE_DIM>
__device__ __forceinline__ void rmsnorm_inline_amd(void const *input_ptr,
                                                   void const *weight_ptr,
                                                   void *output_ptr,
                                                   float eps = 1e-5f) {
  bf16 const *__restrict__ d_weight = static_cast<bf16 const *>(weight_ptr);

  constexpr int VEC_SIZE = 8;   // 8 bf16 per 128-bit load
  constexpr int NTHREADS = 256; // block size for gang RMSNorm
  int const tid = threadIdx.x;
  int const nthreads = MPK_NT;
  constexpr int VEC_ITERS = STORAGE_DIM / (NTHREADS * VEC_SIZE);

  // ── Phase 1: sum of squares + cache input in registers ──
  // Cache d_input values to avoid re-reading from HBM in Phase 4.
  // GPT-OSS 120B: STORAGE_DIM=3072, 256 threads, VEC_SIZE=8 → VEC_ITERS=1,
  // tail=1024 elems → 4/thread. Total cache: 8 + 4 = 12 floats/thread.
  constexpr int _VEC_END = VEC_ITERS * NTHREADS * VEC_SIZE;
  constexpr int _TAIL_ELEMS = STORAGE_DIM - _VEC_END;
  constexpr int _MAX_TAIL = (_TAIL_ELEMS + NTHREADS - 1) / NTHREADS;

#pragma unroll 1
  for (int _row = 0; _row < NUM_ROWS; ++_row) {
  bf16 const *__restrict__ d_input =
      static_cast<bf16 const *>(input_ptr) + (size_t)_row * ROW_STRIDE;
  bf16 *__restrict__ d_output =
      static_cast<bf16 *>(output_ptr) + (size_t)_row * ROW_STRIDE;
  float sum = 0.0f;

  // Vectorized cache: raw uint64_t pairs (2 per iter = lo + hi)
  uint64_t in_cache_lo[VEC_ITERS > 0 ? VEC_ITERS : 1];
  uint64_t in_cache_hi[VEC_ITERS > 0 ? VEC_ITERS : 1];
#pragma unroll 1
  for (int v = 0; v < VEC_ITERS; v++) {
    int offset = (v * nthreads + tid) * VEC_SIZE;
    uint64_t in_lo = *reinterpret_cast<uint64_t const *>(&d_input[offset]);
    uint64_t in_hi = *reinterpret_cast<uint64_t const *>(&d_input[offset + 4]);
    in_cache_lo[v] = in_lo;
    in_cache_hi[v] = in_hi;
    bf16 const *lo = reinterpret_cast<bf16 const *>(&in_lo);
    bf16 const *hi = reinterpret_cast<bf16 const *>(&in_hi);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float vlo = __bfloat162float(lo[i]);
      float vhi = __bfloat162float(hi[i]);
      if constexpr (NORM_SPAN == STORAGE_DIM) {
        sum += vlo * vlo;
        sum += vhi * vhi;
      } else {
        sum += (offset + i < NORM_SPAN) ? vlo * vlo : 0.0f;
        sum += (offset + 4 + i < NORM_SPAN) ? vhi * vhi : 0.0f;
      }
    }
  }
  // Scalar tail cache
  float tail_cache[_MAX_TAIL > 0 ? _MAX_TAIL : 1];
  int n_tail = 0;
  int const VEC_END = _VEC_END;
  for (int i = VEC_END + tid; i < STORAGE_DIM; i += nthreads) {
    float val = __bfloat162float(d_input[i]);
    tail_cache[n_tail++] = val;
    if constexpr (NORM_SPAN == STORAGE_DIM) {
      sum += val * val;
    } else {
      sum += (i < NORM_SPAN) ? val * val : 0.0f;
    }
  }

// ── Phase 2: wavefront reduction (AMD wavefront = 64 lanes) ──
#if MPK_RMSNORM_DPP
  sum = _mpk_wave_sum_to_lane0(sum);
#else
#pragma unroll
  for (int offset = 32; offset > 0; offset >>= 1) {
    sum += __shfl_xor(sum, offset);
  }
#endif

  // ── Phase 3: cross-wavefront reduction via shared memory ──
  // Use a static __shared__ array — independent of CK pipeline's dynamic LDS.
  __shared__ float red[16]; // up to 16 wavefronts (1024 threads)
  int wave_id = tid >> 6;
  int lane_id = tid & 63;
  int num_waves = nthreads >> 6;

  if (lane_id == 0) {
    red[wave_id] = sum;
  }
  __syncthreads();

  if (wave_id == 0) {
    sum = (lane_id < num_waves) ? red[lane_id] : 0.0f;
    // Reduce across num_waves (typically 4 for blockDim=256)
    for (int offset = num_waves >> 1; offset > 0; offset >>= 1) {
      sum += __shfl_xor(sum, offset);
    }
    if (lane_id == 0) {
      red[0] = sum;
    }
  }
  __syncthreads();

  float rms_rcp = rsqrtf(red[0] / float(ACTUAL_HIDDEN_DIM) + eps);

// ── Phase 4: apply normalization using cached input (no HBM re-read) ──
#pragma unroll 1
  for (int v = 0; v < VEC_ITERS; v++) {
    int offset = (v * nthreads + tid) * VEC_SIZE;
    // Reuse cached input from Phase 1
    uint64_t in_lo = in_cache_lo[v];
    uint64_t in_hi = in_cache_hi[v];
    uint64_t w_lo = *reinterpret_cast<uint64_t const *>(&d_weight[offset]);
    uint64_t w_hi = *reinterpret_cast<uint64_t const *>(&d_weight[offset + 4]);
    bf16 const *in_lo_a = reinterpret_cast<bf16 const *>(&in_lo);
    bf16 const *in_hi_a = reinterpret_cast<bf16 const *>(&in_hi);
    bf16 const *w_lo_a = reinterpret_cast<bf16 const *>(&w_lo);
    bf16 const *w_hi_a = reinterpret_cast<bf16 const *>(&w_hi);

    bf16 out[VEC_SIZE];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      out[i] = __float2bfloat16(__bfloat162float(in_lo_a[i]) * rms_rcp *
                                __bfloat162float(w_lo_a[i]));
      out[4 + i] = __float2bfloat16(__bfloat162float(in_hi_a[i]) * rms_rcp *
                                    __bfloat162float(w_hi_a[i]));
    }
    *reinterpret_cast<uint64_t *>(&d_output[offset]) =
        *reinterpret_cast<uint64_t *>(&out[0]);
    *reinterpret_cast<uint64_t *>(&d_output[offset + 4]) =
        *reinterpret_cast<uint64_t *>(&out[4]);
  }
  // Scalar tail: use cached float values
  int ti = 0;
  for (int i = VEC_END + tid; i < STORAGE_DIM; i += nthreads) {
    float val = tail_cache[ti++];
    float w = __bfloat162float(d_weight[i]);
    d_output[i] = __float2bfloat16(val * rms_rcp * w);
  }
  // Not optional at NUM_ROWS > 1: red[] is reused by the next row's
  // cross-wave reduction, and a fast wave would overwrite red[wave_id] while
  // a slow one is still reading red[0] for this row's rms_rcp.
  __syncthreads();
  } // _row

  // Workgroup-scope fence: ensure stores visible to the linear loads below
  // within this workgroup. (Cross-XCD coherence not needed: each XCD
  // computes the same value redundantly and reads its own writes.)
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  __syncthreads();
}

} // namespace gang_rmsnorm_detail

// Fused RMSNorm + Gang Linear + Bias.
//
// Step 1: every worker on every XCD computes the same normalized output to
//         norm_output_ptr (idempotent concurrent writes).
// Step 2: this workgroup's gang-linear tile reads from norm_output_ptr,
//         applies bias in the epilogue.
template <typename T,
          int BATCH_SIZE,
          int REDUCTION_SIZE,
          int ACTUAL_HIDDEN_DIM = REDUCTION_SIZE,
          int NORM_SPAN = REDUCTION_SIZE>
__device__ __forceinline__ void gang_rmsnorm_linear_bias_kernel(
    void const *norm_input_ptr,  // [batch, REDUCTION_SIZE]
    void const *norm_weight_ptr, // [REDUCTION_SIZE]  (zero-padded past ACTUAL)
    void *norm_output_ptr,       // [batch, REDUCTION_SIZE]
    void const *linear_weight_ptr, // [chunk_N, REDUCTION_SIZE]
    void const *bias_ptr,          // [1, full_N]
    void *linear_output_ptr,       // [batch, o_stride]
    int num_active_tokens,
    int tile_n,
    int o_stride,
    int m_tiles,
    int n_tiles,
    int wgm,
    int tile_idx) {
  // Step 1: redundant RMSNorm.
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
    MPK_BSDBG_SEQ(26, norm_input_ptr, REDUCTION_SIZE, "d_attn_proj",
                  BATCH_SIZE, REDUCTION_SIZE, 24);
  }

  //
  // BATCH_SIZE rows, not one. BATCH_SIZE here is m_per_tile, and the callers
  // that need more than one row all leave m_tiles at 1, so it is the token
  // count -- which is exactly the number of rows the gang linear below will
  // read out of norm_output_ptr. Normalizing only row 0 left the second
  // token's row as whatever the scratch last held, and the row stride is the
  // reduction extent for both.
  gang_rmsnorm_detail::rmsnorm_inline_amd<REDUCTION_SIZE,
                                          ACTUAL_HIDDEN_DIM,
                                          NORM_SPAN,
                                          BATCH_SIZE,
                                          REDUCTION_SIZE>(
      norm_input_ptr, norm_weight_ptr, norm_output_ptr);

  // Step 2: gang linear with bias, reading from norm_output_ptr.
  gang_linear_kernel<T, BATCH_SIZE, REDUCTION_SIZE>(norm_output_ptr,
                                                    linear_weight_ptr,
                                                    linear_output_ptr,
                                                    num_active_tokens,
                                                    tile_n,
                                                    o_stride,
                                                    m_tiles,
                                                    n_tiles,
                                                    wgm,
                                                    tile_idx,
                                                    bias_ptr);
}

// Croc-style fused gate + TopK: 128 workers (one per expert).
//
// Replaces the 8-worker small_router_linear approach with 128-way parallel
// dot products, matching croc's mega_phase4_gate pattern:
//   - Each worker computes 1 expert's gate logit (not 16 serially)
//   - RMSNorm is fused into the GEMV (no intermediate BF16 round-trip)
//   - 128 workers = 16/XCD, vs old 1/XCD = 8x more parallelism
//   - Expected: ~3-5 us/layer (down from 25.3 us/layer)
//
// Cross-XCD synchronization: same atomic counter barrier as before.
// The LAST worker (count == 128) computes TopK softmax inline.
//
// Memory ordering: write-through stores (sc0 sc1) for logits bypass L2.
// buffer_inv on the TopK reader invalidates stale L2 entries.
namespace gang_rmsnorm_topk_detail {
using bf16_t = __hip_bfloat16;

__device__ __forceinline__ int get_xcd_id() {
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
  int xcd_id;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcd_id));
  return xcd_id;
#else
  return 0;
#endif
}

// Noinline wrapper for TopK to prevent code bloat in the megakernel's hot path.
// Without this, inlining TopK into the persistent_kernel function causes the
// compiler's instruction scheduler to pessimize the CK GEMM pipeline, making
// ALL task types ~5-12% slower.
// ROUTING_ROW_STRIDE is routing_indices' ALLOCATED row stride, i.e. the
// graph's BATCH_SIZE, which is not the same thing as num_active_tokens once
// BATCH_SIZE > 1. 0 keeps the pre-MTP behaviour (stride == num_rows). See the
// long comment on the parameter in moe_topk_sigmoid_bias_mi300.cuh.
template <typename T, int NUM_EXPERTS, int K, int ROUTING_ROW_STRIDE = 0>
__device__ __attribute__((noinline)) void
    topk_noinline(void *logits_scratch_ptr,
                  void *topk_weight_ptr,
                  void *routing_indices_ptr,
                  void *active_expert_ids_ptr,
                  void *gang_counter_ptr,
                  int num_active_tokens) {
  constexpr int CHUNK_N = NUM_EXPERTS / 8;
  int xcd_id = get_xcd_id();
  void *logits_base = static_cast<T *>(logits_scratch_ptr) -
                      static_cast<int64_t>(xcd_id) * CHUNK_N;

  // Invalidate L2 cache before reading logits. Write-through stores from
  // all 128 workers bypassed L2 → HBM. buffer_inv ensures the reader's L2
  // fetches fresh data from HBM (zeroing stores from previous iteration may
  // have populated L2 with stale zeros).
#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  unsigned long long _tki_t0 = __builtin_amdgcn_s_memrealtime();
#endif
  asm volatile("buffer_inv" ::: "memory");
#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  // buffer_inv invalidates this XCD's ENTIRE L2, not just the 256-byte logit
  // line, so it costs the completer both the invalidate itself and every
  // subsequent miss on data it had cached. Timed separately from the softmax
  // body because the fix differs: the invalidate can be narrowed (the logits
  // are st_wt, so a plain load with sc0 sc1 reads past L2 without nuking it),
  // whereas the body would need restructuring.
  if (threadIdx.x == 0) {
    atomicAdd(&g_tk_inv_ns, __builtin_amdgcn_s_memrealtime() - _tki_t0);
  }
#endif

  topk_softmax_mi300_task_impl<T,
                               /*VPT=*/8,
                               NUM_EXPERTS,
                               /*WARPS_PER_CTA=*/4,
                               /*BYTES_PER_LDG=*/16,
                               ROUTING_ROW_STRIDE>(logits_base,
                                                     topk_weight_ptr,
                                                     num_active_tokens,
                                                     K,
                                                     routing_indices_ptr,
                                                     active_expert_ids_ptr,
                                                     0,
                                                     NUM_EXPERTS,
                                                     true);

  // Reset counter for the next layer's use. Ordinary store: the buffer_wbl2 in
  // the release fence below retires it before any consumer runs. It must stay
  // ordinary -- st_wt here is part of the measured-negative change described at
  // that fence.
  if (threadIdx.x == 0) {
    *static_cast<int *>(gang_counter_ptr) = 0;
  }
}

// `noaux_tc` counterpart of topk_noinline, for GLM / DeepSeek-style routers.
//
// The bias behaves differently here and that difference reaches back into the
// GEMV: `e_score_correction_bias` steers the top-k *choice* through
// sigmoid(logit) + bias, but the weight that gets emitted comes from the
// unbiased sigmoid. So the caller must leave the logit alone and hand the full
// bias vector down to this tail, instead of folding one element into its own
// logit the way the softmax path does.
template <typename T, int NUM_EXPERTS, int K, int ROUTING_ROW_STRIDE = 0>
__device__ __attribute__((noinline)) void
    topk_sigmoid_noinline(void *logits_scratch_ptr,
                          void *bias_ptr,
                          void *topk_weight_ptr,
                          void *routing_indices_ptr,
                          void *active_expert_ids_ptr,
                          void *gang_counter_ptr,
                          int num_active_tokens,
                          bool renormalize,
                          float routed_scaling_factor,
                          int num_shared_experts,
                          int *routing_ready_ptr,
                          int epoch_hint) {
  constexpr int CHUNK_N = NUM_EXPERTS / 8;
  int xcd_id = get_xcd_id();
  void *logits_base = static_cast<T *>(logits_scratch_ptr) -
                      static_cast<int64_t>(xcd_id) * CHUNK_N;
  // #135. Only the elected block runs this, so these four marks cost one PCIe
  // write each per MoE layer -- and they are the difference between "wedged
  // somewhere in the router" and "wedged in the serial tail".
  int const tid = threadIdx.x;
  (void)tid;
  MPK_WS_MARK(775, xcd_id);

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  // Split SP6[4] (the 7.6 us serial tail) into its two halves, so the next
  // attempt on it is aimed rather than guessed:
  //   SP6[6] = buffer_inv + the selection itself
  //   SP6[7] = counter reset + syncthreads + release fence + epoch publish
  // Only the elected block runs any of this, so these are wall microseconds
  // per MoE layer, not worker-seconds. SP6[5] already carries the event count.
  unsigned long long _tk_t0 = __builtin_amdgcn_s_memrealtime();
#endif

  asm volatile("buffer_inv" ::: "memory");

  topk_sigmoid_bias_mi300_task_impl<T,
                                    /*VPT=*/8,
                                    NUM_EXPERTS,
                                    /*WARPS_PER_CTA=*/4,
                                    /*BYTES_PER_LDG=*/16,
                                    /*K_STATIC=*/K,
                                    ROUTING_ROW_STRIDE,
                                    /*SKIP_CLEARS=*/(MPK_TOPK_SKIP_CLEARS &&
                                                     ROUTING_ROW_STRIDE == 1)>(
      logits_base,
      bias_ptr,
      topk_weight_ptr,
      num_active_tokens,
      K,
      routing_indices_ptr,
      active_expert_ids_ptr,
      0,
      NUM_EXPERTS,
      renormalize,
      routed_scaling_factor,
      num_shared_experts);
  MPK_WS_MARK(776, xcd_id);

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  unsigned long long _tk_t1 = __builtin_amdgcn_s_memrealtime();
  if (threadIdx.x == 0 && g_subphase_active) {
    atomicAdd(&g_subphase_ns[6][6], (_tk_t1 - _tk_t0) * 10); // selection
  }
#endif

  // Reset counter for the next layer's use. Ordinary store: the buffer_wbl2 in
  // the release fence below retires it before any consumer runs. It must stay
  // ordinary -- st_wt here is part of the measured-negative change described at
  // that fence.
  if (threadIdx.x == 0) {
    *static_cast<int *>(gang_counter_ptr) = 0;
  }

  // Release the MoE workers, when a fused caller has parked them on this
  // epoch. Lifted from gang_linear_mxfp4_res_bias_rmsnorm_topk_mi300.cuh:1070,
  // fence and all.
  //
  // The fence is required, not an optimization. The consumers are MoE workers
  // on *other* XCDs, and what they read after it is active_expert_ids and
  // routing_indices, written just above by all 256 threads of this block.
  // __syncthreads orders those within this block only; it says nothing about
  // when they reach another XCD's L2. The release flags go out via st_wt,
  // bypassing L2, so without a GPU-scope fence a flag can land in HBM ahead of
  // the routing data it advertises.
  //
  // The failure is silent rather than a crash: every stale value is still in
  // range, so a token is simply routed through the previous layer's experts.
  //
  // MEASURED, do not re-attempt: converting the tail's last two ordinary
  // stores to st_wt and downgrading this to `s_waitcnt vmcnt(0)` is a 2.2%
  // regression (12.897 vs 12.615 ms, two matched repeats each, 2026-08-19) and
  // leaves SP6[4] -- this tail's own cost -- unchanged at 0.58 s aggregate.
  // The 7.6 us tail is the TopK compute, not this fence. The likely reason
  // removing it costs: buffer_wbl2 drains this XCD's dirty L2 during the one
  // window where 231 workers are parked anyway; defer it and the writeback
  // lands on the MoE critical path instead.
  MPK_WS_MARK(777, xcd_id);
  if (routing_ready_ptr) {
    __syncthreads();
    if (threadIdx.x == 0) {
      threadfence_gpu();
      // Publish the epoch the consumers will actually wait for.
      //
      // The read-modify-write below is only equivalent to that when the
      // publication count and the consumer's layer index advance in lockstep,
      // and under EP they do not: the EP_TAIL_ONLY layer folds the last real
      // layer's MoE output and returns before the router ever runs, so it
      // consumes an `ml` slot -- hence a `task_layer_idx` -- without
      // publishing. One skipped bump per decode iteration, and the fused
      // caller's `routing_expected = task_layer_idx + 1` runs permanently
      // ahead of this counter. The first symptom is a hard hang at the Phase 4
      // routing poll of pc_iter 2, layer 0, with all 240 workers parked.
      //
      // Storing the caller's value instead of incrementing makes writer and
      // reader agree by construction, exactly as every Mechanism C barrier
      // flag in this file already does. It also drops an uncached HBM round
      // trip (0.30-0.44 us measured in gpt-oss) from the serial TopK tail that
      // 239 workers are blocked on. epoch_hint < 0 keeps the old read for
      // callers with no layer index to hand.
      int epoch =
          (epoch_hint >= 0) ? epoch_hint : (ld_nt_s32(routing_ready_ptr) + 1);
      st_flag_u32((void *)routing_ready_ptr, (unsigned)epoch);
      for (int x = 0; x < 8; x++) {
        st_flag_u32((void *)&routing_ready_ptr[(1 + x) * 16], (unsigned)epoch);
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }
  }
  MPK_WS_MARK(778, xcd_id);
  // Stage stamp 35: routing published, by the one elected block. S34 (max)
  // -> S35 is the serial TopK tail; S35 -> S5 is the routing poll seeing it.
  if (threadIdx.x == 0) {
    mpk_stage_stamp(35);
  }

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  if (threadIdx.x == 0 && g_subphase_active) {
    atomicAdd(&g_subphase_ns[6][7],
              (__builtin_amdgcn_s_memrealtime() - _tk_t1) * 10); // release
  }
#endif
}

// MPK_ROUTER_KSPLIT engages only on this shape; the fused caller tests the same
// predicate to place the TopK on its own worker.
template <int BATCH_SIZE,
          int REDUCTION_SIZE,
          int NUM_EXPERTS,
          bool SIGMOID_BIAS,
          bool OPROJ_BARRIER,
          int EXPERTS_PER_TILE,
          bool SUM_LL>
constexpr bool router_ksplit_on() {
  return MPK_ROUTER_KSPLIT && MPK_ROUTER_XSPLIT && MPK_ROUTER_LL && SUM_LL &&
         OPROJ_BARRIER && SIGMOID_BIAS && BATCH_SIZE == 1 &&
         EXPERTS_PER_TILE == 2 && NUM_EXPERTS == 256 &&
         REDUCTION_SIZE % (16 * 64) == 0;
}

#if MPK_ROUTER_LL
// MPK_ROUTER_LL's TopK, run by one router block per XCD once its own logits
// are out: wait until all NUM_EXPERTS words carry this layer's epoch, then
// select into this XCD's routing copy and release this XCD's MoE workers
// through its L2. Every XCD selects from the same bf16 logits with the same
// code, so the eight copies are identical.
// KSPLIT (MPK_ROUTER_KSPLIT): the logits arrive as 16 per-slice partials per
// expert in the XCDs' exchange rows (ks_all = XCD 0's block), unscaled; this
// XCD's 16 sums of squares give 1/rms. The release also waits for this XCD's
// 16 normed-slice flags.
template <typename T, int NUM_EXPERTS, int K, bool KSPLIT = false>
__device__ __attribute__((noinline)) void
    topk_ll_noinline(int *rll,
                     int xcd,
                     unsigned epoch,
                     void *bias_ptr,
                     int num_active_tokens,
                     bool renormalize,
                     float routed_scaling_factor,
                     int num_shared_experts,
                     int const *ks_all = nullptr,
                     float ks_hidden = 0.0f) {
  static_assert(NUM_EXPERTS == MPK_NT, "one logit word per thread");
  static_assert(NUM_EXPERTS * 8 <= MPK_RLL_WORD_LINES * 64,
                "the logit words overrun their lines");
  static_assert(NUM_EXPERTS + 16 <= MPK_RLL_IDS_LINES * 16,
                "the routing copies overrun their lines");
  int const tid = threadIdx.x;
  unsigned long long const *const words =
      reinterpret_cast<unsigned long long const *>(rll);
  __shared__ unsigned short s_logits[NUM_EXPERTS];
  unsigned long long w;
  if constexpr (KSPLIT && MPK_ROUTE_LL_ON) {
    // GLM_DENSE_FUSED: routed_scaling_factor 0 marks a dense layer whose MLP
    // is K + num_shared always-active virtual experts, so there is nothing to
    // select: slot k is expert k at weight 1. The router tiles' (zero)
    // partials go unread. Same ordering as below: the words are written by
    // the wave that saw the normed flags.
    if (routed_scaling_factor == 0.0f) {
      if (tid < 64) {
        unsigned const *const ks_flag = reinterpret_cast<unsigned const *>(
            ks_all + xcd * MPK_XSPLIT_XCD_INTS);
        while (true) {
          unsigned seen = epoch;
          if (tid < 16) {
            asm volatile("global_load_dword %0, %1, off nt\n"
                         "s_waitcnt vmcnt(0)"
                         : "=v"(seen)
                         : "v"(ks_flag + tid)
                         : "memory");
          }
          if (__ballot(seen >= epoch) == ~0ull) {
            break;
          }
          __builtin_amdgcn_s_sleep(1);
        }
        unsigned long long *const ll_dense =
            reinterpret_cast<unsigned long long *>(
                rll + MPK_RLL_WORD_LINES * 16 + xcd * MPK_RLL_XCD_INTS + 32 +
                MPK_RLL_IDS_LINES * 16);
        if (tid < K + num_shared_experts) {
          unsigned long long const hi = (unsigned long long)epoch << 32;
          ll_dense[tid] = hi | (unsigned)tid;
          ll_dense[16 + tid] = hi | __float_as_uint(1.0f);
        }
      }
      return;
    }
  }
  if constexpr (KSPLIT) {
    static_assert(NUM_EXPERTS == 256, "32 experts per XCD, 16 slices");
    extern __shared__ char _topk_dyn_smem[];
    T *const s_bias = reinterpret_cast<T *>(_topk_dyn_smem);
    float *const s_ssq = reinterpret_cast<float *>(_topk_dyn_smem + 512);
    T const b = static_cast<T const *>(bias_ptr)[tid];
    unsigned long long const *const blk =
        reinterpret_cast<unsigned long long const *>(
            ks_all + (tid >> 5) * MPK_XSPLIT_XCD_INTS +
            MPK_XSPLIT_FLAG_LINES * 16) +
        (tid & 31);
    // Each round re-issues every word still stale, all in flight at once:
    // re-polling them one at a time costs a round trip per late tile. The
    // first 16 threads carry this XCD's sums of squares in the same rounds.
    unsigned long long const *const sp =
        reinterpret_cast<unsigned long long const *>(
            ks_all + xcd * MPK_XSPLIT_XCD_INTS + MPK_XSPLIT_FLAG_LINES * 16) +
        (tid & 15) * 64 + 32;
    bool const has_sw = tid < 16;
    unsigned long long pw[16];
    unsigned long long sw = (unsigned long long)epoch << 32;
#pragma unroll
    for (int t = 0; t < 16; t++) {
      asm volatile("global_load_dwordx2 %0, %1, off sc0 sc1"
                   : "=v"(pw[t])
                   : "v"(blk + t * 64)
                   : "memory");
    }
    if (has_sw) {
      asm volatile("global_load_dwordx2 %0, %1, off sc0 sc1"
                   : "=v"(sw)
                   : "v"(sp)
                   : "memory");
    }
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    while (true) {
      bool ok = (unsigned)(sw >> 32) == epoch;
#pragma unroll
      for (int t = 0; t < 16; t++) {
        ok = ok && (unsigned)(pw[t] >> 32) == epoch;
      }
      if (ok) {
        break;
      }
      __builtin_amdgcn_s_sleep(1);
#pragma unroll
      for (int t = 0; t < 16; t++) {
        if ((unsigned)(pw[t] >> 32) != epoch) {
          asm volatile("global_load_dwordx2 %0, %1, off sc0 sc1"
                       : "+v"(pw[t])
                       : "v"(blk + t * 64)
                       : "memory");
        }
      }
      if ((unsigned)(sw >> 32) != epoch) {
        asm volatile("global_load_dwordx2 %0, %1, off sc0 sc1"
                     : "+v"(sw)
                     : "v"(sp)
                     : "memory");
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }
    if (tid == 0) {
      mpk_stage_stamp(40);
    }
    if (has_sw) {
      s_ssq[tid] = __uint_as_float((unsigned)sw);
    }
    s_bias[tid] = b;
    bias_ptr = s_bias;
    __syncthreads();
    float tot = 0.0f;
#pragma unroll
    for (int t = 0; t < 16; t++) {
      tot += s_ssq[t];
    }
    float const irms = rsqrtf(tot / ks_hidden + 1e-5f);
    float s = 0.0f;
#pragma unroll
    for (int t = 0; t < 16; t++) {
      s += __uint_as_float((unsigned)pw[t]);
    }
    __hip_bfloat16 const lb = __float2bfloat16(s * irms);
    unsigned short u;
    __builtin_memcpy(&u, &lb, 2);
    w = u;
  } else {
#if MPK_TOPK_LL_PREBIAS
  // MPK_TOPK_LL_PREBIAS: the correction bias does not depend on the logits,
  // so its load rides the first poll round and the selection reads it out of
  // LDS; and each thread spins on its own word, so the last word to land is
  // seen one load after it lands rather than one block-wide round after.
  // Dynamic LDS, at its base: static LDS is at its 8 KB budget, and nothing
  // of this worker's is in flight there between the o_proj prefetch window
  // (retired at Phase 8) and the MoE's, which starts after routing.
  extern __shared__ char _topk_dyn_smem[];
  T *const s_bias = reinterpret_cast<T *>(_topk_dyn_smem);
  T const b = static_cast<T const *>(bias_ptr)[tid];
  while ((unsigned)((w = ld_sys_u64(const_cast<unsigned long long *>(
                         words + tid))) >>
                    32) != epoch) {
    __builtin_amdgcn_s_sleep(1);
  }
  s_bias[tid] = b;
  bias_ptr = s_bias;
#else
  while (true) {
    w = ld_sys_u64(const_cast<unsigned long long *>(words + tid));
    if (__syncthreads_and((unsigned)(w >> 32) == epoch)) {
      break;
    }
    __builtin_amdgcn_s_sleep(1);
  }
#endif
  }
  s_logits[tid] = static_cast<unsigned short>(w & 0xFFFFu);
  __syncthreads();
  // Stage stamps 41/42/43: logits ready, selection done, normed flags seen.
  if (tid == 0) {
    mpk_stage_stamp(41);
  }
  int *const xb = rll + MPK_RLL_WORD_LINES * 16 + xcd * MPK_RLL_XCD_INTS;
  // MPK_ROUTE_LL: the routing goes out as epoch-tagged words that the MoE
  // tiles validate themselves, in the routing_indices lines (which the
  // one-row TP decode never reads). Words written after the normed flags are
  // seen, so a tile that sees its word also sees the normed row in L2.
  constexpr bool ROUTE_LL = KSPLIT && MPK_ROUTE_LL_ON;
  unsigned long long *const ll_route =
      ROUTE_LL ? reinterpret_cast<unsigned long long *>(
                     xb + 32 + MPK_RLL_IDS_LINES * 16)
               : nullptr;
  if constexpr (ROUTE_LL) {
    if (tid < 64) {
      unsigned const *const ks_flag =
          reinterpret_cast<unsigned const *>(ks_all + xcd * MPK_XSPLIT_XCD_INTS);
      while (true) {
        unsigned seen = epoch;
        if (tid < 16) {
          asm volatile("global_load_dword %0, %1, off nt\n"
                       "s_waitcnt vmcnt(0)"
                       : "=v"(seen)
                       : "v"(ks_flag + tid)
                       : "memory");
        }
        if (__ballot(seen >= epoch) == ~0ull) {
          break;
        }
        __builtin_amdgcn_s_sleep(1);
      }
    }
    __syncthreads();
  }
  // Trace stamp 63: normed flags seen; S63 -> S42 is the selection alone.
  if (tid == 0) {
    mpk_stage_stamp(63);
  }
  // MPK_TOPK_RANK's scratch: past s_bias [0, 512) and s_ssq [512, 1536).
  extern __shared__ char _topk_rank_smem[];
  topk_sigmoid_bias_mi300_task_impl<T,
                                    /*VPT=*/8,
                                    NUM_EXPERTS,
                                    /*WARPS_PER_CTA=*/4,
                                    /*BYTES_PER_LDG=*/16,
                                    /*K_STATIC=*/K,
                                    /*ROUTING_ROW_STRIDE=*/1,
                                    // The logits are this call's own LDS copy.
                                    // MEASURED NEUTRAL here too (2026-10-08,
                                    // 6.510 6.488 -> 6.487 6.492 ms).
                                    /*SKIP_CLEARS=*/MPK_TOPK_SKIP_CLEARS != 0,
                                    /*L2_STORES=*/true>(
      s_logits,
      bias_ptr,
      /*topk_weight=*/xb + 16,
      num_active_tokens,
      K,
      /*routing_indices=*/ROUTE_LL ? nullptr
                                   : xb + 32 + MPK_RLL_IDS_LINES * 16,
      /*active_expert_ids=*/xb + 32,
      0,
      NUM_EXPERTS,
      renormalize,
      routed_scaling_factor,
      num_shared_experts,
      ll_route,
      epoch,
      MPK_TOPK_RANK ? reinterpret_cast<unsigned *>(_topk_rank_smem + 2048)
                    : nullptr);
  if (tid == 0) {
    mpk_stage_stamp(42);
  }
  if constexpr (ROUTE_LL) {
    if (tid == 0) {
      mpk_stage_stamp(35);
    }
    return;
  }
  // Every wave wrote some of the copy (the zero fill is block-wide), so each
  // drains its own stores before the barrier the release sits behind.
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  __syncthreads();
  if constexpr (KSPLIT) {
    // The MoE reads the normed row this XCD's router tiles wrote into its L2.
    if (tid < 64) {
      unsigned const *const ks_flag =
          reinterpret_cast<unsigned const *>(ks_all + xcd * MPK_XSPLIT_XCD_INTS);
      while (true) {
        unsigned seen = epoch;
        if (tid < 16) {
          asm volatile("global_load_dword %0, %1, off nt\n"
                       "s_waitcnt vmcnt(0)"
                       : "=v"(seen)
                       : "v"(ks_flag + tid)
                       : "memory");
        }
        if (__ballot(seen >= epoch) == ~0ull) {
          break;
        }
        __builtin_amdgcn_s_sleep(1);
      }
    }
    __syncthreads();
    if (tid == 0) {
      mpk_stage_stamp(43);
    }
#if MPK_TOPK_PAD_NS > 0
    // PRICING PROBE: a fixed delay ahead of the routing release, every rank,
    // every layer. The wall's slope against it says whether this tail is on
    // the critical path.
    if (tid == 0) {
      unsigned long long const t0 = __builtin_amdgcn_s_memrealtime();
      while ((__builtin_amdgcn_s_memrealtime() - t0) * 10ull <
             (unsigned long long)MPK_TOPK_PAD_NS) {
      }
    }
    __syncthreads();
#endif
  }
  if (tid == 0) {
    asm volatile("global_store_dword %0, %1, off\n"
                 "s_waitcnt vmcnt(0)"
                 :
                 : "v"(xb), "v"(epoch)
                 : "memory");
    mpk_stage_stamp(35);
  }
}
#endif // MPK_ROUTER_LL
} // namespace gang_rmsnorm_topk_detail

// Croc-style fused RMSNorm + Gate GEMV + TopK.
//
// 128 workers (16/XCD), one expert per worker. Each worker:
//   1. Computes RMSNorm (irms) from hidden state — redundant, same result
//   2. Fused GEMV: dp += gate_w[expert, i] * (hidden[i] * irms * gamma[i])
//      Also writes norm_output (for downstream MoE FP8 quant) as side-effect
//   3. Writes 1 logit via write-through store
//   4. atomicAdd barrier; last worker runs TopK softmax
//
// SIGMOID_BIAS switches the tail to the `noaux_tc` router (GLM / DeepSeek):
// sigmoid scores with an additive selection bias instead of a softmax. Steps
// 1-3 are identical; only the treatment of `bias_ptr` and the tail differ.
template <typename T,
          int BATCH_SIZE,
          int REDUCTION_SIZE,
          int ACTUAL_HIDDEN_DIM,
          int NUM_EXPERTS,
          int K,
          bool SIGMOID_BIAS = false,
          // When true the caller has just produced norm_input_ptr with a GEMM
          // and has already counted its arrival at a cross-XCD barrier; this
          // kernel owns the *wait*. It takes it over so that the gamma and
          // gate-weight loads -- which do not depend on the barrier at all --
          // can be issued before the poll and be in flight while it spins,
          // instead of paying their full latency after it. Without this a
          // fused caller is strictly slower than the dispatch it replaced:
          // the barrier serialises where a task boundary would have let the
          // next task's loads start. Lifted from
          // gang_linear_mxfp4_res_bias_rmsnorm_topk_mi300.cuh:716.
          bool OPROJ_BARRIER = false,
          // Experts per call. 1 is the shape this was written for -- one
          // worker per expert -- and is bit-identical to the code before this
          // parameter existed. Above 1 the call walks the row once and emits
          // EXPERTS_PER_TILE logits from it, which is the only lever on the
          // *round count*: GLM-5's 256 experts are 32 tiles per XCD against 29
          // workers, so at 1 the makespan is two calls where the mean is 1.10,
          // and the second call re-pays the barrier spin, the two block-wide
          // reductions and the arrival atomic to add one dot product. The row
          // read, the RMSNorm and the normed write are all shared across a
          // tile's experts; only the gate row and the accumulator are not.
          int EXPERTS_PER_TILE = 1,
          // MPK_OPROJ_RP: the row is not in norm_input_ptr but split over
          // SUM_SLOTS f32 rank partials at `sum_slots`, REDUCTION_SIZE apart.
          // Step 1 sums them in slot order and rounds to bf16 once -- the same
          // row on every rank -- keeps it in registers for Step 2, and writes
          // this tile's 1/total_gang_tiles share of it to `sum_hidden_out`.
          int SUM_SLOTS = 0,
          // The slots hold f32 partials; false is bf16 (MPK_OPROJ_RP_BF16).
          bool SUM_F32 = true,
          // MPK_ROUTER_STREAM: the barrier above releases on this rank's own
          // slot only; each peer slot is added in as its own signal lands, in
          // slot order, so the row is bit-identical to the unstreamed sum.
          bool SUM_STREAM = false,
          // MPK_OPROJ_LL: the slots are (epoch << 32 | f32) words, each one
          // its own signal, so neither the per-peer signals nor the o_proj
          // release are waited on.
          bool SUM_LL = false>
#if MPK_ROUTER_INLINE
__device__ __attribute__((always_inline)) void
#else
__device__ __attribute__((noinline)) void
#endif
gang_rmsnorm_linear_bias_topk_kernel(
    void const *norm_input_ptr,  // input_ptrs[0]: [batch, REDUCTION_SIZE]
    void const *norm_weight_ptr, // input_ptrs[1]: [REDUCTION_SIZE]
    void *norm_output_ptr, // input_ptrs[2]: [batch, REDUCTION_SIZE] scratch
    void const
        *gate_weight_ptr,     // input_ptrs[3]: [chunk_N, REDUCTION_SIZE] bf16
    void const *bias_ptr,     // input_ptrs[4]: [chunk_N] bf16 (XCD-partitioned;
                              // the whole [NUM_EXPERTS] under SIGMOID_BIAS)
    void *logits_scratch_ptr, // input_ptrs[5]: XCD-partitioned [batch, chunk_N]
    void *gang_counter_ptr,   // input_ptrs[6]: [1] int32 atomic counter
    void *topk_weight_ptr,    // output_ptrs[0]: [batch, K] float
    void *routing_indices_ptr,   // output_ptrs[1]: [NUM_EXPERTS, batch] int32
    void *active_expert_ids_ptr, // output_ptrs[2]: [NUM_EXPERTS+1] int32
    int num_active_tokens,
    int tile_n,
    int o_stride,
    int m_tiles,
    int n_tiles,
    int wgm,
    int tile_idx,
    int total_gang_tiles,
    // SIGMOID_BIAS only; ignored by the softmax tail.
    bool renormalize = true,
    float routed_scaling_factor = 1.0f,
    int num_shared_experts = 0,
    // OPROJ_BARRIER only. Mechanism C: hier[x * 16] is XCD x's release flag,
    // and the caller has already done the arrival and the last-arriver
    // fan-out. `oproj_release_expected` must be read by the caller *before*
    // it produces its output, not here -- see the fused wrapper.
    void *oproj_hier_barrier_ptr = nullptr,
    int oproj_xcd_id = 0,
    int oproj_release_expected = 0,
    // SIGMOID_BIAS only. Non-null when a fused caller has MoE workers parked
    // behind this router; the TopK tail fans out the release. Layout is
    // gpt-oss's: [0] epoch, [(1 + x) * 16] XCD x's flag.
    int *routing_ready_ptr = nullptr,
    // SIGMOID_BIAS only. The epoch value the TopK tail publishes into
    // routing_ready. Pass the same number the caller's consumers wait on;
    // < 0 falls back to incrementing whatever is there.
    int routing_epoch_hint = -1,
    // Optional LDS scratch letting one worker amortise Step 1 over several
    // experts. Every worker re-norms the whole row to get a scalar that does
    // not depend on which expert it is doing, so a worker holding two experts
    // computes the same `irms` twice, bit for bit. Point this at a __shared__
    // float, set it negative once per layer, and the second call skips Step 1
    // entirely. Null keeps the old behaviour.
    float *irms_cache = nullptr,
    // ── router fold ────────────────────────────────────────────────────────
    // Non-null means the caller's o_proj epilogue already accumulated both
    // contractions this kernel would otherwise run -- sum_i h_i*gamma_i*W[e,i]
    // for every expert, and sum_i h_i^2 -- each over the rank's own column
    // slice, and reduced across the ranks on the o_proj all-gather's
    // rendezvous. Steps 1 and 2 then collapse to a FOLDED_RANKS-long sum and
    // a scale by irms, and the only thing left that has to walk the row is the
    // normed write, which the MoE needs and the TopK does not.
    //
    // Layout: FOLDED_RANKS lines of FOLDED_STRIDE floats, line p written by
    // PE p. Element FOLDED_EXPERT_BASE + e of a line is expert e of this XCD's
    // range; element NUM_EXPERTS is the sum-of-squares.
    float const *folded_lines = nullptr,
    int folded_ranks = 0,
    int folded_stride = 0,
    int folded_expert_base = 0,
    // SUM_SLOTS > 0 only; see the template parameter. sum_tile is this
    // tile's index among the total_gang_tiles.
    void const *sum_slots = nullptr,
    void *sum_hidden_out = nullptr,
    int sum_tile = 0,
    // SUM_STREAM only: this rank's index, and the u64 signal line slot p's
    // writer bumps to `stream_expected` once slot p is whole, at
    // stream_sig + p * stream_sig_stride.
    int stream_my_pe = 0,
    unsigned long long const *stream_sig = nullptr,
    int stream_sig_stride = 0,
    unsigned long long stream_expected = 0,
    // MPK_ROUTER_XSPLIT: this XCD's exchange block (flag line, then the row).
    int *xsplit = nullptr,
    // MPK_ROUTER_LL: the logit words, then the eight per-XCD routing copies.
    int *rll = nullptr) {

  using bf16 = __hip_bfloat16;
  bf16 const *__restrict__ d_hidden = static_cast<bf16 const *>(norm_input_ptr);
  bf16 const *__restrict__ d_gamma = static_cast<bf16 const *>(norm_weight_ptr);
  bf16 *__restrict__ d_normed = static_cast<bf16 *>(norm_output_ptr);
  bf16 const *__restrict__ d_gate_w =
      static_cast<bf16 const *>(gate_weight_ptr);
  bf16 const *__restrict__ d_bias = static_cast<bf16 const *>(bias_ptr);
  bf16 *__restrict__ d_logits = static_cast<bf16 *>(logits_scratch_ptr);

  int const tid = threadIdx.x;
  int const lane = tid & 63;
  int const wave = tid >> 6;
  constexpr int NUM_WAVES = 4; // 256 threads / 64 lanes

  // This tile's first expert, in the caller's index space -- XCD-local for the
  // fused router, where gate_weight_ptr and logits_scratch_ptr are already
  // this XCD's slice. At EXPERTS_PER_TILE 1 this is `tile_idx` and every
  // expression below reduces to what it was.
  int const first_expert = tile_idx * EXPERTS_PER_TILE;

  // ═══ Step 0: prefetch across the O-proj barrier, then wait ═══
  // gamma and this worker's gate row are data-independent of the GEMM the
  // caller just ran, so they go out before the poll and land while it spins.
  // `nt` keeps them out of L2, which matters: the buffer_inv that closes the
  // barrier would otherwise throw them away again.
  typedef int __attribute__((ext_vector_type(2))) i32x2_pf_t;
  constexpr int H4_PF = REDUCTION_SIZE >> 2;
  constexpr int MAX_ITERS_PF = OPROJ_BARRIER ? ((H4_PF + 255) / 256) : 1;
  i32x2_pf_t g_pf[MAX_ITERS_PF];
  i32x2_pf_t w_pf[EXPERTS_PER_TILE][MAX_ITERS_PF];
  // Declared here rather than at Step 1 because it also decides WHICH
  // prefetch is issued. The unfolded path wants gamma and every gate row of
  // the tile, whole, because every worker walks the whole row for its dot
  // product. The folded path reads no gate weight at all -- that is the 24 KB
  // per worker, 3.1 MB per rank per layer, the fold exists to delete -- and
  // walks no whole row either: all that is left of Step 2 is this tile's
  // slice of the normed write, so it prefetches exactly that slice of gamma
  // and of the hidden row and nothing else.
  bool const folded = (folded_lines != nullptr);
  // The fold's share-out of the normed write. Only reachable from the fused
  // router, where this kernel's expert range is already one XCD's chunk of
  // NUM_EXPERTS, so the tile count is that chunk over the tile width.
  constexpr int FOLD_TILES = (NUM_EXPERTS / 8) / EXPERTS_PER_TILE;
  constexpr int FOLD_COLS = REDUCTION_SIZE / (FOLD_TILES > 0 ? FOLD_TILES : 1);
  constexpr int FOLD_PF = ((FOLD_COLS / 4) + 255) / 256;
  static_assert(!OPROJ_BARRIER || FOLD_TILES < 1 ||
                    REDUCTION_SIZE % (4 * FOLD_TILES) == 0,
                "the fold's normed write splits the row into whole "
                "dwordx2-aligned per-tile slices");
  i32x2_pf_t fg_pf[FOLD_PF]; // gamma, this tile's slice
  int const fold_col0 = tile_idx * FOLD_COLS;
  // MPK_ROUTER_KSPLIT: this tile's 384 columns of all 32 of the XCD's gate
  // rows (thread t: row t / 8, 8-column chunks t % 8 + 8j), and gamma's
  // same 384 columns on the slice-summing threads.
  constexpr bool KSPLIT = gang_rmsnorm_topk_detail::router_ksplit_on<
      BATCH_SIZE, REDUCTION_SIZE, NUM_EXPERTS, SIGMOID_BIAS, OPROJ_BARRIER,
      EXPERTS_PER_TILE, SUM_LL>();
  constexpr int KS_SLICE = REDUCTION_SIZE / 16;
  constexpr int KS_CH = KS_SLICE / 64;
  typedef int __attribute__((ext_vector_type(4))) i32x4_ks_t;
  i32x4_ks_t ks_w[KSPLIT ? KS_CH : 1];
  i32x2_pf_t ks_g = {0, 0};
  if constexpr (KSPLIT) {
    char const *const ks_wb =
        (char const *)gate_weight_ptr +
        ((size_t)(tid >> 3) * REDUCTION_SIZE + (size_t)tile_idx * KS_SLICE +
         (tid & 7) * 8) * 2;
#pragma unroll
    for (int j = 0; j < KS_CH; j++) {
      asm volatile("global_load_dwordx4 %0, %1, off sc0 nt"
                   : "=v"(ks_w[j])
                   : "v"(ks_wb + j * 128)
                   : "memory");
    }
    if (tid < KS_SLICE / 4) {
      asm volatile("global_load_dwordx2 %0, %1, off sc0 nt"
                   : "=v"(ks_g)
                   : "v"((char const *)norm_weight_ptr +
                         ((size_t)tile_idx * KS_SLICE + tid * 4) * 2)
                   : "memory");
    }
  }
  if constexpr (OPROJ_BARRIER) {
    if (!KSPLIT && folded) {
      // gamma only. The hidden row is precisely what the barrier below
      // guards -- the o_proj all-gather has not finished writing it yet --
      // so it is read after the spin like it always was, just 384 columns at
      // a time on sixteen tiles instead of 6144 on one.
      char const *fg_base =
          (char const *)norm_weight_ptr + (size_t)fold_col0 * 2;
#pragma unroll
      for (int iter = 0; iter < FOLD_PF; iter++) {
        int i_cur = tid + iter * 256;
        if (i_cur >= (FOLD_COLS >> 2)) {
          break;
        }
        // Same `sc0 nt` as the unfolded prefetch, and for the same reason:
        // the buffer_inv that closes the barrier would throw an L2-cached
        // line away again before Step 2 could use it.
        asm volatile("global_load_dwordx2 %0, %1, off sc0 nt"
                     : "=v"(fg_pf[iter])
                     : "v"(fg_base + i_cur * 8)
                     : "memory");
      }
    }
    if (!KSPLIT && !folded) {
    char const *g_base_pf = (char const *)norm_weight_ptr;
    char const *w_base_pf = (char const *)gate_weight_ptr +
                            (int64_t)first_expert * REDUCTION_SIZE * 2;
#pragma unroll
    for (int iter = 0; iter < MAX_ITERS_PF; iter++) {
      int i_cur = tid + iter * 256;
      if (i_cur >= H4_PF) {
        break;
      }
      int byte_off = i_cur * 8;
      asm volatile("global_load_dwordx2 %0, %1, off sc0 nt"
                   : "=v"(g_pf[iter])
                   : "v"(g_base_pf + byte_off)
                   : "memory");
      // Every expert in the tile, so all EXPERTS_PER_TILE gate rows are in
      // flight across the barrier rather than one across it and the rest
      // behind it. Costs EXPERTS_PER_TILE * MAX_ITERS_PF * 2 live VGPRs; see
      // the note on the asm barrier below, which is what keeps the *address*
      // arithmetic from costing as many again.
#pragma unroll
      for (int e = 0; e < EXPERTS_PER_TILE; e++) {
        asm volatile("global_load_dwordx2 %0, %1, off sc0 nt"
                     : "=v"(w_pf[e][iter])
                     : "v"(w_base_pf + (int64_t)e * REDUCTION_SIZE * 2 +
                           byte_off)
                     : "memory");
      }
      }
    }
    int *hier = static_cast<int *>(oproj_hier_barrier_ptr);
    // Self-heal, see MPK_FL_REPUBLISH_SPINS in
    // gang_mla_full_layer_fused_mi300.cuh. The release is eight independent
    // write-through stores from one elected thread, issued once, never
    // retried; lose the one addressed to this XCD and every worker on it
    // spins here for the rest of the run. The arrival count is not plumbed
    // down to this kernel, but the other seven flags are just as conclusive:
    // all eight are written by the same thread in the same loop, so any peer
    // at or past the epoch proves the release fired and this line is simply
    // short. Republishing it is idempotent -- same monotonic absolute value.
    {
#ifdef MPK_ENABLE_SUBPHASE_TIMING
      unsigned long long _sp_spin0 = __builtin_amdgcn_s_memrealtime();
#endif
      int _spins = 0;
      // #135. The caller stamps phase 73 before this call and 74 after it, so
      // a worker wedged anywhere inside the router shows up as "73" and
      // nothing more. Give this spin its own barrier id (770) and the region
      // past it its own marks, so the next capture says *which* of the two it
      // is instead of leaving the whole kernel as one bucket.
      MPK_WS_WAIT_BEGIN(770, oproj_release_expected);
      int _obs770;
      while (!SUM_LL && (_obs770 = ld_nt_s32(&hier[oproj_xcd_id * 16])) <
                            oproj_release_expected) {
        MPK_WS_WAIT_TICK(_obs770, _spins);
        if ((++_spins & (MPK_FL_REPUBLISH_SPINS - 1)) == 0) {
          // #135. The old aux reported hier[0], which answers nothing when
          // the question is *which* of the eight flags is short -- a capture
          // showed a0=222 while workers on XCD 0 had demonstrably already
          // read 223 and moved on, i.e. the value was simply stale by the
          // time it was sampled. Report instead the two values that fork the
          // diagnosis, computed in the same pass the heal already makes:
          //   a0 = the barrier's own arrival counter, at [8*HIER_STRIDE].
          //        Not a multiple of total_barrier_arrivals => an arrival was
          //        lost in some earlier layer, the modular election never
          //        fired for this epoch, and NO flag was ever written. That
          //        is invisible to atomicMax publication, which only makes a
          //        write that happens order-independent.
          //   a1 = low 8 bits, the mask of XCDs whose flag is at or past the
          //        epoch; bits 8..11, this worker's oproj_xcd_id.
          //        mask == 0  -> the election fired and published stale, or
          //                      never fired (read a0).
          //        mask != 0  -> a peer holds it, so the heal below ran; if
          //                      we are still spinning the store did not
          //                      stick, which after atomicMax would mean the
          //                      flag line itself is being written by
          //                      something else.
          int _mask = 0;
          for (int x = 0; x < 8; x++) {
            if (ld_nt_s32(&hier[x * 16]) >= oproj_release_expected) {
              _mask |= 1 << x;
            }
          }
          MPK_WS_WAIT_AUX(ld_nt_s32(&hier[8 * 16]),
                          _mask | (oproj_xcd_id << 8), 0, 0);
          if (_mask) {
            st_flag_u32((void *)&hier[oproj_xcd_id * 16],
                        (unsigned)oproj_release_expected);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          }
        }
        __builtin_amdgcn_s_sleep(1);
      }
      // #135, capture 4. The p135d capture showed three workers whose `spins`
      // was FROZEN at 57344 across four consecutive host dumps while every
      // other worker's grew by ~20M -- i.e. those waves were not executing the
      // loop above at all. `spins` positive only proves the last write to the
      // slot was a wait tick, and the aux slot is sticky, so "still polling"
      // and "left the poll and hung immediately after" are indistinguishable
      // from that dump. This mark separates them: reaching it makes the slot
      // NEGATIVE. If the next capture still shows a positive frozen spins, the
      // wave is genuinely inside the poll; if it shows -779..., the poll was
      // satisfied and the hang is the buffer_inv / s_waitcnt vmcnt(0) below,
      // which drains the Step-0 prefetch -- and that is VMEM state carried in
      // from the previous layer's MoE k-loop, which is exactly what
      // MPK_MOE_PF_GROUPS_W13 changes.
      MPK_WS_MARK(779, oproj_xcd_id);
#ifdef MPK_ENABLE_SUBPHASE_TIMING
      // The caller charges this whole kernel to SP3[2] "Router", so without
      // this the spin and the router's actual work are one number. Guarded
      // exactly as SP3[2] is -- every worker's thread 0, both calls when a
      // worker carries two experts -- so the two are directly comparable and
      // SP3[2] minus this is the router's real compute.
      if (tid == 0 && g_subphase_active) {
        atomicAdd(&g_subphase_ns[6][0],
                  (__builtin_amdgcn_s_memrealtime() - _sp_spin0) * 10);
        atomicAdd(&g_subphase_cnt[6], 1ULL);
      }
#endif
    }
    // Plain buffer_inv, no sc1: drop the vL1 so d_hidden is re-read, but
    // leave this XCD's own L2 lines alone.
    asm volatile("buffer_inv" ::: "memory");
    // Drain the prefetch. nt loads bypass L2 and are unaffected by the inv.
    // Under SUM_LL not until Step 2 uses it: vmcnt retires in issue order,
    // so a wave with XGMI pushes in flight would sit here on their acks.
    if constexpr (!SUM_LL) {
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }
    MPK_WS_MARK(771, tile_idx);
    // Stage stamp 33: o_proj all-gather observed (router workers). S31 -> S33
    // is the o_proj rendezvous as the router sees it, S33 -> S34 its work.
    if (tid == 0) {
      mpk_stage_stamp(33);
    }
  }

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  // SP6[0] is the o_proj barrier spin above. [1]..[4] split what is left of
  // this kernel, so SP3[2] "Router" decomposes without a second run:
  //   [1] Step 1 RMSNorm   [2] Step 2 gate dot + normed write
  //   [3] Step 3 logit store + arrival atomic   [4] Step 4 TopK tail
  // Every one is guarded `tid == 0 && g_subphase_active`, exactly as SP6[0]
  // and SP3[2] are, so all six are directly comparable.
  unsigned long long _rt_t0 = __builtin_amdgcn_s_memrealtime();
#endif

  // ═══ Step 1: RMSNorm — compute irms from hidden state ═══
  // All 256 threads collaborate. Vectorized 4-wide bf16 loads.
  //
  // Skipped outright when a previous call on this block already computed it
  // for the same row. The cache is LDS written by tid 0 and read by all 256,
  // which is safe without a fence of its own: the producing call ends in
  // __syncthreads (Step 3) and this read is the next call's first LDS access.
  // red[] carries, in order, the per-(row, wave) sum of squares and then the
  // per-(row, expert, wave) gate partial. The second is the wider of the two.
  // At BATCH_SIZE 1 this is the 16 floats it always was.
  constexpr int RED_SLOTS = BATCH_SIZE * EXPERTS_PER_TILE * NUM_WAVES;
  __shared__ float red[RED_SLOTS > 16 ? RED_SLOTS : 16];
  // Separate from red[] rather than folded into it: the dp reduction below
  // overwrites red while irms is still needed, and at BATCH_SIZE > 1 the
  // read-then-overwrite ordering that made a single red[0] safe stops being
  // obvious. BATCH_SIZE floats of LDS is not a number worth being clever for.
  __shared__ float s_irms[BATCH_SIZE];
  // SUM_SLOTS keeps the summed row for Step 2, so it cannot skip Step 1.
  bool const irms_cached = (SUM_SLOTS == 0) && (irms_cache != nullptr) &&
                           (irms_cache[0] > 0.0f);
  float ssq[BATCH_SIZE];
#pragma unroll
  for (int m = 0; m < BATCH_SIZE; m++) {
    ssq[m] = 0.0f;
  }
  constexpr int SUM_ITERS = (SUM_SLOTS > 0) ? MAX_ITERS_PF : 1;
  float hs[SUM_ITERS][4];
  if constexpr (SUM_SLOTS > 0 && SUM_STREAM) {
    static_assert(SUM_F32 && OPROJ_BARRIER && BATCH_SIZE == 1 &&
                      REDUCTION_SIZE % (4 * 256) == 0,
                  "the streamed slot sum takes f32 partials and one row");
    typedef float sum_f4_t __attribute__((ext_vector_type(4)));
    float const *const sl = static_cast<float const *>(sum_slots);
    unsigned short *const hout = static_cast<unsigned short *>(sum_hidden_out);
    int const q_per = H4_PF / total_gang_tiles;
    if (q_per * total_gang_tiles != H4_PF) {
      __builtin_trap();
    }
    int const q_lo = sum_tile * q_per;
    float hacc[SUM_ITERS][4];
#pragma unroll
    for (int iter = 0; iter < SUM_ITERS; iter++) {
#pragma unroll
      for (int j = 0; j < 4; j++) {
        hacc[iter][j] = 0.0f;
      }
    }
    // Slot order, not arrival order: the row has to round the same on every
    // rank, and the same as the unstreamed sum (0 + s0 + s1 + ...), so the
    // output is bit-identical to it. This rank's own slot is covered by the
    // barrier above. A round takes slot p and, if it is already signalled,
    // p + 1, issuing both slots' loads before accumulating either.
    __shared__ int s_stream_two;
    auto slot_ready = [&](int s) -> bool {
      return s == stream_my_pe ||
             ld_sys_u64(const_cast<unsigned long long *>(
                 stream_sig + (size_t)s * stream_sig_stride)) >=
                 stream_expected;
    };
#if MPK_ROUTER_XSPLIT
    // The XCD's xs_n router tiles split the row xs_n ways: each sums its
    // slice over every slot and the slices meet in an XCD-private row in L2,
    // so a peer's line is read cold once per XCD rather than once per tile,
    // and the walk is one round trip instead of a poll-probe-load chain per
    // slot pair. Per element the slot order, the bf16 rounding and the
    // per-thread ssq order are the walk's below, so the result is identical.
    int const xs_n = total_gang_tiles / 8;
    int const xs_f4 = (xs_n > 0) ? H4_PF / xs_n : 0;
    int const xs_x = (xs_n > 0) ? sum_tile / xs_n : -1;
    int const xs_t = (xs_n > 0) ? sum_tile % xs_n : 0;
    if (xsplit == nullptr || xs_n * 8 != total_gang_tiles || xs_n > 64 ||
        xs_f4 * xs_n != H4_PF || xs_f4 > 256 ||
        xs_x != gang_rmsnorm_topk_detail::get_xcd_id()) {
      __builtin_trap();
    }
    static_assert(REDUCTION_SIZE * 2 <= MPK_XSPLIT_ROW_LINES * 64,
                  "the summed bf16 row overruns its exchange block");
    unsigned *const xs_flag = reinterpret_cast<unsigned *>(xsplit);
    unsigned short *const xs_row =
        reinterpret_cast<unsigned short *>(xsplit + MPK_XSPLIT_FLAG_LINES * 16);
    unsigned const xs_epoch = static_cast<unsigned>(stream_expected);
    MPK_WS_MARK(790, xs_t);
    MPK_WS_WAIT_AUX(gang_rmsnorm_topk_detail::get_xcd_id(), xs_x, xs_t,
                    (int)xs_epoch);
    if constexpr (!SUM_LL) {
      if (tid < SUM_SLOTS) {
        while (!slot_ready(tid)) {
          __builtin_amdgcn_s_sleep(1);
        }
      }
      __syncthreads();
    }
    // Four words per slot, two 16-byte system-scope loads each: the peers'
    // words arrive over XGMI, so nothing on this side may cache them.
    typedef unsigned int ll_u4_t __attribute__((ext_vector_type(4)));
    ll_u4_t llw[SUM_LL ? SUM_SLOTS : 1][2];
    if constexpr (SUM_LL) {
      // One pass over the slice first -- by now it has usually landed --
      // and only while some of it is missing, a spin on one sentinel word
      // per (slot, 32-row o_proj tile) before the next pass. Re-reading the
      // whole slice per spin, from every router tile on the GPU, put ~3 MB
      // of system-scope loads on the fabric per round and slowed the run by
      // 0.5 ms. Waves 0-1 only: MPK_OPROJ_LL == 2 keeps its XGMI stores, and
      // so their acks, on wave 3.
      constexpr int LL_TILE_ROWS = 32;
      unsigned long long const *const ll =
          static_cast<unsigned long long const *>(sum_slots);
      int const tiles_per_slice = (xs_f4 * 4) / LL_TILE_ROWS;
      int const sentinels = SUM_SLOTS * tiles_per_slice;
      if ((xs_f4 * 4) % LL_TILE_ROWS != 0 || sentinels > 128 || xs_f4 > 128) {
        __builtin_trap();
      }
      unsigned const want = static_cast<unsigned>(stream_expected);
      int const i0 = xs_t * xs_f4 + tid;
      while (true) {
        bool ok = true;
        if (tid < xs_f4) {
#pragma unroll
          for (int s = 0; s < SUM_SLOTS; s++) {
#pragma unroll
            for (int h = 0; h < 2; h++) {
              asm volatile("global_load_dwordx4 %0, %1, off sc0 sc1"
                           : "=v"(llw[s][h])
                           : "v"(ll + (size_t)s * REDUCTION_SIZE + i0 * 4 +
                                 h * 2)
                           : "memory");
            }
          }
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#pragma unroll
          for (int s = 0; s < SUM_SLOTS; s++) {
#pragma unroll
            for (int h = 0; h < 2; h++) {
              ok = ok && llw[s][h][1] == want && llw[s][h][3] == want;
            }
          }
        }
        if (__syncthreads_and(ok)) {
          break;
        }
        if (tid < sentinels) {
          int const s = tid / tiles_per_slice;
          int const j = tid % tiles_per_slice;
          unsigned long long *const sp = const_cast<unsigned long long *>(
              ll + (size_t)s * REDUCTION_SIZE + xs_t * xs_f4 * 4 +
              j * LL_TILE_ROWS);
          while ((unsigned)(ld_sys_u64(sp) >> 32) != want) {
            __builtin_amdgcn_s_sleep(1);
          }
        }
        __syncthreads();
      }
    }
    MPK_WS_MARK(791, xs_t);
    // Stage stamps 36/37/38/39 split S33 -> S34: slice validated, XCD row
    // exchange passed, norm done, logits stored (before the Step 3 drain).
    if (tid == 0) {
      mpk_stage_stamp(36);
    }
    if constexpr (KSPLIT) {
      // Partial logits and the slice's sum of squares go to this XCD's row
      // area as 64-word blocks, one per tile: words 0..31 the XCD's experts,
      // word 32 the sum of squares. Every XCD sums every slice identically, so
      // any XCD's 16 sums of squares give the same 1/rms bit for bit.
      if (xs_f4 * 4 != KS_SLICE || xs_t != tile_idx || rll == nullptr ||
          routing_epoch_hint <= 0) {
        __builtin_trap();
      }
      extern __shared__ char _rks_dyn_smem[];
      float *const ks_lds = reinterpret_cast<float *>(_rks_dyn_smem + 2048);
      float *const ks_ssq = reinterpret_cast<float *>(_rks_dyn_smem + 4096);
      unsigned long long *const ks_own =
          reinterpret_cast<unsigned long long *>(xsplit +
                                                 MPK_XSPLIT_FLAG_LINES * 16);
      unsigned long long const ks_ep =
          (unsigned long long)(unsigned)routing_epoch_hint << 32;
      float hk[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      float g4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      float ssq_t = 0.0f;
      int const i = xs_t * xs_f4 + tid;
      if (tid < xs_f4) {
        float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int s = 0; s < SUM_SLOTS; s++) {
          a[0] += __uint_as_float(llw[s][0][0]);
          a[1] += __uint_as_float(llw[s][0][2]);
          a[2] += __uint_as_float(llw[s][1][0]);
          a[3] += __uint_as_float(llw[s][1][2]);
        }
        unsigned b[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
          bf16 const r = __float2bfloat16(a[j]);
          unsigned short u;
          __builtin_memcpy(&u, &r, 2);
          b[j] = u;
          hk[j] = __uint_as_float(b[j] << 16);
          ssq_t += hk[j] * hk[j];
        }
        if (i / (H4_PF / 8) == xs_x) {
          st_wt_u64((void *)(hout + i * 4),
                    (unsigned long long)(b[0] | (b[1] << 16)) |
                        ((unsigned long long)(b[2] | (b[3] << 16)) << 32));
        }
        unsigned const glo = (unsigned)ks_g[0];
        unsigned const ghi = (unsigned)ks_g[1];
        g4[0] = __uint_as_float(glo << 16);
        g4[1] = __uint_as_float(glo & 0xFFFF0000u);
        g4[2] = __uint_as_float(ghi << 16);
        g4[3] = __uint_as_float(ghi & 0xFFFF0000u);
        typedef float ks_f4_t __attribute__((ext_vector_type(4)));
        ks_f4_t hg;
        hg[0] = hk[0] * g4[0];
        hg[1] = hk[1] * g4[1];
        hg[2] = hk[2] * g4[2];
        hg[3] = hk[3] * g4[3];
        *reinterpret_cast<ks_f4_t *>(ks_lds + tid * 4) = hg;
      }
      {
        float v = ssq_t;
#pragma unroll
        for (int off = 32; off > 0; off >>= 1) {
          v += __shfl_xor(v, off);
        }
        if (lane == 0) {
          red[wave] = v;
        }
      }
      __syncthreads();
      if (tid == 0) {
        mpk_stage_stamp(37);
      }
      // SUM_LL's deferred prefetch drain: the gate chunks are used next.
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      float acc = 0.0f;
#pragma unroll
      for (int j = 0; j < KS_CH; j++) {
        typedef float ks_f4_t __attribute__((ext_vector_type(4)));
        int const c = (tid & 7) + 8 * j;
        ks_f4_t const h0 = *reinterpret_cast<ks_f4_t const *>(ks_lds + c * 8);
        ks_f4_t const h1 =
            *reinterpret_cast<ks_f4_t const *>(ks_lds + c * 8 + 4);
        unsigned const w0 = (unsigned)ks_w[j][0];
        unsigned const w1 = (unsigned)ks_w[j][1];
        unsigned const w2 = (unsigned)ks_w[j][2];
        unsigned const w3 = (unsigned)ks_w[j][3];
        acc += __uint_as_float(w0 << 16) * h0[0] +
               __uint_as_float(w0 & 0xFFFF0000u) * h0[1] +
               __uint_as_float(w1 << 16) * h0[2] +
               __uint_as_float(w1 & 0xFFFF0000u) * h0[3] +
               __uint_as_float(w2 << 16) * h1[0] +
               __uint_as_float(w2 & 0xFFFF0000u) * h1[1] +
               __uint_as_float(w3 << 16) * h1[2] +
               __uint_as_float(w3 & 0xFFFF0000u) * h1[3];
      }
      acc += __shfl_xor(acc, 1);
      acc += __shfl_xor(acc, 2);
      acc += __shfl_xor(acc, 4);
      if (tid == 0) {
        mpk_stage_stamp(38);
      }
      unsigned long long *const ks_blk = ks_own + xs_t * 64;
      if ((tid & 7) == 0) {
        st_wt_u64((void *)(ks_blk + (tid >> 3)), ks_ep | __float_as_uint(acc));
      }
      if (tid == 0) {
        float const s = ((red[0] + red[1]) + red[2]) + red[3];
        st_wt_u64((void *)(ks_blk + 32), ks_ep | __float_as_uint(s));
        mpk_stage_stamp(34);
      }
      // This tile's slice of the normed row, from the XCD's 16 sums of
      // squares in tile order -- the TopK's own sum, so the MoE input and
      // the logits share one 1/rms.
      if (tid < 16) {
        unsigned long long sw;
        while ((unsigned)((sw = ld_sys_u64(ks_own + tid * 64 + 32)) >> 32) !=
               (unsigned)routing_epoch_hint) {
          __builtin_amdgcn_s_sleep(1);
        }
        ks_ssq[tid] = __uint_as_float((unsigned)sw);
      }
      __syncthreads();
      if (tid < xs_f4) {
        float tot = 0.0f;
#pragma unroll
        for (int t = 0; t < 16; t++) {
          tot += ks_ssq[t];
        }
        float const irms_k = rsqrtf(tot / (float)ACTUAL_HIDDEN_DIM + 1e-5f);
        unsigned nb[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
          bf16 const r = __float2bfloat16(hk[j] * irms_k * g4[j]);
          unsigned short u;
          __builtin_memcpy(&u, &r, 2);
          nb[j] = u;
        }
        unsigned long long const npk =
            (unsigned long long)(nb[0] | (nb[1] << 16)) |
            ((unsigned long long)(nb[2] | (nb[3] << 16)) << 32);
        asm volatile("global_store_dwordx2 %0, %1, off"
                     :
                     : "v"(d_normed + i * 4), "v"(npk)
                     : "memory");
      }
      if (__builtin_amdgcn_readfirstlane(tid) < xs_f4) {
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      }
      __syncthreads();
      if (tid == 0) {
        asm volatile("global_store_dword %0, %1, off\n"
                     "s_waitcnt vmcnt(0)"
                     :
                     : "v"(xs_flag + xs_t), "v"((unsigned)routing_epoch_hint)
                     : "memory");
        mpk_stage_stamp(39);
      }
      // The TopK runs on the XCD's first worker past the router tiles (the
      // fused caller), so it polls from the start rather than behind this.
      return;
    }
    if (tid < xs_f4) {
      int const i = xs_t * xs_f4 + tid;
      sum_f4_t v[SUM_SLOTS];
      if constexpr (SUM_LL) {
#pragma unroll
        for (int s = 0; s < SUM_SLOTS; s++) {
          v[s][0] = __uint_as_float(llw[s][0][0]);
          v[s][1] = __uint_as_float(llw[s][0][2]);
          v[s][2] = __uint_as_float(llw[s][1][0]);
          v[s][3] = __uint_as_float(llw[s][1][2]);
        }
      } else {
#pragma unroll
        for (int s = 0; s < SUM_SLOTS; s++) {
          v[s] = __builtin_nontemporal_load(reinterpret_cast<sum_f4_t const *>(
              sl + (size_t)s * REDUCTION_SIZE + i * 4));
        }
      }
      float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int s = 0; s < SUM_SLOTS; s++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
          a[j] += v[s][j];
        }
      }
      unsigned b[4];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        bf16 const r = __float2bfloat16(a[j]);
        unsigned short u;
        __builtin_memcpy(&u, &r, 2);
        b[j] = u;
      }
      unsigned long long const pk =
          (unsigned long long)(b[0] | (b[1] << 16)) |
          ((unsigned long long)(b[2] | (b[3] << 16)) << 32);
      asm volatile("global_store_dwordx2 %0, %1, off"
                   :
                   : "v"(xs_row + i * 4), "v"(pk)
                   : "memory");
      // Every XCD sums every element; the one whose eighth of the row it is
      // publishes it.
      if (i / (H4_PF / 8) == xs_x) {
        st_wt_u64((void *)(hout + i * 4), pk);
      }
    }
    // Under SUM_LL only the waves that stored the slice drain: a wave with
    // XGMI pushes still in flight would hold the flag back for their acks.
    // Wave-uniform, so the wait is branched around rather than masked.
    if (!SUM_LL || __builtin_amdgcn_readfirstlane(tid) < xs_f4) {
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }
    __syncthreads();
    if (tid == 0) {
      asm volatile("global_store_dword %0, %1, off\n"
                   "s_waitcnt vmcnt(0)"
                   :
                   : "v"(xs_flag + xs_t), "v"(xs_epoch)
                   : "memory");
    }
    MPK_WS_MARK(792, xs_t);
    // The flags and the row are only ever written from this XCD, so its L2
    // is where they are coherent. `nt` is the load that misses the vL1 and
    // still hits the L2: sc0 alone is Hit-LRU in the vL1, sc1 bypasses the
    // L2 on this multi-L2 part, and a bare buffer_inv is a NOP -- each of
    // those spins on a stale copy of the line forever. All of wave 0 polls
    // and leaves on one ballot, so a capture can say whose flag is missing.
    if (tid < 64) {
      int spins = 0;
      while (true) {
        unsigned seen = xs_epoch;
        if (tid < xs_n) {
          asm volatile("global_load_dword %0, %1, off nt\n"
                       "s_waitcnt vmcnt(0)"
                       : "=v"(seen)
                       : "v"(xs_flag + tid)
                       : "memory");
        }
        unsigned long long const ok = __ballot(seen >= xs_epoch);
        if (ok == ~0ull) {
          break;
        }
        if ((++spins & 4095) == 0) {
          MPK_WS_MARK(794, (int)(~ok & 0xFFFFull));
        }
        __builtin_amdgcn_s_sleep(1);
      }
    }
    __syncthreads();
    MPK_WS_MARK(793, xs_t);
    if (tid == 0) {
      mpk_stage_stamp(37);
    }
#pragma unroll
    for (int iter = 0; iter < SUM_ITERS; iter++) {
      int const i = tid + iter * 256;
      unsigned long long const pk = __builtin_nontemporal_load(
          reinterpret_cast<unsigned long long const *>(xs_row + i * 4));
#pragma unroll
      for (int j = 0; j < 4; j++) {
        unsigned const b = (unsigned)(pk >> (16 * j)) & 0xFFFFu;
        hs[iter][j] = __uint_as_float(b << 16);
        ssq[0] += hs[iter][j] * hs[iter][j];
      }
    }
    (void)hacc;
    (void)q_lo;
#else
    int p = 0;
#pragma unroll 1
    while (p < SUM_SLOTS) {
      if (tid == 0) {
        while (!slot_ready(p)) {
          __builtin_amdgcn_s_sleep(1);
        }
        s_stream_two = (p + 1 < SUM_SLOTS && slot_ready(p + 1)) ? 1 : 0;
      }
      __syncthreads();
      bool const two = s_stream_two != 0;
      asm volatile("buffer_inv" ::: "memory");
      sum_f4_t va[SUM_ITERS], vb[SUM_ITERS];
#pragma unroll
      for (int iter = 0; iter < SUM_ITERS; iter++) {
        va[iter] = *reinterpret_cast<sum_f4_t const *>(
            sl + (size_t)p * REDUCTION_SIZE + (tid + iter * 256) * 4);
      }
      if (two) {
#pragma unroll
        for (int iter = 0; iter < SUM_ITERS; iter++) {
          vb[iter] = *reinterpret_cast<sum_f4_t const *>(
              sl + (size_t)(p + 1) * REDUCTION_SIZE + (tid + iter * 256) * 4);
        }
      }
#pragma unroll
      for (int iter = 0; iter < SUM_ITERS; iter++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
          hacc[iter][j] += va[iter][j];
        }
      }
      if (two) {
#pragma unroll
        for (int iter = 0; iter < SUM_ITERS; iter++) {
#pragma unroll
          for (int j = 0; j < 4; j++) {
            hacc[iter][j] += vb[iter][j];
          }
        }
      }
      // s_stream_two is rewritten next round.
      __syncthreads();
      p += two ? 2 : 1;
    }
#pragma unroll
    for (int iter = 0; iter < SUM_ITERS; iter++) {
      int const i = tid + iter * 256;
      int const base = i * 4;
      unsigned b[4];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        bf16 const r = __float2bfloat16(hacc[iter][j]);
        unsigned short u;
        __builtin_memcpy(&u, &r, 2);
        b[j] = u;
        hs[iter][j] = __uint_as_float(b[j] << 16);
        ssq[0] += hs[iter][j] * hs[iter][j];
      }
      if (i >= q_lo && i < q_lo + q_per) {
        st_wt_u64((void *)(hout + base),
                  (unsigned long long)(b[0] | (b[1] << 16)) |
                      ((unsigned long long)(b[2] | (b[3] << 16)) << 32));
      }
    }
#endif // MPK_ROUTER_XSPLIT
  } else if constexpr (SUM_SLOTS > 0) {
    static_assert(BATCH_SIZE == 1 && OPROJ_BARRIER &&
                      REDUCTION_SIZE % (4 * 256) == 0,
                  "the slot sum walks one row in whole 256 x 4 passes, the "
                  "same walk Step 2's prefetch path indexes");
    typedef float sum_f4_t __attribute__((ext_vector_type(4)));
    float const *const sl = static_cast<float const *>(sum_slots);
    unsigned short *const hout = static_cast<unsigned short *>(sum_hidden_out);
    int const q_per = H4_PF / total_gang_tiles;
    if (q_per * total_gang_tiles != H4_PF) {
      __builtin_trap();
    }
    int const q_lo = sum_tile * q_per;
#pragma unroll
    for (int iter = 0; iter < SUM_ITERS; iter++) {
      int const i = tid + iter * 256;
      int const base = i * 4;
      float f[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      if constexpr (SUM_F32) {
        sum_f4_t pk[SUM_SLOTS];
#pragma unroll
        for (int p = 0; p < SUM_SLOTS; p++) {
          pk[p] = *reinterpret_cast<sum_f4_t const *>(
              sl + (size_t)p * REDUCTION_SIZE + base);
        }
#pragma unroll
        for (int p = 0; p < SUM_SLOTS; p++) {
#pragma unroll
          for (int j = 0; j < 4; j++) {
            f[j] += pk[p][j];
          }
        }
      } else {
        unsigned short const *const sl16 =
            static_cast<unsigned short const *>(sum_slots);
        uint2 pk[SUM_SLOTS];
#pragma unroll
        for (int p = 0; p < SUM_SLOTS; p++) {
          pk[p] = *reinterpret_cast<uint2 const *>(
              sl16 + (size_t)p * REDUCTION_SIZE + base);
        }
#pragma unroll
        for (int p = 0; p < SUM_SLOTS; p++) {
          f[0] += __uint_as_float(pk[p].x << 16);
          f[1] += __uint_as_float(pk[p].x & 0xFFFF0000u);
          f[2] += __uint_as_float(pk[p].y << 16);
          f[3] += __uint_as_float(pk[p].y & 0xFFFF0000u);
        }
      }
      unsigned b[4];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        bf16 const r = __float2bfloat16(f[j]);
        unsigned short u;
        __builtin_memcpy(&u, &r, 2);
        b[j] = u;
        hs[iter][j] = __uint_as_float(b[j] << 16);
        ssq[0] += hs[iter][j] * hs[iter][j];
      }
      if (i >= q_lo && i < q_lo + q_per) {
        st_wt_u64((void *)(hout + base),
                  (unsigned long long)(b[0] | (b[1] << 16)) |
                      ((unsigned long long)(b[2] | (b[3] << 16)) << 32));
      }
    }
  } else if (!irms_cached && !folded) {
    int const h4 = REDUCTION_SIZE >> 2;
    for (int i = tid; i < h4; i += (int)MPK_NT) {
      int base = i * 4;
      // Rows are REDUCTION_SIZE apart; the row loop is unrolled so `m` stays
      // a literal and ssq[] never leaves registers. At BATCH_SIZE 1 the row
      // offset is a constant zero and this is the original loop.
#pragma unroll
      for (int m = 0; m < BATCH_SIZE; m++) {
        bf16 const *hr = d_hidden + (size_t)m * REDUCTION_SIZE;
        float v0 = __bfloat162float(hr[base]);
        float v1 = __bfloat162float(hr[base + 1]);
        float v2 = __bfloat162float(hr[base + 2]);
        float v3 = __bfloat162float(hr[base + 3]);
        ssq[m] += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
      }
    }
    // Scalar tail
    for (int i = (h4 << 2) + tid; i < REDUCTION_SIZE; i += (int)MPK_NT) {
#pragma unroll
      for (int m = 0; m < BATCH_SIZE; m++) {
        float v = __bfloat162float(d_hidden[(size_t)m * REDUCTION_SIZE + i]);
        ssq[m] += v * v;
      }
    }
  }

  float irms[BATCH_SIZE];
  if (irms_cached) {
#pragma unroll
    for (int m = 0; m < BATCH_SIZE; m++) {
      irms[m] = irms_cache[m];
    }
  } else if (folded) {
    // FOLDED_RANKS floats, read by one thread. The whole of Step 1 -- a 12 KB
    // row read plus a two-level block reduction -- is this.
    if (tid == 0) {
      float tot = 0.0f;
      for (int p = 0; p < folded_ranks; p++) {
        tot += folded_lines[(size_t)p * folded_stride + NUM_EXPERTS];
      }
      // The fold is single-row by construction -- the o_proj epilogue that
      // produces folded_lines accumulates one row's two contractions -- and
      // the caller disables it at BATCH_SIZE > 1 for exactly that reason.
      s_irms[0] = rsqrtf(tot / (float)ACTUAL_HIDDEN_DIM + 1e-5f);
    }
    __syncthreads();
#pragma unroll
    for (int m = 0; m < BATCH_SIZE; m++) {
      irms[m] = s_irms[0];
    }
    if (irms_cache != nullptr && tid == 0) {
#pragma unroll
      for (int m = 0; m < BATCH_SIZE; m++) {
        irms_cache[m] = irms[m];
      }
    }
  } else {
// Wave-level reduction (64 lanes), one chain per row
#pragma unroll
    for (int m = 0; m < BATCH_SIZE; m++) {
      float v = ssq[m];
#if MPK_RMSNORM_DPP
      v = gang_rmsnorm_detail::_mpk_wave_sum_to_lane0(v);
#else
#pragma unroll
      for (int off = 32; off > 0; off >>= 1) {
        v += __shfl_xor(v, off);
      }
#endif
      // Cross-wave reduction via LDS
      if (lane == 0) {
        red[m * NUM_WAVES + wave] = v;
      }
    }
    __syncthreads();

    if (tid == 0) {
#pragma unroll
      for (int m = 0; m < BATCH_SIZE; m++) {
        float tot = 0.0f;
        for (int w = 0; w < NUM_WAVES; w++) {
          tot += red[m * NUM_WAVES + w];
        }
        s_irms[m] = rsqrtf(tot / (float)ACTUAL_HIDDEN_DIM + 1e-5f);
      }
    }
    __syncthreads();
#pragma unroll
    for (int m = 0; m < BATCH_SIZE; m++) {
      irms[m] = s_irms[m];
    }
    // `red` is reused by the dp reduction below; s_irms is not, which is why
    // it is its own array.
    if (irms_cache != nullptr && tid == 0) {
#pragma unroll
      for (int m = 0; m < BATCH_SIZE; m++) {
        irms_cache[m] = irms[m];
      }
    }
  }

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  unsigned long long _rt_t1 = __builtin_amdgcn_s_memrealtime();
  if (tid == 0 && g_subphase_active) {
    atomicAdd(&g_subphase_ns[6][1], (_rt_t1 - _rt_t0) * 10); // Step 1 norm
  }
#endif
  // SUM_LL's deferred prefetch drain (see the barrier above).
  if constexpr (OPROJ_BARRIER && SUM_LL) {
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  }
  if (tid == 0) {
    mpk_stage_stamp(38);
  }

  // ═══ Step 2: Fused Gate GEMV + norm write ═══
  // One expert per worker. Each thread handles REDUCTION_SIZE/blockDim.x
  // elements in a single pass: compute normed, write to d_normed (for
  // downstream MoE FP8 quant), accumulate gate dot product.
  //
  // Gate weight layout: [chunk_N, REDUCTION_SIZE] bf16, row-major.
  // tile_idx = expert index within this XCD (0..chunk_N-1).
  // One dot product per (row, expert). The gate row is row-independent, so
  // widening this costs BATCH_SIZE accumulators and no extra weight traffic --
  // which is the whole reason the row loop sits INSIDE the k loop below.
  float dp[BATCH_SIZE][EXPERTS_PER_TILE];
#pragma unroll
  for (int m = 0; m < BATCH_SIZE; m++) {
#pragma unroll
    for (int e = 0; e < EXPERTS_PER_TILE; e++) {
      dp[m][e] = 0.0f;
    }
  }
  bf16 const *my_gate = d_gate_w + (int64_t)first_expert * REDUCTION_SIZE;
  // Rows past the live token count hold whatever the last step left behind,
  // so their normed row and their logit are garbage. Computing them is free
  // (the loops are unrolled on a compile-time bound); STORING them is not
  // always harmless -- a NaN in a dead hidden row would reach the MoE
  // quantizer -- so the stores, and only the stores, are masked.
  int const live_rows =
      (BATCH_SIZE == 1)
          ? 1
          : (num_active_tokens < BATCH_SIZE ? num_active_tokens : BATCH_SIZE);

  // Every worker walks the whole row for its own gate dot product, so every
  // worker *can* write the normed row -- and until now every worker did, all
  // NUM_EXPERTS/8 of them per XCD, to the same global buffer. The values are
  // identical, so all but one copy is pure store traffic on the critical path:
  // for GLM that is 8 workers x 4 KB x 8 XCDs = 256 KB per layer where 4 KB is
  // needed. One writer per XCD is enough (leaving the copies XCD-local, which
  // costs nothing and keeps the store in the writer's own L2).
  bool const write_normed = (first_expert == 0);

  if (folded) {
    // Both contractions are already reduced across the ranks, and the gate
    // weight is not read here at all -- it was read once, transposed and four
    // columns at a time, by the o_proj tiles that own those columns. What is
    // left of Step 2 is the normed row the MoE quantizer reads.
    //
    // Every tile writes FOLD_COLS of it rather than tile 0 writing all of it.
    // Unfolded, the one-writer rule was free: the write hid inside sixteen
    // tiles' worth of gate GEMV. Folded there is no GEMV to hide inside, so a
    // whole-row write on one tile is fifteen idle workers and a routing
    // barrier that waits for all of it -- measured as +961 ns on this step
    // and +0.38 ms/iter on RoutingWait. Split sixteen ways it is 384 columns
    // per tile, already in registers from the Step 0 prefetch.
    {
      constexpr int FOLD_H4 = FOLD_COLS >> 2;
#pragma unroll
      for (int iter = 0; iter < FOLD_PF; iter++) {
        int i_cur = tid + iter * 256;
        if (i_cur >= FOLD_H4) {
          break;
        }
        bf16 const *gp = reinterpret_cast<bf16 const *>(&fg_pf[iter]);
        int const base = fold_col0 + i_cur * 4;
#pragma unroll
        for (int j = 0; j < 4; j++) {
          d_normed[base + j] = __float2bfloat16(
              __bfloat162float(d_hidden[base + j]) * irms[0] *
              __bfloat162float(gp[j]));
        }
      }
    }
    // irms factored out of the o_proj-side accumulation, applied once here.
    //
    // One lane per rank, not one thread per everything. These eight addresses
    // are on the symmetric heap, written write-through by the peers and read
    // on the far side of a buffer_inv, so every one of them is an uncached
    // round trip -- and the loop as first written had all 256 threads reissue
    // all eight of them per expert. 4096 uncached loads is why this step got
    // *more* expensive after its GEMV was deleted (2282 -> 3187 ns). Only
    // wave 0 runs it because only tid 0's copy is read, by the red[] store
    // below; the other waves' slots are zeroed there.
    if (wave == 0) {
#pragma unroll
      for (int e = 0; e < EXPERTS_PER_TILE; e++) {
        float s = (lane < folded_ranks)
                      ? folded_lines[(size_t)lane * folded_stride +
                                     folded_expert_base + first_expert + e]
                      : 0.0f;
#pragma unroll
        for (int off = 32; off > 0; off >>= 1) {
          s += __shfl_xor(s, off);
        }
        dp[0][e] = s * irms[0];
      }
    }
  } else if constexpr (OPROJ_BARRIER) {
    // Same arithmetic as the branch below, but iterating over the prefetch
    // slots so `iter` is a literal -- an array indexed by a runtime value
    // would be spilled to scratch and the prefetch would be worthless.
    // gamma and the gate row come from registers; only d_hidden is loaded.
#pragma unroll
    for (int iter = 0; iter < MAX_ITERS_PF; iter++) {
      int i = tid + iter * 256;
      if (i >= H4_PF) {
        break;
      }
      int base = i * 4;
      bf16 const *gp = reinterpret_cast<bf16 const *>(&g_pf[iter]);
      // Row loop innermost so gamma and the EXPERTS_PER_TILE gate rows are
      // fetched once and reused across rows; only d_hidden is re-read. At
      // BATCH_SIZE 1 the offsets are constant zeroes and this is the loop it
      // was.
#pragma unroll
      for (int m = 0; m < BATCH_SIZE; m++) {
        float h0, h1, h2, h3;
        if constexpr (SUM_SLOTS > 0) {
          h0 = hs[iter][0];
          h1 = hs[iter][1];
          h2 = hs[iter][2];
          h3 = hs[iter][3];
        } else {
          bf16 const *hr = d_hidden + (size_t)m * REDUCTION_SIZE;
          h0 = __bfloat162float(hr[base]);
          h1 = __bfloat162float(hr[base + 1]);
          h2 = __bfloat162float(hr[base + 2]);
          h3 = __bfloat162float(hr[base + 3]);
        }

        float n0 = h0 * irms[m] * __bfloat162float(gp[0]);
        float n1 = h1 * irms[m] * __bfloat162float(gp[1]);
        float n2 = h2 * irms[m] * __bfloat162float(gp[2]);
        float n3 = h3 * irms[m] * __bfloat162float(gp[3]);

        if (write_normed && (BATCH_SIZE == 1 || m < live_rows)) {
          bf16 *nr = d_normed + (size_t)m * REDUCTION_SIZE;
          nr[base] = __float2bfloat16(n0);
          nr[base + 1] = __float2bfloat16(n1);
          nr[base + 2] = __float2bfloat16(n2);
          nr[base + 3] = __float2bfloat16(n3);
        }

#pragma unroll
        for (int e = 0; e < EXPERTS_PER_TILE; e++) {
          bf16 const *wp = reinterpret_cast<bf16 const *>(&w_pf[e][iter]);
          dp[m][e] += __bfloat162float(wp[0]) * n0 +
                      __bfloat162float(wp[1]) * n1 +
                      __bfloat162float(wp[2]) * n2 +
                      __bfloat162float(wp[3]) * n3;
        }
      }
    }
    static_assert(!OPROJ_BARRIER || (REDUCTION_SIZE % 4) == 0,
                  "the prefetch path has no scalar tail, so K must be a "
                  "multiple of 4");
  } else {
    int const h4 = REDUCTION_SIZE >> 2;
    for (int i = tid; i < h4; i += (int)MPK_NT) {
      int base = i * 4;
      float g0 = __bfloat162float(d_gamma[base]);
      float g1 = __bfloat162float(d_gamma[base + 1]);
      float g2 = __bfloat162float(d_gamma[base + 2]);
      float g3 = __bfloat162float(d_gamma[base + 3]);

#pragma unroll
      for (int m = 0; m < BATCH_SIZE; m++) {
        bf16 const *hr = d_hidden + (size_t)m * REDUCTION_SIZE;
        float h0 = __bfloat162float(hr[base]);
        float h1 = __bfloat162float(hr[base + 1]);
        float h2 = __bfloat162float(hr[base + 2]);
        float h3 = __bfloat162float(hr[base + 3]);

        float n0 = h0 * irms[m] * g0;
        float n1 = h1 * irms[m] * g1;
        float n2 = h2 * irms[m] * g2;
        float n3 = h3 * irms[m] * g3;

        // Write normed output (redundant across 128 workers, idempotent).
        // Needed by downstream MoE FP8 quant task.
        if (write_normed && (BATCH_SIZE == 1 || m < live_rows)) {
          bf16 *nr = d_normed + (size_t)m * REDUCTION_SIZE;
          nr[base] = __float2bfloat16(n0);
          nr[base + 1] = __float2bfloat16(n1);
          nr[base + 2] = __float2bfloat16(n2);
          nr[base + 3] = __float2bfloat16(n3);
        }

        // Gate GEMV: accumulate dot product
#pragma unroll
        for (int e = 0; e < EXPERTS_PER_TILE; e++) {
          bf16 const *ge = my_gate + (int64_t)e * REDUCTION_SIZE;
          float w0 = __bfloat162float(ge[base]);
          float w1 = __bfloat162float(ge[base + 1]);
          float w2 = __bfloat162float(ge[base + 2]);
          float w3 = __bfloat162float(ge[base + 3]);
          dp[m][e] += w0 * n0 + w1 * n1 + w2 * n2 + w3 * n3;
        }
      }
    }
    // Scalar tail
    for (int i = (h4 << 2) + tid; i < REDUCTION_SIZE; i += (int)MPK_NT) {
      float g = __bfloat162float(d_gamma[i]);
#pragma unroll
      for (int m = 0; m < BATCH_SIZE; m++) {
        float h = __bfloat162float(d_hidden[(size_t)m * REDUCTION_SIZE + i]);
        float n = h * irms[m] * g;
        if (write_normed && (BATCH_SIZE == 1 || m < live_rows)) {
          d_normed[(size_t)m * REDUCTION_SIZE + i] = __float2bfloat16(n);
        }
#pragma unroll
        for (int e = 0; e < EXPERTS_PER_TILE; e++) {
          dp[m][e] +=
              __bfloat162float(my_gate[(int64_t)e * REDUCTION_SIZE + i]) * n;
        }
      }
    }
  }

  // Wave-level reduction, then one LDS slot per (wave, expert). `red` is 16
  // floats, so EXPERTS_PER_TILE * NUM_WAVES has to fit -- 4 experts at four
  // waves is the ceiling, and nothing here wants to go that wide.
  static_assert(BATCH_SIZE * EXPERTS_PER_TILE * NUM_WAVES <= 32,
                "red[] holds NUM_WAVES partial sums per (row, expert)");
  if (folded) {
    // dp is already whole and identical in every thread -- there is nothing
    // to reduce. Land it in the slot the store below reads so that store stays
    // one code path.
    //
    // The sync is not the store's; it is irms's. Step 1's folded branch left
    // irms in red[0], and on this path Step 2 is empty for 15 of 16 tiles, so
    // wave 0 can reach the overwrite below before wave 3 has read it. The
    // legacy path is only safe here because its Step 2 is thousands of cycles
    // long, which is not a property worth inheriting.
    __syncthreads();
    if (tid == 0) {
#pragma unroll
      for (int e = 0; e < EXPERTS_PER_TILE; e++) {
        for (int w = 1; w < NUM_WAVES; w++) {
          red[e * NUM_WAVES + w] = 0.0f;
        }
        red[e * NUM_WAVES] = dp[0][e];
      }
    }
  } else {
#pragma unroll
    for (int m = 0; m < BATCH_SIZE; m++) {
#pragma unroll
      for (int e = 0; e < EXPERTS_PER_TILE; e++) {
        float v = dp[m][e];
#pragma unroll
        for (int off = 32; off > 0; off >>= 1) {
          v += __shfl_xor(v, off);
        }
        if (lane == 0) {
          red[(m * EXPERTS_PER_TILE + e) * NUM_WAVES + wave] = v;
        }
      }
    }
  }
  __syncthreads();

  // tid==0 writes logit + bias via write-through store. One logit row per
  // token; moe_gate_out is [batch, NUM_EXPERTS] and this pointer is already
  // this XCD's column slice, so NUM_EXPERTS is still the row stride.
  if (tid == 0) {
#pragma unroll
    for (int m = 0; m < BATCH_SIZE; m++) {
    if (BATCH_SIZE > 1 && m >= live_rows) {
      break;
    }
#pragma unroll
    for (int e = 0; e < EXPERTS_PER_TILE; e++) {
      float s = 0.0f;
      for (int w = 0; w < NUM_WAVES; w++) {
        s += red[(m * EXPERTS_PER_TILE + e) * NUM_WAVES + w];
      }
      // `noaux_tc` keeps its bias out of the logit -- it only steers
      // selection, and the weight that gets emitted comes from the unbiased
      // sigmoid. The tail applies it. Everyone else folds it in here as an
      // ordinary GEMV bias.
      if (!SIGMOID_BIAS && d_bias) {
        s += __bfloat162float(d_bias[first_expert + e]);
      }
      bf16 bval = __float2bfloat16(s);
#if MPK_ROUTER_LL
      if (rll != nullptr) {
        // The epoch rides in the same 8-byte store as the logit, so a reader
        // that sees this layer's epoch has this layer's logit.
        unsigned short u;
        __builtin_memcpy(&u, &bval, 2);
        int const g = gang_rmsnorm_topk_detail::get_xcd_id() *
                          (NUM_EXPERTS / 8) +
                      first_expert + e;
        st_wt_u64((void *)(reinterpret_cast<unsigned long long *>(rll) + g),
                  ((unsigned long long)(unsigned)routing_epoch_hint << 32) |
                      u);
        continue;
      }
#endif
      st_wt_u16(&d_logits[(size_t)m * NUM_EXPERTS + first_expert + e],
                *reinterpret_cast<unsigned short *>(&bval));
    }
    }
    mpk_stage_stamp(39);
  }

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  unsigned long long _rt_t2 = __builtin_amdgcn_s_memrealtime();
  if (tid == 0 && g_subphase_active) {
    atomicAdd(&g_subphase_ns[6][2], (_rt_t2 - _rt_t1) * 10); // Step 2 gate dot
  }
#endif

  // ═══ Step 3: Cross-XCD barrier via atomic counter ═══
  // Write-through stores (sc0 sc1) bypass L2 → HBM. s_waitcnt ensures
  // stores are globally visible before incrementing counter.
  __syncthreads();
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

  // MEASURED, do not re-attempt: splitting this into a two-level counter --
  // one 64-byte line per XCD at [(1 + x) * 16] closed by that XCD's 32nd
  // arrival, then eight arrivals on [0] -- is a 6.0% regression (decode
  // 13.031 / 12.863 ms against 12.281 / 12.150 for this line, four runs
  // interleaved in one session, 2026-08-19). The premise was that 256
  // device-scope atomicAdds on one word serialise at the memory-side atomic
  // unit; SP6[3]'s 2.10 us mean is arrival *skew*, not that contention, so
  // there is nothing to win and the second, dependent atomic that the electing
  // block has to issue before the TopK tail can start is pure added latency on
  // the path all 232 workers' RoutingWait hangs off.
  //
  // Two further traps if it is ever revisited anyway. The nine-word variant
  // must not be reset by the tail: zeroing nine words instead of one widens the
  // window in which a worker already into the next layer reads a half-cleared
  // counter, and that deadlocked at iteration 0 in three runs out of three. Run
  // the words monotonic with a modular test instead, the way the o_proj barrier
  // in gang_oproj_router_fused_mi300.cuh does -- that ran, it was merely slow.
  // And the split is only sound because the router is entered by all eight XCDs
  // or by none; the dense prefix layers and the EP_TAIL_ONLY variant return
  // before Phase 3 uniformly, so no XCD's line drifts out of step.
  __shared__ int s_completed;
  MPK_WS_MARK(772, tile_idx);
#if MPK_ROUTER_LL
  // The logit words are the arrival; nobody counts and nobody is elected.
  bool const rll_on = rll != nullptr;
  if (rll_on && (BATCH_SIZE != 1 || !SIGMOID_BIAS || routing_epoch_hint <= 0)) {
    __builtin_trap();
  }
#else
  bool const rll_on = false;
#endif
  if (tid == 0) {
    s_completed = rll_on
                      ? 0
                      : atomicAdd(static_cast<int *>(gang_counter_ptr), 1) + 1;
  }
  __syncthreads();
  int completed = s_completed;
  MPK_WS_MARK(773, completed);
  // Stage stamp 34: this router tile's logits are stored and it has arrived.
  // The max over workers is the last arrival, where the TopK tail starts.
  if (tid == 0) {
    mpk_stage_stamp(34);
  }

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  unsigned long long _rt_t3 = __builtin_amdgcn_s_memrealtime();
  if (tid == 0 && g_subphase_active) {
    atomicAdd(&g_subphase_ns[6][3], (_rt_t3 - _rt_t2) * 10); // Step 3 arrival
  }
#endif

  // ═══ Step 4: Last worker runs TopK ═══
#if MPK_ROUTER_LL
  if constexpr (SIGMOID_BIAS && BATCH_SIZE == 1) {
    if (rll_on && tile_idx == 0) {
      gang_rmsnorm_topk_detail::topk_ll_noinline<T, NUM_EXPERTS, K>(
          rll,
          gang_rmsnorm_topk_detail::get_xcd_id(),
          (unsigned)routing_epoch_hint,
          const_cast<void *>(bias_ptr),
          num_active_tokens,
          renormalize,
          routed_scaling_factor,
          num_shared_experts);
    }
  }
#endif
  if (!rll_on && completed == total_gang_tiles) {
    if constexpr (SIGMOID_BIAS) {
      gang_rmsnorm_topk_detail::
          topk_sigmoid_noinline<T, NUM_EXPERTS, K, /*ROUTING_ROW_STRIDE=*/
                                BATCH_SIZE>(
          logits_scratch_ptr,
          const_cast<void *>(bias_ptr),
          topk_weight_ptr,
          routing_indices_ptr,
          active_expert_ids_ptr,
          gang_counter_ptr,
          num_active_tokens,
          renormalize,
          routed_scaling_factor,
          num_shared_experts,
          routing_ready_ptr,
          routing_epoch_hint);
    } else {
      gang_rmsnorm_topk_detail::
          topk_noinline<T, NUM_EXPERTS, K, /*ROUTING_ROW_STRIDE=*/BATCH_SIZE>(
          logits_scratch_ptr,
          topk_weight_ptr,
          routing_indices_ptr,
          active_expert_ids_ptr,
          gang_counter_ptr,
          num_active_tokens);
    }
#ifdef MPK_ENABLE_SUBPHASE_TIMING
    // Only the elected block reaches here, so SP6[4] is the serial TopK tail
    // itself -- one sample per layer, not per worker. g_subphase_cnt only has
    // SUBPHASE_SLOTS entries and slot 6 already counts router-kernel calls, so
    // SP6[5] carries this tail's own event count instead (a raw count, not ns).
    if (tid == 0 && g_subphase_active) {
      atomicAdd(&g_subphase_ns[6][4],
                (__builtin_amdgcn_s_memrealtime() - _rt_t3) * 10);
      atomicAdd(&g_subphase_ns[6][5], 1ULL);
    }
#endif
  }
  MPK_WS_MARK(774, completed);
}

} // namespace kernel
