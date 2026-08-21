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
  int const nthreads = blockDim.x;
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

#pragma unroll
  for (int offset = 32; offset > 0; offset >>= 1) {
    sum += __shfl_xor(sum, offset);
  }

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

template <int STORAGE_DIM, int ACTUAL_HIDDEN_DIM, int NORM_SPAN = STORAGE_DIM>
__device__ __forceinline__ void rmsnorm_inline_amd(void const *input_ptr,
                                                   void const *weight_ptr,
                                                   void *output_ptr,
                                                   float eps = 1e-5f) {
  bf16 const *__restrict__ d_input = static_cast<bf16 const *>(input_ptr);
  bf16 const *__restrict__ d_weight = static_cast<bf16 const *>(weight_ptr);
  bf16 *__restrict__ d_output = static_cast<bf16 *>(output_ptr);

  constexpr int VEC_SIZE = 8;   // 8 bf16 per 128-bit load
  constexpr int NTHREADS = 256; // block size for gang RMSNorm
  int const tid = threadIdx.x;
  int const nthreads = blockDim.x;
  constexpr int VEC_ITERS = STORAGE_DIM / (NTHREADS * VEC_SIZE);

  // ── Phase 1: sum of squares + cache input in registers ──
  // Cache d_input values to avoid re-reading from HBM in Phase 4.
  // GPT-OSS 120B: STORAGE_DIM=3072, 256 threads, VEC_SIZE=8 → VEC_ITERS=1,
  // tail=1024 elems → 4/thread. Total cache: 8 + 4 = 12 floats/thread.
  constexpr int _VEC_END = VEC_ITERS * NTHREADS * VEC_SIZE;
  constexpr int _TAIL_ELEMS = STORAGE_DIM - _VEC_END;
  constexpr int _MAX_TAIL = (_TAIL_ELEMS + NTHREADS - 1) / NTHREADS;
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
#pragma unroll
  for (int offset = 32; offset > 0; offset >>= 1) {
    sum += __shfl_xor(sum, offset);
  }

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
  gang_rmsnorm_detail::
      rmsnorm_inline_amd<REDUCTION_SIZE, ACTUAL_HIDDEN_DIM, NORM_SPAN>(
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
                                    ROUTING_ROW_STRIDE>(
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
      st_wt_u32((void *)routing_ready_ptr, (unsigned)epoch);
      for (int x = 0; x < 8; x++) {
        st_wt_u32((void *)&routing_ready_ptr[(1 + x) * 16], (unsigned)epoch);
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }
  }

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  if (threadIdx.x == 0 && g_subphase_active) {
    atomicAdd(&g_subphase_ns[6][7],
              (__builtin_amdgcn_s_memrealtime() - _tk_t1) * 10); // release
  }
#endif
}
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
          int EXPERTS_PER_TILE = 1>
__device__ __attribute__((noinline)) void gang_rmsnorm_linear_bias_topk_kernel(
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
    int folded_expert_base = 0) {

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
  if constexpr (OPROJ_BARRIER) {
    if (folded) {
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
    if (!folded) {
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
      while (ld_nt_s32(&hier[oproj_xcd_id * 16]) < oproj_release_expected) {
        if ((++_spins & (MPK_FL_REPUBLISH_SPINS - 1)) == 0) {
          for (int x = 0; x < 8; x++) {
            if (ld_nt_s32(&hier[x * 16]) >= oproj_release_expected) {
              st_wt_u32((void *)&hier[oproj_xcd_id * 16],
                        (unsigned)oproj_release_expected);
              asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
              break;
            }
          }
        }
        __builtin_amdgcn_s_sleep(1);
      }
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
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
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
  bool const irms_cached = (irms_cache != nullptr) && (irms_cache[0] > 0.0f);
  float ssq[BATCH_SIZE];
#pragma unroll
  for (int m = 0; m < BATCH_SIZE; m++) {
    ssq[m] = 0.0f;
  }
  if (!irms_cached && !folded) {
    int const h4 = REDUCTION_SIZE >> 2;
    for (int i = tid; i < h4; i += (int)blockDim.x) {
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
    for (int i = (h4 << 2) + tid; i < REDUCTION_SIZE; i += (int)blockDim.x) {
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
#pragma unroll
      for (int off = 32; off > 0; off >>= 1) {
        v += __shfl_xor(v, off);
      }
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
        bf16 const *hr = d_hidden + (size_t)m * REDUCTION_SIZE;
        float h0 = __bfloat162float(hr[base]);
        float h1 = __bfloat162float(hr[base + 1]);
        float h2 = __bfloat162float(hr[base + 2]);
        float h3 = __bfloat162float(hr[base + 3]);

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
    for (int i = tid; i < h4; i += (int)blockDim.x) {
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
    for (int i = (h4 << 2) + tid; i < REDUCTION_SIZE; i += (int)blockDim.x) {
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
      st_wt_u16(&d_logits[(size_t)m * NUM_EXPERTS + first_expert + e],
                *reinterpret_cast<unsigned short *>(&bval));
    }
    }
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
  if (tid == 0) {
    s_completed = atomicAdd(static_cast<int *>(gang_counter_ptr), 1) + 1;
  }
  __syncthreads();
  int completed = s_completed;

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  unsigned long long _rt_t3 = __builtin_amdgcn_s_memrealtime();
  if (tid == 0 && g_subphase_active) {
    atomicAdd(&g_subphase_ns[6][3], (_rt_t3 - _rt_t2) * 10); // Step 3 arrival
  }
#endif

  // ═══ Step 4: Last worker runs TopK ═══
  if (completed == total_gang_tiles) {
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
}

} // namespace kernel
