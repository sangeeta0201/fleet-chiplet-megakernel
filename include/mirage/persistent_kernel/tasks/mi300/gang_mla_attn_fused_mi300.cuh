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

// The attention half of a GLM decoder layer in one gang task: input RMSNorm +
// [q_a | kv_a] projection, q_a RMSNorm + absorbed q_b + latent KV-cache append,
// absorbed MLA decode, and the split-KV merge.
//
// Companion to gang_oproj_router_fused_mi300.cuh, which carries the other half
// (o_proj + post-attention RMSNorm + router + TopK + MoE). Together they put a
// decoder layer in two tasks -- gpt-oss's shape, and the reason
// gang_full_layer_fused_mi300.cuh exists there. GLM was running nine tasks per
// layer; the MoE fusion took that to five and 5.362 -> 4.973 ms/token, and
// these four attention dispatches are the rest of the way.
//
// Structure is the same wrapper-over-existing-kernels shape as the MoE half.
// No sub-kernel is rewritten; each is given a WRITE_THROUGH epilogue for the
// case where its consumer now sits across a barrier instead of across a task
// graph event.
//
// Barriers. Three Mechanism-C barriers in one counter tensor of 29 * 16 int32,
// laid out exactly like the MoE task's:
//
//   [0 .. 8]    qkv_a  -> q_b.     flags at [x * 16],        counter [8 * 16]
//   [10 .. 18]  q_b    -> decode.  flags at [(10 + x) * 16], counter [18 * 16]
//   [20 .. 28]  decode -> merge.   flags at [(20 + x) * 16], counter [28 * 16]
//
// Every dispatched worker arrives at all three -- unlike the MoE task, where
// splitting the router's barrier off is what lets the MoE-only workers skip
// it. Here the widest phase is q_b (28 tiles per XCD on GLM-4.7-Flash) and the
// task is dispatched at exactly that width, so "every worker" and "every
// producer" are the same set and there is nothing to split. What the narrow
// phases do instead is stop *waiting* early: only the decode's ranks wait on
// the q_b barrier, and past the decode barrier only the merge's four ranks per
// XCD remain -- the other 24 return to the scheduler a whole merge stage
// ahead of them.
//
// Cache discipline, same rule as the MoE half: same-XCD producer/consumer
// pairs use ordinary stores plus s_waitcnt, cross-XCD pairs use st_wt and a
// consumer-side buffer_inv. Unlike gpt-oss -- where kv_head == xcd_id keeps
// the whole attention chain XCD-local -- every producer here is read by every
// XCD, because MLA has a single shared latent head and the split is over the
// sequence instead:
//
//   qkv_a_out    the q_a columns are read by all 8 q_b GEMMs, and the latent
//                columns, which live on XCDs 4 and 5, are read by XCD 0's
//                tile 0.
//   q_workspace  a q group is 16 heads wide and the q_b GEMM spreads 3 heads
//                per XCD, so one decode tile reads six XCDs' output.
//   kv_cache     written by XCD 0's tile 0 alone, read by all 8.
//   o_acc / lse  XCD x computes chunk x; a merge task reduces all 8 chunks.
//
// So all three producing kernels run with WRITE_THROUGH=true. The merge's own
// output is not write-through by default: it is consumed by the next task
// across a real event boundary, which does the buffer_wbl2 itself. The knob
// is still plumbed, since it measured a dead heat as a standalone task and is
// worth re-measuring from inside here.
//
// Dispatch: tiles_per_xcd is the max over the four phases, and has to stay
// under the resident worker count or the in-kernel barriers deadlock. tile_idx
// is global -- this task registers as a variant of TASK_GANG_MLA_DECODE_MI300,
// which is in runtime.cc's n_tile_start list -- so tile_idx = xcd_id *
// tiles_per_xcd + xcd_rank. The two GEMM phases index tiles *within* an XCD
// and are handed xcd_rank; the decode and the merge want a global work-item
// index, which is synthesized from (xcd_id, xcd_rank) against their own
// per-XCD width rather than the dispatch width.

#pragma once
#include "tasks/ampere/merge_splitkv.cuh"
#include "tasks/mi300/gang_mla_decode_mi300.cuh"
#include "tasks/mi300/gang_rmsnorm_linear_mxfp8_bias_mla_kvupd_mi300.cuh"

namespace kernel {

template <int BATCH_SIZE,
          // ── qkv_a: input_layernorm + [q_a_proj | kv_a_proj_with_mqa] ──
          int QKV_OUTPUT_PER_WG,
          int QKV_REDUCTION_SIZE, // == hidden_size
          int QKV_ACTUAL_HIDDEN,  // RMSNorm divisor, <= QKV_REDUCTION_SIZE
          // ── q_b: q_a_layernorm + absorbed q_b_proj + latent append ──
          int QB_OUTPUT_PER_WG,  // == QK_ROPE_HEAD_DIM
          int QB_REDUCTION_SIZE, // == q_lora_pad
          int QB_ACTUAL_HIDDEN,  // == q_lora
          int KV_LORA_RANK,
          int QK_ROPE_HEAD_DIM,
          int KV_INPUT_STRIDE, // qkv_a_out's row width
          int KV_CACHE_STRIDE,
          int MAX_SEQ_LEN,
          int PAGE_SIZE,
          int KV_INPUT_OFFSET, // where the latent starts in the qkv_a row
          // ── decode + merge ──
          int NUM_Q_HEADS,
          int NUM_KV_CHUNKS,
          int Q_WORKSPACE_STRIDE,
          int MERGE_DIM_SPLITS,
          bool MERGE_WRITE_THROUGH>
__device__ __attribute__((always_inline)) void gang_mla_attn_fused_kernel_mi300(
    // ── inputs ──
    void const *x_ptr,               // [0]  residual stream
    void const *pre_norm_weight_ptr, // [1]  input_layernorm gamma
    void *pre_norm_scratch_ptr,      // [2]
    void const *qkv_weight_ptr,      // [3]  MXFP8, this XCD's chunk
    void const *qkv_bias_ptr,        // [4]  this XCD's slice
    void const *q_a_norm_weight_ptr, // [5]
    void *q_a_norm_scratch_ptr,      // [6]
    void const *qb_weight_ptr,       // [7]  MXFP8, this XCD's chunk
    void const *qb_bias_ptr,         // [8]  this XCD's slice
    void const *kv_norm_weight_ptr,  // [9]  kv_a_layernorm gamma
    void const *cos_ptr,             // [10]
    void const *sin_ptr,             // [11]
    void *kv_cache_ptr,              // [12] paged latent cache, written
    void *attn_counters_ptr,         // [13] this task's three barriers
    void const *moe_ws_f32_ptr,      // [14] previous layer's MoE accumulator
    // ── outputs ──
    void *qkv_a_out_ptr,   // [0] [q_a | latent], declared whole
    void *q_workspace_ptr, // [1] absorbed queries, declared whole
    void *lse_ptr,         // [2] per-chunk LSE
    void *o_acc_ptr,       // [3] per-chunk f32 partials
    void *attn_out_ptr,    // [4] merged bf16 attention output
    void *x_out_ptr,       // [5] this layer's resolved residual stream
    // ── indptr buffers ──
    int const *qo_indptr,
    int const *kv_indptr,
    int const *kv_indices,
    int const *kv_last_page_len,
    int16_t request_id,
    // ── parameters ──
    int num_active_tokens,
    int qkv_n_wgs_per_xcd,
    int qkv_output_stride,
    int qb_n_wgs_per_xcd,
    int qb_output_stride,
    int mla_tiles_per_xcd,
    int mla_total_work_items,
    int merge_tiles_per_xcd,
    int tiles_per_xcd,
    float scale_s,
    float kv_eps,
    int tile_idx,
    // Release values supplied by a caller that has already snapshotted them.
    // Negative means "snapshot them yourself", which is what the standalone
    // dispatch of this task does. The fused whole-layer task passes real
    // values because by the time this body runs, seven phases deep, the
    // snapshot is no longer behind the previous layer's event boundary -- see
    // gang_mla_full_layer_fused_mi300.cuh.
    int qkv_expected_in = -1,
    int qb_expected_in = -1,
    int decode_expected_in = -1) {

  int const tid = threadIdx.x;
  int const xcd_id = tile_idx / tiles_per_xcd;
  int const xcd_rank = tile_idx % tiles_per_xcd;

  // The q_b phase is one tile wider than its workgroup count: tile 0 carries
  // the latent row rather than a GEMM tile. See the kvupd header.
  int const qkv_tiles_per_xcd = BATCH_SIZE * qkv_n_wgs_per_xcd;
  int const qb_tiles_per_xcd = BATCH_SIZE * qb_n_wgs_per_xcd + 1;

  constexpr int HIER_STRIDE = 16;
  int *qkv_barrier = static_cast<int *>(attn_counters_ptr);
  int *qb_barrier = qkv_barrier + 10 * HIER_STRIDE;
  int *decode_barrier = qkv_barrier + 20 * HIER_STRIDE;
  int const arrivals = tiles_per_xcd * 8;

  // Release values are read once, up front, before anything in this layer has
  // run -- the same argument as the MoE task's s_expected block. Reading a
  // release value beside its own arrival atomic races within the last
  // arriving block: a straggler wave there can observe its own thread 0's
  // bump, compute expected = published + 1, and spin forever.
  __shared__ int s_expected[3];
  if (qkv_expected_in < 0) {
    if (tid == 0) {
      s_expected[0] = ld_nt_s32(&qkv_barrier[xcd_id * HIER_STRIDE]) + 1;
      s_expected[1] = ld_nt_s32(&qb_barrier[xcd_id * HIER_STRIDE]) + 1;
      s_expected[2] = ld_nt_s32(&decode_barrier[xcd_id * HIER_STRIDE]) + 1;
    }
    __syncthreads();
  }
  // The override is a kernel argument, so the branch above is block-uniform
  // and the __syncthreads inside it is safe.
  int const qkv_expected = qkv_expected_in < 0 ? s_expected[0] : qkv_expected_in;
  int const qb_expected = qb_expected_in < 0 ? s_expected[1] : qb_expected_in;
  int const decode_expected =
      decode_expected_in < 0 ? s_expected[2] : decode_expected_in;

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  // Slot 4 is ATTN: [0]=qkv_a [1]=qkv barrier [2]=q_b+kvupd [3]=q_b barrier
  // [4]=MLA decode [5]=decode barrier [6]=merge.
  unsigned long long _sp_t0 = __builtin_amdgcn_s_memrealtime();
#endif

  // ══════════════════════════════════════════════════════════════════════
  // Phase 1: residual resolve + input RMSNorm + [q_a_proj | kv_a_proj_with_mqa]
  // ══════════════════════════════════════════════════════════════════════
  // x_ptr is the *previous* layer's pre-MoE residual and moe_ws_f32_ptr its
  // MoE's f32 accumulator; their sum is this layer's input, and used to be
  // produced by a MOE_RESIDUAL_ADD_F32 task of its own -- grid_dim (1,1,1),
  // 47 dispatches per token, 239 of 240 workers idle behind a full event
  // boundary. FUSE_RESADD folds it into the prologue that was going to read
  // the row anyway, which is what gpt-oss's
  // gang_resaddf32_rmsnorm_linear_mxfp4_bias_kernel does. The resolved row
  // still goes out to x_out_ptr, because the o_proj that follows this task
  // adds it back as its own residual.
  //
  // The first layer has no MoE behind it; the demo hands it a zeroed
  // workspace rather than a second task variant, at the cost of one 8 KB
  // read of zeros per token.
  // The GEMM addresses its output as [wg_idx * OPW + ...] within an XCD's
  // column chunk, so it wants that chunk's base. qkv_a_out has to be declared
  // whole here -- Phase 3 norms a prefix of it that spans four XCDs, and
  // reads a latent slice that spans two more -- so the slice the partition
  // map used to supply is reconstructed instead, exactly as the MoE task
  // reconstructs o_proj's.
  if (xcd_rank < qkv_tiles_per_xcd) {
    unsigned short *xcd_out =
        static_cast<unsigned short *>(qkv_a_out_ptr) +
        static_cast<size_t>(xcd_id) * qkv_n_wgs_per_xcd * QKV_OUTPUT_PER_WG;
    gang_rmsnorm_linear_mxfp8_bias_kernel<BATCH_SIZE,
                                          QKV_OUTPUT_PER_WG,
                                          QKV_REDUCTION_SIZE,
                                          QKV_ACTUAL_HIDDEN,
                                          /*WRITE_THROUGH=*/true,
                                          /*FUSE_RESADD=*/true>(
        /*norm_input_ptr=*/x_ptr, // the residual, under FUSE_RESADD
        pre_norm_weight_ptr,
        pre_norm_scratch_ptr,
        qkv_weight_ptr,
        qkv_bias_ptr,
        xcd_out,
        num_active_tokens,
        qkv_n_wgs_per_xcd,
        qkv_output_stride,
        xcd_rank,
        moe_ws_f32_ptr,
        x_out_ptr);
  }

  // ══════════════════════════════════════════════════════════════════════
  // Phase 2: qkv_a -> q_b barrier
  // ══════════════════════════════════════════════════════════════════════
  // __syncthreads is execution-only; s_waitcnt is what retires all 256
  // threads' stores, and it has to precede the release atomic or a waiting
  // XCD can be let through ahead of the data. threadfence_gpu() would also
  // order this and is what a first draft reaches for, but it lowers to
  // buffer_wbl2 sc1 -- a writeback of the whole L2, ~0.16 ms/iter in
  // gpt-oss's measurement. st_wt on the producer side is what makes the
  // cheap fence sufficient.
  __syncthreads();
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#ifdef MPK_ENABLE_SUBPHASE_TIMING
  {
    unsigned long long _t = __builtin_amdgcn_s_memrealtime();
    if (tid == 0 && g_subphase_active) {
      atomicAdd(&g_subphase_ns[4][0], (_t - _sp_t0) * 10);
    }
    _sp_t0 = _t;
  }
#endif
  if (tid == 0) {
    int prev = atom_add_release_gpu_s32(&qkv_barrier[8 * HIER_STRIDE], 1);
    // Modular test rather than a reset: the counter is monotonic for the
    // whole run, so no worker from the next layer can observe a zeroed one.
    if ((prev % arrivals) == arrivals - 1) {
      for (int x = 0; x < 8; x++) {
        st_wt_u32((void *)&qkv_barrier[x * HIER_STRIDE], (unsigned)qkv_expected);
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }
    while (ld_nt_s32(&qkv_barrier[xcd_id * HIER_STRIDE]) < qkv_expected) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
  __syncthreads();
  asm volatile("buffer_inv" ::: "memory");

  // (The MoE accumulator this layer's W2 will use is zeroed by the o_proj task
  // itself, up front, where the Phase 6 W13->W2 barrier already orders it
  // against the first accumulate. It used to be done here.)
#ifdef MPK_ENABLE_SUBPHASE_TIMING
  {
    unsigned long long _t = __builtin_amdgcn_s_memrealtime();
    if (tid == 0 && g_subphase_active) {
      atomicAdd(&g_subphase_ns[4][1], (_t - _sp_t0) * 10);
    }
    _sp_t0 = _t;
  }
#endif

  // ══════════════════════════════════════════════════════════════════════
  // Phase 3: q_a RMSNorm + absorbed q_b + latent KV-cache append
  // ══════════════════════════════════════════════════════════════════════
  // tile_idx 0 is the latent row and returns immediately on every XCD but 0;
  // the GEMM tiles are shifted up by one inside the kernel. Handing it
  // xcd_rank reproduces the standalone dispatch exactly.
  if (xcd_rank < qb_tiles_per_xcd) {
    unsigned short *xcd_q_ws =
        static_cast<unsigned short *>(q_workspace_ptr) +
        static_cast<size_t>(xcd_id) * qb_n_wgs_per_xcd * QB_OUTPUT_PER_WG;
    gang_rmsnorm_linear_mxfp8_bias_mla_kvupd_kernel<BATCH_SIZE,
                                                    QB_OUTPUT_PER_WG,
                                                    QB_REDUCTION_SIZE,
                                                    QB_ACTUAL_HIDDEN,
                                                    KV_LORA_RANK,
                                                    QK_ROPE_HEAD_DIM,
                                                    KV_INPUT_STRIDE,
                                                    KV_CACHE_STRIDE,
                                                    MAX_SEQ_LEN,
                                                    PAGE_SIZE,
                                                    KV_INPUT_OFFSET,
                                                    /*WRITE_THROUGH=*/true>(
        qkv_a_out_ptr,
        q_a_norm_weight_ptr,
        q_a_norm_scratch_ptr,
        qb_weight_ptr,
        qb_bias_ptr,
        qkv_a_out_ptr, // kv_latent: the same row, at KV_INPUT_OFFSET
        kv_norm_weight_ptr,
        cos_ptr,
        sin_ptr,
        xcd_q_ws,
        kv_cache_ptr,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        request_id,
        num_active_tokens,
        qb_n_wgs_per_xcd,
        qb_output_stride,
        xcd_rank,
        kv_eps);
  }

  // ══════════════════════════════════════════════════════════════════════
  // Phase 4: q_b -> decode barrier
  // ══════════════════════════════════════════════════════════════════════
  __syncthreads();
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#ifdef MPK_ENABLE_SUBPHASE_TIMING
  {
    unsigned long long _t = __builtin_amdgcn_s_memrealtime();
    if (tid == 0 && g_subphase_active) {
      atomicAdd(&g_subphase_ns[4][2], (_t - _sp_t0) * 10);
    }
    _sp_t0 = _t;
  }
#endif
  if (tid == 0) {
    int prev = atom_add_release_gpu_s32(&qb_barrier[8 * HIER_STRIDE], 1);
    if ((prev % arrivals) == arrivals - 1) {
      for (int x = 0; x < 8; x++) {
        st_wt_u32((void *)&qb_barrier[x * HIER_STRIDE], (unsigned)qb_expected);
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }
  }
  // Only the decode ranks need what this barrier protects. Everyone else
  // falls through to the decode barrier's arrival, which is what orders them
  // against the merge.
  if (xcd_rank < mla_tiles_per_xcd) {
    if (tid == 0) {
      while (ld_nt_s32(&qb_barrier[xcd_id * HIER_STRIDE]) < qb_expected) {
        __builtin_amdgcn_s_sleep(1);
      }
    }
    __syncthreads();
    asm volatile("buffer_inv" ::: "memory");
#ifdef MPK_ENABLE_SUBPHASE_TIMING
    {
      unsigned long long _t = __builtin_amdgcn_s_memrealtime();
      if (tid == 0 && g_subphase_active) {
        atomicAdd(&g_subphase_ns[4][3], (_t - _sp_t0) * 10);
      }
      _sp_t0 = _t;
    }
#endif

    // ════════════════════════════════════════════════════════════════════
    // Phase 5: absorbed MLA decode, split over (q_head_group, kv_chunk)
    // ════════════════════════════════════════════════════════════════════
    // The decode wants a *global* work-item index, not a per-XCD tile: it
    // decomposes it into (q_head_group, kv_chunk, request). Synthesizing it
    // against mla_tiles_per_xcd rather than tiles_per_xcd reproduces the
    // standalone task's mapping, where the gang dispatch width was the
    // decode's own.
    gang_mla_decode_kernel<bfloat16,
                           NUM_Q_HEADS,
                           KV_LORA_RANK,
                           QK_ROPE_HEAD_DIM,
                           PAGE_SIZE,
                           MAX_SEQ_LEN,
                           NUM_KV_CHUNKS,
                           Q_WORKSPACE_STRIDE,
                           KV_CACHE_STRIDE,
                           /*WRITE_THROUGH=*/true>(
        q_workspace_ptr,
        kv_cache_ptr,
        o_acc_ptr,
        lse_ptr,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        mla_total_work_items,
        xcd_id * mla_tiles_per_xcd + xcd_rank,
        scale_s);
#ifdef MPK_ENABLE_SUBPHASE_TIMING
    {
      unsigned long long _t = __builtin_amdgcn_s_memrealtime();
      if (tid == 0 && g_subphase_active) {
        atomicAdd(&g_subphase_ns[4][4], (_t - _sp_t0) * 10);
      }
      _sp_t0 = _t;
    }
#endif
  }

  // ══════════════════════════════════════════════════════════════════════
  // Phase 6: decode -> merge barrier
  // ══════════════════════════════════════════════════════════════════════
  // Every worker arrives; only the merge ranks wait. The rest return here, a
  // whole merge stage early, rather than spinning -- the same trade the MoE
  // task makes at its W13 -> W2 barrier.
  __syncthreads();
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  if (tid == 0) {
    int prev = atom_add_release_gpu_s32(&decode_barrier[8 * HIER_STRIDE], 1);
    if ((prev % arrivals) == arrivals - 1) {
      for (int x = 0; x < 8; x++) {
        st_wt_u32((void *)&decode_barrier[x * HIER_STRIDE],
                  (unsigned)decode_expected);
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }
  }
  if (xcd_rank >= merge_tiles_per_xcd) {
    return;
  }
  if (tid == 0) {
    while (ld_nt_s32(&decode_barrier[xcd_id * HIER_STRIDE]) < decode_expected) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
  __syncthreads();
  asm volatile("buffer_inv" ::: "memory");
#ifdef MPK_ENABLE_SUBPHASE_TIMING
  {
    unsigned long long _t = __builtin_amdgcn_s_memrealtime();
    if (tid == 0 && g_subphase_active) {
      atomicAdd(&g_subphase_ns[4][5], (_t - _sp_t0) * 10);
    }
    _sp_t0 = _t;
  }
#endif

  // ══════════════════════════════════════════════════════════════════════
  // Phase 7: split-KV merge
  // ══════════════════════════════════════════════════════════════════════
  // merge_task_offset is (q_group * MERGE_DIM_SPLITS + dim_slice), which the
  // kernel decomposes itself; all this has to supply is a bijection onto
  // [0, NUM_Q_GROUPS * MERGE_DIM_SPLITS). The standalone task got it from
  // bid.y of a (requests, 32, 1) grid -- there is no bid.y in a gang task, so
  // it comes from the worker's own coordinates instead.
  constexpr int NUM_Q_GROUPS = NUM_Q_HEADS / 16;
  merge_splitkv_ck_fmha<bfloat16,
                        /*NUM_QO_HEADS_PER_KV=*/16,
                        NUM_Q_GROUPS,
                        /*HEAD_DIM=*/KV_LORA_RANK,
                        NUM_KV_CHUNKS,
                        /*KV_CHUNK_SIZE=*/128,
                        PAGE_SIZE,
                        MERGE_WRITE_THROUGH,
                        MERGE_DIM_SPLITS>(
      reinterpret_cast<float const *>(lse_ptr),
      reinterpret_cast<float const *>(o_acc_ptr),
      qo_indptr,
      kv_indptr,
      kv_last_page_len,
      request_id,
      reinterpret_cast<bfloat16 *>(attn_out_ptr),
      xcd_id * merge_tiles_per_xcd + xcd_rank);
#ifdef MPK_ENABLE_SUBPHASE_TIMING
  {
    unsigned long long _t = __builtin_amdgcn_s_memrealtime();
    if (tid == 0 && g_subphase_active) {
      atomicAdd(&g_subphase_ns[4][6], (_t - _sp_t0) * 10);
      atomicAdd(&g_subphase_cnt[4], 1ULL);
    }
  }
#endif
}

} // namespace kernel
