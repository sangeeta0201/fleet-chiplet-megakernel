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

// Fused q_a_layernorm + absorbed q_b_proj + MLA KV cache update.
//
// This is the MLA counterpart of gang_rmsnorm_linear_mxfp4_bias_kvupd_kernel:
// it folds what used to be a separately dispatched MLA_KV_CACHE_UPDATE task
// into the epilogue of the GEMM that feeds it. That task was the worst cost
// per unit of work in the decode graph -- 377 us/token across 47 layers for
// ~37 KB of traffic -- because it ran as a *single* workgroup that all 240
// workers had to wait behind, plus a dispatch and two event barriers.
//
// The old task did two independent things (mla_kv_cache_update_mi300.cuh):
//
//   Step 1  q_absorbed -> q_workspace: bit-copy the absorbed c-part of each
//           head, partial interleaved RoPE over the trailing rope slice
//   Step 2  kv_latent -> paged cache: kv_a_layernorm over c_kv, RoPE over
//           k_rope, append the row at the current position
//
// Neither survives here as a pass of its own.
//
// Step 1 disappears entirely. It was a copy of the very buffer this GEMM had
// just written, so the GEMM now writes q_workspace directly and q_absorbed
// stops existing. What is left of it is the RoPE, and the tiling makes that
// nearly free: a head is QK_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM columns, and
// with GLM's 512 + 64 against tile_n = 64 the rope slice of every head is
// *exactly one tile*. So one worker in nine owns a whole rope slice, alone,
// and rotates its own 64 values in place after the MFMA. The other eight are
// already correct as written. No cross-workgroup exchange, and none of the
// LDS staging the GQA version needs -- there the rope span is a full HEAD_DIM
// spread over several waves.
//
// Step 2 gets a dispatch slot of its own inside this task -- tile 0, with the
// GEMM's tiles shifted up by one -- rather than becoming a phase or riding on
// a worker that also has a tile. It reads kv_latent, which the *previous* task
// produced, so it depends on nothing here and needs no barrier; on its own
// slot it simply runs alongside the other 288 workers' MFMA. Bolting it onto a
// worker that also owns a tile does not work, and measurably did not: that
// worker then finishes ~5 us behind its peers and the gang waits for it, which
// is the same stall the standalone task caused, just relocated. The event
// between this task and gang_mla_decode orders the cache write against the
// read, exactly as it ordered the old task.
//
// The arithmetic is unchanged from mla_kv_cache_update_impl, including the
// 1e-6 kv_a_layernorm epsilon -- this is a scheduling change and should be
// bit-identical.

#pragma once

#include "gang_linear_mi300.cuh"
#include "gang_rmsnorm_linear_bias_mi300.cuh"
#include "mla_kv_cache_update_mi300.cuh"
#include <hip/hip_bf16.h>

namespace kernel {

namespace gang_mla_kvupd_detail {
using bf16 = __hip_bfloat16;

// Flatten latent_to_cache's index chase (see the block comment inside it).
//
// MEASURED NULL, and defaulted off for that reason. NP=4, bs=1, devices 4-7,
// paired same session, n=6 each:
//
//   MPK_KVUPD_FAST=1   11.221 ms/tok  sd 0.045
//   MPK_KVUPD_FAST=0   11.183 ms/tok  sd 0.048   <- shipped
//
// +0.038 with t = 1.4, i.e. inside the 0.26 ms noise floor in the wrong
// direction. The transform itself is real -- four dependent HBM round trips
// become two, one full re-read of the latent row disappears, and a
// __syncthreads goes away -- but it buys the wall nothing, and the reason is
// in this file's own header comment: step 2 was given a dispatch slot of its
// own (tile 0) precisely so it runs *alongside* the other 288 workers' MFMA.
// It is not the last arriver of the q_b segment, so shortening it moves
// nothing. Kept behind the flag as a priced null, not deleted -- if the q_b
// phase is ever re-tiled such that tile 0 does set the arrival, this is the
// version to switch on.
#ifndef MPK_KVUPD_FAST
#define MPK_KVUPD_FAST 0
#endif

// Step 2, verbatim from mla_kv_cache_update_impl minus the Q half. Kept
// noinline so its LDS and register pressure stay off the 287 workers that
// never call it.
// Store one bf16 into the latent cache row.
//
// The default lands in this XCD's L2; the task graph's event boundary is what
// republishes it to the other seven. A fused caller has no such boundary and
// every XCD's MLA decode reads this row, so it asks for WRITE_THROUGH.
template <bool WRITE_THROUGH>
__device__ __forceinline__ void cache_store(bf16 *dst, float val) {
  if constexpr (WRITE_THROUGH) {
    bf16 b = static_cast<bf16>(val);
    unsigned short raw;
    __builtin_memcpy(&raw, &b, 2);
    st_wt_u16((void *)dst, raw);
  } else {
    *dst = static_cast<bf16>(val);
  }
}

template <int KV_LORA_RANK,
          int QK_ROPE_HEAD_DIM,
          int KV_INPUT_STRIDE,
          int KV_CACHE_STRIDE,
          int MAX_SEQ_LEN,
          int PAGE_SIZE,
          int KV_INPUT_OFFSET,
          bool WRITE_THROUGH = false>
__device__ __attribute__((noinline)) void
    latent_to_cache(void const *kv_latent_ptr,
                    void *paged_kv_cache_ptr,
                    int const *qo_indptr,
                    int const *kv_indptr,
                    int const *kv_indices,
                    int const *kv_last_page_len,
                    int16_t request_id,
                    void const *kv_norm_weight_ptr,
                    void const *cos_ptr,
                    void const *sin_ptr,
                    float kv_eps) {
  constexpr int ROPE_HALF = QK_ROPE_HEAD_DIM / 2;
  constexpr int NUM_THREADS = 256;
  constexpr int NUM_WARPS = NUM_THREADS / 64;
  constexpr int MAX_PAGES_PER_REQUEST =
      (MAX_SEQ_LEN + PAGE_SIZE - 1) / PAGE_SIZE;
  constexpr int PER_THREAD = (KV_LORA_RANK + NUM_THREADS - 1) / NUM_THREADS;

  static_assert(QK_ROPE_HEAD_DIM % 2 == 0, "rope dim must be even");
  static_assert(KV_CACHE_STRIDE >= KV_LORA_RANK + QK_ROPE_HEAD_DIM,
                "cache row must hold c_kv and k_rope");
  static_assert(KV_INPUT_OFFSET + KV_LORA_RANK + QK_ROPE_HEAD_DIM <=
                    KV_INPUT_STRIDE,
                "latent slice must fit in the projection row");

  int const req = request_id;

#if MPK_KVUPD_FAST
  // The index chase, flattened.
  //
  // All five of these are indexed by `req` alone, so they are mutually
  // independent -- but the old form put the qo_indptr equality test in front
  // of the kv_indptr / kv_last_page_len loads, and vmcnt is in-order, so the
  // compare forced an s_waitcnt vmcnt(0) and the second group paid a second
  // cold HBM round trip. Issuing all five above the branch collapses that to
  // one wait. The shipped ISA showed the serialized form directly:
  //
  //   flat_load_dwordx2 v[4:5], v[4:5]        ; qo_indptr[req], [req+1]
  //   s_waitcnt vmcnt(0) lgkmcnt(0)           ; <- the branch's fault
  //   v_cmp_ne_u32_e32 vcc, v5, v4
  //   s_and_saveexec_b64 s[6:7], vcc
  //   flat_load_dwordx2 v[6:7], v[6:7]        ; kv_indptr, a full trip later
  //   flat_load_dword   v12, v[10:11]
  //   s_waitcnt vmcnt(0) lgkmcnt(0)
  int const first_token_pos = qo_indptr[req];
  int const last_token_pos = qo_indptr[req + 1];
  int const first_page_pos = kv_indptr[req];
  int const last_page_pos = kv_indptr[req + 1];
  int const last_page_len = kv_last_page_len[req];
  if (first_token_pos == last_token_pos) {
    return;
  }
  int const num_tokens = last_token_pos - first_token_pos;
  int const num_pages = last_page_pos - first_page_pos;
  int const global_seq_len = (num_pages - 1) * PAGE_SIZE + last_page_len;
#else
  int const first_token_pos = qo_indptr[req];
  int const last_token_pos = qo_indptr[req + 1];
  if (first_token_pos == last_token_pos) {
    return;
  }
  int const num_tokens = last_token_pos - first_token_pos;

  int const first_page_pos = kv_indptr[req];
  int const num_pages = kv_indptr[req + 1] - first_page_pos;
  int const global_seq_len =
      (num_pages - 1) * PAGE_SIZE + kv_last_page_len[req];
#endif

  int const tid = threadIdx.x;
  int const warp_idx = tid >> 6;
  int const lane_idx = tid & 63;

  __shared__ float s_reduce[NUM_WARPS];
#if !MPK_KVUPD_FAST
  __shared__ int page_indices[MAX_PAGES_PER_REQUEST];
  for (int i = tid; i < num_pages; i += NUM_THREADS) {
    page_indices[i] = kv_indices[first_page_pos + i];
  }
#else
  (void)num_pages;
#endif

  bf16 const *__restrict__ d_kv = reinterpret_cast<bf16 const *>(kv_latent_ptr) +
                                  (long)first_token_pos * KV_INPUT_STRIDE +
                                  KV_INPUT_OFFSET;
  bf16 *__restrict__ d_cache = reinterpret_cast<bf16 *>(paged_kv_cache_ptr);
  bf16 const *__restrict__ kv_weight =
      reinterpret_cast<bf16 const *>(kv_norm_weight_ptr);
  bf16 const *__restrict__ d_cos = reinterpret_cast<bf16 const *>(cos_ptr);
  bf16 const *__restrict__ d_sin = reinterpret_cast<bf16 const *>(sin_ptr);

#if !MPK_KVUPD_FAST
  __syncthreads();
#endif

  for (int token = 0; token < num_tokens; token++) {
    int const pos = global_seq_len - num_tokens + token;
#if MPK_KVUPD_FAST
    // Straight to the one page this token lands on. Staging the whole
    // MAX_PAGES_PER_REQUEST table in LDS bought a decode nothing -- it read
    // exactly one entry back out, behind a __syncthreads.
    int const page_idx = kv_indices[first_page_pos + pos / PAGE_SIZE];
#else
    int const page_idx = page_indices[pos / PAGE_SIZE];
#endif
    int const dst_row = page_idx * PAGE_SIZE + (pos % PAGE_SIZE);
    bf16 const *src = d_kv + (long)token * KV_INPUT_STRIDE;
    bf16 *dst = d_cache + (long)dst_row * KV_CACHE_STRIDE;

    float sum_sq = 0.0f;
#if MPK_KVUPD_FAST
    // Hold the row in registers: PER_THREAD is 2 at GLM's 512/256. The old
    // form read src[] twice, once for the sum of squares and once for the
    // scaled store, and the second pass could not issue until rms_rcp was
    // known -- so it was a second full latency, not just a second KB.
    float xv[PER_THREAD];
#pragma unroll
    for (int u = 0; u < PER_THREAD; u++) {
      int const i = tid + u * NUM_THREADS;
      xv[u] = (i < KV_LORA_RANK) ? __cvt_bf16_to_f32_mla(src[i]) : 0.0f;
      sum_sq += xv[u] * xv[u];
    }
    // The rope operands depend on nothing the reduction produces, so pull
    // them in now and let them fly under the shuffle chain and the barrier.
    bf16 const *cos_data = d_cos + (long)pos * QK_ROPE_HEAD_DIM;
    bf16 const *sin_data = d_sin + (long)pos * QK_ROPE_HEAD_DIM;
    bool const has_rope = tid < ROPE_HALF;
    float rx0 = 0.0f, rx1 = 0.0f, rc = 0.0f, rs = 0.0f;
    if (has_rope) {
      rx0 = __cvt_bf16_to_f32_mla(src[KV_LORA_RANK + 2 * tid]);
      rx1 = __cvt_bf16_to_f32_mla(src[KV_LORA_RANK + 2 * tid + 1]);
      rc = __cvt_bf16_to_f32_mla(cos_data[tid]);
      rs = __cvt_bf16_to_f32_mla(sin_data[tid]);
    }
    static_assert(ROPE_HALF <= NUM_THREADS,
                  "the hoisted rope operands assume one pair per thread");
#else
    for (int i = tid; i < KV_LORA_RANK; i += NUM_THREADS) {
      float const val = __cvt_bf16_to_f32_mla(src[i]);
      sum_sq += val * val;
    }
#endif
#pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
      sum_sq += __shfl_xor(sum_sq, offset);
    }
    if (lane_idx == 0) {
      s_reduce[warp_idx] = sum_sq;
    }
    __syncthreads();
    sum_sq = 0.0f;
#pragma unroll
    for (int w = 0; w < NUM_WARPS; w++) {
      sum_sq += s_reduce[w];
    }
    float const rms_rcp = rsqrtf(sum_sq / float(KV_LORA_RANK) + kv_eps);

#if MPK_KVUPD_FAST
#pragma unroll
    for (int u = 0; u < PER_THREAD; u++) {
      int const i = tid + u * NUM_THREADS;
      if (i < KV_LORA_RANK) {
        cache_store<WRITE_THROUGH>(
            &dst[i], xv[u] * rms_rcp * __cvt_bf16_to_f32_mla(kv_weight[i]));
      }
    }
    if (has_rope) {
      cache_store<WRITE_THROUGH>(&dst[KV_LORA_RANK + tid],
                                 rx0 * rc - rx1 * rs);
      cache_store<WRITE_THROUGH>(&dst[KV_LORA_RANK + ROPE_HALF + tid],
                                 rx1 * rc + rx0 * rs);
    }
#else
    for (int i = tid; i < KV_LORA_RANK; i += NUM_THREADS) {
      float const val = __cvt_bf16_to_f32_mla(src[i]) * rms_rcp *
                        __cvt_bf16_to_f32_mla(kv_weight[i]);
      cache_store<WRITE_THROUGH>(&dst[i], val);
    }

    bf16 const *cos_data = d_cos + (long)pos * QK_ROPE_HEAD_DIM;
    bf16 const *sin_data = d_sin + (long)pos * QK_ROPE_HEAD_DIM;
    for (int j = tid; j < ROPE_HALF; j += NUM_THREADS) {
      float const x0 = __cvt_bf16_to_f32_mla(src[KV_LORA_RANK + 2 * j]);
      float const x1 = __cvt_bf16_to_f32_mla(src[KV_LORA_RANK + 2 * j + 1]);
      float const c = __cvt_bf16_to_f32_mla(cos_data[j]);
      float const s = __cvt_bf16_to_f32_mla(sin_data[j]);
      cache_store<WRITE_THROUGH>(&dst[KV_LORA_RANK + j], x0 * c - x1 * s);
      cache_store<WRITE_THROUGH>(&dst[KV_LORA_RANK + ROPE_HALF + j],
                                 x1 * c + x0 * s);
    }
#endif
    // The next token's reduction overwrites s_reduce.
    __syncthreads();
  }
}

// Step 1's remainder: rotate this worker's rope tile where it already sits.
// Only the ROPE_HALF pairing crosses lanes, and it stays inside the tile, so
// the read set is captured in registers and one barrier separates it from the
// write set -- (2j, 2j+1) read, (j, j + ROPE_HALF) written.
//
// `out` lands the rotated tile somewhere other than where it was read from,
// which is what the un-absorbed q_b needs: there the GEMM's output row is the
// 256-wide [nope | rope] scratch W_UK reduces over, while MLA decode wants the
// roped 64 in the 576-wide query row. Since the read set is already in
// registers before the barrier, redirecting the write costs nothing.
template <int QK_ROPE_HEAD_DIM, bool WRITE_THROUGH = false>
__device__ __forceinline__ void rope_tile_inplace(bf16 *tile,
                                                  bf16 const *cos_data,
                                                  bf16 const *sin_data,
                                                  bf16 *out = nullptr) {
  constexpr int ROPE_HALF = QK_ROPE_HEAD_DIM / 2;
  int const tid = threadIdx.x;

  float x0 = 0.0f, x1 = 0.0f, c = 0.0f, s = 0.0f;
  if (tid < ROPE_HALF) {
    x0 = __cvt_bf16_to_f32_mla(tile[2 * tid]);
    x1 = __cvt_bf16_to_f32_mla(tile[2 * tid + 1]);
    c = __cvt_bf16_to_f32_mla(cos_data[tid]);
    s = __cvt_bf16_to_f32_mla(sin_data[tid]);
  }
  __syncthreads();
  bf16 *dst = out ? out : tile;
  if (tid < ROPE_HALF) {
    cache_store<WRITE_THROUGH>(&dst[tid], x0 * c - x1 * s);
    cache_store<WRITE_THROUGH>(&dst[tid + ROPE_HALF], x1 * c + x0 * s);
  }
}

} // namespace gang_mla_kvupd_detail

// See the file header. Template parameters up to NORM_SPAN are
// gang_rmsnorm_linear_bias_kernel's; the rest describe the latent row.
template <typename T,
          int BATCH_SIZE,
          int REDUCTION_SIZE,
          int ACTUAL_HIDDEN_DIM,
          int NORM_SPAN,
          int KV_LORA_RANK,
          int QK_ROPE_HEAD_DIM,
          int KV_INPUT_STRIDE,
          int KV_CACHE_STRIDE,
          int MAX_SEQ_LEN,
          int PAGE_SIZE,
          int KV_INPUT_OFFSET>
__device__ __attribute__((noinline)) void
    gang_rmsnorm_linear_bias_mla_kvupd_kernel(
        void const *norm_input_ptr,    // [batch, REDUCTION_SIZE] (q_a half)
        void const *norm_weight_ptr,   // [REDUCTION_SIZE]
        void *norm_output_ptr,         // [batch, REDUCTION_SIZE] scratch
        void const *linear_weight_ptr, // [chunk_N, REDUCTION_SIZE]
        void const *bias_ptr,          // [1, full_N]
        void const *kv_latent_ptr,     // [batch, KV_INPUT_STRIDE] from kv_a
        void const *kv_norm_weight_ptr, // [KV_LORA_RANK]
        void const *cos_ptr,
        void const *sin_ptr,
        void *q_workspace_ptr,      // [batch, o_stride], was q_absorbed
        void *paged_kv_cache_ptr,   // [pages, PAGE_SIZE, 1, KV_CACHE_STRIDE]
        int const *qo_indptr,
        int const *kv_indptr,
        int const *kv_indices,
        int const *kv_last_page_len,
        int16_t request_id,
        int num_active_tokens,
        int tile_n,
        int o_stride,
        int m_tiles,
        int n_tiles,
        int wgm,
        int tile_idx,
        float kv_eps) {
  using bf16 = __hip_bfloat16;
  constexpr int QK_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM;

  // Tile 0 carries the latent row and nothing else; the GEMM's own tiles are
  // shifted up by one. Handing it a dispatch slot of its own is the whole
  // point -- bolted onto a worker that also has a tile it just runs ahead of
  // that worker's MFMA, so the worker finishes behind its peers and the gang
  // waits for it exactly as it waited for the old standalone task. On its own
  // slot it overlaps everyone else's MFMA and costs nothing. The other seven
  // XCDs' tile 0 is dead, which is seven dispatch slots against 288 real
  // ones.
  if (tile_idx == 0) {
    if (gang_rmsnorm_topk_detail::get_xcd_id() != 0) {
      return;
    }
    gang_mla_kvupd_detail::latent_to_cache<KV_LORA_RANK,
                                           QK_ROPE_HEAD_DIM,
                                           KV_INPUT_STRIDE,
                                           KV_CACHE_STRIDE,
                                           MAX_SEQ_LEN,
                                           PAGE_SIZE,
                                           KV_INPUT_OFFSET>(kv_latent_ptr,
                                                            paged_kv_cache_ptr,
                                                            qo_indptr,
                                                            kv_indptr,
                                                            kv_indices,
                                                            kv_last_page_len,
                                                            request_id,
                                                            kv_norm_weight_ptr,
                                                            cos_ptr,
                                                            sin_ptr,
                                                            kv_eps);
    return;
  }
  int const gemm_tile_idx = tile_idx - 1;

  gang_rmsnorm_linear_bias_kernel<T,
                                  BATCH_SIZE,
                                  REDUCTION_SIZE,
                                  ACTUAL_HIDDEN_DIM,
                                  NORM_SPAN>(norm_input_ptr,
                                             norm_weight_ptr,
                                             norm_output_ptr,
                                             linear_weight_ptr,
                                             bias_ptr,
                                             q_workspace_ptr,
                                             num_active_tokens,
                                             tile_n,
                                             o_stride,
                                             m_tiles,
                                             n_tiles,
                                             wgm,
                                             gemm_tile_idx);

  // Does this worker own a rope slice? A head spans QK_DIM columns and the
  // slice is its last QK_ROPE_HEAD_DIM, so with tile_n dividing both the test
  // is positional and needs no knowledge of which XCD we are on -- QK_DIM
  // divides the per-XCD chunk.
  int m_tile, n_tile;
  if (!gang_linear_tile_coords(
          gemm_tile_idx, m_tiles, n_tiles, wgm, &m_tile, &n_tile)) {
    return;
  }
  int const tiles_per_head = QK_DIM / tile_n;
  if (n_tile % tiles_per_head != tiles_per_head - 1) {
    return;
  }

  int const req = request_id;
  int const first_token_pos = qo_indptr[req];
  int const num_tokens = qo_indptr[req + 1] - first_token_pos;
  if (num_tokens == 0) {
    return;
  }
  int const first_page_pos = kv_indptr[req];
  int const global_seq_len =
      (kv_indptr[req + 1] - first_page_pos - 1) * PAGE_SIZE +
      kv_last_page_len[req];

  // Make this workgroup's own MFMA stores visible to its loads below: the CK
  // epilogue writes non-temporally, so the barrier alone is not enough.
  __syncthreads();
  asm volatile("buffer_inv" ::: "memory");

  bf16 const *d_cos = reinterpret_cast<bf16 const *>(cos_ptr);
  bf16 const *d_sin = reinterpret_cast<bf16 const *>(sin_ptr);
  bf16 *tile_base = reinterpret_cast<bf16 *>(q_workspace_ptr) +
                    (long)m_tile * BATCH_SIZE * o_stride +
                    (long)n_tile * tile_n;

  for (int r = 0; r < BATCH_SIZE; r++) {
    int const row = m_tile * BATCH_SIZE + r;
    if (row < first_token_pos || row >= first_token_pos + num_tokens ||
        row >= num_active_tokens) {
      continue;
    }
    int const pos = global_seq_len - num_tokens + (row - first_token_pos);
    gang_mla_kvupd_detail::rope_tile_inplace<QK_ROPE_HEAD_DIM>(
        tile_base + (long)r * o_stride,
        d_cos + (long)pos * QK_ROPE_HEAD_DIM,
        d_sin + (long)pos * QK_ROPE_HEAD_DIM);
  }
}

} // namespace kernel
