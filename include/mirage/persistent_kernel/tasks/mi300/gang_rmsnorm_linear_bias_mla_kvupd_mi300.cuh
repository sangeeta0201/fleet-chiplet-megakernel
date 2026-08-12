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

// Step 2, verbatim from mla_kv_cache_update_impl minus the Q half. Kept
// noinline so its LDS and register pressure stay off the 287 workers that
// never call it.
template <int KV_LORA_RANK,
          int QK_ROPE_HEAD_DIM,
          int KV_INPUT_STRIDE,
          int KV_CACHE_STRIDE,
          int MAX_SEQ_LEN,
          int PAGE_SIZE,
          int KV_INPUT_OFFSET>
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

  static_assert(QK_ROPE_HEAD_DIM % 2 == 0, "rope dim must be even");
  static_assert(KV_CACHE_STRIDE >= KV_LORA_RANK + QK_ROPE_HEAD_DIM,
                "cache row must hold c_kv and k_rope");
  static_assert(KV_INPUT_OFFSET + KV_LORA_RANK + QK_ROPE_HEAD_DIM <=
                    KV_INPUT_STRIDE,
                "latent slice must fit in the projection row");

  int const req = request_id;
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

  int const tid = threadIdx.x;
  int const warp_idx = tid >> 6;
  int const lane_idx = tid & 63;

  __shared__ int page_indices[MAX_PAGES_PER_REQUEST];
  __shared__ float s_reduce[NUM_WARPS];
  for (int i = tid; i < num_pages; i += NUM_THREADS) {
    page_indices[i] = kv_indices[first_page_pos + i];
  }

  bf16 const *__restrict__ d_kv = reinterpret_cast<bf16 const *>(kv_latent_ptr) +
                                  (long)first_token_pos * KV_INPUT_STRIDE +
                                  KV_INPUT_OFFSET;
  bf16 *__restrict__ d_cache = reinterpret_cast<bf16 *>(paged_kv_cache_ptr);
  bf16 const *__restrict__ kv_weight =
      reinterpret_cast<bf16 const *>(kv_norm_weight_ptr);
  bf16 const *__restrict__ d_cos = reinterpret_cast<bf16 const *>(cos_ptr);
  bf16 const *__restrict__ d_sin = reinterpret_cast<bf16 const *>(sin_ptr);

  __syncthreads();

  for (int token = 0; token < num_tokens; token++) {
    int const pos = global_seq_len - num_tokens + token;
    int const page_idx = page_indices[pos / PAGE_SIZE];
    int const dst_row = page_idx * PAGE_SIZE + (pos % PAGE_SIZE);
    bf16 const *src = d_kv + (long)token * KV_INPUT_STRIDE;
    bf16 *dst = d_cache + (long)dst_row * KV_CACHE_STRIDE;

    float sum_sq = 0.0f;
    for (int i = tid; i < KV_LORA_RANK; i += NUM_THREADS) {
      float const val = __cvt_bf16_to_f32_mla(src[i]);
      sum_sq += val * val;
    }
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

    for (int i = tid; i < KV_LORA_RANK; i += NUM_THREADS) {
      float const val = __cvt_bf16_to_f32_mla(src[i]) * rms_rcp *
                        __cvt_bf16_to_f32_mla(kv_weight[i]);
      dst[i] = static_cast<bf16>(val);
    }

    bf16 const *cos_data = d_cos + (long)pos * QK_ROPE_HEAD_DIM;
    bf16 const *sin_data = d_sin + (long)pos * QK_ROPE_HEAD_DIM;
    for (int j = tid; j < ROPE_HALF; j += NUM_THREADS) {
      float const x0 = __cvt_bf16_to_f32_mla(src[KV_LORA_RANK + 2 * j]);
      float const x1 = __cvt_bf16_to_f32_mla(src[KV_LORA_RANK + 2 * j + 1]);
      float const c = __cvt_bf16_to_f32_mla(cos_data[j]);
      float const s = __cvt_bf16_to_f32_mla(sin_data[j]);
      dst[KV_LORA_RANK + j] = static_cast<bf16>(x0 * c - x1 * s);
      dst[KV_LORA_RANK + ROPE_HALF + j] = static_cast<bf16>(x1 * c + x0 * s);
    }
    // The next token's reduction overwrites s_reduce.
    __syncthreads();
  }
}

// Step 1's remainder: rotate this worker's rope tile where it already sits.
// Only the ROPE_HALF pairing crosses lanes, and it stays inside the tile, so
// the read set is captured in registers and one barrier separates it from the
// write set -- (2j, 2j+1) read, (j, j + ROPE_HALF) written.
template <int QK_ROPE_HEAD_DIM>
__device__ __forceinline__ void rope_tile_inplace(bf16 *tile,
                                                  bf16 const *cos_data,
                                                  bf16 const *sin_data) {
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
  if (tid < ROPE_HALF) {
    tile[tid] = static_cast<bf16>(x0 * c - x1 * s);
    tile[tid + ROPE_HALF] = static_cast<bf16>(x1 * c + x0 * s);
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
