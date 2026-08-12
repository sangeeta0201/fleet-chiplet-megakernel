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

// Latent KV cache update for absorbed MLA (GLM-5, GLM-4.7-Flash).
//
// This is Phase A of the MLA attention path, the counterpart of
// kv_cache_update_impl in the GQA path:
//   Phase A: kv_a_layernorm + partial interleaved RoPE, append one latent row
//            to the paged cache, write the roped Q to a workspace  <-- here
//   Phase B: gang_mla_decode_kernel
//   Phase C: merge_splitkv_ck_fmha, when NUM_KV_CHUNKS > 1
//
// It does *not* fit kv_cache_update_impl, which assumes separate K and V
// caches of NUM_KV_HEADS x HEAD_DIM each. Under absorption there is a single
// latent row per token,
//
//     kv_row = [ c_kv (KV_LORA_RANK) | k_rope (QK_ROPE_HEAD_DIM) ]
//
// and V is the leading KV_LORA_RANK dims of that same row rather than a
// tensor of its own. This is the exact row layout gang_mla_decode_mi300.cuh
// reads, so the two files share a dtype (plain __hip_bfloat16, no ck_tile) and
// the same v_cvt_f32_bf16 convert idiom borrowed from
// paged_attention_decode_minimal_hd64_mi300.cuh.
//
// Both inputs arrive already projected:
//   q_absorbed  [tokens, NUM_QO_HEADS * QK_DIM]  from q_b_proj with W_UK
//                                                folded in, so each head is
//                                                [q_nope @ W_UK | q_rope]
//   kv_latent   [tokens, KV_INPUT_STRIDE]        from kv_a_proj_with_mqa,
//                                                un-normalised, starting at
//                                                column KV_INPUT_OFFSET (the
//                                                demo fuses q_a and kv_a into
//                                                one GEMM, so the latent is
//                                                the tail of a wider row)
// so the work here is norm + RoPE + placement, no GEMM. kv_a_layernorm has to
// happen on this side of the cache, not on the read side: kv_b_proj (hence
// W_UK / W_UV) is linear in the *normalised* latent, which is what makes the
// absorption exact.
//
// Three differences from the GQA Phase A worth noting:
//   * Q carries no norm. q_a_layernorm was already applied by the preceding
//     fused rmsnorm+linear task, and MLA has no per-head Q norm.
//   * RoPE is partial and interleaved: only the trailing QK_ROPE_HEAD_DIM dims
//     rotate, pairing (2j, 2j+1) and writing to (j, j + ROPE/2). See
//     rope_interleave_partial in rotary_embedding_mi300.cuh for why that
//     output permutation is the matching one.
//   * Nothing is staged in LDS. The GQA version needs shared memory because it
//     normalises Q and K in place; here the RoPE source (q_absorbed /
//     kv_latent) and destination (q_workspace / the cache) are distinct
//     buffers, so the read and write index sets cannot collide and the task is
//     a streaming copy plus one block-wide reduction for the norm.

#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

namespace kernel {

// bf16 -> f32 via the gfx950 convert, matching __load_bf16x4_to_fp16's idiom.
// The instruction reads the bf16 from the low half of a 32-bit lane.
__device__ __forceinline__ float __cvt_bf16_to_f32_mla(__hip_bfloat16 x) {
  unsigned w = *reinterpret_cast<unsigned short const *>(&x);
  float r;
  asm("v_cvt_f32_bf16 %0, %1" : "=v"(r) : "v"(w));
  return r;
}

template <typename T,
          int NUM_QO_HEADS,
          int KV_LORA_RANK,
          int QK_ROPE_HEAD_DIM,
          int Q_INPUT_STRIDE,
          int KV_INPUT_STRIDE,
          int KV_CACHE_STRIDE,
          int MAX_SEQ_LEN,
          int PAGE_SIZE,
          int Q_WORKSPACE_STRIDE,
          int KV_INPUT_OFFSET = 0>
__device__ __noinline__ void
    mla_kv_cache_update_impl(void const *q_absorbed_ptr,
                             void const *kv_latent_ptr,
                             void *paged_kv_cache_ptr,
                             void *q_workspace_ptr,
                             int const *qo_indptr,
                             int const *kv_indptr,
                             int const *kv_indices,
                             int const *kv_last_page_len,
                             int16_t request_id,
                             void const *kv_norm_weight_ptr,
                             void const *cos_ptr,
                             void const *sin_ptr,
                             float kv_eps) {
  using bf16 = __hip_bfloat16;

  constexpr int QK_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM;
  constexpr int ROPE_HALF = QK_ROPE_HEAD_DIM / 2;
  constexpr int NUM_THREADS = 256;
  constexpr int NUM_WARPS = NUM_THREADS / 64;
  constexpr int MAX_PAGES_PER_REQUEST =
      (MAX_SEQ_LEN + PAGE_SIZE - 1) / PAGE_SIZE;
  // uint4 = 16 bytes = 8 bf16. The absorbed part of Q is a pure bit copy, so
  // it moves at the widest granularity the layout allows.
  constexpr int VEC_SIZE = 8;
  constexpr int LORA_VECS = KV_LORA_RANK / VEC_SIZE;

  static_assert(KV_LORA_RANK % VEC_SIZE == 0,
                "KV_LORA_RANK must be a multiple of the 8-wide copy");
  static_assert(QK_ROPE_HEAD_DIM % 2 == 0, "rope dim must be even");
  static_assert(Q_WORKSPACE_STRIDE >= NUM_QO_HEADS * QK_DIM,
                "q workspace row must hold every absorbed head");
  static_assert(KV_CACHE_STRIDE >= QK_DIM,
                "cache row must hold c_kv and k_rope");
  static_assert(KV_INPUT_OFFSET + QK_DIM <= KV_INPUT_STRIDE,
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

  bf16 const *__restrict__ d_q = reinterpret_cast<bf16 const *>(q_absorbed_ptr) +
                                 (long)first_token_pos * Q_INPUT_STRIDE;
  bf16 const *__restrict__ d_kv = reinterpret_cast<bf16 const *>(kv_latent_ptr) +
                                  (long)first_token_pos * KV_INPUT_STRIDE +
                                  KV_INPUT_OFFSET;
  bf16 *__restrict__ d_cache = reinterpret_cast<bf16 *>(paged_kv_cache_ptr);
  bf16 *__restrict__ d_q_ws = reinterpret_cast<bf16 *>(q_workspace_ptr) +
                              (long)first_token_pos * Q_WORKSPACE_STRIDE;
  bf16 const *__restrict__ kv_weight =
      reinterpret_cast<bf16 const *>(kv_norm_weight_ptr);
  bf16 const *__restrict__ d_cos = reinterpret_cast<bf16 const *>(cos_ptr);
  bf16 const *__restrict__ d_sin = reinterpret_cast<bf16 const *>(sin_ptr);

  __syncthreads();

  // ==========================================================================
  // Step 1: Q -> workspace. Bit copy of the absorbed c-part, partial
  //         interleaved RoPE on the trailing rope slice.
  // ==========================================================================
  for (int token = 0; token < num_tokens; token++) {
    int const pos = global_seq_len - num_tokens + token;
    bf16 const *cos_data = d_cos + (long)pos * QK_ROPE_HEAD_DIM;
    bf16 const *sin_data = d_sin + (long)pos * QK_ROPE_HEAD_DIM;
    bf16 const *q_row = d_q + (long)token * Q_INPUT_STRIDE;
    bf16 *ws_row = d_q_ws + (long)token * Q_WORKSPACE_STRIDE;

    for (int vec_idx = tid; vec_idx < NUM_QO_HEADS * LORA_VECS;
         vec_idx += NUM_THREADS) {
      int const head = vec_idx / LORA_VECS;
      int const col = (vec_idx % LORA_VECS) * VEC_SIZE;
      int const off = head * QK_DIM + col;
      *reinterpret_cast<uint4 *>(&ws_row[off]) =
          *reinterpret_cast<uint4 const *>(&q_row[off]);
    }

    for (int idx = tid; idx < NUM_QO_HEADS * ROPE_HALF; idx += NUM_THREADS) {
      int const head = idx / ROPE_HALF;
      int const j = idx % ROPE_HALF;
      bf16 const *src = &q_row[head * QK_DIM + KV_LORA_RANK];
      bf16 *dst = &ws_row[head * QK_DIM + KV_LORA_RANK];
      float const x0 = __cvt_bf16_to_f32_mla(src[2 * j]);
      float const x1 = __cvt_bf16_to_f32_mla(src[2 * j + 1]);
      float const c = __cvt_bf16_to_f32_mla(cos_data[j]);
      float const s = __cvt_bf16_to_f32_mla(sin_data[j]);
      dst[j] = static_cast<bf16>(x0 * c - x1 * s);
      dst[j + ROPE_HALF] = static_cast<bf16>(x1 * c + x0 * s);
    }
  }

  // ==========================================================================
  // Step 2: latent -> paged cache. kv_a_layernorm over c_kv, partial
  //         interleaved RoPE over k_rope.
  // ==========================================================================
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

} // namespace kernel
