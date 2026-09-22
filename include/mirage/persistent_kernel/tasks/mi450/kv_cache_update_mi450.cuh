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

#pragma once

#include "mirage/persistent_kernel/arch_traits.cuh"

namespace kernel {
namespace mi450 {

// 16-byte vector copy helpers.
//
// mi300 takes these from multitoken_paged_attention_ck_mi300.cuh, which the
// mi450 build cannot include at all -- it is a CK FMHA header and CK's WMMA
// path requires target feature wmma-128b-insts, which gfx1250 does not have.
// They are four lines of type punning with no arch content, so they are
// restated here rather than pulled out into a shared header: this is the only
// mi450 consumer, and moving them would edit a file on the (currently green)
// gfx950 build path for no gain.
//
// Declared inside kernel::mi450 so they cannot collide with the identically
// named templates in kernel:: if both headers end up in one TU.
struct kv450_float4_t {
  float x, y, z, w;
};

template <typename T>
__device__ __forceinline__ void vec_load_8(T *dst, T const *src) {
  *reinterpret_cast<kv450_float4_t *>(dst) =
      *reinterpret_cast<kv450_float4_t const *>(src);
}

template <typename T>
__device__ __forceinline__ void vec_store_8(T *dst, T const *src) {
  *reinterpret_cast<kv450_float4_t *>(dst) =
      *reinterpret_cast<kv450_float4_t const *>(src);
}

// KV cache update -- phase A of attention, ported from kv_cache_update_mi300.
//
// WHY THIS IS A PORT AND NOT AN INCLUDE
//
// This file contains no inline asm at all, so at first read it looks like a
// tier-2 candidate (share the mi300 source, change nothing). It is not, for
// one reason: it reduces across the wave, and the mi300 source spells the wave
// width as the literal 64 in six places --
//
//   NUM_THREADS_PER_WARP = 64        the butterfly sweep start
//   NUM_WARPS = NUM_THREADS / 64     the count of cross-wave scratch slots
//   threadIdx.x / 64                 which slot this wave writes
//   threadIdx.x % 64 == 0            which lane does the writing
//
// -- and it calls shfl_xor_sync() from tasks/common/utils.cuh, which on AMD is
// hardcoded to `__shfl_xor(x, lane_mask, 64)`. At wave32 every one of those is
// wrong, and wrong in the quiet way: a butterfly over 64 lanes on a 32-lane
// wave still executes, `threadIdx.x % 64 == 0` still elects a writer (just one
// per two waves, so half the scratch slots are never written and are read back
// as whatever LDS held), and the kernel produces a plausible RMS norm rather
// than a fault. Nothing about the output shape would tell you.
//
// So the reduction is re-expressed through mirage::arch, which is written
// against WAVE_SIZE and is therefore correct at both widths. Everything else --
// the paged-cache addressing, the RoPE rotation, the vec_load_8 tiling -- is
// wave-agnostic and is carried over unchanged, deliberately: those parts have
// been exercised on gfx950 and a gratuitous rewrite would only add risk.
//
// WHAT IS DIFFERENT AT WAVE32 BEYOND THE LITERALS
//
// NUM_WAVES doubles (256 threads = 8 waves, not 4), so s_reduce needs 8 slots
// instead of 4 and the second-stage reduce folds 8 partials instead of 4. That
// stays inside one wave (8 <= 32), which is what makes the two-stage shape
// valid at all; it is asserted below rather than assumed, because a config that
// violated it would again reduce over a subset and return a plausible number.
template <typename T,
          int NUM_QO_HEADS,
          int NUM_KV_HEADS,
          int NUM_QO_GROUPS,
          int KV_CACHE_STRIDE,
          int QKV_STRIDE,
          int HEAD_DIM,
          int MAX_SEQ_LEN,
          int PAGE_SIZE,
          int MAX_TOKENS,
          int Q_WORKSPACE_STRIDE>
__device__ __forceinline__ void
    kv_cache_update_impl(void const *qkv_ptr,
                         void *paged_k_cache_ptr,
                         void *paged_v_cache_ptr,
                         void *q_workspace_ptr,
                         int const *qo_indptr_buffer_ptr,
                         int const *paged_kv_indptr_buffer_ptr,
                         int const *paged_kv_indices_buffer_ptr,
                         int const *paged_kv_last_page_len_buffer_ptr,
                         int16_t request_id,
                         bool qk_norm,
                         bool rope,
                         void const *q_norm_weight_ptr,
                         void const *k_norm_weight_ptr,
                         void const *cos_ptr,
                         void const *sin_ptr,
                         float q_eps,
                         float k_eps) {

  constexpr int NUM_QO_PER_KV = NUM_QO_HEADS / NUM_KV_HEADS;
  constexpr int NUM_THREADS = 256;
  constexpr int WAVE = mirage::arch::WAVE_SIZE;
  constexpr int NUM_WAVES = NUM_THREADS / WAVE;
  static_assert(NUM_WAVES <= WAVE,
                "the second reduction stage folds NUM_WAVES partials inside a "
                "single wave; if NUM_WAVES exceeded WAVE_SIZE it would silently "
                "reduce only the first WAVE_SIZE of them");
  constexpr int MAX_PAGES_PER_REQUEST =
      (MAX_SEQ_LEN + PAGE_SIZE - 1) / PAGE_SIZE;
  constexpr int VEC_SIZE = 8;
  constexpr int HEAD_DIM_VECS = HEAD_DIM / VEC_SIZE;

  using bf16 = ck_tile::bf16_t;

  int const first_token_pos = qo_indptr_buffer_ptr[request_id];
  int const last_token_pos = qo_indptr_buffer_ptr[request_id + 1];
  if (first_token_pos == last_token_pos) {
    return;
  }
  int const num_tokens = last_token_pos - first_token_pos;

  int const first_page_pos = paged_kv_indptr_buffer_ptr[request_id];
  int const last_page_pos = paged_kv_indptr_buffer_ptr[request_id + 1];
  int const num_pages = last_page_pos - first_page_pos;
  int const global_seq_len = (num_pages - 1) * PAGE_SIZE +
                             paged_kv_last_page_len_buffer_ptr[request_id];

  int const wave_idx = mirage::arch::wave_of(threadIdx.x);
  int const lane_idx = mirage::arch::lane_of(threadIdx.x);

  // Load page indices to shared memory
  __shared__ __align__(16) int page_indices[MAX_PAGES_PER_REQUEST];
  for (int i = threadIdx.x; i < num_pages; i += NUM_THREADS) {
    page_indices[i] = paged_kv_indices_buffer_ptr[first_page_pos + i];
  }

  extern __shared__ char smem[];
  // s_q: NUM_QO_PER_KV * num_tokens * HEAD_DIM, s_k: num_tokens * HEAD_DIM,
  // s_reduce: NUM_WAVES floats. Note S_REDUCE_SIZE is twice the mi300 value at
  // wave32 -- it is derived from NUM_WAVES rather than restated.
  constexpr int Q_ROWS = MAX_TOKENS * NUM_QO_PER_KV;
  constexpr size_t S_Q_SIZE = sizeof(bf16) * Q_ROWS * HEAD_DIM;
  constexpr size_t S_K_SIZE = sizeof(bf16) * MAX_TOKENS * HEAD_DIM;
  constexpr size_t S_REDUCE_OFFSET = ((S_Q_SIZE + S_K_SIZE + 15) & ~15);
  constexpr size_t S_REDUCE_SIZE = sizeof(float) * NUM_WAVES;
  (void)S_REDUCE_SIZE;

  bf16 *s_q = reinterpret_cast<bf16 *>(smem);
  bf16 *s_k = reinterpret_cast<bf16 *>(smem + S_Q_SIZE);
  float *s_reduce = reinterpret_cast<float *>(smem + S_REDUCE_OFFSET);

  bf16 const *__restrict__ d_q =
      reinterpret_cast<bf16 const *>(qkv_ptr) + first_token_pos * QKV_STRIDE;
  bf16 const *__restrict__ d_k = d_q + NUM_QO_PER_KV * HEAD_DIM;
  bf16 const *__restrict__ d_v = d_k + HEAD_DIM;
  bf16 *__restrict__ d_paged_k_cache =
      reinterpret_cast<bf16 *>(paged_k_cache_ptr);
  bf16 *__restrict__ d_paged_v_cache =
      reinterpret_cast<bf16 *>(paged_v_cache_ptr);

  __syncthreads();

  // Block-wide sum of a per-thread partial. Two stages: butterfly inside each
  // wave, one slot per wave in LDS, then a second butterfly over those slots
  // inside wave 0. Broadcast through s_reduce[0] because only thread 0 holds
  // the total after the second stage.
  auto block_sum = [&](float v) -> float {
    v = mirage::arch::wave_reduce_sum(v);
    if (lane_idx == 0) {
      s_reduce[wave_idx] = v;
    }
    __syncthreads();
    float t = (threadIdx.x < NUM_WAVES) ? s_reduce[threadIdx.x] : 0.0f;
    t = mirage::arch::wave_reduce_sum(t);
    if (threadIdx.x == 0) {
      s_reduce[0] = t;
    }
    __syncthreads();
    return s_reduce[0];
  };

  // =========================================================================
  // Step 1: Load Q from QKV buffer, apply QK norm + RoPE, write to workspace
  // =========================================================================
  int total_q_vecs = num_tokens * NUM_QO_PER_KV * HEAD_DIM_VECS;
  for (int vec_idx = threadIdx.x; vec_idx < total_q_vecs;
       vec_idx += NUM_THREADS) {
    int row = vec_idx / HEAD_DIM_VECS;
    int vec_col = vec_idx % HEAD_DIM_VECS;
    int token = row / NUM_QO_PER_KV;
    int head = row % NUM_QO_PER_KV;
    vec_load_8(&s_q[row * HEAD_DIM + vec_col * VEC_SIZE],
               &d_q[token * QKV_STRIDE + head * HEAD_DIM + vec_col * VEC_SIZE]);
  }
  __syncthreads();

  if (qk_norm) {
    bf16 const *q_weight = reinterpret_cast<bf16 const *>(q_norm_weight_ptr);
    for (int token = 0; token < num_tokens; token++) {
      int pos = global_seq_len - num_tokens + token;
      bf16 const *cos_data =
          rope ? reinterpret_cast<bf16 const *>(cos_ptr) + pos * HEAD_DIM
               : nullptr;
      bf16 const *sin_data =
          rope ? reinterpret_cast<bf16 const *>(sin_ptr) + pos * HEAD_DIM
               : nullptr;
      for (int head = 0; head < NUM_QO_PER_KV; head++) {
        int row = token * NUM_QO_PER_KV + head;
        bf16 *q_head = s_q + row * HEAD_DIM;

        float sum_sq = 0.0f;
        for (int i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
          float val = ck_tile::type_convert<float>(q_head[i]);
          sum_sq += val * val;
        }
        float const total = block_sum(sum_sq);
        float rms_rcp = rsqrt(total / float(HEAD_DIM) + q_eps);

        for (int i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
          float val = ck_tile::type_convert<float>(q_head[i]) * rms_rcp *
                      ck_tile::type_convert<float>(q_weight[i]);
          q_head[i] = ck_tile::type_convert<bf16>(val);
        }
        __syncthreads();

        if (rope) {
          for (int i = threadIdx.x; i < HEAD_DIM / 2; i += NUM_THREADS) {
            float v0 = ck_tile::type_convert<float>(q_head[i]);
            float v1 = ck_tile::type_convert<float>(q_head[i + HEAD_DIM / 2]);
            float c = ck_tile::type_convert<float>(cos_data[i]);
            float s_val = ck_tile::type_convert<float>(sin_data[i]);
            q_head[i] = ck_tile::type_convert<bf16>(v0 * c - v1 * s_val);
            q_head[i + HEAD_DIM / 2] =
                ck_tile::type_convert<bf16>(v0 * s_val + v1 * c);
          }
          __syncthreads();
        }
      }
    }
  } else if (rope) {
    for (int token = 0; token < num_tokens; token++) {
      int pos = global_seq_len - num_tokens + token;
      bf16 const *cos_data =
          reinterpret_cast<bf16 const *>(cos_ptr) + pos * HEAD_DIM;
      bf16 const *sin_data =
          reinterpret_cast<bf16 const *>(sin_ptr) + pos * HEAD_DIM;
      for (int idx = threadIdx.x; idx < NUM_QO_PER_KV * HEAD_DIM / 2;
           idx += NUM_THREADS) {
        int head = idx / (HEAD_DIM / 2);
        int d = idx % (HEAD_DIM / 2);
        int base = (token * NUM_QO_PER_KV + head) * HEAD_DIM;
        float q0 = ck_tile::type_convert<float>(s_q[base + d]);
        float q1 = ck_tile::type_convert<float>(s_q[base + d + HEAD_DIM / 2]);
        float c = ck_tile::type_convert<float>(cos_data[d]);
        float sv = ck_tile::type_convert<float>(sin_data[d]);
        s_q[base + d] = ck_tile::type_convert<bf16>(q0 * c - q1 * sv);
        s_q[base + d + HEAD_DIM / 2] =
            ck_tile::type_convert<bf16>(q0 * sv + q1 * c);
      }
    }
    __syncthreads();
  }

  // Write processed Q to workspace.
  bf16 *d_q_workspace = reinterpret_cast<bf16 *>(q_workspace_ptr) +
                        first_token_pos * Q_WORKSPACE_STRIDE;
  for (int vec_idx = threadIdx.x; vec_idx < total_q_vecs;
       vec_idx += NUM_THREADS) {
    int row = vec_idx / HEAD_DIM_VECS;
    int vec_col = vec_idx % HEAD_DIM_VECS;
    int token = row / NUM_QO_PER_KV;
    int head = row % NUM_QO_PER_KV;
    vec_store_8(&d_q_workspace[token * Q_WORKSPACE_STRIDE + head * HEAD_DIM +
                               vec_col * VEC_SIZE],
                &s_q[row * HEAD_DIM + vec_col * VEC_SIZE]);
  }

  __syncthreads();

  // =========================================================================
  // Step 2: Load new K tokens, apply QK norm + RoPE, write to KV cache
  // =========================================================================
  int total_k_vecs = num_tokens * HEAD_DIM_VECS;
  for (int vec_idx = threadIdx.x; vec_idx < total_k_vecs;
       vec_idx += NUM_THREADS) {
    int token = vec_idx / HEAD_DIM_VECS;
    int vec_col = vec_idx % HEAD_DIM_VECS;
    vec_load_8(&s_k[token * HEAD_DIM + vec_col * VEC_SIZE],
               &d_k[token * QKV_STRIDE + vec_col * VEC_SIZE]);
  }
  __syncthreads();

  bf16 const *k_weight = reinterpret_cast<bf16 const *>(k_norm_weight_ptr);

  if (qk_norm) {
    for (int k_tok = 0; k_tok < num_tokens; k_tok++) {
      int pos = global_seq_len - num_tokens + k_tok;
      bf16 const *cos_data =
          rope ? reinterpret_cast<bf16 const *>(cos_ptr) + pos * HEAD_DIM
               : nullptr;
      bf16 const *sin_data =
          rope ? reinterpret_cast<bf16 const *>(sin_ptr) + pos * HEAD_DIM
               : nullptr;
      bf16 *k_head = s_k + k_tok * HEAD_DIM;

      float sum_sq = 0.0f;
      for (int i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
        float val = ck_tile::type_convert<float>(k_head[i]);
        sum_sq += val * val;
      }
      float const total = block_sum(sum_sq);
      float rms_rcp = rsqrt(total / float(HEAD_DIM) + k_eps);

      for (int i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
        float val = ck_tile::type_convert<float>(k_head[i]) * rms_rcp *
                    ck_tile::type_convert<float>(k_weight[i]);
        k_head[i] = ck_tile::type_convert<bf16>(val);
      }
      __syncthreads();

      if (rope) {
        for (int i = threadIdx.x; i < HEAD_DIM / 2; i += NUM_THREADS) {
          float v0 = ck_tile::type_convert<float>(k_head[i]);
          float v1 = ck_tile::type_convert<float>(k_head[i + HEAD_DIM / 2]);
          float c = ck_tile::type_convert<float>(cos_data[i]);
          float s_val = ck_tile::type_convert<float>(sin_data[i]);
          k_head[i] = ck_tile::type_convert<bf16>(v0 * c - v1 * s_val);
          k_head[i + HEAD_DIM / 2] =
              ck_tile::type_convert<bf16>(v0 * s_val + v1 * c);
        }
        __syncthreads();
      }
    }
  } else if (rope) {
    for (int k_tok = 0; k_tok < num_tokens; k_tok++) {
      int pos = global_seq_len - num_tokens + k_tok;
      bf16 const *cos_data =
          reinterpret_cast<bf16 const *>(cos_ptr) + pos * HEAD_DIM;
      bf16 const *sin_data =
          reinterpret_cast<bf16 const *>(sin_ptr) + pos * HEAD_DIM;
      for (int d = threadIdx.x; d < HEAD_DIM / 2; d += NUM_THREADS) {
        float k0 = ck_tile::type_convert<float>(s_k[k_tok * HEAD_DIM + d]);
        float k1 = ck_tile::type_convert<float>(
            s_k[k_tok * HEAD_DIM + d + HEAD_DIM / 2]);
        float c = ck_tile::type_convert<float>(cos_data[d]);
        float s = ck_tile::type_convert<float>(sin_data[d]);
        s_k[k_tok * HEAD_DIM + d] =
            ck_tile::type_convert<bf16>(k0 * c - k1 * s);
        s_k[k_tok * HEAD_DIM + d + HEAD_DIM / 2] =
            ck_tile::type_convert<bf16>(k0 * s + k1 * c);
      }
    }
    __syncthreads();
  }

  // Write processed K to the paged cache.
  for (int k_tok = 0; k_tok < num_tokens; k_tok++) {
    int global_pos = global_seq_len - num_tokens + k_tok;
    int page_num = global_pos / PAGE_SIZE;
    int page_offset = global_pos % PAGE_SIZE;
    int page_idx = page_indices[page_num];
    int dst_idx = page_idx * PAGE_SIZE + page_offset;
    for (int vec_col = threadIdx.x; vec_col < HEAD_DIM_VECS;
         vec_col += NUM_THREADS) {
      vec_store_8(
          &d_paged_k_cache[dst_idx * KV_CACHE_STRIDE + vec_col * VEC_SIZE],
          &s_k[k_tok * HEAD_DIM + vec_col * VEC_SIZE]);
    }
  }

  // =========================================================================
  // Step 3: Write new V tokens to the paged cache (V needs no norm/RoPE)
  // =========================================================================
  for (int v_tok = 0; v_tok < num_tokens; v_tok++) {
    int global_pos = global_seq_len - num_tokens + v_tok;
    int page_num = global_pos / PAGE_SIZE;
    int page_offset = global_pos % PAGE_SIZE;
    int page_idx = page_indices[page_num];
    int dst_idx = page_idx * PAGE_SIZE + page_offset;
    for (int vec_col = threadIdx.x; vec_col < HEAD_DIM_VECS;
         vec_col += NUM_THREADS) {
      vec_store_8(
          &d_paged_v_cache[dst_idx * KV_CACHE_STRIDE + vec_col * VEC_SIZE],
          &d_v[v_tok * QKV_STRIDE + vec_col * VEC_SIZE]);
    }
  }
}

} // namespace mi450
} // namespace kernel
