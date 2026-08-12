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
#include "tasks/common/common_header.cuh"
#include <hip/hip_cooperative_groups.h>

namespace kernel {

template <typename T,
          typename InputSmem,
          int NUM_HEAD,
          int WINDOW_SIZE,
          int HEAD_DIM = 128>
__device__ __forceinline__ void rotary_embedding(InputSmem smem_input,
                                                 T const *cos_ptr,
                                                 T const *sin_ptr,
                                                 int token_offset = 0) {
  // Avoid sync divergence dead lock.
  static_assert(HEAD_DIM < NUM_THREADS || HEAD_DIM % NUM_THREADS == 0);
  constexpr int ROTARY_PARTICIPATING_THREADS =
      (NUM_THREADS < HEAD_DIM ? NUM_THREADS : HEAD_DIM);
  // HIP: wavefront size is 64, tile size must be <= 64 and power of 2
  constexpr int HIP_TILE_SIZE =
      (ROTARY_PARTICIPATING_THREADS > 64) ? 64 : ROTARY_PARTICIPATING_THREADS;
  static_assert(HIP_TILE_SIZE <= 64 &&
                    (HIP_TILE_SIZE & (HIP_TILE_SIZE - 1)) == 0,
                "HIP tile size must be <= 64 and power of 2");
  auto block_group = cooperative_groups::this_thread_block();
  auto participating_group =
      cooperative_groups::tiled_partition<HIP_TILE_SIZE>(block_group);
#pragma unroll
  for (int win_idx = 0; win_idx < WINDOW_SIZE; ++win_idx) {

    int smem_seq_idx = token_offset + win_idx;

#pragma unroll
    for (int head_idx = 0; head_idx < NUM_HEAD; ++head_idx) {

      T const *cur_cos_ptr = cos_ptr + win_idx * HEAD_DIM;
      T const *cur_sin_ptr = sin_ptr + win_idx * HEAD_DIM;

#pragma unroll
      for (uint32_t i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
        int offset = (i / HEAD_DIM) * HEAD_DIM + i;

        int row = smem_seq_idx * NUM_HEAD + head_idx;
        int col = i;

        float cos = static_cast<float>(cur_cos_ptr[offset]);
        float sin = static_cast<float>(cur_sin_ptr[offset]);

        float v_rot;

        participating_group.sync();

        if (i < HEAD_DIM / 2) {
          float v1 = static_cast<float>(smem_input.at(row, col));
          float v2 = static_cast<float>(smem_input.at(row, col + HEAD_DIM / 2));
          v_rot = v1 * cos - v2 * sin;
        } else {
          float v1 = static_cast<float>(smem_input.at(row, col));
          float v2 = static_cast<float>(smem_input.at(row, col - HEAD_DIM / 2));
          v_rot = v1 * cos + v2 * sin;
        }

        participating_group.sync();
        smem_input.at(row, col) = static_cast<T>(v_rot);
      }
    }
  }
}

// Partial + interleaved RoPE, as used by GLM-5 (`rope_interleave: true`,
// qk_rope_head_dim 64 of qk_head_dim 256) and DeepSeek-V3.2.
//
// Two differences from the half-split RoPE the GQA path uses in
// kv_cache_update_mi300.cuh:
//
//   1. Partial: only the trailing ROPE_DIM dims of each head are rotated; the
//      leading (HEAD_DIM - ROPE_DIM) "nope" dims are copied through untouched.
//   2. Interleaved: the rotary pair for angle j is (2j, 2j+1), not
//      (j, j + ROPE_DIM/2).
//
// The rotated pair is written back to (j, j + ROPE_DIM/2) rather than in
// place. That is not a bug — it is exactly what transformers'
// apply_rotary_pos_emb_interleave() does: it reads the even/odd slices and
// returns `cat([even*cos - odd*sin, odd*cos + even*sin], dim=-1)`. The output
// permutation is the same for Q and K, so the QK dot product is unchanged, and
// matching it keeps us bit-comparable with the HF reference.
//
// `heads` points at NUM_HEADS heads laid out contiguously as
// [NUM_HEADS][HEAD_DIM]. cos_ptr/sin_ptr point at this token's row of a
// [max_seq, ROPE_DIM] table in HF layout (emb = cat(freqs, freqs)); only the
// first ROPE_DIM/2 entries are read, the same convention the existing
// half-split RoPE follows.
//
// Reads (2j, 2j+1) and writes (j, j + ROPE_DIM/2), so the read and write index
// sets overlap across threads: the rotation is staged in registers with a
// barrier in between.
template <typename T,
          int HEAD_DIM,
          int ROPE_DIM,
          int NUM_HEADS,
          int BLOCK_THREADS>
__device__ __forceinline__ void rope_interleave_partial(T *heads,
                                                        T const *cos_ptr,
                                                        T const *sin_ptr) {
  static_assert(ROPE_DIM % 2 == 0, "ROPE_DIM must be even");
  static_assert(ROPE_DIM <= HEAD_DIM, "ROPE_DIM must fit inside HEAD_DIM");
  constexpr int HALF = ROPE_DIM / 2;
  constexpr int NOPE_DIM = HEAD_DIM - ROPE_DIM;
  constexpr int TOTAL = NUM_HEADS * HALF;
  constexpr int PER_THREAD = (TOTAL + BLOCK_THREADS - 1) / BLOCK_THREADS;

  float rot_even[PER_THREAD];
  float rot_odd[PER_THREAD];

#pragma unroll
  for (int it = 0; it < PER_THREAD; ++it) {
    int const idx = threadIdx.x + it * BLOCK_THREADS;
    if (idx < TOTAL) {
      int const head = idx / HALF;
      int const j = idx % HALF;
      T const *rot = heads + head * HEAD_DIM + NOPE_DIM;
      float const x0 = static_cast<float>(rot[2 * j]);
      float const x1 = static_cast<float>(rot[2 * j + 1]);
      float const c = static_cast<float>(cos_ptr[j]);
      float const s = static_cast<float>(sin_ptr[j]);
      rot_even[it] = x0 * c - x1 * s;
      rot_odd[it] = x1 * c + x0 * s;
    }
  }
  __syncthreads();

#pragma unroll
  for (int it = 0; it < PER_THREAD; ++it) {
    int const idx = threadIdx.x + it * BLOCK_THREADS;
    if (idx < TOTAL) {
      int const head = idx / HALF;
      int const j = idx % HALF;
      T *rot = heads + head * HEAD_DIM + NOPE_DIM;
      rot[j] = static_cast<T>(rot_even[it]);
      rot[j + HALF] = static_cast<T>(rot_odd[it]);
    }
  }
  __syncthreads();
}

} // namespace kernel
