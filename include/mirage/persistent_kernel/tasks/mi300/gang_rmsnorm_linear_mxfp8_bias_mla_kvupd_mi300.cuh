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

// MXFP8 twin of gang_rmsnorm_linear_bias_mla_kvupd_mi300.cuh: fused
// q_a_layernorm + absorbed q_b_proj + MLA KV cache update, with the q_b weight
// stored as MXFP8 instead of bf16.
//
// q_b is 1024 x 18432 per layer, 1.77 GB/token across GLM's 47 layers -- the
// largest single bf16 GEMM left after the MoE and the LM head went to MXFP8,
// and 19% of the whole weight budget.
//
// Everything around the GEMM is reused verbatim: latent_to_cache and
// rope_tile_inplace come from the bf16 header, and the tile-0-carries-the-
// latent-row dispatch is unchanged. Only two things differ.
//
// The GEMM body is gang_rmsnorm_linear_mxfp8_bias_kernel rather than
// gang_rmsnorm_linear_bias_kernel. That kernel takes the narrowed reduction
// implicitly -- it uses REDUCTION_SIZE as the norm span, the GEMM extent and
// the row stride all at once -- which is correct here for the same reason the
// bf16 path's `reduction_size` parameter is: BATCH_SIZE is 1, so the row
// stride is never applied. GLM's q_a is a 1024-wide prefix of the 2048-wide
// [q_a | latent] row that qkv_a emitted, and the trailing latent columns are
// dropped from the GEMM entirely rather than multiplied against zero weights.
//
// The rope-slice test moves from tile coordinates to workgroup index. The
// MXFP8 weight is packed per workgroup of OUTPUT_PER_WG columns, so a flat
// tile resolves to (tok_idx, wg_idx) instead of (m_tile, n_tile), and wg_idx
// plays exactly the role n_tile did: a head spans QK_DIM columns, its rope
// slice is the last QK_ROPE_HEAD_DIM of them, and with OUTPUT_PER_WG equal to
// the rope width that slice is one workgroup. GLM: 512 + 64 against 64 gives
// 9 workgroups a head, and every ninth owns a rope slice alone.

#pragma once

#include "tasks/mi300/gang_rmsnorm_linear_bias_mla_kvupd_mi300.cuh"
#include "tasks/mi300/gang_rmsnorm_linear_mxfp8_bias_mi300.cuh"

namespace kernel {

// Template parameters up to ACTUAL_HIDDEN_DIM are
// gang_rmsnorm_linear_mxfp8_bias_kernel's; the rest describe the latent row
// and match the bf16 kvupd kernel one for one.
template <int BATCH_SIZE,
          int OUTPUT_PER_WG,
          int REDUCTION_SIZE,
          int ACTUAL_HIDDEN_DIM,
          int KV_LORA_RANK,
          int QK_ROPE_HEAD_DIM,
          int KV_INPUT_STRIDE,
          int KV_CACHE_STRIDE,
          int MAX_SEQ_LEN,
          int PAGE_SIZE,
          int KV_INPUT_OFFSET,
          bool WRITE_THROUGH = false>
__device__ __attribute__((noinline)) void
    gang_rmsnorm_linear_mxfp8_bias_mla_kvupd_kernel(
        void const *norm_input_ptr,  // [batch, KV_INPUT_STRIDE] (q_a prefix)
        void const *norm_weight_ptr, // [REDUCTION_SIZE]
        void *norm_output_ptr,       // [batch, KV_INPUT_STRIDE] scratch
        void const *weight_ptr,      // [n_wgs_per_xcd, wg_bytes] packed MXFP8
        void const *bias_ptr,        // [1, output_size_per_xcd]
        void const *kv_latent_ptr,   // [batch, KV_INPUT_STRIDE] from kv_a
        void const *kv_norm_weight_ptr, // [KV_LORA_RANK]
        void const *cos_ptr,
        void const *sin_ptr,
        void *q_workspace_ptr,    // [batch, o_stride], was q_absorbed
        void *paged_kv_cache_ptr, // [pages, PAGE_SIZE, 1, KV_CACHE_STRIDE]
        int const *qo_indptr,
        int const *kv_indptr,
        int const *kv_indices,
        int const *kv_last_page_len,
        int16_t request_id,
        int num_active_tokens,
        int n_wgs_per_xcd,
        int o_stride,
        int tile_idx,
        float kv_eps) {
  using bf16 = __hip_bfloat16;
  constexpr int QK_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM;

  // A head's rope slice has to be exactly one workgroup, or the in-place
  // rotation would straddle two of them.
  static_assert(OUTPUT_PER_WG == QK_ROPE_HEAD_DIM,
                "OUTPUT_PER_WG must equal the rope width");
  static_assert(QK_DIM % OUTPUT_PER_WG == 0,
                "a head must be a whole number of workgroups");

  // Tile 0 carries the latent row and nothing else; the GEMM's tiles are
  // shifted up by one. See the bf16 header for why this gets a dispatch slot
  // of its own rather than riding on a worker that also has a tile.
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
                                           KV_INPUT_OFFSET,
                                           WRITE_THROUGH>(kv_latent_ptr,
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

  gang_rmsnorm_linear_mxfp8_bias_kernel<BATCH_SIZE,
                                        OUTPUT_PER_WG,
                                        REDUCTION_SIZE,
                                        ACTUAL_HIDDEN_DIM,
                                        WRITE_THROUGH>(norm_input_ptr,
                                                           norm_weight_ptr,
                                                           norm_output_ptr,
                                                           weight_ptr,
                                                           bias_ptr,
                                                           q_workspace_ptr,
                                                           num_active_tokens,
                                                           n_wgs_per_xcd,
                                                           o_stride,
                                                           gemm_tile_idx);

  // Does this worker own a rope slice? Positional, and needs no knowledge of
  // which XCD we are on, because QK_DIM divides the per-XCD chunk.
  int const tok_idx = gemm_tile_idx / n_wgs_per_xcd;
  int const wg_idx = gemm_tile_idx % n_wgs_per_xcd;
  constexpr int WGS_PER_HEAD = QK_DIM / OUTPUT_PER_WG;
  if (wg_idx % WGS_PER_HEAD != WGS_PER_HEAD - 1) {
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

  // The rope tile spans all four waves' output columns, so the barrier is
  // load-bearing; buffer_inv is kept from the bf16 path so a stale L1 line
  // cannot shadow this workgroup's own stores.
  __syncthreads();
  asm volatile("buffer_inv" ::: "memory");

  bf16 const *d_cos = reinterpret_cast<bf16 const *>(cos_ptr);
  bf16 const *d_sin = reinterpret_cast<bf16 const *>(sin_ptr);
  bf16 *tile_base = reinterpret_cast<bf16 *>(q_workspace_ptr) +
                    (long)tok_idx * o_stride +
                    (long)wg_idx * OUTPUT_PER_WG;

  int const row = tok_idx;
  if (row < first_token_pos || row >= first_token_pos + num_tokens ||
      row >= num_active_tokens) {
    return;
  }
  int const pos = global_seq_len - num_tokens + (row - first_token_pos);
  gang_mla_kvupd_detail::rope_tile_inplace<QK_ROPE_HEAD_DIM, WRITE_THROUGH>(
      tile_base,
      d_cos + (long)pos * QK_ROPE_HEAD_DIM,
      d_sin + (long)pos * QK_ROPE_HEAD_DIM);
}

} // namespace kernel
