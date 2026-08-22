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
// slice is the last QK_ROPE_HEAD_DIM of them, so it lands in the head's last
// workgroup whenever OUTPUT_PER_WG is at least the rope width. GLM absorbed:
// 512 + 64 against 64 gives 9 workgroups a head, and every ninth owns a rope
// slice that fills it exactly. GLM un-absorbed at OUTPUT_PER_WG 128: 192 + 64
// against 128 gives 2 workgroups a head, and the second owns the rope slice as
// its trailing half.
//
// ── QK_NOPE_HEAD_DIM > 0: q_b with W_UK left out of it ──────────────────
//
// Absorbing W_UK widens q_b's output from qk_nope + qk_rope to kv_lora +
// qk_rope per head -- 256 to 576 on GLM-5 -- and the weight is [rows, q_lora],
// so the absorbed form costs 77.9 MB a layer against 34.6 MB plus 6.5 MB for
// the W_UK stack. Same bytes-for-ops trade as the o_proj/W_UV side, and the
// same exactness argument: MLA's QK product is linear in the cached latent.
//
// The only thing that changes in here is where the output goes. A head is
// four workgroups instead of nine, the first three carry nope rows and land
// contiguously in a [num_heads, qk_nope + qk_rope] scratch that the W_UK GEMV
// reduces over, and the fourth is still the rope slice alone -- rotated out of
// that scratch and into the 576-wide query row at head * QK_DIM +
// KV_LORA_RANK, where MLA decode expects it. The scratch's own rope columns
// keep the un-rotated values and are never read.

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
          bool WRITE_THROUGH = false,
          // > 0 un-absorbs W_UK: q_b's output row is per head
          // QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM wide instead of QK_DIM, and
          // q_workspace_ptr addresses that scratch rather than the query row.
          int QK_NOPE_HEAD_DIM = 0,
          // Hand the rotation to the caller instead of doing it here. The
          // only reason the rope has a say in OUTPUT_PER_WG is that the slice
          // must not be split across workgroups with no barrier between them;
          // a caller that already runs a barrier after this GEMM -- the
          // un-absorbed path's XCD-local W_UK release -- can rotate on the far
          // side of it and free the tile width entirely. See the note on the
          // static_assert below.
          bool DEFER_ROPE = false>
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
        float kv_eps,
        // Un-absorbed only: the 576-wide query row MLA decode reads, whole
        // rather than this XCD's slice, since the head index below is global.
        void *q_rope_out_ptr = nullptr) {
  using bf16 = __hip_bfloat16;
  constexpr int QK_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM;
  constexpr bool UNABSORB_K = QK_NOPE_HEAD_DIM > 0;
  constexpr int HEAD_SPAN =
      UNABSORB_K ? (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM) : QK_DIM;

  // A head's rope slice has to sit inside ONE workgroup: the rotation reads
  // (2j, 2j+1) and writes (j, j + ROPE_HALF) across the whole slice, so a
  // slice split over two workers -- which have no barrier between them --
  // would read half of it before the other half was written.
  //
  // It does not have to *be* the workgroup, which is what this used to
  // require. The slice is the tail of a head's span, so whenever the span
  // divides into whole workgroups and a workgroup is at least as wide as the
  // slice, the head's last workgroup contains it outright. Decoupling the two
  // is what lets q_b pick its tile width on scheduling grounds instead of
  // inheriting the rope's: at GLM-5's 8 heads per XCD and a 256-wide head
  // span, OUTPUT_PER_WG 64 gives 32 tiles against 29 workers -- two
  // grid-stride rounds for four of them and one for the rest -- while 128
  // gives 16 and clears in a single round.
  //
  // DEFER_ROPE drops even that. Under the EP head shard q_b covers one head
  // per XCD, so at OUTPUT_PER_WG 64 the whole phase is four tiles against 29
  // workers and its makespan is one 135 KB tile -- 25 workers idle through
  // 11 us. The only thing standing between that and a 16-wide tile is this
  // assert, and the caller that wants it already has the barrier the rotation
  // needs. So it rotates there, and this kernel just leaves the un-roped
  // columns in the scratch.
  static_assert(DEFER_ROPE || OUTPUT_PER_WG >= QK_ROPE_HEAD_DIM,
                "a head's rope slice must fit inside one workgroup");
  static_assert(!DEFER_ROPE || UNABSORB_K,
                "only the un-absorbed path has a barrier after this GEMM and "
                "a separate scratch to leave the un-roped columns in");
  static_assert(HEAD_SPAN % OUTPUT_PER_WG == 0,
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

  // norm_input_ptr is the [q_a | latent] row, KV_INPUT_STRIDE wide, and this
  // GEMM reduces over its REDUCTION_SIZE-wide q_a prefix. The two differ, so
  // the row stride has to be handed over explicitly; without it the second
  // token's row lands REDUCTION_SIZE in rather than KV_INPUT_STRIDE.
  gang_rmsnorm_linear_mxfp8_bias_kernel<BATCH_SIZE,
                                        OUTPUT_PER_WG,
                                        REDUCTION_SIZE,
                                        ACTUAL_HIDDEN_DIM,
                                        WRITE_THROUGH,
                                        /*FUSE_RESADD=*/false,
                                        /*EP_PEER_SLOTS=*/0,
                                        /*EP_PRE_FOLDED=*/false,
                                        /*SP_QKV=*/false,
                                        /*PRO_PUB=*/false,
                                        // q_b keeps the row on the tile index:
                                        // 4 tiles per XCD against 29 workers
                                        // is one round at bs=2 either way.
                                        /*FOLD_ROWS=*/false,
                                        KV_INPUT_STRIDE>(norm_input_ptr,
                                                           norm_weight_ptr,
                                                           norm_output_ptr,
                                                           weight_ptr,
                                                           bias_ptr,
                                                           q_workspace_ptr,
                                                           num_active_tokens,
                                                           n_wgs_per_xcd,
                                                           o_stride,
                                                           gemm_tile_idx);

  if constexpr (DEFER_ROPE) {
    // The caller rotates, after its own barrier. Nothing below this point --
    // the rope-slice test, the position arithmetic, the cos/sin loads -- has
    // any other purpose, so the whole tail goes away and the tile is a plain
    // GEMM again.
    return;
  }

  // Does this worker own a rope slice? Positional, and needs no knowledge of
  // which XCD we are on, because QK_DIM divides the per-XCD chunk.
  int const tok_idx = gemm_tile_idx / n_wgs_per_xcd;
  int const wg_idx = gemm_tile_idx % n_wgs_per_xcd;
  constexpr int WGS_PER_HEAD = HEAD_SPAN / OUTPUT_PER_WG;
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
  // The rope slice is the *tail* of this workgroup's columns, not all of them,
  // whenever OUTPUT_PER_WG is wider than the rope.
  bf16 *tile_base = reinterpret_cast<bf16 *>(q_workspace_ptr) +
                    (long)tok_idx * o_stride +
                    (long)wg_idx * OUTPUT_PER_WG +
                    (OUTPUT_PER_WG - QK_ROPE_HEAD_DIM);

  int const row = tok_idx;
  if (row < first_token_pos || row >= first_token_pos + num_tokens ||
      row >= num_active_tokens) {
    return;
  }
  int const pos = global_seq_len - num_tokens + (row - first_token_pos);

  // Absorbed: the rope slice is already sitting in the query row, so it is
  // rotated where it is. Un-absorbed: it is sitting in the [nope | rope]
  // scratch instead, and the query row is 576 wide with the roped 64 last, so
  // the rotation writes across. The head index has to be global for that --
  // q_workspace_ptr is this XCD's slice of the scratch, but q_rope_out_ptr is
  // the whole query row.
  bf16 *rope_out = nullptr;
  if constexpr (UNABSORB_K) {
    // `if constexpr (DEFER_ROPE) return;` above discards the branch, not the
    // rest of the function, so this tail is still instantiated when the caller
    // does the rotation itself -- hence the DEFER_ROPE term. Without it a
    // BATCH_SIZE > 1 build fails here on code that never runs.
    static_assert(DEFER_ROPE || BATCH_SIZE == 1,
                  "the query row's stride is not plumbed through here, so the "
                  "cross-write is only addressable at one token");
    // The nope rows do NOT have to fill whole workgroups. At OUTPUT_PER_WG 128
    // the head's second workgroup straddles nope[128:192] and the rope slice,
    // and that is harmless in exactly this branch: `out` is non-null, so the
    // rotation writes to the query row and leaves the scratch's rope columns
    // holding their un-roped values -- which W_UK never reads, since it
    // reduces over head_in[0:QK_NOPE_HEAD_DIM] only.
    int const xcd_id = gang_rmsnorm_topk_detail::get_xcd_id();
    int const head = xcd_id * (n_wgs_per_xcd / WGS_PER_HEAD) +
                     wg_idx / WGS_PER_HEAD;
    rope_out = reinterpret_cast<bf16 *>(q_rope_out_ptr) +
               (long)head * QK_DIM + KV_LORA_RANK;
  }
  gang_mla_kvupd_detail::rope_tile_inplace<QK_ROPE_HEAD_DIM, WRITE_THROUGH>(
      tile_base,
      d_cos + (long)pos * QK_ROPE_HEAD_DIM,
      d_sin + (long)pos * QK_ROPE_HEAD_DIM,
      rope_out);
}

} // namespace kernel
