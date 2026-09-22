/* Paged PREFILL attention (seqlen_q > 1) for gfx1250 (MI450), HEAD_DIM=64, WMMA.
 *
 * The companion to paged_attention_wmma_decode_hd64_mi450.cuh. On mi300 this
 * case goes to CK FMHA, which cannot target gfx1250 at all: CK routes WMMA
 * through __builtin_amdgcn_wmma_*_w32_gfx12, which requires the target feature
 * wmma-128b-insts, and gfx1250 does not have it. So this is written fresh, not
 * ported.
 *
 * ── Relationship to the decode kernel ──
 *
 * Structurally this IS the decode kernel with one axis reinterpreted, and that
 * is deliberate rather than lazy. Decode takes the QK product as A=K, B=Q to
 * get D[kv][q], so lane l owns query column (l%16) and a contiguous run of KV
 * rows -- which keeps each query's softmax reduction inside one lane. There,
 * the 16 columns are the NUM_QO_PER_KV *query heads* of a single token. Here
 * they are 16 *query tokens* of a single head. Every fragment mapping, the
 * shfl_xor(16) redistribution before PV, and the accumulator's column/row
 * disagreement with the softmax stats are all unchanged, because none of them
 * ever cared what the 16 columns meant.
 *
 * Consequences worth stating, since they are the cost of that choice:
 *   - the outer loop is over query heads, one 16-token tile at a time, exactly
 *     as CK's prefill loops `for qo_h in NUM_QO_PER_KV`. No head-batching.
 *   - all 4 waves redundantly recompute QK, each owning 16 of the 64 head dims.
 *     Same as decode. It trades ALU for having no cross-wave communication.
 *
 * ── The part that is genuinely new: masking ──
 *
 * Decode has one query at the very end of the sequence, so every KV token is
 * visible and there is no causal mask at all -- only a length mask. Prefill has
 * seqlen_q query rows that must not see the future, and the convention is
 * BOTTOM-RIGHT causal, not top-left. This is derived from CK's own code rather
 * than from its comment, because getting it backwards is silently wrong on
 * exactly the case that matters (a chunked prefill appending to a non-empty
 * cache):
 *
 *   make_generic_attention_mask_from_lr_window<SimplifiedGenericAttentionMask<true>>(
 *       left_size, 0, seqlen_q, seqlen_k, /*is_top_left=* /false)
 *
 * with left_size = (sliding_window > 0) ? sliding_window - 1 : -1. Feeding that
 * through make_generic_attention_mask_coordinates_from_lr_window:
 *
 *   left_size = -1  ->  seqlen_k - 1              (i.e. unbounded)
 *   x = 1 + right_size + (x_total - y_total) = 1 + (seqlen_k - seqlen_q)
 *   y = 1 + left_size  + (y_total - x_total) = 1 + left_size + seqlen_q - seqlen_k
 *
 * and then IsOutOfBound(i_y, i_x) rejects i_x < x_start || i_x >= x_end, with
 *
 *   x_end   = min(i_y + x, seqlen_k)  ->  i_x <= i_y + (seqlen_k - seqlen_q)
 *   x_start = -y + i_y + 1            ->  i_x >= q_abs - left_size
 *
 * So with delta = seqlen_k - seqlen_q, query row i_y sits at absolute position
 * q_abs = i_y + delta and attends to [q_abs - W + 1, q_abs]. Note the upper
 * bound is q_abs, NOT i_y: when the KV cache already holds `delta` tokens, a
 * top-left reading would let every query see only the first few cached tokens
 * and hide the ones it is actually supposed to attend to. The output stays
 * finite and plausible either way, which is why this is spelled out here.
 *
 * SimplifiedGenericAttentionMask applies x_start unconditionally (there is no
 * IsLocal branch as there is in GenericAttentionMask), so the sliding-window
 * lower bound is always live, not just in a "local" mode.
 *
 * ── What is NOT done here ──
 *
 * V is still loaded as 16 scalar strided loads per fragment rather than staged
 * transposed through LDS, and Q is re-read from global for every KV tile rather
 * than held in LDS across the tile loop. Both are real bandwidth costs and both
 * are left alone on purpose: FFM has no cycle model and AM's numbers would only
 * describe the model, so any tuning here would be guesswork dressed up as
 * optimisation. Correctness first; the shape is deliberately easy to re-tile
 * once silicon can measure it.
 */
#pragma once

#include "mirage/persistent_kernel/arch_traits.cuh"
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

namespace kernel {
namespace mi450 {

#if defined(MIRAGE_ARCH_GFX1250)

// Q tokens per tile. This is the WMMA N dimension of the QK product, so it is
// fixed at 16 by the instruction, not tunable.
static constexpr int ATTN_Q_TILE = 16;

// Prefill attention over a paged KV cache.
//
// Signature and template list match paged_attention_ck_fmha_prefill's, minus
// the ck_tile::index_t spelling, so the dispatch wrapper below is a drop-in for
// paged_attention_ck_fmha_split_kv_impl.
//
// Launch geometry: 128 threads = 4 waves of 32, same as the decode kernel.
template <typename T,
          int NUM_QO_PER_KV,
          int HEAD_DIM,
          int PAGE_SIZE,
          int MAX_SEQ_LEN,
          int NUM_KV_CHUNKS,
          int Q_WORKSPACE_STRIDE,
          int KV_CACHE_STRIDE,
          int NUM_KV_HEADS>
__device__ __noinline__ void
    paged_attention_wmma_prefill_hd64(void const *q_workspace_ptr,
                                      void *paged_k_cache_ptr,
                                      void *paged_v_cache_ptr,
                                      void *output_ptr,
                                      void *lse_ptr,
                                      int const *qo_indptr,
                                      int const *kv_indptr,
                                      int const *kv_indices,
                                      int const *kv_last_page_len,
                                      int16_t request_id,
                                      int kv_head_idx,
                                      int kv_chunk_idx,
                                      float scale_s,
                                      int seqlen_q,
                                      int sliding_window = 0,
                                      void const *sinks_ptr = nullptr) {
  using bf16 = __hip_bfloat16;
  static_assert(HEAD_DIM == 64, "This kernel is HD=64 only");

  int const req = request_id;
  int const query_start = qo_indptr[req];
  if (seqlen_q <= 0) {
    return;
  }

  int const first_page = kv_indptr[req];
  int const num_pages = kv_indptr[req + 1] - first_page;
  int const seqlen_k = (num_pages - 1) * PAGE_SIZE + kv_last_page_len[req];

  // Bottom-right alignment: query row i maps to absolute KV position i + delta.
  int const delta = seqlen_k - seqlen_q;

  int const tid = threadIdx.x;
  int const wave_id = tid / mirage::arch::WAVE_SIZE; // 0..3, owns 16 head dims
  int const lane = tid % mirage::arch::WAVE_SIZE;    // 0..31
  int const frag_row = lane % 16;
  int const frag_half = lane / 16;

  int const my_q = frag_row;                  // query token column of this lane
  int const kv_lo = frag_half * 8;            // first KV of this lane's group
  int const my_dim = wave_id * 16 + frag_row; // head dim this lane outputs

  char const *k_base = reinterpret_cast<char const *>(paged_k_cache_ptr);
  char const *v_base = reinterpret_cast<char const *>(paged_v_cache_ptr);
  bf16 const *d_sinks = reinterpret_cast<bf16 const *>(sinks_ptr);

  constexpr int LSE_STRIDE = NUM_KV_HEADS * NUM_KV_CHUNKS * NUM_QO_PER_KV;
  constexpr int O_ACC_STRIDE = LSE_STRIDE * HEAD_DIM;

  // Page-indirected byte offset of a KV token's row. Identical to decode's.
  auto kv_row_off = [&](int global_tok) -> long {
    int const pid = kv_indices[first_page + global_tok / PAGE_SIZE];
    return (static_cast<long>(pid) * PAGE_SIZE * KV_CACHE_STRIDE +
            static_cast<long>(global_tok % PAGE_SIZE) * KV_CACHE_STRIDE) *
           2;
  };

  int const num_q_tiles = (seqlen_q + ATTN_Q_TILE - 1) / ATTN_Q_TILE;

  for (int qo_h = 0; qo_h < NUM_QO_PER_KV; ++qo_h) {
    int const head = kv_head_idx * NUM_QO_PER_KV + qo_h;

    for (int m0 = 0; m0 < num_q_tiles; ++m0) {
      int const q_local = m0 * ATTN_Q_TILE + my_q; // this lane's query token
      bool const q_valid = q_local < seqlen_q;
      int const q_abs = q_local + delta;

      // ── Q fragment ──
      // Lane l needs Q[token = m0*16 + l%16][d = ks*32 + (l/16)*16 .. +15]:
      // 16 contiguous bf16, one 32-byte load per k-step. Rows past the end of
      // the sequence read row 0 and are discarded at the store.
      attn_ab_t q_frag[HEAD_DIM / ATTN_WMMA_K];
      {
        int const q_row_idx = q_valid ? q_local : 0;
        bf16 const *q_row =
            reinterpret_cast<bf16 const *>(q_workspace_ptr) +
            static_cast<long>(query_start + q_row_idx) * Q_WORKSPACE_STRIDE +
            static_cast<long>(head) * HEAD_DIM;
#pragma unroll
        for (int ks = 0; ks < HEAD_DIM / ATTN_WMMA_K; ++ks) {
          bf16 const *src = q_row + ks * ATTN_WMMA_K + frag_half * 16;
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            q_frag[ks][i] = static_cast<__bf16>(src[i]);
          }
        }
      }

      // ── KV range for this query tile ──
      // Upper bound comes from the LAST query row in the tile, lower bound from
      // the FIRST: a tile is a band, not a single diagonal. Per-element masking
      // below still enforces the exact per-row bounds; this only avoids
      // iterating over tiles that are entirely masked.
      int const q_hi_abs =
          min(m0 * ATTN_Q_TILE + ATTN_Q_TILE - 1, seqlen_q - 1) + delta;
      int const q_lo_abs = m0 * ATTN_Q_TILE + delta;
      int kv_end = q_hi_abs + 1; // exclusive
      if (kv_end > seqlen_k) {
        kv_end = seqlen_k;
      }
      int kv_begin = 0;
      if (sliding_window > 0) {
        kv_begin = q_lo_abs - sliding_window + 1;
        if (kv_begin < 0) {
          kv_begin = 0;
        }
        // Align down so tiles start on a KV_TILE boundary, matching decode's
        // policy. Masking is per element, so this only affects how much work is
        // skipped, never which tokens are visible.
        kv_begin = (kv_begin / ATTN_KV_TILE) * ATTN_KV_TILE;
      }

      int ntiles = (kv_end > kv_begin)
                       ? (kv_end - kv_begin + ATTN_KV_TILE - 1) / ATTN_KV_TILE
                       : 0;
      int tile_first = 0;
      int tile_last = ntiles;

      if constexpr (NUM_KV_CHUNKS > 1) {
        int const per_chunk = (ntiles + NUM_KV_CHUNKS - 1) / NUM_KV_CHUNKS;
        tile_first = kv_chunk_idx * per_chunk;
        tile_last = tile_first + per_chunk;
        if (tile_last > ntiles) {
          tile_last = ntiles;
        }
      }

      float *lse_slot = reinterpret_cast<float *>(lse_ptr) +
                        static_cast<long>(query_start + q_local) * LSE_STRIDE +
                        kv_head_idx * NUM_KV_CHUNKS * NUM_QO_PER_KV +
                        kv_chunk_idx * NUM_QO_PER_KV + qo_h;

      if (tile_first >= tile_last) {
        // Nothing for this chunk to do. LSE must still be stamped -inf or the
        // merge reads whatever was in the slot; the output accumulator is
        // multiplied by exp(LSE - m_global) == 0, but a stale NaN would survive
        // that multiply, so zero it too rather than rely on the weight.
        if (q_valid && wave_id == 0 && frag_half == 0) {
          *lse_slot = -1e30f;
        }
        if constexpr (NUM_KV_CHUNKS > 1) {
          // Clear in the ROW distribution -- the same one the real store below
          // uses. Guarding this on q_valid (a *column* predicate, about my_q)
          // would be a category error: it would leave rows uncleared whenever
          // the lane's column happened to be past the end.
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            int const q_row = m0 * ATTN_Q_TILE + frag_half * 8 + i;
            if (q_row >= seqlen_q) {
              continue;
            }
            float *o = reinterpret_cast<float *>(output_ptr) +
                       static_cast<long>(query_start + q_row) * O_ACC_STRIDE +
                       static_cast<long>(kv_head_idx) * NUM_KV_CHUNKS *
                           NUM_QO_PER_KV * HEAD_DIM +
                       static_cast<long>(kv_chunk_idx) * NUM_QO_PER_KV *
                           HEAD_DIM +
                       static_cast<long>(qo_h) * HEAD_DIM;
            o[my_dim] = 0.f;
          }
        }
        continue;
      }

      // ── Main loop ──
      attn_acc_t o_acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      float m_running = -INFINITY;
      float l_running = 0.f;

      for (int t = tile_first; t < tile_last; ++t) {
        int const tile_start = kv_begin + t * ATTN_KV_TILE;

        // QK over the two 16-token halves. A = K, exactly as decode.
        float scores[2][8];
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          attn_acc_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
          int const kv_abs_row = tile_start + half * 16 + frag_row;
          bool const kv_valid = kv_abs_row < seqlen_k;
          long const row_off = kv_valid ? kv_row_off(kv_abs_row) : 0;

#pragma unroll
          for (int ks = 0; ks < HEAD_DIM / ATTN_WMMA_K; ++ks) {
            attn_ab_t k_frag;
            if (kv_valid) {
              bf16 const *src =
                  reinterpret_cast<bf16 const *>(k_base + row_off) +
                  ks * ATTN_WMMA_K + frag_half * 16;
#pragma unroll
              for (int i = 0; i < 16; ++i) {
                k_frag[i] = static_cast<__bf16>(src[i]);
              }
            } else {
#pragma unroll
              for (int i = 0; i < 16; ++i) {
                k_frag[i] = static_cast<__bf16>(0.f);
              }
            }
            acc = __builtin_amdgcn_wmma_f32_16x16x32_bf16(
                false, k_frag, false, q_frag[ks], 0, acc, false, false);
          }

          // acc[i] is the score of query my_q against KV
          // tile_start + half*16 + kv_lo + i. Apply the causal / window /
          // length mask here, in absolute KV coordinates.
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            int const kv_abs = tile_start + half * 16 + kv_lo + i;
            bool ok = (kv_abs < seqlen_k) && (kv_abs <= q_abs);
            if (sliding_window > 0 && kv_abs < q_abs - sliding_window + 1) {
              ok = false;
            }
            scores[half][i] = ok ? acc[i] * scale_s : -INFINITY;
          }
        }

        // Online softmax, identical to decode: 16 values in-lane plus one
        // exchange with the lane holding the complementary KV group.
        float tile_max = -INFINITY;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            tile_max = fmaxf(tile_max, scores[half][i]);
          }
        }
        tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 16));

        float const new_max = fmaxf(m_running, tile_max);

        // A fully masked tile leaves new_max at -inf, and exp2(-inf - -inf) is
        // NaN. Decode handles this as a first-tile-only edge case; here it is
        // routine, because with a sliding window an interior tile can be fully
        // masked for some query rows and not others.
        //
        // That "some rows and not others" is the trap. `if (new_max ==
        // -INFINITY) continue;` looks like the obvious guard and is WRONG:
        // new_max is per-lane (each lane owns query column frag_row, hence a
        // different q_abs and a different window), so the branch is not
        // wave-uniform. Lanes that took it sit out the rest of the iteration --
        // including attn_swap_half's __shfl_xor and the PV WMMA, which are
        // collective. The surviving lanes then trade with inactive partners and
        // read garbage. This was not hypothetical: it failed
        // "window=48, q=24 k=96" at rel 7e-2 and "sinks+window" at 3e16.
        //
        // So: no branch. Substitute a finite max and let the arithmetic fall
        // out. Every score is -inf, so every prob is 0 and tile_sum is 0;
        // m_running stays -inf (correctly recording "nothing seen yet") because
        // new_max is what gets stored, not safe_max. And new_max == -inf
        // implies m_running == -inf (new_max is a max over it), so rescale is
        // 0 and the accumulator, itself still 0, is unaffected.
        float const safe_max = (new_max == -INFINITY) ? 0.f : new_max;
        float const rescale =
            (m_running == -INFINITY) ? 0.f : attn_exp2(m_running - safe_max);

        float probs[2][8];
        float tile_sum = 0.f;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            float const p = (scores[half][i] == -INFINITY)
                                ? 0.f
                                : attn_exp2(scores[half][i] - safe_max);
            probs[half][i] = p;
            tile_sum += p;
          }
        }
        tile_sum += __shfl_xor(tile_sum, 16);

        l_running = l_running * rescale + tile_sum;
        m_running = new_max;

        // PV A-fragment assembly: keep one 8-group, trade the other with lane
        // l^16. See the decode kernel's header for the derivation.
        float traded0[8], traded1[8];
        attn_swap_half(probs[0], traded0);
        attn_swap_half(probs[1], traded1);

        attn_ab_t p_frag;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          float const lo = (frag_half == 0) ? probs[0][i] : traded1[i];
          float const hi = (frag_half == 0) ? traded0[i] : probs[1][i];
          p_frag[i] = static_cast<__bf16>(lo);
          p_frag[8 + i] = static_cast<__bf16>(hi);
        }

        // V fragment: one head dim across 16 KV tokens.
        attn_ab_t v_frag;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          int const kv_abs = tile_start + frag_half * 16 + i;
          if (kv_abs < seqlen_k) {
            long const off = kv_row_off(kv_abs);
            v_frag[i] = static_cast<__bf16>(
                reinterpret_cast<bf16 const *>(v_base + off)[my_dim]);
          } else {
            // Masked KV: probability is already 0, but the value must still be
            // finite -- a NaN here would survive the multiply by zero.
            v_frag[i] = static_cast<__bf16>(0.f);
          }
        }

        // The accumulator is distributed by query ROW, the softmax stats by
        // query COLUMN, so the rescale for slot i must come from the lane
        // owning query (l/16)*8 + i. Same broadcast decode needs.
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          o_acc[i] *= __shfl(rescale, frag_half * 8 + i);
        }

        o_acc = __builtin_amdgcn_wmma_f32_16x16x32_bf16(
            false, p_frag, false, v_frag, 0, o_acc, false, false);
      }

      // ── Epilogue ──
      // o_acc[i] is the unnormalised output for query token (frag_half*8 + i)
      // of this tile, at head dim my_dim.
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        int const q_in_tile = frag_half * 8 + i;
        int const q_row = m0 * ATTN_Q_TILE + q_in_tile;
        if (q_row >= seqlen_q) {
          continue;
        }
        float const l_q = __shfl(l_running, q_in_tile);
        float const m_q = __shfl(m_running, q_in_tile);
        float inv_l = (l_q > 0.f) ? (1.f / l_q) : 0.f;

        if constexpr (NUM_KV_CHUNKS == 1) {
          if (d_sinks != nullptr) {
            float const lse = attn_lse_natural(m_q, l_q);
            float const sink = static_cast<float>(d_sinks[head]);
            inv_l *= 1.f / (1.f + expf(sink - lse));
          }
          bf16 *o = reinterpret_cast<bf16 *>(output_ptr) +
                    static_cast<long>(query_start + q_row) *
                        Q_WORKSPACE_STRIDE +
                    static_cast<long>(head) * HEAD_DIM;
          o[my_dim] = static_cast<bf16>(o_acc[i] * inv_l);
        } else {
          float *o = reinterpret_cast<float *>(output_ptr) +
                     static_cast<long>(query_start + q_row) * O_ACC_STRIDE +
                     static_cast<long>(kv_head_idx) * NUM_KV_CHUNKS *
                         NUM_QO_PER_KV * HEAD_DIM +
                     static_cast<long>(kv_chunk_idx) * NUM_QO_PER_KV *
                         HEAD_DIM +
                     static_cast<long>(qo_h) * HEAD_DIM;
          o[my_dim] = o_acc[i] * inv_l;
        }
      }

      // LSE, one writer per query row: wave 0's lanes 0..15 hold the column
      // distribution, so they write without a broadcast.
      if (wave_id == 0 && frag_half == 0 && q_valid) {
        *lse_slot = attn_lse_natural(m_running, l_running);
      }
    }
  }
}

// Dispatch on seqlen_q, mirroring paged_attention_ck_fmha_split_kv_impl so that
// tasks/mi450/task_header.cuh can alias this over the mi300 entry point without
// touching any call site. DECODE_ONLY exists for the same reason it does there:
// callers that statically know they never prefill can drop this kernel's code
// entirely.
template <typename T,
          int NUM_QO_PER_KV,
          int HEAD_DIM,
          int PAGE_SIZE,
          int MAX_SEQ_LEN,
          int NUM_KV_CHUNKS,
          int Q_WORKSPACE_STRIDE,
          int KV_CACHE_STRIDE,
          int NUM_KV_HEADS,
          bool DECODE_ONLY = false>
__device__ __forceinline__ void paged_attention_wmma_split_kv_impl(
    void const *q_workspace_ptr,
    void *paged_k_cache_ptr,
    void *paged_v_cache_ptr,
    void *o_acc_ptr,
    void *lse_acc_ptr,
    int const *qo_indptr,
    int const *kv_indptr,
    int const *kv_indices,
    int const *kv_last_page_len,
    int16_t request_id,
    int kv_head_idx,
    int kv_chunk_idx,
    float scale_s,
    int sliding_window = 0,
    void const *sinks_ptr = nullptr) {
  int const req = request_id;
  int const query_start = qo_indptr[req];
  int const query_end = qo_indptr[req + 1];
  if (query_start == query_end) {
    return;
  }
  int const seqlen_q = query_end - query_start;

  if (seqlen_q == 1) {
    paged_attention_wmma_decode_hd64<T,
                                     NUM_QO_PER_KV,
                                     HEAD_DIM,
                                     PAGE_SIZE,
                                     MAX_SEQ_LEN,
                                     NUM_KV_CHUNKS,
                                     Q_WORKSPACE_STRIDE,
                                     KV_CACHE_STRIDE,
                                     NUM_KV_HEADS>(q_workspace_ptr,
                                                   paged_k_cache_ptr,
                                                   paged_v_cache_ptr,
                                                   o_acc_ptr,
                                                   lse_acc_ptr,
                                                   qo_indptr,
                                                   kv_indptr,
                                                   kv_indices,
                                                   kv_last_page_len,
                                                   request_id,
                                                   kv_head_idx,
                                                   kv_chunk_idx,
                                                   scale_s,
                                                   sliding_window,
                                                   sinks_ptr);
  } else {
    if constexpr (!DECODE_ONLY) {
      paged_attention_wmma_prefill_hd64<T,
                                        NUM_QO_PER_KV,
                                        HEAD_DIM,
                                        PAGE_SIZE,
                                        MAX_SEQ_LEN,
                                        NUM_KV_CHUNKS,
                                        Q_WORKSPACE_STRIDE,
                                        KV_CACHE_STRIDE,
                                        NUM_KV_HEADS>(q_workspace_ptr,
                                                      paged_k_cache_ptr,
                                                      paged_v_cache_ptr,
                                                      o_acc_ptr,
                                                      lse_acc_ptr,
                                                      qo_indptr,
                                                      kv_indptr,
                                                      kv_indices,
                                                      kv_last_page_len,
                                                      request_id,
                                                      kv_head_idx,
                                                      kv_chunk_idx,
                                                      scale_s,
                                                      seqlen_q,
                                                      sliding_window,
                                                      sinks_ptr);
    }
  }
}

#else // !MIRAGE_ARCH_GFX1250

// Host-pass declarations. Same rationale as linear_wmma_mi450.cuh's: the arch
// macro comes from __gfx1250__, which exists only in the device pass, but
// gang_attention_mi300.cuh names the dispatch symbol (through the
// paged_attention_ck_fmha_split_kv_impl alias) in a header that the host pass
// also parses. Without these the host pass fails with "no member named
// paged_attention_wmma_split_kv_impl". Empty bodies are safe -- HIP does not
// codegen __device__ bodies in the host pass, so they never execute.
//
// Both the decode and prefill kernels need one, because the dispatch wrapper
// calls them; declaring only the wrapper would just move the error.

template <typename T,
          int NUM_QO_PER_KV,
          int HEAD_DIM,
          int PAGE_SIZE,
          int MAX_SEQ_LEN,
          int NUM_KV_CHUNKS,
          int Q_WORKSPACE_STRIDE,
          int KV_CACHE_STRIDE,
          int NUM_KV_HEADS>
__device__ __forceinline__ void
    paged_attention_wmma_prefill_hd64(void const *,
                                      void *,
                                      void *,
                                      void *,
                                      void *,
                                      int const *,
                                      int const *,
                                      int const *,
                                      int const *,
                                      int16_t,
                                      int,
                                      int,
                                      float,
                                      int,
                                      int = 0,
                                      void const * = nullptr) {
}

template <typename T,
          int NUM_QO_PER_KV,
          int HEAD_DIM,
          int PAGE_SIZE,
          int MAX_SEQ_LEN,
          int NUM_KV_CHUNKS,
          int Q_WORKSPACE_STRIDE,
          int KV_CACHE_STRIDE,
          int NUM_KV_HEADS,
          bool DECODE_ONLY = false>
__device__ __forceinline__ void
    paged_attention_wmma_split_kv_impl(void const *,
                                       void *,
                                       void *,
                                       void *,
                                       void *,
                                       int const *,
                                       int const *,
                                       int const *,
                                       int const *,
                                       int16_t,
                                       int,
                                       int,
                                       float,
                                       int = 0,
                                       void const * = nullptr) {
}

#endif // MIRAGE_ARCH_GFX1250

} // namespace mi450
} // namespace kernel
