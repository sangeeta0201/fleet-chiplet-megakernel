/* Paged decode attention for gfx1250 (MI450), HEAD_DIM=64, written on WMMA.
 *
 * This is a from-scratch kernel, not a port of the gfx950 MFMA version. The
 * structure (split-KV, online softmax in log2 space, fused attention sink,
 * LSE in the CK FMHA convention) follows
 * tasks/mi300/paged_attention_decode_minimal_hd64_mi300.cuh so the two produce
 * comparable numbers, but every data-distribution decision is different:
 * v_mfma_f32_16x16x16bf16 over a 64-lane wave and v_wmma_f32_16x16x32_bf16
 * over a 32-lane wave do not agree on which lane holds which element, so a
 * line-by-line translation would compile and silently attend to the wrong
 * tokens. CK is not an option either -- its WMMA path requires the
 * wmma-128b-insts target feature, which gfx1250 does not have.
 *
 * ── Fragment layout (validated in tests/mi450/test_gemm_wmma.hip) ──
 *
 * The instruction computes D[m][n] = sum_k A[m][k] * B[n][k].
 *
 *   A/B operand: lane l holds row (l % 16); the half selector (l / 16) picks
 *                which 16 of the 32 K values, so the lane's elements are
 *                [l % 16][(l / 16) * 16 .. + 15].
 *   D/C accum:   lane l holds column (l % 16), rows (l / 16) * 8 + [0..7].
 *
 * ── Why QK is computed as A=K, B=Q ──
 *
 * The softmax reduces over KV. With A=Q, B=K the accumulator would give each
 * lane one KV token and eight *queries*, scattering each query's scores across
 * 16 lanes and turning every max and every sum into a 16-lane butterfly.
 * Taking A=K, B=Q instead yields D[kv][q]: lane l holds query (l % 16) and KV
 * rows (l / 16) * 8 + [0..7]. Each lane then owns a contiguous run of KV for a
 * single query, so the reduction is 16 values in registers plus exactly one
 * cross-lane step. That is the same shape the MFMA kernel gets from its
 * scores[0..3] + two permlane swaps.
 *
 * ── Why KV_TILE is 32, not 16 ──
 *
 * WMMA's K depth is 32, and the PV product contracts over KV. A 16-token tile
 * would leave half of every PV instruction multiplying zeros. Two QK tiles of
 * 16 KV each fill one 32-deep PV, so the tile is 32 and the QK runs twice in M.
 *
 * ── The one redistribution that is genuinely needed ──
 *
 * After QK, lane l holds query (l % 16) with KV {(l/16)*8 + [0..7]} from tile 0
 * and {16 + (l/16)*8 + [0..7]} from tile 1 -- i.e. lanes 0-15 hold KV 0-7 and
 * 16-23, lanes 16-31 hold KV 8-15 and 24-31. The PV A-fragment wants lane l to
 * hold query (l % 16) with KV (l/16)*16 + [0..15] -- lanes 0-15 want KV 0-15,
 * lanes 16-31 want KV 16-31. Each lane therefore holds exactly one 8-element
 * group it needs and one it must trade with lane l^16, which is a single
 * __shfl_xor(.., 16) per value rather than a general permute.
 *
 * Head dims are split 4 ways across the 4 waves (wave w owns dims
 * [w*16, w*16+16)), matching the MFMA kernel's 4-warps-of-16-dims split. Each
 * wave recomputes the same QK scores, exactly as the MFMA kernel does.
 */
#pragma once

#include "mirage/persistent_kernel/arch_traits.cuh"
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

namespace kernel {
namespace mi450 {

#if defined(MIRAGE_ARCH_GFX1250)

typedef __bf16 attn_ab_t __attribute__((ext_vector_type(16)));
typedef float attn_acc_t __attribute__((ext_vector_type(8)));

static constexpr int ATTN_WMMA_M = 16;
static constexpr int ATTN_WMMA_N = 16;
static constexpr int ATTN_WMMA_K = 32;
// KV tokens consumed per iteration of the main loop. Fixed by the PV product's
// contraction depth (see the header comment), not tunable independently.
static constexpr int ATTN_KV_TILE = 32;

// exp2. The MFMA kernel reaches for `v_exp_f32` in inline asm; the intrinsic
// lowers to the same instruction on gfx1250 and keeps the file asm-free.
__device__ __forceinline__ float attn_exp2(float x) {
  return __builtin_amdgcn_exp2f(x);
}

// Convert the running softmax state to a natural-log LSE.
//
// This is a domain conversion, not a formatting detail. scale_s arrives with
// log2(e) already folded in (see src/kernel/task_register.cc:1650), so every
// score -- and therefore m_running -- is in log2 units, and the softmax below
// uses exp2 throughout. But both LSE consumers want natural log:
// merge_splitkv_ck_fmha multiplies the stored value by log2(e) on the way in
// ("CK FMHA stores LSE in natural log scale"), and the sink correction is
// sigmoid(LSE - sink) with sink in natural units.
//
// So the conversion has to be applied to the log2 part only:
//   LSE_nat = ln(sum exp2(s_i)) = m_log2 * ln2 + ln(l)
// Writing `m_running + logf(l)` instead -- mixing a log2 max with a natural
// log sum -- is off by m_running * (1 - ln2), which is a plain constant offset
// that leaves the attention output itself perfectly correct and only shows up
// once the LSE is consumed. Note tasks/mi300's decode kernel has exactly that
// expression; see the note in tests/mi450/test_attention_wmma.hip.
__device__ __forceinline__ float attn_lse_natural(float m_log2, float l) {
  return (l > 0.f) ? (m_log2 * 0.693147180559945f + logf(l)) : -1e30f;
}

// Trade an 8-element score group with lane l^16. Used to assemble the PV
// A-fragment; see "the one redistribution" in the header comment.
__device__ __forceinline__ void attn_swap_half(float const *__restrict__ src,
                                               float *__restrict__ dst) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    dst[i] = __shfl_xor(src[i], 16);
  }
}

// Decode attention over a paged KV cache.
//
// NUM_QO_PER_KV query heads share one KV head. Q is read from the workspace
// laid out [token][head][dim]; K and V pointers are pre-offset by
// kv_head_idx * HEAD_DIM, matching the convention in gang_attention_mi300.cuh.
//
// Launch geometry: 128 threads = 4 waves of 32. Callers that still assume the
// gfx950 256-thread block must halve it; a wave32 part needs half the threads
// to field the same four waves.
// MAX_SEQ_LEN is unused here -- this kernel keeps no per-sequence scratch, it
// streams KV a tile at a time. It stays in the parameter list, in mi300's
// position, so tasks/mi450/task_header.cuh can alias this over
// paged_attention_minimal_decode_hd64 without rewriting call sites.
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
    paged_attention_wmma_decode_hd64(void const *q_workspace_ptr,
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
                                     int sliding_window = 0,
                                     void const *sinks_ptr = nullptr) {
  using bf16 = __hip_bfloat16;
  static_assert(HEAD_DIM == 64, "This kernel is HD=64 only");
  static_assert(NUM_QO_PER_KV <= ATTN_WMMA_M,
                "query heads per KV head must fit the 16-row WMMA tile");

  int const req = request_id;
  int const query_start = qo_indptr[req];
  if (query_start == qo_indptr[req + 1]) {
    return;
  }

  int const first_page = kv_indptr[req];
  int const num_pages = kv_indptr[req + 1] - first_page;
  int const seqlen_k = (num_pages - 1) * PAGE_SIZE + kv_last_page_len[req];

  int const tid = threadIdx.x;
  int const wave_id = tid / mirage::arch::WAVE_SIZE; // 0..3, owns 16 head dims
  int const lane = tid % mirage::arch::WAVE_SIZE;    // 0..31
  int const frag_row = lane % 16; // operand row / accumulator column
  int const frag_half = lane / 16;

  // Constants for the accumulator mapping, named once so the store and the
  // softmax agree about what a lane holds.
  int const my_q = frag_row;           // query this lane's scores belong to
  int const kv_lo = frag_half * 8;     // first KV of this lane's group
  int const my_dim = wave_id * 16 + frag_row; // head dim this lane outputs

  char const *k_base = reinterpret_cast<char const *>(paged_k_cache_ptr);
  char const *v_base = reinterpret_cast<char const *>(paged_v_cache_ptr);

  // ── Q fragment, loaded once ──
  // Lane l needs Q[q = l%16][d = ks*32 + (l/16)*16 .. +15] for ks in {0,1},
  // which is 16 contiguous bf16 per k-step -- one 32-byte load, no LDS.
  // Lanes whose frag_row is past the real head count read head 0 and are
  // discarded at the store, mirroring the MFMA kernel's guard.
  attn_ab_t q_frag[HEAD_DIM / ATTN_WMMA_K];
  {
    int const q_head = (my_q < NUM_QO_PER_KV) ? my_q : 0;
    bf16 const *q_row =
        reinterpret_cast<bf16 const *>(q_workspace_ptr) +
        static_cast<long>(query_start) * Q_WORKSPACE_STRIDE +
        static_cast<long>(kv_head_idx * NUM_QO_PER_KV + q_head) * HEAD_DIM;
#pragma unroll
    for (int ks = 0; ks < HEAD_DIM / ATTN_WMMA_K; ++ks) {
      bf16 const *src = q_row + ks * ATTN_WMMA_K + frag_half * 16;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        q_frag[ks][i] = static_cast<__bf16>(src[i]);
      }
    }
  }

  // ── Sliding window and split-KV partitioning ──
  // Identical policy to the MFMA kernel: align the window start down to a tile
  // boundary, then carve the tile range into NUM_KV_CHUNKS contiguous slices.
  int kv_start = 0;
  if (sliding_window > 0 && seqlen_k > sliding_window) {
    kv_start = ((seqlen_k - sliding_window) / ATTN_KV_TILE) * ATTN_KV_TILE;
  }
  int effective_len = seqlen_k - kv_start;
  int ntiles = (effective_len + ATTN_KV_TILE - 1) / ATTN_KV_TILE;

  constexpr int LSE_STRIDE = NUM_KV_HEADS * NUM_KV_CHUNKS * NUM_QO_PER_KV;
  float *lse_slot = reinterpret_cast<float *>(lse_ptr) +
                    static_cast<long>(query_start) * LSE_STRIDE +
                    kv_head_idx * NUM_KV_CHUNKS * NUM_QO_PER_KV +
                    kv_chunk_idx * NUM_QO_PER_KV;

  if constexpr (NUM_KV_CHUNKS > 1) {
    int const tiles_per_chunk =
        (ntiles + NUM_KV_CHUNKS - 1) / NUM_KV_CHUNKS;
    int const chunk_first = kv_chunk_idx * tiles_per_chunk;
    int chunk_last = chunk_first + tiles_per_chunk;
    if (chunk_last > ntiles) {
      chunk_last = ntiles;
    }
    if (chunk_first >= ntiles) {
      // Empty chunk. The merge weights partials by exp(LSE - m_global), so the
      // output buffer can be left alone, but LSE must be stamped -inf or the
      // merge picks up whatever the previous iteration left in the slot.
      if (wave_id == 0 && frag_half == 0 && my_q < NUM_QO_PER_KV) {
        lse_slot[my_q] = -1e30f;
      }
      return;
    }
    kv_start += chunk_first * ATTN_KV_TILE;
    effective_len = (chunk_last - chunk_first) * ATTN_KV_TILE;
    int const remaining = seqlen_k - kv_start;
    if (effective_len > remaining) {
      effective_len = remaining;
    }
    ntiles = chunk_last - chunk_first;
  }
  if (ntiles == 0) {
    return;
  }

  // Page-indirected byte offset of a KV token's row. The cache is
  // [page][slot][KV_CACHE_STRIDE] and the pointers are pre-offset to this
  // kv_head, so only the token index varies here.
  auto kv_row_off = [&](int global_tok) -> long {
    int const pid = kv_indices[first_page + global_tok / PAGE_SIZE];
    return (static_cast<long>(pid) * PAGE_SIZE * KV_CACHE_STRIDE +
            static_cast<long>(global_tok % PAGE_SIZE) * KV_CACHE_STRIDE) *
           2;
  };

  // ── Main loop ──
  attn_acc_t o_acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  float m_running = -INFINITY; // per-lane, for query my_q
  float l_running = 0.f;

  for (int t = 0; t < ntiles; ++t) {
    int const tile_start = t * ATTN_KV_TILE;
    int const tile_len = min(effective_len - tile_start, ATTN_KV_TILE);

    // QK for the two 16-token halves of this tile.
    //
    // A = K: lane l supplies K[kv = half*16 + l%16][d = ks*32 + (l/16)*16 ..].
    // Those 16 dims are contiguous in the cache row, so this is one 32-byte
    // load per lane per k-step and needs no LDS staging at all -- the MFMA
    // kernel only stages K through LDS because its fragment wants 8 dims per
    // lane from a row shared by four lanes.
    float scores[2][8];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      attn_acc_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      int const kv_local = half * 16 + frag_row;
      bool const kv_valid = kv_local < tile_len;
      long const row_off =
          kv_valid ? kv_row_off(kv_start + tile_start + kv_local) : 0;

#pragma unroll
      for (int ks = 0; ks < HEAD_DIM / ATTN_WMMA_K; ++ks) {
        attn_ab_t k_frag;
        if (kv_valid) {
          bf16 const *src = reinterpret_cast<bf16 const *>(k_base + row_off) +
                            ks * ATTN_WMMA_K + frag_half * 16;
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            k_frag[i] = static_cast<__bf16>(src[i]);
          }
        } else {
          // Past the end of the sequence. Zeroing the operand keeps the
          // instruction well-defined; the score is masked to -inf below, so the
          // zeros never reach the softmax.
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            k_frag[i] = static_cast<__bf16>(0.f);
          }
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x32_bf16(
            false, k_frag, false, q_frag[ks], 0, acc, false, false);
      }

      // acc[i] is the score of query my_q against KV (half*16 + kv_lo + i).
      // scale_s carries the 1/sqrt(d) factor *and* log2(e), so the softmax
      // below is base-2 throughout and can use v_exp_f32 directly.
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        int const kv_idx = half * 16 + kv_lo + i;
        scores[half][i] =
            (kv_idx < tile_len) ? acc[i] * scale_s : -INFINITY;
      }
    }

    // Online softmax. This lane holds 16 of the tile's 32 KV for one query, so
    // the tile max is 16 values in registers plus a single exchange with the
    // lane holding the complementary KV group.
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
    // A tile that is entirely masked leaves new_max at -inf; exp2 of (-inf -
    // -inf) is NaN, so the first-tile case is special-cased to 0 exactly as the
    // MFMA kernel does.
    float const rescale =
        (m_running == -INFINITY) ? 0.f : attn_exp2(m_running - new_max);

    float probs[2][8];
    float tile_sum = 0.f;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float const p = (scores[half][i] == -INFINITY)
                            ? 0.f
                            : attn_exp2(scores[half][i] - new_max);
        probs[half][i] = p;
        tile_sum += p;
      }
    }
    tile_sum += __shfl_xor(tile_sum, 16);

    l_running = l_running * rescale + tile_sum;
    m_running = new_max;

    // Assemble the PV A-fragment: lane l wants query (l%16) with KV
    // (l/16)*16 + [0..15]. It holds the tile-0 group and the tile-1 group for
    // its own frag_half, so it keeps one and trades the other with lane l^16.
    float traded0[8], traded1[8];
    attn_swap_half(probs[0], traded0);
    attn_swap_half(probs[1], traded1);

    attn_ab_t p_frag;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      // frag_half 0 covers KV 0-15: its own tile-0 group, then the partner's.
      // frag_half 1 covers KV 16-31: the partner's tile-1 group, then its own.
      float const lo = (frag_half == 0) ? probs[0][i] : traded1[i];
      float const hi = (frag_half == 0) ? traded0[i] : probs[1][i];
      p_frag[i] = static_cast<__bf16>(lo);
      p_frag[8 + i] = static_cast<__bf16>(hi);
    }

    // V fragment: lane l supplies V^T[d = wave*16 + l%16][kv = (l/16)*16 + ..],
    // i.e. one head dim across 16 KV tokens. That walks the cache with the
    // token stride, so unlike K it is 16 scalar loads rather than one vector
    // load. Staging V through LDS transposed would make these contiguous; left
    // as direct loads until the kernel is correct, since FFM cannot measure the
    // difference anyway.
    attn_ab_t v_frag;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      int const kv_local = frag_half * 16 + i;
      if (kv_local < tile_len) {
        long const off = kv_row_off(kv_start + tile_start + kv_local);
        v_frag[i] = static_cast<__bf16>(
            reinterpret_cast<bf16 const *>(v_base + off)[my_dim]);
      } else {
        // Masked KV: probability is already 0, so the value is arbitrary, but
        // it must be finite -- a NaN here would survive the multiply by zero.
        v_frag[i] = static_cast<__bf16>(0.f);
      }
    }

    // Rescale the accumulator before adding this tile's contribution.
    //
    // The accumulator is distributed by *query row*: lane l holds queries
    // (l/16)*8 + [0..7]. The softmax stats are distributed by *query column*:
    // lane l holds query l%16. So the rescale factor for accumulator slot i has
    // to come from the lane that owns query (l/16)*8 + i. This is the one place
    // the two distributions disagree and a broadcast is unavoidable.
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      o_acc[i] *= __shfl(rescale, frag_half * 8 + i);
    }

    o_acc = __builtin_amdgcn_wmma_f32_16x16x32_bf16(
        false, p_frag, false, v_frag, 0, o_acc, false, false);
  }

  // ── Epilogue ──
  // o_acc[i] is the unnormalised output for query (frag_half*8 + i) at head dim
  // my_dim. The normaliser and the LSE live in the column distribution, so they
  // need the same broadcast the rescale did.
  bf16 const *d_sinks = reinterpret_cast<bf16 const *>(sinks_ptr);

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int const q_head = frag_half * 8 + i;
    if (q_head >= NUM_QO_PER_KV) {
      continue;
    }
    float const l_q = __shfl(l_running, q_head);
    float const m_q = __shfl(m_running, q_head);
    float inv_l = (l_q > 0.f) ? (1.f / l_q) : 0.f;

    if constexpr (NUM_KV_CHUNKS == 1) {
      // Fuse the per-head attention sink:
      //   out *= sigmoid(LSE - sink) = 1 / (1 + exp(sink - LSE))
      // which is what lets decode skip the standalone attention_sink task.
      if (d_sinks != nullptr) {
        float const lse = attn_lse_natural(m_q, l_q);
        float const sink = static_cast<float>(
            d_sinks[kv_head_idx * NUM_QO_PER_KV + q_head]);
        inv_l *= 1.f / (1.f + expf(sink - lse));
      }
      bf16 *o = reinterpret_cast<bf16 *>(output_ptr) +
                static_cast<long>(query_start) * Q_WORKSPACE_STRIDE +
                static_cast<long>(kv_head_idx * NUM_QO_PER_KV + q_head) *
                    HEAD_DIM;
      o[my_dim] = static_cast<bf16>(o_acc[i] * inv_l);
    } else {
      constexpr int O_S = LSE_STRIDE * HEAD_DIM;
      float *o = reinterpret_cast<float *>(output_ptr) +
                 static_cast<long>(query_start) * O_S +
                 static_cast<long>(kv_head_idx) * NUM_KV_CHUNKS *
                     NUM_QO_PER_KV * HEAD_DIM +
                 static_cast<long>(kv_chunk_idx) * NUM_QO_PER_KV * HEAD_DIM +
                 static_cast<long>(q_head) * HEAD_DIM;
      o[my_dim] = o_acc[i] * inv_l;
    }
  }

  // LSE, one writer per query head. Wave 0 lanes 0..NUM_QO_PER_KV-1 own the
  // column distribution, so they can write directly without a broadcast.
  if (wave_id == 0 && frag_half == 0 && my_q < NUM_QO_PER_KV) {
    // Natural log, matching the CK FMHA convention the merge step expects --
    // deliberately not log2, even though the softmax above runs base-2.
    lse_slot[my_q] = attn_lse_natural(m_running, l_running);
  }
}

#else // !MIRAGE_ARCH_GFX1250

// Host-pass declaration. MIRAGE_ARCH_GFX1250 comes from __gfx1250__, which the
// host pass never defines, so without this any shared header that names this
// kernel fails to parse for host. The prefill header's dispatch wrapper calls
// it for seqlen_q == 1, and gang_attention_mi300.cuh reaches that wrapper
// through the paged_attention_ck_fmha_split_kv_impl alias, so the host pass
// does see this name. An empty body is safe: HIP does not codegen __device__
// bodies in the host pass.
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
    paged_attention_wmma_decode_hd64(void const *,
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
