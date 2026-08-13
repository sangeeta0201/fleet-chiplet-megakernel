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
// MFMA MLA decode for GLM-5 (GlmMoeDsaForCausalLM), in the absorbed
// formulation. Structurally this is paged_attention_decode_minimal_hd64 with a
// different tiling; it reuses that file's MFMA / bf16-convert / exp2 helpers
// verbatim rather than re-deriving them.
//
// Absorbed MLA is MQA with asymmetric head dims. After folding W_UK into the
// query projection, each of the NUM_Q_HEADS query heads carries
//
//   q_absorbed = [ q_nope @ W_UK  (KV_LORA_RANK) | q_rope (QK_ROPE_HEAD_DIM) ]
//
// and the paged cache holds a *single* shared latent row per token
//
//   kv_row     = [ c_kv          (KV_LORA_RANK) | k_rope (QK_ROPE_HEAD_DIM) ]
//
// so QK reduces over QK_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM (576 for GLM-5)
// and PV accumulates over the leading KV_LORA_RANK dims of the *same* row —
// V is not a separate tensor. That halves the cache traffic relative to the
// GQA path and is why only one LDS staging buffer appears below.
//
// The kernel emits the KV_LORA_RANK-wide latent attention output per head;
// W_UV is folded into o_proj downstream, so no separate up-projection task is
// needed.
//
// Chiplet mapping. The GQA gang attention pins kv_head == xcd_id
// (gang_attention_mi300.cuh). MLA has one shared latent head, so instead the
// work is split two ways: the NUM_Q_HEADS query heads chunk into
// NUM_Q_GROUPS groups of 16 (one MFMA M tile each), and the sequence chunks
// into NUM_KV_CHUNKS. GLM-5 at NUM_Q_HEADS=64 with NUM_KV_CHUNKS=2 gives
// exactly 8 work items, one per XCD, and the latent cache is 1.15 KB/token so
// each XCD's slice stays L2-resident.
//
// A q-head group plays exactly the role a kv head plays in the GQA path, so
// the split-KV o_acc/lse_acc layouts below are bit-for-bit the ones
// merge_splitkv_ck_fmha already consumes: instantiate it with
// NUM_QO_HEADS_PER_KV = Q_HEADS_PER_GROUP, NUM_QO_GROUPS = NUM_Q_GROUPS,
// HEAD_DIM = KV_LORA_RANK, and pass q_head_group as its kv_head_idx.
//
// 256 threads, 4 warps x 64 lanes:
//   - QK_DIM / 32 QK MFMAs cover the reduction (18 for GLM-5)
//   - KV_LORA_RANK / 64 PV MFMAs per tile, each warp covering 16 output dims
//   - LDS: one KV_TILE x QK_DIM fp16 tile (18 KB for GLM-5), plain row-major
//
// Online softmax runs in log2 base (scale_s carries the log2(e) factor, same
// convention as the CK FMHA path).

#include <hip/hip_bf16.h>

// __mfma_qk_hd64 / __mfma_pv_hd64 / __fast_exp2_hd64 / __load_bf16x4_to_fp16 /
// __load_bf16x4_raw / __cvt_bf16x4_to_fp16 live here. They are tiling-agnostic
// despite the hd64 name.
#include "tasks/mi300/paged_attention_decode_minimal_hd64_mi300.cuh"

namespace kernel {

template <typename T,
          int NUM_Q_HEADS,
          int KV_LORA_RANK,
          int QK_ROPE_HEAD_DIM,
          int PAGE_SIZE,
          int MAX_SEQ_LEN,
          int NUM_KV_CHUNKS,
          int Q_WORKSPACE_STRIDE,
          int KV_CACHE_STRIDE,
          bool WRITE_THROUGH = false>
__device__ __noinline__ void
    mla_decode_absorbed(void const *q_workspace_ptr,
                        void const *paged_kv_cache_ptr,
                        void *output_ptr,
                        void *lse_ptr,
                        int const *qo_indptr,
                        int const *kv_indptr,
                        int const *kv_indices,
                        int const *kv_last_page_len,
                        int16_t request_id,
                        int q_head_group,
                        int kv_chunk_idx,
                        float scale_s) {
  using bf16 = __hip_bfloat16;

  constexpr int QK_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM;
  constexpr int Q_HEADS_PER_GROUP = 16; // MFMA M tile
  constexpr int NUM_Q_GROUPS = NUM_Q_HEADS / Q_HEADS_PER_GROUP;
  constexpr int KV_TILE = 16;
  constexpr int NUM_K32 = QK_DIM / 32;            // QK MFMA steps
  constexpr int NUM_V_BLOCKS = KV_LORA_RANK / 64; // PV MFMA steps
  constexpr int LDG_PER_TILE = QK_DIM / 64;       // 4-elt loads per thread

  static_assert(NUM_Q_HEADS % Q_HEADS_PER_GROUP == 0,
                "NUM_Q_HEADS must be a multiple of the MFMA M tile (16)");
  static_assert(KV_LORA_RANK % 64 == 0,
                "KV_LORA_RANK must be a multiple of 64 (4 warps x 16 dims)");
  static_assert(QK_DIM % 64 == 0,
                "QK_DIM must be a multiple of 64 so the LDS tile loads evenly");
  static_assert(KV_CACHE_STRIDE >= QK_DIM,
                "KV cache rows must hold the latent + rope dims");

  int const req = request_id;
  int const query_start = qo_indptr[req];
  if (query_start == qo_indptr[req + 1]) {
    return;
  }

  int const first_page = kv_indptr[req];
  int const num_pages = kv_indptr[req + 1] - first_page;
  int const seqlen_k = (num_pages - 1) * PAGE_SIZE + kv_last_page_len[req];

  int const tid = threadIdx.x;
  int const warp_id = tid / 64;
  int const lane = tid & 63;
  int const midx = lane & 15;
  int const kgrp = lane >> 4;

  char const *kv_base = reinterpret_cast<char const *>(paged_kv_cache_ptr);

  // LDS: KV[KV_TILE][QK_DIM] fp16 (18 KB for GLM-5). K and V are the same
  // rows, so unlike the GQA decode there is only one buffer.
  extern __shared__ char _mla_decode_smem[];
  _Float16 *lds_kv = reinterpret_cast<_Float16 *>(_mla_decode_smem);

  // 256 threads x 4 elements = 16 tok x 64 dim per round, LDG_PER_TILE rounds
  int const my_tok = tid / 16;
  int const my_dim0 = (tid % 16) * 4;

  // Load this lane's Q slice: head (q_head_group * 16 + midx), dims
  // [kc * 32 + kgrp * 8, +8) for each MFMA step.
  _Float16 qr[NUM_K32][8];
  {
    int const q_head = q_head_group * Q_HEADS_PER_GROUP + midx;
    char const *q_ptr = reinterpret_cast<char const *>(q_workspace_ptr) +
                        (static_cast<long>(query_start) * Q_WORKSPACE_STRIDE +
                         static_cast<long>(q_head) * QK_DIM) *
                            2;
#pragma unroll
    for (int kc = 0; kc < NUM_K32; kc++) {
      int dim_off = kc * 32 + kgrp * 8;
      uint4 raw = *reinterpret_cast<uint4 const *>(q_ptr + dim_off * 2);
      unsigned words[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float lo_f, hi_f;
        asm("v_cvt_f32_bf16 %0, %1" : "=v"(lo_f) : "v"(words[i]));
        asm("v_cvt_f32_bf16 %0, %1" : "=v"(hi_f) : "v"(words[i] >> 16));
        qr[kc][i * 2] = (_Float16)lo_f;
        qr[kc][i * 2 + 1] = (_Float16)hi_f;
      }
    }
  }

  int kv_start = 0;
  int effective_len = seqlen_k;
  int ntiles = (effective_len + KV_TILE - 1) / KV_TILE;

  // Split-KV partitioning: each chunk takes a contiguous slice of tiles.
  // For NUM_KV_CHUNKS==1 this is a no-op. An empty chunk stamps LSE=-inf and
  // exits so the merge gives it zero weight instead of picking up stale data.
  int chunk_first_tile = 0;
  int chunk_last_tile = ntiles;
  if constexpr (NUM_KV_CHUNKS > 1) {
    int tiles_per_chunk = (ntiles + NUM_KV_CHUNKS - 1) / NUM_KV_CHUNKS;
    chunk_first_tile = kv_chunk_idx * tiles_per_chunk;
    chunk_last_tile = chunk_first_tile + tiles_per_chunk;
    if (chunk_last_tile > ntiles) {
      chunk_last_tile = ntiles;
    }
    if (chunk_first_tile >= ntiles) {
      if (warp_id == 0 && kgrp == 0) {
        constexpr int LSE_STRIDE =
            NUM_Q_GROUPS * NUM_KV_CHUNKS * Q_HEADS_PER_GROUP;
        float *lse_out = reinterpret_cast<float *>(lse_ptr) +
                         static_cast<long>(query_start) * LSE_STRIDE +
                         q_head_group * NUM_KV_CHUNKS * Q_HEADS_PER_GROUP +
                         kv_chunk_idx * Q_HEADS_PER_GROUP + midx;
        if constexpr (WRITE_THROUGH) {
          float const empty = -1e30f;
          unsigned raw;
          __builtin_memcpy(&raw, &empty, 4);
          st_wt_u32((void *)lse_out, raw);
        } else {
          *lse_out = -1e30f;
        }
      }
      return;
    }
    kv_start += chunk_first_tile * KV_TILE;
    effective_len = (chunk_last_tile - chunk_first_tile) * KV_TILE;
    int remaining = seqlen_k - kv_start;
    if (effective_len > remaining) {
      effective_len = remaining;
    }
    ntiles = chunk_last_tile - chunk_first_tile;
  }
  if (ntiles == 0) {
    return;
  }

  // Byte offset of a token's latent row (bf16 = 2 bytes per element).
  auto get_kv_row = [&](int global_tok) -> long {
    int pid = kv_indices[first_page + global_tok / PAGE_SIZE];
    return (static_cast<long>(pid) * PAGE_SIZE * KV_CACHE_STRIDE +
            static_cast<long>(global_tok % PAGE_SIZE) * KV_CACHE_STRIDE) *
           2;
  };

  // ===== PROLOGUE: KV[0] -> LDS =====
  {
    int tile0_len = (KV_TILE < effective_len) ? KV_TILE : effective_len;
    long row = (my_tok < tile0_len) ? get_kv_row(kv_start + my_tok) : 0;
#pragma unroll
    for (int r = 0; r < LDG_PER_TILE; r++) {
      int dim = my_dim0 + r * 64;
      _Float16 kv_fp[4] = {0, 0, 0, 0};
      if (my_tok < tile0_len) {
        __load_bf16x4_to_fp16(kv_fp, kv_base + row + dim * 2);
      }
      *(uint64_t *)&lds_kv[my_tok * QK_DIM + dim] = *(uint64_t *)kv_fp;
    }
  }

  // Prefetch KV[1] -> registers as raw dwords
  uint2 kv_pre[LDG_PER_TILE];
  bool has_pre = false;
  if (ntiles > 1) {
    int tile1_len = ((effective_len - KV_TILE) < KV_TILE)
                        ? (effective_len - KV_TILE)
                        : KV_TILE;
    if (my_tok < tile1_len) {
      long row = get_kv_row(kv_start + KV_TILE + my_tok);
#pragma unroll
      for (int r = 0; r < LDG_PER_TILE; r++) {
        __load_bf16x4_raw(&kv_pre[r], kv_base + row + (my_dim0 + r * 64) * 2);
      }
      has_pre = true;
    }
  }

  // ===== MAIN LOOP =====
  __mfma_hd64_fp32x4 const zero4 = {0, 0, 0, 0};
  __mfma_hd64_fp32x4 o_acc[NUM_V_BLOCKS];
#pragma unroll
  for (int vb = 0; vb < NUM_V_BLOCKS; vb++) {
    o_acc[vb] = zero4;
  }
  float m_running = -INFINITY;
  float l_head[4] = {0, 0, 0, 0};

  for (int t = 0; t < ntiles; t++) {
    int tile_start = t * KV_TILE;
    int tile_len = ((effective_len - tile_start) < KV_TILE)
                       ? (effective_len - tile_start)
                       : KV_TILE;

    __syncthreads();

    // QK: lane (midx, kgrp) reads K[tok=midx][dim=kc*32+kgrp*8..+7]
    __mfma_hd64_fp32x4 scores = {0, 0, 0, 0};
#pragma unroll
    for (int kc = 0; kc < NUM_K32; kc++) {
      _Float16 kr[8];
      _Float16 const *k_ptr = &lds_kv[midx * QK_DIM + kc * 32 + kgrp * 8];
#pragma unroll
      for (int i = 0; i < 8; i++) {
        kr[i] = k_ptr[i];
      }
      scores = __mfma_qk_hd64(scores, kr, qr[kc]);
    }

    scores[0] *= scale_s;
    scores[1] *= scale_s;
    scores[2] *= scale_s;
    scores[3] *= scale_s;

#pragma unroll
    for (int h = 0; h < 4; h++) {
      if (kgrp * 4 + h >= tile_len) {
        scores[h] = -INFINITY;
      }
    }

    // Online softmax
    float tile_max =
        fmaxf(fmaxf(scores[0], scores[1]), fmaxf(scores[2], scores[3]));
    {
      float a = tile_max, b = tile_max;
      asm volatile("s_nop 1\n\tv_permlane32_swap_b32_e32 %0, %1"
                   : "+v"(a), "+v"(b));
      tile_max = fmaxf(a, b);
      a = tile_max;
      b = tile_max;
      asm volatile("s_nop 1\n\tv_permlane16_swap_b32_e32 %0, %1"
                   : "+v"(a), "+v"(b));
      tile_max = fmaxf(a, b);
    }

    float new_max = fmaxf(m_running, tile_max);
    float rescale =
        (m_running == -INFINITY) ? 0.0f : __fast_exp2_hd64(m_running - new_max);

    float w0 = __fast_exp2_hd64(scores[0] - new_max);
    float w1 = __fast_exp2_hd64(scores[1] - new_max);
    float w2 = __fast_exp2_hd64(scores[2] - new_max);
    float w3 = __fast_exp2_hd64(scores[3] - new_max);

    l_head[0] = l_head[0] * rescale + w0;
    l_head[1] = l_head[1] * rescale + w1;
    l_head[2] = l_head[2] * rescale + w2;
    l_head[3] = l_head[3] * rescale + w3;
    m_running = new_max;

    __mfma_hd64_fp16x4 pb;
    pb[0] = (_Float16)w0;
    pb[1] = (_Float16)w1;
    pb[2] = (_Float16)w2;
    pb[3] = (_Float16)w3;

    // PV: A = V transposed, lane (midx, kgrp) reads
    // V[tok=kgrp*4+j][dim=vb*64+warp_id*16+midx]. V is the latent prefix of
    // the same LDS rows the QK step just read.
#pragma unroll
    for (int vb = 0; vb < NUM_V_BLOCKS; vb++) {
      __mfma_hd64_fp16x4 va;
      _Float16 const *v_ptr =
          &lds_kv[(kgrp * 4) * QK_DIM + vb * 64 + warp_id * 16 + midx];
      va[0] = v_ptr[0 * QK_DIM];
      va[1] = v_ptr[1 * QK_DIM];
      va[2] = v_ptr[2 * QK_DIM];
      va[3] = v_ptr[3 * QK_DIM];
      o_acc[vb][0] *= rescale;
      o_acc[vb][1] *= rescale;
      o_acc[vb][2] *= rescale;
      o_acc[vb][3] *= rescale;
      o_acc[vb] = __mfma_pv_hd64(
          o_acc[vb], (_Float16 const *)&va, (_Float16 const *)&pb);
    }

    // The QK and PV steps both read the whole LDS tile, so the refill has to
    // wait for every lane rather than slotting between them the way the HD=64
    // decode does.
    __syncthreads();

    if (has_pre) {
#pragma unroll
      for (int r = 0; r < LDG_PER_TILE; r++) {
        _Float16 kv_fp[4];
        __cvt_bf16x4_to_fp16(kv_fp, kv_pre[r]);
        *(uint64_t *)&lds_kv[my_tok * QK_DIM + my_dim0 + r * 64] =
            *(uint64_t *)kv_fp;
      }
    } else if (t + 1 < ntiles) {
      // This lane's token is past the end of tile t+1: zero it so the stale
      // rows cannot leak into the next tile's QK.
      _Float16 zero[4] = {0, 0, 0, 0};
#pragma unroll
      for (int r = 0; r < LDG_PER_TILE; r++) {
        *(uint64_t *)&lds_kv[my_tok * QK_DIM + my_dim0 + r * 64] =
            *(uint64_t *)zero;
      }
    }

    // Prefetch KV[t+2]
    has_pre = false;
    if (t + 2 < ntiles) {
      int t2_start = (t + 2) * KV_TILE;
      int t2_len = ((effective_len - t2_start) < KV_TILE)
                       ? (effective_len - t2_start)
                       : KV_TILE;
      if (my_tok < t2_len) {
        long row = get_kv_row(kv_start + t2_start + my_tok);
#pragma unroll
        for (int r = 0; r < LDG_PER_TILE; r++) {
          __load_bf16x4_raw(&kv_pre[r], kv_base + row + (my_dim0 + r * 64) * 2);
        }
        has_pre = true;
      }
    }
  }

  // ===== Output =====
  // l_head lives on the kgrp lanes; fold them into one value per q head.
  float l_sum = l_head[0] + l_head[1] + l_head[2] + l_head[3];
  l_sum += __shfl_xor(l_sum, 16);
  l_sum += __shfl_xor(l_sum, 32);

  float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
  int const q_head_local = midx;

  if constexpr (NUM_KV_CHUNKS == 1) {
    constexpr int OUT_STRIDE = NUM_Q_HEADS * KV_LORA_RANK;
    bf16 *o = reinterpret_cast<bf16 *>(output_ptr) +
              static_cast<long>(query_start) * OUT_STRIDE +
              static_cast<long>(q_head_group * Q_HEADS_PER_GROUP +
                                q_head_local) *
                  KV_LORA_RANK;
#pragma unroll
    for (int vb = 0; vb < NUM_V_BLOCKS; vb++) {
#pragma unroll
      for (int h = 0; h < 4; h++) {
        int dim_offset = vb * 64 + warp_id * 16 + kgrp * 4 + h;
        o[dim_offset] = static_cast<bf16>(o_acc[vb][h] * inv_l);
      }
    }
  } else {
    // Split-KV partial output: float + LSE, laid out exactly the way
    // merge_splitkv_ck_fmha indexes it with q_head_group as kv_head_idx.
    constexpr int LSE_S = NUM_Q_GROUPS * NUM_KV_CHUNKS * Q_HEADS_PER_GROUP;
    constexpr int O_S = LSE_S * KV_LORA_RANK;
    float *o = reinterpret_cast<float *>(output_ptr) +
               static_cast<long>(query_start) * O_S +
               static_cast<long>(q_head_group) * NUM_KV_CHUNKS *
                   Q_HEADS_PER_GROUP * KV_LORA_RANK +
               static_cast<long>(kv_chunk_idx) * Q_HEADS_PER_GROUP *
                   KV_LORA_RANK +
               static_cast<long>(q_head_local) * KV_LORA_RANK;
#pragma unroll
    for (int vb = 0; vb < NUM_V_BLOCKS; vb++) {
      int dim_offset = vb * 64 + warp_id * 16 + kgrp * 4;
      if constexpr (WRITE_THROUGH) {
        // The four dims are contiguous and 4-aligned, so one
        // global_store_dwordx4 carries them past L2 in a single instruction.
        unsigned raw[4];
#pragma unroll
        for (int h = 0; h < 4; h++) {
          float v = o_acc[vb][h] * inv_l;
          __builtin_memcpy(&raw[h], &v, 4);
        }
        st_wt_u128(&o[dim_offset], raw[0], raw[1], raw[2], raw[3]);
      } else {
#pragma unroll
        for (int h = 0; h < 4; h++) {
          o[dim_offset + h] = o_acc[vb][h] * inv_l;
        }
      }
    }
  }

  // LSE for the split-KV merge. One lane per q head writes.
  //
  // scale_s carries log2(e), so m_running is a log2 exponent and l_sum sums
  // exp2 terms; merge_splitkv_ck_fmha wants natural-log LSE (it multiplies by
  // log2(e) on the way back in), hence the ln(2) on m_running.
  if (warp_id == 0 && kgrp == 0) {
    constexpr int LSE_STRIDE =
        NUM_Q_GROUPS * NUM_KV_CHUNKS * Q_HEADS_PER_GROUP;
    float *lse_out = reinterpret_cast<float *>(lse_ptr) +
                     static_cast<long>(query_start) * LSE_STRIDE +
                     q_head_group * NUM_KV_CHUNKS * Q_HEADS_PER_GROUP +
                     kv_chunk_idx * Q_HEADS_PER_GROUP + q_head_local;
    float lse_val = (l_sum > 0.0f)
                        ? (m_running * 0.69314718055994530942f + logf(l_sum))
                        : -1e30f;
    if constexpr (WRITE_THROUGH) {
      unsigned raw;
      __builtin_memcpy(&raw, &lse_val, 4);
      st_wt_u32((void *)lse_out, raw);
    } else {
      *lse_out = lse_val;
    }
  }
}

// Gang MLA decode: NUM_Q_GROUPS * NUM_KV_CHUNKS tasks broadcast to workers,
// one per XCD at the GLM-5 shape (4 q-head groups x 2 sequence chunks).
//
// tile_idx decomposition (q-head group fastest, so the first 8 tiles land on
// distinct XCDs the same way kv_head does in gang_attention_mi300.cuh):
//   q_head_group = tile_idx % NUM_Q_GROUPS
//   kv_chunk     = (tile_idx / NUM_Q_GROUPS) % NUM_KV_CHUNKS
//   request_id   = tile_idx / (NUM_Q_GROUPS * NUM_KV_CHUNKS)
//
// The latent cache append (c_kv + k_rope for the new token) is done by the
// preceding task, so unlike gang_attention_split_kv_kernel there is no
// Phase A here.
template <typename T,
          int NUM_Q_HEADS,
          int KV_LORA_RANK,
          int QK_ROPE_HEAD_DIM,
          int PAGE_SIZE,
          int MAX_SEQ_LEN,
          int NUM_KV_CHUNKS,
          int Q_WORKSPACE_STRIDE,
          int KV_CACHE_STRIDE,
          bool WRITE_THROUGH = false>
__device__ __noinline__ void
    gang_mla_decode_kernel(void const *q_workspace_ptr,
                           void const *paged_kv_cache_ptr,
                           void *output_ptr,
                           void *lse_ptr,
                           int const *qo_indptr,
                           int const *kv_indptr,
                           int const *kv_indices,
                           int const *kv_last_page_len,
                           int total_work_items,
                           int tile_idx,
                           float scale_s) {
  if (tile_idx >= total_work_items) {
    return;
  }

  constexpr int NUM_Q_GROUPS = NUM_Q_HEADS / 16;

  int const q_head_group = tile_idx % NUM_Q_GROUPS;
  int const kv_chunk_idx = (tile_idx / NUM_Q_GROUPS) % NUM_KV_CHUNKS;
  int16_t const request_id =
      static_cast<int16_t>(tile_idx / (NUM_Q_GROUPS * NUM_KV_CHUNKS));

  mla_decode_absorbed<T,
                      NUM_Q_HEADS,
                      KV_LORA_RANK,
                      QK_ROPE_HEAD_DIM,
                      PAGE_SIZE,
                      MAX_SEQ_LEN,
                      NUM_KV_CHUNKS,
                      Q_WORKSPACE_STRIDE,
                      KV_CACHE_STRIDE,
                      WRITE_THROUGH>(q_workspace_ptr,
                                       paged_kv_cache_ptr,
                                       output_ptr,
                                       lse_ptr,
                                       qo_indptr,
                                       kv_indptr,
                                       kv_indices,
                                       kv_last_page_len,
                                       request_id,
                                       q_head_group,
                                       kv_chunk_idx,
                                       scale_s);
}

} // namespace kernel
