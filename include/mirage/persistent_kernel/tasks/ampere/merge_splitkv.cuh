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

// this kernel merges the result of one KV head chunk back to the full KV
// cache，
// it taks the output of multitoken_paged_attention_task_impl_32_64_split_kv and
// the log exp sum as input,

// ---------------------------------------------------------------------------
// MPK_MERGE_GLOBAL: name the address space of the split-KV merge's two reads.
//
// merge_splitkv_ck_fmha's `lse_ptr` / `o_ptr` are plain `float const *`, so the
// compiler cannot prove they are global and emits `flat_load_dword` for both --
// 32 + 32 of them per instantiation in the fused MLA layer, and another 16 + 16
// each in worker_kernel / persistent_kernel. They are the o_acc / lse_acc HBM
// staging buffers, so addrspace(1) is what they actually are.
//
// This body is the online-softmax rescan: NUM_KV_CHUNKS fully-unrolled
// dependent (lse, o) pairs per output dim. Flat raises lgkmcnt as well as
// vmcnt, so every one of those loads fences against LDS traffic it never
// touches. Unlike the MoE k-loop negative, there is no loop-carried s_waitcnt
// count to change here -- the whole chain is unrolled flat.
//
// MEASURED NEUTRAL. It does exactly what it says to the image -- fused layer
// 280 -> 216 flat, worker_kernel 126 -> 94, persistent_kernel 133 -> 101, all
// of the delta being these two lines -- and the wall does not move:
//
//   GLM-5, NP=4, devices 4-7, bs=1, one session
//   0   10.068 10.087 10.156 10.110 10.078 10.086 10.103 10.063   n=8 mean 10.094
//   1   10.020 10.065 10.123 10.092 10.084 10.068                 n=6 mean 10.075
//
// -0.019 against a 0.26 ms n=1 noise floor: not resolvable, and the ranges
// interleave. Default is 1 anyway because addrspace(1) is what these pointers
// ACTUALLY are, and the two prior wins in this class both came from saying so
// -- but do not count this as one of them.
//
// Taken with MPK_EP_ASSUME_DIRECT (dead flat, also neutral), this is the second
// measurement saying the flat-op vein is spent at NP=4: even flat that EXECUTES
// buys nothing once it is outside a k-loop and outside a phase that is on the
// critical path. The merge is not.
// ---------------------------------------------------------------------------
#ifndef MPK_MERGE_GLOBAL
#define MPK_MERGE_GLOBAL 1
#endif

// MPK_MERGE_TWO_PASS: gpt-oss's KV-outer two-pass split-KV merge
// (amd_mi355_gpt_oss120b f3c40ed: merge 3.11 -> 2.61 us at 31 chunks; ships
// default ON there). The running-max form in merge_splitkv_ck_fmha is a serial
// chain over NUM_KV_CHUNKS: chunk k's rescale needs m_global from chunk k-1.
// Reducing the lse values to m_max first makes every weight known up front,
// so all o loads are independent. Same result in exact arithmetic; the low
// bits move (FMA contraction / reassociation), so the gate is numerical, not a
// text hash. GLM ships VAL_PER_THREAD == 1 (HEAD_DIM 512, DIM_SPLITS 32), so
// the KV-outer half's per-dim hoisting is moot here and only the chain break
// matters.
//
// MEASURED NULL 2026-09-23, 1024/1024, chunks=32, GPUs 4-7, one MPK_BAR_SKEW=3
// pair, both G1 PASS with coherent text:
//   ISA        v_exp_f32 328 -> 200 (4 inlined merges x 32), 147022 -> 145850
//              insns. Standalone merge<wt=1, splits=32, 32 chunks>: loads in
//              flight before the first wait 16 -> 36, 932 -> 647 insns.
//   merge      critical-worker busy 4.16 -> 3.80 us/layer, span 10.12 -> 9.95
//   decode_min (instrumented) 14.588 -> 14.555 ms
// ~0.02 ms/token, the size gpt-oss measured (-0.017). The running form was
// already 15 loads deep, and the merge is ~4 us of a ~185 us layer. Correct:
// tests/standalone/test_mla_merge.hip passes at 16 and 32 chunks with the
// same max error (1.95e-3, bf16 rounding) as the running form. Kept off.
#ifndef MPK_MERGE_TWO_PASS
#define MPK_MERGE_TWO_PASS 0
#endif

namespace kernel {

#if MPK_MERGE_GLOBAL
using _mrg_gf32 = __attribute__((address_space(1))) float const *;
#define MPK_MRG_G(p) ((::kernel::_mrg_gf32)(p))
#else
#define MPK_MRG_G(p) (p)
#endif

template <typename T,
          int NUM_QO_HEADS_PER_KV,
          int NUM_KV_HEADS,
          int NUM_QO_GROUPS,
          int HEAD_DIM,
          int MAX_TOKENS = 8,
          bool PARTITION_KV = true,
          int NUM_KV_CHUNKS = 1,
          int KV_CHUNK_SIZE = 256,
          int PAGE_SIZE = 4096,
          typename OaccT = T>
__device__ __forceinline__ void
    merge_splitkv(void const *lse,
                  void const *o,
                  int const *qo_indptr_buffer_ptr,
                  int const *paged_kv_indptr_buffer_ptr,
                  int const *paged_kv_last_page_len_buffer_ptr,
                  int16_t request_id,
                  void *output,
                  int merge_task_offset) {
#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  unsigned long long _t0 = __builtin_amdgcn_s_memrealtime();
#endif
  OaccT const *o_ptr = reinterpret_cast<OaccT const *>(o);
  T *output_ptr = reinterpret_cast<T *>(output);
  float const *lse_ptr = reinterpret_cast<float const *>(lse);
  // constexpr int GLOBAL_ITERS_M = (NUM_QO_HEADS_PER_KV + 64 - 1) / 64;

  int const first_page_pos = paged_kv_indptr_buffer_ptr[request_id];
  int const last_page_pos = paged_kv_indptr_buffer_ptr[request_id + 1];
  int const num_pages = last_page_pos - first_page_pos;
  int seq_len = (num_pages - 1) * PAGE_SIZE +
                paged_kv_last_page_len_buffer_ptr[request_id];

  int num_chunks = (seq_len + KV_CHUNK_SIZE - 1) / KV_CHUNK_SIZE;

  // size of o and output is NUM_QO_HEADS_PER_KV * MAX_TOKENS * 128
  // let each thread process one
  //  constexpr int NUM_QO_PER_KV = NUM_QO_HEADS_PER_KV / NUM_KV_HEADS;
  //  constexpr int NUM_Q = MAX_TOKENS * NUM_QO_PER_KV;
  //  constexpr int GLOBAL_ITERS_M = (NUM_Q + 64 - 1) / 64;

  int const first_token_pos = qo_indptr_buffer_ptr[request_id];
  int const last_token_pos = qo_indptr_buffer_ptr[request_id + 1];
  // Exit the current task is number of query tokens is zero
  if (first_token_pos == last_token_pos) {
    return;
  }
  int const num_tokens = last_token_pos - first_token_pos;

  constexpr int THREADS_PER_TOKEN = 16; // let 16 threads process one head
  constexpr int VAL_PER_THREAD = HEAD_DIM / THREADS_PER_TOKEN;
  constexpr int num_groups = NUM_THREADS / THREADS_PER_TOKEN;

  int thread_in_group = threadIdx.x % THREADS_PER_TOKEN;
  int group_id = threadIdx.x / THREADS_PER_TOKEN;
  int head_partition = thread_in_group;

  // let 16 threads to process one head_dim
#pragma unroll 1
  for (int tok = group_id; tok < num_tokens * NUM_QO_HEADS_PER_KV;
       tok += num_groups) {

    int token_idx = tok / NUM_QO_HEADS_PER_KV;
    int head_idx = tok % NUM_QO_HEADS_PER_KV;

#pragma unroll 1
    for (int i = 0; i < VAL_PER_THREAD; ++i) {
      float m_global = -inf;
      float d_global = 1.f;
      float o_global = 0.f;
#pragma unroll
      for (int kv_idx = 0; kv_idx < num_chunks; ++kv_idx) {
        // process 8 tokens
        float m_prev = m_global,
              d_prev = d_global; // save previous values
        // int lse_offset = kv_idx * (MAX_TOKENS * NUM_QO_HEADS_PER_KV) + tok;
        // int o_offset = (kv_idx * (MAX_TOKENS * NUM_QO_HEADS_PER_KV) + tok) *
        // HEAD_DIM +  head_partition * VAL_PER_THREAD + i;

        int lse_offset = head_idx + kv_idx * NUM_QO_HEADS_PER_KV +
                         (first_token_pos + token_idx) * NUM_QO_GROUPS *
                             NUM_KV_CHUNKS * NUM_QO_HEADS_PER_KV;
        // int lse_offset = merge_task_offset * NUM_QO_HEADS_PER_KV + head_idx +
        // kv_idx * NUM_QO_HEADS_PER_KV + token_idx * NUM_QO_GROUPS *
        // NUM_KV_CHUNKS * NUM_QO_HEADS_PER_KV;
        int o_offset =
            lse_offset * HEAD_DIM + head_partition * VAL_PER_THREAD + i;

        float other_m = lse_ptr[lse_offset], other_d = 1;
        m_global = max(m_prev, other_m);
        d_global = d_prev * ptx_exp2(m_prev - m_global) +
                   other_d * ptx_exp2(other_m - m_global);
        // accumulate o
        float other_o = (float)o_ptr[o_offset];

        o_global = o_global * ptx_exp2(m_prev - m_global) +
                   other_o * ptx_exp2(other_m - m_global);
      }
      T out_val = (T)__fdividef(o_global, d_global);
      output_ptr[(first_token_pos + token_idx) * NUM_QO_GROUPS *
                     NUM_QO_HEADS_PER_KV * HEAD_DIM +
                 head_idx * HEAD_DIM + head_partition * VAL_PER_THREAD + i] =
          out_val;
    }
  }
#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  __syncthreads();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    unsigned long long _dur = (__builtin_amdgcn_s_memrealtime() - _t0) * 10;
    printf("[ATTN_MERGE] dur_us=%.1f\n", (double)_dur / 1000.0);
  }
#endif
}

// G consecutive chunks of the online-softmax merge: all 2G loads first, then
// the updates in exactly the flat loop's order, so the result is the same.
template <int G, typename LseP, typename OP>
__device__ __forceinline__ void merge_online_group(LseP lse_g,
                                                   OP o_g,
                                                   int lse_linear0,
                                                   int lse_step,
                                                   int head_dim,
                                                   int o_col,
                                                   float &m_global,
                                                   float &d_global,
                                                   float &o_global) {
  float other_m[G], other_o[G];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    int const lse_linear = lse_linear0 + g * lse_step;
    other_m[g] = lse_g[lse_linear] * 1.44269504088896340736f;
    other_o[g] = o_g[lse_linear * head_dim + o_col];
  }
#pragma unroll
  for (int g = 0; g < G; ++g) {
    float const m_prev = m_global, d_prev = d_global;
    m_global = max(m_prev, other_m[g]);
    d_global = d_prev * ptx_exp2(m_prev - m_global) +
               ptx_exp2(other_m[g] - m_global);
    o_global = o_global * ptx_exp2(m_prev - m_global) +
               other_o[g] * ptx_exp2(other_m[g] - m_global);
  }
}

// CK FMHA merge variant: reads float o_acc/lse_acc with interleaved kv_head
// layout lse layout:  lse[token * (NUM_QO_GROUPS * NUM_KV_CHUNKS * QO_PER_KV)
//                  + kv_head * NUM_KV_CHUNKS * QO_PER_KV
//                  + chunk * QO_PER_KV + head]
// o layout:    same indexing * HEAD_DIM + d
// output layout: output[token * NUM_QO_GROUPS * QO_PER_KV * HEAD_DIM
//                       + kv_head * QO_PER_KV * HEAD_DIM + head * HEAD_DIM + d]
//
// DIM_SPLITS fans one kv_head's merge across several tasks, each owning a
// contiguous HEAD_DIM / DIM_SPLITS slice. The parallelism here is otherwise
// capped at NUM_QO_GROUPS blocks, which is fine at GQA head dims but starves
// on absorbed MLA: HEAD_DIM is the 512-wide latent, so a 16-head group leaves
// each thread carrying HEAD_DIM / 16 = 32 fully-unrolled online-softmax chains
// of NUM_KV_CHUNKS dependent loads apiece, on one CU of 256. Splitting the dim
// range costs nothing but a re-read of the (tiny) LSE column per slice -- the
// o_acc reads stay perfectly partitioned. DIM_SPLITS == 1 is the original
// code path exactly.
//
// The slice index rides in on kv_head_idx rather than a separate argument so
// that callers keep passing merge_task_offset straight through; the kernel
// decomposes it because only it knows DIM_SPLITS.
template <typename T,
          int NUM_QO_HEADS_PER_KV,
          int NUM_QO_GROUPS,
          int HEAD_DIM,
          int NUM_KV_CHUNKS,
          int KV_CHUNK_SIZE = 128,
          int PAGE_SIZE = 4096,
          bool WRITE_THROUGH = false,
          int DIM_SPLITS = 1,
          // Merge only heads [head_base, head_base + HEADS_N) of the group:
          // GLM's head-local decode fills a 16-head group of which a rank at
          // NP=8 owns 8, and the other 8 are never read.
          int HEADS_N = NUM_QO_HEADS_PER_KV>
__device__ __forceinline__ void
    merge_splitkv_ck_fmha(float const *lse_ptr,
                          float const *o_ptr,
                          int const *qo_indptr_buffer_ptr,
                          int const *paged_kv_indptr_buffer_ptr,
                          int const *paged_kv_last_page_len_buffer_ptr,
                          int16_t request_id,
                          T *output_ptr,
                          int merge_task_offset,
                          void const *sinks_ptr = nullptr,
                          int head_base = 0,
                          // qo_indptr[request_id] and [request_id + 1], when
                          // the caller read them ahead of its barrier; the
                          // lse/o addresses depend on them.
                          int first_token_pos_hint = -1,
                          int last_token_pos_hint = -1,
                          // MPK_ATTN_OUT_LL: each bf16 pair leaves as one
                          // (ll_epoch << 32 | pair) word at ll_out, indexed
                          // by (global head - ll_head0) * HEAD_DIM + column,
                          // halved; only the halves path can produce it.
                          unsigned long long *ll_out = nullptr,
                          unsigned ll_epoch = 0,
                          int ll_head0 = 0,
                          // MPK_DEC_MERGE_LL: the decode's partials as epoch
                          // words (o: bf16 pairs at half the f32 index, lse:
                          // f32), validated here; halves path only.
                          unsigned long long const *ll_o_in = nullptr,
                          unsigned long long const *ll_lse_in = nullptr,
                          unsigned ll_in_epoch = 0) {
  static_assert(HEADS_N >= 1 && HEADS_N <= NUM_QO_HEADS_PER_KV,
                "HEADS_N is a sub-range of the group");

  static_assert(HEAD_DIM % DIM_SPLITS == 0,
                "DIM_SPLITS must divide HEAD_DIM evenly");
  int const kv_head_idx =
      (DIM_SPLITS == 1) ? merge_task_offset : (merge_task_offset / DIM_SPLITS);
  int const dim_slice =
      (DIM_SPLITS == 1) ? 0 : (merge_task_offset % DIM_SPLITS);

  int const first_token_pos = first_token_pos_hint >= 0
                                  ? first_token_pos_hint
                                  : qo_indptr_buffer_ptr[request_id];
  int const last_token_pos = first_token_pos_hint >= 0
                                 ? last_token_pos_hint
                                 : qo_indptr_buffer_ptr[request_id + 1];
  if (first_token_pos == last_token_pos) {
    return;
  }
  int const num_tokens = last_token_pos - first_token_pos;

  // Use NUM_KV_CHUNKS (template param) as the chunk count — the CK FMHA
  // pipeline processes exactly NUM_KV_CHUNKS chunks worth of data into
  // o_acc/lse_acc
  constexpr int num_chunks = NUM_KV_CHUNKS;

  // Full token stride matches CK FMHA's LSE_STRIDE = NUM_KV_HEADS *
  // NUM_KV_CHUNKS * QO_PER_KV
  constexpr int LSE_TOKEN_STRIDE =
      NUM_QO_GROUPS * NUM_KV_CHUNKS * NUM_QO_HEADS_PER_KV;
  // Offset to this kv_head's slice within one token
  int const lse_kv_offset = kv_head_idx * NUM_KV_CHUNKS * NUM_QO_HEADS_PER_KV;
  // Output stride (full width across all kv_heads)
  constexpr int OUT_TOKEN_STRIDE =
      NUM_QO_GROUPS * NUM_QO_HEADS_PER_KV * HEAD_DIM;

  // Use 32 threads per head to match work items (1 token * 8 heads = 8 groups =
  // 256/32)
  // A sub-range keeps 16 threads per head when the slice is too narrow for
  // 32 (GLM: 512 / 32 splits = 16 dims); half the groups then idle, and each
  // item reads half the o_acc.
  constexpr int THREADS_PER_TOKEN =
      (NUM_QO_HEADS_PER_KV <= 8 ||
       (HEADS_N <= 8 && HEAD_DIM / DIM_SPLITS >= 32))
          ? 32
          : 16;
  constexpr int DIM_PER_SLICE = HEAD_DIM / DIM_SPLITS;
  constexpr int VAL_PER_THREAD = DIM_PER_SLICE / THREADS_PER_TOKEN;
  constexpr int num_groups = NUM_THREADS / THREADS_PER_TOKEN;
  static_assert(VAL_PER_THREAD >= 1,
                "DIM_SPLITS too large: fewer than one dim per thread");
  int const dim_base = dim_slice * DIM_PER_SLICE;

  int thread_in_group = threadIdx.x % THREADS_PER_TOKEN;
  int group_id = threadIdx.x / THREADS_PER_TOKEN;

  auto const lse_g = MPK_MRG_G(lse_ptr);
  auto const o_g = MPK_MRG_G(o_ptr);

  // Optional sink correction (GPT-OSS): out *= 1 / (1 + exp(sink -
  // LSE_natural)) Layout: sinks[num_q_heads] in bf16, indexed by
  // kv_head_idx*NUM_QO_HEADS_PER_KV + head.
  using __sink_bf16 = __hip_bfloat16;
  __sink_bf16 const *d_sinks = reinterpret_cast<__sink_bf16 const *>(sinks_ptr);

#if MPK_MERGE_HALVES
  // MPK_MERGE_HALVES: one row, one dim per thread, and half the groups idle
  // (GLM at NP=8: 8 heads, 16 groups of 16). Give every head two groups --
  // group g takes head g >> 1 over chunk half g & 1 -- so each thread issues
  // its half's lse and o loads together in one round trip instead of two
  // dependent 32-chunk batches, and the partner halves, 16 lanes apart in
  // one wave, combine with a shuffle.
  if constexpr (HEADS_N * 2 == num_groups && VAL_PER_THREAD == 1 &&
                NUM_KV_CHUNKS % 2 == 0 && WRITE_THROUGH &&
                THREADS_PER_TOKEN == 16) {
    if (num_tokens == 1 && sinks_ptr == nullptr) {
      constexpr int HC = NUM_KV_CHUNKS / 2;
      int const head_idx = head_base + (group_id >> 1);
      int const half = group_id & 1;
      int const lse0 = head_idx + first_token_pos * LSE_TOKEN_STRIDE +
                       lse_kv_offset + half * HC * NUM_QO_HEADS_PER_KV;
      int const o_col = dim_base + thread_in_group;
      float l[HC], o[HC];
      if (ll_o_in != nullptr) {
        unsigned long long lw[HC], ow[HC];
        while (true) {
#pragma unroll
          for (int c = 0; c < HC; ++c) {
            int const ll = lse0 + c * NUM_QO_HEADS_PER_KV;
            asm volatile("global_load_dwordx2 %0, %1, off sc0 sc1"
                         : "=v"(lw[c])
                         : "v"(ll_lse_in + ll)
                         : "memory");
            asm volatile("global_load_dwordx2 %0, %1, off sc0 sc1"
                         : "=v"(ow[c])
                         : "v"(ll_o_in + ((ll * HEAD_DIM + o_col) >> 1))
                         : "memory");
          }
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          bool ok = true;
#pragma unroll
          for (int c = 0; c < HC; ++c) {
            // An empty chunk stamps lse -1e30 and no o.
            bool const empty = (unsigned)lw[c] == 0xF149F2CAu;
            ok = ok && (unsigned)(lw[c] >> 32) == ll_in_epoch &&
                 (empty || (unsigned)(ow[c] >> 32) == ll_in_epoch);
          }
          if (ok) {
            break;
          }
          __builtin_amdgcn_s_sleep(1);
        }
#pragma unroll
        for (int c = 0; c < HC; ++c) {
          bool const empty = (unsigned)lw[c] == 0xF149F2CAu;
          l[c] = __uint_as_float((unsigned)lw[c]) * 1.44269504088896340736f;
          unsigned const half16 =
              (o_col & 1) ? (unsigned)(ow[c] >> 16) & 0xFFFFu
                          : (unsigned)ow[c] & 0xFFFFu;
          o[c] = empty ? 0.f : __uint_as_float(half16 << 16);
        }
      } else {
#pragma unroll
      for (int c = 0; c < HC; ++c) {
        int const ll = lse0 + c * NUM_QO_HEADS_PER_KV;
        l[c] = lse_g[ll] * 1.44269504088896340736f;
        o[c] = o_g[ll * HEAD_DIM + o_col];
      }
      }
      float m = -inf;
#pragma unroll
      for (int c = 0; c < HC; ++c) {
        m = max(m, l[c]);
      }
      float d = 0.f, os = 0.f;
#pragma unroll
      for (int c = 0; c < HC; ++c) {
        float const w = ptx_exp2(l[c] - m);
        d += w;
        os += w * o[c];
      }
      float const mp = __shfl_xor(m, 16);
      float const dp = __shfl_xor(d, 16);
      float const op = __shfl_xor(os, 16);
      float const mm = max(m, mp);
      float const a = ptx_exp2(m - mm);
      float const b = ptx_exp2(mp - mm);
      float const out_f = __fdividef(os * a + op * b, d * a + dp * b);
      float const partner = __shfl_down(out_f, 1, THREADS_PER_TOKEN);
      if (half == 0 && (thread_in_group & 1) == 0) {
        int const out_offset = first_token_pos * OUT_TOKEN_STRIDE +
                               kv_head_idx * NUM_QO_HEADS_PER_KV * HEAD_DIM +
                               head_idx * HEAD_DIM + o_col;
        __hip_bfloat16 v0 = (__hip_bfloat16)out_f;
        __hip_bfloat16 v1 = (__hip_bfloat16)partner;
        uint16_t lo, hi;
        memcpy(&lo, &v0, 2);
        memcpy(&hi, &v1, 2);
        uint32_t const packed = lo | ((uint32_t)hi << 16);
        if (ll_out != nullptr) {
          int const gh = kv_head_idx * NUM_QO_HEADS_PER_KV + head_idx;
          st_wt_u64((void *)(ll_out + ((gh - ll_head0) * HEAD_DIM + o_col) / 2),
                    ((unsigned long long)ll_epoch << 32) | packed);
        } else {
          st_wt_u32((void *)&output_ptr[out_offset], packed);
        }
      }
      return;
    }
  }
#endif
  // The consumer polls words only the halves path writes, and the decode's
  // words only the halves path reads.
  if (ll_out != nullptr || ll_o_in != nullptr) {
    __builtin_trap();
  }

#pragma unroll
  for (int tok = group_id; tok < num_tokens * HEADS_N; tok += num_groups) {
    int token_idx = tok / HEADS_N;
    int head_idx = head_base + tok % HEADS_N;

    // Load this head's sink once (independent of dim).
    float sink_val_log2 = 0.0f;
    if (sinks_ptr != nullptr) {
      sink_val_log2 =
          static_cast<float>(
              d_sinks[kv_head_idx * NUM_QO_HEADS_PER_KV + head_idx]) *
          1.44269504088896340736f; // convert sink from ln to log2
    }

    // Base output offset for this token+head (dims start here)
    int out_offset_base = (first_token_pos + token_idx) * OUT_TOKEN_STRIDE +
                          kv_head_idx * NUM_QO_HEADS_PER_KV * HEAD_DIM +
                          head_idx * HEAD_DIM + dim_base +
                          thread_in_group * VAL_PER_THREAD;

    // Compute all VAL_PER_THREAD dimensions
    float out_vals[VAL_PER_THREAD];
#if MPK_MERGE_TWO_PASS
    {
      int const lse_base0 = head_idx +
                            (first_token_pos + token_idx) * LSE_TOKEN_STRIDE +
                            lse_kv_offset;
      // Pass 1: every lse, independent loads, then the max.
      float lse_log2[NUM_KV_CHUNKS];
#pragma unroll
      for (int kv_idx = 0; kv_idx < num_chunks; ++kv_idx) {
        lse_log2[kv_idx] = lse_g[lse_base0 + kv_idx * NUM_QO_HEADS_PER_KV] *
                           1.44269504088896340736f;
      }
      float m_max = -inf;
#pragma unroll
      for (int kv_idx = 0; kv_idx < num_chunks; ++kv_idx) {
        m_max = max(m_max, lse_log2[kv_idx]);
      }
      // Pass 2: weights known, so the o loads carry no dependency on each
      // other. Empty chunks (lse = -1e30) get weight 0, as in the running form.
      float d_sum = 0.f;
      float o_sum[VAL_PER_THREAD];
#pragma unroll
      for (int i = 0; i < VAL_PER_THREAD; ++i) {
        o_sum[i] = 0.f;
      }
#pragma unroll
      for (int kv_idx = 0; kv_idx < num_chunks; ++kv_idx) {
        float const w = ptx_exp2(lse_log2[kv_idx] - m_max);
        d_sum += w;
        int const o_base =
            (lse_base0 + kv_idx * NUM_QO_HEADS_PER_KV) * HEAD_DIM + dim_base +
            thread_in_group * VAL_PER_THREAD;
#pragma unroll
        for (int i = 0; i < VAL_PER_THREAD; ++i) {
          o_sum[i] += o_g[o_base + i] * w;
        }
      }
#pragma unroll
      for (int i = 0; i < VAL_PER_THREAD; ++i) {
        float out_f = __fdividef(o_sum[i], d_sum);
        if (sinks_ptr != nullptr) {
          float lse_log2_tot = m_max + ptx_log2(d_sum);
          float diff = sink_val_log2 - lse_log2_tot;
          out_f *= __fdividef(1.0f, 1.0f + ptx_exp2(diff));
        }
        out_vals[i] = out_f;
      }
    }
#else
#pragma unroll
    for (int i = 0; i < VAL_PER_THREAD; ++i) {
      float m_global = -inf;
      float d_global = 1.f;
      float o_global = 0.f;
      if constexpr (NUM_KV_CHUNKS > 32) {
        // Past 32 chunks the full unroll below either fails (128 compiles to
        // a rolled loop that waits on every chunk's loads) or does not finish
        // compiling (64 ran 18 minutes). Groups of 32: one exposed load
        // latency per group, same update order.
        constexpr int MRG_G = 32;
        constexpr int MRG_FULL = NUM_KV_CHUNKS / MRG_G;
        constexpr int MRG_TAIL = NUM_KV_CHUNKS % MRG_G;
        int const lse_linear0 = head_idx +
                                (first_token_pos + token_idx) * LSE_TOKEN_STRIDE +
                                lse_kv_offset;
        int const o_col = dim_base + thread_in_group * VAL_PER_THREAD + i;
#pragma unroll 1
        for (int grp = 0; grp < MRG_FULL; ++grp) {
          merge_online_group<MRG_G>(
              lse_g, o_g, lse_linear0 + grp * MRG_G * NUM_QO_HEADS_PER_KV,
              NUM_QO_HEADS_PER_KV, HEAD_DIM, o_col, m_global, d_global,
              o_global);
        }
        if constexpr (MRG_TAIL > 0) {
          merge_online_group<MRG_TAIL>(
              lse_g, o_g, lse_linear0 + MRG_FULL * MRG_G * NUM_QO_HEADS_PER_KV,
              NUM_QO_HEADS_PER_KV, HEAD_DIM, o_col, m_global, d_global,
              o_global);
        }
      } else {
#pragma unroll
      for (int kv_idx = 0; kv_idx < num_chunks; ++kv_idx) {
        float m_prev = m_global, d_prev = d_global;

        int lse_linear = head_idx + kv_idx * NUM_QO_HEADS_PER_KV +
                         (first_token_pos + token_idx) * LSE_TOKEN_STRIDE +
                         lse_kv_offset;
        int lse_offset = lse_linear;
        int o_offset = lse_linear * HEAD_DIM + dim_base +
                       thread_in_group * VAL_PER_THREAD + i;

        // CK FMHA stores LSE in natural log scale; convert to log2 for ptx_exp2
        float other_m = lse_g[lse_offset] * 1.44269504088896340736f,
              other_d = 1;
        m_global = max(m_prev, other_m);
        d_global = d_prev * ptx_exp2(m_prev - m_global) +
                   other_d * ptx_exp2(other_m - m_global);
        float other_o = o_g[o_offset];
        o_global = o_global * ptx_exp2(m_prev - m_global) +
                   other_o * ptx_exp2(other_m - m_global);
      }
      }
      float out_f = __fdividef(o_global, d_global);
      if (sinks_ptr != nullptr) {
        float lse_log2 = m_global + ptx_log2(d_global);
        float diff = sink_val_log2 - lse_log2;
        float correction = __fdividef(1.0f, 1.0f + ptx_exp2(diff));
        out_f *= correction;
      }
      out_vals[i] = out_f;
    }
#endif

    // Write output: either write-through (st_wt) or regular global store
    if constexpr (WRITE_THROUGH) {
      static_assert(VAL_PER_THREAD % 2 == 0 || VAL_PER_THREAD == 1,
                    "WRITE_THROUGH requires even VAL_PER_THREAD or 1");
      if constexpr (VAL_PER_THREAD == 1) {
        // One dim per thread -- DIM_SPLITS == HEAD_DIM / THREADS_PER_TOKEN,
        // the finest split the merge supports. The paired loop below cannot
        // run here: it would read out_vals[1] out of bounds and store two
        // elements, the second garbage, over the next thread's dim. That is
        // a silent corruption of half of attn_out, invisible unless
        // WRITE_THROUGH is on, which is why it survived until whole-layer
        // fusion forced the flag and DIM_SPLITS=32 started emitting nonsense.
        //
        // The obvious repair -- st_wt_u16 -- is correct but slow: a 2-byte
        // write-through is a partial-line write and the memory side turns it
        // into a read-modify-write. Measured 4.408 ms against 4.002 for the
        // (wrong) dword version, so the store width, not the merge, was
        // carrying that difference.
        //
        // So pair the threads instead. thread_in_group t and t+1 own
        // contiguous dims of the same head -- out_offset_base is
        // ... + thread_in_group * 1 -- and THREADS_PER_TOKEN divides 64, so a
        // group never straddles a wave and shfl_down(1) from an even lane
        // always lands on its own partner. Even lanes do one dword store,
        // odd lanes none.
        float const partner = __shfl_down(out_vals[0], 1, THREADS_PER_TOKEN);
        if ((thread_in_group & 1) == 0) {
          __hip_bfloat16 v0 = (__hip_bfloat16)out_vals[0];
          __hip_bfloat16 v1 = (__hip_bfloat16)partner;
          uint16_t lo, hi;
          memcpy(&lo, &v0, 2);
          memcpy(&hi, &v1, 2);
          st_wt_u32((void *)&output_ptr[out_offset_base],
                    lo | ((uint32_t)hi << 16));
        }
      } else {
        // Pack pairs of bf16 into uint32 and write-through to HBM
#pragma unroll
        for (int i = 0; i < VAL_PER_THREAD; i += 2) {
          __hip_bfloat16 v0 = (__hip_bfloat16)out_vals[i];
          __hip_bfloat16 v1 = (__hip_bfloat16)out_vals[i + 1];
          uint32_t packed;
          uint16_t lo, hi;
          memcpy(&lo, &v0, 2);
          memcpy(&hi, &v1, 2);
          packed = lo | ((uint32_t)hi << 16);
          st_wt_u32((void *)&output_ptr[out_offset_base + i], packed);
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < VAL_PER_THREAD; ++i) {
        output_ptr[out_offset_base + i] = (T)out_vals[i];
      }
    }
  }
}

} // namespace kernel