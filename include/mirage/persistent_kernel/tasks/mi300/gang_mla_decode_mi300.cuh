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

namespace gang_mla_decode_detail {
// POD vectors, used instead of uint2/uint4 for anything loaded through an
// addrspace(1) pointer: the HIP_vector_type classes have a copy constructor
// taking a generic `uint2 const&`, which undoes the cast before the load ever
// happens. ext_vector_type is loaded directly and keeps .x/.y/.z/.w.
typedef unsigned int __attribute__((ext_vector_type(2))) u32x2_t;
typedef unsigned int __attribute__((ext_vector_type(4))) u32x4_t;

// A load that is *known* to come from device memory.
//
// Clang infers address spaces intraprocedurally, so a pointer that arrives as
// an argument to this __noinline__ kernel stays generic and every dereference
// is emitted as flat_load rather than global_load. On gfx9 a flat instruction
// increments **both** vmcnt and lgkmcnt, so the `s_waitcnt lgkmcnt(0)` that
// retires an LDS read also waits on every outstanding KV load -- which defeats
// the whole point of the tile prefetch below. Same fix and same reasoning as
// gang_gemv_mxfp8_detail::ld_g; the cast is two-step because clang rejects a
// reinterpret_cast that changes pointee type and address space at once.
//
// Safe because every pointer handed to this kernel is device global: the q
// workspace, the paged latent cache, the index arrays and the outputs all come
// from the megakernel's workspace or from a task descriptor.
template <typename T>
__device__ __forceinline__ T ld_g(void const *p) {
  T const *q = static_cast<T const *>(p);
  return *(__attribute__((address_space(1))) T const *)q;
}

// Native bf16 MFMA. gfx950 runs bf16 at the same rate as fp16, so the latent
// cache -- which is bf16 in memory -- has no reason to be widened to f32 and
// narrowed to fp16 on the way to the matrix core. Dropping that round trip
// removes ~340 VALU ops per invocation, but far more importantly it removes a
// *dependency*: with fp16 the loaded dwords had to be converted before they
// could be used, so the Q load and the KV prologue each stalled on their own
// data. Loaded straight as bf16 they are already the MFMA operand registers.
//
// It is also strictly better numerically on the K side: bf16 -> fp16 narrows
// the exponent from 8 bits to 5 and can overflow to inf on a large latent
// value. The P operand loses mantissa (8 bits vs 11), which is the usual
// bf16 flash-attention tradeoff and is absorbed by the fp32 accumulator.
typedef __bf16 __attribute__((ext_vector_type(8))) bf16x8_t;
typedef __bf16 __attribute__((ext_vector_type(4))) bf16x4_t;

__device__ __forceinline__ __mfma_hd64_fp32x4
    mfma_qk_bf16(__mfma_hd64_fp32x4 c, __bf16 const *a, __bf16 const *b) {
  bf16x8_t av, bv;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    av[i] = a[i];
    bv[i] = b[i];
  }
  return __builtin_amdgcn_mfma_f32_16x16x32_bf16(av, bv, c, 0, 0, 0);
}

__device__ __forceinline__ __mfma_hd64_fp32x4
    mfma_pv_bf16(__mfma_hd64_fp32x4 c, __bf16 const *a, __bf16 const *b) {
  bf16x4_t av, bv;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    av[i] = a[i];
    bv[i] = b[i];
  }
  return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, bv, c, 0, 0, 0);
}

// The bf16x4 loads from paged_attention_decode_minimal_hd64, re-expressed so
// the load itself goes through ld_g. The convert half is reused verbatim.
__device__ __forceinline__ void __ldg_bf16x4_raw(uint2 *__restrict__ dst,
                                                 void const *__restrict__ src) {
  u32x2_t const v = ld_g<u32x2_t>(src);
  dst->x = v.x;
  dst->y = v.y;
}
} // namespace gang_mla_decode_detail

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
// THIS function sets the whole megakernel's register allocation, and two
// obvious ways to fix that are closed. Recorded so neither gets retried.
//
// Under -fgpu-rdc every phase is its own __noinline__ ELF symbol (125
// s_swappc_b64 in the image), so a kernel's .vgpr_count is a call-graph
// quantity, not its own body's pressure (worker_kernel's body is 226; it
// reports 284). At the default budget the 284 is 248 arch + 36 acc and BOTH
// terms are set here -- 248 arch ties with three other functions, and the 36
// AGPRs are this function's alone.
//
// CLOSED 1: a budget on this function. __attribute__((amdgpu_waves_per_eu))
// does not compile on a device function -- clang restricts it to kernels
// ("'amdgpu_waves_per_eu' attribute only applies to kernel functions") and
// LLVM 20.0.0git as shipped in ROCm 7.0 has no llc equivalent and no
// "amdgpu-agpr-alloc". A non-kernel function with no waves-per-eu attribute
// is allocated against the full 512-entry unified file, which is what this
// one does.
//
// CLOSED 2: inlining it into a kernel that CAN be budgeted. Dropping
// __noinline__ here and on gang_mla_full_layer_fused_kernel_mi300 does not
// inline anything -- at -O2 the inliner declines a ~950-instruction callee
// and both symbols survive with .vgpr_count unmoved at 284.
//
// What actually worked is MPK_WORKER_WAVES_PER_EU in persistent_kernel.cuh:
// budget both KERNELS and let LLVM's AMDGPUAttributor push the constraint
// down the call graph. Long note there.
//
// Do not read this function's 240 scratch ops as pressure -- they are the ABI
// callee-save prologue/epilogue (v40-v143, 60 dwords in at instr 1-60, back
// out at 880-939). The body spills nothing at 284. Its live set is genuine:
// arch peaks at 248 in the same window where 24-36 AGPRs are live, so moving
// the MFMA accumulators to arch VGPRs trades 36 acc for 36 arch and loses.
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
  using gang_mla_decode_detail::__ldg_bf16x4_raw;
  using gang_mla_decode_detail::bf16x4_t;
  using gang_mla_decode_detail::ld_g;
  using gang_mla_decode_detail::mfma_pv_bf16;
  using gang_mla_decode_detail::mfma_qk_bf16;
  using gang_mla_decode_detail::u32x4_t;

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
  int const query_start = ld_g<int>(&qo_indptr[req]);
  if (query_start == ld_g<int>(&qo_indptr[req + 1])) {
    return;
  }

  int const first_page = ld_g<int>(&kv_indptr[req]);
  int const num_pages = ld_g<int>(&kv_indptr[req + 1]) - first_page;
  int const seqlen_k =
      (num_pages - 1) * PAGE_SIZE + ld_g<int>(&kv_last_page_len[req]);

  int const tid = threadIdx.x;
  int const warp_id = tid / 64;
  int const lane = tid & 63;
  int const midx = lane & 15;
  int const kgrp = lane >> 4;

  char const *kv_base = reinterpret_cast<char const *>(paged_kv_cache_ptr);

  // LDS: KV[KV_TILE][QK_DIM] bf16 (18 KB for GLM-5), the cache's own format.
  // K and V are the same rows, so unlike the GQA decode there is only one
  // buffer.
  extern __shared__ char _mla_decode_smem[];
  __bf16 *lds_kv = reinterpret_cast<__bf16 *>(_mla_decode_smem);

  // 256 threads x 4 elements = 16 tok x 64 dim per round, LDG_PER_TILE rounds
  int const my_tok = tid / 16;
  int const my_dim0 = (tid % 16) * 4;

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  // Slot 2 is free on the MXFP8 fused path (its only other user is
  // gang_linear_mxfp4_res_bias, which that path never calls). Phases:
  //   2 = chunk partition      3 = cold start: Q + KV tiles 0/1 + LDS write
  //   4 = per-tile QK/softmax/PV (incl. the entry __syncthreads)
  //   5 = per-tile refill: __syncthreads + vmcnt drain + LDS write + prefetch
  //   6 = epilogue      (the count goes to g_subphase_cnt[2])
  unsigned long long _d_t0 = __builtin_amdgcn_s_memrealtime();
  unsigned long long _d_compute = 0, _d_refill = 0;
#endif

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
  //
  // The page-table read is a *dependent* load, and it used to sit in front of
  // every tile's prefetch: the asm showed `flat_load_dword` (kv_indices) then
  // `s_waitcnt vmcnt(0) lgkmcnt(0)` then the nine KV loads, once per tile. So
  // each tile paid a full round trip before its first KV byte could even be
  // addressed, and the drain retired the *previous* tile's prefetch at the
  // same time -- the software pipeline never actually overlapped anything.
  //
  // A worker's whole chunk is at most a few tiles of PAGE_SIZE-token pages, so
  // the page index is very nearly loop-invariant. Cache the last (page, base)
  // pair and reload only when it actually changes; at the GLM shape that is
  // once, in the prologue.
  int cached_page = -1;
  long cached_page_base = 0;
  auto get_kv_row = [&](int global_tok) -> long {
    int const page = global_tok / PAGE_SIZE;
    if (page != cached_page) {
      int const pid = ld_g<int>(&kv_indices[first_page + page]);
      cached_page = page;
      cached_page_base =
          static_cast<long>(pid) * PAGE_SIZE * KV_CACHE_STRIDE * 2;
    }
    return cached_page_base +
           static_cast<long>(global_tok % PAGE_SIZE) * KV_CACHE_STRIDE * 2;
  };

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  unsigned long long _d_t1 = __builtin_amdgcn_s_memrealtime();
#endif

  // ===== PROLOGUE: page table first, then KV[0], KV[1] and Q all in flight =====
  //
  // Order here is load-bearing, and the obvious order is the wrong one.
  //
  // The page-table read is the only *dependent* load in the cold start: no KV
  // address can be formed until it lands. With it issued after the Q block,
  // the only `s_waitcnt` that retires it is `vmcnt(0)` -- which also retires
  // all 18 Q loads. The asm showed exactly that: 18 `global_load_dwordx4`,
  // `s_waitcnt vmcnt(0)`, page load, `s_waitcnt vmcnt(0)`, then tile 0. Three
  // serialized round trips, which is what the profile charged 5.25 us of a
  // 10.5 us decode for. Putting Q first only reordered them.
  //
  // So: resolve both page rows first, when nothing else is outstanding and the
  // wait costs exactly one round trip. Everything after that point is
  // independent and can sit in flight together. Then issue tile 0, tile 1 and
  // Q *in that order*, so the LDS write below -- the first real consumer --
  // waits on `vmcnt(27)` (tile 1 + Q still outstanding) rather than draining
  // the world. With one wave per SIMD there is no occupancy to hide any of
  // this, so overlap has to be built by hand.
  int const tile0_len = (KV_TILE < effective_len) ? KV_TILE : effective_len;
  int const tile1_len = ((effective_len - KV_TILE) < KV_TILE)
                            ? (effective_len - KV_TILE)
                            : KV_TILE;
  bool const do_t0 = (my_tok < tile0_len);
  bool const do_t1 = (ntiles > 1) && (my_tok < tile1_len);
  long row0 = 0, row1 = 0;
  if (do_t0) {
    row0 = get_kv_row(kv_start + my_tok);
  }
  if (do_t1) {
    row1 = get_kv_row(kv_start + KV_TILE + my_tok);
  }

  uint2 kv0[LDG_PER_TILE];
  if (do_t0) {
#pragma unroll
    for (int r = 0; r < LDG_PER_TILE; r++) {
      __ldg_bf16x4_raw(&kv0[r], kv_base + row0 + (my_dim0 + r * 64) * 2);
    }
  }

  // Prefetch KV[1] -> registers as raw dwords
  uint2 kv_pre[LDG_PER_TILE];
  bool has_pre = false;
  if (do_t1) {
#pragma unroll
    for (int r = 0; r < LDG_PER_TILE; r++) {
      __ldg_bf16x4_raw(&kv_pre[r], kv_base + row1 + (my_dim0 + r * 64) * 2);
    }
    has_pre = true;
  }

  // Q slice for this lane: head (q_head_group * 16 + midx), dims
  // [kc * 32 + kgrp * 8, +8) for each MFMA step.
  //
  // Issued last of the three, and consumed last -- the first MFMA is past the
  // loop's opening `__syncthreads()`. At bf16 these land directly in the MFMA
  // operand registers, so no convert sits between the load and its use and
  // nothing forces an early drain. All four waves read the same 16 heads, so
  // waves 1..3 hit L1.
  __bf16 qr[NUM_K32][8];
  {
    int const q_head = q_head_group * Q_HEADS_PER_GROUP + midx;
    char const *q_ptr = reinterpret_cast<char const *>(q_workspace_ptr) +
                        (static_cast<long>(query_start) * Q_WORKSPACE_STRIDE +
                         static_cast<long>(q_head) * QK_DIM) *
                            2;
#pragma unroll
    for (int kc = 0; kc < NUM_K32; kc++) {
      int dim_off = kc * 32 + kgrp * 8;
      u32x4_t const raw = ld_g<u32x4_t>(q_ptr + dim_off * 2);
      __builtin_memcpy(&qr[kc][0], &raw, 16);
    }
  }

  // Now consume tile 0. The cache is bf16 and so is the LDS tile, so this is
  // a straight 8-byte copy -- no convert between the load and the store.
  {
    uint2 const zero2 = {0u, 0u};
#pragma unroll
    for (int r = 0; r < LDG_PER_TILE; r++) {
      uint2 const v = do_t0 ? kv0[r] : zero2;
      *(uint64_t *)&lds_kv[my_tok * QK_DIM + my_dim0 + r * 64] =
          *(uint64_t *)&v;
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

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  unsigned long long _d_t2 = __builtin_amdgcn_s_memrealtime();
#endif

  for (int t = 0; t < ntiles; t++) {
#ifdef MPK_ENABLE_SUBPHASE_TIMING
    unsigned long long _d_a = __builtin_amdgcn_s_memrealtime();
#endif
    int tile_start = t * KV_TILE;
    int tile_len = ((effective_len - tile_start) < KV_TILE)
                       ? (effective_len - tile_start)
                       : KV_TILE;

    __syncthreads();

    // QK: lane (midx, kgrp) reads K[tok=midx][dim=kc*32+kgrp*8..+7]
    __mfma_hd64_fp32x4 scores = {0, 0, 0, 0};
#pragma unroll
    for (int kc = 0; kc < NUM_K32; kc++) {
      __bf16 kr[8];
      __bf16 const *k_ptr = &lds_kv[midx * QK_DIM + kc * 32 + kgrp * 8];
#pragma unroll
      for (int i = 0; i < 8; i++) {
        kr[i] = k_ptr[i];
      }
      scores = mfma_qk_bf16(scores, kr, qr[kc]);
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

    bf16x4_t pb;
    pb[0] = (__bf16)w0;
    pb[1] = (__bf16)w1;
    pb[2] = (__bf16)w2;
    pb[3] = (__bf16)w3;

    // PV: A = V transposed, lane (midx, kgrp) reads
    // V[tok=kgrp*4+j][dim=vb*64+warp_id*16+midx]. V is the latent prefix of
    // the same LDS rows the QK step just read.
#pragma unroll
    for (int vb = 0; vb < NUM_V_BLOCKS; vb++) {
      bf16x4_t va;
      __bf16 const *v_ptr =
          &lds_kv[(kgrp * 4) * QK_DIM + vb * 64 + warp_id * 16 + midx];
      va[0] = v_ptr[0 * QK_DIM];
      va[1] = v_ptr[1 * QK_DIM];
      va[2] = v_ptr[2 * QK_DIM];
      va[3] = v_ptr[3 * QK_DIM];
      o_acc[vb][0] *= rescale;
      o_acc[vb][1] *= rescale;
      o_acc[vb][2] *= rescale;
      o_acc[vb][3] *= rescale;
      o_acc[vb] = mfma_pv_bf16(
          o_acc[vb], (__bf16 const *)&va, (__bf16 const *)&pb);
    }

    // The QK and PV steps both read the whole LDS tile, so the refill has to
    // wait for every lane rather than slotting between them the way the HD=64
    // decode does.
#ifdef MPK_ENABLE_SUBPHASE_TIMING
    unsigned long long _d_b = __builtin_amdgcn_s_memrealtime();
    _d_compute += _d_b - _d_a;
#endif
    __syncthreads();

    if (has_pre) {
#pragma unroll
      for (int r = 0; r < LDG_PER_TILE; r++) {
        *(uint64_t *)&lds_kv[my_tok * QK_DIM + my_dim0 + r * 64] =
            *(uint64_t *)&kv_pre[r];
      }
    } else if (t + 1 < ntiles) {
      // This lane's token is past the end of tile t+1: zero it so the stale
      // rows cannot leak into the next tile's QK.
      uint64_t const zero8 = 0;
#pragma unroll
      for (int r = 0; r < LDG_PER_TILE; r++) {
        *(uint64_t *)&lds_kv[my_tok * QK_DIM + my_dim0 + r * 64] = zero8;
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
          __ldg_bf16x4_raw(&kv_pre[r], kv_base + row + (my_dim0 + r * 64) * 2);
        }
        has_pre = true;
      }
    }
#ifdef MPK_ENABLE_SUBPHASE_TIMING
    _d_refill += __builtin_amdgcn_s_memrealtime() - _d_b;
#endif
  }

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  unsigned long long _d_t3 = __builtin_amdgcn_s_memrealtime();
#endif

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

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  if (tid == 0 && g_subphase_active) {
    unsigned long long _d_t4 = __builtin_amdgcn_s_memrealtime();
    atomicAdd(&g_subphase_ns[2][2], (_d_t1 - _d_t0) * 10);
    atomicAdd(&g_subphase_ns[2][3], (_d_t2 - _d_t1) * 10);
    atomicAdd(&g_subphase_ns[2][4], _d_compute * 10);
    atomicAdd(&g_subphase_ns[2][5], _d_refill * 10);
    atomicAdd(&g_subphase_ns[2][6], (_d_t4 - _d_t3) * 10);
    // Must be g_subphase_cnt, not another g_subphase_ns phase: the printer
    // skips any slot whose cnt is still zero.
    atomicAdd(&g_subphase_cnt[2], 1ULL);
  }
#endif
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
