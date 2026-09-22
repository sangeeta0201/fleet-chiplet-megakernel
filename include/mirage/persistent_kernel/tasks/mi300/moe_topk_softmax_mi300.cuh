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

// MoE TopK Softmax routing kernel for MI300/MI350.
// Fused softmax + top-k selection + routing index generation.
// Adapted from topk_softmax_sm100.cuh for HIP/AMD.

#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include "arch_traits.cuh" // WAVE_SIZE, lane_mask_t, ballot
#include "mpk_atoms.cuh"   // st_wt_u16, st_wt_u32

namespace kernel {

// AMD wavefront size. Not 64 unconditionally any more: gfx1250 (MI450) runs
// wave32, and every use below -- the row-to-lane mapping, the ballot mask
// width, and the expert compaction stride -- has to follow the real width or
// the kernel silently processes the wrong rows. mirage::arch::WAVE_SIZE is a
// constant expression (unlike HIP's warpSize), so it can size the mapping math.
static constexpr int MI300_WARP_SIZE = mirage::arch::WAVE_SIZE;

// Fused TopK softmax kernel for AMD MI300/MI350.
// Uses wave-level reductions with __shfl_xor for max/sum across experts.
//
// Block size: 256 threads (4 wavefronts of 64).
// Each wavefront processes multiple rows in parallel.
//
// Template parameters:
//   T           - data type (bf16)
//   VPT         - values per thread (power of 2)
//   NUM_EXPERTS - total number of experts (power of 2)
//   WARPS_PER_CTA - number of wavefronts per CTA (typically 4)
//   BYTES_PER_LDG - bytes per vectorized load (8 or 16)
template <typename T,
          int VPT,
          int NUM_EXPERTS,
          int WARPS_PER_CTA,
          int BYTES_PER_LDG>
__device__ __forceinline__ void topk_softmax_mi300_task_impl(
    void *__restrict__ input_ptr,  // [num_rows, NUM_EXPERTS]
    void *__restrict__ output_ptr, // [num_rows, k] (float weights)
    int const num_rows,
    // Allocated row stride of routing_indices, which is BATCH_SIZE -- not
    // num_rows. The two differ whenever fewer tokens are active than the batch
    // was sized for (the fused router passes num_active_tokens as num_rows).
    // The MoE tile decode indexes this buffer as `d_routing + expert*BATCH_SIZE`
    // and reads lanes [0, BATCH_SIZE), so writing or clearing at a narrower
    // stride puts every expert's row at the wrong offset and leaves the tail
    // holding a previous layer's routing values.
    int const routing_row_stride,
    int const k,
    void *__restrict__ routing_indices_ptr,   // [NUM_EXPERTS, stride] int32
    void *__restrict__ active_expert_ids_ptr, // [NUM_EXPERTS + 1] int32
    int const start_expert,
    int const end_expert,
    bool const renormalize) {
  T *input = static_cast<T *>(input_ptr);
  float *output = static_cast<float *>(output_ptr);
  int *routing_indices = static_cast<int *>(routing_indices_ptr);
  int *active_expert_ids = static_cast<int *>(active_expert_ids_ptr);

  // Union of experts activated by *any* row, marked during TopK and compacted
  // at the bottom. One byte per local expert; the write is idempotent (every
  // writer stores 1), so no atomics are needed.
  __shared__ unsigned char s_hit[NUM_EXPERTS];

  // Initialize routing indices to 0.
  // active_expert_ids initialization is NOT needed: the compaction pass at the
  // bottom writes [0..count-1] densely and then the count at [NUM_EXPERTS], and
  // no consumer reads past the count.
  for (int expert = start_expert + threadIdx.x; expert < end_expert;
       expert += blockDim.x) {
    if (routing_indices != nullptr) {
      // Clear the whole allocated row, not just the active prefix: the MoE
      // decode ballots over all BATCH_SIZE lanes, so a stale nonzero in
      // [num_rows, stride) would compact a token that does not exist into an
      // MFMA column.
      //
      // Write-through for the same reason as the logits reset below: the
      // winners are published with `st_wt_u32` straight to HBM, so a plain
      // store leaves a dirty zero line in this XCD's L2 whose writeback can
      // land after a later layer has written the same address. Here the
      // consequence is worse than a perturbed weight -- a dropped routing
      // index silently removes a token from an expert's MFMA column set.
      for (int row = 0; row < routing_row_stride; ++row) {
        st_wt_u32((void *)&routing_indices[expert * routing_row_stride + row],
                  0u);
      }
    }
  }
  for (int e = threadIdx.x; e < NUM_EXPERTS; e += blockDim.x) {
    s_hit[e] = 0;
  }
  __syncthreads();

  // Compile-time constants
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
  static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
  static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
  static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

  static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
  static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS),
                "NUM_EXPERTS must be power of 2");
  static_assert(MI300_WARP_SIZE % THREADS_PER_ROW == 0,
                "THREADS_PER_ROW must divide warp size");

  static constexpr int ELTS_PER_WARP = MI300_WARP_SIZE * VPT;
  static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;

  int const warp_idx = threadIdx.x / MI300_WARP_SIZE;
  int const lane_idx = threadIdx.x % MI300_WARP_SIZE;
  int const warp_base_row = warp_idx * ROWS_PER_WARP;
  int const thread_row_in_warp = lane_idx / THREADS_PER_ROW;
  int const thread_row = warp_base_row + thread_row_in_warp;
  int const thread_group_idx = lane_idx % THREADS_PER_ROW;

  if (thread_row < num_rows) {
    // Load row data
    T *thread_row_ptr = input + thread_row * ELTS_PER_ROW;
    int const first_elt = thread_group_idx * ELTS_PER_LDG;
    T *thread_read_ptr = thread_row_ptr + first_elt;

    float row_chunk[VPT];
    // Vectorized load
    for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) {
      int base = ldg * ELTS_PER_LDG;
      int src_offset = ldg * THREADS_PER_ROW * ELTS_PER_LDG;
      for (int e = 0; e < ELTS_PER_LDG; ++e) {
        row_chunk[base + e] =
            static_cast<float>(thread_read_ptr[src_offset + e]);
      }
    }

    // Reset input buffer to 0 (for split-k gate linear compatibility).
    //
    // WRITE-THROUGH, to match the producer. The 128 router workers publish
    // each logit with `st_wt_u16` (sc0 sc1), which bypasses L2 and lands in
    // HBM, and they sit on all 8 XCDs while this reader is on one. A plain
    // store here leaves a *dirty* copy of the line in this XCD's L2. That line
    // is not merely stale-on-read -- the acquire in topk_noinline drops it
    // before the next read -- it is a pending writeback of zeros. Whenever it
    // is evicted after the next layer's router has already written that
    // address in HBM, the eviction overwrites a live logit with 0.
    //
    // A zeroed logit does not crash and does not change the ranking: 0 is far
    // below the winners, so TopK returns the same experts in the same order
    // and `routing_indices` / `active_expert_ids` come out bit-identical. But
    // the softmax denominator is a sum over *all* NUM_EXPERTS logits, so the
    // renormalized weights all shift in the low bits.
    //
    // Do NOT read the paragraph above as an explanation of any observed
    // moe_topk_weight nondeterminism. It was, and it was the wrong suspect: the
    // real cause of run-to-run moe_topk_weight divergence at batch_size >= 8
    // was a dim-0 `input_map` on the output tensor, which made the host offset
    // the base pointer by whole rows. See the NOTE at the `topk_weight`
    // new_input() sites in persistent_kernel.py. The write-through reset here
    // is still correct and still required for the L2 reason given above; it
    // just shifts low bits, not whole rows. Define MPK_NO_LOGIT_RESET to keep
    // the logits readable when dumping tensors for a golden reference.
#ifndef MPK_NO_LOGIT_RESET
    for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) {
      int src_offset = ldg * THREADS_PER_ROW * ELTS_PER_LDG;
      for (int e = 0; e < ELTS_PER_LDG; ++e) {
        T zero = static_cast<T>(0);
        st_wt_u16(&thread_read_ptr[src_offset + e],
                  *reinterpret_cast<unsigned short *>(&zero));
      }
    }
#endif

    // Max reduction within subgroup
    float thread_max = row_chunk[0];
    for (int ii = 1; ii < VPT; ++ii) {
      thread_max = fmaxf(thread_max, row_chunk[ii]);
    }
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      float other = __shfl_xor(thread_max, mask, THREADS_PER_ROW);
      thread_max = fmaxf(thread_max, other);
    }

    // Softmax
    float row_sum = 0.f;
    for (int ii = 0; ii < VPT; ++ii) {
      row_chunk[ii] = __builtin_amdgcn_exp2f((row_chunk[ii] - thread_max) *
                                             1.4426950408889634f);
      row_sum += row_chunk[ii];
    }
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      row_sum += __shfl_xor(row_sum, mask, THREADS_PER_ROW);
    }
    float const inv_sum = 1.f / row_sum;
    for (int ii = 0; ii < VPT; ++ii) {
      row_chunk[ii] *= inv_sum;
    }

    // Fused Top-K selection — Step 1 uses inline asm for branchless local
    // argmax.
    int const start_col = first_elt;
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;
    float row_sum_for_renorm = 0.f;
    float topk_vals[8];

    // Precompute expert column indices for branchless local argmax.
    int col[VPT];
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      col[i] = start_col + i;
    }

    for (int k_idx = 0; k_idx < k; ++k_idx) {
      // ── Step 1: Branchless local argmax over VPT=8 elements ──
      // Portable form of what used to be a hand-written v_cmp/v_cndmask
      // chain. The asm existed to guarantee branchless select rather than an
      // s_and_saveexec diverge-and-rejoin; a ternary on a scalar float lowers
      // to exactly v_cmp_gt_f32 + v_cndmask_b32 on both gfx950 and gfx1250, so
      // the codegen is unchanged and the wave-width assumption baked into the
      // literal `vcc` operand is gone. (`vcc` is 64-bit on gfx9 and 32-bit on
      // gfx12; naming it explicitly is exactly the kind of thing that would
      // have silently mismatched at wave32.)
      //
      // Ties: strict `>` keeps the lowest index, matching the original.
      float max_val = row_chunk[0];
      int expert = col[0];
#pragma unroll
      for (int i = 1; i < VPT; ++i) {
        bool const gt = row_chunk[i] > max_val;
        expert = gt ? col[i] : expert;
        max_val = gt ? row_chunk[i] : max_val;
      }

      // ── Step 2: Branchless argmax reduce across subgroup ──
      // Butterfly over THREADS_PER_ROW lanes. The compare+select was hand
      // written asm to keep it branchless rather than an s_and_saveexec
      // diverge-and-rejoin; two ternaries on the same predicate lower to the
      // identical v_cmp_gt_f32 + two v_cndmask_b32 on gfx950 and gfx1250, so
      // the codegen is unchanged and the explicit `vcc` operand -- 64-bit on
      // gfx9, 32-bit on gfx12 -- is gone.
      //
      // Ties: `>` keeps the *current* lane's expert, so the winner is the
      // lowest-numbered lane holding the max, exactly as before.
      for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        float const other_max = __shfl_xor(max_val, mask, THREADS_PER_ROW);
        int const other_expert = __shfl_xor(expert, mask, THREADS_PER_ROW);
        bool const gt = other_max > max_val;
        expert = gt ? other_expert : expert;
        max_val = gt ? other_max : max_val;
      }

      // ── Step 3: Write top-k result ──
      if (thread_group_idx == 0) {
        bool const node_uses = (expert >= start_expert && expert < end_expert);
        int const out_idx = k * thread_row + k_idx;
        st_wt_u32((void *)&output[out_idx], __float_as_uint(max_val));
        topk_vals[k_idx] = max_val;
        row_sum_for_renorm += max_val;

        if (node_uses && routing_indices != nullptr) {
          int const local_expert = expert - start_expert;
          st_wt_u32((void *)&routing_indices[local_expert * routing_row_stride +
                                             thread_row],
                    (unsigned)(k_idx + 1));
          // Mark, do not write: at num_rows > 1 several rows reach this line
          // with different `expert` for the same k_idx, so indexing the output
          // by k_idx made rows overwrite each other. Marking a per-expert byte
          // is idempotent and deduplicates for free -- which the MoE tile
          // decode requires, since it maps tile -> active_expert_ids[i] and
          // would otherwise run the same expert twice and drop another.
          s_hit[local_expert] = 1;
        }
      }

      // ── Step 4: Branchless blanking of winner ──
      // expert == col[i] matches exactly one thread + one element.
      if (k_idx + 1 < k) {
        float const neg_inf = -10000.f;
        // Was v_cmp_eq_u32 + v_cndmask_b32 by hand, for the same
        // branchless-select reason as Steps 1 and 2. The ternary lowers to the
        // same pair on both targets without naming the wave-width-dependent
        // `vcc` register.
#pragma unroll
        for (int i = 0; i < VPT; ++i) {
          row_chunk[i] = (expert == col[i]) ? neg_inf : row_chunk[i];
        }
      }
    }

    // Optional renormalization (write-through stores, using cached values)
    if (renormalize && thread_group_idx == 0) {
      float inv = 1.f / row_sum_for_renorm;
      for (int k_idx = 0; k_idx < k; ++k_idx) {
        int const out_idx = k * thread_row + k_idx;
        st_wt_u32((void *)&output[out_idx],
                  __float_as_uint(topk_vals[k_idx] * inv));
      }
    }
  }
  __syncthreads();

  // ── Compact the hit set into a dense, ascending, deduplicated list ──
  // Wavefront 0 alone, so the running `count` needs no atomics: it is a plain
  // register carried across the chunk loop, uniform across the wave because
  // every lane derives it from the same ballot masks.
  //
  // At num_rows == 1 this reproduces the old behaviour up to ordering: exactly
  // k distinct experts win, so count == k, and the ids come out sorted instead
  // of in topk order. No consumer depends on that order -- they either scan
  // [0, count) or index by tile.
  if (active_expert_ids != nullptr && threadIdx.x < MI300_WARP_SIZE) {
    int const lane = threadIdx.x;
    // Width-correct prefix mask. `1ULL << lane` was safe only because lane was
    // known to be < 64; at wave32 the mask type shrinks with the wave, and
    // lane_mask_t/FULL_WAVE_MASK keep the two in step. Note __ballot returns
    // 64 bits on both targets (the upper half is simply zero at wave32), so
    // the AND with lane_mask is what actually bounds the popcount.
    using mask_t = mirage::arch::lane_mask_t;
    mask_t const lane_mask =
        (mask_t)((lane == 0) ? 0 : (mirage::arch::FULL_WAVE_MASK >>
                                    (mirage::arch::WAVE_SIZE - lane)));
    int count = 0;
    for (int base = 0; base < NUM_EXPERTS; base += MI300_WARP_SIZE) {
      int const e = base + lane;
      bool const hit = (e < NUM_EXPERTS) && (s_hit[e] != 0);
      unsigned long long const ballot = __ballot(hit);
      if (hit) {
        int const pos =
            count + __popcll(ballot & (unsigned long long)lane_mask);
        st_wt_u32((void *)&active_expert_ids[pos], (unsigned)(start_expert + e));
      }
      count += __popcll(ballot);
    }
    // Sentinel-fill the unused slots. The MoE tile decode must decide "is this
    // tile padding?" from a value that is *per-slot*, never from the count:
    // the count is one word shared by all 36 layers, so two workers that read
    // it at different moments can disagree about how many slots are live, and
    // then disagree about which tiles are W13. That disagreement is a deadlock,
    // not drift -- the skipped tiles never reach the per-expert arrival, the
    // `% W13_TILES` release never fires, and the W2 workers spin forever.
    //
    // A slot read is not immune to the same cross-layer skew, but it degrades
    // the way the mask already did before packing: a worker gets some layer's
    // expert id and produces slightly wrong numbers. Every worker that looks at
    // slot s reads the same address, so they cannot disagree about whether s is
    // padding while agreeing on everything else.
    for (int e = count + lane; e < NUM_EXPERTS; e += MI300_WARP_SIZE) {
      st_wt_u32((void *)&active_expert_ids[e], (unsigned)-1);
    }
    if (lane == 0) {
      st_wt_u32((void *)&active_expert_ids[NUM_EXPERTS], (unsigned)count);
    }
  }
}

} // namespace kernel
