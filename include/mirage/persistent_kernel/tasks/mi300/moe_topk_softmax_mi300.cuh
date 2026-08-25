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

namespace kernel {

static constexpr int MI300_WARP_SIZE = 64; // AMD wavefront size

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
          int BYTES_PER_LDG,
          // routing_indices' ALLOCATED row stride -- the graph's BATCH_SIZE,
          // which stops matching num_rows once BATCH_SIZE > 1. 0 keeps the
          // pre-MTP behaviour. Same bug and same fix as the sigmoid_bias
          // variant; see the long comment there. GLM does not take this path,
          // so this edit is a no-op for every current caller.
          int ROUTING_ROW_STRIDE = 0>
__device__ __forceinline__ void topk_softmax_mi300_task_impl(
    void *__restrict__ input_ptr,  // [num_rows, NUM_EXPERTS]
    void *__restrict__ output_ptr, // [num_rows, k] (float weights)
    int const num_rows,
    int const k,
    void *__restrict__ routing_indices_ptr,   // [NUM_EXPERTS, num_rows] int32
    void *__restrict__ active_expert_ids_ptr, // [NUM_EXPERTS + 1] int32
    int const start_expert,
    int const end_expert,
    bool const renormalize) {
  T *input = static_cast<T *>(input_ptr);
  float *output = static_cast<float *>(output_ptr);
  int *routing_indices = static_cast<int *>(routing_indices_ptr);
  int *active_expert_ids = static_cast<int *>(active_expert_ids_ptr);
  int const rstride = ROUTING_ROW_STRIDE > 0 ? ROUTING_ROW_STRIDE : num_rows;

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  unsigned long long _tks_t0 = __builtin_amdgcn_s_memrealtime();
#endif

  // Initialize routing indices to 0.
  // active_expert_ids initialization is NOT needed: we write directly to
  // active_expert_ids[0..k-1] during TopK and set the count after.
  for (int expert = start_expert + threadIdx.x; expert < end_expert;
       expert += MPK_NT) {
    if (routing_indices != nullptr) {
      for (int row = 0; row < rstride; ++row) {
        routing_indices[expert * rstride + row] = 0;
      }
    }
  }
  __syncthreads();

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  // Split the completer's 3.0 us: everything before this point is the
  // 128-expert zero-init, everything after is the actual routing math.
  unsigned long long _tks_t1_l = __builtin_amdgcn_s_memrealtime();
  if (threadIdx.x == 0) {
    atomicAdd(&g_tk_zinit_ns, _tks_t1_l - _tks_t0);
  }
#endif

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

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
    // The logit load is 128 bf16 that every one of the 128 router tiles just
    // pushed past L2 with st_wt, so this is a cold HBM read no matter what.
    // row_chunk[0]*0 keeps the timestamp below the load's s_waitcnt.
    if (threadIdx.x == 0) {
      atomicAdd(&g_tk_lload_ns,
                __builtin_amdgcn_s_memrealtime() +
                    (unsigned long long)(row_chunk[0] * 0.0f) - _tks_t1_l);
    }
#endif

    // Reset input buffer to 0 (for split-k gate linear compatibility)
    for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) {
      int src_offset = ldg * THREADS_PER_ROW * ELTS_PER_LDG;
      for (int e = 0; e < ELTS_PER_LDG; ++e) {
        thread_read_ptr[src_offset + e] = static_cast<T>(0);
      }
    }

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
#ifdef MPK_TK_DEFER
    int topk_experts[8];
#endif

    // Precompute expert column indices for branchless local argmax.
    int col[VPT];
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      col[i] = start_col + i;
    }

    for (int k_idx = 0; k_idx < k; ++k_idx) {
      // ── Step 1: Branchless local argmax over VPT=8 elements ──
      float max_val;
      int expert;
      asm volatile("v_mov_b32 %[mv], %[r0]\n"
                   "v_mov_b32 %[ex], %[c0]\n"
                   "v_cmp_gt_f32 vcc, %[r1], %[mv]\n"
                   "v_cndmask_b32 %[mv], %[mv], %[r1], vcc\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c1], vcc\n"
                   "v_cmp_gt_f32 vcc, %[r2], %[mv]\n"
                   "v_cndmask_b32 %[mv], %[mv], %[r2], vcc\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c2], vcc\n"
                   "v_cmp_gt_f32 vcc, %[r3], %[mv]\n"
                   "v_cndmask_b32 %[mv], %[mv], %[r3], vcc\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c3], vcc\n"
                   "v_cmp_gt_f32 vcc, %[r4], %[mv]\n"
                   "v_cndmask_b32 %[mv], %[mv], %[r4], vcc\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c4], vcc\n"
                   "v_cmp_gt_f32 vcc, %[r5], %[mv]\n"
                   "v_cndmask_b32 %[mv], %[mv], %[r5], vcc\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c5], vcc\n"
                   "v_cmp_gt_f32 vcc, %[r6], %[mv]\n"
                   "v_cndmask_b32 %[mv], %[mv], %[r6], vcc\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c6], vcc\n"
                   "v_cmp_gt_f32 vcc, %[r7], %[mv]\n"
                   "v_cndmask_b32 %[mv], %[mv], %[r7], vcc\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c7], vcc\n"
                   : [mv] "=&v"(max_val), [ex] "=&v"(expert)
                   : [r0] "v"(row_chunk[0]),
                     [r1] "v"(row_chunk[1]),
                     [r2] "v"(row_chunk[2]),
                     [r3] "v"(row_chunk[3]),
                     [r4] "v"(row_chunk[4]),
                     [r5] "v"(row_chunk[5]),
                     [r6] "v"(row_chunk[6]),
                     [r7] "v"(row_chunk[7]),
                     [c0] "v"(col[0]),
                     [c1] "v"(col[1]),
                     [c2] "v"(col[2]),
                     [c3] "v"(col[3]),
                     [c4] "v"(col[4]),
                     [c5] "v"(col[5]),
                     [c6] "v"(col[6]),
                     [c7] "v"(col[7])
                   : "vcc");

      // ── Step 2: Branchless argmax reduce across subgroup ──
      // Uses __shfl_xor for cross-lane communication, inline asm for
      // branchless compare+select (eliminates s_and_saveexec divergence).
      for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        float other_max = __shfl_xor(max_val, mask, THREADS_PER_ROW);
        int other_expert = __shfl_xor(expert, mask, THREADS_PER_ROW);
        asm volatile("v_cmp_gt_f32 vcc, %[om], %[mv]\n"
                     "v_cndmask_b32 %[mv], %[mv], %[om], vcc\n"
                     "v_cndmask_b32 %[ex], %[ex], %[oe], vcc\n"
                     : [mv] "+v"(max_val), [ex] "+v"(expert)
                     : [om] "v"(other_max), [oe] "v"(other_expert)
                     : "vcc");
      }

      // ── Step 3: Write top-k result ──
#ifdef MPK_TK_DEFER
      // Record only; every store moves below the loop. See the block after it
      // for why this is the same bytes in the same places.
      if (thread_group_idx == 0) {
        topk_experts[k_idx] = expert;
        topk_vals[k_idx] = max_val;
        row_sum_for_renorm += max_val;
      }
#else
      if (thread_group_idx == 0) {
        bool const node_uses = (expert >= start_expert && expert < end_expert);
        int const out_idx = k * thread_row + k_idx;
        st_wt_u32((void *)&output[out_idx], __float_as_uint(max_val));
        topk_vals[k_idx] = max_val;
        row_sum_for_renorm += max_val;

        if (node_uses && routing_indices != nullptr) {
          int const local_expert = expert - start_expert;
          st_wt_u32(
              (void *)&routing_indices[local_expert * rstride + thread_row],
              (unsigned)(k_idx + 1));
          if (active_expert_ids != nullptr) {
            st_wt_u32((void *)&active_expert_ids[k_idx], (unsigned)expert);
          }
        }
      }
#endif

      // ── Step 4: Branchless blanking of winner ──
      // expert == col[i] matches exactly one thread + one element.
      if (k_idx + 1 < k) {
        float const neg_inf = -10000.f;
#pragma unroll
        for (int i = 0; i < VPT; ++i) {
          asm volatile("v_cmp_eq_u32 vcc, %[ex], %[ci]\n"
                       "v_cndmask_b32 %[rc], %[rc], %[ni], vcc\n"
                       : [rc] "+v"(row_chunk[i])
                       : [ex] "v"(expert), [ci] "v"(col[i]), [ni] "v"(neg_inf)
                       : "vcc");
        }
      }
    }

#ifdef MPK_TK_DEFER
    // All of Step 3, hoisted out of the k-loop.
    //
    // Why this is worth doing: the loop body alternates st_wt_u32 with
    // `asm volatile` blocks (Step 4's branchless blanking). LLVM's waitcnt
    // pass cannot see inside inline asm, so it must conservatively drain
    // outstanding VMEM before each asm block -- every iteration pays an
    // s_waitcnt vmcnt(0) on write-through stores that went past L2 to HBM.
    // With k=4 that is up to 4 serialized HBM store round trips inside what
    // should be pure register math, and this is the single completer
    // workgroup with 239 other workers parked on its release flag.
    //
    // Why it is the same result: nothing in the loop READS output,
    // routing_indices, or active_expert_ids -- the winners are already kept in
    // topk_vals/topk_experts for the renorm pass, and Step 4 blanks
    // row_chunk in registers. Store order among these three arrays is not
    // observable either: the consumers are gated on the routing_ready flag,
    // published after a threadfence_gpu in the caller.
    //
    // The `output` store in the loop was also dead outright: renormalize is
    // passed literally true at the only two call sites, so the renorm block
    // below unconditionally overwrote it.
    if (thread_group_idx == 0) {
      float inv = renormalize ? (1.f / row_sum_for_renorm) : 1.f;
      for (int k_idx = 0; k_idx < k; ++k_idx) {
        int const out_idx = k * thread_row + k_idx;
        st_wt_u32((void *)&output[out_idx],
                  __float_as_uint(topk_vals[k_idx] * inv));
        int const expert = topk_experts[k_idx];
        bool const node_uses = (expert >= start_expert && expert < end_expert);
        if (node_uses && routing_indices != nullptr) {
          int const local_expert = expert - start_expert;
          st_wt_u32(
              (void *)&routing_indices[local_expert * rstride + thread_row],
              (unsigned)(k_idx + 1));
          if (active_expert_ids != nullptr) {
            st_wt_u32((void *)&active_expert_ids[k_idx], (unsigned)expert);
          }
        }
      }
    }
#else
    // Optional renormalization (write-through stores, using cached values)
    if (renormalize && thread_group_idx == 0) {
      float inv = 1.f / row_sum_for_renorm;
      for (int k_idx = 0; k_idx < k; ++k_idx) {
        int const out_idx = k * thread_row + k_idx;
        st_wt_u32((void *)&output[out_idx],
                  __float_as_uint(topk_vals[k_idx] * inv));
      }
    }
#endif
  }
  __syncthreads();

  // Set active expert count (thread 0 only, single wavefront handles all rows
  // for batch=1)
  if (active_expert_ids != nullptr && threadIdx.x == 0) {
    st_wt_u32((void *)&active_expert_ids[NUM_EXPERTS], (unsigned)k);
  }
}

} // namespace kernel
