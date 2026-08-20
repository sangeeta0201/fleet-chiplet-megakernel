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

// MoE TopK sigmoid + correction-bias routing kernel for MI300/MI350.
// This is the `noaux_tc` router used by GLM-5 / GLM-4.x-MoE (and DeepSeek-V3):
//
//   scores            = sigmoid(gate_logits)                      [fp32]
//   scores_for_choice = scores + e_score_correction_bias          [fp32]
//   topk_ids          = topk(scores_for_choice, k)
//   topk_weights      = scores.gather(topk_ids)   <- UNBIASED scores
//   if norm_topk_prob: topk_weights /= sum(topk_weights)
//   topk_weights     *= routed_scaling_factor
//
// GLM-5 sets n_group = topk_group = 1, so the DeepSeek group-limited routing
// degenerates to a plain top-k over all experts and no group masking is needed.
//
// GLM's always-on shared expert is emitted as one extra routing slot rather
// than as a separate pair of GEMMs (the AITER `shared_expert_id` trick): with
// `num_shared_experts == 1` the router appends expert id NUM_EXPERTS at slot k
// with weight 1.0, so the routed-MoE tasks run it alongside the selected
// experts and `moe_mul_sum_add` folds it into the same weighted sum. The
// shared expert has exactly a routed expert's shape, so it is just row
// NUM_EXPERTS of the stacked weights.
//
// Structurally a clone of moe_topk_softmax_mi300.cuh (same wave layout, same
// branchless argmax / blanking inline asm); only the scoring function and the
// bias/scale plumbing differ. The one extra wrinkle is that selection uses the
// biased score while the emitted weight is the unbiased one, so the argmax
// carries the unbiased score as a third payload register alongside
// (max, expert) instead of re-reading it after the cross-lane reduction.

#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

// Reuse MI300_WARP_SIZE from the softmax router (same wave layout).
#include "moe_topk_softmax_mi300.cuh"

namespace kernel {

// Hardware-accelerated sigmoid: rcp(1 + exp(-x)).
// Same V_RCP_F32 trick as fast_silu() in silu_mul_mi300.cuh (~4 cycles vs ~16
// for fdiv).
__device__ __forceinline__ float fast_sigmoid(float x) {
  return __builtin_amdgcn_rcpf(1.0f + __expf(-x));
}

// Fused sigmoid + bias TopK kernel for AMD MI300/MI350.
//
// Block size: 256 threads (4 wavefronts of 64).
// Each wavefront processes ROWS_PER_WARP rows in parallel.
//
// Template parameters:
//   T           - data type (bf16)
//   VPT         - values per thread (power of 2)
//   NUM_EXPERTS - total number of experts (power of 2)
//   WARPS_PER_CTA - number of wavefronts per CTA (typically 4)
//   BYTES_PER_LDG - bytes per vectorized load (8 or 16)
//   K_STATIC      - top-k when the caller knows it at compile time, else 0.
//
// K_STATIC is a performance parameter, not a functional one: `k` is still
// honoured either way. It exists because `topk_vals[k_idx]` is indexed by the
// selection loop's induction variable, and with a runtime bound the loop can
// not unroll, so the index stays dynamic and AMDGCN backs the array with
// scratch -- private memory, which is HBM. That turns eight register writes
// plus the renormalisation loop's eight reads into sixteen ~400-cycle round
// trips sitting on the router's serial critical path. Passing the bound as a
// constant unrolls both loops and keeps the array in VGPRs.
template <typename T,
          int VPT,
          int NUM_EXPERTS,
          int WARPS_PER_CTA,
          int BYTES_PER_LDG,
          int K_STATIC = 0>
__device__ __forceinline__ void topk_sigmoid_bias_mi300_task_impl(
    void *__restrict__ input_ptr, // [num_rows, NUM_EXPERTS]
    void *__restrict__ bias_ptr,  // [NUM_EXPERTS] e_score_correction_bias
    void *__restrict__ output_ptr, // [num_rows, k] (float weights)
    int const num_rows,
    int const k,
    void *__restrict__ routing_indices_ptr,   // [NUM_EXPERTS + S, num_rows] i32
    void *__restrict__ active_expert_ids_ptr, // [NUM_EXPERTS + S + 1] int32
    int const start_expert,
    int const end_expert,
    bool const renormalize,
    float const routed_scaling_factor,
    int const num_shared_experts) {
  T *input = static_cast<T *>(input_ptr);
  T *bias = static_cast<T *>(bias_ptr);
  float *output = static_cast<float *>(output_ptr);
  int *routing_indices = static_cast<int *>(routing_indices_ptr);
  int *active_expert_ids = static_cast<int *>(active_expert_ids_ptr);

  // Slot count per token: the k routed experts, plus the shared expert.
  int const k_total = k + num_shared_experts;

  // SP7 splits SP6[6] -- the 5.78 us selection half of the router's serial
  // tail -- five ways, so the next attempt at it aims at the right microsecond.
  // No s_waitcnt is inserted at the boundaries, deliberately: the stores here
  // are fire-and-forget and the drain is the release fence's (SP6[7]), so a
  // SP7 sum well short of SP6[6] localises the cost to that drain rather than
  // to any of these five.
#ifdef MPK_ENABLE_SUBPHASE_TIMING
  unsigned long long _tk_a0 = __builtin_amdgcn_s_memrealtime();
#endif

  // Initialize routing indices to 0.
  // active_expert_ids initialization is NOT needed: we write directly to
  // active_expert_ids[0..k-1] during TopK and set the count after.
  for (int expert = start_expert + threadIdx.x; expert < end_expert;
       expert += blockDim.x) {
    if (routing_indices != nullptr) {
      for (int row = 0; row < num_rows; ++row) {
        routing_indices[expert * num_rows + row] = 0;
      }
    }
  }
  // The shared expert takes slot k of every token, unconditionally.
  //
  // Write-through, unlike the zero fill above. The zeros are never read -- the
  // MoE tile decoder indexes routing_indices only at experts named by
  // active_expert_ids -- but the shared expert is always named, so this row is
  // read by W13 workgroups on all eight XCDs. Per-XCD L2 is not coherent: as a
  // standalone task the event boundary's buffer_wbl2 publishes it, and a fused
  // caller has no such boundary. Failure is silent and looks like a token
  // routed through the previous layer's shared expert.
  if (num_shared_experts > 0 && routing_indices != nullptr) {
    for (int row = threadIdx.x; row < num_rows; row += blockDim.x) {
      st_wt_u32((void *)&routing_indices[NUM_EXPERTS * num_rows + row],
                (unsigned)(k + 1));
    }
  }
  __syncthreads();

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  unsigned long long _tk_a1 = __builtin_amdgcn_s_memrealtime();
  if (threadIdx.x == 0 && g_subphase_active) {
    atomicAdd(&g_subphase_ns[7][0], (_tk_a1 - _tk_a0) * 10); // zero fill
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
  static_assert(VPT == 8,
                "branchless argmax asm below is specialized to VPT=8");

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
    T *thread_bias_ptr = bias + first_elt;

    // score_chunk: sigmoid(logit), the weight actually emitted.
    // row_chunk:   score + correction bias, used only for selection.
    float score_chunk[VPT];
    float row_chunk[VPT];
    // Vectorized load
    for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) {
      int base = ldg * ELTS_PER_LDG;
      int src_offset = ldg * THREADS_PER_ROW * ELTS_PER_LDG;
      for (int e = 0; e < ELTS_PER_LDG; ++e) {
        float const s =
            fast_sigmoid(static_cast<float>(thread_read_ptr[src_offset + e]));
        score_chunk[base + e] = s;
        row_chunk[base + e] =
            s + static_cast<float>(thread_bias_ptr[src_offset + e]);
      }
    }

#ifdef MPK_ENABLE_SUBPHASE_TIMING
    unsigned long long _tk_a2 = __builtin_amdgcn_s_memrealtime();
    if (threadIdx.x == 0 && g_subphase_active) {
      // The sigmoid consumes each loaded value, so the compiler's waitcnt is
      // inside the loop and this does capture the logit + bias fetch. Both
      // rows were published with sc0 sc1 stores by 256 blocks on eight XCDs,
      // so they are a cold 512 + 512 B read from HBM, not from this XCD's L2.
      atomicAdd(&g_subphase_ns[7][1], (_tk_a2 - _tk_a1) * 10); // logit+bias ld
    }
#endif

    // The zero fill of the logit row used to sit here, ahead of selection.
    // It now runs after it, because the sort-then-merge path below recovers
    // the eight winners' unbiased weights by re-reading their logits, and a
    // cleared row would hand it eight sigmoid(0) = 0.5. Nothing between the
    // two reads the row, so the move is order-only.

    // No max/sum reduction here: sigmoid is elementwise, unlike softmax.

    // Fused Top-K selection — Step 1 uses inline asm for branchless local
    // argmax.
    int const start_col = first_elt;
    float row_sum_for_renorm = 0.f;
    float topk_vals[8];
    int topk_experts[8];

    // Precompute expert column indices for branchless local argmax.
    int col[VPT];
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      col[i] = start_col + i;
    }

    // ── Branchless top-8 by sort-then-merge ────────────────────────────────
    // The k-loop below runs K strictly dependent rounds: local argmax, a
    // log2(THREADS_PER_ROW)-step shfl_xor reduce, then blank the winner so the
    // next round can run. Nothing in it overlaps -- round k+1's argmax needs
    // round k's blank, which needs round k's reduce -- and at GLM-5's 256
    // experts on 32 lanes it measured 3454 ns of the router's 8590 ns serial
    // tail (SP7[3] of SP6[4]), with 231 other blocks idle for all of it.
    //
    // Sorting has about the same operation count and no chain. Each lane
    // bitonic-sorts its own VPT=8 keys -- six stages of four independent
    // compare-exchanges -- and then log2(THREADS_PER_ROW) shfl_xor rounds each
    // merge two descending 8-lists into the top 8 of their union: with both
    // sorted descending, max(a[i], b[7-i]) is a bitonic sequence holding
    // exactly those eight, and three more stages sort it. Every lane in the
    // group ends with the same answer, so the winners need no final broadcast.
    //
    // One register per candidate, not three. The comparison key packs the
    // expert id into the low 8 bits of the order-preserving u32 image of the
    // biased score, so v_max_u32/v_min_u32 carry the id for free -- that is
    // what deletes the two v_cndmask per compare that made the argmax's
    // payload expensive, and it is why a network with more comparators than
    // the k-loop has fewer instructions. The cost is eight mantissa bits of
    // the *selection* key; the logits and the correction bias are both bf16
    // (8-bit mantissa), so two experts that collide in the surviving 16 bits
    // were already a tie, and the packed id then breaks it toward the higher
    // id, deterministically. The emitted weight is never truncated: it is
    // recomputed at full precision from the logit for the eight winners.
    constexpr bool FAST_SORT_MERGE = (K_STATIC == 8) && (VPT == 8) &&
                                     (LDG_PER_THREAD == 1) &&
                                     (NUM_EXPERTS <= 256);
    bool used_fast = false;
#define MTK_D(i, j)                                                            \
  {                                                                            \
    unsigned _a = key[i], _b = key[j];                                         \
    key[i] = _a > _b ? _a : _b;                                                \
    key[j] = _a > _b ? _b : _a;                                                \
  }
#define MTK_A(i, j)                                                            \
  {                                                                            \
    unsigned _a = key[i], _b = key[j];                                         \
    key[i] = _a > _b ? _b : _a;                                                \
    key[j] = _a > _b ? _a : _b;                                                \
  }
    if constexpr (FAST_SORT_MERGE) {
      if (k == 8) {
        used_fast = true;
        unsigned key[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          // Order-preserving float -> u32: flip the sign bit for positives,
          // invert everything for negatives. Biased scores are sigmoid plus a
          // signed correction bias, so both halves are reachable.
          unsigned b = __float_as_uint(row_chunk[i]);
          b = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
          key[i] = (b & 0xFFFFFF00u) | (unsigned)col[i];
        }
        // Bitonic sort of 8, descending: the textbook ascending network with
        // every arrow reversed. MTK_D puts the larger at the lower index.
        MTK_D(0, 1) MTK_A(2, 3) MTK_D(4, 5) MTK_A(6, 7)
        MTK_D(0, 2) MTK_D(1, 3) MTK_A(4, 6) MTK_A(5, 7)
        MTK_D(0, 1) MTK_D(2, 3) MTK_A(4, 5) MTK_A(6, 7)
        MTK_D(0, 4) MTK_D(1, 5) MTK_D(2, 6) MTK_D(3, 7)
        MTK_D(0, 2) MTK_D(1, 3) MTK_D(4, 6) MTK_D(5, 7)
        MTK_D(0, 1) MTK_D(2, 3) MTK_D(4, 5) MTK_D(6, 7)
#pragma unroll
        for (int mask = 1; mask < THREADS_PER_ROW; mask <<= 1) {
          unsigned o[8];
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            o[i] = (unsigned)__shfl_xor((int)key[i], mask, THREADS_PER_ROW);
          }
          // Reversing the partner's descending list and taking the elementwise
          // max keeps exactly the top 8 of the 16, as a bitonic sequence.
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            unsigned const b = o[7 - i];
            key[i] = key[i] > b ? key[i] : b;
          }
          MTK_D(0, 4) MTK_D(1, 5) MTK_D(2, 6) MTK_D(3, 7)
          MTK_D(0, 2) MTK_D(1, 3) MTK_D(4, 6) MTK_D(5, 7)
          MTK_D(0, 1) MTK_D(2, 3) MTK_D(4, 5) MTK_D(6, 7)
        }
        if (thread_group_idx == 0) {
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            int const e = (int)(key[i] & 0xFFu);
            // Full precision, and from this lane's L1: the row was read a few
            // hundred nanoseconds ago by this same wavefront.
            float const s = fast_sigmoid(static_cast<float>(thread_row_ptr[e]));
            topk_experts[i] = e;
            topk_vals[i] = s;
            row_sum_for_renorm += s;
          }
        }
      }
    }
#undef MTK_D
#undef MTK_A

    // topk_vals is 8 wide, so k has always been capped at 8 here; the unroll
    // bound just makes that cap explicit. The `break` is dead code whenever
    // K_STATIC == k, and it is what keeps the runtime-k callers correct.
    constexpr int K_UNROLL = (K_STATIC > 0) ? K_STATIC : 8;
#pragma unroll
    for (int k_idx = 0; !used_fast && k_idx < K_UNROLL; ++k_idx) {
      if (k_idx >= k) {
        break;
      }
      // ── Step 1: Branchless local argmax over VPT=8 elements ──
      // Carries three registers: the biased max (compare key), the expert id,
      // and the unbiased score (the value we ultimately emit).
      float max_val;
      int expert;
      float score;
      asm volatile("v_mov_b32 %[mv], %[r0]\n"
                   "v_mov_b32 %[ex], %[c0]\n"
                   "v_mov_b32 %[sc], %[s0]\n"
                   "v_cmp_gt_f32 vcc, %[r1], %[mv]\n"
                   "v_cndmask_b32 %[mv], %[mv], %[r1], vcc\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c1], vcc\n"
                   "v_cndmask_b32 %[sc], %[sc], %[s1], vcc\n"
                   "v_cmp_gt_f32 vcc, %[r2], %[mv]\n"
                   "v_cndmask_b32 %[mv], %[mv], %[r2], vcc\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c2], vcc\n"
                   "v_cndmask_b32 %[sc], %[sc], %[s2], vcc\n"
                   "v_cmp_gt_f32 vcc, %[r3], %[mv]\n"
                   "v_cndmask_b32 %[mv], %[mv], %[r3], vcc\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c3], vcc\n"
                   "v_cndmask_b32 %[sc], %[sc], %[s3], vcc\n"
                   "v_cmp_gt_f32 vcc, %[r4], %[mv]\n"
                   "v_cndmask_b32 %[mv], %[mv], %[r4], vcc\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c4], vcc\n"
                   "v_cndmask_b32 %[sc], %[sc], %[s4], vcc\n"
                   "v_cmp_gt_f32 vcc, %[r5], %[mv]\n"
                   "v_cndmask_b32 %[mv], %[mv], %[r5], vcc\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c5], vcc\n"
                   "v_cndmask_b32 %[sc], %[sc], %[s5], vcc\n"
                   "v_cmp_gt_f32 vcc, %[r6], %[mv]\n"
                   "v_cndmask_b32 %[mv], %[mv], %[r6], vcc\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c6], vcc\n"
                   "v_cndmask_b32 %[sc], %[sc], %[s6], vcc\n"
                   "v_cmp_gt_f32 vcc, %[r7], %[mv]\n"
                   "v_cndmask_b32 %[mv], %[mv], %[r7], vcc\n"
                   "v_cndmask_b32 %[ex], %[ex], %[c7], vcc\n"
                   "v_cndmask_b32 %[sc], %[sc], %[s7], vcc\n"
                   : [mv] "=&v"(max_val), [ex] "=&v"(expert), [sc] "=&v"(score)
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
                     [c7] "v"(col[7]),
                     [s0] "v"(score_chunk[0]),
                     [s1] "v"(score_chunk[1]),
                     [s2] "v"(score_chunk[2]),
                     [s3] "v"(score_chunk[3]),
                     [s4] "v"(score_chunk[4]),
                     [s5] "v"(score_chunk[5]),
                     [s6] "v"(score_chunk[6]),
                     [s7] "v"(score_chunk[7])
                   : "vcc");

      // ── Step 2: Branchless argmax reduce across subgroup ──
      // Uses __shfl_xor for cross-lane communication, inline asm for
      // branchless compare+select (eliminates s_and_saveexec divergence).
      for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        float other_max = __shfl_xor(max_val, mask, THREADS_PER_ROW);
        int other_expert = __shfl_xor(expert, mask, THREADS_PER_ROW);
        float other_score = __shfl_xor(score, mask, THREADS_PER_ROW);
        asm volatile("v_cmp_gt_f32 vcc, %[om], %[mv]\n"
                     "v_cndmask_b32 %[mv], %[mv], %[om], vcc\n"
                     "v_cndmask_b32 %[ex], %[ex], %[oe], vcc\n"
                     "v_cndmask_b32 %[sc], %[sc], %[os], vcc\n"
                     : [mv] "+v"(max_val), [ex] "+v"(expert), [sc] "+v"(score)
                     : [om] "v"(other_max),
                       [oe] "v"(other_expert),
                       [os] "v"(other_score)
                     : "vcc");
      }

      // ── Step 3: Record the winner ──
      // Registers only. Publishing here instead would put three st_wt_u32 in
      // the loop body, and each of those is an `asm volatile ... : "memory"`,
      // which is a full compiler barrier: 24 of them across the eight passes
      // pin the shuffle reduce, the blanking and the address arithmetic into
      // strict program order and leave the scheduler nothing to overlap. The
      // stores go out in one burst below.
      if (thread_group_idx == 0) {
        topk_vals[k_idx] = score;
        topk_experts[k_idx] = expert;
        row_sum_for_renorm += score;
      }

      // ── Step 4: Branchless blanking of winner ──
      // expert == col[i] matches exactly one thread + one element.
      // Only the biased array needs blanking: score_chunk is read solely via
      // the payload register carried out of Step 1.
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

#ifdef MPK_ENABLE_SUBPHASE_TIMING
    unsigned long long _tk_a3 = __builtin_amdgcn_s_memrealtime();
    if (threadIdx.x == 0 && g_subphase_active) {
      // Selection, whichever path ran: the sort-then-merge network, or the k
      // dependent argmax passes it replaced. Measured from _tk_a2 now that the
      // zero fill has moved past selection -- the slot still means the same
      // thing, and SP7[2] still means the fill.
      atomicAdd(&g_subphase_ns[7][3], (_tk_a3 - _tk_a2) * 10); // select
    }
#endif

    // Reset input buffer to 0 (for split-k gate linear compatibility).
    // Deliberately after selection: the winners' weights are recomputed from
    // this row. Lane 0 read eight arbitrary elements of it a few instructions
    // ago and every lane is about to overwrite its own eight, so drain first
    // -- the addresses cross lanes, which is exactly the case neither the
    // compiler's alias analysis nor same-wave VMEM ordering covers.
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) {
      int src_offset = ldg * THREADS_PER_ROW * ELTS_PER_LDG;
      for (int e = 0; e < ELTS_PER_LDG; ++e) {
        thread_read_ptr[src_offset + e] = static_cast<T>(0);
      }
    }

#ifdef MPK_ENABLE_SUBPHASE_TIMING
    unsigned long long _tk_a4 = __builtin_amdgcn_s_memrealtime();
    if (threadIdx.x == 0 && g_subphase_active) {
      atomicAdd(&g_subphase_ns[7][2], (_tk_a4 - _tk_a3) * 10); // logit clear
    }
#endif

    // Publish all k slots in one burst, folding the renormalisation in.
    //
    // The weight is the UNBIASED sigmoid score. The shared expert sits outside
    // the renormalised sum -- GLM adds it with weight 1 -- so only the k routed
    // slots are written here. Previously the loop stored score *
    // routed_scaling_factor for every slot and this block immediately
    // overwrote all k of them; that store was dead on the renormalising path,
    // which is every GLM call.
    if (thread_group_idx == 0) {
      float const inv = renormalize ? (routed_scaling_factor /
                                       row_sum_for_renorm)
                                    : routed_scaling_factor;
#pragma unroll
      for (int k_idx = 0; k_idx < K_UNROLL; ++k_idx) {
        if (k_idx >= k) {
          break;
        }
        int const expert = topk_experts[k_idx];
        st_wt_u32((void *)&output[k_total * thread_row + k_idx],
                  __float_as_uint(topk_vals[k_idx] * inv));
        if (expert >= start_expert && expert < end_expert &&
            routing_indices != nullptr) {
          st_wt_u32((void *)&routing_indices[(expert - start_expert) * num_rows +
                                             thread_row],
                    (unsigned)(k_idx + 1));
          if (active_expert_ids != nullptr) {
            st_wt_u32((void *)&active_expert_ids[k_idx], (unsigned)expert);
          }
        }
      }
    }

    // Shared expert: slot k, weight 1.0, unscaled and unrenormalized.
    if (num_shared_experts > 0 && thread_group_idx == 0) {
      st_wt_u32((void *)&output[k_total * thread_row + k],
                __float_as_uint(1.0f));
    }
  }
  __syncthreads();

  // Set the active expert list tail (thread 0 only, single wavefront handles
  // all rows for batch=1): the shared expert id, then the slot count.
  if (active_expert_ids != nullptr && threadIdx.x == 0) {
    if (num_shared_experts > 0) {
      st_wt_u32((void *)&active_expert_ids[k], (unsigned)NUM_EXPERTS);
    }
    st_wt_u32((void *)&active_expert_ids[NUM_EXPERTS + num_shared_experts],
              (unsigned)k_total);
  }

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  if (threadIdx.x == 0 && g_subphase_active) {
    // Whole-impl total, not a fifth segment: the renorm/shared-slot/tail
    // piece is [4] - ([0] + [1] + [2] + [3]). Measured from _tk_a0 because the
    // three inner timestamps are scoped to the `thread_row < num_rows` block.
    // [4] against SP6[6] also says how much of the selection half is the
    // buffer_inv and the noinline wrapper rather than this function.
    atomicAdd(&g_subphase_ns[7][4],
              (__builtin_amdgcn_s_memrealtime() - _tk_a0) * 10);
    // Must be g_subphase_cnt, not another g_subphase_ns phase: the dump loop
    // in persistent_kernel.cuh skips a whole slot on `g_subphase_cnt[s] == 0`
    // before it ever looks at the phases, so a slot that only writes ns is
    // silently dropped.
    atomicAdd(&g_subphase_cnt[7], 1ULL); // tail event count
  }
#endif
}

} // namespace kernel
