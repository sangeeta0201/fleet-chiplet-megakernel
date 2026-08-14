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

// Whole-layer fused gang task for GLM on MI300/MI350.
//
// One gang dispatch per decoder layer. The GLM counterpart of gpt-oss's
// gang_full_layer_fused_mi300.cuh, and built the same way: the two existing
// half-layer gang kernels run back to back inside one task body, with an
// in-kernel cross-XCD barrier where the task graph used to put an event.
//
// 15-phase pipeline (single dispatch, all 240 workers stay alive):
//
//   -- attention half, gang_mla_attn_fused_kernel_mi300 --
//   Phase  1: residual resolve + input RMSNorm + [q_a_proj | kv_a_proj]
//   Phase  2: qkv_a -> q_b barrier
//   Phase  3: q_a RMSNorm + absorbed q_b_proj + latent KV append
//   Phase  4: q_b -> decode barrier
//   Phase  5: absorbed MLA decode, split over (q_head_group, kv_chunk)
//   Phase  6: decode -> merge barrier
//   Phase  7: split-KV merge
//   -- the boundary this task exists to remove --
//   Phase  8: attention -> o_proj cross-XCD barrier
//   -- MoE half, gang_oproj_router_fused_kernel_mi300 --
//   Phase  9: absorbed o_proj (+ residual add)
//   Phase 10: o_proj -> router barrier
//   Phase 11: post-attention RMSNorm + sigmoid/bias router + TopK
//   Phase 12: routing-ready wait
//   Phase 13: MoE W13 + SwiGLU
//   Phase 14: W13 -> W2 barrier
//   Phase 15: MoE W2 + MulSumAdd
//
// Six dispatched tasks per layer become one. What is actually removed is five
// scheduler round trips and five event boundaries; what is added is one
// in-kernel barrier (Phase 8). The other thirteen phase boundaries already
// existed in one of the two halves.
//
// ── the counter buffer ────────────────────────────────────────────────────
// One buffer, because the two halves brought three between them and the input
// list only has 28 slots. HIER_STRIDE == 16 int32 (one cache line) per slot,
// exactly as in both halves; the sub-kernels are handed base pointers into
// this buffer and index it the way they always did.
//
//   [ 0 ..  9] * 16 : attention qkv_a -> q_b barrier
//   [10 .. 19] * 16 : attention q_b -> decode barrier
//   [20 .. 29] * 16 : attention decode -> merge barrier
//   [30 .. 39] * 16 : attention -> o_proj barrier   (new, this task)
//   [40 .. 49] * 16 : o_proj -> router barrier
//   [50 .. 59] * 16 : routing-ready epoch
//   [60 .. 69] * 16 : MoE W13 -> W2 barrier
//   [70]       * 16 : the router's own TopK arrival counter
//
// All of them are monotonic and never reset, so one buffer serves every layer
// of every iteration.
//
// ── why the release values are snapshotted here and not in the halves ─────
// Both halves read their barriers' release values ("current + 1") at the top
// of their own body, and both carry the same comment explaining why: reading
// a release value next to its own arrival atomic races inside the last
// arriving block. The read is only safe if it is ordered before anything in
// this layer could have published.
//
// Fusing breaks that for the MoE half. Its `routing_ready` snapshot used to
// sit behind an event boundary, so no worker could be running this layer's
// o_proj when any other worker took the snapshot. Inlined here it would sit
// *after* seven attention phases, and the argument no longer holds: a fast
// worker can finish o_proj and publish routing_ready while a slow one is
// still in the attention half and has not read the flag yet. It would then
// compute expected = published + 1 and spin for a layer that never comes.
//
// So all six values are snapshotted once, up here, before Phase 1 -- which is
// the earliest point in the task and therefore behind the same event boundary
// the halves used to rely on individually. The values are passed down into
// both halves, whose own snapshot blocks are compiled out.
//
// gpt-oss solves the same problem differently, with a deterministic layer
// counter published by the host into task_metadata._linear_reserved and read
// back as `layer_counter + 1`. It has to: it fuses all 36 layers into one
// task, which destroys the per-layer event boundary entirely, so there is no
// "before this layer ran" moment left to snapshot at. GLM still dispatches
// one task per layer and that boundary still exists. When task #14 lands the
// 47-layer loop, this block is what has to go and the layer counter is what
// replaces it -- see the layer_counter comment in
// gang_full_layer_fused_mi300.cuh.
//
// ── what the fusion requires of the halves ────────────────────────────────
// MERGE_WRITE_THROUGH must be on. The split-KV merge writes only this XCD's
// four tiles of attn_out and Phase 9's o_proj reduces over the whole row, so
// the consumer is on another XCD. Across an event that was free -- the
// scheduler's boundary makes the write visible. Across an in-kernel barrier
// it is not: a plain store lands in the producing XCD's L2 and the consumer's
// buffer_inv only drops its own vL1. The static_assert below is the whole
// safety net for that, and it is why demo.py forces the flag on this path
// rather than leaving it at the GLM_MLA_MERGE_WT default of 0.
//
// x_out needs nothing: the residual resolve already stores it with st_wt_u64.

#pragma once

#include "tasks/mi300/gang_mla_attn_fused_mi300.cuh"
// make_w_buffer_rsrc + __llvm_amdgcn_raw_buffer_load_lds, for the Phase 8
// o_proj weight prefetch. Reached transitively too, but the prefetch is a
// direct user.
#include "tasks/mi300/gang_moe_linear_mxfp4_mi300.cuh"
#include "tasks/mi300/gang_oproj_router_fused_mi300.cuh"

namespace kernel {

// Slot bases into the single counter buffer, in HIER_STRIDE units.
static constexpr int FULL_LAYER_ATTN_SLOT = 0;
static constexpr int FULL_LAYER_ATTN_RELEASE_SLOT = 30;
static constexpr int FULL_LAYER_OPROJ_SLOT = 40;
static constexpr int FULL_LAYER_ROUTER_COUNTER_SLOT = 70;
static constexpr int FULL_LAYER_COUNTER_SLOTS = 71;

template <
    // ── shared ──
    int BATCH_SIZE,
    // ── attention half ──
    int QKV_OUTPUT_PER_WG,
    int QKV_REDUCTION_SIZE,
    int QKV_ACTUAL_HIDDEN,
    int QB_OUTPUT_PER_WG,
    int QB_REDUCTION_SIZE,
    int QB_ACTUAL_HIDDEN,
    int KV_LORA_RANK,
    int QK_ROPE_HEAD_DIM,
    int KV_INPUT_STRIDE,
    int KV_CACHE_STRIDE,
    int MAX_SEQ_LEN,
    int PAGE_SIZE,
    int KV_INPUT_OFFSET,
    int NUM_Q_HEADS,
    int NUM_KV_CHUNKS,
    int Q_WORKSPACE_STRIDE,
    int MERGE_DIM_SPLITS,
    bool MERGE_WRITE_THROUGH,
    // ── MoE half ──
    int OPROJ_REDUCTION_SIZE,
    int OPROJ_ROWS_PER_WG,
    int HIDDEN_SIZE,
    int ACTUAL_HIDDEN_DIM,
    int NUM_EXPERTS,
    int TOPK_K,
    int MOE_INTERMEDIATE,
    int MOE_NUM_EXPERTS,
    int MOE_NUM_TOPK,
    int MOE_W13_TILES_PER_EXPERT,
    int MOE_W2_TILES_PER_EXPERT,
    int MOE_W13_OPW,
    int MOE_W2_OPW,
    bool MOE_WEIGHT_FP4 = false>
__device__ __noinline__ void gang_mla_full_layer_fused_kernel_mi300(
    // Pointer arrays are passed whole rather than unpacked into 38 named
    // parameters, which is what gpt-oss's full-layer task does and for the
    // same reason: the unpacked form costs a few hundred bytes of stack frame
    // per thread for pointers that are read once each.
    void *const *input_ptrs,  // 27, see the demo's layer for the map
    void *const *output_ptrs, // 11
    // ── runtime config ──
    int const *qo_indptr,
    int const *kv_indptr,
    int const *kv_indices,
    int const *kv_last_page_len,
    int num_active_tokens,
    // ── attention parameters ──
    int qkv_n_wgs_per_xcd,
    int qkv_output_stride,
    int qb_n_wgs_per_xcd,
    int qb_output_stride,
    int mla_tiles_per_xcd,
    int mla_total_work_items,
    int merge_tiles_per_xcd,
    float scale_s,
    float kv_eps,
    // ── MoE parameters ──
    int oproj_tiles_per_xcd,
    int router_tile_n,
    int total_barrier_arrivals,
    int total_router_tiles,
    bool renormalize,
    float routed_scaling_factor,
    int num_shared_experts,
    int moe_w13_tiles_per_xcd,
    int moe_w2_tiles_per_xcd,
    // ── shared parameters ──
    int tiles_per_xcd,
    int tile_idx) {

  static_assert(MERGE_WRITE_THROUGH,
                "the fused layer reads attn_out across an in-kernel barrier "
                "instead of an event, so the merge has to write through");
  static_assert(HIDDEN_SIZE == QKV_REDUCTION_SIZE,
                "o_proj's N is the next layer's qkv_a K; they are one row");

  int const tid = threadIdx.x;
  int const xcd_id = tile_idx / tiles_per_xcd;
  int const xcd_rank = tile_idx % tiles_per_xcd;

  // Drop anything vL1 is still holding from the previous layer's task. The
  // halves do not do this themselves -- they were entered through a dispatch,
  // which does it for them.
  asm volatile("buffer_inv" ::: "memory");

  constexpr int HIER_STRIDE = 16;
  int *const counters = static_cast<int *>(input_ptrs[14]);
  int *const attn_counters = counters + FULL_LAYER_ATTN_SLOT * HIER_STRIDE;
  int *const attn_release = counters + FULL_LAYER_ATTN_RELEASE_SLOT * HIER_STRIDE;
  int *const oproj_counters = counters + FULL_LAYER_OPROJ_SLOT * HIER_STRIDE;
  int *const router_counter =
      counters + FULL_LAYER_ROUTER_COUNTER_SLOT * HIER_STRIDE;

  // All six release values, plus this task's own, read before Phase 1. See
  // the header: this is the only point in the fused body that is still
  // behind the previous layer's event, and the two halves' own snapshots are
  // suppressed by passing these down.
  __shared__ int s_exp[7];
  if (tid == 0) {
    s_exp[0] = ld_nt_s32(&attn_counters[xcd_id * HIER_STRIDE]) + 1;
    s_exp[1] = ld_nt_s32(&attn_counters[(10 + xcd_id) * HIER_STRIDE]) + 1;
    s_exp[2] = ld_nt_s32(&attn_counters[(20 + xcd_id) * HIER_STRIDE]) + 1;
    s_exp[3] = ld_nt_s32(&attn_release[xcd_id * HIER_STRIDE]) + 1;
    s_exp[4] = ld_nt_s32(&oproj_counters[xcd_id * HIER_STRIDE]) + 1;
    // routing_ready's global epoch lives at its slot 0; the per-XCD release
    // flags are at [(1 + xcd) * HIER_STRIDE]. That asymmetry is the MoE
    // half's, reproduced here because it is the half that reads it back.
    s_exp[5] = ld_nt_s32(&oproj_counters[10 * HIER_STRIDE]) + 1;
    s_exp[6] = ld_nt_s32(&oproj_counters[(20 + xcd_id) * HIER_STRIDE]) + 1;
  }
  __syncthreads();
  int const attn_release_expected = s_exp[3];

  // ══════════════════════════════════════════════════════════════════════
  // Phases 1-7: the attention half
  // ══════════════════════════════════════════════════════════════════════
  // Workers past merge_tiles_per_xcd return out of this call early, from its
  // Phase 7 guard. They land on the Phase 8 barrier below with nothing to do
  // -- which is exactly where the o_proj weight prefetch is issued from.
  gang_mla_attn_fused_kernel_mi300<BATCH_SIZE,
                                   QKV_OUTPUT_PER_WG,
                                   QKV_REDUCTION_SIZE,
                                   QKV_ACTUAL_HIDDEN,
                                   QB_OUTPUT_PER_WG,
                                   QB_REDUCTION_SIZE,
                                   QB_ACTUAL_HIDDEN,
                                   KV_LORA_RANK,
                                   QK_ROPE_HEAD_DIM,
                                   KV_INPUT_STRIDE,
                                   KV_CACHE_STRIDE,
                                   MAX_SEQ_LEN,
                                   PAGE_SIZE,
                                   KV_INPUT_OFFSET,
                                   NUM_Q_HEADS,
                                   NUM_KV_CHUNKS,
                                   Q_WORKSPACE_STRIDE,
                                   MERGE_DIM_SPLITS,
                                   MERGE_WRITE_THROUGH>(
      /*x=*/input_ptrs[0],
      /*pre_norm_weight=*/input_ptrs[1],
      /*pre_norm_scratch=*/input_ptrs[2],
      /*qkv_weight=*/input_ptrs[3],
      /*qkv_bias=*/input_ptrs[4],
      /*q_a_norm_weight=*/input_ptrs[5],
      /*q_a_norm_scratch=*/input_ptrs[6],
      /*qb_weight=*/input_ptrs[7],
      /*qb_bias=*/input_ptrs[8],
      /*kv_norm_weight=*/input_ptrs[9],
      /*cos=*/input_ptrs[10],
      /*sin=*/input_ptrs[11],
      /*kv_cache=*/input_ptrs[12],
      /*attn_counters=*/attn_counters,
      /*moe_ws_f32=*/input_ptrs[13],
      /*qkv_a_out=*/output_ptrs[0],
      /*q_workspace=*/output_ptrs[1],
      /*lse=*/output_ptrs[2],
      /*o_acc=*/output_ptrs[3],
      /*attn_out=*/output_ptrs[4],
      /*x_out=*/output_ptrs[5],
      qo_indptr,
      kv_indptr,
      kv_indices,
      kv_last_page_len,
      /*request_id=*/(int16_t)0,
      num_active_tokens,
      qkv_n_wgs_per_xcd,
      qkv_output_stride,
      qb_n_wgs_per_xcd,
      qb_output_stride,
      mla_tiles_per_xcd,
      mla_total_work_items,
      merge_tiles_per_xcd,
      tiles_per_xcd,
      scale_s,
      kv_eps,
      tile_idx,
      /*qkv_expected_in=*/s_exp[0],
      /*qb_expected_in=*/s_exp[1],
      /*decode_expected_in=*/s_exp[2]);

  // ══════════════════════════════════════════════════════════════════════
  // Phase 8: attention -> o_proj cross-XCD barrier
  // ══════════════════════════════════════════════════════════════════════
  // The boundary this whole task exists to remove. In the six-task layer it
  // was an event: the merge task retired, the scheduler noticed, and the
  // o_proj task was dispatched. Here it is one atomic and one poll.
  //
  // Every dispatched worker arrives, including the ones that fell out of the
  // merge guard, so the count is the full dispatch width. Mechanism C: bump a
  // global counter, let the last arriver fan a write-through release out to
  // all eight per-XCD flags, and have everyone poll their own XCD's flag with
  // a non-temporal load. Never reset -- the release value came from the
  // snapshot at the top, so it is just "one more than last layer's".
#ifdef MPK_ENABLE_SUBPHASE_TIMING
  unsigned long long _fl_t0 = __builtin_amdgcn_s_memrealtime();
#endif
  {
    int const arrivals = tiles_per_xcd * 8;
    __syncthreads();
    // Not redundant with the __syncthreads: that orders execution, not the
    // visibility of stores issued by the 255 threads that are not tid 0. The
    // release atomic below has to be ordered after all of them.
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#ifdef MPK_ENABLE_SUBPHASE_TIMING
    // Phase 8 is the largest line in the profile by a wide margin, and "all
    // 240 workers wait 20 us" has two very different explanations: a straggler
    // the barrier is legitimately waiting for, or the barrier mechanism itself
    // costing that much. Split it four ways so the answer is read off rather
    // than argued about. Every arrival hits ONE address with a device-scope
    // atomic, so _b_atomic is where 240-way same-line contention would show.
    unsigned long long _b_t0 = 0, _b_t1 = 0, _b_t2 = 0, _b_t3 = 0;
    _b_t0 = __builtin_amdgcn_s_memrealtime();
#endif
    if (tid == 0) {
      int const prev =
          atom_add_release_gpu_s32(&attn_release[8 * HIER_STRIDE], 1);
#ifdef MPK_ENABLE_SUBPHASE_TIMING
      _b_t1 = __builtin_amdgcn_s_memrealtime();
#endif
      if ((prev % arrivals) == arrivals - 1) {
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        for (int x = 0; x < 8; x++) {
          st_wt_u32((void *)&attn_release[x * HIER_STRIDE],
                    (unsigned)attn_release_expected);
        }
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      }
#ifdef MPK_ENABLE_SUBPHASE_TIMING
      _b_t2 = __builtin_amdgcn_s_memrealtime();
#endif
    }
    // ── o_proj weight DMA, issued before the poll ────────────────────────
    // gpt-oss does exactly this at the equivalent barrier
    // (gang_full_layer_fused_mi300.cuh, Phase 6): 42 buffer_load_dwordx4-to-
    // LDS are issued and then the wave falls into the spin, so the DMA and
    // the wait overlap instead of running back to back. This barrier is the
    // single largest line in the subphase profile -- 151 s aggregate, ~27 us
    // per worker per layer -- because 208 of 240 workers drop out of the
    // merge guard and then spin through the whole decode tail. That is the
    // window being filled.
    //
    // The LDS copy is never read. Phase 9's gang_gemv_mxfp8_kernel loads its
    // weights straight from global, exactly as gpt-oss's Phase 7 o_proj does
    // (gang_linear_mxfp4_res_bias_mi300.cuh reads wg_data/wg_scales from the
    // global pointer, not from the prefetch buffer). So no LDS read path and
    // no MPK_W13_LDS_PREFETCH port is involved here: the DMA exists purely to
    // pull the 165 KB workgroup weight block up the hierarchy during a stall
    // the worker was going to eat anyway.
    //
    // One 4 KB LDS window, reused by all 42 loads, rather than gpt-oss's
    // j * 4096 ladder. Two reasons. gpt-oss's o_proj tile is small enough to
    // land a distinct slice per load; GLM's is OPROJ_ROWS_PER_WG *
    // OPROJ_REDUCTION_SIZE = 160 KB, over the LDS budget. And a moving
    // destination means a new M0 per load, which makes the compiler insert
    // s_waitcnt vmcnt(0) between them and serializes what should be 42
    // concurrent DMAs -- the same trap gang_moe_fused_mxfp4_mi300.cuh:285
    // works around with a single inline-asm block. A constant destination
    // sidesteps it without the asm.
    {
      constexpr int PF_WG_DATA = OPROJ_ROWS_PER_WG * OPROJ_REDUCTION_SIZE;
      constexpr int PF_WG_SCALE = OPROJ_ROWS_PER_WG * (OPROJ_REDUCTION_SIZE / 32);
      constexpr int PF_WG_BYTES = PF_WG_DATA + PF_WG_SCALE;
      constexpr int PF_N16 = (PF_WG_BYTES + 15) / 16;
      constexpr int PF_LPT = (PF_N16 + 255) / 256;
      // Past the activation staging area gang_gemv_mxfp8_kernel will fill in
      // Phase 9, so an in-flight load cannot land on top of its s_a.
      constexpr int PF_LDS_OFF =
          ((int)(sizeof(unsigned short) * BATCH_SIZE * OPROJ_REDUCTION_SIZE) +
           255) /
          256 * 256;
      static_assert(PF_LDS_OFF + 4096 <=
                        mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE,
                    "the o_proj prefetch window has to fit past the GEMV's "
                    "activation staging");

      if (xcd_rank < oproj_tiles_per_xcd) {
        extern __shared__ char _fused_smem[];
        i32x4_t const pf_rsrc = make_w_buffer_rsrc(
            input_ptrs[15],
            static_cast<uint32_t>(oproj_tiles_per_xcd) * PF_WG_BYTES);
        uint32_t const pf_wg_voff =
            static_cast<uint32_t>(xcd_rank) * PF_WG_BYTES;
        // buffer_load_lds writes to M0 + lane_id * 16, so one wave64 fills
        // 1024 bytes and the four waves need distinct 1 KB slices.
        auto *pf_dst =
            (__attribute__((address_space(3)))
             uint32_t *)(_fused_smem + PF_LDS_OFF + (tid >> 6) * 1024);
        // One incrementing voffset register, not PF_LPT independent ones.
        // The loads are deliberately issued back to back, so an expression
        // that depends on j keeps all PF_LPT of them live at once -- 42 VGPRs
        // here. That is what pushed worker_kernel past its register budget
        // when the same idiom was added a second time for the shared expert
        // (332 VGPRs / 0 spills -> 349 / 8, and 4.115 -> 4.762 ms). The tail
        // is clamped on the register instead of per load, since PF_N16 is not
        // a multiple of 256 and the last round would run off the slab.
        int pf_voff = static_cast<int>(pf_wg_voff) + tid * 16;
        int const pf_last = static_cast<int>(pf_wg_voff) + (PF_N16 - 1) * 16;
#pragma unroll
        for (int j = 0; j < PF_LPT; j++) {
          int const voff = pf_voff < pf_last ? pf_voff : pf_last;
          pf_voff += 4096;
          // aux = sc0, deliberately without gpt-oss's `nt`. gpt-oss passes 3
          // (sc0|nt); nt marks the line evict-first in L2, which is right
          // when the DMA's only purpose is to reach LDS. Here the point is
          // for Phase 9 to hit the line in L2, and GLM's o_proj block is
          // 165 KB per workgroup against gpt-oss's much smaller MXFP4 one,
          // so evict-first throws the prefetch away before it is used and
          // the weight is fetched from HBM twice. Measured end to end:
          // no prefetch 4.525 ms, aux=3 (nt) 4.674 ms, aux=1 4.486 ms.
          //
          // What the subphase counters say about the 4.486 (aggregate worker
          // ns over the run, MPK_SUBPHASE_TIMING=1):
          //
          //   SP3[0] o_proj GEMV   30.67 s -> 19.70 s   -36%
          //   SP3[3] RoutingWait   79.89 s -> 70.32 s   -12%
          //   SP5[0] this barrier  151.0 s -> 164.9 s    +9%
          //
          // So the L2 hit is real and large, and roughly two thirds of it is
          // handed back at the barrier: the 21 MB of prefetch traffic is
          // issued by all 240 workers while the 32 merge workers this
          // barrier is actually waiting on are still reading, and delays the
          // release by ~2.5 us per worker per layer. Prefetching less shrinks
          // both sides in the same proportion, so this is about the best the
          // idiom can do here; the remaining 27 us of Phase 8 spin is a
          // load-balance problem, not a latency-hiding one. See task #30.
          __llvm_amdgcn_raw_buffer_load_lds(pf_rsrc, pf_dst, 16, voff, 0, 0, 1);
          // Opaque to the optimizer, and load-bearing twice over. Without
          // it LLVM reassociates the PF_LPT offsets into PF_LPT
          // simultaneously-live VGPRs -- 42 here, 17 for the shared-expert
          // prefetch -- which together took worker_kernel from 332 VGPRs /
          // 0 spills to 349 / 8 and cost 4.115 -> 4.762 ms. Rolling the
          // loop up with #pragma unroll 1 fixes the registers but sinks
          // the m0 setup into the loop body, and then the DMA no longer
          // finishes under the spin: o_proj went 18.5 -> 33.5 s, worse
          // than not prefetching at all. This keeps both properties --
          // back-to-back loads, one offset register.
          asm volatile("" : "+v"(pf_voff) : : "memory");
        }
      }
    }
#ifdef MPK_ENABLE_SUBPHASE_TIMING
    _b_t3 = __builtin_amdgcn_s_memrealtime();
#endif
    if (tid == 0) {
      int *const my_flag = &attn_release[xcd_id * HIER_STRIDE];
      while (ld_nt_s32(my_flag) < attn_release_expected) {
        __builtin_amdgcn_s_sleep(1);
      }
    }
#ifdef MPK_ENABLE_SUBPHASE_TIMING
    if (tid == 0 && g_subphase_active) {
      unsigned long long _b_t4 = __builtin_amdgcn_s_memrealtime();
      atomicAdd(&g_subphase_ns[5][1], (_b_t1 - _b_t0) * 10); // arrival atomic
      atomicAdd(&g_subphase_ns[5][2], (_b_t2 - _b_t1) * 10); // release fan-out
      atomicAdd(&g_subphase_ns[5][3], (_b_t3 - _b_t2) * 10); // prefetch issue
      atomicAdd(&g_subphase_ns[5][4], (_b_t4 - _b_t3) * 10); // spin on flag
      // Split the spin by population. Phase 7's merge runs on the
      // merge_tiles_per_xcd ranks only; if those are the stragglers this
      // barrier waits for, their own spin is ~0 and everyone else's is the
      // full wait. If BOTH populations spin the same, the straggler is
      // somewhere else entirely and widening the merge would buy nothing.
      if (xcd_rank < merge_tiles_per_xcd) {
        atomicAdd(&g_subphase_ns[5][5], (_b_t4 - _b_t3) * 10); // merge ranks
      } else {
        atomicAdd(&g_subphase_ns[5][6], (_b_t4 - _b_t3) * 10); // idle ranks
      }
      // The merge ranks arrive ~23 us after everyone else but Phase 7's merge
      // compute is only ~2.4 us, so the time is in the entry region: the
      // __syncthreads plus the s_waitcnt vmcnt(0) that drains Phase 7's
      // write-through attn_out stores. Attribute it to confirm.
      if (xcd_rank < merge_tiles_per_xcd) {
        atomicAdd(&g_subphase_ns[5][7], (_b_t0 - _fl_t0) * 10);
      }
    }
#endif
    __syncthreads();
    // Plain buffer_inv, not an agent-scope acquire fence. The fence would
    // emit `buffer_inv sc1`, which also invalidates L2 and would throw away
    // the merge output this XCD just wrote for itself -- and the o_proj
    // weights the prefetch above just pulled into it.
    asm volatile("buffer_inv" ::: "memory");
    // Retire the prefetch DMA. Nothing reads its LDS window, but every phase
    // from 9 on reuses that memory, so the loads have to have landed before
    // one of them writes there.
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  }
#ifdef MPK_ENABLE_SUBPHASE_TIMING
  {
    unsigned long long _t = __builtin_amdgcn_s_memrealtime();
    if (tid == 0 && g_subphase_active) {
      atomicAdd(&g_subphase_ns[5][0], (_t - _fl_t0) * 10);
      atomicAdd(&g_subphase_cnt[5], 1ULL);
    }
  }
#endif

  // ══════════════════════════════════════════════════════════════════════
  // Phases 9-15: the MoE half
  // ══════════════════════════════════════════════════════════════════════
  // attn_out is output_ptrs[4] here rather than an input of its own: the two
  // halves declared it identically (unpartitioned) so one slot serves both,
  // and the input list has no room for a second.
  gang_oproj_router_fused_kernel_mi300<BATCH_SIZE,
                                       OPROJ_REDUCTION_SIZE,
                                       OPROJ_ROWS_PER_WG,
                                       HIDDEN_SIZE,
                                       ACTUAL_HIDDEN_DIM,
                                       NUM_EXPERTS,
                                       TOPK_K,
                                       MOE_INTERMEDIATE,
                                       MOE_NUM_EXPERTS,
                                       MOE_NUM_TOPK,
                                       MOE_W13_TILES_PER_EXPERT,
                                       MOE_W2_TILES_PER_EXPERT,
                                       MOE_W13_OPW,
                                       MOE_W2_OPW,
                                       MOE_WEIGHT_FP4>(
      /*oproj_input=*/output_ptrs[4],
      /*oproj_weight=*/input_ptrs[15],
      /*oproj_residual=*/input_ptrs[16],
      /*norm_weight=*/input_ptrs[17],
      /*norm_output=*/input_ptrs[18],
      /*router_weight=*/input_ptrs[19],
      /*router_bias=*/input_ptrs[20],
      /*logits_scratch=*/input_ptrs[21],
      /*router_counter=*/router_counter,
      /*oproj_counters=*/oproj_counters,
      /*moe_gate_up_weight=*/input_ptrs[22],
      /*moe_down_weight=*/input_ptrs[23],
      /*moe_w13_bias=*/input_ptrs[24],
      /*moe_w2_bias=*/input_ptrs[25],
      /*moe_swiglu_out=*/input_ptrs[26],
      /*hidden=*/output_ptrs[6],
      /*topk_weight=*/output_ptrs[7],
      /*routing_indices=*/output_ptrs[8],
      /*active_expert_ids=*/output_ptrs[9],
      /*moe_workspace_f32=*/output_ptrs[10],
      num_active_tokens,
      oproj_tiles_per_xcd,
      router_tile_n,
      tiles_per_xcd,
      total_barrier_arrivals,
      total_router_tiles,
      renormalize,
      routed_scaling_factor,
      num_shared_experts,
      moe_w13_tiles_per_xcd,
      moe_w2_tiles_per_xcd,
      tile_idx,
      /*oproj_expected_in=*/s_exp[4],
      /*routing_expected_in=*/s_exp[5],
      /*w13_expected_in=*/s_exp[6]);
}

} // namespace kernel
