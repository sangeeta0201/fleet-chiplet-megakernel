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

// Fused absorbed-o_proj + post-attention RMSNorm + router + TopK, for GLM.
//
// The GLM counterpart of gpt-oss's gang_oproj_topk_moe_fused_mi300.cuh, and
// built the same way: a thin wrapper that runs two existing gang kernels back
// to back with an in-kernel barrier where the task graph used to put a
// dispatch. Neither kernel is modified -- the o_proj GEMV only gains a
// WRITE_THROUGH epilogue, and the router is called with exactly the arguments
// its standalone registrar passes.
//
// Why these two. Of GLM's nine per-layer stages this boundary is the one where
// a dispatch buys the least: o_proj is 128 workgroups of ~15 us and the router
// is 64 workgroups that spend most of their span waiting to be handed a tile
// (the profile shows a 168 us/iter *start skew* on the TopK stage, as large as
// its serial tail). Fusing removes the event, the scheduler task, and the skew
// in one go, because the router's workers are already resident and holding
// their operands.
//
// Where it differs from gpt-oss. There the barrier is a cheap per-XCD epoch,
// because GQA pins kv_head == xcd_id and an XCD only ever reads what it wrote.
// GLM's post-attention RMSNorm spans the whole 2048-wide row while each XCD
// produced only its own 256 columns, so every worker here needs all eight
// XCDs' output. That forces the expensive form: one global arrival counter,
// the last arriver fanning out eight per-XCD release flags with st_wt_u32,
// and every thread polling its own XCD's flag with ld_nt_s32. It is the
// Mechanism C barrier from gang_linear_mxfp4_res_bias_rmsnorm_topk_mi300.cuh,
// unchanged.
//
// Result: correct, and slower. 5.44 ms against 5.34 with GLM_FUSE_OPROJ_ROUTER
// at 0 on the same binary, 50/50 identical tokens. The +0.10 ms is not
// compilation (that A/B is one .so), not the write-through epilogue (turning
// it off buys 0.02 ms), and not the poll traffic (releasing the 64 workers
// with no Phase 3 work buys nothing). Under --profiling the fusion does
// exactly what it was built to do -- 47 dispatch gaps gone, idle 1304.6 ->
// 1137.8 us/iter -- and still loses on the clock, so the barrier costs more
// than the boundary. Roughly: the dispatch it replaces is ~2.5 us/layer, the
// barrier ~4.5.
//
// The consequence is bigger than this one task. Every GLM stage boundary needs
// the global form of the barrier, for the reason in the paragraph above, so
// whole-layer fusion would pay this five or six times per layer to remove
// boundaries that MPK_PRECOMPUTED_DISPATCH has already made cheap. gpt-oss
// gets whole-layer fusion for free only because GQA lets its barriers stay
// per-XCD. Off by default; kept as the measurement behind that conclusion.
//
// Dispatch: tiles_per_xcd = max(oproj_tiles_per_xcd, router_tile_n), and
// tile_idx is global (this task type is in runtime.cc's n_tile_start list),
// so tile_idx = xcd_id * tiles_per_xcd + xcd_rank. Both sub-kernels index
// tiles *within* an XCD, so each phase is handed xcd_rank, not tile_idx.

#pragma once
#include "tasks/mi300/gang_gemv_mxfp8_mi300.cuh"
#include "tasks/mi300/gang_rmsnorm_linear_bias_mi300.cuh"

namespace kernel {

template <int BATCH_SIZE,
          int OPROJ_REDUCTION_SIZE, // absorbed o_proj K (10240 for GLM)
          int OPROJ_ROWS_PER_WG,    // output columns per workgroup
          int HIDDEN_SIZE,          // o_proj N == router reduction
          int ACTUAL_HIDDEN_DIM,    // RMSNorm divisor, <= HIDDEN_SIZE
          int NUM_EXPERTS,
          int TOPK_K>
__device__ __attribute__((always_inline)) void
    gang_oproj_router_fused_kernel_mi300(
        // ── o_proj inputs ──
        void const *oproj_input_ptr,    // input_ptrs[0] attn_out, replicated
        void const *oproj_weight_ptr,   // input_ptrs[1] MXFP8, XCD chunk
        void const *oproj_residual_ptr, // input_ptrs[2] x, XCD column slice
        // ── router inputs ──
        void const *norm_weight_ptr,   // input_ptrs[3] post-attn gamma
        void *norm_output_ptr,         // input_ptrs[4] normed row, for the MoE
        void const *router_weight_ptr, // input_ptrs[5] gate, XCD chunk
        void const *router_bias_ptr,   // input_ptrs[6] e_score_correction_bias
        void *logits_scratch_ptr,      // input_ptrs[7] XCD slice
        void *router_counter_ptr,      // input_ptrs[8] the router's own TopK
                                       //               arrival counter
        void *oproj_counters_ptr,      // input_ptrs[9] this task's barrier
        // ── outputs ──
        void *hidden_ptr,            // output_ptrs[0] attn_proj_out, whole row
        void *topk_weight_ptr,       // output_ptrs[1]
        void *routing_indices_ptr,   // output_ptrs[2]
        void *active_expert_ids_ptr, // output_ptrs[3]
        // ── parameters ──
        int num_active_tokens,
        int oproj_tiles_per_xcd,
        int router_tile_n,
        int tiles_per_xcd,
        int total_barrier_arrivals,
        int total_router_tiles,
        bool renormalize,
        float routed_scaling_factor,
        int num_shared_experts,
        int tile_idx) {

  int const tid = threadIdx.x;
  int const xcd_id = tile_idx / tiles_per_xcd;
  int const xcd_rank = tile_idx % tiles_per_xcd;

  // Mechanism C barrier layout, HIER_STRIDE int32 (one cache line) per slot:
  //   [x * 16] per-XCD release flag, monotonically increasing
  //   [8 * 16] global arrival counter, never reset
  constexpr int HIER_STRIDE = 16;
  int *hier_barrier = static_cast<int *>(oproj_counters_ptr);

  // Read the release value this layer will publish *before* Phase 1, not
  // after it. Reading it next to the arrival atomic is the natural place and
  // is what the standalone O-proj kernel does, but it races within the last
  // arriving block: a straggler wave there can read the flag after its own
  // thread 0 has already bumped it, compute expected = published + 1, and
  // spin forever. Up here no block has arrived yet, so no flag can have
  // moved, and one __syncthreads makes the value block-uniform.
  __shared__ int s_oproj_expected;
  if (tid == 0) {
    s_oproj_expected = ld_nt_s32(&hier_barrier[xcd_id * HIER_STRIDE]) + 1;
  }
  __syncthreads();
  int const oproj_expected = s_oproj_expected;

  // ════════════════════════════════════════════════════════════════════════
  // Phase 1: absorbed o_proj (MXFP8 GEMV + residual)
  // ════════════════════════════════════════════════════════════════════════
  // The GEMV addresses its output as [n_tile * ROWS_PER_WG + row] within an
  // XCD's slice, so it wants the XCD's base pointer. `hidden_ptr` is the whole
  // row here -- it has to be, since Phase 3 norms all of it -- so the slice is
  // reconstructed rather than handed over by the partition map.
  if (xcd_rank < oproj_tiles_per_xcd) {
    unsigned short *xcd_out =
        static_cast<unsigned short *>(hidden_ptr) +
        static_cast<size_t>(xcd_id) * oproj_tiles_per_xcd * OPROJ_ROWS_PER_WG;
    gang_gemv_mxfp8_kernel<BATCH_SIZE,
                           OPROJ_REDUCTION_SIZE,
                           OPROJ_ROWS_PER_WG,
                           /*HAS_RESIDUAL=*/true,
                           /*WRITE_THROUGH=*/true>(oproj_input_ptr,
                                                   oproj_weight_ptr,
                                                   oproj_residual_ptr,
                                                   xcd_out,
                                                   num_active_tokens,
                                                   OPROJ_ROWS_PER_WG,
                                                   HIDDEN_SIZE,
                                                   /*m_tiles=*/1,
                                                   oproj_tiles_per_xcd,
                                                   /*wgm=*/0,
                                                   xcd_rank);
  }

  // ════════════════════════════════════════════════════════════════════════
  // Phase 2: global barrier
  // ════════════════════════════════════════════════════════════════════════
  // __syncthreads is execution-only; s_waitcnt is what actually retires all
  // 256 threads' stores, and it has to happen before the release atomic or a
  // waiting XCD can be let through ahead of the data.
  __syncthreads();
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  {
    if (tid == 0) {
      int prev = atom_add_release_gpu_s32(&hier_barrier[8 * HIER_STRIDE], 1);
      // Modular test rather than a reset: the counter is monotonic for the
      // whole run, so there is no window in which a fast worker from the next
      // layer can observe a zeroed counter.
      if ((prev % total_barrier_arrivals) == total_barrier_arrivals - 1) {
        for (int x = 0; x < 8; x++) {
          st_wt_u32((void *)&hier_barrier[x * HIER_STRIDE],
                    (unsigned)oproj_expected);
        }
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      }
    }
    // Only the workers that have Phase 3 work wait. The other half of each
    // XCD's tiles are o_proj-only: once their stores are retired and their
    // arrival is counted they have nothing left to contribute, so they return
    // to the scheduler instead of joining the poll. That is worth doing for
    // its own sake -- it frees 64 of the 128 workers a whole router stage
    // early -- but mostly it halves the number of wavefronts hammering the
    // eight release lines with non-temporal loads while the stragglers are
    // still finishing their GEMV.
    if (xcd_rank >= router_tile_n) {
      return;
    }
    // Every thread polls for itself: no __syncthreads on the far side, and
    // ld_nt from 256 threads onto one line coalesces into a single request.
    while (ld_nt_s32(&hier_barrier[xcd_id * HIER_STRIDE]) < oproj_expected) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
  // Plain `buffer_inv`, no sc1: this invalidates the vL1 so the RMSNorm below
  // re-reads the o_proj output, and deliberately leaves L2 alone -- an
  // agent-scope acquire would throw away the lines this XCD wrote moments ago.
  // The o_proj epilogue is write-through, so the data is already past L2.
  asm volatile("buffer_inv" ::: "memory");

  // ════════════════════════════════════════════════════════════════════════
  // Phase 3: RMSNorm + router GEMV + sigmoid/bias TopK
  // ════════════════════════════════════════════════════════════════════════
  // One worker per expert, each redundantly re-norming the row; the kernel's
  // own atomic counter picks the last of the 64 to run the TopK tail. Workers
  // past router_tile_n must not enter, or that counter would over-count.
  if (xcd_rank < router_tile_n) {
    gang_rmsnorm_linear_bias_topk_kernel<__hip_bfloat16,
                                         BATCH_SIZE,
                                         HIDDEN_SIZE,
                                         ACTUAL_HIDDEN_DIM,
                                         NUM_EXPERTS,
                                         TOPK_K,
                                         /*SIGMOID_BIAS=*/true>(
        hidden_ptr,
        norm_weight_ptr,
        norm_output_ptr,
        router_weight_ptr,
        router_bias_ptr,
        logits_scratch_ptr,
        router_counter_ptr,
        topk_weight_ptr,
        routing_indices_ptr,
        active_expert_ids_ptr,
        num_active_tokens,
        /*tile_n=*/1,
        NUM_EXPERTS,
        /*m_tiles=*/1,
        router_tile_n,
        /*wgm=*/0,
        xcd_rank,
        total_router_tiles,
        renormalize,
        routed_scaling_factor,
        num_shared_experts);
  }
}

} // namespace kernel
