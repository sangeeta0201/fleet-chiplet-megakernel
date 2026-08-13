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
// The barrier. gpt-oss uses the same global form -- one arrival counter, the
// last arriver fanning out eight per-XCD release flags with st_wt_u32, every
// thread polling its own flag with ld_nt_s32. That is Mechanism C from
// gang_linear_mxfp4_res_bias_rmsnorm_topk_mi300.cuh:689, taken unchanged. It
// is *not* a cheap per-XCD epoch there either, so nothing about GLM's geometry
// makes this boundary more expensive to fuse than gpt-oss's.
//
// What the barrier does cost is a dependent stall, and the thing that pays for
// it is prefetch. gamma and the gate row do not depend on the o_proj output,
// so they are issued before the poll and land while it spins; only then is the
// fusion worth as much as the dispatch it replaces. The wait therefore does
// not happen in this wrapper -- it is handed to the router kernel under
// OPROJ_BARRIER, which is the only place those loads can be issued from and
// still be consumed in registers. Getting this wrong is expensive and quiet:
// the first version of this file did the wait here and called the router as an
// opaque __noinline__ afterwards, which measured 5.44 ms against 5.34
// unfused. Adding the prefetch alone recovered 0.06 of that 0.10.
//
// Where it stands: 5.381 ms fused against 5.362 unfused, three samples each on
// one binary, tokens identical. Parity, inside the +-0.035 run-to-run noise --
// the 47 dispatch gaps the fusion removes (idle 1304.6 -> 1137.8 us/iter under
// --profiling) now roughly cancel the barrier. Off by default until it wins.
// The remaining exposed latency is Step 1 of the router: a dependent 4 KB
// re-read of the row plus a two-__syncthreads block reduction, all after the
// barrier, purely to get the RMSNorm sum of squares. Each o_proj workgroup
// already holds its 16 outputs in registers, so that sum can be reduced into
// the barrier itself and read as a single float. That is the next thing to
// try, and it is the one that would turn parity into a win.
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

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  // Slot 3 is OPROJ_TOPK, the same slot gpt-oss's monolith uses, and the same
  // phase numbering: 0 = o_proj compute, 1 = barrier wait, 2 = router.
  unsigned long long _sp_t0 = __builtin_amdgcn_s_memrealtime();
#endif

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
#ifdef MPK_ENABLE_SUBPHASE_TIMING
  {
    unsigned long long _sp_t1 = __builtin_amdgcn_s_memrealtime();
    if (tid == 0 && g_subphase_active) {
      atomicAdd(&g_subphase_ns[3][0], (_sp_t1 - _sp_t0) * 10); // OProjCompute
    }
    _sp_t0 = _sp_t1;
  }
#endif
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
    // Only the workers with Phase 3 work go on. The rest are o_proj-only:
    // their stores are retired and their arrival is counted, so they return
    // to the scheduler a whole router stage early instead of spinning.
    if (xcd_rank >= router_tile_n) {
      return;
    }
  }
  // The *wait* deliberately does not happen here, and neither does the
  // acquire. Both are handed to the router kernel, which issues its gamma and
  // gate-weight loads before polling so they are in flight while the barrier
  // spins. Doing the wait here, with the router called as an opaque
  // __noinline__ afterwards, exposes the full post-barrier load latency and is
  // what made the first version of this fusion slower than the dispatch it
  // replaced -- 5.44 ms against 5.34.
#ifdef MPK_ENABLE_SUBPHASE_TIMING
  {
    unsigned long long _sp_t2 = __builtin_amdgcn_s_memrealtime();
    if (tid == 0 && g_subphase_active) {
      atomicAdd(&g_subphase_ns[3][1], (_sp_t2 - _sp_t0) * 10); // BarrierWait
    }
    _sp_t0 = _sp_t2;
  }
#endif

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
                                         /*SIGMOID_BIAS=*/true,
                                         /*OPROJ_BARRIER=*/true>(
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
        num_shared_experts,
        hier_barrier,
        xcd_id,
        oproj_expected);
  }
#ifdef MPK_ENABLE_SUBPHASE_TIMING
  {
    unsigned long long _sp_t3 = __builtin_amdgcn_s_memrealtime();
    if (tid == 0 && g_subphase_active) {
      atomicAdd(&g_subphase_ns[3][2], (_sp_t3 - _sp_t0) * 10); // Router
      // Only the router tiles reach here, so cnt is the divisor for phases 1
      // and 2. Phase 0 is summed over all tiles and needs 2x this.
      atomicAdd(&g_subphase_cnt[3], 1ULL);
    }
  }
#endif
}

} // namespace kernel
