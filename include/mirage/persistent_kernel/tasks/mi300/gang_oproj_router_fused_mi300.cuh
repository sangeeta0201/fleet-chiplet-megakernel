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

// Fused absorbed-o_proj + post-attention RMSNorm + router + TopK + MoE
// (W13 + SwiGLU + W2 + MulSumAdd), for GLM.
//
// The GLM counterpart of gpt-oss's gang_oproj_topk_moe_fused_mi300.cuh, and
// built the same way: a thin wrapper that runs existing gang kernels back to
// back with in-kernel barriers where the task graph used to put dispatches.
// None of the sub-kernels is modified beyond a WRITE_THROUGH epilogue on the
// two producers whose consumers now sit on the other side of a barrier
// instead of the other side of an event.
//
// Why the whole tail of the layer and not one boundary at a time. Fusing just
// o_proj+router measured at parity (5.381 fused vs 5.362 unfused): the barrier
// it adds costs about what the dispatch it removes cost. That is the expected
// result for a pairwise fusion, and it is not the reason gpt-oss is fast.
// gpt-oss runs a layer in one or two tasks; GLM ran it in nine. The profile
// says the cost is not the dispatch per se but the *occupancy* around it --
// mean busy workers 85.7 of 240, and a 3.2x depwait/compute ratio on MoE W13.
// A 14%-occupancy stage like the router does not deserve its own 240-worker
// barrier; it deserves to ride inside a wide stage's workgroup. That only
// happens if the stages share a task.
//
// Barrier structure. Two independent barriers, exactly as gpt-oss splits them,
// in one counter tensor of 29 * 16 int32:
//
//   [0 .. 8]      o_proj -> router. Mechanism C: per-XCD release flags at
//                 [x * 16], global arrival counter at [8 * 16]. Arrivals are
//                 oproj_topk_tiles_per_xcd * 8, NOT workers_per_xcd * 8 --
//                 only the workers with o_proj/router work arrive here.
//   [10 .. 18]    routing_ready. Epoch at [10 * 16], per-XCD flags at
//                 [(11 + x) * 16]. Published by whichever block runs the TopK
//                 tail, polled by every worker before it touches routing data.
//                 Keeping this separate from the o_proj barrier is what lets
//                 the MoE-only workers skip the o_proj barrier entirely.
//   [20 .. 28]    W13 -> W2. Mechanism C again, flags at [(20 + x) * 16],
//                 counter at [28 * 16], workers_per_xcd * 8 arrivals -- every
//                 worker runs a W13 tile, so every worker arrives.
//
// Prefetch across the o_proj barrier is load-bearing and is why the *wait* for
// barrier 1 does not happen in this wrapper: it is handed to the router
// kernel under OPROJ_BARRIER, the only place gamma and the gate row can be
// issued from and still be consumed in registers. Doing the wait here with the
// router called as an opaque __noinline__ afterwards measured 5.44 ms against
// 5.34 unfused.
//
// Cache discipline. `buffer_inv` without sc1 drops vL1 and leaves this XCD's
// L2 alone, so the rule for every producer->consumer pair below is: same XCD,
// ordinary stores plus s_waitcnt; different XCD, st_wt. That makes three
// producers write-through -- o_proj's hidden row (read by all 8 routers), the
// TopK's routing outputs (read by all 8 XCDs' MoE tiles, including the shared
// expert's routing_indices row), and W13's SwiGLU activations (an expert's
// intermediate is spread over all 8 XCDs and W2 reduces over all of it). The
// normed row is deliberately NOT write-through: one router worker per XCD
// writes the whole row into its own L2 and only that XCD's W13 tiles read it.
//
// Dispatch: tiles_per_xcd = workers_per_xcd (30 for GLM), not the o_proj tile
// count -- the MoE phases need every worker. tile_idx is global (this task
// type is in runtime.cc's n_tile_start list), so tile_idx = xcd_id *
// tiles_per_xcd + xcd_rank. Every sub-kernel indexes tiles *within* an XCD, so
// each phase is handed xcd_rank, not tile_idx.

#pragma once
#include "tasks/mi300/gang_gemv_mxfp8_mi300.cuh"
#include "tasks/mi300/gang_moe_linear_mxfp8_mi300.cuh"
#include "tasks/mi300/gang_rmsnorm_linear_bias_mi300.cuh"

namespace kernel {

template <int BATCH_SIZE,
          int OPROJ_REDUCTION_SIZE, // absorbed o_proj K (10240 for GLM)
          int OPROJ_ROWS_PER_WG,    // output columns per workgroup
          int HIDDEN_SIZE,          // o_proj N == router reduction == MoE K
          int ACTUAL_HIDDEN_DIM,    // RMSNorm divisor, <= HIDDEN_SIZE
          int NUM_EXPERTS,          // routed experts (router N)
          int TOPK_K,               // routed experts per token
          int MOE_INTERMEDIATE,     // per-expert intermediate (1536 for GLM)
          int MOE_NUM_EXPERTS,      // routed + shared (65 for GLM)
          int MOE_NUM_TOPK,         // TOPK_K + shared (5 for GLM)
          int MOE_W13_TILES_PER_EXPERT,
          int MOE_W2_TILES_PER_EXPERT,
          int MOE_W13_OPW,
          int MOE_W2_OPW,
          // Expert weights as E2M1 nibbles rather than E4M3 bytes. Only the
          // two MoE sub-kernels see it; o_proj and the router stay MXFP8.
          bool MOE_WEIGHT_FP4 = false>
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
        void *oproj_counters_ptr,      // input_ptrs[9] this task's barriers
        // ── MoE inputs ──
        void const *moe_gate_up_weight_ptr, // input_ptrs[10] MXFP8
        void const *moe_down_weight_ptr,    // input_ptrs[11] MXFP8
        void const *moe_w13_bias_ptr,       // input_ptrs[12]
        void const *moe_w2_bias_ptr,        // input_ptrs[13]
        void *moe_swiglu_out_ptr,           // input_ptrs[14] [b, topk, inter]
        // ── outputs ──
        void *hidden_ptr,            // output_ptrs[0] attn_proj_out, whole row
        void *topk_weight_ptr,       // output_ptrs[1]
        void *routing_indices_ptr,   // output_ptrs[2]
        void *active_expert_ids_ptr, // output_ptrs[3]
        void *moe_workspace_f32_ptr, // output_ptrs[4]
        // ── parameters ──
        int num_active_tokens,
        int oproj_tiles_per_xcd,
        int router_tile_n,
        int tiles_per_xcd, // == workers_per_xcd
        int total_barrier_arrivals,
        int total_router_tiles,
        bool renormalize,
        float routed_scaling_factor,
        int num_shared_experts,
        int moe_w13_tiles_per_xcd,
        int moe_w2_tiles_per_xcd,
        int tile_idx,
        // Release values supplied by a caller that has already snapshotted
        // them. Negative means "snapshot them yourself", which is what the
        // standalone dispatch of this task does. The fused whole-layer task
        // passes real values because inlined there this body no longer sits
        // directly behind the previous layer's event boundary, and the
        // argument in the comment below stops holding for routing_ready --
        // see gang_mla_full_layer_fused_mi300.cuh.
        int oproj_expected_in = -1,
        int routing_expected_in = -1,
        int w13_expected_in = -1) {

  int const tid = threadIdx.x;
  int const xcd_id = tile_idx / tiles_per_xcd;
  int const xcd_rank = tile_idx % tiles_per_xcd;

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  // Slot 3 is OPROJ_TOPK, the same slot gpt-oss's monolith uses, and the same
  // phase numbering: 0 = o_proj compute, 1 = barrier wait, 2 = router,
  // 3 = routing-ready wait, 4 = MoE W13, 5 = W13->W2 barrier, 6 = MoE W2.
  unsigned long long _sp_t0 = __builtin_amdgcn_s_memrealtime();
#endif

  // Mechanism C barrier layout, HIER_STRIDE int32 (one cache line) per slot.
  constexpr int HIER_STRIDE = 16;
  int *hier_barrier = static_cast<int *>(oproj_counters_ptr);
  int *routing_ready = hier_barrier + 10 * HIER_STRIDE;
  int *w13_barrier = hier_barrier + 20 * HIER_STRIDE;

  // Read the release values this layer will publish *before* Phase 1, not
  // after. Reading next to the arrival atomic is the natural place and is what
  // the standalone O-proj kernel does, but it races within the last arriving
  // block: a straggler wave there can read the flag after its own thread 0 has
  // already bumped it, compute expected = published + 1, and spin forever. Up
  // here nothing in this layer has run, so no flag can have moved, and one
  // __syncthreads makes the values block-uniform.
  //
  // The same argument covers the MoE-only workers reading `routing_ready`
  // ahead of a TopK tail that could in principle already have published it:
  // every worker runs this task for every layer, and this layer's task cannot
  // start on any worker until the previous layer's event has fired, so the
  // earliest possible publication is a whole o_proj GEMM after the latest
  // possible arrival here.
  __shared__ int s_expected[3];
  if (oproj_expected_in < 0) {
    if (tid == 0) {
      s_expected[0] = ld_nt_s32(&hier_barrier[xcd_id * HIER_STRIDE]) + 1;
      s_expected[1] = ld_nt_s32(routing_ready) + 1;
      s_expected[2] = ld_nt_s32(&w13_barrier[xcd_id * HIER_STRIDE]) + 1;
    }
    __syncthreads();
  }
  // The override is a kernel argument, so the branch above is block-uniform
  // and the __syncthreads inside it is safe.
  int const oproj_expected =
      oproj_expected_in < 0 ? s_expected[0] : oproj_expected_in;
  int const routing_expected =
      routing_expected_in < 0 ? s_expected[1] : routing_expected_in;
  int const w13_expected =
      w13_expected_in < 0 ? s_expected[2] : w13_expected_in;

  // Zero the f32 MoE accumulator that this layer's W2 will atomicAdd into.
  //
  // Here, rather than in the task that reads it: the previous value is the
  // *previous* layer's MoE output, and its only reader is this layer's RMSNorm
  // prologue (the FUSE_RESADD fold), which is a whole task earlier in the event
  // chain. So by the time any worker reaches this line every read of the old
  // value has retired, and the Phase 6 W13->W2 barrier -- which every worker
  // arrives at -- separates the zero from the first accumulate. No barrier of
  // its own, and it serves the fused and unfused attention paths alike.
  //
  // Write-through: the consumer is a device-scope atomicAdd, performed past the
  // XCD's L2, so a dirty local line holding the zero could later be written
  // back over the accumulated result. One workgroup per XCD, each taking its
  // own eighth of the row, so the 8 KB is written exactly once.
  if (xcd_rank == 0) {
    constexpr int WS_TOTAL = BATCH_SIZE * HIDDEN_SIZE;
    static_assert(WS_TOTAL % (8 * 4) == 0,
                  "the workspace has to split into eight dwordx4-aligned "
                  "slices, one per XCD");
    constexpr int WS_PER_XCD = WS_TOTAL / 8;
    float *ws = static_cast<float *>(moe_workspace_f32_ptr) + xcd_id * WS_PER_XCD;
    for (int i = tid * 4; i < WS_PER_XCD; i += 256 * 4) {
      st_wt_u128((void *)(ws + i), 0u, 0u, 0u, 0u);
    }
  }

  // Workers with o_proj or router work. The rest fall straight through to the
  // routing-ready poll: they must not arrive at the o_proj barrier, whose
  // arrival count is sized to this set.
  int const oproj_topk_tiles_per_xcd =
      oproj_tiles_per_xcd > router_tile_n ? oproj_tiles_per_xcd : router_tile_n;

  if (xcd_rank < oproj_topk_tiles_per_xcd) {
    // ══════════════════════════════════════════════════════════════════════
    // Phase 1: absorbed o_proj (MXFP8 GEMV + residual)
    // ══════════════════════════════════════════════════════════════════════
    // The GEMV addresses its output as [n_tile * ROWS_PER_WG + row] within an
    // XCD's slice, so it wants the XCD's base pointer. `hidden_ptr` is the
    // whole row here -- it has to be, since Phase 3 norms all of it -- so the
    // slice is reconstructed rather than handed over by the partition map.
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

    // ══════════════════════════════════════════════════════════════════════
    // Phase 2: o_proj -> router barrier (arrival only)
    // ══════════════════════════════════════════════════════════════════════
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
    // The *wait* deliberately does not happen here, and neither does the
    // acquire. Both are handed to the router kernel, which issues its gamma
    // and gate-weight loads before polling so they are in flight while the
    // barrier spins.
#ifdef MPK_ENABLE_SUBPHASE_TIMING
    {
      unsigned long long _sp_t2 = __builtin_amdgcn_s_memrealtime();
      if (tid == 0 && g_subphase_active) {
        atomicAdd(&g_subphase_ns[3][1], (_sp_t2 - _sp_t0) * 10); // BarrierWait
      }
      _sp_t0 = _sp_t2;
    }
#endif

    // ══════════════════════════════════════════════════════════════════════
    // Phase 3: RMSNorm + router GEMV + sigmoid/bias TopK
    // ══════════════════════════════════════════════════════════════════════
    // One worker per expert, each redundantly re-norming the row; the kernel's
    // own atomic counter picks the last of the 64 to run the TopK tail.
    // Workers past router_tile_n must not enter, or that counter would
    // over-count. That tail publishes `routing_ready` for the MoE phases.
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
          oproj_expected,
          routing_ready);
    }
#ifdef MPK_ENABLE_SUBPHASE_TIMING
    {
      unsigned long long _sp_t3 = __builtin_amdgcn_s_memrealtime();
      if (tid == 0 && g_subphase_active) {
        atomicAdd(&g_subphase_ns[3][2], (_sp_t3 - _sp_t0) * 10); // Router
        atomicAdd(&g_subphase_cnt[3], 1ULL);
      }
      _sp_t0 = _sp_t3;
    }
#endif
  }

  // ════════════════════════════════════════════════════════════════════════
  // Phase 4: wait for routing
  // ════════════════════════════════════════════════════════════════════════
  // Every worker, including the ones that just ran the router -- for the block
  // that ran the TopK tail this poll succeeds on its first load. Separate from
  // the o_proj barrier so that the MoE-only workers, which never arrived
  // there, still have something to wait on.
  if (tid == 0) {
    int *my_flag = &routing_ready[(1 + xcd_id) * HIER_STRIDE];
    while (ld_nt_s32(my_flag) < routing_expected) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
  __syncthreads();
  // Acquire: drop this CU's vL1 so the routing data is re-read. No sc1 -- the
  // producer wrote through, and this XCD's L2 still holds the normed row that
  // Phase 5 is about to consume.
  asm volatile("buffer_inv" ::: "memory");
#ifdef MPK_ENABLE_SUBPHASE_TIMING
  {
    unsigned long long _sp_t4 = __builtin_amdgcn_s_memrealtime();
    if (tid == 0 && g_subphase_active) {
      atomicAdd(&g_subphase_ns[3][3], (_sp_t4 - _sp_t0) * 10); // RoutingWait
    }
    _sp_t0 = _sp_t4;
  }
#endif

  // ════════════════════════════════════════════════════════════════════════
  // Phase 5: MoE W13 (gate+up) with the SwiGLU folded into the epilogue
  // ════════════════════════════════════════════════════════════════════════
  for (int t = xcd_rank; t < moe_w13_tiles_per_xcd; t += tiles_per_xcd) {
    gang_moe_w13_linear_mxfp8_kernel<BATCH_SIZE,
                                     2 * MOE_INTERMEDIATE,
                                     2 * MOE_INTERMEDIATE,
                                     HIDDEN_SIZE,
                                     MOE_NUM_EXPERTS,
                                     MOE_NUM_TOPK,
                                     MOE_W13_TILES_PER_EXPERT,
                                     MOE_W13_OPW,
                                     /*FUSE_SWIGLU=*/true,
                                     /*WRITE_THROUGH=*/true,
                                     MOE_WEIGHT_FP4>(
        norm_output_ptr,
        moe_gate_up_weight_ptr,
        routing_indices_ptr,
        active_expert_ids_ptr,
        moe_w13_bias_ptr,
        moe_swiglu_out_ptr,
        t);
  }
#ifdef MPK_ENABLE_SUBPHASE_TIMING
  {
    unsigned long long _sp_t5 = __builtin_amdgcn_s_memrealtime();
    if (tid == 0 && g_subphase_active) {
      atomicAdd(&g_subphase_ns[3][4], (_sp_t5 - _sp_t0) * 10); // MoeW13
    }
    _sp_t0 = _sp_t5;
  }
#endif

  // ════════════════════════════════════════════════════════════════════════
  // Phase 6: W13 -> W2 barrier
  // ════════════════════════════════════════════════════════════════════════
  // Every worker arrives; only the W2 workers wait. The ones past
  // moe_w2_tiles_per_xcd return to the scheduler a whole W2 stage early
  // instead of spinning, which is the same trade Phase 2 makes for o_proj.
  __syncthreads();
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  if (tid == 0) {
    int prev = atom_add_release_gpu_s32(&w13_barrier[8 * HIER_STRIDE], 1);
    int const arrivals = tiles_per_xcd * 8;
    if ((prev % arrivals) == arrivals - 1) {
      for (int x = 0; x < 8; x++) {
        st_wt_u32((void *)&w13_barrier[x * HIER_STRIDE], (unsigned)w13_expected);
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }
  }
  if (xcd_rank >= moe_w2_tiles_per_xcd) {
    return;
  }
  if (tid == 0) {
    int *my_flag = &w13_barrier[xcd_id * HIER_STRIDE];
    while (ld_nt_s32(my_flag) < w13_expected) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
  __syncthreads();
  asm volatile("buffer_inv" ::: "memory");
#ifdef MPK_ENABLE_SUBPHASE_TIMING
  {
    unsigned long long _sp_t6 = __builtin_amdgcn_s_memrealtime();
    if (tid == 0 && g_subphase_active) {
      atomicAdd(&g_subphase_ns[3][5], (_sp_t6 - _sp_t0) * 10); // W13Barrier
    }
    _sp_t0 = _sp_t6;
  }
#endif

  // ════════════════════════════════════════════════════════════════════════
  // Phase 7: MoE W2 (down) with the routing-weight mul-sum-add folded in
  // ════════════════════════════════════════════════════════════════════════
  // The routing weights come from output[1], which the TopK tail wrote through
  // in Phase 3 -- W2 does not need a slot of its own for them.
  for (int t = xcd_rank; t < moe_w2_tiles_per_xcd; t += tiles_per_xcd) {
    gang_moe_w2_linear_mxfp8_kernel<BATCH_SIZE,
                                    HIDDEN_SIZE,
                                    HIDDEN_SIZE,
                                    MOE_INTERMEDIATE,
                                    MOE_NUM_EXPERTS,
                                    MOE_NUM_TOPK,
                                    MOE_W2_TILES_PER_EXPERT,
                                    MOE_W2_OPW,
                                    /*FUSE_MULSUMADD=*/true,
                                    MOE_WEIGHT_FP4>(
        moe_swiglu_out_ptr,
        moe_down_weight_ptr,
        routing_indices_ptr,
        active_expert_ids_ptr,
        moe_w2_bias_ptr,
        moe_workspace_f32_ptr,
        t,
        topk_weight_ptr);
  }
#ifdef MPK_ENABLE_SUBPHASE_TIMING
  {
    unsigned long long _sp_t7 = __builtin_amdgcn_s_memrealtime();
    if (tid == 0 && g_subphase_active) {
      atomicAdd(&g_subphase_ns[3][6], (_sp_t7 - _sp_t0) * 10); // MoeW2
    }
  }
#endif
}

} // namespace kernel
