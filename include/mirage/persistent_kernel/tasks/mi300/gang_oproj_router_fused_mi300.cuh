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
// in one counter tensor of 39 * 16 int32:
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
//   [30 .. 38]    W_UV -> o_proj, WUV_ROWS_PER_WG > 0 only. Mechanism C,
//                 workers_per_xcd * 8 arrivals, arrive and wait in the same
//                 place. Unused slots when kv_b_v stays absorbed.
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

// Self-heal gate for the Mechanism-C flag polls. Same value and same
// reasoning as the copy in gang_mla_full_layer_fused_mi300.cuh, which carries
// the full note; duplicated under #ifndef because the monoliths land in this
// translation unit in include order and any one of them may be first.
#ifndef MPK_FL_REPUBLISH_SPINS
#define MPK_FL_REPUBLISH_SPINS 1024
#endif
// One boolean for "the W13 ceiling probe forced the flat arrival", so the
// self-heal's quota and the arrival cannot disagree across the #ifdef.
#ifdef MPK_W13_EARLY_REL
#define MPK_W13_EARLY_REL_ON 1
#else
#define MPK_W13_EARLY_REL_ON 0
#endif

// Layout of the symmetric EP signal array, in uint64 units. Duplicated from
// FULL_LAYER_EP_SIGNAL_STRIDE in gang_full_layer_fused_mi300.cuh rather than
// read from it -- the monoliths land in this translation unit in include order
// and either may be first. gang_mla_full_layer_fused_mi300.cuh static_asserts
// the two agree, from a point where both are in scope.
//
// One 64-byte line per PE, written only by that PE (on every peer's copy), so
// two independent per-layer signals can share it with no false sharing: slot 0
// is Phase 9's MoE fold, slot 1 is this task's o_proj all-gather.
static constexpr int OPROJ_EP_SIGNAL_STRIDE = 8;
static constexpr int OPROJ_EP_SIGNAL_SLOT = 1;

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
          bool MOE_WEIGHT_FP4 = false,
          // Expert parallelism. Only the two MoE sub-kernels see it: o_proj,
          // the RMSNorm and the router are replicated on every rank, because
          // the router has to produce the SAME activated list everywhere or
          // the ranks would disagree about who owns what. 1/0 compiles the
          // whole remap out.
          int EP_WORLD_SIZE = 1,
          int EP_MY_PE = 0,
          int EP_SHARED_PE = 0,
          // ── un-absorbed kv_b_v ────────────────────────────────────────
          // 0 keeps the absorbed o_proj: OPROJ_REDUCTION_SIZE is
          // NUM_Q_HEADS * KV_LORA_RANK and W_UV is already folded into the
          // weight. Non-zero makes OPROJ_REDUCTION_SIZE NUM_Q_HEADS *
          // WUV_V_HEAD_DIM and adds Phase 0, a block-diagonal GEMV
          //
          //   v[h * V_HEAD + j] = sum_c attn_out[h * KV_LORA + c] * W_UV[h][c][j]
          //
          // ahead of it. See the byte argument in the Phase 0 comment.
          int WUV_ROWS_PER_WG = 0,
          int WUV_REDUCTION = 0,  // == KV_LORA_RANK
          int WUV_V_HEAD_DIM = 0, // == v_head_dim
          // Experts per router tile. See Phase 3. 1 is one worker per expert,
          // which at GLM-5's 256 experts is 32 tiles per XCD against 29
          // workers -- two rounds for a mean of 1.10. `router_tile_n` is the
          // TILE count, so the caller divides by this.
          int ROUTER_EXPERTS_PER_TILE = 1>
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
        int w13_expected_in = -1,
        // ── un-absorbed kv_b_v, WUV_ROWS_PER_WG > 0 only ──
        // Trailing rather than grouped with the o_proj inputs so that adding
        // them renumbers neither the input nor the output list.
        void const *wuv_weight_ptr = nullptr, // MXFP8 [H * V_HEAD, KV_LORA]
        void *v_out_ptr = nullptr,            // [b, H * V_HEAD], o_proj's input
        int wuv_tiles_per_xcd = 0,
        int wuv_expected_in = -1,
        // Null means "slots [30 .. 38] of my own counter tensor", which is
        // where the standalone dispatch puts them. The whole-layer task passes
        // an explicit base instead: its merged buffer gives this task only
        // slots [40 .. 69], and 0/10/20 already fill them.
        void *wuv_counters_ptr = nullptr,
        // Symmetric [EP_WORLD_SIZE * 8] uint64 signal array, the same object
        // the Phase-9 EP fold uses. Slot 0 of a PE's 64-byte line is the
        // fold's; slot 1 is this task's o_proj all-gather. Sharing the line is
        // safe because both are written only by that PE, on every peer, and
        // both carry the same run-monotonic layer count. Null on the
        // standalone dispatch, and null is also what disables the all-gather.
        void *ep_signal_ptr = nullptr,
        // ── router fold, ROUTER_FOLD only ──────────────────────────────────
        // logit_e = irms * sum_i h_i * gamma_i * W[e,i], and irms is a
        // positive scalar over the whole row, so both the gate dot and the
        // RMSNorm's sum-of-squares are contractions over exactly the hidden
        // partition the o_proj shard already imposes. Phase 1 can therefore
        // compute this rank's share of both before the barrier instead of
        // Phase 3 recomputing all of it after -- see the block in Phase 1.
        //
        // [OPROJ_TP_COLS, NUM_EXPERTS] bf16: the gate weight transposed and
        // sliced to this rank's columns. Transposed because a tile owns four
        // hidden columns and all 256 experts, so the row-major form would have
        // it read 256 lines at a 12 KB stride to use 8 bytes of each.
        void const *router_weight_t_ptr = nullptr,
        // Symmetric float scratch, [8 + EP_WORLD_SIZE][NUM_EXPERTS + 1]. The
        // first eight lines are this rank's per-XCD accumulators, reduced
        // locally; the rest are the per-rank lines the peers push. Element
        // NUM_EXPERTS of a line is the sum-of-squares, which rides along
        // because it reduces over the identical partition.
        void *router_partials_ptr = nullptr) {

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
  // Phase 0's W_UV -> o_proj barrier. The full-layer caller hands down its own
  // base (its [30] is the attention release); standalone falls back to [30].
  // Only ever dereferenced under `WUV_ROWS_PER_WG > 0` -- with W_UV absorbed
  // the standalone counter tensor is 29 slots and [30] is off the end.
  int *wuv_barrier = wuv_counters_ptr ? static_cast<int *>(wuv_counters_ptr)
                                      : hier_barrier + 30 * HIER_STRIDE;

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
  __shared__ int s_expected[4];
  if (oproj_expected_in < 0) {
    if (tid == 0) {
      s_expected[0] = ld_nt_s32(&hier_barrier[xcd_id * HIER_STRIDE]) + 1;
      s_expected[1] = ld_nt_s32(routing_ready) + 1;
      s_expected[2] = ld_nt_s32(&w13_barrier[xcd_id * HIER_STRIDE]) + 1;
      // Guarded: with W_UV absorbed the standalone counter tensor is 29
      // slots and [30 + xcd_id] is off the end. The snapshot below the
      // branch stays branch-free either way.
      s_expected[3] = (WUV_ROWS_PER_WG > 0)
                          ? ld_nt_s32(&wuv_barrier[xcd_id * HIER_STRIDE]) + 1
                          : 0;
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
  int const wuv_expected =
      wuv_expected_in < 0 ? s_expected[3] : wuv_expected_in;

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

  constexpr bool UNABSORB_V = WUV_ROWS_PER_WG > 0;
  // ════════════════════════════════════════════════════════════════════════
  // Phase 0: un-absorbed kv_b_v (W_UV), a block-diagonal GEMV
  // ════════════════════════════════════════════════════════════════════════
  // Why undo an absorption that was put in deliberately. Absorption is a
  // FLOP-for-bytes trade, and at GLM-5 on 8 GPUs the bytes are all that
  // matter. Attention is replicated on every rank (DP attention, EP MoE), so
  // each rank streams the whole of it per token, and per layer that is
  //
  //   qkv_a       6144 x 2624   =  16.1 M
  //   q_b         2048 x 36864  =  75.5 M   (W_UK absorbed)
  //   o_proj      6144 x 32768  = 201.3 M   (W_UV absorbed)
  //                             = 292.9 M weights, 302 MB at MXFP8
  //
  // against ~40 MB/layer for the MoE half at EP=8 and top-8 of 256. o_proj
  // alone is 60% of the token's bytes. Un-absorbed it is 6144 x 16384 =
  // 100.7 M, and W_UV is 64 x 512 x 256 = 8.4 M: 201.3 -> 109.1 M, i.e.
  // -95 MB per layer, -7.4 GB per token over 78 layers.
  //
  // What makes it exact at decode is that the PV accumulation is linear in
  // the cached latent. The decode's per-head output is sum_t a_t * c_t over
  // the KV_LORA-wide latent rows, so applying W_UV[h] to that sum is the same
  // as having applied it to every c_t -- which is what absorbing it into
  // o_proj did. It costs one 512x256 matvec per head, 8.4 M MACs on a single
  // token, i.e. nothing but its own weight bytes.
  //
  // Shape. The GEMV is the same gang_gemv_mxfp8_kernel o_proj runs; only the
  // activation pointer moves per tile. Output row n belongs to head
  // n / V_HEAD_DIM and reduces over that head's KV_LORA latent slice, so
  // WUV_ROWS_PER_WG has to divide V_HEAD_DIM for a workgroup's rows to share
  // one head. The packed weight is W_UV.transpose(1, 2) flattened to
  // [H * V_HEAD_DIM, KV_LORA_RANK], so a global tile index addresses it the
  // same way it addresses any other workgroup-packed weight.
  //
  // The barrier. Phase 1 reduces over the whole v row, so W_UV has to be
  // complete on every XCD before o_proj starts -- one more cross-XCD barrier
  // than the absorbed form needs. It should be a cheap one: unlike the
  // attention->o_proj barrier, which waits on a merge that only four ranks per
  // XCD run, every worker here gets one or two equal tiles, so they arrive
  // together.
  //
  // MEASURED, GLM-5 744B, NP=8 EP, 78 layers, rank 0. GLM_UNABSORB_OPROJ=0 is
  // the ablation. Aggregate worker-seconds, MPK_SUBPHASE_TIMING=1, SP3 cnt
  // 2209800:
  //
  //   SP3 slot            absorbed   un-absorbed   delta
  //   [0] OProjCompute     120.65        58.86     -61.79
  //   [7] Wuv                  --        30.02     +30.02
  //   [2] Router            52.94        33.41     -19.53
  //   SP5[0] Phase-8 bar    56.71        50.73      -5.98
  //   SP4 (control)        110.87       108.81      -2.06
  //
  // Three uninstrumented runs each, decode host clock:
  //   absorbed     17.441 / 17.134 / 17.254  mean 17.28 ms
  //   un-absorbed  14.779 / 15.055 / 14.706  mean 14.85 ms   -14.1%
  //
  // o_proj halves because its K halves, and Router falls with it -- the 95 MB
  // per layer this stops reading is 95 MB the router's own weights no longer
  // contend with. The Wuv phase costs back 8.4 MB of weight plus a barrier.
  //
  // Accuracy: demo/glm5/run_correctness_suite.sh, 4 prompts, NP=8. Both forms
  // agree with the torch reference to exactly the same prefix (0/90, 28/256,
  // 13/256, 0/256) and all 8 ranks are identical in both. The two forms
  // diverge from each other at token 14-51, as expected -- un-absorbed
  // quantizes W_UV to MXFP8 on its own and rounds v to bf16, where the
  // absorbed form quantized the product.
  if constexpr (UNABSORB_V) {
    static_assert(WUV_V_HEAD_DIM % WUV_ROWS_PER_WG == 0,
                  "a workgroup's rows must sit inside one head, or its "
                  "activation slice would not be contiguous");
    static_assert(OPROJ_REDUCTION_SIZE % WUV_V_HEAD_DIM == 0,
                  "o_proj's K is the whole v row, H * V_HEAD_DIM");
    constexpr int TILES_PER_HEAD = WUV_V_HEAD_DIM / WUV_ROWS_PER_WG;
    // W_UV is the only one of the three bf16-activation GEMVs whose shape the
    // FP8 MFMA can take. o_proj wants 4-row tiles at the NP=8 shard (96
    // columns per XCD, 24 tiles against 29 workers) where the MFMA's unit is
    // 64; and W_UK reduces over QK_NOPE_HEAD_DIM = 192, which is not even a
    // multiple of the 128-wide k-tile, let alone the depth-4 pipeline's 512.
    // Padding 192 -> 512 would be 2.7x the bytes on a memory-bound stage.
    // The row width is not a free choice under MPK_WUV_MFMA -- the MFMA
    // kernel's OUTPUT_PER_WG is a hardcoded 4 waves x 16 rows -- so the demo
    // pins GLM_WUV_GEMV_ROWS to 64 when it sets the flag, and this falls back
    // rather than mis-tiling if some other width reaches here.
#if defined(MPK_WUV_MFMA)
    constexpr bool WUV_USE_MFMA =
        (WUV_ROWS_PER_WG == 64) && (WUV_REDUCTION % 512 == 0);
#else
    constexpr bool WUV_USE_MFMA = false;
#endif
    // `wuv_weight_ptr` is this XCD's dim-0 slice of the packed weight, the
    // same partitioning o_proj's input [15] gets, so the GEMV is handed the
    // *local* tile index against wuv_tiles_per_xcd n_tiles. Only the head
    // lookup and the output column need the global index -- and the column is
    // biased into the pointer, exactly like Phase 1's xcd_out.
    unsigned short *xcd_v_out =
        static_cast<unsigned short *>(v_out_ptr) +
        static_cast<size_t>(xcd_id) * wuv_tiles_per_xcd * WUV_ROWS_PER_WG;
#ifdef MPK_ENABLE_SUBPHASE_TIMING
    // [3][7] below is "W_UV + its GPU-wide barrier". Slot 0's spare entry [7]
    // is the GEMV loop alone and [2][7] the barrier, so the two can be told
    // apart; the tile count comes from the W_UK counter's twin at [7][5].
    unsigned long long _sp_wuv = __builtin_amdgcn_s_memrealtime();
    int _wuv_tiles = 0;
#endif
    for (int t = xcd_rank; t < wuv_tiles_per_xcd; t += tiles_per_xcd) {
      int const g = xcd_id * wuv_tiles_per_xcd + t;
      unsigned short const *head_in =
          static_cast<unsigned short const *>(oproj_input_ptr) +
          static_cast<size_t>(g / TILES_PER_HEAD) * WUV_REDUCTION;
      if constexpr (WUV_USE_MFMA) {
        // FP8 activation. The GEMV expands the weight to bf16 a pair at a time
        // (v_cvt_scalef32_pk_bf16_fp8) and accumulates with v_dot2c_f32_bf16,
        // ~82 VALU ops per 128 B of weight per lane; this quantizes the
        // activation to FP8 as well and lets the MFMA's scale operands do the
        // dequant in hardware. There is no third option on gfx950 -- the fp8
        // VALU dots (v_dot4_f32_fp8_fp8 and friends) need target feature
        // dot11-insts, which gfx942 has and gfx950 does not, so an fp8
        // activation means v_mfma_scale_f32_16x16x128_f8f6f4 or nothing.
        //
        // The packed weight is byte-identical: pack_dense_mxfp8 lays a
        // workgroup out as [OPW][K] E4M3 then [OPW][K/32] E8M0, and both
        // kernels compute the same WG_BYTES from it. Only the gather differs.
        //
        // Measured over a 4 GiB HBM-resident weight buffer at K=10240
        // (tests/standalone/test_fp8_act_mfma_bw.hip), GB/s per workgroup:
        //
        //   blocks   streaming roof   GEMV (LDS act)   this
        //      128           37.44             18.7    21.02
        //      240           22.37             17.2    19.30
        //
        // +12%, taking the stage from 77% to 86% of the achievable streaming
        // rate. Not the 4x the instruction count suggests: the loop was never
        // VALU-throughput-bound, the VALU was crowding load issue, and only
        // part of that recovers.
        gang_linear_mxfp8_kernel<BATCH_SIZE,
                                 WUV_REDUCTION,
                                 /*WRITE_THROUGH=*/true>(head_in,
                                                         wuv_weight_ptr,
                                                         xcd_v_out,
                                                         num_active_tokens,
                                                         WUV_ROWS_PER_WG,
                                                         OPROJ_REDUCTION_SIZE,
                                                         /*output_size=*/
                                                         OPROJ_REDUCTION_SIZE,
                                                         /*m_tiles=*/1,
                                                         wuv_tiles_per_xcd,
                                                         /*wgm=*/0,
                                                         t,
                                                         /*bias_ptr=*/nullptr);
      } else {
        gang_gemv_mxfp8_kernel<BATCH_SIZE,
                               WUV_REDUCTION,
                               WUV_ROWS_PER_WG,
                               /*HAS_RESIDUAL=*/false,
                               /*WRITE_THROUGH=*/true>(head_in,
                                                       wuv_weight_ptr,
                                                       /*residual=*/nullptr,
                                                       xcd_v_out,
                                                       num_active_tokens,
                                                       WUV_ROWS_PER_WG,
                                                       OPROJ_REDUCTION_SIZE,
                                                       /*m_tiles=*/1,
                                                       wuv_tiles_per_xcd,
                                                       /*wgm=*/0,
                                                       t);
      }
#ifdef MPK_ENABLE_SUBPHASE_TIMING
      ++_wuv_tiles;
#endif
    }
#ifdef MPK_ENABLE_SUBPHASE_TIMING
    {
      unsigned long long _t = __builtin_amdgcn_s_memrealtime();
      if (tid == 0 && g_subphase_active) {
        atomicAdd(&g_subphase_ns[0][7], (_t - _sp_wuv) * 10);
        atomicAdd(&g_subphase_ns[7][5], (unsigned long long)_wuv_tiles);
      }
      _sp_wuv = _t;
    }
#endif
    // Mechanism C, arrive and wait, the same shape as Phase 6's W13 -> W2.
    // Every worker arrives and every worker waits: Phase 1's participant set
    // is narrower, but a worker that skipped the wait would fall through to
    // the routing-ready poll and could reach Phase 4 -- which reads the normed
    // row derived from o_proj -- before o_proj had a correct v to read.
    __syncthreads();
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    int const wuv_arrivals = tiles_per_xcd * 8;
    bool const wuv_tree = MPK_BAR_TREE && (wuv_arrivals == tiles_per_xcd * 8);
    if (tid == 0) {
      if (hier_barrier_arrive(wuv_barrier, HIER_STRIDE, wuv_arrivals,
                              tiles_per_xcd, xcd_id, wuv_tree)) {
        for (int x = 0; x < 8; x++) {
          st_wt_u32((void *)&wuv_barrier[x * HIER_STRIDE],
                    (unsigned)wuv_expected);
        }
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      }
      int *my_flag = &wuv_barrier[xcd_id * HIER_STRIDE];
      MPK_WS_WAIT_BEGIN(767, wuv_expected);
      int _spins = 0;
      int _obs;
      while ((_obs = ld_nt_s32(my_flag)) < wuv_expected) {
        ++_spins;
        MPK_WS_WAIT_TICK(_obs, _spins);
        if ((_spins & (MPK_FL_REPUBLISH_SPINS - 1)) == 0) {
          if (ld_nt_s32(&wuv_barrier[8 * HIER_STRIDE]) >=
              hier_barrier_heal_quota(wuv_arrivals, wuv_tree) * wuv_expected) {
            st_wt_u32((void *)my_flag, (unsigned)wuv_expected);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          }
        }
        __builtin_amdgcn_s_sleep(1);
      }
    }
    __syncthreads();
    asm volatile("buffer_inv" ::: "memory");
#ifdef MPK_ENABLE_SUBPHASE_TIMING
    if (tid == 0 && g_subphase_active) {
      atomicAdd(&g_subphase_ns[2][7],
                (__builtin_amdgcn_s_memrealtime() - _sp_wuv) * 10);
    }
#endif
  }
#ifdef MPK_ENABLE_SUBPHASE_TIMING
  {
    unsigned long long _sp_tv = __builtin_amdgcn_s_memrealtime();
    if (tid == 0 && g_subphase_active) {
      atomicAdd(&g_subphase_ns[3][7], (_sp_tv - _sp_t0) * 10); // Wuv
    }
    _sp_t0 = _sp_tv;
  }
#endif

  // Workers with o_proj or router work. The rest fall straight through to the
  // routing-ready poll: they must not arrive at the o_proj barrier, whose
  // arrival count is sized to this set.
  int const oproj_topk_tiles_per_xcd =
      oproj_tiles_per_xcd > router_tile_n ? oproj_tiles_per_xcd : router_tile_n;

  if (xcd_rank < oproj_topk_tiles_per_xcd) {
    MPK_WS_PHASE(71, routing_expected, xcd_id);
    // ══════════════════════════════════════════════════════════════════════
    // Phase 1: absorbed o_proj (MXFP8 GEMV + residual)
    // ══════════════════════════════════════════════════════════════════════
    // The GEMV addresses its output as [n_tile * ROWS_PER_WG + row] within an
    // XCD's slice, so it wants the XCD's base pointer. `hidden_ptr` is the
    // whole row here -- it has to be, since Phase 3 norms all of it -- so the
    // slice is reconstructed rather than handed over by the partition map.
    // Grid-stride: at GLM-5's hidden 6144 o_proj wants 48 tiles per XCD
    // against 30 resident workers, and the dispatch width is capped at the
    // worker count because a tile that has to wait for a worker deadlocks the
    // barrier below. Where the count fits the loop runs once.
    //
    // m_tiles is 1, so every pass round this loop has the same `tile_input`
    // and the GEMV's 64 KB LDS staging of it is identical work. Stage on the
    // first pass only. Nothing between the passes writes _fused_smem: the GEMV
    // itself only reads s_a after the staging block, and the router below runs
    // after the barrier.
    //
    // GLM_OPROJ_RESTAGE=1 is the ablation: it re-stages on every pass, i.e.
    // the previous behaviour.
    //
    // Measured, GLM-5 744B, NP=8 EP, 78 layers, MPK_SUBPHASE_TIMING=1, rank 0,
    // aggregate worker-seconds over SP3 cnt 2209800:
    //
    //   SP3 slot          restage  stage-once  delta
    //   [0] OProjCompute   123.71      120.65  -3.06  (-2.5%)
    //   [2] Router          55.35       52.94  -2.41
    //   others                     within +-1.0 (noise)
    //   SP4 (control)      110.89      110.87   0.00
    //
    // End-to-end is flat: SP3[0] is 25% of ~495 worker-s, so -2.5% there is
    // ~0.1 ms, under the run-to-run spread. Three uninstrumented runs, decode
    // device clock: 17.474 / 17.496 / 17.229 ms (mean 17.400) vs 17.284 /
    // 17.550 / 17.412 (mean 17.415). Kept anyway -- it is a strict removal of
    // 128 KB of traffic per second pass with bit-identical output, and it gets
    // less hidden as the barrier waits come down.
#ifdef MPK_GLM_OPROJ_RESTAGE
    constexpr bool RESTAGE = true;
#else
    constexpr bool RESTAGE = false;
#endif
    bool stage_a = true;
    // ── rank-sharded o_proj (OPROJ_TP) ────────────────────────────────────
    // With ATTN_DP at batch 1 every rank holds the same attn_out and computes
    // the same o_proj from the same 103.8 MB of weight -- 8x the bytes for one
    // copy of the answer, and the single largest replicated item in the layer
    // (see the profile note above: o_proj is 26.7% of this task and runs at
    // 76% of HBM peak, so the only lever left on it is bytes).
    //
    // Sharding it output-wise: rank p keeps rows [p * HIDDEN_SIZE/EP, +that)
    // of the weight and computes exactly those columns of the hidden row,
    // residual included. The columns are disjoint across ranks, so completing
    // the row is an all-gather and not a reduction, and every rank's slice is
    // already final -- nothing is added to it afterwards.
    //
    // It is detected, not plumbed. demo.py slices the weight and
    // oproj_tiles_per_xcd falls out of its dim 0, so "my tiles cover a
    // 1/EP-th of the row" is the whole condition, and A/B is a demo.py env
    // flag with no template argument, no new task input and no arity change.
    constexpr int OPROJ_TP_COLS = HIDDEN_SIZE / EP_WORLD_SIZE;
    bool const oproj_tp =
        (EP_WORLD_SIZE > 1) && (ep_signal_ptr != nullptr) &&
        (oproj_tiles_per_xcd * OPROJ_ROWS_PER_WG * 8 == OPROJ_TP_COLS);
    // The output columns and the residual columns are the SAME columns, so both
    // are derived here from one base rather than the output being offset by the
    // kernel and the residual by the input partition map -- which has no rank
    // axis and so cannot express the sharded base at all.
    size_t const oproj_col_base =
        (oproj_tp ? (size_t)EP_MY_PE * OPROJ_TP_COLS : (size_t)0) +
        static_cast<size_t>(xcd_id) * oproj_tiles_per_xcd * OPROJ_ROWS_PER_WG;
    unsigned short *const xcd_out =
        static_cast<unsigned short *>(hidden_ptr) + oproj_col_base;
    unsigned short const *const xcd_res =
        static_cast<unsigned short const *>(oproj_residual_ptr) +
        oproj_col_base;
    // One delta per peer, resolved once. Both the hidden row and the signal
    // array are symmetric-heap objects, so the same heap-wide delta addresses
    // either; that is the property the Phase-9 fold relies on too.
    constexpr int OPROJ_NPEER = (EP_WORLD_SIZE > 1) ? (EP_WORLD_SIZE - 1) : 1;
    int64_t oproj_peer_delta[OPROJ_NPEER];
    bool oproj_all_mapped = oproj_tp;
    if (oproj_tp) {
#pragma unroll
      for (int q = 0; q < OPROJ_NPEER; q++) {
        oproj_peer_delta[q] = 0;
        if (!mpk_shmem_peer_delta((q < EP_MY_PE) ? q : (q + 1),
                                  &oproj_peer_delta[q])) {
          // No direct mapping means no place to push to. Fall back to the
          // replicated form for this run rather than to a staged collective:
          // the weight is already sliced by then, so the row would simply be
          // wrong. This is a fail-loud configuration error, not a fast path.
          oproj_all_mapped = false;
        }
      }
    }
    bool const oproj_push = oproj_tp && oproj_all_mapped;
    // The fold rides the same shard: it is exactly the o_proj column slice
    // that makes each rank's share of the two contractions well defined, so
    // there is no fold without the push. Both pointers null is the opt-out,
    // and the standalone dispatch takes it.
    bool const router_fold = oproj_push && (router_weight_t_ptr != nullptr) &&
                             (router_partials_ptr != nullptr);
    __hip_bfloat16 const *const d_norm_w =
        static_cast<__hip_bfloat16 const *>(norm_weight_ptr);
    __hip_bfloat16 const *const d_router_wt =
        static_cast<__hip_bfloat16 const *>(router_weight_t_ptr);
    // Lines [0 .. 7] are the per-XCD accumulators, one per XCD of this rank;
    // lines [8 .. 8+EP-1] are the per-rank sums, indexed by the PE that wrote
    // them. Only the first eight are touched here.
    float *const xcd_acc =
        static_cast<float *>(router_partials_ptr) +
        (size_t)xcd_id * (NUM_EXPERTS + 1);
    for (int t = xcd_rank; t < oproj_tiles_per_xcd; t += tiles_per_xcd) {
      gang_gemv_mxfp8_kernel<BATCH_SIZE,
                             OPROJ_REDUCTION_SIZE,
                             OPROJ_ROWS_PER_WG,
                             /*HAS_RESIDUAL=*/true,
                             /*WRITE_THROUGH=*/true>(UNABSORB_V
                                                         ? (void const *)
                                                               v_out_ptr
                                                         : oproj_input_ptr,
                                                     oproj_weight_ptr,
                                                     xcd_res,
                                                     xcd_out,
                                                     num_active_tokens,
                                                     OPROJ_ROWS_PER_WG,
                                                     HIDDEN_SIZE,
                                                     /*m_tiles=*/1,
                                                     oproj_tiles_per_xcd,
                                                     /*wgm=*/0,
                                                     t,
                                                     /*bias_ptr=*/nullptr,
                                                     stage_a);
      stage_a = RESTAGE;
      // Push the tile straight into every peer's copy of the hidden row, at
      // the identical offset -- the slices are disjoint, so the all-gather is
      // EP_NPEER stores of the bytes this workgroup just produced and no
      // staging buffer at all. Per worker per layer that is
      // OPROJ_ROWS_PER_WG/2 dwords x 7 links = 56 stores at GLM-5's shape.
      //
      // Pushed here rather than by the elected barrier leader on the whole
      // 1/EP-th row: the leader is one workgroup and would serialize 1.5 KB
      // x 7 behind the last arrival, whereas here the links are loaded by all
      // eight XCDs while the o_proj tiles are still finishing, and the leader
      // is left with nothing to do but the signal and the wait.
      //
      // The read-back has to bypass L1: the GEMV's epilogue is WRITE_THROUGH,
      // i.e. sc0 sc1, so the bytes are in memory but this CU's vL1 may hold
      // the line these lanes read before it. ld_nt_s32 is the sc0 sc1 load,
      // and it carries its own vmcnt drain.
      if (oproj_push) {
        static_assert((OPROJ_ROWS_PER_WG % 2) == 0,
                      "peer stores are packed 32-bit, so a tile must be an "
                      "even number of bf16");
        constexpr int OPROJ_TILE_W32 = OPROJ_ROWS_PER_WG / 2;
        __syncthreads();
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        unsigned int *const src32 = reinterpret_cast<unsigned int *>(
            xcd_out + (size_t)t * OPROJ_ROWS_PER_WG);
        for (int w = tid; w < OPROJ_TILE_W32; w += (int)blockDim.x) {
          unsigned int const v =
              (unsigned int)ld_nt_s32(reinterpret_cast<int *>(src32 + w));
          // Unrolled over peers so oproj_peer_delta stays in registers: a
          // runtime index into a per-thread array is a scratch spill.
#pragma unroll
          for (int q = 0; q < OPROJ_NPEER; q++) {
            st_wt_u32((void *)(reinterpret_cast<char *>(src32 + w) +
                               oproj_peer_delta[q]),
                      v);
          }
        }
      }

      // ── the router fold ─────────────────────────────────────────────────
      // This tile now holds four FINAL hidden columns -- residual included,
      // and nothing is added to a rank's own slice afterwards -- so it can
      // contribute those columns' share of both contractions the router would
      // otherwise run after the barrier:
      //
      //   ssq      += sum_j h_j^2
      //   logit[e] += sum_j h_j * gamma_j * Wt[col_j][e]     for all 256 e
      //
      // irms = rsqrt(ssq/H + eps) is a positive scalar over the whole row, so
      // it factors out of the dot and can be applied once, after the reduce.
      //
      // Thread e owns expert e. Wt's row for one hidden column is 256
      // contiguous bf16, so the 256 threads read one 512-byte line per column
      // and the tile touches 2 KB total -- against the 12 KB row and the
      // 24 KB of gate weight the two Phase-3 steps read per worker.
      //
      // The four h values are read back through ld_nt_s32 for the same reason
      // the peer push does: the GEMV epilogue is WRITE_THROUGH, so this CU's
      // vL1 may hold the pre-store line. Reusing that read is why the fold
      // needs nothing from the GEMV kernel itself.
      if (router_fold) {
        static_assert(!(OPROJ_ROWS_PER_WG & 1),
                      "the fold reads the tile back as packed 32-bit pairs");
        __syncthreads();
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        int const col0 = xcd_id * oproj_tiles_per_xcd * OPROJ_ROWS_PER_WG +
                         t * OPROJ_ROWS_PER_WG;
        // Rank-local column index: Wt is already sliced to this rank, so it is
        // indexed by the offset within the 1/EP-th row, not by oproj_col_base.
        unsigned short const *const hrow =
            reinterpret_cast<unsigned short const *>(xcd_out) +
            (size_t)t * OPROJ_ROWS_PER_WG;
        float hg[OPROJ_ROWS_PER_WG];
        float ssq_tile = 0.0f;
#pragma unroll
        for (int j = 0; j < OPROJ_ROWS_PER_WG; j += 2) {
          unsigned int const pair = (unsigned int)ld_nt_s32(
              reinterpret_cast<int *>(
                  const_cast<unsigned short *>(hrow) + j));
          unsigned short const lo = (unsigned short)(pair & 0xFFFFu);
          unsigned short const hi = (unsigned short)(pair >> 16);
          float const h0 = __bfloat162float(*(bf16 const *)&lo);
          float const h1 = __bfloat162float(*(bf16 const *)&hi);
          ssq_tile += h0 * h0 + h1 * h1;
          size_t const gcol = oproj_col_base + (size_t)t * OPROJ_ROWS_PER_WG;
          hg[j] = h0 * __bfloat162float(d_norm_w[gcol + j]);
          hg[j + 1] = h1 * __bfloat162float(d_norm_w[gcol + j + 1]);
        }
        // gamma is indexed globally -- it is the un-sharded [HIDDEN] vector --
        // while Wt below is indexed rank-locally. The two differ by exactly
        // EP_MY_PE * OPROJ_TP_COLS, which is the whole content of the shard.
        for (int e = tid; e < NUM_EXPERTS; e += (int)blockDim.x) {
          float acc = 0.0f;
#pragma unroll
          for (int j = 0; j < OPROJ_ROWS_PER_WG; j++) {
            acc += hg[j] * __bfloat162float(d_router_wt[(size_t)(col0 + j) *
                                                            NUM_EXPERTS +
                                                        e]);
          }
          // Distinct address per thread; the contention is the 24 tiles of
          // this XCD, and it is L2-local because the accumulator line is the
          // XCD's own.
          atomicAdd(&xcd_acc[e], acc);
        }
        if (tid == 0) {
          atomicAdd(&xcd_acc[NUM_EXPERTS], ssq_tile);
        }
      }
    }

    MPK_WS_PHASE(72, routing_expected, xcd_id);
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
    // The election result is broadcast to the whole block rather than kept in
    // tid 0, because under the fold the elected block has 2056 floats to
    // reduce and 257 to push -- work for 256 threads, not for one. Everything
    // downstream of the election still runs on tid 0 alone.
    __shared__ int s_oproj_leader;
    if (tid == 0) {
      // Modular test rather than a reset: the counter is monotonic for the
      // whole run, so there is no window in which a fast worker from the next
      // layer can observe a zeroed counter.
      //
      // Nothing downstream of this barrier reads the arrival counter -- the
      // wait at MPK_WS_WAIT_BEGIN(765) polls routing_ready's epoch, not the
      // count -- so the tree changes only who does the counting, and there is
      // no heal quota here to keep in step.
      s_oproj_leader =
          hier_barrier_arrive(hier_barrier, HIER_STRIDE,
                              total_barrier_arrivals,
                              total_barrier_arrivals / 8, xcd_id,
                              MPK_BAR_TREE != 0 &&
                                  (total_barrier_arrivals % 8) == 0)
              ? 1
              : 0;
    }
    __syncthreads();
    if (s_oproj_leader) {
      // ── the router fold's rank-local reduce and all-reduce ───────────────
      // Being elected means every one of this rank's arrivals is in, so all
      // eight per-XCD accumulators are final. Sum them into this rank's line
      // and push that line to the peers, ahead of the signal below -- the
      // signal's release then covers the logits exactly as it covers the
      // hidden-row slices, and the fold costs no rendezvous of its own.
      //
      // 257 floats x 7 links = 7.2 KB per rank per layer, against the 1.38 MB
      // it would be if all 192 contributing workers pushed their own partials
      // and skipped this reduce.
      if (router_fold) {
        constexpr int RF_LINE = NUM_EXPERTS + 1;
        float *const parts = static_cast<float *>(router_partials_ptr);
        // Parity-indexed so a peer that reaches its next layer's push before
        // this rank's Phase 3 has read the line cannot overwrite it. The
        // hidden row itself is not double-buffered and relies on the peer
        // being a whole Phase 3-7 behind; a 257-float line is cheap enough
        // not to have to make that argument again.
        float *const my_line =
            parts + (size_t)(8 + (oproj_expected & 1) * EP_WORLD_SIZE +
                             EP_MY_PE) *
                        RF_LINE;
        for (int e = tid; e < RF_LINE; e += (int)blockDim.x) {
          float s = 0.0f;
#pragma unroll
          for (int x = 0; x < 8; x++) {
            s += parts[(size_t)x * RF_LINE + e];
          }
          my_line[e] = s;
          // Clear for the next layer. Safe here and only here: no worker on
          // this rank is released until the fan-out below, so none can be in
          // the next layer's Phase 1 adding to these lines yet.
#pragma unroll
          for (int x = 0; x < 8; x++) {
            parts[(size_t)x * RF_LINE + e] = 0.0f;
          }
        }
        __syncthreads();
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        for (int e = tid; e < RF_LINE; e += (int)blockDim.x) {
          float const v = my_line[e];
#pragma unroll
          for (int q = 0; q < OPROJ_NPEER; q++) {
            st_wt_u32((void *)(reinterpret_cast<char *>(my_line + e) +
                               oproj_peer_delta[q]),
                      __float_as_uint(v));
          }
        }
        __syncthreads();
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      }
      if (tid == 0) {
        // ── the all-gather's rendezvous rides this barrier ────────────────
        // Under OPROJ_TP the row is not complete when the local arrivals are
        // in; it is complete when every peer's slice has landed too. This
        // thread is the one the modular test elected, so it is the one that
        // has observed all eight local XCDs -- which is exactly the condition
        // for telling the peers, and it is already the thread that fans the
        // release out. Folding the peer wait in here means ONE thread on the
        // rank polls the remote lines (which must be sc0 sc1, L2-bypassing --
        // see ld_sys_u64) and the other 231 workers keep polling a local flag
        // that is now simply published later. No second barrier, no change to
        // the router-side wait, and the router's self-heal stays sound: it
        // republishes only from a flag that some other XCD already holds, and
        // all eight are written below, after this wait.
        if (oproj_push) {
          unsigned long long *const ep_sig =
              static_cast<unsigned long long *>(ep_signal_ptr);
          unsigned long long *const my_line =
              ep_sig + (size_t)EP_MY_PE * OPROJ_EP_SIGNAL_STRIDE +
              OPROJ_EP_SIGNAL_SLOT;
          // All EP_NPEER stores back to back, then one drain: distinct peers
          // are distinct XGMI links and pipeline.
#pragma unroll
          for (int q = 0; q < OPROJ_NPEER; q++) {
            st_wt_u64((void *)(reinterpret_cast<char *>(my_line) +
                               oproj_peer_delta[q]),
                      (unsigned long long)oproj_expected);
          }
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          // Poll all peers concurrently off one bitmask rather than in rank
          // order, so a slow link costs its own latency and not the sum.
          unsigned remaining = (1u << OPROJ_NPEER) - 1u;
          while (remaining) {
#pragma unroll
            for (int q = 0; q < OPROJ_NPEER; q++) {
              if (remaining & (1u << q)) {
                int const p = (q < EP_MY_PE) ? q : (q + 1);
                if (ld_sys_u64(ep_sig + (size_t)p * OPROJ_EP_SIGNAL_STRIDE +
                               OPROJ_EP_SIGNAL_SLOT) >=
                    (unsigned long long)oproj_expected) {
                  remaining &= ~(1u << q);
                }
              }
            }
            if (remaining) {
              __builtin_amdgcn_s_sleep(1);
            }
          }
          // The peer slices arrived as sc0 sc1 stores, so they are in memory
          // and not in any cache this rank can see stale. Drop vL1 anyway
          // before the release: the router re-reads the whole row and its own
          // buffer_inv is downstream of a flag this thread has not written
          // yet, which is the wrong order to rely on.
          asm volatile("buffer_inv" ::: "memory");
        }
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

    MPK_WS_PHASE(73, routing_expected, xcd_id);
    // ══════════════════════════════════════════════════════════════════════
    // Phase 3: RMSNorm + router GEMV + sigmoid/bias TopK
    // ══════════════════════════════════════════════════════════════════════
    // One worker per expert, each redundantly re-norming the row; the kernel's
    // own atomic counter picks the last of the 64 to run the TopK tail.
    // Workers past router_tile_n must not enter, or that counter would
    // over-count. That tail publishes `routing_ready` for the MoE phases.
    // Grid-stride over router TILES, each carrying ROUTER_EXPERTS_PER_TILE
    // experts. One worker per expert (EPT 1) is the shape this was written
    // for, but 256 experts is 32 per XCD against 29 workers, so a second round
    // ran for a mean of 1.10 -- and that second call re-paid the barrier spin,
    // the redundant RMSNorm, two block-wide reductions and the arrival atomic
    // to add one dot product. At EPT 2 the tiles are 16 per XCD and the round
    // count is one; the gang counter sees exactly total_router_tiles arrivals
    // per layer however they are distributed, so the TopK tail still fires
    // exactly once either way.
    //
    // The LDS irms cache below is what made EPT 1's second round survivable
    // and is kept: it costs nothing at EPT 2 (one call, one norm) and still
    // covers any geometry where the tiles outnumber the workers.
    // Reset per layer: `hidden` is a different row each time.
    static_assert(NUM_EXPERTS % (8 * ROUTER_EXPERTS_PER_TILE) == 0,
                  "the router's experts split evenly over the XCDs and then "
                  "over a tile; router_tile_n is that quotient");
    __shared__ float s_router_irms;
    if (tid == 0) {
      s_router_irms = -1.0f;
    }
    __syncthreads();
    for (int t = xcd_rank; t < router_tile_n; t += tiles_per_xcd) {
      gang_rmsnorm_linear_bias_topk_kernel<__hip_bfloat16,
                                           BATCH_SIZE,
                                           HIDDEN_SIZE,
                                           ACTUAL_HIDDEN_DIM,
                                           NUM_EXPERTS,
                                           TOPK_K,
                                           /*SIGMOID_BIAS=*/true,
                                           /*OPROJ_BARRIER=*/true,
                                           ROUTER_EXPERTS_PER_TILE>(
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
          t,
          total_router_tiles,
          renormalize,
          routed_scaling_factor,
          num_shared_experts,
          hier_barrier,
          xcd_id,
          oproj_expected,
          routing_ready,
          /*routing_epoch_hint=*/routing_expected,
          /*irms_cache=*/&s_router_irms,
          // Under the fold both contractions are already done and reduced
          // across the ranks; the router's job shrinks to scaling by irms,
          // and it is handed the parity block's line 0 plus the offset of
          // this XCD's expert range inside a line.
          /*folded_lines=*/
          router_fold ? (static_cast<float const *>(router_partials_ptr) +
                         (size_t)(8 + (oproj_expected & 1) * EP_WORLD_SIZE) *
                             (NUM_EXPERTS + 1))
                      : nullptr,
          /*folded_ranks=*/router_fold ? EP_WORLD_SIZE : 0,
          /*folded_stride=*/NUM_EXPERTS + 1,
          /*folded_expert_base=*/xcd_id * (NUM_EXPERTS / 8));
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

  MPK_WS_PHASE(74, routing_expected, xcd_id);
  // ════════════════════════════════════════════════════════════════════════
  // Phase 4: wait for routing
  // ════════════════════════════════════════════════════════════════════════
  // Every worker, including the ones that just ran the router -- for the block
  // that ran the TopK tail this poll succeeds on its first load. Separate from
  // the o_proj barrier so that the MoE-only workers, which never arrived
  // there, still have something to wait on.
  if (tid == 0) {
    int *my_flag = &routing_ready[(1 + xcd_id) * HIER_STRIDE];
    // Self-heal, see MPK_FL_REPUBLISH_SPINS in
    // gang_mla_full_layer_fused_mi300.cuh. Unlike the barriers there this one
    // has no arrival counter -- but it does not need one. The TopK tail
    // (gang_rmsnorm_linear_bias_mi300.cuh) publishes slot 0 and then the eight
    // per-XCD lines from the same thread in the same loop, so slot 0 at the
    // epoch proves the release fired and this XCD's line is merely short.
    MPK_WS_WAIT_BEGIN(765, routing_expected);
    int _spins = 0;
    int _obs;
    while ((_obs = ld_nt_s32(my_flag)) < routing_expected) {
      ++_spins;
      MPK_WS_WAIT_TICK(_obs, _spins);
      if ((_spins & (MPK_FL_REPUBLISH_SPINS - 1)) == 0) {
        if (ld_nt_s32(routing_ready) >= routing_expected) {
          st_wt_u32((void *)my_flag, (unsigned)routing_expected);
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        }
      }
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

  MPK_WS_PHASE(76, routing_expected, xcd_id);
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
                                     MOE_WEIGHT_FP4,
                                     EP_WORLD_SIZE,
                                     EP_MY_PE,
                                     /*EP_NUM_ROUTED=*/NUM_EXPERTS,
                                     EP_SHARED_PE>(
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

  MPK_WS_PHASE(77, routing_expected, xcd_id);
  // ════════════════════════════════════════════════════════════════════════
  // Phase 6: W13 -> W2 barrier
  // ════════════════════════════════════════════════════════════════════════
  // Every worker arrives; only the W2 workers wait. The ones past
  // moe_w2_tiles_per_xcd return to the scheduler a whole W2 stage early
  // instead of spinning, which is the same trade Phase 2 makes for o_proj.
  __syncthreads();
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  if (tid == 0) {
    int const arrivals = tiles_per_xcd * 8;
    // MPK_W13_EARLY_REL needs the raw modular position, not just "am I last",
    // so this site keeps the flat arrival whenever the ceiling probe is on.
#ifdef MPK_W13_EARLY_REL
    int prev = atom_add_release_gpu_s32(&w13_barrier[8 * HIER_STRIDE], 1);
#else
    bool const _w13_owes =
        hier_barrier_arrive(w13_barrier, HIER_STRIDE, arrivals, tiles_per_xcd,
                            xcd_id, MPK_BAR_TREE != 0);
#endif
    // MPK_W13_EARLY_REL: fire the release at FRAC/16 of the arrivals instead of
    // all of them. WRONG OUTPUT by construction -- W2 reads swiglu columns
    // whose producers have not run. Ported from gpt-oss
    // (gang_moe_fused_mxfp4_mi300.cuh:1731) for the same purpose: price the
    // ceiling of every "narrow the dependency" scheme before paying for the
    // index surgery. If releasing at half the arrivals does not move the token,
    // the wait is upstream arrival SPREAD, not the barrier's count, and no
    // per-expert / split-K narrowing of the count can help.
#ifdef MPK_W13_EARLY_REL
    int const _rel_at_raw = (arrivals * MPK_W13_EARLY_REL) / 16;
    int const _rel_at = (_rel_at_raw < 1) ? 1 : _rel_at_raw;
    if ((prev % arrivals) == _rel_at - 1) {
#else
    if (_w13_owes) {
#endif
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
    // Self-heal, see MPK_FL_REPUBLISH_SPINS in
    // gang_mla_full_layer_fused_mi300.cuh. The counter just above is the
    // truth and advances by exactly `arrivals` per layer, so at or past
    // arrivals * w13_expected the release is owed and republishing this XCD's
    // line is idempotent -- same monotonic absolute value.
    int const arrivals = tiles_per_xcd * 8;
    MPK_WS_WAIT_BEGIN(766, w13_expected);
    int _spins = 0;
    int _obs;
    while ((_obs = ld_nt_s32(my_flag)) < w13_expected) {
      ++_spins;
      MPK_WS_WAIT_TICK(_obs, _spins);
      if ((_spins & (MPK_FL_REPUBLISH_SPINS - 1)) == 0) {
        // Quota is 8 under the tree -- the global counter is bumped once per
        // XCD -- except when MPK_W13_EARLY_REL forced the flat arrival above.
        bool const _w13_tree =
            MPK_BAR_TREE != 0 && !MPK_W13_EARLY_REL_ON;
        if (ld_nt_s32(&w13_barrier[8 * HIER_STRIDE]) >=
            hier_barrier_heal_quota(arrivals, _w13_tree) * w13_expected) {
          st_wt_u32((void *)my_flag, (unsigned)w13_expected);
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        }
      }
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

  MPK_WS_PHASE(78, routing_expected, xcd_id);
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
                                    MOE_WEIGHT_FP4,
                                    EP_WORLD_SIZE,
                                    EP_MY_PE,
                                    /*EP_NUM_ROUTED=*/NUM_EXPERTS,
                                    EP_SHARED_PE,
                                    MPK_W2_KSPLIT>(
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
