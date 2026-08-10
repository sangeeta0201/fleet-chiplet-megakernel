/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

// Full-layer fused gang task for MI300/MI350.
//
// Combines task 214 (QKV+Attn) and task 215 (O-proj+TopK+MoE) into ONE
// gang task per transformer decoder layer, eliminating one inter-task
// event barrier and one scheduler dispatch per layer.
//
// 8-phase pipeline (single dispatch, all 240 workers stay alive):
//   Phase 1: QKV GEMM          — workers 0..(qkv_tiles_per_xcd-1) per XCD
//   Phase 2: QKV barrier       — epoch-based, all QKV workers wait
//   Phase 3: Parallel attention — workers 0..(NUM_KV_CHUNKS-1) each run one
//   chunk Phase 4: Chunk barrier      — last chunk worker runs merge inline
//   Phase 5: Merge + flush     — merge_splitkv_ck_fmha → bf16, write-through,
//   signal Phase 6: Cross-XCD barrier — all 30 workers poll attn_global Phase
//   7: O-proj + RMSNorm + Router + TopK Phase 8: MoE (W13+SwiGLU+W2)
//
//   Phase 9: EP combine       — expert-parallel only; see below
//
// Counter buffer slot map (extends type 215's layout):
//   oproj_counters_ptr + 0..18*16-1  : type 215's counters
//   oproj_counters_ptr + 19*16       : attn_global_counter (cross-XCD sync)
//   oproj_counters_ptr + 20*16       : qkv_epoch[0..7] per-XCD epoch flags
//   oproj_counters_ptr + 28*16       : chunk_barrier[0..7] per-XCD chunk
//   arrival
//   oproj_counters_ptr + 36*16       : attn_xcd_release[0..7]
//   oproj_counters_ptr + 768         : ep_moe_done   (EP: global MoE arrival)
//   oproj_counters_ptr + 784 (+4 ea) : ep_combine_release[0..7] (EP)
//   oproj_counters_ptr + 816 (+4 ea) : ep_fold_done[0..7]       (EP)
//
// Stack-frame optimization: input/output pointers are NOT unpacked into
// local variables. Instead, input_ptrs[N] and output_ptrs[N] are accessed
// directly, saving ~272 bytes of stack frame per thread.

#pragma once
#include "comm/mpk_comm.cuh"
#include "tasks/mi300/gang_linear_mxfp4_res_bias_rmsnorm_topk_mi300.cuh"
#include "tasks/mi300/gang_moe_fused_mxfp4_mi300.cuh"
#include "tasks/mi300/gang_rmsnorm_linear_mxfp4_bias_mi300.cuh"
#include "tasks/mi300/paged_attention_ck_fmha_split_kv_mi300.cuh"

namespace kernel {

static constexpr int FULL_LAYER_ATTN_GLOBAL_COUNTER_SLOT = 19 * 16;
static constexpr int FULL_LAYER_QKV_EPOCH_SLOT = 20 * 16;
static constexpr int FULL_LAYER_CHUNK_BARRIER_SLOT = 28 * 16;
static constexpr int FULL_LAYER_ATTN_XCD_RELEASE_SLOT = 36 * 16;
// Phase 9 (expert-parallel combine) barrier slots. Same 16-int (64-byte)
// per-XCD spacing as every other barrier here, so no two XCDs share a line.
// These push the counter buffer past the 832 ints the non-EP path allocates;
// demo.py sizes it to EP_COUNTER_SIZE when the inline combine is on.
static constexpr int FULL_LAYER_EP_MOE_DONE_SLOT = 48 * 16;      // 768
static constexpr int FULL_LAYER_EP_RELEASE_SLOT = 49 * 16;       // 784..896
static constexpr int FULL_LAYER_EP_FOLD_DONE_SLOT = 58 * 16;     // 928..1040
// Nine lines, not eight: [0] is the accumulator the 8 slice-reducers count into
// and [1+x] is XCD x's release flag. The reduce is sliced across XCDs, so an
// XCD's own completion no longer means the residual is whole.
static constexpr int FULL_LAYER_EP_COMBINE_DONE_SLOT = 67 * 16;  // 1072..1216
// Per-XCD MoE arrival counters for the two-level 9a tree. One 64-byte line per
// XCD, so the 30 workers of an XCD contend only with each other.
static constexpr int FULL_LAYER_EP_XCD_ARRIVE_SLOT = 77 * 16;    // 1232..1344
static constexpr int FULL_LAYER_EP_XCD_STRIDE = 16;
// Highest int the EP phase touches, + one slot of headroom. Kept next to the
// slots so the two cannot drift apart; demo.py asserts against it indirectly
// by allocating this many ints.
static constexpr int FULL_LAYER_EP_COUNTER_SIZE = 86 * 16;       // 1376
// Stride between per-PE signal slots, in uint64. One 64-byte line each so a
// peer's SIGNAL_ADD never shares a line with another peer's.
static constexpr int FULL_LAYER_EP_SIGNAL_STRIDE = 8;

// Hoisted from the Phase 9 block: MPK_EP_WAIT_AT_USE's wait sits in Phase 1,
// upstream of where this used to be defined, and must not compile in under an
// ablation that suppresses the peer signal it waits on.
#ifndef MPK_EP_ABLATE
#define MPK_EP_ABLATE 0
#endif

// Move Phase 9d's peer wait to its point of use in the next layer's QKV
// prologue. Both placements are correct -- see the long comment at the Phase 1
// site -- so this is a pure scheduling knob, not an ablation. Forced off under
// any ablation that drops the peer store (1, 5) or Phase 9 as a whole (2):
// with nothing signalling, a deferred wait would spin forever.
#ifndef MPK_EP_WAIT_AT_USE
#define MPK_EP_WAIT_AT_USE 0
#endif
#if MPK_EP_ABLATE != 0
#undef MPK_EP_WAIT_AT_USE
#define MPK_EP_WAIT_AT_USE 0
#endif

// Fold this rank's MoE partial out of the f32 workspace into bf16, optionally
// adding the residual, and zero the workspace.
//
// This is moe_residual_add_f32_mi300_impl's job, but that kernel always adds a
// residual, so the non-folding ranks would need a zero buffer plumbed through
// the graph purely to be added to nothing. FOLD is a template parameter here
// instead, so the add compiles away on the ranks that must not perform it and
// no zero_residual tensor is needed.
//
// Zeroing is not incidental: the next layer's QKV prologue computes
// x = workspace_f32 + residual, and under EP the complete sum is handed to it
// as `residual`. The workspace must therefore read as zero, or this layer's
// partial would be counted twice.
// peer_out, when non-null, is this same slot's address in the PEER's address
// space (mpk_shmem_peer_ptr). Each thread stores the value it just computed to
// both places, so the cross-GPU transfer is the fold's own epilogue: no staging
// buffer to re-read, no DMA descriptor, no separate transfer step to wait on.
// The remote store is write-through (sc0 sc1) because the peer must observe it
// without a flush of this rank's L2 -- the same reason every cross-XCD release
// flag here is st_wt_u32.
// col_lo/col_hi restrict the fold to a column slice, so the 8 XCD workgroups
// that are awake anyway can each take 1/8 of the work instead of one workgroup
// doing all of it while 239 workers idle. The slices are disjoint, so no
// coordination is needed between them beyond the arrival count that follows.
template <int BATCH_SIZE, int OUTPUT_SIZE, int OUTPUT_STRIDE, bool FOLD>
__device__ __forceinline__ void _full_layer_ep_fold_partial(
    void *workspace_f32_ptr, void const *residual_ptr, void *output_ptr,
    void *peer_out_ptr = nullptr, int col_lo = 0, int col_hi = OUTPUT_SIZE) {
  float *__restrict__ d_ws = static_cast<float *>(workspace_f32_ptr);
  unsigned short const *__restrict__ d_res =
      static_cast<unsigned short const *>(residual_ptr);
  unsigned short *__restrict__ d_out =
      static_cast<unsigned short *>(output_ptr);
  unsigned short *d_peer = static_cast<unsigned short *>(peer_out_ptr);

  for (int row = 0; row < BATCH_SIZE; ++row) {
    float *ws_row = d_ws + row * OUTPUT_STRIDE;
    unsigned short const *res_row = d_res + row * OUTPUT_STRIDE;
    unsigned short *out_row = d_out + row * OUTPUT_STRIDE;
    unsigned short *peer_row = d_peer ? d_peer + row * OUTPUT_STRIDE : nullptr;
    // Two bf16 per thread per step, so the remote store is a 32-bit write
    // rather than two 16-bit ones. A 2-byte store still occupies a full XGMI
    // transaction, so the narrow form would double the packet count for the
    // same payload. OUTPUT_SIZE is the padded hidden dim (even); the odd tail
    // below exists only so the helper stays correct for any width.
    int const pair_lo = col_lo >> 1;
    int const pair_hi = col_hi >> 1;
    for (int p = pair_lo + threadIdx.x; p < pair_hi; p += blockDim.x) {
      int off = p << 1;
      unsigned packed = 0;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        float v = ws_row[off + j];
        if constexpr (FOLD) {
          unsigned rbits = (unsigned)res_row[off + j] << 16;
          float rv;
          __builtin_memcpy(&rv, &rbits, 4);
          v += rv;
        }
        unsigned u;
        __builtin_memcpy(&u, &v, 4);
        unsigned rounding_bias = ((u >> 16) & 1) + 0x7FFFu;
        unsigned short bf = (unsigned short)((u + rounding_bias) >> 16);
        packed |= ((unsigned)bf) << (16 * j);
      }
      // The LOCAL slot is written write-through too, as one 32-bit store rather
      // than two 16-bit ones -- `packed` is already assembled for the peer.
      //
      // The point is not the store width, it is what write-through lets the
      // caller drop. This slot's consumer is the next layer's QKV prologue on
      // every XCD, so a plain store needs an agent-scope release to become
      // visible, and threadfence_gpu() on gfx950 is `buffer_wbl2 sc1` -- an
      // L2->HBM writeback of the WHOLE cache, run by all 8 folding workgroups
      // on all 36 layers. st_wt puts these bytes past L2 on the store itself,
      // which is the only data the fence was there to publish (the workspace
      // zeroing below is read back by this same XCD's Phase 8 atomicAdds, and
      // those go through L2 anyway).
      st_wt_u32((void *)&out_row[off], packed);
      // The zeroing has to be write-through for the same reason, and it is the
      // one that makes dropping the fence sound. Its consumer is the next
      // layer's Phase 8 atomicAdd, and a W2 tile writing these columns can land
      // on ANY XCD -- the tile->column map is by wg_idx, not by XCD -- so a
      // plain store into this XCD's L2 is not enough. Two floats, one 8-byte
      // store, the same pair the fold just read.
      st_wt_u64((void *)&ws_row[off], 0ull);
      if (peer_row) {
        st_wt_u32((void *)&peer_row[off], packed);
      }
    }
    if ((OUTPUT_SIZE & 1) && threadIdx.x == 0 && col_hi == OUTPUT_SIZE) {
      int off = OUTPUT_SIZE - 1;
      float v = ws_row[off];
      if constexpr (FOLD) {
        unsigned rbits = (unsigned)res_row[off] << 16;
        float rv;
        __builtin_memcpy(&rv, &rbits, 4);
        v += rv;
      }
      unsigned u;
      __builtin_memcpy(&u, &v, 4);
      unsigned rounding_bias = ((u >> 16) & 1) + 0x7FFFu;
      unsigned short bf = (unsigned short)((u + rounding_bias) >> 16);
      st_wt_u16((void *)&out_row[off], bf);
      st_wt_u32((void *)&ws_row[off], 0u);
      if (peer_row) {
        st_wt_u16((void *)&peer_row[off], bf);
      }
    }
  }
}

// Wait until every OTHER PE's slot of the symmetric gather buffer has landed
// for this layer. Single-threaded; the caller decides which thread runs it.
//
// At world size 2 this is one plain non-temporal load in a loop rather than a
// call into rocSHMEM: the direct peer store and rocSHMEM's SIGNAL_ADD land at
// the same symmetric address, and ld_nt_u64 observes it past the local cache
// hierarchy either way, which was the only reason the backend primitive was
// needed. Wider worlds keep the staged path.
//
// Many workers may poll the same address concurrently and that is fine: a load
// is not a coherence transaction the way an atomic is, and the line is
// read-only until its one writer touches it once per layer. The 240-deep
// arrival tree in 9a exists because atomicAdds to one address serialize; reads
// to one address do not.
template <int EP_WORLD_SIZE, int EP_MY_PE>
__device__ __forceinline__ void
    _full_layer_ep_wait_peers(uint64_t *ep_signal, uint64_t ep_sig_expected) {
  if constexpr (EP_WORLD_SIZE == 2) {
    uint64_t *peer_sig =
        ep_signal + (size_t)(1 - EP_MY_PE) * FULL_LAYER_EP_SIGNAL_STRIDE;
    while (ld_nt_u64(reinterpret_cast<unsigned long long *>(peer_sig)) <
           (unsigned long long)ep_sig_expected) {
      __builtin_amdgcn_s_sleep(1);
    }
  } else {
    // Staged fallback: one transfer per peer carries the whole slot and
    // signals slot 0 of p's line, so every waiter watches that one slot.
    for (int p = 0; p < EP_WORLD_SIZE; p++) {
      if (p == EP_MY_PE) {
        continue;
      }
      mpk_shmem_signal_wait_ge(
          ep_signal + (size_t)p * FULL_LAYER_EP_SIGNAL_STRIDE,
          ep_sig_expected);
    }
  }
}

template <int QKV_BATCH_SIZE,
          int QKV_OUTPUT_PER_WG,
          int QKV_REDUCTION_SIZE,
          int ACTUAL_HIDDEN_DIM,
          int HEAD_DIM,
          int NUM_Q_PER_KV,
          int PAGE_SIZE,
          int MAX_SEQ_LEN,
          int NUM_KV_CHUNKS,
          int Q_WORKSPACE_STRIDE,
          int KV_CACHE_STRIDE,
          int NUM_KV_HEADS,
          int SLIDING_WINDOW,
          int HAS_SINKS,
          int OPROJ_OUTPUT_PER_WG,
          int OPROJ_REDUCTION_SIZE,
          int NUM_EXPERTS,
          int TOPK_K,
          int MOE_INTERMEDIATE_SIZE,
          int MOE_HIDDEN_SIZE,
          int MOE_W13_OUTPUT_PER_WG,
          int MOE_W2_OUTPUT_PER_WG,
          bool DECODE_ONLY = false,
          // Expert-parallel ownership. This rank computes only experts in
          // [MOE_EXPERT_BASE, MOE_EXPERT_BASE + MOE_NUM_LOCAL_EXPERTS) and its
          // weight/bias arrays hold only those, indexed by local id. Defaults
          // are the single-GPU / replicated-MoE identity, so every existing
          // instantiation is unchanged.
          int MOE_EXPERT_BASE = 0,
          int MOE_NUM_LOCAL_EXPERTS = NUM_EXPERTS,
          // Inline expert-parallel combine (Phase 9). EP_WORLD_SIZE == 1 is
          // the single-GPU / replicated-MoE identity: Phase 9 compiles away
          // entirely, so every existing instantiation is unchanged and pays
          // nothing. EP_MY_PE is this rank; EP_FOLD_PE is the one rank that
          // folds the real residual into its partial, so that after the SUM
          // the residual appears exactly once.
          int EP_WORLD_SIZE = 1,
          int EP_MY_PE = 0,
          int EP_FOLD_PE = 0,
          // Slot-parallel expert split (see gang_moe_fused_mxfp4_mi300.cuh).
          // Replaces the id-range split when > 1; requires replicated weights.
          int EP_SLOT_WS = 1,
          int EP_SLOT_ME = 0,
          // The EP reduction, dissolved into this layer's QKV prologue.
          //
          // > 1 means input_ptrs[26] is the PREVIOUS layer's symmetric gather
          // buffer -- [EP_PREV_SLOTS, batch, QKV_REDUCTION_SIZE], one slot per
          // rank holding that rank's partial -- and Phase 1 reads it in place
          // of input_ptrs[1], summing the slots in the same pass it already
          // makes over that vector to compute the sum of squares. There is no
          // separate reduce and no barrier around one.
          //
          // Layer 0 has no predecessor and so passes 1: its residual is the
          // embedding, an ordinary 2-D tensor in input_ptrs[1].
          int EP_PREV_SLOTS = 1,
          // Whether this layer's Phase 9 also materializes the sum into
          // output_ptrs[11]. Only the LAST layer needs it, because the tail
          // (final RMSNorm + LM head) is a separate task that reads a plain 2-D
          // tensor and has no slot-summing prologue to fold the reduce into.
          // Every other layer leaves the exchange as the last thing it does.
          bool EP_WRITE_COMBINED = false>
__device__ __noinline__ void
    gang_full_layer_fused_kernel_mi300(void *const *input_ptrs,
                                       void *const *output_ptrs,
                                       void const *cos_ptr,
                                       void const *sin_ptr,
                                       int const *qo_indptr,
                                       int const *kv_indptr,
                                       int const *kv_indices,
                                       int const *kv_last_page_len,
                                       int num_active_tokens,
                                       int qkv_n_wgs_per_xcd,
                                       int kv_stride,
                                       int q_ws_stride,
                                       float attn_scale,
                                       int total_qkv_tiles_per_xcd,
                                       int oproj_n_wgs_per_xcd,
                                       int oproj_output_stride,
                                       int router_tile_n,
                                       int total_oproj_tiles,
                                       int total_topk_tiles,
                                       int oproj_tiles_per_xcd,
                                       int moe_total_tiles_per_xcd,
                                       int workers_per_xcd,
                                       int tile_idx,
                                       int task_layer_idx) {
  // (All phases enabled — buffer_inv before Phase 6 poll fixes L2 stale read)
  // input_ptrs layout:
  //  [0] workspace_f32    [1] residual          [2] norm_weight_pre
  //  [3] norm_scratch_pre [4] qkv_weight        [5] qkv_bias
  //  [6] attn_sinks       [7] qkv_barrier       [8] lse_acc
  //  [9] oproj_weight     [10] oproj_bias        [11] norm_weight_post
  //  [12] norm_scratch_post [13] router_weight   [14] router_bias
  //  [15] logits_scratch   [16] oproj_counters   [17] moe_gate_up_weight
  //  [18] moe_down_weight  [19] moe_w13_bias     [20] moe_w2_bias
  //  [21] moe_barrier      [22] moe_swiglu_out   [23] o_acc_f32
  //  [24] ep_gather        [25] ep_signal   (EP_WORLD_SIZE > 1 only; both are
  //                                          symmetric-heap allocations)
  //
  // output_ptrs layout:
  //  [0] x_output         [1] k_cache           [2] v_cache
  //  [3] q_workspace      [4] o_acc             [5] attn_proj_out
  //  [6] topk_weight      [7] routing_indices   [8] active_expert_ids
  //  [9] moe_routing_weight [10] moe_workspace_f32
  //  [11] ep_combined     (EP_WORLD_SIZE > 1 only)

  int xcd_id;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcd_id));

  int xcd_rank = tile_idx % workers_per_xcd;
  int tid = threadIdx.x;

  // Invalidate vL1 to ensure we read fresh MoE atomicAdd results from L2.
  // Without this, stale zeros from QKV's workspace zeroing (flat_store in
  // previous iteration) can persist in vL1 across gang task boundaries.
  asm volatile("buffer_inv" ::: "memory");

  // NOTE: the layer counter that the MoE W13->W2 barrier derives its release
  // value from is published further down, once qkv_epoch_expected is known.
  // See the LAYER_IDX_SMEM_OFF store just before Phase 1.

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  unsigned long long _fused_t0 = __builtin_amdgcn_s_memrealtime();
  unsigned long long _fused_t0a = 0, _fused_t0b = 0, _fused_t0c = 0,
                     _fused_t0d = 0, _merge_done = 0;
  // Phase 9 (inline EP combine) sub-timestamps. Taken on a CORRECT run --
  // this is what prices the collective without the ablations' routing drift.
  //   _ep_t0 entry (MoE done, this worker)  _ep_t1 after 9a GPU-wide barrier
  //   _ep_t2 after 9b/9c fold+put           _ep_t3 after 9d wait+reduce
  //   _ep_t4 after 9e exit barrier
  unsigned long long _ep_t0 = 0, _ep_t1 = 0, _ep_t2 = 0, _ep_t3 = 0,
                     _ep_t4 = 0;
#endif

#ifdef MPK_ENABLE_MOE_SUBPHASE
  // Activate MoE subphase timing on first decode iteration (nat <= 1).
  if (tid == 0 && num_active_tokens <= 1) {
    int old = atomicCAS(&g_subphase_active, 0, 1);
    if (old == 0) {
      for (int s = 4; s <= 5; s++) {
        for (int p = 0; p < SUBPHASE_MAX_PHASES; p++) {
          g_subphase_ns[s][p] = 0;
        }
        g_subphase_cnt[s] = 0;
      }
      __threadfence();
    }
  }
#endif

  MPK_TW_SUB(1, tile_idx);

  int *oproj_counters_base = static_cast<int *>(input_ptrs[16]);
  int *attn_global = oproj_counters_base + FULL_LAYER_ATTN_GLOBAL_COUNTER_SLOT;
  int *qkv_epoch = oproj_counters_base + FULL_LAYER_QKV_EPOCH_SLOT;
  int *chunk_barrier = oproj_counters_base + FULL_LAYER_CHUNK_BARRIER_SLOT;
  int *routing_ready = oproj_counters_base + 10 * 16;
  int *attn_release = oproj_counters_base + FULL_LAYER_ATTN_XCD_RELEASE_SLOT;

  // Barrier release values, derived from the layer counter rather than read.
  //
  // These used to be snapshots of "current value + 1". That is only correct if
  // this worker reads *before* this layer's producer bumps the counter, and
  // nothing guaranteed it: all 36 layers run inside a single task (see the ml
  // loop in persistent_kernel.cuh) with only a per-block __syncthreads between
  // them, so workers skew freely across layer boundaries. A worker that fell a
  // full layer behind could read an already-bumped counter and then wait for a
  // bump this layer never produces -- an intermittent deadlock, because it
  // needs a full-layer skew to happen.
  //
  // f1fa720 fixed that by forcing *every* worker on the XCD to arrive at the
  // Phase 2 barrier, which ordered each read before every producer. It worked,
  // and it also cost 2.19 -> 2.46 ms/iter. 2c1071c recovered most of that by
  // splitting arrival from waiting (all 30 arrive, only ranks < NUM_KV_CHUNKS
  // block), reaching 2.39. The remaining ~0.2ms was *not* recovered by this
  // change and is still unattributed -- device timing under MPK_DEVICE_TIMING=1
  // cannot localize it, because the ~147k printfs inflate iterations to ~56ms
  // and the skew lands in Phase 6's xcd_barrier, whose median then moves
  // opposite to real latency. This change is a correctness fix; treat its
  // latency effect as neutral.
  //
  // The layer counter removes the race at its source. All three counters start
  // at 0, bump exactly once per layer, and are never reset, so the value this
  // layer drives them to is a pure function of the layer index -- no shared
  // read, nothing to order, and no arrival requirement. See the
  // _linear_reserved store in the ml loop for how it is published.
  //
  //   qkv_epoch[x]      bumped once by the last worker to arrive on XCD x
  //   attn_release[x]   written by the last XCD to reach the attn_global
  //                     barrier, using its *own* expected value -- so producer
  //                     and consumer now agree by construction
  //   routing_ready[*]  read-modify-written once by the single TopK completer
  //
  // Because they are all the same per-layer count, all three expected values
  // are the same number.
  int const layer_counter = task_layer_idx;
  int const routing_expected = layer_counter + 1;
  int const attn_release_expected = layer_counter + 1;
  int const qkv_epoch_expected = layer_counter + 1;

  // Only the workers that read this layer's QKV output take part in the epoch
  // barrier. The expected values above no longer depend on arrival ordering,
  // so the participant set is free to be the set that actually needs the
  // barrier. This is the pre-f1fa720 participant set, now safe to use because
  // nothing reads a shared counter -- it measured neutral, not faster.
  int const qkv_epoch_participants = total_qkv_tiles_per_xcd > NUM_KV_CHUNKS
                                         ? total_qkv_tiles_per_xcd
                                         : NUM_KV_CHUNKS;

  // Publish the layer counter the MoE W13->W2 barrier keys off.
  //
  // That barrier derives its release value as layer_idx + 1
  // (gang_moe_fused_mxfp4_mi300.cuh), and its d_barrier is monotonic -- never
  // reset. So layer_idx must be monotonic across the whole run, not per-layer.
  // This slot used to be stored as 0 on every entry, which made release_val
  // permanently 1: after the very first layer wrote 1, the barrier was already
  // satisfied for every later layer, so W2 workers stopped waiting for their
  // own layer's W13 and could read swiglu_out before it was written. The tile
  // ordering (all W13 tiles precede all W2 tiles, padded so every worker
  // starts on W13) usually hid it, but nothing enforced it.
  //
  // qkv_epoch_expected is exactly the counter needed: it counts
  // (iterations * num_layers + layer), monotonic and never reset. It is now a
  // pure function of the layer index rather than a snapshot, so every worker
  // -- on this XCD or any other -- computes the same value with no barrier
  // required to make it agree. That is strictly stronger than what this slot
  // relied on before.
  {
    constexpr int LAYER_IDX_SMEM_OFF =
        mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE -
        mirage::runtime::LAYER_IDX_SMEM_OFFSET_FROM_END;
    extern __shared__ char _layer_smem_init[];
    if (tid == 0) {
      *reinterpret_cast<int *>(&_layer_smem_init[LAYER_IDX_SMEM_OFF]) =
          qkv_epoch_expected;
    }
  }
  __syncthreads();

  // ══════════════════════════════════════════════════════════════════
  // Phase 1: QKV GEMM
  // ══════════════════════════════════════════════════════════════════
  MPK_TW_SUB(10, xcd_rank);
  if (xcd_rank < total_qkv_tiles_per_xcd) {
    // Under EP the "residual" this prologue reads is the PREVIOUS layer's
    // symmetric gather buffer, and the prologue performs the cross-rank SUM
    // itself as part of the pass it already makes over that vector. See
    // EP_PEER_SLOTS in gang_rmsnorm_linear_mxfp4_bias_mi300.cuh: it is what
    // lets Phase 9 end at the exchange instead of running a reduce and an exit
    // barrier after it. Slot 0 of the gather buffer sits exactly where an
    // ordinary residual would, so the pointer arithmetic is identical and
    // EP_PREV_SLOTS == 1 (layer 0, or no EP at all) reads it as one.
    void const *qkv_residual =
        (EP_PREV_SLOTS > 1) ? input_ptrs[26] : input_ptrs[1];

    // The peer's slot is NOT waited on here by default, even though these 8
    // workers are its only readers and this is the first instruction that
    // needs it.
    //
    // Moving the wait here was the obvious "dependency at the point of use"
    // play and it measured slower: 2.542 vs 2.528 ms/token. The reason is
    // where each placement sits relative to the rest of the layer. At the end
    // of the previous layer the wait trails everything -- the peer is running
    // the same layer on the same schedule, so its store has effectively
    // already landed by the time anyone looks. Here it sits UPSTREAM of the
    // Phase 2 QKV barrier, which gates all of attention, so any residual
    // latency serializes with the whole layer instead of overlapping the
    // layer boundary.
    //
    // MPK_EP_WAIT_AT_USE=1 re-tests that placement. The original measurement
    // was taken at a ~2.53 ms operating point, before the write-through fold
    // and the two L2-writeback removals; the peer wait is a much larger share
    // of what remains now, so the balance may have moved. What makes the move
    // legal at all is that the gather buffers are per-layer (ep_gather_{li} in
    // demo.py) -- layer i's slot is written only in layer i and read only in
    // layer i+1, so deferring the wait cannot expose a WAR hazard. The
    // threshold is layer_counter, not layer_counter + 1: what is awaited is
    // the PREVIOUS layer's signal, and that layer published (its own
    // layer_counter) + 1 == this layer's layer_counter.
    //
    // Only the local self-wait stays in 9d under this flag. It is not about
    // the peer at all -- it covers two purely local WAR hazards (the fold
    // reads attn_proj_out, which the next layer's Phase 7 overwrites, and it
    // zeroes moe_workspace_f32, which the next layer's Phase 8 atomicAdds
    // into), so it cannot move to a consumer that touches neither.
#if MPK_EP_WAIT_AT_USE
    if constexpr (EP_PREV_SLOTS > 1 && EP_WORLD_SIZE > 1) {
      if (tid == 0) {
        uint64_t *_prev_peer_sig = reinterpret_cast<uint64_t *>(input_ptrs[25]) +
                                   (size_t)(1 - EP_MY_PE) *
                                       FULL_LAYER_EP_SIGNAL_STRIDE;
        while (ld_nt_u64(reinterpret_cast<unsigned long long *>(
                   _prev_peer_sig)) < (unsigned long long)layer_counter) {
          __builtin_amdgcn_s_sleep(1);
        }
      }
      __syncthreads();
      // The peer's bytes arrived by write-through store, so they are in HBM,
      // not in this XCD's L2 -- but a stale line for those addresses may still
      // sit in this workgroup's vL1 from the previous layer's read of the same
      // per-layer buffer's neighbours. Plain buffer_inv (no sc1) is the right
      // scope: vL1 only, leaving L2 alone.
      asm volatile("buffer_inv" ::: "memory");
    }
#endif
    gang_resaddf32_rmsnorm_linear_mxfp4_bias_kvupd_kernel<QKV_BATCH_SIZE,
                                                          QKV_OUTPUT_PER_WG,
                                                          QKV_REDUCTION_SIZE,
                                                          ACTUAL_HIDDEN_DIM,
                                                          HEAD_DIM,
                                                          NUM_Q_PER_KV,
                                                          PAGE_SIZE,
                                                          EP_PREV_SLOTS>(
        input_ptrs[0],
        qkv_residual,
        input_ptrs[2],
        input_ptrs[3],
        input_ptrs[4],
        input_ptrs[5],
        output_ptrs[0],
        output_ptrs[1],
        output_ptrs[2],
        output_ptrs[3],
        cos_ptr,
        sin_ptr,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_active_tokens,
        qkv_n_wgs_per_xcd,
        kv_stride,
        q_ws_stride,
        xcd_rank);

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
    _fused_t0a = __builtin_amdgcn_s_memrealtime();
#endif
  } // end Phase 1: QKV GEMM

  // ══════════════════════════════════════════════════════════════════
  // Phase 2: QKV barrier — epoch-based
  //
  // Joined only by the workers that produce or consume this layer's QKV
  // output: the QKV GEMM workers and the attention-chunk workers. The MoE-only
  // workers beyond both skip it entirely and go straight to Phase 6, where
  // their O-proj weight DMA can start overlapping immediately.
  //
  // This barrier no longer carries any ordering for the other two counters.
  // It used to: every worker had to arrive so that the attn_release /
  // routing_ready snapshots were ordered against their producers. Those are
  // now derived from the layer counter and read nothing shared, so the
  // participant set is just the set that needs the barrier for its own sake --
  // workers that read K/V in Phase 3 must see the GEMM land first.
  //
  // The arrival count is qkv_epoch_participants, and the modular test only
  // fires when it matches the number of workers that actually arrive -- so
  // this constant and the guard below must stay in agreement.
  // ══════════════════════════════════════════════════════════════════
  MPK_TW_SUB(20, qkv_epoch_expected);
  MPK_WS_PHASE(20, qkv_epoch_expected, xcd_id);
  if (xcd_rank < qkv_epoch_participants) {
    __shared__ int s_prev;
    if (tid == 0) {
      s_prev = atom_add_release_gpu_s32(
          &static_cast<int *>(input_ptrs[7])[xcd_id], 1);
    }
    __syncthreads();

    if ((s_prev % qkv_epoch_participants) == qkv_epoch_participants - 1) {
      // Last worker to arrive: bump epoch (no reset needed — modular check)
      if (tid == 0) {
        atom_add_release_gpu_s32(&qkv_epoch[xcd_id * 16], 1);
      }
    }

    // Every participant polls. The participant set is now exactly the workers
    // that need this barrier, so there is no longer a subset that arrives only
    // to carry ordering for someone else.
    MPK_WS_WAIT_BEGIN(20, qkv_epoch_expected);
    if (tid == 0) {
      int _obs;
      int _spins = 0;
      while ((_obs = __atomic_load_n(&qkv_epoch[xcd_id * 16],
                                     __ATOMIC_RELAXED)) < qkv_epoch_expected) {
        MPK_WS_WAIT_TICK(_obs, _spins);
        _spins++;
        __builtin_amdgcn_s_sleep(1);
      }
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
    __syncthreads();
    asm volatile("buffer_inv" ::: "memory");
  }

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  _fused_t0b = __builtin_amdgcn_s_memrealtime();
#endif

  // ══════════════════════════════════════════════════════════════════
  // Phase 3: Parallel attention chunks
  // Workers 0..(NUM_KV_CHUNKS-1) each run one chunk.
  //
  // NOTE: this block is deliberately NOT nested inside the Phase 1 QKV guard.
  // It used to be, which capped the usable chunk count at
  // total_qkv_tiles_per_xcd (10 for GPT-OSS 120B): with NUM_KV_CHUNKS > 10 only
  // 10 workers ever reached the chunk barrier, the
  // `(s_chunk_prev % NUM_KV_CHUNKS) == NUM_KV_CHUNKS-1` merge condition never
  // fired, and the megakernel deadlocked. Attention chunks are now served by
  // any of the workers_per_xcd (30) workers on this XCD.
  MPK_TW_SUB(30, xcd_rank);
  MPK_WS_PHASE(30, qkv_epoch_expected, xcd_id);
  {
    if (xcd_rank < NUM_KV_CHUNKS) {
      int kv_chunk_idx = xcd_rank;
      using bf16_t = __hip_bfloat16;
      void const *offset_k = reinterpret_cast<bf16_t const *>(output_ptrs[1]) +
                             static_cast<size_t>(xcd_id) * HEAD_DIM;
      void const *offset_v = reinterpret_cast<bf16_t const *>(output_ptrs[2]) +
                             static_cast<size_t>(xcd_id) * HEAD_DIM;

      // Write float32 partials to o_acc_f32 (input_ptrs[23])
      // Write LSE to lse_acc (input_ptrs[8])
      // NO sinks for per-chunk — sinks applied in merge step
      paged_attention_ck_fmha_split_kv_impl<bfloat16,
                                            NUM_Q_PER_KV,
                                            HEAD_DIM,
                                            PAGE_SIZE,
                                            MAX_SEQ_LEN,
                                            NUM_KV_CHUNKS,
                                            Q_WORKSPACE_STRIDE,
                                            KV_CACHE_STRIDE,
                                            NUM_KV_HEADS,
                                            DECODE_ONLY>(
          output_ptrs[3], // q_workspace
          const_cast<void *>(offset_k),
          const_cast<void *>(offset_v),
          input_ptrs[23], // o_acc_f32 (float32 partials)
          input_ptrs[8],  // lse_acc
          qo_indptr,
          kv_indptr,
          kv_indices,
          kv_last_page_len,
          /*request_id=*/0,
          /*kv_head_idx=*/xcd_id,
          kv_chunk_idx,
          attn_scale,
          SLIDING_WINDOW,
          nullptr); // no sinks per-chunk

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
      _fused_t0c = __builtin_amdgcn_s_memrealtime();
#endif

      // ══════════════════════════════════════════════════════════════════
      // Phase 4: Chunk barrier — CROC last-chunk-worker runs merge
      // ══════════════════════════════════════════════════════════════════
      MPK_TW_SUB(40, kv_chunk_idx);
      MPK_WS_PHASE(40, qkv_epoch_expected, xcd_id);
      // Flush this chunk's o_acc/lse_acc partials before arriving.
      //
      // The chunk kernel writes its partials with ordinary stores, which may
      // still be sitting in this CU's write buffer / vL1 when the atomic
      // below retires. atom_add_release_gpu_s32 orders *this thread's* prior
      // writes, but tid 0 is not the thread that wrote most of this chunk's
      // partials -- the other 255 threads did, and __syncthreads is a
      // block-execution barrier, not a memory-visibility one to other CUs.
      // Without this, the merging worker (a *different* block, possibly a
      // different CU) can read a partially-written o_acc/lse_acc slot.
      //
      // The result is silent numerical corruption, not a crash: merge weights
      // each chunk by exp2(lse - m_global), so a stale/torn lse reads as
      // garbage magnitude. A torn lse that lands large makes m_global huge and
      // drives every other chunk's weight to zero; one that lands as raw
      // uninitialized bits can be NaN, which propagates through the whole
      // attention output. That is the seq-len-dependent part: at 512 tokens
      // ntiles=32 so 14 of 30 chunks exit early via the empty-chunk path and
      // never race, while at 32k ntiles=2048 gives every one of the 30 chunks
      // real work to write, so the window is open on all of them every layer.
      //
      // Scope: this barrier is per-XCD (chunk_barrier[xcd_id * 16]), so the
      // chunk workers and the merging worker are always on the same XCD and
      // share one 32MB L2. Making the partials visible therefore only requires
      // getting them *into* L2 -- not flushing L2 to HBM.
      //
      // db48239 used threadfence_gpu() here, which is agent scope and lowers to
      // `buffer_wbl2 sc1; s_waitcnt vmcnt(0)` (verified in gfx950 ISA). The
      // buffer_wbl2 is an L2->HBM writeback of the whole cache, paid by every
      // chunk worker on every layer, and it buys nothing a same-XCD consumer
      // can observe: see the intra-XCD section of mpk_atoms.cuh, "all CUs
      // within an XCD share the same 32MB L2 ... No buffer_wbl2 required".
      // Dropping it is worth ~0.16 ms/iter at seq 512.
      //
      // What remains is the part that is actually load-bearing: s_waitcnt
      // vmcnt(0) retires all 256 threads' outstanding stores so they have
      // reached L2 before tid 0's release atomic. Workgroup-scope fences emit
      // no instruction at all on gfx950, so they cannot substitute -- the
      // consumer is a different block.
      __syncthreads();
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      __shared__ int s_chunk_prev;
      if (tid == 0) {
        s_chunk_prev = atom_add_release_gpu_s32(&chunk_barrier[xcd_id * 16], 1);
      }
      __syncthreads();

      if ((s_chunk_prev % NUM_KV_CHUNKS) == NUM_KV_CHUNKS - 1) {
        // Last chunk worker: run merge (no reset needed — modular check)
        //
        // Acquire the partials the other chunks released above. buffer_inv
        // drops this CU's stale vL1 lines so the merge reads what they
        // actually wrote rather than a cached copy from a previous layer --
        // vL1 is per-CU, so this is required even though the producers share
        // our L2.
        //
        // Plain `buffer_inv` (no sc1) invalidates vL1 only. The acquire fence
        // that used to precede it was agent scope, which emits `buffer_inv
        // sc1` and additionally invalidates L2 -- discarding lines this XCD's
        // own chunk workers had just written, and forcing them to be re-read
        // from HBM. The producers are all on this XCD (the barrier is
        // per-XCD), so their stores are already in our L2 and invalidating it
        // is both unnecessary and actively harmful.
        asm volatile("buffer_inv" ::: "memory");

        // ══════════════════════════════════════════════════════════════════
        // Phase 5: Merge (write-through fused) + signal
        // ══════════════════════════════════════════════════════════════════
        MPK_TW_SUB(50, s_chunk_prev);
        MPK_WS_PHASE(50, qkv_epoch_expected, xcd_id);
        // WRITE_THROUGH=true: merge writes bf16 output directly via st_wt,
        // eliminating the separate __syncthreads + readback + flush pass.
        merge_splitkv_ck_fmha<__hip_bfloat16,
                              NUM_Q_PER_KV,
                              NUM_KV_HEADS,
                              HEAD_DIM,
                              NUM_KV_CHUNKS,
                              128,
                              4096,
                              /*WRITE_THROUGH=*/true>(
            reinterpret_cast<float const *>(input_ptrs[8]),  // lse_acc
            reinterpret_cast<float const *>(input_ptrs[23]), // o_acc_f32
            qo_indptr,
            kv_indptr,
            kv_last_page_len,
            /*request_id=*/0,
            reinterpret_cast<__hip_bfloat16 *>(
                output_ptrs[4]), // attn_out (bf16)
            /*kv_head_idx=*/xcd_id,
            HAS_SINKS ? input_ptrs[6] : nullptr); // sinks applied here

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
        _merge_done = __builtin_amdgcn_s_memrealtime();
#endif
        // Wait for all write-through stores from merge to complete
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        if (tid == 0) {
          int attn_expected =
              (__atomic_load_n(attn_global, __ATOMIC_RELAXED) | 7) + 1;
          int prev = atom_add_release_gpu_s32(attn_global, 1);
          if (prev == attn_expected - 1) {
            // LAST XCD to arrive: fan out per-XCD release flags via st_wt
            // (write-through to HBM, bypasses L2 — guarantees ld_nt sees it).
            // This eliminates the cross-XCD poll of attn_global, which hangs
            // because ld_nt/relaxed loads read stale L2 cached values.
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            for (int x = 0; x < 8; x++) {
              st_wt_u32((void *)&attn_release[x * 16],
                        (unsigned)attn_release_expected);
            }
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          }
        }
#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
        _fused_t0d = __builtin_amdgcn_s_memrealtime();
#endif
      }
    }
  }

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  unsigned long long _fused_t1 = __builtin_amdgcn_s_memrealtime();
#endif

  // ══════════════════════════════════════════════════════════════════
  // Phase 6: Cross-XCD attention barrier + O-proj weight DMA
  // Issue buffer_load_lds for O-proj weights BEFORE the barrier poll
  // so DMA runs in the background during the spin-wait (~5-60us).
  // ══════════════════════════════════════════════════════════════════
  MPK_TW_SUB(60, attn_release_expected);
  MPK_WS_PHASE(60, qkv_epoch_expected, xcd_id);
  int oproj_topk_tiles_per_xcd =
      oproj_tiles_per_xcd > router_tile_n ? oproj_tiles_per_xcd : router_tile_n;

  {
    constexpr int OPROJ_WG_DATA =
        OPROJ_OUTPUT_PER_WG * (OPROJ_REDUCTION_SIZE / 2);
    constexpr int OPROJ_NUM_B32 = OPROJ_REDUCTION_SIZE / 32;
    constexpr int OPROJ_WG_SCALE = OPROJ_OUTPUT_PER_WG * OPROJ_NUM_B32;
    constexpr int OPROJ_WG_BYTES = OPROJ_WG_DATA + OPROJ_WG_SCALE;
    constexpr int OPROJ_N16_DATA = OPROJ_WG_DATA / 16;
    constexpr int OPROJ_LPT = (OPROJ_N16_DATA + 255) / 256;
    constexpr int OPROJ_DATA_PAD = OPROJ_LPT * 256 * 16;
    constexpr int OPROJ_N16_SCALE = (OPROJ_WG_SCALE + 15) / 16;
    constexpr int OPROJ_SLPT = (OPROJ_N16_SCALE + 255) / 256;
    constexpr int OPROJ_SCALE_PAD = OPROJ_SLPT * 256 * 16;

    // Activations are FP8 E4M3: one byte per element, one E8M0 scale byte per
    // 32-element block. See _gang_wave_parallel_fp8_quant.
    constexpr int OPROJ_FP_TOK = OPROJ_REDUCTION_SIZE;
    constexpr int OPROJ_FP_SCL = OPROJ_NUM_B32;
    constexpr int OPROJ_LDS_W_OFF =
        ((OPROJ_FP_TOK + OPROJ_FP_SCL + 15) / 16) * 16;

    extern __shared__ char _oproj_pf_smem[];

    int oproj_tile_idx_pf = xcd_id * oproj_topk_tiles_per_xcd + xcd_rank;
    int oproj_tok_pf =
        (xcd_rank % oproj_topk_tiles_per_xcd) / oproj_n_wgs_per_xcd;
    int oproj_wg_pf =
        (xcd_rank % oproj_topk_tiles_per_xcd) % oproj_n_wgs_per_xcd;

    if (xcd_rank < oproj_topk_tiles_per_xcd &&
        oproj_tok_pf < num_active_tokens) {
      uint8_t const *oproj_W = (uint8_t const *)input_ptrs[9];
      uint32_t oproj_buf_range =
          static_cast<uint32_t>(oproj_n_wgs_per_xcd) * OPROJ_WG_BYTES;
      i32x4_t oproj_rsrc = make_w_buffer_rsrc(oproj_W, oproj_buf_range);
      uint32_t oproj_wg_voff =
          static_cast<uint32_t>(oproj_wg_pf) * OPROJ_WG_BYTES;

      auto *oproj_lds_base = (__attribute__((address_space(3)))
                              uint32_t *)(_oproj_pf_smem + OPROJ_LDS_W_OFF);
      // buffer_load_lds writes to M0 + lane_id*16.  Each wave fills 1024 bytes.
      // Must offset by warp_id*1024 so 4 waves write to distinct slices.
      int const pf_warp_id = tid >> 6;

#pragma unroll
      for (int j = 0; j < OPROJ_LPT; j++) {
        int idx = tid + j * 256;
        int clamped = idx < OPROJ_N16_DATA ? idx : OPROJ_N16_DATA - 1;
        uint32_t voff = oproj_wg_voff + static_cast<uint32_t>(clamped) * 16;
        auto *lds_dst =
            (__attribute__((address_space(3)))
             uint32_t *)((uint8_t
                          __attribute__((address_space(3))) *)oproj_lds_base +
                         j * 4096 + pf_warp_id * 1024);
        __llvm_amdgcn_raw_buffer_load_lds(
            oproj_rsrc, lds_dst, 16, static_cast<int>(voff), 0, 0, 3);
      }
#pragma unroll
      for (int j = 0; j < OPROJ_SLPT; j++) {
        int idx = tid + j * 256;
        int clamped = idx < OPROJ_N16_SCALE ? idx : OPROJ_N16_SCALE - 1;
        uint32_t voff =
            oproj_wg_voff + OPROJ_WG_DATA + static_cast<uint32_t>(clamped) * 16;
        auto *lds_dst =
            (__attribute__((address_space(3)))
             uint32_t *)((uint8_t
                          __attribute__((address_space(3))) *)oproj_lds_base +
                         OPROJ_DATA_PAD + j * 4096 + pf_warp_id * 1024);
        __llvm_amdgcn_raw_buffer_load_lds(
            oproj_rsrc, lds_dst, 16, static_cast<int>(voff), 0, 0, 3);
      }
    }
  }

  MPK_WS_WAIT_BEGIN(60, attn_release_expected);
  {
    int _obs;
    int _spins = 0;
    while ((_obs = ld_nt_s32(&attn_release[xcd_id * 16])) <
           attn_release_expected) {
      MPK_WS_WAIT_TICK(_obs, _spins);
      _spins++;
      __builtin_amdgcn_s_sleep(1);
    }
  }
  asm volatile("buffer_inv" ::: "memory");
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  unsigned long long _fused_t2 = __builtin_amdgcn_s_memrealtime();
#endif

  // ══════════════════════════════════════════════════════════════════
  // Phase 7: O-proj + RMSNorm + Router + TopK
  // ══════════════════════════════════════════════════════════════════
  MPK_TW_SUB(70, oproj_topk_tiles_per_xcd);
  MPK_WS_PHASE(70, qkv_epoch_expected, xcd_id);
  {
    if (xcd_rank < oproj_topk_tiles_per_xcd) {
      int oproj_tile_idx = xcd_id * oproj_topk_tiles_per_xcd + xcd_rank;
      void const *oproj_residual_ptr =
          reinterpret_cast<__hip_bfloat16 const *>(output_ptrs[0]) +
          static_cast<size_t>(xcd_id) * oproj_n_wgs_per_xcd *
              OPROJ_OUTPUT_PER_WG;
      gang_linear_mxfp4_res_bias_rmsnorm_topk_kernel<QKV_BATCH_SIZE,
                                                     OPROJ_OUTPUT_PER_WG,
                                                     OPROJ_REDUCTION_SIZE,
                                                     ACTUAL_HIDDEN_DIM,
                                                     NUM_EXPERTS,
                                                     TOPK_K>(
          output_ptrs[4],
          input_ptrs[9],
          oproj_residual_ptr,
          input_ptrs[10],
          input_ptrs[11],
          input_ptrs[12],
          input_ptrs[13],
          input_ptrs[14],
          input_ptrs[15],
          input_ptrs[16],
          output_ptrs[5],
          output_ptrs[6],
          output_ptrs[7],
          output_ptrs[8],
          num_active_tokens,
          oproj_n_wgs_per_xcd,
          oproj_output_stride,
          router_tile_n,
          total_oproj_tiles,
          total_topk_tiles,
          oproj_topk_tiles_per_xcd,
          oproj_tile_idx,
          routing_ready,
          // ts_base: the O-proj/TopK kernel's optional per-sub-op timestamp
          // sink. Nothing here reads those slots -- the [FUSED_PHASE] printf
          // below derives every number from _fused_t0.._fused_t4, which are
          // taken in this function. This used to name a _ts_base that was
          // never declared, so MPK_DEVICE_TIMING=1 did not compile.
          nullptr);
    }
  }

  // ══════════════════════════════════════════════════════════════════
  // Phase 7b: wait for TopK (XCD-local release flag)
  // ══════════════════════════════════════════════════════════════════
  // Cross-workgroup sync: the last OProj workgroup runs TopK and
  // signals routing_ready. Other workgroups poll here until TopK
  // results are globally visible.
  // All threads poll independently — eliminates __syncthreads overhead.
  MPK_TW_SUB(75, routing_expected);
  MPK_WS_PHASE(75, qkv_epoch_expected, xcd_id);
  {
    int *my_release = &routing_ready[(1 + xcd_id) * 16];
    MPK_WS_WAIT_BEGIN(75, routing_expected);
    int _obs;
    int _spins = 0;
    while ((_obs = ld_nt_s32(my_release)) < routing_expected) {
      MPK_WS_WAIT_TICK(_obs, _spins);
      _spins++;
      __builtin_amdgcn_s_sleep(1);
    }
  }
  asm volatile("buffer_inv" ::: "memory");

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  unsigned long long _fused_t3 = __builtin_amdgcn_s_memrealtime();
  // Entry spread into Phase 8, measured the same way as the 9a arrival spread.
  // 9a's spread is 19.4 us; the question this answers is whether MoE CREATES
  // that skew (entry spread small, exit spread large -> tile imbalance, fix the
  // partition) or merely INHERITS it (both large -> the skew predates MoE and
  // no MoE-side balancing helps). Reset by the same closer that reads it.
  if (threadIdx.x == 0) {
    atomicMin(&g_ep8_arr_min, _fused_t3);
    atomicMax(&g_ep8_arr_max, _fused_t3);
  }
#endif

  // ══════════════════════════════════════════════════════════════════
  // Phase 8: MoE (W13+SwiGLU+W2)
  // ══════════════════════════════════════════════════════════════════
#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  {
    int _wid = xcd_id * workers_per_xcd + xcd_rank;
    if (threadIdx.x == 0 && _wid < MOE_OCC_WORKERS) {
      unsigned long long _n = 0;
      for (int _t = xcd_rank; _t < moe_total_tiles_per_xcd;
           _t += workers_per_xcd) {
        _n++;
      }
      atomicAdd(&g_moe_tiles[_wid], _n);
    }
  }
#endif
  for (int moe_t = xcd_rank; moe_t < moe_total_tiles_per_xcd;
       moe_t += workers_per_xcd) {
    MPK_TW_SUB(80, moe_t);
    MPK_WS_PHASE(80, qkv_epoch_expected, xcd_id);
    gang_moe_fused_mxfp4_kernel_mi300<QKV_BATCH_SIZE,
                                      MOE_INTERMEDIATE_SIZE,
                                      MOE_HIDDEN_SIZE,
                                      NUM_EXPERTS,
                                      TOPK_K,
                                      MOE_W13_OUTPUT_PER_WG,
                                      MOE_W2_OUTPUT_PER_WG,
                                      MOE_EXPERT_BASE,
                                      MOE_NUM_LOCAL_EXPERTS,
                                      EP_SLOT_WS,
                                      EP_SLOT_ME>(input_ptrs[12],
                                                            input_ptrs[17],
                                                            input_ptrs[18],
                                                            output_ptrs[7],
                                                            output_ptrs[8],
                                                            input_ptrs[19],
                                                            input_ptrs[20],
                                                            output_ptrs[9],
                                                            input_ptrs[22],
                                                            output_ptrs[10],
                                                            input_ptrs[21],
                                                            moe_t);
  }

  // ══════════════════════════════════════════════════════════════════
  // Phase 9: inline expert-parallel combine
  //
  // Under EP each rank owns MOE_NUM_LOCAL_EXPERTS of the NUM_EXPERTS experts,
  // so what Phase 8 leaves in moe_workspace_f32 is a PARTIAL weighted sum.
  // The complete value is the sum across ranks, and the layer's residual must
  // be added to it exactly once.
  //
  // This used to be three dispatched tasks after the monolith --
  // moe_residual_add_f32 -> identity -> allreduce. That cost two scheduler
  // round-trips per layer and, worse, put a SECOND gang group between
  // consecutive monolith dispatches: workers blocked in the collective's gang
  // barrier while other workers blocked in the next layer's Phase 6 barrier,
  // each group waiting on the other. It deadlocked. Doing the exchange here
  // removes the second gang group along with the round-trips: one dispatch
  // per layer, and the scheduler is not involved in the collective at all.
  //
  //   9a  every worker on every XCD arrives -- the partial is only complete
  //       once the last W2 atomicAdd anywhere on this GPU has landed
  //   9b  one workgroup folds f32 partial (+ residual on EP_FOLD_PE) to bf16
  //       into this rank's slot of the symmetric gather buffer, and zeroes
  //       the workspace for the next layer
  //   9c  that workgroup puts its slot to every peer and signals
  //   9d  wait for all peers' slots to arrive AND for 9b's local fold, then
  //       every XCD's workgroup 0 reduces the gather buffer into the output
  //   9e  every other worker waits for its XCD's combine before leaving the
  //       layer -- without this the layer has no exit fence at all
  //
  // For EP_WORLD_SIZE == 1 (single GPU, or multi-GPU with replicated MoE) the
  // whole block compiles away and the next layer's QKV prologue closes out the
  // workspace exactly as before.
  // ══════════════════════════════════════════════════════════════════
  // ── Ablation knobs (MPK_EP_ABLATE, off by default) ──────────────────
  //
  // Phase 9 bundles four costs that the end-to-end number cannot separate:
  // the GPU-wide barrier (9a), the fold, the cross-GPU transfer (9c/9d), and
  // the exit barrier (9e). These strip them one at a time so each can be
  // priced. BOTH SETTINGS PRODUCE WRONG OUTPUT -- they exist only to
  // attribute latency, and every run using them must be reported as such.
  //
  //   MPK_EP_ABLATE=1  skip the put and the peer signal-wait. All barriers,
  //                    the fold and the local reduce stay. The reduce then
  //                    sums this rank's slot with a stale peer slot, so the
  //                    answer is garbage but the intra-GPU control flow and
  //                    every memory access is byte-for-byte the real one.
  //                    Difference vs. full = cost of the network round trip.
  //   MPK_EP_ABLATE=2  skip Phase 9 entirely. Experts are still sliced 64/64,
  //                    so this prices the expert-work SAVING against the DP
  //                    baseline with none of the combine's cost. Output is a
  //                    partial sum -- garbage, and the next layer's prologue
  //                    reads the un-zeroed workspace too.
  //   MPK_EP_ABLATE=6  keep 9a, the fold and the peer put, but let the 232
  //                    workers that do NOT fold leave without waiting in 9d.
  //                    The folding workgroups (xcd_rank == 0) still take the
  //                    full wait, so the transfer is still ordered; what goes
  //                    away is 232 workers sitting on a signal for work they
  //                    never read. Prices the ceiling of ping-pong buffering
  //                    attn_proj_out / moe_workspace_f32, which is what would
  //                    make this legal -- both hazards 9d covers are WAR
  //                    against the NEXT layer's reuse of those two buffers.
  //                    WRONG OUTPUT as written: with one buffer a fast worker
  //                    does clobber them.
#ifndef MPK_EP_ABLATE
#define MPK_EP_ABLATE 0
#endif
  if constexpr (EP_WORLD_SIZE > 1 && MPK_EP_ABLATE != 2) {
    int *ep_moe_done = oproj_counters_base + FULL_LAYER_EP_MOE_DONE_SLOT;
    int *ep_release = oproj_counters_base + FULL_LAYER_EP_RELEASE_SLOT;
    int *ep_fold_done = oproj_counters_base + FULL_LAYER_EP_FOLD_DONE_SLOT;
    int *ep_combine_done =
        oproj_counters_base + FULL_LAYER_EP_COMBINE_DONE_SLOT;
    int *ep_xcd_arrive = oproj_counters_base + FULL_LAYER_EP_XCD_ARRIVE_SLOT;
    // input_ptrs[24]: [EP_WORLD_SIZE, batch, QKV_REDUCTION_SIZE] bf16
    // symmetric gather buffer. Slot p holds rank p's folded partial.
    // input_ptrs[25]: [EP_WORLD_SIZE * 8] uint64 symmetric signal counters.
    //
    // QKV_REDUCTION_SIZE, not OPROJ_REDUCTION_SIZE: what is exchanged is a
    // residual-stream vector (padded hidden), whereas OPROJ_REDUCTION_SIZE is
    // o_proj's *input* width (num_heads * head_dim).
    __hip_bfloat16 *ep_gather =
        reinterpret_cast<__hip_bfloat16 *>(input_ptrs[24]);
    uint64_t *ep_signal = reinterpret_cast<uint64_t *>(input_ptrs[25]);
    constexpr size_t EP_SLOT_ELEMS =
        (size_t)QKV_BATCH_SIZE * QKV_REDUCTION_SIZE;
    constexpr size_t EP_SLOT_BYTES = EP_SLOT_ELEMS * sizeof(__hip_bfloat16);

    // The release/signal thresholds ride the same run-monotonic layer counter
    // every other barrier here uses, so nothing is reset between layers and no
    // worker has to observe a shared value to agree on the target.
    int const ep_expected = layer_counter + 1;
    // One signal line per producing PE, and the consumer waits on each line
    // separately, so a line's threshold is just the layer count -- one arrival
    // per layer from its one writer. This used to be (EP_WORLD_SIZE - 1) *
    // ep_expected, which is the same thing at world size 2 and too high at any
    // larger size: a line would never reach it and the wait would hang.
    uint64_t const ep_sig_expected = (uint64_t)ep_expected;

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
    _ep_t0 = __builtin_amdgcn_s_memrealtime();
#endif

    MPK_WS_PHASE(91, qkv_epoch_expected, xcd_id);
    // ── 9a: GPU-wide MoE barrier, as a two-level tree ───────────────────
    // Every worker, not just this XCD's: W2 tiles for an expert can be
    // executed by any XCD, so the partial is not complete until all 240 have
    // finished. vmcnt(0) retires this block's atomicAdds to L2 before the
    // arrival is visible.
    //
    // The arrival used to be a single atomicAdd to ep_moe_done from all 240
    // workers. Every one of those is a far atomic serialized at the same
    // coherence point, so the barrier cost scaled with the full worker count
    // -- 240 deep, and it measured like it (9a p50 5.4 us, p90 22.2 us, for a
    // barrier whose actual work is zero). Every other barrier in this file
    // already avoids that by fanning out over 8 per-XCD lines; 9a was the one
    // that did not.
    //
    // Two levels: 30 workers contend per XCD line (8 lines in parallel), then
    // the 8 XCD leaders contend on the single global line. Depth 30 + 8
    // instead of 240, with identical semantics -- the global counter still
    // only reaches its target once every worker on the GPU has arrived.
    __syncthreads();
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    bool is_last_on_gpu = false;
    if (tid == 0) {
#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
      // Arrival instant of THIS workgroup, before it touches the counter.
      // min/max over the 240 give the spread the barrier is absorbing.
      {
        unsigned long long _a = _ep_t0;
        atomicMin(&g_ep9_arr_min, _a);
        atomicMax(&g_ep9_arr_max, _a);
        // Phase 8 occupancy for THIS worker: entry (_fused_t3) to arrival.
        int _wid = xcd_id * workers_per_xcd + xcd_rank;
        if (_wid < MOE_OCC_WORKERS && _a >= _fused_t3) {
          atomicAdd(&g_moe_busy_ns[_wid], _a - _fused_t3);
        }
      }
#endif
      int prev_x = atom_add_release_gpu_s32(
          &ep_xcd_arrive[xcd_id * FULL_LAYER_EP_XCD_STRIDE], 1);
      if (prev_x % workers_per_xcd == workers_per_xcd - 1) {
        // Leader of this XCD: one arrival on behalf of all 30.
#if MPK_EP_ABLATE == 3
        // Ablation: drop the CROSS-XCD level. Each XCD's leader acts as if it
        // closed the whole GPU, so the rendezvous never spans XCDs. Prices the
        // GPU-wide-ness of 9a against its arrival depth -- the two levels cost
        // very different things and the end-to-end number cannot separate them.
        // 8 workgroups then fold concurrently and every XCD reduces against a
        // partial that may still be growing: WRONG OUTPUT, latency only.
        is_last_on_gpu = true;
#else
        int prev = atom_add_release_gpu_s32(ep_moe_done, 1);
        if (prev % 8 == 7) {
          is_last_on_gpu = true;
        }
#endif
      }
    }

    MPK_WS_PHASE(92, qkv_epoch_expected, xcd_id);
    // ── 9b/9c: fold + stream, run BY the worker that closed the barrier ──
    //
    // This used to be a separate step: 9a released 8 flags, XCD 0's workgroup 0
    // woke up, folded, released 8 more flags, and only then could 9d proceed.
    // Two full flag round-trips to hand 0.1 us of folding from one workgroup to
    // another. The worker that closes 9a already knows the GPU is done -- it is
    // the one that observed the last arrival -- so it just does the fold itself
    // and publishes once. One release chain per layer instead of three.
    //
    // No threadfence_gpu() for the closer either. It is a release fence, and
    // between its arrival atomic and here it has written nothing to release:
    // the ep_release flags below are st_wt_u32 and the fold's stores are all
    // write-through now. The acquire it actually needs -- seeing the W2
    // atomicAdds other XCDs made into the workspace -- is the buffer_inv before
    // the fold, which is a different instruction and still there. This one is
    // left over from when 9a's job was to release flags to a separate folding
    // workgroup, and it costs a whole-L2 writeback on the single worker every
    // other worker on the GPU is waiting for.
#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
    // Read the arrival spread from the closer: it has just observed the 240th
    // arrival, so min/max are final for this layer and no one has started the
    // next. Reset here for the same reason.
    if (is_last_on_gpu && tid == 0) {
      unsigned long long lo = g_ep9_arr_min, hi = g_ep9_arr_max;
      unsigned long long span = (hi >= lo) ? (hi - lo) : 0;
      atomicAdd(&g_ep9_arr_span_sum, span);
      atomicMax(&g_ep9_arr_span_max, span);
      unsigned long long lo8 = g_ep8_arr_min, hi8 = g_ep8_arr_max;
      unsigned long long span8 = (hi8 >= lo8) ? (hi8 - lo8) : 0;
      atomicAdd(&g_ep8_arr_span_sum, span8);
      atomicMax(&g_ep8_arr_span_max, span8);
      unsigned long long an = atomicAdd(&g_ep9_arr_n, 1ull);
      g_ep9_arr_min = ~0ull;
      g_ep9_arr_max = 0ull;
      g_ep8_arr_min = ~0ull;
      g_ep8_arr_max = 0ull;
      if (an == 36ull * 100ull) {
        printf("[EP9ARR] n=%llu p8_entry_spread_us mean=%.3f max=%.3f | "
               "p9_entry_spread_us mean=%.3f max=%.3f\n",
               an,
               (double)g_ep8_arr_span_sum / an / 100.0,
               (double)g_ep8_arr_span_max / 100.0,
               (double)g_ep9_arr_span_sum / an / 100.0,
               (double)g_ep9_arr_span_max / 100.0);
        // Per-worker MoE occupancy, one line per XCD to keep the printf
        // count trivial. Values are per layer-instance: busy_us is Phase 8
        // entry -> 9a arrival, tiles is loop trips handed to that worker.
        for (int x = 0; x < 8; x++) {
          double bmin = 1e30, bmax = 0.0, bsum = 0.0;
          unsigned long long tmin = ~0ull, tmax = 0;
          for (int r = 0; r < 30; r++) {
            int w = x * 30 + r;
            // One sample per worker per layer, and `an` counts layers, so this
            // is already per-layer; ticks -> us is /100.
            double b = (double)g_moe_busy_ns[w] / an / 100.0;
            unsigned long long t = g_moe_tiles[w];
            bsum += b;
            if (b < bmin) {
              bmin = b;
            }
            if (b > bmax) {
              bmax = b;
            }
            if (t < tmin) {
              tmin = t;
            }
            if (t > tmax) {
              tmax = t;
            }
          }
          printf("[MOEOCC] xcd=%d busy_us min=%.2f mean=%.2f max=%.2f | "
                 "tiles min=%llu max=%llu\n",
                 x,
                 bmin,
                 bsum / 30.0,
                 bmax,
                 tmin,
                 tmax);
        }
        // Per-TILE cost, averaged over every tile executed on this rank.
        // w2_bar is the W13->W2 per-expert barrier poll, which is included in
        // w2_tot; w2_tot - w2_bar is W2's own compute.
        printf("[MOETILE] w13 n=%llu us=%.2f | w2 n=%llu tot=%.2f bar=%.2f "
               "compute=%.2f\n",
               g_moe_w13_n,
               g_moe_w13_n ? (double)g_moe_w13_ns / g_moe_w13_n / 100.0 : 0.0,
               g_moe_w2_n,
               g_moe_w2_n ? (double)g_moe_w2_ns / g_moe_w2_n / 100.0 : 0.0,
               g_moe_w2_n ? (double)g_moe_w2bar_ns / g_moe_w2_n / 100.0 : 0.0,
               g_moe_w2_n ? (double)(g_moe_w2_ns - g_moe_w2bar_ns) /
                                g_moe_w2_n / 100.0
                          : 0.0);
      }
    }
#endif
    // The closer releases every XCD immediately, before folding anything. The
    // fold is then done by all 8 XCD workgroups in parallel, each on its own
    // column slice, rather than by this one workgroup while 239 workers idle.
    //
    // Measured: the bare rendezvous costs 0.29 ms/token (MPK_EP_ABLATE=5) and
    // adding the single-workgroup fold + reduce took it to 0.59 -- the fold was
    // as expensive as the barrier that gated it, because 11.8 KB of f32 reads
    // and 5.8 KB of XGMI stores were running at 1/240 of the machine. Slicing
    // it 8 ways puts that work on hardware that is already awake and waiting.
#ifdef MPK_EP_SKEW_PROBE
    // Read the probe from the ONE worker that has just observed every W2 tile
    // on the GPU. At this instant g_ep_slice_last[] and g_ep_gpu_last are final
    // for this layer and nothing has started the next one, so the sample needs
    // no barrier of its own. Reset here too, for the same reason.
    if (is_last_on_gpu) {
      unsigned long long gpu_last = g_ep_gpu_last;
      int slot = g_ep_skew_n % EP_SKEW_HIST;
      unsigned long long span_min = ~0ull;
      for (int s = 0; s < EP_SKEW_SLICES; s++) {
        unsigned long long sl = g_ep_slice_last[s];
        // Headroom for slice s: how long it sat complete while the barrier
        // waited for the rest of the GPU.
        g_ep_skew_hist[slot][s] = (sl > 0 && gpu_last >= sl) ? (gpu_last - sl) : 0;
        if (sl > 0 && sl < span_min) {
          span_min = sl;
        }
        g_ep_slice_last[s] = 0;
      }
      // Span = last tile anywhere minus the FIRST slice to complete: the total
      // spread of W2 completion, i.e. the most any scheme could overlap.
      g_ep_skew_span[slot] = (span_min != ~0ull) ? (gpu_last - span_min) : 0;
      g_ep_gpu_last = 0;
      g_ep_skew_n = g_ep_skew_n + 1;
      // Dump from here rather than the TERMINATE path: this block is on the
      // hot path and certain to run, and at n=3600 we are well past prefill
      // (72 iters x 36 layers = 2592) and into decode steady state. One
      // worker, once, so the printf does not perturb the measurement.
      if (g_ep_skew_n == 3600) {
        printf("[EPSKEW] n=%d window=%d unit=us (1 tick = 10ns)\n",
               g_ep_skew_n,
               EP_SKEW_HIST);
        for (int s = 0; s < EP_SKEW_SLICES; s++) {
          unsigned long long sum = 0, mx = 0;
          for (int i = 0; i < EP_SKEW_HIST; i++) {
            unsigned long long v = g_ep_skew_hist[i][s];
            sum += v;
            if (v > mx) {
              mx = v;
            }
          }
          printf("[EPSKEW] slice%d mean=%.3f max=%.3f\n",
                 s,
                 (double)sum / EP_SKEW_HIST / 100.0,
                 (double)mx / 100.0);
        }
        unsigned long long ssum = 0, smx = 0;
        for (int i = 0; i < EP_SKEW_HIST; i++) {
          unsigned long long v = g_ep_skew_span[i];
          ssum += v;
          if (v > smx) {
            smx = v;
          }
        }
        printf("[EPSKEW] span mean=%.3f max=%.3f\n",
               (double)ssum / EP_SKEW_HIST / 100.0,
               (double)smx / 100.0);
      }
    }
#endif
    if (is_last_on_gpu) {
      for (int x = 0; x < 8; x++) {
        st_wt_u32((void *)&ep_release[x * FULL_LAYER_EP_XCD_STRIDE],
                  (unsigned)ep_expected);
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }

    // Every XCD's workgroup 0 waits for the release, then folds its slice.
    __shared__ int s_ep_closer;
    if (tid == 0) {
      s_ep_closer = 0;
      if (xcd_rank == 0) {
        int _obs;
        while ((_obs = ld_nt_s32(
                    &ep_release[xcd_id * FULL_LAYER_EP_XCD_STRIDE])) <
               ep_expected) {
          __builtin_amdgcn_s_sleep(1);
        }
        s_ep_closer = 1;
      }
    }
    __syncthreads();

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
    // 9a ends HERE, not after the fold. The old placement folded the two
    // together and reported 9bc as 0, which is how the fold's cost stayed
    // hidden inside the "barrier" number.
    _ep_t1 = __builtin_amdgcn_s_memrealtime();
#endif

    // Column slice for this XCD. Rounded to an even boundary so the packed
    // 32-bit peer stores never straddle two slices.
    constexpr int EP_FOLD_COLS = QKV_REDUCTION_SIZE;
    constexpr int EP_FOLD_CHUNK = ((EP_FOLD_COLS + 7) / 8 + 1) & ~1;
    int const ep_col_lo = xcd_id * EP_FOLD_CHUNK;
    int const ep_col_hi = (ep_col_lo + EP_FOLD_CHUNK) < EP_FOLD_COLS
                              ? (ep_col_lo + EP_FOLD_CHUNK)
                              : EP_FOLD_COLS;

    // MPK_EP_ABLATE=5 keeps the 9a rendezvous and the 9e exit fence but drops
    // everything in between: no fold, no transfer, no reduce. The barrier
    // structure alone. This is the one measurement that separates "the GPU-wide
    // rendezvous is inherently expensive" from "the work it gates is". WRONG
    // OUTPUT: nothing writes ep_combined and the workspace is never zeroed.
#if MPK_EP_ABLATE == 5
    if (s_ep_closer && tid == 0) {
      for (int x = 0; x < 8; x++) {
        st_wt_u32((void *)&ep_release[x * FULL_LAYER_EP_XCD_STRIDE],
                  (unsigned)ep_expected);
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }
    if (false) {
#else
    if (s_ep_closer) {
#endif
      MPK_WS_PHASE(93, qkv_epoch_expected, xcd_id);
      asm volatile("buffer_inv" ::: "memory");
      //
      // REQUIRES patches/rocshmem-ipc-shmem-ptr.patch. Stock rocSHMEM stubs
      // out IPCContext::shmem_ptr to return nullptr, so ep_direct is false and
      // every layer silently takes the staged putmem_signal fallback below --
      // which is what all EP numbers before 2026-08-09 were measuring. With
      // the patch: 2.378 -> 2.229 ms/iter, 4/4 correctness.
      //
      // The failure mode is silent by construction, because nullptr is also
      // the legitimate "no peer mapping" answer. Confirm the direct path is
      // live with MPK_EP_SIG_DBG=1 and read the [EPPATH] line; ep_direct=1 on
      // both ranks is the only evidence that these comments describe the code
      // that actually runs. The patch is still required after the peer-delta
      // change below: init samples the mapping through rocshmem_ptr once, so a
      // stub there yields no valid delta and every peer stays unmapped.
      //
      // Fold straight into the peer's gather slot as well as my own. peer_slot
      // is this rank's slot in the PEER's copy of the symmetric buffer, so
      // after the fold both GPUs hold my partial and nothing further has to be
      // transferred. For EP_WORLD_SIZE == 2 there is exactly one peer, which is
      // the configuration this path serves; wider worlds fall back to the
      // staged put below.
      //
      // One delta, two addresses. The gather slot and the signal line are both
      // symmetric-heap objects and the local->peer offset is heap-wide (see
      // mpk_comm.cuh), so translating the signal below costs an add and no
      // second lookup. Resolving both here also means the per-layer cost of the
      // translation is that add, not a walk down the rocSHMEM context.
      __hip_bfloat16 *peer_slot = nullptr;
      int64_t ep_peer_delta = 0;
#if MPK_EP_ABLATE != 1
      if constexpr (EP_WORLD_SIZE == 2) {
        if (mpk_shmem_peer_delta(1 - EP_MY_PE, &ep_peer_delta)) {
          peer_slot = reinterpret_cast<__hip_bfloat16 *>(
              reinterpret_cast<char *>(ep_gather + EP_MY_PE * EP_SLOT_ELEMS) +
              ep_peer_delta);
        }
      }
#endif
      // Whether the direct path is live has to be a work-group-wide decision,
      // not a per-thread one: the staged fallback below is a work-group
      // collective and every thread must agree on whether to enter it.
      bool const ep_direct = (peer_slot != nullptr);
#ifdef MPK_EP_SIG_DBG
      // Which of the two publication paths is actually live. The direct one
      // needs mpk_shmem_peer_ptr to hand back a usable mapping of the peer's
      // symmetric allocation; if it returns null every layer silently goes
      // through the staged putmem_signal instead, and no amount of tuning the
      // direct path changes anything.
      if (tid == 0 && xcd_id == 0 && layer_counter == 0) {
        printf("[EPPATH] pe=%d ep_direct=%d peer_slot=%p\n",
               EP_MY_PE, (int)ep_direct, (void *)peer_slot);
      }
#endif
      // This XCD's slice only. All 8 run concurrently on disjoint columns.
      _full_layer_ep_fold_partial<QKV_BATCH_SIZE,
                                  QKV_REDUCTION_SIZE,
                                  QKV_REDUCTION_SIZE,
                                  (EP_MY_PE == EP_FOLD_PE)>(
          output_ptrs[10],                      // moe_workspace_f32
          output_ptrs[5],                       // attn_proj_out (residual)
          ep_gather + EP_MY_PE * EP_SLOT_ELEMS, // my slot
          peer_slot,                            // peer's copy of my slot
          ep_col_lo,
          ep_col_hi);
      __syncthreads();
      // No threadfence_gpu() here. Every store the fold makes -- the local slot,
      // the peer's slot, and the workspace zeroing -- is now write-through, so
      // there is nothing sitting in this XCD's L2 for an agent-scope release to
      // publish. What remains is the part that was always load-bearing:
      // s_waitcnt vmcnt(0) retires all 256 threads' stores before tid 0's
      // arrival atomic below, so a consumer that observes the count observes
      // the bytes.
      //
      // The fence was `buffer_wbl2 sc1` on gfx950, an L2->HBM writeback of the
      // whole cache paid by all 8 folding workgroups on all 36 layers -- the
      // same cost the Phase 4/5 chunk barrier removed for the same reason (see
      // the note on db48239 above).
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

#if MPK_EP_ABLATE != 1
      if (ep_direct) {
        // My slice is in the peer's memory, ordered ahead of the arrival by the
        // fence above. Count the slice in; the 8th one to land tells the peer.
        //
        // The signal is a store of the run-monotonic count rather than an
        // accumulate, so it stays idempotent and the consumer's threshold is
        // unchanged from the staged path -- but it must be issued only once the
        // whole payload is there, hence the local count first.
        //
        // One signal per PE, not one per (PE, XCD). Splitting it so each XCD
        // pair handshakes privately is the obvious next move -- the consumer
        // below reads only its own columns -- and it deadlocks. Left as one
        // signal until that is understood; see the note on the consumer.
        if (tid == 0) {
          int prev_f = atom_add_release_gpu_s32(ep_fold_done, 1);
          if (prev_f % 8 == 7) {
            // Same heap-wide delta resolved above; ep_direct being true is
            // what makes it valid.
            uint64_t *peer_sig = reinterpret_cast<uint64_t *>(
                reinterpret_cast<char *>(ep_signal +
                                         (size_t)EP_MY_PE *
                                             FULL_LAYER_EP_SIGNAL_STRIDE) +
                ep_peer_delta);
            st_wt_u64((void *)peer_sig, (unsigned long long)ep_sig_expected);
            // Same store, local copy: this thread has just observed all 8
            // local slices, which is precisely what a worker leaving this
            // layer needs to know about its OWN rank's slot. Publishing it on
            // the local signal line rather than a separate flag means the
            // consumer's wait is two loads in one loop instead of two
            // rendezvous -- and it is written by the thread that already knew
            // the answer, so it adds no hop.
            st_wt_u64((void *)(ep_signal + (size_t)EP_MY_PE *
                                               FULL_LAYER_EP_SIGNAL_STRIDE),
                      (unsigned long long)ep_sig_expected);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          }
        }
      } else {
        // No direct mapping (or world > 2): stage it. putmem_signal is a
        // work-group collective and sends the whole slot, so only one XCD may
        // issue it, and only after every slice has been folded locally.
        __shared__ int s_ep_put;
        if (tid == 0) {
          int prev_f = atom_add_release_gpu_s32(ep_fold_done, 1);
          s_ep_put = (prev_f % 8 == 7) ? 1 : 0;
        }
        __syncthreads();
        if (s_ep_put && tid == 0) {
          st_wt_u64((void *)(ep_signal +
                             (size_t)EP_MY_PE * FULL_LAYER_EP_SIGNAL_STRIDE),
                    (unsigned long long)ep_sig_expected);
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        }
        if (s_ep_put) {
          asm volatile("buffer_inv" ::: "memory");
          for (int p = 0; p < EP_WORLD_SIZE; p++) {
            if (p == EP_MY_PE) {
              continue;
            }
            mpk_putmem_signal_block(
                ep_gather + EP_MY_PE * EP_SLOT_ELEMS,
                ep_gather + EP_MY_PE * EP_SLOT_ELEMS,
                EP_SLOT_BYTES,
                ep_signal + (size_t)EP_MY_PE * FULL_LAYER_EP_SIGNAL_STRIDE,
                1,
                MPK_SIGNAL_ADD,
                p);
          }
        }
      }
#else
      (void)ep_direct;
      if (tid == 0) {
        atom_add_release_gpu_s32(ep_fold_done, 1);
      }
#endif
    }

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
    _ep_t2 = __builtin_amdgcn_s_memrealtime();
#endif

    MPK_WS_PHASE(94, qkv_epoch_expected, xcd_id);
    // ── 9d: wait for THIS rank's slot to be complete. No reduce. ────────
    //
    // There is no reduce pass here any more and no exit barrier after it. The
    // next layer's QKV prologue reads the gather buffer directly and sums the
    // slots as part of the pass it already makes over that vector
    // (EP_PEER_SLOTS in gang_rmsnorm_linear_mxfp4_bias_mi300.cuh).
    //
    // What that removes is a hop, not just work. The old shape was
    // peer -> combiner (9d) -> everyone (9e): the release could not start
    // until a reduce it gated had finished, so every worker paid the peer
    // latency plus a reduce plus a second GPU-internal round trip. Now the
    // reduce lives in a consumer that was already loading those bytes and
    // there is nothing to release.
    //
    // The PEER's slot is not waited on here. Its only reader is the next
    // layer's QKV prologue, which is 8 of 240 workers, so that wait belongs at
    // the point of use (Phase 1, gated on EP_PREV_SLOTS) rather than at the
    // layer boundary where it stalls 232 workers that never touch those bytes.
    // Per-layer gather buffers are what make that safe: layer i's buffer is
    // written only in layer i and read only in layer i+1, so a worker running
    // ahead cannot clobber a slot the peer has yet to fill.
    //
    // Every worker, not workgroup 0: with no combiner there is no one to
    // release the others, and nothing to release them for.
    {
      if (tid == 0) {
        // This rank's OWN slot: its 8 column slices are written by 8 different
        // XCDs, so a worker cannot know the local half is complete from its own
        // store alone. The gate is published on this rank's signal line by the
        // same thread that observed the 8th local fold, so it costs a load and
        // no extra rendezvous -- and unlike the peer's line it is purely
        // on-GPU, no XGMI round trip in the critical path.
        //
        // It also covers the two WAR hazards the old exit barrier covered: the
        // fold reads attn_proj_out (which the next layer's Phase 7 overwrites)
        // and zeroes moe_workspace_f32 (which the next layer's Phase 8
        // atomicAdds into). Both are ordered behind this signal, and both are
        // strictly local -- another reason the peer's line does not belong
        // here.
#if MPK_EP_ABLATE != 1 && MPK_EP_ABLATE != 5
#if MPK_EP_ABLATE == 6
        // Only the folding workgroups wait. See the knob comment above.
        if (xcd_rank == 0) {
#endif
        uint64_t *self_sig =
            ep_signal + (size_t)EP_MY_PE * FULL_LAYER_EP_SIGNAL_STRIDE;
        while (ld_nt_u64(reinterpret_cast<unsigned long long *>(self_sig)) <
               (unsigned long long)ep_sig_expected) {
          __builtin_amdgcn_s_sleep(1);
        }
        // The peer's slot, waited on here rather than at its point of use in
        // the next layer's QKV prologue. Both ranks run the same layer on the
        // same schedule, so by the time a worker reaches this line the peer's
        // store has effectively already landed and the poll is free; pushing
        // it into Phase 1 instead puts it upstream of the QKV barrier that
        // gates all of attention, and measured 2.542 vs 2.528 ms/token.
        //
        // MPK_EP_WAIT_AT_USE=1 moves it to Phase 1 anyway, to re-price that
        // measurement now that the fold is write-through and the two L2
        // writebacks are gone. The last layer is the exception: its consumer
        // is the tail task, not a QKV prologue, so there is nowhere downstream
        // to move the wait to and EP_WRITE_COMBINED reads all slots right
        // below. It keeps the wait here under either setting.
#if !MPK_EP_WAIT_AT_USE
        _full_layer_ep_wait_peers<EP_WORLD_SIZE, EP_MY_PE>(ep_signal,
                                                           ep_sig_expected);
#else
        if constexpr (EP_WRITE_COMBINED) {
          _full_layer_ep_wait_peers<EP_WORLD_SIZE, EP_MY_PE>(ep_signal,
                                                             ep_sig_expected);
        }
#endif
#if MPK_EP_ABLATE == 6
        }
#endif
#else
        (void)ep_signal;
        (void)ep_sig_expected;
#endif
      }
      __syncthreads();
      asm volatile("buffer_inv" ::: "memory");
      // For every layer but the last, nothing else happens here. There used to
      // be a reduce writing output_ptrs[11], then a release flag, then an exit
      // barrier every worker waited on; all three are gone, because the sum is
      // produced by the NEXT layer's QKV prologue straight out of the gather
      // buffer (EP_PEER_SLOTS) -- so there is no combined vector to publish and
      // no window between publishing and consuming for a barrier to protect.
      //
      // The last layer is the exception: its consumer is the tail (final
      // RMSNorm + LM head), a separate task with a plain 2-D input and no
      // slot-summing prologue to fold into. So exactly one layer out of 36
      // still materializes the sum and still needs the exit barrier that makes
      // it visible. 1/36 of the old cost.
      if constexpr (EP_WRITE_COMBINED) {
        __hip_bfloat16 *ep_out =
            reinterpret_cast<__hip_bfloat16 *>(output_ptrs[11]);
        // EXACTLY the columns this XCD folded, on every row. That identity is
        // what lets the wait above be a per-slot poll rather than a full
        // cross-XCD rendezvous: this workgroup consumes only its own local
        // store and the matching remote slice.
        for (int row = 0; row < QKV_BATCH_SIZE; ++row) {
          size_t const row_base = (size_t)row * QKV_REDUCTION_SIZE;
          for (int c = ep_col_lo + tid; c < ep_col_hi; c += blockDim.x) {
            size_t idx = row_base + c;
            float acc = 0.0f;
            for (int p = 0; p < EP_WORLD_SIZE; p++) {
              acc += (float)ep_gather[(size_t)p * EP_SLOT_ELEMS + idx];
            }
            __hip_bfloat16 v = (__hip_bfloat16)acc;
            unsigned short bits;
            __builtin_memcpy(&bits, &v, 2);
            st_wt_u16((void *)&ep_out[idx], bits);
          }
        }
        __syncthreads();
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        threadfence_gpu();
        if (tid == 0) {
          int prev_c = atom_add_release_gpu_s32(ep_combine_done, 1);
          if (prev_c % 8 == 7) {
            for (int x = 0; x < 8; x++) {
              st_wt_u32(
                  (void *)&ep_combine_done[(1 + x) * FULL_LAYER_EP_XCD_STRIDE],
                  (unsigned)ep_expected);
            }
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          }
        }
      }
    }

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
    _ep_t3 = __builtin_amdgcn_s_memrealtime();
#endif

    MPK_WS_PHASE(95, qkv_epoch_expected, xcd_id);
    // ── 9e: exit barrier, last layer only. ───────────────────────────────
    //
    // This used to run on every layer: the combiner published a per-XCD release
    // flag once its slice of the reduce had landed and all 240 workers polled
    // it. It existed because the reduce was a distinct step producing a
    // distinct buffer (output_ptrs[11]) that the next layer read, so the layer
    // needed an exit fence between "combined" and "consumed".
    //
    // With the reduce dissolved into the next layer's QKV prologue there is no
    // such buffer and no such window. Everything a worker must establish before
    // leaving -- both slots of the gather buffer complete, hence the fold's
    // reads of attn_proj_out and its zeroing of moe_workspace_f32 done -- is
    // established by the two-slot wait in 9d, which every worker takes. The
    // release chain went from
    //   peer -> combiner -> everyone  (two sequential hops)
    // to
    //   peer -> everyone              (one, polled in parallel by all 240).
    //
    // It survives only on the last layer, where output_ptrs[11] IS written and
    // a separate tail task reads it.
    if constexpr (EP_WRITE_COMBINED) {
      if (tid == 0) {
        while (ld_nt_s32(&ep_combine_done[(1 + xcd_id) *
                                          FULL_LAYER_EP_XCD_STRIDE]) <
               ep_expected) {
          __builtin_amdgcn_s_sleep(1);
        }
      }
      __syncthreads();
    }
    asm volatile("buffer_inv" ::: "memory");
#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
    _ep_t4 = __builtin_amdgcn_s_memrealtime();
#endif
  }

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  __syncthreads();
  // Phase 9 breakdown. Still printed from two roles: xcd_rank==0 also FOLDS
  // (9b/9c) while every other rank only waits, so their 9bc/9d split differs
  // even though both now take the same 9d wait. 9e is gone and reads ~0.
  if constexpr (EP_WORLD_SIZE > 1) {
    // Accumulate, do NOT print per layer. Printing here fires 36 layers x 16
    // reporting workgroups x 126 iters = ~146k printfs and takes the iteration
    // from 2.5 ms to 441 ms -- the breakdown then describes the printf, not the
    // collective. Accumulate into globals and dump once, from the hot path,
    // deep enough into decode that prefill is out of the average.
    if (tid == 0 && (xcd_rank == 0 || xcd_rank == 1) && _ep_t0 > 0) {
      int r = (xcd_rank == 0) ? 0 : 1;
      atomicAdd(&g_ep9_ns[r][0], (unsigned long long)(_ep_t1 - _ep_t0));
      atomicAdd(&g_ep9_ns[r][1], (unsigned long long)(_ep_t2 - _ep_t1));
      atomicAdd(&g_ep9_ns[r][2], (unsigned long long)(_ep_t3 - _ep_t2));
      atomicAdd(&g_ep9_ns[r][3], (unsigned long long)(_ep_t4 - _ep_t3));
      unsigned long long n = atomicAdd(&g_ep9_cnt[r], 1ull);
      // 8 workgroups per role per layer x 36 layers = 288 per iteration.
      // Dump at iteration ~100, well past the 72 prefill tokens.
      if (r == 0 && n == 288ull * 100ull) {
        for (int rr = 0; rr < 2; rr++) {
          unsigned long long c = g_ep9_cnt[rr];
          if (c == 0) {
            continue;
          }
          // Ticks -> us per layer-instance: 1 tick = 10 ns.
          printf("[EP9] role=%s n=%llu per_layer_us 9a=%.3f 9bc=%.3f "
                 "9d=%.3f 9e=%.3f tot=%.3f | x36 ms=%.3f\n",
                 rr == 0 ? "folder" : "follower",
                 c,
                 (double)g_ep9_ns[rr][0] / c / 100.0,
                 (double)g_ep9_ns[rr][1] / c / 100.0,
                 (double)g_ep9_ns[rr][2] / c / 100.0,
                 (double)g_ep9_ns[rr][3] / c / 100.0,
                 (double)(g_ep9_ns[rr][0] + g_ep9_ns[rr][1] + g_ep9_ns[rr][2] +
                          g_ep9_ns[rr][3]) /
                     c / 100.0,
                 (double)(g_ep9_ns[rr][0] + g_ep9_ns[rr][1] + g_ep9_ns[rr][2] +
                          g_ep9_ns[rr][3]) /
                     c / 100.0 * 36.0 / 1000.0);
        }
      }
    }
  }
  // Phases 1-8, accumulated and dumped once -- the same treatment as the EP9
  // block above and for the same reason. This is the breakdown for the 1.923 ms
  // that Phase 9 does NOT explain: until now the only per-phase numbers came
  // from the per-layer [FUSED_PHASE] printf below, which perturbs the iteration
  // by ~200x and so cannot be read even relatively.
  //
  // One reporting workgroup per XCD (xcd_rank == 0), which is the same worker
  // the printf below uses, so the two describe the same thing at very different
  // cost. That worker holds a real W13 tile in Phase 8, so its p8 is the
  // W13 -> barrier -> W2 chain rather than a padding tile's early exit.
  {
    unsigned long long _fp_t4 = __builtin_amdgcn_s_memrealtime();
    if (tid == 0 && xcd_rank == 0 && _fused_t0a > 0 && _fused_t0b > 0) {
      atomicAdd(&g_fp_ns[0], (unsigned long long)(_fused_t0a - _fused_t0));
      atomicAdd(&g_fp_ns[1], (unsigned long long)(_fused_t0b - _fused_t0a));
      // attn / merge only exist on workers that ran an attention chunk.
      if (_fused_t0c > 0) {
        atomicAdd(&g_fp_ns[2], (unsigned long long)(_fused_t0c - _fused_t0b));
        if (_fused_t0d > 0) {
          atomicAdd(&g_fp_ns[3], (unsigned long long)(_fused_t0d - _fused_t0c));
          atomicAdd(&g_fp_ns[4], (unsigned long long)(_fused_t1 - _fused_t0d));
        } else {
          atomicAdd(&g_fp_ns[4], (unsigned long long)(_fused_t1 - _fused_t0c));
        }
      }
      atomicAdd(&g_fp_ns[5], (unsigned long long)(_fused_t2 - _fused_t1));
      atomicAdd(&g_fp_ns[6], (unsigned long long)(_fused_t3 - _fused_t2));
      atomicAdd(&g_fp_ns[7], (unsigned long long)(_fp_t4 - _fused_t3));
      unsigned long long fn = atomicAdd(&g_fp_cnt, 1ull);
      // 8 XCDs x 36 layers = 288 per iteration; dump at ~iteration 100, past
      // the 72 prefill tokens. Same cadence as the EP9 dump.
      if (fn == 288ull * 100ull) {
        double c = (double)g_fp_cnt;
        // Ticks -> us per layer-instance: 1 tick = 10 ns, so /100.
        double v[8];
        for (int k = 0; k < 8; k++) {
          v[k] = (double)g_fp_ns[k] / c / 100.0;
        }
        double tot = 0;
        for (int k = 0; k < 8; k++) {
          tot += v[k];
        }
        printf("[FP18] n=%.0f per_layer_us qkv_gemm=%.3f qkv_bar=%.3f "
               "attn=%.3f merge=%.3f wait=%.3f xcd_bar=%.3f oproj_topk=%.3f "
               "moe=%.3f | tot=%.3f x36ms=%.3f\n",
               c,
               v[0],
               v[1],
               v[2],
               v[3],
               v[4],
               v[5],
               v[6],
               v[7],
               tot,
               tot * 36.0 / 1000.0);
      }
    }
  }

  // MPK_EP9_ONLY suppresses the per-layer full-phase dump so the Phase 9
  // accumulators above can be read from a run that still executes at ~2.5 ms.
  // With both live the printf volume alone costs 440 ms/iter.
#ifndef MPK_EP9_ONLY
  if (tid == 0 && xcd_rank == 0) {
    unsigned long long _fused_t4 = __builtin_amdgcn_s_memrealtime();
    double p1_5 = (double)(_fused_t1 - _fused_t0) * 10.0 / 1000.0;
    double p6 = (double)(_fused_t2 - _fused_t1) * 10.0 / 1000.0;
    double p7 = (double)(_fused_t3 - _fused_t2) * 10.0 / 1000.0;
    double p8 = (double)(_fused_t4 - _fused_t3) * 10.0 / 1000.0;
    double total = (double)(_fused_t4 - _fused_t0) * 10.0 / 1000.0;
    // Fine-grained sub-ops (xcd_rank==0 is a QKV+attn worker)
    double qkv_gemm = (_fused_t0a > 0)
                          ? (double)(_fused_t0a - _fused_t0) * 10.0 / 1000.0
                          : -1;
    double qkv_barrier = (_fused_t0b > 0)
                             ? (double)(_fused_t0b - _fused_t0a) * 10.0 / 1000.0
                             : -1;
    double attn = (_fused_t0c > 0)
                      ? (double)(_fused_t0c - _fused_t0b) * 10.0 / 1000.0
                      : -1;
    double merge_only = (_merge_done > 0)
                            ? (double)(_merge_done - _fused_t0c) * 10.0 / 1000.0
                            : -1;
    double flush_sig = (_fused_t0d > 0 && _merge_done > 0)
                           ? (double)(_fused_t0d - _merge_done) * 10.0 / 1000.0
                           : -1;
    double merge_flush = (_fused_t0d > 0)
                             ? (double)(_fused_t0d - _fused_t0c) * 10.0 / 1000.0
                             : -1;
    double wait_others =
        (_fused_t0d > 0)   ? (double)(_fused_t1 - _fused_t0d) * 10.0 / 1000.0
        : (_fused_t0c > 0) ? (double)(_fused_t1 - _fused_t0c) * 10.0 / 1000.0
                           : -1;
    printf("[FUSED_PHASE] xcd=%d qkv_attn=%.1f xcd_barrier=%.1f "
           "oproj_topk=%.1f moe=%.1f total=%.1f"
           " | qkv_gemm=%.1f qkv_bar=%.1f attn=%.1f merge=%.1f(m=%.1f+f=%.1f) "
           "wait=%.1f\n",
           xcd_id,
           p1_5,
           p6,
           p7,
           p8,
           total,
           qkv_gemm,
           qkv_barrier,
           attn,
           merge_flush,
           merge_only,
           flush_sig,
           wait_others);
  }
#endif // MPK_EP9_ONLY
#endif
  MPK_TW_SUB(90, tile_idx);
  MPK_WS_PHASE(90, qkv_epoch_expected, xcd_id);
}

} // namespace kernel
