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

// The attention half of a GLM decoder layer in one gang task: input RMSNorm +
// [q_a | kv_a] projection, q_a RMSNorm + absorbed q_b + latent KV-cache append,
// absorbed MLA decode, and the split-KV merge.
//
// Companion to gang_oproj_router_fused_mi300.cuh, which carries the other half
// (o_proj + post-attention RMSNorm + router + TopK + MoE). Together they put a
// decoder layer in two tasks -- gpt-oss's shape, and the reason
// gang_full_layer_fused_mi300.cuh exists there. GLM was running nine tasks per
// layer; the MoE fusion took that to five and 5.362 -> 4.973 ms/token, and
// these four attention dispatches are the rest of the way.
//
// Structure is the same wrapper-over-existing-kernels shape as the MoE half.
// No sub-kernel is rewritten; each is given a WRITE_THROUGH epilogue for the
// case where its consumer now sits across a barrier instead of across a task
// graph event.
//
// Barriers. Three Mechanism-C barriers in one counter tensor of 29 * 16 int32,
// laid out exactly like the MoE task's:
//
//   [0 .. 8]    qkv_a  -> q_b.     flags at [x * 16],        counter [8 * 16]
//   [10 .. 18]  q_b    -> decode.  flags at [(10 + x) * 16], counter [18 * 16]
//   [20 .. 28]  decode -> merge.   flags at [(20 + x) * 16], counter [28 * 16]
//
// Every dispatched worker arrives at all three -- unlike the MoE task, where
// splitting the router's barrier off is what lets the MoE-only workers skip
// it. Here the widest phase is q_b (28 tiles per XCD on GLM-4.7-Flash) and the
// task is dispatched at exactly that width, so "every worker" and "every
// producer" are the same set and there is nothing to split. What the narrow
// phases do instead is stop *waiting* early: only the decode's ranks wait on
// the q_b barrier, and past the decode barrier only the merge's four ranks per
// XCD remain -- the other 24 return to the scheduler a whole merge stage
// ahead of them.
//
// Cache discipline, same rule as the MoE half: same-XCD producer/consumer
// pairs use ordinary stores plus s_waitcnt, cross-XCD pairs use st_wt and a
// consumer-side buffer_inv. Unlike gpt-oss -- where kv_head == xcd_id keeps
// the whole attention chain XCD-local -- every producer here is read by every
// XCD, because MLA has a single shared latent head and the split is over the
// sequence instead:
//
//   qkv_a_out    the q_a columns are read by all 8 q_b GEMMs, and the latent
//                columns, which live on XCDs 4 and 5, are read by XCD 0's
//                tile 0.
//   q_workspace  a q group is 16 heads wide and the q_b GEMM spreads 3 heads
//                per XCD, so one decode tile reads six XCDs' output.
//   kv_cache     written by XCD 0's tile 0 alone, read by all 8.
//   o_acc / lse  XCD x computes chunk x; a merge task reduces all 8 chunks.
//
// So all three producing kernels run with WRITE_THROUGH=true. The merge's own
// output is not write-through by default: it is consumed by the next task
// across a real event boundary, which does the buffer_wbl2 itself. The knob
// is still plumbed, since it measured a dead heat as a standalone task and is
// worth re-measuring from inside here.
//
// Dispatch: tiles_per_xcd is the max over the four phases, and has to stay
// under the resident worker count or the in-kernel barriers deadlock. tile_idx
// is global -- this task registers as a variant of TASK_GANG_MLA_DECODE_MI300,
// which is in runtime.cc's n_tile_start list -- so tile_idx = xcd_id *
// tiles_per_xcd + xcd_rank. The two GEMM phases index tiles *within* an XCD
// and are handed xcd_rank; the decode and the merge want a global work-item
// index, which is synthesized from (xcd_id, xcd_rank) against their own
// per-XCD width rather than the dispatch width.

// A fourth barrier appears when W_UK is un-absorbed from q_b (Phase 3b), and
// it is the only XCD-local one in either monolith. q_b's workgroups and the
// W_UK GEMV's tiles are both head-aligned onto the same eight heads per XCD --
// 32 workgroups of four and 64 tiles of eight, at GLM-5's 64 heads -- so the
// producer and the consumer of the nope scratch are always on one chiplet and
// the release never has to wait on another. Its counters live in a region of
// their own: for XCD x, the flag at [x * 16] and the arrival counter at
// [x * 16 + 8].
//
#pragma once
#include "tasks/ampere/merge_splitkv.cuh"
#include "tasks/mi300/mpk_bsdbg.cuh"
#include "tasks/mi300/gang_gemv_mxfp8_mi300.cuh"
#include "tasks/mi300/gang_mla_decode_mi300.cuh"
#include "tasks/mi300/gang_rmsnorm_linear_mxfp8_bias_mla_kvupd_mi300.cuh"

namespace kernel {

// Self-heal gate for the Mechanism-C flag polls. Same value and same
// reasoning as the copy in gang_mla_full_layer_fused_mi300.cuh, which carries
// the full note; duplicated under #ifndef because the two monoliths land in
// this translation unit in include order and either one may be first.
#ifndef MPK_FL_REPUBLISH_SPINS
#define MPK_FL_REPUBLISH_SPINS 1024
#endif

// Layout of the symmetric EP signal array, in uint64 units. Duplicated from
// FULL_LAYER_EP_SIGNAL_STRIDE rather than read from it, for the same reason
// OPROJ_EP_SIGNAL_STRIDE is: the monoliths land in this translation unit in
// include order and any of them may be first. gang_mla_full_layer_fused_mi300
// static_asserts that all three agree, from a point where all are in scope.
//
// One 64-byte line per PE, written only by that PE (on every peer's copy), so
// three independent per-layer signals share it with no false sharing: slot 0
// is Phase 9's MoE fold, slot 1 the o_proj all-gather, slot 2 this task's
// head-sharded q_b/W_UK all-gather.
static constexpr int QB_EP_SIGNAL_STRIDE = 8;
static constexpr int QB_EP_SIGNAL_SLOT = 2;

template <int BATCH_SIZE,
          // ── qkv_a: input_layernorm + [q_a_proj | kv_a_proj_with_mqa] ──
          int QKV_OUTPUT_PER_WG,
          int QKV_REDUCTION_SIZE, // == hidden_size
          int QKV_ACTUAL_HIDDEN,  // RMSNorm divisor, <= QKV_REDUCTION_SIZE
          // ── q_b: q_a_layernorm + absorbed q_b_proj + latent append ──
          int QB_OUTPUT_PER_WG,  // >= QK_ROPE_HEAD_DIM, divides the head span
          int QB_REDUCTION_SIZE, // == q_lora_pad
          int QB_ACTUAL_HIDDEN,  // == q_lora
          int KV_LORA_RANK,
          int QK_ROPE_HEAD_DIM,
          int KV_INPUT_STRIDE, // qkv_a_out's row width
          int KV_CACHE_STRIDE,
          int MAX_SEQ_LEN,
          int PAGE_SIZE,
          int KV_INPUT_OFFSET, // where the latent starts in the qkv_a row
          // ── decode + merge ──
          int NUM_Q_HEADS,
          int NUM_KV_CHUNKS,
          int Q_WORKSPACE_STRIDE,
          int MERGE_DIM_SPLITS,
          bool MERGE_WRITE_THROUGH,
          // ── expert parallelism ──
          // > 1 turns Phase 1's residual resolve into the cross-rank sum:
          // x_ptr is then the previous layer's symmetric gather buffer,
          // EP_PEER_SLOTS bf16 planes of [batch, QKV_REDUCTION_SIZE], and
          // moe_ws_f32_ptr goes unread. See _rnlm8_resadd_norm_rcp.
          int EP_PEER_SLOTS = 0,
          // ── un-absorbed W_UK ──
          // Both > 0 turn on Phase 3b: q_b emits qk_nope + qk_rope per head
          // instead of kv_lora + qk_rope, and a block-diagonal GEMV applies
          // W_UK afterwards. 0 keeps the absorbed q_b, which is what the
          // dense prologue layers and the standalone dispatch use.
          int QK_NOPE_HEAD_DIM = 0,
          int WUK_ROWS_PER_WG = 0,
          // ── head-sharded q_b + W_UK ──
          // Only used to derive this rank's head base and to address peers;
          // the shard itself is detected from the tile count, exactly as the
          // o_proj one is. Defaulted so the standalone dispatch and every
          // single-rank build are untouched.
          int EP_MY_PE = 0,
          int EP_WORLD_SIZE = 1>
__device__ __attribute__((always_inline)) void gang_mla_attn_fused_kernel_mi300(
    // ── inputs ──
    void const *x_ptr,               // [0]  residual stream
    void const *pre_norm_weight_ptr, // [1]  input_layernorm gamma
    void *pre_norm_scratch_ptr,      // [2]
    void const *qkv_weight_ptr,      // [3]  MXFP8, this XCD's chunk
    void const *qkv_bias_ptr,        // [4]  this XCD's slice
    void const *q_a_norm_weight_ptr, // [5]
    void *q_a_norm_scratch_ptr,      // [6]
    void const *qb_weight_ptr,       // [7]  MXFP8, this XCD's chunk
    void const *qb_bias_ptr,         // [8]  this XCD's slice
    void const *kv_norm_weight_ptr,  // [9]  kv_a_layernorm gamma
    void const *cos_ptr,             // [10]
    void const *sin_ptr,             // [11]
    void *kv_cache_ptr,              // [12] paged latent cache, written
    void *attn_counters_ptr,         // [13] this task's three barriers
    void const *moe_ws_f32_ptr,      // [14] previous layer's MoE accumulator
    // ── outputs ──
    void *qkv_a_out_ptr,   // [0] [q_a | latent], declared whole
    void *q_workspace_ptr, // [1] absorbed queries, declared whole
    void *lse_ptr,         // [2] per-chunk LSE
    void *o_acc_ptr,       // [3] per-chunk f32 partials
    void *attn_out_ptr,    // [4] merged bf16 attention output
    void *x_out_ptr,       // [5] this layer's resolved residual stream
    // ── indptr buffers ──
    int const *qo_indptr,
    int const *kv_indptr,
    int const *kv_indices,
    int const *kv_last_page_len,
    int16_t request_id,
    // ── parameters ──
    int num_active_tokens,
    int qkv_n_wgs_per_xcd,
    int qkv_output_stride,
    int qb_n_wgs_per_xcd,
    int qb_output_stride,
    int mla_tiles_per_xcd,
    int mla_total_work_items,
    int merge_tiles_per_xcd,
    int tiles_per_xcd,
    float scale_s,
    float kv_eps,
    int tile_idx,
    // Release values supplied by a caller that has already snapshotted them.
    // Negative means "snapshot them yourself", which is what the standalone
    // dispatch of this task does. The fused whole-layer task passes real
    // values because by the time this body runs, seven phases deep, the
    // snapshot is no longer behind the previous layer's event boundary -- see
    // gang_mla_full_layer_fused_mi300.cuh.
    int qkv_expected_in = -1,
    int qb_expected_in = -1,
    int decode_expected_in = -1,
    // ── un-absorbed W_UK, Phase 3b; all unused when QK_NOPE_HEAD_DIM == 0 ──
    void const *wuk_weight_ptr = nullptr, // packed MXFP8, this XCD's slice
    void *q_nope_ptr = nullptr,           // [batch, H * (nope + rope)] scratch
    int wuk_tiles_per_xcd = 0,
    int wuk_expected_in = 0,
    void *wuk_counters_ptr = nullptr,
    // Symmetric [EP_WORLD_SIZE * 8] uint64 signal array -- the same object the
    // Phase-9 EP fold and the o_proj all-gather use, at slot 2 of this PE's
    // 64-byte line. Null on the standalone dispatch, and null is also what
    // disables the head shard.
    void *ep_signal_ptr = nullptr) {

  int const tid = threadIdx.x;
  int const xcd_id = tile_idx / tiles_per_xcd;
  int const xcd_rank = tile_idx % tiles_per_xcd;

  // ── bs=2 bisection probe ────────────────────────────────────────────────
  //
  // This task is the only instrument the dense prologue has. MPK_BSDBG_N keys
  // off task_layer_idx, which only exists inside a fused layer, and
  // fuse_full_layer requires layer.is_moe -- so model layers 0-2 are invisible
  // to it, and that is precisely the window the bs=2 break was localized to
  // (the embedding output is bit-identical between the arms, the first fused
  // layer's input is not).
  //
  // These fire on FUSED layers, not on the dense prologue -- do not read them
  // as a dense-layer probe. The premise that they would was wrong and cost a
  // run: GLM_FUSE_ATTN defaults to 0, so `fuse_attn` is false for the three
  // dense layers and true only via `fuse_full_layer`, which requires
  // layer.is_moe. The tell is in the log rather than in the source: x_ptr
  // walks by exactly EP_WORLD_SIZE * BATCH_SIZE * hidden * 2 + 128 bytes per
  // visit, because the fused caller passes input_ptrs[27] -- the per-layer
  // ep_gather buffer -- and not input_ptrs[0]. A dense visit would show the
  // one embed_out pointer, then layer_out twice.
  //
  // For the dense prologue use stages 23/25/26, which sit in the three
  // unfused kernels only those layers dispatch.
  //
  // All three buffers are read at task entry, i.e. behind the previous task's
  // event boundary, so none of them races a producer. x_out and attn_out
  // therefore carry the PREVIOUS layer's values -- which is the useful pairing:
  // x_in(L) against attn_out(L-1) separates a layer's attention half from its
  // MLP half without needing a probe that waits on this task's own phases.
  if (tid == 0 && xcd_id == 0 && xcd_rank == 0) {
    MPK_BSDBG_SEQ(20,
                  x_ptr,
                  QKV_REDUCTION_SIZE,
                  "dl_x_in",
                  BATCH_SIZE,
                  QKV_REDUCTION_SIZE,
                  3);
    MPK_BSDBG_SEQ(21,
                  x_out_ptr,
                  QKV_REDUCTION_SIZE,
                  "dl_x_res_prev",
                  BATCH_SIZE,
                  QKV_REDUCTION_SIZE,
                  3);
    MPK_BSDBG_SEQ(22,
                  attn_out_ptr,
                  NUM_Q_HEADS * KV_LORA_RANK,
                  "dl_attn_out_prev",
                  BATCH_SIZE,
                  NUM_Q_HEADS * KV_LORA_RANK,
                  3);
  }

  // The q_b phase is one tile wider than its workgroup count: tile 0 carries
  // the latent row rather than a GEMM tile. See the kvupd header.
  // MPK_QKV_FOLD_ROWS: put the batch row on the MFMA's output columns instead
  // of on the tile index. At BATCH_SIZE 1 the 16x16x128 scaled MFMA computes
  // 16 output columns and the epilogue reads one of them, so a second row is
  // free in the instruction -- what it is not free in is the tile space, and
  // 2 * qkv_n_wgs_per_xcd tiles is what pushes qkv_a from one grid-stride
  // round to two at bs=2 (S17-S16 went 9.22 -> 17.23 us/layer in the spec
  // decomposition). Folded, the round count is back to one and each tile does
  // both rows' RMSNorm+quant against a single fetch of its weight slab.
  //
  // Not applied to q_b: its tile space is 4 per XCD against 29 workers, so at
  // bs=2 it is still one round and the fold would only move work around.
  //
  // MEASURED NEGATIVE -- off by default, kept as a probe. Paired same-batch
  // BAR_SKEW A/B, one prompt, spec decode at bs=2:
  //
  //     region              ctl      fold     delta
  //     S16->S17 qkv_a   17.150    16.535    -0.615 us/layer
  //     S17->S18          6.919     7.882    +0.963
  //
  // The fold does delete the redundant MFMA -- but only 0.615 us of the ~4 us
  // it removes shows up, and the next region gives it all back, because each
  // folded tile now runs both rows' RMSNorm+quant serially and that lengthens
  // the single-tile critical path. Net +0.348 us/layer; wall +0.244 ms/iter at
  // n=1, inside the 0.26 ms floor and the wrong sign. Output is correct
  // (cross-rank identical, 4/4 ranks agree).
  //
  // The lesson is the standing one: cutting work inside a phase is absorbed.
  // The second decode row's cost at qkv_a is the PER-ROW prologue -- a
  // different token's norm and quant, genuinely not redundant -- not the
  // duplicated weight-side MFMA. Do not port this to o_proj; same shape, same
  // absorption.
  constexpr bool QKV_FOLD_ROWS =
#if defined(MPK_QKV_FOLD_ROWS)
      (BATCH_SIZE > 1);
#else
      false;
#endif
  int const qkv_tiles_per_xcd =
      (QKV_FOLD_ROWS ? 1 : BATCH_SIZE) * qkv_n_wgs_per_xcd;
  int const qb_tiles_per_xcd = BATCH_SIZE * qb_n_wgs_per_xcd + 1;

  constexpr int HIER_STRIDE = 16;
  int *qkv_barrier = static_cast<int *>(attn_counters_ptr);
  int *qb_barrier = qkv_barrier + 10 * HIER_STRIDE;
  int *decode_barrier = qkv_barrier + 20 * HIER_STRIDE;
  int const arrivals = tiles_per_xcd * 8;
  // Two-level arrival for the GPU-wide rendezvous in this task. True by
  // construction here -- `arrivals` is literally tiles_per_xcd * 8 -- but
  // written as the test anyway so the flat fallback is one edit away and the
  // self-heal quota below cannot drift out of step with it.
  bool const bar_tree = MPK_BAR_TREE && (arrivals == tiles_per_xcd * 8);

  // ── pair-local decode -> merge (the gpt-oss CROC port) ──────────────────
  // gpt-oss's chunk barrier is per-XCD because it maps kv_head == xcd_id, so
  // one chiplet owns a head's whole chunk set and the last chunk to arrive
  // just runs the merge inline. MLA has a single latent head, so the default
  // decode map -- item = xcd_id * mla_tiles_per_xcd + t, decomposed by the
  // decode kernel as q_group = item % NUM_Q_GROUPS -- scatters a q_group's
  // NUM_KV_CHUNKS chunks across every XCD, and Phase 6 has to rendezvous
  // GPU-wide. Measured 13.23 us/worker/layer of spin on the 128 merge ranks.
  //
  // The *merge* map is already pair-aligned and nobody noticed: its offset is
  // xcd_id * merge_tiles_per_xcd + t, and merge_tiles_per_xcd is
  // NUM_Q_GROUPS * MERGE_DIM_SPLITS / 8, so offset / MERGE_DIM_SPLITS --
  // which is what the merge kernel reads as its q_group -- collapses to
  // xcd_id * NUM_Q_GROUPS / 8. At GLM-5's four q_groups that is xcd_id / 2:
  // XCDs 2k and 2k+1 both merge q_group k already. Only the decode disagrees.
  //
  // So re-cut the decode instead: XCD pair (2k, 2k+1) computes every chunk of
  // q_group k, half the chunks each. The merge's reads then never leave the
  // pair, and the rendezvous drops from 8 XCDs to 2. Nothing else moves --
  // o_acc and lse are indexed by the decode kernel from its own decomposed
  // (q_group, chunk), so permuting which XCD runs which item is invisible.
  //
  // CLOSED 2026-08-21. This is the whole of the "port gpt-oss's CROC chunk
  // barrier to MLA" item, and it is measured out. PAIR_MERGE cuts the
  // rendezvous from 8 XCDs to 2 and measured **11.074 vs an 11.005 baseline
  // (+0.07, n=1, inside a 0.19 ms spread)** at NUM_KV_CHUNKS=16, having
  // measured +0.6% at 4 chunks back in 69770d6. The remaining variant in the
  // note -- give ONE XCD a whole q_group so the barrier is per-XCD and the
  // last arriver runs the merge inline -- goes 2 -> 1. It cannot pay when
  // 8 -> 2 paid nothing, and it costs the merge a doubling (16 tiles/XCD on 8
  // becomes 32 on 4). Do not build it.
  //
  // The 13.23 us/worker/layer of counted spin below is real and is still not
  // a lever, for the reason that has now closed five separate barrier
  // experiments on this branch: **counted spin at a rendezvous is skew
  // absorption, and narrowing or deleting the rendezvous relocates the skew
  // rather than removing it.** Same verdict as the arrival tree (-44% in the
  // null probe, -0.001 ms real), as narrowing W13's barrier (0.51 ms of
  // counted time deleted, 0.10 ms of wall), and as deleting rendezvous 767
  // outright (8.28 us/layer of UNIFORM spin removed, wall moved 0.003 ms).
  // Price the arrival spread before building anything that touches a barrier.
  constexpr int NUM_Q_GROUPS = NUM_Q_HEADS / 16;
  constexpr int XCDS_PER_GROUP =
      (NUM_Q_GROUPS > 0 && (8 % NUM_Q_GROUPS) == 0) ? 8 / NUM_Q_GROUPS : 1;
#ifdef MPK_GLM_MLA_PAIR_MERGE
  // Needs 8 % NUM_Q_GROUPS == 0 for the pairing to exist at all, and
  // XCDS_PER_GROUP | NUM_KV_CHUNKS so each half of a pair gets a whole number
  // of chunks. Both hold at GLM-5 (4 groups, 4 chunks); neither is checked at
  // runtime because a false one silently drops decode items.
  constexpr bool PAIR_MERGE = (8 % NUM_Q_GROUPS) == 0 &&
                              (NUM_KV_CHUNKS % XCDS_PER_GROUP) == 0;
#else
  constexpr bool PAIR_MERGE = false;
#endif
  int const pair_id = xcd_id / XCDS_PER_GROUP;
  int const pair_half = xcd_id % XCDS_PER_GROUP;
  // Slot 8's line carries the single global arrival counter at offset 0 in
  // the default mode and nothing else, so in pair mode the per-pair counters
  // ride in the same line at 4-int spacing. A run is one mode or the other,
  // and pair 0 reuses the old address. Four atomics on one line is no worse
  // than the one address all 232 workers hit today, and it costs no slot --
  // the standalone dispatch only allocates decode_barrier[0..8].
  int *const _dec_cnt =
      &decode_barrier[8 * HIER_STRIDE + (PAIR_MERGE ? 4 * pair_id : 0)];
  int const dec_arrivals =
      PAIR_MERGE ? tiles_per_xcd * XCDS_PER_GROUP : arrivals;
  // PAIR_MERGE already narrows this rendezvous to two XCDs and moves its
  // counter to a pair-private address inside slot 8, so the eight-way tree
  // neither applies nor addresses the right word; only the GPU-wide default
  // takes it, and there _dec_cnt IS decode_barrier[8 * HIER_STRIDE].
  bool const dec_tree = bar_tree && !PAIR_MERGE;
  constexpr bool UNABSORB_K = QK_NOPE_HEAD_DIM > 0 && WUK_ROWS_PER_WG > 0;
  // Only dereferenced under UNABSORB_K; the caller owns the region and the
  // standalone dispatch does not allocate one.
  int *wuk_barrier = static_cast<int *>(wuk_counters_ptr);

  // ── head-sharded q_b + W_UK (QB_TP) ─────────────────────────────────────
  // With ATTN_DP at batch 1 every rank runs the same 64 heads from the same
  // weights: q_b is 34.6 MB of MXFP8 per rank per layer and W_UK another 6.6,
  // eight copies of one answer. Sharding is head-wise -- rank p owns heads
  // [p * NUM_Q_HEADS/EP, +that) -- because a head is the unit both stages are
  // already blocked on, and because that keeps Phase 3b's barrier XCD-local:
  // one head per XCD instead of eight, producer and consumer still paired.
  //
  // What it actually buys is not bytes but makespan. Replicated, q_b hands an
  // XCD 33 tiles against 29 workers, so 25 of 29 idle a whole 10.28 us tile in
  // the round-two straggler (see the Phase 3b note below), and the tile count
  // is locked by the shape: OPW must divide the 256-wide head span, so per XCD
  // it is 32, 16, 8 or 4 and never 29. Sharded, an XCD owns one head -- four
  // GEMM tiles plus the latent -- and the second round disappears.
  //
  // Detected, not plumbed, exactly like the o_proj shard: "my per-XCD chunk
  // covers a 1/EP-th of the nope row" is unambiguous, since nothing between
  // the whole row and exactly 1/world is a legal packing.
  constexpr int QB_HEAD_SPAN = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM;
  constexpr int QB_QK_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM;
  constexpr int QB_TP_HEADS =
      (EP_WORLD_SIZE > 1) ? (NUM_Q_HEADS / EP_WORLD_SIZE) : NUM_Q_HEADS;
  constexpr int QB_NPEER = (EP_WORLD_SIZE > 1) ? (EP_WORLD_SIZE - 1) : 1;
  bool const qb_tp = UNABSORB_K && (EP_WORLD_SIZE > 1) &&
                     (ep_signal_ptr != nullptr) &&
                     (qb_n_wgs_per_xcd * QB_OUTPUT_PER_WG * 8 ==
                      QB_TP_HEADS * QB_HEAD_SPAN);
  // Both scratch rows stay declared at the full 64 heads; only the weights are
  // sliced, so every offset this rank touches is its own head base plus the
  // per-XCD offset the replicated form already used.
  int const qb_head_base = qb_tp ? EP_MY_PE * QB_TP_HEADS : 0;

  // ── deferred rope ───────────────────────────────────────────────────────
  // Sharded, q_b's makespan is one tile: four of them per XCD against 29
  // workers. Narrowing the tile is the whole remaining lever there, and the
  // only thing that pinned OPW at >= QK_ROPE_HEAD_DIM was that the rotation
  // reads (2j, 2j+1) and writes (j, j + ROPE_HALF) across a head's whole rope
  // slice, so the slice could not straddle two workgroups with no barrier
  // between them. Phase 3b's XCD-local W_UK release *is* such a barrier, so
  // below that width the kvupd kernel leaves the columns un-roped and the
  // last W_UK tile of each head rotates them -- the same workgroup that
  // already carries the head's rope tail into the peers.
  constexpr bool QB_DEFER_ROPE =
      UNABSORB_K && (QB_OUTPUT_PER_WG < QK_ROPE_HEAD_DIM);

  // Release values are read once, up front, before anything in this layer has
  // run -- the same argument as the MoE task's s_expected block. Reading a
  // release value beside its own arrival atomic races within the last
  // arriving block: a straggler wave there can observe its own thread 0's
  // bump, compute expected = published + 1, and spin forever.
  __shared__ int s_expected[3];
  if (qkv_expected_in < 0) {
    if (tid == 0) {
      s_expected[0] = ld_nt_s32(&qkv_barrier[xcd_id * HIER_STRIDE]) + 1;
      s_expected[1] = ld_nt_s32(&qb_barrier[xcd_id * HIER_STRIDE]) + 1;
      s_expected[2] = ld_nt_s32(&decode_barrier[xcd_id * HIER_STRIDE]) + 1;
    }
    __syncthreads();
  }
  // The override is a kernel argument, so the branch above is block-uniform
  // and the __syncthreads inside it is safe.
  int const qkv_expected = qkv_expected_in < 0 ? s_expected[0] : qkv_expected_in;
  int const qb_expected = qb_expected_in < 0 ? s_expected[1] : qb_expected_in;
  int const decode_expected =
      decode_expected_in < 0 ? s_expected[2] : decode_expected_in;

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  // Slot 4 is ATTN: [0]=qkv_a [1]=qkv barrier [2]=q_b+kvupd [3]=q_b barrier
  // [4]=MLA decode [5]=decode barrier [6]=merge [7]=W_UK + its XCD-local
  // barrier, un-absorbed only.
  unsigned long long _sp_t0 = __builtin_amdgcn_s_memrealtime();
#endif

  MPK_WS_PHASE(21, qkv_expected, xcd_id);
  // Stage stamp 16: attention entry: first instruction of the attn half.
  if (tid == 0) {
    mpk_stage_stamp(16);
  }
  // ══════════════════════════════════════════════════════════════════════
  // Phase 1: residual resolve + input RMSNorm + [q_a_proj | kv_a_proj_with_mqa]
  // ══════════════════════════════════════════════════════════════════════
  // x_ptr is the *previous* layer's pre-MoE residual and moe_ws_f32_ptr its
  // MoE's f32 accumulator; their sum is this layer's input, and used to be
  // produced by a MOE_RESIDUAL_ADD_F32 task of its own -- grid_dim (1,1,1),
  // 47 dispatches per token, 239 of 240 workers idle behind a full event
  // boundary. FUSE_RESADD folds it into the prologue that was going to read
  // the row anyway, which is what gpt-oss's
  // gang_resaddf32_rmsnorm_linear_mxfp4_bias_kernel does. The resolved row
  // still goes out to x_out_ptr, because the o_proj that follows this task
  // adds it back as its own residual.
  //
  // The first layer has no MoE behind it; the demo hands it a zeroed
  // workspace rather than a second task variant, at the cost of one 8 KB
  // read of zeros per token.
  // The GEMM addresses its output as [wg_idx * OPW + ...] within an XCD's
  // column chunk, so it wants that chunk's base. qkv_a_out has to be declared
  // whole here -- Phase 3 norms a prefix of it that spans four XCDs, and
  // reads a latent slice that spans two more -- so the slice the partition
  // map used to supply is reconstructed instead, exactly as the MoE task
  // reconstructs o_proj's.
  // Grid-stride, not one tile per worker: at GLM-5's widths a phase can want
  // more tiles per XCD than there are resident workers (o_proj 48, the router
  // 32, W2 108 against 30), and the dispatch width is capped at the worker
  // count because a tile that has to wait for a worker deadlocks the in-kernel
  // barriers. Every phase therefore strides by tiles_per_xcd. Where the count
  // fits -- which is all of them at GLM-4.7-Flash -- the loop runs once and
  // the generated code is what it was.
  // MPK_ABL_QKV: delete the whole Phase 1 tile loop, keeping every barrier and
  // every other phase. WRONG OUTPUT by construction. Companion to
  // MPK_MLA_SKIP_DECODE, for the phase SP4[0]+SP4[1] price at 29.7 us/layer
  // (2.32 ms/token) against 3.2 us/layer of bytes. MPK_ATTN_HALFK already
  // priced halving this stage's bytes AND its MFMAs at ~0, so if this probe
  // also comes back near zero the phase is absorbed by the barrier behind it
  // and no qkv_a lever -- sharding q_a_proj included -- can pay.
  // MPK_QKV_EP_FOLD: hoist the EP reduction out of the tile.
  //
  // The prologue is written as "the consumer IS the reduction", which is right
  // at gpt-oss's shape and wrong at GLM-5's: EP_PEER_SLOTS is 8 and the gang is
  // 24 wide per XCD, so all 192 workgroups per rank sum the same eight 12 KB
  // planes. Measured at 20.7 MB/layer/rank of identical traffic -- more than
  // qkv_a's own 16.1 MB of weight -- and ~50% of a 13.50 us tile.
  //
  // Six workgroups per XCD fold one sixth of the row each, publish it to
  // x_out_ptr (the same buffer, the same bytes, the same rounding `store_x`
  // was already writing), and one XCD-LOCAL release lets the 24 tiles read it.
  // XCD-local, not GPU-wide, on purpose: a rendezvous costs 3.77 us GPU-wide
  // and the whole win here is ~5 us/layer, so a global barrier would eat it.
  // Every XCD folds its own copy to the same addresses -- identical bytes per
  // L2, which is exactly the argument `store_x`'s one-WG-per-XCD store makes.
  // MPK_QKV_PRO_HOIST: hoist the WHOLE prologue, not just the fold.
  //
  // 48fea7f split the tile and the fold is the cheap third of the prologue:
  // resolve + LDS staging + the RMSNorm reciprocal is 6.107 us of a 17.199 us
  // tile, the quantizer another 0.869, and the MFMA K-loop that reads all 98 KB
  // of weight is only 5.407 -- 90% of the per-CU byte roof. So the GEMM is
  // saturated and 41% of the tile is a norm+quant that 192 tiles per layer per
  // rank derive from the same 6144-element row. Hoisting the fold alone
  // measured neutral (c72b559), which is consistent: the fold's 8 planes are
  // L2-resident and were never the expensive part.
  //
  // The same six workgroups per XCD now fold, reduce, normalize and quantize,
  // and publish E4M3 + one E8M0 per 128 into the tail of pre_norm_scratch --
  // the buffer the fused path otherwise leaves unwritten. The tiles copy 6192
  // bytes into LDS instead of staging 24 KB of bf16 and re-deriving. The
  // release below is the fold's, unchanged, so the layer gains no rendezvous.
#if defined(MPK_QKV_PRO_HOIST)
  constexpr bool QKV_PRO_HOIST = (EP_PEER_SLOTS > 1);
#else
  constexpr bool QKV_PRO_HOIST = false;
#endif
#if defined(MPK_QKV_EP_FOLD)
  constexpr bool QKV_EP_FOLD = (EP_PEER_SLOTS > 1) && !QKV_PRO_HOIST;
#else
  constexpr bool QKV_EP_FOLD = false;
#endif
  // Per-XCD publish block in the tail of pre_norm_scratch, past the
  // REDUCTION_SIZE bf16 the buffer nominally holds.
  //
  // Layout, per XCD: BATCH_SIZE consecutive (E4M3 row, E8M0 scales) pairs, then
  // one 256 B line per token holding the PRO_WGS f32 partials and, 128 B on,
  // the PRO_WGS epoch stamps. The tokens must not share stamps -- the second
  // token would find the first's stamps already at this epoch and read its
  // partials. The partials and the stamps get separate 128 B lines for the same
  // reason: an nt read of a stamp must not be served a line that also carries a
  // partial the writer has not stored yet. Only the leading pair is addressed
  // by the consumer, which is why its stride is derivable from REDUCTION_SIZE
  // alone.
  constexpr int PRO_WGS = 6; // 6144 / (256 * 4) = 6 float4 passes
  constexpr int PUB_DATA = QKV_REDUCTION_SIZE;
  constexpr int PUB_SCALES = QKV_REDUCTION_SIZE / 128;
  constexpr int PUB_TOK = PUB_DATA + PUB_SCALES;
  constexpr int PUB_PART_OFF = ((PUB_TOK * BATCH_SIZE + 127) / 128) * 128;
  constexpr int PUB_STRIDE =
      ((PUB_PART_OFF + BATCH_SIZE * 256 + 255) / 256) * 256;
  uint8_t *const pro_pub =
      (uint8_t *)pre_norm_scratch_ptr +
      (size_t)BATCH_SIZE * QKV_REDUCTION_SIZE * 2 + (size_t)xcd_id * PUB_STRIDE;

  if constexpr (QKV_PRO_HOIST || QKV_EP_FOLD) {
    constexpr int FOLD_WGS = PRO_WGS;
    if (xcd_rank < FOLD_WGS) {
      for (int tok = 0; tok < num_active_tokens; tok++) {
        if constexpr (QKV_PRO_HOIST) {
          _rnlm8_pro_publish<QKV_REDUCTION_SIZE,
                             QKV_ACTUAL_HIDDEN,
                             EP_PEER_SLOTS,
                             BATCH_SIZE * QKV_REDUCTION_SIZE,
                             FOLD_WGS>(
              static_cast<unsigned short const *>(x_ptr) +
                  tok * QKV_REDUCTION_SIZE,
              static_cast<unsigned short *>(x_out_ptr) +
                  tok * QKV_REDUCTION_SIZE,
              static_cast<unsigned short const *>(pre_norm_weight_ptr),
              pro_pub + tok * PUB_TOK,
              pro_pub + tok * PUB_TOK + PUB_DATA,
              (int *)(pro_pub + PUB_PART_OFF + tok * 256),
              (int *)(pro_pub + PUB_PART_OFF + tok * 256 + 128),
              xcd_rank,
              qkv_expected,
              /*eps=*/1e-5f);
        } else {
          _rnlm8_ep_fold_slice<QKV_REDUCTION_SIZE,
                               EP_PEER_SLOTS,
                               BATCH_SIZE * QKV_REDUCTION_SIZE,
                               FOLD_WGS>(
              static_cast<unsigned short const *>(x_ptr) +
                  tok * QKV_REDUCTION_SIZE,
              static_cast<unsigned short *>(x_out_ptr) +
                  tok * QKV_REDUCTION_SIZE,
              xcd_rank);
        }
      }
    }
    __syncthreads();
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    if (tid == 0) {
      // Hosted in wuk_barrier's spare lanes: that array is XCD-local already
      // and uses only offsets 0 (flag) and 8 (counter) of its stride-16 block.
      int *const _cnt = &wuk_barrier[xcd_id * HIER_STRIDE + 9];
      int *const _flag = &wuk_barrier[xcd_id * HIER_STRIDE + 1];
      // Stage stamp 45/44: this XCD-local EP-fold rendezvous sits INSIDE the
      // qkv_a span, between the layer-entry release (slot 0) and qkv_a tiles
      // done (slot 17). Charging 0->17 to "busy" would bill this spin as work.
      mpk_stage_stamp(45);
      int prev = atom_add_release_gpu_s32(_cnt, 1);
      if ((prev % tiles_per_xcd) == tiles_per_xcd - 1) {
        st_wt_u32((void *)_flag, (unsigned)qkv_expected);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      }
      MPK_WS_WAIT_BEGIN(769, qkv_expected);
      int _spins = 0;
      int _obs;
      while ((_obs = ld_nt_s32(_flag)) < qkv_expected) {
        ++_spins;
        MPK_WS_WAIT_TICK(_obs, _spins);
        if ((_spins & (MPK_FL_REPUBLISH_SPINS - 1)) == 0) {
          if (ld_nt_s32(_cnt) >= tiles_per_xcd * qkv_expected) {
            st_wt_u32((void *)_flag, (unsigned)qkv_expected);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          }
        }
        __builtin_amdgcn_s_sleep(1);
      }
      mpk_stage_stamp(44);
    }
    __syncthreads();
    asm volatile("buffer_inv" ::: "memory");
  }

#ifndef MPK_ABL_QKV
  // MPK_QKVA_REPS: price an EXTRA un-hidden qkv_a pass. 1 is the shipping
  // path; the long note is at the define in mpk_atoms.cuh. The body is
  // idempotent on the unfolded path, so output stays correct and the wall
  // number is gateable. `#pragma unroll 1` keeps the compiler from cloning
  // the 600-instruction body, which would change the I-cache footprint and
  // confound the byte question with an issue question.
#pragma unroll 1
  for (int _qrep = 0; _qrep < MPK_QKVA_REPS; ++_qrep)
  for (int t = xcd_rank; t < qkv_tiles_per_xcd; t += tiles_per_xcd) {
#ifdef MPK_ENABLE_SUBPHASE_TIMING
    // SP4[0] is 20.06 us/layer and the phase is one grid-stride round, so
    // either a tile really costs 20 us or most of SP4[0] is not tile work.
    // Bank 0 slot 1/2/3 cannot answer it -- its guard is tile_idx == 0 in a
    // kernel four other callers also reach, so its cnt is 31x qkv_a's share.
    // Time the tile here instead, against a count of this loop's own trips:
    // [0][0] is the ns, [2][1] the tiles, so [0][0]/[2][1] is per-tile.
    unsigned long long _sp_qkv0 = __builtin_amdgcn_s_memrealtime();
#endif
    unsigned short *xcd_out =
        static_cast<unsigned short *>(qkv_a_out_ptr) +
        static_cast<size_t>(xcd_id) * qkv_n_wgs_per_xcd * QKV_OUTPUT_PER_WG;
    gang_rmsnorm_linear_mxfp8_bias_kernel<BATCH_SIZE,
                                          QKV_OUTPUT_PER_WG,
                                          QKV_REDUCTION_SIZE,
                                          QKV_ACTUAL_HIDDEN,
                                          /*WRITE_THROUGH=*/true,
                                          /*FUSE_RESADD=*/true,
                                          EP_PEER_SLOTS,
                                          /*EP_PRE_FOLDED=*/QKV_EP_FOLD ||
                                              QKV_PRO_HOIST,
                                          /*SP_QKV=*/true,
                                          /*PRO_PUB=*/QKV_PRO_HOIST,
                                          /*FOLD_ROWS=*/QKV_FOLD_ROWS>(
        // Pre-folded, the resolved row is in x_out_ptr and the gather buffer
        // is not read again.
        /*norm_input_ptr=*/(QKV_EP_FOLD || QKV_PRO_HOIST)
            ? (void const *)x_out_ptr
            : x_ptr,
        pre_norm_weight_ptr,
        pre_norm_scratch_ptr,
        qkv_weight_ptr,
        qkv_bias_ptr,
        xcd_out,
        num_active_tokens,
        qkv_n_wgs_per_xcd,
        qkv_output_stride,
        t,
        moe_ws_f32_ptr,
        x_out_ptr,
        pro_pub);
#ifdef MPK_ENABLE_SUBPHASE_TIMING
    if (tid == 0 && g_subphase_active) {
      atomicAdd(&g_subphase_ns[0][0],
                (__builtin_amdgcn_s_memrealtime() - _sp_qkv0) * 10);
      atomicAdd(&g_subphase_ns[2][1], 1ULL);
    }
#endif
  }
#endif

  MPK_WS_PHASE(22, qkv_expected, xcd_id);
  // Stage stamp 17: qkv_a tiles done.
  if (tid == 0) {
    mpk_stage_stamp(17);
  }
  // ══════════════════════════════════════════════════════════════════════
  // Phase 2: qkv_a -> q_b barrier
  // ══════════════════════════════════════════════════════════════════════
  // __syncthreads is execution-only; s_waitcnt is what retires all 256
  // threads' stores, and it has to precede the release atomic or a waiting
  // XCD can be let through ahead of the data. threadfence_gpu() would also
  // order this and is what a first draft reaches for, but it lowers to
  // buffer_wbl2 sc1 -- a writeback of the whole L2, ~0.16 ms/iter in
  // gpt-oss's measurement. st_wt on the producer side is what makes the
  // cheap fence sufficient.
  __syncthreads();
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#ifdef MPK_ENABLE_SUBPHASE_TIMING
  {
    unsigned long long _t = __builtin_amdgcn_s_memrealtime();
    if (tid == 0 && g_subphase_active) {
      atomicAdd(&g_subphase_ns[4][0], (_t - _sp_t0) * 10);
    }
    _sp_t0 = _t;
  }
#endif
  if (tid == 0) {
    // Modular test rather than a reset: the counter is monotonic for the
    // whole run, so no worker from the next layer can observe a zeroed one.
    if (hier_barrier_arrive(qkv_barrier, HIER_STRIDE, arrivals, tiles_per_xcd,
                            xcd_id, bar_tree, /*skew_slot=*/2)) {
      for (int x = 0; x < 8; x++) {
        st_wt_u32((void *)&qkv_barrier[x * HIER_STRIDE], (unsigned)qkv_expected);
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }
    // Self-heal, see MPK_FL_REPUBLISH_SPINS: the fan-out above is one
    // write-through store per XCD, issued once, by one thread, with no retry.
    // Lose one and every worker on that XCD spins here forever. The arrival
    // counter is the truth -- monotonic at `arrivals` per layer -- so a waiter
    // that has spun past the gate consults it and publishes its own flag.
    int *const _qkv_flag = &qkv_barrier[xcd_id * HIER_STRIDE];
    MPK_WS_WAIT_BEGIN(762, qkv_expected);
    int _spins = 0;
    int _obs;
    while ((_obs = ld_nt_s32(_qkv_flag)) < qkv_expected) {
      ++_spins;
      MPK_WS_WAIT_TICK(_obs, _spins);
      if ((_spins & (MPK_FL_REPUBLISH_SPINS - 1)) == 0) {
        int const _cnt = ld_nt_s32(&qkv_barrier[8 * HIER_STRIDE]);
        // A worker parked here with its spin counter frozen across every host
        // dump is either not executing at all or looping with the heal
        // permanently declined; the two need different fixes and observed/
        // expected cannot tell them apart. aux[0] is the live arrival counter
        // and aux[1] the number of heal rounds this waiter has run, both
        // written from inside the heal, so a frozen aux[1] means the wave
        // stopped and a growing one means the quota test below keeps failing.
        MPK_WS_WAIT_AUX(_cnt, _spins / MPK_FL_REPUBLISH_SPINS, 0, 0);
        if (_cnt >= hier_barrier_heal_quota(arrivals, bar_tree) *
                        qkv_expected) {
          st_wt_u32((void *)_qkv_flag, (unsigned)qkv_expected);
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        }
      }
      __builtin_amdgcn_s_sleep(1);
    }
  }
  __syncthreads();
  asm volatile("buffer_inv" ::: "memory");

  // (The MoE accumulator this layer's W2 will use is zeroed by the o_proj task
  // itself, up front, where the Phase 6 W13->W2 barrier already orders it
  // against the first accumulate. It used to be done here.)
#ifdef MPK_ENABLE_SUBPHASE_TIMING
  {
    unsigned long long _t = __builtin_amdgcn_s_memrealtime();
    if (tid == 0 && g_subphase_active) {
      atomicAdd(&g_subphase_ns[4][1], (_t - _sp_t0) * 10);
    }
    _sp_t0 = _t;
  }
#endif

  MPK_WS_PHASE(23, qkv_expected, xcd_id);
  // Stage stamp 18: qkv_a -> q_b barrier passed.
  if (tid == 0) {
    mpk_stage_stamp(18);
  }
  // ══════════════════════════════════════════════════════════════════════
  // Phase 3: q_a RMSNorm + absorbed q_b + latent KV-cache append
  // ══════════════════════════════════════════════════════════════════════
  // tile_idx 0 is the latent row and returns immediately on every XCD but 0;
  // the GEMM tiles are shifted up by one inside the kernel. Handing it
  // xcd_rank reproduces the standalone dispatch exactly.
  for (int t = xcd_rank; t < qb_tiles_per_xcd; t += tiles_per_xcd) {
#ifdef MPK_ENABLE_SUBPHASE_TIMING
    // The W_UK barrier at [0][4] charges 10.6 us/layer of spin, and q_b's own
    // 33-tiles-on-29-workers straggler only accounts for ~5 of it. Time the
    // tiles here to tell a fat tile from arrival skew: [7][6]/[7][7] are the
    // GEMM tiles and their count, [2][0] is tile 0 -- the latent row and the
    // KV-cache append, which is real work on XCD 0 and a no-op elsewhere.
    unsigned long long _sp_qb0 = __builtin_amdgcn_s_memrealtime();
#endif
    // Un-absorbed, the GEMM's output row is the nope scratch instead of the
    // query row; the rope workgroup crosses back into the query row itself,
    // which is why that one is handed over whole and un-biased.
    //
    // Under QB_TP the base carries this rank's head slice as well. The nope
    // scratch is declared whole on every rank, so rank p's XCD x writes the
    // columns of global head p * QB_TP_HEADS + x and nothing else ever does.
    unsigned short *xcd_q_ws =
        static_cast<unsigned short *>(UNABSORB_K ? q_nope_ptr
                                                 : q_workspace_ptr) +
        static_cast<size_t>(qb_head_base) * QB_HEAD_SPAN +
        static_cast<size_t>(xcd_id) * qb_n_wgs_per_xcd * QB_OUTPUT_PER_WG;
    gang_rmsnorm_linear_mxfp8_bias_mla_kvupd_kernel<BATCH_SIZE,
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
                                                    /*WRITE_THROUGH=*/true,
                                                    QK_NOPE_HEAD_DIM,
                                                    QB_DEFER_ROPE>(
        qkv_a_out_ptr,
        q_a_norm_weight_ptr,
        q_a_norm_scratch_ptr,
        qb_weight_ptr,
        qb_bias_ptr,
        qkv_a_out_ptr, // kv_latent: the same row, at KV_INPUT_OFFSET
        kv_norm_weight_ptr,
        cos_ptr,
        sin_ptr,
        xcd_q_ws,
        kv_cache_ptr,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        request_id,
        num_active_tokens,
        qb_n_wgs_per_xcd,
        qb_output_stride,
        t,
        kv_eps,
        // The kvupd kernel derives the head from xcd_id and its own per-XCD
        // workgroup count, which under QB_TP is a *rank-local* head index.
        // Biasing the pointer by this rank's head base is what makes it land
        // on the global head, and costs the kernel nothing.
        /*q_rope_out_ptr=*/static_cast<unsigned short *>(q_workspace_ptr) +
            static_cast<size_t>(qb_head_base) * QB_QK_DIM);
#ifdef MPK_ENABLE_SUBPHASE_TIMING
    if (tid == 0 && g_subphase_active) {
      unsigned long long _d =
          (__builtin_amdgcn_s_memrealtime() - _sp_qb0) * 10;
      if (t == 0) {
        atomicAdd(&g_subphase_ns[2][0], _d);
      } else {
        atomicAdd(&g_subphase_ns[7][6], _d);
        atomicAdd(&g_subphase_ns[7][7], 1ULL);
      }
    }
#endif
  }

  // ══════════════════════════════════════════════════════════════════════
  // Phase 3b: W_UK, un-absorbed -- q_nope[h] * W_UK[h] -> the query row
  // ══════════════════════════════════════════════════════════════════════
  // Block-diagonal GEMV, KV_LORA_RANK rows per head over a QK_NOPE_HEAD_DIM
  // reduction, exactly the shape the W_UV side runs before o_proj. The
  // barrier ahead of it is XCD-local: see the header note.
  //
  // MEASURED 2026-08-19, GLM-5 NP=8 EP, subphase slot 0. This phase's slot
  // [4][7] is 1.045 ms/iter and reads like a slow GEMV; it is not. Split, it
  // is 0.829 of barrier and 0.216 of GEMV -- a tile is 2.52 us and a worker
  // runs 1.06 of them per layer. The spin is q_b's, not W_UK's: q_b dispatches
  // 33 tiles per XCD (32 GEMM of 10.28 us, plus tile 0's latent append at
  // 2.09) against 29 workers, so 25 of 29 finish a round early and wait
  // 10.3 us. 25/29 * 10.28 = 8.9 of the 12.4 us mean spin; the rest is skew.
  //
  // The tile count is locked by the shape, not by tuning. Per XCD q_b covers
  // 8 heads * 256 columns and QB_OUTPUT_PER_WG must divide the 256-wide head
  // span, so the legal tile counts per XCD are 32, 16, 8, 4 -- never 29. The
  // OPW sweep found 64 best and cannot reach one round. Narrowing loses for
  // the same reason it lost in the MoE (see that header): the tile is already
  // 67% of the 20.2 GB/s per-CU HBM share, so a quarter-width tile costs far
  // more than a quarter. What removes the second round is fewer heads per
  // XCD, i.e. sharding heads across the EP ranks.
  //
  // Merging this phase into q_b's pool with a per-head release instead of the
  // barrier is worth much less than the 12.4 us suggests: heads 0-6 are ready
  // after round 1, but head 7 is produced by tiles 29-32, which are exactly
  // the round-2 tiles, so the makespan floor stays 20.6 + 2.52.
  if constexpr (UNABSORB_K) {
    static_assert(KV_LORA_RANK % WUK_ROWS_PER_WG == 0,
                  "a head's absorbed rows must fill whole GEMV tiles");
    constexpr int TILES_PER_HEAD = KV_LORA_RANK / WUK_ROWS_PER_WG;
    constexpr int QK_DIM_ = KV_LORA_RANK + QK_ROPE_HEAD_DIM;

#ifdef MPK_ENABLE_SUBPHASE_TIMING
    // Close [2] here so it stays "q_b" and [7] is "W_UK + its barrier". Phase
    // 4 still adds to [2] below; with this path on, what it adds is the
    // handful of instructions between the end of this block and its own
    // __syncthreads.
    {
      unsigned long long _t = __builtin_amdgcn_s_memrealtime();
      if (tid == 0 && g_subphase_active) {
        atomicAdd(&g_subphase_ns[4][2], (_t - _sp_t0) * 10);
      }
      _sp_t0 = _t;
    }
#endif

    __syncthreads();
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    if (tid == 0) {
      int *const _cnt = &wuk_barrier[xcd_id * HIER_STRIDE + 8];
      int *const _flag = &wuk_barrier[xcd_id * HIER_STRIDE];
      int prev = atom_add_release_gpu_s32(_cnt, 1);
      if ((prev % tiles_per_xcd) == tiles_per_xcd - 1) {
        st_wt_u32((void *)_flag, (unsigned)wuk_expected_in);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      }
      // Self-heal, see MPK_FL_REPUBLISH_SPINS. The predicate is the whole
      // release condition here, not half of it: this barrier publishes one
      // flag and protects one XCD's writes, so its own counter reaching
      // tiles_per_xcd * expected is exactly what the flag stands for.
      MPK_WS_WAIT_BEGIN(768, wuk_expected_in);
      int _spins = 0;
      int _obs;
      while ((_obs = ld_nt_s32(_flag)) < wuk_expected_in) {
        ++_spins;
        MPK_WS_WAIT_TICK(_obs, _spins);
        if ((_spins & (MPK_FL_REPUBLISH_SPINS - 1)) == 0) {
          if (ld_nt_s32(_cnt) >= tiles_per_xcd * wuk_expected_in) {
            st_wt_u32((void *)_flag, (unsigned)wuk_expected_in);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          }
        }
        __builtin_amdgcn_s_sleep(1);
      }
    }
    __syncthreads();
    asm volatile("buffer_inv" ::: "memory");

#ifdef MPK_ENABLE_SUBPHASE_TIMING
    // [7] above is "W_UK + its barrier", which is not enough to tell an
    // issue-starved GEMV from a straggler on the XCD-local release. Slot 0's
    // spare entries split it: [4] the barrier, [5] the grid-stride GEMV loop,
    // [6] the tiles this worker actually ran, so [5]/[6] is a per-tile figure.
    unsigned long long _sp_wuk = __builtin_amdgcn_s_memrealtime();
    if (tid == 0 && g_subphase_active) {
      atomicAdd(&g_subphase_ns[0][4], (_sp_wuk - _sp_t0) * 10);
    }
    int _wuk_tiles = 0;
#endif

    // The weight is dim-0 partitioned per XCD, so `wuk_weight_ptr` is already
    // this XCD's slice and the GEMV gets the *local* tile index against
    // wuk_tiles_per_xcd. Only the head lookup needs the global index, and the
    // output column is biased into the pointer -- the same shape as Phase 8b
    // on the W_UV side, except that the query row's per-head span (QK_DIM) is
    // wider than the rows a head writes (KV_LORA_RANK), so the bias carries a
    // per-head skip as well.
    // One delta per peer under QB_TP, resolved once for the whole loop. Kept
    // inside this scope rather than at function scope so the seven int64 do
    // not sit live across the decode, which is the register-hungriest phase.
    int64_t qb_peer_delta[QB_NPEER];
    bool qb_all_mapped = qb_tp;
    if (qb_tp) {
#pragma unroll
      for (int q = 0; q < QB_NPEER; q++) {
        qb_peer_delta[q] = 0;
        if (!mpk_shmem_peer_delta((q < EP_MY_PE) ? q : (q + 1),
                                  &qb_peer_delta[q])) {
          // No direct mapping is a fail-loud configuration error, not a slow
          // path: the weights are already sliced, so falling back would just
          // leave seven eighths of the query row stale.
          qb_all_mapped = false;
        }
      }
    }
    bool const qb_push = qb_tp && qb_all_mapped;

    // Rows this task actually carries. BATCH_SIZE is the compile-time width of
    // every scratch row; num_tokens is how many of them hold a live token, and
    // the two differ whenever the graph is built wider than the step being
    // run. At BATCH_SIZE 1 this folds to the constant 1 and nothing below it
    // changes shape -- the bs == 1 codegen is meant to stay identical.
    int qb_rows = 1;
    if constexpr (BATCH_SIZE > 1) {
      int const nt = qo_indptr[request_id + 1] - qo_indptr[request_id];
      qb_rows = nt < BATCH_SIZE ? nt : BATCH_SIZE;
      if (qb_rows < 1) {
        qb_rows = 1;
      }
    }
    // Deferred rope: the position arithmetic the kvupd kernel used to do.
    // Hoisted out of the tile loop because it is four scalar loads and does
    // not depend on the tile; `rope_pos` is the position of token row 0, and
    // row r sits at rope_pos + r, the same walk latent_to_cache does.
    int rope_pos = 0;
    if constexpr (QB_DEFER_ROPE) {
      int const req = request_id;
      int const first_token_pos = qo_indptr[req];
      int const num_tokens = qo_indptr[req + 1] - first_token_pos;
      int const first_page_pos = kv_indptr[req];
      int const global_seq_len =
          (kv_indptr[req + 1] - first_page_pos - 1) * PAGE_SIZE +
          kv_last_page_len[req];
      rope_pos = global_seq_len - num_tokens;
    }

    for (int t = xcd_rank; t < wuk_tiles_per_xcd; t += tiles_per_xcd) {
      int const heads_per_xcd = wuk_tiles_per_xcd / TILES_PER_HEAD;
      int const head = qb_head_base + xcd_id * heads_per_xcd +
                       t / TILES_PER_HEAD;
      unsigned short const *head_in =
          static_cast<unsigned short const *>(q_nope_ptr) +
          static_cast<size_t>(head) *
              (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM);
      // out[t] wants head * QK_DIM + (t % TPH) * ROWS, and the kernel adds
      // t * ROWS, so the difference is the base -- monotonic in t, never
      // stepping below q_workspace_ptr.
      unsigned short *tile_out =
          static_cast<unsigned short *>(q_workspace_ptr) +
          static_cast<size_t>(qb_head_base) * QK_DIM_ +
          static_cast<size_t>(xcd_id) * heads_per_xcd * QK_DIM_ +
          static_cast<size_t>(t / TILES_PER_HEAD) * QK_ROPE_HEAD_DIM;
      // Both strides are the compile-time row widths, not qb_output_stride:
      // the nope scratch and the query row are declared at the full head count
      // on every rank (see the QB_TP note above), and qb_output_stride means
      // different things to this task's two callers. At BATCH_SIZE 1 neither
      // is read at all -- m_tiles is 1, so the row term is zero -- which is
      // exactly why the overloading survived this long.
      gang_gemv_mxfp8_kernel<BATCH_SIZE, QK_NOPE_HEAD_DIM, WUK_ROWS_PER_WG,
                             /*HAS_RESIDUAL=*/false, /*WRITE_THROUGH=*/true,
                             NUM_Q_HEADS * QB_HEAD_SPAN>(
          head_in, wuk_weight_ptr, /*residual=*/nullptr, tile_out,
          num_active_tokens, WUK_ROWS_PER_WG, NUM_Q_HEADS * QK_DIM_,
          /*m_tiles=*/1, wuk_tiles_per_xcd, /*wgm=*/0, t);
      // Push the rows this workgroup just produced straight into every peer's
      // copy of the query row, at the identical offset. The head slices are
      // disjoint across ranks, so the all-gather is QB_NPEER stores of those
      // bytes and needs no staging buffer -- the same shape as the o_proj
      // gather, and pushed here rather than by the barrier leader so all eight
      // XCDs drive the links while W_UK is still running.
      //
      // The last tile of a head carries QK_ROPE_HEAD_DIM more: the roped
      // columns sit immediately after the head's latent rows in the query row,
      // and they were written back in Phase 3 by this same XCD -- which the
      // XCD-local barrier above has already ordered. Contiguous, so it is a
      // wider push and not a second one.
      //
      // ld_nt_s32 is the sc0 sc1 load: the GEMV epilogue and the rope tile are
      // both write-through, so the bytes are in memory but this CU's vL1 may
      // hold the line these lanes read before them.
      // Deferred rotation. The kvupd kernel left this head's rope columns
      // un-roped in the nope scratch because the tile was too narrow to hold
      // the slice; the XCD-local barrier above is the ordering the rotation
      // needed, so it happens here, on the last W_UK tile of the head -- the
      // same workgroup that widens the push below to carry the rope tail.
      // Gated on QB_DEFER_ROPE alone, not on qb_push: a single-rank narrow-OPW
      // build has no push and still needs the query row roped.
      //
      // rope_tile_inplace has a __syncthreads in it, so the guard has to be
      // block-uniform; t and TILES_PER_HEAD both are.
      if constexpr (QB_DEFER_ROPE) {
        if ((t % TILES_PER_HEAD) == TILES_PER_HEAD - 1) {
          using rope_bf16 = gang_mla_kvupd_detail::bf16;
          // One rotation per live token row. Both scratch rows are declared at
          // the full head count on every rank (see the QB_TP note above), so
          // their strides are compile-time -- not qb_output_stride, which the
          // two callers below pass different meanings of and which nothing
          // constrains at BATCH_SIZE 1. rope_tile_inplace has a __syncthreads
          // in it, so the bound has to be block-uniform, which qb_rows is.
          for (int r = 0; r < qb_rows; ++r) {
            gang_mla_kvupd_detail::rope_tile_inplace<QK_ROPE_HEAD_DIM,
                                                     /*WRITE_THROUGH=*/true>(
                reinterpret_cast<rope_bf16 *>(q_nope_ptr) +
                    static_cast<size_t>(r) * NUM_Q_HEADS * QB_HEAD_SPAN +
                    static_cast<size_t>(head) * QB_HEAD_SPAN +
                    QK_NOPE_HEAD_DIM,
                reinterpret_cast<rope_bf16 const *>(cos_ptr) +
                    static_cast<size_t>(rope_pos + r) * QK_ROPE_HEAD_DIM,
                reinterpret_cast<rope_bf16 const *>(sin_ptr) +
                    static_cast<size_t>(rope_pos + r) * QK_ROPE_HEAD_DIM,
                reinterpret_cast<rope_bf16 *>(q_workspace_ptr) +
                    static_cast<size_t>(r) * NUM_Q_HEADS * QK_DIM_ +
                    static_cast<size_t>(head) * QK_DIM_ + KV_LORA_RANK);
          }
        }
      }
      if (qb_push) {
        static_assert((WUK_ROWS_PER_WG % 2) == 0 &&
                          (QK_ROPE_HEAD_DIM % 2) == 0,
                      "peer stores are packed 32-bit, so both the tile and the "
                      "rope tail must be an even number of bf16");
        int const push_w32 =
            (((t % TILES_PER_HEAD) == TILES_PER_HEAD - 1)
                 ? (WUK_ROWS_PER_WG + QK_ROPE_HEAD_DIM)
                 : WUK_ROWS_PER_WG) /
            2;
        __syncthreads();
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        // One push per live token row, at that row's offset in the query row.
        for (int r = 0; r < qb_rows; ++r) {
          unsigned int *const src32 = reinterpret_cast<unsigned int *>(
              tile_out + (size_t)r * NUM_Q_HEADS * QK_DIM_ +
              (size_t)t * WUK_ROWS_PER_WG);
          for (int w = tid; w < push_w32; w += (int)blockDim.x) {
            unsigned int const v =
                (unsigned int)ld_nt_s32(reinterpret_cast<int *>(src32 + w));
            // Unrolled over peers so qb_peer_delta stays in registers: a
            // runtime index into a per-thread array is a scratch spill.
#pragma unroll
            for (int q = 0; q < QB_NPEER; q++) {
              st_wt_u32((void *)(reinterpret_cast<char *>(src32 + w) +
                                 qb_peer_delta[q]),
                        v);
            }
          }
        }
      }
#ifdef MPK_ENABLE_SUBPHASE_TIMING
      ++_wuk_tiles;
#endif
    }
#ifdef MPK_ENABLE_SUBPHASE_TIMING
    {
      unsigned long long _t = __builtin_amdgcn_s_memrealtime();
      if (tid == 0 && g_subphase_active) {
        atomicAdd(&g_subphase_ns[4][7], (_t - _sp_t0) * 10);
        atomicAdd(&g_subphase_ns[0][5], (_t - _sp_wuk) * 10);
        atomicAdd(&g_subphase_ns[0][6], (unsigned long long)_wuk_tiles);
        // Both dump sites skip a slot whose count is zero, and nothing on the
        // GLM path writes g_subphase_cnt[0] -- qkv_a reports into [4][0].
        atomicAdd(&g_subphase_cnt[0], 1ULL);
      }
      _sp_t0 = _t;
    }
#endif
  }

  MPK_WS_PHASE(24, qkv_expected, xcd_id);
  // Stage stamp 19: q_b tiles + KV-cache update done.
  if (tid == 0) {
    mpk_stage_stamp(19);
  }
  // ══════════════════════════════════════════════════════════════════════
  // Phase 4: q_b -> decode barrier
  // ══════════════════════════════════════════════════════════════════════
  __syncthreads();
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#ifdef MPK_ENABLE_SUBPHASE_TIMING
  {
    unsigned long long _t = __builtin_amdgcn_s_memrealtime();
    if (tid == 0 && g_subphase_active) {
      atomicAdd(&g_subphase_ns[4][2], (_t - _sp_t0) * 10);
    }
    _sp_t0 = _t;
  }
#endif
  if (tid == 0) {
    if (hier_barrier_arrive(qb_barrier, HIER_STRIDE, arrivals, tiles_per_xcd,
                            xcd_id, bar_tree, /*skew_slot=*/3)) {
      // ── the head shard's rendezvous rides this barrier ──────────────────
      // Under QB_TP the query row is not complete when the local arrivals are
      // in; it is complete when every peer's eight heads have landed too. This
      // thread is the one the modular test elected, so it has observed all
      // eight local XCDs -- exactly the condition for telling the peers -- and
      // it is already the thread that fans the release out. One thread on the
      // rank polls the remote lines and the other 231 workers keep polling a
      // local flag that is simply published later. No second barrier.
      // MPK_QB_SKIP_PEER_WAIT: delete the q_b head-shard CROSS-RANK gather --
      // this rank's 7 peer stores and the poll of all 7 peers -- while keeping
      // the local hierarchical barrier, the flag release and every other
      // phase. WRONG OUTPUT by construction: the query row keeps the peers'
      // previous-layer heads, so decode reads 7/8 stale heads.
      //
      // Same purpose as MPK_MLA_SKIP_DECODE next door. It prices the ceiling
      // on head-sharding attention end-to-end, which would delete this
      // rendezvous outright rather than narrow it (the one shape the
      // "skew just relocates" rule does not obviously kill, since it removes
      // the rendezvous AND its producer set). The number that decides that
      // rewrite is NOT how much leaves S19->S20 -- it is how much survives at
      // the wall after S22 and S28 re-absorb the freed skew.
#ifndef MPK_QB_SKIP_PEER_WAIT
      if (qb_tp) {
        int64_t d[QB_NPEER];
        bool mapped = true;
#pragma unroll
        for (int q = 0; q < QB_NPEER; q++) {
          d[q] = 0;
          if (!mpk_shmem_peer_delta((q < EP_MY_PE) ? q : (q + 1), &d[q])) {
            mapped = false;
          }
        }
        if (mapped) {
          unsigned long long *const ep_sig =
              static_cast<unsigned long long *>(ep_signal_ptr);
          unsigned long long *const my_line =
              ep_sig + (size_t)EP_MY_PE * QB_EP_SIGNAL_STRIDE +
              QB_EP_SIGNAL_SLOT;
          // All QB_NPEER stores back to back, then one drain: distinct peers
          // are distinct XGMI links and pipeline.
#pragma unroll
          for (int q = 0; q < QB_NPEER; q++) {
            st_wt_u64(
                (void *)(reinterpret_cast<char *>(my_line) + d[q]),
                (unsigned long long)qb_expected);
          }
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          // Poll all peers off one bitmask rather than in rank order, so a
          // slow link costs its own latency and not the sum.
          unsigned remaining = (1u << QB_NPEER) - 1u;
          while (remaining) {
#pragma unroll
            for (int q = 0; q < QB_NPEER; q++) {
              if (remaining & (1u << q)) {
                int const p = (q < EP_MY_PE) ? q : (q + 1);
                if (ld_sys_u64(ep_sig + (size_t)p * QB_EP_SIGNAL_STRIDE +
                               QB_EP_SIGNAL_SLOT) >=
                    (unsigned long long)qb_expected) {
                  remaining &= ~(1u << q);
                }
              }
            }
            if (remaining) {
              __builtin_amdgcn_s_sleep(1);
            }
          }
          // The peer heads arrived as sc0 sc1 stores, so they are in memory.
          // Drop this CU's vL1 anyway before the release: the decode re-reads
          // the whole query row and its own buffer_inv is downstream of a flag
          // this thread has not written yet, which is the wrong order to rely
          // on.
          asm volatile("buffer_inv" ::: "memory");
        }
      }
#endif
      for (int x = 0; x < 8; x++) {
        st_wt_u32((void *)&qb_barrier[x * HIER_STRIDE], (unsigned)qb_expected);
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }
  }
  // Only the decode ranks need what this barrier protects. Everyone else
  // falls through to the decode barrier's arrival, which is what orders them
  // against the merge.
  if (xcd_rank < mla_tiles_per_xcd) {
    if (tid == 0) {
      // Self-heal, see MPK_FL_REPUBLISH_SPINS.
      int *const _qb_flag = &qb_barrier[xcd_id * HIER_STRIDE];
      MPK_WS_WAIT_BEGIN(763, qb_expected);
      int _spins = 0;
      int _obs;
      while ((_obs = ld_nt_s32(_qb_flag)) < qb_expected) {
        ++_spins;
        MPK_WS_WAIT_TICK(_obs, _spins);
        if ((_spins & (MPK_FL_REPUBLISH_SPINS - 1)) == 0) {
          // Under QB_TP the local arrival count is only HALF the release
          // condition -- the peer heads must have landed too, and only the
          // elected leader has waited for them. Healing off the counter would
          // do exactly what the barrier-901 bug did: let workers past a
          // rendezvous whose remote half has not happened. Heal off another
          // XCD's flag instead, which the leader writes after the peer wait,
          // so observing it implies the whole predicate.
          bool _heal;
          if (qb_tp) {
            _heal = false;
            for (int x = 0; x < 8; x++) {
              if (x != xcd_id &&
                  ld_nt_s32(&qb_barrier[x * HIER_STRIDE]) >= qb_expected) {
                _heal = true;
                break;
              }
            }
          } else {
            _heal = ld_nt_s32(&qb_barrier[8 * HIER_STRIDE]) >=
                    hier_barrier_heal_quota(arrivals, bar_tree) * qb_expected;
          }
          if (_heal) {
            st_wt_u32((void *)_qb_flag, (unsigned)qb_expected);
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
      unsigned long long _t = __builtin_amdgcn_s_memrealtime();
      if (tid == 0 && g_subphase_active) {
        atomicAdd(&g_subphase_ns[4][3], (_t - _sp_t0) * 10);
      }
      _sp_t0 = _t;
    }
#endif

    MPK_WS_PHASE(25, qkv_expected, xcd_id);
    // Stage stamp 20: q_b -> decode barrier passed (NESTED -- check cnt).
    if (tid == 0) {
      mpk_stage_stamp(20);
    }
    // ════════════════════════════════════════════════════════════════════
    // Phase 5: absorbed MLA decode, split over (q_head_group, kv_chunk)
    // ════════════════════════════════════════════════════════════════════
    // The decode wants a *global* work-item index, not a per-XCD tile: it
    // decomposes it into (q_head_group, kv_chunk, request). Synthesizing it
    // against mla_tiles_per_xcd rather than tiles_per_xcd reproduces the
    // standalone task's mapping, where the gang dispatch width was the
    // decode's own.
    // MPK_MLA_SKIP_DECODE: run every barrier and every other phase, but do no
    // decode work at all. WRONG OUTPUT by construction -- o_acc/lse keep the
    // previous layer's values. Same purpose as MPK_W13_EARLY_REL in
    // gang_oproj_router_fused_mi300.cuh: price the ceiling before building.
    // The comment in gang_mla_full_layer_fused_mi300.cuh:1391 attributes
    // 1.88 ms/iter of makespan to 232 workers waiting on the 16 that run this
    // loop. If deleting the loop entirely does not move the wall, that
    // attribution is wrong the same way the W13->W2 barrier's was, and
    // widening the decode (more kv chunks, more q groups) cannot pay.
#ifndef MPK_MLA_SKIP_DECODE
    for (int t = xcd_rank; t < mla_tiles_per_xcd; t += tiles_per_xcd) {
      // Under PAIR_MERGE this XCD owns chunks [pair_half * mla_tiles_per_xcd,
      // +mla_tiles_per_xcd) of q_group pair_id, and the decode kernel wants
      // chunk * NUM_Q_GROUPS + q_group. Still a bijection onto
      // [0, NUM_Q_GROUPS * NUM_KV_CHUNKS), just a different one.
      int const decode_item =
          PAIR_MERGE
              ? (pair_half * mla_tiles_per_xcd + t) * NUM_Q_GROUPS + pair_id
              : xcd_id * mla_tiles_per_xcd + t;
      // PAIR_MERGE's remap is a bijection onto [0, NUM_Q_GROUPS *
      // NUM_KV_CHUNKS) with no room for the token factor, and it is a
      // default-off knob that already measured neutral. Fail the build rather
      // than silently decode row 0 twice.
      static_assert(!(PAIR_MERGE && BATCH_SIZE > 1),
                    "MPK_GLM_MLA_PAIR_MERGE does not carry the token "
                    "dimension; it is incompatible with BATCH_SIZE > 1");
      gang_mla_decode_kernel<bfloat16,
                             NUM_Q_HEADS,
                             KV_LORA_RANK,
                             QK_ROPE_HEAD_DIM,
                             PAGE_SIZE,
                             MAX_SEQ_LEN,
                             NUM_KV_CHUNKS,
                             Q_WORKSPACE_STRIDE,
                             KV_CACHE_STRIDE,
                             /*WRITE_THROUGH=*/true,
                             BATCH_SIZE>(
          q_workspace_ptr,
          kv_cache_ptr,
          o_acc_ptr,
          lse_ptr,
          qo_indptr,
          kv_indptr,
          kv_indices,
          kv_last_page_len,
          mla_total_work_items,
          decode_item,
          scale_s);
    }
#endif
#ifdef MPK_ENABLE_SUBPHASE_TIMING
    {
      unsigned long long _t = __builtin_amdgcn_s_memrealtime();
      if (tid == 0 && g_subphase_active) {
        atomicAdd(&g_subphase_ns[4][4], (_t - _sp_t0) * 10);
      }
      _sp_t0 = _t;
    }
#endif
  }

  MPK_WS_PHASE(26, qkv_expected, xcd_id);
  // Stage stamp 21: MLA decode tiles done.
  if (tid == 0) {
    mpk_stage_stamp(21);
  }
  // ══════════════════════════════════════════════════════════════════════
  // Phase 6: decode -> merge barrier
  // ══════════════════════════════════════════════════════════════════════
  // Every worker arrives; only the merge ranks wait. The rest return here, a
  // whole merge stage early, rather than spinning -- the same trade the MoE
  // task makes at its W13 -> W2 barrier.
  __syncthreads();
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  if (tid == 0) {
    bool _dec_owes;
    if (dec_tree) {
      _dec_owes = hier_barrier_arrive(decode_barrier, HIER_STRIDE, dec_arrivals,
                                      tiles_per_xcd, xcd_id, true,
                                      /*skew_slot=*/4);
    } else {
      int prev = atom_add_release_gpu_s32(_dec_cnt, 1);
      _dec_owes = (prev % dec_arrivals) == dec_arrivals - 1;
    }
    if (_dec_owes) {
      if constexpr (PAIR_MERGE) {
        // Release only this pair. The other three pairs are computing chunks
        // this pair's merge never reads.
        for (int h = 0; h < XCDS_PER_GROUP; h++) {
          st_wt_u32(
              (void *)&decode_barrier[(pair_id * XCDS_PER_GROUP + h) *
                                      HIER_STRIDE],
              (unsigned)decode_expected);
        }
      } else {
        for (int x = 0; x < 8; x++) {
          st_wt_u32((void *)&decode_barrier[x * HIER_STRIDE],
                    (unsigned)decode_expected);
        }
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }
  }
  if (xcd_rank >= merge_tiles_per_xcd) {
    return;
  }
  if (tid == 0) {
    // Self-heal, see MPK_FL_REPUBLISH_SPINS. Note the arrival above is
    // unconditional but this wait is merge-ranks-only, so the counter still
    // advances by the full `dec_arrivals` per layer and the quota test holds.
    // Under PAIR_MERGE the counter is the pair's, and the pair's quota is the
    // *whole* predicate this flag stands for -- everything this XCD's merge
    // reads was written by the pair. Healing on a partial predicate is what
    // broke the NP=8 EP barrier; there is no partial one to heal on here.
    int *const _dec_flag = &decode_barrier[xcd_id * HIER_STRIDE];
    MPK_WS_WAIT_BEGIN(764, decode_expected);
    int _spins = 0;
    int _obs;
    while ((_obs = ld_nt_s32(_dec_flag)) < decode_expected) {
      ++_spins;
      MPK_WS_WAIT_TICK(_obs, _spins);
      if ((_spins & (MPK_FL_REPUBLISH_SPINS - 1)) == 0) {
        if (ld_nt_s32(_dec_cnt) >=
            hier_barrier_heal_quota(dec_arrivals, dec_tree) * decode_expected) {
          st_wt_u32((void *)_dec_flag, (unsigned)decode_expected);
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
    unsigned long long _t = __builtin_amdgcn_s_memrealtime();
    if (tid == 0 && g_subphase_active) {
      atomicAdd(&g_subphase_ns[4][5], (_t - _sp_t0) * 10);
    }
    _sp_t0 = _t;
  }
#endif

  MPK_WS_PHASE(27, qkv_expected, xcd_id);
  // Stage stamp 22: decode -> merge barrier passed.
  if (tid == 0) {
    mpk_stage_stamp(22);
  }
  // ══════════════════════════════════════════════════════════════════════
  // Phase 7: split-KV merge
  // ══════════════════════════════════════════════════════════════════════
  // merge_task_offset is (q_group * MERGE_DIM_SPLITS + dim_slice), which the
  // kernel decomposes itself; all this has to supply is a bijection onto
  // [0, NUM_Q_GROUPS * MERGE_DIM_SPLITS). The standalone task got it from
  // bid.y of a (requests, 32, 1) grid -- there is no bid.y in a gang task, so
  // it comes from the worker's own coordinates instead. NUM_Q_GROUPS is
  // declared up with the pair-merge block, which needs it far earlier.
  for (int t = xcd_rank; t < merge_tiles_per_xcd; t += tiles_per_xcd) {
    merge_splitkv_ck_fmha<bfloat16,
                          /*NUM_QO_HEADS_PER_KV=*/16,
                          NUM_Q_GROUPS,
                          /*HEAD_DIM=*/KV_LORA_RANK,
                          NUM_KV_CHUNKS,
                          /*KV_CHUNK_SIZE=*/128,
                          PAGE_SIZE,
                          MERGE_WRITE_THROUGH,
                          MERGE_DIM_SPLITS>(
        reinterpret_cast<float const *>(lse_ptr),
        reinterpret_cast<float const *>(o_acc_ptr),
        qo_indptr,
        kv_indptr,
        kv_last_page_len,
        request_id,
        reinterpret_cast<bfloat16 *>(attn_out_ptr),
        xcd_id * merge_tiles_per_xcd + t);
  }
#ifdef MPK_ENABLE_SUBPHASE_TIMING
  {
    unsigned long long _t = __builtin_amdgcn_s_memrealtime();
    if (tid == 0 && g_subphase_active) {
      atomicAdd(&g_subphase_ns[4][6], (_t - _sp_t0) * 10);
      atomicAdd(&g_subphase_cnt[4], 1ULL);
    }
  }
#endif
}

} // namespace kernel
