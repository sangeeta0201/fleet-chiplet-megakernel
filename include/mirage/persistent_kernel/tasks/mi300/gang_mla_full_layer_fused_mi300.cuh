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
//   [71 .. 79] * 16 : layer-entry barrier   (multi-layer mode only)
//   [80 .. 87] * 16 : reserved, unused      (EP)
//   [88]       * 16 : EP fold arrivals      (EP)
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

// Inline-EP ablation. EVERY non-zero setting PRODUCES WRONG OUTPUT -- each
// rank folds against whatever happens to be in its gather slots -- so no
// latency from one of these is an EP number. They exist to split the layer's
// cross-GPU cost into its two halves, which the aggregate cannot do:
//
//   0  full EP: peer signal store + peer wait.
//   1  neither. Prices the whole cross-GPU rendezvous against the MOE_EP=0
//      replica baseline. Measured NP=8 GLM-4.7-Flash: 4.768 ms/iter decode vs
//      3.637 for MOE_EP=0, so EP's compute reshape is ~1.1 ms and everything
//      else the full path costs is in the rendezvous.
//   2  store, no wait. The difference between 1 and 2 is what the seven
//      st_wt_u64s cost; the difference between 2 and 0 is what waiting for
//      them to become visible costs. Splitting these matters because the fold
//      already writes each peer's gather slot over the same links and that
//      transport is inside the 4.768, so a slow signal is a visibility
//      problem, not a bandwidth one.
//
// Same knob and same meaning as gang_full_layer_fused's. The two monoliths
// share a translation unit, so the guard is #ifndef, not a redefinition.
// Same default as gang_full_layer_fused_mi300.cuh, repeated because this
// header does not include that one and the two are only guaranteed to share a
// translation unit, not an order. Both guards are #ifndef, so a -D on the
// compile line still wins whichever is seen first.
#ifndef MPK_EP_WAIT_TIMEOUT
#define MPK_EP_WAIT_TIMEOUT 0
#endif
#ifndef MPK_EP_ABLATE
#define MPK_EP_ABLATE 0
#endif
#ifndef MPK_EP_TMO_PRINT_LAYERS
#define MPK_EP_TMO_PRINT_LAYERS 2
#endif

namespace kernel {

// Slot bases into the single counter buffer, in HIER_STRIDE units.
static constexpr int FULL_LAYER_ATTN_SLOT = 0;
static constexpr int FULL_LAYER_ATTN_RELEASE_SLOT = 30;
static constexpr int FULL_LAYER_OPROJ_SLOT = 40;
static constexpr int FULL_LAYER_ROUTER_COUNTER_SLOT = 70;
// Layer-entry barrier, used only in multi-layer mode. Mechanism C like the
// rest: per-XCD release flags at [71..78], global arrival counter at [79].
static constexpr int FULL_LAYER_ENTRY_SLOT = 71;
// ── expert parallelism (EP_WORLD_SIZE > 1 only) ──────────────────────────
// [80..87] are the eight per-XCD release flags for the EP rendezvous. They
// were reserved for a Mechanism-C exit barrier, went unused while every worker
// polled the signal lines directly, and are now load-bearing: the poll has to
// be ld_sys_u64 (L2-bypassing) and 240 workers doing that against seven remote
// lines is what made NP=8 unusable. One thread waits, these eight flags carry
// the result. The hazard they were originally reserved for -- MULTI_GPU_NOTES.md
// records it as *the* bug that made gpt-oss's EP emit garbage: a worker leaves
// the layer while another XCD's column slice is still being folded, and the
// residual resolve reads a half-written row -- is covered by the same release,
// since the thread that fans it out is the one that counted the eighth local
// fold slice in.
//
// [88] is the arrival counter the eight folding work-groups bump.
//
// MLA_ prefix, not the bare FULL_LAYER_EP_* gpt-oss uses: both monoliths are
// included into the same translation unit and share this namespace, and its
// slot map is its own (its EP barriers live at [48..85] of a 1216-int buffer).
// MPK_EP_FOLD_WGS: how many work-groups PER XCD fold the EP partial.
//
// It has always been 1 -- `xcd_rank == 0` -- so eight work-groups out of 232
// do the whole exchange while 224 sit in the release poll. The stage stamps
// price that slice at 5.37 us/layer (S3 - S0), which is 0.42 ms of wall for
// ~150 bytes per thread of traffic: the cost is the seven-peer store stream
// and its drain, not the arithmetic. Widening it is the one part of the EP
// collective that is mechanism rather than inter-rank skew.
//
// The arrival counter, the leader election and the self-heal quota all key
// off the folder count, so they move together with this.
#ifndef MPK_EP_FOLD_WGS
#define MPK_EP_FOLD_WGS 1
#endif
static constexpr int FULL_LAYER_EP_FOLDERS = 8 * MPK_EP_FOLD_WGS;

static constexpr int FULL_LAYER_MLA_EP_RELEASE_SLOT = 80;
static constexpr int FULL_LAYER_MLA_EP_FOLD_DONE_SLOT = 88;
// Phase 8b's W_UV -> o_proj barrier, un-absorbed kv_b_v only: per-XCD release
// flags at [96 .. 103], arrival counter at [104]. Past the EP slots rather
// than inside the MoE half's [40 .. 69] region, which 0/10/20 already fill.
static constexpr int FULL_LAYER_MLA_WUV_SLOT = 96;
// Phase 3b's q_b -> W_UK barrier, un-absorbed kv_b_k only. XCD-local, so it
// has no global counter: XCD x owns the flag at [106 + x] and the arrival
// counter eight ints into that same stride.
static constexpr int FULL_LAYER_MLA_WUK_SLOT = 106;
// MPK_NULL_PHASES: up to four extra GPU-wide rendezvous inserted at the head
// of the layer, each with its own nine lines (eight per-XCD release flags and
// an arrival counter) at [114 + 9k .. 122 + 9k].
//
// This is a PRICING probe and, unlike the ceiling probes, it is
// correctness-preserving: nothing is written between the null barriers, so the
// generated text is unchanged and the wall number is valid. That matters here
// because every wrong-output probe on this branch sits upstream of the router
// and is invalid for exactly that reason.
//
// What it answers: every work-cut, phase-cut and barrier-cut measured on GLM
// has been absorbed, while cutting worker count is strongly negative. That is
// the signature of a critical path made of the phase *sequence* rather than
// the work in any phase -- a per-phase fixed cost no work-ablation can see.
// Adding a phase that does no work prices that cost directly. If one null
// rendezvous is worth ~8.5 us/layer, phase-count reduction is the lever and
// the layer's 16 phases are worth ~5 ms; if it is worth ~2.5 us, it is not,
// and the 13 MB/layer/rank exchange rate from the W_UV wash is the whole story.
// ── MPK_BAR_TREE's per-XCD arrival counters ──────────────────────────────
// Not a region of its own: each real barrier's eight counters live at
// `its own base + 114`, so the eight blocks are scattered through [114 .. 217]
// at the same spacing as the barrier bases themselves. One uniform offset is
// legal because the closest two GLM barrier bases -- OPROJ's W13 at 60 and
// ENTRY at 71 -- are 11 slots apart and a tree block is 8. The largest base
// that takes a tree block is W_UV's 96, so the region ends at 96 + 114 + 7.
// Must agree with MPK_BAR_TREE_OFF in mpk_atoms.cuh; the host allocation in
// demo.py is sized against FULL_LAYER_COUNTER_SLOTS below, unconditionally,
// so turning the tree on never changes the buffer length across ranks.
static constexpr int FULL_LAYER_TREE_OFF_SLOTS = MPK_BAR_TREE_OFF;
static constexpr int FULL_LAYER_TREE_END_SLOT =
    FULL_LAYER_MLA_WUV_SLOT + FULL_LAYER_TREE_OFF_SLOTS + 8;
static_assert(FULL_LAYER_TREE_OFF_SLOTS == 114,
              "the tree offset is baked into FULL_LAYER_TREE_END_SLOT's "
              "arithmetic and into demo.py's allocation comment");

static constexpr int FULL_LAYER_NULL_PHASE_SLOT = FULL_LAYER_TREE_END_SLOT;
static constexpr int FULL_LAYER_MAX_NULL_PHASES = 4;
// Per null barrier: [0..7] per-XCD release flags, [8] the global arrival
// counter, [9..16] the per-XCD arrival counters used only by the tree variant.
// Each on its own 64-byte line -- eight counters sharing a line would
// serialize exactly the way the flat counter does, which is the thing under
// test.
static constexpr int FULL_LAYER_NULL_PHASE_STRIDE = 24;
#ifndef MPK_NULL_PHASES
#define MPK_NULL_PHASES 0
#endif
// MPK_NULL_TREE: same null barrier, two-level arrival. Each worker bumps its
// own XCD's counter (29 arrivals, eight lines in parallel); the last arriver
// per XCD bumps the global one (8 arrivals); the last of those fans out the
// release. Serialized atomics per rendezvous drop from 232 to 29 + 8 = 37.
//
// The flat probe measured 3.77 us/layer per rendezvous, and 232 atomics on one
// line at ~16 ns of L2 throughput each is 3.7 us -- so the hypothesis is that a
// rendezvous IS its atomic serialization. Running the two mechanisms through
// the same probe is the controlled test of that.
#ifndef MPK_NULL_TREE
#define MPK_NULL_TREE 0
#endif
// MPK_NULL_TILES: give each null rendezvous an empty grid-stride TILE LOOP in
// front of it, so the probe prices a whole *round* rather than just its
// barrier. `MPK_NULL_PHASES=4, MPK_NULL_TILES=0` is four bare rendezvous;
// `MPK_NULL_PHASES=4, MPK_NULL_TILES=24` is four rounds shaped exactly like
// qkv_a's -- 24 tiles per XCD strided over 29 workers -- whose tiles return
// immediately. The difference between the two is the per-round cost that is
// NOT the rendezvous: loop setup, the strided tile walk, and whatever the
// dispatch of a round costs beyond its sync.
//
// Why it decides the next lever. The layer is 15 sequential grid-stride
// rounds and round_count x one-tile makespan is the wall. If a round's fixed
// floor is the rendezvous alone (3.77 us, already measured), then fusing two
// rounds into one heterogeneous round is worth one rendezvous and one tile of
// makespan, and round-count reduction is arithmetic. If the empty-tile round
// costs materially more than the bare barrier, the round has a dispatch cost
// of its own and the fusion is worth more than that.
//
// Correctness-preserving like MPK_NULL_PHASES: the loop writes nothing.
//
// ── MEASURED. A ROUND'S FIXED COST *IS* ITS RENDEZVOUS. ──────────────────
// MPK_NULL_PHASES=4 both arms, alternated OFF/ON in one batch, n=4 each, one
// -D the variable:
//
//   NULL_TILES=0    11.858 / 11.688 / 11.810 / 12.044   mean 11.850
//   NULL_TILES=24   12.166 / 11.992 / 11.769 / 11.642   mean 11.892
//
// +0.042 ms over 4 x 78 = 312 added rounds, i.e. **0.14 us per round** of
// dispatch, against 2.7-3.8 us per rendezvous (this batch's 4 null barriers
// cost +0.845 ms against the 11.005 n=5 baseline = 2.71 us each; the earlier
// flat-probe batch said 3.77). Dispatch is 4% of a round. Nothing about a
// grid-stride round costs anything except the barrier in front of it and the
// work inside it.
//
// So round-count reduction is priced exactly: fusing two rounds is worth ONE
// rendezvous (~0.25 ms over 78 layers) plus whichever of the two rounds'
// makespans was the shorter. It is NOT worth a hidden per-round dispatch fee,
// because there isn't one. A fusion that costs any real byte traffic has to
// beat 0.25 ms + the shorter round to be worth building.
#ifndef MPK_NULL_TILES
#define MPK_NULL_TILES 0
#endif
static_assert(MPK_NULL_PHASES >= 0 &&
                  MPK_NULL_PHASES <= FULL_LAYER_MAX_NULL_PHASES,
              "MPK_NULL_PHASES must be 0..4; the counter buffer sizes for 4");
static constexpr int FULL_LAYER_COUNTER_SLOTS =
    FULL_LAYER_NULL_PHASE_SLOT +
    FULL_LAYER_MAX_NULL_PHASES * FULL_LAYER_NULL_PHASE_STRIDE;

// Self-heal gate for the Mechanism-C flag polls.
//
// Mechanism C splits "everyone arrived" into two separate facts: a global
// arrival counter, which is atomic and unambiguous, and eight per-XCD release
// flags, which are eight independent write-through stores issued by whichever
// worker happened to win a modular test on that counter. The counter is the
// truth; the flags are a cache of it, published once, by one thread, with no
// retry. Anything that costs one of those eight stores -- a lost store on one
// XCD's path, or an election that fires for the wrong layer once the workers
// straddle two of them -- wedges every worker on that XCD forever, while the
// other seven XCDs sail on. That is exactly the shape of the NP=8 hang: 170 of
// 240 workers spinning on `obs=26 exp=27` while 70 are already a layer ahead.
//
// So a waiter that has spun this long stops trusting the flag and consults the
// counter itself. The counter advances by exactly `arrivals` per fused layer,
// so `counter >= arrivals * expected` means every producer for this layer has
// arrived and the release is owed; the waiter then publishes its own XCD's
// flag. The store is the same monotonic absolute value the elected worker
// would have written, so re-issuing it is idempotent by construction -- the
// same argument MPK_EP_REPUBLISH_SPINS rests on in the gpt-oss monolith.
//
// Cost in a healthy run is zero: the gate is a spin count a satisfied poll
// never reaches. This barrier's tail is ~20 us and a spin round is ~0.4 us, so
// a normal wait clears in tens of rounds; 1024 rounds is ~400 us.
#ifndef MPK_FL_REPUBLISH_SPINS
#define MPK_FL_REPUBLISH_SPINS 1024
#endif

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
    bool MOE_WEIGHT_FP4 = false,
    // ── expert parallelism ──
    // 1 / 0 / 0 compiles the whole EP block out, so the single-GPU kernel is
    // byte-for-byte what it was before EP existed.
    int EP_WORLD_SIZE = 1,
    int EP_MY_PE = 0,
    // The one rank that folds the real residual into its MoE partial, so the
    // residual appears exactly once after the cross-rank sum.
    int EP_FOLD_PE = 0,
    // ── the EP tail ──────────────────────────────────────────────────────
    // Because the fold sits at the HEAD of the layer, the LAST fused layer's
    // MoE output is never folded by anyone: there is no layer L+1 to do it.
    // gpt-oss does not have this problem -- it folds at the tail, so its last
    // layer's own fold is the last thing it does (its EP_WRITE_COMBINED
    // variant).
    //
    // The fix is a variant that runs the layer-entry barrier, the fold, the
    // exchange and the peer wait, and then returns before Phase 1. demo.py
    // emits it as one extra gang_mla_full_layer_fused_layer call immediately
    // after the last real layer, with nothing between the two -- which is what
    // gets it picked up by the multi-layer scan in persistent_kernel.cuh (it
    // groups *consecutive* runs of this task type, and swaps variant_id per
    // layer out of ml_variant_ids, so a different instantiation per fused
    // layer is exactly what that table is for). Joining the batch is not
    // cosmetic: task_layer_idx has to keep counting, or the tail's signal-line
    // threshold would not agree across ranks.
    //
    // The LM head then reads the gather buffer directly through EP_PEER_SLOTS,
    // so no separate combine pass is needed.
    bool EP_TAIL_ONLY = false,
    // ── un-absorbed kv_b_v ────────────────────────────────────────────────
    // 0 keeps W_UV folded into o_proj. Non-zero adds Phase 8b, the
    // block-diagonal GEMV that applies it to the merged attention output, and
    // makes OPROJ_REDUCTION_SIZE the narrower NUM_Q_HEADS * WUV_V_HEAD_DIM.
    // See the Phase 0 comment in gang_oproj_router_fused_mi300.cuh -- at
    // GLM-5 on 8 ranks the absorbed o_proj is 60% of the token's bytes.
    // Adds input [29] (EP) / [27] (no EP) and output [11].
    int WUV_ROWS_PER_WG = 0,
    int WUV_V_HEAD_DIM = 0,
    // ── un-absorbed kv_b_k ────────────────────────────────────────────────
    // The q side of the same trade. 0 keeps W_UK folded into q_b, whose
    // output is then KV_LORA_RANK + QK_ROPE_HEAD_DIM per head; non-zero
    // narrows it to QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM and adds Phase 3b.
    // 77.9 MB a layer becomes 34.6 plus 6.5 for the W_UK stack.
    // Adds input [30] (EP) / [28] (no EP) and output [12].
    int QK_NOPE_HEAD_DIM = 0,
    int WUK_ROWS_PER_WG = 0,
    // Experts per router tile; `router_tile_n` below is the tile count, not
    // the expert count. See Phase 3 in gang_oproj_router_fused_mi300.cuh.
    int ROUTER_EXPERTS_PER_TILE = 1,
    // ── the router fold ───────────────────────────────────────────────────
    // Move the router's two contractions -- the gate GEMV and the RMSNorm's
    // sum-of-squares -- out of Phase 3 and into the o_proj epilogue that
    // already produced the row they contract over, reducing them across the
    // ranks on the all-gather rendezvous instead of after it. Adds two
    // inputs; needs the o_proj column shard, so it is EP-only.
    // See ROUTER_FOLD in gang_oproj_router_fused_mi300.cuh.
    bool ROUTER_FOLD = false>
__device__ __noinline__ void gang_mla_full_layer_fused_kernel_mi300(
    // Pointer arrays are passed whole rather than unpacked into 38 named
    // parameters, which is what gpt-oss's full-layer task does and for the
    // same reason: the unpacked form costs a few hundred bytes of stack frame
    // per thread for pointers that are read once each.
    void *const *input_ptrs,  // 29, see the demo's layer for the map
                              // ([27] and [28] are EP-only and may be null
                              //  at EP_WORLD_SIZE == 1)
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
    // Phase 8b's tile count, per XCD. 0 whenever WUV_ROWS_PER_WG is 0.
    int wuv_tiles_per_xcd,
    // Phase 3b's, likewise 0 whenever WUK_ROWS_PER_WG is 0.
    int wuk_tiles_per_xcd,
    // ── shared parameters ──
    int tiles_per_xcd,
    int tile_idx,
    // Multi-layer mode (task #14). `ml_num_layers` is runtime_config's, and is
    // 0 whenever the scheduler is dispatching one task per layer -- which is
    // the only case where the snapshot below is safe. `task_layer_idx` is the
    // monotonic (iteration * num_layers + layer) counter the scheduler writes
    // into task_metadata._linear_reserved before each layer of the batched
    // loop; it is meaningless when ml_num_layers is 0.
    int ml_num_layers,
    int task_layer_idx) {

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
  int *const entry_bar = counters + FULL_LAYER_ENTRY_SLOT * HIER_STRIDE;
  // W_UV's packed MXFP8 weight, un-absorbed kv_b_v only. Appended after the
  // EP pair so that turning EP on or off renumbers nothing: the input list is
  // 0..26 always, 27/28 under EP, and this last.
  constexpr int FL_WUV_WEIGHT_IN = (EP_WORLD_SIZE > 1) ? 29 : 27;
  // W_UK's, appended after it in a fixed order (W_UV then W_UK) and closing
  // the gap when W_UV is absorbed, so either can be on alone.
  constexpr int FL_WUK_WEIGHT_IN =
      FL_WUV_WEIGHT_IN + ((WUV_ROWS_PER_WG > 0) ? 1 : 0);
  // The fold's pair, appended last and closing both gaps ahead of it, so it
  // does not renumber under either un-absorption. Unlike the W_UV / W_UK
  // slots these cannot be read unconditionally: nothing is allocated for them
  // when the fold is off, so the index would run past the end of the list.
  // That is what ROUTER_FOLD is for -- the guard has to be compile-time.
  constexpr int FL_ROUTER_WT_IN =
      FL_WUK_WEIGHT_IN + ((WUK_ROWS_PER_WG > 0) ? 1 : 0);
  constexpr int FL_ROUTER_PARTS_IN = FL_ROUTER_WT_IN + 1;
  constexpr int FL_QNOPE_OUT = (WUV_ROWS_PER_WG > 0) ? 12 : 11;
  // Phase 8b's W_UV -> o_proj barrier. The MoE half's own [30 .. 38] would
  // land on top of the router counter here, so it gets its own base past the
  // EP slots and is handed down explicitly.
  int *const wuv_counters =
      counters + FULL_LAYER_MLA_WUV_SLOT * HIER_STRIDE;
  // Phase 3b's q_b -> W_UK barrier, XCD-local: per XCD a flag at [x * 16] and
  // an arrival counter at [x * 16 + 8], eight slots in all.
  int *const wuk_counters =
      counters + FULL_LAYER_MLA_WUK_SLOT * HIER_STRIDE;

  // Multi-layer mode: the scheduler is running every layer inside one task,
  // so the per-layer event boundary the snapshot below depends on is gone.
  bool const ml_mode = ml_num_layers > 0;

  // ── layer-entry barrier (multi-layer mode only) ──────────────────────────
  // Phase 1 resolves the residual stream, which the *previous* layer's MoE W2
  // wrote. With one dispatch per layer that ordering came free: the scheduler
  // would not dispatch this task until every worker had retired the last one.
  // Inside the batched loop nothing separates them -- the loop body is a bare
  // threadfence plus a __syncthreads, which orders one workgroup's 256 threads
  // and nothing else. A worker that finishes its W2 tiles early would walk
  // straight into the next layer's residual resolve and read a row other
  // workers have not finished accumulating into, and the resolve zeroes
  // moe_ws_f32 as it consumes it, so the loser's atomicAdd would land in an
  // already-drained accumulator.
  //
  // gpt-oss's fused layer has the same shape and does not do this. Its comment
  // reasons about visibility (threadfence, buffer_inv) rather than arrival,
  // which is a different property: the last barrier before its layer boundary
  // is the per-expert W13->W2 one, so its race window is one W2 tile wide and
  // evidently narrow enough not to fire. That is not an argument for leaving
  // it open here.
  //
  // Mechanism C, identical to Phase 8 below: bump a global counter, let the
  // last arriver fan a write-through release out to all eight per-XCD flags,
  // poll your own XCD's flag. Costs one all-XCD barrier per layer, which is
  // what the removed event boundary was anyway; what multi-layer mode buys is
  // the scheduler round trip on either side of it, not the sync itself.
  if (ml_mode) {
    int const entry_expected = task_layer_idx + 1;
    int const arrivals = tiles_per_xcd * 8;
    __syncthreads();
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    // Two-level arrival when the population divides across the eight XCDs,
    // which for this barrier it does by construction (arrivals is
    // tiles_per_xcd * 8). See hier_barrier_arrive.
    bool const entry_tree = MPK_BAR_TREE && (arrivals == tiles_per_xcd * 8);
    if (tid == 0) {
      if (hier_barrier_arrive(entry_bar, HIER_STRIDE, arrivals, tiles_per_xcd,
                              xcd_id, entry_tree, /*skew_slot=*/0)) {
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        for (int x = 0; x < 8; x++) {
          st_wt_u32((void *)&entry_bar[x * HIER_STRIDE],
                    (unsigned)entry_expected);
        }
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      }
      int *const my_flag = &entry_bar[xcd_id * HIER_STRIDE];
      MPK_WS_WAIT_BEGIN(761, entry_expected);
      int _spins = 0;
      int _obs;
      while ((_obs = ld_nt_s32(my_flag)) < entry_expected) {
        ++_spins;
        MPK_WS_WAIT_TICK(_obs, _spins);
        // Self-heal, see MPK_FL_REPUBLISH_SPINS. Same one-shot fan-out as
        // Phase 8 and the same failure mode, so the same escape: the arrival
        // counter is monotonic at `arrivals` per layer, and once it is at or
        // past arrivals * entry_expected the release is owed.
        if ((_spins & (MPK_FL_REPUBLISH_SPINS - 1)) == 0) {
          if (ld_nt_s32(&entry_bar[8 * HIER_STRIDE]) >=
              hier_barrier_heal_quota(arrivals, entry_tree) * entry_expected) {
            st_wt_u32((void *)my_flag, (unsigned)entry_expected);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          }
        }
        __builtin_amdgcn_s_sleep(1);
      }
    }
    __syncthreads();
    // Same reasoning as Phase 8: plain buffer_inv, not an agent-scope acquire.
    asm volatile("buffer_inv" ::: "memory");
    // Stage stamp 0: the layer-entry release is OBSERVED here. Measured from
    // the barrier's last arrival, so this interval is pure release
    // propagation -- the fan-out store plus one poll period.
    if (tid == 0) {
      mpk_stage_stamp(0);
    }

    // ── MPK_NULL_PHASES: price one rendezvous ────────────────────────────
    // Byte-for-byte the barrier above, including the buffer_inv, with no work
    // between them. See the note at FULL_LAYER_NULL_PHASE_SLOT.
#if MPK_NULL_PHASES > 0
    int *const null_bar =
        counters + FULL_LAYER_NULL_PHASE_SLOT * HIER_STRIDE;
#pragma unroll
    for (int k = 0; k < MPK_NULL_PHASES; k++) {
      int *const kb = null_bar + k * FULL_LAYER_NULL_PHASE_STRIDE * HIER_STRIDE;
#if MPK_NULL_TILES > 0
      // The empty tile loop. Same shape every real phase uses, same worker
      // stride, a body the compiler cannot delete but that touches no memory.
      {
        int _acc = 0;
        for (int t = xcd_rank; t < MPK_NULL_TILES; t += tiles_per_xcd) {
          asm volatile("" : "+v"(_acc) : : "memory");
          _acc += t;
        }
        // Never taken; keeps the loop live without a store on the real path.
        if (_acc == 0x7fffffff) {
          kb[8 * HIER_STRIDE] = _acc;
        }
      }
#endif
      __syncthreads();
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      if (tid == 0) {
        bool release;
        if (MPK_NULL_TREE) {
          // Two-level: 29 arrivals on this XCD's own line, then 8 on the
          // global one. The XCD counter's quota is tiles_per_xcd, so its
          // modular test is the per-XCD analogue of the flat one.
          int const xprev = atom_add_release_gpu_s32(
              &kb[(9 + xcd_id) * HIER_STRIDE], 1);
          release = false;
          if ((xprev % tiles_per_xcd) == tiles_per_xcd - 1) {
            int const gprev =
                atom_add_release_gpu_s32(&kb[8 * HIER_STRIDE], 1);
            release = (gprev % 8) == 7;
          }
        } else {
          int const prev = atom_add_release_gpu_s32(&kb[8 * HIER_STRIDE], 1);
          release = (prev % arrivals) == arrivals - 1;
        }
        if (release) {
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          for (int x = 0; x < 8; x++) {
            st_wt_u32((void *)&kb[x * HIER_STRIDE], (unsigned)entry_expected);
          }
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        }
        int *const my_flag = &kb[xcd_id * HIER_STRIDE];
        // Self-heal quota. Under the tree the global counter is bumped once
        // per XCD, not once per worker, so it tops out at 8 per layer. That
        // is still the WHOLE predicate the flag stands for -- it only reaches
        // 8 * entry_expected once every XCD's local counter hit its own
        // quota, i.e. once all 232 workers arrived. Healing on half a
        // predicate is what broke the NP=8 EP barrier.
        int const heal_quota = MPK_NULL_TREE ? 8 : arrivals;
        int _spins = 0;
        while (ld_nt_s32(my_flag) < entry_expected) {
          ++_spins;
          if ((_spins & (MPK_FL_REPUBLISH_SPINS - 1)) == 0) {
            if (ld_nt_s32(&kb[8 * HIER_STRIDE]) >= heal_quota * entry_expected) {
              st_wt_u32((void *)my_flag, (unsigned)entry_expected);
              asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            }
          }
          __builtin_amdgcn_s_sleep(1);
        }
      }
      __syncthreads();
      asm volatile("buffer_inv" ::: "memory");
    }
#endif
  }

  // ══════════════════════════════════════════════════════════════════════
  // Phase 0-EP: fold this rank's MoE partial and exchange it
  // ══════════════════════════════════════════════════════════════════════
  // gpt-oss's phase 9, moved to the HEAD of the layer. That relocation is the
  // one structural difference between the two ports, and it is what buys GLM
  // the 9a barrier for free.
  //
  // gpt-oss folds at the TAIL of layer L, so it has to introduce a GPU-wide
  // MoE barrier of its own (its 9a, a two-level arrival tree) to know every
  // W2 tile has landed. GLM already pays exactly that barrier -- the layer
  // entry barrier immediately above, added for task #14 because Phase 1 used
  // to consume moe_ws_f32 destructively. Same guarantee, same arrival count
  // (tiles_per_xcd * 8). Folding here rather than at the tail means the sync
  // is reused instead of duplicated, so EP costs GLM one rendezvous per layer
  // (the fold-done wait below) where it costs gpt-oss two.
  //
  // What it folds is therefore the PREVIOUS layer's output:
  //   input_ptrs[13] : moe_ws_f32, this rank's f32 expert partials
  //   input_ptrs[0]  : the residual stream (the previous layer's `hidden`)
  // and the sum of the two, rounded to bf16, is published into slot EP_MY_PE
  // of every rank's copy of input_ptrs[27]. Phase 1 then sums all
  // EP_WORLD_SIZE slots as part of the pass it already makes over that row --
  // see the EP_PEER_SLOTS note in gang_rmsnorm_linear_mxfp8_bias_mi300.cuh.
  // Exactly one rank (EP_FOLD_PE) adds the residual, so it survives the
  // cross-rank sum once.
  //
  // ── one gather buffer per fused layer ────────────────────────────────
  // input_ptrs[27] is both the fold's destination and Phase 1's source, in
  // the same layer, which reads oddly until you notice the fold is at the
  // head. It still has to be a distinct buffer per fused layer: after the
  // peer wait below, a peer is free to run ahead and fold layer L+1 while
  // this rank is still in layer L's Phase 1 reading the row. One buffer would
  // be a cross-rank WAR hazard with nothing ordering it. The multi-layer
  // input table hands each fused layer its own, which is 46 * 8 * 4 KB for
  // GLM-4.7-Flash.
  //
  // ── ml_mode is required ──────────────────────────────────────────────
  // The thresholds ride task_layer_idx, which is only meaningful inside the
  // batched loop. The single-dispatch path would need a snapshot instead, and
  // there is nothing safe to snapshot: the local signal line is written
  // asynchronously by peers. demo.py refuses to build an EP configuration
  // without precomputed dispatch rather than making that work.
  if constexpr (EP_WORLD_SIZE > 1) {
    MPK_WS_PHASE(11, task_layer_idx, xcd_id);
    int *const ep_fold_done =
        counters + FULL_LAYER_MLA_EP_FOLD_DONE_SLOT * HIER_STRIDE;
    // input_ptrs[27]: [EP_WORLD_SIZE, BATCH_SIZE, QKV_REDUCTION_SIZE] bf16
    // symmetric gather buffer; slot p holds rank p's folded partial.
    // input_ptrs[28]: [EP_WORLD_SIZE * 8] uint64 symmetric signal counters.
    //
    // QKV_REDUCTION_SIZE, not OPROJ_REDUCTION_SIZE: what is exchanged is a
    // residual-stream vector (padded hidden), whereas OPROJ_REDUCTION_SIZE is
    // o_proj's *input* width (num_heads * v_head_dim).
    unsigned short *const ep_gather =
        static_cast<unsigned short *>(input_ptrs[27]);
    uint64_t *const ep_signal = static_cast<uint64_t *>(input_ptrs[28]);
    constexpr size_t EP_SLOT_ELEMS =
        (size_t)BATCH_SIZE * QKV_REDUCTION_SIZE;
    constexpr size_t EP_SLOT_BYTES =
        EP_SLOT_ELEMS * sizeof(unsigned short);

    // Same run-monotonic counter every other barrier in this task uses, so
    // nothing is reset between layers and no worker has to observe a shared
    // value to agree on the target. One arrival per layer per signal line
    // from its single writer, so the line's threshold is just the layer
    // count -- not (EP_WORLD_SIZE - 1) * it, which would never be reached.
    int const ep_expected = task_layer_idx + 1;
    uint64_t const ep_sig_expected = (uint64_t)ep_expected;

    // Is every peer directly mapped? Hoisted out of the fold block, which
    // only the eight folding work-groups enter, because the peer wait further
    // down runs on EVERY worker and needs the same answer to pick its poll
    // shape. Pure function of the init-time delta table -- same value on
    // every thread, recomputed rather than communicated.
    bool ep_any_direct = true;
    for (int q = 0; q < EP_WORLD_SIZE - 1; q++) {
      int64_t _d = 0;
      if (!mpk_shmem_peer_delta((q < EP_MY_PE) ? q : (q + 1), &_d)) {
        ep_any_direct = false;
      }
    }
#ifdef MPK_EP_FORCE_STAGED
    // Bisection knob, not a tuning one. The direct path and the staged
    // rocSHMEM path publish the same value at the same symmetric address, so
    // forcing the staged one isolates "is the peer store landing" from
    // everything else in the layer. Correct but slower; never leave it on.
    ep_any_direct = false;
#endif

    // Column slice for this XCD. Rounded to an even boundary so the packed
    // 32-bit peer stores never straddle two slices.
    constexpr int EP_FOLD_COLS = QKV_REDUCTION_SIZE;
    constexpr int EP_FOLD_CHUNK = ((EP_FOLD_COLS + 7) / 8 + 1) & ~1;
    int const ep_xcd_lo = xcd_id * EP_FOLD_CHUNK;
    int const ep_xcd_hi = (ep_xcd_lo + EP_FOLD_CHUNK) < EP_FOLD_COLS
                              ? (ep_xcd_lo + EP_FOLD_CHUNK)
                              : EP_FOLD_COLS;
    // Sub-slice for this folding work-group. Even, same reason the XCD chunk
    // is even: the peer stores are packed 32-bit and must not straddle two
    // slices. The last folder can come out empty when the rounding overshoots
    // -- it still bumps the arrival counter, which is what the count is for.
    constexpr int EP_FOLD_SUB =
        ((EP_FOLD_CHUNK + MPK_EP_FOLD_WGS - 1) / MPK_EP_FOLD_WGS + 1) & ~1;
    int const ep_col_lo_raw = ep_xcd_lo + xcd_rank * EP_FOLD_SUB;
    int const ep_col_lo =
        (ep_col_lo_raw < ep_xcd_hi) ? ep_col_lo_raw : ep_xcd_hi;
    int const ep_col_hi = (ep_col_lo + EP_FOLD_SUB) < ep_xcd_hi
                              ? (ep_col_lo + EP_FOLD_SUB)
                              : ep_xcd_hi;

    // Set on the one thread, rank-wide, that counts the eighth local slice in.
    // That thread is the one that publishes, so it is also the one that waits;
    // every other worker polls a local flag instead. See the Mechanism C note
    // at the wait below for why that indirection is not optional at NP=8.
    bool ep_leader = false;
    int *const ep_release =
        counters + FULL_LAYER_MLA_EP_RELEASE_SLOT * HIER_STRIDE;

    // One folding work-group per XCD, eight in total, on disjoint columns.
    // xcd_rank 0 is the same rank that leads every other per-XCD job here.
    // An overshooting last sub-slice comes out empty (lo == hi) and folds
    // nothing -- but it still enters, because FULL_LAYER_EP_FOLDERS is a
    // compile-time count and the arrival counter has to see every one of them
    // or the leader is never elected.
    if (xcd_rank < MPK_EP_FOLD_WGS) {
      // One delta, two addresses, per peer. The gather slot and the signal
      // line are both symmetric-heap objects and the local->peer offset is
      // heap-wide, so translating the signal costs an add and no second
      // lookup.
      constexpr int EP_NPEER = EP_WORLD_SIZE - 1;
      void *peer_slot[EP_NPEER];
      int64_t ep_peer_delta[EP_NPEER];
      int ep_peer_pe[EP_NPEER];
      bool ep_all_mapped = true;
#pragma unroll
      for (int q = 0; q < EP_NPEER; q++) {
        peer_slot[q] = nullptr;
        ep_peer_delta[q] = 0;
        // Peers in rank order, skipping self: q -> the q'th other PE.
        ep_peer_pe[q] = (q < EP_MY_PE) ? q : (q + 1);
      }
#pragma unroll
      for (int q = 0; q < EP_NPEER; q++) {
        if (mpk_shmem_peer_delta(ep_peer_pe[q], &ep_peer_delta[q])) {
          peer_slot[q] = reinterpret_cast<void *>(
              reinterpret_cast<char *>(ep_gather + EP_MY_PE * EP_SLOT_ELEMS) +
              ep_peer_delta[q]);
        } else {
          // One unmapped peer disqualifies the direct path for the whole
          // work-group: the staged fallback is a collective that sends to
          // every peer, so it cannot be run for a subset.
          ep_all_mapped = false;
        }
      }
      // Work-group-wide decision, not per-thread: the staged fallback is a
      // work-group collective and every thread must agree on entering it.
      bool const ep_mapped_raw = ep_all_mapped;
#ifdef MPK_EP_FORCE_STAGED
      ep_all_mapped = false;
#endif
      bool const ep_direct = ep_all_mapped;
#ifdef MPK_EP_SIG_DBG
      // Which of the two publication paths is actually live. If
      // mpk_shmem_peer_delta hands back nothing usable, every layer silently
      // goes through the staged putmem_signal and no amount of reasoning about
      // st_wt_u64 explains anything.
      if (tid == 0 && xcd_id == 0 && task_layer_idx == 0) {
        printf("[EPPATH] pe=%d world=%d npeer=%d mapped_raw=%d ep_direct=%d "
               "any_direct=%d peer_slot0=%p delta0=%lld\n",
               EP_MY_PE, EP_WORLD_SIZE, EP_NPEER, (int)ep_mapped_raw,
               (int)ep_direct, (int)ep_any_direct, peer_slot[0],
               (long long)ep_peer_delta[0]);
      }
#endif

      // Note what this does NOT need to be templated on: the fold also zeroes
      // moe_ws_f32 as it reads it, which is redundant with the zeroing
      // gang_oproj_router_fused_mi300.cuh does in Phase 9 of this same layer.
      // Both land before any W2 accumulate, so the duplicate is 8 KB of
      // write-through stores and no hazard. Left duplicated rather than
      // threading a flag through the MoE half for it.
      _full_layer_ep_fold_partial<BATCH_SIZE,
                                  QKV_REDUCTION_SIZE,
                                  QKV_REDUCTION_SIZE,
                                  (EP_MY_PE == EP_FOLD_PE),
                                  EP_NPEER>(
          input_ptrs[13],                       // moe_ws_f32
          input_ptrs[0],                        // residual stream
          ep_gather + EP_MY_PE * EP_SLOT_ELEMS, // my slot
          peer_slot,                            // every peer's copy of my slot
          ep_col_lo,
          ep_col_hi);
      __syncthreads();
      // No threadfence_gpu(). Every store the fold makes -- local slot, peer
      // slots, workspace zeroing -- is write-through, so there is nothing in
      // this XCD's L2 for an agent-scope release to publish, and the fence is
      // `buffer_wbl2 sc1` on gfx950: a whole-L2 writeback, paid by all eight
      // folding work-groups on all 46 layers. What remains is the part that
      // was always load-bearing -- retire all 256 threads' stores before tid
      // 0's arrival atomic, so a consumer that observes the count observes
      // the bytes.
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      // Stage stamp 3: this XCD's column slice folded, its local and seven
      // peer copies stored and drained. S3 - S0 is the LOCAL half of the EP
      // cost; S1 - S3 is the cross-rank half. Only the eight folding
      // work-groups reach here.
      if (tid == 0) {
        mpk_stage_stamp(3);
      }

      if (ep_direct) {
        // My slice is in the peer's memory, ordered ahead of the arrival by
        // the drain above. Count the slice in; the 8th to land tells the
        // peers. The signal is a store of the run-monotonic count rather than
        // an accumulate, so it stays idempotent.
        if (tid == 0) {
          int const prev_f = atom_add_release_gpu_s32(ep_fold_done, 1);
          ep_leader = (prev_f % FULL_LAYER_EP_FOLDERS ==
                       FULL_LAYER_EP_FOLDERS - 1);
#ifdef MPK_EP_SIG_DBG
          // Which of the eight folding work-groups actually arrived, and in
          // what order. Leader election is `prev_f % 8 == 7` on a counter that
          // is never reset, so ONE missing arrival on one rank silently means
          // no leader, no publish, and every other rank waiting on that rank
          // forever -- which is the NP=8 signature (peers see p7 stuck one
          // layer behind while p0..p6 advance). Only the first two layers, and
          // only one line per work-group: 8 ranks x 8 XCDs x 2 layers.
          // Bounded by the same knob as [EPTMO] so the two reports cover the
          // same window. task_layer_idx is run-monotonic across decode
          // iterations, so a limit of ml_num_layers covers exactly the first
          // iteration -- which is where every failure localized so far has
          // landed, and 8 folders x 8 ranks x 47 layers of printf is a size
          // the buffer survives.
          if (task_layer_idx < MPK_EP_TMO_PRINT_LAYERS) {
            printf("[EPFOLD] pe=%d xcd=%d layer=%d prev_f=%d leader=%d\n",
                   EP_MY_PE, xcd_id, task_layer_idx, prev_f, (int)ep_leader);
          }
#endif
          if (prev_f % FULL_LAYER_EP_FOLDERS ==
              FULL_LAYER_EP_FOLDERS - 1) {
#if MPK_EP_ABLATE != 1
            // All EP_NPEER stores issued back to back, then a single drain:
            // distinct peers are distinct XGMI links and these pipeline, so
            // N-1 peers cost one drain rather than N-1 round trips.
#pragma unroll
            for (int q = 0; q < EP_NPEER; q++) {
              uint64_t *peer_sig = reinterpret_cast<uint64_t *>(
                  reinterpret_cast<char *>(
                      ep_signal +
                      (size_t)EP_MY_PE * FULL_LAYER_EP_SIGNAL_STRIDE) +
                  ep_peer_delta[q]);
              st_wt_u64((void *)peer_sig, (unsigned long long)ep_sig_expected);
            }
#endif
            // Same store, local copy. This thread has just observed all eight
            // local slices, which is exactly what a worker entering Phase 1
            // needs to know about its OWN rank's slot; publishing it on the
            // local signal line makes the wait below two loads in one loop
            // instead of two rendezvous.
            st_wt_u64((void *)(ep_signal + (size_t)EP_MY_PE *
                                               FULL_LAYER_EP_SIGNAL_STRIDE),
                      (unsigned long long)ep_sig_expected);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          }
        }
      } else {
        // No direct mapping: stage it. putmem_signal is a work-group
        // collective and sends the whole slot, so only one XCD may issue it,
        // and only after every slice has been folded locally.
        __shared__ int s_ep_put;
        if (tid == 0) {
          int const prev_f = atom_add_release_gpu_s32(ep_fold_done, 1);
          s_ep_put = (prev_f % FULL_LAYER_EP_FOLDERS ==
                      FULL_LAYER_EP_FOLDERS - 1)
                         ? 1
                         : 0;
          ep_leader = (prev_f % FULL_LAYER_EP_FOLDERS ==
                       FULL_LAYER_EP_FOLDERS - 1);
        }
        __syncthreads();
        if (s_ep_put && tid == 0) {
          st_wt_u64((void *)(ep_signal +
                             (size_t)EP_MY_PE * FULL_LAYER_EP_SIGNAL_STRIDE),
                    (unsigned long long)ep_sig_expected);
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        }
#if MPK_EP_ABLATE != 1
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
#endif
      }
    }

    // ── wait: this rank's slot complete, then every peer's ────────────────
    // Mechanism C. Exactly one thread on the rank waits -- the same thread
    // that published, which by construction counted the eighth local slice in
    // and therefore needs no self-signal wait at all -- and it fans a
    // write-through release out to eight per-XCD flags. Everyone else polls
    // its own XCD's flag with ld_nt_s32, which is L2-resident and local.
    //
    // The shape this replaced had every worker's tid 0 poll the EP_WORLD_SIZE
    // signal lines itself. That is a genuinely bad shape -- the poll has to be
    // ld_sys_u64, whose sc0 sc1 bypasses L1 *and* L2, so 240 workers x
    // (EP_WORLD_SIZE - 1) lines separated by nothing but s_sleep(1) aims
    // billions of uncached 8-byte reads per second at a handful of cache
    // lines -- and it is why this is written the house way. But do not read
    // the NP=8 history as evidence for it:
    //
    //   all-workers poll, NP=8 : 375.9 ms/iter (correct output)
    //   Mechanism C,     NP=8 : no measurable improvement
    //
    // The 375.9 is not this barrier's doing. MPK_EP_ABLATE=1 (no peer store,
    // no peer wait) still costs 4.768 ms/iter against 3.637 for the MOE_EP=0
    // replica baseline, and MOE_EP=0 at NP=8 -- no collective anywhere in the
    // kernel -- itself produced a 3.595 to 1347 ms/iter spread across the
    // eight ranks, with the slow ranks reshuffling run to run. Something makes
    // a random subset of concurrent GLM replicas ~350x slow on this box; a
    // per-layer rendezvous then makes every rank run at the slowest one's
    // speed. This barrier amplifies that fault, it does not cause it, and
    // changing its shape cannot fix it. See task #43.
    //
    // LATER, and it revises the paragraph above rather than replacing it: a
    // NP=8 MOE_EP=1 run with MPK_EP_WAIT_TIMEOUT=50000 and MPK_EP_SIG_DBG=1
    // came in at 4.002 ms/iter uniform across all eight ranks (range
    // 3.942-4.050), with correct generated text on every rank, and with
    // NEITHER bound ever firing -- no [EPTMO], no [EPREL]. So the ~350x
    // straggler is not a permanent property of eight concurrent replicas on
    // this box; it is intermittent, and when it does not happen the EP path is
    // 4.002 against 3.634-3.667 for the MOE_EP=0 replica baseline. That number
    // is NOT quotable as a ship figure -- it came off a build with both
    // diagnostic knobs compiled in, and the knobs are still the leading
    // suspect for why that run did not hang.
    //
    // What the release carries is strictly stronger than what the two waits it
    // replaces carried: the leader observed all eight local fold slices (it
    // was the eighth) *and* every peer's signal. The local half of that is the
    // part that is not optional -- it covers the two WAR hazards the removed
    // exit barrier used to, the fold reading input_ptrs[0] which Phase 9
    // overwrites via `hidden`, and the fold zeroing moe_ws_f32 which Phase 15
    // atomicAdds into.
    //
    // The peer wait stays here rather than moving to its point of use in
    // Phase 1. gpt-oss measured both placements (2.528 ms here vs 2.542 ms at
    // use) and the reason generalizes: pushing it into Phase 1 puts it
    // upstream of the qkv_a -> q_b barrier that gates all of attention, so one
    // late peer stalls the whole XCD instead of one worker.
    MPK_WS_PHASE(12, task_layer_idx, xcd_id);
    if (tid == 0) {
      if (ep_leader) {
#if MPK_EP_ABLATE == 0
        MPK_WS_PHASE(13, task_layer_idx, xcd_id);
        _full_layer_ep_wait_peers<EP_WORLD_SIZE, EP_MY_PE>(
            ep_signal, ep_sig_expected, ep_any_direct, tid);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        // Stage stamp 4: every peer's signal observed, by the one thread on
        // the rank that waits. S4 - S3 is pure cross-rank wait; S1 - S4 is
        // the eight-flag release fan-out.
        mpk_stage_stamp(4);
#endif
        for (int x = 0; x < 8; x++) {
          st_wt_u32((void *)&ep_release[x * HIER_STRIDE],
                    (unsigned)ep_expected);
        }
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      } else {
        // Barrier id 901: the per-XCD EP release. aux0 is the raw
        // ep_fold_done counter, so a stall here still distinguishes "the
        // leader never got its eighth local slice" from "the leader is stuck
        // on a peer" -- for the latter, look for bid 900 on one worker.
        int *const my_flag = &ep_release[xcd_id * HIER_STRIDE];
        MPK_WS_WAIT_BEGIN(901, ep_expected);
        int _ep_spins = 0;
        int _ep_obs;
        while ((_ep_obs = ld_nt_s32(my_flag)) < ep_expected) {
          if ((++_ep_spins & (MPK_WS_WAIT_REFRESH - 1)) == 0) {
            MPK_WS_WAIT_TICK(_ep_obs, _ep_spins);
            MPK_WS_WAIT_AUX(ld_nt_s32(ep_fold_done), _ep_obs, 0, 0);
          }
          // Self-heal, see MPK_FL_REPUBLISH_SPINS. The release is owed only
          // when BOTH halves of what the leader attests to hold: eight local
          // column slices folded (ep_fold_done == 8 * ep_expected, eight
          // folding work-groups per layer) AND every peer's signal in.
          //
          // The local half alone is NOT the predicate, though it reads like
          // the obvious one and was what this tested first. ep_fold_done
          // reaches 8 * ep_expected the instant this rank finishes its own
          // fold -- which is exactly when the leader ENTERS the peer wait, not
          // when it leaves. So a heal on the local half fires ~1024 spins into
          // every layer whose peers are even slightly late, and releases 239
          // workers into the next layer's QKV prologue, the one place that
          // sums the gather slots, before the peers have written them. Two
          // faults, and both were observed at NP=8:
          //
          //   correctness -- one rank's residual stream built from the
          //                  previous layer's peer slots, silently;
          //   liveness    -- those 239 hit the next layer's qkv barrier
          //                  without their leader, so that round sits one
          //                  arrival short (240*L - 1 on the counter) until
          //                  the leader escapes, and the whole rank parks.
          //
          // One healer per XCD, not 239 per rank. The peer read is
          // EP_WORLD_SIZE - 1 ld_sys_u64s, whose sc0 sc1 bypasses L1 and L2,
          // and the flag it would republish is per-XCD -- so 30 of the 30
          // workers on an XCD issuing it buys nothing that the first one does
          // not. xcd_rank 1 is already this file's designated one-per-XCD
          // reporter (see the [EPREL] print below); reusing it keeps the
          // uncached load rate at 8 x (EP_WORLD_SIZE - 1) per rank per 1024
          // spins instead of 239 x that.
          if (xcd_rank == 1 &&
              (_ep_spins & (MPK_FL_REPUBLISH_SPINS - 1)) == 0) {
            // Under MPK_EP_ABLATE the peers never signal, so the peer half is
            // vacuous and testing it would turn the backstop off entirely.
#if MPK_EP_ABLATE == 0
            bool const _ep_peers_in =
                _full_layer_ep_peers_ready<EP_WORLD_SIZE, EP_MY_PE>(
                    ep_signal, ep_sig_expected);
#else
            bool const _ep_peers_in = true;
#endif
            if (ld_nt_s32(ep_fold_done) >=
                    FULL_LAYER_EP_FOLDERS * ep_expected &&
                _ep_peers_in) {
              st_wt_u32((void *)my_flag, (unsigned)ep_expected);
              asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            }
          }
#if MPK_EP_WAIT_TIMEOUT
          // Bounded for the same reason as the peer wait -- see
          // MPK_EP_WAIT_TIMEOUT in gang_full_layer_fused_mi300.cuh. Without
          // this the rank whose leader was never elected is precisely the rank
          // that can never exit, so its printf buffer never flushes and the
          // only rank that knows why is the only one that cannot say.
          //
          // ep_fold_done is the payload: at layer L it must read 8L+8 once
          // every folding work-group has arrived. Anything less names how many
          // are missing. One line per XCD, not per worker.
          //
          // x256, and the factor is load-bearing rather than a fudge. A spin
          // here is one ld_nt_s32 off an L2-resident flag; a spin in the peer
          // wait is EP_WORLD_SIZE-1 ld_sys_u64s that miss L1 and L2 and go to
          // the fabric. At an equal spin budget these waiters give up orders of
          // magnitude sooner in wall time than the leader they are waiting for,
          // so the first run with this instrument had every rank time out at
          // layer 0 with fold_done=8 -- the leader was fine and simply had not
          // escaped its own bound yet. That reports nothing and, worse, drops
          // 240 workers through a barrier and corrupts the rest of the run.
          //
          // The ordering this factor buys is the whole diagnostic: the leader
          // always escapes first and releases, so these waiters fire ONLY when
          // no leader was elected -- which is the case worth naming.
          if (_ep_spins > MPK_EP_WAIT_TIMEOUT * 256) {
            if (xcd_rank == 1 && task_layer_idx < MPK_EP_TMO_PRINT_LAYERS) {
              printf("[EPREL] pe=%d xcd=%d layer=%d TIMEOUT obs=%d exp=%d "
                     "fold_done=%d (want %d)\n",
                     EP_MY_PE, xcd_id, task_layer_idx, _ep_obs, ep_expected,
                     ld_nt_s32(ep_fold_done),
                     FULL_LAYER_EP_FOLDERS * (task_layer_idx + 1));
            }
            break;
          }
#endif
          __builtin_amdgcn_s_sleep(1);
        }
      }
      MPK_WS_PHASE(14, task_layer_idx, xcd_id);
    }
    __syncthreads();
    asm volatile("buffer_inv" ::: "memory");
    // Stage stamp 1: Phase 0-EP done -- this rank's MoE partial folded,
    // published to all 8 peers, and every peer's partial observed. This sits
    // INSIDE the measured qkv_a gap and is a cross-rank dependency, so it is
    // a fourth candidate alongside propagation / dispatch / first tile.
    if (tid == 0) {
      mpk_stage_stamp(1);
    }
  }

  // The tail variant's whole job is the block above: fold the last real
  // layer's MoE output and publish it, so the LM head has something to sum.
  // Everything below would run a 47th attention/MoE layer on weights demo.py
  // did not bind.
  static_assert(!EP_TAIL_ONLY || EP_WORLD_SIZE > 1,
                "EP_TAIL_ONLY is meaningless without expert parallelism");
  if constexpr (EP_TAIL_ONLY) {
    return;
  }

  // All six release values, plus this task's own.
  //
  // One dispatch per layer: read them here, before Phase 1. See the header --
  // this is the only point in the fused body that is still behind the previous
  // layer's event, and the two halves' own snapshots are suppressed by passing
  // these down.
  //
  // Multi-layer mode: there is no such point, so the snapshot is replaced by
  // the deterministic layer counter, exactly as gpt-oss's full-layer task does
  // (see the layer_counter comment in gang_full_layer_fused_mi300.cuh). Every
  // one of these counters is monotonic, never reset, and bumped exactly once
  // per fused layer -- five are release flags written with the very value
  // being computed here, and routing_ready's epoch is read-modify-written once
  // by the single TopK completer -- so after fused layer L each of them holds
  // L + 1. Deriving that from the layer index instead of from a load means
  // every worker on every XCD agrees with no ordering requirement at all,
  // which is strictly stronger than the snapshot it replaces.
  __shared__ int s_exp[9];
  if (tid == 0) {
    if (ml_mode) {
#pragma unroll
      for (int i = 0; i < 9; i++) {
        s_exp[i] = task_layer_idx + 1;
      }
    } else {
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
      // Guarded: with W_UV absorbed the caller sizes the counter tensor at
      // 96 slots and [96 + xcd_id] is off the end.
      s_exp[7] = (WUV_ROWS_PER_WG > 0)
                     ? ld_nt_s32(&wuv_counters[xcd_id * HIER_STRIDE]) + 1
                     : 0;
      s_exp[8] = (WUK_ROWS_PER_WG > 0)
                     ? ld_nt_s32(&wuk_counters[xcd_id * HIER_STRIDE]) + 1
                     : 0;
    }
  }
  __syncthreads();
  int const attn_release_expected = s_exp[3];

  // ══════════════════════════════════════════════════════════════════════
  // Phases 1-7: the attention half
  // ══════════════════════════════════════════════════════════════════════
  // Workers past merge_tiles_per_xcd return out of this call early, from its
  // Phase 7 guard. They land on the Phase 8 barrier below with nothing to do
  // -- which is exactly where the o_proj weight prefetch is issued from.
  // Stage stamp 2: the last instruction before qkv_a tile code. S2 - S1 is
  // per-worker wake/dispatch (the release-value snapshot and the call setup);
  // gap[2] - S2 is the tile itself plus its barrier arrival.
  if (tid == 0) {
    mpk_stage_stamp(2);
  }
  MPK_WS_PHASE(20, task_layer_idx, xcd_id);
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
                                   MERGE_WRITE_THROUGH,
                                   /*EP_PEER_SLOTS=*/EP_WORLD_SIZE,
                                   QK_NOPE_HEAD_DIM,
                                   WUK_ROWS_PER_WG,
                                   EP_MY_PE,
                                   EP_WORLD_SIZE>(
      // Under EP the residual stream this prologue resolves is the symmetric
      // gather buffer the EP block above just folded into -- EP_WORLD_SIZE
      // bf16 slots holding the PREVIOUS layer's per-rank partials -- and the
      // prologue sums them itself, as part of the pass it already makes over
      // the row. See the EP_PEER_SLOTS note in
      // gang_rmsnorm_linear_mxfp8_bias_mi300.cuh for why the reduction lives
      // at the consumer rather than behind an exit barrier at the producer.
      //
      // Slot 0 sits exactly where an ordinary residual would, so at
      // EP_WORLD_SIZE == 1 this is input_ptrs[0] and the pointer arithmetic is
      // unchanged.
      /*x=*/(EP_WORLD_SIZE > 1) ? input_ptrs[27] : input_ptrs[0],
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
      /*decode_expected_in=*/s_exp[2],
      /*wuk_weight=*/input_ptrs[FL_WUK_WEIGHT_IN],
      /*q_nope=*/output_ptrs[FL_QNOPE_OUT],
      wuk_tiles_per_xcd,
      /*wuk_expected_in=*/s_exp[8],
      /*wuk_counters=*/wuk_counters,
      // Same object, same reasoning, as the o_proj all-gather's below: only
      // under EP and only in ml_mode, because the head shard's release value
      // is qb_expected, which is task_layer_idx + 1 there and a load off a
      // local counter otherwise -- and a local counter's value is not the
      // value a peer would publish. Slot 2 of the per-PE line; see
      // QB_EP_SIGNAL_SLOT in gang_mla_attn_fused_mi300.cuh.
      /*ep_signal=*/(EP_WORLD_SIZE > 1 && ml_mode) ? input_ptrs[28] : nullptr);

  MPK_WS_PHASE(60, task_layer_idx, xcd_id);
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
    bool const rel_tree = MPK_BAR_TREE && (arrivals == tiles_per_xcd * 8);
    if (tid == 0) {
      bool const _owes = hier_barrier_arrive(attn_release, HIER_STRIDE,
                                             arrivals, tiles_per_xcd, xcd_id,
                                             rel_tree, /*skew_slot=*/1);
#ifdef MPK_ENABLE_SUBPHASE_TIMING
      _b_t1 = __builtin_amdgcn_s_memrealtime();
#endif
      if (_owes) {
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
    // Phase 60's marker covers everything from the top of this barrier to the
    // MoE call, which is three very different places to be wedged: before the
    // arrival atomic, spinning on the flag, or parked in the trailing
    // __syncthreads while another wave of the same block is still behind.
    // 61/62/63/64 split them, because a capture with four workers "at the
    // barrier" cannot be read without knowing which.
    MPK_WS_PHASE(61, task_layer_idx, xcd_id);
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
    //
    // ── does this still pay at GLM-5 shapes? ─────────────────────────────
    // The numbers above were measured on GLM-4.7-Flash, where PF_WG_BYTES is
    // 160 KB. GLM-5 instantiates OPROJ_REDUCTION_SIZE 32768 (kv_b_v absorbed)
    // and OPROJ_ROWS_PER_WG 16, so
    //
    //   PF_WG_BYTES = 16 * 32768 * 33/32 = 528 KB per worker
    //   per XCD     = 29 workers * 528 KB = 15.3 MB
    //   L2 per XCD  = 4 MB (rocminfo)
    //
    // i.e. the prefetch working set is 3.8x L2, so most of what it pulls up is
    // evicted before Phase 9 reads it. PF_LPT scales with it too: 42 loads per
    // thread on Flash, 132 here, and 42 was already the count that made LLVM's
    // offset reassociation spill.
    //
    // Two consequences. Widening the prefetch to cover the whole grid-stride
    // sequence is wrong, even though it looks like an obvious bug that only
    // tile `xcd_rank` is fetched while gang_oproj_router_fused_mi300.cuh:261
    // strides over 48 tiles with 29 workers: that would take the working set
    // to 7.6x L2. And the existing one-round prefetch is itself suspect here.
    // GLM_OPROJ_PREFETCH=0 is the ablation.
    //
    // MEASURED at GLM-5, 78 layers, NP=8:
    //   prefetch on   18.2 ms/iter (17.993 / 18.220 / 18.227 / 18.110)
    //   prefetch off  see below
#ifndef MPK_GLM_OPROJ_PREFETCH_OFF
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

      // Only tile `xcd_rank`, deliberately, even though Phase 9 grid-strides
      // (gang_oproj_router_fused_mi300.cuh:261):
      //   for (t = xcd_rank; t < oproj_tiles_per_xcd; t += tiles_per_xcd)
      // At GLM-5 that is 48 tiles over 29 workers, so ranks 0..18 run a second
      // tile that this prefetch does not cover -- 19 of 48 tiles, 40% of
      // o_proj, cold from HBM. That looks like a bug. Covering it is a loss.
      //
      // Measured, GLM-5 78 layers NP=8, MPK_SUBPHASE_TIMING=1, aggregate
      // worker seconds at cnt 2209800 (GLM_OPROJ_PREFETCH=0 is the ablation):
      //
      //                        off       1 round    2 rounds
      //   SP3[0] o_proj      146.85 s    123.48 s    119.11 s
      //   SP5[0] barrier      59.25 s     57.61 s     77.75 s
      //   SP3[2] Router       58.33 s     53.87 s     61.20 s
      //   end to end        20.006 ms   19.227 ms   19.892 ms
      //
      // A second round buys 4.4 s on o_proj and pays 20.1 s at the Phase 8
      // barrier, because the extra traffic is issued by all 232 workers while
      // the 32 merge workers the barrier waits on are still reading -- the
      // same effect the +9% above documents, just bigger. One round is the
      // operating point. Note also that one round is already 29 * 528 KB =
      // 15.3 MB per XCD against 4 MB of L2; the prefetch pays anyway (off is
      // 19% worse on o_proj), so L2 residency is not the limit, barrier
      // interference is.
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
#endif // MPK_GLM_OPROJ_PREFETCH_OFF
#ifdef MPK_ENABLE_SUBPHASE_TIMING
    _b_t3 = __builtin_amdgcn_s_memrealtime();
#endif
    MPK_WS_PHASE(62, task_layer_idx, xcd_id);
    if (tid == 0) {
      int *const my_flag = &attn_release[xcd_id * HIER_STRIDE];
      // Watch this poll. a0 is the raw global arrival counter: it is
      // monotonic at `arrivals` per fused layer, so a0 >= arrivals *
      // attn_release_expected proves every worker arrived and the release
      // fired, and a stuck waiter is then a visibility fault on its own flag
      // rather than a missing producer. a1 is the XCD, since the release is a
      // fan-out to eight separate lines and only one of them may be short.
      // 760, not 860: the dump decoder reads any id in [800,900) as the MoE
      // W13->W2 barrier for expert (id-800), so 860 came out as two
      // contradictory decodings of the same aux pair and the counter it
      // printed could not be trusted. Keep this one out of that range.
      MPK_WS_WAIT_BEGIN(760, attn_release_expected);
      int _spins = 0;
      int _obs = 0;
      while ((_obs = ld_nt_s32(my_flag)) < attn_release_expected) {
        // Incremented outside the macro on purpose: MPK_WS_WAIT_TICK expands
        // to ((void)0) in the ship build, so `++_spins` inside it would never
        // be evaluated there and the republish gate below would fire on every
        // round instead of every MPK_FL_REPUBLISH_SPINS.
        ++_spins;
        MPK_WS_WAIT_TICK(_obs, _spins);
#ifdef MPK_WORKER_STATE
        // Guarded: MPK_WS_WAIT_AUX is unconditional, so an unguarded argument
        // would put this extra load in the ship build's hottest barrier.
        if ((_spins & (MPK_WS_WAIT_REFRESH - 1)) == 0) {
          MPK_WS_WAIT_AUX(ld_nt_s32(&attn_release[8 * HIER_STRIDE]),
                          xcd_id,
                          0,
                          0);
        }
#endif
        // Self-heal: see MPK_FL_REPUBLISH_SPINS. The counter is the truth, the
        // flag is a one-shot cache of it; if the cache is short and the
        // counter is not, publish the flag ourselves.
        if ((_spins & (MPK_FL_REPUBLISH_SPINS - 1)) == 0) {
          if (ld_nt_s32(&attn_release[8 * HIER_STRIDE]) >=
              hier_barrier_heal_quota(arrivals, rel_tree) *
                  attn_release_expected) {
            st_wt_u32((void *)my_flag, (unsigned)attn_release_expected);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          }
        }
        __builtin_amdgcn_s_sleep(1);
      }
    }
    MPK_WS_PHASE(63, task_layer_idx, xcd_id);
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
      // Attributed, and the entry-region guess above was wrong. Slot [7] is
      // _b_t0 - _fl_t0, i.e. exactly that __syncthreads + s_waitcnt, and it
      // measures 0.36 us/layer -- not the ~23 it was supposed to explain.
      //
      // Where the merge population's 27.2 us/layer of lateness actually goes,
      // NP=8, 75 fused layers, merge_tiles_per_xcd=16 of tiles_per_xcd=29
      // (divide each slot by ITS OWN population, not by 232):
      //
      //   Phase 5  decode compute    16 ranks    6.9 us each (SP4[4])
      //   Phase 6  decode->merge     128 ranks  22.1 us      (SP4[5])
      //   Phase 7  merge             128 ranks   1.8 us      (SP4[6])
      //   Phase 8  entry             128 ranks   0.4 us      (SP5[7])
      //   Phase 8  spin, merge pop   128 ranks   1.7 us      (SP5[5])
      //   Phase 8  spin, idle pop    104 ranks  28.9 us      (SP5[6])
      //
      // So the merge is not the straggler and neither is this barrier: both
      // populations are waiting, one phase apart, on the SAME thing -- Phase
      // 5's decode, which runs on num_q_groups * NUM_KV_CHUNKS work items.
      // At 4 groups x 4 chunks that is 16 of 232 workers. 128 wait for them
      // at Phase 6 and the other 104 wait at Phase 8.
      //
      // Averaged into makespan that is (128*22.1 + 104*28.9) / 232 = 25.1
      // us/layer = 1.88 ms/iter, the largest single line in the profile.
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
    MPK_WS_PHASE(64, task_layer_idx, xcd_id);
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
  MPK_WS_PHASE(70, task_layer_idx, xcd_id);
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
                                       MOE_WEIGHT_FP4,
                                       EP_WORLD_SIZE,
                                       EP_MY_PE,
                                       // The shared expert must be computed
                                       // exactly once across the world, and
                                       // the rank that already folds the
                                       // residual is the natural place to put
                                       // it: it is the one rank whose slot is
                                       // guaranteed non-trivial anyway.
                                       /*EP_SHARED_PE=*/EP_FOLD_PE,
                                       WUV_ROWS_PER_WG,
                                       /*WUV_REDUCTION=*/KV_LORA_RANK,
                                       WUV_V_HEAD_DIM,
                                       ROUTER_EXPERTS_PER_TILE>(
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
      /*w13_expected_in=*/s_exp[6],
      /*wuv_weight=*/input_ptrs[FL_WUV_WEIGHT_IN],
      /*v_out=*/output_ptrs[11],
      wuv_tiles_per_xcd,
      /*wuv_expected_in=*/s_exp[7],
      /*wuv_counters=*/wuv_counters,
      // Only under EP, and only in ml_mode: the all-gather's release value is
      // oproj_expected, which is task_layer_idx + 1 there and a load off a
      // counter otherwise -- and a value read from a local counter is not the
      // value a peer would publish. input_ptrs[28] is the same signal array
      // Phase 9 uses; see OPROJ_EP_SIGNAL_SLOT for how the line is split.
      /*ep_signal=*/(EP_WORLD_SIZE > 1 && ml_mode) ? input_ptrs[28] : nullptr,
      // The fold reduces on this same all-gather's rendezvous, so it is off
      // in exactly the cases the signal is: no EP, or the single-layer
      // dispatch, which has no all-gather to ride.
      /*router_weight_t=*/
      (ROUTER_FOLD && ml_mode) ? input_ptrs[FL_ROUTER_WT_IN] : nullptr,
      /*router_partials=*/
      (ROUTER_FOLD && ml_mode) ? input_ptrs[FL_ROUTER_PARTS_IN] : nullptr);
  static_assert(OPROJ_EP_SIGNAL_STRIDE == FULL_LAYER_EP_SIGNAL_STRIDE &&
                    QB_EP_SIGNAL_STRIDE == FULL_LAYER_EP_SIGNAL_STRIDE,
                "the o_proj and q_b all-gathers share the EP fold's signal "
                "array, so all three must agree on its per-PE stride");
  static_assert(QB_EP_SIGNAL_SLOT != OPROJ_EP_SIGNAL_SLOT &&
                    QB_EP_SIGNAL_SLOT != 0 && OPROJ_EP_SIGNAL_SLOT != 0 &&
                    QB_EP_SIGNAL_SLOT < FULL_LAYER_EP_SIGNAL_STRIDE,
                "the three signals must occupy distinct slots of the line, "
                "and slot 0 belongs to the Phase-9 fold");
  MPK_WS_PHASE(90, task_layer_idx, xcd_id);
}

} // namespace kernel
