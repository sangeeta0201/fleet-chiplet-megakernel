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

// MI450 (gfx1250) task implementations.
//
// This header deliberately does NOT mirror tasks/mi300/task_header.cuh. The
// mi300 header includes every task variant the MI350 build has ever used,
// including several that cannot compile for gfx1250 at all:
//
//   - anything reaching __builtin_amdgcn_mfma_*        (needs gfx950-insts /
//     mai-insts; gfx1250 has WMMA instead, with different shapes)
//   - anything reaching __builtin_amdgcn_cvt_scalef32_*(needs
//     fp8-cvt-scale-insts)
//   - CK's FMHA attention, which routes WMMA through
//     __builtin_amdgcn_wmma_*_w32_gfx12 and so requires target feature
//     wmma-128b-insts. gfx1250 does not have that feature and its WMMA
//     builtins are a disjoint set with different shapes, so CK attention is a
//     rewrite, not a patch.
//
// Rather than include those and #ifdef them out from the inside, this header
// starts from the empty set and adds each task only once it has been built and
// tested for gfx1250. That way "it is in this file" means "it compiles and has
// a test", and the porting frontier is visible in one place.
//
// A note on why so few files needed changes to be shared with mi300: the
// gfx9-only *asm* in the leaf tasks turned out to be almost entirely
// s_memrealtime behind #ifdef MPK_ENABLE_DEVICE_TASK_TIMING (off by default,
// and abstracted by mirage::arch::realtime_ticks() when on). The one real
// blocker, buffer_inv in moe_residual_add_f32, now goes through
// mirage::arch::inv_l2(). Tasks below are therefore literally the mi300
// sources, not forks -- no divergent copies to keep in sync.

#pragma once

// clang-format off

// ---------------------------------------------------------------------------
// Tier 1: architecture-independent (plain C++/HIP, no wave-width assumptions)
// ---------------------------------------------------------------------------
#include "tasks/ampere/embedding.cuh"
#include "tasks/ampere/identity.cuh"
#include "tasks/ampere/reduction.cuh"

// ---------------------------------------------------------------------------
// Tier 2: shared with mi300, verified to contain no gfx9-only instruction
// ---------------------------------------------------------------------------
#include "tasks/mi300/argmax_mi300.cuh"
#include "tasks/mi300/rotary_embedding_mi300.cuh"
#include "tasks/mi300/bias_add_mi300.cuh"
#include "tasks/mi300/moe_mul_sum_add_mi300.cuh"
#include "tasks/mi300/rmsnorm_mi300.cuh"
#include "tasks/mi300/silu_mul_mi300.cuh"
#include "tasks/mi300/swigluoai_mi300.cuh"
#include "tasks/mi300/moe_residual_add_f32_mi300.cuh"
#include "tasks/mi300/moe_topk_softmax_mi300.cuh"
// Pure index arithmetic over the MoE workspace: no asm, no cross-lane ops, no
// wave-width literal anywhere in the file, so wave32 needs nothing from it.
#include "tasks/mi300/moe_ws_layout.cuh"
// The attention epilogue. Scanned line by line: no asm, no cross-lane ops, and
// its only literal 64 is a head count, not a wave width. It takes its
// ck_tile/vec_load_8 symbols from whatever included it, so it must come after
// the tier-1 and common headers -- which is why it sits here rather than first.
#include "tasks/mi300/attention_sink_mi300.cuh"
// Split-KV merge. Indexing plus ptx_exp2/ptx_log2 from tasks/common/utils.cuh;
// its thread partitioning is by THREADS_PER_TOKEN (16 or 32), which is a tile
// width rather than a wave width, so it is wave-agnostic as written.
#include "tasks/ampere/merge_splitkv.cuh"

// ---------------------------------------------------------------------------
// Tier 3: gfx1250-native (WMMA / MX), written against arch_traits
// ---------------------------------------------------------------------------
#include "tasks/mi450/mx_layout_mi450.cuh"
#include "tasks/mi450/mx_wmma_mi450.cuh"
#include "tasks/mi450/gemm_handtuned_mi450.cuh"
#include "tasks/mi450/gang_moe_linear_mxfp4_mi450.cuh"
#include "tasks/mi450/paged_attention_wmma_decode_hd64_mi450.cuh"
// Must follow the decode header: its dispatch wrapper calls the decode kernel
// for seqlen_q == 1, and it reuses attn_exp2 / attn_swap_half / attn_lse_natural
// and the attn_ab_t / attn_acc_t fragment types defined there.
#include "tasks/mi450/paged_attention_wmma_prefill_hd64_mi450.cuh"
#include "tasks/mi450/linear_wmma_mi450.cuh"
#include "tasks/mi450/gang_rmsnorm_linear_mxfp4_bias_mi450.cuh"
#include "tasks/mi450/gang_moe_fused_mxfp4_mi450.cuh"
#include "tasks/mi450/kv_cache_update_mi450.cuh"

// Phase A of attention. Aliased rather than included from mi300 because the
// mi300 source spells the wave width as a literal 64 in six places and calls
// utils.cuh's shfl_xor_sync(), which hardcodes __shfl_xor(x, mask, 64) on AMD.
// None of that faults at wave32 -- it reduces over the wrong lane set and
// returns a plausible RMS norm -- so sharing the file was not an option. Same
// aliasing shape as the decode kernel below: identical signature and template
// list, so callers need no change.
#define kv_cache_update_impl ::kernel::mi450::kv_cache_update_impl

// Dense GEMM: point the gang wrappers at the WMMA kernel instead of CK's
// pipeline. This must be defined BEFORE gang_linear_mi300.cuh is included, so
// the include of that file below is deliberately after this #define -- the
// header only reads MPK_LINEAR_KERNEL at parse time, and it has an #ifndef
// guard, so defining it first is what selects the WMMA path.
//
// Why a macro and not the #define-alias trick used for attention: CK's
// linear_kernel_ck must remain nameable (gang_linear_silu_kernel still uses
// CK's pipeline types directly, and other off-path headers call it), so the
// name cannot be globally rewritten -- only the two hot-path call sites move.
#define MPK_LINEAR_KERNEL ::kernel::mi450::linear_kernel_wmma
#include "tasks/mi300/gang_linear_mi300.cuh"

// Fused RMSNorm + gang linear + bias. Step 1 (the wave-reduction prologue) was
// ported in place and is pinned bit-exactly by tests/mi450/test_rmsnorm_wave.hip;
// step 2 is gang_linear_kernel, which the macro above now routes to WMMA. That
// was the only thing this file was blocked on.
#include "tasks/mi300/gang_rmsnorm_linear_bias_mi300.cuh"

// Decode attention: route the mi300 entry point at the gfx1250 WMMA kernel.
// The signatures and template parameter lists are identical (MAX_SEQ_LEN is
// carried unused for exactly this reason), so callers need no change. This is
// the aliasing approach the plan settled on instead of duplicating codegen.
#define paged_attention_minimal_decode_hd64                                    \
  ::kernel::mi450::paged_attention_wmma_decode_hd64

// Attention dispatch (decode AND prefill). This replaces CK's
// paged_attention_ck_fmha_split_kv_impl, which is what gang_attention_mi300.cuh
// calls and which cannot target gfx1250 at all (its WMMA path needs the
// wmma-128b-insts target feature; gfx1250 does not have it).
//
// The replacement has the same shape: it reads seqlen_q from qo_indptr and
// routes 1 to the decode kernel and >1 to the prefill kernel, with the same
// DECODE_ONLY template parameter in the same position. So aliasing the name is
// enough and gang_attention_mi300.cuh needs no edit.
//
// Note this also removes the reason gang_full_layer_fused_mi300.cuh's
// unconditional CK include existed for mi450 -- see the note in the NOT YET
// PORTED block below, which is now stale for attention specifically.
#define paged_attention_ck_fmha_split_kv_impl                                  \
  ::kernel::mi450::paged_attention_wmma_split_kv_impl

// ---------------------------------------------------------------------------
// NOT YET PORTED -- tracked so the frontier stays explicit.
//
//   kv_cache_update             PORTED (2026-08-31), see the #define above.
//                               Writes new K/V into the paged cache and
//                               preprocesses Q (RoPE + QK-norm) into the
//                               workspace the decode kernel reads. Ported
//                               rather than shared because the mi300 source
//                               spells the wave width as a literal 64 in six
//                               places and calls utils.cuh's shfl_xor_sync(),
//                               which hardcodes width 64.
//                               tests/mi450/test_kv_cache_update.hip covers 5
//                               cases (decode with norm+rope / norm only /
//                               rope only, multi-token across a page boundary,
//                               aligned end of page) against a host double
//                               reference with a reversed page table and
//                               poisoned cache; 4/4 mutants killed.
//   gang_linear_silu_kernel     NOT PORTED. Unlike its two siblings in
//                               gang_linear_mi300.cuh it cannot go through
//                               MPK_LINEAR_KERNEL: it names CK's pipeline types
//                               (GemmPipelineAGmemBGmemCRegV2, TileGemmShape,
//                               MFMA WarpTile) directly in its body rather than
//                               calling linear_kernel_ck. Reached only from
//                               gang_rmsnorm_linear (no _bias), which is not on
//                               the gang_full_layer_with_lmhead_fused path, so
//                               it is deferred rather than blocking. Note it
//                               fuses interleaved gate/up weight tiles, so the
//                               WMMA version is not just a call swap.
//   gang_rmsnorm_linear_bias    PORTED (2026-08-28). Its wave64 assumptions are
//                               all gone (butterfly width, tid>>6 / tid&63,
//                               NUM_WAVES=4 cross-wave loop bound, float[16]
//                               LDS scratch) and its 3 asm sites now go through
//                               arch_traits. tests/mi450/test_rmsnorm_wave.hip
//                               pins the prologue bit-exactly against a host
//                               double reference at both wave widths, and was
//                               mutation-checked: restoring `nthreads >> 6`
//                               fails all 2880 elements at rel=0.3964.
//                               Step 2 (gang_linear_kernel) now resolves to
//                               linear_kernel_wmma via MPK_LINEAR_KERNEL, and
//                               the whole kernel instantiates clean for gfx1250
//                               at <bfloat16, 16, 3072, 2880>. Note what is and
//                               is not proven: both halves are separately
//                               validated against host references, but the
//                               fused kernel still has no end-to-end numerical
//                               test of its own. Phase 6 landed
//                               (tests/mi450/e2e/test_e2e_mi450.hip: a real
//                               multi-task graph through the megakernel
//                               scheduler, emitting the correct argmax token)
//                               but that harness runs its own small payloads to
//                               keep the reference obviously correct, so it
//                               validates the *dispatch path*, not this fused
//                               kernel's arithmetic.
//   paged attention             DECODE IS PORTED (above). Written fresh on
//                               gfx1250 WMMA, not translated: the WMMA
//                               accumulator distributes an output tile by
//                               (column=lane%16, rows=(lane/16)*8+[0..7]),
//                               which shares no axis assignment with MFMA's,
//                               so the QK product is taken as A=K,B=Q to keep
//                               each query's scores inside one lane.
//                               tests/mi450/test_attention_wmma.hip covers 8
//                               configurations (tile-aligned and ragged
//                               lengths, multi-page indirection with a
//                               non-identity page table, sliding window,
//                               sinks, split-KV with empty chunks) against a
//                               host double reference, and was mutation-
//                               checked against three distinct mapping bugs.
//                               PREFILL IS PORTED (2026-09-02). Same fragment
//                               types and data distribution as decode -- the 16
//                               accumulator columns just carry query *tokens*
//                               instead of query *heads*, which no fragment
//                               math notices. The only genuinely new part is
//                               the mask, and it was derived from CK's source
//                               rather than its comment: CK passes
//                               is_top_left=false, so query row i sits at
//                               q_abs = i + (seqlen_k - seqlen_q). The two
//                               conventions coincide exactly when delta == 0,
//                               so a fresh-prefill-only test cannot tell them
//                               apart, and getting it wrong yields finite,
//                               plausible output.
//                               tests/mi450/test_attention_prefill_wmma.hip
//                               runs 10 cases over both the fused and split
//                               paths (18 configurations) and asserts its own
//                               discriminating power -- it computes BOTH mask
//                               conventions on the host and fails if the case
//                               list ever stops separating them (currently 7
//                               cases separate them, max output delta 60.10).
//                               Mutation-checked with the mutant proven
//                               compiled in by in-place header swap, not by
//                               md5: top-left masking kills every delta>0 case
//                               at rel ~0.97 while all delta==0 still pass.
//                               One real bug it found: a `continue` on a
//                               fully-masked tile is NOT wave-uniform here
//                               (new_max is per-lane, since each lane owns a
//                               different query column and therefore a
//                               different window), so lanes taking it skipped
//                               the __shfl_xor and PV WMMA and survivors then
//                               traded with inactive partners. It is branchless
//                               now; see the comment at the safe_max/rescale
//                               pair.
//                               Dispatch goes through
//                               paged_attention_wmma_split_kv_impl (the #define
//                               above), which reads seqlen_q from qo_indptr and
//                               routes 1 to decode, >1 to prefill.
//   gang_rmsnorm_linear_mxfp4_bias
//                               PORTED (2026-08-28). Re-expressed through
//                               mx_wmma_mi450.cuh; the mi300 AGPR asm block is
//                               gone rather than translated.
//   gang_moe_fused_mxfp4        PORTED (2026-08-28). Likewise MX-native, no
//                               inline asm. tests/mi450/test_moe_fused_mxfp4
//                               covers four shapes -- SINGLE_TOK scalar decode,
//                               PACK_N ballot compaction, W13 K-split, and W2
//                               NOT K-split -- against an independent host
//                               E2M1/E8M0 reference, with poison and
//                               discriminating-power checks. Mutation-checked:
//                               swapped gate/up, ignored wg_idx, a wrong column
//                               in the K-split epilogue, and a dropped routing
//                               weight are all killed.
//
//                               Three real bugs the test found, all of which
//                               segfaulted FFM rather than returning garbage:
//                                 (a) d_mask[expert_idx] read past the
//                                     NUM_EXPERTS+1 mask -- expert_idx is
//                                     bounded by the PADDED tile space, not by
//                                     MAX_ACTIVATED. mi300 has the identical
//                                     unguarded read; it is latent there only
//                                     because W13_TILES is 48 at the registered
//                                     shape.
//                                 (b) the barrier was sized NUM_EXPERTS but is
//                                     indexed by that same padded expert_idx.
//                                     Fixed via moe450_barrier_slots(); the
//                                     overrun was an atomic RMW, i.e. a write.
//                                 (c) the K-split reduce epilogues read
//                                     s_tok_of_col[]/s_slot_of_col[], which
//                                     only exist under PACK_N -- on the other
//                                     decode paths they are extent-1 and never
//                                     written. gfx950 never hit this because
//                                     MFMA geometry gave W2_WAVES_PER_TILE==1.
//
//                               What is NOT proven: the W13->W2 barrier and its
//                               release/acquire pairing. The test runs the two
//                               phases as separate launches precisely because
//                               FFM cannot model the cross-workgroup wait.
//   gang_full_layer_fused / _with_lmhead
//                               orchestrators; land after their leaves
// ---------------------------------------------------------------------------

// clang-format on
