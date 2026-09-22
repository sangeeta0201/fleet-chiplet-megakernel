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

// Fused W13 + SwiGLU + W2 MoE gang kernel for MI450 (gfx1250).
//
// Port of tasks/mi300/gang_moe_fused_mxfp4_mi300.cuh (2372 lines). This file
// is roughly a fifth of that size, and the difference is almost entirely the
// eight hand-scheduled MFMA pipelines and the 48 direct-to-LDS asm loads that
// are gone -- see "WHAT WAS DELETED" below. The parts that carry the
// correctness of this kernel -- the compile-time tile space, the slot-indexed
// per-expert barrier, its cache-line-per-slot layout, the no-early-return
// window, and both epilogues -- are ported line for line and are annotated
// where they are subtle.
//
// ═══════════════════════════════════════════════════════════════════════════
// WHAT CHANGED, AND WHY EACH CHANGE IS FORCED
// ═══════════════════════════════════════════════════════════════════════════
//
// 1. 8 wave32s, not 4 wave64s. `warp_id = tid >> 5`, `lane_id = tid & 31`.
//
//    THIS IS NOT A RESCALING. gfx950 computes
//
//        W2_TILES_PER_WAVE = W2_OUTPUT_PER_WG / 16 / NUM_WAVES
//
//    which at the shape Fleet actually registers (W2_OUTPUT_PER_WG = 64, see
//    persistent_kernel.py) is 64/16/4 = 1 on gfx950 and **64/16/8 = 0** here.
//    Transcribing that expression would give a loop that runs zero times, so
//    the W2 phase would compute nothing, store nothing, and still arrive at
//    every barrier -- a silent all-zeros MLP output with no hang and no fault.
//
//    So the wave split is 2D, exactly as in
//    gang_rmsnorm_linear_mxfp4_bias_mi450.cuh which hit this first:
//    N_GROUPS waves take distinct output tiles, and WAVES_PER_TILE waves
//    cooperate on the K extent of one tile with their partials summed through
//    LDS. At the registered shape that is W13: 8 tiles -> 8x1 (pure
//    N-parallel, no reduction), W2: 4 tiles -> 4 groups x 2 K-waves.
//
// 2. Accumulator geometry. gfx950's MFMA gives each lane 4 rows at stride 4
//    (`g*4 + i`, g = lane>>4); gfx1250's WMMA gives each lane 8 CONSECUTIVE
//    rows of one column (`8*(lane/16) + i`). Both are read through
//    mx_acc_row()/mx_acc_col() rather than open-coded.
//
//    The SwiGLU epilogue survives this unchanged, and that is luck worth
//    stating: it pairs acc[i] (gate) with acc[i+1] (up) because the W13
//    weights interleave gate and up on consecutive output rows. That pairing
//    needs slots i and i+1 to be ADJACENT rows, which is true of both layouts
//    for even i -- gfx950 by `g*4+i`, gfx1250 by consecutiveness. If either
//    had strided its slots by 2 the epilogue would have silently paired a
//    gate with the wrong up. Same argument as in
//    gang_moe_linear_mxfp4_mi450.cuh's FUSE_SWIGLU path.
//
// 3. Token compaction ballot is wave32. `__ballot` returns a 64-bit mask on
//    both targets, but only 32 bits are meaningful here, and the prefix mask
//    `(1ull << lane_id) - 1` is correct at either width. What is NOT correct
//    is the implicit assumption that one wave covers 64 columns: MFMA_N is 16
//    and BATCH_SIZE <= 16 under packing, so a single wave32 still covers every
//    column. Asserted below rather than left to be rediscovered.
//
// 4. Scale granularity is 32, not 128. gfx950's scaled MFMA takes ONE E8M0
//    byte per 128-element K tile; gfx1250's WMMA takes FOUR, one per 32
//    elements, packed into a single 32-bit operand. So the LDS scale row pitch
//    is K/32 here where gfx950 used ceil(K/128 / 4)*4. Carrying gfx950's
//    stride over would put every scale after the first on the wrong K range --
//    a wrong answer, not a crash. mx_quant_fp8_gather static_asserts against
//    an undersized stride for exactly this reason.
//
// ═══════════════════════════════════════════════════════════════════════════
// WHAT WAS DELETED, DELIBERATELY
// ═══════════════════════════════════════════════════════════════════════════
//
// a. All inline asm. gfx1250 rejects the entire gfx9 vocabulary this file used
//    (`s_waitcnt`, `buffer_load_dwordx4 ... lds`, `v_mfma_scale_*`,
//    `v_accvgpr_read_b32`, `buffer_inv`). The waits go through
//    mirage::arch::wait_vmem(); the MFMA pipelines go through
//    mx_k_loop_f4xf8.
//
// b. The LDS-resident weight staging, i.e. the two 24-load
//    `s_mov_b32 m0` + `buffer_load_dwordx4 ... offen sc0 nt lds` blocks and
//    everything downstream of them (the per-wave weight tiles, their
//    padding arithmetic, the scale scatter, the LDS budget static_asserts).
//
//    On gfx950 that existed because MFMA is fed from AGPRs and a ds_read
//    stream pipelines better than an HBM stream. gfx1250 has no AGPRs, and
//    both already-validated mi450 MX kernels read weights straight from
//    global through the depth-2 pipelined mx_k_loop_f4xf8. Following that
//    precedent removes 48 direct-to-LDS sites instead of translating them to
//    `global_load_async_to_lds_b128` -- which would have been the single
//    riskiest construct in the port, because gfx1250's async path has NO
//    buffer-descriptor `num_records` clamp (the mi300 code leans on that for
//    OOB) and lands on a separate `asynccnt` counter that `wait_vmem()` does
//    not cover.
//
//    THIS IS A BEHAVIOURAL CHANGE, NOT A CLEANUP, and it is unmeasured. FFM
//    cannot compare the two. The reasoning is the same as documented at
//    length in gang_moe_linear_mxfp4_mi450.cuh, but on silicon this is the
//    first thing to re-examine if the MoE is slow.
//
//    One thing the deleted code did that is worth keeping is the W2 weight
//    prefetch issued BEFORE the barrier poll, so HBM latency overlaps the
//    wait instead of serializing after it. That intent is preserved with
//    mx_prefetch_operand_fp4_global, which is legal to issue before the
//    barrier because W2's WEIGHTS do not depend on the barrier -- only its
//    INPUT (the SwiGLU output written by another XCD) does.
//
// c. The MOE_DBG_SUBPHASE / MOE_DBG_ENTRY tripwire markers. They compile to
//    nothing unless MPK_NIL_TRIPWIRE and MPK_TW_SUB are both defined, so they
//    cost nothing -- but their macro definitions live in the mi300 header and
//    duplicating them here would collide if both headers are ever pulled into
//    one TU. The MPK_WS_MARK codes, which are what the wait-state dump keys
//    off, are kept with their original numbering so an mi300 dump and an
//    mi450 dump read the same.
//
// ═══════════════════════════════════════════════════════════════════════════
// WHAT FFM CAN AND CANNOT TELL US HERE
// ═══════════════════════════════════════════════════════════════════════════
//
// FFM-Lite can check the arithmetic: both phases against an independent host
// reference, the token compaction, the tile decode, the epilogue index math.
//
// It CANNOT check the thing this kernel is mostly made of. The W13->W2
// barrier is a cross-workgroup, cross-XCD release/acquire, and FFM runs
// workgroups serially (or time-sliced) on one modeled die. Worse, this was
// measured, not assumed: deleting a load-bearing __syncthreads() from the
// sibling kernel gang_rmsnorm_linear_mxfp4_bias_mi450 was run as a mutant and
// ALL FOUR test shapes still passed bit-exactly. A green FFM suite is not
// evidence that a barrier here is correct or that one is unnecessary.
//
// Specifically unverifiable, and needing silicon:
//   - the release/acquire pairing between atom_add_release_gpu_s32 and the
//     st_wt fan-out, and the cache-line separation that makes it work;
//   - that mirage::arch::xcd_id() returns anything at all (FFM cannot decode
//     s_sendmsg_rtn_b32; builds here define MIRAGE_XCD_ID_FALLBACK=1, which
//     pins it to 0 -- so every FFM run has exercised exactly one XCD and the
//     tile distribution is untested);
//   - every performance claim, including item (b) above.

#pragma once
#include "mirage/persistent_kernel/arch_traits.cuh"
#include "mirage/persistent_kernel/tasks/mi450/gang_moe_linear_mxfp4_mi450.cuh"
#include "mirage/persistent_kernel/tasks/mi450/mx_layout_mi450.cuh"
#include "mirage/persistent_kernel/tasks/mi450/mx_pipeline_mi450.cuh"
#include "mirage/persistent_kernel/tasks/mi450/mx_quant_mi450.cuh"
#include "tasks/mi300/moe_ws_layout.cuh"  // MOE_WS_SLOTS, moe_ws_offset()
#include "tasks/mi300/swigluoai_mi300.cuh" // fast_swigluoai()

// ── Wait-state tripwire macros ────────────────────────────────────────────
// These come from persistent_kernel.cuh, which includes this file (via
// tasks/mi450/task_header.cuh) only AFTER defining them -- so in the real
// build the fallbacks below never fire.
//
// The mi300 original simply assumed that ordering. That makes the header
// impossible to compile standalone, which in turn makes it impossible to write
// a focused unit test or an ISA probe for it -- and this is a kernel whose
// correctness argument leans hard on being able to test it in isolation. The
// no-op fallbacks cost nothing and remove the assumption.
#ifndef MPK_WS_MARK
#define MPK_WS_MARK(code, aux) ((void)0)
#endif
#ifndef MPK_WS_WAIT_BEGIN
#define MPK_WS_WAIT_BEGIN(barrier_id, expected) ((void)0)
#endif
#ifndef MPK_WS_WAIT_TICK
#define MPK_WS_WAIT_TICK(observed, spins) ((void)0)
#endif
#ifndef MPK_WS_WAIT_AUX
#define MPK_WS_WAIT_AUX(a0, a1, a2, a3) ((void)0)
#endif
#ifndef MPK_WS_WAVE_CLEAR
#define MPK_WS_WAVE_CLEAR(wave) ((void)0)
#endif
#ifndef MPK_WS_WAVE_EXIT
#define MPK_WS_WAVE_EXIT(wave) ((void)0)
#endif
#ifndef MPK_WS_WAIT_REFRESH
#define MPK_WS_WAIT_REFRESH 4096
#endif

namespace kernel {

// Per-expert MoE barrier geometry. Byte-identical to the gfx950 layout, and
// it must stay that way: the host allocates this buffer and both targets can
// in principle run against the same task graph.
//
// One 64-byte line per slot is a CORRECTNESS requirement, not padding for
// speed. The release fan-out uses write-through stores that bypass L2 and land
// in HBM, while the arrival counter is an ordinary L2-resident atomic RMW. If
// the two share a line, the L2 copy -- still holding the OLD release values --
// can be written back over the fresh write-through data, silently reverting
// slots that were already released; the W2 workers for that expert then wait
// forever on a release that did happen. That is a captured deadlock on gfx950,
// not a hypothetical, and nothing about gfx1250 removes the mechanism.
//   [xcd * MOE_BAR_LINE]        per-XCD release flag  (st_wt, HBM)
//   [MOE_BAR_COUNTER_SLOT * ..] global arrival count  (atomic, L2)
constexpr int MOE450_BAR_LINE = 16;        // int32 per cache line
constexpr int MOE450_BAR_COUNTER_SLOT = 8; // line index of the arrival counter
constexpr int MOE450_BAR_SLOTS = 10;       // lines reserved per expert
constexpr int MOE450_BAR_STRIDE = MOE450_BAR_SLOTS * MOE450_BAR_LINE;

namespace moe450_detail {

// ── LDS plan ──────────────────────────────────────────────────────────────
//
// Deliberately OUTSIDE the MIRAGE_ARCH_GFX1250 guard, for the same reason
// mx_rnlm_smem_bytes() is: the HOST has to size the dynamic shared memory for
// the launch, and on the host pass __gfx1250__ is not defined. Guarding it
// would make the host see no declaration at all. That failure is at least
// loud, but the workaround a caller would reach for -- hardcoding the byte
// count at the launch site -- is exactly the silent drift this exists to
// prevent.
//
// The +16 row pad is carried over from gfx950 and is still load-bearing,
// though for a slightly different reason. There it spread ds_read_b128 across
// LDS bank groups. Here the token operand is read with int4 loads whose
// address varies with the lane's token column, so an unpadded K-multiple pitch
// would again pile the 16 columns into one bank group. K+16 keeps
// (stride/16) odd for the shapes Fleet uses (3072+16 = 3088, /16 = 193, odd).

constexpr int moe450_round16(int x) {
  return ((x + 15) / 16) * 16;
}

// Scale row pitch, in E8M0 bytes. K/32 -- see change 4 in the file header.
// Rounded to 16 so the token region stays 16B aligned for whatever follows it.
constexpr int moe450_sc_stride(int k) {
  return moe450_round16(k / kernel::mi450::MX_K_PER_SCALE_BLOCK);
}

constexpr int moe450_tok_row_stride(int k) {
  return k + 16;
}

// ── How many barrier slots the host must allocate ───────────────────────────
//
// NOT NUM_EXPERTS. The W13 arrival indexes the barrier by `expert_idx`, which
// is `global_tile / W13_TILES` over the PADDED tile space -- TOTAL_W13 is
// rounded up to PAD_MULTIPLE (240), so expert_idx runs to
// TOTAL_W13/W13_TILES - 1, which exceeds MAX_ACTIVATED whenever the padding
// adds whole experts' worth of tiles. At the registered GPT-OSS shape
// (W13_TILES=48, MAX_ACTIVATED=4 at bs1) that is 5 slots against 128 experts,
// so sizing by NUM_EXPERTS happens to cover it and the dependency is invisible.
// At INTERMEDIATE_SIZE=512 / W13_OPW=128 it is 30 slots against 8 experts, and
// a NUM_EXPERTS-sized buffer is overrun by an atomic RMW -- a write, into
// whatever follows it.
//
// Padding slots are not free to skip: they arrive at the barrier like every
// other tile (see the "nothing read from the mask may steer control flow" note
// in the kernel), which is exactly why their counter must exist.
template <int BATCH_SIZE, int NUM_TOPK, int NUM_EXPERTS, int INTERMEDIATE_SIZE,
          int W13_OUTPUT_PER_WG>
__host__ __device__ constexpr int moe450_barrier_slots() {
  constexpr int PAD_MULTIPLE = 240;
  constexpr int W13_WGS = (2 * INTERMEDIATE_SIZE) / W13_OUTPUT_PER_WG;
  constexpr int W13_TILES =
      (BATCH_SIZE <= 16) ? W13_WGS : BATCH_SIZE * W13_WGS;
  constexpr int MAX_ACTIVATED = (NUM_TOPK * BATCH_SIZE < NUM_EXPERTS)
                                    ? NUM_TOPK * BATCH_SIZE
                                    : NUM_EXPERTS;
  constexpr int TOTAL_W13_REAL = MAX_ACTIVATED * W13_TILES;
  constexpr int TOTAL_W13 =
      ((TOTAL_W13_REAL + PAD_MULTIPLE - 1) / PAD_MULTIPLE) * PAD_MULTIPLE;
  // Ceiling: the last partial group still indexes its own slot.
  return (TOTAL_W13 + W13_TILES - 1) / W13_TILES;
}

// Bytes of dynamic shared memory this kernel needs, for the phase-max of W13
// and W2. Callers must allocate at least this much.
template <int BATCH_SIZE, int INTERMEDIATE_SIZE, int HIDDEN_SIZE,
          int W2_OUTPUT_PER_WG>
__host__ __device__ constexpr int moe450_smem_bytes() {
  // Mirrors the in-kernel geometry; see the kernel body for the derivations.
  constexpr int TOK_ROWS = (BATCH_SIZE <= 16) ? BATCH_SIZE : 1;
  constexpr int NUM_WAVES = 8;

  constexpr int W13_REGION =
      TOK_ROWS * moe450_tok_row_stride(HIDDEN_SIZE) +
      TOK_ROWS * moe450_sc_stride(HIDDEN_SIZE);
  constexpr int W2_REGION =
      TOK_ROWS * moe450_tok_row_stride(INTERMEDIATE_SIZE) +
      TOK_ROWS * moe450_sc_stride(INTERMEDIATE_SIZE);
  constexpr int TOK_BYTES =
      moe450_round16(W13_REGION > W2_REGION ? W13_REGION : W2_REGION);

  // Cross-wave K-reduction scratch. Needed only when some phase splits K
  // across waves; at the registered shape that is W2 (4 tiles, 8 waves).
  // Indexed by ABSOLUTE warp_id so every wave has a private slab, hence
  // NUM_WAVES -- never WAVES_PER_TILE, and never the gfx950 literal 4.
  constexpr int W2_N_TILES = W2_OUTPUT_PER_WG / 16;
  constexpr bool NEED_REDUCE = W2_N_TILES < NUM_WAVES;
  constexpr int REDUCE_BYTES =
      NEED_REDUCE ? NUM_WAVES * 16 * TOK_ROWS * (int)sizeof(float) : 0;

  return TOK_BYTES + REDUCE_BYTES;
}

} // namespace moe450_detail

#if defined(MIRAGE_ARCH_GFX1250)

using kernel::mi450::mx_acc_col;
using kernel::mi450::mx_acc_row;
using kernel::mi450::mx_f32x8_t;
using kernel::mi450::mx_global_u8;
using kernel::mi450::mx_i32x16_t;
using kernel::mi450::mx_k_loop_f4xf8;
using kernel::mi450::mx_load_operand_fp4_global;
using kernel::mi450::mx_load_operand_fp8;
using kernel::mi450::mx_operand_mn;
using kernel::mi450::mx_pack_scales;
using kernel::mi450::mx_pack_scales_global;
using kernel::mi450::mx_prefetch_operand_fp4_global;
using kernel::mi450::mx_quant_fp8;
using kernel::mi450::mx_quant_fp8_gather;
using kernel::mi450::mx_quant_fp8_nt;
using kernel::mi450::mx_to_global;
using kernel::mi450::MX_K_PER_SCALE_BLOCK;
using kernel::mi450::MX_K_PER_TILE;

// ── VGPR allocation: 272, and that is deliberate ─────────────────────────
//
// This kernel allocates 272 VGPRs, against ~88 for every other mi450 task.
// It looks alarming and is not. The excess is entirely inside the W13 K-loop
// (measured by bisect: gutting the K-loop drops it to 70, gutting W2 or the
// quantizer changes nothing), where the compiler puts the LDS B-operand
// ping-pong into the high bank via S_SET_VGPR_MSB. mx_k_loop_f4xf8 compiled
// standalone at the same K=3072 costs 31 VGPRs, so it is this kernel's
// live state across the loop, not the helper.
//
// Do NOT "fix" it by adding __launch_bounds__(256, 4) / waves_per_eu(4) to
// the enclosing persistent_kernel. That does drop this to 120 with no
// high-bank use -- and it also introduces 36 bytes/lane of scratch that is
// not there today. Per MPK_MAX_RESIDENT_BLOCKS in persistent_kernel.cuh,
// scratch, not LDS, is what FFM-Lite runs out of, and scratch per block is
// what caps co-residency of a kernel whose blocks never retire. Trading
// registers for scratch here is the wrong direction.
//
// Why 272 is safe (MI400 Shader Programming Guide S3.3): MI450 has 1024
// VGPRs per SIMD allocated from 64 blocks of 16 (wave32), and one WGP is
// 4 SIMD32. At WORKER_NUM_THREADS=256 a block is 8 wave32s = 2 waves/SIMD,
// so 272 VGPRs is ceil(272/16)=17 blocks x 2 waves = 34 of 64 -- it fits
// with room to spare, and there are no spills (vgpr_spill_count 0,
// private_segment_fixed_size 0). Residency is not at risk; for a persistent
// megakernel that is the only property that is load-bearing, since a block
// the hardware declines to dispatch is waited on forever.
//
// What this does cost is occupancy, which under FFM cannot be measured at
// all. Revisit on silicon: if profiling shows the MoE task latency-bound
// rather than LDS/issue-bound, the fix is to shorten live ranges in the
// K-loop, not to force the allocator.
template <int BATCH_SIZE,
          int INTERMEDIATE_SIZE,
          int HIDDEN_SIZE,
          int NUM_EXPERTS,
          int NUM_TOPK,
          int W13_OUTPUT_PER_WG,
          int W2_OUTPUT_PER_WG>
__device__ __noinline__ void gang_moe_fused_mxfp4_kernel_mi450(
    void const *input_ptr,          // [batch, hidden] BF16
    void const *gate_up_weight_ptr, // [E, W13_WGS, wg_bytes] MXFP4
                                    // (gate/up interleaved on output rows)
    void const *down_weight_ptr,    // [E, W2_WGS, wg_bytes] MXFP4
    void const *routing_ptr,        // [E, batch] int32
    void const *mask_ptr,           // [E+1] int32
    void const *w13_bias_ptr,       // [E, 2*INTERMEDIATE_SIZE] BF16
    void const *w2_bias_ptr,        // [E, HIDDEN_SIZE] BF16
    void const *routing_weight_ptr, // [batch, NUM_TOPK] float32
    void *swiglu_out_ptr,           // [batch, topk, INTERMEDIATE_SIZE] BF16
    void *workspace_f32_ptr,        // [batch, topk, HIDDEN_SIZE] float32
    void *barrier_ptr,              // [moe450_barrier_slots() * BAR_STRIDE] i32
    int tile_idx) {

  using namespace moe450_detail;

  // ── N-axis packing geometry ─────────────────────────────────────────────
  // Token `c` lives at LDS row `c` and feeds N column `c` of the 16x16x128
  // WMMA, so an expert's whole routed token set costs one tile sweep instead
  // of one per token.
  //
  // Batches wider than one WMMA N-tile keep the legacy token-per-tile decode.
  // A block-per-16-tokens compaction would leave some (expert, block) tiles
  // with no live token, and an empty tile still has to arrive at the W13->W2
  // barrier or the `% W13_TILES` modulus below stops being exact -- which is
  // the bs>1 hang this packing exists to fix.
  constexpr int WMMA_N = 16;
  constexpr bool PACK_N = BATCH_SIZE <= WMMA_N;
  constexpr int TOK_ROWS = PACK_N ? BATCH_SIZE : 1;
  // Distinguishes "one staged row because the batch is one" (fold the token
  // index to a literal 0, keeping every derived address wave-uniform) from
  // "one staged row because we fell back to the legacy decode" (token index
  // comes from the tile).
  constexpr bool SINGLE_TOK = PACK_N && BATCH_SIZE == 1;

  // One wave32 must cover every packed column, since the ballot that builds
  // the compaction tables runs on wave 0 alone. WMMA_N is 16, so this holds
  // with room to spare -- but it is the assumption that would break first if
  // WMMA_N ever grew.
  static_assert(!PACK_N || WMMA_N <= mirage::arch::WAVE_SIZE,
                "token compaction ballot assumes one wave covers all packed "
                "columns");

  // ── W13 constants (gate+up interleaved, reduction over hidden_size) ──────
  constexpr int W13_OUTPUT_SIZE = 2 * INTERMEDIATE_SIZE;
  constexpr int W13_K = HIDDEN_SIZE;
  constexpr int W13_NUM_BLK32 = W13_K / MX_K_PER_SCALE_BLOCK;
  constexpr int W13_WG_DATA = W13_OUTPUT_PER_WG * (W13_K / 2);
  constexpr int W13_WG_SCALE = W13_OUTPUT_PER_WG * W13_NUM_BLK32;
  constexpr int W13_WG_BYTES = W13_WG_DATA + W13_WG_SCALE;
  constexpr int W13_WGS = W13_OUTPUT_SIZE / W13_OUTPUT_PER_WG;
  constexpr int64_t W13_EXPERT_BYTES =
      static_cast<int64_t>(W13_WGS) * W13_WG_BYTES;
  // Tile space is weight groups only under packing: the token axis moved into
  // the WMMA's N dimension. This is also what makes the barrier modulus at the
  // bottom of Phase 0 exact -- see the note there.
  constexpr int W13_TILES = PACK_N ? W13_WGS : BATCH_SIZE * W13_WGS;

  // ── W2 constants (down projection, reduction over intermediate_size) ─────
  constexpr int W2_OUTPUT_SIZE = HIDDEN_SIZE;
  constexpr int W2_K = INTERMEDIATE_SIZE;
  constexpr int W2_NUM_BLK32 = W2_K / MX_K_PER_SCALE_BLOCK;
  constexpr int W2_WG_DATA = W2_OUTPUT_PER_WG * (W2_K / 2);
  constexpr int W2_WG_SCALE = W2_OUTPUT_PER_WG * W2_NUM_BLK32;
  constexpr int W2_WG_BYTES = W2_WG_DATA + W2_WG_SCALE;
  constexpr int W2_WGS = W2_OUTPUT_SIZE / W2_OUTPUT_PER_WG;
  constexpr int64_t W2_EXPERT_BYTES =
      static_cast<int64_t>(W2_WGS) * W2_WG_BYTES;
  constexpr int W2_TILES = PACK_N ? W2_WGS : BATCH_SIZE * W2_WGS;

  static_assert(W13_K % MX_K_PER_TILE == 0 && W2_K % MX_K_PER_TILE == 0,
                "both reductions must be a multiple of the 128-element WMMA "
                "K tile");

  // ── 2D wave split, per phase ────────────────────────────────────────────
  // See change 1 in the file header. N_GROUPS waves take distinct output
  // tiles; within a group, WAVES_PER_TILE waves split K and sum through LDS.
  // At the registered shape: W13 is 8x1 (no reduction), W2 is 4x2.
  constexpr int NUM_WAVES = 8;

  constexpr int W13_N_TILES = W13_OUTPUT_PER_WG / 16;
  constexpr int W13_N_GROUPS =
      W13_N_TILES < NUM_WAVES ? W13_N_TILES : NUM_WAVES;
  constexpr int W13_WAVES_PER_TILE = NUM_WAVES / W13_N_GROUPS;
  constexpr int W13_TILES_PER_GROUP = W13_N_TILES / W13_N_GROUPS;

  constexpr int W2_N_TILES = W2_OUTPUT_PER_WG / 16;
  constexpr int W2_N_GROUPS = W2_N_TILES < NUM_WAVES ? W2_N_TILES : NUM_WAVES;
  constexpr int W2_WAVES_PER_TILE = NUM_WAVES / W2_N_GROUPS;
  constexpr int W2_TILES_PER_GROUP = W2_N_TILES / W2_N_GROUPS;

  static_assert(NUM_WAVES % W13_N_GROUPS == 0 && NUM_WAVES % W2_N_GROUPS == 0,
                "wave count must divide evenly into output-tile groups");
  static_assert(W13_N_TILES % W13_N_GROUPS == 0 &&
                    W2_N_TILES % W2_N_GROUPS == 0,
                "output tiles must divide evenly across tile groups");
  static_assert((W13_K / MX_K_PER_TILE) % W13_WAVES_PER_TILE == 0 &&
                    (W2_K / MX_K_PER_TILE) % W2_WAVES_PER_TILE == 0,
                "K tiles must divide evenly across the waves cooperating on "
                "one output tile");
  // Non-zero by construction given the divisibility asserts above, but the
  // gfx950 expression it replaces evaluated to zero at this exact shape and
  // failed silently, so state it.
  static_assert(W13_TILES_PER_GROUP >= 1 && W2_TILES_PER_GROUP >= 1,
                "each wave group must own at least one output tile");

  // ── Pointer setup ───────────────────────────────────────────────────────
  unsigned short const *A = (unsigned short const *)input_ptr;
  uint8_t const *W_gate_up = (uint8_t const *)gate_up_weight_ptr;
  uint8_t const *W_down = (uint8_t const *)down_weight_ptr;
  int const *d_routing = (int const *)routing_ptr;
  int const *d_mask = (int const *)mask_ptr;
  unsigned short const *d_w13_bias = (unsigned short const *)w13_bias_ptr;
  unsigned short const *d_w2_bias = (unsigned short const *)w2_bias_ptr;
  float const *d_routing_weight = (float const *)routing_weight_ptr;
  // The SwiGLU intermediate is always BF16. An FP8 intermediate was tried on
  // gfx950 and was wrong; nothing here revisits that.
  unsigned short *d_swiglu_out = (unsigned short *)swiglu_out_ptr;
  float *d_workspace_f32 = (float *)workspace_f32_ptr;
  int *d_barrier = (int *)barrier_ptr;

  extern __shared__ char _fused_smem[];

  int const tid = threadIdx.x;
  int const warp_id = tid >> 5; // wave 0..7  (gfx950: tid >> 6)
  int const lane_id = tid & 31; // lane 0..31 (gfx950: tid & 63)
  // The accumulator column and the operand column are the same expression on
  // this target, but they are different concepts -- one is where a result
  // lands, the other is what a lane feeds in. Named separately so a future
  // layout change cannot conflate them.
  int const acc_col = mx_acc_col(lane_id);
  int const op_mn = mx_operand_mn(lane_id);

  // ── Tile decode (phase-ordered: padded W13, then all W2) ────────────────
  // total_w13 is padded to a multiple of 240 (30 workers x 8 XCDs) so every
  // worker's first tile is a W13 tile. Without it, W2 workers start polling
  // the barrier while W13 work remains. Padding tiles run empty and cost
  // almost nothing -- but they DO arrive at the barrier; see below.
  constexpr int PAD_MULTIPLE = 240;

  // Read once. On gfx950 this was s_getreg and nearly free; here it is
  // s_sendmsg_rtn, which the MI400 guide warns has very limited bandwidth and
  // must not be issued per wave. Hoisting it is not a micro-optimization.
  int xcd_id = mirage::arch::xcd_id();

  // Read for diagnostics only. Nothing that decides tile partitioning or
  // padding may derive from this word -- see the block comment below.
  int num_activated_experts = d_mask[NUM_EXPERTS];
#ifdef MPK_MOE_SINGLE_EXPERT
  constexpr bool MPK_MOE_SINGLE_EXPERT_ACTIVE = true;
  num_activated_experts = min(num_activated_experts, 1);
#else
  constexpr bool MPK_MOE_SINGLE_EXPERT_ACTIVE = false;
#endif
  (void)num_activated_experts;

  // ── Tile space is compile-time, not routing-dependent ───────────────────
  // MAX_ACTIVATED must equal the host's `max_activated` in
  // persistent_kernel.py, which is what sizes moe_total_tiles_per_xcd.
  //
  // The partitioning below (TOTAL_W13, the is_w2 split, expert_idx) MUST NOT
  // depend on num_activated_experts. There is one moe_mask buffer shared by
  // all layers and no barrier at the layer boundary, so workers straddle
  // layers -- a worker still finishing layer L can read layer L+1's mask. Once
  // the count became a true per-layer union over routed tokens it varies, and
  // two workers computing this arithmetic from different counts disagree about
  // which tiles are W13; the `% W13_TILES` release then never fires and the W2
  // workers spin forever. Deriving it from a constant makes every worker agree
  // by construction, whichever layer's mask it happened to read.
  constexpr int MAX_ACTIVATED = (NUM_TOPK * BATCH_SIZE < NUM_EXPERTS)
                                    ? NUM_TOPK * BATCH_SIZE
                                    : NUM_EXPERTS;

  // The output workspace is indexed by top-k slot, so its slot count must be
  // this model's experts-per-token. If they disagree, slot writes alias across
  // tokens (too few) or the consumer sums uninitialized slabs (too many) --
  // both silent. Consumers derive the same stride from MOE_WS_SLOTS.
  static_assert(NUM_TOPK == MOE_WS_SLOTS,
                "MOE_WS_SLOTS in moe_ws_layout.cuh must equal NUM_TOPK");

  int global_tile = tile_idx * 8 + xcd_id;
  constexpr int TOTAL_W13_REAL = MAX_ACTIVATED * W13_TILES;
  constexpr int TOTAL_W13 =
      ((TOTAL_W13_REAL + PAD_MULTIPLE - 1) / PAD_MULTIPLE) * PAD_MULTIPLE;
  constexpr int TOTAL_W2 = MAX_ACTIVATED * W2_TILES;
  constexpr int TOTAL_TILES = TOTAL_W13 + TOTAL_W2;
  if (global_tile >= TOTAL_TILES) {
    MPK_WS_MARK(8100, global_tile); // exit: past end of tile range
    return;
  }

  bool is_w2 = (global_tile >= TOTAL_W13);
  int expert_idx, phase_tile;
  if (!is_w2) {
    expert_idx = global_tile / W13_TILES;
    phase_tile = global_tile % W13_TILES;
  } else {
    int w2_tile = global_tile - TOTAL_W13;
    expert_idx = w2_tile / W2_TILES;
    phase_tile = w2_tile % W2_TILES;
  }

  // ── Nothing read from the mask may steer control flow ───────────────────
  // Two workers can read DIFFERENT layers' masks for the same slot, which
  // rules out deciding anything structural from mask contents -- not the tile
  // partitioning, not whether a slot is padding, and not the barrier address.
  // (Indexing the barrier by mask-derived expert_id split one slot's arrivals
  // across two counters on gfx950; that is why `base` below uses expert_idx.)
  //
  // So: the tile space is compile-time, the barrier is slot-indexed, and
  // padding tiles do NOT return -- they fall through with no active token, run
  // their WMMA over the clamped row, store nothing, and arrive like every
  // other tile. Every slot in [0, MAX_ACTIVATED) therefore arrives exactly
  // W13_TILES times per layer no matter what the mask says, which is what
  // makes the release unconditional.
  //
  // The mask is still read for expert_id, but only to pick weights. A stale
  // read there is benign numeric drift, not a deadlock.
  //
  // expert_idx is bounded by TOTAL_W13/W13_TILES, NOT by MAX_ACTIVATED: the
  // 240-tile padding rounds TOTAL_W13 up, so the slots in
  // [MAX_ACTIVATED, TOTAL_W13/W13_TILES) are pure padding and index past the
  // end of the mask, which the host allocates with exactly NUM_EXPERTS+1
  // entries. At the registered GPT-OSS shape W13_TILES is 48 and the largest
  // expert_idx is 4, far inside a 129-entry mask, so this never bites there --
  // but it is a property of the shape, not of the code. At INTERMEDIATE_SIZE
  // 512 / W13_OPW 128, W13_TILES is 8 and expert_idx runs to 29 against a
  // 9-entry mask: an out-of-bounds read that segfaulted FFM outright.
  // mi300 has the identical unguarded read; it is latent there for the same
  // shape-dependent reason. Guard it rather than constrain the shape -- the
  // out-of-range slot is padding by definition, so reading the sentinel and
  // reading nothing at all are the same answer.
  int expert_id_raw =
      (expert_idx < NUM_EXPERTS) ? d_mask[expert_idx] : -1;
  bool const is_padding_slot =
      (expert_id_raw < 0) || (expert_id_raw >= NUM_EXPERTS) ||
      (MPK_MOE_SINGLE_EXPERT_ACTIVE && expert_idx >= 1);
  // Clamp before it reaches any pointer arithmetic: the sentinel is -1 and the
  // weight/routing bases are built unconditionally below.
  int expert_id = is_padding_slot ? 0 : expert_id_raw;
  if (is_padding_slot) {
    MPK_WS_MARK(is_w2 ? 8105 : 8101, global_tile);
  }

  int n_wgs = is_w2 ? W2_WGS : W13_WGS;
  // Under packing the tile space IS the weight-group space, so the divide
  // folds away and every tile covers all of the expert's routed tokens.
  int tok_idx = PACK_N ? 0 : phase_tile / n_wgs;
  int wg_idx = PACK_N ? phase_tile : phase_tile % n_wgs;

  int const *expert_routing = d_routing + expert_id * BATCH_SIZE;

  if (!PACK_N && tok_idx >= BATCH_SIZE) {
    MPK_WS_MARK(8102, global_tile); // exit: token out of batch
    return;
  }

  // ── Per-expert token compaction ─────────────────────────────────────────
  // Which of the batch's tokens routed to this expert, packed into WMMA N
  // columns 0..n_tok-1. This replaces 16 separate tile launches (one per
  // token, 15 of which used to early-return) with 16 scalar loads.
  //
  // The early return those launches took is exactly the bs>1 hang: a tile that
  // returns above never reaches the arrival at the end of Phase 0, so
  // `prev_global % W13_TILES == W13_TILES - 1` never fires and the W2 workers
  // spin forever. NOTHING BELOW THIS POINT MAY RETURN EARLY on the W13 side.
  __shared__ int s_tok_of_col[PACK_N ? WMMA_N : 1];
  __shared__ int s_slot_of_col[PACK_N ? WMMA_N : 1];
  __shared__ int s_row_off[PACK_N ? WMMA_N : 1];
  __shared__ int s_n_tok;

  int my_tok, topk_slot, n_tok;
  bool tok_active;

  if constexpr (SINGLE_TOK) {
    // BATCH_SIZE == 1: there is one token, it is token 0, and an expert only
    // appears in the activated list because that token routed to it. Keep the
    // scalar path -- no table, no extra barrier, and the token index stays a
    // compile-time literal so every address derived from it is uniform.
    int route_val = expert_routing[0];
    if (route_val == 0) {
      MPK_WS_MARK(8103, global_tile); // exit: token not routed here
      return;
    }
    my_tok = 0;
    topk_slot = route_val - 1;
    n_tok = 1;
    tok_active = (acc_col == 0);
  } else if constexpr (PACK_N) {
    if (warp_id == 0) {
      // Clear then compact, both from wave 0: LDS ops from one wave retire in
      // program order, so the pad entries are overwritten by the compaction
      // and never read stale.
      bool const in_range = lane_id < WMMA_N;
      if (in_range) {
        s_tok_of_col[lane_id] = 0;
        s_slot_of_col[lane_id] = 0;
        s_row_off[lane_id] = 0;
      }

      // The ballot runs on the WHOLE wave, not inside the `lane_id < 16`
      // branch: it reports only currently-active lanes, so issuing it under
      // divergence would make the prefix count depend on which lanes happen to
      // be on. A padding slot contributes no tokens -- forced to 0 rather than
      // skipped, because expert_id was clamped to 0 for padding and
      // expert_routing therefore points at a real expert's row whose tokens
      // must not be compacted into a tile that has to produce nothing.
      int rv = (in_range && lane_id < BATCH_SIZE && !is_padding_slot)
                   ? expert_routing[lane_id]
                   : 0;
      unsigned long long hit = __ballot(rv != 0);
      int dst = __popcll(hit & ((1ull << lane_id) - 1));
      if (rv != 0) {
        s_tok_of_col[dst] = lane_id;
        s_slot_of_col[dst] = rv - 1;
        // W13 reads the token straight out of the norm buffer; W2 reads the
        // SwiGLU output, which is indexed by (token, top-k slot). `is_w2` is
        // uniform across the block, so one table serves whichever phase this
        // tile is in.
        s_row_off[dst] = is_w2 ? lane_id * (NUM_TOPK * INTERMEDIATE_SIZE) +
                                     (rv - 1) * INTERMEDIATE_SIZE
                               : lane_id * W13_K;
      }
      if (lane_id == 0) {
        s_n_tok = (int)__popcll(hit);
      }
    }
    __syncthreads();
    n_tok = s_n_tok;
    if (n_tok == 0) {
      // Should be unreachable: an expert is in active_expert_ids only because
      // some token routed to it. Mark rather than return -- returning is what
      // breaks the barrier modulus.
      MPK_WS_MARK(8104, global_tile);
    }
    tok_active = acc_col < n_tok;
    int const src_col = tok_active ? acc_col : 0;
    my_tok = s_tok_of_col[src_col];
    topk_slot = s_slot_of_col[src_col];
  } else {
    // Legacy per-token tile decode for batches wider than one WMMA N-tile.
    int route_val = expert_routing[tok_idx];
    if (route_val == 0) {
      MPK_WS_MARK(8103, global_tile); // exit: token not routed here
      return;
    }
    my_tok = tok_idx;
    topk_slot = route_val - 1;
    n_tok = 1;
    tok_active = (acc_col == 0);
  }

  // ── Column -> (token, slot), valid on ALL THREE decode paths ────────────
  // The K-split reduce epilogues (W13 and W2) redistribute the reduced values
  // across the whole block by a flat `idx` whose column component `t_col` is
  // NOT the reading thread's own acc_col -- a thread routinely finalizes a
  // column it did not accumulate. It therefore cannot use my_tok/topk_slot and
  // must map t_col back itself.
  //
  // s_tok_of_col/s_slot_of_col only exist under PACK_N: on the other two paths
  // they are declared with extent 1 and never written, so reading them yields
  // whatever LDS held, which then indexes d_routing_weight and the workspace.
  // That is not a wrong number -- it is a wild address, and it segfaulted FFM
  // on the first W2 tile of the bs1 shape (W2 4x2 takes the K-split path while
  // BATCH_SIZE 1 takes the SINGLE_TOK decode; the two never met on gfx950
  // because MFMA's tile geometry gave W2_WAVES_PER_TILE == 1 there).
  //
  // On the non-packed paths a tile has exactly one token, so column 0 is
  // (my_tok, topk_slot) and every other column is inactive.
  auto col_tok = [&](int c) -> int {
    if constexpr (PACK_N && !SINGLE_TOK) {
      return s_tok_of_col[c];
    } else {
      return my_tok;
    }
  };
  auto col_slot = [&](int c) -> int {
    if constexpr (PACK_N && !SINGLE_TOK) {
      return s_slot_of_col[c];
    } else {
      return topk_slot;
    }
  };

  // ── LDS plan (must agree with moe450_smem_bytes()) ──────────────────────
  constexpr int W13_TOK_ROW_STRIDE = moe450_tok_row_stride(W13_K);
  constexpr int W13_SC_STRIDE = moe450_sc_stride(W13_K);
  constexpr int W2_TOK_ROW_STRIDE = moe450_tok_row_stride(W2_K);
  constexpr int W2_SC_STRIDE = moe450_sc_stride(W2_K);
  constexpr int TOK_BYTES = moe450_round16(
      (TOK_ROWS * W13_TOK_ROW_STRIDE + TOK_ROWS * W13_SC_STRIDE) >
              (TOK_ROWS * W2_TOK_ROW_STRIDE + TOK_ROWS * W2_SC_STRIDE)
          ? TOK_ROWS * W13_TOK_ROW_STRIDE + TOK_ROWS * W13_SC_STRIDE
          : TOK_ROWS * W2_TOK_ROW_STRIDE + TOK_ROWS * W2_SC_STRIDE);

  // Indexed by ABSOLUTE warp_id: [warp][16 rows][TOK_ROWS cols]. Sized from
  // NUM_WAVES, never WAVES_PER_TILE and never a literal.
  float *s_reduce = (float *)(_fused_smem + TOK_BYTES);
  constexpr int REDUCE_SLAB = 16 * TOK_ROWS;

  static_assert(moe450_smem_bytes<BATCH_SIZE, INTERMEDIATE_SIZE, HIDDEN_SIZE,
                                  W2_OUTPUT_PER_WG>() >= TOK_BYTES,
                "moe450_smem_bytes() must cover the in-kernel LDS plan");
  static_assert(moe450_smem_bytes<BATCH_SIZE, INTERMEDIATE_SIZE, HIDDEN_SIZE,
                                  W2_OUTPUT_PER_WG>() <=
                    mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE -
                        mirage::runtime::LAYER_IDX_SMEM_OFFSET_FROM_END,
                "MoE fused LDS plan exceeds the MI450 dynamic shared memory "
                "budget");

  uint8_t *s_tok_fp8 = (uint8_t *)_fused_smem;

  // ════════════════════════════════════════════════════════════════════════
  // PHASE 0: W13 + SwiGLU -> write BF16 to swiglu_out
  // ════════════════════════════════════════════════════════════════════════
  if (!is_w2) {
    MPK_WS_MARK(8200, global_tile); // W13 compute

    // A tile with no routed token has nothing to compute, but it still has to
    // ARRIVE -- the release fires on `% W13_TILES`, which counts every tile in
    // the compile-time space whether or not the mask filled its slot. Skipping
    // only the compute keeps the barrier exact and the cost proportional to
    // the tokens actually routed: the GEMM below streams the full expert
    // weight tile from HBM, so running it for an empty slot would make latency
    // scale with MAX_ACTIVATED rather than with the experts a batch touches.
    //
    // Jumping rather than nesting the phase in an `if`: every path from here
    // must reach the arrival, and a jump makes that structurally obvious.
    if (n_tok == 0) {
      goto w13_arrive;
    }
    // Braced so the goto above does not jump across these initializations --
    // the label must sit outside their scope.
    {
      uint8_t *s_tok_scales = s_tok_fp8 + TOK_ROWS * W13_TOK_ROW_STRIDE;

      // ── Quantize this expert's routed tokens to FP8 in LDS ──────────────
      if constexpr (PACK_N && !SINGLE_TOK) {
        // s_row_off holds `tok * W13_K` for this phase (`is_w2` picked the
        // formula at compaction time).
        mx_quant_fp8_gather<W13_K, TOK_ROWS, W13_TOK_ROW_STRIDE, W13_SC_STRIDE,
                            /*NT_LOAD=*/false>(
            A, s_row_off, n_tok, s_tok_fp8, s_tok_scales);
      } else {
        mx_quant_fp8<W13_K>(A + tok_idx * W13_K, s_tok_fp8, s_tok_scales);
      }
      // The quantizer is block-cooperative and every wave's WMMA reads bytes
      // written by other waves. Load-bearing; see the FFM caveat in the header.
      __syncthreads();

      // B-operand base for THIS LANE: the token column it supplies. Inactive
      // lanes clamp to row 0 rather than skipping -- they still owe their
      // share of a wave-level WMMA that reads B from all 32 lanes. Folded to a
      // literal at TOK_ROWS == 1 so the address stays uniform.
      //
      // The row is folded into the base pointer rather than passed as
      // mx_load_operand_fp8's `row`, because that helper multiplies row by
      // k_stride while the real LDS pitch here is the padded W13_TOK_ROW_STRIDE.
      // Passing row directly would walk the token rows at the wrong stride.
      int const b_row = (TOK_ROWS == 1) ? 0 : (tok_active ? op_mn : 0);
      uint8_t const *b_tok = s_tok_fp8 + b_row * W13_TOK_ROW_STRIDE;
      uint8_t const *b_scl = s_tok_scales + b_row * W13_SC_STRIDE;

      uint8_t const *expert_weight =
          W_gate_up + static_cast<int64_t>(expert_id) * W13_EXPERT_BYTES;
      uint8_t const *wg_data =
          expert_weight + static_cast<int64_t>(wg_idx) * W13_WG_BYTES;
      uint8_t const *wg_scales = wg_data + W13_WG_DATA;

      mx_global_u8 *g_data = mx_to_global(wg_data);
      mx_global_u8 *g_scales = mx_to_global(wg_scales);

      constexpr int W13_K_PER_WAVE = W13_K / W13_WAVES_PER_TILE;
      int const grp = warp_id / W13_WAVES_PER_TILE;
      int const krank = warp_id % W13_WAVES_PER_TILE;

      for (int tile_iter = 0; tile_iter < W13_TILES_PER_GROUP; tile_iter++) {
        int const wave_tile = grp + tile_iter * W13_N_GROUPS;
        // The weight row THIS LANE SUPPLIES -- not the row its accumulator
        // ends up holding. The epilogue uses mx_acc_row() for that.
        // Conflating them yields a plausible-looking transpose.
        int const w_row = wave_tile * 16 + op_mn;

        mx_f32x8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

        if constexpr (W13_WAVES_PER_TILE == 1) {
          acc = mx_k_loop_f4xf8<W13_K>(
              g_data, g_scales, w_row, b_tok, b_scl, acc);
        } else {
          // A K sub-range. Deliberately NOT mx_k_loop_f4xf8: that helper
          // derives its scale-block stride and tail handling from its template
          // parameter, which here would be the slice length while the row pitch
          // is still W13_K. Passing the slice length would silently walk rows
          // at the wrong stride.
          int const k_off = krank * W13_K_PER_WAVE;
#pragma unroll 1
          for (int kt = k_off; kt < k_off + W13_K_PER_WAVE;
               kt += MX_K_PER_TILE) {
            mx_i32x16_t a =
                mx_load_operand_fp4_global(g_data, w_row, kt, W13_K);
            mx_i32x16_t b = mx_load_operand_fp8(b_tok, 0, kt, W13_K);
            unsigned int sa = mx_pack_scales_global(
                g_scales + (size_t)w_row * W13_NUM_BLK32 +
                kt / MX_K_PER_SCALE_BLOCK);
            unsigned int sb =
                mx_pack_scales(b_scl + kt / MX_K_PER_SCALE_BLOCK);
            acc = kernel::mi450::_gang_wmma_f4xf8(a, b, acc, (int)sa, (int)sb);
          }
        }

        // ── SwiGLU epilogue ─────────────────────────────────────────────
        // acc[i] and acc[i+1] are ADJACENT output rows on this target, which
        // is what lets them be the gate/up pair of one activation. See change
        // 2 in the file header.
        constexpr int ACT_STRIDE = W13_OUTPUT_SIZE / 2; // == INTERMEDIATE_SIZE
        if constexpr (W13_WAVES_PER_TILE == 1) {
          if (tok_active) {
#pragma unroll
            for (int i = 0; i < 8; i += 2) {
              int const out_n = wg_idx * W13_OUTPUT_PER_WG + wave_tile * 16 +
                                mx_acc_row(lane_id, i);
              if (out_n + 1 < W13_OUTPUT_SIZE) {
                unsigned bt_g =
                    (unsigned)d_w13_bias[expert_id * W13_OUTPUT_SIZE + out_n]
                    << 16;
                unsigned bt_u = (unsigned)
                                    d_w13_bias[expert_id * W13_OUTPUT_SIZE +
                                               out_n + 1]
                                << 16;
                float bias_g, bias_u;
                __builtin_memcpy(&bias_g, &bt_g, 4);
                __builtin_memcpy(&bias_u, &bt_u, 4);
                float activated =
                    fast_swigluoai(acc[i] + bias_g, acc[i + 1] + bias_u);
                int const act_n = out_n / 2;
                int const out_idx = my_tok * (NUM_TOPK * ACT_STRIDE) +
                                    topk_slot * ACT_STRIDE + act_n;
                st_wt_u16(&d_swiglu_out[out_idx],
                          _gang_float_to_bf16_mi450(activated));
              }
            }
          }
        } else {
          // Sum this tile's K partials across its cooperating waves, THEN
          // apply SwiGLU. Order matters: SwiGLU is non-linear, so activating
          // partial sums and adding the results is a different (wrong)
          // function. The gfx950 code never had to think about this because
          // it never split K.
          if (tok_active) {
#pragma unroll
            for (int i = 0; i < 8; i++) {
              s_reduce[warp_id * REDUCE_SLAB +
                       mx_acc_row(lane_id, i) * TOK_ROWS + acc_col] = acc[i];
            }
          }
          // Every wave in the group must have landed its partial before any
          // wave reads them, and the next tile_iter overwrites these slots.
          //
          // FFM CANNOT VERIFY THIS BARRIER. Deleting the equivalent one in
          // gang_rmsnorm_linear_mxfp4_bias_mi450 was run as a mutant and all
          // four shapes still passed bit-exactly. Do not "simplify" it away on
          // the strength of a green suite; the suite cannot observe it.
          __syncthreads();

          // One thread per (group, gate/up pair, token column). Strided
          // because W13_N_GROUPS * 8 * n_tok can exceed the block size.
          int const n_out = W13_N_GROUPS * 8 * n_tok;
          for (int idx = tid; idx < n_out; idx += blockDim.x) {
            int const t_grp = idx / (8 * n_tok);
            int const rem = idx - t_grp * (8 * n_tok);
            int const t_pair = rem / n_tok; // 0..7 -> rows 2*t_pair, +1
            int const t_col = rem - t_pair * n_tok;
            int const t_row = t_pair * 2;

            float g_sum = 0.0f, u_sum = 0.0f;
#pragma unroll
            for (int w = 0; w < W13_WAVES_PER_TILE; w++) {
              int const slab = (t_grp * W13_WAVES_PER_TILE + w) * REDUCE_SLAB;
              g_sum += s_reduce[slab + t_row * TOK_ROWS + t_col];
              u_sum += s_reduce[slab + (t_row + 1) * TOK_ROWS + t_col];
            }

            int const out_n = wg_idx * W13_OUTPUT_PER_WG +
                              (t_grp + tile_iter * W13_N_GROUPS) * 16 + t_row;
            if (out_n + 1 < W13_OUTPUT_SIZE) {
              unsigned bt_g =
                  (unsigned)d_w13_bias[expert_id * W13_OUTPUT_SIZE + out_n]
                  << 16;
              unsigned bt_u =
                  (unsigned)d_w13_bias[expert_id * W13_OUTPUT_SIZE + out_n + 1]
                  << 16;
              float bias_g, bias_u;
              __builtin_memcpy(&bias_g, &bt_g, 4);
              __builtin_memcpy(&bias_u, &bt_u, 4);
              float activated =
                  fast_swigluoai(g_sum + bias_g, u_sum + bias_u);
              int const act_n = out_n / 2;
              int const out_idx = col_tok(t_col) * (NUM_TOPK * ACT_STRIDE) +
                                  col_slot(t_col) * ACT_STRIDE + act_n;
              st_wt_u16(&d_swiglu_out[out_idx],
                        _gang_float_to_bf16_mi450(activated));
            }
          }
          // The next tile_iter reuses these slabs.
          __syncthreads();
        }
      }

      // The SwiGLU stores must be visible before this tile's arrival is
      // counted, or a W2 worker released by that arrival can read a row that
      // has not landed. This is the release side of the barrier and the reason
      // the wait is here rather than after the atomic.
      mirage::arch::wait_vmem();
      __syncthreads();
    } // end W13 compute region

  w13_arrive:
    MPK_WS_MARK(8201, global_tile); // W13 done, arriving at barrier
    if (tid == 0) {
      // Indexed by SLOT, not expert_id: the id comes from the shared mask and
      // can differ between two workers straddling a layer boundary, which would
      // split one slot's arrivals across two counters. The slot is derived from
      // the compile-time tile space, so every worker agrees on it.
      int const base = expert_idx * MOE450_BAR_STRIDE;

      // Single global arrival; all W13 tiles increment one counter.
      //
      // The counter must NOT share a cache line with the release slots below.
      // See the layout note at the top of this file -- this is the captured
      // gfx950 deadlock, and the mechanism is a cache property, not an ISA one.
      int prev_global = atom_add_release_gpu_s32(
          &d_barrier[base + MOE450_BAR_COUNTER_SLOT * MOE450_BAR_LINE], 1);

      // Exact only if every tile counted by W13_TILES arrives here. Under
      // packing W13_TILES == W13_WGS and no tile returns between the decode and
      // this line, so it is.
      if ((prev_global % W13_TILES) == W13_TILES - 1) {
        constexpr int LAYER_IDX_SMEM_OFF =
            mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE -
            mirage::runtime::LAYER_IDX_SMEM_OFFSET_FROM_END;
        int layer_idx =
            *reinterpret_cast<int *>(&_fused_smem[LAYER_IDX_SMEM_OFF]);
        int release_val = layer_idx + 1;
        for (int x = 0; x < 8; x++) {
          st_wt_u32((void *)&d_barrier[base + x * MOE450_BAR_LINE],
                    (unsigned)release_val);
        }
        mirage::arch::wait_vmem();
      }
    }
    return;
  }

  // ════════════════════════════════════════════════════════════════════════
  // PHASE 1: W2 (down projection) -> accumulate into the f32 workspace
  // ════════════════════════════════════════════════════════════════════════

  // No routed token: nothing to reduce, and unlike W13 there is no arrival to
  // preserve, so this returns outright. It must return BEFORE the poll below:
  // an empty slot's W13 tiles reach the arrival but write no output, so waiting
  // on that release would be waiting to read nothing.
  if (n_tok == 0) {
    MPK_WS_MARK(8106, global_tile); // exit: W2 tile with no routed token
    return;
  }

  MPK_WS_MARK(8300, global_tile); // W2 entry
  {
    uint8_t *s_tok_scales = s_tok_fp8 + TOK_ROWS * W2_TOK_ROW_STRIDE;

    uint8_t const *expert_weight =
        W_down + static_cast<int64_t>(expert_id) * W2_EXPERT_BYTES;
    uint8_t const *wg_data =
        expert_weight + static_cast<int64_t>(wg_idx) * W2_WG_BYTES;
    uint8_t const *wg_scales = wg_data + W2_WG_DATA;

    mx_global_u8 *g_data = mx_to_global(wg_data);
    mx_global_u8 *g_scales = mx_to_global(wg_scales);

    constexpr int W2_K_PER_WAVE = W2_K / W2_WAVES_PER_TILE;
    int const grp = warp_id / W2_WAVES_PER_TILE;
    int const krank = warp_id % W2_WAVES_PER_TILE;

    // ── Weight prefetch issued BEFORE the barrier poll ──────────────────
    // This is what survives of the gfx950 "issue buffer_load_lds before the
    // poll" overlap. Legal here because the WEIGHTS do not depend on the
    // barrier -- only the SwiGLU input does. GLOBAL_PREFETCH_B8 returns no
    // data and touches no counter, so it needs no wait and cannot be observed
    // by anything downstream; if the poll turns out to be short, the worst
    // case is a wasted prefetch.
    {
      int const pf_tile = grp + 0 * W2_N_GROUPS;
      int const pf_row = pf_tile * 16 + op_mn;
      int const pf_k = krank * W2_K_PER_WAVE;
#pragma unroll
      for (int j = 0; j < 2; j++) {
        int const k = pf_k + j * MX_K_PER_TILE;
        if (k < pf_k + W2_K_PER_WAVE) {
          mx_prefetch_operand_fp4_global(g_data, pf_row, k, W2_K);
        }
      }
    }

    int const base = expert_idx * MOE450_BAR_STRIDE;
    int w2_expected;
    {
      constexpr int LAYER_IDX_SMEM_OFF =
          mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE -
          mirage::runtime::LAYER_IDX_SMEM_OFFSET_FROM_END;
      int layer_idx =
          *reinterpret_cast<int *>(&_fused_smem[LAYER_IDX_SMEM_OFF]);
      w2_expected = layer_idx + 1;
    }

    // ── W13 -> W2 acquire ───────────────────────────────────────────────
    // Every thread polls independently and the poll is DELIBERATELY divergent:
    // there is no __syncthreads here, so waves leave as they clear. Adding one
    // would be wrong, not merely slow -- a block whose waves are split across
    // the barrier is the expected state.
    {
      int const expected = w2_expected;
      // The barrier id encodes the expert so a dump can say which activated
      // expert never got its W13 release.
      MPK_WS_WAIT_BEGIN(800 + expert_idx, expected);
      MPK_WS_WAVE_CLEAR(warp_id);
      int _obs;
      int _spins = 0;
      while ((_obs = ld_nt_s32(&d_barrier[base + xcd_id * MOE450_BAR_LINE])) <
             expected) {
        MPK_WS_WAIT_TICK(_obs, _spins);
        if ((_spins & (MPK_WS_WAIT_REFRESH - 1)) == 0) {
          // Refresh the discriminating values: the raw arrival counter
          // (whether it sits on a multiple of W13_TILES separates "release
          // fired but was lost" from "arrivals never landed") and how many of
          // the 8 per-XCD slots agree. All 8 are written by one producer in
          // one loop, so any spread means releases are being lost.
          int _n_ok = 0, _mn = 0x7fffffff, _mx = -0x7fffffff;
          for (int _x = 0; _x < 8; _x++) {
            int _v = ld_nt_s32(&d_barrier[base + _x * MOE450_BAR_LINE]);
            if (_v >= expected) {
              _n_ok++;
            }
            if (_v < _mn) {
              _mn = _v;
            }
            if (_v > _mx) {
              _mx = _v;
            }
          }
          MPK_WS_WAIT_AUX(ld_nt_s32(&d_barrier[base + MOE450_BAR_COUNTER_SLOT *
                                                          MOE450_BAR_LINE]),
                          expert_id,
                          _n_ok * 1000000 + (_mx - _mn),
                          -1);
        }
        _spins++;
        __builtin_amdgcn_s_sleep(1);
      }
      MPK_WS_WAVE_EXIT(warp_id);
    }
    MPK_WS_MARK(8302, global_tile); // W2: cleared W13->W2 barrier

    // No cache invalidate needed: the reads below are non-temporal and the
    // releasing stores were write-through, so neither side is served by L2.

    // ── Quantize the SwiGLU output ──────────────────────────────────────
    MPK_WS_MARK(8303, global_tile);
    if constexpr (PACK_N && !SINGLE_TOK) {
      // s_row_off already holds tok*(NUM_TOPK*INTERMEDIATE) + slot*INTERMEDIATE
      // for this phase. NT loads: this was written by a W13 tile on another XCD
      // and is read exactly once.
      mx_quant_fp8_gather<W2_K, TOK_ROWS, W2_TOK_ROW_STRIDE, W2_SC_STRIDE,
                          /*NT_LOAD=*/true>(
          d_swiglu_out, s_row_off, n_tok, s_tok_fp8, s_tok_scales);
    } else {
      unsigned short const *w2_input_base =
          d_swiglu_out + my_tok * (NUM_TOPK * INTERMEDIATE_SIZE) +
          topk_slot * INTERMEDIATE_SIZE;
      mx_quant_fp8_nt<W2_K>(w2_input_base, s_tok_fp8, s_tok_scales);
    }
    __syncthreads();

    int const b_row = (TOK_ROWS == 1) ? 0 : (tok_active ? op_mn : 0);
    uint8_t const *b_tok = s_tok_fp8 + b_row * W2_TOK_ROW_STRIDE;
    uint8_t const *b_scl = s_tok_scales + b_row * W2_SC_STRIDE;

    // The workspace stores below are 4-wide when a lane owns 8 contiguous
    // output columns. Both terms of the address must be 4-float aligned for
    // that; asserted rather than assumed.
    static_assert(W2_OUTPUT_PER_WG % 4 == 0 && HIDDEN_SIZE % 4 == 0,
                  "float4 workspace stores require 4-element aligned output "
                  "tiles and hidden size");

    MPK_WS_MARK(8305, global_tile); // W2: WMMA loop
    for (int tile_iter = 0; tile_iter < W2_TILES_PER_GROUP; tile_iter++) {
      int const wave_tile = grp + tile_iter * W2_N_GROUPS;
      int const w_row = wave_tile * 16 + op_mn;

      mx_f32x8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

      if constexpr (W2_WAVES_PER_TILE == 1) {
        acc = mx_k_loop_f4xf8<W2_K>(g_data, g_scales, w_row, b_tok, b_scl, acc);
      } else {
        int const k_off = krank * W2_K_PER_WAVE;
#pragma unroll 1
        for (int kt = k_off; kt < k_off + W2_K_PER_WAVE; kt += MX_K_PER_TILE) {
          mx_i32x16_t a = mx_load_operand_fp4_global(g_data, w_row, kt, W2_K);
          mx_i32x16_t b = mx_load_operand_fp8(b_tok, 0, kt, W2_K);
          unsigned int sa = mx_pack_scales_global(
              g_scales + (size_t)w_row * W2_NUM_BLK32 +
              kt / MX_K_PER_SCALE_BLOCK);
          unsigned int sb = mx_pack_scales(b_scl + kt / MX_K_PER_SCALE_BLOCK);
          acc = kernel::mi450::_gang_wmma_f4xf8(a, b, acc, (int)sa, (int)sb);
        }
      }

      // ── Workspace epilogue ────────────────────────────────────────────
      // Plain write-through stores, NOT atomicAdd: this tile is the sole
      // writer of its (token, top-k slot, hidden range), and the consumer runs
      // on another XCD. See moe_ws_layout.cuh for why there is no per-layer
      // zeroing to race against.
      if constexpr (W2_WAVES_PER_TILE == 1) {
        int const out_n_base =
            wg_idx * W2_OUTPUT_PER_WG + wave_tile * 16 + mx_acc_row(lane_id, 0);
        if (tok_active && out_n_base + 7 < W2_OUTPUT_SIZE) {
          float const rw = d_routing_weight[my_tok * NUM_TOPK + topk_slot];
          unsigned short const *bias =
              &d_w2_bias[expert_id * W2_OUTPUT_SIZE + out_n_base];
          int const ws_base =
              moe_ws_offset(my_tok, topk_slot, HIDDEN_SIZE) + out_n_base;
#pragma unroll
          for (int q = 0; q < 8; q += 4) {
            float bv[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
              unsigned bt = (unsigned)bias[q + j] << 16;
              __builtin_memcpy(&bv[j], &bt, 4);
            }
            float4 wv = {(acc[q + 0] + bv[0]) * rw,
                         (acc[q + 1] + bv[1]) * rw,
                         (acc[q + 2] + bv[2]) * rw,
                         (acc[q + 3] + bv[3]) * rw};
            st_wt_f32x4(&d_workspace_f32[ws_base + q], wv);
          }
        }
      } else {
        if (tok_active) {
#pragma unroll
          for (int i = 0; i < 8; i++) {
            s_reduce[warp_id * REDUCE_SLAB + mx_acc_row(lane_id, i) * TOK_ROWS +
                     acc_col] = acc[i];
          }
        }
        // Load-bearing, and unverifiable under FFM -- see the W13 twin above.
        __syncthreads();

        int const n_out = W2_N_GROUPS * 16 * n_tok;
        for (int idx = tid; idx < n_out; idx += blockDim.x) {
          int const t_grp = idx / (16 * n_tok);
          int const rem = idx - t_grp * (16 * n_tok);
          int const t_row = rem / n_tok;
          int const t_col = rem - t_row * n_tok;

          float v = 0.0f;
#pragma unroll
          for (int w = 0; w < W2_WAVES_PER_TILE; w++) {
            v += s_reduce[(t_grp * W2_WAVES_PER_TILE + w) * REDUCE_SLAB +
                          t_row * TOK_ROWS + t_col];
          }

          int const out_n = wg_idx * W2_OUTPUT_PER_WG +
                            (t_grp + tile_iter * W2_N_GROUPS) * 16 + t_row;
          if (out_n < W2_OUTPUT_SIZE) {
            int const o_tok = col_tok(t_col);
            int const o_slot = col_slot(t_col);
            float const rw = d_routing_weight[o_tok * NUM_TOPK + o_slot];
            unsigned bt =
                (unsigned)d_w2_bias[expert_id * W2_OUTPUT_SIZE + out_n] << 16;
            float bv;
            __builtin_memcpy(&bv, &bt, 4);
            float const out = (v + bv) * rw;
            unsigned ou;
            __builtin_memcpy(&ou, &out, 4);
            int const ws_base =
                moe_ws_offset(o_tok, o_slot, HIDDEN_SIZE) + out_n;
            st_wt_u32((void *)&d_workspace_f32[ws_base], ou);
          }
        }
        // The next tile_iter reuses these slabs.
        __syncthreads();
      }
    }
  }

  __syncthreads();

  // No barrier reset: every counter uses a monotonically increasing expected
  // value (per-XCD release = layer_idx + 1, arrivals checked modulo
  // W13_TILES), which is what keeps stale L2 lines from being mistaken for a
  // fresh release on the next layer.
}

#endif // MIRAGE_ARCH_GFX1250

} // namespace kernel
