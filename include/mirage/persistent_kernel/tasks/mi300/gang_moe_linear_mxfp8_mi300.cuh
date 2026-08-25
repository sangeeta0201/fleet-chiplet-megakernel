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

// Gang MoE W13/W2 with MXFP8 weights and FP8xFP8 MFMA (gfx950).
//
// This is the largest single item in the token's weight budget: five active
// experts per layer move 94.4 MB of the 180.9 MB a MoE layer touches, which is
// 47.4% of the whole 9.17 GB token. Halving it is worth more than every
// scheduling change on this branch put together.
//
// Two lineages meet here. The GEMM body is gpt-oss's
// gang_moe_linear_mxfp4_kernel_mi300 -- flat round-robin tile distribution over
// the 8 XCDs, per-token tiles, on-the-fly FP8 quantization of the activation
// into LDS -- with the weight operand widened 4 bits to 8 exactly as
// gang_linear_mxfp8_mi300.cuh does it, and with that file's depth-4 software
// pipeline over the k-loop. The epilogues are GLM's, transplanted unchanged
// from gang_moe_linear_mi300.cuh: the fused SiLU-mul on W13 and the fused
// topk-weight + f32 atomicAdd on W2. Those two fusions are worth ~500 us
// between them and are independent of the weight format, so the port keeps
// them rather than reverting to gpt-oss's swigluoai and its separate reduce.
//
// Structural difference from GLM's CK kernels worth naming: those tile W13's M
// dimension in blocks of 16 tokens, while this decomposes per token like the
// gpt-oss kernel does. At batch 1 the two produce the same 48 tiles per expert,
// so tiles_per_expert does not move; above batch 1 the tile count grows
// linearly here instead of in steps of 16.
//
// MPK_SHARED_DUP -- CORRECT-OUTPUT pricing probe for the shared-expert
// imbalance. Long note at the define in mpk_atoms.cuh. Guarded here as well
// because the tile decode below is included by callers that do not pull in
// mpk_atoms.cuh first.
#ifndef MPK_SHARED_DUP
#define MPK_SHARED_DUP 0
#endif
//
// See gang_linear_mxfp8_mi300.cuh for why the scale indexing is what it is --
// the MFMA addresses its scale operand by matrix position, not by which bytes
// the lane loaded, and the two only coincide at 4 bits.
//
// Weight format, per workgroup of OUTPUT_PER_WG rows of one expert:
//   [FP8 E4M3 data: OPW * K bytes][E8M0 scales: OPW * (K/32) bytes]
// Experts are contiguous: expert e starts at e * EXPERT_WGS * WG_BYTES.
//
// WEIGHT_FP4 narrows that data half to OPW * (K/2) bytes of E2M1 nibbles and
// nothing else. The scales are already E8M0-per-32 in both formats, the tile
// decode, the LDS activation and both epilogues are format-blind, and the MFMA
// is the same v_mfma_scale_f32_16x16x128_f8f6f4 with cbsz switched from FP8 to
// FP4 -- so the two paths differ in exactly three expressions (the row stride,
// the A-operand load, and cbsz) and are one template rather than two files.
// The activation stays FP8: this is W4A8, and quantizing the token to 4 bits
// as gpt-oss's f4xf4 path does would spend the quality budget twice over for
// no bandwidth, since the token is 2 KB against the weight's 48 MB.

#pragma once
// Inherits the same include ordering as gang_linear_mxfp8_mi300.cuh; see the
// note there. silu_mul_mi300.cuh is named explicitly rather than inherited
// because unlike gang_moe_linear_mi300.cuh -- the other header that supplies
// fast_silu -- it is CK-free, so naming it costs nothing.
#include "tasks/mi300/gang_linear_mxfp8_mi300.cuh"
#include "tasks/mi300/silu_mul_mi300.cuh"

// W2 reduction split. Compile-time, and the host multiplies W2's dispatched
// tile count by the same number, so every rank must build with the same value
// or the MoE loop bound and the tile space disagree. See the note at the
// K_SPLITS constant in the W2 kernel.
#ifndef MPK_W2_KSPLIT
#define MPK_W2_KSPLIT 1
#endif

// Restore W2's pre-fdca420 activation staging: quantize the WHOLE reduction
// into LDS whatever K window the tile will consume. Only meaningful with
// MPK_W2_KSPLIT > 1 (at 1 the window IS the whole reduction). Exists so the
// staging A/B is one -D inside one build instead of a cross-batch comparison.
#ifndef MPK_W2_STAGE_FULL
#define MPK_W2_STAGE_FULL 0
#endif

// Hand the W13 -> W2 activation over as MXFP8 rather than bf16, so W2's tiles
// stage bytes instead of re-deriving them. Producer side is the W13 kernel's
// EMIT_FP8 template parameter, consumer side is W2's INPUT_FP8; they are one
// knob because the layouts have to agree. 0 restores the bf16 handoff, which
// is what makes this an A/B inside a single build.
//
// MEASURED NEUTRAL, defaulted OFF. Same-batch A/B, n=3 each, correct output on
// all 4 correctness prompts: OFF 10.724 (10.623/10.772/10.777), ON 10.824
// (10.751/10.777/10.943). +0.100 ms, inside the 0.26 ms noise floor and the
// wrong sign. The redundancy is real -- 96 W2 tiles per expert each re-derive
// the same 2048-element activation -- but it is 4 KB against a 68 KB weight
// tile, the quantize is VALU work that overlaps the tile's own MFMA loop, and
// the producer side has to pay a __syncthreads plus an LDS round trip to emit
// the E8M0-per-32 layout. Deleting redundant work inside a GLM phase is
// absorbed; this is the sixth instance. Kept behind the knob because the
// FP8-handoff plumbing is the prerequisite for any future W2 change that
// consumes MXFP8 directly.
#ifndef MPK_MOE_ACT_FP8
#define MPK_MOE_ACT_FP8 0
#endif

// ── K-MAJOR WEIGHT ─────────────────────────────────────────────────────────
//
// MPK_MOE_KMAJOR: permute the data half of the per-workgroup weight so a
// wave's k-group is contiguous. 0 = the row-major layout everything shipped
// with; 1 = data half K-major; 2 = data and E8M0 scales both K-major.
//
// The bytes are identical and so is the arithmetic -- this is a permutation of
// the packed buffer plus the address expression that reads it. The lever is
// COALESCING, not traffic. Row-major stores a workgroup as [row][k], so the
// sixteen rows one wave covers at k-tile k are sixteen 64-byte pieces (FP4)
// W_ROW_BYTES = 3072 apart. One `global_load_dwordx4` across the 64 lanes
// therefore fans out to sixteen distinct 128-byte L2 requests, each half used
// on this trip. K-major stores it as [row/16][k][row%16], which makes the same
// wave read 16*64 = 1024 CONTIGUOUS bytes -- eight fully-used requests.
//
// Row-major does not over-fetch: the unused half of each line is consumed at
// k+1, and at MPK_MOE_PF_GROUPS=4 four k-tiles (256 B/row) are in flight. So
// the byte count is unchanged and only the request count moves, 16 -> 8 for
// the data at FP4 and 16 -> 1 for the scales at KMAJOR=2 (sixteen 4-byte reads
// K/32 = 192 apart become one 64-byte run).
//
// Predicted from tests/standalone/test_w13_scale_locality.hip, which measured
// this exact permutation on top of the shipping GROUPS=4:
//
//   GROUPS=4                        11.85 us/tile   16.6 GB/s/CU
//   GROUPS=4 + K-major weight        9.70           20.3   <- -2.15, -18%
//
// At the conversion rate the prefetch itself established (-4.44 us/tile
// standalone bought -0.142 ms of wall at NP=4) that projects to about -0.07 ms
// -- under the 0.26 ms noise floor, so it needs n>=6 pooling to resolve.
//
// EVERY RANK MUST BUILD WITH THE SAME VALUE, and so must the packer:
// pack_mxfp8_workgroup in demo/glm5/demo.py reads MPK_MOE_KMAJOR and applies
// the same permutation to the MoE call sites only. The dense/attention-half
// weights go through pack_dense_mxfp8 into a different kernel and stay
// row-major.
// MEASURED AT NP=4, bs=1, devices 4-7, paired n=8 vs n=8 (clean runs only;
// this box intermittently wedges a run at launch_persistent_kernel ENTER at
// BOTH settings, so hung runs are excluded from both arms):
//
//   KMAJOR=0   mean 10.960   range 10.933 .. 11.006
//   KMAJOR=1   mean 10.682   range 10.639 .. 10.720   -0.278 ms
//
// The ranges do not overlap, so this clears the 0.26 ms noise floor on the
// pooled comparison rather than on a single pair. It is 4x the -0.07 ms this
// header projected from the MPK_MOE_PF_GROUPS conversion rate -- that estimate
// priced the coalescing win as if it were a bandwidth win, and it is not: the
// tile class is latency-bound, so collapsing 16 L2 requests to 8 buys back
// request-issue latency, not bytes.
//
// Correctness: G1 (cross-rank agreement) PASSES 4/4 ranks; the generated text
// is coherent and factually correct. G2 flags the continuation as repetitive,
// but the control scores the same top-bigram count on the same prompt -- it is
// the documented bs=1 two-attractor split, not a numerics failure.
//
// LEVEL 2 -- THE SCALE HALF -- MEASURED 2026-08-25, and it is the default.
//
// Everything above is about the DATA half. Collapsing it to 8 requests left
// the E8M0 scales exactly as they were: four bytes per row per k-tile, so one
// wave still gathers sixteen 4-byte pieces K/32 apart. That is 16 L2 requests
// for 64 useful bytes -- WORSE, per instruction, than the data half was before
// this file touched it. The scales are ~3% of the weight bytes, which is why
// no byte-side argument ever looked at them and why level 2 sat implemented on
// both the packer and the kernel side for two days without ever being run.
// Bytes are the wrong meter for this class; request count is the meter.
//
// _gang_moe_sc_kstride() above already switches 4 -> 64 at level 2, and
// pack_mxfp8_workgroup's `kmajor >= 2` branch already emits the matching
// _kmajor_permute(scales, opw, 4). This was a measurement, not a build.
//
// Paired against the shipping operating point (66edd8d: dense K-major at
// level 2, MoE at level 1), NP=4 bs=1 on devices 4-7:
//
//   MPK_MOE_KMAJOR    per-iter min                 clean avg
//   1 (control)       10.343  n=6, 10.262..10.409  10.504  n=6
//   2                 10.280  n=5, 10.248..10.335  10.431  n=4
//                     -0.063                       -0.073
//
// Both statistics agree in sign and in magnitude, and four of the arm's five
// runs sit below the control's BEST run. n=4 on the avg because one arm run
// took a 2.7 s single-iteration stall (this box does that intermittently at
// any setting); per-iter min is immune to it, which is why it is the primary
// statistic here. -0.063 is under the 0.26 ms single-run wall noise floor, so
// it is the pooled distributions that carry it, not any one pair.
//
// The dense twin (MPK_DENSE_KMAJOR 1 -> 2) bought -0.086 by the same
// mechanism. Together the scale half is worth -0.149 ms.
//
// Correctness: G1 (cross-rank) PASSES 4/4 ranks on every prompt. Prompt 0
// scores a G2 coherence FAIL at distinct=0.087 / topbigram=36 -- the control
// tag `dkmaj` scores 0.087 / 33 on that same prompt with the same reasoning
// trace and the same correct answer, so it is the documented bs=1
// two-attractor split and not this layout.
#ifndef MPK_MOE_KMAJOR
#define MPK_MOE_KMAJOR 2
#endif

namespace kernel {

// The weight operand at whichever width it is packed. FP4 fills only the lower
// 128 bits of the i32x8 and is contiguous there, so it is one 16-byte load
// against FP8's split gather of two.
template <bool FP4>
__device__ __forceinline__ i32x8_t _gang_load_w_mfma_a(uint8_t const *row,
                                                       int kt,
                                                       int g) {
  if constexpr (FP4) {
    return _gang_load_fp4_mfma_b(row, kt, g);
  } else {
    return _gang_load_fp8_mfma_b(row, kt, g);
  }
}

// The same load addressed by a BYTE offset the caller has already scaled,
// rather than by a k-element index this helper halves for FP4. Both layouts
// (row-major and K-major) then reduce to `base + k * w_kstride + g*16`, so the
// k-loops carry one runtime stride instead of a layout template parameter.
template <bool FP4>
__device__ __forceinline__ i32x8_t
    _gang_load_w_mfma_a_at(uint8_t const *base, int byte_off, int g) {
  // kt = 0: _gang_load_fp4_mfma_b halves it and _gang_load_fp8_mfma_b does
  // not, so passing zero makes the two paths agree on a byte-addressed base.
  if constexpr (FP4) {
    return _gang_load_fp4_mfma_b(base + byte_off, 0, g);
  } else {
    return _gang_load_fp8_mfma_b(base + byte_off, 0, g);
  }
}

// Global-addressed twin of the above, for MPK_MOE_WGLOBAL. Identical
// addressing; the gathers go through _gang_ld_g so they emit global_load
// instead of flat_load and stop incrementing lgkmcnt.
template <bool FP4>
__device__ __forceinline__ i32x8_t
    _gang_load_w_mfma_a_at_g(uint8_t const *base, int byte_off, int g) {
  if constexpr (FP4) {
    return _gang_load_fp4_mfma_b_g(base + byte_off, 0, g);
  } else {
    return _gang_load_fp8_mfma_b_g(base + byte_off, 0, g);
  }
}

// Bytes one weight row contributes to one 128-element k-tile. W_ROW_BYTES is
// exactly MFMA_ITERS of these.
template <bool FP4>
__device__ __forceinline__ constexpr int _gang_moe_bytes_per_ktile() {
  return FP4 ? 64 : 128;
}

// The two runtime strides the k-loop walks, given the layout. Data: the next
// k-tile of THIS row is one row-chunk away when row-major, sixteen away when
// K-major (the intervening fifteen belong to the other rows of the 16-row
// block this wave covers). Scales: 4 E8M0 bytes per k-tile per row, so 4 when
// row-major and 16*4 when K-major.
template <bool FP4>
__device__ __forceinline__ constexpr int _gang_moe_w_kstride() {
  return (MPK_MOE_KMAJOR >= 1 ? 16 : 1) * _gang_moe_bytes_per_ktile<FP4>();
}
__device__ __forceinline__ constexpr int _gang_moe_sc_kstride() {
  return MPK_MOE_KMAJOR >= 2 ? 64 : 4;
}

template <bool FP4>
__device__ __forceinline__ f32x4_t _gang_mfma_w_x_f8(
    i32x8_t a, i32x8_t b, f32x4_t c, int scale_a, int scale_b) {
  if constexpr (FP4) {
    return _gang_mfma_f4xf8(a, b, c, scale_a, scale_b);
  } else {
    return _gang_mfma_f8xf8(a, b, c, scale_a, scale_b);
  }
}

// ── DEEP PREFETCH ──────────────────────────────────────────────────────────
//
// MPK_MOE_PF_GROUPS: how many k-groups the MoE k-loop holds in registers while
// that many more are in flight. 0 keeps the shipping loop below byte for byte.
//
// WHY. The shipping loop declares a0..a3 and reloads each one immediately
// after the MFMA that consumes it, which reads as a depth-4 software pipeline
// but is not one. Dumped ISA, and every k-iteration compiles to:
//
//   .LBB_11:                      ; k-loop header
//     global_load_dwordx4 x4      ; a0..a3 for the NEXT k-group
//     global_load_ubyte   x4      ; its scales
//     s_waitcnt vmcnt(1)          ; <-- waits for 7 of the 8, immediately
//     4x v_mfma_scale_f32_16x16x128_f8f6f4
//     s_waitcnt vmcnt(0)
//
// A load and its use are ~3 MFMAs apart, about 96 cycles, against an HBM
// latency near 800. There is effectively no overlap.
//
// Measured in tests/standalone/test_w13_scale_locality.hip against the real
// W13 geometry at the real live grid width of 64, 306 MB so the MALL cannot
// serve it (14560af). Same arithmetic, same row-major weight, same global
// scales -- the ONLY change is how far ahead the loads are issued:
//
//   shipping loop, reproduced   16.22 us/tile   12.1 GB/s/CU
//   GROUPS=2                    16.47           11.9   <- where shipping sits
//   GROUPS=4                    11.80           16.7
//   GROUPS=8                    11.62           16.9
//   GROUPS=16                   12.79           15.4   <- past the knee
//
// 16 is measured worse, matching test_narrow_grid_bandwidth.hip, where depth 4
// is the optimum at every grid width and depth 8 is 15% worse at narrow ones.
//
// The same file measures the two candidate multipliers ON TOP OF GROUPS=4,
// which is what ships. Both had only ever been measured on top of the
// standalone's own optimum of 8, and one of them does not survive the move:
//
//   GROUPS=4                        11.85 us/tile   16.6 GB/s/CU
//   GROUPS=4 + LDS-staged scales    12.24           16.1   <- +0.39, NEGATIVE
//   GROUPS=4 + K-major weight        9.70           20.3   <- -2.15, -18%
//   GROUPS=4 + both                 10.92           18.0
//
// LDS scale staging pays at depth 8 (10.72 vs 11.66) and costs at depth 4. The
// file's own arms 3/5 already showed the scale load is nearly free and that
// what the LDS arm really buys is a coalesced burst from ANY address, so it was
// never a scale fix; at depth 4 the prefetch has already bought that pacing and
// the extra barriers are pure cost.
//
// K-MAJOR IS THE REMAINING LEVER AND IT IS UNBUILT. It repacks the weight so a
// wave's k-group is 1024 contiguous bytes instead of sixteen 64-byte pieces
// strided W_ROW_BYTES apart. Same bytes, same MFMA, same pipeline -- purely a
// packer change (pack_mxfp8_workgroup in demo/glm5/demo.py) plus the address
// arithmetic here. At the PF conversion rate measured above (-4.44 us/tile
// standalone bought -0.142 ms of wall) it projects to about -0.07 ms, which is
// under the 0.26 ms noise floor and needs n=6 pooling to resolve.
//
// IT IS NULL AT NP=8 AND A WIN AT NP=4. Default is 4, which is the NP=4
// optimum. Both halves are measured; the NP=8 nulls below are kept because they
// are the reason this stayed off for so long, not because they generalize.
//
// NP=8, three paired same-batch A/Bs, MPK_NUM_WORKERS=232, all three null:
//
//   W13, guarded form      control 10.460  ->  PF=4 10.491   (+0.031, n=3)
//   W13, guard-free form   control 10.503  ->  PF=4 10.517   (+0.014, n=3)
//   W2,  guard-free form   control 10.526  ->  PF=4 10.521   (-0.005, n=6)
//
// All inside the 0.26 ms noise floor against a predicted -0.37. The first has
// an ISA cause (the guards forced a full drain; see the note on the helper);
// the other two do not. The W13 null was explained as absorption -- W13's 16.4
// us/layer is eaten by the spread of the barrier behind it -- but W2 is the
// UNABSORBED phase (memory: glm-w2-imbalance-is-the-only-unabsorbed-spread) and
// read null too, so absorption was never the whole story.
//
// NP=4 (devices 4,5,6,7, the shipping config), same knob, both call sites live,
// two independent batches of n=3 pooled per arm:
//
//   control  n=6  mean 11.441  sd 0.087  min 11.344
//   PF=4     n=6  mean 11.298  sd 0.016  min 11.280   (-0.142)
//   PF=8     n=3  mean 11.701           min 11.621   (+0.260, clearly worse)
//
// The two batches agree to 0.014 ms and the populations do not overlap; PF=4's
// run-to-run spread is 5x tighter than the control's. WHY IT CONVERTS HERE AND
// NOT AT NP=8: at NP=4 a rank owns ~2 activated experts instead of ~1, so the
// MoE phase runs 16 W13 and 24 W2 tiles per XCD against 29 workers rather than
// 8 and 12. The phase is one full round of nearly-saturated workers instead of
// a half-empty one, so per-tile time is the makespan rather than something the
// round quantization rounds away (memory:
// glm-moe-phase-is-round-quantized-and-w2-is-latency-bound).
//
// 8 is worse at the wall even though the standalone put its knee at 8; that
// matches test_narrow_grid_bandwidth.hip, where depth 4 wins at every grid
// width and depth 8 costs 15% at narrow ones.
#ifndef MPK_MOE_PF_GROUPS
#define MPK_MOE_PF_GROUPS 4
#endif

// MPK_MOE_WGLOBAL: address the weight *and its E8M0 scales* through
// addrspace(1), AND hoist the whole GROUPS-wide batch of B-operand ds_reads
// above the MFMA group. The two halves only work together, which is why they
// share one knob.
//
// THE PATHOLOGY, read off the shipped W13 body [0x41894..0x41a28]:
//
//   s_waitcnt vmcnt(0)                          <- top of trip, FULL DRAIN
//   flat_load_dwordx4 v[52:55], v[6:7]          \
//   flat_load_dwordx4 v[48:51], v[6:7] off:1024  |  the next block's weights
//   flat_load_dwordx4 v[36:39], v[6:7] off:2048 /
//   ds_read_b128 x5 ; ds_read_u8 x4             <- this block's B operand
//   s_waitcnt lgkmcnt(0)                        <- DRAINS THE WEIGHTS TOO
//   v_mfma_scale_f32_16x16x128_f8f6f4
//   ds_read_b128 ; s_waitcnt lgkmcnt(0) ; v_mfma
//   ds_read_b128 x2 ; s_waitcnt lgkmcnt(0) ; v_mfma
//
// Three `s_waitcnt lgkmcnt(0)` per trip, each of which -- because the weight
// loads are FLAT, and a flat op increments both vmcnt and lgkmcnt -- retires
// every outstanding weight load as well. The prefetch is dead on arrival:
// `isa_loads_in_flight.py` reports `issued 0 / minvm 0` on this loop, and
// MPK_MOE_PF_GROUPS=4 only buys anything because the four loads at the top
// overlap *each other* for one memory latency before being drained.
//
// WHY THE PREVIOUS ATTEMPT LOST. Task #77 cast the weight pointer to
// addrspace(1) and nothing else, and measured **+0.87 ms** with W13's
// s_waitcnt count going 40 -> 54 (memory:
// glm-moe-addrspace1-is-a-negative-waitcnt-count-is-the-metric). That is the
// expected result of doing half of this: with the weights out of lgkmcnt the
// compiler still needs the three lgkmcnt(0) for the interleaved ds_reads AND
// now needs separate vmcnt waits for the weights, so the count goes up while
// the pipeline stays broken. The B-side hoist is what removes the other two
// lgkm waits, and only then does the decoupling pay for itself.
//
// So the arithmetic per trip is 1 lgkm + partial vmcnts, against today's
// 1 full vmcnt + 3 full lgkm, and -- the point -- the vmcnt waits become
// PARTIAL, because vmcnt is in-order on gfx9 and global_load only touches it.
// GROUPS weight loads really do stay in flight across GROUPS MFMAs.
//
// Correctness: _gang_ld_g is only ever pointed at `w_data_row` and
// `wg_scales`, both of which are the megakernel workspace's weight buffer. The
// B operand keeps the plain (LDS) loader. Never point _gang_ld_g at LDS.
#ifndef MPK_MOE_WGLOBAL
// Measured -0.108 ms at NP=4 bs=1 (arm n=6 mean 9.841 vs control n=7 mean
// 9.949; the two ranges do not overlap). Default ON; =0 is the ablation.
#define MPK_MOE_WGLOBAL 1
#endif


// The k-loop, with the prefetch distance as a parameter. Accumulates
// [0, KI_END) and returns the MFMA accumulator; the caller owns the epilogue.
//
// EVERY LOAD AND EVERY MFMA HERE IS UNCONDITIONAL, and that is the whole
// point. The first version of this helper carried `if (kk < ki_end)` guards so
// GROUPS need not divide the range. It compiled, ran, produced correct output,
// and bought NOTHING (10.460 -> 10.491 ms, n=3 paired). The ISA says why: the
// guards put the prefetch loads in their own basic blocks, so the number of
// outstanding vm ops at the first MFMA differs per path and SIInsertWaitcnts
// falls back to a full drain --
//
//   flat_load_dwordx4 v[16:19], ...     ; next block, partially issued
//   s_waitcnt vmcnt(0) lgkmcnt(0)       ; <- waits for the prefetch too
//   v_mfma_scale_f32_16x16x128_f8f6f4   ; x4
//
// -- which is exactly the shipping loop's behaviour with more registers live.
// A prefetch that the wait drains is not a prefetch. Hence the compile-time
// trip count and the peeled last block: the steady-state body is straight
// line code, so the wait in front of the MFMAs can be a partial vmcnt and
// GROUPS loads really do stay in flight across GROUPS MFMAs.
//
// TOK_SC_FP8 selects the activation-scale layout: W2 stages an MXFP8 vector
// whose E8M0 run is per (block, g) and reads base[k*4+g]; W13 stages one scale
// per 32-element block and reads base[k]. Same loop otherwise.
template <bool WEIGHT_FP4, bool TOK_SC_FP8, int GROUPS, int KI_END>
__device__ __forceinline__ f32x4_t
    _gang_moe_kloop_deep(uint8_t const *w_data_row,
                         uint8_t const *wg_scales,
                         int row_scale_base,
                         uint8_t const *s_tok_fp8,
                         uint8_t const *s_tok_scales,
                         int g) {
  // Every caller declares this as a local constant of the same value; the
  // scaled MFMA's K is fixed by the instruction, not by the layer shape.
  constexpr int K_PER_MFMA = 128;
  static_assert(KI_END % GROUPS == 0,
                "MPK_MOE_PF_GROUPS must divide the k-loop trip count");
  constexpr int NBLK = KI_END / GROUPS;
  // Layout-dependent byte strides; both callers have already folded the layout
  // into w_data_row / row_scale_base, so the loop below is layout-blind.
  constexpr int W_KS = _gang_moe_w_kstride<WEIGHT_FP4>();
  constexpr int SC_KS = _gang_moe_sc_kstride();
  auto tok_sc_at = [&](int k) -> int {
    return TOK_SC_FP8 ? (int)s_tok_scales[k * 4 + g] : (int)s_tok_scales[k];
  };
  // Declared before `consume` captures it.
  f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};
  // The weight (A) operand and its E8M0 scale. Under MPK_MOE_WGLOBAL both go
  // through addrspace(1) so they leave lgkmcnt; see the knob's header.
  auto load_w = [&](int kk) -> i32x8_t {
    if constexpr (MPK_MOE_WGLOBAL) {
      return _gang_load_w_mfma_a_at_g<WEIGHT_FP4>(w_data_row, kk * W_KS, g);
    } else {
      return _gang_load_w_mfma_a_at<WEIGHT_FP4>(w_data_row, kk * W_KS, g);
    }
  };
  auto load_ws = [&](int kk) -> int {
    int const off = row_scale_base + kk * SC_KS + g;
    if constexpr (MPK_MOE_WGLOBAL) {
      return (int)_gang_ld_g<uint8_t>(wg_scales + off);
    } else {
      return (int)wg_scales[off];
    }
  };
  // Consume GROUPS k-tiles. Under MPK_MOE_WGLOBAL the whole batch of LDS
  // reads is issued BEFORE the first MFMA, so SIInsertWaitcnts needs one
  // lgkmcnt(0) for the batch instead of one per MFMA. That is the half of the
  // knob task #77 was missing: with the weights out of lgkmcnt, each of those
  // per-MFMA lgkm waits would still have drained the pipeline.
  auto consume = [&](int base, i32x8_t const *Ap, int const *Sp) {
    if constexpr (MPK_MOE_WGLOBAL) {
      i32x8_t B[GROUPS];
      int BS[GROUPS];
#pragma unroll
      for (int j = 0; j < GROUPS; j++) {
        B[j] = _gang_load_fp8_mfma_b(s_tok_fp8, (base + j) * K_PER_MFMA, g);
        BS[j] = tok_sc_at(base + j);
      }
#pragma unroll
      for (int j = 0; j < GROUPS; j++) {
        acc = _gang_mfma_w_x_f8<WEIGHT_FP4>(Ap[j], B[j], acc, Sp[j], BS[j]);
      }
    } else {
#pragma unroll
      for (int j = 0; j < GROUPS; j++) {
        i32x8_t b =
            _gang_load_fp8_mfma_b(s_tok_fp8, (base + j) * K_PER_MFMA, g);
        acc = _gang_mfma_w_x_f8<WEIGHT_FP4>(Ap[j], b, acc, Sp[j],
                                            tok_sc_at(base + j));
      }
    }
  };

  i32x8_t A[GROUPS];
  int S[GROUPS];
#pragma unroll
  for (int j = 0; j < GROUPS; j++) {
    A[j] = load_w(j);
    S[j] = load_ws(j);
  }

// IMPORTANT: #pragma unroll 1 prevents ROCm miscompilation. Same reason as the
// shipping loop; the GROUPS-wide bodies inside are fully unrolled.
#pragma unroll 1
  for (int blk = 0; blk < NBLK - 1; blk++) {
    int const base = blk * GROUPS;
    i32x8_t N[GROUPS];
    int NS[GROUPS];
    // Issue the whole NEXT block before consuming the current one, so GROUPS
    // loads are outstanding across GROUPS MFMAs instead of ~1 across 3.
#pragma unroll
    for (int j = 0; j < GROUPS; j++) {
      int kk = base + GROUPS + j;
      N[j] = load_w(kk);
      NS[j] = load_ws(kk);
    }
    consume(base, A, S);
#pragma unroll
    for (int j = 0; j < GROUPS; j++) {
      A[j] = N[j];
      S[j] = NS[j];
    }
  }

  // Peeled last block: consume what the previous iteration prefetched, and
  // issue nothing past the end of the row.
  constexpr int TAIL = (NBLK - 1) * GROUPS;
  consume(TAIL, A, S);
  return acc;
}

// ── DOUBLE-BUFFERED DEEP LOOP ──────────────────────────────────────────────
//
// WHAT IS WRONG WITH THE LOOP ABOVE, read off the shipped gfx950 code object
// (demo/glm5/isa_loads_in_flight.py over the W13 body at 0x3e614..0x3e7a8):
//
//   s_waitcnt vmcnt(0)                     <- top of trip, FULL DRAIN
//   12 x v_mov / v_mov_b64                 <- the A[j]=N[j] copy, materialised
//   flat_load_dwordx4  x4                  \  the 8 loads of the next block
//   flat_load_ubyte    x4                  /
//   4 x v_mfma_scale_f32_16x16x128_f8f6f4
//   s_waitcnt vmcnt(0)                     <- bottom of trip, FULL DRAIN AGAIN
//   v_mov_b32 v12, v87                     <- the reason for it: one scale byte
//   s_cbranch_scc1 <top>
//
// So the header's claim above -- "the wait in front of the MFMAs can be a
// partial vmcnt and GROUPS loads really do stay in flight across GROUPS MFMAs"
// -- is not what the compiler emitted. TWO vmcnt(0) per trip. The outstanding
// count returns to zero every k-block, which is the unroll=1 point on the
// curve tests/standalone/test_waves_per_simd_payoff.hip measured on this part:
//
//   loads in flight   1 -> 3446 GB/s      4 -> 5334      8 -> 5446
//
// The cause is `A[j] = N[j]; S[j] = NS[j];`. N[] are load destinations, so a
// copy out of them at the backedge is a use, and every outstanding load must
// land before the branch. SIInsertWaitcnts has no cheaper way to express it.
//
// THE FIX is to stop copying: unroll the block loop by two and swap the roles
// of the two buffers instead, so a value stays in the register its load wrote.
// Same registers live (A[] and N[] are both live in the loop above already),
// same loads, same MFMAs, same order of issue-then-consume. The only change is
// that the wait in front of the MFMAs can now name the eight just-issued loads
// and let them ride across the branch.
//
// Requires NBLK even and >= 4, which the two GLM-5 MoE shapes both satisfy at
// GROUPS=4: W13 KI_END=48 -> NBLK=12, W2 KI_END=16 -> NBLK=4. Anything else
// falls back to the copying form above rather than growing a remainder path --
// a remainder is a branch, a branch is a basic block, and the header's note on
// the guarded first draft records that a prefetch in its own basic block gets
// drained by the wait and is not a prefetch.
template <bool WEIGHT_FP4, bool TOK_SC_FP8, int GROUPS, int KI_END>
__device__ __forceinline__ f32x4_t
    _gang_moe_kloop_dbuf(uint8_t const *w_data_row,
                         uint8_t const *wg_scales,
                         int row_scale_base,
                         uint8_t const *s_tok_fp8,
                         uint8_t const *s_tok_scales,
                         int g) {
  constexpr int K_PER_MFMA = 128;
  constexpr int NBLK = KI_END / GROUPS;
  static_assert(KI_END % GROUPS == 0, "GROUPS must divide the trip count");
  static_assert(NBLK >= 4 && NBLK % 2 == 0, "dbuf needs an even NBLK >= 4");
  constexpr int W_KS = _gang_moe_w_kstride<WEIGHT_FP4>();
  constexpr int SC_KS = _gang_moe_sc_kstride();
  auto tok_sc_at = [&](int k) -> int {
    return TOK_SC_FP8 ? (int)s_tok_scales[k * 4 + g] : (int)s_tok_scales[k];
  };

  f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};
  i32x8_t A[GROUPS], N[GROUPS];
  int S[GROUPS], NS[GROUPS];

  // Issue block `b` into the given buffer. Taken by reference so the arrays
  // keep their identity -- passing them by value would reintroduce the copy
  // this whole variant exists to delete.
  auto issue = [&](i32x8_t (&dst)[GROUPS], int (&dsc)[GROUPS], int b) {
#pragma unroll
    for (int j = 0; j < GROUPS; j++) {
      int const kk = b * GROUPS + j;
      dst[j] = _gang_load_w_mfma_a_at<WEIGHT_FP4>(w_data_row, kk * W_KS, g);
      dsc[j] = (int)wg_scales[row_scale_base + kk * SC_KS + g];
    }
  };
  auto consume = [&](i32x8_t const (&src)[GROUPS], int const (&ssc)[GROUPS],
                     int b) {
#pragma unroll
    for (int j = 0; j < GROUPS; j++) {
      int const kk = b * GROUPS + j;
      i32x8_t bb = _gang_load_fp8_mfma_b(s_tok_fp8, kk * K_PER_MFMA, g);
      acc = _gang_mfma_w_x_f8<WEIGHT_FP4>(src[j], bb, acc, ssc[j],
                                          tok_sc_at(kk));
    }
  };

  issue(A, S, 0);

// IMPORTANT: #pragma unroll 1 prevents ROCm miscompilation. Same reason as the
// two loops above; the GROUPS-wide bodies inside are fully unrolled.
#pragma unroll 1
  for (int blk = 0; blk <= NBLK - 4; blk += 2) {
    issue(N, NS, blk + 1);
    consume(A, S, blk);
    issue(A, S, blk + 2);
    consume(N, NS, blk + 1);
  }

  // Tail: blocks NBLK-2 (already in A) and NBLK-1, issuing nothing past the
  // end of the row.
  issue(N, NS, NBLK - 1);
  consume(A, S, NBLK - 2);
  consume(N, NS, NBLK - 1);
  return acc;
}

// MPK_MOE_PF_DBUF: select the double-buffered form above over the copying one.
//
// MEASURED AND OFF: it does everything to the ISA it was supposed to do and it
// is 0.226 ms SLOWER, because it costs 26 unified VGPRs of the WHOLE
// megakernel's allocation.
//
// NP=4, devices 4-7, bs=1, one batch, arm first then control, n=5 each,
// per-iteration min (the stall-immune statistic on this box):
//
//   MPK_MOE_PF_DBUF=1   mean 10.492  min 10.428  max 10.528
//   MPK_MOE_PF_DBUF=0   mean 10.266  min 10.175  max 10.309   (shipping)
//
// The two populations do not overlap. The ISA change is exactly as designed --
// W13's steady-state body went from 60 instructions / 8 loads / 12 v_mov /
// TWO s_waitcnt vmcnt(0) per k-block, to 99 instructions / 16 loads / ZERO
// v_mov / ONE vmcnt(0) per TWO k-blocks. Drain frequency per MFMA fell 4x and
// every register copy is gone.
//
// What it cost, from the code object's own metadata (vgpr_spill_count is 0 in
// both, so this is allocation, not spilling):
//
//                     worker_kernel .vgpr_count   .agpr_count
//   DBUF=0                          284           36
//   DBUF=1                          310           58
//
// worker_kernel and persistent_kernel share one allocation for every tile
// kernel in the binary, so 26 VGPRs bought here are 26 VGPRs charged to the
// whole decode. That is the same lock the occupancy work hit from the other
// side (memory: glm-occupancy-lock-is-lds-not-registers,
// glm-agpr-is-spill-slack-lds-half-is-free), and it is why the copying form
// wins despite the worse schedule: keeping A[] and N[] live across EIGHT
// MFMAs instead of four extends both live ranges past what the allocator can
// absorb.
//
// THE GENERAL RESULT, which is the useful part: on this kernel, prefetch
// restructuring is register-bound, not schedule-bound. Any variant that widens
// the steady-state body pays a binary-wide VGPR bill first. It also retires the
// open question in memory glm-occupancy-ladder-is-dead-unroll-instead -- the
// MoE k-loop is NOT at the unroll=1 / 3.45 TB/s point; it holds four loads in
// flight, and buying eight costs more than it returns.
#ifndef MPK_MOE_PF_DBUF
#define MPK_MOE_PF_DBUF 0
#endif

// The shipping k-loop, hoisted verbatim out of the two call sites so the
// prefetch depth can be selected with `if constexpr` instead of `#if`. Declares
// a0..a3 and reloads each one right after the MFMA that consumes it; see the
// ISA dump above for why that is a depth-1 pipeline in practice and not the
// depth-4 it reads as. Kept because it is the only form that does not need
// GROUPS to divide the trip count.
template <bool WEIGHT_FP4, bool TOK_SC_FP8, int KI_END>
__device__ __forceinline__ f32x4_t
    _gang_moe_kloop_ship(uint8_t const *w_data_row,
                         uint8_t const *wg_scales,
                         int row_scale_base,
                         uint8_t const *s_tok_fp8,
                         uint8_t const *s_tok_scales,
                         int g) {
  constexpr int K_PER_MFMA = 128;
  // See the deep loop: the layout lives entirely in these two strides plus the
  // bases the caller passed in.
  constexpr int W_KS = _gang_moe_w_kstride<WEIGHT_FP4>();
  constexpr int SC_KS = _gang_moe_sc_kstride();
  auto tok_sc_at = [&](int k) -> int {
    return TOK_SC_FP8 ? (int)s_tok_scales[k * 4 + g] : (int)s_tok_scales[k];
  };

  f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};
  i32x8_t a0 = _gang_load_w_mfma_a_at<WEIGHT_FP4>(w_data_row, 0 * W_KS, g);
  int sa0 = (int)wg_scales[row_scale_base + 0 * SC_KS + g];
  i32x8_t a1 = _gang_load_w_mfma_a_at<WEIGHT_FP4>(w_data_row, 1 * W_KS, g);
  int sa1 = (int)wg_scales[row_scale_base + 1 * SC_KS + g];
  i32x8_t a2 = _gang_load_w_mfma_a_at<WEIGHT_FP4>(w_data_row, 2 * W_KS, g);
  int sa2 = (int)wg_scales[row_scale_base + 2 * SC_KS + g];
  i32x8_t a3 = _gang_load_w_mfma_a_at<WEIGHT_FP4>(w_data_row, 3 * W_KS, g);
  int sa3 = (int)wg_scales[row_scale_base + 3 * SC_KS + g];

// IMPORTANT: #pragma unroll 1 prevents ROCm miscompilation.
#pragma unroll 1
  for (int ki = 0; ki < KI_END; ki += 4) {
    {
      i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, ki * K_PER_MFMA, g);
      acc = _gang_mfma_w_x_f8<WEIGHT_FP4>(a0, b, acc, sa0, tok_sc_at(ki));
    }
    if (ki + 4 < KI_END) {
      int k4 = ki + 4;
      a0 = _gang_load_w_mfma_a_at<WEIGHT_FP4>(w_data_row, k4 * W_KS, g);
      sa0 = (int)wg_scales[row_scale_base + k4 * SC_KS + g];
    }

    {
      i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 1) * K_PER_MFMA, g);
      acc = _gang_mfma_w_x_f8<WEIGHT_FP4>(a1, b, acc, sa1, tok_sc_at(ki + 1));
    }
    if (ki + 5 < KI_END) {
      int k5 = ki + 5;
      a1 = _gang_load_w_mfma_a_at<WEIGHT_FP4>(w_data_row, k5 * W_KS, g);
      sa1 = (int)wg_scales[row_scale_base + k5 * SC_KS + g];
    }

    {
      i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 2) * K_PER_MFMA, g);
      acc = _gang_mfma_w_x_f8<WEIGHT_FP4>(a2, b, acc, sa2, tok_sc_at(ki + 2));
    }
    if (ki + 6 < KI_END) {
      int k6 = ki + 6;
      a2 = _gang_load_w_mfma_a_at<WEIGHT_FP4>(w_data_row, k6 * W_KS, g);
      sa2 = (int)wg_scales[row_scale_base + k6 * SC_KS + g];
    }

    if (ki + 3 < KI_END) {
      i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 3) * K_PER_MFMA, g);
      acc = _gang_mfma_w_x_f8<WEIGHT_FP4>(a3, b, acc, sa3, tok_sc_at(ki + 3));
    }
    if (ki + 7 < KI_END) {
      int k7 = ki + 7;
      a3 = _gang_load_w_mfma_a_at<WEIGHT_FP4>(w_data_row, k7 * W_KS, g);
      sa3 = (int)wg_scales[row_scale_base + k7 * SC_KS + g];
    }
  }
  return acc;
}

// Largest divisor of KI that is at most REQ and at least 2, or 0 if there is
// none. The deep loop's trip count is a compile-time constant with no
// remainder handling -- that is what keeps its steady-state body straight-line
// and its s_waitcnt partial -- so a shape whose k-loop does not divide has to
// fall back rather than fail to compile. GLM-5 is 48 (W13) and 16 (W2), both
// clean at 4; GLM-4.7-Flash's W2 runs 11 iterations and takes the fallback.
__device__ __host__ constexpr int _gang_moe_pf_groups(int ki, int req) {
  for (int gr = (req < ki ? req : ki); gr >= 2; gr--) {
    if (ki % gr == 0) {
      return gr;
    }
  }
  return 0;
}

// Prefetch-depth dispatch. GROUPS_REQ == 0, or a trip count no divisor fits,
// selects the shipping loop.
template <bool WEIGHT_FP4, bool TOK_SC_FP8, int GROUPS_REQ, int KI_END>
__device__ __forceinline__ f32x4_t
    _gang_moe_kloop(uint8_t const *w_data_row,
                    uint8_t const *wg_scales,
                    int row_scale_base,
                    uint8_t const *s_tok_fp8,
                    uint8_t const *s_tok_scales,
                    int g) {
  constexpr int GR = _gang_moe_pf_groups(KI_END, GROUPS_REQ);
  constexpr int NB = GR >= 2 ? KI_END / GR : 0;
  if constexpr (MPK_MOE_PF_DBUF && GR >= 2 && NB >= 4 && NB % 2 == 0) {
    return _gang_moe_kloop_dbuf<WEIGHT_FP4, TOK_SC_FP8, GR, KI_END>(
        w_data_row, wg_scales, row_scale_base, s_tok_fp8, s_tok_scales, g);
  } else if constexpr (GR >= 2) {
    return _gang_moe_kloop_deep<WEIGHT_FP4, TOK_SC_FP8, GR, KI_END>(
        w_data_row, wg_scales, row_scale_base, s_tok_fp8, s_tok_scales, g);
  } else {
    return _gang_moe_kloop_ship<WEIGHT_FP4, TOK_SC_FP8, KI_END>(
        w_data_row, wg_scales, row_scale_base, s_tok_fp8, s_tok_scales, g);
  }
}

// Resolve a gang tile index to (expert_id, token, workgroup).
//
// Tiles interleave round-robin across the XCDs -- global_tile = tile_idx*8 +
// xcd_id -- rather than blocking, so that all 8 XCDs stay busy when fewer than
// 8 experts are active, which at top-4-plus-shared is always.
//
// Returns false when this worker has no tile, has run past the activated
// experts, or the token does not route to this expert.
//
// ── expert parallelism ────────────────────────────────────────────────────
// Under EP each rank stores and computes only EP_NUM_ROUTED / EP_WORLD_SIZE of
// the routed experts (the id split gpt-oss calls `ep_slice`; GLM cannot use its
// slot split, because top-4 does not span 8 ranks). Routing, the mask and the
// activated list stay REPLICATED and keyed by the global expert id -- only the
// weight and bias tensors are local, so only their index is remapped, to
// *local_eid.
//
// The shared expert has no id range to fall in: it is id EP_NUM_ROUTED, every
// token routes to it, and it must be computed exactly once across the whole
// world or the cross-rank sum would count it EP_WORLD_SIZE times. One rank
// (EP_SHARED_PE) owns it; the others store a copy they never read, which is
// 36 MB of dead weight per rank at GLM-5.2 dims and not worth a ragged tensor
// shape to avoid.
//
// MEASURED 2026-08-21 -- do not retry the obvious rebalance. EP_SHARED_PE
// carries one expert MORE than its routed share in EVERY layer, and the MoE
// phase ends when the last rank does, so splitting the shared expert across
// the ranks looks free: the weights are already replicated on all of them (see
// the paragraph above), so it is a change to this decode and nothing else.
// It is WRONG OUTPUT. An expert's TILES_PER_EXPERT tiles partition its
// INTERMEDIATE dimension, and W13 writes that intermediate to a rank-LOCAL
// scratch (moe_swiglu_out_ptr) which this rank's W2 then reduces over in full.
// Give rank r only the r-th row slice and the other 7/8 of W2's reduction
// window reads stale memory. The replicated thing is the WEIGHTS; the
// intermediate ACTIVATION is not replicated, and it is the one that matters.
// Gate 1 still passed (all 8 ranks agreed -- they agree on garbage), so this
// is only caught by reading the text.
// The correct shape is a K-shard: rank r computes W13 rows [r*I/W, (r+1)*I/W)
// AND runs W2 with its reduction restricted to that same slice, letting the EP
// collective sum the partials. That is blocked here -- W2 has MFMA_ITERS = 16
// and the depth-4 pipeline needs the per-rank window to be a multiple of 4, so
// an 8-way split leaves 2. See MPK_W2_KSPLIT for the intra-rank version and
// why more, smaller W2 tiles lose anyway.
//
// The tile space is built over the OWNED subsequence of the activated list,
// not over the whole list with the non-owned tiles early-returning. gpt-oss
// measured why: owned tiles come in runs of TILES_PER_EXPERT, and a run
// against the stride-N worker map aliases into 2 real tiles on one worker
// where the tile count should never exceed 1 -- and that straggler is exactly
// what the next barrier waits out. Compacting costs two scans of the activated
// list (at most TOPK_K + 1 loads, hoisted uniformly across the workgroup) per
// tile, against a tile that is a whole GEMV.
template <int BATCH_SIZE,
          int NUM_EXPERTS,
          int TILES_PER_EXPERT,
          int WGS,
          int EP_WORLD_SIZE = 1,
          int EP_MY_PE = 0,
          // Routed experts only; the shared expert is id EP_NUM_ROUTED.
          int EP_NUM_ROUTED = NUM_EXPERTS,
          int EP_SHARED_PE = 0,
          // MPK_SHARED_DUP pricing probe: give the shared expert
          // 1 + DUP_SHARED consecutive slots in the owned subsequence, so
          // every one of its tiles runs that many times. DUP_SHARED is the
          // number of EXTRA copies: 1 doubles, 2 triples. Two magnitudes are
          // what separate a LINEAR peer-idle->wall response from a slack
          // THRESHOLD. Long note at the define in mpk_atoms.cuh. Only the W13
          // caller may set this -- W2's epilogue is an atomicAdd and would
          // double-count. Default 0 is byte-for-byte the old decode.
          int DUP_SHARED = 0>
__device__ __forceinline__ bool _gang_moe_mxfp8_tile(int tile_idx,
                                                     int const *d_mask,
                                                     int const *d_routing,
                                                     int *expert_id,
                                                     int *local_eid,
                                                     int *tok_idx,
                                                     int *wg_idx,
                                                     int *topk_slot) {
  int const num_activated_experts = d_mask[NUM_EXPERTS];
  int global_tile = tile_idx * 8 + _gang_moe_get_xcd_id();
  int e, leid;
  if constexpr (EP_WORLD_SIZE > 1) {
    static_assert(EP_NUM_ROUTED % EP_WORLD_SIZE == 0,
                  "ep_slice needs the routed expert count to divide by the "
                  "world size");
    constexpr int EP_LOCAL_ROUTED = EP_NUM_ROUTED / EP_WORLD_SIZE;
    constexpr int EP_BASE = EP_MY_PE * EP_LOCAL_ROUTED;
    int const owned_rank = global_tile / TILES_PER_EXPERT;
    int seen = -1;
    e = -1;
    for (int i = 0; i < num_activated_experts; i++) {
      int const cand = d_mask[i];
      bool const owned =
          (cand >= EP_NUM_ROUTED)
              ? (EP_MY_PE == EP_SHARED_PE)
              : (cand >= EP_BASE && cand < EP_BASE + EP_LOCAL_ROUTED);
      // At DUP_SHARED > 0 the shared expert consumes 1 + DUP_SHARED slots
      // instead of one. With mult == 1 `seen` still steps by exactly one from
      // -1, so `>=` first fires exactly where `== owned_rank` did: the
      // default path is unchanged.
      int const mult =
          (DUP_SHARED > 0 && cand >= EP_NUM_ROUTED) ? (1 + DUP_SHARED) : 1;
      if (owned) {
        seen += mult;
        if (seen >= owned_rank) {
          e = cand;
          break;
        }
      }
    }
    if (e < 0) {
      return false;
    }
    // Local weight layout: the owned routed range packed down to
    // [0, EP_LOCAL_ROUTED), then the shared expert.
    leid = (e >= EP_NUM_ROUTED) ? (EP_LOCAL_ROUTED + (e - EP_NUM_ROUTED))
                                : (e - EP_BASE);
  } else {
    if (global_tile >= num_activated_experts * TILES_PER_EXPERT) {
      return false;
    }
    e = d_mask[global_tile / TILES_PER_EXPERT];
    leid = e;
  }
  int const within = global_tile % TILES_PER_EXPERT;
  int const tok = within / WGS;
  if (tok >= BATCH_SIZE) {
    return false;
  }
  int const route_val = d_routing[e * BATCH_SIZE + tok];
  if (route_val == 0) {
    return false;
  }
  *expert_id = e;
  *local_eid = leid;
  *tok_idx = tok;
  *wg_idx = within % WGS;
  *topk_slot = route_val - 1;
  return true;
}

// Gang MoE W13 (gate+up) with MXFP8 weights.
//
// Input:  [batch, K = hidden]                          bf16
// Weight: [num_experts, WGS, WG_BYTES]                 MXFP8 packed
// Output: [batch, topk, OUTPUT_STRIDE]                 bf16, or
//         [batch, topk, OUTPUT_STRIDE/2] when FUSE_SWIGLU
//
// FUSE_SWIGLU requires the weight rows pairwise interleaved -- row 2j is
// gate_j, row 2j+1 is up_j -- because a lane owns acc[0..3] at four
// *consecutive* N, so gate and its up partner only meet in registers when they
// are adjacent columns. The packer preserves row order, so the same
// interleave the bf16 path already applies carries over untouched.
template <int BATCH_SIZE,
          int OUTPUT_SIZE,
          int OUTPUT_STRIDE,
          int REDUCTION_SIZE,
          int NUM_EXPERTS,
          int NUM_TOPK,
          int TILES_PER_EXPERT,
          int OUTPUT_PER_WG,
          bool FUSE_SWIGLU = false,
          // Set by a fused caller that runs W2 in the same task, with only an
          // in-kernel barrier between them. W2 for one expert reduces over
          // that expert's whole intermediate, whose workgroups are spread
          // across all 8 XCDs, so it reads activations this XCD produced.
          // Per-XCD L2 is not coherent on MI300/MI350: as separate tasks the
          // event boundary's buffer_wbl2 is what makes them visible, and a
          // fused caller has no such boundary. Writing through costs the
          // store's L2 residency, which nothing on this XCD wants back.
          bool WRITE_THROUGH = false,
          // E2M1 nibbles instead of E4M3 bytes on the weight side only; see
          // the note at the top of the file.
          bool WEIGHT_FP4 = false,
          // Expert parallelism; see the note on _gang_moe_mxfp8_tile. 1/0
          // compiles the remap out and local_eid == expert_id.
          int EP_WORLD_SIZE = 1,
          int EP_MY_PE = 0,
          int EP_NUM_ROUTED = NUM_EXPERTS,
          int EP_SHARED_PE = 0,
          // ── Emit the SwiGLU result as MXFP8 instead of bf16 ──────────────
          //
          // The consumer of this activation is W2, and W2's first act is to
          // quantize it: every one of the ~96 tiles per expert re-derives the
          // same E4M3 bytes and the same E8M0 scales from the same 2048
          // element vector. That is pure redundancy -- 4 KB of activation in
          // front of a 68 KB weight tile, so no byte-side lever touches it --
          // and it is redundant work this kernel is uniquely placed to
          // delete, because the SwiGLU result is already in its registers in
          // f32 and each of its workgroups owns a contiguous, 32-aligned
          // slice of the intermediate.
          //
          // Deleting it costs one cross-wave amax. A workgroup's slice is
          // OUTPUT_PER_WG/2 activation columns and the emitting lanes are
          // spread over all four waves (col == 0, four g each), so the two
          // f32 results per lane go through LDS and a short pass at the end
          // of the tile packs them. See the write-out block after the MFMA
          // branches.
          //
          // Requires the N-parallel branch: the K-parallel workgroup owns 8
          // activation columns, which is narrower than a scale block.
          bool EMIT_FP8 = false>
__device__ __noinline__ void
    gang_moe_w13_linear_mxfp8_kernel(void const *input_ptr,
                                     void const *weight_ptr,
                                     void const *routing_ptr,
                                     void const *mask_ptr,
                                     void const *bias_ptr,
                                     void *output_ptr,
                                     int tile_idx) {
  // CLOSED 2026-08-21: this assert is why W13 tile narrowing has no middle
  // point left. GLM_MOE_W13_OPW already defaults to 64, so 16 -- the measured
  // -1.34 ms (95a044a) -- was the ONLY legal value below the default. 32 is
  // rejected here and would starve the 16-row MFMA anyway (8 rows over 4
  // waves), which is the same reason 16 lost. Do not reopen looking for a
  // middle point; there is none.
  static_assert(OUTPUT_PER_WG % 64 == 0 || OUTPUT_PER_WG == 16,
                "OUTPUT_PER_WG is either N-parallel (a multiple of 64 = 4 "
                "waves x 16 rows) or the K-parallel width, 16");
  static_assert(REDUCTION_SIZE % 128 == 0,
                "K must be a multiple of 128 for FP8 MFMA");

  constexpr int NUM_BLOCKS_32 = REDUCTION_SIZE / 32;
  constexpr int W_ROW_BYTES = WEIGHT_FP4 ? REDUCTION_SIZE / 2 : REDUCTION_SIZE;
  // Bytes one row contributes to one k-tile; W_ROW_BYTES is MFMA_ITERS of it.
  constexpr int W_BPK = _gang_moe_bytes_per_ktile<WEIGHT_FP4>();
  static_assert(MPK_MOE_KMAJOR == 0 || OUTPUT_PER_WG % 16 == 0,
                "K-major groups the rows a wave covers 16 at a time");
  constexpr int WG_DATA_BYTES = OUTPUT_PER_WG * W_ROW_BYTES;
  constexpr int WG_SCALE_BYTES = OUTPUT_PER_WG * NUM_BLOCKS_32;
  constexpr int WG_BYTES = WG_DATA_BYTES + WG_SCALE_BYTES;
  constexpr int EXPERT_WGS = OUTPUT_STRIDE / OUTPUT_PER_WG;
  constexpr int64_t EXPERT_BYTES = static_cast<int64_t>(EXPERT_WGS) * WG_BYTES;

  constexpr int K_PER_MFMA = 128;
  constexpr int MFMA_ITERS = REDUCTION_SIZE / K_PER_MFMA;
  // Same depth-4 pipeline as the dense MXFP8 kernel, and the same constraint:
  // only slot 3 carries a tail guard, so a partial final group would let slots
  // 1 and 2 compute k-tiles that do not exist.
  static_assert(MFMA_ITERS >= 4 && MFMA_ITERS % 4 == 0,
                "Depth-4 pipeline requires REDUCTION_SIZE % 512 == 0");

  constexpr int NUM_WAVES = 4;
  // N-parallel splits the workgroup's rows across the waves; K-parallel gives
  // every wave the same 16 rows and splits the reduction, so it has exactly
  // one tile. See the branch below.
  constexpr bool K_PARALLEL = (OUTPUT_PER_WG < 64);
  constexpr int TILES_PER_WAVE =
      K_PARALLEL ? 1 : (OUTPUT_PER_WG / 16 / NUM_WAVES);
  constexpr int FP8_TOK_DATA = REDUCTION_SIZE;

  unsigned short const *A = (unsigned short const *)input_ptr;
  uint8_t const *W = (uint8_t const *)weight_ptr;
  unsigned short const *d_bias = (unsigned short const *)bias_ptr;
  unsigned short *d_output = (unsigned short *)output_ptr;

  int expert_id, local_eid, tok_idx, wg_idx, topk_slot;
  if (!_gang_moe_mxfp8_tile<BATCH_SIZE,
                            NUM_EXPERTS,
                            TILES_PER_EXPERT,
                            EXPERT_WGS,
                            EP_WORLD_SIZE,
                            EP_MY_PE,
                            EP_NUM_ROUTED,
                            EP_SHARED_PE,
                            // W13 ONLY. Its epilogue is a plain (or
                            // write-through) store of a deterministic value,
                            // so a duplicated tile rewrites the same bits and
                            // the output is unchanged. Never set this on W2.
                            /*DUP_SHARED=*/(MPK_SHARED_DUP)>(
                                          tile_idx,
                                          (int const *)mask_ptr,
                                          (int const *)routing_ptr,
                                          &expert_id,
                                          &local_eid,
                                          &tok_idx,
                                          &wg_idx,
                                          &topk_slot)) {
    return;
  }

  uint8_t const *wg_data = W + static_cast<int64_t>(local_eid) * EXPERT_BYTES +
                           static_cast<int64_t>(wg_idx) * WG_BYTES;
  uint8_t const *wg_scales = wg_data + WG_DATA_BYTES;

  extern __shared__ char _gang_moe_mxfp8_smem[];
  uint8_t *s_tok_fp8 = (uint8_t *)_gang_moe_mxfp8_smem;
  uint8_t *s_tok_scales = s_tok_fp8 + FP8_TOK_DATA;

  // K-parallel reduces four waves' partial sums through LDS. Placed AFTER the
  // token scratch rather than aliased over it, which is what the dense MXFP8
  // kernel does: a wave that reaches the reduction while wave 0 is still
  // consuming the low end of s_tok_fp8 would otherwise overwrite k-tiles wave
  // 0 has not read. 256 bytes against a 6.3 KB allocation is not worth the
  // race.
  float *s_reduce =
      (float *)(((uintptr_t)(s_tok_scales + NUM_BLOCKS_32) + 15u) &
                ~(uintptr_t)15);

  // EMIT_FP8 scratch, past s_reduce so it does not alias the K-parallel
  // reduction (which it never coexists with, but the offsets are constexpr
  // either way). ACT_COLS floats of SwiGLU result plus one f32 scale and one
  // E8M0 byte per 32-column block: 160 B at OUTPUT_PER_WG = 64.
  constexpr int W13_ACT_COLS = EMIT_FP8 ? OUTPUT_PER_WG / 2 : 0;
  constexpr int W13_ACT_BLKS = EMIT_FP8 ? W13_ACT_COLS / 32 : 0;
  float *s_act = s_reduce + (K_PARALLEL ? NUM_WAVES * OUTPUT_PER_WG : 0);
  float *s_act_scale_f = s_act + W13_ACT_COLS;
  if constexpr (EMIT_FP8) {
    static_assert(!K_PARALLEL,
                  "EMIT_FP8 needs the N-parallel branch: a K-parallel "
                  "workgroup owns 8 activation columns, under one scale "
                  "block of 32");
    static_assert(FUSE_SWIGLU,
                  "EMIT_FP8 quantizes the SwiGLU result; there is no other "
                  "activation for it to emit");
    static_assert(OUTPUT_SIZE == OUTPUT_STRIDE,
                  "EMIT_FP8 packs a dense byte row; a short OUTPUT_SIZE would "
                  "leave holes the consumer still reads");
    static_assert((OUTPUT_PER_WG / 2) % 32 == 0,
                  "a workgroup must own a whole number of 32-element scale "
                  "blocks");
  }

  int const tid = threadIdx.x;
  int const warp_id = tid >> 6;
  int const lane_id = tid & 63;
  int const col = lane_id & 15; // weight row within the 16x16 MFMA tile
  int const g = lane_id >> 4;   // K-group (0..3)

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  // Slot 1 is otherwise unused. [0]/[1] split this tile into the token-quant
  // prologue and everything after it, [6] counts tiles, so ns/count is a
  // per-tile microsecond figure without needing a worker/layer divisor.
  unsigned long long _sp_q0 = __builtin_amdgcn_s_memrealtime();
#endif
  // Phase 1: quantize this token's activation to FP8 E4M3 in LDS.
  _gang_wave_parallel_fp8_quant<REDUCTION_SIZE>(
      A + static_cast<size_t>(tok_idx) * REDUCTION_SIZE,
      s_tok_fp8,
      s_tok_scales);
#ifdef MPK_ENABLE_SUBPHASE_TIMING
  unsigned long long _sp_q1 = __builtin_amdgcn_s_memrealtime();
  if (tid == 0 && g_subphase_active) {
    atomicAdd(&g_subphase_ns[1][0], (_sp_q1 - _sp_q0) * 10);
    atomicAdd(&g_subphase_ns[1][6], 1ULL);
    atomicAdd(&g_subphase_cnt[1], 1ULL);
  }
#endif

  // Phase 2 epilogue, hoisted out of both parallelization branches. `acc`
  // holds one lane's four output columns, starting at out_base; at one token
  // per tile only col 0 carries a result, so every caller guards on that.
  auto emit = [&](f32x4_t const &acc, int const out_base) {
    unsigned short const *bias_row = d_bias + local_eid * OUTPUT_STRIDE;

    if constexpr (FUSE_SWIGLU) {
      constexpr int ACT_STRIDE = OUTPUT_STRIDE / 2;
      static_assert(2 * ACT_STRIDE == OUTPUT_STRIDE,
                    "FUSE_SWIGLU expects a half-width activation output");
      // out_base is a multiple of 4, so the four accumulators are always
      // two whole gate/up pairs starting on a pair boundary.
      unsigned short *act_addr = d_output +
                                 tok_idx * (NUM_TOPK * ACT_STRIDE) +
                                 topk_slot * ACT_STRIDE + (out_base >> 1);
      unsigned short act[2];
      bool act_ok[2];
      if constexpr (EMIT_FP8) {
        // Park the f32 result; the block amax is not known until every wave
        // of this workgroup has emitted. `out_base >> 1` is this lane's first
        // activation column and wg_idx * ACT_COLS is the workgroup's base, so
        // the difference is the LDS slot directly.
        int const act_local = (out_base >> 1) - wg_idx * W13_ACT_COLS;
#pragma unroll
        for (int p = 0; p < 2; p++) {
          int const out_n = out_base + 2 * p;
          float const gate = acc[2 * p] + _gang_bf16_to_float(bias_row[out_n]);
          float const up =
              acc[2 * p + 1] + _gang_bf16_to_float(bias_row[out_n + 1]);
          s_act[act_local + p] = fast_silu(gate) * up;
        }
        return;
      }
#pragma unroll
      for (int p = 0; p < 2; p++) {
        int const out_n = out_base + 2 * p;
        act_ok[p] = (out_n + 1 < OUTPUT_SIZE);
        if (act_ok[p]) {
          float const gate = acc[2 * p] + _gang_bf16_to_float(bias_row[out_n]);
          float const up =
              acc[2 * p + 1] + _gang_bf16_to_float(bias_row[out_n + 1]);
          act[p] = _gang_float_to_bf16(fast_silu(gate) * up);
        }
      }
      if constexpr (WRITE_THROUGH) {
        // The two activations are adjacent columns of the same row, and
        // out_base is a multiple of 4, so act_addr is 4-byte aligned and the
        // pair is one dword -- half as many write-through stores as doing
        // them separately.
        if (act_ok[0] && act_ok[1]) {
          unsigned packed = (unsigned)act[0] | ((unsigned)act[1] << 16);
          st_wt_u32((void *)act_addr, packed);
        } else {
#pragma unroll
          for (int p = 0; p < 2; p++) {
            if (act_ok[p]) {
              st_wt_u16((void *)&act_addr[p], act[p]);
            }
          }
        }
      } else {
#pragma unroll
        for (int p = 0; p < 2; p++) {
          if (act_ok[p]) {
            act_addr[p] = act[p];
          }
        }
      }
    } else {
      unsigned short *out_addr = d_output +
                                 tok_idx * (NUM_TOPK * OUTPUT_STRIDE) +
                                 topk_slot * OUTPUT_STRIDE + out_base;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        int const out_n = out_base + i;
        if (out_n < OUTPUT_SIZE) {
          out_addr[i] = _gang_float_to_bf16(
              acc[i] + _gang_bf16_to_float(bias_row[out_n]));
        }
      }
    }
  };

  // Phase 2: depth-4 pipelined FP8(weight) x FP8(token) MFMA.
  //
  // Two parallelizations, chosen by OUTPUT_PER_WG. N-parallel is the wide
  // form: the workgroup's rows are dealt out across the four waves. It wants
  // OUTPUT_PER_WG >= 64 and gives the largest tile, which is right when there
  // are more tiles than workers.
  //
  // Under expert parallelism there are not. _gang_moe_mxfp8_tile builds the
  // tile space over the OWNED activated experts, and at EP=8 a rank owns ~2
  // activated experts (one routed of eight, plus the shared expert on
  // EP_SHARED_PE), so a GLM-5 layer has 127 live W13 tiles at
  // OUTPUT_PER_WG=64 -- 15.9 per XCD against 29 workers. K-parallel is the
  // narrow form that was supposed to fix it: all four waves take the same 16
  // rows and split the reduction, so the tile count goes up 4x and each tile
  // was expected to cost a quarter as much. Same trade the dense MXFP8 kernel
  // makes for qkv_a at QKV_MXFP8_OPW=16, and #19 for o_proj.
  //
  // MEASURED NEGATIVE, keep the branch but do not default to it. GLM-5,
  // NP=8 EP, GLM_MOE_W13_OPW=16 vs 64: 13.539 vs 12.196 ms/iter, and
  // SP3[4] MoeW13 2.008 vs 0.923 ms/iter. Per-tile subphase slot 1 says why.
  // The N-parallel OPW=64 tile is 0.86 us of token quant + 13.9 us of body
  // for 208 KB of weight, i.e. 15.0 GB/s against the 20.2 GB/s per-CU share
  // of HBM -- already 74% of roof. The K-parallel tile moves a quarter of
  // those bytes and still costs 12.2 us, 4.0 GB/s. A 12-iteration k-loop
  // cannot keep enough loads in flight to reach the roof, so the split buys
  // 4x the tiles at 4x the cost per byte and the makespan gets worse, not
  // better. The prologue is NOT the fixed cost that would have made
  // subdivision pay: at 0.86 us it is 6% of the tile.
  if constexpr (!K_PARALLEL) {
  for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; tile_iter++) {
    int const wave_tile = warp_id + tile_iter * NUM_WAVES;
    int const w_row = wave_tile * 16 + col;
    // Row base under whichever weight layout was packed. Row-major is
    // [row][k]; K-major is [row/16][k][row%16], so the row's k-tile 0 sits at
    // its 16-row block, then its lane slot inside that block's first k-tile.
    // See MPK_MOE_KMAJOR.
    uint8_t const *w_data_row =
        wg_data + (MPK_MOE_KMAJOR >= 1
                       ? static_cast<size_t>(w_row / 16) * (16 * W_ROW_BYTES) +
                             static_cast<size_t>(w_row % 16) * W_BPK
                       : static_cast<size_t>(w_row) * W_ROW_BYTES);
    int const row_scale_base =
        (MPK_MOE_KMAJOR >= 2
             ? (w_row / 16) * (16 * NUM_BLOCKS_32) + (w_row % 16) * 4
             : w_row * NUM_BLOCKS_32);

    // MFMA_ITERS is 48 for GLM-5's 6144 hidden, so the default depth of 4 is
    // taken here.
    f32x4_t acc = _gang_moe_kloop<WEIGHT_FP4, false, MPK_MOE_PF_GROUPS,
                                  MFMA_ITERS>(
        w_data_row, wg_scales, row_scale_base, s_tok_fp8, s_tok_scales, g);

    // Epilogue. acc[i] = C[g*4+i][col]; at one token per tile only col 0 holds
    // a result.
    if (col == 0) {
      emit(acc, wg_idx * OUTPUT_PER_WG + wave_tile * 16 + g * 4);
    }
  }
  } else {
    // K-parallel: all four waves take the same 16 output rows and split the
    // reduction between them, then reduce through LDS.
    static_assert(OUTPUT_PER_WG == 16,
                  "the K-parallel branch covers 16 output rows per workgroup");
    // This branch keeps the pre-MPK_MOE_KMAJOR row-major addressing. It is
    // measured negative and off by default, so it is guarded rather than
    // ported. The `|| !K_PARALLEL` keeps the condition dependent -- a
    // non-dependent static_assert fires even in a discarded if-constexpr arm.
    static_assert(MPK_MOE_KMAJOR == 0 || !K_PARALLEL,
                  "K-major weight is not wired into the K-parallel branch");
    constexpr int ITERS_PER_WAVE = MFMA_ITERS / NUM_WAVES;
    static_assert(MFMA_ITERS % NUM_WAVES == 0,
                  "MFMA_ITERS must be divisible by NUM_WAVES for K-parallel");
    // Same reason the N-parallel loop needs MFMA_ITERS % 4 == 0: only slot 3
    // carries a tail guard, so a partial final group would let slots 1 and 2
    // compute k-tiles outside this wave's range.
    static_assert(ITERS_PER_WAVE >= 4 && ITERS_PER_WAVE % 4 == 0,
                  "Depth-4 K-parallel requires ITERS_PER_WAVE a multiple of 4");

    int const ki_start = warp_id * ITERS_PER_WAVE;
    int const ki_end = ki_start + ITERS_PER_WAVE;
    int const w_row = col; // all four waves, same 16 rows
    uint8_t const *w_data_row =
        wg_data + static_cast<size_t>(w_row) * W_ROW_BYTES;
    int const row_scale_base = w_row * NUM_BLOCKS_32;

    f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

    i32x8_t a0 =
        _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, ki_start * K_PER_MFMA, g);
    int sa0 = (int)wg_scales[row_scale_base + ki_start * 4 + g];
    i32x8_t a1 = _gang_load_w_mfma_a<WEIGHT_FP4>(
        w_data_row, (ki_start + 1) * K_PER_MFMA, g);
    int sa1 = (int)wg_scales[row_scale_base + (ki_start + 1) * 4 + g];
    i32x8_t a2 = _gang_load_w_mfma_a<WEIGHT_FP4>(
        w_data_row, (ki_start + 2) * K_PER_MFMA, g);
    int sa2 = (int)wg_scales[row_scale_base + (ki_start + 2) * 4 + g];
    i32x8_t a3 = _gang_load_w_mfma_a<WEIGHT_FP4>(
        w_data_row, (ki_start + 3) * K_PER_MFMA, g);
    int sa3 = (int)wg_scales[row_scale_base + (ki_start + 3) * 4 + g];

// IMPORTANT: #pragma unroll 1 prevents ROCm miscompilation.
#pragma unroll 1
    for (int ki = ki_start; ki < ki_end; ki += 4) {
      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, ki * K_PER_MFMA, g);
        acc = _gang_mfma_w_x_f8<WEIGHT_FP4>(a0, b, acc, sa0,
                                            (int)s_tok_scales[ki]);
      }
      if (ki + 4 < ki_end) {
        int kt4 = (ki + 4) * K_PER_MFMA;
        a0 = _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, kt4, g);
        sa0 = (int)wg_scales[row_scale_base + kt4 / 32 + g];
      }

      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 1) * K_PER_MFMA, g);
        acc = _gang_mfma_w_x_f8<WEIGHT_FP4>(a1, b, acc, sa1,
                                            (int)s_tok_scales[ki + 1]);
      }
      if (ki + 5 < ki_end) {
        int kt5 = (ki + 5) * K_PER_MFMA;
        a1 = _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, kt5, g);
        sa1 = (int)wg_scales[row_scale_base + kt5 / 32 + g];
      }

      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 2) * K_PER_MFMA, g);
        acc = _gang_mfma_w_x_f8<WEIGHT_FP4>(a2, b, acc, sa2,
                                            (int)s_tok_scales[ki + 2]);
      }
      if (ki + 6 < ki_end) {
        int kt6 = (ki + 6) * K_PER_MFMA;
        a2 = _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, kt6, g);
        sa2 = (int)wg_scales[row_scale_base + kt6 / 32 + g];
      }

      if (ki + 3 < ki_end) {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 3) * K_PER_MFMA, g);
        acc = _gang_mfma_w_x_f8<WEIGHT_FP4>(a3, b, acc, sa3,
                                            (int)s_tok_scales[ki + 3]);
      }
      if (ki + 7 < ki_end) {
        int kt7 = (ki + 7) * K_PER_MFMA;
        a3 = _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, kt7, g);
        sa3 = (int)wg_scales[row_scale_base + kt7 / 32 + g];
      }
    }

    if (col == 0) {
#pragma unroll
      for (int i = 0; i < 4; i++) {
        s_reduce[warp_id * OUTPUT_PER_WG + g * 4 + i] = acc[i];
      }
    }
    __syncthreads();

    if (warp_id == 0 && col == 0) {
      f32x4_t sum = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int w = 0; w < NUM_WAVES; w++) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
          sum[i] += s_reduce[w * OUTPUT_PER_WG + g * 4 + i];
        }
      }
      emit(sum, wg_idx * OUTPUT_PER_WG + g * 4);
    }
  }

  __syncthreads();

  // ── EMIT_FP8 write-out ────────────────────────────────────────────────────
  //
  // Every wave has now parked its SwiGLU results in s_act, so the workgroup's
  // W13_ACT_COLS contiguous activation columns are complete and any thread can
  // see all of them. Two short passes: one thread per 32-column block computes
  // the block amax and its E8M0 scale, then one thread per four columns packs
  // a dword of E4M3 and stores it.
  //
  // The FP8 row and its scales ride in the SAME buffer the bf16 activation
  // used. A (token, slot) pair owns 2 * ACT_STRIDE bytes there; MXFP8 needs
  // ACT_STRIDE + ACT_STRIDE/32, which is 2112 of 4096 for GLM-5. Reusing the
  // slab keeps this change inside the two kernels -- no new tensor, no task
  // registration, no host plumbing.
  //
  // The stores are write-through for exactly the reason the bf16 path is: the
  // consumer is a W2 tile on another XCD and there is no event boundary
  // between them to write L2 back.
  if constexpr (EMIT_FP8) {
    constexpr int ACT_STRIDE = OUTPUT_STRIDE / 2;
    constexpr int W13_ACT_PACKS = W13_ACT_COLS / 4;
    uint8_t *slot_base =
        (uint8_t *)(d_output +
                    static_cast<size_t>(tok_idx) * (NUM_TOPK * ACT_STRIDE) +
                    static_cast<size_t>(topk_slot) * ACT_STRIDE);
    uint8_t *g_act_fp8 = slot_base + wg_idx * W13_ACT_COLS;
    uint8_t *g_act_scale = slot_base + ACT_STRIDE + wg_idx * W13_ACT_BLKS;

    if (tid < W13_ACT_BLKS) {
      float amax = 0.0f;
#pragma unroll
      for (int j = 0; j < 32; j++) {
        amax = fmaxf(amax, fabsf(s_act[tid * 32 + j]));
      }
      uint8_t const se = _gang_compute_e8m0_fp8(amax);
      if (se == 0) {
        s_act_scale_f[tid] = 1.0f;
      } else {
        union {
          float f;
          uint32_t u;
        } sv;
        sv.u = (uint32_t)se << 23;
        s_act_scale_f[tid] = sv.f;
      }
      st_wt_u8((void *)&g_act_scale[tid], se);
    }
    __syncthreads();

    if (tid < W13_ACT_PACKS) {
      int const c0 = tid * 4;
      float const sf = s_act_scale_f[c0 >> 5];
      fp8x4_t pk = _gang_quant_4xfp8(s_act[c0], s_act[c0 + 1], s_act[c0 + 2],
                                     s_act[c0 + 3], sf);
      // wg_idx * W13_ACT_COLS is 32-aligned and c0 is 4-aligned, so the dword
      // store is aligned whatever the workgroup index.
      st_wt_u32((void *)&g_act_fp8[c0], *(unsigned const *)&pk);
    }
  }
#ifdef MPK_ENABLE_SUBPHASE_TIMING
  if (tid == 0 && g_subphase_active) {
    atomicAdd(&g_subphase_ns[1][1],
              (__builtin_amdgcn_s_memrealtime() - _sp_q1) * 10);
  }
#endif
}

// Gang MoE W2 (down projection) with MXFP8 weights.
//
// Input:  [batch, topk, K = intermediate]              bf16
// Weight: [num_experts, WGS, WG_BYTES]                 MXFP8 packed
// Output: [batch, topk, OUTPUT_STRIDE] bf16, or the [batch, hidden] f32
//         workspace when FUSE_MULSUMADD
//
// FUSE_MULSUMADD folds the topk weighting and the cross-expert sum into this
// epilogue: scale by routing_weight[tok, slot] and f32-atomicAdd into a
// workspace, instead of writing a [batch, topk, hidden] slab that a later task
// re-reads and reduces. The accumulation stays fp32 all the way to the residual
// add. Atomics make the summation order non-deterministic, which is the trade
// gpt-oss already takes.
//
// The activation this reads is produced by another XCD's W13 tile, so it is a
// cross-XCD read and uses the NT-load quantizer to avoid polluting L2 with a
// line nobody will read again.
template <int BATCH_SIZE,
          int OUTPUT_SIZE,
          int OUTPUT_STRIDE,
          int REDUCTION_SIZE,
          int NUM_EXPERTS,
          int NUM_TOPK,
          int TILES_PER_EXPERT,
          int OUTPUT_PER_WG,
          bool FUSE_MULSUMADD = false,
          bool WEIGHT_FP4 = false,
          // Expert parallelism; see the note on _gang_moe_mxfp8_tile. 1/0
          // compiles the remap out and local_eid == expert_id.
          int EP_WORLD_SIZE = 1,
          int EP_MY_PE = 0,
          int EP_NUM_ROUTED = NUM_EXPERTS,
          int EP_SHARED_PE = 0,
          // Reduction split; see the K_SPLITS note in the body. 1 is the
          // identity and every other instantiation in the megakernel keeps it.
          int W2_K_SPLITS = 1,
          // The activation is already MXFP8, laid out by the W13 kernel's
          // EMIT_FP8 epilogue: E4M3 bytes in the low REDUCTION_SIZE bytes of
          // the (token, slot) slab and one E8M0 per 32 elements right after
          // them. The prologue becomes a copy and the token scale gains the
          // per-32 `* 4 + g` selector the weight scale already has.
          bool INPUT_FP8 = false>
__device__ __noinline__ void
    gang_moe_w2_linear_mxfp8_kernel(void const *input_ptr,
                                    void const *weight_ptr,
                                    void const *routing_ptr,
                                    void const *mask_ptr,
                                    void const *bias_ptr,
                                    void *output_ptr,
                                    int tile_idx,
                                    void const *routing_weight_ptr = nullptr) {
  static_assert(OUTPUT_PER_WG % 64 == 0 || OUTPUT_PER_WG == 16,
                "OUTPUT_PER_WG is either N-parallel (a multiple of 64 = 4 "
                "waves x 16 rows) or the K-parallel width, 16");
  static_assert(REDUCTION_SIZE % 128 == 0,
                "K must be a multiple of 128 for FP8 MFMA");

  constexpr int NUM_BLOCKS_32 = REDUCTION_SIZE / 32;
  constexpr int W_ROW_BYTES = WEIGHT_FP4 ? REDUCTION_SIZE / 2 : REDUCTION_SIZE;
  // Bytes one row contributes to one k-tile; W_ROW_BYTES is MFMA_ITERS of it.
  constexpr int W_BPK = _gang_moe_bytes_per_ktile<WEIGHT_FP4>();
  static_assert(MPK_MOE_KMAJOR == 0 || OUTPUT_PER_WG % 16 == 0,
                "K-major groups the rows a wave covers 16 at a time");
  constexpr int WG_DATA_BYTES = OUTPUT_PER_WG * W_ROW_BYTES;
  constexpr int WG_SCALE_BYTES = OUTPUT_PER_WG * NUM_BLOCKS_32;
  constexpr int WG_BYTES = WG_DATA_BYTES + WG_SCALE_BYTES;
  constexpr int EXPERT_WGS = OUTPUT_STRIDE / OUTPUT_PER_WG;
  constexpr int64_t EXPERT_BYTES = static_cast<int64_t>(EXPERT_WGS) * WG_BYTES;

  constexpr int K_PER_MFMA = 128;
  constexpr int MFMA_ITERS = REDUCTION_SIZE / K_PER_MFMA;
  static_assert(MFMA_ITERS >= 4 && MFMA_ITERS % 4 == 0,
                "Depth-4 pipeline requires REDUCTION_SIZE % 512 == 0");

  // ── W2 split-K ───────────────────────────────────────────────────────────
  // Under EP, GLM-5 activates ~1 routed expert per rank, so W2's real tile
  // count is EXPERT_WGS = HIDDEN/OPW = 6144/64 = 96 globally, i.e. 12 per XCD
  // against 29 workers -- 41% of the machine, and the other 17 workers per XCD
  // spin in the barrier that follows. Narrowing the tile in N is the wrong fix
  // and is already measured out (OPW=16 is the K_PARALLEL branch below and
  // costs 4x per byte, -1.34 ms). Splitting the *reduction* instead keeps every
  // tile at the full 64-row N-parallel shape -- same MFMA, same bytes per row --
  // and just gives each tile 1/K_SPLITS of the K range.
  //
  // No combine step is needed: FUSE_MULSUMADD's epilogue already atomicAdds
  // into the f32 workspace that the cross-expert sum uses, and that is a linear
  // reduction, so a partial K sum is just another addend. Only the bias has to
  // be gated to split 0 or it lands K_SPLITS times.
  //
  // The split is in whole depth-4 pipeline groups, so K_SPLITS must divide
  // MFMA_ITERS/4. GLM-5's W2 has K = MOE_INTERMEDIATE = 2048, K_PER_MFMA = 128,
  // so MFMA_ITERS = 16 and only 2 and 4 are legal above 1. (3 is not: the
  // instantiation is <...,2048,...> and 16 % 12 = 4, which is what the assert
  // below reports if you try it.)
  // MEASURED A LARGE NEGATIVE, off by default (2026-08-21, n=3 each, control
  // in the same batch): K_SPLITS=1 11.020 ms/iter mean (10.932/11.185/10.943)
  // against K_SPLITS=2 15.136 (14.970/15.372/15.066). **+4.12 ms.**
  //
  // The premise was that W2's 12 tiles/XCD leave 17 of 29 workers idle, so
  // splitting K would fill them. It does, and it still loses, because W2's
  // per-tile cost is dominated by FIXED work that the split does not divide:
  // every tile stages and quantizes the whole FP8_TOK_DATA = REDUCTION_SIZE
  // activation into LDS regardless of which K window it will consume, and every
  // tile runs the full 64-row atomicAdd epilogue. Doubling the tile count
  // doubles both while halving only the MFMA loop, which was never the
  // bottleneck. The host loop bound doubles too, so every worker also pays
  // ceil(216/29) = 8 tile-decode iterations per layer instead of 4.
  //
  // The inversion is the useful part: at constant total FLOPs, W2 gets sharply
  // worse with MORE tiles. Both this and OPW=16 (-1.34 ms) are the same
  // finding from two directions. If W2 is ever revisited, go the other way --
  // WIDER tiles via GLM_MOE_W2_OPW, amortising the staging and the epilogue
  // over more output rows.
  //
  // Kept behind the flag at the identity rather than deleted: the control above
  // shows K_SPLITS=1 is byte-for-byte the old path, and this is the only worked
  // example in the tree of splitting a reduction across the EP tile space.
  //
  // A template parameter and not the macro directly: this kernel is
  // instantiated at several reduction sizes in one translation unit, and only
  // the GLM fused-layer call site has a host loop bound that was widened to
  // match. Reading the macro here made the legality assert fire on every other
  // instantiation.
  constexpr int K_SPLITS = W2_K_SPLITS;
  static_assert(K_SPLITS >= 1, "W2_K_SPLITS is at least 1");
  static_assert(MFMA_ITERS % (4 * K_SPLITS) == 0,
                "split-K must divide the reduction into whole depth-4 groups");
  static_assert(K_SPLITS == 1 || FUSE_MULSUMADD,
                "split-K needs the atomicAdd epilogue; the bf16 store path "
                "overwrites rather than accumulates");
  constexpr int SPLIT_ITERS = MFMA_ITERS / K_SPLITS;

  constexpr int NUM_WAVES = 4;
  // See the note on the W13 kernel's branch: K_PARALLEL is the narrow-tile
  // form that keeps all 29 workers per XCD fed when expert parallelism has
  // left the layer with only one owned expert's worth of tiles.
  constexpr bool K_PARALLEL = (OUTPUT_PER_WG < 64);
  constexpr int TILES_PER_WAVE =
      K_PARALLEL ? 1 : (OUTPUT_PER_WG / 16 / NUM_WAVES);
  constexpr int FP8_TOK_DATA = REDUCTION_SIZE;

  unsigned short const *A = (unsigned short const *)input_ptr;
  uint8_t const *W = (uint8_t const *)weight_ptr;
  unsigned short const *d_bias = (unsigned short const *)bias_ptr;
  unsigned short *d_output = (unsigned short *)output_ptr;
  float *d_workspace = (float *)output_ptr;
  float const *d_routing_weight = (float const *)routing_weight_ptr;

  int expert_id, local_eid, tok_idx, wg_idx, topk_slot;
  // The split index rides in the workgroup index: the per-expert tile space is
  // EXPERT_WGS * K_SPLITS wide and the decode is otherwise untouched, so EP's
  // owned-subsequence compaction still sees a dense run per expert. The split
  // is the HIGH digit so that consecutive tiles -- which land on consecutive
  // workers -- cover different output rows, keeping the weight stream spread
  // rather than three neighbours hammering the same 64 rows.
  if (!_gang_moe_mxfp8_tile<BATCH_SIZE,
                            NUM_EXPERTS,
                            TILES_PER_EXPERT * K_SPLITS,
                            EXPERT_WGS * K_SPLITS,
                            EP_WORLD_SIZE,
                            EP_MY_PE,
                            EP_NUM_ROUTED,
                            EP_SHARED_PE>(tile_idx,
                                          (int const *)mask_ptr,
                                          (int const *)routing_ptr,
                                          &expert_id,
                                          &local_eid,
                                          &tok_idx,
                                          &wg_idx,
                                          &topk_slot)) {
    return;
  }
  int const k_split = (K_SPLITS == 1) ? 0 : (wg_idx / EXPERT_WGS);
  if constexpr (K_SPLITS > 1) {
    wg_idx = wg_idx % EXPERT_WGS;
  }
  // K offset of this split, in reduction elements. SPLIT_ITERS * K_PER_MFMA is
  // a multiple of 512, so every base derived from it stays 128 B-aligned on
  // both the FP8 and the FP4 packing.
  int const k_base = k_split * SPLIT_ITERS * K_PER_MFMA;

  uint8_t const *wg_data = W + static_cast<int64_t>(local_eid) * EXPERT_BYTES +
                           static_cast<int64_t>(wg_idx) * WG_BYTES;
  uint8_t const *wg_scales = wg_data + WG_DATA_BYTES;

  extern __shared__ char _gang_moe_mxfp8_smem[];
  uint8_t *s_tok_fp8 = (uint8_t *)_gang_moe_mxfp8_smem;
  uint8_t *s_tok_scales = s_tok_fp8 + FP8_TOK_DATA;

  // See the W13 kernel: after the token scratch, not aliased over it.
  float *s_reduce =
      (float *)(((uintptr_t)(s_tok_scales + NUM_BLOCKS_32) + 15u) &
                ~(uintptr_t)15);

  int const tid = threadIdx.x;
  int const warp_id = tid >> 6;
  int const lane_id = tid & 63;
  int const col = lane_id & 15;
  int const g = lane_id >> 4;

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  unsigned long long _sp_q0 = __builtin_amdgcn_s_memrealtime();
#endif
  // Stage only THIS split's K window, not the whole reduction. The original
  // staged all REDUCTION_SIZE bytes whatever k_base was, which is what made
  // K_SPLITS=2 cost +4.12 ms: the split divided the MFMA loop but doubled the
  // staging. At K_SPLITS=1, SPLIT_LEN == REDUCTION_SIZE and k_base == 0, so
  // this is byte-for-byte the old call and the default path is untouched.
  // The LDS allocation stays full-width and the window is written at its own
  // offset, so the MFMA loop's s_tok_k = s_tok_fp8 + k_base is unchanged; the
  // bytes outside the window are never read.
  // MPK_W2_STAGE_FULL=1 restores the old unwindowed stage, so the A/B is one
  // -D inside one build rather than a comparison against an older batch.
  constexpr int SPLIT_LEN =
      MPK_W2_STAGE_FULL ? REDUCTION_SIZE : (REDUCTION_SIZE / K_SPLITS);
  static_assert(MPK_W2_STAGE_FULL || SPLIT_LEN == SPLIT_ITERS * K_PER_MFMA,
                "the staged window must be exactly the split's MFMA range");
  int const stage_base = MPK_W2_STAGE_FULL ? 0 : k_base;
  if constexpr (INPUT_FP8) {
    // Same slab as the bf16 path, reinterpreted: the producer packed E4M3 into
    // the low REDUCTION_SIZE bytes and the per-32 E8M0 scales into the
    // REDUCTION_SIZE/32 bytes after them. 2 * REDUCTION_SIZE bytes were
    // reserved for the bf16 row, so both fit with room to spare.
    uint8_t const *slot_base =
        (uint8_t const *)(A +
                          static_cast<size_t>(tok_idx) *
                              (NUM_TOPK * REDUCTION_SIZE) +
                          static_cast<size_t>(topk_slot) * REDUCTION_SIZE);
    _gang_stage_mxfp8_nt<SPLIT_LEN>(slot_base + stage_base,
                                    slot_base + REDUCTION_SIZE +
                                        stage_base / 32,
                                    s_tok_fp8 + stage_base,
                                    s_tok_scales + stage_base / 32);
  } else {
    _gang_wave_parallel_fp8_quant_nt<SPLIT_LEN>(
        A + static_cast<size_t>(tok_idx) * (NUM_TOPK * REDUCTION_SIZE) +
            static_cast<size_t>(topk_slot) * REDUCTION_SIZE + stage_base,
        s_tok_fp8 + stage_base,
        s_tok_scales + stage_base / 32);
  }
#ifdef MPK_ENABLE_SUBPHASE_TIMING
  unsigned long long _sp_q1 = __builtin_amdgcn_s_memrealtime();
  if (tid == 0 && g_subphase_active) {
    atomicAdd(&g_subphase_ns[1][2], (_sp_q1 - _sp_q0) * 10);
    atomicAdd(&g_subphase_ns[1][7], 1ULL);
  }
#endif

  float rw = 0.0f;
  if constexpr (FUSE_MULSUMADD) {
    rw = d_routing_weight[tok_idx * NUM_TOPK + topk_slot];
  }

  // Token scale for MFMA iteration k. The quantizer coarsens to one E8M0 per
  // 128 elements and passes the same byte to all four g; the producer-emitted
  // layout keeps the hardware's per-32 granularity, so g selects its own
  // sub-block exactly as it does on the weight side.
  auto tok_sc = [&](uint8_t const *base, int k) -> int {
    return INPUT_FP8 ? (int)base[k * 4 + g] : (int)base[k];
  };

  // Epilogue, hoisted out of both parallelization branches; see W13.
  auto emit = [&](f32x4_t const &acc, int const out_base) {
    unsigned short const *bias_row = d_bias + local_eid * OUTPUT_STRIDE;

    if constexpr (FUSE_MULSUMADD) {
      // The shared expert rides in routing slot NUM_TOPK-1 with weight 1.0,
      // so it needs no special case here.
      float *ws_addr = d_workspace +
                       static_cast<size_t>(tok_idx) * OUTPUT_STRIDE + out_base;
      // Split-K: every split adds its own partial, so the bias belongs to
      // exactly one of them. rw multiplies the whole sum and so multiplies
      // each addend -- it needs no gate.
      bool const add_bias = (K_SPLITS == 1) || (k_split == 0);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        if (out_base + i < OUTPUT_SIZE) {
          atomicAdd(&ws_addr[i],
                    (acc[i] + (add_bias ? _gang_bf16_to_float(
                                              bias_row[out_base + i])
                                        : 0.0f)) *
                        rw);
        }
      }
    } else {
      unsigned short *out_addr =
          d_output + static_cast<size_t>(tok_idx) * (NUM_TOPK * OUTPUT_STRIDE) +
          static_cast<size_t>(topk_slot) * OUTPUT_STRIDE + out_base;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        if (out_base + i < OUTPUT_SIZE) {
          out_addr[i] = _gang_float_to_bf16(
              acc[i] + _gang_bf16_to_float(bias_row[out_base + i]));
        }
      }
    }
  };

  if constexpr (!K_PARALLEL) {
  for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; tile_iter++) {
    int const wave_tile = warp_id + tile_iter * NUM_WAVES;
    int const w_row = wave_tile * 16 + col;
    // Split-K enters here and nowhere else in the loop: the weight row, its
    // scale run, and the LDS activation are all advanced to this split's K
    // window, so the body below still counts from zero and only its trip count
    // changes. k_base is 0 when K_SPLITS == 1, so the unsplit code is
    // byte-for-byte what it was.
    // Row base + this split's k-window. Expressed in k-TILES times the
    // layout's k-stride, which reduces to the old k_base/2 (FP4) and
    // k_base/32 under the row-major default. See MPK_MOE_KMAJOR.
    int const k_tile_base = k_base / 128;
    uint8_t const *w_data_row =
        wg_data +
        (MPK_MOE_KMAJOR >= 1
             ? static_cast<size_t>(w_row / 16) * (16 * W_ROW_BYTES) +
                   static_cast<size_t>(w_row % 16) * W_BPK
             : static_cast<size_t>(w_row) * W_ROW_BYTES) +
        static_cast<size_t>(k_tile_base) * _gang_moe_w_kstride<WEIGHT_FP4>();
    int const row_scale_base =
        (MPK_MOE_KMAJOR >= 2
             ? (w_row / 16) * (16 * NUM_BLOCKS_32) + (w_row % 16) * 4
             : w_row * NUM_BLOCKS_32) +
        k_tile_base * _gang_moe_sc_kstride();
    uint8_t *s_tok_k = s_tok_fp8 + k_base;
    uint8_t *s_tok_sc_k = s_tok_scales + k_base / 32;

    // W2 is the phase whose tile time is NOT absorbed by the barrier behind it
    // (memory: glm-w2-imbalance-is-the-only-unabsorbed-spread), so this is the
    // half of the knob with a makespan story. SPLIT_ITERS is 16 at the default
    // K_SPLITS=1, which the default depth of 4 divides.
    f32x4_t acc = _gang_moe_kloop<WEIGHT_FP4, INPUT_FP8, MPK_MOE_PF_GROUPS,
                                  SPLIT_ITERS>(
        w_data_row, wg_scales, row_scale_base, s_tok_k, s_tok_sc_k, g);

    if (col == 0) {
      emit(acc, wg_idx * OUTPUT_PER_WG + wave_tile * 16 + g * 4);
    }
  }
  } else {
    // `|| !K_PARALLEL` keeps the condition dependent on a template parameter.
    // A non-dependent static_assert inside a discarded if-constexpr branch
    // still fires, which would break every OPW=64 build under split-K.
    static_assert(K_SPLITS == 1 || !K_PARALLEL,
                  "split-K and the K-parallel narrow tile are two ways to "
                  "split the same reduction; enabling both would double-count");
    // K-parallel; see the W13 kernel for the shape argument.
    static_assert(OUTPUT_PER_WG == 16,
                  "the K-parallel branch covers 16 output rows per workgroup");
    // This branch keeps the pre-MPK_MOE_KMAJOR row-major addressing. It is
    // measured negative and off by default, so it is guarded rather than
    // ported. The `|| !K_PARALLEL` keeps the condition dependent -- a
    // non-dependent static_assert fires even in a discarded if-constexpr arm.
    static_assert(MPK_MOE_KMAJOR == 0 || !K_PARALLEL,
                  "K-major weight is not wired into the K-parallel branch");
    constexpr int ITERS_PER_WAVE = MFMA_ITERS / NUM_WAVES;
    static_assert(MFMA_ITERS % NUM_WAVES == 0,
                  "MFMA_ITERS must be divisible by NUM_WAVES for K-parallel");
    static_assert(ITERS_PER_WAVE >= 4 && ITERS_PER_WAVE % 4 == 0,
                  "Depth-4 K-parallel requires ITERS_PER_WAVE a multiple of 4");

    int const ki_start = warp_id * ITERS_PER_WAVE;
    int const ki_end = ki_start + ITERS_PER_WAVE;
    int const w_row = col;
    uint8_t const *w_data_row =
        wg_data + static_cast<size_t>(w_row) * W_ROW_BYTES;
    int const row_scale_base = w_row * NUM_BLOCKS_32;

    f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

    i32x8_t a0 =
        _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, ki_start * K_PER_MFMA, g);
    int sa0 = (int)wg_scales[row_scale_base + ki_start * 4 + g];
    i32x8_t a1 = _gang_load_w_mfma_a<WEIGHT_FP4>(
        w_data_row, (ki_start + 1) * K_PER_MFMA, g);
    int sa1 = (int)wg_scales[row_scale_base + (ki_start + 1) * 4 + g];
    i32x8_t a2 = _gang_load_w_mfma_a<WEIGHT_FP4>(
        w_data_row, (ki_start + 2) * K_PER_MFMA, g);
    int sa2 = (int)wg_scales[row_scale_base + (ki_start + 2) * 4 + g];
    i32x8_t a3 = _gang_load_w_mfma_a<WEIGHT_FP4>(
        w_data_row, (ki_start + 3) * K_PER_MFMA, g);
    int sa3 = (int)wg_scales[row_scale_base + (ki_start + 3) * 4 + g];

// IMPORTANT: #pragma unroll 1 prevents ROCm miscompilation.
#pragma unroll 1
    for (int ki = ki_start; ki < ki_end; ki += 4) {
      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, ki * K_PER_MFMA, g);
        acc = _gang_mfma_w_x_f8<WEIGHT_FP4>(a0, b, acc, sa0,
                                            tok_sc(s_tok_scales, ki));
      }
      if (ki + 4 < ki_end) {
        int kt4 = (ki + 4) * K_PER_MFMA;
        a0 = _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, kt4, g);
        sa0 = (int)wg_scales[row_scale_base + kt4 / 32 + g];
      }

      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 1) * K_PER_MFMA, g);
        acc = _gang_mfma_w_x_f8<WEIGHT_FP4>(a1, b, acc, sa1,
                                            tok_sc(s_tok_scales, ki + 1));
      }
      if (ki + 5 < ki_end) {
        int kt5 = (ki + 5) * K_PER_MFMA;
        a1 = _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, kt5, g);
        sa1 = (int)wg_scales[row_scale_base + kt5 / 32 + g];
      }

      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 2) * K_PER_MFMA, g);
        acc = _gang_mfma_w_x_f8<WEIGHT_FP4>(a2, b, acc, sa2,
                                            tok_sc(s_tok_scales, ki + 2));
      }
      if (ki + 6 < ki_end) {
        int kt6 = (ki + 6) * K_PER_MFMA;
        a2 = _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, kt6, g);
        sa2 = (int)wg_scales[row_scale_base + kt6 / 32 + g];
      }

      if (ki + 3 < ki_end) {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 3) * K_PER_MFMA, g);
        acc = _gang_mfma_w_x_f8<WEIGHT_FP4>(a3, b, acc, sa3,
                                            tok_sc(s_tok_scales, ki + 3));
      }
      if (ki + 7 < ki_end) {
        int kt7 = (ki + 7) * K_PER_MFMA;
        a3 = _gang_load_w_mfma_a<WEIGHT_FP4>(w_data_row, kt7, g);
        sa3 = (int)wg_scales[row_scale_base + kt7 / 32 + g];
      }
    }

    if (col == 0) {
#pragma unroll
      for (int i = 0; i < 4; i++) {
        s_reduce[warp_id * OUTPUT_PER_WG + g * 4 + i] = acc[i];
      }
    }
    __syncthreads();

    if (warp_id == 0 && col == 0) {
      f32x4_t sum = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int w = 0; w < NUM_WAVES; w++) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
          sum[i] += s_reduce[w * OUTPUT_PER_WG + g * 4 + i];
        }
      }
      emit(sum, wg_idx * OUTPUT_PER_WG + g * 4);
    }
  }

  __syncthreads();
#ifdef MPK_ENABLE_SUBPHASE_TIMING
  if (tid == 0 && g_subphase_active) {
    atomicAdd(&g_subphase_ns[1][3],
              (__builtin_amdgcn_s_memrealtime() - _sp_q1) * 10);
  }
#endif
}

} // namespace kernel
