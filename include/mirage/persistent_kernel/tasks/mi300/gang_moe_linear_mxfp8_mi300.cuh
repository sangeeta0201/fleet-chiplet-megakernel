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
//
// MEASURED NOT TO WORK HERE, off the shipping 9.308 image (task #106):
//
//   loop                     flat  global  vmcnt waits
//   gang_gemv_mxfp8          0     16      7 6 5 4 3 2 1 ...
//   mla_decode_absorbed      0      9      17 ... 0
//   gang_rmsnorm_linear x4   0     12      0 1
//   gang_moe_w13             8      0      0 0
//   gang_moe_w2              8      0      0 0
//
// Every other MAC loop in the megakernel is global-addressed with a PARTIAL
// vmcnt.  The two MoE ones are the only all-flat, all-drain loops left, and
// they are precisely the ones MPK_MOE_WGLOBAL was written for.  So the knob's
// -0.108 ms came entirely from its OTHER half -- the B-side ds_read hoist --
// and the address space never landed.
//
// WHY.  `_gang_ld_g` casts at the LEAF, one dereference deep, and here that
// leaf sits three inlines below a `[&]` lambda whose captured base pointer is
// loop-carried:  load_w -> _gang_load_w_mfma_a_at_g -> _gang_load_fp4_mfma_b_g
// -> _gang_ld_g.  InferAddressSpaces has to walk that whole chain back to a
// provably-global root and does not; the dense kernel's `_rnlm8_load_w` calls
// the same helper one level down from a plain local and keeps its global_load.
// The cast is not wrong, it is just not load-bearing at this depth.
//
// THE FIX is to put the address space in the pointer TYPE and hand it in from
// the top of the k-loop, so no inference is needed: every byte of arithmetic
// below already happens on an addrspace(1) pointer.  See MPK_MOE_WGPTR.
template <bool FP4>
__device__ __forceinline__ i32x8_t
    _gang_load_w_mfma_a_at_g(uint8_t const *base, int byte_off, int g) {
  if constexpr (FP4) {
    return _gang_load_fp4_mfma_b_g(base + byte_off, 0, g);
  } else {
    return _gang_load_fp8_mfma_b_g(base + byte_off, 0, g);
  }
}

// The weight pointer as a TYPE, not as a cast at the point of use.
using _gang_gp8 = __attribute__((address_space(1))) uint8_t const *;

// ── MPK_MOE_STREAM_NT ────────────────────────────────────────────────────────
// `nt` on the MoE k-loop's weight and E8M0-scale loads. A REPLACEMENT-POLICY
// hint only: no scope bit changes, so coherence is untouched and this cannot
// affect correctness.
//
// Found by the task #97 ISA census of the shipping image. Every `nt` in the
// megakernel lives in exactly two places, and the k-loop is not one of them:
//
//   function                              loads          cache bits
//   gang_moe_w13_linear_mxfp8_kernel      16 dwordx4     (none)
//                                         16 ubyte       (none)
//   gang_moe_w2_linear_mxfp8_kernel       32 dwordx4     (none)
//                                         32 ubyte       (none)
//   gang_moe_w2_linear_mxfp8_kernel        8 dwordx4     sc0 sc1 nt   <- #52's
//                                                                     prefetch
//   gang_rmsnorm_linear_bias_topk_kernel  34             sc0 nt
//
// So W2's prefetch-across-barrier loads are already marked non-temporal and
// the k-loop three lines away from them is not. That asymmetry is an accident,
// not a decision -- nothing in either header argues for it.
//
// The stream has ZERO reuse, at every level:
//   * within a tile, each k-group's bytes are consumed once and dropped;
//   * across tiles on one XCD, each tile owns distinct N columns of the same
//     expert, so no two tiles read the same weight byte;
//   * across XCDs, global_tile = t * 8 + xcd_id spreads an expert's tiles over
//     all eight, and they are still distinct N -- no cross-XCD reuse either;
//   * across tokens, it is ~57 MB/layer/rank x 75 layers = 4.3 GB against a
//     256 MB MALL, so nothing survives to the next token.
//
// Yet it is the single largest consumer of cache capacity in the layer, and it
// is the only resident big enough to explain the recorded null
// glm-decode-hole-prefetch-is-evicted-not-absorbed -- "the decode hole is FREE
// but the prefetch is EVICTED". The lines that get evicted are precisely the
// ones tasks #98 (qkv_a L+1), #116 (o_proj) and #137 (speculative expert) put
// there, which is why all three measured neutral despite landing in free holes.
//
// This is the INVERSE of the MPK_BAR_POLL_NT finding. There, `nt` on the most
// read-shared line in the kernel was backwards and dropping it bought
// -0.206 ms. Here, the absence of `nt` on a never-re-read stream is backwards
// for the same reason, read the other way round.
//
// GATE: glm-moe-addrspace1-is-a-negative says the metric for this loop is
// s_waitcnt COUNT, not instruction form -- MPK_MOE_WGLOBAL cost +0.87 ms by
// perturbing the schedule, not the addressing. `__builtin_nontemporal_load`
// should set one bit on the same instruction at the same point in the
// schedule; verify nt count UP and waitcnt count UNCHANGED in the ISA before
// spending a GPU run.
//
// GATE RESULT -- PASSED, exactly. Per-function, arm 0 vs arm 1:
//   W13  loads 54 -> 54   nt  0 -> 32   s_waitcnt 70  -> 70   mfma 16 -> 16
//   W2   loads 100 -> 100 nt  8 -> 72   s_waitcnt 126 -> 126  mfma 32 -> 32
// Only cache bits move. Instruction count, schedule and MFMA placement are
// bit-identical, so this is not the MPK_MOE_WGLOBAL bet in disguise.
//
// WALL, NP=4 / bs=1 / devices 4-7, both arm orders, baseline as its own
// control in each batch (one 48 ms wedged run dropped from B/arm1):
//
//   batch          arm 1 (nt)        arm 0 (control)   delta
//   A  nt first    9.055  (n=5)      9.107  (n=5)      -0.052
//   B  ctl first   9.027  (n=4)      9.098  (n=5)      -0.071
//   pooled         9.042  (n=9)      9.103  (n=10)     -0.061
//
// Per-iter min tracks it: 8.918 vs 8.960, -0.043. One overlapping pair out of
// 90, so the separation is clean at this n even though -0.061 sits under the
// 0.26 ms single-run noise floor. Reproduces under order reversal, and it is
// free -- no extra instruction, no register, no schedule change. Default ON.
#ifndef MPK_MOE_STREAM_NT
#define MPK_MOE_STREAM_NT 1
#endif

// Pointer-typed twin of _gang_load_w_mfma_a_at_g.  Byte-for-byte the same
// addressing -- FP4 takes one 16-byte chunk at `+g*16`, FP8 takes that plus a
// second at `+g*16+64` -- but the base arrives already in addrspace(1), so the
// only cast left is a pointee-type reinterpret WITHIN that address space,
// which needs no inference at all.
template <bool FP4>
__device__ __forceinline__ i32x8_t
    _gang_load_w_mfma_a_at_gp(_gang_gp8 base, int byte_off, int g) {
  using gv4 = __attribute__((address_space(1))) i32x4_t const *;
  _gang_gp8 const p = base + byte_off + g * 16;
#if MPK_MOE_STREAM_NT
  i32x4_t lo = __builtin_nontemporal_load((gv4)p);
#else
  i32x4_t lo = *(gv4)p;
#endif
  i32x8_t r = {};
  r[0] = lo[0];
  r[1] = lo[1];
  r[2] = lo[2];
  r[3] = lo[3];
  if constexpr (!FP4) {
#if MPK_MOE_STREAM_NT
    i32x4_t hi = __builtin_nontemporal_load((gv4)(p + 64));
#else
    i32x4_t hi = *(gv4)(p + 64);
#endif
    r[4] = hi[0];
    r[5] = hi[1];
    r[6] = hi[2];
    r[7] = hi[3];
  }
  return r;
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

// Load the GROUPS-wide E8M0 scale batch off ONE base pointer with compile-time
// immediate offsets. See load_ws_batch's header in _gang_moe_kloop_deep for
// the sixteen-VGPR-for-four-bytes reading that motivates it.
//
// Measured -0.123 ms at NP=4 bs=1 on top of MPK_MOE_WGLOBAL=1: arm n=5 mean
// 9.718 (9.735 9.704 9.733 9.695 9.725) against control n=6 mean 9.841
// (min 9.790).  Arm max 9.735 < control min 9.790, so the ranges do not
// overlap.  Default ON; =0 is the ablation.
#ifndef MPK_MOE_SCBASE
#define MPK_MOE_SCBASE 1
#endif

// MPK_MOE_WGPTR: make MPK_MOE_WGLOBAL actually land, by threading
// addrspace(1) through the k-loop's pointer TYPE instead of casting at the
// leaf. See the long note on _gang_load_w_mfma_a_at_g for the image census
// that says the leaf cast is inert here -- the two MoE k-loops are the ONLY
// all-flat, all-`vmcnt(0)` MAC loops left in the megakernel, and they are the
// two the knob was written for.
//
// Nothing about the addressing changes: same base, same byte offsets, same
// number of loads, same MFMAs. Only the opcode (global_load, not flat_load)
// and therefore which counter the loads sit in. That is the whole point: a
// flat op increments lgkmcnt as well as vmcnt, so the two `s_waitcnt
// lgkmcnt(0)` that land this trip's B-operand ds_reads also retire every
// outstanding weight load. Read off the shipping W2 body [0x4278c..0x4290c]:
//
//   ds_read_b128 x3 ; ds_read_u8 x4        <- B operand, hoisted (WGLOBAL half 2)
//   s_waitcnt lgkmcnt(3) ; v_mfma          <- partial, good
//   s_waitcnt lgkmcnt(2) ; v_mfma
//   flat_load_ubyte    x4                  \  next block's weights + scales
//   flat_load_dwordx4  x3                  /
//   s_waitcnt lgkmcnt(0) ; v_mfma          <- DRAINS ALL SEVEN
//   flat_load_dwordx4  x1
//   s_waitcnt lgkmcnt(0) ; v_mfma          <- drains the eighth
//   s_waitcnt vmcnt(0)                     <- and again at the backedge
//
// Requires MPK_MOE_WGLOBAL=1 to have any effect; =0 keeps the generic loaders.
#ifndef MPK_MOE_WGPTR
#define MPK_MOE_WGPTR 1
#endif


// ── MPK_MOE_BSCHED ─────────────────────────────────────────────────────────
// Stop LLVM sinking the B-operand LDS reads below the MFMAs that consume them.
//
// THE DEFECT, read off the shipping gfx950 image (demo/glm5/isa_outstanding.py
// plus the raw body). The W13 trip at [0x4896c..0x48ac4] and both W2 trips at
// [0x49554..0x496a8] / [0x499b8..0x49b14] are HEALTHY ON THE VMEM SIDE -- 8
// loads carried across the backedge, peak 8, and ZERO `s_waitcnt vmcnt(0)` in
// W13. Every earlier attempt on this loop (MPK_MOE_PF_DBUF, MPK_MOE_PF_GROUPS,
// MPK_MOE_WGLOBAL) was aimed at that side, and it is now closed: the HBM
// pipeline is on the flat part of the loads-in-flight curve that
// tests/standalone/test_waves_per_simd_payoff.hip measured (4 -> 5334 GB/s,
// 8 -> 5446).
//
// What is left exposed is the LDS side. `consume` below issues all GROUPS B
// tiles before the first MFMA in SOURCE order, but the emitted W13 trip splits
// them 3 / 3 / 2 and RECYCLES v[12:19] as the B operand of MFMA 0, MFMA 2 and
// MFMA 3, so the trip carries TWO full `s_waitcnt lgkmcnt(0)`:
//
//   ds_read_b128 v[12:15] / v[16:19] / v[20:23]   <- MFMA 0's B, MFMA 1's lo
//   ds_read_u8   x4                               <- the four token scales
//   s_waitcnt lgkmcnt(3) ; lgkmcnt(2)
//   v_mfma ... v[12:19]
//   ds_read_b128 v[24:27] / v[12:15] / v[16:19]   <- v[12:19] REUSED
//   s_waitcnt lgkmcnt(4) ; lgkmcnt(2)
//   v_mfma ... v[20:27]
//   s_waitcnt lgkmcnt(0)                          <- FULL LDS DRAIN
//   v_mfma ... v[12:19]
//   ds_read_b128 v[12:15] / v[16:19]              <- REUSED AGAIN
//   s_waitcnt lgkmcnt(0)                          <- FULL LDS DRAIN
//   v_mfma ... v[12:19]
//
// GROUPS live i32x8_t of B is 32 VGPRs, and the allocator will not hold them,
// so it recycles and pays a drain for the WAR.
//
// THE CAUSE IS ALREADY DOCUMENTED ON THIS BRANCH, for the sibling dense kernel
// at gang_rmsnorm_linear_mxfp8_bias_mi300.cuh:2421: "the machine scheduler
// sinks all but two tiles back below the first MFMA -- its register-pressure
// heuristic targets an occupancy this kernel does not have (the megakernel runs
// one wave per SIMD) and it happily trades 16 loads in flight for a smaller
// live set." That kernel pins its batch with __builtin_amdgcn_sched_barrier(0)
// and keeps it. The MoE twin never got the same line: it is the ONLY
// sched_barrier in tasks/mi300, and this loop is the other caller of the same
// _gang_load_fp8_mfma_b batch idiom.
//
// 1  sched_barrier(0) between the B batch and the MFMA batch, which is the
//    dense kernel's exact form. Emits NO instruction -- it is a scheduling
//    fence, not code -- so the only thing it can cost is allocation.
//
//    MEASURED ON THE IMAGE AND REJECTED, standalone gfx950 build of
//    tests/standalone/test_mxfp8_moe.hip, W13 trip:
//
//                      insns  mfma  loads  ds  carry  lgkm(0)  vm(0)
//      BSCHED=0           72     4     12  12     12        2      1
//      BSCHED=1          105     4     12  12      8        1      1
//
//    It does exactly what it claims -- all eight ds_read_b128 land above the
//    first MFMA and one of the two LDS drains is gone -- and it is still a
//    loss, for the third time on this loop and by the same mechanism. The
//    +33 instructions are 19 `v_accvgpr_read_b32` and 5
//    `v_accvgpr_write_b32`: holding GROUPS B tiles live pushes the batch into
//    a[8:43], the weight loads follow it into a[0:7], and every one of them
//    then has to be copied back out because srcA is wanted in a VGPR. Two
//    dozen register moves to retire one drain, and HBM loads in flight fall
//    12 -> 8 on top. This is MPK_MOE_PF_DBUF's bill again, paid in AGPR
//    shuffles instead of VGPR count -- so it is refuted against the image and
//    NOT measured on the wall, exactly as MPK_DENSE_SCBASE was (`8e949cd`).
//
// 2  sched_group_barrier interleave. Same goal, register-light by
//    construction: place the ds_reads for B tile j+1 above the MFMA for tile
//    j so only ~2 tiles are live, which lets the wait name a PARTIAL lgkmcnt
//    instead of draining, without asking the allocator for 32 more registers.
//    VMEM is deliberately named in no group, so the healthy weight pipeline
//    (carry 8-12, zero vmcnt(0) in W13) is left for the scheduler to place as
//    it does today.
//
//    MEASURED ON THE IMAGE AND ALSO REJECTED, same build, same three loops:
//
//                      insns  carry  lgkm(0)  accvgpr
//      BSCHED=0    W13     72     12        2        0
//      BSCHED=2    W13     66     12        2        0
//      BSCHED=0    W2a     92     12        2        7
//      BSCHED=2    W2a     85     12        2        5
//      BSCHED=0    W2b     88     12        2        7
//      BSCHED=2    W2b     84     12        3        5
//
//    It is register-neutral as designed -- carry holds at 12, no AGPR shuffle
//    appears, and the body is 4-7 instructions SHORTER -- and it does not do
//    the one thing it was built for: the LDS drain count is unmoved at 2, and
//    W2's second loop gets a THIRD. Interleaving the reads does not help
//    because the wait LLVM emits is not placed by read order; the operand of
//    MFMA j is in a register the allocator is reusing, so the drain is a WAR
//    on the register, not a shortage of issue distance.
//
// SO THE DRAINS ARE NOT ADDRESSABLE FROM SOURCE AT GROUPS=4, in either
// direction. Pinning the batch buys the drain and pays 24 register moves;
// interleaving it is free and buys nothing. Both are off by default and this
// header is the record, so the next reader does not spend a third build on
// the LDS side of this loop. What the census DID establish is worth keeping
// separately: the vmem side is finished. carry 8-12 with zero-to-one vmcnt(0)
// per trip is the flat part of the loads-in-flight curve, so
// MPK_MOE_PF_GROUPS-style depth changes can no longer be justified as
// "more loads in flight" -- if that axis pays now it pays for another reason.
//
// The two masks this file needs, from the AMDGPU backend's
// SchedGroupMask: DS read is 0x100, MFMA/WMMA is 0x8. An instruction matching
// no named group is left to the scheduler, which is why VMEM is absent below.
#define MPK_MOE_SGB_DSREAD 0x100
#define MPK_MOE_SGB_MFMA 0x008
//
// GATE, and it is mandatory before any GPU run, per
// [[glm-moe-addrspace1-is-a-negative]]: the metric for this loop is s_waitcnt
// COUNT, not instruction form. Require, per kernel, lgkmcnt(0) per trip DOWN,
// v_mfma count UNCHANGED, and worker_kernel .vgpr_count NOT UP. A variant that
// raises the unified VGPR count is MPK_MOE_PF_DBUF in disguise -- that one
// bought a better schedule for +26 to +31 registers binary-wide and was
// measured a loss twice -- and must be dropped on the image, without spending
// a wall run.
//
// SECOND GATE, numerics, because this loop has a recorded SILENT miscompile:
// task #115 found the double-buffered form bit-exact at 4 live i32x8_t and
// 330x wrong at 8 (2.161e-04 vs 6.577e-07 max abs error) with no fault and no
// warning. Pinning the batch does not change the MFMA order -- `acc` is still
// accumulated in ascending j -- so a correct build is bit-identical to the
// shipping one, and tests/standalone/test_mxfp8_moe.hip is what says so.
#ifndef MPK_MOE_BSCHED
#define MPK_MOE_BSCHED 0
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
  // Both weight pointers re-typed into addrspace(1) ONCE, at the top of the
  // loop, so every byte of offset arithmetic below happens on a pointer that
  // is already global. MPK_MOE_WGPTR's header has the image census showing
  // why the old leaf-level cast in _gang_ld_g does not survive the inline
  // chain out of these lambdas.
  _gang_gp8 const wdr_g = (_gang_gp8)w_data_row;
  _gang_gp8 const wsc_g = (_gang_gp8)wg_scales;
  // The weight (A) operand and its E8M0 scale. Under MPK_MOE_WGLOBAL both go
  // through addrspace(1) so they leave lgkmcnt; see the knob's header.
  auto load_w = [&](int kk) -> i32x8_t {
    if constexpr (MPK_MOE_WGLOBAL && MPK_MOE_WGPTR) {
      return _gang_load_w_mfma_a_at_gp<WEIGHT_FP4>(wdr_g, kk * W_KS, g);
    } else if constexpr (MPK_MOE_WGLOBAL) {
      return _gang_load_w_mfma_a_at_g<WEIGHT_FP4>(w_data_row, kk * W_KS, g);
    } else {
      return _gang_load_w_mfma_a_at<WEIGHT_FP4>(w_data_row, kk * W_KS, g);
    }
  };
  auto load_ws = [&](int kk) -> int {
    int const off = row_scale_base + kk * SC_KS + g;
    if constexpr (MPK_MOE_WGLOBAL && MPK_MOE_WGPTR) {
#if MPK_MOE_STREAM_NT
      return (int)__builtin_nontemporal_load(wsc_g + off);
#else
      return (int)wsc_g[off];
#endif
    } else if constexpr (MPK_MOE_WGLOBAL) {
      return (int)_gang_ld_g<uint8_t>(wg_scales + off);
    } else {
      return (int)wg_scales[off];
    }
  };
  // Batched form of load_ws: ONE address register plus compile-time immediate
  // offsets for the whole GROUPS-wide batch, instead of GROUPS independent
  // 64-bit address chains.
  //
  // Read off the W2 body [0x4243c..0x425b8] of the shipping WGLOBAL=1 image:
  // the four scale bytes cost SIXTEEN VGPRs -- four 64-bit running offsets
  // (v[2:3] v[4:5] v[12:13] v[14:15], each bumped by s[0:1] at the backedge)
  // and four 64-bit addresses derived from them (v[52:53] v[54:55]
  // v[100:101] v[102:103]).  The weight side of the same loop already folds
  // to one address with offset:1024/2048/3072, so the four scale chains are
  // pure allocator waste; the compiler cannot fold them itself because
  // `row_scale_base + kk * SC_KS + g` is a signed 32-bit add whose sext does
  // not distribute (the classic sext-of-add strength-reduction blocker).
  //
  // Why it should matter beyond the register count: W2's trip opens with a
  // FULL `s_waitcnt vmcnt(0)`, which is the `minvm 0` the census reports.
  // Its cause is a WAR, not a true dependence -- the allocator picked
  // v[36:39] as the destination of the fourth weight load while v[38:39] is
  // that load's own address register, so the next trip's
  // `v_lshl_add_u64 v[38:39], ...` cannot issue until load #8 has landed.
  // Handing the allocator back twelve VGPRs is the source-level way to make
  // that collision unnecessary.  Register-REDUCING, hence not the same bet as
  // MPK_MOE_PF_DBUF, which lost on a +26 VGPR bill.
  auto load_ws_batch = [&](int kbase, int *out) {
    if constexpr (MPK_MOE_SCBASE && MPK_MOE_WGLOBAL && MPK_MOE_WGPTR) {
      // Same single base pointer, carried in addrspace(1) so the batch emits
      // global_load_ubyte off one address with immediate offsets.
      _gang_gp8 const p = wsc_g + (row_scale_base + kbase * SC_KS + g);
#pragma unroll
      for (int j = 0; j < GROUPS; j++) {
#if MPK_MOE_STREAM_NT
        out[j] = (int)__builtin_nontemporal_load(p + j * SC_KS);
#else
        out[j] = (int)p[j * SC_KS];
#endif
      }
    } else if constexpr (MPK_MOE_SCBASE) {
      uint8_t const *p = wg_scales + (row_scale_base + kbase * SC_KS + g);
#pragma unroll
      for (int j = 0; j < GROUPS; j++) {
        // j is a compile-time constant after the unroll, so j * SC_KS is an
        // immediate byte offset off the single pointer p.
        out[j] = MPK_MOE_WGLOBAL ? (int)_gang_ld_g<uint8_t>(p + j * SC_KS)
                                 : (int)p[j * SC_KS];
      }
    } else {
#pragma unroll
      for (int j = 0; j < GROUPS; j++) {
        out[j] = load_ws(kbase + j);
      }
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
#if MPK_MOE_BSCHED == 1
      // Keep the whole batch on this side of the MFMAs; see MPK_MOE_BSCHED.
      __builtin_amdgcn_sched_barrier(0);
#endif
#pragma unroll
      for (int j = 0; j < GROUPS; j++) {
        acc = _gang_mfma_w_x_f8<WEIGHT_FP4>(Ap[j], B[j], acc, Sp[j], BS[j]);
      }
#if MPK_MOE_BSCHED == 2
      // One B tile is two ds_read_b128 plus, amortised over the batch, one
      // ds_read_u8 of token scale: 3 DS reads per MFMA, 3 * GROUPS in the
      // trip. Lead by one tile, then step one tile per MFMA, so tile j+1 is
      // in flight while tile j's MFMA runs. See MPK_MOE_BSCHED.
      // 6 + 3*(GROUPS-2) == 3*GROUPS, so the groups name every DS read in the
      // trip exactly once and no more; over-naming would leave the scheduler
      // an empty group to fill and it silently drops the whole pipeline.
      static_assert(GROUPS >= 3,
                    "the leading two-tile DS group needs GROUPS >= 3");
      __builtin_amdgcn_sched_group_barrier(MPK_MOE_SGB_DSREAD, 6, 0);
#pragma unroll
      for (int j = 0; j < GROUPS - 2; j++) {
        __builtin_amdgcn_sched_group_barrier(MPK_MOE_SGB_MFMA, 1, 0);
        __builtin_amdgcn_sched_group_barrier(MPK_MOE_SGB_DSREAD, 3, 0);
      }
      __builtin_amdgcn_sched_group_barrier(MPK_MOE_SGB_MFMA, 2, 0);
#endif
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
  load_ws_batch(0, S);
#pragma unroll
  for (int j = 0; j < GROUPS; j++) {
    A[j] = load_w(j);
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
    load_ws_batch(base + GROUPS, NS);
#pragma unroll
    for (int j = 0; j < GROUPS; j++) {
      N[j] = load_w(base + GROUPS + j);
    }
    consume(base, A, S);
    // NO-GO, measured: sinking copy j to just after slot j's MFMA.
    //
    // The motivation is real. The block below reads N[GROUPS-1], the LAST load
    // issued, and vmcnt is in-order, so the whole block takes `vmcnt(0)` --
    // and it lands between MFMA 0 and MFMA 1 with three of the four still to
    // issue. Copy j alone reads only load j, so an interleaved form should
    // admit `vmcnt(GROUPS-1-j)` and let the drain walk down with the MFMAs.
    //
    // It emits a **byte-identical** hot body (W13 `[0x4178c..0x418e8]`) at an
    // unchanged 333 VGPRs / 0 spills. LLVM's scheduler already sinks and
    // hoists these copies freely; source placement is not the constraint. Not
    // measured on the GPU -- refuted against the image, as MPK_DENSE_SCBASE
    // was (`8e949cd`).
    //
    // What IS the constraint is the allocation. The rotate form lets the
    // allocator coalesce A[j] with N[j] into one physical register -- which is
    // why only three or four `v_mov_b64` survive here rather than 2*GROUPS --
    // and the `vmcnt(0)` is the price of that coalescing. Forbidding it is
    // exactly MPK_MOE_PF_DBUF, which costs +31 unified VGPRs (333 -> 364) and
    // was measured a loss twice. The two are alternatives, and LLVM already
    // picked the one the wall agrees with. Attack the *dependence*, not the
    // order: break the WAR that makes the 4th weight load's destination
    // collide with what the rotation reads.
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
//
// ── CLOSED NO-GO 2026-08-25 (task #115): CORRECT AT GROUPS=2, BROKEN AT 4 ──
//
// The claim two paragraphs up -- "same loads, same MFMAs, same order of
// issue-then-consume" -- is true of the SOURCE and false of what the machine
// computes at GROUPS=4. Measured on tests/standalone/test_mxfp8_moe.hip, W13
// max abs error against a double-precision dequant reference, arm vs the
// shipping deep loop on identical bytes:
//
//   deep@4  (shipping, 4 live i32x8_t)      6.577e-07
//   dbuf@2  (4 live)                        6.577e-07   <- bit-exact, correct
//   dbuf@4  (8 live)                        2.161e-04   <- 330x, BROKEN
//
// and in the model (NP=4, devices 4-7, bs=1), MPK_MOE_PF_DBUF_W13=1 at the
// shipping GROUPS=4 emits deterministic garbage 4/4 runs (correctness_gate
// distinct 0.010 / topbigram 99, prefix 0) while measuring -0.267 ms. The two
// oracles agree: the ONLY width that wins is the one that is wrong.
//
// It is not an index bug. The consume schedule was rewritten below to hoist the
// LDS B reads exactly as the deep loop does; that changed the error magnitude
// (1.122e-04 -> 2.161e-04) without fixing it. An off-by-one would be invariant
// to scheduling. The discriminator that does track the failure is the live
// buffer count -- 4 live is correct in either loop form, 8 live is wrong in
// either -- and at 8 live the standalone's allocator spills A[]/N[] to 900 B of
// scratch per lane. Suspect the spill/remat, not the loop algebra.
//
// Why this closes rather than continuing: dbuf@2 is correct but GROUPS=2
// narrows W13's block width at a measured +0.783 ms, which swamps the copy
// deletion (the p115 three-arm run: ctl 9.3077, gr2 10.0903, dbuf2 9.4420).
// There is no correct-and-winning point in the space. Do not re-price this
// without first re-running the standalone at GROUPS=4 and getting 6.577e-07.
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

  // Both weight pointers re-typed into addrspace(1) ONCE, exactly as the deep
  // loop does -- see MPK_MOE_WGPTR's header for why a leaf-level cast does not
  // survive the inline chain out of these lambdas.
  //
  // This variant needs it more than the deep loop does, and that is the whole
  // reason for the re-price. dbuf's recorded loss is a +26..31 VGPR bill from
  // holding SIXTEEN loads in flight instead of eight. A flat_ load carries a
  // full 64-bit VGPR address per chain; a global_ load off one addrspace(1)
  // base carries a 32-bit voffset plus an SGPR base with immediate offsets. If
  // the bill is addressing rather than data, doubling the loads in flight is
  // exactly where the two forms diverge most, and the earlier pricings were
  // both taken with dbuf on the flat_ path.
  _gang_gp8 const wdr_g = (_gang_gp8)w_data_row;
  _gang_gp8 const wsc_g = (_gang_gp8)wg_scales;

  // Issue block `b` into the given buffer. Taken by reference so the arrays
  // keep their identity -- passing them by value would reintroduce the copy
  // this whole variant exists to delete.
  auto issue = [&](i32x8_t (&dst)[GROUPS], int (&dsc)[GROUPS], int b) {
    int const kbase = b * GROUPS;
    // Scales first and batched: one base pointer plus compile-time immediate
    // offsets for the whole GROUPS-wide batch, instead of GROUPS independent
    // 64-bit address chains. Same argument as MPK_MOE_SCBASE in the deep loop,
    // and register-REDUCING, which is the axis dbuf loses on.
    if constexpr (MPK_MOE_SCBASE && MPK_MOE_WGLOBAL && MPK_MOE_WGPTR) {
      _gang_gp8 const p = wsc_g + (row_scale_base + kbase * SC_KS + g);
#pragma unroll
      for (int j = 0; j < GROUPS; j++) {
#if MPK_MOE_STREAM_NT
        dsc[j] = (int)__builtin_nontemporal_load(p + j * SC_KS);
#else
        dsc[j] = (int)p[j * SC_KS];
#endif
      }
    } else if constexpr (MPK_MOE_SCBASE) {
      uint8_t const *p = wg_scales + (row_scale_base + kbase * SC_KS + g);
#pragma unroll
      for (int j = 0; j < GROUPS; j++) {
        dsc[j] = MPK_MOE_WGLOBAL ? (int)_gang_ld_g<uint8_t>(p + j * SC_KS)
                                 : (int)p[j * SC_KS];
      }
    } else {
#pragma unroll
      for (int j = 0; j < GROUPS; j++) {
        dsc[j] = (int)wg_scales[row_scale_base + (kbase + j) * SC_KS + g];
      }
    }
#pragma unroll
    for (int j = 0; j < GROUPS; j++) {
      int const kk = kbase + j;
      if constexpr (MPK_MOE_WGLOBAL && MPK_MOE_WGPTR) {
        dst[j] = _gang_load_w_mfma_a_at_gp<WEIGHT_FP4>(wdr_g, kk * W_KS, g);
      } else if constexpr (MPK_MOE_WGLOBAL) {
        dst[j] = _gang_load_w_mfma_a_at_g<WEIGHT_FP4>(w_data_row, kk * W_KS, g);
      } else {
        dst[j] = _gang_load_w_mfma_a_at<WEIGHT_FP4>(w_data_row, kk * W_KS, g);
      }
    }
  };
  // Same shape as the deep loop's `consume`, INCLUDING the MPK_MOE_WGLOBAL
  // B-hoist. Getting that hoist here is not (only) a schedule question: the
  // interleaved form below issues an LDS read between every pair of MFMAs
  // while sixteen weight loads are outstanding, so SIInsertWaitcnts has to
  // interleave vmcnt and lgkmcnt waits inside the unrolled body. The hoisted
  // form asks for one lgkmcnt(0) up front and leaves vmcnt alone -- the same
  // structure the deep loop has been validated at.
  auto consume = [&](i32x8_t const (&src)[GROUPS], int const (&ssc)[GROUPS],
                     int b) {
    if constexpr (MPK_MOE_WGLOBAL) {
      i32x8_t B[GROUPS];
      int BS[GROUPS];
#pragma unroll
      for (int j = 0; j < GROUPS; j++) {
        int const kk = b * GROUPS + j;
        B[j] = _gang_load_fp8_mfma_b(s_tok_fp8, kk * K_PER_MFMA, g);
        BS[j] = tok_sc_at(kk);
      }
#pragma unroll
      for (int j = 0; j < GROUPS; j++) {
        acc = _gang_mfma_w_x_f8<WEIGHT_FP4>(src[j], B[j], acc, ssc[j], BS[j]);
      }
    } else {
#pragma unroll
      for (int j = 0; j < GROUPS; j++) {
        int const kk = b * GROUPS + j;
        i32x8_t bb = _gang_load_fp8_mfma_b(s_tok_fp8, kk * K_PER_MFMA, g);
        acc = _gang_mfma_w_x_f8<WEIGHT_FP4>(src[j], bb, acc, ssc[j],
                                            tok_sc_at(kk));
      }
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
// RE-PRICED ON TOP OF MPK_MOE_WGLOBAL=1 (2026-08-25) and it still loses, by
// about the same margin: n=4 mean 10.062 min 10.015 max 10.100, against the
// WGLOBAL-only n=6 mean 9.841 max 9.905. Non-overlapping again. So the
// register bill is charged whether the loads are flat_ or global_, and
// WGLOBAL's win does not come from anything dbuf would have supplied. The two
// are alternatives, not complements. worker_kernel vgpr 333 -> 364.
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
//
// It must also leave the pipeline INTACT, which is a second condition and was
// found the expensive way. The deep loop peels its last block, so the
// pipelined body runs NBLK - 1 = ki/gr - 1 times; at NBLK = 2 that is one
// trip, LLVM peels it, and the k-loop degenerates to straight-line code with
// no software pipeline at all. Measured on W2 (ki = 16) at gr = 8: the kernel
// goes from two 4-MFMA loops to ZERO MFMA loops, 32 MFMAs inlined, AGPR
// shuffles 16 -> 64, and the wall pays +0.238 ms. Requiring NBLK >= 3 makes a
// too-wide request fall back to the widest width that still pipelines --
// W2 at req 8 resolves to 4 -- instead of silently losing the loop.
__device__ __host__ constexpr int _gang_moe_pf_groups(int ki, int req) {
  for (int gr = (req < ki ? req : ki); gr >= 2; gr--) {
    if (ki % gr == 0 && ki / gr >= 3) {
      return gr;
    }
  }
  return 0;
}

// W13 and W2 select their prefetch width and their loop form INDEPENDENTLY.
//
// Measured 2026-08-25 (NP=4, bs=1, n=3 each, ctl 9.3077): the two are not one
// knob. Moving BOTH kernels together from the shipping GROUPS=4 deep loop to
// a GROUPS=2 double-buffered loop nets +0.14 ms, but that is the sum of two
// large and opposite effects --
//
//   ctl    DBUF=0 GR=4   9.323 / 9.290 / 9.310   -> 9.3077
//   gr2    DBUF=0 GR=2  10.071 / 10.073 / 10.127 -> 10.090   +0.78  narrowing
//   dbuf2  DBUF=1 GR=2   9.464 / 9.428 / ...     -> ~9.446   -0.64  copy kill
//
// -- so the copy deletion is worth 0.64 ms and only fails to ship because it
// was bundled with the narrowing that pays for it. The narrowing is forced:
// dbuf at GR=4 costs 335 -> 364 unified VGPRs binary-wide (see PF_DBUF's
// header), while GR=2 and GR=3 are both free at 335. Splitting the knobs lets
// each kernel take the widest dbuf-capable width its own trip count allows.
// W13 STAYS AT THE UNIFIED DEFAULT. It was briefly set to 8 on a -0.256 ms
// wall measurement; that measurement IS RETRACTED -- W13=8 EMITS GARBAGE.
//
// The wall numbers, NP=4 / bs=1 / devices 4-7, n=4 per arm, one build per arm,
// are reproducible (p130 g8 8.877/8.797 vs p131 w2_ctl 8.876/8.799, same
// config in separate batches, agreeing to 0.001 ms):
//
//   W13 depth, W2 held at 4        avg mean   per-iter-min
//     4 (ships)                      9.133        8.994
//     6                              8.966        8.849   -0.167
//     8                              8.877        8.797   -0.256   RETRACTED
//    12                              8.884        8.806   -0.249   RETRACTED
//
//   W2 depth, W13 held at 8        avg mean   per-iter-min
//     4                              8.876        8.799
//     8                              9.352        9.268   +0.476
//    16                              ~9.33        9.298   +0.499
//
// ...and they are meaningless, because at 128 generated tokens W13=8 answers
// "The capital of France is" with
//
//     <think>aniumaniumaniumaniumanium...        (135 of 136 tokens one bigram)
//
// against the control's clean list of European capitals, on two INDEPENDENT
// builds -- the correctness suite's g8 arm and a separate three-repeat
// recheck. The MoE output feeds the next layer's router, so garbage rewrites
// TopK and every downstream layer runs a different expert set: the timing is
// not a timing of this model. See
// [[glm-wrong-output-probes-upstream-of-router-are-invalid]] -- the MoE is
// downstream of ITS router but upstream of the NEXT one.
//
// The hangs are the same bug, not a separate flake. W13=8 finished 1 of 3
// standalone repeats and 1 of 8 correctness-suite runs; W13=4 finished 5 of 6.
// A run whose routing has gone degenerate spins somewhere it should not.
//
// It was NOT registers and NOT arithmetic. /tmp/vgpr.sh on the real image,
// worker_kernel .vgpr_count:
//
//   W13   4    6    8   12   16   24          W2 (at W13=8)   4    8   16
//       325  325  325  327  344  512(180sp)                 325  362  362
//
// -- W13 is 325 with ZERO spills at 4, 6 and 8, at MPK_MAX_SEQ_LENGTH 128 and
// 512 alike, so the famous 325 -> 362 step is W2 crossing depth 8 and W13
// never paid it. And the MFMA chain is `acc = mfma(A[j], B[j], acc)` in
// ascending j for every GROUPS, so the accumulation ORDER is identical at any
// depth: a correct deep loop is bit-identical to the shipping one.
//
// ROOT CAUSE, task #133, FIXED: batching the token-scale loads made
// s_tok_scales[k] provably wave-uniform, LLVM scalarised it through
// v_readfirstlane_b32, and v_mfma_scale_f32_16x16x128_f8f6f4 SILENTLY
// COMPUTES GARBAGE when a scale operand is an SGPR. See the writeup above
// _gang_mfma_f8xf8 in gang_linear_mxfp8_mi300.cuh; the fix is
// MPK_MFMA_VSCALE, an asm("" : "+v"(s)) launder that costs zero
// instructions. Only W13 was ever hit: it reads s_tok_scales[k]
// (lane-invariant), W2 reads s_tok_scales[k*4+g] (divergent, never
// scalarised). Oracle: tests/standalone/test_moe_kloop_width.hip, 20 s.
//
// So the axis is REAL and worth ~0.25 ms -- but every number in the two
// tables above was measured on a garbage build and must be re-taken on top
// of the fix (task #134). 3/6/12/24 remain unreachable through
// MPK_MOE_PF_GROUPS (it asserts membership in ("0","2","4","8","16") and
// W13's trip count is 48). Anything set here MUST be gated on generated
// text, not on the wall.
//
// ── TASK #134 RE-TAKEN, 2026-08-28: THE WALL REPRODUCES, THE TEXT DOES NOT ──
//
// The knobs were unreachable for a second reason nobody had hit: they were in
// env_common.sh's forward list but had NO wiring in persistent_kernel.py, so
// setting one changed nothing at all. That is now fixed, with a divisibility
// assert -- a width that does not divide the trip count makes
// _gang_moe_pf_groups fall back to the shipping loop, and the arm would
// silently measure the control.
//
// W13 depth 6 on top of MPK_MFMA_VSCALE=1, NP=4 / bs=1 / devices 4-7:
//
//   arm  MPK_MOE_PF_GROUPS_W13=6   8.810 8.873 8.938 8.805  -> 8.857  (n=4)
//   ctl  shipping depth 4          9.066 9.112             -> 9.089  (n=2)
//
// -0.232 ms, NON-OVERLAPPING (arm max 8.938 < ctl min 9.066), and the same
// sign and order as the retracted -0.167. Two runs were dropped as wedges
// (64.982 and 60.798 ms, the known cold-start stall, one in each arm).
//
// Every mechanical gate PASSES:
//   * tests/standalone/test_moe_kloop_width.hip reports MATCH -- bit-identical
//     accumulators -- at ship / deep 4,6,8,12,16 / dbuf 4,6,8.
//   * ZERO SGPR scale operands on v_mfma_scale_* in the arm image, so the
//     task #133 scalarisation this axis was retracted for is provably absent.
//   * worker_kernel 284 VGPR / 36 AGPR / 0 spills in BOTH arms. Depth 6 is
//     register-free, exactly as the earlier census claimed for 4, 6 and 8.
//   * G1 cross-rank identity PASSES, 4 of 4 ranks byte-identical, both prompts.
//
// The text sampling was initially AMBIGUOUS and is what the in-situ checksum
// below was built to resolve. On "The capital of France is", scored by the
// correctness gate's own G2 (distinct ratio < 0.30 or a bigram >= 15x), the
// arm was degenerate in 3 of 7 samples and the control in 0 of 4 -- Fisher
// p ~ 0.20, not significant, but the same failure SHAPE that retracted depth
// 8. Text cannot settle this at any affordable n, because
// correctness_gate.py's own header measures ~50% disagreement between two runs
// of the SAME build: the EP fold and the atomic accumulations retire in
// arrival order, that flips argmax on near-ties, and one flip cascades.
//
// ── SETTLED BY IN-SITU CHECKSUM, NOT BY SAMPLING ───────────────────────────
//
// MPK_BS_DEBUG=2 MPK_BSDBG_LAYER0=0 dumps a double-precision sum, absmax and
// the first two elements of every stage buffer, on all four ranks, for the
// first two fused layers. Arm against control, one run each, both builds
// verified distinct from the hipcc command line in their own logs (the arm
// carries -DMPK_MOE_PF_GROUPS_W13=6 on all four ranks; the control run logs
// mpirun's "could not find environment variable" for it):
//
//   68 comparable records, 68 IDENTICAL, 0 mismatches.
//
// That covers stage 5 (moe_norm_out, W13's INPUT) and stage 6 (swiglu_out,
// W13's OUTPUT) directly, and it covers the whole MoE half transitively --
// layer 1's stage 0 resid_in and stage 1 qkv_a_out are bit-identical too, and
// layer 1's input IS layer 0's complete output, W2's atomicAdd fold included.
//
// This is the gate the standalone oracle could not be: same register
// pressure, same routing, same EP, same four ranks, real weights. It answers
// the one open worry directly -- depth 6 holds A[6] + N[6] = TWELVE live
// i32x8_t, above the 8-live point where task #115 found a silent 330x error,
// and at twelve live this kernel is exact. The degenerate text samples are the
// documented bs=1 attractor nondeterminism, which both arms have and neither
// arm causes.
//
// ── WHY W13 AND W2 WANT DIFFERENT DEPTHS: IT IS NBLK, NOT REGISTERS ────────
//
// The header above says the two kernels "select their prefetch width and their
// loop form INDEPENDENTLY" and leaves that as an observation. It has a
// mechanism, and the mechanism also closes W2 for good.
//
// The deep loop peels its last block, so the PIPELINED body runs
// NBLK - 1 = KI_END/GROUPS - 1 times. Trip counts are 48 for W13 and 16 for
// W2, which is a 3x difference in how much room there is to spend on depth:
//
//   depth   W13: NBLK-1 trips        W2: NBLK-1 trips
//     2         23                        7
//     4         11                        3   <- W2 ships here
//     6          7   <- -0.232 ms         1   ! degenerate
//     8          5                        0   ! no loop at all
//    16          2                        0
//
// W2 at depth 8 was re-taken and it LOSES, +0.238 ms (9.327 n=3, variance
// 9.315-9.347, against the 9.089 control) -- the same SIGN as the old table's
// +0.476, so that number was not a garbage-build artifact after all. But the
// old note blamed "W2 crossing depth 8" on the 325 -> 362 register step, and
// that is NOT what the image says: worker_kernel is 284 VGPR / 36 AGPR / 0
// spills in BOTH arms, unchanged.
//
// What actually happens, per-function on the shipping vs depth-8 image:
//
//                     fn insns   total mfma   MFMA LOOPS      accvgpr
//   W2 depth 4 (ship)      660           16   [4-mfma, 4-mfma]     16
//   W2 depth 8             892           32   []                   64
//
// ZERO MFMA loops. At NBLK = 2 the pipelined body runs once, LLVM peels it,
// and the entire k-loop becomes straight-line code -- the software pipeline
// this whole file is about stops existing, eight tiles go live at once, and
// the allocator pays 64 AGPR shuffles for it. Depth 16 is worse still: NBLK-1
// is 0 and there is no loop to peel.
//
// So W2 is at its structural optimum at 4. Depth 2 is the only other
// non-degenerate width and it was measured +0.78 ms (the narrowing). W13 has
// 3x the trip count and therefore real room -- which is the whole asymmetry.
// The Python wiring now asserts NBLK-1 >= 2 so this cannot be selected by
// accident and re-measured a third time.
//
// SO THE DEFAULT STAYS 4. What is now settled and did not survive before: the
// wiring exists, the wall number is real and reproduced on a build whose known
// miscompile is gone, and the remaining blocker is a named, measurable one --
// get the arm's degenerate rate to the control's over ~20 samples per arm, or
// find what makes twelve live tiles differ from four. Do not re-derive the
// wall number; it is above.
// W13 DEFAULTS TO 6, NOT TO THE UNIFIED KNOB. -0.232 ms, and the numerics are
// verified BIT-IDENTICAL IN SITU, which is the gate the earlier attempt on
// this axis could not clear. See the task #134 block above for the wall
// numbers and the two-instrument correctness argument. W2 keeps inheriting the
// unified default because 4 is its structural maximum -- its trip count is 16,
// so anything wider stops pipelining.
#ifndef MPK_MOE_PF_GROUPS_W13
#define MPK_MOE_PF_GROUPS_W13 6
#endif
#ifndef MPK_MOE_PF_GROUPS_W2
#define MPK_MOE_PF_GROUPS_W2 MPK_MOE_PF_GROUPS
#endif
#ifndef MPK_MOE_PF_DBUF_W13
#define MPK_MOE_PF_DBUF_W13 MPK_MOE_PF_DBUF
#endif
#ifndef MPK_MOE_PF_DBUF_W2
#define MPK_MOE_PF_DBUF_W2 MPK_MOE_PF_DBUF
#endif

// Prefetch-depth dispatch. GROUPS_REQ == 0, or a trip count no divisor fits,
// selects the shipping loop.
template <bool WEIGHT_FP4,
          bool TOK_SC_FP8,
          int GROUPS_REQ,
          int KI_END,
          bool WANT_DBUF>
__device__ __forceinline__ f32x4_t
    _gang_moe_kloop(uint8_t const *w_data_row,
                    uint8_t const *wg_scales,
                    int row_scale_base,
                    uint8_t const *s_tok_fp8,
                    uint8_t const *s_tok_scales,
                    int g) {
  constexpr int GR = _gang_moe_pf_groups(KI_END, GROUPS_REQ);
  constexpr int NB = GR >= 2 ? KI_END / GR : 0;
  if constexpr (WANT_DBUF && GR >= 2 && NB >= 4 && NB % 2 == 0) {
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
// ── REOPENED 2026-09-01: that blocker is world-size-specific ──────────────
// "an 8-way split leaves 2" is arithmetic about EP_WORLD_SIZE == 8. It does
// not hold at 4, which is what the GLM-5 bring-up actually runs:
//
//   ranks  per-rank W2 reduction  MFMA_ITERS  static_assert(>=4 && %4==0)
//     8            2048/8 = 256           2   FAILS  <- the note above
//     4            2048/4 = 512           4   PASSES, exactly at the minimum
//
// The assert one screen down spells its own condition as "REDUCTION_SIZE %
// 512 == 0", and 512 divides 512. So at 4 ranks the K-shard is legal.
//
// Why it is worth the trouble, measured rather than argued. Per-rank MoE span,
// mean over ~38k layers, BARSTAGEWS slots 5..8, NP=4:
//
//   rank 0  19.418 us/layer      rank 2  13.716
//   rank 1  14.493               rank 3  14.145
//
// Rank 0 is EP_SHARED_PE, and it is 5.702 us/layer -- 0.367 ms/token -- slower
// than rank 2 in EVERY layer. The MoE phase ends when the last rank does, so
// that entire gap is on the critical path. Sharding the shared expert 4 ways
// takes the max from 3.0 expert-equivalents to 2.25.
//
// Two cautions carried forward from the note above, both still live:
//   - W13 is NOT affected by the assert. It reduces over HIDDEN (6144, so
//     MFMA_ITERS = 48) and is sharded on its OUTPUT dim, which leaves its
//     reduction untouched. Only W2's reduction is being cut.
//   - W2 at MFMA_ITERS == 4 sits at the exact pipeline minimum: every
//     iteration is fill and none is steady state. That is the shape that cost
//     W_UV its bake-off (see WUV_MFMA in demo.py), so some of the 0.367 ms
//     will be handed back. Price it before believing the whole number.
//
// And heed the failure mode recorded above: wrong output here passed Gate 1
// because all ranks agreed on garbage. Gate on generated text, or on
// MPK_BS_DEBUG=2 checksums against a 1-rank reference -- not on rank agreement.
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
    f32x4_t acc = _gang_moe_kloop<WEIGHT_FP4, false, MPK_MOE_PF_GROUPS_W13,
                                  MFMA_ITERS, (bool)MPK_MOE_PF_DBUF_W13>(
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
    f32x4_t acc = _gang_moe_kloop<WEIGHT_FP4, INPUT_FP8, MPK_MOE_PF_GROUPS_W2,
                                  SPLIT_ITERS, (bool)MPK_MOE_PF_DBUF_W2>(
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
