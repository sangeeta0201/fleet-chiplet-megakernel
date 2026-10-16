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

#pragma once
// MFMA MLA decode for GLM-5 (GlmMoeDsaForCausalLM), in the absorbed
// formulation. Structurally this is paged_attention_decode_minimal_hd64 with a
// different tiling; it reuses that file's MFMA / bf16-convert / exp2 helpers
// verbatim rather than re-deriving them.
//
// Absorbed MLA is MQA with asymmetric head dims. After folding W_UK into the
// query projection, each of the NUM_Q_HEADS query heads carries
//
//   q_absorbed = [ q_nope @ W_UK  (KV_LORA_RANK) | q_rope (QK_ROPE_HEAD_DIM) ]
//
// and the paged cache holds a *single* shared latent row per token
//
//   kv_row     = [ c_kv          (KV_LORA_RANK) | k_rope (QK_ROPE_HEAD_DIM) ]
//
// so QK reduces over QK_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM (576 for GLM-5)
// and PV accumulates over the leading KV_LORA_RANK dims of the *same* row —
// V is not a separate tensor. That halves the cache traffic relative to the
// GQA path and is why only one LDS staging buffer appears below.
//
// The kernel emits the KV_LORA_RANK-wide latent attention output per head;
// W_UV is folded into o_proj downstream, so no separate up-projection task is
// needed.
//
// Chiplet mapping. The GQA gang attention pins kv_head == xcd_id
// (gang_attention_mi300.cuh). MLA has one shared latent head, so instead the
// work is split two ways: the NUM_Q_HEADS query heads chunk into
// NUM_Q_GROUPS groups of 16 (one MFMA M tile each), and the sequence chunks
// into NUM_KV_CHUNKS. GLM-5 at NUM_Q_HEADS=64 with NUM_KV_CHUNKS=2 gives
// exactly 8 work items, one per XCD, and the latent cache is 1.15 KB/token so
// each XCD's slice stays L2-resident.
//
// A q-head group plays exactly the role a kv head plays in the GQA path, so
// the split-KV o_acc/lse_acc layouts below are bit-for-bit the ones
// merge_splitkv_ck_fmha already consumes: instantiate it with
// NUM_QO_HEADS_PER_KV = Q_HEADS_PER_GROUP, NUM_QO_GROUPS = NUM_Q_GROUPS,
// HEAD_DIM = KV_LORA_RANK, and pass q_head_group as its kv_head_idx.
//
// 256 threads, 4 warps x 64 lanes:
//   - QK_DIM / 32 QK MFMAs cover the reduction (18 for GLM-5)
//   - KV_LORA_RANK / 64 PV MFMAs per tile, each warp covering 16 output dims
//   - LDS: one KV_TILE x QK_DIM fp16 tile (18 KB for GLM-5), plain row-major
//
// Online softmax runs in log2 base (scale_s carries the log2(e) factor, same
// convention as the CK FMHA path).

#include <hip/hip_bf16.h>

// MPK_MLA_DECODE_DBLBUF: alternate the KV prefetch between two register
// buffers so tile t+2's loads survive the tile loop's backedge. Long note at
// kv_pre_odd in the prologue for the hazard it is meant to remove.
//
// MEASURED AND REFUTED, 2026-08-29. Keep it off. GLM-5 744B, NP=4, MI350,
// paired runs either side of one build each:
//
//   arm      8.980 mean / 8.936 min (n=3)
//   control  8.858 min (n=4; the 9.160 mean carries a 9.996 outlier)
//
// so roughly +0.08 ms, and an n=1 probe agreed at +0.108. The disassembly says
// why, and the reason is not the hazard analysis being wrong -- it is that
// there is no register headroom to pay for the fix:
//
//   loop           insns  loads  carry  drains   image scratch ops
//   single buffer    348     10      0       2               902
//   two buffers      495     20      0       5              1054
//
// Carry does not move. The +18 VGPRs (9 uint2) push a function whose arch peak
// is already 248 into spilling -- 152 more scratch ops in the image -- and the
// parity branch duplicates the refill/prefetch pair, which is where the extra
// three vmcnt(0) drains come from. See the register note above
// mla_decode_absorbed: this function sets the WHOLE megakernel's allocation,
// so it is the worst place in the tree to spend registers.
//
// Anything further here has to be carry-neutral in registers, or has to buy
// the headroom first.
//
// ── THE CARRY-NEUTRAL FORM: buffer_load_lds. 2026-09-02, NOT YET BUILT ─────
// The one shape that satisfies that constraint is a DMA prefetch.
// buffer_load_lds has NO DESTINATION VGPR, so depth is free in registers; the
// worked primitive and its hazards are in
// tests/standalone/test_w13_scale_locality.hip ("THE buffer_load_lds RING").
//
// That file's SIXTH PASS VERDICT closes the DMA axis for W13, and its point 2
// is "THE DMA PAYS A STRUCTURAL LDS TAX THE REGISTER PATH DOES NOT: a global
// load lands in a VGPR; a DMA load lands in LDS and must then be ds_read out."
// **That reasoning inverts here and the verdict does not carry over.** W13's
// register path delivers straight to MFMA operands and never touches LDS. This
// kernel's does the opposite -- look at refill_lds() below: the prefetch
// registers exist ONLY to be ds_written into lds_kv, so the path today is
//
//   global -> kv_pre_odd (VGPR) -> ds_write -> lds_kv -> ds_read -> MFMA
//
// and the DMA form deletes two of those stages rather than adding one. It is
// LDS-traffic NEGATIVE (the 9 ds_writes per tile go away), and it is register
// POSITIVE: kv_pre_odd's 9 uint2 = 18 VGPRs come back on the function that
// sets the whole megakernel's allocation. Double buffering then costs a second
// 18 KB lds_kv tile against the ~77 KB of free worker LDS, not 18 more VGPRs.
// So it fixes carry 0 and pays for itself twice, which is why it is worth
// building even though the register form measured +0.08 ms.
//
// TWO OBSTACLES, both real, neither obviously fatal:
//
//  1. LDS ADDRESSING IS RIGID. buffer_load_lds writes lds at
//     M0 + inst_offset + lane*4; it cannot honour this kernel's per-lane
//     lds_kv[my_tok * QK_DIM + my_dim0 + r * 64]. Either the LDS tile is
//     relaid out to the DMA's natural contiguous image (and the QK/PV
//     ds_reads re-indexed to match -- QK_DIM = 576 is not a power of two, so
//     check the bank conflicts before assuming this is free), or the DMA is
//     issued per token row with exec masking.
//
//  2. THE KV IS PAGED. Each lane resolves its own get_kv_row(), so a tile is
//     a 16-row GATHER in general and one DMA cannot fetch 16 arbitrary rows.
//     What may rescue it: KV_TILE is 16 tokens and page_size is 4096, so 16
//     CONSECUTIVE tokens fall inside one page except when the tile straddles a
//     boundary, and inside a page the rows are contiguous -- 16 * 576 * 2 =
//     18432 B, which is 4.5 dwordx4 instructions across 256 lanes. Confirm the
//     contiguity from get_kv_row before building, and keep the existing
//     register path as the straddle fallback.
//
// THE REGISTER HALF OF THE PAYOFF IS ZERO. CHECKED 2026-09-02 BEFORE BUILDING,
// which is the trap point 3 of the W13 verdict documents. On today's image
// worker_kernel is 325 VGPR / 69 AGPR / 0 spills. gfx950's unified file is 512
// per thread, so 325 is already 1 wave/SIMD and 2 waves needs <= 256. Handing
// back kv_pre_odd's 18 takes 325 -> ~307: the same 1 wave/SIMD, no occupancy
// step, nothing at the wall. Cutting 69 to reach 256 is the only register
// change here that would buy anything, and 18 is not a quarter of it.
//
// So the case rests ENTIRELY on the overlap half, and that case is weaker than
// the VGPR arithmetic above makes it look, for a reason the DBLBUF measurement
// already showed: adding a second register buffer did NOT move carry off 0.
// The drain is not (only) a WAR hazard on kv_pre_odd -- it is the
// __syncthreads() this loop needs because QK and PV both read the WHOLE
// lds_kv tile, so no lane may refill it until every lane is done (see the
// comment above refill_lds). A second REGISTER buffer cannot remove that
// barrier; a second LDS buffer can, which is the real argument for the DMA
// form and should be stated that way rather than as a register saving:
//
//   what it buys   one fewer full-block barrier per tile, and tile t+2's
//                  fetch in flight across tile t+1's 18 QK + 8 PV MFMAs and
//                  50 LDS reads
//   what it costs  a second 18 KB lds_kv against ~77 KB free
//   what it does   NOT buy: registers, occupancy, or LDS traffic beyond the
//                  9 ds_writes/tile that the DMA deletes
//
// ── PRICED 2026-09-02, AND THE WHOLE AXIS IS VOID AT THE BENCHMARK SHAPE ───
// Subphase slots for this kernel (MPK_SUBPHASE_TIMING=1, SP bank 2, ns summed
// over workers and layers, one 24-token run):
//
//   slot                                        total      share
//   3  cold start (Q + KV tiles 0/1 + LDS wr)   247.9M      29%
//   4  per-tile QK / softmax / PV               231.0M      27%
//   6  epilogue                                 224.6M      26%
//   5  per-tile refill  <- the DMA target       129.3M      15%
//   2  chunk partition                           29.5M       3%
//
// Slot 5 is 15% of the decode, and the DMA form removes only part of it, so
// even a perfect implementation is worth ~0.05-0.09 ms against a 0.26 ms noise
// floor. Do not build it. But the reason is sharper than "the slot is small":
//
// **THE TILE LOOP IS SINGLE-TRIP HERE, SO THERE IS NO BACKEDGE TO CARRY LOADS
// ACROSS.** KV_TILE is 16 and the decode splits the sequence over
// GLM_MLA_NUM_KV_CHUNKS = 16 chunks, so chunk_len = ceil(seq/16) and
// ntiles = ceil(chunk_len/16) is ONE for any seq <= 256. The benchmark runs
// max_seq_length 128 (and the live KV is shorter still -- a 10-token prompt
// plus 24 generated). So:
//
//   * `prefetch_t2` never fires: `t + 2 < ntiles` is false at ntiles == 1.
//   * carry is 0 because the loop has no second trip, not because of a WAR
//     hazard or the __syncthreads. THAT is why DBLBUF "did not move carry"
//     and measured +0.08 ms -- it bought a second buffer for a loop that
//     never goes round, so the measurement captured its cost and none of its
//     benefit. The same is true of any pipelining change here.
//   * cold start + epilogue is 55% of the decode, larger than the tile
//     compute, exactly as a single-trip loop implies.
//
// CONSEQUENCES, and they reach past this kernel:
//   1. Nothing on the software-pipelining axis for this loop can be evaluated
//      at max_seq_length <= 256. First re-measure at seq >= 512, where
//      ntiles >= 2 gives the loop a backedge.
//   2. The 1.13 ms "MLA decode ceiling" in CLOSING_LEDGER §8 is, at this
//      shape, almost entirely per-invocation FIXED cost: 64 tiles
//      (q_groups 4 x chunks 16) each running one nearly-empty 16-token tile
//      over a KV of ~34. The levers it admits are invocation count and the
//      prologue/epilogue, NOT the tile loop.
//   3. That also re-reads the chunks sweep (4 -> 14.189, 16 -> 11.464 ms):
//      what 16 chunks parallelised was the FIXED cost, since the work per
//      chunk is one tile either way.
//
// Caveat on the table: the instrumented build runs 13.3 ms against 8.75
// uninstrumented, so absolute values are inflated. Slots 3-6 are each stamped
// once per invocation at ntiles == 1, so the shares are comparable to each
// other; do not read them as ms of the shipping wall.
#ifndef MPK_MLA_DECODE_DBLBUF
#define MPK_MLA_DECODE_DBLBUF 0
#endif

// MPK_MLA_DECODE_BAR_LDS: HIP __syncthreads() is
//   s_waitcnt vmcnt(0) ; s_barrier ; s_waitcnt vmcnt(0)
// and that is 16 of the decode kernel's 26 full vmcnt(0) drains (8 s_barrier
// in the image). The tile loop only needs an LDS barrier: the first wait is
// "KV rows are in lds_kv" (ds_write, lgkmcnt), the second is "every lane is
// done with QK/PV ds_reads" (also lgkmcnt). Draining vmcnt at those points
// is what made MPK_MLA_DECODE_DBLBUF a null -- tile t+1's global loads cannot
// live across the backedge if the barrier retires them. gpt-oss's counted
// vmcnt pattern is the same idea: do not wait on a payload you will not use
// yet. Default OFF. Compile-time; every rank must agree.
//
// MEASURED NULL at 1024/1024, 2026-09-24 (3 alternating pairs, G1 PASS):
// decode median 12.064/12.053/12.071 -> 12.061/12.055/12.049 ms. The loop
// has no load in flight at either barrier to protect (the register prefetch
// it was written for is gone), so the vmcnt(0) it removes was already free.
//
// MEASURED NULL at 1024/1024, 2026-09-24 (3 alternating pairs, G1 PASS):
// decode median 12.064/12.053/12.071 -> 12.061/12.055/12.049 ms. The loop
// has no load in flight at either barrier to protect (the register prefetch
// it was written for is gone), so the vmcnt(0) it removes was already free.
#ifndef MPK_MLA_DECODE_BAR_LDS
#define MPK_MLA_DECODE_BAR_LDS 0
#endif
#if MPK_MLA_DECODE_BAR_LDS
__device__ __forceinline__ void mla_tile_bar() {
  asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
  __builtin_amdgcn_s_barrier();
}
#else
__device__ __forceinline__ void mla_tile_bar() { __syncthreads(); }
#endif

// MPK_MLA_DECODE_DMA_PF: fetch the next full KV tile straight into a second
// LDS tile buffer (buffer_load_dwordx4 ... lds) while the current tile's
// QK/softmax/PV runs. This is the carry-neutral form
// described above, and unlike the short-prompt shape it was priced against,
// 1k/1k gives the loop 3-4 trips: per-token time steps up 0.76 ms at every
// 512 tokens of KV (seqlen 513/1025/1537, flat in between), i.e. ~10 us of
// layer makespan per serial 18 KB trip.
//
// Issued from inline asm on purpose: the llvm.amdgcn LDS-DMA builtin is
// tracked as an LDS write of unknown extent, so the waitcnt pass drains it
// before the first ds_read of the current tile and the overlap is gone. The
// compute section issues no VMEM, so the DMA flies until the trip's trailing
// barrier retires it.
//
// Requires PAGE_SIZE % KV_TILE == 0 (a tile never straddles a page, so its
// rows are one contiguous run), KV_CACHE_STRIDE == QK_DIM (rows land in LDS
// verbatim) and KV_LDS_BYTES more dynamic LDS past the o_acc spill. Partial
// tiles keep the register path, which zero-fills the rows past tile_len: a
// DMA'd stale row can hold a NaN pattern, and 0 * NaN poisons PV.
//
// MEASURED 2026-09-24, on top of MPK_MLA_OACC_INTERLEAVE: correct (46/46),
// but the tile trip was never load-bound. Isolated: -0.5 us per trip at 3-4
// trips, +0.6 us at one. 1024/1024, 3 alternating pairs, all G1 PASS:
//   decode median  9.851/9.866/9.862 -> 9.856/9.880/9.837 ms   (null)
//   prefill avg    9.269/9.290/9.280 -> 9.383/9.396/9.353 ms   (+0.1, loss)
// Off.
#ifndef MPK_MLA_DECODE_DMA_PF
#define MPK_MLA_DECODE_DMA_PF 0
#endif

// MPK_MLA_OACC_VGPR: keep the 32 loop-carried o_acc floats in VGPRs and issue
// the trip's eight PV MFMAs back to back with VGPR accumulators, instead of
// the LDS spill below (ds_read, lgkmcnt(0), v_accvgpr_write, MFMA, 32 nops,
// v_accvgpr_read, ds_write, lgkmcnt(0) -- eight serial rounds per trip). The
// spill was introduced for a fused ntiles>=2 collapse blamed on RA; the
// allocation-dependent VALU->MFMA hazard fixed in a013ffd fits that symptom
// too, and mla_pv8_vgpr pads both hazards itself.
//
// Deletion probes on the isolated decode (128 items, 32 chunks, 256 threads
// at the megakernel's 512-register budget -- without __launch_bounds__(256,1)
// the bench spills and the comparison inverts), us per 16-token trip:
//   shipped 3.45   no QK 2.53   no PV 1.77   no KV load ~3.1
//   OACC_VGPR 1.85 (PV 1.68 -> ~0.1, QK 0.92 -> 0.41); one-trip launch
//   10.09 -> 8.69 us. test_mla_decode 46/46 at both register budgets.
//
// MEASURED 2026-10-02, NP=8 GPUs 0-7, 1024/1024, 2 alternating pairs, G1 PASS,
// coherent text: decode median 9.613/9.580 -> 9.288/9.318 ms (-0.294).
// Gates (NP=4): ppl512 2.6148, ppl128 2.9048 (torch 2.4073, ratio 1.207),
// longseq 256/512/1024 G1 PASS, test_mla_merge and test_st_wt PASS. Default 1.
// MPK_MLA_KV_SWZ: XOR-swizzled KV tile in LDS; see mla_kv_swz_src. Off, the
// addressing below is the unswizzled code verbatim (routing it through
// mla_kv_idx cost 10 VGPRs and 36 B of scratch on the whole megakernel).
// MEASURED NEGATIVE 2026-10-11, NP=8 1024/1024, with the reads routed through
// mla_kv_idx: test_mla_decode's SQ_LDS_BANK_CONFLICT 1,747,200 -> 0 on the
// 64-chunk kernels, but the megakernel 4.278 4.276 -> 4.345 4.343 ms (382
// VGPRs, 384 B scratch). The per-lane bases below (370 VGPRs, 160 B, the 8
// extra in AGPRs) make the standalone kernels 2-4% faster instead, e.g. the
// 64-chunk pair 13.04 / 13.81 -> 12.49 / 13.64 us; 60/60 both budgets.
// Megakernel NULL 2026-10-11 on MPK_WUK_EARLY_PF + MPK_ROPE_REMAP: 4.130 vs
// 4.132 4.128 ms. That was before the QK loop kept K reads in flight
// (MPK_MLA_QK_PIPE), when each read was waited for on its own. With the
// pipelined loop running on the swizzled tile it pays: DEC_Q_INC +
// RESOLVE_POLL_INC 3.789 3.783 -> + KV_SWZ 3.764 3.763 -> + QK_PIPE2 3.752
// 3.753 ms, text identical (MPK_TRACE: the decode is ~6 us/layer of the
// critical path, MPK_MLA_SKIP_DECODE 3.377 vs 3.845). Default on (gates g14).
#ifndef MPK_MLA_KV_SWZ
#define MPK_MLA_KV_SWZ 1
#endif
#ifndef MPK_MLA_OACC_VGPR
#define MPK_MLA_OACC_VGPR 1
#endif

// Two smaller trip levers on top of MPK_MLA_OACC_VGPR, same isolated bench,
// us per trip: OACC_VGPR 1.875; + QK_2ACC 1.77; + DECODE_BATCH=4 1.57; both
// 1.38. 46/46 at both register budgets for each.
//
// MEASURED 2026-10-02, NP=8 GPUs 0-7, 1024/1024, decode median, G1 PASS:
//   OACC_VGPR (4 runs)   9.277 9.307 9.344 9.279   mean 9.302
//   + QK_2ACC            9.210 9.276               mean 9.243  (-0.06)
//   + DECODE_BATCH=4     9.242 9.253               mean 9.248  (-0.05)
// Both inside the control spread at n=2, as the bench predicts (-0.1 and
// -0.3 us/trip at ~3.5 trips). Off. Every run pair, controls included, first
// diverges at token 92-126, so off-topic tails are not evidence either way.
// Together, re-measured 2026-10-08 at the 6.47 ms EP_LL default (n=2):
// 6.471 6.476 -> 6.484 6.488 ms, and QK_2ACC's summation order moves the
// tokens. Off.
// QK_2ACC alone again 2026-10-10 on the PRELOAD=2 default, n=3 interleaved:
// 4.779 4.791 4.794 -> 4.776 4.777 4.769 ms, tokens move. Off.
// DECODE_BATCH re-measured 2026-10-11 on 5df165e (every item two tiles at
// 1024-2048 context): control 4.004, =2 4.079, =4 4.062 ms. Off.
//
// MPK_MLA_QK_2ACC: split the 18-step QK MFMA chain over a[32:35] (even k32
// steps) and a[36:39] (odd), so consecutive MFMAs are independent.
//
// MPK_MLA_DECODE_BATCH=NB: land up to NB KV tiles in NB LDS buffers at once
// (full tiles by LDS DMA, the partial tail by the register path), so a chunk
// pays the global latency once per batch, and the trips inside a batch run
// with no barrier. The buffers start at lds_kv and cover the o_acc spill, so
// it needs OACC_VGPR; NB * 18 KB of dynamic LDS.
#ifndef MPK_MLA_QK_2ACC
#define MPK_MLA_QK_2ACC 0
#endif
// MPK_DEC_QK_KSPLIT: the four waves split QK's 18 k32 steps instead of each
// running all 18 for the same scores -- wave w takes steps w, w + 4, w + 8,
// ... (5, 5, 4, 4, one swizzle parity each), stages only those Q fragments,
// and the four 16 x 16 partials are summed in wave order through the dead Q
// staging block after one LDS barrier. A wave's chain is 5 deep, not 18, and
// it reads a quarter of the K tile and of Q. VALUE CHANGE: the score sum is
// reassociated (((p0 + p1) + p2) + p3), so the tokens may move; ppl-gated.
// test_mla_decode 66/66 at both WRITE_THROUGH arms (the test sizes its LDS
// for the 96 KB exchange). Final ISA: 5 QK MFMAs a wave instead of 18,
// ds_read_b128 72 -> 28, one more s_barrier a trip.
// MEASURED SLOWER 2026-10-15 on 792d601, NP=8 1024/1024, n=3 interleaved,
// tokens moved, text coherent, G1 PASS: 3.377 3.380 3.367 vs 3.365 3.371
// 3.365 ms (+0.008). Sparse trace (S48/S75): the trip-0 QK is 0.40 us as one
// 18-deep chain and 0.44 us split -- the chain was never latency-bound, the
// four waves run it side by side on four SIMDs, and the exchange costs what
// the shorter chain saves. The Q stage shrinks 0.44 -> 0.28 us, but ~1% of
// items (the newest chunks', three tiles) grow ~1 us and one of them is the
// last item. Trip 0 on the default: QK 0.40, softmax 0.16, PV 0.44 us. Off.
#ifndef MPK_DEC_QK_KSPLIT
#define MPK_DEC_QK_KSPLIT 0
#endif
// MPK_DEC_WPT: a chunk of 2..4 KV tiles runs its tiles side by side, one per
// wave, with one cross-wave exchange of the tile maxima and weights before
// PV, instead of one trip per tile on all four waves (see the tile loop).
// VALUE CHANGE: the softmax is combined against the chunk max in one step
// rather than online across trips; ppl-gated.
// test_mla_decode 66/66 at both WRITE_THROUGH arms; worker_kernel unchanged
// (offline: 390 VGPRs, 340 SGPR spills). Sparse trace (S48/S77/S49): a
// two-tile item's QK + exchange + PV is 1.92 us against two 1.0 us trips,
// compute 2.20 -> 2.08 us and the first tile barrier 0.40 -> 0.20 us; the
// merge start follows the last decode item, which stays the rendezvous' last
// arrival by ~4 us.
// MEASURED 2026-10-16 on 1b7c0ff, NP=8 1024/1024, interleaved, tokens moved,
// text coherent, G1 PASS: 3.356 3.363 3.353 vs 3.366 3.359 3.373 ms; again
// 3.352 3.357 vs 3.368 3.359; again 3.351 3.358 vs 3.364 3.370. Pooled n=7
// pairs: 3.356 vs 3.366 (-0.010); confirmed as the default 3.352 3.349 vs
// 3.359 3.366. Standalone 90/0 and longseq 256/512/1024 G1 PASS.
// FAILS PPL: ppl512 2.4493 on all 8 ranks against 2.3670 (=0, same stage).
// Per-position NLL is identical through position 14 and departs at 15, the
// first step with two live chunks -- below 1025 tokens every chunk is one
// tile, so the WPT branch never runs there. It reproduces with the branch
// compiled in but unreachable (wpt &&= seqlen_k < 0): 2.4493, same first
// position. So the code's presence changes the one-tile path's result,
// deterministically; the register-path KV loads stay behind the buffer_inv
// in both images. Not localized -- a compile-sensitive hazard in the default
// tile loop the standalone test does not see. Off.
#ifndef MPK_DEC_WPT
#define MPK_DEC_WPT 0
#endif
// MPK_DEC_VT: under MPK_DEC_WPT, a two-tile chunk's preloaded tiles also
// exist as V^T -- [512 dims][16 tokens] bf16 (rows padded to 20) at 36 KB of
// dynamic LDS -- so PV reads a
// lane's four tokens of one dim as one ds_read_b64 instead of four
// ds_read_u16 out of the swizzled rows. They are transposed in the decode's
// prologue, ahead of the query poll (the span the query takes to land); a
// tile on the register path -- the current row's -- keeps the u16 reads, so
// the last item pays no transpose after its query. Same values:
// bit-identical to MPK_DEC_WPT (test_mla_decode 66/66, preload cases
// bit-identical; megakernel tokens identical to WPT's).
// MEASURED NULL 2026-10-16 on 1b7c0ff + MPK_DEC_WPT, n=2 interleaved: 3.350
// 3.355 vs WPT alone 3.351 3.358 ms. The median item's compute drops 2.04 ->
// 1.64 us (sparse trace), but the worker image grows 390 -> 394 VGPRs and
// re-allocates. A first form that also transposed the register-path tile
// after its load, with a bank-conflicted transpose, put ~1.2 us ahead of the
// last item's QK: +0.03 ms. Off.
#ifndef MPK_DEC_VT
#define MPK_DEC_VT 0
#endif
#if MPK_DEC_VT && !MPK_DEC_WPT
#error "MPK_DEC_VT's transposed tiles are read by MPK_DEC_WPT's PV"
#endif
// V^T row stride in bf16: 16 tokens padded to 20 (40 B), so the transpose's
// lane-per-dim b64 stores and PV's lane-per-(dim, token quad) b64 loads both
// stay within two-way bank conflicts. Two tiles at 36 KB: past the two
// preloaded tiles, in the buffers MPK_DEC_WPT only uses at three or four
// tiles, where V^T is off.
#define MPK_DEC_VTS 20
#define MPK_DEC_VT_TILE (512 * MPK_DEC_VTS)
#define MPK_DEC_VT_LDS_OFF (36 * 1024)
#ifndef MPK_MLA_DECODE_BATCH
#define MPK_MLA_DECODE_BATCH 0
#endif
#if MPK_MLA_DECODE_BATCH && !MPK_MLA_OACC_VGPR
#error "MPK_MLA_DECODE_BATCH lays its tile buffers over the o_acc spill, so it needs MPK_MLA_OACC_VGPR"
#endif
#if MPK_MLA_DECODE_BATCH && MPK_MLA_DECODE_DMA_PF
#error "MPK_MLA_DECODE_BATCH and MPK_MLA_DECODE_DMA_PF both own the KV tile buffers"
#endif

// __mfma_qk_hd64 / __mfma_pv_hd64 / __fast_exp2_hd64 / __load_bf16x4_to_fp16 /
// __load_bf16x4_raw / __cvt_bf16x4_to_fp16 live here. They are tiling-agnostic
// despite the hd64 name.
#include "tasks/mi300/paged_attention_decode_minimal_hd64_mi300.cuh"
#include "mpk_atoms.cuh"
#if MPK_DEC_PRE_NOBAR && (MPK_MLA_DECODE_BATCH || MPK_MLA_DECODE_DMA_PF)
#error "MPK_DEC_PRE_NOBAR's barrier plan is the default tile loop's"
#endif
#if MPK_DEC_QK2T && (MPK_MLA_QK_SPLIT || MPK_MLA_QK_2ACC || !MPK_MLA_QK_PIPE2 || \
                     !MPK_MLA_KV_SWZ || MPK_MLA_DECODE_BATCH || MPK_MLA_DECODE_DMA_PF)
#error "MPK_DEC_QK2T is a variant of the pipelined swizzled QK and owns a[36:39]"
#endif
#if MPK_DEC_QK_KSPLIT && (MPK_MLA_QK_SPLIT || MPK_MLA_QK_2ACC || MPK_DEC_QK2T || \
                          !MPK_MLA_KV_SWZ || !MPK_MLA_OACC_VGPR ||          \
                          MPK_MLA_DECODE_BATCH || MPK_MLA_DECODE_DMA_PF || \
                          MPK_DEC_PRE_NOBAR)
// PRE_NOBAR: the score exchange relies on every trip's closing barrier to
// order the next trip's partial writes behind this trip's reads.
#error "MPK_DEC_QK_KSPLIT replaces the swizzled single-chain QK of the default tile loop"
#endif
#if MPK_DEC_WPT && (MPK_DEC_QK_KSPLIT || MPK_MLA_QK_SPLIT || MPK_MLA_QK_2ACC || \
                    MPK_DEC_QK2T || MPK_MLA_DECODE_BATCH || MPK_MLA_DECODE_DMA_PF)
#error "MPK_DEC_WPT replaces the default tile loop's trips"
#endif

#if MPK_DEC_OUT_B128 || MPK_DEC_MERGE_LL32
typedef int __attribute__((ext_vector_type(4))) mpk_dec_i32x4_t;
extern "C" __device__ void
__mpk_dec_raw_buffer_store_v4i32(mpk_dec_i32x4_t data, mpk_dec_i32x4_t rsrc,
                                 int voffset, int soffset, int aux)
    __asm("llvm.amdgcn.raw.buffer.store.v4i32");
#endif

namespace kernel {

namespace gang_mla_decode_detail {
// POD vectors, used instead of uint2/uint4 for anything loaded through an
// addrspace(1) pointer: the HIP_vector_type classes have a copy constructor
// taking a generic `uint2 const&`, which undoes the cast before the load ever
// happens. ext_vector_type is loaded directly and keeps .x/.y/.z/.w.
typedef unsigned int __attribute__((ext_vector_type(2))) u32x2_t;
typedef unsigned int __attribute__((ext_vector_type(4))) u32x4_t;

// A load that is *known* to come from device memory.
//
// Clang infers address spaces intraprocedurally, so a pointer that arrives as
// an argument to this __noinline__ kernel stays generic and every dereference
// is emitted as flat_load rather than global_load. On gfx9 a flat instruction
// increments **both** vmcnt and lgkmcnt, so the `s_waitcnt lgkmcnt(0)` that
// retires an LDS read also waits on every outstanding KV load -- which defeats
// the whole point of the tile prefetch below. Same fix and same reasoning as
// gang_gemv_mxfp8_detail::ld_g; the cast is two-step because clang rejects a
// reinterpret_cast that changes pointee type and address space at once.
//
// Safe because every pointer handed to this kernel is device global: the q
// workspace, the paged latent cache, the index arrays and the outputs all come
// from the megakernel's workspace or from a task descriptor.
template <typename T>
__device__ __forceinline__ T ld_g(void const *p) {
  T const *q = static_cast<T const *>(p);
  return *(__attribute__((address_space(1))) T const *)q;
}

// MPK_MLA_KV_SWZ: LDS chunk q (16 B) of the tile -- row q / 72, chunk c --
// holds the row's chunk c ^ (row >> 1). Rows are 72 chunks (576 bf16), 8 mod 16,
// so unswizzled the 16 rows a QK ds_read_b128 spans sit on two of gfx950's 16
// 16-byte bank groups (8-way); swizzled, chunk ^ rotl(row) covers all 16.
// Returns the byte offset into the source tile for LDS chunk q.
__device__ __forceinline__ unsigned mla_kv_swz_src(unsigned q) {
  unsigned const r = q / 72u;
  unsigned const c = q - r * 72u;
  return (r * 72u + (c ^ (r >> 1))) * 16u;
}
// Element (row, d) of a KV tile in LDS, in bf16 units; see mla_kv_swz_src.
template <int QK_DIM>
__device__ __forceinline__ int mla_kv_idx(int row, int d) {
#if MPK_MLA_KV_SWZ
  return row * QK_DIM + ((((d >> 3) ^ (row >> 1)) << 3) | (d & 7));
#else
  return row * QK_DIM + d;
#endif
}
#if MPK_MLA_DECODE_DMA_PF || MPK_MLA_DECODE_BATCH || MPK_DEC_PRELOAD
typedef int __attribute__((ext_vector_type(4))) mla_i32x4_t;

// Raw buffer V# over [base, base + range_bytes), wave-uniform.
__device__ __forceinline__ mla_i32x4_t mla_buffer_rsrc(void const *base,
                                                       unsigned range_bytes) {
  unsigned long long const a = reinterpret_cast<unsigned long long>(base);
  mla_i32x4_t r;
  r[0] = static_cast<int>(
      __builtin_amdgcn_readfirstlane(static_cast<unsigned>(a)));
  r[1] = static_cast<int>(
      __builtin_amdgcn_readfirstlane(static_cast<unsigned>(a >> 32)));
  r[2] = static_cast<int>(range_bytes);
  r[3] = 0x00020000;
  return r;
}

// One 18 KB KV tile by LDS DMA: four 4 KB rounds on every wave, plus a fifth
// on waves 0-1 for the last 2 KB. Round j, lane l copies 16 bytes from
// rsrc + voff_l + 4096 j to LDS m0 + 16 l, with m0 = this wave's 1 KB slice
// of round j. Default cache policy, the same as ld_g.
//
// Every round sits in ONE asm block. Anything the compiler places between
// two rounds that touches VMEM -- a scratch reload under this function's
// register pressure, as the first build showed -- arrives with a vmcnt wait,
// and vmcnt retires in issue order, so it drains every round before it.
//
// Each block opens with s_nop 4: rsrc's base words come from v_readfirstlane,
// a VALU write of an SGPR needs 5 wait states before a VMEM instruction reads
// it, and the compiler does not insert them in front of inline asm.
#define MLA_DMA_SGPR_WAIT "s_nop 4\n\t"
#define MLA_DMA_ROUND(M, S)                                                    \
  "s_mov_b32 m0, %[" M "]\n\t"                                                 \
  "s_nop 0\n\t"                                                                \
  "buffer_load_dwordx4 %[voff], %[rsrc], %[" S "] offen lds\n\t"
__device__ __forceinline__ void mla_dma_tile(mla_i32x4_t rsrc,
                                             unsigned m0_base, unsigned voff,
                                             bool tail_wave) {
  unsigned const m1 = m0_base + 4096u, m2 = m0_base + 8192u,
                 m3 = m0_base + 12288u, m4 = m0_base + 16384u;
#if MPK_MLA_KV_SWZ
  // voff is tid * 16; round k fills LDS chunks 256 k + tid.
  unsigned const q0 = voff / 16u;
  unsigned const v0 = mla_kv_swz_src(q0), v1 = mla_kv_swz_src(q0 + 256u),
                 v2 = mla_kv_swz_src(q0 + 512u), v3 = mla_kv_swz_src(q0 + 768u),
                 v4 = mla_kv_swz_src(q0 + 1024u < 1152u ? q0 + 1024u : 0u);
  unsigned const z = 0u;
  if (tail_wave) {
    asm volatile(MLA_DMA_SGPR_WAIT
                 "s_mov_b32 m0, %[m0]\n\ts_nop 0\n\t"
                 "buffer_load_dwordx4 %[v0], %[rsrc], %[z] offen lds\n\t"
                 "s_mov_b32 m0, %[m1]\n\ts_nop 0\n\t"
                 "buffer_load_dwordx4 %[v1], %[rsrc], %[z] offen lds\n\t"
                 "s_mov_b32 m0, %[m2]\n\ts_nop 0\n\t"
                 "buffer_load_dwordx4 %[v2], %[rsrc], %[z] offen lds\n\t"
                 "s_mov_b32 m0, %[m3]\n\ts_nop 0\n\t"
                 "buffer_load_dwordx4 %[v3], %[rsrc], %[z] offen lds\n\t"
                 "s_mov_b32 m0, %[m4]\n\ts_nop 0\n\t"
                 "buffer_load_dwordx4 %[v4], %[rsrc], %[z] offen lds"
                 :
                 : [rsrc] "s"(rsrc), [z] "s"(z), [v0] "v"(v0), [v1] "v"(v1),
                   [v2] "v"(v2), [v3] "v"(v3), [v4] "v"(v4), [m0] "s"(m0_base),
                   [m1] "s"(m1), [m2] "s"(m2), [m3] "s"(m3), [m4] "s"(m4)
                 : "memory", "m0");
  } else {
    asm volatile(MLA_DMA_SGPR_WAIT
                 "s_mov_b32 m0, %[m0]\n\ts_nop 0\n\t"
                 "buffer_load_dwordx4 %[v0], %[rsrc], %[z] offen lds\n\t"
                 "s_mov_b32 m0, %[m1]\n\ts_nop 0\n\t"
                 "buffer_load_dwordx4 %[v1], %[rsrc], %[z] offen lds\n\t"
                 "s_mov_b32 m0, %[m2]\n\ts_nop 0\n\t"
                 "buffer_load_dwordx4 %[v2], %[rsrc], %[z] offen lds\n\t"
                 "s_mov_b32 m0, %[m3]\n\ts_nop 0\n\t"
                 "buffer_load_dwordx4 %[v3], %[rsrc], %[z] offen lds"
                 :
                 : [rsrc] "s"(rsrc), [z] "s"(z), [v0] "v"(v0), [v1] "v"(v1),
                   [v2] "v"(v2), [v3] "v"(v3), [m0] "s"(m0_base),
                   [m1] "s"(m1), [m2] "s"(m2), [m3] "s"(m3)
                 : "memory", "m0");
  }
  return;
#endif
  unsigned const s0 = 0u, s1 = 4096u, s2 = 8192u, s3 = 12288u, s4 = 16384u;
  if (tail_wave) {
    asm volatile(MLA_DMA_SGPR_WAIT MLA_DMA_ROUND("m0", "s0")
                     MLA_DMA_ROUND("m1", "s1") MLA_DMA_ROUND("m2", "s2")
                         MLA_DMA_ROUND("m3", "s3") MLA_DMA_ROUND("m4", "s4")
                 :
                 : [rsrc] "s"(rsrc), [voff] "v"(voff), [m0] "s"(m0_base),
                   [m1] "s"(m1), [m2] "s"(m2), [m3] "s"(m3), [m4] "s"(m4),
                   [s0] "s"(s0), [s1] "s"(s1), [s2] "s"(s2), [s3] "s"(s3),
                   [s4] "s"(s4)
                 : "memory", "m0");
  } else {
    asm volatile(MLA_DMA_SGPR_WAIT MLA_DMA_ROUND("m0", "s0")
                     MLA_DMA_ROUND("m1", "s1") MLA_DMA_ROUND("m2", "s2")
                         MLA_DMA_ROUND("m3", "s3")
                 :
                 : [rsrc] "s"(rsrc), [voff] "v"(voff), [m0] "s"(m0_base),
                   [m1] "s"(m1), [m2] "s"(m2), [m3] "s"(m3), [s0] "s"(s0),
                   [s1] "s"(s1), [s2] "s"(s2), [s3] "s"(s3)
                 : "memory", "m0");
  }
}
#undef MLA_DMA_ROUND
#undef MLA_DMA_SGPR_WAIT
#endif

// Native bf16 MFMA. gfx950 runs bf16 at the same rate as fp16, so the latent
// cache -- which is bf16 in memory -- has no reason to be widened to f32 and
// narrowed to fp16 on the way to the matrix core. Dropping that round trip
// removes ~340 VALU ops per invocation, but far more importantly it removes a
// *dependency*: with fp16 the loaded dwords had to be converted before they
// could be used, so the Q load and the KV prologue each stalled on their own
// data. Loaded straight as bf16 they are already the MFMA operand registers.
//
// It is also strictly better numerically on the K side: bf16 -> fp16 narrows
// the exponent from 8 bits to 5 and can overflow to inf on a large latent
// value. The P operand loses mantissa (8 bits vs 11), which is the usual
// bf16 flash-attention tradeoff and is absorbed by the fp32 accumulator.
typedef __bf16 __attribute__((ext_vector_type(8))) bf16x8_t;
typedef __bf16 __attribute__((ext_vector_type(4))) bf16x4_t;

// QK is pinned to a[32:35] so the builtin QK MFMA cannot land on a[0:3].
// PV uses the GPT-OSS pattern: o_acc lives in VGPRs and the builtin PV
// MFMA borrows a[0:3] only for that instruction. The fused ntiles>=2
// collapse was RA parking the 32 o_acc floats in a[0:31] AND in VGPRs,
// then issuing the next tile's KV loads into those VGPRs. Loop-carried
// o_acc therefore goes through LDS, not AGPRs.
__device__ __forceinline__ void mla_qk_acc32_zero() {
  asm volatile("v_accvgpr_write_b32 a32, 0\n"
               "v_accvgpr_write_b32 a33, 0\n"
               "v_accvgpr_write_b32 a34, 0\n"
               "v_accvgpr_write_b32 a35, 0\n"
               :
               :
               : "a32", "a33", "a34", "a35");
}

#if MPK_DEC_QK2T
// mla_qk_acc32_mfma_kv on a[36:39]: MPK_DEC_QK2T's second tile's chain.
__device__ __forceinline__ void mla_qk_acc36_mfma_kv(bf16x8_t av,
                                                    __bf16 const *b) {
  bf16x8_t bv;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    bv[i] = b[i];
  }
  u32x4_t avu, bvu;
  __builtin_memcpy(&avu, &av, 16);
  __builtin_memcpy(&bvu, &bv, 16);
  asm volatile("s_nop 1\n"
               "v_mfma_f32_16x16x32_bf16 a[36:39], %0, %1, a[36:39]\n"
               :
               : "v"(avu), "v"(bvu)
               : "a36", "a37", "a38", "a39");
}

// Both chains' scores, a[32:35] and a[36:39], kept apart.
__device__ __forceinline__ void
    mla_qk_acc2_read_pair(__mfma_hd64_fp32x4 &s, __mfma_hd64_fp32x4 &t) {
  float s0, s1, s2, s3, t0, t1, t2, t3;
  asm volatile("s_nop 15\n"
               "s_nop 15\n"
               "v_accvgpr_read_b32 %0, a32\n"
               "v_accvgpr_read_b32 %1, a33\n"
               "v_accvgpr_read_b32 %2, a34\n"
               "v_accvgpr_read_b32 %3, a35\n"
               "v_accvgpr_read_b32 %4, a36\n"
               "v_accvgpr_read_b32 %5, a37\n"
               "v_accvgpr_read_b32 %6, a38\n"
               "v_accvgpr_read_b32 %7, a39\n"
               : "=v"(s0), "=v"(s1), "=v"(s2), "=v"(s3), "=v"(t0), "=v"(t1),
                 "=v"(t2), "=v"(t3)
               :
               : "a32", "a33", "a34", "a35", "a36", "a37", "a38", "a39");
  s[0] = s0;
  s[1] = s1;
  s[2] = s2;
  s[3] = s3;
  t[0] = t0;
  t[1] = t1;
  t[2] = t2;
  t[3] = t3;
}
#endif

#if MPK_MLA_QK_PIPE || MPK_MLA_QK_PIPE2
// mla_qk_acc32_mfma with the K chunk already in registers.
__device__ __forceinline__ void mla_qk_acc32_mfma_kv(bf16x8_t av,
                                                    __bf16 const *b) {
  bf16x8_t bv;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    bv[i] = b[i];
  }
  u32x4_t avu, bvu;
  __builtin_memcpy(&avu, &av, 16);
  __builtin_memcpy(&bvu, &bv, 16);
  asm volatile("s_nop 1\n"
               "v_mfma_f32_16x16x32_bf16 a[32:35], %0, %1, a[32:35]\n"
               :
               : "v"(avu), "v"(bvu)
               : "a32", "a33", "a34", "a35");
}
#endif

__device__ __forceinline__ void mla_qk_acc32_mfma(__bf16 const *a,
                                                 __bf16 const *b) {
  bf16x8_t av, bv;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    av[i] = a[i];
    bv[i] = b[i];
  }
  u32x4_t avu, bvu;
  __builtin_memcpy(&avu, &av, 16);
  __builtin_memcpy(&bvu, &bv, 16);
  // The s_nop is CDNA4 ISA Table 38: a VALU write of a VGPR followed by an
  // MFMA read of it needs 2 wait states (no forwarding path). The hazard
  // recognizer does not pad in front of an MFMA it cannot see inside asm, and
  // the operands here are VALU-produced (v_accvgpr_read out of the AGPRs Q is
  // parked in), so without it correctness depends on register allocation:
  // one allocation put the last v_accvgpr_read one instruction ahead of this
  // MFMA and 42 of 46 test_mla_decode cases failed.
  asm volatile("s_nop 1\n"
               "v_mfma_f32_16x16x32_bf16 a[32:35], %0, %1, a[32:35]\n"
               :
               : "v"(avu), "v"(bvu)
               : "a32", "a33", "a34", "a35");
}

// MPK_MLA_QK_2ACC chains. The AGPRs are named in the asm, so RA can never
// copy them between steps.
__device__ __forceinline__ void mla_qk_acc2_zero() {
  asm volatile("v_accvgpr_write_b32 a32, 0\n"
               "v_accvgpr_write_b32 a33, 0\n"
               "v_accvgpr_write_b32 a34, 0\n"
               "v_accvgpr_write_b32 a35, 0\n"
               "v_accvgpr_write_b32 a36, 0\n"
               "v_accvgpr_write_b32 a37, 0\n"
               "v_accvgpr_write_b32 a38, 0\n"
               "v_accvgpr_write_b32 a39, 0\n"
               :
               :
               : "a32", "a33", "a34", "a35", "a36", "a37", "a38", "a39");
}

template <int ODD>
__device__ __forceinline__ void mla_qk_acc2_mfma(__bf16 const *a,
                                                __bf16 const *b) {
  bf16x8_t av, bv;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    av[i] = a[i];
    bv[i] = b[i];
  }
  u32x4_t avu, bvu;
  __builtin_memcpy(&avu, &av, 16);
  __builtin_memcpy(&bvu, &bv, 16);
  // s_nop 1: VALU-produced operands, as in mla_qk_acc32_mfma.
  if constexpr (ODD) {
    asm volatile("s_nop 1\n"
                 "v_mfma_f32_16x16x32_bf16 a[36:39], %0, %1, a[36:39]\n"
                 :
                 : "v"(avu), "v"(bvu)
                 : "a36", "a37", "a38", "a39");
  } else {
    asm volatile("s_nop 1\n"
                 "v_mfma_f32_16x16x32_bf16 a[32:35], %0, %1, a[32:35]\n"
                 :
                 : "v"(avu), "v"(bvu)
                 : "a32", "a33", "a34", "a35");
  }
}

__device__ __forceinline__ __mfma_hd64_fp32x4 mla_qk_acc2_read() {
  float s0, s1, s2, s3, t0, t1, t2, t3;
  asm volatile("s_nop 15\n"
               "s_nop 15\n"
               "v_accvgpr_read_b32 %0, a32\n"
               "v_accvgpr_read_b32 %1, a33\n"
               "v_accvgpr_read_b32 %2, a34\n"
               "v_accvgpr_read_b32 %3, a35\n"
               "v_accvgpr_read_b32 %4, a36\n"
               "v_accvgpr_read_b32 %5, a37\n"
               "v_accvgpr_read_b32 %6, a38\n"
               "v_accvgpr_read_b32 %7, a39\n"
               : "=v"(s0), "=v"(s1), "=v"(s2), "=v"(s3), "=v"(t0), "=v"(t1),
                 "=v"(t2), "=v"(t3)
               :
               : "a32", "a33", "a34", "a35", "a36", "a37", "a38", "a39");
  __mfma_hd64_fp32x4 r;
  r[0] = s0 + t0;
  r[1] = s1 + t1;
  r[2] = s2 + t2;
  r[3] = s3 + t3;
  return r;
}

__device__ __forceinline__ __mfma_hd64_fp32x4 mla_qk_acc32_read() {
  float s0, s1, s2, s3;
  asm volatile("s_nop 15\n"
               "s_nop 15\n"
               "v_accvgpr_read_b32 %0, a32\n"
               "v_accvgpr_read_b32 %1, a33\n"
               "v_accvgpr_read_b32 %2, a34\n"
               "v_accvgpr_read_b32 %3, a35\n"
               : "=v"(s0), "=v"(s1), "=v"(s2), "=v"(s3)
               :
               : "a32", "a33", "a34", "a35");
  __mfma_hd64_fp32x4 r;
  r[0] = s0;
  r[1] = s1;
  r[2] = s2;
  r[3] = s3;
  return r;
}

__device__ __forceinline__ __mfma_hd64_fp32x4
    mfma_qk_bf16(__mfma_hd64_fp32x4 c, __bf16 const *a, __bf16 const *b) {
  bf16x8_t av, bv;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    av[i] = a[i];
    bv[i] = b[i];
  }
  return __builtin_amdgcn_mfma_f32_16x16x32_bf16(av, bv, c, 0, 0, 0);
}

__device__ __forceinline__ __mfma_hd64_fp32x4
    mfma_pv_bf16(__mfma_hd64_fp32x4 c, __bf16 const *a, __bf16 const *b) {
  bf16x4_t av, bv;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    av[i] = a[i];
    bv[i] = b[i];
  }
  return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, bv, c, 0, 0, 0);
}

// GPT-OSS HD=64 keeps one 4-wide o_acc in VGPRs. MLA has eight PV
// blocks. Loop-carried o_acc goes through LDS, not AGPRs: fused RA
// otherwise parks a[0:31] copies in VGPRs that the next tile's KV
// loads then overwrite (tile 0 hides it, rescale==0).
//
// ds addresses are LDS-absolute. The megakernel has ~7 KB of static
// __shared__ that slides extern __shared__, so tid*128+KV_BYTES from
// 0 lands inside lds_kv and fused decode emits token 0. Standalone
// has no static prefix, which is why the oracle still passed.
// Base = byte address of lds_kv plus the lane's slot (tid*4 float-major,
// tid*128 lane-major; see MPK_MLA_OACC_INTERLEAVE); KV_BYTES is in the ds
// offset field so RA cannot drop it.
constexpr int MLA_OACC_V_BLOCKS = 8;
constexpr int MLA_OACC_FLOATS = MLA_OACC_V_BLOCKS * 4; // 32

// MPK_MLA_OACC_INTERLEAVE: lay the o_acc spill out float-major, [32][256],
// instead of lane-major, [256][32]. Lane-major puts consecutive lanes 128 B
// = 32 dwords apart, so every o_acc ds_read_b32/ds_write_b32 -- eight per PV
// block, eight blocks per tile trip, each behind an lgkmcnt(0) -- sends all
// 64 lanes to one LDS bank. Float-major makes each of those instructions
// touch 64 consecutive dwords. Same 32 KB footprint, offsets still constant
// (max KV_BYTES + 31 * 1024 = 50176, inside the 16-bit ds offset).
//
// MEASURED 2026-09-24. Isolated decode (128 work items, 32 chunks): 9.5 ->
// 3.5 us per 16-token tile trip, 53.2 -> 22.8 us at 4 tiles per chunk. At
// 1024/1024 on GPUs 4-7, 3 alternating pairs, all G1 PASS, coherent text:
//   decode median  12.157/12.119/12.099 -> 9.863/9.886/9.887 ms
//   decode avg     11.896/11.856/11.861 -> 9.866/9.897/9.890 ms
//   prefill avg    10.378/10.346/10.342 -> 9.284/9.303/9.312 ms/iter
// The per-token staircase (one step per extra trip at seqlen 513/1025/1537)
// went from +0.76 ms per step to +0.29.
#ifndef MPK_MLA_OACC_INTERLEAVE
#define MPK_MLA_OACC_INTERLEAVE 1
#endif
#if MPK_MLA_OACC_INTERLEAVE
constexpr int MLA_OACC_LANE_SHIFT = 2;     // lane stride 4 B
constexpr int MLA_OACC_ELEM_STRIDE = 1024; // float j at j * 256 * 4 B
#else
constexpr int MLA_OACC_LANE_SHIFT = 7;     // lane stride 128 B
constexpr int MLA_OACC_ELEM_STRIDE = 4;
#endif

__device__ __forceinline__ unsigned mla_oacc_lds_base(int tid,
                                                     void const *lds_kv) {
  unsigned base;
  unsigned const kv =
      static_cast<unsigned>(reinterpret_cast<uintptr_t>(lds_kv));
  asm volatile("v_lshlrev_b32 %[b], %c[sh], %[tid]\n\t"
               "v_add_u32_e32 %[b], %[kv], %[b]"
               : [b] "=v"(base)
               : [tid] "v"(tid), [kv] "v"(kv), [sh] "n"(MLA_OACC_LANE_SHIFT));
  return base;
}

template <int KV_BYTES>
__device__ __forceinline__ void mla_oacc_zero_lds(unsigned &base) {
  unsigned z;
  asm volatile("v_mov_b32_e32 %[z], 0\n\t"
               "ds_write_b32 %[base], %[z] offset:%c[o0]\n\t"
               "ds_write_b32 %[base], %[z] offset:%c[o1]\n\t"
               "ds_write_b32 %[base], %[z] offset:%c[o2]\n\t"
               "ds_write_b32 %[base], %[z] offset:%c[o3]\n\t"
               "ds_write_b32 %[base], %[z] offset:%c[o4]\n\t"
               "ds_write_b32 %[base], %[z] offset:%c[o5]\n\t"
               "ds_write_b32 %[base], %[z] offset:%c[o6]\n\t"
               "ds_write_b32 %[base], %[z] offset:%c[o7]\n\t"
               "ds_write_b32 %[base], %[z] offset:%c[o8]\n\t"
               "ds_write_b32 %[base], %[z] offset:%c[o9]\n\t"
               "ds_write_b32 %[base], %[z] offset:%c[o10]\n\t"
               "ds_write_b32 %[base], %[z] offset:%c[o11]\n\t"
               "ds_write_b32 %[base], %[z] offset:%c[o12]\n\t"
               "ds_write_b32 %[base], %[z] offset:%c[o13]\n\t"
               "ds_write_b32 %[base], %[z] offset:%c[o14]\n\t"
               "ds_write_b32 %[base], %[z] offset:%c[o15]\n\t"
               "ds_write_b32 %[base], %[z] offset:%c[o16]\n\t"
               "ds_write_b32 %[base], %[z] offset:%c[o17]\n\t"
               "ds_write_b32 %[base], %[z] offset:%c[o18]\n\t"
               "ds_write_b32 %[base], %[z] offset:%c[o19]\n\t"
               "ds_write_b32 %[base], %[z] offset:%c[o20]\n\t"
               "ds_write_b32 %[base], %[z] offset:%c[o21]\n\t"
               "ds_write_b32 %[base], %[z] offset:%c[o22]\n\t"
               "ds_write_b32 %[base], %[z] offset:%c[o23]\n\t"
               "ds_write_b32 %[base], %[z] offset:%c[o24]\n\t"
               "ds_write_b32 %[base], %[z] offset:%c[o25]\n\t"
               "ds_write_b32 %[base], %[z] offset:%c[o26]\n\t"
               "ds_write_b32 %[base], %[z] offset:%c[o27]\n\t"
               "ds_write_b32 %[base], %[z] offset:%c[o28]\n\t"
               "ds_write_b32 %[base], %[z] offset:%c[o29]\n\t"
               "ds_write_b32 %[base], %[z] offset:%c[o30]\n\t"
               "ds_write_b32 %[base], %[z] offset:%c[o31]\n\t"
               "s_waitcnt lgkmcnt(0)"
               : [z] "=&v"(z), [base] "+v"(base)
               : [o0] "n"(KV_BYTES + 0 * MLA_OACC_ELEM_STRIDE), [o1] "n"(KV_BYTES + 1 * MLA_OACC_ELEM_STRIDE),
                 [o2] "n"(KV_BYTES + 2 * MLA_OACC_ELEM_STRIDE), [o3] "n"(KV_BYTES + 3 * MLA_OACC_ELEM_STRIDE),
                 [o4] "n"(KV_BYTES + 4 * MLA_OACC_ELEM_STRIDE), [o5] "n"(KV_BYTES + 5 * MLA_OACC_ELEM_STRIDE),
                 [o6] "n"(KV_BYTES + 6 * MLA_OACC_ELEM_STRIDE), [o7] "n"(KV_BYTES + 7 * MLA_OACC_ELEM_STRIDE),
                 [o8] "n"(KV_BYTES + 8 * MLA_OACC_ELEM_STRIDE), [o9] "n"(KV_BYTES + 9 * MLA_OACC_ELEM_STRIDE),
                 [o10] "n"(KV_BYTES + 10 * MLA_OACC_ELEM_STRIDE), [o11] "n"(KV_BYTES + 11 * MLA_OACC_ELEM_STRIDE),
                 [o12] "n"(KV_BYTES + 12 * MLA_OACC_ELEM_STRIDE), [o13] "n"(KV_BYTES + 13 * MLA_OACC_ELEM_STRIDE),
                 [o14] "n"(KV_BYTES + 14 * MLA_OACC_ELEM_STRIDE), [o15] "n"(KV_BYTES + 15 * MLA_OACC_ELEM_STRIDE),
                 [o16] "n"(KV_BYTES + 16 * MLA_OACC_ELEM_STRIDE), [o17] "n"(KV_BYTES + 17 * MLA_OACC_ELEM_STRIDE),
                 [o18] "n"(KV_BYTES + 18 * MLA_OACC_ELEM_STRIDE), [o19] "n"(KV_BYTES + 19 * MLA_OACC_ELEM_STRIDE),
                 [o20] "n"(KV_BYTES + 20 * MLA_OACC_ELEM_STRIDE), [o21] "n"(KV_BYTES + 21 * MLA_OACC_ELEM_STRIDE),
                 [o22] "n"(KV_BYTES + 22 * MLA_OACC_ELEM_STRIDE), [o23] "n"(KV_BYTES + 23 * MLA_OACC_ELEM_STRIDE),
                 [o24] "n"(KV_BYTES + 24 * MLA_OACC_ELEM_STRIDE), [o25] "n"(KV_BYTES + 25 * MLA_OACC_ELEM_STRIDE),
                 [o26] "n"(KV_BYTES + 26 * MLA_OACC_ELEM_STRIDE), [o27] "n"(KV_BYTES + 27 * MLA_OACC_ELEM_STRIDE),
                 [o28] "n"(KV_BYTES + 28 * MLA_OACC_ELEM_STRIDE), [o29] "n"(KV_BYTES + 29 * MLA_OACC_ELEM_STRIDE),
                 [o30] "n"(KV_BYTES + 30 * MLA_OACC_ELEM_STRIDE), [o31] "n"(KV_BYTES + 31 * MLA_OACC_ELEM_STRIDE)
               : "memory");
}

#define MLA_DEF_PV_BLOCK(NAME, DELTA)                                          \
  template <int KV_BYTES>                                                      \
  __device__ __forceinline__ void NAME(unsigned &base,                         \
                                       float s,                                \
                                       bf16x4_t va,                            \
                                       bf16x4_t pb) {                          \
    float t0, t1, t2, t3;                                                      \
    u32x2_t avu, bvu;                                                          \
    __builtin_memcpy(&avu, &va, 8);                                            \
    __builtin_memcpy(&bvu, &pb, 8);                                            \
    asm volatile("ds_read_b32 %[t0], %[base] offset:%c[o0]\n\t"                \
                 "ds_read_b32 %[t1], %[base] offset:%c[o1]\n\t"                \
                 "ds_read_b32 %[t2], %[base] offset:%c[o2]\n\t"                \
                 "ds_read_b32 %[t3], %[base] offset:%c[o3]\n\t"                \
                 "s_waitcnt lgkmcnt(0)\n\t"                                    \
                 "v_mul_f32_e32 %[t0], %[s], %[t0]\n\t"                        \
                 "v_mul_f32_e32 %[t1], %[s], %[t1]\n\t"                        \
                 "v_mul_f32_e32 %[t2], %[s], %[t2]\n\t"                        \
                 "v_mul_f32_e32 %[t3], %[s], %[t3]\n\t"                        \
                 "v_accvgpr_write_b32 a0, %[t0]\n\t"                           \
                 "v_accvgpr_write_b32 a1, %[t1]\n\t"                           \
                 "v_accvgpr_write_b32 a2, %[t2]\n\t"                           \
                 "v_accvgpr_write_b32 a3, %[t3]\n\t"                           \
                 "s_nop 1\n\t" /* VALU write -> MFMA read: 2 waits */          \
                 "v_mfma_f32_16x16x16_bf16 a[0:3], %[va], %[pb], a[0:3]\n\t"   \
                 "s_nop 15\n\t"                                                \
                 "s_nop 15\n\t"                                                \
                 "v_accvgpr_read_b32 %[t0], a0\n\t"                            \
                 "v_accvgpr_read_b32 %[t1], a1\n\t"                            \
                 "v_accvgpr_read_b32 %[t2], a2\n\t"                            \
                 "v_accvgpr_read_b32 %[t3], a3\n\t"                            \
                 "ds_write_b32 %[base], %[t0] offset:%c[o0]\n\t"               \
                 "ds_write_b32 %[base], %[t1] offset:%c[o1]\n\t"               \
                 "ds_write_b32 %[base], %[t2] offset:%c[o2]\n\t"               \
                 "ds_write_b32 %[base], %[t3] offset:%c[o3]\n\t"               \
                 "s_waitcnt lgkmcnt(0)"                                        \
                 : [t0] "=&v"(t0), [t1] "=&v"(t1), [t2] "=&v"(t2),             \
                   [t3] "=&v"(t3), [base] "+v"(base)                           \
                 : [s] "v"(s), [va] "v"(avu), [pb] "v"(bvu),                   \
                   [o0] "n"(KV_BYTES + ((DELTA) / 4 + 0) * MLA_OACC_ELEM_STRIDE), \
                   [o1] "n"(KV_BYTES + ((DELTA) / 4 + 1) * MLA_OACC_ELEM_STRIDE), \
                   [o2] "n"(KV_BYTES + ((DELTA) / 4 + 2) * MLA_OACC_ELEM_STRIDE), \
                   [o3] "n"(KV_BYTES + ((DELTA) / 4 + 3) * MLA_OACC_ELEM_STRIDE)  \
                 : "a0", "a1", "a2", "a3", "memory");                          \
  }

MLA_DEF_PV_BLOCK(mla_flash_pv_b0, 0)
MLA_DEF_PV_BLOCK(mla_flash_pv_b1, 16)
MLA_DEF_PV_BLOCK(mla_flash_pv_b2, 32)
MLA_DEF_PV_BLOCK(mla_flash_pv_b3, 48)
MLA_DEF_PV_BLOCK(mla_flash_pv_b4, 64)
MLA_DEF_PV_BLOCK(mla_flash_pv_b5, 80)
MLA_DEF_PV_BLOCK(mla_flash_pv_b6, 96)
MLA_DEF_PV_BLOCK(mla_flash_pv_b7, 112)
#undef MLA_DEF_PV_BLOCK

// The eight PV blocks of one tile trip with o_acc in VGPRs. LLVM's hazard
// recognizer cannot see inside the asm, so it pads both sides: s_nop 1 for
// VALU-written srcA/B/C (2 wait states), s_nop 15 for XDL write -> VALU/VMEM/
// LDS read of the accumulators (11 for an 8-pass MFMA).
__device__ __forceinline__ void mla_pv8_vgpr(__mfma_hd64_fp32x4 (&acc)[8],
                                             u32x2_t const (&va)[8],
                                             u32x2_t pb) {
  asm volatile("s_nop 1\n\t"
               "v_mfma_f32_16x16x16_bf16 %[c0], %[a0], %[pb], %[c0]\n\t"
               "v_mfma_f32_16x16x16_bf16 %[c1], %[a1], %[pb], %[c1]\n\t"
               "v_mfma_f32_16x16x16_bf16 %[c2], %[a2], %[pb], %[c2]\n\t"
               "v_mfma_f32_16x16x16_bf16 %[c3], %[a3], %[pb], %[c3]\n\t"
               "v_mfma_f32_16x16x16_bf16 %[c4], %[a4], %[pb], %[c4]\n\t"
               "v_mfma_f32_16x16x16_bf16 %[c5], %[a5], %[pb], %[c5]\n\t"
               "v_mfma_f32_16x16x16_bf16 %[c6], %[a6], %[pb], %[c6]\n\t"
               "v_mfma_f32_16x16x16_bf16 %[c7], %[a7], %[pb], %[c7]\n\t"
               "s_nop 15"
               : [c0] "+v"(acc[0]), [c1] "+v"(acc[1]), [c2] "+v"(acc[2]),
                 [c3] "+v"(acc[3]), [c4] "+v"(acc[4]), [c5] "+v"(acc[5]),
                 [c6] "+v"(acc[6]), [c7] "+v"(acc[7])
               : [a0] "v"(va[0]), [a1] "v"(va[1]), [a2] "v"(va[2]),
                 [a3] "v"(va[3]), [a4] "v"(va[4]), [a5] "v"(va[5]),
                 [a6] "v"(va[6]), [a7] "v"(va[7]), [pb] "v"(pb));
}

template <int KV_BYTES, int VB>
__device__ __forceinline__ void
    mla_oacc_load_block(unsigned &base, __mfma_hd64_fp32x4 &out) {
  float t0, t1, t2, t3;
  constexpr int off = KV_BYTES + VB * 4 * MLA_OACC_ELEM_STRIDE;
  asm volatile("ds_read_b32 %[t0], %[base] offset:%c[o0]\n\t"
               "ds_read_b32 %[t1], %[base] offset:%c[o1]\n\t"
               "ds_read_b32 %[t2], %[base] offset:%c[o2]\n\t"
               "ds_read_b32 %[t3], %[base] offset:%c[o3]\n\t"
               "s_waitcnt lgkmcnt(0)"
               : [t0] "=v"(t0), [t1] "=v"(t1), [t2] "=v"(t2), [t3] "=v"(t3),
                 [base] "+v"(base)
               : [o0] "n"(off + 0 * MLA_OACC_ELEM_STRIDE),
                 [o1] "n"(off + 1 * MLA_OACC_ELEM_STRIDE),
                 [o2] "n"(off + 2 * MLA_OACC_ELEM_STRIDE),
                 [o3] "n"(off + 3 * MLA_OACC_ELEM_STRIDE)
               : "memory");
  out[0] = t0;
  out[1] = t1;
  out[2] = t2;
  out[3] = t3;
}

template <int KV_BYTES>
__device__ __forceinline__ void
    mla_oacc_from_lds(unsigned &base, __mfma_hd64_fp32x4 *o_acc) {
  mla_oacc_load_block<KV_BYTES, 0>(base, o_acc[0]);
  mla_oacc_load_block<KV_BYTES, 1>(base, o_acc[1]);
  mla_oacc_load_block<KV_BYTES, 2>(base, o_acc[2]);
  mla_oacc_load_block<KV_BYTES, 3>(base, o_acc[3]);
  mla_oacc_load_block<KV_BYTES, 4>(base, o_acc[4]);
  mla_oacc_load_block<KV_BYTES, 5>(base, o_acc[5]);
  mla_oacc_load_block<KV_BYTES, 6>(base, o_acc[6]);
  mla_oacc_load_block<KV_BYTES, 7>(base, o_acc[7]);
}

// MPK_DEC_EXP2_BUILTIN: the softmax's v_exp_f32 through the builtin instead
// of __fast_exp2_hd64's asm, which LLVM cannot speculate: the rescale's
// `(m_running == -inf) ? 0 : exp2(..)` is an exec-mask branch around it in
// the final ISA, and a v_cndmask with the builtin. Same instruction on the
// same operands: bit-exact (GPT-OSS 28b19f3).
__device__ __forceinline__ float mla_exp2(float x) {
#if MPK_DEC_EXP2_BUILTIN
  return __builtin_amdgcn_exp2f(x);
#else
  return __fast_exp2_hd64(x);
#endif
}

// The bf16x4 loads from paged_attention_decode_minimal_hd64, re-expressed so
// the load itself goes through ld_g. The convert half is reused verbatim.
__device__ __forceinline__ void __ldg_bf16x4_raw(uint2 *__restrict__ dst,
                                                 void const *__restrict__ src) {
  u32x2_t const v = ld_g<u32x2_t>(src);
  dst->x = v.x;
  dst->y = v.y;
}
#if MPK_DEC_VT
// V^T of one swizzled KV tile (MPK_MLA_KV_SWZ: 16-byte chunk c of row r at
// c ^ (r >> 1)): vt[d * MPK_DEC_VTS + r] = tile[r][d] for the 512 latent
// dims. Wave w, round j takes token quad (w * 8 + j) & 3 and dims
// ((w * 8 + j) >> 2) * 64 + lane: each of its four u16 reads is one row's 64
// consecutive dims (one 128 B swizzle group, conflict-free), and each lane
// stores its dim's four tokens as one b64.
template <int QK_DIM>
__device__ __forceinline__ void mla_vt_transpose(__bf16 const *tile,
                                                 __bf16 *vt) {
  int const lane = threadIdx.x & 63;
  int const w = threadIdx.x >> 6;
#pragma unroll
  for (int j = 0; j < 8; j++) {
    int const k = w * 8 + j;
    int const tg = k & 3;
    int const d = (k >> 2) * 64 + lane;
    unsigned short u[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int const r = tg * 4 + i;
      int const idx = r * QK_DIM + ((((d >> 3) ^ (r >> 1)) << 3) | (d & 7));
      u[i] = reinterpret_cast<unsigned short const *>(tile)[idx];
    }
    unsigned long long const v =
        (unsigned long long)u[0] | ((unsigned long long)u[1] << 16) |
        ((unsigned long long)u[2] << 32) | ((unsigned long long)u[3] << 48);
    *reinterpret_cast<unsigned long long *>(vt + d * MPK_DEC_VTS + tg * 4) = v;
  }
}
#endif
} // namespace gang_mla_decode_detail

template <typename T,
          int NUM_Q_HEADS,
          int KV_LORA_RANK,
          int QK_ROPE_HEAD_DIM,
          int PAGE_SIZE,
          int MAX_SEQ_LEN,
          int NUM_KV_CHUNKS,
          int Q_WORKSPACE_STRIDE,
          int KV_CACHE_STRIDE,
          bool WRITE_THROUGH = false,
          // Compile-time width of the query-row dimension. 1 is the decode
          // shape and keeps the codegen below bit-identical to what it was;
          // > 1 is MTP verify, where one request carries several query rows
          // over one page list. See the token_idx block in the body.
          int BATCH_SIZE = 1>
// THIS function sets the whole megakernel's register allocation, and two
// obvious ways to fix that are closed. Recorded so neither gets retried.
//
// Under -fgpu-rdc every phase is its own __noinline__ ELF symbol (125
// s_swappc_b64 in the image), so a kernel's .vgpr_count is a call-graph
// quantity, not its own body's pressure (worker_kernel's body is 226; it
// reports 284). At the default budget the 284 is 248 arch + 36 acc and BOTH
// terms are set here -- 248 arch ties with three other functions, and the 36
// AGPRs are this function's alone.
//
// CLOSED 1: a budget on this function. __attribute__((amdgpu_waves_per_eu))
// does not compile on a device function -- clang restricts it to kernels
// ("'amdgpu_waves_per_eu' attribute only applies to kernel functions") and
// LLVM 20.0.0git as shipped in ROCm 7.0 has no llc equivalent and no
// "amdgpu-agpr-alloc". A non-kernel function with no waves-per-eu attribute
// is allocated against the full 512-entry unified file, which is what this
// one does.
//
// CLOSED 2: inlining it into a kernel that CAN be budgeted. Dropping
// __noinline__ here and on gang_mla_full_layer_fused_kernel_mi300 does not
// inline anything -- at -O2 the inliner declines a ~950-instruction callee
// and both symbols survive with .vgpr_count unmoved at 284.
//
// What actually worked is MPK_WORKER_WAVES_PER_EU in persistent_kernel.cuh:
// budget both KERNELS and let LLVM's AMDGPUAttributor push the constraint
// down the call graph. Long note there.
//
// Do not read this function's 240 scratch ops as pressure -- they are the ABI
// callee-save prologue/epilogue (v40-v143, 60 dwords in at instr 1-60, back
// out at 880-939). The body spills nothing at 284. Its live set is genuine:
// arch peaks at 248 in the same window where 24-36 AGPRs are live, so moving
// the MFMA accumulators to arch VGPRs trades 36 acc for 36 arch and loses.
//
// That prologue/epilogue is not free, though: forcing the GLM fused layer
// always_inline (MPK_FL_INLINE) deleted its 60-VGPR one and measured -0.18
// ms. MPK_DECODE_INLINE forces the same here, which plain removal of
// __noinline__ did not (CLOSED 2 above).
#if MPK_DECODE_INLINE
__device__ __attribute__((always_inline)) void
#else
__device__ __noinline__ void
#endif
    mla_decode_absorbed(void const *q_workspace_ptr,
                        void const *paged_kv_cache_ptr,
                        void *output_ptr,
                        void *lse_ptr,
                        int const *qo_indptr,
                        int const *kv_indptr,
                        int const *kv_indices,
                        int const *kv_last_page_len,
                        int16_t request_id,
                        int token_idx,
                        int q_head_group,
                        int kv_chunk_idx,
                        float scale_s,
                        // Heads of the group whose partials are stored; the
                        // rest are computed (the MFMA is 16 rows either way)
                        // and dropped. See MPK_MLA_MERGE_OWN_HEADS.
                        int out_head_lo = 0,
                        int out_heads = 16,
                        // The chunk's first `tiles_in_lds` tiles are already
                        // in LDS, full, tile t at lds_kv + t * KV_LDS_BYTES
                        // (MPK_DEC_PRELOAD); those trips skip their load.
                        int tiles_in_lds = 0,
                        // MPK_DEC_HINTS, one row: qo_indptr[req] and the
                        // request's (first page, seqlen), as the caller read
                        // them ahead of its barrier. -1 loads them here.
                        int hint_q_row = -1,
                        int hint_first_page = -1,
                        int hint_seqlen = -1,
                        // MPK_DEC_MERGE_LL: the partials leave as epoch words
                        // -- o as (ll_epoch << 32 | bf16x2) at half the f32
                        // index, lse as (ll_epoch << 32 | f32) at its index --
                        // so the merge validates them instead of waiting.
                        unsigned long long *ll_o = nullptr,
                        unsigned long long *ll_lse = nullptr,
                        unsigned ll_epoch = 0,
                        // MPK_DEC_Q_LL, one row: the own heads' query rows as
                        // (q_ll_epoch << 32 | bf16x2) words, head h at
                        // q_ll + h * QK_DIM / 2 for q heads q_ll_own_lo + h,
                        // h < 8, staged through LDS instead of read from
                        // q_workspace behind the q_b barrier; kv_flag carries
                        // q_ll_epoch once the current token's cache row is
                        // stored, and the chunk holding that row waits on it.
                        unsigned long long const *q_ll = nullptr,
                        unsigned q_ll_epoch = 0,
                        int q_ll_own_lo = 0,
                        unsigned long long const *kv_flag = nullptr) {
  using bf16 = __hip_bfloat16;
  using gang_mla_decode_detail::__ldg_bf16x4_raw;
  using gang_mla_decode_detail::bf16x4_t;
  using gang_mla_decode_detail::ld_g;
  using gang_mla_decode_detail::mla_flash_pv_b0;
  using gang_mla_decode_detail::mla_flash_pv_b1;
  using gang_mla_decode_detail::mla_flash_pv_b2;
  using gang_mla_decode_detail::mla_flash_pv_b3;
  using gang_mla_decode_detail::mla_flash_pv_b4;
  using gang_mla_decode_detail::mla_flash_pv_b5;
  using gang_mla_decode_detail::mla_flash_pv_b6;
  using gang_mla_decode_detail::mla_flash_pv_b7;
  using gang_mla_decode_detail::mla_oacc_from_lds;
  using gang_mla_decode_detail::mla_oacc_lds_base;
  using gang_mla_decode_detail::mla_oacc_zero_lds;
  using gang_mla_decode_detail::mla_qk_acc32_mfma;
  using gang_mla_decode_detail::mla_qk_acc32_read;
  using gang_mla_decode_detail::mla_qk_acc32_zero;
  using gang_mla_decode_detail::u32x2_t;
  using gang_mla_decode_detail::u32x4_t;

  constexpr int QK_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM;
  constexpr int Q_HEADS_PER_GROUP = 16; // MFMA M tile
  constexpr int NUM_Q_GROUPS = NUM_Q_HEADS / Q_HEADS_PER_GROUP;
  constexpr int KV_TILE = 16;
  constexpr int NUM_K32 = QK_DIM / 32;            // QK MFMA steps
  constexpr int NUM_V_BLOCKS = KV_LORA_RANK / 64; // PV MFMA steps
  constexpr int LDG_PER_TILE = QK_DIM / 64;       // 4-elt loads per thread

  static_assert(NUM_Q_HEADS % Q_HEADS_PER_GROUP == 0,
                "NUM_Q_HEADS must be a multiple of the MFMA M tile (16)");
  static_assert(KV_LORA_RANK % 64 == 0,
                "KV_LORA_RANK must be a multiple of 64 (4 warps x 16 dims)");
  static_assert(KV_LORA_RANK / 64 == gang_mla_decode_detail::MLA_OACC_V_BLOCKS,
                "LDS o_acc spill is 8 PV blocks x 4");
  static_assert(QK_DIM % 64 == 0,
                "QK_DIM must be a multiple of 64 so the LDS tile loads evenly");
  static_assert(KV_CACHE_STRIDE >= QK_DIM,
                "KV cache rows must hold the latent + rope dims");
  constexpr int KV_LDS_BYTES = KV_TILE * QK_DIM * (int)sizeof(__bf16);
  static_assert(KV_LDS_BYTES +
                        (gang_mla_decode_detail::MLA_OACC_FLOATS - 1) *
                            gang_mla_decode_detail::MLA_OACC_ELEM_STRIDE +
                        4 <=
                    65536,
                "o_acc ds offsets must fit in the 16-bit ds offset field");

  int const req = request_id;
  bool const hinted = BATCH_SIZE == 1 && hint_q_row >= 0;
  int const query_start = hinted ? hint_q_row : ld_g<int>(&qo_indptr[req]);
  int const query_end =
      hinted ? hint_q_row + 1 : ld_g<int>(&qo_indptr[req + 1]);
  if (query_start == query_end) {
    return;
  }

  // ── which query row, and how much of the cache it may see ──────────────
  //
  // BATCH_SIZE is the compile-time width of the row dimension; num_tokens is
  // how many of those rows hold a live token this step. MTP verify runs two --
  // the accepted token and the draft -- appended to the SAME request's page
  // list, so row r sits at sequence position (seqlen - num_tokens + r) and
  // must not attend to the rows after it. Shortening seqlen_k by the suffix
  // this row may not see IS the causal mask: the per-tile
  // `kgrp * 4 + h >= tile_len` clamp further down then falls out of the
  // shortened effective_len for free, so there is no second mask to write and
  // no branch in the MFMA loop.
  //
  // At BATCH_SIZE 1 token_idx is a literal 0, this block compiles to nothing,
  // and q_row / seqlen_k are exactly the query_start and full length they
  // always were.
  int q_row = query_start;
  int tok_back = 0;
  if constexpr (BATCH_SIZE > 1) {
    int const num_tokens = query_end - query_start;
    if (token_idx >= num_tokens) {
      // A row the graph is wide enough for that this step does not fill.
      // Return without stamping LSE: the merge bounds its own token loop by
      // the same num_tokens, so nothing ever reads this row's partials.
      return;
    }
    q_row = query_start + token_idx;
    tok_back = num_tokens - 1 - token_idx;
  }

  int first_page;
  int seqlen_k;
  if (hinted) {
    first_page = hint_first_page;
    seqlen_k = hint_seqlen;
  } else {
    first_page = ld_g<int>(&kv_indptr[req]);
    int const num_pages = ld_g<int>(&kv_indptr[req + 1]) - first_page;
    seqlen_k = (num_pages - 1) * PAGE_SIZE +
               ld_g<int>(&kv_last_page_len[req]) - tok_back;
  }

  int const tid = threadIdx.x;
  int const warp_id = tid / 64;
  int const lane = tid & 63;
  int const midx = lane & 15;
  int const kgrp = lane >> 4;

  char const *kv_base = reinterpret_cast<char const *>(paged_kv_cache_ptr);

  // LDS: KV[KV_TILE][QK_DIM] bf16 (18 KB for GLM-5), then per-thread o_acc
  // (32 floats x 256 threads = 32 KB) at +KV_TILE*QK_DIM*2, addressed with
  // one VGPR base and immediate ds offsets.
  extern __shared__ char _mla_decode_smem[];
  __bf16 *lds_kv = reinterpret_cast<__bf16 *>(_mla_decode_smem);
  unsigned oacc_base = mla_oacc_lds_base(tid, lds_kv);

  // 256 threads x 4 elements = 16 tok x 64 dim per round, LDG_PER_TILE rounds
  int const my_tok = tid / 16;
  int const my_dim0 = (tid % 16) * 4;
#if MPK_MLA_KV_SWZ
  // 16-byte chunk c of tile row r sits at c ^ (r >> 1). Per lane that is one
  // register-path write column, two QK bases (even / odd k-chunk) and two PV
  // bases (rows kgrp*4 + {0,1} / {2,3}), each taking immediate offsets.
  int const my_wcol = (((my_dim0 >> 3) ^ (my_tok >> 1)) << 3) | (my_dim0 & 7);
  int const kswz_f = midx >> 1;
  int const kswz_lane = midx * QK_DIM + ((kgrp ^ (kswz_f & 3)) << 3);
  int const kswz_e = kswz_lane + ((kswz_f >> 2) << 5);
  int const kswz_o = kswz_lane - ((kswz_f >> 2) << 5);
  int const vswz_h = (2 * warp_id + (midx >> 3)) ^ (2 * kgrp);
  int const vswz_b0 = kgrp * 4 * QK_DIM + (vswz_h << 3) + (midx & 7);
  int const vswz_b2 =
      kgrp * 4 * QK_DIM + 2 * QK_DIM + ((vswz_h ^ 1) << 3) + (midx & 7);
#endif

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  // Slot 2 is free on the MXFP8 fused path (its only other user is
  // gang_linear_mxfp4_res_bias, which that path never calls). Phases:
  //   2 = chunk partition      3 = cold start: Q + KV tiles 0/1 + LDS write
  //   4 = per-tile QK/softmax/PV (incl. the entry __syncthreads)
  //   5 = per-tile refill: __syncthreads + vmcnt drain + LDS write + prefetch
  //   6 = epilogue      (the count goes to g_subphase_cnt[2])
  unsigned long long _d_t0 = __builtin_amdgcn_s_memrealtime();
  unsigned long long _d_compute = 0, _d_refill = 0;
#endif

  int kv_start = 0;
  int effective_len = seqlen_k;
  int ntiles = (effective_len + KV_TILE - 1) / KV_TILE;
  // The sequence's tile count; the split below narrows ntiles to the chunk's.
  int const ntiles_seq = ntiles;

  // Split-KV partitioning: each chunk takes a contiguous slice of tiles.
  // For NUM_KV_CHUNKS==1 this is a no-op. An empty chunk stamps LSE=-inf and
  // exits so the merge gives it zero weight instead of picking up stale data.
  int chunk_first_tile = 0;
  int chunk_last_tile = ntiles;
  if constexpr (NUM_KV_CHUNKS > 1) {
    int tiles_per_chunk = (ntiles + NUM_KV_CHUNKS - 1) / NUM_KV_CHUNKS;
    chunk_first_tile = kv_chunk_idx * tiles_per_chunk;
    chunk_last_tile = chunk_first_tile + tiles_per_chunk;
    if (chunk_last_tile > ntiles) {
      chunk_last_tile = ntiles;
    }
    if (chunk_first_tile >= ntiles) {
      // MPK_DEC_MERGE_LL32: the merge derives the live chunks from the
      // sequence length and reads no word of an empty one.
      if (MPK_DEC_MERGE_LL32 && ll_lse != nullptr) {
        return;
      }
      if (warp_id == 0 && kgrp == 0) {
        constexpr int LSE_STRIDE =
            NUM_Q_GROUPS * NUM_KV_CHUNKS * Q_HEADS_PER_GROUP;
        float *lse_out = reinterpret_cast<float *>(lse_ptr) +
                         static_cast<long>(q_row) * LSE_STRIDE +
                         q_head_group * NUM_KV_CHUNKS * Q_HEADS_PER_GROUP +
                         kv_chunk_idx * Q_HEADS_PER_GROUP + midx;
        if (ll_lse != nullptr) {
          float const empty = -1e30f;
          st_wt_u64((void *)(ll_lse + (lse_out - reinterpret_cast<float *>(
                                                     lse_ptr))),
                    ((unsigned long long)ll_epoch << 32) |
                        __float_as_uint(empty));
        }
        if (ll_lse != nullptr && MPK_DML_DBG != 1) {
        } else if constexpr (WRITE_THROUGH) {
          float const empty = -1e30f;
          unsigned raw;
          __builtin_memcpy(&raw, &empty, 4);
          st_wt_u32((void *)lse_out, raw);
        } else {
          *lse_out = -1e30f;
        }
      }
      return;
    }
    kv_start += chunk_first_tile * KV_TILE;
    effective_len = (chunk_last_tile - chunk_first_tile) * KV_TILE;
    int remaining = seqlen_k - kv_start;
    if (effective_len > remaining) {
      effective_len = remaining;
    }
    // Recompute from the clamped length. Using (last-first) leaves a phantom
    // extra tile when the tail chunk is shorter than tiles_per_chunk*KV_TILE
    // -- tile_len goes negative and the softmax sees leftover LDS.
    ntiles = (effective_len + KV_TILE - 1) / KV_TILE;
  }
  if (ntiles == 0) {
    return;
  }
  // Whether this chunk holds the current token's cache row, the one XCD 0's
  // latent tile appends and flags. MPK_DEC_KVF_LAST: against the sequence's
  // tile count -- ntiles is the chunk's by now, which every chunk's last tile
  // index reaches, so every decode item waited on the flag.
  auto holds_cur = [&]() {
    return chunk_last_tile >= (MPK_DEC_KVF_LAST ? ntiles_seq : ntiles);
  };

  // Byte offset of a token's latent row (bf16 = 2 bytes per element).
  //
  // Reload kv_indices every tile. Caching (page, base) across QK/PV is
  // unsafe under fused VGPR pressure: LLVM may reuse those registers, and
  // a later get_kv_row with page unchanged would skip the reload. The
  // seq>256 collapse is a different fused-loop hazard (o_acc rescale
  // copies vs next-tile KV fills); this just keeps the address path honest.
  auto get_kv_row = [&](int global_tok) -> long {
    int const page = global_tok / PAGE_SIZE;
    // volatile + memory clobber: a plain ld_g is hoistable when every token
    // in the chunk sits on page 0 (PAGE_SIZE=4096, seq<=512), which puts pid
    // back in a VGPR that QK/PV then reuse.
    asm volatile("" ::: "memory");
    int const pid =
        *(__attribute__((address_space(1))) int const volatile
              *)static_cast<int const *>(&kv_indices[first_page + page]);
    long row = static_cast<long>(pid) * PAGE_SIZE * KV_CACHE_STRIDE * 2 +
               static_cast<long>(global_tok % PAGE_SIZE) * KV_CACHE_STRIDE * 2;
    asm volatile("" : "+v"(row) : : "memory");
    return row;
  };

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  unsigned long long _d_t1 = __builtin_amdgcn_s_memrealtime();
#endif
  // Trace stamps 46..50 split the decode item: indptrs read and chunk
  // partitioned, Q issued, first KV tile in LDS, loop done, output issued.
  if (tid == 0) {
    mpk_stage_stamp(46);
  }

#if MPK_DEC_VT
  // MPK_DEC_VT: V^T of the tiles a caller left in LDS (MPK_DEC_PRELOAD), ahead
  // of the query poll -- the span the query takes to land. Their loads are
  // retired here if the caller has not already (the fused caller has).
  __bf16 *const vt_lds =
      reinterpret_cast<__bf16 *>(_mla_decode_smem + MPK_DEC_VT_LDS_OFF);
  // Two-tile chunks only (every chunk at 1025..2048 tokens): at three or
  // four, MPK_DEC_WPT's register-path buffers sit where V^T does.
  int const nvt = (ntiles == 2) ? (tiles_in_lds < 2 ? tiles_in_lds : 2) : 0;
  if (nvt > 0) {
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
    __builtin_amdgcn_s_barrier();
    for (int t = 0; t < nvt; t++) {
      gang_mla_decode_detail::mla_vt_transpose<QK_DIM>(
          lds_kv + t * (KV_TILE * QK_DIM), vt_lds + t * MPK_DEC_VT_TILE);
    }
  }
#endif

  // Q first, then one KV tile per loop trip from HBM.
#if MPK_DEC_QK_KSPLIT
  // Four waves, as every per-lane index in this kernel already assumes.
  constexpr int QKS_J = (NUM_K32 + 3) / 4;
  int const qks_w = __builtin_amdgcn_readfirstlane(warp_id);
  __bf16 qr[QKS_J][8];
#else
  __bf16 qr[NUM_K32][8];
#endif
  if (BATCH_SIZE == 1 && q_ll != nullptr) {
    constexpr int Q_LL_HEADS = 8;
    constexpr int Q_WORDS = Q_LL_HEADS * QK_DIM / 2;
    static_assert(Q_WORDS % 256 == 0 && QK_DIM % 2 == 0,
                  "the staged rows split evenly over 256 threads");
    constexpr int Q_PER = Q_WORDS / 256;
    constexpr int Q_LDS_OFF = 96 * 1024;
    static_assert(Q_LDS_OFF >= 65536 && Q_LDS_OFF + Q_WORDS * 4 <= 140 * 1024,
                  "past the decode's tiles and o_acc, short of MPK_ATTN_META_PF's "
                  "never-read window at 140 KB");
    unsigned *const q_lds =
        reinterpret_cast<unsigned *>(_mla_decode_smem + Q_LDS_OFF);
    unsigned long long w[Q_PER];
#if MPK_DEC_KVF_EARLY
    // MPK_DEC_KVF_EARLY: thread 0 reads the cache-row flag in the same round
    // as the query words, so the flag costs no round trip of its own after
    // them; it was set long before (S66 ~14.7 us vs S67 ~19 us).
    bool kvf_seen = !(kv_flag != nullptr && holds_cur());
#endif
#if MPK_DEC_Q_INC && !MPK_DEC_KVF_EARLY
    // MPK_DEC_Q_INC: each thread re-reads only its words still on an older
    // epoch and spins on its own; the barrier after the cache-row flag below
    // orders the LDS writes for every reader, as before.
    {
      static_assert(Q_PER <= 32, "one valid bit per staged word");
      constexpr unsigned ALL_Q = (Q_PER == 32) ? ~0u : ((1u << Q_PER) - 1u);
      unsigned vq = 0u;
#if MPK_DEC_KVF_INC
      // MPK_DEC_KVF_INC: the chunk holding the current row reads the
      // cache-row flag in these rounds (thread 0), not in a round trip of its
      // own after them.
      bool kvf_ok = !(MPK_PROBE_DEC_NOQ != 1 && kv_flag != nullptr &&
                      tid == 0 && holds_cur());
#endif
      // Trace stamps 63/53 (thread 0): last round issued, first round back.
      bool first = true;
      while (true) {
        if (tid == 0) {
          mpk_stage_stamp(63);
        }
#pragma unroll
        for (int j = 0; j < Q_PER; j++) {
          if (!(vq & (1u << j))) {
            w[j] = __hip_atomic_load(q_ll + tid + 256 * j, __ATOMIC_RELAXED,
                                     __HIP_MEMORY_SCOPE_SYSTEM);
          }
        }
#if defined(MPK_TRACE) && MPK_TRACE
        if (tid == 0 && first) {
          asm volatile("" ::"v"(w[0]), "v"(w[Q_PER - 1]) : "memory");
          mpk_stage_stamp(53);
        }
#endif
        first = false;
#if MPK_DEC_KVF_INC
        unsigned long long kvf = 0;
        if (!kvf_ok) {
          kvf = __hip_atomic_load(kv_flag, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_SYSTEM);
        }
#endif
#pragma unroll
        for (int j = 0; j < Q_PER; j++) {
          if ((unsigned)(w[j] >> 32) == q_ll_epoch) {
            vq |= 1u << j;
          }
        }
#if MPK_DEC_KVF_INC
        if (!kvf_ok) {
          kvf_ok = (unsigned)(kvf >> 32) == q_ll_epoch;
        }
        if ((vq == ALL_Q || MPK_PROBE_DEC_NOQ) && kvf_ok) {
          break;
        }
#else
        if (vq == ALL_Q || MPK_PROBE_DEC_NOQ) {
          break;
        }
#endif
        __builtin_amdgcn_s_sleep(1);
      }
    }
#else
    while (true) {
      // Compiler-visible loads: separate asm loads ahead of one vmcnt(0)
      // leave the outputs free for the allocator to move before they land
      // (see merge_splitkv_ck_fmha's MPK_DEC_MERGE_LL read).
#pragma unroll
      for (int j = 0; j < Q_PER; j++) {
        w[j] = __hip_atomic_load(q_ll + tid + 256 * j, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_SYSTEM);
      }
#if MPK_DEC_KVF_EARLY
      unsigned long long kvf = 0;
      if (tid == 0 && !kvf_seen) {
        kvf = __hip_atomic_load(kv_flag, __ATOMIC_RELAXED,
                                __HIP_MEMORY_SCOPE_SYSTEM);
      }
#endif
      bool ok = true;
#pragma unroll
      for (int j = 0; j < Q_PER; j++) {
        ok = ok && (unsigned)(w[j] >> 32) == q_ll_epoch;
      }
#if MPK_DEC_KVF_EARLY
      if (tid == 0 && !kvf_seen) {
        kvf_seen = (unsigned)(kvf >> 32) == q_ll_epoch;
        ok = ok && kvf_seen;
      }
#endif
      if (__syncthreads_and(ok)) {
        break;
      }
      __builtin_amdgcn_s_sleep(1);
    }
#endif
#pragma unroll
    for (int j = 0; j < Q_PER; j++) {
      q_lds[tid + 256 * j] = (unsigned)w[j];
    }
    // Trace stamp 67: the query words validated, before the cache-row flag.
    if (tid == 0) {
      mpk_stage_stamp(67);
    }
#if !MPK_DEC_KVF_EARLY && !(MPK_DEC_KVF_INC && MPK_DEC_Q_INC)
    if (MPK_PROBE_DEC_NOQ != 1 && kv_flag != nullptr && tid == 0 &&
        holds_cur()) {
      while (true) {
        unsigned long long f;
        asm volatile("global_load_dwordx2 %0, %1, off sc0 sc1\n"
                     "s_waitcnt vmcnt(0)"
                     : "=v"(f)
                     : "v"(kv_flag)
                     : "memory");
        if ((unsigned)(f >> 32) == q_ll_epoch) {
          break;
        }
        __builtin_amdgcn_s_sleep(1);
      }
    }
#endif
    __syncthreads();
    asm volatile("buffer_inv" ::: "memory");
    int const own = midx - q_ll_own_lo;
    bool const mine = own >= 0 && own < Q_LL_HEADS;
    unsigned short const *const q_row_lds =
        reinterpret_cast<unsigned short const *>(q_lds) +
        (mine ? own : 0) * QK_DIM;
#if MPK_DEC_QK_KSPLIT
    // This wave's k32 steps only: qr[j] is step qks_w + 4 j.
#pragma unroll
    for (int j = 0; j < QKS_J; j++) {
      int const kc = qks_w + 4 * j;
      u32x4_t raw = {0u, 0u, 0u, 0u};
      if (mine && kc < NUM_K32) {
        raw = *reinterpret_cast<u32x4_t const *>(q_row_lds + kc * 32 +
                                                 kgrp * 8);
      }
      __builtin_memcpy(&qr[j][0], &raw, 16);
    }
#else
#pragma unroll
    for (int kc = 0; kc < NUM_K32; kc++) {
      int dim_off = kc * 32 + kgrp * 8;
      u32x4_t raw = {0u, 0u, 0u, 0u};
      if (mine) {
        raw = *reinterpret_cast<u32x4_t const *>(q_row_lds + dim_off);
      }
      __builtin_memcpy(&qr[kc][0], &raw, 16);
    }
#endif
  } else {
    int const q_head = q_head_group * Q_HEADS_PER_GROUP + midx;
    char const *q_ptr = reinterpret_cast<char const *>(q_workspace_ptr) +
                        (static_cast<long>(q_row) * Q_WORKSPACE_STRIDE +
                         static_cast<long>(q_head) * QK_DIM) *
                            2;
#if MPK_DEC_QK_KSPLIT
#pragma unroll
    for (int j = 0; j < QKS_J; j++) {
      int const kc = qks_w + 4 * j;
      u32x4_t raw = {0u, 0u, 0u, 0u};
      if (kc < NUM_K32) {
        raw = ld_g<u32x4_t>(q_ptr + (kc * 32 + kgrp * 8) * 2);
      }
      __builtin_memcpy(&qr[j][0], &raw, 16);
    }
#else
#pragma unroll
    for (int kc = 0; kc < NUM_K32; kc++) {
      int dim_off = kc * 32 + kgrp * 8;
      u32x4_t const raw = ld_g<u32x4_t>(q_ptr + dim_off * 2);
      __builtin_memcpy(&qr[kc][0], &raw, 16);
    }
#endif
  }

  // ===== MAIN LOOP =====
#if MPK_MLA_OACC_VGPR
  __mfma_hd64_fp32x4 o_acc_r[NUM_V_BLOCKS];
#pragma unroll
  for (int vb = 0; vb < NUM_V_BLOCKS; vb++) {
    o_acc_r[vb] = __mfma_hd64_fp32x4{0.0f, 0.0f, 0.0f, 0.0f};
  }
#else
  mla_oacc_zero_lds<KV_LDS_BYTES>(oacc_base);
#endif
  float m_running = -INFINITY;
  float l_head[4] = {0, 0, 0, 0};
#if MPK_MLA_QK_SPLIT
  __mfma_hd64_fp32x4 qks_next = {0.0f, 0.0f, 0.0f, 0.0f};
  bool qks_have_next = false;
#endif
#if MPK_DEC_QK2T
  __mfma_hd64_fp32x4 qk2_next = {0.0f, 0.0f, 0.0f, 0.0f};
  bool qk2_have = false;
#endif

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  unsigned long long _d_t2 = __builtin_amdgcn_s_memrealtime();
#endif
  if (tid == 0) {
    mpk_stage_stamp(47);
  }

#if MPK_MLA_DECODE_DMA_PF
  static_assert(PAGE_SIZE % KV_TILE == 0,
                "a DMA'd KV tile must not straddle a page");
  static_assert(KV_CACHE_STRIDE == QK_DIM,
                "DMA copies cache rows into the LDS tile verbatim");
  static_assert(KV_LDS_BYTES % 1024 == 0,
                "an LDS-DMA round is 1 KB per wave");
  constexpr int DMA_TAIL_WAVES = (KV_LDS_BYTES % 4096) / 1024;
  static_assert(KV_LDS_BYTES / 4096 == 4 && DMA_TAIL_WAVES == 2,
                "mla_dma_tile is written for an 18 KB tile (GLM's 576 dims)");
  // Second tile buffer, past the o_acc spill so the default layout (and the
  // o_acc ds offsets) are untouched.
  __bf16 *const lds_kv_b = reinterpret_cast<__bf16 *>(
      _mla_decode_smem + KV_LDS_BYTES +
      256 * gang_mla_decode_detail::MLA_OACC_FLOATS * (int)sizeof(float));
  bool dma_have = false; // tile t was DMA'd into its buffer by trip t - 1
#endif

#if MPK_DEC_WPT
  // MPK_DEC_WPT: the chunk's 2..4 tiles side by side, one per wave, instead
  // of one trip per tile on all four waves. Every tile gets its own buffer
  // (tile t at lds_kv + t * tile; the preloaded ones are already there, the
  // rest go in on the register path), wave t runs tile t's QK and its tile
  // softmax, and parks the tile max and its f32 weights in the dead Q block;
  // then every wave rescales all tiles' weights against the chunk max and runs
  // its own 128 dims of PV over all of them. One exchange barrier replaces the
  // trips' two barriers each, and the per-trip o_acc rescale goes away.
  bool const wpt = ntiles >= 2 && ntiles <= 4;
  if (wpt) {
    static_assert(MPK_MLA_OACC_VGPR && MPK_MLA_KV_SWZ && MPK_MLA_QK_PIPE2,
                  "MPK_DEC_WPT is written for the default tile loop");
#pragma unroll 1
    for (int t = tiles_in_lds; t < ntiles; t++) {
      __bf16 *const tkv = lds_kv + t * (KV_TILE * QK_DIM);
      int const tstart = t * KV_TILE;
      int tlen = effective_len - tstart;
      tlen = tlen > KV_TILE ? KV_TILE : (tlen < 0 ? 0 : tlen);
      if (my_tok < tlen) {
        long const row = get_kv_row(kv_start + tstart + my_tok);
#pragma unroll
        for (int r = 0; r < LDG_PER_TILE; r++) {
          u32x2_t const v =
              ld_g<u32x2_t>(kv_base + row + (my_dim0 + r * 64) * 2);
          uint64_t val;
          __builtin_memcpy(&val, &v, 8);
          *(__attribute__((address_space(3))) uint64_t *)&tkv
              [my_tok * QK_DIM + my_wcol + r * 64] = val;
        }
      } else {
#pragma unroll
        for (int r = 0; r < LDG_PER_TILE; r++) {
          *(__attribute__((address_space(3))) uint64_t *)&tkv
              [my_tok * QK_DIM + my_wcol + r * 64] = 0ull;
        }
      }
    }
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    mla_tile_bar();
    // MPK_DEC_VT: the prologue's V^T stores are ordered by the barrier above.
    if (tid == 0) {
      mpk_stage_stamp(48);
    }
    // The Q block is dead: every wave's qr reads retired ahead of the
    // barrier above.
    float *const wpt_p = reinterpret_cast<float *>(_mla_decode_smem + 96 * 1024);
    float *const wpt_m = wpt_p + 4 * 256;
    int const wv = __builtin_amdgcn_readfirstlane(warp_id);
    if (wv < ntiles) {
      __bf16 const *const tkv = lds_kv + wv * (KV_TILE * QK_DIM);
      int tlen = effective_len - wv * KV_TILE;
      tlen = tlen > KV_TILE ? KV_TILE : (tlen < 0 ? 0 : tlen);
      mla_qk_acc32_zero();
      {
        using kq_t = gang_mla_decode_detail::bf16x8_t;
        auto kaddr = [&](int kc) -> kq_t const * {
          return reinterpret_cast<kq_t const *>(
              &tkv[((kc & 1) ? kswz_o : kswz_e) + kc * 32]);
        };
        kq_t kq[3];
        kq[0] = *kaddr(0);
        kq[1] = *kaddr(1);
#pragma unroll
        for (int kc = 0; kc < NUM_K32; kc++) {
          if (kc + 2 < NUM_K32) {
            kq[(kc + 2) % 3] = *kaddr(kc + 2);
          }
          gang_mla_decode_detail::mla_qk_acc32_mfma_kv(kq[kc % 3], qr[kc]);
        }
      }
      __mfma_hd64_fp32x4 s = mla_qk_acc32_read();
#pragma unroll
      for (int h = 0; h < 4; h++) {
        s[h] *= scale_s;
        if (kgrp * 4 + h >= tlen) {
          s[h] = -INFINITY;
        }
      }
      float tmax = fmaxf(fmaxf(s[0], s[1]), fmaxf(s[2], s[3]));
      {
        float a = tmax, b = tmax;
        asm volatile("s_nop 1\n\tv_permlane32_swap_b32_e32 %0, %1"
                     : "+v"(a), "+v"(b));
        tmax = fmaxf(a, b);
        a = tmax;
        b = tmax;
        asm volatile("s_nop 1\n\tv_permlane16_swap_b32_e32 %0, %1"
                     : "+v"(a), "+v"(b));
        tmax = fmaxf(a, b);
      }
      using gang_mla_decode_detail::mla_exp2;
      __mfma_hd64_fp32x4 w;
#pragma unroll
      for (int h = 0; h < 4; h++) {
        w[h] = (tmax == -INFINITY) ? 0.0f : mla_exp2(s[h] - tmax);
      }
      reinterpret_cast<__mfma_hd64_fp32x4 *>(wpt_p)[wv * 64 + lane] = w;
      if (kgrp == 0) {
        wpt_m[wv * 16 + midx] = tmax;
      }
    }
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    __builtin_amdgcn_s_barrier();
    float mt[4];
    float cmax = -INFINITY;
#pragma unroll
    for (int t = 0; t < 4; t++) {
      mt[t] = (t < ntiles) ? wpt_m[t * 16 + midx] : -INFINITY;
      cmax = fmaxf(cmax, mt[t]);
    }
    using gang_mla_decode_detail::mla_exp2;
    float lp = 0.0f;
#pragma unroll
    for (int t = 0; t < 4; t++) {
      if (t < ntiles) {
        float const sc = (mt[t] == -INFINITY) ? 0.0f : mla_exp2(mt[t] - cmax);
        __mfma_hd64_fp32x4 const w =
            reinterpret_cast<__mfma_hd64_fp32x4 const *>(wpt_p)[t * 64 + lane];
        bf16x4_t pb;
#pragma unroll
        for (int h = 0; h < 4; h++) {
          float const p = w[h] * sc;
          lp += p;
          pb[h] = (__bf16)p;
        }
        __bf16 const *const tkv = lds_kv + t * (KV_TILE * QK_DIM);
        u32x2_t vau[NUM_V_BLOCKS];
#if MPK_DEC_VT
        if (t < nvt) {
          // V^T[d][kgrp * 4 .. + 3], d = vb * 64 + warp_id * 16 + midx.
          __bf16 const *const vrow = vt_lds + t * MPK_DEC_VT_TILE +
                                     (warp_id * 16 + midx) * MPK_DEC_VTS +
                                     kgrp * 4;
#pragma unroll
          for (int vb = 0; vb < NUM_V_BLOCKS; vb++) {
            vau[vb] = *reinterpret_cast<u32x2_t const *>(
                vrow + vb * 64 * MPK_DEC_VTS);
          }
        } else
#endif
        {
#pragma unroll
        for (int vb = 0; vb < NUM_V_BLOCKS; vb++) {
          bf16x4_t va;
          va[0] = tkv[vswz_b0 + vb * 64];
          va[1] = tkv[vswz_b0 + QK_DIM + vb * 64];
          va[2] = tkv[vswz_b2 + vb * 64];
          va[3] = tkv[vswz_b2 + QK_DIM + vb * 64];
          __builtin_memcpy(&vau[vb], &va, 8);
        }
        }
        u32x2_t pbu;
        __builtin_memcpy(&pbu, &pb, 8);
        gang_mla_decode_detail::mla_pv8_vgpr(o_acc_r, vau, pbu);
      }
    }
    m_running = cmax;
    l_head[0] = lp;
    l_head[1] = 0.0f;
    l_head[2] = 0.0f;
    l_head[3] = 0.0f;
    if (tid == 0) {
      mpk_stage_stamp(77);
    }
    // Hands the tiles and the exchange back to the caller, as the last trip's
    // closing barrier does.
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    __builtin_amdgcn_s_barrier();
  }
  if (!wpt) {
#endif
  // One trip at a time. The compiler otherwise pipelines tile t+1's LDS
  // stores into tile t's QK/PV reads; those stores go through a generic
  // uint64_t* and LLVM does not see them as aliasing the bf16 LDS loads.
#pragma unroll 1
  for (int t = 0; t < ntiles; t++) {
#ifdef MPK_ENABLE_SUBPHASE_TIMING
    unsigned long long _d_a = __builtin_amdgcn_s_memrealtime();
#endif
    int tile_start = t * KV_TILE;
    int tile_len = effective_len - tile_start;
    if (tile_len > KV_TILE) {
      tile_len = KV_TILE;
    }
    if (tile_len < 0) {
      tile_len = 0;
    }
#if MPK_MLA_DECODE_BATCH
    constexpr int NB = MPK_MLA_DECODE_BATCH;
    static_assert(PAGE_SIZE % KV_TILE == 0,
                  "a DMA'd KV tile must not straddle a page");
    static_assert(KV_CACHE_STRIDE == QK_DIM,
                  "DMA copies cache rows into the LDS tile verbatim");
    static_assert(KV_LDS_BYTES / 4096 == 4 && (KV_LDS_BYTES % 4096) / 1024 == 2,
                  "mla_dma_tile is written for an 18 KB tile (GLM's 576 dims)");
    int const bslot = t % NB;
    __bf16 *const cur_kv = lds_kv + bslot * (KV_TILE * QK_DIM);
    if (bslot == 0) {
      int const nb = (ntiles - t < NB) ? (ntiles - t) : NB;
      unsigned const wid = __builtin_amdgcn_readfirstlane(warp_id);
      for (int b = 0; b < nb; b++) {
        int const bstart = tile_start + b * KV_TILE;
        int blen = effective_len - bstart;
        if (blen > KV_TILE) {
          blen = KV_TILE;
        }
        __bf16 *const bkv = lds_kv + b * (KV_TILE * QK_DIM);
        if (blen == KV_TILE) {
          long const brow = get_kv_row(kv_start + bstart);
          gang_mla_decode_detail::mla_i32x4_t const rsrc =
              gang_mla_decode_detail::mla_buffer_rsrc(kv_base + brow,
                                                      KV_LDS_BYTES);
          unsigned const m0_base =
              __builtin_amdgcn_readfirstlane(static_cast<unsigned>(
                  reinterpret_cast<uintptr_t>(bkv))) +
              wid * 1024u;
          gang_mla_decode_detail::mla_dma_tile(
              rsrc, m0_base, static_cast<unsigned>(tid) * 16u, wid < 2u);
        } else if (my_tok < blen) {
          long const row = get_kv_row(kv_start + bstart + my_tok);
#pragma unroll
          for (int r = 0; r < LDG_PER_TILE; r++) {
            u32x2_t const v =
                ld_g<u32x2_t>(kv_base + row + (my_dim0 + r * 64) * 2);
            uint64_t val;
            __builtin_memcpy(&val, &v, 8);
            auto *dst = (__attribute__((address_space(3))) uint64_t *)&bkv
#if MPK_MLA_KV_SWZ
                            [my_tok * QK_DIM + my_wcol + r * 64];
#else
                            [my_tok * QK_DIM + my_dim0 + r * 64];
#endif
            *dst = val;
          }
        } else {
          uint64_t const zero8 = 0;
#pragma unroll
          for (int r = 0; r < LDG_PER_TILE; r++) {
            auto *dst = (__attribute__((address_space(3))) uint64_t *)&bkv
#if MPK_MLA_KV_SWZ
                            [my_tok * QK_DIM + my_wcol + r * 64];
#else
                            [my_tok * QK_DIM + my_dim0 + r * 64];
#endif
            *dst = zero8;
          }
        }
      }
      asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
      mla_tile_bar();
    }
#else
#if MPK_MLA_DECODE_DMA_PF
    __bf16 *const cur_kv = (t & 1) ? lds_kv_b : lds_kv;
    if (dma_have) {
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    } else {
#else
    __bf16 *const cur_kv =
        (t < tiles_in_lds) ? lds_kv + t * (KV_TILE * QK_DIM) : lds_kv;
    {
#endif

      // addrspace(3) so the stores are ds_write (lgkmcnt), not generic/flat
      // (vmcnt+lgkmcnt). HIP __syncthreads is
      //   s_waitcnt vmcnt(0) ; s_barrier ; s_waitcnt vmcnt(0)
      // and does not wait lgkmcnt. Do not put vmcnt(0) inside the my_tok <
      // tile_len branch: that wait is divergent on a partial tile and hangs
      // the fused kernel from the first prefill step.
      if (t < tiles_in_lds) {
      } else if (my_tok < tile_len) {
        long const row = get_kv_row(kv_start + tile_start + my_tok);
#pragma unroll
        for (int r = 0; r < LDG_PER_TILE; r++) {
          u32x2_t const v =
              ld_g<u32x2_t>(kv_base + row + (my_dim0 + r * 64) * 2);
          uint64_t val;
          __builtin_memcpy(&val, &v, 8);
          auto *dst = (__attribute__((address_space(3))) uint64_t *)&cur_kv
#if MPK_MLA_KV_SWZ
                          [my_tok * QK_DIM + my_wcol + r * 64];
#else
                          [my_tok * QK_DIM + my_dim0 + r * 64];
#endif
          *dst = val;
        }
      } else {
        uint64_t const zero8 = 0;
#pragma unroll
        for (int r = 0; r < LDG_PER_TILE; r++) {
          auto *dst = (__attribute__((address_space(3))) uint64_t *)&cur_kv
#if MPK_MLA_KV_SWZ
                          [my_tok * QK_DIM + my_wcol + r * 64];
#else
                          [my_tok * QK_DIM + my_dim0 + r * 64];
#endif
          *dst = zero8;
        }
      }
      asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    }
#if MPK_DEC_PRE_NOBAR
    // A preloaded tile was retired and made visible by the barrier behind
    // the query wait, and no trip writes its buffer.
    if (t >= tiles_in_lds) {
      mla_tile_bar();
    }
#else
    mla_tile_bar();
#endif
#endif
    if (tid == 0 && t == 0) {
      mpk_stage_stamp(48);
    }

#if MPK_MLA_DECODE_DMA_PF
    // Every wave is past trip t - 1, so the other buffer is free. Only full
    // tiles go by DMA; a partial one takes the register path next trip.
    dma_have = (t + 1 < ntiles) && (effective_len - tile_start >= 2 * KV_TILE);
    if (dma_have) {
      long const nrow = get_kv_row(kv_start + tile_start + KV_TILE);
      gang_mla_decode_detail::mla_i32x4_t const rsrc =
          gang_mla_decode_detail::mla_buffer_rsrc(kv_base + nrow,
                                                  KV_LDS_BYTES);
      unsigned const wid = __builtin_amdgcn_readfirstlane(warp_id);
      unsigned const m0_base = __builtin_amdgcn_readfirstlane(
          static_cast<unsigned>(reinterpret_cast<uintptr_t>(
              (t & 1) ? lds_kv : lds_kv_b)) +
          wid * 1024u);
      gang_mla_decode_detail::mla_dma_tile(
          rsrc, m0_base, static_cast<unsigned>(tid) * 16u,
          wid < static_cast<unsigned>(DMA_TAIL_WAVES));
    }
#endif

    {
#if MPK_MLA_QK_SPLIT && MPK_MLA_QK_PIPE2 && MPK_MLA_KV_SWZ && \
    MPK_MLA_OACC_VGPR && !MPK_MLA_DECODE_DMA_PF
      // MPK_MLA_QK_SPLIT: the 16 x 16 score block is the same for all four
      // waves, which each read the whole 18 KB K tile for it. Wave 0 runs
      // tile t's chain and, when tile t + 1 is already in LDS, wave 1 runs
      // its chain beside it; both leave their scores in the dead Q staging
      // block and every wave reads them back. Same chains: same bits.
      __mfma_hd64_fp32x4 scores;
      if (qks_have_next) {
        scores = qks_next;
        qks_have_next = false;
      } else {
        bool const pair = t + 1 < ntiles && t + 1 < tiles_in_lds;
        float *const s_sc =
            reinterpret_cast<float *>(_mla_decode_smem + 96 * 1024);
        if (warp_id == 0 || (pair && warp_id == 1)) {
          __bf16 const *const kv =
              (warp_id == 0) ? cur_kv : lds_kv + (t + 1) * (KV_TILE * QK_DIM);
          mla_qk_acc32_zero();
          using kq_t = gang_mla_decode_detail::bf16x8_t;
          auto kaddr = [&](int kc) -> kq_t const * {
            return reinterpret_cast<kq_t const *>(
                &kv[((kc & 1) ? kswz_o : kswz_e) + kc * 32]);
          };
          kq_t kq[3];
          kq[0] = *kaddr(0);
          kq[1] = *kaddr(1);
#pragma unroll
          for (int kc = 0; kc < NUM_K32; kc++) {
            if (kc + 2 < NUM_K32) {
              kq[(kc + 2) % 3] = *kaddr(kc + 2);
            }
            gang_mla_decode_detail::mla_qk_acc32_mfma_kv(kq[kc % 3], qr[kc]);
          }
          __mfma_hd64_fp32x4 const sc = mla_qk_acc32_read();
#pragma unroll
          for (int h = 0; h < 4; h++) {
            s_sc[warp_id * 256 + lane * 4 + h] = sc[h];
          }
        }
        mla_tile_bar();
#pragma unroll
        for (int h = 0; h < 4; h++) {
          scores[h] = s_sc[lane * 4 + h];
        }
        if (pair) {
#pragma unroll
          for (int h = 0; h < 4; h++) {
            qks_next[h] = s_sc[256 + lane * 4 + h];
          }
          qks_have_next = true;
        }
      }
#elif MPK_MLA_QK_2ACC
      static_assert(NUM_K32 % 2 == 0, "QK splits into two equal chains");
      gang_mla_decode_detail::mla_qk_acc2_zero();
#pragma unroll
      for (int kc = 0; kc < NUM_K32; kc += 2) {
        __bf16 kr0[8], kr1[8];
#if MPK_MLA_KV_SWZ
        __bf16 const *k_ptr = &cur_kv[kswz_e + kc * 32];
        __bf16 const *k_ptr1 = &cur_kv[kswz_o + (kc + 1) * 32];
#else
        __bf16 const *k_ptr = &cur_kv[midx * QK_DIM + kc * 32 + kgrp * 8];
        __bf16 const *k_ptr1 = k_ptr + 32;
#endif
#pragma unroll
        for (int i = 0; i < 8; i++) {
          kr0[i] = k_ptr[i];
          kr1[i] = k_ptr1[i];
        }
        gang_mla_decode_detail::mla_qk_acc2_mfma<0>(kr0, qr[kc]);
        gang_mla_decode_detail::mla_qk_acc2_mfma<1>(kr1, qr[kc + 1]);
      }
      __mfma_hd64_fp32x4 scores = gang_mla_decode_detail::mla_qk_acc2_read();
#elif MPK_DEC_QK_KSPLIT
      __mfma_hd64_fp32x4 scores;
      {
        using kq_t = gang_mla_decode_detail::bf16x8_t;
        // Steps qks_w + 4 j share one parity, so one swizzle base.
        __bf16 const *const kb =
            &cur_kv[((qks_w & 1) ? kswz_o : kswz_e) + qks_w * 32];
        kq_t kq[QKS_J];
#pragma unroll
        for (int j = 0; j < QKS_J; j++) {
          if (qks_w + 4 * j < NUM_K32) {
            kq[j] = *reinterpret_cast<kq_t const *>(kb + j * 128);
          }
        }
        mla_qk_acc32_zero();
#pragma unroll
        for (int j = 0; j < QKS_J; j++) {
          if (qks_w + 4 * j < NUM_K32) {
            gang_mla_decode_detail::mla_qk_acc32_mfma_kv(kq[j], qr[j]);
          }
        }
        __mfma_hd64_fp32x4 const part = mla_qk_acc32_read();
        // The Q staging block is dead past the first trip's tile barrier,
        // which every wave's Q reads retired ahead of; each trip's closing
        // barrier orders the next trip's writes after this one's reads.
        __mfma_hd64_fp32x4 *const qks_x = reinterpret_cast<__mfma_hd64_fp32x4 *>(
            _mla_decode_smem + 96 * 1024);
        qks_x[qks_w * 64 + lane] = part;
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        __builtin_amdgcn_s_barrier();
        __mfma_hd64_fp32x4 const p0 = qks_x[lane];
        __mfma_hd64_fp32x4 const p1 = qks_x[64 + lane];
        __mfma_hd64_fp32x4 const p2 = qks_x[128 + lane];
        __mfma_hd64_fp32x4 const p3 = qks_x[192 + lane];
#pragma unroll
        for (int h = 0; h < 4; h++) {
          scores[h] = ((p0[h] + p1[h]) + p2[h]) + p3[h];
        }
      }
#elif (MPK_MLA_QK_PIPE || MPK_MLA_QK_PIPE2) && MPK_MLA_KV_SWZ
      __mfma_hd64_fp32x4 scores;
#if MPK_DEC_QK2T
      // MPK_DEC_QK2T: tile t + 1 preloaded as well, so this wave runs its
      // QK chain beside tile t's -- a[36:39] beside a[32:35], step for
      // step -- and trip t + 1 takes its scores from registers. Each chain
      // is the one-tile loop's: same bits.
      if (qk2_have) {
        scores = qk2_next;
        qk2_have = false;
      } else if (t + 1 < ntiles && t + 1 < tiles_in_lds) {
        using kq_t = gang_mla_decode_detail::bf16x8_t;
        __bf16 const *const kv1 = lds_kv + (t + 1) * (KV_TILE * QK_DIM);
        auto kaddr = [&](__bf16 const *kv, int kc) -> kq_t const * {
          return reinterpret_cast<kq_t const *>(
              &kv[((kc & 1) ? kswz_o : kswz_e) + kc * 32]);
        };
        gang_mla_decode_detail::mla_qk_acc2_zero();
        kq_t ka[3], kb[3];
        ka[0] = *kaddr(cur_kv, 0);
        kb[0] = *kaddr(kv1, 0);
        ka[1] = *kaddr(cur_kv, 1);
        kb[1] = *kaddr(kv1, 1);
#pragma unroll
        for (int kc = 0; kc < NUM_K32; kc++) {
          if (kc + 2 < NUM_K32) {
            ka[(kc + 2) % 3] = *kaddr(cur_kv, kc + 2);
            kb[(kc + 2) % 3] = *kaddr(kv1, kc + 2);
          }
          gang_mla_decode_detail::mla_qk_acc32_mfma_kv(ka[kc % 3], qr[kc]);
          gang_mla_decode_detail::mla_qk_acc36_mfma_kv(kb[kc % 3], qr[kc]);
        }
        gang_mla_decode_detail::mla_qk_acc2_read_pair(scores, qk2_next);
        qk2_have = true;
      } else
#endif
      {
      mla_qk_acc32_zero();
      {
        // The pipelined loop on the swizzled tile: the K reads stop being
        // 8-way bank-conflicted, so the reads in flight can actually overlap.
        using kq_t = gang_mla_decode_detail::bf16x8_t;
        constexpr int QD = MPK_MLA_QK_PIPE2 ? 2 : 1;
        auto kaddr = [&](int kc) -> kq_t const * {
          return reinterpret_cast<kq_t const *>(
              &cur_kv[((kc & 1) ? kswz_o : kswz_e) + kc * 32]);
        };
        kq_t kq[3];
#pragma unroll
        for (int d = 0; d < QD; d++) {
          kq[d] = *kaddr(d);
        }
#pragma unroll
        for (int kc = 0; kc < NUM_K32; kc++) {
          if (kc + QD < NUM_K32) {
            kq[(kc + QD) % 3] = *kaddr(kc + QD);
          }
          gang_mla_decode_detail::mla_qk_acc32_mfma_kv(kq[kc % 3], qr[kc]);
        }
      }
      scores = mla_qk_acc32_read();
      }
#elif MPK_MLA_QK_PIPE2 && !MPK_MLA_KV_SWZ
      mla_qk_acc32_zero();
      {
        // Two chunks in flight ahead of the MFMA: one LDS round trip per step
        // was the whole step (MPK_TRACE: QK 0.68 us per tile).
        using kq_t = gang_mla_decode_detail::bf16x8_t;
        __bf16 const *const k_row = &cur_kv[midx * QK_DIM + kgrp * 8];
        kq_t kq[3];
        kq[0] = *reinterpret_cast<kq_t const *>(k_row);
        kq[1] = *reinterpret_cast<kq_t const *>(k_row + 32);
#pragma unroll
        for (int kc = 0; kc < NUM_K32; kc++) {
          if (kc + 2 < NUM_K32) {
            kq[(kc + 2) % 3] =
                *reinterpret_cast<kq_t const *>(k_row + (kc + 2) * 32);
          }
          gang_mla_decode_detail::mla_qk_acc32_mfma_kv(kq[kc % 3], qr[kc]);
        }
      }
      __mfma_hd64_fp32x4 scores = mla_qk_acc32_read();
#elif MPK_MLA_QK_PIPE && !MPK_MLA_KV_SWZ
      mla_qk_acc32_zero();
      {
        // One chunk in flight ahead of the MFMA that consumes the previous.
        using kq_t = gang_mla_decode_detail::bf16x8_t;
        __bf16 const *const k_row = &cur_kv[midx * QK_DIM + kgrp * 8];
        kq_t kq0 = *reinterpret_cast<kq_t const *>(k_row);
        kq_t kq1;
#pragma unroll
        for (int kc = 0; kc < NUM_K32; kc++) {
          if (kc + 1 < NUM_K32) {
            kq_t const nx =
                *reinterpret_cast<kq_t const *>(k_row + (kc + 1) * 32);
            if (kc & 1) {
              kq0 = nx;
            } else {
              kq1 = nx;
            }
          }
          gang_mla_decode_detail::mla_qk_acc32_mfma_kv((kc & 1) ? kq1 : kq0,
                                                       qr[kc]);
        }
      }
      __mfma_hd64_fp32x4 scores = mla_qk_acc32_read();
#else
      mla_qk_acc32_zero();
#pragma unroll
      for (int kc = 0; kc < NUM_K32; kc++) {
        __bf16 kr[8];
#if MPK_MLA_KV_SWZ
        __bf16 const *k_ptr = &cur_kv[((kc & 1) ? kswz_o : kswz_e) + kc * 32];
#else
        __bf16 const *k_ptr = &cur_kv[midx * QK_DIM + kc * 32 + kgrp * 8];
#endif
#pragma unroll
        for (int i = 0; i < 8; i++) {
          kr[i] = k_ptr[i];
        }
        mla_qk_acc32_mfma(kr, qr[kc]);
      }
      __mfma_hd64_fp32x4 scores = mla_qk_acc32_read();
#endif
      // Trace stamps 75/76/77 (trip 0): QK done, softmax done, PV done.
      if (tid == 0 && t == 0) {
        mpk_stage_stamp(75);
      }
      scores[0] *= scale_s;
      scores[1] *= scale_s;
      scores[2] *= scale_s;
      scores[3] *= scale_s;
#pragma unroll
      for (int h = 0; h < 4; h++) {
        if (kgrp * 4 + h >= tile_len) {
          scores[h] = -INFINITY;
        }
      }
      float tile_max =
          fmaxf(fmaxf(scores[0], scores[1]), fmaxf(scores[2], scores[3]));
      {
        float a = tile_max, b = tile_max;
        asm volatile("s_nop 1\n\tv_permlane32_swap_b32_e32 %0, %1"
                     : "+v"(a), "+v"(b));
        tile_max = fmaxf(a, b);
        a = tile_max;
        b = tile_max;
        asm volatile("s_nop 1\n\tv_permlane16_swap_b32_e32 %0, %1"
                     : "+v"(a), "+v"(b));
        tile_max = fmaxf(a, b);
      }
      float const new_max = fmaxf(m_running, tile_max);
      using gang_mla_decode_detail::mla_exp2;
      float const rescale = (m_running == -INFINITY)
                                ? 0.0f
                                : mla_exp2(m_running - new_max);
      float const w0 = mla_exp2(scores[0] - new_max);
      float const w1 = mla_exp2(scores[1] - new_max);
      float const w2 = mla_exp2(scores[2] - new_max);
      float const w3 = mla_exp2(scores[3] - new_max);
      l_head[0] = l_head[0] * rescale + w0;
      l_head[1] = l_head[1] * rescale + w1;
      l_head[2] = l_head[2] * rescale + w2;
      l_head[3] = l_head[3] * rescale + w3;
      m_running = new_max;

      bf16x4_t pb;
      pb[0] = (__bf16)w0;
      pb[1] = (__bf16)w1;
      pb[2] = (__bf16)w2;
      pb[3] = (__bf16)w3;
      if (tid == 0 && t == 0) {
        mpk_stage_stamp(76);
      }
#if MPK_MLA_OACC_VGPR && MPK_MLA_PV_PIPE && !MPK_MLA_KV_SWZ
      {
        u32x2_t vau[NUM_V_BLOCKS];
        __bf16 const *const v_base =
            &cur_kv[(kgrp * 4) * QK_DIM + warp_id * 16 + midx];
        bf16x4_t va0, va1;
        va0[0] = v_base[0 * QK_DIM];
        va0[1] = v_base[1 * QK_DIM];
        va0[2] = v_base[2 * QK_DIM];
        va0[3] = v_base[3 * QK_DIM];
#pragma unroll
        for (int vb = 0; vb < NUM_V_BLOCKS; vb++) {
          if (vb + 1 < NUM_V_BLOCKS) {
            __bf16 const *const nx = v_base + (vb + 1) * 64;
            bf16x4_t vn;
            vn[0] = nx[0 * QK_DIM];
            vn[1] = nx[1 * QK_DIM];
            vn[2] = nx[2 * QK_DIM];
            vn[3] = nx[3 * QK_DIM];
            if (vb & 1) {
              va0 = vn;
            } else {
              va1 = vn;
            }
          }
          // Left alone, the scheduler sinks these reads below the pack.
          __builtin_amdgcn_sched_barrier(0);
          bf16x4_t const va = (vb & 1) ? va1 : va0;
          __builtin_memcpy(&vau[vb], &va, 8);
          o_acc_r[vb] *= rescale;
        }
        u32x2_t pbu;
        __builtin_memcpy(&pbu, &pb, 8);
        gang_mla_decode_detail::mla_pv8_vgpr(o_acc_r, vau, pbu);
      }
#elif MPK_MLA_OACC_VGPR && MPK_MLA_PV_ASM && MPK_MLA_KV_SWZ
      {
        // The swizzled tile's 32 V reads from two per-lane bases (rows
        // kgrp*4 + {0,1} and + {2,3}), all into distinct registers, one
        // wait. Swizzled, each LDS round trip is short, so the C++ form's
        // eight serialized groups were most of the PV step (MPK_TRACE: 0.45
        // us per tile against eight independent MFMAs).
        static_assert(NUM_V_BLOCKS == 8 && QK_DIM == 576,
                      "the read offsets below are block * 128 + row * 1152");
        unsigned const b0 = static_cast<unsigned>(
            reinterpret_cast<uintptr_t>(&cur_kv[vswz_b0]));
        unsigned const b2 = static_cast<unsigned>(
            reinterpret_cast<uintptr_t>(&cur_kv[vswz_b2]));
        unsigned t[32];
        asm volatile("ds_read_u16 %0, %16 offset:0\n"
                     "ds_read_u16 %1, %16 offset:1152\n"
                     "ds_read_u16 %2, %17 offset:0\n"
                     "ds_read_u16 %3, %17 offset:1152\n"
                     "ds_read_u16 %4, %16 offset:128\n"
                     "ds_read_u16 %5, %16 offset:1280\n"
                     "ds_read_u16 %6, %17 offset:128\n"
                     "ds_read_u16 %7, %17 offset:1280\n"
                     "ds_read_u16 %8, %16 offset:256\n"
                     "ds_read_u16 %9, %16 offset:1408\n"
                     "ds_read_u16 %10, %17 offset:256\n"
                     "ds_read_u16 %11, %17 offset:1408\n"
                     "ds_read_u16 %12, %16 offset:384\n"
                     "ds_read_u16 %13, %16 offset:1536\n"
                     "ds_read_u16 %14, %17 offset:384\n"
                     "ds_read_u16 %15, %17 offset:1536"
                     : "=&v"(t[0]), "=&v"(t[1]), "=&v"(t[2]), "=&v"(t[3]),
                       "=&v"(t[4]), "=&v"(t[5]), "=&v"(t[6]), "=&v"(t[7]),
                       "=&v"(t[8]), "=&v"(t[9]), "=&v"(t[10]), "=&v"(t[11]),
                       "=&v"(t[12]), "=&v"(t[13]), "=&v"(t[14]), "=&v"(t[15])
                     : "v"(b0), "v"(b2)
                     : "memory");
        asm volatile("ds_read_u16 %0, %16 offset:512\n"
                     "ds_read_u16 %1, %16 offset:1664\n"
                     "ds_read_u16 %2, %17 offset:512\n"
                     "ds_read_u16 %3, %17 offset:1664\n"
                     "ds_read_u16 %4, %16 offset:640\n"
                     "ds_read_u16 %5, %16 offset:1792\n"
                     "ds_read_u16 %6, %17 offset:640\n"
                     "ds_read_u16 %7, %17 offset:1792\n"
                     "ds_read_u16 %8, %16 offset:768\n"
                     "ds_read_u16 %9, %16 offset:1920\n"
                     "ds_read_u16 %10, %17 offset:768\n"
                     "ds_read_u16 %11, %17 offset:1920\n"
                     "ds_read_u16 %12, %16 offset:896\n"
                     "ds_read_u16 %13, %16 offset:2048\n"
                     "ds_read_u16 %14, %17 offset:896\n"
                     "ds_read_u16 %15, %17 offset:2048\n"
                     "s_waitcnt lgkmcnt(0)"
                     : "=&v"(t[16]), "=&v"(t[17]), "=&v"(t[18]), "=&v"(t[19]),
                       "=&v"(t[20]), "=&v"(t[21]), "=&v"(t[22]), "=&v"(t[23]),
                       "=&v"(t[24]), "=&v"(t[25]), "=&v"(t[26]), "=&v"(t[27]),
                       "=&v"(t[28]), "=&v"(t[29]), "=&v"(t[30]), "=&v"(t[31])
                     : "v"(b0), "v"(b2)
                     : "memory");
#pragma unroll
        for (int i = 0; i < 16; i++) {
          asm volatile("" : "+v"(t[i]));
        }
        u32x2_t vau[NUM_V_BLOCKS];
#pragma unroll
        for (int vb = 0; vb < NUM_V_BLOCKS; vb++) {
          vau[vb][0] = t[4 * vb] | (t[4 * vb + 1] << 16);
          vau[vb][1] = t[4 * vb + 2] | (t[4 * vb + 3] << 16);
          o_acc_r[vb] *= rescale;
        }
        u32x2_t pbu;
        __builtin_memcpy(&pbu, &pb, 8);
        gang_mla_decode_detail::mla_pv8_vgpr(o_acc_r, vau, pbu);
      }
#elif MPK_MLA_OACC_VGPR && MPK_MLA_PV_ASM && !MPK_MLA_KV_SWZ
      {
        static_assert(NUM_V_BLOCKS == 8 && QK_DIM == 576,
                      "the read offsets below are block * 128 + row * 1152");
        unsigned const v_addr = static_cast<unsigned>(reinterpret_cast<uintptr_t>(
            &cur_kv[(kgrp * 4) * QK_DIM + warp_id * 16 + midx]));
        unsigned t[32];
        // Every read into its own register, so no block's reads wait on the
        // previous block's pack. Blocks 0-3 here, 4-7 and the one wait below.
        asm volatile("ds_read_u16 %0, %16 offset:0\n"
                     "ds_read_u16 %1, %16 offset:1152\n"
                     "ds_read_u16 %2, %16 offset:2304\n"
                     "ds_read_u16 %3, %16 offset:3456\n"
                     "ds_read_u16 %4, %16 offset:128\n"
                     "ds_read_u16 %5, %16 offset:1280\n"
                     "ds_read_u16 %6, %16 offset:2432\n"
                     "ds_read_u16 %7, %16 offset:3584\n"
                     "ds_read_u16 %8, %16 offset:256\n"
                     "ds_read_u16 %9, %16 offset:1408\n"
                     "ds_read_u16 %10, %16 offset:2560\n"
                     "ds_read_u16 %11, %16 offset:3712\n"
                     "ds_read_u16 %12, %16 offset:384\n"
                     "ds_read_u16 %13, %16 offset:1536\n"
                     "ds_read_u16 %14, %16 offset:2688\n"
                     "ds_read_u16 %15, %16 offset:3840"
                     : "=&v"(t[0]), "=&v"(t[1]), "=&v"(t[2]), "=&v"(t[3]),
                       "=&v"(t[4]), "=&v"(t[5]), "=&v"(t[6]), "=&v"(t[7]),
                       "=&v"(t[8]), "=&v"(t[9]), "=&v"(t[10]), "=&v"(t[11]),
                       "=&v"(t[12]), "=&v"(t[13]), "=&v"(t[14]), "=&v"(t[15])
                     : "v"(v_addr)
                     : "memory");
        asm volatile("ds_read_u16 %0, %16 offset:512\n"
                     "ds_read_u16 %1, %16 offset:1664\n"
                     "ds_read_u16 %2, %16 offset:2816\n"
                     "ds_read_u16 %3, %16 offset:3968\n"
                     "ds_read_u16 %4, %16 offset:640\n"
                     "ds_read_u16 %5, %16 offset:1792\n"
                     "ds_read_u16 %6, %16 offset:2944\n"
                     "ds_read_u16 %7, %16 offset:4096\n"
                     "ds_read_u16 %8, %16 offset:768\n"
                     "ds_read_u16 %9, %16 offset:1920\n"
                     "ds_read_u16 %10, %16 offset:3072\n"
                     "ds_read_u16 %11, %16 offset:4224\n"
                     "ds_read_u16 %12, %16 offset:896\n"
                     "ds_read_u16 %13, %16 offset:2048\n"
                     "ds_read_u16 %14, %16 offset:3200\n"
                     "ds_read_u16 %15, %16 offset:4352\n"
                     "s_waitcnt lgkmcnt(0)"
                     : "=&v"(t[16]), "=&v"(t[17]), "=&v"(t[18]), "=&v"(t[19]),
                       "=&v"(t[20]), "=&v"(t[21]), "=&v"(t[22]), "=&v"(t[23]),
                       "=&v"(t[24]), "=&v"(t[25]), "=&v"(t[26]), "=&v"(t[27]),
                       "=&v"(t[28]), "=&v"(t[29]), "=&v"(t[30]), "=&v"(t[31])
                     : "v"(v_addr)
                     : "memory");
#pragma unroll
        for (int i = 0; i < 16; i++) {
          asm volatile("" : "+v"(t[i]));
        }
        u32x2_t vau[NUM_V_BLOCKS];
#pragma unroll
        for (int vb = 0; vb < NUM_V_BLOCKS; vb++) {
          vau[vb][0] = t[4 * vb] | (t[4 * vb + 1] << 16);
          vau[vb][1] = t[4 * vb + 2] | (t[4 * vb + 3] << 16);
          o_acc_r[vb] *= rescale;
        }
        u32x2_t pbu;
        __builtin_memcpy(&pbu, &pb, 8);
        gang_mla_decode_detail::mla_pv8_vgpr(o_acc_r, vau, pbu);
      }
#elif MPK_MLA_OACC_VGPR
      {
        u32x2_t vau[NUM_V_BLOCKS];
#pragma unroll
        for (int vb = 0; vb < NUM_V_BLOCKS; vb++) {
          bf16x4_t va;
#if MPK_MLA_KV_SWZ
          va[0] = cur_kv[vswz_b0 + vb * 64];
          va[1] = cur_kv[vswz_b0 + QK_DIM + vb * 64];
          va[2] = cur_kv[vswz_b2 + vb * 64];
          va[3] = cur_kv[vswz_b2 + QK_DIM + vb * 64];
#else
          __bf16 const *v_ptr =
              &cur_kv[(kgrp * 4) * QK_DIM + vb * 64 + warp_id * 16 + midx];
          va[0] = v_ptr[0 * QK_DIM];
          va[1] = v_ptr[1 * QK_DIM];
          va[2] = v_ptr[2 * QK_DIM];
          va[3] = v_ptr[3 * QK_DIM];
#endif
          __builtin_memcpy(&vau[vb], &va, 8);
          o_acc_r[vb] *= rescale;
        }
        u32x2_t pbu;
        __builtin_memcpy(&pbu, &pb, 8);
        gang_mla_decode_detail::mla_pv8_vgpr(o_acc_r, vau, pbu);
      }
#else
#pragma unroll
      for (int vb = 0; vb < NUM_V_BLOCKS; vb++) {
        bf16x4_t va;
#if MPK_MLA_KV_SWZ
        va[0] = cur_kv[vswz_b0 + vb * 64];
        va[1] = cur_kv[vswz_b0 + QK_DIM + vb * 64];
        va[2] = cur_kv[vswz_b2 + vb * 64];
        va[3] = cur_kv[vswz_b2 + QK_DIM + vb * 64];
#else
        __bf16 const *v_ptr =
            &cur_kv[(kgrp * 4) * QK_DIM + vb * 64 + warp_id * 16 + midx];
        va[0] = v_ptr[0 * QK_DIM];
        va[1] = v_ptr[1 * QK_DIM];
        va[2] = v_ptr[2 * QK_DIM];
        va[3] = v_ptr[3 * QK_DIM];
#endif
        if (vb == 0) {
          mla_flash_pv_b0<KV_LDS_BYTES>(oacc_base, rescale, va, pb);
        } else if (vb == 1) {
          mla_flash_pv_b1<KV_LDS_BYTES>(oacc_base, rescale, va, pb);
        } else if (vb == 2) {
          mla_flash_pv_b2<KV_LDS_BYTES>(oacc_base, rescale, va, pb);
        } else if (vb == 3) {
          mla_flash_pv_b3<KV_LDS_BYTES>(oacc_base, rescale, va, pb);
        } else if (vb == 4) {
          mla_flash_pv_b4<KV_LDS_BYTES>(oacc_base, rescale, va, pb);
        } else if (vb == 5) {
          mla_flash_pv_b5<KV_LDS_BYTES>(oacc_base, rescale, va, pb);
        } else if (vb == 6) {
          mla_flash_pv_b6<KV_LDS_BYTES>(oacc_base, rescale, va, pb);
        } else {
          mla_flash_pv_b7<KV_LDS_BYTES>(oacc_base, rescale, va, pb);
        }
      }
#endif
    }

    if (tid == 0 && t == 0) {
      mpk_stage_stamp(77);
    }
    // The QK and PV steps both read the whole LDS tile, so the refill has to
    // wait for every lane rather than slotting between them the way the HD=64
    // decode does.
#ifdef MPK_ENABLE_SUBPHASE_TIMING
    unsigned long long _d_b = __builtin_amdgcn_s_memrealtime();
    _d_compute += _d_b - _d_a;
#endif
    // Next trip overwrites LDS; wait for every lane's QK/PV reads. Kept under
    // MPK_MLA_DECODE_DMA_PF too: without it test_mla_decode failed every
    // 32-chunk case (warp 3's partials zero), single-trip ones included, for
    // reasons not pinned down; with it all 46 pass. Its vmcnt(0) also retires
    // the next tile's DMA, which by then has had this trip's QK/softmax/PV to
    // land in. Under MPK_MLA_DECODE_BATCH only the batch's last trip refills.
#if MPK_MLA_DECODE_BATCH
    if (bslot == MPK_MLA_DECODE_BATCH - 1 || t == ntiles - 1) {
      mla_tile_bar();
    }
#elif MPK_DEC_PRE_NOBAR
    // Only a register-path trip writes LDS, into buffer 0, and the last
    // trip's barrier hands the tiles back to the caller.
    if (t + 1 >= tiles_in_lds) {
      mla_tile_bar();
    }
#else
    mla_tile_bar();
#endif
#ifdef MPK_ENABLE_SUBPHASE_TIMING
    _d_refill += __builtin_amdgcn_s_memrealtime() - _d_b;
#endif
  }
#if MPK_DEC_WPT
  } // !wpt
#endif

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  unsigned long long _d_t3 = __builtin_amdgcn_s_memrealtime();
#endif
  if (tid == 0) {
    mpk_stage_stamp(49);
  }

  // ===== Output =====
  // l_head lives on the kgrp lanes; fold them into one value per q head.
  float l_sum = l_head[0] + l_head[1] + l_head[2] + l_head[3];
  l_sum += __shfl_xor(l_sum, 16);
  l_sum += __shfl_xor(l_sum, 32);

  float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
  int const q_head_local = midx;

  __mfma_hd64_fp32x4 o_acc[NUM_V_BLOCKS];
#if MPK_MLA_OACC_VGPR
#pragma unroll
  for (int vb = 0; vb < NUM_V_BLOCKS; vb++) {
    o_acc[vb] = o_acc_r[vb];
  }
#else
  mla_oacc_from_lds<KV_LDS_BYTES>(oacc_base, o_acc);
#endif

  if constexpr (NUM_KV_CHUNKS == 1) {
    constexpr int OUT_STRIDE = NUM_Q_HEADS * KV_LORA_RANK;
    bf16 *o = reinterpret_cast<bf16 *>(output_ptr) +
              static_cast<long>(q_row) * OUT_STRIDE +
              static_cast<long>(q_head_group * Q_HEADS_PER_GROUP +
                                q_head_local) *
                  KV_LORA_RANK;
#pragma unroll
    for (int vb = 0; vb < NUM_V_BLOCKS; vb++) {
#pragma unroll
      for (int h = 0; h < 4; h++) {
        int dim_offset = vb * 64 + warp_id * 16 + kgrp * 4 + h;
        o[dim_offset] = static_cast<bf16>(o_acc[vb][h] * inv_l);
      }
    }
  } else if (q_head_local - out_head_lo >= 0 &&
             q_head_local - out_head_lo < out_heads) {
    // Split-KV partial output: float + LSE, laid out exactly the way
    // merge_splitkv_ck_fmha indexes it with q_head_group as kv_head_idx.
    constexpr int LSE_S = NUM_Q_GROUPS * NUM_KV_CHUNKS * Q_HEADS_PER_GROUP;
    constexpr int O_S = LSE_S * KV_LORA_RANK;
    float *o = reinterpret_cast<float *>(output_ptr) +
               static_cast<long>(q_row) * O_S +
               static_cast<long>(q_head_group) * NUM_KV_CHUNKS *
                   Q_HEADS_PER_GROUP * KV_LORA_RANK +
               static_cast<long>(kv_chunk_idx) * Q_HEADS_PER_GROUP *
                   KV_LORA_RANK +
               static_cast<long>(q_head_local) * KV_LORA_RANK;
#if MPK_DEC_OUT_B128
    // One V# over the whole partial buffer; the offsets below stay well
    // inside 2^31 bytes. aux 17 = sc0 sc1, st_wt's write-through.
    mpk_dec_i32x4_t o_rsrc;
    {
      unsigned long long const ob = (unsigned long long)output_ptr;
      o_rsrc[0] = (int)__builtin_amdgcn_readfirstlane((unsigned)ob);
      o_rsrc[1] = (int)__builtin_amdgcn_readfirstlane((unsigned)(ob >> 32));
      o_rsrc[2] = (int)0x7FFFFFFF;
      o_rsrc[3] = (int)0x00020000u;
    }
#endif
#if MPK_DEC_MERGE_LL32
    // MPK_DEC_MERGE_LL32's words over ll_o, [chunk][own head][dim]: a lane's
    // four dims are four words, two dwordx4 of (f32, epoch) pairs.
    mpk_dec_i32x4_t w_rsrc;
    {
      unsigned long long const wb = (unsigned long long)ll_o;
      w_rsrc[0] = (int)__builtin_amdgcn_readfirstlane((unsigned)wb);
      w_rsrc[1] = (int)__builtin_amdgcn_readfirstlane((unsigned)(wb >> 32));
      w_rsrc[2] = (int)0x7FFFFFFF;
      w_rsrc[3] = (int)0x00020000u;
    }
    int const w_row =
        (kv_chunk_idx * out_heads + (q_head_local - out_head_lo)) * KV_LORA_RANK;
#endif
#pragma unroll
    for (int vb = 0; vb < NUM_V_BLOCKS; vb++) {
      int dim_offset = vb * 64 + warp_id * 16 + kgrp * 4;
      if constexpr (WRITE_THROUGH) {
        // The four dims are contiguous and 4-aligned, so one
        // global_store_dwordx4 carries them past L2 in a single instruction.
        unsigned raw[4];
#pragma unroll
        for (int h = 0; h < 4; h++) {
          float v = o_acc[vb][h] * inv_l;
          __builtin_memcpy(&raw[h], &v, 4);
        }
        // Two dwordx2 stores, not one dwordx4, and the reason is NOT that
        // st_wt_u128 is broken -- tests/standalone/test_st_wt.hip writes a
        // known pattern through it into a poisoned buffer and every dword
        // comes back correct, so the two other call sites (the MoE workspace
        // zeroing in gang_oproj_router_fused and the EP publish in
        // xrank_sum_add) are fine and should be left alone.
        //
        // It is wrong HERE, deterministically. test_mla_decode.hip
        // instantiated with WRITE_THROUGH=true -- which only the fused layer
        // does, and which no test covered until now -- fails every split-KV
        // case with the dwordx4 form, including chunks=2 seqlen=1, and passes
        // all 19 with this one. The wrong values are always the 4th dword of
        // each group (dims == 3 mod 4).
        //
        // The difference this function brings is register pressure: it sets
        // the whole megakernel's allocation at 248 arch + 36 AGPR (see the
        // note above mla_decode_absorbed), and the dwordx4 asm needs a
        // 4-aligned VGPR quad for its payload operand. Do not "simplify" this
        // back to one store without re-running the WRITE_THROUGH=true arm.
#if MPK_DEC_MERGE_LL32
        if (ll_o != nullptr) {
          int const ep = (int)ll_epoch;
          mpk_dec_i32x4_t const w01 = {(int)raw[0], ep, (int)raw[1], ep};
          mpk_dec_i32x4_t const w23 = {(int)raw[2], ep, (int)raw[3], ep};
          __mpk_dec_raw_buffer_store_v4i32(w01, w_rsrc,
                                           (w_row + dim_offset) * 8, 0, 17);
          __mpk_dec_raw_buffer_store_v4i32(w23, w_rsrc,
                                           (w_row + dim_offset) * 8 + 16, 0, 17);
        }
#else
        if (ll_o != nullptr) {
          unsigned long long *const ow =
              ll_o + ((&o[dim_offset] - reinterpret_cast<float *>(output_ptr)) >>
                      1);
          unsigned long long const ep = (unsigned long long)ll_epoch << 32;
          auto pk = [](unsigned a, unsigned b) -> unsigned {
            __hip_bfloat16 const x = __float2bfloat16(__uint_as_float(a));
            __hip_bfloat16 const y = __float2bfloat16(__uint_as_float(b));
            unsigned short ux, uy;
            __builtin_memcpy(&ux, &x, 2);
            __builtin_memcpy(&uy, &y, 2);
            return (unsigned)ux | ((unsigned)uy << 16);
          };
          st_wt_u64((void *)ow, ep | pk(raw[0], raw[1]));
          st_wt_u64((void *)(ow + 1), ep | pk(raw[2], raw[3]));
        }
#endif
        if (ll_o == nullptr || MPK_DML_DBG == 1) {
#if MPK_DEC_PART_F16
          // MPK_DEC_PART_F16: the four partials as f16, at the same element
          // index of an f16 view of the buffer -- one write-through dwordx2.
          _Float16 h4[4];
#pragma unroll
          for (int h = 0; h < 4; h++) {
            h4[h] = (_Float16)__uint_as_float(raw[h]);
          }
          unsigned long long hv;
          __builtin_memcpy(&hv, h4, 8);
          st_wt_u64((void *)(reinterpret_cast<_Float16 *>(output_ptr) +
                             (&o[dim_offset] -
                              reinterpret_cast<float *>(output_ptr))),
                    hv);
#elif MPK_DEC_OUT_B128
          mpk_dec_i32x4_t const d4 = {(int)raw[0], (int)raw[1], (int)raw[2],
                                      (int)raw[3]};
          __mpk_dec_raw_buffer_store_v4i32(
              d4, o_rsrc,
              (int)((&o[dim_offset] - reinterpret_cast<float *>(output_ptr)) *
                    4),
              0, 17);
#else
          unsigned long long lo, hi;
          __builtin_memcpy(&lo, &raw[0], 8);
          __builtin_memcpy(&hi, &raw[2], 8);
          st_wt_u64((void *)&o[dim_offset], lo);
          st_wt_u64((void *)&o[dim_offset + 2], hi);
#endif
        }
      } else {
#pragma unroll
        for (int h = 0; h < 4; h++) {
          o[dim_offset + h] = o_acc[vb][h] * inv_l;
        }
      }
    }
  }

#if MPK_DEC_LSE_LL
  // MPK_DEC_LSE_LL: the lse word is the merge's signal that this item's f32
  // partials are in memory, so every wave's stores retire before it goes.
  if (ll_lse != nullptr && ll_o == nullptr) {
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __syncthreads();
  }
#endif
  // LSE for the split-KV merge. One lane per q head writes.
  //
  // scale_s carries log2(e), so m_running is a log2 exponent and l_sum sums
  // exp2 terms; merge_splitkv_ck_fmha wants natural-log LSE (it multiplies by
  // log2(e) on the way back in), hence the ln(2) on m_running.
  if (warp_id == 0 && kgrp == 0 && q_head_local - out_head_lo >= 0 &&
      q_head_local - out_head_lo < out_heads) {
    constexpr int LSE_STRIDE =
        NUM_Q_GROUPS * NUM_KV_CHUNKS * Q_HEADS_PER_GROUP;
    float *lse_out = reinterpret_cast<float *>(lse_ptr) +
                     static_cast<long>(q_row) * LSE_STRIDE +
                     q_head_group * NUM_KV_CHUNKS * Q_HEADS_PER_GROUP +
                     kv_chunk_idx * Q_HEADS_PER_GROUP + q_head_local;
    float lse_val = (l_sum > 0.0f)
                        ? (m_running * 0.69314718055994530942f + logf(l_sum))
                        : -1e30f;
    if (ll_lse != nullptr) {
#if MPK_DEC_MERGE_LL32
      // [own head][chunk], so head 0's words are the merge gate's 8 lines.
      unsigned long long *const lw =
          ll_lse + (q_head_local - out_head_lo) * NUM_KV_CHUNKS + kv_chunk_idx;
#else
      unsigned long long *const lw =
          ll_lse + (lse_out - reinterpret_cast<float *>(lse_ptr));
#endif
      st_wt_u64((void *)lw, ((unsigned long long)ll_epoch << 32) |
                                __float_as_uint(lse_val));
    }
    if (ll_lse != nullptr && MPK_DML_DBG != 1) {
    } else if constexpr (WRITE_THROUGH) {
      unsigned raw;
      __builtin_memcpy(&raw, &lse_val, 4);
      st_wt_u32((void *)lse_out, raw);
    } else {
      *lse_out = lse_val;
    }
  }

  if (tid == 0) {
    mpk_stage_stamp(50);
  }
#ifdef MPK_ENABLE_SUBPHASE_TIMING
  if (tid == 0 && g_subphase_active) {
    unsigned long long _d_t4 = __builtin_amdgcn_s_memrealtime();
    atomicAdd(&g_subphase_ns[2][2], (_d_t1 - _d_t0) * 10);
    atomicAdd(&g_subphase_ns[2][3], (_d_t2 - _d_t1) * 10);
    atomicAdd(&g_subphase_ns[2][4], _d_compute * 10);
    atomicAdd(&g_subphase_ns[2][5], _d_refill * 10);
    atomicAdd(&g_subphase_ns[2][6], (_d_t4 - _d_t3) * 10);
    // Must be g_subphase_cnt, not another g_subphase_ns phase: the printer
    // skips any slot whose cnt is still zero.
    atomicAdd(&g_subphase_cnt[2], 1ULL);
  }
#endif
}

// Gang MLA decode: NUM_Q_GROUPS * NUM_KV_CHUNKS tasks broadcast to workers,
// one per XCD at the GLM-5 shape (4 q-head groups x 2 sequence chunks).
//
// tile_idx decomposition (q-head group fastest, so the first 8 tiles land on
// distinct XCDs the same way kv_head does in gang_attention_mi300.cuh):
//   q_head_group = tile_idx % NUM_Q_GROUPS
//   kv_chunk     = (tile_idx / NUM_Q_GROUPS) % NUM_KV_CHUNKS
//   token        = (tile_idx / (NUM_Q_GROUPS * NUM_KV_CHUNKS)) % BATCH_SIZE
//   request_id   =  tile_idx / (NUM_Q_GROUPS * NUM_KV_CHUNKS * BATCH_SIZE)
//
// The token dimension sits INSIDE the request one because a request's rows
// share its page list -- a work item is (row, q group, kv chunk), and the
// caller's total_work_items has to carry the BATCH_SIZE factor to match. At
// BATCH_SIZE 1 the token term is a constant 0 and request_id is unchanged, so
// this is the same decomposition it always was.
//
// The latent cache append (c_kv + k_rope for the new token) is done by the
// preceding task, so unlike gang_attention_split_kv_kernel there is no
// Phase A here.
template <typename T,
          int NUM_Q_HEADS,
          int KV_LORA_RANK,
          int QK_ROPE_HEAD_DIM,
          int PAGE_SIZE,
          int MAX_SEQ_LEN,
          int NUM_KV_CHUNKS,
          int Q_WORKSPACE_STRIDE,
          int KV_CACHE_STRIDE,
          bool WRITE_THROUGH = false,
          int BATCH_SIZE = 1>
#if MPK_DECODE_INLINE
__device__ __attribute__((always_inline)) void
#else
__device__ __noinline__ void
#endif
    gang_mla_decode_kernel(void const *q_workspace_ptr,
                           void const *paged_kv_cache_ptr,
                           void *output_ptr,
                           void *lse_ptr,
                           int const *qo_indptr,
                           int const *kv_indptr,
                           int const *kv_indices,
                           int const *kv_last_page_len,
                           int total_work_items,
                           int tile_idx,
                           float scale_s,
                           int out_head_lo = 0,
                           int out_heads = 16,
                           // MPK_DEC_PRELOAD: the caller already DMA'd this
                           // chunk's first tiles, full; see mla_decode_absorbed.
                           int tiles_in_lds = 0,
                           // MPK_DEC_HINTS; see mla_decode_absorbed.
                           int hint_q_row = -1,
                           int hint_first_page = -1,
                           int hint_seqlen = -1,
                           // MPK_DEC_MERGE_LL; see mla_decode_absorbed.
                           unsigned long long *ll_o = nullptr,
                           unsigned long long *ll_lse = nullptr,
                           unsigned ll_epoch = 0,
                           // MPK_DEC_Q_LL; see mla_decode_absorbed.
                           unsigned long long const *q_ll = nullptr,
                           unsigned q_ll_epoch = 0,
                           int q_ll_own_lo = 0,
                           unsigned long long const *kv_flag = nullptr) {
  if (tile_idx >= total_work_items) {
    return;
  }

  constexpr int NUM_Q_GROUPS = NUM_Q_HEADS / 16;

  int const q_head_group = tile_idx % NUM_Q_GROUPS;
  int const kv_chunk_idx = (tile_idx / NUM_Q_GROUPS) % NUM_KV_CHUNKS;
  int const token_idx =
      (BATCH_SIZE == 1)
          ? 0
          : ((tile_idx / (NUM_Q_GROUPS * NUM_KV_CHUNKS)) % BATCH_SIZE);
  int16_t const request_id = static_cast<int16_t>(
      tile_idx / (NUM_Q_GROUPS * NUM_KV_CHUNKS * BATCH_SIZE));

  mla_decode_absorbed<T,
                      NUM_Q_HEADS,
                      KV_LORA_RANK,
                      QK_ROPE_HEAD_DIM,
                      PAGE_SIZE,
                      MAX_SEQ_LEN,
                      NUM_KV_CHUNKS,
                      Q_WORKSPACE_STRIDE,
                      KV_CACHE_STRIDE,
                      WRITE_THROUGH,
                      BATCH_SIZE>(q_workspace_ptr,
                                       paged_kv_cache_ptr,
                                       output_ptr,
                                       lse_ptr,
                                       qo_indptr,
                                       kv_indptr,
                                       kv_indices,
                                       kv_last_page_len,
                                       request_id,
                                       token_idx,
                                       q_head_group,
                                       kv_chunk_idx,
                                       scale_s,
                                       out_head_lo,
                                       out_heads,
                                       tiles_in_lds,
                                       hint_q_row,
                                       hint_first_page,
                                       hint_seqlen,
                                       ll_o,
                                       ll_lse,
                                       ll_epoch,
                                       q_ll,
                                       q_ll_epoch,
                                       q_ll_own_lo,
                                       kv_flag);
}

} // namespace kernel
