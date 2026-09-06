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

// __mfma_qk_hd64 / __mfma_pv_hd64 / __fast_exp2_hd64 / __load_bf16x4_to_fp16 /
// __load_bf16x4_raw / __cvt_bf16x4_to_fp16 live here. They are tiling-agnostic
// despite the hd64 name.
#include "tasks/mi300/paged_attention_decode_minimal_hd64_mi300.cuh"
#include "mpk_atoms.cuh"

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
  asm volatile("v_mfma_f32_16x16x32_bf16 a[32:35], %0, %1, a[32:35]\n"
               :
               : "v"(avu), "v"(bvu)
               : "a32", "a33", "a34", "a35");
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
// Base = byte address of lds_kv plus tid*128; KV_BYTES is in the
// ds offset field (fits in 16 bits: 18432+124) so RA cannot drop it.
constexpr int MLA_OACC_V_BLOCKS = 8;
constexpr int MLA_OACC_FLOATS = MLA_OACC_V_BLOCKS * 4; // 32

__device__ __forceinline__ unsigned mla_oacc_lds_base(int tid,
                                                     void const *lds_kv) {
  unsigned base;
  unsigned const kv =
      static_cast<unsigned>(reinterpret_cast<uintptr_t>(lds_kv));
  asm volatile("v_lshlrev_b32 %[b], 7, %[tid]\n\t"
               "v_add_u32_e32 %[b], %[kv], %[b]"
               : [b] "=v"(base)
               : [tid] "v"(tid), [kv] "v"(kv));
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
               : [o0] "n"(KV_BYTES + 0), [o1] "n"(KV_BYTES + 4),
                 [o2] "n"(KV_BYTES + 8), [o3] "n"(KV_BYTES + 12),
                 [o4] "n"(KV_BYTES + 16), [o5] "n"(KV_BYTES + 20),
                 [o6] "n"(KV_BYTES + 24), [o7] "n"(KV_BYTES + 28),
                 [o8] "n"(KV_BYTES + 32), [o9] "n"(KV_BYTES + 36),
                 [o10] "n"(KV_BYTES + 40), [o11] "n"(KV_BYTES + 44),
                 [o12] "n"(KV_BYTES + 48), [o13] "n"(KV_BYTES + 52),
                 [o14] "n"(KV_BYTES + 56), [o15] "n"(KV_BYTES + 60),
                 [o16] "n"(KV_BYTES + 64), [o17] "n"(KV_BYTES + 68),
                 [o18] "n"(KV_BYTES + 72), [o19] "n"(KV_BYTES + 76),
                 [o20] "n"(KV_BYTES + 80), [o21] "n"(KV_BYTES + 84),
                 [o22] "n"(KV_BYTES + 88), [o23] "n"(KV_BYTES + 92),
                 [o24] "n"(KV_BYTES + 96), [o25] "n"(KV_BYTES + 100),
                 [o26] "n"(KV_BYTES + 104), [o27] "n"(KV_BYTES + 108),
                 [o28] "n"(KV_BYTES + 112), [o29] "n"(KV_BYTES + 116),
                 [o30] "n"(KV_BYTES + 120), [o31] "n"(KV_BYTES + 124)
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
                   [o0] "n"(KV_BYTES + (DELTA) + 0),                           \
                   [o1] "n"(KV_BYTES + (DELTA) + 4),                           \
                   [o2] "n"(KV_BYTES + (DELTA) + 8),                           \
                   [o3] "n"(KV_BYTES + (DELTA) + 12)                           \
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

template <int KV_BYTES, int VB>
__device__ __forceinline__ void
    mla_oacc_load_block(unsigned &base, __mfma_hd64_fp32x4 &out) {
  float t0, t1, t2, t3;
  constexpr int off = KV_BYTES + VB * 16;
  asm volatile("ds_read_b32 %[t0], %[base] offset:%c[o0]\n\t"
               "ds_read_b32 %[t1], %[base] offset:%c[o1]\n\t"
               "ds_read_b32 %[t2], %[base] offset:%c[o2]\n\t"
               "ds_read_b32 %[t3], %[base] offset:%c[o3]\n\t"
               "s_waitcnt lgkmcnt(0)"
               : [t0] "=v"(t0), [t1] "=v"(t1), [t2] "=v"(t2), [t3] "=v"(t3),
                 [base] "+v"(base)
               : [o0] "n"(off + 0), [o1] "n"(off + 4), [o2] "n"(off + 8),
                 [o3] "n"(off + 12)
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

// The bf16x4 loads from paged_attention_decode_minimal_hd64, re-expressed so
// the load itself goes through ld_g. The convert half is reused verbatim.
__device__ __forceinline__ void __ldg_bf16x4_raw(uint2 *__restrict__ dst,
                                                 void const *__restrict__ src) {
  u32x2_t const v = ld_g<u32x2_t>(src);
  dst->x = v.x;
  dst->y = v.y;
}
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
__device__ __noinline__ void
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
                        float scale_s) {
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
  static_assert(KV_LDS_BYTES + 124 < 65536,
                "o_acc ds offsets must fit in the 16-bit ds offset field");

  int const req = request_id;
  int const query_start = ld_g<int>(&qo_indptr[req]);
  int const query_end = ld_g<int>(&qo_indptr[req + 1]);
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

  int const first_page = ld_g<int>(&kv_indptr[req]);
  int const num_pages = ld_g<int>(&kv_indptr[req + 1]) - first_page;
  int const seqlen_k = (num_pages - 1) * PAGE_SIZE +
                       ld_g<int>(&kv_last_page_len[req]) - tok_back;

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
      if (warp_id == 0 && kgrp == 0) {
        constexpr int LSE_STRIDE =
            NUM_Q_GROUPS * NUM_KV_CHUNKS * Q_HEADS_PER_GROUP;
        float *lse_out = reinterpret_cast<float *>(lse_ptr) +
                         static_cast<long>(q_row) * LSE_STRIDE +
                         q_head_group * NUM_KV_CHUNKS * Q_HEADS_PER_GROUP +
                         kv_chunk_idx * Q_HEADS_PER_GROUP + midx;
        if constexpr (WRITE_THROUGH) {
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

  // Q first, then one KV tile per loop trip from HBM.
  __bf16 qr[NUM_K32][8];
  {
    int const q_head = q_head_group * Q_HEADS_PER_GROUP + midx;
    char const *q_ptr = reinterpret_cast<char const *>(q_workspace_ptr) +
                        (static_cast<long>(q_row) * Q_WORKSPACE_STRIDE +
                         static_cast<long>(q_head) * QK_DIM) *
                            2;
#pragma unroll
    for (int kc = 0; kc < NUM_K32; kc++) {
      int dim_off = kc * 32 + kgrp * 8;
      u32x4_t const raw = ld_g<u32x4_t>(q_ptr + dim_off * 2);
      __builtin_memcpy(&qr[kc][0], &raw, 16);
    }
  }

  // ===== MAIN LOOP =====
  mla_oacc_zero_lds<KV_LDS_BYTES>(oacc_base);
  float m_running = -INFINITY;
  float l_head[4] = {0, 0, 0, 0};

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  unsigned long long _d_t2 = __builtin_amdgcn_s_memrealtime();
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

    // addrspace(3) so the stores are ds_write (lgkmcnt), not generic/flat
    // (vmcnt+lgkmcnt). HIP __syncthreads is
    //   s_waitcnt vmcnt(0) ; s_barrier ; s_waitcnt vmcnt(0)
    // and does not wait lgkmcnt. Do not put vmcnt(0) inside the my_tok <
    // tile_len branch: that wait is divergent on a partial tile and hangs
    // the fused kernel from the first prefill step.
    if (my_tok < tile_len) {
      long const row = get_kv_row(kv_start + tile_start + my_tok);
#pragma unroll
      for (int r = 0; r < LDG_PER_TILE; r++) {
        u32x2_t const v =
            ld_g<u32x2_t>(kv_base + row + (my_dim0 + r * 64) * 2);
        uint64_t val;
        __builtin_memcpy(&val, &v, 8);
        auto *dst = (__attribute__((address_space(3))) uint64_t *)&lds_kv
                        [my_tok * QK_DIM + my_dim0 + r * 64];
        *dst = val;
      }
    } else {
      uint64_t const zero8 = 0;
#pragma unroll
      for (int r = 0; r < LDG_PER_TILE; r++) {
        auto *dst = (__attribute__((address_space(3))) uint64_t *)&lds_kv
                        [my_tok * QK_DIM + my_dim0 + r * 64];
        *dst = zero8;
      }
    }
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    __syncthreads();

    {
      mla_qk_acc32_zero();
#pragma unroll
      for (int kc = 0; kc < NUM_K32; kc++) {
        __bf16 kr[8];
        __bf16 const *k_ptr = &lds_kv[midx * QK_DIM + kc * 32 + kgrp * 8];
#pragma unroll
        for (int i = 0; i < 8; i++) {
          kr[i] = k_ptr[i];
        }
        mla_qk_acc32_mfma(kr, qr[kc]);
      }
      __mfma_hd64_fp32x4 scores = mla_qk_acc32_read();
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
      float const rescale = (m_running == -INFINITY)
                                ? 0.0f
                                : __fast_exp2_hd64(m_running - new_max);
      float const w0 = __fast_exp2_hd64(scores[0] - new_max);
      float const w1 = __fast_exp2_hd64(scores[1] - new_max);
      float const w2 = __fast_exp2_hd64(scores[2] - new_max);
      float const w3 = __fast_exp2_hd64(scores[3] - new_max);
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
#pragma unroll
      for (int vb = 0; vb < NUM_V_BLOCKS; vb++) {
        bf16x4_t va;
        __bf16 const *v_ptr =
            &lds_kv[(kgrp * 4) * QK_DIM + vb * 64 + warp_id * 16 + midx];
        va[0] = v_ptr[0 * QK_DIM];
        va[1] = v_ptr[1 * QK_DIM];
        va[2] = v_ptr[2 * QK_DIM];
        va[3] = v_ptr[3 * QK_DIM];
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
    }

    // The QK and PV steps both read the whole LDS tile, so the refill has to
    // wait for every lane rather than slotting between them the way the HD=64
    // decode does.
#ifdef MPK_ENABLE_SUBPHASE_TIMING
    unsigned long long _d_b = __builtin_amdgcn_s_memrealtime();
    _d_compute += _d_b - _d_a;
#endif
    // Next trip overwrites LDS; wait for every lane's QK/PV reads.
    __syncthreads();
#ifdef MPK_ENABLE_SUBPHASE_TIMING
    _d_refill += __builtin_amdgcn_s_memrealtime() - _d_b;
#endif
  }

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  unsigned long long _d_t3 = __builtin_amdgcn_s_memrealtime();
#endif

  // ===== Output =====
  // l_head lives on the kgrp lanes; fold them into one value per q head.
  float l_sum = l_head[0] + l_head[1] + l_head[2] + l_head[3];
  l_sum += __shfl_xor(l_sum, 16);
  l_sum += __shfl_xor(l_sum, 32);

  float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
  int const q_head_local = midx;

  __mfma_hd64_fp32x4 o_acc[NUM_V_BLOCKS];
  mla_oacc_from_lds<KV_LDS_BYTES>(oacc_base, o_acc);

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
  } else {
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
        unsigned long long lo, hi;
        __builtin_memcpy(&lo, &raw[0], 8);
        __builtin_memcpy(&hi, &raw[2], 8);
        st_wt_u64((void *)&o[dim_offset], lo);
        st_wt_u64((void *)&o[dim_offset + 2], hi);
      } else {
#pragma unroll
        for (int h = 0; h < 4; h++) {
          o[dim_offset + h] = o_acc[vb][h] * inv_l;
        }
      }
    }
  }

  // LSE for the split-KV merge. One lane per q head writes.
  //
  // scale_s carries log2(e), so m_running is a log2 exponent and l_sum sums
  // exp2 terms; merge_splitkv_ck_fmha wants natural-log LSE (it multiplies by
  // log2(e) on the way back in), hence the ln(2) on m_running.
  if (warp_id == 0 && kgrp == 0) {
    constexpr int LSE_STRIDE =
        NUM_Q_GROUPS * NUM_KV_CHUNKS * Q_HEADS_PER_GROUP;
    float *lse_out = reinterpret_cast<float *>(lse_ptr) +
                     static_cast<long>(q_row) * LSE_STRIDE +
                     q_head_group * NUM_KV_CHUNKS * Q_HEADS_PER_GROUP +
                     kv_chunk_idx * Q_HEADS_PER_GROUP + q_head_local;
    float lse_val = (l_sum > 0.0f)
                        ? (m_running * 0.69314718055994530942f + logf(l_sum))
                        : -1e30f;
    if constexpr (WRITE_THROUGH) {
      unsigned raw;
      __builtin_memcpy(&raw, &lse_val, 4);
      st_wt_u32((void *)lse_out, raw);
    } else {
      *lse_out = lse_val;
    }
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
__device__ __noinline__ void
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
                           float scale_s) {
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
                                       scale_s);
}

} // namespace kernel
