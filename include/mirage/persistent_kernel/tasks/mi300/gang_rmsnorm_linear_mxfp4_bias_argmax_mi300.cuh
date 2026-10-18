/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

// Fused RMSNorm + MXFP4 Gang Linear + Bias + Argmax for MI300/MI350.
//
// Norm once, FP8 quant once, internal tile loop over all WGs
// assigned to this worker. Argmax accumulated in registers across all tiles.
// Each worker writes one (bf16 max_val, int64 abs_idx) pair per batch row.
//
// total_tiles_per_xcd = workers_per_xcd (30 on MI300X).
// tile_idx = worker rank within XCD (0..29).
// Internal loop: wg = tile_idx, tile_idx + workers_per_xcd, ...
//
// ── Batching (bs > 1) rides the MFMA's idle N dimension ──────────────────
// The instruction is mfma_scale_f32_16x16x128_f8f6f4: M=16 weight rows,
// N=16 token columns, K=128. Lane l supplies A[m = l&15][k-blk = l>>4] and
// B[n = l&15][k-blk = l>>4], and receives D[m = (l>>4)*4 + i][n = l&15].
// The single-token version broadcast the same token into all 16 N columns
// and read back only n == 0, so 15/16 of every MFMA was thrown away. Here
// token `col` occupies N column `col`, so up to TOK_ROWS = 16 batch rows
// cost exactly what 1 row used to: same MFMA count, same weight traffic.
//
// Above 16 rows the batch is swept in column blocks (NUM_BBLK passes). That
// pass loop has to sit OUTSIDE the weight-tile loop, because 64 fp8 rows of
// K=2944 are 188 KB and the LDS budget is 152 KB — the token rows for the
// whole batch cannot be resident at once no matter how the weights are
// staged. So bs>16 re-reads the LM-head weights once per 16-row block; the
// alternative (re-quantizing inside the tile loop) would pay the quant's
// fixed ~15 us launch latency ~50x per worker and is far worse.
//
// Eliminates:
//   - 393 redundant RMSNorm + FP8 quant (down to 30 — one per worker)
//   - ~786 KB HBM logits write + read
//   - argmax_partial task dispatch

#pragma once
#include "tasks/mi300/gang_moe_linear_mxfp4_mi300.cuh"
#include "tasks/mi300/gang_rmsnorm_linear_bias_mi300.cuh"

#ifdef MPK_EAGLE3
#include "eagle3_draft_mi300.cuh"
#endif
#ifdef MPK_DFLASH
#include "dflash_draft_mi300.cuh"
#endif
namespace kernel {

#if defined(MPK_LM_BF16) && \
    (defined(MPK_LM_HEAD_GROUP_PIPELINE) || defined(MPK_LM_PF0) || defined(MPK_LMT_FQ))
#error "MPK_LM_BF16 streams BF16 weight records into registers; it replaces the MXFP4 group pipeline, MPK_LM_PF0 and MPK_LMT_FQ"
#endif
#if defined(MPK_LM_HEAD_GROUP_PIPELINE) && !defined(MPK_LM_HEAD_KMAJOR)
#error "MPK_LM_HEAD_GROUP_PIPELINE requires MPK_LM_HEAD_KMAJOR"
#endif
#if defined(MPK_LM_HEAD_GROUP_PIPELINE) && !defined(MPK_LM_HEAD_WAVE_TILE_DMA)
#error "MPK_LM_HEAD_GROUP_PIPELINE requires MPK_LM_HEAD_WAVE_TILE_DMA"
#endif

#ifdef MPK_LM_MALL_SC0
#define MPK_LM_PLAIN_MOD " sc0"
#else
#define MPK_LM_PLAIN_MOD ""
#endif

#ifdef MPK_LM_HEAD_GROUP_PIPELINE
// gfx950 does not interlock a SALU write to M0 against the following
// load-to-LDS MUBUF. Keep the required independent wait state in the same asm
// block as the destination update and request.
__device__ __forceinline__ void lm_head_load_lds_dwordx4_hazard_safe(
    i32x4_t resource,
    __attribute__((address_space(3))) uint32_t *destination,
    uint32_t vector_offset,
    bool non_temporal = true) {
  unsigned const lds_offset = __builtin_amdgcn_readfirstlane(
      static_cast<unsigned>(reinterpret_cast<uintptr_t>(destination)));
  if (non_temporal) {
    asm volatile(
        "s_mov_b32 m0, %[lds_offset]\n"
        "s_nop 0\n"
        "buffer_load_dwordx4 %[vector_offset], %[resource], 0 offen sc0 nt "
        "lds\n"
        :
        : [vector_offset] "v"(vector_offset), [resource] "s"(resource),
          [lds_offset] "s"(lds_offset)
        : "memory", "m0");
  } else {
    asm volatile(
        "s_mov_b32 m0, %[lds_offset]\n"
        "s_nop 0\n"
        "buffer_load_dwordx4 %[vector_offset], %[resource], 0 offen" MPK_LM_PLAIN_MOD " lds\n"
        :
        : [vector_offset] "v"(vector_offset), [resource] "s"(resource),
          [lds_offset] "s"(lds_offset)
        : "memory", "m0");
  }
}
#endif

#if defined(MPK_LM_HEAD_GROUP_PIPELINE) && defined(MPK_LM_PIPE_S)
// Same request, destination already scalar.
__device__ __forceinline__ void lm_head_load_lds_dwordx4_s(i32x4_t resource, unsigned lds_offset,
                                                           uint32_t vector_offset,
                                                           bool non_temporal = true) {
  if (non_temporal) {
    asm volatile(
        "s_mov_b32 m0, %[lds_offset]\n"
        "s_nop 0\n"
        "buffer_load_dwordx4 %[vector_offset], %[resource], 0 offen sc0 nt lds\n"
        :
        : [vector_offset] "v"(vector_offset), [resource] "s"(resource),
          [lds_offset] "s"(lds_offset)
        : "memory", "m0");
  } else {
    asm volatile(
        "s_mov_b32 m0, %[lds_offset]\n"
        "s_nop 0\n"
        "buffer_load_dwordx4 %[vector_offset], %[resource], 0 offen" MPK_LM_PLAIN_MOD " lds\n"
        :
        : [vector_offset] "v"(vector_offset), [resource] "s"(resource),
          [lds_offset] "s"(lds_offset)
        : "memory", "m0");
  }
}
#endif

// MPK_LM_MALL_PCT: which groups stream non-temporally. The first PCT% of an
// XCD's groups load plain and stay MALL-resident across tokens; everything
// else per token (MoE, QKV, O-proj data) is `sc0 nt`, which does not allocate.
#ifdef MPK_LM_MALL_PCT
#define MPK_LM_NT(g) ((g) * 100 >= MPK_LM_MALL_PCT * n_wgs_per_xcd)
#else
#define MPK_LM_NT(g) true
#endif

#ifdef MPK_LMXB
// [LMX] stamps for MPK_LMXB_N sampled steps: (t0, t1, hw) per worker.
#define MPK_LMXB_N 64
#define MPK_LMXB_W 256
__device__ unsigned long long g_lmxb[MPK_LMXB_N * MPK_LMXB_W * 3];
#endif

#ifdef MPK_ARGMAX_DUAL_REDUCE
// ── The LM head's g-group argmax, as two VALU swaps instead of four LDS ─────
//
// The reduction this replaces is only two steps wide -- xor-16 then xor-32,
// walking the four `g` groups and leaving `col` alone -- but each step costs
// two `__shfl_xor`, one for the value and one for the index, and `__shfl_xor`
// lowers to ds_bpermute: an LDS round trip and an lgkmcnt wait per shuffle for
// data that never left the register file. Four round trips, two waits.
//
// Both steps have an exact VALU-native equivalent on gfx950 and they are the
// only two the whole reduction needs: `v_permlane16_swap_b32` is xor-16 and
// `v_permlane32_swap_b32` is xor-32, so unlike the router's six-step chain
// this one needs no DPP row_shl stages at all.
//
// The swaps exchange one operand's low half with the other's high half, so
// seeding each peer with a copy of its own value makes swap-then-select a
// lane-uniform butterfly: in the low lanes the peer holds lane i^k's value
// and in the high lanes it holds the same, which is exactly what the shuffle
// delivered.
//
// The value and the index are two independent partials over the same lane
// crossing, so they interleave: the swap's result-hazard slot is filled by the
// other partial's swap rather than by an `s_nop`. That is why this is worth
// doing at two steps -- the pair costs barely more than one would.
//
// ── Contract: the result is valid in g == 0 (lanes 0..15) ──────────────────
//
// The value is lane-uniform and matches the butterfly everywhere. The *index*
// does not, and the reason is the same asymmetry that makes the sum version
// free: the swap leaves the lower member of each pair holding its own value in
// `val` and its partner's in `peer`, and the upper member holding them the
// other way round. A commutative reduction cannot see that. A strict `>`
// tie-break can -- in the upper member the incumbent is `peer`, so keeping
// `val` on a tie keeps the wrong side.
//
// Lanes 0..15 are the lower member at both steps (bit 4 clear for the xor-16
// swap, bit 5 clear for the xor-32 swap), so their tie-break is unchanged.
// That is all the caller reads: `if (g == 0) s_red_val[...] = thread_max`.
// Hence the name -- do not lift this to a caller that consumes g1..g3.
//
// Verified over 4096 cases including an all-equal block where every
// comparison is a tie: g0 value and index bit-identical to the butterfly,
// value bit-identical in all 64 lanes.
// See tests/standalone/test_argmax_dual_reduce.hip.
__device__ __forceinline__ void argmax_dual_wave_reduce_to_g0(float &val,
                                                              int &idx) {
  float val_peer = val;
  int idx_peer = idx;
  asm volatile(
      // ── xor-16: walk g groups 0<->1 and 2<->3 ──
      "s_nop 1\n"
      "v_permlane16_swap_b32 %[val], %[val_peer]\n"
      "v_permlane16_swap_b32 %[idx], %[idx_peer]\n"
      "v_cmp_gt_f32 vcc, %[val_peer], %[val]\n"
      "v_cndmask_b32 %[idx], %[idx], %[idx_peer], vcc\n"
      "v_cndmask_b32 %[val], %[val], %[val_peer], vcc\n"
      // Re-seed: the swaps are destructive in both operands, so the peer has
      // to be a fresh copy of the survivor rather than carried across steps.
      "v_mov_b32_e32 %[val_peer], %[val]\n"
      "v_mov_b32_e32 %[idx_peer], %[idx]\n"
      // ── xor-32: walk the two halves ──
      "s_nop 1\n"
      "v_permlane32_swap_b32 %[val], %[val_peer]\n"
      "v_permlane32_swap_b32 %[idx], %[idx_peer]\n"
      "v_cmp_gt_f32 vcc, %[val_peer], %[val]\n"
      "v_cndmask_b32 %[idx], %[idx], %[idx_peer], vcc\n"
      "v_cndmask_b32 %[val], %[val], %[val_peer], vcc"
      : [val] "+v"(val),
        [idx] "+v"(idx),
        [val_peer] "+v"(val_peer),
        [idx_peer] "+v"(idx_peer)
      :
      : "vcc");
}
#endif

#ifdef MPK_LM_PF0
// MPK_LM_PF0_V1
#if !defined(MPK_LM_HEAD_GROUP_PIPELINE) || !defined(MPK_LM_HEAD_KMAJOR) ||   \
    !defined(MPK_LM_HEAD_WAVE_TILE_DMA)
#error "MPK_LM_PF0 needs MPK_LM_HEAD_GROUP_PIPELINE, MPK_LM_HEAD_KMAJOR and MPK_LM_HEAD_WAVE_TILE_DMA"
#endif
// Under MPK_LM_MALL_PCT the prefetched group zero stays non-temporal.
#if defined(MPK_LM_HW_XCC2)
#error "MPK_LM_PF0 does not reproduce MPK_LM_HW_XCC2"
#endif
// Group zero of a worker's tile, issued by the worker loop before the task's
// dependency wait (the weights depend on nothing). Offsets match the kernel's
// LDS layout (static_assert in the kernel). The tag word is s_red_val[0]:
// tile_idx + 1 when issued, 0 otherwise; the task writes s_red_* only after
// its tile loop.
template <int BATCH_SIZE, int OUTPUT_PER_WG, int REDUCTION_SIZE>
struct LmPf0Layout {
  static constexpr int NUM_BLOCKS_32 = REDUCTION_SIZE / 32;
  static constexpr int WG_DATA_BYTES = OUTPUT_PER_WG * (REDUCTION_SIZE / 2);
  static constexpr int WG_BYTES = WG_DATA_BYTES + OUTPUT_PER_WG * NUM_BLOCKS_32;
  static constexpr int MFMA_ITERS = REDUCTION_SIZE / 128;
  static constexpr int TOK_ROWS = BATCH_SIZE < 16 ? BATCH_SIZE : 16;
  static constexpr int TILE_DATA = 16 * (REDUCTION_SIZE / 2);
  static constexpr int TILE_SCALE = 16 * NUM_BLOCKS_32;
  static constexpr int TILE_DATA_PADDED =
      ((TILE_DATA / 16 + 255) / 256) * 256 * 16;
  static constexpr int TILE_BYTES = TILE_DATA_PADDED + TILE_SCALE;
  static constexpr int RED_OFF =
      ((TOK_ROWS * (REDUCTION_SIZE + 16) +
        TOK_ROWS * (((MFMA_ITERS + 3) / 4) * 4) + 15) /
       16) *
      16;
  static constexpr int LDS_OFF = ((RED_OFF + 4 * 16 * 8 + 15) / 16) * 16;
};

template <int BATCH_SIZE,
          int OUTPUT_PER_WG,
          int REDUCTION_SIZE,
          int ACTUAL_HIDDEN_DIM = REDUCTION_SIZE>
__device__ __forceinline__ void
gang_rmsnorm_linear_mxfp4_bias_argmax_pf0(void const *weight_ptr,
                                          int n_wgs_per_xcd,
                                          int workers_per_xcd,
                                          int tile_idx) {
  using L = LmPf0Layout<BATCH_SIZE, OUTPUT_PER_WG, REDUCTION_SIZE>;
  static_assert(BATCH_SIZE <= 16 && OUTPUT_PER_WG == 64,  // the kernel checks the layout below
                "MPK_LM_PF0: up to 16 rows (one MFMA column block), 64 rows per group");
  static_assert(L::TILE_DATA % 1024 == 0 && L::TILE_SCALE > 1024 &&
                    (L::TILE_SCALE - 1024) % 16 == 0,
                "MPK_LM_PF0: tile must be whole 1 KiB requests");
  extern __shared__ char _rnlm_smem[];
  int const wg = tile_idx < 0 ? n_wgs_per_xcd : tile_idx % workers_per_xcd;
  if (wg >= n_wgs_per_xcd) {
    if (threadIdx.x == 0) {
      *(volatile int *)(_rnlm_smem + L::RED_OFF) = 0;
    }
    return;
  }
  int const warp = threadIdx.x >> 6;
  int const lane = threadIdx.x & 63;
  uint8_t *const tile =
      (uint8_t *)_rnlm_smem + L::LDS_OFF + warp * L::TILE_BYTES;
  i32x4_t const rsrc =
      make_w_buffer_rsrc((uint8_t const *)weight_ptr,
                         static_cast<uint32_t>(n_wgs_per_xcd) * L::WG_BYTES);
  unsigned const lds_base = __builtin_amdgcn_readfirstlane(
      static_cast<unsigned>(reinterpret_cast<uintptr_t>(
          (__attribute__((address_space(3))) uint8_t *)tile)));
  uint32_t voff = static_cast<uint32_t>(wg) * L::WG_BYTES +
                  static_cast<uint32_t>(warp * L::TILE_DATA) +
                  static_cast<uint32_t>(lane * 16);
  asm volatile("s_nop 4\n"
               "s_mov_b32 m0, %[lds_base]\n"
               "s_nop 0\n"
               "buffer_load_dwordx4 %[voff], %[rsrc], 0 offen sc0 nt lds\n"
               ".rept %c[reps]\n"
               "s_addk_i32 m0, 0x400\n"
               "v_add_u32_e32 %[voff], 0x400, %[voff]\n"
               "buffer_load_dwordx4 %[voff], %[rsrc], 0 offen sc0 nt lds\n"
               ".endr\n"
               : [voff] "+v"(voff)
               : [rsrc] "s"(rsrc),
                 [lds_base] "s"(lds_base),
                 [reps] "n"(L::TILE_DATA / 1024 - 1)
               : "memory", "m0", "scc");
  auto *const sc = (__attribute__((address_space(3))) uint32_t *)(
      tile + L::TILE_DATA_PADDED);
  uint32_t const soff = static_cast<uint32_t>(wg) * L::WG_BYTES +
                        static_cast<uint32_t>(L::WG_DATA_BYTES) +
                        static_cast<uint32_t>(warp * L::TILE_SCALE) +
                        static_cast<uint32_t>(lane * 16);
  lm_head_load_lds_dwordx4_hazard_safe(rsrc, sc, soff, false);
  if (lane < (L::TILE_SCALE - 1024) / 16) {
    lm_head_load_lds_dwordx4_hazard_safe(
        rsrc, sc + 1024 / sizeof(uint32_t), soff + 1024, false);
  }
  if (threadIdx.x == 0) {
    *(volatile int *)(_rnlm_smem + L::RED_OFF) = tile_idx + 1;
  }
}
#endif

#if defined(MPK_LMT_FQ) && !defined(MPK_LMT_NORM)
#error "MPK_LMT_FQ is the FP8 quantization of MPK_LMT_NORM's rows"
#endif
#ifdef MPK_LMT_NORM
#if defined(MPK_LMT_FQ) && defined(MPK_LMT_GLB)
#error "MPK_LMT_GLB: the bf16 rows in the global norm buffer, not with MPK_LMT_FQ"
#endif
#if defined(MPK_LM_NORM_LDS) || (defined(MPK_LM_PF0) && !defined(MPK_LMT_FQ) && !defined(MPK_LMT_GLB))
#error "MPK_LMT_NORM stages the rows in the LM-head weight tiles: not with MPK_LM_NORM_LDS or MPK_LM_PF0 (MPK_LMT_FQ: PF0 ok)"
#endif
#ifdef MPK_LMT_FQ
// DPP max over the 16 lanes of a row (quad_perm [1,0,3,2], [2,3,0,1], row_half_mirror, row_mirror): every lane gets
// the row's maximum. fmaxf is exact and order-independent, so this equals any other reduction order.
template <int CTRL>
__device__ __forceinline__ float lmt_dpp_f(float x) {
  return __int_as_float(__builtin_amdgcn_update_dpp(__float_as_int(x), __float_as_int(x), CTRL, 0xf, 0xf, false));
}
__device__ __forceinline__ float lmt_rowmax16(float m) {
  m = fmaxf(m, lmt_dpp_f<0xB1>(m));
  m = fmaxf(m, lmt_dpp_f<0x4E>(m));
  m = fmaxf(m, lmt_dpp_f<0x141>(m));
  m = fmaxf(m, lmt_dpp_f<0x140>(m));
  return m;
}
__device__ __forceinline__ float lmt_wavemax(float m) {
  m = lmt_rowmax16(m);
  float const a = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(m), 0));
  float const b = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(m), 16));
  float const c = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(m), 32));
  float const d = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(m), 48));
  return fmaxf(fmaxf(a, b), fmaxf(c, d));
}
__device__ __forceinline__ float lmt_e8m0_scale(uint8_t se) {
  return se == 0 ? 1.0f : __uint_as_float((uint32_t)se << 23);
}
#endif
// MPK_LMT_NORM: the LM head's RMSNorm of every live row in one pass, into LDS. Each row keeps rmsnorm_inline_amd's
// per-thread element order, wave tree and cross-wave association ((w0 + w2) + (w1 + w3)), and the same rsqrtf and
// bf16 rounding, so the rows are bit-identical. Gamma is loaded once and three barriers replace three per row. The
// rows go to LDS (the weight tiles, idle until the tile loop) instead of the AID-shared global norm buffer, which
// only the FP8 quantizer read back. Dead rows are zeroed so their (discarded) MFMA columns stay finite.
// RA (MPK_LM_RESADD): `in` is the residual and `ws` the last layer's MoE f32 workspace ([row][4 slots][dim]); each
// input element is bf16((((w0 + w1) + w2) + w3) + res), the order and rounding of the in-kernel residual add
// (and of moe_residual_add_f32_mi300), computed in registers instead of staged in LDS.
template <int STORAGE_DIM, int ACTUAL_HIDDEN_DIM, int ROWS, bool RA = false, bool FQ = false, int TRS = 0,
          int SCS = 0>
__device__ __forceinline__ void lmt_rmsnorm_rows(unsigned short const *in, void const *weight_ptr,
                                                 unsigned short *rows, float *red, int nrows, bool ts = false,
                                                 float const *ws = nullptr, uint8_t *tok_fp8 = nullptr,
                                                 uint8_t *tok_sc = nullptr, float *amx = nullptr) {
  using bf16 = __hip_bfloat16;
  constexpr int VEC = 8, NT = 256;
  static_assert(STORAGE_DIM / (NT * VEC) == 1, "MPK_LMT_NORM: one 8-wide vector per thread, then a tail");
  static_assert(ROWS <= 16, "MPK_LMT_NORM: at most 16 rows (4 lanes per row in the cross-wave step)");
  constexpr int VEC_END = NT * VEC;
  constexpr int MAX_TAIL = (STORAGE_DIM - VEC_END + NT - 1) / NT;
  int const tid = threadIdx.x;
  int const off = tid * VEC;
  bf16 const *const d_w = static_cast<bf16 const *>(weight_ptr);
  uint64_t const w_lo = *reinterpret_cast<uint64_t const *>(&d_w[off]);
  uint64_t const w_hi = *reinterpret_cast<uint64_t const *>(&d_w[off + 4]);
  float wt[MAX_TAIL];
#pragma unroll
  for (int t = 0; t < MAX_TAIL; t++) {
    int const i = VEC_END + tid + t * NT;
    wt[t] = i < STORAGE_DIM ? __bfloat162float(d_w[i]) : 0.f;
  }
  uint64_t xl[ROWS], xh[ROWS];
  float xt[ROWS][MAX_TAIL];
#if defined(MPK_LMT_SL) || defined(MPK_LMT_RG) || defined(MPK_LMT_DRA)
  // MPK_LMT_SL: no branch around a row's loads, so every row's loads issue together. A dead row reads row 0 and a
  // tail lane past the row reads its last element (both in bounds); a select zeroes them, as the branches did.
  // MPK_LMT_RG=k: the same in groups of k rows; a group past the live rows is skipped (its rows stay zero).
#ifdef MPK_LMT_RG
  constexpr int RG = MPK_LMT_RG;
#else
  constexpr int RG = ROWS;
#endif
#pragma unroll
  for (int r = 0; r < ROWS; r++) {
    xl[r] = 0ull;
    xh[r] = 0ull;
#pragma unroll
    for (int t = 0; t < MAX_TAIL; t++) {
      xt[r][t] = 0.f;
    }
  }
#pragma unroll
  for (int g0 = 0; g0 < ROWS; g0 += RG) {
   if (g0 < nrows) {
#pragma unroll
   for (int r = g0; r < (g0 + RG < ROWS ? g0 + RG : ROWS); r++) {
    bool const live = r < nrows;
    int const rs = live ? r : g0;
    if constexpr (RA) {
      auto f2bf16 = [](float f) -> unsigned {
        unsigned const u = __float_as_uint(f);
        return (u + (((u >> 16) & 1u) + 0x7FFFu)) >> 16;
      };
      float const *const wr = ws + (rs * 4) * STORAGE_DIM;
      unsigned short const *const rr = in + rs * STORAGE_DIM;
      float4 a0, a1;
      __builtin_memcpy(&a0, wr + off, 16);
      __builtin_memcpy(&a1, wr + off + 4, 16);
#pragma unroll
      for (int sl = 1; sl < 4; sl++) {
        float4 b0, b1;
        __builtin_memcpy(&b0, wr + sl * STORAGE_DIM + off, 16);
        __builtin_memcpy(&b1, wr + sl * STORAGE_DIM + off + 4, 16);
        a0.x += b0.x;
        a0.y += b0.y;
        a0.z += b0.z;
        a0.w += b0.w;
        a1.x += b1.x;
        a1.y += b1.y;
        a1.z += b1.z;
        a1.w += b1.w;
      }
      uint64_t rl, rh;
      __builtin_memcpy(&rl, rr + off, 8);
      __builtin_memcpy(&rh, rr + off + 4, 8);
      float const av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
      uint64_t lo = 0ull, hi = 0ull;
#pragma unroll
      for (int e = 0; e < 4; e++) {
        float const s0 = av[e] + __uint_as_float((unsigned)((rl >> (16 * e)) & 0xFFFFull) << 16);
        float const s1 = av[4 + e] + __uint_as_float((unsigned)((rh >> (16 * e)) & 0xFFFFull) << 16);
        lo |= (uint64_t)f2bf16(s0) << (16 * e);
        hi |= (uint64_t)f2bf16(s1) << (16 * e);
      }
      xl[r] = live ? lo : 0ull;
      xh[r] = live ? hi : 0ull;
#pragma unroll
      for (int t = 0; t < MAX_TAIL; t++) {
        int const i = VEC_END + tid + t * NT;
        int const ic = i < STORAGE_DIM ? i : STORAGE_DIM - 1;
        float a = wr[ic];
        a += wr[STORAGE_DIM + ic];
        a += wr[2 * STORAGE_DIM + ic];
        a += wr[3 * STORAGE_DIM + ic];
        float const sv = a + __uint_as_float((unsigned)rr[ic] << 16);
        xt[r][t] = (live && i < STORAGE_DIM) ? __uint_as_float(f2bf16(sv) << 16) : 0.f;
      }
    } else {
      bf16 const *const x = reinterpret_cast<bf16 const *>(in) + rs * STORAGE_DIM;
      uint64_t const l = *reinterpret_cast<uint64_t const *>(&x[off]);
      uint64_t const h = *reinterpret_cast<uint64_t const *>(&x[off + 4]);
      xl[r] = live ? l : 0ull;
      xh[r] = live ? h : 0ull;
#pragma unroll
      for (int t = 0; t < MAX_TAIL; t++) {
        int const i = VEC_END + tid + t * NT;
        int const ic = i < STORAGE_DIM ? i : STORAGE_DIM - 1;
        float const v = __bfloat162float(x[ic]);
        xt[r][t] = (live && i < STORAGE_DIM) ? v : 0.f;
      }
    }
  }
   }
  }
#else
#pragma unroll
  for (int r = 0; r < ROWS; r++) {
    xl[r] = 0ull;
    xh[r] = 0ull;
#pragma unroll
    for (int t = 0; t < MAX_TAIL; t++) {
      xt[r][t] = 0.f;
    }
    if (r < nrows) {
      if constexpr (RA) {
        auto f2bf16 = [](float f) -> unsigned {
          unsigned const u = __float_as_uint(f);
          return (u + (((u >> 16) & 1u) + 0x7FFFu)) >> 16;
        };
        float const *const wr = ws + (r * 4) * STORAGE_DIM;
        unsigned short const *const rr = in + r * STORAGE_DIM;
        float4 a0, a1;
        __builtin_memcpy(&a0, wr + off, 16);
        __builtin_memcpy(&a1, wr + off + 4, 16);
#pragma unroll
        for (int sl = 1; sl < 4; sl++) {
          float4 b0, b1;
          __builtin_memcpy(&b0, wr + sl * STORAGE_DIM + off, 16);
          __builtin_memcpy(&b1, wr + sl * STORAGE_DIM + off + 4, 16);
          a0.x += b0.x;
          a0.y += b0.y;
          a0.z += b0.z;
          a0.w += b0.w;
          a1.x += b1.x;
          a1.y += b1.y;
          a1.z += b1.z;
          a1.w += b1.w;
        }
        uint64_t rl, rh;
        __builtin_memcpy(&rl, rr + off, 8);
        __builtin_memcpy(&rh, rr + off + 4, 8);
        float const av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        uint64_t lo = 0ull, hi = 0ull;
#pragma unroll
        for (int e = 0; e < 4; e++) {
          float const s0 = av[e] + __uint_as_float((unsigned)((rl >> (16 * e)) & 0xFFFFull) << 16);
          float const s1 = av[4 + e] + __uint_as_float((unsigned)((rh >> (16 * e)) & 0xFFFFull) << 16);
          lo |= (uint64_t)f2bf16(s0) << (16 * e);
          hi |= (uint64_t)f2bf16(s1) << (16 * e);
        }
        xl[r] = lo;
        xh[r] = hi;
#pragma unroll
        for (int t = 0; t < MAX_TAIL; t++) {
          int const i = VEC_END + tid + t * NT;
          if (i < STORAGE_DIM) {
            float a = wr[i];
            a += wr[STORAGE_DIM + i];
            a += wr[2 * STORAGE_DIM + i];
            a += wr[3 * STORAGE_DIM + i];
            float const sv = a + __uint_as_float((unsigned)rr[i] << 16);
            xt[r][t] = __uint_as_float(f2bf16(sv) << 16);
          }
        }
      } else {
        bf16 const *const x = reinterpret_cast<bf16 const *>(in) + r * STORAGE_DIM;
        xl[r] = *reinterpret_cast<uint64_t const *>(&x[off]);
        xh[r] = *reinterpret_cast<uint64_t const *>(&x[off + 4]);
#pragma unroll
        for (int t = 0; t < MAX_TAIL; t++) {
          int const i = VEC_END + tid + t * NT;
          if (i < STORAGE_DIM) {
            xt[r][t] = __bfloat162float(x[i]);
          }
        }
      }
    }
  }
#endif  // MPK_LMT_SL
  float sum[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; r++) {
    float s = 0.0f;
    bf16 const *lo = reinterpret_cast<bf16 const *>(&xl[r]);
    bf16 const *hi = reinterpret_cast<bf16 const *>(&xh[r]);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float vlo = __bfloat162float(lo[i]);
      s += vlo * vlo;
      float vhi = __bfloat162float(hi[i]);
      s += vhi * vhi;
    }
#pragma unroll
    for (int t = 0; t < MAX_TAIL; t++) {
      if (VEC_END + tid + t * NT < STORAGE_DIM) {
        float val = xt[r][t];
        s += val * val;
      }
    }
    if (r < nrows) {  // uniform: dead rows skip the reduction (their sum is 0 and unused)
#ifdef MPK_RMSNORM_DPP_REDUCE
      rmsnorm_wave_sum_to_lane_zero(s);
#else
#pragma unroll
      for (int offset = 32; offset > 0; offset >>= 1) {
        s += __shfl_xor(s, offset);
      }
#endif
    }
    sum[r] = s;
  }
#ifdef MPK_LMT_TS
  if (ts && tid == 0) g_lmt_ts[5] = __builtin_amdgcn_s_memrealtime();
#endif
  int const wave = tid >> 6, lane = tid & 63;
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < ROWS; r++) {
      red[wave * ROWS + r] = sum[r];
    }
  }
  __syncthreads();
  if (wave == 0) {  // lane 4r + w holds wave w's partial of row r; the xor-2 / xor-1 steps stay inside each 4-lane group
    int const r = lane >> 2, w = lane & 3;
    float v = lane < 4 * ROWS ? red[w * ROWS + r] : 0.0f;
    v += __shfl_xor(v, 2);
    v += __shfl_xor(v, 1);
    if (w == 0 && lane < 4 * ROWS) {
      red[4 * ROWS + r] = v;
    }
  }
  __syncthreads();
#ifdef MPK_LMT_TS
  if (ts && tid == 0) g_lmt_ts[6] = __builtin_amdgcn_s_memrealtime();
#endif
  bf16 const *w_lo_a = reinterpret_cast<bf16 const *>(&w_lo);
  bf16 const *w_hi_a = reinterpret_cast<bf16 const *>(&w_hi);
#ifdef MPK_LMT_FQ
  if constexpr (FQ) {
    // FQ: _gang_multirow_fp8_quant's bytes, computed from the registers. It takes each 128-element super-block's
    // amax over the bf16-rounded values (fmaxf from 0), an E8M0 scale and a per-element cvt with it; the same per
    // element here gives the same bytes. A VEC super-block is one 16-lane row (threads 16s..16s+15); a tail
    // super-block 16 + 2t + h is 128 threads (h = tid >> 7) holding one element each, so its amax crosses two waves
    // through `amx`. Dead rows are written as zeros (what the staged path quantized).
    float tv[ROWS][MAX_TAIL];
#pragma unroll
    for (int r = 0; r < ROWS; r++) {
      uint8_t *const q = tok_fp8 + r * TRS;
      float m = 0.0f;
      float v[VEC];
      if (r < nrows) {
        float const rms_rcp = rsqrtf(red[4 * ROWS + r] / float(ACTUAL_HIDDEN_DIM) + 1e-5f);
        bf16 const *in_lo_a = reinterpret_cast<bf16 const *>(&xl[r]);
        bf16 const *in_hi_a = reinterpret_cast<bf16 const *>(&xh[r]);
#pragma unroll
        for (int i = 0; i < 4; i++) {
          v[i] = __bfloat162float(__float2bfloat16(__bfloat162float(in_lo_a[i]) * rms_rcp * __bfloat162float(w_lo_a[i])));
          v[4 + i] =
              __bfloat162float(__float2bfloat16(__bfloat162float(in_hi_a[i]) * rms_rcp * __bfloat162float(w_hi_a[i])));
        }
#pragma unroll
        for (int t = 0; t < MAX_TAIL; t++) {
          int const i = VEC_END + tid + t * NT;
          tv[r][t] = i < STORAGE_DIM ? __bfloat162float(__float2bfloat16(xt[r][t] * rms_rcp * wt[t])) : 0.0f;
        }
      } else {
#pragma unroll
        for (int i = 0; i < VEC; i++) {
          v[i] = 0.0f;
        }
#pragma unroll
        for (int t = 0; t < MAX_TAIL; t++) {
          tv[r][t] = 0.0f;
        }
      }
#pragma unroll
      for (int i = 0; i < VEC; i++) {
        m = fmaxf(m, fabsf(v[i]));
      }
      m = lmt_rowmax16(m);
      uint8_t const se = _gang_compute_e8m0_fp8(m);
      float const sf = lmt_e8m0_scale(se);
      fp8x4_t p0 = {}, p1 = {};
      p0 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p0, v[0], v[1], sf, false);
      p0 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p0, v[2], v[3], sf, true);
      p1 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p1, v[4], v[5], sf, false);
      p1 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(p1, v[6], v[7], sf, true);
      uint2 pk;
      pk.x = *reinterpret_cast<unsigned const *>(&p0);
      pk.y = *reinterpret_cast<unsigned const *>(&p1);
      *reinterpret_cast<uint2 *>(q + off) = pk;
      if ((tid & 15) == 0) {
        tok_sc[r * SCS + (tid >> 4)] = se;
      }
#pragma unroll
      for (int t = 0; t < MAX_TAIL; t++) {
        float const wm = lmt_wavemax(fabsf(tv[r][t]));
        if (lane == 0) {
          amx[(wave * ROWS + r) * MAX_TAIL + t] = wm;
        }
      }
    }
    __syncthreads();
    int const h = tid >> 7;
#pragma unroll
    for (int r = 0; r < ROWS; r++) {
      uint8_t *const q = tok_fp8 + r * TRS;
#pragma unroll
      for (int t = 0; t < MAX_TAIL; t++) {
        int const i = VEC_END + tid + t * NT;
        if (i < STORAGE_DIM) {
          float const bm = fmaxf(amx[((2 * h) * ROWS + r) * MAX_TAIL + t], amx[((2 * h + 1) * ROWS + r) * MAX_TAIL + t]);
          uint8_t const se = _gang_compute_e8m0_fp8(bm);
          fp8x4_t pt = {};
          pt = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(pt, tv[r][t], 0.0f, lmt_e8m0_scale(se), false);
          q[i] = (uint8_t)(*reinterpret_cast<unsigned const *>(&pt) & 0xFFu);
          if ((tid & 127) == 0) {
            tok_sc[r * SCS + (i >> 7)] = se;
          }
        }
      }
    }
  } else
#endif  // MPK_LMT_FQ
  {
#pragma unroll
  for (int r = 0; r < ROWS; r++) {
    unsigned short *const o = rows + r * STORAGE_DIM;
    if (r < nrows) {
      float const rms_rcp = rsqrtf(red[4 * ROWS + r] / float(ACTUAL_HIDDEN_DIM) + 1e-5f);
      bf16 const *in_lo_a = reinterpret_cast<bf16 const *>(&xl[r]);
      bf16 const *in_hi_a = reinterpret_cast<bf16 const *>(&xh[r]);
      bf16 out[VEC];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        out[i] = __float2bfloat16(__bfloat162float(in_lo_a[i]) * rms_rcp * __bfloat162float(w_lo_a[i]));
        out[4 + i] = __float2bfloat16(__bfloat162float(in_hi_a[i]) * rms_rcp * __bfloat162float(w_hi_a[i]));
      }
      *reinterpret_cast<uint64_t *>(&o[off]) = *reinterpret_cast<uint64_t *>(&out[0]);
      *reinterpret_cast<uint64_t *>(&o[off + 4]) = *reinterpret_cast<uint64_t *>(&out[4]);
#pragma unroll
      for (int t = 0; t < MAX_TAIL; t++) {
        int const i = VEC_END + tid + t * NT;
        if (i < STORAGE_DIM) {
          bf16 const b = __float2bfloat16(xt[r][t] * rms_rcp * wt[t]);
          o[i] = *reinterpret_cast<unsigned short const *>(&b);
        }
      }
    } else {
      *reinterpret_cast<uint64_t *>(&o[off]) = 0ull;
      *reinterpret_cast<uint64_t *>(&o[off + 4]) = 0ull;
#pragma unroll
      for (int t = 0; t < MAX_TAIL; t++) {
        int const i = VEC_END + tid + t * NT;
        if (i < STORAGE_DIM) {
          o[i] = 0;
        }
      }
    }
  }
  }
  __syncthreads();
#ifdef MPK_LMT_TS
  if (ts && tid == 0) g_lmt_ts[7] = __builtin_amdgcn_s_memrealtime();
#endif
}
#endif

#ifdef MPK_LMT_DRA
#if !defined(MPK_LMT_NORM) || !defined(MPK_LM_RESADD) || !defined(MPK_AID_NC_REP)
#error "MPK_LMT_DRA: the MPK_LMT_NORM residual add (MPK_LM_RESADD) with the MPK_AID_NC_REP scratch"
#endif
// MPK_LMT_DRA: in each AID half of the NC scratch, one XCD-local barrier counter per XCD (256 B apart, in the
// per-launch clear, between the O-proj and argmax-key regions) and a bf16 copy of the rows per XCD (past the
// router / row slots).
constexpr int MPK_NC_DRA_CNT_INTS = 10240;
constexpr int MPK_NC_DRA_INTS = 1048576;
static_assert(MPK_NC_OPROJ_INTS + 8 * 16 <= MPK_NC_DRA_CNT_INTS && MPK_NC_DRA_CNT_INTS + 4 * 64 <= 12288,
              "MPK_LMT_DRA counters overlap a neighbouring NC region or leave the per-launch clear");
// Rank k of the XCD's nranks LM-head workers: elements [k * slice, k * slice + slice) of every live row, the
// in-kernel residual add's expression and rounding, into the XCD's copy; then wait for the other workers.
template <int DIM, int ROWS>
__device__ __forceinline__ unsigned short const *lmt_dra_resadd(float const *ws, void const *res_v, int nrows,
                                                                int rank, int nranks) {
  static_assert(MPK_NC_DRA_INTS + 4 * ROWS * DIM / 2 <= (32 << 20) / 4, "MPK_LMT_DRA rows inside the NC half");
  int xcc;
  MPK_GET_XCC_ID16(xcc);
  int *const nc = g_aid_nc_rep[(xcc & 7) >> 2];
  unsigned short *const buf = reinterpret_cast<unsigned short *>(nc + MPK_NC_DRA_INTS) + (xcc & 3) * ROWS * DIM;
  int *const cnt = nc + MPK_NC_DRA_CNT_INTS + (xcc & 3) * 64;
  unsigned short const *const res = static_cast<unsigned short const *>(res_v);
  int const slice = (DIM + nranks - 1) / nranks;
  for (int idx = threadIdx.x; idx < ROWS * slice; idx += blockDim.x) {
    int const r = idx / slice;
    int const e = rank * slice + (idx - r * slice);
    if (r < nrows && e < DIM) {
      float const *const w = ws + (r * 4) * DIM + e;
      float a = w[0];
      a += w[DIM];
      a += w[2 * DIM];
      a += w[3 * DIM];
      float const sv = a + __uint_as_float((unsigned)res[r * DIM + e] << 16);
      unsigned const u = __float_as_uint(sv);
      buf[r * DIM + e] = (unsigned short)((u + (((u >> 16) & 1u) + 0x7FFFu)) >> 16);
    }
  }
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");  // this wave's stores are in the XCD's L2
  __syncthreads();
  if (threadIdx.x == 0) {
    int const old = __hip_atomic_fetch_add(cnt, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    int const target = (old / nranks + 1) * nranks;  // nranks arrivals per pass, the counter cleared per launch
    while (__hip_atomic_load(cnt, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < target) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
  __syncthreads();
  asm volatile("buffer_inv sc0" ::: "memory");  // this CU's L1 may hold the previous pass's rows
  return buf;
}
#endif

template <int BATCH_SIZE,
          int OUTPUT_PER_WG,
          int REDUCTION_SIZE,
          int ACTUAL_HIDDEN_DIM = REDUCTION_SIZE>
#ifdef MPK_LMHEAD_INLINE
__device__ __forceinline__ void
#else
__device__ __noinline__ void
#endif
gang_rmsnorm_linear_mxfp4_bias_argmax_kernel(
    void const *norm_input_ptr,
    void const *norm_weight_ptr,
    void *norm_output_ptr,
    void const *weight_ptr,
    void const *bias_ptr,
    void *argmax_val_ptr, // [num_workers] bf16: per-worker max value
    void *argmax_idx_ptr, // [num_workers] int64: per-worker absolute index
    void *logits_out_ptr, // [max_seq_length, output_stride] f32, or nullptr.
                          // Perplexity mode only: when non-null the full logit
                          // row is also written to HBM at row `step`. Null on
                          // the serving path, which keeps logits in registers
                          // and pays no HBM traffic.
    int step,             // row of logits_out_ptr to write (ignored if null)
    int num_active_tokens,
    int n_wgs_per_xcd,
    int workers_per_xcd,
    int output_stride,
    int tile_idx) { // tile_idx = partition_index * workers_per_xcd +
                    // worker_rank

  static_assert(OUTPUT_PER_WG >= 64,
                "Argmax fusion only supports N-parallel path (OPW>=64)");
  static_assert(OUTPUT_PER_WG % 16 == 0);
  static_assert(REDUCTION_SIZE % 128 == 0);

  constexpr int NUM_BLOCKS_32 = REDUCTION_SIZE / 32;
  constexpr int WG_DATA_BYTES = OUTPUT_PER_WG * (REDUCTION_SIZE / 2);
  constexpr int WG_SCALE_BYTES = OUTPUT_PER_WG * NUM_BLOCKS_32;
  constexpr int WG_BYTES = WG_DATA_BYTES + WG_SCALE_BYTES;
  constexpr int K_PER_MFMA = 128;
  constexpr int MFMA_ITERS = REDUCTION_SIZE / K_PER_MFMA;
  constexpr int NUM_WAVES = 4;
  constexpr int TILES_PER_WAVE = OUTPUT_PER_WG / 16 / NUM_WAVES;

  // Batch tiling over the MFMA N dimension.
  constexpr int MFMA_N = 16;
  constexpr int TOK_ROWS = BATCH_SIZE < MFMA_N ? BATCH_SIZE : MFMA_N;
  constexpr int NUM_BBLK = (BATCH_SIZE + TOK_ROWS - 1) / TOK_ROWS;

  // LDS weight tile layout (padded for buffer_load_lds alignment)
  constexpr int QKV_TILE_ROWS = 16;
  constexpr int QKV_TILE_DATA = QKV_TILE_ROWS * (REDUCTION_SIZE / 2);
  constexpr int QKV_TILE_SCALE = QKV_TILE_ROWS * NUM_BLOCKS_32;
  constexpr int qkv_n16_data = QKV_TILE_DATA / 16;
  constexpr int QKV_LPT = (qkv_n16_data + 255) / 256;
  constexpr int QKV_TILE_DATA_PADDED = QKV_LPT * 256 * 16;
  constexpr int QKV_TILE_BYTES = QKV_TILE_DATA_PADDED + QKV_TILE_SCALE;
#ifdef MPK_LM_HEAD_GROUP_PIPELINE
  constexpr int QKV_SCALE_TILE_PADDED =
      ((QKV_TILE_SCALE + 1023) / 1024) * 1024;
  constexpr int QKV_SCALE_PIPELINE_BYTES = QKV_SCALE_TILE_PADDED * NUM_WAVES;
#else
  constexpr int QKV_SCALE_PIPELINE_BYTES = 0;
#endif

  // FP8 token rows sit at the start of LDS, one row per MFMA N column.
  // The +16 pad keeps consecutive rows off one bank group: the 16 lanes of a
  // g-group each read 16 B at TOK_ROW_STRIDE spacing, and an unpadded 2944 B
  // stride (736 dwords, 736 % 32 == 0) would serialize all 16 into one bank.
  constexpr int TOK_ROW_STRIDE = REDUCTION_SIZE + 16;
  constexpr int SC_STRIDE = ((MFMA_ITERS + 3) / 4) * 4;
  constexpr int TOK_REGION = TOK_ROWS * TOK_ROW_STRIDE;
  constexpr int SC_REGION = TOK_ROWS * SC_STRIDE;
  // Cross-wave argmax scratch: NUM_WAVES x MFMA_N of (float val, int idx).
  constexpr int RED_OFF = ((TOK_REGION + SC_REGION + 15) / 16) * 16;
  constexpr int RED_REGION = NUM_WAVES * MFMA_N * 8;
  constexpr int QKV_LDS_OFF = ((RED_OFF + RED_REGION + 15) / 16) * 16;
  static_assert(QKV_LDS_OFF + QKV_TILE_BYTES * NUM_WAVES +
                            2 * QKV_SCALE_PIPELINE_BYTES <=
                    mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE -
                        mirage::runtime::LAYER_IDX_SMEM_OFFSET_FROM_END,
                "LM-head LDS exceeds the MI350X dynamic LDS budget");
  static_assert(TOK_ROW_STRIDE % 16 == 0,
                "token row stride must keep the i32x4 B-operand loads aligned");
#ifdef MPK_LM_NORM_LDS
  // The normalized rows get their own block past the weight staging, so no
  // region they share is live at the same time.
  constexpr int LM_NORM_OFF = ((QKV_LDS_OFF + QKV_TILE_BYTES * NUM_WAVES +
                                2 * QKV_SCALE_PIPELINE_BYTES + 15) /
                               16) *
                              16;
  static_assert(NUM_BBLK == 1, "MPK_LM_NORM_LDS stages every batch row at once");
  static_assert(LM_NORM_OFF + BATCH_SIZE * REDUCTION_SIZE * 2 <=
                    mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE -
                        mirage::runtime::LAYER_IDX_SMEM_OFFSET_FROM_END,
                "MPK_LM_NORM_LDS: the normalized rows exceed the LDS budget");
#endif
#ifdef MPK_LM_HEAD_GROUP_PIPELINE
#ifdef MPK_VT2
  static_assert(BATCH_SIZE <= MPK_ROP_NTOK,
                "MPK_LM_HEAD_GROUP_PIPELINE: 1 or 2 rows (4 with MPK_VTN_N7B)");
#else
  static_assert(BATCH_SIZE == 1,
                "MPK_LM_HEAD_GROUP_PIPELINE is batch-1 only");
#endif
  static_assert(REDUCTION_SIZE == 2944,
                "MPK_LM_HEAD_GROUP_PIPELINE requires K=2944");
  static_assert(OUTPUT_PER_WG == 64,
                "MPK_LM_HEAD_GROUP_PIPELINE requires OPW=64");
  static_assert(NUM_WAVES == 4 && TILES_PER_WAVE == 1,
                "MPK_LM_HEAD_GROUP_PIPELINE requires four waves, one tile each");
  static_assert(QKV_TILE_DATA == MFMA_ITERS * 1024,
                "each K128 fragment must be one 64-lane 16-byte request");
#endif

#if defined(MPK_LM_RESADD) && !defined(MPK_LMT_NORM)
#ifdef MPK_LM_NORM_LDS
  constexpr int LM_RESADD_OFF =
      LM_NORM_OFF + ((BATCH_SIZE * REDUCTION_SIZE * 2 + 15) / 16) * 16;
#else
  constexpr int LM_RESADD_OFF = ((QKV_LDS_OFF + QKV_TILE_BYTES * NUM_WAVES +
                                  2 * QKV_SCALE_PIPELINE_BYTES + 15) /
                                 16) *
                                16;
#endif
  static_assert(LM_RESADD_OFF + BATCH_SIZE * REDUCTION_SIZE * 2 <=
                    mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE -
                        mirage::runtime::LAYER_IDX_SMEM_OFFSET_FROM_END,
                "MPK_LM_RESADD: the residual rows exceed the LDS budget");
  static_assert(REDUCTION_SIZE % 4 == 0, "MPK_LM_RESADD works in float4");
#endif
  uint8_t const *W = (uint8_t const *)weight_ptr;
#ifdef MPK_LMNORM_AID
  // One normalized-row copy per memory range. Every worker writes the same
  // values, so no publish is needed: each AID's workers fill and read their
  // own copy, halving both the write and the read fan-in and keeping both
  // local. BATCH_SIZE rows per copy.
  unsigned x_lmn;
  MPK_GET_XCC_ID32(x_lmn);
  long long const lmnorm_aid_off =
      ((int)(x_lmn & 0x7) >= (MPK_NUM_XCDS / 2))
          ? (long long)MPK_MAX_NUM_BATCHED_TOKENS * REDUCTION_SIZE
          : 0;
#else
  long long const lmnorm_aid_off = 0;
#endif
  unsigned short const *d_bias = (unsigned short const *)bias_ptr;

  extern __shared__ char _rnlm_smem[];
  uint8_t *s_tok_fp8 = (uint8_t *)_rnlm_smem;
  uint8_t *s_tok_scales = s_tok_fp8 + TOK_REGION;
  float *s_red_val = (float *)((uint8_t *)_rnlm_smem + RED_OFF);
  int *s_red_idx = (int *)(s_red_val + NUM_WAVES * MFMA_N);
  uint8_t *qkv_lds_w = (uint8_t *)_rnlm_smem + QKV_LDS_OFF;
#ifdef MPK_LM_HEAD_GROUP_PIPELINE
  uint8_t *scale_pipeline_lds = qkv_lds_w + QKV_TILE_BYTES * NUM_WAVES;
#endif
#ifdef MPK_LM_PF0
  using LmPf0L = LmPf0Layout<BATCH_SIZE, OUTPUT_PER_WG, REDUCTION_SIZE>;
  static_assert(LmPf0L::LDS_OFF == QKV_LDS_OFF && LmPf0L::RED_OFF == RED_OFF &&
                    LmPf0L::TILE_BYTES == QKV_TILE_BYTES &&
                    LmPf0L::TILE_DATA_PADDED == QKV_TILE_DATA_PADDED &&
                    LmPf0L::TILE_DATA == QKV_TILE_DATA &&
                    LmPf0L::TILE_SCALE == QKV_TILE_SCALE &&
                    LmPf0L::WG_BYTES == WG_BYTES &&
                    LmPf0L::WG_DATA_BYTES == WG_DATA_BYTES,
                "MPK_LM_PF0: prefetch layout differs from the kernel's");
  // Group zero of this tile is already in (or on its way to) the LDS tiles.
  bool const lm_pf0 =
      *(volatile int const *)((uint8_t *)_rnlm_smem + RED_OFF) == tile_idx + 1;
#endif

  int const tid = threadIdx.x;
  int const warp_id = tid >> 6;
  int const lane_id = tid & 63;
  int const col = lane_id & 15; // MFMA N index == token within block
  int const g = lane_id >> 4;   // MFMA k-block / output-row group

  unsigned short *argmax_vals =
      reinterpret_cast<unsigned short *>(argmax_val_ptr);
  long long *argmax_idxs = reinterpret_cast<long long *>(argmax_idx_ptr);

  int batch_count =
      (num_active_tokens < BATCH_SIZE) ? num_active_tokens : BATCH_SIZE;
#ifdef MPK_EAGLE3
#ifdef MPK_EAGLE3_TS
  if (tile_idx == 0 && threadIdx.x == 0) g_e3lm_ts[0] = __builtin_amdgcn_s_memrealtime();
#endif
#ifdef MPK_E3_STATE1
  E3St const e3st = e3_state();
  mpk_eagle3_features(tile_idx, e3st);
#else
  mpk_eagle3_features(tile_idx);
#endif
#ifdef MPK_EAGLE3_TS
  if (tile_idx == 0 && threadIdx.x == 0) g_e3lm_ts[1] = __builtin_amdgcn_s_memrealtime();
#endif
#endif
#ifdef MPK_DFLASH
  DfSt const dfst = df_state();
  mpk_dflash_features(tile_idx, dfst);
#endif
  int partition_index = tile_idx / workers_per_xcd; // logical XCD (0..7)
  int worker_rank = tile_idx % workers_per_xcd;     // 0..workers_per_xcd-1
#ifdef MPK_LM_HW_XCC2
  // Bind the weight slice to the PHYSICAL XCC. partition_index above is the
  // LOGICAL id and the imap handed this worker slice `partition_index`, but
  // block b runs on physical xcd (b-1)%8, so without this the 315 MB vocab
  // stream is homed in a neighbouring AID -- while MPK_LMNORM_AID above
  // already binds its norm copy to the hardware id. Same kernel, two
  // different bindings.
  {
    unsigned _x_hw;
    MPK_GET_XCC_ID32(_x_hw);
    int const _hw = (int)(_x_hw & (unsigned)(MPK_NUM_XCDS - 1));
    W += (int64_t)((_hw - partition_index) & (MPK_NUM_XCDS - 1)) *
         (int64_t)n_wgs_per_xcd * (int64_t)WG_BYTES;
    partition_index = _hw;
  }
#endif
  int partition_start = partition_index * n_wgs_per_xcd * OUTPUT_PER_WG;
  // argmax_part_* are [batch, num_workers], XCD-partitioned along dim 1, so
  // the pointer already points at this XCD's worker slice while the row
  // stride is still the full worker count.
  int const argmax_row_stride = workers_per_xcd * MPK_NUM_XCDS;

#ifdef MPK_LM_INNER
  // Function scope on purpose: _lmi_t1/_lmi_t2 are written inside the bblk
  // loop below and read after it.
  unsigned long long _lmi_t0 = 0, _lmi_t1 = 0, _lmi_t2 = 0;
  if (tid == 0) {
    asm volatile("s_memrealtime %0" : "=s"(_lmi_t0));
  }
#endif
#ifdef MPK_LMX
  unsigned long long const _lmx_t0 = __builtin_amdgcn_s_memrealtime();
#endif
#ifdef MPK_LM_STAMPS
  unsigned long long lmst[20];
  int lmst_n = 0;
  if (lmst_n < 20) lmst[lmst_n++] = __builtin_amdgcn_s_memrealtime();
#endif

  // ── Step 1: RMSNorm ─────────────────────────────────────────────────────
  // DO NOT add a __syncthreads() to this loop body. Measured at bs=2: without
  // it the run completes and both requests decode correct prose; with it the
  // kernel deadlocks before a single task retires (tasks_done=0), and that
  // holds for BOTH a runtime bound (batch_count) and a compile-time one
  // (BATCH_SIZE), so it is the barrier and not the trip count.
  // rmsnorm_inline_amd already ends with its own cross-wave barrier pair; an
  // extra one here is what breaks it.
#if defined(MPK_LM_RESADD) && !defined(MPK_LMT_NORM)
  // Final residual add, done here instead of in its own task: norm_input is
  // the residual and logits_out_ptr carries the MoE f32 workspace. Same loop
  // shape, slot order and rounding as moe_residual_add_f32_mi300, so the row
  // is bit-identical to that task's output. Staged in LDS for the RMSNorm.
  // The worker's dependency check already ran an agent-scope acquire.
  float const *lm_ws = static_cast<float const *>(logits_out_ptr);
  logits_out_ptr = nullptr;
  unsigned short *lm_x =
      (unsigned short *)((uint8_t *)_rnlm_smem + LM_RESADD_OFF);
  for (int b = 0; b < batch_count; b++) {
    float const *ws_row = lm_ws + (b * 4) * REDUCTION_SIZE;
    unsigned short const *res_row =
        (unsigned short const *)norm_input_ptr + b * REDUCTION_SIZE;
    for (int off = tid * 4; off < REDUCTION_SIZE; off += 256 * 4) {
      float4 ws4;
      __builtin_memcpy(&ws4, ws_row + off, 16);
#pragma unroll
      for (int s = 1; s < 4; s++) {
        float4 slot4;
        __builtin_memcpy(&slot4, ws_row + s * REDUCTION_SIZE + off, 16);
        ws4.x += slot4.x;
        ws4.y += slot4.y;
        ws4.z += slot4.z;
        ws4.w += slot4.w;
      }
      uint2 res_packed;
      __builtin_memcpy(&res_packed, res_row + off, 8);
      float const s0 = ws4.x + __uint_as_float((res_packed.x & 0xFFFFu) << 16);
      float const s1 = ws4.y + __uint_as_float(res_packed.x & 0xFFFF0000u);
      float const s2 = ws4.z + __uint_as_float((res_packed.y & 0xFFFFu) << 16);
      float const s3 = ws4.w + __uint_as_float(res_packed.y & 0xFFFF0000u);
      auto f2bf16 = [](float f) -> unsigned {
        unsigned const u = __float_as_uint(f);
        return (u + (((u >> 16) & 1u) + 0x7FFFu)) >> 16;
      };
      uint2 out_packed;
      out_packed.x = f2bf16(s0) | (f2bf16(s1) << 16);
      out_packed.y = f2bf16(s2) | (f2bf16(s3) << 16);
      __builtin_memcpy(lm_x + b * REDUCTION_SIZE + off, &out_packed, 8);
    }
  }
  __syncthreads();
#endif
#ifdef MPK_LMT_NORM
#ifdef MPK_LMT_FQ
  // FQ writes s_tok_fp8 / s_tok_scales directly; its reduction scratch (5 x rows floats, then the tail amax of
  // 4 waves x rows x 4) is the cross-wave argmax scratch at RED_OFF, idle until the epilogue (the LM_PF0 tag there
  // was read at entry).
  static_assert(NUM_BBLK == 1 && TOK_ROWS == BATCH_SIZE, "MPK_LMT_FQ: one column block");
  static_assert(5 * BATCH_SIZE <= 32 && (32 + 4 * BATCH_SIZE * 4) * 4 <= RED_REGION,
                "MPK_LMT_FQ: the reduction scratch fits the cross-wave argmax scratch");
  float *const lmt_red = reinterpret_cast<float *>((uint8_t *)_rnlm_smem + RED_OFF);
#elif defined(MPK_LMT_GLB)
  // GLB: the rows go to the global norm buffer (as rmsnorm_inline_amd's did) and the reduction scratch (5 x rows
  // floats) to the cross-wave argmax scratch at RED_OFF, idle until the epilogue: the weight tiles stay free.
  static_assert(5 * BATCH_SIZE * 4 <= RED_REGION, "MPK_LMT_GLB: the reduction scratch fits the argmax scratch");
  float *const lmt_red = reinterpret_cast<float *>((uint8_t *)_rnlm_smem + RED_OFF);
#else
  static_assert(BATCH_SIZE * REDUCTION_SIZE * 2 + 5 * BATCH_SIZE * 4 <= QKV_TILE_BYTES * NUM_WAVES,
                "MPK_LMT_NORM: the rows and their reduction scratch fit the weight tiles");
  float *const lmt_red = reinterpret_cast<float *>(qkv_lds_w + BATCH_SIZE * REDUCTION_SIZE * 2);
#endif
#ifdef MPK_LMT_GLB
  unsigned short *const lmt_rows = (unsigned short *)norm_output_ptr + lmnorm_aid_off;
#else
  unsigned short *const lmt_rows = reinterpret_cast<unsigned short *>(qkv_lds_w);
#endif
#ifdef MPK_LM_RESADD
  // The worker's dependency check already ran an agent-scope acquire (as for the LDS-staged form).
  float const *const lm_ws_ra = static_cast<float const *>(logits_out_ptr);
  logits_out_ptr = nullptr;
#ifdef MPK_LMT_DRA
  unsigned short const *const lmt_in =
      lmt_dra_resadd<REDUCTION_SIZE, BATCH_SIZE>(lm_ws_ra, norm_input_ptr, batch_count, worker_rank, workers_per_xcd);
  float const *const lm_ws = nullptr;
  constexpr bool LMT_RA = false;
#else
  unsigned short const *const lmt_in = static_cast<unsigned short const *>(norm_input_ptr);
  float const *const lm_ws = lm_ws_ra;
  constexpr bool LMT_RA = true;
#endif
#else
  unsigned short const *const lmt_in = static_cast<unsigned short const *>(norm_input_ptr);
  float const *const lm_ws = nullptr;
  constexpr bool LMT_RA = false;
#endif
  lmt_rmsnorm_rows<REDUCTION_SIZE, ACTUAL_HIDDEN_DIM, BATCH_SIZE, LMT_RA
#ifdef MPK_LMT_FQ
                   , true, TOK_ROW_STRIDE, SC_STRIDE
#endif
                   >(
      lmt_in, norm_weight_ptr, lmt_rows, lmt_red, batch_count
#ifdef MPK_LMT_TS
      , tile_idx == 0
#else
      , false
#endif
      , lm_ws
#ifdef MPK_LMT_FQ
      , s_tok_fp8, s_tok_scales, lmt_red + 32
#endif
      );
#else
  for (int b = 0; b < batch_count; b++) {
#ifdef MPK_LM_RESADD
    unsigned short const *row_in = lm_x + b * REDUCTION_SIZE;
#else
    unsigned short const *row_in =
        (unsigned short const *)norm_input_ptr + (long long)b * REDUCTION_SIZE;
#endif
#ifdef MPK_LM_NORM_LDS
    unsigned short *row_out = (unsigned short *)((uint8_t *)_rnlm_smem + LM_NORM_OFF) + b * REDUCTION_SIZE;
#else
    unsigned short *row_out =
        (unsigned short *)norm_output_ptr + lmnorm_aid_off +
        (long long)b * REDUCTION_SIZE;
#endif
    gang_rmsnorm_detail::rmsnorm_inline_amd<REDUCTION_SIZE, ACTUAL_HIDDEN_DIM>(
        row_in, norm_weight_ptr, row_out);
  }
#endif  // MPK_LMT_NORM

#ifdef MPK_LM_STAMPS
  if (lmst_n < 20) lmst[lmst_n++] = __builtin_amdgcn_s_memrealtime();
#endif
#ifdef MPK_LMT_TS
  if (tile_idx == 0 && threadIdx.x == 0) g_lmt_ts[0] = __builtin_amdgcn_s_memrealtime();
#endif
  // SRD for weight buffer (hoisted outside every loop)
  uint32_t qkv_buf_range = static_cast<uint32_t>(n_wgs_per_xcd) * WG_BYTES;
  i32x4_t qkv_rsrc = make_w_buffer_rsrc(W, qkv_buf_range);

#ifdef MPK_LM_HEAD_WAVE_TILE_DMA
  // Scalar LDS destination for the Phase A weight train, hoisted out of every
  // loop. buffer_load_dwordx4 ... lds takes its LDS base from M0, which is
  // scalar; leaving the destination as a pointer derived from threadIdx makes
  // the compiler re-prove uniformity and emit a v_readfirstlane + s_mov m0 per
  // request. Reading it once here leaves the train nothing to do but
  // s_addk_i32 m0 between loads. Casting through the address_space(3) pointer
  // (32-bit) rather than truncating the generic 64-bit address keeps this off
  // any assumption about where the LDS aperture sits.
  unsigned const lm_lds_wave_base = __builtin_amdgcn_readfirstlane(
      static_cast<unsigned>(reinterpret_cast<uintptr_t>(
          (__attribute__((address_space(3)))
           uint8_t *)(qkv_lds_w + warp_id * QKV_TILE_BYTES))));
#endif

  // ── Batch column-block sweep ────────────────────────────────────────────
  // NUM_BBLK == 1 for bs <= 16, so this collapses away entirely and the
  // bs<=16 path costs exactly what the old single-token path did.
#pragma unroll 1
  for (int bblk = 0; bblk < NUM_BBLK; bblk++) {
    int const tok_row = bblk * MFMA_N + col;
    bool const tok_active = tok_row < batch_count;

    // ── Step 2: FP8 quant of this block's token rows ──────────────────────
#if !defined(MPK_LMT_FQ) && !defined(MPK_LM_BF16)
    _gang_multirow_fp8_quant<REDUCTION_SIZE, TOK_ROWS, BATCH_SIZE,
                             TOK_ROW_STRIDE, SC_STRIDE>(
#ifdef MPK_LM_NORM_LDS
        (unsigned short const *)(unsigned short *)((uint8_t *)_rnlm_smem + LM_NORM_OFF), REDUCTION_SIZE, bblk * MFMA_N,
#elif defined(MPK_LMT_NORM) && !defined(MPK_LMT_GLB)
        (unsigned short const *)lmt_rows, REDUCTION_SIZE, bblk * MFMA_N,
#else
        (unsigned short const *)norm_output_ptr + lmnorm_aid_off,
        REDUCTION_SIZE, bblk * MFMA_N,
#endif
        TOK_ROWS, s_tok_fp8, s_tok_scales);
#endif  // MPK_LMT_FQ
#ifdef MPK_LMT_TS
  if (bblk == 0 && tile_idx == 0 && threadIdx.x == 0) g_lmt_ts[1] = __builtin_amdgcn_s_memrealtime();
#endif

    // Each lane owns exactly one token (N column `col`), so the running
    // argmax stays a scalar per lane.
    float thread_max = -1e30f;
    int thread_max_idx = -1;

    // B operand base for this lane's token. Inactive lanes clamp to row 0 to
    // keep their LDS reads in range; their results are discarded below.
    uint8_t const *b_data = s_tok_fp8 + (tok_active ? col : 0) * TOK_ROW_STRIDE;
    uint8_t const *b_scale =
        s_tok_scales + (tok_active ? col : 0) * SC_STRIDE;

    // ── Step 3: Internal tile loop + argmax ────────────────────────────────
#ifdef MPK_LM_INNER
    // Boundary: rmsnorm + FP8 quant complete, the weight stream begins here.
    if (tid == 0 && bblk == 0) {
      asm volatile("s_memrealtime %0" : "=s"(_lmi_t1));
    }
#endif
#ifdef MPK_LM_BF16
    // -- BF16 LM head -------------------------------------------------------
    // The host shuffles the BF16 weight into K-major records: record (wg, tile t,
    // step s) holds the 64 lanes' A fragments of rows wg * 64 + 16 t + (lane & 15),
    // K = 32 s + 8 (lane >> 4) .. +8, so a wave reads 1 KiB contiguous per step.
    // Wave t streams tile t through v_mfma_f32_16x16x32_bf16, whose D layout is
    // the scaled MFMA's (lane (g, col): rows 4 g + i of token col). The ring holds
    // LMB_D steps; its last LMB_D refills fetch the next group's first steps, so
    // LMB_D must divide the step count for the slots to line up across groups.
    {
      constexpr int LMB_K = REDUCTION_SIZE / 32;
      constexpr int LMB_D = 23;
      static_assert(LMB_K % LMB_D == 0, "MPK_LM_BF16: LMB_D must divide K / 32");
      static_assert(OUTPUT_PER_WG == 64 && NUM_WAVES == 4,
                    "MPK_LM_BF16: one 16-row tile per wave");
      typedef unsigned int lmb_u4 __attribute__((ext_vector_type(4)));
      typedef __bf16 lmb_bf8 __attribute__((ext_vector_type(8)));
      typedef float lmb_f4 __attribute__((ext_vector_type(4)));
      typedef __attribute__((address_space(1))) lmb_u4 const lmb_gu4;
      typedef __attribute__((address_space(1))) unsigned long long const lmb_gb8;
      constexpr size_t LMB_WG_BYTES = (size_t)OUTPUT_PER_WG * REDUCTION_SIZE * 2;
#if defined(MPK_LM_NORM_LDS) || (defined(MPK_LMT_NORM) && !defined(MPK_LMT_GLB))
      typedef __attribute__((address_space(3))) unsigned short const lmb_xs;
      typedef __attribute__((address_space(3))) lmb_u4 const volatile lmb_xu4;
#else
      // volatile on global memory would be a system-scope load: plain here
      typedef __attribute__((address_space(1))) unsigned short const lmb_xs;
      typedef __attribute__((address_space(1))) lmb_u4 const lmb_xu4;
#endif
#if defined(MPK_LM_NORM_LDS)
      lmb_xs *const lmb_x0 = (lmb_xs *)((uint8_t *)_rnlm_smem + LM_NORM_OFF);
#elif defined(MPK_LMT_NORM) && !defined(MPK_LMT_GLB)
      lmb_xs *const lmb_x0 = (lmb_xs *)lmt_rows;
#else
      lmb_xs *const lmb_x0 =
          (lmb_xs *)((unsigned short const *)norm_output_ptr + lmnorm_aid_off);
#endif
      lmb_xs *const lmb_x =
          lmb_x0 + (size_t)(tok_active ? tok_row : bblk * MFMA_N) * REDUCTION_SIZE + g * 8;
      auto lmb_rec = [&](int wgi) -> lmb_gu4 * {
        return (lmb_gu4 *)(W + (size_t)wgi * LMB_WG_BYTES +
                           (size_t)warp_id * (LMB_K * 1024)) +
               lane_id;
      };
      lmb_u4 lmb_w[LMB_D];
      int lmb_wg = worker_rank;
      {
        lmb_gu4 *const r0 = lmb_rec(lmb_wg < n_wgs_per_xcd ? lmb_wg : 0);
#pragma unroll
        for (int s = 0; s < LMB_D; ++s) lmb_w[s] = __builtin_nontemporal_load(r0 + s * 64);
      }
      for (; lmb_wg < n_wgs_per_xcd; lmb_wg += workers_per_xcd) {
        int const nwg = lmb_wg + workers_per_xcd;
        lmb_gu4 *const r = lmb_rec(lmb_wg);
        lmb_gu4 *const rn = lmb_rec(nwg < n_wgs_per_xcd ? nwg : lmb_wg);
        // This tile's 4 bias values, loaded before the refills so their wait
        // does not drain the ring (vmcnt retires in issue order).
        unsigned long long const lmb_bias =
            *(lmb_gb8 *)(d_bias + lmb_wg * OUTPUT_PER_WG + warp_id * 16 + g * 4);
        lmb_f4 acc = {0.f, 0.f, 0.f, 0.f};
        // The activation slice is re-read from LDS one step ahead, as volatile
        // loads: it is the same in every group, and hoisted out of the group loop
        // it pins 92 x 4 registers, which pushes the weight ring into AGPRs and
        // turns its waits into vmcnt(0).
        lmb_u4 lmb_xc = *(lmb_xu4 *)lmb_x;
#pragma unroll
        for (int s = 0; s < LMB_K; ++s) {
          lmb_u4 const w = lmb_w[s % LMB_D];
          if (s + LMB_D < LMB_K)
            lmb_w[s % LMB_D] = __builtin_nontemporal_load(r + (s + LMB_D) * 64);
          else
            lmb_w[s % LMB_D] = __builtin_nontemporal_load(rn + (s + LMB_D - LMB_K) * 64);
          lmb_u4 lmb_xn = lmb_xc;
          if (s + 1 < LMB_K) lmb_xn = *(lmb_xu4 *)(lmb_x + (s + 1) * 32);
          lmb_bf8 a, b;
          __builtin_memcpy(&a, &w, 16);
          __builtin_memcpy(&b, &lmb_xc, 16);
          acc = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, acc, 0, 0, 0);
          lmb_xc = lmb_xn;
        }
        if (tok_active) {
#pragma unroll
          for (int i = 0; i < 4; i++) {
            int const rel_idx = warp_id * 16 + g * 4 + i;
            int const abs_idx = partition_start + lmb_wg * OUTPUT_PER_WG + rel_idx;
            unsigned const bt = (unsigned)((lmb_bias >> (16 * i)) & 0xffffu) << 16;
            float bv;
            __builtin_memcpy(&bv, &bt, 4);
            float const val = acc[i] + bv;
            if (val > thread_max) {
              thread_max = val;
              thread_max_idx = abs_idx;
            }
            if (logits_out_ptr != nullptr) {
              reinterpret_cast<float *>(
                  logits_out_ptr)[(long long)(step + tok_row) * output_stride + abs_idx] = val;
            }
          }
        }
      }
    }
#else
    int group_iteration = 0;
    for (int wg_idx = worker_rank; wg_idx < n_wgs_per_xcd;
         wg_idx += workers_per_xcd, ++group_iteration) {
      uint32_t qkv_wg_voff = static_cast<uint32_t>(wg_idx) * WG_BYTES;

    // ── Phase A: Prefetch weight data into LDS via buffer_load_dwordx4 lds:1
    // ──
#ifdef MPK_LM_HEAD_GROUP_PIPELINE
      // Group zero remains synchronous. Every later group was streamed into
      // this recycled tile by the preceding group's MFMA loop. Fleet's queue
      // stages TaskDesc only immediately before its dependency acquire; unlike
      // an immutable fixed-rank plan, it has no publication-safe earlier
      // descriptor from which to prefetch group zero. Adding that hook requires
      // a separate scheduler protocol change, so this patch deliberately does
      // not move task-descriptor reads across the dependency loop.
#ifdef MPK_LM_PF0
      if (group_iteration == 0 && !lm_pf0) {
#else
      if (group_iteration == 0) {
#endif
#endif
#ifdef MPK_LM_HEAD_WAVE_TILE_DMA
      // ── One wave, one tile, one instruction train ─────────────────────────
      //
      // The striped form below has all four waves interleave 16-byte slots
      // across all four tiles: 24 intrinsic calls per thread, each recomputing
      // a clamped index and an LDS destination the compiler has to re-scalarize
      // for M0. But a wave's 64 lanes already cover exactly 1 KiB, the tile is
      // an exact multiple of 1 KiB, and Phase C has wave `w` read tile `w`
      // (TILES_PER_WAVE == 1) -- so the natural unit is the whole tile, issued
      // by its own consumer as one chain of loads that advances the LDS base in
      // M0 and the global offset in a VGPR together. The clamp disappears
      // (nothing is partial), the duplicate loads it produced disappear, and
      // the per-request address math collapses to one s_addk_i32 + one
      // v_add_u32.
      //
      // The gfx950 M0-to-MUBUF wait state is why the s_nop pair guards the
      // s_mov: a buffer_load ... lds issued too close to the M0 write reads the
      // stale base.
      {
        static_assert(TILES_PER_WAVE == 1,
                      "the wave-tile train assumes wave w owns exactly tile w; "
                      "with more tiles per wave the train has to be repeated "
                      "per tile and the LDS base recomputed between them");
        static_assert(QKV_TILE_DATA % 1024 == 0,
                      "a tile must be a whole number of 64-lane x 16 B loads");
        static_assert(QKV_TILE_DATA_PADDED % 1024 == 0);
#ifdef MPK_LM_HEAD_KMAJOR
        // K-major, Phase C reads exactly [0, QKV_TILE_DATA): 16 B per lane,
        // MFMA_ITERS blocks of 1 KiB. Nothing past the data is ever touched.
        constexpr int LM_DMA_CHUNKS = QKV_TILE_DATA / 1024;
#else
        // Row-major, Phase C's 32-byte reads run 16 B past the last row's last
        // fragment, so the train covers the padded region too. That last chunk
        // reads from the record's scale suffix rather than out of the buffer --
        // in range, and it lands only in the don't-care upper half of an FP4
        // A-operand.
        constexpr int LM_DMA_CHUNKS = QKV_TILE_DATA_PADDED / 1024;
        static_assert(NUM_WAVES * QKV_TILE_DATA +
                              (LM_DMA_CHUNKS * 1024 - QKV_TILE_DATA) <=
                          WG_BYTES,
                      "the padded tail must stay inside the workgroup record");
#endif
        uint32_t voff = qkv_wg_voff +
                        static_cast<uint32_t>(warp_id * QKV_TILE_DATA) +
                        static_cast<uint32_t>(lane_id * 16);
#ifdef MPK_LM_MALL_PCT
        if (!MPK_LM_NT(wg_idx)) {
          asm volatile(
              "s_nop 4\n"
              "s_mov_b32 m0, %[lds_base]\n"
              "s_nop 0\n"
              "buffer_load_dwordx4 %[voff], %[rsrc], 0 offen" MPK_LM_PLAIN_MOD " lds\n"
              ".rept %c[reps]\n"
              "s_addk_i32 m0, 0x400\n"
              "v_add_u32_e32 %[voff], 0x400, %[voff]\n"
              "buffer_load_dwordx4 %[voff], %[rsrc], 0 offen" MPK_LM_PLAIN_MOD " lds\n"
              ".endr\n"
              : [voff] "+v"(voff)
              : [rsrc] "s"(qkv_rsrc),
                [lds_base] "s"(lm_lds_wave_base),
                [reps] "n"(LM_DMA_CHUNKS - 1)
              : "memory", "m0");
        } else
#endif
        asm volatile(
            "s_nop 4\n"
            "s_mov_b32 m0, %[lds_base]\n"
            "s_nop 0\n"
            "buffer_load_dwordx4 %[voff], %[rsrc], 0 offen sc0 nt lds\n"
            ".rept %c[reps]\n"
            "s_addk_i32 m0, 0x400\n"
            "v_add_u32_e32 %[voff], 0x400, %[voff]\n"
            "buffer_load_dwordx4 %[voff], %[rsrc], 0 offen sc0 nt lds\n"
            ".endr\n"
            : [voff] "+v"(voff)
            : [rsrc] "s"(qkv_rsrc),
              [lds_base] "s"(lm_lds_wave_base),
              [reps] "n"(LM_DMA_CHUNKS - 1)
            : "memory", "m0");
      }
#else
      {
        auto *qkv_lds_warp_base = (__attribute__((address_space(3)))
                                   uint32_t *)(qkv_lds_w + warp_id * 1024);
#pragma unroll
      for (int t = 0; t < NUM_WAVES; t++) {
#pragma unroll
        for (int j = 0; j < QKV_LPT; j++) {
          int idx = tid + j * 256;
          int clamped = idx < qkv_n16_data ? idx : qkv_n16_data - 1;
          uint32_t voff =
              qkv_wg_voff +
              static_cast<uint32_t>(t * QKV_TILE_ROWS * (REDUCTION_SIZE / 2)) +
              static_cast<uint32_t>(clamped * 16);
          auto *lds_dst = (__attribute__((address_space(3)))
                           uint32_t *)((uint8_t __attribute__((address_space(
                                           3))) *)qkv_lds_warp_base +
                                       t * QKV_TILE_BYTES + j * 4096);
          __llvm_amdgcn_raw_buffer_load_lds(
              qkv_rsrc, lds_dst, 16, static_cast<int>(voff), 0, 0, 3);
        }
      }
    }
#endif
#ifdef MPK_LM_HEAD_GROUP_PIPELINE
      }
#endif

      // ── Phase B: Drain data loads, load scales via VGPR, scatter to LDS ──
      constexpr int QKV_SC_DW4_PER_TILE = QKV_TILE_SCALE / 16;
      constexpr int QKV_TOTAL_SC_DW4 = QKV_SC_DW4_PER_TILE * NUM_WAVES;
      constexpr int QKV_SC_LPT = (QKV_TOTAL_SC_DW4 + 255) / 256;

      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
#ifdef MPK_LM_STAMPS
    if (lmst_n < 20) lmst[lmst_n++] = __builtin_amdgcn_s_memrealtime();
#endif

      // Issue group-zero scales from HBM via VGPR. Later groups use the
      // ping-pong direct-to-LDS scale buffers filled below.
#ifdef MPK_LM_HEAD_GROUP_PIPELINE
#ifdef MPK_LM_PF0
      if (group_iteration == 0 && !lm_pf0) {
#else
      if (group_iteration == 0) {
#endif
#endif
      uint8_t const *wg_scales_hbm =
          W + static_cast<int64_t>(wg_idx) * WG_BYTES + WG_DATA_BYTES;
      i32x4_t qkv_sc_buf[QKV_SC_LPT];
      {
        i32x4_t const *sc_src = (i32x4_t const *)wg_scales_hbm;
#pragma unroll
      for (int j = 0; j < QKV_SC_LPT; j++) {
        int idx = tid + j * 256;
        if (idx < QKV_TOTAL_SC_DW4) {
          qkv_sc_buf[j] = sc_src[idx];
        }
      }
    }

    // Drain scale loads, scatter to per-tile LDS slots
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    {
#pragma unroll
      for (int j = 0; j < QKV_SC_LPT; j++) {
        int idx = tid + j * 256;
        if (idx < QKV_TOTAL_SC_DW4) {
          int tile = idx / QKV_SC_DW4_PER_TILE;
          int off = idx % QKV_SC_DW4_PER_TILE;
          i32x4_t *dst_sc = (i32x4_t *)(qkv_lds_w + tile * QKV_TILE_BYTES +
                                        QKV_TILE_DATA_PADDED);
          dst_sc[off] = qkv_sc_buf[j];
        }
      }
    }
#ifdef MPK_LM_HEAD_GROUP_PIPELINE
      }
#endif

    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    __syncthreads();
#ifdef MPK_LMT_TS
    if (group_iteration == 0 && bblk == 0 && tile_idx == 0 && threadIdx.x == 0)
      g_lmt_ts[2] = __builtin_amdgcn_s_memrealtime();
#endif

#ifdef MPK_LM_HEAD_GROUP_PIPELINE
    // Stage the following group's 1,472-byte scale tile per wave into one of
    // two disjoint buffers. One wave request covers 1 KiB and a second,
    // partially-executing request covers the exact 448-byte tail.
    int const pipelined_next_wg_idx = wg_idx + workers_per_xcd;
#ifdef MPK_LM_PIPE_S
#ifndef MPK_LM_HEAD_WAVE_TILE_DMA
#error "MPK_LM_PIPE_S takes the scalar wave base of MPK_LM_HEAD_WAVE_TILE_DMA"
#endif
    int const lm_pnw = __builtin_amdgcn_readfirstlane(pipelined_next_wg_idx);
    uint32_t const lm_pvoff = static_cast<uint32_t>(lm_pnw) * WG_BYTES +
                              static_cast<uint32_t>(warp_id * QKV_TILE_DATA) +
                              static_cast<uint32_t>(lane_id * 16);
#endif
    if (pipelined_next_wg_idx < n_wgs_per_xcd) {
      constexpr int SCALE_TAIL_VECTORS = (QKV_TILE_SCALE - 1024) / 16;
      static_assert(QKV_TILE_SCALE > 1024);
      static_assert(QKV_TILE_SCALE - 1024 == SCALE_TAIL_VECTORS * 16);
      int const next_scale_buffer = group_iteration & 1;
      auto *next_scale_wave_lds = (__attribute__((address_space(3)))
                                   uint32_t *)(scale_pipeline_lds +
                                               next_scale_buffer *
                                                   QKV_SCALE_PIPELINE_BYTES +
                                               warp_id *
                                                   QKV_SCALE_TILE_PADDED);
      uint32_t next_scale_offset =
          static_cast<uint32_t>(pipelined_next_wg_idx) * WG_BYTES +
          WG_DATA_BYTES + warp_id * QKV_TILE_SCALE + lane_id * 16;
      lm_head_load_lds_dwordx4_hazard_safe(qkv_rsrc, next_scale_wave_lds,
                                           next_scale_offset, false);
      if (lane_id < SCALE_TAIL_VECTORS) {
        lm_head_load_lds_dwordx4_hazard_safe(
            qkv_rsrc, next_scale_wave_lds + 1024 / sizeof(uint32_t),
            next_scale_offset + 1024, false);
      }
    }
#endif

    // ── Phase C: MFMA loop reading from LDS ──
    for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; tile_iter++) {
      int wave_tile = warp_id + tile_iter * NUM_WAVES;
      uint8_t const *tile_data_lds =
          (uint8_t const *)(qkv_lds_w + wave_tile * QKV_TILE_BYTES);
      uint8_t const *tile_scale_lds;
#ifdef MPK_LM_HEAD_GROUP_PIPELINE
      if (group_iteration == 0) {
        tile_scale_lds = tile_data_lds + QKV_TILE_DATA_PADDED;
      } else {
        int const current_scale_buffer = (group_iteration - 1) & 1;
        tile_scale_lds =
            scale_pipeline_lds +
            current_scale_buffer * QKV_SCALE_PIPELINE_BYTES +
            warp_id * QKV_SCALE_TILE_PADDED;
      }
#else
      tile_scale_lds = tile_data_lds + QKV_TILE_DATA_PADDED;
#endif

      int w_row_in_tile = col;
      int const row_scale_base = w_row_in_tile * NUM_BLOCKS_32;

#ifdef MPK_LM_HEAD_GROUP_PIPELINE
      // This readonly request is older than all 23 next-group data requests.
      // vmcnt(23) below therefore retires it without draining that train.
      using BiasDwords = unsigned int __attribute__((ext_vector_type(2)));
      BiasDwords prefetched_bias;
      int const rel_idx_base = wave_tile * 16 + g * 4;
      if (tok_active) {
        auto const *bias_address =
            d_bias + wg_idx * OUTPUT_PER_WG + rel_idx_base;
        asm volatile("global_load_dwordx2 %0, %1, off"
                     : "=v"(prefetched_bias)
                     : "v"(bias_address)
                     : "memory");
      }
#endif

      // ── Weight fragment addressing ──────────────────────────────────────
      //
      // Row-major (default): lane (g, col) reads
      //   col * (REDUCTION_SIZE / 2) + KI * 64 + g * 16.
      // At REDUCTION_SIZE 2944 the row stride is 1472 B = 368 dwords and
      // 368 % 32 == 16, so the sixteen values of `col` land on just two bank
      // groups; with g * 16 adding four more the whole wave starts on eight
      // distinct banks, eight lanes deep on each -- an 8-way conflict on
      // every weight ds_read in the K loop. It also reads 32 B per lane when
      // an FP4 A-operand needs 16.
      //
      // K-major (MPK_LM_HEAD_KMAJOR): the record is repacked offline from
      // [tile][row][k128][quarter][16B] to [tile][k128][quarter][row][16B],
      // so the lane's fragment sits at lane_id * 16 within its K block --
      // lane_id == g * 16 + col is exactly quarter-major-then-row. The wave
      // then reads 64 consecutive 16-byte fragments, which is the
      // conflict-free pattern ds_read_b128 is built for, and the HBM side of
      // Phase A is a byte-for-byte image of the tile, so the permutation
      // carries into LDS with no change to the DMA.
      //
      // Only the address changes; the same bytes reach the same lane, so this
      // is bit-exact. The scale suffix stays row-major in both layouts (one
      // byte per 32-element block, read as a scalar), hence the shared
      // row_scale_base. Host side: shuffle_lm_head_record_kmajor() in
      // demo/gpt_oss/demo.py, gated on the same MPK_LM_HEAD_KMAJOR.
#ifdef MPK_LM_HEAD_KMAJOR
      static_assert(QKV_TILE_ROWS == 16,
                    "MPK_LM_HEAD_KMAJOR assumes a 16-row MFMA tile: lane_id "
                    "spans exactly 16 rows x 4 K-quarters, which is what "
                    "makes lane_id * 16 the fragment address.");
      // One K128 block holds all 16 rows x 4 quarters x 16 B.
      constexpr int LM_LDS_K_STRIDE = QKV_TILE_ROWS * (K_PER_MFMA / 2);
      static_assert(((MFMA_ITERS - 1) * LM_LDS_K_STRIDE) + 64 * 16 <=
                        QKV_TILE_DATA,
                    "K-major fragment sweep must stay inside the tile");
      int const lds_data_lane_offset = lane_id * 16;
#define MPK_LM_W_FRAG(KI)                                                      \
  _gang_lm_load_fp4_a_kmajor(tile_data_lds + lds_data_lane_offset +            \
                             (KI)*LM_LDS_K_STRIDE)
#else
      int const row_data_base = w_row_in_tile * (REDUCTION_SIZE / 2);
#define MPK_LM_W_FRAG(KI)                                                      \
  (*(i32x8_t const *)(tile_data_lds + row_data_base +                          \
                      (KI) * (K_PER_MFMA / 2) + g * 16))
#endif

      f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

      i32x8_t a0 = MPK_LM_W_FRAG(0);
      int sa0 = (int)tile_scale_lds[row_scale_base + 0 * 4 + g];
      i32x8_t a1 = MPK_LM_W_FRAG(1);
      int sa1 = (int)tile_scale_lds[row_scale_base + 1 * 4 + g];
      i32x8_t a2 = MPK_LM_W_FRAG(2);
      int sa2 = (int)tile_scale_lds[row_scale_base + 2 * 4 + g];
      i32x8_t a3 = MPK_LM_W_FRAG(3);
      int sa3 = (int)tile_scale_lds[row_scale_base + 3 * 4 + g];

#if defined(MPK_LM_HEAD_GROUP_PIPELINE) && defined(MPK_LM_PIPE_S)
#define MPK_LM_PIPE_NEXT(KI)                                                   \
  do {                                                                         \
    if (lm_pnw < n_wgs_per_xcd) {                                              \
      lm_head_load_lds_dwordx4_s(qkv_rsrc, lm_lds_wave_base + (KI) * 1024u,    \
                                 lm_pvoff + (KI) * 1024u, MPK_LM_NT(lm_pnw));  \
    }                                                                          \
  } while (0)
#elif defined(MPK_LM_HEAD_GROUP_PIPELINE)
#define MPK_LM_PIPE_NEXT(KI)                                                   \
  do {                                                                         \
    if (pipelined_next_wg_idx < n_wgs_per_xcd) {                               \
      uint32_t const next_vector_offset =                                      \
          static_cast<uint32_t>(pipelined_next_wg_idx) * WG_BYTES +            \
          static_cast<uint32_t>(warp_id * QKV_TILE_DATA) +                     \
          static_cast<uint32_t>((KI) * 1024 + lane_id * 16);                   \
      auto *next_destination = (__attribute__((address_space(3)))              \
                                 uint32_t *)(qkv_lds_w +                        \
                                             warp_id * QKV_TILE_BYTES +        \
                                             (KI) * 1024);                     \
      lm_head_load_lds_dwordx4_hazard_safe(                                    \
          qkv_rsrc, next_destination, next_vector_offset,                      \
          MPK_LM_NT(pipelined_next_wg_idx));                                   \
    }                                                                          \
  } while (0)
#endif

#pragma unroll 1
      for (int ki = 0; ki < MFMA_ITERS; ki += 4) {
        {
          i32x8_t b = _gang_load_fp8_mfma_b(b_data, ki * K_PER_MFMA, g);
          int sb = (int)b_scale[ki];
          acc = _gang_mfma_f4xf8(a0, b, acc, sa0, sb);
        }
#ifdef MPK_LM_HEAD_GROUP_PIPELINE
        MPK_LM_PIPE_NEXT(ki);
#endif
        if (ki + 4 < MFMA_ITERS) {
          a0 = MPK_LM_W_FRAG(ki + 4);
          sa0 = (int)tile_scale_lds[row_scale_base + (ki + 4) * 4 + g];
        }
        {
          i32x8_t b = _gang_load_fp8_mfma_b(b_data, (ki + 1) * K_PER_MFMA, g);
          int sb = (int)b_scale[ki + 1];
          acc = _gang_mfma_f4xf8(a1, b, acc, sa1, sb);
        }
#ifdef MPK_LM_HEAD_GROUP_PIPELINE
        MPK_LM_PIPE_NEXT(ki + 1);
#endif
        if (ki + 5 < MFMA_ITERS) {
          a1 = MPK_LM_W_FRAG(ki + 5);
          sa1 = (int)tile_scale_lds[row_scale_base + (ki + 5) * 4 + g];
        }
        {
          i32x8_t b = _gang_load_fp8_mfma_b(b_data, (ki + 2) * K_PER_MFMA, g);
          int sb = (int)b_scale[ki + 2];
          acc = _gang_mfma_f4xf8(a2, b, acc, sa2, sb);
        }
#ifdef MPK_LM_HEAD_GROUP_PIPELINE
        MPK_LM_PIPE_NEXT(ki + 2);
#endif
        if (ki + 6 < MFMA_ITERS) {
          a2 = MPK_LM_W_FRAG(ki + 6);
          sa2 = (int)tile_scale_lds[row_scale_base + (ki + 6) * 4 + g];
        }
        if (ki + 3 < MFMA_ITERS) {
          i32x8_t b = _gang_load_fp8_mfma_b(b_data, (ki + 3) * K_PER_MFMA, g);
          int sb = (int)b_scale[ki + 3];
          acc = _gang_mfma_f4xf8(a3, b, acc, sa3, sb);
#ifdef MPK_LM_HEAD_GROUP_PIPELINE
          MPK_LM_PIPE_NEXT(ki + 3);
#endif
        }
        if (ki + 7 < MFMA_ITERS) {
          a3 = MPK_LM_W_FRAG(ki + 7);
          sa3 = (int)tile_scale_lds[row_scale_base + (ki + 7) * 4 + g];
        }
      }
#undef MPK_LM_W_FRAG
#ifdef MPK_LM_HEAD_GROUP_PIPELINE
#undef MPK_LM_PIPE_NEXT
      // Final emitted order per wave is: two next-scale requests, one current
      // bias request, then exactly 23 next-data requests. Keep the 23 younger
      // data requests in flight across argmax while retiring every current
      // consumer. The next iteration's vmcnt(0)/lgkmcnt(0) drains all of them
      // before the barrier makes the recycled LDS tile visible.
      if (pipelined_next_wg_idx < n_wgs_per_xcd) {
        asm volatile("s_waitcnt vmcnt(23)" ::: "memory");
      } else {
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      }
#endif

      // Argmax epilogue: accumulate across ALL tiles in registers.
      // The guard is on `tok_active`, not `col == 0`: lane (g, col) now holds
      // the 4 outputs for token `col`, so all 16 N columns are live. It sits
      // strictly AFTER the MFMA -- narrowing exec around the ds_read_b128 /
      // MFMA pair lets the compiler hoist the mask and feed stale B operands
      // to the lanes that ARE active.
      if (tok_active) {
        for (int i = 0; i < 4; i++) {
          int rel_idx = wave_tile * 16 + g * 4 + i;
          int abs_idx = partition_start + wg_idx * OUTPUT_PER_WG + rel_idx;
          float sum = acc[i];
#ifdef MPK_LM_HEAD_GROUP_PIPELINE
          unsigned const packed_bias = prefetched_bias[i >> 1];
          unsigned const bt = (i & 1) != 0 ? packed_bias & 0xffff0000U
                                            : packed_bias << 16;
#else
          unsigned bt = (unsigned)d_bias[wg_idx * OUTPUT_PER_WG + rel_idx]
                        << 16;
#endif
          float bv;
          __builtin_memcpy(&bv, &bt, 4);
          float val = sum + bv;
          if (val > thread_max) {
            thread_max = val;
            thread_max_idx = abs_idx;
          }
          // Perplexity mode: also spill the logit to HBM. abs_idx is the
          // vocab column this lane owns, so the writes across all workers
          // tile the row exactly once -- no atomics, no races.
          //
          // Stored as f32, not bf16: bf16 carries ~0.4% relative precision,
          // which is the same order as the GEMM error this buffer exists to
          // measure. Truncating here would fold a comparable one-sided error
          // into the very quantity under test.
          //
          // `step` is the position scored by token row 0 of this iteration;
          // row `tok_row` scored the position `tok_row` later, so each token
          // gets its own logits row and the rows still tile exactly once.
          if (logits_out_ptr != nullptr) {
            reinterpret_cast<float *>(
                logits_out_ptr)[(long long)(step + tok_row) * output_stride +
                                abs_idx] = val;
          }
        }
      }
    }
    } // end tile loop
#endif  // MPK_LM_BF16
#ifdef MPK_LMT_TS
    if (bblk == 0 && tile_idx == 0 && threadIdx.x == 0) g_lmt_ts[3] = __builtin_amdgcn_s_memrealtime();
#endif
#ifdef MPK_LM_INNER
    // Sample every 32nd worker. Emitting from all 248 x 86 iterations was
    // ~21,300 printfs and hit the 700 s timeout; ~670 samples is ample for
    // a p50/p95 split. Same approach MPK_MOE_INNER_TIMING uses (every 37th).
    if (tid == 0 && bblk == 0 && (tile_idx & 31) == 0) {
      asm volatile("s_memrealtime %0" : "=s"(_lmi_t2));
      // s_memrealtime is the 100 MHz constant clock: 1 tick = 10 ns.
      printf("[LM_INNER] w=%d normquant=%llu weight=%llu total=%llu\n",
             tile_idx,
             (unsigned long long)((_lmi_t1 - _lmi_t0) * 10ull),
             (unsigned long long)((_lmi_t2 - _lmi_t1) * 10ull),
             (unsigned long long)((_lmi_t2 - _lmi_t0) * 10ull));
    }
#endif

#ifdef MPK_LMX
#ifdef MPK_LMX_EVERY
    if (tid == 0 && bblk == 0 && step >= MPK_LMX &&
        (step - MPK_LMX) % MPK_LMX_EVERY == 0) {
#else
    if (tid == 0 && bblk == 0 && step == MPK_LMX) {
#endif
      unsigned long long const _lmx_t1 = __builtin_amdgcn_s_memrealtime();
      unsigned _lmx_x;
      MPK_GET_XCC_ID32(_lmx_x);
#ifdef MPK_LMXB
      int const _lmx_i = (step - MPK_LMX) / MPK_LMX_EVERY;
      if (_lmx_i < MPK_LMXB_N && tile_idx < MPK_LMXB_W) {
        unsigned long long *const _q =
            &g_lmxb[((size_t)_lmx_i * MPK_LMXB_W + tile_idx) * 3];
        _q[0] = _lmx_t0;
        _q[1] = _lmx_t1;
        _q[2] = (unsigned long long)(_lmx_x & 7u) + 1ull;
      }
#else
      printf("[LMX] w=%d p=%d hw=%u t0=%llu t1=%llu\n", tile_idx,
             partition_index, _lmx_x & 7u, _lmx_t0, _lmx_t1);
#endif
    }
#ifdef MPK_LMXB
    if (tid == 0 && bblk == 0 && tile_idx == 0 &&
        step == MPK_LMX + (MPK_LMXB_N + 1) * MPK_LMX_EVERY) {
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, MPK_DEV_FENCE_ACQ);
      asm volatile("buffer_inv" MPK_DEV_SC_INV "" ::: "memory");
      for (int i = 0; i < MPK_LMXB_N; i++)
        for (int w = 0; w < MPK_LMXB_W; w++) {
          unsigned long long const *const _q =
              &g_lmxb[((size_t)i * MPK_LMXB_W + w) * 3];
          unsigned long long const hw = __atomic_load_n(&_q[2], __ATOMIC_RELAXED);
          if (hw)
            printf("[LMX] w=%d p=%d hw=%llu t0=%llu t1=%llu\n", w, i,
                   hw - 1ull, __atomic_load_n(&_q[0], __ATOMIC_RELAXED),
                   __atomic_load_n(&_q[1], __ATOMIC_RELAXED));
        }
    }
#endif
#endif
    // ── Reduce within the wave, across g only ───────────────────────────
#ifdef MPK_LM_STAMPS
    if (tid == 0 && worker_rank == 0 && step == 1060) {
      unsigned long long const lmst_end = __builtin_amdgcn_s_memrealtime();
      printf("[LMST] p=%d n=%d %llu %llu %llu %llu %llu %llu %llu %llu %llu "
             "%llu %llu %llu %llu %llu %llu %llu %llu end=%llu\n",
             partition_index, lmst_n, lmst_n > 1 ? lmst[1] - lmst[0] : 0ull, lmst_n > 2 ? lmst[2] - lmst[0] : 0ull, lmst_n > 3 ? lmst[3] - lmst[0] : 0ull, lmst_n > 4 ? lmst[4] - lmst[0] : 0ull, lmst_n > 5 ? lmst[5] - lmst[0] : 0ull, lmst_n > 6 ? lmst[6] - lmst[0] : 0ull, lmst_n > 7 ? lmst[7] - lmst[0] : 0ull, lmst_n > 8 ? lmst[8] - lmst[0] : 0ull, lmst_n > 9 ? lmst[9] - lmst[0] : 0ull, lmst_n > 10 ? lmst[10] - lmst[0] : 0ull, lmst_n > 11 ? lmst[11] - lmst[0] : 0ull, lmst_n > 12 ? lmst[12] - lmst[0] : 0ull, lmst_n > 13 ? lmst[13] - lmst[0] : 0ull, lmst_n > 14 ? lmst[14] - lmst[0] : 0ull, lmst_n > 15 ? lmst[15] - lmst[0] : 0ull, lmst_n > 16 ? lmst[16] - lmst[0] : 0ull, lmst_n > 17 ? lmst[17] - lmst[0] : 0ull,
             lmst_end - lmst[0]);
    }
#endif
    // NOT the full 64-lane butterfly: lanes differing in `col` hold different
    // tokens, so mixing them would return one token's argmax for all 16.
    // XOR on bits 4 and 5 walks the 4 g-groups and leaves `col` alone.
#ifdef MPK_ARGMAX_DUAL_REDUCE
    // Same two steps, same order, no LDS. Valid in g == 0 only, which is the
    // only group the `if (g == 0)` write below reads -- see the contract note
    // at argmax_dual_wave_reduce_to_g0.
    argmax_dual_wave_reduce_to_g0(thread_max, thread_max_idx);
#else
#pragma unroll
    for (int offset = 16; offset <= 32; offset <<= 1) {
      float other_val = __shfl_xor(thread_max, offset, 64);
      int other_idx = __shfl_xor(thread_max_idx, offset, 64);
      if (other_val > thread_max) {
        thread_max = other_val;
        thread_max_idx = other_idx;
      }
    }
#endif

    // ── Cross-wave reduce via LDS, one slot per (wave, token) ───────────
    // No barrier before the write: s_red_* is disjoint from the token region
    // the MFMA loop just read, and the previous block's readers were drained
    // by the barrier at the bottom of this sweep.
    if (g == 0) {
      s_red_val[warp_id * MFMA_N + col] = thread_max;
      s_red_idx[warp_id * MFMA_N + col] = thread_max_idx;
    }
    __syncthreads();

    // ── Write per-(token, worker) result ────────────────────────────────
    if (tid < MFMA_N) {
      int out_row = bblk * MFMA_N + tid;
      if (out_row < batch_count) {
        float best_val = -1e30f;
        int best_idx = -1;
        for (int w = 0; w < NUM_WAVES; w++) {
          float v = s_red_val[w * MFMA_N + tid];
          if (v > best_val) {
            best_val = v;
            best_idx = s_red_idx[w * MFMA_N + tid];
          }
        }
        unsigned int vbits;
        __builtin_memcpy(&vbits, &best_val, 4);
        unsigned short bf16_val = (unsigned short)(vbits >> 16);
        long long out_off = (long long)out_row * argmax_row_stride + worker_rank;
        argmax_vals[out_off] = bf16_val;
        argmax_idxs[out_off] = (long long)best_idx;
#ifdef MPK_EAGLE3
        if (g_e3.enabled && out_row < (E3_R > 4 ? E3_R : 4)) {  // all extend rows past block 4
          e3_st2(E3_V_YV + out_row * 256 + tile_idx, (unsigned)bf16_val);
          e3_st2(E3_V_YI + out_row * 256 + tile_idx, (unsigned)best_idx);
        }
#endif
#ifdef MPK_DFLASH
        if (g_df.enabled && out_row < DF_B) {
          df_st2(DF_V_YV + out_row * 256 + tile_idx, (unsigned)bf16_val);
          df_st2(DF_V_YI + out_row * 256 + tile_idx, (unsigned)best_idx);
        }
#endif
#ifdef MPK_ARGMAX_FUSED
        {
          static_assert(BATCH_SIZE == 1, "MPK_ARGMAX_FUSED is batch 1");
          unsigned amx_x;
          MPK_GET_XCC_ID32(amx_x);
          amx_x &= 7u;
          unsigned long long *const amx_slot =
              reinterpret_cast<unsigned long long *>(
                  g_aid_nc_rep[amx_x >> 2] + MPK_NC_AMX_INTS + amx_x * 64);
          unsigned long long const amx_key =
              mpk_amx_key(bf16_val, tile_idx, best_idx);
          asm volatile("global_atomic_umax_x2 %0, %1, off sc1" ::"v"(amx_slot),
                       "v"(amx_key)
                       : "memory");
        }
#endif
      }
    }
#ifdef MPK_LMT_TS
    if (bblk == 0 && tile_idx == 0 && threadIdx.x == 0) g_lmt_ts[4] = __builtin_amdgcn_s_memrealtime();
#endif
    // The next block's quantizer overwrites s_tok_fp8, and s_red_* is reused.
    // Compiled out at bs <= 16, where there is no next block.
    if constexpr (NUM_BBLK > 1) {
      __syncthreads();
    }
  } // end batch column-block sweep
#ifdef MPK_EAGLE3
#ifdef MPK_EAGLE3_TS
  if (tile_idx == 0 && threadIdx.x == 0) g_e3lm_ts[2] = __builtin_amdgcn_s_memrealtime();
  if (threadIdx.x == 0 && g_e3.enabled &&
      E3_SPEC(2) == *g_e3.prompt_len + 20) {
    e3_st_sys_u32(g_e3.vec + E3_V_SPAN + 3 * 248 + tile_idx, (unsigned)__builtin_amdgcn_s_memrealtime());
  }
#endif
#ifdef MPK_E3_STATE1
  mpk_eagle3_phase(tile_idx, tile_idx / workers_per_xcd, e3st);
#else
  mpk_eagle3_phase(tile_idx, tile_idx / workers_per_xcd);
#endif
#ifdef MPK_END_POLL
  // the end scheduler reads this pass's drafts and target argmax as soon as the LM-head event completes: every
  // thread's stores here retire before the task's completion is signalled
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  __syncthreads();
#endif
#endif
#ifdef MPK_DFLASH
  mpk_dflash_phase(tile_idx, tile_idx / workers_per_xcd, dfst);
#endif
}

} // namespace kernel
