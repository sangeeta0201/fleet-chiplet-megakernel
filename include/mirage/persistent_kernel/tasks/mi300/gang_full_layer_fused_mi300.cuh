// MPK_VT2_PATCHED
#if defined(MPK_QKV_EPOCH_NORET) &&                                                  \
    (!defined(MPK_QKV_EPOCH_DIRECT) || !defined(MPK_AID_QKV_ARRIVE) ||                    \
     !defined(MPK_AID_SPLIT_FLAGS) || defined(MPK_ATTN_QWAIT))
#error "MPK_QKV_EPOCH_NORET: the direct AID-replica QKV epoch (not with MPK_ATTN_QWAIT)"
#endif
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
// Counter buffer slot map (extends type 215's layout):
//   oproj_counters_ptr + 0..18*16-1  : type 215's counters
//   oproj_counters_ptr + 19*16       : attn_global_counter (cross-XCD sync)
//   oproj_counters_ptr + 20*16       : qkv_epoch[0..7] per-XCD epoch flags
//   oproj_counters_ptr + 28*16       : chunk_barrier[0..7] per-XCD chunk
//   arrival
//
// Stack-frame optimization: input/output pointers are NOT unpacked into
// local variables. Instead, input_ptrs[N] and output_ptrs[N] are accessed
// directly, saving ~272 bytes of stack frame per thread.

#pragma once
#ifndef MPK_TKOPS_AT
#define MPK_TKOPS_AT(k, cond) do {} while (0)
#endif
#ifndef MPK_TIDX_F_F
#if defined(MPK_TID_FRESH) || defined(MPK_TID_FRESH_F)
// threadIdx.x through a volatile v_mov: nothing derived from it can be hoisted out of the
// persistent loop into kernel-lifetime registers.
__device__ __forceinline__ unsigned mpk_tid_fresh_F() {
  unsigned t;
  asm volatile("v_mov_b32 %0, %1\n\ts_nop 1" : "=v"(t) : "v"(threadIdx.x));
  return t;
}
#define MPK_TIDX_F (mpk_tid_fresh_F())
#else
#define MPK_TIDX_F (threadIdx.x)
#endif
#endif
#ifdef MPK_TOPK_CHECK
// passes checked, ties inside the top four, ties at the fourth place, mismatches
__device__ unsigned long long g_tkc[4];
#endif
#ifdef MPK_ML_TABLE_INLAYER
// Layers per token and ml-table widths for the in-layer row handoff.
#define MPK_MLT_LAYERS 36
#define MPK_MLT_NIN (mirage::runtime::MAX_INPUTS_PER_TASK)
#define MPK_MLT_NOUT (mirage::runtime::MAX_OUTPUTS_PER_TASK)
#endif
#ifdef MPK_GATE_DEFER
#ifndef MPK_GD_LAYERS
#define MPK_GD_LAYERS 36
#endif
// Layer L's gate threshold, carried by tid 0 into layer L+1's call.
__shared__ int s_gd_prev;
#endif
#ifdef MPK_ATTN_META_LDS
#ifndef MPK_ATTN_META_EARLY
#error "MPK_ATTN_META_LDS caches the MPK_ATTN_META_EARLY metadata"
#endif
#ifndef MPK_AM_LAYERS
#define MPK_AM_LAYERS 36
#endif
// The pass's attention metadata, written on its first layer, read by later layers of either instantiation.
__shared__ int s_am_meta[6];
#endif

// MPK_WAIT_SKIP: same bisection discipline as MPK_P7_SKIP -- remove one class
// of WAIT at a time, never a release, so nothing can deadlock. Results are
// garbage by design; the timing delta against the control is the cost of the
// waits that were removed.
#ifdef MPK_WAIT_SKIP
#define MPK_WS_SK(bit) (((MPK_WAIT_SKIP) & (bit)) != 0)
#else
#define MPK_WS_SK(bit) false
#endif

#include "tasks/mi300/gang_linear_mxfp4_res_bias_rmsnorm_topk_mi300.cuh"
#include "tasks/mi300/gang_moe_fused_mxfp4_mi300.cuh"
#ifdef MPK_GATE_SNAP
#define MPK_GS_STAMP(k)                                                        \
  do {                                                                         \
    unsigned const _gs_l = (unsigned)(layer_counter - (MPK_GATE_SNAP) * 36);   \
    unsigned const _gs_w = (unsigned)(xcd_id * workers_per_xcd + xcd_rank);    \
    if (_gs_l < 36u && _gs_w < 256u) {                                         \
      g_gs_snap[(_gs_l * 256u + _gs_w) * 4u + (k)] =                           \
          __builtin_amdgcn_s_memrealtime();                                    \
    }                                                                          \
  } while (0)
#else
#define MPK_GS_STAMP(k) ((void)0)
#endif
#include "tasks/mi300/gang_rmsnorm_linear_mxfp4_bias_mi300.cuh"
#include "tasks/mi300/paged_attention_ck_fmha_split_kv_mi300.cuh"

namespace kernel {
#ifdef MPK_VT2_SCHED4
#ifndef MPK_VT2_SCHED3
#error "MPK_VT2_SCHED4 is a variant of MPK_VT2_SCHED3 (its arrival sets)"
#endif
// Round i of rank r (23 groups per pair member): (moe_i << 8) | (token << 7) | group, or -1.
__device__ __forceinline__ int mpk_vt2_sched4(int i, int r) {
  if (i == 0) return r <= 22 ? r : ((1 << 7) | (r - 23));                  // W13 t0 0-22 | W13 t1 0-7
  if (i == 1) return r <= 14 ? ((1 << 7) | (8 + r)) : ((1 << 8) | (r - 15)); // W13 t1 8-22 | W2 t0 0-15
#if MPK_VT2_SCHED4 == 1
  if (r < 20) return -1;
  int const k = (i - 2) * 11 + (r - 20);
  if (k < 7) return (1 << 8) | (16 + k);                                   // W2 t0 16-22
  if (k < 30) return (1 << 8) | (1 << 7) | (k - 7);                        // W2 t1 0-22
  return -1;
#elif MPK_VT2_SCHED4 == 2
  if (i == 2) {
    if (r <= 6) return (1 << 8) | (16 + r);                                // W2 t0 16-22
    if (r <= 14) return (1 << 8) | (1 << 7) | (r - 7);                     // W2 t1 0-7
    if (r < 20) return -1;
    return (1 << 8) | (1 << 7) | (8 + r - 20);                             // W2 t1 8-18
  }
  if (i == 3 && r >= 20 && r <= 23) return (1 << 8) | (1 << 7) | (19 + r - 20);  // W2 t1 19-22
  return -1;
#else
#error "MPK_VT2_SCHED4 must be 1 or 2"
#endif
}
#define MPK_VT2_NROUNDS 5
#else
#define MPK_VT2_NROUNDS 3
#endif
#ifdef MPK_VT2_QKV_EARLY
__constant__ int g_mpk_qkve_rt;  // env MPK_QKVE
#endif
#ifdef MPK_VT2_T1ROW_LATE
__constant__ int g_mpk_t1late_rt;  // env MPK_T1LATE
#endif
#ifdef MPK_ML_STAGE
__constant__ int g_mpk_mlstage_rt;  // env MPK_MLSTAGE: boundary rows staged in LDS
#endif
#ifdef MPK_VTN_BAL
__constant__ int g_mpk_vtnbal_rt;  // env MPK_VTNBAL: union MoE W2 tiles by rank availability
#endif
#ifdef MPK_ATTN_FIXED_MERGER_RT
__constant__ int g_mpk_fm_rt;  // env MPK_FM
#endif
#ifdef MPK_ATTN_META_LDS_RT
__constant__ int g_mpk_metalds_rt;  // env MPK_METALDS
#endif
#ifdef MPK_VTN_N7B2
__constant__ int g_mpk_n7b2_rt;  // env MPK_N7B2: per-wave TopK gather (4-token pass)
#endif
#if defined(MPK_VTN_N7S2) && defined(MPK_VTN_N7S2_RT)
__constant__ int g_mpk_n7s2_rt;  // env MPK_N7S2: split second Phase 7b round on (1, default) / off (0)
#endif
#ifdef MPK_VT2_W2DELAY
__constant__ int g_mpk_vt2_w2d;
#ifdef MPK_VT2_QSPLIT_RT
__constant__ int g_mpk_qsplit_rt;  // env MPK_QSPLIT
#endif
#ifdef MPK_VT2_MPOLL_RT
__constant__ int g_mpk_mpoll_rt;  // env MPK_MPOLL
#endif
#endif
#ifdef MPK_VT2_KSEL2_CHECK
__device__ unsigned int g_ks2chk[4];
#endif
#ifdef MPK_VTN_UNIONM_CHECK
__device__ unsigned int g_unionm_chk[2];  // MPK_VTN_UNIONM_CHECK: checks, mismatches
#endif
#ifdef MPK_VTN_KREG_CHECK
__device__ unsigned int g_kreg_chk[2];  // MPK_VTN_KREG_CHECK: tokens, differing selections
#endif
#ifdef MPK_VTN_TKDEF
#ifdef MPK_VTN_TKDEF_CHECK
__device__ unsigned int g_tkdef_chk[2];  // tokens checked, tokens with any differing weight bit
#endif
// MPK_VTN_TKDEF: token nt's routing weights from the logits and the four selection keys Phase 7b left in the
// VTN scratch. Wave w takes tokens 4w + r in 16-lane row r: Phase 7b's statistics on the row's lanes (8 logits
// each, row-local DPP reductions), the row's lane 0 broadcast in place of readfirstlane, the exact tail on the
// row's lane 0. Per-token arithmetic identical to Phase 7b.
template <int NT>
__device__ __forceinline__ void mpk_vtn_tkdef_tail(char *smem, int tid) {
  unsigned short const *const nlg = (unsigned short const *)(smem + MPK_VTN_SCR + 2048 + NT * 132 * 4);
  unsigned const *const ntk = (unsigned const *)(smem + MPK_VTN_SCR + 2048 + NT * (132 * 4 + 128 * 2));
  int const ln = tid & 63;
  int const lr = ln & 15;
  int const row = ln >> 4;
  int const nt = (tid >> 6) * 4 + row;
  if ((tid >> 6) * 4 >= NT) return;
  bool const live = nt < NT;
  unsigned short const *const lg = nlg + (live ? nt : 0) * 128;
  float rc[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    rc[e] = __uint_as_float(((unsigned)lg[lr * 8 + e]) << 16);
  }
  float tmax = rc[0];
#pragma unroll
  for (int ii = 1; ii < 8; ++ii) {
    tmax = fmaxf(tmax, rc[ii]);
  }
  tmax = topk_dpp_row_max_to_lane_zero(tmax);
  {
    int const t0 = __builtin_amdgcn_readlane(__float_as_int(tmax), 0);
    int const t1 = __builtin_amdgcn_readlane(__float_as_int(tmax), 16);
    int const t2 = __builtin_amdgcn_readlane(__float_as_int(tmax), 32);
    int const t3 = __builtin_amdgcn_readlane(__float_as_int(tmax), 48);
    tmax = __int_as_float(row == 0 ? t0 : (row == 1 ? t1 : (row == 2 ? t2 : t3)));
  }
  float rsum = 0.f;
#pragma unroll
  for (int ii = 0; ii < 8; ++ii) {
    rc[ii] = __builtin_amdgcn_exp2f((rc[ii] - tmax) * 1.4426950408889634f);
    rsum += rc[ii];
  }
  rsum = topk_dpp_row_sum_to_lane_zero(rsum);
  {
    int const t0 = __builtin_amdgcn_readlane(__float_as_int(rsum), 0);
    int const t1 = __builtin_amdgcn_readlane(__float_as_int(rsum), 16);
    int const t2 = __builtin_amdgcn_readlane(__float_as_int(rsum), 32);
    int const t3 = __builtin_amdgcn_readlane(__float_as_int(rsum), 48);
    rsum = __int_as_float(row == 0 ? t0 : (row == 1 ? t1 : (row == 2 ? t2 : t3)));
  }
  float const inv = 1.f / rsum;
  if (lr == 0 && live) {
    unsigned kg[4];
#pragma unroll
    for (int k_idx = 0; k_idx < 4; ++k_idx) kg[k_idx] = ntk[nt * 4 + k_idx];
    float vt2_p[4];
    float vt2_rs = 0.f;
#pragma unroll
    for (int k_idx = 0; k_idx < 4; ++k_idx) {
      unsigned const o = kg[k_idx] >> 16;
      unsigned const b = (o & 0x8000u) ? (o & 0x7FFFu) : (~o & 0xFFFFu);
      float const e = __builtin_amdgcn_exp2f(
          __fmul_rn(__fsub_rn(__uint_as_float(b << 16), tmax), 1.4426950408889634f));
      vt2_p[k_idx] = __fmul_rn(e, inv);
      vt2_rs = __fadd_rn(vt2_rs, vt2_p[k_idx]);
    }
    float const vt2_rinv = __fdiv_rn(1.f, vt2_rs);
#ifdef MPK_VTN_TKDEF_CHECK
    unsigned bad = 0u;
#pragma unroll
    for (int k_idx = 0; k_idx < 4; ++k_idx) {
      bad |= __float_as_uint(s_vt2_w[nt][k_idx]) ^ __float_as_uint(__fmul_rn(vt2_p[k_idx], vt2_rinv));
    }
    bad |= __float_as_uint(s_vt2_w[nt][4]) ^ __float_as_uint(tmax);
    bad |= __float_as_uint(s_vt2_w[nt][5]) ^ __float_as_uint(inv);
    unsigned const n = atomicAdd(&g_tkdef_chk[0], 1u);
    if (bad != 0u) atomicAdd(&g_tkdef_chk[1], 1u);
    if ((n & 0xFFFFFu) == 0xFFFFFu) printf("[TKDEFCHK] checks=%u bad=%u\n", n + 1u, g_tkdef_chk[1]);
#else
    s_vt2_w[nt][4] = tmax;
    s_vt2_w[nt][5] = inv;
#pragma unroll
    for (int k_idx = 0; k_idx < 4; ++k_idx) {
      s_vt2_w[nt][k_idx] = __fmul_rn(vt2_p[k_idx], vt2_rinv);
    }
#endif
  }
}
#endif

#ifdef MPK_ATS_MERGE
#define MPK_ATSP9(k, cond) do {} while (0)
#else
#define MPK_ATSP9(k, cond) MPK_ATSOPS_AT(k, cond)
#endif
#ifndef MPK_ATSOPS_AT
// MPK_ATS_OPS stamps (timing tool) compile to nothing unless the phase-slot recorder defined them.
#define MPK_ATSOPS_PUT(k, cond, val) \
  do {                               \
  } while (0)
#define MPK_ATSOPS_AT(k, cond) \
  do {                         \
  } while (0)
#endif
#ifdef MPK_ATTK_LDS
#define MPK_ATSTAMP_L(k)                                  \
  do {                                                    \
    if (MPK_TIDX_F == 0 && (k) == 0) {                   \
      s_ak_sw = (SLIDING_WINDOW > 0) ? 1u : 0u;           \
    }                                                     \
    MPK_ATSTAMP(k);                                       \
  } while (0)
#define MPK_ATVALUE_L(k, v) \
  do {                    \
  } while (0)
#elif defined(MPK_ATTSUB_SLIDING)
#define MPK_ATSTAMP_L(k)          \
  do {                            \
    if (SLIDING_WINDOW > 0) {     \
      MPK_ATSTAMP(k);             \
    }                             \
  } while (0)
#define MPK_ATVALUE_L(k, v)       \
  do {                            \
    if (SLIDING_WINDOW > 0) {     \
      MPK_ATVALUE(k, v);          \
    }                             \
  } while (0)
#else
#define MPK_ATSTAMP_L(k) MPK_ATSTAMP(k)
#define MPK_ATVALUE_L(k, v) MPK_ATVALUE(k, v)
#endif

static constexpr int FULL_LAYER_ATTN_GLOBAL_COUNTER_SLOT = 19 * 16;
static constexpr int FULL_LAYER_QKV_EPOCH_SLOT = 20 * 16;

#ifdef MPK_SEP_LDS
// LDS-DMA warm: stream [wbase, wbase+nbytes) into MALL via buffer_load sc0 lds
// (full bandwidth, no VGPR serialization). 32-way over 8 warmers x 4 waves;
// 23KB/wave ring in dynamic LDS (free on idle ranks during attention).
__device__ __forceinline__ void mpk_sep_lds_warm(uint8_t const *wbase,
                                                 long long nbytes, int warmer,
                                                 int warp_id, int lane_id,
                                                 char *sink) {
  i32x4_t rsrc = make_w_buffer_rsrc(wbase, (uint32_t)nbytes);
  int const wave_g = warmer * 4 + warp_id;
  long long const per = (nbytes + 31) / 32;
  uint32_t voff =
      (uint32_t)((long long)wave_g * per) + (uint32_t)(lane_id * 16);
  unsigned const m0b = __builtin_amdgcn_readfirstlane(
      (unsigned)(uintptr_t)(sink + warp_id * 23552));
  int const nbatches = (int)((per + 23 * 1024 - 1) / (23 * 1024));
  for (int b = 0; b < nbatches; b++) {
    asm volatile(
        "s_mov_b32 m0, %[m0b]\n"
        "buffer_load_dwordx4 %[voff], %[rsrc], 0 offen sc0 lds\n"
        ".rept 22\n"
        "s_addk_i32 m0, 0x400\n"
        "v_add_u32_e32 %[voff], 0x400, %[voff]\n"
        "buffer_load_dwordx4 %[voff], %[rsrc], 0 offen sc0 lds\n"
        ".endr\n"
        "v_add_u32_e32 %[voff], 0x400, %[voff]\n"
        : [voff] "+v"(voff)
        : [rsrc] "s"(rsrc), [m0b] "s"(m0b)
        : "memory", "m0", "scc");
  }
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
}
#endif
static constexpr int FULL_LAYER_ATTN_XCD_RELEASE_SLOT = 36 * 16;

// The chunk barrier is the one counter whose size depends on NUM_REQS: it needs
// one line per (XCD, request) so that the last chunk to arrive *for a given
// request* is the worker that merges *that* request. It therefore spans
// 8 * NUM_REQS lines = 128 * NUM_REQS ints.
//
// It used to live at 28 * 16, which is fine at NUM_REQS == 1 (8 lines, ending
// just below attn_xcd_release at 36 * 16) but grows straight through the
// fused-tail counters that gang_full_layer_with_lmhead_fused_mi300.cuh pins at
// the absolute literals 44*16 .. 47*16. At NUM_REQS >= 3 chunk arrivals would
// be written on top of the MoE-done / resadd-done / lmhead-done flags. That is
// latent today only because type 217 is dead behind an early `return`, and it
// would come back as silent corruption the moment FUSE_TAIL is re-enabled.
//
// Moving the barrier above every fixed slot instead of moving attn_xcd_release
// keeps every other counter at the address it has today, so nothing else in
// either file has to change. Callers must size the counter buffer to
// FULL_LAYER_CHUNK_BARRIER_SLOT + 128 * NUM_REQS (see demo.py counter_size).
static constexpr int FULL_LAYER_CHUNK_BARRIER_SLOT = 48 * 16;

// Layer-boundary global barrier: one line per XCD, immediately above the chunk
// barrier's 8 * NUM_REQS lines. See the barrier itself at the end of this file
// for why a *global* (all-XCD) rendezvous is required and a threadfence is not
// enough. Callers must size the counter buffer to
// FULL_LAYER_LAYER_BARRIER_SLOT + 272: 8 per-XCD arrival lines, one global
// arrival line, and 8 per-XCD release lines (see demo.py counter_size).
static constexpr int FULL_LAYER_LAYER_BARRIER_SLOT(int num_reqs) {
  return FULL_LAYER_CHUNK_BARRIER_SLOT + 128 * num_reqs;
}

// Per-XCD O-proj slice-ready flags for MPK_ROUTER_XCD_FOLD. 8 lines, 16 ints
// each, immediately above the layer-barrier region. demo.py counter_size
// adds 128 ints for this. Local-last of 23 O-proj tiles publishes its XCD
// here so the router can FMA that 368-col slice before the other XCDs land.
#ifdef MPK_P9_DEFER
#if !defined(MPK_P9_POLL_GLOBAL2) || !defined(MPK_LEAN_ARRIVE) || \
    !defined(MPK_PREFETCH_NEXT_QKV) || !defined(MPK_W2_CONSUMER_GATE) || \
    defined(MPK_NO_LAYER_BARRIER)
#error "MPK_P9_DEFER is wired for P9_POLL_GLOBAL2 + LEAN_ARRIVE + PREFETCH_NEXT_QKV + W2_CONSUMER_GATE"
#endif
// The layer-gate target carried from one layer's Phase 9 to the next layer's
// entry. It lives in the alignment padding after task_metadata in the task's
// shared-memory TaskDesc slot, which the ml loop never writes between layers
// (it rewrites only the pointer arrays, variant_id and _linear_reserved).
__device__ __forceinline__ int *mpk_p9_word(void *const *input_ptrs) {
  using TD = mirage::runtime::TaskDesc;
  constexpr size_t kWord = offsetof(TD, task_metadata) +
                           sizeof(mirage::runtime::FullTaskDesc::TaskMetadata);
  static_assert(kWord + sizeof(int) <= sizeof(TD),
                "TaskDesc has no padding word after task_metadata");
  char *td = reinterpret_cast<char *>(const_cast<void **>(input_ptrs)) -
             offsetof(TD, input_ptrs);
  return reinterpret_cast<int *>(td + kWord);
}
#endif

#ifdef MPK_ATTN_Q_EPOCH
#if defined(MPK_QKV_KSPLIT) || defined(MPK_ATTN_SPLIT_CHUNK) || \
    defined(MPK_QKV_EPOCH_POLL_SC1) || !defined(MPK_QKV_EPOCH_PRODUCER_DRAIN)
#error "MPK_ATTN_Q_EPOCH is wired for the unsplit QKV/attention and the drained epoch"
#endif
// Per-XCD Q-only arrival lines, then per-XCD Q release lines, 16 ints each,
// after the LTK and O-proj tile-flag words (demo.py grows the block).
static constexpr int MPK_ATTN_QEPOCH_SLOT = 2048 + 128 + 8 * 32 + 256;
#endif
#ifdef MPK_ATTN_QWAIT
#if !defined(MPK_QKV_EPOCH_DIRECT) || !defined(MPK_AID_QKV_ARRIVE) ||          \
    !defined(MPK_AID_SPLIT_FLAGS) || !defined(MPK_ATTN_META_EARLY) ||          \
    !defined(MPK_QKV_EPOCH_PRODUCER_DRAIN) || defined(MPK_ATTN_Q_EPOCH) ||     \
    defined(MPK_QKV_KSPLIT) || defined(MPK_ATTN_SPLIT_CHUNK) ||                \
    defined(MPK_KV_SW_1TILE) || defined(MPK_KV_FULL_TPC) ||                    \
    defined(MPK_QKV_EPOCH_POLL_SC1) || defined(MPK_QKV_EPOCH_POLL_ARRIVE) ||   \
    defined(MPK_ROTATE_QKV_ATTN_RANKS) || defined(MPK_QKV_KV_RANK_SWAP) ||     \
    defined(MPK_KV_SPREAD)
#error "MPK_ATTN_QWAIT: the direct AID-replica QKV epoch, META_EARLY metadata, plain rank map"
#endif
// Per-XCD Q-only arrival counters (16 ints each) in the AID flag replica, in the gap between the
// QKV arrival counters and the chunk barrier; the per-launch replica clear resets them too.
static constexpr int MPK_AID_QKVQ_BASE_INTS = 24704;
#ifdef MPK_ATTN_QWAIT_RT
// Runtime switch (host: env MPK_QWAIT_RT, default 1): one binary runs both arms.
__constant__ int g_mpk_qwait_rt = 1;
#define MPK_QWAIT_ON (g_mpk_qwait_rt != 0)
#else
#define MPK_QWAIT_ON true
#endif
static_assert(MPK_AID_QKVARR_BASE_INTS + 8 * 16 <= MPK_AID_QKVQ_BASE_INTS &&
                  MPK_AID_QKVQ_BASE_INTS + 8 * 16 <= MPK_AID_CHUNKBAR_BASE_INTS,
              "MPK_ATTN_QWAIT: Q-only counters must sit between QKVARR and CHUNKBAR");
// The chunk of the HD64 decode's split-KV partition that holds the newest of sk tokens:
// kv_start / ntiles as in the decode, mpk_kv_chunks_eff's adaptive count, then tiles per chunk.
// Whether chunk c of the HD64 decode's split-KV partition holds the newest of sk tokens, with no
// runtime division: in the adaptive split n = ceil(ntiles/4)/2 satisfies 8n-3 <= ntiles <= 8n+4,
// so a chunk holds 8 tiles (ntiles <= 8n) or 9; at n = 8 and at the NUM_KV_CHUNKS cap the divisor is
// a constant. Checked against the decode for every sk < 34100, NC 1..31, W 0/128.
template <int NUM_KV_CHUNKS, int SLIDING_WINDOW>
__device__ __forceinline__ bool mpk_qep_holds_newest(int c, int sk) {
  int kv_start = 0;
  if (SLIDING_WINDOW > 0 && sk > SLIDING_WINDOW) {
    kv_start = ((sk - SLIDING_WINDOW) / 16) * 16;
  }
  int const ntiles = (sk - kv_start + 15) / 16;
#ifdef MPK_KV_CHUNKS_ADAPTIVE
  int const n_raw = ((ntiles + 3) / 4) / 2;
  int tpc;
  if (NUM_KV_CHUNKS <= 8 || n_raw >= NUM_KV_CHUNKS) {
    tpc = (ntiles + NUM_KV_CHUNKS - 1) / NUM_KV_CHUNKS;
  } else if (n_raw <= 8) {
    tpc = (ntiles + 7) / 8;
  } else {
    tpc = ntiles <= 8 * n_raw ? 8 : 9;
  }
#else
  int const tpc = (ntiles + NUM_KV_CHUNKS - 1) / NUM_KV_CHUNKS;
#endif
  int const t = ntiles - 1;
  return c * tpc <= t && t < (c + 1) * tpc;
}
template <int NUM_KV_CHUNKS, int SLIDING_WINDOW>
__device__ __forceinline__ int mpk_qep_newest_chunk(int sk) {
  int kv_start = 0;
  if (SLIDING_WINDOW > 0 && sk > SLIDING_WINDOW) {
    kv_start = ((sk - SLIDING_WINDOW) / 16) * 16;
  }
  int const ntiles = (sk - kv_start + 15) / 16;
#ifdef MPK_KV_CHUNKS_ADAPTIVE
  int n = ((ntiles + 3) / 4) / 2;
  n = n < 8 ? 8 : n;
  n = n < NUM_KV_CHUNKS ? n : NUM_KV_CHUNKS;
#else
  int const n = NUM_KV_CHUNKS;
#endif
  int const tpc = (ntiles + n - 1) / n;
  return tpc > 0 ? (ntiles - 1) / tpc : 0;
}
#endif
static constexpr int FULL_LAYER_OPROJ_XCD_READY_SLOT(int num_reqs) {
  return FULL_LAYER_LAYER_BARRIER_SLOT(num_reqs) + 272;
}

// MPK_XCD_LOCAL_BARRIER: drop `sc1` from the arrival atomics of the barriers
// whose counters never leave one XCD.
//
// Fleet routes every barrier arrival in this file through
// atom_add_release_gpu_s32, which is `flat_atomic_add ... sc0 sc1`. On MI350
// `sc1` is what forces the operation out of the XCD's private L2 and through
// the device-wide coherency point. For a counter that another XCD reads --
// attn_global, layer_global -- that is exactly right and must stay. For a
// counter indexed by the arriving worker's own `xcd_id` on both the write and
// the read side, it is a cross-die round trip taken to publish a value no
// other die will ever load.
//
// Three of this file's barriers are in the second category (each verified by
// grepping every access to the counter):
//
//   Phase 2  input_ptrs[7][xcd_id] and qkv_epoch[xcd_id * 16]
//   Phase 4  chunk_barrier[(attn_kv_head * NUM_REQS + attn_req) * 16]
//   Phase 9  layer_local[xcd_id * 16]   (layer_global stays sc0 sc1)
//
// The releases are untouched: those really are cross-XCD and keep their st_wt
// fan-out. This changes only how the arrival is counted, not who waits for
// what, so the ordering argument at each site is unaffected.
#ifdef MPK_XCD_LOCAL_BARRIER
#define MPK_XCD_LOCAL_ATOM_ADD(addr, val) atom_add_xcd_local_s32((addr), (val))
#ifdef MPK_P9_ARRREC
__device__ unsigned long long g_p9_arrrec[8][2];
#endif
#else
#define MPK_XCD_LOCAL_ATOM_ADD(addr, val)                                      \
  atom_add_release_gpu_s32((addr), (val))
#endif

#ifdef MPK_ATTN_KV_EARLY
// Issue this wave's share of a full-attention chunk's past K/V as raw bf16
// LDS DMAs into its staging region (see __attn_wave_local_scan_hd64). Same
// chunk / wave / tile split as paged_attention_minimal_decode_hd64 and the
// scan; the newest token's tile is skipped (it is written by this layer's
// QKV). Tiles are 16 tokens x 128 B; lane l of a dwordx4 DMA moves token
// 8j + l/8, dims (l%8)*8.., landing at M0 + 16 l = [token][dim] rows.
template <int PAGE_SIZE, int HEAD_DIM, int NUM_KV_CHUNKS, int KV_CACHE_STRIDE,
          int TPW>
__device__ __noinline__ void mpk_attn_stage_kv_hd64(char const *k_base,
                                                    char const *v_base,
                                                    int const *kv_indices,
                                                    int k0,
                                                    int k1,
                                                    int lpl,
                                                    int chunk) {
  constexpr int KV_TILE = 16;
  static_assert(HEAD_DIM == 64, "128 B token rows");
  static_assert(MPK_ATTN_STAGE_OFF + 4 * TPW * 4096 <= 140 * 1024,
                "MPK_ATTN_KV_EARLY staging does not fit this max sequence");
  int const num_pages = k1 - k0;
  if (num_pages <= 0) {
    return;
  }
  int const seqlen_k = (num_pages - 1) * PAGE_SIZE + lpl;
  int const ntiles = (seqlen_k + KV_TILE - 1) / KV_TILE;
  int const mpk_neff = mpk_kv_chunks_eff<NUM_KV_CHUNKS>(ntiles);
  int const tpc = (ntiles + mpk_neff - 1) / mpk_neff;
  int const c_first = chunk * tpc;
  int c_last = c_first + tpc;
  if (c_last > ntiles) {
    c_last = ntiles;
  }
  int const nt = c_last - c_first;
  if (nt < MPK_ATTN_WAVE_LOCAL_MIN_TILES) {
    return;
  }
  int const wave_id = MPK_TIDX_F >> 6;
  int const lane = MPK_TIDX_F & 63;
  int const tpw = (nt + 3) / 4;
  int const w_first = wave_id * tpw;
  if (w_first >= nt) {
    return;
  }
  int w_last = w_first + tpw;
  if (w_last > nt) {
    w_last = nt;
  }
  int const newest_rel = (c_last == ntiles) ? nt - 1 : -1;
  extern __shared__ char mpk_stage_smem[];
#ifdef MPK_ATTN_KV_EARLY2
  auto *const stg = (__attribute__((address_space(3))) char *)(
      mpk_stage_smem + MPK_ATTN_STAGE2_OFF + w_first * 4096);
#else
  auto *const stg = (__attribute__((address_space(3))) char *)(
      mpk_stage_smem + MPK_ATTN_STAGE_OFF + wave_id * (TPW * 4096));
#endif
  int const voff0 = (lane >> 3) * KV_CACHE_STRIDE * 2 + (lane & 7) * 16;
  for (int t = 0; t < w_last - w_first; t++) {
    if (w_first + t == newest_rel) {
      continue;
    }
    int const tok0 = (c_first + w_first + t) * KV_TILE;
    int const pid = __builtin_amdgcn_readfirstlane(
        kv_indices[k0 + tok0 / PAGE_SIZE]);
    long const toff = (static_cast<long>(pid) * PAGE_SIZE * KV_CACHE_STRIDE +
                       static_cast<long>(tok0 % PAGE_SIZE) * KV_CACHE_STRIDE) *
                      2;
    i32x4_t const krs =
        make_w_buffer_rsrc(k_base + toff, KV_TILE * KV_CACHE_STRIDE * 2);
    i32x4_t const vrs =
        make_w_buffer_rsrc(v_base + toff, KV_TILE * KV_CACHE_STRIDE * 2);
#pragma unroll
    for (int j = 0; j < 2; j++) {
      int const voff = voff0 + 8 * j * KV_CACHE_STRIDE * 2;
      __llvm_amdgcn_raw_buffer_load_lds(
          krs,
          (__attribute__((address_space(3))) uint32_t *)(stg + t * 4096 +
                                                          j * 1024),
          16, voff, 0, 0, 0);
      __llvm_amdgcn_raw_buffer_load_lds(
          vrs,
          (__attribute__((address_space(3))) uint32_t *)(stg + t * 4096 +
                                                          2048 + j * 1024),
          16, voff, 0, 0, 0);
    }
  }
}
#endif

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
          int NUM_REQS = 1>
#if defined(MPK_FUSED_INLINE)
__device__ __forceinline__ void
    gang_full_layer_fused_kernel_mi300_body(void *const *input_ptrs,
#elif defined(MPK_INLINE_FUSED_LAYER)
__device__ __forceinline__ void
    gang_full_layer_fused_kernel_mi300(void *const *input_ptrs,
#else
__device__ __noinline__ void
    gang_full_layer_fused_kernel_mi300(void *const *input_ptrs,
#endif
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
  //
  // output_ptrs layout:
  //  [0] x_output         [1] k_cache           [2] v_cache
  //  [3] q_workspace      [4] o_acc             [5] attn_proj_out
  //  [6] topk_weight      [7] routing_indices   [8] active_expert_ids
  //  [9] moe_routing_weight [10] moe_workspace_f32

  MPK_ILSTAMP(6, s_ilsub_ml == MPK_ILSUB_L0 + 1);
#ifdef MPK_INTERLAYER_SPLIT
  unsigned long long _il_entry = __builtin_amdgcn_s_memrealtime();
#endif

  int xcd_id;
  MPK_GET_XCC_ID16(xcd_id);
  // HW gives the PHYSICAL die (0-7) even when only a subset runs. Every
  // per-XCD array here is sized for MPK_NUM_XCDS, so fold it. Identity
  // in the default 8-XCD build.
  xcd_id %= MPK_NUM_XCDS;

  int xcd_rank = tile_idx % workers_per_xcd;
  int tid = MPK_TIDX_F;
#ifdef MPK_SEP_PREFETCH
  // MPK_SEP_PREFETCH prefetch: idle ranks (>=23) warm this die's predicted
  // W13 half into cache during attention. Prediction = prev token's expert
  // for this (layer,die). The real TopK still routes, so this never changes
  // the output; a miss only wastes prefetch bandwidth in the idle window.
  {
#ifdef MPK_SEP_ORACLE
    long long const sep_slot = (long long)task_layer_idx * MPK_NUM_XCDS + xcd_id;
    bool const sep_have = (sep_slot >= 0 && sep_slot < MPK_SEP_CAP_N);
#else
    int const sep_slot = (task_layer_idx % 36) * MPK_NUM_XCDS + xcd_id;
    bool const sep_have = (task_layer_idx >= 36);
#endif
#ifdef MPK_SEP_LDS
    extern __shared__ char mpk_stage_smem[];
#endif
    int const sep_w = xcd_rank - 23;
    long long const sep_w13b = g_sep_w13_bytes;
    if (sep_have && sep_w >= 0 && sep_w < 8 && sep_w13b > 0 &&
        input_ptrs[17] != nullptr) {
      int const sep_e = g_sep_cap[sep_slot];
      if (sep_e >= 0 && sep_e < NUM_EXPERTS) {
        int const sep_half = (xcd_id >= MPK_NUM_XCDS / 2) ? 1 : 0;
        long long const sep_off =
            (long long)(sep_e + sep_half * (NUM_EXPERTS - 1)) * (sep_w13b / 2);
        uint8_t const *sep_b =
            static_cast<uint8_t const *>(input_ptrs[17]) + sep_off;
        long long const sep_n = sep_w13b / 2;
        i32x4_t const sep_rsrc = make_w_buffer_rsrc(sep_b, (uint32_t)sep_n);
        long long const sep_share = (sep_n + 7) / 8;
        long long const sep_lo = (long long)sep_w * sep_share;
        long long const sep_hi =
            (sep_lo + sep_share < sep_n) ? (sep_lo + sep_share) : sep_n;
        unsigned sep_sink = 0u;
#ifdef MPK_SEP_LDS
        (void)sep_lo; (void)sep_hi;
        mpk_sep_lds_warm(sep_b, sep_n, sep_w, tid >> 6, tid & 63, mpk_stage_smem);
#else
        for (long long o = sep_lo + (long long)tid * 16; o + 16 <= sep_hi;
             o += 256LL * 16) {
          int4 sv;
          asm volatile("buffer_load_dwordx4 %0, %1, %2, 0 offen sc0"
                       : "=v"(sv) : "v"((uint32_t)o), "s"(sep_rsrc) : "memory");
          sep_sink += sv.x;
        }
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
#ifdef MPK_SEP_W2
        // Also prefetch this die's predicted W2 half (input_ptrs[18]) so
        // BOTH weight streams are MALL-resident (free-both floor = 0.996).
        long long const sep_w2b = g_sep_w2_bytes;
        if (sep_w2b > 0 && input_ptrs[18] != nullptr) {
          long long const sep2_off =
              (long long)(sep_e + sep_half * (NUM_EXPERTS - 1)) * (sep_w2b / 2);
          uint8_t const *sep2_b =
              static_cast<uint8_t const *>(input_ptrs[18]) + sep2_off;
          long long const sep2_n = sep_w2b / 2;
          i32x4_t const sep2_rsrc = make_w_buffer_rsrc(sep2_b, (uint32_t)sep2_n);
          long long const sep2_share = (sep2_n + 7) / 8;
          long long const sep2_lo = (long long)sep_w * sep2_share;
          long long const sep2_hi =
              (sep2_lo + sep2_share < sep2_n) ? (sep2_lo + sep2_share) : sep2_n;
#ifdef MPK_SEP_LDS
          (void)sep2_lo; (void)sep2_hi;
          mpk_sep_lds_warm(sep2_b, sep2_n, sep_w, tid >> 6, tid & 63, mpk_stage_smem);
#else
          for (long long o = sep2_lo + (long long)tid * 16; o + 16 <= sep2_hi;
               o += 256LL * 16) {
            int4 sv2;
            asm volatile("buffer_load_dwordx4 %0, %1, %2, 0 offen sc0"
                         : "=v"(sv2) : "v"((uint32_t)o), "s"(sep2_rsrc) : "memory");
            sep_sink += sv2.x;
          }
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
        }
#endif
        if (sep_sink == 0x7fffffffu) g_sep_cov[1] += sep_sink;
      }
    }
  }
#endif
#ifdef MPK_WS_FARCOPY
  // MPK_WSFC_V2: this layer's W2 epilogue counts into the mailbox; the
  // MoE's internal barriers order this reset before it.
  if (tid == 0) g_wsfc_cnt = 0;
#endif

  // ── Optional: rotate QKV/attention ownership off the O-proj ranks ─────────
  //
  // Phase 7 (O-proj + router + TopK) is owned by ranks
  // [0, oproj_topk_tiles_per_xcd); the ranks above it have no Phase 7 work and
  // therefore reach the MoE phase -- and finish its W2 tail -- ahead of
  // everyone else. Those are precisely the workers that have slack to spend on
  // the *next* layer's QKV weight prefetch and its GEMM, and today they are
  // the ones that never get either: Phase 1 and Phase 3 are both keyed on
  // `xcd_rank < K` with small K, so QKV and attention land on ranks 0..K-1 --
  // the same ranks that are still busy in Phase 7.
  //
  // Rotating by exactly the size of the no-Phase-7 block hands QKV and
  // attention to that block. It is a rotation, not a reassignment: `(r + d) %
  // workers_per_xcd` is a bijection on [0, workers_per_xcd), so every guard of
  // the form `rank < K` still selects exactly K workers and every tile index
  // derived from it is still covered exactly once. Only *which* physical
  // worker runs a given QKV tile or attention chunk changes.
  //
  // O-proj, TopK, MoE and every barrier arrival keep the physical rank. That
  // asymmetry is the point -- rotating those too would just relabel the same
  // overlap. It also keeps the Phase 9 barrier accounting untouched, which
  // counts arrivals per XCD and does not care who did what.
  //
  // The rotation distance is computed from the topology rather than hardcoded
  // for a 32-worker/23-tile layout, so it tracks fleet's 31 workers.
  //
  // ── DEFAULT OFF: it measures slower here, and the reason is structural ────
  //
  // 1.892 ms/token against 1.877 for the same build with the flag off (three
  // runs each, per-run medians, alternating, generated text bit-identical to
  // the reference on all six). The first numbers this produced were faster --
  // 1.854 to 1.871 -- but those runs were wrong: the rotation moved the layer
  // gate's waiter set off its arriver set, and the eight workers per XCD that
  // stopped waiting are the whole apparent gain. See the union computation at
  // MPK_LAYER_GATE_ARRIVES.
  //
  // Once the gate is honest the rotation cannot pay, because fleet already
  // spends the same slack twice over. In that topology ranks 23..31 are idle in the
  // W2 tail, so handing them QKV is free; fleet's are not idle, they are
  // *excused* -- MPK_W2_ONLY_ARRIVE keeps them out of the layer-gate arrival
  // and MPK_W2_CONSUMER_GATE keeps them out of its wait, precisely because
  // they have no second MoE tile. Rotating QKV onto them makes them waiters
  // again, which puts the arrival count back from 22 to 30 of 31 and costs
  // more than the prefetch overlap returns. The two optimizations are
  // competing for one resource; fleet's existing pair wins.
  //
  // Kept, behind the flag, because it is correct and because that trade could
  // invert at a geometry where the no-Phase-7 block and the no-second-tile
  // block stop coinciding.
  int const oproj_topk_ranks =
      oproj_tiles_per_xcd > router_tile_n ? oproj_tiles_per_xcd : router_tile_n;
#ifdef MPK_ROTATE_QKV_ATTN_RANKS
#ifdef MPK_QKV_ROT_OVERRIDE
  // Diagnostic only: pick the rotation distance by hand. Used to separate
  // "the rotation itself is unsound" from "the destination block of ranks is
  // special" -- a distance that lands QKV/attention on ranks that already run
  // Phase 7 exercises the same relabelling without moving the work off the
  // O-proj workers.
  int const qkv_rot = MPK_QKV_ROT_OVERRIDE % workers_per_xcd;
#else
  int const qkv_rot = workers_per_xcd > oproj_topk_ranks
                          ? workers_per_xcd - oproj_topk_ranks
                          : 0;
#endif
  int const qkv_attn_rank = (xcd_rank + qkv_rot) % workers_per_xcd;
#elif defined(MPK_QKV_KV_RANK_SWAP)
#if defined(MPK_QKV_PF_SKIP_W2) || defined(MPK_ABLATE_KV_TILES) || \
    defined(MPK_KV_SPREAD) || defined(MPK_QKV_KSPLIT) || defined(MPK_ATTN_SPLIT_CHUNK)
#error "MPK_QKV_KV_RANK_SWAP is wired for the plain ten-tile QKV rank map"
#endif
  // K/V tiles (qkv_attn_rank 8, 9) on physical ranks R, R+1: movers that
  // finish their W2 group early and stage the K/V weights before the gate.
  // Physical ranks 8-9 take the movers' former qkv_attn_rank.
  constexpr int kKvR = MPK_QKV_KV_RANK_SWAP;
  static_assert(kKvR >= 10 && kKvR + 1 < 31, "MPK_QKV_KV_RANK_SWAP: R, R+1 must be non-QKV ranks");
  int const qkv_attn_rank = xcd_rank == 8          ? kKvR
                            : xcd_rank == 9        ? kKvR + 1
                            : xcd_rank == kKvR     ? 8
                            : xcd_rank == kKvR + 1 ? 9
                                                   : xcd_rank;
#else
  int const qkv_attn_rank = xcd_rank;
#endif
#if defined(MPK_OPROJ_SWAP_ATTN) && defined(MPK_ROTATE_QKV_ATTN_RANKS)
#error "MPK_OPROJ_SWAP_ATTN keeps QKV/attn on the prefix; it is incompatible with rotating those ranks onto the idle block"
#endif
#if defined(MPK_ATTN_SPLIT_CHUNK) && defined(MPK_OPROJ_SWAP_ATTN)
#error "MPK_ATTN_SPLIT_CHUNK uses idle ranks 23-30 for the high half of each attention chunk; incompatible with SWAP_ATTN"
#endif
#if defined(MPK_ATTN_SPLIT_CHUNK) && defined(MPK_ROTATE_QKV_ATTN_RANKS)
#error "MPK_ATTN_SPLIT_CHUNK uses idle ranks 23-30; incompatible with ROTATE_QKV_ATTN_RANKS"
#endif

  // Attention work is (request, kv_head, kv_chunk). kv_head is bound to the
  // XCD (kv_head_idx = xcd_id below), so the request and chunk axes both have
  // to fit in xcd_rank -- the 30 workers on this XCD. The host keeps
  // NUM_KV_CHUNKS = workers_per_xcd / NUM_REQS so this never exceeds 30.
  //
  // NUM_REQS is the *compile-time* MPK_MAX_NUM_BATCHED_REQUESTS, never the live
  // request count. prepare_next_batch pads unused slots so
  // qo_indptr[i] == qo_indptr[i+1], and every attention/merge callee
  // early-returns on that *inside itself* -- so a padded slot still arrives at
  // the chunk barrier below. Keeping the participant set fixed is what makes
  // every modulus here independent of how many requests happen to be live.
  // kv_head was bound to the XCD (kv_head_idx = xcd_id). That is only a
  // tiling when there is one kv head per XCD. At MPK_NUM_XCDS=4 (DPX) with
  // NUM_KV_HEADS=8 it silently drops heads 4..7: they are never computed, so
  // half of the OPROJ_REDUCTION_SIZE-wide reduction reads whatever the
  // previous layer left behind, and the attention arrival count no longer
  // matches the number of merges issued -- which is the Phase 6 barrier
  // hanging one arrival short rather than any memory-ordering bug.
  //
  // So the rank decomposition grows a kv-head axis. Barriers below are keyed
  // by the GLOBAL kv head rather than by xcd_id, which keeps every array that
  // was sized NUM_KV_HEADS-wide correct and is a no-op when the two coincide.
  constexpr int KV_HEADS_PER_XCD = NUM_KV_HEADS / MPK_NUM_XCDS;
  static_assert(NUM_KV_HEADS % MPK_NUM_XCDS == 0,
                "kv heads must divide evenly across XCDs");
  constexpr int ATTN_PARTICIPANTS =
      NUM_REQS * NUM_KV_CHUNKS * KV_HEADS_PER_XCD;
#ifdef MPK_ATTN_KV_OFF
#if defined(MPK_ROTATE_QKV_ATTN_RANKS) || defined(MPK_QKV_KV_RANK_SWAP) || \
    defined(MPK_ATTN_SPLIT_CHUNK) || defined(MPK_KV_SPREAD) || \
    defined(MPK_QKV_KSPLIT) || defined(MPK_ATTN_KV_PREFETCH2)
#error "MPK_ATTN_KV_OFF is wired for the plain ten-tile QKV rank map"
#endif
  static_assert(NUM_REQS == 1 && KV_HEADS_PER_XCD == 1 && ATTN_PARTICIPANTS > 10 &&
                    ATTN_PARTICIPANTS + 2 <= 31,
                "MPK_ATTN_KV_OFF: one request, one KV head per XCD, chunks 8-9 on two free ranks");
  // Attention role, separate from the QKV role: the K/V producers (8, 9) hold no chunk, and
  // chunks 8, 9 run on ranks ATTN_PARTICIPANTS, +1.
  int const attn_rank = (qkv_attn_rank == 8 || qkv_attn_rank == 9) ? ATTN_PARTICIPANTS + 2
                        : qkv_attn_rank == ATTN_PARTICIPANTS       ? 8
                        : qkv_attn_rank == ATTN_PARTICIPANTS + 1   ? 9
                                                                   : qkv_attn_rank;
  constexpr int ATTN_RANK_SPAN = ATTN_PARTICIPANTS + 2;
#else
#ifdef MPK_ATTN_FIXED_MERGER
#if defined(MPK_ROTATE_QKV_ATTN_RANKS) || defined(MPK_QKV_KV_RANK_SWAP) || defined(MPK_ATTN_SPLIT_CHUNK) || \
    defined(MPK_ATTN_KV_PREFETCH2) || !defined(MPK_AID_CHUNK_BAR) || !defined(MPK_AID_SPLIT_FLAGS)
#error "MPK_ATTN_FIXED_MERGER is wired for the plain rank map with the AID chunk-barrier replica"
#endif
  static_assert(NUM_REQS == 6 && NUM_KV_CHUNKS == 5 && KV_HEADS_PER_XCD == 1,
                "MPK_ATTN_FIXED_MERGER: T=6 geometry (ranks 23-28 merge)");
  // Request q's last chunk and its merge on rank 23 + q, which holds no O-proj tile; chunks 0-3 of every
  // request on ranks 0-22 and 29. Same (request, chunk) work and merge order; only the ranks move.
#ifdef MPK_ATTN_FIXED_MERGER_RT
#define MPK_FM_ON (g_mpk_fm_rt != 0)
#else
#define MPK_FM_ON true
#endif
#ifndef MPK_ATTN_FM_V
#define MPK_ATTN_FM_V 0
#endif
#if (MPK_ATTN_FM_V & 1) && !defined(MPK_ATTN_FIXED_MERGER_RT)
  // r < 23: r + r / 4; 23..28: 5 r - 111; 29: 28 (the same map as below).
  int const attn_rank = qkv_attn_rank < 23   ? qkv_attn_rank + (qkv_attn_rank >> 2)
                        : qkv_attn_rank < 29 ? 5 * qkv_attn_rank - 111
                        : qkv_attn_rank == 29 ? 28
                                              : qkv_attn_rank;
#else
  int const fm_i = qkv_attn_rank < 23 ? qkv_attn_rank : 23;
  int const attn_rank = !MPK_FM_ON                                ? qkv_attn_rank
                        : (qkv_attn_rank >= 23 && qkv_attn_rank < 29) ? 5 * (qkv_attn_rank - 23) + 4
                        : qkv_attn_rank < 30                        ? (fm_i / 4) * 5 + (fm_i % 4)
                                                                    : qkv_attn_rank;
#endif
#else
  int const attn_rank = qkv_attn_rank;
#endif
  constexpr int ATTN_RANK_SPAN = ATTN_PARTICIPANTS;
#endif
  int const attn_kvh_local = attn_rank / (NUM_REQS * NUM_KV_CHUNKS);
  int const attn_req = (attn_rank / NUM_KV_CHUNKS) % NUM_REQS;
  int const attn_chunk = attn_rank % NUM_KV_CHUNKS;
#if defined(MPK_KV_SW_IDLE) && defined(MPK_KV_CHUNKS_ADAPTIVE)
  // A sliding window spans at most (W + 30) / 16 tiles; at <= 64 tiles the
  // adaptive split uses 8 chunks, so workers >= 8 hold no tiles.
#ifdef MPK_KV_SW_1TILE
  // One tile per chunk: the window's tiles land on chunks 0..SW1_N-1.
  constexpr int MPK_SW1_N =
      SLIDING_WINDOW > 0
          ? ((SLIDING_WINDOW + 30) / 16 < NUM_KV_CHUNKS
                 ? (SLIDING_WINDOW + 30) / 16
                 : NUM_KV_CHUNKS)
          : NUM_KV_CHUNKS;
  bool const attn_idle_sw = (SLIDING_WINDOW > 0) &&
                            (NUM_KV_CHUNKS > 8) && (attn_chunk >= MPK_SW1_N);
#else
  bool const attn_idle_sw = (SLIDING_WINDOW > 0) &&
                            ((SLIDING_WINDOW + 30) / 16 <= 64) &&
                            (NUM_KV_CHUNKS > 8) && (attn_chunk >= 8);
#endif
#ifdef MPK_KV_FULL_IDLE
  // Full-attention layers: the adaptive chunk count for one token more than
  // the metadata reports. Read before the QKV epoch, the metadata lags the
  // current token by at most one and the count is monotone in the context,
  // so a worker at or above it holds no tiles.
  int attn_ne_up = NUM_KV_CHUNKS;
  if constexpr (SLIDING_WINDOW == 0 && NUM_KV_CHUNKS > 8) {
    if (attn_rank < ATTN_PARTICIPANTS && attn_chunk >= 8) {
      int const np = kv_indptr[attn_req + 1] - kv_indptr[attn_req];
      int const seq = (np - 1) * PAGE_SIZE + kv_last_page_len[attn_req] + 1;
      attn_ne_up = mpk_kv_chunks_eff<NUM_KV_CHUNKS>((seq + 15) / 16);
    }
  }
  bool const attn_idle_full = (SLIDING_WINDOW == 0) && (NUM_KV_CHUNKS > 8) &&
                              (attn_chunk >= attn_ne_up);
#else
  constexpr bool attn_idle_full = false;
  constexpr int attn_ne_up = NUM_KV_CHUNKS;
#endif
  bool const attn_idle = attn_idle_sw || attn_idle_full;
#else
  constexpr bool attn_idle = false;
#endif
  int const attn_kv_head = xcd_id * KV_HEADS_PER_XCD + attn_kvh_local;
#ifdef MPK_ATTN_SPLIT_CHUNK
  static_assert(NUM_REQS == 1,
                "MPK_ATTN_SPLIT_CHUNK maps one idle rank per chunk");
  int const split_idle_n = workers_per_xcd > oproj_topk_ranks
                               ? workers_per_xcd - oproj_topk_ranks
                               : 0;
  int const split_hchunk = xcd_rank - oproj_topk_ranks;
  bool const is_split_helper = (split_idle_n == ATTN_PARTICIPANTS) &&
                               (split_hchunk >= 0) &&
                               (split_hchunk < NUM_KV_CHUNKS);
#endif

  // Optional: swap O-proj tiles [0, ATTN) onto the idle ranks
  // [oproj_topk_ranks, W) so attention workers skip Phase 7.
  //
  // Dual of MPK_ROTATE_QKV_ATTN_RANKS. That flag moves QKV/attn onto the
  // idle ranks and lost because those ranks are the W2_ONLY_ARRIVE
  // excusees -- putting QKV on them made them layer-gate waiters again.
  // This keeps QKV/attn on the prefix (waiter set unchanged) and moves the
  // eight attention-owned O-proj tiles onto ranks that skip QKV+attn, so
  // they can issue the Phase 6 weight DMA during that work and GEMM after
  // attn_release with hot LDS. Hier-barrier arrival stays exactly
  // oproj_topk_ranks: the workers that enter Phase 7. Ranks 0..ATTN-1 wait
  // at the skip-gate the way 23..30 do today.
  //
  // Shipped geometry: W=31, oproj=23, ATTN=8. Ranks 8-22 keep tiles 8-22;
  // ranks 23-30 take tiles 0-7; ranks 0-7 skip. If idle_n != ATTN the swap
  // is a no-op rather than a mismatched arrival count.
#ifdef MPK_OPROJ_SWAP_ATTN
  int const oproj_idle_n = workers_per_xcd > oproj_topk_ranks
                                ? workers_per_xcd - oproj_topk_ranks
                                : 0;
  bool const oproj_do_swap =
      (oproj_idle_n == ATTN_PARTICIPANTS) && (oproj_idle_n > 0);
  bool const does_oproj =
      oproj_do_swap ? (xcd_rank >= ATTN_PARTICIPANTS)
                    : (xcd_rank < oproj_topk_ranks);
  int const oproj_local =
      !does_oproj ? 0
      : (oproj_do_swap && xcd_rank >= oproj_topk_ranks)
            ? (xcd_rank - oproj_topk_ranks)
            : xcd_rank;
#else
  bool const does_oproj = xcd_rank < oproj_topk_ranks;
  int const oproj_local = xcd_rank;
#endif

#ifdef MPK_QKV_KSPLIT
  // 2-way K-split. Ranks [0, T) do K-lo; ranks [W-T, W) do K-hi of the same
  // tile. Prefix mapping (K-hi on 10-19) measured +141 / +92 / +103 µs:
  // those ranks are the ones that overlap O-proj DMA with QKV+attn. High
  // ranks skip O-proj (23-30) or sit at its tail (21-22), so they can take
  // the extra GEMM without stealing that overlap. T=10, W=31 → K-hi on
  // 21-30. QKV epoch is no longer a prefix of 20; it is the two ranges.
  int const qkv_k1_base = workers_per_xcd - total_qkv_tiles_per_xcd;
  bool const qkv_does_k0 = qkv_attn_rank < total_qkv_tiles_per_xcd;
  bool const qkv_does_k1 = qkv_attn_rank >= qkv_k1_base;
  bool const qkv_does_qkv = qkv_does_k0 || qkv_does_k1;
  int const qkv_work_slots = total_qkv_tiles_per_xcd * 2;
  constexpr int QKV_MFMA_ITERS = QKV_REDUCTION_SIZE / 128;
  constexpr int QKV_K_LO = (QKV_MFMA_ITERS + 1) / 2;
#else
#ifdef MPK_KV_SPREAD
#if defined(MPK_ABLATE_KV_TILES) || defined(MPK_QKV_PF_SKIP_W2) ||            \
    defined(MPK_QKV_PF_EARLY) || defined(MPK_QKV_PF_WAVE_SPLIT) ||             \
    defined(MPK_ATTN_Q_EPOCH) || defined(MPK_ROTATE_QKV_ATTN_RANKS)
#error "MPK_KV_SPREAD: K/V ride on the Q ranks; incompatible with this QKV rank map"
#endif
  // K and V run as a second MFMA chain on the NUM_Q_PER_KV Q tiles, so the
  // two K/V ranks (which also carry a W2 group) produce no QKV.
  bool const qkv_does_qkv = qkv_attn_rank < NUM_Q_PER_KV;
  int const qkv_work_slots = NUM_Q_PER_KV;
#elif defined(MPK_VT2) && defined(MPK_VT2_QKV)
  // Token t's T tiles on ranks [t T, (t + 1) T): two batch-1 QKV calls in parallel.
  // MPK_VTN_N7B (QKV): above 2 tokens rank group g = rank / T (3 groups) runs tokens g, g + 3, ... in turn.
  constexpr int qkv_vt2_nt = QKV_BATCH_SIZE == 2 ? 2 : (QKV_BATCH_SIZE > 2 ? 3 : 1);
#ifdef MPK_VTN_QMP
#if !defined(MPK_VTN_QROW) || !defined(MPK_VTN_QROW_TAG) || !defined(MPK_VTN_T) || MPK_VTN_T <= 4 || \
    defined(MPK_VTN_RESTAGE)
#error "MPK_VTN_QMP: VTN above 4 tokens, QROW_TAG, every round on the rank's own tile (no restage)"
#endif
#define MPK_QMP_ON 1
#else
#define MPK_QMP_ON 0
#endif
#ifdef MPK_VTN_QROW
  // MPK_VTN_QROW: rank 3 T (token 3, tile 0) publishes token 3's quantized row in round 1; ranks 1..T-1 take
  // it in round 2, rank 0 skips round 2.
  // MPK_VTN_T6: above 4 tokens the producer still runs token 3 (group 0's round-2 token); the later
  // tokens of groups 1-2 (4, 5, ...) fold and norm themselves.
  static_assert(QKV_BATCH_SIZE >= 4 && QKV_BATCH_SIZE <= 8, "MPK_VTN_QROW: 4..8 tokens");
  bool const qkv_qrow_prod = qkv_attn_rank == qkv_vt2_nt * total_qkv_tiles_per_xcd;
  bool const qkv_does_qkv = qkv_attn_rank < qkv_vt2_nt * total_qkv_tiles_per_xcd || qkv_qrow_prod;
  int const qkv_work_slots = qkv_vt2_nt * total_qkv_tiles_per_xcd + 1;
#else
  bool const qkv_does_qkv = qkv_attn_rank < qkv_vt2_nt * total_qkv_tiles_per_xcd;
  int const qkv_work_slots = qkv_vt2_nt * total_qkv_tiles_per_xcd;
#endif
#else
  bool const qkv_does_qkv = qkv_attn_rank < total_qkv_tiles_per_xcd;
  int const qkv_work_slots = total_qkv_tiles_per_xcd;
#endif
#endif
#if defined(MPK_VT2) && defined(MPK_VT2_QKV)
#if defined(MPK_QKV_KSPLIT) || defined(MPK_KV_SPREAD) || defined(MPK_ROTATE_QKV_ATTN_RANKS) || \
    defined(MPK_QKV_PF_SKIP_W2) || defined(MPK_QKV_PF_DELAY_US) || defined(MPK_QKV_PF_DELAY_NS) || \
    defined(MPK_PFD89_RT) || defined(MPK_P9_KV_EARLY_DMA) || defined(MPK_KV_EARLY_STAGE) || \
    defined(MPK_QKV_PF_ORACLE89) || defined(MPK_ABLATE_KV_TILES) || defined(MPK_ATTN_KV_OFF) || \
    defined(MPK_QKV_KV_PF_PLAIN) || defined(MPK_QKV_KV_PF_SC0) || defined(MPK_ATTN_QWAIT)
#error "MPK_VT2_QKV: these flags key the K/V tiles off the rank, not the tile"
#endif
  constexpr int QKV_TB = QKV_BATCH_SIZE >= 2 ? 1 : QKV_BATCH_SIZE;
  int const qkv_vt2_tok = QKV_BATCH_SIZE >= 2 ? qkv_attn_rank / total_qkv_tiles_per_xcd : 0;
  int const qkv_pf_tile =
      QKV_BATCH_SIZE >= 2 ? qkv_attn_rank - qkv_vt2_tok * total_qkv_tiles_per_xcd : qkv_attn_rank;
#else
  constexpr int QKV_TB = QKV_BATCH_SIZE;
  int const qkv_vt2_tok = 0;
  int const qkv_pf_tile = qkv_attn_rank;
#endif

  // Layer-boundary ACQUIRE for moe_workspace_f32. Plain `buffer_inv` (vL1
  // only) -- NOT `buffer_inv sc1`.
  //
  // ── Why vL1-only is sufficient here, and how that was established ────────
  //
  // This site and four others (see the back-references below) were `sc1`
  // (vL1 + L2). The reasoning was: W2 writes the workspace with `st_wt_f32x4`
  // (sc0 sc1), bypassing L2 into HBM; the producing tile is on a different
  // XCD; MI300/MI350 L2 is not coherent across XCDs; so a line this XCD still
  // holds from last layer's read of the same address would be served stale
  // out of L2.
  //
  // Every step of that is true about the *hardware*. What it omits is the
  // Phase 9 layer barrier at the end of this function, which did not exist
  // when these were written. Phase 9 is a global all-XCD rendezvous whose
  // release path drains stores (`s_waitcnt vmcnt(0)`) and fans out
  // write-through flags. No worker can reach this line for layer N+1 until
  // every worker has passed Phase 9 for layer N, by which point the producers'
  // write-through stores have retired to HBM. The stale-L2 window the `sc1`
  // was closing is a window Phase 9 already closes.
  //
  // Measured, not argued from the memory model. Ablated at HEAD across five
  // configurations, every run compared bitwise against an unablated reference:
  //
  //   B=8 x 5 runs, B=1 x 3 runs, B=16 x 3 runs, 36 layers, 200 positions
  //     -> all KV caches and all output tokens bitwise identical, and
  //        bitwise identical to the `sc1` build. Same arithmetic, not merely
  //        a different-but-stable answer.
  //   `--max-layers 2` x 4 runs at B=8 -> also identical. This one is the
  //        load-bearing check: a 36-layer run streams ~1.7 GB of MoE weights
  //        through a ~4 MB L2 per layer, so a stale line might simply be
  //        evicted by capacity pressure rather than by any guarantee, which
  //        would make the result accidental. At 2 layers that pressure is
  //        largely gone and a stale line can survive. It still matched.
  //
  // Cost recovered: 3.455 -> 2.527 ms/iter at B=1, seq 512 (-27%). The five
  // sites are not individually free (0.03-0.27 ms each) but they overlap; the
  // combined figure is the one that matters.
  //
  // ── What would invalidate this ───────────────────────────────────────────
  //
  // The dependency is on Phase 9. If the layer barrier is weakened, moved
  // above a workspace consumer, or removed, these five sites must go back to
  // `sc1` -- or the ablation must be re-run. Phase 9 is not optional for its
  // own sake either: removing it fails this same gate immediately and loudly
  // (all 72 KV tensors differ, output tokens differ, every run).
  //
  // Note this is an ablation result, not a proof. It says removal is harmless
  // across every gate above; it does not derive safety from the memory model.
  //
  // MPK_W2_CONSUMER_GATE is exactly the "weakened" case this warning names, so
  // under that flag the acquire moves to the consumer gate in Phase 1 and goes
  // back to `sc1`. See the gate site for the full argument.
  //
  // ── The bs>1 arm: one system-scope acquire for the whole layer ────────────
  //
  // Everything above is a bs=1 argument. It leans on Phase 9 having drained
  // every producer's write-through stores before any consumer reaches this
  // line, which is true for the workspace but says nothing about data
  // published *within* a layer by one XCD and read by another -- and at bs>1
  // there is such data: rmsnorm_out_moe, written by one router worker per XCD
  // with a plain global_store into that XCD's non-coherent L2, then read in
  // Phase 8 by workers that never entered Phase 7 (xcd_rank >=
  // oproj_topk_tiles_per_xcd) and so never touched the line on their own die.
  //
  // That was the bs>1 MoE W13 corruption. It was first fixed with a scoped
  // `sc1` at the Phase 7b acquire, patching the one site that reproduced.
  // What ships instead is one system-scope acquire for the whole layer, and
  // the difference in shape is the point.
  //
  // Why the broad form is preferable to the narrow patch: the per-site fix is
  // only known-correct at the site that happened to reproduce. This file has
  // seven bare `buffer_inv` sites, all argued safe from bs=1 ablations, and
  // the bs=4 evidence shows how weak that kind of argument is here -- with the
  // scoped fix REMOVED, bs=4 still reports 0/720 bad weight groups across
  // three reps, not because it is safe but because twelve active experts
  // stream enough MXFP4 weight to evict the stale line by L2 capacity before
  // it is read. The same capacity accident the `--max-layers 2` ablation above
  // was designed to defeat. Correctness that depends on cache pressure is not
  // correctness, so gate a coherence fix at the LOWEST batch that reproduces
  // (bs=2 here), never the highest: bad-group counts fall 30 / 27 / 0 as the
  // batch grows.
  //
  // Cost: `sc0 sc1` here drops L2 once per layer, at a point where the layer is
  // about to stream its weights in anyway, rather than mid-Phase-7 after the
  // MXFP4 expert weights are already resident -- which is what made the
  // unconditional form at that site cost +0.259 ms/token. QKV_BATCH_SIZE is a
  // template parameter, so the bs=1 build compiles to exactly the `#else` and
  // pays nothing.
#ifdef MPK_INTERLAYER_SPLIT
  unsigned long long _il_binv0 = __builtin_amdgcn_s_memrealtime();
#endif
#ifdef MPK_NPS2_L2_ACQUIRE
  // SPX+NPS2 needs this acquire unconditionally, including at QKV_BATCH_SIZE
  // == 1 where the branch below performs none at all.
  //
  // The `#else` is sound only because plain hipMalloc is MTYPE_RW in NPS1, so
  // the hardware keeps L2 coherent across XCDs and a consumer cannot hold a
  // stale line. Under aid_local_xcp_nc=Y a spanning BO is local to neither
  // memory range and is demoted to MTYPE_NC, which removes that guarantee: a
  // line this XCD cached in an earlier layer is served stale, and the release
  // poll built on it spins until the line is evicted by capacity. That is the
  // observed batch-1 deadlock.
  asm volatile("buffer_inv sc0 sc1" ::: "memory");
#else
  if (QKV_BATCH_SIZE > 1) {
#ifdef MPK_BS_LAYER_INV_L1
#if !defined(MPK_P9_PREINV) || defined(MPK_P9_PREINV_SKIP)
#error "MPK_BS_LAYER_INV_L1 leaves L2 to MPK_P9_PREINV's per-die invalidate (and not MPK_P9_PREINV_SKIP)"
#endif
    asm volatile("buffer_inv sc0" ::: "memory");
#elif defined(MPK_VT2) && defined(MPK_VT2_LAYER_INV) && defined(MPK_ROUTER_OPROJ_PARTIAL)
#if MPK_VT2_LAYER_INV == 1
    asm volatile("buffer_inv sc0" ::: "memory");
#else
    asm volatile("buffer_inv" ::: "memory");
#endif
#elif defined(MPK_BS_LAYER_INV_W0)
    if (__builtin_amdgcn_readfirstlane(threadIdx.x) < 64) {
      asm volatile("buffer_inv sc0 sc1" ::: "memory");
    } else {
      asm volatile("buffer_inv sc0" ::: "memory");
    }
#else
    asm volatile("buffer_inv sc0 sc1" ::: "memory");
#endif
  } else {
#ifndef MPK_W2_CONSUMER_GATE
    asm volatile("buffer_inv" ::: "memory");
#endif
  }
#endif
#ifdef MPK_INTERLAYER_SPLIT
  // buffer_inv has no completion counter to wait on, so this brackets issue
  // cost plus whatever the invalidate stalls behind it, not drain latency.
  unsigned long long _il_binv1 = __builtin_amdgcn_s_memrealtime();
#endif

#ifdef MPK_ATTN_KV_PREFETCH
  // Idle ranks (no QKV, no O-proj DMA) issue this XCD's past-KV into L2
  // while ranks 0-9 run QKV. Attention is on a different CU of the same
  // XCD, so L2 hits are what we want; vL1 on these CUs is irrelevant.
  // Mapping is physical idle rank -> chunk, not qkv_attn_rank % CHUNKS
  // (rank 23 would decode as request 2).
  static_assert(NUM_REQS == 1,
                "MPK_ATTN_KV_PREFETCH maps one idle rank per chunk at "
                "NUM_REQS == 1");
  {
    int const idle_n = workers_per_xcd > oproj_topk_ranks
                           ? workers_per_xcd - oproj_topk_ranks
                           : 0;
    int const pf_chunk = xcd_rank - oproj_topk_ranks;
#ifdef MPK_ATTN_KV_PREFETCH2
    if ((SLIDING_WINDOW > 0 || MPK_ATTN_KV_PREFETCH2 >= 2) && idle_n > 0 && pf_chunk >= 0 &&
        pf_chunk < idle_n) {
#else
    if (idle_n == ATTN_PARTICIPANTS && pf_chunk >= 0 &&
        pf_chunk < NUM_KV_CHUNKS) {
#endif
      using bf16_t = __hip_bfloat16;
      char const *k_base = reinterpret_cast<char const *>(
          reinterpret_cast<bf16_t const *>(output_ptrs[1]) +
          MPK_KV_HEAD_OFF(xcd_id, HEAD_DIM));
      char const *v_base = reinterpret_cast<char const *>(
          reinterpret_cast<bf16_t const *>(output_ptrs[2]) +
          MPK_KV_HEAD_OFF(xcd_id, HEAD_DIM));
#ifdef MPK_ATTN_KV_PREFETCH2
      mpk_prefetch_kv_strided_l2<PAGE_SIZE, HEAD_DIM, KV_CACHE_STRIDE>(
          k_base, v_base, kv_indptr, kv_indices, kv_last_page_len,
          /*request_id=*/0, pf_chunk, idle_n, SLIDING_WINDOW);
#else
      mpk_prefetch_kv_chunk_l2<PAGE_SIZE, HEAD_DIM, NUM_KV_CHUNKS,
                               KV_CACHE_STRIDE>(k_base,
                                                v_base,
                                                kv_indptr,
                                                kv_indices,
                                                kv_last_page_len,
                                                /*request_id=*/0,
                                                pf_chunk,
                                                SLIDING_WINDOW);
#endif
    }
  }
#endif

  // NOTE: the layer counter that the MoE W13->W2 barrier derives its release
  // value from is published further down, once qkv_epoch_expected is known.
  // See the LAYER_IDX_SMEM_OFF store just before Phase 1.

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  unsigned long long _fused_t0 = __builtin_amdgcn_s_memrealtime();
  unsigned long long _fused_t0a = 0, _fused_t0b = 0, _fused_t0c = 0,
                     _fused_t0d = 0, _merge_done = 0;
#endif

  // Phase slot index. xcd_rank is fixed for the life of the task and xcd_id
  // is the die, so (xcd, rank) is a stable per-worker identity and the trace's
  // per-rank bands line up across runs.
  int const _pslot_w = xcd_id * workers_per_xcd + xcd_rank;
  MPK_PHASE_MARK(_pslot_w, 0);
#ifdef MPK_VT2_DUMP
  // Debug: previous layer's MoE output (this layer's fold input) per token and slot, first pass.
  if (xcd_id == 0 && xcd_rank == 0 && threadIdx.x == 0 && task_layer_idx >= 1 &&
      task_layer_idx <= 2) {
    float const *const vws = (float const *)input_ptrs[0];
    for (int t = 0; t < QKV_BATCH_SIZE; ++t) {
      double cs[4];
      for (int k = 0; k < 4; ++k) {
        double acc = 0.0;
        for (int i = 0; i < 2880; ++i)
          acc += (double)__hip_atomic_load(&vws[(t * 4 + k) * 2944 + i], __ATOMIC_RELAXED,
                                           __HIP_MEMORY_SCOPE_SYSTEM);
        cs[k] = acc;
      }
      printf("[VT2DUMP] L%d t%d ws %.6e %.6e %.6e %.6e\n", task_layer_idx, t, cs[0], cs[1],
             cs[2], cs[3]);
    }
  }
#endif
#ifdef MPK_QKV_SPLIT
  // Function scope, unconditional read: s_memrealtime is scalar and cheap,
  // and keeping it out of divergent flow is what MPK_LM_INNER got wrong.
  unsigned long long _qkvs_t0 = __builtin_amdgcn_s_memrealtime();
  unsigned long long _qkvs_t1 = _qkvs_t0;
#endif
  MPK_QKVSUB(6);
#if defined(MPK_QKV_KV_L2WARM) || defined(MPK_QKV_L2PRE)
  // Next layer's QKV weight pointer, taken while the TaskDesc slot is known
  // to hold it (the ml loop wrote it before this layer started).
  void const *const mpk_warm_qkvw = input_ptrs[24];
#endif
  MPK_ILSTAMP(7, s_ilsub_ml == MPK_ILSUB_L0 + 1);
#ifdef MPK_ILSUB
  if (tid == 0 && s_ilsub_ml == MPK_ILSUB_L0 + 1) {
    g_ilsub[blockIdx.x * 32 + 10] = (unsigned long long)(_pslot_w + 1);
  }
#endif

#ifdef MPK_INTERLAYER_SPLIT
  // Gated on the same arm as the phase slots, so these three segments are
  // directly subtractable from the PSLOTW slot-0 column. Ungated they would
  // average 70 context-growing prefill iterations into the answer.
  if (tid == 0 && g_phase_arm) {
    unsigned long long _il_slot0 = __builtin_amdgcn_s_memrealtime();
    atomicAdd(&g_il_fnpre_sum, (_il_binv0 - _il_entry) * 10);
    atomicAdd(&g_il_binv_sum, (_il_binv1 - _il_binv0) * 10);
    atomicAdd(&g_il_post_sum, (_il_slot0 - _il_binv1) * 10);
    atomicAdd(&g_il_n, 1ULL);
  }
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
#if defined(MPK_AID_P9_NC)
  // Phase 9's counter base, loaded at entry: next to the arrival every
  // asm "memory" clobber forced a reload onto its critical path.
  int *const p9n_base = g_aid_nc_rep[xcd_id >> 2];
#elif defined(MPK_AID_P9_LOCAL)
  int *const p9_rep_base = g_aid_flag_rep[xcd_id >> 2];
#endif
  int *attn_global = oproj_counters_base + FULL_LAYER_ATTN_GLOBAL_COUNTER_SLOT;
  int *qkv_epoch_shared = oproj_counters_base + FULL_LAYER_QKV_EPOCH_SLOT;
#if defined(MPK_AID_SPLIT_QKV) && defined(MPK_AID_SPLIT_FLAGS)
  // Arrival and both polls are confined to this XCD's own slot, so redirecting
  // the base is the whole change: producer and consumer are on the same die and
  // therefore in the same AID, and cannot end up on different buffers.
  int *qkv_epoch =
      mpk_aid_flags(qkv_epoch_shared, xcd_id, MPK_AID_REGION_QKV_EPOCH);
#else
  int *qkv_epoch = qkv_epoch_shared;
#endif
#ifdef MPK_ATTN_Q_EPOCH
  int *const mpk_qe_count = oproj_counters_base + MPK_ATTN_QEPOCH_SLOT;
  int *const mpk_qe_rel = mpk_qe_count + 8 * 16;
#endif
  int *chunk_barrier = oproj_counters_base + FULL_LAYER_CHUNK_BARRIER_SLOT;
  int *routing_ready = oproj_counters_base + 10 * 16;
  int *attn_release_shared =
      oproj_counters_base + FULL_LAYER_ATTN_XCD_RELEASE_SLOT;
#ifdef MPK_AID_SPLIT_FLAGS
  // Poll the replica homed in this XCD's own AID. Both replicas carry the same
  // eight flags at the same offsets, so every consumer below -- the Phase 6
  // gate here and the per-wave slice wait inside the Phase 7 kernel -- is
  // unchanged apart from which copy it reads.
  int *attn_release = mpk_aid_flags(
      attn_release_shared, xcd_id, MPK_AID_REGION_ATTN_RELEASE);
#else
  int *attn_release = attn_release_shared;
#endif

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
#ifdef MPK_ML_TABLE_INLAYER
  // Next layer's ml-table entry for this wave-1 lane (Phase 9 -> end).
  unsigned long long mlt_v;
  int const mlt_ml = task_layer_idx % MPK_MLT_LAYERS;
  bool const mlt_go = g_aid_ml_in[xcd_id >> 2] != nullptr &&
                      g_aid_ml_out[xcd_id >> 2] != nullptr &&
                      mlt_ml + 1 < MPK_MLT_LAYERS;
#endif
#ifdef MPK_ABIP
  // MPK_ABIP: on-pass predicate (pass = layer / 36, parity MPK_ABIP_PH) for features switched every other pass.
#ifndef MPK_ABIP_PH
#define MPK_ABIP_PH 1
#endif
  bool const mpk_abip_odd = ((layer_counter / 36) & 1) == (MPK_ABIP_PH);
#endif
#if defined(MPK_VTN_AOQ) && (defined(MPK_VTN_AOQ_ABIP) || defined(MPK_VTN_AOQ_MINN))
  // MPK_VTN_AOQ_MINN: FP8 rows only in passes with at least N live tokens (the O-proj tests the same).
  if (threadIdx.x == 0) {
    bool aoq_on = true;
#ifdef MPK_VTN_AOQ_ABIP
    aoq_on = mpk_abip_odd;
#endif
#ifdef MPK_VTN_AOQ_MINN
    aoq_on = aoq_on && num_active_tokens >= (MPK_VTN_AOQ_MINN);
#endif
    s_aoq_on = aoq_on ? 1 : 0;
  }
#endif
  int const routing_expected = layer_counter + 1;
  int const attn_release_expected = layer_counter + 1;
  int const qkv_epoch_expected = layer_counter + 1;
#if defined(MPK_EARLY_ROUTING) || defined(MPK_LOCAL_TOPK)
  // Packed into MoE tile_idx[15:8] as expert_id+1. 0 means "no early expert".
  int routed_expert0 = 0;
#ifdef MPK_VT2
  int routed_expert1 = 0;
#endif
#endif

  // Only the workers that read this layer's QKV output take part in the epoch
  // barrier. The expected values above no longer depend on arrival ordering,
  // so the participant set is free to be the set that actually needs the
  // barrier. This is the pre-f1fa720 participant set, now safe to use because
  // nothing reads a shared counter -- it measured neutral, not faster.
  //
  // The max() is what keeps the guard and the modulus in agreement: attention
  // workers *wait* on this epoch, so they must also be counted as arriving.
  // With multiple requests the attention set is ATTN_PARTICIPANTS, not
  // NUM_KV_CHUNKS, and this has to track it or the modulus never fires.
  int const qkv_epoch_participants = qkv_work_slots > ATTN_RANK_SPAN
                                         ? qkv_work_slots
                                         : ATTN_RANK_SPAN;

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
#ifdef MPK_P9_DEFER
  // The previous layer's gate. input_ptrs[25] is non-null exactly when a
  // previous layer of this task ran, i.e. when the word was written.
  if (tid == 0 && input_ptrs[25] != nullptr) {
    int const p9_target = *mpk_p9_word(input_ptrs);
    if (p9_target > 0) {
      int *const p9_global =
          oproj_counters_base + FULL_LAYER_LAYER_BARRIER_SLOT(NUM_REQS) + 8 * 16;
#ifdef MPK_POLL_PIPE
      mpk_poll_ge_s32(p9_global, p9_target);
#else
      while (MPK_LD_GATE(p9_global) < p9_target) {
#ifndef MPK_LAYER_GATE_BUSY_POLL
        __builtin_amdgcn_s_sleep(1);
#endif
      }
#endif
#ifdef MPK_P9_DEFER_INV_L1
      asm volatile("buffer_inv sc0" ::: "memory");
#else
      asm volatile("buffer_inv" MPK_DEV_SC_INV "" ::: "memory");
#endif
    }
  }
#endif
  __syncthreads();

  // ══════════════════════════════════════════════════════════════════
  // Phase 1: QKV GEMM
  // ══════════════════════════════════════════════════════════════════
#ifdef MPK_QKV_SPLIT
  // Setup and the block barrier end here; the QKV GEMM starts next.
  _qkvs_t1 = __builtin_amdgcn_s_memrealtime();
#endif
#ifdef MPK_GATE_DEFER
#if !defined(MPK_P9_FLAT) || !defined(MPK_W2_CONSUMER_GATE) ||              \
    !defined(MPK_PREFETCH_NEXT_QKV) || !defined(MPK_AID_SPLIT_FLAGS) ||       \
    !defined(MPK_P9_PREINV) || !defined(MPK_LEAN_ARRIVE) ||                   \
    defined(MPK_QKV_KSPLIT) || defined(MPK_QKV_PF_WAVE_SPLIT) ||              \
    MPK_NUM_XCDS != 8
#error "MPK_GATE_DEFER supports the shipped recipe's Phase 9 only"
#endif
  // Layer L-1 deferred its gate: prefetch this layer's QKV weights, then
  // wait for the gate before anything reads what layer L-1 wrote.
  if ((task_layer_idx % MPK_GD_LAYERS) != 0) {
#ifdef MPK_GATE_ATTN_JOIN
    bool const gd_joins =
        qkv_does_qkv || (attn_rank < ATTN_PARTICIPANTS);
#else
    bool const gd_joins = qkv_does_qkv;
#endif
#ifdef MPK_GATE_DEFER_PF_LATE
    if (qkv_does_qkv && input_ptrs[4] != nullptr &&
        input_ptrs[25] == input_ptrs[4]) {
      qkv_prefetch_weights_lds<QKV_TB,
                               QKV_OUTPUT_PER_WG,
                               QKV_REDUCTION_SIZE>(
          input_ptrs[4], qkv_n_wgs_per_xcd, qkv_pf_tile);
    }
#endif
    if (tid == 0 && gd_joins) {
      int *const gd_rel = mpk_aid_flags(
          oproj_counters_base + FULL_LAYER_LAYER_BARRIER_SLOT(NUM_REQS) +
              8 * 16 + 16,
          xcd_id, MPK_AID_REGION_LAYER_RELEASE);
#if defined(MPK_XSNAP) && defined(MPK_PHASE_SNAP) && MPK_XSNAP_LAT == 2
      if (tid == 0) {
        unsigned int const _xl_l =
            s_phase_layers -
            (unsigned int)(MPK_PHASE_SNAP * MPK_PHASE_LAYERS_PER_ITER);
        int const *const _xl_loc = g_rop_rep[xcd_id >> 2] + 103680;
        int const *const _xl_rem = g_rop_rep[1 - (xcd_id >> 2)] + 103680;
        int _xl_v0, _xl_v1;
        unsigned long long const _xl_t0 = __builtin_amdgcn_s_memrealtime();
        unsigned long long const _xl_t1 = __builtin_amdgcn_s_memrealtime();
        asm volatile("global_load_dword %0, %1, off sc0 sc1\n"
                     "s_waitcnt vmcnt(0)"
                     : "=v"(_xl_v0)
                     : "v"(_xl_loc)
                     : "memory");
        unsigned long long const _xl_t2 = __builtin_amdgcn_s_memrealtime();
        asm volatile("global_load_dword %0, %1, off sc0 sc1\n"
                     "s_waitcnt vmcnt(0)"
                     : "=v"(_xl_v1)
                     : "v"(_xl_rem)
                     : "memory");
        unsigned long long const _xl_t3 = __builtin_amdgcn_s_memrealtime();
        if (_xl_l < (unsigned int)MPK_PHASE_LAYERS_PER_ITER &&
            (unsigned int)s_xs_worker < (unsigned int)MPK_PHASE_MAX_WORKERS) {
          unsigned long long *const _xl_p =
              &g_xs_snap[((size_t)_xl_l * MPK_PHASE_MAX_WORKERS + s_xs_worker) * 16];
          _xl_p[12] = _xl_t1 - _xl_t0;
          _xl_p[14] = _xl_t2 - _xl_t1 + (unsigned long long)(_xl_v0 & 0);
          _xl_p[15] = _xl_t3 - _xl_t2 + (unsigned long long)(_xl_v1 & 0);
        }
      }
#endif
      MPK_XS_MARK(0);
      MPK_GS_STAMP(1);
      MPK_ATSP9(12, true);
      while (ld_aid_min8_s32(gd_rel) <= s_gd_prev) {
        __builtin_amdgcn_s_sleep(1);
      }
      MPK_GS_STAMP(2);
      MPK_ATSP9(13, true);
      asm volatile("buffer_inv sc0" ::: "memory");
      MPK_XS_MARK(1);
    }
    __syncthreads();
  }
#endif
  if (tid == 0) MPK_GS_STAMP(3);
  MPK_ATSOPS_AT(14, tid == 0);
#ifdef MPK_ATTN_SL_PRE_TOP
#if !defined(MPK_ATTN_SL_PRE) || !defined(MPK_ATTN_META_LDS) || !defined(MPK_ATTN_META_EARLY)
#error "MPK_ATTN_SL_PRE_TOP stages at the layer top from the MPK_ATTN_META_LDS pass metadata"
#endif
  // This rank's first committed K/V tiles, issued beside the QKV fold's loads (pass metadata from LDS).
  mpk_slpre_t top_slp = MPK_SLPRE_NONE;
  if ((task_layer_idx % MPK_AM_LAYERS) != 0 && attn_rank < ATTN_PARTICIPANTS && s_am_meta[1] > s_am_meta[0]) {
    using bf16_t = __hip_bfloat16;
    top_slp = mpk_attn_sl_prestage<PAGE_SIZE, NUM_KV_CHUNKS, KV_CACHE_STRIDE>(
        reinterpret_cast<char const *>(reinterpret_cast<bf16_t const *>(output_ptrs[1]) +
                                       MPK_KV_HEAD_OFF(attn_kv_head, HEAD_DIM)),
        reinterpret_cast<char const *>(reinterpret_cast<bf16_t const *>(output_ptrs[2]) +
                                       MPK_KV_HEAD_OFF(attn_kv_head, HEAD_DIM)),
        attn_req, attn_chunk, SLIDING_WINDOW, s_am_meta[2], s_am_meta[3], s_am_meta[4], s_am_meta[5]);
  }
#endif
  MPK_TW_SUB(10, xcd_rank);
#ifdef MPK_ABLATE_KV_TILES
#ifndef MPK_QKV_PF_SKIP_W2
#error "MPK_ABLATE_KV_TILES needs MPK_QKV_PF_SKIP_W2"
#endif
  if (qkv_does_qkv && xcd_rank < 8) {
#else
  if (qkv_does_qkv
#if defined(MPK_VT2) && defined(MPK_VT2_QKV)
      && (MPK_QMP_ON || qkv_vt2_tok < num_active_tokens)  // a dead token's ranks still arrive below
#endif
  ) {
#endif
#ifdef MPK_QKV_KSPLIT
    int const qkv_k_part = qkv_does_k1 ? 1 : 0;
    int const qkv_tile =
        qkv_does_k1 ? qkv_attn_rank - qkv_k1_base : qkv_attn_rank;
#else
    int const qkv_tile = qkv_pf_tile;
#endif
#if !defined(MPK_ONLY_OP) || (MPK_ONLY_OP & (1 << 1))
#ifdef MPK_RESID_REP
#ifndef MPK_AID_SPLIT_OUT
#error "MPK_RESID_REP reads the MPK_AID_SPLIT_OUT replica"
#endif
#if defined(MPK_RESID_REP_RING)
#ifndef MPK_RESID_RING_LAYERS
#define MPK_RESID_RING_LAYERS 36
#endif
    // Ring: layer L > 0 of a token reads the previous layer's O-proj row
    // (ring copy L-1), which is what this AID's replica holds.
    void const *const _pro_resid =
        (task_layer_idx % MPK_RESID_RING_LAYERS != 0)
#else
    void const *const _pro_resid =
        (input_ptrs[1] == output_ptrs[5])
#endif
            ? mpk_aid_out_base(input_ptrs[1], xcd_id,
                               QKV_BATCH_SIZE * QKV_REDUCTION_SIZE * 2)
            : input_ptrs[1];
#else
    void const *const _pro_resid = input_ptrs[1];
#endif
    MPK_QKVSUB(7);
#if defined(MPK_VT2) && defined(MPK_VT2_QKV)
    // MPK_VTN_N7B: above 2 tokens a rank runs tokens qkv_vt2_tok, + 3, ... (same tile); later rounds stage
    // their own weights.
    constexpr int vq_rounds = QKV_BATCH_SIZE > 2 ? (QKV_BATCH_SIZE + 2) / 3 : 1;
#if MPK_QMP_ON
    // MPK_VTN_QMP: rank (tile j, slot s) runs its tile for tokens (g + n k) mod T, g = its round-0 index over
    // the 31 ranks, n = ranks on the tile (4 on tile 0, 3 elsewhere): round 0 covers every token, the rank
    // with g = t publishes token t's quantized row, and every later round takes its token's row.
    int const vq_n = qkv_pf_tile == 0 ? 4 : 3;
    int const vq_g = 3 * qkv_pf_tile + (qkv_pf_tile > 0 ? 1 : 0) + qkv_vt2_tok;
    bool vq_did = false;
#endif
#ifdef MPK_VQ_TS
    // MPK_VQ_TS (timing tool): wave 0's time per QKV round into the [TOPKSUB] slots.
    unsigned int vq_ts0 = (unsigned int)__builtin_amdgcn_s_memrealtime();
#endif
#pragma unroll
    for (int vq_r = 0; vq_r < vq_rounds; ++vq_r) {
#if MPK_QMP_ON
    if (qkv_vt2_tok + vq_n * vq_r >= QKV_BATCH_SIZE) break;  // past this rank's share of its tile
    long long const vq_t = (vq_g + vq_n * vq_r) % QKV_BATCH_SIZE;
    if (vq_t >= num_active_tokens) continue;
#ifdef MPK_VTN_QNODRAIN
    if (vq_did) asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");  // LDS only
#else
    if (vq_did) __syncthreads();
#endif
    vq_did = true;
    int const vq_role = vq_r > 0 ? 2 : (vq_g < QKV_BATCH_SIZE ? 1 : 0);
#if defined(MPK_VTN_QTRI) && !defined(MPK_VTN_QPAIR)
#error "MPK_VTN_QTRI extends MPK_VTN_QPAIR"
#endif
#ifdef MPK_VTN_QPAIR
    // MPK_VTN_QPAIR v2: this rank's last two rounds run as one call, the second token as MFMA column 1
    // (take + take, or fold + take when the rank has two rounds).
    int vq_t2 = -1;
#ifdef MPK_VTN_QTRI
    // MPK_VTN_QTRI: a rank with three rounds runs all three in its round-0 call (fold + two takes).
    int vq_t3 = -1;
#endif
    {
      int const vq_nr = (QKV_BATCH_SIZE - qkv_vt2_tok + vq_n - 1) / vq_n;  // this rank's rounds
#ifdef MPK_VTN_QTRI
      if (vq_r + 3 == vq_nr) {
        long long const vq_ta = (vq_g + vq_n * (vq_r + 1)) % QKV_BATCH_SIZE;
        long long const vq_tb = (vq_g + vq_n * (vq_r + 2)) % QKV_BATCH_SIZE;
        if (vq_ta < num_active_tokens) {
          vq_t2 = (int)vq_ta;
          if (vq_tb < num_active_tokens) vq_t3 = (int)vq_tb;
        } else if (vq_tb < num_active_tokens) {
          vq_t2 = (int)vq_tb;
        }
      } else
#endif
      if (vq_r + 2 == vq_nr) {
        long long const vq_tn = (vq_g + vq_n * (vq_r + 1)) % QKV_BATCH_SIZE;
        if (vq_tn < num_active_tokens) vq_t2 = (int)vq_tn;
      }
    }
#endif
#else
    long long const vq_t = qkv_vt2_tok + vq_r * qkv_vt2_nt;
    if (vq_r > 0) {
      if (vq_t >= QKV_BATCH_SIZE || vq_t >= num_active_tokens) break;
      __syncthreads();
    }
#ifdef MPK_VTN_QROW
#if defined(MPK_VTN_T) && MPK_VTN_T > 4
    // MPK_VTN_T6: the producer runs round 0 only; group 0 takes token 3's row in its token-3 round (rank 0
    // skips that round), every other later-round token folds and norms itself.
    if (vq_r > 0 && qkv_qrow_prod) break;
    bool const vq_take = vq_t == qkv_vt2_nt && !qkv_qrow_prod;
    int const vq_role = qkv_qrow_prod ? 1 : (vq_take ? 2 : 0);
    if (vq_take && qkv_pf_tile == 0) continue;  // token 3's tile 0 ran on the producer rank
#else
    int const vq_role = qkv_qrow_prod ? 1 : (vq_r > 0 ? 2 : 0);
    if (vq_r > 0 && qkv_pf_tile == 0) break;  // token 3's tile 0 ran on the producer rank
#endif
#endif
#endif  // MPK_QMP_ON
#define MPK_VQ_WS (void *)((float *)input_ptrs[0] + vq_t * MOE_WS_SLOTS * QKV_REDUCTION_SIZE)
#define MPK_VQ_RES (void const *)((unsigned short const *)_pro_resid + vq_t * QKV_REDUCTION_SIZE)
#define MPK_VQ_NS (void *)((unsigned short *)input_ptrs[3] + vq_t * QKV_REDUCTION_SIZE)
#define MPK_VQ_XO (void *)((unsigned short *)output_ptrs[0] + vq_t * QKV_REDUCTION_SIZE)
#define MPK_VQ_QW (void *)((unsigned short *)output_ptrs[3] + vq_t * q_ws_stride)
#define MPK_VQ_NACT 1
#ifdef MPK_VTN_QROW
#if defined(MPK_VTN_T) && MPK_VTN_T > 4 && !defined(MPK_VTN_RESTAGE)
// MPK_VTN_PRE_ALL (default above 4 tokens; -DMPK_VTN_RESTAGE restores the restage): every later round runs
// the round-1 tile and keeps its weights in LDS (the fold's slabs and gammas stage above G::W_END, the token
// row below G::W_OFF), as a take round already does. T=6 -0.95% (A/B/A/B/A), T=8 -1.3%, lossless.
#define MPK_VQ_PRE
#elif defined(MPK_VTN_T) && MPK_VTN_T > 4
// MPK_VTN_T6: a take round runs the round-1 tile with its weights still in LDS; a later full round restages.
#define MPK_VQ_PRE && (vq_r == 0 || vq_role == 2)
#else
#define MPK_VQ_PRE  // round 2 runs the round-1 tile: its weights are still in LDS
#endif
#else
#define MPK_VQ_PRE && vq_r == 0
#endif
#else
#define MPK_VQ_WS input_ptrs[0]
#define MPK_VQ_RES _pro_resid
#define MPK_VQ_NS input_ptrs[3]
#define MPK_VQ_XO output_ptrs[0]
#define MPK_VQ_QW output_ptrs[3]
#define MPK_VQ_NACT num_active_tokens
#define MPK_VQ_PRE
#endif
    gang_resaddf32_rmsnorm_linear_mxfp4_bias_kvupd_kernel<QKV_TB,
                                                          QKV_OUTPUT_PER_WG,
                                                          QKV_REDUCTION_SIZE,
                                                          ACTUAL_HIDDEN_DIM,
                                                          HEAD_DIM,
                                                          NUM_Q_PER_KV,
                                                          PAGE_SIZE>(
        MPK_VQ_WS,
        MPK_VQ_RES,
        input_ptrs[2],
        MPK_VQ_NS,
        input_ptrs[4],
        input_ptrs[5],
        MPK_VQ_XO,
        output_ptrs[1],
        output_ptrs[2],
        MPK_VQ_QW,
        cos_ptr,
        sin_ptr,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        MPK_VQ_NACT,
        qkv_n_wgs_per_xcd,
        kv_stride,
        q_ws_stride,
        qkv_tile,
#ifdef MPK_PREFETCH_NEXT_QKV
        // Skip the DMA: the previous layer's Phase 9 already staged these
        // exact bytes into this exact LDS region during its barrier spin.
        // input_ptrs[25] is this layer's own weight pointer when the previous
        // layer prefetched and null otherwise (it is null on layer 0, which
        // has no previous layer), so this reduces to "was there a producer".
        // The equality check is not redundant: it is the one place the two
        // ends of the hand-off can disagree, and a mismatch here would be
        // silent wrong numerics rather than a fault.
        /*weights_preloaded=*/input_ptrs[25] == input_ptrs[4] MPK_VQ_PRE
#ifdef MPK_QKV_PF_SKIP_W2
#if !defined(MPK_MOE_W2_REMAP) || defined(MPK_ROTATE_QKV_ATTN_RANKS) || \
    defined(MPK_QKV_KSPLIT) || defined(MPK_QKV_PF_WAVE_SPLIT) ||          \
    defined(MPK_QKV_PF_EARLY)
#error "MPK_QKV_PF_SKIP_W2 is wired for MPK_MOE_W2_REMAP's rank map, unrotated and unsplit"
#endif
            // Ranks >= 8 run a W2 group under MPK_MOE_W2_REMAP and do not
            // stage at Phase 9 (the same test guards the prefetch there).
#ifdef MPK_QKV_PF_SKIP_ALL
            && false
#else
            && xcd_rank < 8
#endif
#endif
#else
        /*weights_preloaded=*/false
#endif
#ifdef MPK_QKV_KSPLIT
        ,
        qkv_k_part
#endif
#ifdef MPK_QKV_POS_CACHE
        // 36 layers per token (as MPK_GD_LAYERS / MPK_RESID_RING_LAYERS).
        ,
        /*pos_cache=*/(task_layer_idx % 36 == 0) ? 1 : 2
#endif
#ifdef MPK_QKV_KV_SHARED_ROW
        ,
        /*kv_row_epoch=*/qkv_epoch_expected
#endif
    
#if defined(MPK_VT2) && defined(MPK_VT2_QKV)
        , (int)vq_t
#ifdef MPK_VTN_QROW
#ifdef MPK_VTN_QPAIR
        , (vq_role != 0 || vq_t2 >= 0) ? qkv_epoch_expected : 0, vq_role, vq_t2
#ifdef MPK_VTN_QTRI
        , vq_t3
#endif
#else
        , vq_role != 0 ? qkv_epoch_expected : 0, vq_role
#endif
#endif
#endif
        );
#if defined(MPK_VT2) && defined(MPK_VT2_QKV)
#ifdef MPK_VQ_TS
    if (tid == 0) {
      unsigned int const vq_ts1 = (unsigned int)__builtin_amdgcn_s_memrealtime();
      if (MPK_TOPK_ARMED()) s_tk_acc[vq_r < 3 ? vq_r : 3] += (vq_ts1 - vq_ts0) * 10;
      vq_ts0 = vq_ts1;
    }
#endif
#ifdef MPK_VTN_QPAIR
    if (vq_t2 >= 0) break;  // the next round ran as this call's column 1
#endif
    }  // MPK_VTN_N7B QKV rounds
#endif
#ifdef MPK_VQ_TS
    if (tid == 0 && MPK_TOPK_ARMED()) s_tk_n++;
#endif
#endif

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
    _fused_t0a = __builtin_amdgcn_s_memrealtime();
#endif
  } // end Phase 1: QKV GEMM
#ifdef MPK_QKV_SPLIT
  if (tid == 0 && g_phase_arm) {
    unsigned long long _qkvs_t2 = __builtin_amdgcn_s_memrealtime();
    atomicAdd(&g_qkv_setup_sum, (_qkvs_t1 - _qkvs_t0) * 10ull);
    atomicAdd(&g_qkv_gemm_sum, (_qkvs_t2 - _qkvs_t1) * 10ull);
    atomicAdd(&g_qkv_n, 1ull);
  }
#endif
#ifdef MPK_XSNAP_EARLY
  MPK_XS_MARK(11);
#endif
  MPK_PHASE_MARK(_pslot_w, 1);
  MPK_ILSTAMP(11, s_ilsub_ml == MPK_ILSUB_L0 + 1);

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
#ifdef MPK_ATTN_META_EARLY
#ifdef MPK_ATTN_SPLIT_CHUNK
#error "MPK_ATTN_META_EARLY is wired for the unsplit attention call"
#endif
  // The attention call's request metadata does not depend on this layer's
  // QKV; fetched here, it lands during the epoch wait.
  int am_q0 = 0, am_q1 = 0, am_k0 = 0, am_k1 = 0, am_lpl = 0;
  int am_pid0 = 0;
  if (attn_rank < ATTN_PARTICIPANTS) {
#ifdef MPK_ATTN_META_LDS
#ifdef MPK_ATTN_META_LDS_RT
    if (g_mpk_metalds_rt == 0 || (task_layer_idx % MPK_AM_LAYERS) == 0) {
#else
    if ((task_layer_idx % MPK_AM_LAYERS) == 0) {
#endif
      am_q0 = qo_indptr[attn_req];
      am_q1 = qo_indptr[attn_req + 1];
      am_k0 = kv_indptr[attn_req];
      am_k1 = kv_indptr[attn_req + 1];
      am_lpl = kv_last_page_len[attn_req];
#ifdef MPK_ATTN_SL_OVERLAP
      am_pid0 = kv_indices[am_k0];
#endif
      if (tid == 0) {
        s_am_meta[0] = am_q0;
        s_am_meta[1] = am_q1;
        s_am_meta[2] = am_k0;
        s_am_meta[3] = am_k1;
        s_am_meta[4] = am_lpl;
        s_am_meta[5] = am_pid0;
      }
    } else {
      am_q0 = s_am_meta[0];
      am_q1 = s_am_meta[1];
      am_k0 = s_am_meta[2];
      am_k1 = s_am_meta[3];
      am_lpl = s_am_meta[4];
      am_pid0 = s_am_meta[5];
    }
#else
    am_q0 = qo_indptr[attn_req];
    am_q1 = qo_indptr[attn_req + 1];
    am_k0 = kv_indptr[attn_req];
    am_k1 = kv_indptr[attn_req + 1];
    am_lpl = kv_last_page_len[attn_req];
#ifdef MPK_ATTN_SL_OVERLAP
    am_pid0 = kv_indices[am_k0];
#endif
#endif
  }
#ifdef MPK_ATTN_SL_PRE
  mpk_slpre_t am_slp = MPK_SLPRE_NONE;
#endif
#endif
#ifdef MPK_MERGE_META_PRE
#ifndef MPK_ATTN_META_EARLY
#error "MPK_MERGE_META_PRE uses the MPK_ATTN_META_EARLY request metadata"
#endif
  // The merge's sink, fetched with the rest of the attention metadata so the
  // merge issues its lse/o_acc loads without waiting for it first.
  float am_sink_log2 = 0.0f;
#ifndef MPK_ATTN_META_LDS
  if (HAS_SINKS && attn_rank < ATTN_PARTICIPANTS) {
    am_sink_log2 =
        merge_splitkv_sink_log2<NUM_Q_PER_KV>(input_ptrs[6], xcd_id);
  }
#endif
#endif
#ifdef MPK_ATTN_PTR_EARLY
  // The attention call's pointer arguments live in the task descriptor. Read
  // here, they stay in registers; read after the epoch's buffer_inv (a memory
  // clobber) they cost a round trip before the call.
  auto const am_p_q = output_ptrs[3];
  auto const am_p_k = output_ptrs[1];
  auto const am_p_v = output_ptrs[2];
  auto const am_p_o = input_ptrs[23];
  auto const am_p_lse = input_ptrs[8];
#ifdef MPK_MERGE_PTR_EARLY
  auto const am_p_attn = output_ptrs[4];
  auto const am_p_sink = input_ptrs[6];
#endif
#define MPK_AMP(expr, early) (early)
#else
#define MPK_AMP(expr, early) (expr)
#endif
  MPK_TW_SUB(20, qkv_epoch_expected);
  MPK_WS_PHASE(20, qkv_epoch_expected, xcd_id);
#ifdef MPK_QKV_KSPLIT
  if (qkv_does_qkv)
#else
  if (qkv_attn_rank < qkv_epoch_participants)
#endif
  {
    __shared__ int s_prev;
#if defined(MPK_QKV_EPOCH_DIRECT) && defined(MPK_AID_QKV_ARRIVE) && \
    defined(MPK_AID_SPLIT_FLAGS)
#ifdef MPK_ATTN_SPLIT_CHUNK
#error "MPK_QKV_EPOCH_DIRECT: split helpers poll qkv_epoch without arriving"
#endif
    int *const qkv_dir = g_aid_flag_rep[xcd_id >> 2]
                             ? g_aid_flag_rep[xcd_id >> 2] +
                                   MPK_AID_QKVARR_BASE_INTS + xcd_id * 16
                             : nullptr;
#else
    int *const qkv_dir = nullptr;
#endif
#ifdef MPK_QKV_SUB_LDS
    unsigned int qs_t0 = 0, qs_t1 = 0, qs_t2 = 0;
    bool const qs_on = (qkv_epoch_expected > 1440);
    if (tid == 0) {
      qs_t0 = (unsigned int)__builtin_amdgcn_s_memrealtime();
      s_qs_n[2] = xcd_id + 1;
    }
#endif
#ifdef MPK_ATTN_Q_EPOCH
    // Issued ahead of the producer drain so its vmcnt(0) covers the latency.
    int mpk_qe_kv0 = 0, mpk_qe_kv1 = 0, mpk_qe_lpl = 0;
    if constexpr (SLIDING_WINDOW == 0 && NUM_REQS == 1) {
      if (tid == 0 && attn_rank < ATTN_PARTICIPANTS) {
        mpk_qe_kv0 = kv_indptr[attn_req];
        mpk_qe_kv1 = kv_indptr[attn_req + 1];
        mpk_qe_lpl = kv_last_page_len[attn_req];
      }
    }
#endif
#ifdef MPK_ATTN_KV_EARLY2
#if !defined(MPK_ATTN_KV_EARLY) || !defined(MPK_QKV_SLAB_LDS) || \
    !defined(MPK_QKV_EPOCH_PRODUCER_DRAIN)
#error "MPK_ATTN_KV_EARLY2 needs MPK_ATTN_KV_EARLY, MPK_QKV_SLAB_LDS and the producer drain"
#endif
    static_assert(MPK_ATTN_STAGE2_OFF +
                          MPK_ATTN_STAGED_TPC(MAX_SEQ_LEN, NUM_KV_CHUNKS) * 4096 <=
                      155600,
                  "MPK_ATTN_KV_EARLY2 staging would reach the layer word");
    // Past K/V for this rank's chunk, issued as soon as QKV returns: the
    // producer drain below absorbs it, well before ranks 8-9 set the epoch.
    if constexpr (SLIDING_WINDOW == 0 && NUM_REQS == 1) {
      if (attn_rank < ATTN_PARTICIPANTS) {
        mpk_attn_stage_kv_hd64<PAGE_SIZE, HEAD_DIM, NUM_KV_CHUNKS,
                               KV_CACHE_STRIDE,
                               MPK_ATTN_STAGED_TPW(MAX_SEQ_LEN,
                                                   NUM_KV_CHUNKS)>(
            reinterpret_cast<char const *>(
                reinterpret_cast<__hip_bfloat16 const *>(output_ptrs[1]) +
                MPK_KV_HEAD_OFF(attn_kv_head, HEAD_DIM)),
            reinterpret_cast<char const *>(
                reinterpret_cast<__hip_bfloat16 const *>(output_ptrs[2]) +
                MPK_KV_HEAD_OFF(attn_kv_head, HEAD_DIM)),
            kv_indices, am_k0, am_k1, am_lpl, attn_chunk);
      }
    }
#endif
#ifdef MPK_ATTN_SL_PRE_PREDRAIN
#if !defined(MPK_ATTN_SL_PRE) || !defined(MPK_ATTN_META_LDS) || !defined(MPK_QKV_EPOCH_PRODUCER_DRAIN) || \
    defined(MPK_ATTN_SL_PRE_TOP)
#error "MPK_ATTN_SL_PRE_PREDRAIN stages before the producer drain from the MPK_ATTN_META_LDS pass metadata"
#endif
    // This rank's first committed K/V tiles, beside the QKV store acks the drain below waits for.
    mpk_slpre_t pd_slp = MPK_SLPRE_NONE;
    if ((task_layer_idx % MPK_AM_LAYERS) != 0 && attn_rank < ATTN_PARTICIPANTS && s_am_meta[1] > s_am_meta[0]) {
      using bf16_t = __hip_bfloat16;
      pd_slp = mpk_attn_sl_prestage<PAGE_SIZE, NUM_KV_CHUNKS, KV_CACHE_STRIDE>(
          reinterpret_cast<char const *>(reinterpret_cast<bf16_t const *>(output_ptrs[1]) +
                                         MPK_KV_HEAD_OFF(attn_kv_head, HEAD_DIM)),
          reinterpret_cast<char const *>(reinterpret_cast<bf16_t const *>(output_ptrs[2]) +
                                         MPK_KV_HEAD_OFF(attn_kv_head, HEAD_DIM)),
          attn_req, attn_chunk, SLIDING_WINDOW, s_am_meta[2], s_am_meta[3], s_am_meta[4], s_am_meta[5]);
    }
#endif
#ifdef MPK_QKV_EPOCH_PRODUCER_DRAIN
    // Publish this workgroup's Q/K/V stores into this XCD's L2 before the
    // arrival atomic. Same intra-XCD pattern as the Phase 4 chunk barrier:
    // s_waitcnt is per-wave, so drain first, then rendezvous, then arrive.
    // The last arriver's epoch bump then means every producer has retired
    // its payload, and the consumer only needs the vL1 `buffer_inv` below
    // -- not the agent fence that `buffer_inv sc1`s this XCD's whole L2.
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __syncthreads();
#endif
#ifdef MPK_QKV_SUB_LDS
    if (tid == 0) qs_t1 = (unsigned int)__builtin_amdgcn_s_memrealtime();
#endif
    MPK_ATSOPS_AT(9, tid == 0);
#ifdef MPK_ATTN_SL_PRE_ARR
#if !defined(MPK_ATTN_SL_PRE) || !defined(MPK_QKV_EPOCH_PRODUCER_DRAIN) || defined(MPK_ATTN_SL_PRE_TOP) || \
    defined(MPK_ATTN_SL_PRE_PREDRAIN) || defined(MPK_ATTN_SL_PRE_POLL1)
#error "MPK_ATTN_SL_PRE_ARR stages between the QKV-epoch producer drain and the arrival atomic"
#endif
    // Behind the producer drain, ahead of the arrival: the arrival does not wait for these loads, and this
    // rank's own epoch observation, which waits behind them, needs them for its first tiles anyway.
    if (!attn_idle && attn_rank < ATTN_PARTICIPANTS && am_q1 > am_q0) {
      using bf16_t = __hip_bfloat16;
      am_slp = mpk_attn_sl_prestage<PAGE_SIZE, NUM_KV_CHUNKS, KV_CACHE_STRIDE>(
          reinterpret_cast<char const *>(reinterpret_cast<bf16_t const *>(MPK_AMP(output_ptrs[1], am_p_k)) +
                                         MPK_KV_HEAD_OFF(attn_kv_head, HEAD_DIM)),
          reinterpret_cast<char const *>(reinterpret_cast<bf16_t const *>(MPK_AMP(output_ptrs[2], am_p_v)) +
                                         MPK_KV_HEAD_OFF(attn_kv_head, HEAD_DIM)),
          attn_req, attn_chunk, SLIDING_WINDOW, am_k0, am_k1, am_lpl, am_pid0);
    }
#endif
#ifdef MPK_QEP_SNAP
    unsigned long long qep_t_arr = 0;
    if (tid == 0) {
      qep_t_arr = __builtin_amdgcn_s_memrealtime();
    }
#endif
    if (tid == 0) {
      // Both counters on this barrier are XCD-private: the arrival line is
      // `[xcd_id]` and the epoch line is `[xcd_id * 16]`, and the only reader
      // of either -- the poll below -- indexes by its own `xcd_id` too. No
      // other XCD ever touches these lines, so the `sc1` that pushed each
      // atomic out to the device coherency point was buying visibility to
      // nobody. `sc0` keeps them in this XCD's L2 where every participant
      // already is.
#if defined(MPK_AID_QKV_ARRIVE) && defined(MPK_AID_SPLIT_FLAGS)
#ifdef MPK_ATTN_Q_EPOCH
#error "MPK_ATTN_Q_EPOCH is wired for the torch-tensor arrival counter, not MPK_AID_QKV_ARRIVE"
#endif
#ifdef MPK_ATTN_QWAIT
    }
    // Lane 0 arrives on the full counter and, for a Q tile, lane 1 on the Q-only counter, in one
    // instruction (both after the producer drain above).
    if (tid == 0 ||
        (tid == 1 && qkv_attn_rank < NUM_Q_PER_KV && g_aid_flag_rep[xcd_id >> 2])) {
      int *const qkv_arr_rep = g_aid_flag_rep[xcd_id >> 2];
      int *const mpk_qep_a =
          tid == 0 ? (qkv_arr_rep ? qkv_arr_rep + MPK_AID_QKVARR_BASE_INTS + xcd_id * 16
                                  : &static_cast<int *>(input_ptrs[7])[xcd_id])
                   : qkv_arr_rep + MPK_AID_QKVQ_BASE_INTS + xcd_id * 16;
      int const mpk_qep_p = MPK_XCD_LOCAL_ATOM_ADD(mpk_qep_a, 1);
      if (tid == 0) {
        s_prev = mpk_qep_p;
      }
#else
      int *const qkv_arr_rep = g_aid_flag_rep[xcd_id >> 2];
#ifdef MPK_QKV_EPOCH_NORET
      if (qkv_dir != nullptr) {
        // DIRECT: the waiters poll this counter itself and nothing reads the old value.
        asm volatile("global_atomic_add %0, %1, off" ::"v"(qkv_dir), "v"(1) : "memory");
        s_prev = 0;
      } else
#endif
      s_prev = MPK_XCD_LOCAL_ATOM_ADD(
          qkv_arr_rep ? qkv_arr_rep + MPK_AID_QKVARR_BASE_INTS + xcd_id * 16
                      : &static_cast<int *>(input_ptrs[7])[xcd_id],
          1);
#endif  // MPK_ATTN_QWAIT
#elif defined(MPK_ATTN_Q_EPOCH)
    }
    // Lane 0 arrives on the full counter and, for a Q tile, lane 1 on the
    // Q-only counter, in one instruction.
    if (tid == 0 || (tid == 1 && qkv_attn_rank < NUM_Q_PER_KV)) {
      int *const mpk_qe_a = tid == 0
                                ? &static_cast<int *>(input_ptrs[7])[xcd_id]
                                : &mpk_qe_count[xcd_id * 16];
      int const mpk_qe_p = MPK_XCD_LOCAL_ATOM_ADD(mpk_qe_a, 1);
      if (tid == 0) {
        s_prev = mpk_qe_p;
      } else if ((mpk_qe_p % NUM_Q_PER_KV) == NUM_Q_PER_KV - 1) {
        MPK_XCD_LOCAL_ATOM_ADD(&mpk_qe_rel[xcd_id * 16], 1);
      }
#else
      s_prev = MPK_XCD_LOCAL_ATOM_ADD(
          &static_cast<int *>(input_ptrs[7])[xcd_id], 1);
#endif
    }
    __syncthreads();
#ifdef MPK_QKV_SUB_LDS
    if (tid == 0) qs_t2 = (unsigned int)__builtin_amdgcn_s_memrealtime();
#endif

    if ((s_prev % qkv_epoch_participants) == qkv_epoch_participants - 1) {
      // Last worker to arrive: bump epoch (no reset needed — modular check)
      if (tid == 0 && qkv_dir == nullptr) {
        MPK_XCD_LOCAL_ATOM_ADD(&qkv_epoch[xcd_id * 16], 1);
      }
    }

    // Every participant polls. The participant set is now exactly the workers
    // that need this barrier, so there is no longer a subset that arrives only
    // to carry ordering for someone else.
#ifdef MPK_ATTN_KV_EARLY
#ifndef MPK_ATTN_META_EARLY
#error "MPK_ATTN_KV_EARLY needs MPK_ATTN_META_EARLY"
#endif
// MPK_KVEHM_PATCHED: head-major K/V staged via MPK_KV_HEAD_OFF (see mpk_attn_stage_kv_hd64).
#if defined(MPK_KV_FULL_TPC)
#error "MPK_ATTN_KV_EARLY sizes its staging for the default adaptive chunk rule, not MPK_KV_FULL_TPC"
#endif
#ifdef MPK_ATTN_NO_WAVE_LOCAL
#error "MPK_ATTN_KV_EARLY stages for the wave-local scan"
#endif
    // Past K/V for this rank's chunk, into LDS while the epoch is pending.
    // Issued after the arrival so wave 0's arrival atomic does not wait on it.
#ifndef MPK_ATTN_KV_EARLY2
    if constexpr (SLIDING_WINDOW == 0 && NUM_REQS == 1) {
      if (attn_rank < ATTN_PARTICIPANTS) {
        mpk_attn_stage_kv_hd64<PAGE_SIZE, HEAD_DIM, NUM_KV_CHUNKS,
                               KV_CACHE_STRIDE,
                               MPK_ATTN_STAGED_TPW(MAX_SEQ_LEN,
                                                   NUM_KV_CHUNKS)>(
            reinterpret_cast<char const *>(
                reinterpret_cast<__hip_bfloat16 const *>(output_ptrs[1]) +
                MPK_KV_HEAD_OFF(attn_kv_head, HEAD_DIM)),
            reinterpret_cast<char const *>(
                reinterpret_cast<__hip_bfloat16 const *>(output_ptrs[2]) +
                MPK_KV_HEAD_OFF(attn_kv_head, HEAD_DIM)),
            kv_indices, am_k0, am_k1, am_lpl, attn_chunk);
      }
    }
#endif
#endif
#if defined(MPK_ATTN_META_LDS) && defined(MPK_MERGE_META_PRE)
    // The merge's sink, issued after the arrival (the merge is ~3 us away).
    if (HAS_SINKS && attn_rank < ATTN_PARTICIPANTS) {
      am_sink_log2 = merge_splitkv_sink_log2<NUM_Q_PER_KV>(input_ptrs[6], xcd_id);
    }
#endif
#ifdef MPK_ATTN_SL_PRE_POLL1
#if !defined(MPK_ATTN_SL_PRE) || !defined(MPK_QKV_EPOCH_DIRECT) || defined(MPK_ATTN_QWAIT) || \
    defined(MPK_QKV_POLL_AGENT) || defined(MPK_QKV_EPOCH_POLL_SC1) || defined(MPK_QKV_EPOCH_POLL_ARRIVE) || \
    defined(MPK_ATTN_Q_EPOCH)
#error "MPK_ATTN_SL_PRE_POLL1 hoists the plain DIRECT epoch poll's first load"
#endif
    // First look at the epoch before the staged loads go out (vmcnt retires in order).
    int slp_obs0 = 0;
    if (tid == 0 && !attn_idle) {
      slp_obs0 = __atomic_load_n(qkv_dir ? qkv_dir : &qkv_epoch[xcd_id * 16], __ATOMIC_RELAXED);
    }
    asm volatile("" ::: "memory");
#endif
#ifdef MPK_ATTN_SL_PRE
#ifndef MPK_ATTN_META_EARLY
#error "MPK_ATTN_SL_PRE uses the MPK_ATTN_META_EARLY request metadata"
#endif
    // This rank's first committed K/V tiles into registers while the epoch is pending (shared-LDS loop);
    // issued after the arrival so wave 0's arrival atomic does not wait on them.
#ifndef MPK_ATTN_SL_PRE_ARR
    if (!attn_idle && attn_rank < ATTN_PARTICIPANTS && am_q1 > am_q0) {
      using bf16_t = __hip_bfloat16;
#ifdef MPK_ATTN_SL_PRE_TOP
      if ((task_layer_idx % MPK_AM_LAYERS) != 0) {
        am_slp = top_slp;
      } else
#endif
#ifdef MPK_ATTN_SL_PRE_PREDRAIN
      if ((task_layer_idx % MPK_AM_LAYERS) != 0) {
        am_slp = pd_slp;
      } else
#endif
      am_slp = mpk_attn_sl_prestage<PAGE_SIZE, NUM_KV_CHUNKS, KV_CACHE_STRIDE>(
          reinterpret_cast<char const *>(reinterpret_cast<bf16_t const *>(MPK_AMP(output_ptrs[1], am_p_k)) +
                                         MPK_KV_HEAD_OFF(attn_kv_head, HEAD_DIM)),
          reinterpret_cast<char const *>(reinterpret_cast<bf16_t const *>(MPK_AMP(output_ptrs[2], am_p_v)) +
                                         MPK_KV_HEAD_OFF(attn_kv_head, HEAD_DIM)),
          attn_req, attn_chunk, SLIDING_WINDOW, am_k0, am_k1, am_lpl, am_pid0);
    }
#endif  // !MPK_ATTN_SL_PRE_ARR
#endif
#ifdef MPK_ATTN_KV_PF_SELF
#ifndef MPK_ATTN_META_EARLY
#error "MPK_ATTN_KV_PF_SELF uses the MPK_ATTN_META_EARLY request metadata"
#endif
#ifndef MPK_KV_PF_SELF_RANKS
#define MPK_KV_PF_SELF_RANKS 8
#endif
    // This rank's own past-KV chunk into L2 while the QKV epoch is pending:
    // Q ranks reach the epoch ~2 us before the K/V ranks release it. Wave 0
    // polls; waves 1-3 load one dword per 128 B token row, so every line of
    // the chunk is touched in either KV layout. Same split as the scan; the
    // last global tile is skipped (this QKV writes the current token there).
    static_assert(HEAD_DIM * 2 == 128, "one line per token row");
    if constexpr (NUM_REQS == 1) {
      if (!attn_idle && qkv_attn_rank < MPK_KV_PF_SELF_RANKS && tid >= 64) {
        int const pf_np = am_k1 - am_k0;
        if (pf_np > 0) {
          int const pf_sk = (pf_np - 1) * PAGE_SIZE + am_lpl;
          int pf_ks = 0;
          if (SLIDING_WINDOW > 0 && pf_sk > SLIDING_WINDOW) {
            pf_ks = ((pf_sk - SLIDING_WINDOW) / 16) * 16;
          }
          int const pf_nt = (pf_sk - pf_ks + 15) / 16;
#ifdef MPK_KV_SW_1TILE
          int const pf_ne =
              SLIDING_WINDOW > 0
                  ? (pf_nt < 1 ? 1 : (pf_nt < NUM_KV_CHUNKS ? pf_nt : NUM_KV_CHUNKS))
                  : mpk_kv_chunks_eff<NUM_KV_CHUNKS>(pf_nt);
#else
          int const pf_ne = mpk_kv_chunks_eff<NUM_KV_CHUNKS>(pf_nt);
#endif
          int const pf_tpc = (pf_nt + pf_ne - 1) / pf_ne;
          int const pf_t0 = attn_chunk * pf_tpc;
          int pf_t1 = pf_t0 + pf_tpc;
          if (pf_t1 > pf_nt - 1) {
            pf_t1 = pf_nt - 1;
          }
          int const pf_rows = (pf_t1 - pf_t0) * 16;
          using bf16_t = __hip_bfloat16;
          char const *pf_k = reinterpret_cast<char const *>(
              reinterpret_cast<bf16_t const *>(output_ptrs[1]) +
              MPK_KV_HEAD_OFF(attn_kv_head, HEAD_DIM));
          char const *pf_v = reinterpret_cast<char const *>(
              reinterpret_cast<bf16_t const *>(output_ptrs[2]) +
              MPK_KV_HEAD_OFF(attn_kv_head, HEAD_DIM));
          int const pf_nthr = (int)blockDim.x - 64;
          for (int i0 = tid - 64; i0 < 2 * pf_rows; i0 += 2 * pf_nthr) {
            uint32_t pf_x[2] = {0u, 0u};
#pragma unroll
            for (int u = 0; u < 2; u++) {
              int const i = i0 + u * pf_nthr;
              if (i < 2 * pf_rows) {
                int const tok = pf_ks + pf_t0 * 16 + (i >> 1);
                int const pid = kv_indices[am_k0 + tok / PAGE_SIZE];
                long const off = (static_cast<long>(pid) * PAGE_SIZE +
                                  tok % PAGE_SIZE) *
                                 KV_CACHE_STRIDE * 2;
                pf_x[u] = *(__attribute__((address_space(1))) uint32_t const *)(
                    ((i & 1) ? pf_v : pf_k) + off);
              }
            }
            asm volatile("" ::"v"(pf_x[0]), "v"(pf_x[1]) : "memory");
          }
        }
      }
    }
#endif
#ifdef MPK_QEP_SNAP
    unsigned long long qep_t_p0 = 0, qep_t_p1 = 0;
    int qep_spins = 0;
#endif
    MPK_WS_WAIT_BEGIN(20, qkv_epoch_expected);
    if (tid == 0 && !attn_idle) {
      int _obs;
      int _spins = 0;
#ifdef MPK_QEP_SNAP
      qep_t_p0 = __builtin_amdgcn_s_memrealtime();
#endif
#if defined(MPK_QKV_EPOCH_POLL_SC1) || defined(MPK_QKV_EPOCH_POLL_ARRIVE) || \
    defined(MPK_ATTN_Q_EPOCH)
#ifdef MPK_QKV_EPOCH_DIRECT
#error "upstream's QKV epoch poll variants are not wired for MPK_QKV_EPOCH_DIRECT"
#endif
      int *mpk_qe_poll = &qkv_epoch[xcd_id * 16];
#ifdef MPK_QKV_EPOCH_POLL_ARRIVE
#if defined(MPK_ATTN_Q_EPOCH) || defined(MPK_QKV_EPOCH_POLL_SC1) || \
    defined(MPK_QKV_KSPLIT) || defined(MPK_ATTN_SPLIT_CHUNK)
#error "MPK_QKV_EPOCH_POLL_ARRIVE is wired for the single-counter epoch"
#endif
      // The arrival counter itself, up to this layer's full count: waiters
      // see the last arrival when it lands, not after the last arriver's
      // atomic returns and its bump lands.
      int *const mpk_qa_ctr = &static_cast<int *>(input_ptrs[7])[xcd_id];
      int const mpk_qa_target =
          (s_prev / qkv_epoch_participants + 1) * qkv_epoch_participants;
#endif
#ifdef MPK_ATTN_Q_EPOCH
      if constexpr (SLIDING_WINDOW == 0 && NUM_REQS == 1) {
        if (attn_rank < ATTN_PARTICIPANTS) {
          // The attention call's partition: 16-token tiles over seqlen_k,
          // ceil(ntiles / NUM_KV_CHUNKS) tiles per chunk.
          int const sk = (mpk_qe_kv1 - mpk_qe_kv0 - 1) * PAGE_SIZE + mpk_qe_lpl;
          int const nt = (sk + 15) / 16;
          int const tpc = (nt + NUM_KV_CHUNKS - 1) / NUM_KV_CHUNKS;
          if (nt > 0 && attn_chunk != (nt - 1) / tpc) {
            mpk_qe_poll = &mpk_qe_rel[xcd_id * 16];
          }
        }
      }
#endif
#ifdef MPK_QKV_EPOCH_POLL_SC1
      auto qkv_epoch_ld = [&]() {
        int v;
        asm volatile("global_load_dword %0, %1, off sc1\n\ts_waitcnt vmcnt(0)"
                     : "=v"(v)
                     : "v"(&qkv_epoch[xcd_id * 16])
                     : "memory");
        return v;
      };
      while ((_obs = qkv_epoch_ld()) < qkv_epoch_expected) {
#elif defined(MPK_QKV_EPOCH_POLL_ARRIVE)
      _obs = 0;
      while (s_prev + 1 != mpk_qa_target &&
             (_obs = __atomic_load_n(mpk_qa_ctr, __ATOMIC_RELAXED)) <
                 mpk_qa_target) {
#else
      while ((_obs = __atomic_load_n(mpk_qe_poll, __ATOMIC_RELAXED)) <
             qkv_epoch_expected) {
#endif
#else
#ifdef MPK_ATTN_QWAIT
      int *_qp = qkv_dir ? qkv_dir : &qkv_epoch[xcd_id * 16];
      int _qt = qkv_dir ? qkv_epoch_participants * qkv_epoch_expected
                        : qkv_epoch_expected;
      if constexpr (NUM_REQS == 1) {
        // K/V producers arrive last: they poll the full epoch, no test on their path.
        if (MPK_QWAIT_ON && qkv_dir != nullptr && attn_rank < ATTN_PARTICIPANTS &&
            !(qkv_attn_rank >= NUM_Q_PER_KV && qkv_attn_rank < qkv_work_slots)) {
          // Landed: the producer drain above waited for the META_EARLY loads.
          int const sk = (am_k1 - am_k0 - 1) * PAGE_SIZE + am_lpl;
          if (sk > 0 &&
              !mpk_qep_holds_newest<NUM_KV_CHUNKS, SLIDING_WINDOW>(attn_chunk, sk) &&
              !mpk_qep_holds_newest<NUM_KV_CHUNKS, SLIDING_WINDOW>(attn_chunk, sk + 1)) {
            // No tile of this token's K/V in this chunk: the 8 Q heads are all it reads.
            _qp = qkv_dir + (MPK_AID_QKVQ_BASE_INTS - MPK_AID_QKVARR_BASE_INTS);
            _qt = NUM_Q_PER_KV * qkv_epoch_expected;
          }
        }
      }
#else
      int *const _qp = qkv_dir ? qkv_dir : &qkv_epoch[xcd_id * 16];
      int const _qt = qkv_dir ? qkv_epoch_participants * qkv_epoch_expected
                              : qkv_epoch_expected;
#endif  // MPK_ATTN_QWAIT
#ifdef MPK_ATTN_SL_PRE_POLL1
      _obs = slp_obs0;
      if (slp_obs0 < _qt)
#endif
#ifdef MPK_QKV_POLL_AGENT
      while (!MPK_WS_SK(1) &&
             (_obs = __hip_atomic_load(_qp, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) < _qt) {
#else
      while (!MPK_WS_SK(1) &&
             (_obs = __atomic_load_n(_qp, __ATOMIC_RELAXED)) < _qt) {
#endif
#endif
        MPK_WS_WAIT_TICK(_obs, _spins);
        _spins++;
        __builtin_amdgcn_s_sleep(1);
      }
#ifdef MPK_QEP_SNAP
      qep_t_p1 = __builtin_amdgcn_s_memrealtime();
      qep_spins = _spins;
#endif
#if !defined(MPK_QKV_GATE_NO_AGENT_FENCE) &&                                    \
    !defined(MPK_QKV_EPOCH_PRODUCER_DRAIN)
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, MPK_DEV_FENCE_ACQ);
#endif
    }
    __syncthreads();
#ifdef MPK_QEP_SNAP
    if (tid == 0) {
      unsigned const qep_l = (unsigned)(layer_counter - (MPK_QEP_SNAP) * 36);
      unsigned const qep_w = (unsigned)(xcd_id * workers_per_xcd + xcd_rank);
      if (qep_l < 36u && qep_w < 256u) {
        g_qep_snap[(qep_l * 256u + qep_w) * 2u] = qep_t_arr;
        g_qep_snap[(qep_l * 256u + qep_w) * 2u + 1u] = __builtin_amdgcn_s_memrealtime();
        g_qep_snap2[(qep_l * 256u + qep_w) * 2u] = qep_t_p0;
        g_qep_snap2[(qep_l * 256u + qep_w) * 2u + 1u] = qep_t_p1;
        g_qep_spins[qep_l * 256u + qep_w] = (unsigned)qep_spins;
      }
    }
#endif
#ifdef MPK_QKV_SUB_LDS
    if (tid == 0 && qs_on) {
      unsigned int const qs_t3 = (unsigned int)__builtin_amdgcn_s_memrealtime();
      s_qs_acc[0] += (qs_t1 - qs_t0) * 10;
      s_qs_acc[1] += (qs_t2 - qs_t1) * 10;
      s_qs_acc[2] += (qs_t3 - qs_t2) * 10;
      s_qs_n[0]++;
    }
#endif
    // The agent-scope fence above is LOAD-BEARING unless the producer already
    // drained Q/K/V into this XCD's L2 before arriving (MPK_QKV_EPOCH_PRODUCER_DRAIN).
    // Do not drop it alone (MPK_QKV_GATE_NO_AGENT_FENCE): that is a ctx-4096
    // race. The drain arm is the Phase 4 publish protocol applied here.
    //
    // The argument for dropping it (MPK_QKV_GATE_NO_AGENT_FENCE, now default
    // off) was: both counters here are XCD-private and written with sc0
    // (MPK_XCD_LOCAL_ATOM_ADD), so the fence's `buffer_inv sc1` throws away
    // this XCD's whole L2 -- including the weights the next phase reads -- to
    // acquire a line that never left it, and the vL1 `buffer_inv` below is the
    // acquire the *counter* needs.
    //
    // Every clause of that is true. The conclusion does not follow. An acquire
    // fence does not only order the load of the flag; it orders every load
    // AFTER it against every write the releaser made BEFORE its release. What
    // this gate publishes is not the counter, it is the QKV output the
    // attention phase is about to read. The vL1-only `buffer_inv` drops this
    // CU's stale lines but carries no such ordering, so the reader can observe
    // the bumped epoch and still miss the writes it was standing for.
    //
    // Measured, not argued: at ctx 4096 three byte-identical runs produced
    // three different token sequences with the fence dropped, and 5/5 runs are
    // identical with it restored (matching the all-flags-off reference). At
    // ctx 512 both forms give the same text hash -- the same seq-len
    // dependence the Phase 4 comment below describes, and the reason an A/B at
    // 512 scored this as free. It costs 1.945 -> 1.981 ms/tok at bs=1.
    asm volatile("buffer_inv" ::: "memory");
  }

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  _fused_t0b = __builtin_amdgcn_s_memrealtime();
#endif
#ifdef MPK_XSNAP_EARLY
  MPK_XS_MARK(12);
#endif
  MPK_PHASE_MARK(_pslot_w, 2);
  MPK_ATSTAMP_L(0);

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
    if (attn_rank < ATTN_PARTICIPANTS) {
      int kv_chunk_idx = attn_chunk;
#ifdef MPK_KV_CHUNKS_ADAPTIVE
      // Set by this worker's own attention call; the merger is always one
      // of the chunk workers.
#if defined(MPK_KV_SW_IDLE)
#ifdef MPK_KV_SW_1TILE
      int attn_live_chunks =
          attn_idle ? (attn_idle_full ? attn_ne_up
                                      : (MPK_SW1_N > 8 ? MPK_SW1_N : 8))
                    : NUM_KV_CHUNKS;
#else
      int attn_live_chunks =
          attn_idle ? (attn_idle_full ? attn_ne_up : 8) : NUM_KV_CHUNKS;
#endif
#else
      int attn_live_chunks = attn_idle ? 8 : NUM_KV_CHUNKS;
#endif
#endif
      using bf16_t = __hip_bfloat16;
      void const *offset_k =
          reinterpret_cast<bf16_t const *>(MPK_AMP(output_ptrs[1], am_p_k)) +
          MPK_KV_HEAD_OFF(attn_kv_head, HEAD_DIM);
      void const *offset_v =
          reinterpret_cast<bf16_t const *>(MPK_AMP(output_ptrs[2], am_p_v)) +
          MPK_KV_HEAD_OFF(attn_kv_head, HEAD_DIM);

      // Write float32 partials to o_acc_f32 (input_ptrs[23])
      // Write LSE to lse_acc (input_ptrs[8])
      // NO sinks for per-chunk — sinks applied in merge step
      MPK_ATSOPS_AT(1, tid == 0);
#ifdef MPK_ATTN_SETPRIO
      asm volatile("s_setprio 1");
#endif
#if !defined(MPK_ONLY_OP) || (MPK_ONLY_OP & (1 << 3))
#ifdef MPK_VT2_ATTN_ORACLE
      if (!attn_idle && attn_req == 0)  // timing only: request 1's chunks skip the call
#else
      if (!attn_idle)
#endif
#ifdef MPK_KV_CHUNKS_ADAPTIVE
      attn_live_chunks =
#endif
      paged_attention_ck_fmha_split_kv_impl<bfloat16,
                                            NUM_Q_PER_KV,
                                            HEAD_DIM,
                                            PAGE_SIZE,
                                            MAX_SEQ_LEN,
                                            NUM_KV_CHUNKS,
                                            Q_WORKSPACE_STRIDE,
                                            KV_CACHE_STRIDE,
                                            NUM_KV_HEADS,
                                            DECODE_ONLY
#ifdef MPK_ATTN_META_EARLY
                                            ,
                                            true
#ifdef MPK_ATTN_KV_EARLY
                                            ,
                                            SLIDING_WINDOW == 0
#endif
#endif
                                            >(
          MPK_AMP(output_ptrs[3], am_p_q), // q_workspace
          const_cast<void *>(offset_k),
          const_cast<void *>(offset_v),
          MPK_AMP(input_ptrs[23], am_p_o), // o_acc_f32 (float32 partials)
          MPK_AMP(input_ptrs[8], am_p_lse), // lse_acc
          qo_indptr,
          kv_indptr,
          kv_indices,
          kv_last_page_len,
          /*request_id=*/attn_req,
          /*kv_head_idx=*/attn_kv_head,
          kv_chunk_idx,
          attn_scale,
          SLIDING_WINDOW,
          nullptr // no sinks per-chunk
#ifdef MPK_ATTN_META_EARLY
          ,
          /*split_part=*/0,
          am_q0,
          am_q1,
          am_k0,
          am_k1,
          am_lpl
#ifdef MPK_ATTN_SL_OVERLAP
          ,
          am_pid0
#endif
#ifdef MPK_ATTN_SL_PRE
          ,
          am_slp
#endif
#endif
      );
#endif
#if defined(MPK_KV_FULL_IDLE) && defined(MPK_KV_SW_IDLE)
      // Inside the merge's loop bound an idle worker's slot still needs the
      // empty-chunk LSE the attention call would have stamped.
      if (attn_idle_full) {
        int const bucket =
            attn_ne_up <= 8 ? 8 : (attn_ne_up <= 16 ? 16 : NUM_KV_CHUNKS);
        if (attn_chunk < bucket && tid < NUM_Q_PER_KV) {
          constexpr int LSE_STRIDE = NUM_KV_HEADS * NUM_KV_CHUNKS * NUM_Q_PER_KV;
          reinterpret_cast<float *>(input_ptrs[8])
              [qo_indptr[attn_req] * LSE_STRIDE +
               attn_kv_head * NUM_KV_CHUNKS * NUM_Q_PER_KV +
               attn_chunk * NUM_Q_PER_KV + tid] = -1e30f;
        }
      }
#endif
#ifdef MPK_ATTN_SETPRIO
      asm volatile("s_setprio 0");
#endif
      MPK_ATSOPS_AT(2, tid == 0);

#ifdef MPK_ATTN_SPLIT_CHUNK
      // Drain this chunk's low-half stores before the fold reads them.
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      __syncthreads();
      // Helper published this chunk's high-half (O, LSE). Fold into the
      // primary slot before the 8-way merge so NUM_KV_CHUNKS stays 8.
      {
        int *split_flag =
            &chunk_barrier[(attn_kv_head * NUM_REQS + attn_req) * 16 + 1 +
                           kv_chunk_idx];
        if (tid == 0) {
          while (!MPK_WS_SK(2) && MPK_LD_GATE_AID(split_flag) < qkv_epoch_expected) {
            __builtin_amdgcn_s_sleep(1);
          }
        }
        __syncthreads();
        asm volatile("buffer_inv" ::: "memory");
        mpk_fold_split_chunk_partials<NUM_Q_PER_KV,
                                      HEAD_DIM,
                                      NUM_KV_HEADS,
                                      NUM_KV_CHUNKS>(
            reinterpret_cast<float *>(input_ptrs[8]),
            reinterpret_cast<float *>(input_ptrs[23]),
            qo_indptr[attn_req],
            xcd_id,
            kv_chunk_idx);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        __syncthreads();
      }
#endif

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
      _fused_t0c = __builtin_amdgcn_s_memrealtime();
#endif

      // ══════════════════════════════════════════════════════════════════
      // Phase 4: Chunk barrier — last-chunk-worker runs merge
      // ══════════════════════════════════════════════════════════════════
      MPK_TW_SUB(40, kv_chunk_idx);
      MPK_ATSTAMP_L(1);
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
      //
      // One barrier line per (XCD, request). The modulus stays NUM_KV_CHUNKS,
      // so the releaser is the last chunk to arrive *for this request* and it
      // merges *this* request -- exactly the invariant the single-request code
      // had, replicated NUM_REQS times.
      //
      // A single shared line with modulus ATTN_PARTICIPANTS would also be a
      // correct barrier, but the releaser's identity would then carry no
      // information about which request finished, so it would have to merge all
      // NUM_REQS requests serially. That puts (NUM_REQS-1) merges on the
      // critical path of every one of the 36 layers, while these per-request
      // merges run on workers that would otherwise just spin in Phase 6.
      // Drain, THEN rendezvous, THEN arrive. The order matters and was wrong
      // here: with __syncthreads() first, wave 0 can clear its own vmcnt and
      // execute tid 0's arrival while waves 1-3 have not yet retired the
      // partials they wrote -- s_waitcnt is a per-wave counter, and s_barrier
      // carries no vmcnt guarantee for the *other* waves. Draining before the
      // barrier makes the rendezvous the point at which all four waves are
      // known to have retired, which is what the arrival needs to publish.
#ifdef MPK_QKV_SUB_LDS
      unsigned int cs_t0 = 0, cs_t1 = 0;
      bool const cs_on = (qkv_epoch_expected > 1440);
      if (tid == 0) cs_t0 = (unsigned int)__builtin_amdgcn_s_memrealtime();
#endif
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      __syncthreads();
#ifdef MPK_QKV_SUB_LDS
      if (tid == 0) cs_t1 = (unsigned int)__builtin_amdgcn_s_memrealtime();
#endif
      MPK_ATSOPS_AT(3, tid == 0);
      __shared__ int s_chunk_prev;
#ifdef MPK_ATTN_FIXED_MERGER
      // Workgroup-uniform. The chunk counter counts NUM_KV_CHUNKS arrivals per layer on each (head, request)
      // line of the flag replica, which is cleared per launch like the QKV arrival counter that the
      // MPK_QKV_EPOCH_DIRECT poll reads with the same participants x layer target.
      bool const fm_merger = MPK_FM_ON && (attn_chunk == NUM_KV_CHUNKS - 1);
      if (MPK_FM_ON) {
      if (tid == 0) {
        int *const fm_rep = g_aid_flag_rep[xcd_id >> 2];
#if !(MPK_ATTN_FM_V & 1)
        if (fm_rep == nullptr) {
          __builtin_trap();
        }
#endif
        int *const fm_line = fm_rep + MPK_AID_CHUNKBAR_BASE_INTS + (attn_kv_head * NUM_REQS + attn_req) * 16;
#if MPK_ATTN_FM_V & 4
        (void)__hip_atomic_fetch_add(fm_line, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
        asm volatile("global_atomic_add %0, %1, off" ::"v"(fm_line), "v"(1) : "memory");
#endif
        if (fm_merger) {
          int const fm_tgt = NUM_KV_CHUNKS * qkv_epoch_expected;
#if MPK_ATTN_FM_V & 2
          while (__hip_atomic_load(fm_line, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < fm_tgt) {
#else
          while (__atomic_load_n(fm_line, __ATOMIC_RELAXED) < fm_tgt) {
#endif
            __builtin_amdgcn_s_sleep(1);
          }
        }
        s_chunk_prev = fm_merger ? NUM_KV_CHUNKS - 1 : 0;
      }
      if (fm_merger) {
        __syncthreads();
      }
#ifdef MPK_ATTN_FIXED_MERGER_RT
      } else {
#else
      }
#endif
#endif
#if !defined(MPK_ATTN_FIXED_MERGER) || defined(MPK_ATTN_FIXED_MERGER_RT)
      if (tid == 0) {
        // XCD-private counter: the line is indexed by this worker's own
        // `xcd_id`, and its only reader is the `% NUM_KV_CHUNKS` test right
        // below on the same XCD. See MPK_XCD_LOCAL_BARRIER at the top.
#if defined(MPK_AID_CHUNK_BAR) && defined(MPK_AID_SPLIT_FLAGS)
        int *const chunk_rep = g_aid_flag_rep[xcd_id >> 2];
        s_chunk_prev = MPK_XCD_LOCAL_ATOM_ADD(
            chunk_rep ? chunk_rep + MPK_AID_CHUNKBAR_BASE_INTS +
                            (attn_kv_head * NUM_REQS + attn_req) * 16
                      : &chunk_barrier[(attn_kv_head * NUM_REQS + attn_req) * 16],
            1);
#else
        s_chunk_prev = MPK_XCD_LOCAL_ATOM_ADD(
            &chunk_barrier[(attn_kv_head * NUM_REQS + attn_req) * 16], 1);
#endif
      }
      __syncthreads();
#endif
#ifdef MPK_ATTN_FIXED_MERGER_RT
      }
#endif
#ifdef MPK_QKV_SUB_LDS
      if (tid == 0 && cs_on) {
        unsigned int const cs_t2 = (unsigned int)__builtin_amdgcn_s_memrealtime();
        s_qs_acc[3] += (cs_t1 - cs_t0) * 10;
        s_qs_acc[4] += (cs_t2 - cs_t1) * 10;
        s_qs_n[1]++;
      }
#endif
      MPK_ATSTAMP_L(2);
      MPK_ATVALUE_L(6, s_chunk_prev % NUM_KV_CHUNKS);
      MPK_ATSOPS_AT(4, tid == 0);
#if defined(MPK_ATS_OPS) && defined(MPK_ATTN_META_EARLY)
      MPK_ATSOPS_PUT(15, tid == 0,
                     (unsigned long long)(attn_req | (attn_chunk << 4) |
                                          (((s_chunk_prev % NUM_KV_CHUNKS) == NUM_KV_CHUNKS - 1) ? 256 : 0) |
                                          ((am_q1 > am_q0) ? 512 : 0) |
                                          (attn_live_chunks << 12) | (SLIDING_WINDOW > 0 ? (1 << 20) : 0)));
#endif

#ifdef MPK_ATTN_FIXED_MERGER
      if (MPK_FM_ON ? fm_merger : ((s_chunk_prev % NUM_KV_CHUNKS) == NUM_KV_CHUNKS - 1)) {
#else
      if ((s_chunk_prev % NUM_KV_CHUNKS) == NUM_KV_CHUNKS - 1) {
#endif
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
        MPK_ATSOPS_AT(5, tid == 0);
        MPK_WS_PHASE(50, qkv_epoch_expected, xcd_id);
        // WRITE_THROUGH=true: merge writes bf16 output directly via st_wt,
        // eliminating the separate __syncthreads + readback + flush pass.
#if (!defined(MPK_ONLY_OP) || (MPK_ONLY_OP & (1 << 5))) && !defined(MPK_ATTN_MERGE_ORACLE)
#ifdef MPK_KV_CHUNKS_ADAPTIVE
#if defined(MPK_MERGE_PTR_EARLY)
#if !defined(MPK_MERGE_META_PRE) || !defined(MPK_ATTN_PTR_EARLY)
#error "MPK_MERGE_PTR_EARLY needs MPK_MERGE_META_PRE and MPK_ATTN_PTR_EARLY"
#endif
#define MPK_MERGE_BUCKET(LOOP)                                                 \
  merge_splitkv_ck_fmha<__hip_bfloat16, NUM_Q_PER_KV, NUM_KV_HEADS, HEAD_DIM,   \
                        NUM_KV_CHUNKS, 128, 4096, /*WRITE_THROUGH=*/true,       \
                        (LOOP), /*META_PRE=*/true>(                             \
      reinterpret_cast<float const *>(am_p_lse),                                \
      reinterpret_cast<float const *>(am_p_o), qo_indptr, kv_indptr,            \
      kv_last_page_len, /*request_id=*/attn_req,                                \
      reinterpret_cast<__hip_bfloat16 *>(am_p_attn),                            \
      /*kv_head_idx=*/attn_kv_head, HAS_SINKS ? am_p_sink : nullptr,            \
      am_q0, am_q1, am_sink_log2)
#elif defined(MPK_MERGE_META_PRE)
#define MPK_MERGE_BUCKET(LOOP)                                                 \
  merge_splitkv_ck_fmha<__hip_bfloat16, NUM_Q_PER_KV, NUM_KV_HEADS, HEAD_DIM,   \
                        NUM_KV_CHUNKS, 128, 4096, /*WRITE_THROUGH=*/true,       \
                        (LOOP), /*META_PRE=*/true>(                             \
      reinterpret_cast<float const *>(input_ptrs[8]),                           \
      reinterpret_cast<float const *>(input_ptrs[23]), qo_indptr, kv_indptr,    \
      kv_last_page_len, /*request_id=*/attn_req,                                \
      reinterpret_cast<__hip_bfloat16 *>(output_ptrs[4]),                       \
      /*kv_head_idx=*/attn_kv_head, HAS_SINKS ? input_ptrs[6] : nullptr,        \
      am_q0, am_q1, am_sink_log2)
#else
#define MPK_MERGE_BUCKET(LOOP)                                                 \
  merge_splitkv_ck_fmha<__hip_bfloat16, NUM_Q_PER_KV, NUM_KV_HEADS, HEAD_DIM,   \
                        NUM_KV_CHUNKS, 128, 4096, /*WRITE_THROUGH=*/true,       \
                        (LOOP)>(                                                \
      reinterpret_cast<float const *>(input_ptrs[8]),                           \
      reinterpret_cast<float const *>(input_ptrs[23]), qo_indptr, kv_indptr,    \
      kv_last_page_len, /*request_id=*/attn_req,                                \
      reinterpret_cast<__hip_bfloat16 *>(output_ptrs[4]),                       \
      /*kv_head_idx=*/attn_kv_head, HAS_SINKS ? input_ptrs[6] : nullptr)
#endif
        // One uniform branch onto a compile-time loop count: a runtime bound
        // inside the merge serialized its loads.
#ifdef MPK_KV_SW_1TILE
        // Window layers: slots SW1_N.. belong to idle workers and are never
        // stamped, so the merge must stop at SW1_N.
        if (SLIDING_WINDOW > 0 && MPK_SW1_N > 8 && attn_live_chunks > 8) {
          MPK_MERGE_BUCKET(MPK_SW1_N);
        } else
#endif
        if constexpr (NUM_KV_CHUNKS > 16) {
          if (attn_live_chunks <= 8) {
            MPK_MERGE_BUCKET(8);
          } else if (attn_live_chunks <= 16) {
            MPK_MERGE_BUCKET(16);
          } else {
            MPK_MERGE_BUCKET(NUM_KV_CHUNKS);
          }
        } else if constexpr (NUM_KV_CHUNKS > 8) {
          if (attn_live_chunks <= 8) {
            MPK_MERGE_BUCKET(8);
          } else {
            MPK_MERGE_BUCKET(NUM_KV_CHUNKS);
          }
        } else {
          MPK_MERGE_BUCKET(NUM_KV_CHUNKS);
        }
#undef MPK_MERGE_BUCKET
#else
        merge_splitkv_ck_fmha<__hip_bfloat16,
                              NUM_Q_PER_KV,
                              NUM_KV_HEADS,
                              HEAD_DIM,
                              NUM_KV_CHUNKS,
                              128,
                              4096,
#ifdef MPK_MERGE_META_PRE
                              /*WRITE_THROUGH=*/true,
                              /*LOOP_CHUNKS=*/NUM_KV_CHUNKS,
                              /*META_PRE=*/true>(
#else
                              /*WRITE_THROUGH=*/true>(
#endif
            reinterpret_cast<float const *>(input_ptrs[8]),  // lse_acc
            reinterpret_cast<float const *>(input_ptrs[23]), // o_acc_f32
            qo_indptr,
            kv_indptr,
            kv_last_page_len,
            /*request_id=*/attn_req,
            reinterpret_cast<__hip_bfloat16 *>(
                output_ptrs[4]), // attn_out (bf16)
            /*kv_head_idx=*/attn_kv_head,
            HAS_SINKS ? input_ptrs[6] : nullptr // sinks applied here
#ifdef MPK_MERGE_META_PRE
            ,
            am_q0,
            am_q1,
            am_sink_log2
#endif
        );
#endif
#endif

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
        _merge_done = __builtin_amdgcn_s_memrealtime();
#endif
        MPK_ATSTAMP_L(3);
        MPK_ATSOPS_AT(6, tid == 0);
        // Wait for all write-through stores from merge to complete.
        //
        // The __syncthreads() is load-bearing and was missing. merge's st_wt
        // stores are issued by all 256 threads (its work loop is strided by
        // `group_id = MPK_TIDX_F / THREADS_PER_TOKEN`, covering every wave),
        // but only tid 0 publishes the release below. s_waitcnt vmcnt(0) is a
        // *per-wave* counter, so wave 0 draining its own stores says nothing
        // about waves 1-3 -- their attn_out writes can still be in flight when
        // attn_global is bumped and the release flags fan out.
        //
        // The consumer is Phase 7's O-proj, which reads attn_out from a
        // different block. The corruption is silent and batch-size-dependent:
        // it shows up as nondeterministic attn_proj_out confined to the
        // workers that clear the release poll earliest -- those with no Phase 1
        // QKV work, i.e. wg_idx >= total_qkv_tiles_per_xcd.
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        __syncthreads();
        MPK_ATSTAMP_L(4);
        MPK_ATSOPS_AT(7, tid == 0);
        // TESTED AND NOT ADOPTED: the lane-parallel release fan-out that pays
        // off at the Phase 9 and O-proj barriers (readfirstlane the epoch out
        // of tid 0, then `if (tid < 8) st_wt_u32`) measures 2.047 here against
        // 2.038 for this serial loop. The transform's premise -- that 239
        // workgroups are already spinning on flags this one lane has not
        // written -- does not hold at this barrier: only the workers with a
        // Phase 1 or attention role join Phase 6, they arrive spread out rather
        // than pre-parked, and this releaser is not on anyone's critical path
        // for the seven store-issues it saves. Left serial.
#ifdef MPK_ATTN_SLICE_RELEASE
        // ── Per-kv-head slice release ────────────────────────────────────
        //
        // This XCD's merge just finished, and what it finished is exactly the
        // 512-element K slice [xcd_id*512, xcd_id*512+512) of attn_out: the
        // merge is called with kv_head_idx == xcd_id, so XCD x owns q heads
        // [8x, 8x+8) and NUM_Q_PER_KV * HEAD_DIM == 8 * 64 == 512 of the
        // O-proj's 4096-wide reduction. Eight XCDs tile it exactly.
        //
        // So there is nothing for this XCD to learn from the other seven. The
        // default path below still routes every arrival through attn_global
        // and has the *last* XCD fan out all eight flags, which makes the
        // slowest merge on the machine the release time for all eight slices.
        // Here each XCD publishes its own slice the moment it is ready and the
        // consumer waits per slice (see the ATTENTION_SLICE_RELEASE arm in
        // gang_linear_mxfp4_res_bias_rmsnorm_topk_mi300.cuh, where wave w
        // polls only XCDs 2w and 2w+1 -- the two slices its K interval reads).
        //
        // The `s_waitcnt vmcnt(0)` + `__syncthreads()` above is what makes
        // this sound: all four merge waves have drained their st_wt stores
        // before any of them reaches here, so tid 0's flag cannot outrun a
        // sibling wave's attn_out writes.
        //
        // NUM_REQS == 1 only. With more requests attn_global is bumped once
        // per (XCD, request) and an XCD's slice is not complete until all of
        // its requests have merged, so a single unconditioned store here would
        // release after the first. The static_assert is at the top of the
        // MPK_ATTN_SLICE_RELEASE block in Phase 6.
        if (tid == 0)
#ifdef MPK_VT2_SLICE_REL
        // Both of this die's mergers drained their attn_out stores above; the one whose increment
        // completes the pair publishes the slice.
#ifdef MPK_ATTN_REL_LOCAL
        if ((MPK_XCD_LOCAL_ATOM_ADD(g_aid_nc_rep[xcd_id >> 2] + MPK_NC_ATTNREL_INTS + xcd_id * 32, 1) %
             NUM_REQS) == NUM_REQS - 1)
#else
        if ((atom_add_release_gpu_s32(g_aid_nc_rep[xcd_id >> 2] + MPK_NC_ATTNREL_INTS + xcd_id * 32, 1) %
             NUM_REQS) == NUM_REQS - 1)
#endif
#endif
        {
#ifdef MPK_AID_SPLIT_FLAGS
          mpk_aid_publish(attn_release_shared,
                          xcd_id,
                          (unsigned)attn_release_expected,
                          MPK_AID_REGION_ATTN_RELEASE);
#else
          st_wt_u32((void *)&attn_release[xcd_id * 16],
                    (unsigned)attn_release_expected);
#endif
#ifndef MPK_PUB_NOWAIT
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
        }
#else
        if (tid == 0) {
          // attn_global is bumped exactly ATTN_ARRIVALS times per layer: the
          // modulus above fires once per (XCD, request), and every empty-work
          // early return lives *inside* a callee, so no arrival is ever
          // skipped. The releaser is therefore the arrival whose *own* return
          // value is the last of a group of ATTN_ARRIVALS -- a pure function
          // of this atomic's result, reading nothing shared.
          //
          // This used to snapshot the counter and round up:
          //
          //   v = relaxed_load(attn_global);
          //   expected = (v / ATTN_ARRIVALS + 1) * ATTN_ARRIVALS;
          //   if (atom_add(attn_global, 1) == expected - 1) release;
          //
          // which is only correct if the snapshot lands in this layer's group.
          // Nothing enforced that. All 36 layers run inside one task with only
          // a per-block __syncthreads between them, so a worker that reaches
          // here before the previous layer's last arrival reads a `v` from the
          // *previous* group, computes `expected` one full ATTN_ARRIVALS too
          // low, and then satisfies `prev == expected - 1` with its own bump --
          // firing this layer's release when as few as one of the merges has
          // run. The consumer's Phase 6 poll clears immediately and O-proj
          // reads attn_out rows that no merge has written yet.
          //
          // That is exactly the observed defect: silent, run-to-run varying,
          // and concentrated in the high-numbered requests (whose merges finish
          // last) and in the workers that reach Phase 6 earliest (xcd_rank >=
          // total_qkv_tiles_per_xcd, i.e. the wg >= 10 block). It also explains
          // why a delay *anywhere* in Phase 6 -- before the poll or after it --
          // cleaned it up completely while no fence on either side did
          // anything: the data was never unflushed, it was simply not computed
          // yet, and the delay just gave the real merges time to land.
          //
          // The modulus removes the race at its source: no shared read, nothing
          // to order, and no dependence on the value being sampled inside the
          // right group. It is also self-healing across kernel relaunches (the
          // counter is a host allocation that is never reset) and correct at
          // every NUM_REQS -- unlike `(v | (ATTN_ARRIVALS-1)) + 1`, which only
          // rounds to a power of two and released early at NUM_REQS in
          // {3,5,6,7}.
          constexpr int ATTN_ARRIVALS = NUM_KV_HEADS * NUM_REQS;
          int prev = atom_add_release_gpu_s32(attn_global, 1);
          if ((prev % ATTN_ARRIVALS) == ATTN_ARRIVALS - 1) {
            // LAST XCD to arrive: fan out per-XCD release flags via st_wt
            // (write-through to HBM, bypasses L2 — guarantees ld_nt sees it).
            // This eliminates the cross-XCD poll of attn_global, which hangs
            // because ld_nt/relaxed loads read stale L2 cached values.
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            for (int x = 0; x < 8; x++) {
#ifdef MPK_AID_SPLIT_FLAGS
              mpk_aid_publish(attn_release_shared,
                              x,
                              (unsigned)attn_release_expected,
                              MPK_AID_REGION_ATTN_RELEASE);
#else
              st_wt_u32((void *)&attn_release[x * 16],
                        (unsigned)attn_release_expected);
#endif
            }
#ifndef MPK_PUB_NOWAIT
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
          }
        }
#endif
        MPK_ATSOPS_AT(8, tid == 0);
#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
        _fused_t0d = __builtin_amdgcn_s_memrealtime();
#endif
      }
#ifdef MPK_ATTN_SPLIT_CHUNK
    } else if (is_split_helper) {
      // Idle rank 23+c: high half of chunk c. Consumer of Q — wait for the
      // epoch without arriving (arrival count stays QKV+attn prefix).
      if (tid == 0) {
        while (!MPK_WS_SK(1) && MPK_LD_GATE_AID(&qkv_epoch[xcd_id * 16]) < qkv_epoch_expected) {
          __builtin_amdgcn_s_sleep(1);
        }
      }
      __syncthreads();
      asm volatile("buffer_inv" ::: "memory");
      {
        using bf16_t = __hip_bfloat16;
        void const *offset_k =
            reinterpret_cast<bf16_t const *>(output_ptrs[1]) +
            static_cast<size_t>(xcd_id) * HEAD_DIM;
        void const *offset_v =
            reinterpret_cast<bf16_t const *>(output_ptrs[2]) +
            static_cast<size_t>(xcd_id) * HEAD_DIM;
#if !defined(MPK_ONLY_OP) || (MPK_ONLY_OP & (1 << 3))
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
            output_ptrs[3],
            const_cast<void *>(offset_k),
            const_cast<void *>(offset_v),
            input_ptrs[23],
            input_ptrs[8],
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            /*request_id=*/0,
            /*kv_head_idx=*/xcd_id,
            split_hchunk,
            attn_scale,
            SLIDING_WINDOW,
            nullptr,
            /*split_part=*/1);
#endif
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      __syncthreads();
      if (tid == 0) {
        st_wt_u32((void *)&chunk_barrier[(xcd_id * NUM_REQS) * 16 + 1 +
                                         split_hchunk],
                  (unsigned)qkv_epoch_expected);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      }
#endif
    }
  }

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  unsigned long long _fused_t1 = __builtin_amdgcn_s_memrealtime();
#endif
  // Slot 3 = end of attention chunk compute + merge; slot 4 = the attn_release
  // wait begins. They are adjacent here because Fleet's O-proj weight DMA is
  // issued inside the Phase 6 block below, so the DMA overlaps the gate's
  // spin-wait rather than serializing ahead of it.
#ifdef MPK_XSNAP_EARLY
  MPK_XS_MARK(14);
#endif
  MPK_PHASE_MARK(_pslot_w, 3);
  MPK_PHASE_MARK(_pslot_w, 4);
  MPK_ATSTAMP_L(5);
  MPK_ILSTAMP(12, s_ilsub_ml == MPK_ILSUB_L0 + 1);

  // ══════════════════════════════════════════════════════════════════
  // Phase 6: Cross-XCD attention barrier + O-proj weight DMA
  // Issue buffer_load_lds for O-proj weights BEFORE the barrier poll
  // so DMA runs in the background during the spin-wait (~5-60us).
  // ══════════════════════════════════════════════════════════════════
  MPK_TW_SUB(60, attn_release_expected);
  MPK_WS_PHASE(60, qkv_epoch_expected, xcd_id);
  // Same quantity the QKV/attention rotation is sized from; computed once at
  // the top of the kernel so the two cannot drift.
  int oproj_topk_tiles_per_xcd = oproj_topk_ranks;

  {
    extern __shared__ char _oproj_pf_smem[];

    // Tile space is (column block, weight group) now, matching the O-proj
    // kernel's decode. The weight group is what selects the DMA source; the
    // column block only decides whether this tile has any live token at all.
    int oproj_bblk_pf = oproj_local / oproj_n_wgs_per_xcd;
    int oproj_wg_pf = oproj_local % oproj_n_wgs_per_xcd;

    if (does_oproj && oproj_bblk_pf * 16 < num_active_tokens) {
      // Only this worker's FIRST tile. Workers that own more than one tile
      // re-stage from inside the kernel's tile-stride pass, which is the same
      // helper -- the LDS layout and the weight region base have to agree
      // byte-for-byte with the MFMA that reads them, so there is one copy of
      // this code and one formula (oproj_lds_w_off) rather than two.
      oproj_stage_weight_lds<QKV_BATCH_SIZE,
                             OPROJ_OUTPUT_PER_WG,
                             OPROJ_REDUCTION_SIZE>(
          (uint8_t const *)input_ptrs[9],
          oproj_n_wgs_per_xcd,
          oproj_wg_pf,
          tid,
          _oproj_pf_smem);
    }
  }

  MPK_WS_WAIT_BEGIN(60, attn_release_expected);
#ifdef MPK_ATTN_SLICE_RELEASE
#ifdef MPK_VT2_SLICE_REL
#if !defined(MPK_AID_NC_REP)
#error "MPK_VT2_SLICE_REL keeps its per-die counter in the NC scratch (MPK_AID_NC_REP)"
#endif
  static_assert(NUM_REQS >= 2, "MPK_VT2_SLICE_REL: the last of a die's NUM_REQS mergers publishes");
#else
  static_assert(NUM_REQS == 1,
                "MPK_ATTN_SLICE_RELEASE publishes one flag per XCD from the "
                "merge, but with NUM_REQS > 1 an XCD's attn_out slice is only "
                "complete after all of its requests have merged. Use the "
                "attn_global rendezvous at more than one request.");
#endif
  static_assert(OPROJ_REDUCTION_SIZE == NUM_KV_HEADS * NUM_Q_PER_KV * HEAD_DIM,
                "the per-kv-head merge slices must tile the O-proj reduction "
                "exactly. This said 8 and read as a per-XCD count, so it still "
                "passed at MPK_NUM_XCDS=4 while heads 4..7 went uncomputed.");
  static_assert(MPK_NUM_XCDS * KV_HEADS_PER_XCD == NUM_KV_HEADS,
                "every kv head must be owned by exactly one XCD");
  // The O-proj/TopK workers do not wait here at all. They are the only
  // consumers of attn_out, and each of their waves waits for just the two
  // slices its own K interval reads, inside the Phase 7 kernel. Waiting for
  // this XCD's flag first would be strictly worse than useless: it is one
  // more slice than wave 0 needs and one fewer than wave 3 needs, so it
  // would delay the early waves without protecting the late ones.
  //
  // Everyone else still waits, unchanged. They read nothing from attn_out --
  // this is the ordering that keeps them from running ahead into Phase 8 --
  // and their own XCD's flag is the cheapest sufficient one to wait on.
#if defined(MPK_VT2_SLICE_REL) && !defined(MPK_OPROJ_SLICE2)
  // MPK_VT2_SLICE_REL: the O-proj workers wait here too at 2 tokens (the O-proj kernel's per-wave
  // slice wait is batch-1 only), for all eight dies' slices.
  if (!does_oproj || QKV_BATCH_SIZE >= 2)
#else
  if (!does_oproj)
#endif
#endif
  {
    int _obs;
    int _spins = 0;
#ifdef MPK_QKV_L2PRE
    // Idle ranks pull this die's next-layer QKV weight records into L2 while the attention runs;
    // the QKV ranks' Phase 9 DMA then reads them from L2 instead of HBM during the W2 phase.
    if (!does_oproj && mpk_warm_qkvw != nullptr) {
#ifdef MPK_OPROJ_SWAP_ATTN
      int const pre_i = oproj_do_swap ? xcd_rank : xcd_rank - oproj_topk_ranks;
#else
      int const pre_i = xcd_rank - oproj_topk_ranks;
#endif
      int const pre_n = workers_per_xcd - oproj_topk_ranks;
      for (int t = pre_i; pre_n > 0 && t < total_qkv_tiles_per_xcd; t += pre_n) {
        qkv_warm_weights_l2<QKV_BATCH_SIZE, QKV_OUTPUT_PER_WG, QKV_REDUCTION_SIZE>(
            mpk_warm_qkvw, qkv_n_wgs_per_xcd, t);
      }
    }
#endif
#ifdef MPK_NARROW_GATE_POLL
    // One thread polls for the block instead of all 256.
    //
    // Every thread reading the same release line turns one arrival into 240
    // (30 workers x 4 waves x 2 sites) coherent reads of a single cacheline
    // per layer, and at `sc0 sc1` each of those is a real trip to the
    // coherency point rather than a vL1 hit. The line is contended precisely
    // when it is about to be written, which is the worst case for it.
    //
    // Free here: the block already rendezvouses immediately below (the
    // `__syncthreads()` that publishes the Phase 6 weight DMA), so the other
    // 255 threads gain nothing from having observed the flag themselves --
    // they cannot pass that barrier until tid 0 has. The `buffer_inv` is
    // still executed by every wave, which is what the acquire needs, since
    // buffer_inv is per-wave.
    if (tid == 0)
#endif
#ifdef MPK_VT2_SLICE_REL
      while (!MPK_WS_SK(4) &&
             (_obs = (does_oproj ? ld_aid_min8_s32(&attn_release[0])
                                 : MPK_LD_GATE_AID(&attn_release[xcd_id * 16]))) <
                 attn_release_expected) {
#else
      while (!MPK_WS_SK(4) && (_obs = MPK_LD_GATE_AID(&attn_release[xcd_id * 16])) <
             attn_release_expected) {
#endif
        MPK_WS_WAIT_TICK(_obs, _spins);
        _spins++;
        __builtin_amdgcn_s_sleep(1);
      }
#ifdef MPK_NARROW_GATE_POLL
    // Order the other waves behind tid 0's observation *before* they run the
    // acquire below: a wave that invalidated early would just refill vL1 with
    // pre-release lines and the acquire would be void.
    __syncthreads();
#endif
  }
  // Cross-XCD ACQUIRE for attn_out. Plain `buffer_inv` (vL1 only), not `sc1`.
  //
  // Under MPK_ATTN_SLICE_RELEASE this one no longer acquires attn_out for the
  // O-proj workers -- they reach it without having observed any release flag,
  // so an attn_out line left in vL1 by the *previous* layer would survive the
  // invalidate and then be hit by this layer's read. Their acquire moved to
  // where their observation is, per wave, right after the slice poll in the
  // Phase 7 kernel. This site is kept for the workers that still wait above,
  // and because it is the acquire for the whole Phase 6 hand-off, not just
  // for attn_out.
  //
  // The producer is the Phase 5 merge, which writes attn_out with st_wt
  // (WRITE_THROUGH=true) -- sc0 sc1, bypassing L2 and landing in HBM. The
  // consumer is Phase 7's O-proj on a *different* XCD (the merge that produced
  // a given kv_head runs on the XCD that owns it), and MI300/MI350 L2 is not
  // coherent across XCDs. This was `sc1` on that reasoning; see the
  // layer-boundary acquire at the top of this function for why vL1-only is
  // sufficient given the Phase 9 barrier, and for the ablation that
  // established it.
  //
  // Note the ordering below is unchanged and still required: the Phase 6
  // weight DMA is drained BEFORE the invalidate, not after.
  //
  // This is distinct from the Phase 5 barrier's `buffer_inv`, which is correct
  // for a different reason: that one is intra-XCD (the chunk barrier is
  // per-XCD), so the producers share this L2 and an `sc1` there would discard
  // their partials -- it must never become `sc1`, Phase 9 or no Phase 9.
  //
  // buffer_load_lds retires on vmcnt, and it was issued above so it overlaps
  // the release poll. Draining first keeps `buffer_inv sc1` -- which drops L2,
  // what those `sc0 nt` global->LDS reads are sourcing -- from landing while
  // they are still in flight. The wait has to happen before the MFMA reads LDS
  // regardless, so ordering it ahead of the invalidate costs nothing.
  //
  // This ordering is defensive, not the fix for any observed bug. The wg >= 10
  // corruption that was originally attributed here was the fused layer's
  // barrier release value being -1 (see the layer_counter comment above and
  // persistent_kernel.cuh's per-layer dispatch loop): every barrier was a
  // no-op, so O-proj ran against attn_out that the merges had not written yet.
  // That also explains the evidence that never fit a DMA race -- a delay
  // *before* the release poll cleaned it up just as completely as one after,
  // and neither an extra invalidate nor a system-scope fence on either side
  // changed anything.
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#if MPK_NUM_XCDS == 4
  // 4 XCDs span two AIDs, and attn_out is read in full here while each XCD
  // only produced its own KV heads' columns. Half the producers are on the far
  // die, whose lines this reader's L2 holds from before the barrier; unscoped
  // buffer_inv is an architectural NOP on gfx950 and does not drop them.
  asm volatile("buffer_inv" MPK_DEV_SC_INV "" ::: "memory");
#else
  asm volatile("buffer_inv" ::: "memory");
#endif
  // Publish the Phase 6 O-proj weight DMA to the whole block.
  //
  // buffer_load_lds retires on vmcnt, and the drain above is per-wave -- it
  // covers only the loads *this* wave issued. Each wave DMAs its own
  // 1024-byte slices (j*4096 + pf_warp_id*1024), and those slices tile the
  // weight region contiguously, so the tile every wave then reads for its
  // K-range is interleaved across all four waves' slices. Wave 0's MFMA reads
  // bytes wave 3 loaded; without a rendezvous it can read them before they
  // land, and the B operand is whatever LDS held from the previous layer.
  //
  // Silent wrong numerics, no fault: the stale bytes are still a valid MXFP4
  // encoding, so the tile computes cleanly against garbage weights. It shows
  // up in attn_proj_out as columns carrying ~2000-magnitude values where the
  // correct output tops out near 33.
  __syncthreads();

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  unsigned long long _fused_t2 = __builtin_amdgcn_s_memrealtime();
#endif
#ifdef MPK_XSNAP_EARLY
  MPK_XS_MARK(15);
#endif
  MPK_PHASE_MARK(_pslot_w, 5);
  MPK_ILSTAMP(13, s_ilsub_ml == MPK_ILSUB_L0 + 1);

  // ══════════════════════════════════════════════════════════════════
  // Phase 7: O-proj + RMSNorm + Router + TopK
  // ══════════════════════════════════════════════════════════════════
  MPK_TW_SUB(70, oproj_topk_tiles_per_xcd);
  MPK_WS_PHASE(70, qkv_epoch_expected, xcd_id);
  {
    if (does_oproj) {
      int oproj_tile_idx = xcd_id * oproj_topk_tiles_per_xcd + oproj_local;
      void const *oproj_residual_ptr =
          reinterpret_cast<__hip_bfloat16 const *>(output_ptrs[0]) +
          static_cast<size_t>(xcd_id) * oproj_n_wgs_per_xcd *
              OPROJ_OUTPUT_PER_WG;
#if !defined(MPK_ONLY_OP) || (MPK_ONLY_OP & (1 << 7))
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
          // layer_epoch: the O-proj hierarchical barrier's release target.
          // Same layer-derived value as the other three counters in this file
          // -- passing it explicitly is what lets that barrier stop
          // snapshotting "current value + 1", which races with the releaser.
          qkv_epoch_expected,
          // ts_base: the O-proj/TopK kernel's optional per-sub-op timestamp
          // sink. Nothing here reads those slots -- the [FUSED_PHASE] printf
          // below derives every number from _fused_t0.._fused_t4, which are
          // taken in this function. This used to name a _ts_base that was
          // never declared, so MPK_DEVICE_TIMING=1 did not compile.
          nullptr,
#ifdef MPK_ATTN_SLICE_RELEASE
          // The kernel does its own per-wave slice wait; the Phase 6 gate
          // above skipped this worker entirely.
          attn_release
#else
          nullptr
#endif
      );
#endif
    }
  }
  MPK_TKOPS_AT(10, tid == 0);
  MPK_PHASE_MARK(_pslot_w, 6);
  MPK_ILSTAMP(14, s_ilsub_ml == MPK_ILSUB_L0 + 1);

  // ══════════════════════════════════════════════════════════════════
  // Phase 7a': O-proj barrier for the workers that skipped Phase 7
  // ══════════════════════════════════════════════════════════════════
  // `does_oproj` above is 23 workers at this model (184 O-proj weight groups /
  // 8 XCDs), while the dispatch is 31 workers/XCD. The other 8 never enter
  // the Phase 7 kernel at all -- so they never reach its O-proj hierarchical
  // barrier -- and then draw MoE tiles in Phase 8 that read rmsnorm_out_moe.
  // Default mapping is xcd_rank < 23; MPK_OPROJ_SWAP_ATTN moves the skippers
  // to ranks 0-7.
  //
  // Phase 7b below does not cover them. It gates on routing_ready, which
  // publishes active_expert_ids / routing_indices; nothing in that release
  // orders the *normed row* those workers are about to consume, and the
  // `buffer_inv` after it invalidates vL1 only, not the per-XCD L2 the plain
  // global_store under MPK_ONE_NORM_WRITER lands in.
  //
  // Measured, 2 layers at bs=2, against an offline MXFP4+SwiGLU oracle with
  // the error normalized by the per-slot scale: the corrupt W13 weight groups
  // are exactly the tiles owned by xcd_rank >= 23, in three separate
  // geometries --
  //
  //   workers/XCD 31 -> skippers 23..30 -> corrupt ranks 26..30 -> 33/360 bad
  //   workers/XCD 26 -> skippers 23..25 -> corrupt ranks 23..25 -> 14/360 bad
  //   workers/XCD 23 -> skippers none   -> corrupt ranks none   ->  0/360 bad
  //
  // Ranks 16..22 also skip the RMSNorm (`local_tile >= router_tile_n`, 16) and
  // are clean in every config, so the discriminator is entering this barrier,
  // not running the norm.
  //
  // bs=1 is clean because the Phase 7 kernel's per-token loop is a
  // compile-time 1 there and its closing __syncthreads compiles out, so there
  // is no partially written multi-token row to catch. The corrupt reads are
  // uncorrelated noise that scores *worse* than all-zeros, which is what a
  // half-filled row looks like -- not a coherent wrong input.
  //
  // Arrival is unchanged: these workers have no O-proj columns to publish, so
  // they must not bump the counter (`tiles_per_xcd` arrivals per XCD is what
  // the release fires on, and an extra arrival would desynchronize the
  // modular release). They only wait.
  //
  // ── This is a PARTIAL fix; the rest of the race is still open ─────────
  //
  // Matched control in one build, gate vs MPK_NO_OPROJ_SKIP_GATE=1: 32 bad
  // weight groups against 52, with XCDs 4 and 5 going 12+8 -> 0. Real, and
  // reproducible. But XCDs 1, 6 and 7 are untouched, and the residual sits in
  // expert slots 4 and 5 -- the last two *real* slots of the 8-slot
  // MAX_ACTIVATED space at bs=2 (6 real, 2 padding).
  //
  // The reason this cannot be the whole fix: the barrier polled here releases
  // *before* the RMSNorm that writes rmsnorm_out_moe, which runs afterwards in
  // the Phase 7 kernel's own Phase 3. So it orders the O-proj columns, not the
  // normed row. What makes the residual genuinely puzzling is that Phase 7b
  // below already polls routing_ready, which the router workers publish only
  // after draining their norm stores -- if that release were sufficient there
  // would be no residual at all, so something about it does not order the norm
  // for these consumers.
  //
  // rmsnorm_out_moe is *correct* in the final dump (max err 0.0039 against a
  // 0.35 scale, zero bad elements) while swiglu_out is wrong, so the MoE reads
  // it transiently before it settles. The buffer is fine; the ordering is not.
  //
  // Counts move run to run (33/22/32/32 across reps at the shipped geometry),
  // so do not size a change here off a single run -- always rebuild with the
  // ablation flag and compare inside one build.
#if !defined(MPK_NO_OPROJ_SKIP_GATE) && !defined(MPK_ROUTER_OPROJ_PARTIAL)
  if (!does_oproj) {
#if defined(MPK_QKV_L2PF)
#if !defined(MPK_MOE_XCD_PAIR) || !defined(MPK_PREFETCH_NEXT_QKV) ||        \
    MPK_NUM_XCDS != 8 || defined(MPK_QKV_KSPLIT)
#error "MPK_QKV_L2PF: 8 XCDs, MPK_MOE_XCD_PAIR, MPK_PREFETCH_NEXT_QKV"
#endif
    // No MoE tile on this rank: warm this XCD's L2 with the next
    // layer's QKV weights while HBM is quiet, so the QKV workers'
    // Phase 9 prefetch of the same lines is served from L2.
    {
      constexpr int kL2pfFirst = 23;
      if (xcd_rank >= kL2pfFirst && input_ptrs[24] != nullptr) {
        for (int t = xcd_rank - kL2pfFirst; t < qkv_n_wgs_per_xcd;
             t += workers_per_xcd - kL2pfFirst) {
          qkv_prefetch_weights_lds<QKV_TB,
                                   QKV_OUTPUT_PER_WG,
                                   QKV_REDUCTION_SIZE>(
              input_ptrs[24], qkv_n_wgs_per_xcd, t);
        }
      }
    }
#endif
#if defined(MPK_OPROJ_TILE_FLAGS) || defined(MPK_OPROJ_POLL_GLOBAL)
#ifdef MPK_AID_SPLIT_FLAGS
#error "the O-proj skip-gate TILE_FLAGS / POLL_GLOBAL variants poll the shared counter block, not the AID replicas"
#endif
    int *oproj_hier = static_cast<int *>(input_ptrs[16]);
#ifdef MPK_OPROJ_TILE_FLAGS
    if (tid < 64) {
      mpk_oproj_tile_flag_poll(&oproj_hier[MPK_OPROJ_TILE_FLAG_SLOT],
                               total_oproj_tiles, qkv_epoch_expected);
    }
    __syncthreads();
#else
    if (tid == 0) {
      while (MPK_LD_GATE2(&oproj_hier[8 * 16]) < 8 * qkv_epoch_expected) {
        __builtin_amdgcn_s_sleep(1);
      }
    }
    __syncthreads();
#endif
#else
    int *oproj_hier_shared = static_cast<int *>(input_ptrs[16]);
#ifdef MPK_AID_SPLIT_FLAGS
    // Second consumer of the O-proj release flags, in a different file from
    // the kernel that publishes them: these are the ranks that never enter
    // Phase 7, so they wait on that barrier here instead. It has to follow
    // the publisher onto the replicas or it polls a line nobody writes.
    int *oproj_hier = mpk_aid_flags(
        oproj_hier_shared, xcd_id, MPK_AID_REGION_HIER_RELEASE);
#else
    int *oproj_hier = oproj_hier_shared;
#endif
#ifdef MPK_NARROW_HIER_POLL
    // Same contention argument as MPK_NARROW_GATE_POLL, and this site is worse:
    // every rank that skips Phase 7 waits here, so ~168 blocks x 4 waves poll a
    // single line. Measured with a release-visibility microbenchmark: one shared release line goes
    // 5.0 us at 64 pollers -> 26.6 at 128 -> 40.1 at 192, while the same fan-out
    // on AID-local RW replicas stays flat at 2.5 us. The comment below ("all
    // threads poll independently -- eliminates __syncthreads overhead") predates
    // that measurement; the __syncthreads is far cheaper than the extra pollers.
    //
    // Sound with one poller: `does_oproj` is a per-worker constant so this
    // branch is block-uniform, and this gate carries no acquire -- it only
    // orders these ranks against Phase 8 -- so tid 0 observing plus a
    // __syncthreads gives every thread the same ordering it had before.
    //
    // MEASURED NEUTRAL, left off. On top of MPK_NARROW_GATE_POLL=1 this is
    // 2.751 / 2.738 against 2.696 / 2.725 without it -- no gain. So poller
    // count alone is not what costs: the ranks waiting here are already idle,
    // so cutting their polling traffic buys nothing and the __syncthreads
    // costs about what the polls saved. MPK_NARROW_GATE_POLL wins for a
    // different reason -- that line is contended exactly when it is about to
    // be written, and its waiters are about to do work. Do not expect the
    // remaining all-threads polls (gang_moe_fused d_barrier_rel and
    // final_release, gang_linear oproj_xcd_ready and the slice rel/rel0/rel1)
    // to pay just because they have many pollers.
    if (tid == 0)
#endif
#ifdef MPK_OPROJ_NARROW_REL
    // Third consumer of MPK_AID_REGION_HIER_RELEASE, in a different file from
    // the publisher. Must follow the narrowed release to slot 0.
#ifdef MPK_OPROJ_AID_TIER
    while (ld_aid_min2_s32(&oproj_hier[16], &oproj_hier[32]) <
           qkv_epoch_expected) {
#elif defined(MPK_OPROJ_FLAT) && defined(MPK_OPROJ_LOCAL_BAR_PROBE)
    while (MPK_LD_GATE_AID(&oproj_hier[(xcd_id >> 2) * 64]) <
           qkv_epoch_expected) {
#else
    while (MPK_LD_GATE_AID(&oproj_hier[0]) < qkv_epoch_expected) {
#endif
#else
    while (MPK_LD_GATE_AID(&oproj_hier[xcd_id * 16]) < qkv_epoch_expected) {
#endif
#ifndef MPK_OPROJ_SKIP_GATE_BUSY_POLL
      __builtin_amdgcn_s_sleep(1);
#endif
    }
#ifdef MPK_NARROW_HIER_POLL
    __syncthreads();
#endif
#endif
  }
#endif

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
#if defined(MPK_LOCAL_TOPK)
    // No routing_ready poll: rebuild this layer's routing from the 128
    // epoch-tagged router logits, then take this XCD pair's pick -- the
    // same selection-order entry the completer's u64 record carries.
#ifdef MPK_AID_SPLIT_FLAGS
    // Tags and this XCD's normed-row flags live in its own AID's replica.
    int const *const ltk_rep =
        mpk_aid_flags_at(routing_ready, xcd_id, MPK_AID_ROUTING_BASE_INTS);
    int const *const ltk_tags = ltk_rep + MPK_AID_LTK_OFF_INTS;
#ifdef MPK_VT2
    int const *const ltk_tags1 = ltk_tags + 512;
#endif
#ifdef MPK_LTK_EARLY_TAG
    int const *const ltk_norm = ltk_rep + MPK_AID_LTK_NORM_OFF_INTS + xcd_id * 32;
#endif
#else
    int const *const ltk_tags = oproj_counters_base + MPK_LTK_TAG_SLOT;
#ifdef MPK_LTK_EARLY_TAG
    int const *const ltk_norm =
        oproj_counters_base + MPK_LTK_NORM_SLOT + xcd_id * 32;
#endif
#endif
#if defined(MPK_ROUTER_OPROJ_PARTIAL)
#if defined(MPK_VT2) && defined(MPK_VTN_MOE)
    if constexpr (QKV_BATCH_SIZE > 2) {
      // MPK_VTN_N7B: Phase 7b for an N-token pass, one token per wave. Token t's die sums at
      // mpk_rop_agg_ints(t); wave t computes token t's logits, selects its top 4 (the 2-token key selection
      // and exact renormalization) and quantizes its row into LDS row t. Per-token arithmetic as at 2 tokens.
      // MPK_VTN_T6: above 4 tokens wave w also takes tokens w + 4, ... (the gather pairs tokens: even counts).
      static_assert(QKV_BATCH_SIZE >= 4 && QKV_BATCH_SIZE % 2 == 0 && QKV_BATCH_SIZE <= MPK_ROP_NTOK &&
                        MPK_VTN_T == QKV_BATCH_SIZE && NUM_EXPERTS == 128 && MPK_AID_ROP_AGG_INTS == 288,
                    "MPK_VTN_N7B: an even token count, wave w takes tokens w, w + 4, 128 experts");
#if !defined(MPK_VT2_QUANT_FAST) || !defined(MPK_VT2_KSEL2) || !defined(MPK_VT2_KSEL2_TAIL) || \
    MPK_VT2_KSEL2_TAIL != 1
#error "MPK_VTN_N7B mirrors the MPK_VT2_QUANT_FAST + MPK_VT2_KSEL2 (TAIL=1) path"
#endif
      typedef int __attribute__((ext_vector_type(4))) rop7_i32x4_t;
      extern __shared__ char _rop_smem[];
      float *const nsum = (float *)(_rop_smem + MPK_VTN_SCR + 2048);
      unsigned short *const nlg =
          (unsigned short *)(_rop_smem + MPK_VTN_SCR + 2048 + QKV_BATCH_SIZE * 132 * 4);
      static_assert(MPK_VTN_SCR + 2048 + QKV_BATCH_SIZE * (132 * 4 + 128 * 2) <=
                        mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE -
                            mirage::runtime::LAYER_IDX_SMEM_OFFSET_FROM_END,
                    "MPK_VTN_N7B: scratch above the MoE staging");
      unsigned const rop_tag = (unsigned)routing_expected;
      int const rop_units = oproj_output_stride / 16;
      int const npar = (routing_expected & 1) * MPK_NUM_XCDS * MPK_AID_ROP_AGG_INTS;
      int const *const nrep = g_rop_rep[xcd_id >> 2];
      int const nw = tid >> 6;
      int const ln = tid & 63;
      // Layer constants first: this lane's two experts' biases and its three units' gammas.
      float nb0 = 0.0f, nb1 = 0.0f;
      if (input_ptrs[14] != nullptr) {
        unsigned short const *rb = (unsigned short const *)input_ptrs[14] -
                                   xcd_id * (NUM_EXPERTS / MPK_NUM_XCDS);
        nb0 = __uint_as_float((unsigned)rb[ln] << 16);
        nb1 = __uint_as_float((unsigned)rb[ln + 64] << 16);
      }
      unsigned short const *const rop_gb = (unsigned short const *)input_ptrs[11];
      int const nua = ln, nub = ln + 64, nuc = ln + 128;
      bool const nvc = nuc < rop_units;
      rop7_i32x4_t g0, g1, g2, g3, g4 = {}, g5 = {};
      {
        rop7_i32x4_t const *gp = (rop7_i32x4_t const *)(rop_gb + nua * 16);
        g0 = gp[0];
        g1 = gp[1];
        gp = (rop7_i32x4_t const *)(rop_gb + nub * 16);
        g2 = gp[0];
        g3 = gp[1];
        if (nvc) {
          gp = (rop7_i32x4_t const *)(rop_gb + nuc * 16);
          g4 = gp[0];
          g5 = gp[1];
        }
      }
#ifdef MPK_VTN_N7B2
      bool const n7b2 = g_mpk_n7b2_rt != 0;
#else
      constexpr bool n7b2 = false;
#endif
      if (!n7b2) {
      // Entry p < 129 of every token's eight die sums (128 expert partials + the sum of squares); the tags
      // poll as in the 2-token gather, two tokens per load batch.
      if (tid < 129) {
#pragma unroll
        for (int tp = 0; tp < QKV_BATCH_SIZE / 2; ++tp) {
          char const *const a0 = (char const *)(nrep + mpk_rop_agg_ints(2 * tp) + npar + 2 * tid);
          char const *const a1 = a0 + 4 * MPK_AID_ROP_AGG_INTS * 4;
          char const *const b0 =
              (char const *)(nrep + mpk_rop_agg_ints(2 * tp + 1) + npar + 2 * tid);
          char const *const b1 = b0 + 4 * MPK_AID_ROP_AGG_INTS * 4;
          unsigned long long e0, e1, e2, e3, e4, e5, e6, e7;
          unsigned long long f0, f1, f2, f3, f4, f5, f6, f7;
          while (true) {
            asm volatile("global_load_dwordx2 %0, %16, off sc0 sc1\n"
                         "global_load_dwordx2 %1, %16, off offset:1152 sc0 sc1\n"
                         "global_load_dwordx2 %2, %16, off offset:2304 sc0 sc1\n"
                         "global_load_dwordx2 %3, %16, off offset:3456 sc0 sc1\n"
                         "global_load_dwordx2 %4, %17, off sc0 sc1\n"
                         "global_load_dwordx2 %5, %17, off offset:1152 sc0 sc1\n"
                         "global_load_dwordx2 %6, %17, off offset:2304 sc0 sc1\n"
                         "global_load_dwordx2 %7, %17, off offset:3456 sc0 sc1\n"
                         "global_load_dwordx2 %8, %18, off sc0 sc1\n"
                         "global_load_dwordx2 %9, %18, off offset:1152 sc0 sc1\n"
                         "global_load_dwordx2 %10, %18, off offset:2304 sc0 sc1\n"
                         "global_load_dwordx2 %11, %18, off offset:3456 sc0 sc1\n"
                         "global_load_dwordx2 %12, %19, off sc0 sc1\n"
                         "global_load_dwordx2 %13, %19, off offset:1152 sc0 sc1\n"
                         "global_load_dwordx2 %14, %19, off offset:2304 sc0 sc1\n"
                         "global_load_dwordx2 %15, %19, off offset:3456 sc0 sc1\n"
                         "s_waitcnt vmcnt(0)"
                         : "=&v"(e0), "=&v"(e1), "=&v"(e2), "=&v"(e3), "=&v"(e4), "=&v"(e5),
                           "=&v"(e6), "=&v"(e7), "=&v"(f0), "=&v"(f1), "=&v"(f2), "=&v"(f3),
                           "=&v"(f4), "=&v"(f5), "=&v"(f6), "=&v"(f7)
                         : "v"(a0), "v"(a1), "v"(b0), "v"(b1)
                         : "memory");
            unsigned const tg = rop_tag;
            if ((unsigned)(e0 >> 32) == tg && (unsigned)(e1 >> 32) == tg &&
                (unsigned)(e2 >> 32) == tg && (unsigned)(e3 >> 32) == tg &&
                (unsigned)(e4 >> 32) == tg && (unsigned)(e5 >> 32) == tg &&
                (unsigned)(e6 >> 32) == tg && (unsigned)(e7 >> 32) == tg &&
                (unsigned)(f0 >> 32) == tg && (unsigned)(f1 >> 32) == tg &&
                (unsigned)(f2 >> 32) == tg && (unsigned)(f3 >> 32) == tg &&
                (unsigned)(f4 >> 32) == tg && (unsigned)(f5 >> 32) == tg &&
                (unsigned)(f6 >> 32) == tg && (unsigned)(f7 >> 32) == tg) {
              break;
            }
            __builtin_amdgcn_s_sleep(1);
          }
          nsum[(2 * tp) * 132 + tid] = __uint_as_float((unsigned)e0) + __uint_as_float((unsigned)e1) +
                                       __uint_as_float((unsigned)e2) + __uint_as_float((unsigned)e3) +
                                       __uint_as_float((unsigned)e4) + __uint_as_float((unsigned)e5) +
                                       __uint_as_float((unsigned)e6) + __uint_as_float((unsigned)e7);
          nsum[(2 * tp + 1) * 132 + tid] =
              __uint_as_float((unsigned)f0) + __uint_as_float((unsigned)f1) +
              __uint_as_float((unsigned)f2) + __uint_as_float((unsigned)f3) +
              __uint_as_float((unsigned)f4) + __uint_as_float((unsigned)f5) +
              __uint_as_float((unsigned)f6) + __uint_as_float((unsigned)f7);
        }
      }
      __syncthreads();
      }
      // MPK_VTN_T6: wave nw takes tokens nw, nw + 4 (one at 4 tokens); nothing per token is block-wide, so the
      // waves' token counts may differ. n7_load: the token's eight-die sums (tags seen) and its O-proj output row;
      // n7_compute: logits, top 4 and the quantized row. MPK_VTN_N7B_PF: a two-token wave loads its second token
      // before the first token's work -- measured a LOSS (T=6 +1.3% A/B/A/B/A, T=8 +1.0%): it waits for the
      // later-published token's tags in front of the first token's work. Opt-in only.
      constexpr int n7_rounds = (QKV_BATCH_SIZE + 3) / 4;
      static_assert(n7_rounds <= 2, "MPK_VTN_N7B: at most two tokens per wave");
      // This wave's rows of the O-proj output (published before the tags it sees), system scope.
      char const *const rop_hb = (char const *)mpk_aid_out_base(
          output_ptrs[5], xcd_id, QKV_BATCH_SIZE * oproj_output_stride * 2);
      // Global (not FLAT) system-scope loads the compiler counts: volatile is sc0 sc1 on gfx950.
      typedef __attribute__((address_space(1))) rop7_i32x4_t const volatile n7_gv;
#ifdef MPK_N7_TS2
      unsigned int n7s2[5] = {0u, 0u, 0u, 0u, 0u};  // MPK_N7_TS2 (timing tool)
#define MPK_N7S2(k) do { if (nt == nw) { unsigned long long _t; asm volatile("s_memrealtime %0\n\ts_waitcnt lgkmcnt(0)" : "=s"(_t) :: "memory"); n7s2[(k)] = (unsigned int)_t; } } while (0)
#else
#define MPK_N7S2(k) do { } while (0)
#endif
#ifdef MPK_VTN_N7S2
#ifndef MPK_VTN_LIVEB
#error "MPK_VTN_N7S2 needs MPK_VTN_LIVEB"
#endif
#define MPK_N7S2_LP , bool n7drow
#define MPK_N7S2_CP , bool n7dsel, bool n7dqnt
#define MPK_N7S2_LA(r) , r
#define MPK_N7S2_CA(a, b) , a, b
#else
#define MPK_N7S2_LP
#define MPK_N7S2_CP
#define MPK_N7S2_LA(r)
#define MPK_N7S2_CA(a, b)
#endif
      auto n7_load = [&](int nt, float &ns0, float &ns1, float &nsq, rop7_i32x4_t &h0, rop7_i32x4_t &h1,
                         rop7_i32x4_t &h2, rop7_i32x4_t &h3, rop7_i32x4_t &h4, rop7_i32x4_t &h5 MPK_N7S2_LP) {
#ifdef MPK_VTN_N7B2
        if (n7b2) {
          // MPK_VTN_N7B2: wave nt gathers its own token: entries ln, ln + 64 and 128 of the eight dies in one tagged
          // load batch, then logits / selection / quantization without a block barrier.
          int const *const nag = nrep + mpk_rop_agg_ints(nt) + npar;
          char const *const ga0 = (char const *)(nag + 2 * ln);
          char const *const ga1 = ga0 + 4 * MPK_AID_ROP_AGG_INTS * 4;
          char const *const gc0 = (char const *)(nag + 2 * 128);
          char const *const gc1 = gc0 + 4 * MPK_AID_ROP_AGG_INTS * 4;
          unsigned long long e0, e1, e2, e3, e4, e5, e6, e7, f0, f1, f2, f3, f4, f5, f6, f7;
          unsigned long long c0, c1, c2, c3, c4, c5, c6, c7;
#ifdef MPK_VTN_N7P1
          {
#ifdef MPK_VTN_N7P1_ABIP
            bool const n7p1_on = mpk_abip_odd
#ifdef MPK_VTN_N7P1_R1
                                 && nt < 4  // MPK_VTN_N7P1_R1: first round only
#endif
                ;
#elif defined(MPK_VTN_N7P1_R1)
            bool const n7p1_on = nt < 4;  // MPK_VTN_N7P1_R1: first round only
#else
            constexpr bool n7p1_on = true;
#endif
#ifdef MPK_N7P1_MINN
            bool const n7p1_go = n7p1_on && num_active_tokens >= (MPK_N7P1_MINN);
#else
            bool const n7p1_go = n7p1_on;
#endif
#ifndef MPK_N7P1_K
#define MPK_N7P1_K 4
#endif
            // MPK_VTN_N7P1: wait on the eight dies' ssq entries (8 lines per attempt), then gather.
            if (n7p1_go) {
              while (true) {
                unsigned long long q0, q1, q2, q3, q4, q5, q6, q7;
                asm volatile("global_load_dwordx2 %0, %8, off sc0 sc1\n"
                             "global_load_dwordx2 %1, %8, off offset:1152 sc0 sc1\n"
                             "global_load_dwordx2 %2, %8, off offset:2304 sc0 sc1\n"
                             "global_load_dwordx2 %3, %8, off offset:3456 sc0 sc1\n"
                             "global_load_dwordx2 %4, %9, off sc0 sc1\n"
                             "global_load_dwordx2 %5, %9, off offset:1152 sc0 sc1\n"
                             "global_load_dwordx2 %6, %9, off offset:2304 sc0 sc1\n"
                             "global_load_dwordx2 %7, %9, off offset:3456 sc0 sc1\n"
                             "s_waitcnt vmcnt(0)"
                             : "=&v"(q0), "=&v"(q1), "=&v"(q2), "=&v"(q3), "=&v"(q4), "=&v"(q5), "=&v"(q6), "=&v"(q7)
                             : "v"(gc0), "v"(gc1)
                             : "memory");
                unsigned const tq = rop_tag;
                int const nrd = ((unsigned)(q0 >> 32) == tq) + ((unsigned)(q1 >> 32) == tq) +
                                ((unsigned)(q2 >> 32) == tq) + ((unsigned)(q3 >> 32) == tq) +
                                ((unsigned)(q4 >> 32) == tq) + ((unsigned)(q5 >> 32) == tq) +
                                ((unsigned)(q6 >> 32) == tq) + ((unsigned)(q7 >> 32) == tq);
                if (nrd >= MPK_N7P1_K) break;
#ifndef MPK_N7_SLEEP
#define MPK_N7_SLEEP 1
#endif
                __builtin_amdgcn_s_sleep(MPK_N7_SLEEP);
              }
            }
          }
#endif
#if defined(MPK_TOPK_SPINREC) && defined(MPK_RESID_MAP)  // MPK_TOPK_SPINREC_V1
          unsigned n7_spins = 0u;
#endif
          while (true) {
            asm volatile("global_load_dwordx2 %0, %24, off sc0 sc1\n"
                           "global_load_dwordx2 %1, %24, off offset:1152 sc0 sc1\n"
                           "global_load_dwordx2 %2, %24, off offset:2304 sc0 sc1\n"
                           "global_load_dwordx2 %3, %24, off offset:3456 sc0 sc1\n"
                           "global_load_dwordx2 %4, %25, off sc0 sc1\n"
                           "global_load_dwordx2 %5, %25, off offset:1152 sc0 sc1\n"
                           "global_load_dwordx2 %6, %25, off offset:2304 sc0 sc1\n"
                           "global_load_dwordx2 %7, %25, off offset:3456 sc0 sc1\n"
                           "global_load_dwordx2 %8, %24, off offset:512 sc0 sc1\n"
                           "global_load_dwordx2 %9, %24, off offset:1664 sc0 sc1\n"
                           "global_load_dwordx2 %10, %24, off offset:2816 sc0 sc1\n"
                           "global_load_dwordx2 %11, %24, off offset:3968 sc0 sc1\n"
                           "global_load_dwordx2 %12, %25, off offset:512 sc0 sc1\n"
                           "global_load_dwordx2 %13, %25, off offset:1664 sc0 sc1\n"
                           "global_load_dwordx2 %14, %25, off offset:2816 sc0 sc1\n"
                           "global_load_dwordx2 %15, %25, off offset:3968 sc0 sc1\n"
                           "global_load_dwordx2 %16, %26, off sc0 sc1\n"
                           "global_load_dwordx2 %17, %26, off offset:1152 sc0 sc1\n"
                           "global_load_dwordx2 %18, %26, off offset:2304 sc0 sc1\n"
                           "global_load_dwordx2 %19, %26, off offset:3456 sc0 sc1\n"
                           "global_load_dwordx2 %20, %27, off sc0 sc1\n"
                           "global_load_dwordx2 %21, %27, off offset:1152 sc0 sc1\n"
                           "global_load_dwordx2 %22, %27, off offset:2304 sc0 sc1\n"
                           "global_load_dwordx2 %23, %27, off offset:3456 sc0 sc1\n"
                           "s_waitcnt vmcnt(0)"
                           : "=&v"(e0), "=&v"(e1), "=&v"(e2), "=&v"(e3), "=&v"(e4), "=&v"(e5), "=&v"(e6), "=&v"(e7), "=&v"(f0), "=&v"(f1), "=&v"(f2), "=&v"(f3), "=&v"(f4), "=&v"(f5), "=&v"(f6), "=&v"(f7), "=&v"(c0), "=&v"(c1), "=&v"(c2), "=&v"(c3), "=&v"(c4), "=&v"(c5), "=&v"(c6), "=&v"(c7)
                           : "v"(ga0), "v"(ga1), "v"(gc0), "v"(gc1)
                           : "memory");
            unsigned const tg = rop_tag;
            if ((unsigned)(e0 >> 32) == tg &&
                   (unsigned)(e1 >> 32) == tg &&
                   (unsigned)(e2 >> 32) == tg &&
                   (unsigned)(e3 >> 32) == tg &&
                   (unsigned)(e4 >> 32) == tg &&
                   (unsigned)(e5 >> 32) == tg &&
                   (unsigned)(e6 >> 32) == tg &&
                   (unsigned)(e7 >> 32) == tg &&
                   (unsigned)(f0 >> 32) == tg &&
                   (unsigned)(f1 >> 32) == tg &&
                   (unsigned)(f2 >> 32) == tg &&
                   (unsigned)(f3 >> 32) == tg &&
                   (unsigned)(f4 >> 32) == tg &&
                   (unsigned)(f5 >> 32) == tg &&
                   (unsigned)(f6 >> 32) == tg &&
                   (unsigned)(f7 >> 32) == tg &&
                   (unsigned)(c0 >> 32) == tg &&
                   (unsigned)(c1 >> 32) == tg &&
                   (unsigned)(c2 >> 32) == tg &&
                   (unsigned)(c3 >> 32) == tg &&
                   (unsigned)(c4 >> 32) == tg &&
                   (unsigned)(c5 >> 32) == tg &&
                   (unsigned)(c6 >> 32) == tg &&
                   (unsigned)(c7 >> 32) == tg) {
              break;
            }
#if defined(MPK_TOPK_SPINREC) && defined(MPK_RESID_MAP)
            if (((++n7_spins) & 1023u) == 0u && (threadIdx.x & 63u) == 0u) {
              unsigned long long const w24[24] = {e0, e1, e2, e3, e4, e5, e6, e7, f0, f1, f2, f3,
                                                  f4, f5, f6, f7, c0, c1, c2, c3, c4, c5, c6, c7};
              unsigned si = 31u, age = 0u;
#pragma unroll
              for (int i = 23; i >= 0; --i) {
                unsigned const t = (unsigned)(w24[i] >> 32);
                if (t != tg) {
                  si = (unsigned)i;
                  age = tg - t;
                }
              }
              unsigned const k = n7_spins >> 10;
#ifdef MPK_MULTI_DEV
              int const tsw = g_mpk_dev_worker_base + (int)blockIdx.x;
#else
              int const tsw = (int)blockIdx.x;
#endif
              mpk_resid_put(768 + tsw, ((k < 0x7fffu ? k : 0x7fffu) << 16) | (si << 11) | (age < 2047u ? age : 2047u));
            }
#endif
#ifndef MPK_N7_SLEEP
#define MPK_N7_SLEEP 1
#endif
#if defined(MPK_ABIP) && defined(MPK_N7_SLEEP_AB)
            if (mpk_abip_odd) {
              __builtin_amdgcn_s_sleep(MPK_N7_SLEEP_AB);
            } else {
              __builtin_amdgcn_s_sleep(MPK_N7_SLEEP);
            }
#else
            __builtin_amdgcn_s_sleep(MPK_N7_SLEEP);
#endif
          }
          ns0 = __uint_as_float((unsigned)e0) + __uint_as_float((unsigned)e1) + __uint_as_float((unsigned)e2) + __uint_as_float((unsigned)e3) + __uint_as_float((unsigned)e4) + __uint_as_float((unsigned)e5) + __uint_as_float((unsigned)e6) + __uint_as_float((unsigned)e7);
          ns1 = __uint_as_float((unsigned)f0) + __uint_as_float((unsigned)f1) + __uint_as_float((unsigned)f2) + __uint_as_float((unsigned)f3) + __uint_as_float((unsigned)f4) + __uint_as_float((unsigned)f5) + __uint_as_float((unsigned)f6) + __uint_as_float((unsigned)f7);
          nsq = __uint_as_float((unsigned)c0) + __uint_as_float((unsigned)c1) + __uint_as_float((unsigned)c2) + __uint_as_float((unsigned)c3) + __uint_as_float((unsigned)c4) + __uint_as_float((unsigned)c5) + __uint_as_float((unsigned)c6) + __uint_as_float((unsigned)c7);
        } else
#endif
        {
          ns0 = nsum[nt * 132 + ln];
          ns1 = nsum[nt * 132 + 64 + ln];
          nsq = nsum[nt * 132 + 128];
        }
#ifdef MPK_VTN_N7S2
        if (n7drow) {
#endif
        char const *const nhb = rop_hb + (long long)nt * oproj_output_stride * 2;
#ifdef MPK_VTN_N7HB
        // MPK_VTN_N7HB: the row as 64-bit relaxed system-scope loads (sc0 sc1, not volatile): one round trip,
        // waited at the quantizer. Lanes past the row read unit nua and keep zeros, as before.
        {
          typedef unsigned long long __attribute__((address_space(1))) n7_u64g;
          auto n7ld = [&](char const *p, rop7_i32x4_t &v) {
            n7_u64g *const q = (n7_u64g *)p;
            unsigned long long const a = __hip_atomic_load(q, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            unsigned long long const b =
                __hip_atomic_load(q + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            v[0] = (int)(unsigned)a;
            v[1] = (int)(unsigned)(a >> 32);
            v[2] = (int)(unsigned)b;
            v[3] = (int)(unsigned)(b >> 32);
          };
          rop7_i32x4_t t4, t5;
          n7ld(nhb + nua * 32, h0);
          n7ld(nhb + nua * 32 + 16, h1);
          n7ld(nhb + nub * 32, h2);
          n7ld(nhb + nub * 32 + 16, h3);
          n7ld(nhb + (nvc ? nuc : nua) * 32, t4);
          n7ld(nhb + (nvc ? nuc : nua) * 32 + 16, t5);
          h4 = rop7_i32x4_t{};
          h5 = rop7_i32x4_t{};
          if (nvc) {
            h4 = t4;
            h5 = t5;
          }
        }
#else
        h0 = *(n7_gv *)(nhb + nua * 32);
        h1 = *(n7_gv *)(nhb + nua * 32 + 16);
        h2 = *(n7_gv *)(nhb + nub * 32);
        h3 = *(n7_gv *)(nhb + nub * 32 + 16);
        h4 = rop7_i32x4_t{};
        h5 = rop7_i32x4_t{};
        if (nvc) {
          h4 = *(n7_gv *)(nhb + nuc * 32);
          h5 = *(n7_gv *)(nhb + nuc * 32 + 16);
        }
#endif
#ifdef MPK_VTN_N7S2
        } else {
          h0 = rop7_i32x4_t{};
          h1 = rop7_i32x4_t{};
          h2 = rop7_i32x4_t{};
          h3 = rop7_i32x4_t{};
          h4 = rop7_i32x4_t{};
          h5 = rop7_i32x4_t{};
        }
#endif
      };
#ifdef MPK_VTN_UNIONE
#if defined(MPK_VTN_N7S2_RT) || defined(MPK_VTN_N7S1_ABIP) || !defined(MPK_VTN_N7S2)
#error "MPK_VTN_UNIONE: needs MPK_VTN_N7S2 without MPK_VTN_N7S2_RT / MPK_VTN_N7S1_ABIP"
#endif
#ifdef MPK_VTN_N7S1
#define MPK_UE_S1(n) ((n) <= 2)
#else
#define MPK_UE_S1(n) false
#endif
#ifdef MPK_VTN_UNIONE_ABIP
#define MPK_UE_AB mpk_abip_odd
#else
#define MPK_UE_AB true
#endif
// Passes whose last Phase 7b job on wave 0 is a selection: 1-2 live tokens (N7S1), 5-6 (N7S2's round 2).
#ifdef MPK_VTN_UNIONE3
#define MPK_UNIONE_ON(n) (MPK_UE_AB && (MPK_UE_S1(n) || (n) == 3 || ((n) > 4 && (n) <= 6)))
#define MPK_UE_W(n) (((n) == 3) ? 3 : 0)  // MPK_VTN_UNIONE3: wave 3 is idle in Phase 7b at 3 live
#else
#define MPK_UNIONE_ON(n) (MPK_UE_AB && (MPK_UE_S1(n) || ((n) > 4 && (n) <= 6)))
#endif
      __shared__ unsigned s_n7ue[QKV_BATCH_SIZE];  // MPK_VTN_UNIONE: per token, the layer's tag once selected
#endif
      auto n7_compute = [&](int nt, float ns0, float ns1, float nsq, rop7_i32x4_t const &h0,
                            rop7_i32x4_t const &h1, rop7_i32x4_t const &h2, rop7_i32x4_t const &h3,
                            rop7_i32x4_t const &h4, rop7_i32x4_t const &h5 MPK_N7S2_CP) {
        // Token nt's logits (experts ln, ln + 64), as the 2-token path computes them.
        MPK_N7S2(0);
        float const nirms = rsqrtf(nsq / (float)ACTUAL_HIDDEN_DIM + 1e-5f);
#ifdef MPK_VTN_N7S2
        if (n7dsel) {
#endif
#ifdef MPK_VTN_KREG
        unsigned short n7lb0, n7lb1;
#endif
        {
          float s0 = ns0;
          s0 *= nirms;
          s0 += nb0;
          nlg[nt * 128 + ln] = __builtin_bit_cast(unsigned short, __float2bfloat16(s0));
          float s1 = ns1;
          s1 *= nirms;
          s1 += nb1;
          nlg[nt * 128 + 64 + ln] = __builtin_bit_cast(unsigned short, __float2bfloat16(s1));
#ifdef MPK_VTN_KREG
          n7lb0 = __builtin_bit_cast(unsigned short, __float2bfloat16(s0));
          n7lb1 = __builtin_bit_cast(unsigned short, __float2bfloat16(s1));
#endif
        }
#if !defined(MPK_VTN_KREG) || !defined(MPK_VTN_TKDEF) || defined(MPK_VTN_TKDEF_CHECK) || \
    defined(MPK_VTN_KREG_CHECK)
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        __builtin_amdgcn_wave_barrier();
#endif
        MPK_N7S2(1);
        {
          unsigned short const *const lg = nlg + nt * 128;
#ifdef MPK_VTN_KREG
          // MPK_VTN_KREG: the lane's own logits (experts ln, ln + 64) are its keys: the same 128 keys, so the same
          // four, without the LDS transpose.
          unsigned k0 = ksel_key(n7lb0, ln);
          unsigned k1 = ksel_key(n7lb1, ln + 64);
#else
          unsigned k0 = ksel_key(lg[2 * ln], 2 * ln);
          unsigned k1 = ksel_key(lg[2 * ln + 1], 2 * ln + 1);
#endif
          unsigned kg[4];
#ifdef MPK_VTN_KMERGE
#ifndef MPK_KSEL_MERGE
#error "MPK_VTN_KMERGE needs MPK_KSEL_MERGE (ksel_top4_merge)"
#endif
          // MPK_VTN_KMERGE: the four largest keys in one merge network (same keys as the four rounds).
          ksel_top4_merge(k0, k1, kg);
#else
#pragma unroll
          for (int k_idx = 0; k_idx < 4; ++k_idx) {
            unsigned const g = ksel_wave_max_fast(k0 > k1 ? k0 : k1);
            kg[k_idx] = g;
            if (k0 == g) k0 = 0u;
            if (k1 == g) k1 = 0u;
          }
#endif
#ifdef MPK_VTN_KREG_CHECK
          {
            // MPK_VTN_KREG_CHECK: the transposed keys' four rounds, against kg.
            unsigned c0 = ksel_key(lg[2 * ln], 2 * ln);
            unsigned c1 = ksel_key(lg[2 * ln + 1], 2 * ln + 1);
            unsigned kbad = 0u;
#pragma unroll
            for (int k_idx = 0; k_idx < 4; ++k_idx) {
              unsigned const g = ksel_wave_max_fast(c0 > c1 ? c0 : c1);
              kbad |= g ^ kg[k_idx];
              if (c0 == g) c0 = 0u;
              if (c1 == g) c1 = 0u;
            }
            if (ln == 0) {
              unsigned const n = atomicAdd(&g_kreg_chk[0], 1u) + 1u;
              if (kbad != 0u) atomicAdd(&g_kreg_chk[1], 1u);
              if ((n & 0x3FFFFu) == 0u) printf("[KREGCHK] checks=%u bad=%u\n", n, g_kreg_chk[1]);
            }
          }
#endif
#ifdef MPK_N7_TS2
          if (nt == nw) {
            unsigned long long _t;
            asm volatile("s_memrealtime %0\n\ts_waitcnt lgkmcnt(0)" : "=s"(_t) : "s"(kg[3]) : "memory");
            n7s2[2] = (unsigned int)_t;
          }
#endif
#ifdef MPK_VTN_TKDEF
          static_assert(MPK_VTN_SCR + 2048 + QKV_BATCH_SIZE * (132 * 4 + 128 * 2 + 16) <=
                            mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE -
                                mirage::runtime::LAYER_IDX_SMEM_OFFSET_FROM_END,
                        "MPK_VTN_TKDEF: key stash above the logits");
          if (ln == 0) {
            unsigned *const ntk =
                (unsigned *)(_rop_smem + MPK_VTN_SCR + 2048 + QKV_BATCH_SIZE * (132 * 4 + 128 * 2));
#pragma unroll
            for (int k_idx = 0; k_idx < 4; ++k_idx) {
              unsigned const g = kg[k_idx];
              int const r = 127 - (int)(g & 0xFFFFu);
              s_vt2_sel[nt][k_idx] = r;
              if (nt == 0) s_ltk_sel[k_idx] = r;
              ntk[nt * 4 + k_idx] = g;
            }
          }
#endif
#if !defined(MPK_VTN_TKDEF) || defined(MPK_VTN_TKDEF_CHECK)
          if (ln < 16) {
            float rc[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
              rc[e] = __uint_as_float(((unsigned)lg[ln * 8 + e]) << 16);
            }
            float tmax = rc[0];
#pragma unroll
            for (int ii = 1; ii < 8; ++ii) {
              tmax = fmaxf(tmax, rc[ii]);
            }
            tmax = topk_dpp_row_max_to_lane_zero(tmax);
            tmax = __int_as_float(__builtin_amdgcn_readfirstlane(__float_as_int(tmax)));
            float rsum = 0.f;
#pragma unroll
            for (int ii = 0; ii < 8; ++ii) {
              rc[ii] = __builtin_amdgcn_exp2f((rc[ii] - tmax) * 1.4426950408889634f);
              rsum += rc[ii];
            }
            rsum = topk_dpp_row_sum_to_lane_zero(rsum);
            rsum = __int_as_float(__builtin_amdgcn_readfirstlane(__float_as_int(rsum)));
            float const inv = 1.f / rsum;
            if (ln == 0) {
#pragma unroll
              for (int k_idx = 0; k_idx < 4; ++k_idx) {
                unsigned const g = kg[k_idx];
                int const r = 127 - (int)(g & 0xFFFFu);
                unsigned const o = g >> 16;
                unsigned const b = (o & 0x8000u) ? (o & 0x7FFFu) : (~o & 0xFFFFu);
                s_vt2_sel[nt][k_idx] = r;
                s_vt2_w[nt][k_idx] = __uint_as_float(b << 16);
                if (nt == 0) s_ltk_sel[k_idx] = r;
              }
              s_vt2_w[nt][4] = tmax;
              s_vt2_w[nt][5] = inv;
              float vt2_p[4];
              float vt2_rs = 0.f;
#pragma unroll
              for (int k_idx = 0; k_idx < 4; ++k_idx) {
                unsigned const o = kg[k_idx] >> 16;
                unsigned const b = (o & 0x8000u) ? (o & 0x7FFFu) : (~o & 0xFFFFu);
                float const e = __builtin_amdgcn_exp2f(
                    __fmul_rn(__fsub_rn(__uint_as_float(b << 16), tmax), 1.4426950408889634f));
                vt2_p[k_idx] = __fmul_rn(e, inv);
                vt2_rs = __fadd_rn(vt2_rs, vt2_p[k_idx]);
              }
              float const vt2_rinv = __fdiv_rn(1.f, vt2_rs);
#pragma unroll
              for (int k_idx = 0; k_idx < 4; ++k_idx) {
                s_vt2_w[nt][k_idx] = __fmul_rn(vt2_p[k_idx], vt2_rinv);
              }
            }
          }
#endif  // !MPK_VTN_TKDEF || MPK_VTN_TKDEF_CHECK
        }
#ifdef MPK_VTN_UNIONE
        if (ln == 0) {  // MPK_VTN_UNIONE: token nt's selection is in LDS (stores kept in order)
          asm volatile("" ::: "memory");
          ((volatile unsigned *)s_n7ue)[nt] = rop_tag;
        }
#endif
        MPK_N7S2(3);
#ifdef MPK_VTN_N7S2
        }
        if (n7dqnt) {
#endif
        // Row nt: the 2-token quantizer (MPK_VT2_QUANT_FAST) with this token's scale, into LDS row nt.
        static_assert(ACTUAL_HIDDEN_DIM % 16 == 0, "MPK_VTN_N7B: whole tail units");
        char *const nrow = _rop_smem + nt * MPK_VTN_ROW;
        auto nquant = [&](rop7_i32x4_t const &qh0, rop7_i32x4_t const &qh1, rop7_i32x4_t const &qg0,
                          rop7_i32x4_t const &qg1, int unit, bool ok) {
          bool const tail = unit >= ACTUAL_HIDDEN_DIM / 16;
          unsigned const hw[8] = {(unsigned)qh0[0], (unsigned)qh0[1], (unsigned)qh0[2], (unsigned)qh0[3],
                                  (unsigned)qh1[0], (unsigned)qh1[1], (unsigned)qh1[2], (unsigned)qh1[3]};
          unsigned const gw[8] = {(unsigned)qg0[0], (unsigned)qg0[1], (unsigned)qg0[2], (unsigned)qg0[3],
                                  (unsigned)qg1[0], (unsigned)qg1[1], (unsigned)qg1[2], (unsigned)qg1[3]};
#ifdef MPK_VTN_TKQ
          // MPK_VTN_TKQ: packed products, packed magnitude max, the bf16 pair straight into the FP8 conversion.
          typedef float tkq_f2 __attribute__((ext_vector_type(2)));
          typedef unsigned short tkq_u2 __attribute__((ext_vector_type(2)));
          unsigned pkw[8];
          tkq_u2 tkq_am = {0, 0};
#pragma unroll
          for (int d = 0; d < 8; ++d) {
            tkq_f2 const hv = {__uint_as_float(hw[d] << 16), __uint_as_float(hw[d] & 0xFFFF0000u)};
            tkq_f2 const gv = {__uint_as_float(gw[d] << 16), __uint_as_float(gw[d] & 0xFFFF0000u)};
            tkq_f2 const nv = (hv * nirms) * gv;
            uint32_t pk;
            asm("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(pk) : "v"(nv[0]), "v"(nv[1]));
            pkw[d] = pk;
            unsigned const ab = pk & 0x7FFF7FFFu;
            tkq_u2 abv;
            __builtin_memcpy(&abv, &ab, 4);
            tkq_am = __builtin_elementwise_max(tkq_am, abv);
          }
          unsigned const tkq_amb = tkq_am[0] > tkq_am[1] ? (unsigned)tkq_am[0] : (unsigned)tkq_am[1];
          float amax = __uint_as_float(tkq_amb << 16);
#else
          float q[16];
          float m[8];
#pragma unroll
          for (int d = 0; d < 8; ++d) {
            float const n0 = __uint_as_float(hw[d] << 16) * nirms * __uint_as_float(gw[d] << 16);
            float const n1 = __uint_as_float(hw[d] & 0xFFFF0000u) * nirms *
                             __uint_as_float(gw[d] & 0xFFFF0000u);
            uint32_t pk;
            asm("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(pk) : "v"(n0), "v"(n1));
            q[2 * d] = __uint_as_float(pk << 16);
            q[2 * d + 1] = __uint_as_float(pk & 0xFFFF0000u);
            m[d] = fmaxf(fabsf(q[2 * d]), fabsf(q[2 * d + 1]));
          }
          float amax = fmaxf(fmaxf(fmaxf(m[0], m[1]), fmaxf(m[2], m[3])),
                             fmaxf(fmaxf(m[4], m[5]), fmaxf(m[6], m[7])));
#endif
          amax = tail ? 0.0f : fmaxf(0.0f, amax);
          amax = fmaxf(amax, __int_as_float(__builtin_amdgcn_update_dpp(
                                 0, __float_as_int(amax), 0xB1, 0xF, 0xF, false)));
          amax = fmaxf(amax, __int_as_float(__builtin_amdgcn_update_dpp(
                                 0, __float_as_int(amax), 0x4E, 0xF, 0xF, false)));
          amax = fmaxf(amax, __int_as_float(__builtin_amdgcn_update_dpp(
                                 0, __float_as_int(amax), 0x141, 0xF, 0xF, false)));
          uint8_t const se = _gang_compute_e8m0_fp8(amax);
          float scale_f = 1.0f;
          if (se != 0) {
            uint32_t const sbits = (uint32_t)se << 23;
            __builtin_memcpy(&scale_f, &sbits, sizeof(scale_f));
          }
          rop7_i32x4_t out;
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            fp8x4_t pk = {};
#ifdef MPK_VTN_TKQ
            typedef __bf16 tkq_b2 __attribute__((ext_vector_type(2)));
            tkq_b2 p0, p1;
            __builtin_memcpy(&p0, &pkw[2 * k], 4);
            __builtin_memcpy(&p1, &pkw[2 * k + 1], 4);
            pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16(pk, p0, scale_f, false);
            pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16(pk, p1, scale_f, true);
#else
            pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(pk, q[4 * k], q[4 * k + 1], scale_f, false);
            pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(pk, q[4 * k + 2], q[4 * k + 3], scale_f, true);
#endif
            int bits;
            __builtin_memcpy(&bits, &pk, sizeof(bits));
            out[k] = tail ? 0 : bits;
          }
          if (ok) {
            *(rop7_i32x4_t *)(nrow + unit * 16) = out;
            if ((unit & 7) == 0) {
              nrow[oproj_output_stride + (unit >> 3)] = (char)se;
            }
          }
        };
#if defined(MPK_VTN_N7QI) && defined(MPK_VTN_TKQ)
#ifdef MPK_VTN_N7QI_ABIP
        if (mpk_abip_odd) {
#else
        {
#endif
          // MPK_VTN_N7QI: the three units interleaved, one branch-free block (same operations per unit).
          typedef float qi_f2 __attribute__((ext_vector_type(2)));
          typedef unsigned short qi_u2 __attribute__((ext_vector_type(2)));
          typedef __bf16 qi_b2 __attribute__((ext_vector_type(2)));
          auto qprod = [&](rop7_i32x4_t const &qh0, rop7_i32x4_t const &qh1, rop7_i32x4_t const &qg0,
                           rop7_i32x4_t const &qg1, unsigned (&pkw)[8]) -> unsigned {
            unsigned const hw[8] = {(unsigned)qh0[0], (unsigned)qh0[1], (unsigned)qh0[2], (unsigned)qh0[3],
                                    (unsigned)qh1[0], (unsigned)qh1[1], (unsigned)qh1[2], (unsigned)qh1[3]};
            unsigned const gw[8] = {(unsigned)qg0[0], (unsigned)qg0[1], (unsigned)qg0[2], (unsigned)qg0[3],
                                    (unsigned)qg1[0], (unsigned)qg1[1], (unsigned)qg1[2], (unsigned)qg1[3]};
            qi_u2 am = {0, 0};
#pragma unroll
            for (int d = 0; d < 8; ++d) {
              qi_f2 const hv = {__uint_as_float(hw[d] << 16), __uint_as_float(hw[d] & 0xFFFF0000u)};
              qi_f2 const gv = {__uint_as_float(gw[d] << 16), __uint_as_float(gw[d] & 0xFFFF0000u)};
              qi_f2 const nv = (hv * nirms) * gv;
              uint32_t pk;
              asm("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(pk) : "v"(nv[0]), "v"(nv[1]));
              pkw[d] = pk;
              unsigned const ab = pk & 0x7FFF7FFFu;
              qi_u2 abv;
              __builtin_memcpy(&abv, &ab, 4);
              am = __builtin_elementwise_max(am, abv);
            }
            return am[0] > am[1] ? (unsigned)am[0] : (unsigned)am[1];
          };
          unsigned pa[8], pb[8], pc[8];
          unsigned const ma = qprod(h0, h1, g0, g1, pa);
          unsigned const mb = qprod(h2, h3, g2, g3, pb);
          unsigned const mc = qprod(h4, h5, g4, g5, pc);
          bool const qta = nua >= ACTUAL_HIDDEN_DIM / 16;
          bool const qtb = nub >= ACTUAL_HIDDEN_DIM / 16;
          bool const qtc = nuc >= ACTUAL_HIDDEN_DIM / 16;
          float xa = qta ? 0.0f : fmaxf(0.0f, __uint_as_float(ma << 16));
          float xb = qtb ? 0.0f : fmaxf(0.0f, __uint_as_float(mb << 16));
          float xc = qtc ? 0.0f : fmaxf(0.0f, __uint_as_float(mc << 16));
#define MPK_QI_DPP(x, c) x = fmaxf(x, __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), c, 0xF, 0xF, false)))
          MPK_QI_DPP(xa, 0xB1);
          MPK_QI_DPP(xb, 0xB1);
          MPK_QI_DPP(xc, 0xB1);
          MPK_QI_DPP(xa, 0x4E);
          MPK_QI_DPP(xb, 0x4E);
          MPK_QI_DPP(xc, 0x4E);
          MPK_QI_DPP(xa, 0x141);
          MPK_QI_DPP(xb, 0x141);
          MPK_QI_DPP(xc, 0x141);
#undef MPK_QI_DPP
          // _gang_compute_e8m0_fp8 without its amax == 0 branch: +-0 * (1 / 448) has exponent and mantissa 0.
          auto qe8 = [](float amax) -> unsigned {
            unsigned const u = __float_as_uint(amax * (1.0f / 448.0f));
            unsigned const e = ((u >> 23) & 0xFFu) + ((u & 0x7FFFFFu) != 0u ? 1u : 0u);
            return e < 255u ? e : 255u;
          };
          unsigned const sea = qe8(xa), seb = qe8(xb), sec = qe8(xc);
          float const sfa = sea != 0u ? __uint_as_float(sea << 23) : 1.0f;
          float const sfb = seb != 0u ? __uint_as_float(seb << 23) : 1.0f;
          float const sfc = sec != 0u ? __uint_as_float(sec << 23) : 1.0f;
          auto qcvt = [&](unsigned const (&pkw)[8], float sf, bool tail) -> rop7_i32x4_t {
            rop7_i32x4_t out;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              qi_b2 p0, p1;
              __builtin_memcpy(&p0, &pkw[2 * k], 4);
              __builtin_memcpy(&p1, &pkw[2 * k + 1], 4);
              fp8x4_t pk = {};
              pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16(pk, p0, sf, false);
              pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16(pk, p1, sf, true);
              int bits;
              __builtin_memcpy(&bits, &pk, sizeof(bits));
              out[k] = tail ? 0 : bits;
            }
            return out;
          };
          rop7_i32x4_t const oa = qcvt(pa, sfa, qta);
          rop7_i32x4_t const ob = qcvt(pb, sfb, qtb);
          rop7_i32x4_t const oc = qcvt(pc, sfc, qtc);
          *(rop7_i32x4_t *)(nrow + nua * 16) = oa;
          *(rop7_i32x4_t *)(nrow + nub * 16) = ob;
          if (nvc) {
            *(rop7_i32x4_t *)(nrow + nuc * 16) = oc;
          }
          if ((nua & 7) == 0) nrow[oproj_output_stride + (nua >> 3)] = (char)sea;
          if ((nub & 7) == 0) nrow[oproj_output_stride + (nub >> 3)] = (char)seb;
          if (nvc && (nuc & 7) == 0) nrow[oproj_output_stride + (nuc >> 3)] = (char)sec;
        }
#ifdef MPK_VTN_N7QI_ABIP
        else {
          nquant(h0, h1, g0, g1, nua, true);
          nquant(h2, h3, g2, g3, nub, true);
          nquant(h4, h5, g4, g5, nuc, nvc);
        }
#endif
#else
        nquant(h0, h1, g0, g1, nua, true);
        nquant(h2, h3, g2, g3, nub, true);
        nquant(h4, h5, g4, g5, nuc, nvc);
#endif
        MPK_N7S2(4);
#ifdef MPK_VTN_N7S2
        }
#endif
      };
#ifdef MPK_TOPK_SUB
      // MPK_N7_TS: wave 1's Phase 7b steps into the [TOPKSUB] accumulators (timing tool).
      unsigned int n7ts[5] = {0u, 0u, 0u, 0u, 0u};
#define MPK_N7_TS(k) \
  do { if (tid == 64) n7ts[(k)] = (unsigned int)__builtin_amdgcn_s_memrealtime(); } while (0)
#else
#define MPK_N7_TS(k) do { } while (0)
#endif
      {
        float a0, a1, aq;
        rop7_i32x4_t ah0, ah1, ah2, ah3, ah4, ah5;
        MPK_N7_TS(0);
#ifdef MPK_VTN_LIVEB  // MPK_VTN_LIVEB_PATCHED
        int const n7_live = num_active_tokens < QKV_BATCH_SIZE ? num_active_tokens : QKV_BATCH_SIZE;
        bool const n7_one = nw < n7_live;
#else
        constexpr bool n7_one = true;
#endif
#ifdef MPK_VTN_N7S1
#if !defined(MPK_VTN_N7S2) || !defined(MPK_VTN_LIVEB)
#error "MPK_VTN_N7S1 needs MPK_VTN_N7S2 and MPK_VTN_LIVEB"
#endif
        // MPK_VTN_N7S1: with 1-2 live tokens waves 0 / 1 select tokens 0 / 1 and waves 2 / 3 quantize their rows.
        bool const n7s1 = n7_live <= 2
#ifdef MPK_VTN_N7S1_ABIP
                          && mpk_abip_odd
#endif
            ;
        int const n7t1 = (n7s1 && nw >= 2) ? nw - 2 : nw;
        bool const n7a1 = n7s1 ? n7t1 < n7_live : n7_one;
        bool const n7sel1 = !n7s1 || nw < 2;
        bool const n7qnt1 = !n7s1 || nw >= 2;
        if (n7a1) n7_load(n7t1, a0, a1, aq, ah0, ah1, ah2, ah3, ah4, ah5 MPK_N7S2_LA(n7qnt1));
#define MPK_N7S1_C1() \
  do { if (n7a1) n7_compute(n7t1, a0, a1, aq, ah0, ah1, ah2, ah3, ah4, ah5 MPK_N7S2_CA(n7sel1, n7qnt1)); } while (0)
#else
        if (n7_one) n7_load(nw, a0, a1, aq, ah0, ah1, ah2, ah3, ah4, ah5 MPK_N7S2_LA(true));
#define MPK_N7S1_C1() \
  do { if (n7_one) n7_compute(nw, a0, a1, aq, ah0, ah1, ah2, ah3, ah4, ah5 MPK_N7S2_CA(true, true)); } while (0)
#endif
        MPK_N7_TS(1);
        MPK_TKOPS_AT(11, tid == 0);
        if constexpr (n7_rounds > 1) {
#ifdef MPK_VTN_LIVEB
          bool const n7_two = nw + 4 < n7_live;
#else
          bool const n7_two = nw + 4 < QKV_BATCH_SIZE;
#endif
          float b0 = 0.0f, b1 = 0.0f, bq = 0.0f;
          rop7_i32x4_t bh0 = {}, bh1 = {}, bh2 = {}, bh3 = {}, bh4 = {}, bh5 = {};
#ifdef MPK_VTN_N7B_PF
          if (n7_two) n7_load(nw + 4, b0, b1, bq, bh0, bh1, bh2, bh3, bh4, bh5 MPK_N7S2_LA(true));
          MPK_N7S1_C1();
          if (n7_two) n7_compute(nw + 4, b0, b1, bq, bh0, bh1, bh2, bh3, bh4, bh5 MPK_N7S2_CA(true, true));
#else
          MPK_N7S1_C1();
          MPK_TKOPS_AT(12, tid == 0);
          MPK_N7_TS(2);
          MPK_N7_TS(3);
#ifdef MPK_VTN_N7S2
          // MPK_VTN_N7S2: with 5 or 6 live tokens waves 0 / 1 select tokens 4 / 5 and waves 2 / 3 quantize their
          // rows; otherwise wave w takes token w + 4 whole.
          bool const n7f = n7_live > 4 && n7_live <= 6
#ifdef MPK_VTN_N7S2_RT
#ifdef MPK_ABIP
                           && (g_mpk_n7s2_rt == 1 || (g_mpk_n7s2_rt == 2 && mpk_abip_odd))
#else
                           && g_mpk_n7s2_rt != 0
#endif
#endif
              ;
          int const n7t2 = (n7f && nw >= 2) ? nw + 2 : nw + 4;
          if (n7t2 < n7_live) {
            bool const n7q2 = !(n7f && nw < 2);
            n7_load(n7t2, b0, b1, bq, bh0, bh1, bh2, bh3, bh4, bh5 MPK_N7S2_LA(n7q2));
            MPK_N7_TS(3);
            n7_compute(n7t2, b0, b1, bq, bh0, bh1, bh2, bh3, bh4, bh5 MPK_N7S2_CA(!(n7f && nw >= 2), n7q2));
            MPK_TKOPS_AT(9, tid == 0);
          }
#else
          if (n7_two) {
            n7_load(nw + 4, b0, b1, bq, bh0, bh1, bh2, bh3, bh4, bh5);
            MPK_N7_TS(3);
            n7_compute(nw + 4, b0, b1, bq, bh0, bh1, bh2, bh3, bh4, bh5);
            MPK_TKOPS_AT(9, tid == 0);
          }
#endif
          MPK_N7_TS(4);
#endif
        } else {
          MPK_N7S1_C1();
          MPK_TKOPS_AT(12, tid == 0);
          MPK_N7_TS(2);
          MPK_N7_TS(3);
          MPK_N7_TS(4);
        }
      }
#if defined(MPK_ABIP) && defined(MPK_ABIP_CAL_NS)
      if (mpk_abip_odd) {
        // MPK_ABIP_CAL_NS: calibration delay after Phase 7b on on-passes.
        unsigned long long const _cal_t0 = __builtin_amdgcn_s_memrealtime();
        while (__builtin_amdgcn_s_memrealtime() - _cal_t0 < (unsigned long long)((MPK_ABIP_CAL_NS) / 10)) {
          __builtin_amdgcn_s_sleep(1);
        }
      }
#endif
#if defined(MPK_TOPK_SUB) && defined(MPK_N7_TS2)
      if (tid == 64 && MPK_TOPK_ARMED()) {
        s_tk_acc[0] += (n7s2[1] - n7s2[0]) * 10;
        s_tk_acc[1] += (n7s2[2] - n7s2[1]) * 10;
        s_tk_acc[2] += (n7s2[3] - n7s2[2]) * 10;
        s_tk_acc[3] += (n7s2[4] - n7s2[3]) * 10;
        s_tk_n++;
      }
#endif
#if defined(MPK_TOPK_SUB) && !defined(MPK_VQ_TS) && !defined(MPK_ROPW_TS) && !defined(MPK_N7_TS2) && \
    !defined(MPK_QT_TS)
      if (tid == 64 && MPK_TOPK_ARMED()) {
        s_tk_acc[0] += (n7ts[1] - n7ts[0]) * 10;
        s_tk_acc[1] += (n7ts[2] - n7ts[1]) * 10;
        s_tk_acc[2] += (n7ts[3] - n7ts[2]) * 10;
        s_tk_acc[3] += (n7ts[4] - n7ts[3]) * 10;
        s_tk_n++;
      }
#endif
#ifdef MPK_VTN_UNIONE
      {
        int const ue_live = num_active_tokens < QKV_BATCH_SIZE ? num_active_tokens : QKV_BATCH_SIZE;
#ifdef MPK_VTN_UNIONE3
        if (MPK_UNIONE_ON(ue_live) && (tid >> 6) == MPK_UE_W(ue_live)) {
#else
        if (MPK_UNIONE_ON(ue_live) && tid < 64) {
#endif
          // MPK_VTN_UNIONE: every live token's selection, then the union tables (the code below the barrier).
          while (true) {
            bool ue_ok = true;
            for (int t = 0; t < ue_live; ++t) {
              ue_ok = ue_ok && (((volatile unsigned *)s_n7ue)[t] == rop_tag);
            }
            if (ue_ok) break;
            __builtin_amdgcn_s_sleep(1);
          }
          asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
#ifdef MPK_VTN_UNIONE3
          int const ue_tid = tid & 63;
          {
          int const tid = ue_tid;
#endif
      // Lane (t, k) holds token t's k-th expert. Union slots in first-appearance order, columns in token
      // order: an expert occurs at most once per token, so its column is the number of earlier lanes
      // holding it.
      extern __shared__ char _vtn_smem[];
      MpkVtnTab *const vt = MPK_VTN_TABP(_vtn_smem);
      // Dead tokens of a partly accepted pass stay out of the union (MPK_VTN_BAL round); live columns unchanged.
      bool const vlive = tid < 4 * (num_active_tokens < MPK_VTN_T ? num_active_tokens : MPK_VTN_T);
      int const vt_t = tid >> 2, vt_k = tid & 3;
      int const ve = vlive ? s_vt2_sel[vt_t][vt_k] : -1 - tid;
      int vrep = tid, vcol = 0, vcnt = 0;
#if defined(MPK_VTN_UNIONM)
      {
        // MPK_VTN_UNIONM: the live lanes holding this lane's expert as one mask, from seven ballots of the id bits
        // (live ids 0..127); vrep / vcol / vcnt = lowest lane / lanes below this one / count: the loop's values.
        unsigned long long vm = __ballot(vlive);
#pragma unroll
        for (int b = 0; b < 7; ++b) {
          bool const vb = ((ve >> b) & 1) != 0;
          unsigned long long const mb = __ballot(vlive && vb);
          vm &= vb ? mb : ~mb;
        }
        if (vlive) {
          vrep = (int)__builtin_ctzll(vm);
          vcol = (int)__popcll(vm & ((1ull << tid) - 1ull));
          vcnt = (int)__popcll(vm);
        }
      }
#endif
#if !defined(MPK_VTN_UNIONM) || defined(MPK_VTN_UNIONM_CHECK)
      {
        int lrep = tid, lcol = 0, lcnt = 0;
#pragma unroll
        for (int j = 0; j < 4 * MPK_VTN_T; ++j) {
          bool const veq = __builtin_amdgcn_readlane(ve, j) == ve;
          lrep = (veq && j < lrep) ? j : lrep;
          lcol += (veq && j < tid) ? 1 : 0;
          lcnt += veq ? 1 : 0;
        }
#if defined(MPK_VTN_UNIONM_CHECK)
        if (vlive && (lrep != vrep || lcol != vcol || lcnt != vcnt)) atomicAdd(&g_unionm_chk[1], 1u);
        if (tid == 0) {
          unsigned const c = atomicAdd(&g_unionm_chk[0], 1u) + 1u;
          if ((c & 0x7FFFu) == 0u) printf("[UNIONMCHK] checks=%u bad=%u\n", c, g_unionm_chk[1]);
        }
#else
        vrep = lrep;
        vcol = lcol;
        vcnt = lcnt;
#endif
      }
#endif
      unsigned long long const vfirst = __ballot(vlive && vrep == tid);
      int const vu = __popcll(vfirst & ((1ull << vrep) - 1));
      if (vlive) {
        vt->ct[vu][vcol] = (unsigned char)vt_t;
        vt->cs[vu][vcol] = (unsigned char)vt_k;
        if (vrep == tid) {
          vt->ue[vu] = ve;
          vt->un[vu] = vcnt;
        }
      }
      if (tid == 0) vt->U = (int)__popcll(vfirst);
#ifdef MPK_VTN_UNIONE3
          }
#endif
        }
      }
#endif
      __syncthreads();
      MPK_TKOPS_AT(13, tid == 0);
    } else
#endif
    {
      // Router logits and this rank's FP8 MoE row, from the O-proj partials.
      // Every XCD's local-last published 129 sums (expert partials over its
      // 368 columns + their sum of squares) into this AID's replica; add the
      // eight, then normalize and quantize the row straight into the W13
      // handoff LDS. Replaces the global O-proj barrier, the 128 router
      // workers and the handoff load.
      static_assert(MPK_AID_ROP_AGG_INTS == 288, "agg offsets below are 1152 B");
#ifdef MPK_VT2
      static_assert(NUM_EXPERTS == 128 && QKV_BATCH_SIZE <= 2, "VT2: 1 or 2 tokens / 128");
#else
      static_assert(NUM_EXPERTS == 128 && QKV_BATCH_SIZE == 1, "bs1 / 128");
#endif
      typedef int __attribute__((ext_vector_type(4))) rop7_i32x4_t;
      __shared__ float s_rop[132];
      extern __shared__ char _rop_smem[];
      unsigned const rop_tag = (unsigned)routing_expected;
      int const rop_units = oproj_output_stride / 16;
      int const *const rop_agg =
          g_rop_rep[xcd_id >> 2] + MPK_AID_ROP_AGG_BASE_INTS +
          (routing_expected & 1) * MPK_NUM_XCDS * MPK_AID_ROP_AGG_INTS;
#ifdef MPK_VT2
#if defined(MPK_ROP_QOVL) || defined(MPK_ROP_KSEL) || defined(MPK_ROP_MERGED_POLL) || \
    defined(MPK_ROP_RELAY) || defined(MPK_ROP_HOME) || defined(MPK_ROP_LOCAL_PROBE) || \
    defined(MPK_ROP_ROW_EARLY)
#error "MPK_VT2 + MPK_ROUTER_OPROJ_PARTIAL: token 1 is wired for the plain router-partial path"
#endif
      // Token 1: die sums at MPK_AID_ROP_AGG1_BASE_INTS, gathered sums + logits in dynamic LDS past the
      // 3 KiB hand-off row (dead until the MoE), FP8 row into this worker's private slot.
      int const *const rop_agg1 = g_rop_rep[xcd_id >> 2] + MPK_AID_ROP_AGG1_BASE_INTS +
                                  (routing_expected & 1) * MPK_NUM_XCDS * MPK_AID_ROP_AGG_INTS;
#ifdef MPK_VTN_MOE
      // MPK_VTN_MOE_PATCHED: token rows at MPK_VTN_ROW stride from LDS 0; scratch above the MoE staging.
      float *const s_rop1 = (float *)(_rop_smem + MPK_VTN_SCR);
      unsigned short *const s_lg1 = (unsigned short *)(_rop_smem + MPK_VTN_SCR + 132 * 4);
#else
      float *const s_rop1 = (float *)(_rop_smem + 3072);
      unsigned short *const s_lg1 = (unsigned short *)(_rop_smem + 3072 + 132 * 4);
#endif
      char *const rop_q1 = (char *)(g_aid_nc_rep[xcd_id >> 2] + MPK_NC_ROP1ROW_INTS +
                                    ((xcd_id & 3) * 32 + xcd_rank) * 1024);
      rop7_i32x4_t r1h0 = {}, r1h1 = {};
#ifdef MPK_VT2_ROP1_L2
#ifndef MPK_W13_HANDOFF_SC1
#error "MPK_VT2_ROP1_L2 leaves the row in L2: the token-1 hand-off load must be sc1 (MPK_W13_HANDOFF_SC1)"
#endif
#define MPK_ROP1_ST_MOD ""
#else
#define MPK_ROP1_ST_MOD " sc0 sc1"
#endif
#endif
#ifdef MPK_ROP_HOME
#if defined(MPK_ROP_RELAY) || defined(MPK_ROP_LOCAL_PROBE)
#error "MPK_ROP_HOME: not with MPK_ROP_RELAY / MPK_ROP_LOCAL_PROBE"
#endif
      // Each XCD's sums live only in its own AID's replica.
      int const *const rop_agg_a0 =
          g_rop_rep[0] + MPK_AID_ROP_AGG_BASE_INTS +
          (routing_expected & 1) * MPK_NUM_XCDS * MPK_AID_ROP_AGG_INTS;
      int const *const rop_agg_a1 =
          g_rop_rep[1] + MPK_AID_ROP_AGG_BASE_INTS +
          (routing_expected & 1) * MPK_NUM_XCDS * MPK_AID_ROP_AGG_INTS;
#endif
#ifdef MPK_TOPK_SUB
      unsigned int _tk0 = 0, _tk1 = 0, _tk2 = 0, _tk3 = 0;
#ifdef MPK_TOPK_SUB2
      unsigned int _tkg1 = 0, _tkr = 0;
      unsigned int _tkx[3] = {0u, 0u, 0u};
#endif
#endif
#ifdef MPK_TOPK_SUB
      if (tid == 0) _tk0 = (unsigned int)__builtin_amdgcn_s_memrealtime();
#endif
      // Layer constants first; they overlap the wait.
      float rop_bias = 0.0f;
      if (tid < NUM_EXPERTS && input_ptrs[14] != nullptr) {
        unsigned short const *rb = (unsigned short const *)input_ptrs[14] -
                                   xcd_id * (NUM_EXPERTS / MPK_NUM_XCDS);
        rop_bias = __uint_as_float((unsigned)rb[tid] << 16);
      }
#if defined(MPK_ROP_QOVL) && defined(MPK_ROP_KSEL)
      // Wave 0 runs the TopK; waves 1 / 3 / 2 quantize units 0-63 / 64-127 /
      // 128-183, one per lane (wave 2 after its softmax stats).
      int const rop_w = tid >> 6;
      bool const rop_qw = rop_w != 0;
      int const rop_ua = (tid & 63) + (rop_w == 3 ? 64 : rop_w == 2 ? 128 : 0);
      int const rop_ub = 0;
      bool const rop_va = rop_qw && rop_ua < rop_units;
      bool const rop_vb = false;
#elif defined(MPK_ROP_QOVL)
      // Waves 1 and 3 quantize while waves 0 and 2 run the TopK: wave 1 owns
      // units [0, 64) and [128, 192), wave 3 owns [64, 128). 8-lane domains
      // stay lane-aligned in both sets.
      bool const rop_qw = ((tid >> 6) & 1) != 0;
      int const rop_ua = (tid & 63) + ((tid >> 7) << 6);
      int const rop_ub = rop_ua + 128;
      bool const rop_va = rop_qw && rop_ua < rop_units;
      bool const rop_vb = rop_qw && rop_ub < rop_units;
#elif defined(MPK_VT2_QSPLIT)
#if !defined(MPK_VT2) || !defined(MPK_VT2_QUANT_FAST) || !defined(MPK_VT2_KSEL2) || \
    defined(MPK_VT2_MPOLL) || defined(MPK_ROP_MERGED_POLL)
#error "MPK_VT2_QSPLIT: 2-token plain gather with MPK_VT2_QUANT_FAST + MPK_VT2_KSEL2"
#endif
      // Waves 2 / 3 quantize rows 0 / 1 (units l, l+64, l+128) while waves 0 / 1 select; 8-lane scale
      // domains stay lane-aligned.
      int const rop_w = tid >> 6;
#ifdef MPK_VT2_QSPLIT_RT
      bool const rop_rt = g_mpk_qsplit_rt != 0;
#else
      constexpr bool rop_rt = true;
#endif
      int const rop_ua = rop_rt ? (tid & 63) : tid;
      int const rop_ub = rop_ua + 64;
      int const rop_uc = rop_ua + 128;
      bool const rop_qs = rop_rt && rop_w >= 2;
      bool const rop_va = rop_rt ? rop_qs : tid < rop_units;
      bool const rop_vb = rop_qs;
      bool const rop_vc = rop_qs && rop_uc < rop_units;
#else
      int const rop_ua = tid;
      int const rop_ub = 0;
      bool const rop_va = tid < rop_units;
      bool const rop_vb = false;
#endif
      rop7_i32x4_t rop_g0 = {}, rop_g1 = {}, rop_h0 = {}, rop_h1 = {};
      rop7_i32x4_t rop_g2 = {}, rop_g3 = {}, rop_h2 = {}, rop_h3 = {};
      unsigned short const *const rop_gb =
          (unsigned short const *)input_ptrs[11]
#ifdef MPK_GAMMA_AID
          + ((xcd_id >= (MPK_NUM_XCDS / 2)) ? oproj_output_stride : 0)
#endif
          ;
      if (rop_va) {
        rop7_i32x4_t const *gp = (rop7_i32x4_t const *)(rop_gb + rop_ua * 16);
        rop_g0 = gp[0];
        rop_g1 = gp[1];
      }
      if (rop_vb) {
        rop7_i32x4_t const *gp = (rop7_i32x4_t const *)(rop_gb + rop_ub * 16);
        rop_g2 = gp[0];
        rop_g3 = gp[1];
      }
#ifdef MPK_VT2_QSPLIT
      rop7_i32x4_t rop_g4 = {}, rop_g5 = {}, rop_h4 = {}, rop_h5 = {};
      if (rop_vc) {
        rop7_i32x4_t const *gp = (rop7_i32x4_t const *)(rop_gb + rop_uc * 16);
        rop_g4 = gp[0];
        rop_g5 = gp[1];
      }
#endif
#if defined(MPK_ROP_RELAY) && defined(MPK_XSNAP_EARLY)
#error "MPK_XSNAP_EARLY reuses XSNAP slots 11/12"
#endif
#ifdef MPK_ROP_RELAY
      // One worker per XCD fetches the eight XCD sums across the fabric; the
      // rest read its copy out of this XCD's L2. Without it all 248 workers
      // poll the same 8 KiB with sc0 sc1, i.e. memory round trips that queue
      // behind each other: pollers on one XCD saw the same publish up to
      // 600 ns apart. The first worker of the XCD to get here is the relay
      // (XCD-local swap of the layer tag).
      __shared__ int s_rop_relay;
      MPK_XS_MARK(4);
      int *const rop_rctr = g_rop_rep[xcd_id >> 2] +
                            MPK_AID_ROP_RLY_CTR_BASE_INTS + xcd_id * 32;
      char *const rop_rslot =
          (char *)(g_rop_rep[xcd_id >> 2] + MPK_AID_ROP_RLY_BASE_INTS +
                   ((routing_expected & 1) * MPK_NUM_XCDS + xcd_id) *
                       MPK_AID_ROP_AGG_INTS);
      if (tid == 0) {
        unsigned rop_prev;
        asm volatile("flat_atomic_swap %0, %1, %2 sc0\n"
                     "s_waitcnt vmcnt(0) lgkmcnt(0)"
                     : "=v"(rop_prev)
                     : "v"(rop_rctr), "v"(rop_tag)
                     : "memory");
        s_rop_relay = rop_prev != rop_tag;
      }
      __syncthreads();
      MPK_XS_MARK(12);
      int const rop_wi = tid < 129 ? tid : 128;
      float rop_sum;
      if (s_rop_relay != 0) {
        char const *a0 = (char const *)(rop_agg + 2 * rop_wi);
        char const *a1 = a0 + 4 * MPK_AID_ROP_AGG_INTS * 4;
        unsigned long long e0, e1, e2, e3, e4, e5, e6, e7;
        while (true) {
          asm volatile("global_load_dwordx2 %0, %8, off sc0 sc1\n"
                       "global_load_dwordx2 %1, %8, off offset:1152 sc0 sc1\n"
                       "global_load_dwordx2 %2, %8, off offset:2304 sc0 sc1\n"
                       "global_load_dwordx2 %3, %8, off offset:3456 sc0 sc1\n"
                       "global_load_dwordx2 %4, %9, off sc0 sc1\n"
                       "global_load_dwordx2 %5, %9, off offset:1152 sc0 sc1\n"
                       "global_load_dwordx2 %6, %9, off offset:2304 sc0 sc1\n"
                       "global_load_dwordx2 %7, %9, off offset:3456 sc0 sc1\n"
                       "s_waitcnt vmcnt(0)"
                       : "=&v"(e0), "=&v"(e1), "=&v"(e2), "=&v"(e3),
                         "=&v"(e4), "=&v"(e5), "=&v"(e6), "=&v"(e7)
                       : "v"(a0), "v"(a1)
                       : "memory");
          if ((unsigned)(e0 >> 32) == rop_tag &&
              (unsigned)(e1 >> 32) == rop_tag &&
              (unsigned)(e2 >> 32) == rop_tag &&
              (unsigned)(e3 >> 32) == rop_tag &&
              (unsigned)(e4 >> 32) == rop_tag &&
              (unsigned)(e5 >> 32) == rop_tag &&
              (unsigned)(e6 >> 32) == rop_tag &&
              (unsigned)(e7 >> 32) == rop_tag) {
            break;
          }
          __builtin_amdgcn_s_sleep(1);
        }
        rop_sum = __uint_as_float((unsigned)e0) +
                  __uint_as_float((unsigned)e1) +
                  __uint_as_float((unsigned)e2) +
                  __uint_as_float((unsigned)e3) +
                  __uint_as_float((unsigned)e4) +
                  __uint_as_float((unsigned)e5) +
                  __uint_as_float((unsigned)e6) +
                  __uint_as_float((unsigned)e7);
        MPK_XS_MARK(11);
        if (tid < 129) {
          // Plain store: lands in this XCD's L2, where the readers poll it.
          unsigned long long const w =
              ((unsigned long long)rop_tag << 32) | __float_as_uint(rop_sum);
          asm volatile("global_store_dwordx2 %0, %1, off" ::"v"(
                           rop_rslot + tid * 8),
                       "v"(w)
                       : "memory");
        }
      } else {
        unsigned long long w;
        while (true) {
          asm volatile("global_load_dwordx2 %0, %1, off sc1\n"
                       "s_waitcnt vmcnt(0)"
                       : "=v"(w)
                       : "v"(rop_rslot + rop_wi * 8)
                       : "memory");
          if ((unsigned)(w >> 32) == rop_tag) {
            break;
          }
          __builtin_amdgcn_s_sleep(1);
        }
        rop_sum = __uint_as_float((unsigned)w);
      }
      MPK_XS_MARK(5);
      // This lane has seen all eight publishes (itself or via the relay), and
      // every XCD's h columns retired before its sums were published.
      char const *const rop_hb = (char const *)mpk_aid_out_base(
          output_ptrs[5], xcd_id, QKV_BATCH_SIZE * oproj_output_stride * 2);
      if (rop_va) {
        char const *hp = rop_hb + rop_ua * 32;
        asm volatile("global_load_dwordx4 %0, %2, off sc0 sc1\n"
                     "global_load_dwordx4 %1, %2, off offset:16 sc0 sc1"
                     : "=&v"(rop_h0), "=&v"(rop_h1)
                     : "v"(hp)
                     : "memory");
      }
      if (rop_vb) {
        char const *hp = rop_hb + rop_ub * 32;
        asm volatile("global_load_dwordx4 %0, %2, off sc0 sc1\n"
                     "global_load_dwordx4 %1, %2, off offset:16 sc0 sc1"
                     : "=&v"(rop_h2), "=&v"(rop_h3)
                     : "v"(hp)
                     : "memory");
      }
      if (tid < 129) {
        s_rop[tid] = rop_sum;
      }
      MPK_XS_MARK(9);
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      MPK_XS_MARK(10);
      __syncthreads();
      MPK_XS_MARK(6);
#else
#if defined(MPK_XSNAP) && defined(MPK_PHASE_SNAP) && MPK_XSNAP_LAT == 1
      if (tid == 0) {
        unsigned int const _xl_l =
            s_phase_layers -
            (unsigned int)(MPK_PHASE_SNAP * MPK_PHASE_LAYERS_PER_ITER);
        int const *const _xl_loc = g_rop_rep[xcd_id >> 2] + 103680;
        int const *const _xl_rem = g_rop_rep[1 - (xcd_id >> 2)] + 103680;
        int _xl_v0, _xl_v1;
        unsigned long long const _xl_t0 = __builtin_amdgcn_s_memrealtime();
        unsigned long long const _xl_t1 = __builtin_amdgcn_s_memrealtime();
        asm volatile("global_load_dword %0, %1, off sc0 sc1\n"
                     "s_waitcnt vmcnt(0)"
                     : "=v"(_xl_v0)
                     : "v"(_xl_loc)
                     : "memory");
        unsigned long long const _xl_t2 = __builtin_amdgcn_s_memrealtime();
        asm volatile("global_load_dword %0, %1, off sc0 sc1\n"
                     "s_waitcnt vmcnt(0)"
                     : "=v"(_xl_v1)
                     : "v"(_xl_rem)
                     : "memory");
        unsigned long long const _xl_t3 = __builtin_amdgcn_s_memrealtime();
        if (_xl_l < (unsigned int)MPK_PHASE_LAYERS_PER_ITER &&
            (unsigned int)s_xs_worker < (unsigned int)MPK_PHASE_MAX_WORKERS) {
          unsigned long long *const _xl_p =
              &g_xs_snap[((size_t)_xl_l * MPK_PHASE_MAX_WORKERS + s_xs_worker) * 16];
          _xl_p[12] = _xl_t1 - _xl_t0;
          _xl_p[14] = _xl_t2 - _xl_t1 + (unsigned long long)(_xl_v0 & 0);
          _xl_p[15] = _xl_t3 - _xl_t2 + (unsigned long long)(_xl_v1 & 0);
        }
      }
#endif
#if defined(MPK_XSNAP) && defined(MPK_PHASE_SNAP) && MPK_XSNAP_LAT == 3
      if (tid == 0) {
        unsigned int const _xl_l =
            s_phase_layers -
            (unsigned int)(MPK_PHASE_SNAP * MPK_PHASE_LAYERS_PER_ITER);
        int *const _xl_loc = g_rop_rep[xcd_id >> 2] + 103680 + 32 * xcd_id;
        int *const _xl_rem = g_rop_rep[1 - (xcd_id >> 2)] + 103680 + 32 * xcd_id;
        int const *const _xl_hot = rop_agg + 2 * 128 + 1;
        int _xl_o = 0;
        unsigned long long const _xl_t0 = __builtin_amdgcn_s_memrealtime();
        asm volatile("global_store_dword %0, %1, off sc0 sc1\n"
                     "s_waitcnt vmcnt(0)" ::"v"(_xl_loc), "v"(0) : "memory");
        unsigned long long const _xl_t1 = __builtin_amdgcn_s_memrealtime();
        asm volatile("global_store_dword %0, %1, off sc0 sc1\n"
                     "s_waitcnt vmcnt(0)" ::"v"(_xl_rem), "v"(0) : "memory");
        unsigned long long const _xl_t2 = __builtin_amdgcn_s_memrealtime();
        asm volatile("flat_atomic_add %0, %1, %2 sc0\n"
                     "s_waitcnt vmcnt(0) lgkmcnt(0)" : "=v"(_xl_o) : "v"(_xl_loc + 16), "v"(1) : "memory");
        unsigned long long const _xl_t3 = __builtin_amdgcn_s_memrealtime();
        if (_xl_l < (unsigned int)MPK_PHASE_LAYERS_PER_ITER &&
            (unsigned int)s_xs_worker < (unsigned int)MPK_PHASE_MAX_WORKERS) {
          unsigned long long *const _xl_p =
              &g_xs_snap[((size_t)_xl_l * MPK_PHASE_MAX_WORKERS + s_xs_worker) * 16];
          _xl_p[12] = _xl_t1 - _xl_t0 + (unsigned long long)(_xl_o & 0);
          _xl_p[14] = _xl_t2 - _xl_t1;
          _xl_p[15] = _xl_t3 - _xl_t2;
        }
      }
#endif
#if defined(MPK_XSNAP) && defined(MPK_PHASE_SNAP) && MPK_XSNAP_LAT == 4
      if (tid == 0) {
        unsigned int const _xl_l =
            s_phase_layers -
            (unsigned int)(MPK_PHASE_SNAP * MPK_PHASE_LAYERS_PER_ITER);
        int *const _xl_loc = g_rop_rep[xcd_id >> 2] + 103680 + 32 * xcd_id;
        int *const _xl_rem = g_rop_rep[1 - (xcd_id >> 2)] + 103680 + 32 * xcd_id;
        int const *const _xl_hot = rop_agg + 2 * 128 + 1;
        int _xl_o = 0;
        unsigned long long const _xl_t0 = __builtin_amdgcn_s_memrealtime();
        asm volatile("global_store_dword %0, %1, off\n"
                     "s_waitcnt vmcnt(0)" ::"v"(_xl_loc + 8), "v"(0) : "memory");
        unsigned long long const _xl_t1 = __builtin_amdgcn_s_memrealtime();
        asm volatile("flat_atomic_add %0, %1, %2 sc0 sc1\n"
                     "s_waitcnt vmcnt(0) lgkmcnt(0)" : "=v"(_xl_o) : "v"(_xl_loc + 24), "v"(1) : "memory");
        unsigned long long const _xl_t2 = __builtin_amdgcn_s_memrealtime();
        asm volatile("global_load_dword %0, %1, off sc0 sc1\n"
                     "s_waitcnt vmcnt(0)" : "=v"(_xl_o) : "v"(_xl_hot) : "memory");
        unsigned long long const _xl_t3 = __builtin_amdgcn_s_memrealtime();
        if (_xl_l < (unsigned int)MPK_PHASE_LAYERS_PER_ITER &&
            (unsigned int)s_xs_worker < (unsigned int)MPK_PHASE_MAX_WORKERS) {
          unsigned long long *const _xl_p =
              &g_xs_snap[((size_t)_xl_l * MPK_PHASE_MAX_WORKERS + s_xs_worker) * 16];
          _xl_p[12] = _xl_t1 - _xl_t0 + (unsigned long long)(_xl_o & 0);
          _xl_p[14] = _xl_t2 - _xl_t1;
          _xl_p[15] = _xl_t3 - _xl_t2;
        }
      }
#endif
#ifdef MPK_ROP_MERGED_POLL
#if !defined(MPK_ROP_GATHER_SC1) || defined(MPK_ROP_LOCAL_PROBE)
#error "MPK_ROP_MERGED_POLL polls the sums at device scope (MPK_ROP_GATHER_SC1), all eight dies"
#endif
#elif defined(MPK_VT2_MPOLL)
#if !defined(MPK_VT2) || !defined(MPK_VT2_QUANT_FAST) || defined(MPK_ROP_LOCAL_PROBE) || defined(MPK_ROP_HOME) || \
    defined(MPK_ROP_RELAY) || defined(MPK_ROP_ROW_EARLY) || defined(MPK_ROP_QOVL)
#error "MPK_VT2_MPOLL is wired for the plain 2-token gather"
#endif
      // No stage-1 wait: the tagged-sums gather below polls every die by itself.
#else
#ifdef MPK_VT2_MPOLL_RT
#if !defined(MPK_VT2_QSPLIT_RT) || MPK_VT2_QSPLIT != 2 || defined(MPK_ROP_RELAY) || defined(MPK_ROP_ROW_EARLY)
#error "MPK_VT2_MPOLL_RT is wired for MPK_VT2_QSPLIT=2 + MPK_VT2_QSPLIT_RT, plain gather"
#endif
      bool const rop_mp = g_mpk_mpoll_rt != 0;
      if (!rop_mp) {
#endif
      // Stage 1: one wave waits for all eight XCDs' sum-of-squares word.
      MPK_XS_MARK(4);
      if (tid < 64) {
#ifdef MPK_ROP_LOCAL_PROBE
        int const *const p =
            rop_agg + ((xcd_id >> 2) * 4 + (tid & 3)) * MPK_AID_ROP_AGG_INTS +
            2 * 128;
#elif defined(MPK_ROP_HOME)
        int const *const p = ((tid & 4) ? rop_agg_a1 : rop_agg_a0) +
                             (tid & 7) * MPK_AID_ROP_AGG_INTS + 2 * 128;
#elif defined(MPK_VT2)
        int const *const p =
            ((QKV_BATCH_SIZE == 2 && (tid & 8)) ? rop_agg1 : rop_agg) +
            (tid & 7) * MPK_AID_ROP_AGG_INTS + 2 * 128;
#else
        int const *const p =
            rop_agg + (tid & 7) * MPK_AID_ROP_AGG_INTS + 2 * 128;
#endif
        while (true) {
#ifdef MPK_ROP_WAIT_SC1
          unsigned long long v;
          asm volatile("global_load_dwordx2 %0, %1, off sc1\n\ts_waitcnt vmcnt(0)"
                       : "=v"(v)
                       : "v"(p)
                       : "memory");
#else
          unsigned long long const v = ld_sys_u64((void *)p);
#endif
          if (__all((unsigned)(v >> 32) == rop_tag)) {
            break;
          }
          __builtin_amdgcn_s_sleep(1);
        }
        MPK_XS_MARK(5);
      }
      __syncthreads();
#ifdef MPK_VT2_MPOLL_RT
      }
#endif
#endif
#ifdef MPK_TOPK_SUB
      if (tid == 0) _tk1 = (unsigned int)__builtin_amdgcn_s_memrealtime();
#endif
#ifdef MPK_ROP_PROBE
      MPK_PHASE_MARK(_pslot_w, 3);
#endif
#ifdef MPK_ROP_GATHER_SC1
#define MPK_ROP_GSC "sc1"
#else
#define MPK_ROP_GSC "sc0 sc1"
#endif
      // Stage 2: the hidden row (its AID-local copy; every XCD's columns
      // retired before that XCD's sums were published) and the 8 x 129 sums.
      char const *const rop_hb = (char const *)mpk_aid_out_base(
          output_ptrs[5], xcd_id, QKV_BATCH_SIZE * oproj_output_stride * 2);
#if !defined(MPK_ROP_MERGED_POLL) && !defined(MPK_VT2_MPOLL)
#ifdef MPK_VT2_QSPLIT
#ifdef MPK_VT2_MPOLL_RT
      if (rop_qs && !rop_mp) {
#else
      if (rop_qs) {
#endif
#if MPK_VT2_QSPLIT == 2
        // Slots A / C (units l, l+128) hold row 1 on wave 2 and row 0 on wave 3; slot B (l+64) the other row.
        long long const r1off = (long long)oproj_output_stride * 2;
        char const *const hba = rop_hb + (rop_w == 2 ? r1off : 0ll);
        char const *const hbb = rop_hb + (rop_w == 2 ? 0ll : r1off);
        char const *const hbc = hba;
#else
        char const *const hb = rop_hb + (rop_w == 3 ? (long long)oproj_output_stride * 2 : 0ll);
        char const *const hba = hb;
        char const *const hbb = hb;
        char const *const hbc = hb;
#endif
        asm volatile("global_load_dwordx4 %0, %2, off " MPK_ROP_GSC "\n"
                     "global_load_dwordx4 %1, %2, off offset:16 " MPK_ROP_GSC
                     : "=&v"(rop_h0), "=&v"(rop_h1)
                     : "v"(hba + rop_ua * 32)
                     : "memory");
        asm volatile("global_load_dwordx4 %0, %2, off " MPK_ROP_GSC "\n"
                     "global_load_dwordx4 %1, %2, off offset:16 " MPK_ROP_GSC
                     : "=&v"(rop_h2), "=&v"(rop_h3)
                     : "v"(hbb + rop_ub * 32)
                     : "memory");
        if (rop_vc) {
          asm volatile("global_load_dwordx4 %0, %2, off " MPK_ROP_GSC "\n"
                       "global_load_dwordx4 %1, %2, off offset:16 " MPK_ROP_GSC
                       : "=&v"(rop_h4), "=&v"(rop_h5)
                       : "v"(hbc + rop_uc * 32)
                       : "memory");
        }
      }
#endif
#if !defined(MPK_VT2_QSPLIT) || defined(MPK_VT2_QSPLIT_RT)
#ifdef MPK_VT2_MPOLL_RT
      if (!rop_rt && !rop_mp)
#elif defined(MPK_VT2_QSPLIT_RT)
      if (!rop_rt)
#endif
      {
      if (rop_va) {
        char const *hp = rop_hb + rop_ua * 32;
        asm volatile("global_load_dwordx4 %0, %2, off " MPK_ROP_GSC "\n"
                     "global_load_dwordx4 %1, %2, off offset:16 " MPK_ROP_GSC
                     : "=&v"(rop_h0), "=&v"(rop_h1)
                     : "v"(hp)
                     : "memory");
      }
#ifdef MPK_VT2
      if (QKV_BATCH_SIZE == 2 && rop_va) {
        char const *hp = rop_hb + (long long)oproj_output_stride * 2 + rop_ua * 32;
        asm volatile("global_load_dwordx4 %0, %2, off " MPK_ROP_GSC "\n"
                     "global_load_dwordx4 %1, %2, off offset:16 " MPK_ROP_GSC
                     : "=&v"(r1h0), "=&v"(r1h1)
                     : "v"(hp)
                     : "memory");
      }
#endif
      if (rop_vb) {
        char const *hp = rop_hb + rop_ub * 32;
        asm volatile("global_load_dwordx4 %0, %2, off " MPK_ROP_GSC "\n"
                     "global_load_dwordx4 %1, %2, off offset:16 " MPK_ROP_GSC
                     : "=&v"(rop_h2), "=&v"(rop_h3)
                     : "v"(hp)
                     : "memory");
      }
      }
#endif
#endif
#ifdef MPK_ROP_ROW_EARLY
#if !defined(MPK_ROP_MERGED_POLL) || MPK_ROP_MERGED_POLL >= 2 || !defined(MPK_ROP_KSEL) || \
    !defined(MPK_ROP_QOVL) || defined(MPK_ROP_RELAY)
#error "MPK_ROP_ROW_EARLY is wired for MPK_ROP_MERGED_POLL=1 with the QOVL + KSEL quantize waves"
#endif
      bool const rop_rre = g_mpk_rre_rt != 0;
      // Waves 2 and 3 have no sums word of their own: they poll the sum-of-squares word.
      int const rop_pw = tid < 129 ? tid : 128;
      if (tid < 129 || rop_rre) {
#else
      int const rop_pw = tid;
      if (tid < 129) {
#endif
#ifdef MPK_ROP_HOME
        char const *a0 = (char const *)(rop_agg_a0 + 2 * rop_pw);
        char const *a1 = (char const *)(rop_agg_a1 + 2 * rop_pw) +
                         4 * MPK_AID_ROP_AGG_INTS * 4;
#else
        char const *a0 = (char const *)(rop_agg + 2 * rop_pw);
        char const *a1 = a0 + 4 * MPK_AID_ROP_AGG_INTS * 4;
#endif
#if defined(MPK_ROP_MERGED_POLL) && MPK_ROP_MERGED_POLL >= 2
        unsigned long long e0 = 0ull, e1 = 0ull, e2 = 0ull, e3 = 0ull;
        unsigned long long e4 = 0ull, e5 = 0ull, e6 = 0ull, e7 = 0ull;
        unsigned rop_done = 0u;
        auto rop_ld = [](char const *p) {
          return __hip_atomic_load(
              (unsigned long long *)const_cast<char *>(p), __ATOMIC_RELAXED,
              __HIP_MEMORY_SCOPE_AGENT);
        };
        auto rop_in = [&](unsigned long long e) {
          return __all((unsigned)(e >> 32) == rop_tag) != 0;
        };
        while (true) {
          if (!(rop_done & 0x01u)) e0 = rop_ld(a0);
          if (!(rop_done & 0x02u)) e1 = rop_ld(a0 + 1152);
          if (!(rop_done & 0x04u)) e2 = rop_ld(a0 + 2304);
          if (!(rop_done & 0x08u)) e3 = rop_ld(a0 + 3456);
          if (!(rop_done & 0x10u)) e4 = rop_ld(a1);
          if (!(rop_done & 0x20u)) e5 = rop_ld(a1 + 1152);
          if (!(rop_done & 0x40u)) e6 = rop_ld(a1 + 2304);
          if (!(rop_done & 0x80u)) e7 = rop_ld(a1 + 3456);
          rop_done |= (rop_in(e0) ? 0x01u : 0u) | (rop_in(e1) ? 0x02u : 0u) |
                      (rop_in(e2) ? 0x04u : 0u) | (rop_in(e3) ? 0x08u : 0u) |
                      (rop_in(e4) ? 0x10u : 0u) | (rop_in(e5) ? 0x20u : 0u) |
                      (rop_in(e6) ? 0x40u : 0u) | (rop_in(e7) ? 0x80u : 0u);
          if (rop_done == 0xFFu) {
            break;
          }
#ifdef MPK_TOPK_SUB2
          _tkr++;
#endif
          if (rop_done == 0u) {
            __builtin_amdgcn_s_sleep(4);
          } else {
            __builtin_amdgcn_s_sleep(1);
          }
        }
#elif defined(MPK_VT2)
        unsigned long long e0, e1, e2, e3, e4, e5, e6, e7;
        unsigned long long f0 = 0ull, f1 = 0ull, f2 = 0ull, f3 = 0ull, f4 = 0ull, f5 = 0ull, f6 = 0ull, f7 = 0ull;
        if constexpr (QKV_BATCH_SIZE == 2) {
          char const *b0 = (char const *)(rop_agg1 + 2 * rop_pw);
          char const *b1 = b0 + 4 * MPK_AID_ROP_AGG_INTS * 4;
          while (true) {
            asm volatile("global_load_dwordx2 %0, %16, off " MPK_ROP_GSC "\n"
                         "global_load_dwordx2 %1, %16, off offset:1152 " MPK_ROP_GSC "\n"
                         "global_load_dwordx2 %2, %16, off offset:2304 " MPK_ROP_GSC "\n"
                         "global_load_dwordx2 %3, %16, off offset:3456 " MPK_ROP_GSC "\n"
                         "global_load_dwordx2 %4, %17, off " MPK_ROP_GSC "\n"
                         "global_load_dwordx2 %5, %17, off offset:1152 " MPK_ROP_GSC "\n"
                         "global_load_dwordx2 %6, %17, off offset:2304 " MPK_ROP_GSC "\n"
                         "global_load_dwordx2 %7, %17, off offset:3456 " MPK_ROP_GSC "\n"
                         "global_load_dwordx2 %8, %18, off " MPK_ROP_GSC "\n"
                         "global_load_dwordx2 %9, %18, off offset:1152 " MPK_ROP_GSC "\n"
                         "global_load_dwordx2 %10, %18, off offset:2304 " MPK_ROP_GSC "\n"
                         "global_load_dwordx2 %11, %18, off offset:3456 " MPK_ROP_GSC "\n"
                         "global_load_dwordx2 %12, %19, off " MPK_ROP_GSC "\n"
                         "global_load_dwordx2 %13, %19, off offset:1152 " MPK_ROP_GSC "\n"
                         "global_load_dwordx2 %14, %19, off offset:2304 " MPK_ROP_GSC "\n"
                         "global_load_dwordx2 %15, %19, off offset:3456 " MPK_ROP_GSC "\n"
                         "s_waitcnt vmcnt(0)"
                         : "=&v"(e0), "=&v"(e1), "=&v"(e2), "=&v"(e3),
                           "=&v"(e4), "=&v"(e5), "=&v"(e6), "=&v"(e7),
                           "=&v"(f0), "=&v"(f1), "=&v"(f2), "=&v"(f3),
                           "=&v"(f4), "=&v"(f5), "=&v"(f6), "=&v"(f7)
                         : "v"(a0), "v"(a1), "v"(b0), "v"(b1)
                         : "memory");
            unsigned const tg = (unsigned)rop_tag;
            if ((unsigned)(e0 >> 32) == tg && (unsigned)(e1 >> 32) == tg &&
                (unsigned)(e2 >> 32) == tg && (unsigned)(e3 >> 32) == tg &&
                (unsigned)(e4 >> 32) == tg && (unsigned)(e5 >> 32) == tg &&
                (unsigned)(e6 >> 32) == tg && (unsigned)(e7 >> 32) == tg &&
                (unsigned)(f0 >> 32) == tg && (unsigned)(f1 >> 32) == tg &&
                (unsigned)(f2 >> 32) == tg && (unsigned)(f3 >> 32) == tg &&
                (unsigned)(f4 >> 32) == tg && (unsigned)(f5 >> 32) == tg &&
                (unsigned)(f6 >> 32) == tg && (unsigned)(f7 >> 32) == tg) {
              break;
            }
            __builtin_amdgcn_s_sleep(1);
          }
          s_rop1[tid] = __uint_as_float((unsigned)f0) + __uint_as_float((unsigned)f1) +
                        __uint_as_float((unsigned)f2) + __uint_as_float((unsigned)f3) +
                        __uint_as_float((unsigned)f4) + __uint_as_float((unsigned)f5) +
                        __uint_as_float((unsigned)f6) + __uint_as_float((unsigned)f7);
        } else {
          while (true) {
            asm volatile("global_load_dwordx2 %0, %8, off " MPK_ROP_GSC "\n"
                         "global_load_dwordx2 %1, %8, off offset:1152 " MPK_ROP_GSC "\n"
                         "global_load_dwordx2 %2, %8, off offset:2304 " MPK_ROP_GSC "\n"
                         "global_load_dwordx2 %3, %8, off offset:3456 " MPK_ROP_GSC "\n"
                         "global_load_dwordx2 %4, %9, off " MPK_ROP_GSC "\n"
                         "global_load_dwordx2 %5, %9, off offset:1152 " MPK_ROP_GSC "\n"
                         "global_load_dwordx2 %6, %9, off offset:2304 " MPK_ROP_GSC "\n"
                         "global_load_dwordx2 %7, %9, off offset:3456 " MPK_ROP_GSC "\n"
                         "s_waitcnt vmcnt(0)"
                         : "=&v"(e0), "=&v"(e1), "=&v"(e2), "=&v"(e3),
                           "=&v"(e4), "=&v"(e5), "=&v"(e6), "=&v"(e7)
                         : "v"(a0), "v"(a1)
                         : "memory");
            unsigned const tg = (unsigned)rop_tag;
            if ((unsigned)(e0 >> 32) == tg && (unsigned)(e1 >> 32) == tg &&
                (unsigned)(e2 >> 32) == tg && (unsigned)(e3 >> 32) == tg &&
                (unsigned)(e4 >> 32) == tg && (unsigned)(e5 >> 32) == tg &&
                (unsigned)(e6 >> 32) == tg && (unsigned)(e7 >> 32) == tg) {
              break;
            }
            __builtin_amdgcn_s_sleep(1);
          }
        }
#else
        unsigned long long e0, e1, e2, e3, e4, e5, e6, e7;
        while (true) {
          asm volatile("global_load_dwordx2 %0, %8, off " MPK_ROP_GSC "\n"
                       "global_load_dwordx2 %1, %8, off offset:1152 " MPK_ROP_GSC "\n"
                       "global_load_dwordx2 %2, %8, off offset:2304 " MPK_ROP_GSC "\n"
                       "global_load_dwordx2 %3, %8, off offset:3456 " MPK_ROP_GSC "\n"
                       "global_load_dwordx2 %4, %9, off " MPK_ROP_GSC "\n"
                       "global_load_dwordx2 %5, %9, off offset:1152 " MPK_ROP_GSC "\n"
                       "global_load_dwordx2 %6, %9, off offset:2304 " MPK_ROP_GSC "\n"
                       "global_load_dwordx2 %7, %9, off offset:3456 " MPK_ROP_GSC "\n"
                       "s_waitcnt vmcnt(0)"
                       : "=&v"(e0), "=&v"(e1), "=&v"(e2), "=&v"(e3),
                         "=&v"(e4), "=&v"(e5), "=&v"(e6), "=&v"(e7)
                       : "v"(a0), "v"(a1)
                       : "memory");
#ifdef MPK_ROP_LOCAL_PROBE
          if ((xcd_id >> 2) == 0
                  ? ((unsigned)(e0 >> 32) == rop_tag &&
                     (unsigned)(e1 >> 32) == rop_tag &&
                     (unsigned)(e2 >> 32) == rop_tag &&
                     (unsigned)(e3 >> 32) == rop_tag)
                  : ((unsigned)(e4 >> 32) == rop_tag &&
                     (unsigned)(e5 >> 32) == rop_tag &&
                     (unsigned)(e6 >> 32) == rop_tag &&
                     (unsigned)(e7 >> 32) == rop_tag)) {
            break;
          }
#else
          if ((unsigned)(e0 >> 32) == rop_tag &&
              (unsigned)(e1 >> 32) == rop_tag &&
              (unsigned)(e2 >> 32) == rop_tag &&
              (unsigned)(e3 >> 32) == rop_tag &&
              (unsigned)(e4 >> 32) == rop_tag &&
              (unsigned)(e5 >> 32) == rop_tag &&
              (unsigned)(e6 >> 32) == rop_tag &&
              (unsigned)(e7 >> 32) == rop_tag) {
            break;
          }
#endif
#ifdef MPK_TOPK_SUB2
          _tkr++;
#endif
          __builtin_amdgcn_s_sleep(1);
        }
#endif
        MPK_XS_MARK(9);
#ifdef MPK_TOPK_SUB2
        if (tid == 0) _tkg1 = (unsigned int)__builtin_amdgcn_s_memrealtime();
#endif
#ifdef MPK_ROP_ROW_EARLY
        if (rop_rre && rop_va) {
          // This wave has seen every die's tag, so every die's row columns are visible.
          char const *hp = rop_hb + rop_ua * 32;
          asm volatile("global_load_dwordx4 %0, %2, off " MPK_ROP_GSC "\n"
                       "global_load_dwordx4 %1, %2, off offset:16 " MPK_ROP_GSC
                       : "=&v"(rop_h0), "=&v"(rop_h1)
                       : "v"(hp)
                       : "memory");
        }
        if (tid < 129)
#endif
        s_rop[tid] = __uint_as_float((unsigned)e0) +
                     __uint_as_float((unsigned)e1) +
                     __uint_as_float((unsigned)e2) +
                     __uint_as_float((unsigned)e3) +
                     __uint_as_float((unsigned)e4) +
                     __uint_as_float((unsigned)e5) +
                     __uint_as_float((unsigned)e6) +
                     __uint_as_float((unsigned)e7);
      }
#ifdef MPK_ROP_ROW_EARLY
      if (!rop_rre)
#endif
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      MPK_XS_MARK(10);
      __syncthreads();
      MPK_XS_MARK(6);
#endif  // MPK_ROP_RELAY
#ifdef MPK_VT2_MPOLL_RT
      // Merged poll: every die's sums carried this layer's tag, so its row columns are visible.
      if (rop_mp) {
        if (rop_qs) {
          long long const r1off = (long long)oproj_output_stride * 2;
          char const *const hba = rop_hb + (rop_w == 2 ? r1off : 0ll);
          char const *const hbb = rop_hb + (rop_w == 2 ? 0ll : r1off);
          asm volatile("global_load_dwordx4 %0, %2, off " MPK_ROP_GSC "\n"
                       "global_load_dwordx4 %1, %2, off offset:16 " MPK_ROP_GSC
                       : "=&v"(rop_h0), "=&v"(rop_h1)
                       : "v"(hba + rop_ua * 32)
                       : "memory");
          asm volatile("global_load_dwordx4 %0, %2, off " MPK_ROP_GSC "\n"
                       "global_load_dwordx4 %1, %2, off offset:16 " MPK_ROP_GSC
                       : "=&v"(rop_h2), "=&v"(rop_h3)
                       : "v"(hbb + rop_ub * 32)
                       : "memory");
          if (rop_vc) {
            asm volatile("global_load_dwordx4 %0, %2, off " MPK_ROP_GSC "\n"
                         "global_load_dwordx4 %1, %2, off offset:16 " MPK_ROP_GSC
                         : "=&v"(rop_h4), "=&v"(rop_h5)
                         : "v"(hba + rop_uc * 32)
                         : "memory");
          }
        }
        if (!rop_rt && rop_va) {
          char const *hp = rop_hb + rop_ua * 32;
          asm volatile("global_load_dwordx4 %0, %2, off " MPK_ROP_GSC "\n"
                       "global_load_dwordx4 %1, %2, off offset:16 " MPK_ROP_GSC
                       : "=&v"(rop_h0), "=&v"(rop_h1)
                       : "v"(hp)
                       : "memory");
          char const *hq = rop_hb + (long long)oproj_output_stride * 2 + rop_ua * 32;
          asm volatile("global_load_dwordx4 %0, %2, off " MPK_ROP_GSC "\n"
                       "global_load_dwordx4 %1, %2, off offset:16 " MPK_ROP_GSC
                       : "=&v"(r1h0), "=&v"(r1h1)
                       : "v"(hq)
                       : "memory");
        }
      }
#endif
#ifdef MPK_VT2_MPOLL
      // Every die's sums carried this layer's tag, so its row columns are visible. The quantize
      // waits for these loads.
      if (rop_va) {
        char const *hp = rop_hb + rop_ua * 32;
        asm volatile("global_load_dwordx4 %0, %2, off " MPK_ROP_GSC "\n"
                     "global_load_dwordx4 %1, %2, off offset:16 " MPK_ROP_GSC
                     : "=&v"(rop_h0), "=&v"(rop_h1)
                     : "v"(hp)
                     : "memory");
      }
      if (QKV_BATCH_SIZE == 2 && rop_va) {
        char const *hp = rop_hb + (long long)oproj_output_stride * 2 + rop_ua * 32;
        asm volatile("global_load_dwordx4 %0, %2, off " MPK_ROP_GSC "\n"
                     "global_load_dwordx4 %1, %2, off offset:16 " MPK_ROP_GSC
                     : "=&v"(r1h0), "=&v"(r1h1)
                     : "v"(hp)
                     : "memory");
      }
#endif
#ifdef MPK_ROP_MERGED_POLL
      // Every die has published (all 129 sums threads saw its tags), so its
      // row columns are visible; the converting waves wait for these at the
      // start of their side work.
#ifdef MPK_ROP_ROW_EARLY
      if (!rop_rre && rop_va) {
#else
      if (rop_va) {
#endif
        char const *hp = rop_hb + rop_ua * 32;
        asm volatile("global_load_dwordx4 %0, %2, off " MPK_ROP_GSC "\n"
                     "global_load_dwordx4 %1, %2, off offset:16 " MPK_ROP_GSC
                     : "=&v"(rop_h0), "=&v"(rop_h1)
                     : "v"(hp)
                     : "memory");
      }
      if (rop_vb) {
        char const *hp = rop_hb + rop_ub * 32;
        asm volatile("global_load_dwordx4 %0, %2, off " MPK_ROP_GSC "\n"
                     "global_load_dwordx4 %1, %2, off offset:16 " MPK_ROP_GSC
                     : "=&v"(rop_h2), "=&v"(rop_h3)
                     : "v"(hp)
                     : "memory");
      }
#endif
#ifdef MPK_TOPK_SUB
      if (tid == 0) _tk2 = (unsigned int)__builtin_amdgcn_s_memrealtime();
#endif
#ifdef MPK_ROP_PROBE
      MPK_PHASE_MARK(_pslot_w, 4);
#endif
      float const rop_irms =
          rsqrtf(s_rop[128] / (float)ACTUAL_HIDDEN_DIM + 1e-5f);
      if (tid < NUM_EXPERTS) {
        float s = s_rop[tid];
        s *= rop_irms;
        s += rop_bias;
        s_ltk_logit[tid] =
            __builtin_bit_cast(unsigned short, __float2bfloat16(s));
      }
#ifdef MPK_VT2
      float const rop_irms1 =
          QKV_BATCH_SIZE == 2 ? rsqrtf(s_rop1[128] / (float)ACTUAL_HIDDEN_DIM + 1e-5f) : 0.0f;
      if (QKV_BATCH_SIZE == 2 && tid < NUM_EXPERTS) {
        float s1 = s_rop1[tid];
        s1 *= rop_irms1;
        s1 += rop_bias;
        s_lg1[tid] = __builtin_bit_cast(unsigned short, __float2bfloat16(s1));
      }
#endif
      // Normed row -> FP8 + E8M0 (128-element domains = 8 lanes), the same
      // bits the router's pass 2 published: bf16 round trip, then quant. One
      // call quantizes one 16-element unit per lane; the whole wave calls it
      // (the amax shuffles), `ok` only gates the stores.
      auto rop_quant = [&](rop7_i32x4_t const &h0, rop7_i32x4_t const &h1,
                           rop7_i32x4_t const &g0, rop7_i32x4_t const &g1,
                           int unit, bool ok) {
        unsigned const hw[8] = {(unsigned)h0[0], (unsigned)h0[1],
                                (unsigned)h0[2], (unsigned)h0[3],
                                (unsigned)h1[0], (unsigned)h1[1],
                                (unsigned)h1[2], (unsigned)h1[3]};
        unsigned const gw[8] = {(unsigned)g0[0], (unsigned)g0[1],
                                (unsigned)g0[2], (unsigned)g0[3],
                                (unsigned)g1[0], (unsigned)g1[1],
                                (unsigned)g1[2], (unsigned)g1[3]};
        float q[16];
        float amax = 0.0f;
#pragma unroll
        for (int d = 0; d < 8; ++d) {
          float n0 = __uint_as_float(hw[d] << 16) * rop_irms *
                     __uint_as_float(gw[d] << 16);
          float n1 = __uint_as_float(hw[d] & 0xFFFF0000u) * rop_irms *
                     __uint_as_float(gw[d] & 0xFFFF0000u);
          int const e = unit * 16 + d * 2;
          if (e >= ACTUAL_HIDDEN_DIM) {
            n0 = 0.0f;
          }
          if (e + 1 >= ACTUAL_HIDDEN_DIM) {
            n1 = 0.0f;
          }
          uint32_t pk;
          asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2"
                       : "=v"(pk)
                       : "v"(n0), "v"(n1));
          q[2 * d] = __uint_as_float(pk << 16);
          q[2 * d + 1] = __uint_as_float(pk & 0xFFFF0000u);
          amax = fmaxf(amax, fmaxf(fabsf(q[2 * d]), fabsf(q[2 * d + 1])));
        }
#ifdef MPK_ROP_KSEL
        // Quad max, then row_half_mirror pairs each quad with the other quad
        // of its 8-lane domain.
        amax = fmaxf(amax, __int_as_float(__builtin_amdgcn_update_dpp(
                               0, __float_as_int(amax), 0xB1, 0xF, 0xF, false)));
        amax = fmaxf(amax, __int_as_float(__builtin_amdgcn_update_dpp(
                               0, __float_as_int(amax), 0x4E, 0xF, 0xF, false)));
        amax = fmaxf(amax, __int_as_float(__builtin_amdgcn_update_dpp(
                               0, __float_as_int(amax), 0x141, 0xF, 0xF, false)));
#else
        amax = fmaxf(amax, __shfl_xor(amax, 1));
        amax = fmaxf(amax, __shfl_xor(amax, 2));
        amax = fmaxf(amax, __shfl_xor(amax, 4));
#endif
        uint8_t const se = _gang_compute_e8m0_fp8(amax);
        float scale_f = 1.0f;
        if (se != 0) {
          uint32_t const sbits = (uint32_t)se << 23;
          __builtin_memcpy(&scale_f, &sbits, sizeof(scale_f));
        }
        rop7_i32x4_t out;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          fp8x4_t pk = {};
          pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(pk, q[4 * k],
                                                        q[4 * k + 1], scale_f,
                                                        false);
          pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(pk, q[4 * k + 2],
                                                        q[4 * k + 3], scale_f,
                                                        true);
          int bits;
          __builtin_memcpy(&bits, &pk, sizeof(bits));
          out[k] = bits;
        }
        if (ok) {
          *(rop7_i32x4_t *)(_rop_smem + unit * 16) = out;
          if ((unit & 7) == 0) {
            _rop_smem[oproj_output_stride + (unit >> 3)] = (char)se;
          }
        }
      };
#ifdef MPK_VT2
      auto rop_quant1 = [&](rop7_i32x4_t const &h0, rop7_i32x4_t const &h1,
                           rop7_i32x4_t const &g0, rop7_i32x4_t const &g1,
                           int unit, bool ok) {
        unsigned const hw[8] = {(unsigned)h0[0], (unsigned)h0[1],
                                (unsigned)h0[2], (unsigned)h0[3],
                                (unsigned)h1[0], (unsigned)h1[1],
                                (unsigned)h1[2], (unsigned)h1[3]};
        unsigned const gw[8] = {(unsigned)g0[0], (unsigned)g0[1],
                                (unsigned)g0[2], (unsigned)g0[3],
                                (unsigned)g1[0], (unsigned)g1[1],
                                (unsigned)g1[2], (unsigned)g1[3]};
        float q[16];
        float amax = 0.0f;
#pragma unroll
        for (int d = 0; d < 8; ++d) {
          float n0 = __uint_as_float(hw[d] << 16) * rop_irms1 *
                     __uint_as_float(gw[d] << 16);
          float n1 = __uint_as_float(hw[d] & 0xFFFF0000u) * rop_irms1 *
                     __uint_as_float(gw[d] & 0xFFFF0000u);
          int const e = unit * 16 + d * 2;
          if (e >= ACTUAL_HIDDEN_DIM) {
            n0 = 0.0f;
          }
          if (e + 1 >= ACTUAL_HIDDEN_DIM) {
            n1 = 0.0f;
          }
          uint32_t pk;
          asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2"
                       : "=v"(pk)
                       : "v"(n0), "v"(n1));
          q[2 * d] = __uint_as_float(pk << 16);
          q[2 * d + 1] = __uint_as_float(pk & 0xFFFF0000u);
          amax = fmaxf(amax, fmaxf(fabsf(q[2 * d]), fabsf(q[2 * d + 1])));
        }
#ifdef MPK_ROP_KSEL
        // Quad max, then row_half_mirror pairs each quad with the other quad
        // of its 8-lane domain.
        amax = fmaxf(amax, __int_as_float(__builtin_amdgcn_update_dpp(
                               0, __float_as_int(amax), 0xB1, 0xF, 0xF, false)));
        amax = fmaxf(amax, __int_as_float(__builtin_amdgcn_update_dpp(
                               0, __float_as_int(amax), 0x4E, 0xF, 0xF, false)));
        amax = fmaxf(amax, __int_as_float(__builtin_amdgcn_update_dpp(
                               0, __float_as_int(amax), 0x141, 0xF, 0xF, false)));
#else
        amax = fmaxf(amax, __shfl_xor(amax, 1));
        amax = fmaxf(amax, __shfl_xor(amax, 2));
        amax = fmaxf(amax, __shfl_xor(amax, 4));
#endif
        uint8_t const se = _gang_compute_e8m0_fp8(amax);
        float scale_f = 1.0f;
        if (se != 0) {
          uint32_t const sbits = (uint32_t)se << 23;
          __builtin_memcpy(&scale_f, &sbits, sizeof(scale_f));
        }
        rop7_i32x4_t out;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          fp8x4_t pk = {};
          pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(pk, q[4 * k],
                                                        q[4 * k + 1], scale_f,
                                                        false);
          pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(pk, q[4 * k + 2],
                                                        q[4 * k + 3], scale_f,
                                                        true);
          int bits;
          __builtin_memcpy(&bits, &pk, sizeof(bits));
          out[k] = bits;
        }
        if (ok) {
#ifdef MPK_VTN_MOE
          *(rop7_i32x4_t *)(_rop_smem + MPK_VTN_ROW + unit * 16) = out;
#else
          asm volatile("global_store_dwordx4 %0, %1, off" MPK_ROP1_ST_MOD ::"v"(rop_q1 + unit * 16),
                       "v"(out)
                       : "memory");
#endif
          if ((unit & 7) == 0) {
#ifdef MPK_VTN_MOE
            _rop_smem[MPK_VTN_ROW + oproj_output_stride + (unit >> 3)] = (char)se;
#else
            asm volatile("global_store_byte %0, %1, off" MPK_ROP1_ST_MOD ::"v"(
                             rop_q1 + oproj_output_stride + (unit >> 3)),
                         "v"((unsigned)se)
                         : "memory");
#endif
          }
        }
      };
#endif

#ifdef MPK_ROP_QUANT_FAST
      static_assert(ACTUAL_HIDDEN_DIM % 16 == 0, "MPK_ROP_QUANT_FAST: whole tail units");
      auto rop_quant_fast = [&](rop7_i32x4_t const &h0, rop7_i32x4_t const &h1,
                                rop7_i32x4_t const &g0, rop7_i32x4_t const &g1,
                                int unit, bool ok) {
        bool const tail = unit >= ACTUAL_HIDDEN_DIM / 16;
        unsigned const hw[8] = {(unsigned)h0[0], (unsigned)h0[1],
                                (unsigned)h0[2], (unsigned)h0[3],
                                (unsigned)h1[0], (unsigned)h1[1],
                                (unsigned)h1[2], (unsigned)h1[3]};
        unsigned const gw[8] = {(unsigned)g0[0], (unsigned)g0[1],
                                (unsigned)g0[2], (unsigned)g0[3],
                                (unsigned)g1[0], (unsigned)g1[1],
                                (unsigned)g1[2], (unsigned)g1[3]};
        float q[16];
        float m[8];
#pragma unroll
        for (int d = 0; d < 8; ++d) {
          float const n0 = __uint_as_float(hw[d] << 16) * rop_irms *
                           __uint_as_float(gw[d] << 16);
          float const n1 = __uint_as_float(hw[d] & 0xFFFF0000u) * rop_irms *
                           __uint_as_float(gw[d] & 0xFFFF0000u);
          uint32_t pk;
          asm("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(pk) : "v"(n0), "v"(n1));
          q[2 * d] = __uint_as_float(pk << 16);
          q[2 * d + 1] = __uint_as_float(pk & 0xFFFF0000u);
          m[d] = fmaxf(fabsf(q[2 * d]), fabsf(q[2 * d + 1]));
        }
        float amax = fmaxf(fmaxf(fmaxf(m[0], m[1]), fmaxf(m[2], m[3])),
                           fmaxf(fmaxf(m[4], m[5]), fmaxf(m[6], m[7])));
        amax = tail ? 0.0f : fmaxf(0.0f, amax);
        amax = fmaxf(amax, __int_as_float(__builtin_amdgcn_update_dpp(
                               0, __float_as_int(amax), 0xB1, 0xF, 0xF, false)));
        amax = fmaxf(amax, __int_as_float(__builtin_amdgcn_update_dpp(
                               0, __float_as_int(amax), 0x4E, 0xF, 0xF, false)));
        amax = fmaxf(amax, __int_as_float(__builtin_amdgcn_update_dpp(
                               0, __float_as_int(amax), 0x141, 0xF, 0xF, false)));
        uint8_t const se = _gang_compute_e8m0_fp8(amax);
        float scale_f = 1.0f;
        if (se != 0) {
          uint32_t const sbits = (uint32_t)se << 23;
          __builtin_memcpy(&scale_f, &sbits, sizeof(scale_f));
        }
        rop7_i32x4_t out;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          fp8x4_t pk = {};
          pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(pk, q[4 * k],
                                                        q[4 * k + 1], scale_f,
                                                        false);
          pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(pk, q[4 * k + 2],
                                                        q[4 * k + 3], scale_f,
                                                        true);
          int bits;
          __builtin_memcpy(&bits, &pk, sizeof(bits));
          out[k] = tail ? 0 : bits;
        }
        if (ok) {
          *(rop7_i32x4_t *)(_rop_smem + unit * 16) = out;
          if ((unit & 7) == 0) {
            _rop_smem[oproj_output_stride + (unit >> 3)] = (char)se;
          }
        }
      };
#endif
#if defined(MPK_VT2) && defined(MPK_VT2_QUANT_FAST)
#ifndef MPK_ROP_QUANT_FAST
#error "MPK_VT2_QUANT_FAST reuses MPK_ROP_QUANT_FAST's quantizer"
#endif
      auto rop_quant1_fast = [&](rop7_i32x4_t const &h0, rop7_i32x4_t const &h1,
                                 rop7_i32x4_t const &g0, rop7_i32x4_t const &g1,
                                 int unit, bool ok) {
        bool const tail = unit >= ACTUAL_HIDDEN_DIM / 16;
        unsigned const hw[8] = {(unsigned)h0[0], (unsigned)h0[1],
                                (unsigned)h0[2], (unsigned)h0[3],
                                (unsigned)h1[0], (unsigned)h1[1],
                                (unsigned)h1[2], (unsigned)h1[3]};
        unsigned const gw[8] = {(unsigned)g0[0], (unsigned)g0[1],
                                (unsigned)g0[2], (unsigned)g0[3],
                                (unsigned)g1[0], (unsigned)g1[1],
                                (unsigned)g1[2], (unsigned)g1[3]};
        float q[16];
        float m[8];
#pragma unroll
        for (int d = 0; d < 8; ++d) {
          float const n0 = __uint_as_float(hw[d] << 16) * rop_irms1 *
                           __uint_as_float(gw[d] << 16);
          float const n1 = __uint_as_float(hw[d] & 0xFFFF0000u) * rop_irms1 *
                           __uint_as_float(gw[d] & 0xFFFF0000u);
          uint32_t pk;
          asm("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(pk) : "v"(n0), "v"(n1));
          q[2 * d] = __uint_as_float(pk << 16);
          q[2 * d + 1] = __uint_as_float(pk & 0xFFFF0000u);
          m[d] = fmaxf(fabsf(q[2 * d]), fabsf(q[2 * d + 1]));
        }
        float amax = fmaxf(fmaxf(fmaxf(m[0], m[1]), fmaxf(m[2], m[3])),
                           fmaxf(fmaxf(m[4], m[5]), fmaxf(m[6], m[7])));
        amax = tail ? 0.0f : fmaxf(0.0f, amax);
        amax = fmaxf(amax, __int_as_float(__builtin_amdgcn_update_dpp(
                               0, __float_as_int(amax), 0xB1, 0xF, 0xF, false)));
        amax = fmaxf(amax, __int_as_float(__builtin_amdgcn_update_dpp(
                               0, __float_as_int(amax), 0x4E, 0xF, 0xF, false)));
        amax = fmaxf(amax, __int_as_float(__builtin_amdgcn_update_dpp(
                               0, __float_as_int(amax), 0x141, 0xF, 0xF, false)));
        uint8_t const se = _gang_compute_e8m0_fp8(amax);
        float scale_f = 1.0f;
        if (se != 0) {
          uint32_t const sbits = (uint32_t)se << 23;
          __builtin_memcpy(&scale_f, &sbits, sizeof(scale_f));
        }
        rop7_i32x4_t out;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          fp8x4_t pk = {};
          pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(pk, q[4 * k],
                                                        q[4 * k + 1], scale_f,
                                                        false);
          pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(pk, q[4 * k + 2],
                                                        q[4 * k + 3], scale_f,
                                                        true);
          int bits;
          __builtin_memcpy(&bits, &pk, sizeof(bits));
          out[k] = tail ? 0 : bits;
        }
        if (ok) {
#ifdef MPK_VTN_MOE
          *(rop7_i32x4_t *)(_rop_smem + MPK_VTN_ROW + unit * 16) = out;
#else
          asm volatile("global_store_dwordx4 %0, %1, off" MPK_ROP1_ST_MOD ::"v"(rop_q1 + unit * 16),
                       "v"(out)
                       : "memory");
#endif
          if ((unit & 7) == 0) {
#ifdef MPK_VTN_MOE
            _rop_smem[MPK_VTN_ROW + oproj_output_stride + (unit >> 3)] = (char)se;
#else
            asm volatile("global_store_byte %0, %1, off" MPK_ROP1_ST_MOD ::"v"(
                             rop_q1 + oproj_output_stride + (unit >> 3)),
                         "v"((unsigned)se)
                         : "memory");
#endif
          }
        }
      };
#endif
#ifdef MPK_ROP_QOVL
      auto rop_side = [&]() {
#ifdef MPK_ROP_MERGED_POLL
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
        if (rop_qw) {
#ifdef MPK_ROP_QUANT_FAST
#ifndef MPK_ROP_KSEL
#error "MPK_ROP_QUANT_FAST is wired for the QOVL + KSEL quantize waves"
#endif
          if (g_mpk_rqf_rt != 0)
            rop_quant_fast(rop_h0, rop_h1, rop_g0, rop_g1, rop_ua, rop_va);
          else
#endif
          rop_quant(rop_h0, rop_h1, rop_g0, rop_g1, rop_ua, rop_va);
#ifndef MPK_ROP_KSEL
          if ((tid >> 6) == 1) {
            rop_quant(rop_h2, rop_h3, rop_g2, rop_g3, rop_ub, rop_vb);
          }
#endif
        }
      };
#ifdef MPK_TOPK_SUB
      if (tid == 0) _tk3 = (unsigned int)__builtin_amdgcn_s_memrealtime();
#endif
#ifdef MPK_ROP_PROBE
      MPK_PHASE_MARK(_pslot_w, 2);
#endif
#ifdef MPK_ROP_PROBE_TWICE
      mpk_local_topk<128, 4>(nullptr, routing_expected);
      MPK_PHASE_MARK(_pslot_w, 1);
#endif
      // The trailing __syncthreads publishes s_ltk_sel and the row.
#ifdef MPK_TOPK_SUB2
      mpk_local_topk<128, 4>(nullptr, routing_expected, nullptr, 0, rop_side, _tkx);
#else
      mpk_local_topk<128, 4>(nullptr, routing_expected, nullptr, 0, rop_side);
#endif
#ifdef MPK_ROP_STAGE_ROW
      // Stage dumps only: the row in the router's prequant layout (FP8 payload, then one E8M0
      // byte per 128 elements), which nothing in the megakernel reads on this path.
      if (xcd_id == 0 && xcd_rank == 0 && input_ptrs[12] != nullptr) {
        char *const stage = (char *)input_ptrs[12];
        if (tid < rop_units) {
          *(rop7_i32x4_t *)(stage + tid * 16) = *(rop7_i32x4_t const *)(_rop_smem + tid * 16);
        }
        if (tid < rop_units / 8) {
          stage[oproj_output_stride + tid] = _rop_smem[oproj_output_stride + tid];
        }
      }
#endif
#else
#ifdef MPK_VT2_MPOLL
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");  // the h' rows
#endif
#ifdef MPK_VT2_MPOLL_RT
      if (rop_mp && !rop_rt) {
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      }
#endif
#if defined(MPK_VT2) && defined(MPK_VT2_QUANT_FAST)
      // Waves 0-2 are fully active here, as the DPP amax needs (lanes 184-191 are tail units).
#if !defined(MPK_VT2_QSPLIT) || defined(MPK_VT2_QSPLIT_RT)
#ifdef MPK_VT2_QSPLIT_RT
      if (!rop_rt)
#endif
      {
        if (tid < 192) {
          rop_quant_fast(rop_h0, rop_h1, rop_g0, rop_g1, rop_ua, rop_va);
        }
        if (QKV_BATCH_SIZE == 2 && tid < 192) {
          rop_quant1_fast(r1h0, r1h1, rop_g0, rop_g1, rop_ua, rop_va);
        }
      }
#endif
#else
      if (tid < 192) {
        rop_quant(rop_h0, rop_h1, rop_g0, rop_g1, rop_ua, rop_va);
      }
#ifdef MPK_VT2
      if (QKV_BATCH_SIZE == 2 && tid < 192) {
        rop_quant1(r1h0, r1h1, rop_g0, rop_g1, rop_ua, rop_va);  // drains during token 0's selection
      }
#endif
#endif
#ifdef MPK_TOPK_SUB
      if (tid == 0) _tk3 = (unsigned int)__builtin_amdgcn_s_memrealtime();
#endif
#ifdef MPK_ROP_PROBE
      MPK_PHASE_MARK(_pslot_w, 2);
#endif
      // mpk_local_topk's __syncthreads publishes s_ltk_logit and the row.
#ifdef MPK_TOPK_WSTAMP
      __shared__ unsigned s_tkw[4];
#endif
#if defined(MPK_VT2) && defined(MPK_VT2_KSEL2)
      if constexpr (QKV_BATCH_SIZE == 2) {
#if !defined(MPK_KSEL_FAST) || !defined(MPK_LTK_SELLOGIT) || !defined(MPK_LTK_TIE_FIX) || defined(MPK_LTK_TIE_RT)
#error "MPK_VT2_KSEL2 needs MPK_KSEL_FAST + MPK_LTK_SELLOGIT + MPK_LTK_TIE_FIX"
#endif
        __syncthreads();  // both tokens' logits in LDS
        if (tid < 128) {
          int const tk = tid >> 6;
          int const ln = tid & 63;
          unsigned short const *const lg = tk ? s_lg1 : s_ltk_logit;
          unsigned k0 = ksel_key(lg[2 * ln], 2 * ln);
          unsigned k1 = ksel_key(lg[2 * ln + 1], 2 * ln + 1);
          unsigned kg[4];
#pragma unroll
          for (int k_idx = 0; k_idx < 4; ++k_idx) {
            unsigned const g = ksel_wave_max_fast(k0 > k1 ? k0 : k1);
            kg[k_idx] = g;
            if (k0 == g) k0 = 0u;
            if (k1 == g) k1 = 0u;
          }
#ifdef MPK_TOPK_WSTAMP
          if (tid == 0) s_tkw[0] = (unsigned)__builtin_amdgcn_s_memrealtime();
#endif
          // The softmax statistics over all 128 logits: 16 lanes x 8 values, mpk_local_topk's reductions.
          if (ln < 16) {
            float rc[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
              rc[e] = __uint_as_float(((unsigned)lg[ln * 8 + e]) << 16);
            }
            float tmax = rc[0];
#pragma unroll
            for (int ii = 1; ii < 8; ++ii) {
              tmax = fmaxf(tmax, rc[ii]);
            }
#ifdef MPK_TOPK_DPP_REDUCE
            tmax = topk_dpp_row_max_to_lane_zero(tmax);
            tmax = __int_as_float(__builtin_amdgcn_readfirstlane(__float_as_int(tmax)));
#else
            for (int mask = 8; mask > 0; mask /= 2) {
              tmax = fmaxf(tmax, __shfl_xor(tmax, mask, 16));
            }
#endif
            float rsum = 0.f;
#pragma unroll
            for (int ii = 0; ii < 8; ++ii) {
              rc[ii] = __builtin_amdgcn_exp2f((rc[ii] - tmax) * 1.4426950408889634f);
              rsum += rc[ii];
            }
#ifdef MPK_TOPK_DPP_REDUCE
            rsum = topk_dpp_row_sum_to_lane_zero(rsum);
            rsum = __int_as_float(__builtin_amdgcn_readfirstlane(__float_as_int(rsum)));
#else
            for (int mask = 8; mask > 0; mask /= 2) {
              rsum += __shfl_xor(rsum, mask, 16);
            }
#endif
            float const inv = 1.f / rsum;
            if (ln == 0) {
              // Raw selected logits + statistics, as mpk_local_topk leaves them before its tail.
#pragma unroll
              for (int k_idx = 0; k_idx < 4; ++k_idx) {
                unsigned const g = kg[k_idx];
                int const r = 127 - (int)(g & 0xFFFFu);
                unsigned const o = g >> 16;
                unsigned const b = (o & 0x8000u) ? (o & 0x7FFFu) : (~o & 0xFFFFu);
                s_vt2_sel[tk][k_idx] = r;
                s_vt2_w[tk][k_idx] = __uint_as_float(b << 16);
                if (tk == 0) s_ltk_sel[k_idx] = r;
              }
              s_vt2_w[tk][4] = tmax;
              s_vt2_w[tk][5] = inv;
#if defined(MPK_VT2_KSEL2_TAIL) && MPK_VT2_KSEL2_TAIL != 0
              {
                float vt2_p[4];
                float vt2_rs = 0.f;
#pragma unroll
                for (int k_idx = 0; k_idx < 4; ++k_idx) {
                  unsigned const o = kg[k_idx] >> 16;
                  unsigned const b = (o & 0x8000u) ? (o & 0x7FFFu) : (~o & 0xFFFFu);
                  float const e = __builtin_amdgcn_exp2f(
                      __fmul_rn(__fsub_rn(__uint_as_float(b << 16), tmax), 1.4426950408889634f));
                  vt2_p[k_idx] = __fmul_rn(e, inv);
#if MPK_VT2_KSEL2_TAIL == 2
                  vt2_rs = __builtin_fmaf(e, inv, vt2_rs);
#else
                  vt2_rs = __fadd_rn(vt2_rs, vt2_p[k_idx]);
#endif
                }
                float const vt2_rinv = __fdiv_rn(1.f, vt2_rs);
#pragma unroll
                for (int k_idx = 0; k_idx < 4; ++k_idx) {
                  s_vt2_w[tk][k_idx] = __fmul_rn(vt2_p[k_idx], vt2_rinv);
                }
              }
#endif
            }
          }
#ifdef MPK_TOPK_WSTAMP
          if (tid == 0) s_tkw[1] = (unsigned)__builtin_amdgcn_s_memrealtime();
#endif
        }
#ifdef MPK_VT2_QSPLIT
        else if (rop_rt) {
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");  // this wave's row loads (wave 3 skips the gather)
#ifdef MPK_TOPK_WSTAMP
          if (tid == 128) s_tkw[2] = (unsigned)__builtin_amdgcn_s_memrealtime();
#endif
#if MPK_VT2_QSPLIT == 2
          if (tid < 192) {
            rop_quant1_fast(rop_h0, rop_h1, rop_g0, rop_g1, rop_ua, true);
            rop_quant1_fast(rop_h4, rop_h5, rop_g4, rop_g5, rop_uc, rop_vc);
            rop_quant_fast(rop_h2, rop_h3, rop_g2, rop_g3, rop_ub, true);
          } else {
            rop_quant1_fast(rop_h2, rop_h3, rop_g2, rop_g3, rop_ub, true);
            rop_quant_fast(rop_h0, rop_h1, rop_g0, rop_g1, rop_ua, true);
            rop_quant_fast(rop_h4, rop_h5, rop_g4, rop_g5, rop_uc, rop_vc);
          }
#else
          if (tid < 192) {
            rop_quant_fast(rop_h0, rop_h1, rop_g0, rop_g1, rop_ua, true);
            rop_quant_fast(rop_h2, rop_h3, rop_g2, rop_g3, rop_ub, true);
            rop_quant_fast(rop_h4, rop_h5, rop_g4, rop_g5, rop_uc, rop_vc);
          } else {
            rop_quant1_fast(rop_h0, rop_h1, rop_g0, rop_g1, rop_ua, true);
            rop_quant1_fast(rop_h2, rop_h3, rop_g2, rop_g3, rop_ub, true);
            rop_quant1_fast(rop_h4, rop_h5, rop_g4, rop_g5, rop_uc, rop_vc);
          }
#endif
#ifdef MPK_TOPK_WSTAMP2
          if (tid == 128) s_tkw[2] = (unsigned)__builtin_amdgcn_s_memrealtime();
#if MPK_VT2_QSPLIT != 2
#error "MPK_TOPK_WSTAMP2 repeats the QSPLIT=2 unit map"
#endif
          if (tid < 192) {
            rop_quant1_fast(rop_h0, rop_h1, rop_g0, rop_g1, rop_ua, true);
            rop_quant1_fast(rop_h4, rop_h5, rop_g4, rop_g5, rop_uc, rop_vc);
            rop_quant_fast(rop_h2, rop_h3, rop_g2, rop_g3, rop_ub, true);
          } else {
            rop_quant1_fast(rop_h2, rop_h3, rop_g2, rop_g3, rop_ub, true);
            rop_quant_fast(rop_h0, rop_h1, rop_g0, rop_g1, rop_ua, true);
            rop_quant_fast(rop_h4, rop_h5, rop_g4, rop_g5, rop_uc, rop_vc);
          }
#endif
#ifdef MPK_TOPK_WSTAMP
          if (tid == 128) s_tkw[3] = (unsigned)__builtin_amdgcn_s_memrealtime();
#endif
        }
#endif
#if !defined(MPK_VT2_KSEL2_TAIL) || MPK_VT2_KSEL2_TAIL == 0
        __syncthreads();
        // mpk_local_topk's SELLOGIT tail (its thread 128), text and LDS operands kept, so the
        // compiler forms the same arithmetic: thread 128 finishes token 0, thread 192 token 1.
        if (tid == 128 || tid == 192) {
          float *const vt2_rw = s_vt2_w[(tid >> 6) - 2];
          float const m = vt2_rw[4];
          float const inv_sum = vt2_rw[5];
          float p[4];
          float row_sum_for_renorm = 0.f;
          for (int k_idx = 0; k_idx < 4; ++k_idx) {
            p[k_idx] = __builtin_amdgcn_exp2f((vt2_rw[k_idx] - m) *
                                              1.4426950408889634f);
            p[k_idx] *= inv_sum;
            row_sum_for_renorm += p[k_idx];
          }
          float const inv = 1.f / row_sum_for_renorm;
          for (int k_idx = 0; k_idx < 4; ++k_idx) {
            vt2_rw[k_idx] = p[k_idx] * inv;
          }
        }
#endif
#ifdef MPK_VT2_T1ROW_LATE
        if (g_mpk_t1late_rt == 0)
#endif
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");  // token 1's row stores
        __syncthreads();
#ifdef MPK_VT2_KSEL2_CHECK
        {
          // Reference: the serial mpk_local_topk path on the same logits, compared bit for bit.
          unsigned short const ks2_sv0 = tid < 128 ? s_ltk_logit[tid] : (unsigned short)0;
          for (int ks2_t = 0; ks2_t < 2; ++ks2_t) {
            if (ks2_t == 1) {
              if (tid < 128) s_ltk_logit[tid] = s_lg1[tid];
              __syncthreads();
            }
            mpk_local_topk<128, 4>(nullptr, routing_expected);
            __syncthreads();
            if (tid == 0) {
              bool bs = false, bw = false;
              for (int k = 0; k < 4; ++k) {
                bs |= s_vt2_sel[ks2_t][k] != s_ltk_sel[k];
                bw |= __float_as_uint(s_vt2_w[ks2_t][k]) != __float_as_uint(s_ltk_w[k]);
              }
              bw |= __float_as_uint(s_vt2_w[ks2_t][4]) != __float_as_uint(s_ltk_w[4]);
              bw |= __float_as_uint(s_vt2_w[ks2_t][5]) != __float_as_uint(s_ltk_w[5]);
              unsigned const n = atomicAdd(&g_ks2chk[0], 1u);
              if (bs) atomicAdd(&g_ks2chk[1], 1u);
              if (bw) atomicAdd(&g_ks2chk[2], 1u);
              if ((bs || bw) && atomicAdd(&g_ks2chk[3], 1u) < 12u) {
                printf("[KS2CHK] L%d t%d x%d r%d sel %d %d %d %d ref %d %d %d %d "
                       "w %08x %08x %08x %08x %08x %08x ref %08x %08x %08x %08x %08x %08x\n",
                       task_layer_idx, ks2_t, xcd_id, xcd_rank, s_vt2_sel[ks2_t][0],
                       s_vt2_sel[ks2_t][1], s_vt2_sel[ks2_t][2], s_vt2_sel[ks2_t][3],
                       s_ltk_sel[0], s_ltk_sel[1], s_ltk_sel[2], s_ltk_sel[3],
                       __float_as_uint(s_vt2_w[ks2_t][0]), __float_as_uint(s_vt2_w[ks2_t][1]),
                       __float_as_uint(s_vt2_w[ks2_t][2]), __float_as_uint(s_vt2_w[ks2_t][3]),
                       __float_as_uint(s_vt2_w[ks2_t][4]), __float_as_uint(s_vt2_w[ks2_t][5]),
                       __float_as_uint(s_ltk_w[0]), __float_as_uint(s_ltk_w[1]),
                       __float_as_uint(s_ltk_w[2]), __float_as_uint(s_ltk_w[3]),
                       __float_as_uint(s_ltk_w[4]), __float_as_uint(s_ltk_w[5]));
              }
              if ((n & 0xFFFFFu) == 0xFFFFFu)
                printf("[KS2CHK] checks=%u selbad=%u wbad=%u\n", n + 1u, g_ks2chk[1],
                       g_ks2chk[2]);
            }
            __syncthreads();
          }
          if (tid < 128) s_ltk_logit[tid] = ks2_sv0;
          if (tid < 8) {
            s_ltk_sel[tid] = s_vt2_sel[0][tid];
            s_ltk_w[tid] = s_vt2_w[0][tid];
          }
          __syncthreads();
        }
#endif
      } else {
        mpk_local_topk<128, 4>(nullptr, routing_expected);
      }
#else
      mpk_local_topk<128, 4>(nullptr, routing_expected);
#endif
#endif
#ifdef MPK_TOPK_SUB
      if (tid == 0 && MPK_TOPK_ARMED()) {
        unsigned int const _tk4 = (unsigned int)__builtin_amdgcn_s_memrealtime();
#ifdef MPK_TOPK_WSTAMP
        s_tk_acc[0] += (s_tkw[0] - _tk3) * 10;
        s_tk_acc[1] += (s_tkw[1] - _tk3) * 10;
        s_tk_acc[2] += (s_tkw[2] - _tk3) * 10;
        s_tk_acc[3] += (s_tkw[3] - _tk3) * 10;
        (void)_tk4;
#else
        s_tk_acc[0] += (_tk1 - _tk0) * 10;
        s_tk_acc[1] += (_tk2 - _tk1) * 10;
        s_tk_acc[2] += (_tk3 - _tk2) * 10;
        s_tk_acc[3] += (_tk4 - _tk3) * 10;
#endif
#ifdef MPK_TOPK_SUB2
        s_tk2_acc[0] += (_tkg1 - _tk1) * 10;
        s_tk2_acc[1] += (_tk2 - _tkg1) * 10;
        s_tk2_acc[2] += (_tkx[0] - _tk3) * 10;
        s_tk2_acc[3] += (_tkx[1] - _tkx[0]) * 10;
        s_tk2_acc[4] += (_tkx[2] - _tkx[1]) * 10;
        s_tk2_acc[5] += (_tk4 - _tkx[2]) * 10;
        s_tk2_acc[6] += _tkr;
#endif
        s_tk_n++;
      }
#endif
#if defined(MPK_VT2) && !defined(MPK_VT2_KSEL2)
      if constexpr (QKV_BATCH_SIZE == 2) {
        __syncthreads();  // MPK_LTK_SELLOGIT: thread 128 finishes s_ltk_w after the call
        if (tid < 8) {
          s_vt2_sel[0][tid] = s_ltk_sel[tid];
          s_vt2_w[0][tid] = s_ltk_w[tid];
        }
        if (tid < NUM_EXPERTS) {
          s_ltk_logit[tid] = s_lg1[tid];
        }
        __syncthreads();
        mpk_local_topk<128, 4>(nullptr, routing_expected);  // token 1, selection only
        __syncthreads();
        if (tid < 8) {
          s_vt2_sel[1][tid] = s_ltk_sel[tid];
          s_vt2_w[1][tid] = s_ltk_w[tid];
        }
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");  // token 1's row stores
        __syncthreads();
      }
#endif
      (void)rop_ub;
    }
#elif defined(MPK_LTK_EARLY_TAG)
    // Writers per XCD: the router's pq_split election (MAX_ITERS tiles).
    constexpr int ltk_row_iters = (ACTUAL_HIDDEN_DIM / 4 + 255) / 256;
    mpk_local_topk<128, 4>(ltk_tags, routing_expected, ltk_norm,
                           router_tile_n >= ltk_row_iters ? ltk_row_iters : 1);
#else
#ifdef MPK_VT2
    if constexpr (QKV_BATCH_SIZE == 2) {
      // Wave 1 polls token 1's words while wave 0 polls token 0's inside the call below; the call's
      // first barrier waits for both.
      if (threadIdx.x >= 64 && threadIdx.x < 128) {
        mpk_vt2_poll_tags(ltk_tags1, routing_expected, s_vt2_logit);
      }
    }
#endif
    mpk_local_topk<128, 4>(ltk_tags, routing_expected);
#ifdef MPK_VT2
    if constexpr (QKV_BATCH_SIZE == 2) {
      // MPK_LTK_SELLOGIT: thread 128 finishes s_ltk_w after the call's last barrier.
      __syncthreads();
      if (threadIdx.x < 8) {
        s_vt2_sel[0][threadIdx.x] = s_ltk_sel[threadIdx.x];
        s_vt2_w[0][threadIdx.x] = s_ltk_w[threadIdx.x];
      }
      __syncthreads();
#ifndef MPK_VT2_SAMESEL
      if (threadIdx.x < 128) {
        s_ltk_logit[threadIdx.x] = s_vt2_logit[threadIdx.x];
      }
      __syncthreads();
      mpk_local_topk<128, 4>(nullptr, routing_expected);  // selection only
#endif
      __syncthreads();
      if (threadIdx.x < 8) {
        s_vt2_sel[1][threadIdx.x] = s_ltk_sel[threadIdx.x];
        s_vt2_w[1][threadIdx.x] = s_ltk_w[threadIdx.x];
      }
      __syncthreads();
    }
#endif
#endif
#if defined(MPK_MOE_SPLIT) || defined(MPK_MOE_XPAIR)
    routed_expert0 = s_ltk_sel[xcd_id & 3];
#ifdef MPK_VT2
    if constexpr (QKV_BATCH_SIZE == 2) {
      routed_expert0 = s_vt2_sel[0][xcd_id & 3];
      routed_expert1 = s_vt2_sel[1][xcd_id & 3];
    }
#endif
#ifdef MPK_VTN_MOE
    static_assert(MPK_VTN_T == QKV_BATCH_SIZE, "MPK_VTN_MOE: MPK_VTN_T is the pass's token count");
    static_assert(TOPK_K == 4 && 4 * MPK_VTN_T <= 64, "MPK_VTN_MOE: one lane per (token, slot) in wave 0");
#ifdef MPK_VTN_UNIONE
    bool const ue_done = MPK_UNIONE_ON(num_active_tokens < QKV_BATCH_SIZE ? num_active_tokens : QKV_BATCH_SIZE);
#else
    constexpr bool ue_done = false;
#endif
    if (tid < 64 && !ue_done) {
      // Lane (t, k) holds token t's k-th expert. Union slots in first-appearance order, columns in token
      // order: an expert occurs at most once per token, so its column is the number of earlier lanes
      // holding it.
      extern __shared__ char _vtn_smem[];
      MpkVtnTab *const vt = MPK_VTN_TABP(_vtn_smem);
      // Dead tokens of a partly accepted pass stay out of the union (MPK_VTN_BAL round); live columns unchanged.
      bool const vlive = tid < 4 * (num_active_tokens < MPK_VTN_T ? num_active_tokens : MPK_VTN_T);
      int const vt_t = tid >> 2, vt_k = tid & 3;
      int const ve = vlive ? s_vt2_sel[vt_t][vt_k] : -1 - tid;
      int vrep = tid, vcol = 0, vcnt = 0;
#if defined(MPK_VTN_UNIONM)
      {
        // MPK_VTN_UNIONM: the live lanes holding this lane's expert as one mask, from seven ballots of the id bits
        // (live ids 0..127); vrep / vcol / vcnt = lowest lane / lanes below this one / count: the loop's values.
        unsigned long long vm = __ballot(vlive);
#pragma unroll
        for (int b = 0; b < 7; ++b) {
          bool const vb = ((ve >> b) & 1) != 0;
          unsigned long long const mb = __ballot(vlive && vb);
          vm &= vb ? mb : ~mb;
        }
        if (vlive) {
          vrep = (int)__builtin_ctzll(vm);
          vcol = (int)__popcll(vm & ((1ull << tid) - 1ull));
          vcnt = (int)__popcll(vm);
        }
      }
#endif
#if !defined(MPK_VTN_UNIONM) || defined(MPK_VTN_UNIONM_CHECK)
      {
        int lrep = tid, lcol = 0, lcnt = 0;
#pragma unroll
        for (int j = 0; j < 4 * MPK_VTN_T; ++j) {
          bool const veq = __builtin_amdgcn_readlane(ve, j) == ve;
          lrep = (veq && j < lrep) ? j : lrep;
          lcol += (veq && j < tid) ? 1 : 0;
          lcnt += veq ? 1 : 0;
        }
#if defined(MPK_VTN_UNIONM_CHECK)
        if (vlive && (lrep != vrep || lcol != vcol || lcnt != vcnt)) atomicAdd(&g_unionm_chk[1], 1u);
        if (tid == 0) {
          unsigned const c = atomicAdd(&g_unionm_chk[0], 1u) + 1u;
          if ((c & 0x7FFFu) == 0u) printf("[UNIONMCHK] checks=%u bad=%u\n", c, g_unionm_chk[1]);
        }
#else
        vrep = lrep;
        vcol = lcol;
        vcnt = lcnt;
#endif
      }
#endif
      unsigned long long const vfirst = __ballot(vlive && vrep == tid);
#ifdef MPK_VTN_USORT
      // MPK_VTN_USORT: rank among the representatives by (column count desc, first appearance).
      int vu = 0;
#pragma unroll
      for (int j = 0; j < 4 * MPK_VTN_T; ++j) {
        int const cj = __builtin_amdgcn_readlane(vcnt, j);
        bool const rj = (vfirst >> j) & 1ull;
        vu += (rj && (cj > vcnt || (cj == vcnt && j < vrep))) ? 1 : 0;
      }
#else
      int const vu = __popcll(vfirst & ((1ull << vrep) - 1));
#endif
      if (vlive) {
        vt->ct[vu][vcol] = (unsigned char)vt_t;
        vt->cs[vu][vcol] = (unsigned char)vt_k;
        if (vrep == tid) {
          vt->ue[vu] = ve;
          vt->un[vu] = vcnt;
        }
      }
      if (tid == 0) vt->U = (int)__popcll(vfirst);
    }
    if (!ue_done) __syncthreads();
#endif
#ifdef MPK_SEP_PREFETCH
    // MPK_SEP_PREFETCH capture: store this die's routed expert for the next
    // token's prefetch, and count prev-token prediction coverage.
    if (xcd_rank == 0 && tid == 0) {
#ifdef MPK_SEP_ORACLE
      long long const sep_slot = (long long)task_layer_idx * MPK_NUM_XCDS + xcd_id;
      if (sep_slot >= 0 && sep_slot < MPK_SEP_CAP_N) {
        int const sep_prev = g_sep_cap[sep_slot];
        if (sep_prev == routed_expert0) atomicAdd(&g_sep_cov[0], 1ull);
        unsigned long long const t = atomicAdd(&g_sep_cov[1], 1ull) + 1ull;
        if ((t & 16383ull) == 0ull)
          printf("[SEPCOV] hits=%llu total=%llu\n",
                 (unsigned long long)g_sep_cov[0], t);
        g_sep_cap[sep_slot] = routed_expert0;
      }
#else
      int const sep_slot = (task_layer_idx % 36) * MPK_NUM_XCDS + xcd_id;
      int const sep_prev = g_sep_cap[sep_slot];
      if (task_layer_idx >= 36) {
        if (sep_prev == routed_expert0) atomicAdd(&g_sep_cov[0], 1ull);
        unsigned long long const t = atomicAdd(&g_sep_cov[1], 1ull) + 1ull;
        if ((t & 8191ull) == 0ull)
          printf("[SEPCOV] hits=%llu total=%llu\n",
                 (unsigned long long)g_sep_cov[0], t);
      }
      g_sep_cap[sep_slot] = routed_expert0;
#endif
    }
#endif
#else
#ifdef MPK_VT2
    if constexpr (QKV_BATCH_SIZE == 2) {
      routed_expert0 = s_vt2_sel[0][xcd_id >> 1];
      routed_expert1 = s_vt2_sel[1][xcd_id >> 1];
#ifdef MPK_VT2_DUMP
      if (xcd_id == 0 && xcd_rank == 0 && threadIdx.x == 0 && task_layer_idx <= 1) {
        for (int t = 0; t < 2; ++t) {
          unsigned short const *nr = (unsigned short const *)input_ptrs[12] +
                                     (long long)t * oproj_output_stride;
          unsigned ck = 0u;
          for (int i = 0; i < 2880; ++i) ck = ck * 31u + (unsigned)nr[i];
          printf("[VT2SEL] L%d t%d sel %d %d %d %d w %08x %08x %08x %08x norm %08x\n",
                 task_layer_idx, t, s_vt2_sel[t][0], s_vt2_sel[t][1], s_vt2_sel[t][2],
                 s_vt2_sel[t][3], __float_as_uint(s_vt2_w[t][0]),
                 __float_as_uint(s_vt2_w[t][1]), __float_as_uint(s_vt2_w[t][2]),
                 __float_as_uint(s_vt2_w[t][3]), ck);
        }
      }
#endif
    } else {
      routed_expert0 = s_ltk_sel[xcd_id >> 1];
    }
#else
    routed_expert0 = s_ltk_sel[xcd_id >> 1];
#endif
#endif
#ifdef MPK_ROP_DBG_FULL
    if (tid == 0 && xcd_rank == 0 && xcd_id == 0 && routing_expected == (MPK_ROP_DBG_FULL)) {
      for (int r = 0; r < 8; ++r) {
        printf("[RFULL] e%d r%d %04x %04x %04x %04x %04x %04x %04x %04x %04x %04x %04x %04x "
               "%04x %04x %04x %04x\n", routing_expected, r,
               s_ltk_logit[r * 16 + 0], s_ltk_logit[r * 16 + 1], s_ltk_logit[r * 16 + 2],
               s_ltk_logit[r * 16 + 3], s_ltk_logit[r * 16 + 4], s_ltk_logit[r * 16 + 5],
               s_ltk_logit[r * 16 + 6], s_ltk_logit[r * 16 + 7], s_ltk_logit[r * 16 + 8],
               s_ltk_logit[r * 16 + 9], s_ltk_logit[r * 16 + 10], s_ltk_logit[r * 16 + 11],
               s_ltk_logit[r * 16 + 12], s_ltk_logit[r * 16 + 13], s_ltk_logit[r * 16 + 14],
               s_ltk_logit[r * 16 + 15]);
      }
    }
#endif
#ifdef MPK_TOPK_CHECK
    if (tid == 0 && xcd_rank == 0) {
      float v[5] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY};
      for (int e = 0; e < 128; ++e) {
        float const x = __uint_as_float((unsigned)s_ltk_logit[e] << 16);
        if (x > v[4]) {
          int j = 4;
          while (j > 0 && x > v[j - 1]) {
            v[j] = v[j - 1];
            --j;
          }
          v[j] = x;
        }
      }
      int se[4];
      float sv[4];
      for (int k = 0; k < 4; ++k) {
        se[k] = s_ltk_sel[k] & 127;
        sv[k] = __uint_as_float((unsigned)s_ltk_logit[se[k]] << 16);
      }
      for (int a = 0; a < 4; ++a)
        for (int b = a + 1; b < 4; ++b)
          if (sv[b] > sv[a]) {
            float const t = sv[a];
            sv[a] = sv[b];
            sv[b] = t;
          }
      bool const distinct = se[0] != se[1] && se[0] != se[2] && se[0] != se[3] &&
                            se[1] != se[2] && se[1] != se[3] && se[2] != se[3];
      bool const ok = distinct && sv[0] == v[0] && sv[1] == v[1] && sv[2] == v[2] &&
                      sv[3] == v[3];
      atomicAdd(&g_tkc[0], 1ull);
      if (v[0] == v[1] || v[1] == v[2] || v[2] == v[3]) atomicAdd(&g_tkc[1], 1ull);
      if (v[3] == v[4]) atomicAdd(&g_tkc[2], 1ull);
      if (!ok) {
        unsigned long long const nb = atomicAdd(&g_tkc[3], 1ull);
        if (nb < 12)
          printf("[TKCHK] BAD x%d e%d L%d sel %d,%d,%d,%d = %.7g %.7g %.7g %.7g | top %.7g %.7g %.7g "
                 "%.7g %.7g\n", xcd_id, routing_expected, task_layer_idx, se[0], se[1], se[2],
                 se[3], sv[0], sv[1], sv[2], sv[3], v[0], v[1], v[2], v[3], v[4]);
      }
      if (xcd_id == 0 && routing_expected % 2304 == 0)
        printf("[TKCHK] e%d checks %llu tie_in_top4 %llu tie_at_4th %llu bad %llu\n",
               routing_expected, g_tkc[0], g_tkc[1], g_tkc[2], g_tkc[3]);
    }
#endif
#ifdef MPK_ROP_DBG
    if (tid == 0 && xcd_rank == 0 && (xcd_id == 0 || xcd_id == 5) &&
        routing_expected <= (MPK_ROP_DBG)) {
      unsigned ck = 0u;
      for (int e = 0; e < 128; ++e) ck = ck * 31u + (unsigned)s_ltk_logit[e];
      printf("[RDBG] x%d e%d L%d s%d,%d,%d,%d l%04x,%04x,%04x,%04x w%.6e,%.6e,%.6e,%.6e"
             " m%.6e i%.6e c%08x\n",
             xcd_id, routing_expected, task_layer_idx, s_ltk_sel[0], s_ltk_sel[1],
             s_ltk_sel[2], s_ltk_sel[3], (unsigned)s_ltk_logit[s_ltk_sel[0] & 127],
             (unsigned)s_ltk_logit[s_ltk_sel[1] & 127], (unsigned)s_ltk_logit[s_ltk_sel[2] & 127],
             (unsigned)s_ltk_logit[s_ltk_sel[3] & 127], s_ltk_w[0], s_ltk_w[1], s_ltk_w[2],
             s_ltk_w[3], s_ltk_w[4], s_ltk_w[5], ck);
    }
#endif
#elif defined(MPK_EARLY_ROUTING)
    MPK_WS_WAIT_BEGIN(75, routing_expected);
    int _obs;
    int _spins = 0;
    unsigned long long _rec = 0;
#ifdef MPK_AID_SPLIT_ROUTING
    // Poll this AID's replica of the (epoch, expert) record rather than the
    // shared NC line; see MPK_AID_ROUTING_BASE_INTS in mpk_atoms.cuh.
    int *const _rr = mpk_aid_flags_at(routing_ready, xcd_id,
                                      MPK_AID_ROUTING_BASE_INTS);
#else
    int *const _rr = routing_ready;
#endif
#ifdef MPK_NARROW_GATE_POLL
    if (tid == 0)
#endif
      do {
        _rec = MPK_LD_GATE_AID_U64(&_rr[(1 + xcd_id) * 16 + 2]);
        _obs = (int)_rec;
        MPK_WS_WAIT_TICK(_obs, _spins);
        _spins++;
      } while (_obs < routing_expected);
#ifdef MPK_NARROW_GATE_POLL
    __syncthreads();
    _rec = MPK_LD_GATE_AID_U64(&_rr[(1 + xcd_id) * 16 + 2]);
#endif
    routed_expert0 = (int)(_rec >> 32);
#else
#if defined(MPK_ROUTING_NARROW_AID)
#if defined(MPK_ROUTING_LANE_RELEASE) || defined(MPK_AID_SPLIT_ROUTING)
#error "MPK_ROUTING_NARROW_AID replaces the per-XCD routing flags"
#endif
    // Must follow the narrowed publisher: slot 1 of this AID's replica.
    int *my_release = &mpk_aid_flags_at(routing_ready, xcd_id,
                                        MPK_AID_ROUTING_BASE_INTS)[1 * 16];
#elif defined(MPK_AID_SPLIT_ROUTING)
    // Poll this AID's replica instead of the shared NC line; see
    // MPK_AID_ROUTING_BASE_INTS in mpk_atoms.cuh.
    int *my_release = &mpk_aid_flags_at(routing_ready, xcd_id,
                                        MPK_AID_ROUTING_BASE_INTS)
                           [(1 + xcd_id) * 16];
#else
    int *my_release = &routing_ready[(1 + xcd_id) * 16];
#endif
    MPK_WS_WAIT_BEGIN(75, routing_expected);
    int _obs;
    int _spins = 0;
#ifdef MPK_NARROW_GATE_POLL
    // Same one-poller-per-block argument as the Phase 6 gate above; see there.
    // This site has no rendezvous of its own -- the comment above the phase
    // says so -- so the narrowing has to supply one.
    if (tid == 0)
#endif
      while ((_obs = MPK_LD_GATE_AID(my_release)) < routing_expected) {
        MPK_WS_WAIT_TICK(_obs, _spins);
        _spins++;
        __builtin_amdgcn_s_sleep(1);
      }
#ifdef MPK_NARROW_GATE_POLL
    __syncthreads();
#endif
#endif
  }
  // Cross-XCD ACQUIRE for the routing data. Bare `buffer_inv`, deliberately.
  //
  // Read the gfx950 rule first, because it is the opposite of what the
  // mnemonic suggests: an *unscoped* buffer_inv invalidates nothing. It is an
  // architectural NOP. `sc1` invalidates L2; `sc0 sc1` is system scope.
  // The disassembly confirms this is the understood rule elsewhere in the
  // kernel -- 376 of the 396 buffer_inv sites in the built binary already
  // carry `sc0 sc1`; this one was among the few that did not.
  //
  // So this instruction does nothing, and that is fine *here*: everything the
  // poll gates -- active_expert_ids / routing_indices / topk_weight -- is
  // written st_wt (sc0 sc1) straight to HBM, bypassing L2, and needs no
  // invalidate to be seen.
  //
  // What is NOT fine, and was the bs>1 MoE W13 corruption, is that Phase 8
  // also reads rmsnorm_out_moe, which under MPK_ONE_NORM_WRITER is published
  // by one router worker *per XCD* with a plain global_store into that XCD's
  // non-coherent L2. That line needs a real acquire, and this NOP was standing
  // in for one. What that line needs, and what the branch below gives it, is
  // an `sc1` invalidate. Applying it to every worker here is correct -- it
  // takes the bs=2 two-layer dump from 32 bad weight groups to 0/360 -- but
  // costs +0.259 ms/token (2.196 vs 1.937 at ctx 512), because dropping all of
  // L2 also throws away the streamed MXFP4 expert weights Phase 8 is about to
  // read. So it is applied only to the workers that need it.
  //
  // Which workers those are is not a guess. The corrupt tiles were exactly the
  // ranks at or past oproj_topk_tiles_per_xcd -- {23..28} at the shipped
  // geometry -- the same set the Phase 7a' gate above singles out, because
  // those are the workers that never entered Phase 7 and so never touched the
  // norm row on their own die. Every other rank read it while producing it.
  //
  // A cheaper form was tried and does not work: passing NT_LOAD to the W13
  // activation gather. `__builtin_nontemporal_load` emits the `nt` cache-
  // replacement hint, not a coherence scope, and left 30 of 32 groups bad.
  //
  // How the defect was localized, since the shape is worth recognizing again:
  // it hit 100% of the affected weight groups on XCDs 1, 6 and 7 and 0% on the
  // other five, was bit-identical under MPK_MOE_ENTRY_DELAY=2000 (2000 x
  // s_sleep(127) before Phase 8), and was unchanged by making the producer's
  // store write-through. A defect that survives an unbounded consumer delay
  // and a write-through producer, while staying deterministic per XCD, is a
  // consumer-side stale line -- not a race, and not a producer visibility gap.
  //
  // The acquire rmsnorm_out_moe needs is NO LONGER HERE. It moved to the top
  // of the layer, as a single `buffer_inv sc0 sc1` under QKV_BATCH_SIZE > 1.
  // See the long note at that site for why the layer-top form is preferred to
  // scoping this one.
  //
  // The scoped form that used to be here was:
  //
  //   if (QKV_BATCH_SIZE > 1 && xcd_rank >= oproj_topk_tiles_per_xcd)
  //     buffer_inv sc1;   else   buffer_inv;
  //
  // measuring 1.940 ms/token against a 1.937 pre-fix baseline, with the three
  // arms at ctx 512 being: unconditional flush at this site 2.196, selective
  // by rank 1.997, selective + batch-gated 1.940. It was correct (0/360 at
  // bs=2) but only at the site that happened to reproduce.
#if MPK_NUM_XCDS == 4
  // Routing data is published by one XCD and read by all four, two of which
  // are on the far AID. See the attn_out acquire above.
  asm volatile("buffer_inv" MPK_DEV_SC_INV "" ::: "memory");
#else
  asm volatile("buffer_inv" ::: "memory");
#endif

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  unsigned long long _fused_t3 = __builtin_amdgcn_s_memrealtime();
#endif
  MPK_TKOPS_AT(14, tid == 0);
  MPK_PHASE_MARK(_pslot_w, 7);
#ifdef MPK_OPROJ_DATA_POLL
  // The routing this worker just waited on was published after the O-proj
  // phase-2 barrier, so every O-proj tile has loaded this layer's attn_out:
  // put this XCD's slice back to the value the next layer's O-proj polls
  // for. The Phase 9 arrival orders these stores before the next layer.
  static_assert(NUM_Q_PER_KV * HEAD_DIM == 512,
                "MPK_OPROJ_DATA_POLL resets 1 KiB of bf16 per XCD");
  if (xcd_rank == 0 && tid < 64) {
    typedef int __attribute__((ext_vector_type(4))) dp_i32x4_t;
    char *const dp_rst = reinterpret_cast<char *>(output_ptrs[4]) +
                         static_cast<size_t>(xcd_id) * 1024 + tid * 16;
    dp_i32x4_t const dp_ones = {-1, -1, -1, -1};
    asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1"
                 :
                 : "v"(dp_rst), "v"(dp_ones)
                 : "memory");
  }
#endif
#ifdef MPK_ILPER
    if (tid == 0 && s_ilsub_ml >= 0 && s_ilsub_ml < 64) {
      g_ilper[blockIdx.x * 128 + 0 + s_ilsub_ml] = __builtin_amdgcn_s_memrealtime();
    }
#endif
  MPK_ILSTAMP(15, s_ilsub_ml == MPK_ILSUB_L0 + 1);

  // ══════════════════════════════════════════════════════════════════
  // Phase 8: MoE (W13+SwiGLU+W2)
  // ══════════════════════════════════════════════════════════════════
#ifdef MPK_MOE_ENTRY_DELAY
  // Diagnostic only: stall every MoE worker before it reads rmsnorm_out_moe.
  // Separates "reads too early" from "reads the wrong thing". If the bs>1 W13
  // corruption is a race, a delay long enough to let every producer finish
  // makes it go away; if the corrupt tiles survive an unbounded wait, no
  // amount of ordering will fix them and the defect is in what is read, not
  // when. Costs a full stall on the hot path -- never ship this on.
  for (int _d = 0; _d < MPK_MOE_ENTRY_DELAY; _d++) {
    __builtin_amdgcn_s_sleep(127);
  }
#endif
#if defined(MPK_MOE_XCD_PAIR) && !defined(MPK_EARLY_ROUTING) && !defined(MPK_LOCAL_TOPK)
#error "MPK_MOE_XCD_PAIR requires MPK_EARLY_ROUTING or MPK_LOCAL_TOPK (Phase 7b)"
#endif
#ifdef MPK_MOE_XCD_PAIR
  // Two packed tiles per rank 0..22: W13 (bit7=0) then W2 (bit7=1). Ranks
  // 23..30 have no pair-map work; they still join Phase 9.
#if MPK_NUM_XCDS == 4
  constexpr int kMoePairRanks = 46;
#else
  constexpr int kMoePairRanks = 23;
#endif
#if MPK_NUM_XCDS == 8 && !defined(MPK_MULTI_PASS_8XCD)
#ifdef MPK_MOE_W2_REMAP
#ifdef MPK_W2_L2_WARM
#error "MPK_MOE_W2_REMAP gives ranks 23..30 W2 groups; MPK_W2_L2_WARM uses them as warmers"
#endif
#if defined(MPK_MOE_W2_REMAP2) && (defined(MPK_QKV_PF_SKIP_W2) || defined(MPK_ABLATE_KV_TILES))
#error "MPK_MOE_W2_REMAP2 leaves ranks 8 and 9 W13-only; the skip-W2 prefetch arms assume they run W2"
#endif
  // Ranks 0..7: W13 only. Ranks 8..22: W13 then W2. Ranks 23..30: W2 group
  // (rank - 23) only, entered right after the TopK.
  constexpr int kW2Moved = 8;
#ifdef MPK_MOE_W2_REMAP3
// MPK_QKV_PF_SKIP_W2 is allowed: ranks 8-9 (the K/V QKV producers) still
// run a W2 group here, which is all its `xcd_rank < 8` test assumes.
#if defined(MPK_MOE_W2_REMAP2) || defined(MPK_ABLATE_KV_TILES) || \
    defined(MPK_W2_L2_WARM)
#error "MPK_MOE_W2_REMAP3 replaces the REMAP rank map; incompatible with this flag"
#endif
  // Ranks 0-7: W13 tile r. Ranks 8-9: W2 group r only, entered with the
  // movers' delay. Ranks 10-22: W13 tile r + W2 group r. Ranks R, R+1
  // (MPK_MOE_W2_REMAP3_R0): W13 tile 8 / 9 + W2 group r-23. Other ranks
  // 23-30: W2 group r-23 (delayed).
#ifndef MPK_MOE_W2_REMAP3_R0
#define MPK_MOE_W2_REMAP3_R0 23
#endif
  constexpr int kMr3R0 = MPK_MOE_W2_REMAP3_R0;
  static_assert(kMr3R0 >= kMoePairRanks && kMr3R0 + 1 < kMoePairRanks + kW2Moved,
                "MPK_MOE_W2_REMAP3_R0 must name two of ranks 23-30");
  bool const mr3_w13_mover = xcd_rank == kMr3R0 || xcd_rank == kMr3R0 + 1;
  bool const mr3_w2_only =
      xcd_rank == 8 || xcd_rank == 9 ||
      (xcd_rank >= kMoePairRanks && xcd_rank < kMoePairRanks + kW2Moved &&
       !mr3_w13_mover);
#ifdef MPK_MOE_W2_REMAP3_NODELAY89
  // Only the movers take the entry delay; ranks 8-9 start at once.
  bool const moe_w2_mover = mr3_w2_only && xcd_rank >= kMoePairRanks;
#else
  bool const moe_w2_mover = mr3_w2_only;
#endif
  bool const mr3_w13 =
      (xcd_rank < kMoePairRanks && !mr3_w2_only) || mr3_w13_mover;
  bool const mr3_w2 = xcd_rank >= 8 && xcd_rank < kMoePairRanks + kW2Moved;
  int const moe_begin = mr3_w13 ? 0 : 1;
  int const moe_end = mr3_w2 ? 2 : 1;
  int const mr3_w13_phase = mr3_w13_mover ? 8 + (xcd_rank - kMr3R0) : xcd_rank;
  int const mr3_w2_phase =
      xcd_rank < kMoePairRanks ? xcd_rank : xcd_rank - kMoePairRanks;
#else
  bool const moe_w2_mover = xcd_rank >= kMoePairRanks &&
                            xcd_rank < kMoePairRanks + kW2Moved;
  int const moe_begin = xcd_rank < kMoePairRanks ? 0 : 1;
#ifdef MPK_MOE_W2_REMAP2
  // Ranks 8 and 9 hand W2 groups 8 and 9 to movers 23 and 24, which run them
  // after their own group; every QKV rank is then W13-only.
  constexpr int kW2Extra = 2;
  int const moe_end =
      xcd_rank < kW2Moved + kW2Extra ? 1
      : xcd_rank < kMoePairRanks     ? 2
      : moe_w2_mover ? (xcd_rank - kMoePairRanks < kW2Extra ? 3 : 2)
                     : 1;
#else
  int const moe_end = xcd_rank < kW2Moved         ? 1
                      : xcd_rank < kMoePairRanks ? 2
                      : moe_w2_mover             ? 2
                                                 : 1;
#endif
  int const moe_phase =
      xcd_rank < kMoePairRanks ? xcd_rank : xcd_rank - kMoePairRanks;
#endif
#else
  int const moe_begin = 0;
  int const moe_end = (xcd_rank < kMoePairRanks) ? 2 : 0;
  int const moe_phase = xcd_rank;
#endif
  int const moe_step = 1;
#elif defined(MPK_MOE_W2_REMAP)
#error "MPK_MOE_W2_REMAP is wired for the 8-XCD single-pass pair map"
#endif
#else
  int const moe_begin = xcd_rank;
  int const moe_end = moe_total_tiles_per_xcd;
  int const moe_step = workers_per_xcd;
#endif
#ifdef MPK_W2_L2_WARM
#ifndef MPK_MOE_XCD_PAIR
#error "MPK_W2_L2_WARM is wired for the MPK_MOE_XCD_PAIR tile map"
#endif
#ifdef MPK_W2_WARM_NT
#define MPK_W2_WARM_MOD "sc0 nt"
#else
#define MPK_W2_WARM_MOD ""
#endif
  if (xcd_rank >= kMoePairRanks) {
    // This XCD's MoE ranks stream W2 groups (xcd_id & 1) * 23 + [0, 23) of
    // the expert routed to this XCD pair; each idle rank touches every 128 B
    // line of an eighth of that range so their W2 weight DMA hits this XCD's
    // L2. Same group geometry as gang_moe_fused_mxfp4_kernel_mi300's W2.
    constexpr int kW2Wg = MOE_W2_OUTPUT_PER_WG * (MOE_INTERMEDIATE_SIZE / 2) +
                          MOE_W2_OUTPUT_PER_WG * (MOE_INTERMEDIATE_SIZE / 32);
    constexpr int64_t kW2Expert =
        static_cast<int64_t>(MOE_HIDDEN_SIZE / MOE_W2_OUTPUT_PER_WG) * kW2Wg;
    constexpr int kWarmers = 8;
    constexpr int kSpan = kMoePairRanks * kW2Wg;
    constexpr int kShare = (kSpan + kWarmers - 1) / kWarmers;
    static_assert((kShare + 256 * 128 - 1) / (256 * 128) == 9,
                  "the warm sequence below hardcodes .rept 8");
    int const warm_j = xcd_rank - kMoePairRanks;
    if (warm_j < kWarmers) {
#if defined(MPK_W2_L2_WARM_DELAY_US) && MPK_W2_L2_WARM_DELAY_US > 0
      unsigned long long const warm_t0 = __builtin_amdgcn_s_memrealtime();
      while (__builtin_amdgcn_s_memrealtime() - warm_t0 <
             100ull * MPK_W2_L2_WARM_DELAY_US) {
        __builtin_amdgcn_s_sleep(8);
      }
#endif
      int const warm_lo = warm_j * kShare;
      int const warm_n = warm_lo + kShare <= kSpan ? kShare : kSpan - warm_lo;
      uint8_t const *warm_base =
          static_cast<uint8_t const *>(input_ptrs[18]) +
          static_cast<int64_t>(routed_expert0) * kW2Expert +
          static_cast<int64_t>((xcd_id & 1) * kMoePairRanks) * kW2Wg + warm_lo;
      // Out-of-range offsets read zero (num_records = warm_n).
      i32x4_t const warm_rsrc =
          make_w_buffer_rsrc(warm_base, static_cast<uint32_t>(warm_n));
      uint32_t warm_voff = static_cast<uint32_t>(tid) * 128u;
      unsigned warm_sink;
      asm volatile("buffer_load_dword %[d], %[voff], %[rsrc], 0 offen " MPK_W2_WARM_MOD "\n"
                   ".rept 8\n"
                   "v_add_u32_e32 %[voff], 0x8000, %[voff]\n"
                   "buffer_load_dword %[d], %[voff], %[rsrc], 0 offen " MPK_W2_WARM_MOD "\n"
                   ".endr\n"
                   "s_waitcnt vmcnt(0)\n"
                   : [d] "=&v"(warm_sink), [voff] "+v"(warm_voff)
                   : [rsrc] "s"(warm_rsrc)
                   : "memory");
    }
  }
#endif
#ifdef MPK_MOE_XCD_PAIR
  // 46 W13 tiles/expert (W13_TILES) must arrive before W2. At 4 XCDs there
  // is no partner die, so this XCD owns all 46 pair-ranks with 31 workers:
  // finish every leftover W13 (moe_i==0) before any W2 (moe_i==1).
#if MPK_NUM_XCDS == 8 && !defined(MPK_MULTI_PASS_8XCD)
#ifdef MPK_VT2
#if defined(MPK_MOE_W2_REMAP) || defined(MPK_MOE_W2_REMAP2) || defined(MPK_MOE_W2_REMAP3)
#error "MPK_VT2: the 8-XCD pair loop is wired for the plain map (no W2 remap)"
#endif
  // W13 of both tokens, then W2 of both: token 0's W2 wait overlaps token 1's W13 stream.
  constexpr int vt2_nt = QKV_BATCH_SIZE == 2 ? 2 : 1;
#ifdef MPK_VT2_SCHED3
  // 92 calls per XCD (2 tokens x 23 W13 + 23 W2 groups) over all 31 ranks, at most 3 each, W13
  // before W2 on every rank: r0 W13 t0 (ranks 0-22) / W13 t1 0-7 (23-30); r1 W13 t1 8-22 (0-14) /
  // W2 t0 0-15 (15-30); r2 W2 t0 16-22 (0-6) / W2 t1 0-22 (7-29).
  static_assert(QKV_BATCH_SIZE != 2 || kMoePairRanks == 23,
                "MPK_VT2_SCHED3: 23 groups per pair member");
#ifdef MPK_VTN_MOE
  {
    // This die's share of every AID half: groups G = 4 p + die (expert-major), W13 then W2, round-robin over
    // its ranks (W13 before W2 on every rank, as SCHED3).
    extern __shared__ char _vtn_smem2[];
    MpkVtnTab const *const vt = MPK_VTN_TABP(_vtn_smem2);
    int const vU = vt->U;
    int const vx = xcd_id & 3;
    int const vn = (23 * vU - vx + 3) / 4;
    int vcall = 0;
    int const vW = workers_per_xcd;
#ifdef MPK_VTN_TKDEF
    bool vtk_done = false;
#endif
#ifdef MPK_VTN_BAL
    // W13 tiles round-robin; W2 tiles by availability (a W13 call ~ 2 W2 calls): ranks [vA, W) are free at step
    // 2 vb, ranks [0, vA) (one W13 more) at 2 vb + 2, and at each step every free rank takes the next W2 tile.
#ifdef MPK_VTN_BAL_CT
    constexpr bool vbal = true;  // MPK_VTN_BAL_CT: no per-layer constant load at MoE entry
#else
    bool const vbal = g_mpk_vtnbal_rt != 0;
#endif
    int const vA = vn % vW, vb = vn / vW, vnA = vW - vA;
    int const vt0 = 2 * vb, vt1 = vt0 + 2;
    int vtt = xcd_rank >= vA ? vt0 : vt1;
#else
    constexpr bool vbal = false;
    int const vA = 0, vnA = 0, vt0 = 0, vt1 = 0;
    int vtt = 0;
#endif
#ifdef MPK_VTN_DYN
    // MPK_VTN_DYN: rank r runs item r, then item 31 + t for each ticket t its calls take, until past 2 vn.
    constexpr bool vdyn = true;
    int const vN = 2 * vn;
    int vcur = xcd_rank;
    int const *const vtks = (int const *)(_vtn_smem2 + MPK_VTN_TKS_OFF);
    int const vlay = *reinterpret_cast<int const *>(
        &_vtn_smem2[mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE -
                    mirage::runtime::LAYER_IDX_SMEM_OFFSET_FROM_END]);
    int *const vq_ctr = g_aid_flag_rep[xcd_id >> 2] + MPK_VTN_Q_INTS + xcd_id * 64 + (vlay & 1);
#else
    constexpr bool vdyn = false;
#endif
#ifdef MPK_VTN_MOETL
    unsigned long long *const vtl_s = (unsigned long long *)(_vtn_smem2 + MPK_VTN_TLS_OFF);
    if (tid == 0) {
      for (int c = 0; c < MPK_VTN_TLC; ++c) vtl_s[3 * c + 2] = 0ull;
    }
#endif
    for (int vk = 0;; ++vk) {
#ifdef MPK_VTN_DYN
      (void)vbal; (void)vA; (void)vnA; (void)vt0; (void)vt1; (void)vtt; (void)vW;
      if (vk != 0) {
        __syncthreads();  // the previous call's ticket, written by its thread 0
        int const vtk = vtks[0];
        if (tid == 0 && vtk == vN - 1) {
          asm volatile("global_atomic_swap %0, %1, off" ::"v"(vq_ctr), "v"(0) : "memory");
        }
        vcur = 31 + vtk;
      }
      if (vcur >= vN) break;
      int const vw2 = vcur >= vn ? 1 : 0;
      int const vitem = vw2 ? vcur - vn : vcur;
#else
      int const vi = xcd_rank + vk * vW;
      int vw2, vitem;
      if (vi < vn) {
        vw2 = 0;
        vitem = vi;
      } else if (!vbal) {
        if (vi >= 2 * vn) break;
        vw2 = 1;
        vitem = vi - vn;
      } else {
        int const vj2 = vtt < vt1 ? (vtt - vt0) * vnA + (xcd_rank - vA)
                                  : (vt1 - vt0) * vnA + (vtt - vt1) * vW + xcd_rank;
        if (vj2 >= vn) break;
        ++vtt;
        vw2 = 1;
        vitem = vj2;
      }
#endif
      int const vg = 4 * vitem + vx;
      int const vu = vg / 23;
      int const vj = vg - vu * 23;
      int const ve = vt->ue[vu];
#ifdef MPK_VTN_TKDEF
      if (vw2 != 0 && !vtk_done) {
        // MPK_VTN_TKDEF: this rank's W2 epilogues read s_vt2_w; Phase 7b left its logits and keys.
        vtk_done = true;
        mpk_vtn_tkdef_tail<QKV_BATCH_SIZE>(_vtn_smem2, tid);
        __syncthreads();
      }
#endif
      if (vcall++ != 0 && !vdyn) __syncthreads();
#ifdef MPK_VTN_STAG_NS
#ifndef MPK_VTN_STAG_MOD
#define MPK_VTN_STAG_MOD 4
#endif
#ifndef MPK_VTN_STAG_DIE
#define MPK_VTN_STAG_DIE 1
#endif
      if (vcall == 1) {
        // MPK_VTN_STAG_NS: offset this rank's first call so the per-call non-streaming phases of the
        // GPU's 248 workgroups do not all fall in the same microsecond.
        int const vst = (xcd_rank + MPK_VTN_STAG_DIE * (xcd_id & 3)) % MPK_VTN_STAG_MOD;
        unsigned long long const vs_t0 = __builtin_amdgcn_s_memrealtime();
#if defined(MPK_VTN_STAG_QUAD)
        (void)vst;
        unsigned long long const vs_lim =
            (unsigned long long)xcd_rank * (unsigned long long)xcd_rank * (MPK_VTN_STAG_NS) / 9000ull;
#elif defined(MPK_VTN_STAG_LIN2)
        (void)vst;
        unsigned long long const vs_lim =
            (unsigned long long)((xcd_rank * 8 + xcd_id) * (MPK_VTN_STAG_NS / 248) / 10);
#elif defined(MPK_VTN_STAG_LIN)
        (void)vst;
        unsigned long long const vs_lim = (unsigned long long)(xcd_rank * (MPK_VTN_STAG_NS / 31) / 10);
#else
        unsigned long long const vs_lim =
            (unsigned long long)(vst * (MPK_VTN_STAG_NS / MPK_VTN_STAG_MOD) / 10);
#endif
#if defined(MPK_VTN_STAG_W13U)
        // MPK_VTN_STAG_W13U=UREF: a rank whose first item is a W13 group waits delay * min(vU, UREF) / UREF and,
        // when scaled, leaves the routing-weight tail to its first W2 call (that call's release wait hides it).
        // W2-first ranks keep the full delay: their fragment loads would otherwise compete with the W13 streams.
        bool const vs_scaled = vw2 == 0 && vU < (MPK_VTN_STAG_W13U);
        unsigned long long const vs_eff =
            vs_scaled ? vs_lim * (unsigned long long)vU / (unsigned long long)(MPK_VTN_STAG_W13U) : vs_lim;
#elif defined(MPK_VTN_STAG_U)
        // MPK_VTN_STAG_U=UREF: the delay scales with the pass's union size, delay * min(vU, UREF) / UREF.
        unsigned long long const vs_eff =
            vU >= (MPK_VTN_STAG_U) ? vs_lim : vs_lim * (unsigned long long)vU / (unsigned long long)(MPK_VTN_STAG_U);
        constexpr bool vs_scaled = false;
#else
        unsigned long long const vs_eff = vs_lim;
        constexpr bool vs_scaled = false;
#endif
#ifndef MPK_VTN_STAG_TK_MIN
#define MPK_VTN_STAG_TK_MIN 0
#endif
#if defined(MPK_VTN_STAG_TK) && defined(MPK_VTN_TKDEF)
        if (xcd_rank >= MPK_VTN_STAG_TK && !vtk_done && !vs_scaled &&
            vs_eff >= (unsigned long long)(MPK_VTN_STAG_TK_MIN)) {
          // MPK_VTN_STAG_TK: the routing-weight tail runs inside this rank's stagger delay (still timed from
          // vs_t0) instead of right before its first W2 call.
          vtk_done = true;
          mpk_vtn_tkdef_tail<QKV_BATCH_SIZE>(_vtn_smem2, tid);
          __syncthreads();
        }
#endif
        (void)vs_scaled;
        while (__builtin_amdgcn_s_memrealtime() - vs_t0 < vs_eff) {
          __builtin_amdgcn_s_sleep(2);
        }
      }
#endif
#ifdef MPK_VTN_MOETL
      unsigned long long const vtl_t0 = __builtin_amdgcn_s_memrealtime();
#endif
      gang_moe_fused_mxfp4_kernel_mi300<1, MOE_INTERMEDIATE_SIZE, MOE_HIDDEN_SIZE, NUM_EXPERTS, TOPK_K,
                                        MOE_W13_OUTPUT_PER_WG, MOE_W2_OUTPUT_PER_WG>(
          input_ptrs[12], input_ptrs[17], input_ptrs[18], output_ptrs[7], output_ptrs[8], input_ptrs[19],
          input_ptrs[20], output_ptrs[9], input_ptrs[22], output_ptrs[10], input_ptrs[21],
          vj | (vw2 << 7) | ((ve + 1) << 8) | (vu << 16) | (vdyn ? (1 << 24) : 0));
#ifdef MPK_VTN_MOETL
      if (tid == 0 && vcall <= MPK_VTN_TLC) {
        vtl_s[3 * (vcall - 1) + 0] = vtl_t0;
        vtl_s[3 * (vcall - 1) + 1] = __builtin_amdgcn_s_memrealtime();
        vtl_s[3 * (vcall - 1) + 2] = (1ull << 63) | (unsigned long long)(unsigned)(
            vw2 | (vj << 1) | (vu << 6) | ((ve & 255) << 11) | (vitem << 19) | ((vt->un[vu] & 15) << 27));
      }
#endif
    }
  }
#define MPK_VTN_S3N 0
#else
#define MPK_VTN_S3N MPK_VT2_NROUNDS
#endif
  for (int vt2_i = 0; vt2_i < (QKV_BATCH_SIZE == 2 ? MPK_VTN_S3N : 0); ++vt2_i) {
#else
  for (int vt2_i = moe_begin * vt2_nt; vt2_i < moe_end * vt2_nt; ++vt2_i) {
#endif
// Token-major: a W2 call depends on per-worker state its own W13 call leaves behind, so an
// interleaved W13(t0) W13(t1) W2(t0) W2(t1) order computes token 0's W2 with token 1's state.
#if defined(MPK_VT2_SCHED3)
#elif !defined(MPK_VT2_INTERLEAVE)
    int const moe_i = vt2_nt == 2 ? (vt2_i & 1) : vt2_i;
    int const vt2_tok = vt2_nt == 2 ? (vt2_i >> 1) : 0;
#else
    int const moe_i = vt2_i / vt2_nt;
    int const vt2_tok = vt2_i % vt2_nt;
#endif
    // Back-to-back MoE calls with no Phase 9 between them: a fast wave would start the next
    // call's LDS weight DMA while a slower wave still reads this call's fragments.
#ifdef MPK_VT2_SCHED3
    int const vr = xcd_rank;
#ifdef MPK_VT2_SCHED4
    int const vt2_code = vr < 31 ? mpk_vt2_sched4(vt2_i, vr) : -1;
    int const moe_i = vt2_code < 0 ? 0 : (vt2_code >> 8);
    int const vt2_tok = vt2_code < 0 ? 0 : ((vt2_code >> 7) & 1);
    int const vt2_tile = vt2_code < 0 ? -1 : (vt2_code & 127);
#else
    int const moe_i = vt2_i == 0 ? 0 : (vt2_i == 1 ? (vr <= 14 ? 0 : 1) : 1);
    int const vt2_tok = vt2_i == 0 ? (vr <= 22 ? 0 : 1)
                        : (vt2_i == 1 ? (vr <= 14 ? 1 : 0) : (vr <= 6 ? 0 : 1));
    int const vt2_tile = vt2_i == 0 ? (vr <= 22 ? vr : vr - 23)
                         : (vt2_i == 1 ? (vr <= 14 ? 8 + vr : vr - 15)
                                       : (vr <= 6 ? 16 + vr : (vr <= 29 ? vr - 7 : -1)));
#endif
    if (vr >= 31 || vt2_tile < 0) continue;
#ifdef MPK_VT2_T1ROW_LATE
#if !defined(MPK_VT2_KSEL2) || defined(MPK_VT2_SCHED4)
#error "MPK_VT2_T1ROW_LATE: SCHED3 order (one token-1 W13 call per rank) with the KSEL2 tail"
#endif
    // Waves 2-3 stored token 1's FP8 row in Phase 7b; this call's waves 0-2 load it.
    bool const t1row_wait = g_mpk_t1late_rt != 0 && moe_i == 0 && vt2_tok == 1;
    if (t1row_wait) asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    if (vt2_i != 0 || t1row_wait) __syncthreads();
#else
    if (vt2_i != 0) __syncthreads();
#endif
#ifdef MPK_VT2_QKV_EARLY
#if !defined(MPK_QKV_W_TOP) || !defined(MPK_VT2_QKV) || !defined(MPK_VT2_QKV_PF_AUX) || \
    defined(MPK_VT2_SCHED4) || defined(MPK_QKV_KSPLIT) || defined(MPK_QKV_PF_WAVE_SPLIT)
#error "MPK_VT2_QKV_EARLY: SCHED3 + VT2_QKV with the plain Phase 9 prefetch and MPK_QKV_W_TOP"
#endif
    // Round 2 is every QKV rank's last MoE call (a W2): the next layer's QKV chunks above the W2
    // window go out now; the callee's entry wait absorbs them inside the W2 release-poll slack.
    if (g_mpk_qkve_rt != 0 && vt2_i == 2 && moe_i == 1 && qkv_does_qkv &&
        input_ptrs[24] != nullptr) {
      qkv_prefetch_weights_lds<QKV_TB, QKV_OUTPUT_PER_WG, QKV_REDUCTION_SIZE,
                               MPK_VT2_QKV_PF_AUX, MPK_KVES_LO>(
          input_ptrs[24], qkv_n_wgs_per_xcd, qkv_pf_tile);
    }
#endif
#ifdef MPK_VT2_W2DELAY
    // Round 1's W2 (ranks 15-30) issues its weight fragments while ranks 0-14 stream token 1's W13:
    // hold it back g_mpk_vt2_w2d ns (runtime, env MPK_VT2_W2D).
    if (vt2_i == 1 && moe_i == 1 && g_mpk_vt2_w2d > 0) {
      unsigned long long const w2d0 = __builtin_amdgcn_s_memrealtime();
      while (__builtin_amdgcn_s_memrealtime() - w2d0 < (unsigned long long)(g_mpk_vt2_w2d / 10)) {
        __builtin_amdgcn_s_sleep(8);
      }
    }
#endif
#else
    int const vt2_tile = moe_phase;
    if (vt2_i != moe_begin * vt2_nt) __syncthreads();
#endif
    {
#else
  for (int moe_i = moe_begin; moe_i < moe_end; moe_i += moe_step) {
    {
#endif
#if defined(MPK_MOE_W2_REMAP3)
    int const moe_t = (moe_i == 0 ? mr3_w13_phase : mr3_w2_phase) |
                      (moe_i << 7) | ((routed_expert0 + 1) << 8) |
                      MPK_W2K_MOVER_BIT(moe_i == 1 && moe_w2_mover)
#ifdef MPK_W2_EARLY
                      // W13+W2 ranks: the W13 call loads the W2 tile, the W2 call finds it.
                      | ((mr3_w13 && mr3_w2)
                             ? (moe_i == 0 ? ((1 << 21) | (mr3_w2_phase << 24)) : (1 << 22))
                             : 0)
#endif
        ;
#elif defined(MPK_MOE_W2_REMAP2)
    int const moe_t =
        moe_i == 2
            ? (kW2Moved + xcd_rank - kMoePairRanks) | (1 << 7) |
                  ((routed_expert0 + 1) << 8)
            : moe_phase | (moe_i << 7) | ((routed_expert0 + 1) << 8);
#elif defined(MPK_VT2)
    int const moe_t = vt2_tile | (moe_i << 7) |
                      (((vt2_tok ? routed_expert1 : routed_expert0) + 1) << 8) |
                      (vt2_tok << 23);
#else
    int const moe_t =
        moe_phase | (moe_i << 7) | ((routed_expert0 + 1) << 8);
#endif
#if defined(MPK_MOE_W2_REMAP) && defined(MPK_MOE_W2_REMAP_DELAY_US) && \
    MPK_MOE_W2_REMAP_DELAY_US > 0
#ifdef MPK_MOE_W2_REMAP2
#ifdef MPK_MOE_W2_REMAP2_NODELAY
    if (moe_w2_mover && moe_i == 1 && xcd_rank - kMoePairRanks >= kW2Extra) {
#else
    if (moe_w2_mover && moe_i == 1) {
#endif
#else
    if (moe_w2_mover) {
#endif
      unsigned long long const w2d_t0 = __builtin_amdgcn_s_memrealtime();
#ifdef MPK_W2_MALLPF
      if (g_mpk_w2pf_mode != 0u) {
        while (__builtin_amdgcn_s_memrealtime() - w2d_t0 < (unsigned long long)g_mpk_w2pf_ticks) {
          __builtin_amdgcn_s_sleep(1);
        }
        mpk_w2_mallpf<MOE_INTERMEDIATE_SIZE, MOE_W2_OUTPUT_PER_WG, NUM_EXPERTS>(
            input_ptrs[18], routed_expert0, xcd_id, xcd_rank, tid);
      }
#endif
#ifdef MPK_KV_EARLY_STAGE
#if !defined(MPK_QKV_W_TOP) || !defined(MPK_MOE_W2_REMAP3) || \
    defined(MPK_MOE_W2_REMAP3_NODELAY89) || defined(MPK_QKV_KSPLIT) || \
    defined(MPK_KV_SPREAD) || defined(MPK_QKV_PF_SKIP_W2) || \
    defined(MPK_QKV_PF_ORACLE89) || defined(MPK_QKV_KV_PF_PLAIN) || \
    defined(MPK_QKV_KV_PF_SC0) || defined(MPK_QKV_L2PRE) || \
    defined(MPK_GATE_DEFER_PF_LATE) || defined(MPK_QKV_PF_EARLY) || \
    defined(MPK_QKV_PF_WAVE_SPLIT) || defined(MPK_QKV_PF_ABLATE) || \
    defined(MPK_QKV_KV_L2WARM)
#error "MPK_KV_EARLY_STAGE: ranks 8-9 must be delayed W2-only movers with the plain Phase 9 prefetch"
#endif
      // K/V ranks: the next layer's weight chunks above the W2 window go out now,
      // while they wait; Phase 9 issues the rest.
      if (qkv_does_qkv && g_mpk_kves != 0 && input_ptrs[24] != nullptr) {
        qkv_prefetch_weights_lds<QKV_TB, QKV_OUTPUT_PER_WG,
                                 QKV_REDUCTION_SIZE, 3, MPK_KVES_LO>(
            input_ptrs[24], qkv_n_wgs_per_xcd, qkv_pf_tile);
      }
#endif
#ifdef MPK_QKV_KV_L2WARM
      // The K/V QKV ranks pull next layer's weight record into L2 while they
      // wait; their Phase 9 prefetch then reads it with a plain DMA.
      if (qkv_does_qkv && mpk_warm_qkvw != nullptr) {
        qkv_warm_weights_l2<QKV_BATCH_SIZE, QKV_OUTPUT_PER_WG,
                            QKV_REDUCTION_SIZE>(
            mpk_warm_qkvw, qkv_n_wgs_per_xcd, qkv_pf_tile);
      }
#endif
#ifdef MPK_D89_RT
      unsigned long long const w2d_lim =
          (xcd_rank == 8 || xcd_rank == 9)
              ? (unsigned long long)g_mpk_d89_ticks
#ifdef MPK_MOE_W2_REMAP_DELAY_NS
              : (unsigned long long)(MPK_MOE_W2_REMAP_DELAY_NS / 10);
#else
              : 100ull * MPK_MOE_W2_REMAP_DELAY_US;
#endif
      while (__builtin_amdgcn_s_memrealtime() - w2d_t0 < w2d_lim) {
#else
      while (__builtin_amdgcn_s_memrealtime() - w2d_t0 <
#ifdef MPK_MOE_W2_REMAP_DELAY_NS
             (unsigned long long)(MPK_MOE_W2_REMAP_DELAY_NS / 10)) {
#else
             100ull * MPK_MOE_W2_REMAP_DELAY_US) {
#endif
#endif
        __builtin_amdgcn_s_sleep(8);
      }
    }
#endif
#else
#ifdef MPK_VT2
  // W13 of both tokens, then W2 of both: token 0's W2 wait overlaps token 1's W13 stream.
  constexpr int vt2_nt = QKV_BATCH_SIZE == 2 ? 2 : 1;
  for (int moe_i = 0; moe_i < 2 * vt2_nt; ++moe_i) {
    for (int pair = xcd_rank; pair < kMoePairRanks; pair += workers_per_xcd) {
      int const vt2_tok = vt2_nt == 2 ? (moe_i & 1) : 0;
      int const vt2_e = vt2_tok ? routed_expert1 : routed_expert0;
#ifdef MPK_VT2_SKIP1
      if (vt2_tok == 1) continue;
#endif
      int const moe_t = pair | ((moe_i / vt2_nt) << 7) | ((vt2_e + 1) << 8) |
                        (vt2_tok << 23);
#else
  for (int moe_i = 0; moe_i < 2; ++moe_i) {
    for (int pair = xcd_rank; pair < kMoePairRanks; pair += workers_per_xcd) {
      int const moe_t =
          pair | (moe_i << 7) | ((routed_expert0 + 1) << 8);
#endif
#endif
#else
  for (int moe_i = moe_begin; moe_i < moe_end; moe_i += moe_step) {
      int const moe_t = moe_i;
#endif
    MPK_TW_SUB(80, moe_t);
    MPK_WS_PHASE(80, qkv_epoch_expected, xcd_id);
#ifdef MPK_MOENORM_AID
    // Read the normed-row copy homed in this XCD's own range (see
    // MPK_MOENORM_AID in gang_linear_mxfp4_res_bias_rmsnorm_topk_mi300.cuh).
    void const *const moe_norm_in =
        (void const *)((unsigned short const *)input_ptrs[12] +
                       ((xcd_id >= (MPK_NUM_XCDS / 2))
                            ? (long long)QKV_BATCH_SIZE * oproj_output_stride
                            : 0));
#else
    void const *const moe_norm_in = input_ptrs[12];
#endif
#ifdef MPK_MOE_CALL_LDS
#if !defined(MPK_MOE_LDS) || !defined(MPK_MOE_XCD_PAIR)
#error "MPK_MOE_CALL_LDS needs MPK_MOE_LDS and MPK_MOE_XCD_PAIR"
#endif
    unsigned long long const _mc0 = __builtin_amdgcn_s_memrealtime();
#endif
#if !defined(MPK_ONLY_OP) || (MPK_ONLY_OP & (1 << 8))
#ifdef MPK_VT2
    // A 2-token pass runs the batch-1 MoE once per token (tile index bit 23).
#ifdef MPK_ROUTER_OPROJ_PARTIAL
    // Token 0's row is in LDS (Phase 7b); token 1's in this worker's private slot.
    void const *const vt2_in =
        ((moe_t >> 23) & 1)
            ? (void const *)(g_aid_nc_rep[xcd_id >> 2] + MPK_NC_ROP1ROW_INTS +
                             ((xcd_id & 3) * 32 + xcd_rank) * 1024)
            : moe_norm_in;
#else
    void const *const vt2_in = (void const *)((unsigned short const *)moe_norm_in +
                                              (long long)((moe_t >> 23) & 1) *
                                                  oproj_output_stride);
#endif
    void *const vt2_ws = (void *)((float *)output_ptrs[10] +
                                  (long long)((moe_t >> 23) & 1) * MOE_WS_SLOTS *
                                      MOE_HIDDEN_SIZE);
#define MPK_VT2_MOE_BS 1
#define MPK_VT2_IN vt2_in
#define MPK_VT2_WS vt2_ws
#else
#define MPK_VT2_MOE_BS QKV_BATCH_SIZE
#define MPK_VT2_IN moe_norm_in
#define MPK_VT2_WS output_ptrs[10]
#endif
    gang_moe_fused_mxfp4_kernel_mi300<MPK_VT2_MOE_BS,
                                      MOE_INTERMEDIATE_SIZE,
                                      MOE_HIDDEN_SIZE,
                                      NUM_EXPERTS,
                                      TOPK_K,
                                      MOE_W13_OUTPUT_PER_WG,
                                      MOE_W2_OUTPUT_PER_WG>(MPK_MOE_HOF_IN(MPK_VT2_IN),
                                                            input_ptrs[17],
                                                            input_ptrs[18],
                                                            output_ptrs[7],
                                                            output_ptrs[8],
                                                            input_ptrs[19],
                                                            input_ptrs[20],
                                                            output_ptrs[9],
                                                            input_ptrs[22],
                                                            MPK_VT2_WS,
                                                            input_ptrs[21],
#if defined(MPK_EARLY_ROUTING) && !defined(MPK_MOE_XCD_PAIR)
                                                            moe_t |
                                                                ((routed_expert0 +
                                                                  1)
                                                                 << 8)
#else
                                                            moe_t
#endif
#ifdef MPK_MOE_XCD_STRIPE_LAYER
                                                            ,
                                                            layer_counter
#endif
#ifdef MPK_EARLY_ROUTING
                                                            ,
                                                            routing_ready,
                                                            routing_expected
#endif
    );
#endif
#ifdef MPK_MOE_CALL_LDS
    {
      unsigned long long const _mc1 = __builtin_amdgcn_s_memrealtime();
      if (tid == 0 && MPK_MOE_ARMED()) {
        s_moe_call[moe_i] += (unsigned int)((_mc1 - _mc0) * 10);
        s_moe_call[2 + moe_i] += 1;
      }
    }
#endif
    if (moe_i == 1) {
      MPK_W2STAMP(7);
    }
  }
#ifdef MPK_MOE_XCD_PAIR
  } // W13-all then W2-all
#endif
  MPK_PHASE_MARK(_pslot_w, 8);
#if defined(MPK_VTN_MOETL) && defined(MPK_VTN_MOE)
  if (tid == 0) {
    unsigned int const vtl_l =
        s_phase_layers - (unsigned int)(MPK_PHASE_SNAP * MPK_PHASE_LAYERS_PER_ITER);
    if (vtl_l < (unsigned int)MPK_PHASE_LAYERS_PER_ITER) {
      extern __shared__ char _vtn_smem3[];
      unsigned long long const *const vtl_r = (unsigned long long const *)(_vtn_smem3 + MPK_VTN_TLS_OFF);
      unsigned long long *const vtl_g =
          MPK_VTN_TL_RD + ((size_t)vtl_l * MPK_PHASE_MAX_WORKERS + _pslot_w) * (MPK_VTN_TLC * 3);
      for (int c = 0; c < 3 * MPK_VTN_TLC; ++c) {
        __hip_atomic_store(&vtl_g[c], vtl_r[c], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      }
    }
  }
#endif
  if (tid == 0) MPK_GS_STAMP(0);
  MPK_WS_PHASE(90, qkv_epoch_expected, xcd_id);
  MPK_ILSTAMP(16, s_ilsub_ml == MPK_ILSUB_L0 + 1);

  // ══════════════════════════════════════════════════════════════════
  // Phase 9: Layer-boundary GLOBAL barrier
  //
  // The next layer's QKV prologue (ResAddF32) reads moe_workspace_f32 -- the
  // buffer this layer's W2 epilogue just wrote. moe_ws_layout.cuh guarantees
  // every (token, slot, hidden) element has exactly one writer, so there is no
  // accumulator race *within* a layer. What was missing is the guarantee that
  // the writer has run *at all* before the next layer reads.
  //
  // Nothing provided it. All 36 layers run inside one persistent dispatch (the
  // `ml` loop in persistent_kernel.cuh), and the only thing between them is:
  //
  //     if (MPK_TIDX_F == 0 && block_xcd_local_rank == 0) threadfence_gpu();
  //     __syncthreads();
  //
  // Both are per-workgroup. `__syncthreads` synchronizes the 256 threads of
  // *this* block and says nothing about the other 239 workers, and
  // `threadfence_gpu` is a memory *ordering* fence, not a rendezvous -- it
  // makes this worker's prior writes visible if someone reads them later, but
  // it never makes anyone wait. So a workgroup that finished its MoE tiles
  // could roll straight into layer N+1 and read workspace slots belonging to
  // tokens whose W2 tiles were still executing on another XCD.
  //
  // The result is not a hang and not garbage: the reader gets the *previous*
  // layer's value for those elements (the buffer is deliberately never zeroed
  // -- coverage is total, so nothing clears it), which is a plausible-looking
  // float. Which elements lose the race depends on inter-XCD timing, so it
  // varies run to run. That is precisely the observed signature: layer 0's K/V
  // is bit-identical across runs because on the first layer the workspace has
  // just been zeroed and there is no prior value to read, while layer 1 -- the
  // first layer with a real predecessor -- diverges at decode step 0.
  //
  // The barrier is a two-level rendezvous over the same counter block every
  // other phase here uses: each XCD elects one worker to publish its arrival,
  // then everyone waits for all 8 XCDs to reach this layer.
  //
  // The release target is a snapshot-free function of task_layer_idx, which
  // persistent_kernel.cuh computes as (pc_iter - 1) * num_layers + ml --
  // monotonic across the whole run and identical on every worker, so there is
  // nothing to race and a relaunch cannot wait on a value already in the past.
  //
  // Cost is one cross-XCD rendezvous per layer. That is the price of the
  // dependency actually being there; the previous code was fast because it
  // simply did not wait.
#ifndef MPK_NO_LAYER_BARRIER
  {
    // Who joins the release wait. Default: everyone, a full gang rendezvous.
    // Under MPK_W2_CONSUMER_GATE: only the workers that read
    // moe_workspace_f32 in the next layer's Phase 1, which is the same
    // `qkv_attn_rank < total_qkv_tiles_per_xcd` set that guards the reader
    // itself (qkv_attn_rank is fixed for the life of the task, so this worker
    // is the one that will do that read). Everyone still *arrives* either way
    // -- the arrival tree is what publishes the release, so narrowing it would
    // deadlock; this narrows only the wait.
#if defined(MPK_W2_CONSUMER_GATE) &&                                           \
    (MPK_NUM_XCDS != 4 || defined(MPK_FORCE_NARROW_LAYER_GATE))
    // MPK_FORCE_NARROW_LAYER_GATE restores the 8-XCD narrow wait at 4 XCDs.
    //
    // Worth trying because this gate is the single most expensive phase in
    // DPX+NPS2: slot 10 measures 15.4 us/layer in NPS1 against 134.1 in NPS2,
    // an 8.7x blowup, and the arm below multiplies an already-expensive NC
    // poll by all 31 workers per XCD instead of just the QKV ranks. The
    // hazard the `#else` documents is real, so this is opt-in and has to be
    // checked against the known-good text, not just against the clock.
#ifdef MPK_GATE_ATTN_JOIN
#if defined(MPK_ROTATE_QKV_ATTN_RANKS) || defined(MPK_QKV_KSPLIT)
#error "MPK_GATE_ATTN_JOIN widens the waiter set; extend this branch's arrival count first"
#endif
    // The premise below ("what keeps the non-consumers in line is Phase 6")
    // holds only for workers with no Phase 1-5 work. Once NUM_KV_CHUNKS
    // exceeds total_qkv_tiles_per_xcd, ranks [T, NUM_KV_CHUNKS) run attention
    // chunks in the next layer without having waited here, and long prompts
    // at 24-31 chunks decode differently on every run (1k prompt, 31 chunks:
    // 0/16 tokens match torch; with this, 16/16 and bit-identical reruns).
    // At NUM_KV_CHUNKS <= T the waiter set is unchanged.
    bool const MPK_LAYER_GATE_JOINS =
        qkv_does_qkv || (attn_rank < ATTN_PARTICIPANTS);
#else
    bool const MPK_LAYER_GATE_JOINS = qkv_does_qkv;
#endif
#else
    // 4 XCDs: every rank produces at least one W2 tile (46 groups / 31
    // workers). CONSUMER_GATE would let ranks 20-30 skip Phase 9 and start
    // the next layer's W13 on the same arrival counter the leftover W2 is
    // still waiting on.
    bool const MPK_LAYER_GATE_JOINS = true;
#endif

    // Two-level rendezvous. Slot layout, one 16-int line each:
    //   layer_local[x]   : arrivals from XCD x's own workers_per_xcd workers
    //   layer_global[0]  : one arrival per XCD, bumped by that XCD's last
    //   layer_release[x] : release flag for XCD x, broadcast by the last XCD
    //
    // The tree is the whole point. A flat barrier -- 240 workgroups each
    // bumping one shared counter and then polling eight remote lines -- is
    // correct but measures ~1.0 ms/iter over 36 layers on this machine,
    // because every one of those 240 atomics is a cross-die read-modify-write
    // serialising on a single cache line, and the poll adds 240 waves x 8
    // cache-bypassing loads of lines homed on other dies. Phase 6's barrier
    // is cheap precisely because it only ever puts 8 * NUM_REQS atomics on
    // its shared line. This gets to the same 8 by aggregating per-XCD first,
    // and borrows Phase 6's arrive-then-broadcast release so no worker polls
    // a line owned by another die.
    int *layer_local =
        oproj_counters_base + FULL_LAYER_LAYER_BARRIER_SLOT(NUM_REQS);
    int *layer_global = layer_local + 8 * 16;
#if defined(MPK_P9_FLAT) &&                                                    \
    (MPK_NUM_XCDS != 8 || !defined(MPK_LEAN_ARRIVE) ||                         \
     !defined(MPK_AID_SPLIT_FLAGS) || defined(MPK_LAYER_NARROW_REL))
#error "MPK_P9_FLAT needs 8 XCDs, MPK_LEAN_ARRIVE, MPK_AID_SPLIT_FLAGS and no MPK_LAYER_NARROW_REL"
#endif
#if defined(MPK_AID_P9_NC) && (!defined(MPK_P9_FLAT) || !defined(MPK_AID_NC_REP))
#error "MPK_AID_P9_NC needs MPK_P9_FLAT and MPK_AID_NC_REP"
#endif
#if defined(MPK_AID_P9_LOCAL) && !defined(MPK_P9_FLAT)
#error "MPK_AID_P9_LOCAL needs MPK_P9_FLAT (epochs derived from the per-launch counter)"
#endif
    int *layer_release_shared = layer_global + 16;
#ifdef MPK_AID_SPLIT_FLAGS
    // Fan-out shape: one XCD publishes all eight slots, rather than each XCD
    // publishing its own. That touches eight far-AID lines from a single die
    // at once, which is what overflows the CC directory -- so this one is only
    // safe on RW replicas. `layer_global`, which the epoch is derived from,
    // stays shared: it is an sc1 atomic, and those are XCD-L2-bound rather
    // than AID-bound, so homing it buys nothing.
    int *layer_release = mpk_aid_flags(
        layer_release_shared, xcd_id, MPK_AID_REGION_LAYER_RELEASE);
#else
    int *layer_release = layer_release_shared;
#endif

    // ── Who *arrives* ────────────────────────────────────────────────────
    //
    // Default: all workers_per_xcd. The barrier exists to order this layer's
    // W2 stores against the next layer's read of moe_workspace_f32, and only
    // the workers that ran a W2 tile have any such store -- so the rest are
    // reporting the retirement of nothing, while the release cannot fire
    // until they do. At the shipped geometry that is 8 of 31 workers per XCD
    // whose arrival is pure latency: moe_total_tiles_per_xcd is 53, the loop
    // strides by workers_per_xcd, so ranks 0..21 get a second tile (global
    // tile >= TOTAL_W13, i.e. a W2 tile) and ranks 22..30 get only their
    // first, a W13 tile.
    //
    // MPK_W2_ONLY_ARRIVE narrows the arrival to exactly those producers,
    // guarding on `xcd_rank < moe_total_tiles_per_xcd - workers_per_xcd`.
    //
    // The non-producers stay ordered for the same reason the non-consumers
    // do under MPK_W2_CONSUMER_GATE: they cannot outrun the next layer's
    // Phase 6, whose release requires every XCD's attention chunks, which
    // requires QKV, which requires this release. What they lose is only the
    // right to *delay* it.
    //
    // The count must be computed exactly as Phase 8's loop bound is, or the
    // local counter's modulus stops being exact and the release never fires.
    //
    // `moe_total_tiles_per_xcd - workers_per_xcd` counts the ranks that get a
    // SECOND tile, which is a producer count only while the loop deals at most
    // two tiles per rank. Past that it exceeds the worker count and becomes a
    // modulus no counter can reach: at bs=4 the tile space is 212 per XCD
    // against 31 workers, so the unclamped form asks for 181 arrivals from a
    // counter that advances 31 per layer, and every layer gate deadlocks. The
    // clamp is also the semantically right answer -- once there are at least
    // as many W2 tiles as workers, every worker is a producer and there is
    // nobody left to exclude, so this degenerates to the default arrival.
#ifdef MPK_MOE_XCD_PAIR
    // Pair map: ranks 0..22 each produce one W2 tile (23 groups × 2 XCDs).
#if MPK_NUM_XCDS == 8 && !defined(MPK_MULTI_PASS_8XCD)
    constexpr int n_w2_workers_per_xcd = kMoePairRanks;
#else
    int const n_w2_workers_per_xcd =
        kMoePairRanks > workers_per_xcd ? workers_per_xcd : kMoePairRanks;
#endif
#else
    int const n_w2_workers_per_xcd =
        moe_total_tiles_per_xcd > workers_per_xcd
            ? moe_total_tiles_per_xcd - workers_per_xcd
            : moe_total_tiles_per_xcd;
#endif
#ifdef MPK_W2_ONLY_ARRIVE
    // ── Every waiter must also arrive ────────────────────────────────────
    //
    // `s_layer_rel_prev` is written only on the arrival path, so a worker
    // that *waits* below without *arriving* still holds the -1 seed. -1 is
    // beaten by the release already sitting in the line -- last layer's --
    // so its gate clears immediately and it runs the next layer's Phase 1
    // against a moe_workspace_f32 this layer's W2 has not published yet.
    // Silent, run-to-run varying wrong numerics, not a hang.
    //
    // The seed comment above names one configuration that trips this
    // (MPK_W2_CONSUMER_GATE=0, where JOINS is unconditionally true while
    // only the W2 producers arrive) and calls it a hang; the -1 seed
    // converted it into this race. MPK_ROTATE_QKV_ATTN_RANKS trips it too,
    // and worse: it moves the waiter set to a cyclic block of ranks chosen
    // *because* they have no Phase 7 work, and those are very nearly the
    // same ranks that get no second MoE tile and therefore do not arrive.
    // Measured at the shipped geometry, rotation on: three runs, three
    // different completions, against a control that is bit-identical over
    // eight.
    //
    // So arrive on the union of "W2 producer" and "gate waiter" rather than
    // on the producers alone, and move the modulus with it. The waiter set
    // is a cyclic block, so the union is a prefix plus at most two pieces
    // and its size is closed-form -- it has to be exact, or the local
    // counter's modulus stops firing and the release never comes.
    int const n_w2_arrivers = n_w2_workers_per_xcd < workers_per_xcd
                                  ? n_w2_workers_per_xcd
                                  : workers_per_xcd;
#if defined(MPK_MOE_W2_REMAP) || defined(MPK_VT2_SCHED3)
    // W2 producers are ranks 8..30 and the gate waiters ranks 0..9: the
    // union is everybody. MPK_VT2_SCHED3 hands W2 groups to all 31 ranks.
    int const arrivers_per_xcd = workers_per_xcd;
    bool const MPK_LAYER_GATE_ARRIVES = true;
#elif !defined(MPK_W2_CONSUMER_GATE)
    // JOINS is `true` here, so the union is everybody.
    int const arrivers_per_xcd = workers_per_xcd;
    bool const MPK_LAYER_GATE_ARRIVES = true;
#elif defined(MPK_ROTATE_QKV_ATTN_RANKS)
    // qkv_attn_rank = (r + qkv_rot) % W, so the waiters
    // `qkv_attn_rank < total_qkv_tiles_per_xcd` are exactly
    // r in [W - qkv_rot, W - qkv_rot + T) taken mod W.
    int const wait_b = (workers_per_xcd - qkv_rot) % workers_per_xcd;
    int const wait_n = qkv_work_slots < workers_per_xcd
                           ? qkv_work_slots
                           : workers_per_xcd;
    int const wait_hi = wait_b + wait_n; // may run past workers_per_xcd
    // Part of the block at or above the producer prefix, before the wrap...
    int const seg_lo = wait_b > n_w2_arrivers ? wait_b : n_w2_arrivers;
    int const seg_hi = wait_hi < workers_per_xcd ? wait_hi : workers_per_xcd;
    int const seg_head = seg_hi > seg_lo ? seg_hi - seg_lo : 0;
    // ...and the wrapped tail [0, wrap), likewise only where it is not
    // already counted by the prefix.
    int const wrap = wait_hi > workers_per_xcd ? wait_hi - workers_per_xcd : 0;
    int const seg_tail = wrap > n_w2_arrivers ? wrap - n_w2_arrivers : 0;
    int const arrivers_per_xcd = n_w2_arrivers + seg_head + seg_tail;
    bool const MPK_LAYER_GATE_ARRIVES =
        (xcd_rank < n_w2_arrivers) || MPK_LAYER_GATE_JOINS;
#else
#ifdef MPK_QKV_KSPLIT
    // Waiters are [0, T) U [W-T, W). Union with W2 producers [0, n_w2).
    // At T=10, W=31, n_w2=22 this is 31: every rank arrives. The last
    // arriver is still a W2 producer; the extra ranks arrive early.
    int const extra_hi = qkv_k1_base >= n_w2_arrivers
                             ? (workers_per_xcd - qkv_k1_base)
                             : (workers_per_xcd - n_w2_arrivers);
    int const arrivers_per_xcd = n_w2_arrivers + extra_hi;
    bool const MPK_LAYER_GATE_ARRIVES =
        (xcd_rank < n_w2_arrivers) || MPK_LAYER_GATE_JOINS;
#else
    // Unrotated: the waiters are the prefix [0, T), which the producer
    // prefix already covers at every shipped geometry.
#ifdef MPK_GATE_ATTN_JOIN
    int const n_waiters_want = qkv_work_slots > ATTN_RANK_SPAN
                                   ? qkv_work_slots
                                   : ATTN_RANK_SPAN;
    int const n_waiters_unrot = n_waiters_want < workers_per_xcd
                                    ? n_waiters_want
                                    : workers_per_xcd;
#else
    int const n_waiters_unrot = qkv_work_slots < workers_per_xcd
                                    ? qkv_work_slots
                                    : workers_per_xcd;
#endif
    int const arrivers_per_xcd =
        n_w2_arrivers > n_waiters_unrot ? n_w2_arrivers : n_waiters_unrot;
    bool const MPK_LAYER_GATE_ARRIVES = (xcd_rank < arrivers_per_xcd);
#endif
#endif
#else
    int const arrivers_per_xcd = workers_per_xcd;
    bool const MPK_LAYER_GATE_ARRIVES = true;
#endif
    (void)n_w2_workers_per_xcd;

    // Every worker must retire its own MoE stores before its arrival is
    // published: vmcnt is per-wave, so neither the release on the atomic nor
    // s_barrier covers the other waves. Drain, then barrier, then arrive --
    // the same ordering the other release sites in this file use.
    //
    // MPK_DRAIN_OVERLAP moves that drain to *after* the arrival, so the store
    // retirement overlaps the barrier spin instead of preceding it. It is a
    // measurement arm, not a shipping mode: see the correctness note at the
    // post-arrival site for why it may not be enabled by default.
    //
    // TESTED AND REJECTED: dropping this pair on the grounds that Phase 8
    // already ends in the same two instructions is *wrong*, and wrong in the
    // silent way. Under it one run in eight produced different generated text
    // at identical flags. The argument fails on the W13 arm: its arrival
    // block runs after Phase 8's `__syncthreads()`, and the last arrival's
    // eight-slot `st_wt_u32` release fan-out is a store issued past it -- so
    // the block is neither drained nor reconverged on that path. The
    // `goto w13_arrive` empty-slot path skips Phase 8's drain outright. The
    // pair stays unconditional.
#ifdef MPK_DRAIN_STATS
    unsigned long long _dr0 = __builtin_amdgcn_s_memrealtime();
#endif
#ifndef MPK_DRAIN_OVERLAP
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
#ifdef MPK_DRAIN_STATS
    unsigned long long _dr1 = __builtin_amdgcn_s_memrealtime();
#endif
    __syncthreads();
#if defined(MPK_PREFETCH_NEXT_QKV) && defined(MPK_QKV_PF_EARLY)
#if defined(MPK_QKV_PF_WAVE_SPLIT) || defined(MPK_QKV_KSPLIT) || \
    defined(MPK_DRAIN_OVERLAP)
#error "MPK_QKV_PF_EARLY is wired for the unsplit prefetch behind the drain"
#endif
    // The next layer's QKV tile, staged before the arrival rather than after
    // it. Waves 1..3 issue all four stripes (wave 1 also wave 0's), so wave 0,
    // whose lane 0 runs the arrival atomics and their vmcnt(0) waits, has none
    // of the DMA outstanding.
    if (input_ptrs[24] != nullptr && qkv_does_qkv && tid >= 64) {
      qkv_prefetch_weights_lds<QKV_TB,
                               QKV_OUTPUT_PER_WG,
                               QKV_REDUCTION_SIZE>(
          input_ptrs[24], qkv_n_wgs_per_xcd, qkv_pf_tile, 0, -1, tid >> 6);
      if ((tid >> 6) == 1) {
        qkv_prefetch_weights_lds<QKV_TB,
                                 QKV_OUTPUT_PER_WG,
                                 QKV_REDUCTION_SIZE>(
            input_ptrs[24], qkv_n_wgs_per_xcd, qkv_pf_tile, 0, -1, 0);
      }
    }
#endif
#ifdef MPK_DRAIN_STATS
    unsigned long long _dr2 = __builtin_amdgcn_s_memrealtime();
#endif

    // Carries "I am the releaser, publish epoch N" from lane 0 out to lanes
    // 1..7 of the same wave; see the fan-out below. Zero means not the
    // releaser, which is unambiguous since a real epoch is >= 8.
    int lean_rel_epoch = 0;

    // Thread-local, not __shared__, and that is the whole point. Every write
    // (the seed below, the snapshot, the MPK_LEAN_ARRIVE derivation) and the
    // only read (the gate poll) is guarded by `tid == 0`, so no thread but
    // lane 0 of wave 0 ever touches it -- LDS was buying nothing.
    //
    // What it cost is visible in the asm. As a __shared__ int the compiler
    // cannot hoist it out of the spin, so the poll loop was
    //
    //     global_load_dword v3, v[0:1], off sc0 sc1
    //     s_waitcnt vmcnt(0)
    //     ds_read_b32 v4, v2 offset:336      <-- reloaded every iteration
    //     s_waitcnt lgkmcnt(0)               <-- and serialized behind it
    //     v_cmp_le_i32 vcc, v3, v4
    //
    // an LDS round trip appended to each poll, after the vmcnt wait, on the
    // one thread whose latency to observe the release is the layer boundary
    // for its entire workgroup. Holding the previous release in a register
    // instead removes the round trip: the compare is then just the load and
    // the cmp.
    //
    // The `__syncthreads()` after the seed is kept: it is a barrier the block
    // needs on this path regardless, and dropping it is a separate change.
    //
    // MPK_GATE_PREV_LDS=1 restores the __shared__ form for A/B.
#ifdef MPK_GATE_PREV_LDS
    __shared__ int s_layer_rel_prev;
    if (tid == 0) {
      s_layer_rel_prev = -1;
    }
#else
    int s_layer_rel_prev = -1;
#endif
#ifdef MPK_ML_TABLE_INLAYER
    // Layer ml+1's row from this AID's table copy, loaded by wave 1:
    // wave 0 carries the arrival, whose release drain would wait for
    // these. Typed AS(1), so no FLAT access disturbs wait counting.
    mlt_v = 0;
    {
      int const mlt_i = tid - 64;
      if (mlt_go && mlt_i >= 0 && mlt_i <= MPK_MLT_NIN + MPK_MLT_NOUT) {
        int const row = xcd_id * MPK_MLT_LAYERS + mlt_ml + 1;
        auto const *in_t =
            (__attribute__((address_space(1))) unsigned long long const *)
                g_aid_ml_in[xcd_id >> 2];
        auto const *out_t =
            (__attribute__((address_space(1))) unsigned long long const *)
                g_aid_ml_out[xcd_id >> 2];
        if (mlt_i < MPK_MLT_NIN) {
          mlt_v = in_t[row * MPK_MLT_NIN + mlt_i];
        } else if (mlt_i < MPK_MLT_NIN + MPK_MLT_NOUT) {
          mlt_v = out_t[row * MPK_MLT_NOUT + (mlt_i - MPK_MLT_NIN)];
        } else if (mlt_ml + 2 < MPK_MLT_LAYERS) {
          mlt_v = in_t[(row + 1) * MPK_MLT_NIN + 4];  // slot 24
        }
      }
    }
#endif
    // Seeded for the workers that do NOT arrive. `s_layer_rel_prev` is written
    // only under `tid == 0 && MPK_LAYER_GATE_ARRIVES`, but the wait below is
    // joined by every worker for which MPK_LAYER_GATE_JOINS holds. Those two
    // sets are equal in the two shipped configurations (both flags on, or both
    // off), which is why this never bit -- but MPK_W2_ONLY_ARRIVE=1 with
    // MPK_W2_CONSUMER_GATE=0 makes JOINS a strict superset of ARRIVES, and
    // ranks 22..30 then spin against uninitialized LDS. Whatever garbage the
    // slot holds is compared `<=` against a release value that only ever
    // increases by 8 per layer, so a large positive leaves the gate closed
    // forever: all 248 workers hang with tasks_done=0 and no error.
    //
    // -1 is below every real release value (they are 8*L, L >= 1), so under
    // MPK_GATE_PREV_LDS a non-arriving worker reads a defined value instead of
    // whatever the slot held.
    //
    // It is NOT, as this comment used to claim, "the correct semantics for a
    // worker that published nothing". A non-arriving *waiter* is beaten by the
    // release already in the line -- last layer's -- so its gate clears
    // without waiting for this layer's release at all, and it reads
    // moe_workspace_f32 in the next layer's Phase 1 before this layer's W2
    // published it. The seed turned a hang into a silent race. What makes
    // the seed harmless is the arriver set above now covering the waiter set
    // by construction, so no waiter reaches this line unseeded; see the
    // union computation at MPK_LAYER_GATE_ARRIVES for the two configurations
    // that used to break it. The seed stays as the belt to that suspenders.
    //
    // Cost is one LDS store per workgroup per layer on a path that already
    // does several.
    __syncthreads();
#ifdef MPK_P9_KV_EARLY_DMA
#if !defined(MPK_PREFETCH_NEXT_QKV) || !defined(MPK_PFIN_RETIRE) || defined(MPK_QKV_KSPLIT) || \
    defined(MPK_KV_SPREAD) || defined(MPK_QKV_PF_SKIP_W2) || defined(MPK_QKV_PF_WAVE_SPLIT) || \
    defined(MPK_QKV_PF_EARLY) || defined(MPK_GATE_DEFER_PF_LATE) || defined(MPK_QKV_KV_PF_PLAIN) || \
    defined(MPK_QKV_KV_PF_SC0) || defined(MPK_QKV_L2PRE) || defined(MPK_KV_EARLY_STAGE)
#error "MPK_P9_KV_EARLY_DMA: plain Phase 9 prefetch with MPK_PFIN_RETIRE only"
#endif
    // K/V ranks: waves 1-3 issue their weight stripes beside wave 0's arrival round trip.
    if (qkv_does_qkv && qkv_attn_rank >= 8 && input_ptrs[24] != nullptr && (tid >> 6) != 0) {
      __builtin_amdgcn_s_waitcnt(0x0F70);  // vmcnt(0): MPK_PFIN_RETIRE's retire point
      qkv_prefetch_weights_lds<QKV_TB, QKV_OUTPUT_PER_WG, QKV_REDUCTION_SIZE>(
          input_ptrs[24], qkv_n_wgs_per_xcd, qkv_pf_tile);
    }
#endif
#ifdef MPK_DRAIN_STATS
    __shared__ unsigned long long s_dr3;
    __shared__ bool s_was_last_local;
    __shared__ bool s_was_last_global;
    if (tid == 0) {
      s_was_last_global = false;
    }
    __syncthreads();
#endif

    if (tid == 0 && MPK_LAYER_GATE_ARRIVES) {
      // Snapshot this XCD's release line *before* arriving. Layer L's release
      // is written only after all 240 workers arrive, this one included, so
      // at this instant the line still holds layer L-1's value -- and it
      // holds at least that, because this worker cleared layer L-1's release
      // to get here. So `rel_prev` is exactly layer L-1's value on every
      // worker, and "layer L has released" is exactly "the line moved".
      //
      // This is why the wait target is a snapshot rather than a function of
      // task_layer_idx: this counter block is a host allocation that is never
      // reset, while task_layer_idx restarts at 0 on every kernel launch
      // (pc_iter is __shared__). A layer-derived target would make every
      // relaunch after a warmup wait on a value already in the past and hang
      // all 240 workers. Same reasoning as the attn_global bump above.
      //
      // MPK_LEAN_ARRIVE removes this load, and the `local_expected` load
      // below, from the critical path. Both are inferable from the value the
      // arrival atomic already returns, so the arrival becomes a single
      // read-modify-write with nothing in front of it -- see the derivation
      // at the post-atomic site.
#ifndef MPK_LEAN_ARRIVE
      s_layer_rel_prev = ld_nt_s32(&layer_release[xcd_id * 16]);
#endif

      // The two arrival targets are division snapshots for the same reason,
      // and are race-free because the barrier each belongs to bounds it.
      // Division, never `(v | mask) + 1`: workers_per_xcd is 30, not a power
      // of two, so the bitwise form would release early.
      //
      // For the local counter: every worker on this XCD has cleared layer
      // L-1's release (so v >= workers_per_xcd * L) and none can reach layer
      // L+1's arrival without layer L's release, which needs this very
      // arrival (so v < workers_per_xcd * (L+1)). For the global counter the
      // same argument runs one level up, over the 8 XCDs.
#ifndef MPK_LEAN_ARRIVE
      int local_expected =
          (__atomic_load_n(&layer_local[xcd_id * 16], __ATOMIC_RELAXED) /
               arrivers_per_xcd +
           1) *
          arrivers_per_xcd;
#endif
      // XCD-private: written and read only at `[xcd_id * 16]`. This is the
      // hottest arrival in the layer -- 30 workers per XCD, 240 per GPU, every
      // layer -- and under the default `sc1` every one of them is a cross-die
      // round trip to a coherency point no other XCD reads through.
      // `layer_global` below is the level that *is* cross-XCD and keeps its
      // `sc0 sc1`, so the two-level tree still publishes exactly as before.
      //
      // The drain that makes this arrival a release is the explicit
      // `s_waitcnt vmcnt(0)` above, not the atomic's scope, so weakening the
      // atomic does not weaken the release: the XCD leader elected here still
      // executes `atom_add_release_gpu_s32(layer_global)` after every one of
      // its XCD's workers has both drained and arrived.
      MPK_ATSP9(10, true);
#ifdef MPK_P9_ARRREC
      unsigned long long const p9r_t0 = __builtin_amdgcn_s_memrealtime();
#endif
#if defined(MPK_AID_P9_LOCAL)
      // In the replica of this die's own AID (physical XCC id), past the MoE
      // flat barrier's region; one 256 B line per die so no two dies share an
      // L2 line (RW lines are coherent inside the AID). Cleared per launch.
      int *const p9_rep = p9_rep_base;
      int local_prev = MPK_XCD_LOCAL_ATOM_ADD(
          p9_rep ? p9_rep + 29696 + xcd_id * 64 : &layer_local[xcd_id * 16], 1);
#elif defined(MPK_AID_P9_NC)
      // The die's AID half of the NC scratch (physical XCC id), 256 B apart.
      int *const p9n = p9n_base;
      int local_prev = MPK_XCD_LOCAL_ATOM_ADD(
          p9n ? p9n + MPK_NC_P9_INTS + xcd_id * 64 : &layer_local[xcd_id * 16], 1);
#else
      int local_prev =
          MPK_XCD_LOCAL_ATOM_ADD(&layer_local[xcd_id * 16], 1);
#endif
#ifdef MPK_P9_ARRREC
      {
        if (local_prev == 0x7fffffff) {
          __builtin_amdgcn_s_sleep(0);
        }
        unsigned long long const p9r_dt = __builtin_amdgcn_s_memrealtime() - p9r_t0;
        int p9r_xcc;
        MPK_GET_XCC_ID16(p9r_xcc);
        p9r_xcc &= 7;
        atomicAdd(&g_p9_arrrec[p9r_xcc][1], p9r_dt);
        unsigned long long const p9r_n = atomicAdd(&g_p9_arrrec[p9r_xcc][0], 1ULL) + 1;
        if (p9r_n == 20000) {
          printf("[P9ARR] die=%d arrivals=%llu arrive_ns=%llu\n", p9r_xcc, p9r_n,
                 g_p9_arrrec[p9r_xcc][1] * 10 / p9r_n);
        }
      }
#endif

#ifdef MPK_LEAN_ARRIVE
      // ── Everything the two removed loads carried, recovered from local_prev
      //
      // Both counters are monotonic and never reset, and the layer barrier
      // bounds them: no worker reaches layer L+1's arrival without layer L's
      // release, which needs every layer-L arrival. So when this worker
      // arrives at layer L the local counter holds
      // `(L-1) * workers_per_xcd + p` with `0 <= p < workers_per_xcd`, and
      // one integer division of the value the atomic already returned splits
      // it into both halves that were previously loaded separately:
      //
      //   p == workers_per_xcd - 1        <=>  this is the XCD's last arrival
      //   local_prev / workers_per_xcd    ==   L - 1
      //
      // The release value written for layer L is `8 * L` (the global counter
      // takes exactly one arrival per XCD per layer), so the previous
      // release -- what the wait below must beat -- is `8 * (L - 1)`. That is
      // still derived from the counter block rather than from
      // task_layer_idx, so it keeps the relaunch-safety the snapshot had:
      // task_layer_idx restarts at 0 on every dispatch, this does not.
      //
      // Net effect: the arrival is one read-modify-write with no dependent
      // load in front of it.
      MPK_ATSP9(11, true);
      int const lean_layers_done = local_prev / arrivers_per_xcd;
      s_layer_rel_prev = MPK_NUM_XCDS * lean_layers_done;
      bool const lean_is_last_local =
          (local_prev - lean_layers_done * arrivers_per_xcd) ==
          arrivers_per_xcd - 1;
#ifdef MPK_DRAIN_STATS
      s_was_last_local = lean_is_last_local;
#endif
      if (lean_is_last_local) {
#if defined(MPK_P9_PREINV) && !defined(MPK_P9_PREINV_SKIP)
        // Every worker on this die has arrived, and none reads the
        // workspace or residual again before the release, so the L2
        // invalidate the gate consumers need can happen here, while other
        // dies are still arriving. Drained before the arrival advertises it.
        asm volatile("buffer_inv" MPK_DEV_SC_INV "\n s_waitcnt vmcnt(0)" ::: "memory");
#endif
#ifdef MPK_P9_FLAT
#if defined(MPK_P9_POLL_GLOBAL2) || defined(MPK_P9_POLL_GLOBAL)
#error "MPK_P9_FLAT replaces the layer_global arrival that MPK_P9_POLL_GLOBAL(2) polls"
#endif
        // Flat: every XCD publishes its own slot as soon as its own workers
        // are in; the gate takes the minimum over all eight.
        int global_expected = MPK_NUM_XCDS * (lean_layers_done + 1);
        bool const is_last_global = true;
#elif defined(MPK_P9_POLL_GLOBAL2)
        // Same gate as P9_POLL_GLOBAL, but the original returning atomic.
        (void)atom_add_release_gpu_s32(layer_global, 1);
        int const global_expected = 0;
        bool const is_last_global = false;
#elif defined(MPK_P9_POLL_GLOBAL)
        // The gate polls layer_global itself, so nobody needs the old value.
        asm volatile("flat_atomic_add %0, %1 sc1" ::"v"(layer_global), "v"(1)
                     : "memory");
        int const global_expected = 0;
        bool const is_last_global = false;
#else
        int global_prev = atom_add_release_gpu_s32(layer_global, 1);
        // Same identity one level up: the counter takes 8 arrivals per layer,
        // so the last XCD is the one whose pre-increment value is 7 mod 8 and
        // the release it publishes is its post-increment value.
        int global_expected = global_prev + 1;
        bool const is_last_global = (global_prev & (MPK_NUM_XCDS - 1)) == (MPK_NUM_XCDS - 1);
#endif
#else
#ifdef MPK_DRAIN_STATS
      s_was_last_local = (local_prev == local_expected - 1);
#endif
      if (local_prev == local_expected - 1) {
        // Last worker on this XCD: this XCD's MoE stores are all retired, so
        // publish one arrival on behalf of the whole die.
        int global_expected =
            (__atomic_load_n(layer_global, __ATOMIC_RELAXED) / MPK_NUM_XCDS + 1) * MPK_NUM_XCDS;
        int global_prev = atom_add_release_gpu_s32(layer_global, 1);
        bool const is_last_global = (global_prev == global_expected - 1);
#endif
#ifdef MPK_DRAIN_STATS
        if (is_last_global) {
          atomicAdd(&g_last_xcd[xcd_id], 1ULL);
          s_was_last_global = true;
        }
#endif
        if (is_last_global) {
          // Last XCD overall: fan the release out with st_wt (write-through,
          // bypasses L2) so every worker polls a line it already owns.
          //
          // The eight stores are handed to lanes 0..7 below rather than
          // issued here, so they become one instruction instead of eight
          // serial ones from the single most critical lane on the GPU -- 239
          // workgroups are spinning on flags it has not written yet, and the
          // eighth XCD would otherwise be released seven store-issues after
          // the first. The drain that has to precede the flags still runs
          // here, on the lane that owns the stores being advertised.
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          lean_rel_epoch = global_expected;
        }
#if MPK_NUM_XCDS == 4
        // 4 XCDs: publish this XCD's own flag as soon as its 31 workers
        // arrived. A last-global fan-out to all 4 lines lets two fast XCDs
        // release the two still in MoE.
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        lean_rel_epoch = qkv_epoch_expected;
#endif
      }
#ifdef MPK_DRAIN_STATS
      s_dr3 = __builtin_amdgcn_s_memrealtime();
#endif
    }

    // Lanes 0..7 of wave 0 publish the eight release flags in one wave
    // instruction. readfirstlane rather than LDS: the arrival above runs with
    // only lane 0 active, so once the `if` closes every lane reads lane 0's
    // value from the first active lane.
    lean_rel_epoch = __builtin_amdgcn_readfirstlane(lean_rel_epoch);
    if (lean_rel_epoch != 0) {
      MPK_XS_MARK(2);
#ifdef MPK_LAYER_NARROW_REL
      // One slot per AID instead of eight. Every slot received the identical
      // `lean_rel_epoch`, so the eight are pure redundancy, and all 16
      // write-through stores drain under the one vmcnt below -- the release
      // therefore costs the slowest of them. MoE narrow release, third use.
      if (tid == 0) {
#ifdef MPK_AID_SPLIT_FLAGS
        mpk_aid_publish(layer_release_shared,
                        0,
                        (unsigned)lean_rel_epoch,
                        MPK_AID_REGION_LAYER_RELEASE);
#else
        st_wt_u32((void *)&layer_release[0], (unsigned)lean_rel_epoch);
#endif
      }
#else
#ifdef MPK_P9_FLAT
      if (tid == 0) {
#else
      if (tid < 8) {
#endif
#ifdef MPK_AID_SPLIT_FLAGS
        mpk_aid_publish(layer_release_shared,
#ifdef MPK_P9_FLAT
                        xcd_id,
#else
                        tid,
#endif
                        (unsigned)lean_rel_epoch,
                        MPK_AID_REGION_LAYER_RELEASE);
#else
        st_wt_u32((void *)&layer_release[tid * 16], (unsigned)lean_rel_epoch);
#endif
        MPK_ATSOPS_AT(0, true);
      }
#endif
#ifndef MPK_PUB_NOWAIT
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
    }

#ifdef MPK_WS_FARCOPY
#if !defined(MPK_WSF32_AID) || !defined(MPK_P9_FLAT)
#error "MPK_WS_FARCOPY needs MPK_WSF32_AID and MPK_P9_FLAT"
#endif
#if defined(MPK_W2_CONSUMER_GATE) && \
    (defined(MPK_ROTATE_QKV_ATTN_RANKS) || defined(MPK_QKV_KSPLIT))
#error "MPK_WS_FARCOPY picks its re-poisoner as the highest rank of a prefix arriver set"
#endif
    // MPK_WSFC_V2 MoE workspace far copy. W2 wrote only its own AID's copy of
    // this layer's ring buffer, and its stores retired before the MoE call
    // returned. Right after this worker's own arrival, 16 lanes of wave 1
    // store its tile (the LDS mailbox) into the other AID's copy: wave 0's
    // arrival and gate poll never wait on those stores, and the gate barrier
    // needs lgkmcnt only. The highest arriving rank also re-poisons the other
    // AID's slots of the ring this layer READ in its own AID's copy: every
    // die's QKV prologue of this layer finished before any worker reached
    // Phase 9 (the MoE needs TopK, TopK every die's O-proj).
    {
      static_assert(QKV_REDUCTION_SIZE == MOE_HIDDEN_SIZE,
                    "workspace slot stride differs between W2 and QKV");
      static_assert(QKV_BATCH_SIZE == 1, "MPK_WS_FARCOPY is bs=1");
      constexpr int FC_H = QKV_REDUCTION_SIZE;
      constexpr int FC_COPY = MOE_WS_SLOTS * FC_H;
      static_assert(FC_H % 8 == 0, "a die's quarter must be whole float4s");
      int const fc_a = xcd_id >> 2;
#ifndef MPK_WSFC_NO_WRITERS
      // MPK_WSFC_V5: no barrier here -- it held waves 1-3's QKV weight
      // prefetch behind wave 0's arrival and publish drain. The barriers
      // after the MoE call already order the mailbox writes.
#ifdef MPK_WSFC_STATS
      if (tid == 64 && g_wsfc_cnt > 0) {
        mpk_wsfc_stat(g_wsfc_cnt == 16 ? 1 : 2, g_wsfc_cnt);
      }
#endif
      if (g_wsfc_cnt == 16 && tid >= 64 && tid < 80) {
        float *const fc_ring = (float *)output_ptrs[10];
        int const i = tid - 64;
        st_wt_f32x4((void *)(fc_ring + (1 - fc_a) * FC_COPY + g_wsfc_base + 4 * i),
                    g_wsfc_val[i]);
      }
      if (MPK_LAYER_GATE_ARRIVES && xcd_rank == arrivers_per_xcd - 1 &&
          task_layer_idx % MPK_WS_FC_LAYERS != 0 && tid >= 64) {
        float *const fc_in = (float *)input_ptrs[0];
        float *pz = fc_in + fc_a * FC_COPY + 2 * (1 - fc_a) * FC_H +
                    (xcd_id & 3) * (FC_H / 2);
        float4 const pv = make_float4(__uint_as_float(MPK_WS_POISON),
                                      __uint_as_float(MPK_WS_POISON),
                                      __uint_as_float(MPK_WS_POISON),
                                      __uint_as_float(MPK_WS_POISON));
        for (int i = tid - 64; i < FC_H / 8; i += (int)blockDim.x - 64) {
          st_wt_f32x4((void *)(pz + 4 * i), pv);
        }
      }
#else
      (void)fc_a;
#endif  // MPK_WSFC_NO_WRITERS
    }
#endif

#ifdef MPK_DRAIN_OVERLAP
    // ── Drain overlapped with the barrier spin (measurement arm) ──────────
    //
    // The arrival above has already been published; this retires the stores
    // it was meant to advertise. Every thread drains, because vmcnt is
    // per-wave and the arrival speaks for all four waves of this block.
    //
    // CORRECTNESS: this is deliberately weaker than the default. The arrival
    // *is* the statement "my W2 stores are visible", so publishing it before
    // the drain lets another XCD observe the arrival, clear the barrier, and
    // read a moe_workspace_f32 slot whose store has not landed -- the exact
    // race Phase 9 exists to close. It survives in practice only because the
    // release still has to cross all 8 XCDs, which takes far longer than the
    // drain; that is a timing accident, not a guarantee. Kept as a flag so
    // the ceiling of the idea is a measured number rather than an argument.
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif

#ifdef MPK_PREFETCH_NEXT_QKV
    // ── Next layer's QKV weight DMA, issued into the barrier spin ────────
    //
    // Measured, this barrier costs mean_wait=7683 ns per worker per layer
    // (n=4.39M): 0.277 ms of the 2.49 ms iteration spent waiting on the
    // slowest XCD. The wait is a scheduling cost, not a bandwidth one, so
    // weight traffic issued here is close to free -- the memory system is
    // otherwise idle for those microseconds.
    //
    // Placement is load-bearing on both sides:
    //
    //   * AFTER the arrival block above, not before it. The last-arriving
    //     XCD's leader executes `s_waitcnt vmcnt(0)` in its release fan-out,
    //     which drains *all* of this wave's outstanding vector memory --
    //     including a prefetch issued earlier. That worker is by definition
    //     the critical path, so it would stall on its own DMA and delay the
    //     release for all 240. Issuing after the fan-out keeps it off.
    //
    //   * AFTER the `__syncthreads()` that opens this block. buffer_load_lds
    //     retires under vmcnt while MoE's ds_writes retire under lgkmcnt, so
    //     the two are not ordered against each other; a Phase 8 LDS write
    //     still in flight could land on top of the staged weights. The
    //     s_barrier inside __syncthreads drains lgkmcnt first.
    //
    // Only Phase-1 workers stage anything: the rest have no QKV tile next
    // layer and their LDS would just be dirtied for nothing. tile_idx there
    // is qkv_attn_rank, and qkv_attn_rank is fixed for the life of the task,
    // so this worker prefetches exactly the tile it will itself execute --
    // which is why the rotation has to be applied here too, not just at the
    // Phase 1 guard.
    //
    // input_ptrs[24] is null on the last layer of the iteration -- see the
    // publish site in persistent_kernel.cuh for why nothing may be staged
    // across the iteration boundary.
#ifdef MPK_RET_WARM
    // The epilogue restores v41 from the frame slot at s33+0, written by
    // the prologue before the MoE stream evicted it: touch it now so the
    // restore hits L2. a31 is caller-saved and unused here; the
    // epilogue's vmcnt(0) retires this load before the return.
    asm volatile("scratch_load_dword a31, off, s33" ::: "a31");
#endif
#ifdef MPK_PFIN_RETIRE
#if !defined(MPK_FUSED_INLINE) || defined(MPK_QKV_PF_EARLY) || defined(MPK_QKV_L2PF)
#error "MPK_PFIN_RETIRE: inlined fused layer, Phase 9 prefetch issued here only"
#endif
    // On every path, before the DMA: nothing the compiler issued is still in
    // flight here, and the worker loop's table registers are then retired at
    // the next boundary instead of waited for behind this DMA.
#ifdef MPK_P9_KV_EARLY_DMA
    if (!(qkv_does_qkv && qkv_attn_rank >= 8 && input_ptrs[24] != nullptr && (tid >> 6) != 0))
#endif
    __builtin_amdgcn_s_waitcnt(0x0F70);  // vmcnt(0)
#endif
#if defined(MPK_QKV_PF_DELAY_US) || defined(MPK_QKV_PF_DELAY_NS)
    // Q ranks: keep the next layer's weight DMA out of the W2 fragment stream.
#ifdef MPK_QKV_PF_DELAY_NS
    unsigned long long const pfd_ticks = (MPK_QKV_PF_DELAY_NS) / 10ull;  // 1 tick = 10 ns
    int const pfd_sleep = 1;
#else
    unsigned long long const pfd_ticks = 100ull * (MPK_QKV_PF_DELAY_US);
    int const pfd_sleep = 8;
#endif
    if (qkv_does_qkv && qkv_attn_rank < 8 && input_ptrs[24] != nullptr) {
      unsigned long long const pfd_t0 = __builtin_amdgcn_s_memrealtime();
      while (__builtin_amdgcn_s_memrealtime() - pfd_t0 < pfd_ticks) {
        __builtin_amdgcn_s_sleep(pfd_sleep);
      }
    }
#endif
#ifdef MPK_PFD89_RT
    // K/V ranks: their own delay before the next layer's weight DMA.
    if (qkv_does_qkv && qkv_attn_rank >= 8 && input_ptrs[24] != nullptr &&
        g_mpk_pfd89_ticks != 0u) {
      unsigned long long const pfk_t0 = __builtin_amdgcn_s_memrealtime();
      while (__builtin_amdgcn_s_memrealtime() - pfk_t0 <
             (unsigned long long)g_mpk_pfd89_ticks) {
        __builtin_amdgcn_s_sleep(1);
      }
    }
#endif
#ifdef MPK_VTN_PFD_PROBE
    if (input_ptrs[24] != nullptr && qkv_does_qkv) {
      // MPK_VTN_PFD_PROBE (timing probe): delay the next layer's QKV weight DMA.
      unsigned long long const pp_t0 = __builtin_amdgcn_s_memrealtime();
      while (__builtin_amdgcn_s_memrealtime() - pp_t0 < (unsigned long long)((MPK_VTN_PFD_PROBE) / 10)) {
        __builtin_amdgcn_s_sleep(1);
      }
    }
#endif
#if defined(MPK_VTN_QWARM)
#ifndef MPK_VT2
#error "MPK_VTN_QWARM is wired for the VT2 / VTN QKV rank map"
#endif
    if (input_ptrs[24] != nullptr && !qkv_does_qkv) {
      // MPK_VTN_QWARM: warm this rank's share of the die's next-layer QKV weight tiles into L2.
      using QG = QkvWeightLds<QKV_TB, QKV_OUTPUT_PER_WG, QKV_REDUCTION_SIZE>;
      int const q0 = qkv_work_slots;  // ranks [0, q0) run QKV tiles (incl. a QROW producer)
      int const nwm = workers_per_xcd - q0;
      int const wi = xcd_rank - q0;
      if (nwm > 0 && wi >= 0) {
        uint32_t const tot = static_cast<uint32_t>(qkv_n_wgs_per_xcd) * QG::WG_BYTES;
        uint32_t const per = ((tot + nwm - 1) / nwm + 32767u) & ~32767u;
        uint32_t const b0 = static_cast<uint32_t>(wi) * per;
        uint8_t const *const qb = (uint8_t const *)input_ptrs[24];
        uint32_t const lt = static_cast<uint32_t>(tid) * 16u;
        for (uint32_t o = b0; o < b0 + per && o < tot; o += 32768u) {
          uint8_t const *pa[8];
#pragma unroll
          for (int k = 0; k < 8; ++k) {
            uint32_t const a = o + static_cast<uint32_t>(k) * 4096u + lt;
            pa[k] = qb + (a + 16u <= tot ? a : 0u);
          }
          i32x4_t d0, d1, d2, d3, d4, d5, d6, d7;
          asm volatile("global_load_dwordx4 %0, %8, off sc0\n"
                       "global_load_dwordx4 %1, %9, off sc0\n"
                       "global_load_dwordx4 %2, %10, off sc0\n"
                       "global_load_dwordx4 %3, %11, off sc0\n"
                       "global_load_dwordx4 %4, %12, off sc0\n"
                       "global_load_dwordx4 %5, %13, off sc0\n"
                       "global_load_dwordx4 %6, %14, off sc0\n"
                       "global_load_dwordx4 %7, %15, off sc0\n"
                       "s_waitcnt vmcnt(0)\n"
                       : "=&v"(d0), "=&v"(d1), "=&v"(d2), "=&v"(d3), "=&v"(d4), "=&v"(d5), "=&v"(d6), "=&v"(d7)
                       : "v"(pa[0]), "v"(pa[1]), "v"(pa[2]), "v"(pa[3]), "v"(pa[4]), "v"(pa[5]), "v"(pa[6]),
                         "v"(pa[7])
                       : "memory");
          (void)d0; (void)d1; (void)d2; (void)d3; (void)d4; (void)d5; (void)d6; (void)d7;
        }
      }
    }
#endif
    if (input_ptrs[24] != nullptr &&
        qkv_does_qkv
#if defined(MPK_GATE_DEFER) && defined(MPK_GATE_DEFER_PF_LATE)
        && false  // prefetched at the top of the next layer instead
#endif
#ifdef MPK_QKV_PF_ABLATE
        && false  // TIMING ONLY: no QKV weight DMA for layers 1..35
#endif
#ifdef MPK_QKV_PF_ORACLE89
        && qkv_attn_rank < 8  // TIMING ONLY: K/V ranks skip the DMA, use stale LDS
#endif
#ifdef MPK_QKV_PF_SKIP_W2
#ifdef MPK_QKV_PF_SKIP_ALL
        && false
#else
        && xcd_rank < 8
#endif
#endif
#ifdef MPK_QKV_PF_EARLY
        && false
#endif
#ifdef MPK_QKV_PF_WAVE_SPLIT
        // ── Keep the poller's wave out of the pre-gate DMA ─────────────────
        //
        // tid 0 is the only thread that spins on `layer_release` below, and it
        // is the thread whose latency the whole workgroup then waits on. Its
        // wave issues 1/4 of this prefetch: NUM_WAVES buffer_load_lds per lane,
        // all still outstanding when it enters the spin. The poll is itself a
        // vector load, so it queues behind them in the same wave's memory
        // pipeline -- the poller pays its own prefetch's latency on every
        // iteration of the loop, and the release it is looking for is seen late
        // by exactly the amount.
        //
        // Waves 1..3 have nothing to do during the spin, so they keep their
        // three quarters here; wave 0's quarter is re-issued after the gate,
        // where it costs the poller nothing. The split is clean because
        // qkv_prefetch_weights_lds partitions LDS by warp_id (warp_id * 1024)
        // and each wave's buffer_load_lds writes only its own 1 KiB block, so
        // deferring one wave defers exactly one quarter of the staged tile and
        // touches nobody else's bytes.
        && tid >= 64
#endif
#ifdef MPK_P9_KV_EARLY_DMA
        && !(qkv_attn_rank >= 8 && (tid >> 6) != 0)  // issued after the post-MoE barrier
#endif
    ) {
#ifdef MPK_QKV_KSPLIT
      int const qkv_k_part = qkv_does_k1 ? 1 : 0;
      int const qkv_tile =
          qkv_does_k1 ? qkv_attn_rank - qkv_k1_base : qkv_attn_rank;
      int const qkv_k_iter0 = qkv_k_part ? QKV_K_LO : 0;
      int const qkv_k_niters =
          qkv_k_part ? (QKV_MFMA_ITERS - QKV_K_LO) : QKV_K_LO;
      qkv_prefetch_weights_lds<QKV_TB,
                               QKV_OUTPUT_PER_WG,
                               QKV_REDUCTION_SIZE>(
          input_ptrs[24],
          qkv_n_wgs_per_xcd,
          qkv_tile,
          qkv_k_iter0,
          qkv_k_niters);
#else
#if defined(MPK_VT2) && defined(MPK_VT2_QKV) && defined(MPK_VT2_QKV_PF_AUX)
      if (QKV_BATCH_SIZE >= 2) {
#ifdef MPK_VT2_QKV_EARLY
        if (g_mpk_qkve_rt != 0) {  // the part above MPK_KVES_LO went out before round 2's W2 call
          qkv_prefetch_weights_lds<QKV_TB, QKV_OUTPUT_PER_WG, QKV_REDUCTION_SIZE,
                                   MPK_VT2_QKV_PF_AUX, 0, MPK_KVES_LO>(
              input_ptrs[24], qkv_n_wgs_per_xcd, qkv_pf_tile);
        } else
#endif
        qkv_prefetch_weights_lds<QKV_TB,
                                 QKV_OUTPUT_PER_WG,
                                 QKV_REDUCTION_SIZE,
                                 /*AUX=*/MPK_VT2_QKV_PF_AUX>(
            input_ptrs[24], qkv_n_wgs_per_xcd, qkv_pf_tile);
      } else
#endif
#ifdef MPK_QKV_L2PRE
      if (true) {
        qkv_prefetch_weights_lds<QKV_TB,
                                 QKV_OUTPUT_PER_WG,
                                 QKV_REDUCTION_SIZE,
                                 /*AUX=*/0>(
            input_ptrs[24], qkv_n_wgs_per_xcd, qkv_pf_tile);
      } else
#endif
#ifdef MPK_QKV_KV_PF_PLAIN
      if (qkv_attn_rank >= 8) {
        qkv_prefetch_weights_lds<QKV_TB,
                                 QKV_OUTPUT_PER_WG,
                                 QKV_REDUCTION_SIZE,
                                 /*AUX=*/0>(
            input_ptrs[24], qkv_n_wgs_per_xcd, qkv_pf_tile);
      } else
#endif
#ifdef MPK_QKV_KV_PF_SC0
      if (qkv_attn_rank >= 8) {
        qkv_prefetch_weights_lds<QKV_TB,
                                 QKV_OUTPUT_PER_WG,
                                 QKV_REDUCTION_SIZE,
                                 /*AUX=*/1>(
            input_ptrs[24], qkv_n_wgs_per_xcd, qkv_pf_tile);
      } else
#endif
#ifdef MPK_KV_EARLY_STAGE
      if (qkv_attn_rank >= 8 && g_mpk_kves != 0) {
        qkv_prefetch_weights_lds<QKV_TB, QKV_OUTPUT_PER_WG,
                                 QKV_REDUCTION_SIZE, 3, 0, MPK_KVES_LO>(
            input_ptrs[24], qkv_n_wgs_per_xcd, qkv_pf_tile);
      } else
#endif
      qkv_prefetch_weights_lds<QKV_TB,
                               QKV_OUTPUT_PER_WG,
                               QKV_REDUCTION_SIZE>(
          input_ptrs[24], qkv_n_wgs_per_xcd, qkv_pf_tile);
#ifdef MPK_KV_SPREAD
      qkv_prefetch_kv_extra_lds<QKV_BATCH_SIZE,
                                QKV_OUTPUT_PER_WG,
                                QKV_REDUCTION_SIZE,
                                NUM_Q_PER_KV>(
          input_ptrs[24], qkv_n_wgs_per_xcd, qkv_pf_tile);
#endif
#endif
    }
#endif

    // ── MPK_W2_CONSUMER_GATE: publish-and-go ─────────────────────────────
    //
    // Under this flag the arrival tree above still runs -- every worker still
    // reports in, the last XCD still fans out `layer_release` -- but nobody
    // waits here. The wait moves to the one place that needs it: the Phase 1
    // workspace consumer of the *next* layer.
    //
    // Why the rendezvous was never the requirement. The hazard Phase 9 exists
    // to close is exactly one edge: the next layer's QKV prologue
    // (gang_resaddf32_rmsnorm_linear_mxfp4_bias_kvupd_kernel) reads
    // moe_workspace_f32, which this layer's W2 epilogue writes. That reader is
    // already guarded by `qkv_attn_rank < total_qkv_tiles_per_xcd`, so only
    // total_qkv_tiles_per_xcd of the workers_per_xcd workers on each XCD ever
    // touch the buffer. Ordering the other ~20 against a buffer they never
    // load is synchronization the dependency does not ask for, and the
    // measured rank spin cliff is that surplus made visible: ranks 0-22 spin
    // ~5.1 us, ranks 23-29 ~25.2 us.
    //
    // What keeps the non-consumers in line is Phase 6. Its attn_release poll
    // is joined by all workers_per_xcd and its release cannot be published
    // until every XCD's attention chunks have arrived, which requires QKV,
    // which requires this release. So the non-consumers are still ordered --
    // transitively, at a gate they already had to pay -- rather than at a
    // second gate of their own.
    //
    // This is the "weakened" case the note at the top of this file warns
    // about, so the layer-boundary acquire moves with the wait and returns to
    // `sc1`: the five sc1 drops in 981e124 were justified by "no worker
    // reaches layer N+1 until every worker passed Phase 9 for layer N", and
    // that premise is exactly what this flag retires. Only the consumers pay
    // the sc1 now, which is what makes it affordable.
    //
    // The wait stays here rather than moving to the top of the next layer:
    // xcd_rank is fixed for the life of the task (same property the QKV
    // prefetch relies on), so the worker that waits at the end of layer N is
    // exactly the worker that reads the workspace in layer N+1. Keeping it
    // here also keeps `s_layer_rel_prev` valid -- it is a snapshot taken in
    // this invocation, and there is no cross-invocation channel to carry a
    // release target through.
#ifdef MPK_DRAIN_STATS
    // Splits the "spin" segment. Everything between s_dr3 and here is the
    // release fan-out plus the next-layer QKV prefetch *issue*, which the
    // original window silently folded into the wait -- which is why the
    // worker that publishes the release, and therefore waits on nobody, still
    // showed ~2 us of "spin".
    unsigned long long _dr35 = __builtin_amdgcn_s_memrealtime();
#endif
    // Same split point MPK_DRAIN_STATS uses: everything before this is the
    // arrival tree plus the release fan-out and the next-layer QKV prefetch
    // *issue*; everything after is the poll proper. Only slot 9->10 is a
    // like-for-like wait span: the pre-gate half is release-publishing work,
    // not waiting, so folding it in overstates the gate.
    MPK_PHASE_MARK(_pslot_w, 9);
#if defined(MPK_P9_DEFER)
#if defined(MPK_GATE_DEFER) || defined(MPK_P9_FLAT) || MPK_NUM_XCDS == 4
#error "MPK_P9_DEFER defers the single-counter layer gate; not wired for MPK_GATE_DEFER / MPK_P9_FLAT / 4 XCDs"
#endif
    if (tid == 0) {
      *mpk_p9_word(input_ptrs) = MPK_LAYER_GATE_JOINS ? s_layer_rel_prev + 8 : 0;
    }
    if (false) {
#elif defined(MPK_GATE_DEFER)
    bool const gd_defer =
        (task_layer_idx % MPK_GD_LAYERS) != MPK_GD_LAYERS - 1;
    if (tid == 0 && MPK_LAYER_GATE_JOINS && gd_defer) {
      s_gd_prev = s_layer_rel_prev;
    }
    if (tid == 0 && MPK_LAYER_GATE_JOINS && !gd_defer) {
#else
    if (tid == 0 && MPK_LAYER_GATE_JOINS) {
#endif
#if MPK_NUM_XCDS == 4
      // Every XCD publishes only its own flag, so the gate reads all four.
      // The reported `observed` packs which flags are short in bits 0..3 and
      // the minimum epoch seen above bit 8.
      MPK_WS_WAIT_BEGIN(90, qkv_epoch_expected);
      int _lg_spins = 0;
      for (;;) {
        // Invalidate L2 before re-reading. These flags live in plain
        // hipMalloc VRAM, which is MTYPE_NC under NPS2 -- cacheable but NOT
        // coherent across XCDs -- so a reader can hold a line indefinitely
        // and never observe another die's write-through publish. Measured on
        // this part: `buffer_inv sc1` is the only acquire that reaches zero
        // stale reads; a bare `buffer_inv` is an architectural NOP on gfx950
        // and invalidates nothing.
        asm volatile("buffer_inv" MPK_DEV_SC_INV "" ::: "memory");
        int _f0 = MPK_LD_GATE_AID(&layer_release[0 * 16]);
        int _f1 = MPK_LD_GATE_AID(&layer_release[1 * 16]);
        int _f2 = MPK_LD_GATE_AID(&layer_release[2 * 16]);
        int _f3 = MPK_LD_GATE_AID(&layer_release[3 * 16]);
        int _lo01 = _f0 < _f1 ? _f0 : _f1;
        int _lo23 = _f2 < _f3 ? _f2 : _f3;
        int _lo = _lo01 < _lo23 ? _lo01 : _lo23;
        if (_lo >= qkv_epoch_expected) {
          break;
        }
        int _sm = (_f0 < qkv_epoch_expected ? 1 : 0) |
                  (_f1 < qkv_epoch_expected ? 2 : 0) |
                  (_f2 < qkv_epoch_expected ? 4 : 0) |
                  (_f3 < qkv_epoch_expected ? 8 : 0);
        (void)_sm;
        MPK_WS_WAIT_TICK((_f0 & 0xf) | ((_f1 & 0xf) << 4) |
                             ((_f2 & 0xf) << 8) | ((_f3 & 0xf) << 12) |
                             ((arrivers_per_xcd & 0xff) << 16) |
                             ((MPK_LD_GATE(&layer_global[0]) & 0xff) << 24),
                         _lg_spins);
        _lg_spins++;
#ifndef MPK_LAYER_GATE_BUSY_POLL
        __builtin_amdgcn_s_sleep(1);
#endif
      }
#else
#if defined(MPK_P9_POLL_GLOBAL) || defined(MPK_P9_POLL_GLOBAL2)
#ifndef MPK_LEAN_ARRIVE
#error "MPK_P9_POLL_GLOBAL relies on MPK_LEAN_ARRIVE's s_layer_rel_prev = 8 * layers done"
#endif
#ifdef MPK_POLL_PIPE
      mpk_poll_ge_s32(layer_global, s_layer_rel_prev + 8);
      while (false) {
#else
      while (MPK_LD_GATE(layer_global) < s_layer_rel_prev + 8) {
#endif
#elif defined(MPK_LAYER_NARROW_REL)
      while (MPK_LD_GATE_AID(&layer_release[0]) <= s_layer_rel_prev) {
#else
#if defined(MPK_P9_FLAT)
      while (ld_aid_min8_s32(layer_release) <= s_layer_rel_prev) {
#else
      while (MPK_LD_GATE_AID(&layer_release[xcd_id * 16]) <= s_layer_rel_prev) {
#endif
#endif
#ifndef MPK_LAYER_GATE_BUSY_POLL
        __builtin_amdgcn_s_sleep(1);
#endif
      }
#endif
#ifndef MPK_W2_CONSUMER_GATE
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, MPK_DEV_FENCE_ACQ);
#endif
#ifdef MPK_W2_CONSUMER_GATE
      // The agent-scope acquire fence is gone from this arm on purpose: it
      // lowers to `buffer_inv sc1`, the exact instruction below, so the two
      // together emitted a full L2 invalidate twice back to back (visible in
      // ATT as the adjacent pair at the gate's exit). The asm carries a
      // "memory" clobber, so the compiler-side ordering the fence provided is
      // unchanged; the bare asm with no fence is what this gate wants.
      //
      // Layer-boundary acquire, relocated from the top of the layer and
      // promoted back to `sc1` (vL1 + L2). W2 writes the workspace
      // write-through from another XCD and MI350 L2 is not coherent across
      // XCDs, so a line this XCD still holds from last layer's read of the
      // same address would otherwise be served stale. The default build gets
      // away with a vL1-only `buffer_inv` because the full rendezvous
      // guarantees every producer retired before any consumer proceeds; this
      // arm retires that guarantee, so the L2 invalidate comes back. Only the
      // gate-joining workers execute it.
#if defined(MPK_P9_PREINV) && defined(MPK_LEAN_ARRIVE)
      // L2 was invalidated by this die's last arriver before the release
      // (MPK_P9_PREINV); only this CU's vL1 can still hold a stale line.
      asm volatile("buffer_inv sc0" ::: "memory");
#elif defined(MPK_P9_LEADER_INV) && defined(MPK_AID_SPLIT_FLAGS)
      {
        // One L2 invalidate per die: the L2 serializes them, and it is
        // shared by every worker on the XCC, so the first joiner's covers
        // the rest. Followers wait for it to retire, then drop their vL1.
        unsigned _xcc;
        MPK_GET_XCC_ID16(_xcc);
        _xcc &= 7;
        int *_l2c = g_aid_flag_rep[_xcc >> 2];
        if (_l2c == nullptr) {
          asm volatile("buffer_inv" MPK_DEV_SC_INV "" ::: "memory");
        } else {
          _l2c += MPK_AID_L2CLEAN_BASE_INTS + (int)_xcc * 32;
          if (atomicMax(_l2c, qkv_epoch_expected) < qkv_epoch_expected) {
            asm volatile("buffer_inv" MPK_DEV_SC_INV "\n s_waitcnt vmcnt(0)" ::: "memory");
            __hip_atomic_store(_l2c + 16, qkv_epoch_expected, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
          } else {
            while (__hip_atomic_load(_l2c + 16, __ATOMIC_RELAXED,
                                     __HIP_MEMORY_SCOPE_AGENT) <
                   qkv_epoch_expected) {
              __builtin_amdgcn_s_sleep(1);
            }
            asm volatile("buffer_inv sc0" ::: "memory");
          }
        }
      }
#elif defined(MPK_P9_NO_L2INV)
      asm volatile("buffer_inv" ::: "memory");
#else
      asm volatile("buffer_inv" MPK_DEV_SC_INV "" ::: "memory");
#endif
#endif
#ifdef MPK_DRAIN_STATS
      unsigned long long _dr4 = __builtin_amdgcn_s_memrealtime();
      // s_memrealtime ticks at 100 MHz -> 10 ns per tick.
      unsigned long long drain = (_dr1 - _dr0) * 10;
      unsigned long long sync = (_dr2 - _dr1) * 10;
      unsigned long long arrive = (s_dr3 - _dr2) * 10;
      unsigned long long spin = (_dr4 - s_dr3) * 10;
      unsigned long long fanpf = (_dr35 - s_dr3) * 10;
      unsigned long long poll = (_dr4 - _dr35) * 10;
      atomicAdd(&g_fanpf_sum, fanpf);
      atomicAdd(&g_poll_sum, poll);
      if (s_was_last_global) {
        atomicAdd(&g_poll_lastglobal, poll);
      }
      unsigned long long n = atomicAdd(&g_drain_n, 1ULL);
      atomicAdd(&g_drain_sum, drain);
      atomicAdd(&g_sync_sum, sync);
      atomicAdd(&g_arrive_sum, arrive);
      atomicAdd(&g_spin_sum, spin);
      atomicAdd(&g_spin_xcd[xcd_id], spin);
      atomicAdd(&g_n_xcd[xcd_id], 1ULL);
      if (xcd_rank < 64) {
        atomicAdd(&g_spin_rank[xcd_rank], spin);
        atomicAdd(&g_n_rank[xcd_rank], 1ULL);
      }
      if (s_was_last_local) {
        atomicAdd(&g_spin_lastlocal, spin);
        atomicAdd(&g_n_lastlocal, 1ULL);
      }
      if (s_was_last_global) {
        atomicAdd(&g_spin_lastglobal, spin);
        atomicAdd(&g_n_lastglobal, 1ULL);
      }
      if (n % 100000 == 0 && n > 0) {
        printf("[DRAIN] n=%llu drain=%llu sync=%llu arrive=%llu spin=%llu "
               "tot=%llu\n",
               n,
               g_drain_sum / n,
               g_sync_sum / n,
               g_arrive_sum / n,
               g_spin_sum / n,
               (g_drain_sum + g_sync_sum + g_arrive_sum + g_spin_sum) / n);
        printf("[XCDSPIN] %llu %llu %llu %llu %llu %llu %llu %llu | last "
               "%llu %llu %llu %llu %llu %llu %llu %llu\n",
               g_spin_xcd[0] / (g_n_xcd[0] ? g_n_xcd[0] : 1),
               g_spin_xcd[1] / (g_n_xcd[1] ? g_n_xcd[1] : 1),
               g_spin_xcd[2] / (g_n_xcd[2] ? g_n_xcd[2] : 1),
               g_spin_xcd[3] / (g_n_xcd[3] ? g_n_xcd[3] : 1),
               g_spin_xcd[4] / (g_n_xcd[4] ? g_n_xcd[4] : 1),
               g_spin_xcd[5] / (g_n_xcd[5] ? g_n_xcd[5] : 1),
               g_spin_xcd[6] / (g_n_xcd[6] ? g_n_xcd[6] : 1),
               g_spin_xcd[7] / (g_n_xcd[7] ? g_n_xcd[7] : 1),
               g_last_xcd[0],
               g_last_xcd[1],
               g_last_xcd[2],
               g_last_xcd[3],
               g_last_xcd[4],
               g_last_xcd[5],
               g_last_xcd[6],
               g_last_xcd[7]);
        printf("[INTER] lastlocal_spin=%llu n=%llu | lastglobal_spin=%llu "
               "n=%llu\n",
               g_spin_lastlocal / (g_n_lastlocal ? g_n_lastlocal : 1),
               g_n_lastlocal,
               g_spin_lastglobal / (g_n_lastglobal ? g_n_lastglobal : 1),
               g_n_lastglobal);
        printf("[SPLIT] fanout+prefetch=%llu poll=%llu | lastglobal_poll=%llu\n",
               g_fanpf_sum / n,
               g_poll_sum / n,
               g_poll_lastglobal / (g_n_lastglobal ? g_n_lastglobal : 1));
        printf("[MLPRO] prologue=%llu copy=%llu n=%llu\n",
               g_mlprologue_sum / (g_mlprologue_n ? g_mlprologue_n : 1),
               g_mlcopy_sum / (g_mlcopy_n ? g_mlcopy_n : 1),
               g_mlprologue_n);
        for (int r0 = 0; r0 < 30; r0 += 10) {
          printf("[RANK%02d] %llu %llu %llu %llu %llu %llu %llu %llu %llu "
                 "%llu\n",
                 r0,
                 g_spin_rank[r0 + 0] / (g_n_rank[r0 + 0] ? g_n_rank[r0 + 0] : 1),
                 g_spin_rank[r0 + 1] / (g_n_rank[r0 + 1] ? g_n_rank[r0 + 1] : 1),
                 g_spin_rank[r0 + 2] / (g_n_rank[r0 + 2] ? g_n_rank[r0 + 2] : 1),
                 g_spin_rank[r0 + 3] / (g_n_rank[r0 + 3] ? g_n_rank[r0 + 3] : 1),
                 g_spin_rank[r0 + 4] / (g_n_rank[r0 + 4] ? g_n_rank[r0 + 4] : 1),
                 g_spin_rank[r0 + 5] / (g_n_rank[r0 + 5] ? g_n_rank[r0 + 5] : 1),
                 g_spin_rank[r0 + 6] / (g_n_rank[r0 + 6] ? g_n_rank[r0 + 6] : 1),
                 g_spin_rank[r0 + 7] / (g_n_rank[r0 + 7] ? g_n_rank[r0 + 7] : 1),
                 g_spin_rank[r0 + 8] / (g_n_rank[r0 + 8] ? g_n_rank[r0 + 8] : 1),
                 g_spin_rank[r0 + 9] / (g_n_rank[r0 + 9] ? g_n_rank[r0 + 9] : 1));
        }
      }
#endif
    }
    // Broadcasts the gate to the other 255 threads: only tid 0 waited (and,
    // under MPK_W2_CONSUMER_GATE, only on gate-joining workers), so this is
    // what makes the release visible to the whole block.
    __syncthreads();
    MPK_PHASE_MARK(_pslot_w, 10);
#ifdef MPK_ILPER
    if (tid == 0 && s_ilsub_ml >= 0 && s_ilsub_ml < 64) {
      g_ilper[blockIdx.x * 128 + 64 + s_ilsub_ml] = __builtin_amdgcn_s_memrealtime();
    }
#endif
    MPK_ILSTAMP(17, s_ilsub_ml == MPK_ILSUB_L0 + 1);
    MPK_ILSTAMP(8, s_ilsub_ml == MPK_ILSUB_L0);

#if defined(MPK_PREFETCH_NEXT_QKV) && defined(MPK_QKV_PF_WAVE_SPLIT)
    // Wave 0's deferred quarter of the staged tile. It has to be issued here,
    // past the gate, because that is the whole point of deferring it: before
    // the gate it would sit in tid 0's memory pipeline ahead of the poll.
    //
    // Issuing it *after* the __syncthreads rather than inside the `tid == 0`
    // block is required -- buffer_load_lds is per lane, so all 64 lanes of
    // wave 0 must run it to fill the 1 KiB block, and only lane 0 was inside.
    //
    // It is still a prefetch: the consumer in the next layer drains with
    // `s_waitcnt vmcnt(0)` before its first ds_read, so the DMA has the whole
    // rest of this layer's epilogue plus the next layer's prologue to land.
    if (input_ptrs[24] != nullptr && qkv_does_qkv &&
        tid < 64) {
#ifdef MPK_QKV_KSPLIT
      int const qkv_k_part = qkv_does_k1 ? 1 : 0;
      int const qkv_tile =
          qkv_does_k1 ? qkv_attn_rank - qkv_k1_base : qkv_attn_rank;
      int const qkv_k_iter0 = qkv_k_part ? QKV_K_LO : 0;
      int const qkv_k_niters =
          qkv_k_part ? (QKV_MFMA_ITERS - QKV_K_LO) : QKV_K_LO;
      qkv_prefetch_weights_lds<QKV_TB,
                               QKV_OUTPUT_PER_WG,
                               QKV_REDUCTION_SIZE>(
          input_ptrs[24],
          qkv_n_wgs_per_xcd,
          qkv_tile,
          qkv_k_iter0,
          qkv_k_niters);
#else
      qkv_prefetch_weights_lds<QKV_TB,
                               QKV_OUTPUT_PER_WG,
                               QKV_REDUCTION_SIZE>(
          input_ptrs[24], qkv_n_wgs_per_xcd, qkv_pf_tile);
#endif
    }
#endif
    // No invalidate here: in the default build the task body re-entered for
    // the next layer opens with the layer-boundary acquire near the top of
    // this function, which is the same instruction against the same data.
    // Issuing it twice per layer costs a full L2 invalidate for nothing.
    // Under MPK_W2_CONSUMER_GATE that top-of-layer site is compiled out and
    // the acquire is the `buffer_inv sc1` above, inside the gate.
  }
#endif

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  __syncthreads();
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
#endif
#ifdef MPK_ML_TABLE_INLAYER
  if (mlt_go) {
    __syncthreads();  // every wave is done with this layer's pointers
    int const mlt_i = tid - 64;
    void **const in_w = const_cast<void **>(input_ptrs);
    void **const out_w = const_cast<void **>(output_ptrs);
    if (mlt_i >= 0 && mlt_i < MPK_MLT_NIN) {
      if (mlt_i != 24 && mlt_i != 25) {
        in_w[mlt_i] = (void *)mlt_v;
      }
      if (mlt_i == 4) {
        in_w[25] = (void *)mlt_v;  // next layer's own QKV weight
      }
    } else if (mlt_i >= MPK_MLT_NIN && mlt_i < MPK_MLT_NIN + MPK_MLT_NOUT) {
      out_w[mlt_i - MPK_MLT_NIN] = (void *)mlt_v;
    } else if (mlt_i == MPK_MLT_NIN + MPK_MLT_NOUT) {
      in_w[24] = (void *)mlt_v;  // layer ml+2's QKV weight, or null
    }
  }
#endif
  MPK_PHASE_MARK(_pslot_w, 11);
  // Back-to-back with slot 11 and nothing in between, so the span reported
  // for slot 12 is the cost of one mark -- the recorder measuring itself.
  MPK_PHASE_MARK(_pslot_w, 12);
  MPK_ILSTAMP(9, s_ilsub_ml == MPK_ILSUB_L0);
  MPK_TW_SUB(90, tile_idx);
  MPK_WS_PHASE(90, qkv_epoch_expected, xcd_id);
}

#ifdef MPK_FUSED_INLINE
// Every caller except the ml loop keeps the out-of-line call.
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
          int NUM_REQS = 1>
__device__ __noinline__ void gang_full_layer_fused_kernel_mi300(void *const *input_ptrs,
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
  gang_full_layer_fused_kernel_mi300_body<QKV_BATCH_SIZE, QKV_OUTPUT_PER_WG, QKV_REDUCTION_SIZE, ACTUAL_HIDDEN_DIM, HEAD_DIM, NUM_Q_PER_KV, PAGE_SIZE, MAX_SEQ_LEN, NUM_KV_CHUNKS, Q_WORKSPACE_STRIDE, KV_CACHE_STRIDE, NUM_KV_HEADS, SLIDING_WINDOW, HAS_SINKS, OPROJ_OUTPUT_PER_WG, OPROJ_REDUCTION_SIZE, NUM_EXPERTS, TOPK_K, MOE_INTERMEDIATE_SIZE, MOE_HIDDEN_SIZE, MOE_W13_OUTPUT_PER_WG, MOE_W2_OUTPUT_PER_WG, DECODE_ONLY, NUM_REQS>(
      input_ptrs,
      output_ptrs,
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
      attn_scale,
      total_qkv_tiles_per_xcd,
      oproj_n_wgs_per_xcd,
      oproj_output_stride,
      router_tile_n,
      total_oproj_tiles,
      total_topk_tiles,
      oproj_tiles_per_xcd,
      moe_total_tiles_per_xcd,
      workers_per_xcd,
      tile_idx,
      task_layer_idx);
}

// ml-loop copy: no call boundary between the Phase-9 gate and the next QKV.
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
          int NUM_REQS = 1>
__device__ __forceinline__ void gang_full_layer_fused_kernel_mi300_inl(void *const *input_ptrs,
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
  gang_full_layer_fused_kernel_mi300_body<QKV_BATCH_SIZE, QKV_OUTPUT_PER_WG, QKV_REDUCTION_SIZE, ACTUAL_HIDDEN_DIM, HEAD_DIM, NUM_Q_PER_KV, PAGE_SIZE, MAX_SEQ_LEN, NUM_KV_CHUNKS, Q_WORKSPACE_STRIDE, KV_CACHE_STRIDE, NUM_KV_HEADS, SLIDING_WINDOW, HAS_SINKS, OPROJ_OUTPUT_PER_WG, OPROJ_REDUCTION_SIZE, NUM_EXPERTS, TOPK_K, MOE_INTERMEDIATE_SIZE, MOE_HIDDEN_SIZE, MOE_W13_OUTPUT_PER_WG, MOE_W2_OUTPUT_PER_WG, DECODE_ONLY, NUM_REQS>(
      input_ptrs,
      output_ptrs,
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
      attn_scale,
      total_qkv_tiles_per_xcd,
      oproj_n_wgs_per_xcd,
      oproj_output_stride,
      router_tile_n,
      total_oproj_tiles,
      total_topk_tiles,
      oproj_tiles_per_xcd,
      moe_total_tiles_per_xcd,
      workers_per_xcd,
      tile_idx,
      task_layer_idx);
}
#endif

} // namespace kernel
