// MPK_VT2_PATCHED
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
// VTN verify engine defaults above 4 tokens (measured, lossless):
//   MPK_VTN_QMP (multi-producer QKV rounds): T=8 -3.4%, T=6 -1.3%; opt out with -DMPK_VTN_NO_QMP.
//   MPK_VTN_ROPW (router-partial die sums on all four waves): T=8 -1.0% (T=6 +0.4%: off there);
//   opt out with -DMPK_VTN_NO_ROPW.
//   MPK_VTN_QPAIR (a rank's last two QKV rounds in one call, the second token as MFMA column 1):
//   T=8 -0.95%, T=6 -0.79% (A/B/A/B/A, lossless); opt out with -DMPK_VTN_NO_QPAIR.
//   MPK_VTN_QTRI (T=7..9: a rank with three QKV rounds runs them as one call, MFMA columns 0-2):
//   T=8 -1.73% on top of QPAIR (A/B/A/B/A, lossless); opt out with -DMPK_VTN_NO_QTRI.
#if defined(MPK_VTN_T) && MPK_VTN_T > 4 && defined(MPK_VTN_QROW_TAG) && !defined(MPK_VTN_RESTAGE) && \
    !defined(MPK_VTN_NO_QMP) && !defined(MPK_VTN_QMP)
#define MPK_VTN_QMP
#endif
#if defined(MPK_VTN_T) && MPK_VTN_T > 6 && !defined(MPK_VTN_NO_ROPW) && !defined(MPK_VTN_ROPW)
#define MPK_VTN_ROPW
#endif
#if defined(MPK_VTN_QMP) && !defined(MPK_VTN_NO_QPAIR) && !defined(MPK_VTN_QPAIR)
#define MPK_VTN_QPAIR
#endif
#if defined(MPK_VTN_QPAIR) && MPK_VTN_T > 6 && MPK_VTN_T <= 9 && !defined(MPK_VTN_NO_QTRI) && \
    !defined(MPK_VTN_QTRI)
#define MPK_VTN_QTRI
#endif

// MPK_MULTI_DEV=N: one process drives N HIP devices -- the DPX (N=2) or CPX (N=8) partitions of one
// GPU -- and each hosts MPK_NUM_XCDS / N logical XCDs of the 8-XCD megakernel. HW_REG_XCC_ID counts
// from 0 on every partition, so the host writes each device's first logical XCD and first worker
// id here. Every XCC_ID read goes through these macros (the original instruction otherwise).
#ifdef MPK_MULTI_DEV
#ifdef MPK_XCD_SUBSET
#error "MPK_MULTI_DEV and MPK_XCD_SUBSET both remap XCC ids"
#endif
__constant__ int g_mpk_dev_xcd_base;
__constant__ int g_mpk_dev_worker_base;
#define MPK_GET_XCC_ID16(v)                                                    \
  do {                                                                         \
    asm volatile("s_getreg_b32 %0, hwreg(" "HW_REG_XCC_ID, 0, 16)" : "=s"(v)); \
    (v) += g_mpk_dev_xcd_base;                                                 \
  } while (0)
#define MPK_GET_XCC_ID32(v) MPK_GET_XCC_ID16(v)
#ifndef MPK_NUM_XCDS
#define MPK_NUM_XCDS 8
#endif
#if MPK_NUM_XCDS / MPK_MULTI_DEV == 1
// CPX: device scope is one XCC, so cross-XCD acquires / releases / fences need system scope.
#define MPK_CPX_SCOPE 1
#endif
#elif defined(MPK_XCCID_CONST_TEST) // timing probe: MPK_MULTI_DEV's id form, base 0
__constant__ int g_mpk_dev_xcd_base;
#define MPK_GET_XCC_ID16(v)                                                    \
  do {                                                                         \
    asm volatile("s_getreg_b32 %0, hwreg(" "HW_REG_XCC_ID, 0, 16)" : "=s"(v)); \
    (v) += g_mpk_dev_xcd_base;                                                 \
  } while (0)
#define MPK_GET_XCC_ID32(v) MPK_GET_XCC_ID16(v)
#else
#define MPK_GET_XCC_ID16(v)                                                    \
  asm volatile("s_getreg_b32 %0, hwreg(" "HW_REG_XCC_ID, 0, 16)" : "=s"(v))
#define MPK_GET_XCC_ID32(v)                                                    \
  asm volatile("s_getreg_b32 %0, hwreg(" "HW_REG_XCC_ID)" : "=s"(v))
#endif
#ifdef MPK_CPX_SCOPE
#ifdef MPK_CPX_SC_AGENT // timing probe: all three groups below stay agent-scope
#define MPK_CPX_INV_AGENT
#define MPK_CPX_WB_AGENT
#define MPK_CPX_FENCE_AGENT
#endif
#ifdef MPK_CPX_INV_AGENT // timing probe: asm buffer_inv stays agent-scope
#define MPK_DEV_SC_INV " sc1"
#else
#define MPK_DEV_SC_INV " sc0 sc1"
#endif
#ifdef MPK_CPX_WB_AGENT // timing probe: asm buffer_wbl2 stays agent-scope
#define MPK_DEV_SC_WB " sc1"
#else
#define MPK_DEV_SC_WB " sc0 sc1"
#endif
#ifdef MPK_CPX_FENCE_AGENT // timing probe: both fence groups below stay agent-scope
#define MPK_CPX_FACQ_AGENT
#define MPK_CPX_FREL_AGENT
#endif
#define MPK_DEV_FENCE ""
#ifdef MPK_CPX_FACQ_AGENT // timing probe: acquire fences stay agent-scope
#define MPK_DEV_FENCE_ACQ "agent"
#else
#define MPK_DEV_FENCE_ACQ ""
#endif
#ifdef MPK_CPX_FREL_AGENT // timing probe: release fences stay agent-scope
#define MPK_DEV_FENCE_REL "agent"
#else
#define MPK_DEV_FENCE_REL ""
#endif
#ifndef MPK_CPX_TF_AGENT // timing probe: __threadfence stays agent-scope
#define __threadfence() __threadfence_system()
#endif
#else
#define MPK_DEV_SC_INV " sc1"
#define MPK_DEV_SC_WB " sc1"
#define MPK_DEV_FENCE "agent"
#define MPK_DEV_FENCE_ACQ "agent"
#define MPK_DEV_FENCE_REL "agent"
#endif

// PERF TEST flags (uncomment to test):
// #define MPK_DISABLE_THREADFENCE  // Disable threadfence_gpu()
#define MPK_USE_RELAXED_ATOMICS
#define MPK_USE_NT_MEMORY // Use non-temporal loads/stores (bypass cache, fence
                          // for ordering)
// #define MPK_ENABLE_TIMING  // Enable timing instrumentation
// #define MPK_ENABLE_DEVICE_TASK_ACCUM  // Enable per-task-type wall-clock
// accumulation #define MPK_ENABLE_SPAN_TIMING  // Enable per-stage wall-clock
// span (first_start→last_end) #define MPK_DISABLE_LINEAR  // Disable LINEAR
// kernel entirely

// LINEAR kernel section flags (uncomment one at a time to find bottleneck):
// #define MPK_SKIP_LOAD_INPUT   // Skip loading input from global memory
// #define MPK_SKIP_LOAD_WEIGHT  // Skip loading weight from global memory
// #define MPK_SKIP_MFMA         // Skip MFMA computation
// #define MPK_SKIP_STORE_OUTPUT // Skip storing output to global memory

#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
#include <atomic>
#endif

// ---------------------------------------------------------------------------
// KV cache layout selector.
//
// Default NHD: (layers, pages, page_size, kv_heads, head_dim). Heads are an
// INNER dim, so head x is strided across the whole cache and cannot be placed
// in one AID -- which is why the kvupd kernel calls it an "un-partitioned
// base".
//
// MPK_KV_HEAD_MAJOR selects H*ND: (layers, kv_heads, pages, page_size,
// head_dim). Head x becomes contiguous, so halves placement puts heads 0-3 in
// range 0 and 4-7 in range 1, and since attn_kv_head == xcd_id each XCD then
// reads only its own AID.
//
// Every KV address in the tree is `tok * tok_stride + head * head_stride`.
// Only the token stride was ever a parameter; these two macros make both
// strides explicit so one expression serves either layout.
#define MPK_KV_TOTAL_TOKENS ((long long)MPK_MAX_NUM_PAGES * MPK_PAGE_SIZE)
#ifdef MPK_KV_HEAD_MAJOR
#define MPK_KV_HEAD_OFF(h, hd) ((size_t)(h) * (size_t)MPK_KV_TOTAL_TOKENS * (hd))
#define MPK_KV_TOK_STRIDE(hd, kvs) (hd)
#else
#define MPK_KV_HEAD_OFF(h, hd) ((size_t)(h) * (size_t)(hd))
#define MPK_KV_TOK_STRIDE(hd, kvs) (kvs)
#endif

#ifdef MPK_AID_SPLIT_FLAGS
// The two COHERENT AID-local flag replicas, [0] homed in AID0 and [1] in AID1,
// each holding eight per-XCD flags one 64 B line apart. Populated once by the
// host before the first launch (see alloc_flag_replicas in aid_local.h) and
// re-zeroed per launch, exactly where the shared counter tensors are.
//
// A null pair means the allocator was unavailable, in which case every site
// guarded by MPK_AID_SPLIT_FLAGS falls back to the shared buffer, so a build
// with this flag on still runs correctly (just without the win) on a stock
// driver or in NPS1.
#ifdef MPK_AID_REP_CONST
// Host-filled before launch, read-only on the device: __constant__ makes
// the loads scalar (lgkmcnt), so their waits cannot drain asm DMA loads.
#define MPK_AID_REP_SPACE __constant__
#else
#define MPK_AID_REP_SPACE __device__
#endif
MPK_AID_REP_SPACE int *g_aid_flag_rep[2];
#ifdef MPK_QEP_SNAP
// QKV-epoch arrival / exit stamps of every epoch participant for the 36 layers of iteration
// MPK_QEP_SNAP (worker = die * 31 + rank); printed per (layer, die) at terminate.
__device__ unsigned long long g_qep_snap[36 * 256 * 2];
// v2: poll start / observation stamps and spin counts of the polling participants.
__device__ unsigned long long g_qep_snap2[36 * 256 * 2];
__device__ unsigned int g_qep_spins[36 * 256];
#endif
#ifdef MPK_GATE_SNAP
// Layer-boundary stamps of iteration MPK_GATE_SNAP (worker = die * 31 + rank): [0] MoE return,
// [1] deferred-gate poll start, [2] gate observation, [3] after the gate block.
__device__ unsigned long long g_gs_snap[36 * 256 * 4];
#endif
// ROUTER_OPROJ_PARTIAL scratch: the AID replicas in NPS2, one plain buffer
// aliased twice otherwise (see launch_persistent_kernel).
MPK_AID_REP_SPACE int *g_rop_rep[2];
// AID-local NC scratch for counters: [0] in AID0, [1] in AID1 (halves of
// one plain allocation). Null unless MPK_AID_NC_REP set it up.
MPK_AID_REP_SPACE int *g_aid_nc_rep[2];
// Layout of each AID half of the NC scratch, in ints. The first 64 KB
// (16384 ints) are cleared per launch.
constexpr int MPK_NC_MOE_INTS = 0;      // one 256 B line per expert id (128)
constexpr int MPK_NC_P9_INTS = 8192;    // Phase 9, one 256 B line per die
constexpr int MPK_NC_OPROJ_INTS = 9216; // O-proj hier_local, HIER_STRIDE per die
constexpr int MPK_NC_ATTNREL_INTS = 13312;  // MPK_VT2_SLICE_REL: one 128 B line per die
static_assert(12288 + 8 * 64 <= MPK_NC_ATTNREL_INTS /* past MPK_NC_AMX_INTS */ &&
                  MPK_NC_ATTNREL_INTS + 8 * 32 <= 14336 /* E3_NC_INTS */,
              "MPK_NC_ATTNREL_INTS overlaps a neighbouring NC region");
static_assert(MPK_NC_MOE_INTS + 128 * 64 <= MPK_NC_P9_INTS &&
                  MPK_NC_P9_INTS + 8 * 64 <= MPK_NC_OPROJ_INTS &&
                  MPK_NC_OPROJ_INTS + 8 * 16 <= 16384,
              "NC scratch regions overlap or leave the per-launch clear");
#ifdef MPK_D89_RT
#if !defined(MPK_MOE_W2_REMAP3) || defined(MPK_MOE_W2_REMAP3_NODELAY89)
#error "MPK_D89_RT: ranks 8-9 are W2-only movers only under MPK_MOE_W2_REMAP3 (without NODELAY89)"
#endif
// W2 entry delay of ranks 8-9 in 10 ns ticks (host: env MPK_D89_NS, default 4000 ns).
__constant__ unsigned g_mpk_d89_ticks = 400u;
#endif
#ifdef MPK_W2_MALLPF
// W2-only movers prefetch the dual ranks' W2 tiles into the MALL (host: env MPK_W2PF = 0 off / 1 on,
// MPK_W2PF_NS = start offset after the mover's MoE entry).
__constant__ unsigned g_mpk_w2pf_mode = 0u;
__constant__ unsigned g_mpk_w2pf_ticks = 0u;
#endif
#ifdef MPK_PFD89_RT
// Delay of ranks 8-9's Phase 9 weight DMA in 10 ns ticks (host: env MPK_PFD89_NS, default 0).
__constant__ unsigned g_mpk_pfd89_ticks = 0u;
#endif
#if defined(MPK_VT2_QKV_EARLY) && !defined(MPK_KVES_LO)
#define MPK_KVES_LO 107264  // same bound: the W2 window ends below it
#endif
#ifdef MPK_KV_EARLY_STAGE
#ifndef MPK_KVES_LO
// First LDS byte the K/V ranks' early chunks may land at (the W2 window ends below it).
#define MPK_KVES_LO 107264
#endif
// Host: env MPK_KVES (default 0 = full Phase 9 prefetch on every rank).
__constant__ int g_mpk_kves = 0;
#endif
#ifdef MPK_KSEL_MERGE
// Runtime switch (host: env MPK_KMRG_RT, default 1): 0 runs the four selection rounds.
#ifdef MPK_RT_CONST
static constexpr int g_mpk_kmrg_rt = 1;
#else
__constant__ int g_mpk_kmrg_rt = 1;
#endif
#endif
#ifdef MPK_ROP_QUANT_FAST
// Runtime switch (host: env MPK_RQF_RT, default 1): 0 runs the original quantization.
#ifdef MPK_RT_CONST
static constexpr int g_mpk_rqf_rt = 1;
#else
__constant__ int g_mpk_rqf_rt = 1;
#endif
#endif
#ifdef MPK_LTK_STATS_W0
// Runtime switch (host: env MPK_STW0_RT, default 1): 0 keeps the statistics on wave 2.
#ifdef MPK_RT_CONST
static constexpr int g_mpk_stw0_rt = 1;
#else
__constant__ int g_mpk_stw0_rt = 1;
#endif
#endif
#ifdef MPK_LTK_STW0_FAST
// Runtime switch (host: env MPK_STWF_RT, default 1): 0 keeps the reduction and the nops.
#ifdef MPK_RT_CONST
static constexpr int g_mpk_stwf_rt = 1;
#else
__constant__ int g_mpk_stwf_rt = 1;
#endif
#endif
#ifdef MPK_ROP_ROW_EARLY
// Runtime switch (host: env MPK_RRE_RT, default 1): 0 issues the row loads after the barrier.
#ifdef MPK_RT_CONST
static constexpr int g_mpk_rre_rt = 1;
#else
__constant__ int g_mpk_rre_rt = 1;
#endif
#endif

#ifdef MPK_ARGMAX_FUSED
#ifndef MPK_AID_NC_REP
#error "MPK_ARGMAX_FUSED keeps its argmax keys in the MPK_AID_NC_REP scratch"
#endif
// One packed argmax key per die, 256 B apart, in the die's AID half.
constexpr int MPK_NC_AMX_INTS = 12288;
static_assert(MPK_NC_OPROJ_INTS + 8 * 16 <= MPK_NC_AMX_INTS &&
                  MPK_NC_AMX_INTS + 8 * 64 <= 16384,
              "argmax keys overlap another region or leave the per-launch clear");
// Runtime switch (host: env MPK_AMX_RT, default 1): 0 runs the original reduce.
#ifdef MPK_RT_CONST
static constexpr int g_mpk_amx_rt = 1;
#else
__constant__ int g_mpk_amx_rt = 1;
#endif
// The reduce's winner order among equal values: its first-stage thread is the
// partial index, each wave reduces with __shfl_down keeping the left lane on
// ties (smallest 6-bit bit-reversed lane wins), then wave 0 reduces the four
// wave results the same way (waves rank 0, 2, 1, 3).
__device__ __forceinline__ unsigned mpk_amx_tie_rank(int i) {
  unsigned const l = (unsigned)i & 63u;
  unsigned const w = ((unsigned)i >> 6) & 3u;
  unsigned const rl = ((l & 1u) << 5) | ((l & 2u) << 3) | ((l & 4u) << 1) |
                      ((l & 8u) >> 1) | ((l & 16u) >> 3) | ((l & 32u) >> 5);
  unsigned const rw = ((w & 1u) << 1) | ((w & 2u) >> 1);
  return (rw << 6) | rl;
}
// High 16 bits: the bf16 value mapped so u32 order is float order, -0 folded
// onto +0, NaN and -inf at 0 (the reduce never selects them). Next 16: the
// inverted tie rank. Low 32: the absolute index.
__device__ __forceinline__ unsigned long long mpk_amx_key(unsigned b, int slot,
                                                          int idx) {
  unsigned o;
  if ((b & 0x7FFFu) > 0x7F80u || b == 0xFF80u) {
    o = 0u;
  } else {
    b = b == 0x8000u ? 0u : b;
    o = (b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u);
  }
  return ((unsigned long long)o << 48) |
         ((unsigned long long)(0xFFFFu - mpk_amx_tie_rank(slot)) << 32) |
         (unsigned long long)(unsigned)idx;
}
#endif

// This die's AID half of the NC scratch at a raw int offset, picked by the
// physical XCC id; falls back to shared_base when the scratch is absent.
__device__ __forceinline__ int *mpk_aid_nc_hw(int *shared_base, int off_ints) {
  int xcc;
  MPK_GET_XCC_ID16(xcc);
  int *rep = g_aid_nc_rep[(xcc & 7) >> 2];
  return rep ? rep + off_ints : shared_base;
}

// Per-AID copies of the multi-layer pointer tables. Every worker re-reads its
// TaskDesc's input/output pointers out of these at every layer boundary, and
// the tables are indexed with xcd_id outermost, so each XCD only ever touches
// its own 1/8 of them. They are read-only after setup and total ~90 KiB, which
// makes them the opposite profile to the MoE weights: tiny, high-fanout, and
// squarely on the critical path at the start of every layer.
MPK_AID_REP_SPACE void **g_aid_ml_in[2];
MPK_AID_REP_SPACE void **g_aid_ml_out[2];

// Each flag family gets its own eight lines in both replicas. All of them are
// cleared per launch, because the shared lines they mirror all live in
// oproj_topk_counters, which demo.py clears wholesale between launches.
constexpr int MPK_AID_REGION_INTS = 8 * 16; // eight 64 B lines
constexpr int MPK_AID_REGION_ATTN_RELEASE = 0;

constexpr int MPK_AID_REGION_LAYER_RELEASE = 1; // reserved, see slot 10 note
constexpr int MPK_AID_REGION_OPROJ_READY = 2; // MPK_ROUTER_XCD_FOLD only
// Only the eight per-XCD release flags of the O-proj hierarchical barrier.
// The global arrival counter at [8*16], topk_counter at [9*16] and the
// XCD-local tree at [28*16] stay shared: they are atomics, and those are
// XCD-L2-bound rather than AID-bound, so placement does nothing for them.
constexpr int MPK_AID_REGION_HIER_RELEASE = 3;
// qkv_epoch's eight per-XCD slots. Unlike the release families this one is
// never read across the AID boundary: the arrival and both polls all address
// `[xcd_id * 16]`, so only the AID running that XCD ever wants the value.
// Hence a single replica write rather than a dual publish, and the arrival
// atomic lands on a line homed in the AID performing it.
constexpr int MPK_AID_REGION_QKV_EPOCH = 4;
// The O-proj barrier's per-XCD ARRIVAL tree. Like qkv_epoch this is never read
// across the AID boundary -- the file calls level 1 "XCD-private, written and
// read only at [xcd_id]" -- so one replica write, read locally, no dual
// publish. Regions 0..5 use 6*128 = 768 ints, still under MPK_AID_MOE_BASE_INTS.
constexpr int MPK_AID_REGION_HIER_LOCAL = 5;
// Per-XCD arrival counters for the top-k barrier. Region 6 keeps
// 7*128 = 896 ints, still below MPK_AID_MOE_BASE_INTS (1024).
constexpr int MPK_AID_REGION_TOPK = 6;
// MPK_OPROJ_AID_TIER: per-AID arrival count of the O-proj phase-2 barrier.
// Slot 0 only; touched by the four XCDs of the AID that owns the replica.
constexpr int MPK_AID_REGION_OPROJ_TIER = 7;
#ifdef MPK_LOCAL_TOPK
#ifndef MPK_MOE_XCD_PAIR
#error "MPK_LOCAL_TOPK needs MPK_MOE_XCD_PAIR (the TopK publishes picks, not the per-expert mask)"
#endif
// Tagged router logits, 128 ints inside the routing region of each replica.
// mpk_ltk_tag() and the per-workgroup s_ltk_* state are declared with the
// local-TopK slot layout further down.
constexpr int MPK_AID_LTK_OFF_INTS = 1024;
// MPK_LTK_EARLY_TAG: per-XCD normed-row flags (32 ints per hardware XCD),
// written and polled only in that XCD's own AID replica.
constexpr int MPK_AID_LTK_NORM_OFF_INTS = MPK_AID_LTK_OFF_INTS + 128;
#endif

// The MoE fused barrier is per-expert (MOE_BAR_STRIDE ints each), so at 128
// experts it needs ~20k ints rather than the eight lines the families above
// use. It gets a raw offset past them instead of a region index. The 2 MiB
// replica has room to spare.
constexpr int MPK_AID_MOE_BASE_INTS = 1024;
// XCD-private barrier counters past the MoE region (~21.5k ints): QKV epoch
// arrivals, one line per XCD (MPK_AID_QKV_ARRIVE), and split-KV chunk
// arrivals, one line per head x request (MPK_AID_CHUNK_BAR).
constexpr int MPK_AID_QKVARR_BASE_INTS = 24576;
constexpr int MPK_AID_CHUNKBAR_BASE_INTS = 24832;

// routing_ready's eight per-XCD release flags. Own raw offset because the family
// spans nine lines (an epoch counter at [0] plus [(1+x)*16] for x in 0..7), one
// more than a region slot holds. Past the MoE family, which runs to ~21.5k ints;
// the 2 MiB replica is 524k ints.
//
// ONLY the per-XCD flags move. The epoch counter at [0] stays shared: it is read
// as `ld_nt_s32(routing_ready) + 1` to DERIVE what the poll waits for, and a
// stale replica there yields an expected value nobody ever publishes -- the
// MPK_LD_EPOCH hazard, a hang rather than a slowdown. Same split the MoE barrier
// and the O-proj hier barrier already use: replicate the flags, share the
// counter.
//
// Why this matters in SPX+NPS2. gang_oproj_topk_moe_fused_mi300.cuh:190 claims
// the poll is "XCD-local (hot in L2, no cross-XCD contention)". That holds in
// NPS1, where the line is MTYPE_RW and cacheable. In NPS2 plain hipMalloc is
// demoted to MTYPE_NC, nothing can be shared through cache, and every poller's
// `sc0 sc1` read goes to the coherency point and serialises. Measured with
// a release-visibility microbenchmark, last-of-7-acks against pollers on ONE line:
//   NPS1  8->2565 ns   64->2561   128->2457   256->2446   (flat)
//   NPS2  8->2426 ns   64->5060   128->26620  256->52251  (21x)
// An AID-local RW replica is flat in NPS2 too (2470-2502), which is the fix.
//
// MEASURED: correct but end-to-end NEUTRAL, so MPK_AID_SPLIT_ROUTING ships off.
// It does move the right numbers -- slot 7 topk_wait 11618 -> 10750, slot 6
// 18469 -> 17334, layer total 76854 -> 75883 ns -- and it does not hang, so the
// replication is sound. It is just ~1 us of a ~25 us gap, because this poll is
// dominated by WAITING FOR TOPK TO COMPUTE rather than by poll serialization;
// gang_linear_...:2757 already says so ("that 5.8 us is the TopK compute the
// release is waiting for, not the release's own latency"). Keep it for the
// correctness of the mechanism and as the record that the NC-poll cliff, though
// real in isolation (21x in relobs3), is NOT what is left in fleet: the other
// three flag families were already replicated.
constexpr int MPK_AID_ROUTING_BASE_INTS = 32768;

// The fused-tail barriers of the LM head task (moe_done, resadd_done,
// lmhead_done). Own offset past routing; the 2 MiB replica is 524k ints and
// the per-launch memset clears it, so these mirrors start at 0 every launch.
// One line each so they never share a cache line.
constexpr int MPK_AID_LMTAIL_BASE_INTS = 40960;

// swiglu_out's two copies, inside the same per-AID replicas the split flags
// use. 5888 ints each (NUM_TOPK * PADDED_INTERMEDIATE_SIZE shorts), clear of
// the flag regions below 1024, LMTAIL at 40960 and EVCTR at 458752.
constexpr int MPK_AID_SWIGLU_BASE_INTS = 200704;
// Layer-gate L2 invalidate election (MPK_P9_LEADER_INV): per physical XCC,
// a claim word at +xcc*32 and the published epoch at +xcc*32+16. Both
// are touched by one XCD only, clear of SWIGLU and ATTNOUT.
constexpr int MPK_AID_L2CLEAN_BASE_INTS = 294912;
// W13's prequantized activation handoff row (MPK_W13_HOF_AIDREP): one copy
// in each AID's replica, stored write-through by the router's norm writers
// and read with sc0 sc1 by every W13 tile on that AID. One 5888 B row at
// bs=1, clear of SWIGLU (200704..206592) and L2CLEAN.
constexpr int MPK_AID_W13HOF_BASE_INTS = 262144;
// MoE f32 output workspace (MPK_WSF32_REP): one copy per AID, stored
// write-through by every W2 tile into both and read by the next layer's QKV
// prologue from its own AID's copy, so that read needs no L2 invalidate.
// 24576 ints (96 KiB), clear of L2CLEAN (294912 + 8 * 32) and ATTNOUT (327680).
constexpr int MPK_AID_WSF32_BASE_INTS = 303104;
constexpr int MPK_AID_WSF32_INTS = 24576;
#ifdef MPK_WSF32_REP
#ifndef MPK_AID_SPLIT_FLAGS
#error "MPK_WSF32_REP needs the MPK_AID_SPLIT_FLAGS replicas"
#endif
#ifdef MPK_WSF32_AID
#error "MPK_WSF32_REP replaces MPK_WSF32_AID's torch-doubled copy"
#endif
#endif
__device__ __forceinline__ float *mpk_wsf32_rep(int aid) {
  return (g_aid_flag_rep[0] != nullptr && g_aid_flag_rep[1] != nullptr)
             ? (float *)(g_aid_flag_rep[aid] + MPK_AID_WSF32_BASE_INTS)
             : nullptr;
}
#ifdef MPK_W13_HOF_AIDREP
#ifndef MPK_AID_SPLIT_FLAGS
#error "MPK_W13_HOF_AIDREP needs the MPK_AID_SPLIT_FLAGS replicas"
#endif
#ifdef MPK_W13_HOF_XCDREP
// Each die writes and reads only its own copy: plain stores stay in
// its L2, where its W13 tiles find them.
#define MPK_HOF_STORE_MOD ""
#else
#define MPK_HOF_STORE_MOD " sc0 sc1"
#endif
#else
#define MPK_HOF_STORE_MOD ""
#endif
// Offset of this die's private copy under MPK_W13_HOF_XCDREP (8 KiB per
// die, four dies per AID replica: 262144..270336, clear of L2CLEAN).
constexpr int MPK_AID_W13HOF_COPY_INTS = 2048;
__device__ __forceinline__ int mpk_hof_copy_ints() {
#ifdef MPK_W13_HOF_XCDREP
  int x;
  MPK_GET_XCC_ID16(x);
  return (x & 3) * MPK_AID_W13HOF_COPY_INTS;
#else
  return 0;
#endif
}
// AID of the physical die; the producer and the consumers must agree.
__device__ __forceinline__ int mpk_hof_aid() {
  int x;
  MPK_GET_XCC_ID16(x);
  return (x & 7) >> 2;
}
#if defined(MPK_W13_HOF_AIDREP) && defined(MPK_W13_PREQUANT)
#define MPK_MOE_HOF_IN(torch_in)                                               \
  ((g_aid_flag_rep[0] != nullptr && g_aid_flag_rep[1] != nullptr)              \
       ? (void const *)(g_aid_flag_rep[mpk_hof_aid()] +                        \
                        MPK_AID_W13HOF_BASE_INTS + mpk_hof_copy_ints())        \
       : (void const *)(torch_in))
#else
#define MPK_MOE_HOF_IN(torch_in) (torch_in)
#endif
constexpr int MPK_AID_LMTAIL_MOE = 0;
constexpr int MPK_AID_LMTAIL_RESADD = 16;
constexpr int MPK_AID_LMTAIL_LMHEAD = 32;

// Replica view at a raw int offset, for families too big for a region slot.
__device__ __forceinline__ int *
    mpk_aid_flags_at(int *shared_base, int xcd_id, int off_ints) {
  int *rep = g_aid_flag_rep[xcd_id >> 2];
  return rep ? rep + off_ints : shared_base;
}

// The replica this XCD should poll. SPX maps XCD x to AID x>>2.
__device__ __forceinline__ int *
    mpk_aid_flags(int *shared_base, int xcd_id, int region) {
  int *rep = g_aid_flag_rep[xcd_id >> 2];
  return rep ? rep + region * MPK_AID_REGION_INTS : shared_base;
}

// Same, but reading the physical XCC id rather than trusting a caller-derived
// one. The O-proj kernel computes its `xcd_id` as `tile_idx / tiles_per_xcd`,
// which is the *intended* binding rather than the hardware's -- and picking
// the wrong replica here sends every reader to the far AID, which is a silent
// slowdown rather than a failure. Cheap enough to just ask.
__device__ __forceinline__ int *
    mpk_aid_flags_hw(int *shared_base, int region) {
  int xcc;
  MPK_GET_XCC_ID16(xcc);
  int *rep = g_aid_flag_rep[(xcc & 7) >> 2];
  return rep ? rep + region * MPK_AID_REGION_INTS : shared_base;
}
#endif

// =============================================================================
// Non-Temporal (NT) Load/Store for MI300X
// =============================================================================
//
// NT loads/stores bypass both L1 and L2 caches, reading/writing directly
// to HBM memory. This avoids relying on cache coherence protocol.
//
// Assembly flags:
//   global_load_dwordx2 %0, %1, off nt
//   global_store_dwordx2 %0, %1, off nt
//
// On MI300X (gfx942), the 'nt' (non-temporal) flag sets:
//   - L1: sc1=1 or NT=1 → MISS EVICT (bypasses L1)
//   - L2: sc0=1 or NT=1 → HIT STREAM / Cache Bypass
//
// Why fence is still needed with NT:
//   NT provides cache bypass but NOT memory ordering. Without fence:
//   1. Hardware can reorder NT loads (read data before flag)
//   2. Compiler can reorder code (asm volatile helps but isn't enough)
//
// The pattern is:
//   Producer: st_nt(data) → fence → st_nt(flag)
//   Consumer: ld_nt(flag) → fence → ld_nt(data)
//
// The fence (SEQ_CST) ensures:
//   - Producer's data write completes before flag write
//   - Consumer's flag read completes before data read
//
// Performance: NT + fence ≈ RELAXED atomics + fence (~17.4ms per token)
// Both are valid approaches; NT is more explicit about bypassing cache.
// =============================================================================

// Non-temporal load (bypasses cache, reads from memory)
__device__ __forceinline__ unsigned long long int
    ld_nt_u64(unsigned long long int *addr) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  unsigned long long int val;
  asm volatile("global_load_dwordx2 %0, %1, off nt\n"
               "s_waitcnt vmcnt(0)"
               : "=v"(val)
               : "v"(addr)
               : "memory");
  return val;
#else
  // NVIDIA: use volatile for non-cached access
  return *reinterpret_cast<unsigned long long int volatile *>(addr);
#endif
}

// Non-temporal 32-bit load (bypasses cache, reads from memory)
// Publish a release flag: write-through, and raise-only by construction.
//
// It must be write-through. These counters are plain hipMalloc VRAM, which is
// MTYPE_NC under SPX/DPX+NPS2 -- non-coherent across XCDs. An atomic (even at
// system scope) can settle in the publishing XCD's L2, while every poller
// reads the flag with ld_sys_s32, an `sc0 sc1` load that goes past L2 to
// memory. The publisher then sees its own value and nobody else does: measured
// as O-proj flags [8,8,7,8] read from two dies while the die owning flag 2 had
// already moved past that same barrier.
//
// Monotonicity therefore cannot come from an atomic max, and does not need to:
// each flag has a single writer (its own XCD's last arrival), which publishes
// once per layer in increasing epoch order, drained by s_waitcnt before the
// store. A blind store only moves a flag backwards when several workgroups
// broadcast into the same flag, which the per-XCD publish removes.
__device__ __forceinline__ void st_rel_max_u32(void *addr, unsigned val) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  asm volatile("global_store_dword %0, %1, off sc0 sc1"
               :
               : "v"(addr), "v"(val)
               : "memory");
#else
  *reinterpret_cast<unsigned int volatile *>(addr) = val;
#endif
}

__device__ __forceinline__ int ld_nt_s32(int *addr) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  int val;
  asm volatile("global_load_dword %0, %1, off nt\n"
               "s_waitcnt vmcnt(0)"
               : "=v"(val)
               : "v"(addr)
               : "memory");
  return val;
#else
  return *reinterpret_cast<int volatile *>(addr);
#endif
}

// System-scope 32-bit load for release polls.
//
// `nt` is only an eviction-policy hint: it does not force the load to miss
// vL1, so a spin built on ld_nt_s32 keeps re-reading whatever line it already
// has until that line happens to be evicted. `sc0 sc1` is the real
// device-coherency read, which is what a release flag written by another XCD
// needs. Every release gate in this kernel polls this way.
__device__ __forceinline__ int ld_sys_s32(int *addr) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  int val;
  asm volatile("global_load_dword %0, %1, off sc0 sc1\n"
               "s_waitcnt vmcnt(0)"
               : "=v"(val)
               : "v"(addr)
               : "memory");
  return val;
#else
  return *reinterpret_cast<int volatile *>(addr);
#endif
}

#ifdef MPK_POLL_PIPE
// Waits until *p >= target with two system-scope loads in flight, issued
// about half a round trip apart, so a release is seen up to half a trip
// sooner than with one load per trip. The first check is not delayed: the
// first load's round trip is longer than the stagger.
__device__ __forceinline__ void mpk_poll_ge_s32(int *p, int target) {
  int a = __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  __builtin_amdgcn_s_sleep(10);
  int b = __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  while (true) {
    if (a >= target) {
      return;
    }
    a = __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    if (b >= target) {
      return;
    }
    b = __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

// Same for two monotonic flags that must both reach `target`: each sample
// reads both, and two samples are in flight.
__device__ __forceinline__ void mpk_poll_ge2_s32(int *p0, int *p1, int target) {
  int a0 = __hip_atomic_load(p0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  int a1 = __hip_atomic_load(p1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  __builtin_amdgcn_s_sleep(10);
  int b0 = __hip_atomic_load(p0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  int b1 = __hip_atomic_load(p1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  while (true) {
    if (a0 >= target && a1 >= target) {
      return;
    }
    a0 = __hip_atomic_load(p0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    a1 = __hip_atomic_load(p1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    if (b0 >= target && b1 >= target) {
      return;
    }
    b0 = __hip_atomic_load(p0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    b1 = __hip_atomic_load(p1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}
#endif

__device__ __forceinline__ unsigned long long ld_sys_u64(void *addr) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  unsigned long long val;
  asm volatile("global_load_dwordx2 %0, %1, off sc0 sc1\n"
               "s_waitcnt vmcnt(0)"
               : "=v"(val)
               : "v"(addr)
               : "memory");
  return val;
#else
  return *reinterpret_cast<unsigned long long volatile *>(addr);
#endif
}

// Two independent system-scope polls, one wait. `ld_sys_s32` serializes each
// load behind its own vmcnt(0), so a two-flag spin pays two HBM round trips
// per miss. Issue both, then one wait.
__device__ __forceinline__ void ld_sys_s32x2(int *addr0, int *addr1, int &out0,
                                             int &out1) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  int v0, v1;
  asm volatile("global_load_dword %0, %2, off sc0 sc1\n"
               "global_load_dword %1, %3, off sc0 sc1\n"
               "s_waitcnt vmcnt(0)"
               : "=&v"(v0), "=&v"(v1)
               : "v"(addr0), "v"(addr1)
               : "memory");
  out0 = v0;
  out1 = v1;
#else
  out0 = *reinterpret_cast<int volatile *>(addr0);
  out1 = *reinterpret_cast<int volatile *>(addr1);
#endif
}

// The poll idiom the release gates use. MPK_SYS_POLL_LOAD swaps every gate
// from the `nt` hint to a true system-scope read in one place, so the two can
// be compared without touching each site.
// It is a level, not a boolean: 1 converts the three fused-layer gates
// (qkv_epoch / attn_release / routing_ready / layer_release), 2 additionally
// converts the two per-XCD barriers inside MoE and O-proj. The second set
// polls a line its own XCD produced, so it is a weaker case for sc1 and worth
// measuring apart from the first.
#if defined(MPK_SYS_POLL_LOAD) && MPK_SYS_POLL_LOAD >= 1
#define MPK_LD_GATE(p) ld_sys_s32(p)
#else
#define MPK_LD_GATE(p) ld_nt_s32(p)
#endif

#if defined(MPK_SYS_POLL_LOAD) && MPK_SYS_POLL_LOAD >= 2
#define MPK_LD_GATE2(p) ld_sys_s32(p)
#else
#define MPK_LD_GATE2(p) ld_nt_s32(p)
#endif


// ---------------------------------------------------------------------------
// AID-local gate load: vL1-bypassing but L2-CACHEABLE.
//
// ld_sys_s32 (`sc0 sc1`) goes past L2 to memory on every spin. Its stated
// reason is that the flags are "plain hipMalloc VRAM, which is MTYPE_NC" --
// true before MPK_AID_SPLIT_FLAGS, false after it. A replica lives in an
// alloc_in_aid BO at mtype_local=0 = MTYPE_RW, which is hardware-coherent
// inside its own AID, so the publisher's write-through store probes this
// reader's L2 copy out and a cached load cannot go stale.
//
// SCOPE per AMD MI300 ISA tables 48/49 (MLSE "Understand cache modifiers").
// SC1:SC0 is a two-bit SCOPE VALUE, not two independent flags:
//   0 wave | 1 group | 2 device | 3 system
// and the two caches are governed by DIFFERENT bits:
//   L1 (TCP, per-CU, read-only): sc1=1 or nt=1 -> MISS_EVICT, else HIT_LRU
//   L2 (TCC):                    sc0=1 or nt=1 -> Coherence_Cache_Bypass,
//                                sc0=0 and nt=0 -> HIT_LRU
// So `sc0 sc1` (system) bypasses L2 BY CONSTRUCTION -- every poll is a memory
// round trip. `sc1` alone is device scope with L1 MISS_EVICT and L2 HIT_LRU,
// which is the weakest scope that is still coherent across XCDs.
// `sc0` alone is GROUP scope, weaker than device: it was tried first and HUNG
// (0 forward passes, hash d41d8cd98f00, EXIT=137) because readers kept
// serving a stale line.
// `nt` is only an eviction hint and never observes a release at all.
//
// The MTYPE still matters independently: L2 HIT_LRU only helps if the line's
// MTYPE gives coherent L2 at device scope, i.e. an AID-local MTYPE_RW replica
// rather than spanning MTYPE_NC.
// Two device-scope polls in flight at once, one wait: the min of both
// words. Used where a waiter needs two AIDs' publishes, so detecting the
// second costs one load latency rather than two.
__device__ __forceinline__ int ld_aid_min2_s32(int *a, int *b) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  int x, y;
  asm volatile("global_load_dword %0, %2, off sc1\n"
               "global_load_dword %1, %3, off sc1\n"
               "s_waitcnt vmcnt(0)"
               : "=&v"(x), "=&v"(y)
               : "v"(a), "v"(b)
               : "memory");
  return x < y ? x : y;
#else
  int x = __atomic_load_n(a, __ATOMIC_ACQUIRE);
  int y = __atomic_load_n(b, __ATOMIC_ACQUIRE);
  return x < y ? x : y;
#endif
}

__device__ __forceinline__ int ld_aid_s32(int *addr) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  int val;
  asm volatile("global_load_dword %0, %1, off sc1\n"
               "s_waitcnt vmcnt(0)"
               : "=v"(val)
               : "v"(addr)
               : "memory");
  return val;
#else
  return *reinterpret_cast<int volatile *>(addr);
#endif
}

// MPK_W2_SPOLL: scalar-path poll. s_load glc misses the scalar cache and reads L2; it is waited
// on lgkmcnt, so it neither waits for the wave's outstanding vector loads nor queues behind the
// CU's. The address must be wave-uniform.
__device__ __forceinline__ int ld_scalar_glc_s32(int const *addr) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  unsigned long long const a = reinterpret_cast<unsigned long long>(addr);
  unsigned long long const u =
      ((unsigned long long)(unsigned)__builtin_amdgcn_readfirstlane((unsigned)(a >> 32)) << 32) |
      (unsigned long long)(unsigned)__builtin_amdgcn_readfirstlane((unsigned)a);
  int val;
  asm volatile("s_load_dword %0, %1, 0x0 glc\n"
               "s_waitcnt lgkmcnt(0)"
               : "=s"(val)
               : "s"(u)
               : "memory");
  return val;
#else
  return *reinterpret_cast<int const volatile *>(addr);
#endif
}

__device__ __forceinline__ unsigned long long ld_aid_u64(void *addr) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  unsigned long long val;
  asm volatile("global_load_dwordx2 %0, %1, off sc1\n"
               "s_waitcnt vmcnt(0)"
               : "=v"(val)
               : "v"(addr)
               : "memory");
  return val;
#else
  return *reinterpret_cast<unsigned long long volatile *>(addr);
#endif
}

// Minimum of eight 64 B-strided flags in one round trip: all eight loads are
// issued before the single wait (ld_aid_s32 waits inside its own asm, so eight
// calls would serialise into eight round trips per poll).
#if defined(MPK_AID_GATE_CACHED)
#define MPK_MIN8_SC "sc1"
#else
#define MPK_MIN8_SC "sc0 sc1"
#endif
__device__ __forceinline__ int ld_aid_min8_s32(int *base) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  int v0, v1, v2, v3, v4, v5, v6, v7;
  asm volatile("global_load_dword %0, %8, off " MPK_MIN8_SC "\n"
               "global_load_dword %1, %8, off offset:64 " MPK_MIN8_SC "\n"
               "global_load_dword %2, %8, off offset:128 " MPK_MIN8_SC "\n"
               "global_load_dword %3, %8, off offset:192 " MPK_MIN8_SC "\n"
               "global_load_dword %4, %8, off offset:256 " MPK_MIN8_SC "\n"
               "global_load_dword %5, %8, off offset:320 " MPK_MIN8_SC "\n"
               "global_load_dword %6, %8, off offset:384 " MPK_MIN8_SC "\n"
               "global_load_dword %7, %8, off offset:448 " MPK_MIN8_SC "\n"
               "s_waitcnt vmcnt(0)"
               : "=&v"(v0), "=&v"(v1), "=&v"(v2), "=&v"(v3), "=&v"(v4),
                 "=&v"(v5), "=&v"(v6), "=&v"(v7)
               : "v"(base)
               : "memory");
  int a = v0 < v1 ? v0 : v1;
  int b = v2 < v3 ? v2 : v3;
  int c = v4 < v5 ? v4 : v5;
  int d = v6 < v7 ? v6 : v7;
  a = a < b ? a : b;
  c = c < d ? c : d;
  return a < c ? a : c;
#else
  int m = reinterpret_cast<int volatile *>(base)[0];
  for (int i = 1; i < 8; ++i) {
    int v = reinterpret_cast<int volatile *>(base)[i * 16];
    m = v < m ? v : m;
  }
  return m;
#endif
}

// MPK_PIPE_POLL: wait until min(*p0, *p1) >= t with three samples (two loads each) in flight,
// issued one s_sleep apart and each reissued as soon as it lands and fails, so a release is seen
// within about a third of a round trip instead of within a whole serial poll period. Loads and
// waits sit in one asm statement (the compiler cannot reuse an in-flight destination); the
// vmcnt(4) waits also cover anything the wave issued before the call, which lands first.
#define MPK_PIPE_GE2_ASM(SC)                                                   \
  asm volatile("global_load_dword %[a0], %[p0], off " SC "\n\t"                \
               "global_load_dword %[a1], %[p1], off " SC "\n\t"                \
               "s_sleep 1\n\t"                                                 \
               "global_load_dword %[b0], %[p0], off " SC "\n\t"                \
               "global_load_dword %[b1], %[p1], off " SC "\n\t"                \
               "s_sleep 1\n\t"                                                 \
               "global_load_dword %[c0], %[p0], off " SC "\n\t"                \
               "global_load_dword %[c1], %[p1], off " SC "\n\t"                \
               "MPK_PG2_L%=:\n\t"                                              \
               "s_waitcnt vmcnt(4)\n\t"                                        \
               "v_min_i32 %[a0], %[a0], %[a1]\n\t"                             \
               "v_cmp_le_i32 vcc, %[t], %[a0]\n\t"                             \
               "s_cbranch_vccnz MPK_PG2_D%=\n\t"                               \
               "global_load_dword %[a0], %[p0], off " SC "\n\t"                \
               "global_load_dword %[a1], %[p1], off " SC "\n\t"                \
               "s_waitcnt vmcnt(4)\n\t"                                        \
               "v_min_i32 %[b0], %[b0], %[b1]\n\t"                             \
               "v_cmp_le_i32 vcc, %[t], %[b0]\n\t"                             \
               "s_cbranch_vccnz MPK_PG2_D%=\n\t"                               \
               "global_load_dword %[b0], %[p0], off " SC "\n\t"                \
               "global_load_dword %[b1], %[p1], off " SC "\n\t"                \
               "s_waitcnt vmcnt(4)\n\t"                                        \
               "v_min_i32 %[c0], %[c0], %[c1]\n\t"                             \
               "v_cmp_le_i32 vcc, %[t], %[c0]\n\t"                             \
               "s_cbranch_vccnz MPK_PG2_D%=\n\t"                               \
               "global_load_dword %[c0], %[p0], off " SC "\n\t"                \
               "global_load_dword %[c1], %[p1], off " SC "\n\t"                \
               "s_branch MPK_PG2_L%=\n\t"                                      \
               "MPK_PG2_D%=:\n\t"                                              \
               "s_waitcnt vmcnt(0)"                                            \
               : [a0] "=&v"(a0), [a1] "=&v"(a1), [b0] "=&v"(b0),               \
                 [b1] "=&v"(b1), [c0] "=&v"(c0), [c1] "=&v"(c1)                \
               : [p0] "v"(p0), [p1] "v"(p1), [t] "v"(t)                        \
               : "memory", "vcc")
template <bool SYS>
__device__ __forceinline__ void mpk_pipe_ge2(int const *p0, int const *p1, int t) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  int a0, a1, b0, b1, c0, c1;
  if constexpr (SYS) {
    MPK_PIPE_GE2_ASM("sc0 sc1");
  } else {
    MPK_PIPE_GE2_ASM("sc1");
  }
#else
  while (*reinterpret_cast<int const volatile *>(p0) < t ||
         *reinterpret_cast<int const volatile *>(p1) < t) {
  }
#endif
}

// Used ONLY at polls whose pointer came from mpk_aid_flags*. Without the
// replicas those pointers fall back to the shared NC buffer, where a cached
// poll would be unsound -- hence the MPK_AID_SPLIT_FLAGS guard.
#if defined(MPK_AID_GATE_CACHED) && defined(MPK_AID_SPLIT_FLAGS)
#define MPK_LD_GATE_AID(p) ld_aid_s32(p)
#define MPK_LD_GATE_AID_U64(p) ld_aid_u64(p)
#else
#define MPK_LD_GATE_AID(p) MPK_LD_GATE(p)
#define MPK_LD_GATE_AID_U64(p) ld_sys_u64(p)
#endif

// The scheduler's event counters are polled by a relaxed load with no
// invalidate inside the loop -- the acquire fence runs only once it exits.
// Sound while L2 is device-coherent (MTYPE_RW, which plain hipMalloc gives in
// NPS1), but in SPX+NPS2 a spanning BO is MTYPE_NC: cacheable and NOT coherent
// across XCDs, so the waiter re-reads its own stale L2 copy forever and an
// increment performed on another XCD is never observed. Measured: converting
// this poll to sc0 sc1 did NOT clear the hang (4/6 still hung). The deadlock
// is the termination race behind MPK_TERM_RECHECK.
#ifdef MPK_NPS2_EVENT_POLL
#define MPK_LD_EVENT(p) ld_sys_u64((void *)(p))
#else
#define MPK_LD_EVENT(p)                                                        \
  __atomic_load_n(reinterpret_cast<unsigned long long *>(p), __ATOMIC_RELAXED)
#endif

// Scoped read of the event-accounting CONFIG (thresholds / trigger counts), as
// opposed to MPK_LD_EVENT which reads the COUNTER. Both matter: a stale counter
// only delays a poll, but a stale threshold makes two XCDs take different
// branches of the two-level accounting, so the global total never reaches the
// exact value the trigger tests for and the event never fires at all.
#ifdef MPK_NPS2_EVENT_POLL
#define MPK_LD_EVCFG(p) ld_sys_s32((int *)(p))
#else
#define MPK_LD_EVCFG(p) (*(p))
#endif

// Non-temporal store (bypasses cache, writes to memory)
__device__ __forceinline__ void st_nt_u64(unsigned long long int *addr,
                                          unsigned long long int val) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  asm volatile("global_store_dwordx2 %0, %1, off nt\n"
               "s_waitcnt vmcnt(0)"
               :
               : "v"(addr), "v"(val)
               : "memory");
#else
  // NVIDIA: use volatile for non-cached access
  *reinterpret_cast<unsigned long long int volatile *>(addr) = val;
#endif
}

// =============================================================================
// Write-Through (WT) Store for MI300X
// =============================================================================
// sc0=1, sc1=1 on stores → "Coherent_Cache_Bypass" in L2
// Data bypasses L2 entirely, writes directly to HBM/MALL.
// No buffer_wbl2 needed after WT stores — data is already in memory.
// Use s_waitcnt vmcnt(0) to ensure stores complete before signaling.

// Write-through 64-bit store (4x bf16 or 2x float)
__device__ __forceinline__ void st_wt_u64(void *addr,
                                          unsigned long long int val) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  asm volatile("global_store_dwordx2 %0, %1, off sc0 sc1"
               :
               : "v"(addr), "v"(val)
               : "memory");
#else
  *reinterpret_cast<unsigned long long int volatile *>(addr) = val;
#endif
}

// Write-through 128-bit zero store (4x dword = 16 bytes, sc0 sc1).
// Uses global_store_dwordx4 for 4x fewer store instructions vs st_wt_u32.
__device__ __forceinline__ void st_wt_zero128(void *addr) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  typedef unsigned int v4u32 __attribute__((ext_vector_type(4)));
  v4u32 zero = {0u, 0u, 0u, 0u};
  asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1"
               :
               : "v"(addr), "v"(zero)
               : "memory");
#else
  reinterpret_cast<unsigned int volatile *>(addr)[0] = 0;
  reinterpret_cast<unsigned int volatile *>(addr)[1] = 0;
  reinterpret_cast<unsigned int volatile *>(addr)[2] = 0;
  reinterpret_cast<unsigned int volatile *>(addr)[3] = 0;
#endif
}

// Write-through 128-bit store (4x float), sc0 sc1.
// Same instruction as st_wt_zero128 but with a caller-supplied value. Needed
// wherever one XCD produces f32 data that another XCD consumes: a plain store
// stops in the producing XCD's L2, which is NOT coherent across XCDs (see
// threadfence_gpu below), so the consumer reads whatever its own L2 holds.
// `addr` must be 16-byte aligned.
#ifdef MPK_WS_FARCOPY
// Marks a workspace slot the other AID's Phase 9 helper has not copied
// yet: a signalling-NaN payload, which arithmetic never produces.
constexpr unsigned MPK_WS_POISON = 0x7FBADBADu;
// MPK_WSFC_V2: the W2 tile's 16 float4 outputs, handed from the MoE
// epilogue to the fused layer's post-arrival far store.
__shared__ float4 g_wsfc_val[16];
__shared__ int g_wsfc_cnt;
__shared__ int g_wsfc_base;
#ifdef MPK_WSFC_STATS
__device__ unsigned g_wsfc_stat[3];
__device__ __noinline__ void mpk_wsfc_stat(int k, int v) {
  unsigned n = atomicAdd(&g_wsfc_stat[k], 1u) + 1u;
  if ((n & (n - 1u)) == 0u || (k == 2 && n <= 8u)) {
    printf("[WSFC] %s=%u v=%d\n",
           k == 0 ? "slow" : (k == 1 ? "copied" : "skipped"), n, v);
  }
}
#endif
#ifndef MPK_WS_FC_LAYERS
#define MPK_WS_FC_LAYERS 36
#endif
typedef float mpk_fc_f4 __attribute__((ext_vector_type(4)));
__device__ __forceinline__ float4 mpk_fc_ld16(float const *p) {
  mpk_fc_f4 v;
  asm volatile("global_load_dwordx4 %0, %1, off sc0 sc1\n\t"
               "s_waitcnt vmcnt(0)"
               : "=v"(v)
               : "v"(p)
               : "memory");
  return make_float4(v[0], v[1], v[2], v[3]);
}
__device__ __forceinline__ void mpk_fc_ld16x4(float const *p0, float const *p1,
                                              float const *p2, float const *p3,
                                              float4 &r0, float4 &r1,
                                              float4 &r2, float4 &r3) {
  mpk_fc_f4 a, b, c, d;
  asm volatile("global_load_dwordx4 %0, %4, off sc0 sc1\n\t"
               "global_load_dwordx4 %1, %5, off sc0 sc1\n\t"
               "global_load_dwordx4 %2, %6, off sc0 sc1\n\t"
               "global_load_dwordx4 %3, %7, off sc0 sc1\n\t"
               "s_waitcnt vmcnt(0)"
               : "=&v"(a), "=&v"(b), "=&v"(c), "=&v"(d)
               : "v"(p0), "v"(p1), "v"(p2), "v"(p3)
               : "memory");
  r0 = make_float4(a[0], a[1], a[2], a[3]);
  r1 = make_float4(b[0], b[1], b[2], b[3]);
  r2 = make_float4(c[0], c[1], c[2], c[3]);
  r3 = make_float4(d[0], d[1], d[2], d[3]);
}
__device__ __forceinline__ bool mpk_fc_poisoned(float4 v) {
  return __float_as_uint(v.x) == MPK_WS_POISON ||
         __float_as_uint(v.y) == MPK_WS_POISON ||
         __float_as_uint(v.z) == MPK_WS_POISON ||
         __float_as_uint(v.w) == MPK_WS_POISON;
}
#endif
__device__ __forceinline__ void st_wt_f32x4(void *addr, float4 val) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  typedef unsigned int v4u32 __attribute__((ext_vector_type(4)));
  v4u32 v;
  __builtin_memcpy(&v, &val, 16);
  asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1"
               :
               : "v"(addr), "v"(v)
               : "memory");
#else
  // float4 has no volatile assignment operator; go component-wise.
  float const *src = reinterpret_cast<float const *>(&val);
  volatile float *dst = reinterpret_cast<volatile float *>(addr);
  dst[0] = src[0];
  dst[1] = src[1];
  dst[2] = src[2];
  dst[3] = src[3];
#endif
}

// Write-through 32-bit store (1x float or 2x bf16)
__device__ __forceinline__ void st_wt_u32(void *addr, unsigned int val) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  asm volatile("global_store_dword %0, %1, off sc0 sc1"
               :
               : "v"(addr), "v"(val)
               : "memory");
#else
  *reinterpret_cast<unsigned int volatile *>(addr) = val;
#endif
}

#ifdef MPK_AID_SPLIT_FLAGS
// Publish per-XCD flag `slot` into both replicas, so a consumer in either AID
// finds it on the copy homed in its own range. Two write-through stores
// instead of one; the publish happens 8 times a layer while the flags are
// polled ~184 times, which is why the trade pays.
__device__ __forceinline__ void
    mpk_aid_publish(int *shared_base, int slot, unsigned int val, int region) {
  int *a = g_aid_flag_rep[0];
  int *b = g_aid_flag_rep[1];
  if (a != nullptr && b != nullptr) {
    int const off = region * MPK_AID_REGION_INTS + slot * 16;
    st_wt_u32((void *)&a[off], val);
    st_wt_u32((void *)&b[off], val);
  } else {
    st_wt_u32((void *)&shared_base[slot * 16], val);
  }
}

// Same, addressing by raw int index rather than 64 B slot.
// 64-bit form, for the (epoch, expert) record the early-routing path writes at
// [(1+x)*16+2]. Alignment holds by construction: that index is 64*(1+x)+8 bytes
// from the family base, and the base offset is itself 8-byte aligned.
__device__ __forceinline__ void mpk_aid_publish_at_u64(void *shared_base,
                                                       int idx,
                                                       unsigned long long val,
                                                       int off_ints) {
  int *a = g_aid_flag_rep[0];
  int *b = g_aid_flag_rep[1];
  if (a != nullptr && b != nullptr) {
    st_wt_u64((void *)&a[off_ints + idx], val);
    st_wt_u64((void *)&b[off_ints + idx], val);
  } else {
    st_wt_u64((void *)&((int *)shared_base)[idx], val);
  }
}

// ?? AID-local copy of the O-proj output row ???????????????????????????????
// The row lives at the same [batch][output_stride] offsets inside each AID
// replica, so an index computed for the shared buffer indexes the replica too.
//
// 256 KiB into the 2 MiB replica, clear of the routing family at 32768 ints.
// The bound is checked identically on the write and the read side: if the row
// does not fit, BOTH fall back to the shared buffer. They must never disagree
// -- a publisher writing the replica while a consumer reads the shared copy is
// a silently stale read, not a hang.
// AID-local copy of attn_out. 1.25 MiB into the 2 MiB replica, past the
// O-proj row region (65536 ints + 1 MiB). Indexed exactly like the shared
// buffer, in 32-bit units: element e of the row lives at int index
// MPK_AID_ATTNOUT_BASE_INTS + e/2, and every slice base is even.
// AID-local mirrors of the scheduler's event counters. 1.75 MiB into the 2 MiB
// replica, past the attn_out region, leaving room for 8192 u64 counters against
// the ~42 this model builds. The shared counter stays authoritative; these are
// published copies that only the dependency POLL reads.
constexpr int MPK_AID_EVCTR_BASE_INTS = 458752;
constexpr int MPK_AID_EVCTR_MAX = 8192;

// Per-AID event accumulator: the middle tier between the per-XCD counters and
// the global one. Each AID replica holds its OWN array, so indexing it through
// this XCD's replica selects that AID's counter and the atomic stays local.
// AID-local mirror of the iteration-boundary gate. 491520 is past EVTIER and
// inside the 2 MiB replica (524288 ints); one line is reserved to avoid false
// sharing. ONE slot per AID, not one per XCD: the publisher wrote the same
// value into all eight, so they were pure redundancy.
constexpr int MPK_AID_ITER_BASE_INTS = 491520;

// Write-through into both replicas. An MTYPE_RW line is coherent only inside
// its own AID and a write-through store is what crosses; a cross-AID atomic
// is the hazard that hung MPK_AID_EVCTR v2.
__device__ __forceinline__ void mpk_aid_iter_publish(unsigned int val) {
  int *a = g_aid_flag_rep[0];
  int *b = g_aid_flag_rep[1];
  if (a) {
    st_wt_u32((void *)(a + MPK_AID_ITER_BASE_INTS), val);
  }
  if (b) {
    st_wt_u32((void *)(b + MPK_AID_ITER_BASE_INTS), val);
  }
}

__device__ __forceinline__ int *mpk_aid_iter_slot(int xcd_id) {
  int *rep = g_aid_flag_rep[xcd_id >> 2];
  return rep ? rep + MPK_AID_ITER_BASE_INTS : nullptr;
}

constexpr int MPK_AID_EVTIER_BASE_INTS = 475136;  // past the EVCTR mirrors
constexpr int MPK_AID_XCDS_PER_AID = 4;

__device__ __forceinline__ unsigned long long *mpk_aid_evtier_ptr() {
  unsigned x;
  MPK_GET_XCC_ID32(x);
  int *r = g_aid_flag_rep[(x & 0x7) >> 2];
  return r ? (unsigned long long *)(r + MPK_AID_EVTIER_BASE_INTS) : nullptr;
}

// Post-increment count for this AID; sets *out_ok to 0 if the tier is
// unavailable, in which case the caller takes the original global path.
__device__ __forceinline__ unsigned long long
    mpk_aid_evtier_add(int idx, unsigned long long *out_ok) {
  unsigned long long *p = mpk_aid_evtier_ptr();
  if (p == nullptr || idx >= MPK_AID_EVCTR_MAX) {
    *out_ok = 0;
    return 0;
  }
  *out_ok = 1;
  return atomicAdd(&p[idx], 1ULL) + 1ULL;
}

// ---------------------------------------------------------------------------
// MPK_AID_EVCTR2: monotone per-XCD mirror slots.
//
// The single-slot mirror above is written by all eight XCDs with unordered
// write-through stores, so it is NOT monotone -- a late store from an earlier
// XCD can move it backwards past a threshold a reader is waiting on. That is
// why the poll has to re-read the authoritative counter periodically, and why
// the saving nets to zero: the mandatory truth reads cost what the cheap
// mirror reads save.
//
// Here each XCD owns one slot, so every slot has exactly ONE writer and can
// only increase. A reader sums the eight slots from the replica homed in its
// own AID and never touches the shared counter.
//
// Layout: MPK_AID_EVCTR_BASE_INTS + (idx * 8 + xcd) * 2 ints.
constexpr int MPK_AID_EVCTR2_MAX = 512; // events; 512 * 8 * 8 B = 32 KiB

__device__ __forceinline__ int mpk_aid_evctr2_xcd() {
  unsigned x;
  MPK_GET_XCC_ID32(x);
  return (int)(x & 0xfu);
}

__device__ __forceinline__ unsigned long long *
    mpk_aid_evctr2_rep(int aid) {
  int *r = g_aid_flag_rep[aid];
  return r ? (unsigned long long *)(r + MPK_AID_EVCTR_BASE_INTS) : nullptr;
}

// Publish this XCD's cumulative contribution. Returns false if unavailable,
// in which case the caller keeps the shared-counter path.
__device__ __forceinline__ bool mpk_aid_evctr2_pub(int idx,
                                                   unsigned long long amt) {
  if (idx >= MPK_AID_EVCTR2_MAX) {
    return false;
  }
  int const xcd = mpk_aid_evctr2_xcd() & 7;
  int const aid = xcd >> 2;
  unsigned long long *near_ = mpk_aid_evctr2_rep(aid);
  unsigned long long *far_ = mpk_aid_evctr2_rep(1 - aid);
  if (near_ == nullptr || far_ == nullptr) {
    return false;
  }
  int const slot = idx * 8 + xcd;
  // Same-AID atomic: coherent inside this AID, and this XCD is the only
  // writer of this slot, so the value is monotone.
  unsigned long long const mine = atomicAdd(&near_[slot], amt) + amt;
  // Cross the boundary with a write-through store -- an atomic would not be
  // visible. Still a single writer, so the far copy is monotone too.
  st_wt_u64((void *)&far_[slot], mine);
  return true;
}

// Sum the eight per-XCD slots from this AID's replica. All reads local.
__device__ __forceinline__ unsigned long long
    mpk_aid_evctr2_sum(void *shared_base, int idx) {
  unsigned long long *sh = (unsigned long long *)shared_base;
  if (idx >= MPK_AID_EVCTR2_MAX) {
    return MPK_LD_EVENT(&sh[idx]);
  }
  int const xcd = mpk_aid_evctr2_xcd() & 7;
  unsigned long long *m = mpk_aid_evctr2_rep(xcd >> 2);
  if (m == nullptr) {
    return MPK_LD_EVENT(&sh[idx]);
  }
  unsigned long long t = 0;
  int const base = idx * 8;
#pragma unroll
  for (int x = 0; x < 8; x++) {
    t += MPK_LD_EVENT(&m[base + x]);
  }
  return t;
}

#ifdef MPK_EVCTR2_SC1
// MPK_EVCTR2_SC1_V1: the same sum at agent scope (sc1: an L2 hit until a slot
// changes), with a system-scope re-read every 64th poll as the hang guard.
__device__ __forceinline__ unsigned long long
    mpk_aid_evctr2_sum_sc1(void *shared_base, int idx, unsigned it) {
  if ((it & 63u) == 63u) {
    return mpk_aid_evctr2_sum(shared_base, idx);
  }
  unsigned long long *sh = (unsigned long long *)shared_base;
  if (idx >= MPK_AID_EVCTR2_MAX) {
    return MPK_LD_EVENT(&sh[idx]);
  }
  int const xcd = mpk_aid_evctr2_xcd() & 7;
  unsigned long long *m = mpk_aid_evctr2_rep(xcd >> 2);
  if (m == nullptr) {
    return MPK_LD_EVENT(&sh[idx]);
  }
  unsigned long long t = 0;
  int const base = idx * 8;
#pragma unroll
  for (int x = 0; x < 8; x++) {
    t += __hip_atomic_load(&m[base + x], __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
  }
  return t;
}
#endif

__device__ __forceinline__ unsigned long long *mpk_aid_evctr_rep(int aid) {
  int *r = g_aid_flag_rep[aid];
  return r ? (unsigned long long *)(r + MPK_AID_EVCTR_BASE_INTS) : nullptr;
}

// atomicMax, not a store: increments from different XCDs retire out of order,
// and a plain store could move a mirror BACKWARDS past a threshold a reader is
// already waiting on, stalling it forever. Max is monotone.
__device__ __forceinline__ void mpk_aid_evctr_pub(int idx,
                                                  unsigned long long v) {
  if (idx >= MPK_AID_EVCTR_MAX) {
    return;
  }
  unsigned long long *a = mpk_aid_evctr_rep(0);
  unsigned long long *b = mpk_aid_evctr_rep(1);
  if (a == nullptr || b == nullptr) {
    return;
  }
  // Write-through, NOT atomicMax. A cross-AID atomic on an MTYPE_RW line is
  // not visible -- RW is coherent only inside its own AID -- which is what
  // hung v2. A write-through store from either AID does reach both replicas,
  // the same mechanism the flag replicas rely on. It is not monotone, so the
  // poll below re-reads the shared counter periodically to recover.
  st_wt_u64((void *)&a[idx], v);
  st_wt_u64((void *)&b[idx], v);
}

__device__ __forceinline__ unsigned long long
    mpk_aid_evctr_load(void *shared_base, int idx) {
  unsigned long long *sh = (unsigned long long *)shared_base;
  if (idx >= MPK_AID_EVCTR_MAX) {
    return MPK_LD_EVENT(&sh[idx]);
  }
  unsigned x;
  MPK_GET_XCC_ID32(x);
  unsigned long long *m = mpk_aid_evctr_rep((x & 0xf) >> 2);
  if (m == nullptr) {
    return MPK_LD_EVENT(&sh[idx]);
  }
  return MPK_LD_EVENT(&m[idx]);
}

// Poll-form read: the mirror on all but every 64th spin, the authoritative
// shared counter on those, so a mirror left stale or regressed by an
// out-of-order write-through can delay a worker but can never deadlock it.
__device__ __forceinline__ unsigned long long
    mpk_aid_evctr_poll(void *shared_base, int idx, unsigned it) {
  unsigned long long *sh = (unsigned long long *)shared_base;
  if ((it & 63u) == 63u) {
    return MPK_LD_EVENT(&sh[idx]);
  }
  return mpk_aid_evctr_load(shared_base, idx);
}

// Expression-form read, so the call site needs no preprocessor block and keeps
// its own semicolon.
#if defined(MPK_AID_EVCTR2) && defined(MPK_AID_SPLIT_FLAGS)
// Monotone per-XCD slots: no fallback needed, so the poll ignores the spin
// counter entirely.
#define MPK_EVCTR_READ(base, idx) mpk_aid_evctr2_sum((base), (idx))
#ifdef MPK_EVCTR2_SC1
#define MPK_EVCTR_POLL(base, idx, it) mpk_aid_evctr2_sum_sc1((base), (idx), (it))
#else
#define MPK_EVCTR_POLL(base, idx, it) mpk_aid_evctr2_sum((base), (idx))
#endif
#elif defined(MPK_AID_EVCTR) && defined(MPK_AID_SPLIT_FLAGS)
#define MPK_EVCTR_READ(base, idx) mpk_aid_evctr_load((base), (idx))
#define MPK_EVCTR_POLL(base, idx, it) mpk_aid_evctr_poll((base), (idx), (it))
#else
#define MPK_EVCTR_READ(base, idx) MPK_LD_EVENT(&(base)[idx])
#define MPK_EVCTR_POLL(base, idx, it) MPK_LD_EVENT(&(base)[idx])
#endif

constexpr int MPK_AID_ATTNOUT_BASE_INTS = 327680;
constexpr int MPK_AID_ATTNOUT_MAX_BYTES = 512 * 1024;

__device__ __forceinline__ bool mpk_aid_attnout_live() {
  return g_aid_flag_rep[0] != nullptr && g_aid_flag_rep[1] != nullptr;
}

// Reader: the attn_out base homed in this XCD's own memory range.
__device__ __forceinline__ void const *
    mpk_aid_attnout_base(void const *shared, int xcd_id) {
  if (!mpk_aid_attnout_live()) {
    return shared;
  }
  return (void const *)(g_aid_flag_rep[xcd_id >> 2] +
                        MPK_AID_ATTNOUT_BASE_INTS);
}

// Writer: mirror one 32-bit pair of bf16 elements into both replicas.
__device__ __forceinline__ void mpk_aid_attnout_publish32(int dw,
                                                          unsigned val) {
  if (!mpk_aid_attnout_live()) {
    return;
  }
  st_wt_u32((void *)&g_aid_flag_rep[0][MPK_AID_ATTNOUT_BASE_INTS + dw], val);
  st_wt_u32((void *)&g_aid_flag_rep[1][MPK_AID_ATTNOUT_BASE_INTS + dw], val);
}

#ifdef MPK_VTN_AOQ
#ifdef MPK_AID_SPLIT_ATTNOUT
#error "MPK_VTN_AOQ uses the attn_out replica region"
#endif
// MPK_VTN_AOQ: attn_out rows as FP8 (4096 bytes, then the 32 blocks' E8M0 scales) in the attn_out replica region.
constexpr int MPK_AOQ_ROW_BYTES = 4096 + 32;
__device__ __forceinline__ char *mpk_aoq_base(int aid) {
  return (char *)(g_aid_flag_rep[aid] + MPK_AID_ATTNOUT_BASE_INTS);
}
typedef int mpk_aoq_i32x4 __attribute__((ext_vector_type(4)));
// System-scope (sc0 sc1) loads the compiler counts: the region is rewritten every layer.
__device__ __forceinline__ mpk_aoq_i32x4 mpk_aoq_ld16(char const *p) {
  typedef unsigned long long __attribute__((address_space(1))) aoq_u64g;
  aoq_u64g *const q = (aoq_u64g *)p;
  unsigned long long const a = __hip_atomic_load(q, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  unsigned long long const b = __hip_atomic_load(q + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  return mpk_aoq_i32x4{(int)(unsigned)a, (int)(unsigned)(a >> 32), (int)(unsigned)b, (int)(unsigned)(b >> 32)};
}
__device__ __forceinline__ unsigned mpk_aoq_ld32(char const *p) {
  typedef unsigned __attribute__((address_space(1))) aoq_u32g;
  return __hip_atomic_load((aoq_u32g *)p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
#ifdef MPK_VTN_AOQ_INPLACE
// MPK_VTN_AOQ_INPLACE: the FP8 row lives in the attn_out row itself (8192 bytes per bf16 row), read like the
// bf16 row it replaces.
constexpr int MPK_AOQ_RSTRIDE = 8192;
// Block b's scale byte at 4096 + (b / 4) * 128 + b % 4: one line per die (four 128-element blocks).
#define MPK_AOQ_SOFF(b) (((b) >> 2) * 128)
#define MPK_AOQ_LD16(p) (*(mpk_aoq_i32x4 const *)(p))
#define MPK_AOQ_LD32(p) (*(unsigned const *)(p))
#else
constexpr int MPK_AOQ_RSTRIDE = MPK_AOQ_ROW_BYTES;
#define MPK_AOQ_SOFF(b) ((b) & ~3)
#define MPK_AOQ_LD16(p) mpk_aoq_ld16(p)
#define MPK_AOQ_LD32(p) mpk_aoq_ld32(p)
#endif
#endif

constexpr int MPK_AID_OUT_BASE_INTS = 65536;
constexpr int MPK_AID_OUT_MAX_BYTES = 1024 * 1024;

// MPK_ROUTER_OPROJ_PARTIAL. Per-WG router partials, [parity][xcd][24][160]
// floats on the XCD's own replica (written and read only through that XCD's
// L2), and the per-XCD sums, [parity][xcd][288] ints = 129 {value, tag} u64 on
// both replicas. Both sit in the free gap between OUT (bs1 ends at 67008) and
// SWIGLU (200704). Slots >= the O-proj worker count are never written and
// stay zero from the per-launch memset.
constexpr int MPK_AID_ROP_WG_BASE_INTS = 131072;
constexpr int MPK_AID_ROP_WG_SLOTS = 24;
constexpr int MPK_AID_ROP_WG_INTS = 160;
constexpr int MPK_AID_ROP_AGG_BASE_INTS = 98304;
constexpr int MPK_AID_ROP_AGG_INTS = 288;
// MPK_VT2: token 1's die sums (both AIDs) and its tile slots (NC scratch, XCD-local exchange).
constexpr int MPK_AID_ROP_AGG1_BASE_INTS = 192512;
static_assert(MPK_AID_ROP_AGG1_BASE_INTS >= MPK_AID_ROP_WG_BASE_INTS + 2 * 8 * MPK_AID_ROP_WG_SLOTS *
                                                 MPK_AID_ROP_WG_INTS &&
                  MPK_AID_ROP_AGG1_BASE_INTS + 2 * 8 * MPK_AID_ROP_AGG_INTS <= 200704,
              "token 1's die sums between the tile slots and the SwiGLU replica");
constexpr int MPK_NC_ROP1_INTS = 65536;
// MPK_VT2: token 1's FP8 hand-off row, one private 4 KiB slot per worker of the AID (4 dies x 32).
constexpr int MPK_NC_ROP1ROW_INTS = MPK_NC_ROP1_INTS + 2 * 8 * 24 * 160;
static_assert(MPK_AID_ROP_WG_BASE_INTS + 2 * 8 * MPK_AID_ROP_WG_SLOTS *
                      MPK_AID_ROP_WG_INTS <= 200704,
              "ROP WG partials overlap SWIGLU");
static_assert(MPK_AID_ROP_AGG_BASE_INTS + 2 * 8 * MPK_AID_ROP_AGG_INTS <=
                  MPK_AID_ROP_WG_BASE_INTS,
              "ROP aggregate overlaps the WG partials");
// MPK_ROP_RELAY, also in the gap below the WG partials and only ever touched
// through the owning XCD's L2: one election word per XCD (128 B apart) and
// the relay's copy of the eight-XCD sums, [parity][xcd][288] ints.
constexpr int MPK_AID_ROP_RLY_CTR_BASE_INTS = 103424;
constexpr int MPK_AID_ROP_RLY_BASE_INTS = 103936;
// MPK_VTN_N7B, token t of an N-token verify pass (t = 0 / 1 are the 2-token regions): tile slots after
// token 1's in the NC scratch (tokens >= 2 inside token 1's FP8-row area, unused under MPK_VTN_MOE, whose
// rows live in LDS), die sums in the replica gap between the relay copy and the WG partials.
__host__ __device__ constexpr int mpk_rop_nc_ints(int t) {
  return MPK_NC_ROP1_INTS + (t - 1) * (2 * 8 * MPK_AID_ROP_WG_SLOTS * MPK_AID_ROP_WG_INTS);
}
__host__ __device__ constexpr int mpk_rop_agg_ints(int t) {
  return t == 0   ? MPK_AID_ROP_AGG_BASE_INTS
         : t == 1 ? MPK_AID_ROP_AGG1_BASE_INTS
#if defined(MPK_VTN_T) && MPK_VTN_T > 6
         // MPK_VTN_T8: tokens 6.. in the W13 hand-off replica region (unused: MPK_W13_HOF_AIDREP is off).
         : t >= 6 ? MPK_AID_W13HOF_BASE_INTS + (t - 6) * 2 * 8 * MPK_AID_ROP_AGG_INTS
#endif
                  : 110592 + (t - 2) * 2 * 8 * MPK_AID_ROP_AGG_INTS;
}
static_assert(mpk_rop_agg_ints(2) >= MPK_AID_ROP_RLY_BASE_INTS + 2 * 8 * MPK_AID_ROP_AGG_INTS &&
                  mpk_rop_agg_ints(5) + 2 * 8 * MPK_AID_ROP_AGG_INTS <= MPK_AID_ROP_WG_BASE_INTS,
              "MPK_VTN_N7B: tokens 2-5's die sums between the relay copy and the WG partials");
static_assert(mpk_rop_nc_ints(4) <= MPK_NC_ROP1ROW_INTS + 128 * 1024,
              "MPK_VTN_N7B: tokens 2-3's tile slots inside token 1's unused FP8-row area");
#if defined(MPK_VT2) && defined(MPK_VTN_MOE)
#if defined(MPK_VTN_T) && MPK_VTN_T > 4
// MPK_VTN_T6: tokens >= 4's tile slots continue past token 1's FP8-row area (nothing else lives above it
// in the NC scratch half); their die sums continue mpk_rop_agg_ints (tokens 2-5 fit below the WG partials).
#define MPK_ROP_NTOK MPK_VTN_T
static_assert(MPK_VTN_T <= 8, "MPK_VTN_T8: die sums for tokens 2-7 (mpk_rop_agg_ints)");
#if MPK_VTN_T > 6
#ifdef MPK_W13_HOF_AIDREP
#error "MPK_VTN_T8: tokens 6-7's die sums live in the W13 hand-off replica region"
#endif
static_assert(mpk_rop_agg_ints(MPK_VTN_T - 1) + 2 * 8 * MPK_AID_ROP_AGG_INTS <= MPK_AID_L2CLEAN_BASE_INTS,
              "MPK_VTN_T8: tokens 6-7's die sums below L2CLEAN");
#endif
static_assert(mpk_rop_nc_ints(MPK_VTN_T) <= (32 << 20) / 4,
              "MPK_VTN_T6: tile slots inside the 32 MiB NC half");
#else
#define MPK_ROP_NTOK 4
#endif
#else
#define MPK_ROP_NTOK 2
#endif
#ifdef MPK_VTN_QMP
#if !defined(MPK_VTN_T) || MPK_VTN_T <= 4 || !defined(MPK_VTN_QROW_TAG)
#error "MPK_VTN_QMP: VTN above 4 tokens with MPK_VTN_QROW_TAG"
#endif
// MPK_VTN_QMP: tagged quantized-row slots per (die, parity, token) in the NC scratch half of the die's AID,
// past the router tile slots; cleared per launch.
constexpr int MPK_NC_QROW_INTS = 524288;
constexpr int MPK_NC_QROW_SLOT_INTS = 1536;
constexpr int MPK_NC_QROW_TOTAL_INTS = 4 * 2 * MPK_VTN_T * MPK_NC_QROW_SLOT_INTS;
static_assert(MPK_NC_QROW_INTS >= mpk_rop_nc_ints(MPK_ROP_NTOK) &&
                  MPK_NC_QROW_INTS + MPK_NC_QROW_TOTAL_INTS <= (32 << 20) / 4,
              "MPK_VTN_QMP: row slots inside the NC half, past the router tile slots");
#endif
static_assert(MPK_AID_ROP_AGG_BASE_INTS + 2 * 8 * MPK_AID_ROP_AGG_INTS <=
                      MPK_AID_ROP_RLY_CTR_BASE_INTS &&
                  MPK_AID_ROP_RLY_CTR_BASE_INTS + 8 * 32 <=
                      MPK_AID_ROP_RLY_BASE_INTS &&
                  MPK_AID_ROP_RLY_BASE_INTS + 2 * 8 * MPK_AID_ROP_AGG_INTS <=
                      MPK_AID_ROP_WG_BASE_INTS,
              "ROP relay overlaps");

__device__ __forceinline__ bool mpk_aid_out_fits(int bytes) {
  return bytes <= MPK_AID_OUT_MAX_BYTES && g_aid_flag_rep[0] != nullptr &&
         g_aid_flag_rep[1] != nullptr;
}

// Reader: the row base homed in this XCD's own memory range.
__device__ __forceinline__ void const *
    mpk_aid_out_base(void const *shared, int xcd_id, int bytes) {
  if (!mpk_aid_out_fits(bytes)) {
    return shared;
  }
  return (void const *)(g_aid_flag_rep[xcd_id >> 2] + MPK_AID_OUT_BASE_INTS);
}

// Writer: mirror one packed-4-bf16 payload into both replicas. `idx` is in
// bf16 elements and is a multiple of 4, and the region base is 256 KiB
// aligned, so the 8-byte store stays aligned.
__device__ __forceinline__ void
    mpk_aid_out_publish(int idx, unsigned long long val, int bytes) {
  if (!mpk_aid_out_fits(bytes)) {
    return;
  }
  unsigned short *a =
      (unsigned short *)(g_aid_flag_rep[0] + MPK_AID_OUT_BASE_INTS);
  unsigned short *b =
      (unsigned short *)(g_aid_flag_rep[1] + MPK_AID_OUT_BASE_INTS);
  st_wt_u64((void *)&a[idx], val);
  st_wt_u64((void *)&b[idx], val);
}

__device__ __forceinline__ void mpk_aid_publish_at(int *shared_base,
                                                   int idx,
                                                   unsigned int val,
                                                   int off_ints) {
  int *a = g_aid_flag_rep[0];
  int *b = g_aid_flag_rep[1];
  if (a != nullptr && b != nullptr) {
    st_wt_u32((void *)&a[off_ints + idx], val);
    st_wt_u32((void *)&b[off_ints + idx], val);
  } else {
    st_wt_u32((void *)&shared_base[idx], val);
  }
}
#endif

// Write-through 16-bit store (1x bf16)
__device__ __forceinline__ void st_wt_u16(void *addr, unsigned short val) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  asm volatile("global_store_short %0, %1, off sc0 sc1"
               :
               : "v"(addr), "v"(val)
               : "memory");
#else
  *reinterpret_cast<unsigned short volatile *>(addr) = val;
#endif
}

// Write-through 8-bit store (1x byte)
__device__ __forceinline__ void st_wt_u8(void *addr, uint8_t val) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  asm volatile("global_store_byte %0, %1, off sc0 sc1"
               :
               : "v"(addr), "v"((unsigned)val)
               : "memory");
#else
  *reinterpret_cast<uint8_t volatile *>(addr) = val;
#endif
}

__device__ __forceinline__ int atom_add_release_gpu_s32(int *addr, int val) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  // Inline asm: no compiler-generated buffer_wbl2/buffer_inv around atomic.
  // Ordering provided by explicit threadfence_gpu() before this call.
  // sc0 sc1 required on GFX942 for cross-CU atomic visibility.
  int old_val;
  asm volatile("flat_atomic_add %0, %1, %2 sc0 sc1\n"
               "s_waitcnt vmcnt(0) lgkmcnt(0)"
               : "=v"(old_val)
               : "v"(addr), "v"(val)
               : "memory");
  return old_val;
#else
  int old_val;
  asm volatile("atom.add.release.gpu.s32 %0,[%1],%2;"
               : "=r"(old_val)
               : "l"(addr), "r"(val)
               : "memory");
  return old_val;
#endif
}

// XCD-local counterpart of atom_add_release_gpu_s32: `sc0` without `sc1`.
//
// `sc1` is what carries an atomic through the device-wide coherency point. On
// MI350 the eight XCDs have separate, non-coherent L2s, so an `sc0 sc1` atomic
// must leave the XCD and serialise at that point; an `sc0`-only atomic is
// resolved in the XCD's own L2 and never crosses the die boundary. `sc0` still
// returns the pre-op value and is still atomic against every CU on this XCD,
// which is the whole participant set of an XCD-private barrier.
//
// Only correct for a counter whose writers AND readers are all on one XCD --
// i.e. every access is indexed by that worker's own `xcd_id`. Using it on a
// counter any other XCD touches drops the update into a private L2 where the
// remote reader cannot see it, and the barrier hangs. `MPK_XCD_LOCAL_BARRIER`
// gates each conversion so the pairing can be re-checked per site.
__device__ __forceinline__ int atom_add_xcd_local_s32(int *addr, int val) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  int old_val;
  asm volatile("flat_atomic_add %0, %1, %2 sc0\n"
               "s_waitcnt vmcnt(0) lgkmcnt(0)"
               : "=v"(old_val)
               : "v"(addr), "v"(val)
               : "memory");
  return old_val;
#else
  int old_val;
  asm volatile("atom.add.release.cta.s32 %0,[%1],%2;"
               : "=r"(old_val)
               : "l"(addr), "r"(val)
               : "memory");
  return old_val;
#endif
}

__device__ __forceinline__ unsigned long long int
    atom_add_release_gpu_u64(unsigned long long int *addr,
                             unsigned long long int val) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  // Inline asm: no compiler-generated buffer_wbl2/buffer_inv around atomic.
  // Ordering provided by explicit threadfence_gpu() before this call.
  // sc0 sc1 required on GFX942 for cross-CU atomic visibility.
  unsigned long long int old_val;
  asm volatile("flat_atomic_add_x2 %0, %1, %2 sc0 sc1\n"
               "s_waitcnt vmcnt(0) lgkmcnt(0)"
               : "=v"(old_val)
               : "v"(addr), "v"(val)
               : "memory");
  return old_val;
#else
  unsigned long long int old_val;
  asm volatile("atom.add.release.gpu.u64 %0,[%1],%2;"
               : "=l"(old_val)
               : "l"(addr), "l"(val)
               : "memory");
  return old_val;
#endif
}

__device__ __forceinline__ unsigned long long int
    atom_cas_release_gpu_u64(unsigned long long int *addr,
                             unsigned long long int cmp,
                             unsigned long long int val) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  // Inline asm: no compiler-generated buffer_wbl2/buffer_inv around atomic.
  // Ordering provided by explicit threadfence_gpu() before this call.
  // sc0 sc1 required on GFX942 for cross-CU atomic visibility.
  // flat_atomic_cmpswap_x2 layout: {swap[63:0], compare[63:0]} in 4 VGPRs.
  typedef unsigned long long v2u64 __attribute__((ext_vector_type(2)));
  v2u64 cmp_swap;
  cmp_swap[0] = val; // swap value (low 64 bits) — new value to write
  cmp_swap[1] = cmp; // compare value (high 64 bits) — expected old value
  unsigned long long int old_val;
  asm volatile("flat_atomic_cmpswap_x2 %0, %1, %2 sc0 sc1\n"
               "s_waitcnt vmcnt(0) lgkmcnt(0)"
               : "=v"(old_val)
               : "v"(addr), "v"(cmp_swap)
               : "memory");
  return old_val;
#else
  unsigned long long int old_val;
  asm volatile("atom.cas.release.gpu.b64 %0,[%1],%2,%3;"
               : "=l"(old_val)
               : "l"(addr), "l"(cmp), "l"(val)
               : "memory");
  return old_val;
#endif
}

__device__ __forceinline__ unsigned long long int
    ld_acquire_gpu_u64(unsigned long long int *addr) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
#if defined(MPK_CPX_SCOPE) // MDEV_CPX_LD: an nt poll stays in this XCC's L2 under CPX
  return ld_sys_u64((void *)addr);
#elif defined(MPK_USE_NT_MEMORY)
  // Non-temporal load bypasses cache, reads from memory
  return ld_nt_u64(addr);
#elif defined(MPK_USE_RELAXED_ATOMICS)
  return __atomic_load_n(addr, __ATOMIC_RELAXED);
#else
  // HIP/AMD: Use SEQ_CST for cross-XCD visibility on MI300
  return __atomic_load_n(addr, __ATOMIC_SEQ_CST);
#endif
#else
  unsigned long long int val;
  asm volatile("ld.acquire.gpu.u64 %0, [%1];" : "=l"(val) : "l"(addr));
  return val;
#endif
}

__device__ __forceinline__ unsigned long long int
    ld_acquire_sys_u64(unsigned long long int *addr) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
#ifdef MPK_USE_RELAXED_ATOMICS
  return __atomic_load_n(addr, __ATOMIC_RELAXED);
#else
  // HIP/AMD: Use SEQ_CST for cross-XCD visibility on MI300
  return __atomic_load_n(addr, __ATOMIC_SEQ_CST);
#endif
#else
  unsigned long long int val;
  asm volatile("ld.acquire.sys.u64 %0, [%1];"
               : "=l"(val)
               : "l"(addr)
               : "memory");
  return val;
#endif
}

__device__ __forceinline__ unsigned long long int
    ld_relaxed_gpu_u64(unsigned long long int *addr) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
#if defined(MPK_CPX_SCOPE) // MDEV_CPX_LD: an nt poll stays in this XCC's L2 under CPX
  return ld_sys_u64((void *)addr);
#elif defined(MPK_USE_NT_MEMORY)
  // Non-temporal load bypasses cache, reads from memory
  return ld_nt_u64(addr);
#else
  // HIP/AMD: use __atomic_load_n with relaxed ordering
  return __atomic_load_n(addr, __ATOMIC_RELAXED);
#endif
#else
  unsigned long long int val;
  asm volatile("ld.relaxed.gpu.u64 %0, [%1];" : "=l"(val) : "l"(addr));
  return val;
#endif
}

__device__ __forceinline__ void st_relaxed_gpu_u64(unsigned long long int *addr,
                                                   unsigned long long int val) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
#if defined(MPK_CPX_SCOPE) // MDEV_CPX_ST: write through to memory for the other XCCs
  st_wt_u64((void *)addr, val);
#elif defined(MPK_USE_NT_MEMORY)
  // Non-temporal store bypasses cache, writes to memory
  st_nt_u64(addr, val);
#else
  // Use RELAXED for stores - threadfence_gpu() after provides ordering
  __atomic_store_n(addr, val, __ATOMIC_RELAXED);
#endif
#else
  asm volatile("st.relaxed.gpu.u64 [%0], %1;" : : "l"(addr), "l"(val));
#endif
}

// Memory fence for GPU scope - ensures all previous memory operations are
// visible Define MPK_DISABLE_THREADFENCE to disable for performance testing
__device__ __forceinline__ void threadfence_gpu() {
#ifdef MPK_DISABLE_THREADFENCE
  // Disabled for performance testing - results may be incorrect
  (void)0;
#elif defined(__HIP_DEVICE_COMPILE__) &&                                       \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  // MI300X: L2 NOT coherent across XCDs — buffer_wbl2 required.
  // Agent scope (sc1) sufficient — no need for system scope (sc0 sc1).
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, MPK_DEV_FENCE_REL);
#else
  __threadfence();
#endif
}

// =========================================================================
// Intra-XCD (same L2 partition) memory operations.
// On MI300X, all CUs within an XCD share the same 32MB L2 cache.
// Regular stores are immediately visible to regular loads on the same XCD.
// No buffer_wbl2 (L2→HBM writeback) or NT loads needed.
// =========================================================================

// Regular volatile store through L2 — visible to same-XCD CUs immediately
__device__ __forceinline__ void st_local_u64(unsigned long long int *addr,
                                             unsigned long long int val) {
  *((unsigned long long int volatile *)addr) = val;
}

// Regular volatile load from L2 — sees same-XCD stores immediately
__device__ __forceinline__ unsigned long long int
    ld_local_u64(unsigned long long int const *addr) {
  return *((unsigned long long int const volatile *)addr);
}

// Compiler fence — ensures store ordering without buffer_wbl2.
// L2 coherency within an XCD means no hardware fence is needed.
__device__ __forceinline__ void fence_local() {
  asm volatile("" ::: "memory");
}

// Device-scope atomicAdd — sufficient for intra-XCD communication.
// No sc0/sc1 system-scope qualifiers needed.
__device__ __forceinline__ unsigned long long int
    atom_add_local_u64(unsigned long long int *addr,
                       unsigned long long int val) {
  return atomicAdd(addr, val);
}

// Device-scope atomicCAS — sufficient for intra-XCD communication.
__device__ __forceinline__ unsigned long long int
    atom_cas_local_u64(unsigned long long int *addr,
                       unsigned long long int cmp,
                       unsigned long long int val) {
  return atomicCAS(addr, cmp, val);
}

#ifdef MPK_LOCAL_TOPK
// MPK_LOCAL_TOPK (bs=1, 128 experts, XCD-pair MoE map): each router tile
// publishes its logit as one epoch-tagged word, (tag << 16) | bf16 bits, and
// every workgroup rebuilds the routing at the Phase 7b gate instead of waiting
// for a serial TopK completer.
//
// The 128 tag words live at this int offset in the oproj_topk_counters block
// (demo.py grows the block to MPK_LTK_TAG_SLOT + 128 under the flag), and
// routing_ready sits at 10 * 16 in the same block.
constexpr int MPK_LTK_TAG_SLOT = 2048;
constexpr int MPK_LTK_ROUTING_REL = MPK_LTK_TAG_SLOT - 10 * 16;
#ifdef MPK_LTK_EARLY_TAG
// One 128 B line per hardware XCD after the tags: normed-row writer t sets
// word t to mpk_ltk_tag(epoch) once its write-through row stores drained.
// demo.py grows the block to MPK_LTK_NORM_SLOT + 8 * 32 under the flag.
constexpr int MPK_LTK_NORM_SLOT = MPK_LTK_TAG_SLOT + 128;
constexpr int MPK_LTK_NORM_REL = MPK_LTK_NORM_SLOT - 10 * 16;
#endif
// Never 0, so the zero-initialized block cannot satisfy a poll, and
// consecutive layers always differ.
__device__ __forceinline__ unsigned mpk_ltk_tag(int epoch) {
  return 1u + (unsigned)epoch % 65535u;
}
// Selection-order picks and renormalized weights: what the completer carries
// to XCD pair k in its u64 record, and what it writes to topk_weight[k].
__shared__ int s_ltk_sel[8];
__shared__ float s_ltk_w[8];
__shared__ unsigned short s_ltk_logit[128];
#ifdef MPK_VT2
// Per-token copies of the local TopK result for a 2-token verify pass (MPK_VTN_T tokens, MPK_VTN_N7B).
#if defined(MPK_VTN_T) && MPK_VTN_T > 2
#define MPK_VT2_NTOK MPK_VTN_T
#else
#define MPK_VT2_NTOK 2
#endif
__shared__ int s_vt2_sel[MPK_VT2_NTOK][8];
__shared__ float s_vt2_w[MPK_VT2_NTOK][8];
#ifndef MPK_ROUTER_OPROJ_PARTIAL
__shared__ unsigned short s_vt2_logit[128];
#endif
#endif
#endif

#ifdef MPK_OPROJ_TILE_FLAGS
#if defined(MPK_OPROJ_ARRIVE_ONLY) || defined(MPK_ROUTER_XCD_FOLD)
#error "MPK_OPROJ_TILE_FLAGS replaces the O-proj release that these flags modify"
#endif
// One ready word per O-proj tile (the layer epoch), after the LTK tag and
// normed-row words. demo.py grows the block to MPK_OPROJ_TILE_FLAG_SLOT + 256
// under the flag.
constexpr int MPK_OPROJ_TILE_FLAG_SLOT = 2048 + 128 + 8 * 32;
// Called by one full wave. Each lane checks up to three words per trip, all
// three loads in flight together.
__device__ __forceinline__ void mpk_oproj_tile_flag_poll(int *flags, int n,
                                                         int expected) {
  int const lane = threadIdx.x & 63;
  if (n > 192) {
    for (int j = lane; j < n; j += 64) {
      while (ld_sys_s32(flags + j) < expected) {
        __builtin_amdgcn_s_sleep(1);
      }
    }
    return;
  }
  int *const p0 = flags + (lane < n ? lane : 0);
  int *const p1 = flags + (lane + 64 < n ? lane + 64 : 0);
  int *const p2 = flags + (lane + 128 < n ? lane + 128 : 0);
  while (true) {
    int a, b, c;
    asm volatile("global_load_dword %0, %3, off sc0 sc1\n"
                 "global_load_dword %1, %4, off sc0 sc1\n"
                 "global_load_dword %2, %5, off sc0 sc1\n"
                 "s_waitcnt vmcnt(0)"
                 : "=&v"(a), "=&v"(b), "=&v"(c)
                 : "v"(p0), "v"(p1), "v"(p2)
                 : "memory");
    bool const ok = a >= expected && b >= expected && c >= expected;
    if (__builtin_amdgcn_ballot_w64(!ok) == 0) {
      break;
    }
    __builtin_amdgcn_s_sleep(1);
  }
}
#endif

// Without MPK_AID_SPLIT_FLAGS the replica helpers are not built; these read
// the shared buffers directly (upstream's behaviour).
#ifndef MPK_MOE_HOF_IN
#define MPK_MOE_HOF_IN(torch_in) (torch_in)
#endif
#ifndef MPK_EVCTR_READ
#define MPK_EVCTR_READ(base, idx) MPK_LD_EVENT(&(base)[idx])
#endif
#ifndef MPK_EVCTR_POLL
#define MPK_EVCTR_POLL(base, idx, it) MPK_LD_EVENT(&(base)[idx])
#endif
