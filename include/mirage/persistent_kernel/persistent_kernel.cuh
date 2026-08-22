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

#include "profiler.h"
#include "tasks/common/copy_sm80.cuh"
#ifdef MPK_ENABLE_TMA
#include "tma.cuh"
#endif
#include "mpk_atoms.cuh"
#include "runtime_header.h"
// Backend-agnostic SHMEM comm shim (NVSHMEM / rocSHMEM / single-PE no-op).
// Provides mpk_putmem_signal_block, MPK_SIGNAL_ADD, and the host mpk_shmem_*
// wrappers to both the megakernel host code and the emitted allreduce tasks.
#include "comm/mpk_comm.cuh"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <map>
#include <set>
#include <thread>
#include <unistd.h>
#include <vector>

// ── Sub-phase timing (tile_idx==0 only, low overhead) ──────────────────
// Must be declared before task_header.cuh so gang task kernels can see them.
#if defined(MPK_ENABLE_SUBPHASE_TIMING) || defined(MPK_ENABLE_MOE_SUBPHASE)
// Slots: 0=QKV 1=QKV_KVUPD 2=OPROJ 3=OPROJ_TOPK 4=MOE_W13 5=MOE_W2
// 6=ROUTER_SPLIT, GLM only: the o_proj barrier wait is served inside the
// router kernel (so the gamma/gate prefetch covers it), which charges it to
// slot 3 phase 2 along with the norm and the gate GEMV. [0] is the spin
// alone; subtract it from SP3[2] for the router's real compute.
// 7=TOPK_SPLIT, GLM only: the inside of the noaux_tc TopK tail, which SP6[6]
// measures whole. One sample per layer -- only the elected block runs it -- so
// aggregate / ranks / iters is already the wall.
#define SUBPHASE_SLOTS 8
#define SUBPHASE_MAX_PHASES 8
__device__ unsigned long long g_subphase_ns[SUBPHASE_SLOTS]
                                           [SUBPHASE_MAX_PHASES];
__device__ unsigned long long g_subphase_cnt[SUBPHASE_SLOTS];
__device__ int g_subphase_active;
// Scratch timestamps in global memory — avoids LDS offset conflicts with
// __shared__ variables (which shift extern __shared__ _fused_smem) and
// avoids VGPR pressure from local variables.
__device__ unsigned long long g_subphase_scratch[8];
#endif

#ifdef MPK_FUSED_PHASE_TIMING
// Fused O-proj+MoE phase timing: [0]=oproj_ns, [1]=poll_ns, [2]=moe_ns,
// [3]=count
__device__ unsigned long long g_fused_phase_ns[4];
#endif

// ── EP skew probe ──────────────────────────────────────────────────────────
//
// Phase 9 waits for ALL 240 workers before folding, but a given column of the
// residual is only written by the W2 tiles whose output slice covers it --
// ~2 tiles per rank at top_k=4, not 240. If those tiles finish much earlier
// than the last worker on the GPU, the barrier is buying nothing for that
// column and a per-slice release could ship it early. If they finish at
// roughly the same time, there is nothing to overlap and the whole
// send-as-you-go idea is dead regardless of how it is implemented.
//
// This measures exactly that gap, and nothing else:
//   g_ep_slice_last[x] = timestamp of the last W2 atomicAdd into XCD x's
//                        column slice (max over contributing tiles)
//   g_ep_gpu_last      = timestamp of the last W2 tile anywhere on the GPU
// The difference, per slice, is the headroom a per-slice release could claim.
//
// Written with plain atomics on device globals from the W2 epilogue only
// (one per tile, not per element), so the probe does not perturb the compute
// it is measuring. Off unless MPK_EP_SKEW_PROBE is defined.
#ifdef MPK_EP_SKEW_PROBE
#define EP_SKEW_SLICES 8
#define EP_SKEW_HIST 64
__device__ unsigned long long g_ep_slice_last[EP_SKEW_SLICES];
__device__ unsigned long long g_ep_gpu_last;
__device__ unsigned long long g_ep_layer_t0;
// Per-layer samples of (gpu_last - slice_last), in units of s_memrealtime
// ticks (100 MHz => 10 ns/tick). Ring of the most recent EP_SKEW_HIST layers.
__device__ unsigned long long g_ep_skew_hist[EP_SKEW_HIST][EP_SKEW_SLICES];
__device__ unsigned long long g_ep_skew_span[EP_SKEW_HIST];
__device__ int g_ep_skew_n;
#endif

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
// Phase 9 breakdown accumulators, in s_memrealtime ticks (10 ns).
// [role][9a, 9bc, 9d, 9e]; role 0 = folder (xcd_rank 0), 1 = follower.
// Accumulated rather than printed per layer -- see the dump site in
// gang_full_layer_fused_mi300.cuh for why.
__device__ unsigned long long g_ep9_ns[2][4];
__device__ unsigned long long g_ep9_cnt[2];
// Phases 1-8 breakdown, same accumulate-and-dump-once treatment as g_ep9_ns
// and for the same reason: the per-layer [FUSED_PHASE] printf takes the
// iteration from 2.5 ms to ~440 ms, so its numbers describe the printf.
// Slots: [0] qkv_gemm  [1] qkv_barrier  [2] attn  [3] merge+flush
//        [4] wait_others (to the cross-XCD barrier)  [5] Phase 6 xcd_barrier
//        [6] Phase 7 oproj+topk  [7] Phase 8 MoE
//        [8] Phase 7 split: oproj+router compute only (so [6] - [8] is the
//            Phase 7b routing_ready wait)
__device__ unsigned long long g_fp_ns[9];
__device__ unsigned long long g_fp_cnt;
// Inside Phase 7's compute half, which is the only part a row-shard of o_proj
// could halve. [0] o_proj MFMA, [1] the GPU-wide o_proj barrier, [2] RMSNorm +
// router GEMV. Only [0] is shardable: RMSNorm and the router read the whole
// hidden vector by definition, so they stay replicated on both ranks.
__device__ unsigned long long g_op3_ns[3];
__device__ unsigned long long g_op3_cnt;
// Worker ARRIVAL spread at 9a: min/max over all 240 workgroups of the instant
// they reach Phase 9, reset each layer. The W2-tile skew probe says tile
// completion spreads only 0.48 us, but a worker's last act is not necessarily a
// W2 tile, so tile spread does not bound worker spread. This measures the thing
// the barrier actually waits on.
// Initialized to ~0 so the first atomicMin takes; the closer resets it to ~0
// again at the end of every layer.
__device__ unsigned long long g_ep9_arr_min = ~0ull;
__device__ unsigned long long g_ep9_arr_max;
__device__ unsigned long long g_ep9_arr_span_sum;
__device__ unsigned long long g_ep9_arr_span_max;
__device__ unsigned long long g_ep9_arr_n;
// Same, for entry into Phase 8 (MoE). Comparing the two spreads says whether
// MoE creates the skew or inherits it.
__device__ unsigned long long g_ep8_arr_min = ~0ull;
__device__ unsigned long long g_ep8_arr_max;
__device__ unsigned long long g_ep8_arr_span_sum;
__device__ unsigned long long g_ep8_arr_span_max;
// Per-worker MoE occupancy, indexed by worker (xcd*30 + xcd_rank).
// g_moe_busy_ns is the time each worker spends between entering Phase 8 and
// reaching 9a; g_moe_tiles is how many tiles it was handed. If busy time is
// flat the barrier is waiting on something other than MoE work; if it tracks
// the tile count, the tile->worker map is the problem.
#define MOE_OCC_WORKERS 240
__device__ unsigned long long g_moe_busy_ns[MOE_OCC_WORKERS];
__device__ unsigned long long g_moe_tiles[MOE_OCC_WORKERS];
__device__ unsigned long long g_moe_occ_iters;

// ── MPK_PERFETTO: raw per-worker phase spans, for a Perfetto timeline ──────
// Everything above reduces timestamps into atomicAdd accumulators, which is
// why the 20x max/min worker spread in [MOEOCC] cannot be attributed: a mean
// says nothing about WHICH worker was slow or what it was doing. This keeps
// the raw pairs instead.
//
// One row per (worker, layer, phase): 6 phase boundaries the fused task
// already timestamps (_fused_t0/t0a/t0b/t1/t2/t3, _ep_t0.._ep_t4). We store
// entry+exit as raw s_memrealtime ticks (100 MHz, 10 ns) and let the host do
// the arithmetic, so no information is lost to a reduction.
//
// Sized for ONE layer-sweep of one decode iteration: 240 workers x 36 layers
// x PERF_NPHASE. At 8 phases x 2 u32 that is 240*36*8*2*4 B = 553 KB of device
// global -- fine. Writes are plain stores to device memory (NOT the pinned
// host buffer MPK_WS_MARK uses, which costs a PCIe write per mark and was
// measured at 2.3x when done four times per tile).
#define PERF_NPHASE 8
#define PERF_NLAYER 36
#define PERF_SLOTS (MOE_OCC_WORKERS * PERF_NLAYER * PERF_NPHASE)
// [slot][0] = entry tick, [slot][1] = exit tick, relative to g_perf_t0.
__device__ unsigned int g_perf_span[PERF_SLOTS][2];
// Kernel-wide time origin, so the u32 deltas cannot overflow: at 100 MHz a
// u32 covers 43 s, and one decode iteration is ~2 ms.
__device__ unsigned long long g_perf_t0;
// Set to 1 by the first worker to reach the capture iteration; the dump
// reads it to know the buffer is populated. Capturing every iteration would
// just overwrite, so the gate picks a single steady-state one.
__device__ int g_perf_armed;
__device__ __forceinline__ void mpk_perf_span(int worker,
                                              int layer,
                                              int phase,
                                              unsigned long long t_in,
                                              unsigned long long t_out) {
  if (worker >= MOE_OCC_WORKERS || layer >= PERF_NLAYER ||
      phase >= PERF_NPHASE) {
    return;
  }
  // g_perf_armed guards the origin: worker 0 of layer 0 publishes g_perf_t0,
  // and every other worker records its spans at the END of its layer, so the
  // origin is set long before it is read. The check makes that ordering
  // explicit rather than assumed -- an unarmed read would see t0 == 0 and
  // turn the subtraction into a full 64-bit tick count, overflowing the u32.
  if (!g_perf_armed) {
    return;
  }
  unsigned long long const t0 = g_perf_t0;
  if (t_in < t0 || t_out < t_in) {
    return;
  }
  int const slot = (worker * PERF_NLAYER + layer) * PERF_NPHASE + phase;
  g_perf_span[slot][0] = (unsigned int)(t_in - t0);
  g_perf_span[slot][1] = (unsigned int)(t_out - t0);
}
// Where a MoE worker's time actually goes: w13 compute, the W13->W2 per-expert
// barrier poll, and W2 compute. The occupancy probe shows workers holding at
// most 2 tiles yet spanning 1-12 us, so the cost is inside a tile, not in the
// number of them. These three say which part.
__device__ unsigned long long g_moe_w13_ns;
__device__ unsigned long long g_moe_w2bar_ns;
__device__ unsigned long long g_moe_w2_ns;
__device__ unsigned long long g_moe_w13_n;
__device__ unsigned long long g_moe_w2_n;
// W13's 7.4 us/tile against a 4.4 us HBM-stream floor for the same bytes: the
// four terms that 7.4 splits into, so the 3.0 us gap can be attributed before
// anything is restructured. Double-buffering the K loop only pays if the MFMA
// term is large enough to hide loads behind; if the gap is in the load drain
// or the SwiGLU epilogue instead, it buys nothing.
//   p: entry -> the 24 dwordx4 buffer_load_lds prefetch is ISSUED
//   a: that -> post FP8 quant of the token
//   b: post-quant -> weight/scale loads drained (s_waitcnt vmcnt(0))
//   c: drain -> MFMA + SwiGLU + swiglu_out store complete
//   d: that -> barrier arrival done (tile exit)
// The first cut folded p into a and reported 3.46 us of "quant". It is not
// quant: the prefetch issue sits in front of it and the quant's own loads
// queue behind ~98 KB of in-flight weight traffic, so p+a+b was measuring the
// HBM stream (4.42 us floor for these bytes at nt=256) more than the ALU.
// pre splits again into addr (t0v/t0m setup, 24 readfirstlane) and issue (the
// 24-load asm block), because the block is supposed to cost ~75 ns and the
// s_mov_b32 m0 in front of every load is the reason to doubt that.
__device__ unsigned long long g_w13_addr_ns;
__device__ unsigned long long g_w13_pre_ns;
__device__ unsigned long long g_w13_quant_ns;
__device__ unsigned long long g_w13_drain_ns;
__device__ unsigned long long g_w13_mfma_ns;
__device__ unsigned long long g_w13_epi_ns;
__device__ unsigned long long g_w13_stdrain_ns;
__device__ unsigned long long g_w13_arrive_ns;

// Phase 1 (QKV GEMM) tile split, same accumulate-and-dump-once treatment as
// the W13 split above. QKV moves 7.8 MB of MXFP4 per rank and the phase takes
// 8.9 us, but dividing those gives 0.9 TB/s and that ratio is meaningless --
// the split below shows the weight stream is only pre+drain = 2.14 us, i.e.
// 3.6 TB/s against a measured 6.59 TB/s peak, with drain at 0.70 us meaning
// the prefetch is already well hidden. The time is elsewhere. The existing
// MPK_ENABLE_SUBPHASE_TIMING slots cannot say where: _sp_t3b in the kvupd
// kernel is declared and never assigned, so its [3]/[4] terms are garbage.
//
// MEASURED (n=288080, both ranks): pre 1.44, rms 3.27, quant 0.65, drain 0.70,
// mfma 1.40, epi 1.12. RMSNorm is 38% of the tile for 23 KB of traffic, and
// all 80 QKV workgroups norm the SAME 2880-element vector.
//
// Terms, in execution order, non-overlapping and summing to the tile total:
//   pre     Phase A buffer_load_lds issue (4 waves x QKV_LPT loads)
//   rms     ResAddF32 + RMSNorm over the hidden vector
//   quant   FP8 activation quant into LDS
//   drain   s_waitcnt on the Phase A loads + scale scatter
//   mfma    the asm MFMA loop
//   epi     KV_UPD / RoPE epilogue + workspace zeroing
__device__ unsigned long long g_qkv_pre_ns;
__device__ unsigned long long g_qkv_rms_ns;
__device__ unsigned long long g_qkv_quant_ns;
__device__ unsigned long long g_qkv_drain_ns;
__device__ unsigned long long g_qkv_mfma_ns;
__device__ unsigned long long g_qkv_epi_ns;
__device__ unsigned long long g_qkv_n;
// rms split three ways: p1 is the resadd + sum-of-squares pass (2 or 3 HBM
// streams: the f32 workspace, the residual, and under EP the peer's gather
// slot), red is the cross-wave ssq reduction (two __syncthreads), p2 is the
// norm-weight apply. p1 and p2 are separated by a full block barrier, so if
// the cost is HBM latency rather than bytes it shows up as p1 >> p2.
__device__ unsigned long long g_qkv_rp1_ns;
__device__ unsigned long long g_qkv_rred_ns;
__device__ unsigned long long g_qkv_rp2_ns;

// Phase 7b's routing_ready wait, 5.4 us/layer, split by CAUSE. The two
// possibilities need opposite fixes, so the split decides which to build:
//
//   arr   time from the LAST router tile arriving at the top-k barrier back to
//         when the FIRST one did -- i.e. arrival spread. If this dominates,
//         the wait is inherited skew and parallelizing top-k buys nothing.
//   tk    top-k compute itself: one workgroup, 128 logits, plus the 8-flag
//         st_wt fanout. If THIS dominates, it is a genuine serial critical
//         path at bs=1 and worth restructuring.
//
// arr is measured on the top-k completer via a min-timestamp every router tile
// atomicMins as it arrives; tk is that completer's own compute span. Both are
// per-layer, reset by the completer after it reads them.
__device__ unsigned long long g_tk_arr_ns;
__device__ unsigned long long g_tk_tk_ns;
__device__ unsigned long long g_tk_fan_ns;
__device__ unsigned long long g_tk_n;
__device__ unsigned long long g_tk_first = ~0ull;
// MEASURED: arr 5.4-5.7, tk 3.0, fan 1.3-1.8 us. Arrival spread is the larger
// term, so the wait is mostly inherited skew -- but tk+fan = 4.4 us is a
// genuinely serial span with 239 workers parked, and that part is attackable.
// Second-level split of those two, since the fixes differ:
//
//   fence   threadfence_gpu() alone. On gfx950 this is `buffer_wbl2 sc1`, a
//           writeback of the WHOLE L2. Every store it must publish is already
//           st_wt EXCEPT routing_indices' zero-init (moe_topk_softmax_mi300
//           :68, a plain store), so the same argument that killed the Phase 9
//           fence in 57422b1 may apply here.
//   flags   the 9 st_wt_u32 epoch publishes + their s_waitcnt.
//   zinit   the 128-expert routing_indices zero-init loop and its syncthreads,
//           run by the single completer block before it touches any logit.
//   sfmx    buffer_inv + the load/softmax/top-k/blank body.
__device__ unsigned long long g_tk_fence_ns;
__device__ unsigned long long g_tk_eload_ns;
__device__ unsigned long long g_tk_flags_ns;
__device__ unsigned long long g_tk_zinit_ns;
__device__ unsigned long long g_tk_sfmx_ns;
//   inv     the buffer_inv that opens topk_noinline -- a WHOLE-L2 invalidate on
//           the completer's XCD, taken to see the logits the 128 router tiles
//           wrote past L2 with st_wt.
//   lload   the 128-bf16 logit read itself, which is a cold HBM hit by
//           construction and is what the inv is paid for.
__device__ unsigned long long g_tk_inv_ns;
__device__ unsigned long long g_tk_lload_ns;
#endif

#ifdef MPK_ENABLE_SPAN_TIMING
// g_span_active declared early so task kernels (included via task_header.cuh)
// can see it
__device__ int g_span_active;
// O-Proj inner breakdown: 6 checkpoints → 5 sub-phases
// CP0=start, CP1=post-quant, CP2=post-MFMA+reduce, CP3=post-barrier,
// CP4=post-RMSNorm, CP5=done
#define OPROJ_INNER_CPS 6
__device__ unsigned long long
    g_oproj_cp_min[OPROJ_INNER_CPS]; // atomicMin → first worker
__device__ unsigned long long
    g_oproj_cp_max[OPROJ_INNER_CPS]; // atomicMax → last worker
__device__ unsigned long long
    g_oproj_span_accum[OPROJ_INNER_CPS - 1]; // accumulated span (ticks)
__device__ int
    g_oproj_inner_count; // per-iteration WG arrival counter (resets each iter)
__device__ int g_oproj_inner_iters;      // total iterations accumulated
__device__ int g_oproj_inner_reset_flag; // CAS flag for per-iter reset
#endif

#ifdef MPK_NIL_TRIPWIRE
// Sub-phase breadcrumbs from inside the gang task kernels.
//
// The outer loop's breadcrumb only resolves to "somewhere inside
// _execute_gang_task", which is the whole layer. These let a task kernel name
// the phase it was in. Declared before task_header.cuh so the task kernels can
// see them; the pointer is published by the worker loop (every block stores
// the same value, so the race is benign).
//
// A worker block's slot index is blockIdx.x: worker_id defaults to blockIdx.x
// for every block that runs the worker loop.
__device__ unsigned long long *g_tw_dev;

__device__ __forceinline__ void
    mpk_tw_sub(int sub, unsigned long long aux, int tid) {
  if (tid == 0 && g_tw_dev != nullptr) {
    unsigned long long *b =
        g_tw_dev + MPK_TW_HDR + blockIdx.x * MPK_TW_PER_WORKER;
    b[4] = (unsigned long long)sub;
    b[5] = aux;
  }
}
#define MPK_TW_SUB(sub, aux) mpk_tw_sub((sub), (unsigned long long)(aux), tid)
#else
#define MPK_TW_SUB(sub, aux) ((void)0)
#endif

// Intra-layer phase breadcrumb for the worker-state dump.
//
// The 40000+ml marker says which layer a stalled worker is in, but a layer
// spans eight phases and several cross-XCD barriers, so it cannot say which
// barrier is holding. MPK_TW_SUB resolves that, but only under
// MPK_NIL_TRIPWIRE, which costs enough to perturb the timing of the very
// race being chased. This writes the same sub-phase code into the existing
// worker-state slot, so MPK_WORKER_STATE alone gets intra-layer resolution:
// a single relaxed store by tid 0, no atomics and no extra buffers.
//
// Encoded as 50000000 + xcd*100000 + phase*1000 + layer%1000, so one int
// carries all three. The XCD matters because every barrier here is either
// per-XCD (the QKV epoch) or cross-XCD (attn_global, MoE W13->W2), and
// telling those apart is exactly what identifies the blocking worker.
__device__ int *g_ws_dev;

// Sentinel the host pre-stamps into every worker's phase slot before the
// launch. Zero is a legal phase and `in_task` is `wphase >= 0`, so a slot the
// device never wrote is otherwise indistinguishable from a worker parked at
// phase 0 -- which is how a whole rank of never-started workers once read as
// "240 workers IN TASK". Negative, so it decodes as "idle at the scheduler"
// rather than as a task, and far from -1 so the dump can count it separately.
#define MPK_WS_UNWRITTEN (-0x40000000)

// Second sentinel, written by each worker block as the first thing it does
// after the launch and before it joins the XCD-ready count. It separates the
// two failures that both present as "ws-unwritten: 240/240":
//
//   all UNWRITTEN  -> the worker kernel is not on the GPU at all
//   N ENTERED, few
//   UNWRITTEN      -> the kernel launched but some blocks never became
//                     resident, so `worker_xcd_ready_count` never reaches
//                     num_workers and every scheduler is still parked in its
//                     bootstrap wait. Nothing is ever dispatched, so no worker
//                     writes a phase -- including the 239 that ARE resident.
//   all ENTERED    -> everyone is resident and the schedulers are the problem.
//
// A persistent kernel assumes every block is co-resident; 240 workers + 8
// schedulers on 256 CUs leaves no slack for a single misplaced block. This is
// the cheapest way to see that happen: one relaxed store per block, once.
#define MPK_WS_ENTERED (-0x3F000000)

// Scheduler slots live past the three per-worker regions (phase, barrier
// watch, wave mask), 4 ints each. 16 covers every configuration here (8
// schedulers on MI350, one per XCD).
#define MPK_WS_SCHED_SLOTS 16
// Scheduler phase codes, written to slot [+3].
#define MPK_WS_SCHED_ENTERED (-0x3E000000)
#define MPK_WS_SCHED_XCD_READY (-0x3D000000)

__device__ __forceinline__ void
    mpk_ws_phase(int phase, int layer, int xcd, int tid) {
  if (tid == 0 && g_ws_dev != nullptr) {
    __atomic_store_n(&g_ws_dev[blockIdx.x * 4 + 3],
                     50000000 + xcd * 100000 + phase * 1000 + (layer % 1000),
                     __ATOMIC_RELAXED);
  }
}
#ifdef MPK_WORKER_STATE
#define MPK_WS_PHASE(phase, layer, xcd)                                        \
  mpk_ws_phase((phase), (layer), (xcd), tid)
#else
#define MPK_WS_PHASE(phase, layer, xcd) ((void)0)
#endif

// Barrier watch: what a spinning worker is actually waiting *for*.
//
// MPK_WS_PHASE names the phase, which narrows a stall to a barrier, but not
// to a cause. Every barrier here is a "poll until counter >= expected" loop,
// and the two ways it hangs are indistinguishable from the phase alone:
// either the producer never arrived (observed < expected, and the shortfall
// says how many arrivals are missing), or the waiter computed an expected
// value the producer will never reach (observed >= expected but the load is
// reading a stale cache line, or expected ran ahead by a full epoch). This
// records observed/expected so the dump can tell those apart.
//
// Lives in the second half of the worker-state buffer -- 4 more ints per
// worker at [num_workers*4 + blockIdx.x*4] -- to avoid widening RuntimeConfig
// for a debug-only path. Slots: 0 = barrier id, 1 = last observed value,
// 2 = expected value, 3 = spin iterations.
//
// Cost in a healthy run is two relaxed stores per barrier entry: the observed
// value is refreshed only once every MPK_WS_WAIT_REFRESH spins, and these
// polls clear in far fewer than that, so the loop body itself stays untouched.
__device__ int g_ws_nworkers;
#define MPK_WS_WAIT_REFRESH 4096

__device__ __forceinline__ void
    mpk_ws_wait_begin(int barrier_id, int expected, int tid) {
  if (tid == 0 && g_ws_dev != nullptr) {
    int *b = g_ws_dev + g_ws_nworkers * 4 + blockIdx.x * 4;
    __atomic_store_n(&b[0], barrier_id, __ATOMIC_RELAXED);
    __atomic_store_n(&b[2], expected, __ATOMIC_RELAXED);
    __atomic_store_n(&b[3], 0, __ATOMIC_RELAXED);
  }
}

__device__ __forceinline__ void
    mpk_ws_wait_tick(int observed, int spins, int tid) {
  if (tid == 0 && g_ws_dev != nullptr) {
    int *b = g_ws_dev + g_ws_nworkers * 4 + blockIdx.x * 4;
    __atomic_store_n(&b[1], observed, __ATOMIC_RELAXED);
    __atomic_store_n(&b[3], spins, __ATOMIC_RELAXED);
  }
}

// Third quarter of the buffer: four barrier-specific auxiliary values.
//
// observed/expected says a release is missing but not why. For the MoE
// W13->W2 barrier the discriminating value is the raw arrival counter: the
// release fires on (prev % W13_TILES) == W13_TILES-1, on a counter that is
// never reset and is shared by every layer that activates the expert. If the
// counter is sitting on a multiple of W13_TILES the arrivals all landed and
// the release value is wrong; if it is not, arrivals were lost in some earlier
// layer and the modular boundary has been permanently skewed past.
__device__ __forceinline__ void
    mpk_ws_wait_aux(int a0, int a1, int a2, int a3, int tid) {
  if (tid == 0 && g_ws_dev != nullptr) {
    int *b = g_ws_dev + g_ws_nworkers * 8 + blockIdx.x * 4;
    __atomic_store_n(&b[0], a0, __ATOMIC_RELAXED);
    __atomic_store_n(&b[1], a1, __ATOMIC_RELAXED);
    // b[2] and b[3] belong to the per-wave masks below -- not written here.
    (void)a2;
    (void)a3;
  }
}
#define MPK_WS_WAIT_AUX(a0, a1, a2, a3)                                        \
  mpk_ws_wait_aux((a0), (a1), (a2), (a3), tid)

// Per-wave exit mark for a *thread-divergent* poll.
//
// Every other tracer here is tid==0 only, which is enough while a barrier is
// entered and left by the whole block together. The MoE W13->W2 poll is not:
// it deliberately has each thread test the release flag on its own with no
// __syncthreads, so tid 0 can clear the poll and run on while other waves of
// the same block are still spinning. From tid 0's slot that is invisible --
// it reports a straight-line mark past the barrier while the block as a whole
// is still stuck at it, which is exactly how the gx_1 capture looked.
//
// One relaxed store per wave, into the aux quarter's low bits: wave w sets
// bit w when it leaves the poll. A block whose waves all exited reads 0xf.
// Anything less names the waves that never got the release.
__device__ __forceinline__ void mpk_ws_wave_exit(int wave, int tid) {
  if ((tid & 63) == 0 && g_ws_dev != nullptr) {
    int *b = g_ws_dev + g_ws_nworkers * 8 + blockIdx.x * 4;
    __atomic_fetch_or(&b[3], 1 << (wave & 31), __ATOMIC_RELAXED);
  }
}
// Each wave clears its OWN bit in both masks. Doing it this way instead of
// "tid 0 zeroes the word, then __syncthreads" matters: adding a __syncthreads
// to a region whose whole point is that it is thread-divergent changes the
// control flow being measured. The ix_1 capture is not trustworthy for that
// reason. Per-wave clears race with nothing, and need no barrier.
__device__ __forceinline__ void mpk_ws_wave_clear(int wave, int tid) {
  if ((tid & 63) == 0 && g_ws_dev != nullptr) {
    int *b = g_ws_dev + g_ws_nworkers * 8 + blockIdx.x * 4;
    __atomic_fetch_and(&b[2], ~(1 << (wave & 31)), __ATOMIC_RELAXED);
    __atomic_fetch_and(&b[3], ~(1 << (wave & 31)), __ATOMIC_RELAXED);
  }
}
#define MPK_WS_WAVE_EXIT(wave) mpk_ws_wave_exit((wave), tid)
#define MPK_WS_WAVE_CLEAR(wave) mpk_ws_wave_clear((wave), tid)

// Same per-wave mask, one slot over, for arrival at a __syncthreads.
//
// hx_2 showed the W13->W2 poll mask at 0xf -- every wave cleared the barrier
// -- with the block still parked at mark 8303, i.e. inside the FP8 quant.
// The only thing in that function that can block is its trailing
// __syncthreads, so the question becomes which wave fails to reach it.
__device__ __forceinline__ void mpk_ws_wave_sync(int wave, int tid) {
  if ((tid & 63) == 0 && g_ws_dev != nullptr) {
    int *b = g_ws_dev + g_ws_nworkers * 8 + blockIdx.x * 4;
    __atomic_fetch_or(&b[2], 1 << (wave & 31), __ATOMIC_RELAXED);
  }
}
#define MPK_WS_WAVE_SYNC(wave) mpk_ws_wave_sync((wave), tid)

// Straight-line progress mark, for code that is not a spin loop.
//
// The barrier watch only sees workers parked in a poll. A worker stuck
// *between* polls -- inside MoE compute, say -- registers nowhere, which is
// exactly the state the first captured deadlock left its two blockers in.
// This writes the same slots with spins = -1 so the host can tell a mark from
// a barrier and print the last code the worker got past.
// One store, not four. These fire per MoE tile -- several times per worker
// per layer -- and g_ws_dev is pinned *host* memory, so every store is a PCIe
// write. The four-store version cost 2.3x (6.99 vs 3.01 ms/token), which is
// far past the point where the instrumentation perturbs the race it is meant
// to catch. Packed as code*100000 + aux%100000, written to the spins slot as
// a negative so the host can distinguish it from a poll count.
__device__ __forceinline__ void mpk_ws_mark(int code, int aux, int tid) {
  if (tid == 0 && g_ws_dev != nullptr) {
    int *b = g_ws_dev + g_ws_nworkers * 4 + blockIdx.x * 4;
    __atomic_store_n(
        &b[3], -(code * 100000 + (aux % 100000)), __ATOMIC_RELAXED);
  }
}
// Compile-time gate for the inline worker-state dump sites below. The runtime
// null check alone still costs a global load and a branch at ~30 sites, several
// of them in the per-layer dispatch loop; measured 2.386 -> 2.321 ms/iter when
// compiled out. Define MPK_WORKER_STATE to get the dump back.
#ifdef MPK_WORKER_STATE
#define MPK_WS_ON(cfg) ((cfg).precomp_dbg_worker_state != nullptr)
#else
#define MPK_WS_ON(cfg) (false)
#endif

#ifdef MPK_WORKER_STATE
#define MPK_WS_MARK(code, aux) mpk_ws_mark((code), (aux), tid)
#else
#define MPK_WS_MARK(code, aux) ((void)0)
#endif

#ifdef MPK_WORKER_STATE
#define MPK_WS_WAIT_BEGIN(barrier_id, expected)                                \
  mpk_ws_wait_begin((barrier_id), (expected), tid)
#define MPK_WS_WAIT_TICK(observed, spins)                                      \
  do {                                                                         \
    if (((spins) & (MPK_WS_WAIT_REFRESH - 1)) == 0) {                          \
      mpk_ws_wait_tick((observed), (spins), tid);                              \
    }                                                                          \
  } while (0)
#else
#define MPK_WS_WAIT_BEGIN(barrier_id, expected) ((void)0)
#define MPK_WS_WAIT_TICK(observed, spins) ((void)0)
#endif

#if defined(MIRAGE_GRACE_HOPPER)
#include "tasks/hopper/task_header.cuh"
#elif defined(MIRAGE_GRACE_BLACKWELL)
#include "tasks/blackwell/task_header.cuh"
#elif defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
#include "tasks/mi300/task_header.cuh"
#else
#include "tasks/ampere/task_header.cuh"
#endif

// HIP/ROCm compatibility macros and type aliases
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
#include <hip/hip_runtime.h>
// CUDA runtime API -> HIP equivalents
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaSetDevice hipSetDevice
#define cudaMemcpy hipMemcpy
#define cudaMemset hipMemset
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaFuncAttributeMaxDynamicSharedMemorySize                            \
  hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaEventDisableTiming hipEventDisableTiming
// HIP's hipFuncSetAttribute requires
// const void*; wrap to cast function
// pointers
#define cudaFuncSetAttribute(func, attr, value)                                \
  hipFuncSetAttribute(reinterpret_cast<const void *>(func), attr, value)
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventRecord hipEventRecord
#define cudaEventDestroy hipEventDestroy
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStreamDestroy hipStreamDestroy
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError cudaStream_t type alias
typedef hipStream_t cudaStream_t;
// __nanosleep is CUDA-specific; use AMD's s_sleep intrinsic for HIP
__device__ __forceinline__ void __nanosleep(unsigned int ns) {
  // AMD GPU s_sleep: each unit is approximately 64 clock cycles
  // For MI300 at ~1.5GHz, 64 cycles ~= 43ns, so 10ns -> 1 s_sleep unit
  // Use at least 1 to ensure we yield
  unsigned int sleep_units = (ns > 0) ? ((ns + 63) / 64) : 1;
  // s_sleep argument must be a compile-time constant, so we use a small fixed
  // value and loop if needed. This is still much better than a busy loop.
  for (unsigned int i = 0; i < sleep_units; i++) {
    __builtin_amdgcn_s_sleep(1); // Sleep for ~64 cycles
  }
}
#endif

// Returns wall-clock time in nanoseconds using a fixed-frequency real-time
// clock. Unlike clock64() whose frequency varies by GPU model, this gives
// consistent nanosecond timestamps across all platforms.
__device__ __forceinline__ unsigned long long get_wallclock_ns() {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  // s_memrealtime runs at 100 MHz (10 ns/tick) on MI300X
  return __builtin_amdgcn_s_memrealtime() * 10;
#else
  unsigned long long ret;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ret));
  return ret; // globaltimer runs at 1 GHz (1 ns/tick)
#endif
}

using bfloat16 = type::bfloat16_t;
using namespace mirage::runtime;
using namespace kernel;
// Configurations for the MPK runtime
// #define MPK_MAX_NUM_BATCHED_REQUESTS 16
// #define MPK_MAX_NUM_BATCHED_TOKENS 64
// #define MPK_MAX_NUM_PAGES 1024
// #define MPK_PAGE_SIZE 64

#ifndef MPK_PROFILING_NUM_ITERS
#define MPK_PROFILING_NUM_ITERS 0
#endif

// Max tokens per request per iteration during prefill.
// Capped by attention kernel LDS usage:
//   MI300X (gfx942, MPK_TARGET_CC==94): 64 KB LDS  -> 8 tokens
//   MI350X (gfx950, MPK_TARGET_CC==95): 160 KB LDS -> 28 tokens
// If not explicitly defined, use full batch size (no per-request cap).
#ifndef MPK_MAX_TOKENS_PER_REQUEST
#if (MPK_TARGET_CC == 95)
// MI350X: 160 KB LDS leaves room for ~28 prefill tokens at head_dim=64,
// qo_per_kv=8 (see task_register.cc max_tokens_lds calc).
#define MPK_MAX_TOKENS_PER_REQUEST 28
#elif defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300) ||            \
    (MPK_TARGET_CC == 94)
#define MPK_MAX_TOKENS_PER_REQUEST 8
#else
#define MPK_MAX_TOKENS_PER_REQUEST MPK_MAX_NUM_BATCHED_TOKENS
#endif
#endif

#if defined(MIRAGE_GRACE_HOPPER)
#define WORKER_NUM_THREADS 256
#define SINGLE_KERNEL_NUM_THREADS 256
#elif defined(MIRAGE_GRACE_BLACKWELL)
#define WORKER_NUM_THREADS 256
#define SINGLE_KERNEL_NUM_THREADS 256
#elif defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
// AMD MI300: Use 256 threads for better memory-level parallelism
// Benchmarks showed 1.44x speedup vs 128 threads due to more outstanding loads
#define WORKER_NUM_THREADS 256
#define SINGLE_KERNEL_NUM_THREADS 256
#else
#define WORKER_NUM_THREADS 128
#define SINGLE_KERNEL_NUM_THREADS 128
#endif
#define INIT_NUM_THREADS 128

#ifndef CUDA_CHECK
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    hipError_t err = call;                                                     \
    if (err != hipSuccess) {                                                   \
      fprintf(stderr,                                                          \
              "HIP error at %s:%d: %s\n",                                      \
              __FILE__,                                                        \
              __LINE__,                                                        \
              hipGetErrorString(err));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#else
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr,                                                          \
              "CUDA error at %s:%d: %s\n",                                     \
              __FILE__,                                                        \
              __LINE__,                                                        \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#endif
#endif

// Verbose output for debugging (disabled by default for performance)
// Uncomment the line below to enable verbose debugging for AMD
// #define MPK_ENABLE_VERBOSE

// =============================================================================
// AMD MI300X XCD-aware synchronization for reduced cross-XCD polling contention
// =============================================================================
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)

// Number of XCDs on MI300X
constexpr int MI300X_NUM_XCDS = 8;

// Get the XCD (XCC) ID for the current thread
__device__ __forceinline__ int get_current_xcd_id() {
  int xcd_id;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcd_id));
  return xcd_id;
}

// Per-block shared state for XCD info (computed once at block start)
struct XCDBlockInfo {
  int xcd_id;        // XCD this block is running on
  int is_xcd_leader; // 1 if this worker is leader for its XCD, 0 otherwise
};

// Elect XCD leaders - first worker on each XCD becomes leader
// Returns true if this worker should be leader
__device__ __forceinline__ bool
    elect_xcd_leader(RuntimeConfig const &config, int worker_id, int xcd_id) {
  // Use atomicCAS to elect first worker on each XCD as leader
  // -1 means no leader yet
  int old = atomicCAS(&config.xcd_leader_worker[xcd_id], -1, worker_id);
  return (old == -1 || old == worker_id);
}

#else
// Stubs for NVIDIA - no XCD concept
__device__ __forceinline__ int get_current_xcd_id() {
  return 0;
}
#endif

__device__ __forceinline__ void
    _execute_task(TaskDesc const *task_desc,
                  RuntimeConfig const &runtime_config);

// Gang task execution: same as _execute_task but with tile_idx for gang
// dispatch
__device__ __forceinline__ void
    _execute_gang_task(TaskDesc const *task_desc,
                       RuntimeConfig const &runtime_config,
                       int tile_idx);

// Helper: check if a task type is a gang task
__device__ __host__ __forceinline__ bool is_gang_task_type(TaskType t) {
  return t == TASK_GANG_LINEAR_MI300 || t == TASK_GANG_LINEAR_RES_MI300 ||
         t == TASK_GANG_LINEAR_SILU_MI300 || t == TASK_GANG_RMS_NORM_MI300 ||
         t == TASK_GANG_SPLITK_LINEAR_RES_MI300 ||
         t == TASK_GANG_KSPLIT_GEMM_MI300 ||
         t == TASK_GANG_KSPLIT_FINALIZE_MI300 ||
         t == TASK_GANG_ATTN_SPLIT_KV_MI300 ||
         t == TASK_GANG_ATTN_MERGE_MI300 ||
         t == TASK_GANG_MOE_W13_LINEAR_MI300 ||
         t == TASK_GANG_MOE_W2_LINEAR_MI300 ||
         t == TASK_GANG_MOE_W13_LINEAR_MXFP4_MI300 ||
         t == TASK_GANG_MOE_W2_LINEAR_MXFP4_MI300 ||
         t == TASK_GANG_MOE_W13_LINEAR_MXFP8_MI300 ||
         t == TASK_GANG_MOE_W2_LINEAR_MXFP8_MI300 ||
         t == TASK_GANG_MOE_FUSED_MXFP4_MI300 ||
         t == TASK_GANG_MOE_SWIGLU_W2_MXFP4_MI300 ||
         t == TASK_GANG_MOE_W13_SWIGLU_MXFP4_MI300 ||
         t == TASK_GANG_LINEAR_BIAS_MI300 ||
         t == TASK_GANG_SPLITK_LINEAR_RES_BIAS_MI300 ||
         t == TASK_GANG_RMSNORM_LINEAR_BIAS_MI300 ||
         t == TASK_GANG_RMSNORM_LINEAR_MXFP4_BIAS_MI300 ||
         t == TASK_GANG_RMSNORM_LINEAR_MXFP8_BIAS_MI300 ||
         t == TASK_GANG_LINEAR_MXFP4_RES_BIAS_MI300 ||
         t == TASK_GANG_MULSUMRADD_RMSNORM_LINEAR_MXFP4_BIAS_MI300 ||
         t == TASK_GANG_RMSNORM_LINEAR_BIAS_TOPK_MI300 ||
         t == TASK_GANG_RMSNORM_LINEAR_MXFP4_BIAS_KVUPD_MI300 ||
         t == TASK_GANG_MULSUMRADD_RMSNORM_LINEAR_MXFP4_BIAS_KVUPD_MI300 ||
         t == TASK_GANG_RESADDF32_RMSNORM_LINEAR_MXFP4_BIAS_MI300 ||
         t == TASK_GANG_RESADDF32_RMSNORM_LINEAR_MXFP4_BIAS_KVUPD_MI300 ||
         t == TASK_GANG_LINEAR_MXFP4_RES_BIAS_RMSNORM_TOPK_MI300 ||
         t == TASK_GANG_QKV_ATTN_FUSED_MI300 ||
         t == TASK_GANG_OPROJ_TOPK_MOE_FUSED_MI300 ||
         t == TASK_GANG_FULL_LAYER_FUSED_MI300 ||
         t == TASK_GANG_FULL_LAYER_WITH_LMHEAD_FUSED_MI300 ||
         t == TASK_GANG_RMSNORM_LINEAR_MXFP4_BIAS_ARGMAX_MI300 ||
         t == TASK_GANG_MLA_DECODE_MI300 ||
         t == TASK_GANG_MLA_FULL_LAYER_FUSED_MI300;
}

__device__ __forceinline__ bool is_termination_event(size_t event_loc,
                                                     EventDesc e) {
  return (event_loc == 0);
}

__device__ __forceinline__ bool is_nvshmem_event(EventId event_id) {
  return (event_id & EVENT_NVSHMEM_TAG) > 0;
}

__device__ __forceinline__ size_t get_event_gpu_id(EventId event_id) {
  return ((event_id >> 32) & 0xffff);
}

__device__ __forceinline__ size_t get_event_position_index(EventId event_id) {
  return (event_id & 0xffffffff);
}

__device__ __forceinline__ size_t get_task_iteration_num(TaskId task_id) {
  return (task_id >> 32);
}

__device__ __forceinline__ size_t get_task_position_index(TaskId task_id) {
  return (task_id & 0xffffffff);
}

__device__ __forceinline__ TaskId compute_task_id(size_t iteration_num,
                                                  size_t position_index) {
  return ((iteration_num << 32) | position_index);
}

__global__ void init_kernel(RuntimeConfig config) {
  assert(gridDim.x == 1);
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);
  // Only a single thread that initializes everything
  if (threadIdx.x == 0) {
    // initialize metadata
#if defined(MODE_OFFLINE) || defined(MODE_ONLINE)
    for (int i = 0; i < config.total_num_requests; i++) {
      config.step[i] = 0;
    }
    *config.next_request_id = 0;
    for (int i = 0; i < MPK_MAX_NUM_BATCHED_REQUESTS; i++) {
      config.request_ids[i] = -1;
    }
    for (int i = 0; i < MPK_MAX_NUM_BATCHED_REQUESTS + 1; i++) {
      config.qo_indptr_buffer[i] = 0;
      config.paged_kv_indptr_buffer[i] = 0;
    }
    // Page manager
    *config.page_queue_head = 0;
    *config.page_queue_tail = MPK_MAX_NUM_PAGES;
    for (int i = 0; i < MPK_MAX_NUM_PAGES; i++) {
      config.page_queue[i] = i;
    }
#endif
  }
}

__global__ void prepare_kernel(RuntimeConfig config,
                               int end_of_task_graph_event_pos) {
  // Initialize worker queue last task id
  // Each worker now maintains a local and a remote worker queue
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < 2 * config.num_workers;
       i += blockDim.x * gridDim.x) {
    config.worker_queue_last_ready_task_id[i] = 0;
  }
  // Initialize scheduler queue last event id
  // We maintain one extra scheduler queue for the global scheduler
  int num_schedulers =
      config.num_local_schedulers + config.num_remote_schedulers;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_schedulers + 1;
       i += blockDim.x * gridDim.x) {
    config.sched_queue_last_ready_event_id[i] = 0;
    config.sched_queue_next_free_event_id[i] = 0;
  }
  // Initialize all event counters
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < config.num_events;
       i += blockDim.x * gridDim.x) {
    config.all_event_counters[i] = 0;
  }

#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
  // Reset per-XCD local event counters and leaders
  if (config.xcd_local_event_counters != nullptr) {
    int total_local_counters = config.num_xcds * config.num_events;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total_local_counters;
         i += blockDim.x * gridDim.x) {
      config.xcd_local_event_counters[i] = 0;
    }
  }
  // Reset XCD leaders (-1 = no leader)
  if (config.xcd_leader_worker != nullptr) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < config.num_xcds;
         i += blockDim.x * gridDim.x) {
      config.xcd_leader_worker[i] = -1;
    }
  }
  // Reset worker XCD map and ready counter for next kernel launch
  if (config.worker_xcd_map != nullptr) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < config.num_workers;
         i += blockDim.x * gridDim.x) {
      config.worker_xcd_map[i] = -1;
    }
  }
  if (config.worker_xcd_ready_count != nullptr && blockIdx.x == 0 &&
      threadIdx.x == 0) {
    *config.worker_xcd_ready_count = 0;
  }
  // Reset per-XCD per-event task thresholds (scheduler will recompute on first
  // iteration) In precomputed dispatch mode, thresholds are set by host — do
  // NOT reset.
#ifndef MPK_PRECOMPUTED_DISPATCH
  if (config.xcd_event_num_tasks != nullptr) {
    int total_xcd_events = config.num_xcds * config.num_events;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_xcd_events;
         i += blockDim.x * gridDim.x) {
      config.xcd_event_num_tasks[i] = 0;
    }
  }
#endif
#ifdef MPK_PRECOMPUTED_DISPATCH
  // Reset XCD rank counters for template copy
  if (config.precomp_xcd_rank_counter != nullptr && blockIdx.x == 0) {
    for (int i = threadIdx.x; i < config.num_xcds; i += blockDim.x) {
      config.precomp_xcd_rank_counter[i] = 0;
    }
  }
  // Reset gang barrier counters
  if (config.precomp_gang_barrier != nullptr) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < config.num_events;
         i += blockDim.x * gridDim.x) {
      config.precomp_gang_barrier[i] = 0;
    }
  }
#endif
  // Reset combined kernel election state
  if (config.xcd_scheduler_claimed != nullptr && blockIdx.x == 0) {
    for (int i = threadIdx.x; i < config.num_xcds; i += blockDim.x) {
      config.xcd_scheduler_claimed[i] = -1;
    }
  }
  if (config.dynamic_worker_id_counter != nullptr && blockIdx.x == 0 &&
      threadIdx.x == 0) {
    *config.dynamic_worker_id_counter = 0;
  }
#endif

  // Send event to scheduler[0]
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    assert(config.all_events[end_of_task_graph_event_pos].event_type ==
           EVENT_END_OF_TASK_GRAPH);
    config.sched_queue_next_free_event_id[0] = 1;
    config.sched_queues[0][0] = end_of_task_graph_event_pos;
    config.sched_queue_last_ready_event_id[0] = 1;
  }
}

#ifdef MPK_SPEC_DECODE
// Speculative decode bookkeeping. Written only by prepare_next_batch, which is
// single-threaded on one scheduler, so these need no atomics.
//
// g_spec_committed is the denominator that matters: the wall clock is per
// ITERATION, and speculation buys a cheaper token by making the iteration more
// expensive, so ms/iter on its own cannot say whether it paid. Report
// ms/token = total_ms / committed.
__device__ int g_spec_proposed;
__device__ int g_spec_accepted;
__device__ int g_spec_committed;
// Decode-only wall clock. [FWD_PASS_TOTAL] averages over prefill chunks too,
// and prefill is a different iteration shape, so dividing that by a decode
// token count would be a category error. g_spec_iter_is_decode is set by
// prepare_next_batch when it dispatches a speculative row, and is read one
// iteration later -- at the point where that iteration's duration is known.
__device__ unsigned long long g_spec_decode_ns;
__device__ int g_spec_decode_iters;
__device__ int g_spec_iter_is_decode;

// How many rows a decode iteration dispatches: 1 verify row + (WIDTH-1) draft
// rows. Only 2 is implemented -- a deeper draft tree needs the draft model to
// run WIDTH-1 times, and the accept scan below is written for one draft.
#ifndef MPK_SPEC_WIDTH
#define MPK_SPEC_WIDTH 2
#endif
static_assert(MPK_SPEC_WIDTH == 2, "only a single draft token is implemented");
#endif

#ifdef MODE_OFFLINE
// TODO: parallelize this processing
__device__ __forceinline__ bool
    prepare_next_batch(RuntimeConfig const &config) {
  __shared__ int smem_kv_indices[MPK_MAX_NUM_PAGES];
  int page_queue_head = *config.page_queue_head;
  int page_queue_tail = *config.page_queue_tail;
#ifdef MPK_SPEC_DECODE
  // Describes the iteration this call is about to set up, and is read one
  // iteration later, when that iteration's duration is known.
  g_spec_iter_is_decode = 0;
#endif
  // Step 1: finalize previous batch
  for (int i = 0; i < MPK_MAX_NUM_BATCHED_REQUESTS; i++) {
    int16_t request_id = config.request_ids[i];
    if (request_id != -1) {
      // Step 1.1: move output_tokens to tokens
      int step = config.step[request_id];
      int qo_indptr = config.qo_indptr_buffer[i];
      int num_tokens = config.qo_indptr_buffer[i + 1] - qo_indptr;
      int prompt_len = config.prompt_length[request_id];
      // How many of the num_tokens rows we keep. Equal to num_tokens
      // everywhere except a speculative decode iteration, where row 1 was
      // conditioned on a guess and is only real if the guess was right.
      int num_committed = num_tokens;
#ifdef MPK_SPEC_DECODE
      if (step >= prompt_len && num_tokens > 1) {
        // Row j predicts the token at position step+j+1. Row 1's input was
        // the draft written into tokens[step+1] at the end of the previous
        // iteration, so row 1's output is the true continuation iff that
        // draft is what row 0 actually predicts. Read the draft BEFORE the
        // commit loop overwrites it.
        long long draft =
            config.tokens[request_id * MPK_MAX_SEQ_LENGTH + step + 1];
        long long verified0 = config.output_tokens[qo_indptr];
        g_spec_proposed++;
        if (draft == verified0) {
          g_spec_accepted++;
          num_committed = 2;
          // EOS as the first of an accepted pair: the second token is past
          // the end of the sequence. Committing it would make speculation
          // change the output, which is the one thing it must never do.
          if (verified0 == config.eos_token_id) {
            num_committed = 1;
          }
        } else {
          num_committed = 1;
        }
      }
      // Generated tokens only: a prefill chunk "commits" prompt positions.
      if (step >= prompt_len) {
        g_spec_committed += num_committed;
      }
#endif
      for (int j = 0; j < num_committed; j++) {
        if (step + j + 1 >= prompt_len &&
            step + j + 1 < config.max_seq_length) {
          config.tokens[request_id * MPK_MAX_SEQ_LENGTH + step + j + 1] =
              config.output_tokens[qo_indptr + j];
        }
      }
      config.step[request_id] = step + num_committed;
#ifdef MPK_ENABLE_PROFILING
      // num_committed, not num_tokens: under speculation the loop bound is a
      // token count, so a rejected draft row must not shorten the run.
      if (config.profiling_num_iters > 0 &&
          step + num_committed >= config.profiling_num_iters)
#else
      if ((step + num_committed + 1 >= config.max_seq_length) ||
          ((config.tokens[request_id * MPK_MAX_SEQ_LENGTH + step +
                          num_committed] == config.eos_token_id) &&
           (step + num_committed >= prompt_len)))
#endif
      {
        // Request is done
        config.request_ids[i] = -1;
        // Free pages
        int kv_indptr = config.paged_kv_indptr_buffer[i];
        int num_pages = config.paged_kv_indptr_buffer[i + 1] - kv_indptr;
        for (int j = 0; j < num_pages; j++) {
          config.page_queue[page_queue_tail % MPK_MAX_NUM_PAGES] =
              config.paged_kv_indices_buffer[kv_indptr + j];
          page_queue_tail++;
        }
      }
    }
  }

  // Step 2: copy kv_indices to shared mem
  int num_pages = config.paged_kv_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];
  for (int i = 0; i < num_pages; i++) {
    smem_kv_indices[i] = config.paged_kv_indices_buffer[i];
  }

  // Step 3: prepare next batch
  int num_reqs = 0, num_tokens = 0;
  num_pages = 0;
  for (int i = 0; i < MPK_MAX_NUM_BATCHED_REQUESTS; i++) {
    int16_t request_id = config.request_ids[i];
    if (request_id != -1) {
      int kv_indptr = config.paged_kv_indptr_buffer[i];
      int num_old_pages = config.paged_kv_indptr_buffer[i + 1] - kv_indptr;
      config.request_ids[num_reqs] = request_id;
      config.qo_indptr_buffer[num_reqs] = num_tokens;
      config.paged_kv_indptr_buffer[num_reqs] = num_pages;
      int step = config.step[request_id];
      int num_new_tokens = config.prompt_length[request_id] - step;
      if (num_new_tokens > 0) {
        // Prefill requests: cap per-request tokens to avoid attention LDS
        // overflow
        num_new_tokens = min(num_new_tokens,
                             min(MPK_MAX_TOKENS_PER_REQUEST,
                                 MPK_MAX_NUM_BATCHED_TOKENS - num_tokens));
      } else {
        // Decode requests
        num_new_tokens = min(1, MPK_MAX_NUM_BATCHED_TOKENS - num_tokens);
#ifdef MPK_SPEC_DECODE
        // Speculative decode: dispatch the verify row plus one draft row, and
        // stage the draft where the input_tokens copy below already looks --
        // tokens[step+1]. That position is exactly where the verify row's own
        // output will land, so a rejected draft is overwritten by the truth on
        // the very next line of the finalize pass and nothing has to unwind.
        //
        // The draft row's KV lands at sequence position step+1. On a reject
        // the next iteration's verify row is that same position and rewrites
        // it, so no KV invalidation is needed either; on an accept it is
        // already the KV of the token we just committed.
        //
        // MPK_SPEC_DRAFT_SELF=1 is the null draft (propose the last committed
        // token). It exists so the harness -- 2-row dispatch, accept scan,
        // step advance, page growth -- can be gated for byte-identical output
        // BEFORE the MTP layer exists to fill the row with a real guess. It
        // will accept occasionally on repeated tokens, which is what exercises
        // the accept branch.
        if (num_new_tokens == 1 &&
            MPK_MAX_NUM_BATCHED_TOKENS - num_tokens >= MPK_SPEC_WIDTH &&
            step + MPK_SPEC_WIDTH < config.max_seq_length) {
#ifdef MPK_SPEC_ORACLE
          // Oracle draft: leave tokens[step+1] alone. The host has pre-filled
          // the token buffer past the prompt with a reference continuation,
          // so the draft is by construction what the verify row is about to
          // predict. This is the acceptance == 1.0 arm: it gates the accept
          // branch end-to-end (the output must still come out identical,
          // because every accepted token is re-derived from output_tokens,
          // not copied from the oracle) and it measures the ceiling on
          // ms/token before the MTP draft layer exists to chase it.
          //
          // The commit loop overwrites tokens[step+1] with the verify row's
          // own output, so a wrong oracle degrades to a normal reject rather
          // than leaking the reference into the answer.
#else
          config.tokens[request_id * MPK_MAX_SEQ_LENGTH + step + 1] =
              config.spec_draft_tokens != nullptr
                  ? config.spec_draft_tokens[i]
                  : config.tokens[request_id * MPK_MAX_SEQ_LENGTH + step];
#endif
          num_new_tokens = MPK_SPEC_WIDTH;
          g_spec_iter_is_decode = 1;
        }
#endif
      }
      // Move tokens to input_tokens
      for (int j = 0; j < num_new_tokens; j++) {
        config.input_tokens[num_tokens + j] =
            config.tokens[request_id * MPK_MAX_SEQ_LENGTH + step + j];
      }
      // Prepare page indptrs
      int num_new_pages =
          (step + num_new_tokens + MPK_PAGE_SIZE - 1) / MPK_PAGE_SIZE;
      {
        int last_len = (step + num_new_tokens) % MPK_PAGE_SIZE;
        config.paged_kv_last_page_len_buffer[num_reqs] =
            last_len == 0 ? MPK_PAGE_SIZE : last_len;
      }
      for (int j = 0; j < num_old_pages; j++) {
        config.paged_kv_indices_buffer[num_pages + j] =
            smem_kv_indices[kv_indptr + j];
      }
      for (int j = num_old_pages; j < num_new_pages; j++) {
        config.paged_kv_indices_buffer[num_pages + j] =
            config.page_queue[page_queue_head % MPK_MAX_NUM_PAGES];
        page_queue_head++;
      }
      num_pages += num_new_pages;
      num_tokens += num_new_tokens;
      num_reqs++;
    }
  }

  // Add new prefill requests until we reach capacity
  while (num_reqs < MPK_MAX_NUM_BATCHED_REQUESTS &&
         num_tokens < MPK_MAX_NUM_BATCHED_TOKENS) {
    int next_request_id = *config.next_request_id;
    if (next_request_id >= config.total_num_requests) {
      break;
    }
    config.request_ids[num_reqs] = next_request_id;
    config.qo_indptr_buffer[num_reqs] = num_tokens;
    config.paged_kv_indptr_buffer[num_reqs] = num_pages;
    // Prefill request: cap per-request tokens to avoid attention LDS overflow
    int num_new_tokens = min(config.prompt_length[next_request_id],
                             min(MPK_MAX_TOKENS_PER_REQUEST,
                                 MPK_MAX_NUM_BATCHED_TOKENS - num_tokens));
    // Move tokens to input tokens
    for (int j = 0; j < num_new_tokens; j++) {
      config.input_tokens[num_tokens + j] =
          config.tokens[next_request_id * MPK_MAX_SEQ_LENGTH + j];
    }
    int num_new_pages = (num_new_tokens + MPK_PAGE_SIZE - 1) / MPK_PAGE_SIZE;
    {
      int last_len = num_new_tokens % MPK_PAGE_SIZE;
      config.paged_kv_last_page_len_buffer[num_reqs] =
          last_len == 0 ? MPK_PAGE_SIZE : last_len;
    }
    for (int j = 0; j < num_new_pages; j++) {
      config.paged_kv_indices_buffer[num_pages + j] =
          config.page_queue[page_queue_head % MPK_MAX_NUM_PAGES];
      page_queue_head++;
    }
    num_tokens += num_new_tokens;
    num_pages += num_new_pages;
    num_reqs++;
    *config.next_request_id = next_request_id + 1;
  }

  // Step 4: Update all unused requests slots
  for (int i = num_reqs; i < MPK_MAX_NUM_BATCHED_REQUESTS; i++) {
    config.request_ids[i] = -1;
  }
  for (int i = num_reqs; i <= MPK_MAX_NUM_BATCHED_REQUESTS; i++) {
    config.qo_indptr_buffer[i] = num_tokens;
    config.paged_kv_indptr_buffer[i] = num_pages;
  }

  // Step 5: update page head tail
  *config.page_queue_head = page_queue_head;
  *config.page_queue_tail = page_queue_tail;

  if (num_tokens == 0) {
    return false;
  } else {
    return true;
  }
}
#endif

#ifdef MODE_ONLINE
__device__ __forceinline__ bool
    prepare_next_batch(RuntimeConfig const &config) {
  int step = config.step[0];
#ifdef MPK_ENABLE_VERBOSE
  printf("step: %d, new_token_num(%p): %d, new_token_ids:\n",
         step,
         config.new_token_nums,
         config.new_token_nums[0]);
  for (int i = 0; i < config.new_token_nums[0]; i++) {
    printf("%lld ", config.tokens[step + 1 + i]);
  }
  printf("\n");
#endif
  config.step[0] = step + config.new_token_nums[0];

#ifdef MPK_ENABLE_PROFILING
  if (config.profiling_num_iters > 0 &&
      step + 1 >= config.profiling_num_iters) {
    return false;
  }
  return true;
#else
  if ((step + 2 >= config.max_seq_length) ||
      (config.tokens[step + 1] == config.eos_token_id)) {
    return false;
  } else {
    return true;
  }
#endif
}
#endif

#ifdef MODE_ONLINE_NOTOKEN
__device__ __forceinline__ bool prepare_next_batch(RuntimeConfig const &config,
                                                   size_t iteration_num = 0) {
  // TODO: iteration_num is a current workaround
  // We may consider split EVENT_END_OF_TASK_GRAPH into
  // EVENT_END_OF_TASK_GRAPH and EVENT_START_OF_TASK_GRAPH
  if (iteration_num > 0) {
    return false;
  } else { // iteration_num == 0
    return true;
  }
}
#endif

__device__ __forceinline__ int get_rand_sched_id(size_t event_index,
                                                 int worker_id,
                                                 int num_workers,
                                                 int num_schedulers,
                                                 int xcd_id = 0) {
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
  // XCD-aligned: worker signals the scheduler on its own XCD.
  // scheduler_kernel block k runs on XCD k, so sched_id == XCD ID.
  return xcd_id;
#else
  size_t x = worker_id;
  return x / ((num_workers + num_schedulers - 1) / num_schedulers);
#endif
}

__device__ __forceinline__ void
    get_first_last_ids(unsigned long long int num_elements,
                       unsigned long long int num_workers,
                       unsigned long long int my_id,
                       unsigned long long int *my_first_element,
                       unsigned long long int *my_last_element) {
  unsigned long long int num_elements_per_worker = num_elements / num_workers;
  unsigned long long int reminder = num_elements % num_workers;
  if (my_id < reminder) {
    *my_first_element = (num_elements_per_worker + 1) * my_id;
    *my_last_element = *my_first_element + num_elements_per_worker + 1;
  } else {
    *my_first_element = num_elements_per_worker * my_id + reminder;
    *my_last_element = *my_first_element + num_elements_per_worker;
  }
}

__device__ __forceinline__ void terminate_schedulers(RuntimeConfig config) {
  // Event ID 0 is the termination event
  int num_schedulers =
      config.num_local_schedulers + config.num_remote_schedulers;
  for (int i = 0; i < num_schedulers; i++) {
    // size_t last_event_id =
    //     atomicAdd(&config.sched_queue_next_free_event_id[i], 1);
    size_t last_event_id =
        atom_add_release_gpu_u64(&config.sched_queue_next_free_event_id[i], 1);
    st_relaxed_gpu_u64(
        &config.sched_queues[i][last_event_id % config.per_sched_queue_len], 0);
    // Memory fence to ensure event store is visible before signaling
    threadfence_gpu();
    // CAS loop to update last_ready_event_id in order
    size_t old;
    do {
      old = atom_cas_release_gpu_u64(&config.sched_queue_last_ready_event_id[i],
                                     last_event_id,
                                     last_event_id + 1);
    } while (old != last_event_id);
  }
}

__device__ __forceinline__ void worker_checker(RuntimeConfig config) {
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);
  // Each worker SM serves a single worker
  // Each scheduelr SM serves four schedulers
  // int num_schedulers =
  //    config.num_local_schedulers + config.num_remote_schedulers;

  assert(gridDim.x == config.num_workers);
  assert(config.num_workers <= MAX_NUM_WORKERS);
  // We will reinterpret TaskDesc as an array of integers to
  // collectively load it from device to shared memory
  static_assert(sizeof(TaskDesc) % sizeof(int) == 0);
}

__device__ __forceinline__ void scheduler_checker(RuntimeConfig config) {
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);
  // Each worker SM serves a single worker
  // Each scheduelr SM serves four schedulers
  // int num_schedulers =
  //    config.num_local_schedulers + config.num_remote_schedulers;

  assert(config.num_workers <= MAX_NUM_WORKERS);
}

__device__ __forceinline__ void persistent_checker(RuntimeConfig config) {
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);
  // Each worker SM serves a single worker
  // Each scheduelr SM serves four schedulers
  int const num_schedulers =
      config.num_local_schedulers + config.num_remote_schedulers;
  int const num_schedulers_per_sm = std::min((int)blockDim.x / 32, 4);
  assert(num_schedulers % num_schedulers_per_sm == 0);
  assert(gridDim.x ==
         config.num_workers + num_schedulers / num_schedulers_per_sm);
  assert(config.num_workers <= MAX_NUM_WORKERS);
  // We will reinterpret TaskDesc as an array of integers to
  // collectively load it from device to shared memory
  static_assert(sizeof(TaskDesc) % sizeof(int) == 0);
  // assert(blockDim.x >= 128);
}

// ── Device-global task-time accumulators (all workers atomicAdd) ──
#ifdef MPK_ENABLE_DEVICE_TASK_ACCUM
// Slots: 0=QKV 1=GATE_UP 2=O_PROJ 3=DOWN_PROJ 4=RMS 5=SILU 6=ATTN_SPLIT
// 7=ATTN_MERGE 8=OTHER MoE: 9=MOE_W13 10=MOE_W2 11=MOE_TOPK(+Router)
// 12=MOE_SWIGLU 13=MOE_MULSUM 14=MOE_BIAS Gap: 15=SCHED_GAP (pure scheduling:
// task end -> next task fetched, excl dep wait)
//      16=DEP_WAIT (dependency wait: task fetched -> dep satisfied)
#define DACCUM_SLOTS 17
__device__ unsigned long long g_daccum_ns[DACCUM_SLOTS];
__device__ unsigned long long g_daccum_cnt[DACCUM_SLOTS];
__device__ int g_daccum_active;       // 0=prefill, 1=decode
__device__ int g_daccum_decode_iters; // count of decode iterations
#endif

// ── Per-stage wall-clock span timing via first-worker-start / last-worker-end
// ──
#ifdef MPK_ENABLE_SPAN_TIMING
// 4 pipeline stages: 0=QKV_KVUPD 1=CK_FMHA 2=OPROJ_TOPK 3=MOE_FUSED
#define SPAN_STAGES 4
// Accumulated event-to-event span (us) across all decode layers
__device__ unsigned long long g_span_accum_us[SPAN_STAGES];
// Accumulated compute span = last_worker_end - first_worker_start (us)
__device__ unsigned long long g_span_compute_us[SPAN_STAGES];
__device__ unsigned long long
    g_span_prev_event_ticks; // previous event fire timestamp
// Per-stage first-worker-start tracking
__device__ unsigned long long
    g_span_first_start[SPAN_STAGES];           // first worker timestamp
__device__ int g_span_first_flag[SPAN_STAGES]; // 0=not set, 1=set
__device__ int g_span_event_count;
// g_span_active declared before task_header.cuh (line ~45)

// Map task type to span stage index (inline, used by worker loop)
__device__ __forceinline__ int _span_stage_for_task(int task_type,
                                                    int variant_id) {
  switch (task_type) {
    case TASK_GANG_RMSNORM_LINEAR_MXFP4_BIAS_KVUPD_MI300:
    case TASK_GANG_MULSUMRADD_RMSNORM_LINEAR_MXFP4_BIAS_KVUPD_MI300:
    case TASK_GANG_RESADDF32_RMSNORM_LINEAR_MXFP4_BIAS_KVUPD_MI300:
    case TASK_GANG_RMSNORM_LINEAR_MXFP4_BIAS_MI300:
    case TASK_GANG_RMSNORM_LINEAR_MXFP8_BIAS_MI300:
    case TASK_GANG_MULSUMRADD_RMSNORM_LINEAR_MXFP4_BIAS_MI300:
    case TASK_GANG_RESADDF32_RMSNORM_LINEAR_MXFP4_BIAS_MI300:
    case TASK_GANG_RMSNORM_LINEAR_MXFP4_BIAS_ARGMAX_MI300:
      return (variant_id == 0) ? 0 : -1;
    case TASK_PAGED_ATTENTION_CK_FMHA_SPLIT_KV_MI300:
    case TASK_PAGED_ATTENTION_SPLIT_KV_MI300:
    case TASK_GANG_ATTN_SPLIT_KV_MI300:
    case TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_MI300:
    case TASK_GANG_ATTN_MERGE_MI300:
      return 1;
    case TASK_GANG_LINEAR_MXFP4_RES_BIAS_RMSNORM_TOPK_MI300:
      return 2;
    case TASK_GANG_MOE_FUSED_MXFP4_MI300:
    case TASK_GANG_MOE_SWIGLU_W2_MXFP4_MI300:
    case TASK_GANG_MOE_W2_LINEAR_MXFP4_MI300:
      return 3;
    default:
      return -1;
  }
}
#endif

// Deferred FWD_PASS log: store per-iteration timing, print at termination.
//
// The ring is bounded, so runs longer than FWDPASS_LOG_MAX iterations cannot
// keep every sample. They must not silently keep only the *first* window
// either: per-iter latency grows with sequence length, so truncating to the
// first 8k iterations biased every reported average downward at long seq len
// (a 49k run reported the latency of its cheapest 8k iterations).
//
// The ring therefore stride-decimates: when it fills it compacts in place,
// keeping every 2nd sample and doubling the stride. Coverage stays uniform
// across the entire run at any length, which is what makes latency-vs-seqlen
// reconstructible. g_fwdpass_stride is the current decimation factor; the
// totals accumulate over every iteration regardless.
#define FWDPASS_LOG_MAX 8192
__device__ int g_fwdpass_count;
__device__ int g_fwdpass_stride = 1;
__device__ unsigned long long
    g_fwdpass_time_ns[FWDPASS_LOG_MAX];           // iter duration
__device__ int g_fwdpass_tokens[FWDPASS_LOG_MAX]; // num_active_tokens
// Untruncated aggregates: cover every iteration regardless of ring capacity.
__device__ int g_fwdpass_dropped;
__device__ unsigned long long g_fwdpass_total_ns;
__device__ int g_fwdpass_total_iters;

__device__ __forceinline__ void execute_worker(RuntimeConfig config,
                                               int assigned_worker_id = -1) {
  // Make sure overall smem usage here do not exceed 3KB
  // last_task_pos: 2 * 8 = 16 B
  // next_task_pos: 2 * 8 = 16 B
  // worker_queue_ids: 2 * 4 = 8 B
  // worker_queues: 2 * 8 = 16 B
  // remaining: 3016 B

  constexpr int TASK_DESCS_BUFFER_LENGTH = std::min(
      (mirage::runtime::WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE - 56) /
          (int)(sizeof(TaskDesc) + sizeof(TaskId)),
      16);
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
  // HIP: Use char arrays and cast to avoid constructor calls on __shared__
  // arrays HIP doesn't support default construction of __shared__ array
  // elements HIP doesn't support alignas with __shared__, use __align__ instead
  static_assert(TASK_DESCS_BUFFER_LENGTH <= 16, "Buffer length exceeds 16");
  __shared__ __align__(
      alignof(TaskDesc)) char task_descs_storage[16 * sizeof(TaskDesc)];
  __shared__ __align__(
      alignof(TaskId)) char task_ids_storage[16 * sizeof(TaskId)];
  TaskDesc *task_descs = reinterpret_cast<TaskDesc *>(task_descs_storage);
  TaskId *task_ids = reinterpret_cast<TaskId *>(task_ids_storage);
#else
  __shared__ TaskDesc task_descs[TASK_DESCS_BUFFER_LENGTH];
  __shared__ TaskId task_ids[TASK_DESCS_BUFFER_LENGTH];
#endif
  __shared__ TaskId *worker_queues[2];
  __shared__ int worker_queue_ids[2];
  __shared__ size_t next_task_pos[2];
  __shared__ size_t last_task_pos[2];

#ifdef MPK_ENABLE_PROFILING
  PROFILER_CLOSURE_PARAMS_DECL;
  PROFILER_INIT(static_cast<uint64_t *>(config.profiler_buffer),
                0,
                1,
                (threadIdx.x % WORKER_NUM_THREADS == 0));

#endif
  int const worker_id =
      (assigned_worker_id >= 0) ? assigned_worker_id : blockIdx.x;
#ifdef MPK_NIL_TRIPWIRE
  // Hand the tripwire buffer to the task kernels, which take input/output
  // pointer arrays rather than the RuntimeConfig and so cannot reach it.
  // Every block stores the same value, so the write race is benign.
  // MPK_TW_SUB indexes by blockIdx.x, which equals worker_id here (nothing
  // passes assigned_worker_id), and MPK_TW_SLOTS covers 256 blocks.
  if (threadIdx.x == 0) {
    g_tw_dev = config.tripwire;
  }
  __syncthreads();
#endif
#ifdef MPK_WORKER_STATE
  // Same publication for the worker-state phase breadcrumb (see MPK_WS_PHASE).
  // Indexed by blockIdx.x, matching the 4-int-per-worker dump layout.
  // Gated like the tripwire above: with MPK_WORKER_STATE off nothing reads
  // g_ws_dev, so this store and its __syncthreads are pure overhead.
  if (threadIdx.x == 0) {
    g_ws_dev = config.precomp_dbg_worker_state;
    // Stride to the barrier-watch half of the buffer (see MPK_WS_WAIT_BEGIN).
    g_ws_nworkers = config.num_workers;
    // Residency mark. Must be before the worker_xcd_ready_count atomicAdd
    // below: the schedulers block until that count reaches num_workers, so if
    // one block is not resident every OTHER block parks at the scheduler with
    // no task and no phase. See MPK_WS_ENTERED.
    if (g_ws_dev != nullptr) {
      __atomic_store_n(
          &g_ws_dev[blockIdx.x * 4 + 3], MPK_WS_ENTERED, __ATOMIC_RELAXED);
    }
  }
  __syncthreads();
#endif
  worker_queues[0] = config.worker_queues[worker_id];
  worker_queue_ids[0] = worker_id;
  int num_worker_queues = 1;
  if (config.num_gpus > 1) {
    worker_queues[num_worker_queues] =
        config.worker_queues[worker_id + config.num_workers];
    worker_queue_ids[num_worker_queues] = worker_id + config.num_workers;
    num_worker_queues++;
  }

#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
  // XCD-aware polling: detect XCD and elect leader
  __shared__ int block_xcd_id;
  __shared__ int block_is_xcd_leader;

  if (threadIdx.x == 0) {
    block_xcd_id = get_current_xcd_id();
    // Write this worker's XCD ID to shared map for scheduler to read
    if (config.worker_xcd_map != nullptr) {
      config.worker_xcd_map[worker_id] = block_xcd_id;
      if (worker_id < 8) {
        printf("[WORKER_XCD] worker_id=%d block=%d xcd=%d\n",
               worker_id,
               (int)blockIdx.x,
               block_xcd_id);
      }
      // Fence ensures map write is visible before incrementing ready counter
      threadfence_gpu();
      atomicAdd(config.worker_xcd_ready_count, 1);
    }
    // Elect leader: first worker on each XCD becomes leader
    // Leader polls global counter, others poll local counter
    if (config.xcd_leader_worker != nullptr) {
      block_is_xcd_leader =
          elect_xcd_leader(config, worker_id, block_xcd_id) ? 1 : 0;
    } else {
      block_is_xcd_leader = 0; // No hierarchical polling if not configured
    }
  }
  __syncthreads();

  int const xcd_id = block_xcd_id;
  int const is_xcd_leader = block_is_xcd_leader;

#ifdef MPK_ENABLE_GANG_TASKS
  // XCD-local rank for gang tasks: computed lazily on first gang task.
  // -1 means not yet computed. Safe to compute after scheduler starts
  // dispatching (all worker_xcd_map entries are valid by then).
  __shared__ int block_xcd_local_rank;
  __shared__ int block_workers_on_xcd;
  if (threadIdx.x == 0) {
    block_xcd_local_rank = -1;
    block_workers_on_xcd = 0;
  }
#endif
#endif
#if !defined(__HIP_PLATFORM_AMD__) && !defined(MIRAGE_AMD_MI300)
  int const xcd_id = 0; // NVIDIA: no XCD concept
#endif

  if (threadIdx.x == 0) {
    for (int i = 0; i < 2; i++) {
      next_task_pos[i] = 0;
    }
    for (int i = 0; i < 2; i++) {
      last_task_pos[i] = 0;
    }
    // num_loaded_tasks = 0;
  }

  int queue_pos = 0, queue_len = 0;
  unsigned long long prev_begin_clk = 0;
  int fwd_pass_count = 0;
#ifdef MPK_ENABLE_GAP_TIMING
  unsigned long long _prev_task_end = 0; // for inter-task gap timing
#endif
#ifdef MPK_ENABLE_PROFILING
  size_t task_counter = 0;
#endif
#ifdef MPK_PRECOMPUTED_DISPATCH
  __shared__ int pc_pos;
  __shared__ size_t pc_iter;
  __shared__ int pc_my_len;
  __shared__ int pc_terminated;
  __shared__ int pc_xcd_rank;  // Persistent template rank within XCD (for gang
                               // tile assignment)
  __shared__ int pc_has_begin; // 1 if this worker's queue starts with
                               // begin_task_graph (position 1)
  size_t *pc_my_queue =
      config.precomp_queue + worker_id * config.precomp_max_tpw;

  // Copy XCD template to per-worker queue based on runtime hardware XCD.
  // Worker atomically claims a rank within its hardware XCD, then copies
  // the template for [hardware_xcd][rank] to its own queue slot.
  {
    __shared__ int pc_my_rank;
    if (threadIdx.x == 0) {
      pc_my_rank = atomicAdd(&config.precomp_xcd_rank_counter[xcd_id], 1);
    }
    __syncthreads();
    int my_rank = pc_my_rank;
    int wpx = config.precomp_workers_per_xcd;
    int tmpl_flat = xcd_id * wpx + my_rank;
    int tmpl_len = config.precomp_xcd_template_len[tmpl_flat];
    size_t *tmpl_src =
        config.precomp_xcd_template + tmpl_flat * config.precomp_max_tpw;
    // Parallel copy: each thread copies a chunk
    for (int i = threadIdx.x; i < tmpl_len; i += blockDim.x) {
      pc_my_queue[i] = tmpl_src[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      config.precomp_queue_len[worker_id] = tmpl_len;
      pc_pos = 0;
      pc_iter = 1;
      pc_my_len = tmpl_len;
      pc_terminated = 0;
      pc_xcd_rank = my_rank;
      pc_has_begin = (tmpl_len > 0 && tmpl_src[0] == 1)
                         ? 1
                         : 0; // position 1 = begin_task_graph
      if (worker_id < 8) {
        printf("[PC_INIT] worker=%d xcd=%d rank=%d tmpl_len=%d\n",
               worker_id,
               xcd_id,
               my_rank,
               tmpl_len);
      }
    }
  }
#endif
  __syncthreads();

  // Timing instrumentation
#ifdef MPK_ENABLE_TIMING
  unsigned long long total_poll_cycles = 0;
  unsigned long long total_dep_wait_cycles = 0;
  unsigned long long total_exec_cycles = 0;
  unsigned long long total_signal_cycles = 0;
  int total_poll_iters = 0;
  int total_dep_wait_iters = 0;
  int total_tasks_executed = 0;
  // Per-task-type timing (major types only)
  unsigned long long linear_cycles = 0, linear_res_cycles = 0;
  unsigned long long attention_cycles = 0, rms_cycles = 0, silu_cycles = 0;
  unsigned long long fused_cycles = 0;
  int linear_count = 0, linear_res_count = 0;
  int attention_count = 0, rms_count = 0, silu_count = 0;
  int fused_count = 0;
#endif

  while (true) {
    // fetch next task from a task queue if task_descs is empty
    if (queue_pos == queue_len) {
#ifdef MPK_PRECOMPUTED_DISPATCH
      // ===== Pre-filled queue path =====
      if (threadIdx.x == 0) {
        if (pc_pos >= pc_my_len) {
          // All tasks done for this iteration — wrap around
          pc_pos = 0;
          pc_iter++;
        }
        // All workers must wait at iteration boundary because orphan tasks
        // (e.g., embed) have no dependency event and would execute immediately,
        // corrupting activation buffers if the previous iteration isn't done.
        while (__atomic_load_n(config.precomp_iter_ready, __ATOMIC_RELAXED) <
               pc_iter) {
          if (__atomic_load_n(config.precomp_terminate, __ATOMIC_RELAXED)) {
            pc_terminated = 1;
            break;
          }
          __builtin_amdgcn_s_sleep(1);
        }
        // Terminate is checked INSIDE the wait loop only, and that is not
        // enough: the scheduler's shutdown path sets terminate and then bumps
        // iter_ready (see the END_OF_TASK_GRAPH handler). A worker that
        // samples iter_ready after that bump never enters the loop at all, so
        // it falls through and starts an iteration the scheduler will never
        // produce. Its first task with a dependent event then waits on a
        // counter that is one arrival short forever, while every worker that
        // did take the break returns -- one resident workgroup, 100% GPU, no
        // progress, and the host blocked in hipStreamSynchronize.
        //
        // Observed at NP=8 EP on GLM-5: 7 of 8 ranks reached
        // finalize_persistent_kernel while rank 5 held exactly one workgroup
        // (block 234) spinning in the dep wait at needed=128 actual=127.
        // This is task #47's ~40%-of-runs cold-start hang, and it is not
        // EP-specific -- EP only makes it reliable by coupling the ranks.
        //
        // The re-check is safe on every normal iteration boundary: terminate
        // is set exactly once, on the final END_OF_TASK_GRAPH, and the bump
        // that accompanies it is the only one that outruns a real iteration.
        // A worker at a boundary always has pc_iter greater than the last
        // completed iteration, so bailing here can never skip real work.
        if (__atomic_load_n(config.precomp_terminate, __ATOMIC_RELAXED)) {
          pc_terminated = 1;
        }
      }
      __syncthreads();
      if (pc_terminated) {
#ifdef MPK_ENABLE_DEVICE_TASK_ACCUM
        if (threadIdx.x == 0 && worker_id == 0) {
          unsigned long long ni = g_daccum_cnt[4] > 0 ? g_daccum_cnt[4] : 1;
          printf("[DACCUM] iters=%llu workers=%d\n", ni, config.num_workers);
          printf("[DACCUM] QKV        total_us=%llu  count=%llu  "
                 "per_iter_us=%.1f  per_iter_cnt=%.1f\n",
                 g_daccum_ns[0] / 1000,
                 g_daccum_cnt[0],
                 (double)g_daccum_ns[0] / 1000.0 / ni,
                 (double)g_daccum_cnt[0] / ni);
          printf("[DACCUM] DOWN_PROJ  total_us=%llu  count=%llu  "
                 "per_iter_us=%.1f  per_iter_cnt=%.1f\n",
                 g_daccum_ns[3] / 1000,
                 g_daccum_cnt[3],
                 (double)g_daccum_ns[3] / 1000.0 / ni,
                 (double)g_daccum_cnt[3] / ni);
          printf("[DACCUM] ATTN_SPLIT total_us=%llu  count=%llu  "
                 "per_iter_us=%.1f  per_iter_cnt=%.1f\n",
                 g_daccum_ns[6] / 1000,
                 g_daccum_cnt[6],
                 (double)g_daccum_ns[6] / 1000.0 / ni,
                 (double)g_daccum_cnt[6] / ni);
          printf("[DACCUM] OTHER      total_us=%llu  count=%llu  "
                 "per_iter_us=%.1f  per_iter_cnt=%.1f\n",
                 g_daccum_ns[8] / 1000,
                 g_daccum_cnt[8],
                 (double)g_daccum_ns[8] / 1000.0 / ni,
                 (double)g_daccum_cnt[8] / ni);
          printf("[DACCUM] MOE_W13    total_us=%llu  count=%llu  "
                 "per_iter_us=%.1f  per_iter_cnt=%.1f\n",
                 g_daccum_ns[9] / 1000,
                 g_daccum_cnt[9],
                 (double)g_daccum_ns[9] / 1000.0 / ni,
                 (double)g_daccum_cnt[9] / ni);
          printf("[DACCUM] SCHED_GAP  total_us=%llu  count=%llu  "
                 "per_iter_us=%.1f  per_iter_cnt=%.1f  avg_us=%.2f\n",
                 g_daccum_ns[15] / 1000,
                 g_daccum_cnt[15],
                 (double)g_daccum_ns[15] / 1000.0 / ni,
                 (double)g_daccum_cnt[15] / ni,
                 g_daccum_cnt[15] > 0
                     ? (double)g_daccum_ns[15] / 1000.0 / g_daccum_cnt[15]
                     : 0.0);
          printf("[DACCUM] DEP_WAIT   total_us=%llu  count=%llu  "
                 "per_iter_us=%.1f  per_iter_cnt=%.1f  avg_us=%.2f\n",
                 g_daccum_ns[16] / 1000,
                 g_daccum_cnt[16],
                 (double)g_daccum_ns[16] / 1000.0 / ni,
                 (double)g_daccum_cnt[16] / ni,
                 g_daccum_cnt[16] > 0
                     ? (double)g_daccum_ns[16] / 1000.0 / g_daccum_cnt[16]
                     : 0.0);
        }
#endif
#if MPK_BAR_SKEW
        if (threadIdx.x == 0 && worker_id == 0) {
          for (int s = 0; s < MPK_BAR_SKEW_SLOTS; s++) {
            if (g_barskew_cnt[s] == 0 && g_barskew_drop[s] == 0) {
              continue;
            }
            printf("BARSKEW %d cnt %llu ns %llu drop %llu gap %llu\n", s,
                   g_barskew_cnt[s], g_barskew_ns[s], g_barskew_drop[s],
                   g_barskew_gap[s]);
          }
          // Reduce the private per-worker rows. Serial in one thread, but it
          // runs once at teardown; the point of the rows is that the hot path
          // never touches a shared line.
          for (int s = 0; s < MPK_STAGE_SLOTS; s++) {
            unsigned long long sum = 0ull, cnt = 0ull, mn = ~0ull, mx = 0ull;
            for (int w = 0; w < MPK_STAGE_WORKERS; w++) {
              int const p = w * MPK_STAGE_SLOTS + s;
              if (g_stage_pcnt[p] == 0ull) {
                continue;
              }
              sum += g_stage_psum[p];
              cnt += g_stage_pcnt[p];
              if (g_stage_pmin[p] < mn) {
                mn = g_stage_pmin[p];
              }
              if (g_stage_pmax[p] > mx) {
                mx = g_stage_pmax[p];
              }
            }
            if (cnt == 0ull) {
              continue;
            }
            printf("BARSTAGE %d cnt %llu sum %llu min %llu max %llu\n", s, cnt,
                   sum, mn, mx);
          }
#if MPK_BAR_SKEW >= 3
          // Per-worker rows. The aggregate above gives the first-to-last
          // spread but not whether the SAME workers are late every layer.
          // Static imbalance is a schedule bug and is fixable; jitter is not,
          // and the two need different work. Only meaningful now that the
          // rows are private -- under the old contended atomics a worker's
          // row was other workers' queueing.
          // Slots 5..9 walk the MoE half in order (W_UV, pre-o_proj,
          // post-router, last W2 tile, entry arrival), so differencing a
          // single worker's row across them says WHICH phase built that
          // worker's lateness.
          //
          // 10..14 continue through the layer boundary (kernel entry, loop
          // bookkeeping done, loop top, pointer refresh issued, block
          // joined). They are here because the AGGREGATE means for those
          // slots are unusable: every slot's max is ~9.4 ms, so a handful of
          // outlier layers dominate the sum, and 12..14 are stamped only for
          // ml > 0 while 9..11 are stamped for every ml -- differencing two
          // slots with different populations is not a duration. Per-worker
          // rows fix both: same worker, same layer count, and
          // max-over-workers of a worker's own mean is the makespan proxy.
          for (int s = 0; s < MPK_STAGE_SLOTS - 1; s++) {
            for (int w = 0; w < MPK_STAGE_WORKERS; w++) {
              int const p = w * MPK_STAGE_SLOTS + s;
              if (g_stage_pcnt[p] == 0ull) {
                continue;
              }
              printf("BARSTAGEWS %d %d %llu %llu\n", s, w, g_stage_pcnt[p],
                     g_stage_psum[p]);
            }
          }
#endif
        }
#endif
#ifdef MPK_ENABLE_SUBPHASE_TIMING
        // Same dump as the TASK_TERMINATE path below. Under precomputed
        // dispatch a worker never fetches a TERMINATE task -- it breaks out of
        // the iter-ready poll on precomp_terminate and returns right here --
        // so without this copy the accumulators are silently dropped.
        if (threadIdx.x == 0 && worker_id == 0) {
          for (int s = 0; s < SUBPHASE_SLOTS; s++) {
            if (g_subphase_cnt[s] == 0) {
              continue;
            }
            printf("SP %d cnt %llu\n", s, g_subphase_cnt[s]);
            for (int p = 0; p < SUBPHASE_MAX_PHASES; p++) {
              if (g_subphase_ns[s][p] > 0) {
                printf("SP %d %d %llu\n", s, p, g_subphase_ns[s][p]);
              }
            }
          }
        }
#endif
        return;
      }

      int num_loaded_tasks = min(pc_my_len - pc_pos, TASK_DESCS_BUFFER_LENGTH);
      // Load task IDs: compute_task_id(iteration, position)
      if (threadIdx.x < num_loaded_tasks) {
        task_ids[threadIdx.x] =
            compute_task_id(pc_iter, pc_my_queue[pc_pos + threadIdx.x]);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        pc_pos += num_loaded_tasks;
      }
      // Load task descs via cp_async (identical to original)
      static_assert(sizeof(TaskDesc) % 16 == 0);
      constexpr int TASK_SIZE_PC = sizeof(TaskDesc) / 16;
      for (int i = threadIdx.x; i < num_loaded_tasks * TASK_SIZE_PC;
           i += blockDim.x) {
        int task_idx = i / TASK_SIZE_PC;
        int offset = i % TASK_SIZE_PC;
        load_smem(reinterpret_cast<char *>(task_descs) + i * 16,
                  reinterpret_cast<char *>(
                      config.all_tasks +
                      get_task_position_index(task_ids[task_idx])) +
                      offset * 16);
      }
      kernel::cp_async_fence();
      kernel::cp_async_wait<0>();
      __syncthreads();
      queue_pos = 0;
      queue_len = num_loaded_tasks;
#else
      // ===== Original scheduler-queue polling path =====
      int queue_idx = 0;
      if (threadIdx.x == 0) {
        int poll_count = 0;
#ifdef MPK_ENABLE_TIMING
        unsigned long long poll_start = clock64();
#endif
        while (next_task_pos[queue_idx] == last_task_pos[queue_idx]) {
          // Scheduler is on same XCD — use L2-local load instead of NT/HBM
          last_task_pos[queue_idx] =
              ld_local_u64(&config.worker_queue_last_ready_task_id
                                [worker_queue_ids[queue_idx]]);
          if (next_task_pos[queue_idx] < last_task_pos[queue_idx]) {
            break;
          } else {
            queue_idx =
                (queue_idx == num_worker_queues - 1) ? 0 : queue_idx + 1;
          }
          // nanosleep to avoid overwhelming I/O
          __nanosleep(10);
#ifdef MPK_ENABLE_TIMING
          total_poll_iters++;
#endif
        }
#ifdef MPK_ENABLE_TIMING
        total_poll_cycles += clock64() - poll_start;
#endif
        assert(next_task_pos[queue_idx] + config.per_worker_queue_len >
               last_task_pos[queue_idx]);
        // Compiler fence for ordering — scheduler is on same XCD, L2 coherent
        fence_local();
      }
      __syncthreads();
      int num_loaded_tasks =
          min((int)(last_task_pos[queue_idx] - next_task_pos[queue_idx]),
              TASK_DESCS_BUFFER_LENGTH);
      // Load task ids
      if (threadIdx.x < num_loaded_tasks) {
        // Scheduler is on same XCD — task data visible via L2
        task_ids[threadIdx.x] = ld_local_u64(
            &worker_queues[queue_idx][(next_task_pos[queue_idx] + threadIdx.x) %
                                      config.per_worker_queue_len]);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
#ifdef MPK_ENABLE_VERBOSE
        for (int i = 0; i < num_loaded_tasks; i++) {
          printf(
              "[%d][FTCH] worker_id(%d) queue_idx(%d) next_task_pos(%llu, "
              "%llu) last_task_pos(%llu, %llu) "
              "task_id(%llu) task_type(%d) event_id(%llx) \n",
              config.my_gpu_id,
              worker_id,
              queue_idx,
              next_task_pos[0],
              next_task_pos[1],
              last_task_pos[0],
              last_task_pos[1],
              get_task_position_index(task_ids[i]),
              config.all_tasks[get_task_position_index(task_ids[i])].task_type,
              config.all_tasks[get_task_position_index(task_ids[i])]
                  .trigger_event);
        }
#endif
        next_task_pos[queue_idx] += num_loaded_tasks;
      }
      // Load task descs
      static_assert(sizeof(TaskDesc) % 16 == 0);
      constexpr int TASK_SIZE = sizeof(TaskDesc) / 16; // 128b copy-async
      for (int i = threadIdx.x; i < num_loaded_tasks * TASK_SIZE;
           i += blockDim.x) {
        int task_idx = i / TASK_SIZE;
        int offset = i % TASK_SIZE;
        load_smem(reinterpret_cast<char *>(task_descs) + i * 16,
                  reinterpret_cast<char *>(
                      config.all_tasks +
                      get_task_position_index(task_ids[task_idx])) +
                      offset * 16);
      }
      cp_async_fence();
      cp_async_wait<0>();
      __syncthreads();
      queue_pos = 0;
      queue_len = num_loaded_tasks;
#endif // MPK_PRECOMPUTED_DISPATCH
    }
    TaskDesc *task_desc = task_descs + queue_pos;

#ifdef MPK_ENABLE_GAP_TIMING
    // Timestamp BEFORE dep check — gap = this minus prev_task_end (pure
    // scheduling)
    unsigned long long _before_dep_t0 = __builtin_amdgcn_s_memrealtime();
#endif

#ifdef MPK_ENABLE_PROFILING
    // FETCHED: stamp BEFORE dep-wait. The gap between this and the next
    // PROFILER_EVENT_START is dep-wait (queue+event-wait) time. The gap
    // between PROFILER_EVENT_START and PROFILER_EVENT_END is pure compute.
    if (task_desc->task_type != TASK_TERMINATE) {
      PROFILER_EVENT_FETCHED(task_desc->task_type, task_counter);
    }
#endif

    // Dependency check: The scheduler only dispatches tasks after their
    // dependency event has fired. The release-acquire chain guarantees:
    //   completing_worker RELEASE → scheduler ACQUIRE → scheduler RELEASE →
    //   this_worker ACQUIRE
    // So by the time we read the task from our queue, the dependency is
    // satisfied.
    if (threadIdx.x == 0) {
#ifdef MPK_PRECOMPUTED_DISPATCH
      // Debug: write current task position (state[0]) before dep check
      if (MPK_WS_ON(config)) {
        int *ws = config.precomp_dbg_worker_state + worker_id * 4;
        __atomic_store_n(&ws[0],
                         (int)get_task_position_index(task_ids[queue_pos]),
                         __ATOMIC_RELAXED);
      }
#endif
      if (task_desc->dependent_event != EVENT_INVALID_ID) {
        EventId event_id = task_desc->dependent_event;
        assert(get_event_gpu_id(event_id) == config.my_gpu_id);
        size_t event_index = get_event_position_index(event_id);
        EventCounter needed_counts =
            static_cast<EventCounter>(
                config.all_event_num_triggers[event_index]) *
            get_task_iteration_num(task_ids[queue_pos]);
        EventCounter actual_counts = 0;
#ifdef MPK_PRECOMPUTED_DISPATCH
        // Debug: write dep_event index (state[1])
        if (MPK_WS_ON(config)) {
          int *ws = config.precomp_dbg_worker_state + worker_id * 4;
          __atomic_store_n(&ws[1], (int)event_index, __ATOMIC_RELAXED);
        }
#endif
        if (is_nvshmem_event(event_id)) {
          // Cross-GPU dependency: a remote PE increments this symmetric event
          // counter via put+signal (SIGNAL_ADD). Wait through the backend's
          // signal primitive so the remote write is observed past the local
          // cache hierarchy. Under USE_ROCSHMEM this was previously compiled
          // out entirely, which let the consumer run before the remote slice
          // landed. GE (not EQ): the counter is a monotonic atomic-add, so a
          // fast peer can push it past the threshold before we sample it.
          mpk_shmem_signal_wait_ge(
              reinterpret_cast<uint64_t *>(
                  &config.all_event_counters[event_index]),
              needed_counts);
        } else {
#ifdef MPK_ENABLE_TIMING
          unsigned long long dep_start = clock64();
#endif
          // Single ACQUIRE load — dependency guaranteed satisfied by scheduler
          // dispatch chain.
          actual_counts =
              __atomic_load_n(reinterpret_cast<unsigned long long *>(
                                  &config.all_event_counters[event_index]),
                              __ATOMIC_RELAXED);
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
          // Agent-scope acquire fence (GPU-only, not system scope)
          __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
#else
          __atomic_thread_fence(__ATOMIC_ACQUIRE);
#endif
          if (actual_counts < needed_counts) {
            // Dependency not yet satisfied — poll until ready
#ifdef MPK_PRECOMPUTED_DISPATCH
            // Mark as spinning: ws[3] = -(needed - actual) to distinguish from
            // "not spinning"
            if (MPK_WS_ON(config)) {
              int *ws = config.precomp_dbg_worker_state + worker_id * 4;
              __atomic_store_n(&ws[3],
                               -((int)(needed_counts - actual_counts)),
                               __ATOMIC_RELAXED);
            }
#endif
#ifdef MPK_PRECOMPUTED_DISPATCH
            // Backstop for the shutdown race closed at the iteration gate
            // above. That fix keeps a worker from entering a phantom
            // iteration; this one gets a worker out if it is already parked
            // on an arrival that will never come. Without it the only exit
            // from this loop is the counter, so a single wedged workgroup
            // holds the whole rank -- and at NP=8 the peer wait then holds
            // every other rank with it.
            //
            // Polled every 1024 spins, not every spin: this is the barrier
            // wait, it is the hottest loop in the kernel, and precomp_terminate
            // is a system-scoped line that 240 workers would otherwise hammer.
            // A 1024-spin granularity is ~50us of extra shutdown latency once
            // per run, against zero steady-state cost.
            int _term_poll = 0;
#endif
            while (actual_counts < needed_counts) {
              actual_counts =
                  __atomic_load_n(reinterpret_cast<unsigned long long *>(
                                      &config.all_event_counters[event_index]),
                                  __ATOMIC_RELAXED);
#ifdef MPK_PRECOMPUTED_DISPATCH
              if ((++_term_poll & 1023) == 0 &&
                  __atomic_load_n(config.precomp_terminate, __ATOMIC_RELAXED)) {
                pc_terminated = 1;
                break;
              }
#endif
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
              __builtin_amdgcn_s_sleep(1);
#endif
#ifdef MPK_ENABLE_TIMING
              total_dep_wait_iters++;
#endif
            }
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
#else
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
#endif
          }
#ifdef MPK_ENABLE_TIMING
          total_dep_wait_cycles += clock64() - dep_start;
#endif
        }
      }
#ifdef MPK_PRECOMPUTED_DISPATCH
      // Phase 10 = dep check done, about to execute task
      if (MPK_WS_ON(config)) {
        int *ws = config.precomp_dbg_worker_state + worker_id * 4;
        __atomic_store_n(&ws[3], 10, __ATOMIC_RELAXED);
      }
#endif
    }
    __syncthreads();

#ifdef MPK_PRECOMPUTED_DISPATCH
    // The dep-wait backstop above bailed out of a wait that will never be
    // satisfied. Rides the __syncthreads() that is already here, so this
    // costs one shared-memory read per task and no extra barrier. The
    // accumulator dumps are skipped deliberately: this path only runs when
    // the run is already being torn down.
    if (pc_terminated) {
      return;
    }
#endif

#ifdef MPK_ENABLE_PROFILING
    if (task_desc->task_type != TASK_TERMINATE) {
      PROFILER_EVENT_START(task_desc->task_type, task_counter);
    }
#endif

#ifdef MPK_ENABLE_GANG_TASKS
    // Gang tasks: tiles executed per worker for event counting (default 1)
    __shared__ int gang_tiles_executed;
    if (threadIdx.x == 0) {
      gang_tiles_executed = 1;
    }
#endif
#ifdef MPK_FUSED_LAYER_BATCHING
    // When the fused-layer batching fast path handles event signaling
    // internally, it sets this flag to prevent the main loop from
    // double-signaling the last task's event.
    __shared__ int _flb_skip_signal;
    if (threadIdx.x == 0) {
      _flb_skip_signal = 0;
    }
#endif
#ifdef MPK_ENABLE_TIMING
    unsigned long long exec_start = clock64();
#endif
#ifdef MPK_ENABLE_DEVICE_TASK_ACCUM
    unsigned long long _accum_t0 = __builtin_amdgcn_s_memrealtime();
#endif
#ifdef MPK_ENABLE_GAP_TIMING
    // Scheduling gap = _before_dep_t0 - _prev_task_end (queue poll + task load,
    // NO dep wait) Dep wait = _accum_t0 - _before_dep_t0 (time spent waiting
    // for dependency)
    unsigned long long _sched_gap = 0;
    unsigned long long _dep_wait = 0;
    if (threadIdx.x == 0 && _prev_task_end != 0 && g_daccum_active) {
      _sched_gap = (_before_dep_t0 - _prev_task_end) * 10; // ns
      _dep_wait = (_accum_t0 - _before_dep_t0) * 10;       // ns
    }
#endif
    // Successfully fetched a new task
#ifdef MPK_K2944_DEBUG
    if (threadIdx.x == 0 && worker_id == 0 && pc_iter <= 2) {
      printf("[K2944] iter=%llu type=%d var=%d gang=%d ntiles=%d\n",
             (unsigned long long)pc_iter,
             (int)task_desc->task_type,
             (int)task_desc->variant_id,
             is_gang_task_type(task_desc->task_type) ? 1 : 0,
             is_gang_task_type(task_desc->task_type)
                 ? (int)task_desc->task_metadata.n_tile_count
                 : 0);
    }
#endif
    if (task_desc->task_type == TASK_TERMINATE) {
      // Terminate
#ifdef MPK_ENABLE_TIMING
      // Print timing stats before returning
      if (threadIdx.x == 0 && worker_id < 8) {
        printf("[TIMING] worker=%d tasks=%d poll_iters=%d dep_iters=%d "
               "poll_cycles=%llu dep_cycles=%llu exec_cycles=%llu "
               "signal_cycles=%llu\n",
               worker_id,
               total_tasks_executed,
               total_poll_iters,
               total_dep_wait_iters,
               total_poll_cycles,
               total_dep_wait_cycles,
               total_exec_cycles,
               total_signal_cycles);
        printf("[TASK_TIME] worker=%d linear=%llu/%d linear_res=%llu/%d "
               "attn=%llu/%d rms=%llu/%d silu=%llu/%d fused=%llu/%d\n",
               worker_id,
               linear_cycles,
               linear_count,
               linear_res_cycles,
               linear_res_count,
               attention_cycles,
               attention_count,
               rms_cycles,
               rms_count,
               silu_cycles,
               silu_count,
               fused_cycles,
               fused_count);
      }
#endif
#ifdef MPK_ENABLE_DEVICE_TASK_ACCUM
      if (threadIdx.x == 0 && worker_id == 0) {
        // g_daccum_decode_iters is incremented by ALL workers that see BEGIN
        // during decode, so divide by approx workers-per-BEGIN to get actual
        // iters. Use a simpler approach: just report raw totals. Derive iter
        // count from RMS_NORM count (1 per decode iter on worker 0)
        // g_daccum_decode_iters has a visibility bug on HIP, so derive from
        // data
        unsigned long long ni = g_daccum_cnt[4] > 0 ? g_daccum_cnt[4] : 1;
        int nw = config.num_workers;
        printf("[DACCUM] iters=%llu workers=%d\n", ni, nw);
        printf("[DACCUM] QKV        total_us=%llu  count=%llu  "
               "per_iter_us=%.1f  per_iter_cnt=%.1f\n",
               g_daccum_ns[0] / 1000,
               g_daccum_cnt[0],
               (double)g_daccum_ns[0] / 1000.0 / ni,
               (double)g_daccum_cnt[0] / ni);
        printf("[DACCUM] GATE_UP    total_us=%llu  count=%llu  "
               "per_iter_us=%.1f  per_iter_cnt=%.1f\n",
               g_daccum_ns[1] / 1000,
               g_daccum_cnt[1],
               (double)g_daccum_ns[1] / 1000.0 / ni,
               (double)g_daccum_cnt[1] / ni);
        printf("[DACCUM] O_PROJ     total_us=%llu  count=%llu  "
               "per_iter_us=%.1f  per_iter_cnt=%.1f\n",
               g_daccum_ns[2] / 1000,
               g_daccum_cnt[2],
               (double)g_daccum_ns[2] / 1000.0 / ni,
               (double)g_daccum_cnt[2] / ni);
        printf("[DACCUM] DOWN_PROJ  total_us=%llu  count=%llu  "
               "per_iter_us=%.1f  per_iter_cnt=%.1f\n",
               g_daccum_ns[3] / 1000,
               g_daccum_cnt[3],
               (double)g_daccum_ns[3] / 1000.0 / ni,
               (double)g_daccum_cnt[3] / ni);
        printf("[DACCUM] RMS_NORM   total_us=%llu  count=%llu  "
               "per_iter_us=%.1f  per_iter_cnt=%.1f\n",
               g_daccum_ns[4] / 1000,
               g_daccum_cnt[4],
               (double)g_daccum_ns[4] / 1000.0 / ni,
               (double)g_daccum_cnt[4] / ni);
        printf("[DACCUM] SILU_MUL   total_us=%llu  count=%llu  "
               "per_iter_us=%.1f  per_iter_cnt=%.1f\n",
               g_daccum_ns[5] / 1000,
               g_daccum_cnt[5],
               (double)g_daccum_ns[5] / 1000.0 / ni,
               (double)g_daccum_cnt[5] / ni);
        printf("[DACCUM] ATTN_SPLIT total_us=%llu  count=%llu  "
               "per_iter_us=%.1f  per_iter_cnt=%.1f\n",
               g_daccum_ns[6] / 1000,
               g_daccum_cnt[6],
               (double)g_daccum_ns[6] / 1000.0 / ni,
               (double)g_daccum_cnt[6] / ni);
        printf("[DACCUM] ATTN_MERGE total_us=%llu  count=%llu  "
               "per_iter_us=%.1f  per_iter_cnt=%.1f\n",
               g_daccum_ns[7] / 1000,
               g_daccum_cnt[7],
               (double)g_daccum_ns[7] / 1000.0 / ni,
               (double)g_daccum_cnt[7] / ni);
        printf("[DACCUM] OTHER      total_us=%llu  count=%llu  "
               "per_iter_us=%.1f  per_iter_cnt=%.1f\n",
               g_daccum_ns[8] / 1000,
               g_daccum_cnt[8],
               (double)g_daccum_ns[8] / 1000.0 / ni,
               (double)g_daccum_cnt[8] / ni);
        printf("[DACCUM] MOE_W13    total_us=%llu  count=%llu  "
               "per_iter_us=%.1f  per_iter_cnt=%.1f\n",
               g_daccum_ns[9] / 1000,
               g_daccum_cnt[9],
               (double)g_daccum_ns[9] / 1000.0 / ni,
               (double)g_daccum_cnt[9] / ni);
        printf("[DACCUM] MOE_W2     total_us=%llu  count=%llu  "
               "per_iter_us=%.1f  per_iter_cnt=%.1f\n",
               g_daccum_ns[10] / 1000,
               g_daccum_cnt[10],
               (double)g_daccum_ns[10] / 1000.0 / ni,
               (double)g_daccum_cnt[10] / ni);
        printf("[DACCUM] MOE_TOPK   total_us=%llu  count=%llu  "
               "per_iter_us=%.1f  per_iter_cnt=%.1f\n",
               g_daccum_ns[11] / 1000,
               g_daccum_cnt[11],
               (double)g_daccum_ns[11] / 1000.0 / ni,
               (double)g_daccum_cnt[11] / ni);
        printf("[DACCUM] MOE_SWIGLU total_us=%llu  count=%llu  "
               "per_iter_us=%.1f  per_iter_cnt=%.1f\n",
               g_daccum_ns[12] / 1000,
               g_daccum_cnt[12],
               (double)g_daccum_ns[12] / 1000.0 / ni,
               (double)g_daccum_cnt[12] / ni);
        printf("[DACCUM] MOE_MULSUM total_us=%llu  count=%llu  "
               "per_iter_us=%.1f  per_iter_cnt=%.1f\n",
               g_daccum_ns[13] / 1000,
               g_daccum_cnt[13],
               (double)g_daccum_ns[13] / 1000.0 / ni,
               (double)g_daccum_cnt[13] / ni);
        printf("[DACCUM] MOE_BIAS   total_us=%llu  count=%llu  "
               "per_iter_us=%.1f  per_iter_cnt=%.1f\n",
               g_daccum_ns[14] / 1000,
               g_daccum_cnt[14],
               (double)g_daccum_ns[14] / 1000.0 / ni,
               (double)g_daccum_cnt[14] / ni);
        printf("[DACCUM] SCHED_GAP  total_us=%llu  count=%llu  "
               "per_iter_us=%.1f  per_iter_cnt=%.1f  avg_us=%.2f\n",
               g_daccum_ns[15] / 1000,
               g_daccum_cnt[15],
               (double)g_daccum_ns[15] / 1000.0 / ni,
               (double)g_daccum_cnt[15] / ni,
               g_daccum_cnt[15] > 0
                   ? (double)g_daccum_ns[15] / 1000.0 / g_daccum_cnt[15]
                   : 0.0);
        printf("[DACCUM] DEP_WAIT   total_us=%llu  count=%llu  "
               "per_iter_us=%.1f  per_iter_cnt=%.1f  avg_us=%.2f\n",
               g_daccum_ns[16] / 1000,
               g_daccum_cnt[16],
               (double)g_daccum_ns[16] / 1000.0 / ni,
               (double)g_daccum_cnt[16] / ni,
               g_daccum_cnt[16] > 0
                   ? (double)g_daccum_ns[16] / 1000.0 / g_daccum_cnt[16]
                   : 0.0);
      }
#endif
#ifdef MPK_ENABLE_SUBPHASE_TIMING
      if (threadIdx.x == 0 && worker_id == 0) {
        // Print raw accumulator values - short lines to avoid GPU printf
        // interleaving Post-process with: grep "^SP " to extract values Format:
        // SP slot phase raw_ns count
        for (int s = 0; s < SUBPHASE_SLOTS; s++) {
          if (g_subphase_cnt[s] == 0) {
            continue;
          }
          printf("SP %d cnt %llu\n", s, g_subphase_cnt[s]);
          for (int p = 0; p < SUBPHASE_MAX_PHASES; p++) {
            if (g_subphase_ns[s][p] > 0) {
              printf("SP %d %d %llu\n", s, p, g_subphase_ns[s][p]);
            }
          }
        }
      }
#endif
#ifdef MPK_FUSED_PHASE_TIMING
      if (threadIdx.x == 0 && worker_id == 0) {
        unsigned long long cnt = g_fused_phase_ns[3];
        if (cnt > 0) {
          unsigned long long oproj = g_fused_phase_ns[0];
          unsigned long long poll = g_fused_phase_ns[1];
          unsigned long long moe = g_fused_phase_ns[2];
          // cnt = total calls across all workers across all iterations
          // Per-call averages in µs (values are in ns):
          printf(
              "[FUSED_PHASE] cnt=%llu oproj_us=%llu poll_us=%llu moe_us=%llu\n",
              cnt,
              oproj / 1000,
              poll / 1000,
              moe / 1000);
          // Per-iteration estimates (36 layers, 30 workers/XCD, 8 XCDs):
          int n_iters = (int)(cnt / (36 * 30 * 8));
          if (n_iters < 1) {
            n_iters = 1;
          }
          printf("[FUSED_PHASE] per_iter: oproj=%.1f poll=%.1f moe=%.1fus "
                 "(n_iters=%d)\n",
                 (double)oproj / 1000.0 / n_iters,
                 (double)poll / 1000.0 / n_iters,
                 (double)moe / 1000.0 / n_iters,
                 n_iters);
        }
      }
#endif
#ifdef MPK_EP_SKEW_PROBE
      if (threadIdx.x == 0 && worker_id == 0) {
        int n = g_ep_skew_n;
        int have = (n < EP_SKEW_HIST) ? n : EP_SKEW_HIST;
        if (have > 0) {
          // s_memrealtime is the 100 MHz constant clock: 1 tick = 10 ns.
          // Report the mean and the max over the retained layers; the max is
          // what matters, because a per-slice release only helps if the slice
          // is reliably early, not occasionally early.
          printf("[EPSKEW] layers=%d (ring holds %d) unit=us\n", n, have);
          for (int s = 0; s < EP_SKEW_SLICES; s++) {
            unsigned long long sum = 0, mx = 0;
            for (int i = 0; i < have; i++) {
              unsigned long long v = g_ep_skew_hist[i][s];
              sum += v;
              if (v > mx) {
                mx = v;
              }
            }
            printf("[EPSKEW] slice%d mean=%.3f max=%.3f\n",
                   s,
                   (double)sum / have / 100.0,
                   (double)mx / 100.0);
          }
          unsigned long long ssum = 0, smx = 0;
          for (int i = 0; i < have; i++) {
            unsigned long long v = g_ep_skew_span[i];
            ssum += v;
            if (v > smx) {
              smx = v;
            }
          }
          // span = gpu_last - earliest slice_last: the total W2 tail spread.
          // No per-slice scheme can claim more than this.
          printf("[EPSKEW] span mean=%.3f max=%.3f\n",
                 (double)ssum / have / 100.0,
                 (double)smx / 100.0);
        }
      }
#endif
#ifdef MPK_ENABLE_SPAN_TIMING
      if (threadIdx.x == 0 && worker_id == 0 && g_span_active) {
        int ne = g_span_event_count;
        int nl = 36;
        int ni = (ne > 0) ? ((ne + 1) / (nl * SPAN_STAGES)) : 1;
        if (ni < 1) {
          ni = 1;
        }
        static char const *stage_names[SPAN_STAGES] = {
            "QKV_KVUPD", "CK_FMHA", "OPROJ_TOPK", "MOE_FUSED"};
        printf("[SPAN] events=%d iters=%d layers=%d\n", ne, ni, nl);
        unsigned long long total_span = 0, total_compute = 0;
        for (int s = 0; s < SPAN_STAGES; s++) {
          double span_pl = (double)g_span_accum_us[s] / ni / nl;
          double comp_pl = (double)g_span_compute_us[s] / ni / nl;
          double gap_pl = span_pl - comp_pl;
          total_span += g_span_accum_us[s];
          total_compute += g_span_compute_us[s];
          printf("[SPAN] %s span=%.2f compute=%.2f gap=%.2f us/layer\n",
                 stage_names[s],
                 span_pl,
                 comp_pl,
                 gap_pl);
        }
        double ts = (double)total_span / ni / nl;
        double tc = (double)total_compute / ni / nl;
        printf("[SPAN] TOTAL span=%.2f compute=%.2f gap=%.2f us/layer\n",
               ts,
               tc,
               ts - tc);
        // O-Proj inner breakdown
        int oi = g_oproj_inner_iters;
        if (oi > 0) {
          static char const *op_names[OPROJ_INNER_CPS - 1] = {
              "Quant", "MFMA+Reduce", "Barrier", "RMSNorm+Router", "TopK"};
          printf("[OPROJ] iters=%d layers=%d\n", oi / nl, nl);
          double otot = 0.0;
          for (int s = 0; s < OPROJ_INNER_CPS - 1; s++) {
            // g_oproj_span_accum[s] is in raw ticks (100MHz clock = 10ns/tick)
            double us_pl = (double)g_oproj_span_accum[s] * 10.0 / 1000.0 / oi;
            otot += us_pl;
            printf("[OPROJ] %s %.2f us/layer\n", op_names[s], us_pl);
          }
          printf("[OPROJ] TOTAL %.2f us/layer\n", otot);
        }
      }
#endif
      return;
    } else if (task_desc->task_type == TASK_BEGIN_TASK_GRAPH) {
      // Track per-iteration wall-clock time on worker 0
      if (threadIdx.x == 0 && worker_id == 0) {
        unsigned long long cur_clk = get_wallclock_ns();
#ifndef MPK_ENABLE_PROFILING
        int num_active_tokens =
            config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];
        if (prev_begin_clk != 0 &&
            (fwd_pass_count < 10 || fwd_pass_count % 50 == 0)) {
          printf("[FWD_PASS] iter=%d time_ms=%.3f num_active_tokens=%d\n",
                 fwd_pass_count,
                 (double)(cur_clk - prev_begin_clk) / 1000000.0,
                 num_active_tokens);
        }
#endif
        prev_begin_clk = cur_clk;
        fwd_pass_count++;
      }
#ifdef MPK_ENABLE_DEVICE_TASK_ACCUM
      // Decode-only accumulation: activate on first decode iter (nat==1)
      if (threadIdx.x == 0) {
        int nat = config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];
        if (nat <= 1) {
#ifdef MPK_ENABLE_GAP_TIMING
          _prev_task_end = 0; // reset gap tracking at each BEGIN
#endif
          // Transition: atomicCAS ensures exactly one worker resets
          int old = atomicCAS(&g_daccum_active, 0, 1);
          if (old == 0) {
            for (int s = 0; s < DACCUM_SLOTS; s++) {
              g_daccum_ns[s] = 0;
              g_daccum_cnt[s] = 0;
            }
            g_daccum_decode_iters = 0;
            __threadfence();
          }
          // Count decode iters: only worker_id 0 increments
          if (worker_id == 0) {
            g_daccum_decode_iters++;
          }
        }
      }
#endif
#ifdef MPK_ENABLE_SUBPHASE_TIMING
      // Sub-phase timing: activate on first decode iter, same as DACCUM
      if (threadIdx.x == 0) {
        int nat = config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];
        if (nat <= 1) {
          int old = atomicCAS(&g_subphase_active, 0, 1);
          if (old == 0) {
            for (int s = 0; s < SUBPHASE_SLOTS; s++) {
              for (int p = 0; p < SUBPHASE_MAX_PHASES; p++) {
                g_subphase_ns[s][p] = 0;
              }
              g_subphase_cnt[s] = 0;
            }
            __threadfence();
          }
        }
      }
#endif
#ifdef MPK_ENABLE_SPAN_TIMING
      // Span timing: activate on first decode iter
      if (threadIdx.x == 0) {
        int nat = config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];
        if (nat <= 1) {
          int old = atomicCAS(&g_span_active, 0, 1);
          if (old == 0) {
            for (int s = 0; s < SPAN_STAGES; s++) {
              g_span_accum_us[s] = 0;
              g_span_compute_us[s] = 0;
              g_span_first_start[s] = 0;
              g_span_first_flag[s] = 0;
            }
            g_span_prev_event_ticks = 0;
            g_span_event_count = 0;
            // Init O-Proj inner breakdown
            for (int s = 0; s < OPROJ_INNER_CPS; s++) {
              g_oproj_cp_min[s] = 0xFFFFFFFFFFFFFFFFULL;
              g_oproj_cp_max[s] = 0;
            }
            for (int s = 0; s < OPROJ_INNER_CPS - 1; s++) {
              g_oproj_span_accum[s] = 0;
            }
            g_oproj_inner_count = 0;
            g_oproj_inner_iters = 0;
            g_oproj_inner_reset_flag = 0;
            __threadfence();
          }
        }
      }
#endif
#ifdef MPK_ENABLE_GANG_TASKS
    } else if (is_gang_task_type(task_desc->task_type)) {
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
      // Gang: compute XCD-local rank + workers_on_xcd lazily on first encounter
      if (threadIdx.x == 0 && block_xcd_local_rank == -1) {
#ifdef MPK_PRECOMPUTED_DISPATCH
        // Use template rank (matches dispatch assignment) to ensure all
        // dispatched workers get tiles. The worker_xcd_map scan gives arbitrary
        // ordering that may not match template rank, causing barrier deadlocks.
        block_xcd_local_rank = pc_xcd_rank;
        block_workers_on_xcd = config.precomp_workers_per_xcd;
#else
        // All worker_xcd_map entries are valid by now (scheduler waited for
        // ready_count)
        int rank = 0;
        int total_on_xcd = 0;
        for (int w = 0; w < config.num_workers; w++) {
          if (config.worker_xcd_map[w] == xcd_id) {
            if (w < worker_id) {
              rank++;
            }
            total_on_xcd++;
          }
        }
        block_xcd_local_rank = rank;
        block_workers_on_xcd = total_on_xcd;
#endif
      }
      __syncthreads();
#ifdef MPK_PRECOMPUTED_DISPATCH
      // Phase 11 = past gang __syncthreads, entering tile loop
      if (threadIdx.x == 0 && MPK_WS_ON(config)) {
        int *ws = config.precomp_dbg_worker_state + worker_id * 4;
        __atomic_store_n(&ws[3], 11, __ATOMIC_RELAXED);
      }

      // Gang barrier disabled — the fused kernel's internal hierarchical
      // barrier already handles cross-XCD synchronization. Workers that arrive
      // early simply wait at the internal barrier for stragglers.
#endif

      // Workers loop over multiple tiles when n_tile_count > workers_per_xcd
      {
#ifdef MPK_ENABLE_SPAN_TIMING
        // Record first-worker-start for this stage (one worker wins the CAS)
        if (threadIdx.x == 0 && g_span_active) {
          int _ss =
              _span_stage_for_task(task_desc->task_type, task_desc->variant_id);
          if (_ss >= 0 && atomicCAS(&g_span_first_flag[_ss], 0, 1) == 0) {
            g_span_first_start[_ss] = __builtin_amdgcn_s_memrealtime();
            __threadfence();
          }
        }
#endif

#if defined(MPK_PRECOMPUTED_DISPATCH) && defined(MPK_FUSED_LAYER_BATCHING)
        // ===== Fused-layer batching fast path =====
        if (task_desc->task_type == TASK_GANG_FULL_LAYER_FUSED_MI300 ||
            task_desc->task_type ==
                TASK_GANG_FULL_LAYER_WITH_LMHEAD_FUSED_MI300 ||
            task_desc->task_type == TASK_GANG_MLA_FULL_LAYER_FUSED_MI300) {

          if (config.ml_num_layers > 0) {
            // ===== Multi-layer all-fused: one task executes all layers =====
            int ml_n_tile_start = (int)task_desc->task_metadata.n_tile_start;
            int ml_n_tile_count = (int)task_desc->task_metadata.n_tile_count;

            // MPK_ML_PTR_PREFETCH: hold the next layer's pointer-table entries
            // in registers instead of loading them at the layer boundary.
            //
            // Measured: the boundary between a worker's last W2 tile and its
            // arrival at the next layer-entry barrier is 16.50 us/layer, and
            // stamp 12 puts 15.33 of that in this copy alone -- 1.20 ms of an
            // 11 ms wall for 47 pointer loads. It is that expensive because
            // it is two DEPENDENT cold round trips: the loads feed shared
            // memory, so the compiler must s_waitcnt before the LDS store,
            // and the layer streams ~65 MB through a 32 MB L2 in between, so
            // the 272-byte slice is evicted every single layer.
            //
            // MAX_INPUTS_PER_TASK is 34 and MAX_OUTPUTS_PER_TASK 13, both
            // below blockDim, so the whole table is one element per thread --
            // four VGPRs, issued before the tile loop and consumed after it.
            // Registers are free here (320 of 512, occupancy pinned by block
            // count, see the depth-8 scheduling result), and the tile loop is
            // ~130 us of cover for a ~1 us latency.
#if MPK_ML_PF
            void *pf_in = nullptr;
            void *pf_out = nullptr;
            int pf_variant = 0;
            bool const pf_in_ok = (int)threadIdx.x < MAX_INPUTS_PER_TASK;
            bool const pf_out_ok = (int)threadIdx.x < MAX_OUTPUTS_PER_TASK;
#endif

            for (int ml = 0; ml < config.ml_num_layers; ml++) {
              // Stage stamp 12: the loop boundary itself. Splits the 16.50 us
              // that S11 - S8 measures into the part below the previous
              // iteration's fence (S12 - S8: return from the fused kernel,
              // the worker-state store, threadfence_gpu, __syncthreads) and
              // the part above the dispatch (S11 - S12: the 34+13 pointer
              // table copy and its two __syncthreads).
              if (threadIdx.x == 0 && ml > 0) {
                mpk_stage_stamp(12);
              }
              // Layer 0: task_desc already loaded from precomputed dispatch
              // buffer with correct per-XCD pointers. Skip the copy.
              if (ml > 0) {
                // Widths must match the host-side ML_N_IN / ML_N_OUT that
                // built these tables (see the multi-layer scan), or the
                // strides disagree and every layer past 0 reads the wrong
                // slots. Both sides key off the TaskDesc capacity so the
                // LM-head variant's input_ptrs[24..27] / output_ptrs[12] get
                // refreshed too.
#if MPK_ML_PF
                // Registers, loaded a whole layer ago. No global traffic and
                // nothing to wait on -- just LDS stores and the barrier.
                if (pf_in_ok) {
                  task_desc->input_ptrs[threadIdx.x] = pf_in;
                }
                if (pf_out_ok) {
                  task_desc->output_ptrs[threadIdx.x] = pf_out;
                }
                if (threadIdx.x == 0) {
                  task_desc->variant_id = pf_variant;
                }
#else
                int ml_in_base =
                    (xcd_id * config.ml_num_layers + ml) * MAX_INPUTS_PER_TASK;
                int ml_out_base =
                    (xcd_id * config.ml_num_layers + ml) * MAX_OUTPUTS_PER_TASK;
                for (int i = threadIdx.x; i < MAX_INPUTS_PER_TASK;
                     i += blockDim.x) {
                  task_desc->input_ptrs[i] =
                      config.ml_input_table[ml_in_base + i];
                }
                for (int i = threadIdx.x; i < MAX_OUTPUTS_PER_TASK;
                     i += blockDim.x) {
                  task_desc->output_ptrs[i] =
                      config.ml_output_table[ml_out_base + i];
                }
                if (threadIdx.x == 0) {
                  task_desc->variant_id = config.ml_variant_ids[ml];
                }
#endif
                // Stage stamp 13: tid 0 has issued its share of the pointer
                // refresh and is about to join the block. S13 - S12 is the
                // ISSUE cost -- the global loads and the LDS stores on one
                // thread. Bisects the 1533 ticks that MPK_ML_BOUNDARY_PAD
                // priced at 1.46 ms of wall: if S13 - S12 is small and
                // S14 - S13 holds the region, the cost is tid 0 waiting for
                // the other 255 threads to arrive, which is intra-block skew
                // inherited from the previous layer and cannot be recovered
                // by deleting work here. If S13 - S12 holds it, the refresh
                // itself is serialized and is deletable.
                if (threadIdx.x == 0) {
                  mpk_stage_stamp(13);
                }
#if !MPK_ABL_ML_BOUNDARY
                __syncthreads();
#endif
                // Stage stamp 14: the block has joined. S14 - S13 is the
                // WAIT.
                if (threadIdx.x == 0) {
                  mpk_stage_stamp(14);
                }
              }
#ifdef MPK_NIL_TRIPWIRE
              // Breadcrumb: which layer this worker reached, and whether the
              // pointer set it is about to hand to the kernel contains a null.
              // Written to pinned host memory so it survives an abort.
              // Deliberately only plain stores to this worker's own slots: no
              // scan over the pointer set (the host-side table validation
              // already covers nulls there) and no global atomics on the hot
              // path. An earlier version scanned all 35 pointers per layer and
              // cost 4.5x -- enough to perturb timing and hide the very race
              // it was meant to catch, the same way rocgdb does.
              if (threadIdx.x == 0 && config.tripwire != nullptr) {
                unsigned long long *b = config.tripwire + MPK_TW_HDR +
                                        worker_id * MPK_TW_PER_WORKER;
                b[0] = (unsigned long long)ml;
                b[1] = 100; // phase: entering layer execution
                b[3] = (unsigned long long)task_desc->input_ptrs[0];
                // Decode iteration: distinguishes "faults on the very first
                // pass" from "faults after hundreds of clean ones". The host
                // only sees that it died within a second of launch, which at
                // ~2.5 ms/iter is anywhere in the first few hundred.
                b[6] = (unsigned long long)pc_iter;
              }
#endif

              // Publish the deterministic layer counter for this layer.
              //
              // The fused-layer task derives its three barrier release values
              // from this instead of snapshotting "current counter + 1". The
              // snapshot was only valid if the reader was ordered before this
              // layer's producer, which nothing guaranteed -- that is the race
              // f1fa720 worked around by making *every* worker arrive at the
              // Phase 2 barrier, at a cost of ~0.27ms/iter.
              //
              // pc_iter is the decode iteration (1-based) and ml the layer, so
              // this counts exactly what the per-XCD epoch counts: one bump per
              // layer, never reset, monotonic across the whole run. Every
              // worker computes the same value with no read of a shared
              // counter, so there is nothing left to race.
              //
              // _linear_reserved is the free int32 of the n_tile union member
              // this task already uses (n_tile_start / n_tile_count are the two
              // uint16s below it); it is declared in runtime_header.h and read
              // nowhere else in the tree.
              //
              // task_desc is a reused shared-memory slot, so this store must be
              // visible to the whole block before any worker enters the task.
              if (threadIdx.x == 0) {
                task_desc->task_metadata._linear_reserved =
                    (int32_t)((pc_iter - 1) * config.ml_num_layers + ml);
              }
              // NOT ablatable. Deleting this one hangs the run at layer 0:
              // task_desc is a shared-memory slot, every worker derives its
              // three barrier release values from _linear_reserved, and
              // without the join the other 255 threads read the previous
              // layer's epoch and wait on a release that never comes. This
              // is a sync-correctness barrier, not boundary bookkeeping.
              __syncthreads();

              // Boundary pricing probe. Sits exactly where the bookkeeping
              // does -- after the pointer refresh, before the dispatch -- and
              // is paid by all 256 threads, so it lengthens the region for
              // every worker uniformly rather than only for tid 0. Compiles
              // to nothing at the default MPK_ML_BOUNDARY_PAD=0.
              mpk_ml_boundary_pad();

              // Stage stamp 11: the multi-layer loop has finished its
              // per-layer bookkeeping (return path from the previous layer,
              // threadfence, input/output pointer-table refresh) and is about
              // to dispatch. See stamp 10 in gang_mla_full_layer_fused.
              if (threadIdx.x == 0) {
                mpk_stage_stamp(11);
              }

#if MPK_ML_PF
              // Issue the NEXT layer's pointer loads here, immediately in
              // front of the tile loop. The values are not touched again
              // until the top of the next iteration, on the far side of
              // ~130 us of layer work, so the round trip is fully covered.
              // Every s_waitcnt inside the tile loop drains vmcnt anyway,
              // which is the point: the load completes during the layer
              // rather than blocking the boundary.
              if (ml + 1 < config.ml_num_layers) {
                int const pf_in_base =
                    (xcd_id * config.ml_num_layers + ml + 1) *
                    MAX_INPUTS_PER_TASK;
                int const pf_out_base =
                    (xcd_id * config.ml_num_layers + ml + 1) *
                    MAX_OUTPUTS_PER_TASK;
                if (pf_in_ok) {
                  pf_in = config.ml_input_table[pf_in_base + threadIdx.x];
                }
                if (pf_out_ok) {
                  pf_out = config.ml_output_table[pf_out_base + threadIdx.x];
                }
                if (threadIdx.x == 0) {
                  pf_variant = config.ml_variant_ids[ml + 1];
                }
              }
#endif

              // Execute this layer
              int my_tiles = 0;
              // Publish the layer index into the worker-state phase slot.
              //
              // ws[3] was set to 11 once at gang entry and then never touched
              // again for the whole task. Since all 36 layers run inside this
              // one task, every worker reported phase=11 for the entire run
              // and a hang dump showed "240 workers at gang entry" no matter
              // which layer they were actually stuck in -- the counter could
              // not distinguish a worker on layer 0 from one on layer 35.
              // Encoding the layer makes the dump localize the stall.
              if (threadIdx.x == 0 && MPK_WS_ON(config)) {
                int *ws = config.precomp_dbg_worker_state + worker_id * 4;
                __atomic_store_n(&ws[3], 40000 + ml, __ATOMIC_RELAXED);
              }
              for (int t = block_xcd_local_rank; t < ml_n_tile_count;
                   t += block_workers_on_xcd) {
#ifdef MPK_NIL_TRIPWIRE
                if (threadIdx.x == 0 && config.tripwire != nullptr) {
                  unsigned long long *b = config.tripwire + MPK_TW_HDR +
                                          worker_id * MPK_TW_PER_WORKER;
                  b[2] = (unsigned long long)(ml_n_tile_start + t);
                  b[1] = 200; // phase: inside gang tile execution
                }
#endif
                _execute_gang_task(task_desc, config, ml_n_tile_start + t);
                my_tiles++;
              }
              if (threadIdx.x == 0) {
                gang_tiles_executed = my_tiles;
              }

              // Inter-layer sync: threadfence_gpu flushes L2 write buffer so
              // next layer's buffer_inv + QKV epoch barrier sees fresh data.
              // __syncthreads ensures all threads finish before task_desc is
              // modified for the next layer.
#if MPK_ABL_ML_BOUNDARY < 2
              if (ml < config.ml_num_layers - 1) {
                if (threadIdx.x == 0 && block_xcd_local_rank == 0) {
                  threadfence_gpu();
                }
                __syncthreads();
              }
#endif
            }

            // Deferred event signal: after compaction, layer 0's trigger_event
            // points to the last layer's event (gates FUSE_TAIL). Signal via
            // the two-level path since this task is in the queue.
            if (threadIdx.x == 0) {
              threadfence_gpu();
              EventId ev_id = config.ml_trigger_events[0];
              size_t ev_idx = get_event_position_index(ev_id);
              int xcd_slot = xcd_id * config.num_events + (int)ev_idx;
              int xcd_thresh = config.xcd_event_num_tasks != nullptr
                                   ? config.xcd_event_num_tasks[xcd_slot]
                                   : 0;
              if (xcd_thresh > 0) {
                TaskId ml_tid =
                    compute_task_id(pc_iter, config.ml_task_positions[0]);
                EventCounter local_cnt =
                    atom_add_local_u64(
                        reinterpret_cast<unsigned long long int *>(
                            &config.xcd_local_event_counters[xcd_slot]),
                        1) +
                    1;
                EventCounter needed_local =
                    static_cast<EventCounter>(xcd_thresh) *
                    get_task_iteration_num(ml_tid);
                if (local_cnt == needed_local) {
                  atom_add_release_gpu_u64(&config.all_event_counters[ev_idx],
                                           1);
                }
              } else {
                atom_add_release_gpu_u64(&config.all_event_counters[ev_idx], 1);
              }
            }

            if (threadIdx.x == 0) {
              _flb_skip_signal = 1;
            }

          } else {
            // ===== Original fast-loop: per-layer dispatch =====
            while (true) {
              int n_tile_start = (int)task_desc->task_metadata.n_tile_start;
              int n_tile_count = (int)task_desc->task_metadata.n_tile_count;
              int my_tiles = 0;

              // Rebase the host-stamped layer index onto the run-monotonic
              // counter the fused kernel's barriers require -- the same value
              // the replay loop computes, (iter-1)*num_layers + layer. The
              // barrier counters are never reset, so a per-iteration index
              // would satisfy every layer's release from iteration 1 onward.
              // task_desc points at a reused shared-memory slot, so the store
              // must be visible block-wide before any worker enters the task.
              if (threadIdx.x == 0 && config.ml_scanned_layers > 0) {
                int stamped_layer = task_desc->task_metadata._linear_reserved;
                task_desc->task_metadata._linear_reserved =
                    (int32_t)((pc_iter - 1) * config.ml_scanned_layers +
                              stamped_layer);
              }
              __syncthreads();

              for (int t = block_xcd_local_rank; t < n_tile_count;
                   t += block_workers_on_xcd) {
                _execute_gang_task(task_desc, config, n_tile_start + t);
                my_tiles++;
              }

              if (threadIdx.x == 0) {
                gang_tiles_executed = my_tiles;
                EventId ev_id = task_desc->trigger_event;
                size_t ev_idx = get_event_position_index(ev_id);
                int num_triggers = config.all_event_num_triggers[ev_idx];
                int xcd_slot = xcd_id * config.num_events + (int)ev_idx;
                int xcd_thresh = config.xcd_event_num_tasks != nullptr
                                     ? config.xcd_event_num_tasks[xcd_slot]
                                     : 0;
                if (xcd_thresh > 0) {
                  EventCounter local_cnt =
                      atom_add_local_u64(
                          reinterpret_cast<unsigned long long int *>(
                              &config.xcd_local_event_counters[xcd_slot]),
                          1) +
                      1;
                  EventCounter needed_local =
                      static_cast<EventCounter>(xcd_thresh) *
                      get_task_iteration_num(task_ids[queue_pos]);
                  if (local_cnt == needed_local) {
                    threadfence_gpu();
                    atom_add_release_gpu_u64(&config.all_event_counters[ev_idx],
                                             1);
                  }
                } else {
                  threadfence_gpu();
                  atom_add_release_gpu_u64(&config.all_event_counters[ev_idx],
                                           1);
                }
              }

              int next_qp = queue_pos + 1;
              if (next_qp >= queue_len) {
                int remaining = pc_my_len - pc_pos;
                if (remaining <= 0) {
                  break;
                }
                int num_to_load = min(remaining, TASK_DESCS_BUFFER_LENGTH);
                __syncthreads();
                if (threadIdx.x < num_to_load) {
                  task_ids[threadIdx.x] = compute_task_id(
                      pc_iter, pc_my_queue[pc_pos + threadIdx.x]);
                }
                __syncthreads();
                if (threadIdx.x == 0) {
                  pc_pos += num_to_load;
                }
                constexpr int TS = sizeof(TaskDesc) / 16;
                for (int i = threadIdx.x; i < num_to_load * TS;
                     i += blockDim.x) {
                  int ti = i / TS, off = i % TS;
                  load_smem(reinterpret_cast<char *>(task_descs) + i * 16,
                            reinterpret_cast<char *>(
                                config.all_tasks +
                                get_task_position_index(task_ids[ti])) +
                                off * 16);
                }
                kernel::cp_async_fence();
                kernel::cp_async_wait<0>();
                __syncthreads();
                queue_pos = 0;
                queue_len = num_to_load;
                next_qp = 0;
              }

              TaskDesc *next_td = task_descs + next_qp;
              if (next_td->task_type != TASK_GANG_FULL_LAYER_FUSED_MI300 &&
                  next_td->task_type !=
                      TASK_GANG_FULL_LAYER_WITH_LMHEAD_FUSED_MI300 &&
                  next_td->task_type !=
                      TASK_GANG_MLA_FULL_LAYER_FUSED_MI300) {
                break;
              }

              queue_pos = next_qp;
              task_desc = task_descs + queue_pos;
            }
            if (threadIdx.x == 0) {
              _flb_skip_signal = 1;
            }
          } // ml_num_layers > 0

        } else
#endif // MPK_FUSED_LAYER_BATCHING
        {
          // Normal (non-batched) gang task execution
          int n_tile_start = (int)task_desc->task_metadata.n_tile_start;
          int n_tile_count = (int)task_desc->task_metadata.n_tile_count;
          int my_tiles = 0;

          for (int t = block_xcd_local_rank; t < n_tile_count;
               t += block_workers_on_xcd) {
            int tile_idx = n_tile_start + t;
#ifdef MPK_PRECOMPUTED_DISPATCH
            // Phase 12 = about to execute gang tile
            if (threadIdx.x == 0 && MPK_WS_ON(config)) {
              int *ws = config.precomp_dbg_worker_state + worker_id * 4;
              __atomic_store_n(&ws[3], 1200 + tile_idx, __ATOMIC_RELAXED);
            }
#endif
            _execute_gang_task(task_desc, config, tile_idx);
            my_tiles++;
          }
#ifdef MPK_PRECOMPUTED_DISPATCH
          // Phase 13 = tile loop done
          if (threadIdx.x == 0 && MPK_WS_ON(config)) {
            int *ws = config.precomp_dbg_worker_state + worker_id * 4;
            __atomic_store_n(&ws[3], 13, __ATOMIC_RELAXED);
          }
#endif
          if (threadIdx.x == 0) {
            gang_tiles_executed = my_tiles;
          }
        }
      }

#else
      // Fallback for non-AMD: just execute as regular task
      _execute_task(task_desc, config);
#endif
#endif // MPK_ENABLE_GANG_TASKS
    } else {
#ifdef MPK_ENABLE_VERBOSE
      if (threadIdx.x == 0) {
        printf("[worker] _execute_task EXECUTE_TASK %d\n",
               task_desc->task_type);
      }
#endif
#ifdef MPK_TASK_DEBUG_PRINTF
      if (threadIdx.x == 0) {
        printf("[TASK_DBG] blk=%d type=%d var=%d req=%d in0=%p in1=%p in2=%p "
               "in3=%p in4=%p out0=%p out1=%p\n",
               (int)blockIdx.x,
               task_desc->task_type,
               task_desc->variant_id,
               (int)task_desc->task_metadata.request_id,
               task_desc->input_ptrs[0],
               task_desc->input_ptrs[1],
               task_desc->input_ptrs[2],
               task_desc->input_ptrs[3],
               task_desc->input_ptrs[4],
               task_desc->output_ptrs[0],
               task_desc->output_ptrs[1]);
      }
      __syncthreads();
#endif
#ifdef MPK_ENABLE_SPAN_TIMING
      // Record first-worker-start for non-gang tasks (e.g. CK_FMHA)
      if (threadIdx.x == 0 && g_span_active) {
        int _ss =
            _span_stage_for_task(task_desc->task_type, task_desc->variant_id);
        if (_ss >= 0 && atomicCAS(&g_span_first_flag[_ss], 0, 1) == 0) {
          g_span_first_start[_ss] = __builtin_amdgcn_s_memrealtime();
          __threadfence();
        }
      }
#endif
#ifdef MPK_NIL_TRIPWIRE
      // Breadcrumb for the non-gang path, so a fault outside the multi-layer
      // loop is still attributable to a worker and a task type.
      if (threadIdx.x == 0 && config.tripwire != nullptr) {
        unsigned long long *b =
            config.tripwire + MPK_TW_HDR + worker_id * MPK_TW_PER_WORKER;
        b[1] = 300; // phase: non-gang _execute_task
        b[2] = (unsigned long long)task_desc->task_type;
        b[3] = (unsigned long long)task_desc->input_ptrs[0];
      }
#endif
      _execute_task(task_desc, config);
    }
#ifdef MPK_PRECOMPUTED_DISPATCH
    // Phase 15 = task execution done, about to syncthreads
    if (threadIdx.x == 0 && MPK_WS_ON(config)) {
      int *ws = config.precomp_dbg_worker_state + worker_id * 4;
      __atomic_store_n(&ws[3], 15, __ATOMIC_RELAXED);
    }
#endif
    __syncthreads();

#ifdef MPK_K2944_DEBUG
    if (threadIdx.x == 0 && worker_id == 0 && pc_iter <= 2) {
      printf("[K2944_DONE] iter=%llu type=%d var=%d\n",
             (unsigned long long)pc_iter,
             (int)task_desc->task_type,
             (int)task_desc->variant_id);
    }
#endif

#ifdef MPK_ENABLE_DEVICE_TASK_ACCUM
    if (threadIdx.x == 0 && g_daccum_active &&
        task_desc->task_type != TASK_BEGIN_TASK_GRAPH) {
      unsigned long long _accum_dur =
          (__builtin_amdgcn_s_memrealtime() - _accum_t0) * 10; // ns
      int slot = 8;                                            // OTHER
      unsigned vid = task_desc->variant_id;
      switch (task_desc->task_type) {
        case TASK_LINEAR:
        case TASK_GANG_LINEAR_MI300:
        case TASK_GANG_LINEAR_BIAS_MI300:
        case TASK_GANG_RMSNORM_LINEAR_BIAS_MI300:
        case TASK_GANG_RMSNORM_LINEAR_MXFP4_BIAS_MI300:
        case TASK_GANG_RMSNORM_LINEAR_MXFP8_BIAS_MI300:
        case TASK_GANG_MULSUMRADD_RMSNORM_LINEAR_MXFP4_BIAS_MI300:
        case TASK_GANG_RMSNORM_LINEAR_MXFP4_BIAS_KVUPD_MI300:
        case TASK_GANG_MULSUMRADD_RMSNORM_LINEAR_MXFP4_BIAS_KVUPD_MI300:
        case TASK_GANG_RESADDF32_RMSNORM_LINEAR_MXFP4_BIAS_MI300:
        case TASK_GANG_RESADDF32_RMSNORM_LINEAR_MXFP4_BIAS_KVUPD_MI300:
        case TASK_GANG_RMSNORM_LINEAR_MXFP4_BIAS_ARGMAX_MI300:
          slot = (vid == 0) ? 0 : 1;
          break; // QKV vs gate_up
        case TASK_GANG_RMSNORM_LINEAR_BIAS_TOPK_MI300:
          slot = 11;
          break; // Router+TopK (fused RMSNorm+Linear+TopK)
        case TASK_LINEAR_WITH_RESIDUAL:
        case TASK_GANG_LINEAR_RES_MI300:
        case TASK_SPLITK_LINEAR_RES_ATOMIC_MI300:
        case TASK_GANG_SPLITK_LINEAR_RES_BIAS_MI300:
        case TASK_GANG_LINEAR_MXFP4_RES_BIAS_MI300:
          slot = (vid == 0) ? 2 : 3;
          break; // o_proj vs down_proj
        case TASK_GANG_LINEAR_MXFP4_RES_BIAS_RMSNORM_TOPK_MI300:
          slot = 3;
          break; // fused o_proj + topk
        case TASK_RMS_NORM:
          slot = 4;
          break;
        case TASK_SILU_MUL:
          slot = 5;
          break;
        case TASK_PAGED_ATTENTION_SPLIT_KV_MI300:
        case TASK_PAGED_ATTENTION_CK_FMHA_SPLIT_KV_MI300:
        case TASK_GANG_ATTN_SPLIT_KV_MI300:
        case TASK_GANG_MLA_DECODE_MI300:
        case TASK_GANG_MLA_FULL_LAYER_FUSED_MI300:
          slot = 6;
          break;
        case TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_MI300:
        case TASK_GANG_ATTN_MERGE_MI300:
          slot = 7;
          break;
        // MoE tasks
        case TASK_MOE_W13_LINEAR_MI300:
        case TASK_MOE_W13_LINEAR_MXFP4_MI300:
        case TASK_MOE_W13_LINEAR_MXFP4_CK_MI300:
        case TASK_GANG_MOE_W13_LINEAR_MI300:
        case TASK_GANG_MOE_W13_LINEAR_MXFP4_MI300:
        case TASK_GANG_MOE_W13_LINEAR_MXFP8_MI300:
        case TASK_GANG_MOE_W13_SWIGLU_MXFP4_MI300:
          slot = 9;
          break;
        case TASK_MOE_W2_LINEAR_MI300:
        case TASK_MOE_W2_LINEAR_MXFP4_MI300:
        case TASK_MOE_W2_LINEAR_MXFP4_CK_MI300:
        case TASK_GANG_MOE_W2_LINEAR_MI300:
        case TASK_GANG_MOE_W2_LINEAR_MXFP4_MI300:
        case TASK_GANG_MOE_W2_LINEAR_MXFP8_MI300:
          slot = 10;
          break;
        case TASK_GANG_MOE_FUSED_MXFP4_MI300:
          slot = 9;
          break; // Reuse W13 slot (fused replaces W13+SwiGLU+W2)
        case TASK_GANG_MOE_SWIGLU_W2_MXFP4_MI300:
          slot = 10;
          break; // W2 slot (fused SwiGLU+W2 replaces SwiGLU+W2)
        case TASK_MOE_TOPK_SOFTMAX_MI300:
        case TASK_MOE_TOPK_SIGMOID_BIAS_MI300:
          slot = 11;
          break;
        case TASK_SWIGLUOAI_MI300:
          slot = 12;
          break;
        case TASK_MOE_MUL_SUM_ADD_MI300:
          slot = 13;
          break;
        case TASK_BIAS_ADD_MI300:
          slot = 14;
          break;
        default:
          slot = 8;
          break;
      }
      atomicAdd(&g_daccum_ns[slot], _accum_dur);
      atomicAdd(&g_daccum_cnt[slot], 1ULL);

#ifdef MPK_ENABLE_GAP_TIMING
      // Accumulate scheduling gap (excl dep wait) and dep wait separately
      if (_sched_gap > 0) {
        atomicAdd(&g_daccum_ns[15], _sched_gap);
        atomicAdd(&g_daccum_cnt[15], 1ULL);
      }
      if (_dep_wait > 0) {
        atomicAdd(&g_daccum_ns[16], _dep_wait);
        atomicAdd(&g_daccum_cnt[16], 1ULL);
      }
      _prev_task_end = __builtin_amdgcn_s_memrealtime();
#endif
    }
#endif

#ifdef MPK_ENABLE_TIMING
    if (threadIdx.x == 0) {
      unsigned long long task_time = clock64() - exec_start;
      total_exec_cycles += task_time;
      total_tasks_executed++;
      // Track per-task-type timing
      switch (task_desc->task_type) {
        case TASK_LINEAR:
        case TASK_SPLITK_LINEAR_MI300:
        case TASK_SPLITK_REDUCE_MI300:
        case TASK_GANG_LINEAR_MI300:
        case TASK_GANG_LINEAR_BIAS_MI300:
        case TASK_GANG_RMSNORM_LINEAR_BIAS_MI300:
        case TASK_GANG_RMSNORM_LINEAR_BIAS_TOPK_MI300:
        case TASK_GANG_RMSNORM_LINEAR_MXFP4_BIAS_MI300:
        case TASK_GANG_RMSNORM_LINEAR_MXFP8_BIAS_MI300:
        case TASK_GANG_MULSUMRADD_RMSNORM_LINEAR_MXFP4_BIAS_MI300:
        case TASK_GANG_RMSNORM_LINEAR_MXFP4_BIAS_KVUPD_MI300:
        case TASK_GANG_MULSUMRADD_RMSNORM_LINEAR_MXFP4_BIAS_KVUPD_MI300:
        case TASK_GANG_RESADDF32_RMSNORM_LINEAR_MXFP4_BIAS_MI300:
        case TASK_GANG_RESADDF32_RMSNORM_LINEAR_MXFP4_BIAS_KVUPD_MI300:
        case TASK_GANG_RMSNORM_LINEAR_MXFP4_BIAS_ARGMAX_MI300:
        case TASK_GANG_LINEAR_RES_MI300:
        case TASK_GANG_LINEAR_SILU_MI300:
          linear_cycles += task_time;
          linear_count++;
          break;
        case TASK_LINEAR_WITH_RESIDUAL:
        case TASK_SPLITK_LINEAR_RES_ATOMIC_MI300:
        case TASK_GANG_SPLITK_LINEAR_RES_BIAS_MI300:
        case TASK_GANG_LINEAR_MXFP4_RES_BIAS_MI300:
          linear_res_cycles += task_time;
          linear_res_count++;
          break;
        case TASK_PAGED_ATTENTION_1:
        case TASK_PAGED_ATTENTION_2:
        case TASK_PAGED_ATTENTION_SPLIT_KV_MI300:
        case TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_MI300:
        case TASK_KV_CACHE_UPDATE_MI300:
        case TASK_MLA_KV_CACHE_UPDATE_MI300:
        case TASK_PAGED_ATTENTION_CK_FMHA_SPLIT_KV_MI300:
        case TASK_GANG_ATTN_SPLIT_KV_MI300:
        case TASK_GANG_ATTN_MERGE_MI300:
        case TASK_GANG_MLA_DECODE_MI300:
        case TASK_GANG_MLA_FULL_LAYER_FUSED_MI300:
          attention_cycles += task_time;
          attention_count++;
          break;
        case TASK_RMS_NORM:
          rms_cycles += task_time;
          rms_count++;
          break;
        case TASK_SILU_MUL:
          silu_cycles += task_time;
          silu_count++;
          break;
        case TASK_SILU_MUL_LINEAR_WITH_RESIDUAL:
          fused_cycles += task_time;
          fused_count++;
          break;
        default:
          break;
      }
    }
    unsigned long long signal_start = clock64();
#endif

#ifdef MPK_ENABLE_PROFILING
    if (task_desc->task_type != TASK_TERMINATE) {
      PROFILER_EVENT_END(task_desc->task_type, task_counter++);
    }
#endif

    // Trigger event
#ifdef MPK_FUSED_LAYER_BATCHING
    if (threadIdx.x == 0 && _flb_skip_signal) {
      // Fast path already signaled all events including this task's.
      // Reset and skip the normal event signal to avoid double-counting.
      _flb_skip_signal = 0;
    } else
#endif
        if (threadIdx.x == 0) {
#ifdef MPK_PRECOMPUTED_DISPATCH
      // Debug: increment per-worker tasks_done (state[2])
      // Phase 20 = about to signal event
      if (MPK_WS_ON(config)) {
        int *ws = config.precomp_dbg_worker_state + worker_id * 4;
        __atomic_store_n(&ws[2],
                         __atomic_load_n(&ws[2], __ATOMIC_RELAXED) + 1,
                         __ATOMIC_RELAXED);
        __atomic_store_n(&ws[3], 20, __ATOMIC_RELAXED);
      }
#endif
      EventId event_id = task_desc->trigger_event;
      size_t event_index = get_event_position_index(event_id);
      if (!is_nvshmem_event(event_id)) {
        size_t gpu_id = get_event_gpu_id(event_id);
        assert(gpu_id == config.my_gpu_id);
        // Case 1: Trigger a local non-nvshmem event
        int num_triggers = config.all_event_num_triggers[event_index];
        EventCounter count = 0;
        bool event_fired = false;

#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
        // Two-level event counting: XCD-local first, batch flush when
        // this XCD's share is done. Reduces buffer_wbl2 from ~24K to ~300.
        int xcd_slot = xcd_id * config.num_events + (int)event_index;
        int xcd_threshold = config.xcd_event_num_tasks != nullptr
                                ? config.xcd_event_num_tasks[xcd_slot]
                                : 0;
        if (xcd_threshold > 0) {
#ifdef MPK_ENABLE_GANG_TASKS
          // Gang tasks: each dispatched worker signals 1 to local counter.
          // Threshold = dispatch_count per XCD. Last worker flushes 1 to
          // global.
#ifdef MPK_PRECOMPUTED_DISPATCH
          // Pre-filled dispatch: threshold = dispatch_count, each worker
          // signals 1
          {
            int event_increment = 1;
#else
          // Scheduler dispatch: threshold = tiles_per_xcd, workers signal tile
          // count
          if (gang_tiles_executed == 0 &&
              is_gang_task_type(task_desc->task_type)) {
            // No work done — skip event signaling entirely
          } else {
            int event_increment = is_gang_task_type(task_desc->task_type)
                                      ? gang_tiles_executed
                                      : 1;
#endif
            EventCounter local_count =
                atom_add_local_u64(
                    reinterpret_cast<unsigned long long int *>(
                        &config.xcd_local_event_counters[xcd_slot]),
                    event_increment) +
                event_increment;
            EventCounter needed_local =
                static_cast<EventCounter>(xcd_threshold) *
                get_task_iteration_num(task_ids[queue_pos]);

            if (local_count == needed_local) {
              // Last worker/task on this XCD — flush + signal global
              threadfence_gpu();
              // Gang: flush 1 (one task done). Non-gang: flush xcd_threshold.
              int flush_amount =
                  is_gang_task_type(task_desc->task_type) ? 1 : xcd_threshold;
              count = atom_add_release_gpu_u64(
                  &config.all_event_counters[event_index], flush_amount);
              event_fired = (count + flush_amount) ==
                            static_cast<EventCounter>(num_triggers) *
                                get_task_iteration_num(task_ids[queue_pos]);
            }
          }
#else
          // Non-gang: each task increments by 1
          EventCounter local_count =
              atom_add_local_u64(
                  reinterpret_cast<unsigned long long int *>(
                      &config.xcd_local_event_counters[xcd_slot]),
                  1) +
              1;
          EventCounter needed_local =
              static_cast<EventCounter>(xcd_threshold) *
              get_task_iteration_num(task_ids[queue_pos]);

          if (local_count == needed_local) {
            threadfence_gpu();
            count = atom_add_release_gpu_u64(
                &config.all_event_counters[event_index], xcd_threshold);
            event_fired = (count + xcd_threshold) ==
                          static_cast<EventCounter>(num_triggers) *
                              get_task_iteration_num(task_ids[queue_pos]);
          }
#endif
          // else: not the last on this XCD — skip fence + global atomic
        } else
#endif
        {
          // Fallback: per-task fence + global atomic (NVIDIA, or uncounted
          // events)
          threadfence_gpu();
          count = atom_add_release_gpu_u64(
              &config.all_event_counters[event_index], 1);
          event_fired =
              (count + 1) == static_cast<EventCounter>(num_triggers) *
                                 get_task_iteration_num(task_ids[queue_pos]);
        }

#ifdef MPK_ENABLE_VERBOSE
        printf("[%d][DONE] worker_id(%d) iter_num(%llu) task_idx(%llu) "
               "event_id(%llu) "
               "event_type(local) count(%llu)\n",
               config.my_gpu_id,
               worker_id,
               get_task_iteration_num(task_ids[queue_pos]),
               get_task_position_index(task_ids[queue_pos]),
               event_id,
               count);
#endif

        if (event_fired) {
#ifdef MPK_ENABLE_SPAN_TIMING
          if (g_span_active) {
            unsigned long long _ev_ticks = __builtin_amdgcn_s_memrealtime();
            int _ev_stage = _span_stage_for_task(task_desc->task_type,
                                                 task_desc->variant_id);
            if (_ev_stage >= 0 && g_span_prev_event_ticks > 0) {
              unsigned long long delta_ticks =
                  _ev_ticks - g_span_prev_event_ticks;
              atomicAdd(&g_span_accum_us[_ev_stage], delta_ticks * 10 / 1000);
              atomicAdd((unsigned long long *)&g_span_event_count, 1ULL);
            }
            // Compute span = last_worker_end (now) - first_worker_start
            if (_ev_stage >= 0 && g_span_first_start[_ev_stage] > 0) {
              unsigned long long cs = _ev_ticks - g_span_first_start[_ev_stage];
              atomicAdd(&g_span_compute_us[_ev_stage], cs * 10 / 1000);
              // Reset flag for next layer
              g_span_first_flag[_ev_stage] = 0;
              g_span_first_start[_ev_stage] = 0;
            }
            if (_ev_stage >= 0) {
              g_span_prev_event_ticks = _ev_ticks;
              __threadfence();
            }
          }
#endif
#ifdef MPK_ENABLE_PROFILING
          PROFILER_EVENT_START(TASK_SCHD_EVENTS, task_counter);
#endif
          EventDesc event_desc = config.all_events[event_index];
          if (event_desc.event_type == EVENT_EMPTY) {
            // Do nothing for empty event
          }
#ifdef MPK_PRECOMPUTED_DISPATCH
          // In pre-filled mode, schedulers only need EVENT_END_OF_TASK_GRAPH.
          // Skip pushing LAUNCH_DEPENDENT_TASKS to scheduler queues entirely.
          else if (event_desc.event_type == EVENT_LAUNCH_DEPENDENT_TASKS ||
                   event_desc.event_type == EVENT_LAUNCH_MASSIVE_TASKS) {
            // Workers self-serve — no scheduler dispatch needed
          }
#endif
          else {
            bool use_bcast_queue = false;
            if (event_desc.event_type == EVENT_LAUNCH_MASSIVE_TASKS ||
                event_desc.event_type == EVENT_LAUNCH_DEPENDENT_TASKS) {
              use_bcast_queue = true;
            }
            int sched_id =
                use_bcast_queue
                    ? config.num_local_schedulers + config.num_remote_schedulers
                    : get_rand_sched_id(event_index,
                                        worker_id,
                                        config.num_workers,
                                        config.num_local_schedulers,
                                        xcd_id);
            if (use_bcast_queue) {
              // Cross-XCD broadcast queue — need global ops
              size_t last_event_pos = atom_add_release_gpu_u64(
                  &config.sched_queue_next_free_event_id[sched_id], 1);
              st_relaxed_gpu_u64(
                  &config.sched_queues[sched_id][last_event_pos %
                                                 config.per_sched_queue_len],
                  event_index);
              threadfence_gpu();
              size_t old;
              do {
                old = atom_cas_release_gpu_u64(
                    &config.sched_queue_last_ready_event_id[sched_id],
                    last_event_pos,
                    last_event_pos + 1);
              } while (old != last_event_pos);
            } else {
              // Same-XCD scheduler queue — use L2-local ops
              size_t last_event_pos = atom_add_local_u64(
                  &config.sched_queue_next_free_event_id[sched_id], 1);
              st_local_u64(
                  &config.sched_queues[sched_id][last_event_pos %
                                                 config.per_sched_queue_len],
                  event_index);
              fence_local();
              size_t old;
              do {
                old = atom_cas_local_u64(
                    &config.sched_queue_last_ready_event_id[sched_id],
                    last_event_pos,
                    last_event_pos + 1);
              } while (old != last_event_pos);
            }
          }
#ifdef MPK_ENABLE_PROFILING
          PROFILER_EVENT_END(TASK_SCHD_EVENTS, task_counter++);
#endif
        }
      } else {
        // Case 2: trigger a nvshmem event
        assert(task_desc->task_type == TASK_NVSHMEM_COPY);
        // Note that nvshmem copy task signal counter during data copy
        // we don't need to do anything here is the task type is NVSHMEM_COPY
#ifdef MPK_ENABLE_VERBOSE
        printf("[%d][DONE] worker_id(%d) task_id(%llu) event_id(%llx) "
               "event_type(remote)\n",
               config.my_gpu_id,
               worker_id,
               get_task_position_index(task_ids[queue_pos]),
               event_id);
#endif
      }
    }
#ifdef MPK_ENABLE_TIMING
    if (threadIdx.x == 0) {
      total_signal_cycles += clock64() - signal_start;
    }
#endif
#ifdef MPK_PRECOMPUTED_DISPATCH
    // Phase 30 = event signaling complete, looping back
    if (threadIdx.x == 0 && MPK_WS_ON(config)) {
      int *ws = config.precomp_dbg_worker_state + worker_id * 4;
      __atomic_store_n(&ws[3], 30, __ATOMIC_RELAXED);
    }
#endif
    queue_pos += 1;
  }
}

// Single-warp scheduler: lane 0 handles all dispatch for this XCD.
__device__ __forceinline__ void execute_scheduler(RuntimeConfig config,
                                                  int offset,
                                                  int assigned_sched_id = -1) {
  int const num_schedulers =
      config.num_local_schedulers + config.num_remote_schedulers;
  // CANNOT use syncthreads below — only thread 0 runs the scheduler
  if (threadIdx.x == 0) {
    int const sched_id =
        (assigned_sched_id >= 0) ? assigned_sched_id : (blockIdx.x + offset);
    int num_sched_queues = 1;
    size_t iteration_num = 0;
    unsigned long long prev_end_of_graph_clk = 0;
    int end_of_graph_count = 0;
    EventId *sched_queues[2];
    int sched_queue_ids[2];
    sched_queues[0] = config.sched_queues[sched_id];
    sched_queue_ids[0] = sched_id;
#ifdef MPK_WORKER_STATE
    // Scheduler residency + liveness, in the slots past the three per-worker
    // regions. Without this a stalled bootstrap is invisible: the schedulers
    // write nothing, so "no task was ever dispatched" and "no scheduler is
    // resident" read identically off the worker slots.
    int *const _ws_sched =
        (config.precomp_dbg_worker_state != nullptr && sched_id >= 0 &&
         sched_id < MPK_WS_SCHED_SLOTS)
            ? config.precomp_dbg_worker_state + config.num_workers * 12 +
                  sched_id * 4
            : nullptr;
    if (_ws_sched != nullptr) {
      __atomic_store_n(&_ws_sched[0], (int)blockIdx.x, __ATOMIC_RELAXED);
      __atomic_store_n(&_ws_sched[3], MPK_WS_SCHED_ENTERED, __ATOMIC_RELAXED);
    }
#endif

    // Build worker list for this scheduler (all workers on this XCD)
    int my_workers[MAX_WORKER_PER_SCHEDULER];
    int my_num_workers = 0;
    int my_xcd = 0; // Scheduler's hardware XCD ID (0 for NVIDIA)

    if (sched_id < config.num_local_schedulers) {
      // local schedulers also (collectively) process events from
      // the global queue
      sched_queues[num_sched_queues] = config.sched_queues[num_schedulers];
      sched_queue_ids[num_sched_queues] = num_schedulers;
      num_sched_queues++;

#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
      // Wait for all workers to report their XCD IDs
      if (config.worker_xcd_map != nullptr) {
        while (true) {
          int const _rdy = atomicAdd(config.worker_xcd_ready_count, 0);
#ifdef MPK_WORKER_STATE
          // The count itself, live. `239` here names the failure outright:
          // one worker block never became resident, and every scheduler is
          // therefore still in this loop rather than dispatching.
          if (_ws_sched != nullptr) {
            __atomic_store_n(&_ws_sched[1], _rdy, __ATOMIC_RELAXED);
          }
#endif
          if (_rdy >= config.num_workers) {
            break;
          }
          __nanosleep(100);
        }
        // Fence ensures we see all worker_xcd_map writes
        threadfence_gpu();
        // Read this scheduler's XCD and collect ALL matching workers
        my_xcd = get_current_xcd_id();
        for (int w = 0; w < config.num_workers; w++) {
          if (config.worker_xcd_map[w] == my_xcd) {
            my_workers[my_num_workers++] = w;
          }
        }
        // Device printf never reaches the log on a hang -- the buffer is
        // flushed at kernel exit and the kernel does not exit -- so the same
        // two facts also go to the pinned worker-state buffer.
#ifdef MPK_WORKER_STATE
        if (_ws_sched != nullptr) {
          __atomic_store_n(&_ws_sched[2], my_xcd * 1000 + my_num_workers,
                           __ATOMIC_RELAXED);
          __atomic_store_n(
              &_ws_sched[3], MPK_WS_SCHED_XCD_READY, __ATOMIC_RELAXED);
        }
#endif
        printf("[SCHED_XCD] sched_id=%d xcd=%d workers_on_xcd=%d block=%d\n",
               sched_id,
               my_xcd,
               my_num_workers,
               (int)blockIdx.x);
      } else
#endif
      {
        // Fallback: stride-based mapping (NVIDIA or if XCD map not available)
        for (int w = sched_id; w < config.num_workers;
             w += config.num_local_schedulers) {
          my_workers[my_num_workers++] = w;
        }
      }
    } else {
      // Remote schedulers: stride across remote worker IDs
      int remote_sched = sched_id - config.num_local_schedulers;
      for (int w = remote_sched; w < config.num_workers;
           w += config.num_remote_schedulers) {
        my_workers[my_num_workers++] = w + config.num_workers;
      }
    }
    size_t cur_event_pos[2], last_event_pos[2];
    for (int i = 0; i < 2; i++) {
      cur_event_pos[i] = 0;
      last_event_pos[i] = 0;
    }

    size_t worker_queue_next_free_task_pos[MAX_WORKER_PER_SCHEDULER];
    for (int i = 0; i < my_num_workers; i++) {
      worker_queue_next_free_task_pos[i] = 0;
    }

    int next_worker_idx = 0;
    int queue_idx = 0;
#ifdef MPK_ENABLE_DISPATCH_TIMING
    unsigned long long last_dep_dispatch_end_ns = 0;
#endif

    while (true) {
#ifdef MPK_WORKER_STATE
      // Liveness once the bootstrap wait has cleared: with [3] at
      // MPK_WS_SCHED_XCD_READY, [1] is this scheduler's event-loop iteration
      // count. Frozen across host dumps means the scheduler itself is stuck,
      // which no worker slot can show.
      if (_ws_sched != nullptr) {
        __atomic_store_n(&_ws_sched[1], (int)iteration_num, __ATOMIC_RELAXED);
      }
#endif
      int poll_count = 0;
      while (cur_event_pos[queue_idx] == last_event_pos[queue_idx]) {
        // queue_idx 0 = own scheduler queue (XCD-local), 1 = broadcast
        // (cross-XCD)
        if (queue_idx == 0) {
          last_event_pos[0] = ld_local_u64(
              &config.sched_queue_last_ready_event_id[sched_queue_ids[0]]);
        } else {
          last_event_pos[queue_idx] =
              ld_acquire_gpu_u64(&config.sched_queue_last_ready_event_id
                                      [sched_queue_ids[queue_idx]]);
        }

        if (cur_event_pos[queue_idx] < last_event_pos[queue_idx]) {
          break;
        } else {
          queue_idx = (queue_idx == num_sched_queues - 1) ? 0 : queue_idx + 1;
        }
        // nanosleep to avoid overwhelming I/O
        __nanosleep(10);
      }
      // Make sure the schedule queue is not overflow
      assert(cur_event_pos[queue_idx] + config.per_sched_queue_len >
             last_event_pos[queue_idx]);
      // Read event from queue — use local ops for own queue, global for
      // broadcast
      EventId event_id;
      if (queue_idx == 0) {
        fence_local();
        event_id = ld_local_u64(
            &sched_queues[0][cur_event_pos[0] % config.per_sched_queue_len]);
      } else {
        threadfence_gpu();
        event_id = ld_relaxed_gpu_u64(
            &sched_queues[queue_idx][cur_event_pos[queue_idx] %
                                     config.per_sched_queue_len]);
      }
      EventDesc e = config.all_events[event_id];
      if (is_termination_event(event_id, e)) {
        // terminate all workers (same XCD — use local ops)
        if (sched_id < config.num_local_schedulers) {
          for (int widx = 0; widx < my_num_workers; widx++) {
            int w = my_workers[widx];
            size_t last_task_id = worker_queue_next_free_task_pos[widx]++;
            st_local_u64(&config.worker_queues[w][last_task_id %
                                                  config.per_worker_queue_len],
                         0);
            fence_local();
            atom_add_local_u64(&config.worker_queue_last_ready_task_id[w], 1);
          }
        }
        return;
      }
      // This is the ending task of the current task graph
      if (e.event_type == EVENT_END_OF_TASK_GRAPH) {
#ifdef MPK_PRECOMPUTED_DISPATCH
        // Debug: if (sched_id == 0) printf("[PC_S0] EVENT_END_OF_TASK_GRAPH
        // iter=%llu\n", iteration_num);
#endif
#ifdef MPK_ENABLE_VERBOSE
        printf("[SCHD] END_OF_TASK_GRAPH\n");
#endif
        unsigned long long iter_end_clk = get_wallclock_ns();
        int num_active_tokens =
            config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];
        // Store per-iteration timing (deferred print at termination)
        if (prev_end_of_graph_clk != 0) {
          int idx = end_of_graph_count; // 1-based after first iter
          unsigned long long dur = iter_end_clk - prev_end_of_graph_clk;
          // Aggregates first: these must cover every iteration, including the
          // ones past the end of the ring.
          g_fwdpass_total_ns += dur;
          g_fwdpass_total_iters++;
#ifdef MPK_SPEC_DECODE
          if (g_spec_iter_is_decode) {
            g_spec_decode_ns += dur;
            g_spec_decode_iters++;
          }
#endif
          // Stride-decimate rather than truncate.
          //
          // The old code kept iterations 0..8191 and dropped every one after,
          // which is precisely backwards for this workload: per-iter latency
          // GROWS with sequence length, so the discarded tail is the only part
          // that shows the scaling. A 32k run reported the average of its
          // cheapest 8k iterations and looked flat no matter how badly the
          // tail degraded.
          //
          // Instead, once the ring fills, halve the resolution and compact:
          // keep every 2nd sample, then every 4th, and so on. The ring then
          // spans the whole run at uniform stride for any length, and the
          // recorded token count lets the host reconstruct latency-vs-seqlen.
          if (idx / g_fwdpass_stride < FWDPASS_LOG_MAX) {
            if (idx % g_fwdpass_stride == 0) {
              g_fwdpass_time_ns[idx / g_fwdpass_stride] = dur;
              g_fwdpass_tokens[idx / g_fwdpass_stride] = num_active_tokens;
              g_fwdpass_count = idx / g_fwdpass_stride + 1;
            }
          } else {
            // Ring full at this stride: compact in place, doubling the stride.
            for (int i = 0; i * 2 + 1 < FWDPASS_LOG_MAX; i++) {
              g_fwdpass_time_ns[i] = g_fwdpass_time_ns[i * 2];
              g_fwdpass_tokens[i] = g_fwdpass_tokens[i * 2];
            }
            g_fwdpass_stride *= 2;
            g_fwdpass_count = FWDPASS_LOG_MAX / 2;
            if (idx % g_fwdpass_stride == 0) {
              g_fwdpass_time_ns[idx / g_fwdpass_stride] = dur;
              g_fwdpass_tokens[idx / g_fwdpass_stride] = num_active_tokens;
              g_fwdpass_count = idx / g_fwdpass_stride + 1;
            }
          }
        }
        prev_end_of_graph_clk = iter_end_clk;
        end_of_graph_count++;
        // Check if we want to continue
#ifdef MODE_ONLINE_NOTOKEN
        if (!prepare_next_batch(config, iteration_num))
#else
        if (!prepare_next_batch(config))
#endif
        {
          unsigned long long prep_done_clk = get_wallclock_ns();
#ifndef MPK_ENABLE_PROFILING
          printf("[ITER_TIME] sched=%d iter=%llu prep_us=%.1f DONE\n",
                 sched_id,
                 iteration_num,
                 (double)(prep_done_clk - iter_end_clk) / 1000.0);
#endif
          // Dump deferred FWD_PASS log (after timing, before terminate).
          // iter is reconstructed from the decimation stride.
          for (int i = 1; i < g_fwdpass_count && i < FWDPASS_LOG_MAX; i++) {
            printf("[FWD_PASS] iter=%d time_ms=%.3f num_active_tokens=%d\n",
                   i * g_fwdpass_stride,
                   (double)g_fwdpass_time_ns[i] / 1000000.0,
                   g_fwdpass_tokens[i]);
          }
          // Untruncated summary. dropped>0 means the per-iter lines above are
          // only the first FWDPASS_LOG_MAX iterations and must not be averaged
          // as if they were the whole run -- use total_ms/iters instead.
          printf("[FWD_PASS_TOTAL] iters=%d total_ms=%.3f avg_ms=%.3f "
                 "dropped=%d\n",
                 g_fwdpass_total_iters,
                 (double)g_fwdpass_total_ns / 1000000.0,
                 g_fwdpass_total_iters > 0
                     ? (double)g_fwdpass_total_ns / 1000000.0 /
                           (double)g_fwdpass_total_iters
                     : 0.0,
                 g_fwdpass_dropped);
#ifdef MPK_SPEC_DECODE
          // ms_per_token is the number speculation is judged on. decode_ms
          // above it is per ITERATION and goes UP when speculation works.
          // Both are decode-only; prefill chunks never speculate and have a
          // different iteration shape.
          printf("[SPEC] proposed=%d accepted=%d accept_rate=%.3f "
                 "committed=%d decode_iters=%d tokens_per_iter=%.3f "
                 "decode_ms_per_iter=%.3f ms_per_token=%.3f\n",
                 g_spec_proposed,
                 g_spec_accepted,
                 g_spec_proposed > 0
                     ? (double)g_spec_accepted / (double)g_spec_proposed
                     : 0.0,
                 g_spec_committed,
                 g_spec_decode_iters,
                 g_spec_decode_iters > 0
                     ? (double)g_spec_committed / (double)g_spec_decode_iters
                     : 0.0,
                 g_spec_decode_iters > 0
                     ? (double)g_spec_decode_ns / 1000000.0 /
                           (double)g_spec_decode_iters
                     : 0.0,
                 g_spec_committed > 0
                     ? (double)g_spec_decode_ns / 1000000.0 /
                           (double)g_spec_committed
                     : 0.0);
#endif
#ifdef MPK_ENABLE_MOE_SUBPHASE
          // Raw timestamps: scratch[0]=entry, [1]=before_lds,
          // [4]=after_compute, [2]=after_barrier
          {
            long long prologue =
                (long long)(g_subphase_scratch[1] - g_subphase_scratch[0]) * 10;
            long long compute =
                (long long)(g_subphase_scratch[4] - g_subphase_scratch[1]) * 10;
            long long barrier =
                (long long)(g_subphase_scratch[2] - g_subphase_scratch[4]) * 10;
            long long total =
                (long long)(g_subphase_scratch[2] - g_subphase_scratch[0]) * 10;
            printf("[MOE_W13] prologue_ns=%lld compute_ns=%lld barrier_ns=%lld "
                   "total_ns=%lld\n",
                   prologue,
                   compute,
                   barrier,
                   total);
          }
#endif
#ifdef MPK_PRECOMPUTED_DISPATCH
          // Signal workers to terminate
          __threadfence();
          atomicExch(config.precomp_terminate, 1);
          __threadfence();
          atomicAdd(config.precomp_iter_ready, 1ULL); // unblock waiting workers
          // Fan out per-XCD release for terminate
          {
            unsigned long long new_val = *config.precomp_iter_ready;
            int *rel = config.precomp_iter_xcd_release;
            for (int x = 0; x < 8; x++) {
              st_wt_u32((void *)&rel[x * 16], (unsigned)new_val);
            }
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          }
#endif
          terminate_schedulers(config);
        } else {
          unsigned long long prep_done_clk = get_wallclock_ns();
#ifdef MPK_PRECOMPUTED_DISPATCH
          // Signal workers to start next iteration.
          // iter_ready starts at 0; every END_OF_TASK_GRAPH bumps it.
          // First bump (from prepare_kernel) enables iteration 1.
          __threadfence();
          atomicAdd(config.precomp_iter_ready, 1ULL);
          // Fan out per-XCD release flags via st_wt (write-through to HBM)
          // so workers can poll via ld_nt without cross-XCD L2 contention.
          {
            unsigned long long new_val = *config.precomp_iter_ready;
            int *rel = config.precomp_iter_xcd_release;
            for (int x = 0; x < 8; x++) {
              st_wt_u32((void *)&rel[x * 16], (unsigned)new_val);
            }
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          }
#else
          // Launch task 1 (begin_task_graph) for the next iteration (same XCD)
          int nw = my_workers[next_worker_idx];
          size_t last_task_id =
              worker_queue_next_free_task_pos[next_worker_idx]++;
          st_local_u64(
              &config.worker_queues[nw]
                                   [last_task_id % config.per_worker_queue_len],
              compute_task_id(iteration_num + 1, 1 /*begin_task_graph*/));
          fence_local();
          atom_add_local_u64(&config.worker_queue_last_ready_task_id[nw], 1);
#ifdef MPK_ENABLE_VERBOSE
          printf("[%d][SCHD]EVENT_END_OF_TASK_GRAPH schd_id(%d) "
                 "iter_num(%llu) task_idx(1) "
                 "worker_id(%d) "
                 "worker_last_ready_pos(%llu)\n",
                 config.my_gpu_id,
                 sched_id,
                 iteration_num + 1,
                 nw,
                 last_task_id + 1);
#endif
          next_worker_idx = (next_worker_idx + 1) % my_num_workers;
#endif // MPK_PRECOMPUTED_DISPATCH
        }
      } else if (e.event_type == EVENT_LAUNCH_DEPENDENT_TASKS) {
        iteration_num = iteration_num + 1;
#ifdef MPK_PRECOMPUTED_DISPATCH
        // Workers self-serve from pre-filled queues — skip all dispatch.
#else
        // assign event in a round-robin fashion
        // Split event across local schedulers (stride-based workers)
        assert(sched_id < config.num_local_schedulers);

#ifdef MPK_ENABLE_GANG_TASKS
        // Check if this event contains gang tasks
        bool is_gang_event = false;
        int gang_task_count = (int)(e.last_task_id - e.first_task_id);
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
        if (gang_task_count > 0 && gang_task_count == config.num_xcds &&
            is_gang_task_type(config.all_tasks[e.first_task_id].task_type)) {
          is_gang_event = true;
        }
#endif

        if (is_gang_event) {
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
          // Gang dispatch: each scheduler takes its XCD's task and broadcasts
          // to all workers on that XCD
          size_t my_gang_task = e.first_task_id + sched_id;
          assert(my_gang_task < e.last_task_id);

          int tiles_per_xcd =
              (int)config.all_tasks[my_gang_task].task_metadata.n_tile_count;
          int dispatch_count =
              (tiles_per_xcd < my_num_workers) ? tiles_per_xcd : my_num_workers;

          assert(dispatch_count > 0);
          assert(dispatch_count <= my_num_workers);

          // Pass 1 (first iteration): set xcd_event_num_tasks for two-level
          // counting
          if (iteration_num == 1 && config.xcd_event_num_tasks != nullptr) {
            int xcd_base = my_xcd * config.num_events;
            EventId trigger_ev = config.all_tasks[my_gang_task].trigger_event;
            int tev_idx = (int)get_event_position_index(trigger_ev);
            config.xcd_event_num_tasks[xcd_base + tev_idx] = tiles_per_xcd;
            fence_local();
          }

          // Pass 2: Broadcast the gang task to dispatch_count workers
          for (int widx = 0; widx < dispatch_count; widx++) {
            int nw = my_workers[widx];
            size_t last_task_id = worker_queue_next_free_task_pos[widx]++;
            st_local_u64(&config.worker_queues[nw][last_task_id %
                                                   config.per_worker_queue_len],
                         compute_task_id(iteration_num, my_gang_task));
            fence_local();
            atom_add_local_u64(&config.worker_queue_last_ready_task_id[nw], 1);
          }
#endif
        } else
#endif // MPK_ENABLE_GANG_TASKS
        {
          // Normal (non-gang) dependent task dispatch

#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
          // Pass 1: Count per-event tasks for two-level event counting
          // Only needed in first iteration (task graph is identical every
          // iteration)
          if (iteration_num == 1 && config.xcd_event_num_tasks != nullptr) {
            int xcd_base = my_xcd * config.num_events;
            size_t num_rounds =
                (e.last_task_id - e.first_task_id + config.num_workers - 1) /
                config.num_workers;
            for (size_t ri = 0; ri < num_rounds; ri++) {
              for (int widx = 0; widx < my_num_workers; widx++) {
                size_t j = my_workers[widx];
                size_t pos = e.first_task_id + ri * config.num_workers + j;
                if (pos < e.last_task_id) {
                  EventId trigger_ev = config.all_tasks[pos].trigger_event;
                  int tev_idx = (int)get_event_position_index(trigger_ev);
#ifdef MPK_ENABLE_GANG_TASKS
                  // Gang tasks: each worker fires per tile processed
                  if (is_gang_task_type(config.all_tasks[pos].task_type)) {
                    int gang_tiles =
                        (int)config.all_tasks[pos].task_metadata.n_tile_count;
                    config.xcd_event_num_tasks[xcd_base + tev_idx] +=
                        gang_tiles;
                  } else
#endif
                  {
                    config.xcd_event_num_tasks[xcd_base + tev_idx] += 1;
                  }
                }
              }
            }
            fence_local(); // Ensure thresholds visible to workers on same XCD
          }
#endif

          // Pass 2: Dispatch tasks to workers
          {
            size_t num_rounds =
                (e.last_task_id - e.first_task_id + config.num_workers - 1) /
                config.num_workers;

            for (size_t i = 0; i < num_rounds; i++) {
              for (int widx = 0; widx < my_num_workers; widx++) {
                size_t j = my_workers[widx];
                size_t position_index =
                    e.first_task_id + i * config.num_workers + j;
                if (position_index < e.last_task_id) {
#if defined(MPK_ENABLE_GANG_TASKS) &&                                          \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
                  // Gang task: broadcast to workers instead of 1-to-1
                  if (is_gang_task_type(
                          config.all_tasks[position_index].task_type)) {
                    int gang_tiles = (int)config.all_tasks[position_index]
                                         .task_metadata.n_tile_count;
                    int gang_dispatch = (gang_tiles < my_num_workers)
                                            ? gang_tiles
                                            : my_num_workers;
                    for (int gw = 0; gw < gang_dispatch; gw++) {
                      int gnw = my_workers[gw];
                      size_t gslot = worker_queue_next_free_task_pos[gw]++;
                      st_local_u64(
                          &config
                               .worker_queues[gnw][gslot %
                                                   config.per_worker_queue_len],
                          compute_task_id(iteration_num, position_index));
                      fence_local();
                      atom_add_local_u64(
                          &config.worker_queue_last_ready_task_id[gnw], 1);
                    }
                  } else
#endif
                  {
                    int nw = my_workers[next_worker_idx];
                    size_t last_task_id =
                        worker_queue_next_free_task_pos[next_worker_idx]++;
                    st_local_u64(
                        &config.worker_queues[nw][last_task_id %
                                                  config.per_worker_queue_len],
                        compute_task_id(iteration_num, position_index));
                    fence_local();
                    atom_add_local_u64(
                        &config.worker_queue_last_ready_task_id[nw], 1);
#ifdef MPK_TRACE_MOE_DISPATCH
                    // Trace MoE task placement (first iteration only)
                    if (iteration_num == 1) {
                      int tt = config.all_tasks[position_index].task_type;
                      if (tt == TASK_MOE_W13_LINEAR_MI300) {
                        int eo = config.all_tasks[position_index]
                                     .task_metadata.expert_offset;
                        printf("[MOE_DISPATCH] sched=%d xcd=%d worker=%d "
                               "pos=%d type=W13 expert_offset=%d\n",
                               sched_id,
                               (int)my_xcd,
                               (int)nw,
                               (int)position_index,
                               eo);
                      }
                    }
#endif
                    next_worker_idx = (next_worker_idx + 1) % my_num_workers;
                  }
                }
              }
            }
          }
        } // end else (non-gang)
#endif // MPK_PRECOMPUTED_DISPATCH (skip dispatch)
      } else {
        TaskId my_first_task = e.first_task_id, my_last_task = e.last_task_id;
        if (e.event_type == EVENT_LAUNCH_MASSIVE_TASKS) {
          // Split event across local schedulers (by sched_id)
          assert(sched_id < config.num_local_schedulers);
          get_first_last_ids(e.last_task_id - e.first_task_id,
                             config.num_local_schedulers,
                             sched_id,
                             &my_first_task,
                             &my_last_task);
          my_first_task += e.first_task_id;
          my_last_task += e.first_task_id;
        }

#if defined(MPK_ENABLE_GANG_TASKS) &&                                          \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
        // Check if this is a gang event (MASSIVE or LAUNCH_TASKS with gang task
        // type)
        bool is_gang_in_else = false;
        if (my_first_task < my_last_task &&
            is_gang_task_type(config.all_tasks[my_first_task].task_type)) {
          is_gang_in_else = true;
        }

        if (is_gang_in_else) {
          // Gang dispatch: each scheduler has 1 gang task, broadcast to workers
          assert(my_last_task - my_first_task == 1 &&
                 "Expected exactly 1 gang task per scheduler");
          size_t my_gang_task = my_first_task;
          int tiles_per_xcd =
              (int)config.all_tasks[my_gang_task].task_metadata.n_tile_count;
          int dispatch_count =
              (tiles_per_xcd < my_num_workers) ? tiles_per_xcd : my_num_workers;

          // Pass 1: xcd_event_num_tasks
          if (iteration_num == 1 && config.xcd_event_num_tasks != nullptr) {
            int xcd_base = my_xcd * config.num_events;
            EventId trigger_ev = config.all_tasks[my_gang_task].trigger_event;
            int tev_idx = (int)get_event_position_index(trigger_ev);
            config.xcd_event_num_tasks[xcd_base + tev_idx] = tiles_per_xcd;
            fence_local();
          }

          // Pass 2: broadcast to dispatch_count workers
          for (int widx = 0; widx < dispatch_count; widx++) {
            int nw = my_workers[widx];
            size_t last_task_id = worker_queue_next_free_task_pos[widx]++;
            st_local_u64(&config.worker_queues[nw][last_task_id %
                                                   config.per_worker_queue_len],
                         compute_task_id(iteration_num, my_gang_task));
            fence_local();
            atom_add_local_u64(&config.worker_queue_last_ready_task_id[nw], 1);
          }
        } else
#endif
        {
          // Normal (non-gang) dispatch
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
          // Count per-event tasks for two-level event counting (first
          // iteration)
          if (iteration_num == 1 && config.xcd_event_num_tasks != nullptr) {
            int xcd_base = my_xcd * config.num_events;
            for (size_t ti = my_first_task; ti < my_last_task; ti++) {
              EventId trigger_ev = config.all_tasks[ti].trigger_event;
              int tev_idx = (int)get_event_position_index(trigger_ev);
              config.xcd_event_num_tasks[xcd_base + tev_idx] += 1;
            }
            fence_local();
          }
#endif

          for (size_t i = my_first_task; i < my_last_task; i++) {
            int nw = my_workers[next_worker_idx];
            size_t last_task_id =
                worker_queue_next_free_task_pos[next_worker_idx]++;
            st_local_u64(&config.worker_queues[nw][last_task_id %
                                                   config.per_worker_queue_len],
                         compute_task_id(iteration_num, i));
            fence_local();
            atom_add_local_u64(&config.worker_queue_last_ready_task_id[nw], 1);

#ifdef MPK_ENABLE_VERBOSE
            printf("[%d][SCHD] EXECUTE_TASK schd_id(%d) iter_num(%llu) "
                   "task_idx(%llu) "
                   "worker_id(%d) "
                   "worker_last_ready_pos(%llu)\n",
                   config.my_gpu_id,
                   sched_id,
                   iteration_num,
                   i,
                   nw,
                   last_task_id + 1);
#endif

            next_worker_idx = (next_worker_idx + 1) % my_num_workers;
          }
        } // end non-gang dispatch in else branch
      }
      cur_event_pos[queue_idx] += 1;
    }
  }
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
// Budgeted on the same switch as worker_kernel below, and it MUST be: LLVM's
// AMDGPUAttributor propagates amdgpu-waves-per-eu from kernels down the call
// graph, but it MERGES over all callers, so the weakest caller wins. This
// kernel reaches the same phase functions, so leaving it at 1 kept every
// device function unconstrained no matter what worker_kernel asked for --
// that is why the first attempt (worker_kernel alone) read 284 at wpe 1, 2
// and 3 alike and looked like the attribute did nothing.
// MPK_WORKER_WAVES_PER_EU is documented next to worker_kernel; predeclare its
// default here so both launch bounds see the same value.
#ifndef MPK_WORKER_WAVES_PER_EU
#define MPK_WORKER_WAVES_PER_EU 1
#endif
#if MPK_WORKER_WAVES_PER_EU >= 2
#define MPK_WORKER_ATTR                                                        \
  __attribute__((amdgpu_waves_per_eu(MPK_WORKER_WAVES_PER_EU)))
#else
#define MPK_WORKER_ATTR
#endif
__global__ MPK_WORKER_ATTR __launch_bounds__(
    WORKER_NUM_THREADS,
    MPK_WORKER_WAVES_PER_EU) void persistent_kernel(RuntimeConfig config) {
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
  // Dynamic role election: each block discovers its XCD, then
  // the first block on each XCD becomes the scheduler for that XCD.
  // All other blocks become workers. This guarantees exactly 1
  // scheduler per XCD regardless of hardware block placement.
  __shared__ int my_role; // 0 = worker, 1 = scheduler
  __shared__ int my_id;   // worker_id or sched_id

  if (threadIdx.x == 0) {
    int my_xcd = get_current_xcd_id();
    // Try to claim this XCD as scheduler
    int old =
        atomicCAS(&config.xcd_scheduler_claimed[my_xcd], -1, (int)blockIdx.x);
    if (old == -1) {
      // Successfully claimed — become scheduler for this XCD
      my_role = 1;
      my_id = my_xcd; // sched_id = XCD ID
    } else {
      // XCD already has a scheduler — become a worker
      my_role = 0;
      my_id = atomicAdd(config.dynamic_worker_id_counter, 1);
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    int my_xcd = get_current_xcd_id();
    if (my_role == 1) {
      printf("[ELECTION] block=%d xcd=%d -> SCHEDULER (sched_id=%d)\n",
             (int)blockIdx.x,
             my_xcd,
             my_id);
    }
  }
  __syncthreads();

  if (my_role == 0) {
    execute_worker(config, my_id);
  } else {
    execute_scheduler(config, 0, my_id);
  }
#else
  // NVIDIA: static role assignment (no XCD concept)
  persistent_checker(config);
  if (blockIdx.x < config.num_workers) {
    execute_worker(config);
  } else {
    execute_scheduler(config, -config.num_workers);
  }
#endif
}

// MPK_WORKER_WAVES_PER_EU: the register budget for the WHOLE megakernel.
//
// In HIP the second __launch_bounds__ argument is NOT CUDA's
// minBlocksPerMultiprocessor -- it is MIN_WARPS_PER_EXECUTION_UNIT, waves per
// SIMD, and it is the direct control on how many VGPRs the allocator may
// spend. gfx950's unified VGPR file is 512/SIMD with an allocation granule of
// 8, so 284 granulates to 288 and 512/288 = 1 wave/SIMD: a second block per CU
// is physically unplaceable. 256 granulates to 256 and gives 2.
//
// TWO THINGS HAD TO BE TRUE FOR THIS KNOB TO DO ANYTHING. Both were found the
// hard way; do not simplify either away.
//
// (1) It must be set on BOTH kernels. See the note at persistent_kernel.
//
// (2) The reported .vgpr_count is not any one function's pressure. The
//     megakernel is built with -fgpu-rdc and every phase is __device__
//     __noinline__ (125 s_swappc_b64 in the image), so the number is a
//     CALL-GRAPH quantity -- and specifically it is
//         max(arch VGPR over the call graph) + max(AGPR over the call graph),
//     the two maxima taken INDEPENDENTLY and then summed. Measured three ways:
//       wpe=1  248 arch (four functions tie) + 36 acc (mla_decode_absorbed
//              alone)                                                 = 284
//       wpe=2  220 arch (persistent_kernel's own body, 0 acc) + 128 acc
//              (gang_mla_full_layer_fused, which the per-function budget lets
//              split 128/128)                                         = 348
//       wpe=3  168 arch + 84 acc                                      = 252
//     wpe=2 is WORSE than no budget: every function individually honours its
//     256 and then the independent-maxima sum blows past it. The knob is not
//     monotonic. Read the number, do not reason about it.
//
// Because of (2), the fix was never "shrink mla_decode_absorbed". Its live set
// is real -- arch peaks at 248 in the same window where 24-36 AGPRs are live,
// and its 240 baseline scratch ops are the ABI callee-save prologue/epilogue
// (v40-v143, 60 dwords in at instr 1-60, back out at 880-939), not pressure.
// The allocator can fit it in 236 when asked; the module-level sum was the
// gate, and the sum is what this knob moves.
//
// Cost at wpe=3, all of it compiler-side and not yet priced on the wall:
// worker_kernel spills 67 VGPRs (0 before) and private_segment_fixed_size goes
// 596 -> 1192 B/thread. LDS is 7296 B and never binding (2 x 7296 = 14.6 KB
// against 160 KB/CU).
//
// This is only HALF of 2 blocks/CU. The other half is having >= ~464
// co-resident workers to place; at MPK_NUM_WORKERS=232 there is 1 block per CU
// regardless, so this knob alone should cost the spill and buy nothing.
//
// ---------------------------------------------------------------------------
// CORRECTION. Everything above is a description of the BUILD, and the build is
// not the run. wpe=3 DOES NOT RUN. Measured at NP=8, MPK_NUM_WORKERS=232, LDS
// left at its default: all eight ranks abort immediately after
// "launch_persistent_kernel ENTER", before a single decode iteration, with
//   HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION ... code: 0x29
// The commit that introduced this knob called the register gate "OPEN" on the
// strength of .vgpr_count alone and never ran it. It is not open.
//
// Ruled out so far, so nobody re-runs these:
//  * A stack undercount from -fgpu-rdc's 125 ABI calls. Rebuilding with
//    -mllvm --amdgpu-assume-external-call-stack-size=16384 leaves
//    private_segment_fixed_size at exactly 1192 -- LLVM resolved the call
//    graph itself and is not falling back to an assumed size.
//  * A metadata inconsistency across the four entry points. prepare_kernel
//    (53 vgpr / 64 B) and scheduler_kernel (71 / 512 B) are byte-identical
//    between wpe=1 and wpe=3; only worker_kernel (284/596/0 spill ->
//    252/1192/67) and persistent_kernel (284/932/0 -> 252/1576/74) move.
//  * The LDS request. wpe=1 with the LDS cut below runs clean and correct, and
//    wpe=3 faults with LDS at its default 155 KB, so the two are independent.
//
//  * The scratch backing store. Raising occupancy doubles what ROCr reserves
//    (1192 B/thread x 256 CU x 4 SIMD x 2 waves x 64 lanes = 149 MB), but
//    HSA_SCRATCH_SINGLE_LIMIT=2G, HSA_NO_SCRATCH_THREAD_LIMITER=1,
//    HSA_NO_SCRATCH_RECLAIM=1 and HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0 all leave
//    it at exactly 7 aperture violations.
//  * A general fragility to the attribute. wpe=2 RUNS CLEAN -- 0 aperture
//    violations, correct output, decode min 10.911 ms (slower, and expected to
//    be: 348 unified VGPRs is worse than the 284 default and buys no
//    occupancy). So only the wpe=3 allocation is broken, not any perturbation.
//  * Inline asm naming a physical VGPR above wpe=3's arch cap. At wpe=3 every
//    phase function is capped at v0-v84 where wpe=2 allows v0-v128, so an asm
//    naming a register in 85..128 would break exactly wpe=3 and nothing else.
//    Grepped: there is none, the only v[128:135]/v[144:151] hits are comments.
//    The one asm block audited in full (the W13 MFMA pipeline) declares every
//    physical register it touches -- v7-v39 and a0-a3 -- in its clobber list.
//
// So: the register lock is still SHUT and its failure is not yet explained.
// Note the shape of it -- wpe=1 declares 596 B of scratch but has ZERO spills,
// so the persistent kernel never touches scratch at all; wpe=3's 67 spills are
// the first scratch traffic it has ever done. wpe=2 spills too and is fine,
// which weakens but does not kill that.
//
// The lower-risk route to <= 256 is NOT this knob. By the sum rule the default
// is 248 arch + 36 acc = 284, and both maxima are set by mla_decode_absorbed.
// Cutting its AGPR term alone from 36 to 8 gives 248 + 8 = 256 -> granule 256
// -> 2 waves/SIMD with NO spilling anywhere, because arch never moves. That is
// a genuine live-set reduction on a 16-worker phase, not a compiler flag.
// ---------------------------------------------------------------------------
#ifndef MPK_WORKER_WAVES_PER_EU
#define MPK_WORKER_WAVES_PER_EU 1
#endif
// Spelled as the direct AMDGPU attribute as well as the launch bound. Kept in
// the header rather than passed as a -D: hipcc re-quotes its argv through a
// wrapper and a -D carrying parens and commas does not survive the round trip
// -- it fails the build outright. MPK_WORKER_ATTR is defined at
// persistent_kernel above, which needs the same budget for the attributor's
// caller merge to have any effect.
__global__ MPK_WORKER_ATTR __launch_bounds__(WORKER_NUM_THREADS,
                                             MPK_WORKER_WAVES_PER_EU) void
    worker_kernel(RuntimeConfig config) {
  worker_checker(config);
  execute_worker(config);
}

__global__ void scheduler_kernel(RuntimeConfig config) {
  scheduler_checker(config);
  execute_scheduler(config, 0);
}
#pragma clang diagnostic pop

template <typename DT>
DT *gpu_malloc(size_t size) {
  void *dst_ptr;
#if defined(USE_NVSHMEM)
  dst_ptr = nvshmem_malloc(size);
#elif defined(USE_ROCSHMEM)
  // Multi-GPU: buffers reached by a cross-PE put+signal (notably
  // all_event_counters, the inter-GPU event-trigger counters) MUST live in the
  // symmetric heap so a remote PE's signal lands at the matching offset. A
  // plain hipMalloc here would make those addresses non-symmetric and corrupt
  // the remote signal, silently yielding wrong allreduce results.
  dst_ptr = mpk_shmem_malloc(size);
#else
  (void)cudaMalloc(&dst_ptr, size);
#endif
  return static_cast<DT *>(dst_ptr);
}

void gpu_free(void *ptr) {
#if defined(USE_NVSHMEM)
  nvshmem_free(ptr);
#elif defined(USE_ROCSHMEM)
  mpk_shmem_free(ptr);
#else
  (void)cudaFree(ptr);
#endif
}

// The following function will be generated by the transpiler
static void _init_persistent_kernel(std::vector<FullTaskDesc> &all_tasks,
                                    std::vector<EventDesc> &all_events,
                                    std::vector<TaskId> &first_tasks,
                                    int num_gpus,
                                    int my_gpu_id);

static RuntimeConfig global_runtime_config;
#ifdef MPK_PRECOMPUTED_DISPATCH
static unsigned long long *g_dbg_h_iter_ready = nullptr;
static int *g_dbg_h_terminate = nullptr;
static unsigned long long *g_dbg_h_tasks_done =
    nullptr; // host-mapped task counter
static int *g_dbg_h_worker_state =
    nullptr; // [num_workers*4] host-mapped debug state
static int g_dbg_num_workers = 0;
static EventCounter *g_dbg_h_event_counters =
    nullptr; // host-mapped event counters
static EventCounter *g_dbg_h_xcd_local_counters =
    nullptr; // host-mapped local counters
static int *g_dbg_h_xcd_event_thresholds = nullptr; // host copy of thresholds
static int g_dbg_num_events = 0;
static int g_dbg_num_xcds = 0;
#endif

#ifdef MPK_NIL_TRIPWIRE
// ── Tripwire for the nil-address GPU memory fault ────────────────────────
// The fault aborts the process. Device printf output is lost, and the GPU
// coredump ROCm writes carries no wavefront state, so the usual channels
// tell us nothing. This buffer is pinned host memory mapped into the GPU
// address space: kernel writes reach host RAM as they happen and survive
// the abort, and a SIGABRT handler prints them.
//
// Layout: MPK_TW_HDR unused header slots, then MPK_TW_PER_WORKER per worker.
// Per worker w, base = MPK_TW_HDR + w * MPK_TW_PER_WORKER:
//   [base+0] last layer this worker entered
//   [base+1] outer phase marker (100 layer entry, 200 gang tile, 300 non-gang)
//   [base+2] last tile_idx
//   [base+3] the pointer this worker was about to dereference
//   [base+4] sub-phase inside the fused layer kernel (see MPK_TW_SUB calls:
//            1 entry, 10 QKV, 20 QKV barrier, 30 attn, 40 chunk barrier,
//            50 merge, 60 cross-XCD barrier, 70 O-proj/TopK, 75 TopK wait,
//            80 MoE, 90 kernel exit)
//   [base+5] sub-phase aux value (meaning depends on the sub-phase)
//   [base+6] decode iteration (pc_iter)
static unsigned long long *g_tw_host = nullptr;
static int g_tw_num_workers = 0;
static bool g_tw_handler_installed = false;

// Async-signal-safe-ish dump. fprintf from a fatal handler is not strictly
// legal, but the process is aborting anyway and getting the data out matters
// more than standards purity here.
static void tripwire_dump(int sig) {
  if (g_tw_host == nullptr) {
    _exit(134);
  }
  fprintf(stderr, "\n===== MPK NIL TRIPWIRE (signal %d) =====\n", sig);
  // Per-worker breadcrumbs only; there is no global header counter, since
  // maintaining one would mean a device-wide atomic on the hot path.
  // What matters is the spread: if one worker is on a different layer from
  // the rest, or holds a null ptr, that names the culprit.
  int printed = 0;
  unsigned long long min_layer = ~0ull, max_layer = 0;
  int nulls = 0;
  for (int w = 0; w < g_tw_num_workers; w++) {
    unsigned long long const *b =
        g_tw_host + MPK_TW_HDR + w * MPK_TW_PER_WORKER;
    if (b[0] == 0 && b[1] == 0 && b[2] == 0 && b[3] == 0) {
      continue; // worker never wrote a breadcrumb
    }
    fprintf(stderr,
            "  w%-3d it=%llu layer=%llu phase=%llu sub=%llu aux=%llu "
            "tile=%llu ptr=0x%llx%s\n",
            w,
            b[6],
            b[0],
            b[1],
            b[4],
            b[5],
            b[2],
            b[3],
            b[3] == 0 ? "   <-- NULL" : "");
    if (b[3] == 0) {
      nulls++;
    }
    if (b[0] < min_layer) {
      min_layer = b[0];
    }
    if (b[0] > max_layer) {
      max_layer = b[0];
    }
    printed++;
  }
  fprintf(
      stderr,
      "(%d workers left breadcrumbs; layer span %llu..%llu; %d null ptrs)\n",
      printed,
      printed ? min_layer : 0,
      max_layer,
      nulls);
  fprintf(stderr, "===== END TRIPWIRE =====\n");
  fflush(stderr);
  // Restore default disposition and re-raise so the exit status is unchanged.
  signal(sig, SIG_DFL);
  raise(sig);
}

// Snapshot the breadcrumbs to disk on a timer.
//
// The signal handler alone is not enough: the ROCm runtime installs its own
// SIGABRT disposition after we arm ours, so on a real memory fault the
// process dies without our handler ever running (observed). A file on disk
// survives regardless of who wins the signal-handler race, and the fault
// aborts within ~1s of launch, so a short interval catches it.
static bool volatile g_tw_snap_stop = false;
static std::thread g_tw_snap_thread;

static void tripwire_snapshot_loop() {
  while (!g_tw_snap_stop) {
    FILE *f = fopen("/tmp/mpk_tripwire.txt", "w");
    if (f != nullptr) {
      for (int w = 0; w < g_tw_num_workers; w++) {
        unsigned long long const *b =
            g_tw_host + MPK_TW_HDR + w * MPK_TW_PER_WORKER;
        if (b[0] == 0 && b[1] == 0 && b[2] == 0 && b[3] == 0) {
          continue;
        }
        fprintf(f,
                "w%-3d it=%llu layer=%llu phase=%llu sub=%llu aux=%llu "
                "tile=%llu ptr=0x%llx%s\n",
                w,
                b[6],
                b[0],
                b[1],
                b[4],
                b[5],
                b[2],
                b[3],
                b[3] == 0 ? "   <-- NULL" : "");
      }
      fclose(f);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

static void install_tripwire_handler() {
  if (g_tw_handler_installed) {
    return;
  }
  signal(SIGABRT, tripwire_dump);
  signal(SIGSEGV, tripwire_dump);
  g_tw_snap_thread = std::thread(tripwire_snapshot_loop);
  g_tw_snap_thread.detach();
  g_tw_handler_installed = true;
}
#endif

// meta_tensors[0]: seq_length
// meta_tensors[1]: tokens
// meta_tensors[2]: input_tokens
// meta_tensors[3]: output_tokens
// meta_tensors[4]: new_tokens_nums
// meta_tensors[5]: prompt_length
// meta_tensors[6]: qo_indptr_buffer
// meta_tensors[7]: paged_kv_indptr_buffer
// meta_tensors[8]: paged_kv_indices_buffer
// meta_tensors[9]: paged_kv_last_page_len_buffer

extern "C" void init_request_resources() {
  init_kernel<<<dim3(1, 1, 1), dim3(INIT_NUM_THREADS, 1, 1)>>>(
      global_runtime_config);
  (void)cudaStreamSynchronize(NULL);
}

extern "C" void set_rope_tables(void *cos_ptr, void *sin_ptr) {
  global_runtime_config.rope_cos_ptr = cos_ptr;
  global_runtime_config.rope_sin_ptr = sin_ptr;
}

#ifdef MPK_SPEC_DECODE
// Hand prepare_next_batch the buffer the draft head writes into. Kept out of
// meta_tensors on purpose: that vector is asserted at size 10 and is shared
// with every other model in the repo.
extern "C" void set_spec_draft_tokens(void *ptr) {
  global_runtime_config.spec_draft_tokens = static_cast<long long *>(ptr);
}
#endif

// Diagnostic: copy `nbytes` from the `index`-th symmetric-heap allocation
// (allocation order recorded in mpk_shmem_alloc_registry) into device buffer
// `dst`. Returns 0 on success, -1 on bad index / size. Used by --verify to
// snapshot the post-allreduce mlp_final tensor -- a symmetric-heap tensor with
// no torch backing, so the Python side cannot otherwise read it.
extern "C" int mpk_read_shmem_alloc(int index, void *dst, size_t nbytes) {
  auto &reg = mpk_shmem_alloc_registry();
  if (index < 0 || index >= (int)reg.size()) {
    return -1;
  }
  if (nbytes > reg[index].second || reg[index].first == nullptr) {
    return -1;
  }
  (void)cudaMemcpy(dst, reg[index].first, nbytes, cudaMemcpyDeviceToDevice);
  (void)cudaDeviceSynchronize();
  return 0;
}

// Diagnostic: number of recorded symmetric-heap allocations.
extern "C" int mpk_num_shmem_allocs() {
  return (int)mpk_shmem_alloc_registry().size();
}

// Diagnostic: byte size of the index-th symmetric-heap allocation, or 0.
extern "C" unsigned long long mpk_shmem_alloc_size(int index) {
  auto &reg = mpk_shmem_alloc_registry();
  if (index < 0 || index >= (int)reg.size()) {
    return 0ULL;
  }
  return (unsigned long long)reg[index].second;
}

extern "C" void init_persistent_kernel(std::vector<void *> meta_tensors,
                                       void *profiler_buffer,
                                       int my_rank,
                                       int num_workers,
                                       int num_local_schedulers,
                                       int num_remote_schedulers,
                                       int max_seq_length,
                                       int total_num_requests,
                                       long long eos_token_id) {
  assert(meta_tensors.size() == 10);
  global_runtime_config.step = static_cast<int *>(meta_tensors[0]);
  global_runtime_config.tokens = static_cast<long long *>(meta_tensors[1]);
  global_runtime_config.input_tokens =
      static_cast<long long *>(meta_tensors[2]);
  global_runtime_config.output_tokens =
      static_cast<long long *>(meta_tensors[3]);
  global_runtime_config.new_token_nums = static_cast<int *>(meta_tensors[4]);
  global_runtime_config.prompt_length = static_cast<int *>(meta_tensors[5]);
  global_runtime_config.qo_indptr_buffer = static_cast<int *>(meta_tensors[6]);
  global_runtime_config.paged_kv_indptr_buffer =
      static_cast<int *>(meta_tensors[7]);
  global_runtime_config.paged_kv_indices_buffer =
      static_cast<int *>(meta_tensors[8]);
  global_runtime_config.paged_kv_last_page_len_buffer =
      static_cast<int *>(meta_tensors[9]);
  global_runtime_config.rope_cos_ptr = nullptr;
  global_runtime_config.rope_sin_ptr = nullptr;
#ifdef MPK_SPEC_DECODE
  global_runtime_config.spec_draft_tokens = nullptr;
#endif
  global_runtime_config.num_workers = num_workers;
  global_runtime_config.num_local_schedulers = num_local_schedulers;
  global_runtime_config.num_remote_schedulers = num_remote_schedulers;
  global_runtime_config.max_seq_length = max_seq_length;
  global_runtime_config.eos_token_id = eos_token_id;
  global_runtime_config.profiler_buffer = profiler_buffer;
  int num_schedulers = num_local_schedulers + num_remote_schedulers;

  // Initialize nvshmem
  (void)cudaSetDevice(my_rank);

#ifdef USE_NVSHMEM
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
  nvshmem_barrier_all();
  int mype = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  printf("mype(%d) npes(%d) mype_node(%d)\n", mype, npes, mype_node);
#elif defined(USE_ROCSHMEM)
  // rocSHMEM bootstraps MPI internally; the target device must already be
  // selected (the cudaSetDevice above) before rocshmem_init().
  rocshmem::rocshmem_init();
  rocshmem::rocshmem_barrier_all();
  int mype = rocshmem::rocshmem_my_pe();
  int npes = rocshmem::rocshmem_n_pes();
  printf("mype(%d) npes(%d)\n", mype, npes);
  // rocSHMEM's IPC init (symmetric-heap reservation + peer-access enable) can
  // leave the current HIP device != my_rank, unlike NVSHMEM which preserves it.
  // Re-assert it so the subsequent device-to-device weight staging (which reads
  // this rank's torch tensors on device my_rank) runs on the right device.
  (void)cudaSetDevice(my_rank);
#else
  int mype = 0;
  int npes = 1;
#endif

#if defined(MODE_OFFLINE) || defined(MODE_ONLINE)
  global_runtime_config.request_ids =
      gpu_malloc<int>(sizeof(int) * (MPK_MAX_NUM_BATCHED_REQUESTS + 1));
  global_runtime_config.next_request_id = gpu_malloc<int>(sizeof(int));
  global_runtime_config.page_queue =
      gpu_malloc<int>(MPK_MAX_NUM_PAGES * sizeof(int));
  global_runtime_config.page_queue_head = gpu_malloc<int>(sizeof(int));
  global_runtime_config.page_queue_tail = gpu_malloc<int>(sizeof(int));
  global_runtime_config.total_num_requests = total_num_requests;
#endif
  global_runtime_config.per_worker_queue_len = 1024;
  global_runtime_config.per_sched_queue_len = 1024;
  global_runtime_config.num_gpus = npes;
  global_runtime_config.my_gpu_id = mype;
  global_runtime_config.num_graphs = 1;
  global_runtime_config.profiling_num_iters = MPK_PROFILING_NUM_ITERS;
  // Split mode: workers and schedulers in separate kernel launches.
  // Note: rocprofv3 --pmc serializes kernel dispatches and deadlocks split
  // mode, but --profiling (Perfetto trace) works fine since profiler writes are
  // worker-only.
  global_runtime_config.split_worker_scheduler = true;

  std::vector<FullTaskDesc> all_fulltasks;
  std::vector<EventDesc> all_events;
  std::vector<TaskId> first_tasks;
  _init_persistent_kernel(all_fulltasks, all_events, first_tasks, npes, mype);

  // Sample the symmetric-heap peer deltas now that _init_persistent_kernel has
  // done every mpk_shmem_malloc. This is what lets the EP combine translate a
  // local symmetric address to its peer with an add instead of a call into
  // rocshmem_ptr on the hot path -- see mpk_comm.cuh. Any recorded symmetric
  // allocation serves as the probe; the delta is heap-wide, not per-object.
  {
    auto &reg = mpk_shmem_alloc_registry();
    void *probe = nullptr;
    for (auto const &a : reg) {
      if (a.first != nullptr) {
        probe = a.first;
        break;
      }
    }
    mpk_shmem_init_peer_deltas(probe);
  }

  std::vector<TaskDesc> all_tasks;
  for (auto const &ft : all_fulltasks) {
    TaskDesc task_desc(ft);
    // if (ft.task_type == TASK_PAGED_ATTENTION_SPLIT_KV_SM100 || ft.task_type
    // == TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_SM100) {
    //   printf("ft.kv_idx %d\n", ft.kv_idx);
    //   printf("ft.merge_task_offset %d\n", ft.merge_task_offset);
    // }
    // Reinterpret part of TaskDesc to save xfer_size information
    if (ft.task_type == TASK_NVSHMEM_COPY) {
      int size_in_bytes = 2;
      for (int i = 0; i < ft.inputs[0].num_dims; i++) {
        size_in_bytes *= ft.inputs[0].dim[i];
      }
      task_desc.task_metadata.xfer_size_in_bytes = size_in_bytes;
    }
    all_tasks.push_back(task_desc);
  }

  // Dump task graph for dispatch benchmark
  {
    char const *dump_path = getenv("MPK_DUMP_TASK_GRAPH");
    if (dump_path) {
      FILE *f = fopen(dump_path, "wb");
      if (f) {
        int n_tasks = (int)all_tasks.size();
        int n_events = (int)all_events.size();
        int n_workers = num_workers;
        fwrite(&n_tasks, sizeof(int), 1, f);
        fwrite(&n_events, sizeof(int), 1, f);
        fwrite(&n_workers, sizeof(int), 1, f);
        // Per task: task_type(int), dep_event_idx(int), trig_event_idx(int),
        //           is_gang(int), n_tile_count(int)
        for (int i = 0; i < n_tasks; i++) {
          int task_type = (int)all_tasks[i].task_type;
          int dep_event = -1;
          if (all_tasks[i].dependent_event != EVENT_INVALID_ID) {
            dep_event = (int)(all_tasks[i].dependent_event & 0xffffffff);
          }
          int trig_event = -1;
          if (all_tasks[i].trigger_event != EVENT_INVALID_ID) {
            trig_event = (int)(all_tasks[i].trigger_event & 0xffffffff);
          }
          int is_gang = is_gang_task_type(all_tasks[i].task_type) ? 1 : 0;
          int n_tile_count =
              is_gang ? (int)all_tasks[i].task_metadata.n_tile_count : 0;
          fwrite(&task_type, sizeof(int), 1, f);
          fwrite(&dep_event, sizeof(int), 1, f);
          fwrite(&trig_event, sizeof(int), 1, f);
          fwrite(&is_gang, sizeof(int), 1, f);
          fwrite(&n_tile_count, sizeof(int), 1, f);
        }
        // Per event: event_type(int), num_triggers(int), first_task(int),
        // last_task(int)
        for (int i = 0; i < n_events; i++) {
          int etype = (int)all_events[i].event_type;
          int ntrig = all_events[i].num_triggers;
          int first = (int)(all_events[i].first_task_id & 0xffffffff);
          int last = (int)(all_events[i].last_task_id & 0xffffffff);
          fwrite(&etype, sizeof(int), 1, f);
          fwrite(&ntrig, sizeof(int), 1, f);
          fwrite(&first, sizeof(int), 1, f);
          fwrite(&last, sizeof(int), 1, f);
        }
        fclose(f);
        printf(
            "[MPK] Dumped task graph: %d tasks, %d events, %d workers -> %s\n",
            n_tasks,
            n_events,
            n_workers,
            dump_path);
      }
    }
  }

#ifdef MPK_PRECOMPUTED_DISPATCH
  // =========================================================================
  // Multi-layer fusion: build pointer table from all 36 layers, then compact
  // the task graph by removing redundant layers 1..35 (280 tasks, 35 events).
  // Must happen BEFORE template queue building so the template uses compacted
  // indices.
  // =========================================================================
#ifdef MPK_FUSED_LAYER_BATCHING
  {
    constexpr int NUM_XCDS_ML = 8;
    // Must cover every slot the fused layer kernels read, not just the ones
    // the plain variant uses. The LM-head variant
    // (TASK_GANG_FULL_LAYER_WITH_LMHEAD_FUSED_MI300, FUSE_TAIL=1) reads
    // input_ptrs[24..27] and output_ptrs[12] -- see
    // gang_full_layer_with_lmhead_fused_mi300.cuh. These were 24 and 11,
    // sized for the plain variant, so with FUSE_TAIL=1 the ml loop would
    // refresh only the first 24/11 slots and leave the LM-head pointers
    // holding layer 0's values for every later layer. Sizing to the TaskDesc
    // capacity keeps this correct for any variant.
    constexpr int ML_N_IN = MAX_INPUTS_PER_TASK;
    constexpr int ML_N_OUT = MAX_OUTPUTS_PER_TASK; // 13
    int n_tasks_pre = (int)all_tasks.size();
    int n_events_pre = (int)all_events.size();

    // Scan for consecutive FULL_LAYER_FUSED gang tasks (8 XCD copies each)
    // We look for groups of 8 consecutive tasks of the same fused type.
    std::vector<size_t> fused_layer_positions; // first task (XCD 0) per layer
    {
      size_t t = 0;
      while (t < all_tasks.size()) {
        if (is_gang_task_type(all_tasks[t].task_type) &&
            (all_tasks[t].task_type == TASK_GANG_FULL_LAYER_FUSED_MI300 ||
             all_tasks[t].task_type ==
                 TASK_GANG_FULL_LAYER_WITH_LMHEAD_FUSED_MI300 ||
             all_tasks[t].task_type ==
                 TASK_GANG_MLA_FULL_LAYER_FUSED_MI300) &&
            t + NUM_XCDS_ML <= all_tasks.size()) {
          fused_layer_positions.push_back(t);
          t += NUM_XCDS_ML;
        } else {
          t++;
        }
      }
    }

    int ml_layers = (int)fused_layer_positions.size();
    printf("[MPK] Multi-layer scan: found %d fused layers (%d tasks "
           "pre-compact, %d events)\n",
           ml_layers,
           n_tasks_pre,
           n_events_pre);
    fflush(stdout);

    // Multi-layer replay is only legal when the fused layers are back-to-back.
    // The replay loop runs all 36 layers inside ONE task without returning to
    // the scheduler, so it cannot execute anything the graph placed *between*
    // layers. Compaction then deletes layers 1..35, and whatever sat between
    // them keeps its dependent events -- events nothing triggers any more. The
    // graph stalls with N-1 dead events and no error.
    //
    // This is exactly what expert-parallel MoE does: it inserts a
    // residual-fold + allreduce after every layer's combine. Detect the gap
    // and fall back to per-layer dispatch (slower, but the graph runs).
    // Stamp each fused task with its layer index.
    //
    // The fused kernel derives every barrier release value from
    // task_layer_idx == task_metadata._linear_reserved (see the layer-counter
    // discussion in gang_full_layer_fused_mi300.cuh). Until now the ONLY
    // writer of that field was the replay loop, so on the per-layer dispatch
    // path the kernel read whatever the host happened to leave in the union --
    // and waited on epoch values no producer would ever reach. Stamping here
    // makes the field meaningful on both paths; replay overwrites it with the
    // same number.
    global_runtime_config.ml_scanned_layers = ml_layers;
    for (int L = 0; L < ml_layers; L++) {
      for (int xcd = 0; xcd < NUM_XCDS_ML; xcd++) {
        all_tasks[fused_layer_positions[L] + xcd]
            .task_metadata._linear_reserved = L;
      }
    }

    // MPK_ML_REPLAY=0 forces the fallback on a graph that would otherwise
    // qualify. This is the only way to run the per-layer fused dispatch path
    // on a config where replay is legal, which is what isolates a bug in that
    // path from a bug in whatever made replay illegal.
    {
      char const *rep = getenv("MPK_ML_REPLAY");
      if (rep && atoi(rep) == 0 && ml_layers > 1) {
        printf("[MPK] Multi-layer table: DISABLED by MPK_ML_REPLAY=0\n");
        fflush(stdout);
        ml_layers = 1;
      }
    }

    int inter_layer_gap = 0;
    for (int L = 1; L < ml_layers; L++) {
      size_t gap = fused_layer_positions[L] -
                   (fused_layer_positions[L - 1] + NUM_XCDS_ML);
      inter_layer_gap += (int)gap;
    }
    if (ml_layers > 1 && inter_layer_gap > 0) {
      printf("[MPK] Multi-layer table: DISABLED -- %d task(s) sit between "
             "fused layers (per-layer collectives?). Replay would delete "
             "them; falling back to per-layer dispatch.\n",
             inter_layer_gap);
      fflush(stdout);
      ml_layers = 1; // skip compaction below
    }

    if (ml_layers > 1) {
      // Build per-XCD pointer tables from all layers BEFORE compaction
      std::vector<void *> h_input_table(NUM_XCDS_ML * ml_layers * ML_N_IN);
      std::vector<void *> h_output_table(NUM_XCDS_ML * ml_layers * ML_N_OUT);
      std::vector<EventId> h_trigger_events(ml_layers);
      std::vector<unsigned> h_variant_ids(ml_layers);

      for (int L = 0; L < ml_layers; L++) {
        size_t pos = fused_layer_positions[L];
        for (int xcd = 0; xcd < NUM_XCDS_ML; xcd++) {
          TaskDesc const &td_xcd = all_tasks[pos + xcd];
          int base = (xcd * ml_layers + L) * ML_N_IN;
          for (int i = 0; i < ML_N_IN; i++) {
            h_input_table[base + i] = td_xcd.input_ptrs[i];
          }
          int obase = (xcd * ml_layers + L) * ML_N_OUT;
          for (int i = 0; i < ML_N_OUT; i++) {
            h_output_table[obase + i] = td_xcd.output_ptrs[i];
          }
        }
        TaskDesc const &td = all_tasks[pos];
        h_trigger_events[L] = td.trigger_event;
        h_variant_ids[L] = td.variant_id;
        printf("[MPK]   Layer %d: pos=%zu variant=%u type=%d\n",
               L,
               pos,
               td.variant_id,
               (int)td.task_type);
      }

      // Validate the tables before upload. The layer loop dereferences every
      // slot the kernel reads, so a null there is a nil-address GPU fault with
      // no usable backtrace -- catching it on the host names the exact slot.
      //
      // Only the slots the task type actually declares are checked. The tables
      // are sized to the TaskDesc capacity so the LM-head variant's higher
      // slots get refreshed, but the plain variant declares 24 inputs / 11
      // outputs (see task_config in graph.cc), and slots past its count are
      // legitimately null -- TaskDesc value-initializes them. Flagging those
      // would report 1152/576 "nulls" on a perfectly healthy graph and train
      // the reader to ignore this line.
      {
        int null_in = 0, null_out = 0, unused_in = 0, unused_out = 0;
        for (int L = 0; L < ml_layers; L++) {
          TaskType lt = all_tasks[fused_layer_positions[L]].task_type;
          bool is_lmhead = lt == TASK_GANG_FULL_LAYER_WITH_LMHEAD_FUSED_MI300;
          // GLM's fused layer declares 27 inputs / 11 outputs, +2 inputs under
          // EP, +1 in / +1 out when kv_b_v is un-absorbed (the packed W_UV
          // weight and its V workspace) and +1 in / +1 out again when kv_b_k
          // is (the W_UK weight and the nope scratch); gpt-oss's plain variant
          // 24 / 11 and its LM-head variant 28 / 13. The arity is detected
          // rather than passed: each of those trailing slots is bound only by
          // the graph that uses it, and on every fused layer if at all.
          TaskDesc const &lt_td = all_tasks[fused_layer_positions[L]];
          int used_in =
              is_lmhead
                  ? 28
                  : (lt == TASK_GANG_MLA_FULL_LAYER_FUSED_MI300
                         ? (lt_td.input_ptrs[30]   ? 31
                            : lt_td.input_ptrs[29] ? 30
                            : lt_td.input_ptrs[28] ? 29
                            : lt_td.input_ptrs[27] ? 28
                                                   : 27)
                         : 24);
          int used_out =
              is_lmhead
                  ? 13
                  : (lt == TASK_GANG_MLA_FULL_LAYER_FUSED_MI300
                         ? (lt_td.output_ptrs[12]   ? 13
                            : lt_td.output_ptrs[11] ? 12
                                                    : 11)
                         : 11);
          for (int xcd = 0; xcd < NUM_XCDS_ML; xcd++) {
            int base = (xcd * ml_layers + L) * ML_N_IN;
            for (int i = 0; i < ML_N_IN; i++) {
              if (h_input_table[base + i] == nullptr) {
                if (i >= used_in) {
                  unused_in++;
                  continue;
                }
                if (null_in < 8) {
                  printf(
                      "[MPK] ML_NULL_IN layer=%d xcd=%d idx=%d\n", L, xcd, i);
                }
                null_in++;
              }
            }
            int obase = (xcd * ml_layers + L) * ML_N_OUT;
            for (int i = 0; i < ML_N_OUT; i++) {
              if (h_output_table[obase + i] == nullptr) {
                if (i >= used_out) {
                  unused_out++;
                  continue;
                }
                if (null_out < 8) {
                  printf(
                      "[MPK] ML_NULL_OUT layer=%d xcd=%d idx=%d\n", L, xcd, i);
                }
                null_out++;
              }
            }
          }
        }
        // Only report when something is actually wrong. A healthy graph has
        // null_in == null_out == 0, and printing that on every run trains the
        // reader to skip the line that matters.
        if (null_in || null_out) {
          printf("[MPK] ML table validation: %d null inputs, %d null outputs "
                 "(%d/%d null in undeclared slots, expected)\n",
                 null_in,
                 null_out,
                 unused_in,
                 unused_out);
          fflush(stdout);
        }
      }

      // === Compact task graph: remove layers 1..35 tasks and inter-layer
      // events === Collect tasks and events to remove
      std::set<size_t> tasks_to_remove; // positions in all_tasks
      std::set<int> events_to_remove;   // indices in all_events

      for (int L = 1; L < ml_layers; L++) {
        size_t pos = fused_layer_positions[L];
        // Remove all 8 XCD copies of this layer
        for (int xcd = 0; xcd < NUM_XCDS_ML; xcd++) {
          tasks_to_remove.insert(pos + xcd);
        }
        // Remove the inter-layer event (dep_event of this layer's tasks)
        EventId dep_ev = all_tasks[pos].dependent_event;
        if (dep_ev != EVENT_INVALID_ID) {
          int dep_idx = (int)(dep_ev & 0xffffffff);
          events_to_remove.insert(dep_idx);
        }
      }

      printf("[MPK] Compaction: removing %d tasks, %d events\n",
             (int)tasks_to_remove.size(),
             (int)events_to_remove.size());

      // Fix layer 0's trigger_event: redirect to last layer's trigger event.
      // The last layer's trigger event gates FUSE_TAIL. After compaction,
      // layers 1..35 are gone, so layer 0 must trigger the FUSE_TAIL consumer.
      {
        size_t layer0_pos = fused_layer_positions[0];
        EventId last_trigger =
            all_tasks[fused_layer_positions[ml_layers - 1]].trigger_event;
        for (int xcd = 0; xcd < NUM_XCDS_ML; xcd++) {
          all_tasks[layer0_pos + xcd].trigger_event = last_trigger;
        }
        // Update h_trigger_events[0] to match the redirected trigger_event
        // (the original was an inter-layer event that gets removed)
        h_trigger_events[0] = last_trigger;
        // Note: the last layer's event's first/last_task_id already point at
        // the FUSE_TAIL consumer tasks (set by dfs_create_events_add_tasks).
        // Don't modify them — the remap will adjust positions correctly.
      }

      // Build remap tables for compaction
      std::vector<int> task_remap(n_tasks_pre, -1);
      std::vector<int> event_remap(n_events_pre, -1);
      {
        int new_idx = 0;
        for (int i = 0; i < n_tasks_pre; i++) {
          if (tasks_to_remove.count(i) == 0) {
            task_remap[i] = new_idx++;
          }
        }
        new_idx = 0;
        for (int i = 0; i < n_events_pre; i++) {
          if (events_to_remove.count(i) == 0) {
            event_remap[i] = new_idx++;
          }
        }
      }

      // Remap EventId fields in all tasks
      auto remap_event = [&](EventId eid) -> EventId {
        if (eid == EVENT_INVALID_ID) {
          return eid;
        }
        int idx = (int)(eid & 0xffffffff);
        uint64_t iter = eid >> 32;
        if (idx < n_events_pre && event_remap[idx] >= 0) {
          return ((uint64_t)event_remap[idx]) | (iter << 32);
        }
        return EVENT_INVALID_ID; // removed event
      };
      for (int i = 0; i < n_tasks_pre; i++) {
        if (tasks_to_remove.count(i)) {
          continue;
        }
        all_tasks[i].trigger_event = remap_event(all_tasks[i].trigger_event);
        all_tasks[i].dependent_event =
            remap_event(all_tasks[i].dependent_event);
      }

      // Remap first_task_id/last_task_id in all events
      for (int i = 0; i < n_events_pre; i++) {
        if (events_to_remove.count(i)) {
          continue;
        }
        int first = (int)(all_events[i].first_task_id & 0xffffffff);
        int last = (int)(all_events[i].last_task_id & 0xffffffff);
        if (first < n_tasks_pre && task_remap[first] >= 0) {
          all_events[i].first_task_id = (size_t)task_remap[first];
        }
        if (last < n_tasks_pre && task_remap[last] >= 0) {
          all_events[i].last_task_id = (size_t)task_remap[last];
        } else if (last > 0 && last - 1 < n_tasks_pre &&
                   task_remap[last - 1] >= 0) {
          all_events[i].last_task_id = (size_t)(task_remap[last - 1] + 1);
        }
      }

      // Compact the vectors
      {
        std::vector<TaskDesc> new_tasks;
        for (int i = 0; i < n_tasks_pre; i++) {
          if (tasks_to_remove.count(i) == 0) {
            new_tasks.push_back(all_tasks[i]);
          }
        }
        all_tasks = std::move(new_tasks);
      }
      {
        std::vector<EventDesc> new_events;
        for (int i = 0; i < n_events_pre; i++) {
          if (events_to_remove.count(i) == 0) {
            new_events.push_back(all_events[i]);
          }
        }
        all_events = std::move(new_events);
      }

      printf("[MPK] Compacted: %d -> %d tasks, %d -> %d events\n",
             n_tasks_pre,
             (int)all_tasks.size(),
             n_events_pre,
             (int)all_events.size());
      fflush(stdout);

      // Update h_trigger_events with remapped event indices for GPU upload
      for (int L = 0; L < ml_layers; L++) {
        h_trigger_events[L] = remap_event(h_trigger_events[L]);
      }

      // Build h_task_positions from layer 0's compacted position
      std::vector<size_t> h_task_positions(ml_layers);
      h_task_positions[0] = task_remap[fused_layer_positions[0]];
      for (int L = 1; L < ml_layers; L++) {
        // Layers 1..35 are removed — use layer 0's position as placeholder
        h_task_positions[L] = h_task_positions[0];
      }

      // Upload pointer tables + multi-layer config to GPU
      global_runtime_config.ml_input_table = gpu_malloc<void *>(
          NUM_XCDS_ML * ml_layers * ML_N_IN * sizeof(void *));
      (void)cudaMemcpy(global_runtime_config.ml_input_table,
                       h_input_table.data(),
                       NUM_XCDS_ML * ml_layers * ML_N_IN * sizeof(void *),
                       cudaMemcpyHostToDevice);
      global_runtime_config.ml_output_table = gpu_malloc<void *>(
          NUM_XCDS_ML * ml_layers * ML_N_OUT * sizeof(void *));
      (void)cudaMemcpy(global_runtime_config.ml_output_table,
                       h_output_table.data(),
                       NUM_XCDS_ML * ml_layers * ML_N_OUT * sizeof(void *),
                       cudaMemcpyHostToDevice);
      global_runtime_config.ml_trigger_events =
          gpu_malloc<EventId>(ml_layers * sizeof(EventId));
      (void)cudaMemcpy(global_runtime_config.ml_trigger_events,
                       h_trigger_events.data(),
                       ml_layers * sizeof(EventId),
                       cudaMemcpyHostToDevice);
      global_runtime_config.ml_task_positions =
          gpu_malloc<size_t>(ml_layers * sizeof(size_t));
      (void)cudaMemcpy(global_runtime_config.ml_task_positions,
                       h_task_positions.data(),
                       ml_layers * sizeof(size_t),
                       cudaMemcpyHostToDevice);
      global_runtime_config.ml_variant_ids =
          gpu_malloc<unsigned>(ml_layers * sizeof(unsigned));
      (void)cudaMemcpy(global_runtime_config.ml_variant_ids,
                       h_variant_ids.data(),
                       ml_layers * sizeof(unsigned),
                       cudaMemcpyHostToDevice);

      // Barrier counters (zeroed)
      global_runtime_config.ml_barrier_arrive =
          gpu_malloc<int>(8 * 16 * sizeof(int));
      (void)cudaMemset(
          global_runtime_config.ml_barrier_arrive, 0, 8 * 16 * sizeof(int));
      global_runtime_config.ml_barrier_global =
          gpu_malloc<int>(16 * sizeof(int));
      (void)cudaMemset(
          global_runtime_config.ml_barrier_global, 0, 16 * sizeof(int));
      global_runtime_config.ml_barrier_release =
          gpu_malloc<int>(8 * 16 * sizeof(int));
      (void)cudaMemset(
          global_runtime_config.ml_barrier_release, 0, 8 * 16 * sizeof(int));

      global_runtime_config.ml_num_layers = ml_layers;
      // ml_workers_per_xcd set later after workers_per_xcd is computed

      printf(
          "[MPK] Multi-layer table: %d fused layers, pointer tables uploaded\n",
          ml_layers);
      fflush(stdout);
    } else {
      global_runtime_config.ml_num_layers = 0;
      printf("[MPK] Multi-layer table: disabled (%d fused layers found)\n",
             ml_layers);
      fflush(stdout);
    }
  }
#else
  global_runtime_config.ml_num_layers = 0;
  global_runtime_config.ml_scanned_layers = 0;
#endif // MPK_FUSED_LAYER_BATCHING

  // =========================================================================
  // Build per-XCD-slot template queues on host.
  // At runtime, workers discover their hardware XCD via get_current_xcd_id(),
  // claim a rank, and copy the template for their XCD slot. This ensures gang
  // tasks land on the correct hardware XCD (fixing the logical-vs-hardware
  // XCD mismatch that caused internal hierarchical barrier deadlocks).
  // =========================================================================
  printf("[MPK] Building XCD-template dispatch queues...\n");
  fflush(stdout);
  {
    int n_tasks = (int)all_tasks.size();
    int n_events = (int)all_events.size();
    constexpr int NUM_XCDS_PC = 8;
    int workers_per_xcd = num_workers / NUM_XCDS_PC;
    int max_tpw = 512;

    // Template queues: indexed by [xcd_slot][rank][task_index]
    // Flattened: xcd_slot * workers_per_xcd * max_tpw + rank * max_tpw + idx
    int total_template_entries = NUM_XCDS_PC * workers_per_xcd;
    std::vector<size_t> h_tmpl(total_template_entries * max_tpw, 0);
    std::vector<int> h_tmpl_len(total_template_entries, 0);

    auto enqueue_tmpl = [&](int xcd_slot, int rank, size_t pos) {
      int flat = xcd_slot * workers_per_xcd + rank;
      if (h_tmpl_len[flat] < max_tpw) {
        h_tmpl[flat * max_tpw + h_tmpl_len[flat]++] = pos;
      }
    };

    // begin_task_graph (position 1) → XCD 0, rank 0
    enqueue_tmpl(0, 0, 1);

    // Find EVENT_LAUNCH_DEPENDENT_TASKS (type 903) → main task range
    size_t main_first = 2, main_last = n_tasks;
    for (int i = 0; i < n_events; i++) {
      if (all_events[i].event_type == EVENT_LAUNCH_DEPENDENT_TASKS) {
        main_first = all_events[i].first_task_id;
        main_last = all_events[i].last_task_id;
        break;
      }
    }

    // Build event groups from sub-events + find orphaned tasks
    struct EventGroup {
      int dep_event;
      size_t first_task, last_task;
      bool is_gang;
    };
    std::vector<EventGroup> event_groups;
    std::set<size_t> covered_tasks;

    for (int i = 2; i < n_events - 1; i++) {
      size_t ef = all_events[i].first_task_id;
      size_t el = all_events[i].last_task_id;
      if (ef < main_first || ef >= main_last) {
        continue;
      }
      bool gang = ((int)(el - ef) == NUM_XCDS_PC &&
                   is_gang_task_type(all_tasks[ef].task_type));
      event_groups.push_back({i, ef, el, gang});
      for (size_t t = ef; t < el; t++) {
        covered_tasks.insert(t);
      }
    }

    // Orphaned tasks: in main range but not in any sub-event group
    std::vector<size_t> orphan_tasks;
    for (size_t t = main_first; t < main_last; t++) {
      if (covered_tasks.find(t) == covered_tasks.end()) {
        orphan_tasks.push_back(t);
      }
    }

    // Dispatch orphans to XCD 0, rank 0 (e.g., task 2: embed, dep=-1)
    for (size_t t : orphan_tasks) {
      enqueue_tmpl(0, 0, t);
    }

    // Round-robin state per XCD
    int rr_idx[NUM_XCDS_PC];
    for (int s = 0; s < NUM_XCDS_PC; s++) {
      rr_idx[s] = 0;
    }

    // Process each event group
    for (auto &eg : event_groups) {
      if (eg.is_gang) {
        // Gang: XCD slot N gets task first_task + N.
        // At runtime, hardware XCD H uses template slot H, so task first_task +
        // H lands on the correct hardware XCD.
        for (int xcd = 0; xcd < NUM_XCDS_PC; xcd++) {
          size_t raw_idx = eg.first_task + xcd;
          int ntiles = (int)all_tasks[raw_idx].task_metadata.n_tile_count;
          int dispatch_count = std::min(ntiles, workers_per_xcd);
          for (int w = 0; w < dispatch_count; w++) {
            enqueue_tmpl(xcd, w, raw_idx);
          }
        }
      } else {
        // Non-gang: round-robin across ranks within each XCD slot
        for (size_t t = eg.first_task; t < eg.last_task; t++) {
          int sched = (int)(t % NUM_XCDS_PC);
          int rank = rr_idx[sched];
          enqueue_tmpl(sched, rank, t);
          rr_idx[sched] = (rank + 1) % workers_per_xcd;
        }
      }
    }

    // Build gang_dispatch_count (for cross-XCD gang barrier)
    int n_xcds = NUM_XCDS_PC;
    std::vector<int> h_gang_dispatch_count(n_events, 0);

    for (auto &eg : event_groups) {
      if (eg.is_gang) {
        for (int xcd = 0; xcd < n_xcds; xcd++) {
          size_t raw_idx = eg.first_task + xcd;
          int ntiles = (int)all_tasks[raw_idx].task_metadata.n_tile_count;
          int dispatch_count = std::min(ntiles, workers_per_xcd);
          EventId trigger_ev = all_tasks[raw_idx].trigger_event;
          int tev_idx = (int)(trigger_ev & 0xffffffff);
          h_gang_dispatch_count[tev_idx] += dispatch_count;
        }
      }
    }

    // Sort each worker's queue into task-index order.
    //
    // Workers execute their queue strictly front-to-back and block on each
    // task's dependency event, so a queue whose order disagrees with the
    // graph's topological order deadlocks: the worker parks on a late task
    // while an earlier one -- which some other worker is waiting on -- sits
    // behind it, unreachable.
    //
    // The queues are built by iterating event groups, and event index order is
    // NOT topological with respect to task index. On the single-fused-layer
    // graphs this never showed, because compaction left one event group per
    // stage and the orders coincided. Under expert parallelism there are ~100
    // groups per layer and they do not: 8 workers (one per XCD) were observed
    // parked on a layer-4 collective while 97 others spun in layer 4's
    // monolith barriers short_by=1, waiting on exactly those 8.
    //
    // dfs_create_events_add_tasks assigns task indices in dependency order, so
    // ascending task index is a valid topological order and sorting is enough.
    // begin_task_graph (position 1) and the orphans are already ahead of the
    // main range by index, so this does not disturb them.
    for (int i = 0; i < total_template_entries; i++) {
      size_t *q = h_tmpl.data() + (size_t)i * max_tpw;
      std::sort(q, q + h_tmpl_len[i]);
    }

    int gang_barrier_events = 0;
    for (int e = 0; e < n_events; e++) {
      if (h_gang_dispatch_count[e] > 0) {
        gang_barrier_events++;
      }
    }

    // Stats
    int min_t = 999999, max_t = 0, sum_t = 0;
    for (int i = 0; i < total_template_entries; i++) {
      sum_t += h_tmpl_len[i];
      min_t = std::min(min_t, h_tmpl_len[i]);
      max_t = std::max(max_t, h_tmpl_len[i]);
    }
    printf("[MPK] XCD-template dispatch: workers=%d orphans=%d min=%d max=%d "
           "avg=%.1f total=%d tasks/worker\n",
           num_workers,
           (int)orphan_tasks.size(),
           min_t,
           max_t,
           (float)sum_t / total_template_entries,
           sum_t);
    printf("[MPK] Gang barrier events: %d (of %d total events)\n",
           gang_barrier_events,
           n_events);
    fflush(stdout);

    // Allocate per-worker queue space (workers fill from templates at runtime)
    global_runtime_config.precomp_max_tpw = max_tpw;
    global_runtime_config.precomp_workers_per_xcd = workers_per_xcd;
    global_runtime_config.precomp_queue =
        gpu_malloc<size_t>(num_workers * max_tpw * sizeof(size_t));
    global_runtime_config.precomp_queue_len =
        gpu_malloc<int>(num_workers * sizeof(int));

    // Upload XCD templates to GPU
    global_runtime_config.precomp_xcd_template =
        gpu_malloc<size_t>(total_template_entries * max_tpw * sizeof(size_t));
    (void)cudaMemcpy(global_runtime_config.precomp_xcd_template,
                     h_tmpl.data(),
                     total_template_entries * max_tpw * sizeof(size_t),
                     cudaMemcpyHostToDevice);
    global_runtime_config.precomp_xcd_template_len =
        gpu_malloc<int>(total_template_entries * sizeof(int));
    (void)cudaMemcpy(global_runtime_config.precomp_xcd_template_len,
                     h_tmpl_len.data(),
                     total_template_entries * sizeof(int),
                     cudaMemcpyHostToDevice);

    // Per-XCD atomic rank counter (workers claim ranks at runtime)
    global_runtime_config.precomp_xcd_rank_counter =
        gpu_malloc<int>(NUM_XCDS_PC * sizeof(int));
    (void)cudaMemset(global_runtime_config.precomp_xcd_rank_counter,
                     0,
                     NUM_XCDS_PC * sizeof(int));

    // Use device memory for iter_ready and terminate (GPU-only polling)
    {
      global_runtime_config.precomp_iter_ready =
          gpu_malloc<unsigned long long>(sizeof(unsigned long long));
      (void)cudaMemset(global_runtime_config.precomp_iter_ready,
                       0,
                       sizeof(unsigned long long));

      global_runtime_config.precomp_terminate = gpu_malloc<int>(sizeof(int));
      (void)cudaMemset(global_runtime_config.precomp_terminate, 0, sizeof(int));

      // Per-XCD release flags for fast cross-XCD iteration barrier
      // [8*16 ints] padded to cache lines to avoid false sharing
      global_runtime_config.precomp_iter_xcd_release =
          gpu_malloc<int>(8 * 16 * sizeof(int));
      (void)cudaMemset(global_runtime_config.precomp_iter_xcd_release,
                       0,
                       8 * 16 * sizeof(int));

      // Debug task counter — device memory, read via cudaMemcpy in debug loop
      global_runtime_config.precomp_dbg_tasks_done =
          gpu_malloc<unsigned long long>(sizeof(unsigned long long));
      (void)cudaMemset(global_runtime_config.precomp_dbg_tasks_done,
                       0,
                       sizeof(unsigned long long));

      // Debug worker state — off by default (the per-task relaxed stores cost
      // a little on the hot path), opt in with MPK_WORKER_STATE=1 to make the
      // host poll loop dump where every worker is parked when it hangs.
      //
      // Host-mapped, not device memory: on a hang the kernel never returns, so
      // the host must be able to read this while the kernel is still spinning.
      global_runtime_config.precomp_dbg_worker_state = nullptr;
      g_dbg_h_worker_state = nullptr;
      g_dbg_num_workers = num_workers;
      {
        char const *ws_env = getenv("MPK_WORKER_STATE");
        if (ws_env != nullptr && atoi(ws_env) != 0) {
          // Three thirds: [0, num_workers*4) is the per-worker phase state,
          // [num_workers*4, num_workers*8) is the barrier watch written by
          // MPK_WS_WAIT_BEGIN / MPK_WS_WAIT_TICK, and
          // [num_workers*8, num_workers*12) is the per-barrier aux written by
          // MPK_WS_WAIT_AUX.
          //
          // Coherent, NOT NonCoherent. This used to pass
          // hipHostMallocNonCoherent, which maps the pages MTYPE_NC on the
          // device side: a store lands in this XCD's L2 and stays there until
          // something evicts it or a system-scope release flushes it. Every
          // writer here (mpk_ws_phase, MPK_WS_WAIT_*) is a plain
          // __atomic_store_n(..., __ATOMIC_RELAXED) at default agent scope, so
          // nothing ever forces that flush. On a hang the kernel never returns
          // either, so there is no end-of-kernel writeback to save us. The
          // result is a debug buffer the host reads as all-zeros for an
          // arbitrary subset of ranks -- and since `in_task` is
          // `wphase >= 0`, an unwritten slot decodes as "IN TASK: phase=0",
          // i.e. the instrument manufactures a rank whose workers appear to be
          // wedged in a task they never entered. /tmp/probe3.log's "rank 3
          // never dispatched a task" was exactly that artifact.
          //
          // hipHostMallocCoherent maps the pages MTYPE_UC, so the relaxed
          // store goes to host memory directly. Slower per store, which is
          // fine: this whole allocation only exists under MPK_WORKER_STATE.
          // Three per-worker regions of 4 ints, then MPK_WS_SCHED_SLOTS
          // scheduler slots of 4 (see MPK_WS_SCHED_ENTERED).
          size_t ws_bytes =
              (size_t)(num_workers * 12 + MPK_WS_SCHED_SLOTS * 4) * sizeof(int);
          int *ws_host = nullptr;
          if (hipHostMalloc(reinterpret_cast<void **>(&ws_host),
                            ws_bytes,
                            hipHostMallocMapped | hipHostMallocCoherent) ==
                  hipSuccess &&
              ws_host != nullptr) {
            memset(ws_host, 0, ws_bytes);
            for (int w = 0; w < num_workers; w++) {
              ws_host[w * 4 + 3] = MPK_WS_UNWRITTEN;
            }
            for (int s = 0; s < MPK_WS_SCHED_SLOTS; s++) {
              ws_host[num_workers * 12 + s * 4 + 3] = MPK_WS_UNWRITTEN;
            }
            int *ws_dev = nullptr;
            if (hipHostGetDevicePointer(reinterpret_cast<void **>(&ws_dev),
                                        ws_host,
                                        0) == hipSuccess) {
              g_dbg_h_worker_state = ws_host;
              global_runtime_config.precomp_dbg_worker_state = ws_dev;
              fprintf(stderr,
                      "[HOST_DBG] worker-state tracing ON (%d workers)\n",
                      num_workers);
            } else {
              (void)hipHostFree(ws_host);
            }
          }
        }
      }

#ifdef MPK_NIL_TRIPWIRE
      // Tripwire for the nil-address memory fault. Backed by *pinned host*
      // memory, not device memory: the fault aborts the process, so anything
      // living only in VRAM (and device printf, and the GPU coredump) is
      // gone by the time we could read it. Writes from the kernel land in
      // host RAM over PCIe as they happen, so whatever the last worker wrote
      // before the fault is still there for the SIGABRT handler to print.
      (void)hipHostMalloc(reinterpret_cast<void **>(&g_tw_host),
                          MPK_TW_SLOTS * sizeof(unsigned long long),
                          hipHostMallocMapped | hipHostMallocNonCoherent);
      if (g_tw_host != nullptr) {
        for (int i = 0; i < MPK_TW_SLOTS; i++) {
          g_tw_host[i] = 0;
        }
        unsigned long long *tw_dev = nullptr;
        (void)hipHostGetDevicePointer(
            reinterpret_cast<void **>(&tw_dev), g_tw_host, 0);
        global_runtime_config.tripwire = tw_dev;
        g_tw_num_workers = num_workers;
        install_tripwire_handler();
        printf("[MPK] nil tripwire armed: host=%p dev=%p slots=%d\n",
               (void *)g_tw_host,
               (void *)tw_dev,
               MPK_TW_SLOTS);
      } else {
        global_runtime_config.tripwire = nullptr;
        printf("[MPK] nil tripwire: hipHostMalloc FAILED, not armed\n");
      }
      fflush(stdout);
#else
      global_runtime_config.tripwire = nullptr;
#endif

      // Clear host debug pointers (no longer host-mapped)
      g_dbg_h_iter_ready = nullptr;
      g_dbg_h_terminate = nullptr;
      g_dbg_h_tasks_done = nullptr;

      printf("[MPK] precomp_iter_ready dev=%p (device memory)\n",
             (void *)global_runtime_config.precomp_iter_ready);
      printf("[MPK] precomp_terminate dev=%p (device memory)\n",
             (void *)global_runtime_config.precomp_terminate);
      fflush(stdout);
    }

    // Build per-XCD per-event task thresholds for two-level event counting.
    // Iterate over template queues and count how many signals each XCD sends
    // to each event. Gang tasks: 1 signal per dispatched worker. Non-gang: 1
    // per task.
    std::vector<int> h_xcd_evt(NUM_XCDS_PC * n_events, 0);
    for (int xcd = 0; xcd < NUM_XCDS_PC; xcd++) {
      for (int rank = 0; rank < workers_per_xcd; rank++) {
        int flat = xcd * workers_per_xcd + rank;
        int qlen = h_tmpl_len[flat];
        for (int qi = 0; qi < qlen; qi++) {
          size_t pos = h_tmpl[flat * max_tpw + qi];
          if (pos >= (size_t)n_tasks) {
            continue;
          }
          EventId trigger_ev = all_tasks[pos].trigger_event;
          if (trigger_ev == EVENT_INVALID_ID) {
            continue;
          }
          int tev_idx = (int)(trigger_ev & 0xffffffff);
          if (tev_idx < 0 || tev_idx >= n_events) {
            continue;
          }
          // Each dispatched worker signals 1 (for both gang and non-gang)
          h_xcd_evt[xcd * n_events + tev_idx] += 1;
        }
      }
    }

    int nonzero_xcd_evt = 0;
    int mismatch_count = 0;
    for (int i = 0; i < NUM_XCDS_PC * n_events; i++) {
      if (h_xcd_evt[i] > 0) {
        nonzero_xcd_evt++;
      }
    }
    // Verify: for non-gang events, sum across XCDs should equal num_triggers.
    // For gang events, sum = dispatch_count but num_triggers = 8 (1 flush per
    // XCD).
    for (int ev = 0; ev < n_events; ev++) {
      int sum = 0;
      for (int xcd = 0; xcd < NUM_XCDS_PC; xcd++) {
        sum += h_xcd_evt[xcd * n_events + ev];
      }
      int nt = all_events[ev].num_triggers;
      // For gang events: sum = total dispatched workers, flush = 1 per XCD →
      // global = 8 = num_triggers For non-gang: sum should equal num_triggers
      // (each XCD flushes its threshold)
      bool is_gang_ev = false;
      for (auto &eg : event_groups) {
        if (eg.is_gang) {
          EventId tev = all_tasks[eg.first_task].trigger_event;
          if ((int)(tev & 0xffffffff) == ev) {
            is_gang_ev = true;
            break;
          }
        }
      }
      if (!is_gang_ev && sum != nt && nt > 0) {
        if (mismatch_count < 10) {
          printf("[MPK] WARN: event %d: sum_xcd_thresholds=%d != "
                 "num_triggers=%d (type=%d)\n",
                 ev,
                 sum,
                 nt,
                 all_events[ev].event_type);
        }
        mismatch_count++;
      }
    }
    if (mismatch_count > 0) {
      printf("[MPK] WARN: %d events have threshold/trigger mismatch!\n",
             mismatch_count);
    }
    printf("[MPK] Pre-filled mode: two-level event counting enabled (%d "
           "non-zero xcd_event thresholds)\n",
           nonzero_xcd_evt);
    fflush(stdout);

    // Allocate gang barrier counter (GPU memory, zeroed) and dispatch count
    // (GPU, pre-filled)
    {
      global_runtime_config.precomp_gang_barrier =
          gpu_malloc<unsigned long long>(n_events * sizeof(unsigned long long));
      (void)cudaMemset(global_runtime_config.precomp_gang_barrier,
                       0,
                       n_events * sizeof(unsigned long long));

      global_runtime_config.precomp_gang_dispatch_count =
          gpu_malloc<int>(n_events * sizeof(int));
      (void)cudaMemcpy(global_runtime_config.precomp_gang_dispatch_count,
                       h_gang_dispatch_count.data(),
                       n_events * sizeof(int),
                       cudaMemcpyHostToDevice);
      printf("[MPK] Allocated gang barriers: %d events\n", n_events);
      fflush(stdout);
    }

    // Allocate and upload xcd_event_num_tasks for two-level event counting
    global_runtime_config.xcd_event_num_tasks =
        gpu_malloc<int>(NUM_XCDS_PC * n_events * sizeof(int));
    (void)cudaMemcpy(global_runtime_config.xcd_event_num_tasks,
                     h_xcd_evt.data(),
                     NUM_XCDS_PC * n_events * sizeof(int),
                     cudaMemcpyHostToDevice);
    // Also need xcd_local_event_counters for two-level counting
    global_runtime_config.xcd_local_event_counters =
        gpu_malloc<EventCounter>(NUM_XCDS_PC * n_events * sizeof(EventCounter));
    (void)cudaMemset(global_runtime_config.xcd_local_event_counters,
                     0,
                     NUM_XCDS_PC * n_events * sizeof(EventCounter));
    g_dbg_num_xcds = NUM_XCDS_PC;

    // Set ml_workers_per_xcd (pointer tables uploaded earlier, before
    // compaction)
    if (global_runtime_config.ml_num_layers > 0) {
      global_runtime_config.ml_workers_per_xcd = workers_per_xcd;
    }
  }
#endif

  // Initialize worker queue last task id
  // Each worker now maintains a local and a remote worker queue
  global_runtime_config.worker_queue_last_ready_task_id =
      gpu_malloc<unsigned long long int>((num_workers * 2) *
                                         sizeof(unsigned long long int));
  // std::vector<unsigned long long int> host_worker_queue_last_task_id;
  // for (int i = 0; i < 2 * num_workers; i++) {
  //   host_worker_queue_last_task_id.push_back(0);
  // }
  // cudaMemcpy(global_runtime_config.worker_queue_last_ready_task_id,
  //            host_worker_queue_last_task_id.data(),
  //            (num_workers * 2) * sizeof(unsigned long long int),
  //            cudaMemcpyHostToDevice);
  //  Initialize scheduler queue last event id
  //  We maintain one extra scheduler queue for the global scheduler
  global_runtime_config.sched_queue_last_ready_event_id =
      gpu_malloc<unsigned long long int>((num_schedulers + 1) *
                                         sizeof(unsigned long long int));
  global_runtime_config.sched_queue_next_free_event_id =
      gpu_malloc<unsigned long long int>((num_schedulers + 1) *
                                         sizeof(unsigned long long int));

  // std::vector<unsigned long long int> host_sched_queue_last_event_id;
  // for (int i = 0; i < (num_schedulers + 1); i++) {
  //   host_sched_queue_last_event_id.push_back(0);
  // }
  // cudaMemcpy(global_runtime_config.sched_queue_last_ready_event_id,
  //            host_sched_queue_last_event_id.data(),
  //            (num_schedulers + 1) * sizeof(unsigned long long int),
  //            cudaMemcpyHostToDevice);
  // cudaMemcpy(global_runtime_config.sched_queue_next_free_event_id,
  //            host_sched_queue_last_event_id.data(),
  //            (num_schedulers + 1) * sizeof(unsigned long long int),
  //            cudaMemcpyHostToDevice);
  //  Initialize all event counters
  global_runtime_config.all_event_counters =
      gpu_malloc<EventCounter>(all_events.size() * sizeof(EventCounter));
#ifdef MPK_PRECOMPUTED_DISPATCH
  g_dbg_num_events = (int)all_events.size();
#endif
  global_runtime_config.all_event_num_triggers =
      gpu_malloc<int>(all_events.size() * sizeof(int));
  std::vector<int> host_all_event_counters;
  for (size_t i = 0; i < all_events.size(); i++) {
    host_all_event_counters.push_back(all_events.at(i).num_triggers);
  }
  (void)cudaMemcpy(global_runtime_config.all_event_num_triggers,
                   host_all_event_counters.data(),
                   all_events.size() * sizeof(int),
                   cudaMemcpyHostToDevice);
  // cudaMemset(global_runtime_config.all_event_counters,
  //            0,
  //            all_events.size() * sizeof(EventCounter));

#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300) ||              \
    defined(MIRAGE_BACKEND_USE_ROCM)
  // Initialize per-XCD local event counters for hierarchical polling
  // This reduces cross-XCD atomic contention by having leaders poll global
  // and other workers poll XCD-local counters
  constexpr int NUM_XCDS_INIT = 8; // MI300X has 8 XCDs
  global_runtime_config.num_xcds = NUM_XCDS_INIT;
  // xcd_local_event_counters may already be allocated by precomputed dispatch
  if (global_runtime_config.xcd_local_event_counters == nullptr) {
    global_runtime_config.xcd_local_event_counters = gpu_malloc<EventCounter>(
        NUM_XCDS_INIT * all_events.size() * sizeof(EventCounter));
    // Initialize to 0 (same initial state as global counters)
    (void)cudaMemset(global_runtime_config.xcd_local_event_counters,
                     0,
                     NUM_XCDS_INIT * all_events.size() * sizeof(EventCounter));
  }

  // Initialize XCD leader array (-1 means no leader yet)
  global_runtime_config.xcd_leader_worker =
      gpu_malloc<int>(NUM_XCDS_INIT * sizeof(int));
  std::vector<int> initial_leaders(NUM_XCDS_INIT, -1);
  (void)cudaMemcpy(global_runtime_config.xcd_leader_worker,
                   initial_leaders.data(),
                   NUM_XCDS_INIT * sizeof(int),
                   cudaMemcpyHostToDevice);

  // Worker XCD map: workers write their hardware XCD ID at kernel startup,
  // schedulers read it to build XCD-aligned worker lists
  global_runtime_config.worker_xcd_map =
      gpu_malloc<int>(num_workers * sizeof(int));
  (void)cudaMemset(
      global_runtime_config.worker_xcd_map, -1, num_workers * sizeof(int));
  global_runtime_config.worker_xcd_ready_count = gpu_malloc<int>(sizeof(int));
  (void)cudaMemset(
      global_runtime_config.worker_xcd_ready_count, 0, sizeof(int));
  // Per-XCD per-event task thresholds for two-level event counting
  // xcd_event_num_tasks may already be allocated+filled by precomputed dispatch
  if (global_runtime_config.xcd_event_num_tasks == nullptr) {
    global_runtime_config.xcd_event_num_tasks =
        gpu_malloc<int>(NUM_XCDS_INIT * all_events.size() * sizeof(int));
    (void)cudaMemset(global_runtime_config.xcd_event_num_tasks,
                     0,
                     NUM_XCDS_INIT * all_events.size() * sizeof(int));
  }
  // Combined kernel: XCD-based scheduler election
  global_runtime_config.xcd_scheduler_claimed =
      gpu_malloc<int>(NUM_XCDS_INIT * sizeof(int));
  (void)cudaMemset(global_runtime_config.xcd_scheduler_claimed,
                   -1,
                   NUM_XCDS_INIT * sizeof(int));
  global_runtime_config.dynamic_worker_id_counter =
      gpu_malloc<int>(sizeof(int));
  (void)cudaMemset(
      global_runtime_config.dynamic_worker_id_counter, 0, sizeof(int));
#endif

  //  Initialize all tasks
  global_runtime_config.all_tasks =
      gpu_malloc<TaskDesc>(all_tasks.size() * sizeof(TaskDesc));
  (void)cudaMemcpy(global_runtime_config.all_tasks,
                   all_tasks.data(),
                   all_tasks.size() * sizeof(TaskDesc),
                   cudaMemcpyHostToDevice);
  // Initialize all events
  global_runtime_config.num_events = (int)all_events.size();
  global_runtime_config.all_events =
      gpu_malloc<EventDesc>(all_events.size() * sizeof(EventDesc));
  (void)cudaMemcpy(global_runtime_config.all_events,
                   all_events.data(),
                   all_events.size() * sizeof(EventDesc),
                   cudaMemcpyHostToDevice);
  // Initialize worker queues
  {
    std::vector<TaskId *> host_worker_queues;
    for (int i = 0; i < (num_workers * 2); i++) {
      TaskId *worker_queue = gpu_malloc<TaskId>(
          global_runtime_config.per_worker_queue_len * sizeof(TaskId));
      host_worker_queues.push_back(worker_queue);
    }
    global_runtime_config.worker_queues =
        gpu_malloc<TaskId *>((num_workers * 2) * sizeof(TaskId *));
    (void)cudaMemcpy(global_runtime_config.worker_queues,
                     host_worker_queues.data(),
                     (num_workers * 2) * sizeof(TaskId *),
                     cudaMemcpyHostToDevice);
  }
  // Initialize scheduler queues
  {
    std::vector<EventId *> host_sched_queues;
    for (int i = 0; i < (num_schedulers + 1); i++) {
      EventId *sched_queue = gpu_malloc<EventId>(
          global_runtime_config.per_sched_queue_len * sizeof(EventId));
      host_sched_queues.push_back(sched_queue);
    }
    global_runtime_config.sched_queues =
        gpu_malloc<EventId *>((num_schedulers + 1) * sizeof(EventId *));
    (void)cudaMemcpy(global_runtime_config.sched_queues,
                     host_sched_queues.data(),
                     (num_schedulers + 1) * sizeof(EventId *),
                     cudaMemcpyHostToDevice);
  }
  // Initialize first tasks
  {
    global_runtime_config.first_tasks =
        gpu_malloc<TaskId>(first_tasks.size() * sizeof(TaskId));
    (void)cudaMemcpy(global_runtime_config.first_tasks,
                     first_tasks.data(),
                     first_tasks.size() * sizeof(TaskId),
                     cudaMemcpyHostToDevice);
  }

  // Set configuration for kernels
  (void)cudaFuncSetAttribute(worker_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MAX_DYNAMIC_SHARED_MEMORY_SIZE);
  (void)cudaFuncSetAttribute(
      scheduler_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 1024);
  (void)cudaFuncSetAttribute(persistent_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MAX_DYNAMIC_SHARED_MEMORY_SIZE);
  // Create worker and scheduler streams
  (void)cudaStreamCreateWithFlags(&global_runtime_config.worker_stream,
                                  cudaStreamNonBlocking);
  (void)cudaStreamCreateWithFlags(&global_runtime_config.scheduler_stream,
                                  cudaStreamNonBlocking);
  // Create events
  (void)cudaEventCreateWithFlags(&global_runtime_config.prepare_done_event,
                                 cudaEventDisableTiming);
  (void)cudaEventCreateWithFlags(&global_runtime_config.worker_done_event,
                                 cudaEventDisableTiming);
  (void)cudaEventCreateWithFlags(&global_runtime_config.scheduler_done_event,
                                 cudaEventDisableTiming);

  init_request_resources();
#if defined(USE_NVSHMEM) || defined(USE_ROCSHMEM)
  // Add a global barrier for all init_kernel to complete
  mpk_shmem_barrier_all();
#endif
}

// Entry point for C/C++
// TODO: change launch config
extern "C" void launch_persistent_kernel(cudaStream_t default_stream) {
  fprintf(stderr, "[HOST_DBG] launch_persistent_kernel ENTER\n");
  // Prepare next persistent kernel by resetting queue pointers
  {
    int end_of_task_graph_event_pos = global_runtime_config.num_events - 1;
    prepare_kernel<<<dim3(global_runtime_config.num_workers, 1, 1),
                     dim3(128, 1, 1),
                     0 /*smem*/,
                     default_stream>>>(global_runtime_config,
                                       end_of_task_graph_event_pos);
    (void)cudaEventRecord(global_runtime_config.prepare_done_event,
                          default_stream);
#if defined(USE_NVSHMEM) || defined(USE_ROCSHMEM)
    mpk_shmem_barrier_all();
#endif
  }
  int num_schedulers = global_runtime_config.num_local_schedulers +
                       global_runtime_config.num_remote_schedulers;
  if (global_runtime_config.split_worker_scheduler) {
    (void)cudaStreamWaitEvent(global_runtime_config.worker_stream,
                              global_runtime_config.prepare_done_event,
                              0);
    (void)cudaStreamWaitEvent(global_runtime_config.scheduler_stream,
                              global_runtime_config.prepare_done_event,
                              0);

    // The split kernel does not support NVSHMEM because
    // nvshmemx_collective_launch launches kernels sequentially, which blocks
    // the interaction between the worker kernel and the scheduler kernel
    worker_kernel<<<dim3(global_runtime_config.num_workers, 1, 1),
                    dim3(WORKER_NUM_THREADS, 1, 1),
                    MAX_DYNAMIC_SHARED_MEMORY_SIZE /*smem*/,
                    global_runtime_config.worker_stream>>>(
        global_runtime_config);
    scheduler_kernel<<<dim3(global_runtime_config.num_local_schedulers, 1, 1),
                       dim3(128, 1, 1),
                       0 /*smem*/,
                       global_runtime_config.scheduler_stream>>>(
        global_runtime_config);

    (void)cudaEventRecord(global_runtime_config.worker_done_event,
                          global_runtime_config.worker_stream);
    (void)cudaEventRecord(global_runtime_config.scheduler_done_event,
                          global_runtime_config.scheduler_stream);

    (void)cudaStreamWaitEvent(
        default_stream, global_runtime_config.worker_done_event, 0);
    (void)cudaStreamWaitEvent(
        default_stream, global_runtime_config.scheduler_done_event, 0);

#ifdef MPK_PRECOMPUTED_DISPATCH
    // Debug: poll device memory via async memcpy on a separate stream
    // Opt-in, and the default matters. The block below is described further
    // down as running "only under MPK_WORKER_STATE debug builds"; it never
    // did. Its only guard is #ifdef MPK_PRECOMPUTED_DISPATCH, which GLM sets
    // unconditionally (every EP threshold is a function of task_layer_idx, so
    // demo.py asserts on it), so this ran on every GLM run including the
    // timed ones.
    //
    // It is not an instrumentation-only path. It spawns a detached thread
    // that issues hipMemcpyAsync D2H plus hipStreamSynchronize on a private
    // stream against a GPU whose 256 CUs are entirely occupied by the
    // persistent kernel. A small D2H is a blit kernel on ROCm, and a blit
    // kernel that cannot find a free CU is exactly the case the hardware
    // scheduler resolves by preempting the resident kernel.
    char const *dbg_poll_env = getenv("MPK_HOST_DBG_POLL");
    bool const dbg_poll_on = dbg_poll_env && atoi(dbg_poll_env) == 1;
    if (dbg_poll_on) {
      fprintf(stderr, "[HOST_DBG] Starting poll loop...\n");
    }
    if (dbg_poll_on && global_runtime_config.precomp_iter_ready &&
        global_runtime_config.precomp_terminate) {
      hipStream_t dbg_stream;
      (void)hipStreamCreate(&dbg_stream);
      // The device-memory poll runs on its own detached thread.
      //
      // This is not premature generality: when the kernel hangs, the ROCm
      // runtime wedges with it, and hipMemcpyAsync/hipStreamSynchronize below
      // block forever *even on a private stream*. Running them inline is what
      // made this instrumentation useless on the one case it exists for -- the
      // hung 32k run printed "Starting poll loop..." and then not one line
      // more, because it never got past the first sync to reach the dump.
      //
      // The worker-state and event-counter buffers are pinned host memory, so
      // the dumps below need no runtime call and keep printing regardless.
      // dev_polls staying at 0 is itself the signal that HIP is stuck.
      static std::atomic<unsigned long long> dbg_a_ir{0};
      static std::atomic<int> dbg_a_term{0};
      static std::atomic<unsigned long long> dbg_a_td{0};
      static std::atomic<unsigned long long> dbg_a_polls{0};
      static std::atomic<bool> dbg_a_stop{false};
      // Detached, and every object it touches has static storage duration --
      // a thread blocked in HIP outlives this scope and must not dangle.
      std::thread([dbg_stream]() {
        while (!dbg_a_stop.load(std::memory_order_relaxed)) {
          unsigned long long ir = 0;
          int term = 0;
          unsigned long long td = 0;
          (void)hipMemcpyAsync(&ir,
                               global_runtime_config.precomp_iter_ready,
                               sizeof(unsigned long long),
                               hipMemcpyDeviceToHost,
                               dbg_stream);
          (void)hipMemcpyAsync(&term,
                               global_runtime_config.precomp_terminate,
                               sizeof(int),
                               hipMemcpyDeviceToHost,
                               dbg_stream);
          if (global_runtime_config.precomp_dbg_tasks_done) {
            (void)hipMemcpyAsync(&td,
                                 global_runtime_config.precomp_dbg_tasks_done,
                                 sizeof(unsigned long long),
                                 hipMemcpyDeviceToHost,
                                 dbg_stream);
          }
          (void)hipStreamSynchronize(dbg_stream);
          dbg_a_ir.store(ir, std::memory_order_relaxed);
          dbg_a_term.store(term, std::memory_order_relaxed);
          dbg_a_td.store(td, std::memory_order_relaxed);
          dbg_a_polls.fetch_add(1, std::memory_order_relaxed);
          if (term) {
            break;
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
      }).detach();
      // A 32k run takes minutes, so the original 30-tick (30s) bound stopped
      // watching long before the interesting window: a hang at iteration
      // ~20000 produced no dump at all, because the loop had already exited
      // into the blocking stream sync below. Poll until the kernel actually
      // reports terminate (the `if (term) break` at the bottom), bounded only
      // by a generous ceiling so a wedged run still exits the loop and lets
      // the enclosing `timeout` reap it.
      int const dbg_max_ticks = 900;
      for (int dbg_i = 0; dbg_i < dbg_max_ticks; dbg_i++) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        unsigned long long ir = dbg_a_ir.load(std::memory_order_relaxed);
        int term = dbg_a_term.load(std::memory_order_relaxed);
        unsigned long long td = dbg_a_td.load(std::memory_order_relaxed);
        unsigned long long polls = dbg_a_polls.load(std::memory_order_relaxed);
        // dev_polls is normally 0 for the whole run -- a healthy 32k run shows
        // 0 at every tick. hipMemcpyAsync does not complete while the
        // persistent kernel occupies the GPU, even on a private stream, so
        // iter_ready/terminate/tasks_done stay 0 and mean nothing until the
        // kernel exits. Do NOT read dev_polls=0 as a hang.
        //
        // The real liveness signal is the per-worker `done` counter in the
        // dump below: it lives in pinned host memory and advances even while
        // HIP is blocked. Frozen `done` across ticks = actually stuck.
        if (dbg_i % 30 == 0 || term) {
          fprintf(
              stderr,
              "[HOST_DBG] t=%ds iter_ready=%llu terminate=%d "
              "tasks_done=%llu dev_polls=%llu (0 is normal; watch `done`)\n",
              dbg_i + 1,
              ir,
              term,
              td,
              polls);
        }
        // Dump per-worker state + summary
        if (g_dbg_h_worker_state && dbg_i >= 2) {
          int phase_hist[4] = {};
          // Fused-layer phase histogram over ALL workers, not just the 32
          // printed individually. A stall where 239 workers sit on one barrier
          // and a single straggler sits elsewhere is invisible in a 32-worker
          // sample, and that straggler is the one holding the barrier.
          // Indexed by (phase-50000)/100; also track min/max layer to expose
          // cross-worker layer skew, which is the real deadlock signature.
          std::map<int, int> fl_phase_hist;
          std::map<int, int> fl_px_hist; // key = phase*10 + xcd
          int fl_layer_min = 1 << 30, fl_layer_max = -1;
          int dep_hist[256] = {};
          int phase_gang_stuck = 0;
          int phase_gang_entry = 0;
          // Workers whose phase slot still holds the pre-launch sentinel, i.e.
          // that have never executed a single MPK_WS write. In a healthy run
          // this drops to 0 within the first tick; if it stays at
          // g_dbg_num_workers, this rank's megakernel is not running at all,
          // which is a completely different failure from any barrier stall.
          int ws_unwritten = 0;
          // Resident but never dispatched a task (see MPK_WS_ENTERED).
          int ws_entered = 0;
          int moe_sub[8] = {};  // 0=w13(2000) 1=w13sig(2001) 2=w2entry(3000)
                                // 3=w2poll(3001) 4=w2polldone(3002)
                                // 5=w2quant(3003) 6=w2lds(3004) 7=w2mfma(3005)
          int moe_epilogue = 0; // 3006
          // Liveness first, so the verbose decision below can use it.
          //
          // Comparing each worker's `done` against the previous tick is the
          // only reliable progress test here (see the dev_polls note above).
          // Tracked per-worker, not as an aggregate, so one frozen straggler
          // among 239 live workers cannot hide.
          static std::vector<int> prev_done;
          static int stall_ticks = 0;
          int frozen = 0;
          bool have_prev = ((int)prev_done.size() == g_dbg_num_workers);
          {
            std::vector<int> cur_done(g_dbg_num_workers);
            for (int w = 0; w < g_dbg_num_workers; w++) {
              cur_done[w] = __atomic_load_n(&g_dbg_h_worker_state[w * 4 + 2],
                                            __ATOMIC_RELAXED);
            }
            if (have_prev) {
              for (int w = 0; w < g_dbg_num_workers; w++) {
                if (cur_done[w] == prev_done[w]) {
                  frozen++;
                }
              }
              stall_ticks = (frozen == g_dbg_num_workers) ? stall_ticks + 1 : 0;
            }
            prev_done.swap(cur_done);
          }
          // Now that the loop runs for the whole kernel rather than 30s, the
          // per-worker lines would be ~32 lines/sec of healthy-run noise and
          // would bury the failure. Print them only while a stall is building
          // or on a periodic checkpoint.
          bool verbose = (stall_ticks >= 1) || (dbg_i % 30 == 0);
          for (int w = 0; w < g_dbg_num_workers; w++) {
            int *ws = g_dbg_h_worker_state + w * 4;
            int tpos = __atomic_load_n(&ws[0], __ATOMIC_RELAXED);
            int dep = __atomic_load_n(&ws[1], __ATOMIC_RELAXED);
            int done = __atomic_load_n(&ws[2], __ATOMIC_RELAXED);
            int phase = __atomic_load_n(&ws[3], __ATOMIC_RELAXED);
            if (phase == MPK_WS_UNWRITTEN) {
              ws_unwritten++;
              continue;
            }
            if (phase == MPK_WS_ENTERED) {
              ws_entered++;
              continue;
            }
            int sc = phase / 100000;
            int moe_xcd = (phase / 10000) % 10;
            int moe_tile = phase % 10000;
            if (verbose && (w < 32 || sc == 3001)) {
              // 40000+ml is the multi-layer marker written in the ml loop.
              // Print it as a layer number rather than a raw phase, since
              // "which layer" is the whole point of that encoding.
              if (phase >= 50000000 && phase < 51000000) {
                // 50000000 + xcd*100000 + phase*1000 + layer%1000.
                int fx = (phase - 50000000) / 100000;
                int fp = ((phase - 50000000) / 1000) % 100;
                int fl = (phase - 50000000) % 1000;
                char const *pn = "?";
                switch (fp) {
                  case 11:
                    pn = "EP-fold-begin";
                    break;
                  case 12:
                    pn = "EP-arrival-done";
                    break;
                  case 13:
                    pn = "EP-self-wait-done";
                    break;
                  case 14:
                    pn = "EP-peer-wait-done";
                    break;
                  case 20:
                    pn = "P2-qkv-epoch";
                    break;
                  // GLM's MLA attention half, one per phase. gpt-oss reuses
                  // 20/30/... for its own coarser split; these are the
                  // intra-attention breakdown the MLA monolith needs.
                  case 21:
                    pn = "MLA-P1-resolve-norm-qkva";
                    break;
                  case 22:
                    pn = "MLA-P2-qkva-qb-barrier";
                    break;
                  case 23:
                    pn = "MLA-P3-qb-kvappend";
                    break;
                  case 24:
                    pn = "MLA-P4-qb-decode-barrier";
                    break;
                  case 25:
                    pn = "MLA-P5-decode";
                    break;
                  case 26:
                    pn = "MLA-P6-decode-merge-barrier";
                    break;
                  case 27:
                    pn = "MLA-P7-merge";
                    break;
                  // GLM's MoE half, phases 9-15 of the monolith.
                  case 71:
                    pn = "MOE-P9-oproj";
                    break;
                  case 72:
                    pn = "MOE-P10-oproj-router-barrier";
                    break;
                  case 73:
                    pn = "MOE-P11-norm-router-topk";
                    break;
                  case 74:
                    pn = "MOE-P12-routing-wait";
                    break;
                  case 76:
                    pn = "MOE-P13-w13";
                    break;
                  case 77:
                    pn = "MOE-P14-w13-w2-barrier";
                    break;
                  case 78:
                    pn = "MOE-P15-w2";
                    break;
                  case 30:
                    pn = "P3-attn-chunk";
                    break;
                  case 40:
                    pn = "P4-chunk-barrier";
                    break;
                  case 50:
                    pn = "P5-merge";
                    break;
                  case 60:
                    pn = "P6-attn-xcd-barrier";
                    break;
                  case 61:
                    pn = "P6a-arrived";
                    break;
                  case 62:
                    pn = "P6b-flag-poll";
                    break;
                  case 63:
                    pn = "P6c-flag-cleared";
                    break;
                  case 64:
                    pn = "P6d-post-inv";
                    break;
                  case 70:
                    pn = "P7-oproj";
                    break;
                  case 75:
                    pn = "P7b-routing-poll";
                    break;
                  case 80:
                    pn = "P8-moe";
                    break;
                  case 90:
                    pn = "P9-layer-done";
                    break;
                }
                fprintf(stderr,
                        "  w%d: pos=%d dep=%d done=%d xcd=%d epoch=%d %s\n",
                        w,
                        tpos,
                        dep,
                        done,
                        fx,
                        fl,
                        pn);
              } else if (phase >= 40000 && phase < 40000 + 1024) {
                fprintf(stderr,
                        "  w%d: pos=%d dep=%d done=%d layer=%d\n",
                        w,
                        tpos,
                        dep,
                        done,
                        phase - 40000);
              } else {
                fprintf(stderr,
                        "  w%d: pos=%d dep=%d done=%d phase=%d (sc=%d xcd=%d "
                        "tile=%d)\n",
                        w,
                        tpos,
                        dep,
                        done,
                        phase,
                        sc,
                        moe_xcd,
                        moe_tile);
              }
            }
            // Exclude the 40000+ml multi-layer marker: it means "executing
            // layer ml", which is normal progress, not a stuck gang tile.
            if (phase >= 50000000 && phase < 51000000) {
              int fp = ((phase - 50000000) / 1000) % 100;
              int fx = (phase - 50000000) / 100000;
              int fl = (phase - 50000000) % 1000;
              fl_phase_hist[fp]++;
              // Per-(phase,XCD) tally: a barrier stall shows as one XCD short
              // of its expected arrivals while the others are complete.
              fl_px_hist[fp * 10 + fx]++;
              if (fl < fl_layer_min) {
                fl_layer_min = fl;
              }
              if (fl > fl_layer_max) {
                fl_layer_max = fl;
              }
            }
            if (phase >= 1200 && phase < 300100000 &&
                !(phase >= 40000 && phase < 40000 + 1024) &&
                !(phase >= 50000000 && phase < 51000000)) {
              phase_gang_stuck++;
            }
            if (sc == 2000) {
              moe_sub[0]++;
            } else if (sc == 2001) {
              moe_sub[1]++;
            } else if (sc == 3000) {
              moe_sub[2]++;
            } else if (sc == 3001) {
              moe_sub[3]++;
            } else if (sc == 3002) {
              moe_sub[4]++;
            } else if (sc == 3003) {
              moe_sub[5]++;
            } else if (sc == 3004) {
              moe_sub[6]++;
            } else if (sc == 3005) {
              moe_sub[7]++;
            } else if (sc == 3006) {
              moe_epilogue++;
            }
            if (phase == 11) {
              phase_gang_entry++;
            }
            if (phase < 0) {
              phase_hist[0]++;
              if (dep >= 0 && dep < 256) {
                dep_hist[dep]++;
              }
            } else if (phase == 10) {
              phase_hist[1]++;
            } else if (phase == 20) {
              phase_hist[2]++;
            } else if (phase == 30) {
              phase_hist[3]++;
            }
          }
          if (have_prev && (verbose || stall_ticks >= 1)) {
            fprintf(stderr,
                    "  progress: %d/%d workers advanced since last tick%s\n",
                    g_dbg_num_workers - frozen,
                    g_dbg_num_workers,
                    stall_ticks >= 2 ? "   *** STALLED: no worker has advanced "
                                       "for 3+ ticks ***"
                                     : "");
          }
          if (ws_unwritten > 0) {
            fprintf(stderr,
                    "  ws-unwritten: %d/%d workers have never written a phase"
                    "%s\n",
                    ws_unwritten,
                    g_dbg_num_workers,
                    ws_unwritten == g_dbg_num_workers
                        ? "   *** this rank's megakernel is not running ***"
                        : "");
          }
          // Residency and bootstrap. Printed whenever anything is still short
          // of its first task, because that window is the whole failure: the
          // schedulers block until every worker has registered its XCD, so one
          // non-resident block leaves all 240 idle and all 8 schedulers spinning
          // in the bootstrap wait, with not a single phase written anywhere.
          if (ws_entered > 0 || ws_unwritten > 0) {
            int const nsched = global_runtime_config.num_local_schedulers +
                               global_runtime_config.num_remote_schedulers;
            fprintf(stderr,
                    "  ws-residency: entered-no-task=%d unwritten=%d of %d"
                    "%s\n",
                    ws_entered,
                    ws_unwritten,
                    g_dbg_num_workers,
                    (ws_unwritten > 0 && ws_entered > 0)
                        ? "   *** some worker blocks are NOT RESIDENT; the "
                          "schedulers can never leave the XCD-ready wait ***"
                        : "");
            for (int s = 0; s < nsched && s < MPK_WS_SCHED_SLOTS; s++) {
              int *sl = g_dbg_h_worker_state + g_dbg_num_workers * 12 + s * 4;
              int sblk = __atomic_load_n(&sl[0], __ATOMIC_RELAXED);
              int sv1 = __atomic_load_n(&sl[1], __ATOMIC_RELAXED);
              int sv2 = __atomic_load_n(&sl[2], __ATOMIC_RELAXED);
              int sph = __atomic_load_n(&sl[3], __ATOMIC_RELAXED);
              char const *sn = "?";
              if (sph == MPK_WS_UNWRITTEN) {
                sn = "NOT-RESIDENT";
              } else if (sph == MPK_WS_SCHED_ENTERED) {
                sn = "in-xcd-ready-wait";
              } else if (sph == MPK_WS_SCHED_XCD_READY) {
                sn = "dispatching";
              }
              fprintf(stderr,
                      "    sched%d: %s block=%d %s=%d xcd/workers=%d\n",
                      s,
                      sn,
                      sblk,
                      sph == MPK_WS_SCHED_XCD_READY ? "iters" : "ready_count",
                      sv1,
                      sv2);
            }
          }
          // Barrier watch. Only meaningful once the run is actually wedged --
          // in a healthy run these slots hold whatever barrier each worker
          // last cleared, which is noise. On a stall it is the whole answer:
          // observed vs expected separates "the producer never arrived"
          // (observed short by N) from "this waiter expects an epoch the
          // producer will never reach" (observed >= expected, i.e. the poll
          // should have cleared and the load is reading a stale line).
          if (stall_ticks >= 2) {
            // key = barrier_id, value = (count, min observed, expected,
            // max spins) aggregated over workers still spinning.
            std::map<int, std::array<long long, 5>> bw;
            // The watch slots are last-write-wins and are never cleared on
            // exit, so a worker that finished its layer long ago still shows
            // whatever mark it passed on the way out. Reading them without
            // cross-checking the live phase is how the first capture produced
            // 56 "blockers" that were all idle at the scheduler (phase == -1)
            // while the one worker actually inside the layer went unprinted.
            // Only a worker whose phase says it is still executing can be
            // holding anything, so gate every watch read on that.
            for (int w = 0; w < g_dbg_num_workers; w++) {
              int wphase = __atomic_load_n(&g_dbg_h_worker_state[w * 4 + 3],
                                           __ATOMIC_RELAXED);
              bool in_task = (wphase >= 0);
              int *b = g_dbg_h_worker_state + g_dbg_num_workers * 4 + w * 4;
              int bid = __atomic_load_n(&b[0], __ATOMIC_RELAXED);
              int obs = __atomic_load_n(&b[1], __ATOMIC_RELAXED);
              int exp_ = __atomic_load_n(&b[2], __ATOMIC_RELAXED);
              int spins = __atomic_load_n(&b[3], __ATOMIC_RELAXED);
              if (!in_task) {
                continue; // idle at the scheduler; its watch slot is stale
              }
              // Whatever this worker's watch slot says, it is genuinely still
              // inside the task -- print the live phase too, since that is the
              // ground truth the watch slot has to be read against.
              {
                char const *pn = "?";
                int fx = -1, fp = -1, fl = -1;
                if (wphase >= 50000000 && wphase < 51000000) {
                  fx = (wphase - 50000000) / 100000;
                  fp = ((wphase - 50000000) / 1000) % 100;
                  fl = (wphase - 50000000) % 1000;
                  switch (fp) {
                    case 20:
                      pn = "P2-qkv-epoch";
                      break;
                    // GLM's MLA monolith markers: 11-14 the EP fold, 21-27 the
                    // attention half, 71-78 the MoE half. Kept in step with the
                    // per-worker switch above -- the same marker has to decode the
                    // same way in the histogram or the two disagree.
                    case 11:
                      pn = "EP-fold-begin";
                      break;
                    case 12:
                      pn = "EP-arrival-done";
                      break;
                    case 13:
                      pn = "EP-self-wait-done";
                      break;
                    case 14:
                      pn = "EP-peer-wait-done";
                      break;
                    case 21:
                      pn = "MLA-P1-resolve-norm-qkva";
                      break;
                    case 22:
                      pn = "MLA-P2-qkva-qb-barrier";
                      break;
                    case 23:
                      pn = "MLA-P3-qb-kvappend";
                      break;
                    case 24:
                      pn = "MLA-P4-qb-decode-barrier";
                      break;
                    case 25:
                      pn = "MLA-P5-decode";
                      break;
                    case 26:
                      pn = "MLA-P6-decode-merge-barrier";
                      break;
                    case 27:
                      pn = "MLA-P7-merge";
                      break;
                    case 71:
                      pn = "MOE-P9-oproj";
                      break;
                    case 72:
                      pn = "MOE-P10-oproj-router-barrier";
                      break;
                    case 73:
                      pn = "MOE-P11-norm-router-topk";
                      break;
                    case 74:
                      pn = "MOE-P12-routing-wait";
                      break;
                    case 76:
                      pn = "MOE-P13-w13";
                      break;
                    case 77:
                      pn = "MOE-P14-w13-w2-barrier";
                      break;
                    case 78:
                      pn = "MOE-P15-w2";
                      break;
                    case 30:
                      pn = "P3-attn-chunk";
                      break;
                    case 40:
                      pn = "P4-chunk-barrier";
                      break;
                    case 50:
                      pn = "P5-merge";
                      break;
                    case 60:
                      pn = "P6-attn-xcd-barrier";
                      break;
                    case 61:
                      pn = "P6a-arrived";
                      break;
                    case 62:
                      pn = "P6b-flag-poll";
                      break;
                    case 63:
                      pn = "P6c-flag-cleared";
                      break;
                    case 64:
                      pn = "P6d-post-inv";
                      break;
                    case 70:
                      pn = "P7-oproj";
                      break;
                    case 75:
                      pn = "P7b-routing-poll";
                      break;
                    case 80:
                      pn = "P8-moe";
                      break;
                    case 90:
                      pn = "P9-layer-done";
                      break;
                    // Phase 9's sub-stages. A hang here is a collective hang,
                    // and which stage it is decides where to look: 9a is the
                    // GPU-wide MoE arrival, 9b/9c the fold + peer store, 9d
                    // the self/peer signal wait, 9e the last-layer exit.
                    case 91:
                      pn = "P9a-moe-arrive";
                      break;
                    case 92:
                      pn = "P9bc-fold-release";
                      break;
                    case 93:
                      pn = "P9bc-folding";
                      break;
                    case 94:
                      pn = "P9d-sig-wait";
                      break;
                    case 95:
                      pn = "P9e-exit";
                      break;
                  }
                }
                int *a = g_dbg_h_worker_state + g_dbg_num_workers * 8 + w * 4;
                int a0 = __atomic_load_n(&a[0], __ATOMIC_RELAXED);
                int a1 = __atomic_load_n(&a[1], __ATOMIC_RELAXED);
                int a2 = __atomic_load_n(&a[2], __ATOMIC_RELAXED);
                int a3 = __atomic_load_n(&a[3], __ATOMIC_RELAXED);
                // aux is printed unconditionally. It used to be read here and
                // then only consumed by the fp == 80 branch below, which meant
                // the one field that distinguishes "one straggling peer" from
                // "no peer ever landed" -- bid 900's aux0, the `remaining`
                // bitmask -- was collected on the device, copied to the host,
                // and thrown away. Four extra ints on a line that is only
                // emitted under MPK_WORKER_STATE costs nothing.
                fprintf(stderr,
                        "    >>> w%d IN TASK: phase=%d (%s xcd=%d epoch=%d) "
                        "watch[bid=%d obs=%d exp=%d spins=%d] "
                        "aux[%d %d %d %d]\n",
                        w,
                        wphase,
                        pn,
                        fx,
                        fl,
                        bid,
                        obs,
                        exp_,
                        spins,
                        a0,
                        a1,
                        a2,
                        a3);
                // Per-wave exit mask for the MoE W13->W2 poll. That poll is
                // thread-divergent by construction (each thread tests the
                // release flag on its own, no __syncthreads), so waves of one
                // block leave it independently. tid 0 leaving is NOT the block
                // leaving: gx_1 showed w30 reporting a straight-line mark past
                // the barrier (8303, inside the quant's trailing
                // __syncthreads) while the block was still short by 2. A mask
                // below 0xf names the waves that never saw the release.
                if (fp == 80) {
                  int smask = a2 & 0xf;
                  fprintf(stderr,
                          "        quant sync mask=0x%x (%s)\n",
                          smask,
                          smask == 0xf
                              ? "all 4 waves reached the quant __syncthreads"
                              : "waves missing -- note NSUBBLOCKS(96) < "
                                "blockDim(256), so waves 1..3 do fewer or "
                                "zero loop trips and legitimately arrive "
                                "early; only a mask frozen across dumps is "
                                "evidence");
                  int mask = a3 & 0xf;
                  fprintf(stderr,
                          "        wave exit mask=0x%x (%s)\n",
                          mask,
                          mask == 0xf
                              ? "all 4 waves cleared the W13->W2 poll"
                              : (mask == 0
                                     ? "no wave cleared it"
                                     : "<== BLOCK SPLIT ACROSS THE BARRIER: "
                                       "some waves cleared, others still "
                                       "spinning -- the cleared waves are "
                                       "parked in the next __syncthreads"));
                }
                // spins == 0 means the poll cleared on its very first load
                // without ever ticking, so obs/exp and the aux below are
                // leftovers from an earlier layer and must not be read as this
                // worker's current state. An in-task worker with spins == 0 is
                // stuck somewhere *past* its last barrier, not at one -- which
                // is exactly what the fx_8 capture showed after the cache-line
                // split, and what the aux misreported as a lost fan-out.
                if (spins == 0) {
                  fprintf(stderr,
                          "        (watch slot STALE: last barrier cleared on "
                          "its first load -- this worker is stuck past it, "
                          "read the mark, not obs/exp)\n");
                }
                // a2/a3 are the per-wave masks now, so the slot-spread
                // decoding below no longer applies. Keep the arrival counter,
                // which still lives in a0.
                if (spins > 0 && bid >= 800 && bid < 900) {
                  fprintf(stderr,
                          "        MoE arrivals=%d (arrivals%%46=%d) "
                          "expert_id=%d\n",
                          a0,
                          a0 % 46,
                          a1);
                }
                // 760 is the attention -> o_proj cross-XCD barrier (phase 8 of
                // the MLA monolith). a0 is the raw global arrival counter,
                // monotonic at `arrivals` per fused layer, and a1 is the XCD
                // whose release flag this worker polls. The counter is what
                // decides the case: at or past arrivals * expected every
                // worker arrived and the last one fanned the release out, so a
                // waiter still short is not missing a producer -- it is not
                // seeing a store that was made. Short of it, a producer really
                // is missing and the count says how many.
                if (spins > 0 && bid == 760) {
                  // arrivals is not carried in the dump; derive it from the
                  // counter itself, which is an exact multiple of it at every
                  // quiescent point. 240 = tiles_per_xcd(30) * 8 for GLM.
                  int const arrivals = 240;
                  int want_ctr = arrivals * exp_;
                  fprintf(stderr,
                          "        attn-xcd barrier: xcd=%d obs=%d want=%d "
                          "counter=%d (need %d, short %d) %s\n",
                          a1,
                          obs,
                          exp_,
                          a0,
                          want_ctr,
                          want_ctr - a0,
                          a0 >= want_ctr
                              ? "<== all arrived, release fired: THIS FLAG IS "
                                "NOT VISIBLE TO THIS WORKER"
                              : "<== producers missing, blame the arrivals");
                }
                // 899 is the EP self-signal wait, 900 the peer wait -- the two
                // halves of one rendezvous. Decoded here rather than left as
                // raw aux ints because which half is stuck decides whether the
                // fault is local (fold arrivals) or remote (publish/transport).
                //
                // 899: a0 is the raw ep_fold_done counter, run-monotonic with
                // exactly 8 arrivals per layer. a0 % 8 != 0 while the folding
                // work-groups are parked means the arrivals themselves are
                // being lost, before any peer is involved.
                if (spins > 0 && bid == 899) {
                  fprintf(stderr,
                          "        EP self-wait: ep_fold_done=%d (%%8=%d, "
                          "layers=%d) sig=%d want=%d %s\n",
                          a0,
                          a0 % 8,
                          a0 / 8,
                          a1,
                          exp_,
                          a0 % 8 != 0
                              ? "<== local fold arrivals LOST"
                              : "<== arrivals complete, publish did not land");
                }
                // 900: a0 is the bitmask of peers whose line is still short.
                // One bit is a straggler and the fault is on that rank; all
                // bits is a transport or address fault and the fault is here.
                // a1 packs two views of the LOWEST missing peer's signal: low
                // 16 bits the copy that peer pushed into this rank's line
                // (what the wait actually tests), high 16 bits the same value
                // read out of that peer's OWN memory through its peer delta.
                // pushed == exp_ - 1 alone only says "that peer is one layer
                // behind"; it is the two together that name the fault.
                //
                //   home > pushed  : the peer really did advance and its store
                //                    into this line is not landing -- and the
                //                    re-publish retry in the wait means it did
                //                    not land on ~15k attempts, so it is the
                //                    mapping or the link, not a dropped store.
                //   home == pushed : the peer never advanced its signal, and
                //                    its own higher threshold came from a
                //                    counter that drifted between ranks. A
                //                    software bug; fix the counter.
                if (spins > 0 && bid == 900) {
                  int nmiss = __builtin_popcount((unsigned)a0);
                  int pushed = a1 & 0xffff;
                  int home = (a1 >> 16) & 0xffff;
                  fprintf(stderr,
                          "        EP peer-wait: missing=0x%02x (%d peers) "
                          "pushed=%d home=%d want=%d %s%s\n",
                          a0,
                          nmiss,
                          pushed,
                          home,
                          exp_,
                          nmiss == 1 ? "<== single straggler, blame that rank"
                                     : "<== broad, blame transport/address",
                          home == pushed ? " [counter drift]"
                                         : " [PUSH NOT LANDING]");
                }
                if (false) {
                  // a0 = raw arrival counter. If a0 is an exact multiple of
                  // W13_TILES every producer arrived and the release fired.
                  // a2 packs (slots_at_or_past_expected)*1e6 + (max-min)
                  // across the 8 per-XCD release slots of this expert; a3 is
                  // the min. All 8 are written by one producer in one loop, so
                  // spread != 0 means the fan-out writes are being lost.
                  int n_ok = a2 / 1000000;
                  int spread = a2 % 1000000;
                  fprintf(stderr,
                          "        MoE aux: arrivals=%d (arrivals%%46=%d) "
                          "expert_id=%d slots_ok=%d/8 spread=%d min=%d %s\n",
                          a0,
                          a0 % 46,
                          a1,
                          n_ok,
                          spread,
                          a3,
                          spread != 0
                              ? "<== SLOTS DISAGREE: release fan-out lost"
                              : (a0 % 46 == 0
                                     ? "<== all arrived, all slots equal: "
                                       "release VALUE is stale"
                                     : "<== arrivals missing"));
                }
              }
              // spins == -1 is a straight-line mark (MPK_WS_MARK), not a poll:
              // the worker is running, and the code says where. Print those
              // separately -- on a total stall they are the blockers, since a
              // worker that is not in any poll is the one everyone waits on.
              if (spins < 0) {
                int mcode = (-spins) / 100000;
                int maux = (-spins) % 100000;
                char const *mn = "?";
                switch (mcode) {
                  case 8000:
                    mn = "MoE tile loop";
                    break;
                  case 8100:
                    mn = "MoE exit: past tile range";
                    break;
                  case 8101:
                    mn = "MoE exit: W13 padding tile";
                    break;
                  case 8102:
                    mn = "MoE exit: token out of batch";
                    break;
                  case 8103:
                    mn = "MoE exit: token not routed";
                    break;
                  case 8200:
                    mn = "MoE W13 compute";
                    break;
                  case 8201:
                    mn = "MoE W13 done, at barrier";
                    break;
                  case 8300:
                    mn = "MoE W2 entry";
                    break;
                  case 8302:
                    mn = "MoE W2: cleared W13->W2 barrier";
                    break;
                  case 8303:
                    mn = "MoE W2: FP8 quant";
                    break;
                  case 8304:
                    mn = "MoE W2: drain HBM loads";
                    break;
                  case 8305:
                    mn = "MoE W2: MFMA loop";
                    break;
                  case 8306:
                    mn = "MoE W2: epilogue";
                    break;
                }
                fprintf(stderr,
                        "    w%d NOT POLLING: mark %d (%s) tile=%d"
                        "   <== running, not blocked; everyone else waits on "
                        "this\n",
                        w,
                        mcode,
                        mn,
                        maux);
                continue;
              }
              // spins==0 means this worker cleared its last barrier without
              // ever ticking; it is not waiting, so it is not the blocker.
              if (spins == 0) {
                continue;
              }
              auto it = bw.find(bid);
              if (it == bw.end()) {
                bw[bid] = {1, obs, exp_, spins, w};
              } else {
                it->second[0]++;
                if (obs < it->second[1]) {
                  it->second[1] = obs;
                }
                if (spins > it->second[3]) {
                  it->second[3] = spins;
                  it->second[4] = w;
                }
              }
            }
            if (!bw.empty()) {
              fprintf(stderr,
                      "  *** BARRIER WATCH (workers still spinning) "
                      "***\n");
              for (auto const &kv : bw) {
                char const *bn = "?";
                int bid = kv.first;
                if (bid >= 800 && bid < 900) {
                  bn = "MoE-W13->W2";
                } else if (bid == 20) {
                  bn = "P2-qkv-epoch";
                } else if (bid == 60) {
                  bn = "P6-attn-xcd";
                } else if (bid == 760) {
                  bn = "attn->o_proj xcd flag";
                } else if (bid == 761) {
                  bn = "layer-entry xcd flag";
                } else if (bid == 762) {
                  bn = "qkv_a->q_b xcd flag";
                } else if (bid == 763) {
                  bn = "q_b->decode xcd flag";
                } else if (bid == 764) {
                  bn = "decode->merge xcd flag";
                } else if (bid == 765) {
                  bn = "routing-ready xcd flag";
                } else if (bid == 766) {
                  bn = "GLM MoE W13->W2 xcd flag";
                } else if (bid == 75) {
                  bn = "P7b-routing";
                }
                fprintf(stderr,
                        "    barrier %d (%s%s%d): %lld workers spinning, "
                        "observed=%lld expected=%lld short_by=%lld "
                        "max_spins=%lld (w%lld)%s\n",
                        bid,
                        bn,
                        bid >= 800 && bid < 900 ? " expert " : "",
                        bid >= 800 && bid < 900 ? bid - 800 : 0,
                        kv.second[0],
                        kv.second[1],
                        kv.second[2],
                        kv.second[2] - kv.second[1],
                        kv.second[3],
                        kv.second[4],
                        kv.second[1] >= kv.second[2]
                            ? "   <== STALE READ: observed >= expected, the "
                              "poll should have cleared"
                            : "");
              }
            }
          }
          if (verbose) {
            if (!fl_phase_hist.empty()) {
              fprintf(stderr,
                      "  fused-layer phases (all %d workers):",
                      g_dbg_num_workers);
              for (auto const &kv : fl_phase_hist) {
                char const *pn = "?";
                switch (kv.first) {
                  case 20:
                    pn = "P2-qkv-epoch";
                    break;
                  // GLM's MLA monolith markers: 11-14 the EP fold, 21-27 the
                  // attention half, 71-78 the MoE half. Kept in step with the
                  // per-worker switch above -- the same marker has to decode the
                  // same way in the histogram or the two disagree.
                  case 11:
                    pn = "EP-fold-begin";
                    break;
                  case 12:
                    pn = "EP-arrival-done";
                    break;
                  case 13:
                    pn = "EP-self-wait-done";
                    break;
                  case 14:
                    pn = "EP-peer-wait-done";
                    break;
                  case 21:
                    pn = "MLA-P1-resolve-norm-qkva";
                    break;
                  case 22:
                    pn = "MLA-P2-qkva-qb-barrier";
                    break;
                  case 23:
                    pn = "MLA-P3-qb-kvappend";
                    break;
                  case 24:
                    pn = "MLA-P4-qb-decode-barrier";
                    break;
                  case 25:
                    pn = "MLA-P5-decode";
                    break;
                  case 26:
                    pn = "MLA-P6-decode-merge-barrier";
                    break;
                  case 27:
                    pn = "MLA-P7-merge";
                    break;
                  case 71:
                    pn = "MOE-P9-oproj";
                    break;
                  case 72:
                    pn = "MOE-P10-oproj-router-barrier";
                    break;
                  case 73:
                    pn = "MOE-P11-norm-router-topk";
                    break;
                  case 74:
                    pn = "MOE-P12-routing-wait";
                    break;
                  case 76:
                    pn = "MOE-P13-w13";
                    break;
                  case 77:
                    pn = "MOE-P14-w13-w2-barrier";
                    break;
                  case 78:
                    pn = "MOE-P15-w2";
                    break;
                  case 30:
                    pn = "P3-attn-chunk";
                    break;
                  case 40:
                    pn = "P4-chunk-barrier";
                    break;
                  case 50:
                    pn = "P5-merge";
                    break;
                  case 60:
                    pn = "P6-attn-xcd-barrier";
                    break;
                  case 61:
                    pn = "P6a-arrived";
                    break;
                  case 62:
                    pn = "P6b-flag-poll";
                    break;
                  case 63:
                    pn = "P6c-flag-cleared";
                    break;
                  case 64:
                    pn = "P6d-post-inv";
                    break;
                  case 70:
                    pn = "P7-oproj";
                    break;
                  case 75:
                    pn = "P7b-routing-poll";
                    break;
                  case 80:
                    pn = "P8-moe";
                    break;
                  case 90:
                    pn = "P9-layer-done";
                    break;
                }
                fprintf(stderr, " %s=%d", pn, kv.second);
              }
              fprintf(stderr,
                      "  epoch=[%d..%d]%s\n",
                      fl_layer_min,
                      fl_layer_max,
                      fl_layer_min != fl_layer_max
                          ? "  *** EPOCH SKEW: workers on different layers ***"
                          : "");
              // Per-XCD breakdown. Each XCD runs 30 workers, so a phase that
              // shows fewer than 30 on some XCD is missing arrivals there --
              // that XCD is the one holding a per-XCD barrier.
              for (auto const &kv : fl_phase_hist) {
                fprintf(stderr, "    phase %d per-xcd:", kv.first);
                for (int x = 0; x < 8; x++) {
                  auto it = fl_px_hist.find(kv.first * 10 + x);
                  fprintf(stderr,
                          " x%d=%d",
                          x,
                          it == fl_px_hist.end() ? 0 : it->second);
                }
                fprintf(stderr, "\n");
              }
            }
            fprintf(
                stderr,
                "  phases: spinning=%d dep_done=%d signaling=%d sig_done=%d "
                "gang_tile=%d gang_entry=%d",
                phase_hist[0],
                phase_hist[1],
                phase_hist[2],
                phase_hist[3],
                phase_gang_stuck,
                phase_gang_entry);
            fprintf(stderr,
                    "\n  moe: w13=%d w13sig=%d w2entry=%d w2poll=%d w2done=%d "
                    "w2quant=%d w2lds=%d w2mfma=%d w2epi=%d",
                    moe_sub[0],
                    moe_sub[1],
                    moe_sub[2],
                    moe_sub[3],
                    moe_sub[4],
                    moe_sub[5],
                    moe_sub[6],
                    moe_sub[7],
                    moe_epilogue);
            if (phase_hist[0] > 0) {
              fprintf(stderr, " | stuck_deps:");
              for (int d = 0; d < 256; d++) {
                if (dep_hist[d] > 0) {
                  fprintf(stderr, " ev%d=%d", d, dep_hist[d]);
                }
              }
            }
            fprintf(stderr, "\n");
          } // verbose
        }
        // Dump event counters to diagnose which events haven't fired
        if (g_dbg_h_event_counters && dbg_i == 5) {
          fprintf(stderr,
                  "  === Event counters: last 20 events + event 158 ===\n");
          // Show events 140-158 (the tail where workers are stuck)
          int start_ev = g_dbg_num_events > 20 ? g_dbg_num_events - 20 : 0;
          for (int ev = start_ev; ev < g_dbg_num_events; ev++) {
            EventCounter gc =
                __atomic_load_n(&g_dbg_h_event_counters[ev], __ATOMIC_RELAXED);
            fprintf(
                stderr, "    ev%d: global=%llu", ev, (unsigned long long)gc);
            if (g_dbg_h_xcd_local_counters && g_dbg_h_xcd_event_thresholds) {
              fprintf(stderr, " |");
              for (int x = 0; x < g_dbg_num_xcds; x++) {
                int slot = x * g_dbg_num_events + ev;
                EventCounter lc = __atomic_load_n(
                    &g_dbg_h_xcd_local_counters[slot], __ATOMIC_RELAXED);
                int thresh = g_dbg_h_xcd_event_thresholds[slot];
                fprintf(
                    stderr, " x%d=%llu/%d", x, (unsigned long long)lc, thresh);
              }
            }
            fprintf(stderr, "\n");
          }
          // Summary: count events with global=0
          int unfired = 0;
          for (int ev = 1; ev < g_dbg_num_events; ev++) {
            EventCounter gc =
                __atomic_load_n(&g_dbg_h_event_counters[ev], __ATOMIC_RELAXED);
            if (gc == 0) {
              unfired++;
            }
          }
          fprintf(stderr,
                  "  Events with global=0: %d / %d\n",
                  unfired,
                  g_dbg_num_events);
          // Also show unique done values to understand queue lengths
          if (g_dbg_h_worker_state) {
            int min_done = 99999, max_done = 0;
            for (int w = 0; w < g_dbg_num_workers; w++) {
              int done = __atomic_load_n(&g_dbg_h_worker_state[w * 4 + 2],
                                         __ATOMIC_RELAXED);
              if (done < min_done) {
                min_done = done;
              }
              if (done > max_done) {
                max_done = done;
              }
            }
            fprintf(
                stderr, "  done range: min=%d max=%d\n", min_done, max_done);
          }
        }
        if (term) {
          break;
        }
      }
      dbg_a_stop.store(true, std::memory_order_relaxed);
      // Deliberately leaking dbg_stream: the poll thread is detached and may
      // still be blocked inside HIP on it. Destroying the stream out from
      // under it would turn a diagnosable hang into a use-after-free. This
      // path only runs under MPK_HOST_DBG_POLL=1.
    }
#endif
    (void)cudaStreamSynchronize(global_runtime_config.worker_stream);
    (void)cudaStreamSynchronize(global_runtime_config.scheduler_stream);
  } else {
    int num_sms_to_use = global_runtime_config.num_workers + num_schedulers;
#ifdef USE_NVSHMEM
    void *args[] = {&global_runtime_config};
    nvshmemx_collective_launch((void const *)persistent_kernel,
                               dim3(num_sms_to_use, 1, 1),
                               dim3(SINGLE_KERNEL_NUM_THREADS, 1, 1),
                               args,
                               MAX_DYNAMIC_SHARED_MEMORY_SIZE /*sharedmem*/,
                               0 /*stream*/);
#else
    persistent_kernel<<<dim3(num_sms_to_use, 1, 1),
                        dim3(SINGLE_KERNEL_NUM_THREADS, 1, 1),
                        MAX_DYNAMIC_SHARED_MEMORY_SIZE /*smem*/>>>(
        global_runtime_config);
#endif
    (void)cudaDeviceSynchronize();
  }
}

extern "C" void finalize_persistent_kernel() {
  gpu_free(global_runtime_config.worker_queue_last_ready_task_id);
  gpu_free(global_runtime_config.sched_queue_last_ready_event_id);
  gpu_free(global_runtime_config.sched_queue_next_free_event_id);
  gpu_free(global_runtime_config.all_event_counters);
  gpu_free(global_runtime_config.all_event_num_triggers);
  gpu_free(global_runtime_config.all_tasks);
  gpu_free(global_runtime_config.all_events);
#if defined(MODE_OFFLINE) || defined(MODE_ONLINE)
  gpu_free(global_runtime_config.next_request_id);
  gpu_free(global_runtime_config.page_queue);
  gpu_free(global_runtime_config.page_queue_head);
  gpu_free(global_runtime_config.page_queue_tail);
#endif
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300) ||              \
    defined(MIRAGE_BACKEND_USE_ROCM)
  // Free per-XCD local event counters
  if (global_runtime_config.xcd_local_event_counters != nullptr) {
    gpu_free(global_runtime_config.xcd_local_event_counters);
  }
  if (global_runtime_config.xcd_leader_worker != nullptr) {
    gpu_free(global_runtime_config.xcd_leader_worker);
  }
  if (global_runtime_config.worker_xcd_map != nullptr) {
    gpu_free(global_runtime_config.worker_xcd_map);
  }
  if (global_runtime_config.worker_xcd_ready_count != nullptr) {
    gpu_free(global_runtime_config.worker_xcd_ready_count);
  }
  if (global_runtime_config.xcd_event_num_tasks != nullptr) {
    gpu_free(global_runtime_config.xcd_event_num_tasks);
  }
  if (global_runtime_config.xcd_scheduler_claimed != nullptr) {
    gpu_free(global_runtime_config.xcd_scheduler_claimed);
  }
  if (global_runtime_config.dynamic_worker_id_counter != nullptr) {
    gpu_free(global_runtime_config.dynamic_worker_id_counter);
  }
#endif
  int num_workers = global_runtime_config.num_workers;
  std::vector<TaskId *> host_worker_queues(num_workers * 2);
  (void)cudaMemcpy(host_worker_queues.data(),
                   global_runtime_config.worker_queues,
                   (num_workers * 2) * sizeof(TaskId *),
                   cudaMemcpyDeviceToHost);
  for (int i = 0; i < 2 * num_workers; i++) {
    gpu_free(host_worker_queues[i]);
  }
  gpu_free(global_runtime_config.worker_queues);
  int num_schedulers = global_runtime_config.num_local_schedulers +
                       global_runtime_config.num_remote_schedulers;
  std::vector<EventId *> host_sched_queues(num_schedulers + 1);
  (void)cudaMemcpy(host_sched_queues.data(),
                   global_runtime_config.sched_queues,
                   (num_schedulers + 1) * sizeof(EventId *),
                   cudaMemcpyDeviceToHost);
  for (int i = 0; i < num_schedulers + 1; i++) {
    gpu_free(host_sched_queues[i]);
  }
  gpu_free(global_runtime_config.sched_queues);
  gpu_free(global_runtime_config.first_tasks);
#if defined(USE_NVSHMEM) || defined(USE_ROCSHMEM)
  mpk_shmem_barrier_all();
  mpk_shmem_finalize();
#endif
  // Free worker and scheduler streams
  (void)cudaEventDestroy(global_runtime_config.prepare_done_event);
  (void)cudaEventDestroy(global_runtime_config.worker_done_event);
  (void)cudaEventDestroy(global_runtime_config.scheduler_done_event);
  (void)cudaStreamDestroy(global_runtime_config.worker_stream);
  (void)cudaStreamDestroy(global_runtime_config.scheduler_stream);
}
