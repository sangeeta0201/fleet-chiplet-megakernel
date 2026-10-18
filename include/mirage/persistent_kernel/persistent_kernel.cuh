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

#ifdef MPK_HOST_STOP
// Every 16th iteration: each read is a round trip to host memory on the iteration boundary.
#define MPK_HOST_STOP_AT(c, it)                                                  \
  ((((unsigned long long)(it)) & 15ull) == 0ull && (c).stop_host != nullptr &&  \
   __hip_atomic_load((c).stop_host, __ATOMIC_RELAXED,                           \
                     __HIP_MEMORY_SCOPE_SYSTEM) != 0)
#else
#define MPK_HOST_STOP_AT(c, it) false
#endif
#ifdef MPK_PPROBE
#include "pprobe.h"
#endif
#ifdef MPK_RPROBE
#include "rprobe.h"
#endif
#ifdef MPK_NC_PROBE
#include "ncprobe.h"
#endif

#if defined(MPK_AID_LOCAL) || defined(MPK_AID_SPLIT_FLAGS) ||                 \
    defined(MPK_LM_WEIGHT_AID)
#include "aid_local.h"
#endif
#ifdef USE_NVSHMEM
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#endif
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
#define SUBPHASE_SLOTS 6
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

#ifdef MPK_QKV_SUBSTAMPS
// Per-worker boundary times inside the Phase 1 QKV kernel for the last
// armed layer: entry, slab pass consumed, RMSNorm, FP8 quant, MFMA, end.
__device__ unsigned long long g_qkvsub[1024 * 8];
#define MPK_QKVSUB(k)                                                  \
  do {                                                                 \
    if (threadIdx.x == 0) {                                            \
      asm volatile("" ::: "memory");                                    \
      g_qkvsub[blockIdx.x * 8 + (k)] = __builtin_amdgcn_s_memrealtime(); \
      asm volatile("" ::: "memory");                                    \
    }                                                                  \
  } while (0)
#else
#define MPK_QKVSUB(k) \
  do {             \
  } while (0)
#endif

#ifdef MPK_ATTSUB
__device__ unsigned long long g_attsub[1024 * 8];
#define MPK_ATSTAMP(k)                                                  \
  do {                                                                  \
    if (threadIdx.x == 0) {                                             \
      asm volatile("" ::: "memory");                                     \
      g_attsub[blockIdx.x * 8 + (k)] = __builtin_amdgcn_s_memrealtime(); \
      asm volatile("" ::: "memory");                                     \
    }                                                                   \
  } while (0)
#define MPK_ATVALUE(k, v)                                               \
  do {                                                                  \
    if (threadIdx.x == 0) {                                             \
      g_attsub[blockIdx.x * 8 + (k)] = (unsigned long long)(v);          \
    }                                                                   \
  } while (0)
#else
#define MPK_ATSTAMP(k) \
  do {             \
  } while (0)
#define MPK_ATVALUE(k, v) \
  do {                \
  } while (0)
#endif
#ifdef MPK_ATTSCAN
__device__ unsigned long long g_attscan[1024 * 8];
#define MPK_ASSTAMP(k)                                                   \
  do {                                                                   \
    if (threadIdx.x == 0) {                                              \
      asm volatile("" ::: "memory");                                      \
      g_attscan[blockIdx.x * 8 + (k)] = __builtin_amdgcn_s_memrealtime(); \
      asm volatile("" ::: "memory");                                      \
    }                                                                    \
  } while (0)
#define MPK_ASVALUE(k, v)                                                \
  do {                                                                   \
    if (threadIdx.x == 0) {                                              \
      g_attscan[blockIdx.x * 8 + (k)] = (unsigned long long)(v);          \
    }                                                                    \
  } while (0)
#else
#define MPK_ASSTAMP(k) \
  do {             \
  } while (0)
#define MPK_ASVALUE(k, v) \
  do {                \
  } while (0)
#endif
#ifdef MPK_W2SUB
__device__ unsigned long long g_w2sub[1024 * 8];
#define MPK_W2STAMP(k)                                                 \
  do {                                                                 \
    if (threadIdx.x == 0) {                                            \
      asm volatile("" ::: "memory");                                    \
      g_w2sub[blockIdx.x * 8 + (k)] = __builtin_amdgcn_s_memrealtime(); \
      asm volatile("" ::: "memory");                                    \
    }                                                                  \
  } while (0)
#else
#define MPK_W2STAMP(k) \
  do {             \
  } while (0)
#endif

// Dynamic LDS for each scheduler block. A worker takes all but ~4.5 KB of a
// CU's 160 KB, so a scheduler asking for more than what is left can never be
// dispatched next to a worker: it takes the grid's spare CU instead. At 0 the
// dispatcher is free to put a spinning scheduler on a worker's CU (7 of 8
// did so in one SPX/NPS2 launch), and that worker runs slower for the whole
// launch.
#ifndef MPK_SCHED_LDS
#define MPK_SCHED_LDS 0
#endif

#ifdef MPK_HWID_MAP
// Where each worker and scheduler block landed: XCC_ID << 32 | HW_ID, taken
// once at kernel entry. A scheduler that shares a CU with a worker steals
// that worker's issue slots for the whole launch.
__device__ unsigned long long g_hwid_w[1024];
__device__ unsigned long long g_hwid_s[64];
__device__ __forceinline__ unsigned long long mpk_hwid_now() {
  unsigned hw, xcc;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s"(hw));
  MPK_GET_XCC_ID16(xcc);
  return ((unsigned long long)xcc << 32) | hw;
}
#endif
#ifdef MPK_RESID_MAP
// Residency map in pinned host memory: slot b = worker block b, slot 1024 + b
// = scheduler block b; 0x80000000 | XCC_ID << 24 | HW_ID[23:0].
__device__ unsigned *g_resid_host = nullptr;
__device__ __forceinline__ void mpk_resid_mark(int slot) {
  unsigned hw, xcc;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s"(hw));
  MPK_GET_XCC_ID16(xcc);
  unsigned *p = g_resid_host;
  if (p != nullptr) {
    __hip_atomic_store(&p[slot],
                       0x80000000u | ((xcc & 0xfu) << 24) | (hw & 0xffffffu),
                       __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_SYSTEM);
  }
}
__device__ __forceinline__ void mpk_resid_put(int slot, unsigned v) {
  unsigned *p = g_resid_host;
  if (p != nullptr) {
    __hip_atomic_store(&p[slot], 0x80000000u | v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}
#define MPK_RESID_PUT(slot, v) mpk_resid_put((slot), (v))
// MPK_GATE_RESCUE_V1: raw counters in the host residency buffer, slots 1088 and up.
__device__ __forceinline__ void mpk_resid_add(int slot, unsigned v) {
  unsigned *p = g_resid_host;
  if (p != nullptr) {
    __hip_atomic_fetch_add(&p[slot], v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}
// CU identity = XCC << 8 | HW_ID[15:8] (CU_ID, SH_ID, SE_ID).
static void mpk_resid_report(unsigned const *hm, int nw, int ns, int gen, int t) {
  int wst = 0, sst = 0, wx[16] = {0}, sx[16] = {0};
  std::map<unsigned, int> cu_w, cu_s;
  char miss_w[512] = {0}, miss_s[256] = {0};
  size_t lw = 0, ls = 0;
  for (int b = 0; b < nw && b < 1024; b++) {
    unsigned const v = __atomic_load_n(&hm[b], __ATOMIC_RELAXED);
    if (v & 0x80000000u) {
      unsigned const x = (v >> 24) & 0xfu;
      wst++;
      wx[x]++;
      cu_w[(x << 8) | ((v >> 8) & 0xffu)]++;
    } else if (lw < sizeof(miss_w) - 8) {
      lw += snprintf(miss_w + lw, sizeof(miss_w) - lw, " %d", b);
    }
  }
  for (int b = 0; b < ns && b < 64; b++) {
    unsigned const v = __atomic_load_n(&hm[1024 + b], __ATOMIC_RELAXED);
    if (v & 0x80000000u) {
      unsigned const x = (v >> 24) & 0xfu;
      sst++;
      sx[x]++;
      cu_s[(x << 8) | ((v >> 8) & 0xffu)]++;
    } else if (ls < sizeof(miss_s) - 8) {
      ls += snprintf(miss_s + ls, sizeof(miss_s) - ls, " %d", b);
    }
  }
  int shared = 0, dupw = 0, dups = 0;
  for (auto const &kv : cu_s) {
    shared += cu_w.count(kv.first) ? 1 : 0;
    dups += kv.second > 1 ? 1 : 0;
  }
  for (auto const &kv : cu_w) {
    dupw += kv.second > 1 ? 1 : 0;
  }
  char when[32];
  if (t < 0) {
    snprintf(when, sizeof when, "done");
  } else {
    snprintf(when, sizeof when, "NOT DONE at %ds", t);
  }
  fprintf(stderr,
          "[RESID] launch %d %s: workers %d/%d schedulers %d/%d; CUs with "
          "worker+sched %d, 2+ workers %d, 2+ scheds %d; distinct CUs %zu\n",
          gen, when, wst, nw, sst, ns, shared, dupw, dups,
          cu_w.size() + cu_s.size() - (size_t)shared);
  fprintf(stderr,
          "[RESID]   per XCC workers %d %d %d %d %d %d %d %d | schedulers %d %d "
          "%d %d %d %d %d %d\n",
          wx[0], wx[1], wx[2], wx[3], wx[4], wx[5], wx[6], wx[7], sx[0], sx[1],
          sx[2], sx[3], sx[4], sx[5], sx[6], sx[7]);
  {  // MPK_GATE_RESCUE_V1
    unsigned r[8], li[8], tot = 0;
    for (int x = 0; x < 8; x++) {
      r[x] = __atomic_load_n(&hm[1088 + x], __ATOMIC_RELAXED);
      li[x] = __atomic_load_n(&hm[1104 + x], __ATOMIC_RELAXED) & 0x7fffffffu;
      tot += r[x];
    }
    if (tot != 0) {
      fprintf(stderr,
              "[RESID]   gate rescues %u: per XCC %u %u %u %u %u %u %u %u | last iter %u %u %u %u %u "
              "%u %u %u\n",
              tot, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], li[0], li[1], li[2], li[3], li[4],
              li[5], li[6], li[7]);
    }
  }
  if (wst < nw) {
    fprintf(stderr, "[RESID]   workers not resident:%s\n", miss_w);
  }
  if (sst < ns) {
    fprintf(stderr, "[RESID]   schedulers not resident:%s\n", miss_s);
  }
  if (t >= 0) {
    for (auto const &kv : cu_s) {
      fprintf(stderr, "[RESID]   sched CU xcc=%u hw15_8=0x%02x n=%d%s\n",
              kv.first >> 8, kv.first & 0xffu, kv.second,
              cu_w.count(kv.first) ? " SHARES A WORKER CU" : "");
    }
  }
}
#endif

#ifdef MPK_ILSUB
// Inter-layer path timestamps for the transition MPK_ILSUB_L0 -> L0 + 1;
// see ilsub_patch.py for the stamp map.
#ifndef MPK_ILSUB_L0
#define MPK_ILSUB_L0 20
#endif
__device__ unsigned long long g_ilsub[1024 * 32];
#ifdef MPK_ILPER
__device__ unsigned long long g_ilper[1024 * 128];
#endif
__shared__ int s_ilsub_ml;
#define MPK_ILSTAMP(k, cond)                                            \
  do {                                                                \
    if (threadIdx.x == 0 && (cond)) {                                 \
      asm volatile("" ::: "memory");                                   \
      g_ilsub[blockIdx.x * 32 + (k)] = __builtin_amdgcn_s_memrealtime(); \
      asm volatile("" ::: "memory");                                   \
    }                                                                 \
  } while (0)
#else
#define MPK_ILSTAMP(k, cond) \
  do {                  \
  } while (0)
#endif

#ifdef MPK_IL_FAST
// Per-iteration copies of values the inter-layer window used to load from
// global memory every layer. See the ml-loop prologue.
// One 16-byte-aligned block whose size is a multiple of 16: the dynamic LDS
// region starts right after static LDS, and an 8-byte-aligned end measured
// the whole kernel ~2x slower (misaligned 16-byte LDS traffic).
struct alignas(16) MpkIlFastLds {
  int qo_len[4];
  int variant[64];
  void *qkvw[64];
};
static_assert(sizeof(MpkIlFastLds) % 16 == 0, "keep static LDS 16-byte sized");
__shared__ MpkIlFastLds s_ilf;
#define s_ml_qo_len (s_ilf.qo_len[0])
#define s_ml_variant (s_ilf.variant)
#define s_ml_qkvw (s_ilf.qkvw)
__device__ __forceinline__ int mpk_ml_qo_len() { return s_ml_qo_len; }
#endif

#ifdef MPK_DRAIN_STATS
// Phase 9 barrier segment attribution: how much of the layer-boundary wait is
// store drain (s_waitcnt vmcnt(0)) vs rendezvous vs spinning on other XCDs.
__device__ unsigned long long g_drain_n;
__device__ unsigned long long g_drain_sum;
__device__ unsigned long long g_sync_sum;
__device__ unsigned long long g_arrive_sum;
__device__ unsigned long long g_spin_sum;
// Per-XCD spin, to tell a fixed slow die from a rotating one.
__device__ unsigned long long g_spin_xcd[8];
__device__ unsigned long long g_n_xcd[8];
// How often each XCD is the last one to arrive (i.e. is the critical path).
__device__ unsigned long long g_last_xcd[8];
// Spin of each XCD's *last local* arriver: that worker has no intra-XCD wait
// left, so its spin is purely the inter-XCD component.
__device__ unsigned long long g_spin_lastlocal;
__device__ unsigned long long g_n_lastlocal;
// Spin of the worker that itself published the global release -- the last
// local arriver on the last global XCD. By construction it waits on nobody:
// no intra-XCD straggler (it is the last local) and no other XCD (its own
// arrival was the eighth). Whatever it still spins is the irreducible
// store-to-poll-observe latency of the release line, so it splits the spin
// total into propagation (this) and skew (the rest).
__device__ unsigned long long g_spin_lastglobal;
__device__ unsigned long long g_n_lastglobal;
// The spin segment split at the poll loop: `fanpf` is the release fan-out
// plus the next-layer QKV prefetch issue that sits in front of the gate,
// `poll` is the gate loop proper.
__device__ unsigned long long g_fanpf_sum;
__device__ unsigned long long g_poll_sum;
__device__ unsigned long long g_poll_lastglobal;
// The inter-layer prologue (ml pointer-table copy + its two __syncthreads),
// which runs after the previous layer's gate cleared and before any of the
// next layer's work.
__device__ unsigned long long g_mlprologue_sum;
__device__ unsigned long long g_mlprologue_n;
// The table read alone, without the two __syncthreads that publish it.
__device__ unsigned long long g_mlcopy_sum;
__device__ unsigned long long g_mlcopy_n;
// Spin by xcd_rank: the intra-XCD load-imbalance map. Rank determines role
// (QKV/attn vs MoE tiles), so this says which workers idle and which straggle.
__device__ unsigned long long g_spin_rank[64];
__device__ unsigned long long g_n_rank[64];
#endif

#ifdef MPK_INTERLAYER_SPLIT
// Splits the slot-11 -> slot-0 span, which the phase table reports as a single
// ~4.3 us/layer block and which MPK_DRAIN_STATS only accounts ~1.7 us of.
// The rest is inside the fused-layer function *before* its slot-0 mark, so no
// existing counter can see it. Three segments, all tid 0, all per layer:
//   fnpre  - function entry to just before the layer's L2 invalidate
//   binv   - the buffer_inv itself
//   post   - invalidate to the slot-0 mark
__device__ unsigned long long g_il_fnpre_sum;
__device__ unsigned long long g_il_binv_sum;
__device__ unsigned long long g_il_post_sum;
__device__ unsigned long long g_il_n;
#endif

#ifdef MPK_PHASE_SLOTS
// Per-worker phase timestamps: one dword pair per slot per layer, reduced
// into per-slot-pair spans.
//
// Why not reuse MPK_DEVICE_TIMING: that path prints one [FUSED_PHASE] line per
// layer per iteration (~262k printfs), which inflates the iteration to ~56 ms
// and dumps the skew into whichever phase happens to hold the barrier -- the
// numbers it produces are of the instrumented kernel, not of the real one.
// This writes one dword pair per slot per layer with no printf on the hot
// path.
//
// Slots:
//   0  layer entry (also accumulates the inter-layer span; see below)
//   1  end of QKV GEMM                    (Phase 1 exit)
//   2  end of QKV epoch barrier
//   3  end of attention chunk compute
//   4  attn_release wait begin
//   5  attn_release wait end
//   6  end of O-proj + router
//   7  end of TopK routing wait
//   8  end of MoE tiles (pre-Phase 9)
//   9  Phase 9 arrival done
//   10 Phase 9 gate cleared
//   11 layer exit
#define MPK_PHASE_SLOT_COUNT 13
// Per-worker stride padded to a full 128 B line. The arrays are indexed
// by STRIDE and iterated by COUNT, so only the spacing changes.
#define MPK_PHASE_SLOT_STRIDE 16
#define MPK_PHASE_PAD_INT 32   /* 32 * 4 B = 128 B */
#define MPK_PHASE_PAD_U64 16   /* 16 * 8 B = 128 B */
#define MPK_PHASE_MAX_WORKERS 256
__device__ unsigned long long
    g_phase_ts[MPK_PHASE_MAX_WORKERS * MPK_PHASE_SLOT_STRIDE];
// Accumulated per-slot-pair spans, so the reduction is over every layer of
// every decode iteration rather than whichever layer happened to be last in
// the buffer. Accumulating rather than sampling one captured layer is
// strictly more robust and costs one atomic per slot.
__device__ unsigned long long
    g_phase_span[MPK_PHASE_MAX_WORKERS * MPK_PHASE_SLOT_STRIDE];
__device__ unsigned long long
    g_phase_n[MPK_PHASE_MAX_WORKERS * MPK_PHASE_PAD_U64];
// Only sample steady-state decode.
//
// `volatile` is load-bearing, not decoration. The whole layer body is inlined
// into the persistent `ml` loop inside one dispatch, so a plain __device__ int
// that no thread in this wave writes is a loop-invariant load: clang hoists it
// out, every mark tests the value read on the very first iteration -- zero --
// and the recorder silently reports nothing at all. Volatile also gets the
// cross-XCD visibility this needs, since only worker 0 ever writes it.
#ifdef MPK_QKV_SPLIT
// Slot 1 split: setup+barrier vs the QKV GEMM. Same shape as the g_il_*
// accumulators above -- atomicAdd from tid 0, printed once at the end.
__device__ unsigned long long g_qkv_setup_sum = 0;
__device__ unsigned long long g_qkv_gemm_sum = 0;
__device__ unsigned long long g_qkv_n = 0;
#endif
__device__ volatile int g_phase_arm;

// Which iteration to start on. There is no device-side signal that separates
// prefill from decode here: the sweep runs prefill at
// --max-num-batched-tokens 1, so `num_active_tokens == 1` on *every* iteration
// of both phases. Arming on that would average 512 prefill layers into the
// answer and bias every span toward short-context. So the caller states the
// boundary, and `[PSLOT] layers_per_worker` is the check: it must equal
// 36 * (total_iters - MPK_PHASE_START_ITER) against the [FWD_PASS] tail.
#ifndef MPK_PHASE_START_ITER
#define MPK_PHASE_START_ITER 600
#endif

// Per-worker "this layer started while armed". The arm fires between two
// layers, so without this the layer straddling it contributes spans for its
// late slots only, yet still bumps g_phase_n at slot 11 -- every early-slot
// mean would then be divided by one layer too many. Set at slot 0, required by
// every other slot.
__device__ int g_phase_live[MPK_PHASE_MAX_WORKERS * MPK_PHASE_PAD_INT];

#ifdef MPK_PHASE_LDS
// LDS-resident recorder: only s_memrealtime and LDS on the hot path, so its
// cost is the same in NPS1 and NPS2 (the global recorder below is not: its
// per-mark g_phase_* traffic lands on MTYPE_NC lines in NPS2). A worker arms
// itself after MPK_PHASE_START_ITER iterations of its own layers, and copies
// the sums to the globals the printer reads once per iteration.
#ifndef MPK_PHASE_LAYERS_PER_ITER
#define MPK_PHASE_LAYERS_PER_ITER 36
#endif
#ifndef MPK_PSNAPD_WPD
#define MPK_PSNAPD_WPD 31
#endif
#ifdef MPK_PHASE_SNAP
// Every slot timestamp of every worker for the layers of iteration
// MPK_PHASE_SNAP, so the printer can take per-layer min/max over workers.
__device__ unsigned long long g_phase_snap[MPK_PHASE_LAYERS_PER_ITER *
                                           MPK_PHASE_MAX_WORKERS *
                                           MPK_PHASE_SLOT_COUNT];
#ifdef MPK_MULTI_DEV // MDEV_PSNAP: every device's workers stamp into device 0's g_phase_snap (the printer's)
__constant__ unsigned long long *g_phase_snap_dst = nullptr;
#define MPK_PSNAP_RD (g_phase_snap_dst != nullptr ? g_phase_snap_dst : g_phase_snap)
#else
#define MPK_PSNAP_RD g_phase_snap
#endif
#endif
#ifdef MPK_VTN_MOETL // MPK_VTN_MOETL: per-call union-MoE timeline of the snapshot pass
#ifndef MPK_PHASE_SNAP
#error "MPK_VTN_MOETL needs MPK_PHASE_SNAP"
#endif
#define MPK_VTN_TLC 16
#ifndef MPK_VTN_MOETL_L0
#define MPK_VTN_MOETL_L0 0
#endif
#ifndef MPK_VTN_MOETL_NL
#define MPK_VTN_MOETL_NL 8
#endif
__device__ unsigned long long g_vtn_tl[MPK_PHASE_LAYERS_PER_ITER * MPK_PHASE_MAX_WORKERS * MPK_VTN_TLC * 3];
#ifdef MPK_MULTI_DEV
__constant__ unsigned long long *g_vtn_tl_dst = nullptr;
#define MPK_VTN_TL_RD (g_vtn_tl_dst != nullptr ? g_vtn_tl_dst : g_vtn_tl)
#else
#define MPK_VTN_TL_RD g_vtn_tl
#endif
#endif
#ifdef MPK_IL_LDS
// Low 32 bits of s_memrealtime at the ml loop's top (0) and just before the
// tile loop (1); spans wrap-safe for anything under 42 s.
__shared__ unsigned int s_il_ts[2];
__shared__ unsigned int s_il_acc[3];
__shared__ unsigned int s_il_n;
__device__ unsigned int g_il_acc[MPK_PHASE_MAX_WORKERS * 3];
__device__ unsigned int g_il_n[MPK_PHASE_MAX_WORKERS];
#define MPK_IL_MARK(k)                                                         \
  do {                                                                         \
    if (threadIdx.x == 0)                                                      \
      s_il_ts[(k)] = (unsigned int)__builtin_amdgcn_s_memrealtime();          \
  } while (0)
#endif
#ifdef MPK_QKV_SUB_LDS
// QKV epoch and chunk barrier split per worker (iterations >= 40), ns sums:
// 0 producer drain, 1 arrival atomic, 2 epoch wait, 3 chunk drain, 4 chunk arrival.
__shared__ unsigned int s_qs_acc[5];
__shared__ unsigned int s_qs_n[3];  // qkv passes, chunk passes, xcd_id + 1
__device__ unsigned int g_qs_acc[MPK_PHASE_MAX_WORKERS * 5];
__device__ unsigned int g_qs_n[MPK_PHASE_MAX_WORKERS * 3];
#endif
#ifdef MPK_QKVK_LDS
// QKV kernel stages per worker, ns sums over armed layers:
// 0 fold + RMSNorm, 1 quant + weight DMA drain, 2 MFMA setup,
// 3 MFMA + RoPE / KV epilogue.
__shared__ unsigned int s_qk_ts[5];
__shared__ unsigned int s_qk_acc[4];
__shared__ unsigned int s_qk_n;
__device__ unsigned int g_qk_acc[MPK_PHASE_MAX_WORKERS * 4];
__device__ unsigned int g_qk_n[MPK_PHASE_MAX_WORKERS];
#define MPK_QKVK_MARK(k)                                                      \
  do {                                                                        \
    if (threadIdx.x == 0) {                                                   \
      s_qk_ts[(k)] = (unsigned int)__builtin_amdgcn_s_memrealtime();         \
      if ((k) == 4 &&                                                         \
          s_phase_layers >= (unsigned int)(MPK_PHASE_START_ITER *              \
                                           MPK_PHASE_LAYERS_PER_ITER)) {       \
        for (int _q = 0; _q < 4; _q++) {                                      \
          s_qk_acc[_q] += (s_qk_ts[_q + 1] - s_qk_ts[_q]) * 10;               \
        }                                                                     \
        s_qk_n++;                                                             \
      }                                                                       \
    }                                                                         \
  } while (0)
#endif
#ifdef MPK_TOPK_SUB
#ifndef MPK_PHASE_LDS
#error "MPK_TOPK_SUB arms with the LDS phase recorder: set MPK_PHASE_SLOTS=1 MPK_PHASE_LDS=1"
#endif
// TopK (ROUTER_OPROJ_PARTIAL phase 7b) steps per worker, ns sums over armed layers:
// 0 wait for all 8 dies' sum-of-squares stamps, 1 gather the row + 8 x 129 sums,
// 2 1/rms + expert scores, 3 top 4 with the FP8 MoE-input conversion overlapped.
__shared__ unsigned int s_tk_acc[4];
#ifdef MPK_TOPK_SUB2
__shared__ unsigned int s_tk2_acc[8];
__device__ unsigned int g_tk2_acc[MPK_PHASE_MAX_WORKERS * 8];
#endif
__shared__ unsigned int s_tk_n;
__device__ unsigned int g_tk_acc[MPK_PHASE_MAX_WORKERS * 4];
__device__ unsigned int g_tk_n[MPK_PHASE_MAX_WORKERS];
#define MPK_TOPK_ARMED()                                                        \
  (s_phase_layers >=                                                          \
   (unsigned int)(MPK_PHASE_START_ITER * MPK_PHASE_LAYERS_PER_ITER))
#endif
#ifdef MPK_OPROJ_SUB
#ifndef MPK_PHASE_LDS
#error "MPK_OPROJ_SUB arms with the LDS phase recorder: set MPK_PHASE_SLOTS=1 MPK_PHASE_LDS=1"
#endif
// O-proj kernel steps per worker, ns sums over armed layers: 0 weight-tile drain, 1 wait for the
// last wave's attention slices, 2 convert + MFMA + wave reduce, 3 residual/bias/store + router
// partial, 4 drain + die combine + publish. s_os_w: each wave's slice-wait end (lane 0).
__shared__ unsigned int s_os_w[4];
__shared__ unsigned int s_os_acc[5];
__shared__ unsigned int s_os_n;
__device__ unsigned int g_os_acc[MPK_PHASE_MAX_WORKERS * 5];
__device__ unsigned int g_os_n[MPK_PHASE_MAX_WORKERS];
#define MPK_OS_ARMED()                                                          \
  (s_phase_layers >=                                                          \
   (unsigned int)(MPK_PHASE_START_ITER * MPK_PHASE_LAYERS_PER_ITER))
#endif
#ifdef MPK_ATTK_LDS
// Attention marks, ns sums per cell = sliding * 12 + MPK_AK_SLOT(prev * 10 + id):
// 0 start>scan0, 1 scan0>pro, 2 pro>ored, 3 ored>pstore, 4 pstore>cbar,
// 5 start>cbar, 6 cbar>cbdone, 7 cbdone>merged, 8 merged>drain,
// 9 drain>end, 10 cbdone>end, 11 other (spans ending at start excluded).
// Kept small: the worker's static LDS has under 2 KB of headroom.
__shared__ unsigned int s_ak_prev;  // id + 1, 0 = none yet
__shared__ unsigned int s_ak_t;
__shared__ unsigned int s_ak_sw;
__shared__ unsigned int s_ak_acc[24];
__shared__ unsigned int s_ak_cnt[24];
__device__ unsigned int g_ak_acc[MPK_PHASE_MAX_WORKERS * 24];
__device__ unsigned int g_ak_cnt[MPK_PHASE_MAX_WORKERS * 24];
#define MPK_AK_SLOT(k)                                                   \
  ((k) == 6 ? 0 : (k) == 67 ? 1 : (k) == 78 ? 2 : (k) == 89 ? 3 :        \
   (k) == 91 ? 4 : (k) == 1 ? 5 : (k) == 12 ? 6 : (k) == 23 ? 7 :        \
   (k) == 34 ? 8 : (k) == 45 ? 9 : (k) == 25 ? 10 : 11)
#define mpk_ak_mark(id_)                                                 \
  do {                                                                   \
    if (threadIdx.x == 0) {                                              \
      unsigned int const _ak_id = (id_);                                 \
      unsigned int const _ak_t =                                         \
          (unsigned int)__builtin_amdgcn_s_memrealtime();                \
      unsigned int const _ak_p = s_ak_prev;                              \
      if (_ak_p != 0 && _ak_id != 0 &&                                   \
          s_phase_layers >= (unsigned int)(MPK_PHASE_START_ITER *         \
                                           MPK_PHASE_LAYERS_PER_ITER)) {  \
        unsigned int const _ak_c =                                       \
            s_ak_sw * 12 + MPK_AK_SLOT((_ak_p - 1) * 10 + _ak_id);       \
        s_ak_acc[_ak_c] += (_ak_t - s_ak_t) * 10;                        \
        s_ak_cnt[_ak_c]++;                                               \
      }                                                                  \
      s_ak_prev = _ak_id + 1;                                            \
      s_ak_t = _ak_t;                                                    \
    }                                                                    \
  } while (0)
#undef MPK_ATSTAMP
#define MPK_ATSTAMP(k) mpk_ak_mark((unsigned int)(k))
#undef MPK_ASSTAMP
#define MPK_ASSTAMP(k)                                   \
  do {                                                   \
    if ((k) == 0 || (k) == 1) {                          \
      mpk_ak_mark(6u + (unsigned int)(k));               \
    } else if ((k) == 5 || (k) == 6) {                   \
      mpk_ak_mark(3u + (unsigned int)(k));               \
    }                                                    \
  } while (0)
#endif
#ifdef MPK_W13_SUB
__shared__ unsigned int s_w13_prev;
__shared__ unsigned int s_w13_acc[10];
__shared__ unsigned int s_w13_n;
__device__ unsigned int g_w13_acc[MPK_PHASE_MAX_WORKERS * 10];
__device__ unsigned int g_w13_n[MPK_PHASE_MAX_WORKERS];
__device__ unsigned int g_w13_hwid[MPK_PHASE_MAX_WORKERS];
#define MPK_W13_ARMED()                                                        \
  (s_phase_layers >=                                                         \
   (unsigned int)(MPK_PHASE_START_ITER * MPK_PHASE_LAYERS_PER_ITER))
#define MPK_W13_START()                                                        \
  do {                                                                         \
    if (threadIdx.x == 0)                                                      \
      s_w13_prev = (unsigned int)__builtin_amdgcn_s_memrealtime();            \
  } while (0)
#define MPK_W13_MARK(k)                                                        \
  do {                                                                         \
    if (threadIdx.x == 0 && MPK_W13_ARMED()) {                                 \
      unsigned int const _w13t = (unsigned int)__builtin_amdgcn_s_memrealtime(); \
      s_w13_acc[(k)] += (_w13t - s_w13_prev) * 10;                             \
      s_w13_prev = _w13t;                                                      \
    }                                                                          \
  } while (0)
#define MPK_W13_DONE()                                                         \
  do {                                                                         \
    if (threadIdx.x == 0 && MPK_W13_ARMED()) s_w13_n++;                        \
  } while (0)
#endif
__shared__ unsigned long long s_phase_ts[MPK_PHASE_SLOT_COUNT];
__shared__ unsigned long long s_phase_span[MPK_PHASE_SLOT_COUNT];
__shared__ unsigned long long s_phase_prev_end;
__shared__ unsigned long long s_phase_span_b;
__shared__ unsigned int s_phase_layers;
#if defined(MPK_LTK_SNAP) && defined(MPK_PHASE_SNAP)
// Local-TopK propagation stamps for the layers of iteration
// MPK_PHASE_SNAP: [0] router logit publish, [1] poll start,
// [2] poll success, [3] TopK end. Worker id from the slot-0 mark.
__device__ unsigned long long
    g_ltk_snap[MPK_PHASE_LAYERS_PER_ITER * MPK_PHASE_MAX_WORKERS * 4];
__shared__ int s_ltk_worker;
#define MPK_LTK_MARK(k)                                                        \
  do {                                                                         \
    if (threadIdx.x == 0) {                                                    \
      unsigned int const _ltk_l =                                              \
          s_phase_layers -                                                     \
          (unsigned int)(MPK_PHASE_SNAP * MPK_PHASE_LAYERS_PER_ITER);          \
      if (_ltk_l < (unsigned int)MPK_PHASE_LAYERS_PER_ITER &&                  \
          (unsigned int)s_ltk_worker < (unsigned int)MPK_PHASE_MAX_WORKERS) {  \
        g_ltk_snap[((size_t)_ltk_l * MPK_PHASE_MAX_WORKERS + s_ltk_worker) *   \
                       4 +                                                     \
                   (k)] = __builtin_amdgcn_s_memrealtime();                    \
      }                                                                        \
    }                                                                          \
  } while (0)
#endif
#if defined(MPK_XSNAP) && defined(MPK_PHASE_SNAP)
// Barrier stamps for the layers of iteration MPK_PHASE_SNAP; see p_xsnap.py.
__device__ unsigned long long
    g_xs_snap[MPK_PHASE_LAYERS_PER_ITER * MPK_PHASE_MAX_WORKERS * 16];
__shared__ int s_xs_worker;
#define MPK_XS_MARK(k)                                                         \
  do {                                                                         \
    if (threadIdx.x == 0) {                                                    \
      unsigned int const _xs_l =                                               \
          s_phase_layers -                                                     \
          (unsigned int)(MPK_PHASE_SNAP * MPK_PHASE_LAYERS_PER_ITER);          \
      if (_xs_l < (unsigned int)MPK_PHASE_LAYERS_PER_ITER &&                   \
          (unsigned int)s_xs_worker < (unsigned int)MPK_PHASE_MAX_WORKERS) {   \
        g_xs_snap[((size_t)_xs_l * MPK_PHASE_MAX_WORKERS + s_xs_worker) * 16 + \
                  (k)] = __builtin_amdgcn_s_memrealtime();                     \
      }                                                                        \
    }                                                                          \
  } while (0)
#endif
#if defined(MPK_TK_OPS) && defined(MPK_PHASE_SNAP)
// MPK_TK_OPS (timing tool): O-proj / Phase 7b stamps of iteration MPK_PHASE_SNAP, 16 per (layer, worker); under
// MPK_MULTI_DEV every device stamps one buffer on device 0 (g_tk_ops_dst, as MDEV_PSNAP).
__device__ unsigned long long g_tk_ops[MPK_PHASE_LAYERS_PER_ITER * MPK_PHASE_MAX_WORKERS * 16];
__constant__ unsigned long long *g_tk_ops_dst = nullptr;
__shared__ int s_tkops_worker;
#ifdef MPK_TK_OPS_LDS
// MPK_TK_OPS_LDS: stamps into LDS, written out by thread 0 at the next layer's slot-0 mark, so no global store
// sits in front of a vmcnt wait on the timed path.
__shared__ unsigned long long s_tkops_v[16];
__shared__ unsigned int s_tkops_l;
#define MPK_TKOPS_AT(k, cond)                                                                          \
  do {                                                                                                 \
    if (cond) {                                                                                        \
      unsigned int const _to_l = s_phase_layers - (unsigned int)(MPK_PHASE_SNAP * MPK_PHASE_LAYERS_PER_ITER); \
      if (_to_l < (unsigned int)MPK_PHASE_LAYERS_PER_ITER &&                                           \
          (unsigned int)s_tkops_worker < (unsigned int)MPK_PHASE_MAX_WORKERS) {                        \
        s_tkops_v[(k)] = __builtin_amdgcn_s_memrealtime();                                             \
        s_tkops_l = _to_l;                                                                             \
      }                                                                                                \
    }                                                                                                  \
  } while (0)
#else
#define MPK_TKOPS_AT(k, cond)                                                                          \
  do {                                                                                                 \
    if (cond) {                                                                                        \
      unsigned int const _to_l = s_phase_layers - (unsigned int)(MPK_PHASE_SNAP * MPK_PHASE_LAYERS_PER_ITER); \
      if (_to_l < (unsigned int)MPK_PHASE_LAYERS_PER_ITER &&                                           \
          (unsigned int)s_tkops_worker < (unsigned int)MPK_PHASE_MAX_WORKERS) {                        \
        unsigned long long const _to_t = __builtin_amdgcn_s_memrealtime();                             \
        __hip_atomic_store(&(g_tk_ops_dst != nullptr ? g_tk_ops_dst : g_tk_ops)                        \
                               [((size_t)_to_l * MPK_PHASE_MAX_WORKERS + s_tkops_worker) * 16 + (k)],  \
                           _to_t, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);                        \
      }                                                                                                \
    }                                                                                                  \
  } while (0)
#endif  // MPK_TK_OPS_LDS
#endif
#if defined(MPK_ATS_OPS) && defined(MPK_PHASE_SNAP)
// MPK_ATS_OPS (timing tool): attention / sync stamps of iteration MPK_PHASE_SNAP, 16 per (layer, worker). Stamps
// go to LDS (no memory traffic on the timed path); the layer-exit phase mark flushes them with plain stores into
// device 0's own g_ats_ops, from dies 0..3 only (device 0 runs the printer).
__device__ unsigned long long g_ats_ops[MPK_PHASE_LAYERS_PER_ITER * MPK_PHASE_MAX_WORKERS * 16];
__constant__ unsigned long long *g_ats_ops_dst = nullptr;
#define MPK_ATS_RD (g_ats_ops_dst != nullptr ? g_ats_ops_dst : g_ats_ops)
__shared__ int s_atsops_worker;
__shared__ unsigned long long s_ats_buf[16];
#define MPK_ATSOPS_PUT(k, cond, val)                 \
  do {                                               \
    if (cond) {                                      \
      s_ats_buf[(k)] = (unsigned long long)(val);    \
    }                                                \
  } while (0)
#define MPK_ATSOPS_AT(k, cond) MPK_ATSOPS_PUT(k, cond, __builtin_amdgcn_s_memrealtime())
#else
#define MPK_ATSOPS_PUT(k, cond, val) \
  do {                               \
  } while (0)
#define MPK_ATSOPS_AT(k, cond) \
  do {                         \
  } while (0)
#endif
__shared__ unsigned int s_phase_n;

#ifdef MPK_OPROJ_LDS
__shared__ unsigned long long s_sub_ts[4];
__shared__ unsigned long long s_sub_span[4];
__shared__ unsigned int s_sub_live;
__shared__ unsigned int s_sub_n;
__shared__ unsigned int s_sub_pub;
__shared__ unsigned long long s_sub_last;
__shared__ unsigned int s_sub_last_n;
__shared__ unsigned long long s_tk_ts[3];
__shared__ unsigned long long s_tk_span[3];
__shared__ unsigned int s_tk_live;
__shared__ unsigned int s_tk_n;
__device__ unsigned long long g_tk_span[MPK_PHASE_MAX_WORKERS * 3];
__device__ unsigned int g_tk_n[MPK_PHASE_MAX_WORKERS];
__device__ unsigned long long g_sub_last[MPK_PHASE_MAX_WORKERS];
__device__ unsigned int g_sub_last_n[MPK_PHASE_MAX_WORKERS];
__device__ unsigned long long g_sub_span[MPK_PHASE_MAX_WORKERS * 4];
__device__ unsigned int g_sub_n[MPK_PHASE_MAX_WORKERS];
__device__ __forceinline__ void mpk_sub_mark(int k) {
  if (threadIdx.x != 0) {
    return;
  }
  asm volatile("" ::: "memory");
  s_sub_ts[k] = __builtin_amdgcn_s_memrealtime();
  asm volatile("" ::: "memory");
  s_sub_live |= 1u << k;
}
#define MPK_SUB_MARK(k) mpk_sub_mark(k)
#define MPK_SUB_TOPK(k)                                                        \
  do {                                                                         \
    if (threadIdx.x == 0) {                                                    \
      s_tk_ts[(k)] = __builtin_amdgcn_s_memrealtime();                         \
      s_tk_live |= 1u << (k);                                                  \
    }                                                                          \
  } while (0)
#define MPK_SUB_PUBLISHER()                                                    \
  do {                                                                         \
    if (threadIdx.x == 0) s_sub_pub = 1u;                                      \
  } while (0)
#else
#define MPK_SUB_MARK(k) do {} while (0)
#endif
#ifdef MPK_MOE_LDS
// Per-worker MoE split, summed over every armed tile: W13 dec/compute/arrive
// and W2 dec/prep/barrier/compute (ns). Copied out with the phase spans.
__shared__ unsigned int s_moe_acc[7];
__shared__ unsigned int s_moe_n[2];
#ifdef MPK_MOE_LDS_MAX
// [0] max W13 arrival ns, [1] arrivals > 1 us, [2] max W2 barrier ns,
// [3] W2 barrier waits > 1 us.
__shared__ unsigned int s_moe_x[4];
__device__ unsigned int g_moe_x[MPK_PHASE_MAX_WORKERS * 4];
#endif
__device__ unsigned int g_moe_acc[MPK_PHASE_MAX_WORKERS * 7];
__device__ unsigned int g_moe_n[MPK_PHASE_MAX_WORKERS * 2];
#ifdef MPK_MOE_CALL_LDS
// Caller-side ns around each MoE call (entry drain + tile + return drain)
// and call counts: [0] W13 pass, [1] W2 pass, [2..3] counts.
__shared__ unsigned int s_moe_call[4];
__device__ unsigned int g_moe_call[MPK_PHASE_MAX_WORKERS * 4];
#endif
#define MPK_MOE_ARMED()                                                        \
  (s_phase_layers >=                                                         \
   (unsigned int)(MPK_PHASE_START_ITER * MPK_PHASE_LAYERS_PER_ITER))
#endif
__device__ __forceinline__ void mpk_phase_lds_init() {
  if (threadIdx.x == 0) {
    for (int s = 0; s < MPK_PHASE_SLOT_COUNT; s++) {
      s_phase_ts[s] = 0;
      s_phase_span[s] = 0;
    }
    s_phase_prev_end = 0;
    s_phase_span_b = 0;
#ifdef MPK_OPROJ_LDS
    for (int k = 0; k < 4; k++) s_sub_span[k] = 0;
    s_sub_live = 0;
    s_sub_n = 0;
    s_sub_pub = 0;
    s_sub_last = 0;
    s_sub_last_n = 0;
    for (int k = 0; k < 3; k++) s_tk_span[k] = 0;
    s_tk_live = 0;
    s_tk_n = 0;
#endif
    s_phase_layers = 0;
    s_phase_n = 0;
#if defined(MPK_TK_OPS) && defined(MPK_PHASE_SNAP) && defined(MPK_TK_OPS_LDS)
    for (int k = 0; k < 16; k++) s_tkops_v[k] = 0ull;
    s_tkops_l = 0xFFFFFFFFu;
#endif
#ifdef MPK_W13_SUB
    for (int k = 0; k < 10; k++) s_w13_acc[k] = 0;
    s_w13_n = 0;
    s_w13_prev = 0;
#endif
#ifdef MPK_IL_LDS
    s_il_ts[0] = 0;
    s_il_ts[1] = 0;
    for (int k = 0; k < 3; k++) s_il_acc[k] = 0;
    s_il_n = 0;
#endif
#ifdef MPK_QKV_SUB_LDS
    for (int k = 0; k < 5; k++) s_qs_acc[k] = 0;
    for (int k = 0; k < 3; k++) s_qs_n[k] = 0;
#endif
#ifdef MPK_QKVK_LDS
    for (int k = 0; k < 4; k++) s_qk_acc[k] = 0;
    s_qk_n = 0;
#endif
#ifdef MPK_OPROJ_SUB
    for (int k = 0; k < 5; k++) s_os_acc[k] = 0;
    s_os_n = 0;
#endif
#ifdef MPK_TOPK_SUB
    for (int k = 0; k < 4; k++) s_tk_acc[k] = 0;
#ifdef MPK_TOPK_SUB2
    for (int k = 0; k < 8; k++) s_tk2_acc[k] = 0;
#endif
    s_tk_n = 0;
#endif
#ifdef MPK_ATTK_LDS
    for (int k = 0; k < 24; k++) {
      s_ak_acc[k] = 0;
      s_ak_cnt[k] = 0;
    }
    s_ak_prev = 0;
    s_ak_sw = 0;
#endif
#ifdef MPK_MOE_LDS
    for (int k = 0; k < 7; k++) s_moe_acc[k] = 0;
    s_moe_n[0] = 0;
    s_moe_n[1] = 0;
#ifdef MPK_MOE_LDS_MAX
    for (int k = 0; k < 4; k++) s_moe_x[k] = 0;
#endif
#endif
#ifdef MPK_MOE_CALL_LDS
    for (int k = 0; k < 4; k++) s_moe_call[k] = 0;
#endif
  }
}

__device__ __forceinline__ void mpk_phase_mark(int worker, int slot) {
  if (threadIdx.x != 0 || worker >= MPK_PHASE_MAX_WORKERS) {
    return;
  }
  asm volatile("" ::: "memory");
  unsigned long long const t = __builtin_amdgcn_s_memrealtime();
  asm volatile("" ::: "memory");
  s_phase_ts[slot] = t;
#if defined(MPK_LTK_SNAP) && defined(MPK_PHASE_SNAP)
  if (slot == 0) {
    s_ltk_worker = worker;
  }
#endif
#if defined(MPK_XSNAP) && defined(MPK_PHASE_SNAP)
  if (slot == 0) {
    s_xs_worker = worker;
  }
#endif
#if defined(MPK_TK_OPS) && defined(MPK_PHASE_SNAP)
  if (slot == 0) {
#ifdef MPK_TK_OPS_LDS
    if (s_tkops_l < (unsigned int)MPK_PHASE_LAYERS_PER_ITER) {
      unsigned long long *const _tkd = (g_tk_ops_dst != nullptr ? g_tk_ops_dst : g_tk_ops) +
                                       ((size_t)s_tkops_l * MPK_PHASE_MAX_WORKERS + worker) * 16;
      for (int _k = 0; _k < 16; _k++) {
        if (s_tkops_v[_k] != 0ull)
          __hip_atomic_store(&_tkd[_k], s_tkops_v[_k], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        s_tkops_v[_k] = 0ull;
      }
    }
    s_tkops_l = 0xFFFFFFFFu;
#endif
    s_tkops_worker = worker;
  }
#endif
#if defined(MPK_ATS_OPS) && defined(MPK_PHASE_SNAP)
  if (slot == 0) {
    s_atsops_worker = worker;
  }
#endif
#ifdef MPK_PHASE_SNAP
  {
    // s_phase_layers still counts completed layers here, so it is the
    // index of the layer this mark belongs to.
    unsigned int const snap_l =
        s_phase_layers -
        (unsigned int)(MPK_PHASE_SNAP * MPK_PHASE_LAYERS_PER_ITER);
    if (snap_l < (unsigned int)MPK_PHASE_LAYERS_PER_ITER) {
#ifdef MPK_MULTI_DEV // MDEV_PSNAP: system scope, so the printer on device 0 sees every device's stamps
      __hip_atomic_store(&(g_phase_snap_dst != nullptr ? g_phase_snap_dst : g_phase_snap)
                             [((size_t)snap_l * MPK_PHASE_MAX_WORKERS + worker) * MPK_PHASE_SLOT_COUNT + slot],
                         (unsigned long long)t, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
      g_phase_snap[((size_t)snap_l * MPK_PHASE_MAX_WORKERS + worker) *
                       MPK_PHASE_SLOT_COUNT +
                   slot] = t;
#endif
    }
  }
#endif
#if defined(MPK_ATS_OPS) && defined(MPK_PHASE_SNAP)
  if (slot == MPK_PHASE_SLOT_COUNT - 1) {
    unsigned int const _ao_l =
        s_phase_layers - (unsigned int)(MPK_PHASE_SNAP * MPK_PHASE_LAYERS_PER_ITER);
    if (_ao_l < (unsigned int)MPK_PHASE_LAYERS_PER_ITER && worker < 4 * MPK_PSNAPD_WPD) {
      unsigned long long *const _ao_g = &MPK_ATS_RD[((size_t)_ao_l * MPK_PHASE_MAX_WORKERS + worker) * 16];
      for (int _k = 0; _k < 16; ++_k) {
        _ao_g[_k] = s_ats_buf[_k];
      }
    }
    for (int _k = 0; _k < 16; ++_k) {
      s_ats_buf[_k] = 0ull;
    }
  }
#endif
  if (slot != MPK_PHASE_SLOT_COUNT - 1) {
    return;
  }
  unsigned int const layers = ++s_phase_layers;
#ifdef MPK_OPROJ_LDS
  unsigned int const _sub_live = s_sub_live;
  s_sub_live = 0;
#endif
  unsigned long long const prev_end = s_phase_prev_end;
  s_phase_prev_end = t;
  if (layers <= (unsigned int)(MPK_PHASE_START_ITER * MPK_PHASE_LAYERS_PER_ITER)) {
    return;
  }
#ifdef MPK_OPROJ_LDS
  if (_sub_live == 0xFu && s_phase_ts[6] >= s_sub_ts[3]) {
    s_sub_span[0] += (s_sub_ts[1] - s_sub_ts[0]) * 10;
    s_sub_span[1] += (s_sub_ts[2] - s_sub_ts[1]) * 10;
    s_sub_span[2] += (s_sub_ts[3] - s_sub_ts[2]) * 10;
    s_sub_span[3] += (s_phase_ts[6] - s_sub_ts[3]) * 10;
    s_sub_n++;
    if (s_tk_live == 0x7u && s_tk_ts[0] >= s_sub_ts[3]) {
      s_tk_span[0] += (s_tk_ts[0] - s_sub_ts[3]) * 10;
      s_tk_span[1] += (s_tk_ts[1] - s_tk_ts[0]) * 10;
      s_tk_span[2] += (s_tk_ts[2] - s_tk_ts[1]) * 10;
      s_tk_n++;
    }
    if (s_sub_pub) {
      s_sub_last += (s_sub_ts[2] - s_sub_ts[1]) * 10;
      s_sub_last_n++;
    }
  }
  s_sub_pub = 0;
  s_tk_live = 0;
#endif
  // Slot 0 is the inter-layer span (previous layer's last mark to this
  // layer's first), exactly as in the global recorder.
  if (prev_end != 0 && s_phase_ts[0] >= prev_end) {
    unsigned long long const d0 = (s_phase_ts[0] - prev_end) * 10; // ns
    if ((layers - 1) % MPK_PHASE_LAYERS_PER_ITER == 0) {
      s_phase_span_b += d0; // layer 0: the span between two forward passes
    } else {
      s_phase_span[0] += d0;
    }
  }
#ifdef MPK_IL_LDS
  if (prev_end != 0 && (layers - 1) % MPK_PHASE_LAYERS_PER_ITER != 0) {
    unsigned int const pe = (unsigned int)prev_end;
    unsigned int const ta = s_il_ts[0], tb = s_il_ts[1];
    unsigned int const tc = (unsigned int)s_phase_ts[0];
    if ((int)(ta - pe) >= 0 && (int)(tb - ta) >= 0 && (int)(tc - tb) >= 0) {
      s_il_acc[0] += (ta - pe) * 10;
      s_il_acc[1] += (tb - ta) * 10;
      s_il_acc[2] += (tc - tb) * 10;
      s_il_n++;
    }
  }
#endif
  for (int s = 1; s < MPK_PHASE_SLOT_COUNT; s++) {
    unsigned long long const a = s_phase_ts[s - 1];
    unsigned long long const b = s_phase_ts[s];
    if (b >= a) {
      s_phase_span[s] += (b - a) * 10;
    }
  }
  unsigned int const n = ++s_phase_n;
  if (n % MPK_PHASE_LAYERS_PER_ITER == 0) {
    int const base = worker * MPK_PHASE_SLOT_STRIDE;
    for (int s = 0; s < MPK_PHASE_SLOT_COUNT; s++) {
      g_phase_span[base + s] = s_phase_span[s];
      g_phase_ts[base + s] = s_phase_ts[s];
    }
    g_phase_span[base + MPK_PHASE_SLOT_COUNT] = s_phase_span_b;
#ifdef MPK_OPROJ_LDS
    for (int k = 0; k < 4; k++) g_sub_span[worker * 4 + k] = s_sub_span[k];
    g_sub_n[worker] = s_sub_n;
    g_sub_last[worker] = s_sub_last;
    g_sub_last_n[worker] = s_sub_last_n;
    for (int k = 0; k < 3; k++) g_tk_span[worker * 3 + k] = s_tk_span[k];
    g_tk_n[worker] = s_tk_n;
#endif
#ifdef MPK_MOE_LDS
    for (int k = 0; k < 7; k++) g_moe_acc[worker * 7 + k] = s_moe_acc[k];
    g_moe_n[worker * 2] = s_moe_n[0];
    g_moe_n[worker * 2 + 1] = s_moe_n[1];
#ifdef MPK_MOE_LDS_MAX
    for (int k = 0; k < 4; k++) g_moe_x[worker * 4 + k] = s_moe_x[k];
#endif
#endif
#ifdef MPK_MOE_CALL_LDS
    for (int k = 0; k < 4; k++) g_moe_call[worker * 4 + k] = s_moe_call[k];
#endif
#ifdef MPK_IL_LDS
    for (int k = 0; k < 3; k++) g_il_acc[worker * 3 + k] = s_il_acc[k];
    g_il_n[worker] = s_il_n;
#endif
#ifdef MPK_QKV_SUB_LDS
    for (int k = 0; k < 5; k++) g_qs_acc[worker * 5 + k] = s_qs_acc[k];
    for (int k = 0; k < 3; k++) g_qs_n[worker * 3 + k] = s_qs_n[k];
#endif
#ifdef MPK_QKVK_LDS
    for (int k = 0; k < 4; k++) g_qk_acc[worker * 4 + k] = s_qk_acc[k];
    g_qk_n[worker] = s_qk_n;
#endif
#ifdef MPK_OPROJ_SUB
    for (int k = 0; k < 5; k++) g_os_acc[worker * 5 + k] = s_os_acc[k];
    g_os_n[worker] = s_os_n;
#endif
#ifdef MPK_TOPK_SUB
    for (int k = 0; k < 4; k++) g_tk_acc[worker * 4 + k] = s_tk_acc[k];
#ifdef MPK_TOPK_SUB2
    for (int k = 0; k < 8; k++) g_tk2_acc[worker * 8 + k] = s_tk2_acc[k];
#endif
    g_tk_n[worker] = s_tk_n;
#endif
#ifdef MPK_ATTK_LDS
    for (int k = 0; k < 24; k++) {
      g_ak_acc[worker * 24 + k] = s_ak_acc[k];
      g_ak_cnt[worker * 24 + k] = s_ak_cnt[k];
    }
#endif
#ifdef MPK_W13_SUB
    for (int k = 0; k < 10; k++) g_w13_acc[worker * 10 + k] = s_w13_acc[k];
    g_w13_n[worker] = s_w13_n;
    {
      unsigned _hw;
      asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s"(_hw));
      g_w13_hwid[worker] = _hw;
    }
#endif
    g_phase_n[worker * MPK_PHASE_PAD_U64] = n;
  }
}
#else
__device__ __forceinline__ void mpk_phase_mark(int worker, int slot) {
  if (threadIdx.x != 0 || worker >= MPK_PHASE_MAX_WORKERS) {
    return;
  }
  if (slot == 0) {
    if (!g_phase_arm) {
      return;
    }
    g_phase_live[(worker) * MPK_PHASE_PAD_INT] = 1;
  } else if (!g_phase_live[(worker) * MPK_PHASE_PAD_INT]) {
    return;
  }
  asm volatile("" ::: "memory");
  unsigned long long const t = __builtin_amdgcn_s_memrealtime();
  asm volatile("" ::: "memory");
  int const base = worker * MPK_PHASE_SLOT_STRIDE;
  // Slot 0 accumulates the *inter-layer* span: last layer's slot 11 to this
  // layer's slot 0. It used to record only a timestamp, which left everything
  // between two layers outside the trace -- and that hole is not small. At
  // ctx 512 the eleven measured spans sum to 1.935 ms/token against a measured
  // 2.021, so ~0.09 ms/token -- a significant share of the per-token budget --
  // was being attributed to nothing at all. Anything the persistent worker loop
  // does between the inlined layer bodies (the `ml` loop's own bookkeeping,
  // the per-layer task dispatch, the inter-layer threadfence/syncthreads
  // pair) lands here.
  //
  // Guarded on g_phase_live having been set by a *previous* layer, which is
  // why the read is of slot 11's timestamp rather than slot -1: on the first
  // armed layer there is no predecessor and prev is 0, which the existing
  // `prev != 0` test already rejects. n is bumped at slot 11, so this span is
  // accumulated over one fewer layer than the others -- at 7560 layers the
  // resulting bias is 0.01%, far below anything read off this table.
  unsigned long long const prev =
      g_phase_ts[base + (slot > 0 ? slot - 1 : MPK_PHASE_SLOT_COUNT - 1)];
  g_phase_ts[base + slot] = t;
  if (prev != 0 && t >= prev) {
    g_phase_span[base + slot] += (t - prev) * 10; // 100 MHz -> ns
  }
  if (slot == MPK_PHASE_SLOT_COUNT - 1) {
    g_phase_n[(worker) * MPK_PHASE_PAD_U64]++;
    g_phase_live[(worker) * MPK_PHASE_PAD_INT] = 0;
  }
}
#endif // MPK_PHASE_LDS
#define MPK_PHASE_MARK(worker, slot) mpk_phase_mark((worker), (slot))
#elif defined(MPK_ATT_MARK)
// Thread-trace phase marker: an s_mov_b32 m0 whose immediate 0x4D500000 | slot
// names the phase. Not s_ttracedata: its user data breaks the ATT decoder's
// code-object mapping, so no instruction can be attributed.
#define MPK_PHASE_MARK(worker, slot)                                           \
  asm volatile("s_mov_b32 m0, %0" ::"i"(0x4D500000 | (slot)) : "m0")
#else
#define MPK_PHASE_MARK(worker, slot)                                           \
  do {                                                                         \
  } while (0)
#endif

#ifdef MPK_FUSED_PHASE_TIMING
// Fused O-proj+MoE phase timing: [0]=oproj_ns, [1]=poll_ns, [2]=moe_ns,
// [3]=count
__device__ unsigned long long g_fused_phase_ns[4];
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
#ifdef MPK_MULTI_DEV
#define MPK_WS_BLK (g_mpk_dev_worker_base + (int)blockIdx.x)
#else
#define MPK_WS_BLK ((int)blockIdx.x)
#endif
__device__ int *g_ws_dev;

__device__ __forceinline__ void
    mpk_ws_phase(int phase, int layer, int xcd, int tid) {
  if (tid == 0 && g_ws_dev != nullptr) {
    __atomic_store_n(&g_ws_dev[MPK_WS_BLK * 4 + 3],
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
    int *b = g_ws_dev + g_ws_nworkers * 4 + MPK_WS_BLK * 4;
    __atomic_store_n(&b[0], barrier_id, __ATOMIC_RELAXED);
    __atomic_store_n(&b[2], expected, __ATOMIC_RELAXED);
    __atomic_store_n(&b[3], 0, __ATOMIC_RELAXED);
  }
}

__device__ __forceinline__ void
    mpk_ws_wait_tick(int observed, int spins, int tid) {
  if (tid == 0 && g_ws_dev != nullptr) {
    int *b = g_ws_dev + g_ws_nworkers * 4 + MPK_WS_BLK * 4;
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
    int *b = g_ws_dev + g_ws_nworkers * 8 + MPK_WS_BLK * 4;
    __atomic_store_n(&b[0], a0, __ATOMIC_RELAXED);
    __atomic_store_n(&b[1], a1, __ATOMIC_RELAXED);
    // b[2] and b[3] belong to the per-wave masks below -- not written here.
    (void)a2;
    (void)a3;
  }
}
// Gated on MPK_WORKER_STATE like MPK_WS_MARK / MPK_WS_WAIT_BEGIN below. This
// one was left ungated: the runtime `g_ws_dev != nullptr` test still emits the
// global load and the branch with tracing off, and these sites sit on the MoE
// W13->W2 barrier path.
#ifdef MPK_WORKER_STATE
#define MPK_WS_WAIT_AUX(a0, a1, a2, a3)                                        \
  mpk_ws_wait_aux((a0), (a1), (a2), (a3), tid)
#else
#define MPK_WS_WAIT_AUX(a0, a1, a2, a3) ((void)0)
#endif

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
    int *b = g_ws_dev + g_ws_nworkers * 8 + MPK_WS_BLK * 4;
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
    int *b = g_ws_dev + g_ws_nworkers * 8 + MPK_WS_BLK * 4;
    __atomic_fetch_and(&b[2], ~(1 << (wave & 31)), __ATOMIC_RELAXED);
    __atomic_fetch_and(&b[3], ~(1 << (wave & 31)), __ATOMIC_RELAXED);
  }
}
// Same gate. These two are the costliest of the ungated set: with tracing off
// they still put two dependent global loads and a device-scope atomic between
// clearing the W13->W2 barrier and the first W2 instruction.
#ifdef MPK_WORKER_STATE
#define MPK_WS_WAVE_EXIT(wave) mpk_ws_wave_exit((wave), tid)
#define MPK_WS_WAVE_CLEAR(wave) mpk_ws_wave_clear((wave), tid)
#else
#define MPK_WS_WAVE_EXIT(wave) ((void)0)
#define MPK_WS_WAVE_CLEAR(wave) ((void)0)
#endif

// Same per-wave mask, one slot over, for arrival at a __syncthreads.
//
// hx_2 showed the W13->W2 poll mask at 0xf -- every wave cleared the barrier
// -- with the block still parked at mark 8303, i.e. inside the FP8 quant.
// The only thing in that function that can block is its trailing
// __syncthreads, so the question becomes which wave fails to reach it.
__device__ __forceinline__ void mpk_ws_wave_sync(int wave, int tid) {
  if ((tid & 63) == 0 && g_ws_dev != nullptr) {
    int *b = g_ws_dev + g_ws_nworkers * 8 + MPK_WS_BLK * 4;
    __atomic_fetch_or(&b[2], 1 << (wave & 31), __ATOMIC_RELAXED);
  }
}
#ifdef MPK_WORKER_STATE
#define MPK_WS_WAVE_SYNC(wave) mpk_ws_wave_sync((wave), tid)
#else
#define MPK_WS_WAVE_SYNC(wave) ((void)0)
#endif

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
    int *b = g_ws_dev + g_ws_nworkers * 4 + MPK_WS_BLK * 4;
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

#if defined(MPK_MULTI_DEV) && defined(MPK_SPEC_VSLOT) // MDEV_SPEC
// The virtual-slot spec state is read and written by workers and schedulers on every device, so it lives in
// one buffer (device 0) whose address each device holds in its constant bank: a __device__ array would exist
// once per device. volatile: on gfx940+ every access is a system-scope (sc0 sc1) load or store.
struct MpkSpecShared {
  int state[4];  // [0] 0 = not admitted, 1 = live, 2 = done; [1] n, [2] start step, [3] kind
  long long prev_in[MPK_MAX_NUM_BATCHED_TOKENS];
  int pages[MPK_MAX_NUM_PAGES + 1];
  unsigned long long stats[4];
};
__constant__ MpkSpecShared volatile *g_spec_sh;
#define g_spec_state (g_spec_sh->state)
#define g_spec_prev_in (g_spec_sh->prev_in)
#define g_spec_pages (g_spec_sh->pages)
#define g_spec_stats (g_spec_sh->stats)
#define MPK_SPEC_CV volatile
#else
#define MPK_SPEC_CV
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
#ifdef MPK_ML_STAGE
// Inline-asm LDS-DMA for the layer-boundary row staging: invisible to the
// compiler's wait model, so no vmcnt is inserted before later LDS reads.
// s_nop 4 covers a readfirstlane'd SGPR operand; s_nop 0 is the M0 wait state.
__device__ __forceinline__ void mpk_mls_dma(i32x4_t rsrc, unsigned lds, unsigned voff,
                                            bool b16) {
  if (b16) {
    asm volatile("s_nop 4\n\ts_mov_b32 m0, %[l]\n\ts_nop 0\n\t"
                 "buffer_load_dwordx4 %[v], %[r], 0 offen lds\n"
                 :
                 : [v] "v"(voff), [r] "s"(rsrc), [l] "s"(lds)
                 : "memory", "m0");
  } else {
    asm volatile("s_nop 4\n\ts_mov_b32 m0, %[l]\n\ts_nop 0\n\t"
                 "buffer_load_dword %[v], %[r], 0 offen lds\n"
                 :
                 : [v] "v"(voff), [r] "s"(rsrc), [l] "s"(lds)
                 : "memory", "m0");
  }
}
#endif
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

#ifndef MPK_NUM_XCDS
#define MPK_NUM_XCDS 8
#endif
// Physical XCDs on an SPX agent, independent of how many we choose to use.
#define MPK_XCD_HW 8
// First physical XCD of the subset. 0 = the dies close to memory range 0,
// 4 = those close to range 1.
#ifndef MPK_XCD_BASE
#define MPK_XCD_BASE 0
#endif
#ifdef MPK_XCD_SUBSET
// Dense per-die worker-rank counters, zeroed by prepare_kernel each launch.
__device__ int g_xcd_subset_rank[MPK_XCD_HW];
__device__ int g_xcd_sched_rank[MPK_XCD_HW];
#endif
#ifdef MPK_XCD_SUBSET
#define MPK_XCD_OVERSUB (2 * MPK_XCD_HW / MPK_NUM_XCDS)
#else
#define MPK_XCD_OVERSUB 1
#endif
constexpr int MI300X_NUM_XCDS = MPK_NUM_XCDS;

// Physical XCC_ID is 0-7 even in DPX. Fold into the local 0..N-1 index
// (DPX partition 1 is XCCs 4-7 -> 0-3).
__device__ __forceinline__ int get_current_xcd_id() {
  int xcd_id;
  MPK_GET_XCC_ID16(xcd_id);
  return xcd_id % MPK_NUM_XCDS;
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
#ifdef MPK_FUSED_INLINE
// Generated next to _execute_gang_task; its fused-layer calls inline.
__device__ __forceinline__ void
    _execute_gang_task_ml(TaskDesc const *task_desc,
                          RuntimeConfig const &runtime_config,
#ifdef MPK_IL_CACHE2
                          int tile_idx, int ml_qo_len);
#else
                          int tile_idx);
#endif
#endif

#ifdef MPK_LM_PF0
// MPK_LM_PF0_V1: generated next to _execute_gang_task (persistent_kernel.py).
__device__ __forceinline__ void _mpk_lm_pf0(TaskDesc const *task_desc,
                                            int tile_idx);
#endif
#ifdef MPK_LM_LEADER_INV
#ifndef MPK_AID_SPLIT_FLAGS
#error "MPK_LM_LEADER_INV needs MPK_AID_SPLIT_FLAGS (the AID flag replicas)"
#endif
// The LM head's dependency acquire with one L2 invalidate per XCD: every
// LM-head worker of an XCD waits on the same event, the first to pass it
// invalidates the shared L2 and publishes the iteration, the rest wait for
// that and drop only their vL1. Claim word at +xcc*32, published iteration
// at +16; the flag replicas are cleared per launch and pc_iter starts at 1.
constexpr int MPK_AID_LMINV_BASE_INTS = MPK_AID_L2CLEAN_BASE_INTS + 8 * 32;
static_assert(MPK_AID_LMINV_BASE_INTS + 8 * 32 <= MPK_AID_WSF32_BASE_INTS,
              "MPK_LM_LEADER_INV words overlap MPK_AID_WSF32");
__device__ __forceinline__ void mpk_lm_dep_acquire(bool lm, int iter) {
  if (lm) {
    unsigned xcc;
    MPK_GET_XCC_ID16(xcc);
    xcc &= 7;
    int *w = g_aid_flag_rep[xcc >> 2];
    if (w != nullptr) {
      w += MPK_AID_LMINV_BASE_INTS + (int)xcc * 32;
      if (atomicMax(w, iter) < iter) {
        asm volatile("buffer_inv" MPK_DEV_SC_INV "\n s_waitcnt vmcnt(0)" ::: "memory");
        __hip_atomic_store(w + 16, iter, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
      } else {
        while (__hip_atomic_load(w + 16, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_AGENT) < iter) {
          __builtin_amdgcn_s_sleep(1);
        }
        asm volatile("buffer_inv sc0" ::: "memory");
      }
      return;
    }
  }
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, MPK_DEV_FENCE_ACQ);
}
#endif

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
         t == TASK_GANG_MOE_FUSED_MXFP4_MI300 ||
         t == TASK_GANG_MOE_SWIGLU_W2_MXFP4_MI300 ||
         t == TASK_GANG_MOE_W13_SWIGLU_MXFP4_MI300 ||
         t == TASK_GANG_LINEAR_BIAS_MI300 ||
         t == TASK_GANG_SPLITK_LINEAR_RES_BIAS_MI300 ||
         t == TASK_GANG_RMSNORM_LINEAR_BIAS_MI300 ||
         t == TASK_GANG_RMSNORM_LINEAR_MXFP4_BIAS_MI300 ||
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
         t == TASK_GANG_RMSNORM_LINEAR_MXFP4_BIAS_ARGMAX_MI300;
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

// Increment an event counter and publish the new value into both AID mirrors.
// Returns the PREVIOUS value, exactly like atom_add_release_gpu_u64, so every
// `count + amount == expected` test at the call sites is unchanged.
__device__ __forceinline__ unsigned long long
    mpk_evctr_add(EventCounter *base, int idx, unsigned long long amt) {
  unsigned long long old = atom_add_release_gpu_u64(
      reinterpret_cast<unsigned long long *>(&base[idx]), amt);
#if defined(MPK_AID_EVCTR2) && defined(MPK_AID_SPLIT_FLAGS)
  // Monotone per-XCD slot; falls back to the shared counter only if the
  // replicas are unavailable.
  (void)mpk_aid_evctr2_pub(idx, amt);
#elif defined(MPK_AID_EVCTR) && defined(MPK_AID_SPLIT_FLAGS)
  mpk_aid_evctr_pub(idx, old + amt);
#endif
  return old;
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
#ifdef MPK_CPX_SCOPE // MDEV_CPX_BOUNDARY
    __hip_atomic_store(config.worker_xcd_ready_count, 0, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_SYSTEM);
#else
    *config.worker_xcd_ready_count = 0;
#endif
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
  // Reset the iteration-release / termination state.
  //
  // These are cudaMemset once at precompute time and were never touched again,
  // which is invisible as long as the process launches the kernel exactly once
  // (the offline demo). A server launches it per request, and launch N+1 then
  // starts with terminate=1 and iter_ready at the previous run's terminal
  // value, so every worker takes the terminate branch at its first queue wrap
  // and the host polls a kernel that has already exited. Workers restart at
  // pc_iter=1 and wait for iter_ready >= pc_iter, so 0 is the correct base:
  // the first END_OF_TASK_GRAPH bump enables iteration 1.
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    if (config.precomp_iter_ready != nullptr) {
      *config.precomp_iter_ready = 0ULL;
    }
    if (config.precomp_terminate != nullptr) {
      *config.precomp_terminate = 0;
    }
    if (config.precomp_dbg_tasks_done != nullptr) {
      *config.precomp_dbg_tasks_done = 0ULL;
    }
  }
  // Per-XCD release flags mirror iter_ready and are polled with ld_nt, so a
  // stale non-zero here would release iteration 1 before it is ready.
  if (config.precomp_iter_xcd_release != nullptr) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < MPK_NUM_XCDS * 16;
         i += blockDim.x * gridDim.x) {
      config.precomp_iter_xcd_release[i] = 0;
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
#ifdef MPK_XCD_SUBSET
  if (blockIdx.x == 0 && threadIdx.x < MPK_XCD_HW) {
    g_xcd_subset_rank[threadIdx.x] = 0;
    g_xcd_sched_rank[threadIdx.x] = 0;
  }
#endif
#endif

  // Send event to scheduler[0]
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    assert(config.all_events[end_of_task_graph_event_pos].event_type ==
           EVENT_END_OF_TASK_GRAPH);
    config.sched_queue_next_free_event_id[0] = 1;
    config.sched_queues[0][0] = end_of_task_graph_event_pos;
    config.sched_queue_last_ready_event_id[0] = 1;
  }
#ifdef MPK_CPX_SCOPE // MDEV_CPX_BOUNDARY: write this kernel's resets back past XCC 0's L2
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
#endif
}

#if defined(MODE_OFFLINE) && defined(MPK_SPEC_VSLOT)
// One real request (0); every token of a pass gets its own batch slot (see the patch notes). Drafts come
// from g_spec_oracle (host: MPK_SPEC_ORACLE / MPK_SPEC_K / MPK_SPEC_CORRUPT).
__device__ long long *g_spec_oracle = nullptr;
#ifdef MPK_SPEC_ORACLE_WINS  // timing tool: oracle drafts win over the drafter's
#define MPK_SPEC_ORACLE_YIELD && g_spec_oracle == nullptr
#else
#define MPK_SPEC_ORACLE_YIELD
#endif
__device__ int g_spec_oracle_n = 0;
__device__ int g_spec_k = 0;
__device__ int g_spec_corrupt = 0;
// MPK_SPEC_CONF > 0 (DFlash): a verify pass checks only the confident prefix of the drafts, while the running
// product of the drafter's chosen-token probabilities stays >= MPK_SPEC_CONF (at least MPK_SPEC_KMIN drafts).
__device__ float g_spec_conf = 0.f;
__device__ int g_spec_kmin = 0;
#if defined(MPK_E3_CONF) && defined(MPK_E3_CONF_LOG)
__device__ float g_e3_conf_log[8];  // the probabilities of the drafts the last verify pass was built from
__device__ int g_e3_conf_logn = 0;
#endif
#if defined(MPK_E3_CONF) && defined(MPK_E3_CONF_D)
__device__ float g_spec_confd = 0.f;  // MPK_E3_CONF_D: the EAGLE3 chain continues while the product >= this
#endif
#ifndef MPK_MULTI_DEV
__device__ __attribute__((aligned(16))) int g_spec_state[4];  // [0] 0 = not admitted, 1 = live, 2 = done; [1] n, [2] start step, [3] kind
__device__ long long g_spec_prev_in[MPK_MAX_NUM_BATCHED_TOKENS];
__device__ int g_spec_pages[MPK_MAX_NUM_PAGES + 1];  // [0] count, then the request's pages in order
__device__ unsigned long long g_spec_stats[4];       // passes, decode passes, drafted, accepted
#endif
#if defined(MPK_TAIL_NO111) && defined(MPK_TAIL_NO111_CHECK)
#error "MPK_TAIL_NO111_CHECK keeps task 111: not with MPK_TAIL_NO111"
#endif
#if (defined(MPK_TAIL_NO111) || defined(MPK_TAIL_NO111_CHECK)) && \
    (!defined(MPK_SPEC_VSLOT) || !defined(MPK_VSLOT_FAST) || !defined(MPK_EAGLE3))
#error "MPK_TAIL_NO111: the MPK_VSLOT_FAST prepare and the EAGLE3 draft"
#endif
#ifndef MPK_VSLOT_FAST
__device__ __forceinline__ bool prepare_next_batch(RuntimeConfig const &config) {
  constexpr int S = MPK_MAX_NUM_BATCHED_TOKENS;
  static_assert(MPK_MAX_NUM_BATCHED_REQUESTS == S, "MPK_SPEC_VSLOT: one batch slot per token");
  int page_queue_head = *config.page_queue_head;
  int page_queue_tail = *config.page_queue_tail;
  int MPK_SPEC_CV *const st = g_spec_state;
  long long *const tok = config.tokens;  // request 0's row
  int const prompt_len = config.prompt_length[0];
  bool done = false;
  // ---- finalize the previous pass
  if (st[0] == 1) {
    int const n = st[1], s = st[2], kind = st[3];
    int step = s + n;
    if (kind == 0) {  // prefill chunk; only a chunk that ends the prompt predicts a new token
      if (step >= prompt_len && step < config.max_seq_length) {
        tok[step] = config.output_tokens[n - 1];
      }
    } else {  // inputs [tok[s], d1..d_(n-1)] at positions s..s+n-1; o_j predicts position s+j+1
      int a = 0;
      while (a < n - 1 && g_spec_prev_in[a + 1] == config.output_tokens[a]) {
        a++;
      }
      step = s;
      for (int j = 0; j <= a; j++) {
        if (s + j + 1 >= config.max_seq_length) {
          break;
        }
        tok[s + j + 1] = config.output_tokens[j];
        step = s + j + 1;
        if (config.output_tokens[j] == config.eos_token_id) {
          break;  // a one-token decode stops right after emitting EOS
        }
      }
#ifdef MPK_SPEC_DEBUG
#ifndef MPK_SPEC_DEBUG_N
#define MPK_SPEC_DEBUG_N 12
#endif
      if (g_spec_stats[1] < MPK_SPEC_DEBUG_N) {
        printf("[SPECDBG] s=%d n=%d in=%lld,%lld,%lld,%lld out=%lld,%lld,%lld,%lld acc=%d\n", s, n,
               g_spec_prev_in[0], n > 1 ? g_spec_prev_in[1] : -1, n > 2 ? g_spec_prev_in[2] : -1,
               n > 3 ? g_spec_prev_in[3] : -1, config.output_tokens[0], n > 1 ? config.output_tokens[1] : -1,
               n > 2 ? config.output_tokens[2] : -1, n > 3 ? config.output_tokens[3] : -1, step - s - 1);
      }
#endif
      g_spec_stats[1]++;
      g_spec_stats[2] += (unsigned long long)(n - 1);
      g_spec_stats[3] += (unsigned long long)(step - s - 1);
#if defined(MPK_E3_CONF) && defined(MPK_E3_CONF_LOG)
      if (g_e3_conf_logn < (MPK_E3_CONF_LOG)) {
        g_e3_conf_logn++;
        printf("[E3CONF] k=%d acc=%d p=%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", n - 1, step - s - 1, g_e3_conf_log[0],
               g_e3_conf_log[1], g_e3_conf_log[2], g_e3_conf_log[3], g_e3_conf_log[4], g_e3_conf_log[5],
               g_e3_conf_log[6]);
      }
#endif
    }
    config.step[0] = step;
    if (step + 1 >= config.max_seq_length ||
        (tok[step] == config.eos_token_id && step >= prompt_len)) {
      done = true;
    }
  }
  if (done) {
    for (int j = 0; j < g_spec_pages[0]; j++) {
      config.page_queue[page_queue_tail % MPK_MAX_NUM_PAGES] = g_spec_pages[1 + j];
      page_queue_tail++;
    }
    g_spec_pages[0] = 0;
    st[0] = 2;
    printf("[SPEC] passes=%llu decode_passes=%llu drafted=%llu accepted=%llu final_step=%d k=%d corrupt=%d\n",
           g_spec_stats[0], g_spec_stats[1], g_spec_stats[2], g_spec_stats[3], config.step[0], g_spec_k,
           g_spec_corrupt);
  }
  if (st[0] == 2) {
    for (int i = 0; i < S; i++) {
      config.request_ids[i] = -1;
    }
    for (int i = 0; i <= S; i++) {
      config.qo_indptr_buffer[i] = 0;
      config.paged_kv_indptr_buffer[i] = 0;
    }
    *config.page_queue_head = page_queue_head;
    *config.page_queue_tail = page_queue_tail;
    return false;
  }
  if (st[0] == 0) {
    if (*config.next_request_id >= config.total_num_requests) {
      return false;
    }
    *config.next_request_id = 1;
    config.step[0] = 0;
    g_spec_pages[0] = 0;
    st[0] = 1;
  }
  int const step = config.step[0];
  // ---- build the next pass
  int n, kind;
  if (step < prompt_len) {
    kind = 0;
    n = min(S, prompt_len - step);
    for (int j = 0; j < n; j++) {
      config.input_tokens[j] = tok[step + j];
      g_spec_prev_in[j] = tok[step + j];
    }
  } else {
    kind = 1;
    int k = min(g_spec_k, S - 1);
    if (step + 1 + k >= config.max_seq_length) {
      k = max(0, config.max_seq_length - step - 2);
    }
#ifdef MPK_DFLASH
    if (kernel::g_df.enabled && g_spec_conf > 0.f) {
      float pr = 1.f;
      int ke = 0;
      for (int i = 1; i <= k; i++) {
        pr *= __uint_as_float((unsigned)__hip_atomic_load(&kernel::g_df.draft_out[kernel::DF_CONF_SLOT + i - 1],
                                                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM));
        if (i > g_spec_kmin && pr < g_spec_conf) {
          break;
        }
        ke = i;
      }
      k = ke;
    }
#endif
#if defined(MPK_EAGLE3) && defined(MPK_E3_CONF) && defined(MPK_E3_CONF_LOG)
    if (kernel::g_e3.enabled)
      for (int i = 0; i < 7; i++)
        g_e3_conf_log[i] = i < kernel::E3_K ? __uint_as_float((unsigned)__hip_atomic_load(
                                                   &kernel::g_e3.draft_out[kernel::E3_CONF_SLOT + i], __ATOMIC_RELAXED,
                                                   __HIP_MEMORY_SCOPE_SYSTEM))
                                             : -1.f;
#endif
#if defined(MPK_EAGLE3) && defined(MPK_E3_CONF)
    if (kernel::g_e3.enabled && g_spec_conf > 0.f) {  // the drafts' running probability product, as the chain cut it
      float pr = 1.f;
      int ke = 0;
      for (int i = 1; i <= k && i <= kernel::E3_K; i++) {
        pr *= __uint_as_float((unsigned)__hip_atomic_load(&kernel::g_e3.draft_out[kernel::E3_CONF_SLOT + i - 1],
                                                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM));
        if (i > g_spec_kmin && pr < g_spec_conf) {
          break;
        }
        ke = i;
      }
      k = ke;
    }
#endif
    n = 1 + k;
    config.input_tokens[0] = tok[step];
    g_spec_prev_in[0] = tok[step];
    for (int i = 1; i <= k; i++) {
      int const q = step + i - prompt_len;
      long long d = (g_spec_oracle != nullptr && q >= 0 && q < g_spec_oracle_n) ? g_spec_oracle[q] : 0;
#ifdef MPK_EAGLE3
      if (kernel::g_e3.enabled MPK_SPEC_ORACLE_YIELD) {
        d = i <= kernel::E3_K  // E3_CHAIN: draft i in draft_out[i - 1]
                ? __hip_atomic_load(&kernel::g_e3.draft_out[i - 1], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)
                : 0;
      }
#endif
#ifdef MPK_DFLASH
      if (kernel::g_df.enabled MPK_SPEC_ORACLE_YIELD) {
        d = i - 1 < kernel::DF_B - 1
                ? __hip_atomic_load(&kernel::g_df.draft_out[i - 1], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)
                : 0;
      }
#endif
      if (g_spec_corrupt > 0 && ((step + i) % g_spec_corrupt) == 0) {
        d = d + 1;  // a wrong id
      }
      config.input_tokens[i] = d;
      g_spec_prev_in[i] = d;
    }
  }
  int const need = (step + n + MPK_PAGE_SIZE - 1) / MPK_PAGE_SIZE;
  while (g_spec_pages[0] < need) {
    g_spec_pages[1 + g_spec_pages[0]] = config.page_queue[page_queue_head % MPK_MAX_NUM_PAGES];
    page_queue_head++;
    g_spec_pages[0]++;
  }
  int np = 0;
  for (int j = 0; j < S; j++) {
    config.qo_indptr_buffer[j] = j < n ? j : n;
    config.paged_kv_indptr_buffer[j] = np;
    if (j < n) {
      config.request_ids[j] = 0;
      int const p = step + j;
      int const pages = p / MPK_PAGE_SIZE + 1;
      for (int t = 0; t < pages; t++) {
        config.paged_kv_indices_buffer[np + t] = g_spec_pages[1 + t];
      }
      config.paged_kv_last_page_len_buffer[j] = p % MPK_PAGE_SIZE + 1;
      np += pages;
    } else {
      config.request_ids[j] = -1;
    }
  }
  config.qo_indptr_buffer[S] = n;
  config.paged_kv_indptr_buffer[S] = np;
  st[1] = n;
  st[2] = step;
  st[3] = kind;
  g_spec_stats[0]++;
  *config.page_queue_head = page_queue_head;
  *config.page_queue_tail = page_queue_tail;
  return true;
}
#else  // MPK_VSLOT_FAST
// The spec-state accesses as relaxed system-scope atomics: the scope of the volatile accesses they replace
// (sc0 sc1) without the s_waitcnt vmcnt(0) the memory legalizer puts after every volatile access, so the
// loads below overlap. The iteration release that follows prepare_next_batch (__threadfence, then the
// release stores) waits for the stores.
#ifdef MPK_TAIL_NO111_CHECK
__device__ unsigned long long g_n111_chk[3];  // rows, published != task 111, lower-worker rule != task 111
#endif
#ifdef MPK_VSF_TS
__device__ unsigned long long g_vsf_ts[5];  // preload, finalize, draft count, build (s_memrealtime ticks), calls
#endif
template <typename T>
__device__ __forceinline__ T vsf_ld(T MPK_SPEC_CV const *p) {
  return __hip_atomic_load(const_cast<T *>(p), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
template <typename T, typename U>
__device__ __forceinline__ void vsf_st(T MPK_SPEC_CV *p, U v) {
  __hip_atomic_store(const_cast<T *>(p), (T)v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
// The same virtual-slot prepare with every load it can need issued first. Under MPK_MULTI_DEV the spec state is a
// system-scope (sc0 sc1) buffer on device 0, and the original walked it in dependent chains (the acceptance compare,
// the confidence product, the drafts, the page list), one memory round trip per step on the iteration boundary.
// Same reads (some extra, all in bounds), same writes, same values.
__device__ __forceinline__ bool prepare_next_batch(RuntimeConfig const &config) {
#ifdef MPK_VSF_PTRS
  // The RuntimeConfig fields, read once: a store through one of its arrays may alias the struct for the
  // compiler, which would otherwise re-read the field (a dependent scalar load) after every store.
  auto const vcf_output_tokens = config.output_tokens;
  auto const vcf_page_queue_head = config.page_queue_head;
  auto const vcf_page_queue_tail = config.page_queue_tail;
  auto const vcf_prompt_length = config.prompt_length;
  auto const vcf_max_seq_length = config.max_seq_length;
  auto const vcf_eos_token_id = config.eos_token_id;
  auto const vcf_step = config.step;
  auto const vcf_page_queue = config.page_queue;
  auto const vcf_request_ids = config.request_ids;
  auto const vcf_qo_indptr_buffer = config.qo_indptr_buffer;
  auto const vcf_paged_kv_indptr_buffer = config.paged_kv_indptr_buffer;
  auto const vcf_next_request_id = config.next_request_id;
  auto const vcf_total_num_requests = config.total_num_requests;
  auto const vcf_input_tokens = config.input_tokens;
  auto const vcf_paged_kv_indices_buffer = config.paged_kv_indices_buffer;
  auto const vcf_paged_kv_last_page_len_buffer = config.paged_kv_last_page_len_buffer;
  auto const vcf_tokens = config.tokens;
#define VCF(f) vcf_##f
#else
#define VCF(f) config.f
#endif
  constexpr int S = MPK_MAX_NUM_BATCHED_TOKENS;
  static_assert(MPK_MAX_NUM_BATCHED_REQUESTS == S, "MPK_SPEC_VSLOT: one batch slot per token");
  constexpr int NPG = MPK_MAX_NUM_PAGES + 1;
#ifdef MPK_VSF_TS
  unsigned long long const vts0 = __builtin_amdgcn_s_memrealtime();
#endif
#ifdef MPK_VSF_HOIST
  // The host-set spec knobs are __device__ globals: read here, with the first batch, not one round trip later.
  int const vh_k = g_spec_k, vh_kmin = g_spec_kmin, vh_cor = g_spec_corrupt, vh_orn = g_spec_oracle_n;
  float const vh_conf = g_spec_conf;
  long long *const vh_or = g_spec_oracle;
#define g_spec_k vh_k
#define g_spec_kmin vh_kmin
#define g_spec_corrupt vh_cor
#define g_spec_oracle_n vh_orn
#define g_spec_conf vh_conf
#define g_spec_oracle vh_or
#endif
  int MPK_SPEC_CV *const st = g_spec_state;
  long long *const tok = VCF(tokens);  // request 0's row
  int const st0 = vsf_ld(&st[0]), st1 = vsf_ld(&st[1]), st2 = vsf_ld(&st[2]), st3 = vsf_ld(&st[3]);
  long long pin[S], otk[S];
#pragma unroll
  for (int j = 0; j < S; j++) {
    pin[j] = vsf_ld(&g_spec_prev_in[j]);
#ifdef MPK_TAIL_NO111
    otk[j] = __hip_atomic_load(&kernel::g_e3.draft_out[kernel::E3_TGT_SLOT + j], __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_SYSTEM);
#else
    otk[j] = VCF(output_tokens)[j];
#endif
  }
  int pg[NPG];
#pragma unroll
  for (int j = 0; j < NPG; j++) {
    pg[j] = vsf_ld(&g_spec_pages[j]);
  }
  unsigned long long sts[4];
#pragma unroll
  for (int j = 0; j < 4; j++) {
    sts[j] = vsf_ld(&g_spec_stats[j]);
  }
#ifdef MPK_EAGLE3
  constexpr int EK = kernel::E3_K;
  bool const e3on = kernel::g_e3.enabled;
  long long e3d[EK];
#ifdef MPK_E3_CONF
  float e3p[EK];
#endif
#pragma unroll
  for (int i = 0; i < EK; i++) {
    e3d[i] = e3on ? __hip_atomic_load(&kernel::g_e3.draft_out[i], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) : 0;
#ifdef MPK_E3_CONF
    e3p[i] = e3on ? __uint_as_float((unsigned)__hip_atomic_load(&kernel::g_e3.draft_out[kernel::E3_CONF_SLOT + i],
                                                                __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM))
                  : 0.f;
#endif
  }
#endif
  int page_queue_head = *VCF(page_queue_head);
  int page_queue_tail = *VCF(page_queue_tail);
  int const prompt_len = VCF(prompt_length)[0];
  bool done = false;
  int step_reg = 0;
  bool tok_known = false;  // tok[step_reg] was written by this call: tok_val
  long long tok_val = 0;
#ifdef MPK_VSF_TS
  asm volatile("" ::"v"(st0), "v"(st1), "v"(st2), "v"(st3), "v"((int)pin[S - 1]), "v"((int)otk[S - 1]),
               "v"(pg[NPG - 1]), "v"((int)sts[3]), "v"(page_queue_head), "v"(page_queue_tail), "v"(prompt_len));
#ifdef MPK_EAGLE3
  asm volatile("" ::"v"((int)e3d[EK - 1]));
#endif
  unsigned long long const vts1 = __builtin_amdgcn_s_memrealtime();
#endif
#ifdef MPK_TAIL_NO111_CHECK
  if (st0 == 1 && e3on) {
#pragma unroll
    for (int j = 0; j < S; j++) {
      if (j < st1) {
        long long const a = __hip_atomic_load(&kernel::g_e3.draft_out[kernel::E3_TGT_SLOT + j], __ATOMIC_RELAXED,
                                              __HIP_MEMORY_SCOPE_SYSTEM);
        long long const b = __hip_atomic_load(&kernel::g_e3.draft_out[kernel::E3_TGT_SLOT + 8 + j], __ATOMIC_RELAXED,
                                              __HIP_MEMORY_SCOPE_SYSTEM);
        g_n111_chk[0] += 1ull;
        g_n111_chk[1] += a != otk[j] ? 1ull : 0ull;
        g_n111_chk[2] += b != otk[j] ? 1ull : 0ull;
      }
    }
  }
#endif
  // ---- finalize the previous pass
  if (st0 == 1) {
    int const n = st1, s = st2, kind = st3;
    int step = s + n;
    if (kind == 0) {  // prefill chunk; only a chunk that ends the prompt predicts a new token
      if (step >= prompt_len && step < VCF(max_seq_length)) {
        long long ov = 0;
#pragma unroll
        for (int j = 0; j < S; j++) {
          if (j == n - 1) ov = otk[j];
        }
        tok[step] = ov;
        tok_known = true;
        tok_val = ov;
      }
    } else {  // inputs [tok[s], d1..d_(n-1)] at positions s..s+n-1; o_j predicts position s+j+1
      int a = 0;
#pragma unroll
      for (int j = 0; j + 1 < S; j++) {
        if (a == j && a < n - 1 && pin[j + 1] == otk[j]) a++;
      }
      step = s;
      bool go = true;  // the original loop's breaks: before j at max_seq_length, after j on EOS
#pragma unroll
      for (int j = 0; j < S; j++) {
        if (go && j <= a) {
          if (s + j + 1 >= VCF(max_seq_length)) {
            go = false;
          } else {
            tok[s + j + 1] = otk[j];
            step = s + j + 1;
            tok_known = true;
            tok_val = otk[j];
            if (otk[j] == VCF(eos_token_id)) {
              go = false;  // a one-token decode stops right after emitting EOS
            }
          }
        }
      }
#ifdef MPK_SPEC_DEBUG
#ifndef MPK_SPEC_DEBUG_N
#define MPK_SPEC_DEBUG_N 12
#endif
      if (sts[1] < MPK_SPEC_DEBUG_N) {
        printf("[SPECDBG] s=%d n=%d in=%lld,%lld,%lld,%lld out=%lld,%lld,%lld,%lld acc=%d\n", s, n, pin[0],
               n > 1 ? pin[1] : -1, n > 2 ? pin[2] : -1, n > 3 ? pin[3] : -1, otk[0], n > 1 ? otk[1] : -1,
               n > 2 ? otk[2] : -1, n > 3 ? otk[3] : -1, step - s - 1);
      }
#endif
      sts[1] += 1ull;
      sts[2] += (unsigned long long)(n - 1);
      sts[3] += (unsigned long long)(step - s - 1);
      vsf_st(&g_spec_stats[1], sts[1]);
      vsf_st(&g_spec_stats[2], sts[2]);
      vsf_st(&g_spec_stats[3], sts[3]);
#if defined(MPK_E3_CONF) && defined(MPK_E3_CONF_LOG)
      if (g_e3_conf_logn < (MPK_E3_CONF_LOG)) {
        g_e3_conf_logn++;
        printf("[E3CONF] k=%d acc=%d p=%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", n - 1, step - s - 1, g_e3_conf_log[0],
               g_e3_conf_log[1], g_e3_conf_log[2], g_e3_conf_log[3], g_e3_conf_log[4], g_e3_conf_log[5],
               g_e3_conf_log[6]);
      }
#endif
    }
    VCF(step)[0] = step;
    step_reg = step;
    if (step + 1 >= VCF(max_seq_length) ||
        ((tok_known ? tok_val : tok[step]) == VCF(eos_token_id) && step >= prompt_len)) {
      done = true;
    }
  }
  if (done) {
#pragma unroll
    for (int j = 0; j + 1 < NPG; j++) {
      if (j < pg[0]) {
        VCF(page_queue)[page_queue_tail % MPK_MAX_NUM_PAGES] = pg[1 + j];
        page_queue_tail++;
      }
    }
    vsf_st(&g_spec_pages[0], 0);
    pg[0] = 0;
    vsf_st(&st[0], 2);
#ifdef MPK_VSF_TS
    if (g_vsf_ts[4] > 0) {
      double const vn = (double)g_vsf_ts[4];
      printf("[VSFTS] n=%llu preload=%.2f finalize=%.2f k=%.2f build=%.2f us\n", g_vsf_ts[4],
             g_vsf_ts[0] / vn / 100.0, g_vsf_ts[1] / vn / 100.0, g_vsf_ts[2] / vn / 100.0, g_vsf_ts[3] / vn / 100.0);
    }
#endif
#ifdef MPK_TAIL_NO111_CHECK
    printf("[N111] rows=%llu mismatch111=%llu mismatch_low=%llu\n", g_n111_chk[0], g_n111_chk[1], g_n111_chk[2]);
#endif
    printf("[SPEC] passes=%llu decode_passes=%llu drafted=%llu accepted=%llu final_step=%d k=%d corrupt=%d\n",
           sts[0], sts[1], sts[2], sts[3], step_reg, g_spec_k, g_spec_corrupt);
  }
#ifdef MPK_VSF_TS
  unsigned long long const vts2 = __builtin_amdgcn_s_memrealtime();
#endif
  int const stn = done ? 2 : st0;
  if (stn == 2) {
    for (int i = 0; i < S; i++) {
      VCF(request_ids)[i] = -1;
    }
    for (int i = 0; i <= S; i++) {
      VCF(qo_indptr_buffer)[i] = 0;
      VCF(paged_kv_indptr_buffer)[i] = 0;
    }
    *VCF(page_queue_head) = page_queue_head;
    *VCF(page_queue_tail) = page_queue_tail;
    return false;
  }
  if (stn == 0) {
    if (*VCF(next_request_id) >= VCF(total_num_requests)) {
      return false;
    }
    *VCF(next_request_id) = 1;
    VCF(step)[0] = 0;
    step_reg = 0;
    vsf_st(&g_spec_pages[0], 0);
    pg[0] = 0;
    vsf_st(&st[0], 1);
  }
  int const step = step_reg;
  // ---- build the next pass
  int n, kind;
  if (step < prompt_len) {
    kind = 0;
    n = min(S, prompt_len - step);
    for (int j = 0; j < n; j++) {
      long long const t = tok[step + j];
      VCF(input_tokens)[j] = t;
      vsf_st(&g_spec_prev_in[j], t);
    }
  } else {
    kind = 1;
    int k = min(g_spec_k, S - 1);
    if (step + 1 + k >= VCF(max_seq_length)) {
      k = max(0, VCF(max_seq_length) - step - 2);
    }
#ifdef MPK_DFLASH
    if (kernel::g_df.enabled && g_spec_conf > 0.f) {
      float pr = 1.f;
      int ke = 0;
      for (int i = 1; i <= k; i++) {
        pr *= __uint_as_float((unsigned)__hip_atomic_load(&kernel::g_df.draft_out[kernel::DF_CONF_SLOT + i - 1],
                                                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM));
        if (i > g_spec_kmin && pr < g_spec_conf) {
          break;
        }
        ke = i;
      }
      k = ke;
    }
#endif
#if defined(MPK_EAGLE3) && defined(MPK_E3_CONF) && defined(MPK_E3_CONF_LOG)
    if (e3on) {
#pragma unroll
      for (int i = 0; i < 7; i++) {
        float lv = -1.f;
#pragma unroll
        for (int q = 0; q < EK; q++) {
          if (q == i) lv = e3p[q];
        }
        g_e3_conf_log[i] = lv;
      }
    }
#endif
#if defined(MPK_EAGLE3) && defined(MPK_E3_CONF)
    if (e3on && g_spec_conf > 0.f) {  // the drafts' running probability product, as the chain cut it
      float pr = 1.f;
      int ke = 0;
      bool cut = false;
#pragma unroll
      for (int i = 1; i <= EK; i++) {
        if (!cut && i <= k) {
          pr *= e3p[i - 1];
          if (i > g_spec_kmin && pr < g_spec_conf) {
            cut = true;
          } else {
            ke = i;
          }
        }
      }
      k = ke;
    }
#endif
    n = 1 + k;
    long long const t0 = tok_known ? tok_val : tok[step];
    VCF(input_tokens)[0] = t0;
    vsf_st(&g_spec_prev_in[0], t0);
#pragma unroll
    for (int i = 1; i < S; i++) {
      if (i <= k) {
        int const q = step + i - prompt_len;
        long long d = (g_spec_oracle != nullptr && q >= 0 && q < g_spec_oracle_n) ? g_spec_oracle[q] : 0;
#ifdef MPK_EAGLE3
        if (e3on MPK_SPEC_ORACLE_YIELD) {
          long long dv = 0;
#pragma unroll
          for (int q2 = 0; q2 < EK; q2++) {
            if (q2 == i - 1) dv = e3d[q2];
          }
          d = i <= EK ? dv : 0;
        }
#endif
#ifdef MPK_DFLASH
        if (kernel::g_df.enabled MPK_SPEC_ORACLE_YIELD) {
          d = i - 1 < kernel::DF_B - 1
                  ? __hip_atomic_load(&kernel::g_df.draft_out[i - 1], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)
                  : 0;
        }
#endif
        if (g_spec_corrupt > 0 && ((step + i) % g_spec_corrupt) == 0) {
          d = d + 1;  // a wrong id
        }
        VCF(input_tokens)[i] = d;
        vsf_st(&g_spec_prev_in[i], d);
      }
    }
  }
#ifdef MPK_VSF_TS
  unsigned long long const vts3 = __builtin_amdgcn_s_memrealtime();
#endif
  int const need = (step + n + MPK_PAGE_SIZE - 1) / MPK_PAGE_SIZE;
  if (pg[0] < need) {  // a new page: the original walk, then reload the list
    while (g_spec_pages[0] < need) {
      g_spec_pages[1 + g_spec_pages[0]] = VCF(page_queue)[page_queue_head % MPK_MAX_NUM_PAGES];
      page_queue_head++;
      g_spec_pages[0]++;
    }
#pragma unroll
    for (int j = 0; j < NPG; j++) {
      pg[j] = vsf_ld(&g_spec_pages[j]);
    }
  }
  int np = 0;
#pragma unroll
  for (int j = 0; j < S; j++) {
    VCF(qo_indptr_buffer)[j] = j < n ? j : n;
    VCF(paged_kv_indptr_buffer)[j] = np;
    if (j < n) {
      VCF(request_ids)[j] = 0;
      int const p = step + j;
      int const pages = p / MPK_PAGE_SIZE + 1;
#pragma unroll
      for (int t = 0; t + 1 < NPG; t++) {
        if (t < pages) {
          VCF(paged_kv_indices_buffer)[np + t] = pg[1 + t];
        }
      }
      VCF(paged_kv_last_page_len_buffer)[j] = p % MPK_PAGE_SIZE + 1;
      np += pages;
    } else {
      VCF(request_ids)[j] = -1;
    }
  }
  VCF(qo_indptr_buffer)[S] = n;
  VCF(paged_kv_indptr_buffer)[S] = np;
  vsf_st(&st[1], n);
  vsf_st(&st[2], step);
  vsf_st(&st[3], kind);
  vsf_st(&g_spec_stats[0], sts[0] + 1ull);
  *VCF(page_queue_head) = page_queue_head;
  *VCF(page_queue_tail) = page_queue_tail;
#ifdef MPK_VSF_TS
  unsigned long long const vts4 = __builtin_amdgcn_s_memrealtime();
  g_vsf_ts[0] += vts1 - vts0;
  g_vsf_ts[1] += vts2 - vts1;
  g_vsf_ts[2] += vts3 - vts2;
  g_vsf_ts[3] += vts4 - vts3;
  g_vsf_ts[4] += 1ull;
#endif
  return true;
}
#undef VCF
#ifdef MPK_VSF_HOIST
#undef g_spec_k
#undef g_spec_kmin
#undef g_spec_corrupt
#undef g_spec_oracle_n
#undef g_spec_conf
#undef g_spec_oracle
#endif
#endif  // MPK_VSLOT_FAST
#endif
#if defined(MODE_OFFLINE) && !defined(MPK_SPEC_VSLOT)
// TODO: parallelize this processing
__device__ __forceinline__ bool
    prepare_next_batch(RuntimeConfig const &config) {
  __shared__ int smem_kv_indices[MPK_MAX_NUM_PAGES];
  int page_queue_head = *config.page_queue_head;
  int page_queue_tail = *config.page_queue_tail;
  // Step 1: finalize previous batch
  for (int i = 0; i < MPK_MAX_NUM_BATCHED_REQUESTS; i++) {
    int16_t request_id = config.request_ids[i];
    if (request_id != -1) {
      // Step 1.1: move output_tokens to tokens
      int step = config.step[request_id];
      int qo_indptr = config.qo_indptr_buffer[i];
      int num_tokens = config.qo_indptr_buffer[i + 1] - qo_indptr;
      int prompt_len = config.prompt_length[request_id];
      for (int j = 0; j < num_tokens; j++) {
        if (step + j + 1 >= prompt_len &&
            step + j + 1 < config.max_seq_length) {
          config.tokens[request_id * MPK_MAX_SEQ_LENGTH + step + j + 1] =
              config.output_tokens[qo_indptr + j];
        }
      }
      config.step[request_id] = step + num_tokens;
#ifdef MPK_ENABLE_PROFILING
      if (config.profiling_num_iters > 0 &&
          step + num_tokens >= config.profiling_num_iters)
#else
      if ((step + num_tokens + 1 >= config.max_seq_length) ||
          ((config.tokens[request_id * MPK_MAX_SEQ_LENGTH + step +
                          num_tokens] == config.eos_token_id) &&
           (step + num_tokens >= prompt_len)))
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

  // Step 2.5: fair-share prefill budget.
  //
  // MPK_MAX_NUM_BATCHED_TOKENS is both the admission pool and the GEMM M
  // dimension, so it cannot be raised to make room (budget 8 is
  // nondeterministic and slower per iteration, 16 hangs). With the pool fixed
  // at B, letting one prefiller take up to MPK_MAX_TOKENS_PER_REQUEST (28)
  // drains it: request 0 grabs the whole pool, requests 1..B-1 get nothing
  // and enter only after it finishes prefilling. Three 512-token prompts
  // therefore prefill *serially* and the "bs=3" decode tail is 255 iterations
  // of one decoder, not three -- 1.803 tok/iter instead of 3.000.
  //
  // Capping each prefiller at its share of the pool makes the requests enter
  // and advance in lockstep. Prefill takes more iterations (each request gets
  // budget/demand tokens per pass instead of up to 28); decode, which is what
  // TPOT measures, goes fully concurrent. At B=1 demand is 1 and the share is
  // the whole budget, so this is structurally a no-op there.
  int prefill_share = MPK_MAX_TOKENS_PER_REQUEST;
#ifndef MPK_NO_FAIR_PREFILL
  {
    int num_live = 0;
    for (int i = 0; i < MPK_MAX_NUM_BATCHED_REQUESTS; i++) {
      if (config.request_ids[i] != -1) {
        num_live++;
      }
    }
    int queued = config.total_num_requests - *config.next_request_id;
    if (queued < 0) {
      queued = 0;
    }
    int demand = num_live + queued;
    if (demand > MPK_MAX_NUM_BATCHED_REQUESTS) {
      demand = MPK_MAX_NUM_BATCHED_REQUESTS;
    }
    if (demand < 1) {
      demand = 1;
    }
    int share = MPK_MAX_NUM_BATCHED_TOKENS / demand;
    if (share < 1) {
      share = 1;
    }
    if (share < prefill_share) {
      prefill_share = share;
    }
  }
#endif

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
        // overflow, and to this request's fair share of the pool
        num_new_tokens = min(num_new_tokens,
                             min(prefill_share,
                                 MPK_MAX_NUM_BATCHED_TOKENS - num_tokens));
      } else {
        // Decode requests
        num_new_tokens = min(1, MPK_MAX_NUM_BATCHED_TOKENS - num_tokens);
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
    // Prefill request: cap per-request tokens to avoid attention LDS overflow,
    // and to this request's fair share of the pool (see Step 2.5)
    int num_new_tokens = min(config.prompt_length[next_request_id],
                             min(prefill_share,
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
// MPK_END_DIRECT under MPK_SPEC_VSLOT: the virtual-slot prepare above is single-threaded and uses no
// LDS, so the argmax worker runs it as is.
#if defined(MODE_OFFLINE) && defined(MPK_END_DIRECT) && defined(MPK_SPEC_VSLOT)
__device__ __forceinline__ bool prepare_next_batch_w(RuntimeConfig const &config) {
  return prepare_next_batch(config);
}
// MPK_END_DIRECT_V2: the worker-side copy of prepare_next_batch (MPK_END_DIRECT).
#elif defined(MODE_OFFLINE) && defined(MPK_END_DIRECT)
// TODO: parallelize this processing
__device__ __forceinline__ bool
    prepare_next_batch_w(RuntimeConfig const &config) {
  int smem_kv_indices[MPK_MAX_NUM_PAGES]; // local: no worker LDS
  int page_queue_head = *config.page_queue_head;
  int page_queue_tail = *config.page_queue_tail;
  // Step 1: finalize previous batch
  for (int i = 0; i < MPK_MAX_NUM_BATCHED_REQUESTS; i++) {
    int16_t request_id = config.request_ids[i];
    if (request_id != -1) {
      // Step 1.1: move output_tokens to tokens
      int step = config.step[request_id];
      int qo_indptr = config.qo_indptr_buffer[i];
      int num_tokens = config.qo_indptr_buffer[i + 1] - qo_indptr;
      int prompt_len = config.prompt_length[request_id];
      for (int j = 0; j < num_tokens; j++) {
        if (step + j + 1 >= prompt_len &&
            step + j + 1 < config.max_seq_length) {
          config.tokens[request_id * MPK_MAX_SEQ_LENGTH + step + j + 1] =
              config.output_tokens[qo_indptr + j];
        }
      }
      config.step[request_id] = step + num_tokens;
#ifdef MPK_ENABLE_PROFILING
      if (config.profiling_num_iters > 0 &&
          step + num_tokens >= config.profiling_num_iters)
#else
      if ((step + num_tokens + 1 >= config.max_seq_length) ||
          ((config.tokens[request_id * MPK_MAX_SEQ_LENGTH + step +
                          num_tokens] == config.eos_token_id) &&
           (step + num_tokens >= prompt_len)))
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

  // Step 2.5: fair-share prefill budget.
  //
  // MPK_MAX_NUM_BATCHED_TOKENS is both the admission pool and the GEMM M
  // dimension, so it cannot be raised to make room (budget 8 is
  // nondeterministic and slower per iteration, 16 hangs). With the pool fixed
  // at B, letting one prefiller take up to MPK_MAX_TOKENS_PER_REQUEST (28)
  // drains it: request 0 grabs the whole pool, requests 1..B-1 get nothing
  // and enter only after it finishes prefilling. Three 512-token prompts
  // therefore prefill *serially* and the "bs=3" decode tail is 255 iterations
  // of one decoder, not three -- 1.803 tok/iter instead of 3.000.
  //
  // Capping each prefiller at its share of the pool makes the requests enter
  // and advance in lockstep. Prefill takes more iterations (each request gets
  // budget/demand tokens per pass instead of up to 28); decode, which is what
  // TPOT measures, goes fully concurrent. At B=1 demand is 1 and the share is
  // the whole budget, so this is structurally a no-op there.
  int prefill_share = MPK_MAX_TOKENS_PER_REQUEST;
#ifndef MPK_NO_FAIR_PREFILL
  {
    int num_live = 0;
    for (int i = 0; i < MPK_MAX_NUM_BATCHED_REQUESTS; i++) {
      if (config.request_ids[i] != -1) {
        num_live++;
      }
    }
    int queued = config.total_num_requests - *config.next_request_id;
    if (queued < 0) {
      queued = 0;
    }
    int demand = num_live + queued;
    if (demand > MPK_MAX_NUM_BATCHED_REQUESTS) {
      demand = MPK_MAX_NUM_BATCHED_REQUESTS;
    }
    if (demand < 1) {
      demand = 1;
    }
    int share = MPK_MAX_NUM_BATCHED_TOKENS / demand;
    if (share < 1) {
      share = 1;
    }
    if (share < prefill_share) {
      prefill_share = share;
    }
  }
#endif

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
        // overflow, and to this request's fair share of the pool
        num_new_tokens = min(num_new_tokens,
                             min(prefill_share,
                                 MPK_MAX_NUM_BATCHED_TOKENS - num_tokens));
      } else {
        // Decode requests
        num_new_tokens = min(1, MPK_MAX_NUM_BATCHED_TOKENS - num_tokens);
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
    // Prefill request: cap per-request tokens to avoid attention LDS overflow,
    // and to this request's fair share of the pool (see Step 2.5)
    int num_new_tokens = min(config.prompt_length[next_request_id],
                             min(prefill_share,
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
#if defined(MPK_END_DIRECT) && !defined(MODE_OFFLINE)
#define prepare_next_batch_w prepare_next_batch
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

#ifdef MPK_MULTI_DEV // MDEV_CHECKER: each device launches its share of the workers
  assert(gridDim.x * MPK_MULTI_DEV == config.num_workers);
#else
  assert(gridDim.x == config.num_workers * MPK_XCD_OVERSUB);
#endif
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
#ifdef MPK_END_DIRECT
// (iteration << 1) | continue, written by the worker that completed that
// iteration before it pushes the end event to the scheduler.
__device__ unsigned long long g_end_direct;
#endif
#if defined(MPK_END_DIRECT) && defined(MODE_ONLINE_NOTOKEN)
#error "MPK_END_DIRECT does not support MODE_ONLINE_NOTOKEN"
#endif
#if defined(MPK_END_FAST) && !defined(MPK_END_DIRECT)
#error "MPK_END_FAST needs MPK_END_DIRECT"
#endif
#if defined(MPK_END_DIRECT_EMB) && \
    !(defined(MPK_END_DIRECT) && defined(MPK_EMB_FIRST) && defined(MPK_PC_DROP_BEGIN))
#error "MPK_END_DIRECT_EMB needs MPK_END_DIRECT, MPK_EMB_FIRST and MPK_PC_DROP_BEGIN"
#endif
#ifdef MPK_ITER_TS
// MPK_ITER_TS_V1: iteration-boundary timestamps, ns (s_memrealtime x 10).
__device__ unsigned long long g_its_argmax_end;
__device__ unsigned long long g_its_rel_ts;
__device__ unsigned long long g_its_sum[8];
__device__ unsigned long long g_its_n;
__device__ unsigned long long g_its_emb_sum[2];
__device__ unsigned long long g_its_emb_n;
// MPK_ITER_TS2_V1
__device__ unsigned long long g_its_tr_sum[5];
__device__ unsigned long long g_its_tr_n;
__device__ unsigned long long g_its_push_end;
#endif
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
#if defined(MPK_PHASE_SLOTS) && defined(MPK_PHASE_LDS)
  mpk_phase_lds_init();
#endif

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
#ifdef MPK_CPX_SCOPE // MDEV_CPX_BOUNDARY
      atomicAdd_system(config.worker_xcd_ready_count, 1);
#else
      atomicAdd(config.worker_xcd_ready_count, 1);
#endif
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

#ifdef MPK_AID_SPLIT_FLAGS
  // Read the ml pointer tables out of this XCD's own AID. Indexing is
  // unchanged -- each replica is a full copy, so only the base moves. Falls
  // back to the shared table when the allocator was unavailable.
  void **const ml_in_tab = (g_aid_ml_in[xcd_id >> 2] != nullptr)
                               ? g_aid_ml_in[xcd_id >> 2]
                               : config.ml_input_table;
  void **const ml_out_tab = (g_aid_ml_out[xcd_id >> 2] != nullptr)
                                ? g_aid_ml_out[xcd_id >> 2]
                                : config.ml_output_table;
#else
  void **const ml_in_tab = config.ml_input_table;
  void **const ml_out_tab = config.ml_output_table;
#endif
  (void)ml_in_tab;
  (void)ml_out_tab;

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
#if defined(MPK_PC_DROP_BEGIN) && !defined(MPK_EMB_FIRST)
#error "MPK_PC_DROP_BEGIN needs MPK_EMB_FIRST (the embedding's acquire)"
#endif
#ifdef MPK_EMB_FIRST
    // The graph-begin marker carries no data: run the embedding it gates
    // first on this worker; its wait on the marker is skipped at dispatch.
    bool const pc_swap =
        tmpl_len >= 2 && tmpl_src[0] == 1 &&
        config.all_tasks[tmpl_src[1]].task_type == TASK_EMBEDDING;
#ifdef MPK_PC_DROP_BEGIN
#ifdef MPK_END_DIRECT_EMB
    // The embedding moves to the worker that runs the argmax reduce. Thread 0
    // decides and copies (no LDS: the worker is at the static-LDS limit); the
    // ordered scan gives every worker the same two slots.
    int pc_drop = 0;
    int pc_add = 0;
    if (threadIdx.x == 0) {
      int emb_slot = -1;
      int amx_slot = -1;
      for (int s2 = 0; s2 < MPK_NUM_XCDS * wpx && (emb_slot < 0 || amx_slot < 0);
           s2++) {
        size_t const *t2 =
            config.precomp_xcd_template + (size_t)s2 * config.precomp_max_tpw;
        int const l2 = config.precomp_xcd_template_len[s2];
        if (emb_slot < 0 && l2 >= 2 && t2[0] == 1 &&
            config.all_tasks[t2[1]].task_type == TASK_EMBEDDING) {
          emb_slot = s2;
        }
        for (int j = 0; amx_slot < 0 && j < l2; j++) {
          if (config.all_tasks[t2[j]].task_type == TASK_ARGMAX_REDUCE) {
            amx_slot = s2;
          }
        }
      }
      bool const move =
          emb_slot >= 0 && amx_slot >= 0 &&
          (amx_slot == emb_slot ||
           config.precomp_xcd_template_len[amx_slot] + 1 <= config.precomp_max_tpw);
      pc_drop = pc_swap ? ((move && tmpl_flat == emb_slot) ? 2 : 1) : 0;
      pc_add = (move && tmpl_flat == amx_slot) ? 1 : 0;
      for (int i = pc_drop; i < tmpl_len; i++) {
        pc_my_queue[i - pc_drop + pc_add] = tmpl_src[i];
      }
      if (pc_add) {
        pc_my_queue[0] = config.precomp_xcd_template[(size_t)emb_slot *
                                                         config.precomp_max_tpw +
                                                     1];
      }
    }
#else
    // MPK_PC_DROP_BEGIN_V1: with the swap, entry 0 of the template is the marker;
    // drop it and shift the rest down by one.
    int const pc_drop = pc_swap ? 1 : 0;
    int const pc_add = 0;
    for (int i = threadIdx.x + pc_drop; i < tmpl_len; i += blockDim.x) {
      pc_my_queue[i - pc_drop] = tmpl_src[i];
    }
#endif
#else
    for (int i = threadIdx.x; i < tmpl_len; i += blockDim.x) {
      pc_my_queue[pc_swap && i < 2 ? 1 - i : i] = tmpl_src[i];
    }
#endif
#else
    // Parallel copy: each thread copies a chunk
    for (int i = threadIdx.x; i < tmpl_len; i += blockDim.x) {
      pc_my_queue[i] = tmpl_src[i];
    }
#endif
    __syncthreads();
    if (threadIdx.x == 0) {
#if defined(MPK_EMB_FIRST) && defined(MPK_PC_DROP_BEGIN)
      config.precomp_queue_len[worker_id] = tmpl_len - pc_drop + pc_add;
      pc_pos = 0;
      pc_iter = 1;
      pc_my_len = tmpl_len - pc_drop + pc_add;
#else
      config.precomp_queue_len[worker_id] = tmpl_len;
      pc_pos = 0;
      pc_iter = 1;
      pc_my_len = tmpl_len;
#endif
      pc_terminated = 0;
      pc_xcd_rank = my_rank;
      pc_has_begin = (tmpl_len > 0 && tmpl_src[0] == 1)
                         ? 1
                         : 0; // position 1 = begin_task_graph
#ifdef MPK_EMB_FIRST
      if (pc_swap) {
        pc_has_begin = 2; // embedding moved ahead of the marker
      }
#ifdef MPK_END_DIRECT_EMB
      if (pc_add) {
        pc_has_begin = 2; // keeps the embedding's acquire on its new worker
      }
#endif
#endif
#ifndef MPK_PC_INIT_N
#define MPK_PC_INIT_N 8
#endif
#ifdef MPK_RESID_MAP
      MPK_RESID_PUT(256 + worker_id, ((unsigned)my_rank << 16) | (unsigned)tmpl_len);
#endif
      if (worker_id < MPK_PC_INIT_N) {
        int const t0 = tmpl_len > 0 ? (int)tmpl_src[0] : -1;
        int const t1 = tmpl_len > 1 ? (int)tmpl_src[1] : -1;
        int const t2 = tmpl_len > 2 ? (int)tmpl_src[2] : -1;
        printf("[PC_INIT] worker=%d block=%d xcd=%d rank=%d wpx=%d tmpl_len=%d "
               "tmpl=%d,%d,%d types=%d,%d,%d\n",
               worker_id, (int)blockIdx.x, xcd_id, my_rank, wpx, tmpl_len, t0, t1, t2,
               t0 >= 0 ? (int)config.all_tasks[t0].task_type : -1,
               t1 >= 0 ? (int)config.all_tasks[t1].task_type : -1,
               t2 >= 0 ? (int)config.all_tasks[t2].task_type : -1);
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

#ifdef MPK_ITER_TS
  unsigned long long its_gate_exit = 0;
  int its_wrap = 0;
#endif
#ifdef MPK_END_FAST
  // MPK_END_FAST_V1: iteration this worker released itself; deferred end event.
  unsigned long long ed_rel_iter = 0;
  long long ed_defer_ev = -1;
  int ed_defer_wait = 0;
#endif
  while (true) {
    // fetch next task from a task queue if task_descs is empty
    if (queue_pos == queue_len) {
#ifdef MPK_PRECOMPUTED_DISPATCH
      // ===== Pre-filled queue path =====
#ifdef MPK_PC_DESC_PREFETCH
      // The descriptors are static (positions never change between
      // iterations): wrap first, then every thread starts copying the next
      // batch into LDS before thread 0 waits at the iteration gate.
      if (threadIdx.x == 0 && pc_pos >= pc_my_len) {
        pc_pos = 0;
        pc_iter++;
#ifdef MPK_ITER_TS
        its_wrap = 1;
#endif
      }
      __syncthreads();
      {
        int const pf_n = min(pc_my_len - pc_pos, TASK_DESCS_BUFFER_LENGTH);
        constexpr int PF_TS = sizeof(TaskDesc) / 16;
        for (int i = threadIdx.x; i < pf_n * PF_TS; i += blockDim.x) {
          int const ti = i / PF_TS;
          int const off = i % PF_TS;
          load_smem(reinterpret_cast<char *>(task_descs) + i * 16,
                    reinterpret_cast<char *>(
                        config.all_tasks + pc_my_queue[pc_pos + ti]) +
                        off * 16);
        }
      }
#endif
      if (threadIdx.x == 0) {
#ifndef MPK_PC_DESC_PREFETCH
        if (pc_pos >= pc_my_len) {
          // All tasks done for this iteration — wrap around
          pc_pos = 0;
          pc_iter++;
#ifdef MPK_ITER_TS
          its_wrap = 1;
#endif
        }
#endif
#ifdef MPK_END_FAST
        // This worker released iteration pc_iter itself, and its prepare said
        // the run continues: the gate is open and nothing terminates it.
        if (ed_rel_iter != (unsigned long long)pc_iter) {
#endif
        // All workers must wait at iteration boundary because orphan tasks
        // (e.g., embed) have no dependency event and would execute immediately,
        // corrupting activation buffers if the previous iteration isn't done.
#ifdef MPK_ITER_XCD_POLL
        // 248 worker blocks polling one spanning-NC u64 is the measured
        // contention cliff (2426 ns at 1 poller -> 52251 at 256). The
        // scheduler already mirrors iter_ready into a per-XCD line right after
        // bumping it; read that instead so the pollers spread over eight
        // lines. All eight slots hold the same value, so which one a worker
        // reads cannot affect correctness -- it only changes contention, which
        // matters because the worker->XCD map is rotated by one.
        // Lags the counter by one store: can delay a release, never grant one
        // early. MPK_LD_GATE2 is ld_sys_s32 (sc0 sc1, past L2); an nt load
        // would never observe the st_wt publish at all.
        int *_itrel = config.precomp_iter_xcd_release;
        if (_itrel != nullptr) {
          unsigned _xcc;
          MPK_GET_XCC_ID32(_xcc);
#ifdef MPK_ITER_AID
          int *_aid = mpk_aid_iter_slot((int)(_xcc & (MPK_NUM_XCDS - 1)));
          int *_slot = _aid ? _aid
                            : _itrel + (int)(_xcc & (MPK_NUM_XCDS - 1)) * 16;
#else
          int *_slot = _itrel + (int)(_xcc & (MPK_NUM_XCDS - 1)) * 16;
#endif
          // BOTH loads must leave the spin. Redirecting only iter_ready is
          // worthless: vmcnt(0) waits for the MAXIMUM of its outstanding
          // loads, so one remaining sc0 sc1 terminate load pins the whole
          // iteration (measured: v1 gave -1.22%, t=-0.59).
          //
          // Checking terminate every 64th trip is safe because the terminate
          // path also bumps iter_ready to release parked workers -- so the
          // mirror wakes us anyway -- and MPK_TERM_RECHECK re-tests it right
          // after this loop. Worst case a terminating worker takes up to 63
          // extra sleeps to notice.
          unsigned _it = 0;
#if defined(MPK_ITER_GATE_SC1) && defined(MPK_ITER_AID)
          // MPK_TAILFIX_V1. The AID slot is an MTYPE_RW replica written with
          // write-through stores, which probe the pollers' L2 copies out inside
          // the AID: poll it like the in-layer replica flags (sc1, an L2 hit
          // until the publish lands) instead of 124 pollers per line past L2.
          while ((unsigned)(_aid ? ld_aid_s32(_slot) : MPK_LD_GATE2(_slot)) <
                 (unsigned)pc_iter) {
#else
          while ((unsigned)MPK_LD_GATE2(_slot) < (unsigned)pc_iter) {
#endif
#if defined(MPK_GATE_RESCUE) && defined(MPK_ITER_GATE_SC1) && defined(MPK_ITER_AID)
            // MPK_GATE_RESCUE_V1: the sc1 poll may be served from this XCD's L2 copy of the slot, which
            // only a hardware probe drops. Every MPK_GATE_RESCUE trips, read the slot past L2: a release
            // found there but not in L2 is taken, and counted per XCC (residency slots 1088 + xcc, the
            // iteration at 1104 + xcc).
            if (_aid != nullptr &&
                (_it & ((unsigned)(MPK_GATE_RESCUE)-1u)) == (unsigned)(MPK_GATE_RESCUE)-1u &&
                (unsigned)ld_sys_s32(_slot) >= (unsigned)pc_iter) {
#ifdef MPK_RESID_MAP
              mpk_resid_add(1088 + (int)(_xcc & 7u), 1u);
              mpk_resid_put(1104 + (int)(_xcc & 7u), (unsigned)pc_iter);
#endif
              break;
            }
#endif
            if ((_it++ & 63u) == 63u &&
                __atomic_load_n(config.precomp_terminate, __ATOMIC_RELAXED)) {
              pc_terminated = 1;
              break;
            }
            __builtin_amdgcn_s_sleep(1);
          }
        } else
#endif
        while (__atomic_load_n(config.precomp_iter_ready, __ATOMIC_RELAXED) <
               pc_iter) {
          if (__atomic_load_n(config.precomp_terminate, __ATOMIC_RELAXED)) {
            pc_terminated = 1;
            break;
          }
          __builtin_amdgcn_s_sleep(1);
        }
#ifdef MPK_TERM_RECHECK
        // The terminate path bumps precomp_iter_ready specifically to release
        // workers parked in the loop above, so a worker whose condition is
        // satisfied by that bump can leave without ever evaluating the check
        // inside it. It then loads tasks for an iteration that will never be
        // produced and blocks forever in the dependency wait, which has no
        // terminate check of its own. Re-test here, where the existing
        // pc_terminated exit below can still catch it.
        if (!pc_terminated &&
            __atomic_load_n(config.precomp_terminate, __ATOMIC_RELAXED)) {
          pc_terminated = 1;
        }
#endif
#ifdef MPK_RESID_MAP
        MPK_RESID_PUT(512 + worker_id, ((unsigned)pc_terminated << 24) | (unsigned)pc_iter);
#endif
#ifdef MPK_END_FAST
        }
#endif
#ifdef MPK_ITER_TS
        if (its_wrap) {
          its_gate_exit = get_wallclock_ns();
          its_wrap = 0;
        }
#endif
      }
      __syncthreads();
      if (pc_terminated) {
#ifdef MPK_RESID_MAP
        if (threadIdx.x == 0) {
          MPK_RESID_PUT(768 + worker_id, (unsigned)pc_iter);
        }
#endif
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
#ifndef MPK_PC_DESC_PREFETCH
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
#endif
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

#ifdef MPK_LM_PF0
    // The LM head's group-zero weights stream during its dependency wait.
    if (task_desc->task_type == TASK_GANG_RMSNORM_LINEAR_MXFP4_BIAS_ARGMAX_MI300) {
      int const _pf_r = block_xcd_local_rank;
      _mpk_lm_pf0(task_desc,
                  (_pf_r >= 0 &&
                   _pf_r < (int)task_desc->task_metadata.n_tile_count)
                      ? (int)task_desc->task_metadata.n_tile_start + _pf_r
                      : -1);
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
#ifdef MPK_EMB_FIRST
      if (pc_has_begin == 2 && task_desc->task_type == TASK_EMBEDDING) {
        // Its only dependency is the graph-begin marker, now queued behind
        // it. Keep the acquire the wait would have ended with.
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, MPK_DEV_FENCE_ACQ);
      } else
#endif
      if (task_desc->dependent_event != EVENT_INVALID_ID) {
        EventId event_id = task_desc->dependent_event;
        assert(get_event_gpu_id(event_id) == config.my_gpu_id);
        size_t event_index = get_event_position_index(event_id);
        EventCounter needed_counts =
            static_cast<EventCounter>(
                  MPK_LD_EVCFG(&config.all_event_num_triggers[event_index])) *
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
#ifdef USE_NVSHMEM
          nvshmem_signal_wait_until(
              reinterpret_cast<uint64_t *>(
                  &config.all_event_counters[event_index]),
              NVSHMEM_CMP_EQ,
              needed_counts);
#endif
        } else {
#ifdef MPK_ENABLE_TIMING
          unsigned long long dep_start = clock64();
#endif
          // Single ACQUIRE load — dependency guaranteed satisfied by scheduler
          // dispatch chain.
          actual_counts =
              MPK_LD_EVENT(&config.all_event_counters[event_index]);
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
          // Agent-scope acquire fence (GPU-only, not system scope)
#ifdef MPK_SCHED_WG_FENCE
          __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
#elif defined(MPK_SCHED_ONE_FENCE)
          // If the poll below runs, its trailing fence is the acquire.
          if (actual_counts >= needed_counts) {
#ifdef MPK_LM_LEADER_INV
            mpk_lm_dep_acquire(task_desc->task_type ==
                                   TASK_GANG_RMSNORM_LINEAR_MXFP4_BIAS_ARGMAX_MI300,
                               (int)pc_iter);
#else
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, MPK_DEV_FENCE_ACQ);
#endif
          }
#else
#ifdef MPK_LM_LEADER_INV
          mpk_lm_dep_acquire(task_desc->task_type ==
                                   TASK_GANG_RMSNORM_LINEAR_MXFP4_BIAS_ARGMAX_MI300,
                               (int)pc_iter);
#else
          __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, MPK_DEV_FENCE_ACQ);
#endif
#endif
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
            unsigned _ev_spin = 0;
#ifdef MPK_P7_SKIP
            // bit 8: skip the dependency wait entirely (timing probe)
            bool const _dep_skip = ((MPK_P7_SKIP) & 8) != 0;
#else
            bool const _dep_skip = false;
#endif
            while (!_dep_skip && actual_counts < needed_counts) {
              actual_counts = MPK_EVCTR_POLL(
                  config.all_event_counters, event_index, _ev_spin++);
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
              __builtin_amdgcn_s_sleep(1);
#endif
#ifdef MPK_ENABLE_TIMING
              total_dep_wait_iters++;
#endif
            }
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
#ifdef MPK_SCHED_WG_FENCE
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
#elif defined(MPK_LM_LEADER_INV)
            mpk_lm_dep_acquire(task_desc->task_type ==
                                   TASK_GANG_RMSNORM_LINEAR_MXFP4_BIAS_ARGMAX_MI300,
                               (int)pc_iter);
#else
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, MPK_DEV_FENCE_ACQ);
#endif
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

#ifdef MPK_ENABLE_PROFILING
    if (task_desc->task_type != TASK_TERMINATE) {
      PROFILER_EVENT_START(task_desc->task_type, task_counter);
    }
#endif
#ifdef MPK_ITER_TS
    if (threadIdx.x == 0 && task_desc->task_type == TASK_EMBEDDING &&
        its_gate_exit != 0) {
      unsigned long long const _t = get_wallclock_ns();
      unsigned long long const _rel =
          __atomic_load_n(&g_its_rel_ts, __ATOMIC_RELAXED);
      if (_rel != 0 && _rel <= its_gate_exit &&
          its_gate_exit - _rel < 1000000ull) {
        atomicAdd(&g_its_emb_sum[0], its_gate_exit - _rel);
        atomicAdd(&g_its_emb_sum[1], _t - its_gate_exit);
        atomicAdd(&g_its_emb_n, 1ull);
      }
      its_gate_exit = 0;
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
        // Inline per-iteration trace. Under MPK_QUIET_FWDPASS this is off: a
        // device printf drains over PCIe inside the iteration it reports, so
        // on the serving path it perturbs the very inter-token latency the
        // benchmark is measuring.
#ifndef MPK_QUIET_FWDPASS
        if (prev_begin_clk != 0 &&
            (fwd_pass_count < 10 || fwd_pass_count % 50 == 0)) {
          printf("[FWD_PASS] iter=%d time_ms=%.3f num_active_tokens=%d\n",
                 fwd_pass_count,
                 (double)(cur_clk - prev_begin_clk) / 1000000.0,
                 num_active_tokens);
        }
#else
        (void)num_active_tokens;
#endif
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

#if defined(MPK_FUSED_LAYER_BATCHING)
        // ===== Fused-layer batching fast path =====
        if (task_desc->task_type == TASK_GANG_FULL_LAYER_FUSED_MI300 ||
            task_desc->task_type ==
                TASK_GANG_FULL_LAYER_WITH_LMHEAD_FUSED_MI300) {

#ifdef MPK_PRECOMPUTED_DISPATCH
          size_t const flb_iter = pc_iter;
#else
          size_t const flb_iter = get_task_iteration_num(task_ids[queue_pos]);
#endif

          if (config.ml_num_layers > 0) {
            // ===== Multi-layer all-fused: one task executes all layers =====
            int ml_n_tile_start = (int)task_desc->task_metadata.n_tile_start;
            int ml_n_tile_count = (int)task_desc->task_metadata.n_tile_count;

#ifdef MPK_ML_TABLE_PREFETCH
            // ── Hide the inter-layer pointer-table read ────────────────────
            //
            // The copy below is 41 pointers (28 in + 13 out) read from two
            // device-global tables, and it sits between the previous layer's
            // Phase 9 gate and the next layer's first instruction -- pure
            // serial latency with nothing to overlap it. Measured at ctx 512
            // it is 765 ns of a 1679 ns inter-layer prologue, 0.028 ms over
            // 36 layers. It is one dependent HBM round trip, not bandwidth:
            // 328 bytes spread over 256 threads.
            //
            // The tables are built once on the host and are invariant for the
            // life of the kernel, so layer ml+1's row can be read during layer
            // ml's body instead. Each thread holds at most one input and one
            // output pointer (MAX_INPUTS_PER_TASK is 28, MAX_OUTPUTS 13, both
            // well under the 256-thread block), so the whole prefetch is two
            // VGPR pairs carried across the layer call.
            //
            // Correctness rests on the tables being read-only after launch and
            // on xcd_id being fixed for the life of the task -- the same
            // property MPK_PREFETCH_NEXT_QKV already relies on. If either ever
            // becomes false this must go back to a post-gate read.
            void *_pf_in = nullptr;
            void *_pf_out = nullptr;
            void *_pf_cur = nullptr;
#ifdef MPK_IL_Q4REG
#if defined(MPK_IL_FAST) || defined(MPK_ML_TABLE_INLAYER)
#error "MPK_IL_Q4REG replaces the slot-24 source of the plain prefetch path"
#endif
            void *_pf_q4 = nullptr;  // thread 4: slot 4 of layer ml+1
#endif
#ifdef MPK_IL_CACHE2
#if !defined(MPK_FUSED_INLINE)
#error "MPK_IL_CACHE2 needs MPK_FUSED_INLINE (the ml-only dispatch carries qo_len)"
#endif
            int _pf_var = 0;      // variant id of layer ml+1, read during layer ml
            int _pf_var_cur = 0;
            int ml_qo_len = 0;    // qo_indptr_buffer[1], read at layer 0
#endif
#ifdef MPK_ML_STAGE
#if !defined(MPK_PREFETCH_NEXT_QKV) || defined(MPK_IL_FAST) || defined(MPK_IL_Q4REG) || \
    defined(MPK_IL_CACHE2) || defined(MPK_ML_TABLE_INLAYER)
#error "MPK_ML_STAGE: plain prefetch path with MPK_PREFETCH_NEXT_QKV only"
#endif
            // [0, 28) inputs, [28, 41) outputs, [42] variant id; q4 = row
            // ml+2's slot 4, slot 24 of the layer after this one.
            __shared__ __attribute__((aligned(16))) unsigned long long s_mls[44];
            __shared__ __attribute__((aligned(16))) unsigned long long s_mls_q4[2];
            bool const _ms_on =
                g_mpk_mlstage_rt != 0 && config.ml_num_layers > 1 &&
                blockDim.x >= 64 &&
                task_desc->task_type == TASK_GANG_FULL_LAYER_FUSED_MI300;
            unsigned const _ms_lds = __builtin_amdgcn_readfirstlane((unsigned)(uintptr_t)(
                (__attribute__((address_space(3))) unsigned long long *)s_mls));
            unsigned const _ms_lds_q4 = __builtin_amdgcn_readfirstlane((unsigned)(uintptr_t)(
                (__attribute__((address_space(3))) unsigned long long *)s_mls_q4));
            uint32_t const _ms_rows = (uint32_t)((xcd_id + 1) * config.ml_num_layers);
            i32x4_t const _ms_rin = make_w_buffer_rsrc(
                (void const *)ml_in_tab, _ms_rows * (uint32_t)(MAX_INPUTS_PER_TASK * 8));
            i32x4_t const _ms_rout = make_w_buffer_rsrc(
                (void const *)ml_out_tab, _ms_rows * (uint32_t)(MAX_OUTPUTS_PER_TASK * 8));
            i32x4_t const _ms_rvar = make_w_buffer_rsrc(
                (void const *)config.ml_variant_ids, (uint32_t)(config.ml_num_layers * 4));
#endif
            bool const _pf_ok =
                (blockDim.x >= (unsigned)MAX_INPUTS_PER_TASK) &&
                config.ml_num_layers > 1;
#ifdef MPK_ML_STAGE
            if (_pf_ok && !_ms_on) {
#else
            if (_pf_ok) {
#endif
              int _pf_base = (xcd_id * config.ml_num_layers + 1);
#ifdef MPK_IL_CACHE2
              if (threadIdx.x == 0) {
                _pf_var = config.ml_variant_ids[1];
              }
#endif
              if ((int)threadIdx.x < MAX_INPUTS_PER_TASK) {
                _pf_in = ml_in_tab[_pf_base * MAX_INPUTS_PER_TASK +
                                               threadIdx.x];
              }
#ifdef MPK_IL_Q4REG
              if (threadIdx.x == 4) {
                _pf_q4 = ml_in_tab[_pf_base * MAX_INPUTS_PER_TASK + 4];
              }
#endif
              if ((int)threadIdx.x < MAX_OUTPUTS_PER_TASK) {
                _pf_out =
                    ml_out_tab[_pf_base * MAX_OUTPUTS_PER_TASK +
                                           threadIdx.x];
              }
            }
#endif
#if defined(MPK_ML_TABLE_INLAYER) && defined(MPK_IL_FAST)
#error "MPK_ML_TABLE_INLAYER and MPK_IL_FAST both replace the per-layer table reads"
#endif
#ifdef MPK_ML_TABLE_INLAYER
            // Layers 1.. get their row written by the previous layer (same
            // condition as the fused layer's mlt_go); only the variant id is
            // read here, one layer ahead so its load retires under the call.
            bool const _mlt_on = g_aid_ml_in[xcd_id >> 2] != nullptr &&
                                 g_aid_ml_out[xcd_id >> 2] != nullptr &&
                                 config.ml_num_layers == MPK_MLT_LAYERS;
            int _mlt_var = config.ml_num_layers > 1
                               ? (int)config.ml_variant_ids[1]
                               : 0;
#endif
#ifdef MPK_IL_FAST
            if (config.ml_num_layers > 64) {
              __builtin_trap();
            }
            if (threadIdx.x == 0) {
              s_ml_qo_len = config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];
            }
            if ((int)threadIdx.x < config.ml_num_layers) {
              s_ml_variant[threadIdx.x] = config.ml_variant_ids[threadIdx.x];
              s_ml_qkvw[threadIdx.x] = config.ml_input_table
                  [(xcd_id * config.ml_num_layers + (int)threadIdx.x) *
                       MAX_INPUTS_PER_TASK +
                   4];
            }
            __syncthreads();
#endif
#if defined(MPK_PFIN_RETIRE) && defined(MPK_ML_TABLE_PREFETCH)
            // The first row's load must be retired in the compiler's wait
            // model before the loop header, or the header joins it as pending
            // and every boundary copy below gets a vmcnt(0) that also drains
            // the Phase 9 QKV weight DMA. The back edge is retired in the fused
            // layer's Phase 9, before that DMA is issued.
            __builtin_amdgcn_s_waitcnt(0x0F70);  // vmcnt(0)
#endif
            for (int ml = 0; ml < config.ml_num_layers; ml++) {
              MPK_ILSTAMP(2, ml == MPK_ILSUB_L0 + 1);
#ifdef MPK_DRAIN_STATS
              unsigned long long _mlt0 = __builtin_amdgcn_s_memrealtime();
#endif
#ifdef MPK_IL_LDS
              MPK_IL_MARK(0);
#endif
              // Layer 0: task_desc already loaded from precomputed dispatch
              // buffer with correct per-XCD pointers. Skip the copy.
#ifdef MPK_ML_TABLE_INLAYER
              if (_mlt_on && ml > 0) {
                if (threadIdx.x == 0) {
                  task_desc->variant_id = _mlt_var;
                }
                _mlt_var = (ml + 1 < config.ml_num_layers)
                               ? (int)config.ml_variant_ids[ml + 1]
                               : 0;
              } else
#endif
#ifdef MPK_ML_STAGE
              if (_ms_on) {
                int const t = threadIdx.x;
                int const N = config.ml_num_layers;
                if (ml > 0) {
                  if (t < MAX_INPUTS_PER_TASK) {
                    unsigned long long v = s_mls[t];
                    if (t == 24) {
                      v = s_mls_q4[0];
                    } else if (t == 25) {
                      v = s_mls[4];
                    }
                    task_desc->input_ptrs[t] = (void *)v;
                  }
                  if (t < MAX_OUTPUTS_PER_TASK) {
                    task_desc->output_ptrs[t] = (void *)s_mls[MAX_INPUTS_PER_TASK + t];
                  }
                  if (t == 0) {
                    task_desc->variant_id = (unsigned)s_mls[42];
                  }
                } else {
                  if (t == 24) {
                    task_desc->input_ptrs[24] =
                        ml_in_tab[(xcd_id * N + 1) * MAX_INPUTS_PER_TASK + 4];
                  }
                  if (t == 25) {
                    task_desc->input_ptrs[25] = nullptr;
                  }
                }
                if (t == 0) {
                  task_desc->task_metadata._linear_reserved =
                      (int32_t)((flb_iter - 1) * N + ml);
#ifdef MPK_ILSUB
                  s_ilsub_ml = ml;
#endif
                }
                __syncthreads();
                if (t < 64) {
                  if (ml + 1 < N) {
                    unsigned const r = (unsigned)(xcd_id * N + ml + 1);
                    if (t < MAX_INPUTS_PER_TASK / 2) {
                      mpk_mls_dma(_ms_rin, _ms_lds,
                                  r * (unsigned)(MAX_INPUTS_PER_TASK * 8) + (unsigned)t * 16u, true);
                    }
                    if (t < MAX_OUTPUTS_PER_TASK * 2) {
                      mpk_mls_dma(_ms_rout, _ms_lds + MAX_INPUTS_PER_TASK * 8,
                                  r * (unsigned)(MAX_OUTPUTS_PER_TASK * 8) + (unsigned)t * 4u, false);
                    }
                    if (t == 0) {
                      mpk_mls_dma(_ms_rvar, _ms_lds + 42 * 8, (unsigned)(ml + 1) * 4u, false);
                    }
                  }
                  if (ml + 2 < N) {
                    if (t < 2) {
                      mpk_mls_dma(_ms_rin, _ms_lds_q4,
                                  ((unsigned)(xcd_id * N + ml + 2) * (unsigned)MAX_INPUTS_PER_TASK + 4u) * 8u +
                                      (unsigned)t * 4u,
                                  false);
                    }
                  } else if (t == 0) {
                    s_mls_q4[0] = 0ull;
                  }
                }
              } else
#endif
              if (ml > 0) {
                // Widths must match the host-side ML_N_IN / ML_N_OUT that
                // built these tables (see the multi-layer scan), or the
                // strides disagree and every layer past 0 reads the wrong
                // slots. Both sides key off the TaskDesc capacity so the
                // LM-head variant's input_ptrs[24..27] / output_ptrs[12] get
                // refreshed too.
                int ml_in_base =
                    (xcd_id * config.ml_num_layers + ml) * MAX_INPUTS_PER_TASK;
                int ml_out_base =
                    (xcd_id * config.ml_num_layers + ml) * MAX_OUTPUTS_PER_TASK;
#ifdef MPK_ML_TABLE_PREFETCH
                if (_pf_ok) {
                  // Retire the values loaded a layer ago, then immediately
                  // re-issue for ml+1 so the next iteration finds them
                  // resident too. The store must precede the re-issue: both
                  // touch the same registers, and the compiler will otherwise
                  // sink the store past the load and serialize them again.
                  if ((int)threadIdx.x < MAX_INPUTS_PER_TASK) {
                    task_desc->input_ptrs[threadIdx.x] = _pf_in;
                  }
                  if ((int)threadIdx.x < MAX_OUTPUTS_PER_TASK) {
                    task_desc->output_ptrs[threadIdx.x] = _pf_out;
                  }
                  // Layer ml's own slot-4 value, kept before the re-issue
                  // below overwrites it. Slot 25 is exactly this, so holding
                  // it here lets thread 4 write both slot 24 and slot 25 from
                  // its own registers instead of thread 0 reading slot 4 back
                  // out of the task descriptor.
                  _pf_cur = _pf_in;
                  int _nl = ml + 1;
#ifdef MPK_IL_CACHE2
                  if (threadIdx.x == 0) {
                    _pf_var_cur = _pf_var;
                    if (_nl < config.ml_num_layers) {
                      _pf_var = config.ml_variant_ids[_nl];
                    }
                  }
#endif
                  if (_nl < config.ml_num_layers) {
                    int _nb = (xcd_id * config.ml_num_layers + _nl);
                    if ((int)threadIdx.x < MAX_INPUTS_PER_TASK) {
                      _pf_in = ml_in_tab[_nb * MAX_INPUTS_PER_TASK +
                                                     threadIdx.x];
                    }
                    if ((int)threadIdx.x < MAX_OUTPUTS_PER_TASK) {
                      _pf_out =
                          ml_out_tab[_nb * MAX_OUTPUTS_PER_TASK +
                                                 threadIdx.x];
                    }
                  }
                } else
#endif
                {
                  for (int i = threadIdx.x; i < MAX_INPUTS_PER_TASK;
                       i += blockDim.x) {
                    task_desc->input_ptrs[i] =
                        ml_in_tab[ml_in_base + i];
                  }
                  for (int i = threadIdx.x; i < MAX_OUTPUTS_PER_TASK;
                       i += blockDim.x) {
                    task_desc->output_ptrs[i] =
                        ml_out_tab[ml_out_base + i];
                  }
                }
                if (threadIdx.x == 0) {
#ifdef MPK_IL_FAST
                  task_desc->variant_id = s_ml_variant[ml];
#elif defined(MPK_IL_CACHE2)
                  task_desc->variant_id =
                      _pf_ok ? _pf_var_cur : config.ml_variant_ids[ml];
#else
                  task_desc->variant_id = config.ml_variant_ids[ml];
#endif
                }
#ifdef MPK_DRAIN_STATS
                // Before the first __syncthreads: isolates the table read from
                // the two block rendezvous that publish it.
                if (threadIdx.x == 0) {
                  unsigned long long _mltc =
                      __builtin_amdgcn_s_memrealtime();
                  atomicAdd(&g_mlcopy_sum, (_mltc - _mlt0) * 10);
                  atomicAdd(&g_mlcopy_n, 1ULL);
                }
#endif
#ifdef MPK_ML_TABLE_PREFETCH
                // Under the prefetch path the only reader of a slot written by
                // another thread was the slot-25 publish, and that now comes
                // from thread 4's own register. Every remaining store lands in
                // task_desc and is consumed after the single __syncthreads
                // below, so this rendezvous has nothing left to order.
                // Measured, the two of them were 914 ns of the 1679 ns
                // inter-layer prologue -- more than the table read itself.
                if (!_pf_ok)
#endif
                  __syncthreads();
              }

              MPK_ILSTAMP(3, ml == MPK_ILSUB_L0 + 1);
#ifdef MPK_PREFETCH_NEXT_QKV
              // Publish the NEXT layer's QKV weight pointer so the fused task
              // can start its HBM->LDS weight DMA during the Phase 9 barrier
              // spin instead of at the top of the next layer.
              //
              // Slot 24 is free for the plain fused variant: it declares 24
              // inputs (0..23) and the tables are sized to the TaskDesc
              // capacity of 28, so the host-side null validator already treats
              // >= 24 as legitimately unused.
              //
              // It is NOT free for the LM-head variant, which really does use
              // 24..27 -- so the whole publish is gated on the task type. That
              // variant is dead today (an early return at the top of
              // gang_full_layer_with_lmhead_fused_mi300.cuh) but FUSE_TAIL is
              // meant to come back, and writing nullptr over a live input
              // would fault rather than fail a gate. The gate costs the
              // prefetch entirely under FUSE_TAIL, which is the right trade
              // until someone gives that variant a slot of its own.
              //
              // Set outside the `ml > 0` guard above: layer 0 skips the table
              // copy (its pointers come from the precomputed dispatch buffer),
              // but it still needs to know layer 1's weights.
              //
              // Null on the last layer. The next reader would be layer 0 of
              // the following decode iteration, and tasks that use LDS run in
              // between (the MoE residual-add tail), so anything staged there
              // would be clobbered before it could be read.
              // Two slots, because the producer and the consumer need
              // different facts at different times and neither can derive the
              // other's:
              //
              //   [24] the NEXT layer's QKV weight pointer -- what Phase 9 of
              //        *this* layer issues its DMA against. Null on the last
              //        layer.
              //   [25] non-null iff the PREVIOUS layer's Phase 9 staged this
              //        layer's weights -- what Phase 1 keys its skip off. That
              //        is exactly `ml > 0`, and the value stored is this
              //        layer's own weight pointer so the two ends can be
              //        checked against each other.
              //
              // Carrying [25] as __shared__ state across the ml loop instead
              // would look cheaper but has no correct initial value: the very
              // first layer of the very first iteration would read whatever
              // was in LDS. Recomputing it host-side from `ml` is stateless.
#ifdef MPK_ML_STAGE
              if (!_ms_on) {
#endif
#ifdef MPK_ML_TABLE_PREFETCH
              // With the table prefetch running, slot 24 needs no HBM read:
              // thread 4 is already holding layer ml+1's input_ptrs[4] in a
              // register (it prefetched that exact element above), which is
              // precisely the next layer's QKV weight pointer. Writing it from
              // thread 4 also removes the cross-thread dependency that forced
              // the second __syncthreads -- slot 25 is thread 4's own
              // register too, so both stores come from the one thread that
              // owns the value and the block rendezvouses once, not twice.
#ifdef MPK_ML_TABLE_INLAYER
              if (_mlt_on && ml > 0) {
                // slots 24 / 25 were written by the previous layer
              } else
#endif
              if (task_desc->task_type == TASK_GANG_FULL_LAYER_FUSED_MI300 &&
                  _pf_ok) {
                if (threadIdx.x == 4) {
                  // _pf_in was re-issued for ml+1 in the copy block above, so
                  // on the last layer it still holds layer ml's value and must
                  // be suppressed -- same "null on the last layer" rule as the
                  // non-prefetch path, for the same reason.
                  task_desc->input_ptrs[24] =
#ifdef MPK_IL_FAST
                      (ml + 1 < config.ml_num_layers) ? s_ml_qkvw[ml + 1] : nullptr;
#elif defined(MPK_IL_Q4REG)
                      (ml + 1 < config.ml_num_layers) ? _pf_q4 : nullptr;
                  if (ml + 2 < config.ml_num_layers) {
                    _pf_q4 = ml_in_tab[(xcd_id * config.ml_num_layers +
                                        ml + 2) * MAX_INPUTS_PER_TASK + 4];
                  }
#else
                      (ml + 1 < config.ml_num_layers) ? _pf_in : nullptr;
#endif
                  // _pf_cur is this layer's slot 4, held from before the
                  // re-issue, so this needs no read of input_ptrs[4] and
                  // therefore no rendezvous with the thread that wrote it.
                  task_desc->input_ptrs[25] = (ml > 0) ? _pf_cur : nullptr;
                }
              } else
#endif
                  if (threadIdx.x == 0 &&
                      task_desc->task_type ==
                          TASK_GANG_FULL_LAYER_FUSED_MI300) {
                int const next_ml = ml + 1;
                task_desc->input_ptrs[24] =
                    (next_ml < config.ml_num_layers)
                        ? ml_in_tab[(xcd_id * config.ml_num_layers +
                                                 next_ml) *
                                                    MAX_INPUTS_PER_TASK +
                                                4]
                        : nullptr;
                task_desc->input_ptrs[25] =
                    (ml > 0) ? task_desc->input_ptrs[4] : nullptr;
              }
              __syncthreads();
#ifdef MPK_ML_STAGE
              }
#endif
#endif
              MPK_ILSTAMP(4, ml == MPK_ILSUB_L0 + 1);
#ifdef MPK_DRAIN_STATS
              // The inter-layer prologue: the ml_input_table / ml_output_table
              // copy plus the two __syncthreads that publish it. This runs
              // AFTER the previous layer's Phase 9 gate cleared, so it is
              // serial critical-path latency between layers -- and it is an
              // HBM read (the tables are device globals), not an LDS one.
              // Passing the pointers as template/function arguments, with the
              // gate *inside* the layer body after the pointers are known,
              // would remove this prologue entirely.
              if (threadIdx.x == 0 && ml > 0) {
                unsigned long long _mlt1 = __builtin_amdgcn_s_memrealtime();
                atomicAdd(&g_mlprologue_sum, (_mlt1 - _mlt0) * 10);
                atomicAdd(&g_mlprologue_n, 1ULL);
              }
#endif
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
                b[6] = (unsigned long long)flb_iter;
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
#ifdef MPK_ML_STAGE
              if (!_ms_on) {
#endif
              if (threadIdx.x == 0) {
                task_desc->task_metadata._linear_reserved =
                    (int32_t)((flb_iter - 1) * config.ml_num_layers + ml);
#ifdef MPK_ILSUB
                s_ilsub_ml = ml;
#endif
              }
              __syncthreads();
#ifdef MPK_ML_STAGE
              }
#endif
              MPK_ILSTAMP(5, ml == MPK_ILSUB_L0 + 1);

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
#ifdef MPK_PRECOMPUTED_DISPATCH
              if (threadIdx.x == 0 && MPK_WS_ON(config)) {
                int *ws = config.precomp_dbg_worker_state + worker_id * 4;
                __atomic_store_n(&ws[3], 40000 + ml, __ATOMIC_RELAXED);
              }
#endif
#ifdef MPK_IL_LDS
              MPK_IL_MARK(1);
#endif
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
#if defined(MPK_FUSED_INLINE) && defined(MPK_IL_CACHE2)
                if (ml == 0) {
                  ml_qo_len = config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];
                }
                _execute_gang_task_ml(task_desc, config, ml_n_tile_start + t, ml_qo_len);
#elif defined(MPK_FUSED_INLINE)
                _execute_gang_task_ml(task_desc, config, ml_n_tile_start + t);
#else
                _execute_gang_task(task_desc, config, ml_n_tile_start + t);
#endif
                my_tiles++;
              }
              if (threadIdx.x == 0) {
                gang_tiles_executed = my_tiles;
              }
              MPK_ILSTAMP(0, ml == MPK_ILSUB_L0);

              // Inter-layer sync: threadfence_gpu flushes L2 write buffer so
              // next layer's buffer_inv + QKV epoch barrier sees fresh data.
              // __syncthreads ensures all threads finish before task_desc is
              // modified for the next layer.
              if (ml < config.ml_num_layers - 1) {
#ifdef MPK_INTERLAYER_FENCE
                // Retired by default: this threadfence_gpu had nothing to
                // flush. It was a buffer_wbl2 sc1 + drain meant to push the
                // layer's output out of L2, but the layer's output never
                // enters L2 -- the MoE W2 epilogue writes the workspace with
                // st_wt_f32x4 (sc0 sc1, write-through straight to HBM), and
                // Phase 9 then drains with an explicit s_waitcnt vmcnt(0)
                // before its release atomic. The next layer's consumers still
                // do their own buffer_inv behind the QKV epoch poll, so the
                // acquire side is unchanged.
                //
                // Measured 2.215 -> 2.204 (n=3 each, 16 tokens) and
                // 2.233 -> 2.202 (n=6, 128 tokens), output byte-identical
                // across eight runs. Define MPK_INTERLAYER_FENCE to restore.
                if (threadIdx.x == 0 && block_xcd_local_rank == 0) {
                  threadfence_gpu();
                }
#endif
                __syncthreads();
              }
              MPK_ILSTAMP(1, ml == MPK_ILSUB_L0);
            }

            // Deferred event signal: after compaction, layer 0's trigger_event
            // points to the last layer's event (gates FUSE_TAIL). Signal via
            // the two-level path since this task is in the queue.
            if (threadIdx.x == 0) {
              // MPK_FI_SIGNAL_DRAIN: every worker runs this once per iteration,
              // right after the last layer. Everything the last layer hands to
              // another XCD is already write-through (W2 workspace st_wt, P9
              // drains before its release), and the KV cache is read back only
              // by this XCD, so retiring the stores is the whole release. The
              // 248 simultaneous L2 writebacks cost ~4 us at 216 -> 212 in
              // SPX/NPS2 against ~0.4 us in NPS1.
              //
              // MPK_FI_SIGNAL_XCDFENCE keeps the full release but pays it once
              // per XCD: each worker only retires its stores into this XCD's L2
              // before the XCD-local arrival, and the last arriver's L2
              // writeback then covers all of them -- the gang task-end shape.
              // In NPS2 the AID-local RW lines are write-back, so each of the
              // 248 buffer_wbl2 has dirty lines to walk; in NPS1 (NC) none do.
#if defined(MPK_TASK_FENCE_DRAIN) || defined(MPK_FI_SIGNAL_DRAIN) || \
    defined(MPK_FI_SIGNAL_XCDFENCE)
              asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#else
              threadfence_gpu();
#endif
              EventId ev_id = config.ml_trigger_events[0];
              size_t ev_idx = get_event_position_index(ev_id);
              int xcd_slot = xcd_id * config.num_events + (int)ev_idx;
              int xcd_thresh = config.xcd_event_num_tasks != nullptr
                    ? MPK_LD_EVCFG(&config.xcd_event_num_tasks[xcd_slot])
                    : 0;
              if (xcd_thresh > 0) {
                TaskId ml_tid =
                    compute_task_id(flb_iter, config.ml_task_positions[0]);
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
#if defined(MPK_FI_SIGNAL_XCDFENCE) && !defined(MPK_TASK_FENCE_DRAIN) && \
    !defined(MPK_FI_SIGNAL_DRAIN)
                  threadfence_gpu();
#endif
#if defined(MPK_AID_EVTIER) && defined(MPK_AID_SPLIT_FLAGS)
                  // AID tier. The four XCDs of one AID accumulate in an AID-local line,
                  // and only the last of them crosses the boundary -- adding 4, so the
                  // global total and every `count == num_triggers * iter` test are
                  // unchanged while boundary-crossing atomics drop from 8 to 2.
                  // Gang events only: num_triggers == 8 is one flush per XCD, hence
                  // exactly 4 per AID per iteration. For any other contributor count the
                  // per-AID total is not a multiple of 4, which would leave the global
                  // permanently short, so those keep the original single increment.
                  int _nt = MPK_LD_EVCFG(&config.all_event_num_triggers[ev_idx]);
                  unsigned long long _ok = 0, _n = 0;
                  if (_nt == 2 * MPK_AID_XCDS_PER_AID) {
                    _n = mpk_aid_evtier_add((int)ev_idx, &_ok);
                  }
                  if (!_ok) {
                    mpk_evctr_add(config.all_event_counters, ev_idx, 1);
                  } else if ((_n % MPK_AID_XCDS_PER_AID) == 0) {
                    mpk_evctr_add(config.all_event_counters, ev_idx,
                                  MPK_AID_XCDS_PER_AID);
                  }
#else
                  mpk_evctr_add(config.all_event_counters, ev_idx, 1);
#endif
                }
              } else {
#if defined(MPK_FI_SIGNAL_XCDFENCE) && !defined(MPK_TASK_FENCE_DRAIN) && \
    !defined(MPK_FI_SIGNAL_DRAIN)
                threadfence_gpu();
#endif
                mpk_evctr_add(config.all_event_counters, ev_idx, 1);
              }
            }

            if (threadIdx.x == 0) {
              _flb_skip_signal = 1;
            }

          } else {
            // ===== Original fast-loop: per-layer dispatch =====
            while (true) {
              // Publish the layer counter here too.
              //
              // The multi-layer branch above stores this once per `ml`, and the
              // fused-layer task derives all four of its barrier release values
              // from it (qkv_epoch, attn_release, routing_ready, and the MoE
              // W13->W2 layer_epoch). This branch never did -- so on any run
              // where multi-layer batching is off (it needs >1 fused layer;
              // `--max-layers 1` disables it, and the host prints "Multi-layer
              // table: disabled"), the field kept the ~0ull that
              // FullTaskDesc's constructor writes into raw_payload.
              //
              // That made task_layer_idx == -1, so every expected value was
              // layer_counter + 1 == 0, and *every* barrier in the fused layer
              // became a no-op: the counters are host allocations that start at
              // 0, so `while (observed < 0)` never spins even once. Verified
              // directly -- an instrumented Phase 6 poll reported
              // ALREADY-PAST on 101440 of 101440 entries, none waiting.
              //
              // With the cross-XCD attention barrier disabled, O-proj reads
              // attn_out while the merges are still running. The damage is
              // silent and lands on the requests whose merges finish last and
              // the workers that reach Phase 6 first (xcd_rank >=
              // total_qkv_tiles_per_xcd) -- the row>=2 x wg>=10 block seen in
              // attn_proj_out, and the reason a delay anywhere in Phase 6 made
              // it disappear while no cache fence on either side helped.
              //
              // Same expression as the ml branch: monotonic, never reset, one
              // bump per layer. There is exactly one fused layer per iteration
              // on this path, so the layer term is 0.
              if (threadIdx.x == 0) {
                task_desc->task_metadata._linear_reserved =
                    (int32_t)(flb_iter - 1);
              }
              __syncthreads();
              int n_tile_start = (int)task_desc->task_metadata.n_tile_start;
              int n_tile_count = (int)task_desc->task_metadata.n_tile_count;
              int my_tiles = 0;
              for (int t = block_xcd_local_rank; t < n_tile_count;
                   t += block_workers_on_xcd) {
                _execute_gang_task(task_desc, config, n_tile_start + t);
                my_tiles++;
              }

              if (threadIdx.x == 0) {
                gang_tiles_executed = my_tiles;
                EventId ev_id = task_desc->trigger_event;
                size_t ev_idx = get_event_position_index(ev_id);
                int num_triggers = MPK_LD_EVCFG(&config.all_event_num_triggers[ev_idx]);
                int xcd_slot = xcd_id * config.num_events + (int)ev_idx;
                int xcd_thresh = config.xcd_event_num_tasks != nullptr
                    ? MPK_LD_EVCFG(&config.xcd_event_num_tasks[xcd_slot])
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
#if defined(MPK_TASK_FENCE_DRAIN) || (defined(MPK_CPX_ML_DRAIN) && defined(MPK_CPX_SCOPE))
                    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#else
                    threadfence_gpu();
#endif
#if defined(MPK_AID_EVTIER) && defined(MPK_AID_SPLIT_FLAGS)
                    // AID tier. The four XCDs of one AID accumulate in an AID-local line,
                    // and only the last of them crosses the boundary -- adding 4, so the
                    // global total and every `count == num_triggers * iter` test are
                    // unchanged while boundary-crossing atomics drop from 8 to 2.
                    // Gang events only: num_triggers == 8 is one flush per XCD, hence
                    // exactly 4 per AID per iteration. For any other contributor count the
                    // per-AID total is not a multiple of 4, which would leave the global
                    // permanently short, so those keep the original single increment.
                    int _nt = num_triggers;
                    unsigned long long _ok = 0, _n = 0;
                    if (_nt == 2 * MPK_AID_XCDS_PER_AID) {
                      _n = mpk_aid_evtier_add((int)ev_idx, &_ok);
                    }
                    if (!_ok) {
                      mpk_evctr_add(config.all_event_counters, ev_idx, 1);
                    } else if ((_n % MPK_AID_XCDS_PER_AID) == 0) {
                      mpk_evctr_add(config.all_event_counters, ev_idx,
                                    MPK_AID_XCDS_PER_AID);
                    }
#else
                    mpk_evctr_add(config.all_event_counters, ev_idx, 1);
#endif
                  }
                } else {
#if defined(MPK_TASK_FENCE_DRAIN) || (defined(MPK_CPX_ML_DRAIN) && defined(MPK_CPX_SCOPE))
                  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#else
                  threadfence_gpu();
#endif
                  mpk_evctr_add(config.all_event_counters, ev_idx, 1);
                }
              }

              int next_qp = queue_pos + 1;
              if (next_qp >= queue_len) {
#ifdef MPK_PRECOMPUTED_DISPATCH
                int remaining = pc_my_len - pc_pos;
                if (remaining <= 0) {
                  break;
                }
                int num_to_load = min(remaining, TASK_DESCS_BUFFER_LENGTH);
                __syncthreads();
                if (threadIdx.x < num_to_load) {
                  task_ids[threadIdx.x] = compute_task_id(
                      flb_iter, pc_my_queue[pc_pos + threadIdx.x]);
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
#else
                break;
#endif
              }

              TaskDesc *next_td = task_descs + next_qp;
              if (next_td->task_type != TASK_GANG_FULL_LAYER_FUSED_MI300 &&
                  next_td->task_type !=
                      TASK_GANG_FULL_LAYER_WITH_LMHEAD_FUSED_MI300) {
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
        case TASK_GANG_MOE_W13_SWIGLU_MXFP4_MI300:
          slot = 9;
          break;
        case TASK_MOE_W2_LINEAR_MI300:
        case TASK_MOE_W2_LINEAR_MXFP4_MI300:
        case TASK_MOE_W2_LINEAR_MXFP4_CK_MI300:
        case TASK_GANG_MOE_W2_LINEAR_MI300:
        case TASK_GANG_MOE_W2_LINEAR_MXFP4_MI300:
          slot = 10;
          break;
        case TASK_GANG_MOE_FUSED_MXFP4_MI300:
          slot = 9;
          break; // Reuse W13 slot (fused replaces W13+SwiGLU+W2)
        case TASK_GANG_MOE_SWIGLU_W2_MXFP4_MI300:
          slot = 10;
          break; // W2 slot (fused SwiGLU+W2 replaces SwiGLU+W2)
        case TASK_MOE_TOPK_SOFTMAX_MI300:
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
        case TASK_PAGED_ATTENTION_CK_FMHA_SPLIT_KV_MI300:
        case TASK_GANG_ATTN_SPLIT_KV_MI300:
        case TASK_GANG_ATTN_MERGE_MI300:
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

#ifdef MPK_ITER_TS
    if (threadIdx.x == 0 && task_desc->task_type == TASK_ARGMAX_REDUCE) {
      __atomic_store_n(&g_its_argmax_end, get_wallclock_ns(),
                       __ATOMIC_RELAXED);
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
#ifdef MPK_ITER_TS
      unsigned long long its_tr[5] = {0, 0, 0, 0, 0};
      bool const its_on = task_desc->task_type == TASK_ARGMAX_REDUCE;
#endif
      EventId event_id = task_desc->trigger_event;
      size_t event_index = get_event_position_index(event_id);
      if (!is_nvshmem_event(event_id)) {
        size_t gpu_id = get_event_gpu_id(event_id);
        assert(gpu_id == config.my_gpu_id);
        // Case 1: Trigger a local non-nvshmem event
        int num_triggers = MPK_LD_EVCFG(&config.all_event_num_triggers[event_index]);
#ifdef MPK_EVT_FAST
        // Read-only and host-built: issued with the counter metadata so its
        // latency overlaps theirs instead of following both atomics.
        EventDesc const evt_fast_desc = config.all_events[event_index];
#endif
        EventCounter count = 0;
        bool event_fired = false;

#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
        // Two-level event counting: XCD-local first, batch flush when
        // this XCD's share is done. Reduces buffer_wbl2 from ~24K to ~300.
        int xcd_slot = xcd_id * config.num_events + (int)event_index;
        int xcd_threshold = config.xcd_event_num_tasks != nullptr
                    ? MPK_LD_EVCFG(&config.xcd_event_num_tasks[xcd_slot])
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
#ifdef MPK_EVT_FAST
            EventCounter needed_local =
                static_cast<EventCounter>(xcd_threshold) *
                get_task_iteration_num(task_ids[queue_pos]);
            // A sole contributor on its XCD is always the last one; the
            // returning L2 atomic would only confirm it.
            EventCounter local_count =
                (xcd_threshold == 1 && event_increment == 1)
                    ? needed_local
                    : atom_add_local_u64(
                          reinterpret_cast<unsigned long long int *>(
                              &config.xcd_local_event_counters[xcd_slot]),
                          event_increment) +
                          event_increment;
#else
            EventCounter local_count =
                atom_add_local_u64(
                    reinterpret_cast<unsigned long long int *>(
                        &config.xcd_local_event_counters[xcd_slot]),
                    event_increment) +
                event_increment;
            EventCounter needed_local =
                static_cast<EventCounter>(xcd_threshold) *
                get_task_iteration_num(task_ids[queue_pos]);
#endif

#ifdef MPK_ITER_TS
            if (its_on) {
              its_tr[0] = get_wallclock_ns();
            }
#endif
            if (local_count == needed_local) {
              // Last worker/task on this XCD — flush + signal global
#ifdef MPK_EVT_FAST
              // The graph-begin marker writes no data, so it has nothing to
              // publish.
              if (task_desc->task_type != TASK_BEGIN_TASK_GRAPH)
#endif
#ifdef MPK_TASK_FENCE_DRAIN
              asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#else
              threadfence_gpu();
#endif
#ifdef MPK_ITER_TS
              if (its_on) {
                its_tr[1] = get_wallclock_ns();
              }
#endif
              // Gang: flush 1 (one task done). Non-gang: flush xcd_threshold.
              int flush_amount =
                  is_gang_task_type(task_desc->task_type) ? 1 : xcd_threshold;
              count = mpk_evctr_add(config.all_event_counters, event_index, flush_amount);
#ifdef MPK_ITER_TS
              if (its_on) {
                its_tr[2] = get_wallclock_ns();
              }
#endif
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
#ifdef MPK_TASK_FENCE_DRAIN
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#else
            threadfence_gpu();
#endif
            count = mpk_evctr_add(config.all_event_counters, event_index, xcd_threshold);
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
#ifdef MPK_TASK_FENCE_DRAIN
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#else
          threadfence_gpu();
#endif
          count = mpk_evctr_add(config.all_event_counters, event_index, 1);
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
#ifdef MPK_EVT_FAST
          EventDesc event_desc = evt_fast_desc;
#else
          EventDesc event_desc = config.all_events[event_index];
#endif
#ifdef MPK_ITER_TS
          if (its_on) {
            its_tr[3] = get_wallclock_ns();
          }
#endif
#ifdef MPK_END_FAST
          bool ed_skip = false;
#endif
#ifdef MPK_END_DIRECT
          if (event_desc.event_type == EVENT_END_OF_TASK_GRAPH) {
            bool const ed_cont =
                prepare_next_batch_w(config) && !MPK_HOST_STOP_AT(config, pc_iter);
            if (ed_cont) {
              __threadfence();
#ifdef MPK_END_FAST
              atomicAdd(config.precomp_iter_ready, 1ULL);
              unsigned long long const ed_new = (unsigned long long)pc_iter + 1ULL;
#else
              unsigned long long const ed_new =
                  atomicAdd(config.precomp_iter_ready, 1ULL) + 1ULL;
#endif
              int *ed_rel = config.precomp_iter_xcd_release;
#ifdef MPK_ITER_AID
              mpk_aid_iter_publish((unsigned)ed_new);
              if (mpk_aid_iter_slot(0) == nullptr) {
                for (int x = 0; x < MPK_NUM_XCDS; x++) {
                  st_wt_u32((void *)&ed_rel[x * 16], (unsigned)ed_new);
                }
              }
#else
              for (int x = 0; x < MPK_NUM_XCDS; x++) {
                st_wt_u32((void *)&ed_rel[x * 16], (unsigned)ed_new);
              }
#endif
#ifdef MPK_END_FAST
              ed_rel_iter = (unsigned long long)pc_iter + 1ULL;
              ed_skip = true;
              ed_defer_ev = (long long)event_index;
              ed_defer_wait = 1;
#endif
            }
            __atomic_store_n(&g_end_direct,
                             ((unsigned long long)pc_iter << 1) |
                                 (ed_cont ? 1ull : 0ull),
                             __ATOMIC_RELAXED);
#ifdef MPK_END_FAST
            if (!ed_cont)
#endif
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          }
#endif
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
#ifdef MPK_END_FAST
          else if (ed_skip) {
            // pushed at the end of this worker's next trigger
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
#ifdef MPK_ITER_TS
          if (its_on) {
            its_tr[4] = get_wallclock_ns();
            unsigned long long const t0 =
                __atomic_load_n(&g_its_argmax_end, __ATOMIC_RELAXED);
            if (t0 != 0 && its_tr[0] >= t0 && its_tr[1] >= its_tr[0] &&
                its_tr[2] >= its_tr[1] && its_tr[3] >= its_tr[2] &&
                its_tr[4] >= its_tr[3] && its_tr[4] - t0 < 1000000ull) {
              g_its_tr_sum[0] += its_tr[0] - t0;
              g_its_tr_sum[1] += its_tr[1] - its_tr[0];
              g_its_tr_sum[2] += its_tr[2] - its_tr[1];
              g_its_tr_sum[3] += its_tr[3] - its_tr[2];
              g_its_tr_sum[4] += its_tr[4] - its_tr[3];
              g_its_tr_n++;
            }
            __atomic_store_n(&g_its_push_end, its_tr[4], __ATOMIC_RELAXED);
          }
#endif
#ifdef MPK_ENABLE_PROFILING
          PROFILER_EVENT_END(TASK_SCHD_EVENTS, task_counter++);
#endif
        }
#ifdef MPK_END_FAST
        if (ed_defer_ev >= 0) {
          if (ed_defer_wait) {
            ed_defer_wait = 0;
          } else {
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            int const ed_sid = get_rand_sched_id((size_t)ed_defer_ev,
                                                 worker_id,
                                                 config.num_workers,
                                                 config.num_local_schedulers,
                                                 xcd_id);
            size_t const ed_pos = atom_add_local_u64(
                &config.sched_queue_next_free_event_id[ed_sid], 1);
            st_local_u64(
                &config.sched_queues[ed_sid][ed_pos % config.per_sched_queue_len],
                (size_t)ed_defer_ev);
            fence_local();
            size_t ed_old;
            do {
              ed_old = atom_cas_local_u64(
                  &config.sched_queue_last_ready_event_id[ed_sid], ed_pos,
                  ed_pos + 1);
            } while (ed_old != ed_pos);
            ed_defer_ev = -1;
          }
        }
#endif
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
#ifdef MPK_CPX_SCOPE // MDEV_CPX_BOUNDARY
        while (atomicAdd_system(config.worker_xcd_ready_count, 0) <
#else
        while (atomicAdd(config.worker_xcd_ready_count, 0) <
#endif
               config.num_workers) {
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
#ifdef MPK_END_POLL
#if !defined(MPK_PRECOMPUTED_DISPATCH) || !defined(MPK_TAIL_NO111) || defined(MPK_END_DIRECT) || defined(MPK_END_FAST)
#error "MPK_END_POLL needs MPK_PRECOMPUTED_DISPATCH and MPK_TAIL_NO111 (an end marker with no work), and neither MPK_END_DIRECT nor MPK_END_FAST"
#endif
    // MPK_END_POLL: after the first end event the end marker pushes, wait for the marker's own dependency (the
    // LM-head task's event) instead of its push; the marker's later pushes are dropped.
    int ep_ev = -1;
    bool ep_drop_end = false;
    EventId ep_end_id = 0;
    EventCounter ep_target = 0, ep_step = 0;
    unsigned long long ep_rel_clk = 0, ep_quiet_until = 0, ep_quiet_ns = 800000ull;
#ifdef MPK_END_POLL_DBG
    unsigned ep_polled = 0, ep_dropped = 0, ep_armed_at = 0;
#endif
#endif
#ifdef MPK_ENABLE_DISPATCH_TIMING
    unsigned long long last_dep_dispatch_end_ns = 0;
#endif

    while (true) {
#ifdef MPK_END_POLL
      bool ep_got = false;
      if (ep_ev >= 0) {
#ifndef MPK_END_POLL_EAGER
        // most of a pass the LM-head event cannot complete: leave the event mirror lines to the layers meanwhile
        while (get_wallclock_ns() < ep_quiet_until) {
          __nanosleep(2000);
        }
#endif
        for (unsigned ep_spin = 0;; ep_spin++) {
          if ((EventCounter)MPK_EVCTR_POLL(config.all_event_counters, ep_ev, ep_spin) >= ep_target) {
            ep_got = true;
            break;
          }
          // termination, and the marker's own (now late) end events, still come through the own queue
          if ((ep_spin & 15) == 15 &&
              ld_local_u64(&config.sched_queue_last_ready_event_id[sched_queue_ids[0]]) > cur_event_pos[0]) {
            break;
          }
          __builtin_amdgcn_s_sleep(1);
        }
        if (ep_got) {
          __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, MPK_DEV_FENCE_ACQ);
          {
            unsigned long long const ep_obs = get_wallclock_ns() - ep_rel_clk;
            if (ep_rel_clk != 0 && ep_obs < 1000000000ull && (ep_obs * 85ull) / 100ull < ep_quiet_ns) {
              ep_quiet_ns = (ep_obs * 85ull) / 100ull;
            }
          }
#ifdef MPK_END_POLL_DBG
          ep_polled++;
#endif
        }
      }
#endif
      int poll_count = 0;
#ifdef MPK_END_POLL
      while (!ep_got && cur_event_pos[queue_idx] == last_event_pos[queue_idx]) {
#else
      while (cur_event_pos[queue_idx] == last_event_pos[queue_idx]) {
#endif
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
#ifdef MPK_END_POLL
      if (ep_got) {
        event_id = ep_end_id;
      } else
#endif
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
#ifdef MPK_END_POLL
      if (!ep_got && ep_drop_end && e.event_type == EVENT_END_OF_TASK_GRAPH) {
        // the marker's push for a pass the poll already handled
#ifdef MPK_END_POLL_DBG
        ep_dropped++;
#endif
        cur_event_pos[queue_idx] += 1;
        continue;
      }
#endif
      if (is_termination_event(event_id, e)) {
#if defined(MPK_END_POLL) && defined(MPK_END_POLL_DBG)
        if (ep_drop_end) {
          printf("[EPOLL] sched=%d armed_at=%u polled=%u dropped=%u ep_ev=%d step=%llu\n", sched_id, ep_armed_at,
                 ep_polled, ep_dropped, ep_ev, (unsigned long long)ep_step);
        }
#endif
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
#ifdef MPK_BOOK_LATE
        // MPK_BOOK_LATE: run after the iteration release (book_late() below); the pass's values are captured here.
        auto book_late = [book_valid = prev_end_of_graph_clk != 0, idx = end_of_graph_count,
                          dur = iter_end_clk - prev_end_of_graph_clk, &num_active_tokens]() {
          if (!book_valid) {
            return;
          }
#else
        if (prev_end_of_graph_clk != 0) {
          int idx = end_of_graph_count; // 1-based after first iter
          unsigned long long dur = iter_end_clk - prev_end_of_graph_clk;
#endif
          // Aggregates first: these must cover every iteration, including the
          // ones past the end of the ring.
          g_fwdpass_total_ns += dur;
          g_fwdpass_total_iters++;
#ifdef MPK_PHASE_SLOTS
          // Arm the phase recorder here and nowhere else.
          //
          // The obvious site -- the worker loop's TASK_BEGIN_TASK_GRAPH branch
          // -- is dead: workers never receive that task type, so an arm placed
          // there leaves g_phase_arm at 0 forever while all 97M marks fall
          // through. This handler is the one place that ticks exactly once per
          // forward pass, and scheduler 0 owns it.
          //
          // Arming between forward passes rather than mid-pass is what makes
          // the slot-0 liveness guard sufficient: every worker sees the flag
          // set before it enters its next layer 0.
          //
          // No sched_id guard: which scheduler drains the global queue's end
          // event is not fixed, and pinning this to sched_id 0 left the flag
          // at 0 for a whole 910-iteration run. Any scheduler reaching here
          // has ticked the counter, and the store is an idempotent 1, so a
          // race between two of them writes the same value.
          if (!g_phase_arm &&
              g_fwdpass_total_iters >= MPK_PHASE_START_ITER) {
            __threadfence();
            g_phase_arm = 1;
            __threadfence();
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
#ifdef MPK_BOOK_LATE
        };
#else
        }
#endif
        prev_end_of_graph_clk = iter_end_clk;
        end_of_graph_count++;
#ifdef MPK_ITER_TS
        unsigned long long const its_t1b = get_wallclock_ns();
#endif
        // Publish decode progress to pinned host memory.
        //
        // config.step[0] -- request 0's token position -- not the iteration
        // count: prefill packs up to MPK_MAX_NUM_BATCHED_TOKENS positions into
        // a single iteration, so iterations and tokens are not the same unit.
        // Publishing iterations made a 1024-token prefill look like 64 steps,
        // and the streaming server then held back every token until the launch
        // was nearly over.
        //
        // step[0] is updated by the *previous* iteration's prepare_next_batch,
        // so at this point it names the positions already written to
        // config.tokens -- a host reader that sees N may safely read N.
        // Written before the continue-check so the final iteration is
        // published even on the terminating path. Release ordering pairs with
        // the host's acquire load: the token stores must be visible before the
        // count that advertises them.
        //
        // One entry per request, not just request 0: a batch coalesces
        // prompts of different lengths, so each request reaches its first
        // output token at a different absolute position.
#ifndef MPK_TAIL_PUB_LATE
        if (config.progress_host != nullptr) {
#ifdef MPK_TOK_HOST // request 0's newest token id beside its position (bs=1 streaming)
          {
            int const tp = (int)config.step[0];
            for (int _j = 0; _j < 16; _j++) {  // spec: backfill accepted positions committed this pass
              int const _p = tp - _j;
              if (_p < 0 || _p >= MPK_MAX_SEQ_LENGTH) break;
              config.progress_host[MPK_MAX_NUM_BATCHED_REQUESTS + _p] = (int)__hip_atomic_load(
                  &config.tokens[_p], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            }
          }
#endif
          __threadfence_system();
          for (int r = 0; r < MPK_MAX_NUM_BATCHED_REQUESTS; r++) {
            __atomic_store_n(
                &config.progress_host[r], config.step[r], __ATOMIC_RELEASE);
          }
        }
#endif
#ifdef MPK_ITER_TS
        unsigned long long const its_t2 = get_wallclock_ns();
#endif
        // Check if we want to continue
#ifdef MPK_END_DIRECT
        unsigned long long const ed_word =
            __atomic_load_n(&g_end_direct, __ATOMIC_RELAXED);
        bool const end_direct =
            (ed_word >> 1) == (unsigned long long)end_of_graph_count;
        if (end_direct ? (ed_word & 1ull) == 0ull
                       : (!prepare_next_batch(config) ||
                          MPK_HOST_STOP_AT(config, end_of_graph_count)))
#elif defined(MODE_ONLINE_NOTOKEN)
        if (!prepare_next_batch(config, iteration_num))
#else
        if (!prepare_next_batch(config) ||
            MPK_HOST_STOP_AT(config, end_of_graph_count))
#endif
        {
#ifdef MPK_BOOK_LATE
          book_late();
#endif
          // Republish after the terminating prepare_next_batch.
          //
          // The publish above runs BEFORE prepare_next_batch, so it reports
          // the step count as of the previous iteration -- the final token is
          // written by this last call and would never be advertised. A
          // streaming reader would then sit on the second-to-last token until
          // the launch ended and the poll loop fell through to its flush,
          // charging the whole FWD_PASS log dump and teardown (~150-490 ms) to
          // the last inter-token gap.
          if (config.progress_host != nullptr) {
#ifdef MPK_TOK_HOST // request 0's newest token id beside its position (bs=1 streaming)
            {
              int const tp = (int)config.step[0];
              for (int _j = 0; _j < 16; _j++) {  // spec: backfill accepted positions committed this pass
                int const _p = tp - _j;
                if (_p < 0 || _p >= MPK_MAX_SEQ_LENGTH) break;
                config.progress_host[MPK_MAX_NUM_BATCHED_REQUESTS + _p] = (int)__hip_atomic_load(
                    &config.tokens[_p], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
              }
            }
#endif
            __threadfence_system();
            for (int r = 0; r < MPK_MAX_NUM_BATCHED_REQUESTS; r++) {
              __atomic_store_n(
                  &config.progress_host[r], config.step[r], __ATOMIC_RELEASE);
            }
          }
          unsigned long long prep_done_clk = get_wallclock_ns();
#ifndef MPK_ENABLE_PROFILING
          printf("[ITER_TIME] sched=%d iter=%llu prep_us=%.1f DONE\n",
                 sched_id,
                 iteration_num,
                 (double)(prep_done_clk - iter_end_clk) / 1000.0);
#endif
#ifdef MPK_ITER_TS
          if (g_its_n > 0) {
            double const n = (double)g_its_n;
            printf("[ITS] n=%llu argmax_to_deq=%.2f book=%.2f progress=%.2f "
                   "prepare=%.2f release=%.2f us\n",
                   g_its_n, g_its_sum[0] / n / 1000.0, g_its_sum[1] / n / 1000.0,
                   g_its_sum[2] / n / 1000.0, g_its_sum[3] / n / 1000.0,
                   g_its_sum[4] / n / 1000.0);
          }
          if (g_its_emb_n > 0) {
            double const n = (double)g_its_emb_n;
            printf("[ITS] emb n=%llu prep_to_gate_exit=%.2f "
                   "gate_exit_to_embedding=%.2f us\n",
                   g_its_emb_n, g_its_emb_sum[0] / n / 1000.0,
                   g_its_emb_sum[1] / n / 1000.0);
          }
          if (g_its_tr_n > 0 && g_its_n > 0) {
            double const n = (double)g_its_tr_n;
            printf("[ITS] trig n=%llu local=%.2f fence=%.2f counter=%.2f desc=%.2f "
                   "push=%.2f push_to_deq=%.2f us\n",
                   g_its_tr_n, g_its_tr_sum[0] / n / 1000.0,
                   g_its_tr_sum[1] / n / 1000.0, g_its_tr_sum[2] / n / 1000.0,
                   g_its_tr_sum[3] / n / 1000.0, g_its_tr_sum[4] / n / 1000.0,
                   g_its_sum[5] / (double)g_its_n / 1000.0);
          }
#endif
          // Dump deferred FWD_PASS log (after timing, before terminate).
          // iter is reconstructed from the decimation stride.
          //
          // Skipped under MPK_QUIET_FWDPASS. The dump is thousands of device
          // printfs drained over PCIe, which costs ~150-500 ms of teardown --
          // negligible once per offline run, but a server pays it on every
          // request, where it lands inside the measured end-to-end latency.
          // The [FWD_PASS_TOTAL] summary below still reports the real
          // per-iteration average, so nothing needed for timing is lost.
#ifndef MPK_QUIET_FWDPASS
          for (int i = 1; i < g_fwdpass_count && i < FWDPASS_LOG_MAX; i++) {
            printf("[FWD_PASS] iter=%d time_ms=%.3f num_active_tokens=%d\n",
                   i * g_fwdpass_stride,
                   (double)g_fwdpass_time_ns[i] / 1000000.0,
                   g_fwdpass_tokens[i]);
          }
#endif
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
#ifdef MPK_DRAIN_STATS
          // The hot-path dump only fires every 100k arrivals; a 64-token
          // decode never reaches that. Print once at terminate so last-XCD
          // identity and per-XCD spin are visible on a normal smoke run.
          {
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
          }
#endif
#ifdef MPK_GATE_SNAP
          // Per layer boundary L-1 -> L: the last W2-rank MoE return of layer L-1 over all dies (ranks
          // 8..30), then the Q ranks' (0..7) first / last gate observation and last QKV start of layer
          // L, and the K/V ranks' (8, 9) last observation; ns after that last W2 return.
          for (int L = 1; L < 36; L++) {
            unsigned long long w2 = 0, qof = ~0ull, qol = 0, qsl = 0, kvl = 0;
            for (int w = 0; w < 248; w++) {
              int const r = w % 31;
              unsigned long long const *a = &g_gs_snap[((size_t)(L - 1) * 256 + w) * 4];
              unsigned long long const *c = &g_gs_snap[((size_t)L * 256 + w) * 4];
              if (r >= 8 && a[0] > w2) {
                w2 = a[0];
              }
              if (r < 8 && c[2] != 0) {
                qof = c[2] < qof ? c[2] : qof;
                qol = c[2] > qol ? c[2] : qol;
                qsl = c[3] > qsl ? c[3] : qsl;
              }
              if ((r == 8 || r == 9) && c[2] > kvl) {
                kvl = c[2];
              }
            }
            if (w2 == 0 || qof == ~0ull) {
              continue;
            }
            auto rel = [&](unsigned long long t) -> long long {
              return t == 0 ? -1ll : ((long long)t - (long long)w2) * 10;
            };
            printf("[GATES] L=%d qobs_first=%lld qobs_last=%lld qstart_last=%lld kvobs_last=%lld\n", L,
                   rel(qof), rel(qol), rel(qsl), rel(kvl));
#ifdef MPK_QEP_SNAP
            {
              // Same base: the epoch marks of layer L (needs MPK_QEP_SNAP == MPK_GATE_SNAP).
              unsigned long long qa = 0, kva = 0, fo = ~0ull, qe = 0, kvs = 0;
              for (int w = 0; w < 248; w++) {
                int const r = w % 31;
                unsigned long long const *e = &g_qep_snap[((size_t)L * 256 + w) * 2];
                unsigned long long const *e2 = &g_qep_snap2[((size_t)L * 256 + w) * 2];
                unsigned long long const *c = &g_gs_snap[((size_t)L * 256 + w) * 4];
                if (r < 8 && e[0] > qa) qa = e[0];
                if ((r == 8 || r == 9) && e[0] > kva) kva = e[0];
                if ((r == 8 || r == 9) && c[3] > kvs) kvs = c[3];
                if (r < 17 && e2[1] != 0 && e2[1] < fo) fo = e2[1];
                if (r < 8 && e[1] > qe) qe = e[1];
              }
              printf("[GATEQ] L=%d kvstart_last=%lld qarr_last=%lld kvarr_last=%lld obs_first=%lld qexit_last=%lld\n",
                     L, rel(kvs), rel(qa), rel(kva), rel(fo == ~0ull ? 0ull : fo), rel(qe));
            }
#endif
          }
#endif
#ifdef MPK_QEP_SNAP
          // Per (layer, die): Q producers' last arrival, K / V arrivals, Q ranks' last epoch exit, K's
          // exit; ns after the die's first arrival in that layer (iteration MPK_QEP_SNAP).
          for (int L = 0; L < 36; L++) {
            for (int d = 0; d < 8; d++) {
              unsigned long long const *q = &g_qep_snap[((size_t)L * 256 + d * 31) * 2];
              unsigned long long b = ~0ull, qa = 0, qe = 0;
              for (int r = 0; r < 31; r++) {
                if (q[r * 2] != 0 && q[r * 2] < b) {
                  b = q[r * 2];
                }
              }
              if (b == ~0ull) {
                continue;
              }
              for (int r = 0; r < 8; r++) {
                qa = q[r * 2] > qa ? q[r * 2] : qa;
                qe = q[r * 2 + 1] > qe ? q[r * 2 + 1] : qe;
              }
              unsigned long long const ka = q[8 * 2], va = q[9 * 2], ke = q[8 * 2 + 1];
              {
                // v2: last arrival over ranks 0..16, first observer, slowest Q exit's split.
                unsigned long long const *q2 = &g_qep_snap2[((size_t)L * 256 + d * 31) * 2];
                unsigned int const *sp = &g_qep_spins[(size_t)L * 256 + d * 31];
                unsigned long long la = 0, fo = ~0ull;
                for (int r = 0; r < 17; r++) {
                  la = q[r * 2] > la ? q[r * 2] : la;
                  if (q2[r * 2 + 1] != 0 && q2[r * 2 + 1] < fo) {
                    fo = q2[r * 2 + 1];
                  }
                }
                int sr = 0;
                for (int r = 1; r < 8; r++) {
                  sr = q[r * 2 + 1] > q[sr * 2 + 1] ? r : sr;
                }
                unsigned long long const sa = q[sr * 2], s0 = q2[sr * 2], s1 = q2[sr * 2 + 1],
                                         se = q[sr * 2 + 1];
                printf("[QEPS2] L=%d d=%d lastarr=%llu firstobs=%llu slowq=%d arr=%llu a2p=%llu p2o=%llu "
                       "o2e=%llu spins=%u\n",
                       L, d, la >= b ? (la - b) * 10 : 0, (fo != ~0ull && fo >= b) ? (fo - b) * 10 : 0, sr,
                       sa >= b ? (sa - b) * 10 : 0, (s0 >= sa && s0) ? (s0 - sa) * 10 : 0,
                       (s1 >= s0 && s0) ? (s1 - s0) * 10 : 0, (se >= s1 && s1) ? (se - s1) * 10 : 0,
                       sp[sr]);
              }
              printf("[QEPS] L=%d d=%d qarr=%llu karr=%llu varr=%llu qexit=%llu kexit=%llu\n", L, d,
                     qa >= b ? (qa - b) * 10 : 0, ka >= b ? (ka - b) * 10 : 0,
                     va >= b ? (va - b) * 10 : 0, qe >= b ? (qe - b) * 10 : 0,
                     ke >= b ? (ke - b) * 10 : 0);
            }
          }
#endif
#ifdef MPK_PHASE_SLOTS
          // One line per (xcd_rank band, slot): mean ns of each slot-to-slot
          // span over every steady-state decode layer. Printed once at
          // termination so nothing lands on the hot path.
          //
          // Emitted per rank band because rank is role here: ranks
          // 0..total_qkv_tiles_per_xcd-1 run QKV and attention, the rest go
          // straight to Phase 6 and only ever run MoE tiles. Averaging the two
          // together reports a worker that does not exist -- the same mistake
          // that made the barrier look uniformly expensive.
          {
            // layers_per_worker must equal 36 * (iters - START_ITER) against
            // the [FWD_PASS_TOTAL] line; arm=0 means the recorder never
            // started and every number below is meaningless.
            printf("[PSLOT] slots=%d layers_per_worker=%llu arm=%d "
                   "total_iters=%d start_iter=%d\n",
                   MPK_PHASE_SLOT_COUNT,
                   g_phase_n[0],
                   (int)g_phase_arm,
                   g_fwdpass_total_iters,
                   MPK_PHASE_START_ITER);
            for (int w = 0; w < MPK_PHASE_MAX_WORKERS; w++) {
              unsigned long long n = g_phase_n[w * MPK_PHASE_PAD_U64];
              if (n == 0) {
                continue;
              }
              // From s=0, not s=1: slot 0 now carries the inter-layer span
              // (see mpk_phase_mark). Emitting it is what makes the row sum
              // to the real per-layer cost instead of leaving a hole.
              printf("[PSLOTW] w=%d n=%llu", w, n);
#ifdef MPK_PHASE_LDS
              for (int s = 0; s <= MPK_PHASE_SLOT_COUNT; s++) {
#else
              for (int s = 0; s < MPK_PHASE_SLOT_COUNT; s++) {
#endif
                printf(" %llu",
                       g_phase_span[w * MPK_PHASE_SLOT_STRIDE + s] / n);
              }
              printf("\n");
              // Raw s_memrealtime ticks (10 ns) of each slot boundary for the
              // last armed layer: a one-layer snapshot of every worker.
              printf("[PHASETS] w=%d", w);
              for (int s = 0; s < MPK_PHASE_SLOT_COUNT; s++) {
                printf(" %llu", g_phase_ts[w * MPK_PHASE_SLOT_COUNT + s]);
              }
              printf("\n");
#ifdef MPK_QKV_SUBSTAMPS
              printf("[QKVSUB] w=%d", w);
              for (int s = 0; s < 8; s++) {
                printf(" %llu", g_qkvsub[w * 8 + s]);
              }
              printf("\n");
#endif
#ifdef MPK_ATTSUB
              printf("[ATTSUB] w=%d", w);
              for (int s = 0; s < 8; s++) {
                printf(" %llu", g_attsub[w * 8 + s]);
              }
              printf("\n");
#endif
#ifdef MPK_ATTSCAN
              printf("[ATTSCAN] w=%d", w);
              for (int s = 0; s < 8; s++) {
                printf(" %llu", g_attscan[w * 8 + s]);
              }
              printf("\n");
#endif
#ifdef MPK_W2SUB
              printf("[W2SUB] w=%d", w);
              for (int s = 0; s < 8; s++) {
                printf(" %llu", g_w2sub[w * 8 + s]);
              }
              printf("\n");
#endif
            }
#ifdef MPK_OPROJ_LDS
            for (int w = 0; w < MPK_PHASE_MAX_WORKERS; w++) {
              unsigned int sn = g_sub_n[w];
              if (sn == 0) {
                continue;
              }
              printf("[PSUBW] w=%d n=%u %llu %llu %llu %llu\n", w, sn,
                     g_sub_span[w * 4 + 0] / sn, g_sub_span[w * 4 + 1] / sn,
                     g_sub_span[w * 4 + 2] / sn, g_sub_span[w * 4 + 3] / sn);
              if (g_tk_n[w] > 0) {
                unsigned int tn = g_tk_n[w];
                printf("[PSUBT] w=%d n=%u %llu %llu %llu\n", w, tn,
                       g_tk_span[w * 3 + 0] / tn, g_tk_span[w * 3 + 1] / tn,
                       g_tk_span[w * 3 + 2] / tn);
              }
              if (g_sub_last_n[w] > 0) {
                printf("[PSUBL] w=%d nl=%u last=%llu\n", w, g_sub_last_n[w],
                       g_sub_last[w] / g_sub_last_n[w]);
              }
            }
#endif
#ifdef MPK_MOE_LDS
            for (int w = 0; w < MPK_PHASE_MAX_WORKERS; w++) {
              unsigned int const a = g_moe_n[w * 2], b = g_moe_n[w * 2 + 1];
              if (a == 0 && b == 0) {
                continue;
              }
              unsigned int const *m = &g_moe_acc[w * 7];
              printf("[MSUBW] w=%d n13=%u dec=%u cmp=%u arr=%u n2=%u "
                     "dec=%u prep=%u bar=%u cmp=%u\n",
                     w, a, a ? m[0] / a : 0, a ? m[1] / a : 0, a ? m[2] / a : 0,
                     b, b ? m[3] / b : 0, b ? m[4] / b : 0, b ? m[5] / b : 0,
                     b ? m[6] / b : 0);
            }
#ifdef MPK_MOE_LDS_MAX
            for (int w = 0; w < MPK_PHASE_MAX_WORKERS; w++) {
              unsigned int const a = g_moe_n[w * 2], b = g_moe_n[w * 2 + 1];
              if (a == 0 && b == 0) {
                continue;
              }
              unsigned int const *x = &g_moe_x[w * 4];
              printf("[MSUBX] w=%d n13=%u arrmax=%u arr1us=%u n2=%u barmax=%u bar1us=%u\n",
                     w, a, x[0], x[1], b, x[2], x[3]);
            }
#endif
#endif
#ifdef MPK_MOE_CALL_LDS
            for (int w = 0; w < MPK_PHASE_MAX_WORKERS; w++) {
              unsigned int const *c = &g_moe_call[w * 4];
              if (c[2] == 0 && c[3] == 0) {
                continue;
              }
              printf("[MCALLW] w=%d n13=%u call13=%u n2=%u call2=%u\n", w, c[2],
                     c[2] ? c[0] / c[2] : 0, c[3], c[3] ? c[1] / c[3] : 0);
            }
#endif
#ifdef MPK_PHASE_SNAP
            for (int L = 0; L < MPK_PHASE_LAYERS_PER_ITER; L++) {
              unsigned long long mn[MPK_PHASE_SLOT_COUNT];
              unsigned long long mx[MPK_PHASE_SLOT_COUNT];
              for (int s = 0; s < MPK_PHASE_SLOT_COUNT; s++) {
                mn[s] = ~0ull;
                mx[s] = 0;
              }
              for (int w = 0; w < MPK_PHASE_MAX_WORKERS; w++) {
                for (int s = 0; s < MPK_PHASE_SLOT_COUNT; s++) {
                  unsigned long long const v =
                      MPK_PSNAP_RD[((size_t)L * MPK_PHASE_MAX_WORKERS + w) *
                                       MPK_PHASE_SLOT_COUNT +
                                   s];
                  if (v == 0) {
                    continue;
                  }
                  mn[s] = v < mn[s] ? v : mn[s];
                  mx[s] = v > mx[s] ? v : mx[s];
                }
              }
              unsigned long long const b = mn[0];
              printf("[PSNAP] L=%d B=%llu", L, b * 10ull);
              for (int s = 0; s < MPK_PHASE_SLOT_COUNT; s++) {
                printf(" %llu:%llu",
                       (mn[s] != ~0ull && mn[s] >= b) ? (mn[s] - b) * 10 : 0,
                       mx[s] >= b ? (mx[s] - b) * 10 : 0);
              }
              printf("\n");
#if defined(MPK_ATS_OPS)
              if (L >= 1 && L <= 8) {
                unsigned long long const *ao = MPK_ATS_RD;
                for (int w = 0; w < MPK_PHASE_MAX_WORKERS; w++) {
                  unsigned long long const *q = &ao[((size_t)L * MPK_PHASE_MAX_WORKERS + w) * 16];
                  bool any = false;
                  for (int k = 0; k < 16; k++) any |= q[k] != 0;
                  if (!any) continue;
                  printf("[ATSOPS] L=%d w=%d", L, w);
                  for (int k = 0; k < 15; k++) {
                    printf(" %lld", q[k] ? ((long long)q[k] - (long long)b) * 10 : -1ll);
                  }
                  printf(" f=%llu |", q[15]);
                  for (int s = 0; s < MPK_PHASE_SLOT_COUNT; s++) {
                    unsigned long long const v =
                        MPK_PSNAP_RD[((size_t)L * MPK_PHASE_MAX_WORKERS + w) * MPK_PHASE_SLOT_COUNT + s];
                    printf(" %lld", v ? ((long long)v - (long long)b) * 10 : -1ll);
                  }
                  printf("\n");
                }
              }
#endif
#ifdef MPK_TK_OPS
              if (L >= 1 && L <= 6) {
                unsigned long long const *tko = g_tk_ops_dst != nullptr ? g_tk_ops_dst : g_tk_ops;
                for (int w = 0; w < MPK_PHASE_MAX_WORKERS; w++) {
                  unsigned long long const *q = &tko[((size_t)L * MPK_PHASE_MAX_WORKERS + w) * 16];
                  bool any = false;
                  for (int k = 0; k < 16; k++) any |= q[k] != 0;
                  if (!any) continue;
                  printf("[TKOPS] L=%d w=%d", L, w);
                  for (int k = 0; k < 16; k++) {
                    printf(" %lld", q[k] ? ((long long)q[k] - (long long)b) * 10 : -1ll);
                  }
                  printf(" |");
                  for (int s = 5; s <= 7; s++) {
                    unsigned long long const v =
                        MPK_PSNAP_RD[((size_t)L * MPK_PHASE_MAX_WORKERS + w) * MPK_PHASE_SLOT_COUNT + s];
                    printf(" %lld", v ? ((long long)v - (long long)b) * 10 : -1ll);
                  }
                  printf("\n");
                }
              }
#endif
            }
            {
              // Per die: the die's last worker at slots 7 / 8 / 9, ns after the
              // layer's first entry, mean over layers; and how often the die
              // was last to finish the MoE (slot 8) / arrive at Phase 9 (slot 9).
              constexpr int kWpd = MPK_PHASE_MAX_WORKERS / 8;
              int const wpd = (int)(MPK_PSNAPD_WPD);
              unsigned long long sum[8][3] = {};
              int last8[8] = {}, last9[8] = {}, nl = 0;
              for (int L = 0; L < MPK_PHASE_LAYERS_PER_ITER; L++) {
                unsigned long long b = ~0ull;
                for (int w = 0; w < 8 * wpd && w < MPK_PHASE_MAX_WORKERS; w++) {
                  unsigned long long const v =
                      MPK_PSNAP_RD[((size_t)L * MPK_PHASE_MAX_WORKERS + w) *
                                   MPK_PHASE_SLOT_COUNT];
                  if (v != 0 && v < b) {
                    b = v;
                  }
                }
                if (b == ~0ull) {
                  continue;
                }
                unsigned long long dmx[8][3] = {};
                for (int d = 0; d < 8; d++) {
                  for (int r = 0; r < wpd; r++) {
                    int const w = d * wpd + r;
                    for (int k = 0; k < 3; k++) {
                      unsigned long long const v =
                          MPK_PSNAP_RD[((size_t)L * MPK_PHASE_MAX_WORKERS + w) *
                                           MPK_PHASE_SLOT_COUNT +
                                       7 + k];
                      if (v > dmx[d][k]) {
                        dmx[d][k] = v;
                      }
                    }
                  }
                }
                int a8 = 0, a9 = 0;
                for (int d = 0; d < 8; d++) {
                  for (int k = 0; k < 3; k++) {
                    sum[d][k] += dmx[d][k] >= b ? (dmx[d][k] - b) * 10 : 0;
                  }
                  a8 = dmx[d][1] > dmx[a8][1] ? d : a8;
                  a9 = dmx[d][2] > dmx[a9][2] ? d : a9;
                }
                last8[a8]++;
                last9[a9]++;
                nl++;
              }
              (void)kWpd;
              for (int d = 0; d < 8 && nl > 0; d++) {
                printf("[PSNAPD] die=%d moe_start_last=%llu moe_end_last=%llu "
                       "p9_arrive_last=%llu last_moe_end=%d last_p9=%d of %d\n",
                       d, sum[d][0] / nl, sum[d][1] / nl, sum[d][2] / nl,
                       last8[d], last9[d], nl);
              }
            }
#ifdef MPK_VTN_MOETL
            for (int L = MPK_VTN_MOETL_L0;
                 L < MPK_VTN_MOETL_L0 + MPK_VTN_MOETL_NL && L < MPK_PHASE_LAYERS_PER_ITER; L++) {
              unsigned long long b = ~0ull;
              for (int w = 0; w < MPK_PHASE_MAX_WORKERS; w++) {
                unsigned long long const v =
                    MPK_PSNAP_RD[((size_t)L * MPK_PHASE_MAX_WORKERS + w) * MPK_PHASE_SLOT_COUNT];
                if (v != 0 && v < b) b = v;
              }
              if (b == ~0ull) continue;
              for (int w = 0; w < MPK_PHASE_MAX_WORKERS; w++) {
                unsigned long long const *ps =
                    &MPK_PSNAP_RD[((size_t)L * MPK_PHASE_MAX_WORKERS + w) * MPK_PHASE_SLOT_COUNT];
                if (ps[0] == 0) continue;
                unsigned long long const *tl =
                    &MPK_VTN_TL_RD[((size_t)L * MPK_PHASE_MAX_WORKERS + w) * MPK_VTN_TLC * 3];
                printf("[MTL] L=%d w=%d s7=%lld s8=%lld", L, w, (long long)(ps[7] - b) * 10,
                       (long long)(ps[8] - b) * 10);
                for (int c = 0; c < MPK_VTN_TLC; c++) {
                  if ((tl[3 * c + 2] >> 63) == 0) break;
                  printf(" %llu:%lld:%lld", tl[3 * c + 2] & 0xFFFFFFFFull,
                         (long long)(tl[3 * c] - b) * 10, (long long)(tl[3 * c + 1] - b) * 10);
                }
                printf("\n");
              }
            }
#endif
#endif
#ifdef MPK_IL_LDS
            for (int w = 0; w < MPK_PHASE_MAX_WORKERS; w++) {
              unsigned int const n_il = g_il_n[w];
              if (n_il == 0) {
                continue;
              }
              printf("[ILSUBW] w=%d n=%u ret=%u setup=%u call=%u\n", w, n_il,
                     g_il_acc[w * 3] / n_il, g_il_acc[w * 3 + 1] / n_il,
                     g_il_acc[w * 3 + 2] / n_il);
            }
#endif
#ifdef MPK_QKV_SUB_LDS
            for (int w = 0; w < MPK_PHASE_MAX_WORKERS; w++) {
              unsigned int const nq = g_qs_n[w * 3], nc = g_qs_n[w * 3 + 1];
              if (nq == 0 && nc == 0) {
                continue;
              }
              unsigned int const *a = &g_qs_acc[w * 5];
              printf("[QSUBW] w=%d x=%d nq=%u drain=%u arrive=%u wait=%u nc=%u "
                     "cdrain=%u carrive=%u\n",
                     w, (int)g_qs_n[w * 3 + 2] - 1, nq, nq ? a[0] / nq : 0,
                     nq ? a[1] / nq : 0, nq ? a[2] / nq : 0, nc,
                     nc ? a[3] / nc : 0, nc ? a[4] / nc : 0);
            }
#endif
#ifdef MPK_OPROJ_SUB
            for (int w = 0; w < MPK_PHASE_MAX_WORKERS; w++) {
              unsigned int const n = g_os_n[w];
              if (n == 0) {
                continue;
              }
              unsigned int const *a = &g_os_acc[w * 5];
              printf("[OPSUB] w=%d n=%u wdrain=%u wait=%u mfma=%u store=%u combine=%u\n",
                     w, n, a[0] / n, a[1] / n, a[2] / n, a[3] / n, a[4] / n);
            }
#endif
#ifdef MPK_TOPK_SUB
            for (int w = 0; w < MPK_PHASE_MAX_WORKERS; w++) {
              unsigned int const n = g_tk_n[w];
              if (n == 0) {
                continue;
              }
              unsigned int const *a = &g_tk_acc[w * 4];
              printf("[TOPKSUB] w=%d n=%u wait=%u gather=%u scores=%u pick=%u\n",
                     w, n, a[0] / n, a[1] / n, a[2] / n, a[3] / n);
#ifdef MPK_TOPK_SUB2
              {
                unsigned int const *b = &g_tk2_acc[w * 8];
                printf("[TOPKSUB2] w=%d n=%u g_w0=%u g_bar=%u p_bar1=%u p_ksel=%u p_bar2=%u "
                       "p_ret=%u retry=%u\n",
                       w, n, b[0] / n, b[1] / n, b[2] / n, b[3] / n, b[4] / n, b[5] / n,
                       (unsigned int)((unsigned long long)b[6] * 1000ull / n));
              }
#endif
            }
#endif
#ifdef MPK_QKVK_LDS
            for (int w = 0; w < MPK_PHASE_MAX_WORKERS; w++) {
              unsigned int const n = g_qk_n[w];
              if (n == 0) {
                continue;
              }
              unsigned int const *a = &g_qk_acc[w * 4];
              printf("[QKVKW] w=%d n=%u foldnorm=%u ready=%u setup=%u mfma_epi=%u\n",
                     w, n, a[0] / n, a[1] / n, a[2] / n, a[3] / n);
            }
#endif
#ifdef MPK_ATTK_LDS
            for (int w = 0; w < MPK_PHASE_MAX_WORKERS; w++) {
              for (int c = 0; c < 24; c++) {
                unsigned int const n = g_ak_cnt[w * 24 + c];
                if (n == 0) {
                  continue;
                }
                printf("[ATTKW] w=%d sw=%d slot=%d n=%u ns=%u\n", w, c / 12,
                       c % 12, n, g_ak_acc[w * 24 + c] / n);
              }
            }
#endif
#ifdef MPK_W13_SUB
            for (int w = 0; w < MPK_PHASE_MAX_WORKERS; w++) {
              unsigned int const n13 = g_w13_n[w];
              if (n13 == 0) {
                continue;
              }
              unsigned int const *a = &g_w13_acc[w * 10];
              printf("[W13SUB] w=%d n=%u %u %u %u %u %u %u %u %u %u %u hw=0x%x\n", w, n13,
                     a[0] / n13, a[1] / n13, a[2] / n13, a[3] / n13, a[4] / n13,
                     a[5] / n13, a[6] / n13, a[7] / n13, a[8] / n13, a[9] / n13,
                     g_w13_hwid[w]);
            }
#endif
            // Raw per-slot timestamps of the last armed layer, so one layer's
            // timeline can be reconstructed per worker (worker % 8 == XCD) and
            // exported as a Perfetto trace. The span table above is a mean and
            // cannot show ordering or overlap; this can.
            for (int w = 0; w < MPK_PHASE_MAX_WORKERS; w++) {
              if (g_phase_n[w * MPK_PHASE_PAD_U64] == 0) {
                continue;
              }
              printf("[PHASETS] w=%d", w);
              for (int s = 0; s < MPK_PHASE_SLOT_COUNT; s++) {
                printf(" %llu", g_phase_ts[w * MPK_PHASE_SLOT_STRIDE + s]);
              }
              printf("\n");
            }
#if defined(MPK_LTK_SNAP) && defined(MPK_PHASE_SNAP)
            for (int L = 0; L < MPK_PHASE_LAYERS_PER_ITER; L++) {
              unsigned long long pub[2] = {0, 0}, okmin[2] = {~0ull, ~0ull};
              unsigned long long okmax[2] = {0, 0}, p0min = ~0ull;
              unsigned long long endmin = ~0ull, endmax = 0, oksum = 0;
              int pubn = 0, okn = 0, pubw = -1;
              for (int w = 0; w < MPK_PHASE_MAX_WORKERS; w++) {
                unsigned long long const *v =
                    &g_ltk_snap[((size_t)L * MPK_PHASE_MAX_WORKERS + w) * 4];
                int const a = (w / MPK_PSNAPD_WPD) >= 4 ? 1 : 0;
                if (v[0]) {
                  if (v[0] > pub[a]) pub[a] = v[0];
                  if (v[0] >= pub[0] && v[0] >= pub[1]) pubw = w;
                  pubn++;
                }
                if (v[1] && v[1] < p0min) p0min = v[1];
                if (v[2]) {
                  if (v[2] < okmin[a]) okmin[a] = v[2];
                  if (v[2] > okmax[a]) okmax[a] = v[2];
                  oksum += v[2];
                  okn++;
                }
                if (v[3]) {
                  if (v[3] < endmin) endmin = v[3];
                  if (v[3] > endmax) endmax = v[3];
                }
              }
              if (pubn == 0 || okn == 0) {
                continue;
              }
              printf("[LTKSNAP] L=%d pubn=%d pubw=%d pub0=%llu pub1=%llu p0min=%llu "
                     "okmin0=%llu okmin1=%llu okmax0=%llu okmax1=%llu okmean=%llu "
                     "endmin=%llu endmax=%llu okn=%d\n",
                     L, pubn, pubw, pub[0], pub[1], p0min, okmin[0], okmin[1],
                     okmax[0], okmax[1], oksum / okn, endmin, endmax, okn);
            }
#endif
#if defined(MPK_XSNAP) && defined(MPK_PHASE_SNAP)
            for (int L = 0; L < MPK_PHASE_LAYERS_PER_ITER; L++) {
              unsigned long long b = ~0ull;
              for (int w = 0; w < MPK_PHASE_MAX_WORKERS; w++) {
                unsigned long long const v =
                    MPK_PSNAP_RD[((size_t)L * MPK_PHASE_MAX_WORKERS + w) *
                                 MPK_PHASE_SLOT_COUNT];
                if (v && v < b) b = v;
              }
              if (b == ~0ull) continue;
              printf("[XSNAPB] L=%d b=%llu\n", L, b);
#ifdef MPK_XSNAP_WORKERS
              if (L >= 1 && L <= (MPK_XSNAP_WORKERS)) {
                for (int w = 0; w < MPK_PHASE_MAX_WORKERS; w++) {
                  printf("[XSW] L=%d w=%d", L, w);
                  for (int k = 0; k < 16; k++) {
                    unsigned long long const v =
                        g_xs_snap[((size_t)L * MPK_PHASE_MAX_WORKERS + w) * 16 + k];
                    printf(" %lld", v ? ((long long)v - (long long)b) * 10 : -1ll);
                  }
                  printf(" |");
                  for (int k = 0; k < MPK_PHASE_SLOT_COUNT; k++) {
                    unsigned long long const v =
                        MPK_PSNAP_RD[((size_t)L * MPK_PHASE_MAX_WORKERS + w) *
                                         MPK_PHASE_SLOT_COUNT + k];
                    printf(" %lld", v ? ((long long)v - (long long)b) * 10 : -1ll);
                  }
                  printf("\n");
                }
              }
#endif
              int const wpd = (int)(MPK_PSNAPD_WPD);
              for (int k = 0; k < 16; k++) {
                printf("[XSNAP] L=%d k=%d", L, k);
                for (int x = 0; x < 8; x++) {
                  unsigned long long mn = ~0ull, mx = 0;
                  for (int w = x * wpd; w < (x + 1) * wpd && w < MPK_PHASE_MAX_WORKERS; w++) {
                    unsigned long long const v =
                        g_xs_snap[((size_t)L * MPK_PHASE_MAX_WORKERS + w) * 16 + k];
                    if (!v) continue;
                    mn = v < mn ? v : mn;
                    mx = v > mx ? v : mx;
                  }
                  if (mx == 0) printf(" -1:-1");
                  else {
#ifdef MPK_XSNAP_LAT
                    long long const bb = (k == 12 || k >= 14) ? 0 : (long long)b;
#else
                    long long const bb = (long long)b;
#endif
                    printf(" %lld:%lld", ((long long)mn - bb) * 10,
                           ((long long)mx - bb) * 10);
                  }
                }
                printf("\n");
              }
            }
#endif
          }
#endif
#ifdef MPK_QKV_SPLIT
          if (g_qkv_n > 0) {
            printf("[QKVSPLIT] n=%llu setup=%llu gemm=%llu total=%llu\n",
                   g_qkv_n,
                   g_qkv_setup_sum / g_qkv_n,
                   g_qkv_gemm_sum / g_qkv_n,
                   (g_qkv_setup_sum + g_qkv_gemm_sum) / g_qkv_n);
          }
#endif
#ifdef MPK_INTERLAYER_SPLIT
          // Accounts the part of the slot-11 -> slot-0 span that lives inside
          // the fused-layer function, which no other counter can see.
          if (g_il_n > 0) {
            unsigned long long n = g_il_n;
            printf("[ILSPLIT] n=%llu fnpre_ns=%llu binv_ns=%llu post_ns=%llu "
                   "total_ns=%llu\n",
                   n,
                   g_il_fnpre_sum / n,
                   g_il_binv_sum / n,
                   g_il_post_sum / n,
                   (g_il_fnpre_sum + g_il_binv_sum + g_il_post_sum) / n);
          }
#endif
#ifdef MPK_ILSUB
          for (int w = 0; w < 1024; w++) {
            if (g_ilsub[w * 32 + 10] == 0) {
              continue;
            }
            printf("[ILSUB] b=%d", w);
            for (int s = 0; s < 32; s++) {
              printf(" %llu", g_ilsub[w * 32 + s]);
            }
            printf("\n");
          }
#ifdef MPK_ILPER
          for (int w = 0; w < 1024; w++) {
            if (g_ilsub[w * 32 + 10] == 0) {
              continue;
            }
            printf("[ILPER] b=%d %llu", w, g_ilsub[w * 32 + 10]);
            for (int s = 0; s < 36; s++) {
              printf(" %llu", g_ilper[w * 128 + s]);
            }
            for (int s = 0; s < 36; s++) {
              printf(" %llu", g_ilper[w * 128 + 64 + s]);
            }
            printf("\n");
          }
#endif
#endif
#ifdef MPK_HWID_MAP
          for (int w = 0; w < 1024; w++) {
            if (g_hwid_w[w] != 0) {
              printf("[HWID] w b=%d xcc=%u hw=%x\n", w,
                     (unsigned)(g_hwid_w[w] >> 32), (unsigned)g_hwid_w[w]);
            }
          }
          for (int s = 0; s < 64; s++) {
            if (g_hwid_s[s] != 0) {
              printf("[HWID] s b=%d xcc=%u hw=%x\n", s,
                     (unsigned)(g_hwid_s[s] >> 32), (unsigned)g_hwid_s[s]);
            }
          }
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
#ifdef MPK_ITER_AID
            // Two stores, not eight, and into AID-local MTYPE_RW replicas
            // rather than the 512 B gpu_malloc line -- which is plain
            // spanning VRAM, i.e. MTYPE_NC wholly inside ONE range, so half
            // the XCDs were polling it remotely AND uncached.
            mpk_aid_iter_publish((unsigned)new_val);
            if (mpk_aid_iter_slot(0) == nullptr) {
              for (int x = 0; x < MPK_NUM_XCDS; x++) {
                st_wt_u32((void *)&rel[x * 16], (unsigned)new_val);
              }
            }
#else
            for (int x = 0; x < MPK_NUM_XCDS; x++) {
              st_wt_u32((void *)&rel[x * 16], (unsigned)new_val);
            }
#endif
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          }
#endif
          terminate_schedulers(config);
#ifdef MPK_END_POLL
          ep_ev = -1;  // the termination event comes through the own queue
#endif
        } else {
          unsigned long long prep_done_clk = get_wallclock_ns();
#ifdef MPK_END_POLL
          if (ep_ev >= 0) {
            ep_target += ep_step;
          } else if (!ep_drop_end && !ep_got && (sched_id != 0 || cur_event_pos[0] > 0)) {
            // an end event the marker pushed (the launch's first one is the host's, scheduler 0 slot 0): the
            // event the marker waits on is the one that launches it
            size_t const end_pos = get_event_position_index(event_id);
            int found = -1;
            for (int ev = config.num_events - 1; ev >= 0 && found < 0; ev--) {
              EventDesc const d = config.all_events[ev];
              if (d.first_task_id == TASK_INVALID_ID || d.last_task_id == TASK_INVALID_ID) {
                continue;
              }
              size_t const t0 = get_task_position_index(d.first_task_id);
              size_t const t1 = get_task_position_index(d.last_task_id);
              for (size_t t = t0; t < t1 && t < t0 + 4; t++) {
                EventId const tr = config.all_tasks[t].trigger_event;
                if (tr != EVENT_INVALID_ID && get_event_position_index(tr) == end_pos) {
                  found = ev;
                  break;
                }
              }
            }
            if (found >= 0) {
              EventCounter const st = (EventCounter)MPK_LD_EVCFG(&config.all_event_num_triggers[found]);
              EventCounter const c = (EventCounter)MPK_EVCTR_READ(config.all_event_counters, found);
              // the marker saw exactly st * (its pass) before pushing; anything else: stay on the queue
              if (st > 0 && c > 0 && c % st == 0) {
                ep_ev = found;
                ep_step = st;
                ep_target = c + st;
                ep_end_id = event_id;
                ep_drop_end = true;
#ifdef MPK_END_POLL_DBG
                ep_armed_at = (unsigned)end_of_graph_count;
#endif
              }
            }
          }
          ep_rel_clk = get_wallclock_ns();
          ep_quiet_until = ep_rel_clk + ep_quiet_ns;
#endif
#ifdef MPK_PRECOMPUTED_DISPATCH
          // Signal workers to start next iteration.
#ifdef MPK_ITER_DELAY_NS
          // MPK_ITER_DELAY_V1: timing probe.
          while (get_wallclock_ns() < prep_done_clk + (MPK_ITER_DELAY_NS)) {
          }
#endif
#ifdef MPK_ITER_TS
          __atomic_store_n(&g_its_rel_ts, prep_done_clk, __ATOMIC_RELAXED);
#endif
          // iter_ready starts at 0; every END_OF_TASK_GRAPH bumps it.
          // First bump (from prepare_kernel) enables iteration 1.
#ifdef MPK_END_DIRECT
          if (!end_direct) {
#endif
          __threadfence();
#ifdef MPK_REL_RET
          unsigned long long const rel_ret = atomicAdd(config.precomp_iter_ready, 1ULL) + 1ULL;
#else
          atomicAdd(config.precomp_iter_ready, 1ULL);
#endif
          // Fan out per-XCD release flags via st_wt (write-through to HBM)
          // so workers can poll via ld_nt without cross-XCD L2 contention.
          {
#ifdef MPK_REL_RET
            unsigned long long new_val = rel_ret;  // what the load behind the add read, without its round trip
#else
            unsigned long long new_val = *config.precomp_iter_ready;
#endif
            int *rel = config.precomp_iter_xcd_release;
#ifdef MPK_ITER_AID
            // Two stores, not eight, and into AID-local MTYPE_RW replicas
            // rather than the 512 B gpu_malloc line -- which is plain
            // spanning VRAM, i.e. MTYPE_NC wholly inside ONE range, so half
            // the XCDs were polling it remotely AND uncached.
            mpk_aid_iter_publish((unsigned)new_val);
            if (mpk_aid_iter_slot(0) == nullptr) {
              for (int x = 0; x < MPK_NUM_XCDS; x++) {
                st_wt_u32((void *)&rel[x * 16], (unsigned)new_val);
              }
            }
#else
            for (int x = 0; x < MPK_NUM_XCDS; x++) {
              st_wt_u32((void *)&rel[x * 16], (unsigned)new_val);
            }
#endif
#ifdef MPK_ITER_TS
          {
            unsigned long long const its_t4 = get_wallclock_ns();
            unsigned long long const its_t0 =
                __atomic_load_n(&g_its_argmax_end, __ATOMIC_RELAXED);
            if (g_fwdpass_total_iters >= 40 && its_t0 != 0 &&
                its_t0 <= iter_end_clk && iter_end_clk - its_t0 < 1000000ull) {
              g_its_sum[0] += iter_end_clk - its_t0;
              g_its_sum[1] += its_t1b - iter_end_clk;
              g_its_sum[2] += its_t2 - its_t1b;
              g_its_sum[3] += prep_done_clk - its_t2;
              g_its_sum[4] += its_t4 - prep_done_clk;
              {
                unsigned long long const pe =
                    __atomic_load_n(&g_its_push_end, __ATOMIC_RELAXED);
                if (pe != 0 && pe <= iter_end_clk && iter_end_clk - pe < 1000000ull) {
                  g_its_sum[5] += iter_end_clk - pe;
                }
              }
              g_its_n++;
            }
          }
#endif
#ifdef MPK_TAIL_PUB_LATE
          // After the release: the next iteration does not need it, and the
          // value read now (after prepare_next_batch) already names the token
          // this iteration produced.
          if (config.progress_host != nullptr) {
#ifdef MPK_TOK_HOST // request 0's newest token id beside its position (bs=1 streaming)
            {
              int const tp = (int)config.step[0];
              for (int _j = 0; _j < 16; _j++) {  // spec: backfill accepted positions committed this pass
                int const _p = tp - _j;
                if (_p < 0 || _p >= MPK_MAX_SEQ_LENGTH) break;
                config.progress_host[MPK_MAX_NUM_BATCHED_REQUESTS + _p] = (int)__hip_atomic_load(
                    &config.tokens[_p], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
              }
            }
#endif
            __threadfence_system();
            for (int r = 0; r < MPK_MAX_NUM_BATCHED_REQUESTS; r++) {
              __atomic_store_n(&config.progress_host[r], config.step[r],
                               __ATOMIC_RELEASE);
            }
          }
#endif
#ifdef MPK_BOOK_LATE
            book_late();
#endif
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          }
#ifdef MPK_END_DIRECT
          }
#endif
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
#ifdef MPK_END_POLL
      if (!ep_got)
#endif
      cur_event_pos[queue_idx] += 1;
    }
  }
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
__global__ __launch_bounds__(WORKER_NUM_THREADS,
                             1) void persistent_kernel(RuntimeConfig config) {
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

#ifdef MPK_PLACE_GUARD
// MPK_PLACE_GUARD_V1: a block that never gets a CU deadlocks the launch (a worker fills a CU, so nothing can share
// one). Every block checks in at entry, before it touches shared state; the first block that sees all of them
// sets GO, a block whose MPK_PLACE_GUARD ms run out first sets ABORT, and every block follows the one decision.
static unsigned *g_mpk_place_h = nullptr;  // MPK_PLACE_SERIAL_V1: host view of the check-in words
__device__ unsigned *g_place_host = nullptr;  // pinned: [0] check-ins, [1] 0 pending / 1 go / 2 abort, [2] blocks
__device__ __forceinline__ bool mpk_place_guard_abort(int is_sched) {  // MPK_PLACE_GUARD_V2 MPK_PLACE_GUARD_V3
  // No __shared__ here: a static LDS variable moves the dynamic LDS rings off 16-byte alignment (1.9x slower).
  unsigned *const g = g_place_host;
  if (g == nullptr) {
    return false;
  }
  if (threadIdx.x == 0) {
    unsigned const want = __hip_atomic_load(&g[2], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    unsigned long long const t0 = __builtin_amdgcn_s_memrealtime();
#ifdef MPK_MULTI_DEV
    int const slot = is_sched ? (int)g[3] + g_mpk_dev_xcd_base + (int)blockIdx.x
                              : g_mpk_dev_worker_base + (int)blockIdx.x;
#else
    int const slot = is_sched ? (int)g[3] + (int)blockIdx.x : (int)blockIdx.x;
#endif
    if (slot >= 0 && slot < 1000) {
      __hip_atomic_store((unsigned long long *)(g + 16) + slot, t0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    __hip_atomic_fetch_add(&g[0], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    while (__hip_atomic_load(&g[1], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) == 0u) {
      unsigned e = 0u;
      if (__hip_atomic_load(&g[0], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) >= want) {
        __hip_atomic_compare_exchange_strong(&g[1], &e, 1u, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                             __HIP_MEMORY_SCOPE_SYSTEM);
      } else if (__builtin_amdgcn_s_memrealtime() - t0 > (unsigned long long)(MPK_PLACE_GUARD) * 100000ull) {
        __hip_atomic_compare_exchange_strong(&g[1], &e, 2u, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                             __HIP_MEMORY_SCOPE_SYSTEM);
      } else {
        __builtin_amdgcn_s_sleep(8);
      }
    }
  }
  __syncthreads();
  // g[1] only changes from 0, so every thread reads the decision thread 0 waited for.
  return __hip_atomic_load(&g[1], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) == 2u;
}
#endif
__global__ __launch_bounds__(WORKER_NUM_THREADS,
                             1) void worker_kernel(RuntimeConfig config) {
#ifdef MPK_RESID_MAP
  if (threadIdx.x == 0 && blockIdx.x < 1024) {
#ifdef MPK_MULTI_DEV // MDEV_RESID
    mpk_resid_mark(g_mpk_dev_worker_base + (int)blockIdx.x);
#else
    mpk_resid_mark(blockIdx.x);
#endif
  }
#endif
#ifdef MPK_PLACE_GUARD
  if (mpk_place_guard_abort(0)) {
    return;
  }
#endif
  worker_checker(config);
#ifdef MPK_CPX_SCOPE // MDEV_CPX_BOUNDARY: no line another kernel left in this XCC's L2
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
#endif
#ifdef MPK_LDS_PAD
  // Static LDS pad of MPK_LDS_PAD bytes: shifts every later LDS offset
  // (and the dynamic base) without adding any work. The asm keeps it live.
  __shared__ unsigned char s_mpk_lds_pad[MPK_LDS_PAD];
  asm volatile("" ::"v"((unsigned)(size_t)s_mpk_lds_pad));
#endif
#ifdef MPK_HWID_MAP
  if (threadIdx.x == 0 && blockIdx.x < 1024) {
    g_hwid_w[blockIdx.x] = mpk_hwid_now();
  }
#endif
#ifdef MPK_XCD_SUBSET
  // Elect by hardware die, never by blockIdx: this gate retires most of the
  // grid at once, the freed CUs are refilled out of round-robin order, and a
  // block's index then says nothing about where it ran. Each surviving block
  // claims a dense rank on its own die, so the participating XCDs see exactly
  // workers_per_xcd workers however the dispatcher placed them.
  __shared__ int s_subset_wid;
  if (threadIdx.x == 0) {
    int phys;
    MPK_GET_XCC_ID16(phys);
    s_subset_wid = -1;
    // get_current_xcd_id() folds with %MPK_NUM_XCDS, which only agrees with
    // (phys - MPK_XCD_BASE) when the base is a whole number of subsets in.
    static_assert(MPK_XCD_BASE % MPK_NUM_XCDS == 0,
                  "MPK_XCD_BASE must be a multiple of MPK_NUM_XCDS");
    int const lx = phys - MPK_XCD_BASE;
    if (lx >= 0 && lx < MPK_NUM_XCDS) {
      int const wpx = config.precomp_workers_per_xcd > 0
                          ? config.precomp_workers_per_xcd
                          : config.num_workers / MPK_NUM_XCDS;
      int const rank = atomicAdd(&g_xcd_subset_rank[lx], 1);
      if (rank < wpx) {
        s_subset_wid = rank * MPK_NUM_XCDS + lx;
      }
    }
  }
  __syncthreads();
  if (s_subset_wid < 0) {
    return;
  }
  execute_worker(config, s_subset_wid);
#elif defined(MPK_MULTI_DEV) // MDEV_WORKER_ID
  execute_worker(config, g_mpk_dev_worker_base + (int)blockIdx.x);
#else
  execute_worker(config);
#endif
}

__global__ void scheduler_kernel(RuntimeConfig config) {
#ifdef MPK_RESID_MAP
  if (threadIdx.x == 0 && blockIdx.x < 64) {
#ifdef MPK_MULTI_DEV // MDEV_RESID
    mpk_resid_mark(1024 + g_mpk_dev_xcd_base + (int)blockIdx.x);
#else
    mpk_resid_mark(1024 + blockIdx.x);
#endif
  }
#endif
#ifdef MPK_PLACE_GUARD
  if (mpk_place_guard_abort(1)) {
    return;
  }
#endif
  scheduler_checker(config);
#ifdef MPK_CPX_SCOPE // MDEV_CPX_BOUNDARY: no line another kernel left in this XCC's L2
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
#endif
#ifdef MPK_HWID_MAP
  if (threadIdx.x == 0 && blockIdx.x < 64) {
    g_hwid_s[blockIdx.x] = mpk_hwid_now();
  }
#endif
#ifdef MPK_XCD_SUBSET
  // One scheduler per participating die, claimed by the first block to land
  // there. The whole block leaves together, so the syncthreads stays uniform
  // -- and it must happen before the return, because execute_scheduler runs
  // thread 0 only and cannot be entered divergently.
  __shared__ int s_sched_id;
  if (threadIdx.x == 0) {
    int phys;
    MPK_GET_XCC_ID16(phys);
    s_sched_id = -1;
    int const lx = phys - MPK_XCD_BASE;
    if (lx >= 0 && lx < MPK_NUM_XCDS &&
        atomicAdd(&g_xcd_sched_rank[lx], 1) == 0) {
      s_sched_id = lx;
    }
  }
  __syncthreads();
  if (s_sched_id < 0) {
    return;
  }
  execute_scheduler(config, 0, s_sched_id);
#elif defined(MPK_MULTI_DEV) // MDEV_SCHED_ID
  execute_scheduler(config, 0, get_current_xcd_id());
#else
  execute_scheduler(config, 0);
#endif
}
#pragma clang diagnostic pop

#ifdef MPK_AID_LOCAL
#include "aid_local.h"
#endif

// Allocator for state that several XCDs read and write.
//
// Under DPX+NPS2 a device is 4 XCDs spanning two AIDs, and device VRAM there
// is coherent only within one AID -- so a counter incremented on one die is
// never observed by the two XCDs on the other. Host memory is system-coherent,
// so every XCD sees the same value. The buffers routed here are a few KB of
// counters polled in spin loops, not bandwidth-bound data.
//
// MPK_UNCACHED_SYNC prefers UNCACHED DEVICE VRAM over host memory. It needs
// the driver's aid_local_uncached_mtype=1, which maps an UNCACHED-marked BO to
// MTYPE_NC instead of upstream's MTYPE_UC -- NC is the one MTYPE this
// protocol was measured correct on, and UC drops the coherence point
// device-scope atomics resolve at, so these counters' atomicAdd would never
// land. Staying on the device avoids a PCIe round trip per poll, which is
// what made the host-memory arm 21.3 ms/token against 3.9 in DPX+NPS1.
template <typename DT>
DT *mpk_sync_malloc(size_t size) {
  static bool const use_uncached = (getenv("MPK_UNCACHED_SYNC") != nullptr &&
                                    atoi(getenv("MPK_UNCACHED_SYNC")) != 0);
  static bool const use_host = (getenv("MPK_HOST_EVCTR") != nullptr &&
                                atoi(getenv("MPK_HOST_EVCTR")) != 0);
  void *dst_ptr = nullptr;
  if (use_uncached) {
    if (hipExtMallocWithFlags(&dst_ptr, size, hipDeviceMallocUncached) ==
        hipSuccess) {
      (void)hipMemset(dst_ptr, 0, size);
      fprintf(stderr, "[SYNCMEM] %zu bytes uncached VRAM @ %p\n", size,
              dst_ptr);
      return static_cast<DT *>(dst_ptr);
    }
    fprintf(stderr, "[SYNCMEM] hipExtMallocWithFlags failed\n");
  }
  if (use_host) {
    if (hipHostMalloc(&dst_ptr, size,
                      hipHostMallocMapped | hipHostMallocCoherent) ==
        hipSuccess) {
      memset(dst_ptr, 0, size);
      fprintf(stderr, "[SYNCMEM] %zu bytes host-coherent @ %p\n", size,
              dst_ptr);
      return static_cast<DT *>(dst_ptr);
    }
    fprintf(stderr, "[SYNCMEM] hipHostMalloc failed, falling back to VRAM\n");
  }
  (void)cudaMalloc(&dst_ptr, size);
  return static_cast<DT *>(dst_ptr);
}

static int g_mpk_unc_seq = 0;  // MPK_UNC_SEQ numbering, shared by every gpu_malloc<DT>
template <typename DT>
DT *gpu_malloc(size_t size) {
  void *dst_ptr;
#ifdef USE_NVSHMEM
  dst_ptr = nvshmem_malloc(size);
#else
  // MPK_UNCACHED_ALL routes every runtime allocation -- task descriptors,
  // the per-worker queues, the multi-layer pointer tables -- to NC as well.
  // A worker reads its queue entry before it can enter its first task, and
  // the multi-layer refresh WRITES task_desc->input_ptrs from the kernel, so
  // this state is not merely host-uploaded and read-only. Blunt on purpose:
  // it leaves only the torch-owned weights and KV cache on RW, which is the
  // bisect that says whether anything outside the weights still needs NC.
  static bool const unc_all = (getenv("MPK_UNCACHED_ALL") != nullptr &&
                               atoi(getenv("MPK_UNCACHED_ALL")) != 0);
  static int unc_lo = -1, unc_hi = -1;
  static bool const unc_log = getenv("MPK_UNC_LOG") != nullptr;
  if (unc_lo < 0 && getenv("MPK_UNC_SEQ") != nullptr) {
    if (sscanf(getenv("MPK_UNC_SEQ"), "%d:%d", &unc_lo, &unc_hi) != 2) {
      unc_lo = unc_hi = 0;
    }
  }
  int const my_seq = g_mpk_unc_seq++;
  bool const unc = unc_all || (unc_lo >= 0 && my_seq >= unc_lo && my_seq < unc_hi);
  if (unc_log) {
    fprintf(stderr, "[UNC] seq=%d bytes=%zu elem=%zu uncached=%d\n", my_seq, size, sizeof(DT),
            (int)unc);
  }
  if (unc) {
    dst_ptr = nullptr;
    if (hipExtMallocWithFlags(&dst_ptr, size, hipDeviceMallocUncached) ==
        hipSuccess) {
      (void)hipMemset(dst_ptr, 0, size);
      return static_cast<DT *>(dst_ptr);
    }
  }
  (void)cudaMalloc(&dst_ptr, size);
#endif
  return static_cast<DT *>(dst_ptr);
}

#ifdef MPK_EVCTR_HOST
// Pinned, device-mapped allocation for the event counters (debug only).
//
// See MPK_EVCTR_HOST at the call sites: device memory is unreadable by the host
// during a hang, because reading it needs HIP and HIP is wedged behind the
// spinning persistent kernel. Falls back to device memory if pinning fails, so
// a build with the flag still runs.
template <typename DT>
DT *host_mapped_malloc(size_t bytes, void **host_out) {
  void *h = nullptr;
  (void)hipHostMalloc(&h, bytes, hipHostMallocMapped);
  if (h == nullptr) {
    fprintf(stderr, "[MPK] EVCTR_HOST: hipHostMalloc(%zu) failed, using VRAM\n",
            bytes);
    *host_out = nullptr;
    return gpu_malloc<DT>(bytes);
  }
  memset(h, 0, bytes);
  void *d = nullptr;
  if (hipHostGetDevicePointer(&d, h, 0) != hipSuccess || d == nullptr) {
    d = h;
  }
  *host_out = h;
  return static_cast<DT *>(d);
}
#endif

void gpu_free(void *ptr) {
#ifdef USE_NVSHMEM
  nvshmem_free(ptr);
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
// Host-side view of the decode-progress counter (pinned, device-mapped).
// Not gated on MPK_PRECOMPUTED_DISPATCH: reset_decode_progress /
// get_decode_progress / launch_persistent_kernel all compile in either mode.
#ifdef MPK_TOK_HOST // request 0's token ids after the progress slots, one int per position
#define MPK_TOK_HOST_N MPK_MAX_SEQ_LENGTH
#else
#define MPK_TOK_HOST_N 0
#endif
static int *g_progress_host = nullptr;
#ifdef MPK_HOST_STOP
static int *g_stop_host = nullptr;
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

#if defined(MODE_OFFLINE) && defined(MPK_SPEC_VSLOT) && defined(MPK_MULTI_DEV)
// MDEV_SPEC: the shared spec-state buffer and the device it was allocated on (set at the first launch).
static void *g_mpk_spec_sh_host = nullptr;
static int g_mpk_spec_sh_dev = 0;
#endif

extern "C" void init_request_resources() {
  init_kernel<<<dim3(1, 1, 1), dim3(INIT_NUM_THREADS, 1, 1)>>>(
      global_runtime_config);
  (void)cudaStreamSynchronize(NULL);
#if defined(MODE_OFFLINE) && defined(MPK_SPEC_VSLOT) // MPK_SPEC_REQ_RESET
  // A new request starts with the virtual-slot spec state at zero, as the first launch does; otherwise the next
  // launch in this process finds request 0 done and ends at once (one launch per served request).
#ifdef MPK_MULTI_DEV
  if (g_mpk_spec_sh_host != nullptr) {
    int cur = 0;
    (void)hipGetDevice(&cur);
    (void)hipSetDevice(g_mpk_spec_sh_dev);
    (void)hipMemset(g_mpk_spec_sh_host, 0, sizeof(MpkSpecShared));
    (void)hipDeviceSynchronize();
    (void)hipSetDevice(cur);
  }
#else
  {
    int const zero4[4] = {0, 0, 0, 0};
    unsigned long long const zst[4] = {0ull, 0ull, 0ull, 0ull};
    (void)hipMemcpyToSymbol(HIP_SYMBOL(g_spec_state), zero4, sizeof zero4);
    (void)hipMemcpyToSymbol(HIP_SYMBOL(g_spec_stats), zst, sizeof zst);
    (void)hipDeviceSynchronize();
  }
#endif
#endif
}

// Zero the progress counters from the host, synchronously.
//
// launch_persistent_kernel already clears them, but that runs on the GPU
// worker thread *after* the streaming poller has started. The poller would
// then read the previous request's terminal position, conclude the whole
// response was already generated, and emit every token at once -- which
// reports a near-zero TTFT and a near-zero inter-token latency for every
// request after the first. Call this before the poller starts.
// Ask the running launch to end. The iteration-end decision reads the word every 16th iteration
// and then takes the normal end-of-request path. reset_decode_progress() clears it.
extern "C" void request_decode_stop() {
#ifdef MPK_HOST_STOP
  if (g_stop_host != nullptr) {
    __atomic_store_n(g_stop_host, 1, __ATOMIC_RELEASE);
  }
#endif
}

extern "C" void reset_decode_progress() {
#ifdef MPK_HOST_STOP
  if (g_stop_host != nullptr) {
    __atomic_store_n(g_stop_host, 0, __ATOMIC_RELEASE);
  }
#endif
  if (g_progress_host == nullptr) {
    return;
  }
  for (int i = 0; i < MPK_MAX_NUM_BATCHED_REQUESTS; i++) {
    __atomic_store_n(&g_progress_host[i], 0, __ATOMIC_RELEASE);
  }
#ifdef MPK_TOK_HOST
  for (int i = 0; i < MPK_TOK_HOST_N; i++) {
    __atomic_store_n(&g_progress_host[MPK_MAX_NUM_BATCHED_REQUESTS + i], -1, __ATOMIC_RELAXED);
  }
#endif
}

// Token position reached by batch slot `slot` in the currently-running launch,
// or -1 if unavailable. Safe to call from another thread while the kernel runs
// -- that is the point.
// Token id at position pos of request 0 in the running launch (MPK_TOK_HOST), or -1. Valid once
// get_decode_progress(0) >= pos: the publish orders the id before the position.
extern "C" int get_decode_token(int pos) {
#ifdef MPK_TOK_HOST
  if (g_progress_host == nullptr || pos < 0 || pos >= MPK_MAX_SEQ_LENGTH) {
    return -1;
  }
  return __atomic_load_n(&g_progress_host[MPK_MAX_NUM_BATCHED_REQUESTS + pos], __ATOMIC_ACQUIRE);
#else
  (void)pos;
  return -1;
#endif
}

extern "C" int get_decode_progress(int slot) {
  if (g_progress_host == nullptr || slot < 0 ||
      slot >= MPK_MAX_NUM_BATCHED_REQUESTS) {
    return -1;
  }
  return __atomic_load_n(&g_progress_host[slot], __ATOMIC_ACQUIRE);
}

// Per-launch stop bound. prepare_next_batch retires a request at
// `step + num_tokens + 1 >= config.max_seq_length` and has no other length
// control, so the generated length is whatever the arena allows. That is fine
// offline, where the arena is sized for the one run, but a server gets a
// different max_tokens per request and cannot recompile. config.max_seq_length
// is a plain int in the by-value config, distinct from the compile-time
// MPK_MAX_SEQ_LENGTH that fixes the token-array stride -- lowering it before a
// launch shortens generation without touching any layout.
//
// Must be <= MPK_MAX_SEQ_LENGTH; a larger value would index past the arena.
extern "C" void set_max_seq_length(int max_seq_length) {
  if (max_seq_length > 0 && max_seq_length <= MPK_MAX_SEQ_LENGTH) {
    global_runtime_config.max_seq_length = max_seq_length;
  }
}

extern "C" void set_rope_tables(void *cos_ptr, void *sin_ptr) {
  global_runtime_config.rope_cos_ptr = cos_ptr;
  global_runtime_config.rope_sin_ptr = sin_ptr;
}

#ifdef MPK_MULTI_DEV // MDEV_FWD_DECL
#if defined(MPK_ATTN_QWAIT_RT) || defined(MPK_RPROBE)
#error "MPK_MULTI_DEV: ATTN_QWAIT_RT / RPROBE set symbols on one device only"
#endif
static void mpk_mdev_init();
#endif
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

#ifdef MPK_LM_WEIGHT_AID
  // Task 218 input 3 is a static dim-0 bijection: descriptor x owns exactly
  // one eighth of the vocabulary weight. Give each consumer its own local RW
  // BO; do not replicate the 315 MiB weight.
  if (!mirage::aid::relocate_gang_sliced_input(
          all_tasks,
          (int)TASK_GANG_RMSNORM_LINEAR_MXFP4_BIAS_ARGMAX_MI300,
          3)) {
    fprintf(stderr, "[AID LM] fatal: LM-head split did not complete\n");
    abort();
  }
#endif

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

  // =========================================================================
  // Multi-layer fusion: build pointer table from all 36 layers, then compact
  // the task graph by removing redundant layers 1..35 (280 tasks, 35 events).
  // Must happen BEFORE template queue building so the template uses compacted
  // indices. Not gated on MPK_PRECOMPUTED_DISPATCH: the fused task graph is
  // the same either way; precomputed dispatch only fills per-worker queues.
  // =========================================================================
#ifdef MPK_FUSED_LAYER_BATCHING
  {
    constexpr int NUM_XCDS_ML = MPK_NUM_XCDS;
    // Must cover every slot the fused layer kernels read, not just the ones
    // the plain variant uses. The LM-head variant
    // (TASK_GANG_FULL_LAYER_WITH_LMHEAD_FUSED_MI300, FUSE_TAIL=1) reads
    // input_ptrs[24..27] and output_ptrs[12] -- see
    // gang_full_layer_with_lmhead_fused_mi300.cuh. These were 24 and 11,
    // sized for the plain variant, so with FUSE_TAIL=1 the ml loop would
    // refresh only the first 24/11 slots and leave the LM-head pointers
    // holding layer 0's values for every later layer. Sizing to the TaskDesc
    // capacity keeps this correct for any variant.
    constexpr int ML_N_IN = MAX_INPUTS_PER_TASK;   // 28
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
                 TASK_GANG_FULL_LAYER_WITH_LMHEAD_FUSED_MI300) &&
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

    if (ml_layers > 1) {
      // Build per-XCD pointer tables from all layers BEFORE compaction
#ifdef MPK_AID_LOCAL
      // Move the XCD-sliced weight tensors named by MPK_AID_LOCAL_SLOTS
      // into AID-local VRAM before the pointer tables are built.
      // all_tasks is still pre-compaction here, so fused_layer_positions[L]
      // + xcd is XCD xcd's descriptor for layer L; the table below is built
      // from all_tasks, so one rewrite covers layer 0 and layers 1+.
      mirage::aid::relocate_xcd_sliced_inputs(
          all_tasks, fused_layer_positions, ML_N_IN);
#endif
#ifdef MPK_PPROBE
      mpk_pprobe::run(all_tasks, fused_layer_positions);
#endif
#ifdef MPK_AID_RW
      {
        // MPK_AID_RW_MAP = "p0:p1:n,...": XCDs of the upper AID read the AID1
        // copy of every listed input. Any slot pointing inside [p0, p0+n) moves
        // to the same offset in p1, so sliced / offset pointers follow too.
        struct RwRange { char *p0, *p1; size_t n; };
        std::vector<RwRange> rr;
        if (char const *m = getenv("MPK_AID_RW_MAP")) {
          char const *q = m;
          while (*q) {
            char *e = nullptr;
            unsigned long long a = strtoull(q, &e, 16);
            if (*e != ':') break;
            unsigned long long b = strtoull(e + 1, &e, 16);
            if (*e != ':') break;
            unsigned long long n = strtoull(e + 1, &e, 16);
            rr.push_back({(char *)a, (char *)b, (size_t)n});
            q = (*e == ',') ? e + 1 : e;
            if (!*e) break;
          }
        }
        int moved = 0, moved_out = 0;
        std::vector<int> per_slot(ML_N_IN, 0);
        for (int L = 0; L < ml_layers; L++) {
          for (int xcd = NUM_XCDS_ML / 2; xcd < NUM_XCDS_ML; xcd++) {
            for (int slot = 0; slot < ML_N_IN; slot++) {
              void *&p = all_tasks[fused_layer_positions[L] + xcd].input_ptrs[slot];
              char *c = (char *)p;
              for (auto const &r : rr) {
                if (c >= r.p0 && c < r.p0 + r.n) {
                  p = r.p1 + (c - r.p0);
                  moved++;
                  per_slot[slot]++;
                  break;
                }
              }
            }
            for (int slot = 0; slot < ML_N_OUT; slot++) {
              void *&p = all_tasks[fused_layer_positions[L] + xcd].output_ptrs[slot];
              char *c = (char *)p;
              for (auto const &r : rr) {
                if (c >= r.p0 && c < r.p0 + r.n) {
                  p = r.p1 + (c - r.p0);
                  moved_out++;
                  break;
                }
              }
            }
          }
        }
        printf("[AID_RW] ranges=%zu repointed=%d out=%d slots:", rr.size(), moved, moved_out);
        for (int s = 0; s < ML_N_IN; s++)
          if (per_slot[s]) printf(" %d:%d", s, per_slot[s]);
        printf("\n");
      }
#endif
#ifdef MPK_MOE_RW
      {
        // XCDs of the upper AID read the AID1 copy of every MoE weight
        // (slots 17 gate_up, 18 down). Rewriting all_tasks before the tables
        // are built covers layer 0 and layers 1+.
        std::vector<std::pair<void *, void *>> rw;
        if (char const *m = getenv("MPK_MOE_RW_MAP")) {
          std::string s(m);
          size_t i = 0;
          while (i < s.size()) {
            size_t const c = s.find(':', i);
            size_t e = s.find(',', i);
            if (c == std::string::npos) break;
            if (e == std::string::npos) e = s.size();
            rw.push_back({(void *)strtoull(s.substr(i, c - i).c_str(), nullptr, 16),
                          (void *)strtoull(s.substr(c + 1, e - c - 1).c_str(), nullptr, 16)});
            i = e + 1;
          }
        }
        int moved = 0;
        for (int L = 0; L < ml_layers; L++) {
          for (int xcd = NUM_XCDS_ML / 2; xcd < NUM_XCDS_ML; xcd++) {
            for (int slot = 17; slot <= 18; slot++) {
              void *&p = all_tasks[fused_layer_positions[L] + xcd].input_ptrs[slot];
              for (auto const &pr : rw) {
                if (pr.first == p) {
                  p = pr.second;
                  moved++;
                  break;
                }
              }
            }
          }
        }
        printf("[MOE_RW] pairs=%zu repointed=%d (expect %d)\n", rw.size(), moved,
               ml_layers * (NUM_XCDS_ML / 2) * 2);
      }
#endif
      if (char const *os = getenv("MPK_ORACLE_MALL_SLOTS")) {
        // MALL-residency probe (wrong output): layers 1+ read layer 0's
        // pointer in the listed slots, per XCD.
        int moved = 0;
        for (char const *q = os; *q;) {
          char *e = nullptr;
          long const slot = strtol(q, &e, 10);
          if (e == q) break;
          if (slot >= 0 && slot < ML_N_IN)
            for (int L = 1; L < ml_layers; L++)
              for (int xcd = 0; xcd < NUM_XCDS_ML; xcd++) {
                all_tasks[fused_layer_positions[L] + xcd].input_ptrs[slot] =
                    all_tasks[fused_layer_positions[0] + xcd].input_ptrs[slot];
                moved++;
              }
          q = (*e == ',') ? e + 1 : e;
        }
        printf("[ORACLE_MALL] slots=%s repointed=%d\n", os, moved);
      }
      if (getenv("MPK_DBG_XCD_PTRS")) {
        // Which descriptors actually carry the LM head weight (slot 26)?
        int shown = 0;
        for (size_t t = 0; t < all_tasks.size() && shown < 24; t++) {
          if (all_tasks[t].input_ptrs[26] != nullptr) {
            printf("[LMSCAN] idx=%zu task_type=%d lm_W=%p lm_bias=%p "
                   "norm_w=%p norm_in=%p\n",
                   t,
                   (int)all_tasks[t].task_type,
                   all_tasks[t].input_ptrs[26],
                   all_tasks[t].input_ptrs[27],
                   all_tasks[t].input_ptrs[24],
                   all_tasks[t].input_ptrs[25]);
            shown++;
          }
        }
        printf("[LMSCAN] total_tasks=%zu with_slot26=%d\n",
               all_tasks.size(), shown);
      }
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
          // LM head pointers live on the TAIL layer, not layer 0, and in
          // slots 24-27 which no existing dump covers. Eight distinct,
          // uniformly-strided values here == a real per-XCD bijection;
          // eight identical values == every XCD reading the same weight.
          if (L == ml_layers - 1 && getenv("MPK_DBG_XCD_PTRS")) {
            printf("[LMPTR] tail L=%d xcd=%d norm_w=%p norm_in=%p "
                   "lm_W=%p lm_bias=%p\n",
                   L,
                   xcd,
                   td_xcd.input_ptrs[24],
                   td_xcd.input_ptrs[25],
                   td_xcd.input_ptrs[26],
                   td_xcd.input_ptrs[27]);
          }
          if (L == 0 && getenv("MPK_DBG_XCD_PTRS")) {
            printf("[XCDPTR] L0 xcd=%d qkv_w=%p oproj_w=%p router_w=%p "
                   "resid=%p w13=%p w2=%p\n",
                   xcd,
                   td_xcd.input_ptrs[4],
                   td_xcd.input_ptrs[9],
                   td_xcd.input_ptrs[13],
                   td_xcd.input_ptrs[1],
                   td_xcd.input_ptrs[17],
                   td_xcd.input_ptrs[18]);
            fflush(stdout);
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
          bool is_lmhead = all_tasks[fused_layer_positions[L]].task_type ==
                           TASK_GANG_FULL_LAYER_WITH_LMHEAD_FUSED_MI300;
          int used_in = is_lmhead ? 28 : 24;
          int used_out = is_lmhead ? 13 : 11;
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

      // Barrier counters (zeroed). This is the cross-XCD layer rendezvous, so
      // it goes through mpk_sync_malloc: in DPX+NPS2 plain VRAM is MTYPE_RW,
      // coherent only within an AID, and a device's 4 XCDs span two -- an
      // arrival published by one pair is never observed by the other, which
      // parks the run at the first layer boundary with zero forward passes.
      global_runtime_config.ml_barrier_arrive =
          mpk_sync_malloc<int>(MPK_NUM_XCDS * 16 * sizeof(int));
      (void)cudaMemset(
          global_runtime_config.ml_barrier_arrive, 0, MPK_NUM_XCDS * 16 * sizeof(int));
      global_runtime_config.ml_barrier_global =
          mpk_sync_malloc<int>(16 * sizeof(int));
      (void)cudaMemset(
          global_runtime_config.ml_barrier_global, 0, 16 * sizeof(int));
      global_runtime_config.ml_barrier_release =
          mpk_sync_malloc<int>(MPK_NUM_XCDS * 16 * sizeof(int));
      (void)cudaMemset(
          global_runtime_config.ml_barrier_release, 0, MPK_NUM_XCDS * 16 * sizeof(int));

      global_runtime_config.ml_num_layers = ml_layers;
      global_runtime_config.ml_workers_per_xcd = num_workers / NUM_XCDS_ML;

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
#endif // MPK_FUSED_LAYER_BATCHING

#ifdef MPK_PRECOMPUTED_DISPATCH
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
    constexpr int NUM_XCDS_PC = MPK_NUM_XCDS;
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
          mpk_sync_malloc<unsigned long long>(sizeof(unsigned long long));
      (void)cudaMemset(global_runtime_config.precomp_iter_ready,
                       0,
                       sizeof(unsigned long long));

      global_runtime_config.precomp_terminate =
          mpk_sync_malloc<int>(sizeof(int));
      (void)cudaMemset(global_runtime_config.precomp_terminate, 0, sizeof(int));

      // Per-XCD release flags for fast cross-XCD iteration barrier
      // [8*16 ints] padded to cache lines to avoid false sharing
      global_runtime_config.precomp_iter_xcd_release =
          gpu_malloc<int>(MPK_NUM_XCDS * 16 * sizeof(int));
      (void)cudaMemset(global_runtime_config.precomp_iter_xcd_release,
                       0,
                       MPK_NUM_XCDS * 16 * sizeof(int));

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
          size_t ws_bytes = (size_t)num_workers * 12 * sizeof(int);
          int *ws_host = nullptr;
          if (hipHostMalloc(reinterpret_cast<void **>(&ws_host),
                            ws_bytes,
                            hipHostMallocMapped | hipHostMallocNonCoherent) ==
                  hipSuccess &&
              ws_host != nullptr) {
            memset(ws_host, 0, ws_bytes);
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
#ifdef MPK_EVCTR_HOST
    {
      void *h = nullptr;
      global_runtime_config.xcd_local_event_counters =
          host_mapped_malloc<EventCounter>(
              NUM_XCDS_PC * n_events * sizeof(EventCounter), &h);
      g_dbg_h_xcd_local_counters = static_cast<EventCounter *>(h);
      // Thresholds never change after upload, so a plain host copy is enough.
      g_dbg_h_xcd_event_thresholds = new int[NUM_XCDS_PC * n_events];
      memcpy(g_dbg_h_xcd_event_thresholds,
             h_xcd_evt.data(),
             NUM_XCDS_PC * n_events * sizeof(int));
      printf("[MPK] EVCTR_HOST: xcd_local host=%p dev=%p thresholds mirrored\n",
             h,
             (void *)global_runtime_config.xcd_local_event_counters);
      fflush(stdout);
    }
#else
    global_runtime_config.xcd_local_event_counters =
        mpk_sync_malloc<EventCounter>(NUM_XCDS_PC * n_events * sizeof(EventCounter));
    (void)cudaMemset(global_runtime_config.xcd_local_event_counters,
                     0,
                     NUM_XCDS_PC * n_events * sizeof(EventCounter));
#endif
    g_dbg_num_xcds = NUM_XCDS_PC;

    // Set ml_workers_per_xcd (pointer tables uploaded earlier, before
    // compaction)
#ifdef MPK_FUSED_LAYER_BATCHING
    if (global_runtime_config.ml_num_layers > 0) {
      global_runtime_config.ml_workers_per_xcd = workers_per_xcd;
    }
#endif
  }
#endif

  // Decode-progress counters in pinned host memory, one per batch slot.
  // Always allocated (one page, one store per slot per iteration); the
  // serving path reads them to stream tokens while the launch is still
  // running, and nothing else touches them. Not gated on
  // MPK_PRECOMPUTED_DISPATCH: the scheduler publishes from the shared
  // EVENT_END_OF_TASK_GRAPH path.
  //
  // Per-slot, not a single counter: a batch coalesces requests with
  // different prompt lengths, so slot r's first *output* token lands at a
  // different absolute position than slot 0's. One shared counter would
  // make every request in the batch inherit request 0's schedule and
  // report the wrong TTFT for all the others.
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
  {
    int *pg_host = nullptr;
    if (hipHostMalloc(reinterpret_cast<void **>(&pg_host),
                      sizeof(int) * (MPK_MAX_NUM_BATCHED_REQUESTS + MPK_TOK_HOST_N),
                      hipHostMallocMapped | hipHostMallocNonCoherent) ==
            hipSuccess &&
        pg_host != nullptr) {
      for (int i = 0; i < MPK_MAX_NUM_BATCHED_REQUESTS + MPK_TOK_HOST_N; i++) {
        pg_host[i] = 0;
      }
      g_progress_host = pg_host;
      int *pg_dev = nullptr;
      if (hipHostGetDevicePointer(reinterpret_cast<void **>(&pg_dev),
                                 pg_host,
                                 0) == hipSuccess) {
        global_runtime_config.progress_host = pg_dev;
      } else {
        (void)hipHostFree(pg_host);
        g_progress_host = nullptr;
        global_runtime_config.progress_host = nullptr;
      }
    }
  }
#endif
#if defined(MPK_HOST_STOP) && (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  {
    // Fine-grained (coherent) so the kernel's system-scope load sees the host's store, portable
    // so every device of an MPK_MULTI_DEV launch can read it.
    int *st_host = nullptr;
    int *st_dev = nullptr;
    if (hipHostMalloc(reinterpret_cast<void **>(&st_host), 64,
                      hipHostMallocMapped | hipHostMallocCoherent |
                          hipHostMallocPortable) == hipSuccess &&
        st_host != nullptr) {
      __atomic_store_n(st_host, 0, __ATOMIC_RELEASE);
      if (hipHostGetDevicePointer(reinterpret_cast<void **>(&st_dev), st_host, 0) ==
          hipSuccess) {
        g_stop_host = st_host;
      } else {
        (void)hipHostFree(st_host);
        st_dev = nullptr;
      }
    }
    global_runtime_config.stop_host = st_dev;
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
#ifdef MPK_EVCTR_HOST
  {
    void *h = nullptr;
    global_runtime_config.all_event_counters = host_mapped_malloc<EventCounter>(
        all_events.size() * sizeof(EventCounter), &h);
    g_dbg_h_event_counters = static_cast<EventCounter *>(h);
    printf("[MPK] EVCTR_HOST: all_event_counters host=%p dev=%p n=%zu\n",
           h,
           (void *)global_runtime_config.all_event_counters,
           all_events.size());
    fflush(stdout);
  }
#else
  global_runtime_config.all_event_counters =
      gpu_malloc<EventCounter>(all_events.size() * sizeof(EventCounter));
#endif
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
  constexpr int NUM_XCDS_INIT = MPK_NUM_XCDS;
  global_runtime_config.num_xcds = NUM_XCDS_INIT;
  // xcd_local_event_counters may already be allocated by precomputed dispatch
  if (global_runtime_config.xcd_local_event_counters == nullptr) {
    global_runtime_config.xcd_local_event_counters = mpk_sync_malloc<EventCounter>(
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
  (void)cudaFuncSetAttribute(scheduler_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MPK_SCHED_LDS > 1024 ? MPK_SCHED_LDS : 1024);
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
#ifdef MPK_MULTI_DEV // MDEV_INIT_CALL
  mpk_mdev_init();
#endif

  init_request_resources();
#ifdef USE_NVSHMEM
  // Add a global barrier for all init_kernel to complete
  nvshmem_barrier_all();
#endif
}

// Entry point for C/C++
// TODO: change launch config

#ifdef MPK_MULTI_DEV // MDEV_HOST
// Devices of the group: the current device at init and the next MPK_MULTI_DEV - 1 visible ones
// (HIP_VISIBLE_DEVICES lists exactly the partitions of one GPU, partition 0 first).
struct MpkMdev {
  int dev[MPK_MULTI_DEV];
  hipStream_t ws[MPK_MULTI_DEV], ss[MPK_MULTI_DEV];
  hipEvent_t wdone[MPK_MULTI_DEV], sdone[MPK_MULTI_DEV];
};
static MpkMdev g_mdev;

static void mpk_mdev_check(hipError_t e, char const *what, int d) {
  if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
    fprintf(stderr, "[MDEV] %s on device %d: %s\n", what, d, hipGetErrorString(e));
    abort();
  }
  (void)hipGetLastError();
}

// Write one host value into a __constant__ symbol on every device of the group.
static void mpk_mdev_symbol_all(void const *sym, void const *src, size_t bytes) {
  int cur = 0;
  (void)hipGetDevice(&cur);
  for (int d = 0; d < MPK_MULTI_DEV; d++) {
    mpk_mdev_check(hipSetDevice(g_mdev.dev[d]), "hipSetDevice", d);
    mpk_mdev_check(hipMemcpyToSymbol(sym, src, bytes), "hipMemcpyToSymbol", d);
  }
  (void)hipSetDevice(cur);
}

static void mpk_mdev_init() {
  constexpr int XPD = MPK_NUM_XCDS / MPK_MULTI_DEV;
  static_assert(MPK_NUM_XCDS % MPK_MULTI_DEV == 0, "MPK_MULTI_DEV must divide MPK_NUM_XCDS");
  int cur = 0, ndev = 0;
  (void)hipGetDevice(&cur);
  (void)hipGetDeviceCount(&ndev);
  if (cur + MPK_MULTI_DEV > ndev) {
    fprintf(stderr, "[MDEV] need %d devices from %d, only %d visible\n", MPK_MULTI_DEV, cur, ndev);
    abort();
  }
  int const nw = global_runtime_config.num_workers;
  int const ns = global_runtime_config.num_local_schedulers;
  if (nw % MPK_MULTI_DEV != 0 || ns != MPK_NUM_XCDS || global_runtime_config.num_remote_schedulers) {
    fprintf(stderr, "[MDEV] workers %d / schedulers %d+%d do not split over %d devices\n", nw, ns,
            global_runtime_config.num_remote_schedulers, MPK_MULTI_DEV);
    abort();
  }
  for (int d = 0; d < MPK_MULTI_DEV; d++) {
    g_mdev.dev[d] = cur + d;
  }
  for (int d = 0; d < MPK_MULTI_DEV; d++) {
    int const hd = g_mdev.dev[d];
    mpk_mdev_check(hipSetDevice(hd), "hipSetDevice", d);
    for (int e = 0; e < MPK_MULTI_DEV; e++) {
      if (e != d) {
        mpk_mdev_check(hipDeviceEnablePeerAccess(g_mdev.dev[e], 0), "hipDeviceEnablePeerAccess", d);
      }
    }
    (void)cudaFuncSetAttribute(worker_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               MAX_DYNAMIC_SHARED_MEMORY_SIZE);
    (void)cudaFuncSetAttribute(scheduler_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               MPK_SCHED_LDS > 1024 ? MPK_SCHED_LDS : 1024);
    int const xb = d * XPD;
    int const wb = d * (nw / MPK_MULTI_DEV);
    mpk_mdev_check(hipMemcpyToSymbol(HIP_SYMBOL(g_mpk_dev_xcd_base), &xb, sizeof xb), "xcd base", d);
    mpk_mdev_check(hipMemcpyToSymbol(HIP_SYMBOL(g_mpk_dev_worker_base), &wb, sizeof wb), "worker base", d);
    if (d == 0) {
      g_mdev.ws[0] = global_runtime_config.worker_stream;
      g_mdev.ss[0] = global_runtime_config.scheduler_stream;
      g_mdev.wdone[0] = global_runtime_config.worker_done_event;
      g_mdev.sdone[0] = global_runtime_config.scheduler_done_event;
    } else {
      mpk_mdev_check(cudaStreamCreateWithFlags(&g_mdev.ws[d], cudaStreamNonBlocking), "stream", d);
      mpk_mdev_check(cudaStreamCreateWithFlags(&g_mdev.ss[d], cudaStreamNonBlocking), "stream", d);
      mpk_mdev_check(cudaEventCreateWithFlags(&g_mdev.wdone[d], cudaEventDisableTiming), "event", d);
      mpk_mdev_check(cudaEventCreateWithFlags(&g_mdev.sdone[d], cudaEventDisableTiming), "event", d);
    }
    char bus[64] = {0};
    (void)hipDeviceGetPCIBusId(bus, sizeof bus, hd);
    printf("[MDEV] device %d = hip %d (%s): logical XCDs %d-%d, workers %d-%d, %d scheduler(s)\n",
           d, hd, bus, xb, xb + XPD - 1, wb, wb + nw / MPK_MULTI_DEV - 1, XPD);
  }
#ifdef MPK_PHASE_SNAP // MDEV_PSNAP
  {
    void *snap0 = nullptr;
    size_t const nb = sizeof g_phase_snap;
    mpk_mdev_check(hipSetDevice(g_mdev.dev[0]), "hipSetDevice", 0);
    mpk_mdev_check(hipMalloc(&snap0, nb), "phase snapshot", 0);
    mpk_mdev_check(hipMemset(snap0, 0, nb), "phase snapshot clear", 0);
    mpk_mdev_symbol_all(&g_phase_snap_dst, &snap0, sizeof snap0);
    printf("[MDEV] phase snapshot: every device stamps one buffer on device 0 at %p\n", snap0);
  }
#endif
#if defined(MPK_TK_OPS) && defined(MPK_PHASE_SNAP)
  {
    void *tko0 = nullptr;
    size_t const nb = sizeof g_tk_ops;
    mpk_mdev_check(hipSetDevice(g_mdev.dev[0]), "hipSetDevice", 0);
    mpk_mdev_check(hipMalloc(&tko0, nb), "tk_ops snapshot", 0);
    mpk_mdev_check(hipMemset(tko0, 0, nb), "tk_ops snapshot clear", 0);
    mpk_mdev_symbol_all(&g_tk_ops_dst, &tko0, sizeof tko0);
  }
#endif
#if defined(MPK_ATS_OPS) && defined(MPK_PHASE_SNAP)
  {
    void *ao0 = nullptr;
    size_t const nb = sizeof g_ats_ops;
    mpk_mdev_check(hipSetDevice(g_mdev.dev[0]), "hipSetDevice", 0);
    mpk_mdev_check(hipMalloc(&ao0, nb), "ats_ops snapshot", 0);
    mpk_mdev_check(hipMemset(ao0, 0, nb), "ats_ops snapshot clear", 0);
    mpk_mdev_symbol_all(&g_ats_ops_dst, &ao0, sizeof ao0);
  }
#endif
#ifdef MPK_VTN_MOETL
  {
    void *tl0 = nullptr;
    size_t const nb = sizeof g_vtn_tl;
    mpk_mdev_check(hipSetDevice(g_mdev.dev[0]), "hipSetDevice", 0);
    mpk_mdev_check(hipMalloc(&tl0, nb), "moe timeline", 0);
    mpk_mdev_check(hipMemset(tl0, 0, nb), "moe timeline clear", 0);
    mpk_mdev_symbol_all(&g_vtn_tl_dst, &tl0, sizeof tl0);
    printf("[MDEV] MoE timeline: every device stamps one buffer on device 0 at %p\n", tl0);
  }
#endif
  (void)hipSetDevice(cur);
  fflush(stdout);
}

#ifdef MPK_AID_SPLIT_FLAGS
// Flag replica pointers for devices 1.. (device 0's go through the single-device path): each
// device gets its own VA of the two shared replica BOs.
static void mpk_mdev_rep_symbols(void *rep[2][MPK_MULTI_DEV]) {
  int cur = 0;
  (void)hipGetDevice(&cur);
  for (int d = 1; d < MPK_MULTI_DEV; d++) {
    int *rp[2] = {static_cast<int *>(rep[0][d]), static_cast<int *>(rep[1][d])};
    mpk_mdev_check(hipSetDevice(g_mdev.dev[d]), "hipSetDevice", d);
    mpk_mdev_check(hipMemcpyToSymbol(HIP_SYMBOL(g_aid_flag_rep), rp, sizeof rp), "g_aid_flag_rep", d);
#ifdef MPK_ROUTER_OPROJ_PARTIAL
    mpk_mdev_check(hipMemcpyToSymbol(HIP_SYMBOL(g_rop_rep), rp, sizeof rp), "g_rop_rep", d);
#endif
  }
  (void)hipSetDevice(cur);
}
#endif

// Worker and scheduler kernels on every device of the group, each after prepare_kernel; the
// caller's stream (device 0) then waits for all of them.
static void mpk_mdev_launch(cudaStream_t default_stream) {
  int cur = 0;
  (void)hipGetDevice(&cur);
  int const wpd = global_runtime_config.num_workers / MPK_MULTI_DEV;
  int const spd = global_runtime_config.num_local_schedulers / MPK_MULTI_DEV;
#if defined(MPK_PLACE_GUARD) && defined(MPK_PLACE_SERIAL)
  // MPK_PLACE_SERIAL_V1: workers of every device first; schedulers once every worker block is resident.
  for (int d = 0; d < MPK_MULTI_DEV; d++) {
    (void)hipSetDevice(g_mdev.dev[d]);
    (void)cudaStreamWaitEvent(g_mdev.ws[d], global_runtime_config.prepare_done_event, 0);
    (void)cudaStreamWaitEvent(g_mdev.ss[d], global_runtime_config.prepare_done_event, 0);
    worker_kernel<<<dim3(wpd, 1, 1), dim3(WORKER_NUM_THREADS, 1, 1),
                    MAX_DYNAMIC_SHARED_MEMORY_SIZE, g_mdev.ws[d]>>>(global_runtime_config);
    (void)cudaEventRecord(g_mdev.wdone[d], g_mdev.ws[d]);
  }
  if (g_mpk_place_h != nullptr) {
    unsigned const want = (unsigned)(global_runtime_config.num_workers * MPK_XCD_OVERSUB);
    auto const t0 = std::chrono::steady_clock::now();
    while (__atomic_load_n(&g_mpk_place_h[0], __ATOMIC_ACQUIRE) < want &&
           std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(2000)) {
      std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
  }
  for (int d = 0; d < MPK_MULTI_DEV; d++) {
    (void)hipSetDevice(g_mdev.dev[d]);
    scheduler_kernel<<<dim3(spd, 1, 1), dim3(128, 1, 1), MPK_SCHED_LDS, g_mdev.ss[d]>>>(
        global_runtime_config);
    (void)cudaEventRecord(g_mdev.sdone[d], g_mdev.ss[d]);
  }
#else
  for (int d = 0; d < MPK_MULTI_DEV; d++) {
    (void)hipSetDevice(g_mdev.dev[d]);
    (void)cudaStreamWaitEvent(g_mdev.ws[d], global_runtime_config.prepare_done_event, 0);
    (void)cudaStreamWaitEvent(g_mdev.ss[d], global_runtime_config.prepare_done_event, 0);
    worker_kernel<<<dim3(wpd, 1, 1), dim3(WORKER_NUM_THREADS, 1, 1),
                    MAX_DYNAMIC_SHARED_MEMORY_SIZE, g_mdev.ws[d]>>>(global_runtime_config);
    scheduler_kernel<<<dim3(spd, 1, 1), dim3(128, 1, 1), MPK_SCHED_LDS, g_mdev.ss[d]>>>(
        global_runtime_config);
    (void)cudaEventRecord(g_mdev.wdone[d], g_mdev.ws[d]);
    (void)cudaEventRecord(g_mdev.sdone[d], g_mdev.ss[d]);
  }
#endif
  (void)hipSetDevice(cur);
  for (int d = 0; d < MPK_MULTI_DEV; d++) {
    (void)cudaStreamWaitEvent(default_stream, g_mdev.wdone[d], 0);
    (void)cudaStreamWaitEvent(default_stream, g_mdev.sdone[d], 0);
  }
}
#endif
extern "C" void launch_persistent_kernel(cudaStream_t default_stream) {
  fprintf(stderr, "[HOST_DBG] launch_persistent_kernel ENTER\n");
  static bool const s_lts = getenv("MPK_LAUNCH_TS") != nullptr;
  if (s_lts) {
    fprintf(stderr, "[LAUNCH_TS] enter %.6f\n",
            std::chrono::duration<double>(
                std::chrono::system_clock::now().time_since_epoch())
                .count());
  }
#ifdef MPK_AID_SPLIT_FLAGS
  // Every replicated flag family also lives in oproj_topk_counters, which
  // demo.py clears wholesale in reset_device_barriers() between launches. The
  // replicas must mirror that exactly: clear them here, per launch, before
  // the kernel starts -- never between layers of one launch.
  //
  // Both directions of divergence hang. A layer-derived target like
  // attn_release would find a stale high flag already satisfying this
  // launch's early gates; a snapshot target like layer_release would snapshot
  // the previous launch's terminal value and then wait for a bump that the
  // freshly-zeroed epoch counter never reaches.
  {
    static void *s_rep[2] = {nullptr, nullptr};
    static bool s_tried = false;
    if (!s_tried) {
      s_tried = true;
#ifdef MPK_MULTI_DEV // MDEV_FLAG_REPS: device 0 below as before, devices 1.. here
      void *s_rep_dev[2][MPK_MULTI_DEV] = {};
      bool const s_rep_ok =
          mirage::aid::alloc_flag_replicas_mdev(s_rep_dev, MPK_MULTI_DEV, g_mdev.dev);
      if (s_rep_ok) {
        s_rep[0] = s_rep_dev[0][0];
        s_rep[1] = s_rep_dev[1][0];
        mpk_mdev_rep_symbols(s_rep_dev);
      }
      if (s_rep_ok) {
#else
      if (mirage::aid::alloc_flag_replicas(s_rep)) {
#endif
        int *dev[2] = {static_cast<int *>(s_rep[0]),
                       static_cast<int *>(s_rep[1])};
        // Async-on-default_stream + synchronize, for the same reason as the
        // clear below: the plain form lands on the NULL stream, which does not
        // synchronize with the hipStreamNonBlocking streams this kernel runs
        // on. If that write raced the launch, some workgroups would read
        // g_aid_flag_rep as null and fall back to the shared buffer while
        // others read the replica -- publisher and consumer split across two
        // different buffers, which deadlocks instead of merely being slow.
#ifdef MPK_MULTI_DEV // MDEV_SYM_BCAST: runtime switches, spec settings and the EAGLE3 table on every device
#define hipMemcpyToSymbol(sym, src, bytes) (mpk_mdev_symbol_all(&(sym), (src), (bytes)), hipSuccess)
#endif
#ifdef MPK_ATTN_QWAIT_RT
        {
          char const *qw = getenv("MPK_QWAIT_RT");
          int const qwv = qw ? atoi(qw) : 1;
          (void)hipMemcpyToSymbol(HIP_SYMBOL(g_mpk_qwait_rt), &qwv, sizeof qwv);
          fprintf(stderr, "[QWAIT] runtime switch %d\n", qwv);
        }
#endif
#ifdef MPK_LTK_TIE_RT
        {
          char const *tr = getenv("MPK_TIE_RT");
          int const trv = tr ? atoi(tr) : 1;
          (void)hipMemcpyToSymbol(HIP_SYMBOL(g_mpk_tie_rt), &trv, sizeof trv);
          fprintf(stderr, "[TIE_RT] runtime switch %d\n", trv);
        }
#endif
#ifdef MPK_ARGMAX_FUSED
#ifndef MPK_RT_CONST
        {
          char const *ar = getenv("MPK_AMX_RT");
          int const arv = ar ? atoi(ar) : 1;
          (void)hipMemcpyToSymbol(HIP_SYMBOL(g_mpk_amx_rt), &arv, sizeof arv);
          fprintf(stderr, "[AMX_RT] runtime switch %d\n", arv);
        }
#endif
#endif
#ifdef MPK_ROP_ROW_EARLY
#ifndef MPK_RT_CONST
        {
          char const *rr = getenv("MPK_RRE_RT");
          int const rrv = rr ? atoi(rr) : 1;
          (void)hipMemcpyToSymbol(HIP_SYMBOL(g_mpk_rre_rt), &rrv, sizeof rrv);
          fprintf(stderr, "[RRE_RT] runtime switch %d\n", rrv);
        }
#endif
#endif
#ifdef MPK_LTK_STATS_W0
#ifndef MPK_RT_CONST
        {
          char const *sw = getenv("MPK_STW0_RT");
          int const swv = sw ? atoi(sw) : 1;
          (void)hipMemcpyToSymbol(HIP_SYMBOL(g_mpk_stw0_rt), &swv, sizeof swv);
          fprintf(stderr, "[STW0_RT] runtime switch %d\n", swv);
        }
#endif
#endif
#ifdef MPK_LTK_STW0_FAST
#ifndef MPK_RT_CONST
        {
          char const *sf = getenv("MPK_STWF_RT");
          int const sfv = sf ? atoi(sf) : 1;
          (void)hipMemcpyToSymbol(HIP_SYMBOL(g_mpk_stwf_rt), &sfv, sizeof sfv);
          fprintf(stderr, "[STWF_RT] runtime switch %d\n", sfv);
        }
#endif
#endif
#ifdef MPK_ROP_QUANT_FAST
#ifndef MPK_RT_CONST
        {
          char const *qf = getenv("MPK_RQF_RT");
          int const qfv = qf ? atoi(qf) : 1;
          (void)hipMemcpyToSymbol(HIP_SYMBOL(g_mpk_rqf_rt), &qfv, sizeof qfv);
          fprintf(stderr, "[RQF_RT] runtime switch %d\n", qfv);
        }
#endif
#endif
#ifdef MPK_KSEL_MERGE
#ifndef MPK_RT_CONST
        {
          char const *km = getenv("MPK_KMRG_RT");
          int const kmv = km ? atoi(km) : 1;
          (void)hipMemcpyToSymbol(HIP_SYMBOL(g_mpk_kmrg_rt), &kmv, sizeof kmv);
          fprintf(stderr, "[KMRG_RT] runtime switch %d\n", kmv);
        }
#endif
#endif
#ifdef MPK_VT2_MPOLL_RT
        {
          char const *mv = getenv("MPK_MPOLL");
          int const mp = mv ? atoi(mv) : 0;
          (void)hipMemcpyToSymbol(HIP_SYMBOL(kernel::g_mpk_mpoll_rt), &mp, sizeof mp);
          fprintf(stderr, "[VT2_MPOLL] runtime switch %d\n", mp);
        }
#endif
#ifdef MPK_VT2_QKV_EARLY
        {
          char const *ev = getenv("MPK_QKVE");
          int const qe = ev ? atoi(ev) : 0;
          (void)hipMemcpyToSymbol(HIP_SYMBOL(kernel::g_mpk_qkve_rt), &qe, sizeof qe);
          fprintf(stderr, "[VT2_QKVE] runtime switch %d\n", qe);
        }
#endif
#ifdef MPK_VT2_T1ROW_LATE
        {
          char const *tv = getenv("MPK_T1LATE");
          int const tl = tv ? atoi(tv) : 0;
          (void)hipMemcpyToSymbol(HIP_SYMBOL(kernel::g_mpk_t1late_rt), &tl, sizeof tl);
          fprintf(stderr, "[VT2_T1LATE] runtime switch %d\n", tl);
        }
#endif
#ifdef MPK_ML_STAGE
        {
          char const *mv = getenv("MPK_MLSTAGE");
          int const ms = mv ? atoi(mv) : 0;
          (void)hipMemcpyToSymbol(HIP_SYMBOL(kernel::g_mpk_mlstage_rt), &ms, sizeof ms);
          fprintf(stderr, "[ML_STAGE] runtime switch %d\n", ms);
        }
#endif
#ifdef MPK_VTN_N7B2
        {
          char const *nv = getenv("MPK_N7B2");
          int const ns = nv ? atoi(nv) : 1;
          (void)hipMemcpyToSymbol(HIP_SYMBOL(kernel::g_mpk_n7b2_rt), &ns, sizeof ns);
          fprintf(stderr, "[VTN_N7B2] runtime switch %d\n", ns);
        }
#endif
#if defined(MPK_VTN_N7S2) && defined(MPK_VTN_N7S2_RT)
        {
          char const *sv2 = getenv("MPK_N7S2");
          int const s2 = sv2 ? atoi(sv2) : 1;
          (void)hipMemcpyToSymbol(HIP_SYMBOL(kernel::g_mpk_n7s2_rt), &s2, sizeof s2);
          fprintf(stderr, "[VTN_N7S2] runtime switch %d\n", s2);
        }
#endif
#ifdef MPK_VTN_BAL
        {
          char const *bv = getenv("MPK_VTNBAL");
          int const bs = bv ? atoi(bv) : 1;
          (void)hipMemcpyToSymbol(HIP_SYMBOL(kernel::g_mpk_vtnbal_rt), &bs, sizeof bs);
          fprintf(stderr, "[VTN_BAL] runtime switch %d\n", bs);
        }
#endif
#ifdef MPK_VT2_QSPLIT_RT
        {
          char const *qv = getenv("MPK_QSPLIT");
          int const qs = qv ? atoi(qv) : 1;
          (void)hipMemcpyToSymbol(HIP_SYMBOL(kernel::g_mpk_qsplit_rt), &qs, sizeof qs);
          fprintf(stderr, "[VT2_QSPLIT] runtime switch %d\n", qs);
        }
#endif
#ifdef MPK_VT2_W2DELAY
        {
          char const *wv = getenv("MPK_VT2_W2D");
          int const w2d = wv ? atoi(wv) : 0;
          (void)hipMemcpyToSymbol(HIP_SYMBOL(kernel::g_mpk_vt2_w2d), &w2d, sizeof w2d);
          fprintf(stderr, "[VT2_W2D] round-1 W2 delay %d ns\n", w2d);
        }
#endif
#ifdef MPK_ATTN_META_LDS_RT
        {
          char const *mv = getenv("MPK_METALDS");
          int const ms = mv ? atoi(mv) : 1;
          (void)hipMemcpyToSymbol(HIP_SYMBOL(kernel::g_mpk_metalds_rt), &ms, sizeof ms);
          fprintf(stderr, "[ATTN_METALDS] runtime switch %d\n", ms);
        }
#endif
#ifdef MPK_ATTN_FIXED_MERGER_RT
        {
          char const *fv = getenv("MPK_FM");
          int const fs = fv ? atoi(fv) : 1;
          (void)hipMemcpyToSymbol(HIP_SYMBOL(kernel::g_mpk_fm_rt), &fs, sizeof fs);
          fprintf(stderr, "[ATTN_FM] runtime switch %d\n", fs);
        }
#endif
#ifdef MPK_D89_RT
        {
          char const *dv = getenv("MPK_D89_NS");
          unsigned const dt = dv ? (unsigned)(atoi(dv) / 10) : 400u;
          (void)hipMemcpyToSymbol(HIP_SYMBOL(g_mpk_d89_ticks), &dt, sizeof dt);
          fprintf(stderr, "[D89_RT] ranks 8-9 W2 entry delay %u ticks\n", dt);
        }
#endif
#if defined(MODE_OFFLINE) && defined(MPK_SPEC_VSLOT)
        {
          static long long *spec_oracle_dev = nullptr;
          static int spec_oracle_n = 0;
          char const *op = getenv("MPK_SPEC_ORACLE");
          if (op != nullptr && op[0] != 0 && spec_oracle_dev == nullptr) {
            std::vector<long long> ids;
            FILE *f = fopen(op, "r");
            if (f != nullptr) {
              long long v;
              while (fscanf(f, "%lld", &v) == 1) {
                ids.push_back(v);
              }
              fclose(f);
            }
            if (!ids.empty()) {
              (void)hipMalloc(&spec_oracle_dev, ids.size() * sizeof(long long));
              (void)hipMemcpy(spec_oracle_dev, ids.data(), ids.size() * sizeof(long long),
                              hipMemcpyHostToDevice);
              spec_oracle_n = (int)ids.size();
            }
          }
          char const *kv = getenv("MPK_SPEC_K");
          char const *cv = getenv("MPK_SPEC_CORRUPT");
          int const k = kv ? atoi(kv) : 0;
          int const corrupt = cv ? atoi(cv) : 0;
          int zero4[4] = {0, 0, 0, 0};
          unsigned long long zst[4] = {0ull, 0ull, 0ull, 0ull};
          (void)hipMemcpyToSymbol(HIP_SYMBOL(g_spec_oracle), &spec_oracle_dev, sizeof spec_oracle_dev);
          (void)hipMemcpyToSymbol(HIP_SYMBOL(g_spec_oracle_n), &spec_oracle_n, sizeof spec_oracle_n);
          (void)hipMemcpyToSymbol(HIP_SYMBOL(g_spec_k), &k, sizeof k);
          (void)hipMemcpyToSymbol(HIP_SYMBOL(g_spec_corrupt), &corrupt, sizeof corrupt);
          char const *cfv = getenv("MPK_SPEC_CONF");
          char const *kmv = getenv("MPK_SPEC_KMIN");
          float const conf = cfv ? (float)atof(cfv) : 0.f;
          int const kmin = kmv ? atoi(kmv) : 0;
          (void)hipMemcpyToSymbol(HIP_SYMBOL(g_spec_conf), &conf, sizeof conf);
          (void)hipMemcpyToSymbol(HIP_SYMBOL(g_spec_kmin), &kmin, sizeof kmin);
#if defined(MPK_E3_CONF) && defined(MPK_E3_CONF_D)
          {
            char const *cdv = getenv("MPK_SPEC_CONF_D");
            float const confd = cdv ? (float)atof(cdv) : conf;
            (void)hipMemcpyToSymbol(HIP_SYMBOL(g_spec_confd), &confd, sizeof confd);
            fprintf(stderr, "[SPEC] EAGLE3 chain continues while the probability product >= %.3f\n", confd);
          }
#endif
#ifdef MPK_MULTI_DEV // MDEV_SPEC: one shared state buffer on device 0, its address on every device
          {
            static void *s_spec_sh = nullptr;
            if (s_spec_sh == nullptr) {
              mpk_mdev_check(hipMalloc(&s_spec_sh, sizeof(MpkSpecShared)), "spec state", 0);
              (void)hipMemsetAsync(s_spec_sh, 0, sizeof(MpkSpecShared), default_stream);
              mpk_mdev_symbol_all(&g_spec_sh, &s_spec_sh, sizeof s_spec_sh);
              g_mpk_spec_sh_host = s_spec_sh;  // MPK_SPEC_REQ_RESET
              (void)hipGetDevice(&g_mpk_spec_sh_dev);
            }
            (void)hipMemsetAsync(s_spec_sh, 0, sizeof zero4, default_stream);
            (void)hipMemsetAsync(static_cast<char *>(s_spec_sh) + offsetof(MpkSpecShared, stats), 0,
                                 sizeof zst, default_stream);
            (void)hipStreamSynchronize(default_stream);
          }
#else
          (void)hipMemcpyToSymbol(HIP_SYMBOL(g_spec_state), zero4, sizeof zero4);
          (void)hipMemcpyToSymbol(HIP_SYMBOL(g_spec_stats), zst, sizeof zst);
#endif
          fprintf(stderr, "[SPEC] virtual slots %d, k=%d, corrupt=%d, oracle %d tokens (%s)\n",
                  MPK_MAX_NUM_BATCHED_TOKENS, k, corrupt, spec_oracle_n, op ? op : "none");
          fprintf(stderr, "[SPEC] confidence gate %.3f, kmin %d\n", conf, kmin);
        }
#endif
#ifdef MPK_EAGLE3
        {
          kernel::MpkEagle3 e3{};
          std::vector<unsigned long long> pp;
          if (char const *pv = getenv("MPK_EAGLE3_PTRS")) {
            char const *c = pv;
            while (*c) {
              char *end = nullptr;
              unsigned long long const v = strtoull(c, &end, 16);
              if (end == c) {
                break;
              }
              pp.push_back(v);
              c = (*end == ',') ? end + 1 : end;
            }
          }
#ifdef MPK_EAGLE3_FP8
          size_t const e3_want = 34;  // + nine per-row scales
#else
          size_t const e3_want = 25;
#endif
          if (pp.size() == e3_want) {
            size_t k = 0;
            auto U = [&]() { return (unsigned short const *)(uintptr_t)pp[k++]; };
            e3.fc = U(); e3.wq = U(); e3.wk = U(); e3.wv = U(); e3.wo = U();
            e3.wg = U(); e3.wu = U(); e3.wd = U(); e3.wlm = U();
            e3.n_hidden = U(); e3.n_input = U(); e3.n_post = U(); e3.n_final = U(); e3.embed = U();
            e3.d2t = (long long const *)(uintptr_t)pp[k++];
            e3.kc = (unsigned short *)(uintptr_t)pp[k++];
            e3.vc = (unsigned short *)(uintptr_t)pp[k++];
            e3.vec = (float *)(uintptr_t)pp[k++];
            e3.draft_out = (long long *)(uintptr_t)pp[k++];
            for (int a = 0; a < 3; ++a) e3.apo[a] = U();
            for (int a = 0; a < 3; ++a) e3.ws[a] = (float const *)(uintptr_t)pp[k++];
#ifdef MPK_EAGLE3_FP8
            auto F = [&]() { return (float const *)(uintptr_t)pp[k++]; };
            e3.sfc = F(); e3.swq = F(); e3.swk = F(); e3.swv = F(); e3.swo = F();
            e3.swg = F(); e3.swu = F(); e3.swd = F(); e3.swlm = F();
#endif
            e3.tokens = global_runtime_config.tokens;
            e3.prompt_len = global_runtime_config.prompt_length;
            char const *wsv = getenv("MPK_EAGLE3_WS_STRIDE");
            char const *mpv = getenv("MPK_EAGLE3_MAXPOS");
            char const *nwv = getenv("MPK_EAGLE3_NW");
            e3.ws_stride = wsv ? atoi(wsv) : 0;
            e3.maxpos = mpv ? atoi(mpv) : 0;
            e3.nworkers = nwv ? atoi(nwv) : 0;
            e3.enabled = (e3.ws_stride > 0 && e3.maxpos > 0 && e3.nworkers > 0) ? 1 : 0;
          }
          (void)hipMemcpyToSymbol(HIP_SYMBOL(kernel::g_e3), &e3, sizeof e3);
          fprintf(stderr, "[EAGLE3] draft head %s: %zu pointers, ws stride %d, maxpos %d, workers %d\n",
                  e3.enabled ? "on" : "OFF", pp.size(), e3.ws_stride, e3.maxpos, e3.nworkers);
        }
#endif
#ifdef MPK_DFLASH
        {
          kernel::MpkDflash df{};
          std::vector<unsigned long long> pp;
          if (char const *pv = getenv("MPK_DFLASH_PTRS")) {
            char const *c = pv;
            while (*c) {
              char *end = nullptr;
              unsigned long long const v = strtoull(c, &end, 16);
              if (end == c) {
                break;
              }
              pp.push_back(v);
              c = (*end == ',') ? end + 1 : end;
            }
          }
          if (pp.size() == 44) {
            size_t k = 0;
            auto B8 = [&]() { return (unsigned char const *)(uintptr_t)pp[k++]; };
            auto F = [&]() { return (float const *)(uintptr_t)pp[k++]; };
            auto U = [&]() { return (unsigned short const *)(uintptr_t)pp[k++]; };
            df.fc = B8(); df.wq = B8(); df.wk = B8(); df.wv = B8(); df.wo = B8(); df.wg = B8(); df.wu = B8();
            df.wd = B8();
            df.sfc = F(); df.swq = F(); df.swk = F(); df.swv = F(); df.swo = F(); df.swg = F(); df.swu = F();
            df.swd = F();
            df.bq = F(); df.bk = F(); df.bv = F(); df.bo = F();
            df.nq = U(); df.nk = U(); df.nin = U(); df.npost = U(); df.nhid = U(); df.nfin = U();
            df.wlm = B8(); df.slm = B8(); df.embed = U(); df.rope = F();
            df.kc = (unsigned short *)(uintptr_t)pp[k++];
            df.vc = (unsigned short *)(uintptr_t)pp[k++];
            df.vec = (float *)(uintptr_t)pp[k++];
            df.draft_out = (long long *)(uintptr_t)pp[k++];
            for (int a = 0; a < kernel::DF_NTL; ++a) df.apo[a] = U();
            for (int a = 0; a < kernel::DF_NTL; ++a) df.ws[a] = F();
            df.tokens = global_runtime_config.tokens;
            df.prompt_len = global_runtime_config.prompt_length;
            char const *wsv = getenv("MPK_DFLASH_WS_STRIDE");
            char const *mpv = getenv("MPK_DFLASH_MAXPOS");
            char const *nwv = getenv("MPK_DFLASH_NW");
            df.ws_stride = wsv ? atoi(wsv) : 0;
            df.maxpos = mpv ? atoi(mpv) : 0;
            df.nworkers = nwv ? atoi(nwv) : 0;
            df.enabled = (df.ws_stride > 0 && df.maxpos > 0 && df.nworkers > 0) ? 1 : 0;
          }
          (void)hipMemcpyToSymbol(HIP_SYMBOL(kernel::g_df), &df, sizeof df);
          fprintf(stderr, "[DFLASH] drafter %s: %zu pointers, block %d, ws stride %d, maxpos %d, workers %d, scratch %d floats\n",
                  df.enabled ? "on" : "OFF", pp.size(), kernel::DF_B, df.ws_stride, df.maxpos, df.nworkers,
                  kernel::DF_V_FLOATS);
        }
#endif
#ifdef MPK_W2_MALLPF
        {
          char const *mv = getenv("MPK_W2PF");
          unsigned const md = mv ? (unsigned)atoi(mv) : 0u;
          char const *tv = getenv("MPK_W2PF_NS");
          unsigned const td = tv ? (unsigned)(atoi(tv) / 10) : 0u;
          (void)hipMemcpyToSymbol(HIP_SYMBOL(g_mpk_w2pf_mode), &md, sizeof md);
          (void)hipMemcpyToSymbol(HIP_SYMBOL(g_mpk_w2pf_ticks), &td, sizeof td);
          fprintf(stderr, "[W2PF] mover W2 MALL prefetch mode %u, start %u ticks\n", md, td);
        }
#endif
#ifdef MPK_PFD89_RT
        {
          char const *pv = getenv("MPK_PFD89_NS");
          unsigned const pt = pv ? (unsigned)(atoi(pv) / 10) : 0u;
          (void)hipMemcpyToSymbol(HIP_SYMBOL(g_mpk_pfd89_ticks), &pt, sizeof pt);
          fprintf(stderr, "[PFD89_RT] ranks 8-9 Phase 9 DMA delay %u ticks\n", pt);
        }
#endif
#ifdef MPK_KV_EARLY_STAGE
        {
          char const *kv = getenv("MPK_KVES");
          int const ks = kv ? atoi(kv) : 0;
          (void)hipMemcpyToSymbol(HIP_SYMBOL(g_mpk_kves), &ks, sizeof ks);
          fprintf(stderr, "[KVES_RT] K/V early staging %d (split at %d)\n", ks,
                  (int)MPK_KVES_LO);
        }
#endif
#ifdef MPK_MULTI_DEV
#undef hipMemcpyToSymbol
#endif
        if (hipMemcpyToSymbolAsync(HIP_SYMBOL(g_aid_flag_rep),
                                   dev,
                                   sizeof dev,
                                   0,
                                   hipMemcpyHostToDevice,
                                   default_stream) != hipSuccess ||
            hipStreamSynchronize(default_stream) != hipSuccess) {
          fprintf(stderr,
                  "[AID] split flags: hipMemcpyToSymbol failed, falling "
                  "back to the shared buffer\n");
          s_rep[0] = s_rep[1] = nullptr;
        }
#ifdef MPK_RPROBE
        mpk_rprobe::g_rep[0] = s_rep[0];
        mpk_rprobe::g_rep[1] = s_rep[1];
#endif
      }
    }
    // On `default_stream`, then synchronized, rather than a bare hipMemset.
    // hipMemset lands on the NULL stream, and every stream this kernel uses is
    // created hipStreamNonBlocking -- which by definition does NOT synchronize
    // with the NULL stream. The clear could therefore still be in flight when
    // the megakernel starts, and zero a flag that had already been published:
    // a lost release, i.e. a hang rather than a wrong answer, on whichever
    // fraction of runs the timing happened to overlap.
    for (int i = 0; i < 2; i++) {
      if (s_rep[i] != nullptr) {
        (void)hipMemsetAsync(
            s_rep[i], 0, mirage::aid::kFlagRepBytes, default_stream);
      }
    }
#ifdef MPK_ROUTER_OPROJ_PARTIAL
    // The router partials need scratch even without the replicas (NPS1, stock
    // driver): one plain buffer then serves as both AIDs' copy. Its tags are
    // layer epochs, so it is cleared per launch like the replicas.
    {
      static void *s_rop[2] = {nullptr, nullptr};
      static bool s_rop_tried = false;
      if (!s_rop_tried) {
        s_rop_tried = true;
        if (s_rep[0] != nullptr && s_rep[1] != nullptr) {
          s_rop[0] = s_rep[0];
          s_rop[1] = s_rep[1];
        } else if (hipExtMallocWithFlags(&s_rop[0],
                                         mirage::aid::kFlagRepBytes,
                                         hipDeviceMallocUncached) ==
                   hipSuccess) {
          // Uncached: in NPS1 the lines interleave over both AIDs, and a
          // reader XCD on the far AID would otherwise keep polling a stale
          // L2 copy of the aggregate tags.
          s_rop[1] = s_rop[0];
          printf("[AID] router partials: shared uncached scratch %p (no replicas)\n",
                 s_rop[0]);
        }
        int *dev[2] = {static_cast<int *>(s_rop[0]),
                       static_cast<int *>(s_rop[1])};
        (void)hipMemcpyToSymbolAsync(HIP_SYMBOL(g_rop_rep),
                                     dev,
                                     sizeof dev,
                                     0,
                                     hipMemcpyHostToDevice,
                                     default_stream);
      }
      if (s_rop[0] != nullptr && s_rop[0] != s_rep[0]) {
        (void)hipMemsetAsync(
            s_rop[0], 0, mirage::aid::kFlagRepBytes, default_stream);
      }
    }
#endif
    (void)hipStreamSynchronize(default_stream);
  }
#if defined(MPK_AID_NC_REP) && defined(MPK_AID_SPLIT_FLAGS)
  // AID-local NC scratch: one plain allocation, large enough to be its own
  // BO, so halves placement puts its first half in AID0 and its second in
  // AID1. Counters there take device-scope atomics at local memory. Cleared
  // per launch on default_stream, like the replicas.
  {
    static char *s_nc = nullptr;
    static bool s_nc_tried = false;
    constexpr size_t kNcBytes = 64ull << 20;
    if (!s_nc_tried) {
      s_nc_tried = true;
      if (hipMalloc(reinterpret_cast<void **>(&s_nc), kNcBytes) == hipSuccess) {
        int *dev[2] = {reinterpret_cast<int *>(s_nc),
                       reinterpret_cast<int *>(s_nc + kNcBytes / 2)};
#ifdef MPK_MULTI_DEV // MDEV_NC_AID1: the AID1 half on the first partition of AID1
        static char *s_nc1 = nullptr;
        {
          int cur = 0;
          (void)hipGetDevice(&cur);
          (void)hipSetDevice(g_mdev.dev[MPK_MULTI_DEV / 2]);
          mpk_mdev_check(hipMalloc(reinterpret_cast<void **>(&s_nc1), kNcBytes / 2), "nc AID1", MPK_MULTI_DEV / 2);
          (void)hipSetDevice(cur);
        }
        dev[1] = reinterpret_cast<int *>(s_nc1);
#endif
        // MPK_AID_NC_REP_NULL: the no-op arm -- allocate and clear, but
        // leave the pointers null so every counter keeps its old address.
        if (getenv("MPK_AID_NC_REP_NULL") != nullptr) {
          dev[0] = nullptr;
          dev[1] = nullptr;
          printf("[AID] nc scratch: MPK_AID_NC_REP_NULL, pointers left null\n");
        }
        if (hipMemcpyToSymbolAsync(HIP_SYMBOL(g_aid_nc_rep), dev, sizeof dev, 0,
                                   hipMemcpyHostToDevice,
                                   default_stream) != hipSuccess ||
            hipStreamSynchronize(default_stream) != hipSuccess) {
          fprintf(stderr, "[AID] nc scratch: hipMemcpyToSymbol failed\n");
          s_nc = nullptr;
        } else {
#ifdef MPK_MULTI_DEV // MDEV_NC_REP: one peer-visible allocation, same pointers everywhere
          mpk_mdev_symbol_all(HIP_SYMBOL(g_aid_nc_rep), dev, sizeof dev);
#endif
          printf("[AID] nc scratch at %p (AID0 half) and %p (AID1 half)\n",
                 (void *)dev[0], (void *)dev[1]);
#ifdef MPK_NC_PROBE
          {
            auto probe_buf = [](char const *tag, char *p, size_t bytes, int n) {
              void *rb = nullptr;
              size_t rs = 0;
              hipMemGetAddressRange((hipDeviceptr_t *)&rb, &rs,
                                    (hipDeviceptr_t)p);
              printf("[NCPROBE] %s ptr %p: allocation range base %p size %zu MiB\n",
                     tag, (void *)p, rb, rs >> 20);
              std::vector<unsigned long long> a;
              for (int k = 0; k < n; k++) {
                a.push_back((unsigned long long)p + (bytes / n) * k + 4096);
              }
              mpk_ncprobe::probe_list(tag, a, (unsigned long long)p);
            };
            probe_buf("nc64", s_nc, kNcBytes, 32);
            char *t256 = nullptr;
            if (hipMalloc(reinterpret_cast<void **>(&t256), 256ull << 20) ==
                hipSuccess) {
              probe_buf("tmp256", t256, 256ull << 20, 16);
              hipFree(t256);
            }
          }
#endif
        }
      } else {
        s_nc = nullptr;
      }
    }
    if (s_nc != nullptr) {
      (void)hipMemsetAsync(s_nc, 0, 64 << 10, default_stream);
      (void)hipMemsetAsync(s_nc + kNcBytes / 2, 0, 64 << 10, default_stream);
#ifdef MPK_VTN_MOE
      // MPK_VTN_N7B: the per-token router tile slots (slot 23 is never written and must read zero).
      (void)hipMemsetAsync(s_nc + (size_t)MPK_NC_ROP1_INTS * 4, 0,
                           (size_t)(mpk_rop_nc_ints(MPK_ROP_NTOK) - MPK_NC_ROP1_INTS) * 4 /* MPK_VTN_T6 */, default_stream);
#ifdef MPK_VTN_QMP
      (void)hipMemsetAsync(s_nc + (size_t)MPK_NC_QROW_INTS * 4, 0, (size_t)MPK_NC_QROW_TOTAL_INTS * 4, default_stream);
#endif
      (void)hipMemsetAsync(s_nc + kNcBytes / 2 + (size_t)MPK_NC_ROP1_INTS * 4, 0,
                           (size_t)(mpk_rop_nc_ints(MPK_ROP_NTOK) - MPK_NC_ROP1_INTS) * 4 /* MPK_VTN_T6 */, default_stream);
#ifdef MPK_VTN_QMP
      (void)hipMemsetAsync(s_nc + kNcBytes / 2 + (size_t)MPK_NC_QROW_INTS * 4, 0, (size_t)MPK_NC_QROW_TOTAL_INTS * 4, default_stream);
#endif
#endif
#ifdef MPK_MULTI_DEV // MDEV_NC_CLEAR: the AID1 half is its own allocation on the first partition of AID1
      {
        int *nc1 = nullptr;
        (void)hipMemcpyFromSymbol(&nc1, HIP_SYMBOL(g_aid_nc_rep), sizeof nc1, sizeof nc1);
        if (nc1 != nullptr) {
          (void)hipMemsetAsync(nc1, 0, 64 << 10, default_stream);
#ifdef MPK_VTN_MOE
          (void)hipMemsetAsync(reinterpret_cast<char *>(nc1) + (size_t)MPK_NC_ROP1_INTS * 4, 0,
                               (size_t)(mpk_rop_nc_ints(MPK_ROP_NTOK) - MPK_NC_ROP1_INTS) * 4 /* MPK_VTN_T6 */, default_stream);
#ifdef MPK_VTN_QMP
          (void)hipMemsetAsync(reinterpret_cast<char *>(nc1) + (size_t)MPK_NC_QROW_INTS * 4, 0, (size_t)MPK_NC_QROW_TOTAL_INTS * 4, default_stream);
#endif
#endif
        }
      }
#endif
      (void)hipStreamSynchronize(default_stream);
    }
  }
#endif
  // Per-AID copies of the ml pointer tables. Read-only after setup, so plain
  // AID_LOCAL (MTYPE_RW) is enough -- no COHERENT, and therefore no DF-CS
  // directory pressure. Done once; nothing rewrites them between launches.
  {
    static bool s_ml_tried = false;
    if (!s_ml_tried) {
      s_ml_tried = true;
      int const nl = global_runtime_config.ml_num_layers;
      if (nl > 0 && global_runtime_config.ml_input_table != nullptr) {
        size_t const in_b = (size_t)8 * nl * MAX_INPUTS_PER_TASK * sizeof(void *);
        size_t const out_b =
            (size_t)8 * nl * MAX_OUTPUTS_PER_TASK * sizeof(void *);
        void *ml[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};
        bool ok = true;
#ifdef MPK_MULTI_DEV // MDEV_ML_TABLES: [aid][in/out][device] VAs of one GEM BO each
        void *mlv[2][2][MPK_MULTI_DEV] = {};
#endif
        for (int aid = 0; aid < 2 && ok; aid++) {
#ifdef MPK_MULTI_DEV
          ok = mirage::aid::alloc_in_aid_mdev(in_b, aid, 0, MPK_MULTI_DEV, g_mdev.dev, mlv[aid][0]) &&
               mirage::aid::alloc_in_aid_mdev(out_b, aid, 0, MPK_MULTI_DEV, g_mdev.dev, mlv[aid][1]);
          ml[aid][0] = ok ? mlv[aid][0][0] : nullptr;
          ml[aid][1] = ok ? mlv[aid][1][0] : nullptr;
#else
          ml[aid][0] = mirage::aid::alloc_in_aid(in_b, aid);
          ml[aid][1] = mirage::aid::alloc_in_aid(out_b, aid);
#endif
          ok = ml[aid][0] != nullptr && ml[aid][1] != nullptr;
          if (ok) {
            ok = hipMemcpy(ml[aid][0],
                           global_runtime_config.ml_input_table,
                           in_b,
                           hipMemcpyDeviceToDevice) == hipSuccess &&
                 hipMemcpy(ml[aid][1],
                           global_runtime_config.ml_output_table,
                           out_b,
                           hipMemcpyDeviceToDevice) == hipSuccess;
          }
        }
        if (ok) {
          void *in_p[2] = {ml[0][0], ml[1][0]};
          void *out_p[2] = {ml[0][1], ml[1][1]};
          ok = hipMemcpyToSymbol(HIP_SYMBOL(g_aid_ml_in),
                                 in_p,
                                 sizeof in_p) == hipSuccess &&
               hipMemcpyToSymbol(HIP_SYMBOL(g_aid_ml_out),
                                 out_p,
                                 sizeof out_p) == hipSuccess;
        }
#ifdef MPK_MULTI_DEV // MDEV_ML_SYMS
        if (ok) {
          int cur = 0;
          (void)hipGetDevice(&cur);
          for (int d = 1; d < MPK_MULTI_DEV; d++) {
            void *ip[2] = {mlv[0][0][d], mlv[1][0][d]};
            void *op[2] = {mlv[0][1][d], mlv[1][1][d]};
            (void)hipSetDevice(g_mdev.dev[d]);
            mpk_mdev_check(hipMemcpyToSymbol(HIP_SYMBOL(g_aid_ml_in), ip, sizeof ip), "g_aid_ml_in", d);
            mpk_mdev_check(hipMemcpyToSymbol(HIP_SYMBOL(g_aid_ml_out), op, sizeof op), "g_aid_ml_out", d);
          }
          (void)hipSetDevice(cur);
        }
#endif
        printf("[AID] ml tables: %s (%zu + %zu B per AID)\n",
               ok ? "replicated per AID" : "unavailable, using shared",
               in_b,
               out_b);
      }
    }
  }
#endif
#ifdef MPK_PLACE_GUARD
  // MPK_PLACE_GUARD_V1: the pinned decision words, mapped on every device.
  static unsigned *s_place_h = nullptr;
  if (s_place_h == nullptr) {
    void *h = nullptr;
    if (hipHostMalloc(&h, 8192, hipHostMallocMapped) == hipSuccess && h != nullptr) {
      memset(h, 0, 8192);
      void *d = nullptr;
      if (hipHostGetDevicePointer(&d, h, 0) != hipSuccess || d == nullptr) {
        d = h;
      }
      if (hipMemcpyToSymbolAsync(HIP_SYMBOL(g_place_host), &d, sizeof d, 0, hipMemcpyHostToDevice,
                                 default_stream) == hipSuccess &&
          hipStreamSynchronize(default_stream) == hipSuccess) {
        s_place_h = static_cast<unsigned *>(h);
        g_mpk_place_h = s_place_h;
#ifdef MPK_MULTI_DEV
        mpk_mdev_symbol_all((void const *)&g_place_host, &d, sizeof d);
#endif
      }
    }
    fprintf(stderr, "[PLACE] guard %s (%d ms)\n", s_place_h ? "armed" : "FAILED", (int)(MPK_PLACE_GUARD));
  }
mpk_place_retry:;
  if (s_place_h != nullptr) {
    memset(s_place_h + 16, 0, 8192 - 64);
    __atomic_store_n(&s_place_h[3], (unsigned)(global_runtime_config.num_workers * MPK_XCD_OVERSUB),
                     __ATOMIC_RELAXED);
    __atomic_store_n(&s_place_h[0], 0u, __ATOMIC_RELAXED);
    __atomic_store_n(&s_place_h[1], 0u, __ATOMIC_RELAXED);
    __atomic_store_n(&s_place_h[2],
                     (unsigned)((global_runtime_config.num_workers + global_runtime_config.num_local_schedulers) *
                                MPK_XCD_OVERSUB),
                     __ATOMIC_RELEASE);
  }
#endif
#ifdef MPK_RESID_MAP
  static unsigned *s_resid_h = nullptr;
  static std::atomic<int> s_resid_done{0};
  static int s_resid_gen = 0;
  if (s_resid_h == nullptr) {
    void *h = nullptr;
    if (hipHostMalloc(&h, 1152 * sizeof(unsigned), hipHostMallocMapped) ==
            hipSuccess &&
        h != nullptr) {
      memset(h, 0, 1152 * sizeof(unsigned));
      void *d = nullptr;
      if (hipHostGetDevicePointer(&d, h, 0) != hipSuccess || d == nullptr) {
        d = h;
      }
      if (hipMemcpyToSymbolAsync(HIP_SYMBOL(g_resid_host), &d, sizeof d, 0,
                                 hipMemcpyHostToDevice,
                                 default_stream) == hipSuccess &&
          hipStreamSynchronize(default_stream) == hipSuccess) {
        s_resid_h = static_cast<unsigned *>(h);
#ifdef MPK_MULTI_DEV // MDEV_RESID
        mpk_mdev_symbol_all((void const *)&g_resid_host, &d, sizeof d);
#endif
      }
    }
    fprintf(stderr, "[RESID] host map %s\n", s_resid_h ? "armed" : "FAILED");
  }
  int const resid_gen = ++s_resid_gen;
  if (s_resid_h != nullptr) {
    memset(s_resid_h, 0, 1152 * sizeof(unsigned));
  }
#endif
  // Progress is per-launch, so clear it before the kernel starts rather than
  // after it ends: a streaming reader polling from another thread must never
  // see the previous request's count and emit tokens that do not exist yet.
  if (g_progress_host != nullptr) {
    for (int i = 0; i < MPK_MAX_NUM_BATCHED_REQUESTS; i++) {
      __atomic_store_n(&g_progress_host[i], 0, __ATOMIC_RELEASE);
    }
  }
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
#ifdef USE_NVSHMEM
    nvshmem_barrier_all();
#endif
  }
  int num_schedulers = global_runtime_config.num_local_schedulers +
                       global_runtime_config.num_remote_schedulers;
  if (global_runtime_config.split_worker_scheduler) {
#ifdef MPK_MULTI_DEV // MDEV_LAUNCH
    mpk_mdev_launch(default_stream);
#ifdef MPK_RESID_MAP // MDEV_RESID
    if (s_resid_h != nullptr) {
      int const rnw = global_runtime_config.num_workers;
      int const rns = global_runtime_config.num_local_schedulers;
      unsigned *const rhm = s_resid_h;
      fprintf(stderr, "[RESID] launch %d: kernels enqueued on %d devices\n", resid_gen,
              MPK_MULTI_DEV);
      std::thread([rhm, rnw, rns, resid_gen]() {
        for (int t = 1; t <= 60; t++) {
          std::this_thread::sleep_for(std::chrono::seconds(1));
          if (s_resid_done.load() >= resid_gen) {
            return;
          }
          if (t == 15 || t == 45) {
            mpk_resid_report(rhm, rnw, rns, resid_gen, t);
            int nogate = 0, exited = 0;
            for (int w = 0; w < rnw && w < 248; w++) {
              unsigned const a = __atomic_load_n(&rhm[256 + w], __ATOMIC_RELAXED);
              unsigned const g = __atomic_load_n(&rhm[512 + w], __ATOMIC_RELAXED);
              unsigned const e = __atomic_load_n(&rhm[768 + w], __ATOMIC_RELAXED);
              nogate += !(g & 0x80000000u);
              exited += (e & 0x80000000u) != 0;
              if (w < rnw / MPK_MULTI_DEV || !(g & 0x80000000u) || (e & 0x80000000u)) {
                fprintf(stderr, "[RESID]   w%d claim=%s rank=%u len=%u | gate=%s iter=%u term=%u | exit=%s\n",
                        w, (a & 0x80000000u) ? "y" : "n", (a >> 16) & 0x7fffu, a & 0xffffu,
                        (g & 0x80000000u) ? "y" : "n", g & 0xffffffu, (g >> 24) & 1u,
                        (e & 0x80000000u) ? "y" : "n");
              }
            }
            fprintf(stderr, "[RESID]   workers never past the gate %d, exited early %d\n", nogate, exited);
#ifdef MPK_TOPK_SPINREC  // MPK_TOPK_SPINREC_V1
            for (int w = 0; w < rnw && w < 248; w++) {
              unsigned const e = __atomic_load_n(&rhm[768 + w], __ATOMIC_RELAXED);
              if (e != 0u && !(e & 0x80000000u)) {
                fprintf(stderr, "[RESID]   w%d topk poll: %uK spins, first stale word %u, %u layers old\n", w,
                        (e >> 16) & 0x7fffu, (e >> 11) & 31u, e & 2047u);
              }
            }
#endif
          }
        }
      }).detach();
    }
#endif
#else
    (void)cudaStreamWaitEvent(global_runtime_config.worker_stream,
                              global_runtime_config.prepare_done_event,
                              0);
    (void)cudaStreamWaitEvent(global_runtime_config.scheduler_stream,
                              global_runtime_config.prepare_done_event,
                              0);

    // The split kernel does not support NVSHMEM because
    // nvshmemx_collective_launch launches kernels sequentially, which blocks
    // the interaction between the worker kernel and the scheduler kernel
    worker_kernel<<<dim3(global_runtime_config.num_workers * MPK_XCD_OVERSUB,
                         1,
                         1),
                    dim3(WORKER_NUM_THREADS, 1, 1),
                    MAX_DYNAMIC_SHARED_MEMORY_SIZE /*smem*/,
                    global_runtime_config.worker_stream>>>(
        global_runtime_config);
    scheduler_kernel<<<dim3(global_runtime_config.num_local_schedulers *
                             MPK_XCD_OVERSUB,
                         1,
                         1),
                       dim3(128, 1, 1),
                       MPK_SCHED_LDS /*smem*/,
                       global_runtime_config.scheduler_stream>>>(
        global_runtime_config);
#ifdef MPK_RESID_MAP
    if (s_resid_h != nullptr) {
      int const rnw = global_runtime_config.num_workers * MPK_XCD_OVERSUB;
      int const rns =
          global_runtime_config.num_local_schedulers * MPK_XCD_OVERSUB;
      unsigned *const rhm = s_resid_h;
      fprintf(stderr, "[RESID] launch %d: worker + scheduler kernels enqueued\n",
              resid_gen);
      std::thread([rhm, rnw, rns, resid_gen]() {
        for (int t = 1; t <= 60; t++) {
          std::this_thread::sleep_for(std::chrono::seconds(1));
          if (s_resid_done.load() >= resid_gen) {
            return;
          }
          if (t == 15 || t == 45) {
            mpk_resid_report(rhm, rnw, rns, resid_gen, t);
          }
        }
      }).detach();
    }
#endif

    (void)cudaEventRecord(global_runtime_config.worker_done_event,
                          global_runtime_config.worker_stream);
    (void)cudaEventRecord(global_runtime_config.scheduler_done_event,
                          global_runtime_config.scheduler_stream);

    (void)cudaStreamWaitEvent(
        default_stream, global_runtime_config.worker_done_event, 0);
    (void)cudaStreamWaitEvent(
        default_stream, global_runtime_config.scheduler_done_event, 0);
#endif

#ifdef MPK_PRECOMPUTED_DISPATCH
    // Hang-diagnosis watchdog. OFF unless MPK_HOST_WATCHDOG=1.
    //
    // This loop gates when launch_persistent_kernel returns, so it is not
    // free: it ticks, and the kernel finishes mid-tick, so every launch pays
    // out the remainder of the current tick before the response can be sent.
    // At the original 1 s granularity that was a median 498 ms per request at
    // conc=1, landing entirely on the final inter-token gap -- +0.55 ms/token
    // once TPOT averages it over ~900 tokens, or 22% of the reported figure.
    // A 12-token coherence check paid 974 ms of it. No compute happens in
    // that window; it is pure sleep.
    //
    // It exists for real failures (see the 32k-hang notes below) and stays
    // available, but a serving run should not carry a debug poller in the
    // response path. The tick is also 5 ms now rather than 1 s, so enabling
    // it costs ~5 ms instead of ~500.
    static bool const dbg_watchdog = [] {
      char const *e = getenv("MPK_HOST_WATCHDOG");
      return e != nullptr && e[0] == '1';
    }();
    if (dbg_watchdog) {
      fprintf(stderr, "[HOST_DBG] Starting poll loop...\n");
    }
    if (dbg_watchdog && global_runtime_config.precomp_iter_ready &&
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
      // Re-arm for this launch. These are function-static so the detached poll
      // thread can outlive the call, which also means launch N+1 inherits
      // launch N's terminal values: dbg_a_stop=true made the new poll thread
      // exit on its first predicate test, and dbg_a_term=1 made the tick loop
      // break at t=1s. That is exactly the misleading
      // "iter_ready=<terminal> terminate=1 dev_polls=1" a server run reports
      // one second after a relaunch, whatever the kernel is actually doing.
      dbg_a_stop.store(false, std::memory_order_relaxed);
      dbg_a_ir.store(0, std::memory_order_relaxed);
      dbg_a_term.store(0, std::memory_order_relaxed);
      dbg_a_td.store(0, std::memory_order_relaxed);
      dbg_a_polls.store(0, std::memory_order_relaxed);
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
      // Tick at 5 ms, not 1 s. This loop gates when launch_persistent_kernel
      // returns, so its sleep granularity lands directly on the client: the
      // kernel finishes mid-tick and the response waits out the remainder of
      // the sleep. Measured at conc=1, that was 1000 - (finish_ms % 1000) on
      // every launch -- a median 498 ms charged to the final inter-token gap,
      // which TPOT then averages over ~900 tokens as +0.55 ms/token. A
      // 12-token request paid 974 ms of it. Nothing was computing during that
      // window.
      //
      // The reporting cadence is deliberately decoupled below (dbg_ticks_per_s)
      // so the log keeps its once-a-second rhythm and the same 900 s ceiling.
      int const dbg_tick_ms = 5;
      int const dbg_ticks_per_s = 1000 / dbg_tick_ms;
      int const dbg_max_ticks = 900 * dbg_ticks_per_s;
      for (int dbg_i = 0; dbg_i < dbg_max_ticks; dbg_i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(dbg_tick_ms));
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
        // dbg_i counts 5 ms ticks now, so both the 30 s reporting interval and
        // the printed timestamp have to be converted back into seconds.
        if (dbg_i % (30 * dbg_ticks_per_s) == 0 || term) {
          fprintf(
              stderr,
              "[HOST_DBG] t=%ds iter_ready=%llu terminate=%d "
              "tasks_done=%llu dev_polls=%llu (0 is normal; watch `done`)\n",
              (dbg_i / dbg_ticks_per_s) + 1,
              ir,
              term,
              td,
              polls);
        }
        // Dump per-worker state + summary
        // Was `dbg_i >= 2`, i.e. "after 2 s of ticks"; keep that wall-clock
        // meaning rather than letting it fire 10 ms in.
        if (g_dbg_h_worker_state && dbg_i >= 2 * dbg_ticks_per_s) {
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
          bool verbose =
              (stall_ticks >= 1) || (dbg_i % (30 * dbg_ticks_per_s) == 0);
          for (int w = 0; w < g_dbg_num_workers; w++) {
            int *ws = g_dbg_h_worker_state + w * 4;
            int tpos = __atomic_load_n(&ws[0], __ATOMIC_RELAXED);
            int dep = __atomic_load_n(&ws[1], __ATOMIC_RELAXED);
            int done = __atomic_load_n(&ws[2], __ATOMIC_RELAXED);
            int phase = __atomic_load_n(&ws[3], __ATOMIC_RELAXED);
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
                  case 20:
                    pn = "P2-qkv-epoch";
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
                }
                int *a = g_dbg_h_worker_state + g_dbg_num_workers * 8 + w * 4;
                int a0 = __atomic_load_n(&a[0], __ATOMIC_RELAXED);
                int a1 = __atomic_load_n(&a[1], __ATOMIC_RELAXED);
                int a2 = __atomic_load_n(&a[2], __ATOMIC_RELAXED);
                int a3 = __atomic_load_n(&a[3], __ATOMIC_RELAXED);
                fprintf(stderr,
                        "    >>> w%d IN TASK: phase=%d (%s xcd=%d epoch=%d) "
                        "watch[bid=%d obs=%d exp=%d spins=%d]\n",
                        w,
                        wphase,
                        pn,
                        fx,
                        fl,
                        bid,
                        obs,
                        exp_,
                        spins);
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
        if (g_dbg_h_event_counters && dbg_i % (5 * dbg_ticks_per_s) == 0) {
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
      // path only runs under MPK_WORKER_STATE debug builds.
    }
#endif
    (void)cudaStreamSynchronize(global_runtime_config.worker_stream);
    (void)cudaStreamSynchronize(global_runtime_config.scheduler_stream);
    if (s_lts) {
      fprintf(stderr, "[LAUNCH_TS] exit %.6f\n",
              std::chrono::duration<double>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count());
    }
#ifdef MPK_RESID_MAP
    if (s_resid_h != nullptr) {
      s_resid_done.store(resid_gen);
      mpk_resid_report(s_resid_h,
                       global_runtime_config.num_workers * MPK_XCD_OVERSUB,
                       global_runtime_config.num_local_schedulers *
                           MPK_XCD_OVERSUB,
                       resid_gen, -1);
    }
#endif
#ifdef MPK_PLACE_GUARD
    if (s_place_h != nullptr) {
      (void)hipStreamSynchronize(default_stream);
      {
        int const pnw = (int)s_place_h[3], pnb = (int)s_place_h[2];
        unsigned long long const *pt = (unsigned long long const *)(s_place_h + 16);
        unsigned long long tmin = ~0ull, tmax = 0ull;
        int last = -1, seen = 0;
        for (int b = 0; b < pnb && b < 1000; b++) {
          unsigned long long const t = __atomic_load_n(&pt[b], __ATOMIC_RELAXED);
          if (t == 0ull) {
            continue;
          }
          seen++;
          tmin = t < tmin ? t : tmin;
          if (t > tmax) {
            tmax = t;
            last = b;
          }
        }
        double const spread_ms = seen ? (double)(tmax - tmin) / 1.0e5 : 0.0;
        if (spread_ms > 20.0 || __atomic_load_n(&s_place_h[1], __ATOMIC_ACQUIRE) == 2u) {
          fprintf(stderr, "[PLACE] start spread %.1f ms over %d blocks; last to start: %s %d\n", spread_ms,
                  seen, last < pnw ? "worker" : "scheduler", last < pnw ? last : last - pnw);
        }
      }
      if (__atomic_load_n(&s_place_h[1], __ATOMIC_ACQUIRE) == 2u) {
        static int s_place_n = 0;
        fprintf(stderr, "[PLACE] launch abandoned: %u of %u blocks resident within %d ms; relaunching (%d)\n",
                __atomic_load_n(&s_place_h[0], __ATOMIC_RELAXED), __atomic_load_n(&s_place_h[2], __ATOMIC_RELAXED),
                (int)(MPK_PLACE_GUARD), ++s_place_n);
        goto mpk_place_retry;
      }
    }
#endif
#ifdef MPK_RPROBE
    {
      static bool s_rprobe_ran = false;
      if (!s_rprobe_ran) {
        s_rprobe_ran = true;
        mpk_rprobe::run(mpk_rprobe::g_rep);
      }
    }
#endif
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
#ifdef USE_NVSHMEM
  nvshmem_barrier_all();
  nvshmem_finalize();
#endif
  // Free worker and scheduler streams
  (void)cudaEventDestroy(global_runtime_config.prepare_done_event);
  (void)cudaEventDestroy(global_runtime_config.worker_done_event);
  (void)cudaEventDestroy(global_runtime_config.scheduler_done_event);
  (void)cudaStreamDestroy(global_runtime_config.worker_stream);
  (void)cudaStreamDestroy(global_runtime_config.scheduler_stream);
}
