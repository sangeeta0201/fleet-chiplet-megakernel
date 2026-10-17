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


// MPK_VPROBE: routing-epoch stride for the [VPROBE] U print in
// gang_oproj_router_fused_mi300.cuh.  0 = off (default).  Measurement only --
// it prints the activated-expert UNION size so a MoE row fold can be priced.
#ifndef MPK_VPROBE
#define MPK_VPROBE 0
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

// =============================================================================
// Non-Temporal (NT) Load/Store for MI300X
// =============================================================================
//
// CORRECTION -- the paragraph that used to sit here claimed "NT loads/stores
// bypass both L1 and L2 caches, reading/writing directly to HBM memory. This
// avoids relying on cache coherence protocol." That is false on gfx942/gfx950,
// and believing it cost this branch a long intermittent-hang hunt. `nt` is a
// *temporal* hint: it marks the line evict-first (MISS_EVICT / HIT_STREAM in
// the cache-policy tables) so it does not pollute the cache for reuse. It does
// not change the access's *scope*, so an `nt` load still hits whatever is
// already resident in the CU's vector L1 and in this XCD's L2.
//
// Scope is `sc0`/`sc1`, and it is orthogonal to `nt`. What each bit actually
// means, taken from what hipcc emits for __hip_atomic_load/store on gfx950
// rather than from the ISA table:
//
//   load  sc0      -- WORKGROUP scope. Misses vL1; answered by this XCD's L2.
//   load  sc1      -- AGENT scope. Pairs with `buffer_inv sc1` as the acquire.
//   load  sc0 sc1  -- SYSTEM scope. Misses vL1 and L2, goes to memory.
//   store sc1      -- agent release; compiler prefixes `buffer_wbl2 sc1`.
//   store sc0 sc1  -- write through past vL1 and L2.
//   buffer_inv     -- invalidate vL1 only        (workgroup-scope acquire)
//   buffer_inv sc1 -- invalidate vL1 and L2      (agent-scope acquire)
//
// The trap is that `sc0` reads like "the cache bit" and is in fact the WEAKEST
// one. On MI300/MI350 the L2 is per-XCD and not snooped, so an `sc0` load is
// answered from a possibly-stale line whenever the producer sat on a different
// XCD. Two consequences bit this branch in sequence: a plain `nt` poll can be
// answered from the CU's own vL1, which a peer's write-through store never
// invalidates; and an `sc0 nt` poll can be answered from a stale XCD-local L2
// line. Anything used as a cross-XCD barrier poll needs `sc1` at minimum --
// see ld_nt_s32 below.
//
// Why fence is still needed:
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

// MPK_POLL_LAT_PROBE (diagnostic, default off): time every spin-poll load with
// s_memrealtime and log any single load slower than MPK_POLL_LAT_PROBE_US into
// a host-pinned ring that the [ITER] heartbeat thread prints as [PLAT]. Tests
// whether a 1k/1k stall is one poll load that does not return: rocgdb found a
// worker parked for minutes at the s_waitcnt behind its flag load, spin
// counter frozen, with the releasing value already in memory.
#ifndef MPK_POLL_LAT_PROBE
#define MPK_POLL_LAT_PROBE 0
#endif
#ifndef MPK_POLL_LAT_PROBE_US
#define MPK_POLL_LAT_PROBE_US 50000
#endif
#define MPK_PLAT_RING 4096
#define MPK_PLAT_WORDS 8
#if MPK_POLL_LAT_PROBE && defined(__HIP_PLATFORM_AMD__)
// [0] = records claimed; record i occupies words [8 * (i + 1), 8 * (i + 2)).
__device__ unsigned int *g_plat_ring;
__device__ __forceinline__ void mpk_plat_check(unsigned long long t0,
                                               void const *addr,
                                               unsigned long long val) {
#if defined(__HIP_DEVICE_COMPILE__)
  // Where the caller's load is a plain atomic rather than asm with its own
  // wait, the compiler would otherwise read t1 before the load returns.
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  unsigned long long const t1 = __builtin_amdgcn_s_memrealtime();
  if (t1 - t0 > (unsigned long long)MPK_POLL_LAT_PROBE_US * 100ull) {
    unsigned long long pc;
    asm volatile("s_getpc_b64 %0" : "=s"(pc));
    unsigned int *const ring = g_plat_ring;
    if (ring != nullptr) {
      unsigned int const slot = __hip_atomic_fetch_add(
          &ring[0], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      if (slot < MPK_PLAT_RING) {
        unsigned int *const r = ring + MPK_PLAT_WORDS * (1 + slot);
        r[1] = (unsigned int)((t1 - t0) / 100ull);
        r[2] = (unsigned int)pc;
        r[3] = (unsigned int)(pc >> 32);
        r[4] = (unsigned int)(unsigned long long)addr;
        r[5] = (unsigned int)((unsigned long long)addr >> 32);
        r[6] = (unsigned int)val;
        r[7] = (unsigned int)(t0 / 100000ull);
        __hip_atomic_store(&r[0],
                           (blockIdx.x + 1u) | (threadIdx.x << 16),
                           __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
      }
    }
  }
#endif
}
#define MPK_PLAT_T0()                                                          \
  unsigned long long const _plat_t0 = __builtin_amdgcn_s_memrealtime()
#define MPK_PLAT_T1(addr, val)                                                 \
  mpk_plat_check(_plat_t0, (void const *)(addr), (unsigned long long)(val))
#else
#define MPK_PLAT_T0()                                                          \
  do {                                                                         \
  } while (0)
#define MPK_PLAT_T1(addr, val)                                                 \
  do {                                                                         \
  } while (0)
#endif

// MPK_PEER_POLL_NT: the `nt` token on the CROSS-RANK peer poll (ld_sys_u64 and
// ld_sys_u64_x8). Default 0 = dropped. Same mechanism and same reasoning as
// MPK_BAR_POLL_NT below, which measured -0.367 us/barrier and -0.206 ms/token.
#ifndef MPK_PEER_POLL_NT
#define MPK_PEER_POLL_NT 0
#endif

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

// Eight signal lines, ONE round trip.
//
// ld_sys_u64 carries its own s_waitcnt, so a poll pass over seven peer signal
// lines is seven SERIALIZED uncached round trips. That much is true of the
// instruction stream. What is NOT true is that it costs anything: replacing
// the seven trips with this one burst moved the measured peer-wait region by
// 0.00 us/layer (17.49 -> 17.49). See the MEASURED NEUTRAL note on
// MPK_EP_POLL_BATCH in gang_full_layer_fused_mi300.cuh -- the poll is not the
// EP wait's cost, inter-rank skew is, and a poll pass is amortized over a
// wait that is 13x longer than it.
//
// Issuing all eight loads before the single s_waitcnt makes a pass cost one
// round trip. The eight lines are FULL_LAYER_EP_SIGNAL_STRIDE (8 uint64 = 64
// bytes) apart and span 512 bytes, so they all reach through the 13-bit
// signed immediate offset off one base address. Reading my own line too and
// discarding it is cheaper than branching around it.
struct mpk_u64x8 {
  unsigned long long v[8];
};
__device__ __forceinline__ mpk_u64x8
    ld_sys_u64_x8(unsigned long long const *base) {
  mpk_u64x8 r;
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  MPK_PLAT_T0();
#if MPK_PEER_POLL_NT
  asm volatile(
               "global_load_dwordx2 %0, %8, off offset:0 sc0 sc1 nt\n"
               "global_load_dwordx2 %1, %8, off offset:64 sc0 sc1 nt\n"
               "global_load_dwordx2 %2, %8, off offset:128 sc0 sc1 nt\n"
               "global_load_dwordx2 %3, %8, off offset:192 sc0 sc1 nt\n"
               "global_load_dwordx2 %4, %8, off offset:256 sc0 sc1 nt\n"
               "global_load_dwordx2 %5, %8, off offset:320 sc0 sc1 nt\n"
               "global_load_dwordx2 %6, %8, off offset:384 sc0 sc1 nt\n"
               "global_load_dwordx2 %7, %8, off offset:448 sc0 sc1 nt\n"
               "s_waitcnt vmcnt(0)"
#else
  asm volatile(
               "global_load_dwordx2 %0, %8, off offset:0 sc0 sc1\n"
               "global_load_dwordx2 %1, %8, off offset:64 sc0 sc1\n"
               "global_load_dwordx2 %2, %8, off offset:128 sc0 sc1\n"
               "global_load_dwordx2 %3, %8, off offset:192 sc0 sc1\n"
               "global_load_dwordx2 %4, %8, off offset:256 sc0 sc1\n"
               "global_load_dwordx2 %5, %8, off offset:320 sc0 sc1\n"
               "global_load_dwordx2 %6, %8, off offset:384 sc0 sc1\n"
               "global_load_dwordx2 %7, %8, off offset:448 sc0 sc1\n"
               "s_waitcnt vmcnt(0)"
#endif
:
                 "=v"(r.v[0]),
                 "=v"(r.v[1]),
                 "=v"(r.v[2]),
                 "=v"(r.v[3]),
                 "=v"(r.v[4]),
                 "=v"(r.v[5]),
                 "=v"(r.v[6]),
                 "=v"(r.v[7])
               : "v"(base)
               : "memory");
  MPK_PLAT_T1(base, r.v[0]);
#else
#pragma unroll
  for (int i = 0; i < 8; i++) {
    r.v[i] = reinterpret_cast<unsigned long long volatile const *>(base)[i * 8];
  }
#endif
  return r;
}

// 32-bit barrier-poll load. Every Mechanism-C barrier poll in the gang tasks is
// built on this, so its coherence is the coherence of every barrier in the
// megakernel: 45 call sites, 33 of them inside a `while`.
//
// It used to be `off nt` and nothing else, on the belief -- stated in the
// section header above, now corrected -- that `nt` bypasses cache. It does not;
// `nt` is a temporal hint and scope is a separate axis. Fixing that to `sc0 nt`
// was still wrong, one scope short, and the NP=8 hang survived it unchanged.
//
// The scope encoding on gfx950, straight out of the compiler rather than
// inferred (hipcc -S, __hip_atomic_load on gfx950):
//
//   workgroup acquire : global_load_dword ... sc0
//   agent     acquire : global_load_dword ... sc1   + buffer_inv sc1
//   agent     release : buffer_wbl2 sc1 + global_store_dword ... sc1
//
// So `sc0` is WORKGROUP scope, not agent. An `sc0` load misses the CU's vector
// L1 and is then answered by this XCD's L2 -- and on MI300/MI350 the L2 is
// per-XCD and not snooped, so a line this XCD cached before the release is
// still there and still stale. Only `sc1` reaches agent scope.
//
// `sc0 sc1` here, i.e. bypass both, which is system scope and strictly stronger
// than needed. The alternative -- `sc1` plus a `buffer_inv sc1` inside every
// poll loop, which is what the compiler emits -- would invalidate the whole L2
// on every spin iteration for 240 workers, throwing away the weight lines the
// prefetch-across-barrier work exists to keep resident. Bypassing on the one
// flag load is the cheaper half of that trade. Same encoding as ld_sys_u64,
// which is the one poll primitive on this branch that never hung.
//
// Cost note: the poll now goes to MALL/HBM instead of L2. It is one dword per
// spin per waiter behind an s_sleep(1), against a barrier whose flag line is
// read-shared -- but if the barrier tail regresses, this is the first thing to
// re-measure.
//
// MPK_BAR_POLL_NT: RE-MEASURED, and the cost note above was right to flag it.
// `nt` is a replacement-policy hint asking the cache NOT to retain the line.
// On the flag it is exactly backwards: 29 blocks share each of the eight lines
// and every one of them polls it, so the line is the most read-shared address
// in the kernel, and `nt` invites MALL to drop it between polls and re-fetch
// from DRAM. Dropping the hint changes NO coherence -- `sc0 sc1` still bypasses
// L1 and the per-XCD L2, which is the whole reason this primitive exists, and
// MALL sits below the L2s so a MALL hit returns the current value.
//
// Measured in tests/standalone/test_barrier_release.hip, a 232-block replica of
// the release block at gang_mla_attn_fused_mi300.cuh:746-784, two independent
// runs of 2000 rounds x 3 reps, min:
//
//   arm                                     us/barrier   vs shipping
//   sc0 sc1 nt   sleep1  8 flags (ships)      4.120        --
//   sc0 sc1      sleep1  8 flags              3.760       -0.361
//   sc0 sc1      sleep0  8 flags              3.746       -0.374
//   sc0 sc1      sleep1  32 flags             3.625       -0.495
//   sc0 sc1 nt   sleep1  1 flag               4.660       +0.540
//   sc0 sc1 nt   sleep1  232 private lines    4.973       +0.852
//   arrival atomic + syncthreads, no release  1.835       -2.285
//
// Read across the table: line sharing is a real cost in BOTH directions -- one
// flag for the grid is 0.54 worse, but a private line per block is 0.85 worse
// still, because the release fan-out is one thread issuing N stores. Eight is
// near the optimum and 32 is slightly better. s_sleep and buffer_inv are free.
// Only the `nt` hint is a straight win, and it is a one-token change.
#ifndef MPK_BAR_POLL_NT
#define MPK_BAR_POLL_NT 0
#endif
__device__ __forceinline__ int ld_nt_s32(int *addr) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  int val;
  MPK_PLAT_T0();
#if MPK_BAR_POLL_NT
  asm volatile("global_load_dword %0, %1, off sc0 sc1 nt\n"
               "s_waitcnt vmcnt(0)"
               : "=v"(val)
               : "v"(addr)
               : "memory");
#else
  asm volatile("global_load_dword %0, %1, off sc0 sc1\n"
               "s_waitcnt vmcnt(0)"
               : "=v"(val)
               : "v"(addr)
               : "memory");
#endif
  MPK_PLAT_T1(addr, val);
  return val;
#else
  return *reinterpret_cast<int volatile *>(addr);
#endif
}

// System-scope 64-bit load: sc0 sc1, so it misses L1 AND the device L2 every
// time and goes to the fabric.
//
// `nt` on its own is only a replacement-policy hint -- an NT load still HITS a
// resident line. That is survivable for the intra-GPU barrier flags, whose
// pollers overwhelmingly arrive after the flag has moved, so their first load
// misses and fetches the new value. It is not survivable for a peer GPU's
// signal line, where a poller routinely arrives BEFORE the remote store lands:
// its first load pulls the stale value into this XCD's L2, and every later
// iteration of the spin hits that line. The store side is already sc0 sc1, so
// the value does reach the fabric -- the reader simply never looks. Nothing in
// the spin loop evicts the line either, so it clears only when unrelated
// traffic or a host-side DMA happens to flush L2, which is why the failure
// looks like ~17 s per layer of "progress" clocked by the 1 Hz debug poll
// rather than like a deadlock.
//
// Pay sc0 sc1 only where a remote agent is the writer; the intra-GPU flags
// keep ld_nt_s32.
__device__ __forceinline__ unsigned long long int
    ld_sys_u64(unsigned long long int *addr) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  unsigned long long int val;
  MPK_PLAT_T0();
#if MPK_PEER_POLL_NT
  asm volatile("global_load_dwordx2 %0, %1, off sc0 sc1 nt\n"
               "s_waitcnt vmcnt(0)"
               : "=v"(val)
               : "v"(addr)
               : "memory");
#else
  // MPK_PEER_POLL_NT=0 (default): same `nt` drop that bought -0.206 ms/token on
  // the intra-GPU barrier poll (commit a494a9d), applied to the CROSS-RANK peer
  // poll. `nt` is a replacement-policy hint only -- sc0 sc1 already bypasses L1
  // and the per-XCD L2, so this changes no coherence and cannot reintroduce the
  // stale-L2 livelock the sc0 sc1 comment above is about. It only stops asking
  // MALL not to retain a signal line that every worker re-reads every spin.
  asm volatile("global_load_dwordx2 %0, %1, off sc0 sc1\n"
               "s_waitcnt vmcnt(0)"
               : "=v"(val)
               : "v"(addr)
               : "memory");
#endif
  MPK_PLAT_T1(addr, val);
  return val;
#else
  return *reinterpret_cast<unsigned long long int volatile *>(addr);
#endif
}

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
//
// What a cross-GPU handoff costs, measured 2026-09-25 by ping-pong between
// one 512-thread workgroup on each of two idle MI350X (GPUs 4-5 and 4-7
// agree), one-way:
//   8-byte epoch store into the peer, peer polls its own memory  0.80-0.88 us
//   1536 B of payload via st_wt into the peer, s_waitcnt vmcnt(0),
//     then the epoch store (this codebase's push + drain + signal)  1.74-1.84 us
//   the same payload as 384 {data, epoch} 8-byte pairs, no drain,
//     the peer polling the pairs themselves (TileRT's one-shot push) 0.96-1.04 us
// So data-as-signal saves ~0.8 us per exchange; the multi-microsecond peer
// waits in the stage stamps are skew between the ranks, not transport.

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

// Write-through 128-bit store (4x dword = 16 bytes, sc0 sc1).
// Same instruction as st_wt_zero128 but with a caller-supplied payload; the
// address must be 16-byte aligned, as global_store_dwordx4 requires.
__device__ __forceinline__ void st_wt_u128(void *addr,
                                           unsigned int v0,
                                           unsigned int v1,
                                           unsigned int v2,
                                           unsigned int v3) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  typedef unsigned int v4u32 __attribute__((ext_vector_type(4)));
  v4u32 payload = {v0, v1, v2, v3};
  asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1"
               :
               : "v"(addr), "v"(payload)
               : "memory");
#else
  reinterpret_cast<unsigned int volatile *>(addr)[0] = v0;
  reinterpret_cast<unsigned int volatile *>(addr)[1] = v1;
  reinterpret_cast<unsigned int volatile *>(addr)[2] = v2;
  reinterpret_cast<unsigned int volatile *>(addr)[3] = v3;
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

// MPK_BAR_FLAG_MAX: publish a Mechanism-C release flag MONOTONICALLY.
//
// Every per-XCD barrier flag in this codebase holds an epoch that only ever
// counts up -- entry_expected, qkv_expected, routing_expected, all of them are
// task_layer_idx + 1, and task_layer_idx is run-monotonic. Two different
// threads write that flag: the last arriver's eight-way release fan-out, and
// the self-heal republish. Both used a plain write-through store, and a plain
// store carries no ordering against a store issued by a DIFFERENT thread on a
// different XCD. So a straggler still inside layer L-1's spin can land its
// republish of `L` AFTER layer L's fan-out has already published `L+1`, and
// the flag goes BACKWARDS. Whoever has not read it yet then waits on an epoch
// that will never be re-published.
//
// That is the exact state task #135 captured: at layer 11 the entry flag for
// XCD 7 read 11 while 29 of that XCD's 30 workers had already observed 12 and
// moved on to the next barrier. One worker, w232, spun 3.4 M times on a flag
// whose correct value had been written and then overwritten. The poll cannot
// be at fault -- ld_nt_s32 issues `sc0 sc1`, which bypasses vL1 and L2 and
// reads memory, so the 11 was really there.
//
// atomicMax makes the write order-independent: a late store of a lower epoch
// is simply dropped. Cost is eight device-scope atomics per barrier, executed
// by the single last-arriver thread, replacing eight write-through stores.
// Same-address atomics pipeline in the L2 atomic unit and the arrival side
// already pays one (measured at 19 ns, see hier_barrier_arrive), so this is
// far below the 3.77 us a rendezvous costs. Set to 0 to get the old stores
// back for an A/B.
#ifndef MPK_BAR_FLAG_MAX
#define MPK_BAR_FLAG_MAX 1
#endif
// MPK_SIG_WBL2 (diagnostic, default off): write this XCD's L2 back after
// every release flag and every cross-rank signal store. Tests whether the
// multi-second 1k/1k stalls are a published value lingering in a writer's L2:
// captures show senders that have moved on while their signal is absent from
// the receiver's HBM, and a sender's own local copy absent from its own.
// MEASURED NULL, 2026-09-23, 2 runs: prefill still stalled (600 and 1031
// ms/iter, single stalls of 476 s and 518 s); decode clean at +1.2 ms median.
#ifndef MPK_SIG_WBL2
#define MPK_SIG_WBL2 0
#endif
#if MPK_SIG_WBL2
#define MPK_SIG_FLUSH()                                                        \
  asm volatile("buffer_wbl2 sc0 sc1\n\ts_waitcnt vmcnt(0)" ::: "memory")
#else
#define MPK_SIG_FLUSH()                                                        \
  do {                                                                         \
  } while (0)
#endif
__device__ __forceinline__ void st_flag_u32(void *addr, unsigned int val) {
#if MPK_BAR_FLAG_MAX
  // Device scope: the reader may be on any XCD. RELAXED, deliberately -- the
  // ordering this needs is already there. Every caller puts an explicit
  // `s_waitcnt vmcnt(0)` ahead of the fan-out, and the payload it advertises
  // was written with st_wt_u32, i.e. `sc0 sc1`, which bypasses L2 and lands in
  // memory. __ATOMIC_RELEASE would add a `buffer_wbl2 sc1` L2 writeback in
  // front of every one of the eight, buying nothing over the store this
  // replaces (a plain store carries no ordering at all). Compiles to a bare
  // `global_atomic_smax`, verified on gfx950.
  __hip_atomic_fetch_max(static_cast<int *>(addr), (int)val, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
#else
  st_wt_u32(addr, val);
#endif
  MPK_SIG_FLUSH();
}

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
#ifdef MPK_USE_NT_MEMORY
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
#ifdef MPK_USE_NT_MEMORY
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
#ifdef MPK_USE_NT_MEMORY
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
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
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

// ───────────────────────────────────────────────────────────────────────────
// Two-level arrival for the Mechanism-C hierarchical barrier.
//
// The flat mechanism has all `arrivals` workers atomically bump ONE counter
// (`bar[8 * HIER_STRIDE]`), so the rendezvous costs `arrivals` serialized L2
// round trips on a single cache line. Measured on GLM-5 at 232 workers that
// is 3.77 us/layer, via the correct-output MPK_NULL_PHASES probe (11.005 ->
// 12.180 ms at four extra rendezvous). The same probe with this two-level
// arrival measures 2.11 us -- 11.663 ms at four extra, -44% of the mechanism.
//
// Each worker bumps its OWN XCD's counter (29 arrivals, eight cache lines
// contending in parallel); the last arriver per XCD bumps the global one
// (8 arrivals); the last of those fans the release out. Serialized atomics
// per rendezvous drop from `arrivals` to `per_xcd + 8` -- 232 to 37.
//
// Layout, all offsets in units of HIER_STRIDE ints from `bar`:
//   [0 .. 7]     per-XCD write-through release flags   (unchanged)
//   [8]          global arrival counter                (unchanged, but now
//                                                       counts XCDs, not
//                                                       workers, so it is
//                                                       monotonic at 8/epoch)
//   [tree + x]   per-XCD arrival counter, one per cache line -- eight of them
//                sharing a line would serialize exactly the way the flat
//                counter does, which is the thing being removed.
//
// The counters stay monotonic and are tested modularly, never reset, so no
// worker from the next epoch can observe a zeroed one. Both levels must use
// release ordering: the XCD-level atomic publishes this worker's stores to
// its XCD leader, and the global one republishes the leader's observation.
//
// Returns true on exactly one thread in the whole grid -- the one that owes
// the release fan-out.
__device__ __forceinline__ bool
    hier_barrier_tree_arrive(int *bar,
                             int tree_off_slots,
                             int hier_stride,
                             int per_xcd,
                             int xcd_id) {
  int *const my_cnt = &bar[(tree_off_slots + xcd_id) * hier_stride];
  int const xprev = atom_add_release_gpu_s32(my_cnt, 1);
  if ((xprev % per_xcd) != per_xcd - 1) {
    return false;
  }
  int const gprev = atom_add_release_gpu_s32(&bar[8 * hier_stride], 1);
  return (gprev % 8) == 7;
}

// MPK_BAR_SKEW: accumulate, per named rendezvous, the nanoseconds between the
// FIRST worker's arrival and the LAST worker's arrival. That interval is the
// part of a barrier's cost that a null-barrier probe cannot see. Correctness-
// preserving and O(1) per epoch. Off by default.
//
// Two accumulators per slot, and the second is the useful one:
//   ns  -- first-arrival to last-arrival at THIS barrier ("spread"). When the
//          phase has fewer tiles per XCD than workers, the first arriver has
//          no tile, so the spread IS the tile's duration.
//   gap -- previous barrier's last arrival to this one's. The barriers of a
//          layer complete in order, so the gaps are per-phase makespans and
//          they SUM TO THE LAYER. Divide either by cnt to get ns per layer
//          per iteration directly -- no iteration count needed.
//
// ── MEASURED. THE FIRST PER-PHASE WALL DECOMPOSITION. ────────────────────
// n=1 (skew3/skew4 agree within 5%), correct generated text, instrument
// costs ~0.35 ms of the 11.3 ms wall so read the SHARE, not the absolute:
//
//   slot  phase                     gap us/layer   spread us/layer   share
//   0     MoE W2 + layer entry          35.18          12.12         23%
//   7     MoE W13                       34.56          16.92         22%
//   2     qkv_a                         32.40          14.83         21%
//   1     decode + merge                18.85          24.60         12%
//   3     q_b (+ W_UK)                  15.27           6.48         10%
//   6     o_proj + router                9.63           2.50          6%
//   5     W_UV                           7.92           5.07          5%
//                                      ------
//                                      153.9 us/layer
//
// The headline: in the three biggest phases MORE THAN HALF the phase is not
// tile work. qkv_a's tiles take 14.83 us and the phase costs 32.40. W13's
// take 16.92 and the phase costs 34.56. W_UV, the phase with the most idle
// workers, has gap - spread = 2.85 us, which agrees with the 2.71 us null
// rendezvous -- so the barrier mechanism is NOT the missing 17 us in the big
// phases. Whatever it is, it is worth more than every remaining named lever
// on the board combined.
//
// slot 1's spread exceeds its gap because the decode and merge populations
// differ from the previous barrier's, so its "first arriver" waited across a
// phase it did not participate in. Only compare spread to gap when the two
// barriers have the same population.
//
// ── MEASURED. THE QKV_A GAP IS THE EP COLLECTIVE, AND 79% OF IT IS SKEW. ──
// The stage stamps below cut the 48.55 us qkv_a phase five ways, all measured
// from the layer-entry barrier's last arrival (BAR_SKEW=1, wall 11.92):
//
//   S0            release propagation                 1.73 us   (min 1.25)
//   S3 - S0       local EP fold + 7 peer stores       5.37 us   (min 4.62)
//   S4 - S3       cross-rank peer wait               18.79 us   (min 10.71)
//   S1 - S4       eight-flag release fan-out          1.33 us
//   S2 - S1       per-worker wake / dispatch          0.88 us
//   gap[2] - S2   qkv_a tile + barrier arrival       20.45 us
//
// So the three candidates that looked plausible from the outside -- release
// propagation, wake/dispatch, cold first-tile latency -- are together under
// 3 us and are dead. The phase's non-tile half is the EP collective: 24.2 us,
// of which 18.8 is one thread waiting on seven peers.
//
// Probe cost is attributed, not assumed: MPK_BAR_SKEW=2 doubles every stamp
// and costs +0.95 ms of wall (+9.85 us/layer summed over the gaps), against
// +1.17 ms for the whole probe over the 10.78 baseline. The cost is linear in
// stamp count and lands mostly on the two MoE barriers (+4.5 and +5.7 us);
// slot 2 moved +0.13 us, so qkv_a's decomposition is not the probe's doing.
// MPK_ML_PTR_PREFETCH: see the multi-layer loop in persistent_kernel.cuh.
// Default on with MPK_MOE_SKIP_TOPK (see there): 4.284 -> 4.260 ms together.
#ifndef MPK_ML_PTR_PREFETCH
#define MPK_ML_PTR_PREFETCH 1
#endif

// MPK_ABL_ML_BOUNDARY: the SUBTRACTIVE ceiling probe for the multi-layer
// loop's per-layer boundary. WRONG OUTPUT by construction.
//
// The naive subtraction -- skip the TaskDesc pointer refresh -- is invalid,
// and the note on MPK_ML_BOUNDARY_PAD below says why: every layer would then
// reuse one layer's weights, turning a 65 MB/layer HBM stream into a MALL
// hit and reporting an enormous spurious win. So this probe deletes the
// boundary's COST while leaving the byte stream alone. It:
//
//   * implies MPK_ML_PTR_PREFETCH, so the 34+13 pointer entries come out of
//     registers loaded a whole layer ago instead of two dependent cold round
//     trips at the boundary -- the loads still happen, just not here;
//   * deletes the __syncthreads after the pointer refresh;
//   * at level 2, also deletes the inter-layer threadfence_gpu and its
//     __syncthreads.
//
// Threads therefore read a mix of layer ml-1 and layer ml pointers -- wrong
// output, but every pointer is a live buffer of the right shape, so no
// fault, and the weight stream stays ~per-layer distinct.
//
// ── MEASURED. THE BOUNDARY IS NOT A LEVER. RETRACTION BELOW. ──────────────
//
// Paired A/B against a control in the same batch, min-of-115 per run, n=3:
//
//   level 1 (prefetch + drop the refresh join)
//     control  10.259 / 10.234 / 10.230   mean 10.241
//     probe    10.274 / 10.244 / 10.252   mean 10.257    +0.016 ms
//   level 2 (also drop the inter-layer threadfence_gpu and its join)
//     control  10.281 / 10.281 / 10.271   mean 10.278
//     probe    10.234 / 10.224 / 10.224   mean 10.227    -0.051 ms
//
// The probe really does less work -- s_barrier in the device image goes
// 342 -> 340 -> 338 and the instruction count 114042 -> 113947 -> 113893.
// Deleting the ENTIRE per-layer boundary is worth 0.051 ms, 0.65 us/layer,
// a fifth of the 0.26 ms wall noise floor and the same size as the batch
// drift between the two controls above.
//
// RETRACTION. MPK_ML_BOUNDARY_PAD's additive slope priced this region at
// 1533 ticks * 1.22 = ~1.4 ms, and the b7aa3aa / 9063140 stage stamps put
// 8.21-16.50 us/layer inside it. Both numbers are real; the INFERENCE from
// them was wrong. A uniform addition at a point on the critical path costs
// what you add, but the converse does not hold: the boundary sits in front
// of a GPU-wide rendezvous whose arrival spread is 82 of 161 us/layer, so
// removing uniform time there just moves every worker's arrival earlier by
// the same amount and the last arriver still sets the release. This is the
// fourth instance of the same rule -- see
// glm-counted-region-time-before-a-barrier-is-not-a-lever and
// glm-deleting-a-whole-rendezvous-is-neutral. A pad slope is an upper bound
// on a deletion only when the region is NOT followed by a barrier that
// re-synchronizes what you saved.
//
// MEASURED NOT ABLATABLE: the __syncthreads after the _linear_reserved
// publish. Deleting it hangs the run at kernel launch -- task_desc is a
// shared slot, every worker derives its three barrier release values from
// that store, and without the join the other 255 threads wait on an epoch
// that never releases. It is sync correctness, not bookkeeping, so it stays
// in both levels.
#ifndef MPK_ABL_ML_BOUNDARY
#define MPK_ABL_ML_BOUNDARY 0
#endif

// MPK_FOLD_W2: GLM's EP fold moves from the head of layer L+1 into layer L's
// W2 epilogue (gang_moe_w2_tp_linear_mxfp8_kernel), which therefore writes
// layer L+1's ep_gather. The multi-layer loop hands it that pointer in
// TaskDesc input slot MPK_FOLD_W2_SLOT, read off the next layer's own input
// [27] -- the same vehicle MPK_QKVA_PF_SLOT uses for the next qkv_a weight.
//
// BROKEN on the 2026-10-08 defaults (MPK_EP_LL, MPK_OPROJ_LL=2,
// MPK_FL_INLINE), not yet localized: alone (MPK_EP_LL=0) it faults with an
// illegal address in the first iteration; with MPK_EP_LL (the W2 epilogue
// pushing LL words, GangW2FoldPush::ll_slot) it wedges in the first
// iteration, yet completes with correct output under MPK_WORKER_STATE=1.
// 2026-10-09 on the dbcd01f defaults: MPK_EP_LL=0 completes, output
// correct, with rank 0 under rocgdb precise-memory (no fault to catch);
// MPK_EP_LL completes 3/3 at full speed but at 6.182 6.184 6.180 ms against
// 5.89, and one of the three runs' text differs from the other two and from
// the default -- a race in the fold path, timing-dependent like the faults.
// Off; a fix has to clear both the race and the +0.29 ms.
#ifndef MPK_FOLD_W2
#define MPK_FOLD_W2 0
#endif
#define MPK_FOLD_W2_SLOT 32

// MPK_ROUTER_XSPLIT: under MPK_ROUTER_STREAM the XCD's router tiles split the
// o_proj slot sum between them and swap slices through an XCD-private row in
// the fused layer's counter buffer (FULL_LAYER_XSPLIT_SLOT), instead of every
// tile reading every slot for the whole row. Per XCD: two 64-byte lines of
// flags (a word per router tile), then MPK_XSPLIT_ROW_LINES lines of bf16 row.
//
// Every per-XCD block is a whole number of 128-byte L2 lines and starts on
// one. The blocks are written with plain stores and live dirty in eight
// non-coherent L2s; a line shared by two XCDs' blocks is written back whole by
// whichever L2 evicts it, which restores the other XCD's stale flags in memory
// -- and an `nt` poll, which streams the line out of L2, then reads them back.
// That hung the first build (193-line blocks: XCD x's row tail shared a line
// with XCD x+1's flags).
//
// MEASURED 2026-10-08, NP=8 1024/1024, n=3 interleaved, every token identical
// to the default: 7.157 7.160 7.159 -> 7.040 7.030 7.025 ms. Step 1 was
// 10.4 of the router's 13.4 instrumented us (SP6[1]). demo.py turns it on
// wherever MPK_ROUTER_STREAM is.
#ifndef MPK_ROUTER_XSPLIT
#define MPK_ROUTER_XSPLIT 0
#endif
#define MPK_XSPLIT_FLAG_LINES 2
#define MPK_XSPLIT_ROW_LINES 192
#define MPK_XSPLIT_XCD_INTS ((MPK_XSPLIT_FLAG_LINES + MPK_XSPLIT_ROW_LINES) * 16)

// MPK_ROUTER_LL: the router tiles publish each logit as one 8-byte
// (epoch << 32 | bf16) word, and one block per XCD polls all of them and runs
// the TopK into XCD-private routing copies. Replaces the arrival atomic, the
// single elected tail with its GPU-wide fence and release, and the MoE
// workers' cold cross-XCD reads of the routing. Region (FULL_LAYER_RLL_SLOT):
// MPK_RLL_WORD_LINES of words, then per XCD a flag line, a line of TopK
// weights, and MPK_RLL_IDS_LINES each of active ids and routing indices.
//
// MEASURED 2026-10-08, NP=8 1024/1024, n=3 interleaved, every token identical
// to the default: 7.159 7.133 7.151 -> 7.079 7.038 7.002 ms. The tail it
// replaces was 7.0 instrumented us per layer (SP6[4]): 4.4 selection, 1.6
// fence + release. On by default under MPK_OPROJ_RP (demo.py).
#ifndef MPK_ROUTER_LL
#define MPK_ROUTER_LL 0
#endif
// Same 128-byte rule as MPK_ROUTER_XSPLIT for the per-XCD copies (36 lines).
// MPK_QKV_XSPLIT: qkv_a's EP resolve (the eight-slot sum + residual every
// tile re-reads for the whole row, ~46% of the tile) split over the XCD's
// first 16 qkv_a tiles, which swap slices through an XCD-private row laid out
// exactly like MPK_ROUTER_XSPLIT's (FULL_LAYER_QXS_SLOT). Same per-element add
// order and per-thread ssq order as the unsplit resolve.
//
// MEASURED 2026-10-08, NP=8 1024/1024, n=3 interleaved against the XSPLIT +
// ROUTER_LL default, tokens identical: 6.903 6.891 6.888 -> 6.977 7.014 6.974
// (+0.09). The unsplit resolve is already one batched pass over L2-resident
// slots; the exchange's extra rendezvous costs more than the re-reads. Off,
// except as MPK_EP_LL's receiver.
#ifndef MPK_QKV_XSPLIT
#define MPK_QKV_XSPLIT 0
#endif

// MPK_EP_LL: the head-of-layer EP fold writes its slot as (epoch << 32 |
// bf16x2) 8-byte words into every rank's copy, and MPK_QKV_XSPLIT's resolvers
// validate them word by word. Gone: the fold's drain of its XGMI pushes, the
// fold-done count, the leader's peer signals and peer wait, and the per-XCD
// release -- the drain-then-signal pair MPK_FOLD_W2's note prices at ~2 us
// of acks. The fold region of ep_gather doubles (world -> 2 * world planes,
// ahead of MPK_OPROJ_RP's). Tail folds (LM head input) keep the bf16 slot.
//
// MEASURED 2026-10-08, NP=8 1024/1024, n=3 interleaved against the own-heads
// default, tokens identical: 6.820 6.819 6.824 -> 6.492 6.504 6.477 ms
// (-0.33).
// Twice what the fold's stamped cross-rank wait promised: the resolve now
// starts on the bytes, not on a release one fan-out behind them. demo.py
// turns it on under MoE EP.
#ifndef MPK_EP_LL
#define MPK_EP_LL 0
#endif
// MPK_FL_INLINE: inline the GLM fused layer into the worker's dispatch. As a
// __noinline__ call it saves and restores its callee-saved VGPRs through
// scratch on every layer -- 60 of them (87 under MPK_OPROJ_LL=2), i.e.
// ~60 KB of spill traffic per worker per layer. Private segment 336 -> 80 B.
// MEASURED 2026-10-08, NP=8 1024/1024, n=3 interleaved, tokens identical:
// 6.497 6.475 6.515 -> 6.312 6.299 6.297 ms (-0.19); with MPK_OPROJ_LL=2,
// whose router-chain gain those spills had been eating, 6.172 6.163 6.130
// (-0.34). Default on.
#ifndef MPK_FL_INLINE
#define MPK_FL_INLINE 1
#endif
// MPK_ROUTER_INLINE: likewise the router tile (gang_rmsnorm_linear_bias_topk_
// kernel), whose ~45 arguments arrive partly on the stack -- 20 scratch loads
// at the head of a critical-path phase.
// MEASURED NEGATIVE 2026-10-08, NP=8 1024/1024, n=3 interleaved on the
// FL_INLINE + OPROJ_LL=2 default: 6.164 6.130 6.180 -> 6.242 6.256 6.250 ms,
// and the tokens move (the inlined gate dot contracts differently). Off.
#ifndef MPK_ROUTER_INLINE
#define MPK_ROUTER_INLINE 0
#endif
// MPK_DECODE_INLINE: likewise the MLA decode (gang_mla_decode_kernel and
// mla_decode_absorbed), one call pair per decode item.
// MEASURED NEUTRAL 2026-10-08, NP=8 1024/1024, n=3 interleaved (on a base
// that also had MPK_ROUTER_INLINE): 6.240 6.254 6.260 -> 6.269 6.256 6.266
// ms. Its prologue is 8 VGPRs now, not the 60 the note in the decode had;
// inlined, the AGPR overflow grows 106 -> 136. Off.
// Re-run 2026-10-13 on 49d37f8: illegal memory access on ranks 0-1, three
// attempts of three, so not measurable as it stands. Not debugged.
#ifndef MPK_DECODE_INLINE
#define MPK_DECODE_INLINE 0
#endif

// MPK_EP_LL_FOLD_LAST: under MPK_EP_LL, fold on each XCD's last worker instead
// of xcd_rank 0, which is also split resolver 0 (see the fold site).
// MEASURED NEUTRAL 2026-10-08, NP=8 1024/1024, n=3 interleaved, tokens
// identical: 6.510 6.488 6.509 -> 6.484 6.467 6.505 (-0.017, overlapping).
// Resolver 0's own acks are back before the peers' words land. Off.
#ifndef MPK_EP_LL_FOLD_LAST
#define MPK_EP_LL_FOLD_LAST 1
#endif
#if MPK_EP_LL && !MPK_QKV_XSPLIT
#error "MPK_EP_LL: the LL slots are only read by MPK_QKV_XSPLIT's resolvers"
#endif
// Under MPK_FOLD_W2 the W2 epilogue pushes the words itself (GangW2FoldPush::
// ll_slot), except into the tail, which keeps the bf16 slot and signal.
#if MPK_EP_LL && (defined(MPK_QKV_EP_FOLD) || defined(MPK_QKV_PRO_HOIST))
#error "MPK_EP_LL: the in-tile EP folds read the gather slots as bf16"
#endif

// MPK_OPROJ_LL: MPK_OPROJ_RP's f32 partials travel as (epoch << 32 | f32)
// 8-byte words, written by the o_proj GEMV straight into every rank's slot,
// and the split router validates them word by word -- no read-back push, no
// drain-then-signal, no wait on the o_proj release. The slot region doubles
// (ep_gather RP planes 3x -> 5x world).
//
// MEASURED NEGATIVE 2026-10-08, NP=8 1024/1024, n=3 interleaved against the
// XSPLIT + ROUTER_LL default, tokens identical: 6.895 6.883 6.902 -> 7.582
// 7.584 7.619 ms (+0.70). Polling one sentinel word per (slot, o_proj tile)
// instead of the whole slice per spin did not move it (7.544 -> 7.6). The
// worker image grows 348 VGPR / 96 AGPR / 496 B private against 319 / 63 /
// 336. Not yet localized; off.
//
// == 2 keeps the GEMV as it is and moves the words into the push: the GEMV
// plain-stores its f32 rows to an XCD-private block in L2 (FULL_LAYER_OLL_SLOT),
// and the tile's wave 3 alone reads them back and writes the words, local and
// peers. vmcnt retires in issue order, loads and stores mixed, so a wave with
// XGMI stores in flight cannot finish any later load until they are acked;
// keeping them on wave 3 leaves the router's validating waves 0-1 free of
// them. The o_proj arrival, election, signals and release are skipped.
// MEASURED NEUTRAL 2026-10-08, NP=8 1024/1024, n=2 interleaved against the
// EP_LL default, tokens identical: 6.491 6.493 -> 6.506 6.505 ms. The worker
// image grows to 352 VGPR / 96 AGPR / 544 B private (319 / 63 / 336), all of
// it callee-saved spill in the fused layer's prologue -- and the stamps put
// the router chain 3.3 us/layer shorter. With that prologue gone
// (MPK_FL_INLINE): 6.312 6.299 6.297 -> 6.172 6.163 6.130 ms (-0.15).
// demo.py sets 2 under MPK_ROUTER_XSPLIT.
#ifndef MPK_OPROJ_LL
#define MPK_OPROJ_LL 0
#endif
// f32 rows per XCD for MPK_OPROJ_LL == 2: 768 at GLM-5's 6144 / 8.
#define MPK_OLL_XCD_INTS 768
// MPK_QKVA_IDLE_PF: the qkv_a-idle workers prefetch the XCD's qkv_a weight
// into L2 while the tiles resolve (gang_mla_attn_fused_mi300.cuh).
// MEASURED 2026-10-09, NP=8 1024/1024, n=3 interleaved, tokens
// identical: 6.160 6.145 6.157 -> 6.119 6.093 6.094 ms (-0.05). Gated with
// MPK_TOPK_LL_PREBIAS: ppl512 2.4838, longseq 256/512/1024 G1 PASS. Default 1.
#ifndef MPK_QKVA_IDLE_PF
#define MPK_QKVA_IDLE_PF 1
#endif

// MPK_QB_IDLE_PF: after MPK_QKVA_IDLE_PF's block, the same idle workers pull
// q_b's weight into L2 too (gang_mla_attn_fused_mi300.cuh).
// MEASURED NEGATIVE 2026-10-09, NP=8 1024/1024, interleaved, tokens
// identical: 6.034 6.041 -> 6.117 6.163 6.153 ms. Its bytes compete with
// qkv_a's own L2 stream. Off.
// Re-measured 2026-10-14 on the MPK_KV_LL defaults, where q_b's A tiles land
// 0.84 us after their issue (MPK_TRACE q05), n=3 interleaved, tokens
// identical: 3.401 3.402 3.397 vs 3.394 3.379 3.391 ms (+0.012).
#ifndef MPK_QB_IDLE_PF
#define MPK_QB_IDLE_PF 0
#endif
// MPK_QB_IDLE_PF_AT_US: under MPK_QB_IDLE_PF, the idle workers issue q_b's
// slabs this many us after the attention half's entry instead of right behind
// their qkv_a block. q_b's A tiles land when the slab fetch does (MPK_TRACE
// q11, MPK_QB_PF_EARLY), MPK_QB_DEC_PF starts it at ~8.6 us, and from the
// layer's entry it competes with qkv_a's own stream.
// MEASURED SLOWER 2026-10-14 on 4a21f5e, n=3 interleaved, rank-0 tokens
// identical: at 4 us 3.411 3.408 3.407, at 6 us 3.404 3.402 3.409 vs 3.393
// 3.396 3.388 ms (+0.017, +0.013). The slab fetch MPK_QB_DEC_PF starts at
// ~8.6 us is the better time. Off.
#ifndef MPK_QB_IDLE_PF_AT_US
#define MPK_QB_IDLE_PF_AT_US 0
#endif
// MPK_QB_SELF_PF: each q_b tile's waves 1-3 prefetch its own slab into L2
// during the qkv_a -> q_b rendezvous, after qkv_a's stream has finished
// (gang_mla_attn_fused_mi300.cuh).
// MEASURED NEUTRAL 2026-10-09, NP=8 1024/1024, interleaved, tokens
// identical: 6.038 6.021 -> 6.043 6.056 6.072 ms. q_b + W_UK is not bound
// by its weight fetch. Off.
// Re-measured on 12ad8d7, where q_b idles in the MPK_QA_LL word poll, n=3:
// 3.698 3.685 3.702 vs 3.690 3.698 3.691 ms, null.
#ifndef MPK_QB_SELF_PF
#define MPK_QB_SELF_PF 0
#endif
// Tried 2026-10-13 and not kept: q_b's hoisted A-tile fill -- at
// K_PER_MFMA 128 a wave's four k-tiles, the tile's whole 32 KB slab --
// issued ahead of the MPK_QA_LL word poll instead of after the norm.
// MEASURED SLOWER on 91c270e, n=3 interleaved, rank-0 tokens identical: 3.534
// 3.524 3.531 vs 3.466 3.471 3.467 ms (+0.06); with MPK_QUANT_WIDE 3.519
// 3.515 3.512. MPK_TRACE: on the XCDs whose q_b workers start polling after
// their own qkv_a tile (1-7), S72 moved 1.4-1.5 us later -- the poll's
// vmcnt(0) retires in issue order, so its first round waited on a slab that
// took ~2.6 us to land -- while on XCD 0, polling from 1.4 us, q_b finished
// 0.53 us earlier. The same order is why MPK_QB_SELF_PF, whose waves also
// poll, was null; MPK_QB_DEC_PF issues the slabs from the decode workers.
// MPK_QB_DEC_PF: the XCD's decode workers, idle from their qkv_a tile
// (~9 us) to their query (~16 us), pull the XCD's q_b slabs into L2 an
// eighth each before their KV preload walk (gang_mla_attn_fused_mi300.cuh).
// The q_b tiles' own fill then hits L2.
// MEASURED 2026-10-13 on 91c270e, n=3 interleaved, rank-0 tokens identical:
// 3.462 3.460 3.464 vs 3.470 3.465 3.471 ms (-0.007); with MPK_QUANT_WIDE
// 3.462 3.466 3.468. MPK_TRACE: S72 unmoved (the slabs do not slow the q_a
// word poll), and S73 -> S51 stays ~1.1 us with the slab in L2 -- the
// K-loop was not waiting on it; that span is the MFMAs, the cross-wave sum
// and the words' store acks.
// Default on with MPK_DEC_KVF_LAST and MPK_DEC_KVF_INC. The three together
// vs all three off, 2026-10-13 on 91c270e, n=3 interleaved, rank-0 tokens
// identical: 3.447 3.450 3.447 vs 3.472 3.473 3.471 ms (-0.024). Gates:
// test_mla_decode (both WRITE_THROUGH arms), test_mla_merge, test_st_wt PASS;
// ppl512 2.3670 on all 8 ranks; longseq 256/512/1024 G1 PASS.
#ifndef MPK_QB_DEC_PF
#define MPK_QB_DEC_PF 1
#endif

// MPK_TOPK_LL_PREBIAS: the LL TopK stages the correction bias into LDS during
// its logit poll and polls per thread (topk_ll_noinline). MEASURED
// 2026-10-08, NP=8 1024/1024, n=3 interleaved, tokens identical: 6.156 6.174
// 6.142 -> 6.084 6.091 6.091 ms (-0.07): the bias read was a cold round trip
// behind the last logit. Default 1.
#ifndef MPK_TOPK_LL_PREBIAS
#define MPK_TOPK_LL_PREBIAS 1
#endif
// MPK_ROUTER_KSPLIT: under MPK_ROUTER_XSPLIT + MPK_ROUTER_LL each router tile
// contracts only its own 384-column slice, against all 32 of its XCD's
// experts, and pushes 32 partial logits plus its slice's sum of squares as LL
// words into its XCD's exchange row; the TopK sums the 16 partials per expert
// and applies 1/rms itself. Deletes the XCD row swap, the whole-row re-read
// and the 256-thread reductions from the router's critical path. Each tile
// then writes its slice of the normed row and flags it; the TopK waits on
// those flags before releasing the MoE.
// MEASURED 2026-10-09, NP=8 1024/1024, interleaved: 6.041 6.034 6.032 ->
// 6.000 6.019 6.027 6.004 ms (-0.02) with the TopK on router tile 0. The
// tokens move (the logit and 1/rms sums reassociate, and bf16 logits flip
// near-ties); ppl512 2.4112 against 2.4838, longseq 256/512/1024 G1 PASS.
// Stamps: router work 4.25 -> 2.38 us, but the TopK tail 2.21 -> 4.41, since
// tile 0 wrote its normed slice first and re-polled stale words one at a
// time. With the TopK on the XCD's first worker past the router tiles
// (gang_oproj_router_fused_mi300.cuh) and whole-round re-polls, same tokens:
// 6.039 6.056 6.035 -> 5.958 5.961 5.966 5.962 ms (-0.08). Default on; it
// only engages under XSPLIT + ROUTER_LL + OPROJ_LL.
#ifndef MPK_ROUTER_KSPLIT
#define MPK_ROUTER_KSPLIT 1
#endif
// MPK_MOE_BS1_DECODE: at one row the TopK always emits NUM_TOPK slots (the
// routed k plus the shared expert) and routes slot s through
// routing_indices[e] = s + 1, so the MoE tile decode needs neither the slot
// count nor the routing_indices load: one dependent L2 round trip (the
// expert id) after the routing release instead of four, and W2's segment
// quantizer loads its weights and activations without the id chain.
// MEASURED 2026-10-09, NP=8 1024/1024, interleaved, tokens identical (ppl512
// 2.4112 = control): 5.953 5.955 5.931 -> 5.898 5.900 5.890 5.889 ms
// (-0.05). Default on; MPK_MOE_TP builds only.
#ifndef MPK_MOE_BS1_DECODE
#define MPK_MOE_BS1_DECODE 1
#endif
// MPK_QXS_XOUT_PLAIN_PROBE: pricing probe, WRONG OUTPUT. The qkv_a split
// resolver's x_out store goes to L2 instead of write-through, to price the
// write-through drain in front of its exchange flag. MEASURED 2026-10-09,
// NP=8 1024/1024: 5.897 5.869 vs 5.885 ms -- nothing to recover there.
#ifndef MPK_QXS_XOUT_PLAIN_PROBE
#define MPK_QXS_XOUT_PLAIN_PROBE 0
#endif
// MPK_ATTN_META_PF: the decode tiles walk the decode's dependent prologue
// loads (indptrs, page id, the chunk's KV rows) into L2 while they wait at
// the q_b -> decode barrier, and the merge tiles read qo_indptr across the
// decode -> merge wait (gang_mla_attn_fused_mi300.cuh).
// MEASURED 2026-10-09, NP=8 1024/1024, interleaved, tokens identical
// (ppl512 2.4098 = control): 5.747 5.740 5.724 -> 5.709 5.703 5.705 5.708
// ms (-0.03). Default on.
#ifndef MPK_ATTN_META_PF
#define MPK_ATTN_META_PF 1
#endif
// MPK_ROUTE_LL: the K-split TopK publishes each slot's expert and weight as
// epoch-tagged words in its XCD's L2 (after the normed flags), and the MoE
// tiles validate the words they need instead of polling a release flag that
// sat behind a store drain (topk_ll_noinline, gang_oproj_router_fused).
// One-row TP decode only.
// MEASURED 2026-10-10, NP=8 1024/1024, interleaved, tokens identical:
// 5.699 5.714 5.713 -> 5.683 5.659 5.688 5.669 ms (-0.03). Default on.
#ifndef MPK_ROUTE_LL
#define MPK_ROUTE_LL 1
#endif
// The publisher (topk_ll_noinline) and the readers (gang_oproj_router_fused)
// must agree, or one side waits on a flag the other no longer writes.
#define MPK_ROUTE_LL_ON (MPK_ROUTE_LL && MPK_MOE_BS1_DECODE && MPK_MOE_TP)
// MPK_TOPK_PAD_NS: pricing probe, correct output. Delays the K-split TopK's
// routing release by this many ns on every rank and layer.
// MPK_WUK_PF: the W_UK workers pull their tile's weight into L2 while they
// wait at Phase 3b's XCD barrier (gang_mla_attn_fused_mi300.cuh). The
// MPK_TRACE profile puts the W_UK step at ~3 us behind that barrier.
// MPK_WUK_HOIST: Phase 3b's deferred-rope position read ahead of its barrier,
// and no peer-delta lookups under head-local (gang_mla_attn_fused_mi300.cuh).
// MEASURED NEUTRAL 2026-10-10, NP=8 1024/1024, n=3 interleaved, text
// identical: 4.779 4.791 4.794 -> 4.794 4.785 4.776 ms. Off.
// MPK_WUK_EARLY_PF: MPK_WUK_PF's weight pull issued ahead of the qkv_a -> q_b
// rendezvous instead of after the worker's q_b tile. Under MPK_WUK_LL the
// W_UK worker whose q_b tile finishes last pays the whole HBM fetch behind
// it: MPK_TRACE puts its GEMV at 2.76 us after its own q_b tile against
// 1.40 us after the last q_b tile for workers that finished q_b early.
// MEASURED 2026-10-11, NP=8 1024/1024, interleaved, text identical:
// 4.242 4.242 4.248 -> 4.207 4.199 4.204 ms (-0.041). Default on.
#ifndef MPK_WUK_EARLY_PF
#define MPK_WUK_EARLY_PF 1
#endif
// MPK_ROPE_REMAP: under MPK_WUK_LL the head's rope tail runs on the XCD's
// first worker past the W_UK tiles instead of after the last W_UK tile's
// GEMV, which put it 0.7 us behind that GEMV on the decode's query path.
// MEASURED 2026-10-11, same batch: 4.164 4.165 ms alone (-0.079), 4.130
// 4.130 with MPK_WUK_EARLY_PF (-0.114). Default on.
#ifndef MPK_ROPE_REMAP
#define MPK_ROPE_REMAP 1
#endif
// MPK_TOPK_FLAG_LATE: the TopK checks its XCD's 16 normed-row flags in front
// of the routing's first ll_route word instead of ahead of the selection,
// which reads only the logits. MPK_TRACE: logits ready -> normed flags seen
// -> select start is 0.53 us of the 1.45 us selection's lead-in.
// MEASURED NULL-NEGATIVE 2026-10-11, NP=8 1024/1024, text identical: 3.894
// 3.885 vs 3.876 3.882 3.875 ms. The gate's own load + vmcnt(0) is an L2
// round trip after the selection even when the flags landed long before.
// AND IT HANGS: 2 of 3 runs stalled in the first decode iteration with one
// rank short of iteration 1, no fault, HBM idle (watchdog, retried clean).
// The clears inside the selection do not touch the flags; cause not found.
// DO NOT ENABLE either flag until it is.
// MPK_TOPK_FLAG_PF (needs FLAG_LATE): issue that read ahead of the selection
// and wait on it only at the gate.
#ifndef MPK_TOPK_FLAG_LATE
#define MPK_TOPK_FLAG_LATE 0
#endif
#ifndef MPK_TOPK_FLAG_PF
#define MPK_TOPK_FLAG_PF 0
#endif
// MPK_TOPK_FLAG_OVL: the TopK's first read of the 16 normed flags issued as
// the partials land (S40) and waited for at the flag poll, so it overlaps the
// logit pass (S40 -> S41 0.45 us; S41 -> S63 0.27 us today). The poll and its
// barrier stay ahead of the selection, unlike MPK_TOPK_FLAG_LATE.
// MEASURED NULL 2026-10-11 r1, text identical: 3.884 vs 3.881. Off.
#ifndef MPK_TOPK_FLAG_OVL
#define MPK_TOPK_FLAG_OVL 0
#endif
// MPK_WUK_REMAP: under MPK_WUK_LL + MPK_DEC_REMAP, W_UK tile t runs on the
// XCD's decode worker t (idle-polling for its query from the attention entry)
// instead of q_b worker t, which reaches it only after its own q_b tile.
// MPK_TRACE: the decode waits on the last W_UK tile, on the last q_b worker,
// 17.20 us vs 16.48 for the first; even idle W_UK workers return 1.7 us after
// the head's last q_b tile. The decode's preload walk follows the W_UK tile.
// MEASURED 2026-10-11, NP=8 1024/1024, text identical: 3.866 -> 4.005 ms
// (+0.139; with MPK_WUK_WFIRST 4.012). The decode workers' KV preload and
// prologue walk, issued at ~11 us today, slide behind the W_UK tile, and the
// decode waits on them. A local all-heads W_UK is no way round it: each decode
// item runs all 8 heads, one per XCD. Off.
#ifndef MPK_WUK_REMAP
#define MPK_WUK_REMAP 0
#endif
// MPK_WUK_LAT_REMAP: under MPK_WUK_LL, XCD 0's W_UK tile 0 (and its early
// weight pull) runs on the XCD's last worker -- an EP folder with nothing to
// do between the fold and o_proj -- instead of xcd_rank 0, which owns q_b's
// latent row and reaches Phase 3b only after the KV-cache append and its
// flag. MPK_TRACE on 12ad8d7: that one tile is the layer's last W_UK return
// in 277 of 292 layers, 16.80 us against 15.89 for the next XCD's, and the
// decode validates its query behind it (17.18 median). Same GEMV on the same
// words, so bit-identical.
// MEASURED 2026-10-12 on 12ad8d7, NP=8 1024/1024, n=3 interleaved, all 1024
// tokens identical on all 8 ranks: 3.689 3.684 3.705 -> 3.597 3.597 3.595 ms
// (-0.097). Default on.
#ifndef MPK_WUK_LAT_REMAP
#define MPK_WUK_LAT_REMAP 1
#endif
#if MPK_WUK_LAT_REMAP && MPK_WUK_REMAP
#error "MPK_WUK_LAT_REMAP moves one tile off the identity map MPK_WUK_REMAP replaces"
#endif
// MPK_WUK_IDLE_MAP: every W_UK tile on a worker with no q_b tile -- xcd_rank 0
// off XCD 0 and the EP folders (the XCD's last ranks) -- since even with its
// weight in L2 a W_UK tile on a q_b worker returns ~0.9 us later than on an
// idle one (MPK_TRACE on 12ad8d7: 2.4 vs 1.55 us after the head's last q_b
// tile). Five idle workers (four on XCD 0) hold 128-row tiles
// (GLM_WUK_GEMV_ROWS=128), not the 64-row default's eight. Supersedes
// MPK_WUK_LAT_REMAP.
// MEASURED 2026-10-12 on g16 (8498895), NP=8 1024/1024, n=3, control 3.591
// 3.595 3.592: 64-row tiles grid-striding over the five (two serial tiles on
// an idle worker), text identical, 3.725 3.723 3.729 (+0.133); 128-row tiles,
// one per idle worker, 3.570 3.572 -- but the 2-lane row reduction moves the
// tokens from the first one. With MPK_WUK_VL4 (the 64-row tiles' reduction
// order), all 1024 tokens identical on all 8 ranks, n=3: 3.563 3.568 3.562
// vs 3.593 3.595 3.598 (-0.031). Default on, with MPK_WUK_VL4 and demo.py's
// 128-row default under it.
// Re-measured 2026-10-14 on 4a21f5e, where q_b's return no longer drains
// (MPK_KVUPD_INLINE): the 64-row tiles back on ranks 0-7 (the q_b workers)
// with MPK_QB_LL_NOOUT, n=2 interleaved, tokens identical, 3.408 3.408 vs
// 3.383 3.381 ms (+0.026); without MPK_WUK_EARLY_PF too, 3.459 3.455.
#ifndef MPK_WUK_IDLE_MAP
#define MPK_WUK_IDLE_MAP 1
#endif
// MPK_QB_LL_NOOUT: under MPK_WUK_LL a q_b tile skips its write-through store
// of the nope scratch row: the W_UK tiles and the deferred rope read the
// XCD's epoch words instead, and nothing reads the row. The store's remote
// ack sat in wave 0's vmcnt ahead of the same worker's W_UK word poll (the
// GEMV stages its activation on wave 0), so a W_UK tile on a q_b worker
// returned ~0.9 us later than one on an idle worker (MPK_TRACE on 12ad8d7).
// MEASURED 2026-10-12 on g16, NP=8 1024/1024, n=3, text identical: 3.590
// 3.585 3.587 vs 3.595 3.596 3.602 (-0.011). Subsumed by MPK_WUK_IDLE_MAP,
// which takes the W_UK tiles off the q_b workers: together 3.566 3.568 vs
// 3.595 3.589, no better than the idle map alone. Off.
#ifndef MPK_QB_LL_NOOUT
#define MPK_QB_LL_NOOUT 0
#endif
#if MPK_WUK_IDLE_MAP && (MPK_WUK_REMAP || !MPK_EP_LL_FOLD_LAST)
#error "MPK_WUK_IDLE_MAP replaces the W_UK tile map and needs the folders last"
#endif
// MPK_WUK_VL4: a 2-lane-per-row MXFP8 GEMV row (W_UK's 128-row tiles) is
// computed in the 4-lane row's chunk split and sum order, so 128-row tiles
// give the 64-row tiles' bits (gang_gemv_mxfp8_mi300.cuh). Pairs with
// MPK_WUK_IDLE_MAP + GLM_WUK_GEMV_ROWS=128.
#ifndef MPK_WUK_VL4
#define MPK_WUK_VL4 1
#endif
// MPK_WUK_VL4_WF: MPK_WUK_WFIRST for the VL4 tile, whose six-trip K-loop
// WFIRST's one-trip limit excludes: the row's weights and scales load ahead
// of the q_b word poll. MPK_TRACE (stamps 78/79): words staged -> row sums
// 0.8 us, all of it after the poll; the last tile's sums -> the decode's
// query ready 1.9 us. Same loads, same arithmetic: bit-identical.
// MEASURED NULL 2026-10-13, n=3 interleaved, rank-0 tokens identical: 3.491
// 3.501 3.499 vs 3.505 3.504 3.498 ms (-0.005). MPK_TRACE: words staged ->
// row sums 0.73 -> 0.67 us. The step is the VALU chain -- 48 fp8 converts
// and packed FMAs a thread at one wave per SIMD -- not the weights' round
// trip. Off.
#ifndef MPK_WUK_VL4_WF
#define MPK_WUK_VL4_WF 0
#endif
#ifndef MPK_WUK_HOIST
#define MPK_WUK_HOIST 0
#endif

// MPK_W2_ACT_LL: W13 publishes its SwiGLU output as epoch-tagged words that
// the TP W2 tiles validate themselves, so the W2 workers skip the W13 -> W2
// wait (gang_oproj_router_fused_mi300.cuh, gang_moe_linear_mxfp8_mi300.cuh).
// MEASURED 2026-10-10, NP=8 1024/1024, n=3 interleaved on the 9ef34be +
// MPK_DEC_PRELOAD=2 default, text identical: 4.779 4.791 4.794 -> 4.728
// 4.722 4.722 ms (-0.064). Default on.
#ifndef MPK_W2_ACT_LL
#define MPK_W2_ACT_LL 1
#endif

// MPK_ATTN_OUT_LL: the split-KV merge publishes this rank's attention output
// as epoch-tagged words that the row-parallel o_proj validates while staging,
// so every worker skips the Phase 8 attention-release wait
// (gang_mla_full_layer_fused_mi300.cuh, merge_splitkv.cuh).
// MEASURED 2026-10-10, NP=8 1024/1024, n=3 interleaved on the 9ef34be +
// MPK_DEC_PRELOAD=2 + MPK_ATTN_PF_GROUPS_K=6 default, text identical:
// 4.762 4.764 4.759 -> 4.711 4.713 4.712 ms (-0.050). Default on.
#ifndef MPK_ATTN_OUT_LL
#define MPK_ATTN_OUT_LL 1
#endif

// MPK_OPROJ_WFIRST: the o_proj MXFP4 GEMV issues its first trip of weight
// loads ahead of the activation staging (gang_gemv_mxfp4_mi300.cuh).
// MEASURED NEGATIVE 2026-10-10, same batch as MPK_ATTN_OUT_LL, text
// identical: 4.762 4.764 4.759 -> 4.784 4.761 4.783 ms (+0.014). Off.
// Re-measured 2026-10-11 under MPK_ATTN_OUT_LL: 4.059 vs 4.031 (+0.028).
#ifndef MPK_OPROJ_WFIRST
#define MPK_OPROJ_WFIRST 0
#endif
// MPK_OPROJ_WFIRST_GATE: those trip-0 loads issued once MPK_LL_POLL_GATE has
// seen half the merge items land, not at entry (~17 us, beside the decode's
// KV and the merge's partial reads). MPK_TRACE: staged -> GEMV returned is
// 2.39 us with every weight load issued after the staging.
#ifndef MPK_OPROJ_WFIRST_GATE
#define MPK_OPROJ_WFIRST_GATE 0
#endif
// MPK_OPROJ_WALL_GATE: all 16 trips' weight and scale loads at that point (32
// vmem, ~80 VGPRs). The C++ K-loop issues trip 8-15's loads only as trip
// 0-7's land, so it pays two HBM round trips after the staging.
// MEASURED NEGATIVE 2026-10-11, NP=8 1024/1024, text identical: control
// 3.882 3.881; WFIRST_GATE 3.898 3.898 (+0.016); WALL_GATE 3.919 (+0.037).
// The more weight traffic issued beside the merge's tail, the worse -- the
// same barrier interference the Phase 8 o_proj prefetch note measures with a
// second round. Off.
#ifndef MPK_OPROJ_WALL_GATE
#define MPK_OPROJ_WALL_GATE 0
#endif

// MPK_TOPK_RANK: the per-XCD routing TopK selects by rank -- one expert per
// thread, counting the keys above its own in LDS -- instead of the 32-lane
// sort-then-merge network. Same keys, order and weights
// (moe_topk_sigmoid_bias_mi300.cuh).
// MEASURED NEGATIVE 2026-10-10, same batch as MPK_ATTN_OUT_LL, text
// identical: 4.762 4.764 -> 4.842 4.861 ms (+0.09). MPK_TRACE puts the
// network at 1.90 us (stamps 63 -> 42); 64 broadcast ds_read_b128 and 256
// compare-adds per thread cost more than its shuffle chain. Off.
#ifndef MPK_TOPK_RANK
#define MPK_TOPK_RANK 0
#endif

// The 2026-10-10 first-iteration wedge was folder 0 = xcd_rank 0 skipping the
// MPK_BAR_TAGGED_HIER entry poll that releases its XCD; xcd_rank 0 now always
// waits, so the skip pays only with MPK_EP_LL_FOLD_LAST (folders at the XCD's
// last ranks). =2 folds from the words but keeps the wait.
// MEASURED 2026-10-10 on the ea9e4d2 default, NP=8 1024/1024, n=3
// interleaved, text identical: 4.615 4.618 4.610 -> with FOLD_LAST:
// 4.565 4.572 4.567 (-0.046); + MPK_ENTRY_NOWAIT and the entry on
// Mechanism C (MPK_BAR_TAGGED=30): 4.471 4.458 4.465 (-0.149). FOLD_LAST
// alone: 4.624 4.644 4.631. All four default on.
// MPK_W2_PARTS_LL: the TP W2 also writes each K-part's f32 rows as
// epoch-tagged words (FULL_LAYER_W2PL_SLOT), and the next layer's EP_LL
// folders fold from those, arriving at the layer-entry rendezvous without
// waiting on it -- the last W2 tile's return path and the rendezvous leave
// the cross-rank critical path (gang_mla_full_layer_fused_mi300.cuh).
#ifndef MPK_W2_PARTS_LL
#define MPK_W2_PARTS_LL 1
#endif
// MPK_ENTRY_NOWAIT: under MPK_W2_PARTS_LL, every worker arrives at the
// layer-entry rendezvous without waiting on it (not the table's first layer
// nor the tail). With MPK_BAR_TAGGED's entry bit set the tagged entry still
// makes xcd_rank 0 wait (it holds its XCD's release), and xcd_rank 0 is a
// qkv_a resolver -- so the entry runs on Mechanism C (MPK_BAR_TAGGED=30).
#ifndef MPK_ENTRY_NOWAIT
#define MPK_ENTRY_NOWAIT 1
#endif
// MPK_DEC_HINTS: under MPK_DEC_PRELOAD the decode tiles hand the decode the
// indptr values their prologue walk already loaded, so its dependent
// qo_indptr -> kv_indptr loads leave the post-barrier path
// (gang_mla_attn_fused_mi300.cuh, gang_mla_decode_mi300.cuh).
#ifndef MPK_DEC_HINTS
#define MPK_DEC_HINTS 0
#endif
// MPK_DEC_MERGE_LL: the decode writes its split-KV partials as epoch words
// (o as bf16 pairs, same bytes as the f32 partials; lse as f32 words in
// FULL_LAYER_DLSE_SLOT) and the merge validates them, so the merge ranks
// skip the decode -> merge wait. Rounds the partials to bf16 before the
// merge, so it moves the tokens: ppl-gated.
#ifndef MPK_DEC_MERGE_LL
#define MPK_DEC_MERGE_LL 0
#endif
// MPK_DEC_LSE_LL: MPK_DEC_MERGE_LL's lse words only. The decode keeps its
// f32 partials, drains them, and only then writes each head's lse as an
// epoch word; the merge ranks skip the decode -> merge rendezvous, poll their
// lse words, and load the partials after -- so the merge waits on its own 64
// decode items, not a 232-tag relay, and its inputs are the same f32 values:
// bit-identical. MPK_ATTN_PROBE_NODECWAIT priced the rendezvous at -0.066 ms
// from a build with 2.6x the default's SGPR spills.
// MEASURED NEGATIVE 2026-10-11 on the g14 defaults, text identical (the
// merge reads its lse from LDS and keeps one copy of the arithmetic; a
// per-lane out-of-line copy contracted differently and moved the tokens):
// 3.754 -> 3.818 ms (+0.064). The partials' round trip now follows the
// words', and the merge ranks poll from their W_UK tile onward. Off.
#ifndef MPK_DEC_LSE_LL
#define MPK_DEC_LSE_LL 0
#endif
// MPK_DEC_MERGE_LL32: MPK_DEC_MERGE_LL with the partials kept f32. Each value
// leaves the decode as one (epoch << 32 | f32) word -- [chunk][own head][dim]
// in o_acc, [own head][chunk] in FULL_LAYER_DLSE_SLOT -- two to a write-through
// dwordx4, with no drain after it and no decode -> merge rendezvous at all.
// The merge derives the live chunks from the sequence length (an empty chunk
// writes nothing), lets wave 0 poll head 0's lse words until every live chunk
// has written, then re-reads its own words until they all carry the epoch and
// runs the halves path's arithmetic on the same f32 values: bit-identical.
// The gate is what MPK_DEC_MERGE_LL lacked: its merge ranks polled every word
// from right after W_UK.
// MEASURED SLOWER 2026-10-12 on 49d37f8, NP=8 1024/1024, r1, rank-0 tokens
// identical: 3.888 vs 3.506 ms (+0.38). MPK_TRACE: the gate passed at 22.57
// us against the stock barrier's 22.48 -- no earlier -- then validating the
// 2.2 MB of words took 3.7 us, and the decode's output step went 0.83 ->
// 1.69 us. The rendezvous was never the slow part. Off.
#ifndef MPK_DEC_MERGE_LL32
#define MPK_DEC_MERGE_LL32 0
#endif
#if MPK_DEC_MERGE_LL32 && (MPK_DEC_MERGE_LL || MPK_DEC_LSE_LL)
#error "MPK_DEC_MERGE_LL32 replaces MPK_DEC_MERGE_LL's and MPK_DEC_LSE_LL's words"
#endif
// MPK_OPROJ_SHIFT: under MPK_ATTN_OUT_LL, o_proj tile t runs on xcd_rank
// t + MPK_OPROJ_SHIFT, off the merge ranks (4 under head-local GLM:
// HL_MRG_ITEMS = 32 merge items over 8 XCDs), which reach o_proj last
// (gang_oproj_router_fused_mi300.cuh, gang_mla_full_layer_fused_mi300.cuh).
// MEASURED 2026-10-10, NP=8 1024/1024, interleaved, text identical:
// 4.437 4.421 4.432 -> 4.347 4.343 4.344 ms (-0.085). Default 4.
#ifndef MPK_OPROJ_SHIFT
#define MPK_OPROJ_SHIFT 4
#endif
// MPK_WUK_LL: q_b -> W_UK without the XCD-local W_UK barrier. q_b's head row
// also leaves as epoch words in the XCD's L2; the W_UK tiles stage their
// activation from those and the head's last tile ropes from them. Trace 5:
// W_UK barrier passed 1.65 us after the slowest q_b tile. MEASURED alone
// 2026-10-10: 4.420 -> 4.427 ms, null -- on only as MPK_DEC_Q_LL's carrier.
#ifndef MPK_WUK_LL
#define MPK_WUK_LL 1
#endif
// MPK_DEC_Q_LL (needs MPK_WUK_LL): q_b/W_UK -> decode without the q_b
// barrier. W_UK's rows and the roped tail also leave as epoch words, the
// latent tile flags the current token's cache row, and the decode stages its
// own heads' query from the words. Trace 5: the decode barrier passed 1.55 us
// after the last W_UK tile, then 0.7 us to the decode's first stamp.
// MEASURED 2026-10-10, NP=8 1024/1024, interleaved, text identical, with
// MPK_WUK_LL: 4.420 4.415 4.421 -> 4.376 4.365 4.370 ms (-0.049); with
// MPK_OPROJ_SHIFT=4 as well, 4.344 4.336 4.350 -> 4.284 4.287 (-0.06 more).
#ifndef MPK_DEC_Q_LL
#define MPK_DEC_Q_LL 1
#endif
// MPK_WUK_LL_NOOUT: under MPK_DEC_Q_LL the W_UK GEMV leaves its rows as the
// decode's query words only, not also as the bf16 query row: the decode
// stages the words and nothing else reads the row. That row was 128 two-byte
// write-through stores a tile -- partial-line read-modify-writes at the
// memory side -- in the window where the words land (gang_gemv_mxfp8_mi300).
// MEASURED NULL 2026-10-13, r1, rank-0 tokens identical: 3.510 vs 3.507
// ms. Off.
#ifndef MPK_WUK_LL_NOOUT
#define MPK_WUK_LL_NOOUT 0
#endif
// MPK_QXS_ROW_LL: the qkv_a split resolve's row exchange as epoch words in the
// XCD's L2 -- no drain, no per-tile flag, no flag poll before the row read.
// Trace 7: the resolve completes 4.8 us after the attention entry with the
// folds long done. MEASURED NEGATIVE 2026-10-11, NP=8 1024/1024, interleaved,
// text identical: 4.243 4.246 4.244 -> 4.278 4.274 4.264 ms (+0.028) -- every
// qkv_a tile polling 24 KB of row words costs more than the flag hop. Off.
#ifndef MPK_QXS_ROW_LL
#define MPK_QXS_ROW_LL 0
#endif
// MPK_OPROJ_SKIP_TOPK: under MPK_OPROJ_SHIFT the routing TopK worker holds no
// o_proj tile either (the tiles past it move up a rank), so its TopK does not
// drain o_proj stores (trace 6: selection 1.19 us, publish 0.78 us).
// MEASURED NEGATIVE 2026-10-10: 4.235 4.239 4.246 -> 4.246 4.252 4.245 ms. Off.
#ifndef MPK_OPROJ_SKIP_TOPK
#define MPK_OPROJ_SKIP_TOPK 0
#endif
// MPK_PROBE_XRANK_NOWAIT: CEILING PROBE, WRONG OUTPUT. Skip the validation
// of the two cross-rank exchanges' words and read whatever is there: 1 both,
// 2 the router's o_proj slots only, 3 the qkv_a resolver's EP slots only.
#ifndef MPK_PROBE_XRANK_NOWAIT
#define MPK_PROBE_XRANK_NOWAIT 0
#endif
// MPK_LL_POLL_GATE: the o_proj GEMV's MPK_ATTN_OUT_LL staging re-reads all
// 2048 words (16 KB) per round from its tile's entry (~19 us into the layer in
// trace 6) to the merge's last item (~30 us): ~30 MB/layer/rank of
// system-scope polling across the decode and the merge. Poll one word per
// merge item until half have landed, then the full rounds. MEASURED
// 2026-10-10, NP=8 1024/1024, interleaved, text identical: 4.287 4.279 4.283
// -> 4.261 4.258 4.267 ms (-0.021). Default on.
#ifndef MPK_LL_POLL_GATE
#define MPK_LL_POLL_GATE 1
#endif
// MPK_LL_POLL_GATE_N: merge items (of 32) the sentinel poll waits for before
// the full 16 KB rounds start; 192 tiles' rounds are ~3 MB each, run beside
// the merge's tail. MPK_OPROJ_WFIRST_GATE's 6 MB of weight loads at the same
// point measured +0.016 ms, so that tail is sensitive to the traffic.
// MEASURED 2026-10-11 r1 (control 3.881): 32 -> 3.919, 24 -> 3.892, 8 ->
// 3.872. Later full rounds cost more than their traffic saves: an item that
// lands while the gate holds is seen by its block-synchronized s_sleep(8)
// rounds, after it by its own thread's spin. With MPK_ROUTER_POLL_INC, n=3
// interleaved, text identical: control 3.885 3.882 3.878; ROUTER_POLL_INC
// 3.855 3.855 3.866; + GATE_N 8 3.850 3.842 3.841 (-0.038); GATE_N 4 alone
// 3.863 3.860 3.865. Value-independent (every word is validated before the
// staging either way); gated with ROUTER_POLL_INC in g13. Default 8.
#ifndef MPK_LL_POLL_GATE_N
#define MPK_LL_POLL_GATE_N 8
#endif
// MPK_LL_POLL_GATE_SLEEP: s_sleep between that sentinel poll's rounds. A
// round is 32 lanes x 16 B per tile, ~100 KB across the GPU.
#ifndef MPK_LL_POLL_GATE_SLEEP
#define MPK_LL_POLL_GATE_SLEEP 8
#endif
// MPK_OPROJ_POLL_INC: the o_proj GEMV's MPK_ATTN_OUT_LL staging, past the
// LL_POLL_GATE sentinel, re-reads only the words still on an older epoch --
// MPK_ROUTER_POLL_INC's shape -- instead of a thread's whole eight each round
// (gang_gemv_mxfp4_mi300.cuh): 192 tiles x 16 KB of system-scope reads a
// round beside the merge's tail. Same words staged: bit-identical.
// MEASURED NULL 2026-10-13, r1, rank-0 tokens identical: 3.512 vs 3.506
// 3.501 ms. Off.
#ifndef MPK_OPROJ_POLL_INC
#define MPK_OPROJ_POLL_INC 0
#endif
// MPK_OPROJ_PF_DEC: under the head-local decode (MPK_DEC_REMAP) a decode
// worker skips its own Phase 8 o_proj weight prefetch and the q_b worker
// dec_tiles ranks below it -- same XCD, same L2, at Phase 8 ~7 us earlier --
// issues that tile's beside its own (gang_mla_attn_fused_mi300.cuh
// oproj_pf_role, gang_mla_full_layer_fused_mi300.cuh Phase 8). MPK_TRACE:
// a decode worker's S24 -> S27 (issue 0.65 us, retire 0.76 us) is the gap
// between its decode item and its o_proj tile. Same weights: bit-identical.
// MEASURED SLOWER 2026-10-13, n=2 interleaved, rank-0 tokens identical:
// 5.113 5.106 vs 3.502 3.495 ms (+1.6 ms, ~20 us a layer). Far past what a
// moved prefetch can cost; the stall was not traced. Off.
#ifndef MPK_OPROJ_PF_DEC
#define MPK_OPROJ_PF_DEC 0
#endif
#if MPK_OPROJ_PF_DEC && (MPK_OPROJ_PF_IDLE || MPK_OPROJ_SKIP_TOPK)
#error "MPK_OPROJ_PF_DEC pairs ranks on the plain MPK_OPROJ_SHIFT map"
#endif
// MPK_ROUTER_POLL_INC: the router's o_proj-slot validation as a per-thread
// spin that re-reads only the slots still carrying an older epoch, instead
// of full pass -> sentinel spin -> full re-pass. MPK_TRACE: o_proj seen ->
// first pass 2.04 us, first pass -> validated 1.37 us; the re-pass is one
// more system-scope round trip after the last word lands.
// MEASURED 2026-10-11, NP=8 1024/1024, text identical: 3.858 vs 3.881 (r1),
// then n=3 interleaved 3.855 3.855 3.866 vs 3.885 3.882 3.878 (-0.023).
// Gates g13 (with LL_POLL_GATE_N): standalone PASS, ppl512 2.3670, longseq
// 256/512/1024 G1 PASS with token dumps identical to g12. Default on.
#ifndef MPK_ROUTER_POLL_INC
#define MPK_ROUTER_POLL_INC 1
#endif
// MPK_RESOLVE_POLL_INC: the same per-thread incremental spin for the qkv_a
// split resolver's EP slots (S70 -> S71 0.19 us in MPK_TRACE: the first pass
// usually lands, so this is the small end of the same fix).
// MEASURED 2026-10-11 r1, text identical: rpi+g4 3.836 -> 3.821; with
// DEC_Q_INC 3.797 -> 3.792. Default on with it (gates g14).
#ifndef MPK_RESOLVE_POLL_INC
#define MPK_RESOLVE_POLL_INC 1
#endif
// MPK_DEC_KVF_EARLY: the decode's cache-row flag read in the query poll's
// round (thread 0) instead of in its own poll after the query validates.
// MPK_TRACE: Q validated -> Q + flag ready 0.6 us, one system-scope round
// trip on a flag set ~4 us earlier.
// The ISA is as designed (flag load in the query round, its poll gone).
// MEASURED NULL 2026-10-11 r1, text identical: 3.883 vs 3.881. Off.
#ifndef MPK_DEC_KVF_EARLY
#define MPK_DEC_KVF_EARLY 0
#endif
// MPK_DEC_KVF_LAST: only the decode item whose chunk holds the current
// token's cache row waits on XCD 0's flag for it. The test compared the
// chunk's last tile index with ntiles after the split had narrowed ntiles to
// the chunk's own count, so it held for every chunk: all ~51 items polled the
// flag once their query words validated, one system-scope round trip
// (MPK_TRACE: S67 -> S47 0.69 us on every item; under MPK_PROBE_DEC_NOQ=2
// all 51 sat until the flag landed, ~15.3 us, not the one chunk). The other
// chunks' rows were all stored steps earlier. Fewer waits: bit-identical.
// MEASURED 2026-10-13 on 91c270e, NP=8 1024/1024, n=3 interleaved, rank-0
// tokens identical: 3.468 3.464 3.469 vs 3.472 3.464 3.476 ms (-0.004).
// MPK_TRACE: S67 -> S47 0.69 -> 0.56 us; the rest is the other threads'
// query words still landing, not the flag. Default on with MPK_DEC_KVF_INC
// and MPK_QB_DEC_PF (the three together: -0.024 ms, see MPK_QB_DEC_PF).
#ifndef MPK_DEC_KVF_LAST
#define MPK_DEC_KVF_LAST 1
#endif
// MPK_DEC_KVF_INC: under MPK_DEC_Q_INC, the items that wait on the cache-row
// flag read it in the query poll's rounds (thread 0) instead of in a round
// trip of their own after the words validate -- MPK_DEC_KVF_EARLY's idea,
// which as written also gave up MPK_DEC_Q_INC's per-thread spin.
// MEASURED 2026-10-13 on 91c270e with MPK_DEC_KVF_LAST, n=3 interleaved,
// rank-0 tokens identical: 3.461 3.467 3.460 vs 3.472 3.464 3.476 ms (-0.008).
// Default on (with MPK_QB_DEC_PF; see there).
#ifndef MPK_DEC_KVF_INC
#define MPK_DEC_KVF_INC 1
#endif
// MPK_DEC_Q_INC: the decode's query poll as a per-thread incremental spin
// (only words still on an older epoch re-read) instead of block-synchronized
// full rounds -- MPK_ROUTER_POLL_INC's shape on the W_UK -> decode handoff.
// A thread's nine words span the eight heads' W_UK tiles on eight XCDs, so
// once most heads land it re-reads one word, not nine: the old rounds put
// ~1.2 MB/round of system-scope reads beside the last heads' stores.
// MEASURED 2026-10-11, NP=8 1024/1024, text identical: r1 on rpi+g4 3.790 vs
// 3.836; on cdd3c7a 3.797 vs 3.845; with RESOLVE_POLL_INC 3.792, then n=3
// 3.789 3.783 vs 3.852 3.840 control. Default on (gates g14).
#ifndef MPK_DEC_Q_INC
#define MPK_DEC_Q_INC 1
#endif
// MPK_PROBE_DEC_NOQ: PRICING PROBE, WRONG OUTPUT. The decode takes its query
// words after one read, valid or not: the Q path's ceiling (q_b, W_UK and
// their two hand-offs) under MPK_DEC_Q_LL, where MPK_ATTN_PROBE_NOQBWAIT has
// no wait left to skip. =1 also skips the cache-row flag; =2 keeps it, so
// the decode starts on XCD 0's latent tile instead.
// MEASURED 2026-10-13, NP=8 1024/1024, n=2 interleaved against 3.505 3.507:
// =1 3.261 3.252 (-0.25 ms), =2 3.412 3.416 (-0.09). MPK_TRACE at =2: every
// stamp from the decode on moves ~1.9 us earlier (S47 17.29 -> 15.39, S69
// 26.79 -> 24.87), and the decode then waits on its own KV preload. The Q
// path is worth up to 0.09 ms with the latent tile's flag where it is
// (S66 ~14.4 us), 0.25 with the flag out of the way too.
#ifndef MPK_PROBE_DEC_NOQ
#define MPK_PROBE_DEC_NOQ 0
#endif
// MPK_XS_POLL_FULL: the qkv_a split resolver re-reads its whole slice of the
// eight EP slots each round instead of spinning on one sentinel per slot and
// then re-reading -- one system-scope round trip less after the last word
// lands, at ~12 KB per tile per round of polling traffic. Trace 5: the first
// resolve lands ~4 us after the rank's own last fold.
#ifndef MPK_XS_POLL_FULL
#define MPK_XS_POLL_FULL 0
#endif
// MPK_DEC_MERGE_LL's fault was the merge's LL half inlined into the attention
// task (64 asm-load outputs; then, with atomic loads, a build that faulted at
// the first task fetch). Out of line (merge_halves_ll_partials) it runs and
// matches, but MEASURED 4.259 -> 4.875 ms: all 32 merge items poll the same
// 512 lse words from right after W_UK, and the merge finishes 5.4 us later.
// MPK_DML_DBG: MPK_DEC_MERGE_LL bisection. 1: the decode writes its LL words
// and its normal partials, the merge keeps the barrier and the normal path --
// ran clean, text identical. 2: LL words only, the merge reads them but still
// waits at the barrier.
#ifndef MPK_DML_DBG
#define MPK_DML_DBG 0
#endif
// MPK_MOE_SKIP_TOPK: the per-XCD routing TopK worker runs no W13 or W2 tile.
// MPK_TRACE (trace 5) has the 8 TopK workers last at every MoE stamp: W13
// start +0.87 us, W13 done +1.70, W2 done +2.63 against the median worker.
// Alone it only exposes the folders' layer boundary (trace 6s: W2 tail
// max-med 4.3 -> 1.45 us, folders' S8 -> S0 3.3 us on the fold path), so it
// ships with MPK_ML_PTR_PREFETCH. MEASURED 2026-10-10, NP=8 1024/1024,
// interleaved, text identical: 4.276 4.290 4.287 -> skip 4.279 4.288 4.279,
// prefetch 4.274 4.275 4.270, both 4.266 4.256 4.259 ms (-0.024). Default on.
#ifndef MPK_MOE_SKIP_TOPK
#define MPK_MOE_SKIP_TOPK 1
#endif

// MPK_ML_NO_FENCE: the multi-layer loop skips its per-layer threadfence_gpu
// (persistent_kernel.cuh); the __syncthreads stays. MEASURED NEUTRAL
// 2026-10-10 on the five-lever default, tokens identical: 4.831 -> 4.829
// 4.822 ms. Re-measured 2026-10-11 on 1f81211 (MPK_ENTRY_NOARRIVE no longer
// drains every worker at the next layer's entry): the fence's one worker per
// XCD -- a resolver -- sat 3.3-3.6 us in it on its own W2 XGMI acks, the
// slowest on its XCD in 286 of 288 layers (MPK_TRACE S74 -> S12). Dropped,
// text identical: 3.949 3.957 3.945 vs 3.961 3.965 3.958, plus 3.944 vs
// 3.950 (-0.011). Small, because the acks are then waited for at that
// worker's next call. Default on.
#ifndef MPK_ML_NO_FENCE
#define MPK_ML_NO_FENCE 1
#endif

// Five levers from the MPK_TRACE profile, measured 2026-10-10 at NP=8
// 1024/1024 as one multi-arm batch against a shared control (two rounds,
// every arm's text identical to the control's): control 5.092 5.087 ms.
//
// MPK_DEC_REMAP: the head-local decode items on the workers past q_b's tiles,
// which skip Phase 3b's wait (gang_mla_attn_fused_mi300.cuh). The default
// ranks were also the W_UK workers, last to the q_b -> decode barrier, so
// META_PF's prologue walk had nothing to hide under. 5.011 4.989 (-0.090).
#ifndef MPK_DEC_REMAP
#define MPK_DEC_REMAP 1
#endif
// MPK_DEC_PRELOAD: the decode worker DMAs its item's first full KV tile (2:
// first two) into the decode's LDS tile buffers during the q_b -> decode
// wait, short of the current token's row. With MPK_DEC_REMAP: 4.904 4.901
// (-0.187); the MPK_TRACE decode item spent 1.72 us of 8.4 on that tile.
// 2 against 1 on the five-lever default, interleaved, tokens identical:
// 4.838 4.830 4.835 -> 4.789 4.783 4.799 4.802 ms (-0.042).
#ifndef MPK_DEC_PRELOAD
#define MPK_DEC_PRELOAD 2
#endif
// MPK_DEC_PRE_NOBAR: the decode's tile loop skips the barriers of a trip
// whose tile MPK_DEC_PRELOAD left in LDS (gang_mla_decode_mi300.cuh). The
// preloaded tiles sit in their own buffers, were retired by the barrier
// behind the query wait, and nothing writes them, so each trip's load barrier
// and closing barrier only made the four waves wait for each other twice a
// trip. Kept: a register-path trip's two (it loads into buffer 0) and the
// last trip's closing one. At 1025-2048 tokens every live item is two
// preloaded trips: 3 of 4 barriers go. Same arithmetic: bit-identical.
// MEASURED NULL 2026-10-13 with MPK_DEC_EXP2_BUILTIN, NP=8 1024/1024, r1,
// rank-0 tokens identical: 3.505 vs 3.507 ms. Off.
#ifndef MPK_DEC_PRE_NOBAR
#define MPK_DEC_PRE_NOBAR 0
#endif
// MPK_DEC_EXP2_BUILTIN: the decode softmax's exp2 as __builtin_amdgcn_exp2f
// (gang_mla_decode_mi300.cuh mla_exp2): speculatable, so the rescale's -inf
// select is a v_cndmask instead of an exec-mask branch. Bit-exact.
// MEASURED NULL with MPK_DEC_PRE_NOBAR, above. Off.
#ifndef MPK_DEC_EXP2_BUILTIN
#define MPK_DEC_EXP2_BUILTIN 0
#endif
// MPK_DEC_QK2T: when trip t + 1's tile is preloaded too (MPK_DEC_PRELOAD=2:
// every live item at 1025-2048 tokens), trip t's wave runs both tiles' QK
// chains interleaved -- a[32:35] and a[36:39], one K read of each per step --
// and trip t + 1 takes its scores from registers. MPK_TRACE: QK 0.41 us a
// trip, an 18-deep dependent MFMA chain; two side by side cost little more
// than one. Softmax and PV keep their order: bit-identical. Unlike
// MPK_MLA_QK_SPLIT there is no hand-off between waves.
// MEASURED SLOWER 2026-10-13 (r1, with MPK_DEC_PRE_NOBAR and
// MPK_DEC_EXP2_BUILTIN), tokens identical: 3.561 vs 3.506 ms. The second
// chain's accumulators spill -- the decode's scratch ops 8 -> 50 in the
// final ISA -- in the function that sets the megakernel's allocation. Off.
#ifndef MPK_DEC_QK2T
#define MPK_DEC_QK2T 0
#endif
// MPK_W13_ACT_EARLY: the W13 workers quantize the normed row into LDS while
// the TopK selects (gang_oproj_router_fused_mi300.cuh). 5.065 5.037 (-0.038).
#ifndef MPK_W13_ACT_EARLY
#define MPK_W13_ACT_EARLY 1
#endif
// MPK_W2_EARLY_PF: the TP W2 workers pull their tile's weight into L2
// across the W13 -> W2 wait (gang_oproj_router_fused_mi300.cuh). 5.062 5.068
// (-0.025).
#ifndef MPK_W2_EARLY_PF
#define MPK_W2_EARLY_PF 1
#endif
// MPK_W2_PF_MODE: which workers issue MPK_W2_EARLY_PF's pull. 0: every W2
// worker. 1: only the W13-idle ones -- a W13 worker issues it right behind
// its W13 tile, and its 67 KB of DMA then sits in the CU's memory pipe ahead
// of the staging poll (MPK_TRACE: staged +3.74 us after the last W13 tile vs
// +2.18 on W13-idle workers, the same 1.40 us k-loop after). 2: as 1, and
// each W13-idle W2 worker also pulls a share of the W13 workers' tiles.
// MEASURED 2026-10-11, NP=8 1024/1024, on MPK_WUK_EARLY_PF + MPK_ROPE_REMAP
// (control 4.132 4.128 ms), text identical: 1 -> 4.084 (-0.048); 2 -> 4.176
// (+0.044, the extra pull competes with W13's own stream). Default 1.
// 3 (mode 2's pull held until W13's words land) MEASURED 2026-10-12 on the
// MPK_WUK_IDLE_MAP defaults, r1: 3.759 vs 3.564 (+0.195).
#ifndef MPK_W2_PF_MODE
#define MPK_W2_PF_MODE 1
#endif
// MPK_W2_WPRE: under MPK_MOE_TP_W2_PF the TP W2 tile reads its segments'
// expert ids from the routing words at entry and issues its whole weight
// reduction into registers before the activation staging, instead of after
// it (the k-loop needed the staging's s_seg_off). MPK_TRACE (trace21): the
// W2 tiles on W13 workers -- the last W2 tile in 292/292 layers -- run a
// 2.04 us k-loop after staging, the HBM round trip of a weight stream
// issued only then; the L2-prefetched W13-idle tiles 1.40. Same loads, same
// MFMA order: bit-identical. =1: every wave preloads, so wave 0, which
// stages, retires its weight loads before its first poll pass (vmcnt is in
// order). =2: waves 1-3 also load wave 0's operands and leave them in LDS,
// so the staging polls have nothing older in flight.
// MEASURED 2026-10-12 on 40c2a2d, NP=8 1024/1024, n=3 interleaved, text
// identical: =1 3.551 3.552 3.551 vs 3.560 3.561 3.554 (-0.007); again
// 3.549 3.548 3.547 vs 3.557 3.559 3.553 (-0.008). =2 3.553 3.557 3.555
// (-0.001). MPK_TRACE (trace22/23): the k-loop after staging drops 2.04 ->
// 0.72 us, but the staging grows 1.96 -> 3.32: the 6.7 MB weight burst at W2
// entry slows the W13 -> W2 activation hand-off, which polls through the
// same memory path. With MPK_TOPK_2L_MERGE: 3.537 3.529 3.534 vs 3.549 3.552
// 3.552 (-0.018). Default 1.
#ifndef MPK_W2_WPRE
#define MPK_W2_WPRE 1
#endif
// MPK_W2_WPRE_BB: under MPK_W2_WPRE, all ten B operands are read from LDS
// before the MFMAs (one wait) instead of one pair per MFMA. Final ISA: one
// lgkmcnt(0), ten back-to-back MFMAs.
// MEASURED SLOWER 2026-10-12 with =1, n=3, text identical: 3.569 3.576 3.574
// vs 3.554 3.559 3.555 (+0.017; -0.007 without it).
#ifndef MPK_W2_WPRE_BB
#define MPK_W2_WPRE_BB 0
#endif
// MPK_W13_NO_ARRIVE: under MPK_W2_ACT_LL in the fused layer nothing waits on
// the W13 -> W2 rendezvous (the W2 tiles validate W13's words, and the
// release value is passed in rather than read back), so skip its store drain
// and arrival on every worker of the layer.
// MEASURED NULL 2026-10-11, same batch: 4.127 vs 4.132 4.128 ms. Off.
#ifndef MPK_W13_NO_ARRIVE
#define MPK_W13_NO_ARRIVE 0
#endif
// MPK_W13_BIAS_PRE: the K_HALF W13 tile's emitting lanes load their four bias
// values ahead of the k-loop instead of behind it, where the dependent
// global read sat between the last MFMA and the activation store.
// MEASURED 2026-10-11, same batch: 4.078 vs 4.132 4.128 ms (-0.054), text
// identical. Default on.
#ifndef MPK_W13_BIAS_PRE
#define MPK_W13_BIAS_PRE 1
#endif
// MPK_KS_SSQ_NT: the K-split router's per-tile sums of squares are written
// and read on one XCD only (its 16 router tiles and its TopK), so the two
// polls of them read through that XCD's L2 (`nt`) instead of at system scope.
// The partial-logit words beside them stay system scope: every XCD's TopK
// reads every XCD's.
// MEASURED NULL 2026-10-11, same batch: 4.139 vs 4.132 4.128 ms. Off.
// The sums of squares are system-scope stores, which bypass L2 on this
// multi-L2 part, so an `nt` poll can hit a line it cached before they landed
// (see MPK_TOPK_2L, which lost 0.106 ms to exactly that).
#ifndef MPK_KS_SSQ_NT
#define MPK_KS_SSQ_NT 0
#endif
// MPK_TOPK_2L: the K-split TopK in two levels. Each XCD's router tiles hold
// the whole logit of that XCD's 32 experts, so its TopK sums them through its
// own L2, keeps their top 8 and publishes those keys with their bf16 logits;
// every TopK then picks the top 8 of the 64 candidates. No XCD holds more
// than 8 of the winners, and the keys (24-bit biased score, 8-bit id) are
// unique, so the same 8 come out in the same order with the same weights:
// bit-identical. Replaces each TopK's system-scope gather of 4096 partials
// and the 256-key sort on one wave (MPK_TRACE on g16: gather 1.44 us after
// the last router store, selection 1.52 us). test_topk_2l: 400 seeded cases
// incl. forced ties, every output identical to the one-level TopK.
// MEASURED 2026-10-12 on g16, NP=8 1024/1024, n=3, text identical: =1
// (level-1 poll through L2, `nt`) 3.706 3.705 3.702 vs 3.595 3.596 3.602
// (+0.106). The router tiles' words are system-scope stores, which bypass L2
// here, so the poll kept hitting the line it had cached before they landed.
// =2 polls at system scope; =3 also reads the normed flags during the
// candidate rounds instead of after the sort; =4 is =3 with the router
// tiles' partials and sums of squares as plain stores read through the XCD's
// L2 -- only that XCD reads them once the TopK is two-level.
// With the branchless sort step (the first build's compare-select compiled
// to an exec branch per bitonic step: 1.0 us a sort), same batches, n=3:
// =2 3.617 (r1) vs 3.589; =1 3.630 (r1); =3 3.594 3.604 3.605 vs 3.596 3.593
// 3.591 (+0.008). On the MPK_WUK_IDLE_MAP defaults, n=3, text identical:
// =4 3.563 3.566 3.566, =4 + MPK_TOPK_2L_KEEP_LDS 3.554 3.561 3.558, vs
// 3.564 3.569 3.569 (-0.002 / -0.009): of the two-level build's -0.009, its
// 544-byte smaller static LDS, which moves every phase's dynamic base, gives
// back 0.007. Default =4 with MPK_TOPK_2L_KEEP_LDS.
#ifndef MPK_TOPK_2L
#define MPK_TOPK_2L 4
#endif
// MPK_TOPK_2L_KEEP_LDS: the two-level TopK still allocates the one-level
// one's 512-byte logit row, so the worker's static LDS (8208 B) and every
// phase's dynamic LDS base stay put. Prices whether the base move matters.
#ifndef MPK_TOPK_2L_KEEP_LDS
#define MPK_TOPK_2L_KEEP_LDS 1
#endif
// MPK_TOPK_2L_MERGE: the two-level TopK's level 2 takes the top 8 of the
// eight XCDs' sorted candidate lists with a three-level merge (max against
// the partner list reversed, then a 3-step clean: 12 compare steps, 2
// cross-row shuffles) instead of a full 64-lane bitonic sort (21 steps, 3
// cross-row). MPK_TRACE (trace21): S41 -> S4 0.52 us. Same winners, same
// order (test_topk_2l: 400 cases, dump identical to the one-level TopK).
// MEASURED 2026-10-12 on 40c2a2d, n=3, text identical: alone 3.566 3.566
// 3.553 vs 3.549 3.552 3.552 (+0.011); with MPK_W2_WPRE=1 3.537 3.529 3.534
// (-0.018, against -0.008 for MPK_W2_WPRE alone). Default on with it.
#ifndef MPK_TOPK_2L_MERGE
#define MPK_TOPK_2L_MERGE 1
#endif
// MPK_W13_SELF_TOPK: under MPK_TOPK_2L each routed W13 tile's wave 0 polls the
// eight XCDs' level-1 candidates and runs the level-2 merge itself, taking its
// slot's expert from it, instead of waiting for its XCD's TopK to merge, see
// the normed flags and publish the routing words (MPK_TRACE: candidates in ->
// W13 route seen ~0.9 us). The shared slot (and a dense layer's virtual
// experts) start without a poll. The TopK still publishes the words W2 reads.
// Same keys, same merge: bit-identical.
// MEASURED 2026-10-13 on 49d37f8, NP=8 1024/1024, n=2 interleaved, all 1024
// rank-0 tokens identical: 3.470 3.471 vs 3.503 3.506 ms (-0.034). ppl512
// 2.3670 on all 8 ranks (= default), longseq 256/512/1024 G1 PASS, standalone
// PASS. Defaulted, n=3 interleaved against =0, tokens identical: 3.480 3.478
// 3.471 vs 3.509 3.504 3.505 ms (-0.030). Default on.
#ifndef MPK_W13_SELF_TOPK
#define MPK_W13_SELF_TOPK 1
#endif
// MPK_W2_FOLD: TileRT's expert_down_allreduce shape for the one-row TP W2.
// A W2 tile is one 64-row block over all nine slots (12 per XCD instead of 24
// half-K tiles), on the 12 MoE workers that run no routed W13 tile: the ten
// W13-idle ones and the shared slot's two, whose W13 starts on the normed row
// without waiting for routing. Each DMAs its nine 8.5 KB weight blocks into
// LDS as soon as the routing words land, so after the SwiGLU words validate
// it only stages them and runs 18 MFMAs from LDS -- and it folds the result
// itself (the two halves' sums in the folder's order, the residual on the
// fold rank, bf16) and pushes the words to every rank's next-layer gather
// slot. The next layer's folders and the W13 workers' own W2 tiles are gone.
// MPK_TRACE (mtrace_c): the last W2 tile is always the last W13 worker's own,
// staged 4.1 us after the last W13 k-loop against 2.2 for a W13-idle worker,
// and the next layer's fold lands 1.2 us after the last W2 tile.
// Bit-identical: same staging, same per-half MFMA order, same fold.
// MEASURED SLOWER 2026-10-13 on 49d37f8, NP=8 1024/1024, interleaved, all
// 1024 rank-0 tokens identical: =1 (LDS) 3.637 3.640 vs 3.497, and 3.598 vs
// 3.511; =2 (registers) 3.537 3.537 3.542 vs 3.511 3.500 3.495 (+0.037);
// MPK_W2F_RT_OFF 3.529 3.535 (+0.035) -- the path itself is neutral, the
// build pays for having it. MPK_TRACE_SPARSE: W13's last k-loop ends 0.5 us
// later (the 7 MB weight pull lands in W13's stream), and the shared slot's
// two workers per XCD stage 1.4 us behind the W13-idle ones (their SwiGLU
// stores retire, then the routing poll, before their weight fetch). With
// MPK_W2_ACT_SENT 3.534, 3.541 3.538; with the routing words polled as whole
// rounds 3.528 3.530 vs 3.499 3.502; on the MPK_W13_SELF_TOPK default with
// MPK_MOE_W13_INLINE + MPK_W13_NO_ARRIVE + MPK_W13_ACT_W3 3.514 3.513 vs
// 3.476 3.478. Off.
#ifndef MPK_W2_FOLD
#define MPK_W2_FOLD 0
#endif
// MPK_MOE_DPP: the MoE chain's intra-wave exchanges as DPP and gfx950
// register swaps instead of ds_bpermute -- the K-split router tile's sum of
// squares (lane 0's tree) and gate-dot reduction, the two cross-row steps of
// the TopK merge (run by every W13 tile under MPK_W13_SELF_TOPK), and the TP
// W2 staging's per-quad scale max. Same partners, same order: bit-identical
// (m_dpp_test: all four exchanges match their shuffle forms on 4096 random
// waves). gang_rmsnorm_linear_bias_mi300.cuh's MPK_RMSNORM_DPP helper is not
// usable here: its row_shr steps leave lane 0 outside its row's sum.
// MEASURED 2026-10-13 on the MPK_W13_SELF_TOPK default, NP=8 1024/1024, n=3
// interleaved, all 1024 rank-0 tokens identical: 3.471 3.462 3.471 vs 3.476
// 3.476 3.475 (-0.008). Defaulted, n=3 against =0: 3.483 3.465 3.468 vs
// 3.473 3.474 3.477 (-0.003); pooled -0.005, near the noise. ppl512 2.3670
// on all 8 ranks, longseq 256/512/1024 G1 PASS, standalone PASS. Default on.
#ifndef MPK_MOE_DPP
#define MPK_MOE_DPP 1
#endif
// MPK_W2_SELF_IDS: under MPK_W13_SELF_TOPK a W13 tile's merge leaves all
// nine slot ids in LDS, and the same worker's W2 tile -- the last W2 tile to
// stage in every layer -- takes its segments' ids from there instead of
// reading the routing words (one L2 round trip, ahead of its weight preload
// and so of its first staging poll). Same ids: bit-identical.
// MEASURED SLOWER 2026-10-13 on the MPK_W13_SELF_TOPK default, n=3
// interleaved, text identical: 3.488 3.484 3.478 vs 3.471 3.473 3.475
// (+0.010): the preload burst lands earlier in W13's tail. Off.
#ifndef MPK_W2_SELF_IDS
#define MPK_W2_SELF_IDS 0
#endif
// MPK_W2_WPRE_LATE: the W2 tiles on W13 workers -- each issued right behind
// its worker's W13 tile, while the last W13 tiles still stream -- skip
// MPK_W2_WPRE and load their weights after the staging, so the 144 x 40 KB
// burst no longer lands in the W13 tail (MPK_W2_SELF_IDS, which issues it
// one L2 round trip earlier, measured +0.017). The W13-idle tiles keep the
// preload. Same loads, same MFMA order: bit-identical.
// MEASURED SLOWER 2026-10-13, r1, text identical: 3.656 vs 3.471-3.476
// (+0.18) -- without the preload the tile's weight stream starts after the
// staging, in the depth-4 k-loop's dependent trips. Off.
#ifndef MPK_W2_WPRE_LATE
#define MPK_W2_WPRE_LATE 0
#endif
// MPK_ROUTE_BATCH: the TP W2's routing-word reads (MPK_W2_EARLY_PF's pull and
// MPK_W2_WPRE's preload) poll all nine words as whole rounds. A W13-idle
// worker gets there before the TopK publishes, finds every word stale, and
// re-polled them one round trip at a time. Timing only: bit-identical.
// MEASURED SLOWER 2026-10-13 on 49d37f8, n=2 interleaved, text identical:
// 3.549 3.536 vs 3.499 3.505 (+0.04); with MPK_W2_PF_MODE=2 3.608 (r1). The
// pulls start earlier, inside W13's stream. Off.
#ifndef MPK_ROUTE_BATCH
#define MPK_ROUTE_BATCH 0
#endif
// MPK_W2F_RT_OFF: PROBE, correct output. MPK_W2_FOLD's code compiled in and
// never taken (the old W2 and fold run), to price its effect on the build.
// MEASURED 2026-10-13: +0.035, see MPK_W2_FOLD.
#ifndef MPK_W2F_RT_OFF
#define MPK_W2F_RT_OFF 0
#endif
// MPK_ROUTER_W_LATE: the K-split router tile's gate-slice and gamma loads
// move behind its o_proj slice validation, and waves 2-3 warm L2 with the same
// bytes at entry instead. Issued up front on every wave, they sat in the
// validating waves' vmcnt ahead of the first pass (MPK_TRACE: S33 -> first
// pass 2.05 us against a ~1 us local round trip).
// It wedged at decode iteration 2 until the validation compares were tied
// below their vmcnt(0) (the scheduler had hoisted them onto words still in
// flight). Since then it runs, text identical, and MEASURED SLOWER 2026-10-11:
// 4.027 vs 4.000 ms. Off.
#ifndef MPK_ROUTER_W_LATE
#define MPK_ROUTER_W_LATE 0
#endif
// MPK_WUK_WFIRST: a one-trip MXFP8 GEMV with LL activation input (W_UK) loads
// its weights and scales ahead of the activation poll and consumes them from
// registers, so the trip's L2 round trip overlaps the poll instead of
// following the staging (MPK_TRACE: the last q_b worker's W_UK GEMV returns
// 2.20 us after its q_b tile, 0.8 us behind early workers).
// MEASURED NULL 2026-10-11, NP=8 1024/1024: 4.041 vs 4.040 ms. Off.
#ifndef MPK_WUK_WFIRST
#define MPK_WUK_WFIRST 0
#endif
// MPK_RESOLVE_NW_LATE: the split resolve loads its norm weight after the EP
// slot validation instead of ahead of it, where the 12 KB (a fresh layer's,
// so not in L2) sat in the validating waves' vmcnt ahead of the first pass.
// It is first used at the gather, behind the XCD exchange.
// MEASURED NULL 2026-10-11, NP=8 1024/1024: 4.040 vs 4.040 ms. Off.
#ifndef MPK_RESOLVE_NW_LATE
#define MPK_RESOLVE_NW_LATE 0
#endif
// MPK_QXS_EARLY_RES: the split resolve's slice sums run on the XCD's last
// MPK_QXS_EARLY_RES_N workers -- past the 24 qkv_a tiles (the kv_a half is
// padded 576 -> 1024 rows), so with no qkv_a tile; at 4 they are the EP
// folders -- instead of on qkv_a tiles 0-15. Those tiles are W13 + W2
// workers: MPK_TRACE has the last one leave its W2 tile 3.4 us before it
// reaches the resolve, while the peers' fold words land ~1.5 us before its
// first pass returns. The tiles keep only the gather. Same slot order and
// rounding: the row is identical.
// MEASURED NEGATIVE 2026-10-11 on the g14 defaults, text identical: 3.877 vs
// 3.750 ms (+0.127). MPK_TRACE: the resolve validates no earlier (last
// 3.04 vs 3.12 us) -- it is bound by the peers' fold words landing, not by
// the resolvers reaching it -- while the folders, now resolving before their
// MPK_QKVA_IDLE_PF share, arrive last at qkv_a -> q_b (+1.58 us) and the late
// prefetch slows the qkv_a K-loop (+0.53 us). Off.
#ifndef MPK_QXS_EARLY_RES
#define MPK_QXS_EARLY_RES 0
#endif
#ifndef MPK_QXS_EARLY_RES_N
#define MPK_QXS_EARLY_RES_N 4
#endif
// MPK_QA_LL: qkv_a -> q_b through epoch words. The qkv_a tiles also write
// their q_a rows as (epoch << 32 | bf16x2) words, write-through (the q_b
// tiles are on every XCD), into the MPK_QXS_ROW_LL region; the q_b GEMM tiles
// validate those words per thread instead of waiting at the gang-wide
// qkv_a -> q_b rendezvous, which only xcd_rank 0 (the latent tile, which
// reads kv_a) still waits on. The norm's traversal is unchanged, so q_b's
// input and output are bit-identical. MPK_TRACE under the no-wait probe
// prices the rendezvous at ~1.1 us of the chain (S50), which that probe's
// build then lost to 2.4x the SGPR spills.
// MEASURED 2026-10-11 on the g14 defaults, NP=8 1024/1024, n=3 interleaved,
// text identical, SGPR spills 300 vs 298: 3.747 3.753 3.754 -> 3.700 3.706
// 3.691 ms (-0.052). Default on.
#ifndef MPK_QA_LL
#define MPK_QA_LL 1
#endif
// MPK_OPROJ_A_PAD: the o_proj GEMV stages its activation row with 16 bytes
// after every 32-element chunk (gang_gemv_mxfp4_mi300.cuh). Its 64
// ds_read_b128 a thread (256 a CU, 1 KB returned each) read eight
// consecutive chunks per wave, 64 bytes apart -- banks 0/16/32/48 twice --
// beside 256 cvt + 256 dot2 per wave; 80 bytes apart the eight hit disjoint
// banks. Same values, same order: bit-identical.
// MEASURED NULL 2026-10-12 on 12ad8d7, NP=8 1024/1024, n=3 interleaved, all
// 1024 tokens identical: 3.701 3.691 3.689 -> 3.686 3.683 3.695 ms (-0.006,
// inside the spread). Off.
#ifndef MPK_OPROJ_A_PAD
#define MPK_OPROJ_A_PAD 0
#endif
// MPK_PROBE_W13_HALFK: PRICING PROBE, WRONG OUTPUT. The K_HALF W13 tile's
// waves run half their k-tiles: how much of a faster W13 stream the wall
// keeps.
#ifndef MPK_PROBE_W13_HALFK
#define MPK_PROBE_W13_HALFK 0
#endif
// MPK_PROBE_W13_NOSHARED: PRICING PROBE, WRONG OUTPUT. The last slot's K_HALF
// W13 tiles -- the shared expert's under MPK_MOE_TP's fixed slots -- skip their
// k-loop and emit zeros, so the W13 round streams 8 of its 9 slots: the most
// that taking the shared expert's W13 off the round could give.
// MEASURED NULL 2026-10-17 on d355390, NP=8 1024/1024, n=2 interleaved, G1
// PASS, control tokens identical to the default: 3.376 3.374 vs 3.365 3.370
// ms (+0.008). With 16 of the round's 144 tiles idle the other 128 finish no
// sooner, so each tile's own stream (~32 KB in flight per CU) sets the round,
// not the aggregate HBM rate -- MPK_PROBE_W13_HALFK, which shortens every
// tile, is -0.087. Running the shared expert's tiles early is not worth it.
#ifndef MPK_PROBE_W13_NOSHARED
#define MPK_PROBE_W13_NOSHARED 0
#endif
// MPK_PROBE_OPROJ_HALFK: PRICING PROBE, WRONG OUTPUT. The MXFP4 o_proj GEMV
// runs half its K-loop: how much of a faster o_proj GEMV the wall keeps.
#ifndef MPK_PROBE_OPROJ_HALFK
#define MPK_PROBE_OPROJ_HALFK 0
#endif
// MPK_PROBE_QKVA_HALFK: PRICING PROBE, WRONG OUTPUT. qkv_a's K-parallel tile
// runs half its k-tiles per wave (6 of 12; the hoisted prefill still seeds
// the first four): how much of a faster qkv_a weight stream the wall keeps.
// Asked before building a qkv_a tile that DMAs its 99 KB slab into LDS while
// it waits on the XSPLIT exchange: final ISA has the quantizer's closing
// barrier behind a vmcnt(9) on the hoisted prefill's first k-tile, so S54 ->
// S62 (1.13 us) is that load's L2 round trip, and the K-loop (1.64 us) is
// two more -- the slab would delete both.
// MEASURED 2026-10-15 on 792d601, NP=8 1024/1024, n=2 interleaved, control
// tokens identical to the default: 3.364 3.362 vs 3.361 3.379 ms (-0.007,
// noise). MPK_TRACE, n=1 each: S61 max -0.71 us, and every later stamp of the
// layer -0.5..-0.7 (period -0.45 us); traced wall 3.536 -> 3.515. The dense
// trace's stamp stores make qkv_a critical; untraced, the wall keeps ~0.1 us
// of a 0.7 us faster tile. The LDS slab is not worth building on this.
#ifndef MPK_PROBE_QKVA_HALFK
#define MPK_PROBE_QKVA_HALFK 0
#endif
// MPK_OPROJ_PF_MIN_US: the Phase 8 o_proj weight prefetch waits until this
// many us past the worker's layer entry. Under MPK_QKVA_SKIP_PAD four workers
// on XCD 6 (ranks 20-23, decode-idle at this context) reach Phase 8 ~3.8 us
// in and run their GEMV ~24 us later: MPK_TRACE has them at 3.04-3.08 us a
// tile against 2.44 for everyone else, every layer, and the slowest o_proj
// tile is the one the router waits for on every rank. Before the skip they
// reached Phase 8 at ~15 us and ran 2.32. 0 is off.
// MEASURED 2026-10-12 on a87dc19: 14 won two clean standalone pairs by
// 0.012 ms, and combined with MPK_ROUTER_L2PF won three more runs at
// 3.496/3.501/3.492 ms versus 3.520/3.509 ms controls. The combined arm was
// token-identical to control and passed PPL (2.3670) and 256/512/1024 longseq.
#ifndef MPK_OPROJ_PF_MIN_US
#define MPK_OPROJ_PF_MIN_US 14
#endif
// MPK_W2_ACT_SENT: a W2 staging thread whose full pass over its 16 SwiGLU
// words misses spins on two of them (one per producing W13 tile) before the
// next full pass. The W13-idle W2 workers poll from the route onward, ~5 us
// beside W13's HBM-bound K-loop, eight dwordx4 a thread a round; the late
// W2 tiles that set the phase's makespan still take their first pass whole.
// MEASURED NULL 2026-10-12 on 12ad8d7, n=3, text identical: 3.703 3.688
// 3.690 -> 3.697 3.689 3.692 ms. Off.
#ifndef MPK_W2_ACT_SENT
#define MPK_W2_ACT_SENT 0
#endif
// MPK_QKVA_SKIP_PAD: qkv_a's row is [q_a | latent | padding] with the latent
// padded 576 -> 1024 for the output alignment, and nothing reads the padding.
// At GLM-5.2 that is 28 of 192 tiles (XCD 7's 24 and XCD 6's last 4), ~15%
// of qkv_a's weight stream, plus the same bytes in MPK_QKVA_IDLE_PF's L2
// prefetch -- all of it zeros, fetched while the real tiles fetch theirs.
// Skip those tiles and prefetch only the live ones.
// MEASURED 2026-10-11 on the g14 defaults, NP=8 1024/1024, n=3 interleaved,
// text identical: 3.747 3.753 3.754 -> 3.734 3.746 3.733 ms (-0.013).
// Default on.
#ifndef MPK_QKVA_SKIP_PAD
#define MPK_QKVA_SKIP_PAD 1
#endif
// MPK_QXS_XOUT_LATE: the split resolver's write-through x_out store goes out
// after the norm instead of ahead of its exchange flag, so the flag's drain
// waits on the L2 row stores only. Its ack then retires under qkv_a's first
// weight wait, an HBM fetch; the qkv_a -> q_b drain orders it for o_proj.
// MEASURED NULL 2026-10-11, NP=8 1024/1024: 4.043 vs 4.037 ms. Off.
// Re-measured on 12ad8d7 (MPK_QA_LL), n=3: 3.698 3.701 3.688 vs 3.701 3.691
// 3.689, null.
#ifndef MPK_QXS_XOUT_LATE
#define MPK_QXS_XOUT_LATE 0
#endif
// MPK_TOPK_ID_EARLY: under MPK_ROUTE_LL the sort-merge TopK writes the eight
// expert-id words (and the shared slot's) as soon as the merge settles, one
// lane each; the W13 tiles start on the id alone, and the weight words follow
// with the renormalisation as before (MPK_TRACE: selection done -> routing
// published 0.6 us, behind a vmcnt(0) drain and the logit clears).
// MEASURED 2026-10-11, NP=8 1024/1024, interleaved, text identical:
// 4.040 4.036 4.035 -> 3.997 3.998 4.002 ms (-0.038). Default on.
#ifndef MPK_TOPK_ID_EARLY
#define MPK_TOPK_ID_EARLY 1
#endif
// MPK_MOE_XCD_EXPERT: under MPK_MOE_TP with 8 routed slots on 8 XCDs, XCD x
// runs routed slot x's whole W13 (16 tiles) and its whole W2 (96 row blocks
// over 16 workers, 6 each, K = one slot), so the W13 -> W2 activation hand-off
// stays in that XCD's L2. The shared slot's W13 is two tiles per XCD and its
// W2 the remaining 16 x 6 row blocks, read across XCDs. W2 writes one f32
// plane per slot and the EP fold sums the nine in slot order. Layers on the
// striped W2 (the dense prologue, and the last layer, whose output the tail
// folds from the workspace) write zeros into the planes they do not fill.
// MEASURED NEGATIVE 2026-10-11, NP=8 1024/1024, untraced, text coherent but
// moved (W2 sum order): control 4.000 / 3.993 ms; alone 4.031, + TOPK_SH_EARLY
// 4.043, + XE_W2_PF 4.016 / 4.021, + W13_NO_ARRIVE 4.012. MPK_TRACE runs show
// the opposite (middle layers -1.8 us each, traced decode 4.258 -> 4.191), so
// the stamps hide whatever costs it; suspects are the fold now waiting on
// nine XCDs' W2 per column instead of two parts, and W13 + W2 serial on the
// same 144 workers (per XCD the routed W2 stages 2.3-2.8 us after its last
// W13 tile, not the ~1 us an L2 hand-off should take). Off.
#ifndef MPK_MOE_XCD_EXPERT
#define MPK_MOE_XCD_EXPERT 0
#endif
// MPK_TOPK_SH_EARLY: under MPK_ROUTE_LL the shared slot's id and weight words
// (always NUM_EXPERTS at 1.0) go out as soon as the TopK has seen the normed
// flags, ahead of the selection, so the shared slot's W13 tiles start on the
// normed row instead of on the routing. Only pays under MPK_MOE_XCD_EXPERT,
// where it measured +0.012 ms on top of it (4.031 -> 4.043). Alone on the TP
// default 2026-10-11 (7bc1c7e): 3.949 vs 3.950 ms (r1), null. Off.
#ifndef MPK_TOPK_SH_EARLY
#define MPK_TOPK_SH_EARLY 0
#endif
// MPK_XE_W2_PF: under MPK_MOE_XCD_EXPERT the six W13-idle MoE workers of an
// XCD pull that XCD's routed W2 row blocks (16 each) into L2 while W13 runs,
// so the W2 workers' loads hit L2. Waves 1-3, never-read LDS window.
// MEASURED 2026-10-11 on MPK_MOE_XCD_EXPERT + MPK_TOPK_SH_EARLY: 4.043 ->
// 4.016 ms (-0.027), still behind the striped control. Off with it.
#ifndef MPK_XE_W2_PF
#define MPK_XE_W2_PF 0
#endif
// MPK_RNLM8_BIAS_PRE: the MXFP8 RMSNorm+linear tile (qkv_a, q_b) loads its
// epilogue lanes' four bias values ahead of the k-loop instead of between the
// last MFMA (or the cross-wave reduction) and the output store, the same
// dependent read MPK_W13_BIAS_PRE took out of W13.
// MEASURED NULL 2026-10-11 on 5df165e: 4.011 vs 4.000 ms. Off.
// Re-measured 2026-10-14 on dfa7148 (q_b's chain now sets the decode's
// query): 3.480 vs 3.443 ms (+0.037, r1, tokens identical). See
// MPK_QB_BIAS_PRE.
#ifndef MPK_RNLM8_BIAS_PRE
#define MPK_RNLM8_BIAS_PRE 0
#endif
// MPK_QB_BIAS_PRE: MPK_RNLM8_BIAS_PRE for q_b's tile alone. Re-measured on
// dfa7148 the shared flag was slower (bp_r1 3.480 vs 3.443): ahead of qkv_a's
// k-loop the read is older than every weight load the loop then waits on, and
// vmcnt retires in order. In q_b's tile it is issued after its four A tiles,
// and MPK_TRACE (q05) puts the epilogue's own read at 0.27 us between the
// cross-wave sum and the words.
// MEASURED NULL 2026-10-14, n=3 interleaved, rank-0 tokens identical: on
// MPK_KVUPD_INLINE + MPK_QB_NORM_DPP 3.429 3.418 3.425 vs 3.412 3.416 3.420
// ms (+0.008); on the MPK_KV_LL defaults 3.402 3.395 vs 3.389 3.390 3.387
// (+0.010). MPK_TRACE (q05): the epilogue's read 0.27 -> 0.16 us, given back
// elsewhere. Off.
#ifndef MPK_QB_BIAS_PRE
#define MPK_QB_BIAS_PRE 0
#endif
// MPK_GEMV4_EPI_PRE: the MXFP4 GEMV's epilogue lanes (o_proj) load their
// residual and bias ahead of the activation staging instead of after the
// k-loop's shuffle reduction. Under MPK_OPROJ_RP the residual rides rank 0's
// partial only, so this is a dependent read on one rank's push that every
// rank's router waits for.
// MEASURED NULL 2026-10-11 on 5df165e, n=3: 3.995 3.997 4.003 vs 4.006 3.994
// 3.998 ms, text identical. (One earlier r1 wedged rank 6 in iteration 1, the
// box's random first-iteration wedge; the rerun was clean.) Off.
#ifndef MPK_GEMV4_EPI_PRE
#define MPK_GEMV4_EPI_PRE 0
#endif
// MPK_FOLD_RES_PRE: the W2-parts EP fold loads its residual pair (fold PE
// only) ahead of the part-word poll instead of after it, where the poll's
// memory clobber kept the read behind the last part's arrival; every rank's
// next resolve waits for that rank's push.
// DO NOT ENABLE: changes the generated text (2026-10-11, 3.997 vs 4.006 ms).
// The residual row is not final when the fold PE reaches the poll -- the
// router's slice sum writes it -- so the early read can see the last layer's.
#ifndef MPK_FOLD_RES_PRE
#define MPK_FOLD_RES_PRE 0
#endif
// MPK_ROUTER_NO_INV: the K-split, LL-summing router tile skips the buffer_inv
// ahead of its slice validation. Everything that path reads afterwards is a
// system-scope load (the o_proj words, the sentinels, the sums of squares) or
// was issued before it (the gate slice), so the L1 drop protects nothing --
// and MPK_TRACE shows the tile 0.9 us between leaving o_proj and S33 with no
// other instruction there that can wait.
#ifndef MPK_ROUTER_NO_INV
#define MPK_ROUTER_NO_INV 0
#endif
// MPK_ROUTER_KS_EARLY: a K-split router tile's gate and gamma slices are
// loaded when its worker enters o_proj (S29, ~18 us ahead) and handed to the
// router, instead of at the router's head. There all 128 tiles of a GPU ask
// for their 24 KB at once, and the slice validation's loads retire behind
// that burst: S33 -> first pass is 1.9-2.4 us however late the tile starts.
// DO NOT ENABLE: wedges in decode iteration 1 (rank 1 alone finishes it), 6/6
// attempts, with and without the validation-compare tie. The addressing
// matches the router's own loads, the by-value copy is stored behind a
// vmcnt(0), and no straight-line asm-output hazard shows; cause unknown.
#ifndef MPK_ROUTER_KS_EARLY
#define MPK_ROUTER_KS_EARLY 0
#endif
// MPK_OLL2_LDS: under MPK_OPROJ_LL == 2 the o_proj GEMV's epilogue lanes also
// leave their row's f32 in LDS, and wave 3 builds the push words from there
// instead of draining every wave's L2 stores and reading the rows back. Both
// hops sit between the last o_proj tile and every rank's router.
// MEASURED 2026-10-11 on 5df165e, n=3: 3.985 3.985 3.984 vs 3.996 3.996 3.998
// ms (-0.012), text identical.
#ifndef MPK_OLL2_LDS
#define MPK_OLL2_LDS 1
#endif
// MPK_OLL2_NOST: under MPK_OLL2_LDS nothing reads the XCD's L2 row block any
// more (the push builds its words from LDS), so the GEMV's epilogue lanes skip
// that plain f32 store, which the GEMV's return drained (vmcnt(0)) between
// the last o_proj FMA and wave 3's push. Same values pushed: bit-identical.
// MEASURED 2026-10-14 on 98f581f, NP=8 1024/1024, n=3 interleaved, all 1024
// rank-0 tokens identical: 3.376 3.372 3.383 vs 3.383 3.393 3.381 ms
// (-0.007, inside the run-to-run spread); test_oproj_lds 2.40 -> 2.36 us a
// tile. Default off.
#ifndef MPK_OLL2_NOST
#define MPK_OLL2_NOST 0
#endif
// MPK_OPROJ_PF_LDS: the Phase 8 o_proj weight prefetch lands the tile itself
// in LDS (68 KB at MPK_OPROJ_W_LDS_OFF, rows XOR-swizzled by 16 B so a wave's
// eight rows hit distinct banks) instead of only warming L2 through a
// never-read 4 KB window, and the MXFP4 GEMV reads its weights and scales
// from LDS (gang_gemv_mxfp4_mi300.cuh, gang_mla_full_layer_fused_mi300.cuh).
// MPK_TRACE: the GEMV is 2.40 us a tile with every weight an L2 hit, two
// batches of eight dwordx4 behind the staging. Same bytes, same FMA order:
// bit-identical.
// MEASURED 2026-10-14 on 98f581f, tests/standalone/test_oproj_lds.hip (192
// tiles, L2-warm weights): 2.48 us a tile from LDS vs 2.40 from L2. The GEMV
// is bound by its activation reads and v_cvt_scalef32_pk_bf16_fp4 (no
// activation reads 2.08, no cvt 2.16, neither 1.68), not by its weight
// loads; from LDS the weights queue behind those 64 ds_read_b128 a lane.
// Default off; not run at NP=8.
#ifndef MPK_OPROJ_PF_LDS
#define MPK_OPROJ_PF_LDS 0
#endif
#if MPK_OPROJ_PF_LDS && (MPK_OPROJ_PF_IDLE || MPK_OPROJ_PF_DEC || \
                         MPK_OPROJ_SKIP_TOPK ||                     \
                         defined(MPK_GLM_OPROJ_PREFETCH_OFF))
#error "MPK_OPROJ_PF_LDS: each worker prefetches the o_proj tile it runs"
#endif
// Past the GEMV's activation staging and LDS_OUT rows, the Phase 8 prefetch
// window (64 KB) and the router's (68 KB).
#define MPK_OPROJ_W_LDS_OFF (72 * 1024)
// Force-inline the task kernels that are otherwise __noinline__. Every call
// into a non-kernel function opens with s_waitcnt vmcnt(0) expcnt(0)
// lgkmcnt(0) -- the callee cannot know what its caller left in flight -- so
// each call drains the wave's stores, XGMI pushes included, before the phase
// starts. MPK_TRACE puts the router's at 0.9 us (S31 -> S33: wave 3's o_proj
// push). Inlined, waits land only on the registers that need them.
//   MPK_MOE_W13_INLINE  gang_moe_w13_linear_mxfp8_kernel
//   MPK_MOE_W2_INLINE   gang_moe_w2_tp_linear_mxfp8_kernel
//   MPK_GEMV4_INLINE    gang_gemv_mxfp4_kernel (o_proj)
//   MPK_RNLM8_INLINE    gang_rmsnorm_linear_mxfp8_bias_kernel (qkv_a, q_b)
// MEASURED 2026-10-11 on 5df165e + the router tie, r1 vs 4.003 ms: router
// inline 4.031, W13 4.042, W2 4.003, GEMV4 wedged in iteration 1 (2/2),
// RNLM8 not run. The drain is real but the inlined bodies cost more than it
// does; removing the drain at the source (MPK_ENTRY_NOARRIVE) is what paid.
// W2 re-measured 2026-10-12 on b6c8fa5, where the W2 return's drain of its
// write-through part words sits on the layer boundary (MPK_ML_BOUNDARY_PAD
// =1000: +0.066 ms, 85% critical) and nothing else drains before the next
// layer's first non-inlined call. n=3, text identical: 3.510 3.519 3.513 vs
// 3.540 3.531 3.533 (-0.021); worker 386 VGPR (366), 0 spills. Default on.
// W13 re-measured 2026-10-13 on the MPK_W13_SELF_TOPK default with
// MPK_W13_NO_ARRIVE=1, text identical: 3.473 3.473 vs 3.476 3.478
// (null); + MPK_W13_ACT_W3 3.489 3.478 (+0.006).
#ifndef MPK_MOE_W13_INLINE
#define MPK_MOE_W13_INLINE 0
#endif
#ifndef MPK_MOE_W2_INLINE
#define MPK_MOE_W2_INLINE 1
#endif
#ifndef MPK_GEMV4_INLINE
#define MPK_GEMV4_INLINE 0
#endif
#ifndef MPK_RNLM8_INLINE
#define MPK_RNLM8_INLINE 0
#endif
// MPK_KVUPD_INLINE: inline q_b's kvupd wrapper
// (gang_rmsnorm_linear_mxfp8_bias_mla_kvupd_kernel) into the attention task, so
// q_b calls its GEMM directly. The wrapper takes five of its arguments on the
// stack: the caller stores them, the wrapper's entry vmcnt(0) waits for those
// stores and it loads them back before it can call the GEMM. MPK_TRACE (q02,
// trace-only stamps in the q_a word poll): S18 -> poll entry 0.7-1.0 us on
// every q_b tile; on XCDs 0-6 that is the critical path, since their q_b
// workers come off their own qkv_a tile and find every q_a word valid on the
// first round.
// MEASURED 2026-10-14 on dfa7148, NP=8 1024/1024, n=3 interleaved, rank-0
// tokens identical: 3.433 3.435 3.434 vs 3.447 3.450 3.449 ms (-0.015).
// worker_kernel 390 VGPR either way. MPK_TRACE (q05): S18 -> poll entry 0.95
// -> 0.64 us, and q_b's return no longer drains its write-through store
// (S85 -> S51 0.54 -> 0.22 us).
#ifndef MPK_KVUPD_INLINE
#define MPK_KVUPD_INLINE 1
#endif
// MPK_QB_NORM_DPP: q_b's q_a RMSNorm reduction (_rnlm8_stage_norm_rcp_ll) as
// the same xor butterfly through permlane swaps and DPP instead of six
// ds_bpermute round trips, and every wave forming the four-wave total itself,
// which drops the second barrier. Same pairs, same order: bit-identical
// (tests/standalone/test_xor_sum_dpp.hip: 1M lanes, 4096 totals). MPK_TRACE
// (q02): q_a words valid -> normed (S45 -> S72) 0.7 us on every q_b tile.
// MEASURED 2026-10-14 on dfa7148, NP=8 1024/1024, rank-0 tokens identical:
// alone 3.433 vs 3.443 ms (r1); with MPK_KVUPD_INLINE, n=3 interleaved,
// 3.412 3.416 3.420 vs 3.449 3.450 3.444 (-0.032 for the pair). MPK_TRACE
// (q05): S45 -> S72 0.71 -> 0.42 us. Default on, with MPK_KV_LL (see there).
#ifndef MPK_QB_NORM_DPP
#define MPK_QB_NORM_DPP 1
#endif
// MPK_KV_LL: under MPK_QA_LL the qkv_a tiles also write the latent rows (the
// [latent | rope] 576 behind q_a) as (epoch << 32 | bf16x2) words, and XCD 0's
// latent tile (latent_to_cache) validates them into LDS and reads its row
// from there, instead of waiting at the qkv_a -> q_b rendezvous for the row.
// That rendezvous then has no waiter, so nobody arrives at it either: its
// write-through tag no longer sits in every worker's wave-0 vmcnt into the
// next call. MPK_TRACE (q02): the rendezvous released the latent tile 2.0 us
// after the last qkv_a tile, and the cache-row flag the decode waits on (S66)
// came 3.7 us after that. Same values, same sums: the cache row is
// bit-identical.
// MEASURED 2026-10-14 on dfa7148, NP=8 1024/1024, n=3 interleaved, rank-0
// tokens identical: alone 3.418 3.412 3.416 vs 3.449 3.450 3.444 ms (-0.033);
// on MPK_KVUPD_INLINE + MPK_QB_NORM_DPP + MPK_QB_BIAS_PRE 3.398 3.389 3.391 vs
// 3.429 3.418 3.425 (-0.031). MPK_TRACE (q09): latent tile S18 10.75 -> 8.24
// us, cache-row flag (S66) 13.99 -> 11.95, q_b's S18 -> poll entry 0.64 ->
// 0.31. The three defaults together vs all three off (HEAD's image), n=3
// interleaved, tokens identical: 3.389 3.390 3.387 vs 3.443 3.447 3.446 ms
// (-0.057).
#ifndef MPK_KV_LL
#define MPK_KV_LL 1
#endif
// MPK_QB_WEARLY: q_b's A tiles issued before its q_a word poll, from waves
// 1-3, with wave 0 alone polling (vmcnt is per wave and in issue order, so a
// polling wave must hold no weight). MPK_TRACE (q05, k-loop stamps 80..85):
// the tiles issued after the norm landed 0.84 us later, just behind the
// quantizer (0.75 us), so the quantizer was hiding them and MPK_QUANT_WIDE
// alone measured null. The K-split's four chains keep their k-tiles and their
// summation order (wave 1 runs two): bit-identical.
// MEASURED SLOWER 2026-10-14 on dfa7148 + MPK_KVUPD_INLINE + MPK_QB_NORM_DPP +
// MPK_QB_BIAS_PRE, rank-0 tokens identical: 3.531 vs 3.423 ms (+0.108); with
// MPK_QUANT_WIDE 3.539. MPK_TRACE (q09): wave 0's poll, entry -> words valid,
// 0.54 -> 1.62 us. vmcnt is per wave but the CU's request path is not: the
// other waves' 32 KB of tiles sit ahead of the poll's loads. Off.
#ifndef MPK_QB_WEARLY
#define MPK_QB_WEARLY 0
#endif
// MPK_QB_PF_EARLY: q_b's hoisted A tiles issued as soon as each wave's q_a
// words validate, ahead of the RMSNorm reduction, instead of after it. The
// issue is TA-bound (48 dwordx4/ubyte loads a tile, 32 KB: S72 -> S80 0.42 us
// in MPK_TRACE q09) and the tiles land 0.84 us after it, behind the
// quantizer, so the K-loop waits on them. Same loads: bit-identical.
// MEASURED SLOWER 2026-10-14 on the MPK_KV_LL defaults, n=3 interleaved,
// rank-0 tokens identical: 3.420 3.422 3.415 vs 3.394 3.379 3.391 ms
// (+0.031). MPK_TRACE (q11): the first tile still lands at ~11.06 us (S80 ->
// S81 0.84 -> 1.42 us) -- its arrival is the slab fetch's (MPK_QB_DEC_PF),
// not the issue's -- and the norm behind the issue goes 0.42 -> 0.74 us. Off.
#ifndef MPK_QB_PF_EARLY
#define MPK_QB_PF_EARLY 0
#endif
// MPK_LMH_BAL: the LM head (vocab shard 19456 rows, 304 tiles of 64 per rank)
// as one tile per worker. 38 tiles per XCD over 29 workers is two rounds: all
// 29 workers stream a 405 KB tile, then 9 of them resolve the row a second
// time and stream another. Here a worker resolves once and takes 5 or 6 of its
// XCD's 152 16-row groups; the four waves split each group's K four ways
// (_rnlm8_bal_kloop), and the head is packed K-major per group so each k-tile
// is one contiguous 2 KB read. The logits change in the last fp32 bits (four
// partial sums added in wave order instead of one MFMA chain); every rank-0
// token of every run matched. The dispatch takes MPK_WORKERS_PER_XCD tiles per
// XCD (persistent_kernel.py). Implies MPK_LMH_WRAP.
// Standalone (tests/standalone/test_lmh_bal.hip, one GPU, 232 blocks, same
// kernel and dispatch): slowest worker 45.4 -> 28.8 us, all 19456 bf16
// logits identical. Row-major +3 us, interleaved K-split +0.8, first block
// ahead of the resolve +1.2, no sched_barriers +3 to +5. MPK_LMH_TRACE on the
// megakernel agrees: head 49-51 -> 31.5-33.7 us.
// MEASURED SLOWER 2026-10-13/14 on e275027, NP=8 1024/1024, interleaved, all
// rank-0 tokens identical. Behind the wall, worker_kernel instruction-identical
// between the arms: MPK_LMH_WRAP 3.450 3.451 3.457 3.450 3.453 3.458 3.452
// 3.453 3.448 3.456 vs this 3.465 3.463 3.464 3.465 3.464 3.467 (+0.012); with
// HEAD 3.447 3.448 3.450 3.443 3.448 3.442. The head's 17 us come back as
// +0.29 us on every MoE layer of the next token (c07 trace: layer period
// 47.46 -> 47.75 us), all of it in the qkv_a resolve (+0.14) and q_b's k-loop
// (+0.14), none in the MoE half. Not the clock (MPK_LMH_TRACE's s_memtime:
// token-mean 2191.8 vs 2192.9 MHz) and not the callees' code placement
// (MPK_LMH_PAD_BYTES putting every callee back at HEAD's address mod 256 or
// mod 8 KB: 3.460 3.458, 3.461 3.461). Non-temporal loads halve the traced
// layer cost but slow the head (3.453 3.465 3.468 3.456); the row-major
// packing 3.473 3.474; tiles held until 45 us past entry 3.489 3.497. With
// every function 256-byte aligned (-falign-functions=256): HEAD 3.472 3.478,
// WRAP 3.488 3.477, this 3.452 3.445 -- where it wins, against a baseline
// the alignment cost 0.027 ms. Off.
// Not a cache footprint the head leaves behind (2026-10-14, q06): sparse
// traces, 160 decode iterations each, behind the wall vs this, at layers 1-2
// (the first after the head) and 40-41. Layers 1-2: layer period +0.075,
// -0.067, -0.123 us on ranks 0, 2, 5. Layers 40-41: this 0.26-0.36 us FASTER,
// nearly all of it in qkv_a's resolve + norm (S71 -> S54, -0.20 to -0.26),
// the interval the c07 dense trace had +0.14 slower. The per-layer sign
// follows the build (traced, untraced, aligned), not the head's traffic.
#ifndef MPK_LMH_BAL
#define MPK_LMH_BAL 0
#endif
#ifndef MPK_WORKERS_PER_XCD
#define MPK_WORKERS_PER_XCD 0
#endif
// MPK_LMH_WRAP: worker_kernel reaches the LM head through _rnlm8_lmh_wall,
// whose declared clobbers are fixed, so that changes to the head stop moving
// worker_kernel (see the head's kernel). AMDGPU allocates a caller's registers
// from its callees' actual clobbers: the first MPK_LMH_BAL clobbered a21-a31
// as well and worker_kernel went 390 -> 402 VGPRs with a different allocation
// through the fused layer, 3.490 3.498 vs 3.451.
// MEASURED (worker_kernel 1348 instructions different from HEAD's): the
// numbers above, +0.003 against HEAD. Off.
#ifndef MPK_LMH_WRAP
#define MPK_LMH_WRAP 0
#endif
// MPK_LMH_TRACE: see mpk_lmh_stamp (MPK_TRACE only). MPK_LMH_WALL_FULL: the
// wall declares every caller-saved register, so that any head -- traced ones
// use more registers -- leaves worker_kernel unchanged.
#ifndef MPK_LMH_TRACE
#define MPK_LMH_TRACE 0
#endif
#ifndef MPK_LMH_WALL_FULL
#define MPK_LMH_WALL_FULL 0
#endif
// MPK_LMH_NT: the unbalanced LM head's weight stream as non-temporal loads.
// Same register footprint as HEAD's head, so worker_kernel is unchanged.
// MEASURED SLOWER 2026-10-13 on e275027, n=3 interleaved, tokens identical:
// 3.460 3.455 3.454 vs 3.452 3.447 3.447 (+0.008); behind the wall 3.466
// 3.471 vs 3.453 3.458. Off.
#ifndef MPK_LMH_NT
#define MPK_LMH_NT 0
#endif
// MPK_LMH_PAD_BYTES: never-executed s_nop after a branch at the end of the
// instance that runs the LM head. The noinline callees behind the head in the
// code object (qkv_a, q_b's kvupd, W_UK, the decode, o_proj, the router, W13)
// start at 4-byte granularity, so a change in the head's size moves their hot
// loops against instruction-cache lines; this puts them back. A tool for
// measurement, not a lever: every callee shifted by 16 / 32 / 48 / 64 / 128 /
// 192 bytes on a752ce8, interleaved, tokens identical, 3.448 3.452 / 3.447
// 3.448 / 3.458 3.450 / 3.446 3.448 / 3.446 3.444 / 3.448 3.453 vs 3.446
// 3.439 3.451 3.445 -- flat.
#ifndef MPK_LMH_PAD_BYTES
#define MPK_LMH_PAD_BYTES 0
#endif
// MPK_LMH_ARGMAX: each LM head tile also leaves the argmax of its own logits
// in g_lmh_key, as one key whose max is the two-level reduce's winner, ties
// included: the key carries the rank the two block_reduce_max_idx tournaments
// (argmax_partial's over a chunk of MPK_LMH_ARGMAX_CHUNK columns, then the
// xrank reduce's over the chunks) break ties by (test_lmh_bal: 200 of 200 rows
// of ties, +-0, -inf and NaN pick the same column). demo.py drops
// argmax_partial, and argmax_reduce_xrank reduces the MPK_LMH_ARGMAX_SLOTS
// keys instead of the partials. Changes the task graph, so worker_kernel too.
// MEASURED NULL 2026-10-14 on e275027, interleaved, all rank-0 tokens
// identical: on the unbalanced head 3.454 3.447 with the callees padded back
// to HEAD's alignment (MPK_LMH_PAD_BYTES=204), 3.460 3.457 without, vs 3.442
// 3.451; on MPK_LMH_BAL 3.461 3.454 vs 3.464 3.467. The ~5 us argmax_partial
// took does not survive the graph change's worker_kernel. Off.
#ifndef MPK_LMH_ARGMAX
#define MPK_LMH_ARGMAX 0
#endif
#ifndef MPK_LMH_ARGMAX_CHUNK
#define MPK_LMH_ARGMAX_CHUNK 0
#endif
// The head's key slots: 8 x tiles per XCD (one per worker balanced).
#ifndef MPK_LMH_ARGMAX_SLOTS
#define MPK_LMH_ARGMAX_SLOTS 0
#endif
#if MPK_LMH_ARGMAX
__device__ unsigned long long g_lmh_key[8 * 64];
// Position t's rank in block_reduce_max_idx, smaller wins a tie: each
// shfl_down step keeps the lower lane, so the bit-reversed lane decides, then
// the same over the four waves' results.
__device__ __forceinline__ unsigned lmh_tree_rank(unsigned t) {
  return ((__builtin_bitreverse32(t >> 6) >> 30) << 6) |
         (__builtin_bitreverse32(t & 63u) >> 26);
}
// Shard column v holding bf16 `bits`: (order key of the value with -0.0 as
// +0.0, 0xFFFF - tie rank, a -0.0 flag, v), or 0 when the value cannot win
// (-inf or NaN, as with the strict > of both levels).
__device__ __forceinline__ unsigned long long lmh_argmax_key(unsigned bits,
                                                             unsigned v) {
  float const x = __uint_as_float(bits << 16);
  if (!(x > -__builtin_huge_valf())) {
    return 0ull;
  }
  unsigned const u = __float_as_uint(x == 0.0f ? 0.0f : x);
  unsigned const ok = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  unsigned const rank =
      (lmh_tree_rank(v / MPK_LMH_ARGMAX_CHUNK) << 8) |
      lmh_tree_rank(v % MPK_LMH_ARGMAX_CHUNK);
  return ((unsigned long long)ok << 32) | ((0xFFFFu - rank) << 16) |
         (bits == 0x8000u ? 0x8000u : 0u) | v;
}
#endif
// MPK_ENTRY_NOARRIVE: under MPK_ENTRY_NOWAIT, the layers nobody waits at
// (pl_layer) skip the layer-entry drain and arrival as well. Both retire in
// issue order behind the previous W2's XGMI part pushes -- the arrival is a
// returning atomic plus vmcnt(0) -- and MPK_TRACE puts the last resolver's
// S14 -> S9 at 1.2 us of exactly that. The first and tail layers still
// arrive and wait; each contributes `arrivals` to the counter, so their
// modular election and release values are unchanged, and the self-heal
// quota is only ever further away.
// MEASURED 2026-10-11 on 5df165e + the router tie: alone 3.967 vs 4.003 ms
// (r1); with MPK_OLL2_LDS, n=3, 3.956 3.959 3.960 vs 3.992 3.995 3.990
// (-0.034), text identical.
#ifndef MPK_ENTRY_NOARRIVE
#define MPK_ENTRY_NOARRIVE 1
#endif
// MPK_ROUTER_KS_MID: the K-split router tile issues its gate and gamma slices
// right after its first validation pass's loads, and that pass waits
// vmcnt(7) -- its own 16 loads, not the 7 behind them. Up front they made
// the pass retire behind the whole 3 MB burst (S33 -> first pass ~2 us for
// every tile); after validation (MPK_ROUTER_W_LATE) the gate dot waited for
// them instead (+0.027 ms). Here they stream during the validation.
// MEASURED SLOWER 2026-10-11 on 1f81211: 4.010 vs 3.950 ms (r1). The first
// pass rarely validates, so the second pass's vmcnt(0) now waits for the
// whole burst; issued up front it had finished by then. Off.
#ifndef MPK_ROUTER_KS_MID
#define MPK_ROUTER_KS_MID 0
#endif
// MPK_ROUTER_HOUT_LATE: the K-split router tile's write-through store of its
// residual slice goes out after its logit words instead of during the slot
// sum, where its ack sat in the vmcnt(0) the gate dot waits on for the gate
// slices (S37 -> S38). The normed-flag drain still retires it before the
// routing, which everything downstream waits on.
// MEASURED 2026-10-12 on g16, NP=8 1024/1024, n=3, all tokens identical on
// all ranks: 3.579 3.584 3.578 vs 3.596 3.593 3.591 (-0.013). Default on.
#ifndef MPK_ROUTER_HOUT_LATE
#define MPK_ROUTER_HOUT_LATE 1
#endif
// MPK_QUANT_WIDE: the RMSNorm + MXFP8 quantizer of qkv_a and q_b gives each
// lane eight contiguous elements, so one 16-lane DPP row is exactly one
// 128-element E8M0 block and its max is four DPP steps. The default gives
// each thread a whole 32-element sub-block -- a serial chain of 32 converts,
// multiplies and maxes, then four ds_bpermute shuffles -- on 192 threads at
// qkv_a and 48 (one wave) at q_b. MPK_TRACE: S54 -> S62 1.16 us (qkv_a),
// S72 -> S73 1.09 us (q_b), both on the critical path. Same products in the
// same order, same block max, same converts: bit-identical.
// MEASURED SLOWER 2026-10-11 on 1f81211, n=3: 3.979 3.980 3.983 vs 3.961
// 3.965 3.958 ms (+0.020), bit-exact in test_qwide (8 seeds x 1536/2048/
// 6144, LDS and global rows). That build was not the design: at q_b the
// x * rcp products packed into v_pk_mul_f32 with rcp broadcast from a VGPR
// pair whose other half was a hoisted weight load in flight, so the waitcnt
// pass put vmcnt(9) at the top of the quantizer -- an L2 round trip before
// any work. And at qkv_a's 6144 it is three trips with a DPP max each, 336
// instructions per wave against the default's 267. Now: rcp from an SGPR,
// and wide only where it is one trip (REDUCTION_SIZE / 8 <= 256, i.e. q_b's
// 2048): prefetch -> barrier 250 -> 109 instructions, spread over 4 waves.
// test_qwide bit-exact. MEASURED NULL 2026-10-11 (r1, text identical): 3.889
// vs 3.876; with MPK_QB_SELF_PF 3.876. The shorter quantizer only moves the
// wait onto the hoisted weight loads, and pulling those into L2 first does
// not move the wall either. Off.
// Re-measured 2026-10-13 on 91c270e, where q_b's last tile per XCD sets the
// W_UK -> decode chain (MPK_WUK_LAT_REMAP moved it there), n=3 interleaved,
// rank-0 tokens identical: 3.479 3.479 3.482 vs 3.466 3.471 3.467 ms
// (+0.013); again on stage d15, 3.479 3.477 3.481 vs 3.470 3.465 3.471
// (+0.010). MPK_TRACE: the quantizer is 0.4-0.7 us instead of 1.06, but the
// W_UK chain does not start earlier.
// Again 2026-10-14 on dfa7148, n=3 interleaved, tokens identical: 3.449 3.454
// 3.461 vs 3.447 3.450 3.449 ms (null). MPK_TRACE (q05, k-loop stamps): the A
// tiles land 0.84 us after their issue and the default quantizer takes 0.75,
// so it was hiding them (MPK_QB_PF_EARLY, MPK_QB_WEARLY).
#ifndef MPK_QUANT_WIDE
#define MPK_QUANT_WIDE 0
#endif
// MPK_W13_ACT_W3: the K_HALF W13 tile parks its SwiGLU epoch words in LDS and
// wave 3 alone stores them. They were issued by waves 0/1 -- exactly the
// waves that then poll W2's staging, where the polls retire in issue order
// behind those write-through stores. MPK_TRACE: the last W2 tile to stage is
// always the last W13 worker's own, 3.12 us after the last W13 k-loop
// against 2.19 for the W2 workers that ran no W13. Pair with
// MPK_W13_NO_ARRIVE, whose all-wave drain would otherwise put it back.
// MEASURED NULL 2026-10-11 on 7bc1c7e, with MPK_W13_NO_ARRIVE: 3.954 vs 3.950
// ms (r1), text identical. Off.
#ifndef MPK_W13_ACT_W3
#define MPK_W13_ACT_W3 0
#endif
// MPK_ROUTER_L2PF: each router-tile worker pulls its own K-split slice of the
// gate rows (32 rows x 768 B) and of gamma into L2 at Phase 8, ~10 us ahead
// of its router tile, through a never-read LDS window as the o_proj prefetch
// does. At the tile's head all 128 tiles of a GPU ask HBM for their 24 KB at
// once, and the slice validation's first pass retires behind that burst
// (MPK_TRACE: S33 -> first pass 2.15 us); from L2 the burst is short.
// MEASURED 2026-10-12 on a87dc19: won three standalone pairs by ~0.020 ms.
// Combined with MPK_OPROJ_PF_MIN_US=14 it ran at 3.496/3.501/3.492 ms versus
// 3.520/3.509 ms controls, with identical tokens, PPL 2.3670, and passing
// 256/512/1024 longseq.
#ifndef MPK_ROUTER_L2PF
#define MPK_ROUTER_L2PF 1
#endif
// MPK_DEC_DIRECT_WAIT: under the head-local decode the merge ranks poll the
// 64 decode workers' tags themselves instead of their XCD's release line.
// The tagged rendezvous' relay -- xcd_rank 0 polls all 232 workers' slots,
// then republishes in L2 -- waits for every worker to reach Phase 6, decode
// or not, and adds its own hop; the merge reads only the decode partials,
// and each decode worker drains them before its tag. MPK_ATTN_PROBE_NODECWAIT
// prices the whole wait at -0.085 ms (7bc1c7e, r1).
// Every merge rank polling the 64 tags itself MEASURED SLOWER 2026-10-11,
// 4.033 vs 3.950 ms (r1), text identical -- thirty-two memory pollers on the
// lines the stragglers write (the reason MPK_BAR_TAGGED_HIER exists). So
// xcd_rank 0 alone polls and relays, as the stock site does; only what it
// waits for narrows. That MEASURED SLOWER too: 4.042 vs 3.946 ms (r1), text
// identical. Starting the merge on the decode workers alone, ahead of the
// rest of the gang, costs ~1.2 us/layer somewhere downstream; the
// NODECWAIT probe's -0.085 came with wrong attention output, hence wrong
// routing and a different MoE, and is not a price for this wait. Off.
#ifndef MPK_DEC_DIRECT_WAIT
#define MPK_DEC_DIRECT_WAIT 0
#endif
// MPK_DEC_OUT_B128: the split-KV decode writes each lane's four f32 partials
// as one write-through buffer_store_dwordx4 (cache policy sc0 sc1, through the
// raw-buffer intrinsic, so the compiler allocates the aligned VGPR quad that
// broke the inline-asm dwordx4 form) instead of two st_wt_u64. MPK_TRACE: the
// output step is 1.40 us on every decode item, all of them on the critical
// path, for 16 KB of stores.
// MEASURED 2026-10-11 on 7bc1c7e, n=3: 3.897 3.890 3.896 vs 3.938 3.939 3.938
// ms (-0.044), text identical; test_mla_decode 60/60 at both WRITE_THROUGH
// arms. Default on.
#ifndef MPK_DEC_OUT_B128
#define MPK_DEC_OUT_B128 1
#endif
// MPK_MERGE_LSE_LDS: the split-KV merge's halves path reads the block's lse
// words once (two per thread) into LDS instead of each thread loading its
// half's HC lse words itself -- the same HC addresses for all 16 lanes of a
// group, so 32 of each thread's 64 scattered global loads were duplicates.
// The merge is load-issue bound like the decode's output was. Bit-exact.
// MEASURED SLOWER 2026-10-11 on d2d99b1: 3.913 vs 3.894 ms (r1). A group's 16
// lanes on one lse address coalesce to one transaction, so the duplicates
// were nearly free and the LDS round trip plus barrier are not. Off.
#ifndef MPK_MERGE_LSE_LDS
#define MPK_MERGE_LSE_LDS 0
#endif
// MPK_MLA_QK_PIPE: the decode's 18-step QK loop reads K chunk kc+1 from LDS
// before chunk kc's MFMA. The volatile MFMA asm kept the scheduler from
// hoisting any LDS read past it, so every step was ds_read_b128 ->
// s_waitcnt lgkmcnt(0) -> MFMA, a full LDS round trip per step (final ISA);
// MPK_TRACE has the one-tile loop at 2.76 us on every decode item. Same MFMA
// order: bit-exact. Final ISA: each step now waits lgkmcnt(1).
// MEASURED 2026-10-11 on d2d99b1, n=3: 3.868 3.870 3.867 vs 3.893 3.889 3.894
// ms (-0.024), text identical; test_mla_decode 60/60 both arms. Default on.
#ifndef MPK_MLA_QK_PIPE
#define MPK_MLA_QK_PIPE 1
#endif
// MPK_MLA_PV_PIPE: the same for the PV step's V reads -- block vb+1's four
// ds_read_u16 go out before block vb is packed. The allocator reused one set
// of temporaries, so each block waited lgkmcnt(0) before the next block's
// reads could issue: eight serialized LDS round trips per tile. Bit-exact.
// NO EFFECT as written, with or without a sched_barrier: the final ISA
// already overlaps half a block (block N's second pair with block N+1's
// first), and what serializes the groups is the allocator reusing the same
// few temporaries -- the next group cannot issue into registers a pending
// pack still reads. More temporaries (or asm with fixed ones) is the fix,
// in the function that sets the megakernel's VGPR budget. Off.
#ifndef MPK_MLA_PV_PIPE
#define MPK_MLA_PV_PIPE 0
#endif
// MPK_MLA_PV_ASM: the PV step's 32 V reads (ds_read_u16, four rows of each of
// the eight 64-column blocks) issued from asm into 32 distinct registers with
// one wait, then packed. The final ISA of the C++ form reuses each block's
// second and fourth read registers as the next block's first and third, so a
// block's reads wait on the previous block's v_perm: eight serialized LDS
// round trips per tile. Same values, same packing, same MFMAs: bit-exact.
// The ISA is as designed: 32 ds_read_u16, one lgkmcnt(0), 16 packs, 8 MFMAs.
// MEASURED NULL 2026-10-11 (r1, text identical): 3.884 vs 3.876 3.882. Off.
// On the swizzled tile (MPK_MLA_KV_SWZ, a two-base variant of the same asm):
// NULL too, 3.754 vs 3.747 on the g14 defaults.
#ifndef MPK_MLA_PV_ASM
#define MPK_MLA_PV_ASM 0
#endif
// MPK_MLA_QK_PIPE2: the QK loop with two K chunks in flight instead of one.
// MPK_TRACE stamps 75-77 (trip 0, median item): QK 0.68 us, softmax 0.20,
// PV 0.48 per tile -- QK at about one LDS round trip per step, against an
// 18-step MFMA chain of ~0.17 us. Same MFMA order: bit-exact.
// Alone on the unswizzled tile NULL (3.846 vs 3.845): its K reads are 8-way
// bank-conflicted, so a second one in flight only queues behind the first.
// With MPK_MLA_KV_SWZ (see gang_mla_decode_mi300.cuh) on DEC_Q_INC +
// RESOLVE_POLL_INC, n=2: 3.789 3.783 -> SWZ 3.764 3.763 -> SWZ + PIPE2 3.752
// 3.753 ms, text identical. Default on with KV_SWZ (gates g14).
#ifndef MPK_MLA_QK_PIPE2
#define MPK_MLA_QK_PIPE2 1
#endif
// MPK_MLA_QK_SPLIT: the decode's 16 x 16 QK score block is computed by one
// wave instead of all four (each read the whole 18 KB K tile for the same
// scores) and handed to the others through the dead Q staging block at
// 96 KB; when the next tile is already in LDS (MPK_DEC_PRELOAD), wave 1 runs
// that tile's chain beside it and the next trip skips QK. Same chains: same
// bits. MPK_MLA_SKIP_DECODE prices the decode item at -0.358 ms (b6c8fa5).
// MEASURED SLOWER 2026-10-17 on 6c58431, NP=8 1024/1024, n=3 interleaved, G1
// PASS, tokens identical to the default in both arms (test_mla_decode 66/66,
// TEST_DUMP byte-identical): 3.419 3.418 3.425 vs 3.368 3.366 3.365 ms
// (+0.054). Trip 0's chain is no shorter on one wave, and every trip pays the
// score exchange. Off.
#ifndef MPK_MLA_QK_SPLIT
#define MPK_MLA_QK_SPLIT 0
#endif
// MPK_PROBE_MERGE63: PRICING PROBE, WRONG OUTPUT. The merge's halves path
// issues 2 x 32 loads before one wait, one more than vmcnt's 63 in flight, so
// the 64th cannot issue until the first returns. The probe drops one.
// MEASURED NULL 2026-10-11 on the g14 defaults: 3.752 vs 3.747 -- the 64th
// load's wait is not a cost worth a redesign (cf. MPK_MERGE_LSE_LDS).
#ifndef MPK_PROBE_MERGE63
#define MPK_PROBE_MERGE63 0
#endif
// (MPK_WUK_PF below: 5.064 5.057, -0.029.) All five together, interleaved
// against their own control, text identical: 5.109 5.110 5.094 -> 4.831
// 4.828 4.821 ms (-0.278).
#ifndef MPK_WUK_PF
#define MPK_WUK_PF 1
#endif

#ifndef MPK_TOPK_PAD_NS
#define MPK_TOPK_PAD_NS 0
#endif
// MPK_MERGE_HALVES: the split-KV merge (tasks/ampere/merge_splitkv.cuh) runs
// each head on two thread groups, one per chunk half, so every thread's lse
// and o loads go out together: one round trip instead of two.
// MEASURED 2026-10-09, NP=8 1024/1024, interleaved: 5.893 5.893 5.901 ->
// 5.756 5.753 5.737 5.734 5.745 ms (-0.15). Tokens move (the merge
// reassociates); ppl512 2.4098 against 2.4112; test_mla_merge own8 cases
// PASS. Default on.
#ifndef MPK_MERGE_HALVES
#define MPK_MERGE_HALVES 1
#endif
// MPK_MERGE_QUAD: the halves path on one wave, lane = (head, half, 4 dims):
// a lane's partials are one dwordx4 per chunk, 32 + 32 loads in flight
// against the halves path's 4 waves x 64 dword loads, and each dim keeps the
// halves path's arithmetic, with the combine's two fmas pinned. Not quite
// bit-identical: test_mla_merge --dump differs from the halves path in 3
// bytes over 18 cases (accuracy PASS), so the tokens may move.
// MEASURED SLOWER 2026-10-13, n=2 interleaved, tokens moved: 3.516 3.516 vs
// 3.502 3.495 ms (+0.017). Off.
#ifndef MPK_MERGE_QUAD
#define MPK_MERGE_QUAD 0
#endif
// MPK_MERGE_LIVE: the merge's halves path reads o only for the chunks the
// decode's partition fills (KV_TILE 16, ceil(tiles / 64) a chunk): 33..64 of
// 64 over a 1024/1024 run, ~48 on average. An empty chunk keeps its own lse
// word (the decode's -1e30, weight 0) and reads chunk 0's o, a line the
// thread fetches anyway; only the address moves, so the loads stay batched.
// Bit-identical (test_mla_merge --dump). (Under MPK_MERGE_QUAD, a per-chunk
// select of the loaded values instead: MEASURED +0.51 ms 2026-10-13, r1 --
// under the attention task's register pressure each chunk's loads were
// waited for in turn.)
// MEASURED NULL 2026-10-13, r1, rank-0 tokens identical: 3.513 vs 3.507 ms.
// A quarter fewer partial lines does not move the merge: it is a round
// trip, not bytes. Off.
#ifndef MPK_MERGE_LIVE
#define MPK_MERGE_LIVE 0
#endif
// MPK_MERGE_EXP2B: the halves path's 34 exp2 as __builtin_amdgcn_exp2f, one
// v_exp_f32, instead of exp2f's denormal-safe form (compare, two selects, an
// add and a v_ldexp around it, plus an s_nop): ~200 of the merge's VALU
// instructions, all after its loads land. The other operations are pinned to
// the forms the default's final ISA has (the vectorizer regroups them around a
// speculatable exp2). The results differ only where exp2f returns a denormal
// (argument < -126), i.e. a weight under 2^-126 beside the max chunk's 1,
// below the f32 ulp of every sum it joins.
// MEASURED 2026-10-14 on 98f581f, NP=8 1024/1024, n=3 interleaved, all 1024
// rank-0 tokens identical: 3.352 3.361 3.361 vs 3.383 3.393 3.381 ms
// (-0.028); again in a03, 3.365 3.359 3.350. MPK_TRACE: the merge item
// 2.00 -> 1.56 us. Defaulted, interleaved n=3 default vs =0: 3.356 3.353
// 3.350 vs 3.394 3.389 3.384 ms (-0.036), all tokens identical; standalone
// 90 PASS / 0 FAIL, ppl512 2.3670 on all 8 ranks, longseq 256/512/1024 G1
// PASS. Default on.
#ifndef MPK_MERGE_EXP2B
#define MPK_MERGE_EXP2B 1
#endif
#if MPK_MERGE_EXP2B && (MPK_MERGE_LSE_LDS || MPK_DEC_LSE_LL || \
                        MPK_DEC_MERGE_LL32 || MPK_MERGE_LIVE ||  \
                        MPK_DEC_PART_F16)
#error "MPK_MERGE_EXP2B pins the halves path's default load branch only"
#endif
// MPK_DEC_PART_F16: the decode's split-KV partials leave as f16 instead of
// f32 -- 8 KB a decode item instead of 16, one write-through dwordx2 a lane
// per block -- and the merge reads them so. A partial is a convex
// combination of latent rows, far inside f16's range; rounding is 2^-11,
// under the merged output's own bf16 rounding. VALUE CHANGE: the tokens
// move, ppl-gated. MPK_TRACE: the output step S49 -> S50 is 0.83 us and the
// return drain 0.43 us, both on the chain; MPK_DEC_MERGE_LL32's 2x bytes
// doubled both.
// MEASURED NULL 2026-10-13, r1: 3.506 vs 3.507 ms (with MPK_MERGE_LIVE
// 3.524). Half the bytes moves neither leg. Off, so never ppl-gated.
#ifndef MPK_DEC_PART_F16
#define MPK_DEC_PART_F16 0
#endif
#if MPK_DEC_PART_F16 && (MPK_DEC_MERGE_LL || MPK_DEC_LSE_LL || MPK_DEC_MERGE_LL32)
#error "MPK_DEC_PART_F16 is the barrier path's partials"
#endif
// Same default as gang_oproj_router_fused_mi300.cuh, which the MoE kernels'
// header is compiled ahead of.
#ifndef MPK_MOE_TP
#define MPK_MOE_TP 0
#endif
#define MPK_RLL_WORD_LINES 32
#define MPK_RLL_IDS_LINES 17
#define MPK_RLL_XCD_INTS ((2 + 2 * MPK_RLL_IDS_LINES) * 16)
#define MPK_RLL_LINES (MPK_RLL_WORD_LINES + 8 * MPK_RLL_XCD_INTS / 16)

// MPK_QKV_MXFP4: GLM's fused-layer qkv_a weight is packed E2M1 instead of
// E4M3 (demo.py GLM_QKV_MXFP4), halving the phase's weight stream. Faked in
// bf16 first (GLM_FAKE_MXFP4_QKV): ppl@1024 2.50 vs 2.57 default.
//
// MEASURED 2026-10-08, NP=8 1024/1024, n=3: 7.178 7.176 7.155 -> 6.957
// 6.978 6.965 ms (-0.21). The E2M1 path is bit-exact against E4M3 on the same
// values (tests/standalone/test_qkv_fp4.hip). But quality: ppl512 is
// deterministic here and reads 2.4838 (x3) against 2.5532 (x3), +2.8%, and
// the 1024-token generation collapses into a repetition loop (distinct 0.078)
// where the default does not. Off by default.
//
// Three corpus windows, ppl512 (PPL_SKIP_TOKENS 0 / 1024 / 2048):
//   E4M3 default   2.4838  2.9497  2.0749
//   E2M1 ties-up   2.5532  3.0345  2.1836   (+2.8% +2.9% +5.2%)
//   E2M1 RNE       2.5205  2.9785  2.1440   (+1.5% +1.0% +3.3%)
// GLM_QKV_FP4_RNE narrows it but never closes it: the format costs quality
// on qkv_a, not the rounding rule. Rejected.
#ifndef MPK_QKV_MXFP4
#define MPK_QKV_MXFP4 0
#endif
#if MPK_QKV_MXFP4 && (MPK_QKVA_ENTRY_PF || MPK_QKVA_PF_KB > 0 || MPK_QKVA_IDLE_PF)
#error "MPK_QKV_MXFP4: the qkv_a prefetchers still size tiles as E4M3"
#endif

// MPK_MOE_SHADOW_KB: kilobytes of cold weight read that each W13-idle worker
// pulls during the W13 phase. CORRECT OUTPUT -- see the long note at the
// probe site in gang_oproj_router_fused_mi300.cuh. 87 is the like-for-like
// qkv_a dose (16.1 MB/layer/rank over 23 tiles/XCD).
#ifndef MPK_MOE_SHADOW_KB
#define MPK_MOE_SHADOW_KB 0
#endif

// MPK_QKVA_PF_KB: kilobytes of the NEXT layer's qkv_a weight that each
// W13-idle worker pulls into cache during this layer's W13 phase. 0 = off.
//
// THE LOOPHOLE THIS EXPLOITS. glm-heterogeneous-worker-groups-impossible was
// closed by MPK_QKVA_REPS=2 measuring +0.669 ms, but that priced moving
// qkv_a's COMPUTE, which really does depend on MoE(L) through the residual and
// the RMSNorm. qkv_a's WEIGHTS do not: they are constants, known a whole
// layer ahead, and moving only the bytes is dependency-free. The capacity
// question was already answered the other way -- MPK_MOE_SHADOW_KB put
// 14.6 MB/layer/rank of cold read on the 21-of-29 W13-idle workers for
// +0.030 ms (glm-moe-phase-has-a-free-worker-hole), which is ~90% of qkv_a's
// entire weight volume, free. So the bytes fit and nothing forbids moving
// them; what is unknown is whether they SURVIVE to the qkv_a phase, since the
// MoE streams ~25 MB/layer/rank through the same L2.
//
// The pointer is a kernel argument (next_qkv_weight_ptr), filled from
// task_desc->input_ptrs[MPK_QKVA_PF_SLOT]. The multi-layer loop writes that
// slot from ml_input_table[ml+1][3] -- the same qkv_a weight the next layer's
// prologue already dereferences. A __device__ side channel (g_ml_next_qkv_w)
// illegal-addressed at KB=1 AT=13 even after zero-init+syncthreads; cause not
// localized. The slot-33 argument path has not faulted at KB=1.
//
// A worker reads [j * KB, (j+1) * KB) of the slice, j = xcd_rank - live,
// and skips itself if that would leave the bounded window -- exactly the guard
// MPK_MOE_SHADOW_KB uses.
#ifndef MPK_QKVA_PF_KB
#define MPK_QKVA_PF_KB 0
#endif
// Past every slot the fused MLA layer declares (used_in maxes at 31 under
// EP + both un-absorptions). MAX_INPUTS_PER_TASK is 34.
#ifndef MPK_QKVA_PF_SLOT
#define MPK_QKVA_PF_SLOT 33
#endif
// 2026-09-22, 1024/1024, chunks=32, GPUs 4-7, g_ml_next_qkv_w side channel:
//   KB=256 AT=2: 9/9 rc=124. Control 11.256 G1 PASS.
//   KB=1  AT=2: hipErrorIllegalAddress, then after zero-init+syncthreads
//               still 5/5 rc=124 (no HIP error -- watchdog).
//   KB=1  AT=13: hipErrorIllegalAddress again.
// Also AT=2 sat AFTER `if (xcd_rank >= moe_w2_tiles_per_xcd) return`, so the
// W2-idle workers never issued the dose. Re-plumb via MPK_QKVA_PF_SLOT and
// issue AT=2 before that return. The oproj parameter is compile-gated so the
// KB=0 build is unchanged.
// 2026-09-23, 1024/1024, chunks=32, GPUs 4-7, slot 33, multi-worker dose:
//   current qkv as fused arg:      11.343 G1 PASS
//   current qkv via slot 33:       11.353 G1 PASS
//   next-layer KB=1:               11.366 G1 PASS (control 11.281; NULL)
//   KB=256:                        one hipErrorIllegalAddress (rank 3), not
//                                  localized -- the <=1.5 MB read is inside
//                                  the 2.32 MB chunk (see the window below)
// NOT ATTRIBUTABLE -- do not cite as variant failures: next-layer 4/4 rc=124
// (before the heap guard), KB=256 single-worker 3/3 rc=124, and the rc=124
// counts in the 09-22 block above. The CONTROL wedged 5 of 7 attempts in the
// same session, including 3/3 on an ISA byte-identical (f6058fb5) to a
// control that then passed first try. The heap guard is a no-op on this box:
// consecutive layers' qkv_a are 18-69 GB apart, under its 256 GB bound, so
// "guarded" and "unguarded" published the same pointer.
// Keep 0: KB=1 is legal and NULL; KB=256 has one unexplained fault.
// The window a worker's dose must fit inside. qkv_a is padded to 3072 rows
// (q_lora 2048 + kv_a 576 -> 1024), so the per-XCD MXFP8 chunk is
// 384 * (6144 + 192) = 2.32 MB and 1.5 MB is in bounds without the kernel
// having to derive the extent it cannot see.
#define MPK_QKVA_PF_WINDOW ((size_t)1536 * 1024)

// Which idle-worker hole issues the dose. 13 = the W13-idle hole (21 of 29
// workers, measured NEUTRAL -- W2 streams over the lines before qkv_a(L+1)
// reads them); 2 = the W2-idle hole (workers with no W2 tile, issued before
// they return to the fused epilogue, concurrent with the W2 wait+tiles).
// See the placement comment in gang_oproj_router_fused_mi300.cuh.
#ifndef MPK_QKVA_PF_AT
#define MPK_QKVA_PF_AT 2
#endif

// MPK_QKVA_REPS: how many times the qkv_a tile loop runs. 1 is the shipping
// path. CORRECT OUTPUT at any value, and that is the point of the probe --
// the loop body is idempotent, so the wall number is gateable.
//
// WHY. e6ccf70 measured that the MoE phase has a free worker-shaped hole big
// enough to hide all of qkv_a, so item 2 (running qkv_a(L+1) concurrently
// with MoE(L)) is now blocked only by the data dependency: qkv_a(L+1) reads
// MoE(L) through the residual. RMSNorm is linear in its argument up to a
// scalar, so that dependency can be split --
//
//   q_a = (1/rms(h2)) * [ (h1 . g) @ W  +  (moe_out . g) @ W ]
//
// with h1 = x + attn_out available right after o_proj(L). The h1 term could
// run in the free MoE hole. But BOTH terms are full-size GEMMs over the same
// 16.1 MB/layer/rank W, so the split does not shrink the un-hidden half; it
// adds a second pass over W. The whole of item 2 therefore turns on one
// measurable quantity: what does an EXTRA un-hidden qkv_a pass cost?
//
//   * ~+1.1 ms (one qkv_a tile makespan, 14.8 us/layer x 78) => the split is
//     neutral at best and item 2 closes by measurement.
//   * ~0 => qkv_a's tile is fully re-absorbed by the barrier behind it, so no
//     qkv_a lever exists at all. That is consistent with MPK_ATTN_HALFK ~= 0
//     and with the barrier-reabsorption rule, now 4-for-4.
//
// ── MEASURED: +0.669 ms. THE FIRST BRANCH. ITEM 2 IS CLOSED. ──
// Paired A/B, min-of-115, alternating arms, same batch, identical generated
// text arm for arm:
//
//   REPS=1   10.250  10.267  10.249   mean 10.255
//   REPS=2   10.882  10.949  10.941   mean 10.924
//   delta    +0.669 ms  =  +8.6 us/layer     (no overlap; floor is 0.26)
//
// Two conclusions, and the second is the more useful one.
//
// (1) The RMSNorm-linearity split cannot pay. It does not shrink the
//     un-hidden GEMM -- term B (moe_out . g) @ W is the same size as today's
//     single pass over the same W -- it only ADDS term A. So the best case is
//     that term A is perfectly absorbed by the MoE hole and the wall is
//     unchanged; the measured case is that an extra pass over W costs 0.669
//     ms whenever it is not absorbed. There is no version of the split where
//     the un-hidden half gets cheaper. Item 2 closes: the free hole from
//     e6ccf70 is real, but no layer-(L+1) work exists that can legally go in
//     it, because everything in qkv_a(L+1) is downstream of MoE(L).
//
// (2) An ADDED pass costs ~1:1. 8.6 of a 14.8 us/layer tile shows up on the
//     wall, and 8.6 is a lower bound since pass 2 re-reads a W that pass 1
//     just pulled through L2 (2 MB/XCD against a 4 MB slice).
//
//     CORRECTION, and it matters: this is an ADDITIVE probe and it does NOT
//     upper-bound a DELETION. That is the exact inference retracted in
//     4fb029a -- MPK_ML_BOUNDARY_PAD's 1.22:1 additive slope priced the layer
//     boundary at ~1.4 ms and deleting the whole boundary bought 0.051 ms.
//     Adding uniform time in front of a rendezvous costs what you add;
//     removing it just moves every worker's arrival earlier and the last
//     arriver still sets the release. The subtractive counterpart for qkv_a
//     already exists and disagrees: MPK_ATTN_HALFK removes half the K-loop's
//     bytes and buys only 4.91 us/layer. So conclusion (1) -- which prices an
//     ADDITION, which is what the linearity split is -- stands; any claim
//     that shrinking a tile pays needs its own subtractive probe.
//
// Idempotency, checked before trusting the number: on the default path
// (MPK_QKV_EP_FOLD=0, MPK_QKV_PRO_HOIST=0) norm_input_ptr is x_ptr, the
// FUSE_RESADD prologue computes d_x_out[h] = bf16(ws_f32[h] + x[h]) as a pure
// store, and the GEMM write-through-stores to qkv_a_out. Sources and
// destinations are disjoint and no accumulation is involved, so pass 2 writes
// the same bytes pass 1 did.
#ifndef MPK_QKVA_REPS
#define MPK_QKVA_REPS 1
#endif
// Preprocessor, not static_assert: MPK_QKV_EP_FOLD / MPK_QKV_PRO_HOIST are
// -D-only and undefined on the default path, so a C++ expression naming them
// would not compile.
#if MPK_QKVA_REPS != 1
#if defined(MPK_QKV_EP_FOLD) || defined(MPK_QKV_PRO_HOIST)
#error "MPK_QKVA_REPS is only idempotent on the unfolded path: under the fold/hoist norm_input_ptr aliases x_out_ptr and pass 2 would re-resolve an already-resolved row"
#endif
#endif

// MPK_W13_REPS: the same probe as MPK_QKVA_REPS, aimed at the MoE W13 phase
// -- the other big tile phase (34.6 us/layer, 16.9 of it tile). 1 is the
// shipping path. CORRECT OUTPUT: W13's epilogue is pure stores into the
// swiglu scratch, disjoint from its inputs, so the body is idempotent.
//
// The pair (QKVA_REPS, W13_REPS) is a MARGINAL-cost map of the two phases
// that hold half the layer. Stage stamps cannot answer this -- they count a
// worker's spin as its region time -- but doubling a phase and reading the
// wall prices exactly the part that is on the critical path.
//
// W2 deliberately has no such knob: its epilogue f32-atomicAdds into the
// residual workspace, so a second pass doubles the layer output and the
// number would not be gateable.
//
// ── MEASURED: +1.152 ms. ──
// Paired A/B, min-of-115, alternating, same batch, correct output both arms:
//
//   REPS=1   10.260  10.221  10.281   mean 10.254
//   REPS=2   11.455  11.390  11.372   mean 11.406
//   delta    +1.152 ms  =  +14.8 us/layer
//
// The W13 phase is 34.6 us/layer of which 16.9 is tile, so a second pass
// costs 88% of a first one. READ THIS AS AN ADDITION ONLY -- see the
// correction under MPK_QKVA_REPS. It says the ~17.7 us/layer of barrier spin
// behind W13 does not absorb added work; it does NOT say that removing W13
// work would save any. The open question is the subtractive counterpart, and
// there is no correct-output way to ask it: a wrong-output W13 corrupts the
// layer output, hence the NEXT layer's TopK and EP balance, which is the trap
// in glm-wrong-output-probes-upstream-of-router-are-invalid.
//
// The structural fact the number does confirm is occupancy, not absorption:
// W13 runs on 8 of 29 workers per XCD (e6ccf70 measured the other 21 idle and
// able to pull 87 KB each for free), so the phase draws roughly a quarter of
// the machine's HBM. Narrowing the tile to fill them is the obvious fix and
// is already measured out from both directions -- OPW=16 costs 1.34 ms
// because 16 output rows starve a 4-wave MFMA, and W2 split-K costs 4.12 ms.
#ifndef MPK_W13_REPS
#define MPK_W13_REPS 1
#endif

// ═══════════════════════════════════════════════════════════════════════════
// MPK_ABL_PIPE_W13W2: the ADJACENT-PHASE OVERLAP CEILING PROBE.
// ═══════════════════════════════════════════════════════════════════════════
// 0 = off (shipping).  1 = control, CORRECT OUTPUT.  2 = probe, WRONG OUTPUT.
//
// THE QUESTION. Every phase in the GLM layer runs on a fraction of the 29
// workers per XCD and the rest spin at the next rendezvous
// (glm-per-phase-occupancy-is-the-whole-story). The standing counter-argument
// to filling those workers is that the layer's cost is a MAX over a static,
// skewed arrival distribution, in which case moving work off a worker that was
// not the last arriver buys nothing -- which is what every deletion probe on
// this branch has measured (glm-additive-probes-overprice-deletions). The
// additive probes (MPK_QKVA_REPS, MPK_W13_REPS) cannot settle it: they add
// work to EVERY worker, so they necessarily lift the max, and that outcome is
// uninformative about whether an overlap would help.
//
// The only probe that settles it MOVES work across a rendezvous. That is this.
//
// THE PAIR. W13 -> W2, the widest idle-worker margin in the layer whose
// successor is large enough to resolve above the 0.26 ms wall floor:
//   * W13 runs on moe_w13_live ~= 8 of 29 workers/XCD -- 21 idle, and
//     glm-moe-phase-has-a-free-worker-hole already measured those 21 absorb an
//     87 KB cold read for +0.030 ms.
//   * W2 is ~12 tiles/XCD, the biggest MoE phase (35.18 us in BAR_SKEW slot 0).
// decode -> merge has a wider margin (27 of 29 idle) but merge is ~13 us/layer
// total, so its whole ceiling is under the noise floor. Not chosen.
//
// THE MECHANISM. The W13-idle workers [moe_w13_live, ...) each take exactly
// one W2 tile off the front of the W2 tile space; the normal W2 loop skips
// exactly that prefix. Work is MOVED, never added -- tile count, f32-atomicAdd
// count and barrier arrival count are all identical to the shipping path, so
// this is NOT another additive probe.
//
//   =1  the moved tiles run AFTER the W13->W2 rendezvous. Output is CORRECT
//       (the swiglu rows they read are finished). This is the control, and it
//       carries the identical tile->worker permutation, so the A/B isolates
//       exactly one variable: whether the moved tiles overlap W13 or not.
//   =2  the same moved tiles run BEFORE the rendezvous, concurrently with the
//       W13 tiles that produce their inputs. WRONG OUTPUT by construction.
//
// Comparing 2 against 1 rather than against the shipping default is the point:
// the permutation is held fixed, so nothing but the overlap changes.
//
// THE CONFOUND, stated rather than fixed. Arm 2's layer output is garbage, so
// the NEXT layer's router sees garbage logits and the TopK -- hence the EP
// balance -- differs from the control's. That is the trap in
// glm-wrong-output-probes-upstream-of-router-are-invalid (MPK_ABL_QKV_PRO did
// strictly LESS work and cost +2.83 ms). Its sign here is unknown, which is
// why the decision rule is one-sided and generous to the probe: a ceiling that
// does not appear is dead, a ceiling that does appear still has to survive a
// correct-output implementation before it is believed.
//
// DECISION RULE (agreed before the run): |arm2 - arm1| < 0.3 ms => adjacent-
// phase pipelining is dead on GLM and is written up as such. Larger => build
// the barrier-epoch restructure and do not measure until it is correct.
//
// ── MEASURED. THE RULE FIRES: PIPELINING IS DEAD. ──
// n=3 paired, alternating, min-of-115. Geometry that ran: owned ~= 1 expert,
// moe_w13_live = 8, moe_w2_live = 12, 21 idle workers/XCD, so _pipe_tiles = 12
// -- ALL 12 of 12 live W2 tiles relocated. The whole phase, not a slice.
//
//   arm 0  shipping                     10.223  10.281  10.229   mean 10.244
//   arm 1  permutation, below barrier   10.400  10.368  10.410   mean 10.393
//   arm 2  permutation, above barrier   10.102  10.111  10.082   mean 10.098
//
//   arm2 - arm1 = -0.295 ms   overlap isolated, permutation held fixed
//   arm2 - arm0 = -0.146 ms   the ceiling a CORRECT build must beat, before
//                             it pays anything for barrier-epoch machinery
//   arm1 - arm0 = +0.149 ms   the permutation ALONE costs money, so a real
//                             pipelined schedule starts in the hole
//
// 0.295 is under the 0.3 threshold and 0.146 is barely half the 0.26 ms wall
// floor. Hiding a phase the per-phase decomposition prices at ~16 us/layer of
// tile time recovered 1.9 us/layer -- the same absorption signature as
// glm-cutting-work-in-a-phase-is-absorbed and
// glm-deleting-a-whole-rendezvous-is-neutral. The layer's time is not in the
// phase boundaries, so re-packing work across them is not a lever. This also
// closes the last live half of glm-heterogeneous-worker-groups-impossible.
//
// W13 -> W2 is the widest idle margin whose successor can resolve at all;
// decode -> merge has 27 of 29 idle but merge is ~13 us/layer total, entirely
// under the floor. There is no better pair to retry with.
//
// ── RE-MEASURED 2026-10-02 AT NP=4, GLM-5.2, 1024/1024: WORSE, NOT BETTER. ──
// 2 alternating reps, decode median, all G1 PASS (arm 2 text garbage):
//
//   arm 0  shipping                     9.727  9.744   mean 9.736
//   arm 1  permutation, below barrier   9.716  9.697   mean 9.706
//   arm 2  permutation, above barrier   9.892  9.865   mean 9.878
//
//   arm2 - arm1 = +0.172 ms. At four ranks each one owns 2-4 experts, so
// only the W13-idle workers' share of W2 moves (all 12 tiles at 2 experts,
// 6 of 18 at 3, none at 4+), and those tiles then compete with the W13
// stragglers for HBM. This is the ceiling of TileRT-style stage streaming
// for the layer's biggest phase pair here, and it is negative.
//
// Kept at 0, which is a strict no-op: _pipe_tiles folds to 0 and the Phase 7
// skip disappears. Arm 0 measured 10.244 against the 10.25 batch baseline.
//
// VERIFICATION NOTE: the evidence that the probe was live is not a
// disassembly diff -- it is that arm 2 emits deterministic garbage ("The
// capital of France is<think>criptsosas variableso...") identical across all
// three reps, while arm 1 emits the control's known-good continuation. A
// compiled-out probe cannot do that.
#ifndef MPK_ABL_PIPE_W13W2
#define MPK_ABL_PIPE_W13W2 0
#endif
// (the MPK_MOE_LIVE_BOUND dependency is asserted at the use site --
// gang_oproj_router_fused_mi300.cuh is where that macro gets its default.)

// MPK_SHARED_DUP: the shared-expert makespan pricing probe. CORRECT OUTPUT,
// unlike every other ceiling probe in this file -- it is ADDITIVE, not an
// ablation, and that is the whole point.
//
// WHAT IT DOES. EP_SHARED_PE (rank 0) owns the shared expert on top of its
// routed share. Setting this to 1 gives that expert TWO consecutive slots in
// rank 0's owned tile subsequence for W13, so every one of its W13 tiles runs
// twice. Both copies read the same weights and the same activation, compute
// the same bits, and store them to the same address; W13's epilogue is a plain
// (or write-through) store, not an accumulate, so the second store is
// idempotent and the layer's output is UNCHANGED. Rank 0 simply does one
// expert's worth of extra W13 work per layer, and no other rank does anything
// different.
//
// W13 ONLY, and that is not a simplification -- it is a correctness
// requirement. W2's epilogue under FUSE_MULSUMADD is an f32 atomicAdd into
// moe_workspace_f32. A duplicated W2 tile would add the shared expert's
// contribution twice. Do not extend DUP_SHARED to the W2 call site.
//
// WHAT IT MEASURES. Two models of the shared-expert imbalance predict
// different walls, and the ledger has held both:
//
//   "absorbed"       -- the shared expert's tiles occupy workers that would
//                       otherwise be idle, every rank runs the same number of
//                       grid-stride rounds, so the extra expert costs ZERO
//                       makespan. Predicts ~0 ms.
//   "on the path"    -- rank 0 is the MoE straggler, its peers idle at the
//                       next cross-rank rendezvous for exactly as long as it
//                       runs over, and W13 carries +6.02 us/layer of that.
//                       Predicts about +6.0 us/layer = +0.46 ms/token.
//
// 0.46 ms is 1.8x the 0.26 ms wall noise floor, so n=3 resolves it. Decide on
// 1 vs 0 with nothing else changed.
//
// HOW TO READ THE RESULT. Per glm-additive-probes-overprice-deletions, adding
// work costs about 1:1 while removing the same work saves about nothing, four
// pairs measured. So a POSITIVE result here is an UPPER bound on what
// eliminating rank 0's W13 excess could buy -- necessary evidence for building
// a shared-expert K-shard, not sufficient. A NULL result is decisive the other
// way: if rank 0 can absorb a whole extra expert of W13 for free, there is no
// makespan there to harvest and the hoist/shard family is closed for good.
//
// This exists because glm-shared-expert-hoist-is-zero-makespan measured
// "absorbed" on 2026-08-21 at 11:44 and the stage-stamp drop-guard bug was
// found at 12:00 the same day, sixteen minutes later. That closure was never
// re-derived, and re-running its own max-max statistic on clean logs now puts
// rank 0's W13 LONGEST of the eight (18.11 vs 12.27 us) rather than shortest
// (8.55). This probe settles it at the wall instead of at an instrument.
#ifndef MPK_SHARED_DUP
#define MPK_SHARED_DUP 0
#endif

// ── MPK_MOE_SHARED_KSHARD: shard the shared expert across the EP ranks ──────
//
// The number of slices to cut the shared expert into; 0 (default) keeps it
// whole on EP_SHARED_PE. Capped at 4 by W2's pipeline (see below), so at
// EP_WORLD_SIZE = 8 it runs on ranks 0..3 and the rest skip the shared expert
// altogether. This is the fix for the imbalance MPK_SHARED_DUP was built
// to price: EP_SHARED_PE carries its routed share PLUS the whole shared
// expert, which measured 19.418 us/layer of MoE against rank 2's 13.716 --
// 5.702 us/layer, 0.367 ms/token, all of it on the critical path because the
// barrier after W2 waits for the slowest rank.
//
// Rank r takes intermediate slice [r*I/W, (r+1)*I/W) of the shared expert:
//
//   W13  its tiles partition the intermediate, so rank r simply runs WGS/W of
//        them. Gate and up are pairwise interleaved, so the slice is a
//        contiguous output-row range and the shard is a flat tile offset.
//        Rank r writes its slice into the same offset of the rank-LOCAL
//        swiglu scratch its own W2 reads back, so no exchange is needed.
//   W2   its tiles partition the hidden output, so every rank runs all of
//        them, each reducing only over its own slice. The partials need no
//        combine step: the epilogue atomicAdds into the f32 workspace and the
//        EP collective already sums that across ranks, both linear. The bias
//        rides the existing split-K add_bias gate and lands on rank 0 only.
//
// WHY THIS IS LEGAL AT 4 RANKS AND WAS NOT AT 8. W2's depth-4 pipeline needs
// MFMA_ITERS >= 4 and divisible by 4. GLM-5's W2 has K = 2048 and
// K_PER_MFMA = 128, so MFMA_ITERS = 16; at W = 4 the shared class gets 4,
// exactly the minimum, and at W = 8 it gets 2 and the assert fires. The
// 2026-08-21 note that closed this off was measured at 8 ranks and does not
// bind here.
//
// EXPECT LESS THAN A CLEAN 1/W. W13 shards exactly, but a W2 tile's cost is
// staging + MFMA + a 64-row atomicAdd epilogue and the shard divides only the
// first two; ranks 1..W-1 also pick up an epilogue they did not run before.
// The makespan drop is ~(W-1)/W of the VARIABLE part only. On top of that the
// shared class sits at the depth-4 minimum, where every iteration is fill
// with no steady state.
//
// CORRECTNESS IS NOT SELF-EVIDENT HERE. The earlier naive attempt at this
// passed its mechanical gate because all four ranks agreed on the same wrong
// answer. Gate on generated text or on MPK_BS_DEBUG=2 checksums against a
// build with this off, not on rank agreement. Note that checksums will NOT be
// bit-identical across the knob and must not be gated as such: the shard
// changes how the reduction is partitioned, so the low bits move legitimately.
//
// ── MEASURED 2026-09-02: CORRECT, AND A LOSS. DEFAULT STAYS 0. ─────────────
//
//   arm         per-run decode min, ms      n=3, NP=4, one -D apart
//   control     8.771 / 8.761 / 8.763       mean 8.765
//   kshard=4    8.987 / 9.028 / 9.020       mean 9.012   **+0.247 ms**
//
// Text was coherent and correct on the arm, so this is a real measurement of
// a real implementation and not a garbage-output artifact. The per-iteration
// MIN is the statistic: this box injects multi-second stalls that ruined one
// run of each arm (117 ms and 931 ms averages), and the min is immune to them.
//
// The mechanism is in the long note above the W13 kernel in
// gang_moe_linear_mxfp8_mi300.cuh. Short version: at the shipping tile widths
// rank 0's three experts already fit in ONE grid-stride round of the 30
// workers per XCD, so the shared expert runs on otherwise-idle workers and is
// nearly free in makespan. The shard does not take work off the critical path,
// it adds a slower tile class to every rank's round.
//
// Kept rather than deleted: it is the only worked example in the tree of a
// cross-rank K-shard with collective-summed partials, and it is the control
// that shows byte-rebalancing is not the lever while the round count holds.
#ifndef MPK_MOE_SHARED_KSHARD
#define MPK_MOE_SHARED_KSHARD 0
#endif
// The MPK_MOE_LIVE_BOUND dependency is checked where that macro is defined,
// in gang_oproj_router_fused_mi300.cuh, which is included after this header.
#if MPK_MOE_SHARED_KSHARD > 0
#if MPK_SHARED_DUP
#error "MPK_MOE_SHARED_KSHARD and MPK_SHARED_DUP both rewrite the shared expert's slot run; DUP is a pricing probe for the imbalance the shard removes, so they answer the same question and must not be combined"
#endif
#if MPK_ABL_PIPE_W13W2
#error "MPK_MOE_SHARED_KSHARD and MPK_ABL_PIPE_W13W2 both re-derive W2 tile bounds; the probe moves tiles off a bound that no longer covers the shared class"
#endif
#endif

#if MPK_ML_PTR_PREFETCH || MPK_ABL_ML_BOUNDARY
#define MPK_ML_PF 1
#else
#define MPK_ML_PF 0
#endif
// Tried 2026-10-15 and not kept: MPK_ML_PF's loads issued behind the layer
// instead of in front of it. In front, the tile loop's header vmcnt(0) waits
// them out on every worker (sparse trace S11 -> S10 0.48 us). Behind is not
// reachable: any load after _execute_gang_task returns, inline asm included,
// gets the join's vmcnt(0) ahead of it (final ISA); and handing them out of
// the fused layer's end needs LDS the worker does not have -- 8176 B static
// + 155648 B dynamic leave 16 B of the CU's 160 KB, and a 488 B transport
// block fails at dispatch (HSA_STATUS_ERROR_INVALID_ALLOCATION). Nor would it
// pay where it counts: the qkv_a tiles' S54 is the same (+-0.05 us) whether
// their worker left the previous layer's W2 2.2, 3.5 or 9.4 us earlier, so
// the boundary does not start qkv_a.

// MPK_OPROJ_MXFP4: o_proj's weight is E2M1 nibbles rather than E4M3 bytes.
// Set by GLM_OPROJ_MXFP4, which also switches the host packer -- the two must
// agree. Off by default so MXFP8 stays the reference.
#ifndef MPK_OPROJ_MXFP4
#define MPK_OPROJ_MXFP4 0
#endif

#ifndef MPK_BAR_SKEW
#define MPK_BAR_SKEW 0
#endif
#if MPK_BAR_SKEW
#define MPK_BAR_SKEW_SLOTS 16
// MPK_BAR_SKEW=2 runs the barrier stamp block TWICE, the second copy into
// scratch. The per-phase gap difference between =2 and =1 is the probe's own
// cost per barrier, which has to come off the gap before any part of it is
// called unexplained -- the stamp sits on the last arriver's critical path,
// immediately before it fans the release out.
#define MPK_BAR_SKEW_REPS (MPK_BAR_SKEW)
__device__ unsigned long long g_barskew_first[MPK_BAR_SKEW_SLOTS];
__device__ unsigned long long g_barskew_ns[MPK_BAR_SKEW_SLOTS];
__device__ unsigned long long g_barskew_cnt[MPK_BAR_SKEW_SLOTS];
__device__ unsigned long long g_barskew_drop[MPK_BAR_SKEW_SLOTS];
// Gap from the PREVIOUS rendezvous's last arrival to this one's. The layer's
// barriers complete in order, so these gaps are the per-phase makespans and
// they sum to the layer. The first->last spread above cannot do this job: a
// worker with no tile in the phase waits at the *next* barrier for the whole
// layer, so its spread reads as one full layer and the slots do not sum.
__device__ unsigned long long g_barskew_gap[MPK_BAR_SKEW_SLOTS];
__device__ unsigned long long g_barskew_prevlast;
// Reference clock for the intra-phase stage stamps: the layer-entry barrier's
// last arrival. Written by slot 0's last arriver before it fans the release
// out, so every worker it releases reads a value from its own layer.
__device__ unsigned long long g_stage_ref;
// 0..4 are the attention-half / EP-collective stamps, 5..8 the MoE half,
// 9..11 the layer boundary, and
// the LAST slot is the MPK_BAR_SKEW=2 scratch that prices the probe itself.
// Every stamp is measured from the LAYER-ENTRY barrier's last arrival on this
// rank, so a stamp's value is its absolute offset into the layer and stamps
// can be differenced against the cumulative BAR_SKEW gap sums. They can NOT
// be compared across ranks -- the eight entry barriers are not synchronized,
// which is the mistake recorded on MPK_EP_POLL_BATCH.
//
// 36..43 are the BARRIER EXIT bank, added 2026-08-23. Every other stamp in
// this map records an ARRIVAL; nothing has ever recorded when a worker
// actually LEAVES a rendezvous, and without that the only available notion of
// "worker w's work" is arr_w(B_i) minus a MAX-over-workers of arrivals at
// B_{i-1} -- a mean minus a max-of-means, 18.2% of whose entries come out
// negative (glm-barskew-per-worker-sum-is-not-busy-time). That defect is the
// whole reason this bank exists.
//
// An exit stamp is taken by the SAME worker, in the SAME epoch, against the
// SAME g_stage_ref as its own arrival stamp, so
//
//     spin_w(B)    = exit_w(B)      - arrival_w(B)
//     busy_w(phase)= arrival_w(B_i) - exit_w(B_{i-1})
//
// are both differences of two per-worker means over an identical layer
// population. They are true durations and they cannot go negative.
//
//   36 entry_bar      full_layer:711     40 attn_release   full_layer:1782
//   37 qkv_barrier    attn:743           41 wuv_barrier    oproj:552
//   38 qb_barrier     attn:1183          42 hier_barrier   oproj:925
//   39 decode_barrier attn:1411          43 w13_barrier    oproj:1595
//
// The two the deliverable needs: qkv_a is (37 arrival = stamp 17) against
// exit 36, and o_proj is (hier arrival = stamp 30) against exit 41.
// rel_tree and the routing poll are polls rather than hier_barrier_arrive
// call sites and are NOT in this bank; neither bounds qkv_a or o_proj.
#define MPK_STAGE_SLOTS 46
// Samples longer than this are treated as stale-reference and dropped. See
// mpk_stage_stamp. 10 ms is the historical value and keeps old runs
// reproducible; 1 ms is what you want for a ~136 us layer.
#ifndef MPK_BAR_SKEW_DROP_NS
#define MPK_BAR_SKEW_DROP_NS 10000000ull
#endif
// PRIVATE per-worker accumulators, not four contended atomics.
//
// The first version of this stamp did atomicAdd/Min/Max on four GPU-wide
// addresses. That is 232-way contention on the same cache line, and it is not
// a small effect: adding two stamps moved the wall 12.862 -> 14.260, ~9 us per
// stamp per layer. Every stamp therefore sat INSIDE the interval measured by
// the next one, and the headline "the layer boundary costs 20.2 us" was
// partly the cost of stamp 8's own atomics. One worker owns one block, so
// blockIdx.x indexes a private row and the four updates become plain stores
// with no coherence traffic at all. Reduced by the single printing thread.
// 512 = MAX_NUM_WORKERS in runtime_header.h, which this header does not
// include. Kept as a literal with the guard below rather than pulling in the
// dependency. Must track it: at 2 blocks/CU the worker count can exceed the
// 256 CUs, and a short row here is an out-of-bounds store from a high worker.
#define MPK_STAGE_WORKERS 512
__device__ unsigned long long g_stage_psum[MPK_STAGE_WORKERS *MPK_STAGE_SLOTS];
__device__ unsigned long long g_stage_pcnt[MPK_STAGE_WORKERS *MPK_STAGE_SLOTS];
__device__ unsigned long long g_stage_pmin[MPK_STAGE_WORKERS *MPK_STAGE_SLOTS];
__device__ unsigned long long g_stage_pmax[MPK_STAGE_WORKERS *MPK_STAGE_SLOTS];

// One worker's arrival at a named point inside a phase, in ns since the
// layer-entry barrier completed. Call from tid == 0 only. The min across
// workers is the earliest anyone got there, which is what the barrier's
// "first arriver" sees; the mean is where the bulk is.
__device__ __forceinline__ void mpk_stage_stamp(int idx) {
  unsigned long long const t =
      (unsigned long long)__builtin_amdgcn_s_memrealtime() & 0xFFFFFFFFFFull;
  unsigned long long const ref = __hip_atomic_load(
      &g_stage_ref, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  if (ref == 0ull || t <= ref) {
    return;
  }
  unsigned long long const d = (t - ref) * 10ull;
  // A sample longer than the drop threshold means the reference was updated
  // underneath us (this worker is a layer behind). Drop rather than skew the
  // mean.
  //
  // The original 10 ms was far too loose. A layer is ~136 us, so at 10 ms a
  // single stale sample survives and lands with ~10000x the weight of a real
  // one. That is not hypothetical: it is what made the layer boundary read as
  // 20.18 us/layer. Slots stamped every ml keep one sample per DECODE
  // ITERATION whose reference is the previous iteration's last barrier -- a
  // ~1.2 ms gap -- while slots stamped only for ml > 0 do not, so
  // differencing the two families reported that whole gap, amortised, as
  // boundary time. Set MPK_BAR_SKEW_DROP_NS=1000000 (1 ms) to cut those and
  // equalise the populations; the default is unchanged so older numbers stay
  // reproducible.
  if (d > (unsigned long long)MPK_BAR_SKEW_DROP_NS) {
    return;
  }
  unsigned int const w = blockIdx.x;
  if (w >= MPK_STAGE_WORKERS || idx >= MPK_STAGE_SLOTS) {
    return;
  }
  unsigned int const p = w * MPK_STAGE_SLOTS + (unsigned int)idx;
  // Plain loads and stores: this row belongs to this block and only tid 0
  // writes it, so there is no race and nothing to make coherent. g_stage_pmin
  // has no ~0 initializer (3952 entries), so the count doubles as the
  // "unset" flag.
  unsigned long long const c = g_stage_pcnt[p];
  g_stage_psum[p] += d;
  g_stage_pcnt[p] = c + 1ull;
  if (c == 0ull || d < g_stage_pmin[p]) {
    g_stage_pmin[p] = d;
  }
  if (d > g_stage_pmax[p]) {
    g_stage_pmax[p] = d;
  }
  // Probe-cost attribution. At MPK_BAR_SKEW=2 every stamp does its work
  // twice, the second set into a scratch slot: the shift in a LATER stage's
  // mean between reps=1 and reps=2 is exactly the cost of one extra stamp on
  // the path in front of it. This mattered enormously when the four updates
  // were GPU-wide atomics; with private rows it should now read ~0, which is
  // the check that the rewrite worked.
#pragma unroll 1
  for (int _r = 1; _r < MPK_BAR_SKEW_REPS; _r++) {
    unsigned int const q = w * MPK_STAGE_SLOTS + (MPK_STAGE_SLOTS - 1);
    g_stage_psum[q] += d;
    g_stage_pcnt[q] += 1ull;
    g_stage_pmax[q] = d;
  }
}
#elif defined(MPK_TRACE) && MPK_TRACE
// MPK_TRACE: every stage stamp of decode iterations [MPK_TRACE_T0,
// + MPK_TRACE_ITERS) as the raw low 32 bits of s_memrealtime (10 ns ticks),
// one fire-and-forget store per stamp, into [iter][layer][stamp][worker].
// MPK_BAR_SKEW's accumulators cost a dependent load-modify-store each (~1 us,
// 8.8 against 5.9 ms/token) and their per-worker means mix in the first
// layer's samples, whose reference predates the prologue. Dumped by the host
// after the kernel returns (MPK_TRACE_OUT.rank<r>.bin); unwritten entries
// stay 0.
#ifndef MPK_TRACE_T0
#define MPK_TRACE_T0 1600
#endif
#ifndef MPK_TRACE_SPARSE
#define MPK_TRACE_SPARSE 0
#endif
#define MPK_TRACE_ITERS 4
#define MPK_TRACE_LAYERS 80
// 46 shared with MPK_BAR_SKEW, then trace-only sub-phase stamps 46..79.
// MPK_TRACE_SLOTS > 80 makes room for stamps past 79 (q_b's k-loop, 80..85).
#ifndef MPK_TRACE_SLOTS
#define MPK_TRACE_SLOTS 80
#endif
#define MPK_STAGE_SLOTS MPK_TRACE_SLOTS
#define MPK_STAGE_WORKERS 512
#define MPK_TRACE_WORDS                                                        \
  ((size_t)MPK_TRACE_ITERS * MPK_TRACE_LAYERS * MPK_STAGE_SLOTS *              \
   MPK_STAGE_WORKERS)
__device__ unsigned g_mpk_trace[MPK_TRACE_WORDS];
// (iter - T0) * MPK_TRACE_LAYERS + ml while a traced layer runs, else -1.
// Written and read by tid 0 only, so no join.
__shared__ int s_mpk_trace_slot;
__device__ __forceinline__ void mpk_trace_set_slot(int slot) {
  s_mpk_trace_slot = slot;
}
// MPK_TRACE_SPARSE=L: only layers L and L + 1 are stamped, in each of
// MPK_TRACE_ITERS * MPK_TRACE_LAYERS / 2 decode iterations from MPK_TRACE_T0
// (slot 2 * (iter - T0) + (ml - L)), and the stamps go to LDS -- the global
// stores of the dense trace count in vmcnt, so every stamp ahead of a poll's
// vmcnt(0) delayed that poll by a store round trip, and by different amounts
// in different designs (a fold that traced 1.3 us/layer faster measured
// 0.03 ms slower). Layer L + 2 flushes the 2 x 80 words to the same
// [slot][stamp][worker] buffer; the other layers run untraced. Cost
// 2026-10-13: traced decode 3.686 against 3.50 untraced (+5%; the dense
// trace +6-8%). m_sparse_chain.py reads the dump.
#if MPK_TRACE_SPARSE > 0
__shared__ unsigned s_mpk_tl[2][MPK_STAGE_SLOTS];
#endif
__device__ __forceinline__ void mpk_stage_stamp(int idx) {
  unsigned const t = (unsigned)__builtin_amdgcn_s_memrealtime();
  int const slot = s_mpk_trace_slot;
  unsigned const w = blockIdx.x;
  if (slot < 0 || w >= MPK_STAGE_WORKERS) {
    return;
  }
#if MPK_TRACE_SPARSE > 0
  s_mpk_tl[slot & 1][idx] = t;
#else
  __builtin_nontemporal_store(
      t, &g_mpk_trace[((size_t)slot * MPK_STAGE_SLOTS + idx) *
                          MPK_STAGE_WORKERS +
                      w]);
#endif
}
#if MPK_TRACE_SPARSE > 0
// The traced slot for layer ml of decode iteration it, else -1.
__device__ __forceinline__ int mpk_trace_sparse_slot(long it, int ml) {
  long const ti = it - MPK_TRACE_T0;
  int const d = ml - MPK_TRACE_SPARSE;
  return (ti >= 0 && ti < MPK_TRACE_ITERS * MPK_TRACE_LAYERS / 2 &&
          (d == 0 || d == 1))
             ? (int)ti * 2 + d
             : -1;
}
// Called by every thread after a block-wide barrier at layer ml's top: at
// ml == L + 2, threads 0..159 move the two traced layers' stamps out of LDS
// and clear them (unwritten stamps stay 0, as in the dense trace).
__device__ __forceinline__ void mpk_trace_sparse_flush(long it, int ml) {
  int const tid = threadIdx.x;
  unsigned const w = blockIdx.x;
  if (ml != MPK_TRACE_SPARSE + 2 || tid >= 2 * MPK_STAGE_SLOTS ||
      w >= MPK_STAGE_WORKERS) {
    return;
  }
  int const s0 = mpk_trace_sparse_slot(it, MPK_TRACE_SPARSE);
  if (s0 < 0) {
    return;
  }
  int const row = tid / MPK_STAGE_SLOTS;
  int const idx = tid % MPK_STAGE_SLOTS;
  unsigned const v = s_mpk_tl[row][idx];
  if (v != 0u) {
    __builtin_nontemporal_store(
        v, &g_mpk_trace[((size_t)(s0 + row) * MPK_STAGE_SLOTS + idx) *
                            MPK_STAGE_WORKERS +
                        w]);
  }
  s_mpk_tl[row][idx] = 0u;
}
#endif
// MPK_LMH_TRACE: under MPK_TRACE, the LM head behind its call wall stamps
// its own entry / resolve / k-loop / exit into the layer slot the loop never
// fills (MPK_TRACE_LAYERS - 1) of each traced iteration, a worker's first
// tile at 0..4 and a second tile at 5..9. Its calls are counted per worker,
// since the head cannot see the iteration.
#if MPK_LMH_TRACE
__device__ unsigned g_lmh_tr_calls[MPK_STAGE_WORKERS];
__device__ __forceinline__ void mpk_lmh_stamp(int slot, int idx) {
  unsigned const t = (unsigned)__builtin_amdgcn_s_memrealtime();
  if (slot >= 0) {
    __builtin_nontemporal_store(
        t, &g_mpk_trace[((size_t)slot * MPK_STAGE_SLOTS + idx) *
                            MPK_STAGE_WORKERS +
                        blockIdx.x]);
  }
}
#endif
#else
__device__ __forceinline__ void mpk_stage_stamp(int) {}
#endif
#if !(defined(MPK_TRACE) && MPK_TRACE) || MPK_BAR_SKEW
__device__ __forceinline__ void mpk_trace_set_slot(int) {}
#endif

// MPK_ML_BOUNDARY_PAD: nanoseconds of pure delay injected into the multi-layer
// loop's per-layer boundary, paid by every thread of every worker.
//
// This is a PRICING probe, not a ceiling probe: it ADDS uniform time to a
// region instead of deleting it, so the output stays correct and the wall
// number can be gated by the correctness suite. It exists because the
// subtractive form of this experiment is not available -- skipping the
// TaskDesc pointer refresh makes all 78 layers reuse layer 0's weights, which
// turns a 65 MB/layer HBM stream into an L2 hit and would report a huge
// spurious win (the failure mode recorded in
// glm-wrong-output-probes-upstream-of-router-are-invalid, in reverse).
//
// What it measures: the SLOPE of wall against boundary time. Every worker pays
// the same pad and the layer's critical path runs through GPU-wide barriers,
// so if boundary time is on the critical path the wall must rise by
// pad * layers * iters, one for one. If the boundary instead sits inside slack
// that the entry barrier's arrival spread already absorbs, the slope is near
// zero. The slope is an UPPER bound on what deleting the real 15.33 us of
// boundary bookkeeping could buy, because a uniform addition costs at least as
// much as a uniform deletion saves.
//
// RETRACTED 2026-08-21. The last sentence is false, and MPK_ABL_ML_BOUNDARY
// above is the subtractive form this note claimed was unavailable: deleting
// the boundary's cost while leaving the byte stream alone buys 0.051 ms, not
// the ~1.4 ms this slope priced. A uniform addition in front of a barrier
// costs more than the matching deletion saves, because the deletion is
// re-absorbed by the barrier's arrival spread.
#ifndef MPK_ML_BOUNDARY_PAD
#define MPK_ML_BOUNDARY_PAD 0
#endif

#if MPK_ML_BOUNDARY_PAD > 0
// s_memrealtime is the 100 MHz constant-rate counter -- one tick is 10 ns,
// the same conversion mpk_stage_stamp uses. s_sleep(1) parks the wave for
// ~64 clocks so the spin does not saturate the scalar unit while every other
// wave on the SIMD is doing the same thing.
__device__ __forceinline__ void mpk_ml_boundary_pad() {
  unsigned long long const t0 =
      (unsigned long long)__builtin_amdgcn_s_memrealtime();
  unsigned long long const ticks = (unsigned long long)MPK_ML_BOUNDARY_PAD / 10;
  while (((unsigned long long)__builtin_amdgcn_s_memrealtime() - t0) < ticks) {
    __builtin_amdgcn_s_sleep(1);
  }
}
#else
__device__ __forceinline__ void mpk_ml_boundary_pad() {}
#endif

// MPK_BAR_TREE: use the two-level arrival above for every GPU-wide
// Mechanism-C rendezvous. Off by default so the flat mechanism stays the
// reference; the counter buffer is sized for the tree either way, so this is
// a pure A/B.
//
// MEASURED TWICE, NULL BOTH TIMES -- and the second measurement is worth more
// than the null, because it prices the arrival atomic directly.
//
//   NP=8:  -0.001 ms.  Dismissed at the time on the grounds that "at a real
//          barrier the arrivals are already spread over microseconds by
//          upstream skew", so the 232-way same-address contention the
//          standalone null-probe measured (3.77 -> 2.11 us, -44%) never
//          actually forms.
//
//   That reason turned out to be FALSE. The makespan predictor measured the
//   top of the arrival distribution two days later and found it FLAT: max
//   minus second-arrival is 0.005-0.369 us at all ten real rendezvous, and the
//   workers within 1 us of max are 232 of 232 at wuv_barrier and 192 of 192 at
//   hier_barrier. That is precisely the null-probe's condition, so the tree
//   should have paid ~1.66 us x 10 barriers x 78 layers = 1.29 ms.
//
//   NP=4:  -0.015 ms on per-iter min (10.265 n=5, 10.244..10.299, against a
//          control of 10.280 n=5, 10.248..10.335) and -0.021 on clean avg.
//          Same sign, both an order of magnitude under the 0.26 ms noise
//          floor, distributions almost entirely overlapping.
//
// So the arrival atomic is not what a rendezvous costs. Divide it out: 10
// barriers x 78 layers is 780 arrivals per iteration, and the tree's whole
// effect is 15 us of wall, i.e. it removes about 19 NANOSECONDS per barrier.
// A 232-way device-scope atomic on one address does not serialize 232 ways --
// the L2 atomic unit pipelines same-address adds -- so there was never 1.66 us
// there to take. The standalone probe measured a loop of nothing but atomics,
// where issue rate is the only thing left to measure.
//
// What that leaves is the RELEASE half, which this flag does not touch: the
// single global last arriver still does `for (x = 0; x < 8; x++) st_wt_u32`
// out to eight per-XCD flags, and every worker then observes a line written
// from a different XCD's L2. At 3.77 us measured per rendezvous against 0.019
// us of arrival, essentially all of a barrier is release fan-out plus
// cross-XCD observation latency. Any further work here belongs on the release
// side; see glm-one-rendezvous-costs-3.77us.
#ifndef MPK_BAR_TREE
#define MPK_BAR_TREE 0
#endif

// MPK_BAR_TAGGED: a bitmask of GLM fused-layer rendezvous to run as tag
// arrays instead of Mechanism C. Each arriving worker stores the epoch into
// its own int32 slot with a write-through store; each waiting worker's wave 0
// polls every slot (one system-scope dwordx4 per lane) until all of them
// carry the epoch. No arrival atomic, no elected last arriver, no eight-flag
// fan-out, no self-heal -- there is no single store whose loss wedges a
// round. The fences either side are the call site's and are unchanged.
//
// Priced first as a null rendezvous (MPK_NULL_TAGGED), 1024/1024, NP=4,
// 2 alternating reps, decode median, all G1 PASS:
//   control 9.857   +4 Mechanism-C nulls 10.931   +4 tagged nulls 10.471
//   i.e. 3.44 vs 1.97 us per rendezvous over 4 x 78 added.
//
// At the five real sites (mask 31, MPK_BAR_TAGGED_HIER=1), 1024/1024, NP=4,
// 3 alternating reps, all G1 PASS, no overlap between arms:
//   decode median  control 9.869 9.861 9.840   tagged 9.742 9.717 9.688
//   mean 9.857 -> 9.716 ms (-0.141), prefill 9.264 -> 9.138 ms/iter
// About 0.36 us per rendezvous, a quarter of what the null probe priced.
// Gates green with it on: standalone 46/46 + 6/6 + 4/4, ppl512 2.5723 (4
// ranks identical), ppl128 ratio 1.153 vs torch, longseq 256/512/1024 G1.
//
// Only legal where every epoch is task_layer_idx + 1 (ml_mode): a site whose
// release value is snapshotted off its own Mechanism-C flag would never see
// that flag advance once the site stops writing it. Outside ml_mode the
// fused layer passes no slot arrays and every site stays Mechanism C.
// 30, not 31: the entry rendezvous is back on Mechanism C, where a skipping
// worker leaves no one stranded (MPK_ENTRY_NOWAIT).
#ifndef MPK_BAR_TAGGED
#define MPK_BAR_TAGGED 30
#endif
#define MPK_TAGBAR_ENTRY 1
#define MPK_TAGBAR_QKV 2
#define MPK_TAGBAR_DECODE 4
#define MPK_TAGBAR_ATTN 8
#define MPK_TAGBAR_W13 16
// MPK_TAG_DIRECT_SITES: MPK_TAGBAR_* bits whose waiters poll the tags directly
// (see tag_bar_wait_site). Trace 7: the merge starts 2.3 us after the last
// decode item behind the two-hop decode -> merge release. MEASURED NEGATIVE
// 2026-10-11, NP=8 1024/1024 (control 4.240): decode site 4.368, qkv site
// 4.485, both 4.459 ms -- the waiters' tag polling costs far more than the
// release hop. MPK_QB_SELF_PF re-measured on the same stack: 4.238 (null).
#ifndef MPK_TAG_DIRECT_SITES
#define MPK_TAG_DIRECT_SITES 0
#endif
// One region per rendezvous, in the order below: 256 int32 slots, then eight
// 128-byte per-XCD release lines (MPK_BAR_TAGGED_HIER).
#define MPK_TAGBAR_SLOTS 512
#define MPK_TAGBAR_XFLAG_OFF 256
// MPK_BAR_TAGGED_HIER: who polls the slot array.
//   0  every waiting worker's wave 0. MEASURED WORSE than Mechanism C at the
//      five real sites (1024/1024, 3 alternating reps, decode median
//      9.846 -> 10.269 ms, all G1 PASS), though it won the null probe: at a
//      real rendezvous the early arrivers poll 1 KB each, continuously, into
//      the same eight lines while the stragglers they wait for are still
//      streaming, where Mechanism C polls one 4-byte flag from one lane.
//   1  one worker per XCD (xcd_rank 0) polls the slot array, then publishes
//      the epoch to its XCD's line with a plain store, which lands in that
//      XCD's L2; everyone else polls that line with an nt load (L1 miss, L2
//      hit). Eight memory pollers instead of 240, and no cross-XCD fan-out.
#ifndef MPK_BAR_TAGGED_HIER
#define MPK_BAR_TAGGED_HIER 1
#endif
enum {
  MPK_TAGBAR_IDX_ENTRY = 0,
  MPK_TAGBAR_IDX_QKV = 1,
  MPK_TAGBAR_IDX_DECODE = 2,
  MPK_TAGBAR_IDX_ATTN = 3,
  MPK_TAGBAR_IDX_W13 = 4,
  MPK_TAGBAR_ARRAYS = 8
};

// Caller: tid 0, after the site's own __syncthreads + s_waitcnt vmcnt(0).
__device__ __forceinline__ void tag_bar_arrive(int *tags, int slot, int epoch) {
  st_wt_u32((void *)&tags[slot], (unsigned)epoch);
}

// Caller: the 64 lanes of wave 0, all of them. Returns once slots
// [0, arrivals) all hold >= epoch.
__device__ __forceinline__ void tag_bar_wait(int *tags, int arrivals,
                                             int epoch, int lane) {
  int const nchunk = (arrivals + 4 * 64 - 1) / (4 * 64);
  for (int c = 0; c < nchunk; c++) {
    int const base = c * 4 * 64 + lane * 4;
    bool const live = base < arrivals;
    int4 v = make_int4(0, 0, 0, 0);
    for (;;) {
      if (live) {
        asm volatile("global_load_dwordx4 %0, %1, off sc0 sc1\n"
                     "s_waitcnt vmcnt(0)"
                     : "=v"(v)
                     : "v"(&tags[base])
                     : "memory");
      }
      bool const ok = !live || ((v.x >= epoch) &&
                                (base + 1 >= arrivals || v.y >= epoch) &&
                                (base + 2 >= arrivals || v.z >= epoch) &&
                                (base + 3 >= arrivals || v.w >= epoch));
      if (__all(ok)) {
        break;
      }
      __builtin_amdgcn_s_sleep(1);
    }
  }
}

// What a converted site calls, from every thread of the workgroup, between
// its tag_bar_arrive and its own trailing __syncthreads + buffer_inv.
// `region` is the rendezvous' MPK_TAGBAR_SLOTS-int region.
// `direct`: every waiter's wave 0 polls the tags itself instead of the
// xcd_rank-0 release line -- one hop instead of two, for a site with few
// waiters. Every waiter of a site must agree, since nobody then writes its
// release line.
__device__ __forceinline__ void tag_bar_wait_site(int *region, int arrivals,
                                                  int epoch, int tid,
                                                  int xcd_id, int xcd_rank,
                                                  bool direct = false) {
  if (direct) {
    if (tid < 64) {
      tag_bar_wait(region, arrivals, epoch, tid);
    }
    return;
  }
#if MPK_BAR_TAGGED_HIER
  // 128 B apart: one L2 line per XCD, so no two XCDs' L2s hold the same line.
  int *const xflag = region + MPK_TAGBAR_XFLAG_OFF + xcd_id * 32;
  if (tid == 0) {
    // The release line is only coherent through the L2 of the XCD that owns
    // it, so a gang tile whose xcd_id is not the XCD it runs on would poll a
    // line nobody on its own L2 ever writes and spin forever. Fail loudly.
    int hw;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 4)" : "=s"(hw));
    if (hw != xcd_id) {
      __builtin_trap();
    }
  }
  if (xcd_rank == 0) {
    if (tid < 64) {
      tag_bar_wait(region, arrivals, epoch, tid);
    }
    if (tid == 0) {
      asm volatile("global_store_dword %0, %1, off"
                   :
                   : "v"(xflag), "v"(epoch)
                   : "memory");
    }
  } else if (tid == 0) {
    int v;
    for (;;) {
      // `nt`, NOT `sc0`: a group-scope load may HIT in the CU's L1 (the ISA
      // load table's Group row is "Hit LRU" there), so a poller on another CU
      // re-reads its own stale copy forever. Measured standalone, 240 x 256
      // on one MI350X: plain store + sc0 poll, 232 of 232 cross-CU pollers
      // timed out; plain store + nt poll, 232 of 232 saw it.
      asm volatile("global_load_dword %0, %1, off nt\n"
                   "s_waitcnt vmcnt(0)"
                   : "=v"(v)
                   : "v"(xflag)
                   : "memory");
      if (v >= epoch) {
        break;
      }
      __builtin_amdgcn_s_sleep(1);
    }
  }
#else
  (void)xcd_id;
  (void)xcd_rank;
  if (tid < 64) {
    tag_bar_wait(region, arrivals, epoch, tid);
  }
#endif
}

// Offset, in HIER_STRIDE slots, from a barrier's own base to its block of
// eight per-XCD arrival counters. One uniform offset works for every barrier
// because the closest two bases in the GLM counter map are 11 slots apart and
// a tree block is 8 -- see FULL_LAYER_TREE_OFF_SLOTS, which must agree with
// this and is what the host allocation is sized against.
#define MPK_BAR_TREE_OFF 114

// One arrival at a Mechanism-C barrier. Returns true on the single thread in
// the grid that owes the release fan-out.
//
// `use_tree` has to be computed the same way at the arrival and at the
// self-heal (see hier_barrier_heal_quota) or a worker heals on a quota the
// counter never reaches and the rendezvous wedges -- that is the failure mode
// recorded in a-self-heal-must-test-the-whole-predicate. Deriving both from
// one expression at the call site is what keeps them in step. The tree is
// only legal when the population divides evenly across the eight XCDs; a
// barrier whose arrivals are not per_xcd * 8 falls back to flat.
__device__ __forceinline__ bool hier_barrier_arrive(int *bar,
                                                    int hier_stride,
                                                    int arrivals,
                                                    int per_xcd,
                                                    int xcd_id,
                                                    bool use_tree,
                                                    int skew_slot = -1) {
  if (use_tree) {
    bool const tlast = hier_barrier_tree_arrive(bar, MPK_BAR_TREE_OFF,
                                                hier_stride, per_xcd, xcd_id);
#if MPK_BAR_SKEW
    // The tree counts per XCD, so a first->last spread here would only be the
    // cross-XCD part. The gap is still exact -- `tlast` is still the global
    // last arriver -- so record that and leave the spread slot empty.
    if (tlast && skew_slot >= 0 && skew_slot < MPK_BAR_SKEW_SLOTS) {
      unsigned long long const t1m =
          (unsigned long long)__builtin_amdgcn_s_memrealtime() &
          0xFFFFFFFFFFull;
      unsigned long long const pl = atomicExch(&g_barskew_prevlast, t1m);
      if (pl != 0ull && t1m > pl) {
        atomicAdd(&g_barskew_gap[skew_slot], (t1m - pl) * 10ull);
        atomicAdd(&g_barskew_cnt[skew_slot], 1ull);
      }
    }
#else
    (void)skew_slot;
#endif
    return tlast;
  }
  int const prev = atom_add_release_gpu_s32(&bar[8 * hier_stride], 1);
  bool const last = (prev % arrivals) == arrivals - 1;
#if MPK_BAR_SKEW
  // ARRIVAL SPREAD. The whole point: a *null* rendezvous costs 2.71 us
  // (glm-round-fixed-cost-is-the-rendezvous) but a real phase's fixed cost
  // reads ~8.5 us (glm-qb-wuk-phase-is-skew-not-straggler). The difference is
  // arrival spread, and nothing has ever measured it directly. Two clock
  // reads per barrier per EPOCH -- not per worker -- so unlike
  // MPK_SUBPHASE_TIMING the cost does not scale with tile count.
  if (skew_slot >= 0 && skew_slot < MPK_BAR_SKEW_SLOTS) {
    int const ph = prev % arrivals;
    unsigned long long const epoch = (unsigned long long)(prev / arrivals);
    // The stamp carries its own epoch in the top 24 bits. Without the tag the
    // last arriver can read the PREVIOUS epoch's t0 -- the first arriver's
    // atomic is ordered, but its plain store is not, so it can land late --
    // and the spread then reads as one whole layer plus the real skew. That
    // is exactly what the first version of this instrument reported: 153 us
    // per epoch against a 136 us layer. A tag turns a wrong sample into a
    // dropped one. `t` is 100 MHz ticks, so 40 bits is ~3 hours.
    if (ph == 0) {
      unsigned long long const t0 =
          (unsigned long long)__builtin_amdgcn_s_memrealtime();
      unsigned long long const packed =
          (t0 & 0xFFFFFFFFFFull) | ((epoch & 0xFFFFFFull) << 40);
      __hip_atomic_store(&g_barskew_first[skew_slot], packed, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
    } else if (last) {
      unsigned long long const t1 =
          (unsigned long long)__builtin_amdgcn_s_memrealtime();
      unsigned long long const packed = __hip_atomic_load(
          &g_barskew_first[skew_slot], __ATOMIC_RELAXED,
          __HIP_MEMORY_SCOPE_AGENT);
      unsigned long long const t0 = packed & 0xFFFFFFFFFFull;
      unsigned long long const t1m = t1 & 0xFFFFFFFFFFull;
      if ((packed >> 40) == (epoch & 0xFFFFFFull) && t0 != 0ull && t1m > t0) {
        atomicAdd(&g_barskew_ns[skew_slot], (t1m - t0) * 10ull);
        atomicAdd(&g_barskew_cnt[skew_slot], 1ull);
      } else {
        atomicAdd(&g_barskew_drop[skew_slot], 1ull);
      }
      unsigned long long const pl =
          atomicExch(&g_barskew_prevlast, t1m);
      if (pl != 0ull && t1m > pl) {
        atomicAdd(&g_barskew_gap[skew_slot], (t1m - pl) * 10ull);
      }
      if (skew_slot == 0) {
        __hip_atomic_store(&g_stage_ref, t1m, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
      }
#pragma unroll 1
      for (int _r = 1; _r < MPK_BAR_SKEW_REPS; _r++) {
        // Identical work, discarded. Prices the block above.
        unsigned long long const t2 =
            (unsigned long long)__builtin_amdgcn_s_memrealtime() &
            0xFFFFFFFFFFull;
        unsigned long long const p2 = __hip_atomic_load(
            &g_barskew_first[skew_slot], __ATOMIC_RELAXED,
            __HIP_MEMORY_SCOPE_AGENT);
        if ((p2 >> 40) == (epoch & 0xFFFFFFull) && t2 > (p2 & 0xFFFFFFFFFFull)) {
          atomicAdd(&g_barskew_ns[MPK_BAR_SKEW_SLOTS - 1], 1ull);
        } else {
          atomicAdd(&g_barskew_drop[MPK_BAR_SKEW_SLOTS - 1], 1ull);
        }
        unsigned long long const q2 = atomicExch(&g_barskew_prevlast, t2);
        if (q2 != 0ull && t2 > q2) {
          atomicAdd(&g_barskew_gap[MPK_BAR_SKEW_SLOTS - 1], 1ull);
        }
        if (skew_slot == 0) {
          __hip_atomic_store(&g_stage_ref, t2, __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_AGENT);
        }
      }
    }
  }
#else
  (void)skew_slot;
#endif
  return last;
}

// Per-epoch increment of bar[8 * hier_stride], i.e. the multiplier the
// self-heal's "the release is owed" test compares against. Eight under the
// tree, because the global counter is bumped once per XCD rather than once
// per worker -- still the WHOLE predicate, since it only reaches 8 * epoch
// once every XCD's own counter hit its own quota.
__device__ __forceinline__ int hier_barrier_heal_quota(int arrivals,
                                                       bool use_tree) {
  return use_tree ? 8 : arrivals;
}

// Drift-immune companion to the quota test above.
//
// The quota test assumes bar[8 * HIER_STRIDE] == quota * epoch, and that is
// FALSE past ~epoch 155 on this model: EP_TAIL_ONLY's early return sits above
// the s_exp block, so one layer per generated token advances task_layer_idx
// (and therefore the epoch) without any worker arriving. The counter falls one
// full round behind per token -- measured 0 / +1 / +2 / +3 in ~76-epoch steps,
// one step per token. Every `counter >= quota * expected` heal is permanently
// false after that, so a single lost release store wedges the run forever.
// See memory glm-barrier-counter-drifts-one-round-per-token.
//
// This predicate never reads the counter. It asks a strictly stronger and
// drift-free question: has ANY OTHER XCD's release flag already reached
// `expected`? Only the elected last arriver writes those flags, and it writes
// all eight, so one peer flag at `expected` proves the election for THIS round
// fired and my own flag is owed. It therefore cannot release a round early --
// the failure mode of the obvious `counter % arrivals == 0 && my_flag ==
// expected - 1` repair, which is true at every round boundary.
//
// Returns a bitmask of the OTHER XCDs already at `expected` (0 = no evidence
// the release fired), so a caller can hand it straight to MPK_WS_WAIT_AUX
// without a second pass. `my_xcd`'s own bit is always clear -- the caller is
// spinning on it, so it carries no information.
//
// This generalizes the qb_tp arm that already existed in
// gang_mla_attn_fused_mi300.cuh; that one was written for a different reason
// (a TP barrier whose global counter is not this barrier's) but is the same
// predicate.
__device__ __forceinline__ int hier_barrier_peer_released(int *bar,
                                                          int hier_stride,
                                                          int expected,
                                                          int my_xcd) {
  // `unroll 1` is a cold-path size choice, nothing more. This is a 1-in-1024
  // path, so serial-with-early-exit costs nothing that matters and keeps the
  // eight loads from going live at once.
  //
  // It is NOT a register fix, whatever an earlier version of this comment
  // claimed. /tmp/regab.sh (an 18-second offline rebuild of the generated TU,
  // no model load, no GPU) measures worker_kernel at 325 vgpr / 69 agpr at
  // BOTH -DMPK_BAR_PEER_HEAL=0 and =1: 325 is the pre-existing baseline and
  // this helper moves it by zero. The silent launch failure that was blamed on
  // it is the co-residency wedge, which hits the exact shipping build at the
  // same rate -- see glm-launch-wedge-is-config-independent-and-precedes-
  // worker-xcd and megakernel-block-co-residency-margin.
  int mask = 0;
#pragma unroll 1
  for (int x = 0; x < 8; x++) {
    if (x != my_xcd && ld_nt_s32(&bar[x * hier_stride]) >= expected) {
      mask |= 1 << x;
      break;
    }
  }
  return mask;
}

// The heal predicate itself: the cheap counter quota OR the drift-immune peer
// scan. Kept as one helper so all seven waiters stay in step. MPK_BAR_PEER_HEAL
// defaults ON; set it to 0 to A/B back to the counter-only form.
#ifndef MPK_BAR_PEER_HEAL
#define MPK_BAR_PEER_HEAL 1
#endif
__device__ __forceinline__ bool hier_barrier_should_heal(int *bar,
                                                         int hier_stride,
                                                         int expected,
                                                         int my_xcd,
                                                         int arrivals,
                                                         bool use_tree) {
  if (ld_nt_s32(&bar[8 * hier_stride]) >=
      hier_barrier_heal_quota(arrivals, use_tree) * expected) {
    return true;
  }
#if MPK_BAR_PEER_HEAL
  return hier_barrier_peer_released(bar, hier_stride, expected, my_xcd) != 0;
#else
  return false;
#endif
}
