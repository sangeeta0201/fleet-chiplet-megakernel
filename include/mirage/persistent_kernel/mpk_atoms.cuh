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
__device__ __forceinline__ int ld_nt_s32(int *addr) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  int val;
  asm volatile("global_load_dword %0, %1, off sc0 sc1 nt\n"
               "s_waitcnt vmcnt(0)"
               : "=v"(val)
               : "v"(addr)
               : "memory");
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
  asm volatile("global_load_dwordx2 %0, %1, off sc0 sc1 nt\n"
               "s_waitcnt vmcnt(0)"
               : "=v"(val)
               : "v"(addr)
               : "memory");
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
#define MPK_STAGE_SLOTS 8
__device__ unsigned long long g_stage_sum[MPK_STAGE_SLOTS];
__device__ unsigned long long g_stage_cnt[MPK_STAGE_SLOTS];
__device__ unsigned long long g_stage_min[MPK_STAGE_SLOTS] = {
    ~0ull, ~0ull, ~0ull, ~0ull, ~0ull, ~0ull, ~0ull, ~0ull};
__device__ unsigned long long g_stage_max[MPK_STAGE_SLOTS];

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
  // A sample longer than 10 ms means the reference was updated underneath us
  // (this worker is a layer behind). Drop rather than skew the mean.
  if (d > 10000000ull) {
    return;
  }
  atomicAdd(&g_stage_sum[idx], d);
  atomicAdd(&g_stage_cnt[idx], 1ull);
  atomicMin(&g_stage_min[idx], d);
  atomicMax(&g_stage_max[idx], d);
  // Probe-cost attribution. At MPK_BAR_SKEW=2 every stamp does its atomics
  // twice, the second set into a scratch slot. The stamp is 232-way
  // contended, so its cost has to be measured, not argued: the shift in a
  // LATER stage's mean between reps=1 and reps=2 is exactly the cost of one
  // extra stamp on the path in front of it.
#pragma unroll 1
  for (int _r = 1; _r < MPK_BAR_SKEW_REPS; _r++) {
    atomicAdd(&g_stage_sum[MPK_STAGE_SLOTS - 1], d);
    atomicAdd(&g_stage_cnt[MPK_STAGE_SLOTS - 1], 1ull);
    atomicMin(&g_stage_min[MPK_STAGE_SLOTS - 1], d);
    atomicMax(&g_stage_max[MPK_STAGE_SLOTS - 1], d);
  }
}
#else
__device__ __forceinline__ void mpk_stage_stamp(int) {}
#endif

// MPK_BAR_TREE: use the two-level arrival above for every GPU-wide
// Mechanism-C rendezvous. Off by default so the flat mechanism stays the
// reference; the counter buffer is sized for the tree either way, so this is
// a pure A/B.
#ifndef MPK_BAR_TREE
#define MPK_BAR_TREE 0
#endif
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
