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

#include "arch_traits.cuh"

// True on an AMD *device* pass. Every asm block below is gated on this; the
// gfx1250 sub-branch inside each is gated on MIRAGE_ARCH_GFX1250 (set by
// arch_traits.cuh from __gfx1250__).
#define MPK_AMD_DEVICE                                                         \
  (defined(__HIP_DEVICE_COMPILE__) &&                                          \
   (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)))

// ---------------------------------------------------------------------------
// gfx1250 (MI450) note for everything in this file
// ---------------------------------------------------------------------------
//
// The gfx950 implementations below are hand-written asm for two reasons: to
// pin the exact cache-bypass bits (nt, sc0, sc1), and to keep the compiler
// from inserting its own buffer_wbl2/buffer_inv around atomics. Neither
// mnemonic nor modifier survives to gfx1250 -- `s_waitcnt`, `buffer_inv`,
// `global_load_dwordx2`, and `flat_atomic_add ... sc0 sc1` are all rejected by
// the gfx1250 assembler.
//
// Rather than transliterate the asm, the gfx1250 paths use compiler
// intrinsics, because on this target they lower to exactly what we want and
// the compiler gets the split wait-counter domains right by construction
// (verified by reading the emitted ISA):
//
//   __builtin_nontemporal_{load,store}   -> th:TH_{LOAD,STORE}_NT
//   __hip_atomic_* at _AGENT scope       -> global_atomic_* scope:SCOPE_DEV
//                                           plus global_inv / global_wb
//
// The cache-bypass *intent* differs subtly and deliberately. On MI300X/MI350
// L2 is not coherent across XCDs, so the cross-XCD protocol is built on NT and
// write-through stores that push data past L2 to HBM. On gfx1250 the same
// guarantee is expressed as scoped atomics and global_inv/global_wb, which is
// both the supported spelling and the one the memory model actually defines.
// The write-through (st_wt_*) helpers therefore become device-scope release
// stores rather than sc0/sc1 stores: same ordering guarantee, expressed in the
// vocabulary gfx1250 has.
//
// NOT VALIDATED ON SILICON. FFM-Lite is functional-only and runs workgroups
// serially or time-sliced, so it cannot observe a missing writeback or a
// too-narrow wait. These paths are correct by construction and by reading ISA;
// they are not correct by experiment.

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
#if MPK_AMD_DEVICE && defined(MIRAGE_ARCH_GFX1250)
  // -> global_load_b64 ... th:TH_LOAD_NT, with the compiler's own
  // s_wait_loadcnt before the use.
  return __builtin_nontemporal_load(addr);
#elif MPK_AMD_DEVICE
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
__device__ __forceinline__ int ld_nt_s32(int *addr) {
#if MPK_AMD_DEVICE && defined(MIRAGE_ARCH_GFX1250)
  return __builtin_nontemporal_load(addr); // -> global_load_b32 th:TH_LOAD_NT
#elif MPK_AMD_DEVICE
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

// Non-temporal store (bypasses cache, writes to memory)
__device__ __forceinline__ void st_nt_u64(unsigned long long int *addr,
                                          unsigned long long int val) {
#if MPK_AMD_DEVICE && defined(MIRAGE_ARCH_GFX1250)
  // -> global_store_b64 ... th:TH_STORE_NT. The gfx950 asm waits vmcnt(0)
  // inline; here the wait is explicit and covers stores (storecnt), which a
  // bare loadcnt wait would not -- see mirage::arch::wait_vmem().
  __builtin_nontemporal_store(val, addr);
  mirage::arch::wait_vmem();
#elif MPK_AMD_DEVICE
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
#if MPK_AMD_DEVICE && defined(MIRAGE_ARCH_GFX1250)
  // Device-scope release store: the gfx1250 spelling of "make this visible
  // outside our own cache". -> global_store_b64 ... scope:SCOPE_DEV.
  __hip_atomic_store(reinterpret_cast<unsigned long long int *>(addr),
                     val,
                     __ATOMIC_RELEASE,
                     __HIP_MEMORY_SCOPE_AGENT);
#elif MPK_AMD_DEVICE
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
#if MPK_AMD_DEVICE && defined(MIRAGE_ARCH_GFX1250)
  // No 128-bit atomic store exists; emit the wide store and follow it with a
  // device-scope writeback, which gives the same "visible past our cache"
  // guarantee that sc0 sc1 gave on gfx950.
  typedef unsigned int v4u32 __attribute__((ext_vector_type(4)));
  v4u32 zero = {0u, 0u, 0u, 0u};
  *reinterpret_cast<v4u32 *>(addr) = zero;
  mirage::arch::wait_vmem();
  mirage::arch::wb_l2();
#elif MPK_AMD_DEVICE
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
__device__ __forceinline__ void st_wt_f32x4(void *addr, float4 val) {
#if MPK_AMD_DEVICE && defined(MIRAGE_ARCH_GFX1250)
  typedef unsigned int v4u32 __attribute__((ext_vector_type(4)));
  v4u32 v;
  __builtin_memcpy(&v, &val, 16);
  *reinterpret_cast<v4u32 *>(addr) = v;
  mirage::arch::wait_vmem();
  mirage::arch::wb_l2();
#elif MPK_AMD_DEVICE
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
#if MPK_AMD_DEVICE && defined(MIRAGE_ARCH_GFX1250)
  __hip_atomic_store(reinterpret_cast<unsigned int *>(addr),
                     val,
                     __ATOMIC_RELEASE,
                     __HIP_MEMORY_SCOPE_AGENT);
#elif MPK_AMD_DEVICE
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
#if MPK_AMD_DEVICE && defined(MIRAGE_ARCH_GFX1250)
  __hip_atomic_store(reinterpret_cast<unsigned short *>(addr),
                     val,
                     __ATOMIC_RELEASE,
                     __HIP_MEMORY_SCOPE_AGENT);
#elif MPK_AMD_DEVICE
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
#if MPK_AMD_DEVICE && defined(MIRAGE_ARCH_GFX1250)
  __hip_atomic_store(reinterpret_cast<uint8_t *>(addr),
                     val,
                     __ATOMIC_RELEASE,
                     __HIP_MEMORY_SCOPE_AGENT);
#elif MPK_AMD_DEVICE
  asm volatile("global_store_byte %0, %1, off sc0 sc1"
               :
               : "v"(addr), "v"((unsigned)val)
               : "memory");
#else
  *reinterpret_cast<uint8_t volatile *>(addr) = val;
#endif
}

__device__ __forceinline__ int atom_add_release_gpu_s32(int *addr, int val) {
#if MPK_AMD_DEVICE && defined(MIRAGE_ARCH_GFX1250)
  // -> global_atomic_add_u32 ... th:TH_ATOMIC_RETURN scope:SCOPE_DEV, with the
  // release writeback and the correct split waits emitted by the compiler.
  return __hip_atomic_fetch_add(
      addr, val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#elif MPK_AMD_DEVICE
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
#if MPK_AMD_DEVICE && defined(MIRAGE_ARCH_GFX1250)
  return __hip_atomic_fetch_add(
      addr, val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#elif MPK_AMD_DEVICE
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
#if MPK_AMD_DEVICE && defined(MIRAGE_ARCH_GFX1250)
  // __hip_atomic_compare_exchange_strong takes `expected` by pointer and
  // OVERWRITES it with the observed value on failure -- so pass a local copy,
  // not the caller's `cmp`, and return that copy. On success it is left equal
  // to cmp, which is also what the gfx950 asm returns (the old value). Callers
  // compare the result against cmp to detect success, so both paths agree.
  unsigned long long int observed = cmp;
  __hip_atomic_compare_exchange_strong(addr,
                                       &observed,
                                       val,
                                       __ATOMIC_RELEASE,
                                       __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
  return observed;
#elif MPK_AMD_DEVICE
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
