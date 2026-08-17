/* mpk_comm.cuh - Backend-agnostic SHMEM communication shim for the Mirage
 * Persistent Kernel (MPK).
 *
 * MPK runs the whole model as one persistent megakernel per GPU and performs
 * inter-GPU communication with device-initiated, one-sided put+signal calls
 * issued from inside the kernel. This header selects the communication backend
 * at compile time so the megakernel and the transpiler-emitted allreduce tasks
 * stay backend-neutral:
 *
 *   USE_NVSHMEM  -> NVIDIA NVSHMEM  (nvshmemx_putmem_signal_block)
 *   USE_ROCSHMEM -> AMD rocSHMEM    (rocshmem_putmem_signal_wg, IPC backend)
 *   (neither)    -> single-PE no-op fallback (single-GPU builds)
 *
 * The mpk_* wrappers below are the only symbols the rest of the codebase uses.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

// ld_nt_u64 / st_wt_u64. This header inlines the rocSHMEM device primitives it
// needs (see mpk_shmem_signal_wait_ge and mpk_shmem_peer_ptr below) and those
// are built out of these, so the dependency is real rather than incidental --
// do not rely on the includer having pulled it in first.
#include "mpk_atoms.cuh"

#if defined(USE_NVSHMEM)
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#elif defined(USE_ROCSHMEM)
#include <mpi.h>
#include <rocshmem/rocshmem.hpp>
#else
// Single-GPU fallback needs the device-runtime allocator declarations.
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
#include <hip/hip_runtime.h>
#endif
#endif

// ---------------------------------------------------------------------------
// Signal operation constant (referenced by emitted put+signal task code).
// ---------------------------------------------------------------------------
#if defined(USE_NVSHMEM)
#define MPK_SIGNAL_ADD NVSHMEM_SIGNAL_ADD
#elif defined(USE_ROCSHMEM)
#define MPK_SIGNAL_ADD rocshmem::ROCSHMEM_SIGNAL_ADD
#else
#define MPK_SIGNAL_ADD 0
#endif

// ---------------------------------------------------------------------------
// Device-side, block / work-group collective put with remote signal update.
// All threads in the block (work-group) must call this collectively.
//   dst/src  : symmetric-heap pointers
//   nbytes   : number of bytes to transfer
//   sig_addr : remote signal (event counter) address on the target PE
//   signal   : value combined into the signal using sig_op
//   sig_op   : MPK_SIGNAL_ADD (atomic add), etc.
//   pe       : target PE (== target gpu id)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void
mpk_putmem_signal_block(void *dst, void const *src, size_t nbytes,
                        uint64_t *sig_addr, uint64_t signal, int sig_op,
                        int pe) {
#if defined(USE_NVSHMEM)
  nvshmemx_putmem_signal_block(dst, src, nbytes, sig_addr, signal, sig_op, pe);
#elif defined(USE_ROCSHMEM)
  rocshmem::rocshmem_putmem_signal_wg(
      dst, src, nbytes, sig_addr, signal, sig_op, pe);
#else
  (void)dst;
  (void)src;
  (void)nbytes;
  (void)sig_addr;
  (void)signal;
  (void)sig_op;
  (void)pe;
#endif
}

// ---------------------------------------------------------------------------
// Device-side wait on a (symmetric) signal/event counter that a remote PE
// increments via put+signal. Blocks the calling thread until *sig_addr >= val.
// This is the consumer side of the cross-GPU put+signal handshake and must use
// the backend's signal-wait primitive (not a plain load) so the remote write is
// made visible past the local cache hierarchy.
//   sig_addr : symmetric event-counter address on THIS PE
//   val      : threshold to wait for (num_triggers * iteration_num)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void
mpk_shmem_signal_wait_ge(uint64_t *sig_addr, uint64_t val) {
#if defined(USE_NVSHMEM)
  nvshmem_signal_wait_until(sig_addr, NVSHMEM_CMP_GE, val);
#elif defined(USE_ROCSHMEM)
#ifdef MPK_COMM_DEBUG
  if (threadIdx.x == 0)
    printf("[COMM] wait ENTER sig=%p val=%llu cur=%llu\n", (void *)sig_addr,
           (unsigned long long)val,
           (unsigned long long)__atomic_load_n(
               reinterpret_cast<unsigned long long *>(sig_addr),
               __ATOMIC_RELAXED));
#endif
  // Inlined rather than calling rocshmem_ulonglong_wait_until. That entry
  // point is declared ATTR_NO_INLINE in rocshmem_COLL.hpp and reaches the
  // comparison through Context's static-cast DISPATCH, so it stays a real
  // out-of-line call across the device link -- it is present in this kernel's
  // disassembly as _ZN8rocshmem29rocshmem_ulonglong_wait_untilEPyiy, and its
  // spin loop spills the loaded value to scratch and reloads it on every poll
  // iteration:
  //
  //   global_load_dwordx2 v[6:7], v[0:1], off sc0 sc1
  //   flat_store_dwordx2  v[4:5], v[6:7] sc0 sc1   <- to private
  //   flat_load_dwordx2   v[6:7], v[4:5] sc0 sc1   <- straight back
  //   v_cmp_ge_u64_e32 vcc, v[6:7], v[2:3]
  //
  // A waiting work-group holds its VGPRs and its slot on the CU either way,
  // so the round-trip buys nothing; it is an artifact of the value crossing a
  // function boundary the library could not inline through. What the library
  // actually does that matters is Context::test -> uncached_load, which on
  // gfx942/gfx950 is `global_load_dwordx2 ... sc0 sc1` + `s_waitcnt vmcnt(0)`
  // -- a load that misses the local cache hierarchy so a peer's store is
  // observed. Use ld_sys_u64, which is exactly that encoding.
  //
  // This used to call ld_nt_u64 (`nt` and nothing else) on the claim that
  // "both bypass L2 on this part". They do not. `nt` is a temporal hint that
  // marks the line evict-first; it does not change the scope of the access, so
  // the load still hits vL1 and this XCD's L2, and a peer's store over XGMI is
  // not guaranteed to be observed at all. Same defect ld_nt_s32 had; see the
  // scope table at the top of mpk_atoms.cuh, and note in particular that `sc0`
  // alone would NOT have been enough here either -- it is workgroup scope.
  //
  // The s_sleep is an addition, not a translation: rocSHMEM's loop is a bare
  // `while (!test(...))`, which issues back-to-back uncached loads at full
  // rate. Backing off keeps a spinning waiter from consuming the memory
  // pipeline that the producer it is waiting on needs.
  {
    unsigned long long *p = reinterpret_cast<unsigned long long *>(sig_addr);
    unsigned long long const want = static_cast<unsigned long long>(val);
    while (ld_sys_u64(p) < want) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
#ifdef MPK_COMM_DEBUG
  if (threadIdx.x == 0)
    printf("[COMM] wait EXIT  sig=%p val=%llu\n", (void *)sig_addr,
           (unsigned long long)val);
#endif
#else
  (void)sig_addr;
  (void)val;
#endif
}

// ---------------------------------------------------------------------------
// Direct peer address translation.
//
// Returns a pointer THIS PE can dereference that aliases the symmetric-heap
// object `dest` on PE `pe`, or nullptr if that PE is not directly addressable.
// On a single node with XGMI and the IPC backend every peer is mapped, so this
// succeeds and ordinary stores to the returned address land in the peer's HBM.
//
// Why this and not putmem_signal: putmem_signal is a work-group collective that
// builds a descriptor and hands the transfer to a DMA engine. That is the right
// trade for a large message, but the EP combine moves 5.8 KB -- 0.09 us of wire
// time at 64 GB/s -- so the fixed cost of setting up the transfer dominates the
// transfer itself. With a peer pointer the producing work-group simply stores
// its result at the remote address as part of the epilogue it was already
// running, and the write travels over the same XGMI link with none of the
// setup. The cost of the exchange collapses to the cost of the stores.
//
// The returned pointer is stable for the lifetime of the allocation, so callers
// resolve it once and reuse it rather than calling this per layer.
//
// Implemented as one add against a per-peer constant instead of a call into
// rocshmem_ptr. The reason that is legitimate, from rocSHMEM's own sources:
// IPCContext::shmem_ptr (and putmem, and getmem -- they all share the
// arithmetic) computes
//
//     ipc_bases[pe] + (dest - ipc_bases[my_pe])
//
// which regroups to `dest + (ipc_bases[pe] - ipc_bases[my_pe])`. That
// parenthesised term does not depend on `dest`. It cannot: ipcHostInit takes a
// SINGLE hipIpcGetMemHandle over the whole symmetric heap base and the peer
// opens that one handle with hipIpcOpenMemHandle, so there is exactly one base
// per PE and one delta covering every symmetric object for the process
// lifetime. Recomputing it per call re-walks a four-deep dependent load chain
// (ROCSHMEM_CTX_DEFAULT -> ctx_opaque -> ipcImpl_.ipc_bases -> two indexed
// loads) to arrive at the same number every time.
//
// So the delta is sampled once at init (mpk_shmem_init_peer_deltas below, which
// is where the one rocSHMEM call now lives) and the hot path is an add.
// ---------------------------------------------------------------------------
#define MPK_MAX_PES 16

#if defined(USE_ROCSHMEM)
// Byte offset from a local symmetric address to the same object on peer `pe`.
// Only meaningful where the matching bit of mpk_peer_heap_valid_d is set --
// a zero delta is legitimate (it is what pe == my_pe gives), so the validity
// has to be carried separately rather than inferred from the value.
__device__ int64_t mpk_peer_heap_delta_d[MPK_MAX_PES];
__device__ uint32_t mpk_peer_heap_valid_d;
#endif

// Fetch the peer delta, or report that this peer has no direct mapping.
// Callers that translate more than one address to the same peer should call
// this once and add the delta themselves.
__device__ __forceinline__ bool mpk_shmem_peer_delta(int pe, int64_t *out) {
#if defined(USE_NVSHMEM)
  (void)pe;
  (void)out;
  return false; // NVSHMEM path keeps using nvshmem_ptr per address.
#elif defined(USE_ROCSHMEM)
  if (pe < 0 || pe >= MPK_MAX_PES ||
      ((mpk_peer_heap_valid_d >> pe) & 1u) == 0u) {
    return false;
  }
  *out = mpk_peer_heap_delta_d[pe];
  return true;
#else
  (void)pe;
  *out = 0;
  return true;
#endif
}

__device__ __forceinline__ void *mpk_shmem_peer_ptr(void const *dest, int pe) {
#if defined(USE_NVSHMEM)
  return nvshmem_ptr(dest, pe);
#else
  int64_t delta = 0;
  if (!mpk_shmem_peer_delta(pe, &delta)) {
    return nullptr;
  }
  return const_cast<char *>(reinterpret_cast<char const *>(dest)) + delta;
#endif
}

#if defined(USE_ROCSHMEM)
// Sample the deltas. One thread, once, at init -- the only place the megakernel
// still calls a rocSHMEM device function on the EP path.
//
// `probe` must be an address in the symmetric heap; any one will do, since the
// delta is heap-wide. A peer whose translation comes back null (or whose result
// is not a constant offset, which would mean the single-base assumption above
// is wrong on this backend) is left invalid, so mpk_shmem_peer_ptr returns
// nullptr for it and callers take their staged fallback exactly as before.
__global__ void mpk_init_peer_deltas_kernel(void *probe, int my_pe, int n_pes,
                                            void **allocs, int n_allocs) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  uint32_t valid = 0;
  for (int pe = 0; pe < n_pes && pe < MPK_MAX_PES; pe++) {
    void *p = rocshmem::rocshmem_ptr(probe, pe);
    if (p == nullptr) {
      continue;
    }
    int64_t delta = reinterpret_cast<char *>(p) - reinterpret_cast<char *>(probe);
    // Cross-check on a second address in the same heap. If the backend ever
    // stops being a single flat mapping this catches it here, at init, rather
    // than as silent corruption 36 layers deep.
    //
    // `probe + 64` alone did NOT do that. `probe` is the FIRST recorded
    // symmetric allocation, so probe+64 is still inside that same object: it
    // proves the mapping is constant across 64 bytes, which no allocator was
    // ever going to fail. Every real hazard here is inter-object -- the heap
    // spilling into a second segment, a differently-aligned tail allocation --
    // and those live at the far end, in the objects this never looked at. The
    // EP signal array is the LAST of 49 allocations and the only one addressed
    // by delta from the first, which is precisely the pair the old check
    // skipped. So walk every recorded allocation.
    bool flat = true;
    for (int a = 0; a < n_allocs; a++) {
      if (allocs[a] == nullptr) {
        continue;
      }
      void *pa = rocshmem::rocshmem_ptr(allocs[a], pe);
      if (pa == nullptr ||
          (reinterpret_cast<char *>(pa) - reinterpret_cast<char *>(allocs[a])) !=
              delta) {
        printf("[MPK] peer delta for pe=%d differs at alloc %d (%p -> %p, "
               "delta %lld, expected %lld); direct peer stores disabled for "
               "that peer\n",
               pe, a, allocs[a], pa,
               (long long)(pa ? (reinterpret_cast<char *>(pa) -
                                 reinterpret_cast<char *>(allocs[a]))
                              : 0),
               (long long)delta);
        flat = false;
        break;
      }
    }
    if (!flat) {
      continue;
    }
    mpk_peer_heap_delta_d[pe] = delta;
    valid |= (1u << pe);
#ifdef MPK_EP_SIG_DBG
    // Printed from init, which runs before demo.py redirects fd 1 into the
    // deferred device log, so these are visible while a run is still going.
    // Two peers sharing a delta means the peer stores alias: every rank would
    // then be publishing into one peer's line and the others would never see
    // the signal advance.
    printf("[EPDELTA] my_pe=%d pe=%d probe=%p peer=%p delta=%lld\n",
           my_pe, pe, probe, p, (long long)delta);
#endif
  }
  (void)my_pe;
  mpk_peer_heap_valid_d = valid;
}
#endif

// Defined below, next to mpk_shmem_malloc which populates it. Declared here
// because the delta init needs the full allocation list to cross-check.
inline std::vector<std::pair<void *, size_t>> &mpk_shmem_alloc_registry();

// Host entry: call once after the symmetric heap has been allocated and before
// the megakernel launches. `probe` is any symmetric-heap pointer.
__host__ inline void mpk_shmem_init_peer_deltas(void *probe) {
#if defined(USE_ROCSHMEM)
  if (probe == nullptr) {
    fprintf(stderr,
            "[MPK] no symmetric allocation to probe; direct peer stores "
            "disabled\n");
    return;
  }
  int my_pe = rocshmem::rocshmem_my_pe();
  int n_pes = rocshmem::rocshmem_n_pes();
  // Stage the allocation list where the kernel can read it. rocshmem_ptr is
  // device-only, so the walk has to happen on the GPU.
  auto &reg = mpk_shmem_alloc_registry();
  int n_allocs = (int)reg.size();
  void **allocs_d = nullptr;
  if (n_allocs > 0) {
    std::vector<void *> host_allocs;
    host_allocs.reserve(n_allocs);
    for (auto const &a : reg) {
      host_allocs.push_back(a.first);
    }
    if (hipMalloc(&allocs_d, sizeof(void *) * n_allocs) != hipSuccess) {
      allocs_d = nullptr;
      n_allocs = 0;
    } else {
      (void)hipMemcpy(allocs_d, host_allocs.data(), sizeof(void *) * n_allocs,
                      hipMemcpyHostToDevice);
    }
  }
  hipLaunchKernelGGL(mpk_init_peer_deltas_kernel, dim3(1), dim3(1), 0, 0, probe,
                     my_pe, n_pes, allocs_d, n_allocs);
  (void)hipDeviceSynchronize();
  if (allocs_d) {
    (void)hipFree(allocs_d);
  }
#else
  (void)probe;
#endif
}

// Optional per-work-group SHMEM context lifetime hooks. NVSHMEM needs none;
// rocSHMEM's IPC single-node backend does not require them either (kept as
// no-op hooks so other rocSHMEM backends can opt in later without touching the
// megakernel).
__device__ __forceinline__ void mpk_shmem_wg_init() {}
__device__ __forceinline__ void mpk_shmem_wg_finalize() {}

// ---------------------------------------------------------------------------
// Host-side wrappers.
// ---------------------------------------------------------------------------
// Diagnostic registry of symmetric-heap allocations, recorded in allocation
// order as (ptr, size). Populated by mpk_shmem_malloc below and read back by
// mpk_read_shmem_alloc() so a --verify pass can snapshot an nvshmem/rocshmem
// tensor (e.g. the post-allreduce mlp_final) that has no torch backing. Host
// side only; mpk_shmem_malloc is called serially during init.
inline std::vector<std::pair<void *, size_t>> &mpk_shmem_alloc_registry() {
  static std::vector<std::pair<void *, size_t>> reg;
  return reg;
}

__host__ inline void *mpk_shmem_malloc(size_t size) {
  void *ptr = nullptr;
#if defined(USE_NVSHMEM)
  ptr = nvshmem_malloc(size);
#elif defined(USE_ROCSHMEM)
  // rocSHMEM's IPC allocator may switch the current HIP device as a side effect
  // (it touches every PE's symmetric heap). Save and restore it so that
  // subsequent cudaMalloc / device-to-device weight staging in the caller stays
  // on this rank's device.
  int prev_dev = 0;
  (void)hipGetDevice(&prev_dev);
  ptr = rocshmem::rocshmem_malloc(size);
  (void)hipSetDevice(prev_dev);
#else
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
  (void)hipMalloc(&ptr, size);
#else
  (void)cudaMalloc(&ptr, size);
#endif
#endif
  mpk_shmem_alloc_registry().push_back(std::make_pair(ptr, size));
  return ptr;
}

__host__ inline void mpk_shmem_free(void *ptr) {
#if defined(USE_NVSHMEM)
  nvshmem_free(ptr);
#elif defined(USE_ROCSHMEM)
  rocshmem::rocshmem_free(ptr);
#else
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
  (void)hipFree(ptr);
#else
  (void)cudaFree(ptr);
#endif
#endif
}

__host__ inline int mpk_shmem_my_pe() {
#if defined(USE_NVSHMEM)
  return nvshmem_my_pe();
#elif defined(USE_ROCSHMEM)
  return rocshmem::rocshmem_my_pe();
#else
  return 0;
#endif
}

__host__ inline int mpk_shmem_n_pes() {
#if defined(USE_NVSHMEM)
  return nvshmem_n_pes();
#elif defined(USE_ROCSHMEM)
  return rocshmem::rocshmem_n_pes();
#else
  return 1;
#endif
}

__host__ inline void mpk_shmem_barrier_all() {
#if defined(USE_NVSHMEM)
  nvshmem_barrier_all();
#elif defined(USE_ROCSHMEM)
  rocshmem::rocshmem_barrier_all();
#endif
}

__host__ inline void mpk_shmem_finalize() {
#if defined(USE_NVSHMEM)
  nvshmem_finalize();
#elif defined(USE_ROCSHMEM)
  rocshmem::rocshmem_finalize();
#endif
}
