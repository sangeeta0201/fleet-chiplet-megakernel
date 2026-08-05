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
#include <utility>
#include <vector>

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
  rocshmem::rocshmem_ulonglong_wait_until(
      reinterpret_cast<unsigned long long *>(sig_addr),
      rocshmem::ROCSHMEM_CMP_GE,
      static_cast<unsigned long long>(val));
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
