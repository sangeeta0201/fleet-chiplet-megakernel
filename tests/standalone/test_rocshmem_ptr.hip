// Does rocshmem_ptr() ever hand back a usable peer mapping on this stack?
//
// The megakernel's EP combine has two publication paths: fold straight into
// the peer's symmetric slot via rocshmem_ptr(), or fall back to a staged
// putmem_signal. The direct path is the one every comment describes and the
// one the latency budget assumes -- but a [EPPATH] probe in
// gang_full_layer_fused_mi300.cuh showed ep_direct=0 / peer_slot=(nil) on
// both ranks, i.e. it has never executed.
//
// ANSWER (2026-08-09): cause 1. Stock rocSHMEM's IPCContext::shmem_ptr is a
// stub returning nullptr -- only the reverse-offload backend ever implemented
// it. See patches/rocshmem-ipc-shmem-ptr.patch. This file is the regression
// test for that patch: run it after rebuilding rocSHMEM to confirm the direct
// path is available before trusting any EP latency number.
//
// Expected with the patch applied, both ranks:
//   [host] pe=N rocshmem_ptr(peer) = 0x...  NON-NULL
//   [host] pe=N peer wrote 0xbeefNNNN (want 0xbeefNNNN) PASS
// Without it, both ranks report NULL and the PASS lines never print.
//
// This reproduces the megakernel's exact sequence -- rocshmem_init, malloc off
// the symmetric heap, then rocshmem_ptr from device code -- at a size the
// megakernel uses, and additionally WRITES through the returned pointer and
// has the peer read it back. A non-null pointer that does not carry data is
// worse than a null one, so the check is end-to-end, not just != nullptr.
//
// Build (matching persistent_kernel.py's rocSHMEM flags, which are load-
// bearing here: -fgpu-rdc for cross-TU device calls, and xnack- because the
// IPC backend requires an xnack-off code object on gfx950):
//
//   hipcc -std=c++17 -fgpu-rdc --hip-link --offload-arch=gfx950:xnack- \
//     -I$ROCSHMEM_INC_PATH -I$MPI_INC_PATH test_rocshmem_ptr.hip \
//     -x none $ROCSHMEM_LIB_PATH/librocshmem.a -L$MPI_LIB_PATH -lmpi \
//     -lhsa-runtime64 -lamdhip64 -o test_rocshmem_ptr
//
// Run: mpirun -np 2 ./test_rocshmem_ptr
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

// Same shape as the EP gather buffer: (world, batch, padded_hidden) bf16.
// Kept as a byte count so this file does not need the model's headers.
#define EP_SLOT_BYTES (2 * 2880)

// Ask for the translation and report it. One thread: rocshmem_ptr is
// documented as callable per-thread with no penalty, and the megakernel calls
// it from every thread of the closer work-group.
__global__ void probe_ptr(void *heap, int my_pe, void **out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    void *p = rocshmem_ptr(heap, 1 - my_pe);
    *out = p;
    printf("[dev] pe=%d local=%p peer=%p\n", my_pe, heap, p);
  }
}

// Write a PE-identifying pattern into the peer's buffer through the
// translated pointer. If the mapping is real this lands in the peer's HBM.
__global__ void poke_peer(void *heap, int my_pe) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    void *p = rocshmem_ptr(heap, 1 - my_pe);
    if (p == nullptr) {
      return;
    }
    // Slot layout mirrors the megakernel's: each PE owns slot[my_pe].
    int *dst = reinterpret_cast<int *>(p);
    dst[my_pe] = 0xBEEF0000 + my_pe;
    __threadfence_system();
  }
}

// Read back what the peer should have written into MY buffer.
__global__ void check_local(void *heap, int my_pe, int *out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int *src = reinterpret_cast<int *>(heap);
    *out = src[1 - my_pe];
  }
}

int main(int argc, char **argv) {
  // Select the device BEFORE rocshmem_init, exactly as the megakernel does
  // (persistent_kernel.cuh calls cudaSetDevice(my_rank) first). This is not
  // cosmetic: rocSHMEM never calls hipSetDevice itself, and its init publishes
  // the default device context with hipMemcpyToSymbol, which writes only to
  // the CURRENT device. Init both ranks on device 0 and rank 1's copy of
  // ROCSHMEM_CTX_DEFAULT stays null forever.
  char const *lr = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  int pre_rank = lr ? atoi(lr) : 0;
  (void)hipSetDevice(pre_rank);
  rocshmem_init();
  int my_pe = rocshmem_my_pe();
  int n_pes = rocshmem_n_pes();
  if (n_pes != 2) {
    if (my_pe == 0) {
      printf("need exactly 2 PEs, got %d\n", n_pes);
    }
    rocshmem_finalize();
    return 1;
  }
  // Match the megakernel: it re-asserts the device after init because the IPC
  // backend can leave the current device pointing elsewhere.
  (void)hipSetDevice(pre_rank);

  void *heap = rocshmem_malloc(EP_SLOT_BYTES * n_pes);
  printf("[host] pe=%d heap=%p\n", my_pe, heap);
  (void)hipMemset(heap, 0, EP_SLOT_BYTES * n_pes);
  rocshmem_barrier_all();

  void **d_out = nullptr;
  (void)hipMalloc(&d_out, sizeof(void *));
  hipLaunchKernelGGL(probe_ptr, dim3(1), dim3(1), 0, 0, heap, my_pe, d_out);
  (void)hipDeviceSynchronize();
  void *h_out = nullptr;
  (void)hipMemcpy(&h_out, d_out, sizeof(void *), hipMemcpyDeviceToHost);
  printf("[host] pe=%d rocshmem_ptr(peer) = %p  %s\n", my_pe, h_out,
         h_out ? "NON-NULL" : "NULL");

  // A non-null pointer still has to carry data. Prove it end to end.
  if (h_out != nullptr) {
    rocshmem_barrier_all();
    hipLaunchKernelGGL(poke_peer, dim3(1), dim3(1), 0, 0, heap, my_pe);
    (void)hipDeviceSynchronize();
    rocshmem_barrier_all();
    int *d_val = nullptr;
    (void)hipMalloc(&d_val, sizeof(int));
    hipLaunchKernelGGL(check_local, dim3(1), dim3(1), 0, 0, heap, my_pe, d_val);
    (void)hipDeviceSynchronize();
    int got = 0;
    (void)hipMemcpy(&got, d_val, sizeof(int), hipMemcpyDeviceToHost);
    int want = 0xBEEF0000 + (1 - my_pe);
    printf("[host] pe=%d peer wrote 0x%08x (want 0x%08x) %s\n", my_pe, got,
           want, got == want ? "PASS" : "FAIL");
  }

  rocshmem_barrier_all();
  rocshmem_finalize();
  return 0;
}
