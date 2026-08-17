// Does the Phase 9 peer-signal rendezvous survive 8 PEs?
//
// WHY THIS EXISTS. The GLM megakernel runs full expert parallelism correctly
// at NP=2 (3.778 ms/iter, text verified) and NP=4 (3.816 ms/iter), and hangs
// at NP=8 inside _full_layer_ep_wait_peers. Every instrument that could read
// the hang out perturbs it away -- MPK_WORKER_STATE completes, a 1 Hz host
// hipMemcpyAsync D2H completes -- and a hung megakernel never flushes its
// device printf buffer, so the one reporting channel is the one the bug
// suppresses. Worse, the edit-build-run loop is ~15 minutes.
//
// This is the same rendezvous with none of the model: same symmetric
// allocations in the same order, same 240x256 launch geometry, same 8
// folding work-groups per GPU, same run-monotonic ep_fold_done counter with
// the same `prev % 8 == 7` leader election, same st_wt_u64 to self plus all
// NPEER peers through a heap-wide delta, and the same two-stage wait (own
// line, then every peer's line). Two minutes to build, seconds to run, and it
// can be run at -np 2 / 4 / 8 to bracket the width where it breaks.
//
// It deliberately does NOT reproduce: the MoE work, the register pressure of
// the real fused layer, or the other 200-odd KB of per-layer traffic. If this
// passes at 8 PEs, that absence is the finding -- the rendezvous is sound and
// the hang needs one of those to reproduce.
//
// Build (see tests/standalone/build.sh; the flags are load-bearing --
// -fgpu-rdc for cross-TU device calls, xnack- because the IPC backend needs
// an xnack-off code object on gfx950):
//
//   hipcc -std=c++17 -fgpu-rdc --hip-link --offload-arch=gfx950:xnack- \
//     -I$ROCSHMEM_INC_PATH -I$MPI_INC_PATH test_ep_signal_np8.hip \
//     -x none $ROCSHMEM_LIB_PATH/librocshmem.a -L$MPI_LIB_PATH -lmpi \
//     -lhsa-runtime64 -lamdhip64 -o test_ep_signal_np8
//
// Run: mpirun -np 8 ./test_ep_signal_np8
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

#define MAX_PES 16
// Megakernel geometry: 240 workers of 256 threads over 8 XCDs, so 30 workers
// per XCD and work-group 0 of each XCD folds.
#define NUM_WORKERS 240
#define WG_THREADS 256
#define WORKERS_PER_XCD 30
#define NUM_XCD 8
#define NUM_LAYERS 46
// FULL_LAYER_EP_SIGNAL_STRIDE, in uint64s. 64 B per PE.
#define SIG_STRIDE 8
// QKV_REDUCTION_SIZE, the fold width in bf16 columns.
#define FOLD_COLS 2048
#define FOLD_CHUNK (((FOLD_COLS + 7) / 8 + 1) & ~1)
// Give up rather than hang, so a failing run still reports.
#define SPIN_LIMIT 40000000

// Verbatim from mpk_atoms.cuh -- copying rather than including keeps this file
// free of the model headers, and these two are the whole transport under test.
__device__ __forceinline__ unsigned long long int
    ld_sys_u64(unsigned long long int *addr) {
  unsigned long long int val;
  asm volatile("global_load_dwordx2 %0, %1, off sc0 sc1 nt\n"
               "s_waitcnt vmcnt(0)"
               : "=v"(val)
               : "v"(addr)
               : "memory");
  return val;
}

__device__ __forceinline__ void st_wt_u64(void *addr,
                                          unsigned long long int val) {
  asm volatile("global_store_dwordx2 %0, %1, off sc0 sc1"
               :
               : "v"(addr), "v"(val)
               : "memory");
}

__device__ __forceinline__ void st_wt_u32(void *addr, unsigned int val) {
  asm volatile("global_store_dword %0, %1, off sc0 sc1"
               :
               : "v"(addr), "v"(val)
               : "memory");
}

__device__ __forceinline__ int atom_add_release_gpu_s32(int *addr, int val) {
  int old;
  asm volatile("flat_atomic_add %0, %1, %2 sc0 sc1\n"
               "s_waitcnt vmcnt(0) lgkmcnt(0)"
               : "=v"(old)
               : "v"(addr), "v"(val)
               : "memory");
  return old;
}

__device__ int g_fail[MAX_PES];
__device__ int g_stuck_layer;
__device__ int g_stuck_kind; // 1 = self wait, 2 = peer wait
__device__ unsigned g_stuck_rem;

// One "layer" of Phase 9, stripped to the rendezvous.
//
// counters[0] is ep_fold_done: run-monotonic, never reset between layers,
// exactly as in the megakernel -- layer L elects on prev == 8L+7.
__global__ __launch_bounds__(WG_THREADS) void ep_rendezvous(
    unsigned short *gather, unsigned long long *sig, int *counters,
    long long const *peer_delta, int my_pe, int n_pes, int slot_elems) {
  int const tid = threadIdx.x;
  int const wid = blockIdx.x;
  // Megakernel mapping: consecutive blocks round-robin the XCDs.
  int const xcd_id = wid % NUM_XCD;
  int const xcd_rank = wid / NUM_XCD;
  int const npeer = n_pes - 1;

  for (int layer = 0; layer < NUM_LAYERS; layer++) {
    unsigned long long const expected = (unsigned long long)(layer + 1);

    // ---- 9b: the fold. Only work-group 0 of each XCD, disjoint columns.
    if (xcd_rank == 0) {
      int const col_lo = xcd_id * FOLD_CHUNK;
      int const col_hi =
          (col_lo + FOLD_CHUNK) < FOLD_COLS ? (col_lo + FOLD_CHUNK) : FOLD_COLS;
      unsigned short *my_slot = gather + (size_t)my_pe * slot_elems;
      // Same shape as _full_layer_ep_fold_partial: write each column into my
      // own slot and into every peer's copy of my slot, packed 32 bits at a
      // time, then one drain for all of them.
      for (int c = col_lo + tid * 2; c < col_hi; c += WG_THREADS * 2) {
        unsigned int v = ((unsigned int)(my_pe + layer) << 16) |
                         (unsigned int)((c & 0xffff));
        st_wt_u32((void *)&my_slot[c], v);
        for (int q = 0; q < npeer; q++) {
          unsigned short *d =
              reinterpret_cast<unsigned short *>(
                  reinterpret_cast<char *>(my_slot) + peer_delta[q]);
          st_wt_u32((void *)&d[c], v);
        }
      }
      __syncthreads();
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

      // ---- 9c: arrival, then the 8th work-group signals.
      if (tid == 0) {
        int prev_f = atom_add_release_gpu_s32(&counters[0], 1);
        if (prev_f % 8 == 7) {
          for (int q = 0; q < npeer; q++) {
            unsigned long long *peer_sig =
                reinterpret_cast<unsigned long long *>(
                    reinterpret_cast<char *>(sig + (size_t)my_pe * SIG_STRIDE) +
                    peer_delta[q]);
            st_wt_u64((void *)peer_sig, expected);
          }
          st_wt_u64((void *)(sig + (size_t)my_pe * SIG_STRIDE), expected);
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        }
      }
    }

    // ---- 9d: EVERY worker waits on its own line, then on all peers'.
    if (tid == 0) {
      unsigned long long *self = sig + (size_t)my_pe * SIG_STRIDE;
      long long spins = 0;
      while (ld_sys_u64(self) < expected) {
        if (++spins > SPIN_LIMIT) {
          g_fail[my_pe] = 1;
          g_stuck_layer = layer;
          g_stuck_kind = 1;
          break;
        }
        __builtin_amdgcn_s_sleep(1);
      }
      unsigned remaining = 0;
      for (int p = 0; p < n_pes; p++) {
        if (p != my_pe) {
          remaining |= (1u << p);
        }
      }
      spins = 0;
      while (remaining) {
        for (int p = 0; p < n_pes; p++) {
          if (remaining & (1u << p)) {
            if (ld_sys_u64(sig + (size_t)p * SIG_STRIDE) >= expected) {
              remaining &= ~(1u << p);
            }
          }
        }
        if (remaining) {
          if (++spins > SPIN_LIMIT) {
            g_fail[my_pe] = 1;
            g_stuck_layer = layer;
            g_stuck_kind = 2;
            g_stuck_rem = remaining;
            break;
          }
          asm volatile("buffer_inv sc1" ::: "memory");
          __builtin_amdgcn_s_sleep(1);
        }
      }
    }
    __syncthreads();
    asm volatile("buffer_inv" ::: "memory");

    // ---- Payload check, on the folding work-groups only: every peer's slice
    // of this layer must be readable. A signal that arrives ahead of its bytes
    // is a different bug from a signal that never arrives, and only a content
    // check separates them.
    //
    // The predicate is `>= layer`, not `== layer`. There is no barrier between
    // this read and the next iteration's fold -- the real megakernel does not
    // have one either -- so a peer that has already started layer L+1 has
    // legitimately overwritten these columns with L+1's pattern. An equality
    // check therefore fails on ordinary skew, which is what the first version
    // of this test reported and what made it cry wolf at layers 42-45. What
    // must never happen is a value from BEFORE this layer: that would mean the
    // signal outran its bytes, which is the bug worth catching.
    if (xcd_rank == 0 && tid == 0) {
      for (int p = 0; p < n_pes; p++) {
        unsigned short const *slot = gather + (size_t)p * slot_elems;
        int c = xcd_id * FOLD_CHUNK;
        int got = (int)slot[c + 1] - p;
        if (got < layer) {
          g_fail[my_pe] = 2;
          g_stuck_layer = layer;
        }
      }
    }
  }
}

__global__ void resolve_deltas(void *probe, int my_pe, int n_pes,
                               long long *out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  int q = 0;
  for (int pe = 0; pe < n_pes; pe++) {
    if (pe == my_pe) {
      continue;
    }
    void *p = rocshmem_ptr(probe, pe);
    out[q] = p ? (reinterpret_cast<char *>(p) - reinterpret_cast<char *>(probe))
                : 0;
    printf("[delta] pe=%d peer=%d ptr=%p delta=%lld\n", my_pe, pe, p,
           (long long)out[q]);
    if (p == nullptr) {
      out[MAX_PES] = 1; // sentinel: at least one peer unmapped
    }
    q++;
  }
}

int main(int argc, char **argv) {
  char const *lr = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  int pre_rank = lr ? atoi(lr) : 0;
  (void)hipSetDevice(pre_rank);
  rocshmem_init();
  int my_pe = rocshmem_my_pe();
  int n_pes = rocshmem_n_pes();
  (void)hipSetDevice(pre_rank);

  int const slot_elems = FOLD_COLS;
  // Allocation ORDER matters: the megakernel's ep_signal is the last of 49
  // symmetric allocations and is addressed by a delta taken from the first.
  // Keep gather first and sig last so the same heap-wide-flatness assumption
  // is the one under test.
  unsigned short *gather =
      (unsigned short *)rocshmem_malloc(sizeof(unsigned short) * slot_elems *
                                        n_pes);
  unsigned long long *sig =
      (unsigned long long *)rocshmem_malloc(sizeof(unsigned long long) *
                                            SIG_STRIDE * n_pes);
  (void)hipMemset(gather, 0, sizeof(unsigned short) * slot_elems * n_pes);
  (void)hipMemset(sig, 0, sizeof(unsigned long long) * SIG_STRIDE * n_pes);

  int *counters = nullptr;
  (void)hipMalloc(&counters, sizeof(int) * 64);
  (void)hipMemset(counters, 0, sizeof(int) * 64);
  long long *deltas = nullptr;
  (void)hipMalloc(&deltas, sizeof(long long) * (MAX_PES + 1));
  (void)hipMemset(deltas, 0, sizeof(long long) * (MAX_PES + 1));

  rocshmem_barrier_all();
  // Deltas are taken from `gather`, and reused for `sig` -- exactly what the
  // megakernel does, and the assumption a non-flat heap would break.
  hipLaunchKernelGGL(resolve_deltas, dim3(1), dim3(1), 0, 0, (void *)gather,
                     my_pe, n_pes, deltas);
  (void)hipDeviceSynchronize();
  rocshmem_barrier_all();

  hipLaunchKernelGGL(ep_rendezvous, dim3(NUM_WORKERS), dim3(WG_THREADS), 0, 0,
                     gather, sig, counters, deltas, my_pe, n_pes, slot_elems);
  hipError_t err = hipDeviceSynchronize();

  int fail[MAX_PES] = {0};
  int stuck_layer = -1, stuck_kind = 0;
  unsigned stuck_rem = 0;
  (void)hipMemcpyFromSymbol(fail, HIP_SYMBOL(g_fail), sizeof(fail));
  (void)hipMemcpyFromSymbol(&stuck_layer, HIP_SYMBOL(g_stuck_layer),
                            sizeof(int));
  (void)hipMemcpyFromSymbol(&stuck_kind, HIP_SYMBOL(g_stuck_kind), sizeof(int));
  (void)hipMemcpyFromSymbol(&stuck_rem, HIP_SYMBOL(g_stuck_rem),
                            sizeof(unsigned));
  printf("[host] pe=%d n_pes=%d hip=%s fail=%d stuck_layer=%d kind=%d "
         "rem=0x%x  %s\n",
         my_pe, n_pes, hipGetErrorString(err), fail[my_pe], stuck_layer,
         stuck_kind, stuck_rem,
         (err == hipSuccess && fail[my_pe] == 0) ? "PASS" : "FAIL");

  rocshmem_barrier_all();
  rocshmem_finalize();
  return 0;
}
