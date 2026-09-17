// nps2 runtime: memory classes and rendezvous primitives for SPX+NPS2.
//
// Every allocation must declare its class. There is no default, because the
// default is what put fleet's MoE weights on spanning NC at 386 ns/access
// against this box's 154 ns AID-local.
//
//   Private8<T>  class A  per-consumer private. 8 slices, slice i homed in
//                         aid_of_xcd(i). The bulk of all memory. Always local.
//   Bcast2<T>    class B  read-only broadcast. One copy per AID, written once
//                         at load. Costs 2x a few KB and removes every
//                         cross-AID read.
//   Gate         class C  the only shared state. Dual-published with
//                         write-through stores into both AID replicas.
//
// Primitive choice is fixed by Stage 0 (nps1/policy, SPX/NPS2, ns/op at 256
// concurrent workers):
//
//   broadcast  DUAL  909   vs FLAT 25860   -- 28x, and FLAT is banned
//   barrier    HIER  2904  vs FLAT 15515   -- below the 3767 null baseline
//   reduce     PEER  2785  vs FLAT 16527
//
// Two hazards that are not optional, both measured:
//   - an MTYPE_RW line is coherent only inside its own AID, so a replica must
//     be published with a write-through STORE. A cross-AID atomic does not
//     cross and the reader times out after 3 s (this is why evctr v2 hung).
//   - `buffer_inv` with no scope is an architectural NOP on gfx950. Every
//     acquire carries sc1.
#pragma once

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

namespace nps2 {

static constexpr int kNumXcds = 8;
static constexpr int kNumAids = 2;
// 64 B apart, so two gates never share a 128 B line. False sharing between a
// read-mostly flag and a written neighbour is the one thing that makes an
// otherwise-cached NC line go to DRAM on every read.
static constexpr int kGateStrideInts = 16;

__host__ __device__ inline int aid_of_xcd(int xcd) {
  return xcd < (kNumXcds / 2) ? 0 : 1;
}

// ---------------------------------------------------------------- device ---
// These are declared unconditionally. A __global__ body is parsed in BOTH the
// host and device passes, so anything hidden behind __HIP_DEVICE_COMPILE__
// disappears from the host pass and every use inside a kernel fails to
// compile. Only the inline asm is guarded, with a portable fallback -- the
// same shape mpk_atoms.cuh uses.

// Read the XCD (XCC) this workgroup is actually executing on.
//
// The previous version returned blockIdx.x % 8, which is documentation, not a
// measurement. Worse, the "probe" that was supposed to validate the
// distribution counted these same labels, so it reported a perfect 23-per-XCD
// split tautologically and validated nothing. Anything that derives a rank or
// a locality decision from a guessed XCD is unverified -- and the MoE parity
// dispatch built on it produced a DIFFERENT checksum on every run, because
// ranks collided and work items were executed twice or not at all.
//
// HW_REG_XCC_ID is the architectural register that answers this on gfx94x /
// gfx95x. kNumXcds masks it so a partition with fewer XCDs still indexes in
// range.
__device__ __forceinline__ int my_xcd() {
#if defined(__HIP_DEVICE_COMPILE__)
  unsigned int id;
  // hwreg(HW_REG_XCC_ID) -> the physical XCC index
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(id));
  return static_cast<int>(id) & (kNumXcds - 1);
#else
  return static_cast<int>(blockIdx.x) % kNumXcds;
#endif
}

// The label the old code assumed, kept only so a probe can compare the two
// and show whether the assumption ever held.
__device__ __forceinline__ int my_xcd_assumed() {
  return static_cast<int>(blockIdx.x) % kNumXcds;
}
__device__ __forceinline__ int my_aid() {
  return aid_of_xcd(my_xcd());
}

__device__ __forceinline__ unsigned int ld_scoped_u32(void const *addr) {
#if defined(__HIP_DEVICE_COMPILE__)
  unsigned int v;
  asm volatile("global_load_dword %0, %1, off sc0 sc1\n"
               "s_waitcnt vmcnt(0)"
               : "=v"(v)
               : "v"(addr)
               : "memory");
  return v;
#else
  return *reinterpret_cast<unsigned int const volatile *>(addr);
#endif
}

__device__ __forceinline__ void st_wt_u32(void *addr, unsigned int val) {
#if defined(__HIP_DEVICE_COMPILE__)
  asm volatile("global_store_dword %0, %1, off sc0 sc1"
               :
               : "v"(addr), "v"(val)
               : "memory");
#else
  *reinterpret_cast<unsigned int volatile *>(addr) = val;
#endif
}

// Scoped acquire. Bare `buffer_inv` invalidates nothing on gfx950.
__device__ __forceinline__ void acquire_l2() {
#if defined(__HIP_DEVICE_COMPILE__)
  asm volatile("buffer_inv sc1" ::: "memory");
#endif
}

__device__ __forceinline__ unsigned long long rt_clock() {
#if defined(__HIP_DEVICE_COMPILE__)
  return __builtin_amdgcn_s_memrealtime();
#else
  return 0ull;
#endif
}

// ------------------------------------------------------------ class C: Gate -
// A gate is one 32-bit epoch, mirrored in both AIDs. The producer publishes
// into both; every consumer polls only the mirror homed in its own range,
// which is why the cost is flat in poller count.
struct GateSet {
  unsigned int *rep[kNumAids]; // device pointers, one per AID
  int n_gates;

  __device__ __forceinline__ void publish(int gate, unsigned int epoch) const {
    // Both stores are write-through, so each crosses into the other AID.
    st_wt_u32(&rep[0][gate * kGateStrideInts], epoch);
    st_wt_u32(&rep[1][gate * kGateStrideInts], epoch);
  }

  // Returns false if the deadline expired. Every spin in this runtime has a
  // deadline; an un-bounded spin on a mode where a missed release is possible
  // turns a bug into a nine-minute hang.
  __device__ __forceinline__ bool wait(int gate,
                                       unsigned int epoch,
                                       unsigned long long deadline_ticks) const {
    unsigned int const *p = &rep[my_aid()][gate * kGateStrideInts];
    unsigned long long const t0 = rt_clock();
    while (ld_scoped_u32(p) < epoch) {
      if (rt_clock() - t0 > deadline_ticks) {
        return false;
      }
    }
    acquire_l2();
    return true;
  }
};

// -------------------------------------------------------- class C: Barrier --
// HIER: every workgroup arrives into a counter homed in its own AID, one
// elected representative per XCD arrives at the top, and the release fans
// back out through the gate replicas. No counter is ever incremented from
// both AIDs, so no atomic ever crosses the boundary.
struct BarrierSet {
  unsigned int *xcd_count[kNumAids]; // [xcd] arrivals, homed per AID
  unsigned int *global_count;        // 8 arrivals per barrier, one per XCD
  GateSet release;                   // single-gate fan-out
  // Expected arrivals per XCD. NOT assumed uniform: the SPX workgroup-to-XCD
  // mapping must be measured, not predicted, and a grid that does not divide
  // evenly by 8 leaves some XCDs short. An over-estimate here deadlocks the
  // barrier, so callers probe the real distribution and fill this in.
  int n_expect[kNumXcds];

  // Three levels, and the important property is that a WAITER POLLS EXACTLY
  // ONE LINE, homed in its own AID.
  //
  //   1. every workgroup bumps its XCD's counter   -- AID-local atomic
  //   2. the last one per XCD bumps a global count -- 8 shared atomics total
  //   3. the last XCD publishes ONE release gate into both AID mirrors
  //
  // The first version had each waiter poll all eight per-XCD gates. That is
  // 8x the poll traffic (2048 concurrent system-scope loads at 256 blocks)
  // and it measured like FLAT -- 15836 ns vs HIER's 2904. Fanning the wait
  // out across gates destroys the very property the hierarchy exists to buy.
  __device__ __forceinline__ bool arrive_and_wait(
      int gate, unsigned int epoch, unsigned long long deadline_ticks) const {
    int const xcd = my_xcd();
    int const aid = aid_of_xcd(xcd);
    __syncthreads();
    if (threadIdx.x == 0) {
      // Level 1: local to this AID, so a plain device atomic is cheap.
      //
      // Counters are never reset -- resetting races the next epoch's
      // arrivals -- so they accumulate and every last-arrival test scales
      // with the epoch. Comparing against a bare count is true only on
      // epoch 1 and deadlocks every barrier after it.
      unsigned int const prev =
          atomicAdd(&xcd_count[aid][xcd * kGateStrideInts], 1u);
      if (prev + 1u == static_cast<unsigned int>(n_expect[xcd]) * epoch) {
        // Level 2: one arrival per XCD, so only 8 atomics ever touch this
        // shared line per barrier. That is far below the poller-contention
        // cliff, which starts to bite in the tens.
        unsigned int const g = atomicAdd(global_count, 1u);
        if (g + 1u == static_cast<unsigned int>(kNumXcds) * epoch) {
          // Level 3: last XCD releases everyone, one dual-published gate.
          release.publish(gate, epoch);
        }
      }
    }
    // Every waiter polls one line, in its own AID.
    return release.wait(gate, epoch, deadline_ticks);
  }
};

// ------------------------------------------------- workgroup -> XCD probe --
// Counts how many workgroups of a given grid actually land on each XCD.
// Never predict this: the rules on this box record a probe that assumed
// blockIdx.x == xcd, never satisfied its predicate, and spun for nine
// minutes. A barrier configured from a wrong prediction deadlocks.
// counts[0..7]   = real distribution, from HW_REG_XCC_ID
// counts[8..15]  = distribution under the blockIdx.x % 8 assumption
// counts[16]     = number of workgroups where the two DISAGREE
__global__ void k_probe_xcd(unsigned int *counts) {
  if (threadIdx.x == 0) {
    int const real = my_xcd();
    int const assumed = my_xcd_assumed();
    atomicAdd(&counts[real], 1u);
    atomicAdd(&counts[kNumXcds + assumed], 1u);
    if (real != assumed) {
      atomicAdd(&counts[2 * kNumXcds], 1u);
    }
  }
}

// ------------------------------------------------------------------- host --
// Host-only helpers. Templates, so they cost nothing unless instantiated, and
// they are never instantiated from device code.

// Class A: 8 per-consumer slices, slice i in the AID that will read it.
// This is the shape that measured 15.89 us/layer against split50's 24.84 --
// placement must follow the consumer, not merely land "in an AID".
template <typename T> struct Private8 {
  T *slice[kNumXcds] = {};
  size_t elems_per_slice = 0;

  // alloc_aid must be aid_local.h's alloc_in_aid (bytes, aid, extra_flags).
  template <typename AllocFn>
  bool init(size_t elems_each, AllocFn alloc_aid) {
    elems_per_slice = elems_each;
    for (int x = 0; x < kNumXcds; x++) {
      slice[x] = static_cast<T *>(
          alloc_aid(elems_each * sizeof(T), aid_of_xcd(x), 0ull));
      if (slice[x] == nullptr) {
        fprintf(stderr, "[nps2] Private8: slice %d failed in AID %d\n", x,
                aid_of_xcd(x));
        return false;
      }
    }
    return true;
  }
};

// Class B: read-only broadcast, one copy per AID.
template <typename T> struct Bcast2 {
  T *copy[kNumAids] = {};
  size_t elems = 0;

  template <typename AllocFn>
  bool init(size_t n, AllocFn alloc_aid) {
    elems = n;
    for (int a = 0; a < kNumAids; a++) {
      copy[a] = static_cast<T *>(alloc_aid(n * sizeof(T), a, 0ull));
      if (copy[a] == nullptr) {
        return false;
      }
    }
    return true;
  }

  // Both copies get the same bytes, once, at load.
  bool fill(T const *src) {
    for (int a = 0; a < kNumAids; a++) {
      if (hipMemcpy(copy[a], src, elems * sizeof(T),
                    hipMemcpyHostToDevice) != hipSuccess) {
        return false;
      }
    }
    return true;
  }
};

} // namespace nps2

