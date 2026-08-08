// Standalone floor for the EP combine's cross-GPU exchange.
//
// WHY THIS EXISTS
// ---------------
// The inline EP combine in gang_full_layer_fused_mi300.cuh Phase 9 costs
// 0.62 ms/token over 36 layers -- 17 us per layer -- to move 5.8 KB between two
// MI350 GPUs. That number came entirely from differencing end-to-end runs with
// #define ablations, which localizes cost to a region but cannot say whether
// 17 us is a bad implementation or what the hardware charges. Every fix
// proposed against it so far has been a guess, and two of them regressed.
//
// This is the missing measurement: the same protocol, the same instructions
// (st_wt_u64 / nt loads / write-through release flags), the same 8-producer
// geometry, with nothing else in the process. A deadlock here is a 2-second
// debug cycle instead of a 15-minute one.
//
// RESULTS (MI350, 2 GPUs over XGMI, 2000 iters, best of 3)
//
//   store          1.53 us   payload only, no handshake
//   sig            2.86 us   handshake only, no payload
//   agg            4.47 us   committed shape: 1 line/PE, 8th fold publishes
//   perxcd         7.92 us   1 line/(PE,XCD), polled one at a time
//   perxcd8        4.02 us   same lines, all 8 polled under one waitcnt
//   full9          8.20 us   + 240-worker 9a rendezvous and its release
//   full9-norel   10.76 us   9a with folders polling the global counter
//
// WHAT THESE SAY
//
// 1. The per-XCD signal split does not pay. It was the obvious fix on paper --
//    publish each slice as it lands instead of behind a GPU-wide count -- and
//    it is 1.8x SLOWER (7.92 vs 4.47) as originally written. The cause is in
//    the poll, not the protocol: ld_nt_u64 has `s_waitcnt vmcnt(0)` baked in,
//    so looping over 8 peer lines is 8 serialized XGMI round trips. Batching
//    them under one waitcnt (perxcd8, via __builtin_nontemporal_load) recovers
//    all of it -- and lands at 4.02 vs 4.47, i.e. a ~0.45 us edge, ~0.016 ms
//    over 36 layers. Not the 0.3 ms the split was pursued for.
//
// 2. The exchange is not where Phase 9's time goes. full9 - perxcd8 = 4.1 us
//    is the 9a rendezvous alone, as much as the entire exchange it gates.
//
// 3. But 9a's cost is the 240-deep arrival, not its release broadcast.
//    Deleting the broadcast and having each folder poll ep_moe_done directly
//    (full9-norel) is 2.6 us WORSE, because 8 pollers hammering one counter
//    that 240 workers are atomically incrementing contend far more than one
//    store to 8 private lines. The committed two-hop release is right.
//
// 4. Floor for the whole thing: 8.2 us/layer = 0.295 ms/token, against 0.62 ms
//    measured in situ. So roughly half of Phase 9 is protocol -- irreducible
//    without changing what is exchanged -- and half is arrival skew between
//    the 8 XCDs, which this harness cannot reproduce because it has no MoE
//    compute in front of the barrier. Skew is the remaining lever, and it is
//    a MoE load-balance problem, not a collective problem.
//
// The conclusion that matters: no rearrangement of this handshake reaches the
// ~0.03 ms/token that EP needs to beat the 2.133 ms 1-GPU baseline, because
// the rendezvous alone is 10x that. EP at bs=1 must exchange LESS OFTEN (fewer
// collectives per token), not faster.
//
// FIDELITY AND ITS LIMITS
// -----------------------
// Faithful: instruction selection, payload size and layout, the number of
// independent producers, the monotonic-counter signal discipline, and the fact
// that both ranks run the same loop in lockstep.
//
// NOT faithful: this uses plain hipDeviceEnablePeerAccess rather than the
// rocSHMEM symmetric heap (the mapping is the same XGMI mapping either way --
// rocshmem_ptr on the IPC backend returns exactly such a peer pointer), and
// there is no MoE compute between iterations, so there is no arrival skew
// between the 8 producers. Real skew makes the collective look WORSE, never
// better -- which is why 8.2 us here reads as a floor under the 17 us measured
// in situ, and why the difference is attributed to skew above.
//
// BUILD/RUN
//   ./build.sh                                   # or, directly:
//   hipcc -o test_ep_collective test_ep_collective.hip --offload-arch=gfx950 \
//         -O3 -std=c++17 -D__HIP_PLATFORM_AMD__
//   HIP_VISIBLE_DEVICES=6,7 ./test_ep_collective [iters]
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define HIP_CHECK(x)                                                           \
  do {                                                                         \
    hipError_t _e = (x);                                                       \
    if (_e != hipSuccess) {                                                    \
      fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x,              \
              hipGetErrorString(_e));                                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Geometry copied from the megakernel, not re-derived: QKV_REDUCTION_SIZE is
// the padded hidden size (2880 -> 2944) and the slot is one row of it in bf16.
static constexpr int kCols = 2944;
static constexpr int kSlotElems = kCols;                  // batch 1
static constexpr int kSlotBytes = kSlotElems * 2;         // 5888 B
static constexpr int kProducers = 8;                      // one per XCD
static constexpr int kChunk = ((kCols + 7) / 8 + 1) & ~1; // even, matches src
// One 64-byte line per (PE, producer). uint64 units: 8 per line, 64 per PE.
static constexpr int kSigStride = 8;
static constexpr int kSigPeStride = 64;

// ---- the exact primitives the megakernel uses -----------------------------
__device__ __forceinline__ void st_wt_u64(void *addr, unsigned long long val) {
  asm volatile("global_store_dwordx2 %0, %1, off sc0 sc1"
               :
               : "v"(addr), "v"(val)
               : "memory");
}
__device__ __forceinline__ unsigned long long ld_nt_u64(unsigned long long *a) {
  unsigned long long v;
  asm volatile("global_load_dwordx2 %0, %1, off nt\n"
               "s_waitcnt vmcnt(0)"
               : "=v"(v)
               : "v"(a)
               : "memory");
  return v;
}
// Poll 8 signal lines with ONE waitcnt instead of eight.
//
// ld_nt_u64 embeds `s_waitcnt vmcnt(0)`, so a loop over 8 lines is 8 serialized
// round trips -- and to peer memory each of those is a full XGMI latency, not an
// L2 hit. That is a property of the poll, not of the protocol: the 8 lines have
// independent writers and can be read concurrently. This issues all 8 loads
// back-to-back and waits once, so the cost is one round trip plus the pipelining
// of 7 more, and returns the minimum (the poll clears when the slowest line has
// reached the threshold).
// __builtin_nontemporal_load emits the same `global_load_dwordx2 ... nt` as the
// hand-written asm but WITHOUT a waitcnt glued to it, so the compiler is free to
// hoist all 8 loads above a single wait. That freedom is the entire point; the
// asm version cannot express it because the waitcnt is inside the asm block.
__device__ __forceinline__ unsigned long long min8_nt(unsigned long long *base) {
  unsigned long long v[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    v[i] = __builtin_nontemporal_load(base + (size_t)i * kSigStride);
  }
  unsigned long long m = v[0];
#pragma unroll
  for (int i = 1; i < 8; i++) {
    m = min(m, v[i]);
  }
  return m;
}

__device__ __forceinline__ int atom_add_release_gpu_s32(int *a, int v) {
  return __hip_atomic_fetch_add(a, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void threadfence_gpu() {
  __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "agent");
}

enum Variant {
  V_PERXCD = 0, // per-(PE,XCD) lines, polled one at a time
  V_AGG = 1,    // one line per PE, published by the 8th local fold
  V_STORE = 2,  // payload only, no handshake
  V_SIG = 3,    // handshake only, no payload
  V_PERXCD8 = 4, // per-(PE,XCD) lines, all 8 polled under one waitcnt
  V_FULL9 = 5,   // 240 workers + 9a rendezvous + perxcd8 exchange
  V_NOREL = 6    // as V_FULL9, but folders poll the global counter directly
};

// One block per producer. Each owns a disjoint column slice of the slot, writes
// that slice to both its own and the peer's gather buffer, then handshakes.
//
// `iters` stands in for layers: the signal threshold is r+1 on a counter that is
// never reset, exactly as layer_counter+1 is in the megakernel, so nothing has
// to be zeroed between iterations and no rendezvous is needed to agree on the
// target.
__global__ __launch_bounds__(256) void ep_exchange_kernel(
    unsigned short *__restrict__ my_gather,   // [2][kSlotElems] local
    unsigned short *__restrict__ peer_gather, // same object on the peer
    unsigned long long *__restrict__ my_sig,
    unsigned long long *__restrict__ peer_sig_base,
    float const *__restrict__ src, // stand-in for moe_workspace_f32
    int *__restrict__ fold_done,   // used by V_AGG only
    int my_pe,
    int iters,
    int variant,
    int *__restrict__ xcc_out,
    int *__restrict__ ctr,     // 9a counters, V_FULL9 only
    int workers_per_xcd) {
  int const tid = threadIdx.x;

  int xcc;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcc));
  if (tid == 0 && blockIdx.x < kProducers) {
    xcc_out[blockIdx.x] = xcc;
  }

  // V_FULL9 runs the real worker count (240) and derives the XCD from hardware,
  // exactly as the megakernel does; every other variant runs one block per XCD
  // and blockIdx IS the XCD (verified by the xcc map this prints).
  bool const full9 = (variant == V_FULL9 || variant == V_NOREL);
  int const blk = full9 ? xcc : (int)blockIdx.x;
  int const xcd_rank = full9 ? ((int)blockIdx.x / kProducers) : 0;
  // 9a counter layout, one 64-byte line apart, mirroring FULL_LAYER_EP_*_SLOT.
  int *ep_xcd_arrive = ctr;            // [8][16]
  int *ep_moe_done = ctr + 8 * 16;     // [1]
  int *ep_release = ctr + 9 * 16;      // [8][16]

  int const col_lo = blk * kChunk;
  int const col_hi = min(col_lo + kChunk, kCols);
  // My slot in both address spaces. Slot index is the PE, so each rank writes a
  // different half and the two never collide.
  unsigned short *mine = my_gather + (size_t)my_pe * kSlotElems;
  unsigned short *theirs = peer_gather + (size_t)my_pe * kSlotElems;

  for (int r = 0; r < iters; r++) {
    unsigned long long const expect = (unsigned long long)(r + 1);

    // ---- 9a: GPU-wide rendezvous, two-level (30 per XCD, then 8) ----------
    //
    // V_FULL9 only. Without this the other variants measure the exchange with
    // 8 blocks that were never gated on anything, which is why they are a floor
    // and not a model of Phase 9. This is the same tree the megakernel runs and
    // is what turns "the exchange costs 4 us" into "Phase 9 costs X".
    __shared__ int s_closer;
    if (full9) {
      __syncthreads();
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      bool last_on_gpu = false;
      if (tid == 0) {
        int px = atom_add_release_gpu_s32(&ep_xcd_arrive[blk * 16], 1);
        if (px % workers_per_xcd == workers_per_xcd - 1) {
          int pg = atom_add_release_gpu_s32(ep_moe_done, 1);
          last_on_gpu = (pg % kProducers == kProducers - 1);
        }
      }
      // V_NOREL drops the broadcast entirely. The release exists only to tell
      // the 8 folders "all 240 have arrived" -- but ep_moe_done ALREADY encodes
      // exactly that, and every folder can read it itself. The broadcast is a
      // second hop carrying information the first hop already published:
      //   FULL9:  arrive -> closer observes -> closer stores 8 flags -> folder
      //           observes flag            (2 serialized cross-XCD hops)
      //   NOREL:  arrive -> folder observes counter   (1 hop)
      // Correctness is unchanged: the counter is monotonic and reaches
      // 8*(r+1) at exactly the moment the last XCD leader arrives, which is the
      // same instant the closer would have begun its broadcast.
      if (variant == V_FULL9 && last_on_gpu) {
        threadfence_gpu();
        for (int x = 0; x < kProducers; x++) {
          asm volatile("global_store_dword %0, %1, off sc0 sc1"
                       :
                       : "v"(&ep_release[x * 16]), "v"((int)(r + 1))
                       : "memory");
        }
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      }
      (void)last_on_gpu;
      // EVERY block waits for the release, not just the folding one.
      //
      // In the megakernel the 240 workers stay in lockstep because they all run
      // the layer's compute and its barriers; here they would free-run, and
      // since the release value is derived from each block's own loop counter,
      // blocks at different r would publish and wait on different values and
      // wedge. Gating all of them on the release restores the lockstep the
      // megakernel gets for free. The non-folding blocks then idle exactly as
      // they do in situ.
      if (tid == 0) {
        if (variant == V_NOREL) {
          int const target = kProducers * (r + 1);
          while ((int)__builtin_nontemporal_load(ep_moe_done) < target) {
            __builtin_amdgcn_s_sleep(1);
          }
        } else {
          while ((int)__builtin_nontemporal_load(&ep_release[blk * 16]) <
                 (int)(r + 1)) {
            __builtin_amdgcn_s_sleep(1);
          }
        }
        s_closer = (xcd_rank == 0) ? 1 : 0;
      }
      __syncthreads();
      // Only one workgroup per XCD folds and exchanges; the other 29 fall
      // straight through to the next iteration, as in the megakernel.
      if (!s_closer) {
        continue;
      }
      asm volatile("buffer_inv" ::: "memory");
    }

    if (variant != V_SIG) {
      // Fold: read f32, convert to bf16, store to both copies. The remote store
      // is the transfer -- there is no staging buffer and no DMA descriptor.
      for (int c = col_lo + tid; c < col_hi; c += blockDim.x) {
        float v = src[c];
        unsigned short b = (unsigned short)(__float_as_uint(v) >> 16);
        mine[c] = b;
        theirs[c] = b;
      }
      __syncthreads();
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      threadfence_gpu();
    }

    if (variant == V_STORE) {
      continue; // transport only: no handshake at all
    }

    if (variant == V_PERXCD || variant == V_PERXCD8 || full9) {
      // Each producer publishes its own line the instant its slice has landed.
      if (tid == 0) {
        size_t off = (size_t)my_pe * kSigPeStride + (size_t)blk * kSigStride;
        st_wt_u64((void *)(peer_sig_base + off), expect);
        st_wt_u64((void *)(my_sig + off), expect);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      }
    } else {
      // V_AGG / V_SIG: count all 8 local folds, and let the 8th publish on
      // behalf of the whole GPU. This is the committed megakernel shape.
      __shared__ int s_last;
      if (tid == 0) {
        int prev = atom_add_release_gpu_s32(fold_done, 1);
        s_last = (prev % kProducers == kProducers - 1) ? 1 : 0;
      }
      __syncthreads();
      if (s_last && tid == 0) {
        size_t off = (size_t)my_pe * kSigPeStride;
        st_wt_u64((void *)(peer_sig_base + off), expect);
        st_wt_u64((void *)(my_sig + off), expect);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      }
    }

    // Wait. Both halves: my own rank's slot must be complete (its slices come
    // from 8 different blocks) and the peer's must have arrived.
    if (tid == 0) {
      for (int p = 0; p < 2; p++) {
        unsigned long long *base = my_sig + (size_t)p * kSigPeStride;
        if (variant == V_PERXCD8 || full9) {
          while (min8_nt(base) < expect) {
            __builtin_amdgcn_s_sleep(1);
          }
        } else {
          int const lines = (variant == V_PERXCD) ? kProducers : 1;
          for (int x = 0; x < lines; x++) {
            while (ld_nt_u64(base + (size_t)x * kSigStride) < expect) {
              __builtin_amdgcn_s_sleep(1);
            }
          }
        }
      }
    }
    __syncthreads();
    asm volatile("buffer_inv" ::: "memory");
  }
}

struct Rank {
  int dev;
  unsigned short *gather;
  unsigned long long *sig;
  float *src;
  int *fold_done;
  int *ctr; // 9a counters (V_FULL9)
  int *xcc;
  hipStream_t stream;
};

static char const *vname(int v) {
  switch (v) {
    case V_PERXCD: return "perxcd";
    case V_AGG:    return "agg";
    case V_STORE:  return "store";
    case V_SIG:    return "sig";
    case V_PERXCD8: return "perxcd8";
    case V_FULL9:  return "full9";
    case V_NOREL:  return "full9-norel";
  }
  return "?";
}

// One timed run of `iters` exchanges on both ranks concurrently. Returns
// us/iteration. Buffers are reset between runs so the monotonic thresholds
// always start from zero.
static double run_variant(Rank *r0, Rank *r1, int variant, int iters) {
  Rank *rs[2] = {r0, r1};
  for (int i = 0; i < 2; i++) {
    HIP_CHECK(hipSetDevice(rs[i]->dev));
    HIP_CHECK(hipMemset(rs[i]->sig, 0, 2 * kSigPeStride * sizeof(unsigned long long)));
    HIP_CHECK(hipMemset(rs[i]->fold_done, 0, sizeof(int)));
    HIP_CHECK(hipMemset(rs[i]->ctr, 0, 17 * 16 * sizeof(int)));
    HIP_CHECK(hipDeviceSynchronize());
  }

  hipEvent_t t0, t1;
  HIP_CHECK(hipSetDevice(r0->dev));
  HIP_CHECK(hipEventCreate(&t0));
  HIP_CHECK(hipEventCreate(&t1));
  HIP_CHECK(hipEventRecord(t0, r0->stream));

  for (int i = 0; i < 2; i++) {
    Rank *me = rs[i], *pe = rs[1 - i];
    HIP_CHECK(hipSetDevice(me->dev));
    // V_FULL9 launches the real occupancy: 240 workers, 30 per XCD. The other
    // variants launch 8 -- one per XCD -- because they deliberately exclude the
    // rendezvous whose whole cost is the 240-deep arrival.
    int const blocks =
        (variant == V_FULL9 || variant == V_NOREL) ? (kProducers * 30)
                                                   : kProducers;
    hipLaunchKernelGGL(ep_exchange_kernel, dim3(blocks), dim3(256), 0,
                       me->stream, me->gather, pe->gather, me->sig, pe->sig,
                       me->src, me->fold_done, i, iters, variant, me->xcc,
                       me->ctr, 30);
  }
  HIP_CHECK(hipSetDevice(r1->dev));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipSetDevice(r0->dev));
  HIP_CHECK(hipEventRecord(t1, r0->stream));
  HIP_CHECK(hipDeviceSynchronize());

  float ms = 0.f;
  HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
  HIP_CHECK(hipEventDestroy(t0));
  HIP_CHECK(hipEventDestroy(t1));
  return (double)ms * 1000.0 / (double)iters;
}

int main(int argc, char **argv) {
  int iters = (argc > 1) ? atoi(argv[1]) : 2000;

  // Same pair the demo scripts use by default (HIP_VISIBLE_DEVICES=6,7); after
  // that masking they are devices 0 and 1 to this process.
  int ndev = 0;
  HIP_CHECK(hipGetDeviceCount(&ndev));
  if (ndev < 2) {
    fprintf(stderr, "need 2 visible GPUs, saw %d\n", ndev);
    return 1;
  }

  Rank r[2];
  for (int i = 0; i < 2; i++) {
    r[i].dev = i;
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipMalloc(&r[i].gather, 2 * kSlotBytes));
    HIP_CHECK(hipMalloc(&r[i].sig, 2 * kSigPeStride * sizeof(unsigned long long)));
    HIP_CHECK(hipMalloc(&r[i].src, kCols * sizeof(float)));
    HIP_CHECK(hipMalloc(&r[i].fold_done, sizeof(int)));
    HIP_CHECK(hipMalloc(&r[i].ctr, 17 * 16 * sizeof(int)));
    HIP_CHECK(hipMalloc(&r[i].xcc, kProducers * sizeof(int)));
    HIP_CHECK(hipMemset(r[i].gather, 0, 2 * kSlotBytes));
    HIP_CHECK(hipMemset(r[i].src, 0, kCols * sizeof(float)));
    HIP_CHECK(hipStreamCreate(&r[i].stream));
  }
  // Peer access both ways. Without this the peer pointer is not dereferenceable
  // and every remote store faults.
  for (int i = 0; i < 2; i++) {
    int can = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&can, i, 1 - i));
    if (!can) {
      fprintf(stderr, "no peer access %d<->%d\n", i, 1 - i);
      return 1;
    }
    HIP_CHECK(hipSetDevice(i));
    hipError_t e = hipDeviceEnablePeerAccess(1 - i, 0);
    if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
      HIP_CHECK(e);
    }
  }

  printf("payload %d B/slot, %d producers, %d iters\n", kSlotBytes, kProducers,
         iters);

  int order[7] = {V_STORE, V_SIG, V_AGG, V_PERXCD, V_PERXCD8, V_FULL9, V_NOREL};
  for (int oi = 0; oi < 7; oi++) {
    int v = order[oi];
    run_variant(&r[0], &r[1], v, 200); // warm up: first touch of the peer
                                       // mapping is not representative
    double best = 1e30;
    for (int rep = 0; rep < 3; rep++) {
      best = std::min(best, run_variant(&r[0], &r[1], v, iters));
    }
    printf("  %-7s %8.3f us/exchange   (x36 layers = %6.3f ms/token)\n",
           vname(v), best, best * 36.0 / 1000.0);
  }

  std::vector<int> xcc(kProducers);
  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipMemcpy(xcc.data(), r[0].xcc, kProducers * sizeof(int),
                      hipMemcpyDeviceToHost));
  printf("block->XCC map (rank 0):");
  for (int i = 0; i < kProducers; i++) {
    printf(" %d", xcc[i]);
  }
  printf("\n");
  return 0;
}
