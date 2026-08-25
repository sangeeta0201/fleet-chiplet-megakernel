// What does the RELEASE half of a Mechanism-C rendezvous actually cost, and
// which of its five ingredients is the bill?
//
// WHY THIS TEST EXISTS. On the GLM megakernel the rendezvous class is 2.371 ms
// of a 10.27 ms token (glm-whole-wall-closes-tiles-are-3x-off-byte-roof) and
// its byte roof is zero -- it is pure mechanism. Commit 2d134ef priced the
// ARRIVAL side directly and got 19 ns: a 232-way device-scope atomic on one
// address does not serialize, the L2 atomic unit pipelines same-address adds.
// Against a measured 3.77 us per rendezvous (glm-one-rendezvous-costs-3.77us)
// that leaves 99.5% of a barrier in the RELEASE half, which nothing has ever
// decomposed. mpk_atoms.cuh's own note ends "Any further work here belongs on
// the release side."
//
// THE SHAPE UNDER TEST, copied from gang_mla_attn_fused_mi300.cuh:746-784 --
// every one of the layer's ten rendezvous is this same block:
//
//   if (tid == 0) {
//     if (hier_barrier_arrive(bar, ...)) {            // atomic, 19 ns
//       for (x = 0; x < 8; x++)                       // ONE thread in the
//         st_wt_u32(&bar[x * HIER_STRIDE], epoch);    // grid, 8 lines
//       s_waitcnt vmcnt(0);
//     }
//     while (ld_nt_s32(&bar[xcd_id * HIER_STRIDE]) < epoch)   // 29 pollers
//       s_sleep(1);                                           // per line
//   }
//   __syncthreads();
//   buffer_inv;
//
// Five candidate ingredients, and the point of the harness is that they are
// separable:
//
//   1. the store's trip to the coherence point before any reader can see it
//   2. POLL STALENESS -- the in-flight load that was issued just before the
//      release lands returns the old value, so detection costs 1-2 whole poll
//      periods, and a period is a full uncached round trip plus the sleep
//   3. LINE CONTENTION -- 29 blocks per flag line, every one of them issuing
//      a cache-bypassing dword at the same address. `nt` is a
//      replacement-policy hint that asks MALL *not* to retain the line, so
//      the read-shared flag may be re-fetched from DRAM by every poller
//      instead of being served hot out of MALL. This is the ingredient the
//      author of ld_nt_s32 flagged: "Cost note: the poll now goes to MALL/HBM
//      instead of L2 ... if the barrier tail regresses, this is the first
//      thing to re-measure."
//   4. the s_sleep(1) quantum itself
//   5. buffer_inv, paid by all 240 blocks after the gate opens
//
// CALIBRATION. This harness must reproduce the in-situ number before any arm
// of it is believed. glm-round-fixed-cost-is-the-rendezvous measured a NULL
// rendezvous -- barriers back to back with no work between them -- at 2.71 us,
// and the in-situ figure with a phase in front of it is 3.77 us. A0 landing in
// that band is the check that the replica is faithful; A0 landing at 0.3 us
// would mean the harness is measuring something else and no arm below is
// evidence about the megakernel.
//
// WHAT IS DELIBERATELY NOT MODELLED: arrival spread. Every block here arrives
// at the same instant, which is the null-rendezvous condition and is also,
// per glm-makespan-predictor-three-regimes, close to the real one -- max minus
// second-arrival is 0.005-0.369 us at all ten real rendezvous.
//
// Build:  hipcc -o test_barrier_release test_barrier_release.hip \
//             -D__HIP_PLATFORM_AMD__ --offload-arch=gfx950 -O3 -std=c++17
// Run:    HIP_VISIBLE_DEVICES=4 ./test_barrier_release [rounds]

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(x)                                                           \
  do {                                                                         \
    hipError_t _e = (x);                                                       \
    if (_e != hipSuccess) {                                                    \
      printf("HIP error %s at line %d\n", hipGetErrorString(_e), __LINE__);    \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Matches the megakernel: 232 worker blocks, 256 threads, 16 int32 (one 64 B
// line) per barrier slot, 8 logical XCDs.
static constexpr int NBLOCKS = 232;
static constexpr int NTHREADS = 256;
static constexpr int HIER_STRIDE = 16;
static constexpr int NXCD = 8;

// ---------------------------------------------------------------------------
// The primitives, lifted verbatim from mpk_atoms.cuh so the arms differ from
// the shipping code in exactly one thing each.
// ---------------------------------------------------------------------------

__device__ __forceinline__ int ld_flag_ntsys(int *addr) {
  int v;
  asm volatile("global_load_dword %0, %1, off sc0 sc1 nt\n"
               "s_waitcnt vmcnt(0)"
               : "=v"(v)
               : "v"(addr)
               : "memory");
  return v;
}

// Arm 1: same coherence (sc0 sc1 bypasses L1 and L2, so this is still a
// system-scope load and still correct) minus the non-temporal hint, so MALL
// is allowed to retain the read-shared flag line.
__device__ __forceinline__ int ld_flag_sys(int *addr) {
  int v;
  asm volatile("global_load_dword %0, %1, off sc0 sc1\n"
               "s_waitcnt vmcnt(0)"
               : "=v"(v)
               : "v"(addr)
               : "memory");
  return v;
}

// Arm 6: four samples of the same line in flight at once. Staleness falls to
// a quarter of a round trip; the round trip itself is unchanged. Reads the
// LAST one, which is the freshest.
__device__ __forceinline__ int ld_flag_pipe4(int *addr) {
  int v0, v1, v2, v3;
  asm volatile("global_load_dword %0, %4, off sc0 sc1 nt\n"
               "global_load_dword %1, %4, off sc0 sc1 nt\n"
               "global_load_dword %2, %4, off sc0 sc1 nt\n"
               "global_load_dword %3, %4, off sc0 sc1 nt\n"
               "s_waitcnt vmcnt(0)"
               : "=v"(v0), "=v"(v1), "=v"(v2), "=v"(v3)
               : "v"(addr)
               : "memory");
  return v3;
}

// TWO-LEVEL RELEASE. The block sweep says the release half scales with the
// NUMBER OF PARTICIPANTS reaching the coherence point -- 0.872 us at one block,
// 3.809 at 232 -- and that it is not flag-line sharing (32 lines is a wash, 232
// private lines is worse) and not poll rate (sleep 0..128 is flat). What is
// left is aggregate uncached request traffic: every one of the 232 pollers
// bypasses L1 AND the per-XCD L2 on every spin, because a flag written from
// another XCD is not visible in this XCD's unsnooped L2.
//
// But that is only true for a flag written from ANOTHER XCD. Blocks sharing a
// physical XCD share an L2, and an L2 is a perfectly good coherence point for
// them. So: let one leader per XCD pay the MALL poll, and have its 28 peers
// observe an L2-resident line instead. MALL pollers drop 232 -> 8, which by
// the sweep is 3.809 -> 1.320 us.
//
// RESULT: FALSIFIED. Once the load bit is right (see below) the arm runs
// correctly -- no heal, no hang -- and lands at 3.801 us against A1's 3.779.
// Cutting MALL pollers 232 -> 8 buys EXACTLY NOTHING. So the release residue is
// not aggregate poll traffic; it is the latency of the publish reaching an
// observer, and the two-level scheme just pays that latency twice in series
// (leader observes MALL, peers observe L2) instead of once in parallel. The
// block sweep's participant scaling is a queueing artefact of 232 blocks
// arriving at once, not a bandwidth wall. The remaining lever on this class is
// to have FEWER RENDEZVOUS, not cheaper ones.
//
// MEASURED SEMANTICS (probe: writer bid 0, reader bid 8 vs bids 1/2/5, and
// bid % 8 == HW_REG_XCC_ID verified 232/232):
//
//   ld sc0        never observes another block's store at all -- 0/500 even
//                 against an sc0 sc1 write-through store. sc0 on a LOAD does
//                 not bypass L1; the poller spins on its own stale L1 line.
//   ld sc1        bypasses L1 and is answered by this XCD's L2. 500/500 at
//                 0.85 us against a same-XCD sc0 store; 0/500 against that
//                 same store from another XCD.
//   st sc0        lands in the writer's XCD L2 and does not leave it.
//
// So `st sc0` + `ld sc1` is exactly the intra-XCD publish/observe pair this
// arm needs, and the first version of it was wrong only in the load bit.
__device__ __forceinline__ int ld_l2_s32(int *addr) {
  int v;
  asm volatile("global_load_dword %0, %1, off sc1\n"
               "s_waitcnt vmcnt(0)"
               : "=v"(v)
               : "v"(addr)
               : "memory");
  return v;
}

// L2-scope store: no sc1, so it lands in this XCD's L2 and stays there.
// The waitcnt is not optional -- without it the leader races on to the next
// round with the publish still in flight, and its peers spin on the old value.
__device__ __forceinline__ void st_l2_u32(void *addr, unsigned val) {
  asm volatile("global_store_dword %0, %1, off sc0\n"
               "s_waitcnt vmcnt(0)"
               :
               : "v"(addr), "v"(val)
               : "memory");
}

// Durable publish: sc0 sc1 writes THROUGH to MALL, so the value survives the
// per-round `buffer_inv` that would otherwise discard an L2-only line. Peers
// still READ it with a plain sc0 load, which their own XCD's L2 can answer
// once the write-through has allocated the line there. This is the POLL == 5
// variant, and the difference between it and POLL == 4 isolates whether the
// two-level arm fails on durability or on the XCD mapping.
__device__ __forceinline__ void st_l2_durable_u32(void *addr, unsigned val) {
  asm volatile("global_store_dword %0, %1, off sc0 sc1\n"
               "s_waitcnt vmcnt(0)"
               :
               : "v"(addr), "v"(val)
               : "memory");
}

// Physical XCD of this block. On MI300/MI350 the dispatcher assigns workgroups
// to XCDs round-robin, so blockIdx % 8 is the physical XCD for a grid at least
// as large as the XCD count. THIS IS THE ASSUMPTION THE WHOLE ARM RESTS ON, and
// it is why the peer poll below carries a heal: if the mapping is wrong the
// peers are polling a line in some other XCD's L2, the heal fires, and the arm
// measures SLOWER rather than hanging. A hang would tell us nothing; a
// slowdown tells us the assumption is false.
__device__ __forceinline__ int phys_xcd(int bid) { return bid & (NXCD - 1); }

__device__ __forceinline__ void st_wt_u32_l(void *addr, unsigned val) {
  asm volatile("global_store_dword %0, %1, off sc0 sc1"
               :
               : "v"(addr), "v"(val)
               : "memory");
}

// ---------------------------------------------------------------------------
// One kernel, templated on the arm, so every arm shares the identical loop,
// grid, and timing path.
//
//   POLL  0 = sc0 sc1 nt (shipping)  1 = sc0 sc1  2 = 4-deep pipelined nt
//   SLEEP     s_sleep argument; 0 means no sleep at all
//   FLAGS 8 = per-XCD lines (shipping)   1 = one line for the whole grid
//   INV   1 = buffer_inv after the gate (shipping)   0 = omitted
// ---------------------------------------------------------------------------
template <int POLL, int SLEEP, int FLAGS, int INV>
__global__ __launch_bounds__(NTHREADS) void barrier_arm(int *bar,
                                                        int rounds,
                                                        unsigned long long *out,
                                                        int arrivals) {
  int const tid = threadIdx.x;
  int const bid = blockIdx.x;
  // Same derivation as the real task when FLAGS == 8: a logical XCD with
  // arrivals/8 blocks behind each line. FLAGS == arrivals gives every block a
  // private line; anything between shares proportionally.
  int const xcd_id =
      (FLAGS == 1) ? 0
                   : ((FLAGS >= NBLOCKS) ? bid : (bid * FLAGS) / arrivals);

  unsigned long long t0 = 0;
  if (tid == 0 && bid == 0) {
    t0 = __builtin_amdgcn_s_memrealtime();
  }

  for (int r = 1; r <= rounds; r++) {
    if (tid == 0) {
      // ---- arrival: one device-scope atomic, priced at 19 ns and not the
      // subject of this test.
      // Counter parked past the widest flag block so the 232-private-line arm
      // cannot alias it. In the megakernel it sits at slot 8 because there are
      // only ever 8 flags.
      int const prev = __hip_atomic_fetch_add(&bar[(NBLOCKS + 1) * HIER_STRIDE],
                                              1,
                                              __ATOMIC_RELEASE,
                                              __HIP_MEMORY_SCOPE_AGENT);
      bool const last = (prev % arrivals) == arrivals - 1;
      // POLL == 3 is the FLOOR arm: it keeps the arrival atomic, the loop, the
      // __syncthreads and the buffer_inv, and drops only the release and the
      // observation. It does NOT synchronise -- that is the point. It is the
      // cost of everything a rendezvous does that is not the release half.
      if (POLL == 3) {
        (void)last;
      } else if (last) {
        // ---- release fan-out: one thread in the grid, FLAGS write-through
        // stores, issued back to back and retired together.
#pragma unroll
        for (int x = 0; x < FLAGS; x++) {
          st_wt_u32_l((void *)&bar[x * HIER_STRIDE], (unsigned)r);
        }
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      }
      // ---- observation
      int *const flag = &bar[xcd_id * HIER_STRIDE];
      // POLL == 4: two-level. The XCD leader observes MALL and republishes to
      // an L2-resident line; the other 28 blocks on its XCD never touch MALL.
      // Leader lines live past the flag block AND past the counter.
      if (POLL == 4 || POLL == 5) {
        int const px = phys_xcd(bid);
        int *const l2flag = &bar[(NBLOCKS + 2 + px) * HIER_STRIDE];
        if (bid < NXCD) {
          // Leader: bid 0..7 are, by the round-robin assumption, one block on
          // each distinct physical XCD.
          while (ld_flag_sys(flag) < r) {
            __builtin_amdgcn_s_sleep(SLEEP);
          }
          if (POLL == 4) {
            st_l2_u32((void *)l2flag, (unsigned)r);
          } else {
            st_l2_durable_u32((void *)l2flag, (unsigned)r);
          }
        } else {
          int spins = 0;
          while (ld_l2_s32(l2flag) < r) {
            // Heal, and the arm's own falsification test: if blockIdx % 8 is
            // not the physical XCD then this fires every time and the arm is
            // simply the shipping poll plus overhead.
            if ((++spins & 63) == 0 && ld_flag_sys(flag) >= r) {
              break;
            }
            __builtin_amdgcn_s_sleep(SLEEP);
          }
        }
      }
      int obs;
      while (POLL != 3 && POLL != 4 && POLL != 5) {
        if (POLL == 0) {
          obs = ld_flag_ntsys(flag);
        } else if (POLL == 1) {
          obs = ld_flag_sys(flag);
        } else {
          obs = ld_flag_pipe4(flag);
        }
        if (obs >= r) {
          break;
        }
        if (SLEEP > 0) {
          __builtin_amdgcn_s_sleep(SLEEP);
        }
      }
    }
    __syncthreads();
    if (INV) {
      asm volatile("buffer_inv" ::: "memory");
    }
  }

  if (tid == 0 && bid == 0) {
    unsigned long long const t1 = __builtin_amdgcn_s_memrealtime();
    *out = (t1 - t0) * 10ull; // 100 MHz constant clock -> ns
  }
}

struct Arm {
  char const *name;
  void (*fn)(int *, int, unsigned long long *, int);
};

// The launcher has to be a real function pointer per arm; a lambda over a
// template parameter will not do because the kernel address is what hipLaunch
// needs.
#define ARM(P, S, F, I) barrier_arm<P, S, F, I>

// The block-count sweep. The arm table above says which INGREDIENT to change;
// this says whether the residue is LATENCY or CONTENTION, which decides
// whether there is anything left to attack at all. The release half is
// 4.120 - 1.835 = 2.285 us and the best ingredient swap takes 0.5 of it; the
// other 1.8 us is worth 1.4 ms/token if it has structure and nothing if it is
// a round trip.
//
//   flat vs block count  -> pure round-trip latency, irreducible from here
//   rising               -> sharing cost, and the slope says how much
//
// Grid is the only variable; `arrivals` tracks it so the barrier still
// completes, and the flag-line count stays at the shipping 8.
template <int POLL>
static double sweep_one(int *bar, unsigned long long *out, int blocks,
                        int rounds, size_t barBytes) {
  double best = 1e30;
  for (int rep = 0; rep < 3; rep++) {
    HIP_CHECK(hipMemset(bar, 0, barBytes));
    hipLaunchKernelGGL((barrier_arm<POLL, 1, 8, 1>), dim3(blocks),
                       dim3(NTHREADS), 0, 0, bar, 64, out, blocks);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemset(bar, 0, barBytes));
    hipLaunchKernelGGL((barrier_arm<POLL, 1, 8, 1>), dim3(blocks),
                       dim3(NTHREADS), 0, 0, bar, rounds, out, blocks);
    HIP_CHECK(hipDeviceSynchronize());
    unsigned long long ns = 0;
    HIP_CHECK(hipMemcpy(&ns, out, sizeof(ns), hipMemcpyDeviceToHost));
    double const per = (double)ns / (double)rounds / 1000.0;
    if (per < best) {
      best = per;
    }
  }
  return best;
}

int main(int argc, char **argv) {
  int const rounds = (argc > 1) ? atoi(argv[1]) : 2000;

  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  printf("device: %s  CUs=%d\n", prop.gcnArchName, prop.multiProcessorCount);
  printf("grid: %d blocks x %d threads, %d flag lines, %d rounds/arm\n\n",
         NBLOCKS, NTHREADS, NXCD, rounds);

  int *bar = nullptr;
  unsigned long long *out = nullptr;
  // Sized for the widest arm: one private 64 B line per block, plus the
  // arrival counter, which the kernel indexes at NXCD * HIER_STRIDE and which
  // therefore has to live clear of the flag block -- put it past all of them.
  size_t const barBytes = (size_t)(NBLOCKS + 2 * NXCD + 4) * HIER_STRIDE *
                          sizeof(int);
  HIP_CHECK(hipMalloc(&bar, barBytes));
  HIP_CHECK(hipMalloc(&out, sizeof(unsigned long long)));

  struct Named {
    char const *name;
    void (*fn)(int *, int, unsigned long long *, int);
  };
  // One variable per arm against A0. A2/A3 change only the sleep, A4 only the
  // flag count, A5 only buffer_inv, A6 only the poll depth.
  std::vector<Named> arms = {
      {"A0 shipping  nt  sleep1  8flags  inv", ARM(0, 1, 8, 1)},
      {"A1 drop nt       sleep1  8flags  inv", ARM(1, 1, 8, 1)},
      {"A2 nt        sleep0  8flags  inv    ", ARM(0, 0, 8, 1)},
      {"A3 nt        sleep4  8flags  inv    ", ARM(0, 4, 8, 1)},
      {"A4 nt        sleep1  1flag   inv    ", ARM(0, 1, 1, 1)},
      {"A5 nt        sleep1  8flags  noinv  ", ARM(0, 1, 8, 0)},
      {"A6 pipe4     sleep1  8flags  inv    ", ARM(2, 1, 8, 1)},
      {"A7 drop nt   sleep0  8flags  inv    ", ARM(1, 0, 8, 1)},
      {"A8 drop nt   sleep1  1flag   inv    ", ARM(1, 1, 1, 1)},
      // The contention axis. A4 says 1 line is 0.478 us worse than 8, i.e.
      // 29 pollers per line already costs something. Push it the other way:
      // 32 lines is 8 per line, 232 is one PRIVATE line per block and zero
      // sharing. The release side pays more stores, but they are issued back
      // to back by one thread and retired by a single s_waitcnt, so they cost
      // issue slots and one round trip, not N round trips.
      {"B0 nt        sleep1  32flags inv    ", ARM(0, 1, 32, 1)},
      {"B1 nt        sleep1  232priv inv    ", ARM(0, 1, 232, 1)},
      {"B2 drop nt   sleep1  32flags inv    ", ARM(1, 1, 32, 1)},
      {"B3 drop nt   sleep1  232priv inv    ", ARM(1, 1, 232, 1)},
      {"B4 drop nt   sleep0  232priv inv    ", ARM(1, 0, 232, 1)},
      // The floor: arrival atomic + __syncthreads + buffer_inv, NO release and
      // NO poll. Whatever this costs is not attackable from the release side
      // at all, and every arm above should be read against it rather than
      // against zero.
      {"F0 arrival + syncthreads only       ", ARM(3, 1, 8, 1)},
      // The poll-RATE axis, which A2/A3 did not actually vary. s_sleep(1) is
      // 64 clocks ~= 26 ns against a ~700 ns uncached round trip, so sleep 0,
      // 1 and 4 are all "poll as fast as the fabric allows" and measured
      // identical. If the bill is aggregate uncached request traffic from 232
      // simultaneous pollers rather than any one line, only a back-off long
      // enough to matter next to the round trip can show it. s_sleep(16) is
      // ~427 ns, s_sleep(64) ~1.7 us. Too long trades detection latency for
      // traffic, so this is an optimum-finding sweep, not a one-way bet.
      {"C0 drop nt   sleep8   8flags inv    ", ARM(1, 8, 8, 1)},
      {"C1 drop nt   sleep16  8flags inv    ", ARM(1, 16, 8, 1)},
      {"C2 drop nt   sleep32  8flags inv    ", ARM(1, 32, 8, 1)},
      {"C3 drop nt   sleep64  8flags inv    ", ARM(1, 64, 8, 1)},
      {"C4 drop nt   sleep128 8flags inv    ", ARM(1, 128, 8, 1)},
      // Two-level: 8 MALL pollers, 224 L2 pollers. See ld_l2_s32.
      {"D0 two-level  L2 pub   sleep1  inv   ", ARM(4, 1, 8, 1)},
      {"D1 two-level  L2 pub   sleep1  noinv ", ARM(4, 1, 8, 0)},
      {"D2 two-level  wt pub   sleep1  inv   ", ARM(5, 1, 8, 1)},
      {"D3 two-level  wt pub   sleep1  noinv ", ARM(5, 1, 8, 0)},
      {"D4 two-level  wt pub   sleep0  inv   ", ARM(5, 0, 8, 1)},
  };

  double base = 0.0;
  for (size_t i = 0; i < arms.size(); i++) {
    // Three timed repeats per arm, report the MIN: this box injects
    // multi-second single-iteration stalls and the min is the stall-immune
    // statistic (glm5-wall-noise-floor-is-0.26ms's method, scaled down).
    double best = 1e30;
    for (int rep = 0; rep < 3; rep++) {
      HIP_CHECK(hipMemset(bar, 0, barBytes));
      // Warm the arm: first launch pays code-object paging and a cold MALL.
      hipLaunchKernelGGL(arms[i].fn, dim3(NBLOCKS), dim3(NTHREADS), 0, 0, bar,
                         64, out, NBLOCKS);
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(hipMemset(bar, 0, barBytes));
      hipLaunchKernelGGL(arms[i].fn, dim3(NBLOCKS), dim3(NTHREADS), 0, 0, bar,
                         rounds, out, NBLOCKS);
      HIP_CHECK(hipDeviceSynchronize());
      unsigned long long ns = 0;
      HIP_CHECK(hipMemcpy(&ns, out, sizeof(ns), hipMemcpyDeviceToHost));
      double const per = (double)ns / (double)rounds / 1000.0; // us
      if (per < best) {
        best = per;
      }
    }
    if (i == 0) {
      base = best;
    }
    printf("%s  %7.3f us/barrier   %+7.3f vs A0  %6.2fx\n", arms[i].name, best,
           best - base, base / best);
    fflush(stdout);
  }

  printf("\n--- block-count sweep, 8 flag lines, sleep1, inv ---\n");
  printf("%8s  %10s  %10s  %10s\n", "blocks", "nt us", "no-nt us", "delta");
  int const bs[] = {1, 8, 32, 64, 116, 232};
  for (int i = 0; i < 6; i++) {
    double const a = sweep_one<0>(bar, out, bs[i], rounds, barBytes);
    double const b = sweep_one<1>(bar, out, bs[i], rounds, barBytes);
    printf("%8d  %10.3f  %10.3f  %+10.3f\n", bs[i], a, b, b - a);
    fflush(stdout);
  }
  printf("Flat in `blocks` means the residue is round-trip latency and there is\n"
         "nothing left to take here; rising means sharing, and the slope prices\n"
         "it. Note 1 block still pays store-then-observe, so the 1-block row IS\n"
         "the latency floor of this mechanism.\n");

  printf("\nIn-situ calibration: a NULL rendezvous measured 2.71 us and a real\n"
         "one 3.77 us. If A0 is far outside that band the replica is wrong and\n"
         "no arm above is evidence about the megakernel.\n");
  printf("Per-token value of a delta: x 10 barriers x 76 layers = x 0.76 ms/us.\n");

  HIP_CHECK(hipFree(bar));
  HIP_CHECK(hipFree(out));
  return 0;
}
