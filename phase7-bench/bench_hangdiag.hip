// Why does the Phase 7 flag handshake hang under a coherent MTYPE in SPX+NPS2?
//
// bench_flagplace spins forever there, which tells us a poller never observes a
// store but not why. This bounds every spin with a deadline and, at the instant
// a wave gives up, reads the very same address four ways:
//
//   plain  global_load_dword sc0 sc1   the load the spin loop was using
//   nosc   global_load_dword           no scope bits, may hit vL1
//   atom   global_atomic_add(0)        an RMW, must be served at the coherence point
//   inv    buffer_inv then sc0 sc1     after explicitly invalidating the cache
//
// Those four discriminate the possibilities:
//
//   all four stale        the store never reached memory at all (producer side)
//   atom fresh, plain stale
//                         the store IS at the coherence point and the poller is
//                         being served a stale cached copy, i.e. the invalidate
//                         probe never arrived (coherence side)
//   inv fresh             confirms it is a cached line, and that invalidating
//                         by hand is enough to see the value
//
// Structure mirrors bench_flagplace: one producer block per XCD writes its own
// flag, and wave w of every block polls flags 2w and 2w+1, so every XCD polls
// flags owned by every other XCD. There is deliberately no cross-block barrier
// here, so a stuck wave can bail and let the kernel retire instead of wedging
// the GPU.
//
// --pdelay matters. If the producer writes before consumers ever read, the
// consumer may fetch the line fresh and never need a probe. Holding the store
// back guarantees each consumer has the stale line cached first, which is the
// case where a lost invalidate turns into an infinite spin.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define NXCD 8
#define NWAVES 4
#define NTHREADS (NWAVES * 64)
#define DBGF 8 // fields per wave in the debug record

#define HIP_OK(call)                                                           \
  do {                                                                         \
    hipError_t _e = (call);                                                    \
    if (_e != hipSuccess) {                                                    \
      fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #call,           \
              hipGetErrorString(_e));                                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

__device__ unsigned int g_claim[NXCD];

__device__ __forceinline__ int get_xcd() {
  int x;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(x));
  return x;
}

__device__ __forceinline__ void st_wt_u32(void *addr, unsigned int val) {
  asm volatile("global_store_dword %0, %1, off sc0 sc1"
               :
               : "v"(addr), "v"(val)
               : "memory");
}

__device__ __forceinline__ int ld_sys_s32(int *addr) {
  int val;
  asm volatile("global_load_dword %0, %1, off sc0 sc1\n"
               "s_waitcnt vmcnt(0)"
               : "=v"(val)
               : "v"(addr)
               : "memory");
  return val;
}

__device__ __forceinline__ int ld_nosc_s32(int *addr) {
  int val;
  asm volatile("global_load_dword %0, %1, off\n"
               "s_waitcnt vmcnt(0)"
               : "=v"(val)
               : "v"(addr)
               : "memory");
  return val;
}

__device__ __forceinline__ int ld_after_inv_s32(int *addr) {
  int val;
  asm volatile("buffer_inv sc0 sc1\n"
               "global_load_dword %0, %1, off sc0 sc1\n"
               "s_waitcnt vmcnt(0)"
               : "=v"(val)
               : "v"(addr)
               : "memory");
  return val;
}

// An atomic RMW cannot be answered from a stale local copy; it is ordered at
// the point of coherence. Adding zero makes it a pure read of that point.
__device__ __forceinline__ int ld_atomic_s32(int *addr) {
  return __hip_atomic_fetch_add(addr, 0, __ATOMIC_ACQUIRE,
                                __HIP_MEMORY_SCOPE_SYSTEM);
}

// Push any dirty line out of this XCD's L2. Under a writeback MTYPE the store
// can sit dirty in the producer's cache, in which case a remote reader only
// ever sees memory, and memory is stale until someone writes it back.
__device__ __forceinline__ void wb_l2() {
  asm volatile("buffer_wbl2 sc0 sc1\n"
               "s_waitcnt vmcnt(0)" ::
                   : "memory");
}

// Is one virtual address one physical location for every XCD?
//
// Every XCD's rank-0 block stores a value only it would write into the SAME
// dword, they all settle, then they all read it back. There are exactly two
// possible outcomes and they mean completely different things:
//
//   all eight read the same value   one location, and the writes are ordered;
//                                   any hang would then be a coherence or
//                                   timing problem
//   each reads back its own value   eight different physical locations behind
//                                   one virtual address, i.e. the accesses are
//                                   being resolved inside each XCD's own
//                                   memory partition. No amount of cache
//                                   maintenance can fix that.
__global__ __launch_bounds__(NTHREADS) void aliastest(int *flags, int *out,
                                                      int tiles,
                                                      unsigned int settle,
                                                      int wb) {
  int const xcd = get_xcd();
  __shared__ int s_local;
  if (threadIdx.x == 0) {
    s_local = (int)atomicAdd(&g_claim[xcd], 1u);
  }
  __syncthreads();
  if (s_local != 0 || threadIdx.x != 0) {
    return;
  }

  int *p = &flags[0];
  int const mine = 0x1000 + xcd;

  unsigned long long const t0 = __builtin_amdgcn_s_memrealtime();
  st_wt_u32(p, (unsigned)mine);
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  if (wb) {
    wb_l2();
  }

  // Let every XCD finish writing before anybody reads.
  while (__builtin_amdgcn_s_memrealtime() < t0 + settle) {
    __builtin_amdgcn_s_sleep(8);
  }

  out[xcd * 4 + 0] = mine;
  out[xcd * 4 + 1] = ld_sys_s32(p);
  out[xcd * 4 + 2] = ld_atomic_s32(p);
  out[xcd * 4 + 3] = ld_after_inv_s32(p);
}

// Which XCD can see which XCD's store, once the reader has the line cached?
//
// The alias test reads each line exactly once and succeeds, so it cannot see
// the failure; a poller fails precisely because it holds the line from before
// the store. This runs the three phases a poller goes through, separated by
// milliseconds so the eight participating blocks stay in step without needing a
// barrier (a barrier would be subject to the very failure being measured):
//
//   1. every XCD reads all eight flags, caching all eight lines while zero
//   2. every XCD stores its own flag
//   3. every XCD reads all eight flags again
//
// Phase 3 is the answer. A blank entry means that reader never saw that
// writer's store. If only the far-AID half is blank the failure is cross-AID;
// if everything off the diagonal is blank, coherence is not working between any
// two XCDs of this compute partition.
__global__ __launch_bounds__(NTHREADS) void vismatrix(int *flags, int *out,
                                                      int stride_ints,
                                                      unsigned int gap) {
  int const xcd = get_xcd();
  __shared__ int s_local;
  if (threadIdx.x == 0) {
    s_local = (int)atomicAdd(&g_claim[xcd], 1u);
  }
  __syncthreads();
  if (s_local != 0 || threadIdx.x != 0) {
    return;
  }

  unsigned long long const t0 = __builtin_amdgcn_s_memrealtime();
  int sink = 0;
  for (int x = 0; x < NXCD; ++x) {
    sink += ld_sys_s32(&flags[x * stride_ints]);
    out[NXCD * NXCD + x] = sink; // keep the reads, and record phase-1 values
  }

  while (__builtin_amdgcn_s_memrealtime() < t0 + gap) {
    __builtin_amdgcn_s_sleep(8);
  }
  st_wt_u32(&flags[xcd * stride_ints], (unsigned)(0x100 + xcd));
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

  while (__builtin_amdgcn_s_memrealtime() < t0 + 3ull * gap) {
    __builtin_amdgcn_s_sleep(8);
  }
  for (int x = 0; x < NXCD; ++x) {
    out[xcd * NXCD + x] = ld_sys_s32(&flags[x * stride_ints]);
  }
}

__global__ __launch_bounds__(NTHREADS) void hangdiag(
    int *flags, int *dbg, int *seen, int *prod_dbg, int tiles, int stride_ints,
    int n_layers, unsigned long long bail_ticks, unsigned int pdelay, int wb,
    int cinv) {
  int const xcd = get_xcd();
  int const tid = threadIdx.x;
  int const warp = tid >> 6;
  int const lane = tid & 63;

  __shared__ int s_local;
  if (tid == 0) {
    s_local = (int)atomicAdd(&g_claim[xcd], 1u);
    atomicAdd(&seen[xcd], 1);
  }
  __syncthreads();
  int const local = s_local;
  if (local >= tiles) {
    return;
  }

  int const first_xcd = warp * 2;
  int const dbg_row = ((int)blockIdx.x * NWAVES + warp) * DBGF;

  for (int layer = 1; layer <= n_layers; ++layer) {
    // ---- producer: rank 0 of each XCD publishes that XCD's own flag ----
    if (local == 0 && tid == 0) {
      if (pdelay) {
        unsigned long long const until =
            __builtin_amdgcn_s_memrealtime() + (unsigned long long)pdelay;
        while (__builtin_amdgcn_s_memrealtime() < until) {
          __builtin_amdgcn_s_sleep(1);
        }
      }
      st_wt_u32(&flags[xcd * stride_ints], (unsigned)layer);
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      if (wb) {
        wb_l2();
      }
      // Did the producer's own store become visible to the producer itself?
      int const r = xcd * 4;
      prod_dbg[r + 0] = layer;
      prod_dbg[r + 1] = ld_sys_s32(&flags[xcd * stride_ints]);
      prod_dbg[r + 2] = ld_atomic_s32(&flags[xcd * stride_ints]);
      prod_dbg[r + 3] = xcd;
    }

    // ---- consumer: every wave waits on the two flags of its slice pair ----
    int stuck = 0;
    unsigned long long const deadline =
        __builtin_amdgcn_s_memrealtime() + bail_ticks;
    for (int k = 0; k < 2 && !stuck; ++k) {
      int *p = &flags[(first_xcd + k) * stride_ints];
      while ((cinv ? ld_after_inv_s32(p) : ld_sys_s32(p)) < layer) {
        if (__builtin_amdgcn_s_memrealtime() > deadline) {
          if (lane == 0) {
            dbg[dbg_row + 0] = 1;
            dbg[dbg_row + 1] = xcd;
            dbg[dbg_row + 2] = first_xcd + k;
            dbg[dbg_row + 3] = layer;
            dbg[dbg_row + 4] = ld_sys_s32(p);
            dbg[dbg_row + 5] = ld_nosc_s32(p);
            dbg[dbg_row + 6] = ld_atomic_s32(p);
            dbg[dbg_row + 7] = ld_after_inv_s32(p);
          }
          stuck = 1;
          break;
        }
        __builtin_amdgcn_s_sleep(1);
      }
    }
    if (stuck) {
      break;
    }
  }
}

int main(int argc, char **argv) {
  int tiles = 23, n_layers = 4;
  long stride_bytes = 64;
  double bail_ms = 200.0;
  unsigned int pdelay = 10000; // ticks; 100 MHz clock, so 100 us
  int wb = 0, cinv = 0, alias = 0, vis = 0;
  char const *tag = "diag";
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "--tiles=", 8)) {
      tiles = atoi(argv[i] + 8);
    } else if (!strncmp(argv[i], "--layers=", 9)) {
      n_layers = atoi(argv[i] + 9);
    } else if (!strncmp(argv[i], "--flagstride=", 13)) {
      stride_bytes = atol(argv[i] + 13);
    } else if (!strncmp(argv[i], "--bailms=", 9)) {
      bail_ms = atof(argv[i] + 9);
    } else if (!strncmp(argv[i], "--pdelay=", 9)) {
      pdelay = (unsigned)atoi(argv[i] + 9);
    } else if (!strncmp(argv[i], "--wb=", 5)) {
      wb = atoi(argv[i] + 5);
    } else if (!strncmp(argv[i], "--cinv=", 7)) {
      cinv = atoi(argv[i] + 7);
    } else if (!strncmp(argv[i], "--alias=", 8)) {
      alias = atoi(argv[i] + 8);
    } else if (!strncmp(argv[i], "--vis=", 6)) {
      vis = atoi(argv[i] + 6);
    } else if (!strncmp(argv[i], "--tag=", 6)) {
      tag = argv[i] + 6;
    }
  }
  int const stride_ints = (int)(stride_bytes / 4);
  int const nblocks = NXCD * tiles;
  size_t const dbg_n = (size_t)nblocks * NWAVES * DBGF;
  unsigned long long const bail_ticks =
      (unsigned long long)(bail_ms * 1e-3 * 100e6);

  int *d_flags = nullptr, *d_dbg = nullptr, *d_seen = nullptr,
      *d_prod = nullptr;
  size_t const flag_bytes = (size_t)NXCD * stride_bytes + 4096;
  HIP_OK(hipMalloc(&d_flags, flag_bytes));
  HIP_OK(hipMalloc(&d_dbg, dbg_n * sizeof(int)));
  HIP_OK(hipMalloc(&d_seen, NXCD * sizeof(int)));
  HIP_OK(hipMalloc(&d_prod, NXCD * 4 * sizeof(int)));
  HIP_OK(hipMemset(d_flags, 0, flag_bytes));
  HIP_OK(hipMemset(d_dbg, 0, dbg_n * sizeof(int)));
  HIP_OK(hipMemset(d_seen, 0, NXCD * sizeof(int)));
  HIP_OK(hipMemset(d_prod, 0, NXCD * 4 * sizeof(int)));

  printf("[%s] flags %p  stride %ld B  %d tiles/XCD  %d waves  bail %.0f ms  "
         "pdelay %u ticks  %d layers  wb=%d cinv=%d\n",
         tag, (void *)d_flags, stride_bytes, tiles, nblocks * NWAVES, bail_ms,
         pdelay, n_layers, wb, cinv);

  if (vis) {
    int *d_vis = nullptr;
    size_t const vn = NXCD * NXCD + NXCD;
    HIP_OK(hipMalloc(&d_vis, vn * sizeof(int)));
    HIP_OK(hipMemset(d_vis, 0, vn * sizeof(int)));
    hipLaunchKernelGGL(vismatrix, dim3(nblocks), dim3(NTHREADS), 0, 0, d_flags,
                       d_vis, stride_ints, 200000u /* 2 ms per phase */);
    HIP_OK(hipDeviceSynchronize());
    HIP_OK(hipGetLastError());
    std::vector<int> v(vn);
    HIP_OK(hipMemcpy(v.data(), d_vis, vn * sizeof(int), hipMemcpyDeviceToHost));

    printf("[%s] every XCD cached all eight lines, then each stored its own "
           "flag, then all re-read. Entry = value the reader saw; xN means the "
           "writer's 0x1xx was observed, '.' means it was not.\n",
           tag);
    printf("             ");
    for (int w = 0; w < NXCD; ++w) {
      printf(" wrote%d", w);
    }
    printf("\n");
    int off_diag_seen = 0, off_diag_total = 0;
    for (int r = 0; r < NXCD; ++r) {
      printf("        xcd%d ", r);
      for (int w = 0; w < NXCD; ++w) {
        int const got = v[r * NXCD + w];
        int const want = 0x100 + w;
        if (r != w) {
          ++off_diag_total;
          if (got == want) {
            ++off_diag_seen;
          }
        }
        printf("%6s", (got == want) ? "seen" : ".");
      }
      printf("\n");
    }
    printf("[%s] remote stores observed: %d of %d\n", tag, off_diag_seen,
           off_diag_total);
    if (off_diag_seen == 0) {
      printf("[%s] VERDICT: no XCD observed ANY other XCD's store, including "
             "XCDs on the same side of the AID boundary. Coherence is not "
             "working between any two XCDs of this compute partition, not "
             "merely across AIDs.\n",
             tag);
    } else if (off_diag_seen == off_diag_total) {
      printf("[%s] VERDICT: every XCD saw every other XCD's store; coherence "
             "is working here.\n",
             tag);
    } else {
      printf("[%s] VERDICT: partial -- compare the pattern against the XCD to "
             "AID map; if the blanks are exactly the far-AID pairs the failure "
             "is cross-AID.\n",
             tag);
    }
    return 0;
  }

  if (alias) {
    hipLaunchKernelGGL(aliastest, dim3(nblocks), dim3(NTHREADS), 0, 0, d_flags,
                       d_prod, tiles, 200000u /* 2 ms */, wb);
    HIP_OK(hipDeviceSynchronize());
    HIP_OK(hipGetLastError());
    std::vector<int> a(NXCD * 4);
    HIP_OK(hipMemcpy(a.data(), d_prod, NXCD * 4 * sizeof(int),
                     hipMemcpyDeviceToHost));
    int hv = 0;
    HIP_OK(hipMemcpy(&hv, d_flags, sizeof(int), hipMemcpyDeviceToHost));
    printf("[%s] all eight XCDs wrote a distinct value to the SAME dword, then "
           "read it back:\n",
           tag);
    printf("        xcd   wrote    read   atomic  after_inv\n");
    int self = 0, same = 0;
    for (int x = 0; x < NXCD; ++x) {
      printf("        %3d  0x%04x  0x%04x  0x%04x     0x%04x%s\n", x,
             a[x * 4 + 0], a[x * 4 + 1], a[x * 4 + 2], a[x * 4 + 3],
             (a[x * 4 + 1] == a[x * 4 + 0]) ? "   <-- read back its own write"
                                            : "");
      if (a[x * 4 + 1] == a[x * 4 + 0]) {
        ++self;
      }
      if (a[x * 4 + 1] == a[0 * 4 + 1]) {
        ++same;
      }
    }
    printf("[%s] host reads 0x%04x at that address\n", tag, hv);
    if (self == NXCD) {
      printf("[%s] VERDICT: every XCD read back its OWN value from one shared "
             "address. One virtual address is resolving to eight different "
             "physical locations -- this is address aliasing, not a cache "
             "coherence failure.\n",
             tag);
    } else if (same == NXCD) {
      printf("[%s] VERDICT: all eight XCDs agree on one value, so the address "
             "is genuinely shared and writes are ordered.\n",
             tag);
    } else {
      printf("[%s] VERDICT: mixed -- %d read their own write, %d agree with "
             "xcd0.\n",
             tag, self, same);
    }
    return 0;
  }

  hipLaunchKernelGGL(hangdiag, dim3(nblocks), dim3(NTHREADS), 0, 0, d_flags,
                     d_dbg, d_seen, d_prod, tiles, stride_ints, n_layers,
                     bail_ticks, pdelay, wb, cinv);
  HIP_OK(hipDeviceSynchronize());
  HIP_OK(hipGetLastError());

  std::vector<int> dbg(dbg_n), prod(NXCD * 4), seen(NXCD);
  HIP_OK(hipMemcpy(dbg.data(), d_dbg, dbg_n * sizeof(int), hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(prod.data(), d_prod, NXCD * 4 * sizeof(int),
                   hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(seen.data(), d_seen, NXCD * sizeof(int),
                   hipMemcpyDeviceToHost));

  // Ground truth: what the host sees in the flag array after the kernel.
  std::vector<int> hflags(NXCD);
  for (int x = 0; x < NXCD; ++x) {
    HIP_OK(hipMemcpy(&hflags[x], d_flags + x * stride_ints, sizeof(int),
                     hipMemcpyDeviceToHost));
  }

  printf("[%s] blocks per XCD:", tag);
  for (int x = 0; x < NXCD; ++x) {
    printf(" %d", seen[x]);
  }
  printf("\n");

  printf("[%s] producer self-check (what each writer saw of its own store):\n",
         tag);
  printf("        xcd   wrote   selfread   selfatomic\n");
  for (int x = 0; x < NXCD; ++x) {
    printf("        %3d %7d %10d %12d%s\n", x, prod[x * 4 + 0],
           prod[x * 4 + 1], prod[x * 4 + 2],
           (prod[x * 4 + 1] != prod[x * 4 + 0]) ? "   <-- own store not visible"
                                                : "");
  }

  printf("[%s] host view of flags after kernel:", tag);
  for (int x = 0; x < NXCD; ++x) {
    printf(" %d", hflags[x]);
  }
  printf("\n");

  int nstuck = 0;
  int matrix[NXCD][NXCD];
  memset(matrix, 0, sizeof(matrix));
  int ex_plain = -1, ex_nosc = -1, ex_atom = -1, ex_inv = -1, ex_layer = -1;
  int n_atom_fresh = 0, n_inv_fresh = 0, n_all_stale = 0, n_host_had_it = 0;
  int n_own_xcd = 0;
  for (size_t i = 0; i < dbg_n; i += DBGF) {
    if (!dbg[i]) {
      continue;
    }
    ++nstuck;
    int const px = dbg[i + 1], tgt = dbg[i + 2], layer = dbg[i + 3];
    int const plain = dbg[i + 4], nosc = dbg[i + 5], atom = dbg[i + 6],
              inv = dbg[i + 7];
    if (px >= 0 && px < NXCD && tgt >= 0 && tgt < NXCD) {
      ++matrix[px][tgt];
    }
    if (atom >= layer) {
      ++n_atom_fresh;
    }
    if (inv >= layer) {
      ++n_inv_fresh;
    }
    if (plain < layer && nosc < layer && atom < layer && inv < layer) {
      ++n_all_stale;
      // The host reads the same address after the kernel. If it sees the value
      // the wave was waiting for, the store did land; the wave simply could
      // not observe it from where it was running.
      if (tgt >= 0 && tgt < NXCD && hflags[tgt] >= layer) {
        ++n_host_had_it;
      }
      if (px == tgt) {
        ++n_own_xcd;
      }
    }
    if (ex_plain < 0) {
      ex_plain = plain;
      ex_nosc = nosc;
      ex_atom = atom;
      ex_inv = inv;
      ex_layer = layer;
    }
  }

  printf("[%s] stuck waves: %d of %d\n", tag, nstuck, nblocks * NWAVES);
  if (nstuck == 0) {
    printf("[%s] VERDICT: no hang, every wave observed its flags\n", tag);
    return 0;
  }

  printf("[%s] example stuck wave: waiting for layer %d, saw"
         " plain=%d nosc=%d atomic=%d after_inv=%d\n",
         tag, ex_layer, ex_plain, ex_nosc, ex_atom, ex_inv);
  printf("[%s] of %d stuck waves: atomic saw fresh value in %d, "
         "after_inv fresh in %d, all four stale in %d\n",
         tag, nstuck, n_atom_fresh, n_inv_fresh, n_all_stale);

  printf("[%s] stuck-wave matrix, rows = polling XCD, cols = flag owner:\n",
         tag);
  printf("            ");
  for (int t = 0; t < NXCD; ++t) {
    printf(" f%-3d", t);
  }
  printf("\n");
  for (int p = 0; p < NXCD; ++p) {
    printf("        xcd%d", p);
    for (int t = 0; t < NXCD; ++t) {
      printf(" %4d", matrix[p][t]);
    }
    printf("\n");
  }

  printf("[%s] of the %d waves that saw a stale value by all four methods, "
         "the host could read the awaited value for %d of them; %d were "
         "waiting on a flag owned by their own XCD\n",
         tag, n_all_stale, n_host_had_it, n_own_xcd);

  if (n_all_stale == nstuck && n_host_had_it == nstuck && n_own_xcd == 0) {
    printf("[%s] VERDICT: every stuck wave was waiting on a flag owned by a "
           "DIFFERENT XCD, the store did land (the host reads it), and no "
           "access method on the consumer side could see it. Cross-XCD "
           "visibility is broken outright, not merely slow.\n",
           tag);
  } else if (n_all_stale == nstuck) {
    printf("[%s] VERDICT: the stores never reached the coherence point "
           "(producer-side failure)\n",
           tag);
  } else if (n_atom_fresh == nstuck) {
    printf("[%s] VERDICT: the value IS at the coherence point but the polling "
           "load is served a stale cached copy -- the invalidate probe never "
           "reached this XCD\n",
           tag);
  } else {
    printf("[%s] VERDICT: mixed, see the matrix above\n", tag);
  }
  return 0;
}
