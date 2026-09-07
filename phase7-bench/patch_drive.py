#!/usr/bin/env python3
"""Give drive_phase7.cu the AID-split + coherent release flags.

drive_phase7 calls the real gang_linear_mxfp4_res_bias_rmsnorm_topk_kernel, and
that kernel takes attn_slice_release as a parameter while the harness itself
performs the producer store. So the whole slicewait fix lands in the harness:
allocate one AID-local COHERENT flag replica per AID, have the producer publish
into both, and hand each XCD the pointer to the replica homed in its own AID.

The kernel's own layer barrier lives in `counters` and is untouched here -- that
one needs the arrival/release sites inside the fleet headers to publish into
both replicas, so it is a separate change.

The AID allocator is lifted verbatim out of bench_phase7.hip so the two
harnesses cannot drift apart.
"""
import sys

DRIVE = "drive_phase7.cu"
BENCH = "bench_phase7.hip"

src = open(DRIVE).read()
bench = open(BENCH).read()
subs = []


def sub(old, new, tag):
    global src
    if src.count(old) != 1:
        print("FAIL %s: found %d occurrences" % (tag, src.count(old)))
        sys.exit(1)
    src = src.replace(old, new)
    subs.append(tag)


# Pull the allocator out of the other harness.
a = bench.index("static constexpr uint64_t GEM_CREATE_AID_LOCAL")
b = bench.index("// The rendezvous counter shares")
allocator = bench[a:b].rstrip() + "\n"
if "alloc_in_aid(size_t bytes, int aid, bool coherent)" not in allocator:
    print("FAIL: bench_phase7.hip allocator not in the expected 3-arg form")
    sys.exit(1)

# 1. headers the allocator needs.
sub(
    """#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>""",
    """#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <drm/amdgpu_drm.h>
#include <drm/drm.h>""",
    "includes",
)

# 2. the allocator itself.
sub(
    """__device__ unsigned int g_local_claim[NXCD];""",
    allocator
    + """
// The rendezvous counter shares the AID-local buffer with the flags, at a 1 MiB
// offset so it cannot land on a flag line.
#define BAR_OFF_INTS (1 << 18)

__device__ unsigned int g_local_claim[NXCD];""",
    "allocator",
)

# 3. kernel takes both replicas.
sub(
    """    int *flags, unsigned int *bar, int *xcd_seen, int n_layers,
    int delay_ticks, int delay_skew, int tiles) {""",
    """    int *flags, int *flags_b, unsigned int *bar, unsigned int *bar_b,
    int *xcd_seen, int n_layers, int delay_ticks, int delay_skew, int tiles,
    int dual, int split_on) {""",
    "kernel signature",
)

# 4. pick this XCD's replica.
sub(
    """  int const tile_idx = xcd * tiles + local;""",
    """  int const tile_idx = xcd * tiles + local;

  // Which replica this XCD touches. The upper half of the XCDs uses the copy
  // homed in AID1 and the lower half the copy in AID0, so every poll is served
  // in the reader's own coherence domain -- the precondition that makes a
  // coherent MTYPE sound on these lines.
  bool const upper = split_on && xcd >= NXCD / 2;
  int *const rel = upper ? flags_b : flags;
  unsigned int *const my_bar = upper ? bar_b : bar;""",
    "replica select",
)

# 5. producer publishes into both replicas.
sub(
    """      if (tid == 0) {
        drv_st_wt_u32(&flags[xcd * FLAG_STRIDE], (unsigned)layer);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      }""",
    """      if (tid == 0) {
        // Same value into both replicas, so a consumer sees the identical fact
        // whichever one it polls. The far-AID store still invalidates the
        // sharers co-located with that line, which is what lets one producer
        // release both halves.
        drv_st_wt_u32(&flags[xcd * FLAG_STRIDE], (unsigned)layer);
        if (dual) {
          drv_st_wt_u32(&flags_b[xcd * FLAG_STRIDE], (unsigned)layer);
        }
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      }""",
    "producer publish",
)

# 6. the real kernel polls the local replica.
sub(
    """        /*attn_slice_release=*/flags);""",
    """        /*attn_slice_release=*/rel);""",
    "kernel arg",
)

# 7. harness rendezvous gets the same treatment, so it is not the confound.
sub(
    """    if (tid == 0) {
      atomicAdd(bar, 1u);
      while (drv_ld_sys_s32((int *)bar) < layer * NXCD * tiles) {""",
    """    if (tid == 0) {
      atomicAdd(bar, 1u);
      if (dual) {
        atomicAdd(bar_b, 1u);
      }
      while (drv_ld_sys_s32((int *)my_bar) < layer * NXCD * tiles) {""",
    "harness rendezvous",
)

# 8. knobs.
sub(
    """  int tiles = TILES_PER_XCD;
  char const *tag = "run";""",
    """  int tiles = TILES_PER_XCD;
  int use_aid = 0, split_on = 1;
  char const *tag = "run";""",
    "knob decls",
)
sub(
    """    } else if (!strncmp(argv[i], "--tag=", 6)) {
      tag = argv[i] + 6;
    }""",
    """    } else if (!strncmp(argv[i], "--aid=", 6)) {
      use_aid = atoi(argv[i] + 6);
    } else if (!strncmp(argv[i], "--coherent=", 11)) {
      g_coherent = atoi(argv[i] + 11) != 0;
    } else if (!strncmp(argv[i], "--split=", 8)) {
      split_on = atoi(argv[i] + 8);
    } else if (!strncmp(argv[i], "--tag=", 6)) {
      tag = argv[i] + 6;
    }""",
    "knob parse",
)

# 9. second replica.
sub(
    """  int *flags = nullptr, *xcd_seen = nullptr;
  unsigned int *bar = nullptr;""",
    """  int *flags = nullptr, *flags_b = nullptr, *xcd_seen = nullptr;
  unsigned int *bar = nullptr, *bar_b = nullptr;
  int dual = 0;""",
    "replica decls",
)
sub(
    """  HIP_OK(hipMemset(bar, 0, sizeof(unsigned int)));
  HIP_OK(hipMemset(xcd_seen, 0, NXCD * sizeof(int)));""",
    """  HIP_OK(hipMemset(bar, 0, sizeof(unsigned int)));
  HIP_OK(hipMemset(xcd_seen, 0, NXCD * sizeof(int)));

  if (use_aid) {
    if (!aid_init()) {
      fprintf(stderr, "ABORT: --aid=1 but AID-local allocation is unavailable "
                      "(stock driver, or not NPS2)\\n");
      return 2;
    }
    size_t const sync_bytes = 2ull << 20;
    void *fa = alloc_in_aid(sync_bytes, 0, g_coherent);
    void *fb = alloc_in_aid(sync_bytes, 1, g_coherent);
    if (fa == nullptr || fb == nullptr) {
      fprintf(stderr, "ABORT: could not place a sync buffer in each AID\\n");
      return 2;
    }
    flags = (int *)fa;
    flags_b = (int *)fb;
    bar = (unsigned int *)(flags + BAR_OFF_INTS);
    bar_b = (unsigned int *)(flags_b + BAR_OFF_INTS);
    HIP_OK(hipMemset(flags, 0, sync_bytes));
    HIP_OK(hipMemset(flags_b, 0, sync_bytes));
    dual = 1;
  } else {
    // Baseline: one hipMalloc'd flag page and one counter, both MTYPE_NC and
    // both read from either AID -- exactly what the model does today.
    flags_b = flags;
    bar_b = bar;
    split_on = 0;
  }""",
    "replica alloc",
)

# 10. reporting + launch.
sub(
    """  printf("[%s] attn_out %p  flags %p  weight %p (%.2f MiB)\\n", tag, attn,
         (void *)flags, weight, weight_bytes / 1048576.0);""",
    """  printf("[%s] attn_out %p  flags %p / %p  weight %p (%.2f MiB)\\n", tag, attn,
         (void *)flags, (void *)flags_b, weight, weight_bytes / 1048576.0);
  printf("[%s] aid=%d coherent=%d split=%d\\n", tag, use_aid, (int)g_coherent,
         split_on);""",
    "report",
)
sub(
    """                     aeid, flags, bar, xcd_seen, n_layers, delay_ticks,
                     delay_skew, tiles);""",
    """                     aeid, flags, flags_b, bar, bar_b, xcd_seen, n_layers,
                     delay_ticks, delay_skew, tiles, dual, split_on);""",
    "launch",
)

open(DRIVE, "w").write(src)
print("patched %d sites: %s" % (len(subs), ", ".join(subs)))
