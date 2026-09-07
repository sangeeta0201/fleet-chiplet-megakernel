#!/usr/bin/env python3
"""Add --dupdata to bench_phase7.hip.

Replicates attn_out into both AIDs so a consumer reads the slice copy homed in
its own AID. The producer writes both copies, exactly as it already does for the
release flags. dupdata=1 places the pair with MTYPE_NC (isolates locality),
dupdata=2 also marks them COHERENT (locality + a cacheable poll-free read).
"""
import re
import sys

PATH = "bench_phase7.hip"
src = open(PATH).read()
orig = src
subs = []


def sub(old, new, tag):
    global src
    if src.count(old) != 1:
        print("FAIL %s: found %d occurrences" % (tag, src.count(old)))
        sys.exit(1)
    src = src.replace(old, new)
    subs.append(tag)


# 1. per-call coherence, so the flags and the slice buffer can differ.
sub(
    """static void *alloc_in_aid(size_t bytes, int aid) {
  uint64_t const flags = GEM_CREATE_AID_LOCAL |
                         (g_range_for_aid[aid] & 1u ? GEM_CREATE_AID_SELECT : 0) |
                         (g_coherent ? GEM_CREATE_COHERENT : 0);""",
    """static void *alloc_in_aid(size_t bytes, int aid, bool coherent) {
  uint64_t const flags = GEM_CREATE_AID_LOCAL |
                         (g_range_for_aid[aid] & 1u ? GEM_CREATE_AID_SELECT : 0) |
                         (coherent ? GEM_CREATE_COHERENT : 0);""",
    "alloc_in_aid signature",
)

# 2. kernel takes the second slice buffer and the knob.
sub(
    """    unsigned short *attn_out, int *flags_a, int *flags_b, unsigned int *bar_a,
    unsigned int *bar_b, unsigned long long *samp_wait,
    unsigned long long *samp_read, unsigned int *sink, int n_layers,
    int delay_ticks, int delay_skew, int dual, int split_on, int tiles,
    int *seen) {""",
    """    unsigned short *attn_out, unsigned short *attn_b, int *flags_a,
    int *flags_b, unsigned int *bar_a, unsigned int *bar_b,
    unsigned long long *samp_wait, unsigned long long *samp_read,
    unsigned int *sink, int n_layers, int delay_ticks, int delay_skew, int dual,
    int split_on, int tiles, int dupdata, int *seen) {""",
    "kernel signature",
)

# 3. read from the local copy; keep a handle on both for the producer.
sub(
    """  unsigned int const *src =
      (unsigned int const *)(attn_out + first_xcd * ATTN_SLICE);
  unsigned int *my_slice = (unsigned int *)(attn_out + xcd * ATTN_SLICE);""",
    """  // With dupdata the slice buffer is replicated as well, so a consumer reads
  // the copy homed in its own AID and no slice read crosses the boundary. This
  // routes on the XCD's AID directly rather than on split_on, which only
  // controls the flags. attn_out is the AID0 copy, attn_b the AID1 copy.
  unsigned short *const attn_rd =
      (dupdata && xcd >= NXCD / 2) ? attn_b : attn_out;
  unsigned int const *src =
      (unsigned int const *)(attn_rd + first_xcd * ATTN_SLICE);
  unsigned int *my_slice = (unsigned int *)(attn_out + xcd * ATTN_SLICE);
  unsigned int *my_slice_b = (unsigned int *)(attn_b + xcd * ATTN_SLICE);""",
    "consumer slice routing",
)

# 4. producer publishes its slice into both copies.
sub(
    """      // 256 threads x 1 dword = 256 dwords = 512 bf16 = this XCD's slice.
      st_wt_u32(&my_slice[tid], (unsigned)(layer * 2654435761u + xcd));
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");""",
    """      // 256 threads x 1 dword = 256 dwords = 512 bf16 = this XCD's slice.
      // One extra KiB of write-through per XCD per layer buys every consumer a
      // local read of all eight slices.
      unsigned int const v = (unsigned)(layer * 2654435761u + xcd);
      st_wt_u32(&my_slice[tid], v);
      if (dupdata) {
        st_wt_u32(&my_slice_b[tid], v);
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");""",
    "producer slice publish",
)

# 5. knobs.
sub(
    """  int tiles = 23; // O-proj workgroups per XCD in production""",
    """  int tiles = 23; // O-proj workgroups per XCD in production
  int dupdata = 0;""",
    "dupdata decl",
)
sub(
    """    } else if (!strncmp(argv[i], "--split=", 8)) {
      split_on = atoi(argv[i] + 8);""",
    """    } else if (!strncmp(argv[i], "--split=", 8)) {
      split_on = atoi(argv[i] + 8);
    } else if (!strncmp(argv[i], "--dupdata=", 10)) {
      dupdata = atoi(argv[i] + 10);""",
    "dupdata parse",
)

# 6. second slice buffer.
sub(
    """  unsigned short *d_attn = nullptr;""",
    """  unsigned short *d_attn = nullptr, *d_attn_b = nullptr;""",
    "d_attn_b decl",
)

# 7. allocate the pair. Must come after aid_init(), so it goes in the use_aid
#    block; the plain hipMalloc above is simply left unused in that case.
sub(
    """    d_flags_a = (int *)alloc_in_aid(sync_bytes, 0);
    d_flags_b = (int *)alloc_in_aid(sync_bytes, 1);""",
    """    d_flags_a = (int *)alloc_in_aid(sync_bytes, 0, g_coherent);
    d_flags_b = (int *)alloc_in_aid(sync_bytes, 1, g_coherent);""",
    "flag alloc call",
)
sub(
    """    HIP_OK(hipMemset(d_flags_a, 0, sync_bytes));
    HIP_OK(hipMemset(d_flags_b, 0, sync_bytes));
    dual = 1;""",
    """    HIP_OK(hipMemset(d_flags_a, 0, sync_bytes));
    HIP_OK(hipMemset(d_flags_b, 0, sync_bytes));
    dual = 1;
    if (dupdata) {
      // The slice buffer needs no coherence to be local, and the consumer
      // issues a buffer_inv before reading anyway, so dupdata=1 (NC) is the
      // measurement of pure locality. dupdata=2 additionally marks the pair
      // COHERENT, which is sound for the same reason it is for the flags:
      // every reader is co-located with the copy it reads.
      bool const dcoh = dupdata >= 2;
      void *a = alloc_in_aid(sync_bytes, 0, dcoh);
      void *b = alloc_in_aid(sync_bytes, 1, dcoh);
      if (a == nullptr || b == nullptr) {
        fprintf(stderr, "ABORT: could not place a slice buffer in each AID\\n");
        return 2;
      }
      d_attn = (unsigned short *)a;
      d_attn_b = (unsigned short *)b;
      HIP_OK(hipMemset(d_attn, 0, REDUCTION * sizeof(unsigned short)));
      HIP_OK(hipMemset(d_attn_b, 0, REDUCTION * sizeof(unsigned short)));
    }""",
    "slice pair alloc",
)

# 8. dupdata needs the AID allocator.
sub(
    """  int *d_probe = nullptr;""",
    """  if (dupdata && !use_aid) {
    fprintf(stderr, "ABORT: --dupdata needs --aid=1\\n");
    return 2;
  }

  int *d_probe = nullptr;""",
    "dupdata guard",
)

# 9. reporting + launch.
sub(
    """  printf("      attn_out %p (NC, cross-AID by design)  flags %p / %p\\n",
         (void *)d_attn, (void *)d_flags_a, (void *)d_flags_b);""",
    """  printf("      attn_out %p / %p (dupdata=%d)  flags %p / %p\\n",
         (void *)d_attn, (void *)d_attn_b, dupdata, (void *)d_flags_a,
         (void *)d_flags_b);""",
    "report line",
)
sub(
    """  hipLaunchKernelGGL(phase7, dim3(nblocks), dim3(NTHREADS), 0, 0, d_attn,
                     d_flags_a, d_flags_b, d_bar_a, d_bar_b, d_wait, d_read,
                     d_sink, n_layers, delay_ticks, delay_skew, dual, split_on,
                     tiles, d_seen);""",
    """  hipLaunchKernelGGL(phase7, dim3(nblocks), dim3(NTHREADS), 0, 0, d_attn,
                     d_attn_b == nullptr ? d_attn : d_attn_b, d_flags_a,
                     d_flags_b, d_bar_a, d_bar_b, d_wait, d_read, d_sink,
                     n_layers, delay_ticks, delay_skew, dual, split_on, tiles,
                     dupdata, d_seen);""",
    "launch",
)
sub(
    """  printf("[%s] aid=%d coherent=%d split=%d  %d tiles/XCD = %d blocks, %d "
         "polling waves, %d layers, delay=%d skew=%d ticks\\n",
         tag, use_aid, (int)g_coherent, split_on, tiles, nblocks,""",
    """  printf("[%s] aid=%d coherent=%d split=%d dupdata=%d  %d tiles/XCD = %d "
         "blocks, %d polling waves, %d layers, delay=%d skew=%d ticks\\n",
         tag, use_aid, (int)g_coherent, split_on, dupdata, tiles, nblocks,""",
    "header line",
)

open(PATH, "w").write(src)
print("patched %d sites: %s" % (len(subs), ", ".join(subs)))
print("delta %+d bytes" % (len(src) - len(orig)))
