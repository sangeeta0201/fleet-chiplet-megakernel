#!/usr/bin/env python3
"""AID-split the Phase 7 hierarchical release flags.

The eight per-XCD release flags and the single global arrival counter both live
in `counters`, addressed off one base, so the harness cannot separate them. The
flags are read 23 times per XCD per layer and want one replica per memory
partition; the counter is an aggregation point and must stay shared. This adds
two optional replica pointers to the kernel, uses them for the release stores
and the poll only, and leaves the arrival counter where it is.

Null replicas reproduce the current behaviour exactly, so the other callers of
this kernel are unaffected.

Run on mi355x-thor-2:  python3 patch_hier_split.py
"""
import shutil
import sys

FLEET = "/home/schowdha/fleet-chiplet-megakernel"
HDR = (FLEET + "/include/mirage/persistent_kernel/tasks/mi300/"
       "gang_linear_mxfp4_res_bias_rmsnorm_topk_mi300.cuh")
DRV = FLEET + "/drive_phase7.cu"

SUFFIX = ".pre-hiersplit"

# ?? header ??????????????????????????????????????????????????????????????????

H1_OLD = """        int const *attn_slice_release = nullptr) {
"""
H1_NEW = """        int const *attn_slice_release = nullptr,
        // Optional: one replica per memory partition of the eight per-XCD
        // hierarchical release flags, same 16-int stride as the copy inside
        // `counters`. In SPX+NPS2 a compute partition spans two memory
        // partitions, and a coherent MTYPE is only sound for readers
        // co-located with the line's home partition -- so the caller can place
        // one replica in each and let each half of the XCDs poll its own. Both
        // replicas are written the same value; only the reads are partitioned.
        // Null means the single shared copy, which is the behaviour every
        // other caller gets.
        int *hier_release_lo = nullptr,
        int *hier_release_hi = nullptr) {
"""

H2_OLD = """    int const oproj_release_expected =
        layer_epoch > 0 ? layer_epoch
                        : ld_nt_s32(&hier_barrier[xcd_id * HIER_STRIDE]) + 1;
"""
H2_NEW = """    // Which copy of the release flags this XCD polls. The global arrival
    // counter at [8 * HIER_STRIDE] is deliberately not routed: it is the one
    // place all eight dies aggregate, so it cannot be replicated.
    int const n_xcds =
        tiles_per_xcd > 0 ? total_oproj_tiles / tiles_per_xcd : 8;
    int *const hier_rel =
        (hier_release_lo && hier_release_hi)
            ? (xcd_id < n_xcds / 2 ? hier_release_lo : hier_release_hi)
            : hier_barrier;

    int const oproj_release_expected =
        layer_epoch > 0 ? layer_epoch
                        : ld_nt_s32(&hier_rel[xcd_id * HIER_STRIDE]) + 1;
"""

H3_OLD = """    if (oproj_rel_epoch != 0) {
      if (tid < 8) {
        st_wt_u32((void *)&hier_barrier[tid * HIER_STRIDE],
                  (unsigned)oproj_rel_epoch);
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }
"""
H3_NEW = """    if (oproj_rel_epoch != 0) {
      if (hier_release_lo && hier_release_hi) {
        // Sixteen flags: eight per replica. Lanes 0..15 of this wave differ
        // only in the address they form, so both replicas are published in
        // the same single store instruction the one-copy form uses.
        if (tid < 16) {
          int *const dst = (tid < 8) ? hier_release_lo : hier_release_hi;
          st_wt_u32((void *)&dst[(tid & 7) * HIER_STRIDE],
                    (unsigned)oproj_rel_epoch);
        }
      } else if (tid < 8) {
        st_wt_u32((void *)&hier_barrier[tid * HIER_STRIDE],
                  (unsigned)oproj_rel_epoch);
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }
"""

H4_OLD = """      while (MPK_LD_GATE2(&hier_barrier[xcd_id * HIER_STRIDE]) <
             oproj_release_expected) {
"""
H4_NEW = """      while (MPK_LD_GATE2(&hier_rel[xcd_id * HIER_STRIDE]) <
             oproj_release_expected) {
"""

# ?? harness ?????????????????????????????????????????????????????????????????

D1_OLD = """#define BAR_OFF_INTS (1 << 18)
"""
D1_NEW = """#define BAR_OFF_INTS (1 << 18)

// The hierarchical-barrier release replicas share the same AID-local buffer, at
// a 1.5 MiB offset so they land on neither a flag line nor the rendezvous
// counter. Eight lines of 16 ints, the same layout the kernel uses inside
// `counters`.
#define HIER_OFF_INTS (3 << 17)
"""

D2_OLD = """    int *flags, int *flags_b, unsigned int *bar, unsigned int *bar_b,
    int *xcd_seen, int n_layers, int delay_ticks, int delay_skew, int tiles,
    int dual, int split_on) {
"""
D2_NEW = """    int *flags, int *flags_b, unsigned int *bar, unsigned int *bar_b,
    int *hier_lo, int *hier_hi,
    int *xcd_seen, int n_layers, int delay_ticks, int delay_skew, int tiles,
    int dual, int split_on) {
"""

D3_OLD = """        /*ts_base=*/nullptr,
        /*attn_slice_release=*/rel);
"""
D3_NEW = """        /*ts_base=*/nullptr,
        /*attn_slice_release=*/rel,
        /*hier_release_lo=*/hier_lo,
        /*hier_release_hi=*/hier_hi);
"""

D4_OLD = """  int *flags = nullptr, *flags_b = nullptr, *xcd_seen = nullptr;
"""
D4_NEW = """  int *flags = nullptr, *flags_b = nullptr, *xcd_seen = nullptr;
  int *hier_lo = nullptr, *hier_hi = nullptr;
"""

D5_OLD = """    bar = (unsigned int *)(flags + BAR_OFF_INTS);
    bar_b = (unsigned int *)(flags_b + BAR_OFF_INTS);
"""
D5_NEW = """    bar = (unsigned int *)(flags + BAR_OFF_INTS);
    bar_b = (unsigned int *)(flags_b + BAR_OFF_INTS);
    // Only with --split=1. Handing both halves the same replica would be the
    // misrouted case: coherent lines whose readers span both partitions, which
    // hangs rather than running slowly.
    if (split_on) {
      hier_lo = flags + HIER_OFF_INTS;
      hier_hi = flags_b + HIER_OFF_INTS;
    }
"""

D6_OLD = """                     aeid, flags, flags_b, bar, bar_b, xcd_seen, n_layers,
                     delay_ticks, delay_skew, tiles, dual, split_on);
"""
D6_NEW = """                     aeid, flags, flags_b, bar, bar_b, hier_lo, hier_hi,
                     xcd_seen, n_layers, delay_ticks, delay_skew, tiles, dual,
                     split_on);
"""

D7_OLD = """  printf("[%s] aid=%d coherent=%d split=%d\\n", tag, use_aid, (int)g_coherent,
         split_on);
"""
D7_NEW = """  printf("[%s] aid=%d coherent=%d split=%d hier_split=%d\\n", tag, use_aid,
         (int)g_coherent, split_on, hier_lo != nullptr);
"""

EDITS = {
    HDR: [("signature", H1_OLD, H1_NEW),
          ("routed base", H2_OLD, H2_NEW),
          ("release fan-out", H3_OLD, H3_NEW),
          ("poll", H4_OLD, H4_NEW)],
    DRV: [("offset", D1_OLD, D1_NEW),
          ("kernel params", D2_OLD, D2_NEW),
          ("kernel call", D3_OLD, D3_NEW),
          ("declarations", D4_OLD, D4_NEW),
          ("replica setup", D5_OLD, D5_NEW),
          ("launch", D6_OLD, D6_NEW),
          ("banner", D7_OLD, D7_NEW)],
}


def apply(path, edits):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        text = fh.read()

    for name, old, new in edits:
        n = text.count(old)
        if n != 1:
            sys.exit("%s: anchor %r matched %d times, expected 1"
                     % (path, name, n))
        text = text.replace(old, new)
        print("  ok  %s" % name)

    shutil.copyfile(path, path + SUFFIX)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


for path, edits in EDITS.items():
    print("== %s" % path)
    apply(path, edits)

print("\npatched; backups at *%s" % SUFFIX)
