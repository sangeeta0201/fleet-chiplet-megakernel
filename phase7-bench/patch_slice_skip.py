#!/usr/bin/env python3
"""Second negative control: break the attention-slice barrier instead.

The hier-poll control shows the check catches a broken hierarchical barrier,
which guards the RMSNorm row read. It says nothing about the other barrier the
AID-split flags fix touches -- the per-wave wait on the eight attn_release
flags, which guards the O-proj reduction's own input.

-DMPK_OPROJ_SKIP_SLICE_POLL drops that wait, so a wave quantizes its two 512-
element slices whether or not the producing XCDs have published them. The
expected signature differs from the hier control: this one corrupts
attn_proj_out (each block's own reduction), and rmsnorm_out follows because it
reads that row.

Run on mi355x-thor-2:  python3 patch_slice_skip.py
"""
import shutil
import sys

HDR = ("/home/schowdha/fleet-chiplet-megakernel/include/mirage/"
       "persistent_kernel/tasks/mi300/"
       "gang_linear_mxfp4_res_bias_rmsnorm_topk_mi300.cuh")
SUFFIX = ".pre-sliceskip"

OLD = """      while (ld_sys_s32(rel0) < layer_epoch || ld_sys_s32(rel1) < layer_epoch) {
#ifndef MPK_SLICE_BUSY_POLL
        __builtin_amdgcn_s_sleep(1);
#endif
      }
"""
NEW = """#ifdef MPK_OPROJ_SKIP_SLICE_POLL
      // Negative control; never define this in a real build. The acquire below
      // still runs, so what changes is only whether the slice was published
      // before this wave read it.
      while (false) {
#else
      while (ld_sys_s32(rel0) < layer_epoch || ld_sys_s32(rel1) < layer_epoch) {
#endif
#ifndef MPK_SLICE_BUSY_POLL
        __builtin_amdgcn_s_sleep(1);
#endif
      }
"""

with open(HDR, "r", encoding="utf-8", newline="") as fh:
    text = fh.read()

n = text.count(OLD)
if n != 1:
    sys.exit("slice poll anchor matched %d times, expected 1" % n)

shutil.copyfile(HDR, HDR + SUFFIX)
with open(HDR, "w", encoding="utf-8", newline="") as fh:
    fh.write(text.replace(OLD, NEW))

print("patched %s (backup at %s%s)" % (HDR, HDR, SUFFIX))
