#!/usr/bin/env python3
"""Add a compile-time escape hatch that breaks the hierarchical barrier.

A correctness check that has never been seen to fail is not evidence. This adds
-DMPK_OPROJ_SKIP_HIER_POLL, which drops the wait on the per-XCD release flag so
RMSNorm reads the output row while other XCDs are still storing into it -- the
exact corruption the barrier exists to prevent.

The point is to confirm the check can tell that build apart from a correct one.
If it cannot, the check is measuring nothing and any "verified" claim resting on
it is empty.

Run on mi355x-thor-2:  python3 patch_poll_skip.py
"""
import shutil
import sys

HDR = ("/home/schowdha/fleet-chiplet-megakernel/include/mirage/"
       "persistent_kernel/tasks/mi300/"
       "gang_linear_mxfp4_res_bias_rmsnorm_topk_mi300.cuh")
SUFFIX = ".pre-pollskip"

OLD = """      while (MPK_LD_GATE2(&hier_rel[xcd_id * HIER_STRIDE]) <
             oproj_release_expected) {
        __builtin_amdgcn_s_sleep(1);
      }
"""
NEW = """#ifdef MPK_OPROJ_SKIP_HIER_POLL
      // Negative control; never define this in a real build. The release flag
      // is still written, just never waited on, so the only thing that changes
      // is whether the row is complete when RMSNorm reads it.
      while (false) {
#else
      while (MPK_LD_GATE2(&hier_rel[xcd_id * HIER_STRIDE]) <
             oproj_release_expected) {
#endif
        __builtin_amdgcn_s_sleep(1);
      }
"""

with open(HDR, "r", encoding="utf-8", newline="") as fh:
    text = fh.read()

n = text.count(OLD)
if n != 1:
    sys.exit("poll anchor matched %d times, expected 1" % n)

shutil.copyfile(HDR, HDR + SUFFIX)
with open(HDR, "w", encoding="utf-8", newline="") as fh:
    fh.write(text.replace(OLD, NEW))

print("patched %s (backup at %s%s)" % (HDR, HDR, SUFFIX))
