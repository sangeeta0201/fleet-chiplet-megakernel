#!/usr/bin/env python3
"""Stage the weight tile once instead of every layer, under a compile switch.

Re-staging per layer costs about 1 ms a layer against roughly 10 us for the
kernel, so the gate was running on a build two orders of magnitude slower than
the one the latency numbers came from. The weights never change between layers,
so the copy only has to happen once -- provided the kernel's own scratch really
does stay below oproj_lds_w_off() and never reaches the weight region.

That proviso is the reason this is a switch rather than an edit. Build both and
compare hashes: if -DMPK_DRV_STAGE_ONCE agrees with per-layer staging, the
region survives the kernel and the fast build is sound to gate on. If it
disagrees, the region is being clobbered and per-layer staging is the only
correct form -- which is worth knowing either way.

Run on mi355x-thor-2:  python3 patch_lds_once.py
"""
import shutil
import sys

DRV = "/home/schowdha/fleet-chiplet-megakernel/drive_phase7.cu"
SUFFIX = ".pre-ldsonce"

OLD = """    {
      // 16 bytes per thread per pass, as one dwordx4. Every base is at least
"""
NEW = """#ifdef MPK_DRV_STAGE_ONCE
    // The weights are the same every layer, so one copy is enough -- as long as
    // nothing in the kernel writes into this LDS region between layers. Compare
    // the hash against the per-layer build to find out.
    if (layer == 1)
#endif
    {
      // 16 bytes per thread per pass, as one dwordx4. Every base is at least
"""

with open(DRV, "r", encoding="utf-8", newline="") as fh:
    text = fh.read()

n = text.count(OLD)
if n != 1:
    sys.exit("staging anchor matched %d times, expected 1" % n)

shutil.copyfile(DRV, DRV + SUFFIX)
with open(DRV, "w", encoding="utf-8", newline="") as fh:
    fh.write(text.replace(OLD, NEW))

print("patched %s (backup at %s%s)" % (DRV, DRV, SUFFIX))
