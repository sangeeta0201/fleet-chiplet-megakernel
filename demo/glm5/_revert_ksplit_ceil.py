#!/usr/bin/env python3
"""Reverse the MPK_OPROJ_KSPLIT_CEIL patch in the tree given as argv[1].

Used only to prove the "default OFF is byte-identical" requirement: build the
pristine tree, build the patched tree with the flag unset, and compare the
device code objects. Re-apply with _apply_ksplit_ceil.py afterwards.
"""
import os
import re
import sys

ROOT = sys.argv[1]
FILES = [
    "include/mirage/persistent_kernel/tasks/mi300/gang_oproj_router_fused_mi300.cuh",
    "include/mirage/persistent_kernel/tasks/mi300/gang_mla_full_layer_fused_mi300.cuh",
]

# Strip "#if MPK_OPROJ_KSPLIT_CEIL >= N ... #else" leaving the #else body, and
# drop the matching "#endif // MPK_OPROJ_KSPLIT_CEIL...".
OPEN = re.compile(
    r"^#if MPK_OPROJ_KSPLIT_CEIL >= \d\n.*?^#else\n", re.S | re.M)
CLOSE = re.compile(r"^#endif // MPK_OPROJ_KSPLIT_CEIL.*?\n", re.M)

rc = 0
for rel in FILES:
    p = os.path.join(ROOT, rel)
    src = open(p).read()
    n_open = len(OPEN.findall(src))
    n_close = len(CLOSE.findall(src))
    if n_open != 1 or n_close != 1:
        print("FAIL %s: found %d #if / %d #endif, expected 1/1"
              % (rel, n_open, n_close))
        rc = 1
        continue
    src = OPEN.sub("", src, count=1)
    src = CLOSE.sub("", src, count=1)
    if "MPK_OPROJ_KSPLIT_CEIL" in src:
        print("FAIL %s: marker survived the revert" % rel)
        rc = 1
        continue
    open(p, "w").write(src)
    print("reverted: %s" % rel)

sys.exit(rc)
