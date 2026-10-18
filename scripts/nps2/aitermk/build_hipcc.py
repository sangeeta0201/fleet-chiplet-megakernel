"""Build a fleet MPK provider DSO by replaying the exported build's own compile command on the exported source.

usage (inside the build container): build_hipcc.py PROVIDER_DIR NAME
The command is the standalone build's (compile_cmd.json) with the source swapped for src/kernel.hip, the Python
include dropped, src/ on the include path (task_graph_blob.inc) and the output renamed. --save-temps stays, so the
device assembly lands next to the DSO for comparison with the standalone build's.
"""
import json
import os
import subprocess
import sys

P, name = sys.argv[1], sys.argv[2]
cmd = json.load(open(os.path.join(P, "src", "compile_cmd.json")))
out = os.path.join(P, "build_hipcc")
os.makedirs(out, exist_ok=True)
so = os.path.join(out, "libaitermk_provider_gpt_oss_gfx950_fleet_mpk_%s.so" % name)
new, skip = [], False
for a in cmd:
    if skip:
        skip = False
        continue
    if a.endswith("test.cu"):
        new.append(os.path.join(P, "src", "kernel.hip"))
    elif a.startswith("-I/usr/include/python"):
        continue
    elif a == "-o":
        new += ["-o", so]
        skip = True
    else:
        new.append(a)
new.insert(1, "-I" + os.path.join(P, "src"))
with open(os.path.join(out, "build_cmd.txt"), "w") as f:
    f.write(" ".join(new) + "\n")
subprocess.check_call(new, cwd=out)
print("BUILT", so, os.path.getsize(so))

