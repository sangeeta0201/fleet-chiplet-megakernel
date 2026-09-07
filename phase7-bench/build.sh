#!/bin/bash
# Compile drive_phase7 with the same flags the model build used.
#
# The flags are extracted from a model build log rather than hardcoded, because
# the Phase 7 kernel's behavior depends on which MPK_* features are enabled --
# MPK_ATTN_SLICE_RELEASE decides whether the per-wave slice wait exists at all,
# MPK_NARROW_OPROJ_HIER and MPK_OPROJ_TREE_BARRIER pick the barrier shape, and
# MPK_SYS_POLL_LOAD=2 decides whether the release polls are system-scope reads.
# A mismatch here silently measures a different code path.
#
# usage: build.sh [path/to/model_build.log]
#   Default log is a previous run under /tmp/aidrun. Any log containing the
#   hipcc argv list printed by the mirage build works.
set -euo pipefail

REPO=${REPO:-/root/schowdha/fleet-chiplet-megakernel}
LOG=${1:-/tmp/aidrun/oproj_nps1w.log}

cd "$REPO"

if [ ! -f drive_phase7.cu ]; then
  echo "drive_phase7.cu not found in $REPO -- copy it there first" >&2
  exit 1
fi

python3 - "$LOG" <<'PYEOS'
import ast, sys

log = sys.argv[1]
cmd = None
for line in open(log, errors="replace"):
    line = line.strip()
    if line.startswith("[") and "hipcc" in line and "offload-arch" in line:
        try:
            cmd = ast.literal_eval(line)
            break
        except Exception:
            pass
if not cmd:
    sys.exit("no hipcc argv list found in %s; point build.sh at a model build log" % log)

# Keep only include paths, feature defines, link flags and the target arch.
# Everything else (-shared, --save-temps, the generated test.cu, the .so output)
# belongs to the python-extension build, not a standalone executable.
keep = [f for f in cmd
        if f.startswith(("-I", "-D", "-L", "-l", "--offload-arch", "-std", "-O"))
        and not f.startswith("-DMPK_PROFILING_NUM_ITERS")]

out = (["/opt/rocm/bin/hipcc", "-x", "hip", "drive_phase7.cu"]
       + keep
       + ["-DMPK_PROFILING_NUM_ITERS=0", "-DMPK_OPROJ_INNER_TIMING",
          "-o", "drive_phase7"])

with open("/tmp/build_drive.sh", "w") as fh:
    fh.write(" ".join(out) + "\n")

defines = [f for f in keep if f.startswith("-D")]
print("carried %d -D flags from %s" % (len(defines), log))
for want in ("-DMPK_ATTN_SLICE_RELEASE", "-DMPK_NARROW_OPROJ_HIER",
             "-DMPK_OPROJ_TREE_BARRIER", "-DMPK_SYS_POLL_LOAD=2"):
    print("  %-32s %s" % (want, "present" if want in defines else "MISSING"))
PYEOS

bash /tmp/build_drive.sh
echo "built: $REPO/drive_phase7"
