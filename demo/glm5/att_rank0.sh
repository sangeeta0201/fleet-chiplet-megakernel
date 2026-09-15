#!/bin/bash
# Per-rank launch wrapper: ATT-trace rank 0 only.
# Used via MPK_RANK_WRAPPER. Same co-residency rule as rocprof_rank0.sh:
# do NOT trace scheduler_kernel, and do not pass --att-serialize-all.
#
# Do not write rocprofv3 output under /tmp (decoder/path bugs). Default
# outdir is under the container home.
#
# Do not wrap this in sudo: MPI_Init then fails (PMIX ext3x Unreachable).
# Include-match worker_kernel and cap consecutive kernels at 1. Tracing
# every dispatch (exclude-only) hangs the persistent megakernel.
set -u

RANK="${OMPI_COMM_WORLD_RANK:-${PMIX_RANK:-0}}"
if [ "$RANK" != "0" ]; then
  exec "$@"
fi

# rocprofv3 LD_PRELOADs the tool into argv0. If argv0 is stdbuf, python is
# a child exec and the tool initializes twice. Drop stdbuf.
while [ $# -gt 0 ]; do
  case "$1" in
    stdbuf|-oL|-eL|-o|-e) shift ;;
    *) break ;;
  esac
done

OUTDIR="${MPK_ATT_OUTDIR:-/home/claudeuser/glm5_att}"
# MUST include-match the worker. An exclude-only filter traces every torch
# and hipcc dispatch (~tens of thousands of .att files) and then stalls the
# persistent worker when the trace buffer fills -- 0% HBM after ENTER.
KFILTER="${MPK_ATT_KERNEL_REGEX:-worker_kernel}"
# rocprofv3 int_auto() only accepts a base-10/hex integer of bytes.
BUF="${MPK_ATT_BUFFER_SIZE:-536870912}"
CU="${MPK_ATT_TARGET_CU:-0}"
mkdir -p "$OUTDIR"

exec rocprofv3 \
  --att \
  --kernel-include-regex "$KFILTER" \
  --kernel-iteration-range "${MPK_ATT_ITER_RANGE:-1-1}" \
  --att-consecutive-kernels "${MPK_ATT_CONSECUTIVE:-1}" \
  --att-target-cu "$CU" \
  --att-buffer-size "$BUF" \
  --att-activity "${MPK_ATT_ACTIVITY:-8}" \
  --att-library-path /opt/rocm/lib \
  -d "$OUTDIR" -o att -f csv \
  -- "$@"
