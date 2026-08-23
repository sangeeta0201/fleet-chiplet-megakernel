#!/bin/bash
# Per-rank launch wrapper: run the command under rocprofv3 on rank 0 only,
# plain on every other rank. Used via MPK_RANK_WRAPPER in
# run_mp8_dp_ep_fused.sh; see profile_tile_class.sh for the caller.
#
# WHY RANK 0 ONLY. rocprofv3 counter collection intercepts every dispatch on
# the agent and adds a per-dispatch counter start/stop. Doing that on all eight
# ranks multiplies the perturbation and, more to the point, the numbers we want
# (MFMA issue occupancy, HBM bytes) are per-GPU quantities -- one rank's worker
# kernel already contains the whole layer stack for that rank.
#
# WHY THE KERNEL FILTER. The megakernel launches worker_kernel and
# scheduler_kernel on two separate streams and REQUIRES them co-resident. If
# rocprofv3 serializes profiled dispatches, profiling both deadlocks the run by
# construction. Restricting counter collection to worker_kernel is the only
# shape that can work at all; if it still hangs, the run times out and that
# fact is the finding.
set -u

RANK="${OMPI_COMM_WORLD_RANK:-${PMIX_RANK:-0}}"
if [ "$RANK" != "0" ]; then
  exec "$@"
fi

PMC="${MPK_PMC:-MfmaUtil FetchSize WriteSize}"
OUTDIR="${MPK_PMC_OUTDIR:-/tmp/glm5_pmc}"
KFILTER="${MPK_PMC_KERNEL_REGEX:-worker_kernel}"
mkdir -p "$OUTDIR"

exec rocprofv3 \
  --pmc $PMC \
  --kernel-include-regex "$KFILTER" \
  -d "$OUTDIR" -o pmc -f csv \
  -- "$@"
