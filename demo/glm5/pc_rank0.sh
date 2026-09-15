#!/bin/bash
# Per-rank launch wrapper: PC-sample rank 0 only. Used via MPK_RANK_WRAPPER.
#
# Same two constraints as att_rank0.sh:
#   * do NOT serialize dispatches -- worker_kernel and scheduler_kernel are
#     launched on separate streams and REQUIRE co-residency, so anything that
#     serializes them deadlocks the run by construction. PC sampling is an
#     interrupt-driven sampler and does not serialize; --kernel-trace only
#     records begin/end timestamps. Neither adds a per-dispatch barrier.
#   * rocprofv3 LD_PRELOADs the tool into argv0, so strip the `stdbuf` prefix
#     the launcher adds, or the tool initializes twice.
#
# Do not write rocprofv3 output under /tmp (decoder/path bugs) and not under /
# on this box (98% full). Default outdir is on /mnt/nvme1.
#
# The megakernel is ONE persistent dispatch, so per-KERNEL attribution is
# useless here. Attribution is per-ADDRESS instead: map the sampled PC back
# through the disassembly (pc_phase_map.py). That is why no kernel filter is
# applied -- a filter would drop the samples, not the dispatches.
set -u

RANK="${OMPI_COMM_WORLD_RANK:-${PMIX_RANK:-0}}"
if [ "$RANK" != "0" ]; then
  exec "$@"
fi

while [ $# -gt 0 ]; do
  case "$1" in
    stdbuf|-oL|-eL|-o|-e) shift ;;
    *) break ;;
  esac
done

OUTDIR="${MPK_PC_OUTDIR:-/mnt/nvme1/stallprobe/pc}"
mkdir -p "$OUTDIR"

exec rocprofv3 \
  --kernel-trace \
  --pc-sampling-beta-enabled true \
  --pc-sampling-unit "${MPK_PC_UNIT:-cycles}" \
  --pc-sampling-method "${MPK_PC_METHOD:-stochastic}" \
  --pc-sampling-interval "${MPK_PC_INTERVAL:-1048576}" \
  -d "$OUTDIR" -o pc -f csv \
  -- "$@"
