#!/bin/bash
# Collect hardware counters for the tile class on the NP=8 GLM-5 decode run.
#
# THE QUESTION (guide ruling, 2026-08-23). The tile class is ONE item:
# 69.6 us/layer against a 22.5 us byte roof, 3.09x. What KIND of bound is that?
#   * latency/occupancy-bound -> achieved HBM rate near roof, MFMA density low
#   * bandwidth-bound below peak -> achieved rate far below roof, and then the
#     dequant/LDS mechanism eating it has to be named.
# The wall cannot resolve this (0.26 ms floor), so this uses counters.
#
# WHAT IT DOES NOT MEASURE. Not the wall. rocprofv3 perturbs dispatch timing,
# so the ms/iter this run prints is NOT a latency number and must not be
# quoted as one. The two quantities taken from it are ratios and absolute byte
# counts, both robust to profiling overhead:
#   MfmaUtil  = SQ_VALU_MFMA_BUSY_CYCLES / (GRBM_GUI_ACTIVE * SIMD_NUM)
#   FetchSize/WriteSize = TCC<->EA bytes, i.e. actual HBM traffic
#
# Counters land in $OUTDIR as pmc_counter_collection.csv. Reduce with
# tile_class_bound.py.
#
#   ./profile_tile_class.sh [outdir]
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

OUTDIR="${1:-/tmp/glm5_pmc}"
rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"

export MODEL_PATH="${MODEL_PATH:-/home/claudeuser/models/glm5-mxfp4}"
ulimit -c 0

# The build is unchanged (this board edits only shell/python), so reuse it --
# a rebuild here would be the only variable that moved.
export KEEP_BUILD=1
export MASTER_PORT=$(( 30600 + (RANDOM % 300) ))
export MPK_PMC="${MPK_PMC:-MfmaUtil FetchSize WriteSize}"
export MPK_PMC_OUTDIR="$OUTDIR"
export MPK_PMC_KERNEL_REGEX="${MPK_PMC_KERNEL_REGEX:-worker_kernel}"
export MPK_RANK_WRAPPER="$PWD/rocprof_rank0.sh"

log="$OUTDIR/run.log"
echo "pmc=[$MPK_PMC] kernel_regex=[$MPK_PMC_KERNEL_REGEX] -> $log"
timeout "${RUN_TIMEOUT:-1800}" ./run_mp8_dp_ep_fused.sh > "$log" 2>&1
echo "rc=$?"
ls -la "$OUTDIR" || true
