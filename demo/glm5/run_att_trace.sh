#!/bin/bash
# ATT thread-trace of rank-0 worker_kernel on a short KEEP_BUILD decode.
#
# NOT a latency number. rocprofv3 perturbs the dispatch, and the persistent
# kernel is one dispatch, so the captured window is whatever fits in the
# ATT buffer (bootstrap + first fused-layer trips unless the buffer wraps).
#
#   ./run_att_trace.sh [outdir]
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

OUTDIR="${1:-/home/claudeuser/glm5_att}"
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

export MODEL_PATH="${MODEL_PATH:-/mnt/nvme1/GLM-5.2-MXFP4}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-4,5,6,7}"
# Match the shipping 1k/1k image so KEEP_BUILD does not rebuild.
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
export KEEP_BUILD=1
export MASTER_PORT=$(( 30600 + (RANDOM % 300) ))
export MPK_ATT_OUTDIR="$OUTDIR"
export MPK_ATT_BUFFER_SIZE="${MPK_ATT_BUFFER_SIZE:-536870912}"
export MPK_RANK_WRAPPER="$PWD/att_rank0.sh"

ulimit -c 0

# shellcheck source=stall_watchdog.sh
. ./stall_watchdog.sh
STALL_SECS="${STALL_SECS:-240}"
log="$OUTDIR/run.log"
echo "ATT out=$OUTDIR buf=$MPK_ATT_BUFFER_SIZE kernel=worker_kernel -> $log"

run_with_watchdog "$log" "$STALL_SECS" \
  ./run_mp8_dp_ep_fused.sh \
    --ignore-eos \
    --prompt "${MPK_ATT_PROMPT:-The capital of France is}" \
    --max-new-tokens "$MAX_NEW_TOKENS"

echo "rc=$?"
find "$OUTDIR" -type f | head -80
