#!/bin/bash
# PC-sample rank 0 of the GLM-5.2 megakernel at the 1k-context decode shape.
#
# NOT a latency number. rocprofv3 perturbs the run; quote the stall SPLIT, not
# the wall. Reuses the existing build (KEEP_BUILD=1) so the sampled PCs map
# onto the disassembly of the image already in permanent_output_dir_rank0 --
# a rebuild would invalidate the address->phase map.
#
#   OSL=16  ./run_pc_sample.sh canary     # schema + non-empty check
#   OSL=128 ./run_pc_sample.sh main
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

TAG="${1:-main}"
ISL="${ISL:-1024}"
OSL="${OSL:-128}"
OUT="${OUT:-/mnt/nvme1/stallprobe/pc_$TAG}"

export MAX_SEQ_LENGTH=2048          # must match the built image
export MAX_NEW_TOKENS="$OSL"
export MAX_SAVE_TOKENS="$OSL"
export MODEL_PATH="${MODEL_PATH:-/mnt/nvme1/GLM-5.2-MXFP4}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-4,5,6,7}"
export KEEP_BUILD=1
export MASTER_PORT=$(( 31100 + (RANDOM % 400) ))

rm -rf "$OUT"; mkdir -p "$OUT"

# The wrapper runs on the rank, which does not inherit arbitrary env; bake the
# settings in with `env` (MPK_RANK_WRAPPER is unquoted and word-splits).
MPK_RANK_WRAPPER="env MPK_PC_OUTDIR=$OUT \
MPK_PC_UNIT=${MPK_PC_UNIT:-cycles} \
MPK_PC_METHOD=${MPK_PC_METHOD:-stochastic} \
MPK_PC_INTERVAL=${MPK_PC_INTERVAL:-1048576} \
$PWD/pc_rank0.sh"
export MPK_RANK_WRAPPER

ulimit -c 0

# shellcheck source=stall_watchdog.sh
. ./stall_watchdog.sh
STALL_SECS="${STALL_SECS:-900}"
log="$OUT/run.log"
echo "PC sample tag=$TAG ISL=$ISL OSL=$OSL out=$OUT"

run_with_watchdog "$log" "$STALL_SECS" \
  ./run_mp8_dp_ep_fused.sh \
    --ignore-eos \
    --prompt-tokens "$ISL" \
    --save-tokens "$OUT/tokens.json"
rc=$?
echo "rc=$rc"
find "$OUT" -type f -printf '%10s  %p\n' | sort -k2 | head -40
