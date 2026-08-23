#!/bin/bash
# Sample per-GPU HBM controller busy % while the NP=8 decode benchmark runs.
#
# WHY THIS EXISTS. tile_class_bound.py computes the achieved HBM rate from
# MODELLED bytes (117.73 MB/layer/rank), which is a LOWER bound on real
# traffic -- if the kernel actually re-read 3x that, the tile class would be AT
# the HBM roof and the "neither unit is saturated" verdict would be wrong.
# rocprofv3 --pmc would settle it directly, but it hung the megakernel at
# launch_persistent_kernel (see TILE_CLASS_BOUND.md); this does not, because it
# is a host-side sysfs read and touches nothing on the device.
#
#   /sys/class/drm/card*/device/mem_busy_percent   HBM controller busy %
#   /sys/class/drm/card*/device/gpu_busy_percent   shader busy %
#
# WHAT IT IS NOT. A utilisation percentage, not a byte count. It cannot
# separate "33% busy at full width" from "100% busy on a third of the
# channels", and the sampling is asynchronous to the phase structure. It is a
# corroboration of the modelled rate at CLASS granularity -- which is exactly
# the granularity the question was asked at -- not a replacement for it.
#
# DECODE LENGTH. The shipping run already decodes to the 128-token seq
# limit -- 118 iterations, ~1.23 s -- so there is a real plateau to sample at
# 10 ms and no argument override is needed. This runs the SAME configuration
# bench_repeat.sh does.
#
#   ./sample_hbm_activity.sh <tag> [interval_s]
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

TAG="${1:?usage: sample_hbm_activity.sh <tag> [interval_s]}"
IVAL="${2:-0.01}"
LOG="/tmp/glm5_${TAG}_bench.log"
SAMP="/tmp/glm5_${TAG}_hbm.csv"
CARDS=$(ls -d /sys/class/drm/card*/device/mem_busy_percent 2>/dev/null \
        | sed 's#/mem_busy_percent##' | sort -t/ -k5 -V)

export MODEL_PATH="${MODEL_PATH:-/home/claudeuser/models/glm5-mxfp4}"
export MASTER_PORT=$(( 30900 + (RANDOM % 300) ))
ulimit -c 0

: > "$SAMP"
(
  while :; do
    ts=$(date +%s.%N); i=0
    for d in $CARDS; do
      echo "$ts,$i,$(cat "$d/mem_busy_percent" 2>/dev/null),$(cat "$d/gpu_busy_percent" 2>/dev/null)" >> "$SAMP"
      i=$((i + 1))
    done
    sleep "$IVAL"
  done
) &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null' EXIT

timeout "${RUN_TIMEOUT:-1800}" ./run_mp8_dp_ep_fused.sh > "$LOG" 2>&1
echo "MPIRUN rc=$?"
kill "$SAMPLER" 2>/dev/null; sleep 1

echo "--- wall ---"
grep -E '^\[1,0\].*(Decode|Prefill):' "$LOG" | tail -2
echo "--- generated text (rank 0) ---"
grep -E '^\[1,0\]' "$LOG" | grep -iE "output|generated|capital of" | tail -4
echo "--- HBM busy %, samples where the shader is busy (gpu_busy > 50) ---"
awk -F, '$4>50 {mb[$2]+=$3; gb[$2]+=$4; n[$2]++; if($3>mx[$2])mx[$2]=$3}
         END{for(g=0;g<8;g++) if(n[g])
               printf "  GPU%d  mem_busy mean %5.1f %%  max %3d %%   gpu_busy mean %5.1f %%   n=%d\n",
                      g, mb[g]/n[g], mx[g], gb[g]/n[g], n[g]}' "$SAMP"
echo "  raw: $SAMP  ($(wc -l < "$SAMP") samples)"
