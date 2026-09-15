#!/bin/bash
# Alternating-arm A/B for the fused-collective campaign, on devices 0-3.
#
#   SEQ='control= prohoist=MPK_QKV_PRO_HOIST=1' REPS=3 ./run_collfuse_pairs.sh
#
# Arms alternate in time rather than being blocked A-A-A-B-B-B, because this
# box drifts and a blocked design aliases the drift onto the arm. Report the
# pair table, never the best number.
#
# Every arm goes through run_collfuse_arm.sh, which is run_flag_ab.sh with the
# cleanup scoped and the VRAM check pointed at the devices actually in use --
# see that file's header for why both matter when another campaign is running
# on devices 4-7 in the same container.
#
# ABSOLUTES ARE NOT COMPARABLE to the 14.0 ms/token canonical baseline, which
# was taken on devices 4-7 with a socket-1 rankfile. The delta is the result.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

SEQ="${SEQ:-control=}"
REPS="${REPS:-1}"
RESULTS="${RESULTS:-/mnt/nvme1/glm5_collfuse_ab}"
mkdir -p "$RESULTS"

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3}"
export NP="${NP:-4}"
export STALL_SECS="${STALL_SECS:-900}"
export ISL="${ISL:-1024}"
export OSL="${OSL:-1024}"

ulimit -c 0

echo "[collfuse] devices=$HIP_VISIBLE_DEVICES seq='$SEQ' reps=$REPS"

for r in $(seq 1 "$REPS"); do
  for item in $SEQ; do
    arm="${item%%=*}"
    env_str="${item#*=}"
    env_str="${env_str//,/ }"
    RESULTS="$RESULTS" ARM="$arm" REP="$r" ARM_ENV="$env_str" \
      ./run_collfuse_arm.sh > "$RESULTS/${arm}_r${r}.out" 2>&1
    echo "=== ${arm} r${r} rc=$? $(date -u +%H:%M:%SZ) ==="
    grep -E "decode_min_ms|decode_avg_ms|G1_cross_rank|G2_distinct|TEXT_TAIL|FAILED|\[isa\]" \
      "$RESULTS/${arm}_r${r}.out" 2>/dev/null | head -6
  done
done

echo
echo "########## PAIR TABLE (devices $HIP_VISIBLE_DEVICES) ##########"
python3 ./summarize_flag_ab.py "$RESULTS"
