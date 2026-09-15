#!/bin/bash
# Run a sequence of compile-time-flag arms at the 1024/1024 latency protocol.
#
#   SEQ='control= mlptr=MPK_ML_PTR_PREFETCH=1' REPS=3 ./run_flag_pairs.sh
#   SEQ='control= pf256=MPK_QKVA_PF_KB=256 pf1024=MPK_QKVA_PF_KB=1024' \
#       REPS=1 ./run_flag_pairs.sh
#
# SEQ is whitespace-separated `name=ENV` items; ENV may itself be several
# comma-separated KEY=VALUE assignments, or empty for the control. The whole
# sequence is repeated REPS times, so arms alternate in time rather than being
# blocked A-A-A-B-B-B: this box drifts (thermals, page cache, the occasional
# 24-26 ms straggler iteration) and a blocked design aliases that drift onto
# the arm. Report the pair table, never the best number.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

SEQ="${SEQ:-control=}"
REPS="${REPS:-1}"
# First rep number, so a second driver invocation extends the pair table
# instead of overwriting rep 1.
REP_START="${REP_START:-1}"
RESULTS="${RESULTS:-/mnt/nvme1/glm5_flag_ab}"
mkdir -p "$RESULTS"

_wait_vram() {
  local used
  for _ in $(seq 1 24); do
    used=$(rocm-smi --showmeminfo vram 2>/dev/null |
           awk '/GPU\[[4-7]\].*Used/ {s+=$NF} END {print int(s/1073741824)}')
    [ "${used:-99}" -lt 8 ] && return 0
    sleep 5
  done
  echo "[pairs] WARNING: VRAM still ${used} GB after 120 s"
}

for r in $(seq "$REP_START" $((REP_START + REPS - 1))); do
  for item in $SEQ; do
    arm="${item%%=*}"
    env_str="${item#*=}"
    env_str="${env_str//,/ }"
    RESULTS="$RESULTS" ARM="$arm" REP="$r" ARM_ENV="$env_str" \
      ./run_flag_ab.sh > "$RESULTS/${arm}_r${r}.out" 2>&1
    echo "=== ${arm} r${r} rc=$? ==="
    grep -E "decode_min_ms|G1_cross_rank|G2_distinct|TEXT_TAIL|FAILED" \
      "$RESULTS/${arm}_r${r}.out" 2>/dev/null | head -5
    _wait_vram
  done
done

echo
echo "########## PAIR TABLE ##########"
python3 ./summarize_flag_ab.py "$RESULTS"
