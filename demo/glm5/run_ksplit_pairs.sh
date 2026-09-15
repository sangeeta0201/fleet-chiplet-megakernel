#!/bin/bash
# Alternating control/variant pairs for the o_proj K-split ceiling probe.
#
#   PAIRS=3 ARMB=ceil2 ./run_ksplit_pairs.sh
#
# ALTERNATING, not blocked: this box drifts (thermals, page cache, the
# occasional 24-26 ms straggler iteration), and a blocked A-A-A-B-B-B design
# aliases that drift onto the arm. Report the pair table, never the best.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

PAIRS="${PAIRS:-3}"
ARMB="${ARMB:-ceil2}"
RESULTS="${RESULTS:-/tmp/ksplit_ab}"
mkdir -p "$RESULTS"

for r in $(seq 1 "$PAIRS"); do
  for arm in control "$ARMB"; do
    RESULTS="$RESULTS" ARM="$arm" REP="$r" ./run_ksplit_ab.sh \
      > "$RESULTS/${arm}_r${r}.out" 2>&1
    echo "=== ${arm} r${r} rc=$? ==="
    grep -E "decode_avg_ms|G1_cross_rank|G2_distinct|TEXT_HEAD|FAILED" \
      "$RESULTS/${arm}_r${r}.out" 2>/dev/null | head -5
    # Never start the next arm while the driver still holds VRAM.
    for _ in $(seq 1 12); do
      used=$(rocm-smi --showmeminfo vram 2>/dev/null |
             awk '/GPU\[[4-7]\].*Used/ {s+=$NF} END {print int(s/1073741824)}')
      [ "${used:-99}" -lt 8 ] && break
      sleep 5
    done
  done
done

echo
echo "########## PAIR TABLE ##########"
python3 ./summarize_ksplit_ab.py "$RESULTS"
