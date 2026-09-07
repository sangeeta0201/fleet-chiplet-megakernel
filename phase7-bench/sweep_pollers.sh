#!/bin/bash
# Sweep the number of workgroups polling the eight attn_release flags.
#
# This is the experiment that distinguishes the two candidate explanations for
# the slicewait regression:
#   - a fixed cost per cross-AID access  -> slicewait stays flat as pollers drop
#   - queueing on the flag cache lines   -> slicewait falls with poller count
#
# Each block runs 4 waves and each wave polls two flags, so polling waves =
# 8 XCDs * tiles * 4. Run this in NPS2 (where the effect exists) and optionally
# in NPS1 as a control.
set -uo pipefail

REPO=${REPO:-/root/schowdha/fleet-chiplet-megakernel}
OUT=${OUT:-/tmp/aidrun}
LAYERS=${LAYERS:-200}
TILE_LIST=${TILE_LIST:-"23 12 4 1"}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-6}

cd "$REPO"
mkdir -p "$OUT"

mode=$(cat /sys/bus/pci/devices/0000:e5:00.0/current_memory_partition)
echo "mode: $mode   layers: $LAYERS"
echo
printf "%-10s %-8s %-14s %-12s %-12s %s\n" \
       tiles blocks pollingwaves slicewait_p50 bar_p50 per_wave_us

for t in $TILE_LIST; do
  log="$OUT/sweep_${mode}_t${t}.log"
  timeout -s KILL 300 ./drive_phase7 --layers="$LAYERS" --tiles="$t" \
      --tag="t$t" > "$log" 2>&1
  rc=$?
  if [ "$rc" != 0 ]; then
    printf "%-10s FAILED rc=%s\n" "$t" "$rc"
    tail -3 "$log"
    continue
  fi
  # Median of the field, taken over the whole log; warmup is a small fraction
  # of a 200-layer run and does not move the median.
  med=$(grep -oP 'slicewait=\K[\d.]+' "$log" | sort -n \
        | awk '{a[NR]=$1} END{printf "%.2f", a[int(NR*0.5)]}')
  barm=$(grep -oP 'bar=\K[\d.]+' "$log" | sort -n \
        | awk '{a[NR]=$1} END{printf "%.2f", a[int(NR*0.5)]}')
  waves=$((8 * t * 4))
  per=$(awk -v m="$med" -v w="$waves" 'BEGIN{printf "%.4f", m/w}')
  printf "%-10s %-8s %-14s %-12s %-12s %s\n" \
         "$t" "$((8 * t))" "$waves" "$med" "$barm" "$per"
done
