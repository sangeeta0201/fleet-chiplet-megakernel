#!/bin/bash
# Run the Phase 7 benchmark in NPS1 and NPS2 and compare.
#
# The settle delay before each mode switch is not optional. set_mode.sh reloads
# the patched amdgpu driver, and if a GPU context from the previous run has not
# been released yet, rmmod fails, the script prints "ABORT: amdgpu held", and it
# leaves the node in the *previous* mode. The benchmark then runs happily and
# every ratio comes out at 1.0x, which looks like a real null result.
set -uo pipefail

REPO=${REPO:-/root/schowdha/fleet-chiplet-megakernel}
SET_MODE=${SET_MODE:-/tmp/aidrun/set_mode.sh}
OUT=${OUT:-/tmp/aidrun}
LAYERS=${LAYERS:-300}
SETTLE=${SETTLE:-8}
# Any single GPU works; the kernel spans all 8 XCDs of that one device.
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-6}

# Reading the mode from sysfs rather than trusting set_mode.sh's exit status is
# the whole point of the validation below.
MODE_FILE=/sys/bus/pci/devices/0000:e5:00.0/current_memory_partition

cd "$REPO"
mkdir -p "$OUT"

run_arm() {
  local want=$1
  echo "===== $want ====="
  sleep "$SETTLE"
  bash "$SET_MODE" "$want" 2>&1 | tail -5
  local have
  have=$(cat "$MODE_FILE")
  if [ "$have" != "$want" ]; then
    echo "ABORT: asked for $want but the node is in $have."
    echo "       The switch failed; comparing these logs would be meaningless."
    return 1
  fi
  echo "confirmed mode: $have"
  timeout -s KILL 300 ./drive_phase7 --layers="$LAYERS" --tag="$want" \
      > "$OUT/bench_${want}.log" 2>&1
  local rc=$?
  echo "  rc=$rc samples=$(grep -c OPROJ_INNER "$OUT/bench_${want}.log")"
  grep -E "blocks per XCD|polling waves" "$OUT/bench_${want}.log"
  [ "$rc" = 0 ] || { echo "ABORT: benchmark failed"; tail -5 "$OUT/bench_${want}.log"; return 1; }
}

run_arm NPS1 || exit 1
echo
run_arm NPS2 || exit 1
echo

python3 "$(dirname "$0")/compare.py" "$OUT/bench_NPS1.log" "$OUT/bench_NPS2.log"
