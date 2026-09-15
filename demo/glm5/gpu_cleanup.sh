#!/bin/bash
# Kill GLM-5.2 GPU runs and confirm VRAM is released.
#
# Lives in a FILE on purpose. Running these patterns inline via `bash -lc`
# makes pkill match the invoking shell's own command line and SIGKILL it
# (exit 137) before the cleanup finishes, leaving ranks alive and VRAM held.
#
# Driver shells die first, otherwise their retry loop respawns mpirun.
set -u

echo "=== killing driver shells ==="
pkill -9 -f 'run_ksplit_ab\.sh'    2>/dev/null
pkill -9 -f 'run_ksplit_pairs\.sh' 2>/dev/null
pkill -9 -f 'run_ksplit_smoke\.sh' 2>/dev/null
pkill -9 -f 'run_with_watchdog'    2>/dev/null
sleep 2

echo "=== killing ranks ==="
for _ in 1 2 3; do
  pkill -9 -f 'demo\.py'  2>/dev/null
  pkill -9 -f 'mpirun'    2>/dev/null
  sleep 5
  pgrep -f 'demo\.py' >/dev/null 2>&1 || break
done

sleep 10
echo "=== survivors ==="
pgrep -af 'demo\.py|mpirun' | cut -c1-60 | head

echo "=== vram per gpu (idle ~0.3 GB) ==="
rocm-smi --showmeminfo vram 2>/dev/null |
  awk '/Used/ {printf "  gpu%d  %.2f GB\n", n++, $NF/1073741824}'

echo "=== disk ==="
df -h / | tail -1
echo "=== gpucore files (must be none) ==="
find /home/claudeuser/fleet-chiplet-megakernel -maxdepth 3 -name 'gpucore*' \
  -printf '%p %s\n' 2>/dev/null | head
