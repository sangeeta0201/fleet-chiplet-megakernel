#!/bin/bash
# Apply the per-BO coherent MTYPE fix to the Phase 7 reproducer.
#
# Unlike bench_aidsplit, this runs the real Phase 7 shape: the producer writes
# its 512-bf16 slice, publishes attn_release[x*16], and every wave then spins on
# slices 2w and 2w+1 before reading 2 KiB, with a per-layer rendezvous across
# all eight XCDs. So it measures the fix against the actual dependency
# structure, including the barrier and the cross-AID attn_out read.
#
# Three cells:
#   base   hipMalloc flags + counter, MTYPE_NC          -- today's behaviour
#   aidnc  AID-split flags + counter, still NC          -- isolates placement
#   aidcc  AID-split flags + counter, coherent MTYPE    -- the fix
set -u
cd ~/fork-fcm/phase7-bench || exit 1
export PATH=$PATH:/opt/rocm/bin
DEV=6
L=${LAYERS:-1200}
T=${TILES:-23}

echo "===== driver state ====="
D=/sys/bus/pci/devices/0000:05:00.0
P=/sys/module/amdgpu/parameters
echo "  mode=$(cat $D/current_compute_partition)+$(cat $D/current_memory_partition)"
for p in aid_local_xcp_nc aid_local_flag_mtype mtype_local; do
  v=$(cat $P/$p 2>/dev/null) && echo "    $p=$v" || echo "    $p ABSENT"
done
[ "$(cat $D/current_memory_partition)" = NPS2 ] || { echo "ABORT: need NPS2"; exit 1; }
[ -e "$P/aid_local_flag_mtype" ] || {
  echo "ABORT: loaded module has no aid_local_flag_mtype -- load the"
  echo "       amdgpu-mtype-test tree first (see run_perbo.sh)"; exit 1; }
[ "$(cat $P/aid_local_flag_mtype)" = 2 ] || {
  echo "ABORT: aid_local_flag_mtype must be 2 for the aidcc cell"; exit 1; }

echo
echo "===== build ====="
hipcc -O3 --offload-arch=gfx950 -std=c++17 -I/usr/include/libdrm \
      bench_phase7.hip -o bench_phase7 2>&1 | grep -E 'error|warning: unused' | head
[ -x ./bench_phase7 ] || { echo "ABORT: build failed"; exit 1; }
echo "  ok"

run() { # tag aid coherent split
  echo
  echo "--- $1 (aid=$2 coherent=$3 split=$4) ---"
  timeout 300 env HIP_VISIBLE_DEVICES=$DEV ./bench_phase7 \
    --layers=$L --tiles=$T --aid=$2 --coherent=$3 --split=$4 --tag=$1 2>&1 \
    | grep -E 'round-robin|COHERENT|us/layer|slicewait|read us|ABORT' \
    || echo "  TIMED OUT (hang)"
}

run base  0 0 0
run aidnc 1 0 1
run aidcc 1 1 1

echo
echo "===== driver path taken ====="
sudo dmesg | grep -i FLAGMTYPE | tail -3
