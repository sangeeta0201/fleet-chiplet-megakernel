#!/bin/bash
# NPS1 reference for the Phase 7 reproducer, then back to SPX+NPS2 to re-measure
# the fix in the same session.
#
# Both modes load the SAME module (the mtype-test tree), so the only difference
# between the NPS1 and NPS2 rows is the memory partition mode and the MTYPE the
# driver derives from it -- not a different driver build.
set -u
cd ~/fork-fcm/phase7-bench || exit 1
export PATH=$PATH:/opt/rocm/bin
DEV=6
L=${LAYERS:-1200}
T=${TILES:-23}
MT=/home/schowdha/aid-local-hbm/src/amdgpu-mtype-test

for s in spxnps1 spxnps2; do
  sed 's|^SRC=.*|SRC='"$MT"'|' /home/schowdha/_apfix/$s.sh > /tmp/${s}_mtype.sh
done

hipcc -O3 --offload-arch=gfx950 -std=c++17 -I/usr/include/libdrm \
      bench_phase7.hip -o bench_phase7 2>&1 | grep -E 'error' | head
[ -x ./bench_phase7 ] || { echo "ABORT: build failed"; exit 1; }

state() {
  D=/sys/bus/pci/devices/0000:05:00.0
  P=/sys/module/amdgpu/parameters
  echo "  mode=$(cat $D/current_compute_partition)+$(cat $D/current_memory_partition)" \
       " mtype_local=$(cat $P/mtype_local 2>/dev/null)" \
       " xcp_nc=$(cat $P/aid_local_xcp_nc 2>/dev/null)" \
       " flag_mtype=$(cat $P/aid_local_flag_mtype 2>/dev/null)"
  echo "  driver says: $(sudo dmesg | grep -o 'Using MTYPE_[A-Z]* for local memory' | tail -1)"
}

run() { # tag aid coherent split
  echo
  echo "--- $1 (aid=$2 coherent=$3 split=$4) ---"
  timeout 300 env HIP_VISIBLE_DEVICES=$DEV ./bench_phase7 \
    --layers=$L --tiles=$T --aid=$2 --coherent=$3 --split=$4 --tag=$1 2>&1 \
    | grep -E 'round-robin|COHERENT|us/layer|slicewait|read us|ABORT|unavailable' \
    || echo "  TIMED OUT (hang)"
}

echo "############ SPX+NPS1 reference ############"
sudo dmesg -C
bash /tmp/spxnps1_mtype.sh "aid_local_steal_gb=0" 2>&1 | tail -2
state
[ "$(cat /sys/bus/pci/devices/0000:05:00.0/current_memory_partition)" = NPS1 ] || {
  echo "ABORT: did not reach NPS1"; exit 1; }
run nps1 0 0 0
# informational: is AID-local placement even expressible with one partition?
run nps1_aid 1 1 1

echo
echo "############ back to SPX+NPS2 with the per-BO knob ############"
sudo dmesg -C
bash /tmp/spxnps2_mtype.sh "aid_local_steal_gb=0 aid_local_spx_nps2=1 \
aid_local_xcp_span=1 aid_local_xcp_nc=1 aid_local_flag_mtype=2" 2>&1 | tail -2
state
[ "$(cat /sys/bus/pci/devices/0000:05:00.0/current_memory_partition)" = NPS2 ] || {
  echo "ABORT: did not get back to NPS2 -- box left in the wrong mode"; exit 1; }
run nps2_base  0 0 0
run nps2_aidcc 1 1 1
