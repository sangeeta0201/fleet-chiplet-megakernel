#!/bin/bash
# Per-BO coherence without a global MTYPE change.
#
# aid_local_flag_mtype keys on the COHERENT GEM flag, which is in
# AMDGPU_GEM_CREATE_SETTABLE_MASK, so one GEM_CREATE can ask for both a home AID
# and a coherent MTYPE. With aid_local_xcp_nc=1 every ordinary page still gets
# MTYPE_NC, so this is the per-BO version of the global mtype_local=2 that
# reached 0.86 us but made all VRAM coherent.
#
# Four cells, and the conclusion needs all four:
#   cand   placed + coherent      -> want ~0.9 us
#   nc     placed, not coherent   -> the same placement under NC, ~7.7 us
#   broken coherent, co-location deliberately broken -> must degrade/stall
#   ord    ordinary hipMalloc     -> must still be ~20 us and must NOT hang
set -u
cd ~/fork-fcm/phase7-bench || exit 1
export PATH=$PATH:/opt/rocm/bin

MT=/home/schowdha/aid-local-hbm/src/amdgpu-mtype-test
DEV=6

echo "===== build ====="
hipcc -O3 --offload-arch=gfx950 bench_aidsplit.hip -o bench_aidsplit 2>&1 \
  | grep -E 'error' | head
[ -x ./bench_aidsplit ] || { echo "ABORT: build failed"; exit 1; }
echo "  ok"

echo
echo "===== the module we are about to load has the knob, and is complete ====="
modinfo -F parm $MT/amd/amdgpu/amdgpu.ko | grep flag_mtype || {
  echo "ABORT: $MT has no aid_local_flag_mtype"; exit 1; }
for m in amd/amdkcl/amdkcl.ko ttm/amdttm.ko scheduler/amd-sched.ko \
         amddrm_buddy.ko amddrm_exec.ko amddrm_ttm_helper.ko \
         amd/amdxcp/amdxcp.ko amd/amdgpu/amdgpu.ko; do
  [ -f "$MT/$m" ] || { echo "ABORT: $MT/$m missing, insmod would leave no driver"; exit 1; }
done
echo "  all modules present"

echo
echo "===== load: xcp_nc=1 (ordinary pages NC) + flag_mtype=2 (COHERENT BOs -> CC) ====="
# Deliberately NOT passing mtype_local: it is the global knob this run replaces.
sed 's|^SRC=.*|SRC='"$MT"'|' /home/schowdha/_apfix/spxnps2.sh \
  > /home/schowdha/_apfix/spxnps2_mtype.sh
sudo dmesg -C
bash /home/schowdha/_apfix/spxnps2_mtype.sh \
  "aid_local_steal_gb=0 aid_local_spx_nps2=1 aid_local_xcp_span=1 aid_local_xcp_nc=1 aid_local_flag_mtype=2" \
  2>&1 | tail -3
D=/sys/bus/pci/devices/0000:05:00.0
P=/sys/module/amdgpu/parameters
echo "  mode=$(cat $D/current_compute_partition)+$(cat $D/current_memory_partition)" \
     " xcp_nc=$(cat $P/aid_local_xcp_nc)" \
     " flag_mtype=$(cat $P/aid_local_flag_mtype)" \
     " mtype_local=$(cat $P/mtype_local)"
[ "$(cat $D/current_memory_partition)" = NPS2 ] || { echo "ABORT: not NPS2"; exit 1; }

run() { # tag split coherent
  echo
  echo "--- $1: split=$2 coherent=$3 ---"
  timeout 200 env HIP_VISIBLE_DEVICES=$DEV ./bench_aidsplit \
    --tiles=23 --layers=120 --split=$2 --coherent=$3 --tag=$1 2>&1 \
    | grep -E 'COHERENT|observed|vis_p50|no successful' \
    || echo "  TIMED OUT AT THE HARNESS LEVEL (hang)"
}

run cand   1 1
run nc     1 0
run broken 0 1

echo
echo "--- ord: ordinary hipMalloc flags must still be NC (~20 us, no hang) ---"
printf "  "
timeout 150 env HIP_VISIBLE_DEVICES=$DEV ./bench_flagplace \
  --layers=120 --tiles=23 --flagstride=64 --tag=ord 2>&1 \
  | grep -o 'poll_p50=[0-9.]* vis_p50=[0-9.]*' \
  || echo "HANG -- ordinary pages became coherent too, per-BO gating failed"

echo
echo "===== did the driver actually take the per-BO path? ====="
sudo dmesg | grep -iE 'FLAGMTYPE|MTYPE_' | tail -6
