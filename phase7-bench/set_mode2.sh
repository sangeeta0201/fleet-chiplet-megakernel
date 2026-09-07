#!/bin/bash
# set_mode2.sh NPS1|NPS2
#
# Like set_mode_flagmtype.sh, but asserts SPX explicitly instead of assuming
# the driver picks it at load. On this build the hive comes up DPX after a
# memory-mode change, and a live write to current_compute_partition is accepted.
set -u
WANT=${1:?NPS1 or NPS2}
SRC=$(ls -d /root/schowdha/aid-local-hbm/src/amdgpu-* 2>/dev/null | head -1)
BDFS="05 15 65 75 85 95 e5 f5"

[ -n "$SRC" ] || { echo "ABORT: no staged driver tree"; exit 1; }
echo "using SRC=$SRC"

unload() {
  for m in amdgpu amdxcp amddrm_ttm_helper amddrm_exec amddrm_buddy amd_sched amdttm amdkcl; do
    lsmod | grep -q "^${m} " && rmmod "$m" 2>/dev/null
  done
  lsmod | grep -q '^amdgpu ' && return 1 || return 0
}
load() {
  for m in drm_display_helper drm_suballoc_helper drm_exec video cec rc_core ib_core; do
    modprobe "$m" 2>/dev/null
  done
  for m in amd/amdkcl/amdkcl.ko ttm/amdttm.ko scheduler/amd-sched.ko amddrm_buddy.ko \
           amddrm_exec.ko amddrm_ttm_helper.ko amd/amdxcp/amdxcp.ko; do
    lsmod | grep -q "^$(basename $m .ko | tr - _) " || insmod "$SRC/$m" || return 1
  done
  insmod "$SRC/amd/amdgpu/amdgpu.ko" \
      aid_local_steal_gb=0 aid_local_spx_nps2=1 \
      aid_local_xcp_span=0 aid_local_xcp_nc=1 \
      aid_local_flag_mtype=2 || return 1
}
assert_spx() {
  local changed=0
  for b in $BDFS; do
    d=/sys/bus/pci/devices/0000:$b:00.0
    [ "$(cat $d/current_compute_partition)" = SPX ] && continue
    echo SPX > "$d/current_compute_partition" 2>/dev/null && changed=1
  done
  [ "$changed" = 1 ] && sleep 5
  return 0
}

echo "===== want SPX+$WANT ====="
unload || { echo "ABORT: amdgpu held"; fuser -v /dev/kfd 2>&1 | head; exit 1; }
load    || { echo "ABORT: insmod"; exit 1; }
sleep 8

need_reload=0
for b in $BDFS; do
  d=/sys/bus/pci/devices/0000:$b:00.0
  cur=$(cat $d/current_memory_partition)
  if [ "$cur" != "$WANT" ]; then
    echo "$WANT" > "$d/current_memory_partition" 2>/dev/null \
      && { echo "  0000:$b:00.0 $cur -> $WANT"; need_reload=1; } \
      || echo "  0000:$b:00.0 request FAILED (is $cur)"
  fi
done

if [ "$need_reload" = 1 ]; then
  echo "===== reload so PSP applies $WANT ====="
  unload || { echo "ABORT: unload for apply"; exit 1; }
  load    || { echo "ABORT: reload"; exit 1; }
  sleep 12
fi

assert_spx

echo "===== final ====="
ok=0
for b in $BDFS; do
  d=/sys/bus/pci/devices/0000:$b:00.0
  cp=$(cat $d/current_compute_partition); mp=$(cat $d/current_memory_partition)
  echo "  0000:$b:00.0  $cp / $mp"
  [ "$cp" = SPX ] && [ "$mp" = "$WANT" ] && ok=$((ok+1))
done
N=$(for n in /sys/class/kfd/kfd/topology/nodes/*/properties; do
      sc=$(grep -m1 '^simd_count' "$n" | awk '{print $2}'); [ "$sc" = "1024" ] && echo x; done | wc -l)
echo "  SPX+$WANT on $ok/8, nodes@1024=$N"
[ "$ok" = 8 ] && [ "$N" -ge 8 ] || { echo "ABORT: hive not uniform"; exit 1; }

for p in aid_local_xcp_nc aid_local_flag_mtype; do
  echo "  $p = $(cat /sys/module/amdgpu/parameters/$p)"
done
echo "READY $WANT"
