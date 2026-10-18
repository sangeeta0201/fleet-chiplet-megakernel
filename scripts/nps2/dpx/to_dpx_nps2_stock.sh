#!/bin/bash
# to_dpx_nps2_stock.sh: DPX+NPS2 on the STOCK amdgpu module (no aid_local parameters).
# The memory partition persists in hardware. If it is NPS2 already, unload whatever module is loaded and
# modprobe the stock one; stock NPS2 offers no SPX, so DPX is written on every GPU. If it is NPS1 (e.g. after
# a Slurm handoff), NPS2 is requested through the stock sysfs file and the stock module is reloaded.
# Exits 1 unless e5 ends at DPX/NPS2 on the stock module.
set -u
IMG=${IMG:?set IMG to a container image that has modprobe (kmod)}
E=/sys/bus/pci/devices/0000:e5:00.0
aid() { ls /sys/module/amdgpu/parameters/ 2>/dev/null | grep -c aid_local; }
state() { echo "$(cat $E/current_compute_partition 2>/dev/null)/$(cat $E/current_memory_partition 2>/dev/null)"; }
priv() {
  docker run --rm --privileged --pid=host --net=host --entrypoint /bin/bash \
    -v /lib/modules:/lib/modules:ro -v /sys:/sys -v /dev:/dev "$IMG" -lc "$1"
}
reload_stock() {
  priv '
  for m in amdgpu amdxcp amddrm_ttm_helper amddrm_exec amddrm_buddy amd_sched amdttm amdkcl; do
    lsmod | grep -q "^${m} " && rmmod "$m" 2>/dev/null
  done
  lsmod | grep -q "^amdgpu " && { echo "ABORT: amdgpu still held"; exit 1; }
  modprobe amdgpu && echo "stock amdgpu loaded"'
  sleep 20
}
echo "before: $(state), aid params $(aid), srcversion $(cat /sys/module/amdgpu/srcversion)"
if [ "$(aid)" = 0 ] && [ "$(state)" = DPX/NPS2 ]; then
  echo "already stock DPX+NPS2"
  docker start fleet_v1 > /dev/null 2>&1
  exit 0
fi
for c in $(docker ps --format '{{.Names}}' | grep '^fleet_'); do docker stop "$c" > /dev/null 2>&1; done
if [ "$(aid)" != 0 ]; then
  reload_stock || exit 1
fi
echo "stock module: $(state), aid params $(aid)"
if [ "$(cat $E/current_memory_partition)" != NPS2 ]; then
  echo "requesting NPS2 (stock sysfs) and reloading"
  priv 'for b in 05 15 65 75 85 95 e5 f5; do echo NPS2 > /sys/bus/pci/devices/0000:$b:00.0/current_memory_partition 2>&1 | head -1; done; echo requested'
  reload_stock || exit 1
  echo "after NPS2 request: $(state)"
fi
priv '
for b in 05 15 65 75 85 95 e5 f5; do
  f=/sys/bus/pci/devices/0000:$b:00.0/current_compute_partition
  [ "$(cat $f)" = DPX ] || echo DPX > $f 2>/dev/null
  echo "$b: $(cat $f)/$(cat /sys/bus/pci/devices/0000:$b:00.0/current_memory_partition)"
done' | tr '\n' ' '
echo
docker start fleet_v1 > /dev/null 2>&1
n=$(docker exec fleet_v1 python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
echo "now: $(state), aid params $(aid) (0 = stock), srcversion $(cat /sys/module/amdgpu/srcversion), HIP devices $n"
[ "$(state)" = DPX/NPS2 ] && [ "$(aid)" = 0 ]

