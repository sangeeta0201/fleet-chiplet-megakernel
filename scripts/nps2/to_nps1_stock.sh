#!/bin/bash
# to_nps1_stock.sh: SPX+NPS1 on the stock amdgpu, the NPS1 baseline. The patched module requests NPS1 and
# reloads so the request takes effect; then the stock module is loaded. The memory partition persists in
# hardware, so the stock module comes up in SPX+NPS1. About 4 minutes; exits 1 unless it got there.
set -u
D=$(cd "$(dirname "$0")" && pwd)
IMG=${IMG:?set IMG to a container image that has modprobe (kmod)}
E=/sys/bus/pci/devices/0000:e5:00.0
aid() { ls /sys/module/amdgpu/parameters/ | grep -c aid_local; }
if [ "$(aid)" = 0 ] && [ "$(cat $E/current_compute_partition)/$(cat $E/current_memory_partition)" = SPX/NPS1 ]; then
  echo "already SPX+NPS1 on the stock driver"
  docker start fleet_v1 > /dev/null 2>&1
  exit 0
fi
WANT=NPS1 IMG=$IMG bash "$D/spx_nps2_halves_span.sh" || echo "(continuing: the stock reload sets the final state)"
for c in $(docker ps --format '{{.Names}}' | grep '^fleet_'); do docker stop "$c" > /dev/null 2>&1; done
docker run --rm --privileged --pid=host --net=host --entrypoint /bin/bash \
  -v /lib/modules:/lib/modules:ro -v /sys:/sys -v /dev:/dev "$IMG" -lc '
  for m in amdgpu amdxcp amddrm_ttm_helper amddrm_exec amddrm_buddy amd_sched amdttm amdkcl; do
    lsmod | grep -q "^${m} " && rmmod "$m" 2>/dev/null
  done
  lsmod | grep -q "^amdgpu " && { echo "ABORT: amdgpu still held"; exit 1; }
  modprobe amdgpu && echo "stock amdgpu loaded"'
sleep 15
docker start fleet_v1 > /dev/null 2>&1
st="$(cat $E/current_compute_partition)/$(cat $E/current_memory_partition)"
echo "now: $st, $(aid) aid_local parameters (0 = stock driver)"
[ "$st" = SPX/NPS1 ] && [ "$(aid)" = 0 ]
