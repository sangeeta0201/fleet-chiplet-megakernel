#!/bin/bash
# The stock driver pairs NPS2 with DPX, which is why the patched module carries a
# hunk to pick SPX at load. Phase 7 needs all eight XCDs behind one device, so
# ask for SPX directly. Compute partition is normally live-switchable when no
# process holds the GPU; if firmware refuses it while NPS2, that is itself the
# finding and the patched module becomes necessary.
set -u
BDFS="05 15 65 75 85 95 e5 f5"
IMG=$(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null |
	grep -viE '<none>|^k8s|pause' | head -1)

docker run --rm --privileged --pid=host --net=host --entrypoint /bin/bash \
	-v /lib/modules:/lib/modules:ro -v /sys:/sys -v /dev:/dev \
	"$IMG" -lc '
set -u
BDFS="'"$BDFS"'"
for b in $BDFS; do
  d=/sys/bus/pci/devices/0000:$b:00.0
  cur=$(cat $d/current_compute_partition 2>/dev/null)
  if [ "$cur" != SPX ]; then
    if echo SPX > $d/current_compute_partition 2>/dev/null; then
      echo "  0000:$b:00.0 $cur -> SPX ok"
    else
      echo "  0000:$b:00.0 SPX REFUSED (still $cur)"
    fi
  else
    echo "  0000:$b:00.0 already SPX"
  fi
done
echo "=== final ==="
ok=0
for b in $BDFS; do
  d=/sys/bus/pci/devices/0000:$b:00.0
  cp=$(cat $d/current_compute_partition 2>/dev/null)
  mp=$(cat $d/current_memory_partition 2>/dev/null)
  echo "  0000:$b:00.0  $cp / $mp"
  [ "$cp" = SPX ] && [ "$mp" = NPS2 ] && ok=$((ok+1))
done
echo "  $ok of 8 in SPX/NPS2"
'

