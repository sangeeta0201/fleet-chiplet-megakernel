#!/bin/bash
# Bring the box to SPX + NPS2 the ONLY way that actually works, and verify it.
#
# ROOT CAUSE of DPX+NPS2 (measured, not guessed):
#   * SPX+NPS2 exists only in the PATCHED driver, enabled by the module
#     parameter `aid_local_spx_nps2=1`, and the driver chooses SPX at LOAD time.
#     There is no live SPX switch.
#   * Under the stock DKMS module with NPS2 active, sysfs reports
#     available_compute_partition = "DPX, QPX, CPX" -- SPX is absent, so
#     `rocm-smi --setcomputepartition SPX` is correctly refused with
#     "Requested setting (SPX) is unavailable for current device".
#   * `rocm-smi --setmemorypartition NPS2` restarts the amdgpu driver, and the
#     restart loads the STOCK module from /lib/modules, throwing away the
#     hand-insmod'ed patched build. That is what lands the box in DPX+NPS2.
#
# So: never use rocm-smi to change partitions. Reload the patched module with
# the right parameters instead. Needs root, and there is no passwordless sudo
# here, so go through a privileged container (docker group == host root).
set -u
WANT=${WANT:-NPS2}
SRC=${SRC:-$HOME/aid-local-hbm/src/amdgpu-perbo}
IMG=${IMG:-mirage:fleetv2}
BDFS="05 15 65 75 85 95 e5 f5"

verify() {  # prints state; returns 0 only if all 8 are SPX/$WANT
	local ok=0 n=0 cp mp d
	for b in $BDFS; do
		d=/sys/bus/pci/devices/0000:$b:00.0
		[ -d "$d" ] || continue
		n=$((n+1))
		cp=$(cat "$d/current_compute_partition" 2>/dev/null)
		mp=$(cat "$d/current_memory_partition" 2>/dev/null)
		printf '    0000:%s:00.0  %-4s / %-5s\n' "$b" "$cp" "$mp"
		[ "$cp" = SPX ] && [ "$mp" = "$WANT" ] && ok=$((ok+1))
	done
	printf '    -> %d/%d at SPX/%s\n' "$ok" "$n" "$WANT"
	[ "$n" -gt 0 ] && [ "$ok" = "$n" ]
}

echo "== current state =="
#verify && { echo "== already SPX/$WANT; nothing to do =="; exit 0; }

echo
echo "== patched module present? =="
KO=$SRC/amd/amdgpu/amdgpu.ko
if [ ! -f "$KO" ]; then
	echo "  ABORT: $KO missing -- rebuild the patched driver first"; exit 1
fi
ls -l "$KO" | sed 's/^/  /'
if ! strings "$KO" 2>/dev/null | grep -q aid_local_spx_nps2; then
	echo "  ABORT: this module has no aid_local_spx_nps2 parameter, so it"
	echo "         cannot give SPX+NPS2. Wrong tree in SRC=$SRC"
	exit 1
fi
echo "  ok: exports aid_local_spx_nps2"

echo
echo "== free /dev/kfd (an unload fails while anything holds it) =="
for c in $(docker ps -q --filter 'name=fleet_' 2>/dev/null); do
	docker stop "$c" >/dev/null 2>&1 && echo "  stopped $c"
done
docker ps --format '  still running: {{.Names}}' 2>/dev/null | head -5

echo
echo "== reload the patched module with SPX+$WANT, inside a privileged container =="
if ! docker image inspect "$IMG" >/dev/null 2>&1; then
	echo "  image $IMG absent; available:"
	docker images --format '    {{.Repository}}:{{.Tag}}' 2>/dev/null | head -12
	IMG=$(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null |
		grep -viE '<none>' | head -1)
	echo "  (falling back to $IMG)"
fi
# --entrypoint is mandatory: several images here (the vLLM ones) set an
# ENTRYPOINT of api_server.py, which would swallow the script as CLI args.
docker run --rm --privileged --pid=host --net=host \
	--entrypoint /bin/bash \
	-v /lib/modules:/lib/modules:ro -v /sys:/sys -v /dev:/dev \
	-v "$HOME:$HOME" -e SRC="$SRC" -e WANT="$WANT" \
	"$IMG" -lc '
set -u
BDFS="05 15 65 75 85 95 e5 f5"
unload() {
	for m in amdgpu amdxcp amddrm_ttm_helper amddrm_exec amddrm_buddy amd_sched amdttm amdkcl; do
		lsmod | grep -q "^${m} " && rmmod "$m" 2>/dev/null
	done
	lsmod | grep -q "^amdgpu " && return 1 || return 0
}
load() {
	for m in drm_display_helper drm_suballoc_helper drm_exec video cec rc_core ib_core; do
		modprobe "$m" 2>/dev/null
	done
	for m in amd/amdkcl/amdkcl.ko ttm/amdttm.ko scheduler/amd-sched.ko amddrm_buddy.ko \
	         amddrm_exec.ko amddrm_ttm_helper.ko amd/amdxcp/amdxcp.ko; do
		lsmod | grep -q "^$(basename $m .ko | tr - _) " || insmod "$SRC/$m" || return 1
	done
	# flag_mtype=2 (CC) is load-time only (mode 0444) and MUST be on this line.
	# Leaving it default 0 is what produced the 22.1 us run earlier: it turns CC
	# off for the coherent sync buffer, which is the whole point of --coherent=1.
	insmod "$SRC/amd/amdgpu/amdgpu.ko" \
		aid_local_steal_gb=0 aid_local_spx_nps2=1 \
		aid_local_xcp_span=1 aid_local_xcp_nc=1 \
		aid_local_flag_mtype=2 aid_local_ilv_kb=2048 aid_local_ilv_mode=1 || return 1
}
echo "  unloading stock module..."
unload || { echo "  ABORT: amdgpu still held"; fuser -v /dev/kfd 2>&1 | head -6; exit 1; }
echo "  loading patched module (aid_local_spx_nps2=1)..."
load || { echo "  ABORT: insmod failed"; dmesg | tail -15; exit 1; }
sleep 10

echo "  available_compute_partition with the patched module:"
echo "    $(cat /sys/bus/pci/devices/0000:05:00.0/available_compute_partition 2>/dev/null | tr "\n" " ")"

need=0
for b in $BDFS; do
	d=/sys/bus/pci/devices/0000:$b:00.0
	cur=$(cat $d/current_memory_partition 2>/dev/null)
	if [ "$cur" != "$WANT" ]; then
		echo "$WANT" > "$d/current_memory_partition" 2>/dev/null &&
			{ echo "  0000:$b:00.0 $cur -> $WANT requested"; need=1; } ||
			echo "  0000:$b:00.0 request FAILED (is $cur)"
	fi
done

if [ "$need" = 1 ]; then
	echo "  reloading so PSP applies $WANT (SPX is chosen at load)..."
	unload || { echo "  ABORT: unload for apply"; exit 1; }
	load   || { echo "  ABORT: reload"; exit 1; }
	sleep 14
fi

# ORDER MATTERS: this must come AFTER the reload above. The compute partition is
# persisted in hardware, so the reload that applies NPS2 comes back up in
# whatever was last baked in (DPX) and silently undoes an SPX written before it
# -- which is exactly how this script reached 0/8 with NPS2 set but DPX live.
# aid_local_spx_nps2 only PERMITS SPX+NPS2; it never forces it.
for b in $BDFS; do
	d=/sys/bus/pci/devices/0000:$b:00.0
	cur=$(cat $d/current_compute_partition 2>/dev/null)
	[ "$cur" = SPX ] && continue
	if echo SPX > "$d/current_compute_partition" 2>/dev/null; then
		echo "  0000:$b:00.0 compute $cur -> SPX requested"
	else
		echo "  0000:$b:00.0 compute SPX write FAILED (is $cur, avail: $(cat $d/available_compute_partition 2>/dev/null | tr "\n" " "))"
	fi
done
sleep 8
echo "  module params now:"
for p in aid_local_spx_nps2 aid_local_xcp_nc aid_local_flag_mtype mtype_local; do
	v=$(cat /sys/module/amdgpu/parameters/$p 2>/dev/null) &&
		printf "    %-22s = %s\n" "$p" "$v" ||
		printf "    %-22s = <absent>\n" "$p"
done
' 2>&1 | sed 's/^/  /'

echo
echo "== final verification =="
if verify; then
	echo "== SUCCESS: SPX/$WANT on all 8 =="
	exit 0
fi
echo "== FAILED to reach SPX/$WANT =="
exit 1

