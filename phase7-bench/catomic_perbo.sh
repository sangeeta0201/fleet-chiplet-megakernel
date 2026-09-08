#!/bin/bash
# Apply the now-working mechanism to the buffer that actually uses it.
#
# --catomic places `counters` (the level-2 arrival counter + topk_counter, both
# touched only by device-scope atomics) as an AID_LOCAL BO. Its comment claims
# "1 -> MTYPE_RW, the MTYPE NPS1 gives this line" -- but under the old driver
# xcp_nc=Y forced is_local=false for every VRAM BO in a spanning XCP, so the BO
# was demoted to NC and --catomic=1 measured NC, never RW. The perbo patch is
# exactly what makes that claim true, so this is the first real test of it.
#
#   0  plain hipMalloc            -> NC (what every number so far measured)
#   1  AID0, non-coherent         -> MTYPE_RW  <-- the untested case
#   2  AID0, coherent             -> CC (via aid_local_flag_mtype=2)
#   3  AID1, non-coherent         -> RW, other partition = placement control
#
# --bsplit=2 is the harness's own sign-flip control (deliberately home each
# half's counter in the WRONG partition), the same methodology the Mosaic paper
# uses for its 28% die-local-vs-remote swing.
set -u
REF=ff1bdbe7ad2c4ed3
cd "$HOME/fleet-chiplet-megakernel" || exit 1
B="--aid=1 --coherent=1 --split=1 --lsplit=1 --hrdv=1"
for a in c0 c1 c2 c3 bs; do rm -f /tmp/$a; done

echo "driver: xcp_nc=$(cat /sys/module/amdgpu/parameters/aid_local_xcp_nc)" \
     "flag_mtype=$(cat /sys/module/amdgpu/parameters/aid_local_flag_mtype)" \
     "srcversion=$(cat /sys/module/amdgpu/srcversion)"
echo

col() {
	local n; n=$(grep -c OPROJ_INNER "$1")
	grep OPROJ_INNER "$1" | sed "s/.*$2=\([0-9.]*\).*/\1/" |
		awk -v n="$n" 'NR>n*0.3' | sort -n |
		awk '{v[NR]=$1} END {printf "%.3f", v[int(NR/2)]}'
}
printf '%-4s %-28s %8s %8s %9s  %s\n' rep arm mfma bar total ok
run() {
	local rep=$1 label=$2 acc=$3; shift 3
	local log=$HOME/nps1/cp_${acc}_$rep.log
	timeout 240 ./drive_phase7 --layers=400 --tiles=23 $B "$@" --tag=cp >"$log" 2>&1
	local rc=$? h t
	if [ $rc != 0 ]; then
		printf '%-4s %-28s %s\n' "$rep" "$label" \
			"$([ $rc = 124 ] && echo HANG || echo "rc=$rc $(grep -iE 'ABORT|error' "$log" | head -1)")"
		return
	fi
	h=$(grep -o 'rmsnorm_out=[0-9a-f]*' "$log" | head -1); h=${h#rmsnorm_out=}
	t=$(col "$log" total); echo "$t" >>/tmp/$acc
	printf '%-4s %-28s %8s %8s %9s  %s\n' "$rep" "$label" \
		"$(col "$log" mfma)" "$(col "$log" bar)" "$t" \
		"$([ "$h" = "$REF" ] && echo OK || echo "WRONG $h")"
}
for r in 1 2 3 4; do
	run "$r" "counters NC (baseline)"     c0
	run "$r" "counters AID0 RW  <-- new"  c1 --catomic=1
	run "$r" "counters AID0 CC"           c2 --catomic=2
	run "$r" "counters AID1 RW (control)" c3 --catomic=3
	run "$r" "bsplit=2 wrong-AID control" bs --catomic=1 --bsplit=2
done
echo
for a in c0 c1 c2 c3 bs; do
	[ -s /tmp/$a ] || continue
	sort -n /tmp/$a | awk -v a="$a" '{v[NR]=$1} END {
		printf "  %-3s total: min %.2f  median %.2f  max %.2f\n", a, v[1], v[int((NR+1)/2)], v[NR]}'
done
echo "  NPS1 on this node: 9.76 / 9.80 / 9.88"
for a in c0 c1 c2 c3 bs; do rm -f /tmp/$a; done

