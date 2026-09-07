#!/bin/bash
# cmp_l2.sh - the level-2 split against the shared counter it replaces, with the
# handshake poll both sleeping and busy.
#
# The split is correct (gate_bsplit.sh) but slower, and the question is whether
# that is the handshake's structure or just the poll's granularity. Under a
# balanced load the peer half has already arrived by the time its counterpart
# looks, so an `s_sleep(1)` between attempts could be the entire difference.
set -u
cd "$HOME/fleet-chiplet-megakernel" || exit 1

LAYERS=${LAYERS:-300}
BASE="--aid=1 --coherent=1 --split=1 --lsplit=1 --hrdv=1"

med() {
	grep OPROJ_INNER "$1" |
		sed 's/.*bar=\([0-9.]*\).*/\1/' |
		awk -v n="$(grep -c OPROJ_INNER "$1")" 'NR>n*0.3' | sort -n |
		awk '{v[NR]=$1} END {printf "%.2f", v[int(NR/2)]}'
}
tot() {
	grep OPROJ_INNER "$1" |
		sed 's/.*total=\([0-9.]*\).*/\1/' |
		awk -v n="$(grep -c OPROJ_INNER "$1")" 'NR>n*0.3' | sort -n |
		awk '{v[NR]=$1} END {printf "%.2f", v[int(NR/2)]}'
}

printf '%-34s %8s %8s\n' config bar total
for bin in drive_phase7 drive_phase7_busy; do
	[ -x "./$bin" ] || continue
	for bs in 0 1; do
		log=/tmp/cmpl2_${bin}_$bs.log
		timeout 300 ./"$bin" --layers="$LAYERS" --tiles=23 $BASE \
			--bsplit="$bs" --tag=cmp >"$log" 2>&1 || {
			printf '%-34s  run failed\n' "$bin --bsplit=$bs"
			continue
		}
		printf '%-34s %8s %8s\n' "$bin --bsplit=$bs" "$(med "$log")" \
			"$(tot "$log")"
	done
done
