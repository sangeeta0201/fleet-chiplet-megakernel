#!/bin/bash
# Default path unchanged after two rounds of patching: same flags, same
# reference hash, same numbers as before --srdv existed.
set -u
cd "$HOME/fleet-chiplet-megakernel" || exit 1
col() {
	local n; n=$(grep -c OPROJ_INNER "$1")
	grep OPROJ_INNER "$1" | sed "s/.*$2=\([0-9.]*\).*/\1/" |
		awk -v n="$n" 'NR>n*0.3' | sort -n |
		awk '{v[NR]=$1} END {printf "%.3f", v[int(NR/2)]}'
}
printf '%-6s %8s %9s  %-16s %s\n' rep bar total hash ok
for r in 1 2 3 4; do
	log=$HOME/nps1/fin$r.log
	./drive_phase7 --layers=400 --tiles=23 --aid=1 --coherent=1 \
		--split=1 --lsplit=1 --hrdv=1 --tag=fin >"$log" 2>&1
	h=$(grep -o 'rmsnorm_out=[0-9a-f]*' "$log" | head -1); h=${h#rmsnorm_out=}
	printf '%-6s %8s %9s  %-16s %s\n' "$r" "$(col "$log" bar)" \
		"$(col "$log" total)" "$h" \
		"$([ "$h" = ff1bdbe7ad2c4ed3 ] && echo OK || echo WRONG)"
done

