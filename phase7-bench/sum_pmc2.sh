#!/bin/bash
# The kernel name is "drive_phase7(void*, void*, ...)" -- it contains commas, so
# fixed column indices shift. The last four fields are always
# Counter_Name, Counter_Value, Start_Timestamp, End_Timestamp, so index from NF.
set -u
get() { # $1=pass dir
	find "$HOME/nps1/pmc/$1" -name '*counter_collection.csv' | head -1
}
dump() {
	awk -F, '/drive_phase7/ {
		cn=$(NF-3); cv=$(NF-2); gsub(/"/,"",cn);
		s[cn]+=cv; n[cn]++
	} END { for (k in s) printf "  %-26s %18.0f   (%d rows)\n", k, s[k], n[k] }' "$1" | sort
}
val() {
	awk -F, -v want="$2" '/drive_phase7/ {
		cn=$(NF-3); cv=$(NF-2); gsub(/"/,"",cn);
		if (cn==want) s+=cv
	} END { printf "%.0f", s+0 }' "$1"
}

for p in l2 ea gmi; do
	f=$(get $p); echo "=== $p ==="; dump "$f"
done

L=$(get l2); E=$(get ea); G=$(get gmi)
hit=$(val "$L" TCC_HIT); miss=$(val "$L" TCC_MISS)
rd=$(val "$E" TCC_EA0_RDREQ); dram=$(val "$E" TCC_EA0_RDREQ_DRAM)
gmi=$(val "$G" TCC_EA0_RDREQ_GMI_32B); all32=$(val "$G" TCC_EA0_RDREQ_32B)

echo
echo "=== derived (8 layers, 184 tiles) ==="
awk -v h="$hit" -v m="$miss" -v rd="$rd" -v d="$dram" -v g="$gmi" -v a="$all32" 'BEGIN {
	tot = h + m;
	if (tot > 0) {
		printf "  L2 requests        : %.0f  (hit %.0f / miss %.0f)\n", tot, h, m;
		printf "  L2 hit rate        : %.2f%%\n", 100*h/tot;
		printf "  L2 miss traffic    : %.2f MiB  (misses x 128 B)\n", m*128/1048576;
	}
	printf "  L2 fill reqs (EA)  : %.0f\n", rd;
	printf "  ... reaching DRAM  : %.0f  -> %.2f MiB at 128 B/req\n", d, d*128/1048576;
	printf "  cross-die GMI 32B  : %.0f   (all 32B reads %.0f)\n", g, a;
	if (rd > 0) printf "  GMI share of fills : %.2f%%\n", 100*g/rd;
}'

