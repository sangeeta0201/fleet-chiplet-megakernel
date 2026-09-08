#!/bin/bash
# Where does Phase 7's read traffic actually come from? The L2-resident claim has
# only ever been inferred (82 ns flat probe, insensitive mfma). Settle it with
# counters:
#
#   TCC_HIT / TCC_MISS        -> L2 hit rate. If ~100%, reads never leave L2.
#   TCC_EA0_RDREQ             -> L2 fill requests leaving to the fabric at all
#   TCC_EA0_RDREQ_DRAM        -> of those, the ones that reached HBM
#   TCC_EA0_RDREQ_GMI_32B     -> of those, the ones that crossed the die-to-die
#                                link. This is the inter-die traffic the Mosaic
#                                paper measures directly; it is the number that
#                                decides whether placement has anything to remove.
#
# Counter slots are limited, so collect in small passes rather than one big set.
set -u
cd "$HOME/fleet-chiplet-megakernel" || exit 1
OUT=$HOME/nps1/pmc; rm -rf "$OUT"; mkdir -p "$OUT"

echo "=== state ==="
printf '  compute/memory partition: %s / %s\n' \
	"$(cat /sys/bus/pci/devices/0000:65:00.0/current_compute_partition 2>/dev/null)" \
	"$(cat /sys/bus/pci/devices/0000:65:00.0/current_memory_partition 2>/dev/null)"
printf '  amdgpu srcversion: %s\n' "$(cat /sys/module/amdgpu/srcversion 2>/dev/null || echo '(none)')"
printf '  aid_local params present: %s\n' \
	"$(ls /sys/module/amdgpu/parameters/ 2>/dev/null | grep -c aid_local)"

# Short run: the question is a ratio, not a duration, so few layers is enough
# and keeps the profiler's serialization overhead from dominating.
ARGS="--layers=8 --tiles=23 --aid=0 --split=0 --lsplit=0 --hrdv=0 --tag=pmc"

pass() {
	local name=$1; shift
	echo
	echo "=== $name : $* ==="
	local d="$OUT/$name"; mkdir -p "$d"
	if timeout 600 rocprofv3 --pmc "$@" --output-format csv -d "$d" \
		-- ./drive_phase7 $ARGS >"$d/run.log" 2>&1; then
		# sum the counter column across every dispatch and every TCC channel
		find "$d" -name '*counter_collection.csv' | head -1 | while read -r f; do
			awk -F, 'NR>1 {
				gsub(/"/,"",$8); gsub(/"/,"",$9);
				s[$8]+=$9
			} END { for (k in s) printf "  %-28s %18.0f\n", k, s[k] }' "$f" | sort
		done
	else
		echo "  FAILED (rc=$?)"; tail -6 "$d/run.log" | sed 's/^/    /'
	fi
}

pass l2   TCC_HIT TCC_MISS
pass ea   TCC_EA0_RDREQ TCC_EA0_RDREQ_DRAM
pass gmi  TCC_EA0_RDREQ_GMI_32B TCC_EA0_RDREQ_32B

echo
echo "logs under $OUT"

