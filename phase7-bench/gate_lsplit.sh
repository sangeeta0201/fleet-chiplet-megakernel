#!/bin/bash
# gate_lsplit.sh  does homing the level-1 arrival lines per AID still compute
# the same thing?
#
# Level 1 is the eight per-XCD arrival counters of the tree barrier. Each line
# is only ever incremented by one XCD's own workers, who are co-located with
# each other, so moving line x into the partition that XCD x lives in should be
# sound under a coherent MTYPE with no replication and no fan-out. "Should be" is
# why this script exists: the arrival counter is a monotonic `% tiles_per_xcd`
# release, so a lost or misrouted increment does not fail loudly -- it releases
# the barrier early, which shows up as a changed rmsnorm_out hash and nothing
# else.
#
# Reference hashes are the ones every configuration in results/nps1-vs-nps2.md
# agrees on. A hang here means the co-location assumption (XCD 0-3 -> AID0) is
# wrong, so every run is under `timeout`.
set -u
cd "$HOME/fleet-chiplet-megakernel" || exit 1

# The recorded reference in results/nps1-vs-nps2.md is specific to 400 layers.
# It has to be: the harness's per-column residual is `0.5 * ((gcol + layer) & 15)`
# and its producer value depends on `layer & 3`, so the final buffer is a
# function of where the layer loop stopped. Running this gate at any other depth
# is still a valid mutual-agreement test, it just cannot be compared to the
# recorded hashes -- so the reference is taken from the first configuration and
# only cross-checked against the recorded pair when LAYERS is 400.
LAYERS=${LAYERS:-400}
REF_OUT=""
REF_NORM=""
FAILED=0

run() {
	local label=$1
	shift
	printf '%-16s ' "$label"
	local log=/tmp/gate_lsplit_$label.log
	timeout 240 ./drive_phase7 --layers="$LAYERS" --tiles=23 "$@" --tag="$label" \
		>"$log" 2>&1
	local rc=$?
	if [ "$rc" != "0" ]; then
		echo "FAIL rc=$rc (240s timeout means it hung)"
		tail -3 "$log" | sed 's/^/    /'
		FAILED=1
		return
	fi
	local h go gn bar
	h=$(grep -o 'attn_proj_out=[0-9a-f]* rmsnorm_out=[0-9a-f]*' "$log")
	go=${h#attn_proj_out=}
	go=${go%% *}
	gn=${h##*rmsnorm_out=}
	# Median of the bar bucket, warmup dropped the same way buckets.sh does it.
	bar=$(grep OPROJ_INNER "$log" | sed 's/.*bar=\([0-9.]*\).*/\1/' |
		awk -v n="$(grep -c OPROJ_INNER "$log")" 'NR>n*0.3' | sort -n |
		awk '{v[NR]=$1} END {printf "%.2f", v[int(NR/2)]}')
	if [ -z "$REF_OUT" ]; then
		REF_OUT=$go
		REF_NORM=$gn
		echo "bar=${bar} us   $go $gn   (reference)"
	elif [ "$go" = "$REF_OUT" ] && [ "$gn" = "$REF_NORM" ]; then
		echo "bar=${bar} us   agrees"
	else
		echo "bar=${bar} us   HASH MISMATCH"
		echo "    got $go $gn"
		echo "    ref $REF_OUT $REF_NORM"
		FAILED=1
	fi
}

echo "=== level-1 AID split gate, $LAYERS layers, $(cat /sys/bus/pci/devices/0000:05:00.0/current_compute_partition)/$(cat /sys/bus/pci/devices/0000:05:00.0/current_memory_partition) ==="
run stock --aid=0
run relsplit --aid=1 --coherent=1 --split=1 --lsplit=0
run rel_and_local --aid=1 --coherent=1 --split=1 --lsplit=1

# Deliberately not tested here: --split=0 --lsplit=1. `--split=0` also unroutes
# the *attention* slice flags (drive_phase7's `rel = upper ? flags_b : flags`),
# so all eight XCDs would poll one coherent line homed in AID0 -- the
# co-location violation violate_colocation.sh exists to demonstrate. It hangs
# and wedges the GPU, which says nothing about level 1.

if [ "$LAYERS" = "400" ]; then
	if [ "$REF_OUT" = "d5f0c50124e2cb83" ] &&
		[ "$REF_NORM" = "ff1bdbe7ad2c4ed3" ]; then
		echo "reference matches results/nps1-vs-nps2.md"
	else
		echo "WARNING: 400 layers but the reference pair is not the recorded one"
		FAILED=1
	fi
fi
[ "$FAILED" = "0" ] && echo "GATE PASS" || echo "GATE FAIL"
exit "$FAILED"
