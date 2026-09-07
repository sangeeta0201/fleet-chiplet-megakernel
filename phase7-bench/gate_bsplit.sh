#!/bin/bash
# gate_bsplit.sh - does splitting the barrier's level-2 aggregation still order
# correctly?
#
# This is the riskiest of the three splits, because it changes the barrier's
# shape rather than just where its lines live. Before, one counter saw all eight
# dies and one releaser published sixteen flags. Now each half counts its own four
# dies, the two half-closers rendezvous through a pair of single-writer slots, and
# each publishes only the four flags its own XCDs read. Two things can go wrong
# and neither is loud:
#
#   - A mispaired handshake slot deadlocks, so every run is under `timeout`.
#   - Releasing on the fourth arrival instead of the eighth does not fail, it
#     lets RMSNorm read a row that is missing four dies' columns. The only
#     evidence is a changed rmsnorm_out hash, which is what this compares.
#
# The second is the reason the flag-only configurations are run alongside: if the
# handshake were wrong in a way that released early, `l2split` would disagree with
# the three configurations that share its placement but not its topology.
set -u
cd "$HOME/fleet-chiplet-megakernel" || exit 1

LAYERS=${LAYERS:-400}
REF_OUT=""
REF_NORM=""
FAILED=0

run() {
	local label=$1
	shift
	printf '%-18s ' "$label"
	local log=/tmp/gate_bsplit_$label.log
	timeout 240 ./drive_phase7 --layers="$LAYERS" --tiles=23 "$@" --tag="$label" \
		>"$log" 2>&1
	local rc=$?
	if [ "$rc" != "0" ]; then
		echo "RUN FAILED rc=$rc (124 = deadlock, check the handshake slots)"
		tail -3 "$log" | sed 's/^/    /'
		FAILED=1
		return
	fi
	local h go gn bar l2
	h=$(grep -o 'attn_proj_out=[0-9a-f]* rmsnorm_out=[0-9a-f]*' "$log")
	go=${h#attn_proj_out=}
	go=${go%% *}
	gn=${h##*rmsnorm_out=}
	l2=$(grep -m1 -o 'l2_split=[01]' "$log")
	bar=$(grep OPROJ_INNER "$log" | sed 's/.*bar=\([0-9.]*\).*/\1/' |
		awk -v n="$(grep -c OPROJ_INNER "$log")" 'NR>n*0.3' | sort -n |
		awk '{v[NR]=$1} END {printf "%.2f", v[int(NR/2)]}')
	if [ -z "$REF_OUT" ]; then
		REF_OUT=$go
		REF_NORM=$gn
		echo "bar=${bar} us  $l2  $go $gn  (reference)"
	elif [ "$go" = "$REF_OUT" ] && [ "$gn" = "$REF_NORM" ]; then
		echo "bar=${bar} us  $l2  agrees"
	else
		echo "bar=${bar} us  $l2  HASH MISMATCH"
		echo "    got $go $gn"
		echo "    ref $REF_OUT $REF_NORM"
		FAILED=1
	fi
}

CP=$(cat /sys/bus/pci/devices/0000:05:00.0/current_compute_partition)
MP=$(cat /sys/bus/pci/devices/0000:05:00.0/current_memory_partition)
echo "=== level-2 split gate, $LAYERS layers, $CP/$MP ==="

run stock --aid=0
run flags_only --aid=1 --coherent=1 --split=1 --lsplit=1
run flags_tree --aid=1 --coherent=1 --split=1 --lsplit=1 --hrdv=1
run l2split --aid=1 --coherent=1 --split=1 --lsplit=1 --bsplit=1
run l2split_tree --aid=1 --coherent=1 --split=1 --lsplit=1 --hrdv=1 --bsplit=1

# --bsplit without --split must refuse rather than run: two releasers with one
# shared replica would have each half publishing into the other's partition,
# which is the co-location violation, and it hangs instead of being slow.
printf '%-18s ' "bsplit_no_split"
if timeout 60 ./drive_phase7 --layers=20 --tiles=23 --aid=1 --coherent=1 \
	--split=0 --bsplit=1 --tag=refuse >/tmp/gate_bsplit_refuse.log 2>&1; then
	echo "DID NOT REFUSE - it should abort, not run"
	FAILED=1
else
	if grep -q 'needs --split=1' /tmp/gate_bsplit_refuse.log; then
		echo "refused, as it should"
	else
		echo "failed for the wrong reason:"
		tail -2 /tmp/gate_bsplit_refuse.log | sed 's/^/    /'
		FAILED=1
	fi
fi

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
