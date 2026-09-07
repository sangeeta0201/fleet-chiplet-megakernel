#!/bin/bash
# gate_hrdv.sh - does the hierarchical inter-layer rendezvous still compute the
# same thing, and does making it AID-aware move `bar`?
#
# The flat rendezvous it replaces has all 184 blocks increment both replicas and
# then poll their own, which is 368 atomics on two lines per layer with half of
# them crossing the partition boundary. Tracing Phase 7's own barrier showed that
# this, and not the barrier, is what staggers the two halves: the eight XCDs enter
# Phase 7 0.84 us apart in NPS2 against 0.09 us in NPS1, and Phase 7 adds only
# 0.05 us of its own. So the tree is aimed one level up from the barrier.
#
# The failure mode is silent, which is why this gate exists. Every counter is
# monotonic and tested on its residue, so a lost or misrouted increment does not
# hang -- it releases a layer early, and the only evidence is a changed
# rmsnorm_out hash. Every run is under `timeout` because the other failure mode,
# a mispaired peer slot, deadlocks instead.
set -u
cd "$HOME/fleet-chiplet-megakernel" || exit 1

# Same reasoning as gate_lsplit.sh: the recorded pair is specific to 400 layers
# because the harness residual is a function of `layer & 15`, so the reference is
# taken from the first configuration and only compared to the recorded pair when
# LAYERS is 400.
LAYERS=${LAYERS:-400}
REF_OUT=""
REF_NORM=""
FAILED=0

run() {
	local label=$1
	shift
	printf '%-18s ' "$label"
	local log=/tmp/gate_hrdv_$label.log
	timeout 240 ./drive_phase7 --layers="$LAYERS" --tiles=23 "$@" --tag="$label" \
		>"$log" 2>&1
	local rc=$?
	if [ "$rc" != "0" ]; then
		echo "RUN FAILED rc=$rc (timeout 240 = deadlock, check the peer slots)"
		tail -3 "$log" | sed 's/^/    /'
		FAILED=1
		return
	fi
	local h go gn bar
	h=$(grep -o 'attn_proj_out=[0-9a-f]* rmsnorm_out=[0-9a-f]*' "$log")
	go=${h#attn_proj_out=}
	go=${go%% *}
	gn=${h##*rmsnorm_out=}
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

CP=$(cat /sys/bus/pci/devices/0000:05:00.0/current_compute_partition)
MP=$(cat /sys/bus/pci/devices/0000:05:00.0/current_memory_partition)
echo "=== hierarchical rendezvous gate, $LAYERS layers, $CP/$MP ==="

run stock --aid=0
run flat_split --aid=1 --coherent=1 --split=1 --lsplit=1
run tree_split --aid=1 --coherent=1 --split=1 --lsplit=1 --hrdv=1
# The shape-versus-placement control: the same tree with one coherence domain and
# both region pointers aliased. If `tree_onedomain` is as fast as `tree_split`
# then the win is the fan-in, not the homing, and vice versa. It also exercises
# the aliased path, where a topology indexed by placement rather than by half
# would release on the fourth arrival instead of the eighth -- silently.
run tree_onedomain --aid=0 --hrdv=1

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
