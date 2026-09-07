#!/bin/bash
# The whole gate in one pass, so the record is one artefact rather than three.
#
# Sections 1-3b ask whether the check can fail; section 4 is the only one that
# says anything about the fix, and it only counts if the earlier ones pass.
#
# Not included: --aid=1 --coherent=1 --split=0. That configuration homes the
# release flags in AID0 under a coherent MTYPE and has all eight XCDs poll
# them, so the four outside that partition spin on a stale line forever. It
# hangs by construction -- the co-location rule failing as documented -- and it
# wedges the GPU until the process is killed, so it is run separately.
set -u
cd "$HOME/fleet-chiplet-megakernel" || exit 1

L=64
H() { grep -o 'hash .*' | head -1; }
W() { grep -o 'wall .*' | head -1; }

echo "=== 0. per-layer cost, so the gate build can be compared to the timed one ==="
timeout 120 ./drive_phase7 --layers=$L --aid=1 --coherent=1 --split=1 --tag=w 2>&1 | W
echo

echo "=== 1. sensitivity: the hash has to differ per layer ==="
echo "    (otherwise reading the previous layer's data hashes the same and the"
echo "     comparison in section 4 proves nothing)"
for n in 16 17 18 19; do
	printf 'layers=%-4s ' "$n"
	timeout 120 ./drive_phase7 --layers=$n --aid=1 --coherent=1 --split=1 --tag=s 2>&1 | H
	echo
done

echo
echo "=== 2. determinism: 6 identical runs ==="
for r in 1 2 3 4 5 6; do
	printf 'rep%-3s ' "$r"
	timeout 120 ./drive_phase7 --layers=$L --aid=1 --coherent=1 --split=1 --tag=d 2>&1 | H
	echo
done

echo
echo "=== 3. teeth A: the hierarchical poll compiled out ==="
echo "    expected to corrupt rmsnorm_out only -- that barrier guards the row"
echo "    read, not each block's own columns"
for r in 1 2 3 4 5 6; do
	printf 'nopoll  rep%-3s ' "$r"
	timeout 120 ./drive_phase7_nopoll --layers=$L --aid=1 --coherent=1 \
		--split=1 --skew=20000 --tag=n 2>&1 | H
	echo
done

echo
echo "=== 3b. teeth B: the attn-slice poll compiled out ==="
echo "    expected to corrupt attn_proj_out as well -- that barrier guards the"
echo "    reduction's own input"
for r in 1 2 3 4 5 6; do
	printf 'noslice rep%-3s ' "$r"
	timeout 120 ./drive_phase7_noslice --layers=$L --aid=1 --coherent=1 \
		--split=1 --skew=20000 --tag=x 2>&1 | H
	echo
done

echo
echo "=== 3c. the correct build under the same skew, for contrast ==="
for r in 1 2 3; do
	printf 'correct rep%-3s ' "$r"
	timeout 120 ./drive_phase7 --layers=$L --aid=1 --coherent=1 \
		--split=1 --skew=20000 --tag=c 2>&1 | H
	echo
done

echo
echo "=== 4. the three sound configurations must produce the same hash ==="
echo "    A  aid=0                     single copy, normal HBM (the stock path)"
echo "    B  aid=1 coherent=0 split=1  AID-local, non-coherent MTYPE"
echo "    C  aid=1 coherent=1 split=1  AID-local, coherent, per-AID replicas"
echo
for cfg in "A --aid=0" "B --aid=1 --coherent=0 --split=1" "C --aid=1 --coherent=1 --split=1"; do
	set -- $cfg
	name=$1
	shift
	for r in 1 2 3 4 5 6; do
		printf '%s rep%-3s ' "$name" "$r"
		timeout 120 ./drive_phase7 --layers=$L "$@" --tag=$name 2>&1 | H
		echo
	done
done
