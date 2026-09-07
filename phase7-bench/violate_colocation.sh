#!/bin/bash
# Violate the co-location rule on purpose, and confirm it hangs.
#
# --aid=1 --coherent=1 --split=0 homes the release flags in AID0 under a
# coherent MTYPE and has all eight XCDs poll them. In SPX a compute partition
# spans both memory partitions, so the four XCDs outside AID0 are cached readers
# of a line homed in the other domain -- and they spin on a stale copy forever.
#
# This is the failure mode the README's co-location rule describes, kept as an
# executable check because "coherent MTYPE is only sound for co-located readers"
# is the whole reason the fix needs two replicas rather than one.
#
# It wedges the GPU until the process dies, so it is deliberately not part of
# gate_all.sh. Run killhang.sh afterwards if the timeout does not clear it.
set -u
cd "$HOME/fleet-chiplet-megakernel" || exit 1

echo "=== aid=1 coherent=1 split=0: coherent flags in AID0, all 8 XCDs polling ==="
timeout -s KILL 45 ./drive_phase7 --layers=8 --aid=1 --coherent=1 --split=0 \
	--tag=violate 2>&1 | grep -E 'hash |wall '
rc=${PIPESTATUS[0]}

if [ "$rc" = "137" ]; then
	echo "HUNG as expected (killed at 45s): out-of-domain XCDs never see the release"
	exit 0
fi
echo "completed with rc=$rc -- the violation did NOT hang, which needs explaining"
exit 1
