#!/bin/bash
# Does staging the weight tile once agree with staging it every layer?
#
# If it does, nothing in the kernel writes into the LDS weight region between
# layers, and the cheap build is the one to gate on. If it does not, the region
# is being clobbered and only per-layer staging is correct.
set -u
cd "$HOME/fleet-chiplet-megakernel" || exit 1

for b in drive_phase7 drive_phase7_once; do
	echo "-- $b --"
	for r in 1 2 3; do
		timeout 120 "./$b" --layers=64 --aid=1 --coherent=1 --split=1 \
			--tag="$b" 2>&1 | grep -E 'wall |hash '
	done
done
