#!/bin/bash
# Tear down a wedged drive_phase7 and anything still driving it.
#
# `pkill -f drive_phase7` does not work here: the pattern matches the ssh
# `bash -c` wrapper running it, so pkill kills its own parent before reaching
# the target. Match the executable name exactly instead, and use a regex the
# script's own command line cannot satisfy for the loop.
set -u

for p in $(pgrep -f 'gate[0-9][.]sh'); do
	echo "killing driver $p"
	kill -9 "$p" 2>/dev/null
done
for p in $(pgrep -x drive_phase7) $(pgrep -x drive_phase7_nopoll) $(pgrep -x drive_phase7_noslice); do
	echo "killing kernel $p"
	kill -9 "$p" 2>/dev/null
done

sleep 6
echo "--- remaining ---"
pgrep -a -x drive_phase7 || echo "none"
echo "--- gpu busy ---"
timeout 30 rocm-smi --showuse 2>/dev/null | grep -E 'GPU\[[0-7]\]'
