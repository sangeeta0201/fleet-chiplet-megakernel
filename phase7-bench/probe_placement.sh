#!/bin/bash
# Measure, per buffer, which XCDs are local to it. The harness's probe does a
# dependent chase in two modes: sc0/sc1 bypasses L2 and reports where the pages
# physically are; nt reports what a cached reader sees.
#
# The question: after --dsplit=3, which buffers are STILL single-homed? The
# source names three that cannot be cached at all -- out (st_wt_u64, acquired
# cross-XCD), counters (atomicAdd from every block), logits (st_wt_u16) -- so
# for those, half the XCDs pay the ~94 ns crossing on every single access and no
# amount of L2 residency hides it. Those are the remaining remote accesses.
set -u
cd "$HOME/fleet-chiplet-megakernel" || exit 1
grep -nE '"--probe' drive_phase7.cu | sed 's/^/  flag: /'

for arm in "unplaced:" "placed:--dsplit=3"; do
	lbl=${arm%%:*}; fl=${arm#*:}
	echo
	echo "=== $lbl ${fl:-(no data placement)} ==="
	timeout 240 ./drive_phase7 --layers=8 --tiles=23 --aid=1 --coherent=1 \
		--split=1 --lsplit=1 --hrdv=1 --probe=1 $fl --tag=pp 2>&1 |
		grep -E "PROBE|skew|local|remote" | sed 's/^/  /'
done

