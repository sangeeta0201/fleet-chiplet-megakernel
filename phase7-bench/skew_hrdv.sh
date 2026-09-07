#!/bin/bash
# skew_hrdv.sh - does the hierarchical rendezvous close the entry stagger?
#
# `bar` alone cannot answer that. The entry columns can: they measure where the
# eight XCDs are when Phase 7 starts, before its barrier has done anything, so a
# change there is attributable to the rendezvous and nothing else.
set -u
cd "$HOME" || exit 1
for a in 0 1; do
	echo "===== --hrdv=$a ====="
	bash trace_bar.sh "hrdv$a" --aid=1 --coherent=1 --split=1 --lsplit=1 \
		--hrdv="$a" 2>&1 |
		grep -E 'entry spread|arrival spread|created inside|Releaser|^  [0-7] {6,}|xcd  '
done
