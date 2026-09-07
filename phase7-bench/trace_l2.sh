#!/bin/bash
# trace_l2.sh - did the level-2 split actually make the level-2 atomic cheaper?
#
# The A/B on `bar` says the split is a net loss, but that is the sum of two
# opposing effects. This separates them: `l2` is the arrival atomic on its own,
# which should fall if the premise was right (a cacheable AID-local line instead
# of the MTYPE_NC one in `counters`), while the handshake it pays for lands in
# `wait`. If `l2` does not fall, the premise itself was wrong.
set -u
cd "$HOME" || exit 1
for b in 0 1; do
	echo "##### --bsplit=$b #####"
	bash trace_bar.sh "l2$b" --aid=1 --coherent=1 --split=1 --lsplit=1 \
		--hrdv=1 --bsplit="$b" 2>&1 |
		grep -E 'level 2:|level 1:|^  all |xcd *drain|Releaser|entry spread|arrival spread'
done
