#!/bin/bash
# Compare every probed buffer across the three breakdown runs, not just the two
# that go into the hash.
#
# The hash covers attn_proj_out and rmsnorm_out. Phase 7 also publishes the
# TopK outputs, and "the output is the same" should mean all of them, so this
# pulls the per-buffer probe lines out of the logs already on disk and diffs
# them. The run tag is stripped first, since that differs by construction.
set -u
cd "$HOME" || exit 1

extract() {
	sed -e 's/^\[[^]]*\] //' "$1" |
		grep -E '^(hash |[a-z_]+\(?[a-z]*\)? +[0-9]+/[0-9]+ nonzero)' |
		sort
}

for f in bk_nps1 bk_nps2_base bk_nps2_fix; do
	[ -f "$f.log" ] || { echo "missing $f.log"; exit 1; }
	extract "$f.log" >"/tmp/$f.probe"
done

echo "=== NPS1 probe (reference) ==="
cat /tmp/bk_nps1.probe

for f in bk_nps2_base bk_nps2_fix; do
	echo
	echo "=== $f vs NPS1 ==="
	if diff -u /tmp/bk_nps1.probe "/tmp/$f.probe" >/tmp/d.txt; then
		echo "IDENTICAL on every probed buffer"
	else
		cat /tmp/d.txt
	fi
done
