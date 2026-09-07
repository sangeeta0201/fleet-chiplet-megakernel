#!/bin/bash
# skew_sweep.sh - is `topk` slower because of NPS2, or because the barrier fixes
# stopped staggering it?
#
# `topk` reads worst in the *fastest* configuration: 1.92 us in NPS2 base, where
# the barrier is 6.28, and 2.20 with the barrier at 2.32. That is the signature of
# bottleneck relocation rather than a regression -- a barrier that releases with
# 6 us of spread trickles the 184 blocks into the next stage, and one that
# releases them together makes them contend.
#
# `--skew=N` re-injects exactly that stagger, N ticks per XCD, without changing
# anything else. If the mechanism is contention from synchronised arrival then
# `topk` falls as the skew rises, while `bar` climbs to absorb it. If `topk` is
# flat, the cause is elsewhere and this hypothesis is wrong.
set -u
cd "$HOME/fleet-chiplet-megakernel" || exit 1

LAYERS=${LAYERS:-300}
ARGS=${ARGS:---aid=1 --coherent=1 --split=1 --lsplit=1 --hrdv=1}

printf '%-8s %8s %8s %8s %8s %8s %8s\n' \
	skew_ns bar topk mfma slicewait rmsnorm total
for s in 0 10 20 40 80; do
	log=/tmp/sk_$s.log
	timeout 300 ./drive_phase7 --layers="$LAYERS" --tiles=23 $ARGS \
		--skew="$s" --tag="sk$s" >"$log" 2>&1 || {
		echo "  run failed at skew=$s"
		continue
	}
	SKEW=$s python3 - "$log" <<'PY'
import os, re, sys, statistics as st

pat = re.compile(r"(\w+)=([0-9.]+)")
need = ("bar", "topk", "mfma", "slicewait", "rmsnorm_router", "total")
rows = []
for line in open(sys.argv[1]):
    if "OPROJ_INNER" not in line:
        continue
    d = dict((k, float(v)) for k, v in pat.findall(line))
    if all(b in d for b in need):
        rows.append(d)
rows = rows[int(len(rows) * 0.30):]
m = lambda b: st.median([r[b] for r in rows])
print("%-8s %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f"
      % (int(os.environ["SKEW"]) * 10, m("bar"), m("topk"), m("mfma"),
         m("slicewait"), m("rmsnorm_router"), m("total")))
PY
done
