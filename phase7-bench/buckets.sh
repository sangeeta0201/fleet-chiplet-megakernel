#!/bin/bash
# buckets.sh <label> [harness args...]
#
# One Phase 7 bucket breakdown plus the correctness hash, from the same build in
# every mode. Writes bk_<label>.tsv for combine.sh to assemble, so the modes can
# be measured either side of a driver reload and still end up in one table.
#
# The hash matters as much as the medians here: it is what says the two modes
# computed the same thing, rather than one of them being fast because it was
# wrong.
set -u
cd "$HOME/fleet-chiplet-megakernel" || exit 1

LABEL=${1:?usage: buckets.sh <label> [harness args...]}
shift
LAYERS=${LAYERS:-400}
OUT="$HOME/bk_$LABEL.log"
TSV="$HOME/bk_$LABEL.tsv"

CP=$(cat /sys/bus/pci/devices/0000:05:00.0/current_compute_partition)
MP=$(cat /sys/bus/pci/devices/0000:05:00.0/current_memory_partition)
echo "=== $LABEL   mode $CP/$MP   $LAYERS layers   args: $* ==="

timeout 900 ./drive_phase7 --layers="$LAYERS" --tiles=23 "$@" --tag="$LABEL" \
	>"$OUT" 2>&1
rc=$?
echo "  rc=$rc samples=$(grep -c OPROJ_INNER "$OUT")"
if [ "$rc" != "0" ]; then
	echo "  ABORT: run failed"
	tail -5 "$OUT"
	exit 1
fi
grep -m1 'hash ' "$OUT"
grep -m1 'wall ' "$OUT"

python3 - "$OUT" "$LABEL" "$CP/$MP" "$TSV" <<'PY'
import re, sys, statistics as st

src, label, mode, tsv = sys.argv[1:5]
buckets = ["slicewait", "mfma", "bar", "rmsnorm_router", "topk", "total"]
pat = re.compile(r"(\w+)=([0-9.]+)")
rows = []
for line in open(src):
    if "[OPROJ_INNER]" not in line:
        continue
    d = dict((k, float(v)) for k, v in pat.findall(line))
    if all(b in d for b in buckets):
        rows.append(d)

rows = rows[int(len(rows) * 0.30):]  # warmup: slicewait elevated while caches fill
with open(tsv, "w") as fh:
    fh.write("# %s\t%s\t%d samples\n" % (label, mode, len(rows)))
    for b in buckets:
        v = sorted(r[b] for r in rows)
        n = len(v)
        fh.write("%s\t%.2f\t%.2f\t%.2f\n" % (
            b, st.median(v), v[int(0.10 * n)], v[min(int(0.90 * n), n - 1)]))
print("  %d samples -> %s" % (len(rows), tsv))
for b in buckets:
    v = sorted(r[b] for r in rows)
    print("    %-16s %6.2f" % (b, st.median(v)))
PY
