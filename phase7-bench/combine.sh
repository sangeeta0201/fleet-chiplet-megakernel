#!/bin/bash
# Assemble whatever bk_*.tsv runs exist into one table.
#
# Column order is fixed rather than glob order so the table reads
# NPS1 -> NPS2 unfixed -> NPS2 fixed, which is the direction the argument goes.
set -u
cd "$HOME" || exit 1

python3 - <<'PY'
import os, glob

order = ["nps1", "nps2_base", "nps2_fix"]
titles = {"nps1": "NPS1", "nps2_base": "NPS2 base", "nps2_fix": "NPS2 fixed"}
buckets = ["slicewait", "mfma", "bar", "rmsnorm_router", "topk", "total"]

data, modes, samples = {}, {}, {}
for label in order:
    path = os.path.expanduser("~/bk_%s.tsv" % label)
    if not os.path.exists(path):
        continue
    with open(path) as fh:
        head = fh.readline().lstrip("#").strip().split("\t")
        modes[label] = head[1] if len(head) > 1 else "?"
        samples[label] = head[2] if len(head) > 2 else "?"
        data[label] = {}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            data[label][f[0]] = (float(f[1]), float(f[2]), float(f[3]))

have = [l for l in order if l in data]
if not have:
    raise SystemExit("no bk_*.tsv found")

print()
for l in have:
    print("  %-10s  %-10s  %s" % (titles[l], modes[l], samples[l]))
print()
print("%-16s%s" % ("bucket", "".join("%12s" % titles[l] for l in have)))

for b in buckets:
    row = "%-16s" % b
    for l in have:
        row += "%12.2f" % data[l][b][0]
    if "nps1" in data and "nps2_fix" in data:
        row += "   %+7.2f vs NPS1" % (data["nps2_fix"][b][0] - data["nps1"][b][0])
    print(row)

if "nps1" in data and "nps2_fix" in data and "nps2_base" in data:
    n1 = data["nps1"]["total"][0]
    nb = data["nps2_base"]["total"][0]
    nf = data["nps2_fix"]["total"][0]
    print()
    print("  NPS2 base  %.2f us = %.2fx NPS1" % (nb, nb / n1))
    print("  NPS2 fixed %.2f us = %.2fx NPS1   (recovered %.2f of the %.2f us gap)"
          % (nf, nf / n1, nb - nf, nb - n1))
PY
