#!/bin/bash
# breakdown_all.sh - the current Phase 7 bucket breakdown, four configurations,
# one build, both partition modes.
#
# combine.sh only knows the three columns the original argument needed. This adds
# the tree rendezvous as a fourth and reports every column against NPS1, which is
# the only comparison that says how much of the partitioning cost is left.
#
# NPS2 is restored unconditionally, including on failure: leaving the node in NPS1
# would silently change every later measurement.
set -u
cd "$HOME" || exit 1

LAYERS=${LAYERS:-400}
export LAYERS

restore() {
	echo ""
	echo "########## restoring NPS2 ##########"
	sudo -n python3 run_root.py set_mode2.sh NPS2 2>&1 | tail -2
	cat /sys/bus/pci/devices/0000:05:00.0/current_memory_partition
}
trap restore EXIT

echo "########## NPS2: base, flags split, flags split + tree rendezvous ##########"
bash buckets.sh nps2_base --aid=0
bash buckets.sh nps2_fix --aid=1 --coherent=1 --split=1 --lsplit=1
bash buckets.sh nps2_tree --aid=1 --coherent=1 --split=1 --lsplit=1 --hrdv=1

echo ""
echo "########## switching to NPS1 ##########"
sudo -n python3 run_root.py set_mode2.sh NPS1 2>&1 | tail -2
MP=$(cat /sys/bus/pci/devices/0000:05:00.0/current_memory_partition)
if [ "$MP" != "NPS1" ]; then
	echo "ABORT: mode is $MP, not NPS1"
	exit 1
fi

# --aid=0 is forced in NPS1: AID-local placement only exists when there are two
# memory partitions to place into. The tree still runs there, with both region
# pointers aliased, which is the shape-without-placement control.
bash buckets.sh nps1 --aid=0
bash buckets.sh nps1_tree --aid=0 --hrdv=1

python3 - <<'PY'
import os

order = ["nps1", "nps1_tree", "nps2_base", "nps2_fix", "nps2_tree"]
titles = {"nps1": "NPS1", "nps1_tree": "NPS1 tree", "nps2_base": "NPS2 base",
          "nps2_fix": "NPS2 fix", "nps2_tree": "NPS2 tree"}
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
hdr = "%-16s" % "bucket" + "".join("%11s" % titles[l] for l in have)
if "nps1" in data and "nps2_tree" in data:
    hdr += "%14s" % "tree vs NPS1"
print(hdr)
print("-" * len(hdr))
for b in buckets:
    row = "%-16s" % b + "".join("%11.2f" % data[l][b][0] for l in have)
    if "nps1" in data and "nps2_tree" in data:
        row += "%+14.2f" % (data["nps2_tree"][b][0] - data["nps1"][b][0])
    print(row)

if "nps1" in data and "nps2_base" in data:
    n1 = data["nps1"]["total"][0]
    print()
    for l in have:
        if l == "nps1":
            continue
        t = data[l]["total"][0]
        print("  %-10s %6.2f us = %.2fx NPS1" % (titles[l], t, t / n1))
    if "nps2_tree" in data:
        nb = data["nps2_base"]["total"][0]
        nt = data["nps2_tree"]["total"][0]
        print()
        print("  recovered %.2f of the %.2f us NPS2 penalty (%.0f%%), %.2f us left"
              % (nb - nt, nb - n1, 100.0 * (nb - nt) / (nb - n1), nt - n1))
PY
