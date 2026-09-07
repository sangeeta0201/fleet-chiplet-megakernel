#!/bin/bash
# trace_both_modes.sh — the same in-barrier decomposition in both partition
# modes, so the +0.72 us residual in `bar` can be charged to a specific step.
#
# NPS1 has to be measured with --aid=0: AID-local allocation only exists when
# there are two memory partitions to place into. That is the same pairing
# results/nps1-vs-nps2.md uses, so the totals stay comparable to the recorded
# table.
#
# NPS2 is restored unconditionally at the end, including on failure -- leaving
# the node in NPS1 would silently change every later measurement.
set -u
cd "$HOME" || exit 1

restore() {
	echo ""
	echo "########## restoring NPS2 ##########"
	sudo -n python3 run_root.py set_mode2.sh NPS2 2>&1 | tail -4
	cat /sys/bus/pci/devices/0000:05:00.0/current_compute_partition \
		/sys/bus/pci/devices/0000:05:00.0/current_memory_partition
}
trap restore EXIT

echo "########## NPS2 base (stock flags, one copy) ##########"
bash trace_bar.sh nps2base --aid=0

echo ""
echo "########## switching to NPS1 ##########"
sudo -n python3 run_root.py set_mode2.sh NPS1 2>&1 | tail -4
MP=$(cat /sys/bus/pci/devices/0000:05:00.0/current_memory_partition)
if [ "$MP" != "NPS1" ]; then
	echo "ABORT: mode is $MP, not NPS1"
	exit 1
fi

echo ""
echo "########## NPS1 ##########"
bash trace_bar.sh nps1 --aid=0
