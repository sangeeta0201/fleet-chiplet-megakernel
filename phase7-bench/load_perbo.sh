#!/bin/bash
# Load the per-BO MTYPE module, then prove what it actually did.
#
# Derived from load_patched.sh by substitution, so every guard in it is
# preserved verbatim: aborts if /dev/kfd is held, and falls back to stock
# modprobe if insmod fails so the node is never left without a driver.
#
# Knobs are deliberately IDENTICAL to what is loaded now (xcp_nc=1,
# flag_mtype=2). The only difference in the whole system is that AID_LOCAL BOs
# are now exempt from the is_local override. That keeps this a single-variable
# change: plain VRAM still NC (so dispatch stays correct and nothing faults),
# coherent AID_LOCAL still CC via FLAGMTYPE (so the Phase 7 sync buffer is on
# the exact path it measures today), and only plain AID_LOCAL BOs move NC -> RW.
set -u
cd "$HOME/nps1" || exit 1

sed 's|SRC=$HOME/aid-local-hbm/src/amdgpu-mtype-test|SRC=$HOME/aid-local-hbm/src/amdgpu-perbo|' \
	load_patched.sh > load_perbo_gen.sh
grep -n "^SRC=" load_perbo_gen.sh | sed 's/^/  /'
grep -q "amdgpu-perbo" load_perbo_gen.sh || { echo "ABORT: substitution failed"; exit 1; }

bash load_perbo_gen.sh 1 2 2>&1 | tail -40

echo
echo "=== did the per-BO path report itself? ==="
dmesg 2>/dev/null | grep -iE "PERBO|FLAGMTYPE|MTYPE_|Using MTYPE" | tail -10 | sed 's/^/  /'
echo
echo "=== module identity (should be the perbo build) ==="
modinfo amdgpu 2>/dev/null | grep -iE "^(filename|srcversion)" | sed 's/^/  /'

