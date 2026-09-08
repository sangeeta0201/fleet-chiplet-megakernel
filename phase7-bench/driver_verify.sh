#!/bin/bash
# Gate the new driver before measuring anything with it. The bar for "no page
# type errors" is: reference hash intact, all 8 XCDs dispatch, no VM fault /
# PERMISSION_FAULTS in dmesg, and the PERBO line proves an AID_LOCAL BO really
# resolved to mtype_local rather than being demoted to NC.
set -u
MARK=$(dmesg 2>/dev/null | wc -l)   # only look at lines this run produces

echo "=== default path, hash gate ==="
bash "$HOME/nps1/final_verify.sh" 2>&1 | sed 's/^/  /'

echo
echo "=== did an AID_LOCAL BO take mtype_local? (the point of the patch) ==="
dmesg 2>/dev/null | tail -n +"$MARK" | grep -iE "PERBO|FLAGMTYPE" |
	sort -u | sed 's/^/  /'
dmesg 2>/dev/null | grep -c "PERBO: AID_LOCAL" |
	awk '{print "  PERBO lines in ring buffer: " $1}'

echo
echo "=== page-type / fault errors introduced by this run ==="
f=$(dmesg 2>/dev/null | tail -n +"$MARK" |
	grep -icE "VM_L2_PROTECTION_FAULT|PERMISSION_FAULTS|page fault|VMC page|no-retry|GPU fault")
echo "  fault-ish lines: $f"
[ "$f" -gt 0 ] && dmesg 2>/dev/null | tail -n +"$MARK" |
	grep -iE "VM_L2_PROTECTION_FAULT|PERMISSION_FAULTS|page fault|no-retry" |
	tail -8 | sed 's/^/    /'

echo
echo "=== all 8 XCDs dispatching? ==="
grep -oE "xcd_seen[^ ]*" "$HOME/nps1/fin1.log" 2>/dev/null | head -2 | sed 's/^/  /'
grep -iE "xcd_seen|XCDs seen|only .* XCD" "$HOME/nps1/fin1.log" 2>/dev/null |
	head -3 | sed 's/^/  /'
echo "  (harness aborts on a short dispatch, so a correct hash already implies 8)"

