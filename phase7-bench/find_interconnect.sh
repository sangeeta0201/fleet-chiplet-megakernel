#!/bin/bash
# Find a usable interconnect (die-to-die) traffic counter.
#
# TCC_EA0_RDREQ_GMI_32B is a dead end: it reads 0 only because there are no 32 B
# requests at all (TCC_EA0_RDREQ_32B = 0 too); every read is 64/128 B. What we
# know so far is 86,210 L2 fills of which 70,019 reached DRAM, leaving 16,191
# (18.8%) going somewhere else -- GMI or IO -- that the 32B-only counters cannot
# separate.
#
# The Mosaic paper says "we measure inter-die traffic directly", so their tree
# should name the counter. Check that first, then the full counter list.
set -u

echo "############ 1. how does Mosaic measure inter-die traffic? ############"
for d in "$HOME/mosaic" "$HOME/aid-local-hbm"; do
	[ -d "$d" ] || continue
	echo "  --- $d ---"
	grep -rniE "inter.?die|GMI|xgmi|remote_traffic|numa_traffic" "$d" \
		--include=*.py --include=*.sh --include=*.cpp --include=*.hpp \
		--include=*.hip --include=*.md --include=*.txt --include=*.json 2>/dev/null |
		grep -viE "algorithm|gmix|gmime" | head -20 | sed 's/^/    /'
done

echo
echo "############ 2. every counter mentioning an interconnect ############"
rocprofv3 --list-avail 2>/dev/null > /tmp/avail.txt
echo "  total lines in --list-avail: $(wc -l < /tmp/avail.txt)"
echo "  --- names matching GMI / XGMI / REMOTE / LINK / SOCKET / DF / IO ---"
grep -oE "\b[A-Z][A-Z0-9_]{3,}\b" /tmp/avail.txt | sort -u |
	grep -iE "GMI|REMOTE|LINK|SOCKET|_DF_|^DF_|_IO_|FABRIC|NUMA|MALL|LLC" |
	head -60 | sed 's/^/    /'

echo
echo "############ 3. all TCC_EA* counters (is there an EA1..n?) ############"
grep -oE "\bTCC_EA[0-9]*_[A-Z0-9_]+\b" /tmp/avail.txt | sort -u | sed 's/^/    /'

echo
echo "############ 4. anything with RDREQ that is not 32B-qualified ############"
grep -oE "\bTCC_[A-Z0-9_]*RDREQ[A-Z0-9_]*\b" /tmp/avail.txt | sort -u | sed 's/^/    /'

