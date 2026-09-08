#!/bin/bash
# GPUDF.csv now exists, but the encodings came from a CPU-DF part, so they could
# be meaningless here. Validate the instrument the only way that counts: does it
# respond to real traffic? Compare an idle window against a window with Phase 7
# hammering memory. A counter that is identical in both is not measuring us.
set -u
G=$HOME/nps1/gpudf_ini
FL=$HOME/fleet-chiplet-megakernel

echo "=== what does the GPUDF CSV contain? ==="
csv=/tmp/me_g_a/csv/GPUDF.csv
echo "  rows: $(wc -l < "$csv")"
echo "  --- header ---"
head -3 "$csv" | cut -c1-220 | sed 's/^/    /'
echo "  --- CAKE metric rows (name + first few instances) ---"
grep -iE "cake|flit|Ipclk" "$csv" | head -25 | cut -c1-200 | sed 's/^/    /'

collect() {  # $1=label  $2=seconds  -> writes /tmp/gd_$1/csv/GPUDF.csv
	local OUT=/tmp/gd_$1; rm -rf "$OUT"; mkdir -p "$OUT"
	sudo -n /usr/bin/MultEvent -O "$OUT" -N "$1" -t "$2" \
		gpu gpu-df -i "$G/gpudf_cake_top.ini" -s 0-3 >/dev/null 2>&1
	sudo -n /usr/bin/MultEvent CLEAR >/dev/null 2>&1
}

echo
echo "=== A: idle 8 s ==="
collect idle 8
echo "  done: $(test -f /tmp/gd_idle/csv/GPUDF.csv && echo ok || echo MISSING)"

echo
echo "=== B: 8 s with Phase 7 running continuously ==="
cd "$FL" || exit 1
# 12000 layers keeps the GPU busy well past the 8 s window
( ./drive_phase7 --layers=12000 --tiles=23 --aid=0 --split=0 --lsplit=0 --hrdv=0 \
	--tag=load >/tmp/gd_load.log 2>&1 ) &
LOADPID=$!
sleep 1
collect busy 8
wait $LOADPID 2>/dev/null
grep -iE "wall|hash" /tmp/gd_load.log | head -3 | sed 's/^/  /'

echo
echo "=== compare: does any CAKE counter move idle -> busy? ==="
python3 - <<'PY'
import csv, os
def load(p):
    if not os.path.exists(p): return {}
    rows = list(csv.reader(open(p)))
    hdr = None
    for r in rows:
        if any(c.strip().startswith('GPUDF_') for c in r): hdr = r; break
    out = {}
    for r in rows:
        if not r or not r[0].strip(): continue
        name = r[0].strip()
        vals = []
        for c in r[1:]:
            c = c.strip()
            try: vals.append(float(c))
            except ValueError: pass
        if vals: out[name] = sum(vals)
    return out, hdr
i, hi = load('/tmp/gd_idle/csv/GPUDF.csv') or ({}, None)
b, hb = load('/tmp/gd_busy/csv/GPUDF.csv') or ({}, None)
if hb: print('  instances:', [c for c in hb if c.strip().startswith('GPUDF_')][:8])
keys = [k for k in b if 'cake' in k.lower() or 'flit' in k.lower()]
print(f'  {"metric":<44}{"idle":>16}{"busy":>16}   ratio')
moved = 0
for k in sorted(keys)[:28]:
    iv, bv = i.get(k, 0.0), b.get(k, 0.0)
    r = (bv/iv) if iv else float('inf') if bv else 0.0
    flag = '  <== MOVES' if (iv == 0 and bv > 0) or (iv and abs(r-1) > 0.25) else ''
    if flag: moved += 1
    print(f'  {k[:44]:<44}{iv:>16.1f}{bv:>16.1f}   {r:>7.2f}{flag}')
print(f'\n  {moved} of {len(keys[:28])} CAKE metrics responded to load')
print('  -> if 0 responded, the CPU-DF encodings do NOT map to GPU DF')
PY

