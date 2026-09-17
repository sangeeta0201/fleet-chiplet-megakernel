#!/bin/bash
# MPK_WAIT_SKIP: remove the remaining per-layer WAITS while keeping every
# RELEASE, exactly like MPK_P7_SKIP. Nothing can deadlock, results are garbage
# by design, and only the timing is meaningful.
#
# Control minus all-waits-removed = the synchronisation cost. That is the
# number the zero-op build kept failing to produce, and it needs no fake
# publishers because no release is ever removed.
#
#   bit 1  qkv_epoch waits        (lines 798, 1201)   -> slot 2
#   bit 2  qkv split_flag wait    (line 914)
#   bit 4  attn_release wait      (line 1393)         -> slot 5
# Combine with the knobs that already exist:
#   MPK_NO_LAYER_BARRIER=1       layer_release wait   -> slots 0/11
#   MPK_P7_SKIP=7                op-7's three waits   -> slots 6/7
set -u
cd "$HOME/fleet-chiplet-megakernel" || exit 1
F=include/mirage/persistent_kernel/tasks/mi300/gang_full_layer_fused_mi300.cuh
P=python/mirage/mpk/persistent_kernel.py
cp -n "$F" "$F.pre-waitskip" 2>/dev/null
cp -n "$P" "$P.pre-waitskip" 2>/dev/null

python3 - "$F" <<'PY'
import sys
p = sys.argv[1]
s = open(p, encoding='utf-8', errors='surrogateescape').read()
if 'MPK_WAIT_SKIP' in s:
    print('  kernel: already patched'); sys.exit(0)

# macro, mirroring MPK_P7_SK
anchor = '#pragma once'
i = s.find(anchor)
if i < 0:
    print('  FAIL: no #pragma once'); sys.exit(1)
j = s.find('\n', i) + 1
mac = '''
// MPK_WAIT_SKIP: same bisection discipline as MPK_P7_SKIP -- remove one class
// of WAIT at a time, never a release, so nothing can deadlock. Results are
// garbage by design; the timing delta against the control is the cost of the
// waits that were removed.
#ifdef MPK_WAIT_SKIP
#define MPK_WS_SK(bit) (((MPK_WAIT_SKIP) & (bit)) != 0)
#else
#define MPK_WS_SK(bit) false
#endif

'''
s = s[:j] + mac + s[j:]

subs = [
    # (old, new, label)
    ('      while ((_obs = __atomic_load_n(&qkv_epoch[xcd_id * 16],',
     '      while (!MPK_WS_SK(1) && (_obs = __atomic_load_n(&qkv_epoch[xcd_id * 16],',
     'qkv_epoch 798'),
    ('          while (MPK_LD_GATE(split_flag) < qkv_epoch_expected) {',
     '          while (!MPK_WS_SK(2) && MPK_LD_GATE(split_flag) < qkv_epoch_expected) {',
     'split_flag 914'),
    ('        while (MPK_LD_GATE(&qkv_epoch[xcd_id * 16]) < qkv_epoch_expected) {',
     '        while (!MPK_WS_SK(1) && MPK_LD_GATE(&qkv_epoch[xcd_id * 16]) < qkv_epoch_expected) {',
     'qkv_epoch 1201'),
    ('      while ((_obs = MPK_LD_GATE(&attn_release[xcd_id * 16])) <',
     '      while (!MPK_WS_SK(4) && (_obs = MPK_LD_GATE(&attn_release[xcd_id * 16])) <',
     'attn_release 1393'),
]
for old, new, label in subs:
    n = s.count(old)
    if n != 1:
        print('  WARN: %s matched %d times; skipped' % (label, n))
        continue
    s = s.replace(old, new, 1)
    print('  patched %s' % label)

open(p, 'w', encoding='utf-8', errors='surrogateescape').write(s)
PY

python3 - "$P" <<'PY'
import sys, re
p = sys.argv[1]
s = open(p, encoding='utf-8', errors='surrogateescape').read()
if 'MPK_WAIT_SKIP' in s:
    print('  py: already registered'); sys.exit(0)
m = re.search(r'^(\s*)_p7skip = int\(os\.environ\.get\("MPK_P7_SKIP", "0"\)\)', s, re.M)
if m is None:
    print('  py: MPK_P7_SKIP anchor not found'); sys.exit(1)
ind = m.group(1)
add = (m.group(0) + '\n'
       + ind + '_wskip = int(os.environ.get("MPK_WAIT_SKIP", "0"))\n'
       + ind + 'if _wskip != 0:\n'
       + ind + '    flags = flags + ["-DMPK_WAIT_SKIP=%d" % _wskip]')
s = s.replace(m.group(0), add, 1)
open(p, 'w', encoding='utf-8', errors='surrogateescape').write(s)
print('  py: MPK_WAIT_SKIP registered')
PY

echo
echo "=== verify ==="
grep -n "MPK_WS_SK(" "$F" | head -8 | sed 's/^/  /'
grep -n "MPK_WAIT_SKIP" "$P" | head -4 | sed 's/^/  /'

