#!/bin/bash
# ITEM 0: gate the 8.952 ms/token MTP number against the project gate, not
# against cross-rank agreement.
#
# Under greedy sampling MTP is supposed to be EXACT: a rejected draft falls
# back to the main model's argmax, so the emitted stream must equal the
# non-speculative stream token for token. f18dc60 only checked that the 8 ranks
# agreed with each other, which they can do on garbage.
#
# One variable: MPK_SPEC_DECODE (+ the two args the harness asserts on).
#   ctl : plain greedy, 1 decode row
#   mtp : MPK_SPEC_DECODE=1 --mtp 1 --max-num-batched-tokens 2
#
# compare_tokens.py's PASS is cross-rank + coherence, NOT exact match
# (agree@0/264 still PASSES), so the arm-vs-arm agree@ count is read out
# explicitly below and that is the number the gate turns on.
set -u
ulimit -c 0
cd /home/claudeuser/fleet-chiplet-megakernel/demo/glm5
D=/tmp/glm5_correctness
rm -f $D/mtpg_*

for ARM in ctl mtp; do
  if [ "$ARM" = ctl ]; then
    export MPK_SPEC_DECODE=0
    unset MPK_EXTRA_ARGS 2>/dev/null || true
    export MPK_EXTRA_ARGS=""
  else
    export MPK_SPEC_DECODE=1
    export MPK_EXTRA_ARGS="--mtp 1 --max-num-batched-tokens 2"
  fi
  export MASTER_PORT_BASE=$((31100 + (RANDOM % 150)))
  ./run_correctness_suite.sh run_mp8_dp_ep_fused.sh "mtpg_$ARM" \
      > "/tmp/mtpg_${ARM}_suite.log" 2>&1
  echo "=== $ARM suite rc=$?"
  echo -n "  SPEC_DECODE in run: "
  grep -ho "MPK_SPEC_DECODE=[01]" $D/mtpg_${ARM}_p*.log 2>/dev/null | sort -u | tr '\n' ' '
  echo
  grep -hoE "(Decode|decode)[^,]*: *[0-9.]+ ms[^,]*" $D/mtpg_${ARM}_p*.log 2>/dev/null | tail -8
  grep -hoE "accept[a-z]*[ =:]+[0-9.]+" $D/mtpg_${ARM}_p*.log 2>/dev/null | tail -4
done

echo "############ compare_tokens ctl vs mtp ############"
python3 compare_tokens.py $D mtpg_ctl mtpg_mtp 2>&1 | tail -40

echo "############ raw exact-prefix check (the real gate) ############"
python3 - <<'PY'
import json, glob, os
D = "/tmp/glm5_correctness"
for p in range(4):
    a = sorted(glob.glob(f"{D}/mtpg_ctl_p{p}_rank*.json"))
    b = sorted(glob.glob(f"{D}/mtpg_mtp_p{p}_rank*.json"))
    if not a or not b:
        print(f"p{p}: MISSING ctl={len(a)} mtp={len(b)}")
        continue
    ta = json.load(open(a[0]))["token_ids"]
    tb = json.load(open(b[0]))["token_ids"]
    n = min(len(ta), len(tb))
    k = 0
    while k < n and ta[k] == tb[k]:
        k += 1
    print(f"p{p}: len ctl={len(ta)} mtp={len(tb)} exact-prefix={k}/{n} "
          f"{'EXACT' if k == n else 'DIVERGES at '+str(k)}")
    if k < n:
        print(f"     ctl[{k}:{k+8}] = {ta[k:k+8]}")
        print(f"     mtp[{k}:{k+8}] = {tb[k:k+8]}")
PY
echo MTPGATE_DONE
