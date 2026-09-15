#!/bin/bash
# Fast liveness reproducer for the K-split ceiling probe.
#
#   ARM=ceil2 ./run_ksplit_smoke.sh
#
# ISL/OSL are tiny on purpose. The 1024/1024 protocol spends 314 s in prefill
# before it can tell you whether the kernel is alive, which makes a deadlock
# cost 20 minutes to observe. At 32/16 the same deadlock shows in ~2 minutes.
# This is a LIVENESS and TEXT check only -- never quote latency from it
# (short prompts are a correctness gate, not a board number).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

ARM="${ARM:-control}"
ISL="${ISL:-32}"
OSL="${OSL:-16}"
RESULTS="${RESULTS:-/tmp/ksplit_smoke}"
mkdir -p "$RESULTS"
DST="$RESULTS/${ARM}_isl${ISL}_osl${OSL}"

ulimit -c 0

case "$ARM" in
  control) unset MPK_OPROJ_KSPLIT_CEIL ;;
  ceil1)   export MPK_OPROJ_KSPLIT_CEIL=1 ;;
  ceil2)   export MPK_OPROJ_KSPLIT_CEIL=2 ;;
  *) echo "unknown ARM=$ARM"; exit 2 ;;
esac

export MODEL_PATH="${MODEL_PATH:-/mnt/nvme1/GLM-5.2-MXFP4}"
export MAX_SEQ_LENGTH=$((ISL + OSL))
export MAX_NEW_TOKENS="$OSL"
export MAX_SAVE_TOKENS="$OSL"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-4,5,6,7}"
export NP="${NP:-4}"
export MASTER_PORT=$(( 30600 + (RANDOM % 300) ))

. ./stall_watchdog.sh
STALL_SECS="${STALL_SECS:-240}"

if [ "${KEEP_BUILD:-0}" != "1" ]; then
  rm -rf permanent_output_dir permanent_output_dir_rank*
fi

echo "########## SMOKE ARM=$ARM ISL=$ISL OSL=$OSL CEIL=${MPK_OPROJ_KSPLIT_CEIL:-unset} ##########"
date -u +%H:%M:%S
run_with_watchdog "${DST}.log" "$STALL_SECS" \
  ./run_mp8_dp_ep_fused.sh --ignore-eos --prompt-tokens "$ISL" \
    --save-tokens "${DST}.json"
rc=$?
date -u +%H:%M:%S
echo "rc=$rc"

grep -E "Decode: [0-9]+ tokens|Decode per-iter|watchdog" "${DST}.log" | tail -4
python3 - "$DST" <<'PY'
import glob, json, sys
dst = sys.argv[1]
ds = sorted(glob.glob(dst + "_rank*.json")) or glob.glob(dst + ".json")
if not ds:
    print("NO DUMP (hang or crash)")
    raise SystemExit(1)
es = [json.load(open(p)) for p in ds]
print("G1_cross_rank=" + ("PASS" if all(e["token_ids"] == es[0]["token_ids"]
                                        for e in es) else "FAIL"))
print("ntok=%d  text=%r" % (len(es[0]["token_ids"]),
                            (es[0].get("text") or "")[:160]))
PY

pkill -9 -f '[d]emo.py' 2>/dev/null
pkill -9 -f '[m]pirun -np' 2>/dev/null
sleep 5
exit "$rc"
