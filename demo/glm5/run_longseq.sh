#!/bin/bash
# Long-sequence decode sweep. The length-dependent collapse (PERPLEXITY.md)
# starts when seq > 256 (16 KV chunks x 16-token tiles). Short G2 prompts
# never hit it. After the MLA tile-loop fix, these are the shapes that would
# have emitted the "0. " loop.
#
#   ./run_longseq.sh
#   SHAPES="256/256 512/512 1024/1024" ./run_longseq.sh
#
# Compiles once at the max ISL+OSL (default 2048) and reuses the build.
# NP=4, devices 4-7. --ignore-eos so OSL is the requested length.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

export MODEL_PATH="${MODEL_PATH:-${GLM_MODEL_PATH:-/mnt/nvme1/GLM-5.2-MXFP4}}"
SHAPES="${SHAPES:-256/256 512/512 1024/1024}"
OUT_DIR="${OUT_DIR:-/tmp/glm5_longseq}"
mkdir -p "$OUT_DIR"

max_seq=0
for s in $SHAPES; do
  isl=${s%%/*}; osl=${s##*/}
  tot=$((isl + osl))
  [ "$tot" -gt "$max_seq" ] && max_seq=$tot
done
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-$max_seq}"

# shellcheck source=stall_watchdog.sh
. ./stall_watchdog.sh
STALL_SECS="${STALL_SECS:-400}"
RETRIES="${RETRIES:-8}"
ulimit -c 0

echo "=== GLM-5.2 longseq compile seq=$MAX_SEQ_LENGTH shapes=$SHAPES ==="
if [ "${KEEP_BUILD:-0}" != "1" ]; then
  rm -rf permanent_output_dir permanent_output_dir_rank*
fi
export KEEP_BUILD=1
fail=0

for s in $SHAPES; do
  isl=${s%%/*}; osl=${s##*/}
  if [ $((isl + osl)) -gt "$MAX_SEQ_LENGTH" ]; then
    echo "SKIP $s: ISL+OSL > MAX_SEQ_LENGTH=$MAX_SEQ_LENGTH"
    continue
  fi
  export MAX_NEW_TOKENS=$osl
  export MAX_SAVE_TOKENS=$osl
  dst="$OUT_DIR/isl${isl}_osl${osl}"
  rc=1
  for attempt in $(seq 0 "$RETRIES"); do
    export MASTER_PORT=$(( 30100 + (RANDOM % 400) ))
    log="${dst}.log"
    run_with_watchdog "$log" "$STALL_SECS" \
      ./run_mp8_dp_ep_fused.sh \
        --ignore-eos --prompt-tokens "$isl" --save-tokens "${dst}.json"
    rc=$?
    if [ "$rc" -eq 0 ] && ls "${dst}"_rank*.json >/dev/null 2>&1; then
      break
    fi
    echo "[longseq] $s attempt $((attempt + 1)) rc=$rc, retrying"
  done
  python3 - "$dst" "$isl" "$osl" "$rc" <<'PY'
import json, glob, os, re, sys, collections
dst, isl, osl, rc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
dumps = sorted(glob.glob(dst + "_rank*.json"))
print(f"=== {isl}/{osl} rc={rc} dumps={len(dumps)} ===")
if not dumps:
    sys.exit(1)
e = json.load(open(dumps[0]))
ids = e["token_ids"]
text = e.get("text") or ""
n = len(ids)
dist = (len(set(ids)) / n) if n else 0.0
head, tail = ids[:128], ids[-128:]
dhead = (len(set(head)) / len(head)) if head else 0.0
g1 = all(json.load(open(p))["token_ids"] == ids for p in dumps)
dec_avg = dec_min = None
for line in open(dst + ".log", errors="replace"):
    m = re.search(r"Decode:\s+(\d+) tokens .* avg ([0-9.]+)ms/iter", line)
    if m:
        dec_avg = float(m.group(2))
    m = re.search(r"Decode per-iter range: min=([0-9.]+)ms", line)
    if m:
        dec_min = float(m.group(1))
print(f"G1={'PASS' if g1 else 'FAIL'} dumped={n} distinct={dist:.3f} first128_distinct={dhead:.3f}")
print(f"decode_avg_ms={dec_avg} decode_min_ms={dec_min}")
print("HEAD:", text[:200].replace("\n", " | "))
print("TAIL:", text[-200:].replace("\n", " | "))
# Collapse signature from the 1k/1k bug: token 15 ('0') dominates.
# An ignore-eos year/list n-gram is a different (host-stop) issue.
top = collections.Counter(ids).most_common(3)
print("top3", top)
zero_n = collections.Counter(ids).get(15, 0)
if ids and top[0][0] == 15 and top[0][1] >= max(8, n // 4):
    print("FAIL: generation dominated by token 15 (0)")
    sys.exit(2)
if dhead < 0.02:
    print("FAIL: first-128 generation is collapsed")
    sys.exit(2)
if zero_n >= max(8, n // 4):
    print("FAIL: token 15 (0) is too frequent")
    sys.exit(2)
PY
  [ $? -ne 0 ] && fail=1
done
exit $fail
