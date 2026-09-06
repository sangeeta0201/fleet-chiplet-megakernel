#!/bin/bash
# Latency protocol: 1024 ISL / 1024 OSL. Always this, never a short prompt.
#
# Short G2 prompts (~20 tokens, ~200 decode) are a correctness gate, not a
# board number. Quote TPOT only from this script, and only if the dump has
# generated text (CLAUDE.md). --ignore-eos so EOS/n-gram halt does not cut
# OSL short.
#
#   ./run_latency_1k1k.sh
#   ISL=1024 OSL=1024 ./run_latency_1k1k.sh
#
# NP=4, HIP_VISIBLE_DEVICES=4,5,6,7. DSA identity holds at kv_len <= 2048.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

ISL="${ISL:-1024}"
OSL="${OSL:-1024}"
export MAX_SEQ_LENGTH=$((ISL + OSL))
export MAX_NEW_TOKENS="$OSL"
export MAX_SAVE_TOKENS="$OSL"
export MODEL_PATH="${MODEL_PATH:-${GLM_MODEL_PATH:-/mnt/nvme1/GLM-5.2-MXFP4}}"

OUT_DIR="${OUT_DIR:-/tmp/glm5_latency_1k1k}"
mkdir -p "$OUT_DIR"
DST="$OUT_DIR/isl${ISL}_osl${OSL}"

# shellcheck source=stall_watchdog.sh
. ./stall_watchdog.sh
# After ENTER the whole 1k prefill + 1k decode is one kernel with FWD_PASS
# captured off fd 1, so the log is silent until mpk() returns. 400s covers
# ~200 ms/iter; a cold-start wedge is still killed.
STALL_SECS="${STALL_SECS:-400}"
RETRIES="${RETRIES:-8}"

ulimit -c 0

echo "=== GLM-5.2 latency ${ISL}/${OSL} ISL/OSL seq=$MAX_SEQ_LENGTH ==="

if [ "${KEEP_BUILD:-0}" != "1" ]; then
  rm -rf permanent_output_dir permanent_output_dir_rank*
fi
export KEEP_BUILD=1

rc=1
for attempt in $(seq 0 "$RETRIES"); do
  export MASTER_PORT=$(( 30100 + (RANDOM % 400) ))
  log="${DST}.log"
  run_with_watchdog "$log" "$STALL_SECS" \
    ./run_mp8_dp_ep_fused.sh \
      --ignore-eos \
      --prompt-tokens "$ISL" \
      --save-tokens "${DST}.json"
  rc=$?
  if [ "$rc" -eq 0 ]; then
    # Launcher used to echo MPIRUN_EXIT and then exit 0. It now propagates
    # mpirun's status, but still require the dump -- a wedge can look quiet.
    if ls "${DST}"_rank*.json >/dev/null 2>&1 || [ -f "${DST}.json" ]; then
      break
    fi
    echo "[1k1k] rc=0 but no dump, retrying" | tee -a "$log"
    rc=1
  fi
  echo "[1k1k] attempt $((attempt + 1)) rc=$rc, retrying"
done

if [ "$rc" -ne 0 ]; then
  echo "[1k1k] FAILED rc=$rc  log=$log"
  exit "$rc"
fi

python3 - <<PY
import glob, json, os, re, sys
dst = "${DST}"
log = dst + ".log"
isl, osl = ${ISL}, ${OSL}
dumps = sorted(glob.glob(dst + "_rank*.json")) or (
    [dst + ".json"] if os.path.exists(dst + ".json") else [])
if not dumps:
    sys.exit("no token dump")
entries = [json.load(open(p)) for p in dumps]
n = [len(e["token_ids"]) for e in entries]
text = entries[0].get("text") or ""
ids = entries[0]["token_ids"]
dist = (len(set(ids)) / len(ids)) if ids else 0.0
# G1: all ranks identical
g1 = all(e["token_ids"] == entries[0]["token_ids"] for e in entries)
decode_avg = decode_min = None
prefill_avg = None
gen_tok = None
for line in open(log, errors="replace"):
    m = re.search(r"Decode:\s+(\d+) tokens .* avg ([0-9.]+)ms/iter", line)
    if m:
        gen_tok, decode_avg = int(m.group(1)), float(m.group(2))
    m = re.search(r"Decode per-iter range: min=([0-9.]+)ms", line)
    if m:
        decode_min = float(m.group(1))
    m = re.search(r"Prefill: .* avg ([0-9.]+)ms/iter", line)
    if m:
        prefill_avg = float(m.group(1))
print(f"ISL={isl} OSL={osl} dumped_osl={n[0]} ranks={len(entries)}")
print(f"G1_cross_rank={'PASS' if g1 else 'FAIL'}")
print(f"G2_distinct={dist:.3f}  dumped_text_chars={len(text)}")
print(f"prefill_avg_ms={prefill_avg}  decode_avg_ms={decode_avg}  decode_min_ms={decode_min}  decode_tokens={gen_tok}")
print("TEXT_HEAD:", text[:240].replace("\n", " | "))
print("TEXT_TAIL:", text[-240:].replace("\n", " | "))
if n[0] != osl:
    print(f"WARNING: dumped OSL {n[0]} != {osl}; do not quote as 1k/1k")
    sys.exit(2)
if not text.strip():
    print("WARNING: empty text; do not quote latency")
    sys.exit(2)
print(f"DUMP={dumps[0]}")
print(f"LOG={log}")
PY
