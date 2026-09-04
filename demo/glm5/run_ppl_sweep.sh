#!/usr/bin/env bash
# Perplexity vs sequence length for GLM-5 744B.
#
# The GLM-5 counterpart of tests/ci-tests/run_gpt_oss_ppl_sweep.sh, and it
# exists for the fixed-prefix column that summarize_glm5_ppl_sweep.py computes.
# A single length in isolation cannot tell "the model got worse" from "the text
# got harder": measured 5.70 at 128 tokens and 15.64 at 256, and without a
# reference arm to move in parallel there is nothing to attribute that to. Every
# run re-scored over the positions all runs share separates the two.
#
# Two differences from the GPT-OSS sweep, both forced:
#
#   * No Torch arm at any length. GLM-5 744B at 4 bits is ~372 GB against
#     288 GB of HBM on one MI350, so the reference does not fit even at 128
#     tokens, where GPT-OSS's fits to 1024.
#
#   * Lengths stop well short of 32768. Each length re-runs the whole 4-rank
#     launch, the sink is max_seq_length x vocab_shard bf16 (0.04 GB at 128,
#     ~2.4 GB at 8192), and the intermittent cold-start wedge gets likelier
#     with length -- 512 wedged 2/2 before run_ppl.sh grew a retry.
#
# Env:
#   MODEL_PATH       GLM-5 MXFP4 checkpoint (defaults to the pinned local dir)
#   PPL_LENS         lengths to sweep (default "128 256 512 1024")
#   PPL_SWEEP_OUT    results dir (default outputs/glm5/ppl_sweep)
#   STALL_SECS       per-attempt log-silence budget (default 300)
#   RETRIES          per-length launch retries (default 2, honoured by run_ppl.sh)
set -uo pipefail   # NOT -e: one length wedging must not kill the sweep

cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$(cd ../.. && pwd)"
export MODEL_PATH="${MODEL_PATH:-/home/claudeuser/models/glm5-mxfp4}"
LENS="${PPL_LENS:-128 256 512 1024}"
OUT="${PPL_SWEEP_OUT:-$ROOT/outputs/glm5/ppl_sweep}"
mkdir -p "$OUT"
LOG="$OUT/sweep.log"
: > "$LOG"

echo "MODEL_PATH=$MODEL_PATH  NP=${NP:-4}  lengths: $LENS" | tee -a "$LOG"
echo "No Torch arm: 744B at 4 bits does not fit on one GPU." | tee -a "$LOG"

for n in $LENS; do
  echo "=== n=$n ===" | tee -a "$LOG"
  # run_ppl.sh owns the watchdog, the launch retry and the per-run gate. Give
  # each length its own dir: the summarizer globs <len>/mpk_ppl_rank*.json, and
  # a shared dir would let one length's dumps be read as another's.
  GLM5_OUTPUT_DIR="$OUT/$n" PPL_MAX_TOKENS="$n" ./run_ppl.sh \
      > "$OUT/$n.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "  FAILED rc=$rc (see $OUT/$n.log)" | tee -a "$LOG"
    tail -4 "$OUT/$n.log" | sed 's/^/    /' | tee -a "$LOG"
  else
    # run_ppl.sh sends the megakernel's own output to $GLM5_OUTPUT_DIR/
    # ppl_run.log, not to its stdout, so read the result from there -- $n.log
    # holds only the wrapper and the pytest summary.
    grep -aE "perplexity      |mean entropy|top-1 accuracy|sink self-check" \
        "$OUT/$n/ppl_run.log" | sed 's/^.*stdout>://; s/^/  /' | tee -a "$LOG"
  fi
done

echo "" | tee -a "$LOG"
python3 "$ROOT/tests/ci-tests/summarize_glm5_ppl_sweep.py" "$OUT" 2>&1 \
    | tee -a "$LOG"
echo "Results in $OUT (per-run logs: $OUT/<len>.log)" | tee -a "$LOG"
