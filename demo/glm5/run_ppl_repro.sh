#!/usr/bin/env bash
# IS THE PERPLEXITY NUMBER REPRODUCIBLE? Runs one identical PPL config N times
# and reports the spread.
#
# This is the continuous counterpart of probe_bs1_determinism.sh, which asks
# the same question about decode TOKENS and answered it "no" -- five identical
# reps of one build split into two attractor continuations. Perplexity is the
# metric the gate actually uses, so the gate's tolerance has to be set from
# perplexity's own spread, not from the token result.
#
# Why this is worth a script rather than three runs by hand: the spread is not
# a fixed property of the kernel, it is a property of how sharp the model's
# rows are. Unprefixed (out-of-distribution, entropy ~4.5 nats) it measured
# 29%; with the [gMASK]<sop> prefix (entropy ~1.3 nats) the same measurement
# gives ~4.5%, because a reordered float sum only changes the answer where two
# logits are nearly tied. Any change that sharpens or flattens the
# distribution moves this number, so it needs re-measuring, not quoting.
#
# Design, following probe_bs1_determinism.sh: BUILD ONCE and discard that rep,
# then N reps that all reuse the same build via PPL_KEEP_BUILD=1. Nothing
# varies between reps -- same binary, same corpus, same teacher-forced prefill,
# and no seed to pin because argmax and cross-entropy are deterministic given
# the logits. Any disagreement among the reps is the kernel's own
# nondeterminism: the EP fold and the MoE W2 f32 atomics retire in arrival
# order, so the reduction is order-dependent.
#
# Env:
#   REPS             identical reps after the build rep (default 5)
#   PPL_MAX_TOKENS   corpus tokens (default 128 -- the length that has never
#                    hit the cold-start wedge)
#   PPL_REPRO_OUT    results dir (default outputs/glm5/ppl_repro)
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$(cd ../.. && pwd)"
export MODEL_PATH="${MODEL_PATH:-/home/claudeuser/models/glm5-mxfp4}"
REPS="${REPS:-5}"
export PPL_MAX_TOKENS="${PPL_MAX_TOKENS:-128}"
OUT="${PPL_REPRO_OUT:-$ROOT/outputs/glm5/ppl_repro}"
mkdir -p "$OUT"
LOG="$OUT/repro.log"
: > "$LOG"

echo "REPS=$REPS  PPL_MAX_TOKENS=$PPL_MAX_TOKENS  NP=${NP:-4}" | tee -a "$LOG"

# Rep B performs the build. Its dump is kept but reported separately: on the
# token probe the build rep was the one that diverged, so "first run after a
# build" is a hypothesis worth being able to check rather than assume away.
for REP in B $(seq 1 "$REPS"); do
  echo "===== REP $REP =====" | tee -a "$LOG"
  GLM5_OUTPUT_DIR="$OUT/r$REP" PPL_KEEP_BUILD="$([ "$REP" = B ] && echo 0 || echo 1)" \
      ./run_ppl.sh > "$OUT/r$REP.out" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "  FAILED rc=$rc (see $OUT/r$REP.out)" | tee -a "$LOG"
  else
    grep -aE "perplexity      |mean entropy|sink self-check" \
        "$OUT/r$REP/ppl_run.log" | sed 's/^.*stdout>://; s/^/  /' \
        | tee -a "$LOG"
  fi
done

echo "" | tee -a "$LOG"
python3 "$ROOT/tests/ci-tests/summarize_glm5_ppl_repro.py" "$OUT" 2>&1 \
    | tee -a "$LOG"
