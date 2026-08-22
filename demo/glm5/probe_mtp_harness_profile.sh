#!/bin/bash
# WHERE DOES THE 3.721 ms MTP HARNESS GO?
#
# 466b940 measured the harness as MTP(16.530) - bs2 plain(12.809) = +3.721 ms,
# which is 1.7x the cost of a whole extra decode row and the single biggest
# unexplained item on the board.  Accounted work is nowhere near that:
#   trailing run = 2 layers   ~0.37 ms at the measured 186.301 us/layer
#   MTP prologue = embed gather + enorm/hnorm + 2 eh_proj GEMVs (5120x5120)
#   a SECOND full lm_head + argmax over vocab (mtp_argmax_in, demo.py:2162)
#
# Accept/reject is GPU-side (prepare_next_batch, demo.py:554), so this is NOT
# host round-trips -- it is on-device work that has to show up per task.
#
# BARSTAGEWS aggregates over layers and cannot separate the trailing run, so
# use the per-task profiler instead.  Per its own docstring the wall under
# --profiling is MEANINGLESS (profiler writes serialise the workers) and only
# the RELATIVE per-task numbers are valid.  That is exactly the use here:
# attribution of a delta already measured uninstrumented.  No wall is quoted
# from this run.
#
# Two arms so per-task-type totals can be diffed; MTP task types are distinctly
# named (mtp_*) but the draft layer reuses the main fused-layer task type.
set -u
ulimit -c 0
cd /home/claudeuser/fleet-chiplet-megakernel/demo/glm5
D=/tmp/prof_harness
mkdir -p $D

export MODEL_PATH=/home/claudeuser/models/glm5-mxfp4
export MAX_SEQ_LENGTH=512
export MAX_NEW_TOKENS=16
export MPK_BAR_SKEW=0
export MPK_SUBPHASE_TIMING=0
export MPK_PROFILING_ITERS=2
PROMPT="Write a short paragraph explaining why the sky appears blue."

run_arm () {
  ARM=$1; shift
  echo "########## ARM $ARM ##########"
  rm -rf permanent_output_dir permanent_output_dir_rank* profile_output.pt
  export KEEP_BUILD=0
  export MASTER_PORT=$((38100 + (RANDOM % 300)))
  timeout 1500 ./run_mp8_dp_ep_fused.sh --prompt "$PROMPT" --profiling \
      > "$D/${ARM}.log" 2>&1
  echo "  rc=$?"
  echo -n "  PROFILING flag in build: "
  grep -ho "\-DMPK_PROFILING_NUM_ITERS=[0-9]*" "$D/${ARM}.log" | sort -u | tr '\n' ' '; echo
  echo -n "  SPEC flag in build: "
  grep -ho "\-DMPK_SPEC_DECODE" "$D/${ARM}.log" | sort -u | tr '\n' ' '; echo
  grep -h "Multi-layer table" "$D/${ARM}.log" | head -1
  for f in profile_output_rank0.pt profile_output.pt; do
    [ -f "$f" ] && cp "$f" "$D/${ARM}_profile.pt" && echo "  saved $f" && break
  done
  ls -la profile_output* 2>/dev/null | head
}

# ---- arm 1: plain 2 decode rows, no speculation ----
unset MPK_SPEC_DECODE
export MPK_EXTRA_ARGS="--max-num-batched-tokens 2"
run_arm bs2

# ---- arm 2: MTP width-1 ----
export MPK_SPEC_DECODE=1
export MPK_EXTRA_ARGS="--mtp 1 --max-num-batched-tokens 2"
run_arm mtp

echo "########## SUMMARIES ##########"
for ARM in bs2 mtp; do
  echo "===== $ARM ====="
  [ -f "$D/${ARM}_profile.pt" ] && \
    python3 /home/claudeuser/fleet-chiplet-megakernel/tests/ci-tests/summarize_mpk_profile.py \
        "$D/${ARM}_profile.pt" --iters 2 --top 40 2>&1 | tail -60 \
    || echo "  no dump"
done
echo PROF_DONE

# ===================== RESULT, 2026-08-22: NEGATIVE =====================
# The obvious fix -- fold the separately-dispatched draft layer into the main
# replay run -- is worth ~0.09 ms of the 3.721 ms harness. Do not build it.
#
# The MTP arm's fused-layer task is perfectly BIMODAL, 232 calls each:
#     main run (76 layers)          27776.62 us/call
#     trailing run (draft+EP tail)    206.04 us/call
#     ratio                            0.742%
# bs2 is unimodal (232 calls, all ~46921 us) -- one run, as expected.
#
# Within-MTP-arm shares of summed worker compute (the iters divisor cancels):
#     fused MAIN run          93.06%
#     RMSNORM_LINEAR_MXFP8    3.99%   (lm_head + eh_proj)
#     LINEAR_RES              1.49%
#     fused TRAILING run      0.69%   <-- the draft layer
#     ARGMAX_PARTIAL          0.05%   (and MTP runs it TWICE: 101 -> 202)
# Everything MTP adds inside the megakernel is at most ~7% of compute, and the
# second lm_head+argmax is 0.05%. The harness is 22.5% of MTP's wall. It is not
# accounted for by megakernel compute share.
#
# It is also not host round-trips: the mirage path times per iteration from a
# device printf ([FWD_PASS] iter=N time_ms=X) and mpk() is ONE call for the
# whole decode loop.
#
# *** CROSS-ARM ABSOLUTE us FROM THIS INSTRUMENT ARE INVALID. ***
# The profiled span is 24336 us/iter for bs2 against a real 12.809 ms wall
# (1.9x) but 15186 us/iter for mtp against a real 16.530 ms wall (0.92x) -- the
# profiler INVERTED the arm ordering. Same 76 layers, same batch size, and the
# bs2 arm's main run is billed 46921 us against mtp's 27777. Only WITHIN-arm
# ratios survive; that is what the bimodal split above uses. The summarizer's
# own docstring says as much ("the wall-clock under --profiling is
# meaningless"), and this run is the concrete demonstration.
#
# Caveat on the dump: MAX_NEW_TOKENS=16 with MPK_PROFILING_ITERS=2 means the
# capture may include prefill. The bimodal RATIO is unaffected (both
# populations sit in the same dump under the same divisor), but do not read the
# absolute us as a decode iteration.
#
# NEXT: the remaining ~3.6 ms needs a WALL-level ablation, not more profiling.
# ========================================================================
