#!/bin/bash
# RE-GATE MTP UNDER G1+G2+G3. The ccf8cfa retraction used exact-token equality,
# which 369032c proved illegal on this model: bs=1 has two attractor
# continuations and splits 2/2 across them, so "MTP diverges from ctl at token
# 0" is what two ctl runs do to each other half the time.
#
# Control set: the four bs=1 runs already on disk in /tmp/det_bs1, SAME prompt,
# same model, greedy. They enumerate both attractor clusters, which is exactly
# what G3 needs. No reason to re-run them.
#
# This script therefore runs only the MTP arm, 3x on that same prompt, and
# gates with demo/glm5/correctness_gate.py.
#
# MTP requires MPK_SPEC_DECODE=1 (asserted demo.py:515) and
# --max-num-batched-tokens >= 2 (asserted demo.py:529).
# ===================== RESULT, 2026-08-22: CONFIRMED =====================
# G1 cross-rank identity : PASS 8/8 on all three reps
# G2 coherence           : PASS (distinct 0.530-0.568, top bigram 9-11)
# G3 attractor           : MEMBER of control cluster B
#
# Pairwise exact-prefix, 4 controls + 3 MTP, 264 tokens each:
#            r1    r2    r5    rB  mtp1  mtp2  mtp3
#      r1     -   122     0     0     0     0     0
#      r2   122     -     0     0     0     0     0
#      r5     0     0     -   124    41    67    81
#      rB     0     0   124     -    41    67    81
#    mtp1     0     0    41    41     -    41    41
#    mtp2     0     0    67    67    41     -    67
#    mtp3     0     0    81    81    41    67     -
#
# A strict ultrametric tree with TWO clusters: A={r1,r2} and
# B={r5,rB,mtp1,mtp2,mtp3}.  Every MTP run shares 41-81 tokens of exact greedy
# output with the bs=1 controls in B and EXACTLY 0 with A.  The load-bearing
# detail: mtp3 agrees with control r5 (81) MORE than with mtp2 (67).  MTP runs
# are closer to bs=1 controls than to each other -- MTP is drawn from the same
# nondeterministic distribution, not a distinct stream.  A real accept/reject
# defect gives the opposite signature (self-consistent MTP, offset from ctl).
#
# NOTE the gate script printed "NEW continuation" here: correctness_gate.py's
# MIN_PREFIX=100 was calibrated on the two deepest control pairs (122/124), but
# drift within a basin is graded while cluster membership is categorical at
# 0-vs-41.  Read the matrix, not the advisory label.
#
# Text check, mtp_r3 vs ctl r5: identical reasoning scaffold, diverging at
# "gas molecules, nitrogen, oxygen" vs "gas molecules like nitrogen and
# oxygen".  A near-tie flip; both correct.
#
#   rep1  accept 0.871  1.875 tok/iter  16.515 ms/iter  8.813 ms/token
#   rep2  accept 0.861  1.861 tok/iter  16.535 ms/iter  8.884 ms/token
#   rep3  accept 0.857  1.861 tok/iter  16.542 ms/iter  8.889 ms/token
#   median: accept 0.861, 16.530 ms/iter, 8.884 ms/token vs 10.619 = 1.196x
#
# ccf8cfa's retraction of 8.952 ms/token is itself RETRACTED.  It rested on
# exact-token equality, which 369032c proved illegal on this model.
#
# GREP TAGS (both of the ones this script looks for are WRONG):
#   the stats line is tagged [SPEC], not [MTP]
#   the flag appears as -DMPK_SPEC_DECODE in the hipcc line, never as
#   MPK_SPEC_DECODE=1 -- so `grep MPK_SPEC_DECODE=` finds nothing on a run
#   where speculation is fully enabled.
# =========================================================================
set -u
ulimit -c 0
cd /home/claudeuser/fleet-chiplet-megakernel/demo/glm5
D=/tmp/mtp_regate
mkdir -p $D

export MODEL_PATH=/home/claudeuser/models/glm5-mxfp4
export MAX_SEQ_LENGTH=512
export MAX_NEW_TOKENS=256
export MAX_SAVE_TOKENS=264
export MPK_BAR_SKEW=0
export MPK_SUBPHASE_TIMING=0
export MPK_SPEC_DECODE=1
export MPK_EXTRA_ARGS="--mtp 1 --max-num-batched-tokens 2"
PROMPT="Write a short paragraph explaining why the sky appears blue."

rm -rf permanent_output_dir permanent_output_dir_rank*
export KEEP_BUILD=0
for REP in 1 2 3; do
  export MASTER_PORT=$((36100 + (RANDOM % 300)))
  echo "===== MTP REP $REP ====="
  timeout 900 ./run_mp8_dp_ep_fused.sh \
      --prompt "$PROMPT" --save-tokens "$D/mtp_r${REP}.json" \
      > "$D/mtp_r${REP}.log" 2>&1
  echo "  rc=$?"
  # Prove the flag actually landed in this build, not just in the environment.
  echo -n "  SPEC_DECODE in log: "
  grep -ho "\-DMPK_SPEC_DECODE" "$D/mtp_r${REP}.log" | sort -u | tr '\n' ' '; echo
  grep -h "\[SPEC\]" "$D/mtp_r${REP}.log" | tail -1
  grep -h "Decode:" "$D/mtp_r${REP}.log" | sed 's/.*(avg /  wall avg /;s/)$//'
  export KEEP_BUILD=1
done

echo "############ G1+G2+G3 GATE ############"
python3 correctness_gate.py --ctl '/tmp/det_bs1/r*_rank*.json' \
                            --arm "$D/mtp_r*_rank*.json"
echo MTPREGATE_DONE
