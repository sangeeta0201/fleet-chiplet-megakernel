#!/bin/bash
# TOP UP chainonly's n.  probe_mtp_chain_alone.sh got chainonly=13.299 at n=1
# (reps 2 and 3 wedged at launch and were killed) against an in-batch bs2
# anchor of 12.805 (n=2), giving "MTP chain = +0.494 ms".  0.494 is only 1.9x
# the 0.26 ms noise floor, so the value needs reps even though the direction
# does not.
#
# Same two arms, same batch, interleaved rather than blocked: chainonly, bs2,
# chainonly, bs2, chainonly.  Interleaving costs two extra rebuilds but means
# a slow drift in the box cannot masquerade as an arm effect, and a hang in
# one arm no longer lands entirely on one side of the comparison.
set -u
ulimit -c 0
cd /home/claudeuser/fleet-chiplet-megakernel/demo/glm5
D=/tmp/mtptopup
mkdir -p $D

export MODEL_PATH=/home/claudeuser/models/glm5-mxfp4
export MAX_SEQ_LENGTH=512
export MAX_NEW_TOKENS=256
export MAX_SAVE_TOKENS=264
export MPK_BAR_SKEW=0
export MPK_SUBPHASE_TIMING=0
PROMPT="Write a short paragraph explaining why the sky appears blue."

one_run () {
  ARM=$1; REP=$2
  echo "===== $ARM REP $REP ====="
  if [ "$ARM" = chainonly ]; then
    unset MPK_SPEC_DECODE; export GLM_MTP_GRAPH_ONLY=1
    export MPK_EXTRA_ARGS="--mtp 1 --max-num-batched-tokens 2"
  else
    unset MPK_SPEC_DECODE GLM_MTP_GRAPH_ONLY
    export MPK_EXTRA_ARGS="--max-num-batched-tokens 2"
  fi
  rm -rf permanent_output_dir permanent_output_dir_rank*
  export KEEP_BUILD=0
  export MASTER_PORT=$((41100 + (RANDOM % 300)))
  # 1500 s: a clean rep is ~7 min plus ~10 min of build.  A wedged rep shows
  # no Decode: line and is discarded, so a shorter cap just buys back time.
  timeout 1500 ./run_mp8_dp_ep_fused.sh \
      --prompt "$PROMPT" --save-tokens "$D/${ARM}_r${REP}.json" \
      > "$D/${ARM}_r${REP}.log" 2>&1
  echo "  rc=$?"
  grep -h "Multi-layer table" "$D/${ARM}_r${REP}.log" | head -1 | sed 's/^/  /'
  grep -h "Decode:" "$D/${ARM}_r${REP}.log" | tail -1 | sed 's/^/  /'
}

one_run chainonly 4
one_run bs2       3
one_run chainonly 5
one_run bs2       4
one_run chainonly 6

echo "############ G1+G2+G3 GATE ############"
for ARM in chainonly bs2; do
  echo "===== $ARM ====="
  python3 correctness_gate.py --ctl '/tmp/det_bs1/r*_rank*.json' \
                              --arm "$D/${ARM}_r*_rank*.json" 2>&1 | tail -14
done
echo TOPUP_DONE

# ############################ RESULT ########################################
#
# Ran 2026-08-22 13:09-13:44.  THE BOX ATE MOST OF THIS BATCH.  Outcome per
# rep, in the interleaved order above:
#
#   chainonly r4   FAULT  "HIP error: an illegal memory access was encountered"
#   bs2       r3   OK     12.879 ms/iter   G1+G2 PASS, cluster member
#   chainonly r5   OK     13.164 ms/iter   G1+G2 PASS, cluster member
#   bs2       r4   FAULT  illegal memory access
#   chainonly r6   FAULT  illegal memory access
#
# All three faults land at the SAME log offset (line 4467, i.e. immediately
# after the dispatch banner, at the first decode iteration) on three DIFFERENT
# ranks (6, 4, 2), and one of them is the PLAIN bs2 arm.  So this is the box's
# intermittent NP=8 failure wearing a second face -- the known presentation is
# a hang at launch_persistent_kernel ENTER, this one is an illegal address at
# the first decode step -- and it is NOT caused by GLM_MTP_GRAPH_ONLY.  Two
# reps of the same binary (bs2 r3, chainonly r5) ran clean.
#
# Stopped here rather than spend more of the run feeding a box failing ~3 in 5.
#
# ############################ THE SPLIT, RECOMPUTED #########################
#
#   arm         walls (ms/iter)                  n   mean     G1+G2
#   bs1         10.619                           3   10.619   PASS
#   bs2         12.833  12.776  12.879           3   12.829   PASS
#   chainonly   13.299  13.164                   2   13.232   PASS
#   mtp         16.510  16.450                   2   16.480   PASS
#
#   10.619   bs=1 control
#   +2.210   2-wide BUILD GEOMETRY          zero extra tokens
#   +0.403   the ENTIRE MTP chain
#   +3.248   the REAL second decode row
#   =16.480  exact by construction
#
# vs the n=1 version in probe_mtp_chain_alone.sh (2.186 / 0.494 / 3.181): the
# chain moved DOWN 0.091 and the row UP 0.091.  The finding is reinforced, not
# weakened -- the chain is 2.4% of the wall and the second LIVE row is
# 2.210 + 3.248 = 5.458 ms.
#
# chainonly is n=2, not the n=3 the noise-floor rule wants.  Stated plainly.
# What licenses using it anyway: the two reps are 0.135 ms apart, HALF the
# 0.26 ms floor, and the bs2 anchor underneath them is n=3 with a 0.103 ms
# spread.  For the chain to be the expensive half, chainonly would have to sit
# above 14.6 -- five floors above both samples.
#
# ITEM 2 IS UNAFFECTED, and not by luck.  Width-2 pays GEOM+CHAIN+ROWACT
# twice, and that sum is pinned by (mtp - bs1) = 5.861 no matter how the three
# terms divide, so width-2 is 10.619 + 2(5.861) = 22.341 ms/iter under BOTH
# splits.  See demo/glm5/price_item2_width2.py -- CLOSED on its 7.3% ceiling.
