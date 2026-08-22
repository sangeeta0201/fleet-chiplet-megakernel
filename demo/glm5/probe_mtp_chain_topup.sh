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
