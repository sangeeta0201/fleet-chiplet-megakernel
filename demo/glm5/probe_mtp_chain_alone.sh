#!/bin/bash
# PRICE THE MTP CHAIN WITH ONE ACTIVE ROW.
#
# probe_row_vs_harness_split.sh measured, one batch, same day:
#   bs1 (1-wide build, 1 row)           10.619
#   bs2 (2-wide build, 1 row)           12.754   +2.135  geometry, zero tokens
#   mtp (2-wide build, 2 rows + chain)  16.480   +3.726  row-2 work AND chain
# and proved the middle arm decodes ONE token per iteration, so the +3.726 is
# a SUM of two things nobody has separated. The arm that would have split it
# from the other side -- MPK_SPEC_DECODE=1 with no --mtp, the self-draft --
# wedges the megakernel 3/3 at launch.
#
# This is the same split from the near side. GLM_MTP_GRAPH_ONLY=1 wires the
# whole MTP chain into the task graph -- embedding gather, enorm/hnorm, both
# eh_proj GEMVs, the draft layer as its own replay run, the second
# lm_head+argmax -- while leaving MPK_SPEC_DECODE off, so prepare_next_batch
# dispatches ONE row (persistent_kernel.cuh:1155) and the draft token is
# thrown away. It is the configuration demo.py:542's assert exists to forbid,
# taken deliberately.
#
#   D - bs2  = the MTP chain, alone, at one active row
#   mtp - D  = the real second decode row
#
# bs2 is re-run in THIS batch as the anchor rather than compared to yesterday's
# number (glm5-baseline-is-11005-over-n5). Both arms are output-gated.
#
# Arm D commits one token per iteration by construction, so its wall is
# ms/iter == ms/token and its text must match a plain bs=1 control -- nothing
# in the chain feeds back into the committed stream. A G1/G2 failure here
# would mean the chain is corrupting the main path, which is worth knowing on
# its own.
#
# Rep 1 of each arm compiles; 2400 s, because 1200 s cut three build-reps off
# in the previous sweep.
set -u
ulimit -c 0
cd /home/claudeuser/fleet-chiplet-megakernel/demo/glm5
D=/tmp/mtpchain
mkdir -p $D

export MODEL_PATH=/home/claudeuser/models/glm5-mxfp4
export MAX_SEQ_LENGTH=512
export MAX_NEW_TOKENS=256
export MAX_SAVE_TOKENS=264
export MPK_BAR_SKEW=0
export MPK_SUBPHASE_TIMING=0
PROMPT="Write a short paragraph explaining why the sky appears blue."

run_arm () {
  ARM=$1; NREP=$2
  echo "########## ARM $ARM ##########"
  rm -rf permanent_output_dir permanent_output_dir_rank*
  export KEEP_BUILD=0
  for REP in $(seq 1 $NREP); do
    export MASTER_PORT=$((39100 + (RANDOM % 300)))
    echo "===== $ARM REP $REP ====="
    timeout 2400 ./run_mp8_dp_ep_fused.sh \
        --prompt "$PROMPT" --save-tokens "$D/${ARM}_r${REP}.json" \
        > "$D/${ARM}_r${REP}.log" 2>&1
    echo "  rc=$?"
    echo -n "  -DMPK_SPEC_DECODE: "
    grep -ho "\-DMPK_SPEC_DECODE" "$D/${ARM}_r${REP}.log" | sort -u | tr '\n' ' '
    echo -n "| NBT: "
    grep -ho "\-DMPK_MAX_NUM_BATCHED_TOKENS=[0-9]*" "$D/${ARM}_r${REP}.log" \
        | sort -u | tr '\n' ' '; echo
    # "2 runs" proves the draft layer is really in the graph for arm D.
    grep -h "Multi-layer table" "$D/${ARM}_r${REP}.log" | head -1 | sed 's/^/  /'
    grep -h "\[SPEC\]" "$D/${ARM}_r${REP}.log" | tail -1 | sed 's/^/  /'
    grep -h "Decode:" "$D/${ARM}_r${REP}.log" | tail -1 | sed 's/^/  /'
    export KEEP_BUILD=1
  done
}

# ---- D: MTP chain in the graph, ONE active row ----
unset MPK_SPEC_DECODE
export GLM_MTP_GRAPH_ONLY=1
export MPK_EXTRA_ARGS="--mtp 1 --max-num-batched-tokens 2"
run_arm chainonly 3

# ---- anchor: plain 2-wide build, one active row, no chain ----
unset GLM_MTP_GRAPH_ONLY
export MPK_EXTRA_ARGS="--max-num-batched-tokens 2"
run_arm bs2 2

echo "############ G1+G2+G3 GATE ############"
for ARM in chainonly bs2; do
  echo "===== $ARM ====="
  python3 correctness_gate.py --ctl '/tmp/det_bs1/r*_rank*.json' \
                              --arm "$D/${ARM}_r*_rank*.json" 2>&1 | tail -14
done
echo MTPCHAIN_DONE
