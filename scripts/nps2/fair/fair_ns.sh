#!/bin/bash
# fair_ns.sh LABEL ARM [NEW]: single-stream production decode on lane fair, node 2 SPX/NPS1 stock, one GPU ($DEV,
# default 0), container fleet_ship2, deployable-NPS1 flags (no per-AID replication: MOE_FLAT / LM_LEADER_INV off).
#   ARM mxfp4: the production MXFP4 LM head.  ARM bf16: MPK_LM_BF16 (BF16 LM head, the MXFP4-only head flags off).
N=/home/user/nps1; LT=/home/user/fleet-lanes/fair
lab=${1:?label}; arm=${2:?mxfp4|bf16}; new=${3:-256}
F="$(grep -v '^#' $LT/NPS2_RECIPE.env | grep = | tr '\n' ' ')"
S="$(cat $N/opt2/best.env | tr '\n' ' ') MPK_END_DIRECT=0 MPK_EVT_FAST=0 MPK_END_FAST=0"
FAIR="MPK_RESID_MAP=1 MPK_MOE_FLAT=0 MPK_LM_LEADER_INV=0 MPK_PLACE_SERIAL=1 MPK_PLACE_GUARD=1000 MPK_GATE_RESCUE=16384"
case $arm in
  mxfp4) X="" ;;
  bf16) S=$(echo "$S" | sed 's/MPK_EXTRA_FLAGS=-DMPK_QKV_PF_DELAY_NS=1400/MPK_EXTRA_FLAGS=-DMPK_QKV_PF_DELAY_NS=1400,-DMPK_LM_BF16/')
        X="MPK_LM_BF16=1 MPK_LM_PF0=0 MPK_LM_PIPE_S=0 MPK_LM_HEAD_GROUP_PIPELINE=0 MPK_LM_HEAD_KMAJOR=0 MPK_LM_HEAD_WAVE_TILE_DMA=0 MPK_LMT_FQ=0" ;;
  *) echo "ARM mxfp4|bf16"; exit 1 ;;
esac
PR=/root/user/nps1/prompt_1k.txt
echo "######## fair_ns $lab arm=$arm NEW=$new $(date -u +%FT%TZ) git=$(git -C $LT log -1 --format=%h)"
export CONTAINER=fleet_ship2 PATH_PREPEND=/root/user/nps1/vllmn/hip70
docker start fleet_ship2 > /dev/null
FLEET_HASH=none LANE_HIP_DEV=${DEV:-0} TIMEOUT=${TO:-900} LANE_ENVS=" " \
  EXTRA_ENVS="$F $S $FAIR $X MPK_CHAT_DATE=2026-09-25 MPK_FIXED_DATE=2026-09-25 MAX_SEQ_LENGTH=$((982 + new)) MAX_NEW_TOKENS=$new MPK_MAX_SAVE_TOKENS=$((new + 64))" \
  bash $N/fleet_run_lane.sh fair $lab --prompt-file $PR --save-tokens /root/user/nps1/opt/spec/tok_$lab.json > /tmp/fair_$lab.out 2>&1
LG=$N/fleet_baselines/lane_fair/$lab.log
echo "decode $(grep -a 'Decode:' $LG | tail -1 | grep -oE 'avg [0-9.]+') ms/token | $(grep -a FWD_PASS_TOTAL $LG | tail -1 | grep -o 'avg_ms=[0-9.]*') | text $(cat $N/fleet_baselines/lane_fair/$lab.texthash 2>/dev/null)"
grep -a -E "LM head (BF16|MXFP4)" $LG | tail -n 1 | cut -c1-200
grep -a -o -- "-DMPK_LM_BF16" $LG | head -n 1
grep -aE "error:|Traceback|Memory access fault|NOT DONE" $LG | grep -v "loop not unrolled" | head -5 | cut -c1-220
