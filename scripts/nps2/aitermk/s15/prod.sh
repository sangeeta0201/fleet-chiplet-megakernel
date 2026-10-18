#!/bin/bash
# prod.sh LABEL [NEW]: the single-stream production decode (opt2/best.env minus the END trio, recipe, DPX
# MULTI_DEV=2) on lane $LANE (default stk15): the non-speculative reference for per-segment comparisons.
# env: PROMPT (default gsm8k_1), PS (extra env), XF (extra -D, comma-led).
N=/home/user/nps1; H=/home/user/mk1/s15
LANE=${LANE:-stk15}
read -r _ DEV BUS CTR < <(awk -v l=$LANE '$1==l' $H/lanes.conf)
lab=${1:?label}; new=${2:-256}
exec 8> $H/.lock_$LANE
flock 8
P=/sys/bus/pci/devices/0000:$BUS:00.0
[ "$(cat $P/current_compute_partition)" = DPX ] || { echo "bus $BUS is not DPX"; exit 1; }
LT=/home/user/fleet-lanes/$LANE
PR=${PROMPT:-/root/user/nps1/opt/spec/bench_prompts/gsm8k_1.txt}
F="$(grep -v '^#' $LT/NPS2_RECIPE.env | grep = | tr '\n' ' ')"
S="$(cat $N/opt2/best.env | tr '\n' ' ') MPK_END_DIRECT=0 MPK_EVT_FAST=0 MPK_END_FAST=0"
[ -n "${XF:-}" ] && S=$(echo "$S" | sed "s/MPK_EXTRA_FLAGS=-DMPK_QKV_PF_DELAY_NS=1400/MPK_EXTRA_FLAGS=-DMPK_QKV_PF_DELAY_NS=1400$XF/")
echo "######## s15 prod $LANE $lab NEW=$new $(date -u +%FT%TZ) prompt=$(basename $PR) git=$(git -C $LT log -1 --format=%h)"
export CONTAINER=$CTR PATH_PREPEND=/root/user/nps1/vllmn/hip70
FLEET_HASH=none LANE_HIP_DEV=$DEV TIMEOUT=${TO:-600} LANE_ENVS=" " \
  EXTRA_ENVS="$F $S MPK_MULTI_DEV=2 MPK_NUM_XCDS=8 MPK_CHAT_DATE=2026-09-25 MPK_FIXED_DATE=2026-09-25 MAX_SEQ_LENGTH=${MSL:-5200} MAX_NEW_TOKENS=$new MPK_FWDTRACE_MIN=1 ${PS:-}" \
  bash $N/fleet_run_lane.sh $LANE $lab --prompt-file $PR --save-tokens /root/user/nps1/opt/spec/tok_$lab.json > /tmp/s15_$lab.out 2>&1
LG=$N/fleet_baselines/lane_$LANE/$lab.log
echo "decode $(grep -a 'Decode:' $LG | tail -1 | grep -oE 'avg [0-9.]+') ms/token | $(grep -a FWD_PASS_TOTAL $LG | tail -1 | grep -o 'avg_ms=[0-9.]*') | text $(cat $N/fleet_baselines/lane_$LANE/$lab.texthash 2>/dev/null)"
grep -aE "error:|Traceback|Memory access fault|NOT DONE" $LG | grep -v "loop not unrolled" | head -5 | cut -c1-200
echo "AGENT_LOOP_WAKE_S15PROD $lab $(date -u +%T)"

