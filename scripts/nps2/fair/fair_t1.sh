#!/bin/bash
# fair_t1.sh [CFGS]: the GSM8K-128 served measurement (t1_table.sh arm A: vllm serve + the original fleet_mk plugin,
# in-process build, 128 requests, 256 tokens) on node 1 e5 (DPX+NPS2, HIP 12,13, container fleet_ak4t1), lane fair:
# per config BF16 LM head / MXFP4 LM head / BF16 LM head, with the hang fixes in every run. No SMI calls.
set -u
CFGS=${1:-"ns e3 df8"}
M=/home/user/mk1; C=fleet_ak4t1; LN=fair
grep -q "^fair " $M/s15/lanes.conf || echo "fair  12,13  e5  fleet_ak4t1" >> $M/s15/lanes.conf
PD=/root/user/nps1/opt/spec/bench_prompts/gsm128
CSD=/root/user/fleet-lanes/stk15/scripts/nps2/spec/s15/gsm128
LD=/home/user/fleet-lanes/stk15/scripts/nps2/spec/s15/lead
BEST="$(cat $LD/best.flags)"; BENV="$(cat $LD/best.env)"
A=$(echo "$BEST" | sed -E 's/,-DMPK_ATTN_FIXED_MERGER(,|$)/\1/; s/,-DMPK_ATTN_FM_V=[0-9]+(,|$)/\1/')
FIX="MPK_PLACE_GUARD=1000 MPK_PLACE_SERIAL=1 MPK_GATE_RESCUE=16384"
BFX="MPK_LM_BF16=1 MPK_LM_PF0=0 MPK_LM_PIPE_S=0 MPK_LM_HEAD_GROUP_PIPELINE=0 MPK_LM_HEAD_KMAJOR=0 MPK_LM_HEAD_WAVE_TILE_DMA=0 MPK_LMT_FQ=0"
RH=$M/results/serve_fair; RC=/root/user/mk1/results/serve_fair
FB=/home/user/nps1/fleet_baselines/lane_$LN; ST=$M/fp6/status_fair_t1.txt
mkdir -p $RH
st() { echo "[FT1] $(date -u +%T) $*" | tee -a $ST; }
docker start $C > /dev/null
# both plugins publish the entry point fleet_mk: the original one (lane vllm_integration, found through serve128.py's
# PYTHONPATH) only wins while libaitermk-vllm is uninstalled; reinstalled (editable, as before) when this script ends
AD=/root/user/mk1/libAiterMK/libAiterMK/engine_adapters/vllm
docker exec $C pip uninstall -y libaitermk-vllm > /dev/null 2>&1 < /dev/null
trap 'docker exec $C pip install --no-deps -e $AD > /dev/null 2>&1 < /dev/null; st "libaitermk-vllm reinstalled: $(docker exec $C pip show libaitermk-vllm 2> /dev/null < /dev/null | grep -c Editable)"' EXIT
trap 'pkill -P $$; exit 1' INT TERM
EP=$(docker exec $C python3 -c 'import importlib.metadata as m; print([e.value for e in m.entry_points(group="vllm.general_plugins") if e.name == "fleet_mk"])' < /dev/null)
st "fleet_mk entry points without PYTHONPATH: $EP (expect [])"
quiet() {
  for i in $(seq 1 90); do
    [ "$(docker exec $C pgrep -fc 'serve128|vllm|demo[.]py|orig_check')" = 0 ] && return 0; sleep 2
  done; return 1
}
st "start CFGS=\"$CFGS\" lane $LN git $(git -C /home/user/fleet-lanes/$LN log -1 --format=%h) dirty $(git -C /home/user/fleet-lanes/$LN status --porcelain -- include demo | wc -l)"
port=8191
for CFG in $CFGS; do
  for run in bf1 mx1 bf2; do
    quiet || { st "container busy before $CFG $run"; exit 1; }
    arm=${run%[0-9]}; lab=fair_${CFG}_$run
    SV="MPK_TOK_HOST=1 MPK_WRAP=/root/user/nps1/vllmn/wrap_sess.sh SESS_SCRIPT=$CSD/serve128.py SESS_PRE=g128"
    SV="$SV SESS_CFG=$CFG SESS_N=128 SESS_NEW=256 SESS_OUT=$RC/$lab.json FLEET_NATIVE_DUMP=$RC/dump_$lab"
    SV="$SV VN_UTIL=0.9 VN_PORT=$port $FIX"
    XB=""
    if [ $arm = bf ]; then SV="$SV $BFX"; XB=",-DMPK_LM_BF16"; fi
    case $CFG in
      ns)  LANE=$LN PROMPT=$PD/q127.txt TO=2400 XF="$XB" PS="$SV" bash $M/s15/prod.sh $lab 264 > /dev/null 2>&1 ;;
      e3)  LANE=$LN DRAFTER=eagle3 PROMPT=$PD/q127.txt T=6 K=5 XF="$BEST$XB" PS="$BENV $SV" TO=2400 \
             bash $M/s15/run.sh $lab 264 > /dev/null 2>&1 ;;
      df8) LANE=$LN DRAFTER=dflash PROMPT=$PD/q127.txt T=8 K=7 XF="$A$XB" PS="MPK_TAIL_PUB_LATE=1 $SV" TO=2400 \
             bash $M/s15/run.sh $lab 264 > /dev/null 2>&1 ;;
    esac
    LOG=$FB/$lab.log
    SUM=$(grep -a '\[SERVE128\] SUMMARY' $LOG | tail -n 1 | cut -c1-300)
    HD=$(grep -a -oE "LM head (BF16|MXFP4)" $LOG | tail -n 1)
    st "$CFG $run head=[$HD] bf16flag=$(grep -a -c -- '-DMPK_LM_BF16' $LOG) hang15=$(grep -ac 'NOT DONE at 15s' $LOG) err=$(grep -a -c -E 'Traceback|error:' $LOG) | $SUM"
    port=$((port + 1))
  done
done
st "AGENT_LOOP_WAKE_FAIRT1"
