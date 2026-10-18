#!/bin/bash
# mk_eval_def.sh LANE BUS LABEL HEAD EVALS [K=V ...] (node 2, default mode = stock SPX+NPS1): accuracy evals of the
# no-speculation megakernel behind `vllm serve` (serve_eval.py session, original fleet_mk plugin, fleet_ship2 = vLLM
# 0.27.1), one request at a time. The deployable default-mode configuration of the blog runs (recipe + opt2/best.env,
# END trio off, no per-die replicas): its outputs equal DPX+NPS2's (checked over 32K tokens). HEAD: mx | bf.
# LANE = a worktree of lane/fair (one per concurrent run; the scripts come from ~/fleet-lanes/fair). EVALS: comma list
# of gsm8k5, chat, mmlupro. Results ~/mk1/results/mk_eval/LABEL/<eval>/, [SERVEEVAL] lines in the run log.
set -u
lane=$1 bus=$2 lab=$3 head=$4 evals=$5; shift 5
cd /home/user/fleet-lanes/fair/scripts/nps2/blog || exit 1
export CONTAINER=fleet_ship2 PATH_PREPEND=/root/user/fleet-lanes/fair/scripts/nps2/vllm/hip70 BUS=$bus LANE=$lane
STATUS=/home/user/mk1/eval/spec/status_mk_eval.txt
. ./common.sh
[ "$(mode)" = SPX/NPS1 ] && stock || { st "$lab: GPU $BUS is $(mode) / not stock: need the default mode"; exit 1; }
docker start $CONTAINER > /dev/null
DEVS=$(first_dev) || { st "$lab: no HIP device for bus $BUS"; exit 1; }
OUTC=/root/user/mk1/results/mk_eval/$lab; mkdir -p /home/user/mk1/results/mk_eval/$lab
case $head in
  mx) ENV="$REC0 $BEST $FAIR" ;;
  bf) ENV="$REC0 $BEST_BF $FAIR $BFX" ;;
  *) st "$lab: unknown head $head"; exit 1 ;;
esac
SV="MPK_TOK_HOST=1 MPK_WRAP=/root/user/nps1/vllmn/wrap_sess.sh SESS_SCRIPT=/root/user/mk1/eval/serve_eval.py"
SV="$SV SESS_EVALS=$evals SESS_EVAL_OUT=$OUTC SESS_MAXLEN=6100 VN_UTIL=0.9 VN_PORT=$((8400 + 16#$bus % 64))"
COM="MPK_CHAT_DATE=2026-09-25 MPK_FIXED_DATE=2026-09-25 MAX_SEQ_LENGTH=6200 MAX_NEW_TOKENS=5900"
PR=/root/user/nps1/opt/spec/bench_prompts/gsm128/q127.txt  # 202 tokens: arena 6,102 >= the server's 6,100
st "$lab: start lane $lane bus $bus (HIP $DEVS) head $head evals $evals extra [$*]"
rm -f /tmp/fleet_lane_${LANE}_$lab.log
CONTAINER=$CONTAINER PATH_PREPEND=$PATH_PREPEND FLEET_HASH=none LANE_HIP_DEV=$DEVS TIMEOUT=14400 LANE_ENVS=" " \
  EXTRA_ENVS="$ENV $COM $SV $*" bash $SC/fleet_run_lane.sh $LANE $lab --prompt-file $PR > /tmp/mkeval_$lab.out 2>&1
grep -a '\[SERVEEVAL\].*RESULT' $LD/$lab.log 2> /dev/null | sed "s/^/[MKEVAL] $lab /" | tee -a $ST
grep -aq '\[SERVEEVAL\].*DONE' $LD/$lab.log 2> /dev/null \
  || st "$lab: no DONE line: $(grep -aE 'Traceback|Error|error:|exited|NOT DONE' $LD/$lab.log 2> /dev/null | tail -3 | tr '\n' ' ' | cut -c1-300)"
st "$lab: end"
