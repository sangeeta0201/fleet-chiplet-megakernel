#!/bin/bash
# specg.sh KIND N LABEL [K=V ...]: one run of the method behind the blog's GSM8K and 1K-conversation speculation
# numbers (~/nps1/specg/specg.sh of Oct 13 2026), with this checkout's compositions (scripts/nps2/{opt2,spec,dpx}).
#   ns   single stream: the recipe + opt2/best.env, END trio off, default attention partition; DPX+NPS2
#   ns7  ns at 7 KV chunks: the 4-token engine's partition, k3's identity reference
#   k1   EAGLE3, 1 draft, 2 virtual slots; DPX+NPS2
#   k3   EAGLE3, 3 chained drafts, 4 virtual slots (the 4-token engine); DPX+NPS2
#   def  ns in the default mode (stock SPX+NPS1, flags as deployed)
# One summary line per run goes to $GSM_OUT/results.txt (q_specg.py), the tokens to $GSM_OUT/tok_<label>.json.
# Environment (defaults = the Oct 18 2026 runs):
#   PROMPT          prompt file, container path (required)
#   MSL             MAX_SEQ_LENGTH (1024)
#   REF             label whose tokens a k1 / k3 run must reproduce
#   GSM_OUT         output directory, host path, under HOST_HOME ($HOST_HOME/mk1/specg_latest)
#   HOST_HOME CTR_HOME  the home directory on the host and inside the containers ($HOME, /root/<your login>)
#   FLEET_LOG_ROOT  run logs: $FLEET_LOG_ROOT/lane_<lane>/<label>.log ($HOST_HOME/nps1/fleet_baselines)
#   DPX_LANE DPX_BUS DPX_CONTAINER  worktree, GPU and container of the DPX+NPS2 runs (fair, e5, fleet_ak4t1)
#   DEF_LANE DEF_BUS DEF_CONTAINER  the same for the default-mode runs (fair2, 05, fleet_ship2)
#   PATH_PREPEND    container path of the ROCm 7.0 hipcc wrapper ($CTR_HOME/fleet-lanes/$DPX_LANE/scripts/nps2/vllm/hip70)
#   TO              timeout per run, seconds (900)
#   LM=bf16         the checkpoint's BF16 LM head (as vLLM runs it) instead of the default build's MXFP4 one, switched
#                   as in the served BF16 builds: the BF16 variables, and -DMPK_LM_BF16 appended to the run's flags
#   DRY=1           print "label|environment|demo arguments" instead of running
HERE=$(cd "$(dirname "$0")" && pwd)
SC=$(cd "$HERE/../.." && pwd)
HOST_HOME=${HOST_HOME:-$HOME}; CTR_HOME=${CTR_HOME:-/root/$(id -un)}
FLEET_LOG_ROOT=${FLEET_LOG_ROOT:-$HOST_HOME/nps1/fleet_baselines}
export HOST_HOME CTR_HOME FLEET_LOG_ROOT
G=${GSM_OUT:-$HOST_HOME/mk1/specg_latest}; GC=$CTR_HOME${G#$HOST_HOME}
DPX_LANE=${DPX_LANE:-fair}; DPX_BUS=${DPX_BUS:-e5}; DPX_CONTAINER=${DPX_CONTAINER:-fleet_ak4t1}
DEF_LANE=${DEF_LANE:-fair2}; DEF_BUS=${DEF_BUS:-05}; DEF_CONTAINER=${DEF_CONTAINER:-fleet_ship2}
PATH_PREPEND=${PATH_PREPEND:-$CTR_HOME/fleet-lanes/$DPX_LANE/scripts/nps2/vllm/hip70}
mkdir -p $G
kind=$1; n=$2; lab=$3; shift 3
XE="$*"
if [ $kind = def ]; then LANE=$DEF_LANE; BUS=$DEF_BUS; C=$DEF_CONTAINER; WANT=SPX/NPS1; else LANE=$DPX_LANE; BUS=$DPX_BUS; C=$DPX_CONTAINER; WANT=DPX/NPS2; fi
LT=$HOST_HOME/fleet-lanes/$LANE
SYS=/sys/bus/pci/devices/0000:$BUS:00.0
mode=$(cat $SYS/current_compute_partition)/$(cat $SYS/current_memory_partition)
[ "${DRY:-0}" = 1 ] || { [ "$mode" = $WANT ] && [ "$(ls /sys/module/amdgpu/parameters | grep -c aid_local)" = 0 ]; } \
  || { echo "specg.sh: GPU $BUS is $mode / not stock, need $WANT: not running $lab"; exit 1; }
first_dev() {
  local i=0 nn p loc=$((16#$BUS * 256))
  for nn in $(ls /sys/class/kfd/kfd/topology/nodes | sort -n); do
    p=/sys/class/kfd/kfd/topology/nodes/$nn/properties
    [ "$(awk '/^simd_count/{print $2}' $p)" = 0 ] && continue
    [ "$(awk '/^location_id/{print $2}' $p)" = "$loc" ] && { echo $i; return 0; }
    i=$((i + 1))
  done
  echo NONE; return 1
}
if [ "${DRY:-0}" = 1 ]; then f=0; else f=$(first_dev) || { echo "no HIP device for bus $BUS"; exit 1; }; fi
if [ $kind = def ]; then DEV=$f; MD=""; else DEV=$f,$((f + 1)); MD="MPK_MULTI_DEV=2 MPK_NUM_XCDS=8"; fi
[ "${DRY:-0}" = 1 ] || ! pgrep -f '^python3? [d]emo[.]py' > /dev/null \
  || { echo "specg.sh: a fleet run is live; not running $lab"; exit 1; }
COM="MPK_CHAT_DATE=2026-09-25 MPK_FIXED_DATE=2026-09-25 MPK_MAX_SAVE_TOKENS=4096 MAX_SEQ_LENGTH=${MSL:-1024} MAX_NEW_TOKENS=$n MPK_FWDTRACE_MIN=1"
DROP="MPK_MOE_XCD_PAIR MPK_MOE_W2_REMAP MPK_MOE_W2_REMAP_DELAY_US MPK_MOE_W2_REMAP3 MPK_MOE_W2_REMAP3_R0 MPK_W2_KWIN
MPK_W13_HOF_AIDREP MPK_LOCAL_TOPK MPK_LTK_DPP MPK_LTK_NO_LOGIT_STORE MPK_LTK_EARLY_TAG MPK_LTK_SELLOGIT MPK_LTK_TIE_FIX
MPK_QKV_GAMMA_NORMREG MPK_ROUTER_OPROJ_PARTIAL MPK_ROP_QOVL MPK_ROP_KSEL MPK_LM_RESADD
MPK_QKV_SLAB_LDS MPK_QKV_ROPE_REG MPK_QKV_RED_LDS_BAR
MPK_OPROJ_FLAT MPK_P9_FLAT MPK_OPROJ_HIER_NC MPK_AID_P9_NC MPK_GATE_DEFER
MPK_ATTN_KV_PREFETCH"
RE="^($(echo $DROP | tr ' ' '|'))="
STEPS="MPK_FUSED_INLINE=1 MPK_PLACE_ARENA=1 MPK_AID_NC_REP=1 MPK_ATTN_INLINE=1 MPK_ATTN_SL_OVERLAP=1 MPK_ATTN_EPI_UNROLL=1
MPK_PUB_NOWAIT=1 MPK_ATTN_PF_DURING_QK=1 MPK_ATTN_EXP2_BUILTIN=1 MPK_OPROJ_IMM=1 MPK_ATTN_CVT_BUILTIN=1
MPK_LLVM_OPTS=-amdgpu-sched-strategy=max-ilp MPK_PFIN_RETIRE=1 MPK_RESID_MAP=1 MPK_OPROJ_SCALAR_XCD=1 MPK_RT_CONST=1
MPK_LDS_DST_S=1 MPK_MFMA_TAIL_NOP=15"
VS="MPK_W13_BIAS_COUNTED_WAIT=0 MPK_ATTN_SLICE_RELEASE=0 MPK_SPEC_VSLOT=1"
SPECBASE="$(grep -v '^#' $LT/NPS2_RECIPE.env | grep = | grep -v -E "$RE" | tr '\n' ' ') $(echo $STEPS) $VS"
FAIR="MPK_RESID_MAP=1 MPK_MOE_FLAT=0 MPK_LM_LEADER_INV=0 MPK_PLACE_SERIAL=1 MPK_PLACE_GUARD=1000 MPK_GATE_RESCUE=16384"
case $kind in
  ns|ns7|def)
    ENV="$(grep -v '^#' $LT/NPS2_RECIPE.env | grep = | tr '\n' ' ') $(tr '\n' ' ' < $SC/opt2/best.env) MPK_END_DIRECT=0 MPK_EVT_FAST=0 MPK_END_FAST=0"
    [ $kind = ns7 ] && ENV="$ENV CK_FMHA_NUM_KV_CHUNKS=7"
    [ $kind = def ] && ENV="$ENV $FAIR"
    XA="" ;;
  k1)
    ENV="$SPECBASE MPK_SPEC_K=1 $(cat $SC/spec/vt2_full7.env) MPK_EXTRA_FLAGS=$(cat $SC/spec/vt2_x14.txt) $(cat $SC/dpx/dpx_spec.env)"
    XA="--max-num-batched-requests 2 --max-num-batched-tokens 2" ;;
  k3)
    ENV="$SPECBASE MPK_SPEC_K=3 $(cat $SC/spec/dflash/n4.env) MPK_EAGLE3=1 MPK_EAGLE3_FP8=1 MPK_E3_OWN_BO=1 MPK_EXTRA_FLAGS=$(cat $SC/spec/dflash/n4_final_x.txt),-DMPK_E3_CHAIN=3 $(cat $SC/spec/dflash/n4_final_env.txt) $(cat $SC/dpx/dpx_spec.env)"
    XA="--max-num-batched-requests 4 --max-num-batched-tokens 4" ;;
  *) echo "specg.sh: KIND must be ns, ns7, k1, k3 or def"; exit 1 ;;
esac
if [ "${LM:-mxfp4}" = bf16 ]; then
  EF=$(echo " $ENV" | grep -oE ' MPK_EXTRA_FLAGS=[^ ]*' | tail -1 | sed 's/^ MPK_EXTRA_FLAGS=//')
  ENV="$ENV MPK_LM_BF16=1 MPK_LM_PF0=0 MPK_LM_PIPE_S=0 MPK_LM_HEAD_GROUP_PIPELINE=0 MPK_LM_HEAD_KMAJOR=0"
  ENV="$ENV MPK_LM_HEAD_WAVE_TILE_DMA=0 MPK_LMT_FQ=0 MPK_EXTRA_FLAGS=${EF:+$EF,}-DMPK_LM_BF16"
fi
if [ "${DRY:-0}" = 1 ]; then
  echo "$lab|$(echo "$ENV $MD $COM $XE" | tr -s ' ' | sed 's/^ //; s/ $//')|--prompt-file ${PROMPT:-} $XA"
  exit 0
fi
live=/tmp/fleet_lane_${LANE}_$lab.log
rm -f $live
CONTAINER=$C PATH_PREPEND=$PATH_PREPEND FLEET_HASH=none LANE_HIP_DEV=$DEV \
  TIMEOUT=${TO:-900} LANE_ENVS=" " EXTRA_ENVS="$ENV $MD $COM $XE" \
  bash $SC/fleet_run_lane.sh $LANE $lab --prompt-file ${PROMPT:?set PROMPT} $XA \
  --save-tokens $GC/tok_$lab.json > /tmp/$lab.out 2>&1 &
pid=$!
fault=0
while kill -0 $pid 2> /dev/null; do
  sleep 5
  # a hang = a launch not done at 45 s with a worker or scheduler not placed (a long launch with every one of them
  # resident is not a hang), or a memory fault
  if [ $fault = 0 ] && { grep -aq "Memory access fault" $live 2> /dev/null \
       || grep -a "NOT DONE at 45s" $live 2> /dev/null | grep -qvE 'workers ([0-9]+)/\1 schedulers ([0-9]+)/\2'; }; then
    fault=1
    for q in $(docker exec $C bash -c 'ps -eo pid,cmd | grep -E "python3? .*demo[.]py" | grep -v grep | awk "{print \$1}"' < /dev/null); do
      docker exec $C kill -9 $q > /dev/null 2>&1 < /dev/null
    done
  fi
done
wait $pid 2> /dev/null
[ $fault = 1 ] && docker exec $C bash -c "rm -f $CTR_HOME/fleet-lanes/$LANE/demo/gpt_oss/gpucore.*" > /dev/null 2>&1 < /dev/null
LANE_DIR=$FLEET_LOG_ROOT/lane_$LANE GSM_OUT=$G python3 $HERE/q_specg.py $lab $kind ${REF:-}

