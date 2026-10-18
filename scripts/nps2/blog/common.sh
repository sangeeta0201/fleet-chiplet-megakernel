#!/bin/bash
# common.sh: the run compositions behind the GPT-OSS-120B blog numbers, shared by run_dpx.sh and run_default.sh.
# Sourced, not run. Every run goes through scripts/nps2/fleet_run_lane.sh with the lane's own NPS2_RECIPE.env.
#
# Environment (defaults = the measurement boxes):
#   LANE            worktree name under $HOST_HOME/fleet-lanes (default: this checkout's directory name)
#   HOST_HOME       host directory holding fleet-lanes/ (default $HOME)
#   CTR_HOME        the same directory inside the container (default /root/<your login>)
#   FLEET_LOG_ROOT  logs go to $FLEET_LOG_ROOT/lane_$LANE/<label>.log (default $HOST_HOME/nps1/fleet_baselines)
#   CONTAINER       container to build and run in (default fleet_v1)
#   PATH_PREPEND    container path of a directory put first on PATH, for images newer than ROCm 7.0: the ROCm 7.0
#                   hipcc wrapper scripts/nps2/vllm/hip70 (it reads ROCM70, default /root/user/rocm70)
#   BUS             PCI bus of the GPU (default e5)
#   WAIT_IDLE=1     wait before each run until no process outside $CONTAINER uses a GPU (default 1)
#   DRY=1           print "label|ENVS|demo args" for each run instead of running it
set -u
BLOG=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SC=$(cd "$BLOG/.." && pwd)
LT=$(cd "$SC/../.." && pwd)
LANE=${LANE:-$(basename "$LT")}
HOST_HOME=${HOST_HOME:-$HOME}
CTR_HOME=${CTR_HOME:-/root/$(id -un)}
FLEET_LOG_ROOT=${FLEET_LOG_ROOT:-$HOST_HOME/nps1/fleet_baselines}
export HOST_HOME CTR_HOME FLEET_LOG_ROOT
CONTAINER=${CONTAINER:-fleet_v1}
PATH_PREPEND=${PATH_PREPEND:-}
BUS=${BUS:-e5}
WAIT_IDLE=${WAIT_IDLE:-1}
DRY=${DRY:-0}
LD=$FLEET_LOG_ROOT/lane_$LANE
TREE_CTR=$CTR_HOME/fleet-lanes/$LANE
P1K=$TREE_CTR/docs/nps2/prompt_1k.txt
ST=${STATUS:-$LD/blog_status.txt}
mkdir -p "$LD" "$LT/demo/gpt_oss/blog_tokens"
SYS=/sys/bus/pci/devices/0000:$BUS:00.0

# --- compositions -----------------------------------------------------------------------------------------------
REC0="$(grep -v '^#' $LT/NPS2_RECIPE.env | grep = | tr '\n' ' ')"
# single-stream: the recipe + the best measured flag set; the END trio is off (wrong past 8,192 tokens)
BEST="$(tr '\n' ' ' < $SC/opt2/best.env) MPK_END_DIRECT=0 MPK_EVT_FAST=0 MPK_END_FAST=0"
# BF16 LM head (the checkpoint's lm_head, as vLLM runs it) in place of the MXFP4 one
BFX="MPK_LM_BF16=1 MPK_LM_PF0=0 MPK_LM_PIPE_S=0 MPK_LM_HEAD_GROUP_PIPELINE=0 MPK_LM_HEAD_KMAJOR=0 MPK_LM_HEAD_WAVE_TILE_DMA=0 MPK_LMT_FQ=0"
BEST_BF="$(echo "$BEST" | sed 's/MPK_EXTRA_FLAGS=-DMPK_QKV_PF_DELAY_NS=1400/MPK_EXTRA_FLAGS=-DMPK_QKV_PF_DELAY_NS=1400,-DMPK_LM_BF16/')"
# phase recorder: per-phase time per token over the 1,024 generated tokens + the critical path of pass 1500
RECORDER="MPK_PHASE_SLOTS=1 MPK_PHASE_LDS=1 MPK_PHASE_START_ITER=982 MPK_PHASE_SNAP=1500"
# DPX+NPS2: one GPU as two logical devices, one per die (persistent_kernel.py adds the placement hang fixes)
MULTI="MPK_MULTI_DEV=2 MPK_NUM_XCDS=8"
# default mode (stock SPX+NPS1) as deployed: no per-die replicas (the stock driver cannot place them), the two flags
# that store into them off, and the launch-placement hang fixes
FAIR="MPK_RESID_MAP=1 MPK_MOE_FLAT=0 MPK_LM_LEADER_INV=0 MPK_PLACE_SERIAL=1 MPK_PLACE_GUARD=1000 MPK_GATE_RESCUE=16384"
# speculative builds: the recipe minus the single-token-only flags, the measured steps, then the VT2 env files
DROP="MPK_MOE_XCD_PAIR MPK_MOE_W2_REMAP MPK_MOE_W2_REMAP_DELAY_US MPK_MOE_W2_REMAP3 MPK_MOE_W2_REMAP3_R0 MPK_W2_KWIN
MPK_W13_HOF_AIDREP MPK_LOCAL_TOPK MPK_LTK_DPP MPK_LTK_NO_LOGIT_STORE MPK_LTK_EARLY_TAG MPK_LTK_SELLOGIT MPK_LTK_TIE_FIX
MPK_QKV_GAMMA_NORMREG MPK_ROUTER_OPROJ_PARTIAL MPK_ROP_QOVL MPK_ROP_KSEL MPK_LM_RESADD
MPK_QKV_SLAB_LDS MPK_QKV_ROPE_REG MPK_QKV_RED_LDS_BAR
MPK_OPROJ_FLAT MPK_P9_FLAT MPK_OPROJ_HIER_NC MPK_AID_P9_NC MPK_GATE_DEFER
MPK_ATTN_KV_PREFETCH"
SPF="$(grep -v '^#' $LT/NPS2_RECIPE.env | grep = | grep -v -E "^($(echo $DROP | tr ' ' '|'))=" | tr '\n' ' ')"
STEPS="MPK_FUSED_INLINE=1 MPK_PLACE_ARENA=1 MPK_AID_NC_REP=1 MPK_ATTN_INLINE=1 MPK_ATTN_SL_OVERLAP=1 MPK_ATTN_EPI_UNROLL=1 MPK_PUB_NOWAIT=1 MPK_ATTN_PF_DURING_QK=1 MPK_ATTN_EXP2_BUILTIN=1 MPK_OPROJ_IMM=1 MPK_ATTN_CVT_BUILTIN=1 MPK_LLVM_OPTS=-amdgpu-sched-strategy=max-ilp MPK_PFIN_RETIRE=1 MPK_RESID_MAP=1 MPK_OPROJ_SCALAR_XCD=1 MPK_RT_CONST=1 MPK_LDS_DST_S=1 MPK_MFMA_TAIL_NOP=15"
SPEC1="$SPF $STEPS MPK_W13_BIAS_COUNTED_WAIT=0 MPK_ATTN_SLICE_RELEASE=0 MPK_SPEC_VSLOT=1 MPK_SPEC_K=1 $(cat $SC/spec/vt2_full7.env) MPK_EXTRA_FLAGS=$(cat $SC/spec/vt2_x14.txt) $(cat $SC/dpx/dpx_spec.env)"
SPEC3="$SPF $STEPS MPK_W13_BIAS_COUNTED_WAIT=0 MPK_ATTN_SLICE_RELEASE=0 MPK_SPEC_VSLOT=1 MPK_MAX_SAVE_TOKENS=4096 MPK_SPEC_K=3 $(cat $SC/spec/dflash/n4.env) MPK_EAGLE3=1 MPK_EAGLE3_FP8=1 MPK_E3_OWN_BO=1 MPK_EXTRA_FLAGS=$(cat $SC/spec/dflash/n4_final_x.txt),-DMPK_E3_CHAIN=3 $(cat $SC/spec/dflash/n4_final_env.txt) $(cat $SC/dpx/dpx_spec.env)"
# EAGLE3 1 draft with the BF16 LM head, switched as in the served BF16 builds (scripts/nps2/fair/fair_t1.sh): the BF16
# variables after the build's own, and -DMPK_LM_BF16 appended to its flag list
SPEC1_BF="$SPEC1 $BFX MPK_EXTRA_FLAGS=$(cat $SC/spec/vt2_x14.txt),-DMPK_LM_BF16"

# --- helpers ----------------------------------------------------------------------------------------------------
st() { echo "[BLOG] $(date -u +%T) $*" | tee -a "$ST"; }
mode() { echo "$(cat $SYS/current_compute_partition 2> /dev/null)/$(cat $SYS/current_memory_partition 2> /dev/null)"; }
stock() { [ "$(ls /sys/module/amdgpu/parameters 2> /dev/null | grep -c aid_local)" = 0 ]; }
first_dev() {  # first HIP device index of $BUS (KFD GPU nodes in numeric order; DPX GPUs count twice)
  local i=0 n p loc=$((16#$BUS * 256))
  for n in $(ls /sys/class/kfd/kfd/topology/nodes | sort -n); do
    p=/sys/class/kfd/kfd/topology/nodes/$n/properties
    [ "$(awk '/^simd_count/{print $2}' $p)" = 0 ] && continue
    [ "$(awk '/^location_id/{print $2}' $p)" = "$loc" ] && { echo $i; return 0; }
    i=$((i + 1))
  done
  echo NONE; return 1
}
others() {  # KFD processes outside $CONTAINER (omnistat's collector excepted)
  local p c id
  id=$(docker inspect -f '{{.Id}}' $CONTAINER 2> /dev/null)
  for p in $(ls /sys/class/kfd/kfd/proc 2> /dev/null); do
    c=$(tr '\0' ' ' < /proc/$p/cmdline 2> /dev/null | cut -c1-80)
    case "$c" in *omnistat*) continue ;; esac
    grep -q "$id" /proc/$p/cgroup 2> /dev/null && continue
    echo "$p:$c;"
  done
}
wait_idle() {
  local f
  [ "$WAIT_IDLE" = 1 ] || return 0
  while true; do
    f=$(others); [ -z "$f" ] && return 0
    st "waiting, other GPU users: $(echo $f | cut -c1-200)"; sleep 30
  done
}
kill_demo() {
  local q
  for q in $(docker exec $CONTAINER bash -c 'ps -eo pid,cmd | grep -E "python3? .*demo[.]py" | grep -v grep | awk "{print \$1}"' < /dev/null); do
    docker exec $CONTAINER kill -9 $q > /dev/null 2>&1 < /dev/null
  done
}

# run LABEL KIND N [K=V ...]   KIND: ns | nsbf | ps | psbf (single-stream; ps = with the recorder) | sp | spbf (EAGLE3,
# 1 draft) | d3 (EAGLE3, 3 drafts, 4-token engine); *bf = BF16 LM head. PROMPT / PL override the prompt (container
# path) / its length.
# Needs MODE_ENV (empty for the default mode, $MULTI for DPX+NPS2) and DEVS (HIP devices) from the caller.
run() {
  local lab=$1 kind=$2 n=$3; shift 3
  local xe="$*" pl=${PL:-982} pr=${PROMPT:-$P1K} ENV XA="" COM try pid d sp
  local msl=$((pl + n))
  COM="MPK_CHAT_DATE=2026-09-25 MPK_FIXED_DATE=2026-09-25 MPK_MAX_SAVE_TOKENS=40000 MAX_SEQ_LENGTH=$msl MAX_NEW_TOKENS=$n"
  case $kind in
    ns) ENV="$REC0 $BEST $MODE_ENV" ;;
    nsbf) ENV="$REC0 $BEST_BF $MODE_ENV $BFX" ;;
    ps) ENV="$REC0 $BEST $MODE_ENV $RECORDER" ;;
    psbf) ENV="$REC0 $BEST_BF $MODE_ENV $BFX $RECORDER" ;;
    sp) ENV="$SPEC1 $MODE_ENV"; XA="--max-num-batched-requests 2 --max-num-batched-tokens 2"
        [ $(( (msl + 4095) / 4096 * 2 )) -gt 16 ] && XA="$XA --max-num-pages 32" ;;
    spbf) ENV="$SPEC1_BF $MODE_ENV"; XA="--max-num-batched-requests 2 --max-num-batched-tokens 2"
        [ $(( (msl + 4095) / 4096 * 2 )) -gt 16 ] && XA="$XA --max-num-pages 32" ;;
    d3) ENV="$SPEC3 $MODE_ENV"; XA="--max-num-batched-requests 4 --max-num-batched-tokens 4"
        COM="MPK_CHAT_DATE=2026-09-25 MPK_FIXED_DATE=2026-09-25 MAX_SEQ_LENGTH=5200 MAX_NEW_TOKENS=$n" ;;
    *) st "unknown kind $kind"; return 1 ;;
  esac
  [ -z "$MODE_ENV" ] && ENV="$ENV $FAIR"
  if [ "$DRY" = 1 ]; then
    echo "$lab|$(echo "$ENV $COM $xe" | tr -s ' ' | sed 's/^ //; s/ $//')|--prompt-file $pr $XA"
    return 0
  fi
  for try in 1 2 3; do
    wait_idle
    rm -f /tmp/fleet_lane_${LANE}_$lab.log
    CONTAINER=$CONTAINER PATH_PREPEND=$PATH_PREPEND FLEET_HASH=none LANE_HIP_DEV=$DEVS TIMEOUT=900 LANE_ENVS=" " \
      EXTRA_ENVS="$ENV $COM $xe" bash $SC/fleet_run_lane.sh $LANE $lab --prompt-file $pr $XA \
      --save-tokens $TREE_CTR/demo/gpt_oss/blog_tokens/tok_$lab.json > /tmp/blog_$lab.out 2>&1 &
    pid=$!
    while kill -0 $pid 2> /dev/null; do
      sleep 5
      # a hang = a launch not done at 45 s with a worker or scheduler not placed; a long run's launch passes 45 s
      # with every one of them resident (BF16 LM head, 32K tokens), which is not a hang
      if grep -aq "Memory access fault" /tmp/fleet_lane_${LANE}_$lab.log 2> /dev/null \
        || grep -a "NOT DONE at 45s" /tmp/fleet_lane_${LANE}_$lab.log 2> /dev/null \
           | grep -qvE 'workers ([0-9]+)/\1 schedulers ([0-9]+)/\2'; then
        st "$lab try $try: launch hang or memory fault, killed"; kill_demo; break
      fi
    done
    wait $pid 2> /dev/null
    grep -aq 'Decode:' $LD/$lab.log 2> /dev/null && break
    sleep 10
  done
  d=$(grep -a 'Decode:' $LD/$lab.log 2> /dev/null | tail -1 | grep -oE 'avg [0-9.]+ ?ms' | grep -oE '[0-9.]+')
  sp=$(grep -a '\[SPEC\] passes' $LD/$lab.log 2> /dev/null | tail -1 | grep -oE 'decode_passes=[0-9]+ drafted=[0-9]+ accepted=[0-9]+')
  st "$lab mode=$(mode) kind=$kind N=$n decode_avg=${d:-FAIL} ${sp:+[$sp]} text=$(cut -c1-12 $LD/$lab.texthash 2> /dev/null) extra=[$xe${PROMPT:+ PROMPT=$(basename $PROMPT)}]"
}

