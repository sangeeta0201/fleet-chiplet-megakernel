#!/bin/bash
# run.sh LABEL [NEW]: one speculative-decode run (DFlash or EAGLE3 drafter + VTN verify engine) on lane $LANE of the
# 1500 tok/s effort. Same flag set as dpxe3/df_vint.sh (n4.env + n4_final_x/env + dpx_spec.env + the mt_vs recipe).
# env: LANE (default stk15; devices/bus/container from s15/lanes.conf), T (block = MPK_VTN_T = batched tokens,
#      default 6), K (drafts, default T-1; K=0 = reference), DRAFTER (dflash | eagle3; KC = EAGLE3 chain, default K),
#      NODF=1 (no drafter at all), PROMPT (prompt file; default gsm8k_1), PS (extra env assignments),
#      XF (extra -D flags, comma-led), XARGS (extra demo.py args), TO (timeout s, default 600).
# Log fleet_baselines/lane_$LANE/LABEL.log, tokens opt/spec/tok_LABEL.json; summary via s15/sum.py.
N=/home/user/nps1; S=$N/opt/spec; D=$N/dpxe3; F=$S/dflash; H=/home/user/mk1/s15
LANE=${LANE:-stk15}
read -r _ DEV BUS CTR < <(awk -v l=$LANE '$1==l' $H/lanes.conf)
[ -n "$CTR" ] || { echo "unknown lane $LANE"; exit 2; }
lab=${1:?label}; new=${2:-256}
# one run at a time per lane: a second chain on the same lane waits here instead of killing this run's demo.py
exec 8> $H/.lock_$LANE
flock 8
T=${T:-6}; K=${K:-$((T-1))}
P=/sys/bus/pci/devices/0000:$BUS:00.0
[ "$(cat $P/current_compute_partition)" = DPX ] || { echo "bus $BUS is not DPX"; exit 1; }
LT=/home/user/fleet-lanes/$LANE; L=$N/fleet_baselines/lane_$LANE
busy() { docker exec $CTR bash -c "for p in \$(pgrep -f 'demo[.]py'); do tr '\\0' '\\n' < /proc/\$p/environ 2>/dev/null | grep -qx HIP_VISIBLE_DEVICES=$DEV && echo \$p; done"; }
b=$(busy); [ -n "$b" ] && { echo "a run is live on $DEV (pid $b); not running"; exit 1; }
PR=${PROMPT:-/root/user/nps1/opt/spec/bench_prompts/gsm8k_1.txt}
echo "######## s15 $LANE $lab drafter=${DRAFTER:-dflash} T=$T K=$K NEW=$new $(date -u +%FT%TZ) mode=$(cat $P/current_compute_partition)/$(cat $P/current_memory_partition) prompt=$(basename $PR) git=$(git -C $LT log -1 --format=%h) dirty=$(git -C $LT status --porcelain -- include python | wc -l)"
E=$(cat $F/n4.env)
X=$(sed "s/-DMPK_VTN_T=4/-DMPK_VTN_T=$T/" $F/n4_final_x.txt | tr -d '\n')
EX=$(cat $F/n4_final_env.txt)
if [ "${DRAFTER:-dflash}" = eagle3 ]; then
  KC=${KC:-$K}; [ "$KC" -lt 1 ] && KC=3
  DFL="MPK_EAGLE3=1 MPK_EAGLE3_FP8=1 MPK_E3_OWN_BO=1"; X="$X,-DMPK_E3_CHAIN=$KC"
else
  DFL="MPK_DFLASH=1"
fi
[ "${NODF:-0}" = 1 ] && DFL=""
DROP="MPK_MOE_XCD_PAIR MPK_MOE_W2_REMAP MPK_MOE_W2_REMAP_DELAY_US MPK_MOE_W2_REMAP3 MPK_MOE_W2_REMAP3_R0 MPK_W2_KWIN
MPK_W13_HOF_AIDREP MPK_LOCAL_TOPK MPK_LTK_DPP MPK_LTK_NO_LOGIT_STORE MPK_LTK_EARLY_TAG MPK_LTK_SELLOGIT MPK_LTK_TIE_FIX
MPK_QKV_GAMMA_NORMREG MPK_ROUTER_OPROJ_PARTIAL MPK_ROP_QOVL MPK_ROP_KSEL MPK_LM_RESADD
MPK_QKV_SLAB_LDS MPK_QKV_ROPE_REG MPK_QKV_RED_LDS_BAR
MPK_OPROJ_FLAT MPK_P9_FLAT MPK_OPROJ_HIER_NC MPK_AID_P9_NC MPK_GATE_DEFER
MPK_ATTN_KV_PREFETCH ${EXTRA_DROP:-}"
grep -v '^#' $LT/NPS2_RECIPE.env | grep = > /tmp/s15_$lab.env
for k in $DROP; do grep -v "^$k=" /tmp/s15_$lab.env > /tmp/s15_$lab.tmp; mv /tmp/s15_$lab.tmp /tmp/s15_$lab.env; done
STEPS="MPK_FUSED_INLINE=1 MPK_PLACE_ARENA=1 MPK_AID_NC_REP=1 MPK_ATTN_INLINE=1 MPK_ATTN_SL_OVERLAP=1 MPK_ATTN_EPI_UNROLL=1
MPK_PUB_NOWAIT=1 MPK_ATTN_PF_DURING_QK=1 MPK_ATTN_EXP2_BUILTIN=1 MPK_OPROJ_IMM=1 MPK_ATTN_CVT_BUILTIN=1
MPK_LLVM_OPTS=-amdgpu-sched-strategy=max-ilp MPK_PFIN_RETIRE=1 MPK_RESID_MAP=1 MPK_OPROJ_SCALAR_XCD=1 MPK_RT_CONST=1
MPK_LDS_DST_S=1 MPK_MFMA_TAIL_NOP=15"
for k in ${EXTRA_DROP:-}; do STEPS=$(echo "$STEPS" | tr ' \n' '\n\n' | grep -v "^$k=" | tr '\n' ' '); done
RF="$(tr '\n' ' ' < /tmp/s15_$lab.env) $(echo $STEPS | tr '\n' ' ')"
VS="MPK_W13_BIAS_COUNTED_WAIT=0 MPK_ATTN_SLICE_RELEASE=0 MPK_SPEC_VSLOT=1 MPK_MAX_SAVE_TOKENS=4096 MPK_FIXED_DATE=2026-09-25"
ALL="$RF $VS MPK_SPEC_K=$K $E MPK_EXTRA_FLAGS=$X${XF:-} MPK_VTNBAL=1 $EX $DFL $(cat $D/dpx_spec.env) MPK_FWDTRACE_MIN=1 ${PS:-}"
export CONTAINER=$CTR PATH_PREPEND=/root/user/nps1/vllmn/hip70
run1() {
  rm -f /tmp/fleet_lane_${LANE}_$1.log
  FLEET_HASH=none LANE_HIP_DEV=$DEV TIMEOUT=${TO:-600} LANE_ENVS=" " \
    EXTRA_ENVS="MPK_CHAT_DATE=2026-09-25 MAX_SEQ_LENGTH=${MSL:-5200} MAX_NEW_TOKENS=$new MPK_MULTI_DEV=2 MPK_NUM_XCDS=8 $ALL" \
    bash $N/fleet_run_lane.sh $LANE $1 --prompt-file $PR --max-num-batched-tokens $T \
    --max-num-batched-requests $T --save-tokens /root/user/nps1/opt/spec/tok_$1.json ${XARGS:-} > /tmp/s15_$1.out 2>&1 &
  local pid=$!
  while kill -0 $pid 2>/dev/null; do
    if [ -z "${NOHANG:-}" ] && grep -aq 'NOT DONE at 45s' /tmp/fleet_lane_${LANE}_$1.log 2>/dev/null; then
      docker exec $CTR bash -c "for p in \$(pgrep -f 'demo[.]py'); do tr '\\0' '\\n' < /proc/\$p/environ 2>/dev/null | grep -qx HIP_VISIBLE_DEVICES=$DEV || continue; case \$(readlink /proc/\$p/cwd) in *$LANE*) kill -9 \$p;; esac; done"
      echo "LAUNCH_HANG_KILLED $1"; break
    fi
    sleep 5
  done
  wait $pid
}
run1 $lab
if [ -z "${NOHANG:-}" ] && grep -aq 'NOT DONE at 45s' /tmp/fleet_lane_${LANE}_$lab.log 2>/dev/null; then
  echo "retrying as ${lab}_r"; run1 ${lab}_r
  mv -f $L/$lab.log $L/${lab}_hang.log
  cp -f $L/${lab}_r.log $L/$lab.log; cp -f $L/${lab}_r.texthash $L/$lab.texthash 2>/dev/null
  cp -f $S/tok_${lab}_r.json $S/tok_$lab.json 2>/dev/null
fi
LG=$L/$lab.log
grep -aE "error:|static assert|#error|Traceback|Memory access fault|AssertionError|NOT DONE" $LG 2>/dev/null | grep -v "loop not unrolled" | sed -E 's|.*/persistent_kernel/||' | sort -u | head -12 | cut -c1-230
# leftovers of this lane only
for q in $(docker exec $CTR bash -c "for p in \$(ps -eo pid,cmd | grep -E 'python3? .*demo[.]py' | grep -v grep | awk '{print \$1}'); do case \$(readlink /proc/\$p/cwd) in *$LANE*) echo \$p;; esac; done"); do docker exec $CTR kill -9 $q; done
python3 $H/sum.py $lab ${REF:-}
echo "AGENT_LOOP_WAKE_S15 $LANE $lab $(date -u +%T)"

