#!/bin/bash
# t1_table.sh [CFGS] [N]: the GSM8K-128 served table on node 1 (lane ak4t1 = DPX devices 0,1 of BDF 05, container
# fleet_ak4t1), with the hang fixes (serialized worker/scheduler launch, launch guard, gate rescue) in both arms.
#   1. export each config with the fixes into ~/mk1/providers_t1/CFG (as exp_run.sh: arena = longest question + 264)
#      and build its provider DSO in the container (build_hipcc.py, ROCm 7.0 hipcc as the standalone build)
#   2. per config: A = vllm serve + the original fleet_mk plugin (in-process build), B = vllm serve + libAiterMK
#      fleet_vllm with that DSO. Tokens are checked against the node 1 standalone references in both arms.
# No SMI calls (rocm-smi / amd-smi) anywhere in here.
set -u
CFGS=${1:-"ns e3 df8"}; N=${2:-128}
M=/home/user/mk1; L=$M/logs; S=$M/scripts
C=fleet_ak4t1; LN=ak4t1
PD=/root/user/nps1/opt/spec/bench_prompts/gsm128
CSD=/root/user/fleet-lanes/stk15/scripts/nps2/spec/s15/gsm128
LD=/home/user/fleet-lanes/stk15/scripts/nps2/spec/s15/lead
BEST="$(cat $LD/best.flags)"; BENV="$(cat $LD/best.env)"
A=$(echo "$BEST" | sed -E 's/,-DMPK_ATTN_FIXED_MERGER(,|$)/\1/; s/,-DMPK_ATTN_FM_V=[0-9]+(,|$)/\1/')
FIX="MPK_PLACE_GUARD=1000 MPK_PLACE_SERIAL=1 MPK_GATE_RESCUE=16384"
PH=$M/providers_t1; PC=/root/user/mk1/providers_t1      # host / container view
RH=$M/results/serve_t1; RC=/root/user/mk1/results/serve_t1
PKG=/root/user/mk1/libAiterMK/libAiterMK/engine_adapters/vllm
FB=/home/user/nps1/fleet_baselines/lane_$LN
mkdir -p $PH $RH
quiet() {   # nothing of ours left running in the container
  for i in $(seq 1 90); do
    [ "$(docker exec $C pgrep -fc 'serve128|vllm|demo[.]py|orig_check')" = 0 ] && return 0; sleep 2
  done; return 1
}
echo "######## t1_table CFGS=\"$CFGS\" N=$N $(date -u +%FT%TZ)"
until grep -q AGENT_LOOP_WAKE_STOPPER $L/bisect3_result.txt 2>/dev/null; do sleep 5; done
quiet || { echo "[T1] container still busy after the trigger hunt"; exit 1; }
QL=$(python3 -c "
import json
L = [len(json.load(open('/home/user/nps1/opt/spec/tok_g128_k0b6_q%03d.json' % q))['prompt_ids']) for q in range(128)]
print('%03d' % max(range(128), key=lambda i: L[i]))")
echo "[T1] arena from q$QL"

# ---- 1. export + build, per config
for CFG in $CFGS; do
  quiet || { echo "[T1] busy before export $CFG"; exit 1; }
  docker exec $C rm -rf $PC/$CFG
  SV="MPK_TOK_HOST=1 MPK_PROVIDER_EXPORT=$PC/$CFG $FIX"
  case $CFG in
    ns)  LANE=$LN PROMPT=$PD/q$QL.txt TO=2400 PS="$SV" bash $M/s15/prod.sh t1x_ns 264 > /dev/null 2>&1 ;;
    e3)  LANE=$LN DRAFTER=eagle3 PROMPT=$PD/q$QL.txt T=6 K=5 XF="$BEST" PS="$BENV $SV" TO=2400 \
           bash $M/s15/run.sh t1x_e3 264 > /dev/null 2>&1 ;;
    df8) LANE=$LN DRAFTER=dflash PROMPT=$PD/q$QL.txt T=8 K=7 XF="$A" PS="MPK_TAIL_PUB_LATE=1 $SV" TO=2400 \
           bash $M/s15/run.sh t1x_df8 264 > /dev/null 2>&1 ;;
  esac
  XL=$FB/t1x_$CFG.log
  FL=$(grep -aoE '\-DMPK_(PLACE_GUARD|PLACE_SERIAL|GATE_RESCUE)(=[0-9]+)?' $XL | sort -u | tr '\n' ' ')
  echo "[T1] export $CFG $(date -u +%T): $(grep -aE 'Decode:' $XL | tail -n 1 | cut -c1-120) | fix flags: $FL"
  [ -d $PH/$CFG/src ] || { echo "[T1] export $CFG FAILED"; grep -aE 'Traceback|Error' $XL | head -n 5; exit 1; }
  docker exec -i $C bash -l > $L/t1_build_$CFG.out 2>&1 <<EOF
export PATH=/root/user/nps1/vllmn/hip70:\$PATH
cd $PC/$CFG
python3 /root/user/mk1/scripts/build_hipcc.py $PC/$CFG $CFG > $PC/$CFG/build_hipcc.log 2>&1
echo "build exit \$?"
nm -D --defined-only \$(ls $PC/$CFG/build_hipcc/*.so) | grep -c ' T fleet_mpk_'
EOF
  echo "[T1] build $CFG $(date -u +%T): $(tr '\n' ' ' < $L/t1_build_$CFG.out | cut -c1-160)"
  grep -q 'build exit 0' $L/t1_build_$CFG.out || { tail -n 5 $PH/$CFG/build_hipcc.log; exit 1; }
  echo "[T1] fix flags in the DSO build command: $(grep -ohE '\-DMPK_(PLACE_GUARD|PLACE_SERIAL|GATE_RESCUE)(=[0-9]+)?' $PH/$CFG/build_hipcc/build_cmd.txt | sort -u | tr '\n' ' ')"
  # the dynamic LDS rings start right after the static block: a static size off 16-byte alignment runs 1.9x slower
  LDS=$(awk '/^[ \t]*\.amdhsa_kernel /{k=$2} /\.amdhsa_group_segment_fixed_size/{if (k ~ /worker_kernel/) print $2}' \
        $PH/$CFG/build_hipcc/kernel-hip-amdgcn-amd-amdhsa-gfx950.s | head -n 1)
  echo "[T1] worker static LDS $CFG: ${LDS:-?} B"
  [ -n "$LDS" ] && [ $((LDS % 16)) = 0 ] || { echo "[T1] worker static LDS is not a multiple of 16: stop"; exit 1; }
done

# ---- 2. served A (original plugin) then B (libAiterMK), per config
port=8091
for CFG in $CFGS; do
  for arm in A B; do
    quiet || { echo "[T1] busy before $CFG $arm"; exit 1; }
    case $arm in
      A)
        docker exec $C pip uninstall -y libaitermk-vllm > /dev/null 2>&1
        SV="MPK_TOK_HOST=1 MPK_WRAP=/root/user/nps1/vllmn/wrap_sess.sh SESS_SCRIPT=$CSD/serve128.py SESS_PRE=g128"
        SV="$SV SESS_CFG=$CFG SESS_N=$N SESS_NEW=256 SESS_OUT=$RC/${CFG}_A.json"
        SV="$SV FLEET_NATIVE_DUMP=$RC/dump_${CFG}_A VN_UTIL=0.9 VN_PORT=$port $FIX"
        case $CFG in
          ns)  LANE=$LN PROMPT=$PD/q127.txt TO=2400 PS="$SV" bash $M/s15/prod.sh t1s_${CFG}_A 264 > /dev/null 2>&1 ;;
          e3)  LANE=$LN DRAFTER=eagle3 PROMPT=$PD/q127.txt T=6 K=5 XF="$BEST" PS="$BENV $SV" TO=2400 \
                 bash $M/s15/run.sh t1s_${CFG}_A 264 > /dev/null 2>&1 ;;
          df8) LANE=$LN DRAFTER=dflash PROMPT=$PD/q127.txt T=8 K=7 XF="$A" PS="MPK_TAIL_PUB_LATE=1 $SV" TO=2400 \
                 bash $M/s15/run.sh t1s_${CFG}_A 264 > /dev/null 2>&1 ;;
        esac
        LOG=$FB/t1s_${CFG}_A.log
        SUM=$(grep -a '\[SERVE128\] SUMMARY' $LOG | tail -n 1 | cut -c1-330) ;;
      B)
        docker exec $C pip install --no-deps -e $PKG > $L/t1_pip.out 2>&1 || \
          docker exec $C pip install --no-deps --no-build-isolation -e $PKG >> $L/t1_pip.out 2>&1
        docker exec $C pip list 2>/dev/null | grep -qi libaitermk-vllm || \
          { echo "[T1] libaitermk-vllm install failed"; tail -n 5 $L/t1_pip.out; exit 1; }
        SO=$(ls $PH/$CFG/build_hipcc/libaitermk_provider_gpt_oss_gfx950_fleet_mpk_*.so | head -n 1)
        SO=${SO/#\/home\/user/\/root\/user}
        LOG=$L/t1s_${CFG}_B.log
        docker exec -i $C bash -l > $LOG 2>&1 <<EOF
export HIP_VISIBLE_DEVICES=0,1
export MODEL_PATH=/root/user/models/gpt-oss-120b
export AITERMK_FLEET_MPK_PROVIDER=$SO
export AITERMK_FLEET_MPK_STATE=$PC/$CFG/state
export SESS_PRE=g128 SESS_CFG=$CFG SESS_N=$N SESS_NEW=256 SESS_OUT=$RC/${CFG}_B.json
export FLEET_NATIVE_DUMP=$RC/dump_${CFG}_B VN_UTIL=0.9 VN_PORT=$port
cd /root/user/mk1
python3 /root/user/mk1/scripts/serve128_aitermk.py
echo "exit \$?"
EOF
        SUM=$(grep -a 'SERVE128AK\] SUMMARY' $LOG | tail -n 1 | cut -c1-330) ;;
    esac
    port=$((port + 1))
    echo "[T1] $CFG $arm $(date -u +%T) hangmarks=$(grep -ac 'NOT DONE at 15s' $LOG) placelog=$(grep -ac '\[PLACE\]' $LOG) $(python3 $S/devall.py $LOG 2>/dev/null) | $SUM"
  done
done
docker exec $C pip install --no-deps -e $PKG > /dev/null 2>&1   # leave libAiterMK's package installed
echo "AGENT_LOOP_WAKE_T1TABLE $(date -u +%T)"

