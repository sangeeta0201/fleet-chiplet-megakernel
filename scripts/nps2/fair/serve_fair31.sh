#!/bin/bash
# serve_fair31.sh: the served GSM8K-128 sweep behind BLOG_REPRODUCTION.md section 1's BF16-head rows (libAiterMK),
# on vLLM 0.31.0. fair_t1.sh's loop with the lane pointed at the vllm/vllm-openai-rocm:v0.31.0 container $C31:
# per configuration (ns, e3 = EAGLE3 block 6 / 5 drafts, df8 = DFlash block 8 / 7 drafts) the arms BF16, MXFP4, BF16
# (f31_<cfg>_bf1, _mx1, _bf2), each a GSM8K-sized build (the longest prompt q127 + 264 tokens) served through the
# megakernel's own plugin (vllm_integration/, via scripts/nps2/spec/s15/gsm128/serve128.py). Runs only the arms whose
# log has no [SERVE128] SUMMARY yet, so it can be rerun after a failure. Results: $RC (JSON per arm, token dumps),
# logs in $LD; tables: blog/blog_tab.py --served-bf $RC --served-bf-logs $LD.
# Needs: GPU e5 in DPX+NPS2 on the stock driver (never switched here); $C31 with mirage's libz3 on its library path
# and `pip install graphviz` (the original plugin imports mirage's extension); the s15 run scripts and lanes.conf
# under ~/mk1/s15 (copies of scripts/nps2/aitermk/s15/); libaitermk-vllm is uninstalled from $C31 while this runs
# (both packages publish the vLLM entry point fleet_mk) and reinstalled at the end.
set -u
N=/home/user/nps1; LD=$N/fleet_baselines/lane_fair
M=/home/user/mk1; ST=$M/fp6/status_serve_fair31.txt; C=fleet_ak4t1; C31=fleet_ak31
st() { echo "[F31] $(date -u +%T) $*" | tee -a $ST; }
E=/sys/bus/pci/devices/0000:e5:00.0
mode() { echo "$(cat $E/current_compute_partition)/$(cat $E/current_memory_partition)"; }
docker start $C $C31 > /dev/null
MINE="$(docker inspect -f '{{.Id}}' $C) $(docker inspect -f '{{.Id}}' $C31)"
foreign() {
  local p c id m
  for p in $(ls /sys/class/kfd/kfd/proc 2> /dev/null); do
    c=$(tr '\0' ' ' < /proc/$p/cmdline 2> /dev/null | cut -c1-90)
    case "$c" in *omnistat*) continue ;; esac
    m=0; for id in $MINE; do grep -q "$id" /proc/$p/cgroup 2> /dev/null && m=1; done
    [ $m = 1 ] && continue
    echo "$p:$c;"
  done
}
wait_idle() {
  local f last=""
  while true; do
    [ "$(mode)" = DPX/NPS2 ] && [ "$(ls /sys/module/amdgpu/parameters | grep -c aid_local)" = 0 ] \
      || { st "e5 is $(mode) / not stock: stopping (this script never switches modes)"; exit 1; }
    f=$(foreign)
    [ -z "$f" ] && return 0
    [ "$f" != "$last" ] && st "waiting, other GPU users: $(echo $f | cut -c1-240)"
    last=$f; sleep 30
  done
}
docker exec $C31 python3 -c "import mirage.core" < /dev/null > /dev/null 2>&1 \
  || docker exec -w /root/user/fleet-lanes/fair/python $C31 python3 -c "import mirage.core" < /dev/null > /dev/null 2>&1 \
  || { st "$C31 cannot import mirage.core (libz3 / graphviz): stopping"; exit 1; }
st "start: the f31 served runs without a SUMMARY"
LC=$M/s15/lanes.conf
sed -i 's/^fair  12,13  e5  fleet_ak4t1$/fair  12,13  e5  fleet_ak31/' $LC
grep '^fair ' $LC | tee -a $ST
AD=/root/user/mk1/libAiterMK/libAiterMK/engine_adapters/vllm
docker exec $C31 pip uninstall -y libaitermk-vllm > /dev/null 2>&1 < /dev/null
PD=/root/user/nps1/opt/spec/bench_prompts/gsm128
CSD=/root/user/fleet-lanes/stk15/scripts/nps2/spec/s15/gsm128
LDS=/home/user/fleet-lanes/stk15/scripts/nps2/spec/s15/lead
SBEST="$(cat $LDS/best.flags)"; SBENV="$(cat $LDS/best.env)"
SA=$(echo "$SBEST" | sed -E 's/,-DMPK_ATTN_FIXED_MERGER(,|$)/\1/; s/,-DMPK_ATTN_FM_V=[0-9]+(,|$)/\1/')
FIX="MPK_PLACE_GUARD=1000 MPK_PLACE_SERIAL=1 MPK_GATE_RESCUE=16384"
BFX="MPK_LM_BF16=1 MPK_LM_PF0=0 MPK_LM_PIPE_S=0 MPK_LM_HEAD_GROUP_PIPELINE=0 MPK_LM_HEAD_KMAJOR=0 MPK_LM_HEAD_WAVE_TILE_DMA=0 MPK_LMT_FQ=0"
RC=/root/user/mk1/results/serve_fair31; mkdir -p /home/user/mk1/results/serve_fair31
port=8231
for CFG in ns e3 df8; do
  for r in bf1 mx1 bf2; do
    arm=${r%[0-9]}; lab=f31_${CFG}_$r
    if grep -aq '\[SERVE128\] SUMMARY' $LD/$lab.log 2> /dev/null; then st "$lab: has a SUMMARY, kept"; continue; fi
    for try in 1 2; do
      wait_idle
      SV="MPK_TOK_HOST=1 MPK_WRAP=/root/user/nps1/vllmn/wrap_sess.sh SESS_SCRIPT=$CSD/serve128.py SESS_PRE=g128"
      SV="$SV SESS_CFG=$CFG SESS_N=128 SESS_NEW=256 SESS_OUT=$RC/$lab.json FLEET_NATIVE_DUMP=$RC/dump_$lab"
      SV="$SV VN_UTIL=0.9 VN_PORT=$port $FIX"; XB=""
      [ $arm = bf ] && { SV="$SV $BFX"; XB=",-DMPK_LM_BF16"; }
      case $CFG in
        ns)  LANE=fair PROMPT=$PD/q127.txt TO=2400 XF="$XB" PS="$SV" bash $M/s15/prod.sh $lab 264 > /dev/null 2>&1 ;;
        e3)  LANE=fair DRAFTER=eagle3 PROMPT=$PD/q127.txt T=6 K=5 XF="$SBEST$XB" PS="$SBENV $SV" TO=2400 \
               bash $M/s15/run.sh $lab 264 > /dev/null 2>&1 ;;
        df8) LANE=fair DRAFTER=dflash PROMPT=$PD/q127.txt T=8 K=7 XF="$SA$XB" PS="MPK_TAIL_PUB_LATE=1 $SV" TO=2400 \
               bash $M/s15/run.sh $lab 264 > /dev/null 2>&1 ;;
      esac
      port=$((port + 1))
      grep -aq '\[SERVE128\] SUMMARY' $LD/$lab.log 2> /dev/null && break
    done
    st "$lab $(grep -a 'version 0\.[0-9]*\.[0-9]*' $LD/$lab.log | head -n 1 | grep -oE 'version 0\.[0-9.]+') | $(grep -a '\[SERVE128\] SUMMARY' $LD/$lab.log | tail -n 1 | grep -oE 'decode [0-9]+ tok/s|kernel arena [0-9]+/[0-9]+|[0-9]+ server restarts|identical to standalone [0-9]+/[0-9]+' | tr '\n' ' ')"
  done
done
docker exec $C31 pip install --no-deps -e $AD > /dev/null 2>&1 < /dev/null
st "libaitermk-vllm reinstalled in $C31: $(docker exec $C31 pip show libaitermk-vllm 2> /dev/null < /dev/null | grep -c Editable)"
sed -i 's/^fair  12,13  e5  fleet_ak31$/fair  12,13  e5  fleet_ak4t1/' $LC
grep '^fair ' $LC | tee -a $ST
st "done"

