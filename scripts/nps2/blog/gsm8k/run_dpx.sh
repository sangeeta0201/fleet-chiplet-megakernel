#!/bin/bash
# run_dpx.sh [QUESTION ...]: the DPX+NPS2 runs of the guide's section 4 (GSM8K speculation and the 1K conversation),
# with the method of Oct 13 2026. Per GSM8K question (default gsm8k_1 gsm8k_2 gsm8k_3): ns and ns7 for 768 tokens;
# E = where each answer ends (first <|return|>); then k1 for ns's E tokens (it must reproduce ns) and k3 for ns7's E
# tokens (it must reproduce ns7). Then the 1K conversation: ns7 and k3 for all 1,024 tokens (CONV=0 skips it; the
# single stream and 1 draft at 1K/1024 are section 2's runs). Labels: TAG_<question>_<kind> (TAG=lg) and
# CTAG_1k_ns7 / CTAG_1k_k3x (CTAG=lc). GPU $DPX_BUS must already be in DPX+NPS2 on the stock driver. Waits while a
# process outside OWN_CONTAINERS (default $DPX_CONTAINER) uses a GPU. Environment: see specg.sh.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
TREE=$(cd "$HERE/../../../.." && pwd)
HOST_HOME=${HOST_HOME:-$HOME}; CTR_HOME=${CTR_HOME:-/root/$(id -un)}
TREE_CTR=$CTR_HOME${TREE#$HOST_HOME}
G=${GSM_OUT:-$HOST_HOME/mk1/specg_latest}; ST=$G/status_dpx.txt
DPX_LANE=${DPX_LANE:-fair}; DPX_BUS=${DPX_BUS:-e5}; DPX_CONTAINER=${DPX_CONTAINER:-fleet_ak4t1}
TAG=${TAG:-lg}; CTAG=${CTAG:-lc}; CONV=${CONV:-1}
P=$TREE_CTR/scripts/nps2/spec/bench_prompts
mkdir -p $G
st() { echo "[GSM8K-DPX] $(date -u +%T) $*" | tee -a $ST; }
MINE=""; for c in ${OWN_CONTAINERS:-$DPX_CONTAINER}; do MINE="$MINE $(docker inspect -f '{{.Id}}' $c 2> /dev/null)"; done
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
wait_idle() { local f; while true; do f=$(foreign); [ -z "$f" ] && return 0; st "waiting: $(echo $f | cut -c1-160)"; sleep 30; done; }
eos() { python3 -c "
import json
t=json.load(open('$G/tok_$1.json'))['token_ids']
print(t.index(200002)+1 if 200002 in t else len(t))" 2> /dev/null || echo 0; }
one() { wait_idle; bash $HERE/specg.sh "$@" | tee -a $ST; }
docker start $DPX_CONTAINER > /dev/null
SYS=/sys/bus/pci/devices/0000:$DPX_BUS:00.0
st "start $(hostname) $DPX_BUS $(cat $SYS/current_compute_partition)/$(cat $SYS/current_memory_partition) driver $(cat /sys/module/amdgpu/srcversion) lane $DPX_LANE $(git -C $HOST_HOME/fleet-lanes/$DPX_LANE log -1 --format=%h)"
for p in ${*:-gsm8k_1 gsm8k_2 gsm8k_3}; do
  export PROMPT=$P/$p.txt
  one ns 768 ${TAG}_${p}_ns
  one ns7 768 ${TAG}_${p}_ns7
  e8=$(eos ${TAG}_${p}_ns); e7=$(eos ${TAG}_${p}_ns7)
  [ "$e8" -gt 1 ] && REF=${TAG}_${p}_ns one k1 $e8 ${TAG}_${p}_k1
  [ "$e7" -gt 1 ] && REF=${TAG}_${p}_ns7 one k3 $e7 ${TAG}_${p}_k3
done
if [ "$CONV" = 1 ]; then
  export PROMPT=$TREE_CTR/docs/nps2/prompt_1k.txt MSL=2048
  one ns7 1024 ${CTAG}_1k_ns7
  REF=${CTAG}_1k_ns7 one k3 1024 ${CTAG}_1k_k3x
fi
st "done"

