#!/bin/bash
# run_default.sh [QUESTION ...]: the default-mode runs of the guide's section 4: the single stream on each GSM8K
# question for 768 tokens (stock SPX+NPS1, flags as deployed; default gsm8k_1 gsm8k_2 gsm8k_3), the placement
# reference for run_dpx.sh's ns runs. Labels TAG_<question>_def (TAG=lg). GPU $DEF_BUS must be in SPX/NPS1 on the
# stock driver. Environment: see specg.sh.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
TREE=$(cd "$HERE/../../../.." && pwd)
HOST_HOME=${HOST_HOME:-$HOME}; CTR_HOME=${CTR_HOME:-/root/$(id -un)}
TREE_CTR=$CTR_HOME${TREE#$HOST_HOME}
G=${GSM_OUT:-$HOST_HOME/mk1/specg_latest}; ST=$G/status_default.txt
DEF_LANE=${DEF_LANE:-fair2}; DEF_BUS=${DEF_BUS:-05}; DEF_CONTAINER=${DEF_CONTAINER:-fleet_ship2}
TAG=${TAG:-lg}
P=$TREE_CTR/scripts/nps2/spec/bench_prompts
mkdir -p $G
st() { echo "[GSM8K-DEF] $(date -u +%T) $*" | tee -a $ST; }
docker start $DEF_CONTAINER > /dev/null
SYS=/sys/bus/pci/devices/0000:$DEF_BUS:00.0
st "start $(hostname) $DEF_BUS $(cat $SYS/current_compute_partition)/$(cat $SYS/current_memory_partition) driver $(cat /sys/module/amdgpu/srcversion) lane $DEF_LANE $(git -C $HOST_HOME/fleet-lanes/$DEF_LANE log -1 --format=%h)"
for p in ${*:-gsm8k_1 gsm8k_2 gsm8k_3}; do
  PROMPT=$P/$p.txt bash $HERE/specg.sh def 768 ${TAG}_${p}_def | tee -a $ST
done
st "done"

