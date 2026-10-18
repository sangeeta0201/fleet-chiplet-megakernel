#!/bin/bash
# Isolated lane runner. One tree, one GPU. Does NOT pkill other lanes.
# Does NOT wrap docker exec in timeout (that kills the client, not demo.py).
set -euo pipefail
LANE=${1:?usage: fleet_run_lane.sh <lane> <label> [demo args]}
LABEL=${2:-$LANE}
shift 2 || true
# HOST_HOME: host directory holding fleet-lanes/<lane>; CTR_HOME: the same directory inside the container;
# FLEET_LOG_ROOT: where lane_<lane>/<label>.log go; MODEL_CTR: the model directory inside the container.
HOST_HOME=${HOST_HOME:-$HOME}
CTR_HOME=${CTR_HOME:-/root/$(id -un)}
FLEET_LOG_ROOT=${FLEET_LOG_ROOT:-$HOST_HOME/nps1/fleet_baselines}
MODEL_CTR=${MODEL_CTR:-$CTR_HOME/models/gpt-oss-120b}
SRC=$HOST_HOME/fleet-lanes/$LANE
[ -f "$SRC/LANE.env" ] || { echo "missing $SRC/LANE.env"; exit 2; }
# shellcheck disable=SC1091
. "$SRC/LANE.env"

FLEET_HASH="${FLEET_HASH:-3d54adb71e19}"
KNOWN="MPK_TERM_RECHECK=1 MPK_NARROW_GATE_POLL=1 MPK_AID_SPLIT_FLAGS=1 \
MPK_AID_SPLIT_OUT=1 MPK_MOE_XCD_PAIR=0 MPK_MOE_REPLICA=1 MPK_AID_EVCTR2=1 \
MPK_SWIGLU_AID=1 MPK_LMNORM_AID=1 MPK_MOENORM_AID=1 MPK_PRENORM_AID=1 \
MPK_GAMMA_AID=1"
ENVS="${LANE_ENVS:-$KNOWN} ${EXTRA_ENVS:-}"
TREE=$CTR_HOME/fleet-lanes/$LANE
LOGDIR=$FLEET_LOG_ROOT/lane_$LANE
mkdir -p "$LOGDIR" /tmp
LOG=/tmp/fleet_lane_${LANE}_${LABEL}.log
HOSTLOG=$LOGDIR/${LABEL}.log

{
  echo "######## lane=$LANE label=$LABEL $(date -u +%FT%TZ) ########"
  echo "TREE $TREE"
  echo "CONTAINER ${CONTAINER:-fleet_v1} PATH_PREPEND ${PATH_PREPEND:-}"
  echo "HIP_VISIBLE_DEVICES $HIP_VISIBLE_DEVICES"
  echo "ENVS $ENVS"
  echo "GIT $(git -C $SRC rev-parse --abbrev-ref HEAD) $(git -C $SRC rev-parse --short HEAD)"
  echo "DIRTY $(git -C $SRC status --porcelain -- include/ python/ | tr '\n' ' ')"
  spx=0; nps2=0
  for b in 05 15 65 75 85 95 e5 f5; do
    d=/sys/bus/pci/devices/0000:$b:00.0
    [ "$(cat $d/current_compute_partition 2>/dev/null)" = SPX ] && spx=$((spx+1))
    [ "$(cat $d/current_memory_partition 2>/dev/null)" = NPS2 ] && nps2=$((nps2+1))
  done
  echo "MODE spx=$spx/8 nps2=$nps2/8"
} > "$LOG"

C=${CONTAINER:-fleet_v1}  # container; PATH_PREPEND = dir put first on PATH inside it
PP=""
[ -n "${PATH_PREPEND:-}" ] && PP="[ -d $PATH_PREPEND ] || { echo PATH_PREPEND_MISSING $PATH_PREPEND; exit 9; }; export PATH=$PATH_PREPEND:\$PATH"
docker start $C >/dev/null

# Kill leftovers whose cwd is THIS lane only (never pkill -f demo.py).
docker exec $C bash -lc "
  for pid in \$(ps -eo pid,cmd | awk '/python.*demo.py/ {print \$1}'); do
    cwd=\$(readlink /proc/\$pid/cwd 2>/dev/null || true)
    case \$cwd in
      *$LANE*) kill -TERM \$pid 2>/dev/null || true ;;
    esac
  done
" || true

# Inner watchdog: kill THIS demo after TIMEOUT seconds, from inside the container.
TO=${TIMEOUT:-480}
docker exec $C bash -lc "
  cd $TREE/demo/gpt_oss || exit 1
  export MODEL_PATH=$MODEL_CTR
  export KEEP_BUILD=\${KEEP_BUILD:-0}
  export MAX_SEQ_LENGTH=\${MAX_SEQ_LENGTH:-128}
  export MAX_NEW_TOKENS=\${MAX_NEW_TOKENS:-16}
  export HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES
  export $ENVS
  $PP
  # inner watchdog
  ( sleep $TO; ps -eo pid,cmd | awk '/demo.py/ {print \$1}' | while read p; do
      tr '\\0' '\\n' < /proc/\$p/environ 2>/dev/null | grep -qx HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES && kill -9 \$p
    done ) >/dev/null 2>&1 &
  wd=\$!
  bash run_1gpu.sh $*
  rc=\$?
  kill \$wd 2>/dev/null || true
  echo WRAPPER_RC=\$rc
  exit \$rc
" >> "$LOG" 2>&1
rc=$?
echo "RC=$rc" >> "$LOG"
if grep -qE 'ImportError|Traceback' "$LOG"; then
  echo "LANE_PYTHON_FAIL" >> "$LOG"
  rc=1
fi
if grep -q '^EXIT=1' "$LOG"; then
  rc=1
fi
cp -f "$LOG" "$HOSTLOG"
# archive
cp -f "$LOG" "$FLEET_LOG_ROOT/fleet_lane_${LANE}_${LABEL}.log" 2>/dev/null || true

hash=$(grep -a assistantanalysis "$LOG" | tail -1 | tr -d '\r\n' | md5sum | awk '{print $1}')
echo "LANE=$LANE HASH=$hash EXPECT=$FLEET_HASH"
grep -E 'FWD_PASS_TOTAL|TIMEOUT|EXIT=|MODE |GIT |ENVS ' "$LOG" | tail -20
echo "$hash" > "$LOGDIR/${LABEL}.hash"
# Full generated text: the line hash above covers one line of a multi-line output.
python3 - "$LOG" > "$LOGDIR/${LABEL}.texthash" 2>/dev/null << 'PYTXT'
import hashlib, re, sys
s = open(sys.argv[1], "rb").read().decode("utf-8", "replace").replace("\r", "")
i = s.find("assistantanalysis")
t = s[i:] if i >= 0 else ""
m = re.search(r"\n(\[[A-Z_]+\]|Decode:|Saved tokens)", t)
t = t[: m.start()] if m else t
print(hashlib.md5(t.encode()).hexdigest()[:12], len(t))
PYTXT
if echo "$hash" | grep -q "^${FLEET_HASH}"; then
  echo LANE_HASH_OK
  exit 0
fi
echo LANE_HASH_FAIL
exit 3

