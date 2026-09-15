#!/bin/bash
# One arm of the fused-collective A/B, on devices 0-3, isolated from a
# concurrent campaign on devices 4-7.
#
#   ARM=control                                  ./run_collfuse_arm.sh
#   ARM=prohoist ARM_ENV="MPK_QKV_PRO_HOIST=1"   ./run_collfuse_arm.sh
#
# This is run_flag_ab.sh with three changes, all forced by sharing one
# container with another agent's A/B. None of them touch the build.
#
#   1. CLEANUP IS SCOPED. run_flag_ab.sh ends in `pkill -9 -f '[d]emo.py'` and
#      `pkill -9 -f '[m]pirun -np'`, which are global: run it while another
#      campaign is mid-decode and you kill that campaign's run. It happened
#      once in this session. This kills only processes in MY isolated tree,
#      which invokes `g5run.py` (a copy of demo.py) under `mpirun -n`, so
#      neither of the other campaign's two patterns can match it either.
#   2. VRAM IS CHECKED ON THE DEVICES IN USE. run_flag_ab.sh greps `GPU[4-7]`
#      unconditionally, so on 0-3 it reports the OTHER campaign's memory and
#      would clear a still-allocated device.
#   3. ALL FOUR RANKS ARE PINNED TO SOCKET 0. run_mp8_dp_ep_fused.sh only
#      builds a single-socket rankfile when HIP_VISIBLE_DEVICES is exactly
#      "4,5,6,7"; on 0-3 it falls through to `--map-by ppr:2:numa`, which
#      splits the ranks across both sockets even though all of GPUs 0-3 hang
#      off NUMA 0 (PCI 05,15,65,75). Both arms get the same pinning, so the
#      DELTA is valid; ABSOLUTES ARE NOT comparable to the 14.0 ms/token
#      canonical baseline, which was taken on 4-7.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

ARM="${ARM:-control}"
REP="${REP:-1}"
ARM_ENV="${ARM_ENV:-}"
TAG="${ARM}_r${REP}"
RESULTS="${RESULTS:-/mnt/nvme1/glm5_collfuse_ab}"
mkdir -p "$RESULTS"

ulimit -c 0

# Every knob this campaign touches, cleared unconditionally, so an arm can
# never inherit the previous arm's setting.
unset MPK_QKV_PRO_HOIST MPK_QKV_EP_FOLD MPK_ABL_QKV_PRO MPK_ABL_QKV
unset GLM_RESADD_BATCH GLM_RESADD_UNROLL GLM_RESADD_GLOBAL

for _kv in $ARM_ENV; do
  export "${_kv?}"
done
unset _kv

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3}"
export NP="${NP:-4}"
export ISL="${ISL:-1024}"
export OSL="${OSL:-1024}"
export OUT_DIR="$RESULTS/$TAG"
export STALL_SECS="${STALL_SECS:-900}"
export RETRIES="${RETRIES:-3}"
export MODEL_PATH="${MODEL_PATH:-/mnt/nvme1/GLM-5.2-MXFP4}"
unset KEEP_BUILD

_RF="/tmp/collfuse_rf_s0_$$.txt"
: > "$_RF"
for _r in $(seq 0 $((NP - 1))); do
  echo "rank $_r=$(hostname) slot=0:*" >> "$_RF"
done
export MPK_MPI_BIND="--rankfile $_RF"
trap 'rm -f "$_RF"' EXIT

echo "########## ARM=$ARM REP=$REP  ARM_ENV='${ARM_ENV}' ##########"
echo "[collfuse] devices=$HIP_VISIBLE_DEVICES bind='$MPK_MPI_BIND'"
date -u +"%Y-%m-%dT%H:%M:%SZ"

./run_latency_1k1k.sh 2>&1 | tee "$RESULTS/$TAG.summary"
rc=${PIPESTATUS[0]}

# -- ISA snapshot, rank 0 --------------------------------------------------
if [ "${KEEP_ISA:-1}" = "1" ]; then
  ISA_DIR="$RESULTS/isa_$TAG"
  rm -rf "$ISA_DIR"; mkdir -p "$ISA_DIR"
  SO=$(ls permanent_output_dir_rank0/*.so 2>/dev/null | head -1)
  if [ -n "$SO" ]; then
    cp "$SO" "$ISA_DIR/"
    ( cd "$ISA_DIR" && /opt/rocm/llvm/bin/llvm-objdump --offloading ./*.so \
        >/dev/null 2>&1
      OBJ=$(ls ./*gfx950* 2>/dev/null | head -1)
      if [ -n "$OBJ" ]; then
        md5sum "$OBJ" > devobj.md5
        # Keep the code object itself: the ISA gate diffs two of these, and
        # rebuilding an arm just to disassemble it is 6 minutes.
        /opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx950 "$OBJ" > dev.s 2>/dev/null
        gzip -f dev.s
        rm -f ./*.so
      fi )
    echo "[isa] $(cat "$ISA_DIR/devobj.md5" 2>/dev/null)"
  else
    echo "[isa] no .so found"
  fi
fi

# -- cleanup, scoped to this tree -----------------------------------------
pkill -9 -f '[g]5run.py' 2>/dev/null
pkill -9 -f '[m]pirun -n .*g5run' 2>/dev/null
sleep 6
pkill -9 -f '[g]5run.py' 2>/dev/null
sleep 4

_dev_re=$(echo "$HIP_VISIBLE_DEVICES" | tr -d ',')
echo "[vram] after $TAG (devices $HIP_VISIBLE_DEVICES):"
for _ in $(seq 1 30); do
  used=$(rocm-smi --showmeminfo vram 2>/dev/null |
         awk -v re="GPU\\\\[[${_dev_re}]\\\\].*Used" \
             '$0 ~ re {s+=$NF} END {print int(s/1073741824)}')
  [ "${used:-99}" -lt 8 ] && break
  sleep 5
done
rocm-smi --showmeminfo vram 2>/dev/null | grep -E "GPU\[[${_dev_re}]\].*Used" || true
df -h / /mnt/nvme1 | tail -2

exit "$rc"
