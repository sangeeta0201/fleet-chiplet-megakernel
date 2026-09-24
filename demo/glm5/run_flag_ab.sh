#!/bin/bash
# One arm of a generic compile-time-flag A/B, at the 1024/1024 latency protocol.
#
#   ARM=control                                   ./run_flag_ab.sh
#   ARM=mlptr   ARM_ENV="MPK_ML_PTR_PREFETCH=1"   ./run_flag_ab.sh
#   ARM=pf256   ARM_ENV="MPK_QKVA_PF_KB=256"      ./run_flag_ab.sh
#
# Generalisation of run_ksplit_ab.sh: same protocol, same cleanup, same ISA
# snapshot, but the arm is a list of KEY=VALUE assignments instead of one
# hardcoded knob. Every knob under test is unset first, so an arm can never
# inherit the previous arm's setting from the caller's shell -- that is the
# difference between a one-variable A/B and a mixed one.
#
# Each invocation rebuilds (the flags are compile-time, and run_latency_1k1k.sh
# removes permanent_output_dir* unless KEEP_BUILD=1).
#
# STALL_SECS defaults to 900: a 1024/1024 run is silent for ~330 s (one kernel
# for the whole prefill+decode, [FWD_PASS] deferred until mpk() returns) and
# every mem_busy_percent on this host reads 0, so the watchdog's HBM liveness
# arm never fires and it degenerates to log growth alone.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

ARM="${ARM:-control}"
REP="${REP:-1}"
ARM_ENV="${ARM_ENV:-}"
TAG="${ARM}_r${REP}"
RESULTS="${RESULTS:-/mnt/nvme1/glm5_flag_ab}"
mkdir -p "$RESULTS"

ulimit -c 0

# Every knob this campaign touches, cleared unconditionally.
unset MPK_ML_PTR_PREFETCH MPK_QKVA_PF_KB MPK_QKVA_PF_AT
unset MPK_ATTN_REL_NARROW MPK_OPROJ_KSPLIT_CEIL
unset MPK_QKV_PRO_HOIST MPK_COLL_FUSE_NORM
unset MPK_MLA_DECODE_DBLBUF MPK_MLA_DECODE_BAR_LDS MPK_RMSNORM_DPP
unset MPK_MLA_OACC_INTERLEAVE MPK_MLA_DECODE_DMA_PF
unset MPK_MERGE_TWO_PASS MPK_QKVA_ENTRY_PF MPK_BAR_SKEW MPK_BAR_SKEW_DROP_NS

for _kv in $ARM_ENV; do
  export "${_kv?}"
done
unset _kv

export ISL="${ISL:-1024}"
export OSL="${OSL:-1024}"
export OUT_DIR="$RESULTS/$TAG"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-4,5,6,7}"
export NP="${NP:-4}"
export STALL_SECS="${STALL_SECS:-900}"
export RETRIES="${RETRIES:-2}"
unset KEEP_BUILD

echo "########## ARM=$ARM REP=$REP  ARM_ENV='${ARM_ENV}' ##########"
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
        rm -f ./*.so
      fi )
    echo "[isa] $(cat "$ISA_DIR/devobj.md5" 2>/dev/null)"
  else
    echo "[isa] no .so found"
  fi
fi

# -- cleanup ---------------------------------------------------------------
pkill -9 -f '[d]emo.py' 2>/dev/null
pkill -9 -f '[m]pirun -np' 2>/dev/null
sleep 6
pkill -9 -f '[d]emo.py' 2>/dev/null
sleep 4
echo "[vram] after $TAG:"
rocm-smi --showmeminfo vram 2>/dev/null | grep -E "GPU\[[4-7]\].*Used" || true
df -h / | tail -1

exit "$rc"
