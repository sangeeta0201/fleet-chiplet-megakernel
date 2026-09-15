#!/bin/bash
# One arm of the o_proj K-split A/B, at the 1024/1024 latency protocol.
#
#   ARM=control ./run_ksplit_ab.sh
#   ARM=ceil2   ./run_ksplit_ab.sh      # MPK_OPROJ_KSPLIT_CEIL=2
#
# Each invocation rebuilds (the flag is compile-time), runs 1024/1024 on
# devices 4-7 at NP=4, snapshots the rank-0 device code object for the ISA
# gate, and leaves the token dump for the generated-text check.
#
# STALL_SECS. The default 400 KILLS HEALTHY RUNS on this box. A 1024/1024 run
# prefills one token per iteration at ~307 ms/iter -- 314 s -- and the whole
# prefill+decode is one kernel whose [FWD_PASS] print is deferred until mpk()
# returns, so the log is silent for ~330 s end to end. stall_watchdog.sh is
# supposed to cover that with HBM-controller liveness, but every
# /sys/class/drm/card*/device/mem_busy_percent on this host reads 0 even at
# 94% GPU busy, so the HBM arm never fires and the watchdog degenerates to
# pure log growth with a 400 s threshold against a 330 s silent window. Three
# consecutive healthy runs were killed at exactly 400 s before this was found.
#
# Cleanup is not optional: a SIGKILLed rank leaves demo.py/mpirun respawning
# and 180-250 GB/GPU of VRAM held, and the next run then dies in hipMalloc and
# looks like a kernel bug.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

ARM="${ARM:-control}"
REP="${REP:-1}"
TAG="${ARM}_r${REP}"
RESULTS="${RESULTS:-/tmp/ksplit_ab}"
mkdir -p "$RESULTS"

ulimit -c 0

case "$ARM" in
  control) unset MPK_OPROJ_KSPLIT_CEIL ;;
  ceil1)   export MPK_OPROJ_KSPLIT_CEIL=1 ;;
  ceil2)   export MPK_OPROJ_KSPLIT_CEIL=2 ;;
  *) echo "unknown ARM=$ARM"; exit 2 ;;
esac

export ISL="${ISL:-1024}"
export OSL="${OSL:-1024}"
export OUT_DIR="$RESULTS/$TAG"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-4,5,6,7}"
export NP="${NP:-4}"
export STALL_SECS="${STALL_SECS:-1200}"
export RETRIES="${RETRIES:-2}"
unset KEEP_BUILD

echo "########## ARM=$ARM REP=$REP  KSPLIT_CEIL=${MPK_OPROJ_KSPLIT_CEIL:-unset} ##########"
date -u +"%Y-%m-%dT%H:%M:%SZ"

./run_latency_1k1k.sh 2>&1 | tee "$RESULTS/$TAG.summary"
rc=${PIPESTATUS[0]}

# -- ISA snapshot, rank 0 --------------------------------------------------
ISA_DIR="$RESULTS/isa_$TAG"
rm -rf "$ISA_DIR"; mkdir -p "$ISA_DIR"
SO=$(ls permanent_output_dir_rank0/*.so 2>/dev/null | head -1)
if [ -n "$SO" ]; then
  cp "$SO" "$ISA_DIR/"
  cp permanent_output_dir_rank0/test.cu "$ISA_DIR/" 2>/dev/null
  ( cd "$ISA_DIR" && /opt/rocm/llvm/bin/llvm-objdump --offloading ./*.so >/dev/null 2>&1
    OBJ=$(ls ./*gfx950* 2>/dev/null | head -1)
    if [ -n "$OBJ" ]; then
      md5sum "$OBJ" > devobj.md5
      /opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx950 "$OBJ" > dev.s 2>&1
    fi )
  echo "[isa] $ISA_DIR/dev.s $(wc -l < "$ISA_DIR/dev.s" 2>/dev/null) lines"
else
  echo "[isa] no .so found"
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
