#!/bin/bash
# Run a gfx1250 binary under FFM-Lite. Takes an ABSOLUTE path to the binary.
#
# Unlike the root-level run_ffm.sh this does not require a sysroot unless the
# host glibc is < 2.38 (set SYSROOT in env.sh if so).
set -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Mandatory for anything persistent-kernel shaped. Without it FFM runs
# workgroups strictly serially in ascending blockIdx, so MPK's first rendezvous
# waits on workers that are not dispatched until it exits. Signature:
# elect=1, s_spin_in=1, s_spin_out=0, disp=0 forever.
export HSA_MODEL_ARGS="${HSA_MODEL_ARGS:-ffm_enable_time_slicing}"
source "$FFM_PKG/ffmlite_env.sh"
BIN="$(readlink -f "$1")"; shift
if [ -n "${SYSROOT:-}" ]; then
  LD="$SYSROOT/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2"
  exec "$LD" --library-path "$SYSROOT/lib/x86_64-linux-gnu:$SYSROOT/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH" "$BIN" "$@"
fi
exec "$BIN" "$@"
