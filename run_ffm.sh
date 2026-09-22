#!/usr/bin/env bash
# Run a gfx1250 binary under FFM-Lite.
#
# The FFM package requires glibc >= 2.38; this host is Ubuntu 22.04 (2.35).
# Rather than upgrade the host glibc -- which would break the system -- we run
# the binary against an isolated Ubuntu 24.04 sysroot's loader. Nothing about
# the host is modified.
#
#   ./run_ffm.sh /tmp/ffmrun/test_gemm_wmma
#
# Build binaries against the FFM package's own HIP runtime so the model
# intercepts dispatches:
#   clang++ -x hip --offload-arch=gfx1250 --rocm-path=$TOOLCHAIN ... \
#       -L$FFM_PKG/rocm -lamdhip64 -Wl,-rpath,$FFM_PKG/rocm
set -euo pipefail

FFM_PKG="${FFM_PKG:-/home/claudeuser/rocdtif-10.1-am+ffmlite-mi400-r9.03}"
SYSROOT="${SYSROOT:-/home/claudeuser/noble-sysroot}"

[ -f "$FFM_PKG/ffmlite_env.sh" ] || { echo "no FFM package at $FFM_PKG" >&2; exit 1; }
[ -x "$SYSROOT/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2" ] || {
  echo "no noble sysroot at $SYSROOT -- rebuild with:" >&2
  echo "  sudo debootstrap --variant=minbase \\" >&2
  echo "    --include=libstdc++6,libgcc-s1,zlib1g,libnuma1,libdrm2,libdrm-amdgpu1 \\" >&2
  echo "    noble $SYSROOT http://archive.ubuntu.com/ubuntu/" >&2
  exit 1; }

# shellcheck disable=SC1091
source "$FFM_PKG/ffmlite_env.sh"

# Time slicing is mandatory for anything persistent-kernel shaped. Without it
# FFM runs workgroups strictly serially in ascending blockIdx, so MPK's very
# first rendezvous -- execute_scheduler spinning on worker_xcd_ready_count --
# waits on workers that will not be dispatched until it exits. The breadcrumb
# signature is unmistakable: elect=1, s_spin_in=1, s_spin_out=0, disp=0 forever.
# ffmlite_env.sh only mentions this in a comment, so set it here rather than
# relying on every caller to remember. Export-if-unset so a caller can still
# override (e.g. to reproduce the serial behaviour deliberately).
export HSA_MODEL_ARGS="${HSA_MODEL_ARGS:-ffm_enable_time_slicing}"

LD="$SYSROOT/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2"
LP="$SYSROOT/lib/x86_64-linux-gnu:$SYSROOT/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# Must be an absolute path: the loader resolves argv[0] itself and a bare
# relative name fails with a misleading "cannot open shared object file".
BIN="$(readlink -f "$1")"; shift
exec "$LD" --library-path "$LP" "$BIN" "$@"
