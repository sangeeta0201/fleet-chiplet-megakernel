#!/bin/bash
# Run a gfx1250 binary under AM (cycle-accurate). ABSOLUTE path required.
#
#   AM_ENV       am_profile_env.sh (per-dispatch counters, 1 XCC)  [default]
#                am_fast_env.sh    (raw counters only, 1 XCC)
#                am_8xcc_env.sh    (8 XCC, DtifNumXcc=8)
#   AM_RUNDIR    scratch dir; AM writes a lot of files into $CWD
#   AM_SINGLE_THREAD=1  disable AM_CLOCK_MT + wgp/sa/se MT (troubleshooting)
#
# Deliberately NOT `set -u`: the packaged env scripts reference unset variables
# and would exit mid-source, killing this wrapper with no output at all.
set -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
AM_ENV="${AM_ENV:-am_profile_env.sh}"
[ -f "$FFM_PKG/$AM_ENV" ] || { echo "no $AM_ENV in $FFM_PKG" >&2; exit 1; }
command -v m4 >/dev/null || { echo "m4 missing: AM cannot preprocess model.conf and aborts with exit 32512" >&2; exit 1; }
RUNDIR="${AM_RUNDIR:-/tmp/am_run}"; mkdir -p "$RUNDIR"; cd "$RUNDIR"
source "$FFM_PKG/$AM_ENV" >/dev/null 2>&1
if [ "${AM_SINGLE_THREAD:-0}" = "1" ]; then
  export AM_CLOCK_MT=0
  # These live inside the -pm4p2_args blob, so rewrite rather than append: a
  # later duplicate key is not guaranteed to win.
  DtifGeneralArgs="${DtifGeneralArgs//test.enable_wgp_mt=true/test.enable_wgp_mt=false}"
  DtifGeneralArgs="${DtifGeneralArgs//test.enable_sa_mt=true/test.enable_sa_mt=false}"
  DtifGeneralArgs="${DtifGeneralArgs//test.enable_se_mt=true/test.enable_se_mt=false}"
  export DtifGeneralArgs
fi
BIN="$(readlink -f "$1")"; shift
echo "[runam] env=$AM_ENV NumXcc=${DtifNumXcc:-default} MT=${AM_CLOCK_MT:-1} cwd=$RUNDIR" >&2
exec "$BIN" "$@"
