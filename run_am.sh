#!/usr/bin/env bash
# Run a gfx1250 binary under AM -- the CYCLE-ACCURATE model -- in its 8-XCC
# config. This is the sibling of run_ffm.sh; the package ships both models and
# the port only ever used FFM-Lite.
#
# Use this, not FFM, when the question involves more than one XCC:
#   - mirage::arch::xcd_id(), whose s_sendmsg_rtn(0x87) FFM cannot even decode
#   - MoE cross-XCD tile distribution (tile_idx * 8 + xcd_id)
#   - anything where cycles matter at all
#
# Prerequisites (both cost real debugging time if missed):
#   sudo apt install m4      # else: "Failed to preprocess parameter file",
#                            # exit code 32512, then a fatal assert
#   the noble sysroot        # same glibc>=2.38 shim run_ffm.sh uses
#
# Expect it to be SLOW: roughly 4 minutes of wall clock just to reach the first
# dispatch, versus ~4 minutes for a whole e2e run under FFM. Budget accordingly
# and prefer small grids.
#
#   ./run_am.sh /abs/path/to/binary [args...]
#
# Env:
#   AM_ENV   which model config to source (default am_8xcc_env.sh)
#            am_env.sh        standard, perftools + thread trace
#            am_fast_env.sh   raw perf counter file only
#            am_profile_env.sh per-dispatch perf counters
#            am_8xcc_env.sh   8 XCC  <-- the reason this script exists

# NOTE: deliberately NOT `set -u`. The packaged env scripts reference unset
# variables, so `set -u` makes them exit mid-source and the wrapper dies with no
# output at all -- which reads as a launcher bug rather than a shell-option one.
set -o pipefail

PKG="${FFM_PKG:-/home/claudeuser/rocdtif-10.1-am+ffmlite-mi400-r9.03}"
SYSROOT="${SYSROOT:-/home/claudeuser/noble-sysroot}"
AM_ENV="${AM_ENV:-am_8xcc_env.sh}"

[ -f "$PKG/$AM_ENV" ] || { echo "no $AM_ENV in $PKG" >&2; exit 1; }
command -v m4 >/dev/null || {
  echo "m4 not installed -- AM cannot preprocess its model.conf and will abort" >&2
  echo "  sudo apt install m4" >&2; exit 1; }

# AM writes a lot of generated files into the CWD. The README is explicit about
# not running it from a directory you care about, so default to a scratch dir.
RUNDIR="${AM_RUNDIR:-/tmp/am_run}"
mkdir -p "$RUNDIR" && cd "$RUNDIR"

# shellcheck disable=SC1090
source "$PKG/$AM_ENV" >/dev/null 2>&1

# AM_DEBUG=1 turns on the model's own logging (AM_README section 8).
#
# This has to happen AFTER the source, not before. am_8xcc_env.sh line 101
# unconditionally exports DtifPrintCondLog=0 DtifPrintTerse=1, so anything you
# put in the environment ahead of the wrapper is silently overwritten and you
# get a completely quiet log while believing debug is on. The commented-out
# line 109 in that script is the intended recipe; this applies it.
#
# Expect a LOT of output (hundreds of MB on a real workload) and a further
# slowdown on a model that is already slow. Use it on one-block repros.
if [ "${AM_DEBUG:-0}" = "1" ]; then
  export DtifConsoleEnabled=1 DtifLogEnabled=1 DtifPm4LogEnabled=1
  export DtifPrintCondLog=1 DtifPrintTerse=0
  export DtifExtraTestArgs="${DtifExtraTestArgs:+$DtifExtraTestArgs }-gclog_enable=all,info-high"
  echo "[run_am] AM_DEBUG=1: verbose model logging enabled" >&2
fi

# AM_SINGLE_THREAD=1 disables AM's clock-domain and shader multithreading.
#
# Motivation: s_sendmsg_rtn hangs under the default 8-XCC config. That message
# needs a round trip -- SQ queues SQ_SPI_MSG_RTN, SQG's ProcessMsgRtnCommand
# answers via generate_and_send_sdata_rtn, and the wave resumes. Everything on
# that path is multithreaded by default (wgp/sa/se worker pools plus
# AM_CLOCK_MT), so a lost or unobserved reply stalls the wave with no error.
# This collapses the model to one thread to test that hypothesis. AM_README
# section 9 explicitly offers AM_CLOCK_MT=0 "for troubleshooting".
#
# Costs simulation throughput, so use it for repros, not for perf runs.
if [ "${AM_SINGLE_THREAD:-0}" = "1" ]; then
  export AM_CLOCK_MT=0
  # These live inside the -pm4p2_args blob, so rewrite rather than append: a
  # later duplicate key is not guaranteed to win.
  DtifGeneralArgs="${DtifGeneralArgs//test.enable_wgp_mt=true/test.enable_wgp_mt=false}"
  DtifGeneralArgs="${DtifGeneralArgs//test.enable_sa_mt=true/test.enable_sa_mt=false}"
  DtifGeneralArgs="${DtifGeneralArgs//test.enable_se_mt=true/test.enable_se_mt=false}"
  export DtifGeneralArgs
  echo "[run_am] AM_SINGLE_THREAD=1: AM_CLOCK_MT=0, wgp/sa/se MT disabled" >&2
fi

LD="$SYSROOT/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2"
LP="$SYSROOT/lib/x86_64-linux-gnu:$SYSROOT/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

BIN="$(readlink -f "$1")"; shift
echo "[run_am] $AM_ENV (DtifNumXcc=${DtifNumXcc:-?}) cwd=$RUNDIR" >&2
exec "$LD" --library-path "$LP" "$BIN" "$@"
