#!/bin/bash
# Build drive_phase7 with extra defines, into a chosen binary name.
#
# Derived from build_drive_aid.sh by substitution rather than restating the flag
# list: the define set selects which code path compiles, so a copy that drifts
# would silently measure something else.
#
#   build_gate.sh ""                              drive_phase7
#   build_gate.sh "-DMPK_OPROJ_SKIP_HIER_POLL" drive_phase7_nopoll
set -u
cd "$HOME/fleet-chiplet-megakernel" || exit 1

EXTRA="${1:-}"
OUT="${2:-drive_phase7}"
CMD=/tmp/build_gate_cmd.sh

sed "s| -o drive_phase7\$| ${EXTRA} -o ${OUT}|" "$HOME/build_drive_aid.sh" >"$CMD"
if ! grep -q -- "-o ${OUT}\$" "$CMD"; then
	echo "ABORT: could not rewrite the output name; build_drive_aid.sh changed shape"
	exit 1
fi

rm -f "./${OUT}"
echo "=== compiling ${OUT} ${EXTRA} ==="
time bash "$CMD" 2>&1 | tail -25
rc=${PIPESTATUS[0]}
echo "hipcc rc=$rc"

if [ ! -x "./${OUT}" ]; then
	echo "ABORT: no binary produced"
	exit 1
fi
ls -la --time-style=+%H:%M:%S "./${OUT}"
