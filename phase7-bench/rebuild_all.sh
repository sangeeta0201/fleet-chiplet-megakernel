#!/bin/bash
# Rebuild the three gate binaries from the current sources.
#
# All three must come from the same source state, or a hash difference between
# them could be the source drift rather than the barrier under test.
set -u
cd "$HOME" || exit 1

bash build_gate.sh "" drive_phase7 2>&1 | tail -2
bash build_gate.sh "-DMPK_OPROJ_SKIP_HIER_POLL" drive_phase7_nopoll 2>&1 | tail -2
bash build_gate.sh "-DMPK_OPROJ_SKIP_SLICE_POLL" drive_phase7_noslice 2>&1 | tail -2
