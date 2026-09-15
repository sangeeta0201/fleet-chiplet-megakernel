#!/bin/bash
# 3-arm fused-collective A/B on the CANONICAL devices 4-7, so absolutes tie
# back to the 14.0 ms/token baseline (no isolated tree: nothing else is
# editing .cuh, so the JIT-poison hazard the isolated tree existed for is gone).
#
#   control  : hoist off, fuse off   == the 14.0 ms canonical build
#   prohoist : hoist on,  fuse off    isolates the producer-row hoist alone
#   collfuse : hoist on,  fuse on     the ported fused collective (redline/ATOM)
#
# The lever under test is  collfuse - prohoist  (the LDS handoff that removes
# the HBM round-trip inside _rnlm8_pro_publish). prohoist - control tells us
# whether the hoist precondition is itself a win, a null, or a regression.
#
# Alternates arms in time (REPS), never blocked. Report the pair table.
# Re-run with REP_START=N to extend instead of overwrite.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

export SEQ='control= prohoist=MPK_QKV_PRO_HOIST=1 collfuse=MPK_QKV_PRO_HOIST=1,MPK_COLL_FUSE_NORM=1'
export REPS="${REPS:-1}"
export REP_START="${REP_START:-1}"
export RESULTS="${RESULTS:-/mnt/nvme1/glm5_collfuse_ab}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-4,5,6,7}"

ulimit -c 0
exec ./run_flag_pairs.sh
