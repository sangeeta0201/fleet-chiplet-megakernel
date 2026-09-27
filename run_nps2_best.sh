#!/bin/bash
# Reproduce the SPX/NPS2 decode number for gpt-oss-120b bs=1 (~1.207 ms/token on
# MI355X, thor-1, 2026-09-27; hash 3d54adb71e19): the lane recipe + the upstream
# chain + the embedding/LM-head tail flags, all from NPS2_RECIPE.env.
#
# Needs: SPX compute / NPS2 memory partition with the patched amdgpu
# (aid_local_* module params). Run inside the fleet container:
#   MODEL_PATH=/root/schowdha/models/gpt-oss-120b HIP_VISIBLE_DEVICES=6 ./run_nps2_best.sh
# From the host through the lane harness (what the measurements used):
#   LANE_HIP_DEV=6 TIMEOUT=600 LANE_ENVS=" " \
#     EXTRA_ENVS="$(grep -v '^#' NPS2_RECIPE.env | tr '\n' ' ')" \
#     bash ~/nps1/fleet_run_lane.sh port_up nps2_best
# Per-phase slots: add MPK_PHASE_SLOTS=1 MPK_PHASE_START_ITER=40 to the environment.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
export $(grep -v '^#' NPS2_RECIPE.env | tr '\n' ' ')
export MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-128}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-16}
exec bash demo/gpt_oss/run_1gpu.sh "$@"
