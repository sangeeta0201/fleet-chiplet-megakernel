#!/bin/bash
# THE BEST VALIDATED GLM-5 744B DECODE CONFIGURATION, bs=1.
#
#   NP=8 (all 8 devices) + in-graph MTP speculative decode
#   ==> 8.754 ms/token, measured n=3 on 2026-08-24.
#
# Both halves were already measured and gated; neither is a new lever. What
# was wrong was the configuration actually being run. Recent A/B work was
# scored against an NP=4 baseline of 11.418 ms/token on devices 4-7 -- a
# hardware-availability constraint that no longer holds. Devices 0-3 are idle.
#
#   11.418  NP=4, no MTP   <- what every recent lever was A/B'd against
#   10.459  NP=8, no MTP   (n=3: 10.649 / 10.368 / 10.360, all rc=0)
#    8.754  NP=8 + MTP     (n=3: 8.854 / 8.840 / 8.568)   -23.3%
#
# MTP measured here: 16.439 ms/iter, 1.86-1.89 tokens/iter, acceptance
# 0.857-0.887. The second row costs 5.98 ms/iter, which is CLOSED -- it splits
# 2.19 ms of 2-wide build geometry + 3.18 ms of real second row (a second
# top-8 expert set, whose weight slabs are not resident) + 0.494 ms of MTP
# chain. See memory glm-mtp-chain-is-cheap-the-row-is-expensive.
#
# CORRECTNESS. Do not quote these numbers without re-checking the text; see
# CLAUDE.md. Exact-token equality is an ILLEGAL gate on this model (bs=1 is not
# bit-reproducible and has two attractor continuations) -- use G1+G2+G3 via
# correctness_gate.py, or at minimum:
#   G1 cross-rank identity: exactly ONE distinct [SPEC] line across all 8 ranks
#   G2 coherence:           read the generated text
# Both passed on all three runs of the batch above.
#
#   ./run_best.sh [extra demo.py args]
#   ./run_best.sh --max-new-tokens 256
#
# To benchmark it n times instead:
#   MPK_SPEC_DECODE=1 ./bench_repeat.sh <tag> 3 -- --mtp 1 --max-num-batched-tokens 2
#
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

export MODEL_PATH="${MODEL_PATH:-/home/claudeuser/models/glm5-mxfp4}"

# MPK_SPEC_DECODE=1 is what makes prepare_next_batch dispatch the draft row;
# --mtp 1 alone only loads the draft layer for the torch reference path
# (demo.py:619 asserts the pair). --max-num-batched-tokens 2 is required
# (demo.py:634) because the draft row needs the second batch slot.
export MPK_SPEC_DECODE=1

# gpucore.<pid> dumps from a GPU fault are ~440 GB each; the disk is not big
# enough for one. (memory: rocgdb-attach-plus-kill-9-fills-the-disk)
ulimit -c 0

exec ./run_mp8_dp_ep_fused.sh --mtp 1 --max-num-batched-tokens 2 "$@"
