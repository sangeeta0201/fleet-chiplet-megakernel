#!/bin/bash
# PRICE THE MTP CHAIN WITH ONE ACTIVE ROW.
#
# probe_row_vs_harness_split.sh measured, one batch, same day:
#   bs1 (1-wide build, 1 row)           10.619
#   bs2 (2-wide build, 1 row)           12.754   +2.135  geometry, zero tokens
#   mtp (2-wide build, 2 rows + chain)  16.480   +3.726  row-2 work AND chain
# and proved the middle arm decodes ONE token per iteration, so the +3.726 is
# a SUM of two things nobody has separated. The arm that would have split it
# from the other side -- MPK_SPEC_DECODE=1 with no --mtp, the self-draft --
# wedges the megakernel 3/3 at launch.
#
# This is the same split from the near side. GLM_MTP_GRAPH_ONLY=1 wires the
# whole MTP chain into the task graph -- embedding gather, enorm/hnorm, both
# eh_proj GEMVs, the draft layer as its own replay run, the second
# lm_head+argmax -- while leaving MPK_SPEC_DECODE off, so prepare_next_batch
# dispatches ONE row (persistent_kernel.cuh:1155) and the draft token is
# thrown away. It is the configuration demo.py:542's assert exists to forbid,
# taken deliberately.
#
#   D - bs2  = the MTP chain, alone, at one active row
#   mtp - D  = the real second decode row
#
# bs2 is re-run in THIS batch as the anchor rather than compared to yesterday's
# number (glm5-baseline-is-11005-over-n5). Both arms are output-gated.
#
# Arm D commits one token per iteration by construction, so its wall is
# ms/iter == ms/token and its text must match a plain bs=1 control -- nothing
# in the chain feeds back into the committed stream. A G1/G2 failure here
# would mean the chain is corrupting the main path, which is worth knowing on
# its own.
#
# Rep 1 of each arm compiles; 2400 s, because 1200 s cut three build-reps off
# in the previous sweep.
#
# ============ RESULT, 2026-08-22.  THE SPLIT INVERTS 466b940 ===============
#
#   arm         rep walls (ms/iter)     n   mean     G1+G2
#   chainonly   13.299  HUNG  HUNG      1   13.299   PASS
#   bs2         12.833  12.776          2   12.805   PASS
#   mtp         16.510  16.450          2   16.480   PASS  (prior batch)
#   bs1         10.619                  3   10.619   PASS  (prior batch)
#
# The in-batch bs2 anchor reproduces the prior batch's 12.754 to within
# 0.051 ms -- one fifth of the 0.26 ms noise floor -- so the cross-batch mtp
# reference is licensed.  Full decomposition of MTP's 16.480:
#
#   10.619   bs=1 control
#   +2.186   2-wide BUILD GEOMETRY          zero extra tokens
#   +0.494   the ENTIRE MTP chain           embed gather, enorm/hnorm, both
#                                           eh_proj GEMVs, the draft layer as
#                                           its own replay run, 2nd lm_head
#   +3.181   the REAL second decode row
#   =16.480  (residual 0.000 by construction; the three terms are the arms)
#
# 466b940 CONCLUDED THE OPPOSITE.  It billed 3.721 ms to "the harness" and
# called it "1.7x a whole extra row."  The harness -- everything speculation
# adds to the graph -- is 0.494 ms, 13% of that.  The expensive half is the
# second decode row, at 3.181 ms: 30% of an entire bs=1 iteration for one
# extra token, on a machine that spends 50% of every layer spinning.
#
# WHAT THIS DOES TO THE BOARD.
#
# CAREFUL WITH THE GEOMETRY TERM -- it is NOT an MTP lever, and the first
# version of this block said it was.  The +2.186 was measured on a build whose
# second row is DEAD and clamped (router zero-fills it, the MoE tile decoder
# drops route_val==0, mla_decode returns before stamping LSE).  It is
# therefore byte movement sized by BATCH_SIZE regardless of liveness: 2 rows
# of EP fold payload, activation scratch, epilogue stores, 2x tile space
# walked.  In a build that under-fills the batch that is pure waste and could
# be sized on num_active_tokens instead.  MTP DOES NOT UNDER-FILL -- both rows
# are live -- so there is nothing there for it to shed.  "At zero geometry
# cost MTP would be 7.68 ms/token" is INVALID; do not quote it.
#
# The additive split 2.186 + 0.494 + 3.181 is exact by construction but the
# terms are not independently removable.  For MTP the meaningful grouping is:
#
#   0.494 ms  the MTP chain        -- overhead proper, and it is small
#   5.367 ms  a second live row    -- 2.186 to widen the build + 3.181 to
#                                     activate the row on top of that
#
# So a second token costs 5.367 ms of machine time against a 10.619 ms
# iteration, and MTP's 1.196x comes from acceptance 0.861 buying that back.
#
# WHERE THE 3.181 GOES, AND WHY ITEM 1 IS HARD.  The arithmetic is not the
# cost: an MFMA tile is M=16/32, so a second row rides free in the same
# instruction -- which is exactly why the row fold measured +0.348 us/layer
# (glm-batch-row-mfma-fold-is-a-negative).  What a second LIVE row really
# adds is a second top-8 expert set.  Under EP a rank owns ~1-2 activated
# experts at bs=1, and a second row with different routes can double that,
# so the MoE runs up to 2x the expert tiles.  The mtp-vs-bs1 region map is
# consistent: W2 +7.520 us/layer on a 8.117 base (+93%), W13 +3.550 on 12.095
# (+29%), and the MoE block is 48% of the captured delta.  That is genuinely
# new work, not overhead, and it is why "a machine that idle should swallow a
# second row" does not follow: the idle workers cannot help, because the
# second row needs weight slabs that are not resident.
#
# The "cut the harness" build implied by glm-mtp-harness-costs-3.7ms's
# sensitivity table is therefore DEAD: its whole budget is 0.494 ms, and the
# ideal-case row of that table (harness -> 0.186) is worth 0.31 ms, not 1.90.
# With the geometry term also off the table as an MTP lever, this probe closes
# out the overhead side of item 1 entirely: there is no 2 ms of MTP bookkeeping
# to delete.  What remains is the expert-set doubling, which is real work.
#
# CAVEAT, STATED PLAINLY: chainonly is n=1.  Reps 2 and 3 both wedged at
# `[HOST_DBG] launch_persistent_kernel ENTER` on the SAME binary that rep 1
# ran clean, GPUs pinned at 100%, and were killed rather than left to burn
# their 2400 s timeouts.  That is the known ~1-in-3 NP=8 hang streaking, not
# an arm defect -- unlike the self-draft arm, which went 0/3.  0.494 ms is
# 1.9x the noise floor, so the exact value needs the top-up reps.  The
# DIRECTION does not: for the chain to be the expensive half, chainonly would
# have to sit at 14.6+, which is 1.3 ms -- five noise floors -- above the
# observed sample.
# ===========================================================================
set -u
ulimit -c 0
cd /home/claudeuser/fleet-chiplet-megakernel/demo/glm5
D=/tmp/mtpchain
mkdir -p $D

export MODEL_PATH=/home/claudeuser/models/glm5-mxfp4
export MAX_SEQ_LENGTH=512
export MAX_NEW_TOKENS=256
export MAX_SAVE_TOKENS=264
export MPK_BAR_SKEW=0
export MPK_SUBPHASE_TIMING=0
PROMPT="Write a short paragraph explaining why the sky appears blue."

run_arm () {
  ARM=$1; NREP=$2
  echo "########## ARM $ARM ##########"
  rm -rf permanent_output_dir permanent_output_dir_rank*
  export KEEP_BUILD=0
  for REP in $(seq 1 $NREP); do
    export MASTER_PORT=$((39100 + (RANDOM % 300)))
    echo "===== $ARM REP $REP ====="
    timeout 2400 ./run_mp8_dp_ep_fused.sh \
        --prompt "$PROMPT" --save-tokens "$D/${ARM}_r${REP}.json" \
        > "$D/${ARM}_r${REP}.log" 2>&1
    echo "  rc=$?"
    echo -n "  -DMPK_SPEC_DECODE: "
    grep -ho "\-DMPK_SPEC_DECODE" "$D/${ARM}_r${REP}.log" | sort -u | tr '\n' ' '
    echo -n "| NBT: "
    grep -ho "\-DMPK_MAX_NUM_BATCHED_TOKENS=[0-9]*" "$D/${ARM}_r${REP}.log" \
        | sort -u | tr '\n' ' '; echo
    # "2 runs" proves the draft layer is really in the graph for arm D.
    grep -h "Multi-layer table" "$D/${ARM}_r${REP}.log" | head -1 | sed 's/^/  /'
    grep -h "\[SPEC\]" "$D/${ARM}_r${REP}.log" | tail -1 | sed 's/^/  /'
    grep -h "Decode:" "$D/${ARM}_r${REP}.log" | tail -1 | sed 's/^/  /'
    export KEEP_BUILD=1
  done
}

# ---- D: MTP chain in the graph, ONE active row ----
unset MPK_SPEC_DECODE
export GLM_MTP_GRAPH_ONLY=1
export MPK_EXTRA_ARGS="--mtp 1 --max-num-batched-tokens 2"
run_arm chainonly 3

# ---- anchor: plain 2-wide build, one active row, no chain ----
unset GLM_MTP_GRAPH_ONLY
export MPK_EXTRA_ARGS="--max-num-batched-tokens 2"
run_arm bs2 2

echo "############ G1+G2+G3 GATE ############"
for ARM in chainonly bs2; do
  echo "===== $ARM ====="
  python3 correctness_gate.py --ctl '/tmp/det_bs1/r*_rank*.json' \
                              --arm "$D/${ARM}_r*_rank*.json" 2>&1 | tail -14
done
echo MTPCHAIN_DONE
