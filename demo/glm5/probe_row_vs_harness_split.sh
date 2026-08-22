#!/bin/bash
# SPLIT THE 3.721 ms "MTP HARNESS" FROM THE REAL SECOND DECODE ROW.
#
# THE CONFOUNDER.  466b940 priced a decode row as
#     bs2 plain (--max-num-batched-tokens 2)  12.809
#   - bs1 control                             10.619
#   = +2.190 ms, C_row = 1.206
# and everything above that as "harness".  But prepare_next_batch's decode
# branch is
#     num_new_tokens = min(1, MPK_MAX_NUM_BATCHED_TOKENS - num_tokens);
#     #ifdef MPK_SPEC_DECODE  ... num_new_tokens = MPK_SPEC_WIDTH;  #endif
# (persistent_kernel.cuh:1155/1197) and --max-num-batched-requests defaults to
# 1 (demo.py:450, and mla_decode asserts it == 1).  So WITHOUT MPK_SPEC_DECODE
# a decode iteration puts exactly ONE token in the batch:
#     num_active_tokens = qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS] = 1
# and every kernel clamps to it (gang_linear_mxfp8_mi300.cuh:205,
# gang_gemv_mi300.cuh:245, gang_oproj_router_fused_mi300.cuh:794,
# argmax_mi300.cuh:97).
#
# So "bs2 plain" is a BATCH_SIZE=2 BUILD DECODING ONE ROW.  Its +2.190 ms is
# build-shape overhead for zero extra tokens.  The real second row's work is
# inside the 3.721 ms that was labelled harness.
#
# THE INSTRUMENT.  MPK_SPEC_DECODE=1 with NO --mtp is legal (demo.py:540 only
# asserts the converse) and gives the self-draft: spec_draft_tokens is nullptr,
# so prepare_next_batch stages tokens[step] as the draft
# (persistent_kernel.cuh:1191).  That arm dispatches TWO real rows -- two
# attention rows, two RMSNorm+quant prologues, two rows of every GEMM epilogue,
# the accept scan, the step advance, the page growth -- and has NO draft layer,
# NO eh_proj, NO second lm_head+argmax.  It is exactly the missing middle term.
#
#   A  bs2 plain      2-wide build, 1 active row
#   B  self-draft     2-wide build, 2 active rows, no MTP layer
#   C  MTP width-1    2-wide build, 2 active rows + draft layer + 2nd lm_head
#
#   B - A  = the real second decode row
#   C - B  = the MTP draft-layer harness proper
#
# One variable per arm, all three in one batch (never against yesterday's
# numbers -- glm5-baseline-is-11005-over-n5).  n=3 each, noise floor 0.26 ms.
#
# Arm B is also the null-draft correctness gate the harness was built for: it
# proposes the last committed token, so it accepts occasionally and must still
# emit a legal continuation.  All three arms are gated with correctness_gate.py
# against the four bs=1 controls in /tmp/det_bs1.
#
# ================== RESULT, 2026-08-22.  CONFOUNDER CONFIRMED ==============
#
#   arm   rep walls (ms/iter)      n   mean     G1+G2
#   bs2   12.822  12.686           2   12.754   PASS   (r1 illegal address)
#   self  HUNG    HUNG    HUNG     0   --       --     3/3 wedged at launch
#   mtp   16.510  16.450           2   16.480   PASS   (r2 timed out)
#   bs1   10.619 (n=3, prior batch, unchanged build shape)
#
# bs2 reproduces the prior batch's 12.809 and mtp its 16.530, so the two
# published numbers stand.  What does NOT stand is their interpretation:
#
#   +2.135 ms   2-wide build decoding ONE row   <- ZERO extra tokens
#   +3.726 ms   the real second row's work AND the MTP chain, together
#
# The bs2 arm's own log settles it without any instrument:
#     "Decode:  496 tokens in 493/496 iters"
# one token per iteration.  So "C_row = 1.206" was never the price of a second
# decode row -- it is what BATCH_SIZE=2 costs a build that still decodes one.
# Every dead-row path is correctly clamped (the router zero-fills inactive
# rows, moe_topk_sigmoid_bias_mi300.cuh:130-176; the MoE tile decoder drops
# route_val==0, gang_moe_linear_mxfp8_mi300.cuh:400; mla_decode returns before
# stamping LSE, gang_mla_decode_mi300.cuh:256), so the 2.135 ms is geometry --
# 2x tile dispatch walked and skipped, 2-row MFMA shapes, 2x workspace -- not
# work.  MTP pays it too.
#
# ARM B HANGS, 3/3, DETERMINISTICALLY.  Not the ~1-in-3 NP=8 flake: all three
# stop at `[HOST_DBG] launch_persistent_kernel ENTER` with no illegal address
# and no output, and rep 2 and 3 reused rep 1's build.  So MPK_SPEC_DECODE=1
# WITHOUT --mtp -- the null self-draft that persistent_kernel.cuh:1169
# advertises as the harness's own correctness gate -- wedges the megakernel.
# The graph is byte-identical to the bs2 arm ("Multi-layer table: 76 fused
# layers" in both); the only difference is prepare_next_batch dispatching two
# rows.  Do not reach for this arm again without fixing it first.
#
# ITEM 2 IS STILL NO-GO, and the confounder does not rescue it.  Re-pricing
# width-2 as geometry(3-wide) + 2x(row+chain) instead of 2x row + 2x harness
# gives 22.341 ms/iter against the old model's 22.441 -- break-even a2 moves
# 0.782 -> 0.769, both far above a1 = 0.861's plausible second-token rate.
#
# WHAT IS NEW: a BATCH_SIZE=2 build costs +2.135 ms (+20%) to produce exactly
# the same one token as a BATCH_SIZE=1 build, and MTP carries that charge on
# every iteration.  At zero geometry cost MTP would be 14.35 ms/iter =
# 7.71 ms/token (1.38x) instead of 8.884.  That is larger than anything else
# outstanding.  Splitting the +3.726 needs an arm that runs the MTP chain with
# one active row -- see GLM_MTP_GRAPH_ONLY in demo.py, which is exactly the
# configuration the assert at demo.py:542 exists to forbid in production.
# ===========================================================================
set -u
ulimit -c 0
cd /home/claudeuser/fleet-chiplet-megakernel/demo/glm5
D=/tmp/rowsplit
mkdir -p $D

export MODEL_PATH=/home/claudeuser/models/glm5-mxfp4
export MAX_SEQ_LENGTH=512
export MAX_NEW_TOKENS=256
export MAX_SAVE_TOKENS=264
export MPK_BAR_SKEW=0
export MPK_SUBPHASE_TIMING=0
PROMPT="Write a short paragraph explaining why the sky appears blue."

run_arm () {
  ARM=$1
  echo "########## ARM $ARM ##########"
  rm -rf permanent_output_dir permanent_output_dir_rank*
  export KEEP_BUILD=0
  for REP in 1 2 3; do
    export MASTER_PORT=$((37100 + (RANDOM % 300)))
    echo "===== $ARM REP $REP ====="
    timeout 1200 ./run_mp8_dp_ep_fused.sh \
        --prompt "$PROMPT" --save-tokens "$D/${ARM}_r${REP}.json" \
        > "$D/${ARM}_r${REP}.log" 2>&1
    echo "  rc=$?"
    # -DMPK_SPEC_DECODE is the ONLY place the flag appears; `MPK_SPEC_DECODE=1`
    # never shows up in the log even when speculation is fully on.
    echo -n "  -DMPK_SPEC_DECODE in build: "
    grep -ho "\-DMPK_SPEC_DECODE" "$D/${ARM}_r${REP}.log" | sort -u | tr '\n' ' '
    echo -n " | MAX_NUM_BATCHED_TOKENS: "
    grep -ho "\-DMPK_MAX_NUM_BATCHED_TOKENS=[0-9]*" "$D/${ARM}_r${REP}.log" \
        | sort -u | tr '\n' ' '; echo
    grep -h "Multi-layer table" "$D/${ARM}_r${REP}.log" | head -1 | sed 's/^/  /'
    # proposed>0 proves the 2-row dispatch actually happened this run.
    grep -h "\[SPEC\]" "$D/${ARM}_r${REP}.log" | tail -1 | sed 's/^/  /'
    grep -h "FWD_PASS_TOTAL" "$D/${ARM}_r${REP}.log" | tail -1 | sed 's/^/  /'
    grep -h "Decode:" "$D/${ARM}_r${REP}.log" | tail -1 | sed 's/^/  /'
    export KEEP_BUILD=1
  done
}

# ---- A: plain 2-wide build, ONE active row ----
unset MPK_SPEC_DECODE
export MPK_EXTRA_ARGS="--max-num-batched-tokens 2"
run_arm bs2

# ---- B: self-draft, TWO active rows, no MTP layer ----
export MPK_SPEC_DECODE=1
export MPK_EXTRA_ARGS="--max-num-batched-tokens 2"
run_arm self

# ---- C: MTP width-1 ----
export MPK_SPEC_DECODE=1
export MPK_EXTRA_ARGS="--mtp 1 --max-num-batched-tokens 2"
run_arm mtp

echo "############ G1+G2+G3 GATE ############"
for ARM in bs2 self mtp; do
  echo "===== $ARM ====="
  python3 correctness_gate.py --ctl '/tmp/det_bs1/r*_rank*.json' \
                              --arm "$D/${ARM}_r*_rank*.json" 2>&1 | tail -25
done
echo ROWSPLIT_DONE
