#!/bin/bash
# MEASURE V -- the exact ceiling on a MoE row fold.  Go/no-go for item 1.
#
# A MoE tile is an (expert, row) pair, so two live decode rows always yield
# 2*TOPK = 16 live tiles regardless of routing overlap.  What overlap saves is
# DUPLICATE WEIGHT-SLAB FETCHES:
#
#     V = 2*TOPK - |topk(t) u topk(t+1)|      0 <= V <= TOPK = 8
#
# MPK_VPROBE=<stride> prints the ACTIVATED EXPERT LIST from rank 0, one
# worker, every <stride> routing epochs.  Measurement only -- it changes no
# tile's work, so this run's wall is NOT a datapoint and is not reported.
#
# WHY bs=1 AND NOT A LIVE 2-ROW BUILD.  Reading U straight off a 2-row build
# is the direct measurement, but on 2026-08-22 the 2-wide build faulted
# ("illegal memory access" at the first decode step) in 4 of 6 attempts while
# every bs=1 run was clean.  bs=1 gives the SAME quantity by pairing
# consecutive tokens offline, on a build that runs: MTP's second row is a
# draft of exactly the next token and is accepted 86% of the time.
#
# Priced ceiling this decides (demo/glm5/price_moe_row_fold.py):
#     saving = (V/8) * 0.808 ms   against a 5.458 ms second live row
#     V >= 4.95 clears the 0.5 ms build bar; V = 8 is the absolute maximum.
set -u
cd "$(dirname "$0")"
D=/tmp/vprobe
mkdir -p $D
ulimit -c 0

# MODEL_PATH MUST be pinned: env_common.sh defaults to GLM-4.7-Flash, and an
# unpinned run dies with "absorbed o_proj reduces over num_q_heads *
# kv_lora_rank = 16384, got 10240" (or benchmarks the wrong model).
export MODEL_PATH=/home/claudeuser/models/glm5-mxfp4
export MAX_SEQ_LENGTH=512

STRIDE=1                 # contiguous epochs; the lag scan below needs them
export MPK_VPROBE=$STRIDE
export MPK_SUBPHASE_TIMING=0
export MPK_BAR_SKEW=0
export MAX_NEW_TOKENS=128
export MAX_SAVE_TOKENS=136
export KEEP_BUILD=${KEEP_BUILD:-0}
PROMPT="Write a short paragraph explaining why the sky appears blue."

ARM=bs1
echo "===== $ARM ====="
[ "$KEEP_BUILD" = 1 ] || rm -rf permanent_output_dir permanent_output_dir_rank* 2>/dev/null
unset GLM_MTP_GRAPH_ONLY MPK_EXTRA_ARGS MPK_SPEC_DECODE
export MASTER_PORT=$((42100 + (RANDOM % 300)))
timeout 1500 ./run_mp8_dp_ep_fused.sh \
    --prompt "$PROMPT" --save-tokens "$D/${ARM}.json" \
    > "$D/${ARM}.log" 2>&1
echo "  rc=$?"
grep -c "DMPK_VPROBE=$STRIDE" "$D/${ARM}.log" | sed 's/^/  hipcc -DMPK_VPROBE lines: /'
grep -c "\[VPROBE\]" "$D/${ARM}.log" | sed 's/^/  VPROBE samples: /'

echo "############ V ############"
python3 vprobe_analyze.py "$D/${ARM}.log"
echo VPROBE_DONE

# ===========================================================================
# RESULT  2026-08-22   (run 14:36, /tmp/vprobe/bs1.log, 38325 rank-0 samples,
#                       128 tokens x 4 prompts, ~504 decode iterations)
# ===========================================================================
#   routed top-k size = 8 in all 38325 samples; expert id 256 is the shared
#   expert and is filtered out.  Epochs 1..38835 with 510 single-epoch holes,
#   and THE HOLES ARE PERIOD-76 -- one silent epoch per decode iteration,
#   which independently confirms 75 printing MoE layers/iteration (78 hidden
#   - 3 dense) without trusting the config.
#
#   lag scan of mean |topk(t) n topk(t+lag)|:
#     lag  76   V = 2.801    <-- same layer, CONSECUTIVE tokens
#     lag 152   V = 2.287    <-- same layer, two tokens apart (decays: real)
#     all 197 other lags     0.234 .. 0.277, mean 0.275
#     independence line     8*8/256 = 0.250
#
#   The scan validates itself three ways: only the two same-layer lags rise
#   above the floor, the floor sits on the independence line, and the signal
#   decays with token distance.  No tuning, no assumed layer count.
#
#   V = 2.801 of 8  ->  35.0% expert reuse between consecutive tokens.
#     priced fold saving = (2.801/8) * 0.830 = 0.291 ms
#                        = 5.3% of the 5.458 ms second live row
#                        = 2.7% of the 10.619 ms bs=1 wall
#     bar = 0.5 ms (needs V >= 4.82)   ->   NO-GO, by 1.7x
#
#   And 0.291 ms is a BYTES CEILING, not a forecast: EP sharding, in-phase
#   absorption and MoE round quantization each discount it and none inflate
#   it (see price_moe_row_fold.py).  The realized number is below the 0.26 ms
#   wall noise floor, so it could not even be measured at n=3.
#
#   VERDICT: the MoE row fold is NOT WORTH BUILDING.  The second decode row's
#   5.458 ms is real per-row expert-weight traffic; the two rows genuinely
#   route to different experts, and the 35% they share is worth a third of
#   the build bar.
# ===========================================================================

# ===========================================================================
# CORRECTION 2026-08-22, same day, same V.  The DENOMINATOR above was wrong.
# ===========================================================================
#   The region map (glm-second-decode-row-region-map) labels S5->S6 "routing"
#   and S6->S7 "W13".  Both labels are wrong.  Every mpk_stage_stamp id has
#   exactly one call site, so the mapping is not ambiguous:
#
#     stamp 5 @gang_oproj_router_fused_mi300.cuh:1220  routing wait done
#     stamp 6 @1521  W13 tiles done      => S5->S6 = Phase 5, W13 TILES  +9.400
#     stamp 7 @1614  W13->W2 release     => S6->S7 = Phase 6, BARRIER    +3.550
#     stamp 8 @1669  W2 tiles done       => S7->S8 = Phase 7, W2 TILES   +7.520
#
#   I priced the fold against the +3.550 BARRIER line as if it were W13.  The
#   foldable MoE TILE delta is 9.400 + 7.520 = 16.920 us/layer, not 11.070.
#   The MoE group total is unchanged (20.470); only the split inside it moves.
#
#     MoE tile share of the second row  1.269 ms  (23.6%, was 0.830 / 15.2%)
#     saving at the measured V = 2.801  0.444 ms  (was 0.291)
#     bar 0.5 ms now needs V >= 3.15    (was 4.82)
#
#   VERDICT UNCHANGED BUT NO LONGER COMFORTABLE: still NO-GO, by 1.13x rather
#   than 1.7x, and 0.444 ms is now ABOVE the 0.26 ms wall noise floor instead
#   of below it.  What still carries it is that 0.444 is a BYTES ceiling and
#   the three discounts are not small: at EP=8 a rank owns ~1-2 activated
#   experts so a duplicate pair must land on the STRAGGLER rank to pay;
#   MoE rounds are ceil(tiles/29) so removing tiles that do not cross a round
#   boundary buys exactly zero; and cutting work inside a phase is absorbed
#   3 times out of 3.  The honest statement is "priced at 0.444 ms ceiling,
#   realized well under, against a 0.5 ms bar" -- not "1.7x short".
# ===========================================================================
