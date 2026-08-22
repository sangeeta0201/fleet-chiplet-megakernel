#!/bin/bash
# DOES RANK 0's SHARED EXPERT COST MAKESPAN?  A correct-output, additive probe.
#
# WHY THIS RUN EXISTS.  Two entries in the ledger disagree, and the
# disagreement decides whether a shared-expert K-shard is worth writing.
#
#   glm-shared-expert-hoist-is-zero-makespan (2026-08-21 11:44)
#       "rank 0's W13 makespan is the SHORTEST of the eight (8.55 vs ~12.5 us)
#        despite the extra expert -- its tiles occupy workers that would
#        otherwise be idle.  The imbalance costs ZERO makespan.  Do not build
#        a hoist, a shard, or a reschedule."
#
#   f31bc61 (2026-08-22), re-running THAT SAME max-max statistic on logs taken
#   with MPK_BAR_SKEW_DROP_NS=1000000 set:
#       statistic            r0 W13   peers   r0-peers   r0 rank
#       median                11.82    5.80     +6.02    #8 of 8
#       makespan (max-max)    18.11   12.27     +5.84    #8 of 8
#   -- rank 0 is the LONGEST of eight, not the shortest.  The peers barely
#   moved (12.56 -> 12.27); rank 0 went 8.55 -> 18.11.  Likely the instrument:
#   glm-stage-stamp-drop-guard-inflated-the-boundary was found at 12:00 on
#   08-21, SIXTEEN MINUTES after the closure was recorded, and the closure was
#   never re-derived.  The MoE-dispatch clamp c8ef8d5 landed at 03:59, before
#   both, so no code change explains it.
#
# Both readings come from the same instrument family.  This probe settles it
# at the WALL instead, where neither drop guard nor order statistic applies.
#
# ONE VARIABLE: MPK_SHARED_DUP (0 vs 1).  At 1, rank 0's shared expert gets two
# consecutive slots in its owned W13 tile subsequence, so every shared-expert
# W13 tile runs TWICE.  Nothing else changes, on any rank.
#
# THIS IS CORRECT OUTPUT, which is the point -- it is ADDITIVE, not an
# ablation, so no wrong-output caveat applies and the wall number is quotable
# (after the gate below).  W13's epilogue is a plain (or write-through) store
# of a deterministic value, so the duplicate tile rewrites the same bits.
# W13 ONLY: W2's epilogue is an f32 atomicAdd and a duplicated W2 tile would
# double-count the shared expert.  DUP_SHARED is not passed at the W2 call
# site and must never be.
#
# WHAT EACH MODEL PREDICTS
#   "absorbed"    the extra expert lands on workers that were idle anyway and
#                 every rank runs the same number of grid-stride rounds
#                 -> ~0 ms
#   "on the path" rank 0 is the MoE straggler and its peers idle at the next
#                 cross-rank rendezvous for exactly its overrun; W13 carries
#                 +6.02 us/layer of that
#                 -> +6.0 us/layer x 76 = +0.46 ms/token
#
# 0.46 ms is 1.8x the 0.26 ms wall noise floor, so n=3 resolves it.
#
# HOW TO READ A POSITIVE RESULT.  Per glm-additive-probes-overprice-deletions
# (four pairs), adding work costs ~1:1 while removing the same work saves ~0.
# So a positive delta here is an UPPER BOUND on what removing rank 0's W13
# excess could buy -- necessary evidence for a K-shard, not sufficient.
# A NULL result is the decisive direction: if rank 0 absorbs a whole extra
# expert of W13 for free, there is no makespan to harvest and the whole
# hoist/shard/reschedule family closes for good.
#
# THREE PHASES
#   A  wall, instruments OFF, n=3 per arm             -> the deciding number
#   B  G1+G2+G3 correctness gate on the dup arm       -> licenses quoting A
#      (reuses the 4-run /tmp/det_bs1 control set, same prompt, as
#       probe_mtp_regate.sh does -- both attractor clusters are in it)
#   C  region counters, MPK_BAR_SKEW=3 on BOTH arms   -> LIVENESS
#
# PHASE C IS NOT OPTIONAL.  A null wall with the flag in the build is
# ambiguous between "absorbed" and "the probe did nothing"
# (glm-moe-kloop-batching-is-neutral: a flag in the log does NOT prove a
# kernel runs).  C reads rank 0's S5->S6 directly: if the duplicate tiles
# execute, it must grow by roughly the shared expert's own W13 time and the
# seven peers must not move.  If S5->S6 does not move, the probe is dead and
# the wall says nothing.
#
# ===================== RESULT: (pending -- filled in on completion) =========
set -u
ulimit -c 0
cd /home/claudeuser/fleet-chiplet-megakernel/demo/glm5

export MODEL_PATH=/home/claudeuser/models/glm5-mxfp4
export MPK_PRINT_ALL_RANKS=0
N="${N:-3}"
D=/tmp/shdup
mkdir -p $D

echo "########################## PHASE A: WALL ##########################"
export MPK_SUBPHASE_TIMING=0
export MPK_BAR_SKEW=0
for ARM in base dup; do
  if [ "$ARM" = base ]; then export MPK_SHARED_DUP=0
  else export MPK_SHARED_DUP=1; fi
  unset KEEP_BUILD
  rm -rf permanent_output_dir permanent_output_dir_rank*
  echo "########## ARM $ARM  MPK_SHARED_DUP=$MPK_SHARED_DUP ##########"
  ./bench_repeat.sh "shdup_${ARM}" "$N"
  # The -D must be in the build on ALL EIGHT ranks.  A flag that never reached
  # the compile is how a null result gets manufactured
  # (mirage-cpp-edits-need-a-two-step-rebuild).
  echo -n "  DMPK_SHARED_DUP in build, rank count: "
  grep -ho "DMPK_SHARED_DUP" /tmp/glm5_shdup_${ARM}_r1.log | wc -l
  # Eyeball the text now, before any number is believed (CLAUDE.md).
  echo "  --- generated text, rep 1 ---"
  grep -hE "^\[1,0\].*(Generated|Output|assistant)" \
      /tmp/glm5_shdup_${ARM}_r1.log | head -3
done

echo "########################## PHASE B: CORRECTNESS ##########################"
# Reuses the dup build left warm by phase A: MAX_NEW_TOKENS / MAX_SAVE_TOKENS
# are runtime, not compile-time, so the megakernel under test is byte-identical
# to the one phase A just timed.  That is deliberate -- gating a DIFFERENT
# build than the one that produced the wall number proves nothing.
export MPK_SHARED_DUP=1
export MAX_SEQ_LENGTH=512
export MAX_NEW_TOKENS=256
export MAX_SAVE_TOKENS=264
export KEEP_BUILD=1
PROMPT="Write a short paragraph explaining why the sky appears blue."
for REP in 1 2 3; do
  export MASTER_PORT=$((37100 + (RANDOM % 300)))
  echo "===== DUP CORRECTNESS REP $REP ====="
  timeout 900 ./run_mp8_dp_ep_fused.sh \
      --prompt "$PROMPT" --save-tokens "$D/dup_r${REP}.json" \
      > "$D/dup_r${REP}.log" 2>&1
  echo "  rc=$?"
  echo -n "  DMPK_SHARED_DUP in build: "
  grep -ho "\-DMPK_SHARED_DUP" "$D/dup_r${REP}.log" | wc -l
done
echo "############ G1+G2+G3 GATE ############"
python3 correctness_gate.py --ctl '/tmp/det_bs1/r*_rank*.json' \
                            --arm "$D/dup_r*_rank*.json"

echo "########################## PHASE C: LIVENESS ##########################"
# MPK_BAR_SKEW=3 on BOTH arms so the instrument's cost is common-mode.
# DROP_NS is MANDATORY -- the default 10 ms guard keeps one ml=0 sample per
# iteration at ~10000x weight and is exactly the bug that produced the closure
# this probe is re-testing.
unset MAX_SAVE_TOKENS
export MPK_BAR_SKEW=3
export MPK_BAR_SKEW_DROP_NS=1000000
export MPK_SUBPHASE_TIMING=0
export MAX_NEW_TOKENS=96
for ARM in base dup; do
  if [ "$ARM" = base ]; then export MPK_SHARED_DUP=0
  else export MPK_SHARED_DUP=1; fi
  unset KEEP_BUILD
  rm -rf permanent_output_dir permanent_output_dir_rank*
  export MASTER_PORT=$((37600 + (RANDOM % 300)))
  echo "===== COUNTER ARM $ARM ====="
  timeout 1800 ./run_mp8_dp_ep_fused.sh \
      --prompt "$PROMPT" > "$D/ctr_${ARM}.log" 2>&1
  echo "  rc=$?  BARSTAGEWS lines: $(grep -c BARSTAGEWS $D/ctr_${ARM}.log)"
  echo -n "  DMPK_SHARED_DUP in build, rank count: "
  grep -ho "DMPK_SHARED_DUP" "$D/ctr_${ARM}.log" | wc -l
done
python3 compare_shared_dup_counters.py

echo "########################## WALL SUMMARY ##########################"
for ARM in base dup; do
  echo -n "$ARM  "
  grep -hE '^\[1,0\].*Decode:' /tmp/glm5_shdup_${ARM}_r*.log \
    | grep -oE 'avg [0-9.]+ms/iter' | grep -oE '[0-9.]+' \
    | awk '{v[n++]=$1; s+=$1; if(n==1||$1<mn)mn=$1; if($1>mx)mx=$1}
           END{if(n)printf "n=%d mean=%.3f min=%.3f max=%.3f spread=%.3f\n",
                             n,s/n,mn,mx,mx-mn; else print "NO RESULTS"}'
done
echo SHDUP_DONE
