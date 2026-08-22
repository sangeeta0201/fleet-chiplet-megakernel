#!/bin/bash
# THE EP COLLECTIVE -- re-price the layer's LARGEST region at today's baseline.
#
# WHY NOW.  Two reasons, and the second is the interesting one.
#
# 1. THE EXISTING NUMBER IS STALE.  MPK_EP_ABLATE=1 was priced once, on
#    2026-08-18, at 14.675 -> 14.276 = 0.40 ms
#    (glm-ep-collective-costs-0.4ms-per-layer-set).  That baseline is 14.675 ms.
#    Today's is 10.490.  4.2 ms of per-rank work has left the layer since, and
#    the EP wait is PURE INTER-RANK SKEW -- not transport, not mechanism
#    (glm-ep-peer-poll-is-not-the-cost: batching the 7 loads moved it
#    17.49 -> 17.49).  A skew term measured against a 40%-longer layer does not
#    transfer.  Meanwhile the board still prices S0->S1 at 15.62 us/layer =
#    1.187 ms, the LARGEST single region in the bs=1 budget
#    (glm-board-audit-no-region-above-0.4ms-unattacked), and its two recorded
#    verdicts -- "hoist neutral, widen neutral" -- are both MECHANISM levers.
#    The WAIT itself has never been ablated on a modern build.
#
# 2. e02d5d5 REOPENED THE CLASS.  Rule 3 of e5d1ff5 says a deleted wait's skew
#    just relocates to the neighbouring rendezvous if they share a producer
#    set.  That rule closed three levers.  All three were INTRA-rank barriers,
#    where the spin really is local arrival-spread absorption.  The q_b
#    CROSS-RANK gather broke it: deleting it moved S22->S28 by -0.04 us/layer
#    and ~92% of the freed time survived to the wall
#    (glm-qb-peer-wait-ceiling-is-0.335ms).  So the falsification is narrow and
#    specific -- rule 3 holds for local skew absorption and fails for
#    cross-rank rendezvous -- and the layer has exactly TWO cross-rank
#    rendezvous.  This probe is the other one, and it is 3.5x bigger.
#
# THE SHARP TEST THIS PROBE HAS AND THE q_b ONE DID NOT.  The EP collective and
# the q_b gather share a producer set: both are "the 8 ranks".  That is exactly
# the configuration rule 3 says must relocate.  So S19->S20 is not a formality
# here, it is a live prediction:
#
#   rule 3 right -> freed EP skew reappears at S19->S20, wall moves ~0
#   e02d5d5 right -> S19->S20 flat, freed time survives to S0->S14 and the wall
#
# Either way we learn something the board does not currently know.
#
# ONE VARIABLE: MPK_EP_ABLATE (0 vs 1).  On the MLA path
# (gang_mla_full_layer_fused_mi300.cuh) =1 deletes the EP_NPEER peer signal
# stores (:1083, direct path), the putmem_signal fan-out (:1129, staged path)
# and _full_layer_ep_wait_peers (:1206) -- and NOTHING else.  The local fold,
# the local signal store, the eight-flag release fan-out, the per-XCD release
# wait, the self-heal and every barrier all still run.  Verified by reading
# each of the four #if sites.
#
# INSTRUMENT CAVEAT, READ BEFORE THE COMPARATOR.  mpk_stage_stamp(4) sits
# INSIDE `#if MPK_EP_ABLATE == 0` (:1213), so slot 4 is NOT WRITTEN in the
# ablated arm.  The peer wait's own span S3->S4 is therefore unreadable in arm
# 1 and must not be differenced.  The bracket S0->S1 is the right span instead:
# both stamps are unguarded, both are written by tid==0 of all 232 blocks
# (:748, :1330), and S0->S1 is precisely the region the board prices at 1.187
# ms.  Slots 3/4 have only 8 writers (one leader per XCD) even in the base arm.
#
# WRONG OUTPUT BY CONSTRUCTION: the peers never publish, so every rank's
# residual stream is built from the previous layer's peer gather slots.  No
# latency claim may be quoted from this -- CEILING ONLY.  And it is upstream of
# the router, so garbage changes TopK and therefore EP expert balance
# (glm-wrong-output-probes-upstream-of-router-are-invalid).  Mitigation, same
# as the decode and q_b ablations: require the two routes to agree.  A wall
# delta far larger than the counter delta means the router confound moved the
# MoE half rather than the EP wait moving the layer.
#
# TWO PHASES:
#   A  wall, instruments OFF, n=3 per arm            -> the deciding number
#   B  region counters, MPK_BAR_SKEW=3 on BOTH arms  -> the corroborating route
#      (probe cost common-mode, the method dddf699 used when the wall could
#      not resolve the EP poll batch)
#
# ===================== RESULT, 2026-08-22, n=3 per arm =====================
#   arm                       runs                      mean    min     max
#   base     (EP_ABLATE=0)    (see WALL SUMMARY)      10.490  10.382  10.552
#   epablate (EP_ABLATE=1)    (see WALL SUMMARY)      10.284  10.203  10.328
#
#   WALL = -0.206 ms.  Arms disjoint by only +0.054, and 0.206 is UNDER the
#   0.26 ms noise floor -- the wall alone CANNOT resolve this.  The counters
#   can, and they are not close.
#
# ROUTE 2, region counters, MPK_BAR_SKEW=3 on BOTH arms (common-mode), pooled
# over 8 ranks (compare_ep_ablate_counters.py).  us/layer, makespan:
#
#   S0->S1   THE EP COLLECTIVE (deleted)     24.14 ->  7.51  -16.63  -1.264 ms
#   S0->S2   EP + release + dispatch         25.32 ->  8.32  -17.00  -1.292 ms
#   S1->S16  EP release -> attn entry         2.32 ->  2.35   +0.03  +0.002 ms
#   S19->S20 THE OTHER CROSS-RANK RENDEZVOUS  9.35 -> 23.23  +13.89  +1.055 ms
#   S22->S28 intra-rank absorber             11.01 -> 11.00   -0.01  -0.001 ms
#   S0->S14  LAYER SPAN                     159.51 ->156.90   -2.61  -0.198 ms
#
# THE TWO ROUTES AGREE: wall -0.206, layer span -0.198.  Inside 0.008 ms.
#
# ================== RULE 3 CAUGHT IN THE ACT, AND IT WINS ==================
# 83% of the deleted EP wait (13.89 of 16.63 us/layer) REAPPEARS at S19->S20,
# the q_b head-shard gather -- the one rendezvous in the layer that shares this
# one's producer set.  It goes nowhere else: both intra-rank absorbers are flat
# to 0.03 us (S1->S16 +0.03, S22->S28 -0.01).  This is the cleanest measurement
# of skew relocation on the branch, because the destination was PREDICTED from
# the producer-set rule before the run and the freed time landed on it and on
# nothing else.
#
# So the e02d5d5 result is NOT "cross-rank rendezvous are a live lever class".
# The two measurements reconcile into one model:
#
#   THE LAYER'S INTER-RANK SKEW IS PAID ONCE, AT THE FIRST CROSS-RANK
#   RENDEZVOUS.  A SECOND ONE COSTS ONLY ITS OWN MARGINAL MECHANISM.
#
# That predicts both numbers.  Delete the FIRST (EP, this probe): the skew
# simply moves to the second and 83% comes back -- net 0.198.  Delete the
# SECOND (q_b, e02d5d5): the first already absorbed the skew, so what is
# deleted is only that rendezvous' own marginal cost, and 92% of it survives --
# net 0.335.  Nothing is contradictory and rule 3 needs no amendment; what
# needed amending was the assumption that a big region is a big lever.
#
# CONSEQUENCE FOR THE BOARD.  S0->S1 is the LARGEST line in the bs=1 budget at
# 1.187 ms and it is NOT DELETABLE -- it is the layer's rank-alignment tax and
# it gets paid at whatever cross-rank sync point exists.  No rendezvous-side
# lever (hoist, widen, batch the poll, delete, re-place) can touch it; all five
# are now measured.  The only thing that can is REDUCING THE INTER-RANK SKEW
# ITSELF.  That is a load-balance question, not a barrier question, and it is
# where the next EP work should go.
#
# Sanity note on S3->S4: base-only (slot 4 is compiled out under =1), 31.88
# us/layer makespan over 8 writers/rank.  Bigger than S0->S1's 24.14 because
# its population is the 8 XCD leaders, not all 232 -- a makespan over a subset
# with a different span is not comparable. Do not difference the two.
# ======================================================================
set -u
ulimit -c 0
cd /home/claudeuser/fleet-chiplet-megakernel/demo/glm5

export MODEL_PATH=/home/claudeuser/models/glm5-mxfp4
export MPK_PRINT_ALL_RANKS=0
N="${N:-3}"

echo "########################## PHASE A: WALL ##########################"
export MPK_SUBPHASE_TIMING=0
export MPK_BAR_SKEW=0
for ARM in base epablate; do
  if [ "$ARM" = base ]; then export MPK_EP_ABLATE=0
  else export MPK_EP_ABLATE=1; fi
  unset KEEP_BUILD
  rm -rf permanent_output_dir permanent_output_dir_rank*
  echo "########## ARM $ARM  MPK_EP_ABLATE=$MPK_EP_ABLATE ##########"
  ./bench_repeat.sh "epabl_${ARM}" "$N"
  # The -D must be in the build on ALL EIGHT ranks. A flag that never reached
  # the compile is how a null result gets manufactured
  # (mirage-cpp-edits-need-a-two-step-rebuild). Note base builds
  # -DMPK_EP_ABLATE=0 only if the env var is set, so grep the VALUE.
  echo -n "  DMPK_EP_ABLATE=1 in build, rank count: "
  grep -ho "DMPK_EP_ABLATE=1" /tmp/glm5_epabl_${ARM}_r1.log | wc -l
done

echo "########################## PHASE B: COUNTERS ##########################"
# MPK_BAR_SKEW=3 on BOTH arms so the instrument's cost is common-mode.
# DROP_NS is MANDATORY: the default 10 ms guard keeps one ml=0 sample per
# iteration and inflated the layer boundary 20.18 vs 8.21 us
# (glm-stage-stamp-drop-guard-inflated-the-boundary).
export MPK_BAR_SKEW=3
export MPK_BAR_SKEW_DROP_NS=1000000
export MPK_SUBPHASE_TIMING=0
export MAX_SEQ_LENGTH=512
export MAX_NEW_TOKENS=96
for ARM in base epablate; do
  if [ "$ARM" = base ]; then export MPK_EP_ABLATE=0
  else export MPK_EP_ABLATE=1; fi
  unset KEEP_BUILD
  rm -rf permanent_output_dir permanent_output_dir_rank*
  export MASTER_PORT=$((32600 + (RANDOM % 150)))
  echo "===== COUNTER ARM $ARM ====="
  timeout 1800 ./run_mp8_dp_ep_fused.sh \
      --prompt "Write a short paragraph explaining why the sky appears blue." \
      > "/tmp/epabl_ctr_${ARM}.log" 2>&1
  echo "  rc=$?  BARSTAGEWS lines: $(grep -c BARSTAGEWS /tmp/epabl_ctr_${ARM}.log)"
  echo -n "  DMPK_EP_ABLATE=1 in build, rank count: "
  grep -ho "DMPK_EP_ABLATE=1" /tmp/epabl_ctr_${ARM}.log | wc -l
done

echo "########################## WALL SUMMARY ##########################"
for ARM in base epablate; do
  echo -n "$ARM  "
  grep -hE '^\[1,0\].*Decode:' /tmp/glm5_epabl_${ARM}_r*.log \
    | grep -oE 'avg [0-9.]+ms/iter' | grep -oE '[0-9.]+' \
    | awk '{v[n++]=$1; s+=$1; if(n==1||$1<mn)mn=$1; if($1>mx)mx=$1}
           END{if(n)printf "n=%d mean=%.3f min=%.3f max=%.3f spread=%.3f\n",
                             n,s/n,mn,mx,mx-mn; else print "NO RESULTS"}'
done
echo EPABL_DONE
