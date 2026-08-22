#!/bin/bash
# THE q_b HEAD-SHARD CROSS-RANK GATHER -- price the ceiling on head-sharding
# attention end-to-end, before writing any of it.
#
# WHY.  5cb68f6 showed the 0.643 ms that SURVIVES the MLA decode ablation
# (f02753c) is not a local barrier: it is the QB_TP cross-rank peer wait at
# gang_mla_attn_fused_mi300.cuh:1193-1240.  Under the q_b head shard the
# elected barrier leader stores this rank's head slice to all 7 peers'
# ep_signal lines and polls all 7 before releasing the local flag.  The layer
# therefore pays TWO cross-rank rendezvous per layer:
#
#     S0->S1   EP collective            17.5 us/layer   1.187 ms
#     S19->S20 q_b head-shard gather      8.8 us/layer   0.667 ms
#
# Head-sharding attention end-to-end (decode + merge + W_UV stay on the rank's
# own heads, the cross-rank combine defers to o_proj's ALREADY EXISTING
# ep_signal all-gather) would DELETE this rendezvous rather than narrow one.
# That is the one shape the "skew just relocates" verdict does not kill on its
# face (rule 3 of e5d1ff5 requires the deleted wait to share a producer set
# with a neighbour; here the rewrite removes the producer set too).
#
# ONE VARIABLE: MPK_QB_SKIP_PEER_WAIT (0 vs 1).  It deletes the 7 peer stores
# and the 7-peer poll and NOTHING else -- the local hierarchical barrier, the
# flag release, the self-heal and every other phase all still run.
#
# THE NUMBER THAT DECIDES THE REWRITE IS NOT "0.667 leaves S19->S20".  It is
# how much survives at the WALL after S22 and S28 re-absorb the freed skew.
# e5d1ff5 is the cautionary case: a whole rendezvous carrying 8.28 us/layer of
# uniform spin was deleted and the wall moved 0.003 ms, because the skew
# reappeared at the next barrier.  So this probe reports BOTH routes and the
# wall is the one that counts.
#
# WRONG OUTPUT BY CONSTRUCTION: the query row keeps the peers' previous-layer
# heads, so decode reads 7/8 stale heads.  No latency claim may be quoted from
# it -- ceiling only.  And it is UPSTREAM OF THE ROUTER, so garbage attention
# changes TopK and therefore EP expert balance
# (glm-wrong-output-probes-upstream-of-router-are-invalid).  Mitigation, same
# as the decode ablation's: require the two routes to agree.  A wall delta far
# larger than the counter delta means the router confound moved the MoE half.
#
# TWO PHASES:
#   A  wall, instruments OFF, n=3 per arm            -> the deciding number
#   B  region counters, MPK_BAR_SKEW=3 on BOTH arms  -> the corroborating route
#      (probe cost common-mode, the method dddf699 used when the wall could
#      not resolve the EP poll batch)
#
# BAR SET BY THE RULING: >= 0.4 ms recovered at the wall -> commit the ceiling
# and come back for a go/no-go before writing the rewrite.  < 0.4 ms -> close
# head-sharding as a NO-GO with the measured number.
#
# ===================== RESULT, 2026-08-22, n=3 per arm =====================
#   arm                          runs                     mean    min     max
#   base     (SKIP_PEER_WAIT=0)  10.677 10.381 10.412   10.490  10.381  10.677
#   skippeer (SKIP_PEER_WAIT=1)  10.103 10.239 10.124   10.155  10.103  10.239
#
#   CEILING ON DELETING THE q_b GATHER = 0.335 ms
#   *** SUPERSEDED -- see the n=6 pooled block below, the ceiling is 0.228 ***
#
# Arms disjoint: base_min 10.381 - skippeer_max 10.239 = +0.142 ms.  Flag
# verified in the build on 8 ranks in the skippeer arm and 0 in base.
#
# ROUTE 2, region counters, MPK_BAR_SKEW=3 on BOTH arms (common-mode), pooled
# over 8 ranks (compare_qb_peer_counters.py).  us/layer, makespan:
#
#   S19->S20  THE PEER WAIT ITSELF     9.22 -> 5.11   -4.11   -0.312 ms
#   S18->S22  qkv_a bar -> dec/merge  33.33 -> 29.55  -3.78   -0.287 ms
#   S18->S28  whole attention tail    44.46 -> 40.65  -3.82   -0.290 ms
#   S22->S28  ABSORBER TO WATCH       11.13 -> 11.09  -0.04   -0.003 ms
#   S0->S14   LAYER SPAN             159.26 -> 155.59 -3.67   -0.279 ms
#
# THE TWO ROUTES AGREE: wall -0.335, layer span -0.279, attention tail -0.290.
# All three inside 0.06 ms of each other.
#
# AND THE SKEW DOES NOT RELOCATE.  S22->S28 moves -0.04 us/layer (median +0.03)
# -- essentially zero.  ~92% of what leaves S19->S20 survives all the way to
# the layer span.  This is NOT the e5d1ff5 pattern; deleting THIS rendezvous
# really does delete its time.  The rule 3 worry was the right one to test and
# it came back clean.
#
# ============ BUT IT IS STILL A NO-GO, AND IT CORRECTS A NUMBER ============
# 5cb68f6 attributed 8.78 us/layer = 0.667 ms to this peer wait, by subtracting
# the local arrival skew at S19 (max-med = 3.90) from the median S19->S20 wait
# (12.68).  THAT SUBTRACTION WAS WRONG BY 2x.  Deleting the store+poll outright
# removes only 4.11 us/layer.  The other ~6.5 us of the S19->S20 wait is local
# barrier mechanism and skew that the deletion does not touch -- "not explained
# by arrival spread" is not the same as "cross-rank", and only the ablation can
# tell them apart.  The two-cross-rank-rendezvous FINDING stands; its PRICE was
# 2x too high.
#
# 0.335 ms is under the ruling's 0.4 ms bar, so head-sharding attention
# end-to-end CLOSES AS A NO-GO.  And 0.335 is a generous ceiling: it assumes
# the rendezvous is deleted at zero cost, whereas the real rewrite still has to
# combine the head-sharded attention output across ranks.  That payload is
# comparable in size to the query row it would stop gathering (128 heads x 512
# latent vs 128 x (512+64)), so the rewrite MOVES a transfer onto o_proj's
# existing gather rather than deleting one.  Realized gain is strictly less
# than 0.335 ms, against a rewrite touching decode, merge, W_UV and o_proj.
#
# =========== RE-RUN, same day, n=3 MORE PER ARM -> n=6 POOLED ===========
# Both phases re-run end to end under identical env.  The verdict is UNCHANGED
# and firmer, but THE CEILING NUMBER IS CORRECTED DOWNWARD: 0.335 -> 0.228 ms.
#
# PHASE A, the deciding number.  Second triple: base 10.559 10.560 10.402,
# skippeer 10.513 10.387 10.258 -> a delta of only +0.121 ms with the arms
# OVERLAPPING.  The first triple's +0.335 was the high tail of a noisy n=3.
#
#   arm       n   mean     min     max     sd
#   base      6   10.498   10.381  10.677  0.118
#   skippeer  6   10.271   10.103  10.513  0.157
#   ------------------------------------------------------------------
#   CEILING ON DELETING THE q_b GATHER = 0.228 ms   (was published as 0.335)
#
# Ranges overlap, but the MEANS are resolvable: pooled sd 0.139, se of the
# difference 0.080, t = 2.84 on df 10 (p ~ 0.02).  So the effect is real and
# small -- 0.228 ms is BELOW the 0.26 ms wall noise floor for a single pair
# (glm5-wall-noise-floor-is-0.26ms) and only n=6 pooling separates it at all.
# This is exactly the case that memory file exists to warn about: the n=3
# number was not reproducible and the direction of the error was optimistic.
# Flag verified in the build on 8 ranks in skippeer, 0 in base, both phases.
#
# PHASE B, the counters, fully re-run.  THE COUNTER ROUTE IS REPRODUCIBLE TO
# ~0.15 us/layer where the wall was not.  Pooled over 8 ranks, us/layer:
#
#   region                          MAKESPAN(crit)      run1     MEDIAN(typ)
#   S19->S20 THE PEER WAIT ITSELF   9.06 ->  5.13  -3.94  (-4.11)  -4.15
#   S18->S22 qkv_a bar -> dec/merge 33.31 -> 29.44 -3.87  (-3.78)  -3.85
#   S18->S28 whole attention tail   44.43 -> 40.72 -3.71  (-3.82)  -3.64
#   S22->S28 ABSORBER TO WATCH      11.12 -> 11.27 +0.16  (-0.04)  +0.22
#   S28->S5  MoE half (CONFOUNDED)  38.57 -> 38.08 -0.49     n/a   -0.54
#   S0->S14  LAYER SPAN            159.74 ->155.92 -3.82  (-3.67)  -4.00
#
#   layer span -3.82 us/layer x 76 = -0.290 ms   (run 1: -0.279)
#
# THE SKEW STILL DOES NOT RELOCATE, now confirmed twice.  S22->S28 moves +0.16
# us/layer makespan / +0.22 median -- zero within the region's own spread, and
# it moved -0.04 the first time, i.e. it flips sign between runs.  ~92% of what
# leaves S19->S20 survives to the layer span.  Deleting THIS rendezvous really
# does delete its time; the e5d1ff5 relocation rule does not bite here.
#
# BUT THE TWO ROUTES NO LONGER AGREE, AND THAT IS THE HONEST HEADLINE.
# Run 1 had wall -0.335 vs span -0.279, inside 0.06.  At n=6 the wall is -0.228
# and the span is -0.290 -- the counters now claim 27% MORE than the wall.  Two
# readings, and this probe cannot separate them:
#   (a) MPK_BAR_SKEW=3 is common-mode in LEVEL but not in SLOPE.  The stamps
#       serialize a store per worker per region; deleting a 4 us wait shortens
#       the window the instrument's own cost hides in, so the instrumented arm
#       can show a larger delta than the uninstrumented one.
#   (b) the router confound.  The skippeer arm's output is COHERENT BUT
#       DIVERGENT (both arms write a sensible paragraph about Rayleigh
#       scattering, with different wording), so TopK differs and the MoE half's
#       expert balance is not the same experiment.  S28->S5 moving -0.49
#       us/layer is that confound, not a result.
# Either way the wall is the route that counts and it is the SMALLER one.
#
# ================== VERDICT: NO-GO, at 0.228 ms not 0.335 ==================
# Under the ruling's 0.4 ms bar by both routes and by every pooling.  Head-
# sharding attention end-to-end is CLOSED.  The generosity argument above only
# gets stronger at the lower number: 0.228 ms is the ceiling for deleting the
# rendezvous at ZERO cost, while the real rewrite still has to combine the
# head-sharded output across ranks -- a payload comparable to the query row it
# stops gathering.  Realized gain is strictly less than 0.228 ms, against a
# rewrite touching decode, merge, W_UV and o_proj.  Not worth it.
#
# METHOD NOTE.  The first base counter arm HUNG: rc=124 at the 1800 s timeout,
# 0 BARSTAGEWS lines, log ending at "launch_persistent_kernel ENTER".  That is
# the known ~1-in-3 NP=8 EP hang (glm-np8-fault-has-two-presentations), NOT a
# consequence of the flag -- the flag is 0 on that arm.  GPUs were confirmed
# idle with no orphan ranks, and the arm alone was re-run under identical env:
# rc=0, 52671 BARSTAGEWS lines.
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
for ARM in base skippeer; do
  if [ "$ARM" = base ]; then export MPK_QB_SKIP_PEER_WAIT=0
  else export MPK_QB_SKIP_PEER_WAIT=1; fi
  unset KEEP_BUILD
  rm -rf permanent_output_dir permanent_output_dir_rank*
  echo "########## ARM $ARM  MPK_QB_SKIP_PEER_WAIT=$MPK_QB_SKIP_PEER_WAIT ##########"
  ./bench_repeat.sh "qbpeer_${ARM}" "$N"
  # The -D must be in the build on ALL EIGHT ranks. A flag that never reached
  # the compile is how a null result gets manufactured
  # (mirage-cpp-edits-need-a-two-step-rebuild).
  echo -n "  DMPK_QB_SKIP_PEER_WAIT in build, rank count: "
  grep -ho "DMPK_QB_SKIP_PEER_WAIT" /tmp/glm5_qbpeer_${ARM}_r1.log | wc -l
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
for ARM in base skippeer; do
  if [ "$ARM" = base ]; then export MPK_QB_SKIP_PEER_WAIT=0
  else export MPK_QB_SKIP_PEER_WAIT=1; fi
  unset KEEP_BUILD
  rm -rf permanent_output_dir permanent_output_dir_rank*
  export MASTER_PORT=$((32600 + (RANDOM % 150)))
  echo "===== COUNTER ARM $ARM ====="
  timeout 1800 ./run_mp8_dp_ep_fused.sh \
      --prompt "Write a short paragraph explaining why the sky appears blue." \
      > "/tmp/qbpeer_ctr_${ARM}.log" 2>&1
  echo "  rc=$?  BARSTAGEWS lines: $(grep -c BARSTAGEWS /tmp/qbpeer_ctr_${ARM}.log)"
  echo -n "  DMPK_QB_SKIP_PEER_WAIT in build, rank count: "
  grep -ho "DMPK_QB_SKIP_PEER_WAIT" /tmp/qbpeer_ctr_${ARM}.log | wc -l
done

echo "########################## WALL SUMMARY ##########################"
for ARM in base skippeer; do
  echo -n "$ARM  "
  grep -hE '^\[1,0\].*Decode:' /tmp/glm5_qbpeer_${ARM}_r*.log \
    | grep -oE 'avg [0-9.]+ms/iter' | grep -oE '[0-9.]+' \
    | awk '{v[n++]=$1; s+=$1; if(n==1||$1<mn)mn=$1; if($1>mx)mx=$1}
           END{if(n)printf "n=%d mean=%.3f min=%.3f max=%.3f spread=%.3f\n",
                             n,s/n,mn,mx,mx-mn; else print "NO RESULTS"}'
done
echo QBPEER_DONE
