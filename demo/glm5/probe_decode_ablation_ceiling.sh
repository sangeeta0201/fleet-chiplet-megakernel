#!/bin/bash
# THE MLA DECODE IDLE HOLE -- price the ceiling before building any overlap.
#
# WHY.  The board audit (77781dc, demo/glm5/board_budget.py) puts S19->S21
# (q_b->decode barrier + MLA decode) at 18.31 us/layer crit against a dTYP of
# only 1.72 -- 16.59 us/layer, 1.26 ms/token, of SPIN.  Decode runs on 64 of
# 232 workers (NUM_Q_GROUPS=4 x NUM_KV_CHUNKS=16); the rest wait.  That is the
# largest single spin term left on the board and the one class of lever this
# branch has not exhausted: it is a SCHEDULING target, not a tuning knob.
#
# Before writing any overlap, measure what the wall actually returns if the
# decode work vanishes.  That delta is the HARD CEILING on any scheme that
# hides decode under another phase -- an overlap can at best recover the time
# the machine spends waiting on decode, never more.
#
# ONE VARIABLE: MPK_MLA_SKIP_DECODE (0 vs 1).  It deletes the Phase 5 decode
# loop (gang_mla_attn_fused_mi300.cuh:1328) and NOTHING else -- every barrier,
# every other phase, the same tile map, the same worker count all still run.
# Wired at python/mirage/mpk/persistent_kernel.py:607 as a -D, so each arm is
# its own build; run_mp8_dp_ep_fused.sh:53 clears permanent_output_dir* on the
# rebuild run, and bench_repeat.sh sets KEEP_BUILD=1 only for runs 2..N, so
# every repeat inside an arm is the same binary.
#
# n=3 per arm, pooled.  The wall noise floor is 0.26 ms and this probe is
# looking for something near it, so n=1 would not resolve the answer
# (glm5-wall-noise-floor-is-0.26ms).
#
# TWO THINGS THIS NUMBER IS NOT, both already on the ledger:
#
#  1. NOT A SPEEDUP.  Wrong output by construction -- o_acc/lse keep the
#     previous layer's values.  No latency claim may be quoted from it.  It is
#     a ceiling and only a ceiling.
#  2. NOT CLEAN OF THE ROUTER CONFOUND.  Decode is UPSTREAM of the router, so
#     garbage attention output changes TopK and therefore EP expert balance
#     (glm-wrong-output-probes-upstream-of-router-are-invalid).  The MoE half's
#     time can move for a reason that has nothing to do with decode.  The
#     mitigation is the one the earlier harvest used: the ablated floor must be
#     INSENSITIVE to decode geometry.  At 4 chunks it read 10.788 and at 16
#     chunks 10.771 -- with decode deleted the chunk count SHOULD be
#     irrelevant, and was.  Treat a floor that lands near ~10.77 as
#     corroborated and a floor far from it as contaminated.
#
# PRIOR: 11.427 -> 10.771 = 0.66 ms at the 16-chunk geometry
# (glm-decode-ablation-is-worth-3.4ms).  The baseline has since moved to
# 10.619, so that 0.66 needs re-pricing against the CURRENT build, which is
# what this script is for.
#
# BAR SET BY THE RULING: if the ceiling comes back under ~0.4 ms, close the
# decode idle hole as a NO-GO rather than build an overlap already priced out.
#
# ===================== RESULT, 2026-08-22, n=3 per arm =====================
#   arm                        runs                    mean    min     max
#   base     (SKIP_DECODE=0)   10.681 10.531 10.553   10.588  10.531  10.681
#   skipdec  (SKIP_DECODE=1)    9.734  9.979  9.804    9.839   9.734   9.979
#
#   CEILING ON DELETING DECODE = 0.749 ms  (9.86 us/layer over 76 layers)
#
# Arms are fully DISJOINT: base_min 10.531 - skipdec_max 9.979 = +0.552 ms.
# 2.9x the 0.26 ms noise floor. Flag verified in the build on all 8 ranks in
# the skipdec arm and on 0 in the base arm (the two-step-rebuild guard).
#
# CORROBORATED by the earlier harvest at a different baseline: 11.427 -> 10.771
# = 0.656 ms then, 10.588 -> 9.839 = 0.749 ms now. The baseline fell 0.84 ms
# between the two on non-decode work and the decode complex stayed ~0.7 ms, so
# the router/EP-balance confound (see caveat 2) did not swamp the signal.
#
# ============ WHAT THE 0.749 SPLITS THE REGION INTO ============
# board_budget.py has S19->S21 (q_b->decode barrier + MLA decode) at 18.31
# us/layer crit = 1.392 ms. The ablation removes the WORK and keeps every
# barrier, so it cleaves the region cleanly:
#
#   decode's own WORK (ablated away)      9.86 us/layer   0.749 ms
#   q_b->decode barrier + skew (SURVIVES) 8.45 us/layer   0.643 ms
#
# ============ THE OVERLAP CEILING IS NOT THE 0.749 ============
# THIS IS THE POINT OF THE PROBE, and it is easy to get backwards.  0.749 ms is
# the ceiling on DELETING decode's work.  An overlap scheme cannot delete it --
# decode's output is required.  Overlap fills the window in which the 168
# non-decode workers are idle with work that would otherwise run LATER, so:
#
#   gain = min(window, legally movable work) = min(0.749 ms, X)
#
# X is the duration of work whose inputs are all ready BEFORE decode starts and
# which currently runs AFTER it.  Enumerated against the region map, X is
# EMPTY.  The intra-layer chain is strictly serial through decode:
#
#   qkv_a -> q_b + KV append -> DECODE -> merge -> W_UV -> o_proj -> router
#         -> W13 -> W2
#
#   * W_UV reads oproj_input_ptr, the MERGED attention output
#     (gang_oproj_router_fused_mi300.cuh:467) -- downstream. Not movable.
#   * o_proj, router, W13, W2 are each downstream of the one before it.
#   * layer L+1's qkv_a needs layer L's MoE output
#     (glm-heterogeneous-worker-groups-impossible).
#   * the KV-cache append must precede decode: decode reads the entry it writes.
#
# The only decode-independent work in the layer is WEIGHT PREFETCH, and both
# forms are already measured out:
#   * the o_proj weight prefetch ALREADY EXISTS at stamps 24-27 and already
#     hides the fetch (glm-mxfp4-oproj-buys-0.14ms-not-0.9).
#   * handing idle workers a cold read measured +0.030, a slight NEGATIVE
#     (glm-moe-phase-has-a-free-worker-hole).
#
# VERDICT: the ceiling CLEARS the 0.4 ms bar, but OVERLAP IS A NO-GO anyway --
# not on the size of the window, on the emptiness of the candidate set.  The
# 0.749 ms is reachable only by making decode's own work cheaper, and that is
# capped: decode width is NUM_Q_GROUPS(4) x NUM_KV_CHUNKS(16) = 64 tiles of 232
# workers, q_groups is fixed at NUM_Q_HEADS/16 by the MFMA tile, and more
# chunks multiply the 3.96 us/call of fixed cost (prologue+epilogue+refill)
# that does not shrink with KV length -- which is why chunks were capped at 16
# and why 8/16 measured null before the harvest
# (glm-decode-ablation-is-worth-3.4ms).
# ======================================================================
set -u
ulimit -c 0
cd /home/claudeuser/fleet-chiplet-megakernel/demo/glm5

export MODEL_PATH=/home/claudeuser/models/glm5-mxfp4
# Instruments OFF. MPK_SUBPHASE_TIMING's cost scales with tile count and decode
# tile count is exactly what this ablation removes -- it would forge the answer
# (glm-subphase-timing-cost-scales-with-tile-count: 3.1 ms at 16 chunks, and it
# once fully masked a real 2.76 ms win).
export MPK_SUBPHASE_TIMING=0
export MPK_BAR_SKEW=0
export MPK_PRINT_ALL_RANKS=0

N="${N:-3}"

for ARM in base skipdec; do
  if [ "$ARM" = base ]; then export MPK_MLA_SKIP_DECODE=0
  else export MPK_MLA_SKIP_DECODE=1; fi
  # -D change => new build. Belt and braces on top of run_mp8_dp_ep_fused.sh:53.
  unset KEEP_BUILD
  rm -rf permanent_output_dir permanent_output_dir_rank*
  echo "########## ARM $ARM  MPK_MLA_SKIP_DECODE=$MPK_MLA_SKIP_DECODE ##########"
  ./bench_repeat.sh "declab_${ARM}" "$N"
  # The flag must be in the build on ALL EIGHT ranks, not just requested in the
  # env. A -D that silently did not reach the compile is how a null result gets
  # manufactured (mirage-cpp-edits-need-a-two-step-rebuild).
  echo -n "  DMPK_MLA_SKIP_DECODE in build, rank count: "
  grep -ho "DMPK_MLA_SKIP_DECODE" /tmp/glm5_declab_${ARM}_r1.log | wc -l
done

echo "########## CEILING ##########"
for ARM in base skipdec; do
  echo -n "$ARM  "
  grep -hE '^\[1,0\].*Decode:' /tmp/glm5_declab_${ARM}_r*.log \
    | grep -oE 'avg [0-9.]+ms/iter' | grep -oE '[0-9.]+' \
    | awk '{v[n++]=$1; s+=$1; if(n==1||$1<mn)mn=$1; if($1>mx)mx=$1}
           END{if(n)printf "n=%d mean=%.3f min=%.3f max=%.3f spread=%.3f\n",
                             n,s/n,mn,mx,mx-mn; else print "NO RESULTS"}'
done
echo DECLAB_DONE
