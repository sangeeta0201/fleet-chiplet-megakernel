#!/bin/bash
# THE GEOM PROBE -- what does WIDENING THE BUILD to 2 rows cost before any
# second row is live?
#
# RENAMED 2026-08-22.  This was written as "where does the second decode row's
# 5.958 ms go", which is the wrong question for this arm.  With
# --max-num-batched-tokens 2 and no MTP the build is BATCH_SIZE=2 but still
# decodes ONE token per iteration -- the second row is dispatched and DEAD.
# So this pair measures GEOM (2.210 ms at the wall, 87106c0), not ROWACT.
# That makes it the right instrument: every microsecond it finds is overhead
# by construction, because nothing reads the row it is spent on.
# Classification: demo/glm5/classify_geom.py.
#
# ONE VARIABLE: --max-num-batched-tokens (1 vs 2). In persistent_kernel.py
# `batch_size = self.max_num_batched_tokens` feeds every task template
# directly, so this is a pure 2-decode-row build: no MTP layer, no draft, no
# accept/reject, no extra task in the graph. The second row just decodes a
# duplicate. That isolates the ROW COST from everything MTP wraps around it.
#
# Instrument: MPK_BAR_SKEW=3 -> private per-worker rows (BARSTAGEWS), which is
# the only mode that yields a makespan rather than a mean. Two traps already
# paid for and not re-entered:
#   * MPK_BAR_SKEW_DROP_NS=1000000 is MANDATORY. Its default 10 ms guard keeps
#     one ml=0 sample per iteration and inflated the layer boundary 20.18 vs
#     8.21 us.
#   * MPK_SUBPHASE_TIMING stays OFF. Its cost scales with tile count, and tile
#     count is exactly the variable here -- it would forge the answer.
#
# Not gated for output: both arms decode the same prompt and bs=2's second row
# is a duplicate, so this is a region-counter measurement, not a wall claim.
# No latency number from this script is quotable as a speedup.
#
# ===================== RESULT, 2026-08-22, rank 0 =====================
# 232 workers reporting in each arm; 52672 / 53116 BARSTAGEWS rows.
#
#   region                          bs=1     bs=2   dCRIT    dTYP   dSPIN  verdict
#   S0->S1  EP collective         15.617   25.381  +9.765  +9.753   0.012  (b)
#   S7->S8  W2 tiles              12.407   19.129  +6.721  +2.271  +4.450  (b)+(c)
#   S29->S30 o_proj                7.392   11.380  +3.988  +3.899   0.089  (b)
#   S5->S6  W13 tiles             18.447   21.081  +2.634  +0.352  +2.282  (c)
#   S30->S31 router                2.690    3.891  +1.201  -0.023  +1.224  (c)
#   S31->S32 router tail          14.152   15.158  +1.006  +2.103  -1.097  (b)
#   S28->S29 attn tail f           9.949   10.663  +0.714  +0.863  -0.150  (b)
#   S6->S7  W13->W2 barrier        5.310    5.861  +0.551  +2.978  -2.427  (b)
#   S8->S12 MoE exit               3.358    3.897  +0.539  -0.068  +0.607  (c)
#   S32->S5 routing poll           4.168    3.571  -0.597  -0.407  -0.190
#   SUM OF REGION DELTAS                          +27.133 +21.844  +5.289
#   LAYER SPAN S0->S14           159.188  186.301 +27.113  = +2.061 ms / 76 layers
#
# The model closes with no residual: the region deltas sum to +27.133 against a
# measured layer span of +27.113. Top four regions are 85% of the cost.
#
# ANSWER TO THE (a)/(b)/(c) QUESTION: 80% (b) longer/more tiles, 20% (c) spin.
# Hypothesis (c) -- "extra rows land on busy workers while idle ones wait" -- is
# NOT what is happening, which retires the "the machine is 50% idle so it should
# swallow a second row" framing.
#
# EP COLLECTIVE SUB-DECOMPOSITION (the biggest line, 36% of the row cost).
# Restricted to the 8 folding work-groups, which are S3/S4's writer set:
#
#   local EP fold (compute + store drain)    4.896 ->  7.195   +2.299   23.6%
#   peer wait (cross-rank publish)           9.331 -> 16.544   +7.213   73.9%
#   release fan-out                          1.537 ->  1.782   +0.245    2.5%
#   whole collective S0->S1                 15.765 -> 25.522   +9.757
#
# A rank-0-only reading of that peer wait says "1.77x for a 2x payload, and all
# ranks are symmetric, so it is publish VOLUME." THAT IS WRONG, and it is wrong
# for the reason already on file: per-rank stamps are relative to each rank's
# OWN barrier and carry no cross-rank arrival order. All 8 ranks emit
# BARSTAGEWS, so run the decomposition per rank and the answer inverts:
#
#   rank |      LOCAL FOLD       |       PEER WAIT
#        |  bs1    bs2   delta   |  bs1     bs2    delta
#     0  | 4.896  7.195  +2.299  |  9.331  16.544  +7.213   <- shared expert
#     1  | 4.831  7.013  +2.183  | 18.563  31.590 +13.027
#     2  | 4.832  7.089  +2.257  | 20.345  32.480 +12.136
#     3  | 4.608  7.136  +2.528  | 18.539  30.561 +12.022
#     4  | 5.013  7.002  +1.989  | 18.782  33.233 +14.451
#     5  | 4.758  7.067  +2.310  | 19.637  31.884 +12.247
#     6  | 4.697  7.146  +2.449  | 19.498  30.986 +11.488
#     7  | 4.798  6.961  +2.162  | 19.484  31.389 +11.905
#
# The LOCAL FOLD grows uniformly (+2.2 on every rank) -- that half really is
# volume. The PEER WAIT does not: rank 0 waits HALF what its peers wait and
# grows half as fast, because rank 0 is the straggler and the other seven are
# waiting on IT. The bs=1 gap of ~9.8 us/layer matches the independently
# measured 9.09 us/layer shared-expert imbalance. So the peer wait's growth is
# SKEW, and the second row roughly doubles it: ~9.8 -> ~15.2 us/layer, which is
# ~1.16 ms over 76 layers.
#
# Rank 0 is the one rank on which this effect is invisible. Measuring only the
# rank you happen to print is how the volume story survives.
#
# TWO GUARDS, both of which produced wrong numbers before they were read:
#   * The 8 folding work-groups have DIFFERENT worker ids in the two arms
#     (43,70,101,... vs 61,65,66,...) -- the fold lands on whichever workers
#     claim the task. Pair within an arm; cross-arm same-worker pairing is void.
#   * S4 is stamped ONCE PER LAYER by whichever group arrives LAST, so its
#     per-worker cnts are 4 / 361 / 1502 / ... / 13726, summing to one per
#     layer. A max-over-workers of S4 is dominated by the worker holding 4
#     samples and reads -12.968 us, i.e. the collective getting FASTER with more
#     work. Use the count-weighted mean; it is a last-arriver by construction
#     and therefore already the critical path.
#
# ===================== THE CLASSIFICATION, 2026-08-22 =====================
# demo/glm5/classify_geom.py assigns each region to one of three classes.
# (i) vs (ii) is NOT decidable from the stamps -- it comes from the task
# builder, where the two shapes are textually distinct:
#   oproj_tiles_per_xcd = n_wgs // 8            <- NO batch_size: FIXED tile
#                                                  count, m_per_tile 1->2
#   moe_w{13,2}_tiles_per_xcd =
#       (min(topk*bs, E) * bs * wgs + 7) // 8   <- QUADRATIC in bs: 4x tile
#                                                  SPACE, and the dead half
#                                                  hits d_routing==0 and
#                                                  returns BEFORE any fetch
#
#   class                                       us/layer     ms   share
#   (i)   DEAD-ROW BYTES  (fixed tile count)       8.085  0.614   29.8%
#   (ii)  EXTRA TILES     (dispatched, dead)       2.770  0.210   10.2%
#   (iii) BARRIER / SKEW  (wider geometry)        15.777  1.199   58.2%
#   unclassified (attn tails, decode/merge)        0.482  0.037    1.8%
#   TOTAL                                         27.113  2.061  = 93% of the
#                                                                  2.210 ms
#                                                                  wall GEOM
#
# GEOM IS NOT DEAD WORK. It is 58% RENDEZVOUS. Only 10.2% is the extra tiles
# the dead row dispatches -- because those tiles return before fetching a
# weight slab, exactly as the source says they do (W13's dTYP is +0.374, which
# IS a decode-and-return). "Stop dispatching tiles for the dead row" is worth
# 0.21 ms, not 2.2.
#
# LARGEST SINGLE TERM: the EP collective's peer wait, +7.368 us/layer on
# rank 0 = 0.560 ms, 27% of GEOM -- and +11.9..+14.1 on the seven ranks that
# wait on rank 0. It is (iii).
#
# ============ VERDICT: GEOM IS A NO-GO. 0.062-0.201 ms deletable ============
# Two corrections landed on top of the table above, both self-caught, both from
# the SAME logs -- no extra GPU run:
#
# 1. THE SKEW HYPOTHESIS IS FALSIFIED (a9018e5). "The wider geometry widens the
#    per-rank spread" is wrong: per-rank layer span dGEOM is +27.11 / +26.58 /
#    +25.79 / +26.45 / +26.73 / +25.68 / +26.19 / +26.54 -- UNIFORM. The
#    cross-rank spread grows only 0.792 -> 1.232 = +0.440 us/layer, 1.6% of
#    GEOM. So the (iii) bucket REDISTRIBUTES; there is no skew to remove.
#
# 2. THE o_proj LEVER IS RETRACTED. It was named as "pure (i) dead-row bytes,
#    0.303 ms". The source says the row is not dead in the sense that matters:
#      gang_gemv_mxfp8_mi300.cuh:396  if (m_tile*BATCH_SIZE+m >=
#        num_active_tokens) continue;  -- inside `if (lane == 0)`, the EPILOGUE.
#        grep -n num_active_tokens on that header returns exactly TWO hits,
#        :180 (the parameter) and :396. The k-loop accumulates acc[m][*] for all
#        BATCH_SIZE rows UNCONDITIONALLY.
#      gang_oproj_router_fused_mi300.cuh:791  push_rows -- the peer all-gather
#        push is ALREADY liveness-clamped.
#    So the +3.988 IS the second row's accumulation, and a LIVE row needs it.
#    Production MTP at bs=2 runs num_active_tokens=2: nothing is dead there.
#
# THE DELETABILITY TEST. A microsecond measured on the dead-row arm is only a
# lever if it stays dead when the row goes live:
#   (i)   0.614 ms -- work a live row also needs.        NOT deletable.
#   (iii) 1.199 ms -- uniform spin, redistributes.       NOT deletable.
#   (ii)  0.210 ms -- the only partly-dead class; MoE tile space is quadratic
#         in bs while live pairs only double. Priced from dTYP of the MoE tile
#         phases (the decode-and-return a dead tile actually costs):
#           peer-typical  0.817 us/layer = 0.062 ms
#           rank 0        2.641 us/layer = 0.201 ms   (shared expert, 2 rows)
#         BOTH under the 0.26 ms wall noise floor.
#
# CONSEQUENCE: this CONFIRMS glm-item2-width2-closed-on-ceiling. C = BS2/BS1 =
# 1.206 prices build width and 1.206 cannot be reduced, so item 2's ceiling
# stands with its largest input independently verified.
# ======================================================================
set -u
ulimit -c 0
cd /home/claudeuser/fleet-chiplet-megakernel/demo/glm5

export MODEL_PATH=/home/claudeuser/models/glm5-mxfp4
export MAX_SEQ_LENGTH=512
export MAX_NEW_TOKENS=96
export MPK_BAR_SKEW=3
export MPK_BAR_SKEW_DROP_NS=1000000
export MPK_SUBPHASE_TIMING=0
export MPK_PRINT_ALL_RANKS=0

for ARM in bs1 bs2; do
  if [ "$ARM" = bs1 ]; then
    export MPK_EXTRA_ARGS=""
  else
    export MPK_EXTRA_ARGS="--max-num-batched-tokens 2"
  fi
  export MASTER_PORT=$((32100 + (RANDOM % 150)))
  # Tile geometry changes with BATCH_SIZE, so each arm needs its own build.
  rm -rf permanent_output_dir permanent_output_dir_rank*
  echo "===== ARM $ARM  EXTRA='${MPK_EXTRA_ARGS}' ====="
  timeout 1800 ./run_mp8_dp_ep_fused.sh \
      --prompt "Write a short paragraph explaining why the sky appears blue." \
      > "/tmp/item11_${ARM}.log" 2>&1
  echo "  rc=$?"
  echo -n "  BATCHED_TOKENS in build: "
  grep -ho "MPK_MAX_NUM_BATCHED_TOKENS=[0-9]*" /tmp/item11_${ARM}.log | sort -u | tr '\n' ' '
  echo
  echo -n "  BARSTAGEWS rows: "
  grep -c BARSTAGEWS "/tmp/item11_${ARM}.log"
  grep -hoE "[0-9.]+ ms/iter|Decode:[^,]*" "/tmp/item11_${ARM}.log" | tail -3
done

echo "############ ITEM 1.1 TABLE ############"
python3 /tmp/ws_phase.py /tmp/item11_bs1.log /tmp/item11_bs2.log 0
echo ITEM11_DONE
