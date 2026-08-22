#!/bin/bash
# ITEM 1.1 -- where does the second decode row's 5.958 ms go?
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
