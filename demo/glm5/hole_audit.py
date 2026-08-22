#!/usr/bin/env python3
"""Where is the FILLABLE hole in the layer, and how big is each one?

Offline.  Reads BARSTAGEWS per-worker rows already on disk.  No GPU run.

barrier_floor_sweep.py answered the first half: 6.498 of 12.098 ms is
LAST-ARRIVER FLOOR, time the critical worker itself spends.  This is the
complement.  If 54% is floor, 46% is slack -- and slack is where a lever has to
put work.  glm-decode-overlap-candidate-set-is-empty is the rule this serves:
price a move as min(window, movable), and the window is what this measures.

THE METRIC, AND THE ONE THAT LOOKED RIGHT AND IS NOT.
The obvious hole for region r is sum_w (max_r - wait_w): "how much spare time
does each worker have before the region's slowest finishes".  It DOUBLE COUNTS.
A worker that finishes a tile phase early stamps b early, so its slack shows up
again as a longer wait in the NEXT region, and again in the one after.  The
tell is that sum_r max_r = 262 us/layer against a 159 us layer -- a "hole"
decomposition that overflows the thing it decomposes is measuring a different
worker's clock in every row.  That version ranked S21->S23 first at 1.226 ms
and totalled 8.303; none of it survives.

The metric below references the FLOOR instead:

    hole_r = mean_w(wait_w,r) - floor_r          (+ the non-writers' full mean)

This telescopes exactly.  sum_r sum_w (wait_w,r - floor_r)
  = 232 * (layer_span - sum_r floor_r), because for a FIXED worker the
consecutive-stamp regions tile its layer with no gaps and no overlap.  Slack
counted in region r is therefore never counted again in r+1, and the totals
close: FLOOR 6.498 + HOLE 5.776 = 12.274 against a 12.017 ms measured span
(the 2% excess is the non-writer term, which is charged at the mean).

WHAT A HOLE IS, BY SHAPE -- do not read the RELEASE and SPREAD rows the same
way.  In a RELEASE region the extra time is provably spin: the workers left
together, so a worker that arrived early sat in the barrier.  In a SPREAD
region the extra time is UNEVEN WORK, not idle, and "filling" it means
rebalancing a tile phase, which is 3 for 3 negative
(glm-counted-region-time-before-a-barrier-is-not-a-lever).  Only the RELEASE
rows are capacity you can hand to someone else, and they total 2.277 ms.

AND A HOLE IN WORKER TIME IS NOT A HOLE IN BANDWIDTH.  The top row, S25->S26,
is the region the o_proj weight prefetch was deliberately built to fly through
(gpt-oss-prefetch-across-barrier-idiom, and S24->S25 is that prefetch's issue).
The VALUs are idle there; the memory system is not.  Check what is already in
flight before pricing a hole as free.
"""
import collections
import re
import statistics
import sys

PAT = re.compile(r"\[1,(\d+)\].*BARSTAGEWS (\d+) (\d+) (\d+) (\d+)")
N_LAYERS = 76

# Same verified map as barrier_floor_sweep.py.
ORDER = [0, 1, 2, 16, 17, 18, 19, 21, 23, 24, 25, 26, 27, 28,
         29, 30, 31, 32, 5, 6, 7, 8, 12, 13, 14]
LABEL = {
    (0, 1): "EP collective", (1, 2): "post-EP -> qkv_a",
    (2, 16): "-> attn call boundary", (16, 17): "qkv_a tiles",
    (17, 18): "qkv_a -> q_b barrier", (18, 19): "q_b tiles + KV append",
    (19, 21): "q_b->dec bar + MLA decode",
    (21, 23): "dec->merge bar + merge",
    (23, 24): "Ph8 attn -> o_proj bar", (24, 25): "o_proj wt prefetch",
    (25, 26): "per-XCD attn release", (26, 27): "prefetch DMA retired",
    (27, 28): "entering the MoE half", (28, 29): "W_UK/W_UV + Mech-C bar",
    (29, 30): "o_proj GEMV + residual", (30, 31): "o_proj -> router barrier",
    (31, 32): "router GEMV + TopK", (32, 5): "routing-ready poll",
    (5, 6): "W13 tiles  [CONTROL]", (6, 7): "W13 -> W2 barrier",
    (7, 8): "W2 tiles", (8, 12): "MoE exit", (12, 13): "(12->13)",
    (13, 14): "layer boundary",
}


def load(path, rank):
    rows = collections.defaultdict(dict)
    for ln in open(path, errors="ignore"):
        m = PAT.search(ln)
        if not m:
            continue
        r, s, w, c, t = (int(x) for x in m.groups())
        if c and r == rank:
            rows[s][w] = t / c / 1000.0
    return rows


def shape_of(sda, sdb):
    if sdb < 0.5 * sda:
        return "RELEASE"
    if sda < 0.5 * sdb:
        return "SPREAD"
    if max(sda, sdb) < 0.5:
        return "UNIFORM"
    return "mixed"


def report(path, rank):
    rows = load(path, rank)
    if 0 not in rows:
        print(f"rank {rank} not in {path}")
        return
    nw = len(set().union(*(set(rows[s]) for s in rows)))
    out, tot_f, tot_h, tot_m, rel_h = [], 0.0, 0.0, 0.0, 0.0
    for a, b in zip(ORDER, ORDER[1:]):
        A, B = rows.get(a, {}), rows.get(b, {})
        both = sorted(set(A) & set(B))
        if len(both) < 8:
            continue
        wait = [B[w] - A[w] for w in both]
        floor, mean = min(wait), statistics.mean(wait)
        sda = statistics.pstdev([A[w] for w in both])
        sdb = statistics.pstdev([B[w] for w in both])
        sh = shape_of(sda, sdb)
        miss = nw - len(both)
        # slack of the participants, plus the non-participants' whole region
        hole = ((mean - floor) * len(both) + miss * mean) / nw
        out.append((hole, a, b, floor, mean, sh, len(both), miss))
        tot_f += floor * N_LAYERS / 1e3
        tot_m += mean * N_LAYERS / 1e3
        tot_h += hole * N_LAYERS / 1e3
        if sh == "RELEASE":
            rel_h += hole * N_LAYERS / 1e3
    print(f"\n{path}   rank {rank}   {nw} workers   {N_LAYERS} layers")
    print("=" * 88)
    print(f"{'region':<38}{'shape':>9}{'n':>5}{'floor':>8}{'mean':>8}"
          f"{'hole':>8}{'hole':>8}")
    print(f"{'':<38}{'':>9}{'':>5}{'us/l':>8}{'us/l':>8}{'us/l':>8}{'ms':>8}")
    print("=" * 88)
    for hole, a, b, floor, mean, sh, n, miss in sorted(out, reverse=True):
        tag = f"S{a}->S{b} " + LABEL.get((a, b), "")
        print(f"{tag:<38}{sh:>9}{n:>5}{floor:>8.2f}{mean:>8.2f}"
              f"{hole:>8.2f}{hole * N_LAYERS / 1e3:>8.3f}")
    print("=" * 88)
    print(f"  layer span (mean per-worker) {tot_m:.3f} ms"
          f"   =   FLOOR {tot_f:.3f}  +  HOLE {tot_h:.3f}  (closes to 2%)")
    print(f"  RELEASE rows only -- provable spin, the only fillable part:"
          f" {rel_h:.3f} ms")
    print("  SPREAD rows are UNEVEN WORK, not idle.  Filling one means "
          "rebalancing a tile\n  phase, which is 3 for 3 negative.")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/shdup1/ctr_base.log"
    for rank in (int(x) for x in (sys.argv[2:] or ["1", "4"])):
        report(path, rank)


if __name__ == "__main__":
    main()

# ============================================================================
# RESULT, 2026-08-22.  /tmp/shdup1/ctr_base.log, ranks 1 and 4 (agree to 3%).
# No GPU run.  THE HOLE RANKING IS THE INVERSE OF THE BOARD RANKING.
# ============================================================================
#
#   FLOOR 6.498 ms  +  HOLE 5.776 ms  =  12.274 vs a 12.017 ms measured span.
#
#   region                          shape   floor  mean    HOLE   board price
#   S25->S26 per-XCD attn release  RELEASE   2.76  14.56  0.896 ms   0.210
#   S32->S5  routing-ready poll    RELEASE   3.80  12.80  0.684      0.289
#   S16->S17 qkv_a tiles            SPREAD   0.76   9.57  0.669      0.918
#   S31->S32 router GEMV + TopK     SPREAD   0.75   9.16  0.649      1.102
#   S21->S23 dec->merge bar+merge    mixed   1.17   9.70  0.648      0.434
#   S6->S7   W13 -> W2 barrier     RELEASE   5.67  12.15  0.492      0.433
#   S19->S21 q_b->dec + MLA decode  SPREAD   1.38   6.27  0.372      1.323
#   S5->S6   W13 tiles [CONTROL]    SPREAD   1.78   6.31  0.344      0.959
#   ...16 more, none above 0.24 ms                        0.922
#   S0->S1   EP collective         UNIFORM  24.92  25.02  0.015 ms   1.899
#
# (1) *** THE BIGGEST ITEM ON THE BOARD HAS NO HOLE AT ALL. ***  The EP
# collective is 1.899 ms of critical path and 0.015 ms of slack -- mean 25.02
# against a floor of 24.92, i.e. all 232 workers on the rank are inside it for
# the same 25 us.  It is the LEAST fillable region in the layer.  Symmetrically
# the top two holes, S25->S26 and S32->S5, are priced 0.210 and 0.289 on the
# board.  A budget built on max(b)-max(a) and a capacity map are close to
# ANTI-correlated, and both are right for their own question
# (glm-layer-is-54pct-last-arriver-floor, finding 4).
#
# (2) ONLY 2.277 ms IS PROVABLE SPIN.  Summing the four RELEASE rows.  The
# SPREAD rows look like holes and are not -- a worker that "has slack" in a
# tile phase is a worker with less work, and the fix is a rebalance, which is
# 3 for 3 negative.  Do not add 5.776 and call it available.
#
# (3) THE METRIC THAT LOOKED RIGHT AND DOUBLE COUNTS.  Referencing each
# worker's slack to the region MAX instead of the region FLOOR gives a
# different ranking (S21->S23 first at 1.226 ms) and a 8.303 ms total inside a
# 12.098 ms layer.  It is wrong because an early finisher's slack reappears in
# every later region: sum_r max_r is 262 us/layer against a 159 us layer.  The
# floor reference telescopes to 232 * (span - sum floor) exactly and is the
# only one that can be summed.  General rule: a decomposition that overflows
# the thing it decomposes is measuring a different clock in every row.
#
# (4) THE TOP HOLE ALREADY HAS A DMA IN IT.  S25->S26 "per-XCD attention
# release" is where the o_proj weight prefetch issued at S24->S25 is in flight
# (gpt-oss-prefetch-across-barrier-idiom).  0.896 ms of idle VALU over a busy
# memory system is not 0.896 ms of free capacity.  A worker-time hole and a
# bandwidth hole are different holes; check what is already flying before
# pricing one as free.  Compare glm-moe-phase-has-a-free-worker-hole, where the
# 21/29 idle workers absorbed a COLD READ for +0.030 -- that hole was free
# because nothing was using the bus.
#
# (5) THE PRICE OF ANY MOVE IS STILL min(window, movable), and this file
# measures only the window.  glm-decode-overlap-candidate-set-is-empty found a
# 0.749 ms window with zero movable work, because the MPK task graph is a
# strictly linear chain (mirage-task-graph-is-a-linear-chain).  Nothing here
# changes that; it says where to look, not that something is there.
#
# ============================================================================
# THE MOVABLE SIDE, same day.  All four RELEASE holes read at the source.
# min(window, movable) = 0 FOR ALL FOUR.  The hole side of the layer is closed.
# ============================================================================
#
#   hole                       ms    what is already in the window
#   S25->S26 attn release   0.896    42-load o_proj weight DMA + 17 loads of
#                                    shared-expert weights, ISSUED AT S24->S25
#                                    exactly so they fly through this spin
#   S32->S5  routing poll   0.684    NOTHING.  Bare spin, idle bus.
#   S6->S7   W13 -> W2      0.492    W2 L2 prefetch, MEASURED NEUTRAL 0.13%
#   S17->S18 qkv_a -> q_b   0.203    under the 0.26 ms noise floor
#
# (6) THE TOP HOLE IS ALREADY FILLED AS FAR AS THE IDIOM GOES.  S24->S25 is the
# prefetch issue and S25->S26 is the spin it flies through
# (gang_mla_full_layer_fused_mi300.cuh:1809-2037, the gpt-oss Phase 6 idiom).
# The in-source measurement is that it "shrinks the release by ~2.5 us per
# worker per layer", against a 11.80 us hole, and the same comment already
# states the verdict: "the remaining 27 us of Phase 8 spin is a load-balance
# problem, not a latency-hiding one."  Widening it is refuted there too -- at
# GLM-5 shapes PF_WG_BYTES is 528 KB/worker = 15.3 MB/XCD against a 4 MB L2,
# so the working set is already 3.8x L2 and most of it is evicted before
# Phase 9 reads it.  Nothing more fits.
#
# (7) *** YOU CANNOT PREFETCH THROUGH A BARRIER WHOSE OUTPUT IS THE ADDRESS. ***
# S32->S5 is the one RELEASE hole in the layer with an idle memory system --
# gang_oproj_router_fused_mi300.cuh:1174-1195 is a bare ld_nt_s32 spin with no
# DMA issued before it.  In bandwidth terms the window is enormous: 9 us at
# 5.17 TB/s is 46 MB, against a 65 MB/layer whole-layer byte budget
# (glm-layer-byte-budget-and-barrier-share).  And the movable set is still
# empty, because the thing the poll is WAITING FOR is expert identity, and
# expert identity is what selects the bytes.  The window and the dependency are
# the same object.  Every unconditionally-known operand is already resident:
# the shared expert's weights were prefetched at the o_proj barrier, and the
# source notes "this XCD's L2 still holds the normed row that Phase 5 is about
# to consume".  The only escape is a SPECULATIVE prefetch of predicted experts,
# which is a different project and needs a prediction first.
#
# (8) SO THE HOLE SIDE MATCHES THE FLOOR SIDE: nothing left that is tuning.
# 2.277 ms of provable spin, and each piece is either already covered by a DMA,
# already measured neutral, under the noise floor, or blocked by the fact that
# its own dependency is an address.  Same conclusion the board reached from the
# critical-path direction (glm-board-audit-no-region-above-0.4ms-unattacked),
# reached independently from the capacity direction.
