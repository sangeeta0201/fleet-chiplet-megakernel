#!/usr/bin/env python3
"""GEOM decomposed by phase and CLASSIFIED.

GEOM is the 2.210 ms that widening the build from 1 to 2 rows costs BEFORE any
second row is live -- `--max-num-batched-tokens 2`, no MTP, no draft, no
accept/reject.  The second row is dispatched but dead: nothing reads it.  It is
the largest pure-overhead term on the board and, unlike per-row expert traffic
(glm-moe-row-fold-ceiling), it has no proven floor.

INSTRUMENT: MPK_BAR_SKEW=3 (private per-worker rows) + MPK_BAR_SKEW_DROP_NS=
1000000, MPK_SUBPHASE_TIMING=0.  Produced by probe_row_cost_phases.sh; the
per-phase arithmetic is ws_phase.py.  Both arms, rank 0, 232 workers each.

    crit(a->b) = max_w mean_w[b] - max_w mean_w[a]     the critical path
    typ(a->b)  = median_w ...                          a normal worker
    dSPIN      = dCRIT - dTYP                          how far the tail moved

THE THREE CLASSES (the board's question):

  (i)   DEAD-ROW BYTES     work sized by BATCH_SIZE through a tile count that
                           does NOT change -- activations, scratch, scales and
                           outputs moved for a row nobody reads.
  (ii)  EXTRA TILES        tiles dispatched for the dead row.
  (iii) BARRIER / SKEW     rendezvous and spin that grew because the geometry
                           is wider, not because more work was done.

(i) vs (ii) is NOT decidable from the stamps -- it is decided from the task
builder, where the two shapes are textually distinct
(python/mirage/mpk/persistent_kernel.py):

  gang_oproj_router_fused_layer:5857  oproj_tiles_per_xcd = n_wgs // 8
      -> NO batch_size.  o_proj's tile count is FIXED; bs=2 widens each tile
         (m_per_tile = batch_size).  Pure (i).

  gang_oproj_router_fused_layer:5856  moe_max_activated = min(topk*bs, E)
                              :5857-5860  moe_w{13,2}_tiles_per_xcd =
                                  (moe_max_activated * batch_size * wgs + 7)//8
      -> QUADRATIC in batch_size: the MoE tile SPACE is 4x at bs=2.  With
         tiles_per_expert = n_tiles * batch_size on the device, half the
         dispatched tiles hit
             if (d_routing[e*BATCH_SIZE + tok] == 0) return false;
         and return BEFORE fetching any weight.  So they are cheap per tile but
         there are twice as many.  (ii).

  gang_mla_full_layer_fused_layer:2894  tiles_per_xcd =
        min(max(batch_size*qkv_n_wgs_per_xcd, ...), num_workers//8)
      -> also bs-scaled, but already clamped to 29 at bs=1, so bs=2 only adds
         grid-stride rounds.  (ii), and measurably tiny.

Run:  python3 demo/glm5/classify_geom.py
"""
import re
from collections import defaultdict

BS1_LOG = "/tmp/item11_bs1.log"
BS2_LOG = "/tmp/item11_bs2.log"
N_LAYERS = 76          # stamped layers per iteration in this build
GEOM_MS = 2.210        # the wall term this table is explaining (87106c0)

# region -> (class, share_of_dCRIT_or_None, why).  A share of None means the
# whole dCRIT goes to that class; a pair splits dTYP / dSPIN between two.
CLASS = {
    (0, 1):   ("SPLIT", "EP collective -- see the per-rank block below"),
    (29, 30): ("i",   "oproj_tiles_per_xcd = n_wgs//8 has NO batch_size, so the"
                      " tile count is FIXED and m_per_tile goes 1->2: a second"
                      " activation row in, a second output row out, per tile."),
    (31, 32): ("i",   "the router TopK loop is over batch_size rows; dTYP"
                      " (+2.103) exceeds dCRIT, so every worker really does it."),
    (28, 29): ("i",   "attn tail: dTYP tracks dCRIT, fixed tile count."),
    (7, 8):   ("ii/iii", "W2: tiles_per_expert = n_tiles*batch_size. dTYP is"
                      " walking the dead row's tiles (decode+return, no fetch);"
                      " dSPIN is the tail that lengthens behind them."),
    (5, 6):   ("ii/iii", "W13: same doubling. dTYP is only +0.352 -- exactly"
                      " what a decode-and-return costs -- and the rest is tail."),
    (2, 16):  ("ii",  "qkv_a tile count is bs-scaled but already clamped to 29"
                      " workers/XCD at bs=1, so bs=2 only adds a stride round."),
    (30, 31): ("iii", "router: dTYP ~0 (-0.023), all of it is tail."),
    (6, 7):   ("iii", "the W13->W2 rendezvous itself."),
    (8, 12):  ("iii", "MoE exit: dTYP ~0 (-0.068)."),
    (32, 5):  ("iii", "the routing poll -- and it goes NEGATIVE (-0.597)."),
}

ORDER = [0, 1, 2, 16, 17, 18, 19, 21, 23, 24, 25, 26, 27, 28,
         29, 30, 31, 32, 5, 6, 7, 8, 12, 13, 14]
LABEL = {(0, 1): "EP collective", (2, 16): "qkv_a tiles",
         (16, 17): "attn: q_b / W_UK", (17, 18): "attn: W_UV",
         (18, 19): "attn: decode prep", (19, 21): "attn: MLA decode",
         (21, 23): "attn: merge", (28, 29): "attn tail f", (29, 30): "o_proj",
         (30, 31): "router", (31, 32): "router tail",
         (32, 5): "routing poll", (5, 6): "W13 tiles",
         (6, 7): "W13->W2 BARRIER", (7, 8): "W2 tiles",
         (8, 12): "MoE exit", (12, 13): "loop top", (13, 14): "pointer refresh"}


def load(path):
    pat = re.compile(r"\[1,(\d+)\].*BARSTAGEWS (\d+) (\d+) (\d+) (\d+)")
    d = defaultdict(lambda: defaultdict(dict))
    for line in open(path, errors="replace"):
        if "BARSTAGEWS" not in line:
            continue
        m = pat.search(line)
        if not m:
            continue
        r, s, w, c, t = (int(x) for x in m.groups())
        if c:
            d[r][s][w] = (c, t)
    return d


def stats(rank_rows):
    out = {}
    for s, ws in rank_rows.items():
        means = sorted(t / c for c, t in ws.values())
        out[s] = {"max": means[-1] / 1000.0,
                  "med": means[len(means) // 2] / 1000.0,
                  "cnt": sum(c for c, _ in ws.values()) // len(ws)}
    return out


def cwmean(sl, workers=None):
    """Count-weighted mean.  MANDATORY for S4: it is stamped once per layer by
    whichever fold group arrives LAST, so per-worker counts run 4 .. 13726 and
    a max-over-workers is dominated by the 4-sample worker (it reads -12.968,
    i.e. the collective getting FASTER with more work)."""
    it = [(c, t) for w, (c, t) in sl.items() if workers is None or w in workers]
    return sum(t for _, t in it) / sum(c for c, _ in it) / 1000.0


A, B = load(BS1_LOG), load(BS2_LOG)
a, b = stats(A[0]), stats(B[0])

# ---------------------------------------------------------------- per-rank EP
print("=" * 78)
print("THE EP COLLECTIVE (S0->S1), PER RANK -- the biggest single line")
print("=" * 78)
print("  Rank 0 alone cannot classify this: per-rank stamps are relative to each")
print("  rank's OWN layer entry and carry no cross-rank arrival order.  All 8")
print("  ranks emit BARSTAGEWS, so read all 8.")
print()
print(f"  {'rank':>4} | {'LOCAL FOLD S0->S3':^25} | {'PEER WAIT S3->S4':^27}")
print(f"  {'':>4} | {'bs1':>7} {'bs2':>7} {'delta':>8} |"
      f" {'bs1':>8} {'bs2':>8} {'delta':>8}")
ep = {}
for r in range(8):
    row = []
    for D in (A, B):
        fw = set(D[r][3])          # the 8 folding groups -- DIFFERENT ids per arm
        row.append((cwmean(D[r][3], fw) - cwmean(D[r][0], fw),
                    cwmean(D[r][4]) - cwmean(D[r][3], fw),
                    cwmean(D[r][1], fw) - cwmean(D[r][4])))
    (f1, p1, r1), (f2, p2, r2) = row
    ep[r] = (f2 - f1, p2 - p1, r2 - r1)
    print(f"  {r:>4} | {f1:7.3f} {f2:7.3f} {f2-f1:+8.3f} |"
          f" {p1:8.3f} {p2:8.3f} {p2-p1:+8.3f}")
fold0, peer0, rel0 = ep[0]
print()
print(f"  LOCAL FOLD grows UNIFORMLY: {min(v[0] for v in ep.values()):+.3f} to"
      f" {max(v[0] for v in ep.values()):+.3f} on all 8 ranks.  That is VOLUME")
print(f"  -- the EP fold buffer is sized by BATCH_SIZE and the dead row's slots")
print(f"  are folded and published like any other.  Class (i).")
print()
print(f"  PEER WAIT does NOT: rank 0 {peer0:+.3f}, peers"
      f" {min(ep[r][1] for r in range(1,8)):+.3f} to"
      f" {max(ep[r][1] for r in range(1,8)):+.3f}.  Rank 0 waits HALF what its")
print(f"  peers wait and grows half as fast, because rank 0 IS the straggler")
print(f"  (the shared expert lives there) and the other seven wait on IT.  A")
print(f"  volume story would grow uniformly, like the fold does.  Class (iii).")
print()
print(f"  Rank 0 is the one rank on which this is invisible.  Reading only the")
print(f"  rank you happen to print is how a volume story survives.")

# ------------------------------------------------------------- the main table
print()
print("=" * 78)
print("GEOM BY PHASE (rank 0, us/layer), CLASSIFIED")
print("=" * 78)
print(f"  {'region':<26} {'dCRIT':>8} {'dTYP':>8} {'dSPIN':>8}  class")
print("  " + "-" * 74)
tot = {"i": 0.0, "ii": 0.0, "iii": 0.0}
sum_dcrit = 0.0
rows = []
for i in range(len(ORDER) - 1):
    s0, s1 = ORDER[i], ORDER[i + 1]
    if s0 not in a or s1 not in a or s0 not in b or s1 not in b:
        continue
    def mism(d):
        lo, hi = sorted((d[s0]["cnt"], d[s1]["cnt"]))
        return (hi - lo) / hi if hi else 1.0
    if mism(a) > 0.01 or mism(b) > 0.01:
        continue                       # different writer populations: not a duration
    dc = (b[s1]["max"] - b[s0]["max"]) - (a[s1]["max"] - a[s0]["max"])
    dt = (b[s1]["med"] - b[s0]["med"]) - (a[s1]["med"] - a[s0]["med"])
    sum_dcrit += dc
    cls = CLASS.get((s0, s1), (None, ""))[0]
    lab = LABEL.get((s0, s1), f"S{s0}->S{s1}")
    if cls == "SPLIT":
        tot["i"] += fold0
        tot["iii"] += peer0 + rel0
        shown = f"(i) {fold0:+.3f} / (iii) {peer0+rel0:+.3f}"
    elif cls == "ii/iii":
        tot["ii"] += dt
        tot["iii"] += dc - dt
        shown = f"(ii) {dt:+.3f} / (iii) {dc-dt:+.3f}"
    elif cls in tot:
        tot[cls] += dc
        shown = f"({cls})"
    else:
        shown = "unclassified (small)"
    rows.append((abs(dc), f"  S{s0:>2}->S{s1:<3} {lab:<17} {dc:8.3f} {dt:8.3f}"
                          f" {dc-dt:8.3f}  {shown}"))
for _, line in sorted(rows, key=lambda t: -t[0]):
    print(line)
print("  " + "-" * 74)
span = ((b[14]["max"] - b[0]["max"]) - (a[14]["max"] - a[0]["max"]))
classified = sum(tot.values())
print(f"  {'SUM OF REGION DELTAS':<26} {sum_dcrit:8.3f}")
print(f"  {'MEASURED LAYER SPAN S0->S14':<26} {span:8.3f}"
      f"   residual {sum_dcrit-span:+.3f} us/layer ({abs(sum_dcrit-span)/span*100:.2f}%)")

print()
print("=" * 78)
print("THE CLASSIFICATION")
print("=" * 78)
for k, name in (("i", "DEAD-ROW BYTES  (fixed tile count, BATCH_SIZE-sized work)"),
                ("ii", "EXTRA TILES     (dispatched for the dead row)"),
                ("iii", "BARRIER / SKEW  (wider geometry, not more work)")):
    ms = tot[k] * N_LAYERS / 1000.0
    print(f"  ({k:<3}) {name:<56} {tot[k]:7.3f} us/layer"
          f"  {ms:6.3f} ms  {100*tot[k]/span:5.1f}%")
resid = span - classified
print(f"  {'unclassified (attn tails a-e, decode/merge, loop top, ptr)':<62}"
      f" {resid:7.3f} us/layer  {resid*N_LAYERS/1000.0:6.3f} ms  {100*resid/span:5.1f}%")
print(f"  {'TOTAL':<62} {span:7.3f} us/layer"
      f"  {span*N_LAYERS/1000.0:6.3f} ms")
print(f"\n  cross-check: {span:.3f} us/layer x {N_LAYERS} ="
      f" {span*N_LAYERS/1000.0:.3f} ms against the {GEOM_MS} ms GEOM measured at"
      f" the wall\n  (87106c0, n=3/n=2).  The instrument accounts for"
      f" {100*span*N_LAYERS/1000.0/GEOM_MS:.0f}% of it.")

# ------------------------------------------------- is GEOM uniform across ranks?
# The (iii) bucket is 58%, so the obvious follow-up is "the wider geometry
# widens the per-rank SPREAD, and a rendezvous charges the spread, not the
# mean".  That is a HYPOTHESIS and this block FALSIFIES it.
def mk(rows, s):
    return max(t / c for c, t in rows[s].values()) / 1000.0


print()
print("=" * 78)
print("IS GEOM UNIFORM ACROSS RANKS?  (the spread hypothesis, FALSIFIED)")
print("=" * 78)
print(f"  {'rank':>4} {'span bs1':>10} {'span bs2':>10} {'dGEOM':>9}")
spans = {}
for arm, D in (("1", A), ("2", B)):
    spans[arm] = [mk(D[r], 14) - mk(D[r], 0) for r in range(8)]
for r in range(8):
    print(f"  {r:>4} {spans['1'][r]:10.3f} {spans['2'][r]:10.3f}"
          f" {spans['2'][r]-spans['1'][r]:+9.3f}")
s1 = max(spans["1"]) - min(spans["1"])
s2 = max(spans["2"]) - min(spans["2"])
dg = [spans["2"][r] - spans["1"][r] for r in range(8)]
print(f"\n  cross-rank SPREAD of the layer span: {s1:.3f} -> {s2:.3f}"
      f"  ({s2-s1:+.3f} us/layer)")
print(f"  GEOM per rank: {min(dg):+.3f} to {max(dg):+.3f}, range"
      f" {max(dg)-min(dg):.3f} us/layer on a ~26.6 mean.")
print(f"""
  GEOM IS UNIFORM.  Every one of the 8 ranks pays +25.7..+27.1 us/layer, and
  the cross-rank spread of the whole layer grows by only {s2-s1:+.3f} us/layer --
  1.6% of GEOM.  The peer wait moves a lot per rank (+7.1 on rank 0, +11.9 to
  +14.1 on the others) but the layer TOTAL does not diverge, because that is
  exactly what a rendezvous does: it re-synchronises.

  So the (iii) bucket is SPIN THAT REDISTRIBUTES, not skew that can be removed.
  There is no rank-local fix and no skew fix that reaches GEOM: shortening one
  rank's share just lengthens its wait at the next collective.  Compare
  glm-counted-region-time-before-a-barrier-is-not-a-lever (3 for 3) and
  glm-deleting-a-whole-rendezvous-is-neutral (8.28 us/layer of UNIFORM spin
  deleted, wall moved 0.003).""")

# ------------------------------------------------ which regions are rank-local?
print()
print("=" * 78)
print("PER-RANK dCRIT: which regions are UNIFORM (attackable) vs RANK-LOCAL")
print("=" * 78)
for s0, s1_, lab in ((29, 30, "o_proj"), (7, 8, "W2 tiles"),
                     (5, 6, "W13 tiles"), (6, 7, "W13->W2 bar"),
                     (2, 16, "qkv_a tiles")):
    d = [(mk(B[r], s1_) - mk(B[r], s0)) - (mk(A[r], s1_) - mk(A[r], s0))
         for r in range(8)]
    peers = sum(d[1:]) / 7
    print(f"  {lab:<13} " + " ".join(f"{x:+6.2f}" for x in d)
          + f"   r0-peers {d[0]-peers:+6.2f}")
print(f"""
  o_proj is UNIFORM (+3.78..+4.50 on all 8) -- real dead-row work everywhere.
  The MoE tile phases are NOT: rank 0 pays +6.72 W2 / +2.63 W13 against peer
  means of +2.65 / +0.64.  Rank 0 owns the shared expert, the shared expert
  processes BATCH_SIZE rows, so at bs=2 it does two -- one of them for the dead
  row.  Rank-0 excess over peers across both MoE tile phases: +6.062 us/layer
  = 0.461 ms, 22% of GEOM.

  CAUTION, and it is not resolved here: glm-shared-expert-hoist-is-zero-makespan
  measured rank 0 as having the SHORTEST W13 makespan of the 8, which is the
  opposite sign to this table.  Different instrument (SP counters vs
  BARSTAGEWS), different build.  Do not spend the 0.461 ms until that is
  reconciled -- and note the uniformity result above says a rank-local win is
  absorbed by the collective anyway.""")

print()
print("=" * 78)
print("THE NEXT LEVER")
print("=" * 78)
print(f"""  SUPERSEDED by the uniformity result above: the EP collective's PEER WAIT,
  {peer0+rel0:+.3f} us/layer on rank 0 = {(peer0+rel0)*N_LAYERS/1000.0:.3f} ms, is the largest classified term by
  size, but it is a redistribution: the layer total does not diverge across
  ranks, so removing it just moves the wait.

  THE LEVER IS o_proj: {(b[30]['max']-b[29]['max'])-(a[30]['max']-a[29]['max']):+.3f} us/layer = {((b[30]['max']-b[29]['max'])-(a[30]['max']-a[29]['max']))*N_LAYERS/1000.0:.3f} ms.  It is the only large term
  that is all three of: pure (i) dead-row bytes, UNIFORM across all 8 ranks
  (+3.78..+4.50, so no rank absorbs it for the others), and dTYP {(b[30]['med']-b[29]['med'])-(a[30]['med']-a[29]['med']):+.3f} of
  dCRIT {(b[30]['max']-b[29]['max'])-(a[30]['max']-a[29]['max']):+.3f} -- every worker really does it, there is no barrier inside it,
  and its tile count does not change (oproj_tiles_per_xcd = n_wgs // 8).  It
  computes and writes a second output row for a token nobody reads.

  0.303 ms is above the 0.26 ms wall noise floor, but only just: gate it on the
  region counter, not on n=1 at the wall.""")
