#!/usr/bin/env python3
"""ITEM 1.1 -- per-phase bs=1 vs bs=2 decomposition from BARSTAGEWS rows.

Input: run logs produced with MPK_BAR_SKEW=3 MPK_BAR_SKEW_DROP_NS=1000000.

`BARSTAGEWS <slot> <worker> <cnt> <sum>` is a PRIVATE per-worker row: worker w's
sum of (ns since this layer's entry reference) over cnt layers. So per worker,
mean_w[s] = sum/cnt is that worker's own average arrival at stamp s.

Two statistics per stamp, and the pair is the whole point:

  makespan(s) = max over workers of mean_w[s]   -- the critical path
  typical(s)  = median over workers of mean_w[s]

A region [a -> b] then splits three ways, which is exactly the (a)/(b)/(c)
question the board asks:

  crit(a->b)   = makespan(b) - makespan(a)   how long the region really takes
  typ(a->b)    = typical(b)  - typical(a)    how long it takes a normal worker
  spin(s)      = makespan(s) - typical(s)    how far the stragglers are out

  (b) LONGER TILES  -> typ rises. Every worker takes longer.
  (c) MORE SPIN     -> typ flat, crit rises: only the tail moved.
  (a) MORE TILES    -> typ rises AND the rise tracks the tile-count ratio;
                       separated from (b) by the geometry, not by the stamps,
                       so tiles/worker is printed alongside when known.

CAUTION baked in from prior wrong turns:
  * Stamps are relative to each rank's OWN layer-entry reference, so these
    carry no cross-rank arrival order. Same-rank regions only.
  * Slots have different populations (cnt). Differencing two slots whose cnt
    disagrees is not a duration -- flagged below rather than silently divided.
  * S6->S7 is the W13->W2 BARRIER, not W13. W13 is S6-S5. Verified at the
    stamp sites 2026-08-22.
"""
import sys, re, statistics
from collections import defaultdict

# Verified against the mpk_stage_stamp() call sites, 2026-08-22.
ORDER = [0, 1, 2, 16, 17, 18, 19, 21, 23, 24, 25, 26, 27, 28,
         29, 30, 31, 32, 5, 6, 7, 8, 12, 13, 14]
LABEL = {
    (0, 1):   "EP collective (fold+peer wait+release)",
    (1, 2):   "post-EP -> qkv_a entry",
    (2, 16):  "qkv_a tiles",
    (16, 17): "attn: q_b / W_UK",
    (17, 18): "attn: W_UV",
    (18, 19): "attn: decode prep",
    (19, 21): "attn: MLA decode",
    (21, 23): "attn: merge",
    (23, 24): "attn tail a", (24, 25): "attn tail b", (25, 26): "attn tail c",
    (26, 27): "attn tail d", (27, 28): "attn tail e", (28, 29): "attn tail f",
    (29, 30): "o_proj",
    (30, 31): "router",
    (31, 32): "router tail",
    (32, 5):  "routing poll -> MoE entry",
    (5, 6):   "W13 tiles",
    (6, 7):   "W13->W2 BARRIER wait",
    (7, 8):   "W2 tiles",
    (8, 12):  "MoE exit -> layer bookkeeping",
    (12, 13): "loop top", (13, 14): "pointer refresh",
}

def load(path, rank="0"):
    pat = re.compile(r"BARSTAGEWS (\d+) (\d+) (\d+) (\d+)")
    rows = defaultdict(dict)   # slot -> worker -> (cnt, sum)
    want = f"[1,{rank}]"
    for line in open(path, errors="replace"):
        if "BARSTAGEWS" not in line:
            continue
        if want not in line and "[1," in line:
            continue
        m = pat.search(line)
        if not m:
            continue
        s, w, c, t = (int(x) for x in m.groups())
        if c:
            rows[s][w] = (c, t)
    return rows

def stats(rows):
    out = {}
    for s, ws in rows.items():
        means = sorted(t / c for c, t in ws.values())
        if not means:
            continue
        out[s] = {
            "n": len(means),
            "cnt": sum(c for c, _ in ws.values()) // max(1, len(ws)),
            "max": means[-1] / 1000.0,
            "med": statistics.median(means) / 1000.0,
        }
    return out

def table(a, b, na, nb):
    print(f"\n{'region':<40} {'crit1':>8} {'crit2':>8} {'dCRIT':>8} "
          f"{'typ1':>8} {'typ2':>8} {'dTYP':>8} {'dSPIN':>8}  verdict")
    print("-" * 116)
    tot = [0.0, 0.0, 0.0]
    for i in range(len(ORDER) - 1):
        s0, s1 = ORDER[i], ORDER[i + 1]
        if not all(s in a and s in b for s in (s0, s1)):
            continue
        # Population guard, RELATIVE not exact. The real hazard is slots with
        # genuinely different writer populations -- one thread vs 232, or
        # stamped only for ml>0 -- which differ by whole factors (S9/S10/S11
        # vs S12/S13/S14 differed by 62562 samples). A 1-3 sample difference
        # out of ~38000 is a dropped sample and rejecting it throws away 21.5
        # of the 27.1 us/layer being measured. 1% is far below any real
        # population split and far above the drop rate.
        def mism(d):
            lo, hi = sorted((d[s0]["cnt"], d[s1]["cnt"]))
            return (hi - lo) / hi if hi else 1.0
        if mism(a) > 0.01 or mism(b) > 0.01:
            print(f"S{s0}->S{s1:<3} SKIPPED: population mismatch "
                  f"({na} {a[s0]['cnt']}/{a[s1]['cnt']}, "
                  f"{nb} {b[s0]['cnt']}/{b[s1]['cnt']}) -- not a duration")
            continue
        c1 = a[s1]["max"] - a[s0]["max"]
        c2 = b[s1]["max"] - b[s0]["max"]
        t1 = a[s1]["med"] - a[s0]["med"]
        t2 = b[s1]["med"] - b[s0]["med"]
        dc, dt = c2 - c1, t2 - t1
        ds = dc - dt
        lab = LABEL.get((s0, s1), f"S{s0}->S{s1}")
        if abs(dc) < 0.30:
            v = "-"
        elif dt > 0.7 * dc:
            v = "(b) LONGER/MORE TILES"
        elif abs(dt) < 0.3 * abs(dc):
            v = "(c) MORE SPIN"
        else:
            v = "(b)+(c) mixed"
        print(f"S{s0:>2}->S{s1:<3} {lab:<29} {c1:8.3f} {c2:8.3f} {dc:8.3f} "
              f"{t1:8.3f} {t2:8.3f} {dt:8.3f} {ds:8.3f}  {v}")
        tot[0] += dc; tot[1] += dt; tot[2] += ds
    print("-" * 116)
    print(f"{'SUM OF REGION DELTAS':<40} {'':>8} {'':>8} {tot[0]:8.3f} "
          f"{'':>8} {'':>8} {tot[1]:8.3f} {tot[2]:8.3f}")
    for s_lo, s_hi in ((0, 14), (0, 8)):
        if s_lo in a and s_hi in a and s_lo in b and s_hi in b:
            span1 = a[s_hi]["max"] - a[s_lo]["max"]
            span2 = b[s_hi]["max"] - b[s_lo]["max"]
            print(f"LAYER SPAN S{s_lo}->S{s_hi} (crit): {span1:.3f} -> "
                  f"{span2:.3f} us/layer, delta {span2-span1:+.3f} "
                  f"= {(span2-span1)*76/1000.0:+.3f} ms over 76 layers")

if __name__ == "__main__":
    f1, f2 = sys.argv[1], sys.argv[2]
    rank = sys.argv[3] if len(sys.argv) > 3 else "0"
    A, B = stats(load(f1, rank)), stats(load(f2, rank))
    print(f"bs=1 log: {f1}\nbs=2 log: {f2}   (rank {rank})")
    print(f"slots: bs1 {sorted(A)}\n       bs2 {sorted(B)}")
    if A:
        s = sorted(A)[0]
        print(f"workers reporting: bs1 {A[s]['n']}, bs2 {B[s]['n'] if s in B else '?'}")
    table(A, B, "bs1", "bs2")
