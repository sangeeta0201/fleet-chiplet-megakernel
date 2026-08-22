#!/usr/bin/env python3
"""LIVENESS + WORK-ADDED read for the MPK_SHARED_DUP probe.

Offline; reads the two phase-C logs written by probe_shared_expert_makespan.sh.
No GPU run.

WHAT THIS ANSWERS.  The wall arm of that probe is ambiguous on a null: a flag
that is in the build does NOT prove the duplicated tiles executed
(glm-moe-kloop-batching-is-neutral).  This reads S5->S6 -- the W13 tile phase --
PER RANK and asks two questions in order:

  1. LIVENESS.  Did rank 0's W13 grow?  MPK_SHARED_DUP runs rank 0's shared
     expert's W13 tiles twice and touches nothing else, so rank 0's S5->S6 must
     grow by roughly one shared expert's W13 time and ranks 1-7 must not move.
     If rank 0 does not move, the probe is dead and the wall says nothing.

  2. WHERE IT LANDS.  If rank 0 grew but the peers' WAIT did not, the extra
     work was absorbed by idle workers -- the "absorbed" model.  If the peers'
     wait at the first cross-rank rendezvous grew by the same amount, rank 0 is
     on the critical path -- the "on the path" model.  Those are the two
     hypotheses the probe exists to separate, and the wall should agree with
     whichever this says.

ESTIMATOR.  Count-weighted mean (`sum(t)/sum(c)`), not an unweighted median
over writers.  Slot 4 is stamped by exactly ONE worker per layer -- whichever
XCD leader actually waited -- and which one varies, so its 8 writers hold 108
to 11433 samples.  An unweighted median weights the 108-sample writer like the
11433-sample one; that error moved a published headline by 12%
(glm-ep-skew-is-half-shared-expert-bias).  Both statistics are printed for W13
because the closure under test used max-max and this one uses the mean, and a
disagreement between them is itself the finding.

WORK vs WAIT.  S5->S6 contains no cross-rank sync, so its per-rank spread is
real imbalance.  S3->S4 contains one, so it is the consequence, not the cause.
Never test for imbalance with a span that contains a cross-rank sync -- the
rendezvous forces the layer periods equal and the span goes flat regardless
(that trap produced and retracted 406375b).
"""
import collections
import os
import re
import sys

ARMS = ("base", "dup")
# MULT is the number of EXTRA shared-expert W13 copies in the dup arm: 1
# doubles rank 0's excess, 2 triples it.  Kept out of the filename default
# so each magnitude keeps its own log dir.
MULT = os.environ.get("MULT", "1")
D = f"/tmp/shdup{MULT}"
LOGS = {a: f"{D}/ctr_{a}.log" for a in ARMS}
PAT = re.compile(r"\[1,(\d+)\].*BARSTAGEWS (\d+) (\d+) (\d+) (\d+)")
N_LAYERS = 76


def load(path):
    """rank -> slot -> worker -> (count, total_ns).  Keep the COUNT."""
    d = collections.defaultdict(lambda: collections.defaultdict(dict))
    for ln in open(path, errors="ignore"):
        m = PAT.search(ln)
        if m:
            r, s, w, c, t = (int(x) for x in m.groups())
            if c:
                d[r][s][w] = (c, t)
    return d


def wmean(rows, s):
    if s not in rows:
        return None
    c = sum(x[0] for x in rows[s].values())
    t = sum(x[1] for x in rows[s].values())
    return t / c / 1000.0 if c else None


def wspan(rows, x, y):
    a, b = wmean(rows, x), wmean(rows, y)
    return None if a is None or b is None else b - a


def mkspan(rows, x, y):
    """max-max: the statistic the closure under test used."""
    if x not in rows or y not in rows:
        return None
    return (max(t / c for c, t in rows[y].values())
            - max(t / c for c, t in rows[x].values())) / 1000.0


R = {}
for a in ARMS:
    try:
        R[a] = load(LOGS[a])
    except FileNotFoundError:
        sys.exit(f"missing {LOGS[a]} -- run probe_shared_expert_makespan.sh")

SPANS = [
    ("WORK", 5, 6, "S5->S6   W13 tiles  <- THE DUPLICATED PHASE"),
    ("WORK", 7, 8, "S7->S8   W2 tiles   (must NOT move)"),
    ("WORK", 20, 14, "S20->S14 whole MoE half"),
    ("WAIT", 3, 4, "S3->S4   EP peer wait (8 XCD leaders)"),
    ("WAIT", 0, 1, "S0->S1   EP bracket"),
    ("TOTAL", 0, 14, "S0->S14  layer span [forced equal -- proves nothing]"),
]

print("=" * 100)
print("PER-RANK, count-weighted mean, us/layer.  base -> dup")
print("=" * 100)
print(f"{'kind':<6}{'span':<52}{'rank0':>9}{'peermn':>9}")
for kind, x, y, lab in SPANS:
    cells = []
    for a in ARMS:
        v = [wspan(R[a].get(r, {}), x, y) for r in range(8)]
        if any(u is None for u in v):
            cells = None
            break
        cells.append((v[0], sum(v[1:]) / 7))
    if not cells:
        continue
    (b0, bp), (d0, dp) = cells
    print(f"{kind:<6}{lab:<52}{b0:9.2f}{bp:9.2f}")
    print(f"{'':<6}{'':<52}{d0:9.2f}{dp:9.2f}")
    print(f"{'':<6}{'  delta (dup - base)':<52}"
          f"{d0-b0:+9.2f}{dp-bp:+9.2f}"
          f"   r0 {(d0-b0)*N_LAYERS/1e3:+.3f} ms,"
          f" peers {(dp-bp)*N_LAYERS/1e3:+.3f} ms")

print()
print("=" * 100)
print("W13 UNDER BOTH STATISTICS -- the closure under test used max-max")
print("=" * 100)
for nm, f in (("count-weighted mean", wspan), ("makespan (max-max)", mkspan)):
    for a in ARMS:
        v = [f(R[a].get(r, {}), 5, 6) for r in range(8)]
        if any(u is None for u in v):
            continue
        pm = sum(v[1:]) / 7
        rank = sorted(range(8), key=lambda r: v[r]).index(0) + 1
        print(f"  {nm:<22}{a:<6} r0 {v[0]:6.2f}  peers {pm:6.2f}  "
              f"r0-peers {v[0]-pm:+6.2f}   r0 is #{rank} of 8 shortest")
    print()

print("""HOW TO DECIDE.
  LIVENESS   rank 0's S5->S6 delta must be clearly positive and the peers'
             delta ~0.  If rank 0 did not move, STOP -- the probe is dead and
             no reading of the wall is valid.
  ABSORBED   rank 0 grows, peers' S3->S4 WAIT does NOT grow, wall ~0.
             -> the shared-expert hoist/shard family closes for good.
  ON PATH    rank 0 grows and the peers' S3->S4 wait grows by about as much,
             wall ~ +0.46 ms.  -> that delta is an UPPER bound on a K-shard's
             harvest (glm-additive-probes-overprice-deletions: added work costs
             ~1:1, removed work saves ~0), so it licenses building one, it does
             not price one.""")
