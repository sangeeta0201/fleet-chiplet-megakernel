#!/usr/bin/env python3
"""WHERE inside the MoE half does rank 0's +10.17 us/layer actually sit?

Offline. Reads the same two Phase-B counter logs as ep_rank_skew_anatomy.py.
No GPU run.

WHY.  ep_rank_skew_anatomy.py showed rank 0 does +10.17 us/layer more WORK than
the peer mean across S20->S14, and waits 10.39 us less at the rendezvous -- the
two closing to 0.21 us.  It then attributed that excess to the SHARED EXPERT,
on two grounds: rank 0 is the shared-expert rank, and
glm-shared-expert-is-the-ep-straggler independently measured 9.09 us/layer.

That is an INFERENCE, not a localization.  S20->S14 is ~102 us wide and
contains decode, merge, o_proj, the router, W13, W2 and the layer boundary.
Any of them could carry the excess.  This script splits the span by stamp and
asks which sub-region is rank-0-heavy.

IT ALSO SETTLES A CONTRADICTION.  glm-shared-expert-hoist-is-zero-makespan
found rank 0 has the SHORTEST W13 makespan of all 8 ranks -- which is the
opposite of what a +10 us shared-expert straggler in W13 predicts.  If the
excess is NOT in S5->S6, both readings can be true at once and the hoist
closure survives.  If it IS in S5->S6, one of them is wrong.

READ THIS BEFORE BELIEVING A ROW.  Slot worker sets differ (20 has the
64-worker decode set, 22 has 128, 29-31 have 192, 3/4 have the 8 XCD leaders,
the rest 232).  Every span below is a median over the workers crossing BOTH of
its stamps.  A region whose two slots have different populations is marked '*'
and is a floor, not a credit -- see board_budget.py.
"""
import re
import collections
import sys

ARMS = ("base", "epablate")
LOGS = {a: f"/tmp/epabl_ctr_{a}.log" for a in ARMS}
PAT = re.compile(r"\[1,(\d+)\].*BARSTAGEWS (\d+) (\d+) (\d+) (\d+)")
N_LAYERS = 76

# The MoE half, split at every stamp, in program order. Labels from
# board_budget.py, which derived them from the mpk_stage_stamp( call sites.
REGIONS = [
    (20, 21, "MLA decode"),
    (21, 23, "decode->merge bar + merge"),
    (23, 24, "Phase 8 attn -> o_proj bar"),
    (24, 25, "o_proj weight prefetch issue"),
    (25, 26, "per-XCD attention release"),
    (26, 27, "prefetch DMA retired"),
    (27, 28, "entering the MoE half"),
    (28, 29, "W_UK / W_UV + Mechanism-C bar"),
    (29, 30, "o_proj GEMV + residual"),
    (30, 31, "o_proj -> router barrier"),
    (31, 32, "router GEMV + TopK"),
    (32, 5, "routing-ready poll"),
    (5, 6, "W13 tiles          <- SHARED EXPERT"),
    (6, 7, "W13 -> W2 barrier"),
    (7, 8, "W2 tiles"),
    (8, 12, "MoE exit"),
    (12, 13, ""),
    (13, 14, "layer boundary"),
]


def load(path):
    d = collections.defaultdict(lambda: collections.defaultdict(dict))
    for ln in open(path, errors="ignore"):
        m = PAT.search(ln)
        if m:
            r, s, w, c, t = (int(x) for x in m.groups())
            if c:
                d[r][s][w] = t / c / 1000.0
    return d


def med(v):
    v = sorted(v)
    return v[len(v) // 2] if v else None


def span(rows, x, y):
    if x not in rows or y not in rows:
        return None
    w = set(rows[x]) & set(rows[y])
    if not w:
        return None
    return med([rows[y][i] for i in w]) - med([rows[x][i] for i in w])


R = {}
for a in ARMS:
    try:
        R[a] = load(LOGS[a])
    except FileNotFoundError:
        sys.exit(f"missing {LOGS[a]} -- run probe_ep_collective_ceiling.sh first")

for a in ARMS:
    print("=" * 104)
    print(f"{a}: MoE half split by stamp, us/layer. 'r0-peers' is the bias we are hunting.")
    print("=" * 104)
    print(f"{'region':<44}{'r0':>8}{'peermn':>8}{'r0-peers':>10}"
          f"{'peerspread':>12}{'ms/tok':>9}")
    tot_r0 = tot_pm = 0.0
    rows_out = []
    for x, y, lab in REGIONS:
        v = [span(R[a].get(r, {}), x, y) for r in range(8)]
        if any(u is None for u in v):
            continue
        pm = sum(v[1:]) / 7
        d = v[0] - pm
        sub = "*" if set(R[a][0].get(x, {})) != set(R[a][0].get(y, {})) else " "
        tot_r0 += v[0]
        tot_pm += pm
        rows_out.append((f"S{x}->S{y}{sub}{lab}", v[0], pm, d,
                         max(v[1:]) - min(v[1:])))
    for n, r0, pm, d, sp in rows_out:
        print(f"{n:<44}{r0:8.2f}{pm:8.2f}{d:+10.2f}{sp:12.2f}"
              f"{d * N_LAYERS / 1e3:+9.3f}")
    print(f"{'SUM of the above':<44}{tot_r0:8.2f}{tot_pm:8.2f}"
          f"{tot_r0 - tot_pm:+10.2f}{'':12}{(tot_r0-tot_pm)*N_LAYERS/1e3:+9.3f}")
    w = span(R[a].get(0, {}), 20, 14)
    pmw = sum(span(R[a].get(r, {}), 20, 14) for r in range(1, 8)) / 7
    print(f"{'S20->S14 measured directly':<44}{w:8.2f}{pmw:8.2f}"
          f"{w - pmw:+10.2f}")
    print()

# ------------------------------------------------------------------ the clash
# glm-shared-expert-hoist-is-zero-makespan measured the SAME slots on
# 2026-08-21 and concluded rank 0's W13 is the SHORTEST of 8.  It used
# max_w(S6) - max_w(S5); this file uses med_w(S6) - med_w(S5).  Two order
# statistics, opposite verdicts.  Settle it on ONE set of logs.
print("=" * 104)
print("THE CLASH: median vs makespan, base arm, same slots, same log")
print("=" * 104)


def stat(rows, x, y, f):
    if x not in rows or y not in rows:
        return None
    w = set(rows[x]) & set(rows[y])
    return f([rows[y][i] for i in w]) - f([rows[x][i] for i in w])


for x, y, lab in ((5, 6, "W13  S5->S6"), (7, 8, "W2   S7->S8"),
                  (5, 8, "MoE  S5->S8")):
    for nm, f in (("median", med), ("makespan(max-max)", max)):
        v = [stat(R["base"].get(r, {}), x, y, f) for r in range(8)]
        if any(u is None for u in v):
            continue
        pm = sum(v[1:]) / 7
        rank = sorted(range(8), key=lambda r: v[r]).index(0) + 1
        print(f"  {lab:<12} {nm:<18} r0 {v[0]:6.2f}  peers {pm:6.2f}  "
              f"r0-peers {v[0]-pm:+6.2f}   r0 is #{rank} of 8 shortest")
    print()
print("""  If the two statistics disagree on the SAME log, the disagreement is a
  property of the worker distribution, not of the run -- and the question
  becomes which statistic gates the cross-rank rendezvous.
""")

print("""HOW TO READ IT.  The bias is localized where 'r0-peers' is large and
POSITIVE.  A region where it is near zero is rank-symmetric and cannot be the
source, no matter how large the region is.  'peerspread' is the spread among
the SEVEN peers only -- if that is comparable to r0-peers, rank 0 is not
special and the excess is ordinary rank noise, not a shared-expert bias.
'*' = the two stamps have different worker sets; that region is a floor.""")
