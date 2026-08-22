#!/usr/bin/env python3
"""Route 2 for the EP collective ceiling: does the freed inter-rank skew LEAVE
the layer, or does it reappear at the OTHER cross-rank rendezvous?

Companion to probe_ep_collective_ceiling.sh.  Reads the two Phase-B counter
logs, both taken with MPK_BAR_SKEW=3 so the instrument's cost is common-mode
(the method dddf699 used when the wall could not resolve the EP poll batch).

THE LIVE PREDICTION.  Rule 3 of e5d1ff5
(glm-deleting-a-whole-rendezvous-is-neutral) says a deleted wait's skew
relocates to a neighbouring rendezvous when the two share a PRODUCER SET.  The
EP collective (S0->S1) and the q_b head-shard gather (S19->S20) are the layer's
only two cross-rank rendezvous and their producer set is identical: the 8
ranks.  This is the textbook configuration for rule 3.  But e02d5d5 showed rule
3 failing for exactly this class -- deleting the q_b gather moved its absorber
S22->S28 by -0.04 us/layer and ~92% of the freed time survived to the wall.  So
S19->S20 below is not a formality:

    rule 3 right   -> S19->S20 grows by roughly what S0->S1 loses, wall ~0
    e02d5d5 right  -> S19->S20 flat, the loss survives to S0->S14 and the wall

SPAN HYGIENE, two separate traps.

1. THE INSTRUMENT IS NOT SYMMETRIC ACROSS THE ARMS.  mpk_stage_stamp(4) sits
   inside `#if MPK_EP_ABLATE == 0` (gang_mla_full_layer_fused_mi300.cuh:1213),
   so slot 4 is absent in the ablated arm.  S3->S4, the peer wait's own span,
   CANNOT be differenced and is reported base-only, for reference.  The
   bracket S0->S1 is the deleted region's readable span: both stamps are
   unguarded and written by tid==0 of all 232 blocks (:748, :1330).

2. THE PARTICIPATION TRAP, from cecadcd.  A stamp is only a synchronization
   point for workers that PARTICIPATE in the phase it ends.  In the attention
   tail the machine splits three ways (64 decode / 64 merge / 104 neither) and
   the 104 cross S21/S23/S24 in program order ~20 us before the decode set
   does.  So every attention-tail span below is taken over a SINGLE set (D,
   from the log's own slot-20 membership).  S21 and S23 are deliberately
   absent.  Slots 3/4 have 8 writers even in the base arm.
"""
import re
import collections
import sys

N_LAYERS = 76
ARMS = ("base", "epablate")
LOGS = {a: f"/tmp/epabl_ctr_{a}.log" for a in ARMS}


def load(path):
    pat = re.compile(r"\[1,(\d+)\].*BARSTAGEWS (\d+) (\d+) (\d+) (\d+)")
    d = collections.defaultdict(lambda: collections.defaultdict(dict))
    for ln in open(path, errors="ignore"):
        m = pat.search(ln)
        if m:
            r, s, w, c, t = (int(x) for x in m.groups())
            if c:
                d[r][s][w] = t / c / 1000.0
    return d


def med(vals):
    v = sorted(vals)
    return v[len(v) // 2]


def mk(vals):
    return max(vals)


R = {}
for a in ARMS:
    try:
        R[a] = load(LOGS[a])
    except FileNotFoundError:
        sys.exit(f"missing {LOGS[a]} -- run probe_ep_collective_ceiling.sh first")
    if not R[a]:
        sys.exit(f"{LOGS[a]} has no BARSTAGEWS rows")

# Slot population per arm -- the asymmetry check. If slot 4 is present in the
# ablated arm the -D never reached the build and nothing below means anything.
print("=" * 88)
print("SLOT POPULATIONS (rank 0) -- slot 4 MUST be absent in the ablated arm")
print("=" * 88)
for a in ARMS:
    rows = R[a].get(0, {})
    for s in (0, 1, 2, 3, 4, 16, 19, 20, 22, 28, 14):
        n = len(rows.get(s, {}))
        print(f"  {a:<10} slot {s:<3} writers={n}")
    print()

# Spans: (from, to, restrict-to-decode-set, label)
SPANS = [
    (0, 1, False, "S0->S1    THE EP COLLECTIVE (the deleted region)"),
    (0, 2, False, "S0->S2    EP + release + dispatch, end of prologue"),
    (1, 16, False, "S1->S16   EP release -> attention entry"),
    (19, 20, True, "S19->S20  THE OTHER CROSS-RANK RENDEZVOUS (absorber)"),
    (22, 28, True, "S22->S28  merge -> MoE entry (absorber #2)"),
    (28, 5, False, "S28->S5   MoE half up to routing (confounded)"),
    (0, 14, False, "S0->S14   LAYER SPAN"),
]


def span(rows, a, b, only_d, stat):
    if a not in rows or b not in rows:
        return None
    sel = set(rows[20]) if only_d and 20 in rows else None
    va = [v for w, v in rows[a].items() if sel is None or w in sel]
    vb = [v for w, v in rows[b].items() if sel is None or w in sel]
    if not va or not vb:
        return None
    return stat(vb) - stat(va)


for stat, sname in ((mk, "MAKESPAN (crit)"), (med, "MEDIAN (typ)")):
    print("=" * 88)
    print(f"{sname}, us/layer, pooled over the 8 ranks")
    print("=" * 88)
    print(f"{'span':<52}{'base':>9}{'abl':>9}{'delta':>9}{'ms/tok':>9}")
    for a, b, only_d, lab in SPANS:
        per = []
        for rk in range(8):
            if rk not in R["base"] or rk not in R["epablate"]:
                continue
            x = span(R["base"][rk], a, b, only_d, stat)
            y = span(R["epablate"][rk], a, b, only_d, stat)
            if x is not None and y is not None:
                per.append((x, y))
        if not per:
            print(f"{lab:<52}{'--- not in both logs ---':>36}")
            continue
        bx = sum(p[0] for p in per) / len(per)
        by = sum(p[1] for p in per) / len(per)
        print(f"{lab:<52}{bx:9.2f}{by:9.2f}{by-bx:+9.2f}"
              f"{(by-bx)*N_LAYERS/1e3:+9.3f}")
    print()

# S3->S4 is base-only: slot 4 is compiled out under the ablation.
print("=" * 88)
print("S3->S4 THE PEER WAIT ITSELF -- BASE ARM ONLY (slot 4 absent under =1)")
print("=" * 88)
for stat, sname in ((mk, "makespan"), (med, "median")):
    per = []
    for rk in range(8):
        if rk in R["base"]:
            v = span(R["base"][rk], 3, 4, False, stat)
            if v is not None:
                per.append(v)
    if per:
        print(f"  {sname:<10} {sum(per)/len(per):6.2f} us/layer "
              f"({sum(per)/len(per)*N_LAYERS/1e3:.3f} ms/token)  over 8 writers/rank")

print("""
HOW TO READ IT.
  S0->S1 is the sanity check: if deleting the peer store and the peer wait does
  not shrink the EP bracket, the -D never reached the build.
  S19->S20 is THE result. Same producer set as the deleted rendezvous, so rule
  3 predicts it absorbs the freed skew. If it stays flat, the cross-rank
  exception found in e02d5d5 replicates on a second, 3.5x larger case, and
  cross-rank rendezvous become a live lever class rather than a closed one.
  S0->S14 is the corroborating route for the wall.
  S28->S5 is CONFOUNDED and reported only for completeness: the probe emits
  wrong output upstream of the router, which changes TopK and therefore EP
  expert balance (glm-wrong-output-probes-upstream-of-router-are-invalid).
  The WALL from Phase A decides. These counters only say whether the wall moved
  for the reason claimed.""")
