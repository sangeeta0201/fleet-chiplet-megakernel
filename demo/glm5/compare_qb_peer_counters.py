#!/usr/bin/env python3
"""Route 2 for the q_b peer-wait ceiling: did the freed skew LEAVE, or did it
just move to S22 / S28?

Companion to probe_qb_peer_wait_ceiling.sh.  Reads the two Phase-B counter
logs, both taken with MPK_BAR_SKEW=3 so the instrument's cost is common-mode
(the method dddf699 used when the wall could not resolve the EP poll batch).

WHAT THIS IS TESTING.  Deleting the peer store+poll must drop S19->S20 by
roughly the 8.8 us/layer that 5cb68f6 attributed to it.  That is NOT the
interesting number -- of course a deleted wait leaves its own region.  The
question e5d1ff5 forces us to ask is whether the freed time survives to the
next genuine rendezvous, or whether S22 and S28 simply absorb it.  e5d1ff5
deleted a whole rendezvous carrying 8.28 us/layer of uniform spin and the wall
moved 0.003 ms for exactly this reason.

SPAN HYGIENE, from cecadcd.  A stamp is only a synchronization point for
workers that PARTICIPATE in the phase it ends.  In the attention tail the
machine splits three ways (64 decode / 64 merge / 104 neither) and the 104
cross S21/S23/S24 in program order ~20 us before the decode set does.  So
every span below is either taken over a SINGLE set (D, from the log's own
slot-20 membership) or between stamps where all sets agree to 0.03 us -- S18,
S22, S28, and the layer bracket S0/S14.  S21 and S23 are deliberately absent.
"""
import re
import collections
import sys

N_LAYERS = 76
ARMS = ("base", "skippeer")
LOGS = {a: f"/tmp/qbpeer_ctr_{a}.log" for a in ARMS}


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
        sys.exit(f"missing {LOGS[a]} -- run probe_qb_peer_wait_ceiling.sh first")
    if not R[a]:
        sys.exit(f"{LOGS[a]} has no BARSTAGEWS rows")

# Spans: (from, to, restrict-to-decode-set, label)
SPANS = [
    (19, 20, True, "S19->S20  THE PEER WAIT ITSELF"),
    (18, 22, True, "S18->S22  qkv_a bar -> dec/merge rendezvous"),
    (18, 28, True, "S18->S28  the whole attention tail"),
    (22, 28, True, "S22->S28  merge -> MoE entry (absorber #1)"),
    (28, 5, False, "S28->S5   MoE half up to routing (confounded)"),
    (0, 14, False, "S0->S14   LAYER SPAN"),
]


def span(rows, a, b, only_d, stat):
    if a not in rows or b not in rows:
        return None
    sel = set(rows[20]) if only_d else None
    va = [v for w, v in rows[a].items() if sel is None or w in sel]
    vb = [v for w, v in rows[b].items() if sel is None or w in sel]
    if not va or not vb:
        return None
    return stat(vb) - stat(va)


for stat, sname in ((mk, "MAKESPAN (crit)"), (med, "MEDIAN (typ)")):
    print("=" * 88)
    print(f"{sname}, us/layer, per rank then pooled")
    print("=" * 88)
    print(f"{'span':<46}{'base':>9}{'skip':>9}{'delta':>9}{'ms/tok':>9}")
    for a, b, only_d, lab in SPANS:
        per = []
        for rk in range(8):
            if rk not in R["base"] or rk not in R["skippeer"]:
                continue
            x = span(R["base"][rk], a, b, only_d, stat)
            y = span(R["skippeer"][rk], a, b, only_d, stat)
            if x is not None and y is not None:
                per.append((x, y))
        if not per:
            print(f"{lab:<46}{'--- not in log ---':>36}")
            continue
        bx = sum(p[0] for p in per) / len(per)
        by = sum(p[1] for p in per) / len(per)
        print(f"{lab:<46}{bx:9.2f}{by:9.2f}{by-bx:+9.2f}"
              f"{(by-bx)*N_LAYERS/1e3:+9.3f}")
    print()

print("""HOW TO READ IT.
  S19->S20 is the sanity check: if deleting the store+poll does not empty that
  region, the -D never reached the build and nothing below means anything.
  S18->S28 is the corroborating route for the wall: it is bracketed by two
  stamps all worker sets cross together, so it is a real machine-wide span.
  S22->S28 is the absorber to watch -- if the peer wait's time reappears here,
  the skew relocated and head-sharding buys nothing, which is exactly what
  happened to the deleted rendezvous in e5d1ff5.
  S28->S5 is CONFOUNDED and reported only for completeness: the probe emits
  wrong attention output, which changes TopK and therefore EP expert balance
  (glm-wrong-output-probes-upstream-of-router-are-invalid).  Do not read a
  lever out of it.
  The WALL from Phase A decides. These counters only say whether the wall
  moved for the reason claimed.""")
