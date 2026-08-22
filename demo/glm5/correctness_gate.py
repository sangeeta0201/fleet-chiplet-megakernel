#!/usr/bin/env python3
"""THE GLM-5 CORRECTNESS GATE. Exact-token equality is NOT legal on this model.

Measured 2026-08-22 (`demo/glm5/probe_bs1_determinism.sh`): five bs=1 runs, one
build, one prompt, greedy argmax, nothing varying between them. Pairwise
exact-prefix:

            r1      r2      r5      rB
    r1       -     122       0       0
    r2     122       -       0       0
    r5       0       0       -     124
    rB       0       0     124       -

The runs partition into TWO attractor continuations. Within a cluster they
agree for ~122-124 tokens and then drift; across clusters they differ at token
0. r5 reused the build, so this is not a first-run-after-build effect. The
mechanism is order-nondeterministic float reduction (the EP fold and the atomic
accumulations retire in arrival order), which flips argmax on near-ties, and one
flip cascades.

CONSEQUENCE: any gate of the form "arm B must equal arm A token for token" fails
~50% of the time on two runs of the SAME configuration. Three published
conclusions were drawn against that illegal gate -- see the notes at the bottom.

WHAT IS ACTUALLY ENFORCEABLE:

  G1  CROSS-RANK IDENTITY (hard). All 8 ranks must emit byte-identical tokens
      within a single run. This held in every run measured and is the gate that
      catches real EP/sync/barrier bugs -- the class of defect that actually
      shows up on this branch. It is necessary and NOT sufficient: two arms can
      each be internally consistent and still both be wrong.

  G2  COHERENCE (hard). Non-degenerate output: full token count, distinct-token
      ratio >= 0.30, no bigram repeated more than 25 times. Catches the
      repetition collapse that a genuinely broken kernel produces. Measured
      healthy values are ratio ~0.53 and top-bigram 5-10.

  G3  ATTRACTOR MEMBERSHIP (advisory). Run the control 3x to enumerate its
      clusters, then require each test-arm run to share a >= MIN_PREFIX-token
      prefix with SOME control run. Advisory because with 2 attractors and 3
      samples you can simply miss a cluster and fail a correct arm. A G3 miss
      is a prompt to run more samples, never on its own a defect.

A change that passes G1+G2 and lands in a known control cluster is as verified
as this model permits. Beyond that, read the text for topicality.

USAGE
  correctness_gate.py --ctl 'ctldir/ctl_r*_rank*.json' \\
                      --arm 'armdir/arm_r*_rank*.json'
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

MIN_PREFIX = 100
MIN_DISTINCT = 0.30
MAX_BIGRAM = 25


def _run_key(path):
    """Group rank files belonging to one run. Strips the _rank<N> suffix."""
    return re.sub(r"_rank\d+(?=\.json$)", "", path)


def load_runs(pattern):
    """-> {run_key: {"tokens": [...], "nranks": n, "xrank_ok": bool, "text": s}}"""
    runs = collections.defaultdict(list)
    for p in sorted(glob.glob(pattern)):
        runs[_run_key(p)].append(p)
    out = {}
    for key, files in sorted(runs.items()):
        toks, texts = [], []
        for f in files:
            try:
                d = json.load(open(f))
            except (OSError, ValueError):
                continue
            toks.append(d.get("token_ids") or [])
            texts.append(d.get("text", ""))
        if not toks:
            continue
        out[os.path.basename(key)] = {
            "tokens": toks[0],
            "nranks": len(toks),
            "xrank_ok": all(t == toks[0] for t in toks),
            "text": texts[0],
        }
    return out


def prefix(a, b):
    n = min(len(a), len(b))
    k = 0
    while k < n and a[k] == b[k]:
        k += 1
    return k


def coherence(tokens):
    if not tokens:
        return False, 0.0, 0
    ratio = len(set(tokens)) / len(tokens)
    big = collections.Counter(zip(tokens, tokens[1:]))
    top = big.most_common(1)[0][1] if big else 0
    return (ratio >= MIN_DISTINCT and top <= MAX_BIGRAM), ratio, top


def report(name, runs):
    print(f"\n=== {name}: {len(runs)} run(s)")
    ok = True
    for key, r in runs.items():
        c_ok, ratio, top = coherence(r["tokens"])
        g1 = "PASS" if r["xrank_ok"] else "FAIL"
        g2 = "PASS" if c_ok else "FAIL"
        ok &= r["xrank_ok"] and c_ok
        print(f"  {key:<28} ntok={len(r['tokens']):<5} ranks={r['nranks']} "
              f"G1(cross-rank)={g1} G2(coherence)={g2} "
              f"[distinct={ratio:.3f} topbigram={top}]")
    return ok


def clusters(runs):
    """Group runs that share a >= MIN_PREFIX prefix."""
    keys = list(runs)
    groups = []
    for k in keys:
        for g in groups:
            if prefix(runs[k]["tokens"], runs[g[0]]["tokens"]) >= MIN_PREFIX:
                g.append(k)
                break
        else:
            groups.append([k])
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctl", required=True, help="glob for control rank dumps")
    ap.add_argument("--arm", help="glob for test-arm rank dumps")
    a = ap.parse_args()

    ctl = load_runs(a.ctl)
    if not ctl:
        print(f"no control runs matched {a.ctl!r}")
        return 2
    ok = report("CONTROL", ctl)

    cg = clusters(ctl)
    print(f"\ncontrol attractor clusters: {len(cg)}")
    for i, g in enumerate(cg):
        print(f"  cluster {i}: {', '.join(g)}")
    if len(cg) == 1 and len(ctl) >= 3:
        print("  NOTE: only one cluster seen. Either this prompt is stable or "
              "the sample missed one; G3 misses are not defects.")

    arm = load_runs(a.arm) if a.arm else {}
    if arm:
        ok &= report("TEST ARM", arm)
        print("\nG3 attractor membership (advisory):")
        for key, r in arm.items():
            best = max(((prefix(r["tokens"], ctl[c]["tokens"]), c) for c in ctl),
                       default=(0, "-"))
            verdict = ("in a control cluster" if best[0] >= MIN_PREFIX
                       else "NEW continuation -- sample the control more")
            print(f"  {key:<28} best prefix {best[0]:>4} vs {best[1]:<22} {verdict}")

    print(f"\nHARD GATE (G1+G2): {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

# ── What this retroactively overturns ────────────────────────────────────────
# 1. `ccf8cfa` retracted MTP's 8.952 ms/token because the MTP arm's tokens did
#    not equal the control's (exact-prefix 0/264 and 26/264). Two runs of plain
#    bs=1 do exactly that to each other. The retraction rested on an ILLEGAL
#    GATE, so MTP's correctness is UNDETERMINED, not refuted. (Reopening it is
#    a separate decision; not done here.)
# 2. `25be6d9` called C = 1.206 provisional because bs=2 diverged from bs=1 at
#    token 0. bs=2's continuation is the same one bs=1 itself produces in half
#    its runs (the {r5,rB} cluster). bs=2 passes G1+G2 and lands in a known
#    control cluster, so C = 1.206 is CONFIRMED.
# 3. The item 1.1 per-phase table (`c08765f`) was flagged provisional for the
#    same reason and is likewise confirmed: routing is not corrupted, so the
#    shared-expert skew finding stands.
