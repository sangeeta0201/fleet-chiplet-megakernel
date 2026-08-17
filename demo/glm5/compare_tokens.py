#!/usr/bin/env python3
"""Compare GLM generated token streams between two runs of the correctness sweep.

Usage:
    python3 compare_tokens.py /tmp/glm5_correctness torch mp8ep

Two gates, and the multi-rank one is the sharper of the two
------------------------------------------------------------
1. **Cross-rank agreement (multi-rank candidates only).** GLM runs data-parallel
   attention: every rank decodes the same prompt, holds the whole latent cache,
   and runs identical attention. Expert parallelism only changes *where* each
   expert's weights live, not what the MoE sums to, and each rank folds its
   peers' partials in the same fixed gather order. So all N ranks must emit the
   same tokens, bit-for-bit. They are not independent samples -- they are N
   copies of one computation, and a fold that drops or misorders a peer
   corrupts exactly the rank that dropped it. This is the gate that actually
   tests the collective, and it needs no reference run at all.

2. **A content keyword per prompt**, against the reference. This is inherited
   from demo/gpt_oss/compare_tokens.py, along with the reason exact match is
   not the criterion: a different reduction order changes low-order bf16 bits,
   and under greedy decode one near-tie logit flips, after which the two
   sequences are legitimately different text and both correct. Distinct-token
   ratio was measured and rejected there -- it ranked word-salad ABOVE good
   text. See that file's docstring for the numbers.

Prefix agreement and repetition are printed as diagnostics only.
"""
import glob
import json
import os
import re
import sys

# Substrings any correct generation must contain (case-insensitive), index-
# aligned with PROMPTS in run_correctness_suite.sh. If the *reference* also
# misses one, the script says so: that is a bad entry here, not a bad candidate.
EXPECTED = {
    0: ["paris"],
    1: ["scatter"],
    2: ["prime"],
    3: ["stack", "queue"],
}


def load_group(d, tag, i):
    """All dumps for one prompt, as {rank_or_None: entry}, lowest rank first."""
    out = {}
    plain = os.path.join(d, f"{tag}_p{i}.json")
    if os.path.exists(plain):
        with open(plain) as f:
            out[None] = json.load(f)
    for p in sorted(glob.glob(os.path.join(d, f"{tag}_p{i}_rank*.json"))):
        m = re.search(r"_rank(\d+)\.json$", p)
        with open(p) as f:
            out[int(m.group(1))] = json.load(f)
    return dict(sorted(out.items(), key=lambda kv: (kv[0] is not None, kv[0])))


def repetition(ids):
    """(distinct ratio, longest run of an immediately repeating bigram)."""
    if not ids:
        return 0.0, 0
    longest = run = 0
    for k in range(2, len(ids)):
        if ids[k] == ids[k - 2]:
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return len(set(ids)) / len(ids), longest


def agreement(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def missing_keywords(entry, i):
    text = entry.get("text", "").lower()
    return [k for k in EXPECTED.get(i, []) if k not in text]


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    d, ref_tag, cand_tag = sys.argv[1:4]

    i = 0
    n_cmp = n_pass = 0
    while True:
        refs = load_group(d, ref_tag, i)
        cands = load_group(d, cand_tag, i)
        if not refs and not cands:
            break
        if not refs or not cands:
            print(f"prompt {i}: MISSING "
                  f"({ref_tag}={len(refs)} files, {cand_tag}={len(cands)})")
            i += 1
            continue
        n_cmp += 1

        # Reference: take the lowest rank present. Under DP they are copies.
        ref = next(iter(refs.values()))
        ref_ids = ref["token_ids"]

        # Gate 1 -- cross-rank agreement among the candidate's ranks.
        ranks = [r for r in cands if r is not None]
        rank_ok = True
        rank_note = ""
        if len(ranks) > 1:
            base = cands[ranks[0]]["token_ids"]
            bad = []
            for r in ranks[1:]:
                other = cands[r]["token_ids"]
                if other != base:
                    bad.append((r, agreement(base, other)))
            rank_ok = not bad
            rank_note = ("all %d ranks identical" % len(ranks) if rank_ok else
                         "ranks diverge from r%d: " % ranks[0] +
                         ", ".join(f"r{r}@{a}" for r, a in bad))

        # Gate 2 -- content keyword, checked on every candidate rank.
        kw_bad = {r: missing_keywords(e, i)
                  for r, e in cands.items() if missing_keywords(e, i)}
        ok = rank_ok and not kw_bad
        n_pass += ok

        cand = next(iter(cands.values()))
        cand_ids = cand["token_ids"]
        agree = agreement(ref_ids, cand_ids)
        exact = cand_ids == ref_ids
        ra, la = repetition(ref_ids)
        rb, lb = repetition(cand_ids)

        print(f"prompt {i}: {'PASS' if ok else 'FAIL'} "
              f"agree@{agree}/{min(len(ref_ids), len(cand_ids))}"
              f"{' (exact)' if exact else ''}")
        if rank_note:
            print(f"    cross-rank: {'OK' if rank_ok else 'FAIL'} -- "
                  f"{rank_note}")
        for r, miss in kw_bad.items():
            print(f"    rank {r} missing expected: {miss}")
        ref_miss = missing_keywords(ref, i)
        if ref_miss:
            print(f"    WARNING: reference {ref_tag} also missing {ref_miss} "
                  f"-- fix EXPECTED[{i}], not the candidate")
        print(f"    {ref_tag:>8}: distinct={ra:.2f} max_bigram_run={la} "
              f"n={len(ref_ids)}")
        print(f"    {cand_tag:>8}: distinct={rb:.2f} max_bigram_run={lb} "
              f"n={len(cand_ids)}")
        i += 1

    print(f"\n{n_pass}/{n_cmp} prompts pass")
    print("Exact match vs the reference is not required -- see the docstring; "
          "cross-rank agreement is.")
    return 0 if n_cmp and n_pass == n_cmp else 1


if __name__ == "__main__":
    sys.exit(main())
