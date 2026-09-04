#!/usr/bin/env python3
"""Tabulate a GLM-5 perplexity-vs-sequence-length sweep produced by
demo/glm5/run_ppl_sweep.sh. Writes summary.md and summary.csv next to the dumps.

Usage: summarize_glm5_ppl_sweep.py <sweep-dir>

Ported from summarize_ppl_sweep.py (the GPT-OSS instrument). Reads
<len>/mpk_ppl_rank*.json.

THE POINT OF THIS SCRIPT IS THE FIXED-PREFIX COLUMN. Each length scores a longer
prefix of the same corpus, so the full-slice perplexity moves with what text got
included, not just with context length. On GPT-OSS that effect was large enough
to look like a kernel regression -- full-slice perplexity climbed 36.0 -> 139.1
from 512 to 1024 tokens on the *Torch reference*, which no kernel change can
explain; WikiText-2 past position ~512 is a dense run of proper nouns averaging
6.27 nats against 3.58 for the first 511 tokens. Restricting every run to the
positions all runs share removes it.

GLM-5 needs this at least as badly, because it has no reference arm to expose
the confound: measured 5.70 at 128 tokens and 15.64 at 256 with no way, from the
plain column alone, to say whether the model got worse or the text did.

Entropy is reported alongside perplexity on purpose: numeric noise flattens the
softmax, which *lowers* NLL at positions the model gets wrong, so perplexity
alone can make a noisier run look like a better one.
"""
import csv
import glob
import json
import math
import os
import re
import sys


def _load_length(d, n):
    """Aggregate the per-rank dumps for one sweep length."""
    files = sorted(glob.glob(os.path.join(d, str(n), "mpk_ppl_rank*.json")))
    if not files:
        return None
    per_rank = []
    for f in files:
        with open(f) as fh:
            per_rank.append(json.load(fh))
    ref = per_rank[0]
    top1, tgts = ref.get("top1") or [], ref.get("targets") or []
    acc = (sum(1 for a, b in zip(top1, tgts) if int(a) == int(b))
           / len(top1)) if top1 else None
    ppls = [r["perplexity"] for r in per_rank]
    return {
        "ranks": len(per_rank),
        "ppl": ref.get("perplexity"),
        "nll": ref.get("mean_nll"),
        "scored": ref.get("scored_positions"),
        "ent": ref.get("mean_entropy"),
        "acc": acc,
        # Cross-rank spread is the EP correctness signal, so carry it into the
        # table rather than leaving it in the pytest log.
        "rank_spread": (max(ppls) - min(ppls)) / min(ppls) if ppls else None,
        "per_pos": ref.get("per_position_nll") or [],
        "prefix": (ref.get("diagnostics") or {}).get("prefix"),
        "widest_zero_run": (ref.get("diagnostics") or {}).get(
            "widest_zero_run"),
        "dup_rows": (ref.get("diagnostics") or {}).get(
            "duplicate_adjacent_rows"),
    }


def main(d):
    lengths = []
    for entry in sorted(os.listdir(d)):
        if re.fullmatch(r"\d+", entry) and os.path.isdir(
                os.path.join(d, entry)):
            lengths.append(int(entry))
    rows = {}
    for n in sorted(lengths):
        got = _load_length(d, n)
        if got:
            rows[n] = got
    if not rows:
        print(f"No <len>/mpk_ppl_rank*.json dumps found under {d}")
        return 1

    common = min((len(v["per_pos"]) for v in rows.values() if v["per_pos"]),
                 default=0)
    for v in rows.values():
        pp = v["per_pos"][:common]
        v["ppl_common"] = math.exp(sum(pp) / len(pp)) if pp else None

    def f(x, spec="9.4f"):
        return format(x, spec) if isinstance(x, (int, float)) else "      n/a"

    out = [
        f"| seq len | scored | ppl | ppl@{common} | entropy | acc | "
        f"rank spread | widest zero run | dup rows |",
        "|--------:|-------:|----:|-------------:|--------:|----:|"
        "------------:|----------------:|---------:|",
    ]
    csv_rows = []
    for n in sorted(rows):
        v = rows[n]
        out.append(
            f"| {n:7d} | {v['scored'] or 0:6d} | {f(v['ppl'])} "
            f"| {f(v['ppl_common'])} | {f(v['ent'], '7.4f')} "
            f"| {f(v['acc'], '6.4f')} | {f(v['rank_spread'], '11.6f')} "
            f"| {v['widest_zero_run'] if v['widest_zero_run'] is not None else 'n/a':>15} "
            f"| {v['dup_rows'] if v['dup_rows'] is not None else 'n/a':>8} |"
        )
        csv_rows.append({
            "seq_len": n, "scored": v["scored"], "ppl": v["ppl"],
            "ppl_common_prefix": v["ppl_common"], "entropy": v["ent"],
            "acc": v["acc"], "rank_spread": v["rank_spread"],
            "prefix": v["prefix"],
            "widest_zero_run": v["widest_zero_run"],
            "duplicate_adjacent_rows": v["dup_rows"],
        })

    table = "\n".join(out)
    print("\nGLM-5 perplexity vs sequence length (WikiText-2 prefix)\n")
    print(table)
    print(f"\nppl@{common} is the SAME first {common} scored positions in "
          f"every run -- read that column for\nthe effect of context length. "
          f"The plain ppl column also moves with which text each\nslice "
          f"includes, so the two columns diverging means the corpus got "
          f"harder, not the\nkernel worse.")
    print("acc = argmax vs corpus targets. rank spread = relative spread of "
          "perplexity across\nEP ranks, which must stay ~0. widest zero run "
          "flags a skipped logit column range.")

    prefixes = {v["prefix"] for v in rows.values()}
    if len(prefixes) > 1:
        print(f"\nWARNING: dumps used different prompt prefixes {prefixes} -- "
              f"these numbers are not comparable.")
    if prefixes and not all(prefixes):
        print("\nWARNING: at least one dump carries NO prompt prefix. GLM-5 "
              "expects '[gMASK]<sop>'; without it the measurement is "
              "out of distribution (268 against 5.70 measured).")

    with open(os.path.join(d, "summary.md"), "w") as fh:
        fh.write("# GLM-5 744B perplexity vs sequence length\n\n"
                 "Corpus: WikiText-2 raw test, first N tokens, prefixed with "
                 "`[gMASK]<sop>`. Prefill-only\n(teacher forced), "
                 "`GLM_LMHEAD_TP=0`.\n\n" + table + "\n")
    with open(os.path.join(d, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nWrote {d}/summary.md and {d}/summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                  else "outputs/glm5/ppl_sweep"))
