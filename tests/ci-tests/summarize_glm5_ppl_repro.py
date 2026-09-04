#!/usr/bin/env python3
"""Summarize a GLM-5 perplexity reproducibility probe (demo/glm5/run_ppl_repro.sh).

Usage: summarize_glm5_ppl_repro.py <repro-dir>

Reports, over N identical reps of ONE build:

  * the perplexity spread, which is what the gate's tolerance must be set from;
  * the first scored position at which the reps disagree, which says whether
    nondeterminism enters immediately or accumulates;
  * how many positions ever disagree, and how many flip their argmax.

The last two are the useful pair. Perplexity is a mean, so it hides whether a
4% spread is every position wobbling slightly or a handful of positions
changing a lot -- and those imply different things about whether a tight gate
is safe. Argmax flips are the ones that would break a token-equality gate.

Exit status is 0 always: this is a measurement, not a gate. The gate is
tests/ci-tests/test_glm5_perplexity.py, which is what should carry the
tolerance this script justifies.
"""
import glob
import json
import os
import statistics as st
import sys


def _load(d):
    reps = {}
    for path in sorted(glob.glob(os.path.join(d, "r*", "mpk_ppl_rank0.json"))):
        name = os.path.basename(os.path.dirname(path))[1:]
        with open(path) as fh:
            reps[name] = json.load(fh)
    return reps


def main(d):
    reps = _load(d)
    if not reps:
        print(f"No r*/mpk_ppl_rank0.json dumps found under {d}")
        return 0

    # The build rep is reported but excluded from the spread: on the token
    # probe it was the rep that diverged, so folding it in would conflate
    # "first run after a build" with run-to-run variance.
    build = reps.pop("B", None)
    names = sorted(reps, key=lambda x: (len(x), x))

    print(f"\nGLM-5 perplexity reproducibility, {len(names)} identical reps "
          f"of one build\n")
    if build:
        print(f"  build rep B : ppl={build['perplexity']:.4f} "
              f"entropy={build.get('mean_entropy')}")
    for n in names:
        r = reps[n]
        print(f"  rep {n:<7}: ppl={r['perplexity']:.4f} "
              f"entropy={r.get('mean_entropy')}")

    if not names:
        return 0
    ppls = [reps[n]["perplexity"] for n in names]
    lo, hi = min(ppls), max(ppls)
    spread = (hi - lo) / lo if lo else float("nan")
    print(f"\n  n={len(ppls)} min={lo:.4f} max={hi:.4f} "
          f"mean={st.mean(ppls):.4f} "
          f"stdev={st.pstdev(ppls):.4f} spread={spread:.2%}")
    if build:
        inside = lo <= build["perplexity"] <= hi
        print(f"  build rep {'IS' if inside else 'is NOT'} inside the "
              f"spread of the reused-build reps"
              f"{'' if inside else ' -- first-run-after-build differs'}")

    # Per-position divergence. Identical inputs and a deterministic host-side
    # cross-entropy mean any difference here came from the kernel.
    ref = reps[names[0]]
    n_pos = min(len(reps[n]["per_position_nll"]) for n in names)
    first_diff = None
    n_diff = 0
    max_delta = 0.0
    for i in range(n_pos):
        vals = [reps[n]["per_position_nll"][i] for n in names]
        if max(vals) != min(vals):
            n_diff += 1
            max_delta = max(max_delta, max(vals) - min(vals))
            if first_diff is None:
                first_diff = i
    flips = 0
    for i in range(n_pos):
        if len({int(reps[n]["top1"][i]) for n in names}) > 1:
            flips += 1
    print(f"\n  per-position NLL: {n_diff}/{n_pos} positions differ across "
          f"reps, first at position "
          f"{first_diff if first_diff is not None else 'none'}, "
          f"largest single-position gap {max_delta:.4f} nats")
    print(f"  argmax: {flips}/{n_pos} positions flip their top-1 across reps "
          f"-- these are what a token-equality gate would trip on")
    if len(names) < 2:
        print("\n  VERDICT: none. One rep cannot disagree with itself -- the "
              "zeros above are an artifact of n=1, not reproducibility. "
              "Re-run with REPS>=2.")
    elif n_diff == 0:
        print("\n  VERDICT: bit-reproducible across these reps. A tight "
              "perplexity gate is safe, and token equality would be legal "
              "at this length.")
    else:
        print(f"\n  VERDICT: not reproducible. The gate tolerance must exceed "
              f"{spread:.2%}, and token equality is not a legal gate. "
              f"Divergence from position {first_diff} onward is consistent "
              f"with the EP fold and the MoE W2 f32 atomics retiring in "
              f"arrival order.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                  else "outputs/glm5/ppl_repro"))
