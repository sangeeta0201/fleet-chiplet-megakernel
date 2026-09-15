#!/usr/bin/env python3
"""Pair table for the o_proj K-split A/B.

  summarize_ksplit_ab.py /tmp/ksplit_ab

Reports BOTH decode statistics, because they answer different questions:

  decode_min  per-iteration minimum. Immune to the box's occasional 24-26 ms
              straggler iteration, which is what contaminated MPK_WUV_IN_MERGE's
              first reading. This is the statistic the 14.0 ms board number is.
  decode_avg  the mean over the 1024 decode iterations. Closer to what a user
              feels, but one outlier iteration moves it by ~0.02 ms/token.

Prints the generated-text head for every run. CLAUDE.md: never quote a latency
number without checking the text, and the flag-on arm here is WRONG OUTPUT by
construction, so its text is expected to be garbage -- that is the check that
the ablation actually did something, not a failure.
"""
import glob
import os
import re
import sys

RES = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ksplit_ab"

FIELDS = {
    "avg": re.compile(r"decode_avg_ms=([0-9.]+|None)"),
    "min": re.compile(r"decode_min_ms=([0-9.]+|None)"),
    "g1": re.compile(r"G1_cross_rank=(\w+)"),
    "dist": re.compile(r"G2_distinct=([0-9.]+)"),
    "osl": re.compile(r"dumped_osl=(\d+)"),
}
HEAD = re.compile(r"TEXT_HEAD: (.*)")


# run_latency_1k1k.sh's own decode_avg regex wants a space before "avg" and the
# demo prints "(avg 16.008ms/iter)", so its avg field is always None in this
# container. Read the Decode line out of the raw log instead of relying on it.
DECODE = re.compile(
    r"Decode:\s+(\d+) tokens in (\d+)/(\d+) iters .*?avg ([0-9.]+)ms/iter")
RANGE = re.compile(r"Decode per-iter range: min=([0-9.]+)ms max=([0-9.]+)ms")


def parse(path):
    out = {}
    txt = ""
    try:
        body = open(path, errors="replace").read()
    except OSError:
        return None
    for k, rx in FIELDS.items():
        m = rx.search(body)
        out[k] = m.group(1) if m else None
    m = HEAD.search(body)
    if m:
        txt = m.group(1)
    out["text"] = txt
    if "FAILED" in body:
        out["failed"] = True

    log = os.path.join(path[:-4], "isl1024_osl1024.log")
    if os.path.exists(log):
        lbody = open(log, errors="replace").read()
        m = DECODE.search(lbody)
        if m:
            out["avg"] = m.group(4)
            out["iters"] = f"{m.group(2)}/{m.group(3)}"
        m = RANGE.search(lbody)
        if m:
            out["min"] = out["min"] or m.group(1)
            out["max"] = m.group(2)
    return out


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    runs = {}
    for p in sorted(glob.glob(os.path.join(RES, "*_r*.out"))):
        base = os.path.basename(p)[:-4]
        arm, _, rep = base.rpartition("_r")
        r = parse(p)
        if r:
            runs[(arm, int(rep))] = r

    arms = sorted({a for a, _ in runs})
    reps = sorted({r for _, r in runs})
    if not arms:
        print(f"no runs under {RES}")
        return 1

    ctl = "control"
    var = [a for a in arms if a != ctl]
    var = var[0] if var else None

    for stat in ("min", "avg"):
        print()
        print(f"=== decode_{stat}_ms, alternating pairs ===")
        hdr = f"{'pair':>5s}" + "".join(f"{a:>12s}" for a in arms)
        if var:
            hdr += f"{'delta':>10s}"
        print(hdr)
        deltas = []
        for rep in reps:
            row = f"{rep:>5d}"
            vals = {}
            for a in arms:
                v = fnum(runs.get((a, rep), {}).get(stat))
                vals[a] = v
                row += f"{v:12.3f}" if v is not None else f"{'--':>12s}"
            if var and vals.get(ctl) is not None and vals.get(var) is not None:
                d = vals[var] - vals[ctl]
                deltas.append(d)
                row += f"{d:+10.3f}"
            print(row)
        if deltas:
            mean = sum(deltas) / len(deltas)
            print(f"{'mean':>5s}" + " " * (12 * len(arms)) + f"{mean:+10.3f}")
            print(f"  n={len(deltas)} pairs; per-pair deltas "
                  f"{' '.join(f'{d:+.3f}' for d in deltas)}")
            print("  wall noise floor is 0.26 ms -- do not claim anything "
                  "smaller than that.")

    print()
    print("=== per-run integrity (never quote latency without this) ===")
    for (a, r) in sorted(runs):
        d = runs[(a, r)]
        print(f"  {a:>8s} r{r}  osl={d['osl']}  G1={d['g1']}  "
              f"distinct={d['dist']}  iters={d.get('iters')}  "
              f"max_iter={d.get('max')}ms  "
              f"{'FAILED' if d.get('failed') else ''}")
        print(f"           text: {d['text'][:110]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
