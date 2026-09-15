#!/usr/bin/env python3
"""Pair table for a generic compile-time-flag A/B on GLM-5.2 decode.

  summarize_flag_ab.py /mnt/nvme1/glm5_flag_ab

Generalisation of summarize_ksplit_ab.py to more than two arms. Every non-
control arm is differenced against the control of the SAME rep, because the
arms alternate in time and the rep is the blocking factor.

Reports BOTH decode statistics, because they answer different questions:

  decode_min  per-iteration minimum. Immune to the box's occasional 24-26 ms
              straggler iteration. This is the statistic the 14.0 ms board
              number is.
  decode_avg  the mean over the 1024 decode iterations. Closer to what a user
              feels, but one outlier iteration moves it by ~0.02 ms/token.

Prints the generated-text tail for every run. CLAUDE.md: never quote a latency
number without checking the text.
"""
import glob
import os
import re
import sys

RES = sys.argv[1] if len(sys.argv) > 1 else "/mnt/nvme1/glm5_flag_ab"
NOISE = 0.26

FIELDS = {
    "avg": re.compile(r"decode_avg_ms=([0-9.]+|None)"),
    "min": re.compile(r"decode_min_ms=([0-9.]+|None)"),
    "g1": re.compile(r"G1_cross_rank=(\w+)"),
    "dist": re.compile(r"G2_distinct=([0-9.]+)"),
    "osl": re.compile(r"dumped_osl=(\d+)"),
}
TAIL = re.compile(r"TEXT_TAIL: (.*)")
DECODE = re.compile(
    r"Decode:\s+(\d+) tokens in (\d+)/(\d+) iters .*?avg ([0-9.]+)ms/iter")
RANGE = re.compile(r"Decode per-iter range: min=([0-9.]+)ms max=([0-9.]+)ms")


def parse(path):
    out = {}
    try:
        body = open(path, errors="replace").read()
    except OSError:
        return None
    for k, rx in FIELDS.items():
        m = rx.search(body)
        out[k] = m.group(1) if m else None
    m = TAIL.search(body)
    out["text"] = m.group(1) if m else ""
    out["failed"] = "FAILED" in body
    m = re.search(r"ARM_ENV='([^']*)'", body)
    out["env"] = m.group(1) if m else ""
    m = re.search(r"\[isa\] ([0-9a-f]{32})", body)
    out["md5"] = m.group(1) if m else None

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
    ctl = "control" if "control" in arms else arms[0]
    arms = [ctl] + [a for a in arms if a != ctl]

    for stat in ("min", "avg"):
        print()
        print(f"=== decode_{stat}_ms, arms alternating in time ===")
        print(f"{'rep':>4s}" + "".join(f"{a:>12s}" for a in arms))
        for rep in reps:
            row = f"{rep:>4d}"
            for a in arms:
                v = fnum(runs.get((a, rep), {}).get(stat))
                row += f"{v:12.3f}" if v is not None else f"{'--':>12s}"
            print(row)
        print(f"--- paired delta vs {ctl} (same rep) ---")
        for a in arms[1:]:
            ds = []
            for rep in reps:
                c = fnum(runs.get((ctl, rep), {}).get(stat))
                v = fnum(runs.get((a, rep), {}).get(stat))
                if c is not None and v is not None:
                    ds.append(v - c)
            if not ds:
                print(f"  {a:>12s}  no complete pair")
                continue
            mean = sum(ds) / len(ds)
            verdict = "NULL (under noise)" if abs(mean) < NOISE else "RESOLVABLE"
            print(f"  {a:>12s}  n={len(ds)}  mean={mean:+.3f}  "
                  f"per-pair {' '.join(f'{d:+.3f}' for d in ds)}   {verdict}")
        print(f"  wall noise floor is {NOISE} ms -- nothing smaller is claimable.")

    print()
    print("=== per-run integrity (never quote latency without this) ===")
    for (a, r) in sorted(runs):
        d = runs[(a, r)]
        print(f"  {a:>12s} r{r}  env='{d['env']}'  osl={d['osl']}  G1={d['g1']}  "
              f"distinct={d['dist']}  iters={d.get('iters')}  "
              f"max_iter={d.get('max')}ms  md5={d.get('md5')}  "
              f"{'FAILED' if d['failed'] else ''}")
        print(f"      tail: {d['text'][:120]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
