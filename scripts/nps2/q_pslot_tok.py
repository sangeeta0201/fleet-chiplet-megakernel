#!/usr/bin/env python3
"""Per-phase time per token from [PSLOTW] (all 248 workers), mean of the given runs.

Each [PSLOTW] row is one worker: w=<id> n=<layers> s0..s13, s_i = mean ns per layer of the span
ending at slot i (s13 = the per-token work outside the layers, spread over the 36 layers). Every
worker's row sums to the same pass time, so the phases add up to the token time.
usage: q_pslot_tok.py LABEL run1.log [run2.log ...]"""
import re
import sys

GROUPS = [
    ("Between layers", [0, 12]),
    ("Gate wait + QKV", [1]),
    ("QKV epoch", [2]),
    ("Attention + merge", [3, 4]),
    ("Attention release", [5]),
    ("O-proj + router", [6]),
    ("TopK", [7]),
    ("MoE", [8]),
    ("Layer gate", [9, 10, 11]),
    ("Outside the layers", [13]),
]


def run_means(path):
    rows = []
    for ln in open(path, errors="replace"):
        if ln.startswith("[PSLOTW]"):
            nums = [int(x) for x in re.findall(r"-?\d+", ln.split("n=", 1)[1])]
            rows.append(nums[1:15])
    k = len(rows)
    return [sum(r[i] for r in rows) / k for i in range(14)], k


label, logs = sys.argv[1], sys.argv[2:]
per_run = []
for p in logs:
    m, k = run_means(p)
    fwd = ""
    for ln in open(p, errors="replace"):
        if "FWD_PASS_TOTAL" in ln or ln.strip().startswith("Decode:"):
            fwd += " | " + ln.strip()[:90]
    tot = sum(m) * 36 / 1000
    print(f"{label} {p.split('/')[-1]}: workers {k}, token {tot:.1f} us{fwd}")
    per_run.append(m)
mean = [sum(r[i] for r in per_run) / len(per_run) for i in range(14)]
total = sum(mean) * 36 / 1000
print(f"{label} MEAN token time {total:.1f} us")
for name, idx in GROUPS:
    us = sum(mean[i] for i in idx) * 36 / 1000
    print(f"  {name:22s} {us:8.1f} us/token  ({us / total * 100:5.1f}%)")
print("PY", label, [round(sum(mean[i] for i in idx) * 36 / 1000, 1) for _, idx in GROUPS])

