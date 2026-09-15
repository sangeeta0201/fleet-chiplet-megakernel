#!/usr/bin/env python3
"""Reduce rocprofv3 ATT stats_*.csv to opcode-class stall share."""
from __future__ import annotations

import csv
import glob
import os
import re
import sys
from collections import defaultdict

CLASS_RULES = [
    ("s_waitcnt", re.compile(r"\bs_waitcnt\b")),
    ("s_waitcnt_vscnt", re.compile(r"\bs_waitcnt_vscnt\b")),
    ("s_barrier", re.compile(r"\bs_barrier\b|\bs_waitcnt.*lgkmcnt|\bs_wakeup\b")),
    ("s_memrealtime/sleep", re.compile(r"\bs_memrealtime\b|\bs_sleep\b|\bs_nop\b")),
    ("atomic/flat", re.compile(r"\b(?:flat|global)_atomic|\bds_atomic|\bbuffer_atomic")),
    ("mfma", re.compile(r"\bv_mfma|\bv_wmma|\bv_mmac")),
    ("buffer_load", re.compile(r"\bbuffer_load|\bglobal_load|\bflat_load|\bscratch_load")),
    ("buffer_store", re.compile(r"\bbuffer_store|\bglobal_store|\bflat_store|\bscratch_store")),
    ("ds_", re.compile(r"\bds_")),
    ("valu", re.compile(r"\bv_(?!mfma|wmma|mmac)")),
    ("salu/smem", re.compile(r"\bs_(?:load|buffer|mov|add|and|or|xor|cmp|cbranch|setpc|swappc|endpgm)")),
]


def classify(inst: str) -> str:
    s = inst.strip()
    for name, rx in CLASS_RULES:
        if rx.search(s):
            return name
    if s.startswith("s_"):
        return "scalar_other"
    if s.startswith("v_"):
        return "valu"
    return "other"


def pick_col(headers, *cands):
    lower = {h.lower(): h for h in headers}
    for c in cands:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def num(v):
    if v is None or v == "":
        return 0.0
    try:
        return float(v.replace(",", ""))
    except ValueError:
        return 0.0


def load_csv(path):
    rows = []
    with open(path, newline="", errors="replace") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            return [], []
        headers = list(r.fieldnames)
        for row in r:
            rows.append(row)
    return headers, rows


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    files = sorted(glob.glob(os.path.join(root, "**", "stats_*.csv"), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(root, "**", "*stats*.csv"), recursive=True))
    if not files:
        print(f"no stats csv under {root}", file=sys.stderr)
        sys.exit(2)

    class_lat = defaultdict(float)
    class_stall = defaultdict(float)
    class_hits = defaultdict(float)
    inst_lat = defaultdict(float)
    inst_stall = defaultdict(float)
    inst_hits = defaultdict(float)
    n_rows = 0
    used = []

    for path in files:
        headers, rows = load_csv(path)
        if not rows:
            continue
        inst_c = pick_col(headers, "Instruction", "instruction", "Inst", "disassembly")
        lat_c = pick_col(headers, "Latency", "latency", "Duration", "cycles", "Cycles")
        stall_c = pick_col(headers, "Stall", "stall", "StallCycles", "stall_cycles")
        idle_c = pick_col(headers, "Idle", "idle")
        hit_c = pick_col(headers, "Hitcount", "hitcount", "Count", "count", "Hits")
        src_c = pick_col(headers, "Source", "source", "File", "file")
        if inst_c is None:
            print(f"skip {path}: headers={headers[:12]}", file=sys.stderr)
            continue
        used.append(path)
        print(f"# {path}")
        print(f"# headers={headers}")
        for row in rows:
            inst = (row.get(inst_c) or "").strip()
            if not inst:
                continue
            lat = num(row.get(lat_c)) if lat_c else 0.0
            stall = num(row.get(stall_c)) if stall_c else 0.0
            idle = num(row.get(idle_c)) if idle_c else 0.0
            hits = num(row.get(hit_c)) if hit_c else 0.0
            # Some dumps put stall in Idle; keep both.
            cost = lat if lat else (stall + idle)
            cls = classify(inst)
            class_lat[cls] += cost
            class_stall[cls] += stall if stall else idle
            class_hits[cls] += hits
            key = inst.split("//")[0].strip()
            if len(key) > 80:
                key = key[:80]
            inst_lat[key] += cost
            inst_stall[key] += stall if stall else idle
            inst_hits[key] += hits
            n_rows += 1
            src = (row.get(src_c) or "").strip() if src_c else ""
            if src:
                inst_lat[key]  # keep
        print()

    tot = sum(class_lat.values()) or 1.0
    tot_s = sum(class_stall.values()) or 1.0
    print(f"files={len(used)} rows={n_rows} latency_sum={tot:.0f} stall_sum={tot_s:.0f}")
    print()
    print(f"{'class':<22} {'lat%':>7} {'stall%':>8} {'hits':>12} {'lat':>14}")
    for cls, lat in sorted(class_lat.items(), key=lambda kv: -kv[1]):
        print(
            f"{cls:<22} {100*lat/tot:6.1f}% {100*class_stall[cls]/tot_s:7.1f}% "
            f"{class_hits[cls]:12.0f} {lat:14.0f}"
        )
    print()
    print("top instructions by latency")
    for inst, lat in sorted(inst_lat.items(), key=lambda kv: -kv[1])[:40]:
        print(
            f"{100*lat/tot:6.2f}% stall={inst_stall[inst]:10.0f} hits={inst_hits[inst]:8.0f}  {inst}"
        )


if __name__ == "__main__":
    main()
