#!/usr/bin/env python3
"""Map ATT stats Vaddrs onto code-object .text symbols."""
from __future__ import annotations

import csv
import re
import subprocess
import collections
import sys


def load_syms(objdump: str, path: str):
    out = subprocess.check_output([objdump, "-t", path], text=True, errors="replace")
    syms = []
    for line in out.splitlines():
        m = re.match(
            r"^([0-9a-fA-F]+)\s+.*\.text\s+([0-9a-fA-F]+)\s+(?:\.protected\s+)?(\S+)",
            line,
        )
        if m:
            addr, size, name = int(m.group(1), 16), int(m.group(2), 16), m.group(3)
            if size > 0:
                syms.append((addr, size, name))
    syms.sort()
    return syms


def short(name: str) -> str:
    if "gang_mla_full_layer_fused" in name:
        return "full_layer_fused"
    if "worker_kernel" in name:
        return "worker_kernel"
    if "gang_rmsnorm_linear_bias_topk" in name:
        return "rmsnorm_qkv_topk"
    if "gang_moe_w2" in name:
        return "moe_w2"
    if "moe_w13" in name:
        return "moe_w13"
    if "gang_mla_decode" in name:
        return "mla_decode"
    if "scheduler" in name:
        return "scheduler"
    if "execute_worker" in name:
        return "execute_worker"
    return name[:48]


def lookup(syms, va: int) -> str:
    for addr, size, name in syms:
        if addr <= va < addr + size:
            return short(name)
    return "other"


def classify(inst: str) -> str:
    if inst.startswith("s_barrier"):
        return "s_barrier"
    if inst.startswith("s_waitcnt"):
        return "s_waitcnt"
    if "mfma" in inst:
        return "mfma"
    if inst.startswith("s_sleep"):
        return "s_sleep"
    if "buffer_load" in inst or "global_load" in inst:
        return "load"
    if "buffer_store" in inst or "global_store" in inst:
        return "store"
    if inst.startswith("ds_"):
        return "lds"
    return "other"


def num(x) -> float:
    try:
        return float(x or 0)
    except ValueError:
        return 0.0


def main():
    stats, co, objdump = sys.argv[1], sys.argv[2], sys.argv[3]
    syms = load_syms(objdump, co)
    rows = list(csv.DictReader(open(stats)))
    tot = sum(num(r["Latency"]) for r in rows) or 1.0
    by = collections.defaultdict(lambda: collections.defaultdict(float))
    for r in rows:
        fn = lookup(syms, int(r["Vaddr"]))
        cls = classify(r["Instruction"].strip())
        lat = num(r["Latency"])
        by[fn][cls] += lat
        by[fn]["ALL"] += lat
    hdr = f"{'func':<22} {'tot%':>7} {'bar%':>7} {'wait%':>7} {'mfma%':>7} {'load%':>7} {'sleep%':>7}"
    print(hdr)
    for fn, d in sorted(by.items(), key=lambda kv: -kv[1]["ALL"])[:12]:
        print(
            f"{fn:<22} {100*d['ALL']/tot:6.2f}% {100*d['s_barrier']/tot:6.2f}% "
            f"{100*d['s_waitcnt']/tot:6.2f}% {100*d['mfma']/tot:6.2f}% "
            f"{100*d['load']/tot:6.2f}% {100*d['s_sleep']/tot:6.2f}%"
        )


if __name__ == "__main__":
    main()
