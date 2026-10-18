#!/usr/bin/env python3
"""q_attr.py: SPX+NPS1 minus DPX+NPS2 per critical-path segment, non-spec and spec (PSNAP recorder runs, same pass).

Chain per layer, gate to gate (q_vtchain.py's anchoring: previous layer's last worker past the gate). Segments are
grouped by what they wait on:
  data  = QKV (gate -> last QKV end: fold, QKV weights, KV write), attention (KV scan), O-proj + router (O-proj
          weights, slice waits), MoE (expert stream, W13 -> W2 hand-off)
  sync  = QKV epoch, attention release, TopK (8-die router sums, selection), P9 arrival, layer gate
Usage: q_attr.py DPX_NS NPS1_NS [DPX_SP NPS1_SP]
  Each argument is a label in $ATTR_LOG_DIR (default fleet_baselines/lane_dpxe3) or a log path without ".log".
  The JSON goes to $ATTR_JSON (default q_attr.json in that directory).
"""
import json
import os
import re
import sys

L = os.environ.get("ATTR_LOG_DIR", "/home/user/nps1/fleet_baselines/lane_dpxe3").rstrip("/") + "/"
SEG = [("QKV", 1, "data"), ("QKV epoch", 2, "sync"), ("attention", 3, "data"), ("release wait", 5, "sync"),
       ("O-proj + router", 6, "data"), ("TopK", 7, "sync"), ("MoE", 8, "data"), ("P9 arrival", 9, "sync"),
       ("layer gate", 10, "sync")]


def chain(lab):
    rows = []
    for ln in open((lab if "/" in lab else L + lab) + ".log", errors="replace"):
        if ln.startswith("[PSNAP]") and "B=" in ln:
            m = re.search(r"L=(\d+) B=(\d+)", ln)
            pairs = re.findall(r"(\d+):(\d+)", ln.split("B=", 1)[1].split(" ", 1)[1])
            if m and len(pairs) >= 11:
                rows.append((int(m.group(1)), int(m.group(2)), [int(b) for _, b in pairs]))
    tot = {s[0]: [] for s in SEG}
    for k in range(1, len(rows)):
        L0, B0, last0 = rows[k - 1]
        L1, B1, last1 = rows[k]
        if L1 != L0 + 1:
            continue
        prev = B0 + last0[10]
        for name, s, _ in SEG:
            t = B1 + last1[s]
            tot[name].append((t - prev) / 1000.0)
            prev = t
    return {n: sum(v) / len(v) for n, v in tot.items() if v}


def ms_of(lab, kind):
    try:
        t = open(L + lab + ".log", errors="replace").read()
    except Exception:
        return None
    d = re.findall(r"avg ([0-9.]+) ?ms/iter", t)
    return float(d[-1]) if d else None


labs = sys.argv[1:5]
ch = [chain(l) for l in labs]
out = {"labels": labs, "segments": []}
pairs = [("non-spec", (ch[0], ch[1]))] + ([("spec", (ch[2], ch[3]))] if len(ch) == 4 else [])
for kind, (a, b) in pairs:
    print("%s  (us per layer, SPX+NPS1 - DPX+NPS2)" % kind)
    print("  %-16s %8s %8s %8s  %s" % ("segment", "DPX", "NPS1", "delta", "class"))
    cls = {"data": 0.0, "sync": 0.0}
    for name, _, c in SEG:
        if name in a and name in b:
            dl = b[name] - a[name]
            cls[c] += dl
            print("  %-16s %8.2f %8.2f %+8.2f  %s" % (name, a[name], b[name], dl, c))
            out["segments"].append(dict(kind=kind, name=name, cls=c, dpx=round(a[name], 3), nps1=round(b[name], 3)))
    ta, tb = sum(a.values()), sum(b.values())
    print("  %-16s %8.2f %8.2f %+8.2f" % ("period", ta, tb, tb - ta))
    print("  data segments %+.2f us/layer, sync segments %+.2f us/layer (x36 layers = %+.0f / %+.0f us per pass)"
          % (cls["data"], cls["sync"], 36 * cls["data"], 36 * cls["sync"]))
json.dump(out, open(os.environ.get("ATTR_JSON", L + "q_attr.json"), "w"), indent=1)

