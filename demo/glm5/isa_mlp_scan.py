#!/usr/bin/env python3
"""Whole-function MLP scan: one row per device function in the image.

Companion to isa_mlp_window.py, which needs a window. This one takes the whole
body, so it covers the phases that have NO load-carrying loop (qkv_a, q_b) and
whose memory behaviour is therefore invisible to isa_outstanding.py.

MLP is the mean loads-outstanding at each retiring s_waitcnt vmcnt(N). Mixing
prologue and epilogue into the body dilutes it, so read this as a screen and
confirm a hit with isa_mlp_window.py over the specific region.

Usage: isa_mlp_scan.py ISA.txt [--filter SUBSTR ...]
Offline; no GPU run.
"""
import argparse
import re

HDR = re.compile(r"^([0-9a-f]+) <(.+)>:")
INS = re.compile(r"^\t([a-z][a-z0-9_]*)\s*(.*?)\s*//\s*([0-9A-F]+):")
VM = ("buffer_load", "global_load", "flat_load", "scratch_load",
      "buffer_store", "global_store", "flat_store", "scratch_store",
      "buffer_atomic", "global_atomic", "flat_atomic")
LDONLY = ("buffer_load", "global_load", "flat_load", "scratch_load")
VMCNT = re.compile(r"vmcnt\((\d+)\)")


def parse(path):
    out, cur = [], None
    for line in open(path):
        h = HDR.match(line)
        if h:
            cur = (h.group(2), [])
            out.append(cur)
            continue
        if cur is None:
            continue
        m = INS.match(line)
        if m:
            cur[1].append((int(m.group(3), 16), m.group(1), m.group(2)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("isa")
    ap.add_argument("--filter", nargs="*", default=[])
    args = ap.parse_args()

    print(f"{'function (whole body)':<60} {'loads':>6} {'MLP':>6} "
          f"{'peak':>5} {'drains':>7} {'waits':>6}")
    print("-" * 96)
    seen = set()
    for name, body in parse(args.isa):
        if args.filter and not any(f in name for f in args.filter):
            continue
        if name in seen or not body:
            continue
        seen.add(name)
        out = peak = 0
        stalls = []
        for _a, op, ops in body:
            if op.startswith(VM):
                out += 1
                peak = max(peak, out)
            if op == "s_waitcnt":
                m = VMCNT.search(ops)
                if m:
                    n = int(m.group(1))
                    if out > n:
                        stalls.append((out, n))
                        out = n
        nload = sum(1 for _a, op, _o in body if op.startswith(LDONLY))
        if nload == 0:
            continue
        mlp = sum(s[0] for s in stalls) / len(stalls) if stalls else float("nan")
        drains = sum(1 for s in stalls if s[1] == 0)
        short = re.sub(r"EEEv.*", "", re.sub(r"^_ZN6kernel\d+", "", name))[:58]
        print(f"{short:<60} {nload:>6} {mlp:>6.2f} {peak:>5} "
              f"{drains:>7} {len(stalls):>6}")


if __name__ == "__main__":
    main()
