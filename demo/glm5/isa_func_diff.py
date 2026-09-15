#!/usr/bin/env python3
"""Per-function instruction diff between two amdgcn disassemblies.

  isa_func_diff.py A/dev.s B/dev.s

The fused-layer bodies are not inside `worker_kernel` -- the task graph puts
them in their own device functions -- so a rendezvous deletion shows up as a
delta in whichever function inlined it, not in the entry point. Print every
function whose instruction count or `s_sleep` count moved, so "the change did
not land where I thought" is visible instead of assumed away.
"""
import collections
import re
import sys

FUNC = re.compile(r"^[0-9a-f]+ <([^>]+)>:")
INSN = re.compile(r"^\t([a-z][a-z0-9_]*)")


def per_func(path):
    cur = None
    ops = collections.defaultdict(collections.Counter)
    tot = collections.Counter()
    for ln in open(path, errors="replace"):
        m = FUNC.match(ln)
        if m:
            cur = m.group(1)
            tot.setdefault(cur, 0)
            continue
        m = INSN.match(ln)
        if m and cur:
            ops[cur][m.group(1)] += 1
            tot[cur] += 1
    return ops, tot


def main():
    A, ta = per_func(sys.argv[1])
    B, tb = per_func(sys.argv[2])
    names = sorted(set(list(ta) + list(tb)))
    print("A = %s" % sys.argv[1])
    print("B = %s" % sys.argv[2])
    print()
    hdr = ("%-58s %8s %8s %7s %6s %6s %6s" %
           ("function", "A tot", "B tot", "d tot", "A slp", "B slp", "d slp"))
    print(hdr)
    print("-" * len(hdr))
    moved = 0
    for n in names:
        da, db = ta.get(n, 0), tb.get(n, 0)
        sa, sb = A[n]["s_sleep"], B[n]["s_sleep"]
        if da != db or sa != sb:
            moved += 1
            print("%-58s %8d %8d %+7d %6d %6d %+6d"
                  % (n[:58], da, db, db - da, sa, sb, sb - sa))
    print()
    print("functions that moved: %d of %d" % (moved, len(names)))
    # Opcode-level detail for the movers, so the delta can be attributed to a
    # deleted poll loop rather than to incidental scheduling.
    for n in names:
        if ta.get(n, 0) == tb.get(n, 0) and A[n]["s_sleep"] == B[n]["s_sleep"]:
            continue
        print()
        print("  %s" % n[:100])
        keys = sorted(set(list(A[n]) + list(B[n])),
                      key=lambda k: -abs(B[n][k] - A[n][k]))
        for k in keys[:12]:
            d = B[n][k] - A[n][k]
            if d == 0:
                break
            print("    %-30s %6d -> %6d  %+d" % (k, A[n][k], B[n][k], d))


if __name__ == "__main__":
    main()
