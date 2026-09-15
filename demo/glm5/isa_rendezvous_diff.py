#!/usr/bin/env python3
"""Count rendezvous/spin sites in a worker_kernel disassembly, and diff two arms.

  isa_rendezvous_diff.py CONTROL/dev.s VARIANT/dev.s

A software rendezvous on this branch is not an `s_barrier` -- it is
`hier_barrier_arrive` (a device-scope atomic) plus a poll loop whose backedge
carries `s_sleep`. So the ISA signature of DELETING one is:

  * one fewer `s_sleep` site (the poll's backedge), and
  * one fewer backwards branch to it.

`s_barrier` is counted too, because __syncthreads() sites bracket every one of
these and a change that accidentally removed one would matter.

Scoped to worker_kernel: the scheduler kernel and rocSHMEM carry their own
sleeps and would swamp a two-site delta.
"""
import re
import sys
import collections

FUNC = re.compile(r"^[0-9a-f]+ <([^>]+)>:")
# llvm-objdump for amdgcn emits "\t<mnemonic> <operands>  // ADDR: ENCODING",
# with no leading address column on the instruction line itself.
INSN = re.compile(r"^\t([a-z][a-z0-9_]*)")

WATCH = ("s_sleep", "s_barrier", "buffer_inv", "global_atomic_add",
         "global_load_dword", "s_setprio")


def counts(path, want="worker_kernel"):
    cur = None
    per = collections.Counter()
    total = collections.Counter()
    nfunc = 0
    with open(path, errors="replace") as fh:
        for ln in fh:
            m = FUNC.match(ln)
            if m:
                cur = m.group(1)
                if want in cur:
                    nfunc += 1
                continue
            m = INSN.match(ln)
            if not m:
                continue
            op = m.group(1)
            total[op] += 1
            if cur and want in cur:
                per[op] += 1
    return per, total, nfunc


def main():
    a, b = sys.argv[1], sys.argv[2]
    pa, ta, na = counts(a)
    pb, tb, nb = counts(b)
    print(f"A = {a}   (worker_kernel symbols: {na})")
    print(f"B = {b}   (worker_kernel symbols: {nb})")
    print()
    print(f"{'insn':24s} {'A(worker)':>10s} {'B(worker)':>10s} {'delta':>8s}"
          f" {'A(all)':>9s} {'B(all)':>9s} {'delta':>8s}")
    print("-" * 82)
    keys = [k for k in WATCH]
    for k in keys:
        pref = [x for x in set(list(ta) + list(tb)) if x.startswith(k)]
        for kk in sorted(pref):
            print(f"{kk:24s} {pa[kk]:10d} {pb[kk]:10d} {pb[kk] - pa[kk]:8d}"
                  f" {ta[kk]:9d} {tb[kk]:9d} {tb[kk] - ta[kk]:8d}")
    print()
    tot_a = sum(pa.values())
    tot_b = sum(pb.values())
    print(f"worker_kernel total instructions: A {tot_a}  B {tot_b}  "
          f"delta {tot_b - tot_a:+d}")
    # Everything that moved by more than a rounding amount, so a change that
    # landed somewhere unexpected is visible rather than assumed away.
    print()
    print("largest per-opcode moves inside worker_kernel:")
    moves = sorted(set(list(pa) + list(pb)),
                   key=lambda k: -abs(pb[k] - pa[k]))
    for k in moves[:15]:
        d = pb[k] - pa[k]
        if d == 0:
            break
        print(f"  {k:28s} {pa[k]:7d} -> {pb[k]:7d}   {d:+d}")


if __name__ == "__main__":
    main()
