#!/usr/bin/env python3
"""MEMORY-LEVEL-PARALLELISM CENSUS over a STRAIGHT-LINE ISA window.

Why this exists when isa_outstanding.py already does: that script only looks at
LOOPS (it keys off backward s_cbranch), so it is blind to every fully-unrolled
region.  The split-KV merge is exactly that -- `#pragma unroll` over
NUM_KV_CHUNKS -- so it appears in no ISA census on this branch even though it is
the largest phase in the layer.

WHAT IT REPORTS.  Walk the window in order, simulating the vmcnt queue:

  at each s_waitcnt vmcnt(N):
      before = loads outstanding when the wave arrives at the wait
      retire = before - N  (loads that must land before the wave proceeds)
      the wave is exposed to memory latency here IF before is small, because
      `before` IS the memory-level parallelism at that stall

  MLP_at_stall  = mean of `before` over waits that actually retire something.
                  1.0 means one load in flight per round trip: fully exposed.
  peak          = max outstanding anywhere in the window
  drains        = count of s_waitcnt vmcnt(0) that retire >= 1 load
  issue_runs    = length histogram of consecutive-load bursts

READING IT.  MLP_at_stall >= 8 is the flat part of the curve measured in
tests/standalone/test_waves_per_simd_payoff.hip (5.45 TB/s); MLP 1 is the
3.45 TB/s point, a 1.58x on that region's bytes.

Usage:
  isa_mlp_window.py ISA.txt --func SUBSTR [--from-idx A --to-idx B]
                            [--anchor-op v_exp_f32_e32 --pad 80] [--dump]
Offline; no GPU run.
"""
import argparse
import re
from collections import Counter

HDR = re.compile(r"^([0-9a-f]+) <(.+)>:")
INS = re.compile(r"^\t([a-z][a-z0-9_]*)\s*(.*?)\s*//\s*([0-9A-F]+):")
LD = ("buffer_load", "global_load", "flat_load", "scratch_load", "ds_read")
VM = ("buffer_load", "global_load", "flat_load", "scratch_load",
      "buffer_store", "global_store", "flat_store", "scratch_store",
      "buffer_atomic", "global_atomic", "flat_atomic")
VMCNT = re.compile(r"vmcnt\((\d+)\)")
LGKM = re.compile(r"lgkmcnt\((\d+)\)")


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


def analyse(seg, dump=False, carry_in=0):
    out = carry_in
    peak = 0
    stalls = []          # (before, n, retired)
    run = 0
    runs = []
    lgkm_waits = 0
    nload = sum(1 for _, op, _ in seg if op.startswith(LD))
    for a, op, ops in seg:
        if op.startswith(VM) and not op.startswith(("ds_",)):
            out += 1
            peak = max(peak, out)
            run += 1
        else:
            if run:
                runs.append(run)
                run = 0
        if op == "s_waitcnt":
            m = VMCNT.search(ops)
            if m:
                n = int(m.group(1))
                if out > n:
                    stalls.append((out, n, out - n))
                    out = n
            if LGKM.search(ops):
                lgkm_waits += 1
        if dump:
            print(f"  {a:#x}  out={out:<3} {op:<24} {ops[:60]}")
    if run:
        runs.append(run)
    return out, peak, stalls, runs, nload, lgkm_waits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("isa")
    ap.add_argument("--func", required=True)
    ap.add_argument("--min-insns", type=int, default=0)
    ap.add_argument("--from-idx", type=int)
    ap.add_argument("--to-idx", type=int)
    ap.add_argument("--from-addr")
    ap.add_argument("--to-addr")
    ap.add_argument("--anchor-op")
    ap.add_argument("--pad", type=int, default=80)
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--loop", action="store_true",
                    help="window is a loop body: settle the vmcnt queue over "
                         "one trip before reporting the next")
    args = ap.parse_args()

    funcs = [(n, b) for n, b in parse(args.isa)
             if args.func in n and len(b) >= args.min_insns]
    if not funcs:
        raise SystemExit(f"no function matching {args.func!r}")
    if args.from_addr:
        a0 = int(args.from_addr, 16)
        funcs = [f for f in funcs if f[1] and f[1][0][0] <= a0 <= f[1][-1][0]] \
            or funcs
    name, body = funcs[0]

    lo = args.from_idx if args.from_idx is not None else 0
    hi = args.to_idx if args.to_idx is not None else len(body)
    if args.from_addr:
        a0 = int(args.from_addr, 16)
        lo = next(i for i, (a, _, _) in enumerate(body) if a >= a0)
    if args.to_addr:
        a1 = int(args.to_addr, 16)
        hi = 1 + max(i for i, (a, _, _) in enumerate(body) if a <= a1)
    if args.anchor_op:
        idx = [i for i, (_, op, _) in enumerate(body) if op == args.anchor_op]
        if not idx:
            raise SystemExit(f"anchor {args.anchor_op} not found")
        lo, hi = max(0, idx[0] - args.pad), min(len(body), idx[-1] + args.pad)

    seg = body[lo:hi]
    # A loop body carries loads across the backedge, so a single pass starting
    # from out=0 under-counts every wait in the first trip. Settle the queue
    # once, then report the steady-state trip.
    carry_in = analyse(seg)[0] if args.loop else 0
    carry, peak, stalls, runs, nload, lgkm = analyse(seg, args.dump, carry_in)
    retiring = [s for s in stalls if s[2] > 0]
    mlp = sum(s[0] for s in retiring) / len(retiring) if retiring else float("nan")
    drains = sum(1 for s in retiring if s[1] == 0)

    print(f"func   {name[:90]}")
    print(f"window idx {lo}..{hi}  addr {seg[0][0]:#x}..{seg[-1][0]:#x}  "
          f"{len(seg)} insns")
    print(f"loads          {nload}")
    print(f"MLP_at_stall   {mlp:.2f}    (mean loads outstanding when the wave "
          f"hits a retiring s_waitcnt)")
    print(f"peak           {peak}")
    print(f"retiring waits {len(retiring)}   of which vmcnt(0) drains: {drains}")
    print(f"lgkmcnt waits  {lgkm}")
    print(f"issue runs     {sorted(Counter(runs).items())}   "
          "(burst length -> count)")
    print(f"carry out      {carry}")
    print("stall profile (before -> vmcnt(N), retired):")
    for before, n, ret in stalls[:40]:
        print(f"   {before:>3} -> {n:<3}  retired {ret}")
    if len(stalls) > 40:
        print(f"   ... {len(stalls) - 40} more")


if __name__ == "__main__":
    main()
