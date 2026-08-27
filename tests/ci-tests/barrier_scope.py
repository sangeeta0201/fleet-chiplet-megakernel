#!/usr/bin/env python3
"""Test each barrier's scope from the trace, independently of the source.

The scope in a barrier's NAME comes from reading the kernel: which atomic it
increments, whose flag it polls, who fans the release out. That reading can be
wrong -- naming a span from its position rather than its source is the bug this
trace tooling exists to catch, and it has happened before. So this script
re-derives scope from the measurement alone and prints both, letting them
disagree out loud rather than trusting the label.

The test is CAUSAL: whose arrival decides when an XCD is released.

  GLOBAL     an XCD leaves a fixed delay after the LAST ARRIVAL ANYWHERE on
             the GPU, so `exit - last_arrival_gpu` is near-constant while
             `exit - last_arrival_own_xcd` swings with how early that die
             happened to finish.
  XCD-LOCAL  the reverse: `exit - last_arrival_own_xcd` is the constant one.

Whichever difference has the lower spread is the one the hardware is actually
waiting on. Both are the same quantity measured on the same spans, so the
comparison survives a slower machine or a different model.

An earlier version of this script compared release SIMULTANEITY instead --
cross-XCD exit spread against within-XCD exit spread -- and got three of five
barriers wrong. A hierarchical barrier (one global arrival, then a serialized
fan-out to 8 per-XCD flags) releases every rank on a die at once, because they
all poll one flag, while the fan-out itself skews the dies apart by a few
hundred ns. Simultaneity reads that as LOCAL. It is not: the arrival that
opens the fan-out is global, and no local rebalancing moves it. Coupling is a
question about causes, so it has to be measured on causes.

Exit time is ts+dur, not ts. Entry is when a worker arrived, which says
nothing about coupling; the release is what the barrier does. dur also
includes each barrier's post-release tail, but that is a constant offset and
this test reads only spread.

  Usage: barrier_scope.py <trace.json>
"""
import json
import statistics
import sys
from collections import defaultdict


# A barrier whose durations barely vary is one where nobody actually waits on
# anybody: every worker pays the same fixed cost. 1 us is ~2.5x the measured
# post-release tail of the tightest barrier here (0.40 us) and well under the
# smallest genuinely-binding one, so it separates the two populations cleanly
# without being tuned to either.
BIND_US = 1.0


def main(path):
    with open(path) as f:
        ev = [e for e in json.load(f)["traceEvents"]
              if e.get("ph") == "X" and "barrier (" in e.get("name", "")]
    if not ev:
        print("no barrier spans in this trace -- was it exported from a build "
              "with the barrier marks (codes 16-21)?", file=sys.stderr)
        return 1

    names = sorted({e["name"] for e in ev})
    wide = max(len(n) for n in names)
    print(f"  {'barrier':<{wide}}  {'claimed':>10}  {'sd|gpu':>8}  "
          f"{'sd|xcd':>8}  {'ratio':>7}  {'measured':>10}")
    print("  " + "-" * (wide + 52))
    bad = []
    for name in names:
        spans = [e for e in ev if e["name"] == name]
        # Last arrival per layer (GPU-wide) and per XCD-layer.
        gpu, own = defaultdict(float), defaultdict(float)
        for e in spans:
            a = e["args"]
            gpu[a["layer"]] = max(gpu[a["layer"]], e["ts"])
            k = (a["layer"], a["xcd"])
            own[k] = max(own[k], e["ts"])
        # One release per XCD-layer, so measure the delay once per XCD-layer
        # (median exit of its ranks) rather than once per worker, which would
        # weight a die by how many workers it happens to hold.
        exits = defaultdict(list)
        for e in spans:
            a = e["args"]
            exits[(a["layer"], a["xcd"])].append(e["ts"] + e["dur"])
        d_gpu, d_own = [], []
        for (l, x), v in exits.items():
            ex = statistics.median(v)
            d_gpu.append(ex - gpu[l])
            d_own.append(ex - own[(l, x)])
        if len(d_gpu) < 2:
            continue
        sg, so = statistics.pstdev(d_gpu), statistics.pstdev(d_own)
        claimed = ("GLOBAL" if "(GLOBAL)" in name else
                   "XCD-LOCAL" if "(XCD-LOCAL)" in name else "?")
        # A barrier nobody waits at carries no scope signal. Both delays are
        # then dominated by the same measurement noise and their ratio lands
        # near 1, which is not evidence of anything -- reporting a verdict off
        # it would manufacture a disagreement out of jitter. Bindingness is a
        # precondition for the test, so it is checked before the test runs.
        # (Cost, as opposed to scope, is what span_table.py reports.)
        spread = statistics.pstdev([e["dur"] for e in spans])
        if spread < BIND_US:
            print(f"  {name:<{wide}}  {claimed:>10}  {sg:>7.2f}u  {so:>7.2f}u  "
                  f"{'--':>7}  {'non-binding':>10}"
                  f"   (dur sd {spread:.2f}u < {BIND_US}u: no signal)")
            continue
        verdict = "GLOBAL" if sg <= so else "XCD-LOCAL"
        agree = "" if claimed == verdict else "   <-- DISAGREES"
        if agree:
            bad.append((name, claimed, verdict))
        print(f"  {name:<{wide}}  {claimed:>10}  {sg:>7.2f}u  {so:>7.2f}u  "
              f"{sg/so if so else float('inf'):>7.2f}  {verdict:>10}{agree}")
    print(f"\n  sd|gpu = std dev of (release - last arrival GPU-wide)\n"
          f"  sd|xcd = std dev of (release - last arrival on own XCD)\n"
          f"  ratio < 1 means the release tracks the whole GPU more closely "
          f"than it tracks\n          this die, so the die cannot free itself "
          f"by finishing early.")
    if bad:
        print(f"\n  {len(bad)} barrier(s) whose name disagrees with the "
              f"measurement -- fix the name or re-read the source.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
