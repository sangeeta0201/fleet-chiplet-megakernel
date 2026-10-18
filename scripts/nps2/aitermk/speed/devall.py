"""devall.py LOG: kernel device time per pass pooled over all server processes in one log.

[FWD_PASS_TOTAL] iters=N total_ms=T is cumulative within a process; a new process starts when iters drops.
Each process contributes its last line.
"""
import re
import sys

procs, prev = [], None
for line in open(sys.argv[1], errors="replace"):
    for m in re.finditer(r"\[FWD_PASS_TOTAL\] iters=(\d+) total_ms=([0-9.]+)", line):
        n, t = int(m.group(1)), float(m.group(2))
        if prev is None or n < prev[0]:
            procs.append((n, t))
        else:
            procs[-1] = (n, t)
        prev = (n, t)
n, t = sum(x for x, _ in procs), sum(y for _, y in procs)
print("device %.4f ms/pass (%d passes, %d process%s)" % (t / n, n, len(procs), "" if len(procs) == 1 else "es")
      if n else "no FWD_PASS_TOTAL")

