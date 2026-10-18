#!/usr/bin/env python3
"""dry_check.py REF DRY: compare the environments of DRY=1 runs with the ones recorded in run logs.

REF and DRY hold "label|ENVS|..." lines (REF: the ENVS line each run's log starts with; DRY: run_dpx.sh /
run_default.sh with DRY=1). For every label in both, the effective value of each variable (the last assignment
wins, as in `export A=1 A=2`) must agree. Prints the differences and exits 1 on any."""
import sys


def load(path):
    out = {}
    for ln in open(path):
        parts = ln.rstrip("\n").split("|")
        if len(parts) >= 2 and "=" in parts[1]:
            d = {}
            for tok in parts[1].split():
                k, _, v = tok.partition("=")
                d[k] = v
            out[parts[0]] = d
    return out


ref, dry = load(sys.argv[1]), load(sys.argv[2])
bad = 0
for lab in sorted(set(ref) & set(dry)):
    a, b = ref[lab], dry[lab]
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    if diff:
        bad += 1
        print("%s: %d differ: %s" % (lab, len(diff), "; ".join("%s ref=%s dry=%s" % (k, a.get(k), b.get(k))
                                                              for k in diff[:8])))
    else:
        print("%s: identical (%d variables)" % (lab, len(a)))
missing = sorted(set(ref) - set(dry))
if missing:
    print("not in the dry run:", " ".join(missing))
sys.exit(1 if bad else 0)

