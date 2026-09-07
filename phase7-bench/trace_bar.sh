#!/bin/bash
# trace_bar.sh [label] [harness args...]
#
# Attribute the Phase 7 `bar` bucket to its four internal steps, per XCD, using
# the MPK_OPROJ_BAR_TRACE build. The question this exists to answer: `bar` is
# +0.6-0.7 us in NPS2-with-the-fix against NPS1, and the two candidate
# explanations -- the single cross-die level-2 counter, or the un-homed level-1
# lines -- both fit that number arithmetically.
#
# What the three printf populations are:
#   [BAR_ARR]  the one worker per XCD that closed level 1 and went on to level 2.
#              Gives l1 (its own local atomic) and l2 (the cross-die atomic).
#   [BAR_OBS]  tile 0 of each XCD. Gives l1 for a worker that is *not* the
#              closer, plus the absolute tick at which it observed the release
#              and the acquire cost after that.
#   [BAR_REL]  the single global releaser. Gives the absolute tick at which the
#              release became visible.
#
# obs - rel is formed by joining on `ep` (layer_epoch), because the two ticks
# come from different blocks. s_memrealtime is device-wide, so that subtraction
# is meaningful across XCDs; the 10 ns tick is the resolution floor.
set -u
cd "$HOME/fleet-chiplet-megakernel" || exit 1

LABEL=${1:-trace}
shift || true
LAYERS=${LAYERS:-150}
LOG="$HOME/tr_$LABEL.log"

CP=$(cat /sys/bus/pci/devices/0000:05:00.0/current_compute_partition)
MP=$(cat /sys/bus/pci/devices/0000:05:00.0/current_memory_partition)
echo "=== $LABEL   $CP/$MP   $LAYERS layers   args: $* ==="

timeout 900 ./drive_phase7_trace --layers="$LAYERS" --tiles=23 "$@" \
	--tag="$LABEL" >"$LOG" 2>&1
rc=$?
if [ "$rc" != "0" ]; then
	echo "  ABORT rc=$rc"
	tail -5 "$LOG"
	exit 1
fi
grep -m1 local_split "$LOG" | sed 's/^/  /'
grep -m1 'hash ' "$LOG" | sed 's/^/  /'
echo "  lines: ARR=$(grep -c BAR_ARR "$LOG") OBS=$(grep -c BAR_OBS "$LOG") REL=$(grep -c BAR_REL "$LOG")"

python3 - "$LOG" <<'PY'
import re, sys, statistics as st

src = sys.argv[1]
arr, obs, rel = [], [], {}
kv = re.compile(r"(\w+)=(-?[0-9.]+)")
for line in open(src):
    if "[BAR_" not in line:
        continue
    d = dict(kv.findall(line))
    if "[BAR_ARR]" in line:
        arr.append((int(d["ep"]), int(d["xcd"]), float(d["l1"]), float(d["l2"])))
    elif "[BAR_OBS]" in line:
        obs.append((int(d["ep"]), int(d["xcd"]), float(d["l1"]),
                    int(d["obs"]), float(d["acq"]), float(d["drain"]),
                    float(d["wait"]), float(d["bar"])))
    elif "[BAR_REL]" in line:
        rel[int(d["ep"])] = (int(d["xcd"]), int(d["rel"]))

# Drop the same 30% warmup buckets.sh drops, by epoch not by row, since the
# three populations have different row counts per layer.
eps = sorted({e for e, *_ in arr} | {e for e, *_ in obs})
if not eps:
    print("  no trace rows"); raise SystemExit
cut = eps[int(len(eps) * 0.30)]
arr = [r for r in arr if r[0] >= cut]
obs = [r for r in obs if r[0] >= cut]

def med(v):
    return st.median(v) if v else float("nan")

print("\n  Where the bar bucket goes  ([BAR_OBS], p50 us, sums to bar)")
print("  %-5s %8s %8s %8s %8s %9s" % ("xcd", "drain", "l1", "wait", "acq",
                                      "bar"))
for x in range(8):
    r = [o for o in obs if o[1] == x]
    if not r:
        continue
    print("  %-5d %8.3f %8.3f %8.3f %8.3f %9.3f"
          % (x, med([q[5] for q in r]), med([q[2] for q in r]),
             med([q[6] for q in r]), med([q[4] for q in r]),
             med([q[7] for q in r])))
print("  %-5s %8.3f %8.3f %8.3f %8.3f %9.3f"
      % ("all", med([q[5] for q in obs]), med([q[2] for q in obs]),
         med([q[6] for q in obs]), med([q[4] for q in obs]),
         med([q[7] for q in obs])))

print("\n  Level 1 and level 2, from the per-XCD closer  ([BAR_ARR], us)")
print("  %-5s %8s %8s %8s %8s   %s" % ("xcd", "l1 p50", "l2 p50", "l2 p10",
                                       "l2 p90", "n"))
for x in range(8):
    l1 = sorted(r[2] for r in arr if r[1] == x)
    l2 = sorted(r[3] for r in arr if r[1] == x)
    if not l2:
        continue
    n = len(l2)
    print("  %-5d %8.3f %8.3f %8.3f %8.3f   %d"
          % (x, med(l1), med(l2), l2[int(.1 * n)], l2[min(int(.9 * n), n - 1)], n))

lo2 = [r[3] for r in arr if r[1] < 4]
hi2 = [r[3] for r in arr if r[1] >= 4]
lo1 = [r[2] for r in arr if r[1] < 4]
hi1 = [r[2] for r in arr if r[1] >= 4]
print("\n  AID halves (XCD 0-3 vs 4-7), p50 us")
print("    level 1:  %.3f  vs  %.3f   delta %+.3f" %
      (med(lo1), med(hi1), med(hi1) - med(lo1)))
print("    level 2:  %.3f  vs  %.3f   delta %+.3f" %
      (med(lo2), med(hi2), med(hi2) - med(lo2)))

print("\n  Release to observation, joined on ep  ([BAR_OBS] obs - [BAR_REL] rel)")
print("  %-5s %10s %10s %10s   %s" % ("xcd", "p50 us", "p10", "p90", "n"))
d_lo, d_hi = [], []
for x in range(8):
    d = []
    for row in obs:
        ep, xc, o = row[0], row[1], row[3]
        if xc != x or ep not in rel:
            continue
        d.append((o - rel[ep][1]) * 10.0 / 1000.0)
    if not d:
        continue
    d.sort()
    n = len(d)
    print("  %-5d %10.3f %10.3f %10.3f   %d"
          % (x, med(d), d[int(.1 * n)], d[min(int(.9 * n), n - 1)], n))
    (d_lo if x < 4 else d_hi).extend(d)
if d_lo and d_hi:
    print("    halves:  %.3f  vs  %.3f   delta %+.3f" %
          (med(d_lo), med(d_hi), med(d_hi) - med(d_lo)))

print("\n  Acquire after observation ([BAR_OBS] acq, p50 us): %.3f"
      % med([r[4] for r in obs]))
print("  Level 1 on a non-closer ([BAR_OBS] l1, p50 us):    %.3f"
      % med([r[2] for r in obs]))

# Which XCD ends up releasing, and whether that changes the release cost. If the
# global counter is homed in one partition, the releaser is whichever XCD's
# closer arrived last, so a skew here is itself informative.
who = {}
for ep, (x, _t) in rel.items():
    if ep >= cut:
        who[x] = who.get(x, 0) + 1
print("  Releaser by XCD: %s" % " ".join("%d:%d" % (k, who[k])
                                         for k in sorted(who)))
PY
