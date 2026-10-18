#!/usr/bin/env python3
# sum.py LABEL [REF...]: one s15 run (lane from env LANE, default stk15). tau = (accepted + decode_passes) /
# decode_passes from [SPEC]; pass mean over the last decode_passes [FWDTRACE]/[FWD_PASS] iterations;
# ms/token = mean / tau; tok/s = 1000 / ms_per_token; token identity vs each REF (opt/spec/tok_<label>.json).
import json
import os
import re
import statistics as st
import sys

LD = "/home/user/nps1/fleet_baselines/lane_" + os.environ.get("LANE", "stk15")
TD = "/home/user/nps1/opt/spec"


def toks(lab):
    q = os.path.join(TD, "tok_%s.json" % lab)
    if not os.path.exists(q):
        return None
    j = json.load(open(q))
    t = j.get("token_ids") if isinstance(j, dict) else j
    if t and isinstance(t[0], list):
        t = t[0]
    return t


lab = sys.argv[1]
p = os.path.join(LD, lab + ".log")
s = open(p, "rb").read().decode("utf-8", "replace").replace("\r", "")
try:
    th = open(os.path.join(LD, lab + ".texthash")).read().strip()
except Exception:
    th = ""
spec = re.findall(r"\[SPEC\] passes=(\d+) decode_passes=(\d+) drafted=(\d+) accepted=(\d+)", s)
it = {}
for ln in re.findall(r"\[FWDTRACE\]([^\n]*)", s):
    for a, b in re.findall(r"(\d+):([0-9.]+)", ln):
        it[int(a)] = float(b)
for a, b in re.findall(r"\[FWD_PASS\] iter=(\d+) time_ms=([0-9.]+)", s):
    it[int(a)] = float(b)
rows = sorted(it.items())
vt = sorted(set(re.findall(r"-DMPK_VTN_T=(\d+)", s)))
kk = sorted(set(re.findall(r"MPK_SPEC_K=(\d+)", s)))
cf = sorted(set(re.findall(r"MPK_SPEC_CONF=([0-9.]+)", s)))
drafter = ("eagle3" if "-DMPK_EAGLE3" in s else ("dflash" if "-DMPK_DFLASH" in s else "none"))
print("== %s: %s T=%s K=%s conf=%s texthash=%s" % (lab, drafter, ",".join(vt), ",".join(kk), ",".join(cf) or "-", th))
if spec and rows:
    pa, dp, dr, ac = map(int, spec[-1])
    thr = rows[-1][0] - dp
    d = [t for i, t in rows if i > thr]
    tau = (ac + dp) / max(dp, 1)
    mean = st.mean(d) if d else 0.0
    msk = mean / tau if tau else 0.0
    print("   [SPEC] passes=%d decode_passes=%d drafted=%d accepted=%d -> accept %.3f, tau %.3f, M_eff %.2f" %
          (pa, dp, dr, ac, ac / max(dr, 1), tau, 1 + dr / max(dp, 1)))
    print("   decode passes %d: mean %.3f median %.3f ms -> %.4f ms/token, %.0f tok/s" %
          (len(d), mean, st.median(d) if d else 0.0, msk, 1000.0 / msk if msk else 0.0))
elif rows:
    print("   no [SPEC] line; %d pass rows" % len(rows))
else:
    print("   no data")
a = toks(lab)
for ref in sys.argv[2:]:
    b = toks(ref)
    if a is None or b is None:
        print("   tokens vs %s: missing" % ref)
        continue
    n = min(len(a), len(b))
    fd = next((i for i in range(n) if a[i] != b[i]), None)
    print("   tokens vs %-14s len %d/%d, first diff %s -> %s" %
          (ref, len(a), len(b), fd, "IDENTICAL" if fd is None and len(a) == len(b) else "DIFFERENT"))

