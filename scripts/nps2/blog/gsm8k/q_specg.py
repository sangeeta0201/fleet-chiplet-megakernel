"""Summarize one specg.sh run (the method of ~/nps1/specg/q_specg.py, Oct 13 2026).

ns / ns7 / def: the answer's natural length E (tokens up to and including the first <|return|>) and the mean decode
time per token over those E tokens, from the per-pass trace ([FWDTRACE], MPK_FWDTRACE_MIN=1). Pass i consumes token
i, so the prompt takes passes 0..PL-1 (the last one yields token 1) and token t>1 comes from pass PL+t-2.
k1 / k3: verify-pass time, tokens per pass and ms/token from [SPEC]; token identity against REF (a ns / ns7 label).
usage: LANE_DIR=<log dir> GSM_OUT=<dir> q_specg.py LABEL KIND [REF]
"""
import json
import os
import re
import sys

G = os.environ.get("GSM_OUT", "/home/user/mk1/specg_latest").rstrip("/") + "/"
L = os.environ.get("LANE_DIR", "/home/user/nps1/fleet_baselines/lane_fair").rstrip("/") + "/"
RET = 200002  # <|return|>: end of the final answer
lab, kind = sys.argv[1], sys.argv[2]
ref = sys.argv[3] if len(sys.argv) > 3 else ""


def tokens(label):
    try:
        d = json.load(open(G + "tok_%s.json" % label))
        return d.get("prompt_ids", []), d.get("token_ids", [])
    except Exception:
        return None, None


try:
    txt = open(L + lab + ".log", errors="replace").read()
except Exception:
    txt = ""
pids, toks = tokens(lab)
dec = re.findall(r"avg ([0-9.]+) ?ms/iter", txt)
out = "%s kind=%s " % (lab, kind)
if not dec or toks is None:
    err = re.findall(r"(Traceback[^\n]{0,60}|NOT DONE at 45s[^\n]{0,60}|Memory access fault[^\n]{0,40}|error: [^\n]{0,90}|TIMEOUT[^\n]{0,30})", txt)
    out += "FAILED %s" % ";".join(err[:3])
else:
    pl = len(pids)
    e = toks.index(RET) + 1 if RET in toks else len(toks)
    out += "prompt=%d new=%d eos_at=%s " % (pl, len(toks), e if RET in toks else "none")
    if kind in ("ns", "ns7", "def"):
        tr = {}
        for line in txt.splitlines():
            if line.startswith("[FWDTRACE]"):
                for tok in line.split()[1:]:
                    k, v = tok.split(":")
                    tr[int(k)] = float(v)
        sel = [tr[i] for i in range(pl, pl + e - 1) if i in tr]
        avg_e = sum(sel) / len(sel) if sel else float("nan")
        out += "decode_avg_all=%s ms_per_token_first_E=%.4f (%d passes traced)" % (dec[-1], avg_e, len(sel))
    else:
        sp = re.findall(r"\[SPEC\] passes=(\d+) decode_passes=(\d+) drafted=(\d+) accepted=(\d+)", txt)
        if sp:
            dp, dr, ac = int(sp[-1][1]), int(sp[-1][2]), int(sp[-1][3])
            tpp = (dp + ac) / dp if dp else 0
            ms = float(dec[-1]) / tpp if tpp else float("nan")
            out += "pass_ms=%s accepted=%d/%d tokens_per_pass=%.3f ms_per_token=%.4f" % (dec[-1], ac, dr, tpp, ms)
        else:
            out += "no [SPEC] line"
    if ref:
        _, rt = tokens(ref)
        if rt:
            m = min(len(toks), len(rt))
            k = next((i for i in range(m) if toks[i] != rt[i]), m)
            out += " | vs %s identical %d of %d" % (ref, k, m)
print(out)
with open(G + "results.txt", "a") as f:
    f.write(out + "\n")

