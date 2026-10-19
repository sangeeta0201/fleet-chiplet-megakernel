#!/usr/bin/env python3
"""blog_tab.py: the GPT-OSS-120B numbers of the blog from the runs of run_dpx.sh / run_default.sh and the served
GSM8K-128 results. Prints markdown tables, BF16 LM head first (the checkpoint's head, as vLLM runs it: the
precision-matched configuration), then the default build's MXFP4 head; --charts writes the data file
blog_charts.py draws from, its figures with the LM head of --charts-head (mx, the default: the MXFP4 head of the
released builds, as in the blog's figures; bf: the BF16 head). The GSM8K and 1K-conversation speculation tables
come from gsm8k/table.py.

usage: blog_tab.py [--dpx DIR] [--default DIR] [--served DIR] [--served-bf DIR] [--served-bf-logs DIR]
                   [--stock JSON] [--charts OUT.json] [--charts-head mx|bf]
  --dpx            logs of run_dpx.sh      (default $FLEET_LOG_ROOT/lane_fair)
  --default        logs of run_default.sh  (default $FLEET_LOG_ROOT/lane_fair2)
  --served         WORK/results of libAiterMK's gpt_oss_build_and_repro.sh (MXFP4 head, the released 32K builds):
                   ns/e3/df8 .json and .log (optional)
  --served-bf      results of the BF16-head serves (scripts/nps2/fair/fair_t1.sh): <pre>_<cfg>_bf<k>.json, several
                   serves of one build averaged (optional)
  --served-bf-logs their logs, <pre>_<cfg>_bf<k>.log (default $FLEET_LOG_ROOT/lane_fair)
  --served-bf-pre  file prefix of those serves (default f31: the vLLM 0.31.0 sweep)
  --stock          stock vLLM summary (gsm128_vllm_summary.py JSON; default stock_vllm_031.json next to this script)
  --charts-head    LM head of the figures blog_charts.py draws (figure 3 swap, figure 4 phase, the served panel):
                   mx (default, the blog's figures) or bf; the file keeps both (swap_mx, swap_bf, served_mx,
                   served_bf, the phase entry's mx and bf values)
A label with a rerun next to it (<label>x.log) uses the rerun.
Definitions: single-stream ms/token = the run's "Decode: ... avg" (generated tokens only); speculative ms/token =
ms per pass / tokens per pass, tokens per pass = (accepted + decode passes) / decode passes ([SPEC] counters);
"x% faster" = default-mode time / DPX+NPS2 time - 1.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("FLEET_LOG_ROOT", "/home/user/nps1/fleet_baselines")
ap = argparse.ArgumentParser()
ap.add_argument("--dpx", default=ROOT + "/lane_fair")
ap.add_argument("--default", default=ROOT + "/lane_fair2")
ap.add_argument("--served")
ap.add_argument("--served-bf")
ap.add_argument("--served-bf-logs", default=ROOT + "/lane_fair")
ap.add_argument("--served-bf-pre", default="f31")
ap.add_argument("--stock", default=os.path.join(HERE, "stock_vllm_031.json"))
ap.add_argument("--charts")
ap.add_argument("--charts-head", choices=("mx", "bf"), default="mx")
A = ap.parse_args()
NOTES = []


def path(d, lab):
    p = os.path.join(d, lab)
    if os.path.exists(p + "x.log"):
        NOTES.append("%s: rerun %sx used" % (lab, lab))
        return p + "x"
    return p


def txt(d, lab):
    try:
        return open(path(d, lab) + ".log", errors="replace").read()
    except OSError:
        return None


def decode(d, lab):
    t = txt(d, lab)
    m = re.findall(r"avg ([0-9.]+) ?ms/iter", t or "")
    return float(m[-1]) if m else None


def spec(d, lab):  # (ms per pass, tokens per pass, ms per token)
    t = txt(d, lab)
    m = re.findall(r"\[SPEC\] passes=\d+ decode_passes=(\d+) drafted=\d+ accepted=(\d+)", t or "")
    ms = decode(d, lab)
    if not m or ms is None:
        return None
    dp, ac = map(int, m[-1])
    tpp = (ac + dp) / dp
    return ms, tpp, ms / tpp


def accepted(d, lab):  # (accepted, drafted) of a speculative run
    m = re.findall(r"\[SPEC\] passes=\d+ decode_passes=\d+ drafted=(\d+) accepted=(\d+)", txt(d, lab) or "")
    return (int(m[-1][1]), int(m[-1][0])) if m else None


def texthash(d, lab):
    try:
        return open(path(d, lab) + ".texthash").read().split()[0]
    except (OSError, IndexError):
        return None


def same(a, b):
    return "yes" if a and b and a == b else ("NO" if a and b else "?")


def pct(a, b):
    return (a / b - 1) * 100 if a and b else float("nan")


def f3(x):
    return "%.3f" % x if x else "missing"


D, F = A.dpx, A.default
charts = {}
print("# GPT-OSS-120B blog numbers on the latest code\n")
print("DPX+NPS2 logs: %s\ndefault-mode logs: %s\n" % (D, F))

LENS = [1024, 2048, 4096, 8192, 16384, 32768]
HEADS = (("bf", "BF16 LM head"), ("mx", "MXFP4 LM head"))


def locality(hd, title):
    """Sections 1-3 for one LM head. Labels: 1K single stream = the phase group's plain run (ph_<mode>_<hd>), the
    others bl_<mode>_ns[bf]N, EAGLE3 bl_dpx_sp[bf]N, weights across dies bl_dpx_{ns,sp}[bf]1024_swap."""
    s = "bf" if hd == "bf" else ""
    lab_d = lambda n: "ph_dpx_%s" % hd if n == 1024 else "bl_dpx_ns%s%d" % (s, n)
    lab_f = lambda n: "ph_spx_%s" % hd if n == 1024 else "bl_spx_ns%s%d" % (s, n)
    print("## 1 (%s). Data placement alone: single stream, 982-token prompt + N generated\n" % title)
    print("| generated | default mode ms/token | DPX+NPS2 ms/token | DPX+NPS2 faster | same output |")
    print("|---|---|---|---|---|")
    loc = {}
    for n in LENS:
        a, b = decode(F, lab_f(n)), decode(D, lab_d(n))
        loc[n] = (a, b)
        print("| %dK | %s | %s | %.1f%% | %s |" % (n // 1024, f3(a), f3(b), pct(a, b),
                                                  same(texthash(F, lab_f(n)), texthash(D, lab_d(n)))))
    print("\n## 2 (%s). Speculative path (EAGLE3, 1 draft) in DPX+NPS2 against the default-mode single stream\n" % title)
    print("| generated | ms per pass | tokens per pass | drafts accepted | ms/token | faster than default single stream |")
    print("|---|---|---|---|---|---|")
    for n in LENS:
        q, ac = spec(D, "bl_dpx_sp%s%d" % (s, n)), accepted(D, "bl_dpx_sp%s%d" % (s, n))
        if q:
            print("| %dK | %.3f | %.3f | %d of %d (%.1f%%) | %.4f | %.1f%% |" % (
                n // 1024, q[0], q[1], ac[0], ac[1], 100.0 * ac[0] / ac[1], q[2], pct(loc[n][0], q[2])))
        else:
            print("| %dK | missing | | | | |" % (n // 1024))
    e3 = texthash(D, "bl_dpx_sp%s1024" % s)
    print("\nEAGLE3 output (1K) == single stream at the same attention partition (15 KV chunks): %s" %
          same(e3, texthash(D, "bl_dpx_ns%s1024_ck15" % s)))
    ck = texthash(D, "bl_dpx_ns%s32768_ck15" % s)
    if ck:
        print("EAGLE3 output (32K; the shorter runs are its prefixes) == single stream at 15 KV chunks: %s" %
              same(texthash(D, "bl_dpx_sp%s32768" % s), ck))
    print("\n## 3 (%s). Same kernel and output, every read-only weight stored on the other die (MPK_W_SWAP=1), 1K\n" % title)
    print("| | same die ms/token | across dies ms/token | slower | same output |")
    print("|---|---|---|---|---|")
    a, b = decode(D, lab_d(1024)), decode(D, "bl_dpx_ns%s1024_swap" % s)
    print("| single stream | %s | %s | +%.1f%% | %s |" % (f3(a), f3(b), pct(b, a), same(texthash(D, lab_d(1024)),
                                                                                 texthash(D, "bl_dpx_ns%s1024_swap" % s))))
    s0, s1 = spec(D, "bl_dpx_sp%s1024" % s), spec(D, "bl_dpx_sp%s1024_swap" % s)
    if s0 and s1:
        print("| EAGLE3, 1 draft | %.3f | %.3f | +%.1f%% | %s |" % (s0[2], s1[2], pct(s1[2], s0[2]),
                                                                  same(e3, texthash(D, "bl_dpx_sp%s1024_swap" % s))))
        if a and b:
            charts["swap_" + hd] = {"ns": [round(a, 3), round(b, 3)],
                                    "e3": [round(s0[2], 3), round(s1[2], 3)], "head": title}
    print()


for hd, title in HEADS:
    locality(hd, title)

# 4. per phase
print("## 4. Per phase (phase recorder over the 1,024 generated tokens, scaled to the plain run's time)\n")
NAMES = ["Between layers", "Gate wait + QKV", "QKV epoch", "Attention + merge", "Attention release",
         "O-proj + router", "TopK", "MoE", "Layer gate", "Outside the layers"]
BLOG = [("Q/K/V hand-off", "sync", 2), ("TopK", "sync", 6), ("attention release", "sync", 4),
        ("O-proj + router combine", "sync", 5), ("MoE expert stream", "bandwidth", 7)]
LMI = 9


def phases(d, mode, arm):
    lab = "ph_%s_%s" % (mode, arm)
    p = path(d, lab + "_ps") + ".log"
    if not os.path.exists(p):
        return None
    out = subprocess.run([sys.executable, os.path.join(HERE, "..", "q_pslot_tok.py"), lab, p], capture_output=True,
                         text=True).stdout
    m = re.search(r"^PY \S+ \[([^\]]*)\]", out, re.M)
    ms = decode(d, lab)
    if not m or ms is None:
        return None
    r = [float(x) for x in m.group(1).split(",")]
    return [x * ms * 1000 / sum(r) for x in r]


ph = {(m, a): phases(D if m == "dpx" else F, m, a) for m in ("spx", "dpx") for a in ("mx", "bf")}
print("| phase | default BF16 | DPX+NPS2 BF16 | default MXFP4 | DPX+NPS2 MXFP4 |")
print("|---|---|---|---|---|")
for i, nm in enumerate(NAMES):
    row = [ph[k][i] if ph[k] else float("nan") for k in (("spx", "bf"), ("dpx", "bf"), ("spx", "mx"), ("dpx", "mx"))]
    print("| %s | %.1f | %.1f | %.1f | %.1f |" % (nm, *row))
print("\nus per token. Speedup per phase (default / DPX+NPS2 - 1):\n")
print("| phase | BF16 LM head | MXFP4 LM head |")
print("|---|---|---|")
sp = {}
for arm in ("bf", "mx"):
    if ph[("spx", arm)] and ph[("dpx", arm)]:
        s, d_ = ph[("spx", arm)], ph[("dpx", arm)]
        sp[arm] = [pct(s[i], d_[i]) for _, _, i in BLOG] + [pct(s[LMI], d_[LMI]), pct(sum(s), sum(d_))]
for j, (nm, kind, _) in enumerate(BLOG + [("outside the layers: LM head, argmax, embedding", "", 0),
                                          ("whole token", "", 0)]):
    print("| %s%s | %s | %s |" % (nm, " (%s)" % kind if kind else "", *["%.1f%%" % sp[a][j] if a in sp else "missing"
                                                                       for a in ("bf", "mx")]))
if "mx" in sp and "bf" in sp:
    charts["phase"] = {"labels": [b[0] for b in BLOG], "kinds": [b[1] for b in BLOG], "lead": A.charts_head,
                       "mx": [round(x, 1) for x in sp["mx"][:5]], "bf": [round(x, 1) for x in sp["bf"][:5]],
                       "lm_label": "outside the layers: LM head, argmax, embedding",
                       "lm": {"mx": round(sp["mx"][5], 1), "bf": round(sp["bf"][5], 1)}}

# 5. critical path of one pass (PSNAP, pass 1500 = ~1.5K context), data vs sync
print("\n## 5. Critical path per layer (pass 1500), default mode - DPX+NPS2\n")
for arm in ("bf", "mx"):
    j = "/tmp/blog_attr_%s_%d.json" % (arm, os.getpid())
    r = subprocess.run([sys.executable, os.path.join(HERE, "..", "dpx", "q_attr.py"), path(D, "ph_dpx_%s_ps" % arm),
                        path(F, "ph_spx_%s_ps" % arm)], capture_output=True, text=True, env=dict(os.environ, ATTR_JSON=j))
    try:
        seg = json.load(open(j))["segments"]
        os.remove(j)
    except (OSError, ValueError, KeyError):
        print("%s: missing" % arm)
        continue
    dl = {c: sum(s["nps1"] - s["dpx"] for s in seg if s["cls"] == c) for c in ("data", "sync")}
    tot = dl["data"] + dl["sync"]
    per = "; ".join("%s %+.2f" % (s["name"], s["nps1"] - s["dpx"]) for s in seg)
    print("- %s LM head: layer %.2f (DPX+NPS2) vs %.2f us (default), +%.2f us: data %+.2f (%.0f%%), sync %+.2f (%.0f%%). "
          "Per segment: %s" % ("MXFP4" if arm == "mx" else "BF16", sum(s["dpx"] for s in seg),
                               sum(s["nps1"] for s in seg), tot, dl["data"], 100 * dl["data"] / tot, dl["sync"],
                               100 * dl["sync"] / tot, per))

# 6. GSM8K and the 1K conversation: gsm8k/table.py prints them (each answer at its own length, the Oct 13 method)
print("\n## 6. GSM8K and the 1K conversation: python3 gsm8k/table.py (runs: gsm8k/run_dpx.sh, gsm8k/run_default.sh)")

# 7. served GSM8K-128 through vllm serve
sys.path.insert(0, HERE)
from spec_records import distinct_full, spec_records  # noqa: E402

PASS_RE = re.compile(r"\[FWD_PASS\] iter=(\d+) time_ms=([0-9.]+)")
CNT = ("questions", "rounds", "tokens", "drafted", "accepted")


def rounds(logp):
    """Acceptance over one record per question, from the kernel's [SPEC] counters (split records rejoined):
    dict(questions, rounds, tokens, drafted, accepted, tpr = tokens per round, kms = kernel ms per round when the
    build printed its passes, else None), or None."""
    try:
        lines = open(logp, "rb").read().decode("utf-8", "replace").splitlines()
    except OSError:
        return None
    recs, _ = spec_records(lines)
    keep = distinct_full(recs)
    if not keep:
        return None
    starts = sorted(i for i, _ in recs)
    passes = [(i, int(a), float(b)) for i, ln in enumerate(lines) for a, b in PASS_RE.findall(ln)]
    r = dict(questions=len(keep), rounds=0, tokens=0, drafted=0, accepted=0)
    tim, timed = 0.0, bool(passes)
    for i, (pa, dp, dr, ac, fs) in keep:
        r["rounds"] += dp
        r["tokens"] += ac + dp
        r["drafted"] += dr
        r["accepted"] += ac
        if timed:  # a launch's pass times follow its [SPEC] line, up to the next record
            nxt = min([s for s in starts if s > i], default=len(lines))
            it = {a: b for j, a, b in passes if i < j < nxt}
            if not it:
                timed = False
                continue
            last = max(it)
            d_ = [t for a, t in it.items() if a > last - dp]
            tim += sum(d_) / len(d_) * dp
    r["tpr"] = r["tokens"] / r["rounds"]
    r["kms"] = tim / r["rounds"] if timed and tim else None
    return r


stock = {}
try:
    st = json.load(open(A.stock))
    pick = lambda pre: max((v for k, v in st.items() if k.split("_")[-1].startswith(pre)), key=lambda v: v["pooled"])
    stock = {"ns": pick("k0"), "e3": pick("e3"), "df": pick("df")}
except (OSError, ValueError):
    pass


def served_block(title, runs, refnote):
    """runs: {cfg: [(json, log), ...]}, one or more serves of the same build (tok/s, TTFT and kernel ms per round
    averaged; the counters of every serve must agree)."""
    print("\n## 7 (%s). Served: GSM8K-128 through vllm serve, 256 tokens, one request at a time\n" % title)
    print("| setup | serves | megakernel tok/s | tokens per round | ms per round | TTFT ms | tokens = %s | streamed == "
          "kernel arena | stock vLLM tok/s (tokens/step, ms/step) | megakernel / stock |" % refnote)
    print("|---|---|---|---|---|---|---|---|---|---|")
    mk, acc = {}, {}
    for c, nm, sk in (("ns", "no speculation", "ns"), ("e3", "EAGLE3", "e3"), ("df8", "DFlash", "df")):
        sms = []
        for jp, lp in runs.get(c, []):
            try:
                sms.append((json.load(open(jp))["summary"], lp))
            except (OSError, ValueError, KeyError):
                pass
        if not sms:
            print("| %s | missing |" % nm)
            continue
        tps = sum(s["decode_tok_s"] for s, _ in sms) / len(sms)
        ttft = sum(s.get("ttft_ms_mean", float("nan")) for s, _ in sms) / len(sms)
        mk[sk] = tps
        rr = None
        if c != "ns":
            rrs = [rounds(lp) for _, lp in sms]
            rr = rrs[0]
            if rr and any(x is None or [x[k] for k in CNT] != [rr[k] for k in CNT] for x in rrs[1:]):
                NOTES.append("%s %s: the serves' [SPEC] counters differ" % (title, nm))
            km = [x["kms"] for x in rrs if x and x["kms"]]
            if rr:
                rr = dict(rr, kms=sum(km) / len(km) if len(km) == len(rrs) else None)
                acc[nm] = (rr, sms[0][0].get("questions", 0))
        if c == "ns":
            tpr, ms_r, star = 1.0, 1000 / tps, ""
        else:
            tpr = rr["tpr"] if rr else None
            ms_r = rr["kms"] if rr and rr["kms"] else (1000 * tpr / tps if tpr else None)
            star = "" if rr and rr["kms"] else "*"
        so = stock.get(sk)
        sdesc = "%.0f (%.2f, %.2f)" % (so["pooled"], so["tok_per_step"], so["ms_per_step"]) if so else "?"
        s0 = sms[0][0]
        print("| %s | %d | %.1f | %s | %s%s | %.1f | %d/%d | %d/%d | %s | %s |" % (
            nm, len(sms), tps, "%.3f" % tpr if tpr else "?", "%.3f" % ms_r if ms_r else "?", star, ttft,
            s0.get("tokens_same", 0), s0.get("questions", 0), s0.get("streamed_eq_arena", 0), s0.get("questions", 0),
            sdesc, "%.2fx" % (tps / so["pooled"]) if so else "?"))
    if stock and len(mk) == 3:
        fs, fm = max(v["pooled"] for v in stock.values()), max(mk.values())
        print("\nfastest of each: megakernel %.1f vs stock vLLM %.1f tok/s = %.2fx" % (fm, fs, fm / fs))
    if any(v[0]["kms"] is None for v in acc.values()):
        print("* ms per round = tokens per round / served decode rate (the build prints no pass times)")
    if acc:
        print("\nAccepted tokens (the kernel's [SPEC] counters, one record per question):\n")
        print("| setup | questions | rounds | drafts proposed | drafts accepted | accepted share | drafts per round | "
              "tokens per round |")
        print("|---|---|---|---|---|---|---|---|")
        for nm, (rr, nq) in acc.items():
            print("| %s | %d of %d | %d | %d | %d | %.1f%% | %.2f | %.3f |" % (
                nm, rr["questions"], nq, rr["rounds"], rr["drafted"], rr["accepted"],
                100 * rr["accepted"] / rr["drafted"], rr["drafted"] / rr["rounds"], rr["tpr"]))
    return mk


if A.served_bf:
    runs = {}
    for c in ("ns", "e3", "df8"):
        js = sorted(glob.glob(os.path.join(A.served_bf, "%s_%s_bf*.json" % (A.served_bf_pre, c))))
        runs[c] = [(j, os.path.join(A.served_bf_logs, os.path.basename(j)[:-5] + ".log")) for j in js]
    mkb = served_block("BF16 LM head, GSM8K-length builds", runs, "MXFP4 reference")
    if stock and len(mkb) == 3:
        charts["served_bf"] = {"stock": {k: round(v["pooled"]) for k, v in stock.items()},
                               "mk": {k: round(v) for k, v in mkb.items()}, "head": "BF16 LM head"}
    # the MXFP4 arm of the same sweep (same builds but for the head, same session, alternated with the BF16 serves)
    runs = {}
    for c in ("ns", "e3", "df8"):
        js = sorted(glob.glob(os.path.join(A.served_bf, "%s_%s_mx*.json" % (A.served_bf_pre, c))))
        runs[c] = [(j, os.path.join(A.served_bf_logs, os.path.basename(j)[:-5] + ".log")) for j in js]
    if any(runs.values()):
        mkx = served_block("MXFP4 LM head, the same GSM8K-length builds and session", runs, "MXFP4 reference")
        for k in mkb:
            if k in mkx:
                NOTES.append("BF16 LM head costs the GSM8K-length build %.1f%% (%s)" % (100 * (1 - mkb[k] / mkx[k]), k))
if A.served:
    runs = {c: [(os.path.join(A.served, c + ".json"), os.path.join(A.served, c + ".log"))] for c in ("ns", "e3", "df8")}
    mkm = served_block("MXFP4 LM head, released 32K builds", runs, "reference")
    if stock and len(mkm) == 3:
        charts["served_mx"] = {"stock": {k: round(v["pooled"]) for k, v in stock.items()},
                               "mk": {k: round(v) for k, v in mkm.items()}, "head": "MXFP4 LM head"}
if NOTES:
    print("\nNotes: " + "; ".join(sorted(set(NOTES))))
if A.charts:
    for k in ("swap", "served"):  # the sections blog_charts.py draws, from the head of --charts-head
        if "%s_%s" % (k, A.charts_head) in charts:
            charts[k] = charts["%s_%s" % (k, A.charts_head)]
    json.dump(charts, open(A.charts, "w"), indent=1)
    print("\nchart data -> %s (%s); figures with the %s" % (A.charts, ", ".join(charts),
                                                         dict(HEADS)[A.charts_head]))

