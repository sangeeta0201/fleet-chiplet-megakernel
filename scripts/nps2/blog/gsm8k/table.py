"""table.py [--out DIR] [--tag lg] [--ctag lc]: the guide's section 4 tables from run_dpx.sh / run_default.sh results
(DIR/results.txt, last line per label, and the DIR/tok_<label>.json token files): per GSM8K question the default
mode, DPX+NPS2 single stream, placement, 1 and 3 drafts, speculation (7-chunk single stream / 3 drafts) and together
(default / 3 drafts); "all three" averages the time per token over all answer tokens; token identity between modes and
against each speculative run's reference. Then the 1K conversation's 7-chunk single stream and 3 drafts."""
import argparse
import json
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.environ.get("GSM_OUT", "/home/user/mk1/specg_latest"))
ap.add_argument("--tag", default="lg")
ap.add_argument("--ctag", default="lc")
A = ap.parse_args()
G = A.out.rstrip("/") + "/"
res = {}
for line in open(G + "results.txt"):
    if line.strip():
        res[line.split()[0]] = line.strip()


def num(lab, key):
    m = re.search(key + r"=([0-9.]+)", res.get(lab, ""))
    return float(m.group(1)) if m else None


def new(lab):
    m = re.search(r"new=(\d+)", res.get(lab, ""))
    return int(m.group(1)) if m else None


def acc(lab):
    m = re.search(r"accepted=(\d+)/(\d+)", res.get(lab, ""))
    return (int(m.group(1)), int(m.group(2))) if m else None


def toks(lab):
    try:
        return json.load(open(G + "tok_%s.json" % lab))["token_ids"]
    except Exception:
        return None


def same(a, b):
    ta, tb = toks(a), toks(b)
    if ta is None or tb is None:
        return "-"
    m = min(len(ta), len(tb))
    return "%d of %d" % (next((i for i in range(m) if ta[i] != tb[i]), m), m)


f4 = lambda x: "%.4f" % x if x else "-"
rx = lambda a, b: "%.3fx" % (a / b) if a and b else "-"
pa = lambda a: "%d of %d (%.1f%%)" % (a[0], a[1], 100.0 * a[0] / a[1]) if a else "-"
print("| question | answer | default | DPX+NPS2 | placement | 1 draft tok/pass | 1 draft accepted | 1 draft | "
      "7-chunk (answer) | 3 drafts tok/pass | 3 drafts accepted | 3 drafts | speculation | together | "
      "default==DPX | 1 draft==ns | 3 drafts==ns7 |")
print("|" + "---|" * 17)
tot = {k: [0.0, 0] for k in ("de", "ns", "k1", "n7", "k3")}
for q in ("gsm8k_1", "gsm8k_2", "gsm8k_3"):
    b = "%s_%s" % (A.tag, q)
    v = {"de": num(b + "_def", "ms_per_token_first_E"), "ns": num(b + "_ns", "ms_per_token_first_E"),
         "n7": num(b + "_ns7", "ms_per_token_first_E"), "k1": num(b + "_k1", "ms_per_token"),
         "k3": num(b + "_k3", "ms_per_token")}
    e, e7 = new(b + "_k1"), new(b + "_k3")
    for k, n in (("de", e), ("ns", e), ("k1", e), ("n7", e7), ("k3", e7)):
        if v[k] and n:
            tot[k][0] += v[k] * n
            tot[k][1] += n
    print("| %s | %s | %s | %s | %s | %s | %s | %s | %s (%s) | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        q, e, f4(v["de"]), f4(v["ns"]), rx(v["de"], v["ns"]), num(b + "_k1", "tokens_per_pass"), pa(acc(b + "_k1")),
        f4(v["k1"]), f4(v["n7"]), e7, num(b + "_k3", "tokens_per_pass"), pa(acc(b + "_k3")), f4(v["k3"]),
        rx(v["n7"], v["k3"]), rx(v["de"], v["k3"]), same(b + "_def", b + "_ns"), same(b + "_k1", b + "_ns"),
        same(b + "_k3", b + "_ns7")))
w = {k: (s / n if n else None) for k, (s, n) in tot.items()}
print("| all three | %d | %s | %s | %s | | | %s | %s (%d) | | | %s | %s | %s | | | |" % (
    tot["ns"][1], f4(w["de"]), f4(w["ns"]), rx(w["de"], w["ns"]), f4(w["k1"]), f4(w["n7"]), tot["n7"][1], f4(w["k3"]),
    rx(w["n7"], w["k3"]), rx(w["de"], w["k3"])))
print("\n1 draft over the DPX+NPS2 single stream: %s" % ", ".join(
    rx(num("%s_%s_ns" % (A.tag, q), "ms_per_token_first_E"), num("%s_%s_k1" % (A.tag, q), "ms_per_token"))
    for q in ("gsm8k_1", "gsm8k_2", "gsm8k_3")) + " (all three %s)" % rx(w["ns"], w["k1"]))
print("\n1K conversation (1,024 tokens):")
for lab in ("%s_1k_ns7" % A.ctag, "%s_1k_k3x" % A.ctag):
    print("  " + res.get(lab, lab + ": missing"))

