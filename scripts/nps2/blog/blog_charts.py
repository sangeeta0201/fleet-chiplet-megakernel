#!/usr/bin/env python3
"""blog_charts.py DATA.json OUTDIR: the blog's GPT-OSS-120B figures redrawn from new measurements.

Each section of DATA.json present draws one figure (same titles, axes and colours as the blog draft):
  phase    -> phase_speedup.png       per-phase speedup, DPX+NPS2 over the default mode (blog figure 4)
              phase_speedup_lm.png    the same per LM-head precision, with the LM-head phase
  swap     -> weights_same_vs_cross.png  same kernel, weights read from the same die or across dies (figure 3)
  served   -> gptoss_speed.png        GSM8K-128 tok/s, stock vLLM vs the megakernel (GPT-OSS panel of figure 1)
              gptoss_speedups.png     speedup per setup (GPT-OSS rows of figure 5)
  matched  -> precision_matched.png   the megakernel with the BF16 LM head vs vLLM with the W4A8 checkpoint
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RED, GRAY, DARK, PINK = "#cc102e", "#8a8d91", "#54585e", "#e8707c"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100})


def save(fig, out, name):
    p = os.path.join(out, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", p)


HEADNAME = {"bf": "BF16 LM head (the checkpoint's, as vLLM runs it)", "mx": "MXFP4 LM head (default build)"}


def phase(d, out):
    lead = d.get("lead", "mx")
    labels, kinds, vals = d["labels"], d["kinds"], d[lead]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    y = range(len(labels))[::-1]
    for yi, lab, k, v in zip(y, labels, kinds, vals):
        ax.barh(yi, v, height=0.62, color=RED if k == "sync" else GRAY)
        ax.text(v + 0.4, yi, "%d%% faster" % round(v), va="center")
    ax.set_yticks(list(y))
    ax.set_yticklabels(["%s (%s)" % (l, k) for l, k in zip(labels, kinds)])
    ax.set_xlim(0, max(30, max(vals) * 1.15))
    ax.set_xlabel("% faster with DPX+NPS2 (that phase only: default-mode time \u00f7 DPX+NPS2 time \u2212 1)")
    ax.set_title(d.get("title", "Cross-die waits speed up the most; the bandwidth-bound MoE stream the least"), pad=18)
    ax.text(0.5, 1.01, HEADNAME[lead], transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color=DARK)
    if d.get("note"):
        fig.text(0.01, -0.04, d["note"], fontsize=8.5, color=DARK)
    save(fig, out, "phase_speedup.png")

    labs = labels + [d["lm_label"]]
    kn = kinds + ["LM head"]
    mx, bf = d["mx"] + [d["lm"]["mx"]], d["bf"] + [d["lm"]["bf"]]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    n = len(labs)
    for i in range(n):
        yi = n - 1 - i
        ax.barh(yi + 0.18, bf[i], height=0.34, color=RED)
        ax.barh(yi - 0.18, mx[i], height=0.34, color=GRAY)
        ax.text(bf[i] + 0.4, yi + 0.18, "%.0f%%" % bf[i], va="center", fontsize=9.5)
        ax.text(mx[i] + 0.4, yi - 0.18, "%.0f%%" % mx[i], va="center", fontsize=9.5)
    ax.set_yticks(list(range(n))[::-1])
    ax.set_yticklabels(["%s (%s)" % (l, k) for l, k in zip(labs, kn)])
    ax.set_xlim(0, max(30, max(mx + bf) * 1.15))
    ax.set_xlabel("% faster with DPX+NPS2 (that phase only: default-mode time \u00f7 DPX+NPS2 time \u2212 1)")
    ax.set_title("Per-phase speedup with the BF16 and the MXFP4 LM head")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=RED), plt.Rectangle((0, 0), 1, 1, color=GRAY)],
              labels=["BF16 LM head (MPK_LM_BF16)", "MXFP4 LM head (default build)"], loc="lower right", frameon=False)
    if d.get("note"):
        fig.text(0.01, -0.03, d["note"], fontsize=8.5, color=DARK)
    save(fig, out, "phase_speedup_lm.png")


def swap(d, out):
    groups = [("no speculation", d["ns"]), ("EAGLE3", d["e3"])]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    w = 0.36
    for i, (name, (same, cross)) in enumerate(groups):
        ax.bar(i - w / 2, same, w, color=RED)
        ax.bar(i + w / 2, cross, w, color=GRAY)
        ax.text(i - w / 2, same + 0.015, "%.3f ms" % same, ha="center", va="bottom", fontsize=10)
        ax.text(i + w / 2, cross + 0.015, "%.3f ms  (+%d%%)" % (cross, round((cross / same - 1) * 100)), ha="center",
                va="bottom", fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylim(0, 2.15)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    ax.set_ylabel("ms per generated token (lower is better)")
    ax.yaxis.grid(True, color="#e6e6e6")
    ax.set_axisbelow(True)
    ax.set_title("Same kernel and output: weight reads same-die vs across dies", pad=18)
    if d.get("head"):
        ax.text(0.5, 1.01, d["head"], transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color=DARK)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=RED), plt.Rectangle((0, 0), 1, 1, color=GRAY)],
              labels=["weights read from the same die's HBM", "weights read across dies (other die's HBM)"],
              loc="upper left", frameon=False, fontsize=9.5, ncol=1)
    if d.get("note"):
        fig.text(0.01, -0.03, d["note"], fontsize=8.5, color=DARK)
    save(fig, out, "weights_same_vs_cross.png")


def served(d, out):
    st, mk = d["stock"], d["mk"]
    fast_st = max(st.values())
    fast_mk = max(mk.values())
    pairs = [("GSM8K, 128 prompts\nsame DFlash drafter", st["df"], mk["df"]),
             ("GSM8K, 128 prompts\nfastest setup tested of each*", fast_st, fast_mk)]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    w = 0.36
    for i, (name, a, b) in enumerate(pairs):
        ax.bar(i - w / 2, a, w, color=GRAY)
        ax.bar(i + w / 2, b, w, color=RED)
        ax.text(i - w / 2, a + 15, "{:,}".format(a), ha="center", va="bottom", fontsize=9)
        ax.text(i + w / 2, b + 15, "{:,}".format(b), ha="center", va="bottom", fontsize=9)
        ax.text(i, b + 120, "%.2fx" % (b / a), ha="center", color=RED, fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([p[0] for p in pairs], fontsize=8.5)
    ax.set_ylim(0, max(1800, fast_mk * 1.25))
    ax.set_ylabel("output tokens / s, one request")
    ax.yaxis.grid(True, color="#e6e6e6")
    ax.set_axisbelow(True)
    ax.set_title("GPT-OSS-120B\n", fontweight="bold")
    ax.text(0.5, 1.02, d.get("subtitle", "vLLM 0.31.0 on one MI355X: stock in the default mode,\nthe megakernel "
                                          "plugin in DPX+NPS2" + (", %s" % d["head"] if d.get("head") else "")),
            transform=ax.transAxes, ha="center", fontsize=8, color=DARK)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GRAY), plt.Rectangle((0, 0), 1, 1, color=RED)],
              labels=["stock vLLM", "megakernel"], loc="upper left", frameon=False, fontsize=9)
    if d.get("note"):
        fig.text(0.01, -0.06, d["note"], fontsize=8, color=DARK)
    save(fig, out, "gptoss_speed.png")

    rows = [("no speculation (vLLM)", st["ns"], mk["ns"]), ("same DFlash drafter (vLLM)", st["df"], mk["df"]),
            ("EAGLE3 on both (vLLM)", st["e3"], mk["e3"]), ("fastest setup of each (vLLM)", fast_st, fast_mk)]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for i, (name, a, b) in enumerate(rows):
        yi = len(rows) - 1 - i
        s = b / a
        ax.barh(yi, s, height=0.62, color=DARK)
        ax.text(s + 0.05, yi, "%.2fx   {:,} \u2192 {:,} tok/s".format(a, b) % s, va="center", fontsize=9.5)
    ax.axvline(1, color="black", lw=1)
    ax.set_yticks(list(range(len(rows)))[::-1])
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 5.2)
    ax.set_xlabel("speedup over the baseline (same engine; GPT-OSS: stock vLLM)")
    ax.set_title(d.get("rows_title", "GPT-OSS-120B on vLLM 0.31.0: the megakernel plugin vs stock vLLM" +
                       (" (%s)" % d["head"] if d.get("head") else "")))
    if d.get("note"):
        fig.text(0.01, -0.06, d["note"], fontsize=8, color=DARK)
    save(fig, out, "gptoss_speedups.png")


def matched(d, out):
    names = [("no speculation", "ns"), ("EAGLE3", "e3"), ("DFlash", "df")]
    a, b = d["w4a8"], d["bf"]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    w = 0.36
    for i, (name, k) in enumerate(names):
        ax.bar(i - w / 2, a[k], w, color=GRAY)
        ax.bar(i + w / 2, b[k], w, color=RED)
        ax.text(i - w / 2, a[k] + 15, "{:,}".format(a[k]), ha="center", va="bottom", fontsize=9)
        ax.text(i + w / 2, b[k] + 15, "{:,}".format(b[k]), ha="center", va="bottom", fontsize=9)
        ax.text(i, b[k] + 110, "%.2fx" % (b[k] / a[k]), ha="center", color=RED, fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n[0] for n in names])
    ax.set_ylim(0, max(b.values()) * 1.3)
    ax.set_ylabel("output tokens / s, one request (GSM8K-128)")
    ax.yaxis.grid(True, color="#e6e6e6")
    ax.set_axisbelow(True)
    ax.set_title("Matched precision: MXFP4 x FP8 GEMMs, BF16 LM head and router, BF16 KV cache", fontsize=11)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GRAY), plt.Rectangle((0, 0), 1, 1, color=RED)],
              labels=[d.get("a_label", "vLLM 0.31.0, AMD W4A8 checkpoint"), d.get("b_label", "megakernel, BF16 LM head")],
              loc="upper left", frameon=False, fontsize=9)
    if d.get("note"):
        fig.text(0.01, -0.04, d["note"], fontsize=8, color=DARK)
    save(fig, out, "precision_matched.png")


data = json.load(open(sys.argv[1]))
out = sys.argv[2]
os.makedirs(out, exist_ok=True)
for key, fn in (("phase", phase), ("swap", swap), ("served", served), ("matched", matched)):
    if key in data:
        fn(data[key], out)

