"""GLM-5 perplexity: the megakernel (MPK) against a corpus, and rank vs rank.

Companion to test_glm5_inference_output.py, and the GLM-5 counterpart of
test_gpt_oss_perplexity.py.

WHY THIS EXISTS. demo/glm5/correctness_gate.py measured five bs=1 runs of one
build on one prompt splitting into TWO attractor continuations that differ at
token 0: the EP fold and the MoE W2 atomics retire in arrival order, which
flips argmax on near-ties, and one flip cascades. So a token-equality gate
fails about half the time comparing a build to ITSELF and cannot separate a bug
from noise. The same reasoning is why demo/gpt_oss/compare_tokens.py rejects
exact match, where correct 2-GPU runs agree for only 3 to 66 tokens.

Perplexity replaces that boolean with a continuous corpus-wide number, and
measures it under teacher forcing. PPL_MODE=1 loads the corpus as one long
prompt and runs prefill only; the megakernel does not overwrite tokens[] inside
the prompt (prepare_next_batch guards the writeback on step+1 >= prompt_len), so
every position conditions on the reference prefix. Teacher forcing is the part
that matters: it removes the cascade, so a single flipped argmax perturbs one
position's NLL instead of every position after it.

TWO GATES, AND WHAT EACH ONE CATCHES

  P1a SANITY BAND (hard). Catches "the model emitted no signal": a logits row
      the kernel never wrote, or a reduction that collapsed, lands near uniform
      (ln(vocab) = 11.95 nats here). This gate has already earned its keep --
      it caught a reproducible failure at ppl ~95000 with 0/127 top-1 on a
      build whose decode output looks fine and which a token or coherence gate
      passes. That defect turned out to be in the HARNESS, not the kernel: the
      corpus was fed without the "[gMASK]<sop>" prefix GLM-5 is trained to see
      (see demo/glm5/PERPLEXITY.md). It runs without a reference
      implementation, which is why it ships first: GLM-5 744B at 4 bits is
      ~372 GB against 288 GB of HBM on one MI350, so the single-GPU Torch arm
      that test_gpt_oss_perplexity.py compares against does not fit here.

  P1b QUALITY CEILING (hard). With the prefix supplied, the canonical slice
      measures 5.64-5.90 and 256 tokens of wikitext2 measures 15.64, so this
      is now a real gate rather than a reported target. It is set with
      headroom over the loosest legitimate slice measured (30.28 for a
      mid-document window, which has almost no context to anchor on), and it
      is what would catch a regression back to the unprefixed 208-268.

  P2  CROSS-RANK AGREEMENT (hard, and specific to this model). PPL_MODE
      requires GLM_LMHEAD_TP=0, so every EP rank scores the FULL vocabulary
      from its own logit row. All ranks decode the same corpus and owe the same
      distribution, so their perplexities must agree to a tight relative
      tolerance. This is the continuous form of correctness_gate.py's G1 and it
      catches the defect class that actually shows up on this branch -- a wrong
      EP fold, a missed barrier, a rank reading a stale gather slot -- which an
      absolute ceiling alone would pass.

  P3  TORCH RATIO (skipped unless a reference dump exists). Populate
      torch_ppl.json and this asserts MPK/Torch <= GLM5_PPL_RATIO_MAX, the same
      way the GPT-OSS test does. Absent, it is reported as skipped rather than
      silently passing.

Top-1 agreement against the target is computed and printed but NOT gated: it is
the sharper discriminator of the two, and also the noisier one across runs of
one build, so gating on it would flake. Read it when P1 fires.

Produce the input first (from repo root):
    PPL_MODE=1 GLM_LMHEAD_TP=0 ./demo/glm5/run_ppl.sh
or by hand, per rank, with --ppl-out outputs/glm5/mpk_ppl.json.

Tunables (env):
    GLM5_OUTPUT_DIR        dir holding the json dumps  (default outputs/glm5)
    GLM5_PPL_SANITY_MAX    near-uniform ceiling, "no signal at all"
    GLM5_PPL_QUALITY_MAX   absolute MPK perplexity ceiling
    GLM5_PPL_RATIO_MAX     max MPK/Torch ratio, if a torch dump exists
    GLM5_PPL_MIN_SCORED    min scored positions to accept a run
    GLM5_PPL_RANK_RTOL     max relative spread across ranks
"""
import glob
import json
import math
import os
import pytest

DEFAULT_OUTPUT_DIR = os.environ.get(
    "GLM5_OUTPUT_DIR", os.path.join("outputs", "glm5")
)

# ── The two ceilings, and why there are two ─────────────────────────────────
# SANITY separates "the model is mediocre on this slice" from "the model
# emitted no signal at all" -- the defect class an unwritten logits row, a dead
# expert, or a malformed prompt produces. It fires near uniform, ln(155136) =
# 11.95 nats. Measured example it catches: 94512/97566 with 6-7 unique argmax
# across 127 positions and 0/127 top-1, mean NLL 11.46.
PPL_SANITY_MAX = float(os.environ.get("GLM5_PPL_SANITY_MAX", "1000"))
# QUALITY, measured 2026-09-03 with the "[gMASK]<sop>" prefix in place:
#
#   128 tok, wikitext2 doc head   5.6970 / 5.8966 / 5.6403  (n=3, same ids)
#   256 tok, wikitext2           15.6414
#   128 tok, mid-document window 30.2805
#
# Same-input spread is 4.5% (n=3), not the 29% measured before the prefix fix:
# in distribution the model's rows are sharp (entropy 1.23-1.45 nats against
# 4.46-4.57 without the prefix), so the order-nondeterministic EP/MoE reduction
# has far fewer near-ties to flip. That is what makes a hard ceiling honest
# here.
#
# 60 is ~2x the loosest legitimate slice measured and ~4x the canonical one.
# It is deliberately not tight to 6: perplexity legitimately depends on how
# much context the slice gives the model (5.70 for a document head, 30.28 for a
# mid-sentence window). What it does catch is the regression class that matters
# -- dropping the prefix scored 208-268, and a near-uniform collapse scores
# ~95000.
PPL_QUALITY_MAX = float(os.environ.get("GLM5_PPL_QUALITY_MAX", "60"))
# A floor as well as a ceiling. Perplexity below this on WikiText-2 means the
# scoring slice is wrong -- most likely the targets are offset so the model is
# being graded against the token it was just given.
PPL_MIN = float(os.environ.get("GLM5_PPL_MIN", "1.5"))
PPL_RATIO_MAX = float(os.environ.get("GLM5_PPL_RATIO_MAX", "3.5"))
MIN_SCORED = int(os.environ.get("GLM5_PPL_MIN_SCORED", "64"))
RANK_RTOL = float(os.environ.get("GLM5_PPL_RANK_RTOL", "0.02"))


def _load(path):
    with open(path) as f:
        data = json.load(f)
    ppl = data.get("perplexity")
    if not isinstance(ppl, (int, float)):
        pytest.fail(f"'perplexity' missing or not a number in {path}")
    if not math.isfinite(ppl):
        pytest.fail(
            f"Non-finite perplexity ({ppl}) in {path} -- the LM head produced "
            f"inf/nan, which is a kernel bug, not a quality result."
        )
    if ppl < 1.0:
        pytest.fail(
            f"Perplexity {ppl} < 1 in {path}, which is impossible for a "
            f"cross-entropy over a real distribution. The scoring slice or "
            f"the vocab truncation is wrong."
        )
    return data


def _mpk_dumps():
    pat = os.path.join(DEFAULT_OUTPUT_DIR, "mpk_ppl*.json")
    files = sorted(glob.glob(pat))
    if not files:
        pytest.fail(
            f"No MPK perplexity dumps matched {pat}. Run "
            f"demo/glm5/run_ppl.sh (PPL_MODE=1, GLM_LMHEAD_TP=0) first."
        )
    return [(f, _load(f)) for f in files]


def _top1_accuracy(d):
    top1, targets = d.get("top1"), d.get("targets")
    if not top1 or not targets:
        return None
    n = min(len(top1), len(targets))
    return sum(1 for i in range(n) if int(top1[i]) == int(targets[i])) / n


def test_glm5_mpk_perplexity():
    dumps = _mpk_dumps()

    # Every dump must have scored the same text, or the numbers below are not
    # comparable and a silent corpus/slice mismatch reads as a quality delta.
    keys = ("corpus", "corpus_desc", "corpus_tokens", "scored_positions")
    ref_path, ref = dumps[0]
    for path, d in dumps[1:]:
        for k in keys:
            if d.get(k) != ref.get(k):
                pytest.fail(
                    f"Dumps scored different slices: {k} is {ref.get(k)!r} in "
                    f"{ref_path} vs {d.get(k)!r} in {path}. Re-run every rank "
                    f"with the same --ppl-corpus and --ppl-max-tokens."
                )

    n_scored = ref.get("scored_positions", 0)
    if n_scored < MIN_SCORED:
        pytest.fail(
            f"Only {n_scored} scored positions (require >= {MIN_SCORED}); too "
            f"few for the mean NLL to be stable. Raise --ppl-max-tokens."
        )

    # An unwritten logits row is all zeros, i.e. uniform over the vocabulary.
    # The demo already warns; fail on it here, because it makes the perplexity
    # meaningless rather than merely worse.
    for path, d in dumps:
        nz = d.get("diagnostics", {}).get("all_zero_rows", 0)
        if nz:
            pytest.fail(
                f"{path}: {nz} of {n_scored} scored rows are all-zero -- the "
                f"LM head never wrote them, so the perplexity is measuring "
                f"a uniform distribution over part of the corpus."
            )

    # The prompt prefix is the single biggest lever on this number -- omitting
    # it measured 208-268 where the prefixed run measures 5.7 -- so surface it
    # before the ceilings rather than leaving it to the failure message.
    prefix = ref.get("diagnostics", {}).get("prefix")
    if not prefix:
        print(
            "[glm5 perplexity] WARNING: this dump records NO prompt prefix. "
            "GLM-5 is trained with '[gMASK]<sop>' at the head of every "
            "sequence, and scoring without it is an out-of-distribution "
            "measurement, not a kernel result."
        )

    for path, d in dumps:
        acc = _top1_accuracy(d)
        acc_str = f"{acc:.2%}" if acc is not None else "n/a"
        print(
            f"[glm5 perplexity] rank {d.get('rank')}: "
            f"ppl={d['perplexity']:.4f} (sanity <= {PPL_SANITY_MAX}, "
            f"quality <= {PPL_QUALITY_MAX}) "
            f"mean_nll={d.get('mean_nll'):.4f} "
            f"entropy={d.get('mean_entropy')} "
            f"top-1={acc_str} over {n_scored} positions of {d.get('corpus')}"
        )

    # ── P1a: sanity band (hard) ──────────────────────────────────────────
    for path, d in dumps:
        ppl = d["perplexity"]
        acc = _top1_accuracy(d)
        if ppl > PPL_SANITY_MAX:
            pytest.fail(
                f"{path}: perplexity {ppl:.4f} exceeds the SANITY ceiling "
                f"{PPL_SANITY_MAX} over {n_scored} positions (top-1 "
                f"{acc if acc is not None else float('nan'):.2%}, entropy "
                f"{d.get('mean_entropy')}). This is not a quality result, it "
                f"is near-uniform output: the model emitted essentially no "
                f"signal. Check the all-zero and pad-column diagnostics, then "
                f"whether every EP rank contributed an expert."
            )
        if ppl < PPL_MIN:
            pytest.fail(
                f"{path}: perplexity {ppl:.4f} is below {PPL_MIN}, which is "
                f"implausibly good on WikiText-2. The most likely cause is a "
                f"target/row off-by-one that grades the model against a token "
                f"it was already given."
            )

    # ── P1b: quality ceiling (hard) ──────────────────────────────────────
    worst_q = max(d["perplexity"] for _, d in dumps)
    if worst_q > PPL_QUALITY_MAX:
        pytest.fail(
            f"perplexity {worst_q:.4f} exceeds the quality ceiling "
            f"{PPL_QUALITY_MAX} over {n_scored} positions, on a slice whose "
            f"canonical value is 5.6-5.9 (128 tok) or 15.6 (256 tok). "
            f"Check FIRST that the prompt carries the '[gMASK]<sop>' prefix "
            f"the model is trained on -- dropping it measured 208-268 here, "
            f"and it is recorded in this dump's corpus_desc "
            f"({ref.get('corpus_desc')!r}). If the prefix is present, this is "
            f"a real quality regression in the 78 layers, the attention, the "
            f"MoE or the EP fold."
        )

    # ── P2: cross-rank agreement ─────────────────────────────────────────
    # Under GLM_LMHEAD_TP=0 every rank scores the full vocabulary from its own
    # logit row, so these are independent measurements of one quantity.
    if len(dumps) > 1:
        ppls = [d["perplexity"] for _, d in dumps]
        lo, hi = min(ppls), max(ppls)
        spread = (hi - lo) / lo
        print(f"[glm5 perplexity] cross-rank: {len(ppls)} ranks, "
              f"min={lo:.4f} max={hi:.4f} relative spread={spread:.4%} "
              f"(tol {RANK_RTOL:.2%})")
        if spread > RANK_RTOL:
            pytest.fail(
                f"Ranks disagree by {spread:.4%} (> {RANK_RTOL:.2%}): "
                + ", ".join(f"rank {d.get('rank')}={d['perplexity']:.4f}"
                            for _, d in dumps)
                + ". Every rank decodes the same corpus and owes the same "
                  "distribution, so this is an EP defect -- a wrong fold, a "
                  "missed barrier, or a stale gather slot -- not quantization "
                  "noise."
            )

    # ── P3: Torch ratio, only if a reference dump exists ─────────────────
    torch_path = os.path.join(DEFAULT_OUTPUT_DIR, "torch_ppl.json")
    if not os.path.exists(torch_path):
        print(
            f"[glm5 perplexity] no Torch reference at {torch_path}: ratio "
            f"gate SKIPPED. GLM-5 744B does not fit on one GPU in the Torch "
            f"path, so this arm needs a sharded reference."
        )
        return
    td = _load(torch_path)
    for k in keys:
        if td.get(k) != ref.get(k):
            pytest.fail(
                f"Torch dump scored a different slice: {k} is {td.get(k)!r} "
                f"vs {ref.get(k)!r} for MPK."
            )
    worst = max(d["perplexity"] for _, d in dumps)
    ratio = worst / td["perplexity"]
    print(f"[glm5 perplexity] torch_ppl={td['perplexity']:.4f} "
          f"worst mpk={worst:.4f} ratio={ratio:.3f} (max {PPL_RATIO_MAX})")
    if ratio > PPL_RATIO_MAX:
        pytest.fail(
            f"MPK perplexity {worst:.4f} is {ratio:.3f}x the Torch reference "
            f"{td['perplexity']:.4f} (max {PPL_RATIO_MAX}) over {n_scored} "
            f"positions. The absolute band passed, so the corpus may just be "
            f"hard -- but the two paths disagree by more than MXFP4 "
            f"run-to-run drift explains."
        )
