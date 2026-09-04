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
# Same-input spread is 16.24% over FIVE identical reps of one build
# (5.4502-6.3355, demo/glm5/run_ppl_repro.sh), down from 29% before the prefix
# fix but well above the 4.5% an earlier n=3 sample suggested -- that reading
# is retracted. 127/128 positions differ across reps, the first at position 1,
# and 40/128 flip their argmax, so the ceiling has to clear 16% of headroom and
# token equality is not a legal gate.
#
# 60 is ~2x the loosest legitimate window measured and ~9x the canonical one.
# It is deliberately not tight to 6: perplexity legitimately depends on where
# the corpus slice STARTS, which the fixed window does not normalise away
# (5.70 for a document head, 30.28 for a mid-sentence window), and run-to-run
# nondeterminism adds 16% on top -- five identical reps of one build measured
# 5.4502-6.3355, see demo/glm5/run_ppl_repro.sh. What it does catch is the
# regression class that matters: dropping the prefix scored 208-268, and a
# near-uniform collapse scores ~95000. A tail that collapses partway is caught
# by the per-block sanity check instead, which is length-independent.
PPL_QUALITY_MAX = float(os.environ.get("GLM5_PPL_QUALITY_MAX", "60"))
# A floor as well as a ceiling. Perplexity below this on WikiText-2 means the
# scoring slice is wrong -- most likely the targets are offset so the model is
# being graded against the token it was just given.
PPL_MIN = float(os.environ.get("GLM5_PPL_MIN", "1.5"))
PPL_RATIO_MAX = float(os.environ.get("GLM5_PPL_RATIO_MAX", "3.5"))
MIN_SCORED = int(os.environ.get("GLM5_PPL_MIN_SCORED", "64"))
RANK_RTOL = float(os.environ.get("GLM5_PPL_RANK_RTOL", "0.02"))
# Widest contiguous run of zero logit columns to tolerate. Zero columns are
# expected as isolated singletons (GPT-OSS measured 763 of 32767x201088 =
# 1.2e-7, first at row 88 column 97897) -- logits that genuinely round to 0.0f.
# A column range the kernel skipped is thousands wide, so this separates them
# by shape rather than by count.
ZERO_RUN_MAX = int(os.environ.get("GLM5_PPL_ZERO_RUN_MAX", "64"))
# Quality is gated on a FIXED WINDOW of scored positions, not on the whole
# slice, because full-slice perplexity moves with which text the slice includes
# and is therefore not comparable across lengths. Measured on one build:
#
#   corpus tokens   full-slice ppl   ppl over the same first 128 positions
#            128            5.6587                                  5.6587
#            256           15.4687                                  5.3406
#            512          435.7983                                  6.6678
#
# A single absolute ceiling on the full-slice column would either pass
# everything or fail a longer run for including harder text. The windowed
# column is the one a ceiling can be set against. demo/glm5/run_ppl_sweep.sh
# and summarize_glm5_ppl_sweep.py report both.
GATE_POSITIONS = int(os.environ.get("GLM5_PPL_GATE_POSITIONS", "128"))
# ...and because a leading window cannot see a tail that collapses, the sanity
# ceiling is ALSO applied per block of consecutive positions. The 512-token run
# above scored mean NLL 9.80 with 0/64 top-1 over its last 64 positions --
# near-uniform (ln(155136) = 11.95) -- while its first 128 stayed healthy.
SANITY_BLOCK = int(os.environ.get("GLM5_PPL_SANITY_BLOCK", "64"))


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


def _windowed_ppl(d, n_positions):
    """Perplexity over the first `n_positions` scored positions.

    Returns (ppl, n_used). Falls back to the whole slice when the dump is
    shorter, so a short run is still gated -- just less comparably.
    """
    pp = d.get("per_position_nll") or []
    if not pp:
        return d.get("perplexity"), d.get("scored_positions") or 0
    used = pp[:n_positions]
    return math.exp(sum(used) / len(used)), len(used)


def _worst_block(d, block):
    """(ppl, start, top1_accuracy) for the worst block of consecutive positions.

    A leading window cannot see a tail that stops carrying signal, and the mean
    over a long slice dilutes it. Scanning blocks finds it.
    """
    pp = d.get("per_position_nll") or []
    if len(pp) < block:
        return None
    top1, tgts = d.get("top1") or [], d.get("targets") or []
    worst = None
    for lo in range(0, len(pp) - block + 1, block):
        seg = pp[lo:lo + block]
        ppl = math.exp(sum(seg) / block)
        if worst is None or ppl > worst[0]:
            acc = None
            if top1 and tgts:
                acc = sum(1 for i in range(lo, lo + block)
                          if int(top1[i]) == int(tgts[i])) / block
            worst = (ppl, lo, acc)
    return worst


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
        diag = d.get("diagnostics", {})
        nz = diag.get("all_zero_rows", 0)
        if nz:
            pytest.fail(
                f"{path}: {nz} of {n_scored} scored rows are all-zero -- the "
                f"LM head never wrote them, so the perplexity is measuring "
                f"a uniform distribution over part of the corpus."
            )
        # ── Sink self-checks, ported from the GPT-OSS instrument ─────────
        # These exist because a partially-written sink still produces a
        # plausible-looking perplexity. On GPT-OSS a wrong input_map wrote
        # alternating 25152-column stripes and "scored 64.5 against a Torch
        # 8.15 while looking like a plausible accuracy result", which the
        # per-row all-zero test above cannot see.
        run = diag.get("widest_zero_run")
        if run is not None and run > ZERO_RUN_MAX:
            pytest.fail(
                f"{path}: widest contiguous run of zero logit columns is "
                f"{run} (> {ZERO_RUN_MAX}). Isolated zeros are expected -- "
                f"logits that genuinely round to 0.0f -- but a run this wide "
                f"is a column range the LM head never wrote. Check the sink's "
                f"input_map: a non-negative map partitions the vocab dim "
                f"across grid_dim.x and pre-shifts each block's base pointer, "
                f"which double-counts the offset."
            )
        row0 = diag.get("row0_max_abs")
        if row0 is not None and row0 != 0.0:
            pytest.fail(
                f"{path}: sink row 0 has max |logit| {row0}, expected 0. In "
                f"PPL_MODE the LM head writes row step+1, so row 0 must stay "
                f"untouched; a written row 0 means ppl_sink did not take "
                f"effect on every store."
            )
        dup = diag.get("duplicate_adjacent_rows")
        if dup:
            pytest.fail(
                f"{path}: {dup} pairs of adjacent scored rows are bit-"
                f"identical. Distinct positions owe distinct distributions, "
                f"so the sink is resolving multiple positions to one row -- "
                f"which still yields a plausible perplexity."
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

    # ── P1a2: per-block sanity (hard) ────────────────────────────────────
    # Catches a slice that starts healthy and stops carrying signal partway.
    for path, d in dumps:
        wb = _worst_block(d, SANITY_BLOCK)
        if wb is None:
            continue
        bppl, blo, bacc = wb
        acc_s = f"{bacc:.1%}" if bacc is not None else "n/a"
        print(f"[glm5 perplexity] rank {d.get('rank')}: worst "
              f"{SANITY_BLOCK}-position block starts at {blo}, "
              f"ppl={bppl:.2f}, top-1={acc_s}")
        if bppl > PPL_SANITY_MAX:
            pytest.fail(
                f"{path}: positions {blo}..{blo + SANITY_BLOCK} score "
                f"ppl {bppl:.2f} (top-1 {acc_s}), above the sanity ceiling "
                f"{PPL_SANITY_MAX}, even though the slice as a whole scores "
                f"{d['perplexity']:.2f}. The model stopped carrying signal "
                f"partway through this sequence. Hard text does not do this: "
                f"check whether quality depends on sequence length, which is "
                f"measurable with demo/glm5/run_ppl_sweep.sh."
            )

    # ── P1b: quality ceiling on a FIXED WINDOW (hard) ────────────────────
    # Windowed, not full-slice: see the GATE_POSITIONS note above.
    worst_q, worst_n, worst_full = 0.0, 0, 0.0
    for _, d in dumps:
        wppl, used = _windowed_ppl(d, GATE_POSITIONS)
        if wppl is not None and wppl > worst_q:
            worst_q, worst_n, worst_full = wppl, used, d["perplexity"]
    print(f"[glm5 perplexity] quality gate: ppl over the first {worst_n} "
          f"scored positions = {worst_q:.4f} (ceiling {PPL_QUALITY_MAX}); "
          f"full-slice {worst_full:.4f} is reported, not gated")
    if worst_q > PPL_QUALITY_MAX:
        pytest.fail(
            f"perplexity {worst_q:.4f} over the first {worst_n} scored "
            f"positions exceeds the quality ceiling {PPL_QUALITY_MAX}. The "
            f"canonical value for this window is 5.3-6.7. Check FIRST that "
            f"the prompt carries the '[gMASK]<sop>' prefix the model is "
            f"trained on -- dropping it measured 208-268 here, and it is "
            f"recorded in this dump's corpus_desc "
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
