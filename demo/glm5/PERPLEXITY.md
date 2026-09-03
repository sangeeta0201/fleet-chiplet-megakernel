# GLM-5 perplexity gate

The correctness instrument GLM-5 was missing. `demo/glm5/demo.py` carried two
comments conceding "a WikiText-2 perplexity sweep is still owed"; this is it.

Run it with `demo/glm5/run_ppl.sh`; the gate is
`tests/ci-tests/test_glm5_perplexity.py`.

## Why perplexity and not tokens

`correctness_gate.py` measured five bs=1 runs of ONE build on ONE prompt
splitting into two attractor continuations that differ at token 0: the EP fold
and the MoE W2 atomics retire in arrival order, which flips argmax on near-ties,
and one flip cascades. A token-equality gate therefore fails about half the time
comparing a build to *itself*. The same reasoning is written into
`demo/gpt_oss/compare_tokens.py`, where correct 2-GPU runs agree for only 3 to
66 tokens.

Perplexity replaces the boolean with a continuous corpus-wide number, and takes
it under **teacher forcing**, which is the part that matters: `PPL_MODE=1` loads
the corpus as one long prompt and runs prefill only, and the megakernel does not
overwrite `tokens[]` inside the prompt, so every position conditions on the
reference prefix. One flipped argmax perturbs one position's NLL instead of
every position after it.

This mirrors how GPT-OSS is verified on this repo
(`tests/ci-tests/test_gpt_oss_perplexity.py`, which records mpk 33.48 against
torch 36.04 on a 512-token slice, ratio 0.929 against a 3.5 ceiling).

## What the gate checks

| Gate | Kind | What it catches |
|---|---|---|
| P1a sanity band | hard | "No signal at all": an unwritten logits row or a collapsed reduction, which lands near uniform (`ln(155136)` = 11.95 nats). Also a target off-by-one, which scores implausibly *low* |
| P1b quality target | reported | Absolute quality against a 5-15 expectation. Not blocking, because the cause is unattributed |
| P2 cross-rank agreement | hard | Wrong EP fold, missed barrier, stale gather slot. Every rank scores the full vocabulary independently and owes the same number |
| P3 Torch ratio | skipped | Needs a reference that does not fit on one GPU. Present only if `torch_ppl.json` exists |

Plus non-gated diagnostics that make a vacuous pass impossible: slice identity
(`corpus`/`corpus_desc`/`corpus_tokens`/`scored_positions` must match across
dumps), all-zero row count, pad-column max |logit|, and top-1 accuracy. Top-1 is
the sharper discriminator but the noisier one across runs of one build (measured
14.17% and 22.05% on two runs of the same slice), so it is printed, never gated.

## Implementation

The GPT-OSS LM head keeps logits in registers and never writes them, so it
needed a separate f32 sink. GLM-5's MXFP8 LM head already writes the full logit
row to HBM and the argmax reduces that same bf16 row, so the port is a
*redirect*, not a new write path: `ppl_sink=True` makes the kernel write row
`step+1` instead of row 0, and `argmax_in` is allocated
`[max_seq_length, vocab_shard]` instead of `[bs, vocab_shard]`. Capturing bf16
is not a precision compromise here -- it is exactly the width the kernel's own
argmax decided on.

`argmax_partial` still reads row 0, which is stale in PPL_MODE. That is
harmless: prefill-only never consumes a sampled token.

### The offset must be compile-time

`PPL_SINK` is a template parameter, not just a defaulted argument, and this cost
a measurement to learn. The kernel is `__noinline__`, so a runtime `logits_row`
that decode always passes as 0 is still materialized into a register and still
costs a 64-bit multiply-add on every output store:

| Form | decode min (ms/token) |
|---|---|
| before any change (n=3) | 8.738 (8.728 / 8.743 / 8.743) |
| `logits_row` as runtime defaulted arg (n=2) | **8.780** (8.761 / 8.800) |
| `PPL_SINK` as template parameter (n=3) | 8.731 (8.686 / 8.768) |

The serving path is back to baseline and the emitted argument list for
`ppl_sink=0` is unchanged.

## Measured, 2026-09-03

GLM-5 744B MXFP4, NP=4 EP on devices 4-7, `GLM_LMHEAD_TP=0`,
`--max-num-batched-tokens 1`, 128 tokens of WikiText-2 (127 scored positions).

### Same-input spread is 29%, so no tight gate on this number is honest

Three runs, one build, **bit-identical** token ids (verified: the `targets`
arrays match element for element):

| Run | input path | perplexity | mean NLL | entropy (nats) | top-1 |
|---|---|---|---|---|---|
| 1 | `wikitext2` | 207.8246 | 5.3367 | 4.485 | 18/127 |
| 2 | `wikitext2` | 209.6810 | 5.3456 | 4.464 | 28/127 |
| 3 | text file, same ids | 268.4578 | 5.5927 | 4.568 | 18/127 |

Runs 1 and 2 agreeing to 0.9% was a two-sample fluke; run 3 is 29% above run 1
on the same input. That is the order-nondeterministic reduction showing up in a
continuous metric, and it is the reason P1b is a reported target rather than a
gate. Top-1 moves even harder (18, 28, 18 of 127), which is why it is printed
and never gated.

All four ranks agreed to four decimal places within every run, so **P2 passes
exactly** -- the EP fold, the barriers and the gather slots are consistent. No
all-zero rows; pad-column max |logit| = 0.0000, so the sink covers every real
vocabulary column and no pad column leaks into the softmax.

### The quality number is POOR and currently UNATTRIBUTED

208-268 is not a plausible WikiText-2 perplexity for a 744B model; a healthy one
is roughly 5-15. The test does **not** set a ceiling that blesses it. Two
candidates, not separated:

* **MXFP4 damage.** Some inflation is expected and is precedented here:
  GPT-OSS's own MXFP4 decode measured a 92-100 spread against a Torch reference
  of ~36, i.e. ~2.7x. But 208 against an expected ~10 is ~20x, past that.
* **A kernel defect.** Not localized. An earlier reading of these dumps
  claimed quality degrades with context position; that does not survive the
  29% same-input spread above and is **retracted**. The apparent late-sequence
  NLL rise sat inside run-to-run noise and on proper nouns that are
  legitimately unpredictable ("B|oul|ter", "John De|ed", "Derek Jac|obi" --
  the slice is a Wikipedia biography).

Target alignment is *not* a candidate: comparing the dumped top-1 against the
target shifted by -3..+3, shift 0 wins in the early window (15.62% vs <=1.61%)
and in the late window (12.70% vs <=3.28%), so the row-to-target mapping is
right.

Separating quantization from defect requires the P3 reference arm. That is the
open item.

## The gate has already found a real bug

**A different 128-token input reproducibly produces near-uniform output.** Same
build, same config, only the corpus text differs:

| Input | perplexity | entropy | unique argmax / 127 | top-1 |
|---|---|---|---|---|
| WikiText-2 head | 208-268 | 4.46-4.57 | many | 18-28/127 |
| WikiText-2 offset by 64 tokens | **97566, 94512** | 9.065, 9.084 | 7, 6 | **0/127** |

Mean NLL 11.46 against uniform's `ln(155136)` = 11.95: the model emits
essentially no signal, from position 1 onward, not after a warm-up. It
reproduced on two independent runs, and 95000 is ~350x outside the 29%
same-input spread, so this is not noise. Six distinct argmax values across 127
positions (one token 103 times) is a collapse, not a quality regression.

Both inputs are valid English prose of the same length, both tokenize to ids
well inside the real vocabulary (max 98867 against 154880), and the offset input
was verified to round-trip token-identically. The distinguishing feature is
therefore the *content*, which in an EP MoE model points at routing -- different
tokens select different experts, so an input that routes onto a broken path
fails while its neighbour does not. Not yet confirmed.

This is the case for the gate: the build that does this passes cross-rank
identity, passes coherence, and produces clean-looking decode text.

## One reliability finding, independent of this change

**Runs at >= 256 corpus tokens wedge at launch.** 128 tokens completed in 5 of
5 attempts; 256 and 512 each wedged with all four ranks stopped at `launch bc: D
both grids enqueued` and no further output, killed by the 300 s silence
watchdog. This is the same symptom `run_correctness_suite.sh` already hit at
`--max-seq-length 512`, so it predates the sink, and the sink is not the cause:
at 256 tokens it is 80 MB.

## Scope: what this run does NOT cover

`GLM_LMHEAD_TP=0` is **not** the shipping decode config (which tiles the vocab
four ways). With the vocab sharded no single rank holds a normalizable logit
row, and a cross-rank logsumexp is not implemented, so the gate requires the
untiled head -- which is also what makes P2 meaningful. What is covered is the
78 layers, the attention, the MoE and the EP fold; what is not is the LM head's
vocab tiling.
