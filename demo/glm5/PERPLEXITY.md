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

## The prompt prefix is load-bearing

**GLM-5 sees `[gMASK]<sop>` at the head of every sequence it was trained on.**
The serving path gets it for free from `apply_chat_template`; the first version
of this harness scored raw text without it, and that single omission dominated
every number the gate produced.

| Corpus (128 tokens) | no prefix | with `[gMASK]<sop>` |
|---|---|---|
| WikiText-2 document head | 268.46 | **5.70** |
| WikiText-2 offset by 64 tokens | 94512 | **30.28** |

Two tokens of context moved one slice by 47x and the other by **3120x**. Note
that `add_special_tokens=True` is a **no-op** for this tokenizer -- it adds
nothing -- so the prefix has to be supplied explicitly. `PPL_PREFIX=''` scores
without it, and the dump records the prefix in `corpus_desc` so a prefixed and
an unprefixed run can never be silently compared.

Why the effect is this large: position 0 is the attention sink every later query
attends to. With no anchor there, an ordinary content token occupies the sink
slot and distorts the whole sequence. The measured signature was
position-independence -- **6 unique argmax values across 127 positions** (one
token 101 times), i.e. a hidden state that had stopped depending on the input,
with a mean NLL of 11.46 against uniform's `ln(155136)` = 11.95.

The prefix positions are context, not predictions the model owes, so they are
excluded from scoring: row `r` is graded against `tokens[r]`, and grading starts
at row 2. `--ppl-max-tokens` counts corpus tokens; the prefix is added on top.

## What the gate checks

| Gate | Kind | What it catches |
|---|---|---|
| P1a sanity band | hard | "No signal at all": an unwritten logits row, a collapsed reduction, or a malformed prompt, all of which land near uniform (11.95 nats). Also a target off-by-one, which scores implausibly *low* |
| P1a2 per-block sanity | hard | The same ceiling applied to every block of 64 consecutive positions, so a slice that starts healthy and stops carrying signal partway cannot average its way to a pass. This is what catches the length-dependent collapse below |
| P1b quality ceiling | hard | Absolute quality, 60, over a **fixed window** of the first 128 scored positions rather than the whole slice. Full-slice perplexity moves with which text a slice includes (5.66 / 15.47 / 435.80 at 128 / 256 / 512 tokens, against 5.66 / 5.34 / 6.67 over the same first 128 positions), so a single ceiling on it is not length-independent. Catches a regression back to the unprefixed 208-268 |
| P2 cross-rank agreement | hard | Wrong EP fold, missed barrier, stale gather slot. Every rank scores the full vocabulary independently and owes the same number |
| P3 Torch ratio | skipped | Needs a reference that does not fit on one GPU. Present only if `torch_ppl.json` exists |

Plus non-gated diagnostics that make a vacuous pass impossible: slice identity
(`corpus`/`corpus_desc`/`corpus_tokens`/`scored_positions` must match across
dumps), a loud warning when a dump records no prompt prefix, all-zero row count,
pad-column max |logit|, and top-1 accuracy. Top-1 is the sharper discriminator
but the noisier one across runs of one build, so it is printed, never gated.

## Measured, 2026-09-03

GLM-5 744B MXFP4, NP=4 EP on devices 4-7, `GLM_LMHEAD_TP=0`,
`--max-num-batched-tokens 1`, prefix `[gMASK]<sop>`.

| Slice | scored | perplexity | mean NLL | entropy (nats) | top-1 |
|---|---|---|---|---|---|
| wikitext2 doc head, 128 tok | 128 | 5.6970 / 5.8966 / 5.6403 | 1.73-1.77 | 1.23-1.45 | 64.8% |
| wikitext2, 256 tok | 256 | 15.6414 | 2.7499 | 2.277 | -- |
| mid-document window, 128 tok | 128 | 30.2805 | 3.4105 | 2.615 | 40.6% |

**5.70 is inside the 5-15 band a healthy 744B model belongs in.** The
implication is that MXFP4 quantization is *not* visibly damaging quality here,
and no kernel defect is indicated -- the earlier 20x gap was entirely the
missing prefix.

Perplexity legitimately rises as a slice gives the model less to anchor on: a
document head (5.70) is easier than a 256-token span (15.64), which is easier
than a window starting mid-sentence (30.28). Compare like slices only.

### Same-input spread is 16%, not the 4.5% an n=3 sample suggested

Five identical reps of one build, bit-identical token ids
(`demo/glm5/run_ppl_repro.sh`, 128 tok):

| rep | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| ppl | 6.3355 | 6.2219 | 6.2184 | 5.4502 | 5.5720 |

min 5.4502, max 6.3355, **spread 16.24%**. An earlier n=3 sample
(5.6970 / 5.8966 / 5.6403) read 4.5% and that number is retracted: it was
three draws from this same distribution, not a tighter one. Before the prefix
fix the measurement spread **29%** (207.82 / 209.68 / 268.46 on ids verified
identical element for element), so the prefix did narrow it -- just to 16%,
not to 4.5%.

Where the nondeterminism enters: **127 of 128 positions** differ across the
five reps, the first at position 1, and **40 of 128 positions flip their
argmax**. Divergence at position 1 means this is not accumulation over the
sequence; it is present immediately, consistent with the EP fold and the MoE
W2 f32 atomics retiring in arrival order. The 40 argmax flips are why token
equality is not a legal gate on this model.

The nondeterminism did not go away -- the reduction still retires in arrival
order. What changed is that an in-distribution model produces sharp rows
(entropy 1.23-1.45 nats, against 4.46-4.57 unprefixed), so there are far fewer
near-ties for a reordered sum to flip. That is what makes a hard quality ceiling
honest here where it would not have been before.

All four ranks agree to four decimal places within every run, so **P2 passes
exactly** -- the EP fold, the barriers and the gather slots are consistent. No
all-zero rows; pad-column max |logit| = 0.0000, so the sink covers every real
vocabulary column and no pad column leaks into the softmax.

## RESOLVED: quality collapse with sequence length

A 512-token run scores **435.80** against 5.66 at 128 tokens. That is not the
corpus getting harder, and it is not the run-to-run spread. Per 64 positions,
mean NLL over four independent 512-token runs:

| positions | 0-64 | 64-128 | 128-192 | 192-256 | 256-320 | 320-384 | 384-448 | 448-512 |
|---|---|---|---|---|---|---|---|---|
| mean NLL | 1.51 | 2.15 | 4.06 | 3.65 | 7.36 | 9.36 | 9.79 | 9.75 |

Fixed 2026-09-05. `GLM_DENSE_MLP_TP=1` interleaved the full gate/up matrices
into eight global XCD slabs and then sliced those slabs by rank. Each local
`silu_mul(grid.x=8)` interpreted its two retained global slabs as eight local
slabs and paired the wrong gate/up values. Slice gate and up by rank first,
then interleave each local pair into eight slabs.

After the fix, on NP=4 devices 4-7:

| corpus tokens | full PPL | first-128 PPL | worst 64-position block |
|---|---:|---:|---:|
| 512 | **2.6020** | **2.8702** | **3.07** (positions 448-511) |

All four ranks are identical. The old collapse measurements below are retained
as the failure signature that localized the bug.

Uniform over this vocabulary is ln(155136) = 11.95 nats, so the last three
blocks carry almost no signal, and top-1 falls to 0-3%. The four runs agree to
+/-0.3 nats per block, so the collapse is reproducible, not a wedge.

**The decisive test** is `PPL_SKIP_TOKENS`, which scores the same corpus window
standalone. A standalone window sees only the 2 prefix tokens of context, so it
should be *harder*:

| corpus tokens | standalone | inside a 512-token run | ratio |
|---|---|---|---|
| 128-256 | 35.7 | 47.6 | 1.3x |
| 320-448 | 22.7 (n=3, 5% spread) | 14941 (n=4) | **659x** |

Corpus tokens 320-448 score 22.7 on their own with 42-45% top-1. The same
tokens inside a 512-token sequence score ~15000 with 1-3% top-1. More context
makes them 659x worse, which no property of the text can explain. The onset is
a cliff between position 256 and 320, not a gradual decay.

What this rules out:

* **Not the corpus.** The standalone arm holds the text fixed.
* **Not nondeterminism.** 659x against a 16% spread, and both arms are tight
  across reps.
* **Not a global config effect.** The first 128 positions of the 512-token runs
  (5.556-6.668) overlap the 128-token runs (5.450-6.336). Only later positions
  break. An earlier reading of a +25% early-position step was n=1 and is
  retracted.
* **Not the MFMA hazards.** Both gfx950 scaled-MFMA hazards fixed for gpt-oss
  are present on this branch and their oracle passes at both loop parities
  (`tests/standalone/test_mfma_pipeline_hazards.hip`: reference 0/19200, fixed
  0/19200, broken 19200/19200). GLM-5's own analogous silent hazard -- an SGPR
  scale operand to `v_mfma_scale_f32_16x16x128_f8f6f4` -- is fixed by
  `MPK_MFMA_VSCALE`, default 1, and `test_moe_kloop_width.hip` reports MATCH at
  every k-loop width.

Not yet localized. The shape -- fine early, cliff past ~256, worsening with
how much KV a position attends over -- points at the prefill attention rather
than at the MoE or the LM head, but that is a hypothesis and not a measurement.
The per-block sanity check in the gate now fails any run with this signature.

## How the earlier (retracted) collapse was localized

Recorded because the diagnostic path was not obvious, and because the first
reading of it was wrong.

1. The failing slice was **not** a degenerate input: it is ordinary WikiText-2
   prose, the same length as the passing one, and in fact a 64-token shift of
   it. Its targets round-tripped token-identically against the tokenizer, and
   its max id (98867) is well inside the 154880 vocabulary.
2. Configuration was byte-identical between the two runs apart from the corpus
   path -- same shard count, same sink shape, same env.
3. The failure was **deterministic** (94512 and 97566 on two runs, per-position
   NLL identical to 2 decimals for the first 12 positions), which ruled out the
   order-nondeterministic reduction.
4. Only **6 unique argmax values across 127 positions** said the hidden state
   entering the LM head had stopped depending on position -- a collapse, not a
   quality regression.
5. Because prefill runs one token per iteration, row 1 conditions on token 0
   alone. Changing **only token 0** (`' He'` to `'He'`, all 127 targets held
   fixed) moved perplexity from 94512 to 3009 -- a 31x swing from one token,
   which no correct model has. That pointed at position 0, hence at the sink,
   hence at the missing anchor.

**An earlier version of this document attributed this to EP MoE routing** on the
grounds that the trigger was content-dependent, and reported it as a probable
kernel defect. That was wrong: content selects the token at position 0, and it
was position 0 that mattered. There is no evidence of a routing defect.

Similarly retracted: a reading that quality degrades with context position. The
unprefixed runs did show a reproducible quartile trend (mean NLL 5.52 / 3.49 /
5.32 / 7.96), but that is the sink distortion spreading through the KV cache,
not a positional bug.

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

The prefix fix itself is host-side and reachable only under `PPL_MODE=1`, so it
cannot touch decode. Re-verified anyway: min 8.739 / 8.757 / 8.746, i.e. 8.739
against the 8.738 pre-change baseline. Decode text re-checked on the same
build -- Rayleigh scattering explained correctly, and the first ten primes
listed correctly with a correct definition.

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

`PPL_MODE` is in `MPK_FORWARD_VARS`: it changes the emitted LM head argument
list, so an unforwarded rank builds a different megakernel and the layer
barriers deadlock.

## Scope: what this run does NOT cover

`GLM_LMHEAD_TP=0` is **not** the shipping decode config (which tiles the vocab
four ways). With the vocab sharded no single rank holds a normalizable logit
row, and a cross-rank logsumexp is not implemented, so the gate requires the
untiled head -- which is also what makes P2 meaningful. What is covered is the
78 layers, the attention, the MoE and the EP fold; what is not is the LM head's
vocab tiling.

No Torch reference arm: GLM-5 744B at 4 bits is ~372 GB against 288 GB of HBM on
one MI350, so the single-GPU leg GPT-OSS compares against does not fit. Drop a
`torch_ppl.json` into the output dir and the test picks up the ratio arm on its
own. With the absolute number now in band, this is a lower-value follow-up than
it was when the gap was 20x.

## Reliability note

An earlier version of this document claimed runs at >= 256 corpus tokens wedge
at launch, as if it were a hard limit. **Corrected:** it is the known
intermittent cold-start hang -- all four ranks stop at `launch bc: D both grids
enqueued` and the 300 s silence watchdog kills them -- and it is probabilistic,
not a ceiling. Observed here:

| corpus tokens | wedged / attempts |
|---|---|
| 128 | 0 / 10 |
| 256 | 1 / 2 (the completing run scored 15.6414) |
| 512 | 2 / 2 |

It is not specific to PPL_MODE and does not come from the sink (80 MB at 256
tokens): a clean `run_correctness_suite.sh` run hit the same stall on prompt 0
and recovered on its retry. `run_ppl.sh` now retries the same way
(`RETRIES=2`, fresh `MASTER_PORT` per attempt), because without it a
512-token gate is a coin flip and a wedge reads as "no dump" rather than
"try again". 512 tokens is still unmeasured.
