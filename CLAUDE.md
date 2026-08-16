# Working in this repo

## Communication

Be terse, logical, to the point. Lead with the answer; supporting detail only
if it changes what to do next. No restating the question, no multi-paragraph
framing, no caveat pile-ups. Use a table where it beats prose.

Keep measured numbers and correctness caveats — those are substance.

## Perf claims

Never quote a latency number without checking the generated text. Two AR
changes on this branch looked like clean speedups and were producing garbage.
`demo/gpt_oss/run_correctness_suite.sh` + `compare_tokens.py` gate this.

Change one variable per run.
