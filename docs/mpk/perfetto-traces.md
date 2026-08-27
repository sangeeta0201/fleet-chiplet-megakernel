# Perfetto traces: seeing the overlap

`MPK_PHASE_SLOTS` answers "where does a layer's time go". It cannot answer
"what ran at the same time", because it reports means: two workers each
averaging 5 us in MoE tell you nothing about whether they overlapped or ran
back to back. Overlap is the entire claim a megakernel makes over a sequence of
kernel launches, so showing it needs absolute timestamps rather than averages.

`MPK_PHASE_TRACE` keeps the raw per-`(worker, layer, slot)` timestamps that the
span table reduces away, and `phase_slots_to_perfetto.py` turns them into a
Chrome Trace Format file that <https://ui.perfetto.dev> opens directly.

## Capture

```bash
cd demo/gpt_oss
MPK_PHASE_SLOTS=1 MPK_PHASE_TRACE=1 MPK_PHASE_START_ITER=600 \
  ./run_1gpu.sh --max-seq-length 512 --max-new-tokens 620 \
  > /tmp/trace_run.log 2>&1

python3 ../../tests/ci-tests/phase_slots_to_perfetto.py \
  /tmp/trace_run.log -o /tmp/trace.json
```

`MPK_PHASE_TRACE=1` without `MPK_PHASE_SLOTS=1` raises rather than tracing
nothing: the buffers and the dump both sit inside the phase-slot `#ifdef`, so
it would otherwise compile clean and produce an empty trace.

Layout in the UI is one process per XCD, one thread per worker, one slice per
phase, with compute phases and spin phases coloured differently. Threads sort
by worker id, so a barrier reads as a wall of slices ending at the same x, and
the stragglers are whatever overhangs it.

## Naming a span for what ran, not for where it sits

Every bug this tooling has had was the same bug: a span inferred its identity
from its position instead of from the kernel. The trace drew 168 of 248 workers
per XCD-layer running a QKV GEMM they had branched past; it named the gap
between the two MoE arms "W13->W2 barrier wait" when the poll is 267 lines
inside the W2 span; it billed two different barriers to `topk_wait`. Each was
plausible from the picture and wrong in the source, and each was settled by a
line number.

So the device now records what it did, and the converter is not allowed to
guess:

- **participation bits** (`p=`) -- a mark on a gated slot is taken by every
  worker that *reaches* the phase, because a mark inside the guard would leave
  non-participants with no timestamp. The bit is the only thing that separates
  "ran it fast" from "fell through the branch". Non-participants are drawn as
  `(skipped: not a <phase> participant)` in grey rather than omitted, so the
  row stays continuous and the skip is visible instead of inferred from a gap.
- **arm codes** (`a=`) -- MoE has no rank guard, so all 248 workers set the
  participation bit while doing different work. The arm records which half
  (W13, W2, both, neither) a worker's tiles actually landed in.
- **tile and barrier sub-marks** -- the finest measurement, and derived
  independently of the bits. The verifier asserts the two agree rather than
  picking a winner.

## The detail row: four weight DMAs, and whether they hide

Each worker gets a second track below its own, carrying the async weight loads.
It is the one place in this trace where two spans at the same x really do mean
"at the same time" — a `buffer_load_lds` in flight while the compute row above
runs something else — which is why it cannot live on the worker's own row,
where Perfetto renders overlap as nesting.

The kernel issues **four** such DMAs into a wait, and for a long time the trace
drew two. The other two were asserted in kernel comments — "~3us overlap
instead of serial", "DMA runs in the background during the spin-wait" — and
never measured. That is the same error as naming a span from its position: a
claim about the code read off the source rather than off the clock.

Each is now `issue -> drain -> done`, straddling the `s_waitcnt` that consumes
it. The first span is how long the load *had* to finish; only the second, the
wait itself, says whether it did. A wide in-flight window is equally consistent
with a fully hidden DMA and with one that stalled at the end of it.

| DMA | in flight (median) | uncovered (median) | at/under instrument floor |
|---|---|---|---|
| O-proj | 17.48 us | 0.44 us | 99% |
| W2 | 2.88 us | 0.44 us | 99% |
| W13 | 0.88 us | 0.72 us | 0% |
| QKV | 3.20 us | no drain mark exists | — |

O-proj and W2 hide completely: their drains sit at the floor, so what those
`s_waitcnt`s measure is the mark, not the kernel. The comments were right, and
are now measurements. W13's is not — none of its 6,659 samples fall under the
floor — so roughly 0.2 us per worker per layer of that load is genuinely
uncovered.

QKV has no drain mark by construction: its consumer is the *next* layer's Phase
1, which skips its own DMA rather than draining this one, so no instant in this
layer means "it finished". Its span runs issue → layer exit and is labelled as
such.

One asymmetry worth the guard it needs: the O-proj *issue* sits inside
`if (xcd_rank < oproj_topk_tiles_per_xcd)` while its drain does not, so 8,927
rows take the drain mark and only 6,624 take the issue. Pairing them
unconditionally drew 2,303 spans a layer named "oproj weight DMA drain" on
workers that issued no O-proj DMA at all. Their `s_waitcnt` is real, but it is
draining somebody else's traffic. The converter requires the issue code before
it will draw the drain, and the verifier restates that rule independently.

## Barriers, and why scope is in the name

Each barrier poll is cut out of its enclosing phase span and labelled
`(GLOBAL)` or `(XCD-LOCAL)`, because scope changes what a stall means.
XCD-LOCAL is 31 workers on one die rendezvousing, so a stall is local imbalance
and rebalancing that die fixes it. GLOBAL couples all 8 XCDs through one
counter, so the slowest die sets everyone's release and local rebalancing
cannot help.

Every label was established twice: from the source (which atomic, whose flag,
who fans out) and from the trace. `barrier_scope.py` does the second
independently -- for each barrier it asks whether the release tracks the last
arrival *GPU-wide* or the last arrival *on its own die*, and reports a
disagreement with the name as a failure.

Two cautions that cost real debugging time:

- **Simultaneity is the wrong test.** Comparing cross-XCD against within-XCD
  release spread calls a hierarchical barrier LOCAL, because one global
  arrival opens a serialized fan-out that releases each die's ranks together
  while skewing the dies apart. Coupling is a question about causes.
- **A non-binding barrier carries no scope signal.** Where pollers and
  producers are disjoint sets (`oproj_hier`: 6.36 us median, 0.16 us standard
  deviation) every poller pays the same fixed latency, both delays are noise,
  and their ratio near 1 means nothing. The script says "non-binding" instead
  of manufacturing a verdict.

## Why the timestamps are comparable across workers

`s_memrealtime` is a constant-rate counter shared by every XCD on the die, so
marks from different workgroups can be compared without cross-calibration.
Measured on an idle gfx950, eight workgroups landing on eight different XCDs
and stamping slot 0 at the same point spread over ~1 us, which is the launch
skew of the dispatch rather than clock divergence.

That is the assumption the whole trace rests on, so the converter checks it:
any span whose end precedes its start is dropped and counted, and a nonzero
count is reported as an anomaly rather than rounding.

## Cost, and what it does to what it measures

The mark itself is unchanged from `MPK_PHASE_SLOTS` — one `s_memrealtime` and
a few stores on thread 0 only, no printf on the hot path. `MPK_PHASE_SLOTS`
itself measured 2.061 ms against a 2.052 ms baseline, where `MPK_DEVICE_TIMING`
costs ~54 ms/iteration; the trace adds stores to a second buffer on the same
already-taken branch.

**With the flags off it costs nothing, measured rather than assumed.** Every
mark is a `do {} while (0)` outside the `#ifdef`, but the two instrumented
kernels also gained a real `int _pslot_w` parameter and both are
`__noinline__`, so the argument is passed whether or not anything reads it.
Three runs each, same GPU, back to back:

```
                median    runs
this branch     1.900 ms  1.937 / 1.897 / 1.900     # flags unset
HEAD~           1.899 ms  1.895 / 1.899 / 1.903
```

0.001 ms apart on a 1.9 ms token, well inside the spread. The 1.937 outlier is
the first run after a fresh build, not the parameter. Reproduce with
`./run_1gpu.sh --max-seq-length 512 --max-new-tokens 70` and read
`[FWD_PASS_TOTAL]`.

What it does add, when the flags ARE on, is memory and teardown printf:

- 864 KB of BSS: the buffer is sized by `MPK_PHASE_MAX_WORKERS` (256) x 36
  layers x 12 slots x 8 bytes, not by the 248 workers actually launched
- tens of thousands of printf lines at termination, against the span table's
  fixed 248

Lower `MPK_PHASE_TRACE_LAYERS` to trace fewer layers if the device printf
buffer truncates. The converter warns when `[PTRACE_END]` is missing, because a
truncated trace looks exactly like a real load imbalance -- the last workers
appear to stop early -- and that is a lie worth refusing to render silently.

## What to look for

- **`attn_release barrier (GLOBAL)`** is the single largest line in the trace
  at 28.7% of all worker time, and it used to be invisible: it sat inside
  `attn_release_wait` along with the attention epilogue. Every worker's slice
  ends at the same x; the spread before it is the attention skew being
  absorbed. It is ~98% poll — the non-poll tail is 0.40 us against a 17.9 us
  median, established from the last worker to arrive at each XCD-layer, which
  by definition waits on nobody.
- **MoE-only ranks** have no QKV or attention work -- but they are *drawn*, as
  `(skipped: not a qkv participant)` in grey, not omitted. Omitting them left a
  gap that read as missing data, and role is not the same as idleness.
- **`inter_layer`** is the gap between one layer's exit and the next layer's
  entry -- the persistent loop's own bookkeeping. It is drawn explicitly so a
  row of layers reads as continuous instead of leaving unexplained whitespace.
- **The instrument floor.** Slot 4 spans two phase marks on adjacent source
  lines with no code between them, so its duration *is* one mark's cost:
  0.52 us median. `span_table.py` prints it, and gives each row a `%flr`
  column — the share of that row's samples at or under the floor. A row at
  99% measured the instrument, not the kernel. Without it, "0.44 us" and
  "0.72 us" look like the same kind of number, and only one of them is.
