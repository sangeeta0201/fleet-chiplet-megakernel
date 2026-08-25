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

What it does add is memory and teardown printf:

- 864 KB of BSS: the buffer is sized by `MPK_PHASE_MAX_WORKERS` (256) x 36
  layers x 12 slots x 8 bytes, not by the 248 workers actually launched
- tens of thousands of printf lines at termination, against the span table's
  fixed 248

Lower `MPK_PHASE_TRACE_LAYERS` to trace fewer layers if the device printf
buffer truncates. The converter warns when `[PTRACE_END]` is missing, because a
truncated trace looks exactly like a real load imbalance -- the last workers
appear to stop early -- and that is a lie worth refusing to render silently.

## What to look for

- **`attn_release_wait`** (slot 5) is where workers converge before the MoE
  half of the layer. Every worker's slice should end at the same x; the spread
  before it is the attention skew being absorbed.
- **MoE-only ranks** have no QKV or attention slices at all -- their slots 1-4
  are zero and the converter emits nothing for them. That is role, not
  idleness, and is why `summarize_phase_slots.py` reports the two bands
  separately.
- **`inter_layer`** is the gap between one layer's exit and the next layer's
  entry -- the persistent loop's own bookkeeping. It is drawn explicitly so a
  row of layers reads as continuous instead of leaving unexplained whitespace.
