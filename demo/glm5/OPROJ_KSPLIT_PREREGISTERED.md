# o_proj K-split — pre-registered prediction

Written **before** the 1024/1024 A/B finished, so the verdict cannot be fitted
to the result. The ceiling probe (`MPK_OPROJ_KSPLIT_CEIL`) was already built and
ISA-gated at this point; only the latency numbers were unknown.

## Prediction

**NULL.** Below the 0.26 ms noise floor. Predicted magnitude ~0.03 ms, i.e.
indistinguishable from zero at n=3 pairs.

## The 2.102 vs 33.93 us/layer discrepancy — resolved

The task brief calibrated on **33.93 us/layer** (32.50 us of it directly
measured spin on `attn_release`), giving 2.58 ms at 1:1 and ~0.77 ms at the
0.3 busy→wall coefficient. That calibration uses the wrong statistic.

| statistic | value | what it measures |
|---|---|---|
| ATOM-comparison profile | 33.93 us/lyr | **mean-worker occupancy** of the attention tail — how much *worker-time* is burned |
| `price_attention_shard.py` row 23 | 2.102 us/lyr | **max-arrival delta** — the critical-path contribution |

Both are correct. They differ by ~16x for a reason that `MAKESPAN_RULE.md`
already documents, in its own straggler table:

| barrier | pop | max | 2nd | med | min | max−2nd | spread | plateau (≤1us) |
|---|---|---|---|---|---|---|---|---|
| attn_release | 232 | 73.25 | 73.24 | 72.92 | 47.93 | 0.005 | **25.32** | **128** |
| wuv_barrier | 232 | 81.02 | 80.94 | 80.68 | 80.47 | 0.083 | **0.55** | **232** |

At `attn_release` the arrival spread is 25.32 us, but **128 of 232 workers
arrive within 1 us of the max**. The ~104 early workers each spin ~25 us — that
spin is real, it is what the profile counted, and it sums to the 32.50 us
mean-worker figure. But the barrier fires at `max_w arrival_w`, and the max is a
128-wide *plateau*, not a straggler. Deleting the wait releases the 104 early
workers; it does not make any of the 128 plateau workers arrive sooner. The wall
does not move.

At `wuv_barrier` the case is even clearer: spread 0.55 us, plateau 232 — *every*
worker arrives within 0.55 us of the max. There is no waiting to remove.

**The spin is conserved, not eliminated.** Released early workers immediately
hit `hier_barrier` at o_proj:925, which survives any o_proj layout and is
explicitly not a target. They spin there instead.

## Why the ATOM comparison is not the refutation it looks like

ATOM genuinely pays 0.00 us here — but it pays 0 because it has no idle workers
in that state to count, not because it removed critical path. Comparing
mpk's worker-time occupancy against ATOM's kernel-boundary cost compares two
different quantities. The 3.45 ms gap is real; this line item is not 33.93
us/layer of it.

## Both target rendezvous were already directly priced, and both are ~0

The repo had already ablated both, which the brief's "UNPRICED" reading missed:

- `attn_release` — per-XCD narrowing replay of the measured arrival log:
  GPU-wide max minus worst-XCD max = **0.000 us** at this barrier
  (`glm-per-xcd-barrier-narrowing-is-zero-by-measurement`). Recorded
  `resolved=True, delta=0.000`.
- `wuv_barrier` — `MPK_WUV_IN_MERGE` (e5d1ff5) **already deletes this
  rendezvous outright**, ten rendezvous become nine, correctness-gated 4/4.
  Measured **−0.003 ms** (n=2 paired), −0.019 on min-iter n=5.

`MAKESPAN_RULE.md`'s validation table also already carries the general row:
*"delete one whole rendezvous | regime A | no work removed | NULL | 0.003 ms"*.

## Regime classification

Deleting a rendezvous **without deleting the work in front of it** is regime A,
not regime B. Regime B ("the segment collapses") requires removing the phase
*and* its rendezvous. Here W_UV and the attention tail still compute; only the
wait disappears. For the max arriver the wait is ~0 by definition — it is the
max, it never waits — so its arrival at the surviving `hier_barrier` is
unchanged. **Regime-A ceiling ≈ the arrival spread of the plateau ≈ 0.55 us/lyr
at `wuv_barrier`, ~0 at `attn_release`.**

0.55 us/layer, scaled by the segment table's own 1 us/layer ↔ 0.0645 ms,
is **0.035 ms** — an order of magnitude under the 0.26 ms noise floor.

## What would falsify this

A measured control−ceil2 delta ≥ 0.26 ms, reproducible across all 3 alternating
pairs. That would mean rendezvous *deletion* escapes the makespan rule in a way
rendezvous *narrowing* does not, and would justify building the full K-split.

---

## OUTCOME (appended after the run; nothing above was edited)

**Prediction held.** Measured **−0.169 ms** (n=2 alternating pairs, 1024/1024)
against the 0.26 ms noise floor. Predicted "NULL, ~0.03 ms"; the sign and the
sub-noise verdict were right, the magnitude was low by ~0.14 ms.

The prediction was made for `ceil2`; the measurement is `ceil1`, because ceil2
hangs deterministically at 1024/1024 (it corrupts the router, so workers wait on
`ep_signal` lines no peer will write). ceil1 additionally emits token garbage,
so −0.169 ms is an upper bound and not a speedup.

Full write-up, with the ISA evidence and the pair table: `OPROJ_KSPLIT_NULL.md`.
