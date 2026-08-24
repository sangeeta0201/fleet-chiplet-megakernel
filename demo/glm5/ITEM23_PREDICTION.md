# v9 Item 2.3 — rendezvous COUNT, re-priced at NP=4

**Written BEFORE the run.** One variable: `GLM_UNABSORB_OPROJ=0`.

## What the knob does

Re-absorbs W_UV into o_proj. That **deletes one GPU-wide rendezvous and one
whole phase** (Regime B, the only regime in the ledger that pays) and **pays
for it in bytes**: o_proj's K doubles 16384 → 32768.

This is the clean half of task #70. The other half, `GLM_UNABSORB_QB=0`, is
**not run** — `qb_tp` in demo.py requires `UNABSORB_K`, so turning absorption
on also turns the q_b head shard off. At NP=8 that confound cost +1.02 ms and
none of it was absorption. At NP=4 it would be 4x replicated q_b instead of
8x — smaller, still confounded. Fixing the confound is a demo.py change, not a
measurement; it is out of scope for this item and named as unfinished.

## The prediction

| term | ms | source |
|---|---:|---|
| o_proj bytes 25.95 → 51.90 MB/layer/rank | +0.376 roof | roofline.py --ranks 4 |
| × the measured byte slope 0.911 | **+0.343** | world_size_fit.py, fixed-NP widen |
| delete P8 attn→o_proj GPU-wide barrier | −0.192 | NP=4 phase table |
| delete the W_UV phase (nested S24..S27) | −0.176 | NP=4 phase table |
| **NET** | **−0.025** | |

Cross-check from the NP=8 measurement instead of the NP=4 table: NP=8 measured
−0.06 against a byte cost of 0.911 × roof(12.98 MB) = +0.171, so the implied
benefit there was 0.231 ms. Carry that benefit unchanged to NP=4 and the net is
0.343 − 0.231 = **+0.112**.

**PREDICTION: neutral. −0.025 to +0.112, i.e. |Δ| < 0.15 ms, inside the
0.26 ms noise floor.** Expected wall 11.39 – 11.53 against the 11.418 baseline.

## What would falsify what

| outcome | reading |
|---|---|
| Δ inside ±0.26 | as predicted. A rendezvous is worth ~its bytes at NP=4 too; the deletion does not scale with world size while the byte cost does. Item 2.3 closes negative. |
| Δ < −0.3 | Regime B pays MORE at NP=4 than the phase table's 0.368 ms says. That would make rendezvous COUNT a live lever and reopen the other nine. |
| Δ > +0.3 | the byte cost is superlinear in o_proj's K — the 0.911 slope does not transfer to a K-doubling inside one GEMM. That would put a caveat on Item 1's slope. |

Note the direction asymmetry this probe tests, which the widen could not: the
widen was a pure ADDITION and additions price ~1:1
(`glm-additive-probes-overprice-deletions`). This probe adds bytes AND deletes
a phase in the same run, so it prices the two directions against each other.

## Result

**MEASURED 11.498 ms, n=3 (11.400 / 11.526 / 11.569), all three texts read and
coherent. Δ = +0.080 ms against the 11.418 baseline.**

`unabsorb_v=0` verified in the log; `layer_o_proj_red` stays at the absorbed
32768 and no W_UV stack is attached (`demo.py:2489`). `permanent_output_dir*`
was removed before the run because the knob changes generated geometry.

**Inside the predicted band (−0.025 … +0.112) and well inside the 0.26 ms
noise floor. Item 2.3 closes NEUTRAL — as predicted, not as hoped.**

### Correcting one term of the prediction

The prediction charged the full +25.95 MB of o_proj and forgot that the 2.16 MB
W_UV read goes away with the phase. Net bytes are +23.79 MB/layer/rank
= +0.345 ms of roof = **+0.314 ms** of wall at the 0.911 slope, not +0.343.
That moves the point prediction to −0.054. It does not change the verdict.

### What the run actually buys us — two things worth more than the null

**1. One rendezvous + its phase is worth ~0.23 ms, and that does NOT scale
with world size.**

| | NP=8 | NP=4 |
|---|---:|---:|
| byte cost of re-absorbing | +0.157 | +0.314 |
| measured net | −0.06 (n=1) | **+0.080 (n=3)** |
| **implied deletion benefit** | **0.217** | **0.234** |

The benefit agrees to 0.017 ms across a 2x world-size change while its byte
price doubles. So **re-absorption gets strictly worse as the world shrinks**,
and the direction is now measured rather than argued. (The NP=8 row is an n=1
screen — its error bar is the full noise floor, so read the agreement as
"consistent", not "confirmed to 17 us".)

**2. Falsifier 3 did not fire.** The 0.911 byte slope was measured on an
addition spread across every MoE weight. Here it was applied to a K-doubling
inside a single GEMM — a completely different geometry — and predicted the
outcome to 0.13 ms. The slope transfers.

### Left unfinished, named

`GLM_UNABSORB_QB=0` — still confounded (`qb_tp` requires `UNABSORB_K`, so it
also disables the q_b head shard). Un-confounding it is a demo.py change, not
a measurement. Given that the clean half of the same trade measures neutral and
gets worse at smaller world sizes, the expected value of building it is low,
but it is **unpriced, not priced out**.
