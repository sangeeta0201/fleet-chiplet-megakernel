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

Filled in after the run.
