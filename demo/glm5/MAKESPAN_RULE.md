# THE MAKESPAN RULE

**A predictor, not a speedup.** Given a proposed change to a tile phase, decide
*before writing kernel code* whether it can move the wall. Regenerate every
number here with `python3 demo/glm5/makespan_predictor.py` (offline, reads a
`MPK_BAR_SKEW=3` log; no GPU run).

This exists because the branch has an asymmetry that made every deletion look
like a failed implementation: **adding work to a tile phase costs ~1:1, removing
it saves ~0.** That is not a measurement artifact and not a hardware effect.

---

## The rule, in one line

> A rendezvous fires at `max_w arrival_w`. **The wall delta is the change in the
> MAXIMUM arrival** — not the change in the mean, and not the change in any
> counted region time.

Three regimes follow. Classify the proposed change, read the ceiling, stop.

| | regime | what it is | ceiling |
|---|---|---|---|
| **C** | ADD work to a phase | runs on every participant *including the max* | **1:1**, always |
| **A** | change work INSIDE a phase, rendezvous stays | helps a strict subset | `max_all(b) − max_{w∉P}(b)` |
| **B** | delete the phase AND its rendezvous | the segment collapses | the segment length |

**Then: if the ceiling is under the 0.26 ms wall noise floor, do not build it.**
The measurement cannot distinguish the result from noise even if the
implementation is perfect.

### Why C is always 1:1
An addition runs on *every* participating worker, including whichever one
happens to be the max. A uniform `+d` shifts the whole arrival distribution by
`+d`, so the max moves `+d`. **There is no way to build an additive probe that
misses the max.** This is why all four additions on this branch read ~1:1 — and
why they are *valid* for pricing a proposed addition and *worthless* as a
deletion ceiling.

### Why A is almost always zero
A deletion helps only the workers that ran the deleted work. Everyone else
arrives when they always did, and the barrier still waits for them.

---

## Table A — regime-A ceilings (measured, `us/layer` and `ms/token`)

Worker→tile: `xcd_id = w/29`, `xcd_rank = w%29`; a phase with T tiles/XCD is run
by `xcd_rank < T` (`gang_mla_attn_fused_mi300.cuh:225-226`).

| barrier | phase | T/XCD | in | out | max_in | max_out | **CEILING** | ms |
|---|---|---|---|---|---|---|---|---|
| qkv_barrier | qkv_a | 21 | 168 | 64 | 31.98 | 31.73 | **0.248** | 0.019 |
| qb_barrier | q_b | 16 | 128 | 104 | 48.99 | 49.48 | **0.000** | 0.000 |
| decode_barrier | MLA decode | 2 | 16 | 216 | 67.27 | 67.79 | **0.000** | 0.000 |
| hier_barrier | o_proj | 24 | 157 | 35 | 98.36 | 98.35 | **0.011** | 0.001 |
| routing poll | router | 16 | 128 | 104 | 115.20 | 114.00 | **1.199** | 0.091 |

**Read the q_b and MLA-decode rows.** `max_out > max_in`: the workers that do
*not* run the phase arrive at its barrier *later* than the ones that do. The
ceiling is not small — it is **zero by construction**. No change to those tiles,
of any size, *up to and including deleting them*, can move that barrier.

*Caveat:* `xcd_rank < T` is right for qkv_a / q_b / o_proj / router but **not**
for the MLA decode, which maps `item = xcd_id*mla_tiles_per_xcd + t` then
`q_group = item % NUM_Q_GROUPS`, scattering chunks across every XCD
(`gang_mla_attn_fused_mi300.cuh:335-341`). The empirical plateau there is
exactly 16 workers wide, which is the check that the note is right.

**Sum of all five regime-A ceilings: 1.46 us/layer = 0.111 ms.** Making
qkv_a, q_b, the decode, o_proj and the router *simultaneously instantaneous*
does not clear the noise floor.

## Table B — regime-B ceilings (the segment table, `max(b) − max(a)`, telescoping)

| segment | span us | % layer | ms/token |
|---|---|---|---|
| layer ref → qkv_barrier | 31.98 | 19.4% | 2.059 |
| qkv_barrier → qb_barrier | 17.50 | 10.6% | 1.127 |
| qb_barrier → decode_barrier | 18.31 | 11.1% | 1.179 |
| decode_barrier → attn_release | 5.46 | 3.3% | 0.352 |
| attn_release → rel_tree | 3.22 | 2.0% | 0.208 |
| rel_tree → wuv_barrier | 4.55 | 2.8% | 0.293 |
| wuv_barrier → hier_barrier | 17.34 | 10.5% | 1.116 |
| hier_barrier → routing poll | 16.84 | 10.2% | 1.084 |
| routing poll → w13_barrier | 22.62 | 13.7% | 1.456 |
| w13_barrier → entry_bar | 27.13 | 16.4% | 1.747 |
| **TOTAL (telescopes exactly)** | **164.95** | 100% | 10.619 |

*The ms column is scaled by k=0.847 to map the instrumented span onto the 10.619
control, so the TOTAL is arithmetic, not evidence. The **split** is what this
table earns. Treat ms as ±10%.*

**The MLA decode has a regime-A ceiling of 0.000 and a regime-B ceiling of
18.31 us/layer. Same phase, same kernel.** That single pair is the whole
asymmetry: `GLM_MLA_NUM_KV_CHUNKS=8` is regime A and measured null;
`MPK_MLA_SKIP_DECODE` is regime B and measured −0.61 ms (44% of its ceiling).

---

## Validation — nine levers already on the branch, six of them nulls

| lever | reg | predicted ceiling | says | measured | ok |
|---|---|---|---|---|---|
| `MPK_ATTN_HALFK` (qkv_a K bytes halved) | A | 0.019 ms | NULL | 0.047 ms | ✓ |
| `GLM_MLA_NUM_KV_CHUNKS=8` (6.89→5.36 us) | A | 0.000 ms | NULL | 14.225 vs 14.189 | ✓ |
| `MPK_W13_EARLY_REL` (12.4 us/lyr of WAIT) | A | 0.001 ms | NULL | 14.204 vs 14.189 | ✓ |
| layer-bdry ptr copy (1.20 ms counted) | A | 0.007 ms | NULL | 0.010 ms | ✓ |
| delete one whole rendezvous | A | no work removed | NULL | 0.003 ms | ✓ |
| W13 barrier narrowing (0.51 ms counted) | A | 0.001 ms | NULL | 0.10 ms | ✓ |
| `MPK_MLA_SKIP_DECODE` (phase + rendezvous) | B | 1.179 ms | MOVE | **−0.61 ms** | ✓ |
| `MPK_QKVA_REPS=2` (+8.6 us/lyr/worker) | C | +0.654 ms | 1:1 | +0.669 ms | ✓ |
| `MPK_W13_REPS=2` (+14.8 us/lyr/worker) | C | +1.125 ms | 1:1 | +1.152 ms | ✓ |

The rule is **not fitted** to these: the A and B ceilings come from arrival
stamps taken before any of these probes ran, and regime C has no free parameter.

---

## The two candidate structures that were ruled out

**1.1 ROUND QUANTIZATION — refuted as the general mechanism.**

| phase | tiles/XCD | workers | rounds | idle slots |
|---|---|---|---|---|
| qkv_a | 21 | 168 | 1 | 8 |
| q_b | 16 | 128 | 1 | 13 |
| W_UK / W_UV | 4 | 32 | 1 | 25 |
| MLA decode | 2 | 16 | 1 | 27 |
| merge | 16 | 128 | 1 | 13 |
| o_proj | 24 | 192 | 1 | 5 |
| router | 16 | 128 | 1 | 13 |
| **MoE W13** | 54 | 232 | **2** | 4 |
| **MoE W2** | 108 | 232 | **4** | 8 |

Eight of ten phases are **exactly one round**, so each participating worker runs
exactly one tile and `phase makespan = max_w(single tile time)` — the tile
*count* is irrelevant below 29/XCD. Refuted additively: `MPK_QKVA_REPS=2`
doubles the tile *without* adding tiles (verified `#pragma unroll 1`, identical
342 `s_barrier` / 736 `mfma` both arms), stays one round, still cost
+8.6 us/layer. Refuted subtractively: OPW=16 takes W2 from 4 rounds to 15 and
would predict ~3.75× the phase; measured −1.34 ms on a ~1 ms phase.
**1.1 is real only for W13 and W2** — exactly the pair whose geometry is already
measured out in both directions.

**1.2 STRAGGLER-SET MAKESPAN — right in shape, wrong in premise.** "Speed up the
one straggler" predicts a large `max − 2nd`. Measured `max − 2nd` is
**0.005–0.369 us at every one of the ten rendezvous**, against spreads up to
26.62 us. **The top of the arrival distribution is a plateau, not a peak.**

| barrier | pop | max | 2nd | med | min | max−2nd | spread | plateau (≤1us) |
|---|---|---|---|---|---|---|---|---|
| entry_bar | 232 | 164.95 | 164.86 | 158.74 | 153.92 | 0.089 | 11.03 | 17 |
| qkv_barrier | 232 | 31.98 | 31.83 | 31.12 | 20.27 | 0.150 | 11.70 | **148** |
| qb_barrier | 232 | 49.48 | 49.23 | 45.58 | 45.16 | 0.247 | 4.32 | 20 |
| decode_barrier | 232 | 67.79 | 67.60 | 47.30 | 46.65 | 0.190 | 21.13 | 16 |
| attn_release | 232 | 73.25 | 73.24 | 72.92 | 47.93 | 0.005 | 25.32 | **128** |
| rel_tree | 232 | 76.47 | 76.42 | 75.57 | 49.85 | 0.048 | 26.62 | **128** |
| wuv_barrier | 232 | 81.02 | 80.94 | 80.68 | 80.47 | 0.083 | 0.55 | **232** |
| hier_barrier | 192 | 98.36 | 98.35 | 97.90 | 97.45 | 0.011 | 0.91 | **192** |
| routing poll | 232 | 115.20 | 114.83 | 112.98 | 90.58 | 0.369 | 24.62 | 3 |
| w13_barrier | 232 | 137.82 | 137.80 | 130.88 | 121.37 | 0.012 | 16.45 | 34 |

To make a barrier fire 1 us earlier you must speed up **every worker in the
plateau column, simultaneously**. At `wuv_barrier` that is 232 of 232.

---

## The plateau identity — and what it says about the 3.55 ms

| barrier | plateau | distinct xcd_ranks on it | spans all 29? |
|---|---|---|---|
| entry_bar | 17 | 15 | no |
| **qkv_barrier** | **148** | **29** | **YES** |
| qb_barrier | 20 | 17 | no |
| decode_barrier | 16 | 12 | no |
| **attn_release** | **128** | **29** | **YES** |
| **rel_tree** | **128** | **29** | **YES** |
| **wuv_barrier** | **232** | **29** | **YES** |
| **hier_barrier** | **192** | **29** | **YES** |
| routing poll | 3 | 3 | no |
| w13_barrier | 34 | 23 | no |

A plateau spanning **all 29** xcd_ranks means the late set includes workers that
run **no tile** in the phase before it. At the qkv_barrier the plateau is 148
workers covering every rank 0..28 — the 8 ranks per XCD that run *no* qkv_a tile
are on it. **Workers doing nothing arrive as late as workers doing a 14.8 us
tile. qkv_a's tile is not what that barrier is waiting for.**

This is the answer to "the ledger prices by bytes and the attention tiles are
not byte-bound." The 55.9 us/layer of non-MoE tile against 8.55 us of bytes is
**real and on the critical path** — but table A says the share *attributable to
any one phase's worker set* is ~zero, because the non-participants arrive just
as late. **The 3.55 ms is not hiding in a tile that can be made faster. It is
the plateau: cost every worker pays regardless of which tile it owns.**

---

## What the rule forbids, and what it leaves open

**FORBIDS**
* Any regime-A change to qkv_a, q_b, the MLA decode, o_proj or the router.
  Combined ceiling 0.111 ms.
* Any argument of the form *"region X counts N ms, therefore removing it buys
  N ms."* Counted time is a mean-side quantity; the wall is a max. This is the
  general form of the three-for-three null in
  `glm-counted-region-time-before-a-barrier-is-not-a-lever`.

**LEAVES OPEN**
1. **Regime B on any phase** — Table B is the price list. 0.208–2.059 ms/row.
2. **Regime A on a 232-wide phase**, where the non-participant set is empty,
   `max_outside` is undefined, and the ceiling is the whole segment. MoE W13 and
   W2 are the only two — and they are exactly the two the ledger already found
   to respond to geometry.
3. **Lowering an entire plateau at once.** A plateau spanning all 29 ranks is
   *uniform* cost — the rendezvous itself plus whatever every worker does
   unconditionally. A uniform cut is regime C run backwards and therefore
   transfers **1:1**. This is the one place where "make the tile cheaper" still
   pays, and it is why item 2's VALU accounting is worth running: **a per-tile
   VALU cut pays only if it lands on all 29 ranks.**
