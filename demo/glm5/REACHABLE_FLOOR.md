# GLM-5 744B bs=1 decode: what floor is actually reachable

Measured on MI350 (gfx950), NP=4, `HIP_VISIBLE_DEVICES=4,5,6,7`, 76 fused
layers. Wall at the time of writing: **8.862 ms/iter**. Byte roofline:
**2.144 ms**. This document answers the question the 4.1x gap invites — how
much of it is actually available — and it is deliberately arithmetic rather
than aspirational.

Unit convention throughout: the instrumented layer span is 164.947 us against a
139.72 us true layer, so **1 us/layer = 0.0644 ms/token** (76 layers, span
corrected). Every per-layer number below is converted with that factor.
Independent cross-check on the factor: non-MoE per-worker busy sums to 53.9
us/layer = 3.47 ms, against the ledger's independently derived 3.55 ms non-MoE
line. Those agree to 2%.

---

## 1. The answer

| tier | floor | what it takes |
|---|---|---|
| today | 8.862 ms | — |
| kill the EP routing tax (§1a) | **~7.9 ms** | expert-TP + shard the shared expert |
| + every intra-layer lever at its measured ceiling | **6.6 – 7.3 ms** | the decode slice release, plus scraps |
| a schedule redesign — fewer, wider phases | ~4.5 – 5.5 ms | not an optimization; a rewrite |
| byte roofline | 2.144 ms | unreachable at any schedule |

**2–3 ms is not reachable.** Two earlier estimates in this work are superseded:
a loose 6–7 ms guess made before the per-phase ceilings were measured, and this
document's own first answer of 7.6–8.3 ms, which was wrong for a more
interesting reason — see §1a.

### 1a. CORRECTED 2026-09-01 — this document excluded its largest lever

The first version of this analysis put the reachable floor at 7.6–8.3 ms and
closed with the prediction that *no single remaining lever is worth > 0.5 ms*.
**That prediction is false, and it was false because the analysis scoped out
the biggest item in the tree and then did not count it.**

`ROOFLINE.md` line 58 already named it, and calls it "the single largest item
the roofline shows":

> Expert-*parallel* is the wrong decomposition at batch 1. Splitting each
> activated expert's weights across all 8 ranks (tensor-parallel *within* an
> expert) makes every rank read exactly one expert-equivalent, with zero
> variance.

Sized for our actual NP=4 geometry, using exact balls-in-bins (`roofline.py`'s
`_emax_balls_in_bins`, top-8 into 4 bins) and the measured W13+W2
critical-worker busy time of 30.8 us/layer:

| | expert-equivalents at the busiest rank | us/layer | ms |
|---|---|---|---|
| routed today (EP) | 3.538 (mean 2.000, imbalance **1.77x**) | | |
| routed under expert-TP | 2.000, zero variance | −10.44 | **−0.672** |
| shared expert, replicated today | 1.000 | | |
| shared expert, sharded 4 ways | 0.250 | −5.09 | **−0.328** |
| **combined** | | **−15.53** | **−1.000** |

One routed expert is 20.05 MB at mxfp4; the busiest rank reads 4.538
expert-equivalents per layer, so W13+W2 cost 6.787 us/layer per
expert-equivalent. These are proportionality estimates off measured busy time,
not measurements — but they are an order of magnitude above the 0.5 ms the
falsified prediction allowed.

Two things make this cheaper to build than "a redesign":

- **The reduction already exists.** W13 column-parallel and W2 row-parallel is
  the standard Megatron MLP split, and it needs no collective between them —
  SwiGLU is elementwise on the sharded intermediate dim. The cross-rank sum of
  W2 partials that it *does* need is exactly what the EP fold already performs
  on `moe_ws_f32`. No new rendezvous.
- **It lands on W13 and W2**, the only 232-wide phases, hence the only ones
  where a cut converts at 1:1 rather than ~0.3.

It also removes the straggler, not just the bytes: every rank reads the same
amount with zero variance, which is the same thing gpt-oss's `9d478ea` bought
one level down by pinning experts to XCD pairs ("the 240-tile padding space and
its straggler imbalance are gone").

The rest of this document — §2 through §7 — was written before this was sized
and still reads as if the intra-layer levers were all there was. Its per-line
reasoning stands; its bottom line does not.

---

## 2. What the roofline is, and what it is not

2.144 ms is a *byte* floor: model bytes divided by achievable HBM bandwidth,
width-corrected. It assumes zero rendezvous, zero arrival skew, and perfect
overlap. The layer has ten GPU-wide rendezvous and two cross-rank ones, and
those cost ~2.94 ms on their own. So the roofline is not a target that any
schedule with barriers can approach — it is a lower bound on a different
program.

### CORRECTED 2026-09-01 — the gpt-oss comparison below was wrong

This section originally read: "gpt-oss at 1.835 ms is a 120B model on MI355.
GLM-5 is 744B. Its byte floor alone is larger than gpt-oss's entire wall." That
compares **models** when the hardware comparison is **per GPU**, and it let a
real gap hide behind a parameter count.

gpt-oss runs on **one** GPU (`export GPU=0` in its README). GLM-5 runs on four.
So the honest comparison is 120B/GPU against 744B/4 = **186B/GPU — 1.55x, not
6x.** Size is not the explanation.

**The actual structural difference is that GLM-5 pays for expert parallelism
and a single-GPU config does not.** From `ROOFLINE.md`, per iteration:

| EP-specific cost | ms | paid by gpt-oss? |
|---|---|---|
| routing imbalance — busiest rank reads ~3 experts where the mean is 1 | 0.58 | no |
| shared expert replicated on every rank (20.05 MB/layer/rank) | 0.25 | no |
| EP cross-rank collective (measured 5.1 us x 75) | 0.38 | no |
| **total** | **~1.21** | |

That is floor, before any inefficiency. On *efficiency* the two are closer than
the walls suggest: GLM-5 achieves 1.691 TB/s, 32.7% of the measured 5.17 TB/s
peak.

Techniques port between the two. Numbers do not — but the reason is expert
parallelism, not 744B.

---

## 3. The four-line budget

From the ledger, taken at the 10.619 ms wall where it was measured:

| line | ms | status |
|---|---|---|
| byte floor | 2.144 | closed |
| rendezvous (10 GPU-wide + 2 cross-rank, x76) | ~2.94 | see §4.2 |
| MoE tile above its byte floor | 1.10 | closed both directions |
| non-MoE tile above its byte floor | 3.55 | mechanically identified, §4.4 |

The wall has since come down to 8.862, so these lines are collectively ~1.76 ms
lighter than tabulated; the shape is what matters, not the last digit.

---

## 4. Reachability, line by line

### 4.1 Byte floor — 2.144 ms, reachable 0

Nothing here without changing precision or the parallelism split. Not in scope.

### 4.2 Rendezvous — ~2.94 ms, reachable ~0.1–0.3 ms

The standalone null probe measures **3.77 us per rendezvous**, which looks like
a 2.9 ms prize. It is not, and three independent measurements say so:

- **Arrival is 0.5% of it.** `MPK_BAR_TREE` cuts serialized arrival atomics from
  232 to 37 and measured null *twice* (−0.015 ms on per-iter min, n=5 either
  side). Dividing out: the tree removes ~19 **nanoseconds** per barrier. A
  232-way same-address atomic does not serialize 232 ways; the L2 pipelines it.
  The probe measured a loop of nothing but atomics, where issue rate is all
  there is to see.
- **Deleting a whole rendezvous buys 0.003 ms.** The skew relocates to the next
  barrier rather than disappearing.
- **Release observation is not staggered.** Measured this session across eight
  barrier-exit slots: the spread in observation time by `xcd_rank` is
  0.005–0.026 us with correlation ≈ 0 (max +0.46 at one slot, on a 0.020 us
  range). Every worker sees every release within tens of nanoseconds.

So the rendezvous line is overwhelmingly **skew being paid somewhere**, not
mechanism that can be deleted. The one sub-item with a real mechanism story
left is the EP cross-rank wait, 6.863 us/layer = 0.442 ms, of which ablation
says 83% relocates rather than vanishes — leaving ~0.075 ms.

### 4.3 MoE tile — 1.10 ms, reachable ~0.1–0.2 ms

W13 and W2 are the only 232-wide phases, so they are the only ones where
tile-interior work converts 1:1. They are also **the only two software-pipelined
loops in the layer** — the ISA census puts W13 at carry 12 and W2 at carry 8,
both on the flat part of the bandwidth curve, where the tool's own guidance says
there is nothing left to take.

Their combined critical-worker busy time is 30.8 us/layer = **1.983 ms**, which
is the hard cap on all tile-interior work in the layer. W13 depth 6 took 0.232
ms of it this month. The ledger records this line closed in both directions.

### 4.4 Non-MoE tile — 3.55 ms, reachable ~0.4–0.8 ms

This is the only line with real room, and it is the one that does not convert.

**Mechanically it is identified.** The gfx950 ISA census:

| loop | loads | carry | drains | shape |
|---|---|---|---|---|
| W13 tile | 12 | 12 | 0 | pipelined |
| W2 tile | 8 | 8 | 1 | pipelined |
| router GEMM | 12 | **0** | **12** | load, `vmcnt(0)`, consume, x12 |
| GEMV (W_UK/W_UV/o_proj) | 16 | **0** | 2 | 8-load bursts, staircase between |
| MLA decode | 10 | **0** | 2 | 9-load burst, `vmcnt(17)`..`vmcnt(0)` staircase |
| KV latent write | 7 | **0** | 3 | small bursts |

Exactly one loop — the router GEMM — is naively serialized. The decode and the
GEMV already emit the counted-vmcnt staircase that gpt-oss had to hand-write for
its W13 handoff. What the non-MoE loops uniformly lack is **cross-backedge
carry**, so each trip exposes a round trip.

**And here is why fixing that does not pay.** The predictor's ceiling for a
phase is `max_all(b) − max_{w∉P}(b)`. Splitting the stamps by participation:

| phase | participants arrive | non-participants | ceiling |
|---|---|---|---|
| o_proj | 90.380 | 90.358 | 0.022 us |
| router | 96.735 | **101.100** | 0 |
| W13 | 119.934 | **121.444** | 0 |
| W2 | 138.953 | **140.452** | 0 |
| **MLA decode** | **58.5** | 41.4 | **17.5 us/layer = 1.13 ms** |

At four of five phases the **non-participants arrive later than the
participants**. The workers that skip a phase are its stragglers. That is the
whole explanation for six historical nulls, and it also kills the obvious fix:
widening a phase to 232 workers does not give its work to idle capacity, it
gives it to workers who are already later. `GLM_MLA_NUM_KV_CHUNKS=32` proved
this directly by widening the decode from 64 workers to 128 for **+0.157 ms**.

The plateau is therefore **not slack**. It is a schedule already balanced in
total work per worker, with phase boundaries falling in different places for
different workers. Two independent measurements of the conversion coefficient
agree: `MPK_ATTN_HALFK` cut 2.3 us/layer uniformly and converted at **0.27**,
and the combined regime-A ceiling for all five non-MoE phases at once is
**0.111 ms**.

So of the 3.55 ms, everything except the decode is capped at ~0.111 ms.

### 4.5 The one open ceiling: MLA decode, 1.13 ms

The decode is the sole phase whose participants set the makespan — 64 workers
(`q_groups=4 x kv_chunks=16`) arrive at 58.5 us while the other 176 are at 41.4.
The slot-21 histogram is cleanly bimodal at exactly 64/176, confirming it
independently. MAKESPAN_RULE lists 0.000 for this phase; that is stale, taken at
the old 16-worker mapping.

The 17.5 us splits into two unequal halves:

| half | us/layer | ms | status |
|---|---|---|---|
| barrier wait before the decode starts | ~11.2 | ~0.72 | needs the slice-release port |
| decode compute | 7.8 | 0.50 | attacked, register-bound |

**The compute half is register-bound, measured.** The loop is already two trips
deep, but `kv_pre` is a single register buffer, so trip t drains it for tile t+1
then issues tile t+2 into the same registers — a WAR edge that stops the issue
being hoisted, which is why carry is 0. Double-buffering by tile parity removes
the hazard and still loses: **8.980 mean / 8.936 min (n=3) against 8.858 min**,
about +0.08 ms. The image says why — carry stays 0, drains go 2→5, and scratch
ops go 902→1054, because the 18 extra VGPRs push a function whose arch peak is
already 248 into spilling. `mla_decode_absorbed` sets the *whole megakernel's*
register allocation, making it the worst place in the tree to spend registers.
Any further work on this loop must be carry-neutral in registers, or must buy
the headroom first.

**The barrier half is the better target and is unattempted.** Decode workers
wait on a GPU-wide q_b→decode rendezvous, but a decode work item only needs the
q_b tiles covering its own 16 heads and the current token's KV row. Replacing
that with a per-slice counted handoff is precisely gpt-oss's
`MPK_ATTN_SLICE_RELEASE` shape, which was worth 1.880 → 1.864 there. Ceiling
here is ~0.72 ms; realistic capture 0.2–0.4 ms, because part of the 11.2 us is
genuine q_b arrival spread that no handoff removes.

One caveat on the 11.2 us: it is `max`-over-workers-of-`mean` compared against a
barrier-exit mean, and by Jensen the true per-layer last arrival is later than
42.779, so some of the gap is hidden skew rather than mechanism. The honest read
is that 11.2 us is an **upper** bound on what the handoff can address. Pricing it
exactly needs per-layer maxima, which `BARSTAGEWS` does not carry (it stores
count and sum only) and which `MPK_SUBPHASE_TIMING` cannot supply — enabling it
puts the wall at 78.8 ms, because its own atomics cost ~9 us per stamp per
layer. Adding `pmin`/`pmax` to the `BARSTAGEWS` dump is the cheap fix and is
the prerequisite for taking this half seriously.

---

## 5. The arithmetic

| line | budget | reachable |
|---|---|---|
| byte floor | 2.144 | 0 |
| rendezvous | ~2.94 | 0.1 – 0.3 |
| MoE tile | 1.10 | 0.1 – 0.2 |
| non-MoE, decode barrier half | (of 3.55) | 0.2 – 0.4 |
| non-MoE, decode compute half | (of 3.55) | 0 – 0.15, needs register headroom |
| non-MoE, everything else | (of 3.55) | ≤ 0.111 |
| **total** | | **0.6 – 1.3 ms** |

**8.862 − (0.6 … 1.3) = 7.6 … 8.3 ms.**

That is the floor reachable by every identified lever landing at its measured
ceiling simultaneously — an optimistic reading, since ceilings are ceilings.

---

## 6. What would change the answer

The binding constraint is not any loop. It is that **the layer has ten
rendezvous and eight distinct phase widths** (232, 232, 192, 168, 128, 128, 64,
...). Every narrow phase creates a population that arrives later than the
participants, which is what drives every conversion coefficient to ~0.3 or 0.

A schedule with, say, four phases all 232 wide would:

- cut rendezvous from ten to four (~1.2 ms instead of ~2.94);
- give every phase a real ceiling, unlocking the non-MoE 3.55 ms at something
  near 1:1 instead of 0.111 ms;
- make the carry-0 loops worth fixing, since their savings would convert.

Rough bound: 2.144 byte floor + ~1.1 MoE tile + ~1.2 rendezvous + residual
non-MoE ≈ **4.5–5.5 ms**. This is a redesign of the phase decomposition and the
tile→worker mapping, not an optimization pass, and the o_proj arithmetic shows
the sort of obstruction it runs into: 1536 columns per rank divide into 24 or 32
tiles, not the 29 that would put exactly one tile on every worker. Uneven tiles
are possible but change every kernel's indexing.

---

## 7. Falsifiable predictions

Recorded so this document can be shown wrong rather than argued with.

1. Pipelining the router GEMM to carry ≥8 will measure **null** (|Δ| < 0.05 ms),
   despite being the only genuinely serialized loop in the layer, because its
   non-participants arrive 4.4 us later than its participants.
2. Widening **any** of o_proj, router, W13 or W2 will measure **≥ 0** (i.e. no
   better, probably worse), for the same reason `kv_chunks=32` cost +0.157 ms.
3. The q_b→decode slice release will land in **0.15–0.45 ms**, not the 0.72 ms
   ceiling.
4. Any change adding ≥12 VGPRs to `mla_decode_absorbed` will regress, regardless
   of how much latency it hides, because the function is at its arch peak.
5. ~~No single lever remaining in this codebase is worth **> 0.5 ms**.~~
   **FALSIFIED 2026-09-01, before anyone else had to.** Expert-TP is ~0.672 ms
   and sharding the shared expert ~0.328 ms, both on the 1:1 phases — see §1a.
   The prediction held only because this document had scoped both out as "a
   redesign" without pricing them. Per the rule below, the budget was
   re-derived rather than patched; §1 and §1a are the result.

If prediction 5 falls, this analysis is wrong in an interesting way and the
budget above should be re-derived rather than patched.

---

## 8. Provenance

- Stamps: `BARSTAGEWS` per-worker count/sum, rank 0, 38321 fused-layer samples,
  `MPK_BAR_STAGE=1`. Analysed with `layer_critical_path.py`.
- ISA: `llvm-objdump -d --mcpu=gfx950` over the JIT image from
  `permanent_output_dir_rank0`, analysed with `isa_outstanding.py`. Note that
  its `carry` counts overlap across the backedge **only**, and reading its
  "STARVED" verdict as "one load at a time" overstates three of four rows —
  check the disassembly before acting on it.
- Walls: `bench_repeat.sh`, n=3 where quoted, per-iter min preferred over mean
  because single runs occasionally carry a >1 ms outlier (one control set here
  had 9.996 against a 8.858 min).
- Correctness: `MPK_BS_DEBUG=2` in-situ checksums, not text sampling — two runs
  of the *same* build disagree on ~50% of tokens because EP atomic accumulation
  order varies.
