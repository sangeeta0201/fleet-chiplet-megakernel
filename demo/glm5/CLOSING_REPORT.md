# GLM-5 744B decode, 8× MI355X, bs=1, one decode row — closing report

**Result: 10.619 ms/token, 5.3× above the 2 ms goal.** Output correct, gated by
`run_correctness_suite.sh` + `correctness_gate.py` (G1+G2+G3; exact-token
equality is an illegal gate at bs=1 — the model has two attractors).

This report closes the *shape* axis. For the per-item lookup table of everything
closed before it, read `demo/glm5/CLOSING_LEDGER.md` (`cd6698c`); its §8
impossibility claim was overruled on 2026-08-23 and this file supersedes it.
The framing here is deliberate: **not "10.619 is the floor," but "10.619 is what
this parallelization reaches, and here is exactly what it could and could not
reach."**

---

## 1. What this parallelization reached

| | ms/token | commit |
|---|---|---|
| start, 2026-08-18 | 14.824 | |
| o_proj sharded across 8 EP ranks | 13.057 | `317c419` |
| o_proj / W_UV GEMV tile retune | 12.256 | `be7947e` |
| router `irms` cached across the 2nd expert | 12.117 | `5e940c2` |
| MoE dispatch clamped to owned tiles | −0.180 | |
| KV chunk granularity 32 → 8 tokens (decode 16 → 64 tiles) | 11.427 | `e158ff4` |
| …and the rest of the ladder | **10.619** | |

1.40× total. Every remaining item below is measured, not asserted.

## 2. What is closed, and by what measurement

Only measurements are listed. Predictions and rooflines are excluded on
purpose — this run learned the hard way what they are worth.

### The budget closes with no room in it

`close_the_wall.py` accounts **10.002 of 10.619 ms**, residual 0.617 (5.8%):
tile 5.292 · cross-rank collective 1.587 · rendezvous 2.371 · boundary 0.751.

### …and the biggest class contains no big item

After correcting the stage-stamp spans for rendezvous spin *inside* a phase
(the router's `OPROJ_BARRIER` at `gang_rmsnorm_linear_bias_mi300.cuh:852` has no
barrier slot of its own, so the span billed 54% poll as busy), the per-phase
headroom table is **flat**: q_b 0.695 [upper bound] · decode 0.597 · qkv_a 0.524
[UB] · W13 0.445 · W2 0.421 · router 0.420 · o_proj 0.285 · merge 0.195. Busy
converts to wall at 0.27 near this operating point (`MPK_ATTN_HALFK`, `73c0afb`),
so **the largest single tile lever in the layer is 0.188 ms — under the 0.26 ms
noise floor.**

### Closed by direct ablation or built probe

| item | measurement | result |
|---|---|---|
| barrier narrowing (per-XCD, all ten at once) | arrival-log replay; GPU-wide max − worst-XCD max = 0.000 µs at all ten | 0.095 ms |
| deleting a whole rendezvous | `MPK_WUV_IN_MERGE`, ten → nine, gated 4/4 | −0.003 ms |
| all-thread barrier polling | measured twice | +0.73 ms |
| q_b head-shard cross-rank gather | `MPK_QB_SKIP_PEER_WAIT`, n=6 pooled, t=2.84/df 10 | 0.228 ms |
| decode → merge re-deal | `MPK_GLM_MLA_PAIR_MERGE`, built | +0.07 ms |
| layer boundary | ablation | 0.051 ms |
| MoE tile narrowing (OPW=16) | built | −1.34 ms |
| W2 split-K (`KSPLIT=2`) | built | **+4.12 ms** |
| W2 L2 prefetch | built | 0.13% |
| shared-expert hoist / duplication | `MPK_SHARED_DUP` | +0.106 ms |
| EP fold hoist; EP fold widen | both built | neutral |
| dense-prologue fusion (`GLM_FUSE_ATTN=1`) | built | +0.996 ms |
| TopK rank-select | built | +33% on the tail |
| FP8 activations | probe vs real stage | +12% at K=10240, **−26%** at W_UV's K=512 |
| occupancy ladder | `wpe=3` faults at launch; LDS-locked at 155/160 KB/CU | dead |
| MTP / spec-decode / bs=2 | measured, then ruled off the board | ceiling ~1.3× on a 5.3× problem |

### Closed on paper, against direct measurements (this run)

| item | why | price |
|---|---|---|
| regime B on its two largest segments | every edge is all-to-all: producer shards the axis the consumer contracts over | 0.000 ms each, vs 2.059 + 1.747 face |
| **MoE EP → TP** | `_rnlm8_ep_fold_slice` **already is** the 8-way hidden-row all-reduce TP needs; 4.0 owned experts still fits one grid-stride round of 29; the 8-way W2 K-split fails `static_assert(MFMA_ITERS % 4 == 0)` | ~1.4 predicted → **~0.03, and it does not compile** |
| **attention sharding** | only 4 of the 10 rendezvous are attention-layout artifacts, and all 4 were already ablated; the largest, `qb_barrier`, does not instantiate (8 owned heads against a 16-head `q_group`) | 0.867 face → **0.228 measured** |

## 3. What was NOT reachable by this parallelization — and why

Not "physics." Three concrete structural facts, each with the measurement that
established it:

1. **Every intra-device edge in the layer is all-to-all.** Each of the ten
   rendezvous separates a producer that shards the very axis its consumer
   contracts over. Measured consequence: narrowing all ten to per-XCD
   simultaneously, legality ignored, is 0.095 ms — because every chiplet holds a
   worker at the global max, so a per-XCD barrier releases seven XCDs early and
   the eighth not at all. The "2.94 ms of rendezvous" was never a barrier-count
   budget; **it is one arrival spread re-counted at ten places.**

2. **The cross-rank cost is a tax, not a lever.** Inter-rank skew is paid once,
   at the *first* cross-rank rendezvous. Deleting the EP collective moves 83% of
   its time onto the q_b gather (S19→S20: 9.35 → 23.23 µs/layer). Deleting the
   second one gives 92% back precisely because the first already absorbed the
   skew. The two cannot be added, and neither can be deleted twice.

3. **54% of the layer is last-arriver floor** — 6.498 of 12.098 ms, of which
   1.506 is cross-rank peer wait the intra-rank estimator cannot see, so the
   honest local floor is ~4.99 ms. The four release holes have
   `min(window, movable) = 0`: there is idle time, and there is no work legally
   movable into it.

The comparison that motivated the shape work stays honest: TileRT runs the same
nine ops per MoE layer with **two** communication points, both fused into the
producing GEMM's epilogue, against our ten rendezvous + EP collective + QB_TP
gather. That difference is real. What this run established is that **it is not
convertible here**: two of our ten are comm points TileRT pays as well, six are
not attention-layout artifacts at all, and the four that are price out at
0.228 ms by ablation. TileRT is also FP8 on B200 with DSA sparse attention and an
MTP headline; we are MXFP4 on MI355X at bs=1 with MTP off the board.

## 4. The methodological finding — two pricing errors, both caught this run

Both were mine, both were caught before spending a build, and both are
transferable. They are the most reusable thing this run produced.

### Error 1 — pricing a phase by scaling bytes when a direct measurement of that exact variable already existed

I priced the MoE EP→TP re-shard by scaling the phase's bytes (80.22 → 22.56 MB)
against its measured 48%-of-roof efficiency, giving 0.388 ms of work saved. The
direct measurement was already on the ledger and said ~15× less: the busiest
rank owns **3× the mean routed bytes** and its MoE makespan is within **1.05 µs**
of every other rank's.

The reason is round quantization, not bandwidth. W13 runs 6 tiles/XCD/expert and
W2 runs 12; both fit **one** grid-stride round of 29 workers up to 4 owned
experts, and the busiest rank sits at exactly 4.0. Cutting it to 1.125 stays in
the same round and removes no makespan.

> **The rule: a byte imbalance costs makespan only if it crosses a grid-stride
> round boundary. Compute tiles/XCD against 29 before believing any roofline
> imbalance line.** Same family as
> `glm-cutting-work-in-a-phase-is-absorbed` and
> `glm-shared-expert-imbalance-is-absorbed`.

The general form is worse than the instance: **a roofline is a bound on a
phase's byte time; it is not an estimator of that phase's response to a change.**
Where an ablation of the same variable exists, the roofline is not a second
opinion — it is simply wrong.

### Error 2 — reading an external op graph's names and inferring a mechanism

I read TileRT's `expert_down_allreduce`, saw we had no op by that name, and
priced deleting our EP routing collective at 1.005 ms. But
`_rnlm8_ep_fold_slice` (`gang_rmsnorm_linear_mxfp8_bias_mi300.cuh:409`) sums
`EP_PEER_SLOTS` copies of the hidden row in fp32 — **it already is an 8-way
all-reduce of the hidden row**, the identical one TP needs. Different name,
different position in the graph, same semantics. The collective term was
**0.000 by construction**, an identity, and no GPU time could have discovered it.

> **The rule: an external op graph tells you the shape; it does not tell you
> which of your own ops already implements it. Grep your own tree for the op
> before pricing its absence.**

### The correction these two produced

Every survivor in the attention pass was priced **only** by an ablation of that
exact variable, with its *n*, and rows without one were to be marked UNPRICED
rather than estimated. No row needed the marker: all four had already been
ablated, three of them by probes built for other reasons. The class did not need
a new measurement — it needed the ledger read in the right order.

A third correction, smaller but the same species: the internal-spin audit found
that stage-stamp spans bill rendezvous *inside* a phase as busy. Applying the
router's measured 46% compute fraction moved it from the #1 headroom row
(0.944 ms) to sixth (0.420), and dropped the tile-class gap from 3.40× to 3.09×.
**An instrument's span is not a phase's work until you have grepped its callees
for `while (`.**

## 5. What is genuinely still open

Stated so the next run does not re-derive the closed set. Nothing below is
recommended on current evidence; all three are named because they are the only
items not measured out.

1. **o_proj N-split → K-split** (TileRT's `unproj_o_allreduce`). The one unbuilt
   item in the attention table. It **compiles** — o_proj has no MFMA, so no
   `MFMA_ITERS` assert — but it is bounded above by the two ablations that
   already deleted the barriers it targets: 0.000 and 0.003 ms.
2. **A different parallelization entirely**, not a different kernel. The 5.3×
   does not live in any single line of the budget; it lives in the fact that
   this layout pays ten all-to-all edges per layer and the arrival spread at
   each of them is the same spread.
3. **The 0.617 ms residual** (5.8%) — dense prologue layers, sampling, harness.
   Never attributed.

---

*Method note kept for the next run:* the wall noise floor is 0.26 ms, so n=1
cannot resolve a sub-0.2 ms lever; a baseline is only ever its own control, never
yesterday's; and `MPK_SUBPHASE_TIMING` costs 3.1 ms at 16 chunks and once fully
masked a real 2.76 ms win. Read `demo/glm5/CLOSING_LEDGER.md` §6 before touching
any instrument.
