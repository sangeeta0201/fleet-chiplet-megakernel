# The shape diff against TileRT — where our 10 rendezvous come from

> **RETRACTED IN PART, 2026-08-23 — see `MOE_TP_FALSIFIED.md`.** The EP→TP
> pricing below (~1.4 ms) is WRONG on both halves. The collective term is
> **0.000**: TP needs the identical 8-way hidden-row all-reduce, and
> `_rnlm8_ep_fold_slice` already IS one. The work term is **~0.03 ms**, not
> 0.388: the measured response of this phase to this exact variable is 3x
> bytes → 1.05 µs of spread, because 4.0 experts still fits one grid-stride
> round. And an 8-way K-split of W2 fails `static_assert(MFMA_ITERS % 4 == 0)`.
> **The reframe in the section "The cause" and below stands; the price table
> does not.**

Offline. **No GPU run, no build.** Reads `~/TileRT` (READ-ONLY reference) and the
closed budget in `demo/glm5/close_the_wall.py`.

## Why do this now

The budget is closed (10.002 of 10.619 accounted) and, after the internal-spin
correction, **flat**: the largest single line item anywhere converts to 0.19 ms
of wall, under the 0.26 ms noise floor. Regime B is closed on both its largest
segments. Every barrier mechanism is measured out. Tuning is finished; the only
remaining variable is shape.

`glm5-2ms-goal-do-not-stop` names the existence proof: TileRT does GLM-5 at
~2.02 ms/token on 8× B200. That is ~26 µs/layer against our 139.72. TileRT ships
as a binary wheel, but its **Python op graph is source** and that is exactly the
layer that carries the shape.

## The op-for-op diff

TileRT's MoE layer, from `register_op` in `tilert/models/glm_5/modules/`:

| # | TileRT op | ours |
|---|---|---|
| 1 | `rmsnorm_projx_wqkva` | qkv_a (+ RMSNorm prologue) |
| 2 | `rmsnorm_projq_wqb` | q_b |
| 3 | `rmsnorm_kv` | (inside qkv_a) |
| 4 | `projq_wqb` | W_UK |
| 5 | `projo_wkvb` | W_UV |
| 6 | **`unproj_o_allreduce`** | o_proj **+ a device rendezvous** |
| 7 | `rmsnorm_expert_proj` | router |
| 8 | `exp_sel_up_gate_silu` | W13 + SwiGLU |
| 9 | **`expert_down_allreduce`** | W2 **+ a device rendezvous** |

The op *list* is essentially ours. The tile phases match one for one. **What
does not match is the communication.**

* **TileRT has exactly TWO communication points per layer, and both are fused
  into the epilogue of the GEMM that produces the data** —
  `unproj_o_allreduce` (o_proj + all-reduce + residual add, one kernel, one
  `flag` argument, `sms = 128`, `dim_per_sm = 6144/128 = 48` output rows per SM)
  and `expert_down_allreduce` (W2 + all-reduce). There is no separate collective
  op and no separate barrier op anywhere in the graph.
* **We have ten rendezvous plus an EP routing collective plus a QB_TP gather.**
  Priced: rendezvous 2.371 ms + cross-rank collective 1.587 ms = **3.958 ms,
  37% of the wall.**

## The cause: TileRT is TP; we are EP

Every TileRT collective is an **all-reduce of the hidden row** (6144 elements,
~12 KB). That is the signature of tensor parallelism: weights are sharded along
the contraction axis, activations are replicated, and the only thing that has to
cross ranks is the partial sum.

We shard the MoE by **expert**. `moe_max_b = 3.0 * per_expert * mq` in
`width_corrected_roofline.py` is the honest statement of the consequence: with 8
activated experts thrown into 8 ranks, the busiest rank holds **E[max] ≈ 3**,
plus the shared expert = **4.0 experts**. Under TP every rank holds exactly
(8 + 1)/8 = **1.125** expert-equivalents, uniformly, by construction.

**This reframes the regime-B result.** `REGIME_B_SEGMENT_LEGALITY.md` proved
every one of our rendezvous separates a producer whose shard is misaligned with
the consumer's deal, and concluded no narrowing is legal. That proof is correct
— **about our shard layout.** It is not a property of the model. TileRT runs the
same nine ops with two communication points because it chose a layout in which
the producer's shard and the consumer's contraction agree, and the misalignment
is paid once, as a fixed-size all-reduce, in the epilogue.

## Pricing the EP → TP re-shard of the MoE  — **RETRACTED, see MOE_TP_FALSIFIED.md**

Numbers from `tile_roof_by_phase.py` and `close_the_wall.py`, same log.

| item | now (EP) | under TP | note |
|---|---|---|---|
| MoE bytes, busiest rank | 80.22 MB | 22.56 MB | 4.0 → 1.125 expert-equiv., **3.56x** |
| byte roof @ 5373 GB/s, 232 wrk | 14.93 µs | 4.20 µs | |
| measured W13+W2 busy | 31.07 µs | ~8.7 µs | holding the measured 48%-of-roof efficiency |
| work saved | | **22.3 µs/lyr** | = 1.436 ms at 1:1, **×0.27 = 0.388 ms** |
| EP routing collective | 15.617 µs/lyr | 0 | TP has no cross-rank expert routing |
| | | **1.005 ms** | whole-phase deletion, regime B, nearer 1:1 |
| | | **≈ 1.4 ms total** | |

**Feasibility — the footprint is identical.** 256 experts × 20.05 MB × 75 MoE
layers = 385 GB of MoE weight. EP: each rank owns 32 whole experts = 48 GB. TP:
each rank owns 1/8 of all 256 = 48 GB. Same bytes, different slicing, and 48 GB
fits 288 GB HBM either way. This is a re-slice, not a capacity change.

**Contiguity holds.** Shard W13 by the intermediate dim (column) and W2 by the
intermediate dim (row) — the standard MoE-TP split, and the one that makes W2
the op needing the all-reduce. That is precisely `expert_down_allreduce`.

## The three risks, stated before any build

1. **Tile count roughly doubles.** 8 experts × 1/8 each is 8 tile groups where
   we now run ~4 whole ones. The MoE phase is round-quantized at
   `ceil(tiles/29)` (`glm-moe-phase-is-round-quantized-and-w2-is-latency-bound`)
   and W2 is latency-bound, not bandwidth-bound. Fixed per-tile cost could eat
   a large share of the 0.388 ms work term. The 1.005 ms collective-deletion
   term is not exposed to this.
2. **The imbalance may already be absorbed.** `glm-ep-routed-imbalance-is-floor-
   not-wall` measured 3x bytes producing only **1.05 µs of cross-rank spread**.
   That says the MoE phase is not byte-bound at this size, which is the same
   thing W13's 63–76% of roof says. So the *work* half of the price is the soft
   half; the *collective deletion* half is the hard half.
3. **It is a large build** — weight loader, dispatch, the W2 kernel, and a new
   fused all-reduce epilogue. Not a one-parameter probe.

## What this does NOT claim

* Not that TileRT's 2.02 ms is reproducible here. TileRT is **FP8 on B200**; we
  are **MXFP4 on MI355X**, and its GLM-5 path also uses DSA sparse attention
  (`index_topk = 2048`, `tilert/models/glm_5/ops/sparse_index_v3.py`) which is a
  long-context lever and irrelevant at our decode length. Its headline chart is
  also an MTP chart — MTP is off our board.
* Not that the 1.4 ms is measured. It is a **prediction from the closed budget**,
  written down before any build, in the same form the regime-B pricing used.

## The one thing it does establish

The 3.958 ms in the rendezvous + collective classes is **not a floor**. We
proved it unremovable *given our sharding*; TileRT demonstrates a sharding of the
same nine ops in which it is two fused epilogues. That is the first item found
this run whose predicted wall delta exceeds 0.5 ms.
