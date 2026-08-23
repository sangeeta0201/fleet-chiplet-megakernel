# The attention-side shard diff: 4 of 10 rendezvous are ours, and all 4 are already ablated

Date 2026-08-23. Branch `merge-rocshmem`. **No GPU run, no build.** Every price
below comes from an ablation already on the ledger. Reproduce the table with
`python3 demo/glm5/price_attention_shard.py`.

This is the attention twin of `MOE_TP_FALSIFIED.md`, run under the same rule
that file's failure produced: **no build is authorized on a scaled number.** A
row is priced only by a measurement of that exact variable, or it is marked
UNPRICED. It turned out that zero rows needed the UNPRICED marker — every
survivor had already been ablated, three of them by probes built for other
reasons.

## Result

| | |
|---|---|
| rendezvous that exist because of our attention shard layout | **4 of 10** |
| their face value in the budget | 0.867 ms |
| largest **directly measured** deletion | **0.228 ms** (qb_barrier) |
| sum of all four measured deletions | 0.161 ms (one is negative) |
| rows needing an estimate | 0 |
| **verdict** | **NO BUILD** — nothing reaches the 0.5 ms bar, and the largest is under the 0.26 ms noise floor |

---

## 1. The shard layout, read off the kernel

Not inferred from op names. That mistake is what produced the MoE's false
~1.4 ms one commit ago.

| stage | site | how it is dealt |
|---|---|---|
| `qkv_a` | `attn:677` | output **column** across XCDs: XCD *x* owns columns `[xS,(x+1)S)` of `[q_a \| latent]`. q_a is the 2048-wide q_lora rank, latent the 512-wide kv_lora rank. **Neither axis is heads.** |
| `q_b`, `W_UK` | `attn:430,439` | **already head-parallel.** `QB_TP_HEADS = 64/8 = 8`; rank *p* owns heads `[8p, 8p+8)`, one head per XCD |
| `decode` | `attn:374` | items are `(q_group, kv_chunk)` = 4 × 16; the unit is a **16-head group** |
| `merge` | `attn:344` | `q_group = xcd_id/2` — pair-aligned already |
| `W_UV` | — | replicated GEMV, 4 tiles/XCD, read by all of o_proj |
| `o_proj` | `oproj:677` | **column-sharded (N-split)** across ranks + all-gather on the `ep_signal` line. Rank *p* emits hidden columns `[768p, +768)` and therefore **contracts over all 64 heads** |

**The single fact that shapes the whole answer is the last row.** Attention is
*already* head-parallel where a head is a legal unit (`q_b`, `W_UK`). What keeps
the attention tail all-to-all is not how the heads are dealt — it is that
o_proj's shard makes every rank contract over every head.

## 2. Which of the ten are ours

| # | rendezvous | stamp | face ms | layout-dependent? |
|---|---|---|---|---|
| 1 | `entry_bar` | 9 | 0.145 | no — MoE(L−1) → layer L, a device-scope atomicAdd over the whole hidden row |
| 2 | `qkv_barrier` | 17 | 0.316 | no — qkv_a's output axis is a lora rank, not heads |
| 3 | **`qb_barrier`** | 19 | **0.582** | **YES** |
| 4 | **`decode_barrier`** | 21 | **0.150** | **YES** |
| 5 | **`attn_release`** | 23 | **0.135** | **YES** |
| 6 | `rel_tree` | 25 | — | no — release/self-heal mechanism, not a data edge |
| 7 | **`wuv_barrier`** | 28 | — | **YES** |
| 8 | `hier_barrier` | 30 | 0.154 | no — o_proj → router; **this is TileRT's own comm point** |
| 9 | `routing poll` | 32 | 0.268 | no — MoE |
| 10 | `w13_barrier` | 6 | 0.336 | no — MoE |

`rel_tree` and `wuv_barrier` carry no separate face value because
`close_the_wall.py`'s budget folds their spin into neighbouring spans. Both are
real `hier_barrier_arrive()` sites and both are priced by ablation below, so the
missing face value changes no verdict.

Two of the six drops matter enough to state plainly:

* **`qkv_barrier` cannot be dealt away by any head layout.** The consumer
  contracts over the whole q_a latent. TileRT does not shard it either —
  `rmsnorm_projx_wqkva` is *replicated*, which removes the rendezvous by
  recompute. Priced already in `REGIME_B_SEGMENT_LEGALITY.md` §2a: qkv_a goes
  11.844 → ~95 µs/layer, **+5.4 ms.** Strictly negative.
* **`hier_barrier` is the comm point TileRT also pays.** `unproj_o_allreduce`
  is exactly this edge. It survives every layout; the only difference is that
  TileRT fuses it into the epilogue, and epilogue-vs-separate is measured
  neutral twice on the EP fold (`glm-qkv-ep-fold-hoist-is-neutral`,
  `glm-widening-the-ep-fold-is-neutral`).

## 3. Compile legality — checked before the price

| candidate | verdict | why |
|---|---|---|
| head-local `qb_barrier` | **DOES NOT INSTANTIATE** | rank *p* owns **8** heads (`QB_TP_HEADS = 64/8`, `attn:430`). The decode's unit is a q_group of **16** (`NUM_Q_GROUPS = NUM_Q_HEADS/16`, `attn:374`, a `constexpr`; 16 is the MFMA M-tile). 8 heads is *half* a q_group — a rank cannot form one. `PAIR_MERGE`'s guards at `attn:382` additionally require `8 % NUM_Q_GROUPS == 0` and `XCDS_PER_GROUP \| NUM_KV_CHUNKS`. |
| `attn_release` + `wuv_barrier` via o_proj N-split → K-split | **COMPILES** | o_proj has **no MFMA at all** (`glm-wuk-wuv-oproj-have-no-mfma`), so there is no `MFMA_ITERS % 4` static_assert to fail — unlike the W2 K-split that blocked the MoE TP shard. This one is legal to build. |
| `decode_barrier` re-deal | **COMPILES, AND IS ALREADY BUILT** | `MPK_GLM_MLA_PAIR_MERGE`, `attn:377`, wired at `persistent_kernel.py:476`. |

## 4. The price — direct measurements only

| rendezvous | face ms | **ablated ms** | compiles | the ablation |
|---|---|---|---|---|
| `qb_barrier` | 0.582 | **0.228** | **no** | `MPK_QB_SKIP_PEER_WAIT` deletes the 7 peer stores + 7-peer poll. **n=6** pooled over two independent triples, 10.498 → 10.271, t = 2.84 on df 10 |
| `decode_barrier` | 0.150 | **−0.070** | yes | `MPK_GLM_MLA_PAIR_MERGE` re-deals decode to match merge. **Built and measured** (`e482f2e`): +0.07 ms at 16 chunks, +0.6% at 4 |
| `attn_release` | 0.135 | **0.000** | yes | per-XCD replay of the measured arrival log: GPU-wide max − worst-XCD max = 0.000 µs here (`glm-per-xcd-barrier-narrowing-is-zero-by-measurement`) |
| `wuv_barrier` | — | **0.003** | yes | `MPK_WUV_IN_MERGE` (`e5d1ff5`) hoists W_UV behind the pair-local barrier and **deletes this rendezvous outright** — ten become nine, correctness gated 4/4. −0.003 ms paired n=2, −0.019 min-iter n=5 |

**Three of the four have already been built and run.** This class did not need a
new probe; it needed the ledger read in the right order.

### And they do not add

`glm-inter-rank-skew-is-paid-once` measured that deleting the EP collective
moves **83%** of its time onto exactly the `qb_barrier` region
(S19→S20: 9.35 → 23.23 µs/layer). The 0.228 is the *second* cross-rank
rendezvous' marginal mechanism and it is only that cheap because the first one
already absorbed the skew. Summing the column would double-count the same skew,
which is the error `price_xcd_narrowing.py` §3.3 already named: *"the 2.94 ms of
rendezvous is not a barrier-count budget; it is one arrival spread re-counted at
ten places."*

## 5. Why the largest survivor is closed even at 0.228

`glm-qb-peer-wait-ceiling-is-0.335ms` ends with the sentence this whole pass was
sent to re-examine: **"Head-sharding attention end-to-end is CLOSED."** Its
reason is structural, and re-reading it against the o_proj shard confirms it:

> the real head-shard rewrite must still combine the head-sharded attention
> output across ranks, and that payload is comparable to the query row it stops
> gathering (128 heads × 512 latent vs 128 × (512+64)). The rewrite **moves** a
> transfer onto o_proj's existing gather rather than deleting one.

That is the same shape as the MoE falsification one commit earlier: the
collective you would "delete" is one the new layout needs too. Realized gain is
**strictly less than 0.228 ms**, against a rewrite touching decode, merge, W_UV
and o_proj — and it does not instantiate at 8 heads against a 16-head q_group.

## 6. What this closes

The attention shard layout is **not** where TileRT's 8-vs-2 rendezvous advantage
converts into wall time for us:

* 6 of our 10 rendezvous are not attention-layout artifacts at all. Two of those
  six are comm points TileRT pays as well.
* The 4 that are, are worth 0.228 ms at the absolute most, by ablation, and one
  of them is worth −0.070.
* The one structural change that *would* fuse two of them — o_proj N-split →
  K-split — compiles, and is the only unbuilt item in the table. It is bounded
  above by the two ablations that already deleted the barriers it targets:
  0.000 and 0.003 ms.

Combined with `MOE_TP_FALSIFIED.md`, both halves of the shape axis are now
priced against direct measurements and both are under the noise floor. See
`demo/glm5/CLOSING_REPORT.md`.
