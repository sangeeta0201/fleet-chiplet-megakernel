# Participant narrowing at `attn_release`: NULL, registered before any build

**Prediction, written before a line of kernel code was changed: NULL, and the
ceiling is 0.000 ms at the real participant set, 0.061 ms under an oracle set
that no implementation can beat.** Against a 0.26 ms wall noise floor.

Registered in the style of `OPROJ_KSPLIT_PREREGISTERED.md`, and for the same
reason: the change is a large, race-prone ordering edit, and the repo already
holds the arrival data that decides it.

---

## 1. What was proposed

Port gpt-oss's participant narrowing
(`gang_full_layer_fused_mi300.cuh:1141-1195`) to GLM's attention -> o_proj
rendezvous. gpt-oss joins its QKV barrier only with the workers that produce or
consume that layer's QKV output:

```
if (xcd_rank < qkv_epoch_participants) { ... atom_add_release_gpu_s32 ...; }
```

and lets "the MoE-only workers beyond both skip it entirely and go straight to
Phase 6, where their O-proj weight DMA can start overlapping immediately."

GLM's `attn_release` (`gang_mla_full_layer_fused_mi300.cuh:1782`) is joined by
**every dispatched worker**: `int const arrivals = tiles_per_xcd * 8;`.

## 2. Three of the four preconditions are already met in GLM

Read before proposing the port, because two of them make the port smaller than
it looks and the third makes it unnecessary.

| gpt-oss precondition | GLM status |
|---|---|
| release values derived from the layer counter, not snapshotted, so the barrier carries no ordering for other counters | **already done** — `s_exp[i] = task_layer_idx + 1` for all nine counters in `ml_mode` (:1434-1440), with the same argument spelled out at :1424-1433 |
| skippers start their o_proj weight DMA immediately | **already done, and it is not gated on skipping** — every rank with an o_proj tile issues `PF_LPT` (132 at GLM-5 shapes) `buffer_load_lds` *above* the poll (:1936-2050), explicitly citing gpt-oss's Phase 6 as the source |
| the participant set is narrower than the dispatch | **true, by 5 of 29 ranks** — see §3 |
| the narrowing moves the wall | **false** — see §4 |

## 3. The participant set, from the live NP=4 run, not from a guess

`[CFG]` lines emitted by `demo.py` on the 1024/1024 control run:

```
[CFG] W_UV   tp=1 rows_per_rank=4096 wuv_rows=32 tiles_per_xcd=16
[CFG] o_proj tp=1 cols_per_rank=1536          tiles_per_xcd=24
[CFG] q_b/W_UK tp=1 heads_per_rank=16 qb_tiles_per_xcd=32 wuk_tiles_per_xcd=8
```

with a dispatch width of `tiles_per_xcd = min(max(...), num_workers/8) = 29`.

* **Producers** of `attn_out`: the split-KV merge, `xcd_rank < merge_tiles_per_xcd`
  = 16 of 29.
* **Consumers**: o_proj, `for (t = xcd_rank; t < oproj_tiles_per_xcd; t += tiles_per_xcd)`
  (`gang_oproj_router_fused_mi300.cuh:261`), so `xcd_rank < 24` of 29.

o_proj is COLUMN-sharded (`OPROJ_TP`, N-split, :875-901) and therefore contracts
over all 64 heads, so a consumer reads the **whole** `attn_out` row — every
XCD's slice. That is why the rendezvous is GPU-wide, and it means the consumer
set cannot be shrunk by dealing heads differently. It shrinks for exactly one
reason: **o_proj is 24 tiles against a 29-wide dispatch**, so ranks 24..28 own
no tile.

**Producers are a subset of consumers**, so the participant set is the consumer
set: 24 of 29 ranks, **192 of 232 workers**. 40 workers (17%) could skip.

So the answer to "can GLM's participant set narrow at all?" is **yes, by 40
workers** — narrower than gpt-oss's but not empty. The port is legal. It is
still not worth building, for the reason in §4.

## 4. The ceiling, replayed from measured arrivals

`demo/glm5/price_participant_narrowing.py`, offline, no GPU run. Same
`MPK_BAR_SKEW=3` arrival log (`/tmp/item11_bs1.log`) and the same
`D[i][w] = arr_w(B_i) - M_{i-1}` invariant `price_xcd_narrowing.py` uses;
identity check reproduces the 158.932 us layer span to 1e-6.

The bound is computed **map-free**. `w` in a `BARSTAGEWS` row is `blockIdx.x`
(`mpk_atoms.cuh:1415`), not `tile_idx`, so `xcd_rank = w % 29` is an assumption
about the dispatch rather than something the log proves — and the 48 workers
absent from `hier_barrier` are scattered rather than a clean residue class.
So instead of assuming a map, the skippers are taken to be the **K latest
arrivers**, which no real participant set can beat: any other set of size K
leaves a later worker behind for the barrier to wait on.

Two skip sets are priced. `[1]` is `attn_release` alone. `[2]` is the maximal
legal skip-ahead — a worker with no o_proj tile has no W_UV tile and no router
tile either (16/16 of 29), so its first genuine data dependency after the merge
is the routing poll; it could skip `attn_release`, `rel_tree`, `wuv_barrier`
and `hier_barrier`.

| K skipped | saved us/lyr [1] | ms/token [1] | saved us/lyr [2] | ms/token [2] |
|---|---|---|---|---|
| 8 | 0.000 | 0.0000 | 0.516 | 0.0345 |
| 16 | 0.000 | 0.0000 | 0.590 | 0.0394 |
| 32 | 0.000 | 0.0000 | 0.641 | 0.0429 |
| **40 (the real set)** | **0.000** | **0.0000** | **0.659** | **0.0440** |
| 48 | 0.000 | 0.0000 | 0.911 | 0.0609 |
| 96 | 0.000 | 0.0000 | 1.582 | 0.1057 |
| 128 | 0.000 | 0.0000 | 1.930 | 0.1289 |

Cross-check with the literal residue-class skipper set (`xcd_rank >= 24`), for
both plausible dispatch widths: **0.000 us at width 29, 0.002 us at width 30**.

**Registered prediction: NULL. 0.000 ms at the real set; 0.061 ms even at the
oracle set with all four barriers skipped. Both under the 0.26 ms floor.**

### Why effect 1 is exactly zero

Dropping the 40 latest arrivers moves `attn_release`'s fire time by 0.255 us —
because 128 of 240 workers arrive within 0.5 us of the max (`MAKESPAN_RULE.md`'s
plateau table, reproduced exactly by this script). The top of the distribution
is a **plateau, not a straggler**; removing 40 workers from a 128-wide plateau
leaves 88 still at the top.

### Why effect 2 is almost zero

The released skippers do not vanish; they arrive at the next rendezvous they do
join. Under `[1]` that is `rel_tree`, one instruction later, and the saving is
identically 0.000. Under `[2]` they run to the routing poll — which fires when
the single TopK completer finishes, an event no skipper influences — so they
spin there instead. **The spin relocates; it does not vanish.** This is the same
sentence `OPROJ_KSPLIT_NULL.md` §6 ends on, now measured for the population
narrowing as well as for the scope narrowing.

## 5. Verdict: do not build

`MAKESPAN_RULE.md`: *"if the ceiling is under the 0.26 ms wall noise floor, do
not build it. The measurement cannot distinguish the result from noise even if
the implementation is perfect."* 0.000 is under 0.26 by construction and 0.061
is under it by 4.3x.

The change is also the expensive kind: `arrivals` would drop from
`tiles_per_xcd * 8` to `oproj_tiles_per_xcd * 8`, which falsifies
`rel_tree`'s `arrivals == tiles_per_xcd * 8` predicate and therefore re-routes
the hierarchical release and the `hier_barrier_should_heal` quota. A count that
disagrees with the arriving population **deadlocks rather than errors**. Paying
that risk for a measured 0.000 ms is not a trade this branch should make.

`MPK_ATTN_REL_NARROW` was therefore **not written**. This file is the record.

## 6. What the profile line actually was

The campaign brief priced this rendezvous at 33.93 us/layer of mpk-vs-ATOM
deficit. That figure is **mean-worker occupancy**; the wall responds to max
arrival, and the same log puts `attn_release`'s critical-path contribution at
2.102 us/layer (`price_attention_shard.py` row 23). The two differ 16x, and
`OPROJ_KSPLIT_NULL.md` §6 already documents the discrepancy. Narrowing the
population does not recover the 33.93 for the same reason deleting the barrier
did not: the 104 early workers really do burn that time, but the 128 plateau
workers do not arrive any sooner.

**This is the fifth consecutive null the makespan rule has predicted in
advance.** Update the tally in `MAKESPAN_RULE.md`.
