# QKV epoch and split-KV chunk barriers on the AID replicas

Code: lane commit `21be579`. Opt-in, use all three together:
`MPK_AID_SPLIT_QKV=1 MPK_AID_QKV_ARRIVE=1 MPK_AID_CHUNK_BAR=1`.

The QKV epoch's arrival counters were one int per XCD packed into
`input_ptrs[7]`, i.e. eight dies' atomics on one shared line, which is
MTYPE_NC in SPX+NPS2. The split-KV chunk barrier counters had the same shape.
Both move to XCD-private 64 B slots on each AID's RW flag replica
(`g_aid_flag_rep`, `MPK_AID_QKVARR_BASE_INTS = 24576`,
`MPK_AID_CHUNKBAR_BASE_INTS = 24832`). Each counter is still touched only by
its own die, so no multi-XCD atomic lands on a replica line.

MI355X, perbo driver, 16-token output hash `3d54adb71e19` on every run; that
hash is the correctness gate. The 5,200-token outputs are not a gate: every
pair of long runs, including NPS2 against NPS1 on the same code, splits
around character 900 inside a repeated "**Answer:** Paris." loop, where
continuing or closing the turn is a near-tie. "31 chunks" means
`CK_FMHA_NUM_KV_CHUNKS=31`, the split a 5,200-token run compiles to, applied
to a 16-token run.

## End to end, `[FWD_PASS_TOTAL] avg_ms`

| run | mode | without | with | change |
|---|---|---|---|---|
| 16 tokens (8 chunks), clean, 3 alternated pairs | NPS2 | 1.463 / 1.463 / 1.462 | 1.445 / 1.440 / 1.438 | -1.5% |
| 16 tokens, 31 chunks, phase slots on | NPS2 | 1.654 / 1.643 | 1.526 / 1.522 | -7.6% |
| 5,200 tokens (31 chunks) | NPS2 | 1.676 | 1.554 | -7.3% |
| 16 tokens (8 chunks), clean | NPS1 | 1.534 / 1.527 / 1.526 | 1.469 / 1.533 (third run hung at launch) | noise |
| 16 tokens, 31 chunks, recorder on | NPS1 | 1.588 / 1.614 | 1.558 / 1.605 | noise |
| 5,200 tokens (31 chunks) | NPS1 | 1.658 | 1.593 | -3.9% (1 run each) |

Same code, both with the fix: NPS2 1.441 vs NPS1 ~1.50 at 16 tokens, and
1.554 vs 1.593 (-2.4%) at 5,200 tokens.

## Where it lands: QKV epoch wait per pass, ns (`MPK_QKV_SUB_LDS`)

| mode | 8 chunks without | 8 chunks with | 31 chunks without | 31 chunks with |
|---|---|---|---|---|
| NPS2 | 1,096 | **622** | 11,464 | **8,960** |
| NPS1 | 859 | 824 | 9,500 | 9,475 |

NPS1 does not move because those lines are RW and hardware-coherent there.
NPS2 goes from slower than NPS1 to faster at both chunk counts. At 31 chunks
every worker joins the epoch, so most of the remaining wait is arrival spread
from the previous layer, not the barrier.

Recorder: `MPK_QKV_SUB_LDS=1` with `MPK_PHASE_SLOTS=1 MPK_PHASE_LDS=1` prints
one `[QSUBW]` line per worker with drain / arrive / wait per QKV-epoch pass
and drain / arrive per chunk-barrier pass.


## Follow-ups: direct poll and flat tree barriers

Code: the commit after `c77ed24`. All opt-in; the recipe now adds
`MPK_QKV_EPOCH_DIRECT=1 MPK_P9_FLAT=1 MPK_OPROJ_FLAT=1` to the three flags above.

- `MPK_QKV_EPOCH_DIRECT`: waiters poll the replica arrival counter for
  participants x epoch, and the last arriver no longer bumps the epoch.
- `MPK_P9_FLAT`: each die's last Phase 9 arriver publishes 8 x layer into its
  own slot of both AID replicas; the gate waits for the minimum over the eight
  slots (`ld_aid_min8_s32`: eight loads, one wait). The cross-die `layer_global`
  atomic and the last-die relay store are gone.
- `MPK_OPROJ_FLAT`: the same for the O-proj phase-2 barrier (no level-2
  `hier_barrier[8*16]` atomic).

| arm (NPS2, 16 tokens) | runs | mean | vs previous |
|---|---|---|---|
| 21be579 flags | 1.444 / 1.441 / 1.441 | 1.442 | |
| + direct poll | 1.432 / 1.435 / 1.430 | 1.432 | -0.67%, t=-5.2, 3/3 |
| + direct (new session) | 1.439 / 1.438 / 1.434 | 1.437 | |
| + direct + P9_FLAT | 1.434 / 1.428 / 1.433 | 1.432 | -0.37%, t=-2.05, 3/3 |
| + direct + P9_FLAT + OPROJ_FLAT | 1.427 / 1.431 / 1.427 | **1.428** | -0.60%, t=-5.2, 3/3 |
| same, after an NPS1 round trip | 1.430 / 1.426 / 1.429 | 1.428 | |

Last-worker critical path per layer: 52.39 -> 52.03 us (O-proj + router
-0.11, Phase 9 arrival -0.06). At 31 chunks the direct poll is null (the wait
there is arrival spread); 5,200-token decode 1.554 (21be579) / 1.562 (direct) /
1.559 (all three), one run each.

NPS1, same code, 16 tokens: 1.530 / 1.513 / 1.502 -> 1.459 / 1.516 / 1.519
(t=-0.63, noise); its lines are hardware-coherent, so the removed hops cost it
less. NPS1 5,200-token decode with all three: 1.708, a uniform +0.12 ms at every
context against the earlier fix-only run -- one run each, unresolved.
