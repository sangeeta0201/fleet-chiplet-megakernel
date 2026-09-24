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

