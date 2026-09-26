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

## Per-die arrival counters on AID-local NC memory (2026-09-26, opt-in)

Every WAIT in the recipe polls an AID-local replica, but three per-die
ARRIVAL counters still sat in AID0-homed torch tensors: the MoE W13->W2
per-expert counter (device-scope atomic, once per W13 tile), Phase 9's
`layer_local` and the O-proj tree barrier's `hier_local` (die-private,
no sc1). Measured per physical die (LDS recorder `MPK_MOE_LDS`,
`MPK_P9_ARRREC`, 16 tokens): arrivals from dies 4-7 cost ~90 ns more (MoE
354 vs 445 ns, Phase 9 237-265 vs 319-358 ns).

What did not work, and why:
- Counters on the AID-local RW replicas (`MPK_MOE_FLAT`, `MPK_AID_P9_LOCAL`):
  an atomic on an RW replica line costs ~350 ns from every die of its AID
  (RW lines are kept coherent inside the AID). 64 B vs 256 B slot spacing
  measured the same, so it is not false sharing.
- The first NC-local attempt read each counter's base pointer from a
  `__device__` array at the arrival. Every asm "memory" clobber forces that
  load again: a dependent ~80-120 ns load in front of the atomic. Base
  pointers are now loaded once per call.

`MPK_AID_NC_REP=1` allocates one 64 MiB hipMalloc; halves placement puts its
first 32 MiB in AID0 and the rest in AID1 (verified with `MPK_NC_PROBE`:
`sc0 sc1` load latency from every physical XCC at every 2 MiB step, 140-175
ns near vs 245-280 ns far, cut exactly at 32 MiB). `MPK_MOE_NCLOCAL`,
`MPK_AID_P9_NC` and `MPK_OPROJ_HIER_NC` put the counters in the die's own
half (`MPK_NC_*_INTS` in mpk_atoms.cuh). Per die: MoE 375 / 376 ns, Phase 9
250-279 / 248-269 ns -- the AID1 penalty is gone.

Timing (NPS2, 16 tokens, A/B/A/B/A): Phase 9 alone 1.425 / 1.425 vs 1.426 /
1.430 / 1.428 ms (-0.21%, just outside the control range); Phase 9 + O-proj
1.425 / 1.425 vs 1.427 / 1.425 / 1.426, no result. That is the expected
size: each counter saves <= ~80 ns per layer, and only when an AID1 die is
the last arriver, i.e. <= 0.2% of the token. The MoE build is ~0.8% SLOWER,
but a run of the same binary with the counters left in place
(`MPK_AID_NC_REP_NULL=1`) is just as slow (1.445 vs 1.428), and padding the
unchanged MoE kernel with 8-32 `s_nop` at entry (`MPK_MOE_NOPS`) costs the
same 1.3% at every size: the MoE kernel is sensitive to code changes near its
entry, not to where the counter lives. Screen MoE code changes against a
padded no-op build. All flags are opt-in; the default build is unchanged.

## Replica pointers as `__constant__`: `MPK_AID_REP_CONST` (default on, 2026-09-26)

`g_aid_flag_rep`, `g_aid_nc_rep`, `g_aid_ml_in` and `g_aid_ml_out` are filled
by the host before launch and only read on the device. Declared `__device__`,
the compiler reads them with VECTOR loads (it cannot prove them unclobbered
past the kernel's atomics and asm "memory" clobbers), and each read gets a
compiler-inserted `s_waitcnt vmcnt(0)`. The compiler cannot count loads issued
from inline asm, so that full wait also drains whatever asm DMA / prefetch is
in flight -- e.g. in the MoE's W2 prologue the replica-pointer arithmetic
drained the two scale DMA loads before the weight prefetch (~1,070 cycles per
call by ATT). As `__constant__` they become `s_load`s waited on lgkmcnt.
Static count: vector pointer loads in the fused layer 18 -> 0, O-proj 2 -> 0,
MoE 1 -> 0; full vmcnt(0) waits fused layer 100 -> 86, worker loop
558 -> 544, O-proj 26 -> 20, MoE 14 -> 12.

Measured (A/B/A/B/A, 16 tokens, NPS2): 1.431 / 1.429 / 1.426 -> 1.376 / 1.381
ms (-3.5%), hash green; 1k prompt at 31 chunks == torch 16/16, 3k at 24 ==
3k at 8; 16k decode 1.562 -> 1.514 ms with the text identical over all
43,433 chars. NPS1 (same build, single runs): 16 tokens 1.520 / 1.469 ->
1.445, 16k decode 1.675 -> 1.586. With it in both modes NPS2 leads by
4.4-4.8% at every context from 250 to 16k tokens and 4.6% at 16 tokens.

It also explains the MoE kernel's "code sensitivity": 8-32 `s_nop` at MoE
entry cost +1.3% (the asm block moved the compiler's pointer-load wait onto
the tile-0 weight burst: a `vmcnt(0)` of 2,229 cycles by ATT), and the MoE
NC-local counter build was +0.8% for the same reason. With the pointers
`__constant__` both are null (1.376 vs 1.376-1.379). `MPK_AID_REP_CONST=0`
restores the old declarations for A/B. `MPK_W2_NO_REFRESH` (opt-in) drops
the W2 poll's diagnostic 8-load refresh; measured null.

## QKV prologue fold made fully local: correct, NOT faster (2026-09-26)

The QKV prologue sums the MoE workspace's four slots (writer-split ring:
slots 0-1 homed on AID0, 2-3 on AID1) plus the residual row, so half its
reads cross the die in both modes, and QKV is the one critical-path segment
where NPS2 does not lead (4.90 vs 4.86 us/layer). All opt-in, all bit-exact
(16-token hash, 1k prompt at 31 chunks == torch, 3k at 24 == 3k at 8):

* `MPK_WS_FARCOPY` (needs `MPK_WSF32_AID`, `MPK_P9_FLAT`, `MPK_LAYER_RING`,
  `MPK_OWN_BO_RINGS=ws`; bs=1): W2 writes only its own AID's copy of the ring
  buffer and drops its 16 float4 into an LDS mailbox; after its own Phase 9
  arrival the worker's wave 1 stores them into the other AID's copy; the
  highest arriving rank re-poisons (0x7FBADBAD) the other AID's slots of the
  ring the layer read. The fold reads its own copy; a poisoned slot makes the
  per-thread sum of squares NaN, and that thread redoes its elements from the
  producer copies (a verbatim copy of the loop). `MPK_WSFC_STATS` counts
  copies (every W2 tile) and slow paths (~0.01% of fold iterations).
* `MPK_RESID_REP_RING`: `MPK_RESID_REP` switches to the O-proj's AID replica
  only when `input_ptrs[1] == output_ptrs[5]`, which is never true under
  `MPK_LAYER_RING` (residual = apo copy L-1, output = copy L) -- the flag has
  been INACTIVE since the ring shipped. RING uses the replica for every layer
  but a token's first.

Measured (A/B/A/B/A, NPS2, 16 tokens): far copy with a per-iteration poison
branch in the fold loop +2.3-2.8%; branch moved out of the loop +1.3%;
post-arrival barrier removed = parity (A 1.378 / 1.379 / 1.379 vs 1.381 /
1.384); far copy + residual replica, i.e. every fold load local, +0.4%
(A 1.381 / 1.378 / 1.378 vs 1.385 / 1.384). Fold locality does not buy time.
The earlier timing oracles (all-local -1.0%, one slab -1.3%) read FEWER
DISTINCT slots, which is what they measured. Per-die phase slots and the
inter-layer split (ret 1466 vs 1460 ns, dies 0-3 vs 4-7) show NPS2 is now
symmetric across dies.

Two rules this cost a run each to learn: never put a data-dependent branch
inside an unrolled load loop on a critical path (it waits for the loads
before the next ones issue; accumulate a flag and fix up afterwards), and
add no barrier between the Phase 9 arrival and the next layer's QKV weight
prefetch issue (it held waves 1-3's prefetch share behind wave 0's publish
drain). Timing probes kept opt-in: `MPK_TOPK_OWN_AID_PROBE` (TopK waits on
own-AID logits only, <= -0.4%), `MPK_QKV_WS_LOCAL_PROBE`, and the ablation
switches `MPK_WSFC_READ_PRODUCER` / `MPK_WSFC_NO_WRITERS`.
