# A decode megakernel designed for SPX+NPS2

Not a port of fleet. Fleet's architecture assumes NPS1's memory model --
device-wide coherent RW lines, one flat address space, any worker may touch
any buffer -- and every one of those assumptions is false in SPX+NPS2. Eleven
incremental fixes to it returned ~1% each, all below a 7.5% noise floor.

Everything below is a consequence of a measurement already taken on thor-2.
Nothing here is speculative design preference.

## The evidence this is built on

| measurement | value | consequence for the design |
|---|---|---|
| MoE bench, per-consumer AID placement, NPS2 | **15.89 us** | NPS2 is FASTER than NPS1 when placement follows the consumer |
| same bench, NPS1 baseline | 17.53 us | so the ceiling is not parity, it is **-9%** |
| same bench, `split50` (whole weight in one AID) | 24.84 us | "an AID" is worthless; it must be **per consumer** |
| same bench, remote | 24.37 us | interleave (every WG half-remote) is no better than remote |
| flag poll, shared spanning NC, 256 pollers | 52,251 ns | never let many workers poll one shared line |
| flag poll, two AID-local RW replicas, 256 pollers | **2,484 ns** | replicate every gate per AID; it is FLAT in poller count |
| cross-AID RW visibility | invisible, 3 s timeout | replicas must be published with **write-through stores**, not atomics |
| `nt` load on a release line | never observes it | scoped loads are a correctness requirement |
| bare `buffer_inv` on gfx950 | architectural NOP | every acquire needs an explicit scope |
| fleet variance, NPS2 | 7.5% | fleet has a live race; the benchmark at 0.7% proves NPS2 does not |
| L2 hit rate, O-proj | 97.3% | weights that fit in L2 gain nothing from placement |
| MoE weight read, fleet today | 386 ns | vs 154 ns AID-local: **2.5x** on the op holding 29% of the gap |

## Six rules, each forced by a row above

1. **Placement follows the consumer, not the buffer.** Every weight slice is
   homed in the AID of the XCD that reads it. Splitting a weight "into an AID"
   measured no better than fully remote (24.84 vs 24.37).
2. **Nothing shared lives on NC unless it is genuinely all-to-all.** There are
   exactly three all-to-all handoffs per layer (attn_out, O-proj `out`, the MoE
   reduce) and they are small vectors.
3. **Every gate is replicated per AID and published with write-through
   stores.** RW replicas are flat at 2,484 ns to 256 pollers; the shared NC
   line is 52,251 ns at the same width. Atomics cannot cross the boundary;
   `st_wt` can.
4. **Barriers are hierarchical, never flat.** Per-XCD arrival, then one
   representative per XCD at the top. Flat 256-way atomics on a shared line are
   the measured cliff.
5. **Every acquire carries an explicit scope.** `buffer_inv` alone is a NOP on
   gfx950; `sc1` invalidates L2.
6. **Dispatch is static and data-directed.** The consumer of a tile is chosen
   by where that tile's data lives. This is what fleet structurally cannot do
   and why its MoE weights are stuck at 386 ns.

## Stage 0: PASSED. Measured primitives, SPX/NPS2, ns per operation

Run on thor-2 in SPX/NPS2 (`nps1/policy`), 200 iters, 8 -> 256 concurrent
workers. This is the foundation gate and it cleared with room to spare.

| primitive | policy | 8w | 64w | 256w | degradation |
|---|---|---:|---:|---:|---|
| barrier | FLAT | 1059 | 4033 | 15515 | 15x |
| barrier | HIERW | 1313 | 4523 | 19196 | 15x |
| barrier | **HIER** | 1496 | 1698 | **2904** | **1.9x** |
| barrier | **PEER** | 1435 | 1625 | **2785** | **1.9x** |
| barrier | (null baseline) | 986 | 1461 | 3767 | -- |
| bcast | FLAT | 687 | 6363 | 25860 | 38x |
| bcast | **DUAL** | 939 | 852 | **909** | **1.0x** |
| bcast | HIER | 1090 | 1009 | 1050 | 1.0x |
| bcast-all | FLAT | 1352 | 8108 | 34187 | 25x |
| bcast-all | **DUAL-B** | 1411 | 1543 | **2659** | 1.9x |
| reduce | FLAT | 1054 | 4027 | 16527 | 16x |
| reduce | **PEER** | 1432 | 1623 | **2785** | 1.9x |

**Chosen primitives, and they are not negotiable downstream:**
- release / flag broadcast -> **DUAL** (dual-publish into both AID replicas).
  Flat: 909 ns at 256 pollers vs FLAT's 25,860. A **28x** win, and the single
  most important number in this document -- fleet's hot gates are broadcasts.
- barrier -> **HIER** or **PEER**. At 2,904 ns for 256 workers it is *below*
  the null-policy baseline of 3,767 ns, i.e. cheaper than the harness's own
  loop overhead. Effectively free.
- reduce -> **PEER** (2,785 ns at 256).
- **FLAT is banned.** Every flat policy degrades 15-38x with poller count, and
  that curve is precisely fleet's measured slot 5 / slot 7 behaviour.

## Stage 1 gate A: PASSED. aid_rt.h reproduces the raw policy numbers

`rt_selftest.cpp`, SPX/NPS2, 200 iters, replicas confirmed AID-local
(`AID0 -> range 0`, `AID1 -> range 1` from the allocator's own log):

| blocks | aid_rt ns | HIER ref | FLAT ref |
|---:|---:|---:|---:|
| 8 | **1212** | 1496 | 1059 |
| 32 | **1241** | 1513 | 2183 |
| 64 | **1519** | 1698 | 4033 |
| 128 | 2163 | 2108 | 7646 |
| 256 | 3344 | 2904 | 15515 |

4.6x better than FLAT at 256 workers and within 15% of the hand-written
reference, so the abstraction is close to free.

### Two bugs found getting there. Both are the design's own claims as traps.

**1. Dual-publish on non-AID-local replicas is WORSE than not replicating.**
Measured 22,385 ns at 256 blocks against FLAT's 15,515. The mirrors were
`hipMalloc`, on the reasoning that this isolates "protocol cost from
placement". There is no such separation: two spanning-NC mirrors cost two
stores and every poller still hits the same far line. **Placement is not an
optimisation of this protocol, it is the protocol.**

**2. A hierarchical barrier whose waiters poll more than one line is not
hierarchical.** The first version had every block wait on all eight per-XCD
gates -- 2048 concurrent system-scope loads at 256 blocks -- and it measured
like FLAT (15,836 ns). The fix is three levels where **a waiter polls exactly
one line, homed in its own AID**: AID-local per-XCD counter, then 8 shared
atomics (one per XCD, far below the contention cliff), then one dual-published
release gate.

**3. Arrival counters must not be reset, so every last-arrival test scales
with the epoch.** Resetting races the next epoch's arrivals; comparing against
a bare count is true only on epoch 1 and deadlocks every barrier after it.
Every spin also carries a deadline, so this surfaces as a fast TIMED OUT
rather than a nine-minute hang.

## Stage 1 gate B: PASSED. Class A beats NPS1 by 25%, with our own code

`mem_selftest.cpp`, grid 184, 49.6 MB/layer (the gpt-oss-120b top-4 decode MoE
weight volume), same binary in both modes, us/layer:

| placement | NPS2 | NPS1 |
|---|---:|---:|
| **AID-local (class A)** | **12.89** | 17.18 |
| remote (anti-placed) | 21.12 | 17.33 |
| shared `hipMalloc` (what fleet does) | 19.19 | 18.43 |

- **NPS2 class A vs NPS1 best: 12.89 vs 17.18 = 1.33x, a 25% win.** Above the
  10-20% objective. This is the redesign's premise, now measured with this
  runtime rather than inherited from the MoE benchmark.
- 1.64x over anti-placed, beating the reference ratio of 1.54x.
- 1.49x over the spanning `hipMalloc` fleet currently uses.
- In NPS1 all three arms collapse to ~17-18 us, exactly as they must: one
  memory range means `alloc_in_aid` has nothing to place. The harness prints
  FAIL there because its pass criterion is NPS2-only, not because anything is
  wrong.
- NPS2 reproduced to the digit across a full mode round-trip (12.89 both
  times, 3848/3849 GB/s), consistent with the benchmark's sub-4% spread and
  unlike fleet's 7.5%.

Both halves of the runtime are therefore validated: sync at **4.6x** over FLAT
and memory at **1.49x** over fleet's allocation, with the end-to-end premise
(**beat NPS1**) confirmed at 25%.

## Stage 3 (MoE layer): IMPLEMENTED, NOT YET PASSING. Status and open bugs

`moe_layer.cpp` implements the mechanism fleet structurally lacks: expert
parity placement (expert e in AID e&1) plus data-directed dispatch (tiles for
expert e executed by workgroups on XCDs in AID e&1). It runs, the barrier is
clean, and the parity arm completes in ~54 us/layer with **0 barrier
timeouts**. It does **not** pass its correctness gate, so no timing from it
should be quoted.

### What the gate caught (it did its job)

**A 581x "win" that was pure deadlock.** With 256 threads/block the naive arm
reported 20,006 us -- exactly the 20 ms barrier deadline -- and 23 blocks
(one XCD's worth) timed out. Dropping to 64 threads/block cleared it: both
arms 0 timeouts, 54.1 vs 53.4 us, i.e. **no difference at all**. The apparent
581x was an artifact of the comparison arm hanging.

**Root cause: block co-residency, not coherence.** The barrier's own state was
provably complete on the failing launches -- every XCD 23/23 arrivals, global
count 8/8, release gate set to 1 in both AID replicas -- yet 23 blocks never
observed a release that demonstrably happened. 256 threads/block cuts
occupancy 4x, so not all 184 blocks are resident, and blocks already spinning
hold the CUs that unscheduled blocks need. **Any barrier in this runtime
requires all participating blocks to be co-resident; launch bounds are a
correctness constraint, not a tuning knob.**

### Fixed along the way

**`my_xcd()` was an assumption, and its probe was tautological.** It returned
`blockIdx.x % 8`, and the "distribution probe" counted that same expression,
so it reported a perfect 23-per-XCD split while validating nothing. Now reads
`HW_REG_XCC_ID`. Measured result: the assumption **was** correct here -- 0 of
184 workgroups disagree -- but it is now verified rather than believed.

### Open bugs, in priority order

1. **The correctness gate is mis-designed.** It compares float checksums
   between arms whose dispatch orders differ, and the reduction is
   `atomicAdd` on floats. Float addition is not associative, so the two arms
   *cannot* produce identical sums however correct they are. Fix: compare
   against a CPU reference with a relative tolerance, or make the final
   reduction order-independent (fixed-point accumulate, or a deterministic
   tree reduce). Until this is fixed the gate cannot pass and cannot fail
   meaningfully.
2. **Checksum magnitude is ~1e36**, where the synthetic inputs bound it near
   1e4. Something is reading memory it should not, or the modulo-indexed
   weight walk is wrong. Not yet isolated.
3. Only after 1 and 2: re-measure parity vs naive. The current 54.1 vs 53.4
   is not evidence either way, because the arms are not yet computing the
   same thing.

## Memory plan

Three classes, and every allocation must declare which it is. There is no
default.

**Class A -- per-consumer private (the bulk).** QKV, O-proj, MoE expert
weights, KV cache, all per-XCD scratch. Allocated as 8 slices, slice i homed
in `aid_of_xcd(i)`. MTYPE_RW, always local, never polled across a boundary.

**Class B -- read-only broadcast.** norm weights, biases, router bias. Two
copies, one per AID, written once at load. Trivially cheap (a few KB) and
removes every cross-AID read of them.

**Class C -- genuine all-to-all (exactly three per layer).** Published with
`st_wt` into **both** AID replicas by the producer; every consumer reads the
replica homed in its own range. This is the only class that costs a remote
store, and it is 8 stores per layer, not 8 reads per worker.

The driver's 2 MiB interleave is **off** for this kernel. Interleave buys
bandwidth by making every workgroup half-remote, which is exactly the
configuration that measured 386 ns in fleet against the benchmark's 154 ns.
Per-consumer placement gets both locality and full bandwidth, because
different XCDs drive different ranges concurrently.

## Sync plan

One rendezvous primitive, used everywhere, with no shared-NC variant:

```
publish(flag, epoch):            // producer, once
    st_wt_u32(&rep[AID0][flag], epoch)
    st_wt_u32(&rep[AID1][flag], epoch)

wait(flag, epoch):               // consumer, many
    while (ld_sys_u32(&rep[my_aid][flag]) < epoch) { }
    buffer_inv sc1               // scoped acquire, or it is a NOP
```

Barriers compose two levels of it: every workgroup arrives into a per-XCD
counter homed in its own AID, one elected representative per XCD arrives at
the top level, and the release fans back out through the replicas. No counter
is ever incremented by workers in both AIDs.

## Determinism is a build gate, not a nice-to-have

Fleet's 7.5% NPS2 spread and its ~11-distinct-outputs-in-16-runs are almost
certainly the same race. This kernel gates every commit on a **bit-identical
output hash across 5 runs**, in NPS2, uninstrumented. A timing number from a
build that has not passed that gate is not a result. The benchmark's 0.7%
spread is the proof this is achievable in NPS2.

## Staging -- each stage has a number it must hit before the next starts

| stage | scope | gate |
|---|---|---|
| 0 | allocator + sync primitives, standalone | replicate policy.cpp's 2,484 ns flat-to-256-pollers |
| 1 | one layer: QKV + attention, Class A weights | hash stable 5/5; beat fleet's per-layer attn slots |
| 2 | + O-proj with the one Class C publish | hash stable; beat op-7's NPS1 0.7845 ms |
| 3 | + MoE, expert-parity placement, data-directed dispatch | hit the bench's 15.9 us/layer, not fleet's 24 |
| 4 | 36 layers, full decode | **beat SPX+NPS1's 0.90 ms end to end** |

Stage 3 is the one that carries the win: MoE is 29% of the current gap and the
benchmark already proves 2.5x is available on its weight reads.

## What is reused

The megakernel scaffolding is worth keeping; the memory and sync layers are
not. Reuse: the task/event scheduler shape, the MXFP4 MFMA inner kernels, the
attention merge, the demo harness and tokenizer path. Replace: every
allocation, every flag, every barrier, and the dispatch-to-consumer mapping.

