# SPX+NPS2: synchronization cost in Phase 7

> **Read this first.** Three separate synchronization designs were measured to
> win 3-10x in isolation and **none of them moved fleet**. Phase 7 sits at
> 1.05x of NPS1 and has ~0.03 ms/token left in it, which is inside the 7.5%
> noise floor. The end-to-end gap is elsewhere: MoE, +0.494 ms, 66% of the
> total. Sections below are kept because the mechanisms and the measurement
> traps are real, but do not read the standalone numbers as available wins.

## THE ARCHITECTURAL RESULT: NPS2 is not slower. SPANNING is.

`onaid.cpp`, SPX/NPS2, 184 blocks, event-counter wait:

| configuration | NPS2 | blocks | XCDs |
|---|---:|---:|---:|
| 8 XCDs, spanning line (**what fleet runs**) | **10,898 ns** | 184 | 8 |
| 4 XCDs 0-3, counter in AID0 (local) | 843 | 92 | 4 |
| 4 XCDs 4-7, counter in AID1 (local) | 837 | 92 | 4 |
| **8 XCDs, two independent per-AID groups** | **842** | **184** | **8** |
| NPS1, 8 XCDs, same code | 1,613 - 1,661 | 184 | 8 |

**Splitting the rendezvous into two 4-XCD groups, each synchronising only
within its own AID, gives 842 ns at FULL occupancy -- 12.9x better than the
spanning configuration and 1.9x faster than NPS1.** Verified by the counter
dump: each AID's counter reaches exactly 4 x iters and each XCD's arrival slot
exactly 23 x iters, with each AID's four slots populated and the other four
untouched.

Poller count explains only part of it: NPS1 also improves 8 -> 4 XCDs
(1,613 -> 841, 1.9x), so of NPS2's 12.9x, about 1.9x is fewer pollers and the
remaining ~6.8x is purely that the line stopped spanning.

**A cross-AID read of an AID-local MTYPE_RW line never completes** -- 242 ms
and 1.6M polls against a spin cap, not slow but broken, because RW is coherent
only inside its own AID. That is the hazard that forces spanning NC for
anything genuinely shared, and it is why "just make it AID-local" cannot be
applied to an 8-way gang barrier.

So the constraint for any redesign is structural, not placement: **the 8-XCD
gang rendezvous fleet is built around is the thing that cannot work in NPS2.**
Two independent 4-XCD groups with one explicit cross-AID handoff per layer is
the shape that does.

**Not yet demonstrated in fleet.** This is a microbenchmark where every block
waits every iteration. Three earlier standalone wins (AID-replicated barrier
flags, `MPK_OPROJ_AID_AGG`, `MPK_AID_EVCTR2`) all failed to transfer, so treat
842 ns as a reason to attempt the restructuring, not as an expected outcome.

## Corrections to earlier claims in this file

| claim | status |
|---|---|
| AID-replicated flags take the barrier to 0.99x of NPS1 | true **standalone**; in fleet Phase 7 went 1.16x -> 1.05x, worth ~0.024 ms/token |
| `MPK_OPROJ_AID_AGG` helps | **unproven.** Sign flips with instrumentation load: `bar` 2.60 with one printf, 2.80-2.88 with another, and 1.80 vs 1.60 on the leanest build |
| `MPK_AID_EVCTR` is worth -6.83% on op 7 | **does not reproduce.** 3 paired reps give t = 0.92 |
| `MPK_AID_EVCTR2` (monotone mirror) fixes the event wait | **1,085 ns standalone (9.6x better than shared, 1.5x faster than NPS1) but NEUTRAL in fleet**: t = 0.94, and ATT total stall 2.25x vs 2.22x unchanged |
| the event wait is producer arrival, not the read | **wrong.** ATT `Idle` is ZERO at every wait site; per-spin stall is 24x worse in NPS2, so it is read latency |
| `bar` is 83% of op 7's gap | that was 83% of the **inner** gap. Phase 7 inner is 28% of mask-128's time, so `bar` is ~3.6% of the total |

## The measurement that actually characterises it

ATT, clean builds, `MPK_ONLY_OP=128`, same flags, both modes
(NPS1 0.817 ms, NPS2 1.016 ms):

| | NPS1 | NPS2 | NPS2 + EVCTR2 |
|---|---:|---:|---:|
| total stall | 486M | 1,078M (2.22x) | 1,092M (2.25x) |
| `s_waitcnt` stall | 175M | 652M | 697M |
| `s_waitcnt` cyc/hit | **156** | **559** | 525 |
| `s_barrier` stall | 267M | 381M | 352M |
| `buffer_load_dwordx4` (MFMA weights) | 15.2M | 16.3M | 4.8M |

`s_waitcnt` and `s_barrier` are 81% and 19% of the increase; the weight stream
is unchanged, which is why every placement lever measured neutral. The single
hottest instruction (`s_waitcnt vmcnt(0) lgkmcnt(0)`, the event-dependency
poll) is 46x worse, but the **population average is 3.6x** -- and the
population is what sets runtime. Quoting the hottest site overstates it.

Per dependency check (74 checks in each trace):

| | NPS1 | NPS2 |
|---|---:|---:|
| spins per check | 631 | 349 |
| stall per check | 569,847 | 7,581,098 |
| **stall per spin** | **903** | **21,723** (24x) |
| idle per check | 35,615 | 15,837 |

NPS2 spins *fewer* times and still costs 13x more per check, with no idle. So
the cost is per-access latency on the polled line, not loop count and not
waiting for a producer.

## Why the monotone mirror does not help fleet

`evwait.cpp` / `evwait2.cpp`, SPX/NPS2, 184 blocks, and this is the cleanest
result in the file:

| design | wait | spins/iter |
|---|---:|---:|
| shared counter only | 10,457 ns | 1.2 |
| mirror + 1/64 fallback (fleet today) | 3,456 ns | 10.9 |
| **per-XCD monotone slots, no fallback** | **1,085 ns** | 2.2 |
| NPS1, shared counter | 1,586 ns | 2.1 |

Today's mirror has EIGHT XCDs publishing a running total to ONE slot with
unordered write-through stores, so a late store from an earlier XCD moves it
BACKWARDS; progress then depends on the periodic shared read, which is the
expensive one, and the saving cancels. Proof: stretching the fallback to
1/1024 costs 16,771 ns and 1/65536 costs 1,536,807 ns with 7,712 spins -- the
mirror is stale, not slow.

Giving each XCD its own slot makes every slot single-writer and therefore
monotone, and the fallback disappears: **1,085 ns, faster than NPS1 itself.**

It still does nothing in fleet (t = 0.94; ATT stall unchanged; `s_sleep`
triples from the extra spinning to sum eight slots). The isolation benchmark
makes every worker wait every iteration, which is not fleet's regime, and that
is the lesson: **an isolation benchmark can prove a mechanism and still
mispredict the system by an order of magnitude.**

## Original finding (standalone, still valid on its own terms)

Measured on mi355x-thor-2, gpt-oss-120b decode, patched amdgpu with
`aid_local_spx_nps2=1` and the 2 MiB VRAM interleave.

## Headline

Fleet's own barrier, extracted standalone at 184 blocks, 200 iterations:

| release path | NPS2 median | vs NPS1 |
|---|---:|---|
| shared spanning-NC flags | **3374 ns** | 1.93x |
| **AID-replicated, dual-published** | **1728 ns** | **0.99x** |
| NPS1 reference (same binary) | 1747 ns | -- |

**Replicating the release flags per AID takes the barrier from 1.93x of NPS1
to marginally faster than NPS1.** Nothing else in the barrier matters:
AID-homing the arrival counters and reducing the cross-AID atomics from 8 to 2
contributed nothing measurable on top (`fleetbar4`, three changes at once:
1948 ns -- *worse* than flags alone).

Harnesses: `fleetbar.cpp` (fleet's shape + skew knob), `fleetbar2.cpp` (flag
A/B), `fleetbar4.cpp` (per-level isolation + store-drain sweep).

## What this refutes

**Arrival skew is NOT the mechanism.** `fleetbar.cpp` injects per-XCD arrival
skew from 0 to 2000 ns. The NPS2-NPS1 delta is **constant at ~+1600 ns at
every skew level** -- 0 ns skew gives 1747 vs 3343, and 2000 ns skew gives
9944 vs 11325. The slopes are identical, so NPS2 does not amplify arrival
spread. Any explanation built on "the barrier bills the slowest die and NPS2
widens the spread" is wrong.

**The store drain is NOT the mechanism.** Fleet's `bar` spans `t1`->`t2`,
which includes `s_waitcnt vmcnt(0)` over each worker's O-proj output stores
into the shared row. Sweeping that from 0 to 4096 B per block moves the
barrier by 390 ns (3377 -> 3767), and by a similar 17% under AID placement.
Fleet's actual stores are far smaller. Negligible.

**Isolated collective benchmarks cannot see this.** `policy.cpp` measures the
*NPS2-aware* policies (HIER/DUAL/PEER), so it looked healthy in NPS2 and the
NPS1 column was never run. It also carries ~3 us of harness overhead -- its
`null` policy costs 3034 ns in NPS1, more than fleet's entire barrier -- so
its absolute numbers were never comparable to fleet's.

## Barrier policy matrix (192 workers, ns, same binary both modes)

| policy | NPS1 | NPS2 |
|---|---:|---:|
| FLAT | 3090 | **13224** |
| HIERW | **2425** | 17315 |
| HIER | 2677 | 2493 |
| PEER | 2808 | **2421** |
| null (harness floor) | 3034 | 2995 |
| fleet's own barrier | **1747** | 3343 |

Three things follow. **FLAT scales fine in NPS1** (1101 -> 3858 over 8 -> 256
workers) and only collapses in NPS2 (1057 -> 15527), so HIER/PEER are repairs
for an NPS2-specific collapse rather than better barriers -- in NPS1 they are
*slower* than FLAT at low width. **Fleet's barrier at 1747 ns beats every
policy in NPS1**, so fleet's shape is already well tuned for NPS1 and swapping
in PEER would regress it. And **best-in-mode is a dead tie** (NPS1 HIERW 2823
vs NPS2 PEER 2788 at 256 workers), so the policies reach parity, never better.

## In-fleet result

`MPK_OPROJ_INNER_TIMING` splits Phase 7 into `mfma` / `bar` / `rmsnorm_router`
/ `topk` (us per layer per XCD, 25056 samples per arm, identical printf
perturbation in both modes):

| component | NPS1 | NPS2 ctl | NPS2 best | ratio |
|---|---:|---:|---:|---|
| mfma (O-proj GEMM) | 2.16 | 2.20 | -- | 1.02x |
| rmsnorm+router | 1.72 | 1.76 | -- | 1.02x |
| **bar** | **2.40** | **3.16** | **2.52** | **1.05x** |
| topk | 0.76 | 0.80 | -- | 1.05x |
| **Phase 7 total** | **7.12** | **7.96** | **7.28** | **1.02x** |

Compute is at parity; the barrier carried the whole inner gap and is now
within 5%. Best arm is `MPK_AID_SPLIT_HIER_LOCAL=1 MPK_OPROJ_AID_AGG=1` on
top of `MPK_AID_SPLIT_FLAGS`.

**Partial application measures as noise.** Individually: `hier_local` alone
3.20, `AID_AGG` alone 2.80, narrow+`hier_local` 3.52, all against a 3.16
control. One remaining remote access on the path costs as much as all of them.
`MPK_NARROW_OPROJ_HIER` alone is 2.60 but **conflicts** with the combination
(2.92 together).

## Where the end-to-end gap actually is

Per-stage increments via `MPK_ONLY_OP`, ms/token:

| stage | NPS1 | NPS2 | gap | share |
|---|---:|---:|---:|---|
| op 7 + all sync (mask 128) | 0.795 | 1.034 | +0.239 | 32% |
| MoE adds (mask 384) | 0.594 | **1.088** | **+0.494** | **66%** |
| QKV + attention + merge add | 0.367 | 0.380 | +0.013 | 2% |
| **full model** | **1.756** | **2.502** | **+0.746** | |

QKV, attention and merge are at parity (1.04x). **MoE is 1.83x and carries
66% of the gap.** Phase 7's inner span is only 28% of mask-128's time, so the
barrier win above is ~0.024 ms end-to-end -- real, but under the noise floor.
The remaining 0.206 ms of op-7's gap is waits *outside* Phase 7: the
inter-layer span, QKV epoch barrier, and Phase 9 gate.

## Closed avenues -- do not re-run

**MoE weight placement.** Slots 17/18 read at 386 ns with `local=0/8`, and
AID placement makes it *worse*: replicate-both 5.420 ms, single-replica 6.981,
against a 2.502 baseline. Cause: routing is dynamic so every XCD reads the
whole weight, and pinning a 2260 MiB/layer streaming weight to one range caps
it at that range's HBM stacks (3529 vs 6346 GB/s). The interleave is correct
for these buffers. Confirms `ab69213` and the 97% L2-hit finding.

**Small intermediates.** The slot map shows `residual`, `workspace_f32`,
`swiglu_out`, `norm_weight` and the biases all at **+-1.5 ns skew**. Only
`o_acc_f32` (-28.6) and `moe_barrier` (-10.1) show anything, and 28 ns against
a 7.5% noise floor is unmeasurable. The note predicting these carry the AID
gradient is wrong.

## Measurement methodology

**`MPK_ONLY_OP` is a BITMASK, not an op index.** `(MPK_ONLY_OP & (1 << n))`
per op: 1 QKV, 3 attention, 5 merge, 7 O-proj, 8 MoE.
- `128` = op 7 only
- `384` = op 7 + MoE
- `426` = every op (same as omitting it)
- **Any mask without bit 7 hangs.** The router/TopK is fused into op 7's
  kernel and publishes `routing_indices` / `active_expert_ids` /
  `routing_ready`; without it every MoE worker waits forever. `0`, `7` and
  `99` all hang for this reason. A zero-op build is therefore impossible.

**The noise floor in SPX+NPS2 is 7.5%.** Three reps of the identical build:
1.191 / 1.283 / 1.208 ms. Any single-shot claim below ~8% is unmeasurable.
Require >= 3 paired reps with arms alternated, and report the control's own
spread.

**The noise is fleet's, not NPS2's.** MoE bench spreads 0.7-3.7% in NPS2 and
0.9-4.3% in NPS1; fleet spreads 1.6% in NPS1 and 7.5% in NPS2. Only that one
cell is noisy, which rules out every memory-side explanation.

**Use `bar` from `MPK_OPROJ_INNER_TIMING`, not `avg_ms`, for barrier work.**
25k samples per run makes it tight, and every arm carries the same printf
perturbation (which inflates the run ~12x -- avg_ms 30.6 vs a real 0.795).

## Traps hit, all real

- **Aliasing in my own harness.** `cnt[xcd*16]` and `rel[xcd*16]` on the same
  base are the SAME word, so the arrival counter overwrote the release flag.
  It completed at 50 iters and deadlocked at 200, and reported a 2190 ns
  baseline against the true 3377. Give every structure a distinct region.
- **Block-buffered stdout loses everything on timeout.** `grep`/`cut` in a
  pipe re-buffer, so a hung run appears to have printed nothing. Write to a
  file, or `stdbuf -o0`.
- **`pgrep -c` prints `0` AND exits non-zero**, so `$(pgrep -c -f x || echo 0)`
  yields `"0 0"` and a `= "0"` test never fires -- an infinite wait loop.
- **Barriers need all blocks co-resident.** At 256 threads/block a 184-block
  barrier deadlocks: spinning blocks hold the CUs unscheduled blocks need. The
  barrier state was provably complete (23/23 per XCD, global 8/8, release set)
  while 23 blocks never observed it. Launch bounds are a correctness
  constraint here, not tuning.
- **`my_xcd()` must read `HW_REG_XCC_ID`**, not assume `blockIdx.x % 8`. A
  probe that counted the assumption validated nothing (it reported a perfect
  23-per-XCD split tautologically). The assumption happens to hold here --
  0 of 184 workgroups disagree -- but verify, don't assert.
- **A Slurm handoff resets the driver to stock.** All `aid_local_*` params
  vanish and all 8 dies return to SPX/NPS1 while `uptime` still shows days.
  Re-run `~/nps1/spx_nps2.sh` before measuring.
