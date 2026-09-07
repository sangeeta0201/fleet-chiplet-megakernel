# Where the residual +0.72 us in `bar` actually goes

`results/nps1-vs-nps2.md` left `bar` as the only real gap after the release-flag
fix: 1.76 us in NPS1 against 2.48 in NPS2, with non-overlapping p10/p90 ranges,
so a floor rather than noise. The guess recorded there was the level-2 global
arrival counter -- "the global arrival counter is inherently one line and half
the XCDs are always remote to it."

**That guess was wrong.** The barrier's memory protocol accounts for about
0.2 us of the gap. The rest is cross-XCD *arrival skew*: in NPS2 the eight XCDs
reach the barrier 0.98 us apart instead of 0.22 us apart, and a barrier costs
whatever the slowest arriver costs.

## How it was measured

`-DMPK_OPROJ_BAR_TRACE` takes five `s_memrealtime` ticks on tid 0 inside the
Phase 2 barrier and prints them *after* `done:`, alongside the existing
`[OPROJ_INNER]` line. Nothing is stored or printed inside the barrier, so the
thing being measured is not perturbed by the measurement -- the cost is wall
time and printf bandwidth, both outside the `t0..t4` window.

    _bt0  arrival instant: after the drain and the block rendezvous,
          immediately before the level-1 atomic
    _bt1  level 1 returned
    _bt2  level 2 returned          (only the one closer per XCD reaches this)
    _bt3  release store retired     (only the single global releaser)
    _bt4  this block's poll cleared

Three populations report, because three different sets of workers hold the
interesting ticks: `[BAR_OBS]` from tile 0 of each XCD, `[BAR_ARR]` from the
worker that closed level 1 on each XCD, `[BAR_REL]` from the one global
releaser. `obs` and `rel` are printed as raw ticks and joined on `layer_epoch`
in post-processing, because release-to-observation spans two different blocks;
`s_memrealtime` is device-wide, so that subtraction is meaningful across XCDs.

Driver: `trace_bar.sh`, and `trace_both_modes.sh` for the partition round trip.
150 layers, `--tiles=23`, first 30% of epochs dropped as warmup. NPS1 and NPS2
base are `--aid=0`; NPS2 fixed is `--aid=1 --coherent=1 --split=1`, the same
pairing the main table uses.

## The decomposition

`bar` splits exactly four ways -- `drain + l1 + wait + acq` -- where `wait` is
everything between a block publishing its own arrival and its poll clearing.
Medians over the eight XCDs' tile 0, microseconds:

| step | NPS1 | NPS2 fixed | delta | NPS2 base |
| --- | --- | --- | --- | --- |
| drain (`s_waitcnt` + `__syncthreads`) | 0.440 | 0.440 | +0.000 | 0.440 |
| l1 (own level-1 atomic) | 0.240 | 0.280 | +0.040 | 0.260 |
| **wait** (everyone else) | **1.000** | **1.840** | **+0.840** | 5.620 |
| acq (`__syncthreads` + `buffer_inv`) | 0.080 | 0.080 | +0.000 | 0.080 |
| **bar** | **1.800** | **2.600** | **+0.800** | 6.400 |

Every step a block performs itself is at parity. The entire gap is `wait`, which
is not a latency this block pays -- it is the latency of whoever arrives last.

## The protocol terms are all small, and all measured

| quantity | NPS1 | NPS2 fixed | NPS2 base |
| --- | --- | --- | --- |
| release -> observation, XCD 0-3 | 0.100 | 0.040 | 0.470 |
| release -> observation, XCD 4-7 | -0.050 | 0.140 | 0.720 |
| level-2 atomic (the closer's) | 0.360 / 0.240 | 0.520 / 0.520 | 0.360 / 0.360 |
| level-1 atomic, XCD 0-3 vs 4-7 | 0.280 / 0.200 | 0.200 / 0.320 | 0.200 / 0.320 |
| acquire | 0.080 | 0.080 | 0.080 |

Three things fall out of this table.

**The release-flag split did exactly what it was supposed to, and the effect is
now visible directly rather than inferred from a queueing slope.**
Release-to-observation is 0.47-0.72 us in base and 0.02-0.14 us once the flags
are replicated per partition -- an order of magnitude, and near the 10 ns tick
resolution afterwards. Negative values are real: they mean the poller's load
returned before the releaser's own post-store timestamp retired, which is what
"already waiting on a local line" looks like.

**Level 2 shows no AID asymmetry at all.** 0.520 for both halves in NPS2, 0.360
vs 0.240 in NPS1. The single global counter costs +0.16 us more per atomic in
NPS2 than NPS1, but it costs the same whichever partition the arriving XCD sits
in -- so the recorded hypothesis that "half the XCDs are always remote to it"
does not show up as a measurable penalty. Only +0.16 us of the +0.80 is here.

**Level 1 does show the expected asymmetry, and fixing it changes nothing.**
See below.

## Splitting level 1 was correct and worthless

`--lsplit=1` homes the eight per-XCD arrival lines per partition
(`hier_local_lo` / `hier_local_hi`). Unlike the release flags this needs no
replication: line `x` is only ever incremented by XCD x's own 23 workers, who
are co-located with each other by construction, so each line can simply live
where its only users are.

It works, in the narrow sense:

| | l1, XCD 0-3 | l1, XCD 4-7 | delta | bar |
| --- | --- | --- | --- | --- |
| `--lsplit=0` | 0.200 | 0.320 | +0.120 | 2.600 |
| `--lsplit=1` | 0.200 | 0.200 | +0.000 | 2.600 |

The cross-AID penalty on the upper half is gone, and the mean `l1` drops from
0.280 to 0.200. `bar` does not move, because the 0.08 us saved moves straight
into `wait`: 1.840 -> 1.920. Arriving sooner does not help when you then wait
for someone else.

This is the general shape of the whole result. **Only one of the 184 level-1
atomics is on the critical path** -- the last one on the last XCD, the one that
gates level 2. Speeding up the other 183 is free and buys nothing. The gate
(`gate_lsplit.sh`) confirms all three configurations still agree bit-for-bit on
both hashes, so the split is sound; it is just not load-bearing.

## What the gap actually is

`obs` is absolute, so each XCD's arrival instant relative to the release is
`(obs - rel) - wait`. Per-XCD medians, microseconds before the release:

| | XCD 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | spread |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NPS1 | -0.98 | -0.96 | -0.84 | -0.92 | -1.01 | -1.06 | -0.99 | -1.04 | **0.22** |
| NPS2 fixed | -2.10 | -2.02 | -2.07 | -1.95 | -1.19 | -1.28 | -1.12 | -1.25 | **0.98** |

Arrival skew grows from 0.22 us to 0.98 us: **+0.76 us against a measured `bar`
delta of +0.80 us.** That is the residual, essentially all of it.

The releaser identity says the same thing independently. The releaser is by
definition on whichever XCD closed level 1 last, so its distribution over 105
sampled layers is a direct read on which XCD is slowest:

    NPS1        0:17  1:4  2:52  3:20  4:1  5:1  6:9  7:1
    NPS2 fixed              4:25            6:80
    NPS2 base               4:84  5:21

In NPS1 the role rotates across all eight XCDs -- no die is systematically last.
In NPS2 it pins to XCD 6, with XCD 4 second, and with `--lsplit=1` it pins
harder still (6:97). The upper half of the XCDs is consistently late, and the
lower half spends 2.1 us waiting for them.

## What this means for the next move

The barrier itself is close to done. Adding up every protocol term that is
genuinely worse in NPS2 -- level 2 at +0.16, level 1 at +0.04, release and
acquire at zero -- leaves roughly **0.2 us of headroom in the barrier**, and the
two obvious structural fixes are both spent: the release flags are already
replicated, and splitting level 1 demonstrably returns nothing.

The remaining ~0.6 us is a load-balance problem, not a synchronization problem.
Something upstream of the barrier makes XCDs 4-7 -- and XCD 6 in particular --
finish Phase 7's earlier stages later than XCDs 0-3, and `bar` is just where the
cost surfaces, because a barrier is where imbalance becomes visible. Note that
`slicewait` and `mfma` are at exact median parity between the two modes, so this
is not a slowdown in the earlier stages; it is a widening of their *spread*,
which a median cannot see.

## Phase 7 inherits the skew, it does not create it

`[BAR_OBS]` now also carries `t0`, the absolute tick at Phase 7 entry, so the
stagger at the door can be separated from the stagger at the barrier. Both are
measured against `rel`, the one tick per layer that every XCD shares, so the two
spreads are directly comparable.

| | entry spread | arrival spread | created inside Phase 7 |
| --- | --- | --- | --- |
| NPS1 | **0.09** | 0.20 | +0.11 |
| NPS2, flags split | **0.82** | 0.90 | +0.08 |
| NPS2 base | 11.04 | 6.19 | -4.85 |

That settles it. In NPS1 the eight dies enter Phase 7 within 90 ns of each other
and leave it within 200 ns. In NPS2 they enter **0.82 us apart**, and Phase 7 adds
0.08 us of its own -- less than NPS1's 0.11. Per-XCD `sw` and `mf` are flat across
all eight dies in both modes (0.36-0.40 and 3.92-3.96 us), so Phase 7's own work
is balanced to the tick.

The barrier was never the problem. It is the instrument that makes an imbalance
created before it ran visible, and no restructuring of it can recover work that
was already lost. The NPS2-base row is the same statement from the other side: its
entry spread is 11 us and the barrier *removes* 4.85 us of it.

## What was upstream: the inter-layer rendezvous

The only thing between one layer's barrier and the next layer's Phase 7 entry is
the harness's own per-layer rendezvous, the one standing in for phase 9. It was
flat:

```c
atomicAdd(bar, 1u);                 // every one of the 184 blocks
if (dual) atomicAdd(bar_b, 1u);     // ... into both replicas
while (ld(my_bar) < layer * NXCD * tiles) sleep();
```

Replicating the two counters fixed the *polling*, which is why `bar` fell from
6.36 to 2.44 us, and left the *arrivals* untouched: 184 blocks x 2 replicas is 368
atomics on two lines per layer, and for 92 of those blocks one of the two is a
read-modify-write across the partition boundary. The two replicas therefore cross
their thresholds at different times, which releases the two halves at different
times, which is exactly the AID-aligned entry stagger measured above.

So the tree belongs one level up from the barrier, on the rendezvous. `--hrdv=1`:

- **Level 1**, per XCD, AID-local. 23 blocks, one line, co-located by
  construction -- partitioned, not replicated, the same argument as `--lsplit`.
- **Level 2**, per half, AID-local. That half's 4 level-1 closers, one line.
- **Level 3**, the handshake. Each half's closer signals the *other* half with a
  single write-through store into a slot homed with its reader, then polls its own
  slot locally, then releases its own half's 92 blocks locally.

Critical-path boundary crossings per layer: **1 store, down from 184 remote
atomics**, and no read-modify-write crosses at all.

Every line is indexed by half as well as by placement. That is load-bearing: it
keeps the topology a genuine 4/4 split when both region pointers alias one buffer,
which is how NPS1 and `--aid=0` run it. Indexing by placement alone would have all
eight XCDs share one level-2 counter and release on the fourth arrival instead of
the eighth -- silently, since the counters are monotonic and tested on residue.

### What it bought

| config | `bar` p50 | entry spread | arrival spread |
| --- | --- | --- | --- |
| NPS1 | 1.72-2.00 | 0.09 | 0.20 |
| NPS2 stock, flat rendezvous | 6.36 | 11.04 | 6.19 |
| NPS2 flags split, flat rendezvous | 2.44 | 0.82 | 0.90 |
| NPS2 flags split, **tree rendezvous** | **2.36** | **0.58** | **0.67** |
| NPS2 one domain, tree rendezvous | 5.16 | -- | -- |

The entry stagger drops 0.82 -> 0.58 us and the arrival spread follows it down
0.90 -> 0.67, which is the causal claim confirmed: the rendezvous was manufacturing
about a third of the skew, and removing its cross-AID arrivals removes that third.
The `bar` spread across XCDs narrows correspondingly, from 2.08-2.76 to 2.24-2.56,
and the releaser stops pinning to XCD 6 (6:70 -> 5:44 6:46).

`bar` p50 itself only moves 0.08 us, because 0.24 us of recovered skew is shared
out among eight dies and only the last arriver is on the critical path. The
last-arriver column is where it shows up: 2.76 -> 2.56.

The one-domain row is the shape-versus-placement control. The tree with no AID
placement at all is worth 1.2 us over stock (6.36 -> 5.16), so the fan-in matters
most when the placement is bad, and the two fixes overlap heavily: once the flags
are replicated, most of what the tree would have saved is already saved.

## Splitting level 2 was correct, and a net loss. The premise was wrong.

The reasoning that motivated this: the level-2 arrival counter is
`hier_barrier[8 * HIER_STRIDE]`, which lives in the plain `hipMalloc`'d
`counters` buffer, so SPX+NPS2 demotes it to MTYPE_NC while NPS1 keeps it
MTYPE_RW. It measures 0.56 us against NPS1's 0.30. It cannot be replicated -- it
is the one place all eight dies aggregate, so a coherent MTYPE would have readers
in both partitions -- but it can be *replaced*, by per-half counters plus the same
two-slot handshake the rendezvous uses. Predicted recovery: most of that 0.26 us.

`--bsplit=1` does exactly that, `gate_bsplit.sh` confirms all five configurations
still agree on both hashes, and it is **0.40 us slower**:

| | level 2, lo / hi | wait | bar |
| --- | --- | --- | --- |
| shared counter | 0.560 / 0.680 | 1.720 | **2.440** |
| split + handshake | 0.520 / 0.640 | 2.080 | **2.840** |

Two things in that table, and both are worth more than the change was.

**The atomic barely moved: 0.56 -> 0.52.** Not 0.26 us, 0.04. Moving the line into
an AID-local buffer with a coherent MTYPE bought essentially nothing, which says
the MTYPE was never what made it cost 0.56. A cacheable line helps a *poll*,
because the second reader can be served from cache. It does nothing for a
read-modify-write, which has to reach the coherence point whether or not anyone
may cache the result. That is the whole difference between this change and the
release-flag fix: the flags are polled 184 times a layer and replicating them was
worth 4 us, while this line is atomically incremented 8 times and never polled at
all. The +0.120 lo-versus-hi asymmetry also survives the move unchanged, so it is
a property of where the coherence point sits, not of where the line is homed.

**The handshake cost 0.36 us, all of it in `wait`.** A shared atomic counter
delivers the global fact -- "all eight dies have arrived" -- in the return value
of a single read-modify-write: one round trip. A two-slot handshake needs two
dependent ones, my store getting out and then the peer's store being observed by
my poll. Two dependent trips beat one only if the single trip costs more than
twice as much, and 0.56 against 0.30 is not that. Busy-polling instead of
`s_sleep` recovers 0.08 of it (`-DMPK_OPROJ_L2_BUSY_POLL`), which rules out poll
granularity as the explanation and leaves the structure.

So the single shared counter is not a wart that survived because nobody split it.
It is the cheapest available mechanism for a global fact, and the kernel comment
that stops at "it cannot be replicated" was right for a better reason than it
gives. Kept behind a default-off flag as the record of what the MTYPE fix does and
does not reach.

## What is left

Every step a block performs itself is now at or better than NPS1: `drain` 0.44
against 0.48, `l1` 0.24 against 0.26, `acq` at parity. AID-local placement is the
faster arrangement, exactly as the absence of an interconnect hop predicts. The
whole of the remaining +0.64 is `wait`, and it is two things:

- **~0.47 us of entry skew**, still arriving from upstream after the rendezvous
  tree took 0.24 off. Entry spread is 0.56 against NPS1's 0.09.
- **~0.26 us of level-2 atomic**, which the section above establishes is not
  reachable by placement. Eight serialized read-modify-writes is already the
  minimum for aggregating eight dies, and the semantics forbid skipping the
  aggregation: RMSNorm needs the whole 2880-element row, which spans all eight.

Only the first is addressable, and the remaining candidate is the one the `drain`
column has been pointing at all along:
the upper half pays +0.08-0.12 us to retire its O-proj stores into
`attn_proj_out`, which is a single hipMalloc'd buffer, single-homed the way
`counters` was. Unlike the barrier flags it is large enough for placement to
matter, and unlike them it is written every layer by all eight XCDs.

That is the next thing to place per AID. The barrier itself has ~0.2 us of
protocol headroom left and is not worth further work.

## Reproducing

```bash
bash build_gate.sh '' drive_phase7
bash gate_lsplit.sh                     # correctness of the level-1 split
bash gate_hrdv.sh                       # correctness of the tree rendezvous
bash build_gate.sh '-DMPK_OPROJ_BAR_TRACE' drive_phase7_trace
bash skew_hrdv.sh                       # entry skew, tree rendezvous on and off
bash trace_bar.sh fix      --aid=1 --coherent=1 --split=1 --lsplit=0
bash trace_bar.sh fixlocal --aid=1 --coherent=1 --split=1 --lsplit=1
sudo -n python3 run_root.py set_mode2.sh NPS1
bash trace_bar.sh nps1 --aid=0
sudo -n python3 run_root.py set_mode2.sh NPS2
```

`trace_both_modes.sh` does the last four steps with the NPS2 restore in an
`EXIT` trap, so a failure part way through does not leave the node in NPS1.
