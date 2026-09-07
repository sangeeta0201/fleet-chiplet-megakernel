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

Two things worth checking next, in order:

1. Whether the skew is already present at Phase 7 entry (`_op_t0`) or is created
   inside `slicewait`/`mfma`. One more absolute tick at `_op_t0` per XCD settles
   it, and it is the difference between "the barrier inherits imbalance from
   upstream" and "Phase 7 creates it."
2. Why XCD 6 specifically. The `drain` column already shows the upper half
   paying +0.08-0.12 us to retire its O-proj stores into a hipMalloc'd
   `attn_proj_out`, which is single-homed like `counters` was -- so the output
   buffer's placement is the next candidate, and unlike the barrier flags it is
   large enough for placement to matter.

## Reproducing

```bash
bash build_gate.sh '-DMPK_OPROJ_BAR_TRACE' drive_phase7_trace
bash gate_lsplit.sh                     # correctness of the level-1 split
bash trace_bar.sh fix      --aid=1 --coherent=1 --split=1 --lsplit=0
bash trace_bar.sh fixlocal --aid=1 --coherent=1 --split=1 --lsplit=1
sudo -n python3 run_root.py set_mode2.sh NPS1
bash trace_bar.sh nps1 --aid=0
sudo -n python3 run_root.py set_mode2.sh NPS2
```

`trace_both_modes.sh` does the last four steps with the NPS2 restore in an
`EXIT` trap, so a failure part way through does not leave the node in NPS1.
