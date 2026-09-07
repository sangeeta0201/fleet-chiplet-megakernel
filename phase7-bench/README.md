# Phase 7 NPS1 vs NPS2 reproducer (MI355X, GPT-OSS-120B decode)

A standalone benchmark that reproduces the decode regression seen when an MI355X
is switched from NPS1 to NPS2, isolated down to a single phase of the fleet
chiplet megakernel.

It calls the **real** `gang_linear_mxfp4_res_bias_rmsnorm_topk_kernel` device
function with the model's exact template instantiation, so the timing buckets
come out of the production code path rather than a reimplementation that might
miss the effect. A run takes about 300 ms and needs no model weights.

Base commit: `9d478ea` on `amd_mi355_gpt_oss120b`. Every number below was
measured on that tree; the probe patch in this directory is already applied to
the kernel header on this branch, guarded by `MPK_OPROJ_INNER_TIMING` so it is
inert unless you define it.

## The regression being chased

| measurement | NPS1 | NPS2 | ratio |
| --- | --- | --- | --- |
| GPT-OSS-120B decode, tree `9d478ea` | 1.532 ms/iter | 10.378 ms/iter | 6.8x |

Narrowing, before this benchmark existed:

1. **Weight placement is not the cause.** A three-arm sweep pinning all weights
   to one AID measured 9.32 ms (AID0), 10.43 ms (allocator-scattered) and
   12.82 ms (AID1). The full range is 3.5 ms against an 8.9 ms gap, and the
   "spread across both AIDs" arm sits in the middle rather than at the good end.
2. **One phase dominates.** The kernel's 12-slot `MPK_PHASE_SLOTS` profiler puts
   68% of the regression in slot 6 (O-proj + RMSNorm + router + TopK) at 7.9x,
   plus 16% in slot 7. QKV GEMM is flat at 0.97x, which rules out weight
   bandwidth.
3. **Inside that phase it is synchronization, not compute.** Splitting Phase 7
   five ways attributes 44% of the regression to the cross-XCD attention slice
   wait (27x) and 16% to the hierarchical barrier (3.8x). The MXFP4 GEMM is
   3.8x and only 9%.

## What this benchmark establishes

Reproduced standalone, medians in microseconds per layer:

| bucket | bench NPS1 | bench NPS2 | bench ratio | model ratio |
| --- | --- | --- | --- | --- |
| `slicewait` (cross-XCD attn) | 0.36 | 9.68 | **26.9x** | 27.4x |
| `bar` (hierarchical barrier) | 1.72 | 6.28 | **3.7x** | 3.8x |
| `mfma` (O-proj MXFP4 GEMM) | 4.00 | 4.66 | 1.2x | 3.8x |
| `rmsnorm_router` | 1.52 | 1.48 | 1.0x | 1.6x |
| `topk` | 2.00 | 2.20 | 1.1x | 1.6x |
| Phase 7 total | 9.66 | 27.52 | 2.8x | 7.3x |

The two synchronization buckets -- together 60% of the regression -- reproduce
within a few percent of the full-model measurement.

### Where it ended up

Same build, same 400 layers, all five columns agreeing on both output hashes
(`breakdown_all.sh`). `fix` is per-AID release replicas plus per-AID level-1
arrival lines; `tree` adds the AID-aware inter-layer rendezvous:

| bucket | NPS1 | NPS1 tree | NPS2 base | NPS2 fix | NPS2 tree | tree vs NPS1 |
| --- | --- | --- | --- | --- | --- | --- |
| `slicewait` | 0.40 | 0.40 | 4.08 | 0.40 | 0.40 | +0.00 |
| `mfma` | 3.96 | 3.96 | 5.00 | 3.96 | 3.96 | +0.00 |
| `bar` | 1.80 | 1.80 | 6.28 | 2.44 | **2.32** | **+0.52** |
| `rmsnorm_router` | 1.52 | 1.52 | 1.48 | 1.76 | 1.56 | +0.04 |
| `topk` | 2.00 | 2.12 | 1.92 | 2.16 | 2.20 | +0.20 |
| **total** | **9.68** | 9.92 | **21.88** | 10.80 | **10.40** | **+0.72** |

NPS2 goes from **2.26x NPS1 to 1.07x**: 11.48 of the 12.20 us partitioning
penalty recovered, 94%, with 0.72 us left and 0.52 of that in `bar`. `slicewait`
and `mfma` are at exact parity, so the entire remaining gap is synchronization.

The NPS1-tree column is the control, and it is the reason the tree is a
partitioning fix rather than a barrier improvement: with one coherence domain
there is no boundary to keep traffic off, so the extra level is pure overhead and
costs 0.24 us. `bar` itself is unchanged at 1.80 either way.

### The cost is queueing, not per-crossing latency

`--tiles=N` varies how many workgroups poll the eight release flags. Every block
runs 4 waves and every wave polls two flags. All NPS2, same binary:

| tiles/XCD | blocks | polling waves | `slicewait` p50 | us per wave |
| --- | --- | --- | --- | --- |
| 23 (model default) | 184 | 736 | 7.76 | 0.0105 |
| 12 | 96 | 384 | 4.36 | 0.0114 |
| 4 | 32 | 128 | 1.48 | 0.0116 |
| 1 | 8 | 32 | 0.64 | 0.0200 |

Per-wave cost is flat within ~10% across a 23x range of poller count, and at 32
pollers NPS2 (0.64 us) has nearly closed on NPS1 (0.36 us). So the 27x is
queueing on eight cache lines that all land in one AID under NPS2 -- not a fixed
per-crossing latency.

### It is not inherited producer delay

The default is `--skew=0`: every producer publishes its slice and flag at the
same instant with no work in between. There is no attention merge to finish
late, yet the 27x still appears. That rules out the explanation that Phase 7 is
merely observing an upstream phase running late.

### Root cause and a measured fix

`ROOT-CAUSE.md` takes this the rest of the way. In one line: the driver maps
VRAM cacheable (`MTYPE_RW`) in NPS1 and non-coherent (`MTYPE_NC`) in SPX+NPS2,
because one compute partition spanning two memory partitions is local to
neither. A cacheable page lets a spinning wave's polls be answered by its own
L2; a non-coherent page forbids that, so all 184 polls per line are serviced at
the line's single home channel and queue there. Hence a cost very nearly linear
in waves-per-line, with production putting 184 on each of eight lines.

**The fix closes the regression rather than mitigating it.** Place one copy of
the flag array in each memory quadrant, have every producer publish into both,
and route each compute die to the copy homed in its own quadrant. Every reader
is then in the same consistency scope as the line it polls, which is what makes
a cacheable memory type safe -- and the cacheable memory type is what absorbs
the 184 readers. It applies to those two allocations alone; every other page in
the process stays non-coherent.

| SPX+NPS2, 23 tiles/XCD | wall us/layer | `slicewait` p50 |
| --- | --- | --- |
| plain `hipMalloc` flags, non-coherent | 30.56 | 14.48 |
| one copy per quadrant, still non-coherent | 15.68 | 6.68 |
| one copy per quadrant, cacheable | **3.83** | **0.80** |
| _SPX+NPS1 reference_ | _4.26_ | _1.04_ |

The middle row is the informative one: splitting alone buys ~2x by halving the
queue at each line, and the remaining ~4x comes from the caching, which only
becomes *legal* once the split guarantees co-location.

Confirmed in production code, not just in the reproducer. `drive_phase7` links
the real kernel, and the fix lands entirely in the harness with **no change to
the fleet headers**, because the kernel already takes the flag array as a
parameter and the producer store lives in the harness:

| bucket | NPS2 base | NPS2 + fix | NPS1 reference |
| --- | --- | --- | --- |
| `slicewait` | 14.16 | **0.36** | 0.36 |
| `bar` | 6.56 | 3.80 | 2.32 |
| `total` | 29.52 | **11.56** | 6.56 |

`slicewait` matches the NPS1 reference exactly. `bar` improved without being
touched -- its lines are still non-coherent, but removing ~1472 uncached polls
per layer freed the channels its own traffic shares. `mfma` went the other way
(3.40 to 4.08 against NPS1's 1.44), which reads as the bottleneck relocating to
the still-non-coherent weight buffer now that the flag wait no longer staggers
the waves; that is the largest remaining gap.

Replicating the flag array 16 times -- same total poll count, 16x fewer readers
per line, no driver knob at all -- takes the wait from 20.5 us to 2.7 us and
remains the fallback if per-allocation memory types are unavailable.

## Using this safely: the co-location rule

`aid_local_flag_mtype=2` hands a coherent memory type to any VRAM buffer that
asks for one, and there is exactly one rule that keeps that safe:

> **A buffer with a coherent MTYPE must only be read by the compute dies
> co-located with the memory quadrant it is homed in.**

Break it and the out-of-domain readers hang. They cache the line in their own
L2, the producer's store never reaches them as an invalidation probe -- DF
probes do not cross the quadrant boundary -- and they spin on a stale value
until their deadline. The `misrouted` cell in `ROOT-CAUSE.md` is that failure
reproduced deliberately.

The rule is narrower than it first sounds, in three ways that matter:

- **It binds only promoted buffers.** Everything else keeps MTYPE_NC, which
  resolves every access at the line's home, so a cross-quadrant read stays
  correct and is merely slow. Reading remote memory was never the problem;
  *caching* remote memory is.
- **It constrains readers, not writers.** A store from a die in the far
  quadrant still correctly invalidates the sharers that *are* co-located with
  the line, which is exactly what lets one producer publish into both replicas.
  `bench_hangdiag` measures this.
- **Neither half is sufficient alone.** The buffer must be AID-placed *and* its
  consumers routed to the replica in their own quadrant. `bench_aidsplit`
  measures all four combinations: placement without coherence is 7.60 us,
  coherence without correct routing hangs, and only both together reach
  0.85 us.

### How to avoid the mistake

- Request `AID_LOCAL | AID_SELECT | COHERENT` in a single `GEM_CREATE`, so a
  buffer can never be marked coherent without also being placed.
- Derive each consumer's pointer from its own XCD id. Handing one pointer to
  all eight dies is the entire failure mode.
- **Confine the coherent MTYPE to synchronization variables.** This is the load
  bearing one. A routing mistake on a flag hangs immediately and visibly; the
  same mistake on a data buffer returns a plausible stale number and corrupts
  results silently. Keeping the promotion inside the category where mistakes
  announce themselves is what makes it safe to ship.
- Do not reach for the global `mtype_local=2` instead. It buys the same latency
  by promoting *all* VRAM, including the activation buffers a megakernel
  rewrites every layer -- precisely the silent case above.
- Gate it: run with `aid_local_flag_mtype=0` and `=2` and compare logits
  bitwise. Bit-identical means no coherent buffer is being read out of domain.

The driver cannot check this for you. It sees where a buffer is placed but not
which dies will read it, so the routing half is a contract the calling code
has to keep. Two details of the current implementation are worth knowing:
the promotion branch tests only `COHERENT`, so a marked-but-unplaced buffer is
promoted anyway and then hangs; and `AID_LOCAL` stays set on a buffer even when
`amdgpu_gmc_aid_aperture()` returns NULL and it was placed unconstrained, so
the flag on its own is not evidence of placement.

## Geometry being modeled

Lifted from the generated `test.cu`, the fused layer instantiates as
`gang_full_layer_fused_kernel_mi300<1, 64, 2944, 2880, 64, 8, 4096, 512, 8, 4096, 512, 8, 128, 1, 16, 4096, 128, 4, 2944, 2944, 128, 64, true, 1>`,
which makes Phase 7
`<BATCH=1, OUTPUT_PER_WG=16, REDUCTION=4096, HIDDEN=2880, NUM_EXPERTS=128, K=4>`
with `n_wgs_per_xcd=23`, `output_stride=2944`, `router_tile_n=16`,
`total_oproj_tiles=184`, `total_topk_tiles=128`.

**The data on the critical path is tiny.** `attn_out` is `[1, 4096]` bf16 =
**8 KiB**, split into 8 slices of 512 bf16 (1 KiB each); XCD *x* produces slice
*x*. Wave *w* waits on XCDs 2*w* and 2*w*+1, so the four waves of every block
collectively read all eight slices -- every XCD reads the whole vector. 8 KiB is
a few nanoseconds of HBM bandwidth against a 23,000 ns wait, so there is no
bandwidth to recover and no benefit from spreading it over more stacks.

**The handshake deliberately bypasses cache on both sides.** The producer uses
`st_wt_u32` (`global_store_dword ... sc0 sc1`), a write-through store; the
consumer spins on `ld_sys_s32` (`global_load_dword ... sc0 sc1`), a system-scope
read. Every poll is a real memory round trip. The eight flags are strided 16
ints (64 B) apart, so all 448 bytes sit inside one 4 KiB page, with 736 waves
polling it. Their *addresses* turn out not to be the mechanism, though -- see
`ROOT-CAUSE.md`, which measures the stride directly and finds NPS1 flat across
it. What differs is the MTYPE the driver puts on the page.

## Reproducing

### 1. Node and container

Direct SSH to the node is gated by `pam_slurm_adopt`, so a Slurm allocation must
exist before any ssh or docker command works.

```bash
# from a Slurm login node
salloc --no-shell -w mi355x-thor-2 -t 04:00:00 --gres=gpu:8 \
       --reservation=<current>          # scontrol show res

ssh mi355x-thor-2
docker start fleet_v1
```

### 2. Apply the probe and build

The kernel already ships four timestamps under `MPK_OPROJ_INNER_TIMING`.
`slicewait-probe.patch` adds a fifth, `_op_t0b`, immediately after the
`buffer_inv` that follows the slice spin. That is what splits the old `mfma`
bucket into `slicewait` plus real compute -- without it the wait is hidden inside
the GEMM number and the regression looks like a 17.5x *compute* problem.

```bash
cd /root/schowdha/fleet-chiplet-megakernel
git apply /path/to/slicewait-probe.patch
cp /path/to/drive_phase7.cu .
bash /path/to/build.sh
```

`build.sh` extracts the compile flags from a previous model build log rather than
hardcoding them. This matters: the kernel's behavior depends on
`MPK_ATTN_SLICE_RELEASE`, `MPK_NARROW_OPROJ_HIER`, `MPK_OPROJ_TREE_BARRIER` and
`MPK_SYS_POLL_LOAD=2`, and a mismatch silently changes which code path runs.

### 3. Run both modes

```bash
export HIP_VISIBLE_DEVICES=6
bash run_both_modes.sh
```

Or by hand:

```bash
bash set_mode.sh NPS1                     # expect "SPX+NPS1 on 8/8" and "READY NPS1"
./drive_phase7 --layers=300 --tag=nps1 > bench_nps1.log

sleep 5                                   # let the GPU context release first
bash set_mode.sh NPS2                     # expect "SPX+NPS2 on 8/8" and "READY NPS2"
./drive_phase7 --layers=300 --tag=nps2 > bench_nps2.log

python3 compare.py bench_nps1.log bench_nps2.log
```

### 4. Validate the run before trusting it

A mode switch reloads the patched amdgpu driver. If the GPU has not settled,
`rmmod` fails, `set_mode.sh` prints `ABORT: amdgpu held`, and **the switch
silently leaves you in the previous mode** -- which yields a meaningless 1.0x
comparison. Check all three:

```
[nps2] blocks per XCD: 23 23 23 23 23 23 23 23        <- one block per XCD, none starved
[nps2] 184 tiles (23/XCD), ... 736 polling waves      <- geometry as intended
[OPROJ_INNER] slicewait=9.68 mfma=4.66 bar=6.28 ...   <- 8 samples per layer
```

and confirm the mode independently:

```bash
cat /sys/bus/pci/devices/0000:e5:00.0/current_memory_partition
```

### 5. The same-build NPS1 vs NPS2 breakdown

The table in `results/nps1-vs-nps2.md` comes from **one binary**, run in both
partition modes. That matters: the correctness changes shifted the NPS2 total by
about 0.5 us, which is the same order as the gap being argued about, so a
comparison assembled from two builds would not settle anything.

Both source changes are **already applied on this branch** -- the kernel header
carries the hier-split parameters and the two negative-control `#ifdef`s, and
`drive_phase7.cu` carries the weight staging. The `patch_*.py` scripts are the
record of how each was derived, with the reasoning in their docstrings; they
apply only to an unpatched tree and will refuse to run against this one, since
each checks its anchor matches exactly once.

So from a checkout of this branch, it is build and run:

```bash
cp phase7-bench/drive_phase7.cu "$HOME/fleet-chiplet-megakernel/"
cp phase7-bench/{build_gate.sh,rebuild_all.sh,gate_all.sh,buckets.sh} "$HOME/"
cp phase7-bench/{combine.sh,cmp_outputs.sh,killhang.sh,violate_colocation.sh} "$HOME/"
cp phase7-bench/{set_mode2.sh,run_root.py} "$HOME/"

bash rebuild_all.sh                 # the three binaries, from one source state
bash gate_all.sh                    # sensitivity, determinism, both controls, 3-way identity
bash violate_colocation.sh          # the rule failing on purpose; then killhang.sh

bash buckets.sh nps2_base --aid=0
bash buckets.sh nps2_fix  --aid=1 --coherent=1 --split=1
sudo -n python3 run_root.py set_mode2.sh NPS1     # ~2 min, reloads amdgpu
bash buckets.sh nps1      --aid=0
sudo -n python3 run_root.py set_mode2.sh NPS2     # restore
bash combine.sh                     # assemble the three runs into one table
bash cmp_outputs.sh                 # every probed buffer, not just the hashed two
```

`buckets.sh` prints the output hash alongside the medians on purpose. Two modes
agreeing on latency means nothing if they disagree on the answer, and one of
them being fast because it was wrong is exactly the failure to rule out.

## Knobs

| flag | default | isolates |
| --- | --- | --- |
| `--layers=N` | 200 | Sample count. 300 yields ~1600 usable samples after warmup. |
| `--tiles=N` | 23 | Workgroups per XCD, so 32*N polling waves. Sweep to separate queueing from per-access latency. |
| `--skew=N` | 0 | Per-XCD staggered producer release, in 10 ns ticks. Injects arrival skew for comparison against the zero-skew baseline. |
| `--delay=N` | 0 | Uniform producer delay in ticks. Models attention/merge compute ahead of the release. |
| `--aid=N` | 0 | Place the sync buffers with `alloc_in_aid` instead of `hipMalloc`. Required by every flag below; there is nothing to place into under one memory partition. |
| `--coherent=N` | 0 | Request a coherent MTYPE on the AID-local buffers. Sound only while every cached reader is co-located with its line's home AID. |
| `--split=N` | 1 | One release-flag replica per partition, each half of the XCDs polling its own. The fix; `--split=0` with `--aid=1` is the co-location violation, and it hangs. |
| `--lsplit=N` | 0 | Home the eight per-XCD level-1 arrival lines per AID. Partitioned, not replicated. Correct, and buys nothing -- see Known gaps. |
| `--hrdv=N` | 0 | Replace the flat inter-layer rendezvous with the AID-aware tree: per-XCD level 1, per-half level 2, then one store across the boundary instead of 184 remote atomics. |

Timestamps come from `s_memrealtime`, which ticks at 100 MHz, so one tick is
10 ns. That is the same conversion the in-kernel printf uses, so benchmark and
model numbers are directly comparable.

## Known gaps

- **`mfma` does not reproduce, and the reason is now specific.** The K-parallel
  arm reads its MFMA B operand from LDS at `oproj_lds_w_off()`, which the fused
  caller fills during Phase 6. This harness stages that tile itself, from the
  *harness* side, outside the kernel's `t0..t4` window -- so the bucket measures
  the LDS read, the FP8 quantize and the MFMA issue, and never the global weight
  fetch. Staging real weights instead of leaving the tile empty moved `mfma` by
  -0.12 us, which is how little it depends on the weights being present at all.
  The model's `mfma` going 1.56 -> 5.92 is a different quantity and is not
  addressed here.
- **The harness has to re-stage the weight tile every layer.** Hoisting the copy
  out of the layer loop (`patch_lds_once.py`, `-DMPK_DRV_STAGE_ONCE`) produces a
  different output hash on every run, so something in the kernel writes into the
  LDS region at or above `oproj_lds_w_off()` between layers. Consistent with the
  real Phase 6 issuing its DMA per layer; worth knowing before assuming that
  region is private.
- **The per-layer rendezvous is itself a cross-AID barrier**, so a little of the
  9.68 us is barrier-release skew rather than pure flag propagation. It is
  bounded by the `bar` number.
- **`MPK_OPROJ_INNER_TIMING` printfs once per XCD per layer**, which inflates
  wall-clock time roughly 100x. The printf lands after the last timestamp so it
  falls outside every measured span, and the bucket ratios reproduce the
  uninstrumented run -- but do not read `wall ... us per layer` as a real
  latency.
- **The residual `bar` gap is resolved, and it is not the barrier.** The
  +0.6-0.7 us that survived the release-flag fix was guessed in
  `results/nps1-vs-nps2.md` to be the level-2 global arrival counter. It is not.
  `-DMPK_OPROJ_BAR_TRACE` (`trace_bar.sh`) times five points inside the barrier
  and splits `bar` into `drain + l1 + wait + acq`: every step a block performs
  itself is at parity between the modes, and the whole +0.80 us sits in `wait`,
  which is not this block's latency but the last arriver's. Cross-XCD arrival
  spread grows from 0.22 us in NPS1 to 0.98 us in NPS2 -- +0.76 us against a
  +0.80 us `bar` delta -- and the releaser, which by construction sits on the
  XCD that finished last, stops rotating across all eight dies and pins to XCD
  6. Level 2 shows *no* AID asymmetry at all (0.520 us either half). So what is
  left is a load-imbalance problem upstream of the barrier, not a
  synchronization one; see `results/bar-attribution.md`.
- **Phase 7 inherits the skew rather than creating it.** `[BAR_OBS]` also carries
  `t0`, the absolute tick at Phase 7 entry, so the stagger at the door is
  separable from the stagger at the barrier. NPS1 enters Phase 7 with all eight
  dies within **0.09 us** and NPS2 with **0.82 us** already baked in, while
  Phase 7's own contribution is 0.08 us in NPS2 against 0.11 us in NPS1 -- it adds
  *less*. Per-XCD `slicewait` and `mfma` are flat across all eight dies in both
  modes, so Phase 7's work is balanced to the tick and the barrier is only the
  place where an imbalance created before it ran becomes visible.
- **The upstream culprit is the inter-layer rendezvous, and the fix is the same
  tree.** The flat rendezvous had all 184 blocks increment *both* replicas and
  poll their own: 368 atomics on two lines per layer, 92 of them read-modify-write
  across the partition boundary. Replicating the counters fixed the polling and
  left the arrivals, so the two replicas crossed their thresholds at different
  times and released the two halves staggered. `--hrdv=1` makes it hierarchical
  and AID-aware -- per-XCD level 1, per-half level 2, then a two-slot handshake
  where each half signals the other with one write-through store into a slot homed
  with its reader and polls its own slot locally. Critical-path boundary crossings
  drop from 184 remote atomics to **one store**. Entry stagger 0.82 -> 0.58 us,
  arrival spread 0.90 -> 0.67, last-arriver `bar` 2.76 -> 2.56, and the releaser
  stops pinning to XCD 6. `bar` p50 moves only 0.08 us, because recovered skew is
  shared across eight dies and only the last arriver is on the critical path.
  Every line is indexed by half as well as by placement, which is load-bearing:
  it keeps the topology a genuine 4/4 split when both region pointers alias one
  buffer, as they do in NPS1. `gate_hrdv.sh` checks all four configurations agree.
- **Homing the level-1 arrival lines per AID is correct and buys nothing.**
  `--lsplit=1` partitions the eight per-XCD arrival counters
  (`hier_local_lo`/`hier_local_hi`); unlike the release flags this needs no
  replication, because line `x` is only ever incremented by XCD x's own 23
  workers, who are co-located with each other by construction. It removes the
  full +0.12 us cross-AID penalty on XCDs 4-7 and `bar` does not move, because
  the saving goes straight into `wait`. Only one of the 184 level-1 atomics is
  ever on the critical path -- the last one on the last XCD -- so speeding up
  the other 183 is free and worthless. `gate_lsplit.sh` confirms the hashes
  still agree, so the split is sound; it is just not load-bearing.
- **Flag placement is resolved.** `bench_flagplace.hip` makes the stride a
  runtime knob: NPS1 is flat across it (2.08 us with all eight flags in one
  cache line vs 1.48 us over eight), so NPS1's advantage never came from
  spreading them over stacks, and neither does AID-locality -- at low fan-out
  NPS2 is the faster mode. The cause is the page MTYPE, MTYPE_RW in NPS1 vs
  MTYPE_NC in SPX+NPS2, and it cannot be switched back because MTYPE_RW makes
  the cross-AID handshake hang. What *does* fix it is more lines rather than
  more distance: see `ROOT-CAUSE.md`.
- **Output is numerically incomplete at `--tiles < 23`**, since fewer weight
  groups are computed. Timing stays valid; the tensor values do not.
- **Correctness is now gated, and the gate has been shown to fail.** It did not
  used to be: every output was identically zero, so a barrier that released
  early read the wrong layer's data and still hashed the same. See
  `results/correctness-gate.md` for what was wrong and the two negative controls
  that now hold the gate honest.

## Files

| file | purpose |
| --- | --- |
| `drive_phase7.cu` | The benchmark. Calls the real Phase 7 kernel; stubs the attention producer. |
| `slicewait-probe.patch` | Adds the `_op_t0b` timestamp that separates the wait from the GEMM. |
| `build.sh` | Compiles the driver with flags extracted from a model build log. |
| `run_both_modes.sh` | NPS1 arm, mode switch with settle delay, NPS2 arm, comparison. |
| `compare.py` | Parses `[OPROJ_INNER]` lines and prints per-bucket medians and ratios. |
| `sweep_pollers.sh` | The `--tiles` sweep that shows the cost is linear in poller count. |
| `bench_phase7.hip` | Hand-written slicewait-only model, no megakernel dependency. Useful for testing the handshake in isolation. |
| `bench_flagplace.hip` | Flag-placement probe. Runtime flag stride, poller count, two-level gate and spin backoff; splits the wait into poll / visibility / producer skew / invalidate. |
| `ROOT-CAUSE.md` | Why NPS2 costs 27x: the page MTYPE plus waves-per-line, with the stride, poller and replication sweeps and a measured 7.5x fix. |
| `results/` | Captured output backing the tables above. |
| `patch_hier_split.py` | The fix: one release replica per memory partition, each half of the XCDs polling its own. |
| `patch_lds_stage.py` | Stands in for Phase 6's weight DMA, fixes the underflowed E8M0 scales, and varies the residual per column. Without it every output is zero. |
| `patch_lds_fast.py` | Makes the staging a `dwordx4` copy; through `unsigned char *` the compiler cannot assume alignment. |
| `patch_lds_once.py` | `-DMPK_DRV_STAGE_ONCE`. Kept because it *fails*: proof the kernel reuses the LDS weight region between layers. |
| `patch_poll_skip.py` | `-DMPK_OPROJ_SKIP_HIER_POLL`. Negative control for the hierarchical barrier. |
| `patch_slice_skip.py` | `-DMPK_OPROJ_SKIP_SLICE_POLL`. Negative control for the attention-slice barrier. |
| `build_gate.sh` | Builds with extra defines, deriving the flag list from `build_drive_aid.sh` rather than restating it. |
| `rebuild_all.sh` | The three gate binaries, from one source state. |
| `gate_all.sh` | The gate: layer sensitivity, determinism, both negative controls, and the three-way identity check. |
| `violate_colocation.sh` | The co-location rule failing on purpose. Wedges the GPU; pair with `killhang.sh`. |
| `buckets.sh` / `combine.sh` | One bucket breakdown per mode, then assembled into a single table. |
| `cmp_outputs.sh` | Diffs every probed buffer across modes, not just the two that go into the hash. |
| `cmp_once.sh` | Reproduces the per-layer-staging finding above. |
| `killhang.sh` | Tears down a wedged run. `pkill -f drive_phase7` does not work: the pattern matches its own ssh wrapper and kills the parent first. |
| `set_mode2.sh` / `run_root.py` | Partition switch asserting SPX explicitly, and the `sudo -n python3` launcher it needs. |
| `results/nps1-vs-nps2.md` | The same-build NPS1 / NPS2-base / NPS2-fixed breakdown. |
| `results/correctness-gate.md` | Why the first gate proved nothing, and the two controls that fixed it. |
| `trace_bar.sh` | `-DMPK_OPROJ_BAR_TRACE`: times five points inside the barrier and splits `bar` into `drain + l1 + wait + acq`, per XCD. Ticks are taken in registers and printed past `done:`, so the barrier is not perturbed by its own measurement. |
| `trace_both_modes.sh` | The same trace either side of a partition switch, with the NPS2 restore in an `EXIT` trap so a mid-run failure cannot leave the node in NPS1. |
| `gate_lsplit.sh` | Correctness gate for `--lsplit`. Derives the reference from the first configuration rather than hardcoding a hash, because the recorded pair is specific to 400 layers -- the harness residual is a function of `layer & 15`. |
| `gate_hrdv.sh` | Correctness gate for `--hrdv`. Four configurations including the tree with one coherence domain, which is both the shape-versus-placement control and the check that the aliased path is still a 4/4 barrier. |
| `skew_hrdv.sh` | Entry stagger with the tree rendezvous on and off. `bar` alone cannot attribute the change; the entry columns can, because they are read before the barrier has done anything. |
| `breakdown_all.sh` | The five-column table above, one build, both partition modes, NPS2 restored in an `EXIT` trap. Includes the tree in NPS1, which is the control that separates the fan-in from the placement. |
| `results/bar-attribution.md` | Where the residual `bar` gap actually goes: arrival skew, not the barrier's memory protocol. Includes the level-1 split that works and buys nothing, the entry-versus-arrival split that exonerates Phase 7, and the tree rendezvous that recovers a third of the skew. |
