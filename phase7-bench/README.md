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
ints (64 B) apart, so all 512 bytes sit inside a single 4 KiB page -- one page,
one AID under NPS2, with 736 waves polling it.

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

## Knobs

| flag | default | isolates |
| --- | --- | --- |
| `--layers=N` | 200 | Sample count. 300 yields ~1600 usable samples after warmup. |
| `--tiles=N` | 23 | Workgroups per XCD, so 32*N polling waves. Sweep to separate queueing from per-access latency. |
| `--skew=N` | 0 | Per-XCD staggered producer release, in 10 ns ticks. Injects arrival skew for comparison against the zero-skew baseline. |
| `--delay=N` | 0 | Uniform producer delay in ticks. Models attention/merge compute ahead of the release. |

Timestamps come from `s_memrealtime`, which ticks at 100 MHz, so one tick is
10 ns. That is the same conversion the in-kernel printf uses, so benchmark and
model numbers are directly comparable.

## Known gaps

- **`mfma` does not reproduce** (1.2x vs 3.8x). The harness has no Phase 6
  weight-DMA overlap and no competing QKV or MoE traffic, and its 6.11 MiB of
  uniform weights behave differently from the real MXFP4 tensors. The
  synchronization buckets reproduce; the compute term still needs the full
  workload to attribute.
- **The per-layer rendezvous is itself a cross-AID barrier**, so a little of the
  9.68 us is barrier-release skew rather than pure flag propagation. It is
  bounded by the `bar` number.
- **`MPK_OPROJ_INNER_TIMING` printfs once per XCD per layer**, which inflates
  wall-clock time roughly 100x. The printf lands after the last timestamp so it
  falls outside every measured span, and the bucket ratios reproduce the
  uninstrumented run -- but do not read `wall ... us per layer` as a real
  latency.
- **Where the flag lines physically land is inferred, not measured.** The eight
  flags span 512 B, so under NPS2 they cannot straddle a 4 KiB page and must
  share one AID. The NPS1 side is *not* verified: whether the finer interleave
  scatters those eight lines across stacks depends on the interleave
  granularity, which we have not confirmed. Note the queueing evidence -- cost
  linear in poller count -- stands regardless of that detail.
- **Output is numerically incomplete at `--tiles < 23`**, since fewer weight
  groups are computed. Timing stays valid; the tensor values do not.

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
| `results/` | Captured output backing the tables above. |
