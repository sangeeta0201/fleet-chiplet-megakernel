# Phase 7 on SPX+NPS2: results and how to reproduce them

Target: close the gap between Phase 7's O-projection in SPX+NPS2 and the SPX+NPS1
baseline. Every run is gated on `rmsnorm_out=ff1bdbe7ad2c4ed3`; a fast run with a
different hash is not a result.

## Numbers

| configuration | median total (?s) |
| --- | --- |
| SPX+NPS1 baseline | 9.76 / 9.80 / 9.88 |
| SPX+NPS2 `--aid=1` (best) | 10.28 - 10.52 |
| NPS2 `--dsplit=1` (data placed) | 11.04 vs 10.48 control |

`mfma` is pinned at 3.960 ?s in 16 of 16 runs across every placement arm, so the
compute half is insensitive to placement and the whole difference lives in the
memory and rendezvous halves.

### Counters (8-layer dispatch, rocprofv3)

| metric | value |
| --- | --- |
| L2 hit rate | 97.32% |
| L2 hits / misses | 9,709,576 / 267,895 of 9,977,471 |
| HBM traffic | 8.55 MiB total (70,019 DRAM fills ? 128 B) |
| per layer | ~1.07 MiB against a 6.1 MB weight |

The O-proj weight is fetched once cold and served from L2 on every later layer.
`--tiles` is capped at `TILES_PER_XCD` = 23, so the working set cannot grow past
6.1 MB (800 KB per XCD) and this is structural, not a tuning artifact.

### Inter-die fabric (MultEvent GPUDF CAKE)

| window | `CAKE0 total bytes moved flit` |
| --- | --- |
| idle | 0.00 GB/s |
| Phase 7 | 0.04 GB/s |
| streaming copy at 4,821 GB/s HBM | 37.71 GB/s |

Phase 7 is ~1000? off the fabric. See `aid-local-hbm/INTERCONNECT.md` for the
method ? rocprofv3 cannot measure this, because every GMI counter is
32B-qualified and Phase 7 issues only 64/128 B requests.

### Placement probe (per-XCD load latency)

| buffer | before per-BO MTYPE fix | after |
| --- | --- | --- |
| weight(AID0) | 182 ns local / 279 ns remote (+97.3 skew) | 97 ns flat, ?0.3 skew |
| out | +108.6 ns skew | unplaced |
| counters | +96.9 ns skew | unplaced |
| logits | +96.6 ns skew | unplaced |

Placement works: every buffer `--dsplit` actually places goes to 97 ns flat on
all 8 XCDs, a 1.87? latency improvement, because RW makes the line L2-cacheable
on both sides. The total does not move, because the data those buffers hold is
already L2-resident.

### Device-scope atomics are homing-insensitive

| arm | median (?s) |
| --- | --- |
| `--catomic=0` (NC baseline) | 10.40 |
| `--catomic=1` (AID0 RW, near) | 10.38 |
| `--catomic=2` (AID0 CC) | 10.36 |
| `--catomic=3` (AID1 RW, far) | 10.44 |

Near vs far is 0.06 ?s apart against 0.3?0.4 ?s per-arm spread. `sc1` atomics
serialize at the device coherency point regardless of where the line is homed, so
placing `counters` cannot help.

`--catomic=1 --bsplit=2` **hangs** under the per-BO driver, where it ran under the
old one. That is positive confirmation the MTYPE really is RW: an RW line is not
coherent across the AID boundary, so the poll never observes the other die's
write-through store. Do not re-run that combination expecting a number.

## Reproducing

### 1. Allocation

```bash
salloc -w mi355x-thor-2 -t 04:00:00 --gres=gpu:8 --reservation=<current>
```

Direct SSH to thor-2 is gated by `pam_slurm_adopt`, so the allocation must exist
first. Run `salloc` from the login node, not thor-2. Reservation names are dated;
check `scontrol show res` if it is rejected.

### 2. Driver

The per-BO MTYPE patch is `aid-local-hbm/0003-amdgpu-per-bo-flag-mtype.patch`.

```bash
~/nps1/load_perbo.sh     # loads the patched amdgpu
~/nps1/set_spx.sh        # it comes up DPX; Phase 7 needs SPX
```

Verify before measuring:
- `PERBO: AID_LOCAL BO is local -> mtype_local=1` in dmesg (enum 1 = MTYPE_RW)
- `FLAGMTYPE knob=2 -> mtype=2` (CC) for the coherent sync buffer
- plain spanning VRAM still NC, zero VM faults
- 8/8 SPX/NPS2

Keep `aid_local_xcp_nc=Y`. With `N` you lose half the dispatch ? only XCDs 0?3
run and it faults at 0x1000000000, which looks like a speedup and is not one.

A node reboot wipes both the patched driver and the partition mode. Rebuild
(~15 min, gcc-13) and re-apply SPX+NPS2 before measuring.

### 3. Build and run

```bash
cd ~/fleet-chiplet-megakernel
./build_drive_aid.sh
# --aid=0 is the SLOW arm (NC sync line, ~22.8 us). It is the control.
./drive_phase7 --layers=400 --tiles=23 --aid=0 --split=0 --lsplit=0 --hrdv=0 --tag=flat

# --aid=1 is the ~10.3 us arm: sync buffer AID_LOCAL + EXT_COHERENT -> CC.
./drive_phase7 --layers=400 --tiles=23 --aid=1 --coherent=1 --tag=best
```

**`--aid` defaults to 0**, and `--aid=0` is the 22.8 us NC-sync-line case, not
the ~10.3 us number in the table above. `--coherent=1` on its own does nothing
(22.84); it only helps once `--aid=1` has made the buffer AID_LOCAL. See
`REPRODUCE-NPS2-ARMS.md` for the full sweep.

Flags that matter:

| flag | effect |
| --- | --- |
| `--aid=1` | replicate the sync buffer per AID (AID_LOCAL + EXT_COHERENT ? CC) |
| `--split` / `--lsplit` / `--hrdv` | hierarchical rendezvous: split, level-1 local, hierarchical dispatch |
| `--dsplit` | partition read-only data (weight, residual, attn_out) across AIDs |
| `--catomic=0..3` | home the atomic counters (0 NC, 1 AID0 RW, 2 AID0 CC, 3 AID1 RW) |
| `--bsplit=2` | deliberately wrong-AID homing, used as a control |

Headers do not trigger a rebuild ? `rm -rf demo/gpt_oss/permanent_output_dir`
after any header change.

### 4. Counters

```bash
./l2_vs_hbm.sh          # TCC_HIT, TCC_MISS, TCC_EA0_RDREQ, TCC_EA0_RDREQ_DRAM
./sum_pmc2.sh           # parse the CSVs
```

The kernel name `drive_phase7(void*, void*, ...)` contains commas, so index
counter columns from the **end** of the CSV row (`$(NF-3)` = name, `$(NF-2)` =
value), not from fixed column numbers.

### 5. Inter-die traffic

```bash
~/aid-local-hbm/tools/interconnect/scripts/build_gpudf_ini.sh
~/aid-local-hbm/tools/interconnect/scripts/ucake.sh 10 phase7 -- \
    ./drive_phase7 --layers=12000 --tiles=23 --tag=ucake
```

12,000 layers ? 10.3 s, which clears MultEvent's ?5 s event-group cycling
requirement.

## Where the remaining gap is, and is not

Ruled out by measurement:

- **Inter-die fabric traffic** ? 0.04 GB/s against a 37.71 GB/s positive control.
- **Placement of the read data** ? the weight is 97.32% L2-resident; placement
  delivers flat 97 ns but no end-to-end change.
- **Homing of the atomic counters** ? near vs far differ by 0.06 ?s, inside noise.

Remaining suspect: the `split` / `lsplit` / `hrdv` rendezvous machinery that NPS2
needs for cross-AID coherence, which NPS1 does not pay for.

Mosaic's own Figure 9(b) measures die-aware placement at ?1.6% average and ?13%
worst case in the `<16 MB` operand-footprint bin, which is exactly where Phase 7's
6.1 MB weight sits. The `--dsplit` regression reproduces their result rather than
contradicting it. Their 10?19% wins are all `>128 MB` memory-bound shapes, and for
a decode megakernel the win is in the KV cache ? 28% throughput swing die-local vs
die-remote, concentrated at GQA group size 1 with long context.

