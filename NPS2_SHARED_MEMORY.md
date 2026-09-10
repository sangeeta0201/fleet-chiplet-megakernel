# Shared memory in SPX + NPS2

How to allocate a buffer that all eight XCDs read, on a part whose memory is
split into two coherence domains, and how to reproduce every number below.

## The three allocators

```c
static void *hipmalloc_shared(size_t n);  // spanning MTYPE_NC == plain hipMalloc
static void *hipmalloc_aid0(size_t n);    // MTYPE_RW pinned to AID0
static void *hipmalloc_aid1(size_t n);    // MTYPE_RW pinned to AID1
```

| | MTYPE | latency, all 8 XCDs | correct for |
|---|---|---|---|
| `hipmalloc_shared` | NC | 81.8 ns cached / 182-292 ns device-scope, +95..109 ns skew | any reader, **given a `buffer_inv sc1` acquire** |
| `hipmalloc_aid0/1` | RW | 97 ns flat, zero skew | readers on the home AID only |

`hipmalloc_shared` needs no ioctl: plain `hipMalloc` already returns spanning
VRAM, and under `aid_local_xcp_nc=Y` the driver demotes a spanning BO to
MTYPE_NC precisely because an XCP straddling two memory ranges is local to
neither. That demotion is the thing that makes it usable.

`hipmalloc_aid0/aid1` go through `alloc_in_aid`: one `GEM_CREATE` with
`AID_LOCAL` (plus `AID_SELECT` for the odd range). One ioctl, whole BO in one
aperture, large pages preserved, no per-access cost. No address swizzle and no
4 KiB page-by-page remap.

## Why the coherent allocation is the wrong tool

The obvious answer, allocate `GEM_CREATE_COHERENT` so hardware keeps the two
dies in sync, is what the harness shipped, and it is why Phase 7 failed its hash
in NPS2.

CC is not simply broken. With a reader that holds no prior copy of the line,
spanning CC read correctly across **~12 million checks** with zero stale reads,
over sharer counts up to 7,168, write-write false sharing of one line by two
dies, 1 GB of eviction churn between release and read, and host visibility after
`hipMemcpy`. The failure is narrower and worse: **a remote cached copy is never
invalidated.** `~/nps1/bits2.sh` pre-warms the reader's cache, has the writer
republish, then has the reader invalidate and re-read. Stale counts out of
16,000, reader doing `buffer_inv sc1` (the best column):

| store | CC same-AID | CC cross-AID | NC cross-AID |
|---|---|---|---|
| `sc0 sc1` write-through | 4075 | 11989 | **0** |
| plain (cached) | 15998 | 15996 | 16000 |
| plain + `buffer_wbl2 sc1` | **2138** | **9992** | **0** |
| `sc1` | 4484 | 11986 | **0** |
| `sc0` | 16000 | 15995 | 16000 |

No store x invalidate combination rescues CC: software cannot evict a coherent
line, because hardware treats the directory as authoritative, and the directory
does not invalidate the remote copies either. Cross-AID is ~4x worse than
same-AID, so whatever partial tracking exists does not reach across the
boundary. NC reaches zero with three of the five store modes, *provided* the
reader invalidates L2 and not just vL1. A bare `buffer_inv` fixes nothing
(16,000 stale); Phase 7 survives only because the layer-boundary acquire in
`gang_full_layer_fused_mi300.cuh:441` is `buffer_inv sc0 sc1`.

MTYPE_NC is **non-coherent, not non-cacheable**, a distinction the old comment
in `drive_phase7.cu` got backwards. `~/nps1/repro2.sh` times a plain load
against `sc0 sc1` on the same lines; cacheable memory makes the plain load much
cheaper. All four buffer types come out at ratio 0.44-0.45 (~684 vs ~1550
ticks), so every one of them is cached. NC works not by bypassing cache but by
being software-invalidatable.

## Why not just pin the shared buffers to one AID

Because a pinned line is MTYPE_RW, and RW is coherent only within its own AID.
The clean comparison holds mode and allocator fixed and varies only the MTYPE
(`load_perbo_gen.sh <xcp_nc> <flag_mtype>`; the knobs are **positional**):

| MTYPE for `out` | Phase 7 `--place=4`, 5 reps |
|---|---|
| NC | **5/5 green** |
| RW | 2/5 |
| CC | 0/5 |

The same RW allocation is green at 9.80 us in NPS1, where one partition makes RW
device-wide. So this is not a bad MTYPE choice, it is a scope mismatch. NC wins
by being effectively uncacheable for a reader that invalidates, so every load is
unconditionally fresh.

## Reproducing

Mode first. **Never** use `rocm-smi --setmemorypartition`: it restarts amdgpu,
the restart loads the stock DKMS module, and the box lands in DPX+NPS2 where
every later `--setcomputepartition SPX` is correctly refused.

```bash
salloc -w mi355x-thor-2 -t 04:00:00 --gres=gpu:8   # no --reservation; none active
bash ~/nps1/spx_nps2.sh                            # verifies 8/8, exits 1 otherwise
WANT=NPS1 bash ~/nps1/spx_nps2.sh                  # the NPS1 control
```

Assert the mode **on every run**. A reader-count sweep once ran in stock NPS1,
where `GEM_CREATE_COHERENT` is a no-op, and produced 3.4 M meaningless
"COHERENT" lines. `assert_mode` in `~/nps1/verify_land.sh` is the guard.

```bash
bash ~/nps1/verify_land.sh     # builds, then 3 hash-gated arms + the probe
bash ~/nps1/bar_gap.sh         # the bar-phase placement matrix
bash ~/nps1/breakdown.sh       # full [OPROJ_INNER] breakdown; WANT=NPS1 for the control
bash ~/nps1/bits2.sh           # the MTYPE x store x invalidate matrix above
bash ~/nps1/repro2.sh          # cacheability (plain vs sc0 sc1 ratio)
bash ~/nps1/repro.sh           # faithful-shaped 184-WG publish/read race
```

Gate every Phase 7 run on `rmsnorm_out=ff1bdbe7ad2c4ed3`. A fast wrong hash is
not a result.

### Build and measurement traps

- `build_drive_aid.sh` is the real build. `place_matrix.sh` uses a bare
  `hipcc -O3 -DMPK_AID_LOCAL` that omits every other `-DMPK_*`, and when it
  fails it only checks `-x ./drive_phase7`, so it silently tests the stale
  binary from the last real build. `verify_land.sh` deletes the binary, fails
  loudly, and checks it is newer than the source.
- `build_drive_aid.sh` ships `-DMPK_OPROJ_NO_WB`, which compiles out the
  write-side `buffer_wbl2 sc1`. That alone moved CC from 0/5 to 2/5.
- The metric is the `[OPROJ_INNER]` breakdown, **not** the `us per layer` line
  (that is ~906 us, the whole fused layer). There is no `median` line at all.
- Within the breakdown, `bar` is the load-bearing column: spread 0.04 us.
  `total` is dominated by `topk`, which swings 0.4 to 6.8 sample to sample.
- Headers do not trigger a rebuild: `rm -rf demo/gpt_oss/permanent_output_dir`
  after editing one.
- A `PERBO:` line in dmesg is **not** proof the patched module is live, since
  dmesg survives reloads. Check that `/sys/module/amdgpu/parameters/aid_local_*`
  exist; absent means stock.
- Partition state is persisted in hardware, so a fresh module load comes up in
  whatever was last baked in. `load_perbo.sh` leaves the box in NPS1 if that is
  where it was; `spx_nps2.sh` is what actually moves it.

## Performance: placement is worth 2.1x in NPS2, and the residual 1 us is not placement

Full `[OPROJ_INNER]` breakdown, same binary, 400 layers, 7 reps, mode asserted,
median of per-run medians:

| mode | sync | slicewait | mfma | **bar** | rmsnorm_router | topk | total |
|---|---|---|---|---|---|---|---|
| NPS1 | spanning | 0.40 | 3.96 | **2.00** | 1.52 | 1.96 | 9.84 |
| NPS1 | AID-local | 0.40 | 3.96 | **2.00** | 1.52 | 2.40 | 10.28 |
| NPS2 | spanning | 4.48 | 5.00 | **6.76** | 1.48 | 1.88 | 22.88 |
| NPS2 | AID-local | 0.40 | 3.96 | **2.64** | 1.64 | 2.24 | 10.84 |

Read the NPS2 rows first. With everything spanning, NPS2 is **22.88 us**, 2.3x
*slower* than NPS1, because spanning VRAM is MTYPE_NC and the sync state, the
weights and the slice handoff all pay for it (`bar` 6.76, `slicewait` 4.48, even
`mfma` 5.00). Putting the sync state in AID-local memory takes that to **10.84
us, a 2.1x speedup**, and restores `slicewait` and `mfma` to their NPS1 values
exactly. That is where the AID-local allocator earns its keep: not as a 10-20%
win over NPS1, but as the thing that makes NPS2 usable at all.

In NPS1, placement changes nothing (`bar` 2.00 either way). With one memory
partition, plain VRAM is already MTYPE_RW and coherent device-wide, so there is
nothing to place.

The residual against NPS1 is **1.00 us of total**, of which +0.64 is `bar`,
+0.12 is `rmsnorm_router`, and the rest is `topk` noise. `mfma` is identical to
the hundredth. That residual is **not** recoverable by placement, measured:

| `--split=1 --lsplit=1` arm | bar | spread |
|---|---|---|
| `counters` SPANNING (the control) | 2.64 | 2.60-2.68 |
| `counters` in AID0 (`--catomic=1`) | 2.64 | 2.60-2.68 |
| `counters` in AID1 (`--catomic=3`) | 2.64 | 2.60-2.68 |

Zero difference, and `counters` was the only buffer the barrier still touches
that was left spanning. Everything else (`flags`, the rendezvous counter, the
hierarchical release replicas, the level-1 arrival lines) is already AID-local
at 97 ns flat. This is the control the earlier reading lacked: comparing
`--catomic=1` against `--catomic=3` is near-versus-far with both arms placed, so
it could never show whether placing `counters` at all helps.

So the last 0.64 us is the intrinsic cost of a device-scope release/acquire
ordering across two coherency domains, plus NC-forced refetch of `out` after the
layer-boundary invalidate. Neither is addressable by moving pages.

## Phase 7 has no *data* placement headroom either

Placing the data buffers is a separate question from placing the sync state, and
there the answer is that there is nothing to gain. Placement works: every buffer
`--dsplit` places goes from a ~97 ns crossing penalty to **97 ns flat on all 8
XCDs with zero skew** (from 182 local / 279 remote), and `--dsplit=3` stays 5/5
green. But the total does not move; it gets slightly worse (10.96 vs 10.80,
`bar` 2.84 vs 2.64). Two independently measured reasons:

1. **The read data is already L2-resident.** `rocprofv3 --pmc TCC_HIT TCC_MISS
   TCC_EA0_RDREQ TCC_EA0_RDREQ_DRAM` over the 8-layer dispatch: **97.32% L2 hit
   rate** (9,709,576 hits / 267,895 misses of 9,977,471 requests) and only
   **8.55 MiB total reaches HBM** (70,019 DRAM fills x 128 B), about 1.07
   MiB/layer against a 6.1 MiB weight. The weight is fetched once cold and
   served from L2 thereafter, so placement only controls where the missing 2.7%
   lands. `mfma` is pinned at 3.960 in 16/16 runs across every placement arm.
2. **There is no inter-die traffic to remove.** MultEvent GPU Data Fabric CAKE
   counters, `CAKE0 total bytes moved flit`: idle **0.00** GB/s, Phase 7
   **0.04** GB/s, a 4,821 GB/s streaming hammer **37.71** GB/s. The positive
   control is what makes Phase 7's ~0 trustworthy: the derived GB/s rows do
   work, Phase 7 simply is not on the fabric.

Two measurement notes. rocprofv3 **cannot** measure inter-die traffic here:
every GMI counter in `--list-avail` is 32B-qualified and Phase 7 issues only
64/128 B requests (`TCC_EA0_RDREQ_32B` = 0), so they read 0 for reasons
unrelated to placement. And the CSV kernel name
`drive_phase7(void*, void*, ...)` contains commas, so index counters from the
END (`$(NF-3)` = name, `$(NF-2)` = value).

This agrees with Mosaic rather than contradicting it. Their Fig 9b puts
die-aware placement at **-1.6% average / -13% worst case** in the `<16 MB`
operand-footprint bin, which is where Phase 7's 6.1 MB O-proj sits. Their 10-19%
wins are all `>128 MB` memory-bound shapes, and for a decode megakernel
specifically it is the KV cache: **28% throughput swing** die-local vs
die-remote, +5.0% against a cache-aware baseline, concentrated at GQA group size
1 with long context.

## Do not re-run these

- Any CC store/invalidate protocol variant. The o-proj publish/consume path was
  audited end to end, every site exercised, hash-gated, 5 reps, mode asserted:
  as-shipped 0/5, plus write-side writeback 2/5, plus `buffer_inv sc1` acquire
  2/5, plus cached publish 1/5, plus device-scope `sc0 sc1` norm reads 2/5,
  NC 5/5. With `sc0 sc1` on both the store and the load the access pattern is
  instruction-for-instruction what NC does, yet NC is 5/5 and CC is 2/5, so the
  bug is not in how the kernel touches `out`.
- Placing `counters` to chase the `bar` gap. Measured at 0.00 us above, against
  the proper spanning control.
- `--catomic=1 --bsplit=2`, which now **hangs**. That is confirmation the MTYPE
  is really RW: an RW line is not coherent across the AID boundary, so the poll
  never sees the other die's write-through store.
- `TCC_EA0_RDREQ_GMI_32B` as evidence of anything. It reads 0 because there are
  no 32 B requests at all.
- An eviction-pressure test using `hipMalloc`'d memory as the churn. NC consumes
  no directory entries; a real CC capacity test must churn *CC* memory.

## Open

- `--place=1` (logits through the GEM path) hangs at `layers=1`
  deterministically, in every configuration including CC, with a
  `PERMISSION_FAULTS: 0x3` retry page fault from TCP. Not the `nt` loads (those
  target `g_base_pf`/`w_base_pf`), and not a size overrun (a 2 MiB BO does not
  fix it). Pre-existing and unrelated to coherence; moot now that shared data
  uses `hipMalloc`.
- Whether the 0.64 us `bar` residual can be attacked at all. Every placement
  lever is exhausted, so any remaining win has to come from changing the
  barrier's algorithm, not its memory.

