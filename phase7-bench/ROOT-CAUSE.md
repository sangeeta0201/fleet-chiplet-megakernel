# Why Phase 7's slice wait costs 27x in NPS2

Two things compose to produce the regression, and they answer different
questions:

- **Why sharing a flag line costs anything at all**: the MTYPE the driver puts
  on every VRAM page. NPS1 gets MTYPE_RW, where hardware keeps the line coherent
  and concurrent system-scope readers are absorbed. SPX+NPS2 gets MTYPE_NC,
  where there is no hardware coherence, so every poll is serviced at the line's
  home and requests to the same line queue.
- **How much it costs**: how many waves share one cache line. In NPS2 the wait
  is very nearly linear in waves-per-line, and the production layout puts 184
  waves on each of 8 lines.

Neither term is about which AID the flags live in. AID placement changes
distance, and distance is not the bottleneck -- at low fan-out NPS2 is the
*faster* mode.

Measured with `bench_flagplace.hip`, which isolates the flag handshake from the
GEMM and splits the wait into terms the in-kernel `slicewait` bucket lumps
together.

## What the flags are

`attn_release` is eight `int32`s, one per XCD, inside `oproj_topk_counters` -- a
plain `torch.int32` device tensor built by `make_tensor` in
`demo/gpt_oss/demo.py` and handed to the fused kernel as `input_ptrs[16]`:

```
int *oproj_counters_base = static_cast<int *>(input_ptrs[16]);
int *attn_release = oproj_counters_base + FULL_LAYER_ATTN_XCD_RELEASE_SLOT;
```

with `FULL_LAYER_ATTN_XCD_RELEASE_SLOT = 36 * 16`. So flag *x* sits at byte
offset `2304 + 64x`, and all eight span 448 B inside the first 4 KiB page.

- **Producer.** XCD *x*'s Phase 6 merge publishes its own slice with
  `st_wt_u32(&attn_release[x * 16], epoch)` -- a write-through `sc0 sc1` store.
- **Consumer.** In Phase 7, wave *w* of *every* O-proj workgroup spins on
  `attn_release[2w * 16]` and `attn_release[(2w+1) * 16]` with `ld_sys_s32`, a
  system-scope `sc0 sc1` load. At 23 workgroups per XCD and 4 waves each that is
  736 waves issuing 1472 polls per layer against 8 cache lines -- 184 waves on
  every line.

Nothing about that allocation differs between NPS1 and NPS2.

## Moving the eight flags apart is not the fix

`--flagstride` makes the 64 B stride a runtime knob. Sweeping it at production
scale (736 polling waves), `poll` p50 in microseconds:

| flag stride | NPS1 | NPS2 |
| --- | --- | --- |
| 4 B (all eight in one line) | 2.08 | 68.16 |
| 32 B | 1.80 | 39.64 |
| 64 B (production) | 1.48 | 19.80 |
| 256 B | 1.56 | 13.00 |
| 1 KiB | 1.52 | 15.20 |
| 4 KiB | 1.60 | 12.64 |
| 64 KiB | 1.56 | 13.40 |
| 16 MiB | -- | 14.04 |
| 256 MiB | -- | 14.60 |

**NPS1 is flat.** Collapsing all eight flags onto a single cache line costs
2.08 us against 1.48 us for eight lines -- 1.4x, not the 8x you would see if
NPS1's advantage came from spreading them over eight stacks. An earlier
write-up of this work claimed the flags "hash across all eight stacks" in NPS1
and that this was why NPS1 was fast. That was wrong in premise, and the
interleave granularity turns out not to matter: NPS1 would still be fast with
every flag in one line.

**NPS2 responds and then saturates**, 68 us down to about 12.5 us and flat from
256 B out to 256 MiB. Note what this sweep does and does not vary: it always has
exactly eight lines and only moves them further apart. Once they are on separate
lines, distance buys nothing. Increasing the *number* of lines is a different
axis, and that one does work -- see below.

## The mechanism is fan-out per line

Sweeping poller count at the production 64 B stride, `poll` p50 in microseconds:

| polling waves | NPS1 | NPS2 |
| --- | --- | --- |
| 32 | 1.12 | 0.96 |
| 128 | 1.36 | 3.12 |
| 384 | 1.44 | 9.04 |
| 736 (production) | 1.44 | 19.96 |

Two things fall out:

1. **At low fan-out NPS2 is the faster mode** (0.96 vs 1.12 us). Per-access
   latency is not the problem -- an AID-local access has a shorter path than the
   NPS1 average. This is the single most useful fact for ruling out placement
   fixes.
2. **NPS1 is flat and NPS2 is linear.** NPS1 absorbs a 23x increase in readers
   for 30% more time. NPS2 pays about 27 ns per additional polling wave.

The harness also confirms the wait is genuinely flag visibility. Producers
publish a release timestamp next to the flag, so `vis` measures store to
observation directly:

| term | NPS2, production |
| --- | --- |
| `poll` (spin loop) | 20.24 us |
| `vis` (producer store -> observed) | 21.25 us |
| `skew` (producer late vs poll start) | 0.03 us |
| `inv` (the `buffer_inv` after the spin) | 0.00 us |

So it is not inherited producer delay and it is not the cache invalidate.

## What actually differs: the page MTYPE

`gmc_v9_0_get_vm_pte_flags` picks the MTYPE for every page. Stock logic marks
local VRAM `is_local` and gives it `mtype_local`, which is **MTYPE_RW** at the
default `amdgpu.mtype_local=0`. The patched driver this box runs adds:

```c
/*
 * An XCP spanning several memory ranges (SPX + NPS2) is local
 * to none of them. Calling it local maps every page MTYPE_RW,
 * which is only coherent inside a single memory partition.
 */
if (amdgpu_aid_local_xcp_nc && adev->xcp_mgr &&
    !adev->xcp_mgr->num_xcp_per_mem_partition &&
    adev->gmc.num_mem_partitions > 1)
        is_local = false;
```

The condition is exactly "one XCP spanning more than one memory partition",
i.e. SPX+NPS2, and it demotes every VRAM page to **MTYPE_NC**. In NPS1
`num_mem_partitions == 1`, so it cannot fire and pages stay MTYPE_RW.

That is what makes line sharing expensive. Under MTYPE_RW the hardware keeps the
line coherent, so 184 waves polling one line are served out of the coherent
hierarchy and absorbed. Under MTYPE_NC there is no hardware coherence, so a
correct `sc0 sc1` poll must be serviced at the line's home every time and those
requests serialize there.

Note the MTYPE is a property of the page mapping and applies to **all** VRAM in
SPX+NPS2, whichever AID the page sits in. An AID-local flag page is still
MTYPE_NC, still has no hardware coherence, and still serializes its readers.

### It cannot simply be switched back

Reloading with `aid_local_xcp_nc=0` in SPX+NPS2, so pages get MTYPE_RW:

```
$ timeout 25 ./bench_flagplace --layers=1 --tiles=1 --flagstride=64
   (times out)
```

The handshake **hangs**, at production scale and equally at 32 polling waves
with a single layer. MTYPE_RW is only coherent inside one memory partition, so a
flag written by a producer on one AID never becomes visible to a poller on the
other and the spin never retires. This is precisely the correctness bug the
patch exists to prevent. MTYPE_NC is load-bearing, and the cost of sharing under
it is the price of correctness -- not a misconfiguration to undo.

## Would one flag page per AID fix it?

Partly, and the measurement says how much. `--copies=K` replicates the
eight-flag array K times; producers write every copy, so the information is
identical and the **total** poll count is unchanged at 1472. Only the sharing of
each line changes. K=2 with one copy per AID is the natural "let each AID poll
its own flags" scheme.

NPS2, 736 polling waves, 64 B stride:

| copies | waves per line | `poll` p50 | `vis` p50 |
| --- | --- | --- | --- |
| 1 (production) | 184 | 20.48 | 21.65 |
| 2 (one per AID) | 96 | 9.52 | 9.04 |
| 4 | 48 | 5.32 | 4.10 |
| 8 | 24 | 3.76 | 2.16 |
| 16 | 16 | **2.84** | **1.48** |
| 23 (one per workgroup rank) | 8 | 3.16 | 1.09 |

So yes -- with K=2 you would still have most of the problem. Halving the readers
per line halves the cost, from 20.5 us to 9.5 us, and 9.5 us is still 6x NPS1.
The win has nothing to do with the two copies being in different AIDs; it is
just two lines instead of one. Distance was never the bottleneck.

The useful part is that the same axis, pushed further, does close the gap.
Because total poll count is held constant across this sweep, it also settles the
earlier ambiguity: the serialization is **per line**, not a global request-rate
limit. At K=23 the visibility term is 1.09 us against NPS1's 1.21 us -- fully
recovered.

## The fix: replicate the flag array

Replication is the cheapest thing that works and it is nearly free in layout
terms: 16 copies x 8 flags x 64 B is 8 KiB. The 256 B stride is marginally
better at low K and converges by K=16:

| variant (NPS2, 736 waves) | `poll` p50 | vs production |
| --- | --- | --- |
| production (1 copy, 64 B) | 20.48 | 1.0x |
| 16 copies, 64 B | 2.84 | 7.2x |
| 16 copies, 256 B | **2.72** | **7.5x** |
| 23 copies, 256 B | 3.04 | 6.7x |

Cost in NPS1, where sharing was already free, is the producer's extra
write-through stores:

| copies (NPS1, 256 B) | `poll` p50 |
| --- | --- |
| 1 | 1.48 |
| 2 | 1.52 |
| 16 | 1.84 |
| 23 | 1.96 |

So K=16 is 7.5x in NPS2 for 24% in NPS1. That is a small enough NPS1 cost that
gating is optional, and unlike the two-level gate below it does not change the
shape of the wait at all -- each wave still observes its own two slices
independently, so the per-slice overlap `MPK_OPROJ_SPLIT_SLICE_WAIT` exploits
survives.

### Alternative: a two-level gate

Fewer polls rather than more lines. Per XCD, the rank-0 block's wave 0 polls all
eight remote flags and republishes one XCD-local flag; every other block's
wave 0 polls only that; waves 1-3 pick it up through `__syncthreads()`.
Flag-polls per layer drop from 1472 to 240.

| variant (NPS2, 736 waves) | `poll` p50 |
| --- | --- |
| two-level (`--hier=2`) | 6.20 |
| two-level, replicated line (`--hier=3`) | 5.80 |
| `--hier=3 --flagstride=256 --sleep=32` | 5.08 |

It also repairs the scaling, 2.80 us at 32 waves to 5.44 at 736 -- 1.9x instead
of 21x. But it is strictly worse than replication (5.08 vs 2.72), costs 2x in
NPS1 (1.48 -> 3.16) because the second hop buys nothing where readers are
already absorbed, and it collapses the eight per-slice observations into one
all-eight-ready event. Replication is the better trade.

A weaker scope is not an option in either design. Republishing with a
device-scope (`sc0`-only) store and polling it with an `sc0`-only load hangs,
for the reason `mpk_atoms.cuh` already documents about `ld_nt_s32`: a load that
may hit vL1 keeps re-reading the line it already holds, so the spin only retires
if that line happens to be evicted. Every gate here is `sc0 sc1`; the wins come
from issuing them against more lines, not from weakening them.

### Caveats before porting into the kernel

- The flags live at `36 * 16` inside a shared counters buffer, so replicating
  them means growing `oproj_topk_counters` and re-checking the other slot
  constants in `gang_full_layer_fused_mi300.cuh`.
- The producer side is in Phase 6 (`gang_full_layer_fused_mi300.cuh`), which
  currently issues one `st_wt_u32`; it would issue K. At K=16 that is 16 stores
  from one lane per XCD per layer, which is what the NPS1 24% buys.
- The consumer needs `attn_release + (rank % K) * 8 * 16` where `rank` is the
  workgroup's index within its XCD, which Phase 7 already has as
  `tile_idx % tiles_per_xcd`.
- The harness measures the gate, not Phase 7 as a whole. `slicewait` was 9.68 us
  of a 27.52 us Phase 7 in the full model; a 7x cut there is worth roughly 8 us
  per layer, but confirming that needs a full-model run.

## Reproducing

```bash
hipcc -O3 --offload-arch=gfx950 -o bench_flagplace bench_flagplace.hip

# production geometry
HIP_VISIBLE_DEVICES=6 ./bench_flagplace --layers=200 --tiles=23 --flagstride=64

# the fix
HIP_VISIBLE_DEVICES=6 ./bench_flagplace --layers=200 --tiles=23 \
    --copies=16 --flagstride=256
```

Switching modes reloads the patched driver; see `set_mode.sh`. Note that script
hardcodes the module source path, and the host and container see that tree at
different mount points, so check `SRC` resolves before running it -- a failed
`insmod` leaves the box with **no** amdgpu driver loaded.

| knob | default | what it isolates |
| --- | --- | --- |
| `--copies=K` | 1 | Replicas of the eight-flag array; block rank `local` reads copy `local % K`. Total poll count is unchanged, so this isolates per-line sharing. |
| `--flagstride=B` | 64 | Byte stride between the eight flags within a copy. Production is 64. |
| `--tiles=N` | 23 | Workgroups per XCD, so `32N` polling waves. |
| `--hier=N` | 0 | 0 flat, 2 two-level, 3 two-level with the local flag replicated over 8 lines. |
| `--sleep=N` | 1 | `s_sleep(1)` repeats per spin iteration, i.e. backoff. |
| `--skew=N` | 0 | Per-XCD staggered producer release, in 10 ns ticks. |
