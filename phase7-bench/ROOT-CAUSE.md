# Why Phase 7's slice wait costs 27x in NPS2

The short version: the attention release flags are allocated identically in both
modes and their addresses are almost irrelevant. What changes is the **MTYPE the
driver puts on every VRAM page**. NPS1 gets MTYPE_RW, where hardware keeps the
line coherent and hundreds of concurrent system-scope readers are absorbed.
SPX+NPS2 gets MTYPE_NC, where there is no hardware coherence, so every poll must
be serviced at the memory point and 1472 of them per layer queue up.

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
  736 waves issuing 1472 polls per layer against 8 cache lines.

Nothing about that allocation differs between NPS1 and NPS2.

## Address placement is not the mechanism

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

**NPS2 does respond to stride**, from 68 us down to about 12.5 us, and then
saturates -- flat from 256 B all the way out to 256 MiB. Once the flags are on
separate lines, moving them further apart buys nothing. There is a floor that
placement cannot break.

## The mechanism is fan-out, not latency

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
   NPS1 average.
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

That single bit explains the whole shape of the data. Under MTYPE_RW the
hardware keeps the line coherent, so a line being polled by 736 waves is served
out of the coherent hierarchy and readers are absorbed -- flat scaling. Under
MTYPE_NC there is no hardware coherence, so a correct `sc0 sc1` poll has to be
serviced at the memory point every time; requests to the same line serialize
there, which is both the linear scaling and the sensitivity to stride (with no
caching, line and channel conflicts become visible).

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
patch exists to prevent. MTYPE_NC is load-bearing, and the 27x is the price of
correctness under it -- not a misconfiguration to undo.

## The fix: fewer system-scope polls, not weaker ones

A weaker scope does not work. Republishing the flag with a device-scope
(`sc0`-only) store and polling it with an `sc0`-only load hangs, for the reason
`mpk_atoms.cuh` already documents about `ld_nt_s32`: a load that may hit vL1
keeps re-reading the line it already holds, so the spin only retires if that
line happens to be evicted.

What does work is cutting the number of `sc0 sc1` polls. `--hier` implements a
two-level gate that is entirely system-scope:

- Per XCD, the rank-0 block's wave 0 polls all eight remote flags, then
  republishes one XCD-local flag (`--hier=3` writes eight copies so the waiting
  blocks do not queue on a single line).
- Every other block's wave 0 polls only that XCD-local flag.
- Waves 1-3 of every block do not poll at all; they pick the result up through
  `__syncthreads()`.

Flag-polls per layer fall from `736 x 2 = 1472` to `8 x 8 + 176 x 1 = 240`.

NPS2 at production scale, 736 waves, `poll` p50:

| variant | us | vs baseline |
| --- | --- | --- |
| baseline (64 B stride) | 20.24 | 1.00x |
| stride 256 B only | 12.96 | 1.56x |
| spin backoff only (`--sleep=64`) | 13.92 | 1.45x |
| two-level gate (`--hier=2`) | 6.20 | 3.3x |
| two-level, replicated line (`--hier=3`) | 5.80 | 3.5x |
| `--hier=3 --flagstride=256 --sleep=32` | **5.08** | **4.0x** |

And it largely repairs the scaling that was the actual defect:

| polling waves | NPS2 baseline | NPS2 two-level |
| --- | --- | --- |
| 32 | 0.96 | 2.80 |
| 128 | 3.12 | 3.60 |
| 384 | 9.04 | 4.28 |
| 736 | 19.96 | 5.44 |

23x more pollers now costs 1.9x instead of 21x. The residual gap to NPS1
(about 5 us against 1.5 us) is the two serial hops the gate introduces plus the
240 polls it still issues.

### Caveats before porting this into the kernel

- **It is a 2x regression in NPS1** (1.52 -> 3.16 us), because the second hop
  buys nothing when readers are already absorbed. Gate it on NPS2.
- The harness measures the gate, not Phase 7 as a whole. Phase 7 also has to
  keep the per-wave `buffer_inv` placed at each wave's own observation point,
  which the `__syncthreads()` fan-out changes the shape of: with the two-level
  gate the block learns "all eight slices are published" at once, so the
  per-slice overlap that `MPK_OPROJ_SPLIT_SLICE_WAIT` exists to exploit is gone.
  Whether that trade is net positive needs a full-model run.
- `--flagstride` cannot be changed in the kernel without growing
  `oproj_topk_counters` and re-checking every other slot constant in
  `gang_full_layer_fused_mi300.cuh`, since the flags live at `36 * 16` inside a
  shared counters buffer.

## Reproducing

```bash
hipcc -O3 --offload-arch=gfx950 -o bench_flagplace bench_flagplace.hip

# production geometry
HIP_VISIBLE_DEVICES=6 ./bench_flagplace --layers=200 --tiles=23 --flagstride=64

# the fix
HIP_VISIBLE_DEVICES=6 ./bench_flagplace --layers=200 --tiles=23 \
    --hier=3 --flagstride=256 --sleep=32
```

Switching modes reloads the patched driver; see `set_mode.sh`. Note that script
hardcodes the module source path, and the host and container see that tree at
different mount points, so check `SRC` resolves before running it -- a failed
`insmod` leaves the box with **no** amdgpu driver loaded.

| knob | default | what it isolates |
| --- | --- | --- |
| `--flagstride=B` | 64 | Byte stride between the eight flags. Production is 64. |
| `--tiles=N` | 23 | Workgroups per XCD, so `32N` polling waves. |
| `--hier=N` | 0 | 0 flat, 2 two-level, 3 two-level with the local flag replicated over 8 lines. |
| `--sleep=N` | 1 | `s_sleep(1)` repeats per spin iteration, i.e. backoff. |
| `--skew=N` | 0 | Per-XCD staggered producer release, in 10 ns ticks. |
