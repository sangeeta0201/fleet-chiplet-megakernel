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

**There is a fix that removes the regression rather than mitigating it.** AID
placement is not useful on its own, for the reason above -- but it is what makes
a *coherent* MTYPE safe to use, and the coherent MTYPE is what absorbs the 184
readers. Placing one copy of the flags in each AID and routing each XCD to its
co-located copy gets visibility to **0.85 us**, against 7.60 us for the same
placement under MTYPE_NC. It needs no global MTYPE change: the coherent MTYPE
applies to those two BOs alone and every other page in the process stays NC. See
[The fix](#the-fix-aid-local-placement-plus-a-per-bo-coherent-mtype). Flag
replication (7.5x, no driver knob at all) remains the fallback and is described
under [replication](#alternative-replicate-the-flag-array).

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

### Why the driver denies the coherent MTYPE, and why NPS1 gets it

#### The MTYPE is the whole mechanism, and the partition mode is not

Everything above compares NPS1 against NPS2, which confounds two variables: the
memory partition mode and the MTYPE the driver derives from it. `mtype_local`
separates them. It is a stock parameter that sets the MTYPE for local VRAM
directly (0 = RW, 1 = NC, 2 = CC), so NPS1 can be made to run on the same
MTYPE_NC that SPX+NPS2 is forced onto, with NPS1's fine interleave and single
coherence domain otherwise untouched.

| SPX+NPS1, `--layers=200 --tiles=23` | stride 4 | stride 64 | stride 256 | 32 pollers | copies=16 |
| --- | --- | --- | --- | --- | --- |
| `mtype_local=0`, MTYPE_RW (default) | 2.08 | **1.52** | 1.52 | 1.16 | 1.80 |
| `mtype_local=2`, MTYPE_CC | 1.92 | **1.52** | 1.56 | - | - |
| `mtype_local=1`, MTYPE_NC | 67.40 | **17.04** | 12.56 | 1.00 | 2.92 |
| SPX+NPS2 for comparison, MTYPE_NC | ~68 | **20.20** | ~13 | - | 2.84 |

poll_p50 in us. NPS1 on MTYPE_NC reproduces the entire regression -- the 27x, the
stride curve, the collapse to ~1 us when only 32 waves poll, and the recovery
under replication -- in one memory partition, at NPS1 interleave, with no AID
boundary anywhere. Conversely both coherent MTYPEs, RW and CC, absorb 184 readers
on one line at 1.52 us.

So the cost is not the AID boundary, not cross-AID distance, and not interleave
granularity. It is MTYPE_NC, and nothing else. An earlier version of this
document attributed NPS1's speed to its fine interleave spreading the directory
load; that is contradicted by the first and third rows above, which differ only
in MTYPE. The interleave argument survives only as an explanation of why the
driver *chooses* NC in the spanning case, not as the reason NC is slow.

#### The documented rationale for the choice

The documented hardware mechanism is a directory with a per-channel capacity budget. AMD's
coherence page for this part (Confluence 369688730, *MI300 Coherence between
Partition and NPS Modes*, quoted in `DF_REMAP.md` in the aid-local-hbm tree)
states it directly:

> **Why SPX+NPS2 is illegal:** DF-CS shadow tags only track ~8 L2 lines per
> channel. Fine NPS1 interleave ? 1 line/XCD/channel. Coarse NPS2 in SPX ? all
> 8 XCDs fill from one channel ? directory overflow, probes don't reach,
> MTYPE_RW cannot be saved by flush.

Each memory channel's DF Coherent Station keeps shadow tags recording which XCD
L2s hold lines it homes. NPS1's fine interleave scatters consecutive lines over
all channels, so a channel sees about one line per XCD and eight XCDs fit the
budget exactly. In NPS2 the AIDs are not interleaved, a contiguous structure
homes on one channel, and the directory overflows; the CS then loses sharers and
the probes that would invalidate their stale lines are never sent. "Cannot be
saved by flush" is the load-bearing part: the failure is a missing invalidate,
not dirty data in a cache, so no software fence recovers it.

Read this as the reason the driver demotes, not as the reason the demotion is
expensive; the `mtype_local` table above settles the latter. The measurements in
the next section also refine the capacity framing: the failure reproduces at
eight lines spread over eight channels and at 32 polling waves, which is far
inside any plausible directory budget, so what breaks in SPX+NPS2 looks
structural to the spanning configuration rather than a working-set effect.
SPX+NPS2 is characterised
as illegal rather than merely untuned: SPX reports `supported_nps_configs=NPS1`,
and the same rule appears on the AWS enablement page ("NPS count cannot exceed
compute partitions. SPX ? NPS1 only"). This benchmark only runs in SPX+NPS2 at
all because of the patched driver.

The same page gives the intended contract for AID-local memory: "Safe uses:
local XCDs only, or non-coherent MTYPE. Decode pinning to XCD0-3 vs XCD4-7 is
the intended software contract." Phase 7 does not satisfy the first option as
written: the K-parallel map forces wave *w* of every workgroup on all eight XCDs
to read slices 2w and 2w+1, so the far-AID XCDs unavoidably touch every flag
line, and re-partitioning O-proj's reduction dimension to fix that is a large
change. That leaves non-coherent MTYPE, which is where the driver already is.

The way out, developed below, is that "local XCDs only" is a constraint on the
*readers of a given buffer*, not on the work decomposition. Replicating the
flags into both AIDs lets every XCD read an in-domain copy while the K-parallel
map is left exactly as it is, because each copy carries all eight releases.
The contract is satisfied by duplicating the data rather than by re-partitioning
the computation.

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

Both coherent MTYPEs fail this way, at every stride:

| SPX+NPS2 | stride 4 | stride 64 | stride 256 |
| --- | --- | --- | --- |
| `mtype_local=0`, MTYPE_RW | hang | hang | hang |
| `mtype_local=2`, MTYPE_CC | hang | hang | hang |

No GPU fault, no ring timeout, no reset -- a clean infinite spin.

### What the hang actually is

`bench_hangdiag.hip` bounds every spin with a deadline and, at the instant a
wave gives up, reads the same address four ways: the `sc0 sc1` load the spin was
using, a plain load, an atomic RMW (which cannot be answered from a stale local
copy), and a load after `buffer_inv sc0 sc1`. It also has the producer read back
its own store, and the host read the flags after the kernel. Under MTYPE_CC in
SPX+NPS2, at 23 tiles and a 200 ms deadline:

```
producer self-check:  every XCD read back its own store correctly
host view of flags:   1 1 1 1 2 2 2 2      (all eight stores landed)
stuck waves:          736 of 736
example stuck wave:   plain=0 nosc=0 atomic=0 after_inv=0
```

The stuck-wave matrix shows each XCD only ever got past the one flag *it owns*:
xcd0 cleared f0 and stuck on f1, xcd2 cleared f2 and stuck on f3. So a store is
visible on its own XCD and to the host, and invisible to every other XCD.

Three candidate explanations, each tested and eliminated:

- *Dirty data trapped in the producer's L2.* Adding `buffer_wbl2 sc0 sc1` after
  every store (`--wb=1`) changes nothing.
- *Stale lines in the consumer's cache that need invalidating.* Issuing
  `buffer_inv sc0 sc1` before every polling load (`--cinv=1`) changes nothing,
  alone or combined with the writeback. All four wb/cinv combinations are
  byte-identical.
- *Address aliasing, one virtual address resolving per memory partition.* The
  `--alias=1` mode has all eight XCDs write a distinct value to the same dword,
  settle, then read it back. All eight read `0x1005`, the last writer's value,
  and so does the host. One address, one location, writes ordered.

What does change the outcome is the deadline. At 1000 ms all 736 waves are
stuck; at 5000 ms, 552 are. Waves do get through, on a timescale four orders of
magnitude beyond the ~1.5 us the same handshake takes under a coherent MTYPE in
NPS1.

That combination is only consistent with one mechanism. A polling wave takes the
line into its L2 before the store happens. Under a coherent MTYPE the L2 is
permitted to answer subsequent `sc0 sc1` loads from that copy, because the
protocol guarantees an invalidate probe will arrive when someone writes. In
SPX+NPS2 the compute partition spans two memory partitions and that probe is
never delivered, so the copy stays stale until the line is evicted by unrelated
capacity pressure -- which is what the 5000 ms partial recovery is. `buffer_inv`
does not rescue it because the invalidate it performs does not reach the level
holding the line, and the alias test succeeds because those reads happen once,
on lines not already cached from before the store.

This is the same failure mode the comment at the top of `bench_flagplace.hip`
records for a weaker load scope, one level further out in the hierarchy. It also
explains why the poller count does not matter: 32 waves hang exactly like 736,
because a single wave with a stale line is sufficient.

MTYPE_NC prevents all of it by forbidding the L2 to answer the poll at all.
Every load goes to the line's home, which is correct, and which is also why 184
of them per line queue into 20 us. The control confirms the instrument: under
MTYPE_NC the identical diagnostic reports 0 of 736 stuck and all four layers
complete.

### Nor per-BO on its own: a coherent MTYPE without placement also hangs

The shadow-tag account above is a *capacity* argument, which suggests an
escape: the global switch fails because the model's whole working set overflows
the directory, but the eight flags are 8 KiB. Give that one BO a coherent MTYPE,
leave every other page MTYPE_NC, and the coherent working set is eight lines.

Tested, and it does not work. `0003-amdgpu-per-bo-flag-mtype.patch` in this
directory adds `aid_local_flag_mtype` (0=off, 1=RW, 2=CC), applying only to
VRAM BOs carrying `AMDGPU_GEM_CREATE_COHERENT` in the spanning case; the
harness marks its flag array with `--flagmem=1`
(`hipDeviceMallocFinegrained`). A `DRM_INFO_ONCE` confirmed the branch fired
with the intended MTYPE.

| flag-page MTYPE | stride 64 | 256 | 512 | 4096 | unmarked control |
| --- | --- | --- | --- | --- | --- |
| NC (production) | 20.20 | 13.00 | -- | 12.64 | -- |
| RW (`flag_mtype=1`) | hang | hang | -- | -- | 20.16 |
| CC (`flag_mtype=2`) | hang | hang | hang | hang | 20.00 |

The controls confirm the change was surgical -- an unmarked BO in the same run
still measures 20 us. Stride was swept because the MI350 channel select hashes
PA[11:8], so a 256 B stride puts each of the eight flags on a different channel
and reduces the directory load to one line per XCD per channel, the same
condition that makes NPS1 work. It still hangs. Eight lines on eight channels is
not enough to stay inside the budget in this mode, so the demotion is not a
capacity workaround that a small BO can dodge.

Two smaller findings from the same work, both about what userspace can reach:
`hipDeviceMallocFinegrained` sets `KFD_IOC_ALLOC_MEM_FLAGS_COHERENT`, which this
ASIC's driver branch computes and then ignores, so it is a no-op on MTYPE
(measured 20.16 vs 19.88 baseline) and therefore usable as a free marker.
`hipDeviceMallocUncached` returns **host** memory, not VRAM (`is_vram=0` in the
probe) -- so the 21.6 us it measures is a page across PCIe, not MTYPE_UC, and
there is no HIP path to `EXT_COHERENT` at all.

What this rules out is *capacity* as the obstacle: shrinking the coherent
working set to eight lines on eight channels does not help, so the demotion is
not a directory-budget workaround that a small BO can dodge.

It does **not** rule out a per-BO coherent MTYPE, and the conclusion originally
drawn here -- "no MTYPE setting recovers this" -- was wrong. The flag array in
this experiment was an ordinary `hipMalloc` allocation: one buffer, one home
AID, and pollers on both sides of the boundary. Half the readers were therefore
outside the coherence domain that homes the line, which is the hang described in
the previous section, reproduced by construction. The missing ingredient is not
a smaller working set but **placement plus routing** -- see the next section,
where the same knob on a buffer whose readers are co-located reaches 0.85 us.

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

## The fix: AID-local placement plus a per-BO coherent MTYPE

Everything above says two things that look contradictory. A coherent MTYPE is
what absorbs 184 readers on a line, and a coherent MTYPE hangs in SPX+NPS2. Both
are true, and the second one has a precondition the first does not need:
coherence works among the XCDs **co-located with a line's home AID**, and fails
only for readers outside that domain. The visibility matrix in
`bench_hangdiag` measured exactly that, including the useful half -- a store
from a far-AID XCD still correctly invalidates the sharers that *are*
co-located with the line.

So the coherent MTYPE is not broken; it is scoped. If a buffer's readers are a
subset of the XCDs co-located with its home AID, coherence is sufficient for it.
That is a property software can arrange: place one copy of the flags in each AID
and route each XCD-half to its own copy. Producers write both copies, so the
information is identical; only the readers are partitioned.

### The approach: the knob already existed

The obvious way to get a coherent MTYPE on one buffer is a new driver
parameter, and the first attempt here was exactly that -- an `aid_local_bo_mtype`
gated on `AMDGPU_GEM_CREATE_AID_LOCAL`. It was unnecessary. Reading the driver
that this box already runs:

```c
if (amdgpu_aid_local_flag_mtype && is_vram && coherent &&
    adev->xcp_mgr &&
    !adev->xcp_mgr->num_xcp_per_mem_partition &&
    adev->gmc.num_mem_partitions > 1) {
        mtype = (amdgpu_aid_local_flag_mtype == 2) ? MTYPE_CC : MTYPE_RW;
```

where, a few lines earlier,

```c
bool coherent = bo->flags & (AMDGPU_GEM_CREATE_COHERENT |
                             AMDGPU_GEM_CREATE_EXT_COHERENT);
```

The comment above that branch says "UNCACHED is the marker", which is stale: the
code tests `COHERENT`, and `AMDGPU_GEM_CREATE_COHERENT` (bit 13) is listed in
`AMDGPU_GEM_CREATE_SETTABLE_MASK` in `amdgpu_gem.h`. So the `GEM_CREATE` ioctl
accepts it directly, and a single ioctl can request a home AID *and* a coherent
MTYPE on the same BO:

```
AMDGPU_GEM_CREATE_AID_LOCAL | AMDGPU_GEM_CREATE_AID_SELECT | AMDGPU_GEM_CREATE_COHERENT
```

`bench_aidsplit --coherent=1` ORs bit 13 into the AID-local allocation it
already performs. No driver change, no rebuild: the module in
`aid-local-hbm/src/amdgpu-mtype-test` already exposes the knob.

Two independent things then have to hold at once, which is why the earlier
attempts each failed on one of them. `aid_local_xcp_nc=1` keeps `is_local` false
so every ordinary page is MTYPE_NC -- unlike the global `mtype_local=2`, which
reached the same latency by promoting all VRAM including activations.
`aid_local_flag_mtype=2` promotes only BOs that ask for it -- unlike the earlier
per-BO test, which asked on a buffer it had not placed.

### The four cells

SPX+NPS2, `aid_local_xcp_nc=1 aid_local_flag_mtype=2`, `mtype_local` left at its
default, 23 tiles x 8 XCDs = 184 blocks, 120 layers, 79,488 samples per cell:

| cell | placement | flag MTYPE | `vis` p50 | p90 | all eight XCDs progressing |
| --- | --- | --- | --- | --- | --- |
| `cand` | one copy per AID, split routing | coherent (CC) | **0.85** | 0.97 | yes, 0 timeouts |
| `nc` | one copy per AID, split routing | NC | 7.60 | 9.61 | yes, 0 timeouts |
| `broken` | both halves on the AID0 copy | coherent (CC) | (0.80) | 0.94 | **no** -- XCD4-7 zero progress |
| `ord` | plain `hipMalloc` | NC | 17.51 | -- | yes (`poll` 20.92) |

`dmesg` confirms which path the driver took rather than leaving it inferred:
`FLAGMTYPE fired: knob=2 -> mtype=2`.

The two middle rows are what make this a mechanism rather than a number.

**`nc` isolates placement.** Identical AID-split buffers and identical routing,
coherence off: 17.51 us to 7.60 us. That is the replication effect and nothing
more -- two copies means two lines, so waves-per-line halves. It is consistent
with the `copies=2` row of the replication sweep and it confirms again that
AID placement per se buys nothing beyond the extra line.

**The coherent MTYPE is the remaining 7.60 to 0.85**, 8.9x, with placement and
routing held fixed. This is the term that no amount of replication reaches,
because it changes what answers the poll: the co-located L2 rather than the
line's home channel.

**`broken` is the correctness control.** Same coherent buffers, but every XCD
routed to the AID0 copy. XCD0-3 continued; XCD4-7 timed out on all 92 layers
they attempted, having never observed a single release. The 0.80 us figure is
in-domain XCDs only and is not a result -- it is there to show the cell ran. This
is the lost-probe failure of the previous section, reproduced deliberately, and
it is what establishes that the routing rather than the allocation is what makes
the coherent MTYPE safe.

**`ord` is the blast-radius control**, and the one the global `mtype_local=2`
failed. Ordinary `hipMalloc` flags in the same driver configuration still
measure 20.92 us and do not hang, so those pages really did stay MTYPE_NC.

### Why this is safe for the rest of the model

The rule is: *a coherent BO's readers must be a subset of the XCDs co-located
with its home AID.* Note it is a property of routing, not of allocation --
`broken` is correctly AID-placed and still fails.

Activations, weights and every other buffer are unaffected because they are
never promoted. They keep MTYPE_NC, and NC resolves every access at the line's
home, so a cross-AID read is always correct and merely slow. That is what the
driver's demotion is for, and it stays in force. This is the whole reason to
prefer the per-BO knob over `mtype_local=2` even though both measure the same
latency: the global switch promotes activation buffers too, and a megakernel
reuses those every layer and every token, so a stale cached line from the
previous iteration is a plausible hit that no probe arrives to invalidate. That
failure returns wrong logits with no crash.

There is a structural reason the per-BO version is easier to trust than "be
careful about placement". Look at how `broken` failed: the out-of-domain XCDs
**hung**, they did not return garbage. A consumer waiting on a flag that never
arrives stops, loudly. A consumer reading an activation from a stale line
returns a plausible number and continues. The two failure modes are separated by
buffer category -- synchronization variables fail loudly, data buffers fail
silently -- so confining the coherent MTYPE to synchronization variables puts
the entire change inside the category where mistakes announce themselves.

Two guards are still worth having:

- **Driver-side.** As written the branch promotes *any* coherent VRAM BO, so it
  trusts userspace to only mark buffers it has actually AID-placed. Marking a
  `hipDeviceMallocFinegrained` buffer that is not placed yields the `broken`
  hang, which is precisely what the earlier `--flagmem=1` experiments hit.
  Gating the branch on `AID_LOCAL && COHERENT` makes that mistake impossible to
  express.
- **Model-side.** Run the model with `aid_local_flag_mtype=0` (everything NC)
  and `=2`, and compare logits bitwise. Any routing mistake on a *data* buffer
  shows up as a numerical difference; a bit-identical result means no coherent
  buffer is being read out of domain.

### Where this stops working

The flags are cheap to do this way for a specific reason: they are tiny, and
their access pattern is *already* partitioned -- each XCD only needs the release
information, and every copy carries all of it, so replicating them costs eight
extra write-through stores per layer and no reads change hands.

That does not generalise to activations. If AID-local activations are ever
wanted for bandwidth, the coherent MTYPE stops being free, because both halves
of the XCDs genuinely need to read each other's output rather than a local copy
of the same information. Satisfying the co-location rule would mean replicating
the tensor and having each half read its own copy, which costs a duplicate write
of real data on the critical path. That is a different tradeoff and a separate
decision; nothing in this section argues for it.

### Caveats

- `bench_aidsplit` and `bench_flagplace` are not directly comparable. They pace
  layers differently and `bench_aidsplit` inherently carries two flag copies, so
  the honest apples-to-apples claim is the 7.60 -> 0.85 us within one harness,
  and 17.51 -> 0.85 us against the ordinary-flag path measured in the same
  session. NPS1's `vis` on `bench_flagplace` is about 1.21 us, so 0.85 us is at
  least in NPS1's league, but confirming "better than NPS1" needs
  `bench_aidsplit` run in NPS1 and it has not been.
- The `broken` cell leaves XCD4-7 spinning to their deadline. Do not run it
  without `--bailms`.
- Phase 7's K-parallel map is what makes the flags readable from both AIDs in
  the first place; this fix does not change the map, it replicates the flags so
  each half has an in-domain copy to read.

## Alternative: replicate the flag array

Replication is the cheapest thing that works without any driver knob, and it is
nearly free in layout terms: 16 copies x 8 flags x 64 B is 8 KiB. The 256 B
stride is marginally better at low K and converges by K=16:

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

### The fix, all four cells

`run_perbo.sh` does the whole thing -- builds, reloads the driver, runs the four
cells, and greps `dmesg` to confirm the driver took the per-BO path. It must
load the module that has the knob, which is **not** the tree `spxnps2.sh`
defaults to:

```bash
bash run_perbo.sh
```

By hand, the part that matters is the module parameters and the two flags on the
benchmark:

```bash
# the module with aid_local_flag_mtype lives in the mtype-test tree; the tree
# named amdgpu-6.16.13-2278356.24.04 has a gfx_v9_4_3.c newer than its own
# headers and does not build.
sed 's|^SRC=.*|SRC=/home/schowdha/aid-local-hbm/src/amdgpu-mtype-test|' \
    /home/schowdha/_apfix/spxnps2.sh > /tmp/spxnps2_mtype.sh

# xcp_nc=1 keeps every ordinary page NC; flag_mtype=2 promotes only BOs that ask.
# Do NOT also pass mtype_local=2 -- that is the global switch this replaces.
bash /tmp/spxnps2_mtype.sh "aid_local_steal_gb=0 aid_local_spx_nps2=1 \
    aid_local_xcp_span=1 aid_local_xcp_nc=1 aid_local_flag_mtype=2"

hipcc -O3 --offload-arch=gfx950 -o bench_aidsplit bench_aidsplit.hip

# cand: placed AND coherent -- the result
HIP_VISIBLE_DEVICES=6 ./bench_aidsplit --tiles=23 --layers=120 \
    --split=1 --coherent=1 --tag=cand

# nc: same placement, coherence off -- isolates the MTYPE's contribution
HIP_VISIBLE_DEVICES=6 ./bench_aidsplit --tiles=23 --layers=120 \
    --split=1 --coherent=0 --tag=nc

# broken: coherent, co-location deliberately violated. XCD4-7 will spin to the
# deadline, so keep --bailms bounded.
HIP_VISIBLE_DEVICES=6 ./bench_aidsplit --tiles=23 --layers=120 \
    --split=0 --coherent=1 --bailms=50 --tag=broken

# ord: ordinary pages must still be NC. ~20 us, and critically NOT a hang.
HIP_VISIBLE_DEVICES=6 ./bench_flagplace --layers=120 --tiles=23 \
    --flagstride=64 --tag=ord

sudo dmesg | grep FLAGMTYPE     # expect: fired: knob=2 -> mtype=2
```

Report all four. `cand` alone does not distinguish the fix from a global MTYPE
change, and without `ord` there is no evidence the rest of VRAM stayed NC.

### The replication fallback

```bash
hipcc -O3 --offload-arch=gfx950 -o bench_flagplace bench_flagplace.hip

# production geometry
HIP_VISIBLE_DEVICES=6 ./bench_flagplace --layers=200 --tiles=23 --flagstride=64

# the fix
HIP_VISIBLE_DEVICES=6 ./bench_flagplace --layers=200 --tiles=23 \
    --copies=16 --flagstride=256
```

To separate the MTYPE from the partition mode, bring the box up in SPX+NPS1 and
vary `mtype_local` alone. This is what shows the regression is not about NPS2:

```bash
# NPS1 on the MTYPE that SPX+NPS2 is forced onto -- reproduces the full 27x
bash spxnps1.sh "aid_local_steal_gb=0 mtype_local=1"
HIP_VISIBLE_DEVICES=6 ./bench_flagplace --layers=200 --tiles=23 --flagstride=64

# same box, same interleave, coherent MTYPE -- 1.52 us
bash spxnps1.sh "aid_local_steal_gb=0 mtype_local=0"   # or 2 for MTYPE_CC
```

Confirm which MTYPE is actually in force with
`dmesg | grep -o 'Using MTYPE_[A-Z]* for local memory'` rather than inferring it
from the parameters, because `aid_local_xcp_nc` overrides `is_local` after that
message is printed.

To diagnose the hang under a coherent MTYPE in SPX+NPS2:

```bash
hipcc -O3 --offload-arch=gfx950 -o bench_hangdiag bench_hangdiag.hip

bash spxnps2.sh "aid_local_steal_gb=0 aid_local_spx_nps2=1 \
    aid_local_xcp_span=1 mtype_local=2"

# bounded spin; reports what each stuck wave could see, four ways
HIP_VISIBLE_DEVICES=6 ./bench_hangdiag --tiles=23 --bailms=1000

# one address, eight writers: is it aliasing or coherence?
HIP_VISIBLE_DEVICES=6 ./bench_hangdiag --alias=1
```

| `bench_hangdiag` knob | default | what it isolates |
| --- | --- | --- |
| `--bailms=N` | 200 | Deadline per spin. Raising it to 5000 lets some waves through, which is how you tell an infinite hang from an eviction-limited one. |
| `--pdelay=N` | 10000 | Ticks the producer waits before storing, so consumers cache the line first. 0 stores before anyone polls. |
| `--wb=1` | 0 | `buffer_wbl2 sc0 sc1` after each store, to test whether data is trapped dirty in the producer's L2. |
| `--cinv=1` | 0 | `buffer_inv sc0 sc1` before every polling load, to test whether a stale consumer line can be invalidated by hand. |
| `--alias=1` | 0 | All eight XCDs write a distinct value to one dword and read it back, separating address aliasing from coherence failure. |

Switching modes reloads the patched driver; see `set_mode.sh`. Note that script
hardcodes the module source path, and the host and container see that tree at
different mount points, so check `SRC` resolves before running it -- a failed
`insmod` leaves the box with **no** amdgpu driver loaded.

| knob | default | what it isolates |
| --- | --- | --- |
| `--copies=K` | 1 | Replicas of the eight-flag array; block rank `local` reads copy `local % K`. Total poll count is unchanged, so this isolates per-line sharing. |
| `--flagstride=B` | 64 | Byte stride between the eight flags within a copy. Production is 64. MI350 selects the channel from PA[11:8], so 256 spreads the eight flags over eight channels. |
| `--flagmem=N` | 0 | Allocation flags for the flag array alone. 0 plain `hipMalloc`, else a `hipExtMallocWithFlags` value: 1 fine-grained (VRAM, marks the BO for `aid_local_flag_mtype`), 3 uncached (host memory, not VRAM). |
| `--tiles=N` | 23 | Workgroups per XCD, so `32N` polling waves. |
| `--hier=N` | 0 | 0 flat, 2 two-level, 3 two-level with the local flag replicated over 8 lines. |
| `--sleep=N` | 1 | `s_sleep(1)` repeats per spin iteration, i.e. backoff. |
| `--skew=N` | 0 | Per-XCD staggered producer release, in 10 ns ticks. |

`bench_aidsplit` knobs, which are the two axes of the fix:

| knob | default | what it isolates |
| --- | --- | --- |
| `--split=N` | 1 | Consumer routing. 1 sends each XCD-half to the copy in its own AID (readers co-located with the home, coherence sufficient). 0 sends everyone to the AID0 copy, which is the deliberate violation. |
| `--coherent=N` | 0 | Whether the two AID-local BOs carry `AMDGPU_GEM_CREATE_COHERENT`, i.e. whether `aid_local_flag_mtype` promotes them. With `--split=1` this isolates the MTYPE from the placement. |
| `--bailms=N` | 50 | Per-spin deadline. Required for `--split=0`, where four XCDs never observe a release. |
| `--flagstride=B` | 64 | Byte stride between the eight flags within a copy, as in `bench_flagplace`. |

Two structural notes on that harness, both there to keep the measurement from
depending on the thing being measured. Layers are paced by absolute deadlines
off the 100 MHz reference rather than a cross-block barrier, because a shared
barrier line would itself be the cross-AID spin under test. And release
timestamps live *inside* the AID-local buffers, at a 1 MiB offset from the
flags, so reading them does not cross the boundary either.
