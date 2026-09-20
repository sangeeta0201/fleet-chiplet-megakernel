# SPX+NPS2: release visibility, poller contention, and the AID0 clamp

Follow-on to `SPX_NPS2.md`, which left SPX+NPS2 at 3.422 ms/token against
SPX+NPS1's 1.522 ms and named two open items: the AID0 VRAM clamp, and
`aid_local_xcp_span=1` breaking the mode outright. Both are resolved here. One
produced a 16.5% win; the other produced a clean negative result.

Everything below was measured on e5 (`HIP_VISIBLE_DEVICES=8` when six dies are
SPX; the index shifts with how many dies are DPX -- resolve it, do not assume),
SPX/NPS2, recipe
`MPK_TERM_RECHECK=1 MPK_AID_SPLIT_FLAGS=1 MPK_MOE_XCD_PAIR=0`.

## 1. Result summary

| change | ms/token | note |
|---|---|---|
| `SPX_NPS2.md` baseline | 3.422 | |
| same recipe, this session | 3.270 | baseline drifts ~0.15 ms between container restarts |
| driver: AID0 clamp fixed, VRAM balanced over both AIDs | 3.418 | **neutral** |
| **`MPK_NARROW_GATE_POLL=1`** | **2.729** | **-16.5%** |
| `+ MPK_OPROJ_PIPE_SLICE_MFMA=1` | 3.272 | neutral |
| `+ MPK_NARROW_HIER_POLL=1` | 2.745 | neutral |

SPX+NPS1 is 1.522 ms, so the gap went from 2.15x to 1.79x.

**Caveat, and it is a real one: the decode output is not reproducible, baseline
included.** See section 6. Every number above is timing-only.

## 2. The driver: `aid_local_xcp_span` was unusable, and the idea does not pay

Patch: `aid-local-hbm/0011-spx-nps2-aid-balance.patch`.

### The gate was ambiguous

`aid_local_xcp_span=1` used to make SPX+NPS2 impossible: all 8 dies came up DPX
despite `aid_local_spx_nps2=Y`, every later SPX write returned EBUSY with nothing
holding `/dev/kfd`, and only a reload with `xcp_span=0` recovered.

The branch in `amdgpu_bo_placement_from_domain()` gated on
`!adev->xcp_mgr->num_xcp_per_mem_partition && num_mem_partitions > 1`. That
counter is `num_xcps / num_mem_partitions`, which is **0 for two different
reasons**: a real SPX+NPS2 (`1/2`), and the `kzalloc`'d state before any
`amdgpu_xcp_init()` has run. That init never runs at load: `aqua_vanjaram.c`
calls `amdgpu_xcp_mgr_init()` with `AMDGPU_UNKNOWN_COMPUTE_PARTITION_MODE`,
which is `-1`, which is `AMDGPU_XCP_MODE_NONE`, and the call is gated on
`init_mode != AMDGPU_XCP_MODE_NONE`.

`gmc.num_mem_partitions` is already 2 by the time TTM comes up, so the old gate
widened the aperture for the driver's **own** early BOs -- GART, PSP
TMR/firmware, CSA, page tables. With `TTM_PL_FLAG_TOPDOWN` those land at the top
of the span, i.e. in AID1, while partition setup and
`__aqua_vanjaram_get_xcp_mem_id()` both nominate range 0. Setup fails,
`__amdgpu_xcp_switch_partition_mode()` re-queries the hardware and leaves DPX,
and later switches fail inside `pre_partition_switch`.

Fix: gate on a positive steady state instead of an uninitialized default.

```c
} else if (amdgpu_aid_local_xcp_span && adev->gmc.mem_partitions &&
           mem_id >= 0 && adev->xcp_mgr &&
           adev->xcp_mgr->num_xcps == 1 &&
           adev->xcp_mgr->mode == AMDGPU_SPX_PARTITION_MODE &&
           adev->gmc.num_mem_partitions > 1 &&
           abo->tbo.type == ttm_bo_type_device) {
```

`num_xcps` is 0 before init, 2/4/8 in DPX/QPX/CPX, and 1 only in an established
SPX; `mode` excludes `AMDGPU_XCP_MODE_TRANS`. `ttm_bo_type_device` keeps the span
to user BOs -- the model weights, which is the whole point -- and leaves every
kernel-internal BO on the stock single-range path, which also covers the
post-switch window. With this, SPX is offered on every die and six come up
SPX+NPS2 (05/15 stay wedged in DPX for unrelated, pre-existing reasons).

### Widening the aperture does nothing; you have to choose per BO

With the gate fixed, `xcp_span=1` still changed **nothing**: the per-XCD skew on
the two 2260 MiB MoE weights stayed at `-99 ns`, byte-identical to `xcp_span=0`.
This part has a large BAR, so `real_vram_size == visible_vram_size`,
`TTM_PL_FLAG_TOPDOWN` is never set, and `drm_buddy` fills bottom-up -- every BO
still landed in range 0. A wider window lifts the cap; it does not distribute.

So the patch instead picks, per BO, the window of the **least-filled** range
(`aid_balance_bytes[]` in `struct amdgpu_gmc`). Each BO stays inside one range
exactly as the stock path does, so nothing straddles the boundary and MTYPE
selection is unchanged, but consecutive large buffers alternate.
gpt-oss-120b's MoE weights arrive as **1126 MiB BOs, two per 2260 MiB tensor**,
so each tensor ends up half in each AID.

### It works, and it is neutral

888 balance decisions, strictly alternating, totals dead even
(`r0=443232 r1=443232 MiB`). The systematic straggler is gone:

| | AID0 clamp | balanced |
|---|---|---|
| `[XCDSPIN]` "last" | XCD4 2338/2500 = **94%** | `1/7/157/1202/183/671/151/760`, XCD4 **6%** |
| spin | 8843 | 8231 |
| drain total | 10753 | 10266 |
| ms/token | 3.422 | **3.418** |

So the +98 ns MoE weight skew is real, and removing it removes the consistent
straggler, but it is **not on the critical path**. That agrees with the
L2-residency measurement (97% hit) and the CAKE fabric reading (~0.04 GB/s) in
`SPX_NPS2.md`, and with this branch's own MoE-weight placement arms. The
hypothesis that the clamp accounted for the residual ~2x is falsified. Keep the
patch for the correctness of `xcp_span` and the loss of the straggler; do not
expect it to close the gap.

Note `gmc_v9_0.c:1195` and `:1219` still use the same `!num_xcp_per_mem_partition`
idiom, but those only pick an MTYPE rather than a placement, and `xcp_nc=Y` is
the configuration every existing measurement was taken under. Do not
"consistency-fix" them without re-baselining.

## 3. What a release actually costs

`relobs2.cpp`. Timed entirely on the **writer's** clock, so nothing assumes
`s_memrealtime` is phase-aligned across XCDs: ping-pong with one reader XCD per
launch (no poll-order bias), plus a fan-out arm.

| flag placement | writer | same-AID | cross-AID | fan-out, last of 7 acks |
|---|---|---|---|---|
| plain `hipMalloc` (spanning, NC) | XCD0 | 781 ns | 992 ns | 2340 ns |
| plain `hipMalloc` (spanning, NC) | XCD4 | 1118 ns | 1045 ns | 3223 ns |
| AID-local RW in AID0 | XCD0 | 846-922 ns | **timeout (3 s)** | - |
| AID-local RW in AID1 | XCD0 | **timeout (3 s)** | 881-917 ns | - |
| spanning NC, `nt` poll | XCD0 | **timeout** | **timeout** | - |

The cross-AID penalty is **+211 ns** round-trip, about 105 ns one way, matching
the ~98 ns crossing seen elsewhere. Two existing rules are confirmed directly
rather than by inference: an AID-local **RW line is invisible to the other AID**
(so split flags must be replicated per AID -- though a writer in AID0 *can*
publish into the AID1 replica, which is what makes replication work), and an
**`nt` poll never observes the release at all**, so `MPK_SYS_POLL_LOAD=2` is a
correctness requirement rather than a tuning knob.

Raw visibility is therefore ~1-3 us and cannot by itself explain a 40 us wait.

## 4. The poller-contention cliff

`relobs3.cpp`, sweeping pollers per XCD. Last-of-7-acks, writer XCD0:

| pollers/XCD | total pollers | single spanning NC line | two AID-local RW replicas |
|---|---|---|---|
| 1 | 8 | 2426 | 2484 |
| 2 | 16 | 2656 | 2477 |
| 4 | 32 | 3243 | 2476 |
| 8 | 64 | 5060 | 2502 |
| 16 | 128 | **26620** | 2491 |
| 24 | 192 | **40135** | 2470 |
| 32 | 256 | **52251** | 2484 |

A shared spanning-NC line collapses past ~64 pollers; AID-local RW replicas are
**flat to 256 pollers**. And it is not simply pollers-per-line: the split arm
carries **128 pollers per replica** at 2484 ns where the single line costs
26620 ns for the same 128. The MTYPE is the variable, not the fan-out width --
which is also why this is mode-specific, since the same shared line is MTYPE_RW
in NPS1 and is demoted to MTYPE_NC in NPS2.

`40135 ns` at 192 pollers is a near-exact match for fleet's measured slot 5
(40288 ns), which is tempting and **wrong**: `attn_release` is already on an RW
replica (`gang_full_layer_fused_mi300.cuh:562`), and the RW column is flat, so
slot 5 is not release contention. It is genuine idle waiting. Record the
coincidence so nobody re-derives it.

## 5. The win: `MPK_NARROW_GATE_POLL`

Fleet's `attn_release_wait` is already split twice over -- `MPK_ATTN_SLICE_RELEASE`
has each XCD publish its own slice as soon as it is ready, each O-proj wave waits
only for the two slices its K interval reads (wave w -> XCDs 2w, 2w+1), and the
O-proj workers skip the slot-5 gate entirely (`if (!does_oproj)`). Slot 5 is paid
only by the ~168 MoE-only workers, which `:1351` states "read nothing from
attn_out": it is a pure ordering fence, and they are transitively gated by
`routing_ready` in slot 7 anyway. **Narrowing the dependency cannot help;
narrowing the poll does.**

`MPK_NARROW_GATE_POLL=1` has one thread poll per block instead of all 256,
cutting ~240 coherent single-line reads per layer to ~30. 3.270 -> **2.729 ms**,
and the per-phase table shows the gain lands in the polling slots rather than
smeared across the layer:

| slot | narrow off | narrow on |
|---|---|---|
| 5 attn_release_wait | 40288 | 37144 |
| 6 oproj+rmsnorm+router | 3243 | 377 |
| 7 topk_wait | 23852 | **17257** |
| TOTAL | 95512 | **84866** ns/layer |

It is sound because the `__syncthreads()` sits **before** the cross-XCD acquire,
so no wave can invalidate early and refill vL1 with pre-release lines; the branch
is block-uniform (`does_oproj` is a per-worker constant); and `buffer_inv` is
still executed per wave, which is what the acquire needs.

`MPK_NARROW_HIER_POLL` applies the same idea to the `oproj_hier` poll at `:1596`,
where ~168 blocks x 4 waves hit one line -- and is **neutral** (2.745 vs 2.711).
Those ranks are already idle, so cutting their polling traffic buys nothing and
the `__syncthreads` costs about what the polls saved. It is left off. Do not
expect the remaining all-threads polls (`gang_moe_fused` `d_barrier_rel` and
`final_release`, `gang_linear` `oproj_xcd_ready` and the slice `rel/rel0/rel1`)
to pay just because they have many pollers.

## 6. Open: the output is not reproducible

Greedy decode (`argmax`, `ignore_eos`, fixed prompt) should be bit-identical run
to run. It is not, **with no flags changed**:

| arm | rep1 | rep2 |
|---|---|---|
| baseline | `...We need to answer: "The capital of France" is Paris` | `The user asks: ... That's a straightforward` |
| `MPK_NARROW_GATE_POLL=1` | `...That's a` | `...That's a` |
| `MPK_OPROJ_PIPE_SLICE_MFMA=1` | `..."The capital of France" is "` | `...Likely they` |

So comparing generated text cannot gate a change: the control does not
reproduce, and a differing continuation is not by itself evidence that a flag
broke something. Use `--verify` (runs PyTorch and Mirage and compares
intermediates) or `PPL_MODE=1 --ppl-corpus wikitext2`. The tree already
describes a likely cause at `:1576` -- "rmsnorm_out_moe is *correct* in the final
dump ... while swiglu_out is wrong, so the MoE reads it transiently before it
settles. The buffer is fine; the ordering is not." This should be closed before
any of the timing results here are treated as final.

## 7. Traps that cost time

- **Do not assume SPX maps workgroup i to XCD i.** The first version of the probe
  picked its writer with `blockIdx.x == wxcd && xcd == wxcd`; that predicate was
  never satisfied, nothing published, and the readers spun for nine minutes.
  Elect one representative block per XCD with `atomicCAS` on a per-XCD slot, and
  give every spin a deadline so a failure prints a code instead of wedging.
- **`kernel.dmesg_restrict = 1`.** Non-root `dmesg` returns stale output, so a
  driver `DRM_INFO` you just added looks like it never fired. Read the log
  through a privileged container.
- **`build.sh` defaults to `CC=gcc-12`, which is absent**; this kernel
  (6.8.0-136-generic) was built with **gcc-13**. Wrong compiler dies at
  `drm_gem_ttm_helper.o` with `Error 127`.
- **Identify the live driver tree by srcversion**, matching
  `/sys/module/amdgpu/srcversion` against
  `modinfo -F srcversion <tree>/amd/amdgpu/amdgpu.ko`. There are six candidate
  trees and dmesg is not proof.
- **`[MAP]` probes only the first 64 MiB at offset 0**, so for a multi-BO tensor
  its `aid=`/`skew=` describes the first chunk, not the tensor. It reported
  `aid=1 skew +95` for a tensor the driver counters prove is split 50/50.

## 8. Slot 8 decomposed: it is VARIANCE, not bandwidth

Slot 8 (`moe w13+swiglu+w2`) is the second-largest NPS2 excess after the waits
(MoE-only ranks: 10224 ns/layer in NPS1 against 18397 in NPS2). Four separate
memory-system explanations were tested and all failed, so the phase was
decomposed with the tree's own `MPK_MOE_INNER_TIMING=1`.

Same instrument, same sample count (n=3132 per arm), mode the only variable.
These runs carry printf overhead (7.0 / 7.6 ms end to end), so the **ratios** are
the result, not the absolute us:

| sub-phase | NPS1 | NPS2 | delta | ratio |
|---|---|---|---|---|
| W13 `dec` | 0.45 | 0.35 | -0.10 | - |
| **W13 `compute`** | 6.89 | 9.42 | **+2.53** | **1.37x** |
| W13 `arrive` | 0.60 | 0.39 | -0.21 | - |
| W2 `dec` | 0.50 | 0.33 | -0.17 | - |
| W2 `prep` | 2.53 | 2.74 | +0.21 | 1.08x |
| **W2 `barrier`** | 6.30 | **15.21** | **+8.91** | **2.41x** |
| **W2 `compute`** | 2.30 | 4.60 | **+2.30** | **2.00x** |
| **MoE total** | **19.57** | **33.05** | **+13.48** | **1.69x** |

`dec`, `arrive` and `prep` are flat or *better* in NPS2. The cost is the W2
`barrier` (+8.91, i.e. 66% of the whole MoE delta) and then real compute
(W13 +2.53, W2 +2.30).

**The load-bearing observation is the mismatch between two of those rows.**
W13's own total grows by only +2.22 us, but the barrier that waits on W13 grows
**+8.91 us** -- four times as much. That barrier waits for the **last** W13 tile,
so it bills *max over tiles*, not the mean. The dominant term is therefore the
**spread across W13 tiles widening in NPS2**, not W13 getting slower on average.
It also explains why narrowing that barrier's poll was neutral
(`MPK_NARROW_MOE_BAR_POLL`): there is no contention to remove, the tiles
genuinely arrive further apart.

So the residual NPS2 gap is a **variance** problem, which is consistent with the
rest of this document: the `[XCDSPIN]` straggler asymmetry, and slots 5 and 7
absorbing skew rather than containing a cause.

### What was ruled out for slot 8, and how

| hypothesis | test | result |
|---|---|---|
| MoE weights are MTYPE_NC in NPS2, RW in NPS1 | `mtypebw.cpp`: same pages, same AID, NC vs RW streaming read | **+4.2%** (3168 vs 3302 GB/s) -- cannot explain 1.8x |
| W13->W2 barrier poll contention | `MPK_NARROW_MOE_BAR_POLL` | **neutral-to-worse** (2.794 vs 2.767 ms; slot 8 up) |
| AID0 VRAM clamp / weight placement | driver balance patch | **neutral** (3.418 vs 3.422 ms) |
| live cross-XCD `swiglu_out` handoff | `handoff2.cpp` | **+19%** for far-side XCDs -- real, but a fraction of one sub-phase |

`mtypebw.cpp` is self-checking: the AID0 buffer is faster from XCD0-3 and the
AID1 buffer from XCD4-7 (~14%), and plain `hipMalloc` is AID0-fast, which
independently confirms the range-0 clamp.

### `handoff2.cpp` and a trap it exposes

The handoff probe writes a buffer from one XCD, publishes an epoch, and has every
other XCD NT-read it -- which is what `gang_moe_fused_mxfp4_mi300.cuh:4741` does
with `d_swiglu_out` ("just written by another XCD"). Consumer read time per
round, 4 MiB, producer XCD4:

| buffer | AID0 consumers | AID1 consumers | correct? |
|---|---|---|---|
| plain `hipMalloc` (NC) | 380k ns | 454k ns (**+19%**) | yes, all 8 |
| `alloc_in_aid(0)` (RW) | 347k ns | **111k ns** | **NO -- stale on 3-4 XCDs** |
| `alloc_in_aid(1)` (RW) | **112k ns** | 349k ns | **NO -- stale on 3-4 XCDs** |

**Do not "optimize" `swiglu_out` onto an AID-local RW buffer.** It looks like a
3.1x win and is silently wrong for half the XCDs: the first version of this probe
had no data check and reported exactly that fake speedup. Every fast number above
is a consumer that never observed the producer's writes. Plain NC is correct for
all 8 consumers, which is why the kernel uses it.

Probe caveat: the NC arm with producer XCD0 times out reproducibly (a rendezvous
artifact in the probe, not a hardware result). The XCD4-producer arm is the clean
one.

### Next

1. **W13 tile skew.** The barrier bills the slowest tile, so measure the per-tile
   W13 completion spread directly -- `MPK_MOE_INNER_WIDE` widens the sample from
   tile 0 to every 37th tile, which is the instrument for exactly this.
2. **W2 `compute` doubling** (2.30 -> 4.60). It reads `swiglu_out` (+19%
   far-side) and the W2 weights (+4%), which do not add up to 2x.

## 9. WHY sync is slower in NPS2: NC polls cannot be cache-shared

Same binary, mode the only variable. `relobs3.cpp`, last-of-7-acks vs pollers on
ONE line:

| pollers | 8 | 64 | 128 | 192 | 256 |
|---|---|---|---|---|---|
| **NPS1** (plain `hipMalloc` = MTYPE_RW) | 2565 | 2561 | 2457 | 2476 | **2446** |
| **NPS2** (plain `hipMalloc` = MTYPE_NC) | 2426 | 5060 | 26620 | 40135 | **52251** |

Single-operation latency is IDENTICAL between modes (round-trip 852 ns NPS1 vs
781 NPS2; cross-AID penalty +220 vs +211). So it is not distance -- it is that an
NC line cannot be shared through cache, so every poller's `sc0 sc1` read goes to
the coherency point and they SERIALISE. On a coherent line 256 readers each hit
their own cached copy and the fan-out stays flat. AID-local RW replicas are flat
in NPS2 too (2470-2502), which is the fix.

`gang_oproj_topk_moe_fused_mi300.cuh:190` states the broken assumption outright:
"polling is XCD-local (hot in L2, no cross-XCD contention)". True in NPS1; in
NPS2 the line is NC and cannot be hot in L2.

This also explains `MPK_NARROW_GATE_POLL` (-16.5% in NPS2, **0% in NPS1**): it
only reduces the NUMBER of NC reads, and in NPS1 there was nothing to serialise.

Localizability, by operation:

| operation | localizable? | status |
|---|---|---|
| release FLAGS (reads) | yes -- per-AID RW replica makes reads cacheable | done: `attn_release`, `layer_release`, `oproj_hier`, MoE per-XCD |
| `routing_ready` | yes, same mechanism | added (`MPK_AID_SPLIT_ROUTING`), **measured neutral**, off |
| arrival COUNTERS (atomics) | **no** -- a counter needs one home, and the tree notes they serialise at the XCD L2 boundary wherever they live | only lever is fewer of them (hierarchy) |

## 10. Session result and the one remaining lever

3.422 -> **2.560 ms**; gap vs SPX+NPS1 (1.695 on the same build/flags)
**2.02x -> 1.51x**. Compute slots 1/2/3 are at parity (+28, +21, -17 ns) and W13
total is now BELOW NPS1 (7.84 vs 7.94).

What is left is **the price of the interleave itself**: it buys bandwidth by
making every MoE weight read ~50% remote, which shows up exactly where predicted
-- W13 compute 6.89 -> 7.92, W2 compute 2.30 -> 3.01 -- and the W13->W2 barrier
then amplifies it because it bills the slowest tile.

So interleave and split are mutually exclusive for the same buffer, and the
remaining headroom needs the SPLIT: place expert E's weights in one AID and route
E's tiles to an XCD pair in that AID, so each half streams its own range locally
(3.5 TB/s each, concurrently, all local) instead of trading locality for
bandwidth. The weights are `[E, W13_WGS, wg_bytes]` indexed purely by
`expert_id`, so this needs ZERO extra memory; the work is making the dispatch
pick the consumer from the data's location, plus a fallback for when the 4
selected experts are not a 2/2 split across AIDs. `MPK_MOE_XCD_PAIR=1` becomes
the enabler rather than the regression it is today, and `aid_local_ilv_mb=0`
turns the interleave off for those slots.
