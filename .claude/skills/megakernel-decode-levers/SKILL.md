---
name: megakernel-decode-levers
description: >-
  Catalog of the decode-latency optimizations that were measured to work in
  fleet's gfx950 persistent megakernel (GPT-OSS 120B, 1.93 -> 1.64 ms/token),
  each with its mechanism, measured delta, precondition, and the null results
  that were kept behind opt-in flags. Use when porting a known-good lever to a
  new model, deciding what to try next on a fused decode path, or checking
  whether a proposed optimization has already been measured.
---

# Megakernel decode levers

Seven levers moved GPT-OSS 120B batch-1 decode from 1.93 to 1.64 ms/token on
MI355X. This catalogs what worked, what did not, and the transferable pattern
behind each, so a port to another model starts from mechanism rather than from
a flag name.

Every number here is a per-run median of alternating control/variant pairs with
the generated-text hash checked identical. Cite the commit, not this file, in
new work.

For the loop that produced them, use `optimize-megakernel`. For ISA and hazard
claims, use `cdna4-expert`. For GLM-5 744B applicability, read
[references/glm5-port-status.md](references/glm5-port-status.md).

---

## The transferable patterns

These are the reusable ideas. The flags below are instances of them.

### 1. Replace a global rendezvous with a per-slice release

A consumer should wait only on the producers of the bytes it actually reads.
An all-XCD barrier makes every consumer wait on the slowest producer on the
machine even when it reads one eighth of the output.

Ask: does this barrier's consumer read the whole output, or a contiguous slice
of it? If a slice, publish per-slice and have consumer `w` wait on only the
producers of its slice.

### 2. Issue last what you will wait on — counted `vmcnt` ordering

gfx950 retires VMEM **in issue order**, so a counted `s_waitcnt vmcnt(N)`
cannot retire a late load without also retiring every earlier one. Issue a
small latency-critical payload *after* a large weight burst and the only wait
reaching it is shallow; issue it first and the wait on it drags the entire
burst.

Two traps: `vmcnt` is **per-wave**, so any work split that makes waves carry
different load counts breaks the counted wait — re-cut the split wave-uniform.
And read `vmcnt(N)` at the first wait before and after any change: an
instruction-count win that *shrinks* depth is a loss in a latency-bound loop.

### 3. Prefer a static ownership map to a data-derived one

Deriving "which expert does this tile run" from routing data costs a dependent
chain of cache-missing loads in every tile prologue, and blocks the weight DMA
until routing completes. A static map (`expert = xcd_id >> 1`) lets the weight
fetch start while the router is still choosing later picks, and lets metadata
ride in spare `tile_idx` bits instead of being looked up.

The win is overlap, not the removed loads.

### 4. Publish producer metadata at selection time — in selection order

Publishing early is only correct if the map is **selection-ordered**. Slot 0 of
an ascending/compacted map is the lowest id, not the first-selected one, so
publishing the first pick into it runs the wrong weights. This failed as
`MPK_EARLY_ROUTING` with three different text hashes in three runs.

### 5. Cross-lane VALU instead of `ds_bpermute`

`__shfl_xor` lowers to `ds_bpermute` — an LDS round trip and an `lgkmcnt` wait
to move data that never left the register file. On gfx950 a six-step xor
butterfly (32/16/8/4/2/1) becomes `v_permlane32_swap_b32` (xor-32),
`v_permlane16_swap_b32` (xor-16), and DPP `row_shl` for the four intra-row
steps, which fold into the add because `v_add_f32` takes the DPP swizzle on
src0. Two swaps and four adds instead of six shuffles, bit-identical order.

**`s_nop 1` before each stage is required, not defensive.** CDNA4 ISA Table 11
gives two wait states for a VALU write followed by a DPP or PERMLANE read of
the same register, and every stage reads what the previous add wrote.

### 6. Hide latency by issue placement, not by a new buffer

Issuing a high-latency fetch just after an existing counted wait lets an
already-present pipeline absorb it, with no extra registers and no change to
resource tiers. Cheaper than double-buffering, which costs registers and can
capture all the cost and none of the benefit (see the GLM-5
`MPK_MLA_DECODE_DBLBUF` note: a second buffer for a loop that never goes round).

### 7. Audit flags already in the tree but shipping off

The single cheapest win in the series was **1.902 -> 1.877 ms for zero code
change** — five previously validated flags that had never been defaulted on.
Before writing anything, enumerate the tree's optimization flags and diff
"validated" against "default".

---

## What worked

| lever | delta (ms) | pattern |
|---|---|---|
| default-on flips (5 flags, no code change) | 1.902 → 1.877 | 7 |
| `MPK_ATTN_SLICE_RELEASE` | 1.880 → 1.864 | 1 |
| `MPK_W13_T0_COUNTED_HANDOFF` | 1.865 → 1.840 | 2 |
| RMSNorm ssq off LDS (`2ee5278`) | 1.851 → 1.835 | 5 |
| `MPK_MOE_XCD_PAIR` (`9d478ea`) | 1.706 → 1.642 avg of 3 pairs | 3, 4 |
| W13 fragment recycle/pipeline (`5a993f4`, `8ddd95f`) | shipped default | 2, 6 |
| W13 tile-1 bias behind tile-0 SwiGLU (`8b9efc7`) | shipped default | 6 |
| routing release epochs derived (`948cc97`) | shipped default | 3 |
| LM-head groups via recycled LDS (`25b5a30`) | shipped default | 6 |

### The five default-on flips (`ff46bd4`)

`MPK_OPROJ_KMAJOR`, `MPK_QKV_LDS_NORM`, `MPK_TOPK_LOCAL_MAX3`,
`MPK_W13_T0_MFMA_UNROLLED`, `MPK_W2_T0_MFMA_UNROLLED`.

### `MPK_ATTN_SLICE_RELEASE`

Each XCD's merge owns a contiguous 512-element slice of the O-proj's 4096-wide
reduction and releases it as it lands. O-proj wave `w` waits on XCDs `2w` and
`2w+1` only, instead of on the slowest merge on the machine. `NUM_REQS == 1`
and the K-parallel O-proj shape are `static_assert`ed, not silently ignored.

### `MPK_W13_T0_COUNTED_HANDOFF`

Issues the W13 handoff payload above the 23 KiB tile-0 weight burst and retires
it with a counted `vmcnt`. Payload-last means the only wait reaching it is
`vmcnt(0)`; payload-first, the two rendezvous retire at `vmcnt(25)` and
`vmcnt(23)` with the weights still in flight and both gated on a weight fetch
neither reads. Both work splits re-cut wave-uniform (46-47 activation
`dwordx4` per wave, one W13 scale tile per wave). Requires `MPK_W13_PREQUANT`
and `MPK_W13_LINEAR_LOAD`, `#error`-enforced.

### `MPK_MOE_XCD_PAIR`

Expert ownership is `xcd_id >> 1`; 46 weight groups split 23 per pair member.
The expert id is published the moment TopK selects it, as a packed
`(epoch, expert)` u64 sent only to the two XCDs that will run it, and the mask
is written in selection order rather than compacted ascending. Tile decode then
reads neither `moe_mask` nor routing: the expert rides in `tile_idx[15:8]` and
the route slot is `expert_idx + 1`. Each rank runs exactly two tiles, W13 then
W2, so the 240-tile padding space and its straggler imbalance are gone, and an
expert's producers and consumers share an XCD pair — making the W13→W2 barrier
pair-local instead of a cross-die fan-out.

---

## What did NOT work

Ported, measured, kept behind opt-in flags with numbers at each definition.
Do not re-run these without changing a precondition first.

| lever | delta (ms) | why |
|---|---|---|
| `MPK_ROUTER_DUAL_REDUCE` | 1.870 → 1.869 | neutral; kept for being bit-identical and shorter |
| `MPK_MOE_DUAL_ACCUMULATOR` | 1.840 → 1.852 | no inter-MFMA `s_nop` to reclaim — 47 of 49 gaps already zero |
| `MPK_LM_HEAD_KMAJOR` | 1.837 → 1.844 | |
| `MPK_LM_HEAD_WAVE_TILE_DMA` | 1.840 → 1.844 | |
| `MPK_EMBED_WIDE` | 1.843 → 1.846 | |
| `MPK_EMBED_PRODUCERS` | 1.843 → 1.852 | |
| `MPK_QKV_MFMA_UNROLLED` | 1.881 → 1.883 | |
| `MPK_ROTATE_QKV_ATTN_RANKS` | 1.877 → 1.892 | |
| `MPK_EARLY_ROUTING` | **incorrect** | ascending map slot 0 ≠ first pick; 3 runs, 3 hashes |

`MPK_MOE_DUAL_ACCUMULATOR` carries the generalizable lesson: a transform that
targets a stall the disassembly shows is already absent cannot pay. Count the
gaps before reclaiming them.

---

## Measurement discipline

Non-negotiable, because two of these levers looked like wins under weaker
protocol:

1. **Alternating control/variant pairs**, at least 3, on an idle GPU. Not
   control-batch-then-variant-batch — baselines drift between batches.
2. **Check the generated text every run.** Hash it. A latency number without a
   text check is not a result.
3. **Record the number at the flag definition**, including neutral and negative
   ones, so the next person does not re-run it.
4. **One variable per checkpoint.**
5. **Confirm the assembly changed the way you predicted** before believing the
   wall. Depth at the first wait, gap histogram, spill count.

On the GLM-5 box specifically, also check for orphaned ranks holding
`/dev/kfd` before scoring — a leaked allocation both slows survivors and wedges
new launches, and it drifted a baseline 0.035 ms. See
`demo/glm5/CLOSING_LEDGER.md`.
