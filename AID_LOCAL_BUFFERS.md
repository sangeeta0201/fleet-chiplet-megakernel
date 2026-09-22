# "Duplicated per AID" is not "placed per AID"

The most productive defect class found in SPX+NPS2 so far, and the easiest to
miss: a buffer that the *kernel* indexes as two per-AID copies, but whose
copies both live in **one** memory range.

Two fixes shipped from it:

| change | result |
|---|---|
| `MPK_MOE_NARROW_RELEASE` (`ee1b248`) | **-5.60%, t = -5.39** |
| `MPK_SWIGLU_AIDREP` (`540e24d`) | **-11.08%, t = -6.90** |

Combined **2.416 -> 2.030 ms (-16.0%)**, hash `3d54adb71e19` on every run.
Against the same-build SPX+NPS1 reference (**1.880 ms**, measured on thor-3
with the stock driver) the gap closes from **1.213x to 1.080x**.

---

## The defect

`MPK_SWIGLU_AID` duplicated `swiglu_out` along dim 0 and had each W2 tile read
the copy chosen by `xcd_id`:

```c
unsigned short const *w2_input_base =
    d_swiglu_out + ((xcd_id >= (MPK_NUM_XCDS / 2)) ? swiglu_aid_half : 0) + ...;
```

The addressing is correct -- the disassembly even shows the selector as
`s_cselect_b32 s4, 0x5c00, 0` (0x5c00 = 23552 = `swiglu_aid_half`). But the
whole doubled tensor is **47 KB**, and the placement map said where it lived:

```
[MAP] slot=22 name=swiglu_out kind=shared alloc_mib=2.00 off_mib=1.73
      lo_ns=102.8 hi_ns=101.3 skew_ns=+1.4 aid=1 placeable=0 local=4/8
```

`alloc_mib=2.00 off_mib=1.73` -- a 47 KB **suballocation inside a 2 MiB torch
pool**. Halves placement splits the *BO* (the pool), not the tensor, so both
copies sat in the same range. Four XCDs gathered the row across the AID
boundary on every read, and because the loads are `nt` they bypass L2, so every
read paid it.

### The signature to look for

| symptom | meaning |
|---|---|
| `skew_ns` ~ 0 **and** `local=4/8` | buffer is wholly in ONE range; half the XCDs are remote |
| `aid=` flips between runs | placement lottery -- no deterministic home |
| `alloc_mib` >> tensor size, `off_mib` > 0 | it is a suballocation, not its own BO |

**A near-zero skew does NOT mean "nothing to gain."** Skew is the *difference*
between the AID halves' latency; a buffer in one range read by all eight XCDs
has both halves equally penalised, so skew is ~0 while everyone is slow.

---

## The fix: reuse the `alloc_in_aid` replicas

`MPK_AID_SPLIT_FLAGS` already allocates exactly what is needed:

```c
static constexpr size_t kFlagRepBytes = 2ull << 20;   // 2 MiB per AID
rep[aid] = alloc_in_aid(kFlagRepBytes, aid, extra);   // real BO, pinned
hipMemset(rep[aid], 0, kFlagRepBytes);
```

Two genuine BOs, one pinned in each range (`GEM_CREATE_AID_LOCAL`, 2 MiB
aligned), zeroed, published to `__device__ int *g_aid_flag_rep[2]`. Any small
buffer can live inside them at a free int offset -- **no new allocation code.**

Offsets in use (2 MiB = 524,288 ints per replica):

| region | base int |
|---|---|
| flag regions | 0 .. 1023 |
| `MPK_AID_MOE_BASE_INTS` | 1024 |
| `MPK_AID_LMTAIL_BASE_INTS` | 40960 |
| **`MPK_AID_SWIGLU_BASE_INTS`** | **200704** (5888 ints) |
| `MPK_AID_EVCTR_BASE_INTS` | 458752 |

### Keep the diff tiny

Point the base pointer at AID0's replica and redefine the "half" offset as the
**pointer difference** to AID1's. Every existing store and read site then works
unchanged:

```c
long long swiglu_aid_half =
    (long long)MPK_MAX_NUM_BATCHED_TOKENS * NUM_TOPK * INTERMEDIATE_SIZE;
if (g_aid_flag_rep[0] != nullptr && g_aid_flag_rep[1] != nullptr) {
  unsigned short *sw0 = (unsigned short *)(g_aid_flag_rep[0] + MPK_AID_SWIGLU_BASE_INTS);
  unsigned short *sw1 = (unsigned short *)(g_aid_flag_rep[1] + MPK_AID_SWIGLU_BASE_INTS);
  d_swiglu_out = sw0;
  swiglu_aid_half = (long long)(sw1 - sw0);
}
```

Three details that matter:
- the offset **must** be `long long` -- two independent BOs can be far apart,
  and the old `int` would overflow
- keep the fallback to the original pointer, as `mpk_aid_flags_at` does, so a
  failed placement degrades instead of corrupting
- **cross-AID publishes must be write-through stores**, never atomics. The
  replicas are `MTYPE_RW`, coherent only inside their own AID; `st_wt_u16/u32`
  crosses, an atomic does not. That is the hazard that hung `MPK_AID_EVCTR` v2.

### Verify, do not assume

The allocator prints `AID0 replica 0x.., AID1 replica 0x..` every run. **Gate
on two distinct pointers.** This is the whole reason the replica version is
trustworthy and the padding version was not.

---

## REJECTED: padding to reach the placement granularity

`MPK_SWIGLU_PLACED` padded each copy to >= 2 MiB hoping halves placement would
split the BO at its midpoint: **-0.56%, t = -1.17**, hash green.

It is not merely weaker, it is **unfalsifiable** -- there is no cheap way to
confirm the copies actually moved, so a null result cannot be distinguished
from "the patch did nothing". Do not reach for padding when `alloc_in_aid`
exists.

---

## How the cause was found: MoE-isolated ATT

`MPK_ONLY_OP` is a **bitmask** over ops 1/3/5/7/8, so **384 = op 7 + op 8**
runs clean. The recorded "`MPK_ONLY_OP=8` HANGS" was only ever op 8 starving
MoE's routing dependency; keeping op 7 fixes it. 128 (op 7 alone) gives a
subtraction baseline.

| arm | NPS1 (thor-3) | NPS2 (thor-2) | ratio |
|---|---|---|---|
| base | 1.880 | 2.281 | 1.213x |
| op7+op8 (384) | 1.557 | 2.087 | 1.340x |
| op7 (128) | 0.849 | 1.146 | 1.350x |

Then filter the ATT stats CSV by **symbol range** from the code object (MoE =
`[0x99a8, 0xce48)`), which is the only way to isolate one kernel:

| region | NPS1 | NPS2 | ratio | delta |
|---|---|---|---|---|
| `worker_kernel` | 43.6 | 172.6 | **3.96x** | **+129.0** |
| `gang_full_layer_fused` A | 92.6 | 171.6 | 1.85x | +79.1 |
| `gang_moe_fused` | 230.5 | 289.1 | 1.25x | +58.6 |
| `gang_linear` (O-proj) | 141.8 | 159.5 | 1.12x | +17.7 |
| LM head | 84.5 | 34.9 | **0.41x** | **-49.5** |

**Do NOT difference mask384 - mask128.** Gating ops changes barrier behaviour;
the `s_barrier` delta came out negative and NPS1's "MoE" difference exceeded
NPS2's, contradicting the timing. Compare mask384 between modes instead.

### Reading ATT correctly

ATT charges a load only for its **issue** stall; the wait for data lands on the
`s_waitcnt`. So `load->LDS` at 0.87x (faster) together with `s_waitcnt` at
1.63x does **not** mean "memory is fine" -- it means the loads issue fine and
complete slower. And `s_barrier` is an on-CU intra-workgroup sync with no
address: its cost is purely how far apart a block's waves arrive, so it is a
**symptom** of divergence, never a cause.

One remote operand poisons a whole drain: a `vmcnt(0)` waits for the **maximum**
of its outstanding loads, so P(all near) = 0.5^N. **Partial locality is nearly
worthless** -- which is why a 47 KB buffer in the wrong range cost 11%.

---

## Still on the list

Buffers that are read many-to-one and are **not** yet in per-AID replicas:

| buffer | size | note |
|---|---|---|
| `workspace_f32` | 47 KB | `MPK_WSF32_AID` exists but broke the output: the task-212 `MOE_RESIDUAL_ADD_F32` consumer was never offset |
| `routing_weight` | **16 B** | read by ~184 MoE tiles -- the worst contention shape in the model; the mirror must go INSIDE `topk_softmax_mi300_task_impl` before its release store |
| `w2_bias` / `w13_bias` | 0.7 / 1.4 MiB | routing-dependent; do NOT replicate along the expert axis (5x regression, memory budget) |
| `sinks`, `router_bias` | tiny | read by every attn / router workgroup |

And the largest open item is not MoE at all: **`worker_kernel` is 3.96x** and
+129.0M of the +246.2M gap, versus MoE's +58.6M. Nothing has looked at it.
