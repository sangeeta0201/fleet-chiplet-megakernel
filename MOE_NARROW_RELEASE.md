# MoE in SPX+NPS2: the release fan-out was the tail

Session result: **2.416 -> 2.281 ms, -5.60%, paired t = -5.39, 3/3 wins**, hash
`3d54adb71e19` (bit-identical to SPX+NPS1) on all six runs. Slot 8 (MoE tiles)
**16.66 -> 14.57 us/layer, -12.5%**. Committed as `ee1b248` on
`lane/moe_aidexp`.

Everything below is measured on thor-2, SPX+NPS2 halves (`aid_local_ilv_mode=1`),
device 14 (BDF f5, 256 CU), on top of the shipped recipe.

---

## THE WIN: `MPK_MOE_NARROW_RELEASE` -- one release slot per AID, not eight

### The defect

`MPK_MOE_INNER_TIMING=1 MPK_MOE_INNER_WIDE=1`, 15761 across-tile samples:

| class | dec | compute | **arrive** | total |
|---|---|---|---|---|
| fastest 50% | 0.28 | 6.84 | **0.36** | 7.60 |
| p90-p98 | 0.32 | 8.48 | **0.36** | 9.32 |
| **slowest 2%** | 0.28 | 8.18 | **6.88** | **15.24** |

The slow tiles are **not** slow in compute (+1.3 us). They are slow in `arrive`
-- 0.36 -> 6.88 us, a 19x jump. `arrive` is the fence + arrival atomic + (on
the last arrival only) the release fan-out. Only one tile per expert runs the
fan-out: **4 of 184 = 2.2%**, which matches the slowest 2% exactly.

The fan-out was:

```c
for (int x = 0; x < 8; x++) {
  mpk_aid_publish_at(d_barrier, base + x * MOE_BAR_LINE,
                     (unsigned)release_val, MPK_AID_MOE_BASE_INTS);
}
asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
```

8 slots x 2 AID replicas = **16 write-through stores**, on one thread, drained
by a single `vmcnt(0)`, with every W2 tile of that expert blocked behind it.
~430 ns per store -- they were not pipelining.

### The fix

All eight slots carried the **same** value (`layer_idx + 1`) and differed only
in which XCD polled them. The poll already goes through the AID-local replica
(`mpk_aid_flags_at`), and `~/nps1/relobs3.sh` measured AID-local RW replicas
**flat at ~2.5 us out to 256 pollers**. So one slot per AID is enough and all
four of that AID's XCDs can poll the same line. **16 stores -> 2.**

Publisher and poller must change together (same macro) or readers poll a slot
nobody writes. The arrival counter stays at `MOE_BAR_COUNTER_SLOT` (line 8),
preserving the documented "counter must not share a cache line with the release
slots" hazard that caused a past deadlock.

### Mechanism confirmed, not just the number

| | control | narrow release |
|---|---|---|
| `arrive`, slowest 2% | **6.88** | **0.40** |
| W2 barrier p50 | 4.16 | **2.20** (-47%) |
| W2 total p50 | 13.56 | **10.44** (-23%) |

### Per-slot attribution -- the win is NOT confined to MoE

| slot | control | narrow rel | diff | pct |
|---|---|---|---|---|
| 0 inter-layer | 9.98 | 9.18 | -0.80 | -8.0% |
| 5 attn_rel end | 10.71 | 9.58 | -1.14 | -10.6% |
| 6 O-proj + router | 19.18 | 16.78 | -2.40 | -12.5% |
| 7 TopK wait | 9.80 | 9.19 | -0.61 | -6.2% |
| **8 MoE tiles** | **16.66** | **14.57** | **-2.09** | **-12.5%** |
| 10 P9 gate | 2.60 | 1.94 | -0.66 | -25.3% |
| **TOTAL** | **76.35** | **68.64** | **-7.70** | **-10.1%** |

Every pure-compute slot (1, 2, 3, 4, 11) is unchanged within noise; every
*wait* slot improved. Removing 6.9 us from the MoE release cut arrival spread
through the rest of the layer -- the same coupling `MPK_MOE_REPLICA` showed.

**So the same defect is worth checking on every other gate.** Slot 6 fell 12.5%
purely as a side effect of fixing MoE's publish.

---

## INSTRUMENT TRAP that produced a wrong diagnosis

`MPK_MOE_INNER_TIMING` **without** `MPK_MOE_INNER_WIDE` samples exactly ONE
fixed tile:

```c
#ifdef MPK_MOE_INNER_WIDE
    if (tid == 0 && (global_tile % 37) == 0) {
#else
    if (tid == 0 && global_tile == 0) {          // w13
    if (tid == 0 && global_tile == TOTAL_W13) {  // w2
#endif
```

So a narrow run's 3132 samples are the **same tile across layers** (86 iters x
36 layers ~ 3096), not a spread across the 46 tiles. Two conclusions drawn from
narrow data had to be withdrawn:

- "W13's compute has a tail" -- FALSE. The across-tile tail is in `arrive`.
- "The barrier costs 9.92 us" -- FALSE. Across tiles it is **4.16 us** p50. The
  9.92 came from the first W2 tile, which the code's own comment calls "the
  earliest worker, so an upper bound on skew rather than a population figure".

**Always set `MPK_MOE_INNER_WIDE=1`.** A narrow number is one tile's history.

---

## REJECTED: expert parallelism by slot (`MPK_MOE_AID_EXPERT`)

AID0's XCDs own routing slots 0,1 and AID1's own 2,3. Hash-green 6/6, and
**+18.3%, paired t = 8.05, 0/3 wins**.

Bisected with `MPK_MOE_AID_EXPERT_W13` (AID map on W13 only) and
`MPK_MOE_AID_NSPLIT` (per slot, tiles 0-22 -> AID0, 23-45 -> AID1):

| arm | swiglu fan-in | wg stride | W2 compute p50 |
|---|---|---|---|
| stock | 23 | 8 | 3.48 |
| W13-only | 23 | 8 | 3.48 |
| N-split | 23 | 4 | 4.00 |
| expert-confined | 46 | 4 | 5.28 |

The +1.80 us is **+0.52 from the stride** (wg_idx walk 8 -> 4) and **+1.28 from
the swiglu reader fan-in** (23 -> 46). Confining W2 to one AID means all four of
its XCDs share one `xcd_id >= 4` value, so all 46 tiles read the SAME
`swiglu_out` copy and the other copy goes unread. That is the shape
`MPK_W2_SWIGLU_PIN0` already measured: fan-in 46 -> 184 cost **5.12 -> 35.44 us**
in NPS2 versus 2.20 -> 2.36 in NPS1.

**Corollary: any change that confines W2 to one AID must also raise the
`swiglu_out` copy count, or it pays the fan-in.**

### The recorded reason for `MPK_MOE_XCD_PAIR`'s +31% is WRONG

The note said the pin only balances when "four routed experts land on four
distinct pairs, 9% of the time". But `expert_idx` there is the routing **slot**
(0..3), always exactly four and always distinct -- there is no lottery. PAIR's
real handicap is `kGroupsPerPairMember = 23`: it uses only ranks 0..22, leaving
**8 of 31 workers per XCD idle**.

---

## NEUTRAL: no tile->XCD mapping beats stock

| arm | result |
|---|---|
| `MPK_MOE_AID_EXPERT_W13` | -0.14%, t = -0.09 |
| `MPK_MOE_AID_NSPLIT` | -0.35%, t = -0.24 |

Per-XCD tile counts are **identical** in every mapping (23 W13 + 23 W2), so a
remap changes only *which expert* a tile belongs to, never how much work a die
does. Measured: W13 total 8.00 vs 7.96 us. Do not expect a mapping change to
reduce work.

---

## CEILING: deleting the barrier is worth -9.2%

`MPK_MOE_SKIP_BAR` removes the W13->W2 **wait** and keeps the release (the
`MPK_WAIT_SKIP` discipline, so it cannot deadlock). Output garbage by design.

| | ms | slot 8 | total us/layer |
|---|---|---|---|
| control | 2.660 | 16.66 | 76.35 |
| wait deleted | **2.414** | **12.62** | 68.92 |

`MPK_MOE_NARROW_RELEASE` captured **61% of the end-to-end ceiling** and **52%
of the slot-8 ceiling** without breaking the hash.

---

## AITER's split-K MoE is SLOWER at bs=1. Do not port it.

Benchmarked on this box, MXFP4 a4w4, E=128, topk=4, D=I=3072 (fleet's 2944
**NaNs** in AITER -- `per_1x32` needs model_dim a multiple of 256):

| arm | M | launch grid | graph us |
|---|---|---|---|
| 1-stage ASM FLAT subGU=128 | 1 | 24x4x1 = **96 WG** | **23.34** |
| 2-stage FlyDSL a4w4 | 1 | auto | 32.11 |
| 1-stage | 8 | 768 WG | 147.18 |
| 2-stage | 8 | auto | **132.15** |
| 1-stage | 128 | 12288 WG | 1885.67 |
| 2-stage | 128 | auto | **548.68** |
| **fleet** | 1 | 184 tiles | **16.7 - 20.0** |

`flat_mode==1` grids as `ceil(inter_dim/sub_GU) x topk x token_cnt`, so at bs=1
it fills 96 of 256 CUs. Occupancy sweep at M=1: 48 WG -> 1611 GB/s, 96 -> 2581,
192 -> 3239, 256 -> 3332. Even full it tops out near 3.3 TB/s. And one TG per
(token, slot) re-streams the whole expert with zero reuse, so time is linear in
M -- 3.4x worse than 2-stage at M=128.

There **is** an MXFP4 1-stage `fmoe` on gfx950
(`hsa/gfx950/fmoe/silu/fmoe_bf16_pertokenMXfp4_g1u1_silu.csv`, 10 kernels, two
`flat=1`). What is absent is MXFP4 in `fmoe_2stages/`.

---

## BLOCKED: the W2 MFMA loop cannot be staged

Staging the barrier (W2 consumes K in G stages, waiting only for the W13 tiles
feeding stage g) is bit-identical in principle -- K order is preserved -- but
the hand-tuned asm carries:

```c
static_assert(W2_MFMA_ITERS % 2 == 1, "MPK_MFMA_PINGPONG_SCHED ...");
static_assert(W2_MFMA_ITERS % 4 == 3, "MPK_MOE_QUAD_ACCUMULATOR ...");
```

`W2_MFMA_ITERS = 2944/128 = 23`. Every stage would need a count that is odd
**and** = 3 mod 4. Two odd numbers cannot sum to 23 (odd), and three values
= 3 mod 4 sum to = 1 mod 4, not 3. **No G works.** Staging means falling back to
the unpipelined loop, documented at ~53 cyc/iter versus ~36 -- a ~30% MFMA loss
against a 4.04 us prize.

Second blocker: `MOE_BAR_SLOTS = 10`, with 0-7 per-XCD releases and 8 the
counter, leaves **one spare line per expert**. Extra stages need more slots and
a larger host allocation.

---

## WHERE THE NEXT HEADROOM IS: bandwidth, not the barrier

MoE moves 55.3 MB/layer:

| | us/layer | GB/s | vs peak |
|---|---|---|---|
| before | 16.66 | 3319 | 52% |
| **after** | **14.57** | **3795** | **60%** |
| both-range peak (`~/nps1/bwsplit.cpp`) | -- | **6346** | 100% |

AITER's independent measurement lands in the same band (2577 GB/s, topping out
~3.3 TB/s fully occupied). **Both implementations are bandwidth-bound at roughly
half the part's both-range capability**, so the remaining MoE headroom is
bandwidth efficiency, not synchronization.

---

## Reproduce

```bash
# lane runner, one GPU, no shared-tree edits
EXTRA_ENVS="MPK_MOE_NARROW_RELEASE=1" TIMEOUT=600 \
  bash ~/nps1/fleet_run_lane.sh moe_aidexp <label>

# hash gate
grep -a assistantanalysis <log> | tail -1 | tr -d '\r\n' | md5sum   # 3d54adb71e19

# across-tile timing (NEVER omit WIDE)
EXTRA_ENVS="MPK_MOE_INNER_TIMING=1 MPK_MOE_INNER_WIDE=1 ..." ...
python3 ~/nps1/w13_tail.py <log>       # histogram + per-field class table
python3 ~/nps1/barrier_after.py        # p50/p98 per field, two arms
python3 ~/nps1/slot_compare.py         # per-slot attribution
```

Logs: `~/nps1/fleet_baselines/lane_moe_aidexp/`.
Flags added this session, all default OFF: `MPK_MOE_NARROW_RELEASE` (ship),
`MPK_MOE_AID_EXPERT`, `MPK_MOE_AID_EXPERT_W13`, `MPK_MOE_AID_NSPLIT`,
`MPK_MOE_SKIP_BAR` (timing probe).
