# GLM-5 744B decode on 8x MI355X — CLOSING LEDGER

**Branch** `merge-rocshmem`. **Date** 2026-08-22. **Target** 2 ms/token of
single-stream decode latency, correct output.

This file is the terminal artifact of the optimization run. It states where the
ladder stopped, what the floor actually is, every item that was closed and with
what number, the two blockers that are *legality* results rather than tuning
gaps, and the traps that would otherwise mislead whoever picks this up next.

Nothing in here is an estimate that has not been ablated, except where it says
so in the line.

---

## 0. THE ONE-PARAGRAPH ANSWER

The megakernel decodes GLM-5 744B at **10.5 ms/iter** at bs=1, and at **8.884
ms/token** with MTP speculation confirmed under the correctness gate. The
width-corrected achievable floor for this parallelization is **2.144 ms**. The
~8.4 ms between them is **not** bandwidth, not tile geometry, not occupancy, and
not the barrier mechanism — all four were attacked to exhaustion and every one
is measured out. It is the **bulk-synchronous ladder itself**: 10 GPU-wide
rendezvous plus 2 cross-rank rendezvous per layer x 76 layers, at 3.77 us each,
whose cost is arrival SKEW and not the barrier code. The two ways to shorten a
ladder — fuse independent rounds, or overlap across the layer boundary — are
both closed by dependency legality, not by effort. **2 ms/token is not reachable
on this hardware with this parallelization.** Section 8 says what it would take.

---

## 1. THE LADDER AS IT STANDS

| stage | ms/token | how measured | gate |
|---|---|---|---|
| starting point (recorded) | 14.824 | — | — |
| bs=1 control, published | **10.609** | RE-STEER v5 board figure | — |
| bs=1 control, *measured* n=3 median | **10.619** | `probe_row_cost_wall.sh:28` — 10.562 / 10.619 / 10.685 | G1+G2 |
| bs=1 base, n=6 pooled, 2026-08-22 | **10.498** | `probe_qb_peer_wait_ceiling.sh` Phase A | G1+G2 |
| **MTP speculative decode** | **8.884** | `probe_mtp_regate.sh`, median of 3 reps | **G1+G2+G3** |

**On the three control numbers.** They are not a contradiction and none of them
should be silently preferred. `10.609` is the board's published figure and is
what every ratio on the board is divided by. `10.619` is the n=3 median actually
measured for that build. `10.498` is a *different, later* build's n=6 pooled
base and is the correct control for anything measured against it on 2026-08-22.
The spread across all three is 0.121 ms, well inside the **0.26 ms wall noise
floor**. Rule that follows: *quote the control that shares a batch with the arm,
never a control from a different day.*

**The MTP figure and its gate.** Three reps, `probe_mtp_regate.sh`:

```
rep1  accept 0.871  1.875 tok/iter  16.515 ms/iter  8.813 ms/token
rep2  accept 0.861  1.861 tok/iter  16.535 ms/iter  8.884 ms/token
rep3  accept 0.857  1.861 tok/iter  16.542 ms/iter  8.889 ms/token
median: accept 0.861, 16.530 ms/iter, 8.884 ms/token vs 10.619 = 1.196x
```

Confirmed in `466b940` under `correctness_gate.py`:

* **G1 cross-rank identity (hard)** — all 8 ranks byte-identical within a run.
  This is the gate that catches real EP/sync/barrier defects. PASS.
* **G2 coherence (hard)** — full token count, distinct-token ratio >= 0.30, no
  bigram repeated > 25x. Measured healthy ~0.53 / 5-10. PASS.
* **G3 attractor membership (advisory)** — every MTP run shares 41-81 tokens of
  exact greedy prefix with the bs=1 controls in cluster B and **exactly 0** with
  cluster A. The load-bearing detail: **mtp3 agrees with control r5 (81 tokens)
  more than with mtp2 (67)**. MTP runs are closer to bs=1 controls than to each
  other, i.e. drawn from the same nondeterministic distribution — not a distinct
  stream. A genuine accept/reject defect gives the opposite signature
  (self-consistent MTP arm, offset from control). PASS as cluster MEMBER.

**Exact-token equality is an ILLEGAL gate on this model** (`369032c`). Five bs=1
runs of one build, one prompt, greedy argmax partition into **two attractor
continuations**: within a cluster they agree 122-124 tokens, across clusters
they differ at token 0. The mechanism is order-nondeterministic float reduction
(the EP fold and the atomic accumulations retire in arrival order) flipping
argmax on near-ties. Any "arm B must equal arm A token for token" gate fails
~50% of the time on two runs of the *same* configuration. Three published
conclusions were drawn against that illegal gate and had to be retracted,
including `ccf8cfa`'s retraction of the MTP number, which is itself retracted.

**MTP acceptance:** 0.861 inside the megakernel (`probe_mtp_regate.sh`), 0.750
end-to-end = 1.75 tok/iter. The cap is the harness, not the model.

---

## 2. THE FLOOR: 2.144 ms, WIDTH-CORRECTED

`demo/glm5/width_corrected_roofline.py` (`1052f86`). No GPU run — both inputs
were already measured.

The old floor (`roofline.py`, `abb5314`) divides every byte by a flat
`HBM_TBS = 5.17 TB/s` and gets **2.319 ms** (1.936 HBM + 0.383 EP). That rate
only exists when the whole GPU fetches at once, and **GLM never does** — the
layer is a ladder of phases at 16, 32, 128, 168, 192 and 232 workers. The
measured achievable aggregate at 16 wide is **707 GB/s, not 5170**
(`tests/standalone/test_narrow_grid_bandwidth.hip`, depth 4, the optimum at
every width):

```
grid width   16     32     64     96    128    232    256
GB/s aggr.  707   1399   2660   3800   4765   5373   5365
```

So the 2.319 ms floor ought to be a fantasy. It is not. Re-pricing every phase
at the measured rate *for its own real width*:

| phase | MB/layer/rank | tiles/XCD | workers | GB/s at that width | us flat | us wide | penalty |
|---|---|---|---|---|---|---|---|
| qkv_a (q_a + kv_a) | 16.63 | 21 | 168 | 4999 | 3.22 | 3.33 | +0.11 |
| q_b | 4.33 | 16 | 128 | 4765 | 0.84 | 0.91 | +0.07 |
| W_UK | 0.81 | 4 | 32 | 1399 | 0.16 | 0.58 | +0.42 |
| W_UV | 1.08 | 4 | 32 | 1399 | 0.21 | 0.77 | +0.56 |
| MLA decode (KV) | 0.07 | 2 | 16 | 707 | 0.01 | 0.10 | +0.09 |
| merge | 0.00 | 16 | 128 | 4765 | 0.00 | 0.00 | 0.00 |
| o_proj | 12.98 | 24 | 192 | 5139 | 2.51 | 2.52 | +0.02 |
| router | 1.62 | 16 | 128 | 4765 | 0.31 | 0.34 | +0.03 |
| MoE W13 | 53.48 | 54 | 232 | 5373 | 10.34 | 9.95 | -0.39 |
| MoE W2 | 26.74 | 108 | 232 | 5373 | 5.17 | 4.98 | -0.20 |
| **LAYER** | **117.73** | | | | **22.77** | **23.49** | **+0.71 us** |

```
75 MoE layers
HBM floor, FLAT 5.17 TB/s ................ 1.708 ms
HBM floor, WIDTH-CORRECTED ............... 1.761 ms
the width penalty ........................ 0.054 ms
+ EP collective (MEASURED 5.1 us x 75) ... 0.383 ms
= WIDTH-CORRECTED ACHIEVABLE FLOOR ....... 2.144 ms
```

**The width penalty on the whole layer is 0.054 ms.** And the entire bandwidth
gain available from magically running *every* phase at the 232-wide rate is
**0.118 ms** (1.761 -> 1.643) — under the noise floor.

**Why:** narrowness and byte weight are **anti-correlated** in this layer. MoE
W13+W2 is 68% of the bytes and already runs at the full 232. The genuinely
narrow phases are byte-trivial — MLA decode is 16 workers and **74 KB**, W_UK is
32 workers and **811 KB**, against a 118 MB layer. The phases that would gain
bandwidth from widening have no bytes to gain it on.

**Consequence: "widen a phase to get more bandwidth" is dead as a CLASS**,
priced at 0.118 ms before anyone writes one instance of it. Any remaining win
from widening a phase must be a *makespan/latency* argument, measured on the
wall.

**Consequence: 2 ms/token is "hit the roofline exactly."** The honest floor is
2.14 ms (width-corrected, MoE layers only) to 2.32 ms (`roofline.py`, which also
carries the 3 dense layers and lm_head). Against 10.5 measured, efficiency is
20.3% and **the whole gap is schedule.** The TileRT reference point for the same
model/hardware is 2.020 ms.

**Achieved vs the corrected roof, the two byte-heavy phases** (times measured,
`SP3[4]` MoeW13 13.06 us/layer, `SP3[6]` MoeW2 13.05, both population 232):

| phase | MB | us | GB/s | % roof | KB/tile | rounds | headroom |
|---|---|---|---|---|---|---|---|
| W13 | 53.48 | 13.06 | 4095 | **76.2%** | 123.8 | 1.86 | 0.233 ms |
| W2 | 26.74 | 13.05 | 2049 | **38.1%** | 30.9 | 3.72 | 0.605 ms |

W2's 0.605 ms is a **real bandwidth gap** with a named mechanism (864 tiles of
30.9 KB, each re-staging the whole activation into LDS and running a full 64-row
atomicAdd epilogue — a fixed cost that neither splitting nor widening divides).
**It is still not a lever**: OPW=16 -1.34 ms, `W2_KSPLIT=2` -4.12 ms, OPW=128
neutral. The MoE half is barrier-bound, so shrinking the phase is absorbed.
**A bandwidth gap and a lever are different things.** This is the worked example.

---

## 3. THE STRUCTURE OF THE 10.5 ms

Three independent decompositions that agree.

### 3a. FLOOR vs HOLE (`0a4f583`, `9238f91`, `a644acb`)

At a rendezvous everyone leaves together, so **shortest wait <=> last arriver**.
With `wait_w = b_w - a_w`: `FLOOR = min_w wait_w`, `SKEW = max_w - min_w`.

```
FLOOR 6.498 ms  +  HOLE 5.776 ms  =  12.274  vs  12.017 ms measured span
```

The model closes with no residual. But:

* **54% of the layer is last-arriver FLOOR** — and **1.506 ms of that 6.498 is
  CROSS-RANK peer wait** that the intra-rank estimator structurally cannot see.
  The EP collective splits **4.47 us local fold / 19.81 us peer wait** per layer.
  **The honest local floor is ~4.99 ms, not 6.498.**
* **Only 2.277 ms of the 5.776 ms hole is provable spin** (the four RELEASE
  rows). Do not add 5.776 and call it available.
* **The fillable-hole ranking is the INVERSE of the board.** The EP collective
  is the single biggest board line (1.899 ms) and the *least* fillable region
  (0.015 ms of slack).

### 3b. THE FOUR RELEASE HOLES — `min(window, movable) = 0` FOR ALL FOUR (`7656a14`)

| hole | ms | what is already in the window |
|---|---|---|
| S25->S26 attention release | 0.896 | 42-load o_proj weight DMA + 17 loads of shared-expert weights, **issued at S24->S25 exactly so they fly through this spin** (the gpt-oss Phase 6 idiom, `gang_mla_full_layer_fused_mi300.cuh:1809-2037`). Widening refuted in-source: at GLM-5 shapes `PF_WG_BYTES` = 528 KB/worker = 15.3 MB/XCD against a 4 MB L2 — 3.8x L2, most evicted before Phase 9 reads it. |
| S32->S5 routing poll | 0.684 | **NOTHING. Bare spin, idle bus.** And still unfillable — see below. |
| S6->S7 W13 -> W2 | 0.492 | W2 L2 prefetch, **measured neutral, 0.13%** |
| S17->S18 qkv_a -> q_b | 0.203 | under the 0.26 ms noise floor |

> **YOU CANNOT PREFETCH THROUGH A BARRIER WHOSE OUTPUT IS THE ADDRESS.**
> S32->S5 is the one release hole in the layer with an idle memory system —
> `gang_oproj_router_fused_mi300.cuh:1174-1195` is a bare `ld_nt_s32` spin with
> no DMA issued before it. In bandwidth terms the window is enormous (9 us at
> 5.17 TB/s = 46 MB, against a 65 MB/layer whole-layer budget). The movable set
> is still **empty**, because the thing the poll is waiting for is **expert
> identity**, and expert identity is what **selects the bytes**. The window and
> the dependency are the same object. Every unconditionally-known operand is
> already resident. The only escape is a *speculative* prefetch of predicted
> experts — a different project, and it needs a prediction first.

### 3c. THE RENDEZVOUS COUNT IS THE COST (`afcb18b`, `534e3f4`)

One GPU-wide rendezvous costs **3.77 us/layer** = 27% of the wall. It **is** the
round's fixed cost — dispatch is 0.14 us. Each rendezvous moves ~13 MB/layer/rank.
The layer pays **10 GPU-wide + 2 cross-rank** rendezvous. The layer model closes:
`tiles 82 + 10 x 2.85 barrier + 17.5 EP = 128 of 135 us`.

**The cost of a rendezvous is arrival SKEW, not the barrier code.** That is why
every mechanism attack failed (Section 5c).

---

## 4. THE TWO STRUCTURAL BLOCKERS — THESE ARE LEGALITY RESULTS, NOT TUNING GAPS

There are exactly two ways to shorten a bulk-synchronous ladder: run fewer
rounds, or overlap rounds. Both are closed by **dependency legality**. Neither
is a "we didn't try hard enough" — trying harder produces wrong output.

### Blocker 1 — NO LEGAL INDEPENDENT-ROUND FUSION EXISTS

Every adjacent pair of rounds in the layer has a true data dependency across the
barrier that separates them. The MPK task graph is a **strictly linear chain**
(`chain_after` is the only escape hatch and it does not create independence).
There is no pair of rounds that can be merged without a read-before-write.

**Therefore the only way to cut barrier count is to DELETE A PHASE**, not to
fuse two. And phase deletion was priced: `f02753c` — the MLA decode ablation
opens a 0.749 ms window and **the legal candidate set of movable work is empty.**

Corroborating measurements, all of which would have been wins if fusion were
legal: deleting a whole GPU-wide rendezvous is **neutral** (`69d4dde`, 10.547 ->
10.544, n=5 paired); narrowing barriers deletes 0.51 ms of counted time for
0.10 ms of wall (`ef87f94`); adjacent-phase pipelining — hiding the **entire**
W2 tile phase under W13, a wrong-output ceiling probe — buys **0.146 ms**
(`37074ae`).

### Blocker 2 — `qkv_a(L+1) <- MoE(L)`

Cross-layer overlap is the other classical answer, and it is blocked by a single
edge: layer L+1's `qkv_a` consumes layer L's MoE output. `959f14a` set out to
build heterogeneous worker groups — half the workers on layer L's tail, half
starting L+1 — and found **no second stream of work exists at batch 1**. Half
the heterogeneous-group hypothesis was refuted outright; the surviving half is
blocked by exactly this one edge.

The layer boundary itself is not the problem — it **ablates to zero** (`5a185a2`,
deleting *all* of it buys 0.051 ms) and it is not the pointer fetch (`37cdcbe`,
neutral). Yet it **pads at 1.22:1** (`3f52067`, 1000 ticks/layer of pad costs
0.953 ms). Both are true: uniform time at the boundary *is* critical path, and
there is no bookkeeping there left to delete. Deleting work you do not do is
worth nothing; the boundary is a *serialization point*, not a cost center.

**Batch width is not a third answer.** It buys tokens/sec, not latency, and it
costs **+2.14 ms of wall** to do it (`a015f60`, `c08765f`). Against a
single-stream latency target that is a regression, not a lever.

---

## 5. EVERY CLOSED ITEM, WITH ITS NUMBER AND ITS COMMIT

Negatives and neutrals are results and are listed as such. **Re-running any of
these is the main way the next run gets wasted.**

### 5a. WINS THAT LANDED (the ladder 14.824 -> ~10.5)

| item | delta | commit |
|---|---|---|
| MLA decode: 8-token KV chunks, not 32 | 14.189 -> 11.464 (-19.2%) | `e158ff4` |
| Two experts per router tile, one round instead of two | 11.721 -> 10.990 | `f098dc3` |
| Defer q_b's rope past the W_UK barrier, freeing OPW to 16 | 11.805 -> 11.500 | `e4e34aa` |
| Head-shard q_b + W_UK across the 8 EP ranks | 12.036 -> 11.805 | `22e7793` |
| Clamp the MoE dispatch to the tiles a rank owns | 11.020 -> 10.840 (**-0.180**) | `c8ef8d5` |
| Stage only the split's K window in W2 | 15.136 -> 11.493 (at K_SPLITS=2) | `fdca420` |
| o_proj weight to MXFP4 | 10.768 -> 10.628 (-0.14) | `7afb1ee` |
| Unroll qkv_a's residual-resolve prologue | 10.432 -> 10.360 | `3cf376f` |
| Batch qkv_a's resolve-loop loads ahead of the LDS store | 10.377 -> 10.261 | `9bf851b` |
| Shard replicated stages via the ep_signal line (o_proj) | -11.9% for one parameter | — |
| MTP draft layer in the task graph | 10.619 -> 8.884 ms/token (1.196x) | `f18dc60`, `466b940` |
| Decode ablation harvested by 8-token KV chunks | 3.40 ms window -> 0.66 remaining | `6bdb9a0` |
| Stage stamp on private per-worker rows (instrument fix) | 14.260 -> 11.881 with same probe | `a4e0853` |

### 5b. OCCUPANCY — THE LADDER IS DEAD

| item | result | commit |
|---|---|---|
| The occupancy ladder (2 blocks/CU) | **DEAD.** The apparent 1.54x on HBM-bound work is a **loads-in-flight artifact**; 4 loads in flight at 1 block/CU beats it (standalone 3446 -> 5446 GB/s by unrolling alone) | `54bc1b4` |
| The lock is **LDS, not registers** | 155 of 160 KB/CU per block pinned 1 block/CU all along; cutting to 72 KB gives 2 blocks — **10.244 -> 10.248, neutral as predicted** | `3902e7e` |
| It needs TWO gates, not one | measured co-residency cliff is 256 VGPR (2 blk/CU) vs 262 (1 blk/CU), but the grid is only 240 blocks on 256 CUs — a register cut alone is null | `2236341` |
| Register gate opened: 284 -> 252 unified VGPR | opened, and then null at the wall | `d90560c` |
| `wpe=3` | **faults at launch.** Not scratch, not the attribute, not the asm — only the 252-VGPR allocation is broken. wpe=2 runs clean at 10.911 | `af2cafa` |
| "registers are free / depth-8 scheduling" | **RETRACTED as a claim**, but the depth-8 null stands. Depth 4 is optimum at EVERY width; depth 8 is 15% WORSE at narrow grids | — |
| Occupancy is pinned by `mla_decode_absorbed`'s registers | — | — |

### 5c. THE BARRIER MECHANISM — ELEVEN ATTACKS, ALL MEASURED OUT

A rendezvous costs **SKEW**. Nothing that touches the barrier *code* can move it.

| item | result | commit |
|---|---|---|
| Barrier narrowing (`MPK_W13_EARLY_REL` family) | 0.51 ms of counted time deleted -> **0.10 ms of wall**; family retired | `ef87f94` |
| Deleting a whole GPU-wide rendezvous | 10.547 -> 10.544, **neutral**, n=5 paired | `69d4dde` |
| All-thread barrier polling | **+0.73 ms**, measured twice | — |
| Barrier arrival tree | 11.005 -> 11.004 **neutral** (-44% in the standalone probe) | `24852c8` |
| `MPK_NUM_WORKERS=200` | neutral | — |
| CROC per-XCD merge barrier port | closed by PAIR_MERGE's own number, 11.074 vs 11.005 (**+0.07**) | `e482f2e` |
| PAIR_MERGE + W_UK rows + KV-chunk retunes (4 knobs) | 10.930 -> 10.929, all inside noise | `a21d215` |
| Entry-barrier spread | is **static worker imbalance**, not jitter | `ebf9eb6` |
| Counted region time before a barrier | **not a lever, 3 for 3.** Check arrival spread first | — |
| Phase 8 "barrier" | is a **straggler**, not a mechanism (91% spin) | `353ad06` |
| decode->merge "wait" | is **serial decode**, not a rendezvous | — |
| q_b / W_UK "phase" | is **skew**, not a straggler; tile-width tuning there is dead | — |

### 5d. PREFETCH / LATENCY HIDING

| item | result | commit |
|---|---|---|
| `MPK_MOE_PF_GROUPS` (deep k-loop prefetch) | **28% standalone tile win, THREE paired A/Bs, THREE nulls.** W13 guarded 10.460 -> 10.491 (+0.031), guard-free 10.503 -> 10.517 (+0.014), W2 n=6 pooled 10.526 -> 10.521 (-0.005). *"Whatever the standalone is measuring, the megakernel's W13/W2 tiles are not paying it."* Retro-explained by the 76%-of-roof correction | `8123795`, `f3ef505`, `7e26c88` |
| W2 L2 prefetch | **0.13% — neutral** | — |
| Layer-boundary pointer prefetch | 10.748 -> 10.758, neutral | `37cdcbe` |
| Bias prefetch in the router TopK tail | cold fetch is only **205 ns** | — |
| Batching the MoE K-loop's weight loads (`MOE_KBATCH`) | 10.258 -> 10.256 — and then found to be a **kernel with no caller**; reverted | `7f0ebb1`, `2aa2bd6` |
| 16-byte loads in the shared quant prologue | 10.776 -> 10.838, neutral | `28bb883` |
| A qkv_a-sized cold read on the W13-idle workers | **+0.030 ms** — the MoE phase really does have a free worker-shaped hole (21/29 idle absorb it) | `c095dc0` |

### 5e. FOLDS AND HOISTS — ALL NEUTRAL OR NEGATIVE

| item | result | commit |
|---|---|---|
| The router fold (`GLM_ROUTER_FOLD`) | **96 ns.** Neutral, defaulted off | `4142bfa` |
| Hoisting qkv_a's EP fold out of the tile | 10.840 -> 10.930, neutral | `c72b559` |
| Widening the EP fold 8 -> 64 work-groups | 10.620 -> 10.640, neutral | `40800a1` |
| Hoisting qkv_a's WHOLE prologue | 10.794 -> 10.981, neutral | `7a5c559` |
| MXFP8 W13 -> W2 activation handoff | 10.724 -> 10.824, **+0.100 neutral** (VALU over an MFMA loop is free) | `38579f0` |
| Folding the MTP draft layer into the main replay run | **~0.09 ms of a 3.721 ms harness — NEGATIVE, do not build it** | `5dff9e9` |
| The bs=2 MFMA row fold onto wasted output columns | S16->S17 17.150 -> 16.535 but S17->S18 6.919 -> 7.882 = **net +0.348 us/layer** | `a7762b7` |
| The MoE row fold | **NO-GO.** V=2.801 of 8 prices it at 0.381-0.444 ms against a 0.5 ms bar | `f862285`, `53b00c3` |
| Fusing the dense prologue (`GLM_FUSE_ATTN=1`) | **+0.996 ms at bs=1, +1.684 at bs=2** | — |
| Hoisting parallel redundancy | a **wash** — 24 concurrent derivations = ONE tile of makespan | — |

### 5f. TILE GEOMETRY — CLOSED FROM BOTH SIDES

| item | result | commit |
|---|---|---|
| `GLM_MOE_W2_OPW=16` (narrower, 4x tiles) | **-1.34 ms** | — |
| `MPK_W2_KSPLIT=2` (2x tiles, half K) | **-4.12 ms** (11.020 -> 15.136) | `421782e` |
| `GLM_MOE_W2_OPW=128` (half the tiles) | 11.020 -> 10.972, **neutral** | `95a044a` |
| W13 narrowing | **no legal middle point**; experts already flattened | `99e8b7c` |
| Doubling W13 (additive probe) | **+1.152 ms** | `6149dfe` |
| An extra un-hidden qkv_a pass (additive probe) | **+0.669 ms** — so the RMSNorm-linearity split cannot pay | `9f43774` |
| W2 tile imbalance | 9.66 us unabsorbed spread with **no reachable tile geometry**: `tiles/XCD = (hidden/OPW)/8`, `hidden % OPW == 0`, so 9 tiles/XCD needs OPW=85.33 — unreachable | `ae9a34a` |
| In-place tile speedups | **transfer at 1.95:1**, they are NOT absorbed (qkv_a tile 13224 -> 12750 ns, wall -0.072) — but only for genuinely in-place work | `d6cc1d6` |
| Round quantization as an argument | pre-rejected in-source. *"An unabsorbed spread is necessary for a lever, not sufficient."* | `demo.py:975-1020` |

> **The asymmetry that governs all of the above:** *cutting* work inside a phase
> is absorbed (only deleting a whole serial phase moves the wall); *adding* work
> costs ~1:1. Additive probes therefore **overprice deletions**. Never price a
> deletion with an additive probe.

### 5g. THE EP COLLECTIVE AND THE SHARED EXPERT

| item | result | commit |
|---|---|---|
| **THE MODEL** | inter-rank skew is paid **ONCE**, at the FIRST cross-rank rendezvous. Deleting the EP collective moves **83%** of it to the q_b gather (10.490 -> 10.284, n=3, wall -0.206, layer span -0.198). **A TAX, not a lever.** | `3c1fbd4` |
| Half the EP skew is rank 0's shared expert | 8.98 us/layer = 0.683 ms, all in W13+W2, paid by peers as IDLE | `f31bc61`, `d809f73` |
| The other half is the last arriver's OWN floor | rank 0 arrives **LAST** and still waits 9.94 us/layer. So the 0.790 ms is **not skew**. Peers uniform to 0.5%. A perfect peer rebalance is worth **0.034 ms** | `66746bf` |
| `MPK_SHARED_DUP=2` (doubling the shared expert) | **+0.106 ms**, inside noise -> a perfect shard is <= 0.103, a 4-way K-shard <= 0.077. **NO-GO** | `49af7c2` |
| `MPK_SHARED_DUP=3` | **+0.527 ms** where doubling cost +0.106 — **the peer-idle -> wall response is CONVEX**: coefficient 0.32 then marginal 1.28. Slack is ~4 us/layer. **Never price a WAIT line at face value** | `f546b32` |
| Sharing the shared expert across EP ranks | **wrong output** — weights are replicated, the activation is not. Reverted | `1e91ff0` |
| The 3x EP routed-expert imbalance | **floor effect, not a wall lever** — 1.05 us of makespan spread | `3aab146` |
| The EP peer wait | **skew, not seven serialized round trips** (17.49 -> 17.49 us/layer) | `e5bd5db` |
| The EP fold is redone **192x** in the qkv_a prologue | hoisting and widening both neutral (above) | — |

### 5h. q_b HEAD-SHARDING — CLOSED AT 0.228 ms (`95b93f6`)

The last open structural item on the board, and the one that moved the most.

`MPK_QB_SKIP_PEER_WAIT=1` deletes the 7 peer stores and the 7-peer poll at
`gang_mla_attn_fused_mi300.cuh:1193-1240` and nothing else. Wrong output by
construction, and **upstream of the router**, so it perturbs TopK and hence EP
balance — the arm is *coherent but divergent*, not garbage.

| arm | triple 1 | triple 2 | n=6 mean | min | max | sd |
|---|---|---|---|---|---|---|
| base | 10.677 / 10.381 / 10.412 | 10.559 / 10.560 / 10.402 | **10.498** | 10.381 | 10.677 | 0.118 |
| skippeer | 10.103 / 10.239 / 10.124 | 10.513 / 10.387 / 10.258 | **10.271** | 10.103 | 10.513 | 0.157 |

**Wall delta = 0.228 ms.** Pooled sd 0.139, se 0.0802, **t = 2.84 on df 10
(p ~ 0.02)**. Triple 1 alone said 0.335 (arms disjoint); triple 2 alone said
0.121 (arms overlapping).

Counters, MAKESPAN criterion, us/layer over 8 ranks:

| region | base -> skip | delta | median | ms |
|---|---|---|---|---|
| S19->S20 the peer wait | 9.06 -> 5.13 | -3.94 | -4.15 | -0.299 |
| S18->S22 | 33.31 -> 29.44 | -3.87 | -3.85 | -0.294 |
| S18->S28 attention tail | 44.43 -> 40.72 | -3.71 | -3.64 | -0.282 |
| **S22->S28 absorber** | 11.12 -> 11.27 | **+0.16** | +0.22 | +0.012 |
| S0->S14 layer span | 159.74 -> 155.92 | -3.82 | -4.00 | -0.290 |

Skew does **not** relocate (the absorber flips sign between runs); ~92% survives
to the layer span. The two routes disagree at n=6 (counters -0.290 vs wall
-0.228), candidates being instrument common-mode in level-but-not-slope, and the
router confound.

**Verdict: NO-GO. Head-sharding attention end-to-end is CLOSED.** 0.228 ms is
below the 0.26 ms noise floor and is the *entire ceiling* of a large refactor.

> **The price moved 0.667 -> 0.335 -> 0.228, downward every time it was
> re-measured. A structural price is optimistic until its ablation is repeated.**
> Corollary already burned once: **never price a rendezvous by residual.**

### 5i. SPECULATION / MTP / BATCH WIDTH

| item | result | commit |
|---|---|---|
| MTP, confirmed | **8.884 ms/token**, 1.196x, G1+G2+G3 | `466b940` |
| MTP acceptance | 0.861 in-megakernel / 0.750 end-to-end = 1.75 tok/iter | — |
| **Item 2 (width-2 speculation) — CLOSED ON ITS CEILING** | a **PERFECT `a2 = 1.0`** second draft is only **8.855 -> 8.208 ms/token (7.3%)**. Break-even `a2 = 0.769` is a coin flip. Not worth building | `1506e67` |
| The MTP chain itself | **0.494 ms** (later 0.403). The expensive half is the second decode row at 3.181 (5.458 at n=3) | `b980171`, `87106c0` |
| "the harness costs 3.7 ms" | superseded — there is **no 2 ms of bookkeeping to delete** | `ec5a7fc` |
| The second decode row: **+2.14 ms is BUILD GEOMETRY** | plain bs=2 decodes ONE token; `C_row = 1.206` prices the build shape (10.619 -> 12.754 for zero extra tokens) | `a015f60`, `25be6d9` |
| GEOM | **NO-GO.** Only 0.062-0.201 ms of the 2.210 ms survives the deletability test — under the noise floor. GEOM is **UNIFORM** (+25.7..+27.1 us/layer on all 8 ranks), i.e. **58% uniform rendezvous**, not skew | `35d143c`, `a9018e5`, `26f8aab` |
| The MTP self-draft arm | **hangs 3/3, deterministically** | — |
| The draft layer needs its own replay run | mechanism note | `a1329d1` |
| Batch width as a latency lever | **rejected.** It buys tokens/sec at the cost of +2.14 ms of wall — a regression against a single-stream latency target | — |

### 5j. NUMERICS / WEIGHT FORMAT

| item | result | commit |
|---|---|---|
| MXFP4 attention weights | pass the numerics gate (4/4 prompts, cross-rank identical), wall unchanged; **capped at ~0.9 ms** | `c10cfed`, `84ead7a` |
| MXFP4 o_proj | **0.14 ms, not 0.9** — the Phase 6 prefetch had already hidden it | `7afb1ee` |
| MXFP4 on MoE | nothing available | — |
| **FP8 activations as a blanket 4x claim** | **FALSE.** +12% at K=10240, **-26% at W_UV's K=512**. The probe measured a steady state the real stage never reaches | `922aaad` |
| gfx950 | **has no fp8 VALU dot** (dot11-insts is gone). The scaled MFMA is the only fp8-activation instruction | — |
| GEMM ISA | matches gpt-oss and is **ahead** — verify ISA on the real image, not a probe | — |

### 5k. THE ROUTER REGION AND THE ATTENTION TAIL

| item | result | commit |
|---|---|---|
| The router region (S31->S32, 1.076 ms) | **CLOSED: 54% barrier poll (6.999 us) / 46% compute (5.877 us, sd 1.3%)**. `cnt[6]/cnt_tail = 128.0` exactly in all 40 samples | `aade4f8` |
| **My own 0.44 ms widening ceiling is RETRACTED** | 128/232 enter, but there are exactly `NUM_EXPERTS/EPT = 128` router TILES and each worker already takes one — **the 104 idle workers have nothing to take.** EPT=1 makes 256 tiles = a second round, measured WORSE | `aade4f8` |
| TopK rank-select | **+33% on the tail.** O(N^2/P) is wrong at P=128 | — |
| o_proj | at **76% of HBM peak** (168 MB/layer replicated 8x); its block is **51% spin**, o_proj itself is 19% | `83ec8c7`, `c866ebf` |
| The attention tail is a THREE-way worker split (64/64/104) | 104 workers idle 33.5 us/layer, and a class of board arithmetic was invalidated by it | `cecadcd` |
| q_b quant prologue | **at the HBM roof; 41% is RMSNorm+quant**, and it is not load-issue-bound | — |
| qkv_a tile | **41% redundant prologue**, 46% residual-resolve, only 41% MFMA K-loop | `48fea7f`, `2b4baa9` |
| Re-absorbing W_UK/W_UV | does not pay — `GLM_UNABSORB_QB=0` kills the q_b head shard | — |

---

## 6. THE CORRECTED TRAPS — READ THIS BEFORE MEASURING ANYTHING

These are the errors that were actually published on this branch and then had to
be retracted. Each one produced a wrong conclusion first.

### 6a. TWO STAMP LABELS WERE WRONG

* **`S2->S16` was labelled "qkv_a tiles". It is not.** It is the *call boundary*
  between the layer task and the attention body, worth **0.094 ms**. The real
  qkv_a tiles are `S16->S17`, **0.900 ms**. Published wrong twice (`26f8aab`,
  `a9018e5`). GEOM totals were unaffected only by luck — S16->S17 was already
  summed, in the unclassified bucket.
* **`S5->S6` ("W13 tiles", 1.402 ms) still carries the verdict "AT THE HBM
  ROOF", and that premise was reached through a denominator that was wrong
  twice.** The chain was 74% -> 36% -> 76%. The *conclusion* survives — W13 is
  at 76% of its real roof and the whole phase only holds 0.233 ms — but it
  survives on the third denominator, not the first. Do not cite the 74% or the
  36%.
* Two more on the same board: `S19->S21` is **not one phase** (0.749 ms of
  decode tile work **plus** 0.643 ms of QB_TP cross-rank peer wait — the layer
  pays **two** cross-rank rendezvous per layer, not one; and that 0.643 estimate
  is now the **0.228 ms ablated** figure of Section 5h). And *"216 workers idle
  through a 16-worker phase"* was stale geometry (it is 64/168 since the KV-chunk
  harvest) **and** mislocated — the idle set crosses S19->S21 in 1.70 us.

**Re-derive the map in one command, do not trust a label:**

```
grep -rn "mpk_stage_stamp(" include/mirage/persistent_kernel/
```

Every id has exactly ONE call site. Two files stamp and both run every layer;
`gang_mla_full_layer_fused_mi300.cuh:1405` includes and calls the attention
kernel immediately after stamp 2, so program order interleaves the two files.

### 6b. THE CONDITIONAL-STAMP DENOMINATOR RULE

**A stamp is only a synchronization point for the workers that PARTICIPATE in
the phase it ends.** Stamp populations on rank 1, universe 232:

```
slots 3, 4  ->   8 writers
slot  20    ->  64
slot  22    -> 128
slots 29-31 -> 192
all others  -> 232
```

Stamps 0/1/3 carry `cnt = 38836` (full: all workers, all 76 layers, all
iterations). **Stamp 4 is CONDITIONAL** — per-worker counts range **32 to
12972**. Differencing a conditional stamp against a full-count stamp is
**illegal**, and the tell that you have done it is a **negative interval**.

Same rule on the subphase counters: **SP slots have per-slot populations —
divide by the guard's writer count, not by 232.** SP4 slots are **deltas**, not
cumulative. **Read a guard before dividing by it.** This has produced wrong
conclusions more than once.

### 6c. THE SLACK-VS-MAX DOUBLE COUNT

Two estimators are legitimate and they answer different questions:

* `max(b) - max(a)` — the board's. **Telescopes** to the layer span. This is the
  critical-path estimator.
* Paired per-worker span — correct for *"how long does a worker spend here"*.
  **Does not telescope.** Do not sum it across regions.

The hole metric must reference the **FLOOR**:
`hole_r = mean_w(wait_w, r) - floor_r` (plus non-writers' full mean), which
telescopes to `232 x (span - sum floor)`.

**The max-referenced alternative DOUBLE COUNTS.** Referencing each worker's
slack to the region's slowest finisher gives a different ranking (S21->S23 first
at 1.226 ms) and an **8.303 ms** total inside a 12.098 ms layer. The tell is
arithmetic: `sum_r max_r = 262 us/layer against a 159 us layer`. The region's
max worker is a *different worker* in every region, so the metric charges every
later region for the same idle time.

### 6d. THE W13 232-WIDE ROOFLINE DENOMINATOR

The `W13 live = 64` label on the narrow-grid bandwidth table came from a **stale
`127 live W13 tiles` kernel comment for a different config.** W13's verified
geometry is **54 tiles/XCD over 29 workers = a 232-WIDE phase**. Per-CU share
23.2 GB/s, aggregate roof 5373 GB/s, achieved 4095 = **76%**.

Not 36%, not "2.5x unexplained in the biggest tile phase." **That single
correction retro-explains three measured nulls** — `MPK_MOE_PF_GROUPS` won 28%
in a standalone and moved the wall 0.000 three times, because at 76% of roof the
whole phase only holds 0.233 ms.

**Rule: before pricing a bandwidth idea, get the phase's real WIDTH and use the
measured curve at that width — not 5.17 TB/s, and not 5.17/256 per CU.**

### 6e. OTHER INSTRUMENT TRAPS THAT COST A RUN

* `MPK_SUBPHASE_TIMING`'s cost **scales with tile count** — 0.4 ms at 4 chunks,
  **3.1 ms at 16**. It once **fully masked a real 2.76 ms win.** Do not leave it
  on while measuring; use ablation probes when tile count is the variable.
* The stage stamp cost **9 us/layer** of its own via GPU-wide atomics until
  private per-worker rows fixed it (`a4e0853`, 14.260 -> 11.881 with the same
  probe). Set `MPK_BAR_SKEW_DROP_NS=1000000` — a 10 ms drop guard was keeping
  one stale sample per iteration and inflated the boundary 8.21 -> 20.18 us
  (`11a9af3`).
* **Host-mapped debug buffers must be allocated Coherent.** Non-coherent ones
  invented **two** wedged-worker findings that did not exist.
* **BSDBG stage 0 is a torn read** and it **cannot see the dense prologue**. Use
  `MPK_BSDBG_SEQ` / `MPK_BSDBG_LAYER0`.
* **The `--profiling` dump is written by all 8 ranks to one path.** Suffix it per
  rank (`9e6ee25`).
* **`.so` md5 + a flag-in-the-log do NOT prove a kernel runs.** `MOE_KBATCH`
  went into a kernel with **no caller** and measured a clean null (`2aa2bd6`).
  C++/header edits need a **two-step** rebuild — `build_ext --inplace` is a no-op
  after a fresh `.a`. **Grep the `.so` for a new string.**
* **Per-rank stamps carry NO cross-rank order.** The EP peer poll is not the
  cost; you cannot infer cross-rank causality from intra-rank stamps.
* **`task_layer_idx` is run-monotonic**, and gang-task tensors are
  **column-sliced by `bid.x`**.
* **The EP tail layer desyncs derived epoch counters** — store `task_layer_idx+1`,
  never increment.
* **`nt` loads hit cache; peer polls need `sc0 sc1`.** Livelock paced by the 1 Hz
  poll is the tell.
* **A self-heal must test the WHOLE predicate.** The partial version *was* the
  NP=8 EP hang.
* **The grid-stride makespan model is falsified.** Do not use it.
* The **NP=8 EP fault has two presentations** (hang at bootstrap, and illegal
  access at iteration 1) and is **~1-in-3, streaky**. **Repeat a run before
  concluding your change caused it, and repeat before blaming an instrument.**
  The NP=8 straggler is **box-level**, not the barriers.

### 6f. THE CORRECTNESS TRAPS

* **Exact-token equality is an ILLEGAL gate** (Section 1). Use `G1+G2+G3` via
  `correctness_gate.py`. `compare_tokens.py`'s gate is **cross-rank identity,
  not exact match** — it is G1 only.
* **`bs=2 emits wrong tokens` is RETRACTED.** bs=2 is fine; the exact-prefix
  0/264 that "proved" it proves nothing.
* **Wrong-output probes UPSTREAM OF THE ROUTER are invalid as ceilings.**
  Garbage changes TopK, hence EP balance, hence the whole MoE phase's shape
  (`92afbb7`: 11.013 -> 13.841 doing **less** work). `MPK_QB_SKIP_PEER_WAIT` is
  upstream of the router — its 0.228 carries that caveat and is reported with it.

---

## 7. HOW TO RUN IT (the operational floor)

```
MODEL_PATH=/home/claudeuser/models/glm5-mxfp4      # PIN IT. env_common.sh
                                                   # defaults to GLM-4.7-Flash
                                                   # (62 GB) and an unpinned run
                                                   # silently reports ~4 ms/iter
MIRAGE_HOME=/home/claudeuser/fleet-chiplet-megakernel
PYTHONPATH=/home/claudeuser/fleet-chiplet-megakernel  # a stale editable install
                                                      # resolves `import mirage`
                                                      # to a checkout with no GLM
MPK_NUM_WORKERS=232        # 240 HANGS at NP=8
ulimit -c 0                # one GPU fault dumped 440 GB and filled the disk
OMP_NUM_THREADS capped     # 8 ranks x 256 threads on 256 cores = 28x oversubscribe
```

* `rm -rf demo/glm5/permanent_output_dir*` after any kernel or header change.
* Long runs need `setsid`; `nohup &` gets SIGKILLed. Freeze the tree — a
  mid-sweep edit is a second variable.
* Killing a run: **split the pattern literal** (`pkill -f "demo""\.py"`) — an
  unsplit `pkill -f mpirun` matches the tool's own command line. **Never kill by
  process name** — `pgrep -x python3` matches the orchestrator. Then **check for
  orphan ranks**; GPUs at 0% utilization during a run is the tell.
* `git status` fails on a submodule error; use
  `git -c status.submodulesummary=0 --no-optional-locks status --short --ignore-submodules=all`.
* If one worker shows `MPK_WS_UNWRITTEN`, that is a block that never got a CU.
  **Cut worker count before blaming a barrier.**

---

## 8. WHAT IT WOULD TAKE TO REACH 2 ms/TOKEN

> **CORRECTION, 2026-08-23.** This section originally concluded "2 ms/token is
> below the floor of this parallelization." **That conclusion is RETRACTED on
> two counts, both of which survive review of the measurements themselves —
> the numbers below are right; the inference from them was not.**
>
> **(A) The margin was misread.** The floor is 2.144 and the goal is *close to*
> 2 ms — a **7% margin**. 2.2-2.5 ms/token satisfies the goal. The finding is
> **"4.1x above the floor,"** not "below" it.
>
> **(B) THE FLOOR IS A BYTE FLOOR, AND THE ATTENTION SIDE IS NOT BYTE-BOUND.**
> Section 2 prices every phase by bytes. Side by side with the measured tile
> time:
>
> | | byte floor | measured tile | ratio |
> |---|---|---|---|
> | MoE W13 + W2 | 14.93 us/layer | 26.11 us/layer | 1.75x |
> | **all non-MoE phases** | **8.55 us/layer** | **55.9 us/layer** | **6.5x** |
>
> That 47.3 us/layer x 76 = **3.55 ms is not accounted for anywhere in this
> ledger** — and it is larger than the 2.94 ms attributed below to all ten
> rendezvous. The cause was already on the branch in two unjoined pieces:
> qkv_a's tile is **41% RMSNorm+quant prologue that touches no HBM**, and that
> prologue is **VALU-bound, not load-issue-bound**. *A VALU-bound region has no
> byte floor and was never going to appear in a roofline.*
>
> Note this does NOT resurrect widening-for-bandwidth (§5, 0.118 ms for the
> class). That prices moving a phase to the 232-wide *rate*. It does not ask
> why qkv_a takes 14.8 us of tile when 3.33 us of bytes cross the bus at its
> own real width of 168. Different question; only the first one is closed.
>
> **What replaced the reasoning below:** `demo/glm5/makespan_predictor.py`
> (`588e1e3`) — the wall delta is the change in the **maximum arrival**, and
> each proposed change falls into one of three regimes with a computable
> ceiling. Read it before using anything in this section.

### The arithmetic

```
measured                     10.498 ms/iter  (bs=1, n=6)
best confirmed               8.884  ms/token (MTP, 1.196x)
width-corrected floor        2.144  ms       (1.761 HBM + 0.383 EP)
target                       2.000  ms
```

Reaching 2 ms/token means running **below the width-corrected HBM+EP floor of
this parallelization** — *if* the floor is what binds. It is not shown to be:
the floor is a byte floor and the non-MoE tiles run at 6.5x theirs (see the
correction above). What the arithmetic does establish is the **4.1x gap** and
that no *byte*-side lever closes it.

### The gap is one thing, and it is not tuning

The distance from 8.884 to 2.144 is **schedule**, and schedule here means
**rendezvous count x arrival skew**:

```
10 GPU-wide rendezvous/layer x 3.77 us  =  37.7 us
+ 2 cross-rank rendezvous/layer
x 76 layers                             =  ~2.9 ms of pure rendezvous
```

plus 6.498 ms of last-arriver floor (4.99 ms of it local), of which only 2.277 ms
is provable spin and **none of that 2.277 has any movable work to put in it**.

Every other class is closed:

| class | status |
|---|---|
| bandwidth / widening | **0.118 ms** for the entire class |
| tile geometry | measured out in **both** directions on both MoE phases |
| occupancy | ladder dead; the lock is LDS; the register gate opened and was null |
| barrier mechanism | **eleven** attacks, all neutral or negative |
| prefetch / latency hiding | the one window with an idle bus has an empty movable set **by construction** |
| numerics / weight format | MXFP4 exhausted at 0.9 + 0.14; fp8 activations are a **net loss** at real K |
| EP collective | a rank-alignment **TAX** — 83% relocates when deleted |
| speculation | MTP landed at 1.196x; width-2 closes on a **7.3% perfect-acceptance ceiling** |
| batch width | +2.14 ms of wall — a latency **regression** |

### What would actually have to change

Not one of these is a tuning knob. Each is a different program.

1. **Delete rendezvous by deleting phases, which requires a different
   decomposition of the layer.** The current layer needs 10 GPU-wide barriers
   because it is 10 data-dependent rounds over one shared worker pool. Cutting
   to, say, 4 rounds means a fundamentally different tiling of MLA + MoE — not a
   fusion of the existing rounds, because **no legal independent-round fusion
   exists** (Blocker 1).

2. **Break `qkv_a(L+1) <- MoE(L)` — which the model forbids.** The only way to
   overlap layers at bs=1 is to have work from L+1 that does not depend on L's
   output. There is none. This is a property of a decoder-only transformer at
   batch 1, not of this implementation. Every classical fix (pipelining,
   heterogeneous groups, second stream) needs a second stream of work, and
   `959f14a` established **there is no second stream at batch 1.**

3. **Speculate deeper, which the acceptance curve does not support.** MTP is
   already 1.196x at `a1 = 0.861`. Width-2 was closed **on its ceiling**: a
   *perfect* second draft (`a2 = 1.0`) is only 8.855 -> 8.208, 7.3%. Depth would
   need acceptance the model does not have.

4. **A fundamentally lower byte count per token.** The floor is 10.01 GB/iter on
   the busiest rank. MXFP4 is already on the attention weights and o_proj; the
   MoE weights are already MXFP4; activations are already FP8 (and going further
   on activations is a measured **loss** at real K). There is no format left.

5. **More GPUs, i.e. a different parallelization.** The floor of 2.144 ms is per
   this 8-rank EP+TP split. Sharding wider would cut the per-rank byte floor —
   and would add cross-rank rendezvous, whose price is **skew, paid once at the
   first cross-rank rendezvous** and already 1.506 ms of the current floor.
   Whether that trade is net-positive is a question this hardware configuration
   cannot answer.

### The plain statement — as corrected

**The gap is 4.1x, and it is not a byte gap.** The achievable *byte* floor is
2.144 ms; the measured bs=1 wall is 10.619 ms. Of the distance:

| | ms | status |
|---|---|---|
| rendezvous (10 GPU-wide + 2 cross-rank, x76) | ~2.94 | item 3 — the CROC per-XCD chunk barrier is on disk and unported |
| non-MoE tile time above its byte floor | **3.55** | item 2 — **PROFILED 2026-08-29, see below** |
| MoE tile above its byte floor | 1.10 | closed both directions |
| byte floor itself | 2.14 | closed |

Two of these four lines are open, and the larger of the two open lines is the
one this ledger never priced. **The intra-layer program is not exhausted; the
BYTE-side intra-layer program is.** Every *tuning* class is measured out and
sums to under the 0.26 ms noise floor — that part stands and is what §5 is for.

What does **not** stand is the inference "therefore 2 ms is unreachable."
Before that can be claimed again, the 3.55 ms has to be attributed at ISA
level, and it has to be attributed **subject to the predictor** — which says
per-phase tile cuts are capped at 0.000-1.199 us/layer because the workers that
skip a phase arrive as late as the workers that run it. The 3.55 ms is
therefore *not* in any one phase's tile. It is in the **plateau**: work every
worker does regardless of which tile it owns.

#### Item 2, profiled — 2026-08-29

`isa_outstanding.py` over today's gfx950 image. The 3.55 ms is **not** VALU and
it is **not** the plateau. It is **memory-latency serialization in every
non-MoE loop**, and the split is total:

| loop | loads | carry | drains | verdict |
|---|---|---|---|---|
| W13 tile | 12 | **12** | 0 | flat part of the curve |
| W2 tile | 8 | **8** | 1 | flat part of the curve |
| router GEMM | 12 | **0** | **12** | one load issued, drained, consumed, x12 |
| GEMV (W_UK / W_UV / o_proj) | 16 | **0** | 2 | issues 8, drains, issues 8, drains |
| MLA decode | 10 | **0** | 2 | same shape |
| KV latent write | 7 | **0** | 3 | same shape |

The two MoE kernels are the only software-pipelined loops in the layer. Every
other loop carries **nothing** across its backedge, so each trip pays a full
round trip it never overlaps. The router GEMM is the extreme: twelve
`s_waitcnt vmcnt(0)` in a 93-instruction body.

**Why fixing it still does not pay, and what would change that.** The
predictor's ceiling for a phase is `max_all(b) − max_{w∉P}(b)`, and it is ~0
for exactly these loops *because they are narrow* — router 128 workers, o_proj
192, MLA decode 64, of 232. The non-participants set the max, so pipelining the
participants' loops moves nothing. W13 and W2 are the only 232-wide phases,
which is both why they are the only ones anyone bothered to pipeline and why
they are the only ones where a tile cut converts 1:1.

So item 2 is now two facts, not one: the time is real and mechanically
identified, and it is unreachable *at the current phase widths*. The lever that
unlocks it is not a better k-loop, it is **making a phase 232 wide** — with no
non-participant there is no one left to set the max, and the phase's ceiling
becomes its whole span. The obstruction is arithmetic: o_proj's 1536 columns
per rank divide into 24 or 32 tiles, not the 29 that would put one tile on
every worker.

---

## 9. INDEX

| topic | file |
|---|---|
| the labelled bs=1 layer budget + verdicts | `demo/glm5/board_budget.py` |
| floor/hole decomposition, the four release holes | `demo/glm5/hole_audit.py` |
| the flat roofline (2.319 ms) | `demo/glm5/roofline.py`, `demo/glm5/ROOFLINE.md` |
| the width-corrected roofline (2.144 ms) | `demo/glm5/width_corrected_roofline.py` |
| THE correctness gate (G1/G2/G3) | `demo/glm5/correctness_gate.py` |
| MTP confirmation | `demo/glm5/probe_mtp_regate.sh`, `gate_mtp_exact.sh` |
| q_b peer-wait ablation, n=6 | `demo/glm5/probe_qb_peer_wait_ceiling.sh` |
| EP collective ceiling | `demo/glm5/probe_ep_collective_ceiling.sh` |
| shared-expert makespan (`MPK_SHARED_DUP`) | `demo/glm5/probe_shared_expert_makespan.sh` |
| MLA decode ablation ceiling | `demo/glm5/probe_decode_ablation_ceiling.sh` |
| width-2 speculation ceiling | `demo/glm5/price_item2_width2.py` |
| MoE row fold ceiling | `demo/glm5/price_moe_row_fold.py` |
| second-row / GEOM classification | `demo/glm5/classify_geom.py` |
| bs=1 nondeterminism (the two attractors) | `demo/glm5/probe_bs1_determinism.sh` |
| paired per-worker span estimator | `demo/glm5/paired_span.py` |
| **the makespan predictor (3 regimes, arrival ceilings)** | **`demo/glm5/makespan_predictor.py`** |
| W2 tile-width ledger | `demo/glm5/demo.py:975-1020` |
| MoE prefetch ledger (`MPK_MOE_PF_GROUPS`, `W2_K_SPLITS`) | `include/mirage/persistent_kernel/tasks/mi300/gang_moe_linear_mxfp8_mi300.cuh:137-204, 1085-1120` |
| the narrow-grid bandwidth curve | `tests/standalone/test_narrow_grid_bandwidth.hip` |
| gpt-oss ledger (diff against it, do not re-derive) | `~/mirage` (READ-ONLY) |
