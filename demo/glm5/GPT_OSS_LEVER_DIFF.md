# RE-STEER v8, Item 1 — the gpt-oss lever walk and the per-layer diff

Reproducer: `demo/glm5/gpt_oss_lever_diff.py`. Every number is cited to a
commit or a measurement. `~/mirage` was read, never written.

Baseline this turn: **10.543 ms/token, n=3** (10.529 / 10.398 / 10.703), all
three texts checked and correct — two runs land on the terse attractor
("…Paris. The capital of Germany is"), one on the reasoning attractor
("Knowledge retrieval: France -> Capital -> Paris"), both legal
(`glm-exact-token-equality-is-an-illegal-gate`). Pooled with the n=5 from
`a678637`: **10.571 ms/token, n=8**. No build change.

---

## §0 The answer, in three lines

1. **Item 1.1/1.2 is empty.** 22 levers from `~/mirage`'s
   3.339 → 1.936 series. **15 PRESENT in GLM, 5 ported-and-null, 1
   structurally inapplicable, 1 not-ported — and that one (XOR swizzle,
   `0352a70`) never landed in gpt-oss's shipping path either**, so it is not
   part of the 27.9 %. There is no unported gpt-oss lever.
2. **Item 1.3 relocates the gap off the tiles.** Normalized per layer,
   **GLM's tile class is 1.13x MORE byte-efficient than gpt-oss's entire
   layer** (1.60 vs 1.42 MB/µs). At gpt-oss's own MB/µs GLM's tiles would cost
   93.06 µs/layer; they cost 82.21.
3. **The gap is per-layer FIXED cost, paid 2.11x as often.** GLM has 76 layers
   to gpt-oss's 36 but only 1.73x the bytes per layer. Rendezvous + boundary
   is **48.5 µs/layer = 41 % of GLM's local layer, 90 % of gpt-oss's ENTIRE
   layer, and it moves no bytes at all.**

---

## §1 gpt-oss's 1.936 ms is a SINGLE-GPU number

Verified this turn, read-only:

| file | evidence |
|---|---|
| `~/mirage/demo/gpt_oss/demo.py:374` | `world_size = 1` |
| `~/mirage/demo/gpt_oss/bench_1k1k.sh` | `HIP_VISIBLE_DEVICES=<one device>` |
| `~/mirage/demo/gpt_oss/bench_vllm_v017.sh` | `HIP_VISIBLE_DEVICES=<one device>` |

gpt-oss pays **zero** cross-rank collective and **zero** cross-rank
rendezvous. GLM pays a measured **1.587 ms** (`coll` class: EP collective
15.617 + QB_TP cross-rank gather 9.034 µs/layer).

**A correction to my own earlier note in this session.** I first put the
cross-rank cost at 3.09 ms by adding the `coll` class (1.587) to the
"1.506 ms cross-rank last-arriver floor". Those are **the same region measured
by two estimators**, not two costs — `barrier_floor_sweep.py` prices the EP
collective at 1.894 ms where the budget's line prices it 1.005 ms. They are
not additive. The cross-rank total is **1.587 ms**, and it is 49 % of the gap,
not 94 %.

---

## §2 Item 1.1/1.2 — the lever table

| commit | lever | delta | status |
|---|---|---:|---|
| `36fb412` | Monotonic barriers | −6.7 % | **P** |
| `74173f1` | Fused-layer batching + MoE scale overlap | −7.5 % | **P** |
| `c156900` | OProj prefetch + FP8 reciprocal | −5.6 % | **P** |
| `5f51a90` | O-proj LDS prefetch | −4.5 % | **P** |
| `1b8787f` | O-proj LDS prefetch correctness fix | −4.2 % | **P** |
| `bda8607` | `global_load_lds_dwordx4` in W13 + W2 | −3.4 % | **P** |
| `e8adbc0` | Fused LM head + argmax | −3.1 % | **(c)** |
| `dff01a2` | W13 K=2944 tile shape | −2.7 % | **(b)** |
| `24b8d4e` | W2 prefetch overlap | −2.3 % | **(b)** |
| `2e0f83c` | Collapse 36 per-layer tasks to 1 | −2.1 % | **P** |
| `85f77bf` | CROC chunk barrier | −1.8 % | **P** |
| `ec9d969` | merge TPT 16 → 32 | −1.4 % | **(b)** |
| `9ea3ebd` | vmcnt serialization fix | −1.4 % | **P** |
| `79d55aa` | Fuse write-through store | −1.2 % | **P** |
| `a188c46` | exp2 TopK | −0.9 % | **P** |
| `c7bf746` | Per-wave W13 decomposition | −0.9 % | **(b)** |
| `92735b1` | Mechanism C | −0.6 % | **P** |
| `78dedf0` | All-thread barrier polling | −0.6 % | **(b)** |
| `681a57e` | 301 → 21 tasks, 42 → 7 events | — | **P** |
| `df77f37` | Multi-layer fusion | — | **P** |
| `0352a70` | XOR swizzle (standalone bench, 7.9x) | — | **(a)** |
| `b94a54e` | Branchless TopK | — | **P** |

### The three findings that had to be checked, not assumed

**The MoE fusion is PRESENT.** gpt-oss has a dedicated
`gang_moe_fused_mxfp4_mi300.cuh` that GLM's include closure does not pull in,
which reads as absent. It is not: `demo/glm5/demo.py:3024` says *"W13+SwiGLU
and W2+MulSumAdd ran as Phases 5 and 7 of the fused task; only the residual add
is left"*, and both folds (`GLM_FUSE_MOE_SWIGLU`, `GLM_FUSE_MOE_MULSUMADD`)
default to 1. GLM fuses the MoE one level higher — into the whole-layer gang
task rather than into a MoE-only one.

**The task-graph compaction is PRESENT.** GLM's log prints
`Number of all tasks: 1374` / `Number of all events: 128`, which against
gpt-oss's post-`681a57e` 21/7 reads as a large absent lever. It is not: that
is the *build-time* graph description. GLM runs the multi-layer loop in-kernel
— `MPK_ABL_ML_BOUNDARY` exists precisely as the ablation probe for its
per-layer bookkeeping, and it prices the **entire** boundary at **0.051 ms**
(`glm-layer-boundary-ablation-is-zero`, paired A/B, min-of-115, n=3).

**The only (a) is not a lever.** XOR swizzle is absent from *both* include
closures. Its 7.9x is a standalone bench (`0352a70`, `bd4a763`) that never
reached gpt-oss's shipping path, so it is not part of the 27.9 % and porting
it would be speculative, not a catch-up.

---

## §3 Item 1.3 — the per-layer normalization

`~/mirage/demo/gpt_oss/ROOFLINE_ANALYSIS.md` (3.339 ms) and
`PHASE_BREAKDOWN.md` (3.25 ms) are both stale, exactly as the v8 brief warned,
and **no per-phase table at 1.936 exists in `~/mirage`**. So the per-phase diff
cannot be done from the source the brief nominated. Normalizing per layer can
be, from numbers on both sides that are current.

| | GLM-5 744B | gpt-oss 120B | ratio |
|---|---:|---:|---:|
| layers | 76 | 36 | 2.11x |
| GB/token/GPU | 10.010 | 2.740 | 3.65x |
| MB/layer/GPU | 131.71 | 76.11 | **1.73x** |
| byte floor, ms/token | 1.936 | 0.530 | 3.65x |
| MEASURED wall, ms/token | 10.571 | 1.936 | 5.46x |
| cross-rank, ms/token | 1.587 | **0.000** | — |
| LOCAL wall, ms/token | 8.984 | 1.936 | 4.64x |
| µs/layer, wall | 139.09 | 53.78 | **2.59x** |
| µs/layer, LOCAL | 118.21 | 53.78 | **2.20x** |

Efficiency vs own byte roof: gpt-oss **27.4 %**, GLM as measured **18.3 %**,
GLM with cross-rank removed **21.6 %**.

### The load-bearing row

| | µs/layer | MB/layer | MB/µs |
|---|---:|---:|---:|
| GLM **tile class only** | 82.21 | 131.71 | **1.60** |
| gpt-oss **ENTIRE layer** | 53.78 | 76.11 | 1.42 |

**GLM's tiles are 1.13x more byte-efficient than gpt-oss's whole layer.** At
gpt-oss's own MB/µs, GLM's tiles would cost 93.06 µs/layer — *more* than the
82.21 they cost. The tile class cannot be the gap. This is consistent with,
and independent of, `TILE_CLASS_BOUND.md`'s finding that the class is
latency-bound with both units idle: the class is inefficient against its
*byte roof* and still ahead of gpt-oss's realized rate.

---

## §4 Reconciliation with the guide's 7.32, and the two board lines

The v8 target is built as `1.936 / 0.279 + 0.383`, where 0.383 is the EP
**byte floor** — i.e. it charges GLM only the *ideal* cross-rank cost. Charging
the *measured* one gives `1.936 / 0.274 + 1.587 = 8.660 ms`. The entire
difference between 7.32 and 8.660 is that single substitution, and it names the
first board line exactly.

| | ms/token |
|---|---:|
| measured, pooled n=8 | **10.571** |
| v8 target | 7.320 |
| **TOTAL GAP** | **3.251** |
| line 1 — cross-rank headroom above its own floor (1.587 − 0.383) | **1.204** — class `coll` |
| line 2 — local gap at gpt-oss's own efficiency | **1.911** — **not** in `tile` |
| line 3 — rounding (0.279 vs 0.274; pooled n=8) | 0.136 |

**Line 2 lives in rendezvous + boundary: 3.123 ms, 48.5 µs/layer.** That is
41 % of GLM's local layer and 90 % of gpt-oss's *entire* layer, and it moves no
bytes. The mechanism is arithmetic, not a defect: GLM pays every per-layer
fixed cost **2.11x as often** as gpt-oss for only **1.73x** the bytes per
layer. Fixed cost does not scale with bytes.

---

## §5 What this does and does not close

**Closed by measurement:** Item 1 in full. Every gpt-oss lever is present,
null, or inapplicable. The one absent lever is absent from gpt-oss's shipping
path too.

**Closed by in-repo pricing, before this turn:** Item 2. `isa_accounting.py`
prices the router's 17-link chain at **≤0.091 ms** and *all five byte-light
phases made instantaneous* at **0.111 ms** — both under the 0.26 ms noise
floor. The quantizer candidate is ~0.05 ms.

**Open, and now precisely located:** the two board lines above, 1.204 and
1.911 ms. Both sit in classes where every *mechanism* lever already measured
null — barrier narrowing, all-thread polling, arrival trees, worker-count 200,
deleting a whole rendezvous (0.10 ms), EP fold hoist, EP fold widen,
shared-expert hoist. What has **not** been attacked is the *count*: 10
rendezvous per layer × 76 layers. `glm-no-legal-independent-round-fusion-exists`
says the only way to cut the count is to delete a phase.

This board is **not** closed on a floor argument, and it does not claim
impossibility. It reports that the gap is 3.251 ms, that it is in two named
lines, and that neither line is in the tile class where the v8 brief's
Items 1 and 2 pointed.

---

## §7 RE-VERIFIED against the 1.93 → 1.64 series — 2026-09-03

§2's walk covered `~/mirage`'s 3.339 → 1.936 series. The
`origin/amd_mi355_gpt_oss120b` branch has since taken gpt-oss to **1.64
ms/token** with nine further levers, so the "no unported lever" claim was
re-checked against them rather than assumed to still hold.

**It holds, and now for a stated reason rather than by enumeration.** The
per-lever assessment lives in
`.claude/skills/megakernel-decode-levers/references/glm5-port-status.md`; the
levers themselves are cataloged in that skill's `SKILL.md`. Summary:

| gpt-oss lever | delta | GLM-5 |
|---|---|---|
| 5 default-on flag flips | 1.902 → 1.877 | **empty** — all 125 off-by-default `MPK_*` flags audited, none has a recorded win |
| `MPK_ATTN_SLICE_RELEASE` | 1.880 → 1.864 | capped 0.095 ms (§4 of `RENDEZVOUS_EDGE_TABLE.md`) |
| `MPK_W13_T0_COUNTED_HANDOFF` | 1.865 → 1.840 | tile channel, closed ×3 |
| RMSNorm ssq off LDS | 1.851 → 1.835 | **portable**, predicted ~0.035 ms — at the noise floor |
| `MPK_MOE_XCD_PAIR` | 1.706 → 1.642 | structurally inapplicable under EP |
| 4 shipped tile levers | — | tile channel, closed ×3 |

Two measurements already in this tree cap the whole class, which is why the
answer is the same as in §0 for a newer series:

1. **Tile-level wins do not convert.** `MPK_MOE_PF_GROUPS` is a 28% standalone
   tile win with three nulls in three paired A/Bs (§5d). `MPK_ATTN_GEMV_PF`
   (`039a35e`) is a fourth instance: load depth 8 → 32, drains 2 → 1, zero
   spills, **neutral** wall. Most of the gpt-oss series is tile-level latency
   hiding.
2. **Barrier-scope narrowing is capped at 0.095 ms** for all ten rendezvous at
   once, because every chiplet holds a worker at the global max.

**`MPK_MOE_XCD_PAIR` is the one worth understanding**, because GLM-5 genuinely
still does what it deleted — the W13 prologue walks `d_mask[i]` then loads
`d_routing[e*B+tok]`, and no weight address exists until that finishes. It
cannot port because the map is only static when #picks == #XCD-pairs:

| | gpt-oss | GLM-5 |
|---|---:|---:|
| routed experts / top-k | 128 / **4** | 256 / **8** |
| EP ranks | **1** | **4** |
| live experts per rank | **4** | **~2** |
| XCDs per GPU | 8 | 8 |

Pinning an expert to an XCD pair would idle ~6 of 8 XCDs. GLM-5's round-robin
stripe is load balancing, not laziness (`gang_moe_linear_mxfp8_mi300.cuh:1571`).

### The part that reframes the goal

gpt-oss runs at `world_size = 1`. It pays **zero** cross-rank collective and
**zero** cross-rank rendezvous; GLM-5 pays a measured **1.587 ms**. That is now
the largest GLM-5 class with no gpt-oss analogue at all — so gpt-oss's 1.64 ms
is not a template for a 4-rank 744B EP decode, and catch-up levers for the
`coll` class must come from a multi-GPU reference, not from that branch.
