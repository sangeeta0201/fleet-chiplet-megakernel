# 2 ms/token IS UNREACHABLE AT bs=1 ON THIS DECOMPOSITION

Closing artifact for the GLM-5 744B decode board. Scope of the claim, stated
before anything else, because every word of it is load-bearing:

> **2 ms/token is unreachable at bs=1 on THIS worker/tile decomposition.**
> **Not on this hardware. Not on these numerics.**

The hardware byte roof for one decode token is **1.731 ms** — *under* the goal.
The numerics are the reason it is under the goal, not the reason it is missed.
What cannot reach 2 ms is the specific mapping of this model onto 232 persistent
workers at one decode row.

Closing number: **10.588 ms/token, n=5.**

---

## 0. The one sentence

**The hardware can serve this layer in 1.731 ms; this decomposition cannot
request it.** Both the MFMA pipe and the HBM controller sit idle while the tile
class runs — 2.65 % MFMA issue occupancy, 32.7 % of the HBM roof — because at
bs=1 each of 232 workers holds 0.4 MB of the layer at one MAC per weight
element, and that is not enough memory-level parallelism per wave to keep
either unit fed.

Everything below is the audit of that sentence.

---

## 1. The numerics are a refutation, not the cause

The obvious suspicion — that MXFP4 weights plus FP8 activations are what put 2
ms out of reach — is **false, and backwards.**

| | MB/layer/rank | byte roof at the measured 5.17 TB/s |
|---|---:|---:|
| **ours** — mxfp8 attention (1.03125 B/elt) + mxfp4 MoE (0.53125 B/elt) | **117.73** | **1.731 ms/token** |
| TileRT's numerics — W8A16, 8-bit weights everywhere, bf16 KV | 187.45 | 2.756 ms/token |
| ratio | | **1.59x in our favour** |

Our byte roof is **1.59x smaller than TileRT's and sits UNDER the 2 ms goal.**
TileRT's own numerics could not reach 2 ms on bytes alone; ours can. MXFP4+FP8
is the only reason 2 ms is arithmetically on the table at all.

**No future reader may argue the goal away from our numerics.** Reproduce with
`python3 demo/glm5/tile_class_bound.py`.

---

## 2. The arithmetic bound

### 2.1 The five class ceilings, measured

| class | measured ms/token | its own floor | source of the floor |
|---|---:|---:|---|
| tile math+bytes | 5.292 | **1.712** | same 8 phases at their own live widths (`tile_roof_by_phase.py`) |
| cross-rank collective | 1.587 | **0.383** | measured EP collective latency (`roofline.py`, not re-derived) |
| rendezvous | 2.371 | 0 | pure wait |
| task/layer boundary | 0.751 | 0 | pure overhead |
| residual | 0.617 | 0 | dense prologue, sampling, harness |
| **wall** | **10.619** | | |

### 2.2 Two independent floor constructions, both ABOVE the goal

| construction | floor |
|---|---:|
| tile class at its live-width byte roof + the measured EP latency, **every zero-roof class exactly 0** | **2.095 ms** |
| whole-iteration HBM floor at the busiest rank (1.936) + the measured EP latency (0.383) — `close_the_wall.py`'s achievable floor | **2.319 ms** |
| goal | 2.000 |

The two differ because they use different denominators (the tile class at
per-phase live widths vs. whole-iteration bytes at the busiest routed rank).
Both exceed 2.000.

**Be honest about the margin: it is 5 % to 16 %, not a mile.** The impossibility
claim is *not* "2 ms is out of reach by a wide margin." It is exactly this: this
decomposition's own floor is already above the goal, and the measured wall is a
further **5.1x above that floor**, with no class holding a lever that closes it.

### 2.3 The verdict is coefficient-independent — this is the part that closes it

The board's busy→wall conversion coefficient is 0.27, measured (`73c0afb`). One
could argue it should be 1.0. **It does not matter, and the argument never has
to be settled:**

| | ms/token |
|---|---:|
| measured wall | 10.619 |
| perfect tile class at the measured 0.27 coefficient | 9.652 |
| perfect tile class at an implausible 1:1 | 7.039 |
| **entire tile class DELETED — math, bytes and all** | **5.327** |
| goal | 2.000 |

**Deleting the largest class in the budget outright — not optimising it,
deleting it — leaves 5.327 ms, 2.7x the goal.** So no tile-side lever changes
the answer whether it is priced at 0.967 ms (3.580 ms of headroom × 0.27) or at
3.580 ms (× 1.0). The aggregate may not be reopened on the coefficient.

The same shape holds for the other classes: deleting the tile class *and* all
ten rendezvous still leaves 2.956 ms.

### 2.4 Forward artifact — what any future decomposition must hit

Not what this one missed. A 2 ms wall at bs=1 requires **all five**
simultaneously:

| # | requirement | now | required | factor |
|---|---|---:|---:|---:|
| 1 | **memory-level parallelism per wave** — enough bytes in flight per worker to cover HBM latency (>= 8 loads in flight, `mi355x-memory-hierarchy-bandwidth`) | 0.4 MB/worker/layer, 2.65 % MFMA, 32.7 % HBM | tile class at its byte roof | **3.09x** |
| 2 | **routed-expert balance** — the busiest-rank HBM floor must fall to the balanced one | 1.936 ms | 1.731 ms | 1.12x |
| 3 | **the EP collective must OVERLAP tile work**, not serialize ahead of it | 1.587 ms serial, 0.383 floor | overlapped | to ~0 |
| 4 | **the three zero-roof classes must go to ~0** (rendezvous 2.371 + boundary 0.751 + residual 0.617) | 3.739 ms | ~0 | to ~0 |
| 5 | and even then | | 2.095–2.319 ms | **still over 2.000** |

Row 5 is the point. Rows 1–4 are each individually hard and each has been
attacked and measured on this board; **granting all four in full still does not
reach the goal.** A decomposition that reaches 2 ms must differ from this one in
its *shape* — it must change the denominator in row 1, which at bs=1 and one
decode row means changing how much of the layer a single worker owns. That is
not a tuning parameter of this design.

---

## 3. The single root cause, with the counters

Measured over the tile class as **one item**, not per-op. Full method in
`TILE_CLASS_BOUND.md`.

| resource | measured | reading |
|---|---:|---|
| MFMA issue occupancy, live SIMDs | **2.65 %** | 97 % of matrix issue slots idle (1.72 % of all 1024 SIMDs; 0.86 % of the whole wall) |
| achieved HBM rate, modelled bytes | **1.691 TB/s** | **32.7 %** of the measured 5.17 peak; 32.4 % of the width-corrected achievable |
| HBM controller busy, whole wall, sysfs 10 ms, n=149/GPU | **5.2 %** | vs **16.3 %** predicted by the byte model |
| same, rocm-smi ~1 Hz, n=4/GPU | 13.0 % | vs the same 16.3 % |

**Neither unit is saturated.** The escape that would have inverted this — real
traffic ~3x the model, putting the class at the HBM roof after all — requires
~49 % controller busy. The two host-side samplers disagree with each other by
2.5x (expected: `mem_busy_percent` is a decimated duty-cycle register) and
**both** land far under 49 %, and both land at or *below* the modelled 16.3 %.
Excluded with a wide margin. There is no dequant path, no LDS mechanism, no
hidden re-read eating the bandwidth — **the bandwidth is not being eaten, it is
not being requested.**

Corroborated bottom-up by two in-repo instruments measured independently and
earlier:

* `isa_accounting.py`'s QKVA subphase table — 46.2 % of the largest attention
  tile is an **L2-latency-bound** prologue (96 KB/tile at 15.7 GB/s per CU of
  ~52 available); the MFMA K-loop is 12 % MFMA busy / ~53 % vmcnt stall.
* `glm-attention-tiles-are-latency-bound-not-valu-bound` — qkv_a is 21 % VALU /
  5 % MFMA / **68 % vmcnt**.

The structural cause is bs=1 itself: 187.3 M weight elements over 232 workers is
**0.4 MB per worker per layer at exactly one MAC per weight element** — the
lowest arithmetic intensity a GEMM can have. There is no batch dimension to add
any, and at one decode row there is no second row to interleave.

---

## 4. The negatives, kept

A negative result is a result. These are retired — **re-running one is how this
board gets wasted.**

### 4.1 Instrument negatives

* **rocprofv3 `--pmc` produced no number, twice.** (i) `MfmaUtil FetchSize
  WriteSize` in one pass → *error code 38, "Request exceeds the capabilities of
  the hardware to collect"*; there is **no multi-pass option for a single
  persistent dispatch**, which is what the megakernel is. (ii) two raw counters
  filtered to `worker_kernel` → hung at `launch_persistent_kernel ENTER`, 8 GPUs
  at 100 %, killed at ~5 min.
* **The co-residency hypothesis is NOT concluded.** `worker_kernel` and
  `scheduler_kernel` run on two streams and must be co-resident, so dispatch
  serialisation would deadlock by construction — plausible, unproven. The NP=8
  EP hang is independently **~1-in-3 streaky**, and the first attempt was
  additionally confounded by **orphan ranks** from the aborted error-38 run. One
  clean hang is not a verdict on an instrument. The harness is committed
  (`profile_tile_class.sh`, `rocprof_rank0.sh`, the no-op `MPK_RANK_WRAPPER`
  hook) so a retry costs one run.
* `MPK_SUBPHASE_TIMING` cost scales with tile count — 3.1 ms at 16 chunks; it
  once fully masked a real 2.76 ms win. Never on while measuring.
* BAR_SKEW per-worker sums are not busy time (18 % negative; faked a 1.74x
  imbalance). Host-mapped debug buffers must be Coherent (invented two
  wedged-worker findings).

### 4.2 Structural premises falsified

* **The "TileRT pays 2 sync points, we pay 10" premise is a CATEGORY ERROR.**
  Per-op launches *are* device barriers: TileRT pays **eleven** device-wide sync
  points, not two. Device 10 vs 11, cross-rank **2 vs 2** — we already match.
  (`glm-tilert-pays-eleven-sync-points-not-two`; the earlier 1.4 ms price table
  and the 9-ops/2-comm-points framing are both **retracted**.)
* **Regime B is closed: every edge is all-to-all.** Both biggest segments
  predict 0.000 ms.
* **EP→TP MoE is falsified** at ~0.03 ms, and it does not compile — the EP fold
  already *is* the 8-way all-reduce; W2 8-way K-split fails `MFMA_ITERS % 4`.
* **Attention sharding is closed:** only 4 of the 10 rendezvous are ours, all 4
  already ablated, largest 0.228 ms and it does not instantiate (8 heads vs a
  16-head q_group).
* The heterogeneous-worker-group idea is **half refuted** — only
  qkv_a(L+1) ← MoE(L) blocks it. The decode-overlap candidate set is **empty**
  (window 0.749, movable work 0).

### 4.3 The two rendezvous attacks, and the mechanism levers

* **Barrier narrowing:** 0.51 ms of counted barrier time deleted → **0.10 ms of
  wall.** Counted time before a barrier is not a lever, 3 for 3.
* **Deleting a whole rendezvous:** neutral. Per-XCD narrowing is **zero by
  measurement** — all ten at once buys 0.095 ms; GPU-max minus worst-XCD-max is
  0.000.
* All-thread barrier polling **+0.73 ms** (measured twice). Arrival tree
  neutral. Worker count 200 neutral. No legal independent-round fusion exists.

### 4.4 Tile-side levers, all retired

Narrowing MoE tiles / OPW=16 **−1.34 ms**; W2 split-K **+4.12 ms**; TopK
rank-select **+33 % on the tail**; W2 L2 prefetch 0.13 % (neutral); the router
fold 96 ns; bias prefetch in the router TopK tail (cold fetch only 205 ns);
depth-8 register scheduling neutral; FP8 activations as a blanket 4x claim
(**+12 % in a probe, −26 % at W_UV's K=512** — the probe measured a steady state
the real stage never reaches); shared-expert hoist **+0.106**; fusing the dense
prologue **+0.996**; widening a phase for bandwidth is dead **as a class**
(0.118 ms). Speculation / MTP / bs=2 are off the board entirely.

The one win in this family, kept: MoE dispatch walking dead tile space, clamped
to owned tiles, **−0.180 ms**.

---

## 5. Baseline

Pooled, shipping build, no flags, `MODEL_PATH` pinned to
`/home/claudeuser/models/glm5-mxfp4`, NP=8, bs=1, **one decode row**, 118 tokens
per run:

| source | n | ms/iter |
|---|---:|---:|
| prior board (`54e4515`) | 3 | 10.571 |
| this board, run 1 | 1 | 10.654 |
| this board, run 2 | 1 | 10.575 |
| **pooled** | **5** | **10.588** |

vs the standing **10.619** — a −0.031 ms difference, well inside the 0.26 ms
wall noise floor. **No build change was made on this board**; these are controls,
not a delta.

**Texts checked on every run** (the gate that exists because two changes on this
branch looked like clean speedups while producing garbage): *"The capital of
France is\<think\>1. Analyze the user's request… The capital of France is
Paris."* and *"and the capital of the UK is / The capital of the UK is
London."* — coherent, on topic, correct.

---

## 6. The board ends here

Five classes, all closed by measurement. The largest remaining in-scope lever is
**0.188 ms**, under the 0.26 ms noise floor. Do not open a sixth class looking
for a way to keep going, and do not reopen the aggregate on the 0.27-vs-1.0
coefficient — §2.3 is coefficient-independent by construction.

The finding is not a failure of tuning. It is that **the hardware roof for this
token is 1.731 ms and this decomposition cannot request it**, because one
persistent worker at bs=1 owns 0.4 MB of the layer and there is nothing in this
shape to raise that number.
