# Fleet on SPX + NPS2 (MI355X, gfx950)

Status of getting `fleet-chiplet-megakernel` to run correctly and at NPS1 speed
in SPX+NPS2. Every number here is from `mi355x-thor-2`, `HIP_DEV=8`
(`0000:e5:00.0`), gpt-oss-120b, batch 1, `MAX_SEQ_LENGTH=128 MAX_NEW_TOKENS=16`.

## Headline

| | ms/token |
|---|---|
| SPX+NPS1 reference (instrumented / plain) | 1.70 / 1.52 |
| SPX+NPS2 baseline | 8.482 |
| SPX+NPS2 with everything below | **3.422** |

2.48x recovered, ~2.0x still missing. The remaining gap has a single identified
cause (see *The AID0 problem*), and it is **not** fixed by anything in this
branch.

## 1. The hang was already fixed, and shipped disabled

`MPK_TERM_RECHECK` defaults to `0`. Turn it on for any NPS2 work.

Without it, SPX+NPS2 hangs about two runs in three. Commit `6bf91f5` fixed this
and recorded "8/8 clean vs 2/3 hung without the fix", but left the flag
opt-in. Independently reproduced here: unmodified NPS2 hung **2 of 3** trials;
with the flag on, **0 of 3** across every arm since.

Mechanism, from that commit: the terminate path sets `precomp_terminate` then
bumps `precomp_iter_ready` to release workers parked at the iteration gate. That
gate only tests terminate *inside* its wait loop, so a worker released by the
bump can leave without seeing the flag, then block forever on an iteration that
is never produced.

This matters beyond the hang: until it is on, two thirds of runs are wasted and
every A/B is noise. Several conclusions in the early part of this investigation
were wrong because of it.

## 2. Memory types in SPX+NPS2

gfx950 PTE encoding is `NC=0, RW=1, CC=2, UC=3`. The loaded per-BO-flag driver
announces its choice in dmesg (`PERBO: AID_LOCAL BO is local -> mtype_local=1`,
`FLAGMTYPE fired: knob=2 -> mtype=2`).

| allocation | MTYPE | behaviour |
|---|---|---|
| plain `hipMalloc` (spans both ranges) | **NC** | cacheable, **not** coherent across XCDs. A poll re-reads its own stale L2 line until capacity evicts it. |
| `AID_LOCAL` | **RW** | coherent **within one memory partition only**. No directory, so it scales. |
| `AID_LOCAL \| COHERENT` | **CC** | coherent everywhere, but backed by DF-CS shadow tags (~8 L2 lines/channel). Overflow loses probes and **hangs**. |
| `AID_LOCAL \| UNCACHED` | **UC** | always correct, always slowest. |

Measured on the flag replicas, identical builds:

| MTYPE | ms/token | note |
|---|---|---|
| NC (no replicas) | 8.535 | |
| UC | 7.037 | correct, uncached |
| CC | 4.11 | fastest but **unusable** — see below |
| **RW** | **4.34** | the right default |

**CC must not be the default.** Same build, three runs: one clean, one clean
with a *different* output text, one hard deadlock before any task retired. RW
costs 0.22 ms and has no directory, so it scales to every flag family.

`aid_local_flag_mtype` is live in the loaded module but is **dead code** in the
`aid-local-hbm/driver/overlay` tree — reading that source alone is misleading.

## 3. What worked: AID-split the polled flag lines

Four families of per-XCD release flags now live in two RW `AID_LOCAL` replicas,
one homed per AID. A publisher writes both; every XCD polls only the copy in its
own AID, so no reader is ever outside its copy's coherence domain. Sound because
a remote **writer** still invalidates sharers co-located with the line; only a
remote **reader** goes unprobed.

Three trials per arm, fresh container each, `MPK_TERM_RECHECK=1`, all 3/3 clean:

| stage | runs (ms) | mean |
|---|---|---|
| baseline | 9.957 / 7.413 / 8.077 | 8.482 |
| `+ attn_release, layer_release` | 3.835 / 4.042 / 4.155 | 4.011 |
| `+ hier_barrier` | 3.836 / 3.637 / 3.810 | 3.761 |
| `+ moe_fused_barrier` | 3.562 / 3.657 / 3.744 | 3.654 |
| `+ MPK_MOE_XCD_PAIR=0` | 3.418 / 3.426 | **3.422** |

Only *polled flag lines* move. Arrival counters and `topk_counter` stay shared:
they are device-scope atomics, which serialize at the **XCD** L2 boundary
(eight of them), not the AID boundary (two), so placement cannot reach them.

`MPK_MOE_XCD_PAIR=0` is an NPS2-specific inversion. On NPS1 that flag is a win
(`9d478ea`: "Decode 1.70 -> 1.64 ms"). In NPS2 it *hurts*: pinning expert slot
`s` to XCD pair `s` makes the AID skew systematic instead of averaging out.

Per-phase result: slot 5 (`attn_release_wait`) reaches NPS1 parity at **1.09x**
(1041 -> 1122 ns/layer), and compute slots are at 1.02-1.04x throughout. The
mechanism works; there just is not much left for it to fix.

## 4. The AID0 problem — the whole remaining gap

**A plain `hipMalloc` fills AID0 first, so the model lands almost entirely in one
memory partition.** From `MPK_AID_LOCAL_MAP=1`:

```
[MAP] slot=17 name=gate_up_weight kind=shared alloc_mib=2260.00 aid=0 lo_ns=335.5 hi_ns=433.4 skew_ns=-97.9
[MAP] slot=18 name=down_weight    kind=shared alloc_mib=2260.00 aid=0 lo_ns=333.5 hi_ns=433.3 skew_ns=-99.8
[MAP] slot=19 name=w13_bias       kind=shared alloc_mib=1014.00 aid=0
[MAP] slot=4  name=qkv_weight     kind=sliced alloc_mib=32.00   aid=0
[MAP] slot=9  name=oproj_weight   kind=sliced alloc_mib=64.00   aid=0
```

17 of 20 slots report `aid=0`. `lo_ns` is XCDs 0-3, `hi_ns` is XCDs 4-7, so
**XCDs 4-7 pay ~98 ns more on every MoE weight access, permanently** — ~29%
worse latency on the two largest buffers in the model.

The source is `amdgpu_object.c`: with `aid_local_xcp_span=0` a spanning XCP
falls through to `places[c].fpfn = mem_partitions[mem_id].range.fpfn`, and XCP
0's home range is range 0 = AID0.

**Why it costs so much: a megakernel layer costs `max` over XCDs, not `mean`.**
`MPK_DRAIN_STATS` on the layer boundary:

```
[DRAIN] n=200000 drain=655 sync=30 arrive=1223 spin=8843 tot=10753
```

82% spin. And `[XCDSPIN]` shows it is not random:

```
[XCDSPIN] 13695 11333 11697 8782 | 6110 7351 7073 4676
   last:      0     3     1    20 | 2338   55   15   69
```

All of AID1 is slower than all of AID0 (low spin = arrives late), and **XCD4 is
last to arrive 2338 of ~2500 times (94%)**. The per-layer barrier converts that
fixed asymmetry into critical-path latency 36 times per token.

For completeness, the rest of the layer boundary is cheap — `MPK_INTERLAYER_SPLIT`:

```
[ILSPLIT] n=410688 fnpre_ns=8 binv_ns=1 post_ns=1880 total_ns=1890
```

TaskDesc setup 8 ns, `buffer_inv` 1 ns. The cost is entirely the barrier wait.

### This explains every failed placement experiment

Relocating any *one* buffer does nothing while GiBs of MoE weights stay in AID0:

| experiment | result vs its baseline |
|---|---|
| O-proj weight slicing (slot 9) | 4.267 vs 4.114 |
| KV cache + QKV weight (slots 1,2,4) | 3.474 vs 3.422 |
| ml pointer tables replicated per AID | 3.423 vs 3.422 |
| MoE weights, full replication | **15.556** vs 3.650 |
| MoE weights, `SINGLE_REPLICA` | 3.623 vs 3.422 |

Full MoE replication is catastrophic and the reason is worth recording: 65
unique segments, 78.68 GiB per AID, and the header's per-range fill counter
reads a comfortable **55%** — because it only counts its own allocations. The
original spanning torch allocation stays resident, so real footprint reaches
~236 GiB of 288 GiB, TTM evicts, and weights return over PCIe. That is exactly
the *"filling a range is the failure mode that looks like success"* case, and
the reassuring percentage is the number that hides it.

`SINGLE_REPLICA` fits and **does** reduce the skew — XCD4's share of
last-arrivals falls from 94% to 41% — confirming the diagnosis. It still loses
overall, because +79.86 GiB of footprint costs as much as the locality saves.

### Why locality does not pay for weights at batch 1

The MoE read is a GEMV whose loads are issued far ahead and retired under one
`s_waitcnt` — one pipelined memory latency with no queue for locality to
shorten. `phase7-bench` reached this independently with `--dupdata` ("read is
0.08 us whether the slice data is local, remote or replicated"), and
`aid/README.md`'s ~1.05x estimate for fleet was right.

### The fix, and why it is blocked

The right fix is to **balance the original allocation**, not add a copy:
round-robin the backing slabs across both ranges so each XCD is ~50% local and
neither AID group is systematically behind — zero extra footprint.

`aid_local_xcp_span=1` looks like exactly that (it gives a spanning XCP the
whole span as its aperture). **It does not work: it is incompatible with
SPX+NPS2 in this driver build.** Loaded with it, all 8 dies came up DPX and
every SPX switch returned EBUSY. The span path and the `aid_local_spx_nps2`
SPX-at-load hunk both key off `num_xcp_per_mem_partition` and conflict.

Remaining option: an LD_PRELOAD allocator that alternates AID per slab.
`aid_interpose.cpp` is the starting point but only supports a single fixed AID,
and it grinds during model load (unusable as-is — the same problem killed the
all-UC experiment).

## 5. Correctness — still open

SPX+NPS2 produces **different generated text run to run**; SPX+NPS1 is identical
across three runs. This predates any change here.

```
NPS1 (x3)      The user asks: "The capital of France is". Likely
NPS2 baseline  We need to answer: "The capital of France". That's a
```

Decoding is greedy argmax with no sampling and no seed, so this is real compute
divergence, not sampling.

`MPK_NPS2_L2_ACQUIRE=1` does **not** fix it. A single run appeared to restore
NPS1's phrasing, but over three trials it varies exactly as much as without, and
it costs 0.9-2.2 ms. That single-run reading was an artifact.

## 6. Things measured and rejected

| knob | result |
|---|---|
| `MPK_NPS2_L2_ACQUIRE=1` | 10.68 vs 8.482 — costs 2.2 ms, fixes nothing |
| `MPK_NPS2_EVENT_POLL=1` | 3.404 vs 3.422 — inside noise |
| busy-poll on all spin loops | 3.609 vs 3.650 — inside noise |
| `MPK_MOE_XCD_STRIPE_LAYER=1` | 3.459 vs 3.422 |
| `MPK_W2_CONSUMER_GATE=0` | 3.665 vs 3.422 |
| `MPK_LEAN_ARRIVE=0` | 3.375 vs 3.422 — inside noise |
| all buffers UC (LD_PRELOAD) | unrunnable — 15 min at 100% GPU without reaching megakernel compile |

## 7. Reproducing

```bash
docker start fleet_v1            # and restart it before EVERY measured run
ENVS="MPK_TERM_RECHECK=1 MPK_AID_SPLIT_FLAGS=1 MPK_MOE_XCD_PAIR=0" \
  TIMEOUT=600 bash ~/nps1/fleet_run3.sh <label>
```

Diagnostics, all opt-in:

| env | what it gives |
|---|---|
| `MPK_PHASE_SLOTS=1 MPK_PHASE_START_ITER=40` | per-phase ns/layer; parse with `tests/ci-tests/summarize_phase_slots.py`. Start-iter defaults to 600, so it **must** be lowered or the recorder never arms |
| `MPK_DRAIN_STATS=1` | `[DRAIN]`, `[XCDSPIN]`, `[MLPRO]` — layer-boundary decomposition and per-XCD straggler counts |
| `MPK_INTERLAYER_SPLIT=1` | `[ILSPLIT]` — needs `MPK_PHASE_SLOTS=1` |
| `MPK_AID_LOCAL_MAP=1 MPK_AID_LOCAL_DRY_RUN=1` | `[MAP]` — home AID, size and per-XCD-group latency of every input slot |
| `MPK_AID_SPLIT_FLAGS_MTYPE=rw\|cc\|uc` | memory type of the flag replicas |

### Measurement hygiene

**Restart the container before every measured run, and never conclude from a
single run.** The box drifts into a state where *everything* hangs after a few
killed runs — four consecutive hangs including the flags-off control, cleared by
`docker restart fleet_v1`, with no GPU fault in dmesg. Run-to-run variance is
also large: the same flags-off config measured 8.54 and 10.24 ms. Per-phase
`[PSLOTW]` deltas are more trustworthy than wall-clock totals.

## 8. Open items

1. **Balanced allocator** — the only known path to the remaining 2x.
2. **Output nondeterminism** — NPS2 disagrees with NPS1; unexplained.
3. **`MPK_MOE_XCD_PAIR` default** — should be off in NPS2, on in NPS1. Currently
   a hand-set env var.
4. Dies `65/75/85/95` were left in DPX by a driver reload during the
   `aid_local_xcp_span` experiment and return EBUSY on SPX. `05/15` were already
   wedged beforehand. `e5`/`f5` are SPX/NPS2.
