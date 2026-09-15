# o_proj K-split (N-split -> K-split): measured NULL, not built

**Verdict: revert-and-record-null.** Do not build the full K-split. The two
rendezvous it would delete are worth **less than the 0.26 ms wall noise floor**,
measured by ceiling probe and independently confirmed by two ablations the repo
already had.

The prediction was registered before the numbers landed, in
`OPROJ_KSPLIT_PREREGISTERED.md`. It said NULL, ~0.03 ms. Measured -0.169 ms
against a 0.26 ms floor. Right call, roughly right magnitude.

---

## 1. What was built

`MPK_OPROJ_KSPLIT_CEIL`, a **ceiling probe**, not the K-split itself. It deletes
the two waits the K-split would make unnecessary, leaving everything else alone:

| level | deletes | file |
|---|---|---|
| `>=1` | `wuv_barrier` local-flag wait | `gang_oproj_router_fused_mi300.cuh:773` |
| `>=2` | `attn_release` Phase-8 wait (+ level 1) | `gang_mla_full_layer_fused_mi300.cuh:2062` |

**Wrong output by construction** — o_proj reads stale peer slices. That is the
point: it prices the *maximum* a correct K-split could win, at a fraction of the
cost of writing one. A correct implementation cannot beat its own ceiling.

Rationale for probing before building: the K-split is a large rewrite
(row-sharding the contraction, re-deriving the reduction into `hier_barrier`,
re-tiling `OPROJ_TP_COLS` and its `static_assert`). If the ceiling is under the
noise floor, the rewrite cannot pay, and the probe answers that in two runs.

## 2. Gate 1 — ISA: the rendezvous really disappeared. PASS

Per-function diff of the gfx950 code object (`demo/glm5/isa_func_diff.py`).
Both deltas land in `gang_mla_full_layer_fused_kernel_mi300`, the one function
that inlines both rendezvous:

| arm | `s_sleep` | instructions | `global_load_dword` |
|---|---|---|---|
| ceil1 (1 wait deleted) | 34 -> 32 (**-2**) | -236 | -8 |
| ceil2 (2 waits deleted) | 34 -> 30 (**-4**) | -402 | -14 |

Two `s_sleep` sites per deleted wait (peeled first iteration + rotated loop),
and the vanished `global_load_dword`s are the poll's `ld_nt_s32`. The counts
scale exactly with the number of waits removed. The change is in the binary,
not just the source.

**Default OFF is byte-identical.** Built the pristine tree and the patched tree
with the flag unset; both code objects hash `f61fa8a35ca3e692cd81b11279aaa0c9`.
Reverting the patch also reproduces `git HEAD` exactly (`git diff --stat` empty).
The shipping build is provably untouched.

**Flag forwarding verified in the live command line**, not just in
`env_common.sh`: `-x MPK_OPROJ_KSPLIT_CEIL` appears in the emitted `mpirun`.
A compile-time flag missing there deadlocks rather than erroring.

## 3. Gate 3 — latency A/B, 1024/1024, NP=4, devices 4-7

Alternating pairs, `decode_min_ms` (per-iteration minimum is the reported
statistic on this box; see the straggler note below).

| pair | control | ceil1 | delta |
|---|---|---|---|
| 1 | 12.707 | 12.532 | **-0.175** |
| 2 | *hang, rc=124* | 12.523 | — |
| 3 | 12.685 | 12.522 | **-0.163** |
| **mean (n=2)** | | | **-0.169** |

**-0.169 ms against a 0.26 ms noise floor. NULL.** Not claimable.

Control is highly reproducible — `12.766 / 12.710 / 12.707 / 12.685` across four
independent runs spanning three sessions and a container restart, and
`decode_avg` 13.983 / 13.952, consistent with the stated 14.0 baseline.

`decode_avg` is **unusable** on this box: single straggler iterations of
18336 ms, 3519 ms and (in an earlier session) 266 s land inside otherwise normal
runs, which is why the min-iter statistic is the one reported.

### The -0.169 ms is an upper bound, and even it is contaminated

Two reasons not to read it as a speedup:

1. **The text is garbage** (section 4), so this is an ablation number.
2. Garbage routing changes which MoE experts fire, which changes MoE work in
   *either* direction. So -0.169 is not even a clean "same work minus a
   barrier" measurement.

Both caveats push the same way: the honest read is "no effect resolvable above
noise", not "-0.169 ms of headroom".

## 4. Correctness — probe FAILS, exactly as designed

Never quote a latency number without reading the text. Tails of the 1024-token
continuations:

| arm | generated tail |
|---|---|
| control r1 | `...noted that Ben Whishaw and Robert Boulter "offer tenderness amid the savagery" in the play Mercury Fur?` |
| control r3 | `...6. **Identify the core entities**: 7. **Identify the core entities**:` |
| **ceil1 r1** | `,,00   topf .-top,-th,8Am_hist 3  , ,,-,12\. 00 db-t,,    30073,43 lost, on3,.J` |
| **ceil1 r3** | `- . healing healing: : ** . - . healing  - - healing of  healing..:` |

Control degenerates into repetition — normal for a 1024-token greedy
`--ignore-eos` continuation — but stays grammatical English. **ceil1 is token
soup.** Token diversity confirms it: 0.096 / 0.093 / 0.071 for ceil1 vs
0.169 / 0.335 for control.

`ppl.sh` and `run_longseq.sh` were **not** run. They gate correctness before a
latency *claim*; there is no claim here, and running perplexity on an arm that
is wrong by construction would only re-measure the garbage above. The shipping
build needs no re-gate because flag-off is md5-identical to pristine (section 2).

**G1 cross-rank PASSED on every run including the garbage ones** — a useful
reminder that G1 proves the four ranks agree, not that they are right.

## 5. Blocker: ceil2 cannot be measured at 1024/1024

The full ceiling (both rendezvous deleted) **hangs deterministically** at the
required shape:

- 2 of 2 attempts killed at `STALL_SECS=1200`, `rc=124`.
- Never emitted a single `[ITER_TIME] ... DONE`, i.e. it dies **before finishing
  iteration 0**. Control on the same box completes iter 0 and reports normally.
- Runs fine at the 32/16 smoke shape, which is how it passed the liveness check.

Mechanism: deleting the `attn_release` wait lets o_proj read unwritten `v_out`,
which corrupts the **router**. Garbage expert selection makes workers wait on
`ep_signal` lines that no peer will ever write, and the surviving `hier_barrier`
never fires. At 32 tokens the corrupted routing still happens to hit live
experts; over 76 layers x 1024 tokens the probability of a fatal selection
approaches 1.

An earlier session mis-attributed this hang to a too-tight watchdog. That was
half right — `STALL_SECS=400` *was* killing healthy control runs, whose silent
prefill takes ~293 s — but at 1200 s the control completes and ceil2 still
hangs. Both bugs were real and independent.

This is why the table above is control-vs-**ceil1**. Section 6 covers the
`attn_release` half by other means.

## 6. The premise was calibrated on the wrong statistic (33.93 vs 2.102 us/layer)

The brief expected 2.58 ms at 1:1, ~0.77 ms at the 0.3 busy->wall coefficient,
from a profiled **33.93 us/layer** (32.50 us of it measured spin). That is a
**mean-worker occupancy** figure. The wall responds to **max arrival**:

| statistic | value | measures |
|---|---|---|
| ATOM-comparison profile | 33.93 us/lyr | worker-time burned spinning |
| `price_attention_shard.py` row 23 | 2.102 us/lyr | critical-path contribution |

Both are correct. They differ ~16x because of the arrival distribution already
tabulated in `MAKESPAN_RULE.md`:

| barrier | pop | max | min | spread | plateau within 1 us of max |
|---|---|---|---|---|---|
| attn_release | 232 | 73.25 | 47.93 | 25.32 | **128** |
| wuv_barrier | 232 | 81.02 | 80.47 | 0.55 | **232** |

At `attn_release`, **128 of 232 workers arrive within 1 us of the max**. The
barrier fires at the max, and the max is a *plateau, not a straggler*. The ~104
early workers really do spin ~25 us each — that is the 32.50 us the profile
counted — but releasing them cannot make the 128 plateau workers arrive sooner.
They are freed only to spin at `hier_barrier` (o_proj:925), which survives any
o_proj layout and was never a target. **The spin is relocated, not removed.**

At `wuv_barrier` it is starker: spread 0.55 us, plateau 232/232. There is no
waiting there to remove.

ATOM genuinely pays 0.00 us here, but it pays 0 because it has no idle workers
in that state to count — not because it removed critical path. The 3.45 ms gap
is real; **this line item is not 33.93 us/layer of it.**

### Regime classification

Deleting a rendezvous *without* deleting the work in front of it is **regime A**,
not regime B. Regime B ("the segment collapses") requires removing the phase and
its rendezvous; here W_UV and the attention tail still compute. For the max
arriver the wait is ~0 by definition — it *is* the max, it never waits — so its
arrival at the surviving `hier_barrier` is unchanged. `MAKESPAN_RULE.md` already
carries the general row: *"delete one whole rendezvous | A | no work removed |
NULL | 0.003 ms"*.

## 7. Both target rendezvous were already priced at ~0

The brief read this row as UNPRICED. Only the *combination* was; each half had
already been ablated:

- **`attn_release`** — per-XCD narrowing replay of the measured arrival log:
  GPU-wide max minus worst-XCD max = **0.000 us**. Recorded `resolved=True,
  delta=0.000` in `price_attention_shard.py`.
- **`wuv_barrier`** — `MPK_WUV_IN_MERGE` (e5d1ff5) already **deletes this
  rendezvous outright and correctly**, ten rendezvous become nine, correctness
  gated 4/4, measured **-0.003 ms** (n=2 paired), -0.019 on min-iter n=5.

That last one is the strongest single result here: a *correct* deletion of
`wuv_barrier` exists, passes correctness, and measures -0.003 ms. This probe's
-0.169 ms garbage-arm number is an upper bound consistent with it.

## 8. Verdict

**Revert and record the null.** The probe stays in-tree behind
`MPK_OPROJ_KSPLIT_CEIL`, default OFF and md5-proven byte-identical when off, as
the recorded evidence.

The makespan rule extends to rendezvous **deletion**, which is what this task set
out to test. That is the finding. It is now 4-for-4 predicting nulls, and the
plateau data explains *why*: barriers whose arrival distribution is a wide
plateau cannot be improved by releasing early arrivers, no matter how much spin
those arrivers are counted as burning.

**Where the 3.45 ms gap is not:** the attention->o_proj rendezvous. Anyone
re-deriving it from a mean-worker profile will get 33.93 us/layer and be wrong
by ~16x. Future gap-closing work should target phases where the arrival
distribution has a genuine tail, or follow the one lever that did move
(`MPK_MLA_SKIP_DECODE`, regime B, -0.61 ms) — remove *work*, not waits.
