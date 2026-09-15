# CLOSED 09-15 — register-resident fusion BUILT, CORRECT, and still not a win

**The "registers are not an option" claim in
`gang_rmsnorm_linear_mxfp8_bias_mi300.cuh:567` was wrong, and it has now been
disproved by building the thing.** That note argued the fold's layout (thread
`t` owns `[4t,4t+4)`) cannot feed the quantizer (which wants 32 contiguous per
lane) without an LDS relayout. That is a property of *reusing*
`_gang_wave_parallel_fp8_quant_rmsnorm`, not of the data: the E8M0 group is 128
elements, so a group is exactly **32 consecutive threads**, and the group amax
is a 32-lane cross-lane max (`__shfl_xor` masks 1..16 never cross the 32-lane
half-wave boundary). Quantizing in place needs no LDS and no re-read — which is
exactly what ATOM's `fused_qk_rmsnorm_group_quant` does.

Built as `MPK_COLL_FUSE_REG` (default off). It **works and is correct**: G1
cross-rank PASS, coherent text. Triple at 16/256, one `-D` apart, `decode_min`
(the avg is unusable here — this box injected a multi-second stall that put
control's avg at 343 ms; min is immune):

| arm | flags | decode_min | Δ vs control |
|---|---|---:|---:|
| control | (none) | **9.721** | — |
| prohoist | `MPK_QKV_PRO_HOIST=1` | 9.860 | +0.139 |
| **collfuse_REG** | `+ MPK_COLL_FUSE_REG=1` | 9.934 | **+0.213** |

`n=1` per arm and every delta is inside the 0.26 ms floor, so the honest
statement is **no win detected**, with a consistently negative monotonic sign.

**Why it cannot win, and this is the transferable part.** The fusion removes
the producer's 2 KB/slice re-read plus a `buffer_inv`. That row is **L2
resident** — qkv_a's 24 tiles per XCD all read it — so the traffic it deletes
was never the queue the prologue waits in. The register form is strictly better
than the LDS form (+0.074 vs +0.30 ms, because it adds no LDS traffic and no
release), which confirms the mechanism was diagnosed right; it just reveals the
target was worth ~0. Compare the `MPK_ABL_ML_BOUNDARY` verdict (deleting the
whole per-layer boundary buys 0.051 ms) and `MPK_QUANT_V16` (0 -> 657
`global_load_dwordx4`, +0.062 ms): **memory-traffic and bookkeeping levers in
this prologue all price at zero.** What is left is occupancy and the serial
stage chain — arrival spread is 82 of 161 us/layer and ~86 of 240 workers are
busy on average.

---

# (earlier, 09-15) the LDS lever is a BUILT + MEASURED LOSS, not merely blocked

**Read this first.** The "blocked" framing below is stale. Two facts close it:

1. **The fusion was already priced and it LOSES.** The design note at
   `gang_rmsnorm_linear_mxfp8_bias_mi300.cuh:586` ("RE-PRICED AT NP=4,
   2026-08-28") measured `MPK_COLL_FUSE_NORM` at NP=4, bs=1, devices 4-7:
   OFF 9.113 vs ON mean 9.396 = **+0.30 ms LOSS**, coherent text, outside the
   0.26 ms floor. The mechanism is architectural and not a flag bug: ATOM's
   `fused_qk_rmsnorm_group_quant` keeps the reduced row **in registers** and
   normalizes in one pass; fleet's EP fold hands thread `t` elements
   `[4t,4t+4)` while the quantizer wants 32 contiguous per lane, so the layout
   must change — forcing an **LDS round-trip** that only removes the re-read,
   not the write, and adds the release the tiles then wait on. Porting ATOM's
   win needs ATOM's register-resident layout, i.e. a rewrite of the fold, not
   `MPK_COLL_FUSE_NORM`.

2. **The isolation, 09-15.** The prior campaign's `collfuse` arm never enabled
   fusion: `MPK_COLL_FUSE_NORM` was **missing from `MPK_FORWARD_VARS`**, so
   `-x` never forwarded it to the ranks — that arm silently ran as `prohoist`.
   Added it to `env_common.sh` and re-ran short-shape (16/256, as claudeuser;
   root has no mpi4py → `world_size(1)` → OOM, a launch trap, not a kernel bug):

   | arm | flags | 16/256 result |
   |---|---|---|
   | `prohoist` | `MPK_QKV_PRO_HOIST=1` | **rc=0, decode_min 9.86 ms, G1 PASS, coherent text.** The hoist is NOT fundamentally broken; its 1024/1024 hang is specific to multi-token prefill. |
   | `collfuse` | both flags | **GPU memory fault** (illegal access 700, sig 6) — a regression in the LDS-handoff path. Moot: even fixed it re-confirms the +0.30 ms loss above. |

**Verdict: do not pursue.** The fused collective joins the shared-expert
K-shard (`gang_moe_linear_mxfp8_mi300.cuh:1812`, built, +0.247 ms) as an
ATOM-inspired lever that is built and measured a LOSS on fleet's fold layout.

---

# (STALE) Fused collective is BLOCKED: its precondition `MPK_QKV_PRO_HOIST` deadlocks at 1024/1024

**Measured 09-15.** The ported fused collective (`MPK_COLL_FUSE_NORM`, the
redline/ATOM "reduce + RMSNorm + FP8-quant in one LDS pass, no HBM round-trip"
lever) cannot be measured, because the producer-hoist it rides on
(`MPK_QKV_PRO_HOIST=1`) **hangs the 1024/1024 megakernel**. The lever itself is
built and its flag-off path is proven inert; the blocker is upstream.

## What was run

3-arm A/B in an isolated tree forked from the Sep-8 container baseline
(`/mnt/nvme1/exp/coll_fuse`), devices 4-7, 1024/1024, `MPK_COLL_FUSE_NORM`
wired and confirmed loaded from the tree (`has COLL_FUSE: True`).

| arm | flags | result |
|---|---|---|
| `control`  | (none) | **rc=0, decode_min 12.729 ms**, G1 PASS, G2 0.277 — matches the canonical 12.642–12.788 min range. Tree + harness are sound. |
| `prohoist`  | `MPK_QKV_PRO_HOIST=1` | **DEADLOCK.** attempt 1 timed out at the 900 s watchdog (rc=124); attempt 2 spun 1000 s at GPU 100 %, `[ITER] n=1`, no completion. |
| `collfuse` | `MPK_QKV_PRO_HOIST=1 MPK_COLL_FUSE_NORM=1` | not reached — would deadlock identically (rides the same hoist). |

Control finishes the whole prefill+decode in ~5 min of kernel time in the same
tree, so the hang is **hoist-specific**, not the tree, the fork, or the disk.

## It is NOT the fused-collective edit (H1, proven statically)

The `prohoist` arm compiles with `-DMPK_QKV_PRO_HOIST` but **without**
`-DMPK_COLL_FUSE_NORM`, so it runs the flag-off path of the edited kernel.
Resolving `MPK_COLL_FUSE_NORM` as undefined in
`gang_rmsnorm_linear_mxfp8_bias_mi300.cuh` and diffing against the pristine
Sep-8 file yields **only a comment block, a behaviorally-inert
`#define MPK_COLL_FUSE_NORM 0`, and one blank line** — zero executable
difference. So `prohoist` deadlocks on essentially pristine hoist code. The
deadlock predates this campaign.

## Where it hangs

The hoist adds an XCD-local counter rendezvous in the qkv_a phase
(`gang_mla_attn_fused_mi300.cuh:624-652`): all `tiles_per_xcd` workgroups
`atom_add`-increment `_cnt` in `wuk_barrier`'s spare lanes, the last arriver
stamps `_flag = qkv_expected`, and everyone spins `while (_flag < qkv_expected)`
with an `s_sleep(1)`. The observed hang — GPUs pegged, `[ITER]` stuck, no log
growth — is that spin never releasing, i.e. the last-arriver condition
`(prev % tiles_per_xcd) == tiles_per_xcd - 1` never fires and the self-heal
`_cnt >= tiles_per_xcd * qkv_expected` never trips.

The code comment on the flag says as much: it *"rides `MPK_QKV_PRO_HOIST`, whose
header already records that a rank which misses that flag disagrees on the
XCD-local release's writer set and deadlocks rather than erroring."* Here all
four ranks got the flag, so it is not rank-inconsistency — it is the in-kernel
writer/arrival accounting at this shape.

**Lead (not yet proven):** the hoist publish loops
`for (tok = 0; tok < num_active_tokens; tok++)` writing `pro_pub + tok*PUB_TOK`,
while `pro_pub`/`PUB_STRIDE` are sized for `BATCH_SIZE` tokens
(`:580-582`). `num_active_tokens` is `qo_indptr[MAX_BATCHED_REQUESTS]` — the
prompt length during **prefill** (up to 1024), vs `BATCH_SIZE`=1 for latency
decode. If `num_active_tokens > BATCH_SIZE` the publish overruns its per-XCD
block. This fits "MEASURED NEUTRAL" (`:529`): the hoist once completed, i.e. it
was validated only at decode/short shapes where `num_active_tokens ≤ BATCH_SIZE`
— never at 1024-token prefill. Pinning the exact cause needs kernel
instrumentation; the buffer/loop mismatch is the first place to look.

## Verdict

The fused collective is the one **un-ported compute lever** with headroom
(~0.35 ms/token projected, regime C, above the 0.26 ms floor). It is
architecturally welded to `_rnlm8_pro_publish`, which "runs only under
`MPK_QKV_PRO_HOIST=1`" (persistent_kernel.py:849). So landing it requires first
making the producer-hoist survive a 1024-token prefill — a deep barrier/buffer
fix on never-validated experimental code, for a modest projected gain.

Shipping build is untouched: `MPK_COLL_FUSE_NORM`, `MPK_QKV_PRO_HOIST`, and
`MPK_OPROJ_KSPLIT_CEIL` all default OFF. Isolated tree kept at
`/mnt/nvme1/exp/coll_fuse` (208 MB) for reuse if the hoist is fixed.
