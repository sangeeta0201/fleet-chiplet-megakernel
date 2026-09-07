# Phase 7 correctness gate

Whether the AID-split sync fixes still order Phase 7 correctly. Measured on
mi355x-thor-2, SPX + NPS2, patched driver with per-BO `aid_local_flag_mtype`.

## Why the first attempt proved nothing

The first version of this gate compared FNV hashes of `attn_proj_out` and
`rmsnorm_out` across configurations. Every hash matched, and the gate was
worthless: every output was identically zero in every configuration, so a
barrier that released early read the wrong layer's data and still produced
zero. Three independent causes, all in the harness rather than the kernel:

1. **No weights.** The K-parallel arm takes its MFMA B operand from LDS at
   `oproj_lds_w_off()`, which the fused caller fills during Phase 6.
   `drive_phase7` invokes Phase 7 standalone and never staged anything, so the
   GEMM multiplied by an unpopulated tile.
2. **Underflowed scales.** The whole weight buffer was `memset` to `0x11`,
   including each group's E8M0 scale half, where `0x11` is 2^-110. Every one of
   the 4096 products underflowed.
3. **A flat row.** Even with 1 and 2 fixed, every weight byte is equal, so all
   184 blocks produce identical output columns. RMSNorm divides a flat row by
   its own RMS and returns exactly 1.0 in every column *regardless of
   magnitude*, which erases the layer dependence the check needs.

`patch_lds_stage.py` fixes all three: it stages the tile from HBM into the LDS
region the kernel reads, fills the data and scale halves separately (`0x22` =
a pair of e2m1 ones, `0x77` = 2^-8), and gives each column its own
layer-dependent residual.

Scale `0x77` is not arbitrary. It puts a 4096-long reduction of ones near 20
rather than 4096, where bf16 resolves 0.125; one XCD's slice going stale moves
the result by 1.0, so the error survives the store instead of rounding away.
The staged arithmetic checks out by hand: `attn_proj_out[0..1]` reads
`419c 41a0` = 19.5 and 20.0, matching 2 x sum(A) plus the column's residual.

## What the barriers guard, and where each failure lands

The two are separable, which is what makes the controls meaningful:

| barrier | guards | a violation shows up in |
| --- | --- | --- |
| `attn_slice_release` | the reduction's own input: `out[col]` sums all 4096 K elements, i.e. all eight XCDs' `attn_out` slices | `attn_proj_out`, and `rmsnorm_out` downstream of it |
| hierarchical release | the row read: RMSNorm reads all 2880 columns, spanning all eight XCDs' blocks | `rmsnorm_out` only |

## Results

400-layer bucket run and 64-layer hash runs, `--skew=20000` where noted.
Correct hashes throughout: `attn_proj_out=d5f0c50124e2cb83`,
`rmsnorm_out=ff1bdbe7ad2c4ed3`.

**1. Sensitivity** — the output has to depend on the layer, or reading the
previous layer's data would hash the same:

| layers | attn_proj_out | rmsnorm_out |
| --- | --- | --- |
| 16 | `d5f0c501...` | `ff1bdbe7...` |
| 17 | `4718710a...` | `54815489...` |
| 18 | `288f320a...` | `a83485d2...` |
| 19 | `27dc14b6...` | `7096fc17...` |

**2. Determinism** — 6/6 identical.

**3. Teeth A**, `-DMPK_OPROJ_SKIP_HIER_POLL`, skew 20000: `attn_proj_out`
unchanged, `rmsnorm_out` wrong in 6/6 and unstable across reps
(`cbc365b4...` x5, `e3d911b9...` x1). Corruption confined to the buffer that
barrier guards.

**3b. Teeth B**, `-DMPK_OPROJ_SKIP_SLICE_POLL`, skew 20000: `attn_proj_out`
wrong in 6/6 (`e5a5ef04...`), `rmsnorm_out` wrong in 6/6 (`d7e3e025...`).
Deterministically wrong rather than unstable — a 20000-tick skew reliably puts
every wave a layer behind.

**3c. Contrast** — the correct build under the same skew: 3/3 correct.

**4. The comparison** — three sound configurations, 6 reps each, 18/18
bit-identical to each other and to the no-skew value:

| | configuration |
| --- | --- |
| A | `--aid=0` — single copy in normal HBM, the stock path |
| B | `--aid=1 --coherent=0 --split=1` — AID-local, non-coherent MTYPE |
| C | `--aid=1 --coherent=1 --split=1` — AID-local, coherent, per-AID replicas |

**5. The co-location rule, violated on purpose** —
`--aid=1 --coherent=1 --split=0` homes the release flags in AID0 under a
coherent MTYPE and has all eight XCDs poll them. It hangs: the four XCDs
outside that partition spin on a stale line forever, killed at 45 s (rc 137).
This is the rule in README.md failing exactly as documented, and it is why
`split=0` does not belong in the comparison above.

## The gate build is the measured build

Wall clock reads ~900-1000 us per layer against a ~10 us kernel, which looks
wrong until you read the header's own note on `MPK_OPROJ_INNER_TIMING`: its
per-XCD-per-layer printf costs ~600 us a layer. That flag is in the model flag
set too, and it sits outside the `t0..t4` window, as does the harness's weight
staging. Confirmed independently: a build staging only on layer 1
(`-DMPK_DRV_STAGE_ONCE`) still costs ~1000 us a layer, so the staging is not
the cost.

Inner buckets on the gate build, 2240 samples, against the medians from the
pre-gate sweep:

| bucket | gate p50 | ref p50 |
| --- | --- | --- |
| slicewait | 0.40 | 0.32 |
| mfma | 3.96 | 4.08 |
| bar | 2.56 | 2.36 |
| rmsnorm_router | 1.64 | 1.56 |
| topk | 2.32 | 1.82 |
| total | 10.66 | 10.18 |

Every bucket within ~0.5 us; `topk` is the noisy one by nature (p10 0.48, p90
6.04) because it folds in the TopK barrier wait. So correctness was checked in
the same timing regime the latency claim came from.

## Two things this does not establish

- **Per-layer staging is mandatory, and that is a finding.** Hoisting the copy
  out of the layer loop produced a *different hash on every run*
  (`5a8be9a0...`, `605cd22c...`, `87cdfe48...`). Something in the kernel writes
  into the LDS region at or above `oproj_lds_w_off()` between layers, so a
  caller has to re-stage every layer — consistent with the real Phase 6 issuing
  its DMA every layer.
- **The harness still does not reproduce the O-proj weight-traffic
  regression.** Staging happens in the harness, outside `t0..t4`, so the `mfma`
  bucket measures the LDS read and not the global weight traffic — 3.96 us here
  against 4.08 before staging, i.e. unchanged by having real weights present.
  The model's `mfma` going 1.56 -> 5.92 is still un-reproduced and still open.

## Reproducing

```bash
python3 patch_lds_stage.py     # stage weights, fix scales, vary residual
python3 patch_lds_fast.py      # dwordx4 copy instead of a byte memcpy
python3 patch_poll_skip.py     # -DMPK_OPROJ_SKIP_HIER_POLL
python3 patch_slice_skip.py    # -DMPK_OPROJ_SKIP_SLICE_POLL
bash rebuild_all.sh            # the three binaries, from one source state
bash gate_all.sh               # sections 1 through 4
bash gate_buckets.sh           # inner buckets vs the reference
bash killhang.sh               # after section 5; `pkill -f` kills its own wrapper
```
