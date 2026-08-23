# THE TILE CLASS IS BOUND BY NEITHER UNIT

The guide ruling of 2026-08-23 asked one question about the largest line in the
budget: the tile class is 69.6 us/layer against a 22.5 us width-corrected byte
roof, 3.09x — **what kind of bound is that?** Answered once, for the whole
class, with counters rather than the wall.

Reproduce: `python3 demo/glm5/tile_class_bound.py [counter_dir]`.

---

## 0. The answer

Neither the MFMA pipe nor the HBM controller is anywhere near saturation while
the tile class runs.

| resource | utilisation over the tile class | how it was obtained |
|---|---|---|
| **MFMA issue occupancy** | **1.72 %** of all 1024 SIMDs; **2.65 %** of each phase's live SIMDs | exact instruction geometry x the repo's own `MFMA_CYC = 32` |
| **achieved HBM rate** | **1.691 TB/s = 32.7 %** of the measured 5.17 TB/s peak, 32.4 % of the width-corrected achievable | modelled bytes / measured tile time |
| **HBM controller busy, whole wall** | **5.2 %** measured at 10 ms (11.6 – 14.5 % at 1 Hz), vs **16.3 %** predicted by the same byte model | `mem_busy_percent` sampled during a shipping run |

That is not the ruling's case (A) — HBM is not near roof — and it is not case
(B) — the achieved rate is not being eaten by a dequant or LDS mechanism that
would show up as *extra* traffic, because the measured controller busy is at or
**below** what the modelled bytes alone predict. It is the third case: **both
units idle, the class is waiting on dependent-issue and memory latency.**

---

## 1. The class as one item

Per layer, per rank, bs=1, NP=8. Byte model, phase widths and measured busy
times are `tile_roof_by_phase.py`'s, unchanged, so the 3.09x reproduces exactly.

| phase | Melem | MB | workers | roof us | meas us | MFMA kcyc |
|---|---:|---:|---:|---:|---:|---:|
| qkv_a | 16.12 | 16.63 | 168 | 3.33 | 10.21 | 251.9 |
| q_b | 4.19 | 4.33 | 128 | 0.91 | 7.31 | 65.5 |
| W_UK | 0.79 | 0.81 | 128 | 0.17 | 1.57 | **0** |
| W_UV | 1.05 | 1.08 | 128 | 0.23 | 1.57 | **0** |
| decode | — | 0.07 | 16 | 0.10 | 7.95 | 213.0 |
| merge / Ph8 | — | 0.00 | 128 | 0.00 | 2.56 | **0** |
| o_proj | 12.58 | 12.98 | 192 | 2.52 | 6.27 | **0** |
| router | 1.57 | 1.62 | 128 | 0.34 | 5.87 | 49.2 |
| W13 | 100.66 | 53.48 | 232 | 9.95 | 15.81 | 1572.9 |
| W2 | 50.33 | 26.74 | 232 | 4.98 | 10.51 | 786.4 |
| **CLASS** | **187.30** | **117.73** | | **22.53** | **69.64** | **2938.9** |

3.09x = 5.292 ms vs 1.712 ms over 76 layers.

Four of the ten phases issue **no MFMA at all** (`glm-wuk-wuv-oproj-have-no-mfma`),
and they hold 11.97 us/layer — 17 % of the class. Two more carry essentially no
weight bytes (decode 0.07 MB, merge 0.00 MB) yet burn 10.51 us/layer between
them. A third of the class is, by construction, neither a matrix op nor a byte
stream.

---

## 2. MFMA issue occupancy — 2.65 % on the live SIMDs

Computed, not sampled, and exact in the numerator. At bs=1 every GEMM has M=1,
so one MFMA instruction covers a whole 16-row output tile and the instruction
count is exactly

```
instructions = weight_elements / (N_TILE * K_TILE)
```

with `v_mfma_scale_f32_16x16x128_f8f6f4` at 16x128 and 32 cycles, and
`v_mfma_f32_16x16x32_bf16` at 16x32 and 16 cycles — the same `MFMA_CYC = 32`
`isa_accounting.py` already uses. Both formats land on the same cycles per
weight element at peak, which is the only property the ratio depends on.

```
MFMA busy   2.939 M SIMD-cycles / layer / rank
live        111.1 M SIMD-cycles   (each phase's own workers x 4 wave64s)
available   171.1 M SIMD-cycles   (all 1024 SIMDs x 69.64 us x 2.4 GHz)
                     -> 2.65 % live, 1.72 % of the whole machine
whole wall           -> 0.86 %
```

This is an **upper bound on required work**, and the live-SIMD denominator is
the fair one (`glm-per-cu-roofline-denominator-is-wrong` — dividing every phase
by all 1024 SIMDs is the known error here).

It is corroborated bottom-up by the repo's own ISA measurement of the one phase
that has a per-subphase breakdown, `isa_accounting.py`'s `QKVA` table: the MFMA
K-loop is 41.0 % of the qkv_a tile and is **12 % MFMA busy / ~53 % vmcnt
stall** inside itself, i.e. ~5 % MFMA over the tile. Same order, from a
completely different instrument.

**No scheduling change to the tile class is limited by the matrix pipe.** The
MFMA units are idle 97 % of the time in the phases that use them at all.

---

## 3. Achieved HBM rate — 32.7 % of peak, and the traffic is not hidden

```
117.73 MB / 69.64 us = 1.691 TB/s   = 32.7 % of the MEASURED 5.17 TB/s peak
                                    = 32.4 % of the width-corrected achievable
```

The obvious objection is that 117.73 MB is a *model*: if the kernel re-reads 3x
that — a dequant path that streams weights twice, a non-coalesced access
pattern, an LDS spill — then the real rate would be at the roof and this would
be case (A) after all. That is the one thing the model cannot answer, so it was
measured.

`mem_busy_percent` from sysfs, sampled during a shipping-configuration NP=8 run
(no flags, no rebuild, host-side read, zero device perturbation), keeping only
samples where the shader is actually running (`gpu_busy > 50`; the kept samples
average **97 % gpu_busy**, so they are on the decode plateau):

| instrument | rate | samples/GPU | mem_busy mean | max |
|---|---|---:|---:|---:|
| sysfs `mem_busy_percent` | 10 ms | ~149 | **5.2 %** (5.1–5.7 across the 8) | 17–18 % |
| `rocm-smi --showmemuse` | ~1 Hz | 4–5 | 11.6–14.5 % | 16 % |
| **byte model predicts** | | | **16.3 %** | |

The two samplers disagree with each other by ~2.5x — expected, since
`mem_busy_percent` is a decimated duty-cycle register read asynchronously to the
phase structure and the 1 Hz sampler has n=4 per GPU. Neither is trustworthy to
better than a factor of ~3 in absolute calibration, and this document does not
rely on one.

What it relies on is the **falsification threshold**, and that has a wide
margin. The escape being tested is "real traffic is ~3x the model, so the class
is at the HBM roof after all"; that requires ~50 % controller busy. Both
instruments are far under it, and both are at or *below* the modelled 16.3 %,
with peak instantaneous samples of 17–18 % — the ceiling is not approached even
momentarily. There is no dequant or LDS mechanism eating the bandwidth, because
the bandwidth is not being eaten — it is not being requested.

---

## 4. So what *is* binding, and what would the fix have to be

Both the compute pipe and the memory pipe are idle, so the binding resource is
the third one: **outstanding requests per wave, and the dependent-issue chains
between them.** The evidence is already in the tree and it is consistent across
three instruments:

* `glm-attention-tiles-are-latency-bound-not-valu-bound` — qkv_a is 21 % VALU /
  5 % MFMA / **68 % vmcnt**.
* `isa_accounting.py`'s `QKVA` breakdown — 46.2 % of the tile is the
  prologue (resolve + EP fold + RMSNorm reciprocal + LDS stage), diagnosed as
  **L2 latency**: 96 KB/tile at 15.7 GB/s per CU of ~52 available.
* `mi355x-memory-hierarchy-bandwidth` — this part needs **>= 8 loads in
  flight** per wave to cover HBM latency at all.

The structural reason is bs=1 itself. 187.3 M weight elements spread over 232
workers is **0.4 MB per worker per layer**, at an arithmetic intensity of
exactly one MAC per weight element — the lowest a GEMM can have. There is not
enough work per wave to hide a load, and there is no batch dimension to add
any.

### The ceiling on any class-wide fix

| assumption | headroom | wall |
|---|---|---|
| every phase reaches its width-corrected byte roof | 3.580 ms of busy | at the MEASURED 0.27 busy->wall coefficient: **0.967 ms** |
| same, priced at 1:1 (upper bound, not the measured behaviour) | 3.580 ms | 3.580 ms |

0.27 is the right coefficient here and it is measured (`73c0afb`,
`glm-cutting-work-in-a-phase-is-absorbed`): a latency fix shortens a tile while
its rendezvous survives, which is regime A of the makespan rule exactly.
**0.967 ms is under the ruling's 1 ms build bar, so no build is authorised on
this axis and the stop is auditable against the table above.**

### And it would not reach the goal even at 1:1

| | ms/token |
|---|---:|
| measured wall | 10.619 |
| perfect tile class at the 0.27 coefficient | 9.652 |
| perfect tile class at 1:1 | 7.039 |
| **entire tile class DELETED — math, bytes and all** | **5.327** |
| entire tile class AND all ten rendezvous deleted | 2.956 |
| goal | 2.000 |

Deleting the biggest class in the budget outright leaves 2.7x the goal. This is
the same shape as the rendezvous result (`RENDEZVOUS_EDGE_TABLE.md` §6): no
single class contains the 5.3x.

---

## 5. The numerics asymmetry, in one line

TileRT's 2.020 ms is W8A16 — 8-bit weights, BF16 activations. Ours is MXFP4
weights (0.53125 B/element with the microscale block) on the MoE and MXFP8
(1.03125 B/element) on attention.

| | MB/layer/rank | tile-class byte roof at 5.17 TB/s |
|---|---:|---:|
| ours — mxfp8 attention + mxfp4 MoE | 117.73 | **1.731 ms/token** |
| TileRT numerics — 8-bit weights everywhere, bf16 KV | 187.45 | 2.756 ms/token |
| ratio | | **1.59x in our favour** |

**Our numerics make the byte roof 1.59x SMALLER than TileRT's, not larger, and
at 1.731 ms/token that roof sits UNDER the 2 ms goal.** MXFP4-weights +
FP8-activations is not what makes 2 ms arithmetically unreachable — it is the
only reason 2 ms is arithmetically on the table at all. The ruling's
option (c) cannot be written on a numerics argument.

---

## 6. What was measured, and what failed

**Control, this turn.** Shipping build, no flags, `MODEL_PATH` pinned to
`/home/claudeuser/models/glm5-mxfp4`, NP=8, bs=1, one decode row, 118 tokens in
115/118 iters. Two runs, host-side sampling only:

| run | ms/iter | generated text (rank 0) |
|---|---:|---|
| 1 | 10.654 | *"The capital of France is\<think\>1. Analyze the user's request… The capital of France is Paris."* |
| 2 | 10.575 | *"and the capital of the UK is / The capital of the UK is London."* |
| **mean n=2** | **10.615** | both coherent, on topic, correct |

Inside the 0.26 ms noise floor of the board's `10.571 n=3` (`54e4515`) and of
the standing 10.619 — the sysfs sampler costs nothing on the device, as
designed. **No change was made to the build on this board**; these are controls,
not a delta.

**rocprofv3 `--pmc` did not produce a number, twice.**

1. `MfmaUtil FetchSize WriteSize` in one pass: *"Could not construct profile cfg
   failed with error code 38: Request exceeds the capabilities of the hardware
   to collect."* Too many counters for one pass; there is no multi-pass option
   for a single persistent dispatch.
2. `SQ_VALU_MFMA_BUSY_CYCLES GRBM_GUI_ACTIVE`, filtered to `worker_kernel`:
   hung at `launch_persistent_kernel ENTER` with all 8 GPUs at 100 %, killed at
   ~5 minutes.

The mechanism is plausible — the megakernel runs `worker_kernel` and
`scheduler_kernel` on two streams and **requires them co-resident**, so any
dispatch serialisation the counter path imposes is a deadlock by construction —
but it is **not concluded here**. The NP=8 EP hang is independently ~1-in-3
(`repeat-runs-before-blaming-the-observer`), and the first attempt was
additionally confounded by orphan ranks from the aborted three-counter run
(`orphan-ranks-survive-a-killed-mpirun`). One clean hang is not a verdict on the
instrument. The harness is committed (`rocprof_rank0.sh`,
`profile_tile_class.sh`, `MPK_RANK_WRAPPER`, a no-op by default) so the next
attempt costs one run, and `tile_class_bound.py` reduces the CSV the moment one
exists.

`sample_hbm_activity.sh` is the instrument that did work, and it perturbs
nothing.
