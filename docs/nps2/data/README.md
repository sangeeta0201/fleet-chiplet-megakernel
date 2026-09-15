# Raw data

All from `mi355x-thor-2`, `HIP_DEV=8` (`0000:e5:00.0`), gpt-oss-120b, batch 1,
`MAX_SEQ_LENGTH=128 MAX_NEW_TOKENS=16`. Every arm restarted the `fleet_v1`
container before each trial. Format is `<arm>_<trial>: <ms/token> | <generated text>`,
with `HANG` where the run did not complete inside `TIMEOUT=600`.

Unless noted, arms carry `MPK_TERM_RECHECK=1 MPK_AID_SPLIT_FLAGS=1`.

## Perf arms

| file | what it is | mean ms |
|---|---|---|
| `ctrlC.txt` | **control**: no AID split flags, no `TERM_RECHECK` | 10.356 (2 of 3 HANG) |
| `flagC.txt` | AID split flags, no `TERM_RECHECK` | 3.881 (2 of 3 HANG) |
| `termA.txt` | `TERM_RECHECK=1` only — the stable baseline | 8.482 |
| `termF.txt` | `+ attn_release, layer_release` replicas | 4.011 |
| `hierA.txt` | `+ hier_barrier` | 3.761 |
| `moeA.txt` | `+ moe_fused_barrier` | 3.654 |
| `pairOff.txt` | `+ MPK_MOE_XCD_PAIR=0` — **best** | 3.422 |

`ctrlC` vs `flagC` is the matched pair proving the hang is *not* caused by the
AID split flags: both hang at the same 2/3 rate without `TERM_RECHECK`.

## Ablations (all on top of the best config)

| file | knob | mean ms |
|---|---|---|
| `strLay.txt` | `MPK_MOE_XCD_STRIPE_LAYER=1` | 3.459 |
| `w2gate.txt` | `MPK_W2_CONSUMER_GATE=0` | 3.665 |
| `lean.txt` | `MPK_LEAN_ARRIVE=0` | 3.375 |
| `busyA.txt` | busy-poll on all spin loops | 3.609 |
| `evp.txt` | `MPK_NPS2_EVENT_POLL=1` | 3.404 |
| `mltab.txt` | ml pointer tables replicated per AID | 3.423 |
| `kvw.txt` | `MPK_AID_LOCAL_SLOTS=1,2,4` (KV + QKV weight) | 3.474 |
| `acqA.txt` | `MPK_NPS2_L2_ACQUIRE=1` | 10.68 |
| `acqB.txt`, `acqBF.txt` | L2 acquire, with and without flags | 10.68 / 4.954 |
| `tok1.txt` | single-token determinism probe (capture failed, text empty) | — |

All of these are inside noise or worse. None is a lever.

## Diagnostics

| file | contents |
|---|---|
| `aid_placement_map.txt` | `MPK_AID_LOCAL_MAP=1` — home AID, size and per-XCD-group latency of every input slot. **This is the evidence that the model sits in AID0**: 17 of 20 slots report `aid=0`, and `gate_up_weight` / `down_weight` (2260 MiB each) show `skew_ns ≈ -98`, i.e. XCDs 4-7 pay ~98 ns more on every access. |
| `layer_boundary_drain.txt` | `MPK_DRAIN_STATS=1` — `[DRAIN]` shows the layer boundary is `spin=8843` of `tot=10753` (82% waiting). `[XCDSPIN]` shows XCD4 arriving **last 2338 of ~2500 times**, with all of AID1 behind all of AID0. |
| `interlayer_split.txt` | `MPK_INTERLAYER_SPLIT=1` — `fnpre=8 binv=1 post=1880`. Rules out TaskDesc setup and `buffer_inv` as the inter-layer cost; it is all barrier wait. |
| `xcdspin_single_replica.txt` | same `[XCDSPIN]` under MoE `SINGLE_REPLICA`. XCD4's share of last-arrivals drops 94% -> 41%, confirming the AID0 diagnosis even though the arm is a net loss on wall clock. |

## Caveats

- Run-to-run variance is large — the same flags-off config measured 8.54 and
  10.24 ms. Treat sub-5% differences as noise.
- Arms with 2 trials are exploratory; the staged perf table (`termA` ->
  `pairOff`) used 3.
- The `[MOE_W13]` subphase timer emits negative `prologue_ns`; its absolute
  numbers are not trustworthy, only the compute-vs-barrier ratio.
