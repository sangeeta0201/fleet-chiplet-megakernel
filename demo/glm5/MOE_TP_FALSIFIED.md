# MPK_MOE_TP: both halves of the 1.4 ms are dead, and the full shard does not compile

Offline. **No GPU run, no build.** This falsifies my own prediction in
`TILERT_SHAPE_DIFF.md` (`7c0c2ee`) before spending the build it asked for.

The ruling authorized the collective-deletion half first, behind a flag, on one
layer. Step 1 of that build is reading the EP fold. Reading it is what killed it.

## Half 1 — the collective term is 0.000 BY CONSTRUCTION, not by measurement

`_rnlm8_ep_fold_slice` (`gang_rmsnorm_linear_mxfp8_bias_mi300.cuh:409`) loads
`EP_PEER_SLOTS` copies of the hidden row at `p * EP_SLOT_ELEMS`, sums them
element-wise in fp32, and writes bf16. **That is an 8-way all-reduce of the
hidden row.** It is not an expert-routing collective and there is nothing
EP-specific about it.

TP needs the identical all-reduce:

| | what rank *r* contributes | reduction |
|---|---|---|
| EP (now) | the experts *r* owns, summed | 8 partials of the 6144-elem hidden row |
| TP | *r*'s 1/8 slice of the intermediate of **every** activated expert | 8 partials of the 6144-elem hidden row |

Same eight partials, same row, same sum, same bytes. TileRT's
`expert_down_allreduce` is that same all-reduce; its only structural difference
is that it sits in W2's epilogue rather than in Phase 0 of the next layer.

**So there is no collective to delete.** And the fusion difference is already
measured out, twice: `glm-qkv-ep-fold-hoist-is-neutral` and
`glm-widening-the-ep-fold-is-neutral`, with the mechanism explained by
`glm-inter-rank-skew-is-paid-once` — the cross-rank skew is a tax paid once at
the first cross-rank rendezvous, and moving where the code sits does not move
when the peers' data arrives.

Predicted 1.005 ms → **0.000 ms.** This half needed no GPU time to settle,
because it is an identity, not a measurement.

## Half 2 — the work term is ~15x smaller than I priced it

I scaled the MoE phase by bytes at its measured 48%-of-roof efficiency: 80.22 →
22.56 MB, so 31.07 → 8.7 µs, saving 22.3 µs/layer. **That estimator is wrong for
this phase, and the direct measurement of exactly this variable already exists.**

`glm-ep-routed-imbalance-is-floor-not-wall`: the busiest rank owns **3x** the
mean routed bytes (60.16 vs 20.05 MB/layer) and its MoE makespan is within
**1.05 µs** of every other rank's.

The reason is round quantization, not bandwidth. W13 runs 6 tiles/XCD/expert and
W2 runs 12; both fit inside **one** grid-stride round of 29 workers **up to 4
owned experts**. The busiest rank is at exactly 4.0. It does not pay an extra
round — the light ranks pay idle workers instead. Cutting 4.0 → 1.125 stays
inside the same single round, so it removes no round and no makespan.

Balancing the entire imbalance is measured at ~1 µs/layer = **0.08 ms**, under
the 0.26 ms noise floor. Predicted 0.388 ms → **~0.03 ms.**

## And the full shard does not compile

Full TP for the MoE is an 8-way **K-split of W2** — sharding the contraction
(intermediate) axis is what makes the all-reduce necessary in the first place.

* `gang_moe_linear_mxfp8_mi300.cuh:532` —
  `static_assert(MFMA_ITERS >= 4 && MFMA_ITERS % 4 == 0)`. W2 has
  `MFMA_ITERS = 16` and a depth-4 pipeline; an 8-way split leaves **2**. The
  static_assert fires. This blocker is already written into the comment at
  `:336`.
* `glm-w2-splitk-is-a-large-negative` — `MPK_W2_KSPLIT=2` measured **+4.12 ms**.
  So the shape is not merely blocked, it is the exact shape of the largest
  measured negative on the ledger.
* The intermediate points are priced too. `glm-ep-routed-imbalance-is-floor-not-wall`
  records **EP=2 × TP=4 as priced NET NEGATIVE**: it saves ~1 W2 round
  (0.23 ms) and costs one NEW cross-rank rendezvous to all-gather the 4 KB
  swiglu row.

## The corrected total

| term | predicted `7c0c2ee` | corrected | why |
|---|---|---|---|
| collective deletion | 1.005 ms | **0.000** | TP needs the same 8-way hidden-row all-reduce; the fold already is one |
| MoE work | 0.388 ms | **~0.03** | 3x bytes measured at 1.05 µs of spread; 4.0 experts still fits one round |
| new all-gather | — | **negative** | EP=2×TP=4 priced net negative |
| **total** | **~1.4 ms** | **~0.03 ms, and it does not compile** | |

## The transferable error

I priced a phase by scaling its bytes against a roofline when a **direct
measurement of that phase's response to that exact variable** was already on the
ledger. The roofline said 15x more than the measurement did. `glm-ep-routed-
imbalance-is-floor-not-wall` states the rule and I did not apply it: *a byte
imbalance only costs makespan if it crosses a grid-stride round boundary —
compute tiles/XCD vs 29 before believing a roofline imbalance line.*

Second, I read TileRT's *op names* and inferred a mechanism from them.
`expert_down_allreduce` and our Phase-0 EP fold have different names, different
positions in the graph, and the same semantics. The op graph tells you the
shape; it does not tell you which of your own ops already implements it.

## What survives of the TileRT diff

The reframe stands and is unaffected: our regime-B closure is a proof about our
shard layout, not about the model. But the specific re-shard it suggested is
dead, and the arithmetic above says the 3.958 ms in the rendezvous + collective
classes is **not reachable by moving the MoE from EP to TP**, because the
cross-rank half of it is the same all-reduce in both layouts and the intra-device
half is the regime-B-closed set.

TileRT's remaining advantage over us on this axis is that it has **two** device
rendezvous where we have **ten** — and the ten are intra-device, already proved
unremovable given our sharding, and unaffected by the MoE's EP/TP choice.
