# GLM-5 decode layer: dependency chain, handoff residency, and overlap

Measured on `amd_mi355_glm5_744b`, NP=8 EP, 78 layers, 232 workers (29 x 8 XCDs),
wall 10.26 ms. Geometry from `[CFG]` lines of `/tmp/glm5_wrep_1_3.log`; register
counts from `demo/glm5/test-hip-amdgcn-amd-amdhsa-gfx950.s`.

Purpose: decide, per edge, whether the producer->consumer handoff can live in
registers/LDS instead of global memory, and which subgraphs can run concurrently.

## Measured geometry

| quantity | value |
|---|---|
| hidden / q_lora / kv_lora+rope / moe_intermediate | 6144 / 2048 / 512+64 / 2048 |
| heads_per_rank (tp=1 head shard) | **8** -> exactly **1 head per XCD** |
| q_b tiles | 16/XCD (opw=16, 256 out/head) |
| W_UK tiles | 4/XCD (ROWS=128, 512 out/head) |
| o_proj | K=16384, 768 cols/rank, 24 tiles/XCD |
| kv_chunks / q_groups / merge_dim_splits | 16 / 4 / 32 |
| decode work items | q_groups x chunks = **16 tiles = 2/XCD = 7% of workers** |
| merge work items | q_groups x dim_splits = 128 = 16/XCD |
| W13 / W2 | OPW=64 -> 64 / 96 tiles per expert |
| activated experts per rank per token | ~1 routed + 1 shared |

## Register pressure: one phase pins the whole kernel

The megakernel is a single launch, so occupancy is set by the max over all
inlined phases. Allocation granule 8, unified VGPR file 512/SIMD.

| device function | unified VGPR | workers used | waves/SIMD supportable |
|---|---|---|---|
| `mla_decode_absorbed` | **284** | **16 of 232** | **1** |
| `..._mxfp8_bias_mla_kvupd` (qkv_a) | 250 | 192 | 2 |
| `..._mxfp8_bias` (q_b) | 232 / 218 | 128 | 2 |
| `gang_moe_w13_linear_mxfp8` | 170 | 232 | 3 |
| `gang_moe_w2_linear_mxfp8` | 169 | 232 | 3 |
| **`worker_kernel` (launched)** | **284**, 0 spill, LDS 6544 B | — | **1** |

284 -> granule 288 -> `512/288 = 1`. **Cutting 28 registers from
`mla_decode_absorbed` alone takes the entire megakernel to 2 waves/SIMD.**
gpt-oss's `worker_kernel` is 261 — 5 registers from the same cliff.

There is no prefill path to compile out: `grep -cE "prefill|seqlen_q"` is 0 in
both `gang_mla_decode_mi300.cuh` and `gang_mla_attn_fused_mi300.cuh`. The
registers are live MFMA state, not a dead branch. So task #16 as written does
not apply; the work is to shrink the live set (fewer accumulators in flight,
shorter software pipeline depth, or splitting the head loop).

## The dependency chain, edge by edge

Granularity column = what the consumer *actually* needs, not what it waits for.

| # | edge | barrier | granularity of the true dependency | producer/consumer XCD span | register/LDS residency possible? |
|---|---|---|---|---|---|
| 1 | resadd+RMSNorm -> qkv_a | — | elementwise | same WG | **already done** — recomputed per tile, 192x, never leaves registers |
| 2 | qkv_a -> q_b | 769/762 | full K-reduction over 6144 | q_a columns read by all 8 XCDs | **no** — output-parallel producer, K-reduction consumer |
| 3 | q_a RMSNorm -> q_b | — | elementwise | same WG | **already done** (fused prologue) |
| 4 | q_b(head h) -> W_UK(head h) | 768 | **per-head** | head h's 16 q_b tiles and 4 W_UK tiles are **both on XCD h** | barrier already XCD-local; full WG-residency would need 1 WG/head (16x less parallelism) |
| 5 | kv_a -> KV-cache append | — | elementwise | XCD 0 tile 0 writes, all 8 read | already in the q_b phase, parallel |
| 6 | q_b/W_UK -> decode | 763 | q for all heads | one decode tile reads 6 XCDs' q | **no** |
| 7 | decode -> merge | 764 | **per (q_group, dim-split)**, reduce over 16 chunks | XCD x computes chunk x for *all* heads -> merge reads all 8 | **no under sequence-split.** Head-split would make it XCD-local at the cost of 8x KV reads |
| 8 | merge(head h) -> W_UV(head h) | 767 | **per-head** | — | candidate: co-locate merge and W_UV per head |
| 9 | W_UV -> o_proj | 767 | full reduction over 64x256 | — | **no** |
| 10 | o_proj -> post-attn norm -> router GEMV -> TopK | — | elementwise / reduction over 6144 | — | **already done** (#63, folded into the o_proj epilogue) |
| 11 | normed row -> W13 | 765 | needs TopK for *routed* experts only | normed row deliberately **not** write-through — one router worker per XCD writes into its own L2, only that XCD's W13 tiles read it | **already XCD-local** |
| 12 | normed row -> shared expert | — | **no TopK dependency at all** | — | **already done** (#64, runs across RoutingWait) |
| 13 | W13(expert e) -> W2(expert e) | 766 | reduction over e's 2048 intermediate | intermediate spread over all 8 XCDs; `arrivals = tiles_per_xcd * 8` | **no** — only ~2 experts of work exist per rank, so it *must* spread. Split-K W2 measured **+4.12 ms** |
| 14 | W2 -> MulSumAdd -> residual | — | elementwise | — | **already done** (#11, #22) |
| 15 | hidden(L) -> qkv_a(L+1) | 761 | full row | — | **the one true cross-layer edge** — see below |
| 16 | MoE partial -> EP all-reduce | 901 | cross-rank | fabric | genuinely global |

### What this says

The elementwise/epilogue subclass is **exhausted** — every edge marked "already
done" is a fusion that cost no parallelism, and all of them have landed.

Every remaining global handoff is a **K-reduction**: the producer is
output-parallel, the consumer reduces over the producer's whole output. Keeping
such an edge in registers forces producer and consumer onto one workgroup, which
is why the two attempts lost:

| | direction | delta |
|---|---|---|
| `5d364ab` W_UV out of o_proj | register-resident -> global + barrier | **17.28 -> 14.85 ms** |
| `981bf62` W_UK out of q_b | register-resident -> global + barrier | 14.94 -> 14.81 ms |
| W2 split-K | global -> register-resident | **+4.12 ms** |

At bs=1 the machine is occupancy-starved, so a 3.77 us round trip is cheaper
than halving the parallelism. **Register residency is the wrong lever on this
workload.** The right one is occupancy.

## Overlap: what is independent and not exploited

Concurrent-capable subgraphs, checked against the chain above:

1. **Shared expert vs routing** — independent, **already exploited** (#64).
2. **KV append vs q_b/W_UK** — independent, already co-phased.
3. **Decode chunks 0..14 vs the current token's KV append** — only chunk 15 needs
   the new token. Already parallel (append sits in the q_b phase).
4. **MoE(L) vs qkv_a(L+1)** — blocked by edge 15, and this is the only remaining
   one. The MoE phase has a measured-free worker hole: 21 of 29 workers per XCD
   own no W13 tile, and giving each 87 KB of cold weight read (14.6 MB/layer/rank,
   ~90% of qkv_a's whole weight volume) costs **+0.030 ms** (`e6ccf70`, n=3
   paired, output verified). The capacity half of "heterogeneous worker groups"
   is therefore **proven**; only the dependency half remains.

   RMSNorm is linear in its argument up to a scalar, so
   `q_a = (1/||h2||) * [(h1*g) @ W + (moe_out*g) @ W]`. The `h1` term is available
   right after o_proj(L), i.e. before the MoE runs. That splits edge 15 into an
   early half that fits the free hole and a late half that does not.

   **Priced and refuted as stated:** an extra qkv_a pass costs **0.669 ms**
   (`glm-qkva-extra-pass-costs-0.669ms`). The split replaces one full-size GEMM
   with a hidden one *plus* an un-hidden one over the same `W`, so the un-hidden
   half is unchanged and the second `W` read is not free. Needs a variant that
   does not re-read `W`.

## Ranked conclusions

| rank | item | size | status |
|---|---|---|---|
| 1 | Cut 28 registers from `mla_decode_absorbed` -> 2 waves/SIMD megakernel-wide | occupancy is the 3.25x gap between 6.3 ms of tile time and 1.94 ms of bytes | **never attempted** |
| 2 | Narrow barriers to true dependency granularity (`766` is GPU-wide for a rank-local dependency) | skew, which is the one thing not absorbed | **never attempted** |
| 3 | Cross-layer overlap into the proven-free MoE hole, without a second `W` read | up to the qkv_a phase, 32.4 us/layer | dependency half open |
| 4 | MTP speculative decode | divides ms/token by accepted length; also raises effective batch, which is the only thing that fills idle workers without changing the schedule | machinery exists (`spec_decode_class`), `modeling_glm_moe_dsa.py:783` drops the layer |

Register residency and further op fusion are **closed** — measured negative three
times, for a structural reason that will not change at batch 1.
