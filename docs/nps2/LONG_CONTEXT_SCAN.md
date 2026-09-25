# Long-context attention: where the time goes, and the scan's load schedule

**Where.** Phase snapshot (last worker per boundary), same build, 31 KV chunks,
~300 vs ~16k tokens of context. Full-attention layers: attention chunks +
merge 5.86 -> 18.72 us, every other phase within +-0.3 us; layer 39.60 ->
52.49 us. Sliding-window layers +1.0 us in total. 18 x 12.9 us is the whole
latency growth from 250 to 16k tokens (1.497 -> 1.755 ms/token). Every rank
group's own chunk time grows the same, so it is the per-worker scan, not a
straggler: each worker streams its ~135 KB of die-local K/V at ~10 GB/s per CU.

**Load schedule of the wave-local scan** (all opt-in, bit-exact: 16k decode
text identical over 51190 chars in every arm):

| flag | tiles in flight / wave | scan VGPRs | fused-layer scratch | 16k ms/token |
|---|---|---|---|---|
| default | 1 | 199 | 16 B | 1.754-1.762 |
| `MPK_ATTN_WL_RING=2` | 2 (registers) | 232 | 16 B | 1.702-1.705 |
| `MPK_ATTN_WL_RING=4` | 4 (registers) | 248 | **80 B** | 1.758 (spill costs +0.03 ms everywhere) |
| `MPK_ATTN_WL_DMA=4` | 4 (LDS, `buffer_load_dwordx4 ... lds`) | 151 | 16 B | 1.701-1.704 |

`MPK_ATTN_WL_DMA=4` vs the default, 16k decode: 16k -3.2%, 12k -2.6%, 8k
-1.7%, 4k -0.7%, <= 2k unchanged; whole decode 1.640-1.645 -> 1.617-1.618.
It matches or slightly beats ring 2 (-0.1 to -0.4%) with 80 fewer VGPRs.
Depth beyond 2 buys almost nothing: what is left per tile is the scan's own
dependency chain (LDS reads, QK MFMA, max reduction, exp, rescale, PV MFMA)
at one wave per SIMD, about 1.2 us per tile per wave at 16k.

Register-ring traps, all visible as `vmcnt(0)` inside the loop: the per-tile
page-id load is a FLAT load (counts on vmcnt and lgkmcnt) -> hoist the two
page ids of the wave's slice; zeroing padding rows in an else branch writes
in-flight load destinations -> load every row, zero at commit; conditional
refills make the committed slot the newest on one path -> refill
unconditionally. The DMA ring avoids all three by construction (explicit
`s_waitcnt vmcnt(4k)`, padding zeroed at the V read).

`MPK_ATTN_PAGE_CACHE` (default-scan only): reload the page id only at a page
crossing. Token-identical across the 4096-token page boundary (5k prompt).
