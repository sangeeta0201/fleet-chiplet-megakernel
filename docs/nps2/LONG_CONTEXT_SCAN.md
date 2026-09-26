# Long-context attention: where the time goes, and the scan's load schedule

**Where.** Phase snapshot (last worker per boundary), same build, 31 KV chunks,
~300 vs ~16k tokens of context. Full-attention layers: attention chunks +
merge 5.86 -> 18.72 us, every other phase within +-0.3 us; layer 39.60 ->
52.49 us. Sliding-window layers +1.0 us in total. 18 x 12.9 us is the whole
latency growth from 250 to 16k tokens (1.497 -> 1.755 ms/token). Each worker
streams its ~135 KB of die-local K/V at ~10 GB/s per CU. (An earlier version
of this note grouped `[PSLOTW]` rows by `w % 31` to argue there is no
straggler. That grouping is not the role rank -- in precomputed dispatch a
worker claims its rank with an atomic at start -- so the claim is withdrawn.)

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

## Split-KV chunk count from the live window (2026-09-25)

`NUM_KV_CHUNKS` is compiled from max_seq_length, so every long decode builds 31
chunks and pays for them at every context: the merge reads 31 partials and 31
workers per die join the QKV epoch and the chunk barrier, most of them with no
tiles. 400-token decode: 8 chunks compiled 1.459-1.463 ms, 31 chunks 1.500-1.506.

- `MPK_KV_CHUNKS_ADAPTIVE=1`: the HD64 decode applies demo.py's rule to the
  live window at run time (`max(8, min(NUM_KV_CHUNKS, ceil(ntiles / 4) / 2))`
  chunks of 16-token tiles) and returns the number of chunks holding tiles. The
  merger is always one of the chunk workers, so it takes that value from its
  own attention call and dispatches the merge onto a compile-time loop of 8, 16
  or NUM_KV_CHUNKS partials (a run-time loop bound serialized the partial
  loads). Up to ~1k tokens of context tiling and merge order are exactly an
  8-chunk build's. A first version re-read kv_indptr / kv_last_page_len after
  the QKV epoch and kept only 0.006 ms of the gain: two serialized flat-load
  round trips per attention worker per layer.
- `MPK_KV_SW_IDLE=1` (with the above): a 128-token sliding window spans at most
  9 tiles, which the rule never splits over more than 8 chunks, so in sliding
  layers chunk workers >= 8 skip the QKV-epoch poll and the attention call.
  They still arrive at both barriers, which count a fixed participant set.

NPS2, recipe + `MPK_ATTN_WL_DMA=4`:

| | 31 chunks | + adaptive | + idle skip |
|---|---|---|---|
| 400-token decode, 31 chunks (A/B/A/B/A) | 1.500 / 1.502 / 1.506 | 1.486 / 1.486 | 1.481 / 1.480 (A 1.484-1.486) |
| 16k decode, context 250 / 1k / 2k | 1.505 / 1.533 / 1.565 | 1.496 / 1.525 / 1.544 | 1.490 / 1.522 / 1.539 |
| 16k decode, context 4k / 8k / 16k | 1.570 / 1.618 / 1.703 | 1.566 / 1.620 / 1.704 | 1.563 / 1.607 / 1.700 |

Gates: 16-token hash unchanged; the 400-token decode is byte-identical to an
8-chunk build; 1k prompt at 31 chunks == the 8-chunk output (== torch); 3k
prompt at 24 chunks == the 24- and 8-chunk outputs; 16k decodes deterministic.
What remains of the 31-chunk cost at short context (phase snapshot, ~390
tokens): QKV-epoch wait +0.3 us and attention + merge +0.3 us per layer.

## SCC and inline asm

`s_addk_i32` writes SCC. The attention DMA issue advances M0 with it and now
declares `"scc"` in its clobbers. Without the clobber the compiler may keep an
SCC value live across the statement; a benchmark that copied this asm did
exactly that (`s_cmp` before it, `s_cselect` after it) and silently skipped a
store. Fleet's ISA had no such case (all 16 sites checked), and the 16k decode
is byte-identical with the clobber. The MoE and LM-head asm use the same
instruction without it; scan the build's `.s` after changing any of them.

## Streaming K/V loads and the scan's bound (2026-09-26)

- `MPK_ATTN_WL_NT=1`: the DMA scan's K/V loads carry `sc0 nt` (KV is read once
  per token per layer), as the weight streams do. Bit-exact. NPS2 16k decode,
  A/B/A/B/A: average 1.599 / 1.601 / 1.603 -> 1.560 / 1.561 (-2.5%); 1k -0.9%,
  4k -1.3%, 8k -2.6%, 12k -3.5%, 16k -4.2% (1.693 -> 1.622); text identical.
  The attention-shaped benchmark (`attn_lat`, 31 WGs/die, pure stream) shows
  why: `sc0 nt` lifts a local stream from ~4.1 to ~4.8 TB/s and does nothing
  for a remote one.
- What bounds the long scan (timing-only `MPK_ATTN_PROBE`, lean loop, 16k):
  no KV traffic 1.643, no tile compute 1.636, both 1.689 -- compute and KV
  streaming each ~0.16 ms/token, partly overlapped.
- Null, kept opt-in: `MPK_ATTN_WL_LEAN` (scalar loop control, full-tile steady
  state, rescale skipped when every factor is exactly 1.0f: 278 -> 219
  instructions per tile, bit-exact, 1.599 vs 1.601) and `MPK_ATTN_WL_PIPE`
  (tile t+1's QK next to tile t's softmax/PV, bit-exact, 1.604-1.607 vs
  1.607). Neither issue slots nor a single dependency chain bound the loop.
- `MPK_CHAT_DATE=YYYY-MM-DD` (demo.py) pins the chat template's date line so
  outputs and torch references compare across days.
- At long context the attention critical path is the slowest chunk (the one
  crossing a KV page boundary) plus the merge run by the die's last arriver
  (~3.5 us per full layer at 16k, both modes).

## Full-attention idle chunk workers: `MPK_KV_FULL_IDLE` (2026-09-26)

With `MPK_KV_CHUNKS_ADAPTIVE` a full-attention layer below ~4k tokens of
context has fewer live chunks than compiled ones (8 up to ~1k tokens, 16 at
~2k, of 31). The workers of the empty chunks still polled the QKV epoch and
made an attention call that found no tiles. `MPK_KV_FULL_IDLE=1` (needs
ADAPTIVE) has each chunk worker derive the live count from `kv_indptr` /
`kv_last_page_len` before the epoch and skip the poll and the call when its
chunk is past it, as `MPK_KV_SW_IDLE` does for sliding layers. Skipped
workers still arrive at both counters. When the merge bucket covers a
skipped chunk (live 9-15 read by the 16-chunk loop), its worker stamps the
chunk's lse slots with -1e30 so the merge weights it 0.

NPS2, 400-token decode compiled at 31 chunks, A/B/A/B/A: 1.479 / 1.477 /
1.483 ms without, 1.468 / 1.465 with (-0.9%, both B below the A range).
Bit-exact: 16-token hash, the 400-token text vs the 8-chunk build, 1k at 31
chunks and 3k at 24 vs their references, and a 16k decode (43,433 chars
identical). From ~4k tokens every chunk is live, so it is neutral there
(16k decode, one run: 250 / 1k / 2k / 4k / 8k / 12k / 16k = 1.481 / 1.498 /
1.519 / 1.543 / 1.575 / 1.603 / 1.634 vs 1.484 / 1.504 / 1.519 / 1.541 /
1.573 / 1.600 / 1.630 ms).
