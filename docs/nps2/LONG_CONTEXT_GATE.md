# Long context decoded wrong: the narrowed Phase 9 gate

**Symptom.** With more than 10 split-KV chunks (`ck_fmha_num_kv_chunks` > 10,
i.e. `max_seq_length` > ~1344), long prompts decoded differently on every run
and mostly as garbage once enough chunks held real KV. Same code in NPS1 and
NPS2. The 16-token gate cannot see it: at 88 tokens the live chunks all sit on
ranks 0-5.

**Cause.** `MPK_W2_CONSUMER_GATE` (default on) lets every worker outside the
QKV set skip the Phase 9 layer gate, on the premise that its first shared
access in the next layer is the Phase 6 `attn_release` poll. Once
`NUM_KV_CHUNKS` exceeds `total_qkv_tiles_per_xcd` (10), ranks
`[10, NUM_KV_CHUNKS)` also run attention chunks (Phases 2-5) of the next
layer, without having waited. The exact racing access is not identified; the
fix is measured, not argued.

**Fix.** `MPK_GATE_ATTN_JOIN` (default on, `=0` to A/B): the waiter set becomes
`qkv_does_qkv || qkv_attn_rank < ATTN_PARTICIPANTS`, and the unrotated arrival
prefix becomes `max(qkv_work_slots, ATTN_PARTICIPANTS)`. At 10 chunks or fewer
nothing changes.

| run (PyTorch reference = same demo.py without `--use-mirage`) | before | after |
|---|---|---|
| 1k prompt, 31 chunks, vs torch (16 tokens) | 0/16, reruns differ | 16/16, reruns bit-identical |
| 3k prompt, 24 chunks | garbage 4 of 5 runs | token-identical to 8 chunks |
| 5k prompt, 31 chunks | garbage 2 of 3 | coherent |
| 1000-token decode, 31 chunks, two runs | split near char 900 | identical, 3718/3718 chars |
| 16-token gate | 3d54adb71e19 | 3d54adb71e19 |
| ms/token, 1k/31 and 5k/31 | 1.508 / 1.556 | 1.508 / 1.559 |

Ruled out on the way: stale vL1 at the QKV-epoch and merge acquires (both are
bare `buffer_inv`, a no-op on gfx950; `buffer_inv sc0` at both changed
nothing), the wave-local scan (8 chunks x 24 tiles is fine), the KV layout
(NHD is also wrong), `MPK_W2_ONLY_ARRIVE=0` (still wrong).
`MPK_W2_CONSUMER_GATE=0` also fixes it but costs +5.8%.

At 3k and 5k the first content word still differs from torch ("The user..."
vs "We need..."), identically at 8 and 24 chunks: a fleet-vs-torch near-tie,
not this bug.
