# MI450 (gfx1250) measurements under FFM-Lite and AM

Captured 2026-09-10 on smci350-rck-g03-f15-21. Reproduce with `./sweep.sh`.

## Correctness under FFM-Lite

All nine kernel harnesses pass, and so does the megakernel e2e chain
(`tests/mi450/e2e`, exact token 55) and `test_hello_megakernel`.
`test_moe_fused_mxfp4` fails and is excluded from the perf table below --
a failing kernel's cycle count is meaningless.

## Cycle counts under AM (`am_profile_env.sh`, 1 XCC, 1.802 GHz modeled)

Six kernels ran to completion under AM **and passed their own correctness
check there**, so these are cycles for runs that computed the right answer.

| kernel | cycles | us | waves | instrs | WMMA ops |
|---|---|---|---|---|---|
| test_mx_wmma | 3,475 | 1.93 | 1 | 18 | 8 |
| test_rmsnorm_wave | 10,635 | 5.90 | 8 | 3,017 | 0 |
| test_gemm_wmma | 18,413 | 10.22 | 4 | 2,744 | 128 |
| test_topk_softmax | 49,386 | 27.41 | 4 | 1,252 | 0 |
| test_kv_cache_update | 51,761 | 28.73 | 16 | 32,274 | 0 |
| test_linear_wmma | 105,619 | 58.62 | 8 | 21,960 | 1,536 |

**These are latency, not throughput.** `CU_UTILIZATION` is 3-6% on every one
(one or two workgroups on a 32-CU config) and `XDL_EFFICIENCY` on the GEMM is
0.7%. They are the correctness harnesses at their built-in shapes, so the
numbers are dominated by cold-miss memory latency on a single CU. A throughput
figure needs the grids widened.

**Ignore the cache-hit columns.** `L1_REQUESTS` reads 1 for a 2,744-instruction
kernel and `L2_HIT_RATE` is exactly 0.000 on all nine, which is not credible.
The cycle, wave and instruction counts are the defensible outputs.

**Reproducibility.** Two back-to-back MT runs of the GEMM both gave exactly
18,413 cycles; `AM_SINGLE_THREAD=1` gives 18,372 and an earlier MT run gave
18,376. The threading mode shifts the result ~0.2%, so quote three significant
figures at most.

## Three kernels abort inside AM

`test_attention_wmma`, `test_rmsnorm_linear_mxfp4_bias` and
`test_moe_linear_mxfp4` all die with the identical AM internal assertion:

    Assertion failed at /src/mi400-am/model/am/blocks/sq/am_inst_buffer.cpp:283
      (IsInstDecoded): m_inst_buffer[ib_index].pc == pc

All three **pass under FFM-Lite**, so this is AM's SQ instruction buffer losing
PC coherence, not a fleet bug. It is deterministic and threading-independent:
re-running all three with `AM_SINGLE_THREAD=1` (AM_CLOCK_MT=0, wgp/sa/se MT
off) reproduces the same assert at the same line, which rules out the data race
AM's own "concurrent clocking is experimental" warning suggests. This is what
blocks attention and the MoE MXFP4 datapath from getting AM numbers.

## The megakernel does not run under AM

Both `test_hello_megakernel` and the e2e chain reach 17 HIP-level dispatches
(init/memset) and then the MPK scheduler hands out **zero tasks**. hello's own
watcher printed `dispatches=0 (no progress)` for 1,160 s before being killed.

This is NOT the clock read. PORTING_MI450.md attributes the megakernel's AM
failure to the unconditional `get_wallclock_ns()` at persistent_kernel.cuh:1998
lowering to `s_sendmsg_rtn`; building with `-DMIRAGE_REALTIME_FALLBACK=1` to
remove that read entirely still yields zero task dispatches, with an identical
watchdog trace. Both variants retire *nonzero* instructions per watchdog window
(1,664 / 19,745 / 20,533), so it is a spin livelock, not a decode stall.

Untested hypothesis: AM has no equivalent of FFM's `ffm_enable_time_slicing`,
so the scheduler's first worker rendezvous never completes. `probe/rendez.hip`
(a bounded two-block rendezvous) is written to settle this and has not been run.

## No end-to-end decode latency is available

Three independent reasons, any one of which is sufficient:
1. The only e2e harness is synthetic -- `hidden=128, vocab=64, 2 layers`, no
   attention, no MoE, no KV cache. GPT-OSS's real layer orchestrators are not
   ported. It is ~5 orders of magnitude less work than a real decode step.
2. The megakernel does not run under AM at all (above).
3. FFM-Lite has no cycle model. Its `[FWD_PASS] time_ms=` is model bookkeeping.

## gfx1250 clock hazard

Only `wall_clock64()` emits a clock read on gfx1250, and it lowers to
`s_sendmsg_rtn_b64 sendmsg(MSG_RTN_GET_REALTIME)`. Both
`__builtin_readcyclecounter()` and `clock64()` compile to a store with **no
clock read at all**, silently. See `probe/clk.hip`.
