# Porting Fleet to MI450 (gfx1250)

Working notes for the port. Everything in the "verified" sections was checked by
compiling to ISA with the gfx1250 toolchain; everything in "unverified" was not,
because there is no gfx1250 hardware on this machine and FFM has not been run.

## The headline

MI450 is **gfx1250**: wave32, WMMA, RDNA-lineage. Fleet targets **gfx950**:
wave64, MFMA, CDNA. This is not a retarget within a family. The CDNA matrix
builtin is rejected outright by the gfx1250 compiler:

    error: '__builtin_amdgcn_mfma_f32_16x16x16bf16_1k' needs target feature mai-insts

The task graph, scheduler, and host runtime should survive. The 50 files /
23,478 lines under `include/mirage/persistent_kernel/tasks/mi300/` mostly do not.

## Toolchain

ROCm 7.0.0 at `/opt/rocm` does **not** know gfx1250 (LLVM knows gfx1200/1201
only). A gfx1250-capable build is at `/home/claudeuser/mi450-toolchain`:

    https://rocm.genesis.amd.com/tarball/therock-dist-linux-gfx1250-7.12.0a20260226.tar.gz

    T=/home/claudeuser/mi450-toolchain
    $T/llvm/bin/clang -x hip --offload-arch=gfx1250 --rocm-path=$T -Iinclude ...

Newer nightlies exist at `rocm.genesis.amd.com/tarball/` if this one goes stale.

## Verified intrinsic mapping

Each row was compiled and the emitted instruction read back from `-S` output.

| Concern | gfx950 (now) | gfx1250 | Status |
|---|---|---|---|
| Matrix bf16 | `v_mfma_f32_16x16x16bf16_1k` | `v_wmma_f32_16x16x32_bf16` | ✅ emits |
| Matrix MX | `v_mfma_scale_f32_16x16x128_f8f6f4` | `v_wmma_scale_f32_16x16x128_f8f6f4` | ✅ emits |
| FP4 dequant | `v_cvt_scalef32_pk_bf16_fp4` (2/call) | `v_cvt_scale_pk8_bf16_fp4` (8/call) | ✅ emits |
| Realtime clock | `s_memrealtime` | `s_sendmsg_rtn_b64` via `wall_clock64()` | ✅ emits |
| Global→LDS | `global_load_lds_dword` | `global_load_async_to_lds_b128` | ✅ emits |
| Wave | 64 | 32 | ✅ |

**The MX GEMM shape is unchanged at 16x16x128 f8f6f4.** Fleet's MoE MXFP4
datapath — the change the README credits with 10.7 → 5.2 ms — maps across
rather than needing a redesign. That is the single best piece of news here.

Two things that *look* portable and are not, both caught only by compiling:

- `__builtin_amdgcn_s_memrealtime` — needs `s-memrealtime`, absent on gfx1250.
  91 call sites in the tree.
- `__builtin_amdgcn_global_load_lds` / `raw_ptr_buffer_load_lds` — need
  `vmem-to-lds-load-insts`, absent on gfx1250. 26 sites.

Async LDS copies complete against a **separate counter**: `s_wait_asynccnt`,
not `vmcnt`. Mixing the wait domains will corrupt silently rather than fail
loudly — the likeliest porting bug in the weight-prefetch path.

## Architectural changes beyond the ISA

From the MI450 kernel-development wiki page:

- **GL2 / XCD coherence changes the barrier story.** MI450 has 2 GL2s (one per
  AID, per 4 XCDs) with *hardware-managed* coherence, versus MI300/MI350 where
  L2 is not coherent across XCDs. Fleet's 240-way (30 workers × 8 XCDs)
  hierarchical barrier and `moe_ws_layout.cuh` are built around the
  non-coherent assumption. Correctness constraints relax; locality still
  matters, since remote-GL2 access costs "roughly the same as HBM."
- **LDS is 320 KB max but shares storage with GL0** (384 KB total, 64 KB
  minimum cache). `MAX_SMEM_SIZE = 160*1024` in `include/mirage/config.h:71` is
  now a cache-vs-LDS tradeoff, not a flat ceiling. `arch_traits.cuh` defaults to
  the 256 KB TCP maximum.
- **New hardware worth exploiting**: TDM (tensor DMA), split/named/cluster
  barriers, workgroup clusters, multicast (up to 5 clients/cycle), LDS
  transpose-on-load, WMMA `reuse_a`/`reuse_b` hints.
- **Attention scheduling**: head-first mapping with spatial swizzling is the
  recommended replacement for round-robin WG dispatch (arXiv:2511.02132).

## Use rocWMMA, not hand-rolled fragment layouts

rocWMMA ships in the gfx1250 toolchain, supports the target
(`ROCWMMA_ARCH_GFX1250` in `internal/config.hpp`), and its fragment API compiles
down to `v_wmma_f32_16x16x32_bf16`. It gets the per-lane operand and accumulator
layouts right without anyone having to re-derive them.

`tasks/mi450/gemm_handtuned_mi450.cuh` currently hand-rolls the layout. It
compiles and emits 16 WMMA ops for 16 K-steps (59 VGPRs, 14 SGPRs), but **the
fragment mapping is unvalidated and is the most likely thing in this tree to be
wrong**. Prefer rocWMMA for the remaining kernels and revisit this file.

## What exists so far

- `include/mirage/persistent_kernel/arch_traits.cuh` — wave size, LDS budget,
  timer, cross-lane reductions, async copy. Compiles clean for **both** gfx1250
  and gfx950; the reduction sweep differs by exactly one butterfly stage
  (9 vs 10 cross-lane ops), which is the wave32/wave64 split showing up in
  codegen as expected.
- `include/mirage/persistent_kernel/tasks/mi450/gemm_handtuned_mi450.cuh` —
  one GEMM ported to WMMA. Compiles. **Not executed.**

## FFM is running

Package: `rocdtif-10.1-am+ffmlite-mi400-r9.03` (ROCm 10.1.0a20260805, FFM-Lite
v9, model `mi400.9102703.540`) from
`atlartifactory.amd.com:8443/artifactory/SW-ROCDTIF-MI-DEV-LOCAL/Packages/AM+FFM-LITE/Release/`.
Extracted at `/home/claudeuser/rocdtif-10.1-am+ffmlite-mi400-r9.03`. It is
self-contained: its own ROCm, HIP runtime, `libhsakmtmodel.so`, and mi450
topology. Source its `ffmlite_env.sh` rather than setting variables by hand —
it sets `HSA_KMT_MODEL_GPUVM_BASE/SIZE` and the fast-copy hook that ad-hoc
setup misses.

**The glibc problem and how it is solved.** The model needs glibc >= 2.38; this
host is Ubuntu 22.04 with 2.35, and the loader aborts with `version GLIBC_2.38
not found`. Upgrading the host glibc would break Ubuntu 22.04 system-wide. So
instead there is an isolated Ubuntu 24.04 (glibc 2.39) sysroot at
`/home/claudeuser/noble-sysroot`, built with `debootstrap`, and binaries are run
against *its* loader. The host is unmodified. `./run_ffm.sh <abs-path-to-binary>`
wraps all of this.

Build test binaries against the FFM package's HIP runtime so the model
intercepts dispatches:

    clang++ -x hip --offload-arch=gfx1250 --rocm-path=$TOOLCHAIN -Iinclude ... \
        -L$FFM_PKG/rocm -lamdhip64 -Wl,-rpath,$FFM_PKG/rocm

The model reports the part correctly — `gfx1250` / `mi450`, 256 CUs,
**wavefront size 32**, which is wave32 confirmed by the model rather than
inferred from docs.

## Validated

`tests/mi450/test_gemm_wmma.hip` — **PASS** under FFM. All 1024 outputs match a
host fp32 reference exactly. Verified non-degenerate by pre-poisoning the output
buffer: 0 elements retained the poison, so every one was genuinely written, and
the far corner `[15,63]` matches (exercising all 4 waves and both accumulator
halves). FFM counters agree: `insts_waves=4`, `insts_valu_xdlmacc=64` = 16
K-steps × 4 waves.

The exact-zero error is expected, not suspicious: inputs are multiples of
0.125/0.25/0.5, so every product and partial sum is exactly representable and
bf16 never rounds.

**This settles the fragment-layout question** — the hand-derived per-lane
mapping in `gemm_handtuned_mi450.cuh` is correct.

## Running on real silicon

Do **not** reuse the FFM build flags. Three of them are FFM workarounds that are
wrong on hardware, and each fails silently rather than loudly:

| Flag | What it does on silicon |
|---|---|
| `-DMIRAGE_XCD_ID_FALLBACK=1` | Hardcodes `xcd_id()` to 0. The MoE tile decode is `tile_idx * 8 + xcd_id`, so 7 of every 8 tiles never execute while 8 XCDs redundantly compute the same tile. Wrong results, no error. |
| `-DMIRAGE_XCD_ID_FROM_BLOCKIDX_Y=1` | Test-only seam. `arch_traits.cuh` says "never define this in a shipping build". |
| `-DMPK_MAX_RESIDENT_BLOCKS=<n>` | Clamps workers to whatever FFM could keep co-resident (9). Silicon has no such ceiling; leaving it defined looks like a performance mystery, not a build error. |

Use `./build_mi450_hw.sh` (which omits all three) and `./run_mi450_hw.sh`, which
walks the bring-up in order: `check` → `tick` → `xcd` → `e2e`.

**The XCD id path has never been executed anywhere.** `xcd_id()` on gfx1250 uses
`__builtin_amdgcn_s_sendmsg_rtn(0x87)` (RTN_GET_SE_HW_ID; `data[3:0]=SE_ID`,
`data[19:16]=Virtual_XCC_ID`). The encoding matches Table 28 of the MI400 Shader
Programming Guide and it compiles — verified, the real path emits
`s_sendmsg_rtn_b32` once the fallback is dropped — but **FFM-Lite cannot decode
the instruction at all**, aborting with `Failed to decode instruction:
s_sendmsg_rtn_b32 ... MSG_RTN_GET_SE_HW_ID`. So no run has ever observed its
return value. `run_mi450_hw.sh xcd` is its first execution. Note the compiler
disassembles 0x87 as `MSG_RTN_GET_SE_AID_ID` while FFM names it
`MSG_RTN_GET_SE_HW_ID` — same opcode, two names, so check the value against
reality rather than trusting the field layout.

Related: the runtime hardcodes **8** XCDs (`NUM_XCDS_INIT`, `NUM_XCDS_PC`),
inherited from MI300X. FFM's mi450 topology reports `num_xcc 1`. If the real part
is not 8, those constants are wrong and the tile decode skips or duplicates tiles
silently.

`MIRAGE_TICK_NS` is still the gfx950 value (10) as an admitted placeholder —
`hipDeviceAttributeWallClockRate` returns 0 under FFM. Run
`tests/mi450/hw/calibrate_tick_ns.hip` on the part and rebuild with
`TICK_NS=<n>`. It cross-checks a reported rate against a measured one and fails
if they disagree by >5%, rather than trusting either alone.

## There is a second model: AM (cycle-accurate), and it does 8 XCCs

The package is `rocdtif-...-am+ffmlite-...`: **AM and FFM-Lite are two different
models** shipping together, and the port only ever used FFM-Lite. This matters
because several things recorded as "silicon only" are not.

| | FFM-Lite | AM |
|---|---|---|
| Fidelity | functional only, no cycles | cycle-accurate |
| XCCs | `num_xcc 1` in its mi450 topology | `DtifNumXcc=8` via `am_8xcc_env.sh` |
| `s_sendmsg_rtn 0x87` | **cannot decode**, aborts loudly | decodes, then **hangs silently** (see below) |
| Everything else | runs | **runs** — AM is usable today, see the end of this section |
| Speed | e2e in ~4 min | ~4 min just to first dispatch |

Env scripts: `am_env.sh` (standard), `am_fast_env.sh` (raw counters only),
`am_profile_env.sh` (per-dispatch counters), `am_8xcc_env.sh` (8 XCC).

Verified here: AM starts, translates gfx1250, and reaches
`Execute Dispatch on pipe 2` on **gpu0 through gpu7** — eight XCCs actually
dispatching. It emits `[rj warn] gfx1250 translation passes through 's_sleep'
unchanged; target-specific handling is not yet implemented`, so gfx1250 support
is real but not complete; treat AM timing as indicative, not authoritative.

Two gotchas cost real time:

- **AM needs `m4`** (`sudo apt install m4`), listed in the README prerequisites.
  Without it the failure is `Failed to preprocess parameter file ... exit code
  32512` followed by a fatal assert, which does not look like a missing package.
- The env scripts reference unset variables, so a launcher wrapper using
  `set -u` dies silently inside `source`. Use `set -o pipefail` only.

This makes AM the right vehicle for the two open silicon questions — the
never-executed `xcd_id()` path (#16) and cross-XCD tile distribution — since it
is the only thing here that models more than one XCC. It does not replace
silicon for absolute performance.

### Attempted on AM: `xcd_id()` — hangs, and the control proves it is the instruction

I ran the XCD probe (8 blocks, one `s_sendmsg_rtn(0x87)` per block) under
`am_8xcc_env.sh`. **It never completed** and produced no probe output. AM's own
watchdog tells the story — it retires instructions for two windows and then goes
permanently idle with the host still spinning:

    ts=5549445   sclk=10000  draw=2  inst=104  glx=16
    ts=11099445  sclk=20000  draw=5  inst=260  glx=84
    ts=16649445  sclk=30000  draw=0  inst=0    glx=0
    ... 19 more windows, all inst=0 ...

Both dispatches issue on gpu0–gpu7, work starts, and then execution stops
without an error, an assert, or a decode complaint. Killed after 18 minutes at
21 idle watchdog rounds.

**A control run makes this attributable.** `/tmp/hwprobe/amctl.hip` is the same
program — 8 blocks, 64 threads, one store per block — with the `sendmsg` line
replaced by a constant. Under the identical `run_am.sh` invocation it **passes**:

    [ctl] sync returned: no error
    [ctl] out[0]=0xa5000000 out[7]=0xa5000007  OVERALL: PASS

two dispatches, 2 watchdog rounds, done in ~5 minutes of wall clock. The same
binary also passes under FFM in seconds. So AM *can* run a gfx1250 HIP program
to completion on 8 XCCs; what it cannot do is retire `s_sendmsg_rtn 0x87`.

This corrects the row in the table above. AM does not reject the instruction the
way FFM does — FFM aborts at decode with a clear message, AM accepts it and then
**hangs silently**, which is the worse of the two failure modes. Take
"accepted, dispatches fine" to mean the decoder accepts it, not that it executes.

**`xcd_id()` therefore remains unexecuted anywhere, and #16 stays open for
silicon.** Neither model runs it. Do not read AM's willingness to compile and
dispatch the probe as evidence the path works. This is the same trap as
[[uninstantiated-templates-hide-asm-errors]] one level up: getting further into
the pipeline is not the same as producing a value. No XCC id has ever been
observed on gfx1250.

Cross-XCD *tile distribution* is still worth testing on AM, since that only
needs 8 XCCs and not the sendmsg path — drive it with
`-DMIRAGE_XCD_ID_FROM_BLOCKIDX_Y=1` and `gridDim.y == 8`, which substitutes a
known XCD id and exercises the distribution logic without the hanging
instruction. That is a real pre-silicon check the port has not yet run.

Repro: `AM_RUNDIR=/tmp/am_run8 ./run_am.sh /tmp/hwprobe/xcdprobe8 8` (hangs)
versus `AM_RUNDIR=/tmp/am_ctl ./run_am.sh /tmp/hwprobe/amctl 8` (passes).

### Trying to fix the AM hang: what was ruled out, and where it actually stops

Four hypotheses, each tested rather than argued:

**1. Is it 0x87 specifically, or `sendmsg_rtn` in general?** A differential probe
(`/tmp/hwprobe/msgdiff.hip`, one kernel per message, selected by argv) says
neither model is simply "missing 0x87":

| message | FFM-Lite | AM (8 XCC) |
|---|---|---|
| none (control, same binary) | PASS `0xc0ffee` | PASS `0xc0ffee` |
| `0x80` GET_DOORBELL | **PASS, returns 1** | hangs |
| `0x83` GET_REALTIME | abort: cannot decode | (not run) |
| `0x87` GET_SE_AID_ID | abort: cannot decode | hangs |

Two corrections fall out. FFM **can** execute `s_sendmsg_rtn` — doorbell works —
so "FFM cannot do sendmsg" was too broad; it lacks specific messages. And AM
hangs on doorbell too, so its problem is not 0x87 either. The `none` control
passing on *the same binary* is what makes both attributable to the instruction
rather than to the program or the launcher.

**2. Is the handler missing from AM?** No — and this is worth knowing. The
handler is in `libsqg_emu.so`, and its jump table covers 0x80–0x88:

    sqg::SqgFuncModel::ProcessMsgRtnCommand    # switch on msg, 0x80..0x88
      msg 0x87 -> mov r9d,[rbp+0x1310]; mov eax,[rbp+0x12fc]; shl r9,0x10
                  -> generate_and_send_sdata_rtn

That is `(XCC << 16) | SE` — **AM's own model independently confirms the field
layout** `data[19:16]=XCC`, `data[3:0]=SE`, matching Table 28 and matching what
`xcd_id()` decodes. Nice corroboration from a third source, but it means the
hang is in the SQ→SPI→SQG round trip (SQ logs `Queuing SQ_SPI_MSG_RTN.`, the
reply never lands), not a missing case.

**3. Is it the multithreaded model losing the reply?** No. `AM_SINGLE_THREAD=1`
(new flag: `AM_CLOCK_MT=0` plus `test.enable_{wgp,sa,se}_mt=false`) hangs the
same way.

**4. Is it the 8-XCC topology?** No. Plain `AM_ENV=am_env.sh` (1 XCC) also hangs.

**Conclusion: not fixable from here.** The break is inside prebuilt,
closed-source model libraries (`libsh_funclib.so`, `libspi_emu.so`,
`libsqg_emu.so`) with no config knob exposed for it. Fixing it means a vendor
build. What this investigation did produce is a precise bug report — *the SQ
queues `SQ_SPI_MSG_RTN` and no reply is ever delivered, for `0x80` as well as
`0x87`, at 1 XCC and 8, single- and multi-threaded* — plus a two-line repro.

Two things did get fixed in the process, both in `run_am.sh`:

- **`AM_DEBUG=1`** turns on the model's own logging. This has to be set *after*
  sourcing the env script: `am_8xcc_env.sh:101` unconditionally exports
  `DtifPrintCondLog=0 DtifPrintTerse=1`, so setting it in the environment ahead
  of the wrapper is silently overwritten and you get a quiet log while believing
  debug is on. Cost a wasted run before I noticed.
- **`AM_SINGLE_THREAD=1`** for troubleshooting, per AM_README §9.

### What this unblocks anyway: `tests/mi450/test_xcd_tile_coverage.hip`

The reason `xcd_id()` matters is the MoE decode `global_tile = tile_idx*8 + xcd_id()`.
That has **two** unknowns: does `xcd_id()` return the right value, and is the
tile arithmetic right? The first is silicon-only. The second is not, and was
never tested — under the constant-0 fallback only multiples of 8 are ever
produced, so 7 of every 8 tiles silently go unexecuted.

The new test substitutes `blockIdx.y` for the XCD id and launches `gridDim.y==8`,
then asserts every tile in `[0, total_tiles)` is produced **exactly once** —
catching gaps (skipped work) and duplicates (doubled accumulation) alike. It
carries an `#error` if built without the seam, and a built-in non-vacuity
control that re-runs with a single XCD and fails the test if that *also* reports
full coverage.

    ./build_mi450_hw.sh tests/mi450/test_xcd_tile_coverage.hip \
        -DMIRAGE_XCD_ID_FROM_BLOCKIDX_Y=1 -o /tmp/xcdcov
    ./run_ffm.sh /tmp/xcdcov          # 8 experts x 2 tiles
    ./run_ffm.sh /tmp/xcdcov 5 3      # ragged: 15 tiles, exercises the guard

### So does anything run on AM? Yes — everything except `s_sendmsg_rtn`

Worth stating plainly, because the section above reads like AM is a dead end and
it is not. The same binary that hangs when it issues `s_sendmsg_rtn` runs to
completion on AM when it does not. The tile-coverage test above passes on AM,
8 XCCs, at both shapes:

    AM_RUNDIR=/tmp/am_tile  ./run_am.sh /tmp/xcdcov 8 2   # covered=16 gaps=0 duplicates=0, PASS
    AM_RUNDIR=/tmp/am_tile2 ./run_am.sh /tmp/xcdcov 5 3   # covered=15 gaps=0 duplicates=0, PASS

both exit 0, with `model.gpu0..gpu7` present in the log, i.e. AM really did
model 8 XCCs rather than collapsing to one. That is a stronger result than the
FFM run of the same test: FFM models a single die, so its "8 XCDs" are 8 values
of `blockIdx.y` on one shader engine, whereas AM dispatched them across eight
modeled XCCs. The tile arithmetic is now confirmed on the multi-XCC model too.

The boundary is therefore narrow and worth remembering as one line: **AM runs
gfx1250 code fine; it hangs the wave on `s_sendmsg_rtn`, any message.** Kernels
that avoid that instruction — which is everything in the port except
`xcd_id()`/`__smid()` — are testable on the cycle-accurate model today. Build
them with `-DMIRAGE_XCD_ID_FALLBACK=1` or the `blockIdx.y` seam and AM is
available; the only thing gated on a vendor fix is `xcd_id()` itself, and that
is separately gated on silicon anyway.

Budget ~4 minutes of wall clock per run before the first dispatch, and prefer
small grids.

Both pass; the control leaves 14/16 and 13/15 tiles uncovered respectively, so
the test demonstrably discriminates. This runs under **FFM**, in seconds — AM
turned out not to be needed for it at all.

It reduces #16 from two unknowns to one. It does **not** show `xcd_id()` returns
a correct value, that the blocks are on different XCDs, or that 8 is the right
XCD count for the part (`NUM_XCDS_PC` is still a hardcoded MI300X inheritance).
Those remain silicon-only.

## Capture/replay (.cap) — the intended flow, currently blocked

`tools/roccap` implements capture/replay, and `tools/roccap/README.md` §4
documents exactly the workflow you would want: **capture on FFM-Lite, play back
on AM**, so the slow cycle-accurate model only replays a recorded AQL packet
trace instead of running the whole app.

    roccap capture <app>     # -> roc_capture_<app>.cap
    roccap extract -l <cap>
    roccap play -r <ranges> <cap>

`tools/rocdtif_ttrace_and_perfreport` wraps this end to end (capture →
disassemble → replay → perf report), with `--disp "*gemm/0-9"` to capture only
selected dispatches.

**It does not work in this environment.** `roccap capture` fails with
`Cannot find capture library. Set ROC_CAPTURE_LIB to pathname of
libcap_hsa64.so.4.6.2`. That message is misleading: the file is present and
`ROC_CAPTURE_LIB` is honoured. The real cause is a glibc mismatch *inside the
package* — `libcap_hsa64.so.4.6.2` needs `GLIBC_2.36`/`2.38` and `GLIBCXX_3.4.32`,
while the `roccap` binary itself links against host glibc 2.35 and satisfies all
its own deps there. So roccap runs on the host loader and then cannot `dlopen`
its own newer capture library, reporting the version error as "cannot find".
Confirmed by dlopen'ing it directly: FAILS under the host loader with
`GLIBC_2.36 not found`, LOADS OK under the noble sysroot loader. Re-exec'ing
roccap under the sysroot loader does not fix it (the sysroot's libc then
conflicts with the host-linked binary).

Fixing this properly means a host with glibc >= 2.38, or a container image built
on Ubuntu 24.04, rather than the sysroot-loader shim used for FFM. Until then,
run tests directly under FFM-Lite or AM; no .cap step is required for either.

## FFM can check more than first assumed

Two things previously recorded as "silicon only" turn out to be testable, via
`ffm_config.toml` options passed in `HSA_MODEL_ARGS`:

    HSA_MODEL_ARGS="ffm_enable_time_slicing ffm_barrier_checks_strict ffm_lds_oob_strict"

The e2e megakernel **passes** under both. Verified non-vacuous: a deliberate
64-float LDS array read at index 4000 passes silently without the flag and
asserts with it (`Strict bounds check: LDS Read extends beyond bounds`). So the
LDS bounds and barrier-misuse classes are now covered pre-silicon.

Three caveats found the hard way:

- **The separator is a space, not a comma.** `HSA_MODEL_ARGS="a,b"` matches no
  section, prints one `Warning:` line, and silently runs with *no* options —
  including dropping `ffm_enable_time_slicing`, which makes the megakernel hang.
  That hang looks exactly like a sync bug and is not one.
- `ffm_exec_order_random` (the mode that would shake out ordering races) is
  **mutually exclusive** with `ffm_enable_time_slicing`, which the persistent
  kernel requires. So wave-order fuzzing remains unavailable for MPK.
- `ffm_validate_gprs_strict` fires immediately (`uninitialized SGPR [S096]`) and
  so `ffm_strict_mode`, which enables it, is unusable here. Whether that is a
  real defect or the model not tracking a register the megakernel legitimately
  sets is unresolved — worth a look, but it is not the barrier/LDS result above.

`FFM_OBSERVER_PLUGINS=<path>/libasync_data_hazard_plugin.so` loads the data
hazard plugin but produced no output on the e2e run, and a deliberately bogus
path produced no error either — so that plugin is **not** yet demonstrated to be
active and its silence should not be read as a pass.

## Not done

No performance data, and FFM cannot provide any — it models no cycles,
bandwidth, or contention, and executes workgroups serially. Correctness only;
performance work needs a different vehicle.

Still unported: 49 remaining files in `tasks/mi300/`, the 91 `s_memrealtime`
sites, the 26 `buffer_load_lds` sites, `MAX_SMEM_SIZE` in
`include/mirage/config.h:71`, and the 240-way XCD barrier (which can relax given
hardware-coherent GL2).

FFM is slow enough that the wiki's own GPT-OSS runs trim the model to 2 layers
with `--load-format dummy`; their posted benchmark log shows 30 hours of wall
clock for 5 requests, all of which failed. Budget accordingly.

The **Data Hazard Plugin** (`FFM_OBSERVER_PLUGINS`) reports missing `s_wait_*`
synchronization as JSON. For a megakernel this sync-heavy, and with the new
`asynccnt` wait domain in play, that is worth wiring up early.
