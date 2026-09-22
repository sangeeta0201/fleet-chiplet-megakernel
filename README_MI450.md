# Fleet on MI450 (gfx1250) -- how to build and run it

Status, prerequisites and the run recipes. For the porting analysis -- the
validated intrinsic mapping, the architectural deltas, and every dead end with
the measurement that closed it -- read **`PORTING_MI450.md`**.

> The scripts at the repo root (`run_ffm.sh`, `build_mi450_hw.sh`,
> `run_mi450_hw.sh`, `scripts/ffm_build_mi450.sh`) hardcode `/home/claudeuser`,
> the container the port was developed in. The tree has since moved, so those
> paths no longer resolve. **Use `tools/mi450-am/`**, which takes every path
> from one `env.sh`.

## What works today

| | status |
|---|---|
| 9 kernel harnesses under FFM-Lite | pass |
| megakernel e2e chain under FFM-Lite | pass (exact token 55) |
| 6 kernels under AM, with cycle counts | pass, see `tools/mi450-am/RESULTS.md` |
| attention / rmsnorm-linear-mxfp4 / moe-linear under AM | abort in an AM internal assert |
| any megakernel under AM | zero task dispatches |
| end-to-end decode latency | not available, see RESULTS.md |
| real silicon | never run; there is no gfx1250 hardware in these labs |

## Prerequisites

1. **gfx1250 toolchain.** ROCm 7.0 at `/opt/rocm` does *not* know gfx1250.
   Get a TheRock build:
   `https://rocm.genesis.amd.com/tarball/therock-dist-linux-gfx1250-7.12.0a20260226.tar.gz`
2. **The model package** `rocdtif-10.1-am+ffmlite-mi400-r9.03`, which ships both
   FFM-Lite and AM plus its own ROCm, HIP runtime, `libhsakmtmodel.so` and mi450
   topology. From artifactory under `Packages/AM+FFM-LITE/Release/`.
3. **`m4`** (`sudo apt install m4`). Without it AM fails to preprocess its
   parameter file, exits 32512 and then asserts -- which does not look like a
   missing package.
4. **glibc >= 2.38** on whatever runs the binary. A bare Ubuntu 24.04 host is
   fine and needs no sysroot. Only set `SYSROOT` if you are on something older,
   such as the Ubuntu 22.04 `mirage:fleetv2` container.

**No GPU is required.** FFM-Lite and AM are CPU-hosted models: `HSA_MODEL_LIB`
replaces the thunk, "VRAM" is a text property file, and parallelism is CPU
threads (`HSA_MODEL_NUM_THREADS`). You want cores and RAM, not an accelerator.

## Quickstart

```bash
cd tools/mi450-am
cp env.sh.example env.sh && $EDITOR env.sh     # FLEET, TOOLCHAIN, FFM_PKG

# one kernel, functional model (seconds)
./build.sh  "$FLEET/tests/mi450/test_gemm_wmma.hip" -o /tmp/gemm
./runffm.sh /tmp/gemm

# same kernel, cycle-accurate model (~20 s)
AM_RUNDIR=/tmp/am_gemm ./runam.sh /tmp/gemm
grep shader_execution_cycles /tmp/am_gemm/perf*_counters_absolute.txt

# the whole sweep: build + FFM validate + AM measure + table
./sweep.sh /tmp/mi450_sweep
```

The megakernel e2e chain uses its own build script, and the flag set is not
negotiable -- see the comment in `build_e2e.sh`:

```bash
./build_e2e.sh "$FLEET/tests/mi450/e2e/test_e2e_mi450.hip" -o /tmp/e2e
./runffm.sh /tmp/e2e 1        # expect: emitted token 55 ... OVERALL: PASS
```

## Traps worth knowing before you start

- **`-O3` is required for correctness**, not speed. At `-O0` the gfx1250 backend
  keeps `__shared__` accesses generic and FFM-Lite's flat sub-dword path returns
  the whole containing dword for a 1-byte `__shared__` read.
- **The megakernel build is schedule-sensitive.** Adding
  `-DMPK_HOST_BREADCRUMB -DMPK_MAX_RESIDENT_BLOCKS=9` to the e2e build flips the
  emitted token from 55 to 29, and so does `-DMIRAGE_REALTIME_FALLBACK=1`. This
  is a latent race (PORTING_MI450.md, task #21), not a build bug. Do not add
  flags to `build_e2e.sh`.
- **`ffm_enable_time_slicing` is mandatory** for anything persistent-kernel
  shaped; `runffm.sh` sets it. Without it FFM dispatches workgroups strictly
  serially and MPK's first rendezvous deadlocks.
- **Never quote a timing number from a build with `MIRAGE_REALTIME_FALLBACK`**,
  and note that FFM has no clock model at all.
- **Do not reuse the FFM flags on silicon.** `build_mi450_hw.sh` exists because
  three of them are wrong on hardware and each fails silently; its header
  explains each one.
