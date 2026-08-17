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

## Not done

Nothing has been run. No numerical validation, no FFM run, no performance data.
The next concrete step is a reference-vs-WMMA correctness check for the single
GEMM under FFM, which is what turns the layout question from a guess into a
fact. FFM is a *functional* model — it can answer "does this compute the right
values" and cannot answer "is this fast."

## FFM setup (from the MI450 validation wiki page)

    export FFM_PATH=<extracted ffm-lite>
    export HSA_MODEL_LIB=$FFM_PATH/libhsakmtmodel.so
    export HSA_MODEL_TOPOLOGY=$FFM_PATH/topology/mi450
    export TARGET_ARCH=gfx1250
    export HSA_ENABLE_SDMA=0 HSA_ENABLE_INTERRUPT=0
    export HSA_MODEL_NUM_THREADS=256

Package: `atlartifactory.amd.com:8443/artifactory/gfxip-mi400-dev-local/gfxip/mi400/main/jitcu/mi450/master/`

FFM is slow enough that the wiki's own GPT-OSS runs trim the model to 2 layers
with `--load-format dummy`; their posted benchmark log shows 30 hours of wall
clock for 5 requests, all of which failed. Budget accordingly, and plan a
separate path for performance work.

The **Data Hazard Plugin** (`FFM_OBSERVER_PLUGINS`) reports missing `s_wait_*`
synchronization as JSON. For a megakernel this sync-heavy, and with the new
`asynccnt` wait domain in play, that is worth wiring up early.
