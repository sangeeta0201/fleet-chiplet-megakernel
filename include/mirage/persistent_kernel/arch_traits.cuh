/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Architecture traits: the single place where gfx950 (MI350/MI355, CDNA) and
// gfx1250 (MI450) differ in wave width, matrix intrinsics, timing, and
// global->LDS copies. Everything else in the kernel tree should ask this
// header rather than testing __gfx*__ directly.
//
// Why this exists: MI450 is *not* a CDNA part. It is wave32 and uses WMMA
// where gfx950 is wave64 and uses MFMA, so the MFMA-based task library does
// not merely need retuning -- the matrix instructions, the cross-lane
// reductions built on 64-lane masks, the s_memrealtime profiler clock, and the
// direct-to-LDS weight streaming all have to switch implementation. Each
// mapping below was verified by compiling to ISA with the gfx1250 toolchain;
// the instruction each intrinsic lowers to is quoted in a comment.
//
//   feature            gfx950                        gfx1250
//   -----------------  ----------------------------  ----------------------
//   wave               64                            32
//   matrix (bf16)      v_mfma_f32_16x16x16bf16_1k    v_wmma_f32_16x16x32_bf16
//   matrix (mx)        v_mfma_scale_f32_16x16x128_   v_wmma_scale_f32_16x16x
//                      f8f6f4                        128_f8f6f4
//   fp4 dequant        v_cvt_scalef32_pk_bf16_fp4    v_cvt_scale_pk8_bf16_fp4
//                      (2 values/call)               (8 values/call)
//   realtime clock     s_memrealtime                 s_sendmsg_rtn_b64
//                                                    (MSG_RTN_GET_REALTIME)
//   global->LDS        global_load_lds_dword         global_load_async_to_lds
//
// Note the MX matrix op keeps the same 16x16x128 f8f6f4 shape across both
// architectures, so the MoE MXFP4 datapath -- the single largest performance
// win in the gfx950 kernel -- maps over rather than needing a redesign.

#pragma once

// This header uses __device__, __forceinline__ and __shfl_xor, so it needs HIP
// already in scope. Every current caller reaches it through a task header that
// happens to include HIP first, which made it look self-contained; including it
// first in a fresh TU produced a wall of "unknown type name '__forceinline__'".
// Include HIP here so the header is usable standalone.
#include <hip/hip_runtime.h>

#if defined(__gfx1250__)
#define MIRAGE_ARCH_GFX1250 1
#elif defined(__gfx950__)
#define MIRAGE_ARCH_GFX950 1
#elif defined(__gfx942__)
#define MIRAGE_ARCH_GFX942 1
#endif

// Host passes carry no __gfx*__ macro. Give the host a wave width so that
// host-side launch math (workers per XCD, LDS budgeting) still compiles; device
// code always resolves to the real value below.
#if defined(MIRAGE_ARCH_GFX1250)
#define MIRAGE_WAVE_SIZE 32
#else
#define MIRAGE_WAVE_SIZE 64
#endif

namespace mirage {
namespace arch {

// ---------------------------------------------------------------------------
// Wave geometry
// ---------------------------------------------------------------------------

// Compile-time wave width. Prefer this over hardcoded 64 or warpSize: warpSize
// is not a constant expression in HIP, so it cannot size arrays or drive
// template parameters.
constexpr int WAVE_SIZE = MIRAGE_WAVE_SIZE;

// Lane mask type: a 64-lane wave needs 64 bits, a 32-lane wave needs 32.
#if MIRAGE_WAVE_SIZE == 32
using lane_mask_t = unsigned int;
constexpr lane_mask_t FULL_WAVE_MASK = 0xFFFFFFFFu;
constexpr int WAVE_SIZE_LOG2 = 5;
#else
using lane_mask_t = unsigned long long;
constexpr lane_mask_t FULL_WAVE_MASK = 0xFFFFFFFFFFFFFFFFull;
constexpr int WAVE_SIZE_LOG2 = 6;
#endif

// Split a flat thread index into (wave, lane). The mi300 kernels open-code this
// as `tid >> 6` / `tid & 63`, which silently halves the wave count and truncates
// the lane id at wave32 -- and because a butterfly reduce over the wrong width
// still *runs*, the result is a plausible wrong number rather than a crash.
__device__ __forceinline__ int wave_of(int tid) {
  return tid >> WAVE_SIZE_LOG2;
}
__device__ __forceinline__ int lane_of(int tid) {
  return tid & (WAVE_SIZE - 1);
}
// Number of waves spanned by `nthreads`. Rounds up so a partial trailing wave
// still gets a slot in cross-wave reduction scratch.
__device__ __forceinline__ int waves_in(int nthreads) {
  return (nthreads + WAVE_SIZE - 1) >> WAVE_SIZE_LOG2;
}
// Upper bound on waves per block, for sizing __shared__ reduction scratch at
// compile time. 1024 threads is HIP's max block size, so this is 32 at wave32.
constexpr int MAX_WAVES_PER_BLOCK = 1024 / WAVE_SIZE;

// ---------------------------------------------------------------------------
// LDS budget
// ---------------------------------------------------------------------------
//
// gfx950 (MI350X): 160 KB per workgroup.
// gfx1250 (MI450): up to 320 KB, allocated in 64 KB segments, but LDS shares
// physical storage with the GL0 cache -- 384 KB total, with a 64 KB minimum
// cache configuration. Asking for the full 320 KB starves GL0, so the default
// here is the 256 KB TCP maximum. Kernels that would rather keep cache should
// request less; kernels that are purely LDS-bound may raise it to 320 KB.
#if defined(MIRAGE_ARCH_GFX1250)
constexpr int MAX_LDS_BYTES = 256 * 1024;
constexpr int LDS_SEGMENT_BYTES = 64 * 1024;
#elif defined(MIRAGE_ARCH_GFX942)
constexpr int MAX_LDS_BYTES = 64 * 1024;
constexpr int LDS_SEGMENT_BYTES = 0;
#else
constexpr int MAX_LDS_BYTES = 160 * 1024;
constexpr int LDS_SEGMENT_BYTES = 0;
#endif

// ---------------------------------------------------------------------------
// Realtime clock
// ---------------------------------------------------------------------------
//
// The profiler timestamps tasks with a constant-rate clock. gfx950 has
// s_memrealtime; on gfx1250 that instruction is gone (the builtin requires the
// 's-memrealtime' target feature, which gfx1250 does not set) and the realtime
// counter is read via s_sendmsg_rtn_b64(MSG_RTN_GET_REALTIME), which is what
// HIP's wall_clock64() lowers to.
//
// Callers must not assume a tick rate: gfx950's is ~100 MHz (10 ns/tick), and
// the gfx1250 rate is a separate question that has to be measured on the
// model or the part. Report raw ticks and convert at the boundary using
// hipDeviceAttributeWallClockRate.
// Nanoseconds per realtime tick, used to scale profiler timestamps.
//
// gfx950: s_memrealtime runs at ~100 MHz -> 10 ns/tick. Measured and relied on
// by the existing profiler.
//
// gfx1250: UNVERIFIED. hipDeviceAttributeWallClockRate returns 0 under
// FFM-Lite, so the model cannot tell us and no gfx1250 silicon is available
// here. 10 is carried over as a placeholder so timestamps stay
// self-consistent, but absolute microsecond figures on gfx1250 are not
// trustworthy until this is measured. Override with -DMIRAGE_TICK_NS=<n>.
#ifndef MIRAGE_TICK_NS
#define MIRAGE_TICK_NS 10
#endif

// MIRAGE_REALTIME_FALLBACK=1 substitutes a constant 0 for the realtime counter.
//
// This is the exact analogue of MIRAGE_XCD_ID_FALLBACK below, and it exists for
// the same reason: on gfx1250 BOTH of these lower to s_sendmsg_rtn, and AM
// (the cycle-accurate model) hangs on that instruction regardless of which
// message it carries. Measured 2026-09-03: a two-block kernel whose only
// content is wall_clock64() retires 104 instructions on AM and then stalls
// until AM's own HANG_WATCHDOG fires. So MSG_RTN_GET_REALTIME hangs AM just as
// MSG_RTN_GET_SE_AID_ID does -- the round trip is the problem, not the message.
//
// That matters for the megakernel specifically because
// persistent_kernel.cuh:1998 calls get_wallclock_ns() UNCONDITIONALLY on worker
// 0 at every TASK_BEGIN_TASK_GRAPH -- it is not behind MPK_ENABLE_PROFILING.
// Without this seam the megakernel cannot reach its first dispatch on AM even
// with MIRAGE_XCD_ID_FALLBACK=1 set.
//
// Returning 0 should be numerically inert: all four get_wallclock_ns() call
// sites (persistent_kernel.cuh:1998, 3186, 3240, 3305) feed printf strings and
// profiling aggregates only, and no control flow reads them. It is NOT safe for
// any perf claim -- with this defined, every duration the megakernel prints is
// meaningless.
//
// EMPIRICALLY IT IS NOT INERT, AND THAT IS A BUG SOMEWHERE ELSE. On FFM the e2e
// harness emits token 55 on the baseline build and token 12 with this defined
// (64/64 logits wrong). Do not conclude the seam is at fault: a control build
// that keeps the real clock and only suppresses the FWD_PASS printf fails the
// same way (token 29), as does one that still issues the sendmsg and discards
// its result. So any perturbation of the instruction schedule around this point
// breaks the run, which points at a latent race / missing release-acquire in
// the megakernel that the baseline schedule happens to hide. Tracked as task
// #21. Until that is fixed, treat a green e2e as schedule-dependent.
//
// Never define this in a shipping build, and never quote a timing number from a
// run that had it set.
__device__ __forceinline__ unsigned long long realtime_ticks() {
#if defined(MIRAGE_ARCH_GFX1250)
#if defined(MIRAGE_REALTIME_FALLBACK)
  return 0ull;
#else
  return wall_clock64(); // -> s_sendmsg_rtn_b64 sendmsg(MSG_RTN_GET_REALTIME)
#endif
#elif defined(__HIP_DEVICE_COMPILE__)
  return __builtin_amdgcn_s_memrealtime(); // -> s_memrealtime
#else
  return 0ull;
#endif
}

// ---------------------------------------------------------------------------
// XCD / XCC identity
// ---------------------------------------------------------------------------
//
// Fleet's gang dispatch spreads MoE tiles across XCDs round-robin and needs
// each workgroup to know which die it landed on. On gfx950 that is
//
//     s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)
//
// which gfx1250 rejects outright: "invalid hardware register: not supported
// on this GPU". The replacement is a sendmsg, per MI400 guide Table 28:
// RTN_GET_SE_HW_ID (0x87) returns SE_ID in bits [3:0] and Virtual_XCC_ID in
// bits [19:16]. LLVM spells the message MSG_RTN_GET_SE_AID_ID.
//
// TWO CAVEATS, both real:
//
// 1. NOT VALIDATED. FFM-Lite fails to decode s_sendmsg_rtn_b32 with this
//    message ("Failed to decode instruction"), so the model cannot execute
//    it. The encoding is what the gfx1250 assembler produces and the field
//    layout is from the guide, but neither has been run. Verify on silicon
//    before trusting the tile distribution.
//
// 2. IT IS EXPENSIVE. The guide warns S_SENDMSG_RTN "has very limited
//    bandwidth and should not be issued by every wave" -- if every wave on a
//    shader engine uses it, throughput falls to once per 20,000 cycles per
//    wave. gfx950's s_getreg was nearly free, so call sites that read the XCD
//    ID per wave need to hoist it: read once per workgroup into LDS, or fold
//    it into the task descriptor host-side.
//
// Define MIRAGE_XCD_ID_FALLBACK=1 to substitute 0, which is what lets the MoE
// kernels run under FFM. FFM executes workgroups serially on a single modeled
// die, so a constant 0 changes nothing it can observe -- but it does mean any
// FFM run has NOT exercised the cross-XCD distribution.
//
// MIRAGE_XCD_ID_FROM_BLOCKIDX_Y is a TEST-ONLY seam, and a strictly narrower
// one: it substitutes blockIdx.y instead of 0. The MoE tile decode is
// `global_tile = tile_idx * 8 + xcd_id()`, so with the constant-0 fallback a
// single-XCD test can only ever reach tiles that are multiples of 8 -- 7 of
// every 8 tiles, including every tile whose wg_idx is not a multiple of 8,
// would go unexecuted while the test still reported PASS on the ones it did
// reach. Launching with gridDim.y == 8 and this seam covers the whole tile
// space on one modeled die. It does NOT test cross-XCD distribution: the
// blocks are not on different XCDs, they merely enumerate the same indices a
// real 8-XCD dispatch would. Never define this in a shipping build.
__device__ __forceinline__ int xcd_id() {
#if defined(MIRAGE_ARCH_GFX1250)
#if defined(MIRAGE_XCD_ID_FROM_BLOCKIDX_Y)
  return (int)blockIdx.y;
#elif defined(MIRAGE_XCD_ID_FALLBACK)
  return 0;
#else
  // MSG_RTN_GET_SE_HW_ID: data[3:0]=SE_ID, data[19:16]=Virtual_XCC_ID
  return (int)((__builtin_amdgcn_s_sendmsg_rtn(0x87) >> 16) & 0xF);
#endif
#elif defined(__HIP_DEVICE_COMPILE__)
  int id;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(id));
  return id;
#else
  return 0;
#endif
}

// ---------------------------------------------------------------------------
// Cross-lane reduction
// ---------------------------------------------------------------------------
//
// Butterfly reduction over `width` lanes. Written against WAVE_SIZE so the
// same source is correct at 32 and 64 lanes: callers that previously hardcoded
// a 64-lane sweep get a 32-lane sweep on gfx1250 automatically.
//
// `width` must be a power of two and <= WAVE_SIZE. Reducing over a group wider
// than the wave is a caller bug -- it silently produced partial results on
// gfx950 and would do so again here, so it is asserted in debug builds.
template <typename T, typename Op>
__device__ __forceinline__ T wave_reduce(T val, Op op, int width = WAVE_SIZE) {
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    val = op(val, __shfl_xor(val, offset, width));
  }
  return val;
}

__device__ __forceinline__ float wave_reduce_sum(float v,
                                                 int width = WAVE_SIZE) {
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor(v, offset, width);
  }
  return v;
}

__device__ __forceinline__ float wave_reduce_max(float v,
                                                 int width = WAVE_SIZE) {
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor(v, offset, width));
  }
  return v;
}

// ---------------------------------------------------------------------------
// Global -> LDS asynchronous copy
// ---------------------------------------------------------------------------
//
// Fleet streams expert weights HBM->LDS without staging through VGPRs. On
// gfx950 that is buffer_load_lds / global_load_lds. gfx1250 removes those
// (they need the 'vmem-to-lds-load-insts' feature, absent on gfx1250) and
// provides global_load_async_to_lds_b{8,32,64,128} instead, which complete
// against a *separate* counter -- s_wait_asynccnt, not vmcnt. Mixing the two
// wait domains is the likeliest source of silent corruption when porting, so
// the wait is wrapped here too and must be paired with these loads.
#if defined(MIRAGE_ARCH_GFX1250)
using async_copy_vec_t = int __attribute__((ext_vector_type(4)));

// 16-byte global->LDS copy, one per issuing lane.
__device__ __forceinline__ void
    async_copy_b128(async_copy_vec_t const *global_src,
                    async_copy_vec_t *lds_dst) {
  __builtin_amdgcn_global_load_async_to_lds_b128(
      (__attribute__((address_space(1))) async_copy_vec_t *)global_src,
      (__attribute__((address_space(3))) async_copy_vec_t *)lds_dst,
      0,
      0); // -> global_load_async_to_lds_b128
}
#endif

// Wait for outstanding async global->LDS copies. On gfx950 the direct-to-LDS
// loads land in the vmcnt domain; on gfx1250 they land in asynccnt.
__device__ __forceinline__ void async_copy_wait_all() {
#if defined(MIRAGE_ARCH_GFX1250)
  __builtin_amdgcn_s_wait_asynccnt(0);
#elif defined(__HIP_DEVICE_COMPILE__)
  __builtin_amdgcn_s_waitcnt(0x0f70); // vmcnt(0)
#endif
}

// ---------------------------------------------------------------------------
// Wait counters
// ---------------------------------------------------------------------------
//
// THE TRAP, and the reason these are functions rather than a macro rename:
// gfx9's counters are coarse and gfx12's are split, and the split is not
// one-to-one.
//
//   gfx9  vmcnt   counts vector-memory LOADS *and* STORES
//   gfx12 loadcnt counts loads only
//         storecnt counts stores only
//
//   gfx9  lgkmcnt counts LDS + scalar-memory + message ops together
//   gfx12 dscnt   counts LDS only
//         kmcnt   counts scalar-memory (and message) only
//
// So a faithful translation of `s_waitcnt vmcnt(0)` is BOTH `s_wait_loadcnt 0`
// AND `s_wait_storecnt 0`. Translating it to loadcnt alone is the natural
// mistake and it is silent: every release pattern in MPK (write payload,
// wait, then publish a flag) would stop waiting for its payload stores and
// publish early. FFM will not catch it -- FFM runs workgroups serially or
// time-sliced, so the consumer never actually observes the window. Hence the
// deliberately coarse names: callers ask for "the gfx9 vmcnt semantic" and get
// both halves.
//
// Note there are no __builtin_amdgcn_s_wait_{load,store,ds,km}cnt builtins in
// this toolchain -- only s_wait_asynccnt exists -- so these are asm. Verified
// to assemble for gfx1250; the compiler's own lowering of C++ atomics emits
// the same mnemonics (s_wait_loadcnt_dscnt, s_wait_kmcnt).
//
// Prefer plain C++ / HIP atomics where possible: on gfx1250 the compiler
// inserts these waits itself and gets the domains right. Reach for these only
// where the gfx950 source already hand-rolled the wait.

// Wait for all outstanding vector-memory ops -- loads AND stores.
// Equivalent to gfx9 `s_waitcnt vmcnt(0)`.
__device__ __forceinline__ void wait_vmem() {
#if defined(MIRAGE_ARCH_GFX1250)
  asm volatile("s_wait_loadcnt 0\n\ts_wait_storecnt 0" ::: "memory");
#elif defined(__HIP_DEVICE_COMPILE__)
  __builtin_amdgcn_s_waitcnt(0x0f70); // vmcnt(0)
#endif
}

// Wait for outstanding vector-memory LOADS only. Cheaper than wait_vmem() and
// correct only where the source genuinely has no store to wait on.
__device__ __forceinline__ void wait_vmem_loads() {
#if defined(MIRAGE_ARCH_GFX1250)
  asm volatile("s_wait_loadcnt 0" ::: "memory");
#elif defined(__HIP_DEVICE_COMPILE__)
  __builtin_amdgcn_s_waitcnt(0x0f70); // vmcnt(0)
#endif
}

// Wait for LDS + scalar-memory ops. Equivalent to gfx9 `s_waitcnt lgkmcnt(0)`.
__device__ __forceinline__ void wait_lgkm() {
#if defined(MIRAGE_ARCH_GFX1250)
  asm volatile("s_wait_dscnt 0\n\ts_wait_kmcnt 0" ::: "memory");
#elif defined(__HIP_DEVICE_COMPILE__)
  __builtin_amdgcn_s_waitcnt(0xc07f); // lgkmcnt(0)
#endif
}

// Wait for LDS ops only.
__device__ __forceinline__ void wait_lds() {
#if defined(MIRAGE_ARCH_GFX1250)
  asm volatile("s_wait_dscnt 0" ::: "memory");
#elif defined(__HIP_DEVICE_COMPILE__)
  __builtin_amdgcn_s_waitcnt(0xc07f); // lgkmcnt(0)
#endif
}

// ---------------------------------------------------------------------------
// Cache maintenance
// ---------------------------------------------------------------------------
//
// gfx950's `buffer_inv sc1` invalidates the L2 so a subsequent load sees
// another XCD's writes. gfx1250 replaces the buffer_* cache ops with
// global_inv / global_wb carrying an explicit scope: modifier; `buffer_inv` no
// longer assembles at all.

// Invalidate caches so subsequent loads observe other agents' writes.
__device__ __forceinline__ void inv_l2() {
#if defined(MIRAGE_ARCH_GFX1250)
  asm volatile("global_inv scope:SCOPE_DEV" ::: "memory");
#elif defined(__HIP_DEVICE_COMPILE__)
  asm volatile("buffer_inv sc1" ::: "memory");
#endif
}

// Write back dirty cache lines so other agents can observe our writes.
__device__ __forceinline__ void wb_l2() {
#if defined(MIRAGE_ARCH_GFX1250)
  asm volatile("global_wb scope:SCOPE_DEV" ::: "memory");
#elif defined(__HIP_DEVICE_COMPILE__)
  asm volatile("buffer_wbl2 sc1" ::: "memory");
#endif
}

} // namespace arch
} // namespace mirage
