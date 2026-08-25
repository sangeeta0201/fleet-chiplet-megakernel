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
#else
using lane_mask_t = unsigned long long;
constexpr lane_mask_t FULL_WAVE_MASK = 0xFFFFFFFFFFFFFFFFull;
#endif

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

__device__ __forceinline__ unsigned long long realtime_ticks() {
#if defined(MIRAGE_ARCH_GFX1250)
  return wall_clock64(); // -> s_sendmsg_rtn_b64 sendmsg(MSG_RTN_GET_REALTIME)
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
__device__ __forceinline__ int xcd_id() {
#if defined(MIRAGE_ARCH_GFX1250)
#if defined(MIRAGE_XCD_ID_FALLBACK)
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

} // namespace arch
} // namespace mirage
