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

// MXFP8 twin of gang_gemv_mi300.cuh: the narrow-tile GEMV reading E4M3 weights
// with one E8M0 exponent per 32 contiguous K elements.
//
// GLM-4.7-Flash's absorbed o_proj is [2048, 10240] per layer, 42 MB bf16, and
// after the MoE, qkv_a, the LM head and q_b went to MXFP8 it is the last big
// bf16 GEMM left: 1.97 GB of the token's weight traffic and 37.5 of a 120 us
// layer. Halving its bytes is worth more than anything else on the list.
//
// The bf16 header says it is "deliberately not templated on a weight format",
// and that is still the right call -- this is a separate kernel rather than a
// template parameter, because nothing about the two bodies wants to share a
// loop. What it does share is everything outside the loop: tile addressing,
// thread mapping, the four accumulator chains, the __shfl_xor reduction and
// the bias/residual epilogue all come over unchanged, and gang_gemv_detail's
// helpers are reused directly rather than copied.
//
// Two things make the port cheap.
//
// The packing needs no rework. pack_dense_mxfp8 stores a workgroup as
// [ROWS_PER_WG][K] E4M3 bytes followed by [ROWS_PER_WG][K/32] E8M0 bytes, both
// plain row-major; the awkward split-gather layout the MXFP8 MFMA path needs
// lives in _gang_load_fp8_mfma_b, i.e. in how the operand is *gathered*, not
// in how it is stored. A scalar reader just walks the row.
//
// And the dequant is one instruction per bf16 pair.
// v_cvt_scalef32_pk_bf16_fp8 takes the E8M0 as an fp32 multiplier and emits a
// packed bf16 pair, which is exactly the operand fma_pair already consumes --
// so the arithmetic downstream of the load is byte-for-byte the bf16 kernel's.
// The conversion is lossless: E4M3 carries 3 mantissa bits against bf16's 7,
// and the scale only moves the exponent.
//
// A lane's 16-element chunk therefore always sits inside one 32-element scale
// block, so the whole iteration costs a single scale byte. Those bytes are
// K/32 of the traffic -- 3% -- and two lanes share each one, so they are read
// straight from global rather than staged through LDS; after the first touch
// they are L1 hits, and issuing them in the same batch as the weight loads
// keeps them off the dependency chain.
//
// The *activation* is the opposite case, and it was the kernel's real limit. A
// bandwidth probe over a 4 GiB weight buffer (tests/standalone/
// test_gemv_mxfp8_bw.hip) put a pure streaming read at 37.8 GB/s per workgroup
// at 128 workgroups while this loop managed 14.5, so the shortfall was inside
// the workgroup rather than in how many of them were running -- re-tiling
// would not have touched it. The cause is that all ROWS_PER_WG rows of a
// workgroup walk the *same* activation row, so the load below was issued once
// per row: at the o_proj tile that is 320 KB of requests against 164 KB of
// weight, contending for the same vmcnt queue as the loads that actually carry
// the weight. Filling LDS with the row once and reading it from there measured
// 18.7 GB/s/WG at 128 workgroups and 17.2 at 240, i.e. 1.30-1.32x, with the
// fill paid per tile as it is here rather than amortized over a block's whole
// share of them. Two variants that looked promising did not survive the probe:
// staging the scale half as well is consistently ~6% slower, and remapping the
// chunks so a lane's are contiguous is slower still -- the strided mapping is
// the coalesced one.
//
// In the megakernel that 1.30x came out as 2.9% on the o_proj subphase and
// nothing at all on the decode step, and the same probe explains why. Kernels
// A-F there all run a grid-stride loop over ~184 tiles, so one tile's drain
// overlaps the next one's ramp. gang_oproj_router_fused gives a worker exactly
// one tile per layer and then a cross-XCD barrier, so all 128 of them ramp and
// drain in lockstep and neither tail overlaps anything: the same kernel over
// the same bytes costs 8.21 us/tile back to back and 11.98 us/tile one at a
// time (kernel G). Deeper unrolling and pipelining the fill against the first
// weight batch were both measured (kernel H) and both did nothing, which is
// what being shape-bound rather than latency-bound looks like. So what is left
// on this kernel is small; the factor of two is in how much consecutive work a
// worker gets between barriers.

#pragma once
#include "tasks/mi300/gang_gemv_mi300.cuh"
#include "tasks/mi300/mpk_bsdbg.cuh"

namespace kernel {

namespace gang_gemv_mxfp8_detail {
typedef __bf16 __attribute__((ext_vector_type(2))) bf16x2_t;
// A POD 16-byte vector, used instead of uint4 for anything that is loaded
// through an addrspace(1) pointer: uint4 is HIP_vector_type, a class, so
// dereferencing an addrspace(1) uint4* runs its copy constructor -- which
// takes a generic `uint4 const&` -- and the cast is undone before the load
// ever happens. The ext_vector_type is loaded directly and keeps .x/.y/.z/.w.
typedef unsigned int __attribute__((ext_vector_type(4))) u32x4_t;

// E8M0 byte to its fp32 value, 2^(e-127). e == 0 gives +0.0f rather than the
// 1.0f the packer's comment describes, which is harmless and deliberate: a
// zero exponent is only ever written for an all-zero block, and 0 * 0 == 0.
__device__ __forceinline__ float e8m0_to_f32(unsigned char e) {
  unsigned u = (unsigned)e << 23;
  float f;
  __builtin_memcpy(&f, &u, 4);
  return f;
}

// Two E4M3 bytes out of `raw` -- the low 16 bits when hi is false, the high 16
// when it is true -- scaled and packed as a bf16 pair in one 32-bit register,
// in the same order fma_pair expects (element 2i low, 2i+1 high).
//
// HI is a template parameter, not an argument: the builtin's word select has
// to be a literal.
template <bool HI>
__device__ __forceinline__ unsigned cvt_fp8_pair(unsigned raw, float scale) {
  bf16x2_t v = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(raw, scale, HI);
  unsigned u;
  __builtin_memcpy(&u, &v, 4);
  return u;
}

// A load that is *known* to come from device memory.
//
// Clang infers address spaces intraprocedurally, so a pointer that arrives as
// an argument to a __noinline__ kernel stays generic and every dereference of
// it is emitted as flat_load rather than global_load. On gfx9 that is not just
// a slower addressing mode: a flat instruction increments **both** vmcnt and
// lgkmcnt, so the `s_waitcnt lgkmcnt(0)` that retires an LDS read also waits
// on every outstanding weight load. That is why staging the activation in LDS
// bought only 1.08x here while the same change in a monolithic __global__
// probe -- where the compiler does see addrspace(1) -- bought 1.30x. Casting
// the pointer to addrspace(1) at the load restores global_load and decouples
// the two counters again.
//
// Safe only because every pointer handed to this kernel is device global:
// weights, activations, residual, bias and output all come from the
// megakernel's workspace or from a task descriptor.
// The cast is two-step because clang rejects a reinterpret_cast that changes
// both the pointee type and the address space at once.
template <typename T>
__device__ __forceinline__ T ld_g(void const *p) {
  T const *q = static_cast<T const *>(p);
  return *(__attribute__((address_space(1))) T const *)q;
}
} // namespace gang_gemv_mxfp8_detail

// Same signature and same tile addressing as gang_gemv_kernel, minus the
// element-type template parameter -- the weight is MXFP8 and the activation,
// residual, bias and output are all bf16.
//
// `weight_ptr` is this XCD's chunk of the workgroup-packed weight,
// [n_tiles, ROWS_PER_WG * (REDUCTION_SIZE + REDUCTION_SIZE/32)] bytes, so
// n_tile indexes a workgroup rather than a row block. As in the bf16 kernel
// the weight row stride is REDUCTION_SIZE while the *input* row stride need
// not be -- INPUT_ROW_STRIDE is that stride, and defaulting it to
// REDUCTION_SIZE is what used to pin this kernel to BATCH_SIZE == 1. W_UV is
// the caller that needs it: input_ptr is one head of attn_out, so the
// reduction is KV_LORA_RANK but the row is NUM_Q_HEADS * KV_LORA_RANK.
//
// WRITE_THROUGH sends the epilogue store past the XCD's L2 (sc0 sc1) instead
// of leaving it dirty there. A standalone task does not need it -- the
// scheduler's end-of-task fence makes the result visible before any consumer
// is dispatched -- but a *fused* caller that follows this GEMM with nothing
// more than an in-kernel barrier does: on MI300/MI350 the L2 is per-XCD and
// not coherent, so a plain store is invisible to the other seven XCDs no
// matter how the barrier is ordered. Same reason the MXFP4 O-proj writes
// st_wt_u64 and the split-KV merge takes a WRITE_THROUGH parameter.
template <int BATCH_SIZE, // = m_per_tile
          int REDUCTION_SIZE,
          int ROWS_PER_WG,
          bool HAS_RESIDUAL,
          bool WRITE_THROUGH = false,
          int INPUT_ROW_STRIDE = REDUCTION_SIZE>
__device__ __noinline__ void
    gang_gemv_mxfp8_kernel(void const *input_ptr,
                           void const *weight_ptr,
                           void const *residual_ptr, // may be null
                           void *output_ptr,
                           int num_active_tokens,
                           int tile_n,
                           int o_stride,
                           int m_tiles,
                           int n_tiles,
                           int wgm,
                           int tile_idx,
                           void const *bias_ptr = nullptr,
                           bool stage_a = true) {
  using gang_gemv_detail::b2f;
  using gang_gemv_detail::f2b;
  using gang_gemv_mxfp8_detail::cvt_fp8_pair;
  using gang_gemv_mxfp8_detail::e8m0_to_f32;
  using gang_gemv_mxfp8_detail::ld_g;
  using gang_gemv_mxfp8_detail::u32x4_t;

  constexpr int NTHREADS = 256;
  constexpr int LANES_PER_ROW = NTHREADS / ROWS_PER_WG;
  constexpr int VEC = 16; // E4M3 bytes per 16-byte load
  constexpr int SCALE_BLOCK = 32;
  static_assert(NTHREADS % ROWS_PER_WG == 0,
                "ROWS_PER_WG must divide the 256-thread block");
  static_assert(LANES_PER_ROW <= 64 && (64 % LANES_PER_ROW) == 0,
                "a row's lanes must be an aligned slice of one wavefront, so "
                "ROWS_PER_WG must be >= 4 and a power of two");
  static_assert(REDUCTION_SIZE % (LANES_PER_ROW * VEC) == 0,
                "K must tile evenly over the row's lanes at 16 fp8 each");
  static_assert(SCALE_BLOCK % VEC == 0,
                "a lane's chunk must sit inside one scale block, or it would "
                "need two exponents");
  static_assert((ROWS_PER_WG * (REDUCTION_SIZE / SCALE_BLOCK)) % 16 == 0,
                "the scale half must keep the next workgroup's data half "
                "16-byte aligned");
  constexpr int ITERS = REDUCTION_SIZE / (LANES_PER_ROW * VEC);
  constexpr int WG_DATA_BYTES = ROWS_PER_WG * REDUCTION_SIZE;
  constexpr int WG_BYTES =
      WG_DATA_BYTES + ROWS_PER_WG * (REDUCTION_SIZE / SCALE_BLOCK);

  assert(tile_idx >= 0);
  assert(tile_n == ROWS_PER_WG);
  // ── bs=2 bisection probe ────────────────────────────────────────────────
  // GLM_FUSE_ATTN defaults to 0, so the three dense prologue layers run this
  // UNFUSED chain and the whole-layer task covers every MoE layer -- which is
  // why the first attempt, a probe inside gang_mla_attn_fused_kernel_mi300,
  // saw fused layers 0-2 and not a single dense one. The dense layers are
  // scheduled first, so seq 0..23 (8 XCD tasks x 3 layers) is exactly the
  // prologue no matter who else calls this kernel later.
  // Read at task entry, i.e. behind the previous task's event boundary, and
  // only on the reduction-dim input, which every XCD sees whole -- an output
  // or residual pointer is column-sliced 8 ways and reading a full row off one
  // slice runs into the next row.
  if constexpr (HAS_RESIDUAL) {
    if (tile_idx == 0 && threadIdx.x == 0) {
      MPK_BSDBG_SEQ(25, input_ptr, REDUCTION_SIZE, "d_attn_out",
                    BATCH_SIZE, INPUT_ROW_STRIDE, 24);
    }
  }


  int m_tile, n_tile;
  if (!gang_linear_tile_coords(
          tile_idx, m_tiles, n_tiles, wgm, &m_tile, &n_tile)) {
    return;
  }

  unsigned short const *A = static_cast<unsigned short const *>(input_ptr);
  unsigned char const *W = static_cast<unsigned char const *>(weight_ptr);
  unsigned short const *R = static_cast<unsigned short const *>(residual_ptr);
  unsigned short const *Bs = static_cast<unsigned short const *>(bias_ptr);
  unsigned short *O = static_cast<unsigned short *>(output_ptr);

  size_t const out_off =
      static_cast<size_t>(m_tile) * BATCH_SIZE * o_stride +
      static_cast<size_t>(n_tile) * ROWS_PER_WG;
  unsigned short const *tile_input =
      A + static_cast<size_t>(m_tile) * BATCH_SIZE * INPUT_ROW_STRIDE;
  unsigned char const *wg = W + static_cast<size_t>(n_tile) * WG_BYTES;

  int const tid = threadIdx.x;
  int const row = tid / LANES_PER_ROW;
  int const lane = tid % LANES_PER_ROW;

  // Dynamic LDS rather than a __shared__ array: the workers are launched with
  // MAX_DYNAMIC_SHARED_MEMORY_SIZE and the persistent kernel runs one
  // workgroup per CU, so this space is already paid for and costs no
  // occupancy, whereas a static array would be added to the frame of every
  // kernel that can reach this one. gang_moe_fused_mxfp4 stages its tokens and
  // weights the same way. Both of this kernel's callers -- the standalone task
  // and gang_oproj_router_fused -- have nothing live in _fused_smem here.
  //
  // `stage_a` lets a caller that calls this kernel more than once for the same
  // activation row skip the copy after the first. o_proj is that caller: at
  // GLM-5's hidden 6144 it runs 48 tiles per XCD over ~29 workers, so 19
  // workers go round the grid-stride loop twice, and with m_tiles == 1 the
  // second tile's `tile_input` is the first tile's -- 64 KB of global reads and
  // 64 KB of ds_writes to rebuild a buffer that is already correct. The caller
  // owns the invariant: pass false only when this block already staged the same
  // `tile_input` and nothing has written _fused_smem since.
  constexpr size_t A_LDS_BYTES =
      sizeof(unsigned short) * BATCH_SIZE * REDUCTION_SIZE;
  constexpr bool STAGE_A =
      A_LDS_BYTES <=
      static_cast<size_t>(mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE);
  extern __shared__ char _fused_smem[];
  unsigned short *s_a = reinterpret_cast<unsigned short *>(_fused_smem);
  if constexpr (STAGE_A) {
    // The tile coords check above is block-uniform, so every thread that
    // reaches this __syncthreads reaches it together, and so is `stage_a`.
    if (stage_a) {
      // Row-wise, because LDS holds the rows compacted at REDUCTION_SIZE
      // while global has them INPUT_ROW_STRIDE apart. At BATCH_SIZE 1 this is
      // one trip of the old flat copy.
      constexpr int VEC_PER_ROW =
          REDUCTION_SIZE / static_cast<int>(sizeof(u32x4_t) /
                                            sizeof(unsigned short));
#pragma unroll
      for (int m = 0; m < BATCH_SIZE; m++) {
        u32x4_t const *src = reinterpret_cast<u32x4_t const *>(
            tile_input + static_cast<size_t>(m) * INPUT_ROW_STRIDE);
        u32x4_t *dst = reinterpret_cast<u32x4_t *>(s_a) + m * VEC_PER_ROW;
#pragma unroll
        for (int i = tid; i < VEC_PER_ROW; i += NTHREADS) {
          dst[i] = ld_g<u32x4_t>(src + i);
        }
      }
      __syncthreads();
    }
  }

  unsigned char const *w_row = wg + static_cast<size_t>(row) * REDUCTION_SIZE;
  unsigned char const *s_row =
      wg + WG_DATA_BYTES +
      static_cast<size_t>(row) * (REDUCTION_SIZE / SCALE_BLOCK);

  // See the bf16 kernel: one wave per SIMD means the only latency hiding is
  // the number of loads a single thread keeps in flight.
  constexpr int UNROLL_MAX = (BATCH_SIZE == 1) ? 8 : 4;
  constexpr int UNROLL = (ITERS % UNROLL_MAX == 0) ? UNROLL_MAX
                         : (ITERS % 4 == 0)        ? 4
                         : (ITERS % 2 == 0)        ? 2
                                                   : 1;

  float acc[BATCH_SIZE][4];
#pragma unroll
  for (int m = 0; m < BATCH_SIZE; m++) {
#pragma unroll
    for (int c = 0; c < 4; c++) {
      acc[m][c] = 0.0f;
    }
  }

  for (int i0 = 0; i0 < ITERS; i0 += UNROLL) {
    // Indexed only by fully unrolled loops, so these stay in VGPRs.
    u32x4_t wv[UNROLL];
    unsigned char sv[UNROLL];
    u32x4_t av[UNROLL][BATCH_SIZE][2];
    // NO-GO, measured: hoisting a base pointer for these two batches.
    //
    // `s_row + k / SCALE_BLOCK` with `k` a signed int is two things LLVM
    // cannot strength-reduce across the unroll -- a *signed* divide, which
    // needs the round-toward-zero correction unless k >= 0 is provable, and a
    // sign-extended add, which does not distribute. So it emits UNROLL
    // independent 64-bit address chains for UNROLL scale bytes: 18
    // `v_lshl_add_u64` per trip and 0 of 8 `global_load_ubyte` in
    // immediate-offset form, against a weight half that folds cleanly onto
    // `v[34:35] offset:512/1024/1536`. That is exactly the detector
    // MPK_MOE_SCBASE was built from, and it was worth -0.123 ms there.
    //
    // Hoisting `wp`/`sp` once per trip and indexing by `u * W_USTRIDE` /
    // `u * (W_USTRIDE / SCALE_BLOCK)` (exact, since LANES_PER_ROW * VEC is a
    // whole multiple of SCALE_BLOCK) lands completely: 18 -> 4
    // `v_lshl_add_u64`, 0/8 -> 7/8 immediate-offset, body 217 -> 201 insns,
    // waitcnt 27 -> 22, `.vgpr_count` unchanged at 333 with 0 spills.
    //
    // It measures **9.752 (n=3) against a 9.718 (n=5) control** -- neutral to
    // marginally worse. The tell is in the same disassembly: the first wait of
    // the trip goes from `vmcnt(7)` to `vmcnt(3)`. The scheduler spent the
    // registers the fold handed back on a SHALLOWER load pipeline, and this
    // loop is latency-bound (128 of its 217 instructions are v_cvt + v_dot2c
    // with no MFMA at all), so depth is worth more than the 16 instructions.
    //
    // Rule this establishes, and the reason MPK_MOE_SCBASE is not a general
    // transform: an address-arithmetic cut only pays if the load depth at the
    // top of the trip does not shrink. Read `vmcnt(N)` at the first wait
    // before and after, not just the instruction count.
#pragma unroll
    for (int u = 0; u < UNROLL; u++) {
      // Chunk index within the row, in units of VEC elements.
      int const c = (i0 + u) * LANES_PER_ROW + lane;
      int const k = c * VEC;
      wv[u] = ld_g<u32x4_t>(w_row + k);
      sv[u] = ld_g<unsigned char>(s_row + k / SCALE_BLOCK);
#pragma unroll
      for (int m = 0; m < BATCH_SIZE; m++) {
        // if constexpr, not a ternary on the pointer: a select between an
        // LDS-derived and a global pointer would collapse both to generic and
        // cost flat_load where ds_read belongs.
        if constexpr (STAGE_A) {
          unsigned short const *a = s_a + m * REDUCTION_SIZE + k;
          av[u][m][0] = *reinterpret_cast<u32x4_t const *>(a);
          av[u][m][1] = *reinterpret_cast<u32x4_t const *>(a + 8);
        } else {
          unsigned short const *a = tile_input + m * INPUT_ROW_STRIDE + k;
          av[u][m][0] = ld_g<u32x4_t>(a);
          av[u][m][1] = ld_g<u32x4_t>(a + 8);
        }
      }
    }
#pragma unroll
    for (int u = 0; u < UNROLL; u++) {
      float const sc = e8m0_to_f32(sv[u]);
      // Weight dword h covers elements [8h, 8h+8); its two halves are the two
      // bf16 pairs of activation dword av[..][h/2].{x,y,z,w}, in order.
      unsigned const wd[4] = {wv[u].x, wv[u].y, wv[u].z, wv[u].w};
#pragma unroll
      for (int m = 0; m < BATCH_SIZE; m++) {
#pragma unroll
        for (int h = 0; h < 4; h++) {
          u32x4_t const a = av[u][m][h >> 1];
          unsigned const ax = (h & 1) ? a.z : a.x;
          unsigned const ay = (h & 1) ? a.w : a.y;
          acc[m][2 * (h & 1)] = gang_gemv_detail::fma_pair(
              cvt_fp8_pair<false>(wd[h], sc), ax, acc[m][2 * (h & 1)]);
          acc[m][2 * (h & 1) + 1] = gang_gemv_detail::fma_pair(
              cvt_fp8_pair<true>(wd[h], sc), ay, acc[m][2 * (h & 1) + 1]);
        }
      }
    }
  }

  float sum[BATCH_SIZE];
#pragma unroll
  for (int m = 0; m < BATCH_SIZE; m++) {
    sum[m] = (acc[m][0] + acc[m][1]) + (acc[m][2] + acc[m][3]);
#pragma unroll
    for (int off = LANES_PER_ROW >> 1; off > 0; off >>= 1) {
      sum[m] += __shfl_xor(sum[m], off);
    }
  }

  if (lane == 0) {
    int const n_local = n_tile * ROWS_PER_WG + row;
    float const bv = Bs ? b2f(ld_g<unsigned short>(Bs + n_local)) : 0.0f;
#pragma unroll
    for (int m = 0; m < BATCH_SIZE; m++) {
      if (m_tile * BATCH_SIZE + m >= num_active_tokens) {
        continue;
      }
      size_t const idx = out_off + static_cast<size_t>(m) * o_stride + row;
      float v = sum[m] + bv;
      if constexpr (HAS_RESIDUAL) {
        v += b2f(ld_g<unsigned short>(R + idx));
      }
      if constexpr (WRITE_THROUGH) {
        unsigned short const o = f2b(v);
        st_wt_u16(&O[idx], o);
      } else {
        O[idx] = f2b(v);
      }
    }
  }
}

} // namespace kernel
