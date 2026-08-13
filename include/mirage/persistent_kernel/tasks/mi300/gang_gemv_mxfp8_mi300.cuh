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

#pragma once
#include "tasks/mi300/gang_gemv_mi300.cuh"

namespace kernel {

namespace gang_gemv_mxfp8_detail {
typedef __bf16 __attribute__((ext_vector_type(2))) bf16x2_t;

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
} // namespace gang_gemv_mxfp8_detail

// Same signature and same tile addressing as gang_gemv_kernel, minus the
// element-type template parameter -- the weight is MXFP8 and the activation,
// residual, bias and output are all bf16.
//
// `weight_ptr` is this XCD's chunk of the workgroup-packed weight,
// [n_tiles, ROWS_PER_WG * (REDUCTION_SIZE + REDUCTION_SIZE/32)] bytes, so
// n_tile indexes a workgroup rather than a row block. As in the bf16 kernel
// the weight row stride is REDUCTION_SIZE while the *input* row stride need
// not be, which stays correct only while BATCH_SIZE == 1.
template <int BATCH_SIZE, // = m_per_tile
          int REDUCTION_SIZE,
          int ROWS_PER_WG,
          bool HAS_RESIDUAL>
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
                           void const *bias_ptr = nullptr) {
  using gang_gemv_detail::b2f;
  using gang_gemv_detail::f2b;
  using gang_gemv_mxfp8_detail::cvt_fp8_pair;
  using gang_gemv_mxfp8_detail::e8m0_to_f32;

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
      A + static_cast<size_t>(m_tile) * BATCH_SIZE * REDUCTION_SIZE;
  unsigned char const *wg = W + static_cast<size_t>(n_tile) * WG_BYTES;

  int const tid = threadIdx.x;
  int const row = tid / LANES_PER_ROW;
  int const lane = tid % LANES_PER_ROW;

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
    uint4 wv[UNROLL];
    unsigned char sv[UNROLL];
    uint4 av[UNROLL][BATCH_SIZE][2];
#pragma unroll
    for (int u = 0; u < UNROLL; u++) {
      // Chunk index within the row, in units of VEC elements.
      int const c = (i0 + u) * LANES_PER_ROW + lane;
      int const k = c * VEC;
      wv[u] = *reinterpret_cast<uint4 const *>(w_row + k);
      sv[u] = s_row[k / SCALE_BLOCK];
#pragma unroll
      for (int m = 0; m < BATCH_SIZE; m++) {
        unsigned short const *a = tile_input + m * REDUCTION_SIZE + k;
        av[u][m][0] = *reinterpret_cast<uint4 const *>(a);
        av[u][m][1] = *reinterpret_cast<uint4 const *>(a + 8);
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
          uint4 const a = av[u][m][h >> 1];
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
    float const bv = Bs ? b2f(Bs[n_local]) : 0.0f;
#pragma unroll
    for (int m = 0; m < BATCH_SIZE; m++) {
      if (m_tile * BATCH_SIZE + m >= num_active_tokens) {
        continue;
      }
      size_t const idx = out_off + static_cast<size_t>(m) * o_stride + row;
      float v = sum[m] + bv;
      if constexpr (HAS_RESIDUAL) {
        v += b2f(R[idx]);
      }
      O[idx] = f2b(v);
    }
  }
}

} // namespace kernel
