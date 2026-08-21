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

// MXFP4 twin of gang_gemv_mxfp8_mi300.cuh: the narrow-tile GEMV reading E2M1
// nibbles with one E8M0 exponent per 32 contiguous K elements.
//
// Why this weight and not qkv_a. Per layer per GPU the un-absorbed attention
// weights are qkv_a 16.1 M, q_b 34.6 M + W_UK 6.5 M, W_UV 8.4 M and o_proj
// 100.7 M. o_proj is not just the biggest, it is the only one measured
// *byte-bound* -- 76% of HBM peak -- so halving its bytes converts to time
// close to one-for-one, which is not true of qkv_a. The ceiling recorded for
// the attention-MXFP4 program as a whole (~0.9 ms) was extrapolated from
// MPK_ATTN_HALFK on qkv_a, and qkv_a is the worst member of the set to
// extrapolate from: it is 41% redundant prologue and its phase is one
// grid-stride round whose makespan is a single tile. o_proj has neither
// property.
//
// The port is small because the fp8 kernel already did the hard parts. Tile
// addressing, the LDS activation stage, the four accumulator chains, the
// __shfl_xor reduction and the bias/residual epilogue are unchanged, and
// gang_gemv_mxfp8_detail's helpers (e8m0_to_f32, ld_g, u32x4_t) are reused
// rather than copied. Three things change:
//
//  1. A 16-byte load now carries 32 elements instead of 16, so VEC doubles and
//     ITERS halves. That is the whole byte saving: the weight row is K/2 bytes.
//
//  2. The scale half does not shrink -- it is still one E8M0 per 32 elements,
//     K/32 bytes per row. So the traffic goes from K + K/32 to K/2 + K/32,
//     i.e. 1.94x rather than a clean 2x, and the scale bytes go from 3% of the
//     weight to 6%. A lane's chunk is now *exactly* one scale block rather
//     than half of one, which is the tightest the format allows and still one
//     scale byte per iteration.
//
//  3. Dequant is v_cvt_scalef32_pk_bf16_fp4 in place of
//     v_cvt_scalef32_pk_bf16_fp8. Both take the E8M0 as an fp32 multiplier and
//     emit a packed bf16 pair, so fma_pair downstream is untouched; the fp4
//     form selects one of the four *bytes* of the source dword with a literal
//     word_sel instead of one of its two halves with a literal bool. Four
//     calls per dword instead of two, for twice the elements -- the same two
//     converts per fma_pair the fp8 path pays. Unlike fp8 the conversion is
//     not lossless against the original bf16, but that is a property of the
//     quantization, not of this kernel, and it is gated separately: see
//     GLM_FAKE_MXFP4_ATTN in demo/glm5/demo.py, which rounds the values
//     through E2M1 while still packing MXFP8 and passes 4/4 correctness
//     prompts with all 8 ranks identical.
//
// Nibble order matches quantize_mxfp4's packer: byte b holds element 2b in the
// low nibble and 2b+1 in the high, and word_sel s selects byte s and emits
// (low, high) as the bf16 pair. So dword h of a 16-byte load covers elements
// [8h, 8h+8) in order, which is exactly one u32x4_t of bf16 activations --
// the operand mapping stays a straight index rather than the interleave the
// fp8 body needs.

#pragma once
#include "tasks/mi300/gang_gemv_mxfp8_mi300.cuh"

namespace kernel {

namespace gang_gemv_mxfp4_detail {
typedef __bf16 __attribute__((ext_vector_type(2))) bf16x2_t;

// Two E2M1 nibbles out of byte SEL of `raw` -- element 2*SEL low, 2*SEL+1
// high -- scaled and packed as a bf16 pair in one 32-bit register, in the
// order fma_pair expects.
//
// SEL is a template parameter, not an argument: the builtin's byte select has
// to be a literal, the same reason cvt_fp8_pair templates on HI.
template <int SEL>
__device__ __forceinline__ unsigned cvt_fp4_pair(unsigned raw, float scale) {
  bf16x2_t v = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(raw, scale, SEL);
  unsigned u;
  __builtin_memcpy(&u, &v, 4);
  return u;
}
} // namespace gang_gemv_mxfp4_detail

// Same signature, same tile addressing and the same caller contract as
// gang_gemv_mxfp8_kernel -- including `stage_a`, whose invariant is unchanged:
// pass false only when this block already staged the same `tile_input` and
// nothing has written _fused_smem since.
//
// `weight_ptr` is this XCD's chunk of the workgroup-packed weight,
// [n_tiles, ROWS_PER_WG * (REDUCTION_SIZE/2 + REDUCTION_SIZE/32)] bytes. The
// packer is the same one: pack_mxfp8_workgroup reads its row width from the
// data tensor, so quantize_mxfp4's [out, K/2] lands in the identical
// data-then-scales layout with no repacking work.
template <int BATCH_SIZE, // = m_per_tile
          int REDUCTION_SIZE,
          int ROWS_PER_WG,
          bool HAS_RESIDUAL,
          bool WRITE_THROUGH = false>
__device__ __noinline__ void
    gang_gemv_mxfp4_kernel(void const *input_ptr,
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
  using gang_gemv_mxfp4_detail::cvt_fp4_pair;
  using gang_gemv_mxfp8_detail::e8m0_to_f32;
  using gang_gemv_mxfp8_detail::ld_g;
  using gang_gemv_mxfp8_detail::u32x4_t;

  constexpr int NTHREADS = 256;
  constexpr int LANES_PER_ROW = NTHREADS / ROWS_PER_WG;
  constexpr int VEC = 32;      // E2M1 elements per 16-byte load
  constexpr int VEC_BYTES = 16;
  constexpr int SCALE_BLOCK = 32;
  static_assert(NTHREADS % ROWS_PER_WG == 0,
                "ROWS_PER_WG must divide the 256-thread block");
  static_assert(LANES_PER_ROW <= 64 && (64 % LANES_PER_ROW) == 0,
                "a row's lanes must be an aligned slice of one wavefront, so "
                "ROWS_PER_WG must be >= 4 and a power of two");
  static_assert(REDUCTION_SIZE % (LANES_PER_ROW * VEC) == 0,
                "K must tile evenly over the row's lanes at 32 fp4 each");
  static_assert(VEC == SCALE_BLOCK,
                "a lane's chunk is exactly one scale block at MXFP4, which is "
                "what keeps it to a single exponent byte per iteration");
  static_assert((ROWS_PER_WG * (REDUCTION_SIZE / SCALE_BLOCK)) % 16 == 0,
                "the scale half must keep the next workgroup's data half "
                "16-byte aligned");
  static_assert((ROWS_PER_WG * (REDUCTION_SIZE / 2)) % 16 == 0,
                "the data half must keep the scale half 16-byte aligned");
  constexpr int ITERS = REDUCTION_SIZE / (LANES_PER_ROW * VEC);
  constexpr int WG_DATA_BYTES = ROWS_PER_WG * (REDUCTION_SIZE / 2);
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

  // Identical to the fp8 kernel: the activation is bf16 either way, so the
  // staging area and its size are unchanged by narrowing the weight.
  constexpr size_t A_LDS_BYTES =
      sizeof(unsigned short) * BATCH_SIZE * REDUCTION_SIZE;
  constexpr bool STAGE_A =
      A_LDS_BYTES <=
      static_cast<size_t>(mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE);
  extern __shared__ char _fused_smem[];
  unsigned short *s_a = reinterpret_cast<unsigned short *>(_fused_smem);
  if constexpr (STAGE_A) {
    if (stage_a) {
      u32x4_t const *src = reinterpret_cast<u32x4_t const *>(tile_input);
      u32x4_t *dst = reinterpret_cast<u32x4_t *>(s_a);
#pragma unroll
      for (int i = tid; i < static_cast<int>(A_LDS_BYTES / sizeof(u32x4_t));
           i += NTHREADS) {
        dst[i] = ld_g<u32x4_t>(src + i);
      }
      __syncthreads();
    }
  }

  unsigned char const *w_row =
      wg + static_cast<size_t>(row) * (REDUCTION_SIZE / 2);
  unsigned char const *s_row =
      wg + WG_DATA_BYTES +
      static_cast<size_t>(row) * (REDUCTION_SIZE / SCALE_BLOCK);

  // One wave per SIMD, so the only latency hiding is how many loads a single
  // thread keeps in flight. ITERS is half the fp8 kernel's for the same K, so
  // a shape that unrolled by 8 there may only manage 4 here; the ladder picks
  // the largest divisor rather than clamping, which keeps the loop exact.
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
    // Indexed only by fully unrolled loops, so these stay in VGPRs. The
    // activation vector is 4 wide rather than 2: 32 elements of weight need
    // 64 bytes of bf16.
    u32x4_t wv[UNROLL];
    unsigned char sv[UNROLL];
    u32x4_t av[UNROLL][BATCH_SIZE][4];
#pragma unroll
    for (int u = 0; u < UNROLL; u++) {
      // Chunk index within the row, in units of VEC elements.
      int const c = (i0 + u) * LANES_PER_ROW + lane;
      int const k = c * VEC;
      wv[u] = ld_g<u32x4_t>(w_row + c * VEC_BYTES);
      // k / SCALE_BLOCK == c, since a chunk is exactly one scale block.
      sv[u] = ld_g<unsigned char>(s_row + c);
#pragma unroll
      for (int m = 0; m < BATCH_SIZE; m++) {
        // if constexpr, not a ternary on the pointer: a select between an
        // LDS-derived and a global pointer would collapse both to generic and
        // cost flat_load where ds_read belongs.
        if constexpr (STAGE_A) {
          unsigned short const *a = s_a + m * REDUCTION_SIZE + k;
#pragma unroll
          for (int h = 0; h < 4; h++) {
            av[u][m][h] = *reinterpret_cast<u32x4_t const *>(a + 8 * h);
          }
        } else {
          unsigned short const *a = tile_input + m * REDUCTION_SIZE + k;
#pragma unroll
          for (int h = 0; h < 4; h++) {
            av[u][m][h] = ld_g<u32x4_t>(a + 8 * h);
          }
        }
      }
    }
#pragma unroll
    for (int u = 0; u < UNROLL; u++) {
      float const sc = e8m0_to_f32(sv[u]);
      // Weight dword h covers elements [8h, 8h+8), which is exactly the eight
      // bf16 of av[..][h]. Byte j of the dword is elements 8h+2j and 8h+2j+1,
      // and dword j of the activation vector is the same pair -- so the two
      // operands line up index for index, with no interleave.
      unsigned const wd[4] = {wv[u].x, wv[u].y, wv[u].z, wv[u].w};
#pragma unroll
      for (int m = 0; m < BATCH_SIZE; m++) {
#pragma unroll
        for (int h = 0; h < 4; h++) {
          u32x4_t const a = av[u][m][h];
          // Four independent accumulator chains, one per byte lane of the
          // weight dword -- the same four the fp8 kernel keeps, so the
          // dependency structure of the inner loop is unchanged.
          acc[m][0] = gang_gemv_detail::fma_pair(
              cvt_fp4_pair<0>(wd[h], sc), a.x, acc[m][0]);
          acc[m][1] = gang_gemv_detail::fma_pair(
              cvt_fp4_pair<1>(wd[h], sc), a.y, acc[m][1]);
          acc[m][2] = gang_gemv_detail::fma_pair(
              cvt_fp4_pair<2>(wd[h], sc), a.z, acc[m][2]);
          acc[m][3] = gang_gemv_detail::fma_pair(
              cvt_fp4_pair<3>(wd[h], sc), a.w, acc[m][3]);
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
