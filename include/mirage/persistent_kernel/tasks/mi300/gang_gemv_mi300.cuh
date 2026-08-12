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

// Narrow-tile gang GEMV for the hidden-width projections at decode.
//
// The CK gang linear tiles N at 64 because that is what a 16x64x256 MFMA tile
// wants. For a projection whose output is only hidden_size wide that is too
// coarse: GLM-4.7-Flash's absorbed o_proj is [2048, 10240], so 2048/64 = 32
// tiles, 4 per XCD, and 208 of the 240 workers sit idle while those 32 read
// 42 MB. Measured, that op is 22% of the token's exclusive critical path at
// 35 GB/s per CU -- already above the 31 GB/s per-CU fair share of an 8 TB/s
// part, so the CU count is what binds, not the kernel.
//
// The fix is more tiles, and the way to get them is to stop using MFMA. At
// batch 1 a 16-row MFMA tile throws away 15/16 of its rows anyway, and once
// the instruction is gone so is the N>=64 constraint. This kernel gives each
// workgroup ROWS_PER_WG output columns and splits the K reduction across
// 256/ROWS_PER_WG lanes, so ROWS_PER_WG=8 turns those 32 tiles into 256.
//
// Buying CUs this way costs nothing, unlike the split-K this replaces: the
// partial sums stay in registers and cross-lane shuffles, so there is no fp32
// workspace round trip and no done-counter spin. The activation is re-read
// once per row instead of once per tile, but every one of those reads is an
// L1/L2 hit on the same few KB.
//
// Deliberately not templated on a weight format: the arithmetic is a handful
// of percent of the runtime here, so there is nothing to gain from MFMA even
// when the weights are quantized.

#pragma once
// For gang_linear_tile_coords: the tile addressing has to match the CK path
// exactly, since a caller swaps between the two by changing one parameter.
#include "tasks/mi300/gang_linear_mi300.cuh"
#include <hip/hip_bf16.h>

namespace kernel {

namespace gang_gemv_detail {
using bf16_t = __hip_bfloat16;

__device__ __forceinline__ float b2f(unsigned short v) {
  unsigned u = (unsigned)v << 16;
  float f;
  __builtin_memcpy(&f, &u, 4);
  return f;
}

__device__ __forceinline__ unsigned short f2b(float f) {
  union {
    float f;
    unsigned u;
  } v;
  v.f = f;
  unsigned bias = ((v.u >> 16) & 1) + 0x7FFF;
  return (unsigned short)((v.u + bias) >> 16);
}

// A packed bf16 pair lives in one 32-bit register: element 2i in the low half,
// 2i+1 in the high half. Widening either to f32 is a single bit shuffle, so the
// pair never has to touch memory -- which matters more than it sounds, because
// reading the halves through a `unsigned short const *` aimed at a local uint4
// makes LLVM materialize that uint4 in scratch and turns every iteration of the
// k-loop into a global round trip.
__device__ __forceinline__ float lo2f(unsigned w) {
  unsigned u = w << 16;
  float f;
  __builtin_memcpy(&f, &u, 4);
  return f;
}

__device__ __forceinline__ float hi2f(unsigned w) {
  unsigned u = w & 0xFFFF0000u;
  float f;
  __builtin_memcpy(&f, &u, 4);
  return f;
}

// acc += dot(unpack(w), unpack(a)) over the packed pair.
__device__ __forceinline__ float fma_pair(unsigned w, unsigned a, float acc) {
  acc = __builtin_fmaf(lo2f(w), lo2f(a), acc);
  acc = __builtin_fmaf(hi2f(w), hi2f(a), acc);
  return acc;
}
} // namespace gang_gemv_detail

// Gang GEMV with optional residual and bias, same tile addressing as
// gang_linear_residual_kernel so the two are interchangeable at the call site.
//
// Thread mapping: tid -> (row = tid / LANES_PER_ROW, lane = tid % LANES_PER_ROW).
// LANES_PER_ROW divides 64, so a row's lanes are a contiguous, aligned slice of
// one wavefront and the final reduction is pure __shfl_xor with no LDS.
//
// The weight row stride is REDUCTION_SIZE (the register function asserts the
// weight is exactly that wide). The *input* row stride is not: a caller may
// narrow the reduction past a padded tail, in which case the input row is
// wider than REDUCTION_SIZE. That only stays correct while m_per_tile == 1,
// which is the same restriction the CK path already carries.
template <typename T,
          int BATCH_SIZE, // = m_per_tile
          int REDUCTION_SIZE,
          int ROWS_PER_WG,
          bool HAS_RESIDUAL>
__device__ __noinline__ void
    gang_gemv_kernel(void const *input_ptr,
                     void const *weight_ptr,  // [chunk_N, REDUCTION_SIZE]
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

  constexpr int NTHREADS = 256;
  constexpr int LANES_PER_ROW = NTHREADS / ROWS_PER_WG;
  constexpr int VEC = 8; // bf16 per 16-byte load
  static_assert(NTHREADS % ROWS_PER_WG == 0,
                "ROWS_PER_WG must divide the 256-thread block");
  static_assert(LANES_PER_ROW <= 64 && (64 % LANES_PER_ROW) == 0,
                "a row's lanes must be an aligned slice of one wavefront, so "
                "ROWS_PER_WG must be >= 4 and a power of two");
  static_assert(REDUCTION_SIZE % (LANES_PER_ROW * VEC) == 0,
                "K must tile evenly over the row's lanes at 8 bf16 each");
  constexpr int ITERS = REDUCTION_SIZE / (LANES_PER_ROW * VEC);

  assert(tile_idx >= 0);
  assert(tile_n == ROWS_PER_WG);

  int m_tile, n_tile;
  if (!gang_linear_tile_coords(
          tile_idx, m_tiles, n_tiles, wgm, &m_tile, &n_tile)) {
    return;
  }

  unsigned short const *A = static_cast<unsigned short const *>(input_ptr);
  unsigned short const *Wt = static_cast<unsigned short const *>(weight_ptr);
  unsigned short const *R =
      static_cast<unsigned short const *>(residual_ptr);
  unsigned short const *Bs = static_cast<unsigned short const *>(bias_ptr);
  unsigned short *O = static_cast<unsigned short *>(output_ptr);

  size_t const out_off =
      static_cast<size_t>(m_tile) * BATCH_SIZE * o_stride +
      static_cast<size_t>(n_tile) * ROWS_PER_WG;
  unsigned short const *tile_input =
      A + static_cast<size_t>(m_tile) * BATCH_SIZE * REDUCTION_SIZE;
  unsigned short const *tile_weight =
      Wt + static_cast<size_t>(n_tile) * ROWS_PER_WG * REDUCTION_SIZE;

  int const tid = threadIdx.x;
  int const row = tid / LANES_PER_ROW;
  int const lane = tid % LANES_PER_ROW;

  unsigned short const *w_row =
      tile_weight + static_cast<size_t>(row) * REDUCTION_SIZE;

  // A megakernel worker is one 256-thread block per CU, i.e. a single wave per
  // SIMD, so there is no second wave to cover a miss: the only latency hiding
  // available is however many loads one thread keeps in flight. Issue a whole
  // batch of them before touching any, rather than leaving it to the scheduler.
  constexpr int UNROLL_MAX = (BATCH_SIZE == 1) ? 8 : 4;
  constexpr int UNROLL = (ITERS % UNROLL_MAX == 0)  ? UNROLL_MAX
                         : (ITERS % 4 == 0)         ? 4
                         : (ITERS % 2 == 0)         ? 2
                                                    : 1;

  // Four chains rather than one: a bf16 pair is two dependent FMAs, and at one
  // wave per SIMD a single chain cannot fill the VALU issue slots.
  float acc[BATCH_SIZE][4];
#pragma unroll
  for (int m = 0; m < BATCH_SIZE; m++) {
#pragma unroll
    for (int c = 0; c < 4; c++) {
      acc[m][c] = 0.0f;
    }
  }

  // One 16-byte weight load per iteration feeds every M row, so widening the
  // batch costs activation loads (L2 hits) and FMAs, not HBM traffic.
  for (int i0 = 0; i0 < ITERS; i0 += UNROLL) {
    // Indexed only by fully unrolled loops, so these stay in VGPRs.
    uint4 wv[UNROLL];
    uint4 av[UNROLL][BATCH_SIZE];
#pragma unroll
    for (int u = 0; u < UNROLL; u++) {
      int const k = ((i0 + u) * LANES_PER_ROW + lane) * VEC;
      wv[u] = *reinterpret_cast<uint4 const *>(w_row + k);
#pragma unroll
      for (int m = 0; m < BATCH_SIZE; m++) {
        av[u][m] = *reinterpret_cast<uint4 const *>(tile_input +
                                                    m * REDUCTION_SIZE + k);
      }
    }
#pragma unroll
    for (int u = 0; u < UNROLL; u++) {
#pragma unroll
      for (int m = 0; m < BATCH_SIZE; m++) {
        acc[m][0] = gang_gemv_detail::fma_pair(wv[u].x, av[u][m].x, acc[m][0]);
        acc[m][1] = gang_gemv_detail::fma_pair(wv[u].y, av[u][m].y, acc[m][1]);
        acc[m][2] = gang_gemv_detail::fma_pair(wv[u].z, av[u][m].z, acc[m][2]);
        acc[m][3] = gang_gemv_detail::fma_pair(wv[u].w, av[u][m].w, acc[m][3]);
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
