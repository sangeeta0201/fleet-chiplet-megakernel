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
#pragma once
#include "tasks/common/common_header.cuh"
namespace kernel {
template <typename T>
__device__ __forceinline__ void warp_reduce_max_idx(T &val, long long &idx) {
#pragma unroll
  for (int offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
    // AMD: use __shfl_down (no _sync mask); width=64 for AMD wavefront
    // reduction
    float tmp = __shfl_down((float)val, offset, NUM_THREADS_PER_WARP);
    T other_val = (T)tmp;
    // AMD: 64-bit shuffle must split into two 32-bit parts
    // __shfl_down only handles 32-bit values on AMD GPUs
    unsigned int idx_lo = static_cast<unsigned int>(idx & 0xFFFFFFFF);
    unsigned int idx_hi = static_cast<unsigned int>((idx >> 32) & 0xFFFFFFFF);
    unsigned int other_idx_lo =
        __shfl_down(idx_lo, offset, NUM_THREADS_PER_WARP);
    unsigned int other_idx_hi =
        __shfl_down(idx_hi, offset, NUM_THREADS_PER_WARP);
    long long other_idx =
        (static_cast<long long>(other_idx_hi) << 32) | other_idx_lo;
    if (other_val > val) {
      val = other_val;
      idx = other_idx;
    }
  }
}

template <typename T>
__device__ __forceinline__ void block_reduce_max_idx(T &val, long long &idx) {
  // Align the shared memory to 128 bytes
  extern __shared__ char smem[];
  long long *smem_idxs =
      (long long *)((reinterpret_cast<uintptr_t>(smem) + 127) / 128 * 128);
  T *smem_vals = reinterpret_cast<T *>(smem_idxs + 32); // max 32 warps

  warp_reduce_max_idx(val, idx);

  int my_lane_id = lane_id();
  int my_warp_id = warp_id();

  // Barrier BEFORE the write, not just after it.
  //
  // Both callers invoke this once per row of the batch, reusing these same
  // smem slots every iteration, and the only barrier used to be the one below.
  // That orders the write against warp 0's read within a single call, but
  // nothing stops warps 1..N from racing ahead into row b+1 and overwriting
  // smem_vals[warp] while warp 0 is still reducing row b. Warp 0 then mixes
  // one row's partial max with another's.
  //
  // Unreachable at num_active_tokens == 1: the loop body runs once, so there
  // is no next iteration to race with. That is exactly why B=1 is stable over
  // ~25 runs while B=8 is not, and why the symptom is a single near-tie
  // argmax position flipping for a subset of rows rather than garbage -- the
  // stolen value is a real logit max from a neighbouring row, so it only
  // changes the outcome when two candidates are close.
  __syncthreads();

  if (my_lane_id == 0) {
    smem_vals[my_warp_id] = val;
    smem_idxs[my_warp_id] = idx;
  }

  __syncthreads();

  // Only thread 0 holds the final result
  if (my_warp_id == 0) {
    T block_max_val = T(-inf);
    long long block_max_idx = -1;

    int num_warps =
        (blockDim.x + NUM_THREADS_PER_WARP - 1) / NUM_THREADS_PER_WARP;
    if (my_lane_id < num_warps) {
      block_max_val = smem_vals[my_lane_id];
      block_max_idx = smem_idxs[my_lane_id];
    }
    warp_reduce_max_idx(block_max_val, block_max_idx);

    if (my_lane_id == 0) {
      val = block_max_val;
      idx = block_max_idx;
    }
  }
}

template <typename T, int BATCH_SIZE, int CHUNK_SIZE, int NUM_PARTIAL_TASKS>
__device__ __forceinline__ void
    argmax_partial_kernel(void const *__restrict__ input_ptr,
                          void *__restrict__ output_val_ptr,
                          void *__restrict__ output_idx_ptr,
                          int num_active_tokens) {
  T const *__restrict__ input = static_cast<T const *>(input_ptr);
  T *__restrict__ output_val = static_cast<T *>(output_val_ptr);
  long long *__restrict__ output_idx = static_cast<long long *>(output_idx_ptr);

  int tidx = threadIdx.x;

// TODO: try vectorize
#pragma unroll
  for (int batch_idx = 0; batch_idx < num_active_tokens; batch_idx++) {
    T local_max = T(-inf);
    long long local_idx = -1;
#pragma unroll
    for (int i = tidx; i < CHUNK_SIZE; i += NUM_THREADS) {
      T val = input[i + batch_idx * CHUNK_SIZE * NUM_PARTIAL_TASKS];
      if (val > local_max) {
        local_max = val;
        local_idx = i;
      }
    }

    block_reduce_max_idx<T>(local_max, local_idx);

    if (tidx == 0) {
      output_val[batch_idx * NUM_PARTIAL_TASKS] = local_max;
      output_idx[batch_idx * NUM_PARTIAL_TASKS] = local_idx;
    }
  }
}

#ifdef MPK_AMX_ROWS
// The shfl_down tournament of warp_reduce_max_idx (strictly greater wins, offsets 32 .. 1, lane 0 keeps the result)
// returns the maximum value, and among lanes holding it the one with the smallest 6-bit-reversed lane index (the
// lower lane of every pair keeps a tie; the last step pairs even with odd lanes, so bit 0 decides first, then bit
// 1, ...). MPK_AMX_ROWS computes exactly that, without the LDS shuffles: a wave max (exact in any order), then a
// wave min of bitrev6(lane) over the lanes equal to it. Values are finite bf16 logits (-inf only for no partial).
// row_shr:N within each 16-lane row; with bound_ctrl off a lane whose source is outside its row keeps x
template <int CTRL>
__device__ __forceinline__ int amx3_dpp(int x) {
  return __builtin_amdgcn_update_dpp(x, x, CTRL, 0xf, 0xf, false);
}
__device__ __forceinline__ float amx3_wave_max(float x) {
  // max over each 16-lane row (row_shr; a lane whose source is outside the row keeps its value), then the 4 rows
  x = fmaxf(x, __int_as_float(amx3_dpp<0x111>(__float_as_int(x))));
  x = fmaxf(x, __int_as_float(amx3_dpp<0x112>(__float_as_int(x))));
  x = fmaxf(x, __int_as_float(amx3_dpp<0x114>(__float_as_int(x))));
  x = fmaxf(x, __int_as_float(amx3_dpp<0x118>(__float_as_int(x))));
  float const r0 = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(x), 15));
  float const r1 = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(x), 31));
  float const r2 = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(x), 47));
  float const r3 = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(x), 63));
  return fmaxf(fmaxf(r0, r1), fmaxf(r2, r3));
}
__device__ __forceinline__ int amx3_wave_min(int x) {
  x = min(x, amx3_dpp<0x111>(x));
  x = min(x, amx3_dpp<0x112>(x));
  x = min(x, amx3_dpp<0x114>(x));
  x = min(x, amx3_dpp<0x118>(x));
  int const r0 = __builtin_amdgcn_readlane(x, 15), r1 = __builtin_amdgcn_readlane(x, 31);
  int const r2 = __builtin_amdgcn_readlane(x, 47), r3 = __builtin_amdgcn_readlane(x, 63);
  return min(min(r0, r1), min(r2, r3));
}
#endif

template <typename T, int BATCH_SIZE, int CHUNK_SIZE, int NUM_PARTIAL_TASKS>
__device__ __forceinline__ void
    argmax_reduce_kernel(void const *__restrict__ input_val_ptr,
                         void const *__restrict__ input_idx_ptr,
                         void *__restrict__ final_output_ptr,
                         int num_active_tokens) {
#ifdef MPK_TAIL_NO111
  // An end-of-pass marker: the EAGLE3 draft already published each row's argmax (this reduction's tie order) and
  // the virtual-slot prepare reads it there; the task stays so the end-of-graph event keeps its worker and scheduler.
  return;
#endif
  T const *__restrict__ partial_vals = static_cast<T const *>(input_val_ptr);
  long long const *__restrict__ partial_idxs =
      static_cast<long long const *>(input_idx_ptr);
  long long *__restrict__ final_output =
      static_cast<long long *>(final_output_ptr);

#ifdef MPK_ARGMAX_FUSED
  // The LM-head workers already max-folded their partials into one key per
  // die (cleared per launch, and below after every read).
  static_assert(BATCH_SIZE == 1 && CHUNK_SIZE == 0 && NUM_PARTIAL_TASKS <= 256,
                "MPK_ARGMAX_FUSED: batch 1, absolute indices, one partial per thread");
  if (g_mpk_amx_rt) {
    if (threadIdx.x == 0) {
      int const *const b0 = g_aid_nc_rep[0] + MPK_NC_AMX_INTS;
      int const *const b1 = g_aid_nc_rep[1] + MPK_NC_AMX_INTS;
      unsigned long long k0, k1, k2, k3, k4, k5, k6, k7;
      asm volatile("global_load_dwordx2 %0, %8, off sc0 sc1\n"
                   "global_load_dwordx2 %1, %8, off offset:256 sc0 sc1\n"
                   "global_load_dwordx2 %2, %8, off offset:512 sc0 sc1\n"
                   "global_load_dwordx2 %3, %8, off offset:768 sc0 sc1\n"
                   "global_load_dwordx2 %4, %9, off offset:1024 sc0 sc1\n"
                   "global_load_dwordx2 %5, %9, off offset:1280 sc0 sc1\n"
                   "global_load_dwordx2 %6, %9, off offset:1536 sc0 sc1\n"
                   "global_load_dwordx2 %7, %9, off offset:1792 sc0 sc1\n"
                   "s_waitcnt vmcnt(0)"
                   : "=&v"(k0), "=&v"(k1), "=&v"(k2), "=&v"(k3), "=&v"(k4),
                     "=&v"(k5), "=&v"(k6), "=&v"(k7)
                   : "v"(b0), "v"(b1)
                   : "memory");
      unsigned long long m = k0 > k1 ? k0 : k1;
      unsigned long long const m2 = k2 > k3 ? k2 : k3;
      unsigned long long const m4 = k4 > k5 ? k4 : k5;
      unsigned long long const m6 = k6 > k7 ? k6 : k7;
      m = m > m2 ? m : m2;
      m = m > m4 ? m : m4;
      m = m > m6 ? m : m6;
      unsigned const idx32 = (unsigned)m;
      final_output[0] = ((m >> 48) == 0ull || idx32 == 0xFFFFFFFFu)
                            ? -1ll
                            : (long long)idx32;
      unsigned long long const z = 0ull;
      asm volatile("global_store_dwordx2 %0, %2, off sc0 sc1\n"
                   "global_store_dwordx2 %0, %2, off offset:256 sc0 sc1\n"
                   "global_store_dwordx2 %0, %2, off offset:512 sc0 sc1\n"
                   "global_store_dwordx2 %0, %2, off offset:768 sc0 sc1\n"
                   "global_store_dwordx2 %1, %2, off offset:1024 sc0 sc1\n"
                   "global_store_dwordx2 %1, %2, off offset:1280 sc0 sc1\n"
                   "global_store_dwordx2 %1, %2, off offset:1536 sc0 sc1\n"
                   "global_store_dwordx2 %1, %2, off offset:1792 sc0 sc1"
                   ::"v"(b0), "v"(b1), "v"(z)
                   : "memory");
    }
    return;
  }
#endif

#ifdef MPK_AMX_ROWS
  // MPK_AMX_ROWS: every row at once, one partial per thread. All rows' loads are straight-line (clamped in bounds);
  // per row the wave's winner is warp_reduce_max_idx's (see amx3_wave_max), and thread 0 then scans the 4 warp
  // winners in the cross-warp tournament's order (warps 0, 2, 1, 3; strictly greater wins). Bit-identical.
  if constexpr (BATCH_SIZE > 1 && BATCH_SIZE <= 8 && NUM_PARTIAL_TASKS <= 256) {
    if (blockDim.x == 256) {
      int const tid = threadIdx.x;
      int const lane = tid & 63, wid = tid >> 6;
      bool const has = tid < NUM_PARTIAL_TASKS;
      int const ti = has ? tid : NUM_PARTIAL_TASKS - 1;
      T cv[BATCH_SIZE];
      long long ci[BATCH_SIZE];
#pragma unroll
      for (int r = 0; r < BATCH_SIZE; r++) {
        cv[r] = partial_vals[ti + r * NUM_PARTIAL_TASKS];
        ci[r] = partial_idxs[ti + r * NUM_PARTIAL_TASKS];
      }
      extern __shared__ char smem[];
      long long *smem_idxs = (long long *)((reinterpret_cast<uintptr_t>(smem) + 127) / 128 * 128);
      float *smem_vals = reinterpret_cast<float *>(smem_idxs + 32);  // 4 warps x BATCH_SIZE <= 32 slots
      int const key = (int)(__brev((unsigned)lane) >> 26);           // bitrev6(lane)
#pragma unroll
      for (int r = 0; r < BATCH_SIZE; r++) {
        // the thread-local step of the original: take the partial if it beats -inf
        float v = -__builtin_inff();
        long long pk = -1;
        if (has && cv[r] > T(-inf)) {
          v = (float)cv[r];
          pk = ((long long)tid << 32) | ci[r];
        }
        float const m = amx3_wave_max(v);
        int const wk = amx3_wave_min(v == m ? key : 64);
        int const wl = (int)(__brev((unsigned)wk) >> 26);
        unsigned const plo = (unsigned)__builtin_amdgcn_readlane((int)(unsigned)(pk & 0xFFFFFFFF), wl);
        unsigned const phi = (unsigned)__builtin_amdgcn_readlane((int)(unsigned)((pk >> 32) & 0xFFFFFFFF), wl);
        if (lane == 0) {
          smem_vals[wid * BATCH_SIZE + r] = m;
          smem_idxs[wid * BATCH_SIZE + r] = (long long)(((unsigned long long)phi << 32) | plo);
        }
      }
      __syncthreads();
      if (tid == 0) {
        int const num_warps = (blockDim.x + NUM_THREADS_PER_WARP - 1) / NUM_THREADS_PER_WARP;
        int const ord[4] = {0, 2, 1, 3};
#pragma unroll
        for (int r = 0; r < BATCH_SIZE; r++) {
          if (r < num_active_tokens) {
            float bv = -__builtin_inff();
            long long bi = -1;
#pragma unroll
            for (int q = 0; q < 4; q++) {
              int const w = ord[q];
              if (w < num_warps) {
                float const ov = smem_vals[w * BATCH_SIZE + r];
                if (ov > bv) {
                  bv = ov;
                  bi = smem_idxs[w * BATCH_SIZE + r];
                }
              }
            }
            if (bi != -1) {
              long long winning_chunk_idx = bi >> 32;
              long long winning_relative_idx = bi & 0xFFFFFFFF;
              final_output[r] = winning_chunk_idx * CHUNK_SIZE + winning_relative_idx;
            } else {
              final_output[r] = -1;
            }
          }
        }
      }
      return;
    }
  }
#endif
  int tidx = threadIdx.x;
// TODO: try vectorize
#pragma unroll
  for (int batch_idx = 0; batch_idx < num_active_tokens; batch_idx++) {
    T local_max = T(-inf);
    // Pack (chunk_index, relative_index) into a single 64-bit integer
    long long local_packed_idx = -1;

#pragma unroll
    for (int i = tidx; i < NUM_PARTIAL_TASKS; i += blockDim.x) {
      T current_val = partial_vals[i + batch_idx * NUM_PARTIAL_TASKS];
#ifdef MPK_ARGMAX_IDX_PF
      long long const current_idx =
          partial_idxs[i + batch_idx * NUM_PARTIAL_TASKS];
#endif
      if (current_val > local_max) {
        local_max = current_val;
        // Higher 32 bits for chunk_index (i), lower 32 for relative_index
#ifdef MPK_ARGMAX_IDX_PF
        local_packed_idx = ((long long)i << 32) | current_idx;
#else
        local_packed_idx = ((long long)i << 32) |
                           partial_idxs[i + batch_idx * NUM_PARTIAL_TASKS];
#endif
      }
    }

    block_reduce_max_idx(local_max, local_packed_idx);

    if (tidx == 0) {
      if (local_packed_idx != -1) {
        long long winning_chunk_idx = local_packed_idx >> 32;
        long long winning_relative_idx = local_packed_idx & 0xFFFFFFFF;
        final_output[batch_idx] =
            winning_chunk_idx * CHUNK_SIZE + winning_relative_idx;
      } else {
        final_output[batch_idx] = -1;
      }
    }
  }
}

} // namespace kernel
