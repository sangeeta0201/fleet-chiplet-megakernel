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
// ---------------------------------------------------------------------------
// Cross-rank sum of one hidden-width partial, plus a residual add.
//
// The three `first_k_dense_replace` layers' MLP is 226.5M parameters each and
// every EP rank reads all of it: ~700 MB per rank per token in MXFP8, the same
// bytes four times over. Sharding the intermediate dimension four ways makes
// gate_up column-parallel and down_proj K-parallel over the matching slice, so
// each rank produces a PARTIAL hidden vector and the four have to be summed.
//
// This task is that sum. Every rank pushes its whole partial into the same slot
// of every peer's copy of one symmetric buffer, publishes an epoch, waits for
// the other three epochs, then reads all four slots and writes
//
//     output = residual + sum_p partial_p
//
// Buffer layout, one contiguous symmetric allocation:
//
//     [0, EP_WORLD_SIZE * BATCH_SIZE * HIDDEN)   bf16 payload, slot p at
//                                                p * BATCH_SIZE * HIDDEN
//     then EP_WORLD_SIZE * 8 uint64             epoch words, 64 bytes per PE
//
// Payload and epoch are separate words with a drain between them, and the
// per-PE epoch is padded to a 64-byte line so two peers' stores never share
// one. Same idiom, and the same reasons, as argmax_reduce_xrank.
//
// One block. The traffic is EP_WORLD_SIZE * HIDDEN * 2 bytes each way -- 49 KB
// at HIDDEN=6144, EP=4 -- which a single CU moves in about the time the peer
// rendezvous itself costs, so fanning this out across XCDs would buy nothing
// and would need a per-block epoch line.
//
// The caller must NOT also add the residual inside the down_proj GEMV: with
// every rank computing every output column, a residual folded into the GEMV
// would be summed EP_WORLD_SIZE times. Pass the GEMV a zero residual and let
// this task add the real one exactly once.
// ---------------------------------------------------------------------------
#include "comm/mpk_comm.cuh"
#include "mpk_atoms.cuh"
#include "tasks/common/common_header.cuh"

namespace kernel {

// 64 bytes per PE for the epoch, matching ARGMAX_XRANK_SLOT_U64's reasoning.
#define XRANK_SUM_SLOT_U64 8

__device__ __forceinline__ float xrank_sum_bf16_to_f32(unsigned short h) {
  return __uint_as_float((unsigned int)h << 16);
}

__device__ __forceinline__ unsigned short xrank_sum_f32_to_bf16(float f) {
  unsigned int u = __float_as_uint(f);
  // Round to nearest, ties to even -- the same rounding the packers use, so a
  // partial that happens to land on one rank alone is bit-identical to the
  // unsharded path.
  unsigned int lsb = (u >> 16) & 1u;
  u += 0x7FFFu + lsb;
  return (unsigned short)(u >> 16);
}

template <typename T,
          int BATCH_SIZE,
          int HIDDEN,
          int EP_WORLD_SIZE,
          int EP_MY_PE>
__device__ __forceinline__ void
    xrank_sum_add_kernel(void const *__restrict__ partial_ptr,
                         void const *__restrict__ residual_ptr,
                         void *__restrict__ xbuf_ptr,
                         void *__restrict__ output_ptr,
                         int num_active_tokens,
                         int step) {
  // Everything is bf16 and the arithmetic is done in fp32 by hand, so the
  // payload path never depends on T's operator overloads.
  static_assert(sizeof(T) == 2, "xrank_sum_add is a bf16 path");
  // 8 bf16 = one 16-byte write-through store.
  constexpr int VEC = 8;
  static_assert(HIDDEN % VEC == 0, "HIDDEN must be a multiple of 8");

  unsigned short const *__restrict__ partial =
      static_cast<unsigned short const *>(partial_ptr);
  unsigned short const *__restrict__ residual =
      static_cast<unsigned short const *>(residual_ptr);
  unsigned short *__restrict__ output =
      static_cast<unsigned short *>(output_ptr);
  unsigned short *__restrict__ payload =
      static_cast<unsigned short *>(xbuf_ptr);
  unsigned long long *__restrict__ epoch_words =
      reinterpret_cast<unsigned long long *>(
          payload + (size_t)EP_WORLD_SIZE * BATCH_SIZE * HIDDEN);

  int tidx = threadIdx.x;

  // Heap-wide constant deltas, sampled once. See mpk_comm.cuh.
  int64_t peer_delta[EP_WORLD_SIZE];
  bool peer_direct[EP_WORLD_SIZE];
#pragma unroll
  for (int p = 0; p < EP_WORLD_SIZE; p++) {
    peer_delta[p] = 0;
    peer_direct[p] = (p == EP_MY_PE) || mpk_shmem_peer_delta(p, &peer_delta[p]);
  }

  // ---- publish this rank's partial into every peer's slot -----------------
  unsigned short *my_slot =
      payload + (size_t)EP_MY_PE * BATCH_SIZE * HIDDEN;
  for (int b = 0; b < num_active_tokens; b++) {
    for (int j = tidx * VEC; j < HIDDEN; j += MPK_NT * VEC) {
      unsigned int const *src = reinterpret_cast<unsigned int const *>(
          partial + (size_t)b * HIDDEN + j);
      unsigned int v0 = src[0], v1 = src[1], v2 = src[2], v3 = src[3];
      char *dst = reinterpret_cast<char *>(my_slot + (size_t)b * HIDDEN + j);
#pragma unroll
      for (int p = 0; p < EP_WORLD_SIZE; p++) {
        if (peer_direct[p]) {
          st_wt_u128((void *)(dst + peer_delta[p]), v0, v1, v2, v3);
        }
      }
    }
  }
  // Each thread drains its own stores; the barrier then makes the whole
  // block's payload visible before anybody publishes an epoch.
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  __syncthreads();

  unsigned long long const epoch = (unsigned long long)step + 1ull;
  if (tidx == 0) {
#pragma unroll
    for (int p = 0; p < EP_WORLD_SIZE; p++) {
      if (peer_direct[p]) {
        st_wt_u64((void *)(reinterpret_cast<char *>(
                               epoch_words +
                               (size_t)EP_MY_PE * XRANK_SUM_SLOT_U64) +
                           peer_delta[p]),
                  epoch);
      } else {
        // No direct mapping. The staged put+signal fallback is a block
        // collective and cannot run under `tidx == 0`; reaching here means
        // mpk_shmem_init_peer_deltas found an unmapped peer, which the layer-0
        // EP fold would have tripped over first.
        __builtin_trap();
      }
    }
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  }
  __syncthreads();

  // ---- wait for all peers -------------------------------------------------
  // One thread per peer, EP_WORLD_SIZE <= 16 so the spin is a single wave.
  // sc0 sc1: a peer GPU is the writer and an L2-resident stale line is a
  // livelock, not a slow path.
  if (tidx < EP_WORLD_SIZE) {
    while (ld_sys_u64(epoch_words + (size_t)tidx * XRANK_SUM_SLOT_U64) <
           epoch) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
  __syncthreads();

  // ---- sum the four slots and add the residual ----------------------------
  for (int b = 0; b < num_active_tokens; b++) {
    for (int j = tidx * VEC; j < HIDDEN; j += MPK_NT * VEC) {
      float acc[VEC];
#pragma unroll
      for (int k = 0; k < VEC; k++) {
        acc[k] = xrank_sum_bf16_to_f32(residual[(size_t)b * HIDDEN + j + k]);
      }
#pragma unroll
      for (int p = 0; p < EP_WORLD_SIZE; p++) {
        unsigned long long *slot = reinterpret_cast<unsigned long long *>(
            payload + ((size_t)p * BATCH_SIZE + b) * HIDDEN + j);
        // Written by a peer, or written through by this block a moment ago --
        // either way this CU's vL1 may hold a pre-write line.
        //
        // MEASURED NEUTRAL, do not re-try: batching these into one
        // `global_load_dwordx4 x4 ... s_waitcnt vmcnt(0)` cuts the loop's
        // s_waitcnt count 15 -> 8 and is bit-identical, and the wall did not
        // move (off 9.180/9.147/9.173 vs on 9.153). ld_sys_u64's built-in
        // drain is not this task's cost -- the ~18 us it spans is the peer
        // WAIT above. It also made the 248-blocks-on-256-CUs bootstrap flaky
        // (1 clean start of 7 vs 3 of 3), so the batched form is gone.
        unsigned long long w0 = ld_sys_u64(slot);
        unsigned long long w1 = ld_sys_u64(slot + 1);
#pragma unroll
        for (int k = 0; k < 4; k++) {
          acc[k] += xrank_sum_bf16_to_f32(
              (unsigned short)((w0 >> (16 * k)) & 0xFFFFull));
          acc[4 + k] += xrank_sum_bf16_to_f32(
              (unsigned short)((w1 >> (16 * k)) & 0xFFFFull));
        }
      }
#pragma unroll
      for (int k = 0; k < VEC; k++) {
        output[(size_t)b * HIDDEN + j + k] = xrank_sum_f32_to_bf16(acc[k]);
      }
    }
  }
  __syncthreads();
}

} // namespace kernel
