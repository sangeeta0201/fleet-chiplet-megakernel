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
// Cross-rank argmax reduce.
//
// The vocab-sharded LM head leaves each EP rank holding the argmax over its own
// VOCAB_SHARD rows and nothing else. This task finishes the job: local reduce
// exactly as argmax_reduce_kernel does, rebase the winning index into the
// global vocab, then exchange the four (value, index) pairs peer-to-peer and
// take the max. Output is bit-identical on every rank, which is what
// correctness_gate.py's G1 tests.
//
// Why it is worth a rendezvous: the LM head is 154880 x 6144. In MXFP8 with its
// E8M0 scales that is 983 MB read per rank per token, identically on all four
// ranks. One rendezvous is ~3.77 us against ~170 us of duplicated weight read.
//
// The payload is one 64-bit word, so the whole exchange is two stores per peer:
//
//     payload = (order_key(value) << 32) | ~(uint32)global_index
//     epoch   = step + 1
//
// order_key is the standard order-preserving float -> uint32 map, so an
// unsigned compare on `payload` ranks by value first. The index is stored
// COMPLEMENTED so that, among equal values, the max payload is the SMALLEST
// index -- matching torch.argmax's first-wins tie-break, and matching what the
// single-rank argmax_reduce_kernel already does (its `>` comparisons keep the
// earlier index).
//
// Payload and epoch are separate words with a drain between them: a peer that
// sees the epoch must see the payload. A single word cannot carry both, and
// double-buffering by parity is not a substitute -- two steps apart the payload
// can repeat exactly, so the reader has nothing to edge-detect on.
// ---------------------------------------------------------------------------
#include "comm/mpk_comm.cuh"
#include "mpk_atoms.cuh"
#include "tasks/common/common_header.cuh"
#include "tasks/mi300/argmax_mi300.cuh"

namespace kernel {

// 64 bytes per PE: payload at word 0, epoch at word 1, the rest padding so two
// peers' stores never share a line.
#define ARGMAX_XRANK_SLOT_U64 8

__device__ __forceinline__ unsigned int argmax_xrank_order_key(float v) {
  unsigned int u = __float_as_uint(v);
  // Negatives reverse under an unsigned compare, so flip every bit; positives
  // only need the sign bit set to sort above them.
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

template <typename T,
          int BATCH_SIZE,
          int CHUNK_SIZE,
          int NUM_PARTIAL_TASKS,
          int EP_WORLD_SIZE,
          int EP_MY_PE,
          int VOCAB_SHARD>
__device__ __forceinline__ void
    argmax_reduce_xrank_kernel(void const *__restrict__ input_val_ptr,
                               void const *__restrict__ input_idx_ptr,
                               void *__restrict__ xrank_ptr,
                               void *__restrict__ final_output_ptr,
                               int num_active_tokens,
                               int step) {
  T const *__restrict__ partial_vals = static_cast<T const *>(input_val_ptr);
  long long const *__restrict__ partial_idxs =
      static_cast<long long const *>(input_idx_ptr);
  long long *__restrict__ final_output =
      static_cast<long long *>(final_output_ptr);
  unsigned long long *__restrict__ xr =
      static_cast<unsigned long long *>(xrank_ptr);

  int tidx = threadIdx.x;

  // The deltas are heap-wide and constant for the process lifetime, so this is
  // an add per peer, not a translation per address. See mpk_comm.cuh.
  int64_t peer_delta[EP_WORLD_SIZE];
  bool peer_direct[EP_WORLD_SIZE];
#pragma unroll
  for (int p = 0; p < EP_WORLD_SIZE; p++) {
    peer_delta[p] = 0;
    peer_direct[p] = (p == EP_MY_PE) || mpk_shmem_peer_delta(p, &peer_delta[p]);
  }

#pragma unroll
  for (int batch_idx = 0; batch_idx < num_active_tokens; batch_idx++) {
    // ---- local reduce over this rank's own vocab slice ---------------------
    T local_max = T(-inf);
    long long local_packed_idx = -1;

#pragma unroll
    for (int i = tidx; i < NUM_PARTIAL_TASKS; i += MPK_NT) {
      T current_val = partial_vals[i + batch_idx * NUM_PARTIAL_TASKS];
      if (current_val > local_max) {
        local_max = current_val;
        local_packed_idx =
            ((long long)i << 32) | partial_idxs[i + batch_idx * NUM_PARTIAL_TASKS];
      }
    }
    block_reduce_max_idx(local_max, local_packed_idx);

    // Epoch has to be per (step, batch slot): two batch slots in one step would
    // otherwise reuse the same threshold and the second could read the first's
    // payload. num_active_tokens is the slot count, so stride by BATCH_SIZE.
    unsigned long long const epoch =
        (unsigned long long)((long long)step * BATCH_SIZE + batch_idx) + 1ull;

    if (tidx == 0) {
      unsigned long long payload;
      if (local_packed_idx == -1) {
        // No live column on this rank. 0x007FFFFF is order_key(-inf), which
        // loses to any real value; the index is complemented, so all-ones
        // here is index 0 and it never wins.
        payload = (0x007FFFFFull << 32) | 0xFFFFFFFFull;
      } else {
        long long chunk = local_packed_idx >> 32;
        long long rel = local_packed_idx & 0xFFFFFFFF;
        unsigned int g =
            (unsigned int)(EP_MY_PE * (long long)VOCAB_SHARD +
                           chunk * CHUNK_SIZE + rel);
        payload =
            ((unsigned long long)argmax_xrank_order_key((float)local_max) << 32) |
            (unsigned long long)(~g);
      }

      size_t const my_word = (size_t)EP_MY_PE * ARGMAX_XRANK_SLOT_U64;
#pragma unroll
      for (int p = 0; p < EP_WORLD_SIZE; p++) {
        if (peer_direct[p]) {
          st_wt_u64((void *)(reinterpret_cast<char *>(xr + my_word) +
                             peer_delta[p]),
                    payload);
        }
      }
      // One drain for all EP_WORLD_SIZE payload stores, then the epochs. Same
      // shape as the full-layer fold's publish: distinct peers, so the stores
      // do not serialise on each other.
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#pragma unroll
      for (int p = 0; p < EP_WORLD_SIZE; p++) {
        if (peer_direct[p]) {
          st_wt_u64((void *)(reinterpret_cast<char *>(xr + my_word + 1) +
                             peer_delta[p]),
                    epoch);
        } else {
          // No direct mapping: fall back to the backend's put+signal, which is
          // a block-collective call and cannot run under `tidx == 0`. Reaching
          // here means mpk_shmem_init_peer_deltas found an unmapped peer, and
          // the EP fold would have taken its own staged path at layer 0 long
          // before the LM head ran.
          __builtin_trap();
        }
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }
    __syncthreads();

    // ---- wait for all peers, then max over the EP_WORLD_SIZE payloads ------
    // One thread per peer. EP_WORLD_SIZE <= 16 < one wavefront, so the spin is
    // a single wave and the reduce below is a wave reduce with no LDS.
    unsigned long long mine = 0ull;
    if (tidx < EP_WORLD_SIZE) {
      unsigned long long *slot = xr + (size_t)tidx * ARGMAX_XRANK_SLOT_U64;
      // sc0 sc1: a peer GPU is the writer, so an L2-resident stale line is a
      // livelock, not a slow path. See ld_sys_u64's comment.
      while (ld_sys_u64(slot + 1) < epoch) {
        __builtin_amdgcn_s_sleep(1);
      }
      mine = ld_sys_u64(slot);
    }
    // Max-reduce across the first EP_WORLD_SIZE lanes.
#pragma unroll
    for (int off = NUM_THREADS_PER_WARP / 2; off > 0; off /= 2) {
      unsigned int lo = (unsigned int)(mine & 0xFFFFFFFFull);
      unsigned int hi = (unsigned int)(mine >> 32);
      unsigned int olo = __shfl_down(lo, off, NUM_THREADS_PER_WARP);
      unsigned int ohi = __shfl_down(hi, off, NUM_THREADS_PER_WARP);
      unsigned long long other =
          ((unsigned long long)ohi << 32) | (unsigned long long)olo;
      if (other > mine) {
        mine = other;
      }
    }

    if (tidx == 0) {
      final_output[batch_idx] =
          (long long)(unsigned int)(~(unsigned int)(mine & 0xFFFFFFFFull));
    }
    __syncthreads();
  }
}

} // namespace kernel
