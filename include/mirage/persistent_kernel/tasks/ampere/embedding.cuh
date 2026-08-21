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

template <typename T, int OUT_DIM>
__device__ __forceinline__ void
    single_embedding_kernel(void const *__restrict__ input_ptr,
                            void const *__restrict__ embedding_ptr,
                            void *__restrict__ output_ptr,
                            int step,
                            long long *tokens) {
  // int64_t const *__restrict__ input_ids =
  //     static_cast<int64_t const *>(input_ptr);
  T const *__restrict__ embedding = static_cast<T const *>(embedding_ptr);
  T *__restrict__ output = static_cast<T *>(output_ptr);
  constexpr int BATCH_SIZE = 1;

  for (int i = threadIdx.x; i < BATCH_SIZE * OUT_DIM; i += blockDim.x) {
    // int idx = i / OUT_DIM;
    int off = i % OUT_DIM;
    // int64_t wordIdx = input_ids[idx];
    int64_t wordIdx = tokens[step];
    output[i] = embedding[wordIdx * OUT_DIM + off];
  }
}

// Under -DMPK_BS_DEBUG the row-bisecting probe needs to know which token each
// row of a multi-token step actually carries: a per-stage checksum that
// disagrees between two arms is only evidence if both arms fed the same
// tokens in. Bounded to the first few steps for the same reason the checksum
// probe is -- a print that runs every iteration is an instrument that changes
// what it measures.
#if defined(MPK_BS_DEBUG) && MPK_BS_DEBUG
__device__ unsigned int g_embdbg_steps;
#endif

template <typename T, int BATCH_SIZE, int CHUNK_SIZE, int OUTPUT_DIM_SIZE>
__device__ __forceinline__ void
    embedding_kernel(void const *__restrict__ input_ptr,
                     void const *__restrict__ embedding_ptr,
                     void *__restrict__ output_ptr) {
  // barrier for first 4 worker warps
  // TODO:(Jianan Ji) In vllm, type should be int32_t instead of int64_t.
  int64_t const *__restrict__ input_ids =
      static_cast<int64_t const *>(input_ptr);
  T const *__restrict__ embedding = static_cast<T const *>(embedding_ptr);
  T *__restrict__ output = static_cast<T *>(output_ptr);

#if defined(MPK_BS_DEBUG) && MPK_BS_DEBUG
  if (threadIdx.x == 0) {
    unsigned int const s = atomicAdd(&g_embdbg_steps, 1u);
    if (s < 6u) {
      long long t0 = (long long)input_ids[0];
      long long t1 = (BATCH_SIZE > 1) ? (long long)input_ids[1] : -1;
      printf("[EMBDBG] step=%u bs=%d tok0=%lld tok1=%lld\n",
             s,
             BATCH_SIZE,
             t0,
             t1);
    }
  }
#endif

#pragma unroll
  for (int batch_idx = 0; batch_idx < BATCH_SIZE; batch_idx++) {
    int64_t wordIdx = input_ids[batch_idx];
#ifdef EMBED_DEBUG
    if (threadIdx.x == 0) {
      printf("[EMBED] wordIdx=%lld output_ptr=%p\n",
             (long long)wordIdx,
             output_ptr);
    }
#endif
    if (wordIdx >= 0) {
#pragma unroll
      for (int i = threadIdx.x; i < CHUNK_SIZE; i += blockDim.x) {
        output[batch_idx * OUTPUT_DIM_SIZE + i] =
            embedding[wordIdx * OUTPUT_DIM_SIZE + i];
      }
    } else {
      // TODO: This might not be necessary
      for (int i = threadIdx.x; i < CHUNK_SIZE;
           i += blockDim.x) { // writing 0 to output
        output[batch_idx * OUTPUT_DIM_SIZE + i] = T(0.0f);
      }
    }
#ifdef EMBED_DEBUG
    __syncthreads();
    if (threadIdx.x == 0) {
      float v0 = static_cast<float>(output[0]);
      float v1 = static_cast<float>(output[1]);
      float v2 = static_cast<float>(output[2]);
      printf("[EMBED] output[0..2]: %f %f %f\n", v0, v1, v2);
    }
#endif
  }

#if defined(MPK_BS_DEBUG) && MPK_BS_DEBUG
  // The same fingerprint the BSDBG stages print, taken on this kernel's OWN
  // output. Without it there is no way to tell "the embedding wrote the wrong
  // thing" from "something clobbered the residual between here and layer 0",
  // and the layer-0 stage-0 dump matches the offline embedding table for row 1
  // and matches nothing at all for row 0 -- which is exactly the ambiguity
  // this resolves. Guarded by the same step bound as the id print.
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned int const s2 = atomicAdd(&g_embdbg_steps, 0u);
    if (s2 <= 6u) {
      for (int b = 0; b < BATCH_SIZE; b++) {
        double sum = 0.0;
        float amax = 0.0f;
        for (int i = 0; i < OUTPUT_DIM_SIZE; i++) {
          float const v = (float)output[b * OUTPUT_DIM_SIZE + i];
          sum += (double)v;
          float const av = v < 0.0f ? -v : v;
          if (!(av <= amax)) {
            amax = av;
          }
        }
        printf("[EMBOUT] step=%u row=%d n=%d sum=%.6f amax=%.6f v0=%.6f "
               "v1=%.6f\n",
               s2 - 1u,
               b,
               OUTPUT_DIM_SIZE,
               sum,
               amax,
               (float)output[b * OUTPUT_DIM_SIZE + 0],
               (float)output[b * OUTPUT_DIM_SIZE + 1]);
      }
    }
  }
#endif
}

} // namespace kernel
