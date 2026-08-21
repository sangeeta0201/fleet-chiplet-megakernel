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

// Per-stage activation checksums, for bisecting a WRONG-OUTPUT build.
//
// Why this exists. Widening the decode path to BATCH_SIZE = 2 (MTP's verify
// step) builds and runs but produces garbage, and it does so with only ONE
// active row -- [FWD_PASS_TOTAL] iters= is identical to the bs=1 arm, so the
// token schedule is the same and the difference is the *build*, not the work.
// Five speculative A/Bs and a full read of every diff failed to localize it.
// A checksum after each stage turns that into one run per arm.
//
// Discipline this instrument obeys, because a debug print that perturbs the
// thing it measures is worse than no print (see the stage-stamp episode:
// four GPU-wide atomics cost ~9 us/layer each and landed inside the next
// interval they were timing):
//
//   * Compiled out entirely unless -DMPK_BS_DEBUG=<n> is passed. n is the
//     number of leading fused layers to instrument.
//   * ONE thread on ONE worker on ONE XCD executes anything at all, and only
//     the first time a (layer, stage) pair is reached -- so the whole run
//     emits MPK_BS_DEBUG * 8 lines per rank, once, on iteration 0. Every
//     other worker's cost is a compare against task_layer_idx.
//   * Called only immediately after the barrier that publishes the buffer it
//     reads, so it never races the producer.
//
// This is a correctness tool. Do NOT quote a latency measured with it on.

#pragma once

#ifndef MPK_BS_DEBUG
#define MPK_BS_DEBUG 0
#endif

#if MPK_BS_DEBUG

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

namespace kernel {

// [layer][stage]. 32 stages is well past what the layer has; the array is
// zero-initialized by the loader and never reset, which is the point -- the
// second iteration's visit finds a 1 and returns.
__device__ unsigned int g_bsdbg_seen[MPK_BS_DEBUG][32];

// Sum, absmax and the first two elements. The sum alone is a weak fingerprint
// (cancellation), the absmax catches a NaN/Inf row, and the first elements
// tell a "shifted by one row" bug from a "wrong values" bug at a glance.
__device__ __forceinline__ void mpk_bsdbg_bf16(int stage,
                                               int layer,
                                               void const *p,
                                               int n,
                                               int pe,
                                               char const *tag) {
  if (layer < 0 || layer >= MPK_BS_DEBUG || stage < 0 || stage >= 32) {
    return;
  }
  if (p == nullptr) {
    return;
  }
  if (atomicExch(&g_bsdbg_seen[layer][stage], 1u) != 0u) {
    return;
  }
  __hip_bfloat16 const *a = (__hip_bfloat16 const *)p;
  double sum = 0.0;
  float amax = 0.0f;
  for (int i = 0; i < n; i++) {
    float const v = __bfloat162float(a[i]);
    sum += (double)v;
    float const av = v < 0.0f ? -v : v;
    if (!(av <= amax)) { // NaN-safe: a NaN takes this branch
      amax = av;
    }
  }
  printf("[BSDBG] pe=%d layer=%d stage=%d %s n=%d sum=%.6f amax=%.6f "
         "v0=%.6f v1=%.6f\n",
         pe,
         layer,
         stage,
         tag,
         n,
         sum,
         amax,
         __bfloat162float(a[0]),
         n > 1 ? __bfloat162float(a[1]) : 0.0f);
}

} // namespace kernel

#define MPK_BSDBG(stage, layer, ptr, n, pe, tag)                               \
  do {                                                                         \
    if ((layer) < MPK_BS_DEBUG && tid == 0 && xcd_id == 0 && xcd_rank == 0) {  \
      ::kernel::mpk_bsdbg_bf16((stage), (layer), (ptr), (n), (pe), (tag));     \
    }                                                                          \
  } while (0)

#else

#define MPK_BSDBG(stage, layer, ptr, n, pe, tag)                               \
  do {                                                                         \
  } while (0)

#endif
