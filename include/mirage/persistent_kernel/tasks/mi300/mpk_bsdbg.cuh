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

// Which forward pass to dump, expressed as the task_layer_idx of its layer 0.
//
// The obvious knob here -- "dump the Nth visit of each (layer, stage)" -- does
// not work, and the reason is worth stating because it silently produces an
// empty log. task_layer_idx is run-monotonic, not per-iteration: iteration k
// covers ml_num_layers * k .. ml_num_layers * k + ml_num_layers - 1 (see the
// pc_iter arithmetic in persistent_kernel.cuh). So `layer < MPK_BS_DEBUG`
// selects the first few layers of iteration 0 ONLY, and every (layer, stage)
// pair it admits is visited exactly once in the whole run.
//
// Offsetting the window is what makes two arms comparable. A bs=1 arm and a
// bs=2 arm consume different numbers of prompt tokens per step, so only their
// step-0 row 0 lines up by default; at 2 tokens per step, the bs=2 arm's
// iteration 0 row 1 is the same math as the bs=1 arm's iteration 1 row 0 --
// both are prompt position 1 attending over a KV cache holding position 0.
// GLM-5 reports "Multi-layer scan: found 76 fused layers", so iteration 1 is
// MPK_BSDBG_LAYER0=76.
#ifndef MPK_BSDBG_LAYER0
#define MPK_BSDBG_LAYER0 0
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
                                               int abs_layer,
                                               void const *p,
                                               int n,
                                               int pe,
                                               char const *tag,
                                               int rows,
                                               int stride) {
  int const layer = abs_layer - MPK_BSDBG_LAYER0;
  if (layer < 0 || layer >= MPK_BS_DEBUG || stage < 0 || stage >= 32) {
    return;
  }
  if (p == nullptr) {
    return;
  }
  if (atomicExch(&g_bsdbg_seen[layer][stage], 1u) != 0u) {
    return;
  }
  for (int r = 0; r < rows; r++) {
    __hip_bfloat16 const *a =
        (__hip_bfloat16 const *)p + (long long)r * (long long)stride;
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
    printf("[BSDBG] pe=%d abs=%d layer=%d stage=%d row=%d %s n=%d sum=%.6f "
           "amax=%.6f v0=%.6f v1=%.6f\n",
           pe,
           abs_layer,
           layer,
           stage,
           r,
           tag,
           n,
           sum,
           amax,
           __bfloat162float(a[0]),
           n > 1 ? __bfloat162float(a[1]) : 0.0f);
  }
}

// The same fingerprint, for a caller that has no task_layer_idx.
//
// The three dense prologue layers do not go through fuse_full_layer (it
// requires layer.is_moe), so they carry no layer index and MPK_BSDBG_N cannot
// see them at all -- which is exactly the blind spot the bs=2 hunt walked
// into: the embedding output is bit-identical between the bs=1 and bs=2 arms,
// the first *fused* layer's input is not, and everything in between was
// uninstrumented.
//
// Ordering instead of indexing. There is a barrier between every task, so the
// n-th visit of a given stage is the n-th layer to reach it -- no layer id
// needed. The gate is left to the caller (`tile_idx == 0 && threadIdx.x == 0`
// in a gang task is one thread per XCD, so a stage emits 8 lines per layer);
// `seq` is printed rather than divided so a caller with a different workgroup
// count is still readable.
//
// The eight lines are NOT eight copies of the same read, and assuming they
// were wasted a run. A gang task's tensors are partitioned across the eight
// workgroups by the task graph's input map -- the residual these probes read
// comes in as `new_input(residual, (1, -1, -1), ...)`, i.e. dim 1 sliced by
// bid.x -- so each XCD's pointer is a different 1/8 of the row. Measured: the
// eight stage-10 lines of one dense layer had eight different sums, exactly
// one of which (arrival order, not XCD order -- `seq` is an atomicAdd) matched
// the known embedding.
//
// Hence two things. The pointer is printed, so the eight lines of a layer can
// be sorted back into column order across arms whose allocations differ. And
// the caller must pass an `n` that fits inside ONE slice, or seven of the eight
// reads run off the end of their slice and, at BATCH_SIZE > 1, straight into
// the next row.
__device__ unsigned int g_bsdbg_seq[32];

__device__ __forceinline__ void mpk_bsdbg_seq_bf16(int stage,
                                                   void const *p,
                                                   int n,
                                                   char const *tag,
                                                   int rows,
                                                   int stride,
                                                   int limit) {
  if (stage < 0 || stage >= 32 || p == nullptr) {
    return;
  }
  unsigned int const seq = atomicAdd(&g_bsdbg_seq[stage], 1u);
  if (seq >= (unsigned int)limit) {
    return;
  }
  for (int r = 0; r < rows; r++) {
    __hip_bfloat16 const *a =
        (__hip_bfloat16 const *)p + (long long)r * (long long)stride;
    double sum = 0.0;
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
      float const v = __bfloat162float(a[i]);
      sum += (double)v;
      float const av = v < 0.0f ? -v : v;
      if (!(av <= amax)) {
        amax = av;
      }
    }
    // The pointer is not decoration. Gating on `tile_idx == 0` fires once per
    // XCD *and* once per task, and those two readings of the same seq stream
    // are indistinguishable from the values alone -- eight hits with eight
    // different sums is either eight tasks or one task whose eight XCDs raced
    // the producer. Same pointer across a run of eight says XCD; a cycling
    // pointer says task.
    printf("[DENSEDBG] seq=%u stage=%d row=%d %s p=%p n=%d sum=%.6f amax=%.6f "
           "v0=%.6f v1=%.6f\n",
           seq,
           stage,
           r,
           tag,
           p,
           n,
           sum,
           amax,
           __bfloat162float(a[0]),
           n > 1 ? __bfloat162float(a[1]) : 0.0f);
  }
}

} // namespace kernel

#define MPK_BSDBG_SEQ(stage, ptr, n, tag, rows, stride, limit)                 \
  do {                                                                         \
    ::kernel::mpk_bsdbg_seq_bf16(                                              \
        (stage), (ptr), (n), (tag), (rows), (stride), (limit));                \
  } while (0)

// `rows` x `stride` dumps more than the first token's row. At BATCH_SIZE > 1
// a bug that only touches row 1 is invisible in row 0, and row 0 is all the
// original one-row form could see -- which is how the two-genuine-row build
// got as far as "correct at one active row" while still emitting garbage at
// two. Callers that do not know their buffer's row stride pass rows=1.
#define MPK_BSDBG_N(stage, layer, ptr, n, pe, tag, rows, stride)               \
  do {                                                                         \
    if ((layer) < MPK_BSDBG_LAYER0 + MPK_BS_DEBUG && tid == 0 &&               \
        xcd_id == 0 && xcd_rank == 0) {                                        \
      ::kernel::mpk_bsdbg_bf16(                                                \
          (stage), (layer), (ptr), (n), (pe), (tag), (rows), (stride));        \
    }                                                                          \
  } while (0)

#define MPK_BSDBG(stage, layer, ptr, n, pe, tag)                               \
  MPK_BSDBG_N(stage, layer, ptr, n, pe, tag, 1, 0)

#else

#define MPK_BSDBG_N(stage, layer, ptr, n, pe, tag, rows, stride)               \
  do {                                                                         \
  } while (0)

#define MPK_BSDBG(stage, layer, ptr, n, pe, tag)                               \
  do {                                                                         \
  } while (0)

#define MPK_BSDBG_SEQ(stage, ptr, n, tag, rows, stride, limit)                 \
  do {                                                                         \
  } while (0)

#endif
