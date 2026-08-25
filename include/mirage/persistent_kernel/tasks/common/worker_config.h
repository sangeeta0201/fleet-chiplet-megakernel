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

// NUM_THREADS: The number of threads used for loop strides in kernels
// Must match the actual thread count used when launching kernels
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
// AMD MI300 uses 256 threads with native 64-thread wavefronts
constexpr int NUM_THREADS = 256;
constexpr int NUM_THREADS_PER_WARP = 64; // Native AMD wavefront size
constexpr int NUM_WARPS = 4;             // 256 / 64 = 4 wavefronts
#elif defined(MIRAGE_GRACE_HOPPER) || defined(MIRAGE_GRACE_BLACKWELL)
// Hopper and Blackwell use 256 threads with 32-thread warps
constexpr int NUM_THREADS = 256;
constexpr int NUM_THREADS_PER_WARP = 32;
constexpr int NUM_WARPS = 8; // 256 / 32 = 8 warps
#else
// Ampere uses 128 threads with 32-thread warps
constexpr int NUM_THREADS = 128;
constexpr int NUM_THREADS_PER_WARP = 32;
constexpr int NUM_WARPS = 4; // 128 / 32 = 4 warps
#endif
constexpr int WARPGROUP_WARPS = 4;

constexpr float inf = 5e4;
// TODO: only setting this for Hopper can have compilation issues on blackwell
// and presumably ampere
#if defined(MIRAGE_GRACE_HOPPER) || defined(MIRAGE_GRACE_BLACKWELL) ||         \
    defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
constexpr int WORKER_NUM_THREADS = 256;   // Grace Hopper/AMD MI300 setting
constexpr int CONSUMER_NUM_THREADS = 128; // Grace Hopper setting
#endif

// ---------------------------------------------------------------------------
// MPK_NT -- the grid-stride step in a task kernel.
//
// `blockDim.x` is NOT a register on gfx9. It is `__ockl_get_local_size(0)`,
// which reads the HSA dispatch packet:
//
//     s_load_dword s5, s[8:9], 0x0        // grid size, to detect a partial WG
//     s_cmp_lt_u32 s12, s5
//     s_cselect_b32 s5, 12, 18            // pick the packet field
//     global_load_ushort v4, v5, s[14:15]
//     s_waitcnt vmcnt(0)                  // <-- FULL VMEM DRAIN
//     v_add_u32_e32 v19, v19, v4          // sb += blockDim.x
//
// The compiler will not hoist it out of a loop, so every `x += blockDim.x`
// backedge pays a vector load plus an `s_waitcnt vmcnt(0)` that fences every
// outstanding VMEM op in the wave. Counted on the built image
// (`llvm-objdump -d -l`, MPK_LINE_TABLES=1) there were 18 such
// ushort-then-drain pairs attributed to amd_hip_runtime.h:263, and 44
// `global_load_ushort` in the fused MLA layer alone -- the hottest kernel in
// the model.
//
// Every task kernel here is launched from worker_kernel or persistent_kernel,
// both `dim3(WORKER_NUM_THREADS)` == `dim3(SINGLE_KERNEL_NUM_THREADS)` ==
// NUM_THREADS. prepare_kernel/scheduler_kernel run at 128 but call no task.
// So the value is a compile-time constant and naming it as one is free.
//
// It also unlocks a second-order win: with a literal step the compiler can
// prove a grid-stride loop whose trip count is below NUM_THREADS runs at most
// once and delete the backedge outright. W2's activation quantizer is exactly
// that shape (NSUBBLOCKS = 2048/32 = 64 against 256 threads).
//
// MPK_CONST_BLOCKDIM=0 restores the dispatch-packet read for A/B.
// ---------------------------------------------------------------------------
#ifndef MPK_CONST_BLOCKDIM
#define MPK_CONST_BLOCKDIM 1
#endif
#if MPK_CONST_BLOCKDIM
#define MPK_NT ((unsigned)NUM_THREADS)
#else
#define MPK_NT (blockDim.x)
#endif

// MEASURED, GLM-5, NP=4, devices 4-7, bs=1, one session, disjoint ranges:
//
//   MPK_CONST_BLOCKDIM=0   10.085 10.055 10.050 10.095   n=4 mean 10.071
//   MPK_CONST_BLOCKDIM=1    9.975  9.977  9.937          n=3 mean  9.963
//
//   delta -0.108 ms.  arm max 9.977 < control min 10.050.
//
// Image, same build recipe both arms (`llvm-objdump -d --mcpu=gfx950`):
//   total global_load_ushort        122 -> 66
//   ushort followed by vmcnt(0)      34 -> 8
//   persistent_kernel                13 -> 0
//   worker_kernel                    12 -> 0
//   topk_sigmoid_noinline             8 -> 0
//   fused MLA layer (x2 instances) 24/20 -> 18/18   (the rest are real bf16)
//
// A run at 111 ms/iter was excluded: all four ranks reported the identical
// avg_ms=102.264, i.e. a uniform box-wide stall, not a rank straggler.
