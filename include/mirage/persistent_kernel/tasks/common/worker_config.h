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
#if defined(__gfx1250__)
// MI450 (gfx1250) is wave32, not wave64.
//
// Keyed off the compiler's __gfx1250__ rather than MIRAGE_AMD_MI450: this
// header is included by host translation units too, and the value has to track
// the *device* wave width in the device pass. Note the consequence -- in a host
// pass __gfx1250__ is absent, so an mi450 host build still sees 64 here. That
// is intentional and matches arch_traits.cuh's own comment: host-side code uses
// this only for launch math, never for lane arithmetic.
//
// This is not cosmetic. NUM_THREADS_PER_WARP feeds warp_id() in utils.cuh and
// the butterfly width in argmax_mi300.cuh's warp_reduce_max_idx. At 64 on a
// 32-lane wave, warp_id() reports 4 waves where there are 8, and the argmax
// block reduction drops the partials of every odd wave -- measured returning
// 16 where the true maximum was 1000. Since argmax emits the output token,
// that is a wrong answer rather than a slow one.
constexpr int NUM_THREADS = 256;
constexpr int NUM_THREADS_PER_WARP = 32;
constexpr int NUM_WARPS = 8; // 256 / 32
#elif defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
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
