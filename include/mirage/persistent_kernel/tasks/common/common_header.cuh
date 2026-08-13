/* Copyright 2025 The Mirage Team
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

#include "bfloat16.h"
#include "copy_sm80.cuh"
#include "dmem_layout.cuh"
#include "runtime_header.h"
#include "utils.cuh"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>

// RMSNorm reciprocal: rsqrt(sumsq/dim + eps).
//
// Written as a divide, `sumsq / (float)DIM + eps` costs a full IEEE-754
// sequence (v_div_scale x2, v_rcp, Newton-Raphson refinement, v_div_fmas,
// v_div_fixup) even though DIM is a compile-time template int -- without
// fast-math the compiler may not fold a reciprocal that is inexact, and
// 1/2880 is inexact. Hoisting the reciprocal to a literal collapses it to one
// v_fmac_f32. The result differs from the divide by <=1 ulp before the rsqrt,
// which is far below the bf16 the normalized activations are stored in.
//
// Set -DMPK_RMSNORM_EXACT_DIV=1 to restore the divide (A/B / bisection).
#if defined(MPK_RMSNORM_EXACT_DIV) && MPK_RMSNORM_EXACT_DIV
#define MPK_RMS_RCP(sumsq, DIM, eps) rsqrtf((sumsq) / (float)(DIM) + (eps))
#else
#define MPK_RMS_RCP(sumsq, DIM, eps)                                           \
  rsqrtf(fmaf((sumsq), 1.0f / (float)(DIM), (eps)))
#endif
