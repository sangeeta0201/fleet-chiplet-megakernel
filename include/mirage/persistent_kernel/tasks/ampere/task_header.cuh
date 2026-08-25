#ifndef MIRAGE_USE_CUTLASS_KERNEL
#define MIRAGE_USE_CUTLASS_KERNEL 1
#endif // MIRAGE_USE_CUTLASS_KERNEL

// MPK_NT (the grid-stride step) must be visible before any task header; see
// worker_config.h for why blockDim.x is not free on gfx9.
#include "../common/worker_config.h"
#include "argmax.cuh"
#include "embedding.cuh"
#include "identity.cuh"
#include "multitoken_paged_attention.cuh"
#include "reduction.cuh"
#include "rmsnorm.cuh"
#include "rotary_embedding.cuh"
#include "silu_mul.cuh"

#if MIRAGE_USE_CUTLASS_KERNEL
#include "linear_cutlass.cuh"
#include "moe_linear.cuh"
#else
#include "linear.cuh"
#endif // MIRAGE_USE_CUTLASS_KERNEL