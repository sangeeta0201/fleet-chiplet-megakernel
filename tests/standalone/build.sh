#!/bin/bash
# Build script for standalone kernel tests

set -e

echo "Building MFMA GEMM test..."
hipcc -o test_mfma_simple test_mfma_simple.hip \
    -D__HIP_PLATFORM_AMD__ \
    -munsafe-fp-atomics \
    -O3 \
    -std=c++17

# gfx950-only: the scaled-MFMA hazard regression uses
# v_mfma_scale_f32_16x16x128_f8f6f4, which does not assemble on other targets.
if [ "${GFX_ARCH:-gfx950}" = "gfx950" ]; then
    echo "Building scaled-MFMA pipeline hazard regression..."
    hipcc -o test_mfma_pipeline_hazards test_mfma_pipeline_hazards.hip \
        -D__HIP_PLATFORM_AMD__ \
        --offload-arch=gfx950 \
        -munsafe-fp-atomics \
        -O3 \
        -std=c++17
fi

echo "Building GLM-5 sigmoid+bias MoE router test..."
hipcc -o test_moe_topk_sigmoid_bias test_moe_topk_sigmoid_bias.hip \
    -D__HIP_PLATFORM_AMD__ \
    -DMIRAGE_BACKEND_USE_ROCM \
    --offload-arch="${GFX_ARCH:-gfx950}" \
    -munsafe-fp-atomics \
    -O3 \
    -std=c++17 \
    -Wno-unused-result \
    -I ../../include \
    -I ../../include/mirage/persistent_kernel

echo "Building GLM-5 partial+interleaved RoPE test..."
hipcc -o test_rope_interleave_partial test_rope_interleave_partial.hip \
    -D__HIP_PLATFORM_AMD__ \
    -DMIRAGE_BACKEND_USE_ROCM \
    --offload-arch="${GFX_ARCH:-gfx950}" \
    -munsafe-fp-atomics \
    -O3 \
    -std=c++17 \
    -Wno-unused-result \
    -I ../../include \
    -I ../../include/mirage/persistent_kernel

echo "Building GLM-5 absorbed MLA decode test..."
hipcc -o test_mla_decode test_mla_decode.hip \
    -D__HIP_PLATFORM_AMD__ \
    -DMIRAGE_BACKEND_USE_ROCM \
    --offload-arch="${GFX_ARCH:-gfx950}" \
    -munsafe-fp-atomics \
    -O3 \
    -std=c++17 \
    -Wno-unused-result \
    -I ../../include \
    -I ../../include/mirage/persistent_kernel

echo "Build complete!"
echo ""
echo "Run with: ./test_mfma_simple"
echo "          ./test_mfma_pipeline_hazards [launches]   # gfx950 only"
echo "          ./test_moe_topk_sigmoid_bias [num_rows]"
echo "          ./test_rope_interleave_partial"
echo "          ./test_mla_decode"
