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

# Needs two peer-capable GPUs at run time, but builds anywhere.
echo "Building EP collective floor benchmark..."
hipcc -o test_ep_collective test_ep_collective.hip \
    -D__HIP_PLATFORM_AMD__ \
    --offload-arch=gfx950 \
    -munsafe-fp-atomics \
    -O3 \
    -std=c++17

echo "Build complete!"
echo ""
echo "Run with: ./test_mfma_simple"
echo "          ./test_mfma_pipeline_hazards [launches]   # gfx950 only"
echo "          HIP_VISIBLE_DEVICES=6,7 ./test_ep_collective [iters]"
