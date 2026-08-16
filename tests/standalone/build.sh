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

echo "Building GLM-5 absorbed MLA latent cache append test..."
hipcc -o test_mla_kv_cache_update test_mla_kv_cache_update.hip \
    -D__HIP_PLATFORM_AMD__ \
    -DMIRAGE_BACKEND_USE_ROCM \
    --offload-arch="${GFX_ARCH:-gfx950}" \
    -munsafe-fp-atomics \
    -O3 \
    -std=c++17 \
    -Wno-unused-result \
    -I ../../include \
    -I ../../include/mirage/persistent_kernel

# gfx950-only, same reason as the hazard regression above.
if [ "${GFX_ARCH:-gfx950}" = "gfx950" ]; then
    echo "Building MXFP8 scaled-MFMA operand layout probe..."
    hipcc -o test_mxfp8_mfma_layout test_mxfp8_mfma_layout.hip \
        -D__HIP_PLATFORM_AMD__ \
        --offload-arch=gfx950 \
        -munsafe-fp-atomics \
        -O3 \
        -std=c++17

    echo "Building MXFP8 dense gang linear test..."
    hipcc -o test_mxfp8_linear test_mxfp8_linear.hip \
        -D__HIP_PLATFORM_AMD__ \
        -DMIRAGE_BACKEND_USE_ROCM \
        --offload-arch=gfx950 \
        -munsafe-fp-atomics \
        -O3 \
        -std=c++17 \
        -Wno-unused-result \
        -I ../../include \
        -I ../../include/mirage/persistent_kernel

    echo "Building MXFP8 MoE W13/W2 gang linear test..."
    hipcc -o test_mxfp8_moe test_mxfp8_moe.hip \
        -D__HIP_PLATFORM_AMD__ \
        -DMIRAGE_BACKEND_USE_ROCM \
        --offload-arch=gfx950 \
        -munsafe-fp-atomics \
        -O3 \
        -std=c++17 \
        -Wno-unused-result \
        -I ../../include \
        -I ../../include/mirage/persistent_kernel

    # Both of these pull in gang_gemv_mxfp8_mi300.cuh, whose WRITE_THROUGH
    # epilogue needs st_wt_u16 out of mpk_atoms.cuh, and whose CK include chain
    # needs the vendored composable_kernel headers.
    echo "Building narrow-tile MXFP8 GEMV bandwidth probe..."
    hipcc -o test_gemv_mxfp8_bw test_gemv_mxfp8_bw.hip \
        -D__HIP_PLATFORM_AMD__ \
        -DMIRAGE_BACKEND_USE_ROCM \
        -DCK_TILE_FMHA_FWD_FAST_EXP2=1 \
        --offload-arch=gfx950 \
        -munsafe-fp-atomics \
        -O3 \
        -std=c++17 \
        -Wno-unused-result \
        -I ../../include \
        -I ../../include/mirage/persistent_kernel \
        -I ../../deps/composable_kernel/include

    echo "Building narrow-tile MXFP8 GEMV accuracy test..."
    hipcc -o test_gemv_mxfp8_accuracy test_gemv_mxfp8_accuracy.hip \
        -D__HIP_PLATFORM_AMD__ \
        -DMIRAGE_BACKEND_USE_ROCM \
        -DCK_TILE_FMHA_FWD_FAST_EXP2=1 \
        --offload-arch=gfx950 \
        -munsafe-fp-atomics \
        -O3 \
        -std=c++17 \
        -Wno-unused-result \
        -I ../../include \
        -I ../../include/mirage/persistent_kernel \
        -I ../../deps/composable_kernel/include
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
echo "          ./test_moe_topk_sigmoid_bias [num_rows] [num_shared]"
echo "          ./test_rope_interleave_partial"
echo "          ./test_mla_decode"
echo "          ./test_mla_kv_cache_update"
echo "          ./test_mxfp8_mfma_layout                  # gfx950 only"
echo "          ./test_mxfp8_linear                       # gfx950 only"
echo "          ./test_mxfp8_moe                          # gfx950 only"
echo "          ./test_gemv_mxfp8_bw                      # gfx950 only, ~2 min"
echo "          ./test_gemv_mxfp8_accuracy                # gfx950 only"
echo "          HIP_VISIBLE_DEVICES=6,7 ./test_ep_collective [iters]"
