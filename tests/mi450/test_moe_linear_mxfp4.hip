/* Exercises the MI450 MoE MXFP4 inner loop and epilogue against a host
 * reference.
 *
 * Scope, stated honestly: this drives the *arithmetic* of
 * gang_moe_linear_mxfp4_kernel_mi450 -- the FP4 token quantization, the
 * interleaved weight load, the packed scale operand, the K-reduction, and the
 * accumulator-to-output mapping. It deliberately does NOT drive the gang
 * dispatch, the XCD tile distribution, or the barrier, because FFM-Lite runs
 * workgroups serially and would pass those regardless of whether they are
 * correct on silicon. Those need hardware.
 *
 * The reference recomputes the whole pipeline independently: it quantizes the
 * token itself, decodes E2M1 and E8M0 from explicit tables, and does the dot
 * product in double. So a shared misunderstanding between kernel and reference
 * would have to be duplicated in two quite different pieces of code.
 *
 * TWO LOOP VARIANTS are run over identical data:
 *
 *   naive     -- one K-tile per trip, generic pointers. The original port.
 *   pipelined -- the shipping form: global-address-space loads, GLOBAL_PREFETCH
 *                one tile ahead, and two register buffers ping-ponged so the
 *                loads for tile i+1 issue before the WMMA for tile i.
 *
 * Both are checked against the reference, and against each other bit-exactly.
 * The optimized loop reorders memory operations but not the arithmetic, so any
 * difference between the two is a bug by construction -- no tolerance applies.
 *
 * Two K values are used: K=512 (4 K-tiles, even) and K=384 (3 K-tiles, odd).
 * The odd one exists because the ping-pong loop steps two tiles per trip and
 * handles the remainder in a separate tail; K=512 alone would never execute
 * that tail, and a dropped final tile is exactly the kind of error that leaves
 * a plausible-looking result.
 *
 *   ./run_ffm.sh /abs/path/to/test_moe_linear_mxfp4
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <vector>

#include "mirage/persistent_kernel/tasks/mi450/mx_layout_mi450.cuh"

#define HIP_CHECK(x)                                                           \
  do {                                                                         \
    hipError_t e = (x);                                                        \
    if (e != hipSuccess) {                                                     \
      printf("HIP error %s at line %d\n", hipGetErrorString(e), __LINE__);     \
      return 2;                                                                \
    }                                                                          \
  } while (0)

static float const kE2M1[16] = {0.0f,
                                0.5f,
                                1.0f,
                                1.5f,
                                2.0f,
                                3.0f,
                                4.0f,
                                6.0f,
                                -0.0f,
                                -0.5f,
                                -1.0f,
                                -1.5f,
                                -2.0f,
                                -3.0f,
                                -4.0f,
                                -6.0f};

static double e8m0(uint8_t e) {
  return std::ldexp(1.0, (int)e - 127);
}

constexpr int OUT_ROWS = 128;
constexpr int NUM_WAVES = 8;
constexpr int N_TILES = OUT_ROWS / 16;
constexpr int TILES_PER_WAVE = N_TILES / NUM_WAVES;

#if defined(__gfx1250__) || !defined(__HIP_DEVICE_COMPILE__)

#if defined(MIRAGE_ARCH_GFX1250)
// Phase 1, shared by both variants: quantize the bf16 token to FP4 in LDS.
template <int K>
__device__ void quantize_token(unsigned short const *tok_bf16,
                               uint8_t *s_tok_fp4,
                               uint8_t *s_tok_scales) {
  using namespace kernel::mi450;
  constexpr int NBLK = K / 32;
  int tid = threadIdx.x;
  // Global-address-space cast, matching the kernel: the activation is a kernel
  // argument, so without it the compiler emits FLAT for a purely global read.
  using g_ushort = __attribute__((address_space(1))) unsigned short const;
  g_ushort *g_src = (g_ushort *)tok_bf16;
  for (int blk = tid; blk < NBLK; blk += blockDim.x) {
    int base = blk * 32;
    float vals[32];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 32; j++) {
      union {
        float f;
        unsigned u;
      } v;
      v.u = ((unsigned)g_src[base + j]) << 16;
      vals[j] = v.f;
      amax = fmaxf(amax, fabsf(vals[j]));
    }
    uint8_t se;
    if (amax == 0.0f) {
      se = 0;
    } else {
      union {
        float f;
        uint32_t u;
      } v;
      v.f = amax * (1.0f / 6.0f);
      int raw_exp = (int)((v.u >> 23) & 0xFF);
      if (v.u & 0x7FFFFF) {
        raw_exp++;
      }
      se = (uint8_t)max(0, min(255, raw_exp));
    }
    s_tok_scales[blk] = se;
    float inv_scale;
    if (se == 0) {
      inv_scale = 1.0f;
    } else {
      union {
        float f;
        uint32_t u;
      } sv;
      sv.u = (uint32_t)(254 - se) << 23;
      inv_scale = sv.f;
    }
    // Dword-at-a-time pack, matching the kernel. See the rationale there:
    // buffering all four dwords for one b128 store costs 44 extra VGPRs.
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      uint32_t acc_w = 0;
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        int j = w * 8 + b * 2;
        uint8_t lo = mx_encode_e2m1(vals[j] * inv_scale);
        uint8_t hi = mx_encode_e2m1(vals[j + 1] * inv_scale);
        acc_w |= (uint32_t)(uint8_t)(lo | (hi << 4)) << (8 * b);
      }
      *(uint32_t *)(s_tok_fp4 + base / 2 + w * 4) = acc_w;
    }
  }
  __syncthreads();
}
#endif

// ---- variant A: the original one-tile-per-trip loop, generic pointers -------
template <int K>
__global__ __launch_bounds__(256) void moe_naive(unsigned short const *tok_bf16,
                                                 uint8_t const *wg_data,
                                                 uint8_t const *wg_scales,
                                                 unsigned short const *bias,
                                                 float *out,
                                                 uint8_t *dbg_tok_fp4,
                                                 uint8_t *dbg_tok_scales) {
#if defined(MIRAGE_ARCH_GFX1250)
  using namespace kernel::mi450;
  constexpr int NBLK = K / 32;
  __shared__ uint8_t s_tok_fp4[K / 2];
  __shared__ uint8_t s_tok_scales[NBLK];

  quantize_token<K>(tok_bf16, s_tok_fp4, s_tok_scales);

  // Publish the quantized token so the host reference can consume the exact
  // same bits -- otherwise a quantization difference would masquerade as a
  // layout error.
  for (int i = threadIdx.x; i < K / 2; i += blockDim.x) {
    dbg_tok_fp4[i] = s_tok_fp4[i];
  }
  for (int i = threadIdx.x; i < NBLK; i += blockDim.x) {
    dbg_tok_scales[i] = s_tok_scales[i];
  }

  int const tid = threadIdx.x;
  int const warp_id = tid >> 5;
  int const lane_id = tid & 31;
  int const col = mx_operand_mn(lane_id);

  for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; ++tile_iter) {
    int wave_tile = warp_id + tile_iter * NUM_WAVES;
    mx_f32x8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    int const w_row = wave_tile * 16 + col;

#pragma unroll 1
    for (int kt = 0; kt < K; kt += MX_K_PER_TILE) {
      mx_i32x16_t a_reg = mx_load_operand_fp4(wg_data, w_row, kt, K);
      mx_i32x16_t b_reg = mx_load_operand_fp4(s_tok_fp4, 0, kt, K);
      unsigned int sa = mx_pack_scales(wg_scales + w_row * NBLK + kt / 32);
      unsigned int sb = mx_pack_scales(s_tok_scales + kt / 32);
      acc = _gang_wmma_f4xf4(a_reg, b_reg, acc, (int)sa, (int)sb);
    }

    if (col == 0) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        int row = mx_acc_row(lane_id, i);
        int out_n = wave_tile * 16 + row;
        union {
          float f;
          unsigned u;
        } b;
        b.u = ((unsigned)bias[out_n]) << 16;
        out[out_n] = acc[i] + b.f;
      }
    }
  }
#endif
}

// ---- variant B: the shipping loop ------------------------------------------
// Structurally identical to gang_moe_linear_mxfp4_kernel_mi450's phase 2:
// address-space-cast global loads, prefetch one tile ahead, two register
// buffers ping-ponged, odd tail handled separately.
template <int K>
__global__ __launch_bounds__(256) void moe_pipelined(
    unsigned short const *tok_bf16,
    uint8_t const *wg_data,
    uint8_t const *wg_scales,
    unsigned short const *bias,
    float *out) {
#if defined(MIRAGE_ARCH_GFX1250)
  using namespace kernel::mi450;
  constexpr int NBLK = K / 32;
  constexpr int K_PER_WMMA = MX_K_PER_TILE;
  constexpr int K_ITERS = K / K_PER_WMMA;
  __shared__ uint8_t s_tok_fp4[K / 2];
  __shared__ uint8_t s_tok_scales[NBLK];

  quantize_token<K>(tok_bf16, s_tok_fp4, s_tok_scales);

  int const tid = threadIdx.x;
  int const warp_id = tid >> 5;
  int const lane_id = tid & 31;
  int const col = mx_operand_mn(lane_id);

  mx_global_u8 *g_data = mx_to_global(wg_data);
  mx_global_u8 *g_scales = mx_to_global(wg_scales);

  for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; ++tile_iter) {
    int wave_tile = warp_id + tile_iter * NUM_WAVES;
    mx_f32x8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    int const w_row = wave_tile * 16 + col;

    mx_i32x16_t a0, a1, b0, b1;
    unsigned int sa0, sa1, sb0, sb1;

    a0 = mx_load_operand_fp4_global(g_data, w_row, 0, K);
    b0 = mx_load_operand_fp4(s_tok_fp4, 0, 0, K);
    sa0 = mx_pack_scales_global(g_scales + w_row * NBLK);
    sb0 = mx_pack_scales(s_tok_scales);

#pragma unroll 1
    for (int kt = 0; kt + 2 * K_PER_WMMA <= K; kt += 2 * K_PER_WMMA) {
      int const k1 = kt + K_PER_WMMA;
      int const k2 = kt + 2 * K_PER_WMMA;

      a1 = mx_load_operand_fp4_global(g_data, w_row, k1, K);
      b1 = mx_load_operand_fp4(s_tok_fp4, 0, k1, K);
      sa1 = mx_pack_scales_global(g_scales + w_row * NBLK + k1 / 32);
      sb1 = mx_pack_scales(s_tok_scales + k1 / 32);

      if (k2 < K) {
        mx_prefetch_operand_fp4_global(g_data, w_row, k2, K);
      }

      acc = _gang_wmma_f4xf4(a0, b0, acc, (int)sa0, (int)sb0);

      if (k2 < K) {
        a0 = mx_load_operand_fp4_global(g_data, w_row, k2, K);
        b0 = mx_load_operand_fp4(s_tok_fp4, 0, k2, K);
        sa0 = mx_pack_scales_global(g_scales + w_row * NBLK + k2 / 32);
        sb0 = mx_pack_scales(s_tok_scales + k2 / 32);
      }

      acc = _gang_wmma_f4xf4(a1, b1, acc, (int)sa1, (int)sb1);
    }

    if constexpr (K_ITERS % 2 == 1) {
      acc = _gang_wmma_f4xf4(a0, b0, acc, (int)sa0, (int)sb0);
    }

    if (col == 0) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        int row = mx_acc_row(lane_id, i);
        int out_n = wave_tile * 16 + row;
        union {
          float f;
          unsigned u;
        } b;
        b.u = ((unsigned)bias[out_n]) << 16;
        out[out_n] = acc[i] + b.f;
      }
    }
  }
#endif
}
#endif

template <int K>
static int run_shape() {
  constexpr int NBLK = K / 32;
  printf("--- shape: OUT_ROWS=%d K=%d (%d K-tiles, %s), %d wave32s, %d "
         "tile(s)/wave ---\n",
         OUT_ROWS,
         K,
         K / 128,
         (K / 128) % 2 ? "odd -> exercises tail" : "even",
         NUM_WAVES,
         TILES_PER_WAVE);

  uint32_t s = 424242u + K;
  auto rnd = [&]() {
    s = s * 1664525u + 1013904223u;
    return s >> 16;
  };

  // Weights: row-major FP4 nibble-packed [OUT_ROWS][K/2] + E8M0 scales.
  std::vector<uint8_t> wcode(OUT_ROWS * K);
  std::vector<uint8_t> wg_data(OUT_ROWS * K / 2, 0);
  std::vector<uint8_t> wg_scales(OUT_ROWS * NBLK);
  for (auto &v : wcode) {
    v = rnd() % 16;
  }
  for (int r = 0; r < OUT_ROWS; ++r) {
    for (int k = 0; k < K; ++k) {
      uint8_t &byte = wg_data[r * (K / 2) + k / 2];
      byte |= (k & 1) ? (wcode[r * K + k] << 4) : wcode[r * K + k];
    }
  }
  for (auto &v : wg_scales) {
    v = 125 + rnd() % 5;
  }

  // Token: bf16, spanning a few magnitudes so the per-block E8M0 scales differ
  // and a dropped scale byte cannot hide.
  std::vector<unsigned short> tok(K);
  std::vector<float> tok_f(K);
  for (int k = 0; k < K; ++k) {
    float mag = std::ldexp(1.0f, (int)(k / 32) % 6 - 3);
    float x = mag * ((float)(rnd() % 2001) / 1000.0f - 1.0f);
    union {
      float f;
      unsigned u;
    } v;
    v.f = x;
    unsigned bias_r = ((v.u >> 16) & 1) + 0x7FFF;
    tok[k] = (unsigned short)((v.u + bias_r) >> 16);
    v.u = ((unsigned)tok[k]) << 16;
    tok_f[k] = v.f; // exactly what the device will see
  }

  std::vector<unsigned short> bias(OUT_ROWS);
  for (int i = 0; i < OUT_ROWS; ++i) {
    union {
      float f;
      unsigned u;
    } v;
    v.f = (float)((int)(rnd() % 21) - 10) * 0.25f;
    unsigned br = ((v.u >> 16) & 1) + 0x7FFF;
    bias[i] = (unsigned short)((v.u + br) >> 16);
  }

  unsigned short *dTok, *dBias;
  uint8_t *dW, *dS, *dTokFp4, *dTokSc;
  float *dOutN, *dOutP;
  HIP_CHECK(hipMalloc(&dTok, tok.size() * 2));
  HIP_CHECK(hipMalloc(&dBias, bias.size() * 2));
  HIP_CHECK(hipMalloc(&dW, wg_data.size()));
  HIP_CHECK(hipMalloc(&dS, wg_scales.size()));
  HIP_CHECK(hipMalloc(&dTokFp4, K / 2));
  HIP_CHECK(hipMalloc(&dTokSc, NBLK));
  HIP_CHECK(hipMalloc(&dOutN, OUT_ROWS * sizeof(float)));
  HIP_CHECK(hipMalloc(&dOutP, OUT_ROWS * sizeof(float)));
  HIP_CHECK(hipMemcpy(dTok, tok.data(), tok.size() * 2, hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(dBias, bias.data(), bias.size() * 2, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dW, wg_data.data(), wg_data.size(), hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(dS, wg_scales.data(), wg_scales.size(), hipMemcpyHostToDevice));

  std::vector<float> poison(OUT_ROWS, -99999.f);
  HIP_CHECK(hipMemcpy(dOutN,
                      poison.data(),
                      poison.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dOutP,
                      poison.data(),
                      poison.size() * sizeof(float),
                      hipMemcpyHostToDevice));

  hipLaunchKernelGGL(moe_naive<K>, dim3(1), dim3(256), 0, 0, dTok, dW, dS,
                     dBias, dOutN, dTokFp4, dTokSc);
  HIP_CHECK(hipGetLastError());
  hipLaunchKernelGGL(moe_pipelined<K>, dim3(1), dim3(256), 0, 0, dTok, dW, dS,
                     dBias, dOutP);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> gotN(OUT_ROWS), gotP(OUT_ROWS);
  std::vector<uint8_t> tok_fp4(K / 2), tok_sc(NBLK);
  HIP_CHECK(hipMemcpy(
      gotN.data(), dOutN, gotN.size() * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(
      gotP.data(), dOutP, gotP.size() * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(tok_fp4.data(), dTokFp4, K / 2, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(tok_sc.data(), dTokSc, NBLK, hipMemcpyDeviceToHost));

  // Independently re-quantize the token on the host and confirm the device
  // agrees bit for bit. If this differs, the mismatch below is a quantization
  // bug, not a layout bug -- worth telling apart.
  int qbad = 0;
  for (int blk = 0; blk < NBLK; ++blk) {
    float amax = 0.f;
    for (int j = 0; j < 32; ++j) {
      amax = fmaxf(amax, fabsf(tok_f[blk * 32 + j]));
    }
    uint8_t se = 0;
    if (amax != 0.f) {
      union {
        float f;
        uint32_t u;
      } v;
      v.f = amax * (1.0f / 6.0f);
      int re = (int)((v.u >> 23) & 0xFF);
      if (v.u & 0x7FFFFF) {
        re++;
      }
      se = (uint8_t)std::max(0, std::min(255, re));
    }
    if (se != tok_sc[blk]) {
      ++qbad;
    }
  }

  int untouchedN = 0, untouchedP = 0;
  for (int i = 0; i < OUT_ROWS; ++i) {
    if (gotN[i] == -99999.f) {
      ++untouchedN;
    }
    if (gotP[i] == -99999.f) {
      ++untouchedP;
    }
  }

  // Reference: decode the device's own quantized token, dot with the weights.
  int badN = 0, badP = 0, diff = 0;
  double worstN = 0.0, worstP = 0.0;
  for (int r = 0; r < OUT_ROWS; ++r) {
    double ref = 0.0;
    for (int k = 0; k < K; ++k) {
      uint8_t tb = tok_fp4[k / 2];
      uint8_t tcode = (k & 1) ? (tb >> 4) : (tb & 0xF);
      ref += (double)kE2M1[wcode[r * K + k]] *
             e8m0(wg_scales[r * NBLK + k / 32]) * (double)kE2M1[tcode] *
             e8m0(tok_sc[k / 32]);
    }
    union {
      float f;
      unsigned u;
    } b;
    b.u = ((unsigned)bias[r]) << 16;
    ref += b.f;

    double eN = std::fabs(gotN[r] - ref), eP = std::fabs(gotP[r] - ref);
    worstN = fmax(worstN, eN);
    worstP = fmax(worstP, eP);
    double tol = 1e-4 * std::fabs(ref) + 1e-4;
    if (eN > tol) {
      if (badN < 3) {
        printf("  NAIVE MISMATCH out[%3d] got=%.6f ref=%.6f\n", r, gotN[r], ref);
      }
      ++badN;
    }
    if (eP > tol) {
      if (badP < 3) {
        printf("  PIPE  MISMATCH out[%3d] got=%.6f ref=%.6f\n", r, gotP[r], ref);
      }
      ++badP;
    }
    // Exact: the optimized loop reorders memory, not arithmetic. Same operands
    // accumulate in the same order, so the two must agree to the bit.
    if (gotN[r] != gotP[r]) {
      if (diff < 3) {
        printf("  NAIVE != PIPE out[%3d] naive=%.9g pipe=%.9g\n",
               r,
               gotN[r],
               gotP[r]);
      }
      ++diff;
    }
  }

  printf("token quant scale mismatches: %d/%d\n", qbad, NBLK);
  printf("naive: untouched=%d/%d mismatches=%d/%d worst_err=%.6g\n",
         untouchedN, OUT_ROWS, badN, OUT_ROWS, worstN);
  printf("pipe : untouched=%d/%d mismatches=%d/%d worst_err=%.6g\n",
         untouchedP, OUT_ROWS, badP, OUT_ROWS, worstP);
  printf("naive vs pipe bit-exact differences: %d/%d\n", diff, OUT_ROWS);

  HIP_CHECK(hipFree(dTok));
  HIP_CHECK(hipFree(dBias));
  HIP_CHECK(hipFree(dW));
  HIP_CHECK(hipFree(dS));
  HIP_CHECK(hipFree(dTokFp4));
  HIP_CHECK(hipFree(dTokSc));
  HIP_CHECK(hipFree(dOutN));
  HIP_CHECK(hipFree(dOutP));

  if (untouchedN || untouchedP) {
    printf("FAIL: outputs never written\n\n");
    return 1;
  }
  if (qbad) {
    printf("FAIL: device and host disagree on token quantization\n\n");
    return 1;
  }
  if (badN || badP) {
    printf("FAIL: does not match the reference\n\n");
    return 1;
  }
  if (diff) {
    printf("FAIL: optimized loop changed the result\n\n");
    return 1;
  }
  printf("PASS\n\n");
  return 0;
}

int main() {
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
    printf("no HIP device\n");
    return 2;
  }
  printf("device: %s (%s)\n\n", prop.name, prop.gcnArchName);

  int rc = 0;
  rc |= run_shape<512>(); // 4 K-tiles, even
  rc |= run_shape<384>(); // 3 K-tiles, odd -> the ping-pong tail

  if (rc) {
    printf("OVERALL: FAIL\n");
    return 1;
  }
  printf("OVERALL PASS: MI450 MoE MXFP4 naive and pipelined loops both match "
         "the reference, and each other exactly\n");
  return 0;
}
