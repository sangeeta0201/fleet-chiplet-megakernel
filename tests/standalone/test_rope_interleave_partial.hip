/* Copyright 2025 CMU
 *
 * Numerical test for GLM-5's partial + interleaved RoPE
 * (rope_interleave_partial() in tasks/mi300/rotary_embedding_mi300.cuh).
 *
 * Reference is transformers' apply_rotary_pos_emb_interleave() from
 * models/glm_moe_dsa/modeling_glm_moe_dsa.py:
 *
 *   cos, sin = cos[..., :d/2], sin[..., :d/2]     # emb == cat(freqs, freqs)
 *   q1, q2   = q[..., 0::2], q[..., 1::2]
 *   q_embed  = cat([q1*cos - q2*sin, q2*cos + q1*sin], dim=-1)
 *
 * applied only to the trailing qk_rope_head_dim dims of each head, with the
 * leading qk_nope_head_dim dims passed through unchanged.
 *
 * Both geometries GLM-5 uses are covered:
 *   Q:     64 heads x (192 nope + 64 rope) = qk_head_dim 256
 *   K_rot:  1 head  x (512 nope + 64 rope) = kv_lora_rank 512 + 64
 *
 * Build: ./build.sh
 * Run:   ./test_rope_interleave_partial
 *
 * Exit code 0 = kernel matches the reference on both geometries.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <vector>

#include "mirage/persistent_kernel/tasks/mi300/rotary_embedding_mi300.cuh"

using bf16 = __hip_bfloat16;

static constexpr int BLOCK_THREADS = 256;
static constexpr int ROPE_DIM = 64;

// GLM-5 query: 64 heads, qk_nope_head_dim 192 + qk_rope_head_dim 64.
static constexpr int Q_HEAD_DIM = 256;
static constexpr int Q_NUM_HEADS = 64;
// GLM-5 latent key: one shared head, kv_lora_rank 512 + qk_rope_head_dim 64.
static constexpr int K_HEAD_DIM = 576;
static constexpr int K_NUM_HEADS = 1;

template <int HEAD_DIM, int NUM_HEADS>
__global__ void rope_entry(void *heads, void const *cos, void const *sin) {
  kernel::rope_interleave_partial<bf16,
                                  HEAD_DIM,
                                  ROPE_DIM,
                                  NUM_HEADS,
                                  BLOCK_THREADS>(
      reinterpret_cast<bf16 *>(heads),
      reinterpret_cast<bf16 const *>(cos),
      reinterpret_cast<bf16 const *>(sin));
}

static float to_bf16_and_back(float x) {
  uint32_t u;
  __builtin_memcpy(&u, &x, 4);
  uint32_t lsb = (u >> 16) & 1u;
  u = (u + 0x7fffu + lsb) & 0xffff0000u;
  float r;
  __builtin_memcpy(&r, &u, 4);
  return r;
}

static uint16_t to_bf16_bits(float x) {
  float r = to_bf16_and_back(x);
  uint32_t u;
  __builtin_memcpy(&u, &r, 4);
  return (uint16_t)(u >> 16);
}

static float from_bf16_bits(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float r;
  __builtin_memcpy(&r, &u, 4);
  return r;
}

template <int HEAD_DIM, int NUM_HEADS>
static int run_case(char const *name, int pos) {
  constexpr int HALF = ROPE_DIM / 2;
  constexpr int NOPE_DIM = HEAD_DIM - ROPE_DIM;
  constexpr int N = NUM_HEADS * HEAD_DIM;

  std::vector<float> host(N);
  std::vector<uint16_t> h_in(N);
  for (int i = 0; i < N; ++i) {
    host[i] = to_bf16_and_back(((float)rand() / (float)RAND_MAX - 0.5f) * 4.0f);
    h_in[i] = to_bf16_bits(host[i]);
  }

  // rope_parameters.rope_theta = 1000000, rope_type "default".
  // inv_freq[j] = 1 / theta^(2j / ROPE_DIM), j in [0, ROPE_DIM/2).
  std::vector<uint16_t> h_cos(ROPE_DIM), h_sin(ROPE_DIM);
  std::vector<float> cos_f(HALF), sin_f(HALF);
  for (int j = 0; j < HALF; ++j) {
    double inv_freq = 1.0 / pow(1000000.0, (double)(2 * j) / (double)ROPE_DIM);
    double angle = (double)pos * inv_freq;
    cos_f[j] = to_bf16_and_back((float)cos(angle));
    sin_f[j] = to_bf16_and_back((float)sin(angle));
    // HF stores emb = cat(freqs, freqs); only the first half is ever read.
    h_cos[j] = to_bf16_bits(cos_f[j]);
    h_sin[j] = to_bf16_bits(sin_f[j]);
    h_cos[j + HALF] = h_cos[j];
    h_sin[j + HALF] = h_sin[j];
  }

  // Host reference.
  std::vector<float> ref(N);
  for (int h = 0; h < NUM_HEADS; ++h) {
    for (int d = 0; d < NOPE_DIM; ++d) {
      ref[h * HEAD_DIM + d] = host[h * HEAD_DIM + d];
    }
    float const *rot_in = &host[h * HEAD_DIM + NOPE_DIM];
    float *rot_out = &ref[h * HEAD_DIM + NOPE_DIM];
    for (int j = 0; j < HALF; ++j) {
      float x0 = rot_in[2 * j];
      float x1 = rot_in[2 * j + 1];
      rot_out[j] = to_bf16_and_back(x0 * cos_f[j] - x1 * sin_f[j]);
      rot_out[j + HALF] = to_bf16_and_back(x1 * cos_f[j] + x0 * sin_f[j]);
    }
  }

  void *d_heads, *d_cos, *d_sin;
  hipMalloc(&d_heads, (size_t)N * 2);
  hipMalloc(&d_cos, (size_t)ROPE_DIM * 2);
  hipMalloc(&d_sin, (size_t)ROPE_DIM * 2);
  hipMemcpy(d_heads, h_in.data(), (size_t)N * 2, hipMemcpyHostToDevice);
  hipMemcpy(d_cos, h_cos.data(), (size_t)ROPE_DIM * 2, hipMemcpyHostToDevice);
  hipMemcpy(d_sin, h_sin.data(), (size_t)ROPE_DIM * 2, hipMemcpyHostToDevice);

  hipLaunchKernelGGL((rope_entry<HEAD_DIM, NUM_HEADS>),
                     dim3(1),
                     dim3(BLOCK_THREADS),
                     0,
                     0,
                     d_heads,
                     d_cos,
                     d_sin);
  hipError_t err = hipDeviceSynchronize();
  if (err != hipSuccess) {
    fprintf(stderr, "%s: launch failed: %s\n", name, hipGetErrorString(err));
    return 1;
  }

  std::vector<uint16_t> h_out(N);
  hipMemcpy(h_out.data(), d_heads, (size_t)N * 2, hipMemcpyDeviceToHost);
  hipFree(d_heads);
  hipFree(d_cos);
  hipFree(d_sin);

  int failures = 0;
  float max_abs = 0.f;
  for (int i = 0; i < N; ++i) {
    float got = from_bf16_bits(h_out[i]);
    float want = ref[i];
    float d = fabsf(got - want);
    max_abs = fmaxf(max_abs, d);
    // Same bf16 rounding on both sides, so mismatches should be exactly 0;
    // one bf16 ulp of slack covers fma contraction in the kernel.
    if (d > 0.02f) {
      if (failures < 10) {
        printf("%s: head %d dim %d: %.6f != %.6f\n",
               name,
               i / HEAD_DIM,
               i % HEAD_DIM,
               got,
               want);
      }
      ++failures;
    }
  }
  if (failures == 0) {
    printf("PASS %s: %d heads x (%d nope + %d rope), max abs err %.3e\n",
           name,
           NUM_HEADS,
           NOPE_DIM,
           ROPE_DIM,
           max_abs);
    return 0;
  }
  printf("FAIL %s: %d mismatch(es)\n", name, failures);
  return 1;
}

int main() {
  srand(4321);
  int rc = 0;
  rc |= run_case<Q_HEAD_DIM, Q_NUM_HEADS>("q_rot", 1337);
  rc |= run_case<K_HEAD_DIM, K_NUM_HEADS>("k_rot", 1337);
  return rc;
}
