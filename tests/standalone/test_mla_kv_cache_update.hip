/* Copyright 2025 CMU
 *
 * Numerical test for the GLM-5 absorbed-MLA latent cache append
 * (mla_kv_cache_update_impl in tasks/mi300/mla_kv_cache_update_mi300.cuh).
 *
 * The kernel is Phase A of the MLA attention path: it applies kv_a_layernorm
 * to the latent, partial+interleaved RoPE to the trailing 64 dims of both the
 * latent and every absorbed query head, then writes one 576-wide row per token
 * into the paged cache and the roped queries into the q workspace.
 *
 * Reference is a host fp32 recomputation of exactly that, with the RoPE
 * following transformers' apply_rotary_pos_emb_interleave:
 *
 *   x0, x1  = x[2j], x[2j+1]
 *   out[j]            = x0*cos[j] - x1*sin[j]
 *   out[j + rope/2]   = x1*cos[j] + x0*sin[j]
 *
 * and the norm following GLM's RMSNorm:
 *
 *   c[i] * rsqrt(mean(c^2) + eps) * weight[i]
 *
 * Two geometries are covered:
 *   GLM-5:            64 q heads, kv_lora 512 + rope 64
 *   GLM-4.7-Flash:    32 q heads (20 real, padded to 2 MFMA M tiles), same
 *                     latent dims
 * and two decode shapes: a single appended token (steady-state decode) and a
 * multi-token append that crosses a page boundary (chunked prefill).
 *
 * Build: ./build.sh
 * Run:   ./test_mla_kv_cache_update
 *
 * Exit code 0 = kernel matches the reference on every case.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <vector>

#include "mirage/persistent_kernel/tasks/mi300/mla_kv_cache_update_mi300.cuh"

using bf16 = __hip_bfloat16;

static constexpr int KV_LORA_RANK = 512;
static constexpr int QK_ROPE_HEAD_DIM = 64;
static constexpr int QK_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM;
static constexpr int ROPE_HALF = QK_ROPE_HEAD_DIM / 2;
static constexpr int PAGE_SIZE = 64;
static constexpr int MAX_SEQ_LEN = 4096;
static constexpr int MAX_PAGES = MAX_SEQ_LEN / PAGE_SIZE;
static constexpr int BLOCK_THREADS = 256;
static constexpr float KV_EPS = 1e-6f;

template <int NUM_QO_HEADS>
__global__ void mla_kvupd_entry(void const *q_absorbed,
                                void const *kv_latent,
                                void *kv_cache,
                                void *q_workspace,
                                int const *qo_indptr,
                                int const *kv_indptr,
                                int const *kv_indices,
                                int const *kv_last_page_len,
                                void const *kv_norm,
                                void const *cos,
                                void const *sin) {
  kernel::mla_kv_cache_update_impl<bf16,
                                   NUM_QO_HEADS,
                                   KV_LORA_RANK,
                                   QK_ROPE_HEAD_DIM,
                                   /*Q_INPUT_STRIDE=*/NUM_QO_HEADS * QK_DIM,
                                   /*KV_INPUT_STRIDE=*/QK_DIM,
                                   /*KV_CACHE_STRIDE=*/QK_DIM,
                                   MAX_SEQ_LEN,
                                   PAGE_SIZE,
                                   /*Q_WORKSPACE_STRIDE=*/NUM_QO_HEADS *
                                       QK_DIM>(q_absorbed,
                                               kv_latent,
                                               kv_cache,
                                               q_workspace,
                                               qo_indptr,
                                               kv_indptr,
                                               kv_indices,
                                               kv_last_page_len,
                                               /*request_id=*/(int16_t)0,
                                               kv_norm,
                                               cos,
                                               sin,
                                               KV_EPS);
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

static bf16 make_bf16(float x) {
  return static_cast<bf16>(x);
}

static float from_bf16(bf16 x) {
  return static_cast<float>(x);
}

// Deterministic pseudo-random in [-1, 1), rounded through bf16 so the host
// reference starts from the same values the device sees.
static float rnd(uint32_t &state) {
  state = state * 1664525u + 1013904223u;
  float v = float((state >> 8) & 0xffff) / 32768.0f - 1.0f;
  return to_bf16_and_back(v);
}

#define HIP_CHECK(cmd)                                                         \
  do {                                                                         \
    hipError_t e = (cmd);                                                      \
    if (e != hipSuccess) {                                                     \
      printf("FAIL: HIP error %s at line %d\n", hipGetErrorString(e),          \
             __LINE__);                                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

template <int NUM_QO_HEADS>
static bool run_case(char const *label, int seq_len, int num_tokens) {
  int const num_pages = (seq_len + PAGE_SIZE - 1) / PAGE_SIZE;
  int const last_page_len = seq_len - (num_pages - 1) * PAGE_SIZE;
  int const q_stride = NUM_QO_HEADS * QK_DIM;

  uint32_t state = 0x1234567u + (uint32_t)seq_len * 7919u + NUM_QO_HEADS;

  std::vector<float> h_q(num_tokens * q_stride);
  for (auto &v : h_q) {
    v = rnd(state);
  }
  std::vector<float> h_kv(num_tokens * QK_DIM);
  for (auto &v : h_kv) {
    v = rnd(state);
  }
  std::vector<float> h_norm(KV_LORA_RANK);
  for (auto &v : h_norm) {
    v = 1.0f + 0.25f * rnd(state);
    v = to_bf16_and_back(v);
  }
  // cos/sin laid out [MAX_SEQ_LEN, QK_ROPE_HEAD_DIM]; only the first
  // QK_ROPE_HEAD_DIM/2 entries per position are read, matching how the demo
  // slices the HF cat(freqs, freqs) table.
  std::vector<float> h_cos(MAX_SEQ_LEN * QK_ROPE_HEAD_DIM);
  std::vector<float> h_sin(MAX_SEQ_LEN * QK_ROPE_HEAD_DIM);
  for (int p = 0; p < MAX_SEQ_LEN; p++) {
    for (int j = 0; j < QK_ROPE_HEAD_DIM; j++) {
      float inv = powf(1e6f, -float(j % ROPE_HALF) * 2.0f / QK_ROPE_HEAD_DIM);
      h_cos[p * QK_ROPE_HEAD_DIM + j] = to_bf16_and_back(cosf(p * inv));
      h_sin[p * QK_ROPE_HEAD_DIM + j] = to_bf16_and_back(sinf(p * inv));
    }
  }

  // Pages are handed out in reverse so the test also exercises a
  // non-contiguous page table.
  std::vector<int> h_indices(num_pages);
  for (int i = 0; i < num_pages; i++) {
    h_indices[i] = num_pages - 1 - i;
  }
  std::vector<int> h_qo_indptr = {0, num_tokens};
  std::vector<int> h_kv_indptr = {0, num_pages};
  std::vector<int> h_last_page = {last_page_len};

  auto to_dev_bf16 = [](std::vector<float> const &src) {
    std::vector<bf16> tmp(src.size());
    for (size_t i = 0; i < src.size(); i++) {
      tmp[i] = make_bf16(src[i]);
    }
    void *d = nullptr;
    HIP_CHECK(hipMalloc(&d, tmp.size() * sizeof(bf16)));
    HIP_CHECK(hipMemcpy(d, tmp.data(), tmp.size() * sizeof(bf16),
                        hipMemcpyHostToDevice));
    return d;
  };
  auto to_dev_int = [](std::vector<int> const &src) {
    void *d = nullptr;
    HIP_CHECK(hipMalloc(&d, src.size() * sizeof(int)));
    HIP_CHECK(hipMemcpy(d, src.data(), src.size() * sizeof(int),
                        hipMemcpyHostToDevice));
    return d;
  };

  void *d_q = to_dev_bf16(h_q);
  void *d_kv = to_dev_bf16(h_kv);
  void *d_norm = to_dev_bf16(h_norm);
  void *d_cos = to_dev_bf16(h_cos);
  void *d_sin = to_dev_bf16(h_sin);
  void *d_qo_indptr = to_dev_int(h_qo_indptr);
  void *d_kv_indptr = to_dev_int(h_kv_indptr);
  void *d_kv_indices = to_dev_int(h_indices);
  void *d_last_page = to_dev_int(h_last_page);

  size_t cache_elems = (size_t)MAX_PAGES * PAGE_SIZE * QK_DIM;
  void *d_cache = nullptr;
  HIP_CHECK(hipMalloc(&d_cache, cache_elems * sizeof(bf16)));
  HIP_CHECK(hipMemset(d_cache, 0, cache_elems * sizeof(bf16)));
  void *d_ws = nullptr;
  HIP_CHECK(hipMalloc(&d_ws, (size_t)num_tokens * q_stride * sizeof(bf16)));
  HIP_CHECK(hipMemset(d_ws, 0, (size_t)num_tokens * q_stride * sizeof(bf16)));

  hipLaunchKernelGGL((mla_kvupd_entry<NUM_QO_HEADS>), dim3(1),
                     dim3(BLOCK_THREADS), 0, 0, d_q, d_kv, d_cache, d_ws,
                     (int const *)d_qo_indptr, (int const *)d_kv_indptr,
                     (int const *)d_kv_indices, (int const *)d_last_page,
                     d_norm, d_cos, d_sin);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<bf16> got_cache(cache_elems);
  std::vector<bf16> got_ws((size_t)num_tokens * q_stride);
  HIP_CHECK(hipMemcpy(got_cache.data(), d_cache, cache_elems * sizeof(bf16),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(got_ws.data(), d_ws,
                      (size_t)num_tokens * q_stride * sizeof(bf16),
                      hipMemcpyDeviceToHost));

  // ---------------------------------------------------------------- reference
  float max_ws_err = 0.0f, max_cache_err = 0.0f;
  int bad = 0;
  for (int t = 0; t < num_tokens && bad < 8; t++) {
    int const pos = seq_len - num_tokens + t;
    float const *cos_data = &h_cos[(size_t)pos * QK_ROPE_HEAD_DIM];
    float const *sin_data = &h_sin[(size_t)pos * QK_ROPE_HEAD_DIM];

    // Q: pass-through on [0, 512), interleaved RoPE on [512, 576).
    for (int h = 0; h < NUM_QO_HEADS && bad < 8; h++) {
      float const *qs = &h_q[(size_t)t * q_stride + h * QK_DIM];
      bf16 const *qg = &got_ws[(size_t)t * q_stride + h * QK_DIM];
      for (int i = 0; i < KV_LORA_RANK; i++) {
        float err = fabsf(from_bf16(qg[i]) - qs[i]);
        max_ws_err = fmaxf(max_ws_err, err);
        if (err != 0.0f && bad < 8) {
          printf("  q pass-through mismatch t=%d h=%d i=%d: got %g want %g\n",
                 t, h, i, from_bf16(qg[i]), qs[i]);
          bad++;
        }
      }
      for (int j = 0; j < ROPE_HALF; j++) {
        float x0 = qs[KV_LORA_RANK + 2 * j];
        float x1 = qs[KV_LORA_RANK + 2 * j + 1];
        float want_lo = x0 * cos_data[j] - x1 * sin_data[j];
        float want_hi = x1 * cos_data[j] + x0 * sin_data[j];
        float got_lo = from_bf16(qg[KV_LORA_RANK + j]);
        float got_hi = from_bf16(qg[KV_LORA_RANK + ROPE_HALF + j]);
        max_ws_err = fmaxf(max_ws_err, fabsf(got_lo - want_lo));
        max_ws_err = fmaxf(max_ws_err, fabsf(got_hi - want_hi));
        if ((fabsf(got_lo - want_lo) > 8e-3f ||
             fabsf(got_hi - want_hi) > 8e-3f) &&
            bad < 8) {
          printf("  q rope mismatch t=%d h=%d j=%d: got (%g,%g) want (%g,%g)\n",
                 t, h, j, got_lo, got_hi, want_lo, want_hi);
          bad++;
        }
      }
    }

    // Latent: RMSNorm on [0, 512), interleaved RoPE on [512, 576).
    float const *ks = &h_kv[(size_t)t * QK_DIM];
    double sum_sq = 0.0;
    for (int i = 0; i < KV_LORA_RANK; i++) {
      sum_sq += (double)ks[i] * (double)ks[i];
    }
    float rms_rcp = 1.0f / sqrtf((float)(sum_sq / KV_LORA_RANK) + KV_EPS);

    int const page = h_indices[pos / PAGE_SIZE];
    bf16 const *row =
        &got_cache[((size_t)page * PAGE_SIZE + (pos % PAGE_SIZE)) * QK_DIM];
    for (int i = 0; i < KV_LORA_RANK; i++) {
      float want = ks[i] * rms_rcp * h_norm[i];
      float err = fabsf(from_bf16(row[i]) - want);
      max_cache_err = fmaxf(max_cache_err, err);
      if (err > 8e-3f && bad < 8) {
        printf("  kv norm mismatch t=%d i=%d: got %g want %g\n", t, i,
               from_bf16(row[i]), want);
        bad++;
      }
    }
    for (int j = 0; j < ROPE_HALF; j++) {
      float x0 = ks[KV_LORA_RANK + 2 * j];
      float x1 = ks[KV_LORA_RANK + 2 * j + 1];
      float want_lo = x0 * cos_data[j] - x1 * sin_data[j];
      float want_hi = x1 * cos_data[j] + x0 * sin_data[j];
      float got_lo = from_bf16(row[KV_LORA_RANK + j]);
      float got_hi = from_bf16(row[KV_LORA_RANK + ROPE_HALF + j]);
      max_cache_err = fmaxf(max_cache_err, fabsf(got_lo - want_lo));
      max_cache_err = fmaxf(max_cache_err, fabsf(got_hi - want_hi));
      if ((fabsf(got_lo - want_lo) > 8e-3f ||
           fabsf(got_hi - want_hi) > 8e-3f) &&
          bad < 8) {
        printf("  kv rope mismatch t=%d j=%d: got (%g,%g) want (%g,%g)\n", t, j,
               got_lo, got_hi, want_lo, want_hi);
        bad++;
      }
    }
  }

  // Rows the append should not have touched must still be zero. This is what
  // catches an off-by-one in the global_seq_len - num_tokens + token mapping,
  // which would otherwise silently corrupt history.
  int untouched_nonzero = 0;
  for (int p = 0; p < num_pages; p++) {
    for (int s = 0; s < PAGE_SIZE; s++) {
      int logical = p * PAGE_SIZE + s;
      if (logical >= seq_len - num_tokens && logical < seq_len) {
        continue;
      }
      int page = h_indices[p];
      bf16 const *row =
          &got_cache[((size_t)page * PAGE_SIZE + s) * QK_DIM];
      for (int i = 0; i < QK_DIM; i++) {
        if (from_bf16(row[i]) != 0.0f) {
          untouched_nonzero++;
          break;
        }
      }
    }
  }
  if (untouched_nonzero) {
    printf("  %d cache row(s) written outside the appended range\n",
           untouched_nonzero);
    bad++;
  }

  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_kv));
  HIP_CHECK(hipFree(d_norm));
  HIP_CHECK(hipFree(d_cos));
  HIP_CHECK(hipFree(d_sin));
  HIP_CHECK(hipFree(d_qo_indptr));
  HIP_CHECK(hipFree(d_kv_indptr));
  HIP_CHECK(hipFree(d_kv_indices));
  HIP_CHECK(hipFree(d_last_page));
  HIP_CHECK(hipFree(d_cache));
  HIP_CHECK(hipFree(d_ws));

  bool ok = (bad == 0);
  printf("%s %s: heads=%d seq_len=%d tokens=%d  max err q %.3e, cache %.3e\n",
         ok ? "PASS" : "FAIL", label, NUM_QO_HEADS, seq_len, num_tokens,
         max_ws_err, max_cache_err);
  return ok;
}

int main() {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  printf("Device: %s (%s)\n\n", prop.name, prop.gcnArchName);

  bool ok = true;
  // GLM-5: 64 heads. Steady-state decode appends one token.
  ok &= run_case<64>("glm5 decode", 1, 1);
  ok &= run_case<64>("glm5 decode", 129, 1);
  // Multi-token append crossing a page boundary (chunked prefill).
  ok &= run_case<64>("glm5 append", 70, 8);
  ok &= run_case<64>("glm5 append", 256, 4);
  // GLM-4.7-Flash: 20 real heads padded to 32 (2 MFMA M tiles).
  ok &= run_case<32>("flash decode", 200, 1);
  ok &= run_case<32>("flash append", 130, 6);

  printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
  return ok ? 0 : 1;
}
