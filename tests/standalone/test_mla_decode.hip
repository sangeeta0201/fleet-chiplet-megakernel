/* Copyright 2025 CMU
 *
 * Numerical test for the GLM-5 absorbed MLA decode kernel
 * (mla_decode_absorbed / gang_mla_decode_kernel in
 * tasks/mi300/gang_mla_decode_mi300.cuh).
 *
 * Geometry is GLM-5's: 64 query heads, kv_lora_rank 512 + qk_rope_head_dim 64
 * = 576 reduction dims, one shared latent KV head, paged cache.
 *
 * Reference is a plain fp32 MQA-with-asymmetric-dims attention on the host:
 *
 *   s[t]   = scale * <q[h][0:576], kv[t][0:576]>
 *   p      = softmax(s)
 *   out[h] = sum_t p[t] * kv[t][0:512]
 *
 * Both the NUM_KV_CHUNKS==1 path (direct bf16 output) and the split-KV path
 * (float o_acc + natural-log LSE) are covered; the split-KV case is merged on
 * the host with the same formula merge_splitkv_ck_fmha uses.
 *
 * Build: ./build.sh
 * Run:   ./test_mla_decode
 *
 * Exit code 0 = kernel matches the reference at every tested sequence length.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <vector>

#include "mirage/persistent_kernel/tasks/mi300/gang_mla_decode_mi300.cuh"

using bf16 = __hip_bfloat16;

static constexpr int NUM_Q_HEADS = 64;
static constexpr int KV_LORA_RANK = 512;
static constexpr int QK_ROPE_HEAD_DIM = 64;
static constexpr int QK_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM;
static constexpr int PAGE_SIZE = 64;
static constexpr int MAX_SEQ_LEN = 4096;
static constexpr int Q_WORKSPACE_STRIDE = NUM_Q_HEADS * QK_DIM;
static constexpr int KV_CACHE_STRIDE = QK_DIM;
static constexpr int NUM_Q_GROUPS = NUM_Q_HEADS / 16;
static constexpr int BLOCK_THREADS = 256;
static constexpr int LDS_BYTES =
    16 * QK_DIM * 2 + 256 * 32 * 4;

static constexpr float LOG2E = 1.44269504088896340736f;

template <int NUM_KV_CHUNKS, bool WRITE_THROUGH>
__global__ void mla_entry(void const *q_workspace,
                          void const *kv_cache,
                          void *output,
                          void *lse,
                          int const *qo_indptr,
                          int const *kv_indptr,
                          int const *kv_indices,
                          int const *kv_last_page_len,
                          int total_work_items,
                          float scale_s) {
  kernel::gang_mla_decode_kernel<bf16,
                                 NUM_Q_HEADS,
                                 KV_LORA_RANK,
                                 QK_ROPE_HEAD_DIM,
                                 PAGE_SIZE,
                                 MAX_SEQ_LEN,
                                 NUM_KV_CHUNKS,
                                 Q_WORKSPACE_STRIDE,
                                 KV_CACHE_STRIDE,
                                 WRITE_THROUGH>(q_workspace,
                                                  kv_cache,
                                                  output,
                                                  lse,
                                                  qo_indptr,
                                                  kv_indptr,
                                                  kv_indices,
                                                  kv_last_page_len,
                                                  total_work_items,
                                                  blockIdx.x,
                                                  scale_s);
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

static float frand() {
  return ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f;
}

template <int NUM_KV_CHUNKS, bool WRITE_THROUGH>
static int run_case(int seqlen) {
  int const num_pages = (seqlen + PAGE_SIZE - 1) / PAGE_SIZE;
  int const last_page_len = seqlen - (num_pages - 1) * PAGE_SIZE;
  // Physical pages are handed out in reverse so the kv_indices indirection is
  // actually exercised rather than degenerating to the identity.
  std::vector<int> h_kv_indices(num_pages);
  for (int p = 0; p < num_pages; ++p) {
    h_kv_indices[p] = num_pages - 1 - p;
  }
  int const num_phys_pages = num_pages;

  // ---- host-side inputs (rounded through bf16 so both sides agree) ----
  std::vector<float> q_f((size_t)NUM_Q_HEADS * QK_DIM);
  std::vector<uint16_t> q_b(q_f.size());
  for (size_t i = 0; i < q_f.size(); ++i) {
    q_f[i] = to_bf16_and_back(frand());
    q_b[i] = to_bf16_bits(q_f[i]);
  }

  // Logical [seqlen][QK_DIM], stored into the paged cache below.
  std::vector<float> kv_f((size_t)seqlen * QK_DIM);
  for (size_t i = 0; i < kv_f.size(); ++i) {
    kv_f[i] = to_bf16_and_back(frand());
  }

  size_t const cache_elems =
      (size_t)num_phys_pages * PAGE_SIZE * KV_CACHE_STRIDE;
  std::vector<uint16_t> kv_b(cache_elems, 0);
  for (int t = 0; t < seqlen; ++t) {
    int phys = h_kv_indices[t / PAGE_SIZE];
    size_t row =
        ((size_t)phys * PAGE_SIZE + (t % PAGE_SIZE)) * KV_CACHE_STRIDE;
    for (int d = 0; d < QK_DIM; ++d) {
      kv_b[row + d] = to_bf16_bits(kv_f[(size_t)t * QK_DIM + d]);
    }
  }

  float const softmax_scale = 1.0f / sqrtf((float)QK_DIM);

  // ---- reference ----
  std::vector<float> ref((size_t)NUM_Q_HEADS * KV_LORA_RANK, 0.f);
  {
    std::vector<float> s(seqlen);
    for (int h = 0; h < NUM_Q_HEADS; ++h) {
      float const *qh = &q_f[(size_t)h * QK_DIM];
      float m = -INFINITY;
      for (int t = 0; t < seqlen; ++t) {
        float dot = 0.f;
        for (int d = 0; d < QK_DIM; ++d) {
          dot += qh[d] * kv_f[(size_t)t * QK_DIM + d];
        }
        s[t] = dot * softmax_scale;
        m = fmaxf(m, s[t]);
      }
      float denom = 0.f;
      for (int t = 0; t < seqlen; ++t) {
        s[t] = expf(s[t] - m);
        denom += s[t];
      }
      float *out = &ref[(size_t)h * KV_LORA_RANK];
      for (int t = 0; t < seqlen; ++t) {
        float p = s[t] / denom;
        float const *kvt = &kv_f[(size_t)t * QK_DIM];
        for (int d = 0; d < KV_LORA_RANK; ++d) {
          out[d] += p * kvt[d];
        }
      }
    }
  }

  // ---- device ----
  int const h_qo_indptr[2] = {0, 1};
  int const h_kv_indptr[2] = {0, num_pages};
  int const h_last_page_len[1] = {last_page_len};

  constexpr int LSE_S = NUM_Q_GROUPS * NUM_KV_CHUNKS * 16;
  size_t const out_elems = (NUM_KV_CHUNKS == 1)
                               ? (size_t)NUM_Q_HEADS * KV_LORA_RANK
                               : (size_t)LSE_S * KV_LORA_RANK;
  size_t const out_bytes = out_elems * (NUM_KV_CHUNKS == 1 ? 2 : 4);

  void *d_q, *d_kv, *d_out, *d_lse;
  int *d_qo_indptr, *d_kv_indptr, *d_kv_indices, *d_last_page_len;
  hipMalloc(&d_q, q_b.size() * 2);
  hipMalloc(&d_kv, kv_b.size() * 2);
  hipMalloc(&d_out, out_bytes);
  hipMalloc(&d_lse, (size_t)LSE_S * 4);
  hipMalloc(&d_qo_indptr, 2 * 4);
  hipMalloc(&d_kv_indptr, 2 * 4);
  hipMalloc(&d_kv_indices, (size_t)num_pages * 4);
  hipMalloc(&d_last_page_len, 4);
  hipMemcpy(d_q, q_b.data(), q_b.size() * 2, hipMemcpyHostToDevice);
  hipMemcpy(d_kv, kv_b.data(), kv_b.size() * 2, hipMemcpyHostToDevice);
  hipMemset(d_out, 0, out_bytes);
  hipMemset(d_lse, 0, (size_t)LSE_S * 4);
  hipMemcpy(d_qo_indptr, h_qo_indptr, 8, hipMemcpyHostToDevice);
  hipMemcpy(d_kv_indptr, h_kv_indptr, 8, hipMemcpyHostToDevice);
  hipMemcpy(d_kv_indices,
            h_kv_indices.data(),
            (size_t)num_pages * 4,
            hipMemcpyHostToDevice);
  hipMemcpy(d_last_page_len, h_last_page_len, 4, hipMemcpyHostToDevice);

  int const total_work_items = NUM_Q_GROUPS * NUM_KV_CHUNKS;
  hipLaunchKernelGGL((mla_entry<NUM_KV_CHUNKS, WRITE_THROUGH>),
                     dim3(total_work_items),
                     dim3(BLOCK_THREADS),
                     LDS_BYTES,
                     0,
                     d_q,
                     d_kv,
                     d_out,
                     d_lse,
                     d_qo_indptr,
                     d_kv_indptr,
                     d_kv_indices,
                     d_last_page_len,
                     total_work_items,
                     softmax_scale * LOG2E);
  hipError_t err = hipDeviceSynchronize();
  if (err != hipSuccess) {
    fprintf(stderr,
            "chunks=%d seqlen=%d: launch failed: %s\n",
            NUM_KV_CHUNKS,
            seqlen,
            hipGetErrorString(err));
    return 1;
  }

  // ---- pull back and (for split-KV) merge ----
  std::vector<float> got((size_t)NUM_Q_HEADS * KV_LORA_RANK);
  if constexpr (NUM_KV_CHUNKS == 1) {
    std::vector<uint16_t> raw(out_elems);
    hipMemcpy(raw.data(), d_out, out_bytes, hipMemcpyDeviceToHost);
    for (size_t i = 0; i < got.size(); ++i) {
      got[i] = from_bf16_bits(raw[i]);
    }
  } else {
    std::vector<float> o_acc(out_elems);
    std::vector<float> lse(LSE_S);
    hipMemcpy(o_acc.data(), d_out, out_bytes, hipMemcpyDeviceToHost);
    hipMemcpy(lse.data(), d_lse, (size_t)LSE_S * 4, hipMemcpyDeviceToHost);
    // Same merge merge_splitkv_ck_fmha performs.
    for (int g = 0; g < NUM_Q_GROUPS; ++g) {
      for (int m = 0; m < 16; ++m) {
        int const h = g * 16 + m;
        float m_global = -INFINITY;
        for (int c = 0; c < NUM_KV_CHUNKS; ++c) {
          m_global = fmaxf(m_global, lse[g * NUM_KV_CHUNKS * 16 + c * 16 + m]);
        }
        float denom = 0.f;
        for (int c = 0; c < NUM_KV_CHUNKS; ++c) {
          denom += expf(lse[g * NUM_KV_CHUNKS * 16 + c * 16 + m] - m_global);
        }
        for (int d = 0; d < KV_LORA_RANK; ++d) {
          float acc = 0.f;
          for (int c = 0; c < NUM_KV_CHUNKS; ++c) {
            int idx = g * NUM_KV_CHUNKS * 16 + c * 16 + m;
            acc += o_acc[(size_t)idx * KV_LORA_RANK + d] *
                   expf(lse[idx] - m_global);
          }
          got[(size_t)h * KV_LORA_RANK + d] = acc / denom;
        }
      }
    }
  }

  hipFree(d_q);
  hipFree(d_kv);
  hipFree(d_out);
  hipFree(d_lse);
  hipFree(d_qo_indptr);
  hipFree(d_kv_indptr);
  hipFree(d_kv_indices);
  hipFree(d_last_page_len);

  // ---- compare ----
  // P is rounded to fp16 before the PV MFMA and the direct path writes bf16,
  // so the error floor is ~1e-3 relative to the head's dynamic range.
  float max_ref = 0.f;
  for (float v : ref) {
    max_ref = fmaxf(max_ref, fabsf(v));
  }
  float max_abs = 0.f;
  int failures = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    float d = fabsf(got[i] - ref[i]);
    max_abs = fmaxf(max_abs, d);
    if (d > 0.02f * max_ref) {
      if (failures < 8) {
        printf("  head %zu dim %zu: %.6f != %.6f\n",
               i / KV_LORA_RANK,
               i % KV_LORA_RANK,
               got[i],
               ref[i]);
      }
      ++failures;
    }
  }
  if (failures == 0) {
    printf("PASS chunks=%d seqlen=%-5d  max abs err %.3e (|ref|max %.3e, "
           "rel %.2e)\n",
           NUM_KV_CHUNKS,
           seqlen,
           max_abs,
           max_ref,
           max_abs / max_ref);
    return 0;
  }
  printf("FAIL chunks=%d seqlen=%d: %d mismatch(es), max abs err %.3e\n",
         NUM_KV_CHUNKS,
         seqlen,
         failures,
         max_abs);
  return 1;
}

template <bool WRITE_THROUGH>
static int run_suite() {
  int rc = 0;
  // 1 exercises the single-token tile, 16/64 the exact-tile cases, the rest
  // partial tails and multi-page indirection.
  for (int seqlen : {1, 7, 16, 17, 64, 65, 129, 333}) {
    rc |= run_case<1, WRITE_THROUGH>(seqlen);
  }
  for (int seqlen : {1, 17, 64, 129, 333}) {
    rc |= run_case<2, WRITE_THROUGH>(seqlen);
  }
  for (int seqlen : {333, 1024}) {
    rc |= run_case<4, WRITE_THROUGH>(seqlen);
  }
  // Production fused shape: 16 chunks, kv_len just past 16*16 so each
  // occupied chunk's tile loop trips twice.
  rc |= run_case<16, WRITE_THROUGH>(257);
  // The fused break point. Fused decode is correct up to seqlen 225 and wrong
  // from 241 on, at 16 AND 32 chunks -- so the trigger is global
  // ntiles = ceil(seqlen/16) >= 16, not tiles_per_chunk (241 and 225 both
  // give tiles_per_chunk 1). These two straddle it; if the kernel is clean
  // here in isolation the fault is not in this file's arithmetic.
  rc |= run_case<16, WRITE_THROUGH>(225);
  rc |= run_case<16, WRITE_THROUGH>(241);
  rc |= run_case<32, WRITE_THROUGH>(257);
  return rc;
}

int main() {
  srand(20250811);
  int rc = 0;
  printf("=== WRITE_THROUGH=false ===\n");
  rc |= run_suite<false>();
  srand(20250811);
  printf("=== WRITE_THROUGH=true ===\n");
  rc |= run_suite<true>();
  return rc;
}
