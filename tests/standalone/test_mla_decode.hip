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
// MPK_MLA_DECODE_DMA_PF adds a second KV tile buffer past the o_acc spill.
// MPK_MLA_DECODE_BATCH lays NB tiles over the same region instead.
static constexpr int LDS_BASE_BYTES =
    16 * QK_DIM * 2 + 256 * 32 * 4 + (MPK_MLA_DECODE_DMA_PF ? 16 * QK_DIM * 2 : 0);
static constexpr int LDS_BATCH_BYTES = MPK_MLA_DECODE_BATCH * 16 * QK_DIM * 2;
static constexpr int LDS_TILE_BYTES =
    LDS_BATCH_BYTES > LDS_BASE_BYTES ? LDS_BATCH_BYTES : LDS_BASE_BYTES;
// MPK_MLA_QK_SPLIT hands the scores over at 96 KB of dynamic LDS,
// MPK_DEC_QK_KSPLIT its four waves' partials (4 KB there), and MPK_DEC_WPT
// its tiles' weights and maxima (4.25 KB). WPT's extra tile buffers sit
// below 72 KB, inside that, as do MPK_DEC_VT's V^T tiles (36..76 KB).
static constexpr int LDS_SCORES_END =
    MPK_DEC_WPT ? 101 * 1024
    : MPK_DEC_QK_KSPLIT ? 100 * 1024 : (MPK_MLA_QK_SPLIT ? 98 * 1024 : 0);
static constexpr int LDS_BYTES =
    LDS_TILE_BYTES < LDS_SCORES_END ? LDS_SCORES_END : LDS_TILE_BYTES;

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
  if (LDS_BYTES > 65536) {
    hipFuncSetAttribute(
        reinterpret_cast<void const *>(&mla_entry<NUM_KV_CHUNKS, WRITE_THROUGH>),
        hipFuncAttributeMaxDynamicSharedMemorySize, LDS_BYTES);
  }
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
#if MPK_DEC_PART_F16
    // The write-through path stores f16 partials at the same element index.
    if (WRITE_THROUGH) {
      std::vector<_Float16> o_h(out_elems);
      hipMemcpy(o_h.data(), d_out, out_elems * 2, hipMemcpyDeviceToHost);
      for (size_t i = 0; i < out_elems; ++i) {
        o_acc[i] = (float)o_h[i];
      }
    } else {
      hipMemcpy(o_acc.data(), d_out, out_bytes, hipMemcpyDeviceToHost);
    }
#else
    hipMemcpy(o_acc.data(), d_out, out_bytes, hipMemcpyDeviceToHost);
#endif
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

#if MPK_DEC_MERGE_LL32
// One q group's chunks, own heads [out_head_lo, +8), as the head-local fused
// decode runs them; ll_o / ll_lse non-null selects MPK_DEC_MERGE_LL32's words.
template <int NUM_KV_CHUNKS>
__global__ void mla_entry_ll32(void const *q_workspace, void const *kv_cache,
                               void *output, void *lse, int const *qo_indptr,
                               int const *kv_indptr, int const *kv_indices,
                               int const *kv_last_page_len, float scale_s,
                               int g, int out_head_lo,
                               unsigned long long *ll_o,
                               unsigned long long *ll_lse, unsigned epoch) {
  kernel::gang_mla_decode_kernel<bf16, NUM_Q_HEADS, KV_LORA_RANK,
                                 QK_ROPE_HEAD_DIM, PAGE_SIZE, MAX_SEQ_LEN,
                                 NUM_KV_CHUNKS, Q_WORKSPACE_STRIDE,
                                 KV_CACHE_STRIDE, /*WRITE_THROUGH=*/true>(
      q_workspace, kv_cache, output, lse, qo_indptr, kv_indptr, kv_indices,
      kv_last_page_len, NUM_Q_GROUPS * NUM_KV_CHUNKS,
      blockIdx.x * NUM_Q_GROUPS + g, scale_s, out_head_lo, /*out_heads=*/8,
      /*tiles_in_lds=*/0, -1, -1, -1, ll_o, ll_lse, epoch);
}

// The words must carry exactly the f32 partials and lse the default path
// stores, with the epoch, for every live chunk, and nothing for an empty one.
template <int NUM_KV_CHUNKS>
static int run_case_ll32(int seqlen, int out_head_lo) {
  int const num_pages = (seqlen + PAGE_SIZE - 1) / PAGE_SIZE;
  int const last_page_len = seqlen - (num_pages - 1) * PAGE_SIZE;
  std::vector<int> h_kv_indices(num_pages);
  for (int p = 0; p < num_pages; ++p) {
    h_kv_indices[p] = num_pages - 1 - p;
  }
  std::vector<uint16_t> q_b((size_t)NUM_Q_HEADS * QK_DIM);
  for (auto &v : q_b) {
    v = to_bf16_bits(frand());
  }
  std::vector<uint16_t> kv_b((size_t)num_pages * PAGE_SIZE * KV_CACHE_STRIDE);
  for (auto &v : kv_b) {
    v = to_bf16_bits(frand());
  }
  int const g = 2;
  constexpr int LSE_S = NUM_Q_GROUPS * NUM_KV_CHUNKS * 16;
  size_t const o_elems = (size_t)LSE_S * KV_LORA_RANK;
  size_t const wo_n = (size_t)NUM_KV_CHUNKS * 8 * KV_LORA_RANK;
  size_t const wl_n = (size_t)8 * NUM_KV_CHUNKS;
  unsigned const epoch = 0x456789au;
  unsigned long long const poison = ((unsigned long long)(epoch - 7) << 32) |
                                    0x7fc00001ull;
  std::vector<unsigned long long> h_poison(wo_n, poison);

  void *d_q, *d_kv, *d_out, *d_lse;
  unsigned long long *d_wo, *d_wl;
  int *d_qo, *d_kvp, *d_kvi, *d_lpl;
  hipMalloc(&d_q, q_b.size() * 2);
  hipMalloc(&d_kv, kv_b.size() * 2);
  hipMalloc(&d_out, o_elems * 4);
  hipMalloc(&d_lse, (size_t)LSE_S * 4);
  hipMalloc(&d_wo, wo_n * 8);
  hipMalloc(&d_wl, wl_n * 8);
  hipMalloc(&d_qo, 8);
  hipMalloc(&d_kvp, 8);
  hipMalloc(&d_kvi, (size_t)num_pages * 4);
  hipMalloc(&d_lpl, 4);
  int const h_qo[2] = {0, 1};
  int const h_kvp[2] = {0, num_pages};
  hipMemcpy(d_q, q_b.data(), q_b.size() * 2, hipMemcpyHostToDevice);
  hipMemcpy(d_kv, kv_b.data(), kv_b.size() * 2, hipMemcpyHostToDevice);
  hipMemset(d_out, 0, o_elems * 4);
  hipMemset(d_lse, 0, (size_t)LSE_S * 4);
  hipMemcpy(d_wo, h_poison.data(), wo_n * 8, hipMemcpyHostToDevice);
  hipMemcpy(d_wl, h_poison.data(), wl_n * 8, hipMemcpyHostToDevice);
  hipMemcpy(d_qo, h_qo, 8, hipMemcpyHostToDevice);
  hipMemcpy(d_kvp, h_kvp, 8, hipMemcpyHostToDevice);
  hipMemcpy(d_kvi, h_kv_indices.data(), (size_t)num_pages * 4,
            hipMemcpyHostToDevice);
  hipMemcpy(d_lpl, &last_page_len, 4, hipMemcpyHostToDevice);
  float const scale = 1.0f / sqrtf((float)QK_DIM) * LOG2E;
  auto *fn = &mla_entry_ll32<NUM_KV_CHUNKS>;
  if (LDS_BYTES > 65536) {
    hipFuncSetAttribute(reinterpret_cast<void const *>(fn),
                        hipFuncAttributeMaxDynamicSharedMemorySize, LDS_BYTES);
  }
  // f32 partials, then the words.
  hipLaunchKernelGGL(fn, dim3(NUM_KV_CHUNKS), dim3(BLOCK_THREADS), LDS_BYTES, 0,
                     d_q, d_kv, d_out, d_lse, d_qo, d_kvp, d_kvi, d_lpl, scale,
                     g, out_head_lo, nullptr, nullptr, 0u);
  hipLaunchKernelGGL(fn, dim3(NUM_KV_CHUNKS), dim3(BLOCK_THREADS), LDS_BYTES, 0,
                     d_q, d_kv, d_out, d_lse, d_qo, d_kvp, d_kvi, d_lpl, scale,
                     g, out_head_lo, d_wo, d_wl, epoch);
  hipError_t const err = hipDeviceSynchronize();
  std::vector<float> o(o_elems), l(LSE_S);
  std::vector<unsigned long long> wo(wo_n), wl(wl_n);
  hipMemcpy(o.data(), d_out, o_elems * 4, hipMemcpyDeviceToHost);
  hipMemcpy(l.data(), d_lse, (size_t)LSE_S * 4, hipMemcpyDeviceToHost);
  hipMemcpy(wo.data(), d_wo, wo_n * 8, hipMemcpyDeviceToHost);
  hipMemcpy(wl.data(), d_wl, wl_n * 8, hipMemcpyDeviceToHost);
  hipFree(d_q); hipFree(d_kv); hipFree(d_out); hipFree(d_lse); hipFree(d_wo);
  hipFree(d_wl); hipFree(d_qo); hipFree(d_kvp); hipFree(d_kvi); hipFree(d_lpl);

  int const ntiles = (seqlen + 15) / 16;
  int const tpc = (ntiles + NUM_KV_CHUNKS - 1) / NUM_KV_CHUNKS;
  int const n_live = tpc > 0 ? (ntiles + tpc - 1) / tpc : 0;
  int bad = 0;
  auto expect = [&](unsigned long long w, float f, bool live, char const *what,
                    int c, int h, int d) {
    unsigned u;
    __builtin_memcpy(&u, &f, 4);
    unsigned long long const want =
        live ? (((unsigned long long)epoch << 32) | u) : poison;
    if (w != want) {
      if (bad < 8) {
        printf("  %s chunk %d head %d dim %d: %016llx != %016llx\n", what, c,
               h, d, w, want);
      }
      ++bad;
    }
  };
  for (int c = 0; c < NUM_KV_CHUNKS; ++c) {
    bool const live = c < n_live;
    for (int h = 0; h < 8; ++h) {
      int const li = g * NUM_KV_CHUNKS * 16 + c * 16 + out_head_lo + h;
      expect(wl[(size_t)h * NUM_KV_CHUNKS + c], l[li], live, "lse", c, h, -1);
      for (int d = 0; d < KV_LORA_RANK; ++d) {
        expect(wo[((size_t)c * 8 + h) * KV_LORA_RANK + d],
               o[(size_t)li * KV_LORA_RANK + d], live, "o", c, h, d);
      }
    }
  }
  if (err != hipSuccess || bad) {
    printf("FAIL LL32 words chunks=%d seqlen=%-5d hb=%d: %d bad (%s)\n",
           NUM_KV_CHUNKS, seqlen, out_head_lo, bad, hipGetErrorString(err));
    return 1;
  }
  printf("PASS LL32 words chunks=%d seqlen=%-5d hb=%d live %d\n",
         NUM_KV_CHUNKS, seqlen, out_head_lo, n_live);
  return 0;
}
#endif

#if MPK_DEC_PRELOAD
// MPK_DEC_PRELOAD as the fused task runs it: up to two full tiles of the
// chunk, short of the last row, DMA'd into the decode's LDS buffers ahead of
// the call and handed over as tiles_in_lds. preload 0 is the plain call.
template <int NUM_KV_CHUNKS>
__global__ void mla_entry_pre(void const *q_workspace, void const *kv_cache,
                              void *output, void *lse, int const *qo_indptr,
                              int const *kv_indptr, int const *kv_indices,
                              int const *kv_last_page_len, float scale_s,
                              int g, int preload, int *npre_out) {
  int npre = 0;
  if (preload) {
    extern __shared__ char _t_pre_smem[];
    int const seqlen = (kv_indptr[1] - kv_indptr[0] - 1) * PAGE_SIZE +
                       kv_last_page_len[0];
    int const ntiles = (seqlen + 15) / 16;
    int const tpc = (ntiles + NUM_KV_CHUNKS - 1) / NUM_KV_CHUNKS;
    int const tok0 = (int)blockIdx.x * tpc * 16;
    if (tok0 < seqlen) {
      int ntok = tpc * 16;
      if (ntok > seqlen - tok0) {
        ntok = seqlen - tok0;
      }
      if (ntok > PAGE_SIZE - tok0 % PAGE_SIZE) {
        ntok = PAGE_SIZE - tok0 % PAGE_SIZE;
      }
      if (tok0 + ntok > seqlen - 1) {
        ntok = seqlen - 1 - tok0;
      }
      int const pid = kv_indices[kv_indptr[0] + tok0 / PAGE_SIZE];
      char const *base = static_cast<char const *>(kv_cache) +
                         ((size_t)pid * PAGE_SIZE + tok0 % PAGE_SIZE) *
                             (size_t)KV_CACHE_STRIDE * 2;
      constexpr int TILE_BYTES = 16 * KV_CACHE_STRIDE * 2;
      unsigned const wid = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
      unsigned const smem0 = __builtin_amdgcn_readfirstlane(
          static_cast<unsigned>(reinterpret_cast<uintptr_t>(_t_pre_smem)));
#pragma unroll
      for (int k = 0; k < 2; k++) {
        if (ntok >= 16) {
          kernel::gang_mla_decode_detail::mla_dma_tile(
              kernel::gang_mla_decode_detail::mla_buffer_rsrc(base,
                                                              TILE_BYTES),
              smem0 + k * TILE_BYTES + wid * 1024u, threadIdx.x * 16u,
              wid < 2u);
          base += TILE_BYTES;
          ntok -= 16;
          npre = k + 1;
        }
      }
    }
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    npre_out[blockIdx.x] = npre;
  }
  kernel::gang_mla_decode_kernel<bf16, NUM_Q_HEADS, KV_LORA_RANK,
                                 QK_ROPE_HEAD_DIM, PAGE_SIZE, MAX_SEQ_LEN,
                                 NUM_KV_CHUNKS, Q_WORKSPACE_STRIDE,
                                 KV_CACHE_STRIDE, /*WRITE_THROUGH=*/true>(
      q_workspace, kv_cache, output, lse, qo_indptr, kv_indptr, kv_indices,
      kv_last_page_len, NUM_Q_GROUPS * NUM_KV_CHUNKS,
      blockIdx.x * NUM_Q_GROUPS + g, scale_s, 0, 16, npre);
}

// The preloaded trips must give the plain call's partials and lse, bit for
// bit (MPK_DEC_PRE_NOBAR / MPK_DEC_QK2T change only who waits and when).
template <int NUM_KV_CHUNKS>
static int run_case_pre(int seqlen) {
  int const num_pages = (seqlen + PAGE_SIZE - 1) / PAGE_SIZE;
  int const last_page_len = seqlen - (num_pages - 1) * PAGE_SIZE;
  std::vector<int> h_kv_indices(num_pages);
  for (int p = 0; p < num_pages; ++p) {
    h_kv_indices[p] = num_pages - 1 - p;
  }
  std::vector<uint16_t> q_b((size_t)NUM_Q_HEADS * QK_DIM);
  for (auto &v : q_b) {
    v = to_bf16_bits(frand());
  }
  std::vector<uint16_t> kv_b((size_t)num_pages * PAGE_SIZE * KV_CACHE_STRIDE);
  for (auto &v : kv_b) {
    v = to_bf16_bits(frand());
  }
  int const g = 1;
  constexpr int LSE_S = NUM_Q_GROUPS * NUM_KV_CHUNKS * 16;
  size_t const o_elems = (size_t)LSE_S * KV_LORA_RANK;
  void *d_q, *d_kv, *d_o0, *d_o1, *d_l0, *d_l1;
  int *d_qo, *d_kvp, *d_kvi, *d_lpl, *d_npre;
  hipMalloc(&d_q, q_b.size() * 2);
  hipMalloc(&d_kv, kv_b.size() * 2);
  hipMalloc(&d_o0, o_elems * 4);
  hipMalloc(&d_o1, o_elems * 4);
  hipMalloc(&d_l0, (size_t)LSE_S * 4);
  hipMalloc(&d_l1, (size_t)LSE_S * 4);
  hipMalloc(&d_qo, 8);
  hipMalloc(&d_kvp, 8);
  hipMalloc(&d_kvi, (size_t)num_pages * 4);
  hipMalloc(&d_lpl, 4);
  hipMalloc(&d_npre, NUM_KV_CHUNKS * 4);
  int const h_qo[2] = {0, 1};
  int const h_kvp[2] = {0, num_pages};
  hipMemcpy(d_q, q_b.data(), q_b.size() * 2, hipMemcpyHostToDevice);
  hipMemcpy(d_kv, kv_b.data(), kv_b.size() * 2, hipMemcpyHostToDevice);
  hipMemset(d_o0, 0, o_elems * 4);
  hipMemset(d_o1, 0, o_elems * 4);
  hipMemset(d_l0, 0, (size_t)LSE_S * 4);
  hipMemset(d_l1, 0, (size_t)LSE_S * 4);
  hipMemcpy(d_qo, h_qo, 8, hipMemcpyHostToDevice);
  hipMemcpy(d_kvp, h_kvp, 8, hipMemcpyHostToDevice);
  hipMemcpy(d_kvi, h_kv_indices.data(), (size_t)num_pages * 4,
            hipMemcpyHostToDevice);
  hipMemcpy(d_lpl, &last_page_len, 4, hipMemcpyHostToDevice);
  float const scale = 1.0f / sqrtf((float)QK_DIM) * LOG2E;
  auto *fn = &mla_entry_pre<NUM_KV_CHUNKS>;
  if (LDS_BYTES > 65536) {
    hipFuncSetAttribute(reinterpret_cast<void const *>(fn),
                        hipFuncAttributeMaxDynamicSharedMemorySize, LDS_BYTES);
  }
  hipLaunchKernelGGL(fn, dim3(NUM_KV_CHUNKS), dim3(BLOCK_THREADS), LDS_BYTES, 0,
                     d_q, d_kv, d_o0, d_l0, d_qo, d_kvp, d_kvi, d_lpl, scale, g,
                     0, d_npre);
  hipLaunchKernelGGL(fn, dim3(NUM_KV_CHUNKS), dim3(BLOCK_THREADS), LDS_BYTES, 0,
                     d_q, d_kv, d_o1, d_l1, d_qo, d_kvp, d_kvi, d_lpl, scale, g,
                     1, d_npre);
  hipError_t const err = hipDeviceSynchronize();
  std::vector<unsigned> o0(o_elems), o1(o_elems), l0(LSE_S), l1(LSE_S);
  std::vector<int> npre(NUM_KV_CHUNKS);
  hipMemcpy(o0.data(), d_o0, o_elems * 4, hipMemcpyDeviceToHost);
  hipMemcpy(o1.data(), d_o1, o_elems * 4, hipMemcpyDeviceToHost);
  hipMemcpy(l0.data(), d_l0, (size_t)LSE_S * 4, hipMemcpyDeviceToHost);
  hipMemcpy(l1.data(), d_l1, (size_t)LSE_S * 4, hipMemcpyDeviceToHost);
  hipMemcpy(npre.data(), d_npre, NUM_KV_CHUNKS * 4, hipMemcpyDeviceToHost);
  hipFree(d_q); hipFree(d_kv); hipFree(d_o0); hipFree(d_o1); hipFree(d_l0);
  hipFree(d_l1); hipFree(d_qo); hipFree(d_kvp); hipFree(d_kvi); hipFree(d_lpl);
  hipFree(d_npre);
  // TEST_DUMP=<file>: append this case's partials and lse (plain call, then
  // preloaded call) so two builds can be compared byte for byte.
  if (char const *dump = getenv("TEST_DUMP")) {
    if (FILE *f = fopen(dump, "ab")) {
      fwrite(o0.data(), 4, o_elems, f);
      fwrite(l0.data(), 4, LSE_S, f);
      fwrite(o1.data(), 4, o_elems, f);
      fwrite(l1.data(), 4, LSE_S, f);
      fclose(f);
    }
  }
  int bad = 0, pre1 = 0, pre2 = 0, nonzero = 0;
  for (int c = 0; c < NUM_KV_CHUNKS; ++c) {
    pre1 += npre[c] == 1;
    pre2 += npre[c] == 2;
  }
  for (size_t i = 0; i < o_elems; ++i) {
    nonzero += o0[i] != 0u;
    if (o0[i] != o1[i]) {
      if (bad < 6) {
        printf("  o[%zu] chunk %zu head %zu dim %zu: %08x != %08x\n", i,
               (i / KV_LORA_RANK / 16) % NUM_KV_CHUNKS,
               (i / KV_LORA_RANK) % 16, i % KV_LORA_RANK, o1[i], o0[i]);
      }
      ++bad;
    }
  }
  for (int i = 0; i < LSE_S; ++i) {
    if (l0[i] != l1[i]) {
      if (bad < 6) {
        printf("  lse[%d]: %08x != %08x\n", i, l1[i], l0[i]);
      }
      ++bad;
    }
  }
  if (err != hipSuccess || bad || nonzero == 0) {
    printf("FAIL preload chunks=%d seqlen=%-5d: %d bad, %d nonzero (%s)\n",
           NUM_KV_CHUNKS, seqlen, bad, nonzero, hipGetErrorString(err));
    return 1;
  }
  printf("PASS preload chunks=%d seqlen=%-5d bit-identical (%d chunks 1 "
         "tile, %d 2 tiles in LDS)\n",
         NUM_KV_CHUNKS, seqlen, pre1, pre2);
  return 0;
}
#endif

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
  // 1k/1k decode shapes: 3-4 trips per chunk, a partial tail tile (1030,
  // 1537) and an exact multiple (2048). Under MPK_MLA_DECODE_DMA_PF these
  // are the cases that take the DMA path and its register-path fallback.
  rc |= run_case<16, WRITE_THROUGH>(1000);
  rc |= run_case<32, WRITE_THROUGH>(1030);
  rc |= run_case<32, WRITE_THROUGH>(1537);
  rc |= run_case<32, WRITE_THROUGH>(2048);
  // MPK_MLA_HEAD_LOCAL's wider splits: 1-2 tiles per chunk at 1k/1k, most
  // chunks empty at a short sequence.
  rc |= run_case<64, WRITE_THROUGH>(17);
  rc |= run_case<64, WRITE_THROUGH>(1030);
  rc |= run_case<64, WRITE_THROUGH>(2048);
  rc |= run_case<128, WRITE_THROUGH>(17);
  rc |= run_case<128, WRITE_THROUGH>(1030);
  rc |= run_case<128, WRITE_THROUGH>(1537);
  rc |= run_case<128, WRITE_THROUGH>(2048);
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
#if MPK_DEC_PRELOAD
  printf("=== MPK_DEC_PRELOAD tiles in LDS ===\n");
  for (int seqlen : {17, 40, 1030, 1537, 2048}) {
    rc |= run_case_pre<64>(seqlen);
  }
  rc |= run_case_pre<32>(1537);
#endif
#if MPK_DEC_MERGE_LL32
  printf("=== MPK_DEC_MERGE_LL32 words ===\n");
  for (int hb : {0, 8}) {
    for (int seqlen : {17, 1030, 1537, 2048}) {
      rc |= run_case_ll32<64>(seqlen, hb);
    }
  }
#endif
  return rc;
}
