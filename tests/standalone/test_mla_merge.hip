/* Copyright 2025 CMU
 *
 * Numerical test for merge_splitkv_ck_fmha, the split-KV merge the GLM-5
 * fused MLA layer runs on device.
 *
 * WHY THIS EXISTS. test_mla_decode.hip covers the decode producer, but it
 * merges the partials ON THE HOST, so the device merge has never been checked
 * numerically. That gap is not hypothetical: instantiating the decode with
 * WRITE_THROUGH=true (which only the fused layer does) exposed a wrong-value
 * bug in st_wt_u128 that every host-merged test passed straight through.
 * The merge has its own WRITE_THROUGH epilogue and its own DIM_SPLITS
 * geometry, and the shipping fused configuration -- WRITE_THROUGH=true,
 * DIM_SPLITS=32 -- selects a VAL_PER_THREAD==1 path with a cross-lane shuffle
 * that no other configuration reaches.
 *
 * Geometry is GLM-5's: 64 q heads as NUM_QO_GROUPS=4 groups of
 * NUM_QO_HEADS_PER_KV=16, HEAD_DIM = kv_lora_rank = 512, NUM_KV_CHUNKS=16.
 *
 * Reference is the same online-softmax merge in double precision on the host:
 *
 *   out[h][d] = sum_c exp(lse[c][h]) * o_acc[c][h][d] / sum_c exp(lse[c][h])
 *
 * Empty chunks (the decode's early return) write lse = -1e30 and never write
 * o_acc, so the test deliberately fills those o_acc slots with garbage and
 * expects the merge to give them zero weight.
 *
 * Build/run: see the command in test_mla_decode.hip's header; same flags.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <vector>

// The LL path stages its lse words in dynamic LDS at MPK_MRG_LL_LDS_OFF.
#ifndef MPK_MRG_LL_LDS_OFF
#define MPK_MRG_LL_LDS_OFF 0
#endif
static constexpr int MRG_DYN_LDS = MPK_MRG_LL_LDS_OFF + 4096;
#include "mirage/persistent_kernel/mpk_atoms.cuh"
#include "mirage/persistent_kernel/tasks/common/utils.cuh"
#include "mirage/persistent_kernel/tasks/ampere/merge_splitkv.cuh"

using bf16 = __hip_bfloat16;

static constexpr int QO_PER_KV = 16;  // NUM_QO_HEADS_PER_KV
static constexpr int QO_GROUPS = 4;   // NUM_QO_GROUPS
static constexpr int HEAD_DIM = 512;  // kv_lora_rank
// 32 is the shipping 1k/1k geometry (GLM_MLA_NUM_KV_CHUNKS=32).
#ifndef TEST_NUM_CHUNKS
#define TEST_NUM_CHUNKS 16
#endif
static constexpr int NUM_CHUNKS = TEST_NUM_CHUNKS; // NUM_KV_CHUNKS
static constexpr int PAGE_SIZE = 4096;
static constexpr int BLOCK_THREADS = 256;

static constexpr int LSE_TOKEN_STRIDE = QO_GROUPS * NUM_CHUNKS * QO_PER_KV;
static constexpr int OUT_TOKEN_STRIDE = QO_GROUPS * QO_PER_KV * HEAD_DIM;

template <bool WRITE_THROUGH, int DIM_SPLITS, int HEADS_N = QO_PER_KV>
__global__ void merge_entry(float const *lse,
                            float const *o_acc,
                            int const *qo_indptr,
                            int const *kv_indptr,
                            int const *kv_last_page_len,
                            bf16 *out,
                            int head_base,
                            unsigned long long const *ll_o = nullptr,
                            unsigned long long const *ll_lse = nullptr,
                            unsigned ll_epoch = 0) {
  kernel::merge_splitkv_ck_fmha<bf16,
                                QO_PER_KV,
                                QO_GROUPS,
                                HEAD_DIM,
                                NUM_CHUNKS,
                                /*KV_CHUNK_SIZE=*/128,
                                PAGE_SIZE,
                                WRITE_THROUGH,
                                DIM_SPLITS,
                                HEADS_N>(lse,
                                         o_acc,
                                         qo_indptr,
                                         kv_indptr,
                                         kv_last_page_len,
                                         /*request_id=*/0,
                                         out,
                                         /*merge_task_offset=*/blockIdx.x,
                                         /*sinks_ptr=*/nullptr,
                                         head_base,
                                         -1,
                                         -1,
                                         nullptr,
                                         0,
                                         0,
                                         ll_o,
                                         ll_lse,
                                         ll_epoch);
}

static unsigned short to_bf16_bits(float f) {
  bf16 const b = (bf16)f;
  unsigned short u;
  memcpy(&u, &b, 2);
  return u;
}
static float from_bf16_bits(unsigned short u) {
  unsigned const w = (unsigned)u << 16;
  float f;
  memcpy(&f, &w, 4);
  return f;
}

static float frand() {
  return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
}

// active_chunks: chunks [0, active_chunks) carry real partials, the rest are
// the decode's empty-chunk early return (lse = -1e30, o_acc never written).
// HEADS_N < 16 merges heads [head_base, head_base + HEADS_N) of each group
// only (MPK_MLA_MERGE_OWN_HEADS), and only those are checked.
// ll: MPK_DEC_MERGE_LL -- the partials arrive as epoch words the way the
// decode writes them (o as bf16 pairs at half the f32 index, lse as f32), and
// the reference uses the bf16-rounded o. Empty chunks' o words keep a stale
// epoch, which the merge must ignore.
template <bool WRITE_THROUGH, int DIM_SPLITS, int HEADS_N = QO_PER_KV>
static int run_case(int active_chunks, char const *tag, int head_base = 0,
                    bool ll = false) {
  size_t const lse_n = (size_t)LSE_TOKEN_STRIDE;
  size_t const o_n = lse_n * HEAD_DIM;
  size_t const out_n = (size_t)OUT_TOKEN_STRIDE;

  std::vector<float> h_lse(lse_n), h_o(o_n);
  for (size_t i = 0; i < o_n; ++i) {
    // Garbage in every slot, including the ones empty chunks never write --
    // large enough that a nonzero weight on them is unmistakable.
    h_o[i] = frand() * 8.0f;
  }
  for (int g = 0; g < QO_GROUPS; ++g) {
    for (int c = 0; c < NUM_CHUNKS; ++c) {
      for (int h = 0; h < QO_PER_KV; ++h) {
        int idx = g * NUM_CHUNKS * QO_PER_KV + c * QO_PER_KV + h;
        if (c < active_chunks) {
          h_lse[idx] = frand() * 3.0f;
          for (int d = 0; d < HEAD_DIM; ++d) {
            h_o[(size_t)idx * HEAD_DIM + d] = frand();
          }
        } else {
          h_lse[idx] = -1e30f; // empty chunk
        }
      }
    }
  }

  unsigned const epoch = 0x1234567u;
  unsigned long long *d_ll_o = nullptr, *d_ll_lse = nullptr;
  if (ll) {
    std::vector<unsigned long long> w_o(o_n / 2), w_lse(lse_n);
    for (size_t i = 0; i < lse_n; ++i) {
      unsigned u;
      memcpy(&u, &h_lse[i], 4);
      w_lse[i] = ((unsigned long long)epoch << 32) | u;
    }
    for (size_t w = 0; w < o_n / 2; ++w) {
      size_t const lse_idx = (2 * w) / HEAD_DIM;
      bool const empty = h_lse[lse_idx] == -1e30f;
      unsigned short const lo = to_bf16_bits(h_o[2 * w]);
      unsigned short const hi = to_bf16_bits(h_o[2 * w + 1]);
      w_o[w] = ((unsigned long long)(empty ? epoch - 1 : epoch) << 32) |
               lo | ((unsigned)hi << 16);
      // The reference sees what the words carry.
      h_o[2 * w] = from_bf16_bits(lo);
      h_o[2 * w + 1] = from_bf16_bits(hi);
    }
    hipMalloc(&d_ll_o, w_o.size() * 8);
    hipMalloc(&d_ll_lse, w_lse.size() * 8);
    hipMemcpy(d_ll_o, w_o.data(), w_o.size() * 8, hipMemcpyHostToDevice);
    hipMemcpy(d_ll_lse, w_lse.data(), w_lse.size() * 8, hipMemcpyHostToDevice);
  }
  float *d_lse, *d_o;
  bf16 *d_out;
  int *d_qo, *d_kvp, *d_lastlen;
  hipMalloc(&d_lse, lse_n * 4);
  hipMalloc(&d_o, o_n * 4);
  hipMalloc(&d_out, out_n * sizeof(bf16));
  hipMalloc(&d_qo, 2 * 4);
  hipMalloc(&d_kvp, 2 * 4);
  hipMalloc(&d_lastlen, 4);
  hipMemcpy(d_lse, h_lse.data(), lse_n * 4, hipMemcpyHostToDevice);
  hipMemcpy(d_o, h_o.data(), o_n * 4, hipMemcpyHostToDevice);
  hipMemset(d_out, 0, out_n * sizeof(bf16));
  int const h_qo[2] = {0, 1};
  int const h_kvp[2] = {0, 1};
  int const h_ll[1] = {257};
  hipMemcpy(d_qo, h_qo, 8, hipMemcpyHostToDevice);
  hipMemcpy(d_kvp, h_kvp, 8, hipMemcpyHostToDevice);
  hipMemcpy(d_lastlen, h_ll, 4, hipMemcpyHostToDevice);

  int const grid = QO_GROUPS * DIM_SPLITS;
  hipLaunchKernelGGL((merge_entry<WRITE_THROUGH, DIM_SPLITS, HEADS_N>),
                     dim3(grid), dim3(BLOCK_THREADS), MRG_DYN_LDS, 0,
                     d_lse, d_o, d_qo, d_kvp, d_lastlen, d_out, head_base,
                     d_ll_o, d_ll_lse, epoch);
  hipError_t err = hipDeviceSynchronize();
  if (err != hipSuccess) {
    printf("FAIL %-28s launch: %s\n", tag, hipGetErrorString(err));
    return 1;
  }

  std::vector<bf16> h_out(out_n);
  hipMemcpy(h_out.data(), d_out, out_n * sizeof(bf16), hipMemcpyDeviceToHost);

  int bad = 0;
  float max_err = 0.0f, ref_max = 0.0f;
  for (int g = 0; g < QO_GROUPS && bad < 8; ++g) {
    for (int h = head_base; h < head_base + HEADS_N && bad < 8; ++h) {
      double denom = 0.0, mx = -1e300;
      for (int c = 0; c < active_chunks; ++c) {
        mx = std::max(mx, (double)h_lse[g * NUM_CHUNKS * QO_PER_KV +
                                        c * QO_PER_KV + h]);
      }
      for (int c = 0; c < active_chunks; ++c) {
        denom += exp((double)h_lse[g * NUM_CHUNKS * QO_PER_KV + c * QO_PER_KV +
                                   h] - mx);
      }
      for (int d = 0; d < HEAD_DIM; ++d) {
        double acc = 0.0;
        for (int c = 0; c < active_chunks; ++c) {
          int idx = g * NUM_CHUNKS * QO_PER_KV + c * QO_PER_KV + h;
          acc += exp((double)h_lse[idx] - mx) * (double)h_o[(size_t)idx * HEAD_DIM + d];
        }
        float ref = (float)(denom > 0.0 ? acc / denom : 0.0);
        float got =
            (float)h_out[(size_t)g * QO_PER_KV * HEAD_DIM + (size_t)h * HEAD_DIM + d];
        ref_max = std::max(ref_max, fabsf(ref));
        float e = fabsf(ref - got);
        max_err = std::max(max_err, e);
        if (e > 6e-3f * std::max(1.0f, fabsf(ref)) && bad < 8) {
          printf("  g%d h%-2d dim %-4d: %10.6f != %10.6f\n", g, h, d, got, ref);
          ++bad;
        }
      }
    }
  }
  hipFree(d_lse); hipFree(d_o); hipFree(d_out);
  if (ll) {
    hipFree(d_ll_o);
    hipFree(d_ll_lse);
  }
  hipFree(d_qo); hipFree(d_kvp); hipFree(d_lastlen);

  if (bad) {
    printf("FAIL %-28s max abs err %.3e\n", tag, max_err);
    return 1;
  }
  printf("PASS %-28s max abs err %.3e (|ref|max %.3e)\n", tag, max_err, ref_max);
  return 0;
}

int main() {
  srand(20250905);
  int rc = 0;
  // The shipping fused configuration is WRITE_THROUGH=true, DIM_SPLITS=32.
  // active_chunks 16 is seqlen>=241 (every chunk carries a tile); 9 and 15 are
  // the shorter sequences that decode correctly today.
  if (NUM_CHUNKS != 16) {
    rc |= run_case<true, 32>(NUM_CHUNKS, "wt=1 splits=32 active=all");
  }
  // 64/128 chunks (MPK_MLA_HEAD_LOCAL): a 1030-token sequence leaves just
  // over half of them occupied.
  if (NUM_CHUNKS > 32) {
    rc |= run_case<true, 32>(NUM_CHUNKS / 2 + 1, "wt=1 splits=32 active=half+1");
    // The shipping NP=8 shape: 8 own heads of the 16-head group, which
    // MPK_MERGE_HALVES runs on both group halves. active=9 leaves the second
    // chunk half empty.
    for (int hb = 0; hb < QO_PER_KV; hb += 8) {
      rc |= run_case<true, 32, 8>(NUM_CHUNKS, "wt=1 splits=32 own8 all", hb);
      rc |= run_case<true, 32, 8>(NUM_CHUNKS / 2 + 1,
                                  "wt=1 splits=32 own8 half+1", hb);
      rc |= run_case<true, 32, 8>(9, "wt=1 splits=32 own8 active=9", hb);
#ifdef TEST_MERGE_LL
      rc |= run_case<true, 32, 8>(NUM_CHUNKS, "LL own8 all", hb, true);
      rc |= run_case<true, 32, 8>(NUM_CHUNKS / 2 + 1, "LL own8 half+1", hb, true);
      rc |= run_case<true, 32, 8>(9, "LL own8 active=9", hb, true);
#endif
    }
  }
  rc |= run_case<false, 32>(16, "wt=0 splits=32 active=16");
  rc |= run_case<true, 32>(16, "wt=1 splits=32 active=16");
  rc |= run_case<true, 32>(15, "wt=1 splits=32 active=15");
  rc |= run_case<true, 32>(9, "wt=1 splits=32 active=9");
  rc |= run_case<true, 16>(16, "wt=1 splits=16 active=16");
  rc |= run_case<false, 1>(16, "wt=0 splits=1  active=16");
  return rc;
}
