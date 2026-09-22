// Correctness for the gfx1250 WMMA decode attention kernel.
//
// The thing under test is a data-distribution derivation, not an algorithm: the
// softmax, the online rescale and the sink correction are all copied in spirit
// from the MFMA kernel and are easy to get right. What is easy to get *wrong*
// is which lane holds which (query, kv, head-dim) triple, because
// v_wmma_f32_16x16x32_bf16 on a 32-lane wave distributes an accumulator
// completely differently from v_mfma_f32_16x16x16bf16 on a 64-lane one.
//
// Every such mistake produces a finite, plausible-looking output -- attending
// to the wrong tokens still yields a convex combination of value vectors. So
// this test is built so that a wrong mapping cannot survive:
//
//  1. Distinguishable values. Every KV token's value vector is a distinct
//     constant (token id + 1) at every dim, so an output is a weighted average
//     of token *ids*. Swap two tokens' weights and the number moves. Contrast
//     with random values, where a permutation error can average out.
//  2. Per-query-head distinct Q, so a kernel that broadcast one head's scores
//     to all heads (the natural frag_row/frag_half confusion) fails.
//  3. Per-head-dim distinct V, so a wrong my_dim fails even if the softmax
//     weights are perfect.
//  4. Poisoned output, checked explicitly -- a lane that never writes cannot
//     pass as a zero.
//  5. Exactly-representable inputs. Q, K entries are small multiples of 1/8,
//     so QK products and their sums are exact in bf16 and fp32; the only
//     inexactness left is exp2 itself, which is the same function on both
//     sides.
//
// Cases cover: the KV_TILE=32 boundary (a tile-aligned length and a ragged
// one), multi-page indirection with a non-identity page table, sliding window,
// and split-KV including a deliberately empty chunk.
//
// ── A note on LSE, and a discrepancy in the gfx950 kernel ──
//
// scale_s arrives with log2(e) folded in (src/kernel/task_register.cc:1650), so
// the running max is in log2 units. Both LSE consumers want natural log:
// merge_splitkv_ck_fmha multiplies the stored value by log2(e) on the way in,
// and the sink correction is sigmoid(LSE - sink) with sink in natural units.
// The conversion therefore has to reach the log2 part only:
//     LSE_nat = m_log2 * ln2 + ln(l)
// tasks/mi300/paged_attention_decode_minimal_hd64_mi300.cuh:468 instead writes
// `m_running + logf(l_sum)`, which is larger by m_running * (1 - ln2) -- for
// these cases, a flat 0.83. This test found that as a *constant* offset in
// every configuration while the attention outputs themselves were correct to
// 5e-3, which is the signature of a domain error rather than a mapping error.
// The mi450 kernel does the conversion properly, so this reference is written
// against the correct value and would fail against mi300's expression. Whether
// mi300 should be fixed is a separate question -- it is a live gfx950 path and
// its error partly cancels between the sink term and the merge -- so it is
// recorded here rather than silently changed.
#include <hip/hip_runtime.h>

#include "mirage/persistent_kernel/tasks/mi450/paged_attention_wmma_decode_hd64_mi450.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using bf16 = __hip_bfloat16;

static constexpr int HEAD_DIM = 64;
static constexpr int PAGE_SIZE = 16;
static constexpr int NUM_KV_HEADS = 2;
static constexpr int NUM_QO_PER_KV = 8;
static constexpr int Q_STRIDE = NUM_KV_HEADS * NUM_QO_PER_KV * HEAD_DIM;
static constexpr int KV_STRIDE = NUM_KV_HEADS * HEAD_DIM;
static constexpr int NTHREADS = 128; // 4 waves of 32

// scale_s carries 1/sqrt(d) AND log2(e), because the kernel's softmax is base-2.
static float const SCALE_S = (1.0f / std::sqrt((float)HEAD_DIM)) * 1.44269504f;

// The kernel body is inside #if defined(MIRAGE_ARCH_GFX1250), which is set only
// in the device pass. The host pass still has to see a __global__ with this
// signature to emit the stub, so the body is empty there -- same arrangement as
// test_gemm_wmma.hip.
template <int NUM_KV_CHUNKS>
__global__ void attn_probe(void const *q,
                           void *k,
                           void *v,
                           void *o,
                           void *lse,
                           int const *qo_indptr,
                           int const *kv_indptr,
                           int const *kv_indices,
                           int const *kv_last_page_len,
                           int kv_head_idx,
                           int kv_chunk_idx,
                           float scale_s,
                           int sliding_window,
                           void const *sinks) {
#if defined(MIRAGE_ARCH_GFX1250)
  kernel::mi450::paged_attention_wmma_decode_hd64<bf16,
                                                  NUM_QO_PER_KV,
                                                  HEAD_DIM,
                                                  PAGE_SIZE,
                                                  4096,
                                                  NUM_KV_CHUNKS,
                                                  Q_STRIDE,
                                                  KV_STRIDE,
                                                  NUM_KV_HEADS>(
      q, k, v, o, lse, qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
      /*request_id=*/0, kv_head_idx, kv_chunk_idx, scale_s, sliding_window,
      sinks);
#endif
}

// Values chosen so bf16 is exact: q in {0, 1/8, ..., 7/8}, k likewise.
//
// The index expression must be a MIXING hash, not an affine function of d.
// The first version of this test used ((head*3 + d*5) % 8) * 0.125, which is
// linear in d modulo 8 and made the scores hugely degenerate: every query head
// got the same LSE, and the max over KV 0-7 equalled the max over KV 8-15.
// Two real mapping bugs survived that data -- dropping the cross-lane tile_max
// exchange, and using the lane-local rescale instead of broadcasting it from
// the query's owning lane -- because both are no-ops when the values they
// disagree about happen to be equal. check_discriminating() below now fails the
// test if the data ever drifts back into that regime.
static unsigned mix(unsigned x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}
static float q_val(int head, int d) {
  return (float)(mix((unsigned)(head * 131 + d * 7 + 1)) % 8) * 0.125f;
}
static float k_val(int tok, int d) {
  return (float)(mix((unsigned)(tok * 911 + d * 13 + 7)) % 8) * 0.125f;
}
// Distinct per (token, dim) so a wrong token *or* a wrong dim both show up.
static float v_val(int tok, int d) {
  return (float)(tok + 1) + (float)d * 0.03125f;
}

// Verify the test data can actually distinguish the mapping bugs it is meant to
// catch. A correctness test whose inputs are degenerate passes for the wrong
// reason, and nothing else in the harness would notice.
//
// Two properties are required, both checked over the same 32-token tile the
// kernel processes:
//
//  1. Per-query-head LSE must differ. The accumulator rescale is broadcast from
//     the lane owning each query (o_acc[i] *= __shfl(rescale, ...)); if all
//     heads share a rescale, dropping that broadcast is undetectable.
//  2. Within a head, max(KV 0-15) must differ from max(KV 16-31). The tile_max
//     reduction ends with one __shfl_xor(.., 16) joining those two halves; if
//     they are already equal, removing it is undetectable.
static bool check_discriminating() {
  int distinct_lse = 0, split_max_differs = 0;
  double first_lse = 0.0;
  for (int h = 0; h < NUM_QO_PER_KV; ++h) {
    double lo = -INFINITY, hi = -INFINITY, sum = 0.0, m = -INFINITY;
    std::vector<double> lg(32);
    for (int t = 0; t < 32; ++t) {
      double dot = 0.0;
      for (int d = 0; d < HEAD_DIM; ++d) {
        dot += (double)q_val(h, d) * (double)k_val(t, d);
      }
      lg[t] = dot / std::sqrt((double)HEAD_DIM);
      (t < 16 ? lo : hi) = std::max(t < 16 ? lo : hi, lg[t]);
      m = std::max(m, lg[t]);
    }
    for (int t = 0; t < 32; ++t) {
      sum += std::exp(lg[t] - m);
    }
    double const lse = m + std::log(sum);
    if (h == 0) {
      first_lse = lse;
    } else if (std::fabs(lse - first_lse) > 1e-2) {
      ++distinct_lse;
    }
    if (std::fabs(lo - hi) > 1e-2) {
      ++split_max_differs;
    }
  }
  bool ok = true;
  if (distinct_lse == 0) {
    printf("FAIL: test data is degenerate -- every query head has the same "
           "LSE, so a missing per-query rescale broadcast cannot be "
           "detected.\n");
    ok = false;
  }
  if (split_max_differs < NUM_QO_PER_KV / 2) {
    printf("FAIL: test data is degenerate -- max(KV 0-15) == max(KV 16-31) for "
           "%d/%d heads, so a missing tile_max cross-lane exchange cannot be "
           "detected.\n",
           NUM_QO_PER_KV - split_max_differs, NUM_QO_PER_KV);
    ok = false;
  }
  return ok;
}

struct Case {
  char const *name;
  int seqlen;
  int sliding_window;
  bool with_sinks;
};

static int failures_total = 0;

template <int NUM_KV_CHUNKS>
static void run_case(Case const &c, int kv_head_idx) {
  int const npages = (c.seqlen + PAGE_SIZE - 1) / PAGE_SIZE;
  int const last_page_len = c.seqlen - (npages - 1) * PAGE_SIZE;

  // Non-identity page table: page i of this request lives at physical page
  // (2*i + 3). An implementation that ignored kv_indices and used the token
  // index directly would read the wrong tokens entirely.
  std::vector<int> h_indices(npages);
  int max_pid = 0;
  for (int i = 0; i < npages; ++i) {
    h_indices[i] = 2 * i + 3;
    max_pid = std::max(max_pid, h_indices[i]);
  }
  int const total_pages = max_pid + 1;

  std::vector<bf16> h_q(Q_STRIDE, __float2bfloat16(0.f));
  for (int h = 0; h < NUM_QO_PER_KV; ++h) {
    for (int d = 0; d < HEAD_DIM; ++d) {
      h_q[(kv_head_idx * NUM_QO_PER_KV + h) * HEAD_DIM + d] =
          __float2bfloat16(q_val(h, d));
    }
  }

  size_t const kv_elems = (size_t)total_pages * PAGE_SIZE * KV_STRIDE;
  std::vector<bf16> h_k(kv_elems, __float2bfloat16(0.f));
  std::vector<bf16> h_v(kv_elems, __float2bfloat16(0.f));
  for (int t = 0; t < c.seqlen; ++t) {
    int const pid = h_indices[t / PAGE_SIZE];
    size_t const row =
        ((size_t)pid * PAGE_SIZE + (t % PAGE_SIZE)) * KV_STRIDE +
        (size_t)kv_head_idx * HEAD_DIM;
    for (int d = 0; d < HEAD_DIM; ++d) {
      h_k[row + d] = __float2bfloat16(k_val(t, d));
      h_v[row + d] = __float2bfloat16(v_val(t, d));
    }
  }

  std::vector<bf16> h_sinks(NUM_KV_HEADS * NUM_QO_PER_KV);
  for (int i = 0; i < NUM_KV_HEADS * NUM_QO_PER_KV; ++i) {
    h_sinks[i] = __float2bfloat16((float)(i % 5) - 2.0f);
  }

  int const h_qo_indptr[2] = {0, 1};
  int const h_kv_indptr[2] = {0, npages};
  int const h_lp[1] = {last_page_len};

  void *d_q, *d_k, *d_v, *d_sinks;
  int *d_qo, *d_kvp, *d_idx, *d_lp;
  hipMalloc(&d_q, h_q.size() * 2);
  hipMalloc(&d_k, kv_elems * 2);
  hipMalloc(&d_v, kv_elems * 2);
  hipMalloc(&d_sinks, h_sinks.size() * 2);
  hipMalloc(&d_qo, 8);
  hipMalloc(&d_kvp, 8);
  hipMalloc(&d_idx, npages * 4);
  hipMalloc(&d_lp, 4);
  hipMemcpy(d_q, h_q.data(), h_q.size() * 2, hipMemcpyHostToDevice);
  hipMemcpy(d_k, h_k.data(), kv_elems * 2, hipMemcpyHostToDevice);
  hipMemcpy(d_v, h_v.data(), kv_elems * 2, hipMemcpyHostToDevice);
  hipMemcpy(d_sinks, h_sinks.data(), h_sinks.size() * 2, hipMemcpyHostToDevice);
  hipMemcpy(d_qo, h_qo_indptr, 8, hipMemcpyHostToDevice);
  hipMemcpy(d_kvp, h_kv_indptr, 8, hipMemcpyHostToDevice);
  hipMemcpy(d_idx, h_indices.data(), npages * 4, hipMemcpyHostToDevice);
  hipMemcpy(d_lp, h_lp, 4, hipMemcpyHostToDevice);

  // Output layout differs between the fused (bf16, NUM_QO_PER_KV*HEAD_DIM
  // within a Q_STRIDE row) and split (fp32, chunk-major) paths.
  constexpr int LSE_STRIDE = NUM_KV_HEADS * NUM_KV_CHUNKS * NUM_QO_PER_KV;
  size_t const o_elems =
      (NUM_KV_CHUNKS == 1) ? (size_t)Q_STRIDE
                           : (size_t)LSE_STRIDE * HEAD_DIM;
  size_t const o_bytes = o_elems * ((NUM_KV_CHUNKS == 1) ? 2 : 4);
  void *d_o, *d_lse;
  hipMalloc(&d_o, o_bytes);
  hipMalloc(&d_lse, LSE_STRIDE * 4);
  hipMemset(d_o, 0xCD, o_bytes);
  hipMemset(d_lse, 0xCD, LSE_STRIDE * 4);

  // The K/V pointers are passed ALREADY OFFSET by kv_head_idx * HEAD_DIM --
  // that is the convention gang_attention_mi300.cuh uses and the kernel relies
  // on it (it offsets Q by the head itself, but not K/V). Passing the unoffset
  // base makes the kernel read head 0, which in this test is all zeros: scores
  // collapse to 0, LSE comes out as exactly ln(seqlen), and every output is 0.
  bf16 *d_k_head = reinterpret_cast<bf16 *>(d_k) + kv_head_idx * HEAD_DIM;
  bf16 *d_v_head = reinterpret_cast<bf16 *>(d_v) + kv_head_idx * HEAD_DIM;

  for (int chunk = 0; chunk < NUM_KV_CHUNKS; ++chunk) {
    hipLaunchKernelGGL((attn_probe<NUM_KV_CHUNKS>), dim3(1), dim3(NTHREADS), 0,
                       0, d_q, d_k_head, d_v_head, d_o, d_lse, d_qo, d_kvp,
                       d_idx, d_lp, kv_head_idx, chunk, SCALE_S,
                       c.sliding_window, c.with_sinks ? d_sinks : nullptr);
  }
  hipError_t err = hipDeviceSynchronize();
  if (err != hipSuccess) {
    printf("FAIL [%s]: %s\n", c.name, hipGetErrorString(err));
    ++failures_total;
    return;
  }

  std::vector<char> h_o(o_bytes);
  std::vector<float> h_lse(LSE_STRIDE);
  hipMemcpy(h_o.data(), d_o, o_bytes, hipMemcpyDeviceToHost);
  hipMemcpy(h_lse.data(), d_lse, LSE_STRIDE * 4, hipMemcpyDeviceToHost);

  // ── Host reference, in double ──
  // The kernel aligns the window start down to a tile boundary, so the
  // reference must too -- otherwise a correct kernel "fails" on a policy
  // difference rather than a bug.
  int kv_start = 0;
  if (c.sliding_window > 0 && c.seqlen > c.sliding_window) {
    kv_start = ((c.seqlen - c.sliding_window) / 32) * 32;
  }

  int local_fail = 0;
  double worst = 0.0;

  for (int h = 0; h < NUM_QO_PER_KV; ++h) {
    // Global softmax over the whole attended range, then per-chunk partials
    // are re-derived from it -- so the reference does not reimplement the
    // kernel's chunking, it only reimplements attention.
    std::vector<double> logits(c.seqlen - kv_start);
    for (int t = kv_start; t < c.seqlen; ++t) {
      double dot = 0.0;
      for (int d = 0; d < HEAD_DIM; ++d) {
        dot += (double)q_val(h, d) * (double)k_val(t, d);
      }
      logits[t - kv_start] = dot / std::sqrt((double)HEAD_DIM);
    }

    int const ntiles = (int)((logits.size() + 31) / 32);
    int const tiles_per_chunk =
        (ntiles + NUM_KV_CHUNKS - 1) / NUM_KV_CHUNKS;

    for (int chunk = 0; chunk < NUM_KV_CHUNKS; ++chunk) {
      int const t0 = chunk * tiles_per_chunk * 32;
      int t1 = std::min((chunk + 1) * tiles_per_chunk * 32, (int)logits.size());
      bool const empty = t0 >= (int)logits.size();

      double m = -INFINITY, l = 0.0;
      std::vector<double> acc(HEAD_DIM, 0.0);
      if (!empty) {
        for (int i = t0; i < t1; ++i) {
          m = std::max(m, logits[i]);
        }
        for (int i = t0; i < t1; ++i) {
          double const p = std::exp(logits[i] - m);
          l += p;
          for (int d = 0; d < HEAD_DIM; ++d) {
            acc[d] += p * (double)v_val(kv_start + i, d);
          }
        }
      }
      double const lse_want = empty ? -1e30 : (m + std::log(l));

      float const lse_got =
          h_lse[kv_head_idx * NUM_KV_CHUNKS * NUM_QO_PER_KV +
                chunk * NUM_QO_PER_KV + h];
      unsigned lse_raw;
      __builtin_memcpy(&lse_raw, &lse_got, 4);
      if (lse_raw == 0xCDCDCDCDu) {
        printf("FAIL [%s] h=%d chunk=%d: LSE never written\n", c.name, h, chunk);
        ++local_fail;
        continue;
      }
      // LSE is a log of a sum of exponentials of bf16-rounded dot products;
      // 3e-3 absolute is well inside one bf16 ulp of the logits and far tighter
      // than any mapping error could survive.
      if (!empty && std::fabs(lse_got - lse_want) > 3e-3) {
        if (++local_fail <= 5) {
          printf("FAIL [%s] h=%d chunk=%d LSE: got %.6f want %.6f\n", c.name, h,
                 chunk, lse_got, lse_want);
        }
      }
      if (empty) {
        if (lse_got > -1e29f) {
          printf("FAIL [%s] h=%d chunk=%d: empty chunk LSE=%g, want -1e30\n",
                 c.name, h, chunk, lse_got);
          ++local_fail;
        }
        continue; // empty chunk writes no output; nothing else to check
      }

      for (int d = 0; d < HEAD_DIM; ++d) {
        double want = acc[d] / l;
        float got;
        if constexpr (NUM_KV_CHUNKS == 1) {
          bf16 const *o = reinterpret_cast<bf16 const *>(h_o.data());
          bf16 const val =
              o[(kv_head_idx * NUM_QO_PER_KV + h) * HEAD_DIM + d];
          unsigned short raw;
          __builtin_memcpy(&raw, &val, 2);
          if (raw == 0xCDCD) {
            if (++local_fail <= 5) {
              printf("FAIL [%s] h=%d d=%d: output still poisoned\n", c.name, h,
                     d);
            }
            continue;
          }
          got = __bfloat162float(val);
          if (c.with_sinks) {
            double const sink = __bfloat162float(h_sinks[
                kv_head_idx * NUM_QO_PER_KV + h]);
            want *= 1.0 / (1.0 + std::exp(sink - lse_want));
          }
        } else {
          float const *o = reinterpret_cast<float const *>(h_o.data());
          got = o[(size_t)kv_head_idx * NUM_KV_CHUNKS * NUM_QO_PER_KV *
                      HEAD_DIM +
                  (size_t)chunk * NUM_QO_PER_KV * HEAD_DIM +
                  (size_t)h * HEAD_DIM + d];
          unsigned raw;
          __builtin_memcpy(&raw, &got, 4);
          if (raw == 0xCDCDCDCDu) {
            if (++local_fail <= 5) {
              printf("FAIL [%s] h=%d chunk=%d d=%d: output still poisoned\n",
                     c.name, h, chunk, d);
            }
            continue;
          }
        }

        // Finiteness before tolerance. The poison check above is not a
        // substitute: arithmetic NaN has different bits from 0xCDCDCDCD, so the
        // slot looks legitimately written, and `rel > 1e-2` is false for NaN --
        // a kernel producing NaN everywhere would report worst_rel=0.00e+00 and
        // pass. Attention is the most exposed kernel here, since a broken
        // softmax denominator or an all--inf row gives 0/0 directly.
        if (!std::isfinite(got) || !std::isfinite(want)) {
          if (++local_fail <= 5) {
            printf("FAIL [%s] h=%d chunk=%d d=%d: NON-FINITE got %f want %f -- "
                   "a tolerance check cannot detect this\n",
                   c.name, h, chunk, d, got, want);
          }
          continue;
        }

        double const rel = std::fabs((got - want) / (want == 0.0 ? 1.0 : want));
        worst = std::max(worst, rel);
        // Outputs are ~O(1..seqlen) weighted averages accumulated in bf16
        // fragments, so 1% is the honest bar. A one-token mapping error moves
        // the result by a whole token id -- orders of magnitude more than this.
        if (rel > 1e-2) {
          if (++local_fail <= 5) {
            printf("FAIL [%s] h=%d chunk=%d d=%d: got %.5f want %.5f "
                   "(rel %.4f)\n",
                   c.name, h, chunk, d, got, want, rel);
          }
        }
      }
    }
  }

  printf("%-28s seq=%-4d win=%-4d sinks=%d chunks=%d kvh=%d worst_rel=%.2e "
         "%s\n",
         c.name, c.seqlen, c.sliding_window, (int)c.with_sinks, NUM_KV_CHUNKS,
         kv_head_idx, worst, local_fail ? "FAIL" : "ok");
  failures_total += local_fail;

  hipFree(d_q);
  hipFree(d_k);
  hipFree(d_v);
  hipFree(d_sinks);
  hipFree(d_o);
  hipFree(d_lse);
  hipFree(d_qo);
  hipFree(d_kvp);
  hipFree(d_idx);
  hipFree(d_lp);
}

int main() {
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);
  printf("device=%s warpSize=%d threads=%d waves=%d\n", prop.gcnArchName,
         prop.warpSize, NTHREADS, NTHREADS / prop.warpSize);
  if (prop.warpSize != 32) {
    printf("SKIP: this kernel is wave32-only (gfx1250).\n");
    return 0;
  }
  if (!check_discriminating()) {
    return 1;
  }

  // seq=32 is exactly one tile; 47 is ragged within the last tile and the last
  // page; 96 is three tiles across six pages; 17 is a single token past a page
  // boundary, which is the decode steady state.
  Case const cases[] = {
      {"one-tile", 32, 0, false},
      {"ragged-tail", 47, 0, false},
      {"multi-page", 96, 0, false},
      {"page-boundary+1", 17, 0, false},
      {"with-sinks", 47, 0, true},
      {"sliding-window", 96, 40, false},
  };
  for (Case const &c : cases) {
    run_case<1>(c, /*kv_head_idx=*/1); // head 1, not 0: catches a dropped
                                       // kv_head_idx offset
  }

  // Split-KV. seq=40 over 4 chunks gives 2 tiles for 4 chunks, so chunks 2 and
  // 3 are empty -- exercising the -1e30 LSE stamp and early return.
  Case const split[] = {
      {"split2-even", 128, 0, false},
      {"split4-empty-chunks", 40, 0, false},
  };
  run_case<2>(split[0], 1);
  run_case<4>(split[1], 1);

  if (failures_total == 0) {
    printf("PASS: WMMA decode attention matches host double reference.\n");
    return 0;
  }
  printf("FAIL: %d mismatch(es).\n", failures_total);
  return 1;
}
