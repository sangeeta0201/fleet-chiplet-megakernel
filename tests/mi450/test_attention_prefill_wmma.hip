// Correctness for the gfx1250 WMMA PREFILL attention kernel (seqlen_q > 1).
//
// The decode kernel's test (test_attention_wmma.hip) already pins the fragment
// mapping, the online softmax, the sink correction and the LSE domain. The
// prefill kernel reuses all of that unchanged -- the 16 accumulator columns are
// query *tokens* here rather than query *heads*, but no fragment math cares.
// So this test deliberately does not re-litigate those. It targets what is
// actually new, which is the mask.
//
// ── The one bug this test exists to catch ──
//
// The convention is BOTTOM-RIGHT causal. Query row i attends to absolute KV
// positions [q_abs - W + 1, q_abs] where q_abs = i + (seqlen_k - seqlen_q).
// The tempting reading -- query i attends to [0, i] -- is TOP-LEFT, and the two
// are identical whenever seqlen_k == seqlen_q. That is the trap: a fresh
// prefill of an empty cache cannot tell them apart, and that is the case
// everyone writes first and tests with.
//
// They diverge only when the KV cache already holds tokens, i.e. chunked
// prefill (delta > 0), which is exactly the production path for a long prompt
// processed in pieces, or a multi-turn conversation. Under a top-left reading
// every query attends far too early in the sequence and never sees the tokens
// it should; the output stays finite and is a perfectly plausible convex
// combination of value vectors. Nothing crashes, and a numerical eyeball does
// not catch it.
//
// Hence: the case list below is built around delta. "aligned" cases have
// delta == 0 and are the weak ones; the "chunked" cases have delta > 0 and are
// the ones that would fail a top-left implementation. assert_delta_matters()
// verifies that claim directly by computing both conventions on the host and
// failing the test if the case list ever stops distinguishing them -- so this
// test cannot quietly decay into passing for the wrong reason the way an
// all-delta-0 case list would.
//
// ── Value design (same rationale as the decode test) ──
//
// V[token][dim] = (token + 1) + dim/32, distinct per (token, dim), so the
// output is a weighted average of token *ids*: attending to the wrong tokens
// moves the number rather than averaging out as it would with random values.
// Q/K are multiples of 1/8 drawn through a mixing hash so bf16 is exact and the
// scores are not degenerate. Outputs are poisoned with 0xCD and the poison is
// checked, so a lane that never writes cannot pass as a zero.
#include <hip/hip_runtime.h>

#include "mirage/persistent_kernel/tasks/mi450/paged_attention_wmma_decode_hd64_mi450.cuh"
#include "mirage/persistent_kernel/tasks/mi450/paged_attention_wmma_prefill_hd64_mi450.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using bf16 = __hip_bfloat16;

static constexpr int HEAD_DIM = 64;
static constexpr int PAGE_SIZE = 16;
static constexpr int NUM_KV_HEADS = 2;
static constexpr int NUM_QO_PER_KV = 8;
static constexpr int Q_STRIDE = NUM_KV_HEADS * NUM_QO_PER_KV * HEAD_DIM;
static constexpr int KV_STRIDE = NUM_KV_HEADS * HEAD_DIM;
static constexpr int NTHREADS = 128; // 4 waves of 32

static float const SCALE_S = (1.0f / std::sqrt((float)HEAD_DIM)) * 1.44269504f;

// Host pass sees an empty kernel namespace (MIRAGE_ARCH_GFX1250 is device-only),
// but still needs the __global__ stub. Same arrangement as the decode test.
template <int NUM_KV_CHUNKS>
__global__ void prefill_probe(void const *q,
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
  // Deliberately entered through the dispatch wrapper, not by calling the
  // prefill kernel directly: the seqlen_q==1 vs >1 routing is part of what is
  // under test, and a test that bypassed it would not notice the wrapper
  // sending prefill work to the decode kernel.
  kernel::mi450::paged_attention_wmma_split_kv_impl<bf16,
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

static unsigned mix(unsigned x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}
// Distinct per (query token, head, dim): a kernel that broadcast one token's
// row to the whole tile, or confused the token axis with the head axis, fails.
static float q_val(int tok, int head, int d) {
  return (float)(mix((unsigned)(tok * 7717 + head * 131 + d * 7 + 1)) % 8) *
         0.125f;
}
static float k_val(int tok, int d) {
  return (float)(mix((unsigned)(tok * 911 + d * 13 + 7)) % 8) * 0.125f;
}
static float v_val(int tok, int d) {
  return (float)(tok + 1) + (float)d * 0.03125f;
}

struct Case {
  char const *name;
  int seqlen_q;
  int seqlen_k; // total cache length; delta = seqlen_k - seqlen_q
  int sliding_window;
  bool with_sinks;
};

// Compute attention for one (query row, head) under a chosen mask convention.
// bottom_right=false is the WRONG convention, used only by
// assert_delta_matters() to prove the case list can tell them apart.
static void ref_attend(int q_row,
                       int h,
                       Case const &c,
                       bool bottom_right,
                       std::vector<double> &out,
                       double &lse_out) {
  int const delta = c.seqlen_k - c.seqlen_q;
  int const q_abs = bottom_right ? (q_row + delta) : q_row;

  int lo = 0, hi = q_abs; // inclusive
  if (c.sliding_window > 0) {
    lo = std::max(0, q_abs - c.sliding_window + 1);
  }
  if (hi >= c.seqlen_k) {
    hi = c.seqlen_k - 1;
  }

  out.assign(HEAD_DIM, 0.0);
  if (hi < lo) {
    lse_out = -INFINITY;
    return;
  }

  std::vector<double> logits(hi - lo + 1);
  double m = -INFINITY;
  for (int t = lo; t <= hi; ++t) {
    double dot = 0.0;
    for (int d = 0; d < HEAD_DIM; ++d) {
      dot += (double)q_val(q_row, h, d) * (double)k_val(t, d);
    }
    logits[t - lo] = dot / std::sqrt((double)HEAD_DIM);
    m = std::max(m, logits[t - lo]);
  }
  double sum = 0.0;
  for (size_t i = 0; i < logits.size(); ++i) {
    sum += std::exp(logits[i] - m);
  }
  for (int t = lo; t <= hi; ++t) {
    double const w = std::exp(logits[t - lo] - m) / sum;
    for (int d = 0; d < HEAD_DIM; ++d) {
      out[d] += w * (double)v_val(t, d);
    }
  }
  lse_out = m + std::log(sum);
}

// Does the case list actually discriminate the mask convention?
//
// Without this, a case list that drifted to all-delta-0 would still report PASS
// while testing nothing about the one thing this kernel adds over decode. This
// computes both conventions on the host and requires that they differ
// materially somewhere.
static bool assert_delta_matters(std::vector<Case> const &cases) {
  int discriminating = 0;
  double biggest = 0.0;
  for (Case const &c : cases) {
    std::vector<double> a, b;
    double la, lb;
    for (int q = 0; q < c.seqlen_q; ++q) {
      ref_attend(q, 0, c, true, a, la);
      ref_attend(q, 0, c, false, b, lb);
      for (int d = 0; d < HEAD_DIM; ++d) {
        biggest = std::max(biggest, std::fabs(a[d] - b[d]));
      }
    }
    if (biggest > 0.5) {
      ++discriminating;
    }
  }
  if (discriminating == 0) {
    printf("FAIL: no case in the list distinguishes bottom-right from "
           "top-left causal masking (max difference %.3f). Every case has "
           "seqlen_k == seqlen_q, so the mask convention -- the whole point "
           "of this test -- is untested.\n",
           biggest);
    return false;
  }
  printf("discriminating: %d case(s) separate bottom-right from top-left "
         "causal, max output difference %.2f\n",
         discriminating, biggest);
  return true;
}

static int failures_total = 0;

// Tolerance, and why it is not a flat constant.
//
// The fused path stores its output as bf16, which carries 8 mantissa bits, so
// the storage step alone costs up to half an ULP = 2^-9 ~= 2.0e-3 relative --
// and these outputs are weighted averages of token ids in the 17..65 range, so
// a single rounding is a large absolute number (0.25 at magnitude 34). An
// initial flat 5e-3 flagged those roundings as failures: the reported errors
// were 0.72-0.86 ULP of bf16, and the SPLIT path -- identical arithmetic,
// fp32 output -- passed the same cases at 3e-3. That is the signature of output
// quantization, not a mapping bug.
//
// So the fused bound is derived rather than tuned: half an ULP for the store,
// plus the ~3.3e-3 the split path already demonstrates for the accumulation
// itself (bf16 P and V operands into an fp32 accumulator), plus a little
// headroom. It is NOT loose enough to hide the bugs this test is about: the
// mask-convention error this file exists to catch moves outputs by up to 60.1
// absolute (see assert_delta_matters), and the fully-masked-tile bug this test
// actually found showed up at 7.5e-2 and 3.2e16. Those are 1-15 orders of
// magnitude above this bound.
//
// The split path keeps the tighter fp32 bound; it has no bf16 store.
static double tol_for(double want) {
  (void)want;
  return 8e-3;
}
static constexpr double TOL_SPLIT = 5e-3;

template <int NUM_KV_CHUNKS>
static void run_case(Case const &c, int kv_head_idx) {
  int const npages = (c.seqlen_k + PAGE_SIZE - 1) / PAGE_SIZE;
  int const last_page_len = c.seqlen_k - (npages - 1) * PAGE_SIZE;
  int const delta = c.seqlen_k - c.seqlen_q;

  // Non-identity page table, as in the decode test: an implementation that
  // ignored kv_indices and indexed the cache directly reads the wrong tokens.
  std::vector<int> h_indices(npages);
  int max_pid = 0;
  for (int i = 0; i < npages; ++i) {
    h_indices[i] = 2 * i + 3;
    max_pid = std::max(max_pid, h_indices[i]);
  }
  int const total_pages = max_pid + 1;

  // Q workspace holds seqlen_q token rows.
  std::vector<bf16> h_q((size_t)c.seqlen_q * Q_STRIDE, __float2bfloat16(0.f));
  for (int t = 0; t < c.seqlen_q; ++t) {
    for (int h = 0; h < NUM_QO_PER_KV; ++h) {
      for (int d = 0; d < HEAD_DIM; ++d) {
        h_q[(size_t)t * Q_STRIDE +
            (kv_head_idx * NUM_QO_PER_KV + h) * HEAD_DIM + d] =
            __float2bfloat16(q_val(t, h, d));
      }
    }
  }

  size_t const kv_elems = (size_t)total_pages * PAGE_SIZE * KV_STRIDE;
  std::vector<bf16> h_k(kv_elems, __float2bfloat16(0.f));
  std::vector<bf16> h_v(kv_elems, __float2bfloat16(0.f));
  for (int t = 0; t < c.seqlen_k; ++t) {
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

  int const h_qo_indptr[2] = {0, c.seqlen_q};
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

  constexpr int LSE_STRIDE = NUM_KV_HEADS * NUM_KV_CHUNKS * NUM_QO_PER_KV;
  constexpr int O_ACC_STRIDE = LSE_STRIDE * HEAD_DIM;
  size_t const o_elems = (NUM_KV_CHUNKS == 1)
                             ? (size_t)c.seqlen_q * Q_STRIDE
                             : (size_t)c.seqlen_q * O_ACC_STRIDE;
  size_t const o_bytes = o_elems * ((NUM_KV_CHUNKS == 1) ? 2 : 4);
  size_t const lse_elems = (size_t)c.seqlen_q * LSE_STRIDE;
  void *d_o, *d_lse;
  hipMalloc(&d_o, o_bytes);
  hipMalloc(&d_lse, lse_elems * 4);
  hipMemset(d_o, 0xCD, o_bytes);
  hipMemset(d_lse, 0xCD, lse_elems * 4);

  // K/V arrive ALREADY OFFSET by kv_head_idx * HEAD_DIM (the
  // gang_attention_mi300.cuh convention the kernel relies on).
  bf16 *d_k_head = reinterpret_cast<bf16 *>(d_k) + kv_head_idx * HEAD_DIM;
  bf16 *d_v_head = reinterpret_cast<bf16 *>(d_v) + kv_head_idx * HEAD_DIM;

  for (int chunk = 0; chunk < NUM_KV_CHUNKS; ++chunk) {
    hipLaunchKernelGGL((prefill_probe<NUM_KV_CHUNKS>), dim3(1), dim3(NTHREADS),
                       0, 0, d_q, d_k_head, d_v_head, d_o, d_lse, d_qo, d_kvp,
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
  std::vector<float> h_lse(lse_elems);
  hipMemcpy(h_o.data(), d_o, o_bytes, hipMemcpyDeviceToHost);
  hipMemcpy(h_lse.data(), d_lse, lse_elems * 4, hipMemcpyDeviceToHost);

  int local_fail = 0, poison = 0;
  double worst = 0.0, worst_lse = 0.0;

  for (int q = 0; q < c.seqlen_q; ++q) {
    for (int h = 0; h < NUM_QO_PER_KV; ++h) {
      std::vector<double> ref;
      double ref_lse;
      ref_attend(q, h, c, /*bottom_right=*/true, ref, ref_lse);

      // Sink correction, applied by the kernel only on the fused path.
      double corr = 1.0;
      if (NUM_KV_CHUNKS == 1 && c.with_sinks) {
        double const sink =
            (double)__bfloat162float(h_sinks[kv_head_idx * NUM_QO_PER_KV + h]);
        corr = 1.0 / (1.0 + std::exp(sink - ref_lse));
      }

      for (int d = 0; d < HEAD_DIM; ++d) {
        double got;
        if (NUM_KV_CHUNKS == 1) {
          bf16 const *op = reinterpret_cast<bf16 const *>(h_o.data());
          size_t const idx = (size_t)q * Q_STRIDE +
                             (size_t)(kv_head_idx * NUM_QO_PER_KV + h) *
                                 HEAD_DIM +
                             d;
          unsigned short raw;
          std::memcpy(&raw, &op[idx], 2);
          if (raw == 0xCDCD) {
            ++poison;
            continue;
          }
          got = (double)__bfloat162float(op[idx]);
        } else {
          // Split path: sum the per-chunk partials the way merge_splitkv does,
          // weighting by exp(LSE_chunk - LSE_global). Done here rather than
          // calling the merge kernel so the reference stays independent.
          double num = 0.0, den = 0.0, mg = -INFINITY;
          for (int ch = 0; ch < NUM_KV_CHUNKS; ++ch) {
            float const l = h_lse[(size_t)q * LSE_STRIDE +
                                  kv_head_idx * NUM_KV_CHUNKS * NUM_QO_PER_KV +
                                  ch * NUM_QO_PER_KV + h];
            if (l > -1e29f) {
              mg = std::max(mg, (double)l);
            }
          }
          for (int ch = 0; ch < NUM_KV_CHUNKS; ++ch) {
            float const l = h_lse[(size_t)q * LSE_STRIDE +
                                  kv_head_idx * NUM_KV_CHUNKS * NUM_QO_PER_KV +
                                  ch * NUM_QO_PER_KV + h];
            if (l <= -1e29f) {
              continue;
            }
            float const *op = reinterpret_cast<float const *>(h_o.data());
            size_t const idx =
                (size_t)q * O_ACC_STRIDE +
                (size_t)kv_head_idx * NUM_KV_CHUNKS * NUM_QO_PER_KV * HEAD_DIM +
                (size_t)ch * NUM_QO_PER_KV * HEAD_DIM + (size_t)h * HEAD_DIM +
                d;
            double const w = std::exp((double)l - mg);
            num += w * (double)op[idx];
            den += w;
          }
          got = (den > 0.0) ? (num / den) : 0.0;
        }

        double const want = ref[d] * corr;
        double const rel =
            std::fabs(got - want) / std::max(1.0, std::fabs(want));
        worst = std::max(worst, rel);
        double const tol = (NUM_KV_CHUNKS == 1) ? tol_for(want) : TOL_SPLIT;
        if (rel > tol) {
          if (local_fail < 4) {
            printf("  [%s] q=%d h=%d d=%d got %.5f want %.5f (rel %.4f)\n",
                   c.name, q, h, d, got, want, rel);
          }
          ++local_fail;
        }
      }

      // LSE is checked only on the fused path; on the split path it is an
      // input to the merge above and is therefore already exercised.
      if (NUM_KV_CHUNKS == 1) {
        float const got_lse =
            h_lse[(size_t)q * LSE_STRIDE + kv_head_idx * NUM_QO_PER_KV + h];
        double const dl = std::fabs((double)got_lse - ref_lse);
        worst_lse = std::max(worst_lse, dl);
        if (dl > 1e-2) {
          if (local_fail < 4) {
            printf("  [%s] q=%d h=%d LSE got %.5f want %.5f\n", c.name, q, h,
                   got_lse, ref_lse);
          }
          ++local_fail;
        }
      }
    }
  }

  if (poison) {
    printf("  [%s] %d output elements were never written (still 0xCDCD)\n",
           c.name, poison);
    local_fail += poison;
  }

  printf("%-34s chunks=%d dq=%3d dk=%3d delta=%3d %s worst_rel=%.2e "
         "worst_lse=%.2e\n",
         c.name, NUM_KV_CHUNKS, c.seqlen_q, c.seqlen_k, delta,
         local_fail ? "FAIL" : "ok  ", worst, worst_lse);
  if (local_fail) {
    ++failures_total;
  }

  hipFree(d_q); hipFree(d_k); hipFree(d_v); hipFree(d_sinks);
  hipFree(d_qo); hipFree(d_kvp); hipFree(d_idx); hipFree(d_lp);
  hipFree(d_o); hipFree(d_lse);
}

int main() {
  std::vector<Case> cases = {
      // delta == 0: fresh prefill of an empty cache. These cannot distinguish
      // the mask convention -- kept because they are the common case and
      // exercise the ragged tail and the Q-tile loop.
      {"fresh, q=16 k=16 (1 q-tile)", 16, 16, 0, false},
      {"fresh, q=40 k=40 (ragged q)", 40, 40, 0, false},
      {"fresh, q=7  k=7  (sub-tile)", 7, 7, 0, false},
      // delta > 0: chunked prefill appending to a populated cache. THESE are
      // the cases that fail under a top-left mask.
      {"chunked, q=16 k=48 (delta=32)", 16, 48, 0, false},
      {"chunked, q=8  k=53 (ragged k)", 8, 53, 0, false},
      {"chunked, q=33 k=97 (multi-tile)", 33, 97, 0, false},
      // Sliding window: exercises the lower bound and, at delta>0, an interior
      // tile that is fully masked for every row of a query tile.
      {"window=32, q=16 k=80", 16, 80, 32, false},
      {"window=48, q=24 k=96", 24, 96, 48, false},
      // Sinks, fused path only.
      {"sinks, q=16 k=48", 16, 48, 0, true},
      {"sinks+window, q=12 k=64", 12, 64, 32, true},
  };

  if (!assert_delta_matters(cases)) {
    printf("OVERALL: FAIL\n");
    return 1;
  }

  printf("\n-- fused path (NUM_KV_CHUNKS=1) --\n");
  for (Case const &c : cases) {
    run_case<1>(c, /*kv_head_idx=*/1);
  }

  // Split-KV. Sinks are not applied on this path (the merge does it), so only
  // the sink-free cases are meaningful here.
  printf("\n-- split path (NUM_KV_CHUNKS=4) --\n");
  for (Case const &c : cases) {
    if (c.with_sinks) {
      continue;
    }
    run_case<4>(c, /*kv_head_idx=*/1);
  }

  printf("\nOVERALL: %s\n", failures_total ? "FAIL" : "PASS");
  return failures_total ? 1 : 0;
}
