// Correctness for kv_cache_update_mi450 -- phase A of the attention path.
//
// What this kernel does: reads Q/K/V out of a fused QKV buffer, applies RMS
// (QK) norm and RoPE to Q and to the new K tokens, writes the processed Q to a
// workspace the decode kernel reads, and scatters the processed K and the raw V
// into the paged cache.
//
// WHY IT NEEDS A TEST OF ITS OWN, GIVEN IT COMPILES
//
// It compiles is worth nothing here. A HIP template that is never instantiated
// is barely parsed -- the port could have had a wrong reduction width, a wrong
// scratch size, or a dangling reference and still produced a clean build. So
// this test instantiates it at a real shape and runs it.
//
// WHAT CAN GO WRONG SILENTLY, AND HOW EACH IS CAUGHT
//
//  1. Wave-width reduction. The whole reason this file is a port rather than a
//     shared include is the RMS norm's cross-wave reduce: mi300 spells the wave
//     width as a literal 64 in six places. At wave32 those do not fault, they
//     reduce over the wrong lane set -- and crucially, `threadIdx.x % 64 == 0`
//     elects only one writer per TWO waves, so half of s_reduce is never
//     written and is read back as stale LDS. The result is a plausible norm.
//     Caught by: the norm is a per-head scalar, so a partial sum is a *wrong
//     scale on every element of that head at once*. The reference computes it
//     in double from the same inputs and the tolerance is tight enough that a
//     half-sum (rms off by sqrt(2)) is four decades outside.
//
//  2. Paged scatter. The K/V destination is
//     page_indices[pos/PAGE_SIZE] * PAGE_SIZE + pos%PAGE_SIZE, which is the
//     one piece of arithmetic where an off-by-one lands in a *valid* buffer.
//     Caught by: a deliberately non-identity page table (reversed), so writing
//     to the linear position instead of the mapped one lands in the wrong page
//     and is detected; plus every unwritten cache slot is poisoned and asserted
//     to still hold poison, so a scatter that writes the right values to the
//     wrong place fails twice -- once as a mismatch, once as a clobbered slot.
//
//  3. RoPE half-rotation. The pairing is (i, i + HEAD_DIM/2), not adjacent
//     pairs. Both produce finite output. Caught by: cos/sin tables that are
//     distinct per (pos, i) and inputs distinct per dim, so pairing i with i+1
//     moves every value.
//
//  4. Q vs K crosstalk. Q and K live at different offsets in the same fused
//     QKV row and use *different* norm weights and eps. Caught by: q_norm and
//     k_norm weights are different, and q_eps != k_eps.
//
//  5. V must NOT be normed or rotated. It is copied raw. Caught by: the
//     reference copies it raw, and V's values are in a range where a norm
//     would visibly change them. Compared with ==, not a tolerance: a copy
//     that is merely close is a copy that is wrong.
//
//  6. NaN. Every value is checked finite BEFORE the tolerance comparison --
//     see bad() below. Without that, a kernel producing NaN everywhere passes
//     with a reported worst error of 0.0, because comparisons against NaN are
//     always false. This is not a hypothetical: it is how mutant m1 (the
//     wave64 restoration) initially survived this very test.
//
// The reference is written from the algorithm description above, not by
// transcribing the kernel -- an independent implementation, so a shared
// misunderstanding has to be made twice.
//
// Build:
//   /tmp/p1/cctest450y.sh tests/mi450/test_kv_cache_update.hip -o <bin>
// Run:
//   ./run_ffm.sh <abs path to bin>

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ck_tile::bf16_t and type_convert, which the kernel uses throughout.
#include "ck_tile/core.hpp"

#include "mirage/persistent_kernel/tasks/mi450/kv_cache_update_mi450.cuh"

using bf16 = ck_tile::bf16_t;

// ── shape ───────────────────────────────────────────────────────────────────
// Deliberately small so the host reference is cheap, but every structural
// feature of the real shape is present: multiple Q heads per KV head, more than
// one page, and a last page that is only partly full.
static constexpr int HEAD_DIM = 64;
static constexpr int NUM_KV_HEADS = 2;
static constexpr int NUM_QO_HEADS = 8;
static constexpr int NUM_QO_PER_KV = NUM_QO_HEADS / NUM_KV_HEADS; // 4
static constexpr int PAGE_SIZE = 16;
static constexpr int MAX_SEQ_LEN = 256;
static constexpr int MAX_TOKENS = 4;
static constexpr int NUM_THREADS = 256;

// The kernel is invoked per (request, kv_head) and indexes the fused QKV row as
// [ Q(NUM_QO_PER_KV * HEAD_DIM) | K(HEAD_DIM) | V(HEAD_DIM) ] for THIS kv head.
// QKV_STRIDE is the full row across all kv heads.
static constexpr int PER_KV_BLOCK = (NUM_QO_PER_KV + 2) * HEAD_DIM;
static constexpr int QKV_STRIDE = NUM_KV_HEADS * PER_KV_BLOCK;
static constexpr int KV_CACHE_STRIDE = NUM_KV_HEADS * HEAD_DIM;
static constexpr int Q_WORKSPACE_STRIDE = NUM_QO_PER_KV * HEAD_DIM;

static constexpr int NUM_PAGES_TOTAL = 8;

static float const Q_EPS = 1e-6f;
static float const K_EPS = 1e-5f; // deliberately different from Q_EPS

// ── inputs ──────────────────────────────────────────────────────────────────
// A mixing hash, not an affine function of the index. Affine data made two real
// cross-lane bugs undetectable in the attention test (see
// test_attention_wmma.hip); the same trap applies to any per-dim mapping error
// here.
static unsigned mix(unsigned x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}
// Values are multiples of 1/8 in [0,1) plus a nonzero floor, so bf16 holds them
// exactly and the only inexactness in the whole pipeline is the rsqrt and the
// bf16 rounding of the *result*.
static float hval(unsigned seed) {
  return 0.125f * (float)(mix(seed) % 8) + 0.125f;
}
static float q_in(int tok, int kvh, int h, int d) {
  return hval((unsigned)(tok * 7919 + kvh * 613 + h * 131 + d * 7 + 1));
}
static float k_in(int tok, int kvh, int d) {
  return hval((unsigned)(tok * 5011 + kvh * 271 + d * 13 + 3));
}
static float v_in(int tok, int kvh, int d) {
  return hval((unsigned)(tok * 3301 + kvh * 449 + d * 17 + 5));
}
static float qw(int d) {
  return hval((unsigned)(d * 29 + 11));
}
static float kw(int d) {
  return hval((unsigned)(d * 37 + 23)); // different from qw
}
// cos/sin are independent per (pos, i) -- NOT an actual rotary table. Using a
// real cos/sin pair would make cos^2+sin^2==1, and a kernel that swapped them
// could survive on symmetric data. These are just two distinct tables.
static float cos_t(int pos, int i) {
  return hval((unsigned)(pos * 8191 + i * 41 + 7));
}
static float sin_t(int pos, int i) {
  return hval((unsigned)(pos * 6337 + i * 53 + 19));
}

static bf16 tobf(float f) {
  return ck_tile::type_convert<bf16>(f);
}
static float tof(bf16 b) {
  return ck_tile::type_convert<float>(b);
}

// bf16 round-trip, used by the reference wherever the kernel stores to a bf16
// buffer and reads it back. Getting this wrong would make the reference
// disagree in the last bits everywhere and force a loose tolerance, which is
// exactly how a real error hides.
static float rt(float f) {
  return tof(tobf(f));
}

// ── device wrapper ──────────────────────────────────────────────────────────
// Body guarded like the other mi450 tests: the host compilation pass still has
// to see the __global__ to emit its stub.
__global__ __launch_bounds__(NUM_THREADS) void kvupd_probe(
    void const *qkv,
    void *k_cache,
    void *v_cache,
    void *q_ws,
    int const *qo_indptr,
    int const *kv_indptr,
    int const *kv_indices,
    int const *kv_last_page_len,
    int request_id,
    bool qk_norm,
    bool rope,
    void const *qnw,
    void const *knw,
    void const *cosp,
    void const *sinp) {
#if defined(MIRAGE_ARCH_GFX1250)
  // The kernel derives its per-kv-head slice from the pointers it is given, so
  // the caller offsets qkv by the kv head's block and the cache by the head's
  // lane. This mirrors what the task dispatcher does with blockIdx.y.
  int const kvh = blockIdx.x;
  bf16 const *qkv_h = reinterpret_cast<bf16 const *>(qkv) + kvh * PER_KV_BLOCK;
  bf16 *kc_h = reinterpret_cast<bf16 *>(k_cache) + kvh * HEAD_DIM;
  bf16 *vc_h = reinterpret_cast<bf16 *>(v_cache) + kvh * HEAD_DIM;
  bf16 *qws_h = reinterpret_cast<bf16 *>(q_ws) +
                kvh * (size_t)MAX_TOKENS * Q_WORKSPACE_STRIDE;

  kernel::mi450::kv_cache_update_impl<bf16,
                                      NUM_QO_HEADS,
                                      NUM_KV_HEADS,
                                      NUM_QO_PER_KV,
                                      KV_CACHE_STRIDE,
                                      QKV_STRIDE,
                                      HEAD_DIM,
                                      MAX_SEQ_LEN,
                                      PAGE_SIZE,
                                      MAX_TOKENS,
                                      Q_WORKSPACE_STRIDE>(
      qkv_h, kc_h, vc_h, qws_h, qo_indptr, kv_indptr, kv_indices,
      kv_last_page_len, (int16_t)request_id, qk_norm, rope, qnw, knw, cosp,
      sinp, Q_EPS, K_EPS);
#else
  (void)qkv; (void)k_cache; (void)v_cache; (void)q_ws; (void)qo_indptr;
  (void)kv_indptr; (void)kv_indices; (void)kv_last_page_len; (void)request_id;
  (void)qk_norm; (void)rope; (void)qnw; (void)knw; (void)cosp; (void)sinp;
#endif
}

#define CHECK(cmd)                                                             \
  do {                                                                         \
    hipError_t _e = (cmd);                                                     \
    if (_e != hipSuccess) {                                                    \
      printf("FAIL %s:%d %s -> %s\n", __FILE__, __LINE__, #cmd,                \
             hipGetErrorString(_e));                                           \
      return 1;                                                                \
    }                                                                          \
  } while (0)

// Poison: a bf16 bit pattern that is not a plausible output. 0x7FC1 is a quiet
// NaN with a distinctive payload, so an untouched slot is both visibly wrong
// and cannot compare equal to anything by accident.
static uint16_t const POISON_BITS = 0x7FC1;

static bool is_poison(bf16 b) {
  uint16_t u;
  std::memcpy(&u, &b, sizeof(u));
  return u == POISON_BITS;
}

// EVERY value must be checked finite BEFORE any tolerance comparison.
//
// This is not defensive boilerplate, it is the single most important line in
// the file. A relative-error test of the form `if (rel > TOL) fail` cannot
// detect a NaN, because every comparison involving NaN evaluates false -- so a
// kernel that produces NaN everywhere passes with a reported worst error of
// exactly 0.0, which reads like an unusually good result rather than a broken
// one.
//
// That is not hypothetical. The wave64 mutant (the exact bug this port exists
// to prevent: restoring `WAVE = 64`, `threadIdx.x % 64 == 0`, and the width-64
// butterfly) made the RMS reduction read stale LDS, giving a sum-of-squares of
// -1.7e9; rsqrt of a negative is NaN; every Q and K element came out NaN; and
// the test printed "worst rel: Q 0 K 0 V 0" and PASSED. It was killed only
// after this check was added. A poison check does not cover it either -- NaN
// from arithmetic has different bits from the poison pattern, so the slot
// looks legitimately written.
static bool bad(double x) {
  return !std::isfinite(x);
}

struct Case {
  char const *name;
  int num_tokens;   // new tokens this step
  int global_len;   // total sequence length after this step
  bool qk_norm;
  bool rope;
};

static int run_case(Case const &c) {
  printf("\n=== %s (tokens=%d seq=%d qk_norm=%d rope=%d) ===\n", c.name,
         c.num_tokens, c.global_len, (int)c.qk_norm, (int)c.rope);

  int const num_pages = (c.global_len + PAGE_SIZE - 1) / PAGE_SIZE;
  int const last_page_len = c.global_len - (num_pages - 1) * PAGE_SIZE;
  if (num_pages > NUM_PAGES_TOTAL) {
    printf("FAIL: case needs %d pages, only %d allocated\n", num_pages,
           NUM_PAGES_TOTAL);
    return 1;
  }

  // Non-identity page table: logical page i lives at physical page
  // (NUM_PAGES_TOTAL-1-i). If the kernel ignored the indirection and used the
  // linear position it would write into the wrong physical page -- which is
  // still inside the allocation, so only the table makes it detectable.
  std::vector<int> h_indices(num_pages);
  for (int i = 0; i < num_pages; ++i) {
    h_indices[i] = NUM_PAGES_TOTAL - 1 - i;
  }
  int const h_qo[2] = {0, c.num_tokens};
  int const h_kvptr[2] = {0, num_pages};
  int const h_lastlen[1] = {last_page_len};

  // ── host buffers ─────────────────────────────────────────────────────────
  size_t const qkv_elems = (size_t)c.num_tokens * QKV_STRIDE;
  std::vector<bf16> h_qkv(qkv_elems);
  for (int t = 0; t < c.num_tokens; ++t) {
    for (int kvh = 0; kvh < NUM_KV_HEADS; ++kvh) {
      bf16 *row = &h_qkv[(size_t)t * QKV_STRIDE + kvh * PER_KV_BLOCK];
      for (int h = 0; h < NUM_QO_PER_KV; ++h) {
        for (int d = 0; d < HEAD_DIM; ++d) {
          row[h * HEAD_DIM + d] = tobf(q_in(t, kvh, h, d));
        }
      }
      for (int d = 0; d < HEAD_DIM; ++d) {
        row[NUM_QO_PER_KV * HEAD_DIM + d] = tobf(k_in(t, kvh, d));
        row[(NUM_QO_PER_KV + 1) * HEAD_DIM + d] = tobf(v_in(t, kvh, d));
      }
    }
  }

  std::vector<bf16> h_qnw(HEAD_DIM), h_knw(HEAD_DIM);
  for (int d = 0; d < HEAD_DIM; ++d) {
    h_qnw[d] = tobf(qw(d));
    h_knw[d] = tobf(kw(d));
  }
  std::vector<bf16> h_cos((size_t)MAX_SEQ_LEN * HEAD_DIM);
  std::vector<bf16> h_sin((size_t)MAX_SEQ_LEN * HEAD_DIM);
  for (int p = 0; p < MAX_SEQ_LEN; ++p) {
    for (int i = 0; i < HEAD_DIM; ++i) {
      h_cos[(size_t)p * HEAD_DIM + i] = tobf(cos_t(p, i));
      h_sin[(size_t)p * HEAD_DIM + i] = tobf(sin_t(p, i));
    }
  }

  size_t const cache_elems =
      (size_t)NUM_PAGES_TOTAL * PAGE_SIZE * KV_CACHE_STRIDE;
  size_t const qws_elems =
      (size_t)NUM_KV_HEADS * MAX_TOKENS * Q_WORKSPACE_STRIDE;

  // ── device buffers, all poisoned ─────────────────────────────────────────
  bf16 *d_qkv, *d_kc, *d_vc, *d_qws, *d_qnw, *d_knw, *d_cos, *d_sin;
  int *d_qo, *d_kvptr, *d_kvidx, *d_lastlen;
  CHECK(hipMalloc(&d_qkv, qkv_elems * sizeof(bf16)));
  CHECK(hipMalloc(&d_kc, cache_elems * sizeof(bf16)));
  CHECK(hipMalloc(&d_vc, cache_elems * sizeof(bf16)));
  CHECK(hipMalloc(&d_qws, qws_elems * sizeof(bf16)));
  CHECK(hipMalloc(&d_qnw, HEAD_DIM * sizeof(bf16)));
  CHECK(hipMalloc(&d_knw, HEAD_DIM * sizeof(bf16)));
  CHECK(hipMalloc(&d_cos, h_cos.size() * sizeof(bf16)));
  CHECK(hipMalloc(&d_sin, h_sin.size() * sizeof(bf16)));
  CHECK(hipMalloc(&d_qo, 2 * sizeof(int)));
  CHECK(hipMalloc(&d_kvptr, 2 * sizeof(int)));
  CHECK(hipMalloc(&d_kvidx, num_pages * sizeof(int)));
  CHECK(hipMalloc(&d_lastlen, sizeof(int)));

  {
    std::vector<bf16> pois_c(cache_elems), pois_w(qws_elems);
    bf16 p;
    std::memcpy(&p, &POISON_BITS, sizeof(p));
    for (auto &x : pois_c) { x = p; }
    for (auto &x : pois_w) { x = p; }
    CHECK(hipMemcpy(d_kc, pois_c.data(), cache_elems * sizeof(bf16),
                    hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_vc, pois_c.data(), cache_elems * sizeof(bf16),
                    hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_qws, pois_w.data(), qws_elems * sizeof(bf16),
                    hipMemcpyHostToDevice));
  }
  CHECK(hipMemcpy(d_qkv, h_qkv.data(), qkv_elems * sizeof(bf16),
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_qnw, h_qnw.data(), HEAD_DIM * sizeof(bf16),
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_knw, h_knw.data(), HEAD_DIM * sizeof(bf16),
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_cos, h_cos.data(), h_cos.size() * sizeof(bf16),
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_sin, h_sin.data(), h_sin.size() * sizeof(bf16),
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_qo, h_qo, sizeof(h_qo), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_kvptr, h_kvptr, sizeof(h_kvptr), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_kvidx, h_indices.data(), num_pages * sizeof(int),
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_lastlen, h_lastlen, sizeof(h_lastlen),
                  hipMemcpyHostToDevice));

  // Dynamic LDS: s_q + s_k + reduction scratch, matching the kernel's layout.
  // NUM_WAVES is 8 at wave32 (256/32), so this is larger than the mi300 value.
  size_t const s_q_size = sizeof(bf16) * MAX_TOKENS * NUM_QO_PER_KV * HEAD_DIM;
  size_t const s_k_size = sizeof(bf16) * MAX_TOKENS * HEAD_DIM;
  size_t const s_red_off = ((s_q_size + s_k_size + 15) & ~(size_t)15);
  size_t const smem = s_red_off + sizeof(float) * (NUM_THREADS / 32);

  hipLaunchKernelGGL(kvupd_probe, dim3(NUM_KV_HEADS), dim3(NUM_THREADS), smem,
                     0, d_qkv, d_kc, d_vc, d_qws, d_qo, d_kvptr, d_kvidx,
                     d_lastlen, 0, c.qk_norm, c.rope, d_qnw, d_knw, d_cos,
                     d_sin);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());

  std::vector<bf16> o_kc(cache_elems), o_vc(cache_elems), o_qws(qws_elems);
  CHECK(hipMemcpy(o_kc.data(), d_kc, cache_elems * sizeof(bf16),
                  hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(o_vc.data(), d_vc, cache_elems * sizeof(bf16),
                  hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(o_qws.data(), d_qws, qws_elems * sizeof(bf16),
                  hipMemcpyDeviceToHost));

  // ── independent host reference ───────────────────────────────────────────
  // Written from the algorithm, in double, with an explicit bf16 round-trip at
  // each point where the kernel stores to LDS or memory as bf16.
  int fail = 0;
  double worst_q = 0.0, worst_k = 0.0, worst_v = 0.0;
  // bf16 has 8 mantissa bits: one rounding is 2^-8 relative. Allow a handful of
  // roundings (norm, weight multiply, RoPE's two products and a sum) plus the
  // f32 rsqrt. A half-sum reduction error is ~41% relative -- two decades out.
  double const TOL = 2e-2;

  for (int kvh = 0; kvh < NUM_KV_HEADS; ++kvh) {
    for (int t = 0; t < c.num_tokens; ++t) {
      int const pos = c.global_len - c.num_tokens + t;

      // --- Q ---
      for (int h = 0; h < NUM_QO_PER_KV; ++h) {
        double ref[HEAD_DIM];
        for (int d = 0; d < HEAD_DIM; ++d) {
          ref[d] = rt(q_in(t, kvh, h, d));
        }
        if (c.qk_norm) {
          double ss = 0.0;
          for (int d = 0; d < HEAD_DIM; ++d) {
            ss += ref[d] * ref[d];
          }
          double const rms = 1.0 / std::sqrt(ss / (double)HEAD_DIM + Q_EPS);
          for (int d = 0; d < HEAD_DIM; ++d) {
            ref[d] = rt(ref[d] * rms * rt(qw(d)));
          }
        }
        if (c.rope) {
          double rot[HEAD_DIM];
          for (int i = 0; i < HEAD_DIM / 2; ++i) {
            double const cc = rt(cos_t(pos, i)), sv = rt(sin_t(pos, i));
            double const v0 = ref[i], v1 = ref[i + HEAD_DIM / 2];
            rot[i] = rt(v0 * cc - v1 * sv);
            rot[i + HEAD_DIM / 2] = rt(v0 * sv + v1 * cc);
          }
          for (int d = 0; d < HEAD_DIM; ++d) {
            ref[d] = rot[d];
          }
        }
        for (int d = 0; d < HEAD_DIM; ++d) {
          size_t idx = (size_t)kvh * MAX_TOKENS * Q_WORKSPACE_STRIDE +
                       (size_t)t * Q_WORKSPACE_STRIDE + h * HEAD_DIM + d;
          if (is_poison(o_qws[idx])) {
            printf("FAIL: q_workspace kvh=%d t=%d h=%d d=%d never written "
                   "(still poison)\n", kvh, t, h, d);
            fail = 1;
            goto done;
          }
          double got = tof(o_qws[idx]);
          if (bad(got) || bad(ref[d])) {
            printf("FAIL: Q kvh=%d t=%d h=%d d=%d is not finite (got %f, want "
                   "%f) -- see the bad() comment: a NaN passes every relative "
                   "error check\n", kvh, t, h, d, got, ref[d]);
            fail = 1;
            goto done;
          }
          double den = std::fabs(ref[d]) > 1e-3 ? std::fabs(ref[d]) : 1e-3;
          double rel = std::fabs(got - ref[d]) / den;
          if (rel > worst_q) { worst_q = rel; }
          if (rel > TOL) {
            printf("FAIL: Q kvh=%d t=%d h=%d d=%d got %.6f want %.6f "
                   "(rel %.4g)\n", kvh, t, h, d, got, ref[d], rel);
            fail = 1;
            goto done;
          }
        }
      }

      // --- K (normed/rotated, scattered through the page table) ---
      {
        double ref[HEAD_DIM];
        for (int d = 0; d < HEAD_DIM; ++d) {
          ref[d] = rt(k_in(t, kvh, d));
        }
        if (c.qk_norm) {
          double ss = 0.0;
          for (int d = 0; d < HEAD_DIM; ++d) {
            ss += ref[d] * ref[d];
          }
          double const rms = 1.0 / std::sqrt(ss / (double)HEAD_DIM + K_EPS);
          for (int d = 0; d < HEAD_DIM; ++d) {
            ref[d] = rt(ref[d] * rms * rt(kw(d)));
          }
        }
        if (c.rope) {
          double rot[HEAD_DIM];
          for (int i = 0; i < HEAD_DIM / 2; ++i) {
            double const cc = rt(cos_t(pos, i)), sv = rt(sin_t(pos, i));
            double const v0 = ref[i], v1 = ref[i + HEAD_DIM / 2];
            rot[i] = rt(v0 * cc - v1 * sv);
            rot[i + HEAD_DIM / 2] = rt(v0 * sv + v1 * cc);
          }
          for (int d = 0; d < HEAD_DIM; ++d) {
            ref[d] = rot[d];
          }
        }
        int const phys = h_indices[pos / PAGE_SIZE] * PAGE_SIZE +
                         (pos % PAGE_SIZE);
        for (int d = 0; d < HEAD_DIM; ++d) {
          size_t idx = (size_t)phys * KV_CACHE_STRIDE + kvh * HEAD_DIM + d;
          if (is_poison(o_kc[idx])) {
            printf("FAIL: k_cache phys=%d kvh=%d d=%d never written\n", phys,
                   kvh, d);
            fail = 1;
            goto done;
          }
          double got = tof(o_kc[idx]);
          if (bad(got) || bad(ref[d])) {
            printf("FAIL: K phys=%d kvh=%d d=%d is not finite (got %f, want "
                   "%f)\n", phys, kvh, d, got, ref[d]);
            fail = 1;
            goto done;
          }
          double den = std::fabs(ref[d]) > 1e-3 ? std::fabs(ref[d]) : 1e-3;
          double rel = std::fabs(got - ref[d]) / den;
          if (rel > worst_k) { worst_k = rel; }
          if (rel > TOL) {
            printf("FAIL: K phys=%d kvh=%d d=%d got %.6f want %.6f (rel %.4g)\n",
                   phys, kvh, d, got, ref[d], rel);
            fail = 1;
            goto done;
          }
        }

        // --- V: raw copy, no norm, no RoPE ---
        for (int d = 0; d < HEAD_DIM; ++d) {
          size_t idx = (size_t)phys * KV_CACHE_STRIDE + kvh * HEAD_DIM + d;
          if (is_poison(o_vc[idx])) {
            printf("FAIL: v_cache phys=%d kvh=%d d=%d never written\n", phys,
                   kvh, d);
            fail = 1;
            goto done;
          }
          double want = rt(v_in(t, kvh, d));
          double got = tof(o_vc[idx]);
          double rel = std::fabs(got - want) / (std::fabs(want) + 1e-9);
          if (rel > worst_v) { worst_v = rel; }
          // V is a pure copy: it must be EXACT, not merely close. A tolerance
          // here would let a stray norm through.
          if (got != want) {
            printf("FAIL: V phys=%d kvh=%d d=%d got %.6f want %.6f -- V must "
                   "be copied unmodified\n", phys, kvh, d, got, want);
            fail = 1;
            goto done;
          }
        }
      }
    }
  }

  // Every cache slot NOT belonging to a new token must still be poison. This is
  // what turns "wrote the right values somewhere" into "wrote them in the right
  // place": a scatter that ignored the page table passes the value checks above
  // only if it happened to land correctly, and fails here otherwise.
  {
    std::vector<char> expect_written(NUM_PAGES_TOTAL * PAGE_SIZE, 0);
    for (int t = 0; t < c.num_tokens; ++t) {
      int const pos = c.global_len - c.num_tokens + t;
      expect_written[h_indices[pos / PAGE_SIZE] * PAGE_SIZE +
                     (pos % PAGE_SIZE)] = 1;
    }
    for (int slot = 0; slot < NUM_PAGES_TOTAL * PAGE_SIZE; ++slot) {
      if (expect_written[slot]) {
        continue;
      }
      for (int e = 0; e < KV_CACHE_STRIDE; ++e) {
        size_t idx = (size_t)slot * KV_CACHE_STRIDE + e;
        if (!is_poison(o_kc[idx]) || !is_poison(o_vc[idx])) {
          printf("FAIL: cache slot %d elem %d was written but should not have "
                 "been -- the scatter went to the wrong page\n", slot, e);
          fail = 1;
          goto done;
        }
      }
    }
  }

  printf("worst rel: Q %.4g  K %.4g  V %.4g (tol %.4g)\n", worst_q, worst_k,
         worst_v, TOL);
  if (!fail) {
    printf("PASS: %s\n", c.name);
  }

done:
  hipFree(d_qkv); hipFree(d_kc); hipFree(d_vc); hipFree(d_qws);
  hipFree(d_qnw); hipFree(d_knw); hipFree(d_cos); hipFree(d_sin);
  hipFree(d_qo); hipFree(d_kvptr); hipFree(d_kvidx); hipFree(d_lastlen);
  return fail;
}

int main() {
  printf("kv_cache_update_mi450: HEAD_DIM=%d KV_HEADS=%d QO_PER_KV=%d "
         "PAGE_SIZE=%d threads=%d\n",
         HEAD_DIM, NUM_KV_HEADS, NUM_QO_PER_KV, PAGE_SIZE, NUM_THREADS);

  int fail = 0;
  // Decode: one new token, sequence spans 3 pages with a partial last page --
  // so the write lands in the middle of a page, not at a boundary.
  fail |= run_case({"decode, norm+rope", 1, 40, true, true});
  // The norm alone, so a RoPE bug cannot mask a reduction bug or vice versa.
  fail |= run_case({"decode, norm only", 1, 40, true, false});
  // RoPE alone.
  fail |= run_case({"decode, rope only", 1, 40, false, true});
  // Multi-token, and deliberately spanning a page boundary: seq 33 with 4 new
  // tokens puts positions 29,30,31 on logical page 1 and 32 on page 2, so the
  // per-token page lookup has to change mid-loop.
  fail |= run_case({"multi-token across a page boundary", 4, 33, true, true});
  // Exactly full last page: last_page_len == PAGE_SIZE, the boundary case where
  // an off-by-one in num_pages/last_page_len shows up.
  fail |= run_case({"aligned end of page", 1, 32, true, true});

  printf("\nOVERALL: %s\n", fail ? "FAIL" : "PASS");
  if (!fail) {
    printf("NOTE: FFM-Lite is functional only. This pins the arithmetic and "
           "the paged scatter; it says nothing about memory ordering between "
           "this task and the attention task that consumes its output.\n");
  }
  return fail;
}
