// Correctness for the gfx1250 WMMA linear kernel (CK replacement).
//
// linear_kernel_wmma is the drop-in for linear_kernel_ck, which cannot target
// gfx1250. It is on the critical path of gang_rmsnorm_linear_bias and both MX
// orchestrators, so a silent error here would surface as "the model emits
// plausible garbage" three phases later. This test pins it against a host
// double reference.
//
// Design, carrying forward what the attention test learned:
//
//  1. Inputs are exactly representable (multiples of 1/8 in a small range) so
//     bf16 never rounds on the way in and the comparison is near-exact rather
//     than tolerance-shaped.
//  2. Indices go through a MIXING hash, not an affine expression. Affine data
//     made two real cross-lane bugs undetectable in the attention port -- see
//     the note in test_attention_wmma.hip. check_discriminating() below fails
//     the test if the data ever drifts back into a degenerate regime.
//  3. Outputs are poisoned with 0xCD and the poison is checked explicitly, so
//     an unwritten element cannot pass as a zero.
//  4. Shapes deliberately exercise the ragged cases: an output_size that is not
//     a multiple of the 128-column block pass, a batch that is not a multiple
//     of the 16-row WMMA tile, and an o_stride wider than output_size (which is
//     the real MoE case, where a worker writes a column slice of a wider
//     matrix -- getting o_stride wrong writes into the neighbouring expert).
#include <hip/hip_runtime.h>

#include "mirage/persistent_kernel/tasks/mi450/linear_wmma_mi450.cuh"
// The kernel header works in __bf16 (the compiler builtin type) and so does not
// pull in hip_bf16.h; the host side of this test needs __hip_bfloat16 and
// __float2bfloat16, so it asks for them directly rather than relying on a
// transitive include.
#include <hip/hip_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using bf16 = __hip_bfloat16;

static constexpr int NTHREADS = 256; // WORKER_NUM_THREADS: 8 waves of 32

static unsigned mix(unsigned x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}
// Values in {-0.5, -0.375, ..., 0.875}: exact in bf16, signed so cancellation
// is exercised rather than every partial sum growing monotonically.
static float a_val(int m, int k) {
  return ((float)(mix((unsigned)(m * 8191 + k * 31 + 1)) % 12) - 4.f) * 0.125f;
}
static float b_val(int n, int k) {
  return ((float)(mix((unsigned)(n * 6151 + k * 17 + 5)) % 12) - 4.f) * 0.125f;
}
static float bias_val(int n) {
  return ((float)(mix((unsigned)(n * 733 + 3)) % 9) - 4.f) * 0.25f;
}
static float res_val(int m, int n) {
  return ((float)(mix((unsigned)(m * 401 + n * 97 + 11)) % 9) - 4.f) * 0.25f;
}

template <int BATCH_SIZE, int REDUCTION_SIZE>
__global__ __launch_bounds__(NTHREADS) void linear_probe(void const *a,
                                                         void const *b,
                                                         void const *res,
                                                         void *o,
                                                         int num_active,
                                                         bool residual_add,
                                                         int output_size,
                                                         int o_stride,
                                                         void const *bias) {
#if defined(MIRAGE_ARCH_GFX1250)
  kernel::mi450::linear_kernel_wmma<bf16, BATCH_SIZE, REDUCTION_SIZE>(
      a, b, res, o, num_active, residual_add, output_size, o_stride, bias);
#endif
}

static int failures_total = 0;

// A GEMM test can be degenerate too: if every output column had the same value,
// a wrong n-mapping would be invisible, and likewise for rows. Verify both axes
// actually vary before trusting a pass.
template <int BATCH_SIZE, int REDUCTION_SIZE>
static bool check_discriminating(int output_size) {
  double c00 = 0.0;
  int col_varies = 0, row_varies = 0;
  for (int n = 0; n < std::min(output_size, 32); ++n) {
    double s = 0.0;
    for (int k = 0; k < REDUCTION_SIZE; ++k) {
      s += (double)a_val(0, k) * (double)b_val(n, k);
    }
    if (n == 0) {
      c00 = s;
    } else if (std::fabs(s - c00) > 1e-6) {
      ++col_varies;
    }
  }
  for (int m = 1; m < std::min(BATCH_SIZE, 16); ++m) {
    double s = 0.0;
    for (int k = 0; k < REDUCTION_SIZE; ++k) {
      s += (double)a_val(m, k) * (double)b_val(0, k);
    }
    if (std::fabs(s - c00) > 1e-6) {
      ++row_varies;
    }
  }
  if (col_varies == 0) {
    printf("FAIL: degenerate data -- output columns do not vary, a wrong "
           "n-mapping would be invisible.\n");
    return false;
  }
  if (row_varies == 0 && BATCH_SIZE > 1) {
    printf("FAIL: degenerate data -- output rows do not vary, a wrong "
           "m-mapping would be invisible.\n");
    return false;
  }
  return true;
}

template <int BATCH_SIZE, int REDUCTION_SIZE>
static void run_case(char const *name,
                     int output_size,
                     int o_stride,
                     int num_active,
                     bool use_bias,
                     bool residual_add) {
  if (!check_discriminating<BATCH_SIZE, REDUCTION_SIZE>(output_size)) {
    ++failures_total;
    return;
  }

  std::vector<bf16> h_a((size_t)BATCH_SIZE * REDUCTION_SIZE);
  std::vector<bf16> h_b((size_t)output_size * REDUCTION_SIZE);
  std::vector<bf16> h_bias(output_size);
  std::vector<bf16> h_res((size_t)BATCH_SIZE * o_stride,
                          __float2bfloat16(0.f));

  for (int m = 0; m < BATCH_SIZE; ++m) {
    for (int k = 0; k < REDUCTION_SIZE; ++k) {
      h_a[(size_t)m * REDUCTION_SIZE + k] = __float2bfloat16(a_val(m, k));
    }
  }
  for (int n = 0; n < output_size; ++n) {
    for (int k = 0; k < REDUCTION_SIZE; ++k) {
      h_b[(size_t)n * REDUCTION_SIZE + k] = __float2bfloat16(b_val(n, k));
    }
    h_bias[n] = __float2bfloat16(bias_val(n));
  }
  for (int m = 0; m < BATCH_SIZE; ++m) {
    for (int n = 0; n < o_stride; ++n) {
      h_res[(size_t)m * o_stride + n] = __float2bfloat16(res_val(m, n));
    }
  }

  void *d_a, *d_b, *d_bias, *d_res, *d_o;
  size_t const o_elems = (size_t)BATCH_SIZE * o_stride;
  hipMalloc(&d_a, h_a.size() * 2);
  hipMalloc(&d_b, h_b.size() * 2);
  hipMalloc(&d_bias, h_bias.size() * 2);
  hipMalloc(&d_res, h_res.size() * 2);
  hipMalloc(&d_o, o_elems * 2);
  hipMemcpy(d_a, h_a.data(), h_a.size() * 2, hipMemcpyHostToDevice);
  hipMemcpy(d_b, h_b.data(), h_b.size() * 2, hipMemcpyHostToDevice);
  hipMemcpy(d_bias, h_bias.data(), h_bias.size() * 2, hipMemcpyHostToDevice);
  hipMemcpy(d_res, h_res.data(), h_res.size() * 2, hipMemcpyHostToDevice);
  hipMemset(d_o, 0xCD, o_elems * 2);

  hipLaunchKernelGGL((linear_probe<BATCH_SIZE, REDUCTION_SIZE>), dim3(1),
                     dim3(NTHREADS), 0, 0, d_a, d_b,
                     residual_add ? d_res : nullptr, d_o, num_active,
                     residual_add, output_size, o_stride,
                     use_bias ? d_bias : nullptr);
  hipError_t err = hipDeviceSynchronize();
  if (err != hipSuccess) {
    printf("FAIL [%s]: %s\n", name, hipGetErrorString(err));
    ++failures_total;
    return;
  }

  std::vector<bf16> h_o(o_elems);
  hipMemcpy(h_o.data(), d_o, o_elems * 2, hipMemcpyDeviceToHost);

  int local_fail = 0;
  double worst = 0.0;
  for (int m = 0; m < BATCH_SIZE; ++m) {
    for (int n = 0; n < o_stride; ++n) {
      bf16 const got_raw = h_o[(size_t)m * o_stride + n];
      unsigned short raw;
      __builtin_memcpy(&raw, &got_raw, 2);
      bool const in_range = (m < num_active) && (n < output_size);

      if (!in_range) {
        // Outside the live region the kernel must not have written. This is the
        // check that catches an o_stride error, which in the MoE case would
        // corrupt a neighbouring expert's columns rather than this one's.
        if (raw != 0xCDCD) {
          if (++local_fail <= 5) {
            printf("FAIL [%s] m=%d n=%d: wrote outside live region "
                   "(num_active=%d output_size=%d)\n",
                   name, m, n, num_active, output_size);
          }
        }
        continue;
      }

      if (raw == 0xCDCD) {
        if (++local_fail <= 5) {
          printf("FAIL [%s] m=%d n=%d: still poisoned (never written)\n", name,
                 m, n);
        }
        continue;
      }

      double want = 0.0;
      for (int k = 0; k < REDUCTION_SIZE; ++k) {
        want += (double)a_val(m, k) * (double)b_val(n, k);
      }
      if (use_bias) {
        want += (double)bias_val(n);
      }
      if (residual_add) {
        want += (double)res_val(m, n);
      }
      float const got = __bfloat162float(got_raw);
      // Finiteness first, and skip the rest on failure. `rel > 6e-3` is false
      // when rel is NaN, so without this a kernel emitting NaN everywhere
      // passes with worst_rel printed as 0.00e+00 -- which reads better than a
      // correct run, not worse. (Seen for real in test_kv_cache_update.hip,
      // where it let a wave64 mutant survive.)
      if (!std::isfinite((double)got) || !std::isfinite(want)) {
        if (++local_fail <= 5) {
          printf("FAIL [%s] m=%d n=%d: NON-FINITE got %f want %f -- a "
                 "tolerance check cannot detect this\n", name, m, n, got, want);
        }
        continue;
      }
      double const denom = std::max(std::fabs(want), 1.0);
      double const rel = std::fabs(got - want) / denom;
      worst = std::max(worst, rel);
      // The accumulation is fp32 over exactly-representable bf16 products; the
      // only rounding is the final bf16 store, ~2^-8 relative.
      if (rel > 6e-3) {
        if (++local_fail <= 5) {
          printf("FAIL [%s] m=%d n=%d: got %.5f want %.5f (rel %.4f)\n", name,
                 m, n, got, want, rel);
        }
      }
    }
  }

  printf("%-26s bs=%-4d k=%-5d N=%-5d ostride=%-5d act=%-4d bias=%d res=%d "
         "worst_rel=%.2e %s\n",
         name, BATCH_SIZE, REDUCTION_SIZE, output_size, o_stride, num_active,
         (int)use_bias, (int)residual_add, worst, local_fail ? "FAIL" : "ok");
  failures_total += local_fail;

  hipFree(d_a);
  hipFree(d_b);
  hipFree(d_bias);
  hipFree(d_res);
  hipFree(d_o);
}

int main() {
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);
  printf("device=%s warpSize=%d threads=%d waves=%d\n", prop.gcnArchName,
         prop.warpSize, NTHREADS, NTHREADS / prop.warpSize);
  if (prop.warpSize != 32) {
    printf("SKIP: wave32-only (gfx1250).\n");
    return 0;
  }

  // Exactly one block pass: 8 waves x 16 cols = 128.
  run_case<16, 128>("aligned-128", 128, 128, 16, false, false);
  // Ragged N: 100 is not a multiple of 128, so the last pass is partial and
  // the b_live guard has to hold.
  run_case<16, 128>("ragged-N", 100, 100, 16, true, false);
  // N wider than one pass: 256 needs two strided passes per wave.
  run_case<16, 256>("two-passes", 256, 256, 16, true, false);
  // Batch above the 16-row WMMA tile: M_TILES loops.
  run_case<32, 128>("multi-mtile", 128, 128, 32, true, false);
  // Ragged M: num_active below BATCH_SIZE, the decode steady state.
  run_case<32, 128>("ragged-M", 128, 128, 19, true, false);
  // o_stride wider than output_size: the MoE column-slice case.
  run_case<16, 128>("wide-ostride", 96, 256, 16, true, false);
  // Residual add path.
  run_case<16, 128>("residual", 128, 128, 16, true, true);
  // GPT-OSS-shaped reduction: 2880 padded to 3072, K % 128 == 0.
  run_case<16, 3072>("gptoss-hidden", 128, 128, 16, true, false);

  if (failures_total == 0) {
    printf("PASS: WMMA linear matches host double reference.\n");
    return 0;
  }
  printf("FAIL: %d mismatch(es).\n", failures_total);
  return 1;
}
