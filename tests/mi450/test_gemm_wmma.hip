/* Correctness harness for the MI450 (gfx1250) WMMA GEMM.
 *
 * The thing this exists to answer: the per-lane fragment layout in
 * tasks/mi450/gemm_handtuned_mi450.cuh was hand-derived and never executed.
 * A wrong layout still compiles and still emits the right instruction -- it
 * just computes the wrong numbers. Only running it against a reference
 * settles that.
 *
 * Build for gfx1250 and run under FFM (no gfx1250 silicon required):
 *
 *   T=/home/claudeuser/mi450-toolchain
 *   $T/llvm/bin/clang -x hip --offload-arch=gfx1250 --rocm-path=$T \
 *       -I../../include -O2 test_gemm_wmma.hip -o test_gemm_wmma \
 *       -L$T/lib -lamdhip64
 *   # then the FFM env from PORTING_MI450.md, and ./test_gemm_wmma
 *
 * The reference is computed on the host in fp32 from the same bf16 inputs the
 * device sees, so the only difference under test is the GEMM itself. The
 * tolerance is loose enough for bf16 accumulation order (bf16 carries ~8 bits
 * of mantissa) and tight enough that a transposed or rotated fragment layout
 * -- the actual failure mode being screened for -- cannot slip through: a
 * wrong layout does not produce slightly-off values, it produces values from
 * the wrong matrix elements entirely.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <vector>

#include "mirage/persistent_kernel/tasks/mi450/gemm_handtuned_mi450.cuh"

#define HIP_CHECK(x)                                                           \
  do {                                                                         \
    hipError_t e = (x);                                                        \
    if (e != hipSuccess) {                                                     \
      printf("HIP error %s at line %d\n", hipGetErrorString(e), __LINE__);     \
      return 2;                                                                \
    }                                                                          \
  } while (0)

constexpr int M = 16;  // BATCH_SIZE
constexpr int K = 512; // REDUCTION_SIZE
constexpr int N = 64;  // output_size, 4 waves x 16 columns

#if defined(__gfx1250__) || !defined(__HIP_DEVICE_COMPILE__)
__global__ __launch_bounds__(128) void gemm_kernel(void const *a,
                                                   void const *b,
                                                   void const *r,
                                                   void *o,
                                                   int num_active_tokens,
                                                   bool residual_add) {
#if defined(MIRAGE_ARCH_GFX1250)
  kernel::mi450::gemm_handtuned_wmma<M, K>(
      a, b, r, o, num_active_tokens, residual_add, N, 0, N);
#endif
}
#endif

static float bf16_to_f32(__bf16 v) {
  return (float)v;
}

int main() {
  int dev = 0;
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, dev) != hipSuccess) {
    printf("no HIP device\n");
    return 2;
  }
  printf("device: %s (%s)\n", prop.name, prop.gcnArchName);

  std::vector<__bf16> hA(M * K), hB(N * K), hR(M * N);
  // Deterministic, non-symmetric values. Symmetric or constant inputs would
  // mask a transposed fragment layout, which is the bug being hunted.
  for (int i = 0; i < M * K; ++i) {
    hA[i] = (__bf16)(((i * 37) % 17 - 8) * 0.125f);
  }
  for (int i = 0; i < N * K; ++i) {
    hB[i] = (__bf16)(((i * 53) % 13 - 6) * 0.25f);
  }
  for (int i = 0; i < M * N; ++i) {
    hR[i] = (__bf16)(((i * 11) % 7 - 3) * 0.5f);
  }

  // Host reference: C[m][n] = sum_k A[m][k] * B[n][k] (+ residual).
  std::vector<float> ref(M * N, 0.f);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.f;
      for (int k = 0; k < K; ++k) {
        acc += bf16_to_f32(hA[m * K + k]) * bf16_to_f32(hB[n * K + k]);
      }
      ref[m * N + n] = acc + bf16_to_f32(hR[m * N + n]);
    }
  }

  __bf16 *dA, *dB, *dR, *dO;
  HIP_CHECK(hipMalloc(&dA, M * K * sizeof(__bf16)));
  HIP_CHECK(hipMalloc(&dB, N * K * sizeof(__bf16)));
  HIP_CHECK(hipMalloc(&dR, M * N * sizeof(__bf16)));
  HIP_CHECK(hipMalloc(&dO, M * N * sizeof(__bf16)));
  HIP_CHECK(hipMemcpy(dA, hA.data(), M * K * sizeof(__bf16), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB, hB.data(), N * K * sizeof(__bf16), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dR, hR.data(), M * N * sizeof(__bf16), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(dO, 0, M * N * sizeof(__bf16)));

  hipLaunchKernelGGL(
      gemm_kernel, dim3(1), dim3(128), 0, 0, dA, dB, dR, dO, M, true);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<__bf16> hO(M * N);
  HIP_CHECK(hipMemcpy(hO.data(), dO, M * N * sizeof(__bf16), hipMemcpyDeviceToHost));

  // Relative tolerance against the magnitude of the reference value. bf16
  // output quantization alone is ~2^-8 relative, so 3% leaves headroom for
  // accumulation-order differences without admitting a layout error.
  int bad = 0;
  float worst = 0.f;
  int worst_m = -1, worst_n = -1;
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float got = bf16_to_f32(hO[m * N + n]);
      float want = ref[m * N + n];
      float denom = fmaxf(fabsf(want), 1.0f);
      float rel = fabsf(got - want) / denom;
      if (rel > worst) {
        worst = rel;
        worst_m = m;
        worst_n = n;
      }
      if (rel > 0.03f) {
        if (bad < 8) {
          printf("  mismatch [%2d,%2d] got %10.3f want %10.3f (rel %.3f)\n",
                 m, n, got, want, rel);
        }
        ++bad;
      }
    }
  }

  printf("worst rel err %.4f at [%d,%d]\n", worst, worst_m, worst_n);
  if (bad) {
    printf("FAIL: %d/%d elements outside tolerance\n", bad, M * N);
    printf("  -> most likely the per-lane fragment layout in "
           "gemm_handtuned_mi450.cuh, not the arithmetic.\n");
    return 1;
  }
  printf("PASS: all %d elements within tolerance\n", M * N);
  return 0;
}
