// Wave-width correctness for the fused RMSNorm prologue.
//
// gang_rmsnorm_linear_bias_mi300.cuh's rmsnorm_inline_amd() reduces a sum of
// squares in two stages: a butterfly within each wave, then a cross-wave sum
// through LDS. Both stages were written against wave64 -- the butterfly started
// at offset 32, the cross-wave loop ran to a hardcoded NUM_WAVES=4, and the LDS
// scratch was float[16]. At wave32 a 256-thread block is 8 waves of 32, so the
// original code would reduce within 32-lane groups using a 64-wide butterfly
// (whose first step is a self-shuffle) and then sum only 4 of 8 wave partials.
//
// Both failure modes produce a *plausible* number, never a crash: RMS comes out
// too large by roughly sqrt(2), so activations are quietly mis-scaled. This test
// pins the exact value instead.
//
// The reference is computed in double on the host. Inputs are drawn from small
// powers of two so every square and every partial sum is exactly representable
// in float and bf16 never rounds -- so the comparison can be near-exact rather
// than tolerance-based, and a real reduction bug cannot hide inside an epsilon.
// hip_runtime.h first: arch_traits.cuh (pulled in transitively below) uses
// __device__/__forceinline__/__shfl_xor without including HIP itself, so it
// only compiles when something has already established them.
#include <hip/hip_runtime.h>

#include "mirage/persistent_kernel/tasks/mi300/gang_rmsnorm_linear_bias_mi300.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using bf16 = __hip_bfloat16;

// GPT-OSS shape: hidden 2880 padded to 3072 for tile alignment.
static constexpr int STORAGE_DIM = 3072;
static constexpr int ACTUAL_HIDDEN = 2880;
static constexpr int NTHREADS = 256;

__global__ void rmsnorm_probe(void const *in, void const *w, void *out) {
  kernel::gang_rmsnorm_detail::rmsnorm_inline_amd<STORAGE_DIM, ACTUAL_HIDDEN>(
      in, w, out);
}

int main() {
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);

  std::vector<float> h_in(STORAGE_DIM), h_w(STORAGE_DIM);
  std::vector<bf16> b_in(STORAGE_DIM), b_w(STORAGE_DIM);

  // Values in {0.25, 0.5, 1, 2} and weights in {0.5, 1, 2}: exact in bf16, and
  // the sum of 2880 such squares stays well inside float's exact-integer range
  // when scaled, so the host double reference and the device float agree to the
  // last bit rather than merely to a tolerance.
  for (int i = 0; i < STORAGE_DIM; ++i) {
    if (i < ACTUAL_HIDDEN) {
      static float const vals[4] = {0.25f, 0.5f, 1.0f, 2.0f};
      static float const wts[3] = {0.5f, 1.0f, 2.0f};
      h_in[i] = vals[i % 4];
      h_w[i] = wts[i % 3];
    } else {
      // The padded tail must be zero in both: the kernel sums squares over the
      // full STORAGE_DIM but divides by ACTUAL_HIDDEN.
      h_in[i] = 0.0f;
      h_w[i] = 0.0f;
    }
    b_in[i] = __float2bfloat16(h_in[i]);
    b_w[i] = __float2bfloat16(h_w[i]);
  }

  size_t const bytes = STORAGE_DIM * sizeof(bf16);
  void *d_in, *d_w, *d_out;
  hipMalloc(&d_in, bytes);
  hipMalloc(&d_w, bytes);
  hipMalloc(&d_out, bytes);
  hipMemcpy(d_in, b_in.data(), bytes, hipMemcpyHostToDevice);
  hipMemcpy(d_w, b_w.data(), bytes, hipMemcpyHostToDevice);
  // Poison, so an untouched output cannot masquerade as a pass.
  hipMemset(d_out, 0xCD, bytes);

  hipLaunchKernelGGL(rmsnorm_probe, dim3(1), dim3(NTHREADS), 0, 0, d_in, d_w,
                     d_out);
  hipError_t err = hipDeviceSynchronize();
  if (err != hipSuccess) {
    printf("FAIL: kernel error: %s\n", hipGetErrorString(err));
    return 1;
  }

  std::vector<bf16> h_out(STORAGE_DIM);
  hipMemcpy(h_out.data(), d_out, bytes, hipMemcpyDeviceToHost);

  // Host reference, in double.
  double ssq = 0.0;
  for (int i = 0; i < STORAGE_DIM; ++i) {
    ssq += (double)h_in[i] * (double)h_in[i];
  }
  double const mean = ssq / (double)ACTUAL_HIDDEN;
  double const rms_rcp = 1.0 / std::sqrt(mean + 1e-5);

  int failures = 0;
  int poisoned = 0;
  double worst = 0.0;
  for (int i = 0; i < STORAGE_DIM; ++i) {
    float const got = __bfloat162float(h_out[i]);
    double const want_f = (double)h_in[i] * rms_rcp * (double)h_w[i];
    float const want = __bfloat162float(__float2bfloat16((float)want_f));

    // 0xCDCD as bf16 is a large negative value; catching it explicitly gives a
    // clearer message than a magnitude mismatch would.
    unsigned short raw;
    __builtin_memcpy(&raw, &h_out[i], 2);
    if (raw == 0xCDCD) {
      if (++poisoned <= 3) {
        printf("FAIL out[%d] still poisoned (kernel never wrote it)\n", i);
      }
      ++failures;
      continue;
    }

    // Finiteness before tolerance. The poison check above does NOT cover this:
    // arithmetic NaN has different bits from 0xCDCD, so the slot looks
    // legitimately written, and `rel > 4e-3` is false for NaN -- so a kernel
    // producing NaN for every element would pass with worst_rel=0.00e+00. That
    // is exactly how a wave64 mutant survived test_kv_cache_update.hip. Which
    // matters here in particular: this test exists to catch a halved cross-wave
    // sum, and a *negative* halved sum feeds rsqrt and yields NaN.
    if (!std::isfinite(got) || !std::isfinite(want)) {
      if (++failures <= 8) {
        printf("FAIL out[%d]: NON-FINITE got %f want %f -- a tolerance check "
               "cannot detect this\n", i, got, want);
      }
      continue;
    }

    double const rel =
        (want == 0.0f) ? std::fabs(got) : std::fabs((got - want) / want);
    if (rel > worst) {
      worst = rel;
    }
    // One bf16 ulp is ~2^-8; anything above that is a real disagreement, not
    // rounding. A halved cross-wave sum would show up as ~0.41 here.
    if (rel > 4e-3) {
      if (++failures <= 8) {
        printf("FAIL out[%d]: got %.6f want %.6f (rel %.4f)\n", i, got, want,
               rel);
      }
    }
  }

  printf("wave_size=%d threads=%d waves=%d storage=%d actual=%d "
         "rms_rcp=%.8f worst_rel=%.2e\n",
         prop.warpSize, NTHREADS, NTHREADS / prop.warpSize, STORAGE_DIM,
         ACTUAL_HIDDEN, rms_rcp, worst);

  if (failures == 0) {
    printf("PASS: rmsnorm prologue matches host double reference at wave%d.\n",
           prop.warpSize);
    return 0;
  }
  printf("FAIL: %d mismatch(es).\n", failures);
  return 1;
}
