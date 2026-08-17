/*
 * Numerical check for the expert-parallel reduction folded into GLM's RMSNorm
 * prologue: _rnlm8_resadd_norm_rcp<REDUCTION_SIZE, EP_PEER_SLOTS, EP_SLOT_ELEMS>
 * in gang_rmsnorm_linear_mxfp8_bias_mi300.cuh.
 *
 * WHY THIS EXISTS
 * ---------------
 * Under EP the residual stream stops being `f32 workspace + bf16 residual` and
 * becomes `sum over ranks of that rank's bf16 partial`. That is not the same
 * arithmetic: each rank rounds its own partial to bf16 before the sum, so the
 * cross-rank total is a sum of N rounded terms rather than one rounding of an
 * f32 total. There is no way to make those agree bit-for-bit, so the question
 * this answers is how far apart they are, and whether the kernel implements
 * the sum it claims to.
 *
 * Two checks:
 *   1. exactness -- the kernel's EP output against a CPU replay of the same
 *      bf16 sum. Must be bit-identical; any difference is an addressing or
 *      unpack bug, not a rounding effect.
 *   2. drift -- the EP output against the single-GPU (f32 workspace) path on
 *      the same underlying values. Expected to differ; the point is to bound
 *      by how much, and to confirm rms_rcp tracks.
 *
 * Build: ./build.sh
 * Run:   ./test_ep_prologue_sum
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <vector>

// Same two shims test_mxfp8_linear.hip supplies: inside the megakernel these
// come from headers task_header.cuh includes earlier, and neither is reached
// by the prologue under test.
#define MPK_WS_WAVE_SYNC(wave) ((void)0)
namespace kernel {
__device__ __forceinline__ int _gang_moe_get_xcd_id() {
  int xcd_id;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcd_id));
  return xcd_id;
}
} // namespace kernel

// st_wt_* for the write-through epilogues down the include chain, same as
// test_gemv_mxfp8_bw.hip. Must come first.
#include "mirage/persistent_kernel/mpk_atoms.cuh"
#include "mirage/persistent_kernel/tasks/mi300/gang_rmsnorm_linear_mxfp8_bias_mi300.cuh"

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t err = call;                                                     \
    if (err != hipSuccess) {                                                   \
      fprintf(stderr,                                                          \
              "HIP error %s:%d: %s\n",                                         \
              __FILE__,                                                        \
              __LINE__,                                                        \
              hipGetErrorString(err));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

using namespace kernel;

// GLM-4.7-Flash's hidden size, batch 1. EP_SLOT_ELEMS == BATCH * REDUCTION is
// what gang_rmsnorm_linear_mxfp8_bias_kernel passes.
static constexpr int N = 2048;
static constexpr int WORLD = 8;

// ── host-side bf16, matching _gang_float_to_bf16's round-to-nearest-even ────
static unsigned short h_f2b(float f) {
  unsigned u;
  __builtin_memcpy(&u, &f, 4);
  unsigned bias = ((u >> 16) & 1u) + 0x7FFFu;
  return (unsigned short)((u + bias) >> 16);
}
static float h_b2f(unsigned short b) {
  unsigned u = (unsigned)b << 16;
  float f;
  __builtin_memcpy(&f, &u, 4);
  return f;
}

// ── kernels under test ─────────────────────────────────────────────────────
// One workgroup, 256 threads, exactly the shape the prologue runs in. s_x is
// the LDS staging buffer the quantizer would read; here it is copied out so
// the summed row can be checked directly.
__global__ void run_ep(unsigned short const *gather, unsigned short *out_row,
                       float *out_rcp) {
  __shared__ unsigned short s_x[N];
  float rcp = _rnlm8_resadd_norm_rcp<N, WORLD, N>(
      nullptr, gather, nullptr, s_x, /*store_x=*/false);
  __syncthreads();
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    out_row[i] = s_x[i];
  }
  if (threadIdx.x == 0) {
    *out_rcp = rcp;
  }
}

__global__ void run_plain(float const *ws, unsigned short const *res,
                          unsigned short *out_row, float *out_rcp) {
  __shared__ unsigned short s_x[N];
  float rcp = _rnlm8_resadd_norm_rcp<N>(ws, res, nullptr, s_x,
                                        /*store_x=*/false);
  __syncthreads();
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    out_row[i] = s_x[i];
  }
  if (threadIdx.x == 0) {
    *out_rcp = rcp;
  }
}

int main() {
  srand(1234);

  // The underlying values: WORLD disjoint expert partials plus one residual.
  // Partials are the small numbers a routed-expert sum actually produces; the
  // residual is O(1) and dominates, which is the realistic ratio and the one
  // that makes the rounding question interesting.
  std::vector<float> part(WORLD * N), res_f(N);
  for (int p = 0; p < WORLD; p++) {
    for (int i = 0; i < N; i++) {
      part[p * N + i] = (rand() / (float)RAND_MAX - 0.5f) * 0.25f;
    }
  }
  for (int i = 0; i < N; i++) {
    res_f[i] = (rand() / (float)RAND_MAX - 0.5f) * 2.0f;
  }

  // EP layout: rank p publishes bf16(its partial), and rank EP_FOLD_PE == 0
  // folds the residual into its own slot so the residual survives the
  // cross-rank sum exactly once.
  std::vector<unsigned short> gather(WORLD * N);
  for (int p = 0; p < WORLD; p++) {
    for (int i = 0; i < N; i++) {
      float v = part[p * N + i] + (p == 0 ? res_f[i] : 0.0f);
      gather[p * N + i] = h_f2b(v);
    }
  }

  // Single-GPU layout: one f32 workspace holding every partial, one bf16
  // residual. This is what the non-EP path reads.
  std::vector<float> ws(N);
  std::vector<unsigned short> res_b(N);
  for (int i = 0; i < N; i++) {
    float acc = 0.0f;
    for (int p = 0; p < WORLD; p++) {
      acc += part[p * N + i];
    }
    ws[i] = acc;
    res_b[i] = h_f2b(res_f[i]);
  }

  unsigned short *d_gather, *d_res, *d_out_ep, *d_out_plain;
  float *d_ws, *d_rcp_ep, *d_rcp_plain;
  HIP_CHECK(hipMalloc(&d_gather, WORLD * N * 2));
  HIP_CHECK(hipMalloc(&d_res, N * 2));
  HIP_CHECK(hipMalloc(&d_out_ep, N * 2));
  HIP_CHECK(hipMalloc(&d_out_plain, N * 2));
  HIP_CHECK(hipMalloc(&d_ws, N * 4));
  HIP_CHECK(hipMalloc(&d_rcp_ep, 4));
  HIP_CHECK(hipMalloc(&d_rcp_plain, 4));
  HIP_CHECK(hipMemcpy(d_gather, gather.data(), WORLD * N * 2,
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_res, res_b.data(), N * 2, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_ws, ws.data(), N * 4, hipMemcpyHostToDevice));

  run_ep<<<1, 256>>>(d_gather, d_out_ep, d_rcp_ep);
  run_plain<<<1, 256>>>(d_ws, d_res, d_out_plain, d_rcp_plain);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<unsigned short> out_ep(N), out_plain(N);
  float rcp_ep, rcp_plain;
  HIP_CHECK(hipMemcpy(out_ep.data(), d_out_ep, N * 2, hipMemcpyDeviceToHost));
  HIP_CHECK(
      hipMemcpy(out_plain.data(), d_out_plain, N * 2, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&rcp_ep, d_rcp_ep, 4, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&rcp_plain, d_rcp_plain, 4, hipMemcpyDeviceToHost));

  // ── check 1: exactness against a CPU replay of the same bf16 sum ──────────
  // The kernel accumulates slot 0 first and then slots 1..N-1 in order, in
  // f32; the replay does the same, so this must match bit-for-bit.
  int bad = 0;
  double ssq_ref = 0.0;
  for (int i = 0; i < N; i++) {
    float acc = h_b2f(gather[i]);
    for (int p = 1; p < WORLD; p++) {
      acc += h_b2f(gather[p * N + i]);
    }
    unsigned short ref = h_f2b(acc);
    if (ref != out_ep[i]) {
      if (bad < 8) {
        printf("  mismatch at %4d: kernel %04x (%.6f)  ref %04x (%.6f)\n", i,
               out_ep[i], h_b2f(out_ep[i]), ref, h_b2f(ref));
      }
      bad++;
    }
    float q = h_b2f(ref);
    ssq_ref += (double)q * q;
  }
  float rcp_ref = 1.0f / sqrtf((float)(ssq_ref / N) + 1e-5f);
  printf("check 1  EP sum vs CPU replay      : %s (%d/%d elements differ)\n",
         bad == 0 ? "PASS" : "FAIL", bad, N);
  printf("         rms_rcp kernel %.8f  ref %.8f  rel %.2e\n", rcp_ep, rcp_ref,
         fabsf(rcp_ep - rcp_ref) / rcp_ref);

  // ── check 2: drift against the single-GPU path ───────────────────────────
  // Not expected to be zero. WORLD bf16 roundings before the sum instead of
  // one after it, so the error floor is ~sqrt(WORLD) bf16 ulps of the partial
  // magnitude -- small next to the residual, which dominates the row.
  double max_abs = 0.0, sum_sq_d = 0.0, sum_sq_v = 0.0;
  int exact = 0;
  for (int i = 0; i < N; i++) {
    float a = h_b2f(out_ep[i]), b = h_b2f(out_plain[i]);
    double d = a - b;
    if (a == b) {
      exact++;
    }
    if (fabs(d) > max_abs) {
      max_abs = fabs(d);
    }
    sum_sq_d += d * d;
    sum_sq_v += (double)b * b;
  }
  double rel_l2 = sqrt(sum_sq_d / sum_sq_v);
  printf("check 2  EP vs single-GPU path     : max|d| %.3e  rel L2 %.3e  "
         "bit-identical %d/%d\n",
         max_abs, rel_l2, exact, N);
  printf("         rms_rcp EP %.8f  single %.8f  rel %.2e\n", rcp_ep, rcp_plain,
         fabsf(rcp_ep - rcp_plain) / rcp_plain);

  // bf16 has 8 mantissa bits, so one ulp is ~2^-8 relative. Eight of them in
  // quadrature is ~1.1e-2 relative to the PARTIAL magnitude, and the partials
  // here are ~1/8 of the row, so ~1.4e-3 relative to the row. Gate a little
  // above that.
  bool drift_ok = rel_l2 < 5e-3;
  printf("\n%s\n", (bad == 0 && drift_ok) ? "PASS" : "FAIL");
  return (bad == 0 && drift_ok) ? 0 : 1;
}
