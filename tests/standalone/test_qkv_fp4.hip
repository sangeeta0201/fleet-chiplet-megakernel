/*
 * MPK_QKV_MXFP4 oracle: the qkv_a GEMM with E2M1 weights against the same
 * values packed as E4M3 (gen_qkv_fp4.py writes both). The two kernel paths
 * multiply identical numbers, so the outputs must agree to accumulation-order
 * noise; anything larger is a layout bug in the WEIGHT_FP4 path.
 *
 * Build: hipcc -O3 --offload-arch=gfx950 -I<repo>/include
 *          -I<repo>/include/mirage/persistent_kernel -D__HIP_PLATFORM_AMD__
 *          -DMIRAGE_BACKEND_USE_ROCM -std=c++17 test_qkv_fp4.hip -o qkv_fp4
 * Run:   ./qkv_fp4 <dir with x.bin gamma.bin w4.bin w8.bin> [rows]
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <vector>

#define MPK_WS_WAVE_SYNC(wave) ((void)0)
// The worker-state tracers the persistent kernel defines; no-ops here.
#define MPK_WS_MARK(code, aux) ((void)0)
#define MPK_WS_PHASE(phase, layer, xcd) ((void)0)
#define MPK_WS_WAIT_BEGIN(barrier_id, expected) ((void)0)
#define MPK_WS_WAIT_TICK(observed, spins) ((void)0)
#define MPK_WS_WAIT_AUX(a0, a1, a2, a3) ((void)0)
namespace kernel {
__device__ __forceinline__ int _gang_moe_get_xcd_id() {
  int xcd_id;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcd_id));
  return xcd_id;
}
} // namespace kernel

#include "mirage/persistent_kernel/mpk_atoms.cuh"
#include "mirage/persistent_kernel/tasks/mi300/gang_rmsnorm_linear_mxfp8_bias_mi300.cuh"

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t err = call;                                                     \
    if (err != hipSuccess) {                                                   \
      fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,             \
              hipGetErrorString(err));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

using namespace kernel;

constexpr int K = 6144;
constexpr int OPW = 16;

template <bool FP4>
__global__ void __launch_bounds__(256, 1)
    run(void const *x, void const *gamma, void *scratch, void const *w,
        void const *bias, void *out, int nwg) {
  gang_rmsnorm_linear_mxfp8_bias_kernel<1, OPW, K, K, false, false, 0, false,
                                        false, false, false, K, false, FP4>(
      x, gamma, scratch, w, bias, out, 1, nwg, nwg * OPW, blockIdx.x);
}

static std::vector<unsigned char> slurp(char const *dir, char const *name) {
  char path[1024];
  snprintf(path, sizeof(path), "%s/%s", dir, name);
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "cannot open %s\n", path);
    exit(1);
  }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<unsigned char> v(n);
  if (fread(v.data(), 1, n, f) != (size_t)n) {
    exit(1);
  }
  fclose(f);
  return v;
}

static float bf2f(unsigned short u) {
  unsigned b = (unsigned)u << 16;
  float f;
  memcpy(&f, &b, 4);
  return f;
}

template <bool FP4>
static std::vector<unsigned short> launch(std::vector<unsigned char> const &w,
                                          void *dx, void *dg, void *dbias,
                                          int rows) {
  int const nwg = rows / OPW;
  void *dw, *dout, *dscr;
  HIP_CHECK(hipMalloc(&dw, w.size()));
  HIP_CHECK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&dout, rows * 2));
  HIP_CHECK(hipMemset(dout, 0, rows * 2));
  HIP_CHECK(hipMalloc(&dscr, K * 2));
  size_t const lds = 64 * 1024;
  HIP_CHECK(hipFuncSetAttribute((void const *)run<FP4>,
                                hipFuncAttributeMaxDynamicSharedMemorySize,
                                (int)lds));
  hipLaunchKernelGGL(run<FP4>, dim3(nwg), dim3(256), lds, 0, dx, dg, dscr, dw,
                     dbias, dout, nwg);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<unsigned short> h(rows);
  HIP_CHECK(hipMemcpy(h.data(), dout, rows * 2, hipMemcpyDeviceToHost));
  hipFree(dw);
  hipFree(dout);
  hipFree(dscr);
  return h;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <dir> [rows]\n", argv[0]);
    return 2;
  }
  int const rows = argc > 2 ? atoi(argv[2]) : 256;
  auto x = slurp(argv[1], "x.bin");
  auto g = slurp(argv[1], "gamma.bin");
  auto w4 = slurp(argv[1], "w4.bin");
  auto w8 = slurp(argv[1], "w8.bin");
  void *dx, *dg, *dbias;
  HIP_CHECK(hipMalloc(&dx, x.size()));
  HIP_CHECK(hipMalloc(&dg, g.size()));
  HIP_CHECK(hipMalloc(&dbias, rows * 2));
  HIP_CHECK(hipMemcpy(dx, x.data(), x.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dg, g.data(), g.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(dbias, 0, rows * 2));
  auto o4 = launch<true>(w4, dx, dg, dbias, rows);
  auto o8 = launch<false>(w8, dx, dg, dbias, rows);
  double maxabs = 0, maxref = 0;
  int worst = -1, bad = 0;
  for (int r = 0; r < rows; r++) {
    double a = bf2f(o4[r]), b = bf2f(o8[r]);
    double d = fabs(a - b);
    maxref = fmax(maxref, fabs(b));
    if (d > maxabs) {
      maxabs = d;
      worst = r;
    }
    if (d > 1e-2 * fmax(fabs(b), 1e-3)) {
      bad++;
    }
  }
  printf("rows=%d max|fp4-fp8|=%.3e (row %d: %.6f vs %.6f) max|fp8|=%.3e "
         "rows off by >1%%: %d\n",
         rows, maxabs, worst, worst >= 0 ? bf2f(o4[worst]) : 0.0,
         worst >= 0 ? bf2f(o8[worst]) : 0.0, maxref, bad);
  for (int r = 0; r < 8; r++) {
    printf("  row %d: fp4 %.6f fp8 %.6f\n", r, bf2f(o4[r]), bf2f(o8[r]));
  }
  printf("%s\n", bad == 0 ? "PASS" : "FAIL");
  return bad == 0 ? 0 : 1;
}
