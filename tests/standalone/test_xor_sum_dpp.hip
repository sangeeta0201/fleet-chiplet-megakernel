// MPK_QB_NORM_DPP (mpk_atoms.cuh): q_b's RMSNorm wave reduction, _rnlm8_xor_sum_dpp, against the __shfl_xor
// 32, 16, 8, 4, 2, 1 butterfly it replaces, and the every-wave four-wave total against wave 0's xor-2, xor-1 one,
// on random floats. Every lane of every block must be bit-equal.
//   hipcc ... -I ../../include test_xor_sum_dpp.hip -o /tmp/test_xor_sum_dpp && /tmp/test_xor_sum_dpp
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#define MPK_WS_WAVE_SYNC(wave) ((void)0)
#define MPK_WS_MARK(code, aux) ((void)0)
#define MPK_WS_WAIT_BEGIN(b, e) ((void)0)
#define MPK_WS_WAIT_TICK(o, s) ((void)0)
#define MPK_WS_WAIT_AUX(...) ((void)0)
namespace kernel {
__device__ __forceinline__ int _gang_moe_get_xcd_id() {
  int xcd_id;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcd_id));
  return xcd_id;
}
} // namespace kernel
#define MPK_QB_NORM_DPP 1
#include "mirage/persistent_kernel/mpk_atoms.cuh"
#include "mirage/persistent_kernel/tasks/mi300/gang_rmsnorm_linear_mxfp8_bias_mi300.cuh"

__global__ void k(float const *in, float *out_ref, float *out_dpp, float *tot_ref, float *tot_dpp) {
  int const tid = threadIdx.x;
  int const lane = tid & 63, wave = tid >> 6;
  float const x = in[blockIdx.x * 256 + tid];
  float a = x;
#pragma unroll
  for (int offset = 32; offset > 0; offset >>= 1) {
    a += __shfl_xor(a, offset);
  }
  float const b = kernel::_rnlm8_xor_sum_dpp(x);
  out_ref[blockIdx.x * 256 + tid] = a;
  out_dpp[blockIdx.x * 256 + tid] = b;
  __shared__ float red[8];
  if (lane == 0) {
    red[wave] = a;
    red[4 + wave] = b;
  }
  __syncthreads();
  if (wave == 0) {
    float s = lane < 4 ? red[lane] : 0.0f;
    for (int offset = 2; offset > 0; offset >>= 1) {
      s += __shfl_xor(s, offset);
    }
    if (lane == 0) {
      tot_ref[blockIdx.x] = s;
    }
  }
  float const t = (red[4] + red[6]) + (red[5] + red[7]);
  if (tid == 255) {
    tot_dpp[blockIdx.x] = t;
  }
}

int main() {
  int const nb = 4096;
  size_t const n = (size_t)nb * 256;
  float *h = (float *)malloc(n * 4);
  srand(1);
  for (size_t i = 0; i < n; i++) {
    float const m = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
    h[i] = (rand() % 7 == 0) ? m * m : ldexpf(m, rand() % 40 - 20);
  }
  float *d_in, *d_a, *d_b, *d_ta, *d_tb;
  (void)hipMalloc(&d_in, n * 4);
  (void)hipMalloc(&d_a, n * 4);
  (void)hipMalloc(&d_b, n * 4);
  (void)hipMalloc(&d_ta, nb * 4);
  (void)hipMalloc(&d_tb, nb * 4);
  (void)hipMemcpy(d_in, h, n * 4, hipMemcpyHostToDevice);
  hipLaunchKernelGGL(k, dim3(nb), dim3(256), 0, 0, d_in, d_a, d_b, d_ta, d_tb);
  (void)hipDeviceSynchronize();
  float *ha = (float *)malloc(n * 4), *hb = (float *)malloc(n * 4);
  float *ta = (float *)malloc(nb * 4), *tb = (float *)malloc(nb * 4);
  (void)hipMemcpy(ha, d_a, n * 4, hipMemcpyDeviceToHost);
  (void)hipMemcpy(hb, d_b, n * 4, hipMemcpyDeviceToHost);
  (void)hipMemcpy(ta, d_ta, nb * 4, hipMemcpyDeviceToHost);
  (void)hipMemcpy(tb, d_tb, nb * 4, hipMemcpyDeviceToHost);
  size_t bad = 0, badt = 0;
  for (size_t i = 0; i < n; i++) {
    bad += memcmp(&ha[i], &hb[i], 4) != 0;
  }
  for (int i = 0; i < nb; i++) {
    badt += memcmp(&ta[i], &tb[i], 4) != 0;
  }
  printf("xor_sum_dpp: lanes differing %zu of %zu, four-wave totals differing %zu of %d\n", bad, n, badt, nb);
  printf("%s\n", (bad == 0 && badt == 0) ? "PASS" : "FAIL");
  return (bad == 0 && badt == 0) ? 0 : 1;
}
