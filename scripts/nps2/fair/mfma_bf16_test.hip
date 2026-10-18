// Operand layout check for v_mfma_f32_16x16x32_bf16 on gfx950 (the LM head's BF16 path):
//   A (16 x 32): lane l holds row l & 15, K elements [(l >> 4) * 8, +8)
//   B (32 x 16): lane l holds column l & 15, K elements [(l >> 4) * 8, +8)
//   D (16 x 16): lane l holds column l & 15, rows 4 * (l >> 4) + i, i = 0..3
// Fills A and B with small integers (exact in bf16 and in f32 sums) and compares every output with a host reference.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

typedef __bf16 bf16x8_t __attribute__((ext_vector_type(8)));
typedef float f32x4_t __attribute__((ext_vector_type(4)));

__global__ void tile(unsigned short const *A, unsigned short const *B, float *D) {
  int const l = threadIdx.x & 63, r = l & 15, g = l >> 4;
  bf16x8_t a, b;
  unsigned short ta[8], tb[8];
  for (int j = 0; j < 8; ++j) {
    ta[j] = A[r * 32 + g * 8 + j];   // A row-major 16 x 32
    tb[j] = B[(g * 8 + j) * 16 + r]; // B row-major 32 x 16 (column r)
  }
  __builtin_memcpy(&a, ta, 16);
  __builtin_memcpy(&b, tb, 16);
  f32x4_t acc = {0.f, 0.f, 0.f, 0.f};
  acc = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, acc, 0, 0, 0);
  for (int i = 0; i < 4; ++i) D[(4 * g + i) * 16 + r] = acc[i];
}

static unsigned short to_bf16(float f) {
  unsigned u;
  memcpy(&u, &f, 4);
  return (unsigned short)(u >> 16);  // small integers are exact
}

int main() {
  unsigned short hA[16 * 32], hB[32 * 16];
  float a[16][32], b[32][16], hD[256];
  srand(3);
  for (int i = 0; i < 16; ++i)
    for (int k = 0; k < 32; ++k) { a[i][k] = (float)(rand() % 9 - 4); hA[i * 32 + k] = to_bf16(a[i][k]); }
  for (int k = 0; k < 32; ++k)
    for (int j = 0; j < 16; ++j) { b[k][j] = (float)(rand() % 9 - 4); hB[k * 16 + j] = to_bf16(b[k][j]); }
  unsigned short *dA, *dB;
  float *dD;
  hipMalloc(&dA, sizeof hA);
  hipMalloc(&dB, sizeof hB);
  hipMalloc(&dD, sizeof hD);
  hipMemcpy(dA, hA, sizeof hA, hipMemcpyHostToDevice);
  hipMemcpy(dB, hB, sizeof hB, hipMemcpyHostToDevice);
  hipLaunchKernelGGL(tile, dim3(1), dim3(64), 0, 0, dA, dB, dD);
  hipMemcpy(hD, dD, sizeof hD, hipMemcpyDeviceToHost);
  int bad = 0;
  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j) {
      float ref = 0;
      for (int k = 0; k < 32; ++k) ref += a[i][k] * b[k][j];
      if (ref != hD[i * 16 + j]) {
        if (bad < 5) printf("mismatch D[%d][%d] = %g, ref %g\n", i, j, hD[i * 16 + j], ref);
        ++bad;
      }
    }
  printf("bf16 16x16x32 layout: %d of 256 outputs wrong\n", bad);
  return bad != 0;
}
