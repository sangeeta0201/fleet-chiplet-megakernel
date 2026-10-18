// Does mfma_scale_f32_16x16x128_f8f6f4 with A = FP6 E2M3 weights (cbsz 2), B = FP8 E4M3 activation give the right
// dot products for "naturally" packed MXFP6 rows: per row and 128-value K tile, 96 bytes; value j of a 32-value block
// at bits [6j, 6j + 6) of that block's 24 bytes (little endian), one E8M0 scale per 32 contiguous values; loaded the
// way the FP4 kernels load their A operand (lane col = l & 15, group g = l >> 4: 24 bytes of row col at byte g * 24).
// B exactly as in mfma_fp4_test.hip (two 16-byte chunks at g * 16 and 64 + g * 16), which was verified there.
// Prints the max error against a host reference, then a single-hot probe pairing A value t of lane group 0 with the
// B position k it multiplies, so a permutation can be read off if natural packing is wrong.
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

typedef int i32x8_t __attribute__((ext_vector_type(8)));
typedef float f32x4_t __attribute__((ext_vector_type(4)));

__global__ void tile(unsigned char const *w, unsigned char const *ws, unsigned char const *x8, int sb_val,
                     float *out) {
  int const lane = threadIdx.x & 63, col = lane & 15, g = lane >> 4;
  int a6[6];
  __builtin_memcpy(a6, w + col * 96 + g * 24, 24);
  i32x8_t a = {a6[0], a6[1], a6[2], a6[3], a6[4], a6[5], 0, 0};
  int const sa = ws[col * 4 + g];
  int b8[8];
  __builtin_memcpy(b8, x8 + g * 16, 16);
  __builtin_memcpy(b8 + 4, x8 + 64 + g * 16, 16);
  i32x8_t b = {b8[0], b8[1], b8[2], b8[3], b8[4], b8[5], b8[6], b8[7]};
  f32x4_t acc = {0.f, 0.f, 0.f, 0.f};
  acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, acc, 2, 0, 0, sa, 0, sb_val);
  if (col == 0)
    for (int i = 0; i < 4; ++i) out[g * 4 + i] = acc[i];
}

static float e2m3(int c) {  // 6-bit code: s e e m m m, bias 1
  int const s = (c >> 5) & 1, e = (c >> 3) & 3, m = c & 7;
  float const v = e == 0 ? m / 8.f : ldexpf(1.f + m / 8.f, e - 1);
  return s ? -v : v;
}
static unsigned char e4m3(float v) {  // exactly representable small values only
  if (v == 0.f) return 0;
  unsigned char s = v < 0 ? 0x80 : 0;
  float a = fabsf(v);
  int e = (int)floorf(log2f(a));
  float m = a / ldexpf(1.f, e) - 1.f;
  return s | (unsigned char)(((e + 7) & 15) << 3) | (unsigned char)(m * 8);
}
static void pack_block(int const *codes32, unsigned char *dst24) {  // natural: value j at bits [6j, 6j + 6)
  memset(dst24, 0, 24);
  for (int j = 0; j < 32; ++j)
    for (int bit = 0; bit < 6; ++bit)
      if ((codes32[j] >> bit) & 1) dst24[(6 * j + bit) / 8] |= (unsigned char)(1u << ((6 * j + bit) % 8));
}

int main() {
  unsigned char *dw, *dws, *dx;
  float *dout;
  hipMalloc(&dw, 16 * 96);
  hipMalloc(&dws, 64);
  hipMalloc(&dx, 128);
  hipMalloc(&dout, 64);
  srand(7);
  int code[16][128];
  unsigned char hw[16 * 96], hws[64], hx[128];
  float xv[128];
  for (int r = 0; r < 16; ++r) {
    for (int k = 0; k < 128; ++k) code[r][k] = rand() & 63;
    for (int b = 0; b < 4; ++b) {
      pack_block(&code[r][32 * b], hw + r * 96 + b * 24);
      hws[r * 4 + b] = (unsigned char)(125 + (rand() % 5));
    }
  }
  float const xs[7] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f};
  for (int k = 0; k < 128; ++k) {
    xv[k] = xs[rand() % 7] * (rand() & 1 ? -1.f : 1.f);
    hx[k] = e4m3(xv[k]);
  }
  hipMemcpy(dw, hw, sizeof hw, hipMemcpyHostToDevice);
  hipMemcpy(dws, hws, sizeof hws, hipMemcpyHostToDevice);
  hipMemcpy(dx, hx, sizeof hx, hipMemcpyHostToDevice);
  hipLaunchKernelGGL(tile, dim3(1), dim3(64), 0, 0, dw, dws, dx, 127, dout);
  float h[16];
  hipMemcpy(h, dout, sizeof h, hipMemcpyDeviceToHost);
  double maxerr = 0, maxref = 0;
  for (int r = 0; r < 16; ++r) {
    double ref = 0;
    for (int k = 0; k < 128; ++k) ref += e2m3(code[r][k]) * ldexp(1.0, hws[r * 4 + k / 32] - 127) * xv[k];
    maxerr = fmax(maxerr, fabs(ref - h[r]));
    maxref = fmax(maxref, fabs(ref));
    if (r < 3) printf("row %d: mfma %g ref %g\n", r, h[r], ref);
  }
  printf("natural FP6 packing: max |err| over 16 rows = %g (max |ref| %g)\n", maxerr, maxref);

  // single-hot probe: row 0, lane group 0, A value t = 1.0 (code 0b001000 = e 1, m 0); B = 1.0 at one position k.
  printf("pairing (lane group 0 value t -> natural B position k):");
  for (int t = 0; t < 32; ++t) {
    int codes[32] = {0};
    codes[t] = 8;
    memset(hw, 0, sizeof hw);
    pack_block(codes, hw);
    memset(hws, 127, sizeof hws);
    hipMemcpy(dw, hw, sizeof hw, hipMemcpyHostToDevice);
    hipMemcpy(dws, hws, sizeof hws, hipMemcpyHostToDevice);
    int found = -1;
    for (int k = 0; k < 128 && found < 0; ++k) {
      memset(hx, 0, sizeof hx);
      hx[k] = e4m3(1.f);
      hipMemcpy(dx, hx, sizeof hx, hipMemcpyHostToDevice);
      hipLaunchKernelGGL(tile, dim3(1), dim3(64), 0, 0, dw, dws, dx, 127, dout);
      hipMemcpy(h, dout, sizeof h, hipMemcpyDeviceToHost);
      if (h[0] != 0.f) found = k;
    }
    printf(" %d", found);
  }
  printf("\n");
  return 0;
}
