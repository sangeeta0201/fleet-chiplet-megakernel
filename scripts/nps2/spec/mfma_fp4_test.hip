// Does mfma_scale_f32_16x16x128_f8f6f4 (A = FP4 weights, B = FP8 activation) give the right dot products
// for naturally packed MXFP4 rows (value 2j in the low nibble of byte j, one E8M0 scale per 32 contiguous
// values) when loaded the way the target kernels load them:
//   A: lane (col = l & 15, g = l >> 4): 16 bytes of row col at byte g * 16 of the 64-byte K tile, scale = block g
//   B: two 16-byte chunks of the natural-order FP8 row at g * 16 and 64 + g * 16, one scale per 128 values
// Prints the max error against a host reference for natural packing, and the pairing of A nibbles with
// B positions (single-hot probe) so a permutation can be read off if natural packing is wrong.
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

typedef int i32x4_t __attribute__((ext_vector_type(4)));
typedef int i32x8_t __attribute__((ext_vector_type(8)));
typedef float f32x4_t __attribute__((ext_vector_type(4)));

__global__ void tile(unsigned char const *w, unsigned char const *ws, unsigned char const *x8, int sb_val,
                     float *out) {
  int const lane = threadIdx.x & 63, col = lane & 15, g = lane >> 4;
  i32x4_t lo4;
  __builtin_memcpy(&lo4, w + col * 64 + g * 16, 16);
  i32x8_t a = {lo4[0], lo4[1], lo4[2], lo4[3], 0, 0, 0, 0};
  int const sa = ws[col * 4 + g];
  i32x4_t blo, bhi;
  __builtin_memcpy(&blo, x8 + g * 16, 16);
  __builtin_memcpy(&bhi, x8 + 64 + g * 16, 16);
  i32x8_t b = {blo[0], blo[1], blo[2], blo[3], bhi[0], bhi[1], bhi[2], bhi[3]};
  f32x4_t acc = {0.f, 0.f, 0.f, 0.f};
  int sbv;  // force a VGPR: an SGPR scale operand is read as 0 (measured)
  asm volatile("v_mov_b32 %0, %1" : "=v"(sbv) : "s"(sb_val));
  acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, acc, 4, 0, 0, sa, 0, sbv);
  if (col == 0)
    for (int i = 0; i < 4; ++i) out[g * 4 + i] = acc[i];
}

static float const kGrid[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
static float fp4(int code) { return (code & 8 ? -1.f : 1.f) * kGrid[code & 7]; }
// e4m3 encode of small exactly representable values (|v| in {0, 0.5, 1, 1.5, 2, 3, 4})
static unsigned char e4m3(float v) {
  if (v == 0.f) return 0;
  unsigned char s = v < 0 ? 0x80 : 0;
  float a = fabsf(v);
  int e = (int)floorf(log2f(a));
  float m = a / ldexpf(1.f, e) - 1.f;  // 0 or 0.5
  return s | (unsigned char)(((e + 7) & 15) << 3) | (unsigned char)(m * 8);
}

int main() {
  unsigned char *dw, *dws, *dx;
  float *dout;
  hipMalloc(&dw, 16 * 64);
  hipMalloc(&dws, 64);
  hipMalloc(&dx, 128);
  hipMalloc(&dout, 64);
  srand(7);
  // natural packing: random codes, scales 127 +- 2
  int code[16][128];
  unsigned char hw[16 * 64], hws[64], hx[128];
  float xv[128];
  for (int r = 0; r < 16; ++r) {
    for (int k = 0; k < 128; ++k) code[r][k] = rand() & 15;
    for (int j = 0; j < 64; ++j) hw[r * 64 + j] = (unsigned char)(code[r][2 * j] | (code[r][2 * j + 1] << 4));
    for (int b = 0; b < 4; ++b) hws[r * 4 + b] = (unsigned char)(125 + (rand() % 5));
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
  double maxerr = 0;
  for (int r = 0; r < 16; ++r) {
    double ref = 0;
    for (int k = 0; k < 128; ++k) ref += fp4(code[r][k]) * ldexp(1.0, hws[r * 4 + k / 32] - 127) * xv[k];
    maxerr = fmax(maxerr, fabs(ref - h[r]));
    if (r < 3) printf("row %d: mfma %g ref %g\n", r, h[r], ref);
  }
  printf("natural packing: max |err| over 16 rows = %g\n", maxerr);
  // scale variants: all A scales 127 / B scale 127; then A 127 and B 128
  for (int v = 0; v < 3; ++v) {
    unsigned char hws2[64];
    memset(hws2, v == 2 ? 128 : 127, sizeof hws2);
    hipMemcpy(dws, hws2, sizeof hws2, hipMemcpyHostToDevice);
    int const sbv = v == 1 ? 128 : 127;
    hipLaunchKernelGGL(tile, dim3(1), dim3(64), 0, 0, dw, dws, dx, sbv, dout);
    hipMemcpy(h, dout, sizeof h, hipMemcpyDeviceToHost);
    double ref = 0;
    for (int k = 0; k < 128; ++k) ref += fp4(code[0][k]) * xv[k];
    printf("variant %d (A scale %d, B scale %d): mfma %g ref(scales 1) %g\n", v, v == 2 ? 128 : 127, sbv, h[0], ref);
  }

  // single-hot probe: row 0, A nibble t (t = 0..31 of lane group g = 0, i.e. byte t/2) = 1.0 (code 2);
  // B = 1.0 at one natural position k. Report the k that pairs with each t.
  printf("pairing (lane group 0 nibble t -> natural B position k):");
  for (int t = 0; t < 32; ++t) {
    memset(hw, 0, sizeof hw);
    hw[t / 2] = (unsigned char)(t & 1 ? 2 << 4 : 2);
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
      if (h[0] != 0.f) { found = k; if (t < 2) printf(" [t=%d val %g]", t, h[0]); }
    }
    printf(" %d", found);
  }
  printf("\n");
  return 0;
}
