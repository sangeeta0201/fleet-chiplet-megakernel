// W13 at OUTPUT_PER_WG = 32 (K_HALF: two 16-row groups x two halves of the
// reduction, joined through LDS) against the shipping N-parallel OPW = 64, in
// gang_moe_w13_linear_mxfp8_kernel, at MPK_MOE_TP's shape: 512 output rows
// (one rank's gate/up slice), K = 6144, 9 activated slots, MXFP4 K-major.
//
// Both widths run on the SAME logical weights, each packed in its own
// workgroup layout. The data makes every partial sum exact in f32 -- FP4
// weights with E8M0 scales 2^-1..2^0, activations +-2^j for j in [-2, 0], so
// products are multiples of 2^-4 and |sum| < 2^16 -- which makes the two
// reduction orders produce the SAME pre-SwiGLU sums. So the two outputs must
// be bit-identical, and both are compared with an f64 SwiGLU reference at a
// bf16 + fast_silu tolerance.
//
//   hipcc -O3 --offload-arch=gfx950 -I../../include \
//     -I../../include/mirage/persistent_kernel -DMIRAGE_BACKEND_USE_ROCM \
//     test_moe_w13_khalf.hip -o test_moe_w13_khalf
//   HIP_VISIBLE_DEVICES=0 ./test_moe_w13_khalf

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <vector>

#define MPK_WS_WAVE_SYNC(wave) ((void)0)
namespace kernel {
__device__ __forceinline__ int _gang_moe_get_xcd_id() {
  return (int)blockIdx.y;
}
} // namespace kernel

#include "mirage/persistent_kernel/mpk_atoms.cuh"
#include "mirage/persistent_kernel/tasks/mi300/gang_moe_linear_mxfp8_mi300.cuh"

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t _e = (call);                                                    \
    if (_e != hipSuccess) {                                                    \
      fprintf(stderr, "HIP %s:%d: %s\n", __FILE__, __LINE__,                   \
              hipGetErrorString(_e));                                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

using namespace kernel;

static_assert(MPK_MOE_KMAJOR == 2, "the host packer below is level 2");

static constexpr int K = 6144;
static constexpr int N = 512;          // gate/up interleaved rows
static constexpr int NUM_EXPERTS = 40; // mask sentinel slot
static constexpr int NUM_TOPK = 9;
static constexpr int ROW_BYTES = K / 2;
static constexpr int BLK32 = K / 32;
static constexpr int KT = K / 128;

static inline uint32_t mix(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

static float e2m1(uint32_t nib) {
  static float const mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  float const v = mag[nib & 7];
  return (nib & 8) ? -v : v;
}

template <int OPW>
__global__ void __launch_bounds__(256, 1)
    run_w13(void const *input,
            void const *weight,
            void const *routing,
            void const *mask,
            void const *bias,
            void *out) {
  gang_moe_w13_linear_mxfp8_kernel<1, N, N, K, NUM_EXPERTS, NUM_TOPK,
                                   N / OPW, OPW,
                                   /*FUSE_SWIGLU=*/true,
                                   /*WRITE_THROUGH=*/true,
                                   /*WEIGHT_FP4=*/true>(
      input, weight, routing, mask, bias, out, blockIdx.x);
}

// [E][WGS][OPW rows K-major data | OPW rows K-major scales], level 2.
template <int OPW>
static std::vector<uint8_t> pack(std::vector<uint8_t> const &nib,
                                 std::vector<uint8_t> const &sc) {
  constexpr size_t WG_BYTES = (size_t)OPW * ROW_BYTES + (size_t)OPW * BLK32;
  constexpr int WGS = N / OPW;
  std::vector<uint8_t> p((size_t)NUM_EXPERTS * WGS * WG_BYTES);
  for (int e = 0; e < NUM_EXPERTS; e++) {
    for (int n = 0; n < N; n++) {
      int const wg = n / OPW, r = n % OPW;
      uint8_t *base = p.data() + ((size_t)e * WGS + wg) * WG_BYTES;
      for (int b = 0; b < ROW_BYTES; b++) {
        uint8_t const v = nib[((size_t)e * N + n) * ROW_BYTES + b];
        base[(r / 16) * (16 * ROW_BYTES) + (b / 64) * 1024 + (r % 16) * 64 +
             b % 64] = (uint8_t)(v | (v << 4));
      }
      for (int s = 0; s < BLK32; s++) {
        base[(size_t)OPW * ROW_BYTES + (r / 16) * (16 * BLK32) +
             (s / 4) * 64 + (r % 16) * 4 + s % 4] =
            sc[((size_t)e * N + n) * BLK32 + s];
      }
    }
  }
  return p;
}

template <int OPW>
static std::vector<unsigned short>
    run_one(std::vector<uint8_t> const &packed, void *d_in, void *d_route,
            void *d_mask, void *d_bias, int n_live) {
  void *d_w, *d_out;
  size_t const out_elems = (size_t)NUM_TOPK * (N / 2);
  HIP_CHECK(hipMalloc(&d_w, packed.size()));
  HIP_CHECK(hipMemcpy(d_w, packed.data(), packed.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&d_out, out_elems * 2));
  HIP_CHECK(hipMemset(d_out, 0xff, out_elems * 2));
  int const tiles = n_live * (N / OPW);
  run_w13<OPW><<<dim3((tiles + 7) / 8, 8), 256, 16384>>>(
      d_in, d_w, d_route, d_mask, d_bias, d_out);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<unsigned short> out(out_elems);
  HIP_CHECK(hipMemcpy(out.data(), d_out, out_elems * 2,
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_out));
  return out;
}

static float bf16_to_f32(unsigned short h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

static bool run_case(int n_live, uint32_t seed) {
  std::vector<uint8_t> nib((size_t)NUM_EXPERTS * N * ROW_BYTES);
  std::vector<uint8_t> sc((size_t)NUM_EXPERTS * N * BLK32);
  for (size_t i = 0; i < nib.size(); i++) {
    nib[i] = (uint8_t)(mix((uint32_t)i * 2654435761u ^ seed) & 15u);
  }
  for (size_t i = 0; i < sc.size(); i++) {
    sc[i] = (uint8_t)(126u + (mix((uint32_t)i ^ (seed * 7919u)) & 1u));
  }
  std::vector<float> act(K);
  std::vector<unsigned short> act_bf16(K);
  for (int k = 0; k < K; k++) {
    uint32_t const r = mix((uint32_t)k ^ (seed * 104729u));
    float const v = ldexpf(1.0f, -(int)(r % 3));
    act[k] = (r & 0x100) ? -v : v;
    uint32_t u;
    memcpy(&u, &act[k], 4);
    act_bf16[k] = (unsigned short)(u >> 16);
  }
  std::vector<int> experts;
  for (uint32_t t = 0; (int)experts.size() < n_live; t++) {
    int const e = (int)(mix(seed * 31u + t) % NUM_EXPERTS);
    bool dup = false;
    for (int x : experts) dup |= (x == e);
    if (!dup) experts.push_back(e);
  }
  std::vector<int> mask(NUM_EXPERTS + 1, 0);
  std::vector<int> routing(NUM_EXPERTS, 0);
  for (int s = 0; s < n_live; s++) {
    mask[s] = experts[s];
    routing[experts[s]] = (s * 4 + 3) % NUM_TOPK + 1;
  }
  mask[NUM_EXPERTS] = n_live;

  void *d_in, *d_route, *d_mask, *d_bias;
  HIP_CHECK(hipMalloc(&d_in, K * 2));
  HIP_CHECK(hipMemcpy(d_in, act_bf16.data(), K * 2, hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&d_route, routing.size() * 4));
  HIP_CHECK(hipMemcpy(d_route, routing.data(), routing.size() * 4,
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&d_mask, mask.size() * 4));
  HIP_CHECK(hipMemcpy(d_mask, mask.data(), mask.size() * 4,
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&d_bias, (size_t)NUM_EXPERTS * N * 2));
  HIP_CHECK(hipMemset(d_bias, 0, (size_t)NUM_EXPERTS * N * 2));

  auto o64 = run_one<64>(pack<64>(nib, sc), d_in, d_route, d_mask, d_bias,
                         n_live);
  auto o32 = run_one<32>(pack<32>(nib, sc), d_in, d_route, d_mask, d_bias,
                         n_live);

  int mism = 0, bad = 0;
  double worst = 0.0;
  for (int s = 0; s < n_live; s++) {
    int const e = experts[s];
    int const slot = routing[e] - 1;
    for (int j = 0; j < N / 2; j++) {
      double sum[2] = {0.0, 0.0};
      for (int h = 0; h < 2; h++) {
        size_t const row = (size_t)e * N + 2 * j + h;
        for (int k = 0; k < K; k++) {
          sum[h] += (double)e2m1(nib[row * ROW_BYTES + k / 2]) *
                    ldexp(1.0, (int)sc[row * BLK32 + k / 32] - 127) *
                    (double)act[k];
        }
      }
      double const ref = sum[0] / (1.0 + exp(-sum[0])) * sum[1];
      size_t const idx = (size_t)slot * (N / 2) + j;
      mism += (o64[idx] != o32[idx]);
      double const got = bf16_to_f32(o32[idx]);
      double const rel = fabs(got - ref) / (fabs(ref) + 1.0);
      worst = rel > worst ? rel : worst;
      bad += (rel > 2e-2);
    }
  }
  bool const pass = (mism == 0 && bad == 0);
  printf("[%s] n_live=%d seed=%u  OPW32 vs OPW64 mismatches %d/%d, vs f64 "
         "worst rel %.2e, bad %d\n",
         pass ? "PASS" : "FAIL", n_live, seed, mism, n_live * N / 2, worst,
         bad);
  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_route));
  HIP_CHECK(hipFree(d_mask));
  HIP_CHECK(hipFree(d_bias));
  return pass;
}

int main() {
  bool ok = true;
  ok &= run_case(9, 1);
  ok &= run_case(3, 2);
  printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
