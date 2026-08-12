/* Copyright 2025 CMU
 *
 * Numerical test for the MXFP8 MoE gang kernels
 * (gang_moe_w13_linear_mxfp8_kernel / gang_moe_w2_linear_mxfp8_kernel in
 * tasks/mi300/gang_moe_linear_mxfp8_mi300.cuh), both with their fused
 * epilogues on -- SwiGLU for W13, topk-weight + f32 atomicAdd for W2, since
 * that is the only configuration GLM-4.7-Flash uses.
 *
 * The GEMM is checked against a scalar dequantize-and-sum over the very same
 * packed bytes and the same LDS-quantized activation, so quantization error is
 * common to both sides and only a layout, routing, or pipeline fault shows.
 * The epilogues are then applied on the host, which is what makes the pairwise
 * gate/up interleave and the routing-weight scaling testable at all -- they are
 * the two places this port could silently produce plausible garbage.
 *
 * The XCD round-robin is exercised for real: _gang_moe_get_xcd_id is stubbed to
 * blockIdx.y and the grid is (tiles_per_xcd, 8), which is exactly the shape the
 * megakernel dispatches -- 8 gangs, one per XCD, each stepping its own tile
 * index. Letting the real s_getreg run would sample whichever XCD the block
 * happened to land on and test nothing.
 *
 * Build: ./build.sh
 * Run:   ./test_mxfp8_moe
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <vector>

// See the note in test_mxfp8_linear.hip. MPK_WS_WAVE_SYNC is barrier-watchdog
// telemetry, already a no-op outside the megakernel.
#define MPK_WS_WAVE_SYNC(wave) ((void)0)
namespace kernel {
__device__ __forceinline__ int _gang_moe_get_xcd_id() {
  return (int)blockIdx.y;
}
} // namespace kernel

#include "mirage/persistent_kernel/tasks/mi300/gang_moe_linear_mxfp8_mi300.cuh"

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

// GLM-4.7-Flash MoE geometry, with the expert count cut to keep the test's
// weights under a gigabyte. Nothing in the kernel depends on NUM_EXPERTS beyond
// the mask's sentinel slot, and 16 is still enough that expert_id and the
// activated-expert index differ.
static constexpr int HIDDEN = 2048;
static constexpr int INTER = 1536;
static constexpr int NUM_EXPERTS = 16;
static constexpr int NUM_TOPK = 5; // top-4 routed + 1 shared
static constexpr int BATCH = 1;
static constexpr int OPW = 64;

static constexpr int W13_N = 2 * INTER; // 3072
static constexpr int W13_K = HIDDEN;
static constexpr int W13_WGS = W13_N / OPW; // 48
static constexpr int W13_TPE = BATCH * W13_WGS;

static constexpr int W2_N = HIDDEN;
static constexpr int W2_K = INTER;
static constexpr int W2_WGS = W2_N / OPW; // 32
static constexpr int W2_TPE = BATCH * W2_WGS;

static constexpr int SMEM_BYTES = W13_K + W13_K / 128 + 256;

// Packed size for an [E, N, K] weight at OPW rows per workgroup.
static constexpr size_t wg_bytes(int k) {
  return (size_t)OPW * k + (size_t)OPW * (k / 32);
}

// ── weight packing ─────────────────────────────────────────────────────────
// One block per (expert, output row), one thread per 32-element scale block.
// Same layout as pack_weights in test_mxfp8_linear.hip, extended with the
// expert-major outer dimension.
__global__ void pack_weights(unsigned short const *src,
                             uint8_t *out,
                             int N,
                             int K) {
  int const e = blockIdx.y;
  int const n = blockIdx.x;
  int const b = threadIdx.x;
  int const blocks32 = K / 32;
  if (b >= blocks32) {
    return;
  }
  size_t const wgb = (size_t)OPW * K + (size_t)OPW * blocks32;
  size_t const data_bytes = (size_t)OPW * K;
  int const wg = n / OPW, row = n % OPW;

  uint8_t *base = out + (size_t)e * (N / OPW) * wgb + (size_t)wg * wgb;
  uint8_t *data = base + (size_t)row * K;
  uint8_t *scales = base + data_bytes + (size_t)row * blocks32;

  unsigned short const *srow = src + ((size_t)e * N + n) * K + b * 32;
  float vals[32];
  float amax = 0.0f;
  for (int j = 0; j < 32; j++) {
    vals[j] = _gang_bf16_to_float(srow[j]);
    amax = fmaxf(amax, fabsf(vals[j]));
  }
  uint8_t se = _gang_compute_e8m0_fp8(amax);
  scales[b] = se;
  float scale_f = 1.0f;
  if (se != 0) {
    union {
      float f;
      uint32_t u;
    } sv;
    sv.u = (uint32_t)se << 23;
    scale_f = sv.f;
  }
  for (int j = 0; j < 32; j += 4) {
    fp8x4_t pk = _gang_quant_4xfp8(
        vals[j], vals[j + 1], vals[j + 2], vals[j + 3], scale_f);
    *(int *)(data + b * 32 + j) = *(int const *)&pk;
  }
}

// ── scalar reference over the same bytes ───────────────────────────────────

__device__ __forceinline__ float fp8_to_f32(uint8_t b) {
  return __builtin_amdgcn_cvt_f32_fp8((int)b, 0);
}

__device__ __forceinline__ float e8m0_to_f32(uint8_t se) {
  if (se == 0) {
    return 1.0f;
  }
  union {
    float f;
    uint32_t u;
  } sv;
  sv.u = (uint32_t)se << 23;
  return sv.f;
}

// One block per activated expert. Quantizes that expert's input row exactly as
// the kernel does, then sums in plain K order. Output is the raw GEMM, before
// bias and before either epilogue.
//
// `absum` receives sum|product| per output. That is the quantity the
// accumulation error rides on, and for a dot product that cancels it can be
// orders of magnitude above |out| -- deriving the tolerance from the output
// magnitude instead, as the dense linear test does, is only safe when the
// terms and the result are the same size, which here they are not.
//
// `flushed` receives the same sum with every product involving an E4M3
// denormal dropped. It is not a reference; it measures how much of the result
// sits at the very bottom of the format, which is what sets the floor on any
// comparison against the MFMA (see MFMA_EPS).
template <int K, int N, bool NT>
__global__ void reference_kernel(unsigned short const *input,
                                 uint8_t const *w,
                                 int const *mask,
                                 int const *routing,
                                 float *out,
                                 float *absum,
                                 float *flushed = nullptr) {
  extern __shared__ char smem[];
  uint8_t *s_tok = (uint8_t *)smem;
  uint8_t *s_scales = s_tok + K;

  int const slot_of_expert = blockIdx.x;
  int const expert_id = mask[slot_of_expert];
  int const topk_slot = routing[expert_id * BATCH] - 1;

  unsigned short const *in_row =
      NT ? input + (size_t)topk_slot * K : input; // W2 reads [topk, K]
  if constexpr (NT) {
    _gang_wave_parallel_fp8_quant_nt<K>(in_row, s_tok, s_scales);
  } else {
    _gang_wave_parallel_fp8_quant<K>(in_row, s_tok, s_scales);
  }

  constexpr int blocks32 = K / 32;
  constexpr size_t wgb = (size_t)OPW * K + (size_t)OPW * blocks32;
  constexpr size_t data_bytes = (size_t)OPW * K;
  uint8_t const *ew = w + (size_t)expert_id * (N / OPW) * wgb;

  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    uint8_t const *base = ew + (size_t)(n / OPW) * wgb;
    uint8_t const *data = base + (size_t)(n % OPW) * K;
    uint8_t const *scales = base + data_bytes + (size_t)(n % OPW) * blocks32;
    // Accumulate in double. Both operands dequantize exactly, so the products
    // are the same fp32 values the MFMA sees and the only thing separating the
    // two results is summation order -- which is exactly what a double
    // reference removes from the comparison instead of adding its own share of.
    double sum = 0.0;
    float mag = 0.0f;
    double norm = 0.0;
    for (int k = 0; k < K; k++) {
      float p = (fp8_to_f32(data[k]) * e8m0_to_f32(scales[k >> 5])) *
                (fp8_to_f32(s_tok[k]) * e8m0_to_f32(s_scales[k >> 7]));
      sum += (double)p;
      mag += fabsf(p);
      // Exponent field zero on either operand: the product is at the bottom of
      // E4M3's range, 2^-9 of a full-scale term or less.
      if ((data[k] & 0x78) != 0 && (s_tok[k] & 0x78) != 0) {
        norm += (double)p;
      }
    }
    out[(size_t)slot_of_expert * N + n] = sum;
    absum[(size_t)slot_of_expert * N + n] = mag;
    if (flushed) {
      flushed[(size_t)slot_of_expert * N + n] = norm;
    }
  }
}

// ── kernel wrappers ────────────────────────────────────────────────────────

__global__ void run_w13(void const *input,
                        void const *weight,
                        void const *routing,
                        void const *mask,
                        void const *bias,
                        void *output) {
  gang_moe_w13_linear_mxfp8_kernel<BATCH,
                                   W13_N,
                                   W13_N,
                                   W13_K,
                                   NUM_EXPERTS,
                                   NUM_TOPK,
                                   W13_TPE,
                                   OPW,
                                   true /*FUSE_SWIGLU*/>(
      input, weight, routing, mask, bias, output, (int)blockIdx.x);
}

template <bool FUSE>
__global__ void run_w2(void const *input,
                       void const *weight,
                       void const *routing,
                       void const *mask,
                       void const *bias,
                       void *output,
                       void const *routing_weight) {
  gang_moe_w2_linear_mxfp8_kernel<BATCH,
                                  W2_N,
                                  W2_N,
                                  W2_K,
                                  NUM_EXPERTS,
                                  NUM_TOPK,
                                  W2_TPE,
                                  OPW,
                                  FUSE>(input,
                                        weight,
                                        routing,
                                        mask,
                                        bias,
                                        output,
                                        (int)blockIdx.x,
                                        routing_weight);
}

// ── host helpers ───────────────────────────────────────────────────────────

static unsigned short f2bf(float f) {
  union {
    float f;
    unsigned u;
  } v;
  v.f = f;
  unsigned bias = ((v.u >> 16) & 1) + 0x7FFF;
  return (unsigned short)((v.u + bias) >> 16);
}

static float bf2f(unsigned short b) {
  union {
    float f;
    unsigned u;
  } v;
  v.u = (unsigned)b << 16;
  return v.f;
}

static float frand() {
  return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
}

static float host_silu(float x) {
  return x / (1.0f + expf(-x));
}

// Per-element budget: a relative term for the bf16 store plus an absolute term
// scaled by the measured sum|product| rather than by the output magnitude.
//
//   REL_TOL   2^-8, for the bf16 store. bf16 rounds at 2^-9; the extra bit
//             covers the reference summing the same products in a different
//             order. Passed as `store_rel` and set to 0 where the result stays
//             fp32, so the f32 workspace is not handed an allowance it cannot
//             use.
//   MFMA_EPS  3e-5 per unit of sum|product|. Not an fp32 accumulation bound --
//             fp32 order noise was measured at 6e-8*sum|product| here, three
//             orders below what the MFMA actually does. The real floor is the
//             bottom of E4M3: the smallest denormal is 2^-9 of a full-scale
//             value, so any disagreement between the MFMA's fp8 decode and
//             __builtin_amdgcn_cvt_f32_fp8 on those terms lands at ~1e-5 of
//             sum|product|, and dropping the denormal terms from the reference
//             outright moves it by 2.0e-5 (printed below). 3e-5 sits just
//             above that.
//
// This is loose in absolute terms and still sharp for its purpose: the operand
// layout probe (test_mxfp8_mfma_layout) puts every wrong data or scale grouping
// at 1.2e-1 to 2.8e-1 on this same scale, and dropping or duplicating a single
// 128-wide k-tile would show at ~7e-3. Both are hundreds of times above the
// budget.
static double REL_TOL = 1.0 / 256.0;
static double MFMA_EPS = 3.0e-5;

static double check(std::vector<double> const &want,
                    std::vector<double> const &got,
                    std::vector<double> const &absum,
                    double store_rel,
                    double *out_worst) {
  double worst = 0.0, budget_at_worst = 0.0;
  for (size_t i = 0; i < want.size(); i++) {
    double const budget = store_rel * fabs(want[i]) + MFMA_EPS * absum[i];
    double const d = fabs(got[i] - want[i]) / budget;
    if (d > worst) {
      worst = d;
      budget_at_worst = budget;
    }
  }
  *out_worst = worst;
  return budget_at_worst;
}

int main() {
  srand(20260812);

  // Routing: 4 routed experts at arbitrary ids plus the shared expert in the
  // last slot with weight 1.0, which is how demo.py lays GLM's out.
  int const num_act = NUM_TOPK;
  int const act_experts[NUM_TOPK] = {3, 11, 0, 14, 7};
  std::vector<int> h_mask(NUM_EXPERTS + 1, 0), h_routing(NUM_EXPERTS * BATCH, 0);
  std::vector<float> h_rw(BATCH * NUM_TOPK);
  for (int s = 0; s < num_act; s++) {
    h_mask[s] = act_experts[s];
    h_routing[act_experts[s] * BATCH + 0] = s + 1;
    h_rw[s] = (s == NUM_TOPK - 1) ? 1.0f : (0.1f + 0.2f * s);
  }
  h_mask[NUM_EXPERTS] = num_act;

  // ── W13 ──────────────────────────────────────────────────────────────────
  // Rows are pairwise interleaved: row 2j is gate_j, row 2j+1 is up_j. Give
  // gate and up visibly different magnitudes so a swapped pair cannot pass.
  std::vector<unsigned short> h_x(W13_K), h_w13((size_t)NUM_EXPERTS * W13_N * W13_K),
      h_b13((size_t)NUM_EXPERTS * W13_N);
  for (int k = 0; k < W13_K; k++) {
    h_x[k] = f2bf(frand());
  }
  for (int e = 0; e < NUM_EXPERTS; e++) {
    for (int n = 0; n < W13_N; n++) {
      float mag = (n & 1) ? 0.25f : 1.0f;                 // up vs gate
      mag *= exp2f((float)((n / 2) % 5) - 2.0f);          // per-pair range
      for (int k = 0; k < W13_K; k++) {
        h_w13[((size_t)e * W13_N + n) * W13_K + k] =
            f2bf(frand() * mag * 0.05f);
      }
      h_b13[(size_t)e * W13_N + n] = f2bf(frand() * 0.05f);
    }
  }

  unsigned short *d_x, *d_w13_bf16, *d_b13, *d_act;
  uint8_t *d_w13;
  float *d_ref13;
  int *d_mask, *d_routing;
  float *d_rw;
  size_t const w13_packed = (size_t)NUM_EXPERTS * W13_WGS * wg_bytes(W13_K);
  HIP_CHECK(hipMalloc(&d_x, W13_K * 2));
  HIP_CHECK(hipMalloc(&d_w13_bf16, h_w13.size() * 2));
  HIP_CHECK(hipMalloc(&d_b13, h_b13.size() * 2));
  HIP_CHECK(hipMalloc(&d_w13, w13_packed));
  HIP_CHECK(hipMalloc(&d_act, (size_t)BATCH * NUM_TOPK * INTER * 2));
  HIP_CHECK(hipMalloc(&d_ref13, (size_t)num_act * W13_N * sizeof(float)));
  float *d_abs13;
  HIP_CHECK(hipMalloc(&d_abs13, (size_t)num_act * W13_N * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_mask, h_mask.size() * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_routing, h_routing.size() * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_rw, h_rw.size() * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), W13_K * 2, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(
      d_w13_bf16, h_w13.data(), h_w13.size() * 2, hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_b13, h_b13.data(), h_b13.size() * 2, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(
      d_mask, h_mask.data(), h_mask.size() * sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_routing,
                      h_routing.data(),
                      h_routing.size() * sizeof(int),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(
      d_rw, h_rw.data(), h_rw.size() * sizeof(float), hipMemcpyHostToDevice));
  // Poison the activation buffer: a tile that never runs must show up as a
  // mismatch rather than as a zero that happens to look plausible.
  HIP_CHECK(hipMemset(d_act, 0x7F, (size_t)BATCH * NUM_TOPK * INTER * 2));

  pack_weights<<<dim3(W13_N, NUM_EXPERTS), W13_K / 32>>>(
      d_w13_bf16, d_w13, W13_N, W13_K);
  HIP_CHECK(hipDeviceSynchronize());

  reference_kernel<W13_K, W13_N, false><<<num_act, 256, SMEM_BYTES>>>(
      d_x, d_w13, d_mask, d_routing, d_ref13, d_abs13);

  int const w13_tiles_per_xcd = (num_act * W13_TPE + 7) / 8;
  run_w13<<<dim3(w13_tiles_per_xcd, 8), 256, SMEM_BYTES>>>(
      d_x, d_w13, d_routing, d_mask, d_b13, d_act);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> ref13((size_t)num_act * W13_N),
      abs13v((size_t)num_act * W13_N);
  std::vector<unsigned short> act((size_t)BATCH * NUM_TOPK * INTER);
  HIP_CHECK(hipMemcpy(ref13.data(),
                      d_ref13,
                      ref13.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(abs13v.data(),
                      d_abs13,
                      abs13v.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(
      hipMemcpy(act.data(), d_act, act.size() * 2, hipMemcpyDeviceToHost));

  std::vector<double> want13, got13, absum13;
  for (int s = 0; s < num_act; s++) {
    int const e = act_experts[s];
    for (int j = 0; j < INTER; j++) {
      float const gate = ref13[(size_t)s * W13_N + 2 * j] +
                         bf2f(h_b13[(size_t)e * W13_N + 2 * j]);
      float const up = ref13[(size_t)s * W13_N + 2 * j + 1] +
                       bf2f(h_b13[(size_t)e * W13_N + 2 * j + 1]);
      want13.push_back(host_silu(gate) * up);
      got13.push_back(bf2f(act[(size_t)s * INTER + j]));
      // Propagate the two GEMM tolerances through silu(g)*u:
      //   d/dg = silu'(g)*u,  |silu'| <= 1.1;   d/du = silu(g).
      absum13.push_back(1.1 * fabs(up) * abs13v[(size_t)s * W13_N + 2 * j] +
                        fabs(host_silu(gate)) *
                            abs13v[(size_t)s * W13_N + 2 * j + 1]);
    }
  }
  double worst13;
  double abs13 = check(want13, got13, absum13, REL_TOL, &worst13);

  // ── W2 ───────────────────────────────────────────────────────────────────
  // Feed W2 the activation the kernel just produced, so the test covers the
  // [batch, topk, K] indexing the real graph uses rather than a fresh buffer.
  std::vector<unsigned short> h_w2((size_t)NUM_EXPERTS * W2_N * W2_K),
      h_b2((size_t)NUM_EXPERTS * W2_N);
  for (int e = 0; e < NUM_EXPERTS; e++) {
    for (int n = 0; n < W2_N; n++) {
      float mag = exp2f((float)(n % 7) - 3.0f);
      for (int k = 0; k < W2_K; k++) {
        h_w2[((size_t)e * W2_N + n) * W2_K + k] = f2bf(frand() * mag * 0.05f);
      }
      h_b2[(size_t)e * W2_N + n] = f2bf(frand() * 0.05f);
    }
  }

  unsigned short *d_w2_bf16, *d_b2, *d_slab;
  uint8_t *d_w2;
  float *d_ref2, *d_ws;
  size_t const w2_packed = (size_t)NUM_EXPERTS * W2_WGS * wg_bytes(W2_K);
  HIP_CHECK(hipMalloc(&d_w2_bf16, h_w2.size() * 2));
  HIP_CHECK(hipMalloc(&d_b2, h_b2.size() * 2));
  HIP_CHECK(hipMalloc(&d_w2, w2_packed));
  HIP_CHECK(hipMalloc(&d_ref2, (size_t)num_act * W2_N * sizeof(float)));
  float *d_abs2;
  HIP_CHECK(hipMalloc(&d_abs2, (size_t)num_act * W2_N * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ws, (size_t)BATCH * W2_N * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_slab, (size_t)BATCH * NUM_TOPK * W2_N * 2));
  HIP_CHECK(hipMemcpy(
      d_w2_bf16, h_w2.data(), h_w2.size() * 2, hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_b2, h_b2.data(), h_b2.size() * 2, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_ws, 0, (size_t)BATCH * W2_N * sizeof(float)));
  HIP_CHECK(hipMemset(d_slab, 0x7F, (size_t)BATCH * NUM_TOPK * W2_N * 2));

  pack_weights<<<dim3(W2_N, NUM_EXPERTS), W2_K / 32>>>(
      d_w2_bf16, d_w2, W2_N, W2_K);
  HIP_CHECK(hipDeviceSynchronize());

  float *d_flush2;
  HIP_CHECK(hipMalloc(&d_flush2, (size_t)num_act * W2_N * sizeof(float)));
  reference_kernel<W2_K, W2_N, true><<<num_act, 256, SMEM_BYTES>>>(
      d_act, d_w2, d_mask, d_routing, d_ref2, d_abs2, d_flush2);

  // Run both epilogues. The unfused slab pins the GEMM per expert, so if the
  // fused number drifts it is the reduction and not the operand layout; with
  // only the fused run, five cancelling terms make those two indistinguishable.
  int const w2_tiles_per_xcd = (num_act * W2_TPE + 7) / 8;
  run_w2<false><<<dim3(w2_tiles_per_xcd, 8), 256, SMEM_BYTES>>>(
      d_act, d_w2, d_routing, d_mask, d_b2, d_slab, nullptr);
  run_w2<true><<<dim3(w2_tiles_per_xcd, 8), 256, SMEM_BYTES>>>(
      d_act, d_w2, d_routing, d_mask, d_b2, d_ws, d_rw);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> ref2((size_t)num_act * W2_N), abs2v((size_t)num_act * W2_N),
      ws(W2_N);
  std::vector<unsigned short> slab((size_t)BATCH * NUM_TOPK * W2_N);
  HIP_CHECK(hipMemcpy(
      ref2.data(), d_ref2, ref2.size() * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(abs2v.data(),
                      d_abs2,
                      abs2v.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(
      hipMemcpy(ws.data(), d_ws, W2_N * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(
      hipMemcpy(slab.data(), d_slab, slab.size() * 2, hipMemcpyDeviceToHost));

  std::vector<double> want2u, got2u, absum2u;
  for (int s = 0; s < num_act; s++) {
    int const e = act_experts[s];
    for (int n = 0; n < W2_N; n++) {
      want2u.push_back((double)ref2[(size_t)s * W2_N + n] +
                       bf2f(h_b2[(size_t)e * W2_N + n]));
      got2u.push_back(bf2f(slab[(size_t)s * W2_N + n]));
      absum2u.push_back(abs2v[(size_t)s * W2_N + n]);
    }
  }
  double worst2u;
  double abs2u = check(want2u, got2u, absum2u, REL_TOL, &worst2u);

  // How much of the dot product lives in E4M3 denormals. This is the floor
  // MFMA_EPS has to clear -- see the comment there.
  std::vector<float> flush2((size_t)num_act * W2_N);
  HIP_CHECK(hipMemcpy(flush2.data(),
                      d_flush2,
                      flush2.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  double den_frac = 0.0;
  for (size_t i = 0; i < flush2.size(); i++) {
    den_frac =
        fmax(den_frac, fabs((double)flush2[i] - (double)ref2[i]) / absum2u[i]);
  }

  // The fused path lands in an f32 workspace, so there is no bf16 store to pay
  // for -- store_rel is 0 and the whole budget is the routing-weighted sum of
  // the five experts' sum|product|.
  std::vector<double> want2, got2, absum2;
  for (int n = 0; n < W2_N; n++) {
    double sum = 0.0, mag = 0.0;
    for (int s = 0; s < num_act; s++) {
      int const e = act_experts[s];
      sum += ((double)ref2[(size_t)s * W2_N + n] +
              bf2f(h_b2[(size_t)e * W2_N + n])) *
             h_rw[s];
      mag += h_rw[s] * abs2v[(size_t)s * W2_N + n];
    }
    want2.push_back(sum);
    got2.push_back(ws[n]);
    absum2.push_back(mag);
  }
  double worst2;
  double abs2 = check(want2, got2, absum2, 0.0, &worst2);

  bool const ok = worst13 < 1.0 && worst2u < 1.0 && worst2 < 1.0;

  printf("MoE MXFP8, %d experts (%d activated), topk %d, batch %d\n",
         NUM_EXPERTS,
         num_act,
         NUM_TOPK,
         BATCH);
  printf("  W13 K=%d N=%d  %d tiles over 8 XCDs (%d each)\n",
         W13_K,
         W13_N,
         num_act * W13_TPE,
         w13_tiles_per_xcd);
  printf("      +SwiGLU     vs exact-dequant : %.2fx budget %s  (abs %.3e)\n",
         worst13,
         worst13 < 1.0 ? "OK" : "FAIL",
         abs13);
  printf("  W2  K=%d N=%d  %d tiles over 8 XCDs (%d each)\n",
         W2_K,
         W2_N,
         num_act * W2_TPE,
         w2_tiles_per_xcd);
  printf("      unfused     vs exact-dequant : %.2fx budget %s  (abs %.3e)\n",
         worst2u,
         worst2u < 1.0 ? "OK" : "FAIL",
         abs2u);
  printf("      +MulSumAdd  vs exact-dequant : %.2fx budget %s  (abs %.3e,"
         " f32 out: no bf16 term)\n",
         worst2,
         worst2 < 1.0 ? "OK" : "FAIL",
         abs2);
  printf("  budget = %.3e*|want| + %.3e*sum|product|; E4M3 denormals carry"
         " %.3e of\n           sum|product| here, which is the floor that"
         " sets the second term\n",
         REL_TOL,
         MFMA_EPS,
         den_frac);
  printf("\n%s\n", ok ? "[PASS]" : "[FAIL]");
  return ok ? 0 : 1;
}
