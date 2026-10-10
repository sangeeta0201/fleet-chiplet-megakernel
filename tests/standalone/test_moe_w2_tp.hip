// Numerical test for MPK_MOE_TP's fused-experts W2 tile,
// gang_moe_w2_tp_linear_mxfp8_kernel in gang_moe_linear_mxfp8_mi300.cuh.
//
// One tile reduces over NSEG segments of I_LOCAL, segment s read from expert
// e_s's weight block and from the activation slab of e_s's routing slot. What
// can go wrong is all addressing: the segment pick in the rolled k-loop, the
// expert stride, the K-major row offsets, the slab gather in the quantizer,
// and the zero padding past the live count. So the data is chosen to make the
// arithmetic exact and leave only addressing to test:
//
//   * activations are +-2^j, j in [-2, 2]: any power-of-two E8M0 scale maps
//     them onto E4M3 exactly, so the kernel's quantization is lossless;
//   * weights are MXFP4 with both nibbles of a byte equal, so the result does
//     not depend on the nibble order inside a byte (the model checks that);
//   * the reference sums the same exact products in double.
//
// Only fp32 summation order separates the two, hence a 1e-5 relative gate
// against sum|terms|.
//
// XCD round-robin is real: _gang_moe_get_xcd_id is blockIdx.y and the grid is
// (tiles_per_xcd, 8), the megakernel's dispatch shape.
//
//   hipcc -o test_moe_w2_tp test_moe_w2_tp.hip \
//     -I../../include -I../../include/mirage/persistent_kernel \
//     -I../../deps/json/include -DMIRAGE_BACKEND_USE_ROCM \
//     --offload-arch=gfx950 -O3
//   HIP_VISIBLE_DEVICES=0 ./test_moe_w2_tp

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_bf16.h>
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

// GLM-5 at 8-way TP: hidden 6144, intermediate 2048 / 8 = 256 per rank,
// top-8 + shared = 9 slots, padded to 10 segments (20 k-tiles, depth 4).
static constexpr int HIDDEN = 6144;
static constexpr int NUM_EXPERTS = 40; // mask sentinel slot; ids up to 39
static constexpr int NUM_TOPK = 9;

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

static unsigned short f32_to_bf16_exact(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return (unsigned short)(u >> 16); // exact: +-2^j has a zero low half
}

template <int OPW, int I_LOCAL, int NSEG, int KP>
__global__ void __launch_bounds__(256, 1)
    run_w2_tp(void const *input,
              void const *weight,
              void const *routing,
              void const *mask,
              void *out,
              void const *routing_weight) {
  gang_moe_w2_tp_linear_mxfp8_kernel<1, HIDDEN, I_LOCAL, NUM_EXPERTS, NUM_TOPK,
                                     OPW, /*WEIGHT_FP4=*/true, NSEG, KP>(
      input, weight, routing, mask, out, blockIdx.x, routing_weight);
}

// MPK_ROUTE_LL: the expert ids and routing weights as epoch-tagged words, as
// the fused layer's TopK publishes them.
template <int OPW, int I_LOCAL, int NSEG, int KP>
__global__ void __launch_bounds__(256, 1)
    run_w2_tp_ll(void const *input,
                 void const *weight,
                 void const *routing,
                 void const *mask,
                 void *out,
                 void const *routing_weight,
                 unsigned long long const *ll,
                 unsigned epoch,
                 unsigned long long const *act_ll) {
  gang_moe_w2_tp_linear_mxfp8_kernel<1, HIDDEN, I_LOCAL, NUM_EXPERTS, NUM_TOPK,
                                     OPW, /*WEIGHT_FP4=*/true, NSEG, KP>(
      input, weight, routing, mask, out, blockIdx.x, routing_weight,
      GangW2FoldPush{}, ll, epoch, act_ll);
}

template <int OPW, int I_LOCAL, int NSEG, int KP = 1>
static bool run_case(int n_live, uint32_t seed, bool ll = false,
                     bool act_words = false) {
  constexpr int ROW_BYTES = I_LOCAL / 2; // MXFP4
  constexpr int BLK32 = I_LOCAL / 32;
  constexpr int WGS = HIDDEN / OPW;
  constexpr size_t WG_BYTES = (size_t)OPW * ROW_BYTES + (size_t)OPW * BLK32;
  constexpr size_t EXPERT_BYTES = (size_t)WGS * WG_BYTES;

  // Logical weights: nibble per (expert, row, byte), E8M0 per (row, block).
  std::vector<uint8_t> nib((size_t)NUM_EXPERTS * HIDDEN * ROW_BYTES);
  std::vector<uint8_t> sc((size_t)NUM_EXPERTS * HIDDEN * BLK32);
  for (size_t i = 0; i < nib.size(); i++) {
    nib[i] = (uint8_t)(mix((uint32_t)i * 2654435761u ^ seed) & 15u);
  }
  for (size_t i = 0; i < sc.size(); i++) {
    sc[i] = (uint8_t)(125u + mix((uint32_t)i ^ (seed * 7919u)) % 5u);
  }

  // Pack, K-major level 2, exactly _kmajor_permute's index math:
  //   data  [wg][r/16][kt][r%16][64 B]   scales [wg][r/16][kt][r%16][4 B]
  std::vector<uint8_t> packed((size_t)NUM_EXPERTS * EXPERT_BYTES);
  for (int e = 0; e < NUM_EXPERTS; e++) {
    for (int n = 0; n < HIDDEN; n++) {
      int const wg = n / OPW, r = n % OPW;
      uint8_t *base = packed.data() + (size_t)e * EXPERT_BYTES +
                      (size_t)wg * WG_BYTES;
      for (int b = 0; b < ROW_BYTES; b++) {
        int const kt = b / 64, bb = b % 64;
        uint8_t const v = nib[((size_t)e * HIDDEN + n) * ROW_BYTES + b];
        base[(r / 16) * (16 * ROW_BYTES) + kt * 1024 + (r % 16) * 64 + bb] =
            (uint8_t)(v | (v << 4));
      }
      for (int s = 0; s < BLK32; s++) {
        int const kt = s / 4, ss = s % 4;
        base[(size_t)OPW * ROW_BYTES + (r / 16) * (16 * BLK32) + kt * 64 +
             (r % 16) * 4 + ss] = sc[((size_t)e * HIDDEN + n) * BLK32 + s];
      }
    }
  }

  // Activated experts (distinct, including high ids) and a shuffled slot map.
  std::vector<int> experts;
  for (uint32_t t = 0; (int)experts.size() < n_live; t++) {
    int const e = (int)(mix(seed * 31u + t) % NUM_EXPERTS);
    bool dup = false;
    for (int x : experts) {
      dup |= (x == e);
    }
    if (!dup) {
      experts.push_back(e);
    }
  }
  std::vector<int> mask(NUM_EXPERTS + 1, 0x7fff0000); // stale garbage
  std::vector<int> routing(NUM_EXPERTS, 0);
  for (int s = 0; s < n_live; s++) {
    mask[s] = experts[s];
#if MPK_MOE_BS1_DECODE
    // The one-row TopK's layout, which the fixed-slot decode relies on: every
    // slot live, slot s routed as s + 1.
    routing[experts[s]] = s + 1;
#else
    routing[experts[s]] = (s * 4 + 3) % NUM_TOPK + 1; // a permutation of 1..9
#endif
  }
  mask[NUM_EXPERTS] = n_live;

  // Routing weight per slot, a distinct power of two each so a slot mix-up
  // shows and weighted activations stay exact in E4M3.
  std::vector<float> rw(NUM_TOPK);
  for (int k = 0; k < NUM_TOPK; k++) {
    rw[k] = ldexpf(1.0f, k - 4);
  }

  // Activation slabs [NUM_TOPK][I_LOCAL], +-2^j.
  std::vector<float> act((size_t)NUM_TOPK * I_LOCAL);
  std::vector<unsigned short> act_bf16(act.size());
  for (size_t i = 0; i < act.size(); i++) {
    uint32_t const r = mix((uint32_t)i ^ (seed * 104729u));
    float const v = ldexpf(1.0f, (int)(r % 5) - 2);
    act[i] = (r & 0x100) ? -v : v;
    act_bf16[i] = f32_to_bf16_exact(act[i]);
  }

  // Reference in double over the logical values.
  std::vector<double> ref(HIDDEN, 0.0), mag(HIDDEN, 0.0);
  for (int s = 0; s < n_live; s++) {
    int const e = experts[s];
    int const slot = routing[e] - 1;
    for (int n = 0; n < HIDDEN; n++) {
      for (int k = 0; k < I_LOCAL; k++) {
        size_t const row = (size_t)e * HIDDEN + n;
        double const w = (double)e2m1(nib[row * ROW_BYTES + k / 2]) *
                         ldexp(1.0, (int)sc[row * BLK32 + k / 32] - 127);
        double const p =
            w * (double)act[(size_t)slot * I_LOCAL + k] * (double)rw[slot];
        ref[n] += p;
        mag[n] += fabs(p);
      }
    }
  }

  void *d_in, *d_w, *d_route, *d_mask, *d_out, *d_rw;
  HIP_CHECK(hipMalloc(&d_rw, rw.size() * 4));
  HIP_CHECK(hipMemcpy(d_rw, rw.data(), rw.size() * 4, hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&d_in, act_bf16.size() * 2));
  HIP_CHECK(hipMalloc(&d_w, packed.size()));
  HIP_CHECK(hipMalloc(&d_route, routing.size() * 4));
  HIP_CHECK(hipMalloc(&d_mask, mask.size() * 4));
  HIP_CHECK(hipMalloc(&d_out, HIDDEN * 4));
  HIP_CHECK(hipMemcpy(d_in, act_bf16.data(), act_bf16.size() * 2,
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_w, packed.data(), packed.size(),
                      hipMemcpyHostToDevice));
#ifdef TP_NEGCTL
#if MPK_MOE_BS1_DECODE
#error "TP_NEGCTL corrupts routing_indices, which the fixed-slot decode never reads: build with -DMPK_MOE_BS1_DECODE=0"
#endif
  // Negative control: the kernel reads one expert's activation from the wrong
  // slab. Every case with n_live > 1 must then FAIL.
  std::vector<int> routing_dev = routing;
  if (n_live > 1) {
    routing_dev[experts[0]] = routing[experts[1]];
  }
  HIP_CHECK(hipMemcpy(d_route, routing_dev.data(), routing.size() * 4,
                      hipMemcpyHostToDevice));
#else
  HIP_CHECK(hipMemcpy(d_route, routing.data(), routing.size() * 4,
                      hipMemcpyHostToDevice));
#endif
  HIP_CHECK(hipMemcpy(d_mask, mask.data(), mask.size() * 4,
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_out, 0, HIDDEN * 4));

  constexpr int SMEM = NSEG * I_LOCAL + NSEG * I_LOCAL / 128 + 256;
  unsigned long long *d_ll = nullptr;
  if (ll) {
    // Ids at [s], weights at [16 + s]; the plain weights are zeroed so a
    // kernel that reads them instead produces all-zero rows.
    constexpr unsigned EPOCH = 0x5a5au;
    std::vector<unsigned long long> words(32, 0ull);
    for (int k = 0; k < n_live; k++) {
      uint32_t wbits;
      memcpy(&wbits, &rw[k], 4);
      words[k] = ((unsigned long long)EPOCH << 32) | (uint32_t)experts[k];
      words[16 + k] = ((unsigned long long)EPOCH << 32) | wbits;
    }
    HIP_CHECK(hipMalloc(&d_ll, words.size() * 8));
    HIP_CHECK(hipMemcpy(d_ll, words.data(), words.size() * 8,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_rw, 0, rw.size() * 4));
    // MPK_W2_ACT_LL: the activation slabs as (EPOCH << 32 | bf16x2) words;
    // the bf16 buffer is zeroed so a kernel reading it gets all-zero rows.
    unsigned long long *d_act_ll = nullptr;
    if (act_words) {
      std::vector<unsigned long long> aw(act_bf16.size() / 2);
      for (size_t i = 0; i < aw.size(); i++) {
        aw[i] = ((unsigned long long)EPOCH << 32) |
                (unsigned long long)((uint32_t)act_bf16[2 * i] |
                                     ((uint32_t)act_bf16[2 * i + 1] << 16));
      }
      HIP_CHECK(hipMalloc(&d_act_ll, aw.size() * 8));
      HIP_CHECK(hipMemcpy(d_act_ll, aw.data(), aw.size() * 8,
                          hipMemcpyHostToDevice));
      HIP_CHECK(hipMemset(d_in, 0, act_bf16.size() * 2));
    }
    run_w2_tp_ll<OPW, I_LOCAL, NSEG, KP><<<dim3((WGS * KP + 7) / 8, 8), 256,
                                           SMEM>>>(
        d_in, d_w, d_route, d_mask, d_out, d_rw, d_ll, EPOCH, d_act_ll);
    if (d_act_ll != nullptr) {
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(hipFree(d_act_ll));
    }
  } else {
    run_w2_tp<OPW, I_LOCAL, NSEG, KP><<<dim3((WGS * KP + 7) / 8, 8), 256,
                                        SMEM>>>(
        d_in, d_w, d_route, d_mask, d_out, d_rw);
  }
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> out(HIDDEN);
  HIP_CHECK(hipMemcpy(out.data(), d_out, HIDDEN * 4, hipMemcpyDeviceToHost));

  double worst = 0.0;
  int worst_n = -1, bad = 0;
  for (int n = 0; n < HIDDEN; n++) {
    double const rel = fabs((double)out[n] - ref[n]) / (mag[n] + 1e-30);
    if (rel > worst) {
      worst = rel;
      worst_n = n;
    }
    bad += (rel > 1e-5);
  }
  bool const pass = (bad == 0);
  int const show = worst_n >= 0 ? worst_n : HIDDEN - 1;
  printf("[%s] OPW=%d I_LOCAL=%d NSEG=%d KP=%d n_live=%d seed=%u%s  worst rel %.3e  row %d: gpu %.6f "
         "ref %.6f (sum|p| %.1f)  bad rows %d/%d\n",
         pass ? "PASS" : "FAIL", OPW, I_LOCAL, NSEG, KP, n_live, seed,
         act_words ? " ll+act" : (ll ? " ll" : ""), worst,
         show, out[show],
         ref[show], mag[show], bad, HIDDEN);
  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_route));
  HIP_CHECK(hipFree(d_mask));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_rw));
  if (d_ll != nullptr) {
    HIP_CHECK(hipFree(d_ll));
  }
  return pass;
}

int main() {
  bool ok = true;
  // Under MPK_MOE_BS1_DECODE the live count is NUM_TOPK by contract, so only
  // the full-slot cases apply; build with -DMPK_MOE_BS1_DECODE=0 for the rest.
  constexpr bool ALL = !MPK_MOE_BS1_DECODE;
  // 8 ranks: 2-tile segments, two per depth-4 block.
  ok &= run_case<64, 256, 10>(9, 1);
  if (ALL) {
    ok &= run_case<64, 256, 10>(5, 2);
    ok &= run_case<64, 256, 10>(1, 3);
  }
  ok &= run_case<128, 256, 10>(9, 4);
  if (ALL) {
    ok &= run_case<128, 256, 10>(7, 5);
  }
  // The same, split into two reduction parts (two atomicAdds per row).
  ok &= run_case<64, 256, 10, 2>(9, 8);
  if (ALL) {
    ok &= run_case<64, 256, 10, 2>(4, 9);
  }
  ok &= run_case<128, 256, 10, 2>(9, 10);
  // 2 ranks: 8-tile segments, two blocks per segment.
  ok &= run_case<128, 1024, 9>(9, 6);
  if (ALL) {
    ok &= run_case<64, 1024, 9>(4, 7);
  }
#if MPK_MOE_BS1_DECODE
  // Routing words instead of routing_indices / routing_weight: the padded
  // tenth segment, two reduction parts, and the 2-rank shape.
  ok &= run_case<64, 256, 10>(9, 11, true);
  ok &= run_case<64, 256, 10, 2>(9, 12, true);
  ok &= run_case<128, 1024, 9>(9, 13, true);
  // MPK_W2_ACT_LL: the activations as words too.
  ok &= run_case<64, 256, 10>(9, 14, true, true);
  ok &= run_case<64, 256, 10, 2>(9, 15, true, true);
  ok &= run_case<128, 1024, 9>(9, 16, true, true);
#endif
  printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
