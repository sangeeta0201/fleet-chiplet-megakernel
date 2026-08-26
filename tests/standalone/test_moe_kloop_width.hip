// Task #133: does _gang_moe_kloop_deep compute the RIGHT answer at GROUPS > 4?
//
// The wall said MPK_MOE_PF_GROUPS_W13=8 is worth -0.256 ms, reproducibly to
// 0.001 ms across separate batches. The generated text said it answers "The
// capital of France is" with `<think>aniumaniumanium...` -- 135 of 136 tokens
// one bigram -- on two independent builds, and it hangs 7 of 8 runs. So the
// deep k-loop is WRONG at width 8, and the whole axis (worth ~0.25 ms if the
// fault is found) is blocked on knowing where.
//
// Two things are already eliminated end-to-end:
//   * NOT registers.  worker_kernel is 325 VGPR / ZERO spills at W13 = 4, 6
//     and 8, at MPK_MAX_SEQ_LENGTH 128 and 512 alike.
//   * NOT arithmetic.  The MFMA chain is `acc = mfma(A[j], B[j], acc)` in
//     ascending j at EVERY GROUPS, so a correct deep loop is bit-identical to
//     the shipping one -- not merely close. That makes this test an EQUALITY
//     test, with no tolerance to argue about.
//
// So this file calls the REAL `_gang_moe_kloop` -- the shipping dispatcher out
// of the shipping header, not a re-implementation -- at every width, on one
// shared set of buffers, and diffs the accumulators bit for bit against the
// GROUPS=0 shipping loop. A 15-minute megakernel run per arm becomes a
// 20-second compile plus a millisecond of GPU.
//
// The addressing is what makes this legal as a stand-in. The loop reads
//   w_data_row[kk * W_KS + g*16 (+64)]   and   wg_scales[base + kk*SC_KS + g]
// and both W_KS and SC_KS are compile-time constants INDEPENDENT of GROUPS
// (they come from MPK_MOE_KMAJOR only). Every width therefore touches exactly
// the same bytes in exactly the same buffers; only the ISSUE ORDER and the
// register pressure change. That is precisely the axis under test, and it
// means the buffers do not have to hold a realistically packed weight -- any
// deterministic fill exercises the same hazard.
//
//   hipcc -o test_moe_kloop_width test_moe_kloop_width.hip \
//     -I../../include -I../../include/mirage/persistent_kernel \
//     -I../../deps/json/include -DMIRAGE_BACKEND_USE_ROCM \
//     --offload-arch=gfx950 -O3
//   HIP_VISIBLE_DEVICES=4 ./test_moe_kloop_width
//
// Reads no GPU timing, so it is safe to run alongside a benchmark.

#include <cstdio>
#include <cstdlib>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <vector>

// Same two shims test_mxfp8_linear.hip uses: symbols the MoE header reaches
// for that live in task_header.cuh inside the megakernel.
#define MPK_WS_WAVE_SYNC(wave) ((void)0)
namespace kernel {
__device__ __forceinline__ int _gang_moe_get_xcd_id() {
  int xcd_id;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcd_id));
  return xcd_id;
}
} // namespace kernel

// The write-through stores the MoE epilogue uses. They live in mpk_atoms.cuh,
// which task_header.cuh pulls in ahead of the task headers inside the
// megakernel; nothing under test calls them, but the epilogues still have to
// compile.
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

// GLM-5's W13: hidden 6144 / 128 = 48 k-tiles. The number that matters, since
// the divisors of 48 are what the sweep can reach.
#ifndef KLOOP_MFMA_ITERS
#define KLOOP_MFMA_ITERS 48
#endif
static constexpr int MFMA_ITERS = KLOOP_MFMA_ITERS;
static constexpr bool WEIGHT_FP4 = false;

// Sized off the loop's own strides so a layout change cannot silently make the
// reads run off the end.
static constexpr int W_KS = 16 * 128; // _gang_moe_w_kstride<false>() at KMAJOR>=1
static constexpr int SC_KS = 64;      // _gang_moe_sc_kstride() at KMAJOR>=2
static constexpr size_t W_BYTES = (size_t)MFMA_ITERS * W_KS + 4096;
static constexpr size_t SC_BYTES = (size_t)MFMA_ITERS * SC_KS + 4096;
static constexpr int TOK_BYTES = MFMA_ITERS * 128 + 256;
static constexpr int TOKSC_BYTES = MFMA_ITERS + 64;

// One 64-lane wave, one accumulator of 4 floats per lane.
static constexpr int LANES = 64;
static constexpr int OUT_FLOATS = LANES * 4;

// Deterministic and host-reproducible, so an arm can be re-run in isolation.
__host__ __device__ static inline uint32_t mix(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

// FP8 E4M3 bytes with the exponent clamped to a narrow band: no NaN (0x7f/0xff)
// and no denormal, so every arm accumulates the same finite magnitudes and a
// bitwise diff means a real ordering/hazard fault rather than an inf.
__host__ static inline uint8_t rand_fp8(uint32_t i) {
  uint32_t r = mix(i);
  uint32_t sign = (r >> 20) & 1u;
  uint32_t exp = 6u + ((r >> 8) & 3u); // 6..9 around the E4M3 bias of 7
  uint32_t man = (r >> 3) & 7u;
  return (uint8_t)((sign << 7) | (exp << 3) | man);
}

// E8M0: 127 is 1.0. Keep the run tight so 48 accumulated products stay well
// inside f32.
__host__ static inline uint8_t rand_e8m0(uint32_t i) {
  return (uint8_t)(126u + (mix(i ^ 0x9e37u) & 3u));
}

// GROUPS_REQ goes straight to the shipping dispatcher, so this kernel selects
// between the SAME three loop bodies the megakernel does -- including the
// GROUPS_REQ=0 shipping path, which is the reference.
template <int GROUPS_REQ, int WANT_DBUF>
__global__ void __launch_bounds__(LANES)
    kloop_arm(uint8_t const *__restrict__ W,
              uint8_t const *__restrict__ WS,
              uint8_t const *__restrict__ TOK,
              uint8_t const *__restrict__ TOKSC,
              float *__restrict__ out) {
  // The token side lives in LDS in the real kernel; keep it there so the
  // ds_read / lgkmcnt structure the deep loop is trying to hide is present.
  __shared__ uint8_t s_tok[TOK_BYTES];
  __shared__ uint8_t s_tok_sc[TOKSC_BYTES];
  for (int i = threadIdx.x; i < TOK_BYTES; i += LANES) {
    s_tok[i] = TOK[i];
  }
  for (int i = threadIdx.x; i < TOKSC_BYTES; i += LANES) {
    s_tok_sc[i] = TOKSC[i];
  }
  __syncthreads();

  int const lane_id = threadIdx.x;
  int const g = lane_id >> 4; // same derivation as the W13 call site

  f32x4_t acc = _gang_moe_kloop<WEIGHT_FP4, false, GROUPS_REQ, MFMA_ITERS,
                                (bool)WANT_DBUF>(
      W, WS, /*row_scale_base=*/0, s_tok, s_tok_sc, g);

#pragma unroll
  for (int i = 0; i < 4; i++) {
    out[lane_id * 4 + i] = acc[i];
  }
}

struct Arm {
  char const *name;
  void (*launch)(uint8_t const *, uint8_t const *, uint8_t const *,
                 uint8_t const *, float *);
};

template <int GR, int DB>
static void launch_arm(uint8_t const *W, uint8_t const *WS, uint8_t const *TOK,
                       uint8_t const *TOKSC, float *out) {
  hipLaunchKernelGGL((kloop_arm<GR, DB>), dim3(1), dim3(LANES), 0, 0, W, WS,
                     TOK, TOKSC, out);
}

int main() {
  printf("MoE W13 k-loop width equality test\n");
  printf("  MFMA_ITERS=%d  KMAJOR=%d WGLOBAL=%d WGPTR=%d SCBASE=%d\n",
         MFMA_ITERS, MPK_MOE_KMAJOR, MPK_MOE_WGLOBAL, MPK_MOE_WGPTR,
         MPK_MOE_SCBASE);
  printf("  every arm reads the same bytes; only issue order differs, so the\n"
         "  correct result is BIT-IDENTICAL to the ship arm.\n\n");

  // KLOOP_ONES=1 replaces the random fill with FP8 1.0 everywhere and E8M0
  // 127 (= 2^0) for every scale. Then EVERY arm has one analytically known
  // answer -- 48 MFMAs x 128 products of 1.0 = 6144.0 in all four slots --
  // so a mismatch names which arm is wrong instead of only saying they
  // disagree. Worth having: the random fill's failures land on exactly 0.0,
  // which is the signature of a lost accumulator rather than a lost term.
  // Modes 2 and 3 name WHICH operand is wrong, which mode 1 cannot. Under
  // mode 1 every byte of every buffer is the same value, so a stale or
  // mis-addressed read still returns the right number -- only a register that
  // no load ever wrote can show up, and an all-zero E8M0 is 2^-127, i.e. a
  // product that prints as 0.0000. Mode 2 puts the weight scale at 2^1 and
  // the token scale at 2^0, mode 3 the reverse; each doubles the answer, so
  // an arm that returns 6144 instead of 12288 has lost exactly the operand
  // that mode's non-default scale rides on.
  int const ones = getenv("KLOOP_ONES") ? atoi(getenv("KLOOP_ONES")) : 0;
  uint8_t const wsc_val = ones == 2 ? 128 : 127;
  uint8_t const tsc_val = ones == 3 ? 128 : 127;
  float const expect = ones ? (ones >= 2 ? 12288.0f : 6144.0f) : 0.0f;
  if (ones) {
    printf("  KLOOP_ONES=%d: data 1.0, w-scale 2^%d, tok-scale 2^%d -> every "
           "slot must be %.1f\n\n", ones, wsc_val - 127, tsc_val - 127, expect);
  }

  std::vector<uint8_t> hW(W_BYTES), hWS(SC_BYTES), hTOK(TOK_BYTES),
      hTOKSC(TOKSC_BYTES);
  for (size_t i = 0; i < W_BYTES; i++) {
    hW[i] = ones ? 0x38 : rand_fp8((uint32_t)i); // 0x38 = E4M3 1.0
  }
  for (size_t i = 0; i < SC_BYTES; i++) {
    hWS[i] = ones ? wsc_val : rand_e8m0((uint32_t)i);
  }
  for (int i = 0; i < TOK_BYTES; i++) {
    hTOK[i] = ones ? 0x38 : rand_fp8((uint32_t)i + 0x5000u);
  }
  for (int i = 0; i < TOKSC_BYTES; i++) {
    hTOKSC[i] = ones ? tsc_val : rand_e8m0((uint32_t)i + 0x600u);
  }

  uint8_t *dW, *dWS, *dTOK, *dTOKSC;
  float *dOut;
  HIP_CHECK(hipMalloc(&dW, W_BYTES));
  HIP_CHECK(hipMalloc(&dWS, SC_BYTES));
  HIP_CHECK(hipMalloc(&dTOK, TOK_BYTES));
  HIP_CHECK(hipMalloc(&dTOKSC, TOKSC_BYTES));
  HIP_CHECK(hipMalloc(&dOut, OUT_FLOATS * sizeof(float)));
  HIP_CHECK(hipMemcpy(dW, hW.data(), W_BYTES, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dWS, hWS.data(), SC_BYTES, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dTOK, hTOK.data(), TOK_BYTES, hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(dTOKSC, hTOKSC.data(), TOKSC_BYTES, hipMemcpyHostToDevice));

  // GROUPS_REQ=0 is the shipping loop and is the reference. 4 is what ships
  // today via the default; it must also match, or the deep loop is wrong at
  // EVERY width and the current default is only accidentally correct.
  Arm const arms[] = {
      {"ship (GR=0)      ", launch_arm<0, 0>},
      {"deep GR=4  (dflt)", launch_arm<4, 0>},
      {"deep GR=6        ", launch_arm<6, 0>},
      {"deep GR=8        ", launch_arm<8, 0>},
      {"deep GR=12       ", launch_arm<12, 0>},
      {"deep GR=16       ", launch_arm<16, 0>},
      {"dbuf GR=4        ", launch_arm<4, 1>},
      {"dbuf GR=6        ", launch_arm<6, 1>},
      {"dbuf GR=8        ", launch_arm<8, 1>},
  };
  int const n_arms = (int)(sizeof(arms) / sizeof(arms[0]));

  std::vector<std::vector<float>> res(n_arms,
                                      std::vector<float>(OUT_FLOATS, 0.0f));
  for (int a = 0; a < n_arms; a++) {
    HIP_CHECK(hipMemset(dOut, 0, OUT_FLOATS * sizeof(float)));
    arms[a].launch(dW, dWS, dTOK, dTOKSC, dOut);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(res[a].data(), dOut, OUT_FLOATS * sizeof(float),
                        hipMemcpyDeviceToHost));
  }

  // Bit-exact, deliberately: see the header. A tolerance here would hide
  // exactly the class of fault being hunted.
  int fails = 0;
  printf("%-18s  %-10s  %-10s  %s\n", "arm", "out[0]", "out[130]", "vs ship");
  for (int a = 0; a < n_arms; a++) {
    int bad = 0;
    float worst = 0.0f;
    for (int i = 0; i < OUT_FLOATS; i++) {
      if (__builtin_bit_cast(uint32_t, res[a][i]) !=
          __builtin_bit_cast(uint32_t, res[0][i])) {
        bad++;
        float d = res[a][i] - res[0][i];
        d = d < 0 ? -d : d;
        if (d > worst) {
          worst = d;
        }
      }
    }
    printf("%-18s  %-10.4f  %-10.4f  ", arms[a].name, res[a][0], res[a][130]);
    if (ones) {
      int off = 0;
      for (int i = 0; i < OUT_FLOATS; i++) {
        if (res[a][i] != expect) {
          off++;
        }
      }
      printf("[abs: %s] ", off == 0 ? "OK " : "BAD");
    }
    if (bad == 0) {
      printf("MATCH\n");
    } else {
      printf("*** MISMATCH: %d/%d lanes-slots, max |d| = %.6f\n", bad,
             OUT_FLOATS, worst);
      if (a != 0) {
        fails++;
      }
    }
  }

  printf("\n%s\n", fails == 0
                       ? "ALL WIDTHS AGREE -- the fault is NOT in the k-loop "
                         "body in isolation."
                       : "*** A WIDTH DISAGREES -- reproduced outside the "
                         "megakernel. Bisect from here.");
  return fails == 0 ? 0 : 1;
}
