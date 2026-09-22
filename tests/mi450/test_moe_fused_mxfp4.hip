/* Validates gang_moe_fused_mxfp4_kernel_mi450 -- the fused MoE task: W13
 * (gate/up projection, MXFP4 weights x FP8-quantized tokens) -> SwiGLU ->
 * W2 (down projection) -> scaled accumulation into an f32 workspace, with a
 * global barrier separating the two phases.
 *
 * WHAT THIS TEST CAN AND CANNOT ESTABLISH
 *
 * This kernel's two phases are coupled by a global barrier and by the SwiGLU
 * buffer in between. Per tests/mi450 precedent and the FFM caveat recorded in
 * the kernel header, FFM-Lite CANNOT test that barrier: waves run closely
 * enough in lockstep that deleting a __syncthreads() still passes bit-exactly
 * (measured on the rmsnorm twin -- all four shapes passed with the reduction
 * barrier removed). So a green run here is evidence about ARITHMETIC AND
 * INDEXING, not about synchronization. Barrier mutants are expected to
 * survive and are not counted as a coverage gap; they need silicon.
 *
 * What it does check, and why each check exists:
 *
 *  1. TWO INDEPENDENT REFERENCES, NOT ONE. E2M1/E8M0/E4M3 are decoded from
 *     tables written from the format definitions, deliberately unlike the
 *     device's bit manipulation, so a shared misunderstanding of the encoding
 *     has to be made twice to go unnoticed.
 *
 *  2. THE SWIGLU INTERMEDIATE IS TAKEN FROM THE DEVICE, NOT MIRRORED. W2's
 *     input is the device's own bf16 SwiGLU output. A host mirror would have
 *     to reproduce fast_swigluoai's exact transcendental, and any difference
 *     there would be indistinguishable from a real W2 bug. So phase 1 is
 *     checked against the device's phase-0 result, and phase 0 is checked
 *     against the host independently. A bug that corrupts the intermediate is
 *     still caught -- by check 3, not by check 4.
 *
 *  3. DISCRIMINATING POWER IS ASSERTED, NOT ASSUMED. Inputs are a mixing hash
 *     of the indices, never an affine function of them: affine test data made
 *     two real cross-lane bugs undetectable earlier in this port. The test
 *     then asserts that outputs actually differ across rows, weight groups,
 *     tokens and top-k slots -- a kernel that ignored wg_idx, or wrote every
 *     token the same value, would otherwise pass.
 *
 *  4. POISON. Every output buffer is pre-filled with a sentinel and survivors
 *     are counted. On the rmsnorm twin the poison check, not the value
 *     comparison, is what killed a mutant. A pass must not be able to come
 *     from memory nobody wrote.
 *
 *  5. A TIGHT RELATIVE BOUND ALONGSIDE THE LOOSE ABSOLUTE ONE. Checking only
 *     |err| < tol lets a kernel that is systematically slightly wrong pass,
 *     because that tolerance has to scale with sqrt(K) and |ref| to admit
 *     legitimate f32 reassociation. The second bound is relative and sized to
 *     f32 accumulation noise alone (4*sqrt(K)*2^-23), which nothing but
 *     rounding fits inside -- a single wrong MXFP4 nibble is ~1e-2 relative,
 *     four decades above it.
 *
 *     This was originally a BIT-EXACT floor (">=10% of values must match the
 *     host exactly"). That check was unpassable by construction and is kept
 *     here as a warning: the host reference sums in double and the device
 *     accumulates in f32 through WMMA, so they disagree in the last ulp almost
 *     everywhere. It measured 0/56, 0/224, 0/112 against a true worst relative
 *     error of 8.2e-7 -- the assertion was wrong, not the kernel. A test whose
 *     failure mode is "always fails" is as useless as one that always passes,
 *     and takes longer to disbelieve.
 *
 * TILE COVERAGE. The decode is `global_tile = tile_idx * 8 + xcd_id()`. Under
 * FFM xcd_id() is pinned to 0, so a plain launch reaches only tiles that are
 * multiples of 8 -- 7 of every 8 tiles never run, and the test would still
 * report PASS on the ones it did. This test is built with
 * MIRAGE_XCD_ID_FROM_BLOCKIDX_Y and launched with gridDim.y == 8 so the whole
 * tile space is covered. That is coverage of the tile SPACE, not of cross-XCD
 * distribution, which remains untested.
 */
#include "mirage/persistent_kernel/runtime_header.h"
#include "mirage/persistent_kernel/mpk_atoms.cuh"
#include "mirage/persistent_kernel/tasks/mi450/gang_moe_fused_mxfp4_mi450.cuh"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define HIP_OK(x)                                                              \
  do {                                                                         \
    hipError_t e = (x);                                                        \
    if (e != hipSuccess) {                                                     \
      printf("HIP error %s at line %d\n", hipGetErrorString(e), __LINE__);     \
      return 2;                                                                \
    }                                                                          \
  } while (0)

// ── Independent decode tables ───────────────────────────────────────────────
static double const kE2M1[16] = {0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
                                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};

static double e4m3(uint8_t b) {
  int s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 0x7;
  double mag = (e == 0) ? std::ldexp((double)m / 8.0, -6)
                        : std::ldexp(1.0 + (double)m / 8.0, e - 7);
  return s ? -mag : mag;
}
static double e8m0(uint8_t e) {
  return std::ldexp(1.0, (int)e - 127);
}

static float bf16_to_f(unsigned short b) {
  unsigned u = (unsigned)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
static unsigned short f_to_bf16_rne(float f) {
  unsigned u;
  std::memcpy(&u, &f, 4);
  return (unsigned short)((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

// Never an affine function of the indices -- see note 3 in the header.
static unsigned mix(unsigned x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

// ── Host mirror of mx_quant_fp8 ─────────────────────────────────────────────
// E8M0 from log2/ceil, deliberately unlike the device's float-bit path.
static uint8_t host_e8m0(float amax) {
  if (amax == 0.0f) {
    return 0;
  }
  double need = (double)amax / 448.0;
  int e = (int)std::ceil(std::log2(need));
  while (std::ldexp(1.0, e) < need) {
    ++e;
  }
  while (e > -127 && std::ldexp(1.0, e - 1) >= need) {
    --e;
  }
  return (uint8_t)std::max(0, std::min(255, e + 127));
}

// float -> E4M3, round-to-nearest-even (the measured hardware rule).
static uint8_t host_f2e4m3(double x, int *saturated) {
  uint8_t sign = (x < 0 || (x == 0 && std::signbit(x))) ? 0x80 : 0x00;
  double a = std::fabs(x);
  if (a == 0.0) {
    return sign;
  }
  int e = (int)std::floor(std::log2(a));
  if (e < -6) {
    e = -6;
  }
  double n = std::nearbyint(a / std::ldexp(1.0, e - 3));
  if (n >= 16.0) {
    ++e;
    n = 8.0;
  }
  if (e > 8 || (e == 8 && n > 14.0)) {
    *saturated += 1;
    return (uint8_t)(sign | 0x7E);
  }
  if (e == -6 && n < 8.0) {
    return (uint8_t)(sign | (int)n);
  }
  return (uint8_t)(sign | ((e + 7) << 3) | ((int)n - 8));
}

static void host_quant_fp8(unsigned short const *row,
                           int K,
                           std::vector<uint8_t> &fp8,
                           std::vector<uint8_t> &scl,
                           int *saturated) {
  int const nblk = K / 32;
  fp8.assign(K, 0);
  scl.assign(nblk, 0);
  for (int b = 0; b < nblk; ++b) {
    float amax = 0.f;
    for (int j = 0; j < 32; ++j) {
      amax = std::fmax(amax, std::fabs(bf16_to_f(row[b * 32 + j])));
    }
    uint8_t se = host_e8m0(amax);
    double s = (se == 0) ? 1.0 : std::ldexp(1.0, (int)se - 127);
    for (int j = 0; j < 32; ++j) {
      fp8[b * 32 + j] =
          host_f2e4m3((double)bf16_to_f(row[b * 32 + j]) / s, saturated);
    }
    scl[b] = se;
  }
}

// One MXFP4 row (packed nibbles + E8M0 scales) dotted with one FP8 row, in
// double. `wg` points at this weight group's data; `row` is the row within it.
static double host_dot_f4xf8(uint8_t const *wg_data,
                             uint8_t const *wg_scl,
                             int row,
                             int K,
                             std::vector<uint8_t> const &tok_fp8,
                             std::vector<uint8_t> const &tok_scl) {
  int const nblk = K / 32;
  uint8_t const *rd = wg_data + (size_t)row * (K / 2);
  uint8_t const *rs = wg_scl + (size_t)row * nblk;
  double acc = 0.0;
  for (int b = 0; b < nblk; ++b) {
    double ws = e8m0(rs[b]);
    double ts = e8m0(tok_scl[b]);
    for (int j = 0; j < 32; ++j) {
      int k = b * 32 + j;
      uint8_t byte = rd[k >> 1];
      int nib = (k & 1) ? (byte >> 4) : (byte & 0xF);
      acc += kE2M1[nib] * ws * e4m3(tok_fp8[k]) * ts;
    }
  }
  return acc;
}

// ── Launcher ────────────────────────────────────────────────────────────────
// The kernel reads layer_idx from the last 4 bytes of the dynamic LDS
// allocation, which the real orchestrator writes before dispatching a task.
// There is no orchestrator here, so the test writes it: the W13 releaser
// stores layer_idx+1 into the barrier and the W2 poller waits for that exact
// value, so if the two read different garbage the poll never clears and the
// kernel spins forever. Zero makes the expected release value 1, matching the
// hipMemset-cleared barrier.
//
// TILE_BASE exists because the W13->W2 barrier is a real cross-workgroup
// dependency: a W2 tile spins until every W13 tile of its expert has arrived.
// One launch covering the whole tile space would therefore deadlock unless all
// 272 workgroups are co-resident, which is exactly the FFM residency ceiling
// recorded in MPK_MAX_RESIDENT_BLOCKS. The test instead runs W13 to completion
// in one launch, then W2 in a second, so the release has already fired before
// any W2 tile polls. That is a weaker execution schedule than production and
// it is the point: it tests the arithmetic without depending on a barrier FFM
// cannot model anyway.
#if defined(__gfx1250__) || !defined(__HIP_DEVICE_COMPILE__)
template <int BATCH, int INTER, int HIDDEN, int EXPERTS, int TOPK, int W13_OPW,
          int W2_OPW>
__global__ __launch_bounds__(256) void moe_fused_kernel(void **p,
                                                        int tile_base) {
#if defined(MIRAGE_ARCH_GFX1250)
  extern __shared__ char _smem[];
  constexpr int LAYER_IDX_OFF =
      mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE -
      mirage::runtime::LAYER_IDX_SMEM_OFFSET_FROM_END;
  if (threadIdx.x == 0) {
    *reinterpret_cast<int *>(&_smem[LAYER_IDX_OFF]) = 0;
  }
  __syncthreads();
  kernel::gang_moe_fused_mxfp4_kernel_mi450<BATCH, INTER, HIDDEN, EXPERTS,
                                            TOPK, W13_OPW, W2_OPW>(
      p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10],
      tile_base + (int)blockIdx.x);
#endif
}
#endif

// ── One shape ───────────────────────────────────────────────────────────────
template <int BATCH, int INTER, int HIDDEN, int EXPERTS, int W13_OPW,
          int W2_OPW>
static int run_case(char const *name) {
  constexpr int TOPK = 4; // must equal kernel::MOE_WS_SLOTS
  constexpr int W13_OUT = 2 * INTER;
  constexpr int W13_K = HIDDEN;
  constexpr int W13_NBLK = W13_K / 32;
  constexpr int W13_WG_DATA = W13_OPW * (W13_K / 2);
  constexpr int W13_WG_SCL = W13_OPW * W13_NBLK;
  constexpr int W13_WG_BYTES = W13_WG_DATA + W13_WG_SCL;
  constexpr int W13_WGS = W13_OUT / W13_OPW;

  constexpr int W2_OUT = HIDDEN;
  constexpr int W2_K = INTER;
  constexpr int W2_NBLK = W2_K / 32;
  constexpr int W2_WG_DATA = W2_OPW * (W2_K / 2);
  constexpr int W2_WG_SCL = W2_OPW * W2_NBLK;
  constexpr int W2_WG_BYTES = W2_WG_DATA + W2_WG_SCL;
  constexpr int W2_WGS = W2_OUT / W2_OPW;

  constexpr int MAX_ACT = (TOPK * BATCH < EXPERTS) ? TOPK * BATCH : EXPERTS;
  constexpr int PAD_MULTIPLE = 240;
  constexpr int W13_TILES = (BATCH <= 16) ? W13_WGS : BATCH * W13_WGS;
  constexpr int W2_TILES = (BATCH <= 16) ? W2_WGS : BATCH * W2_WGS;
  constexpr int TOTAL_W13_REAL = MAX_ACT * W13_TILES;
  constexpr int TOTAL_W13 =
      ((TOTAL_W13_REAL + PAD_MULTIPLE - 1) / PAD_MULTIPLE) * PAD_MULTIPLE;
  constexpr int TOTAL_TILES = TOTAL_W13 + MAX_ACT * W2_TILES;

  // The wave split the kernel derives, recomputed here so the log says which
  // geometry actually ran -- the K-split paths are different code.
  constexpr int W13_NT = W13_OPW / 16;
  constexpr int W13_NG = W13_NT < 8 ? W13_NT : 8;
  constexpr int W2_NT = W2_OPW / 16;
  constexpr int W2_NG = W2_NT < 8 ? W2_NT : 8;
  printf("\n--- %s  batch=%d hidden=%d inter=%d E=%d  W13 %dx%d  W2 %dx%d "
         "(tiles: %d W13 padded, %d total) ---\n",
         name, BATCH, HIDDEN, INTER, EXPERTS, W13_NG, 8 / W13_NG, W2_NG,
         8 / W2_NG, TOTAL_W13, TOTAL_TILES);

  // ── Routing ───────────────────────────────────────────────────────────────
  // routing[e][b] is 0 (not routed) or slot+1. Every token gets exactly TOPK
  // distinct experts; the mask lists the union of activated experts.
  std::vector<int> h_routing((size_t)EXPERTS * BATCH, 0);
  std::vector<int> h_mask(EXPERTS + 1, -1);
  std::vector<int> act; // activated expert ids, in mask order
  for (int b = 0; b < BATCH; ++b) {
    for (int s = 0; s < TOPK; ++s) {
      // Spread experts so different tokens hit different (and overlapping)
      // experts -- an expert serving two tokens is what exercises the N-axis
      // packing, and one serving none exercises the padding path.
      int e = (int)(mix((unsigned)(b * 7919 + s * 104729)) % EXPERTS);
      // Keep the TOPK experts of a token distinct: a repeat would write the
      // same (token, slot) workspace slab twice and make the reference
      // ambiguous rather than wrong.
      while (h_routing[(size_t)e * BATCH + b] != 0) {
        e = (e + 1) % EXPERTS;
      }
      h_routing[(size_t)e * BATCH + b] = s + 1;
    }
  }
  for (int e = 0; e < EXPERTS && (int)act.size() < MAX_ACT; ++e) {
    for (int b = 0; b < BATCH; ++b) {
      if (h_routing[(size_t)e * BATCH + b] != 0) {
        act.push_back(e);
        break;
      }
    }
  }
  int const n_act = (int)act.size();
  for (int i = 0; i < n_act; ++i) {
    h_mask[i] = act[i];
  }
  // Slots past the activated count stay -1: the padding path. Those tiles must
  // run, arrive at the barrier, and store nothing.
  h_mask[EXPERTS] = n_act;
  printf("routing: %d/%d expert slots activated (%d padding slots)\n", n_act,
         MAX_ACT, MAX_ACT - n_act);

  // ── Inputs ────────────────────────────────────────────────────────────────
  std::vector<unsigned short> h_in((size_t)BATCH * HIDDEN);
  for (size_t i = 0; i < h_in.size(); ++i) {
    // Small odd-ish values around 1.0: large enough that E4M3 quantization is
    // not trivially exact, small enough that the K-length sum stays in range.
    float v = ((float)(int)(mix((unsigned)i) % 2001) - 1000.f) / 700.f;
    h_in[i] = f_to_bf16_rne(v);
  }

  std::vector<uint8_t> h_w13((size_t)EXPERTS * W13_WGS * W13_WG_BYTES);
  for (size_t i = 0; i < h_w13.size(); ++i) {
    h_w13[i] = (uint8_t)(mix((unsigned)(i * 3 + 11)) & 0xFF);
  }
  std::vector<uint8_t> h_w2((size_t)EXPERTS * W2_WGS * W2_WG_BYTES);
  for (size_t i = 0; i < h_w2.size(); ++i) {
    h_w2[i] = (uint8_t)(mix((unsigned)(i * 5 + 23)) & 0xFF);
  }
  // Random bytes in the scale regions would be E8M0 exponents spanning the
  // full 2^-127..2^128 range, which overflows the reference before it tests
  // anything. Clamp the scale bytes to a narrow band around 2^0.
  auto tame_scales = [](std::vector<uint8_t> &w, int wgs, int wg_bytes,
                        int data_bytes, int experts) {
    for (int e = 0; e < experts; ++e) {
      for (int g = 0; g < wgs; ++g) {
        uint8_t *s = w.data() + ((size_t)e * wgs + g) * wg_bytes + data_bytes;
        int n = wg_bytes - data_bytes;
        for (int i = 0; i < n; ++i) {
          s[i] = (uint8_t)(120 + (s[i] % 15)); // 2^-7 .. 2^7
        }
      }
    }
  };
  tame_scales(h_w13, W13_WGS, W13_WG_BYTES, W13_WG_DATA, EXPERTS);
  tame_scales(h_w2, W2_WGS, W2_WG_BYTES, W2_WG_DATA, EXPERTS);

  std::vector<unsigned short> h_b13((size_t)EXPERTS * W13_OUT);
  for (size_t i = 0; i < h_b13.size(); ++i) {
    h_b13[i] = f_to_bf16_rne(((float)(int)(mix((unsigned)(i + 7)) % 401) -
                              200.f) / 400.f);
  }
  std::vector<unsigned short> h_b2((size_t)EXPERTS * W2_OUT);
  for (size_t i = 0; i < h_b2.size(); ++i) {
    h_b2[i] = f_to_bf16_rne(((float)(int)(mix((unsigned)(i + 91)) % 401) -
                             200.f) / 400.f);
  }
  // Routing weights: distinct per (token, slot) so a kernel that swapped them
  // produces a different answer rather than the same one.
  std::vector<float> h_rw((size_t)BATCH * TOPK);
  for (size_t i = 0; i < h_rw.size(); ++i) {
    h_rw[i] = 0.15f + (float)(mix((unsigned)(i + 555)) % 700) / 1000.f;
  }

  // ── Device buffers, all poisoned ──────────────────────────────────────────
  size_t const swiglu_n = (size_t)BATCH * TOPK * INTER;
  size_t const ws_n = (size_t)BATCH * kernel::MOE_WS_SLOTS * HIDDEN;
  // Barrier slots come from the PADDED tile space, not from EXPERTS -- see
  // moe450_barrier_slots(). At these test shapes the two differ by ~4x and the
  // EXPERTS-sized buffer is overrun by the arrival atomic.
  size_t const bar_n =
      (size_t)kernel::moe450_detail::moe450_barrier_slots<
          BATCH, TOPK, EXPERTS, INTER, W13_OPW>() *
      kernel::MOE450_BAR_STRIDE;

  unsigned short const POISON_U16 = 0xDEAD;
  float const POISON_F32 = -1.0e30f;

  std::vector<unsigned short> h_swiglu(swiglu_n, POISON_U16);
  std::vector<float> h_ws(ws_n, POISON_F32);

  void *d_in, *d_w13, *d_w2, *d_routing, *d_mask, *d_b13, *d_b2, *d_rw;
  void *d_swiglu, *d_ws, *d_bar, *d_argv;
  HIP_OK(hipMalloc(&d_in, h_in.size() * 2));
  HIP_OK(hipMalloc(&d_w13, h_w13.size()));
  HIP_OK(hipMalloc(&d_w2, h_w2.size()));
  HIP_OK(hipMalloc(&d_routing, h_routing.size() * 4));
  HIP_OK(hipMalloc(&d_mask, h_mask.size() * 4));
  HIP_OK(hipMalloc(&d_b13, h_b13.size() * 2));
  HIP_OK(hipMalloc(&d_b2, h_b2.size() * 2));
  HIP_OK(hipMalloc(&d_rw, h_rw.size() * 4));
  HIP_OK(hipMalloc(&d_swiglu, swiglu_n * 2));
  HIP_OK(hipMalloc(&d_ws, ws_n * 4));
  HIP_OK(hipMalloc(&d_bar, bar_n * 4));
  HIP_OK(hipMalloc(&d_argv, 11 * sizeof(void *)));

  HIP_OK(hipMemcpy(d_in, h_in.data(), h_in.size() * 2, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_w13, h_w13.data(), h_w13.size(), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_w2, h_w2.data(), h_w2.size(), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_routing, h_routing.data(), h_routing.size() * 4,
                   hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_mask, h_mask.data(), h_mask.size() * 4,
                   hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_b13, h_b13.data(), h_b13.size() * 2,
                   hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_b2, h_b2.data(), h_b2.size() * 2, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_rw, h_rw.data(), h_rw.size() * 4, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_swiglu, h_swiglu.data(), swiglu_n * 2,
                   hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_ws, h_ws.data(), ws_n * 4, hipMemcpyHostToDevice));
  HIP_OK(hipMemset(d_bar, 0, bar_n * 4));

  void *h_argv[11] = {d_in,  d_w13,    d_w2, d_routing, d_mask, d_b13,
                      d_b2,  d_rw,     d_swiglu, d_ws,  d_bar};
  HIP_OK(hipMemcpy(d_argv, h_argv, sizeof(h_argv), hipMemcpyHostToDevice));

  // Launch the FULL budget, not moe450_smem_bytes(). That helper reports what
  // the kernel's own tile plan needs (~1-4 KB here), but the barrier release
  // reads layer_idx from the last 4 bytes of MAX_DYNAMIC_SHARED_MEMORY_SIZE --
  // persistent_kernel.cuh always launches the whole fixed allocation and the
  // orchestrator writes layer_idx there. Sizing this launch from the helper
  // makes that an out-of-bounds LDS read; it crashed FFM outright rather than
  // returning garbage, which is the lucky outcome.
  constexpr int SMEM = mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE;
  static_assert(
      kernel::moe450_detail::moe450_smem_bytes<BATCH, INTER, HIDDEN, W2_OPW>() <=
          SMEM,
      "tile plan must fit inside the launched allocation");

  // layer_idx is read from LDS, which is uninitialized at launch here (no
  // orchestrator). Its only use is the release value written to the barrier,
  // and nothing in this test reads the barrier back, so any value is fine --
  // but it must be a value the kernel can legally load, hence the full SMEM.

  // gridDim.x enumerates tile_idx; gridDim.y stands in for xcd_id via the
  // test-only seam, so tile_idx*8 + y covers the tile range. See the header.
  //
  // Two launches, W13 then W2 -- see the note above moe_fused_kernel. Each
  // covers its own half of the tile space; the split point is TOTAL_W13, which
  // is where the kernel's own `is_w2` predicate flips.
  // The split must land on a tile_idx boundary, or the second launch's
  // tile_base skips or repeats tiles. TOTAL_W13 is a multiple of PAD_MULTIPLE
  // (240), so this holds -- but it holds by arithmetic, not by intent.
  static_assert(TOTAL_W13 % 8 == 0,
                "W13/W2 split point must be a multiple of the 8-XCD stride");
  int const w13_x = (TOTAL_W13 + 7) / 8;
  int const w2_x = (TOTAL_TILES - TOTAL_W13 + 7) / 8;
  hipLaunchKernelGGL(
      (moe_fused_kernel<BATCH, INTER, HIDDEN, EXPERTS, TOPK, W13_OPW, W2_OPW>),
      dim3(w13_x, 8), dim3(256), SMEM, 0, (void **)d_argv, 0);
  HIP_OK(hipDeviceSynchronize());
  HIP_OK(hipGetLastError());
  hipLaunchKernelGGL(
      (moe_fused_kernel<BATCH, INTER, HIDDEN, EXPERTS, TOPK, W13_OPW, W2_OPW>),
      dim3(w2_x, 8), dim3(256), SMEM, 0, (void **)d_argv, TOTAL_W13 / 8);
  HIP_OK(hipDeviceSynchronize());
  HIP_OK(hipGetLastError());

  HIP_OK(hipMemcpy(h_swiglu.data(), d_swiglu, swiglu_n * 2,
                   hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(h_ws.data(), d_ws, ws_n * 4, hipMemcpyDeviceToHost));

  // ── Check 1: poison ───────────────────────────────────────────────────────
  // Only (token, slot) pairs that were actually routed are expected to be
  // written. A slot the routing never used must stay poisoned -- if it does
  // not, some tile wrote outside its slab.
  int sg_untouched = 0, sg_stray = 0, ws_untouched = 0, ws_stray = 0;
  for (int b = 0; b < BATCH; ++b) {
    for (int s = 0; s < TOPK; ++s) {
      bool routed = false;
      for (int e = 0; e < EXPERTS; ++e) {
        if (h_routing[(size_t)e * BATCH + b] == s + 1) {
          routed = true;
          break;
        }
      }
      for (int j = 0; j < INTER; ++j) {
        bool p = h_swiglu[((size_t)b * TOPK + s) * INTER + j] == POISON_U16;
        if (routed && p) {
          ++sg_untouched;
        }
        if (!routed && !p) {
          ++sg_stray;
        }
      }
      for (int j = 0; j < HIDDEN; ++j) {
        bool p = h_ws[((size_t)b * kernel::MOE_WS_SLOTS + s) * HIDDEN + j] ==
                 POISON_F32;
        if (routed && p) {
          ++ws_untouched;
        }
        if (!routed && !p) {
          ++ws_stray;
        }
      }
    }
  }

  // ── Check 2: W13 + SwiGLU, against the host reference ─────────────────────
  // Checked in the pre-activation domain: the host recomputes the gate/up
  // pair, and the comparison is against what fast_swigluoai would have to have
  // been fed. Rather than mirror the activation (see note 2), compare the
  // device's own bf16 output to the host activation only loosely, and rely on
  // the exact check for W2.
  int sat = 0;
  int w13_checked = 0, w13_over = 0;
  double w13_worst = 0.0;
  std::vector<std::vector<uint8_t>> tok_fp8(BATCH), tok_scl(BATCH);
  for (int b = 0; b < BATCH; ++b) {
    host_quant_fp8(h_in.data() + (size_t)b * HIDDEN, HIDDEN, tok_fp8[b],
                   tok_scl[b], &sat);
  }

  // Sample rather than sweep: a full sweep is O(batch*topk*inter*hidden) in
  // double on the host and dominates the runtime under FFM. The stride is
  // coprime with every tile dimension so the sample is not aligned to any of
  // them -- a bug confined to one wave or one weight group is still hit.
  constexpr int SAMPLE_STRIDE = 37;
  for (int b = 0; b < BATCH; ++b) {
    for (int e = 0; e < EXPERTS; ++e) {
      int rv = h_routing[(size_t)e * BATCH + b];
      if (rv == 0) {
        continue;
      }
      int slot = rv - 1;
      bool in_mask = false;
      for (int i = 0; i < n_act; ++i) {
        if (act[i] == e) {
          in_mask = true;
        }
      }
      if (!in_mask) {
        continue; // expert past MAX_ACTIVATED: no tile computes it
      }
      for (int n = 0; n < INTER; n += SAMPLE_STRIDE) {
        // Output row pair (gate, up) = rows 2n, 2n+1 of the W13 output.
        int const og = 2 * n, ou = 2 * n + 1;
        int const wg_g = og / W13_OPW, row_g = og % W13_OPW;
        int const wg_u = ou / W13_OPW, row_u = ou % W13_OPW;
        uint8_t const *base_g =
            h_w13.data() + ((size_t)e * W13_WGS + wg_g) * W13_WG_BYTES;
        uint8_t const *base_u =
            h_w13.data() + ((size_t)e * W13_WGS + wg_u) * W13_WG_BYTES;
        double g = host_dot_f4xf8(base_g, base_g + W13_WG_DATA, row_g, W13_K,
                                  tok_fp8[b], tok_scl[b]) +
                   bf16_to_f(h_b13[(size_t)e * W13_OUT + og]);
        double u = host_dot_f4xf8(base_u, base_u + W13_WG_DATA, row_u, W13_K,
                                  tok_fp8[b], tok_scl[b]) +
                   bf16_to_f(h_b13[(size_t)e * W13_OUT + ou]);
        // fast_swigluoai, per swigluoai_mi300.cuh: clamped SwiGLU-OAI.
        double gc = g > 7.0 ? 7.0 : g;
        double uc = u > 7.0 ? 7.0 : (u < -7.0 ? -7.0 : u);
        double ref = gc / (1.0 + std::exp(-1.702 * gc)) * (uc + 1.0);

        float got = bf16_to_f(h_swiglu[((size_t)b * TOPK + slot) * INTER + n]);
        ++w13_checked;
        // Finiteness before tolerance: `err > tol` is false when err is NaN, so
        // without this a kernel producing NaN throughout reports zero
        // mismatches and a worst error of 0.0 -- a better-looking result than a
        // correct run. Counted as an over-tolerance failure so the existing
        // pass criterion picks it up.
        if (!std::isfinite((double)got) || !std::isfinite(ref)) {
          ++w13_over;
          if (w13_over <= 4) {
            printf("  W13 NON-FINITE b=%d e=%d n=%d: got %f ref %f -- a "
                   "tolerance check cannot detect this\n", b, e, n, got, ref);
          }
          continue;
        }
        double err = std::fabs((double)got - ref);
        double tol = 0.06 * (std::fabs(ref) + 1.0); // bf16 out + fast exp2
        if (err > tol) {
          ++w13_over;
          if (err > w13_worst) {
            w13_worst = err;
          }
          if (w13_over <= 4) {
            printf("  W13 mismatch b=%d e=%d n=%d: got %.5f ref %.5f "
                   "(g=%.4f u=%.4f)\n",
                   b, e, n, got, ref, g, u);
          }
        }
      }
    }
  }

  // ── Check 3: W2, against the DEVICE's SwiGLU output ───────────────────────
  // The workspace slab for (token, slot) is written by exactly one expert --
  // the one that token routed to in that slot -- so this is a plain reference,
  // not an accumulation.
  int w2_checked = 0, w2_over = 0, w2_exact = 0;
  // Worst |got-ref|/|ref|, the quantity the pass criterion below is built on.
  double w2_worst_rel = 0.0;
  double w2_worst = 0.0;
  for (int b = 0; b < BATCH; ++b) {
    for (int e = 0; e < EXPERTS; ++e) {
      int rv = h_routing[(size_t)e * BATCH + b];
      if (rv == 0) {
        continue;
      }
      int slot = rv - 1;
      bool in_mask = false;
      for (int i = 0; i < n_act; ++i) {
        if (act[i] == e) {
          in_mask = true;
        }
      }
      if (!in_mask) {
        continue;
      }
      // Quantize the device's own SwiGLU row exactly as the kernel does.
      std::vector<uint8_t> s_fp8, s_scl;
      host_quant_fp8(h_swiglu.data() + ((size_t)b * TOPK + slot) * INTER, W2_K,
                     s_fp8, s_scl, &sat);
      float const rw = h_rw[(size_t)b * TOPK + slot];
      for (int n = 0; n < HIDDEN; n += SAMPLE_STRIDE) {
        int const wg = n / W2_OPW, row = n % W2_OPW;
        uint8_t const *base =
            h_w2.data() + ((size_t)e * W2_WGS + wg) * W2_WG_BYTES;
        double d = host_dot_f4xf8(base, base + W2_WG_DATA, row, W2_K, s_fp8,
                                  s_scl);
        double ref = (d + bf16_to_f(h_b2[(size_t)e * W2_OUT + n])) * (double)rw;
        float got = h_ws[((size_t)b * kernel::MOE_WS_SLOTS + slot) * HIDDEN + n];
        ++w2_checked;
        // Finiteness before tolerance -- see the W13 check above. Note the
        // exact-match counter below is already NaN-safe (`==` is false for
        // NaN), but the tolerance test is not, and w2_exact is reported rather
        // than asserted, so it would not fail the run on its own.
        if (!std::isfinite((double)got) || !std::isfinite(ref)) {
          ++w2_over;
          if (w2_over <= 4) {
            printf("  W2 NON-FINITE b=%d e=%d n=%d: got %f ref %f -- a "
                   "tolerance check cannot detect this\n", b, e, n, got, ref);
          }
          continue;
        }
        if ((double)got == ref) {
          ++w2_exact;
        }
        {
          double const rel =
              std::fabs((double)got - ref) / (std::fabs(ref) + 1e-30);
          if (rel > w2_worst_rel) {
            w2_worst_rel = rel;
          }
        }
        // f32 accumulation over K in a different order than the host's
        // sequential double sum; bound scales with K and magnitude.
        double tol = 1e-4 * (std::fabs(ref) + 1.0) * std::sqrt((double)W2_K);
        double err = std::fabs((double)got - ref);
        if (err > tol) {
          ++w2_over;
          if (err > w2_worst) {
            w2_worst = err;
          }
          if (w2_over <= 4) {
            printf("  W2 mismatch b=%d e=%d n=%d: got %.6f ref %.6f "
                   "(err %.3g tol %.3g)\n",
                   b, e, n, got, ref, err, tol);
          }
        }
      }
    }
  }

  // ── Check 4: discriminating power ─────────────────────────────────────────
  // Without this, a kernel that ignored wg_idx (writing one weight group's
  // answer everywhere), or that wrote every token the same value, passes every
  // check above on data that happens not to distinguish them.
  int distinct_rows = 0, distinct_slabs = 0;
  {
    std::vector<double> seen;
    for (int n = 0; n < HIDDEN; n += SAMPLE_STRIDE) {
      double v = h_ws[n];
      bool dup = false;
      for (double s : seen) {
        if (s == v) {
          dup = true;
          break;
        }
      }
      if (!dup) {
        seen.push_back(v);
        ++distinct_rows;
      }
    }
    std::vector<double> slabs;
    for (int b = 0; b < BATCH; ++b) {
      for (int s = 0; s < TOPK; ++s) {
        double v = h_ws[((size_t)b * kernel::MOE_WS_SLOTS + s) * HIDDEN];
        if (v == POISON_F32) {
          continue;
        }
        bool dup = false;
        for (double q : slabs) {
          if (q == v) {
            dup = true;
            break;
          }
        }
        if (!dup) {
          slabs.push_back(v);
          ++distinct_slabs;
        }
      }
    }
  }
  // Rows must span more than one weight group, or "ignores wg_idx" is not
  // actually being tested by the row count.
  int const rows_sampled = (HIDDEN + SAMPLE_STRIDE - 1) / SAMPLE_STRIDE;
  int const wgs_spanned = (rows_sampled * SAMPLE_STRIDE) / W2_OPW + 1;

  printf("poison: swiglu untouched=%d stray=%d | ws untouched=%d stray=%d\n",
         sg_untouched, sg_stray, ws_untouched, ws_stray);
  printf("W13: checked=%d over_tol=%d worst=%.4g\n", w13_checked, w13_over,
         w13_worst);
  printf("W2:  checked=%d over_tol=%d worst_abs=%.4g worst_rel=%.4g "
         "(f32 floor %.4g) bit-exact=%d/%d\n",
         w2_checked, w2_over, w2_worst, w2_worst_rel,
         4.0 * std::sqrt((double)W2_K) * 1.19209290e-7, w2_exact, w2_checked);
  printf("discriminating: distinct rows=%d/%d (spanning %d wgs)  distinct "
         "slabs=%d\n",
         distinct_rows, rows_sampled, wgs_spanned, distinct_slabs);
  if (sat) {
    printf("note: %d E4M3 saturations in the host mirror\n", sat);
  }

  int fail = 0;
  if (sg_untouched || sg_stray) {
    printf("FAIL: SwiGLU poison check\n");
    fail = 1;
  }
  if (ws_untouched || ws_stray) {
    printf("FAIL: workspace poison check\n");
    fail = 1;
  }
  if (w13_over) {
    printf("FAIL: %d W13/SwiGLU values outside tolerance\n", w13_over);
    fail = 1;
  }
  if (w2_over) {
    printf("FAIL: %d W2 values outside tolerance\n", w2_over);
    fail = 1;
  }
  // Note 5, corrected. The original check here demanded that ≥10% of W2 values
  // match the host reference BIT-EXACTLY, on the theory that a systematically
  // slightly-wrong kernel stays under a loose tolerance while the exact rate
  // collapses. The premise is wrong and the check could never pass: the host
  // sums K products sequentially in `double`, the device accumulates in f32
  // through WMMA in tile order, so the two differ in the last f32 ulp on
  // essentially every element by construction. Measured exact rate: 0/56,
  // 0/224, 0/112 -- while the worst RELATIVE error was 8.2e-7, comfortably
  // under the ~2.7e-6 f32 K=512 accumulation floor. The kernel was right and
  // the assertion was wrong.
  //
  // What actually catches the "systematically slightly wrong" case is a
  // relative bound tight enough that only f32 rounding fits inside it. The
  // loose per-element tolerance above scales with sqrt(K) and |ref|+1, so it
  // stays permissive for large values; this one does not. sqrt(K)*2^-23 is the
  // random-walk bound on f32 accumulation over K terms; 4x gives headroom for
  // the differing summation order without admitting a real error, which would
  // show up orders of magnitude above it (the MXFP4 quantum is ~6% -- one
  // wrong E2M1 nibble is 1e-2 relative, four decades clear of this).
  double const w2_rel_floor = 4.0 * std::sqrt((double)W2_K) * 1.19209290e-7;
  if (w2_checked && w2_worst_rel > w2_rel_floor) {
    printf("FAIL: W2 worst relative error %.4g exceeds the f32 accumulation "
           "floor %.4g -- this is a real error, not rounding\n",
           w2_worst_rel, w2_rel_floor);
    fail = 1;
  }
  if (distinct_rows < rows_sampled / 2 || wgs_spanned < 2) {
    printf("FAIL: output rows insufficiently distinct (%d/%d over %d wgs) -- "
           "the data cannot discriminate a wg_idx bug\n",
           distinct_rows, rows_sampled, wgs_spanned);
    fail = 1;
  }
  if (distinct_slabs < 2) {
    printf("FAIL: all (token, slot) slabs identical -- token/slot indexing is "
           "not being tested\n");
    fail = 1;
  }
  printf("%s: %s\n", name, fail ? "FAIL" : "PASS");

  hipFree(d_in);
  hipFree(d_w13);
  hipFree(d_w2);
  hipFree(d_routing);
  hipFree(d_mask);
  hipFree(d_b13);
  hipFree(d_b2);
  hipFree(d_rw);
  hipFree(d_swiglu);
  hipFree(d_ws);
  hipFree(d_bar);
  hipFree(d_argv);
  return fail;
}

int main() {
  printf("=== gang_moe_fused_mxfp4_mi450 ===\n");
  printf("NOTE: FFM-Lite cannot test the W13->W2 barrier or the release/\n"
         "acquire pairing. A pass here is about arithmetic and indexing.\n");
  int fail = 0;
  // Single token, both phases at the registered geometry (W13 8x1, W2 4x2):
  // exercises the SINGLE_TOK scalar decode and the W2 K-split reduction.
  fail |= run_case<1, 512, 512, 8, 128, 64>("bs1 registered-shape geometry");
  // Multi-token: the N-axis packing, the ballot compaction, the gather
  // quantizer, and experts serving more than one token.
  fail |= run_case<4, 512, 512, 8, 128, 64>("bs4 packed N-axis");
  // Both phases K-split (W13 4x2 as well): the W13 s_reduce path, which the
  // registered shape never takes, and where SwiGLU must follow the sum.
  fail |= run_case<2, 512, 512, 8, 64, 64>("bs2 W13 K-split");
  // W2 NOT K-split (W2_OPW=128 -> 8 N-tiles over 8 waves -> WAVES_PER_TILE=1).
  //
  // Added because a mutant survived: deleting the routing weight from the
  // non-split W2 epilogue changed nothing, since all three shapes above use
  // W2_OPW=64, which is 4 N-tiles over 8 waves and therefore takes the
  // s_reduce path exclusively. The `if constexpr (W2_WAVES_PER_TILE == 1)`
  // branch -- the float4 write-through epilogue, the one the registered
  // GPT-OSS geometry actually runs at HIDDEN_SIZE 3072 -- was never executed
  // by any test. Both epilogues compute the same thing by different means, so
  // covering one says nothing about the other.
  fail |= run_case<1, 512, 512, 8, 128, 128>("bs1 W2 non-split epilogue");
  printf("\nOVERALL: %s\n", fail ? "FAIL" : "PASS");
  return fail;
}
