/* MEASURES the gfx1250 FP8 B-operand lane->K mapping.
 *
 * test_mx_fp8_layout.hip falsified the natural inference (that the FP8 byte
 * index equals the FP4 nibble index): all 256 outputs mismatched. So the
 * partition must be measured rather than reasoned about.
 *
 * METHOD. Set the A operand (FP4 weights) so that exactly one K index carries
 * 1.0 and every other K carries 0.0, for every row. Then
 *     D[m][n] = sum_k A[m][k] * B[n][k] = B[n][k_hot]
 * so the output column n reads out, directly, whichever FP8 element the
 * hardware believes lane-position (n, k_hot) holds.
 *
 * Fill B so each of the 32 lanes x 64 byte positions holds a value that names
 * its own position: E4M3 with exponent field (1 + pos/8) and mantissa (pos%8)
 * is distinct for pos in 0..63 and exactly representable. Reading a value back
 * therefore identifies (lane, byte) uniquely.
 *
 * One launch per k_hot, and only a handful of k_hot values are needed to see
 * the pattern -- not the 2048-launch sweep a naive delta probe would need.
 *
 *   ./run_ffm.sh /abs/path/to/probe_fp8_layout
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <vector>

#include "mirage/persistent_kernel/tasks/mi450/mx_layout_mi450.cuh"
#include "mirage/persistent_kernel/tasks/mi450/mx_wmma_mi450.cuh"

#define HIP_CHECK(x)                                                           \
  do {                                                                         \
    hipError_t e = (x);                                                        \
    if (e != hipSuccess) {                                                     \
      printf("HIP error %s at line %d\n", hipGetErrorString(e), __LINE__);     \
      return 2;                                                                \
    }                                                                          \
  } while (0)

constexpr int M = 16, N = 16, K = 128;

// value stored at lane L, byte position P -> unique, exactly representable.
static uint8_t pos_code(int p) {
  int exp = 1 + p / 8;  // 1..8
  int man = p % 8;      // 0..7
  return (uint8_t)((exp << 3) | man);
}
static double e4m3(uint8_t b) {
  int sign = (b >> 7) & 1;
  int exp = (b >> 3) & 0xF;
  int man = b & 0x7;
  double mag = (exp == 0) ? std::ldexp((double)man / 8.0, -6)
                          : std::ldexp(1.0 + (double)man / 8.0, exp - 7);
  return sign ? -mag : mag;
}

#if defined(__gfx1250__) || !defined(__HIP_DEVICE_COMPILE__)
// Raw operands: 64 bytes per lane, lane-contiguous. No helper involved.
__global__ __launch_bounds__(32) void probe(uint8_t const *a_raw,
                                            uint8_t const *b_raw,
                                            float *out) {
#if defined(MIRAGE_ARCH_GFX1250)
  using namespace kernel::mi450;
  int lane = threadIdx.x;
  mx_i32x16_t a = ((mx_i32x16_t const *)a_raw)[lane];
  mx_i32x16_t b = ((mx_i32x16_t const *)b_raw)[lane];
  mx_f32x8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  acc = _gang_wmma_f4xf8(a, b, acc, 127, 127); // 2^0 both sides
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[mx_acc_row(lane, i) * N + mx_acc_col(lane)] = acc[i];
  }
#endif
}

// Control: identical A placement, but B consumed as FP4 instead of FP8.
//
// The F4xF4 form's A layout is the one mx_load_operand_fp4 implements and
// test_mx_layout.hip passes on. If a k_hot that is dark in the mixed kernel
// above lights up here, then the A layout depends on B's *format*, which
// would be a genuine hardware behaviour and not a bug in the probe. If it is
// dark in both, the A placement itself is what is wrong.
__global__ __launch_bounds__(32) void probe_f4xf4(uint8_t const *a_raw,
                                                  uint8_t const *b_raw,
                                                  float *out) {
#if defined(MIRAGE_ARCH_GFX1250)
  using namespace kernel::mi450;
  int lane = threadIdx.x;
  mx_i32x16_t a = ((mx_i32x16_t const *)a_raw)[lane];
  mx_i32x16_t b = ((mx_i32x16_t const *)b_raw)[lane];
  mx_f32x8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  acc = _gang_wmma_f4xf4(a, b, acc, 127, 127);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[mx_acc_row(lane, i) * N + mx_acc_col(lane)] = acc[i];
  }
#endif
}
#endif

int main() {
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
    printf("no HIP device\n");
    return 2;
  }
  printf("device: %s  warpSize=%d\n", prop.gcnArchName, prop.warpSize);

  size_t const kBytes = 32 * 64;
  std::vector<uint8_t> hA(kBytes, 0), hB(kBytes, 0);

  // B: every (lane, byte) holds a code naming itself.
  for (int lane = 0; lane < 32; ++lane) {
    for (int p = 0; p < 64; ++p) {
      hB[(size_t)lane * 64 + p] = pos_code(p);
    }
  }

  // Reverse lookup: value -> byte position (lane is identified separately, by
  // which output column lights up).
  double val_of_pos[64];
  for (int p = 0; p < 64; ++p) {
    val_of_pos[p] = e4m3(pos_code(p));
  }

  uint8_t *dA, *dB;
  float *dOut;
  HIP_CHECK(hipMalloc(&dA, kBytes));
  HIP_CHECK(hipMalloc(&dB, kBytes));
  HIP_CHECK(hipMalloc(&dOut, (size_t)M * N * sizeof(float)));
  HIP_CHECK(hipMemcpy(dB, hB.data(), kBytes, hipMemcpyHostToDevice));

  // The FP4 A-operand mapping is already measured and trusted
  // (mx_load_operand_fp4, pinned by test_mx_layout.hip): lane l holds row
  // l%16, and nibble index mx_nibble_of_k(k) covers K k when
  // mx_lane_has_k(l, k). Use it to place a single hot 1.0.
  printf("\n k_hot | which FP8 byte position each column read\n");
  printf("       | (blank = column read zero)\n");

  int const probe_ks[] = {0, 1, 15, 16, 31, 32, 63, 64, 95, 96, 127};
  for (int idx = 0; idx < (int)(sizeof(probe_ks) / sizeof(probe_ks[0])); ++idx) {
    int k_hot = probe_ks[idx];
    std::fill(hA.begin(), hA.end(), (uint8_t)0);
    for (int lane = 0; lane < 32; ++lane) {
      // Same expressions as mx_lane_has_k / mx_nibble_of_k in
      // mx_layout_mi450.cuh, restated here because those live inside the
      // MIRAGE_ARCH_GFX1250 guard and so are not visible in the host pass.
      if ((lane >> 4) != ((k_hot % 64) / 32)) {
        continue;
      }
      int nib = 32 * (k_hot / 64) + (k_hot % 32);
      // E2M1 code 2 == 1.0
      size_t byte = (size_t)lane * 64 + nib / 2;
      if (nib & 1) {
        hA[byte] |= (uint8_t)(2 << 4);
      } else {
        hA[byte] |= (uint8_t)2;
      }
    }
    HIP_CHECK(hipMemcpy(dA, hA.data(), kBytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dOut, 0, (size_t)M * N * sizeof(float)));
    hipLaunchKernelGGL(probe, dim3(1), dim3(32), 0, 0, dA, dB, dOut);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> got((size_t)M * N);
    HIP_CHECK(hipMemcpy(
        got.data(), dOut, got.size() * sizeof(float), hipMemcpyDeviceToHost));

    // Row 0 is representative: every row got the same hot K.
    printf("  %4d | ", k_hot);
    for (int n = 0; n < N; ++n) {
      double v = got[n];
      if (v == 0.0) {
        printf("  .  ");
        continue;
      }
      int best = -1;
      for (int p = 0; p < 64; ++p) {
        if (std::fabs(val_of_pos[p] - v) < 1e-9) {
          best = p;
          break;
        }
      }
      if (best < 0) {
        printf(" ?%.2f", v);
      } else {
        printf(" %3d ", best);
      }
    }
    printf("\n");
  }

  printf("\nInterpretation: for a given k_hot, column n printing byte position "
         "P means the hardware paired A's K=%s with the FP8 byte at position P "
         "of the lane holding column n.\n",
         "k_hot");

  // ── Control: is A's live K range format-dependent? ──────────────────────
  // Run the SAME A placements through the F4xF4 form. B is nonzero everywhere
  // in both cases, so a k_hot that produces output there but not above tells
  // us the mixed form consumes a different (smaller) K range from A.
  printf("\ncontrol: nonzero output count per k_hot, F4xF8 vs F4xF4\n");
  printf(" k_hot | f4xf8 | f4xf4\n");
  for (int idx = 0; idx < (int)(sizeof(probe_ks) / sizeof(probe_ks[0])); ++idx) {
    int k_hot = probe_ks[idx];
    std::fill(hA.begin(), hA.end(), (uint8_t)0);
    for (int lane = 0; lane < 32; ++lane) {
      if ((lane >> 4) != ((k_hot % 64) / 32)) {
        continue;
      }
      int nib = 32 * (k_hot / 64) + (k_hot % 32);
      size_t byte = (size_t)lane * 64 + nib / 2;
      if (nib & 1) {
        hA[byte] |= (uint8_t)(2 << 4);
      } else {
        hA[byte] |= (uint8_t)2;
      }
    }
    HIP_CHECK(hipMemcpy(dA, hA.data(), kBytes, hipMemcpyHostToDevice));

    int counts[2] = {0, 0};
    for (int which = 0; which < 2; ++which) {
      HIP_CHECK(hipMemset(dOut, 0, (size_t)M * N * sizeof(float)));
      if (which == 0) {
        hipLaunchKernelGGL(probe, dim3(1), dim3(32), 0, 0, dA, dB, dOut);
      } else {
        hipLaunchKernelGGL(probe_f4xf4, dim3(1), dim3(32), 0, 0, dA, dB, dOut);
      }
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipDeviceSynchronize());
      std::vector<float> got((size_t)M * N);
      HIP_CHECK(hipMemcpy(
          got.data(), dOut, got.size() * sizeof(float), hipMemcpyDeviceToHost));
      for (float v : got) {
        if (v != 0.f) {
          ++counts[which];
        }
      }
    }
    printf("  %4d | %5d | %5d\n", k_hot, counts[0], counts[1]);
  }

  printf("PROBE COMPLETE (measurement, not pass/fail)\n");
  return 0;
}
