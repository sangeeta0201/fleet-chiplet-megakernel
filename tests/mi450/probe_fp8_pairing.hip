/* MEASURES the gfx1250 F4xF8 WMMA operand pairing, assuming nothing.
 *
 * Two earlier attempts failed for the same reason: they assumed the A (FP4)
 * operand's lane->K mapping carried over from the F4xF4 form. The control in
 * probe_fp8_layout.hip showed it does not -- placements that are live in
 * F4xF4 are dark in F4xF8 and vice versa. So this probe assumes NO mapping on
 * either side and measures the pairing directly.
 *
 * METHOD. For each A nibble position `an` within a lane's 32-byte record:
 *   - set that one nibble to E2M1 1.0, in one lane only, everything else zero
 *   - fill B so that (lane, byte position) holds a value naming that position
 *   - read the single nonzero output; its value names the B byte position
 *     that the hardware contracted against this A nibble
 *
 * That gives the pairing an -> bp within a lane, with no reference to what K
 * index either one "is". Since the contraction is over K, two positions that
 * pair are by definition the same K -- which is all a loader needs to know.
 *
 * Swept for a lane in the low half and one in the high half, because the two
 * halves cover different K ranges and need not be symmetric.
 *
 *   ./run_ffm.sh /abs/path/to/probe_fp8_pairing
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

constexpr int M = 16, N = 16;

static uint8_t pos_code(int p) {
  int exp = 1 + p / 8; // 1..8 for p in 0..63
  int man = p % 8;
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
__global__ __launch_bounds__(32) void probe(uint8_t const *a_raw,
                                            uint8_t const *b_raw,
                                            float *out) {
#if defined(MIRAGE_ARCH_GFX1250)
  using namespace kernel::mi450;
  int lane = threadIdx.x;
  mx_i32x16_t a = ((mx_i32x16_t const *)a_raw)[lane];
  mx_i32x16_t b = ((mx_i32x16_t const *)b_raw)[lane];
  mx_f32x8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  // All four E8M0 scale bytes = 127 (2^0). Passing a bare 127 sets only byte
  // 0, and bytes 1..3 then read as E8M0 0 = 2^-127, which flushes K>=32 to
  // zero -- the trap documented in mx_wmma_mi450.cuh. An earlier run of this
  // probe did exactly that and reported a spurious "K=32 only" record.
  acc = _gang_wmma_f4xf8(a, b, acc, 0x7F7F7F7F, 0x7F7F7F7F);
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
  // B must identify (lane, position), not just position. Earlier version used
  // the same values in every lane, so a pairing with lane 0 byte 0 and one
  // with lane 16 byte 0 were indistinguishable -- which is exactly the
  // ambiguity that made the first reading of this probe unusable.
  // Only lanes 0..15 and 16..31 need to be told apart, so the sign bit does it.
  for (int lane = 0; lane < 32; ++lane) {
    for (int p = 0; p < 64; ++p) {
      uint8_t c = pos_code(p);
      if (lane >= 16) {
        c |= 0x80;
      }
      hB[(size_t)lane * 64 + p] = c;
    }
  }
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

  // Sweep A nibble positions. FP4 uses the low 32 bytes = 64 nibbles per lane.
  int const probe_lanes[] = {0, 16};
  for (int li = 0; li < 2; ++li) {
    int L = probe_lanes[li];
    printf("\n=== A lane %d: nibble -> paired B byte position ===\n", L);
    printf("  (A nibble index, then the B byte it contracted with)\n");
    int paired[128];
    for (int an = 0; an < 128; ++an) {
      paired[an] = -1;
      std::fill(hA.begin(), hA.end(), (uint8_t)0);
      size_t byte = (size_t)L * 64 + an / 2;
      hA[byte] = (uint8_t)((an & 1) ? (2 << 4) : 2); // E2M1 code 2 == 1.0
      HIP_CHECK(hipMemcpy(dA, hA.data(), kBytes, hipMemcpyHostToDevice));
      HIP_CHECK(hipMemset(dOut, 0, (size_t)M * N * sizeof(float)));
      hipLaunchKernelGGL(probe, dim3(1), dim3(32), 0, 0, dA, dB, dOut);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipDeviceSynchronize());
      std::vector<float> got((size_t)M * N);
      HIP_CHECK(hipMemcpy(
          got.data(), dOut, got.size() * sizeof(float), hipMemcpyDeviceToHost));
      for (int s = 0; s < M * N; ++s) {
        if (got[s] == 0.f) {
          continue;
        }
        double v = (double)got[s];
        int half = (v < 0) ? 1 : 0; // sign bit encodes the B lane half
        for (int p = 0; p < 64; ++p) {
          if (std::fabs(val_of_pos[p] - std::fabs(v)) < 1e-9) {
            paired[an] = p + 64 * half; // >=64 means B came from lanes 16..31
            break;
          }
        }
        break;
      }
    }
    for (int an = 0; an < 128; ++an) {
      if (an % 16 == 0) {
        printf("\n  n%02d: ", an);
      }
      if (paired[an] < 0) {
        printf("  . ");
      } else {
        printf(" %3d", paired[an]);
      }
    }
    printf("\n");
  }

  printf("\nPROBE COMPLETE (measurement, not pass/fail)\n");
  return 0;
}
