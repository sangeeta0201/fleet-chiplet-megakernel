// Stage 1 gate B: does Private8 (class A) actually deliver AID-local speed?
//
// The sync half of aid_rt.h is validated (barrier 3344 ns at 256 blocks vs
// FLAT's 15515). This is the memory half. It reproduces the MoE benchmark's
// shape -- 49.6 MB of weight streamed per layer, grid 184 = bs=1 top-4 decode
// occupancy -- under three placements:
//
//   local   slice i in aid_of_xcd(i)          the design's class A
//   remote  slice i in the OPPOSITE AID       every read crosses
//   shared  one plain hipMalloc for all       spanning NC, what fleet does
//
// Reference from aid-local-hbm/MOE_LOCAL_PLACEMENT.md, grid 184:
//   AID-local 15.7 us, remote 24.2 us, split50 24.6 us  (NPS1 baseline 18.0)
//
// PASS = local is ~1.5x faster than remote, i.e. Private8 places what it says
// it places. If local == remote, the allocator is not doing what the API
// claims and nothing built on it can work.
#include "aid_rt.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#if !defined(__HIP_DEVICE_COMPILE__)
#include "aid_local.h"
static void *rt_alloc_aid(size_t bytes, int aid) {
  return mirage::aid::alloc_in_aid(bytes, aid, 0ull);
}
#else
static void *rt_alloc_aid(size_t, int) {
  return nullptr;
}
#endif

#define HIP_OK(x)                                                              \
  do {                                                                         \
    hipError_t e = (x);                                                        \
    if (e != hipSuccess) {                                                     \
      printf("HIP FAIL %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
      return 2;                                                                \
    }                                                                          \
  } while (0)

using namespace nps2;

// 49.6 MB/layer, the gpt-oss-120b top-4 decode MoE weight volume.
static constexpr size_t kLayerBytes = 49600000ull;
static constexpr size_t kSliceBytes = kLayerBytes / kNumXcds;
static constexpr size_t kSliceF4 = kSliceBytes / sizeof(float4);

struct Weight8 {
  float4 const *slice[kNumXcds];
};

// Each block streams only its own XCD's slice, which is the access pattern
// per-consumer placement exists to serve. A dot-product accumulate keeps the
// loads live so nothing is optimised away.
__global__ __launch_bounds__(256) void k_stream(Weight8 w, int iters,
                                                float *out) {
  int const xcd = my_xcd();
  float4 const *base = w.slice[xcd];
  float acc = 0.0f;
  for (int it = 0; it < iters; it++) {
    for (size_t i = threadIdx.x + (size_t)(blockIdx.x / kNumXcds) * blockDim.x;
         i < kSliceF4;
         i += (size_t)blockDim.x * (gridDim.x / kNumXcds)) {
      float4 const v = base[i];
      acc += v.x + v.y + v.z + v.w;
    }
  }
  if (acc == 1234.5678f) { // never true; keeps acc live
    out[blockIdx.x] = acc;
  }
}

static bool alloc_placed(Weight8 &w, bool local, bool shared, void *&shared_p) {
  if (shared) {
    if (hipMalloc(&shared_p, kLayerBytes) != hipSuccess) {
      return false;
    }
    if (hipMemset(shared_p, 1, kLayerBytes) != hipSuccess) {
      return false;
    }
    for (int x = 0; x < kNumXcds; x++) {
      // every XCD reads the same spanning buffer, offset to its own slice
      w.slice[x] = reinterpret_cast<float4 const *>(
          static_cast<char *>(shared_p) + (size_t)x * kSliceBytes);
    }
    return true;
  }
  for (int x = 0; x < kNumXcds; x++) {
    int const home = local ? aid_of_xcd(x) : (1 - aid_of_xcd(x));
    void *p = rt_alloc_aid(kSliceBytes, home);
    if (p == nullptr) {
      printf("  alloc_in_aid failed for slice %d (AID %d)\n", x, home);
      return false;
    }
    if (hipMemset(p, 1, kSliceBytes) != hipSuccess) {
      return false;
    }
    w.slice[x] = static_cast<float4 const *>(p);
  }
  return true;
}

static double time_arm(Weight8 const &w, int nblk, int iters, float *out) {
  hipEvent_t a, b;
  hipEventCreate(&a);
  hipEventCreate(&b);
  // warm
  hipLaunchKernelGGL(k_stream, dim3(nblk), dim3(256), 0, 0, w, 1, out);
  hipDeviceSynchronize();
  hipEventRecord(a);
  hipLaunchKernelGGL(k_stream, dim3(nblk), dim3(256), 0, 0, w, iters, out);
  hipEventRecord(b);
  hipDeviceSynchronize();
  float ms = 0.0f;
  hipEventElapsedTime(&ms, a, b);
  hipEventDestroy(a);
  hipEventDestroy(b);
  return (double)ms * 1000.0 / (double)iters; // us per "layer"
}

int main(int argc, char **argv) {
  int const nblk = argc > 1 ? atoi(argv[1]) : 184;
  int const iters = argc > 2 ? atoi(argv[2]) : 50;

  hipDeviceProp_t prop;
  HIP_OK(hipGetDeviceProperties(&prop, 0));
  printf("=== Private8 placement gate: %s, %d CUs, grid %d ===\n",
         prop.gcnArchName, prop.multiProcessorCount, nblk);
  printf("  %.1f MB per layer, %.1f MB per XCD slice\n",
         kLayerBytes / 1e6, kSliceBytes / 1e6);

  float *out;
  HIP_OK(hipMalloc(&out, (size_t)nblk * sizeof(float)));

  double t_local = 0, t_remote = 0, t_shared = 0;
  {
    Weight8 w{};
    void *sp = nullptr;
    if (alloc_placed(w, true, false, sp)) {
      t_local = time_arm(w, nblk, iters, out);
    }
  }
  {
    Weight8 w{};
    void *sp = nullptr;
    if (alloc_placed(w, false, false, sp)) {
      t_remote = time_arm(w, nblk, iters, out);
    }
  }
  {
    Weight8 w{};
    void *sp = nullptr;
    if (alloc_placed(w, false, true, sp)) {
      t_shared = time_arm(w, nblk, iters, out);
    }
  }

  printf("\n  placement            us/layer   GB/s\n");
  printf("  AID-local (class A)  %8.2f   %6.0f\n", t_local,
         t_local > 0 ? kLayerBytes / t_local / 1e3 : 0.0);
  printf("  remote (anti-placed) %8.2f   %6.0f\n", t_remote,
         t_remote > 0 ? kLayerBytes / t_remote / 1e3 : 0.0);
  printf("  shared hipMalloc     %8.2f   %6.0f\n", t_shared,
         t_shared > 0 ? kLayerBytes / t_shared / 1e3 : 0.0);
  if (t_local > 0 && t_remote > 0) {
    printf("\n  local is %.2fx faster than remote  (reference: 24.2/15.7 = 1.54x)\n",
           t_remote / t_local);
    printf("  %s\n", (t_remote / t_local) > 1.25
                         ? "PASS -- Private8 places what it claims"
                         : "FAIL -- placement is not taking effect");
  }
  return 0;
}

