// Measure nanoseconds-per-realtime-tick on gfx1250, for -DMIRAGE_TICK_NS=<n>.
//
// Why this exists: arch_traits.cuh carries MIRAGE_TICK_NS 10, inherited from
// gfx950 (s_memrealtime ~100 MHz). For gfx1250 that number is a placeholder and
// is documented as such -- FFM-Lite returns 0 for
// hipDeviceAttributeWallClockRate, so the model cannot supply it and no gfx1250
// part was available while the port was written. Every profiler figure
// (profiler.h:136, persistent_kernel.cuh:397 and :2610) is scaled by it, so
// until it is measured the megakernel's microsecond numbers are self-consistent
// but not absolute. This program is the first thing to run on real silicon,
// before believing any timing output.
//
// It cannot fail closed. A wrong tick rate does not corrupt results, it only
// mislabels them -- which is exactly why it is easy to never notice.
//
// Method: two independent estimates that must agree.
//
//   (1) The device tells you.  hipDeviceAttributeWallClockRate reports the
//       constant REFCLK frequency in kHz. If the runtime knows it, that is
//       authoritative and needs no timing loop.
//
//   (2) Measure it.  Spin on the device wall clock for a fixed tick count while
//       the host times the kernel with a steady clock. Ticks/second follows.
//
// Agreement between a reported value and a measured one is the actual check.
// A single number from either source alone could be wrong in a way nothing here
// would reveal, so both are printed and compared rather than one being trusted.
//
// Build (from the repo root):
//   ./build_mi450_hw.sh tests/mi450/hw/calibrate_tick_ns.hip -o /tmp/calib_tick
// Run on the part, then rebuild everything with the printed value:
//   TICK_NS=<n> ./build_mi450_hw.sh ...

#include <chrono>
#include <cstdio>
#include <hip/hip_runtime.h>

// wall_clock64() lowers to s_sendmsg_rtn_b64 sendmsg(MSG_RTN_GET_REALTIME) on
// gfx1250 -- the same path realtime_ticks() uses in arch_traits.cuh, so this
// measures the counter the profiler actually reads, not a similar one.
//
// MI400 Shader Programming Guide, Table 28: RTN_GET_REALTIME (SSRC0 0x83) is
// "the constant frequency (REFCLK) time counter". Constant-frequency matters:
// it does not vary with the shader clock, so one calibration holds across DVFS
// states and this program does not need to pin clocks to be valid.
__global__ void spin_ticks(unsigned long long target_ticks,
                           unsigned long long *out_elapsed) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    unsigned long long t0 = wall_clock64();
    unsigned long long t1 = t0;
    // Re-read until the counter has advanced by target_ticks. The loop body is
    // deliberately trivial: we are timing the counter, not the loop.
    while ((t1 - t0) < target_ticks) {
      t1 = wall_clock64();
    }
    *out_elapsed = t1 - t0;
  }
}

int main(int argc, char **argv) {
  int dev = 0;
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, dev) != hipSuccess) {
    printf("FAIL: no device\n");
    return 1;
  }
  printf("device: %s (gcnArch %s), wavefrontSize %d\n",
         prop.name,
         prop.gcnArchName,
         prop.warpSize);

  // ---- Estimate (1): ask the runtime -------------------------------------
  int wall_khz = 0;
  hipError_t ae =
      hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, dev);
  double reported_ns = 0.0;
  if (ae == hipSuccess && wall_khz > 0) {
    // kHz -> ns/tick:  1e6 ns/ms / (ticks/ms)
    reported_ns = 1.0e6 / (double)wall_khz;
    printf("reported : hipDeviceAttributeWallClockRate = %d kHz -> %.4f ns/tick\n",
           wall_khz,
           reported_ns);
  } else {
    // This is the FFM-Lite behaviour, and possibly some ROCm builds. Not fatal:
    // estimate (2) stands on its own, it just loses the cross-check.
    printf("reported : unavailable (rate=%d, %s) -- measurement below is "
           "unchecked, treat with corresponding suspicion\n",
           wall_khz,
           hipGetErrorString(ae));
  }

  // ---- Estimate (2): time a known number of ticks -------------------------
  unsigned long long *d_elapsed = nullptr;
  if (hipMalloc(&d_elapsed, sizeof(unsigned long long)) != hipSuccess) {
    printf("FAIL: hipMalloc\n");
    return 1;
  }

  // Warm up so first-launch overhead is not folded into the timed run. Kernel
  // launch latency is a fixed additive error here: it inflates the measured
  // wall time without inflating the tick count, biasing ns/tick high. Running
  // long enough that launch overhead is negligible is the mitigation, hence the
  // 100 ms-ish target rather than a short burst.
  unsigned long long const warm = 1000ull;
  hipLaunchKernelGGL(spin_ticks, dim3(1), dim3(64), 0, 0, warm, d_elapsed);
  if (hipDeviceSynchronize() != hipSuccess) {
    printf("FAIL: warmup launch (wall_clock64 unsupported here?)\n");
    return 1;
  }

  // Default target assumes roughly 100 MHz, giving ~0.1 s. If the real clock is
  // far from that the run gets shorter or longer but stays correct -- the tick
  // count is exact by construction; only wall time is measured.
  unsigned long long target = (argc > 1) ? strtoull(argv[1], nullptr, 10)
                                         : 10000000ull;

  double best_ns = 1e30;
  double results[3];
  for (int rep = 0; rep < 3; rep++) {
    auto h0 = std::chrono::steady_clock::now();
    hipLaunchKernelGGL(spin_ticks, dim3(1), dim3(64), 0, 0, target, d_elapsed);
    hipError_t e = hipDeviceSynchronize();
    auto h1 = std::chrono::steady_clock::now();
    if (e != hipSuccess) {
      printf("FAIL: launch %d: %s\n", rep, hipGetErrorString(e));
      return 1;
    }
    unsigned long long ticks = 0;
    hipMemcpy(&ticks, d_elapsed, sizeof(ticks), hipMemcpyDeviceToHost);
    double ns =
        std::chrono::duration<double, std::nano>(h1 - h0).count();
    double ns_per_tick = ns / (double)ticks;
    results[rep] = ns_per_tick;
    printf("measured : rep %d  ticks=%llu  wall=%.3f ms  -> %.4f ns/tick\n",
           rep,
           ticks,
           ns / 1.0e6,
           ns_per_tick);
    // The minimum across reps is the right estimator, not the mean: every
    // source of error here (launch overhead, host scheduling, preemption) adds
    // wall time and none subtracts it, so the fastest observation is the least
    // contaminated one.
    if (ns_per_tick < best_ns) {
      best_ns = ns_per_tick;
    }
  }

  // Spread across reps is the honesty check on the measurement. Tight spread
  // means the number is repeatable; wide spread means something (a busy host, a
  // migrating thread) is polluting it and the value should not be baked in.
  double spread = 0.0;
  for (int i = 0; i < 3; i++) {
    double d = results[i] - best_ns;
    if (d > spread) {
      spread = d;
    }
  }
  printf("measured : best %.4f ns/tick (spread %.4f ns, %.1f%%)\n",
         best_ns,
         spread,
         100.0 * spread / best_ns);
  if (spread / best_ns > 0.02) {
    printf("WARN: >2%% spread across reps -- rerun on an idle host before "
           "trusting this\n");
  }

  if (reported_ns > 0.0) {
    double rel = (best_ns - reported_ns) / reported_ns;
    if (rel < 0) {
      rel = -rel;
    }
    printf("cross-check: measured vs reported differ by %.2f%%\n", 100.0 * rel);
    if (rel > 0.05) {
      // Deliberately not a silent pass. Disagreement means one of the two is
      // measuring something other than what it claims, and baking in either
      // number without resolving that would put a wrong scale factor into every
      // profiler figure the port produces from then on.
      printf("FAIL: >5%% disagreement. Do not bake in either value until this "
             "is explained.\n");
      return 1;
    }
  }

  // Prefer the reported value when it agrees: it is exact, whereas the measured
  // one carries host-timing noise.
  double chosen = (reported_ns > 0.0) ? reported_ns : best_ns;
  printf("\nUse: -DMIRAGE_TICK_NS=%.0f   (exact %.4f ns/tick, %.3f MHz)\n",
         chosen,
         chosen,
         1000.0 / chosen);
  if (chosen < 0.5) {
    // MIRAGE_TICK_NS is consumed as an integer multiplier. A sub-nanosecond
    // tick cannot be represented and would silently round to 0 or 1, so say so
    // rather than print a number that will be wrong when used.
    printf("WARN: tick is sub-nanosecond; MIRAGE_TICK_NS is an integer and "
           "cannot express this. Rescale at the profiler boundary instead.\n");
  }
  if ((double)(long long)(chosen + 0.5) != chosen) {
    printf("NOTE: rounding %.4f -> %lld introduces %.2f%% error in every "
           "reported microsecond figure.\n",
           chosen,
           (long long)(chosen + 0.5),
           100.0 * ((double)(long long)(chosen + 0.5) - chosen) / chosen);
  }
  printf("OVERALL: PASS\n");
  hipFree(d_elapsed);
  return 0;
}
