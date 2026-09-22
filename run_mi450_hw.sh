#!/usr/bin/env bash
# Bring-up sequence for Fleet on real MI450 (gfx1250) silicon.
#
# Run the steps in order. Each one exists because it can fail in a way that is
# invisible if you skip straight to the megakernel: a wrong XCD id gives wrong
# results with no error, a wrong tick rate gives wrong microseconds with no
# error, and an under-provisioned worker count looks like a performance problem
# rather than a build flag.
#
#   ./run_mi450_hw.sh check     # 0. the part is what we think it is
#   ./run_mi450_hw.sh tick      # 1. calibrate MIRAGE_TICK_NS
#   ./run_mi450_hw.sh xcd       # 2. the XCD id path, which has NEVER executed
#   ./run_mi450_hw.sh e2e       # 3. end-to-end, real worker count
#
# Nothing here is destructive and nothing leaves the machine.
set -euo pipefail

R="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-/tmp/mi450_hw}"
mkdir -p "$OUT"

build() { "$R/build_mi450_hw.sh" "$@"; }

case "${1:-help}" in

# ---------------------------------------------------------------------------
check)
  # Confirm this is gfx1250 and, importantly, that wavefrontSize is 32. The
  # entire port rests on wave32: every cross-lane reduction was re-derived for
  # it. If a part or runtime reports 64 here, stop -- the reductions are wrong
  # and they are wrong silently, producing plausible numbers.
  cat > "$OUT/devcheck.hip" <<'EOF'
#include <hip/hip_runtime.h>
#include <cstdio>
int main() {
  hipDeviceProp_t p{};
  if (hipGetDeviceProperties(&p, 0) != hipSuccess) { printf("FAIL: no device\n"); return 1; }
  printf("name=%s arch=%s CUs=%d wavefrontSize=%d sharedPerBlock=%zu KB\n",
         p.name, p.gcnArchName, p.multiProcessorCount, p.warpSize,
         p.sharedMemPerBlock / 1024);
  int rate = 0;
  hipDeviceGetAttribute(&rate, hipDeviceAttributeWallClockRate, 0);
  printf("wallClockRate=%d kHz\n", rate);
  if (p.warpSize != 32) {
    printf("FAIL: wavefrontSize %d, expected 32. Every cross-lane reduction in "
           "the mi450 port assumes wave32 and will be silently wrong.\n", p.warpSize);
    return 1;
  }
  printf("OVERALL: PASS\n");
  return 0;
}
EOF
  build "$OUT/devcheck.hip" -o "$OUT/devcheck"
  "$OUT/devcheck"
  ;;

# ---------------------------------------------------------------------------
tick)
  # MIRAGE_TICK_NS is currently the gfx950 value (10) carried over as an
  # admitted placeholder -- FFM returns 0 for the clock rate so it could not be
  # measured pre-silicon. Until this runs, every profiler microsecond the
  # megakernel prints is self-consistent but not absolute.
  build "$R/tests/mi450/hw/calibrate_tick_ns.hip" -o "$OUT/calib_tick"
  "$OUT/calib_tick"
  echo
  echo "Rebuild everything with TICK_NS=<printed value> ./build_mi450_hw.sh ..."
  ;;

# ---------------------------------------------------------------------------
xcd)
  # THE HIGHEST-RISK STEP. Read this before running it.
  #
  # gfx1250 has no HW_REG_XCC_ID, so xcd_id() uses
  #   __builtin_amdgcn_s_sendmsg_rtn(0x87)  -> RTN_GET_SE_HW_ID
  #   data[3:0] = SE_ID, data[19:16] = Virtual_XCC_ID
  # which matches Table 28 of the MI400 Shader Programming Guide. The encoding
  # is confirmed against the guide and it compiles, but it has NEVER BEEN
  # EXECUTED anywhere: FFM-Lite cannot decode the instruction at all (it aborts
  # with "Failed to decode instruction: s_sendmsg_rtn_b32 ... MSG_RTN_GET_SE_HW_ID",
  # verified), so every FFM run to date used -DMIRAGE_XCD_ID_FALLBACK=1 and saw
  # a constant 0. This program is its first execution, ever.
  #
  # Note the compiler disassembles 0x87 as MSG_RTN_GET_SE_AID_ID while FFM names
  # it MSG_RTN_GET_SE_HW_ID. Same opcode, two names, and neither tool agrees on
  # the mnemonic -- which is exactly why the returned VALUE must be checked
  # against reality here rather than assumed from the field layout.
  #
  # What a correct result looks like: across enough workgroups you should see
  # more than one distinct xcc value, and the set should cover 0..N-1 for the
  # part's XCD count. If every block reports 0, the field offset is wrong (or
  # this part reports a single virtual XCC) -- and note the runtime hardcodes 8
  # XCDs (NUM_XCDS_INIT / NUM_XCDS_PC at persistent_kernel.cuh:4477 and :4938,
  # inherited from MI300X). If this part is not 8, those constants are wrong
  # too, and the MoE tile decode `tile_idx * 8 + xcd_id` skips or duplicates
  # tiles with no diagnostic.
  cat > "$OUT/xcdprobe.hip" <<'EOF'
#include <hip/hip_runtime.h>
#include <cstdio>
#include <set>
__global__ void k(unsigned *out) {
  if (threadIdx.x == 0) {
    unsigned raw = (unsigned)__builtin_amdgcn_s_sendmsg_rtn(0x87);
    out[blockIdx.x] = raw;
  }
}
int main(int argc, char **argv) {
  int nb = (argc > 1) ? atoi(argv[1]) : 256;
  unsigned *d; hipMalloc(&d, nb * sizeof(unsigned));
  hipMemset(d, 0xFF, nb * sizeof(unsigned));
  hipLaunchKernelGGL(k, dim3(nb), dim3(64), 0, 0, d);
  hipError_t e = hipDeviceSynchronize();
  printf("sync: %s\n", hipGetErrorString(e));
  if (e != hipSuccess) { printf("FAIL: the sendmsg path faulted\n"); return 1; }
  unsigned *h = new unsigned[nb];
  hipMemcpy(h, d, nb * sizeof(unsigned), hipMemcpyDeviceToHost);
  std::set<unsigned> ses, xccs;
  for (int i = 0; i < nb; i++) { ses.insert(h[i] & 0xF); xccs.insert((h[i] >> 16) & 0xF); }
  printf("raw[0]=0x%08x  distinct SE_ID=%zu  distinct Virtual_XCC_ID=%zu\n",
         h[0], ses.size(), xccs.size());
  printf("XCC ids seen:"); for (unsigned v : xccs) printf(" %u", v); printf("\n");
  if (xccs.size() == 1 && *xccs.begin() == 0) {
    printf("WARN: every workgroup reports XCC 0 over %d blocks. Either this part "
           "presents one virtual XCC, or data[19:16] is the wrong field. Do NOT "
           "run the MoE path until this is resolved: the tile decode "
           "tile_idx*8+xcd_id would leave 7 of 8 tiles unexecuted.\n", nb);
    return 1;
  }
  printf("NOTE: runtime hardcodes 8 XCDs; %zu distinct seen. If these disagree, "
         "fix NUM_XCDS_INIT/NUM_XCDS_PC in persistent_kernel.cuh.\n", xccs.size());
  printf("OVERALL: PASS\n");
  return 0;
}
EOF
  build "$OUT/xcdprobe.hip" -o "$OUT/xcdprobe"
  "$OUT/xcdprobe" "${2:-256}"
  ;;

# ---------------------------------------------------------------------------
e2e)
  # The end-to-end harness at the real worker count. Under FFM this could only
  # run with 1 worker (the model keeps ~9 blocks of this megakernel co-resident;
  # a persistent kernel deadlocks if any block is not co-resident). Silicon has
  # no such ceiling, so this is the first run where task ordering is actually
  # exercised -- with one worker every task lands on the same queue and runs in
  # push order regardless of the dependency edges, which is why the mutation
  # test that deleted an edge SURVIVED under FFM.
  #
  # That makes this the test for task #15. Run it, then re-run the edge-deletion
  # mutant: with many workers a deleted dependency edge should now produce a
  # wrong token or a hang. If it still passes, the edges genuinely do nothing
  # and that is a real finding.
  build "$R/tests/mi450/e2e/test_e2e_mi450.hip" -o "$OUT/e2e450"
  echo "running with ${2:-64} workers (FFM was limited to 1)"
  "$OUT/e2e450" "${2:-64}"
  ;;

*)
  sed -n '2,20p' "${BASH_SOURCE[0]}"
  ;;
esac
