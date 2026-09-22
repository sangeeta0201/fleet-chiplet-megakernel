// Does the MoE tile decode cover every tile exactly once across 8 XCDs?
//
// WHY THIS TEST EXISTS
//
// The MoE kernels distribute work with
//     global_tile = tile_idx * 8 + xcd_id()
// (gang_moe_linear_mxfp4_mi450.cuh:327, gang_moe_fused_mxfp4_mi450.cuh:512).
// On gfx1250, xcd_id() is __builtin_amdgcn_s_sendmsg_rtn(0x87), and that
// instruction has never executed anywhere: FFM-Lite aborts at decode, and AM
// decodes it and then hangs the wave forever with no diagnostic. So every
// model run of the port so far used -DMIRAGE_XCD_ID_FALLBACK=1, i.e. a
// constant 0.
//
// A constant 0 makes the decode collapse to `tile_idx * 8`. Only tiles that
// are multiples of 8 are ever produced -- 7 of every 8 tiles go unexecuted
// while the kernel still returns successfully with plausible numbers for the
// tiles it did reach. That is precisely the failure this test is here to
// catch, and it cannot be caught by a numerical check on a single XCD.
//
// WHAT THIS TEST DOES AND DOES NOT PROVE
//
// It substitutes blockIdx.y for the XCD id (the MIRAGE_XCD_ID_FROM_BLOCKIDX_Y
// seam) and launches gridDim.y == 8, so the decode is exercised over the full
// range of XCD values on one modeled die. It asserts every tile in
// [0, total_tiles) is produced EXACTLY once -- catching both gaps (skipped
// work, wrong results) and duplicates (two XCDs computing the same tile,
// which for an accumulating kernel means doubled output).
//
// It does NOT prove:
//   - that xcd_id() returns the right value on real hardware. Nothing can
//     prove that pre-silicon; the instruction does not run on either model.
//   - that the 8 blocks are actually on 8 different XCDs. They are not. This
//     enumerates the indices a real 8-XCD dispatch would produce.
//   - that 8 is the right XCD count for the part. NUM_XCDS_PC is hardcoded to
//     8 (persistent_kernel.cuh:4477), inherited from MI300X and unverified
//     against MI450. If the real part is not 8, this test passes and the
//     kernel is still wrong. See run_mi450_hw.sh xcd.
//
// So: a PASS here means the tile arithmetic is right given a correct xcd_id().
// It converts "two unknowns" into one, and the remaining one is silicon-only.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef MIRAGE_XCD_ID_FROM_BLOCKIDX_Y
#error "build this test with -DMIRAGE_XCD_ID_FROM_BLOCKIDX_Y=1 -- otherwise " \
       "xcd_id() is either a constant 0 (proves nothing) or a real sendmsg " \
       "(hangs AM, aborts FFM)"
#endif

#define NUM_XCDS 8

// Mirror of the decode in gang_moe_linear_mxfp4_mi450.cuh. Kept as a separate
// literal copy on purpose: if someone changes the kernel's formula, this test
// should FAIL rather than silently follow the change. A shared helper would
// make the test agree with the kernel by construction and test nothing.
__global__ void count_tiles(int *visits, int total_tiles, int tiles_per_grid) {
  int const my_xcd = (int)blockIdx.y;   // the seam under test
  for (int tile_idx = blockIdx.x; tile_idx < tiles_per_grid;
       tile_idx += gridDim.x) {
    if (threadIdx.x != 0) continue;
    int global_tile = tile_idx * NUM_XCDS + my_xcd;
    if (global_tile >= total_tiles) continue;   // same guard as the kernel
    atomicAdd(&visits[global_tile], 1);
  }
}

int main(int argc, char **argv) {
  // Default shape mirrors the e2e config: 8 experts x 2 tiles per expert.
  int num_experts    = (argc > 1) ? atoi(argv[1]) : 8;
  int tiles_per_expert = (argc > 2) ? atoi(argv[2]) : 2;
  int total_tiles    = num_experts * tiles_per_expert;
  // ceil, so the last partial group is covered and the >= guard is exercised
  int tiles_per_grid = (total_tiles + NUM_XCDS - 1) / NUM_XCDS;

  printf("experts=%d tiles/expert=%d total_tiles=%d tiles_per_grid=%d\n",
         num_experts, tiles_per_expert, total_tiles, tiles_per_grid);

  int *d_visits;
  if (hipMalloc(&d_visits, total_tiles * sizeof(int)) != hipSuccess) {
    printf("FAIL: hipMalloc\n");
    return 1;
  }
  // Poison, so an untouched buffer cannot masquerade as a pass.
  hipMemset(d_visits, 0, total_tiles * sizeof(int));

  hipLaunchKernelGGL(count_tiles, dim3(tiles_per_grid, NUM_XCDS), dim3(32),
                     0, 0, d_visits, total_tiles, tiles_per_grid);
  hipError_t e = hipDeviceSynchronize();
  if (e != hipSuccess) {
    printf("FAIL: kernel: %s\n", hipGetErrorString(e));
    return 1;
  }

  std::vector<int> h(total_tiles, -1);
  hipMemcpy(h.data(), d_visits, total_tiles * sizeof(int), hipMemcpyDeviceToHost);

  int gaps = 0, dups = 0;
  for (int t = 0; t < total_tiles; t++) {
    if (h[t] == 0) { if (gaps < 8) printf("  GAP: tile %d never computed\n", t); gaps++; }
    else if (h[t] > 1) { if (dups < 8) printf("  DUP: tile %d computed %dx\n", t, h[t]); dups++; }
  }

  printf("covered=%d gaps=%d duplicates=%d\n", total_tiles - gaps, gaps, dups);

  if (gaps || dups) {
    printf("FAIL: tile decode does not partition the tile space.\n");
    if (gaps == total_tiles - (total_tiles + NUM_XCDS - 1) / NUM_XCDS) {
      printf("      The gap count matches what a constant xcd_id()==0 would\n"
             "      produce -- check that MIRAGE_XCD_ID_FROM_BLOCKIDX_Y is\n"
             "      really compiled in and gridDim.y is 8.\n");
    }
    return 1;
  }

  // Non-vacuity: the test must be able to fail. Re-run with every block
  // claiming XCD 0 (what the FFM fallback does) and confirm we detect it.
  hipMemset(d_visits, 0, total_tiles * sizeof(int));
  hipLaunchKernelGGL(count_tiles, dim3(tiles_per_grid, 1), dim3(32),
                     0, 0, d_visits, total_tiles, tiles_per_grid);
  hipDeviceSynchronize();
  hipMemcpy(h.data(), d_visits, total_tiles * sizeof(int), hipMemcpyDeviceToHost);
  int gaps0 = 0;
  for (int t = 0; t < total_tiles; t++) if (h[t] == 0) gaps0++;
  if (total_tiles > NUM_XCDS && gaps0 == 0) {
    printf("FAIL: the single-XCD control ALSO reported full coverage, so this\n"
           "      test cannot distinguish a broken xcd_id() from a working\n"
           "      one. The pass above is meaningless.\n");
    return 1;
  }
  printf("non-vacuity: single-XCD control leaves %d/%d tiles uncovered, as expected\n",
         gaps0, total_tiles);

  printf("OVERALL: PASS\n");
  printf("NOTE: this validates the tile arithmetic, NOT that xcd_id() returns\n"
         "      a correct value on silicon -- that instruction still has never\n"
         "      executed. Run ./run_mi450_hw.sh xcd on hardware.\n");
  return 0;
}
