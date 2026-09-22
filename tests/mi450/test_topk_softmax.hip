// test_topk_softmax.hip -- MoE router (softmax + top-k) on gfx1250.
//
// This is the wave-width test, not a numerics test. topk_softmax_mi300 used to
// contain three hand-written v_cmp/v_cndmask blocks that named `vcc` as an
// explicit operand, and MI300_WARP_SIZE was a literal 64. Both are wrong at
// wave32:
//
//   - `vcc` is 64-bit on gfx9 and 32-bit on gfx12, so the asm would either
//     fail to assemble or mask the wrong lanes.
//   - MI300_WARP_SIZE drives the row-to-lane mapping (ROWS_PER_WARP,
//     thread_row, thread_group_idx) and the compaction stride. At wave32 with
//     a literal 64 the kernel addresses rows that its wave does not own, so
//     half the rows are silently never written -- which a spot check of row 0
//     would not catch.
//
// So this test checks *every* row, and pre-poisons the outputs so an untouched
// row fails rather than passing by accident.
//
// EXACTNESS
//
// Softmax involves exp and a reciprocal, so bit-exactness against a host double
// reference is not available the way it is for the GEMM tests. The reference
// here is instead structural, and it is exact:
//
//   - inputs are built so that each row's expert scores are *distinct* and
//     separated by wide margins, which makes the top-k *selection* (the ids and
//     their order) a discrete, exactly-checkable fact regardless of rounding.
//   - the returned weights are checked against a host softmax in double with a
//     loose tolerance, since only the selection logic was ported.
//   - active_expert_ids must come out ascending, deduplicated, and sentinel
//     (-1) filled -- the property the MoE tile decode depends on. That is
//     checked exactly.
//
// A softmax that is off by an ulp still passes; a wave-width bug cannot,
// because it moves which experts are selected and which rows are written.

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "arch_traits.cuh"
#include "tasks/mi300/moe_topk_softmax_mi300.cuh"

#define CHECK(x)                                                               \
  do {                                                                         \
    hipError_t _e = (x);                                                       \
    if (_e != hipSuccess) {                                                    \
      printf("HIP error %s at line %d\n", hipGetErrorString(_e), __LINE__);     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static constexpr int NUM_EXPERTS = 32;
static constexpr int VPT = 8;
static constexpr int WARPS_PER_CTA = 4;
static constexpr int BYTES_PER_LDG = 16;
static constexpr int TOPK = 4;
static int NUM_ROWS = 16; // overridable from argv to expose row-guard bugs
static int BATCH = 16;                    // routing_row_stride

__global__ void run_topk(void *in,
                         void *out,
                         int num_rows,
                         int batch,
                         int k,
                         int *routing_indices,
                         int *active_expert_ids) {
  kernel::topk_softmax_mi300_task_impl<__hip_bfloat16,
                                       VPT,
                                       NUM_EXPERTS,
                                       WARPS_PER_CTA,
                                       BYTES_PER_LDG>(
      in,
      out,
      num_rows,
      /*routing_row_stride=*/batch,
      k,
      routing_indices,
      active_expert_ids,
      /*start_expert=*/0,
      /*end_expert=*/NUM_EXPERTS,
      /*renormalize=*/false);
}

int main(int argc, char **argv) {
  if (argc > 1) {
    NUM_ROWS = atoi(argv[1]);
  }
  BATCH = NUM_ROWS;
  // ---- build inputs -------------------------------------------------------
  // Row r gives expert e the score ((e + r) % NUM_EXPERTS) * 0.5. Distinct
  // within a row, and the winning set *rotates* with r, so a kernel that mixes
  // rows up produces the wrong expert ids rather than coincidentally-right
  // ones. bf16 represents every one of these exactly (multiples of 0.5 below
  // 16), so the input is not itself a source of error.
  std::vector<float> h_in_f(NUM_ROWS * NUM_EXPERTS);
  std::vector<__hip_bfloat16> h_in(NUM_ROWS * NUM_EXPERTS);
  for (int r = 0; r < NUM_ROWS; ++r) {
    for (int e = 0; e < NUM_EXPERTS; ++e) {
      float v = (float)((e + r) % NUM_EXPERTS) * 0.5f;
      h_in_f[r * NUM_EXPERTS + e] = v;
      h_in[r * NUM_EXPERTS + e] = __hip_bfloat16(v);
    }
  }

  __hip_bfloat16 *d_in = nullptr;
  float *d_out = nullptr;
  int *d_routing = nullptr;
  int *d_active = nullptr;
  CHECK(hipMalloc(&d_in, h_in.size() * sizeof(__hip_bfloat16)));
  CHECK(hipMalloc(&d_out, NUM_ROWS * TOPK * sizeof(float)));
  CHECK(hipMalloc(&d_routing, NUM_EXPERTS * BATCH * sizeof(int)));
  CHECK(hipMalloc(&d_active, (2 * NUM_EXPERTS + 1) * sizeof(int)));
  CHECK(hipMemcpy(d_in,
                  h_in.data(),
                  h_in.size() * sizeof(__hip_bfloat16),
                  hipMemcpyHostToDevice));

  // Poison every output. A row the kernel never touches must fail, not inherit
  // a plausible zero.
  CHECK(hipMemset(d_out, 0xCD, NUM_ROWS * TOPK * sizeof(float)));
  CHECK(hipMemset(d_routing, 0, NUM_EXPERTS * BATCH * sizeof(int)));
  CHECK(hipMemset(d_active, 0xCD, (2 * NUM_EXPERTS + 1) * sizeof(int)));

  // Block size must be WARPS_PER_CTA waves of the *device's* wave width.
  //
  // Neither obvious spelling works here. `warpSize` is device-only in HIP and
  // not a constant expression. `mirage::arch::WAVE_SIZE` is a constant, but in
  // the *host* pass it falls back to 64 (host translation units carry no
  // __gfx*__ macro), so on a wave32 part it would launch 8 waves where the
  // kernel's row-to-lane mapping expects 4 -- an over-launch that puts warps
  // 4..7 on rows the caller never asked for. Ask the driver instead: the
  // device pass and the device agree on the real width.
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, 0));
  int const dev_wave = prop.warpSize;

  hipLaunchKernelGGL(run_topk,
                     dim3(1),
                     dim3(WARPS_PER_CTA * dev_wave),
                     0,
                     0,
                     d_in,
                     d_out,
                     NUM_ROWS,
                     BATCH,
                     TOPK,
                     d_routing,
                     d_active);
  CHECK(hipDeviceSynchronize());

  std::vector<float> h_out(NUM_ROWS * TOPK);
  std::vector<int> h_routing(NUM_EXPERTS * BATCH);
  std::vector<int> h_active(2 * NUM_EXPERTS + 1);
  CHECK(hipMemcpy(h_out.data(),
                  d_out,
                  h_out.size() * sizeof(float),
                  hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(h_routing.data(),
                  d_routing,
                  h_routing.size() * sizeof(int),
                  hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(h_active.data(),
                  d_active,
                  h_active.size() * sizeof(int),
                  hipMemcpyDeviceToHost));

  int failures = 0;

  // ---- 1. weights: host softmax in double, loose tolerance ---------------
  // Only the *selection* was ported, so the weights only need to be sane.
  for (int r = 0; r < NUM_ROWS; ++r) {
    double mx = -1e30, sum = 0.0;
    for (int e = 0; e < NUM_EXPERTS; ++e) {
      mx = std::max(mx, (double)h_in_f[r * NUM_EXPERTS + e]);
    }
    std::vector<double> p(NUM_EXPERTS);
    for (int e = 0; e < NUM_EXPERTS; ++e) {
      p[e] = std::exp((double)h_in_f[r * NUM_EXPERTS + e] - mx);
      sum += p[e];
    }
    for (int e = 0; e < NUM_EXPERTS; ++e) {
      p[e] /= sum;
    }

    // Expected top-k: the k largest scores. Scores in a row are distinct, so
    // this ordering is unambiguous.
    std::vector<int> order(NUM_EXPERTS);
    for (int e = 0; e < NUM_EXPERTS; ++e) {
      order[e] = e;
    }
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      return h_in_f[r * NUM_EXPERTS + a] > h_in_f[r * NUM_EXPERTS + b];
    });

    for (int j = 0; j < TOPK; ++j) {
      double want = p[order[j]];
      double got = (double)h_out[r * TOPK + j];
      // Poison check: 0xCDCDCDCD as a float is about -4.3e8.
      if (!(got > -1.0 && got < 2.0)) {
        printf("FAIL row %d slot %d: output never written (got %g)\n",
               r,
               j,
               got);
        ++failures;
        continue;
      }
      if (std::fabs(got - want) > 1e-3 * std::max(1.0, want)) {
        printf("FAIL row %d slot %d: weight %g want %g (expert %d)\n",
               r,
               j,
               got,
               want,
               order[j]);
        ++failures;
      }
    }

    // ---- 2. selection: routing_indices must mark exactly the k winners ----
    // routing_indices[e * BATCH + r] == j+1 iff expert e was this row's j-th
    // pick, 0 otherwise. This is the check a wave-width bug cannot survive: it
    // is per (expert, row), so a wave that addresses the wrong rows lands the
    // marks in the wrong columns.
    for (int e = 0; e < NUM_EXPERTS; ++e) {
      int want_rank = 0;
      for (int j = 0; j < TOPK; ++j) {
        if (order[j] == e) {
          want_rank = j + 1;
        }
      }
      int got_rank = h_routing[e * BATCH + r];
      if (got_rank != want_rank) {
        printf("FAIL routing[expert=%d][row=%d]: got %d want %d\n",
               e,
               r,
               got_rank,
               want_rank);
        ++failures;
      }
    }
  }

  // ---- 3. active_expert_ids: ascending, deduplicated, sentinel filled -----
  // The MoE tile decode reads this per slot and must be able to tell padding
  // from a live expert without consulting the count, so the sentinel fill is
  // as load-bearing as the ids.
  int count = h_active[NUM_EXPERTS];
  std::vector<char> want_active(NUM_EXPERTS, 0);
  for (int r = 0; r < NUM_ROWS; ++r) {
    std::vector<int> order(NUM_EXPERTS);
    for (int e = 0; e < NUM_EXPERTS; ++e) {
      order[e] = e;
    }
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      return h_in_f[r * NUM_EXPERTS + a] > h_in_f[r * NUM_EXPERTS + b];
    });
    for (int j = 0; j < TOPK; ++j) {
      want_active[order[j]] = 1;
    }
  }
  std::vector<int> want_ids;
  for (int e = 0; e < NUM_EXPERTS; ++e) {
    if (want_active[e]) {
      want_ids.push_back(e);
    }
  }

  if (count != (int)want_ids.size()) {
    printf("FAIL active count: got %d want %zu\n", count, want_ids.size());
    ++failures;
  }
  for (int i = 0; i < (int)want_ids.size() && i < NUM_EXPERTS; ++i) {
    if (h_active[i] != want_ids[i]) {
      printf("FAIL active_expert_ids[%d]: got %d want %d\n",
             i,
             h_active[i],
             want_ids[i]);
      ++failures;
    }
  }
  for (int i = (int)want_ids.size(); i < NUM_EXPERTS; ++i) {
    if (h_active[i] != -1) {
      printf("FAIL active_expert_ids[%d]: got %d want sentinel -1\n",
             i,
             h_active[i]);
      ++failures;
    }
  }

  // On failure, dump the active-expert set: the routing marks and the active
  // set are written from the same branch, so seeing both says which is lying.
  if (failures != 0) {
    printf("[dbg] active ids:");
    for (int i = 0; i < NUM_EXPERTS; ++i) {
      printf(" %d", h_active[i]);
    }
    printf(" | count=%d\n", h_active[NUM_EXPERTS]);
  }

  printf("wave_size=%d (host arch_traits says %d) rows=%d experts=%d k=%d "
         "active=%d\n",
         dev_wave,
         mirage::arch::WAVE_SIZE,
         NUM_ROWS,
         NUM_EXPERTS,
         TOPK,
         count);
  if (failures == 0) {
    printf("PASS: topk_softmax selection, routing marks and expert compaction "
           "are correct at wave%d.\n",
           dev_wave);
    return 0;
  }
  printf("FAIL: %d check(s) failed.\n", failures);
  return 1;
}
