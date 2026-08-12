/* Copyright 2025 CMU
 *
 * Numerical test for the GLM-5 `noaux_tc` router
 * (tasks/mi300/moe_topk_sigmoid_bias_mi300.cuh).
 *
 * Reference semantics are aiter's biased_grouped_topk_torch
 * (~/aiter/aiter/ops/topk.py), specialized to n_group = topk_group = 1 as
 * GLM-5's config.json sets them, which makes the group masking a no-op:
 *
 *   scores            = sigmoid(gate_logits)
 *   scores_for_choice = scores + e_score_correction_bias
 *   topk_ids          = topk(scores_for_choice, k)
 *   topk_weights      = scores.gather(topk_ids)      <- UNBIASED
 *   topk_weights     /= topk_weights.sum()           (norm_topk_prob)
 *   topk_weights     *= routed_scaling_factor        (2.5)
 *
 * The kernel's weights are compared with a relative tolerance because it uses
 * the hardware-approximate v_exp_f32 / v_rcp_f32 pair for sigmoid (the same
 * fast_silu() trick as silu_mul_mi300.cuh), while the host reference uses
 * libm. Expert *ids* are required to match exactly; the generator rejects any
 * row whose 8th and 9th biased scores are closer than SEP_EPS, so an
 * approximation-induced reordering cannot be mistaken for correct behaviour.
 *
 * Build: ./build.sh   (or see the hipcc line in that script)
 * Run:   ./test_moe_topk_sigmoid_bias
 *
 * Exit code 0 = kernel matches the reference on every row.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <vector>

#include "mirage/persistent_kernel/mpk_atoms.cuh"
#include "mirage/persistent_kernel/tasks/mi300/moe_topk_sigmoid_bias_mi300.cuh"

// GLM-5: n_routed_experts = 256, num_experts_per_tok = 8,
// routed_scaling_factor = 2.5, norm_topk_prob = true.
static constexpr int NUM_EXPERTS = 256;
static constexpr int TOPK = 8;
static constexpr float ROUTED_SCALING = 2.5f;
// Minimum gap between the k-th and (k+1)-th biased score for a row to be used.
static constexpr float SEP_EPS = 1e-3f;

__global__ void topk_sigmoid_bias_entry(void *input,
                                        void *bias,
                                        void *output,
                                        int num_rows,
                                        void *routing_indices,
                                        void *active_expert_ids,
                                        int num_shared_experts) {
  kernel::topk_sigmoid_bias_mi300_task_impl<hip_bfloat16,
                                            8,
                                            NUM_EXPERTS,
                                            4,
                                            16>(
      input,
      bias,
      output,
      num_rows,
      TOPK,
      routing_indices,
      active_expert_ids,
      0,
      NUM_EXPERTS,
      /*renormalize=*/true,
      ROUTED_SCALING,
      num_shared_experts);
}

// Round-trip through bf16 so the host reference sees exactly the values the
// kernel loads.
static float to_bf16_and_back(float x) {
  uint32_t u;
  __builtin_memcpy(&u, &x, 4);
  // hip_bfloat16's host conversion is round-to-nearest-even.
  uint32_t lsb = (u >> 16) & 1u;
  u = (u + 0x7fffu + lsb) & 0xffff0000u;
  float r;
  __builtin_memcpy(&r, &u, 4);
  return r;
}

static uint16_t to_bf16_bits(float x) {
  float r = to_bf16_and_back(x);
  uint32_t u;
  __builtin_memcpy(&u, &r, 4);
  return (uint16_t)(u >> 16);
}

int main(int argc, char **argv) {
  int num_rows = (argc > 1) ? atoi(argv[1]) : 1;
  if (num_rows < 1 || num_rows > 8) {
    fprintf(stderr, "num_rows must be in [1, 8]\n");
    return 2;
  }
  // GLM's always-on shared expert: id NUM_EXPERTS in slot TOPK, weight 1.
  int const num_shared = (argc > 2) ? atoi(argv[2]) : 1;
  if (num_shared != 0 && num_shared != 1) {
    fprintf(stderr, "num_shared must be 0 or 1\n");
    return 2;
  }
  int const SLOTS = TOPK + num_shared;
  int const TOTAL_EXPERTS = NUM_EXPERTS + num_shared;

  srand(1234);
  auto urand = []() { return (float)rand() / (float)RAND_MAX; };

  std::vector<float> logits((size_t)num_rows * NUM_EXPERTS);
  std::vector<float> bias(NUM_EXPERTS);
  std::vector<uint16_t> h_input((size_t)num_rows * NUM_EXPERTS);
  std::vector<uint16_t> h_bias(NUM_EXPERTS);

  // e_score_correction_bias is a small learned offset; magnitudes in released
  // DeepSeek/GLM checkpoints sit well under 0.1.
  for (int e = 0; e < NUM_EXPERTS; ++e) {
    bias[e] = to_bf16_and_back((urand() - 0.5f) * 0.2f);
    h_bias[e] = to_bf16_bits(bias[e]);
  }

  // Host reference, recomputed per row; rows are regenerated until the top-k
  // boundary is unambiguous.
  std::vector<int> ref_ids((size_t)num_rows * TOPK);
  std::vector<float> ref_w((size_t)num_rows * TOPK);

  for (int row = 0; row < num_rows; ++row) {
    std::vector<std::pair<float, int>> biased(NUM_EXPERTS);
    std::vector<float> scores(NUM_EXPERTS);
    for (int attempt = 0;; ++attempt) {
      if (attempt > 100) {
        fprintf(stderr, "could not generate a well-separated row\n");
        return 2;
      }
      for (int e = 0; e < NUM_EXPERTS; ++e) {
        float l = to_bf16_and_back((urand() - 0.5f) * 8.0f);
        logits[(size_t)row * NUM_EXPERTS + e] = l;
        scores[e] = 1.0f / (1.0f + expf(-l));
        biased[e] = {scores[e] + bias[e], e};
      }
      // Descending by value, ties broken by lower expert id — the order the
      // kernel's v_cmp_gt_f32 argmax produces.
      std::stable_sort(biased.begin(),
                       biased.end(),
                       [](auto const &a, auto const &b) {
                         if (a.first != b.first) {
                           return a.first > b.first;
                         }
                         return a.second < b.second;
                       });
      if (biased[TOPK - 1].first - biased[TOPK].first > SEP_EPS) {
        break;
      }
    }
    for (int e = 0; e < NUM_EXPERTS; ++e) {
      h_input[(size_t)row * NUM_EXPERTS + e] =
          to_bf16_bits(logits[(size_t)row * NUM_EXPERTS + e]);
    }
    float sum = 0.f;
    for (int j = 0; j < TOPK; ++j) {
      ref_ids[(size_t)row * TOPK + j] = biased[j].second;
      ref_w[(size_t)row * TOPK + j] = scores[biased[j].second];
      sum += scores[biased[j].second];
    }
    for (int j = 0; j < TOPK; ++j) {
      ref_w[(size_t)row * TOPK + j] *= ROUTED_SCALING / sum;
    }
  }

  void *d_input, *d_bias, *d_output, *d_routing, *d_active;
  hipMalloc(&d_input, (size_t)num_rows * NUM_EXPERTS * 2);
  hipMalloc(&d_bias, (size_t)NUM_EXPERTS * 2);
  hipMalloc(&d_output, (size_t)num_rows * SLOTS * 4);
  hipMalloc(&d_routing, (size_t)TOTAL_EXPERTS * num_rows * 4);
  hipMalloc(&d_active, (size_t)(TOTAL_EXPERTS + 1) * 4);
  hipMemcpy(d_input,
            h_input.data(),
            (size_t)num_rows * NUM_EXPERTS * 2,
            hipMemcpyHostToDevice);
  hipMemcpy(
      d_bias, h_bias.data(), (size_t)NUM_EXPERTS * 2, hipMemcpyHostToDevice);
  hipMemset(d_output, 0, (size_t)num_rows * SLOTS * 4);
  hipMemset(d_routing, 0, (size_t)TOTAL_EXPERTS * num_rows * 4);
  hipMemset(d_active, 0, (size_t)(TOTAL_EXPERTS + 1) * 4);

  hipLaunchKernelGGL(topk_sigmoid_bias_entry,
                     dim3(1),
                     dim3(256),
                     0,
                     0,
                     d_input,
                     d_bias,
                     d_output,
                     num_rows,
                     d_routing,
                     d_active,
                     num_shared);
  hipError_t err = hipDeviceSynchronize();
  if (err != hipSuccess) {
    fprintf(stderr, "launch failed: %s\n", hipGetErrorString(err));
    return 2;
  }

  std::vector<float> out((size_t)num_rows * SLOTS);
  std::vector<int> routing((size_t)TOTAL_EXPERTS * num_rows);
  std::vector<int> active(TOTAL_EXPERTS + 1);
  hipMemcpy(out.data(),
            d_output,
            (size_t)num_rows * SLOTS * 4,
            hipMemcpyDeviceToHost);
  hipMemcpy(routing.data(),
            d_routing,
            (size_t)TOTAL_EXPERTS * num_rows * 4,
            hipMemcpyDeviceToHost);
  hipMemcpy(active.data(),
            d_active,
            (size_t)(TOTAL_EXPERTS + 1) * 4,
            hipMemcpyDeviceToHost);

  int failures = 0;
  float max_rel = 0.f;

  for (int row = 0; row < num_rows; ++row) {
    for (int j = 0; j < TOPK; ++j) {
      float got = out[(size_t)row * SLOTS + j];
      float want = ref_w[(size_t)row * TOPK + j];
      float rel = fabsf(got - want) / fmaxf(fabsf(want), 1e-6f);
      max_rel = fmaxf(max_rel, rel);
      if (rel > 1e-3f) {
        printf("row %d slot %d: weight %.7f != %.7f (rel %.2e)\n",
               row,
               j,
               got,
               want,
               rel);
        ++failures;
      }
    }
    // routing_indices[expert][row] == slot + 1 for the k selected experts, 0
    // everywhere else.
    std::vector<int> want_routing(TOTAL_EXPERTS, 0);
    for (int j = 0; j < TOPK; ++j) {
      want_routing[ref_ids[(size_t)row * TOPK + j]] = j + 1;
    }
    if (num_shared) {
      // The shared expert is unconditionally in slot TOPK of every token.
      want_routing[NUM_EXPERTS] = TOPK + 1;
      float const w = out[(size_t)row * SLOTS + TOPK];
      if (fabsf(w - 1.0f) > 1e-6f) {
        printf("row %d shared slot: weight %.7f != 1\n", row, w);
        ++failures;
      }
    }
    for (int e = 0; e < TOTAL_EXPERTS; ++e) {
      int got = routing[(size_t)e * num_rows + row];
      if (got != want_routing[e]) {
        printf("row %d expert %d: routing_index %d != %d\n",
               row,
               e,
               got,
               want_routing[e]);
        ++failures;
      }
    }
  }

  // active_expert_ids is only meaningful for a single row (all rows race on
  // the same k slots) — matching the existing softmax router's contract.
  if (num_rows == 1) {
    for (int j = 0; j < TOPK; ++j) {
      if (active[j] != ref_ids[j]) {
        printf(
            "slot %d: active_expert_id %d != %d\n", j, active[j], ref_ids[j]);
        ++failures;
      }
    }
  }
  if (num_shared && active[TOPK] != NUM_EXPERTS) {
    printf("shared slot: active_expert_id %d != %d\n", active[TOPK],
           NUM_EXPERTS);
    ++failures;
  }
  if (active[TOTAL_EXPERTS] != SLOTS) {
    printf("active_expert_ids[%d] = %d, expected %d\n",
           TOTAL_EXPERTS,
           active[TOTAL_EXPERTS],
           SLOTS);
    ++failures;
  }

  // Sanity: renormalized + scaled weights must sum to routed_scaling_factor.
  for (int row = 0; row < num_rows; ++row) {
    float s = 0.f;
    for (int j = 0; j < TOPK; ++j) {
      s += out[(size_t)row * SLOTS + j];
    }
    if (fabsf(s - ROUTED_SCALING) > 1e-3f) {
      printf("row %d: weights sum to %.6f, expected %.6f\n",
             row,
             s,
             ROUTED_SCALING);
      ++failures;
    }
  }

  hipFree(d_input);
  hipFree(d_bias);
  hipFree(d_output);
  hipFree(d_routing);
  hipFree(d_active);

  if (failures == 0) {
    printf("PASS: %d row(s), %d experts (+%d shared), top-%d — max weight rel "
           "err %.2e\n",
           num_rows,
           NUM_EXPERTS,
           num_shared,
           TOPK,
           max_rel);
    return 0;
  }
  printf("FAIL: %d mismatch(es)\n", failures);
  return 1;
}
