// Phase 6 -- torch-free end-to-end run of the MPK megakernel on gfx1250.
//
// WHAT THIS IS FOR
//
// Every mi450 test before this one launches a single task kernel directly, with
// the host placing its inputs and reading its outputs. That validates
// arithmetic and nothing else. This one runs a *task graph* through the real
// persistent megakernel: the scheduler hands out task ids, workers pull them
// off their queues, each task's completion triggers an event, and the event
// releases its dependents. The values therefore have to survive the actual
// dispatch path -- correct pointers in the TaskDesc, correct event wiring,
// correct ordering between a producer and its consumer.
//
// The distinction matters because it is precisely the layer the per-kernel
// tests cannot reach. A kernel can be bit-exact in isolation and still receive
// the wrong buffer, run before its input exists, or never run at all.
//
// THE MODEL
//
// A 2-layer synthetic transformer-shaped chain, deliberately built from the
// tasks that are actually ported for gfx1250 rather than from GPT-OSS's real
// layer (whose orchestrators are not ported yet -- see task_header.cuh):
//
//   embedding -> [ layer 0: rmsnorm -> linear ] -> [ layer 1: same ] -> argmax
//
// Each arrow is a real data dependency expressed as an MPK event, so a task
// that runs early reads poison and the check fails.
//
// WHY THE CHAIN IS SHAPED LIKE THIS
//
// It ends in argmax on purpose. argmax is the task that picks the emitted
// token, so it converts every upstream numerical error into a *discrete*
// failure: the harness asserts the exact token id, not a tolerance. It is also
// where a live wave-width bug was found while writing this file -- utils.cuh's
// lane_id() masked with 0x3f, so at wave32 only one thread per two waves wrote
// the cross-wave scratch and the block reduction silently dropped half the
// candidates (measured: returned 16 where the true max was 1000). A logit
// tolerance would not have caught that; an exact token id does.
//
// CORRECTNESS WITHOUT TORCH
//
// PyTorch has no gfx1250 code objects (torch.ones(device='cuda') fails with
// hipErrorInvalidImage), so there is no reference implementation to compare
// against. Instead, following tests/mi450/test_gemm_wmma.hip: all weights and
// activations are small multiples of powers of two, so every product and every
// partial sum in the matmuls is exactly representable and never rounds.
//
// Note the limit of that trick, which cost a debugging round to notice: it does
// NOT make this chain bit-exact, because RMS norm scales by
// rsqrtf(mean + eps), which is irrational. Everything downstream of the first
// norm is a rounded f32. The check is therefore a scaled tolerance (~1-2 ulp,
// derived below), not equality -- but the *token* assertion is still exact,
// which is the point of ending the chain in argmax.
//
// Three anti-false-pass measures, each of which has caught a real bug in this
// port:
//   * poison. Every buffer is pre-filled with a NaN payload and asserted to
//     have been overwritten. A task that never ran cannot pass as a zero.
//   * isfinite before tolerance. A NaN passes every `if (rel > TOL)` check,
//     because comparisons with NaN are false -- a kernel emitting NaN
//     everywhere reports a worst error of 0.0 and looks perfect. This is how a
//     wave64 mutant survived test_kv_cache_update until the check was added.
//   * a discriminating-power assertion on the reference itself, so the test
//     cannot pass because the data was degenerate.
//
// WHAT A PASS DOES NOT MEAN
//
// FFM-Lite is functional only, and time-slices workgroups rather than running
// them concurrently. So this shows the graph is wired correctly and the
// arithmetic survives the dispatch path. It does NOT show the memory ordering
// is right: serial execution hides missing s_wait_*, which is the bug class the
// gfx9 -> gfx12 wait-counter split introduces most. It also cannot see
// performance, occupancy, or cross-XCD behaviour. Green here is necessary and
// not sufficient; silicon is still required.
//
// One specific gap, measured rather than assumed. Three mutants were run
// against this harness:
//
//   deleted the argmax task's dependent_event edge   -> SURVIVED
//   wave64 lane mask in the argmax cross-wave fold   -> killed (NaN + token)
//   swapped the two layers' norm weights             -> killed (token 57 vs 55)
//
// The survivor is not a hole in the assertions, it is a hole in the *platform*,
// and it bounds what this file can claim. FFM co-schedules only 2 blocks for
// this binary -- verified by running the known-good test_hello_megakernel,
// which passes at 1 worker and hangs at 2 with the scheduler still waiting on
// worker 1. Two blocks means one scheduler and one worker, so every task lands
// in a single queue and executes in push order whether or not the edges exist.
// Dependency ordering is therefore untestable here by construction, exactly as
// barriers are untestable under FFM. It needs >= 3 co-resident blocks, i.e.
// silicon or a larger model budget.
//
// Stated plainly: this file verifies the dispatch path and the values. It does
// not verify that the ordering edges do anything.
//
// Build:
//   /tmp/p1/cctest450y.sh tests/mi450/e2e/test_e2e_mi450.hip -o <bin>
// Run:
//   HSA_MODEL_ARGS="ffm_enable_time_slicing" ./run_ffm.sh <abs path to bin>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "mirage/persistent_kernel/persistent_kernel.cuh"

using namespace mirage::runtime;

// ── model shape ─────────────────────────────────────────────────────────────
// Small enough that the host reference is trivial and FFM finishes quickly,
// but every structural property of the real path is preserved: K is a multiple
// of 128 (linear_kernel_wmma static_asserts K % 32 == 0, and GPT-OSS keeps
// K % 128 == 0), and there is more than one layer so a layer-indexing bug has
// somewhere to show up.
static constexpr int HIDDEN = 128;
static constexpr int VOCAB = 64;
static constexpr int NUM_LAYERS = 2;
static constexpr int BATCH = 1;

// ── device-side state the tasks and the host share ──────────────────────────
__device__ int g_task_runs[16];   // per-task execution counter
__device__ unsigned long long *g_progress; // host-mapped liveness counter

// ── the payload kernels ─────────────────────────────────────────────────────
//
// These are written here rather than reused from tasks/ for one reason: the
// point of this harness is the *graph*, and using tiny hand-written bodies
// keeps the host reference obviously correct. The ported task kernels each have
// their own dedicated test; duplicating that coverage here would only make a
// failure harder to localise.
//
// They are still real device code running inside the megakernel worker loop,
// reached through the same _execute_task dispatch the generated code uses.

// out[h] = embedding[token * HIDDEN + h]
__device__ __forceinline__ void e2e_embedding(float const *emb,
                                              long long token,
                                              float *out) {
  for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) {
    out[i] = emb[token * HIDDEN + i];
  }
}

// RMS norm with a per-channel weight. Reduction goes through mirage::arch so it
// is correct at both wave widths -- the same reason kv_cache_update needed a
// port rather than a shared include.
__device__ __forceinline__ void e2e_rmsnorm(float const *in,
                                            float const *w,
                                            float *out,
                                            float eps) {
  __shared__ float s_part[mirage::arch::MAX_WAVES_PER_BLOCK];
  __shared__ float s_total;

  float acc = 0.0f;
  for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) {
    acc += in[i] * in[i];
  }
  acc = mirage::arch::wave_reduce_sum(acc);
  int const wave = mirage::arch::wave_of(threadIdx.x);
  int const lane = mirage::arch::lane_of(threadIdx.x);
  int const nwaves = mirage::arch::waves_in((int)blockDim.x);
  if (lane == 0) {
    s_part[wave] = acc;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.0f;
    for (int i = 0; i < nwaves; ++i) {
      t += s_part[i];
    }
    s_total = t;
  }
  __syncthreads();

  float const scale = rsqrtf(s_total / (float)HIDDEN + eps);
  for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) {
    out[i] = in[i] * scale * w[i];
  }
}

// out[n] = sum_k in[k] * w[n * HIDDEN + k]   (row-major, N-major weight)
__device__ __forceinline__ void
    e2e_linear(float const *in, float const *w, float *out, int n_out) {
  for (int n = threadIdx.x; n < n_out; n += blockDim.x) {
    float acc = 0.0f;
    for (int k = 0; k < HIDDEN; ++k) {
      acc += in[k] * w[(size_t)n * HIDDEN + k];
    }
    out[n] = acc;
  }
}

// Single-block argmax over VOCAB logits. Deliberately uses the shared
// lane_of/wave_of helpers, so if the wave-width fix in utils.cuh/worker_config.h
// were reverted this would pick the wrong token.
__device__ __forceinline__ void e2e_argmax(float const *logits,
                                           long long *out_token) {
  __shared__ float s_val[mirage::arch::MAX_WAVES_PER_BLOCK];
  __shared__ int s_idx[mirage::arch::MAX_WAVES_PER_BLOCK];

  float best = -1e30f;
  int besti = -1;
  for (int i = threadIdx.x; i < VOCAB; i += blockDim.x) {
    if (logits[i] > best) {
      best = logits[i];
      besti = i;
    }
  }
  int const wave = mirage::arch::wave_of(threadIdx.x);
  int const lane = mirage::arch::lane_of(threadIdx.x);
  int const nwaves = mirage::arch::waves_in((int)blockDim.x);
  // Butterfly over the wave, carrying the index alongside the value.
  for (int off = mirage::arch::WAVE_SIZE / 2; off > 0; off >>= 1) {
    float ov = __shfl_xor(best, off, mirage::arch::WAVE_SIZE);
    int oi = __shfl_xor(besti, off, mirage::arch::WAVE_SIZE);
    // Tie-break on the lower index so the result is deterministic regardless of
    // lane assignment -- otherwise a wave-width change could flip the answer on
    // a tie and this test would be flaky rather than wrong.
    if (ov > best || (ov == best && oi >= 0 && (besti < 0 || oi < besti))) {
      best = ov;
      besti = oi;
    }
  }
  if (lane == 0) {
    s_val[wave] = best;
    s_idx[wave] = besti;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float bv = -1e30f;
    int bi = -1;
    for (int i = 0; i < nwaves; ++i) {
      if (s_val[i] > bv || (s_val[i] == bv && s_idx[i] >= 0 &&
                            (bi < 0 || s_idx[i] < bi))) {
        bv = s_val[i];
        bi = s_idx[i];
      }
    }
    *out_token = (long long)bi;
  }
  __syncthreads();
}

// ── task type encoding ──────────────────────────────────────────────────────
//
// The graph reuses existing TaskType values rather than adding new enum
// entries: adding to the enum would mean touching runtime_header.h, which is on
// the live gfx950 path, for a test-only concern. variant_id carries which of
// this harness's payloads to run. That is exactly what variant_id is for in the
// generated code (QKV vs gate_up on the same task type).
enum E2EVariant {
  V_EMBED = 0,
  V_RMSNORM = 1,
  V_LINEAR = 2,
  V_ARGMAX = 3,
};

__device__ __forceinline__ void
    _execute_task(TaskDesc const *task_desc, RuntimeConfig const &runtime_config) {
  if (threadIdx.x == 0 && g_progress != nullptr) {
    __hip_atomic_fetch_add(
        &g_progress[0], 1ull, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
  if (task_desc->task_type != TASK_IDENTITY) {
    return;
  }

  unsigned const v = task_desc->variant_id;
  // Pointer slots are assigned by the host below. Reading a slot the host did
  // not set would be a null deref rather than a silent wrong answer, which is
  // the failure mode we want.
  void const *in0 = task_desc->input_ptrs[0];
  void const *in1 = task_desc->input_ptrs[1];
  void *out0 = task_desc->output_ptrs[0];

  switch (v) {
    case V_EMBED:
      e2e_embedding(static_cast<float const *>(in0),
                    runtime_config.tokens[0],
                    static_cast<float *>(out0));
      break;
    case V_RMSNORM:
      e2e_rmsnorm(static_cast<float const *>(in0),
                  static_cast<float const *>(in1),
                  static_cast<float *>(out0),
                  1e-6f);
      break;
    case V_LINEAR:
      e2e_linear(static_cast<float const *>(in0),
                 static_cast<float const *>(in1),
                 static_cast<float *>(out0),
                 // The last linear projects to VOCAB, the others to HIDDEN.
                 // Encoded in the output tensor's dim rather than a second
                 // variant, so the task body reads the shape from the descriptor
                 // the way real tasks do.
                 task_desc->task_metadata.n_tile_count);
      break;
    case V_ARGMAX:
      e2e_argmax(static_cast<float const *>(in0),
                 static_cast<long long *>(out0));
      break;
    default:
      break;
  }

  if (threadIdx.x == 0) {
    __hip_atomic_fetch_add(
        &g_task_runs[v], 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
}

__device__ __forceinline__ void
    _execute_gang_task(TaskDesc const *task_desc,
                       RuntimeConfig const &runtime_config,
                       int tile_idx) {
  (void)task_desc;
  (void)runtime_config;
  (void)tile_idx;
}

// ── host-side buffers, filled before init and read after the run ────────────
struct Buffers {
  float *emb;                       // [VOCAB, HIDDEN]
  float *norm_w[NUM_LAYERS];        // [HIDDEN]
  float *lin_w[NUM_LAYERS];         // [HIDDEN or VOCAB, HIDDEN]
  float *act[NUM_LAYERS + 2];       // intermediate activations
  long long *token_out;
};
static Buffers g_buf;

// Exactly representable values: multiples of 1/8 in [-1, 1). Every product and
// partial sum below stays exact in f32, so the host reference matches to the
// bit and the tolerance can be zero.
static float exact_val(unsigned seed) {
  unsigned x = seed;
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return ((float)(int)(x % 9) - 4.0f) * 0.125f;
}

// The task graph. Hand-written stand-in for construct_task_graph().
//
// Shape mirrors the generated graph: task 0 is TERMINATE, task 1 is
// BEGIN_TASK_GRAPH, event 0 is TERMINATION, the last event is
// END_OF_TASK_GRAPH, and first_tasks names the first real task.
static void _init_persistent_kernel(std::vector<FullTaskDesc> &all_tasks,
                                    std::vector<EventDesc> &all_events,
                                    std::vector<TaskId> &first_tasks,
                                    int num_gpus,
                                    int my_gpu_id) {
  (void)num_gpus;
  auto ev = [&](size_t pos) -> EventId {
    return (static_cast<EventId>(my_gpu_id) << 32) | pos;
  };

  auto tensor = [](void *p, int d0, int d1) {
    TensorDesc t{};
    t.num_dims = 2;
    t.base_ptr = p;
    t.data_type = 0;
    t.dim[0] = d0;
    t.dim[1] = d1;
    t.stride[0] = d1;
    t.stride[1] = 1;
    return t;
  };

  // Event structure copies register_mugraph()'s, which is not the obvious one.
  //
  // The intuitive encoding -- a chain of LAUNCH_DEPENDENT_TASKS events, each
  // launching the next task -- does not work, and fails in a way worth
  // recording: the runtime dispatches the first two tasks and then stalls
  // forever with no diagnostic. LAUNCH_DEPENDENT_TASKS is a *per-iteration*
  // event (persistent_kernel.cuh:3347 increments iteration_num on it, and the
  // worker's dependency check at :1612 multiplies num_triggers by the task's
  // iteration number), so using one per edge desynchronises the counters. The
  // dispatch scan at :4497 also takes only the FIRST such event as the main
  // task range, so the later ones define no range at all.
  //
  // What the generated graph actually does (runtime.cc:632-649): ONE
  // LAUNCH_DEPENDENT_TASKS covering [2, n_tasks) prelaunches every task at the
  // start of the iteration, and each real dependency edge is an EVENT_EMPTY
  // that the consumer names in `dependent_event`. Workers then block on the
  // edge's counter before executing. So the ordering is enforced worker-side,
  // not dispatch-side -- which is exactly what makes the megakernel work with
  // fewer workers than tasks.
  //
  //   t0 TERMINATE      t1 BEGIN -> ev1 prelaunches t2..t7
  //   t2 embed          trig ev2   dep --
  //   t3 L0 rmsnorm     trig ev3   dep ev2
  //   t4 L0 linear      trig ev4   dep ev3
  //   t5 L1 rmsnorm     trig ev5   dep ev4
  //   t6 L1 linear      trig ev6   dep ev5
  //   t7 argmax         trig ev7   dep ev6   (ev7 = END_OF_TASK_GRAPH)
  size_t const kNumTasks = 8;
  all_events.push_back(EventDesc(EVENT_TERMINATION, 1, 0, 0)); // ev0
  all_events.push_back(
      EventDesc(EVENT_LAUNCH_DEPENDENT_TASKS, 1, 2, (TaskId)kNumTasks)); // ev1
  for (int i = 2; i <= 6; ++i) {
    // Pure dependency edges. first/last are 0 because an EMPTY event launches
    // nothing -- it only counts.
    all_events.push_back(EventDesc(EVENT_EMPTY, 1, 0, 0));
  }
  all_events.push_back(EventDesc(EVENT_END_OF_TASK_GRAPH, 1, 0, 0)); // ev7

  all_tasks.push_back(FullTaskDesc(TASK_TERMINATE, 0)); // t0

  FullTaskDesc begin(TASK_BEGIN_TASK_GRAPH, 0); // t1
  begin.trigger_event = ev(1);
  all_tasks.push_back(begin);

  // t2: embedding -> act[0]
  {
    FullTaskDesc t(TASK_IDENTITY, V_EMBED);
    t.inputs[0] = tensor(g_buf.emb, VOCAB, HIDDEN);
    t.num_inputs = 1;
    t.outputs[0] = tensor(g_buf.act[0], BATCH, HIDDEN);
    t.num_outputs = 1;
    t.trigger_event = ev(2);
    // No dependent_event: ev1's [first,last) range is what makes the first task
    // runnable. Naming ev1 here instead deadlocks -- the task would wait on an
    // event whose only trigger is the task itself. (Verified in
    // test_hello_megakernel: with the edge present the megakernel spins
    // forever; without it, it terminates.)
    all_tasks.push_back(t);
  }

  // Layers. Each is rmsnorm then linear; the last linear widens to VOCAB.
  for (int l = 0; l < NUM_LAYERS; ++l) {
    int const out_dim = (l == NUM_LAYERS - 1) ? VOCAB : HIDDEN;

    FullTaskDesc rn(TASK_IDENTITY, V_RMSNORM);
    rn.inputs[0] = tensor(g_buf.act[l], BATCH, HIDDEN);
    rn.inputs[1] = tensor(g_buf.norm_w[l], 1, HIDDEN);
    rn.num_inputs = 2;
    rn.outputs[0] = tensor(g_buf.act[l + 1], BATCH, HIDDEN);
    rn.num_outputs = 1;
    rn.trigger_event = ev(3 + 2 * l);
    // Waits on whatever produced act[l]: the embedding for layer 0, the
    // previous layer's linear otherwise.
    rn.dependent_event = ev(2 + 2 * l);
    all_tasks.push_back(rn);

    FullTaskDesc ln(TASK_IDENTITY, V_LINEAR);
    ln.inputs[0] = tensor(g_buf.act[l + 1], BATCH, HIDDEN);
    ln.inputs[1] = tensor(g_buf.lin_w[l], out_dim, HIDDEN);
    ln.num_inputs = 2;
    // The linear writes into the NEXT layer's input slot; the final one writes
    // the logits buffer, which is act[NUM_LAYERS + 1].
    ln.outputs[0] = tensor(g_buf.act[(l == NUM_LAYERS - 1) ? (NUM_LAYERS + 1)
                                                           : (l + 1)],
                           BATCH, out_dim);
    ln.num_outputs = 1;
    ln.task_metadata.n_tile_count = (uint16_t)out_dim;
    ln.trigger_event = ev(4 + 2 * l);
    ln.dependent_event = ev(3 + 2 * l); // waits on this layer's rmsnorm
    all_tasks.push_back(ln);
  }

  // argmax over the logits
  {
    FullTaskDesc t(TASK_IDENTITY, V_ARGMAX);
    t.inputs[0] = tensor(g_buf.act[NUM_LAYERS + 1], BATCH, VOCAB);
    t.num_inputs = 1;
    t.outputs[0] = tensor(g_buf.token_out, 1, 1);
    t.num_outputs = 1;
    t.trigger_event = ev(7);
    t.dependent_event = ev(6); // waits on the final linear
    all_tasks.push_back(t);
  }

  // The event/task counts the graph was written against. If someone adds a
  // layer without widening the event list, ev1's range and the dependent_event
  // arithmetic silently disagree, and the symptom is a hang rather than a
  // wrong answer -- so check it here where the cause is visible.
  if (all_tasks.size() != kNumTasks || all_events.size() != kNumTasks) {
    fprintf(stderr,
            "FAIL: graph built %zu tasks / %zu events, expected %zu of each\n",
            all_tasks.size(), all_events.size(), kNumTasks);
    abort();
  }

  first_tasks.push_back(2);
}

#define CHECK(cmd)                                                             \
  do {                                                                         \
    hipError_t _e = (cmd);                                                     \
    if (_e != hipSuccess) {                                                    \
      fprintf(stderr, "FAIL %s:%d %s -> %s\n", __FILE__, __LINE__, #cmd,       \
              hipGetErrorString(_e));                                          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

// NaN payload. An untouched buffer is both visibly wrong and cannot compare
// equal to a real result.
static float poison_f() {
  uint32_t bits = 0x7FC00001u;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

int main(int argc, char **argv) {
  int num_workers = (argc > 1) ? atoi(argv[1]) : 8;
  long long const input_token = 7;

  printf("[e2e] hidden=%d vocab=%d layers=%d workers=%d\n", HIDDEN, VOCAB,
         NUM_LAYERS, num_workers);

  // ── allocate and fill ─────────────────────────────────────────────────────
  size_t const emb_n = (size_t)VOCAB * HIDDEN;
  std::vector<float> h_emb(emb_n);
  for (size_t i = 0; i < emb_n; ++i) {
    h_emb[i] = exact_val((unsigned)(i * 7 + 1));
  }
  CHECK(hipMalloc(&g_buf.emb, emb_n * sizeof(float)));
  CHECK(hipMemcpy(g_buf.emb, h_emb.data(), emb_n * sizeof(float),
                  hipMemcpyHostToDevice));

  std::vector<std::vector<float>> h_nw(NUM_LAYERS), h_lw(NUM_LAYERS);
  for (int l = 0; l < NUM_LAYERS; ++l) {
    h_nw[l].resize(HIDDEN);
    for (int i = 0; i < HIDDEN; ++i) {
      // Keep norm weights away from zero so a dropped weight is visible.
      h_nw[l][i] = 0.5f + 0.125f * (float)((i + l) % 4);
    }
    CHECK(hipMalloc(&g_buf.norm_w[l], HIDDEN * sizeof(float)));
    CHECK(hipMemcpy(g_buf.norm_w[l], h_nw[l].data(), HIDDEN * sizeof(float),
                    hipMemcpyHostToDevice));

    int const out_dim = (l == NUM_LAYERS - 1) ? VOCAB : HIDDEN;
    h_lw[l].resize((size_t)out_dim * HIDDEN);
    for (size_t i = 0; i < h_lw[l].size(); ++i) {
      h_lw[l][i] = exact_val((unsigned)(i * 13 + l * 977 + 3));
    }
    CHECK(hipMalloc(&g_buf.lin_w[l], h_lw[l].size() * sizeof(float)));
    CHECK(hipMemcpy(g_buf.lin_w[l], h_lw[l].data(),
                    h_lw[l].size() * sizeof(float), hipMemcpyHostToDevice));
  }

  // Activations, all poisoned: act[0..NUM_LAYERS] are HIDDEN-wide, the last is
  // the VOCAB-wide logits.
  {
    std::vector<float> pois(std::max(HIDDEN, VOCAB), poison_f());
    for (int i = 0; i <= NUM_LAYERS + 1; ++i) {
      int const w = (i == NUM_LAYERS + 1) ? VOCAB : HIDDEN;
      CHECK(hipMalloc(&g_buf.act[i], (size_t)w * sizeof(float)));
      CHECK(hipMemcpy(g_buf.act[i], pois.data(), (size_t)w * sizeof(float),
                      hipMemcpyHostToDevice));
    }
  }
  long long const poison_tok = -12345;
  CHECK(hipMalloc(&g_buf.token_out, sizeof(long long)));
  CHECK(hipMemcpy(g_buf.token_out, &poison_tok, sizeof(long long),
                  hipMemcpyHostToDevice));

  int zeros[16] = {0};
  CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_task_runs), zeros, sizeof(zeros)));

  unsigned long long *prog_host = nullptr;
  CHECK(hipHostMalloc((void **)&prog_host, sizeof(unsigned long long),
                      hipHostMallocMapped | hipHostMallocCoherent));
  *prog_host = 0;
  unsigned long long *prog_dev = nullptr;
  CHECK(hipHostGetDevicePointer((void **)&prog_dev, prog_host, 0));
  CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_progress), &prog_dev,
                          sizeof(unsigned long long *)));

  // ── meta tensors ─────────────────────────────────────────────────────────
  // Same reasoning as test_hello_megakernel: all-zero meta is NOT neutral. It
  // means prompt_length == 0, so prepare_next_batch() admits a request with no
  // tokens, returns false, and the runtime terminates before the graph runs
  // once -- which looks like a pass-shaped exit with every counter at zero.
  std::vector<void *> meta(10, nullptr);
  size_t const kMetaBytes = 64 * 1024;
  for (auto &p : meta) {
    CHECK(hipMalloc(&p, kMetaBytes));
    CHECK(hipMemset(p, 0, kMetaBytes));
  }
  int const one = 1;
  CHECK(hipMemcpy(meta[5], &one, sizeof(int), hipMemcpyHostToDevice));
  int const qo[2] = {0, 1};
  CHECK(hipMemcpy(meta[6], qo, sizeof(qo), hipMemcpyHostToDevice));
  // meta[1] is the token buffer that runtime_config.tokens points at; the
  // embedding task reads tokens[0].
  CHECK(hipMemcpy(meta[1], &input_token, sizeof(long long),
                  hipMemcpyHostToDevice));

  init_persistent_kernel(meta,
                         /*profiler_buffer=*/nullptr,
                         /*my_rank=*/0,
                         num_workers,
                         /*num_local_schedulers=*/1,
                         /*num_remote_schedulers=*/0,
                         /*max_seq_length=*/2,
                         /*total_num_requests=*/1,
                         /*eos_token_id=*/-1);

  printf("[e2e] init done, launching\n");
  fflush(stdout);

  // Watcher started BEFORE the launch: launch_persistent_kernel() ends with its
  // own device synchronise, so starting it after would produce no output in
  // exactly the hung case it exists for.
  std::atomic<bool> stop{false};
  std::thread watcher([&] {
    unsigned long long last = 0;
    while (!stop.load(std::memory_order_relaxed)) {
      for (int s = 0; s < 10 && !stop.load(std::memory_order_relaxed); ++s) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
      if (stop.load(std::memory_order_relaxed)) {
        break;
      }
      unsigned long long now =
          *static_cast<unsigned long long volatile *>(prog_host);
      printf("[watch] dispatches=%llu %s\n", now,
             now == last ? "(no progress)" : "(progressing)");
      fflush(stdout);
      last = now;
    }
  });

  launch_persistent_kernel(/*default_stream=*/0);

  stop.store(true, std::memory_order_relaxed);
  watcher.join();

  hipError_t launch_err = hipGetLastError();
  printf("[e2e] post-launch status: %s\n", hipGetErrorString(launch_err));
  CHECK(hipDeviceSynchronize());

  // ── read back ────────────────────────────────────────────────────────────
  int runs[16];
  CHECK(hipMemcpyFromSymbol(runs, HIP_SYMBOL(g_task_runs), sizeof(runs)));
  std::vector<float> got_logits(VOCAB);
  CHECK(hipMemcpy(got_logits.data(), g_buf.act[NUM_LAYERS + 1],
                  VOCAB * sizeof(float), hipMemcpyDeviceToHost));
  long long got_token = poison_tok;
  CHECK(hipMemcpy(&got_token, g_buf.token_out, sizeof(long long),
                  hipMemcpyDeviceToHost));

  finalize_persistent_kernel();

  printf("[e2e] task runs: embed=%d rmsnorm=%d linear=%d argmax=%d\n",
         runs[V_EMBED], runs[V_RMSNORM], runs[V_LINEAR], runs[V_ARGMAX]);

  int fail = 0;

  // Every task must have run, and the two per-layer ones exactly NUM_LAYERS
  // times. A graph that dropped a task would otherwise show up only as a
  // numerical mismatch, which is much harder to read.
  if (runs[V_EMBED] != 1 || runs[V_ARGMAX] != 1 ||
      runs[V_RMSNORM] != NUM_LAYERS || runs[V_LINEAR] != NUM_LAYERS) {
    printf("FAIL: task execution counts wrong (expected embed=1 rmsnorm=%d "
           "linear=%d argmax=1)\n", NUM_LAYERS, NUM_LAYERS);
    fail = 1;
  }

  // ── host reference, in double, mirroring the chain exactly ───────────────
  std::vector<double> x(HIDDEN);
  for (int i = 0; i < HIDDEN; ++i) {
    x[i] = h_emb[(size_t)input_token * HIDDEN + i];
  }
  for (int l = 0; l < NUM_LAYERS; ++l) {
    double ss = 0.0;
    for (int i = 0; i < HIDDEN; ++i) {
      ss += x[i] * x[i];
    }
    double const scale = 1.0 / std::sqrt(ss / (double)HIDDEN + 1e-6);
    std::vector<double> n(HIDDEN);
    for (int i = 0; i < HIDDEN; ++i) {
      n[i] = x[i] * scale * h_nw[l][i];
    }
    int const out_dim = (l == NUM_LAYERS - 1) ? VOCAB : HIDDEN;
    std::vector<double> y(out_dim);
    for (int o = 0; o < out_dim; ++o) {
      double acc = 0.0;
      for (int k = 0; k < HIDDEN; ++k) {
        acc += n[k] * h_lw[l][(size_t)o * HIDDEN + k];
      }
      y[o] = acc;
    }
    x.assign(y.begin(), y.end());
  }

  // Discriminating power: the reference must not be degenerate, or the test
  // passes for the wrong reason. Two things matter -- the logits must actually
  // vary, and the argmax must not be a near-tie, or a harmless last-bit
  // difference could flip the token and make this flaky.
  {
    double lo = x[0], hi = x[0], best = x[0], second = -1e300;
    int bi = 0;
    for (int i = 0; i < VOCAB; ++i) {
      lo = std::min(lo, x[i]);
      hi = std::max(hi, x[i]);
      if (x[i] > best) {
        second = best;
        best = x[i];
        bi = i;
      } else if (x[i] > second) {
        second = x[i];
      }
    }
    printf("[e2e] reference logits: min %.6f max %.6f, argmax %d, margin "
           "%.6g\n", lo, hi, bi, best - second);
    if (hi - lo < 1e-3) {
      printf("FAIL: reference logits are nearly constant -- this data cannot "
             "distinguish a correct argmax from a broken one\n");
      fail = 1;
    }
    if (best - second < 1e-4) {
      printf("FAIL: top-2 logits are within %.6g -- the expected token is not "
             "robust to last-bit differences, so this test would be flaky\n",
             best - second);
      fail = 1;
    }
  }

  // Logits tolerance.
  //
  // The powers-of-two trick makes the embedding and both matmuls exact -- every
  // product and partial sum is representable -- but it does NOT make the whole
  // chain exact, because RMS norm multiplies by rsqrtf(mean + eps), which is
  // irrational. From there on every value is a rounded f32, and the second
  // matmul accumulates those roundings over 128 terms. So a zero tolerance is
  // simply the wrong assertion here; measured disagreement is 1.5e-6 absolute,
  // about 1-2 ulp at this magnitude, and demanding bit-exactness would be a
  // test that can only fail.
  //
  // The bound is scaled to the data rather than picked to fit the observed
  // number: 8 * sqrt(HIDDEN) * 2^-24 is the usual f32 dot-product error growth
  // with headroom, times the largest logit magnitude to convert it to absolute
  // terms. That lands ~30x above the measured error and still ~4 orders of
  // magnitude below the argmax margin -- so it accepts rounding while a real
  // structural bug (a dropped wave in the reduction, a stale buffer) is a
  // fractional error and cannot slip through.
  //
  // isfinite is checked FIRST and separately. A NaN passes every `d > tol`
  // comparison, because comparisons with NaN are false, so a kernel emitting
  // NaN everywhere would otherwise report a worst error of 0.0 and pass. That
  // is not hypothetical -- it is how a wave64 mutant survived
  // tests/mi450/test_kv_cache_update.hip until this check was added.
  int expect_token = 0;
  double max_mag = 0.0;
  {
    double best = -1e300;
    for (int i = 0; i < VOCAB; ++i) {
      max_mag = std::max(max_mag, std::fabs(x[i]));
      if (x[i] > best) {
        best = x[i];
        expect_token = i;
      }
    }
  }
  double const tol = 8.0 * std::sqrt((double)HIDDEN) * std::pow(2.0, -24) *
                     std::max(max_mag, 1.0);

  int mismatches = 0;
  double worst = 0.0;
  for (int i = 0; i < VOCAB; ++i) {
    if (!std::isfinite(got_logits[i])) {
      printf("FAIL: logit %d is not finite (%f) -- note this cannot be caught "
             "by a tolerance check\n", i, got_logits[i]);
      fail = 1;
      break;
    }
    double d = std::fabs((double)got_logits[i] - x[i]);
    worst = std::max(worst, d);
    if (d > tol && ++mismatches <= 5) {
      printf("FAIL: logit %d got %.9g want %.9g (diff %.3g > tol %.3g)\n", i,
             (double)got_logits[i], x[i], d, tol);
    }
  }
  if (mismatches > 0) {
    printf("FAIL: %d/%d logits exceed tolerance %.3g (worst %.3g)\n",
           mismatches, VOCAB, tol, worst);
    fail = 1;
  }

  if (got_token == poison_tok) {
    printf("FAIL: output token was never written (still poison)\n");
    fail = 1;
  } else if (got_token != expect_token) {
    printf("FAIL: emitted token %lld, expected %d\n", got_token, expect_token);
    fail = 1;
  }

  printf("\n[e2e] emitted token %lld (expected %d), worst logit diff %.3g\n",
         got_token, expect_token, worst);
  printf("OVERALL: %s\n", fail ? "FAIL" : "PASS");
  if (!fail) {
    printf("NOTE: this shows the task graph is wired correctly and the values "
           "survive the megakernel dispatch path under FFM-Lite. FFM is "
           "functional only and time-slices workgroups, so it does NOT show "
           "the memory ordering is correct on silicon -- serial execution "
           "hides missing s_wait_*, the bug class this port introduces most.\n");
  }
  return fail;
}
