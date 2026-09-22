// test_hello_megakernel.hip -- Phase 0 of the MI450 (gfx1250) port.
//
// This test deliberately computes nothing. It exists to answer one question
// that has to be settled before any kernel is worth porting:
//
//   does MPK's scheduler/worker handshake make forward progress on gfx1250
//   under FFM-Lite, at the real workgroup count?
//
// MPK is a persistent megakernel: workers spin on per-worker queues, a
// scheduler hands out task ids, and completion is published through event
// counters in device memory. That is a producer/consumer pattern, and FFM-Lite
// by default runs workgroups *serially in ascending blockIdx*. Under that
// policy a worker that spins waiting for a scheduler which has not been given
// a turn yet never yields, and the model wedges. HSA_MODEL_ARGS=
// "ffm_enable_time_slicing" is what makes the pattern runnable, so this test is
// also the check that time slicing actually holds at ~240 workgroups rather
// than only in the small producer/consumer probes used earlier.
//
// The task graph is the smallest legal one:
//
//   task 0 : TASK_TERMINATE        (index 0 by convention; never scheduled)
//   task 1 : TASK_BEGIN_TASK_GRAPH (entry point)
//   task 2 : TASK_IDENTITY         (the single trivial payload task)
//
//   event 0 : EVENT_TERMINATION
//   event 1 : EVENT_LAUNCH_DEPENDENT_TASKS -> [2, 3)   (begin -> payload)
//   event 2 : EVENT_END_OF_TASK_GRAPH
//
// mirroring the shape the real generated graph uses (verified against
// demo/gpt_oss/task_graph.json: task 0 is TERMINATE, event 0 is TERMINATION,
// the last event is END_OF_TASK_GRAPH, and first_tasks names the first real
// task).
//
// WHAT A PASS DOES AND DOES NOT MEAN
//
// Pass means: the graph was consumed, the payload task ran, and the kernel
// terminated instead of livelocking. That is a real result -- livelock was the
// single most likely way this port dies early.
//
// Pass does NOT mean the handshake is correct on silicon. FFM-Lite is
// functional only: no cycles, no bandwidth, no contention, and workgroups are
// time sliced rather than genuinely concurrent. Serial execution *hides*
// missing s_wait_* and memory-ordering races -- exactly the bug class the
// gfx9 -> gfx12 wait-counter split introduces. Time slicing also cuts the other
// way: it can mask a deadlock real hardware would hit. Green here is necessary,
// not sufficient.
//
// Build (against the FFM package's HIP runtime so the model sees dispatches):
//   $TOOLCHAIN/llvm/bin/clang++ -x hip --offload-arch=gfx1250 \
//       --rocm-path=$TOOLCHAIN -std=c++17 \
//       -I../../include -I../../include/mirage/hip_compat \
//       -I../../include/mirage/persistent_kernel \
//       -I../../deps/composable_kernel/include -I../../deps/json/include \
//       -D__HIP_PLATFORM_AMD__=1 -DMIRAGE_AMD_MI300 -DMIRAGE_AMD_MI450 \
//       -DMPK_TARGET_CC=125 -DMODE_OFFLINE ... \
//       test_hello_megakernel.hip -o test_hello_megakernel \
//       -L$FFM_PKG/rocm -lamdhip64 -Wl,-rpath,$FFM_PKG/rocm
// Run:
//   HSA_MODEL_ARGS="ffm_enable_time_slicing" ../../run_ffm.sh <abs path to bin>

#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <vector>
#include <atomic>
#include <chrono>
#include <thread>

#include "mirage/persistent_kernel/persistent_kernel.cuh"

using namespace mirage::runtime;

// ---------------------------------------------------------------------------
// The two functions the MPK JIT normally generates from the task graph. Here
// they are written by hand -- that is the whole point of a torch-free harness.
// ---------------------------------------------------------------------------

// Set by the payload task so the host can prove the task body actually ran and
// did not merely get counted as complete by the scheduler.
__device__ int g_hello_ran;

// Host-mapped progress counter.
//
// Distinguishing "deadlocked" from "running but astronomically slow" matters
// more here than anywhere else in the port, and neither device printf nor
// hipMemcpy can tell them apart: printf output is buffered and only flushes
// when the kernel exits, and hipMemcpyAsync does not complete while a
// persistent kernel occupies the GPU (the runtime's own comments at
// persistent_kernel.cuh:5040 say exactly this). Both go silent in either case.
//
// Host-mapped memory has neither problem: the device writes it directly and
// the host reads the same physical page with a plain load, no HIP call and no
// kernel exit required. If this counter climbs, the megakernel is alive and
// FFM is just slow; if it is pinned at zero, it is wedged.
__device__ unsigned long long *g_progress;

__device__ __forceinline__ void
    _execute_task(TaskDesc const *task_desc,
                  RuntimeConfig const &runtime_config) {
  (void)runtime_config;
  if (threadIdx.x == 0 && g_progress != nullptr) {
    // Every task dispatch, not just the payload: this counts scheduler/worker
    // handshakes, which is the liveness signal we actually want.
    __hip_atomic_fetch_add(
        &g_progress[0], 1ull, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
  if (task_desc->task_type == TASK_IDENTITY) {
    if (threadIdx.x == 0) {
      // Relaxed is deliberate: this is a liveness flag, not a synchronisation
      // edge. Ordering of the task itself is the runtime's job, and using a
      // release here would hide a missing release in the runtime.
      __hip_atomic_fetch_add(
          &g_hello_ran, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
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

// ---------------------------------------------------------------------------
// The task graph. This is the hand-written stand-in for construct_task_graph().
// ---------------------------------------------------------------------------

static void _init_persistent_kernel(std::vector<FullTaskDesc> &all_tasks,
                                    std::vector<EventDesc> &all_events,
                                    std::vector<TaskId> &first_tasks,
                                    int num_gpus,
                                    int my_gpu_id) {
  (void)num_gpus;

  // Event ids carry the owning gpu in the high 32 bits.
  auto ev = [&](size_t pos) -> EventId {
    return (static_cast<EventId>(my_gpu_id) << 32) | pos;
  };

  // --- events -------------------------------------------------------------
  // 0: termination. 1: begin -> payload. 2: end of graph.
  all_events.push_back(EventDesc(EVENT_TERMINATION, 1, 0, 0));
  all_events.push_back(EventDesc(EVENT_LAUNCH_DEPENDENT_TASKS, 1, 2, 3));
  all_events.push_back(EventDesc(EVENT_END_OF_TASK_GRAPH, 1, 0, 0));

  // --- tasks --------------------------------------------------------------
  // Task 0 is TASK_TERMINATE by convention; it is never scheduled, it is the
  // sentinel the runtime jumps to. Its events stay invalid.
  all_tasks.push_back(FullTaskDesc(TASK_TERMINATE, 0));

  // Task 1 kicks the graph off and triggers event 1.
  FullTaskDesc begin(TASK_BEGIN_TASK_GRAPH, 0);
  begin.trigger_event = ev(1);
  all_tasks.push_back(begin);

  // Task 2 is the payload. Note it does NOT set dependent_event: in the real
  // generated graph the first task of the main range has dependent_event =
  // EVENT_INVALID_ID, because event 1's [first,last) *range* is what makes it
  // runnable, not an incoming dependency edge. Setting dependent_event = 1 here
  // instead deadlocks -- the task waits on an event whose only trigger is the
  // task itself. (Verified: with the edge present the megakernel spins forever;
  // without it, it terminates.)
  FullTaskDesc hello(TASK_IDENTITY, 0);
  hello.trigger_event = ev(2);
  all_tasks.push_back(hello);

  first_tasks.push_back(2);
}

// ---------------------------------------------------------------------------

#define CHECK(cmd)                                                             \
  do {                                                                         \
    hipError_t _e = (cmd);                                                     \
    if (_e != hipSuccess) {                                                    \
      fprintf(stderr,                                                          \
              "FAIL %s:%d %s -> %s\n",                                         \
              __FILE__,                                                        \
              __LINE__,                                                        \
              #cmd,                                                            \
              hipGetErrorString(_e));                                          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main(int argc, char **argv) {
  // Default to the real shape (240 workers) so this exercises the scale that
  // matters; allow a smaller count for bisecting a hang.
  int num_workers = (argc > 1) ? atoi(argv[1]) : 240;
  int num_schedulers = (argc > 2) ? atoi(argv[2]) : 1;

  printf("[hello_mpk] workers=%d local_schedulers=%d\n",
         num_workers,
         num_schedulers);

  // The 10 meta tensors init_persistent_kernel() asserts on. TASK_IDENTITY
  // reads none of them, but the *scheduler* does: MODE_OFFLINE's
  // prepare_next_batch() decides whether to run another iteration by building
  // a batch out of these, and returns false when it comes up with zero tokens.
  //
  // All-zero meta is therefore not a neutral choice -- it means prompt_length
  // is 0, so the very first prepare_next_batch() admits a request with no
  // tokens, returns false, and the runtime terminates *before* the task graph
  // has run even once. That looks like a pass-shaped exit with ran == 0.
  //
  // So give it exactly enough to run one iteration and then stop:
  //   prompt_length[0]   = 1   -> first call yields 1 token, returns true
  //   max_seq_length     = 2   -> second call retires the request, and with
  //                              next_request_id == total_num_requests == 1
  //                              there is nothing to admit, so it returns false
  // One iteration is all this test wants; letting it run to a 4096-token
  // max_seq_length would multiply FFM's simulated spin cost by 4096 for no
  // extra signal.
  std::vector<void *> meta(10, nullptr);
  size_t const kMetaBytes = 64 * 1024;
  for (auto &p : meta) {
    CHECK(hipMalloc(&p, kMetaBytes));
    CHECK(hipMemset(p, 0, kMetaBytes));
  }
  // meta[5] is prompt_length, indexed by request_id.
  int const one = 1;
  CHECK(hipMemcpy(meta[5], &one, sizeof(int), hipMemcpyHostToDevice));
  // meta[6] is qo_indptr_buffer: request 0 owns [0, 1).
  int const qo[2] = {0, 1};
  CHECK(hipMemcpy(meta[6], qo, sizeof(qo), hipMemcpyHostToDevice));

  int zero = 0;
  CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_hello_ran), &zero, sizeof(int)));

  // Host-mapped progress counter (see the g_progress comment above).
  unsigned long long *prog_host = nullptr;
  CHECK(hipHostMalloc((void **)&prog_host,
                      sizeof(unsigned long long),
                      hipHostMallocMapped | hipHostMallocCoherent));
  *prog_host = 0;
  unsigned long long *prog_dev = nullptr;
  CHECK(hipHostGetDevicePointer((void **)&prog_dev, prog_host, 0));
  CHECK(hipMemcpyToSymbol(
      HIP_SYMBOL(g_progress), &prog_dev, sizeof(unsigned long long *)));

#ifdef MPK_HOST_BREADCRUMB
  // Same mechanism, but for the runtime's own role election, so a stall can be
  // localised to before election / worker entry / scheduler entry.
  unsigned long long *bc_host = nullptr;
  CHECK(hipHostMalloc((void **)&bc_host,
                      16 * sizeof(unsigned long long),
                      hipHostMallocMapped | hipHostMallocCoherent));
  for (int i = 0; i < 16; ++i) {
    bc_host[i] = 0;
  }
  unsigned long long *bc_dev = nullptr;
  CHECK(hipHostGetDevicePointer((void **)&bc_dev, bc_host, 0));
  CHECK(hipMemcpyToSymbol(
      HIP_SYMBOL(g_mpk_breadcrumb), &bc_dev, sizeof(unsigned long long *)));
#endif

  init_persistent_kernel(meta,
                         /*profiler_buffer=*/nullptr,
                         /*my_rank=*/0,
                         num_workers,
                         /*num_local_schedulers=*/num_schedulers,
                         /*num_remote_schedulers=*/0,
                         /*max_seq_length=*/2,
                         /*total_num_requests=*/1,
                         /*eos_token_id=*/-1);

  printf("[hello_mpk] init done, launching\n");
  fflush(stdout);

  // The watcher must start BEFORE the launch: launch_persistent_kernel() ends
  // with its own cudaDeviceSynchronize() in the non-split path, so it does not
  // return until the megakernel has finished. Starting the watcher after it
  // would produce no output at all in exactly the hung case it exists for.
  //
  // It reads a plain volatile load of host-mapped memory -- deliberately no HIP
  // call, which would queue behind the persistent kernel and never complete.
  std::atomic<bool> watch_stop{false};
  std::thread watcher([&] {
    unsigned long long last = 0;
    for (int t = 0; !watch_stop.load(std::memory_order_relaxed); ++t) {
      for (int s = 0; s < 10 && !watch_stop.load(std::memory_order_relaxed);
           ++s) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
      if (watch_stop.load(std::memory_order_relaxed)) {
        break;
      }
      unsigned long long now =
          *static_cast<unsigned long long volatile *>(prog_host);
#ifdef MPK_HOST_BREADCRUMB
      auto bc = [&](int i) {
        return *(static_cast<unsigned long long volatile *>(bc_host) + i);
      };
      // Slot map matches the MPK_CRUMB call sites in persistent_kernel.cuh.
      printf("[watch] t=%3ds disp=%llu | elect=%llu w_branch=%llu "
             "s_branch=%llu | w_xcdpub=%llu s_spin_in=%llu s_spin_out=%llu | "
             "w_loop=%llu w_iter=%llu s_loop=%llu s_iter=%llu %s\n",
             (t + 1) * 10,
             now,
             bc(0),
             bc(1),
             bc(2),
             bc(7),
             bc(8),
             bc(9),
             bc(3),
             bc(4),
             bc(5),
             bc(6),
             now == last ? "(no dispatch progress)" : "(progressing)");
#else
      printf("[watch] t=%3ds dispatches=%llu %s\n",
             (t + 1) * 10,
             now,
             now == last ? "(no progress)" : "(progressing)");
#endif
      fflush(stdout);
      last = now;
    }
  });

  launch_persistent_kernel(/*default_stream=*/0);

  watch_stop.store(true, std::memory_order_relaxed);
  watcher.join();

  // launch_persistent_kernel discards every HIP status with (void), so a
  // rejected launch is otherwise indistinguishable from a hang.
  hipError_t launch_err = hipGetLastError();
  printf("[hello_mpk] post-launch status: %s\n", hipGetErrorString(launch_err));
  fflush(stdout);
  CHECK(hipDeviceSynchronize());

  int ran = -1;
  CHECK(hipMemcpyFromSymbol(&ran, HIP_SYMBOL(g_hello_ran), sizeof(int)));

  printf("[hello_mpk] kernel returned; payload task ran %d time(s)\n", ran);

  finalize_persistent_kernel();

  if (ran < 1) {
    printf("FAIL: megakernel completed but the payload task never executed "
           "(ran=%d). The graph was consumed without running its body.\n",
           ran);
    return 1;
  }
  printf("PASS: megakernel reached termination and the payload task ran.\n");
  printf("NOTE: FFM-Lite is functional only. This shows liveness, not that "
         "the memory ordering is correct on silicon.\n");
  return 0;
}
