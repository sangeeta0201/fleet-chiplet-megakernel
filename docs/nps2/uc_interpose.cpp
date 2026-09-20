// LD_PRELOAD shim: route every hipMalloc to an UNCACHED allocation.
//
// Purpose is a correctness experiment, not performance. In SPX+NPS2 a plain
// hipMalloc VRAM buffer spans both memory ranges, is local to neither, and the
// driver demotes it to MTYPE_NC -- cacheable but NOT coherent across XCDs. The
// hypothesis is that this single fact explains both SPX+NPS2 defects: a reader
// that hits a stale L2 line sometimes gets a slightly wrong activation (fleet
// then emits different text from SPX+NPS1) and sometimes misses a flag update
// (fleet hangs).
//
// MTYPE_UC is coherent by construction -- every access goes to memory. So if
// NPS2 under this shim reproduces the NPS1 text and stops hanging, NC
// incoherence is confirmed as the root cause and everything after it is purely
// recovering the speed UC gives up.
//
// UNCACHED is deliberately the only thing changed. The BO keeps its normal
// spanning placement, so this does NOT also move memory between AIDs the way
// an AID_LOCAL interposer would -- one variable, not two. It is also the only
// allocation flag that reaches the driver through hipExtMallocWithFlags, so no
// DRM ioctl or dmabuf import is needed.
//
// Build:
//   hipcc -O2 -fPIC -shared -o libucalloc.so uc_interpose.cpp -ldl
// Use:
//   LD_PRELOAD=/path/libucalloc.so UC_ALLOC=1 python demo.py
//
// Env:
//   UC_ALLOC=1            enable; without it the library is inert, so the same
//                         LD_PRELOAD serves as its own control arm
//   UC_ALLOC_MIN_BYTES=N  allocations below N fall through (default 0)
//   UC_ALLOC_MAX_BYTES=N  allocations above N fall through (default SIZE_MAX)
//   UC_ALLOC_VERBOSE=1    per-allocation log plus a summary at exit
//
// UC_ALLOC_MAX_BYTES exists because making everything uncached is not
// runnable: with the ~60 GiB of MXFP4 weights uncached, model load alone ran
// 15 minutes at 100% GPU without reaching megakernel compilation. It is also
// better targeted -- weights are read-only once loaded, so they cannot be a
// source of incoherence. Only read-write buffers (activations, barriers,
// counters, workspaces, KV) can serve a stale line, and those are all small.
//
// Pair it with PYTORCH_NO_CUDA_MEMORY_CACHING=1. Otherwise torch's caching
// allocator hands out small tensors from inside large segments, so a size
// threshold applied at hipMalloc granularity would miss them.

#define _GNU_SOURCE
#include <dlfcn.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

using malloc_fn = hipError_t (*)(void **, size_t);
using extmalloc_fn = hipError_t (*)(void **, size_t, unsigned int);

malloc_fn real_malloc = nullptr;
extmalloc_fn real_extmalloc = nullptr;

bool g_on = false;
bool g_verbose = false;
size_t g_min_bytes = 0;
size_t g_max_bytes = ~(size_t)0;

// Guards against re-entry: if the HIP runtime implements
// hipExtMallocWithFlags on top of hipMalloc, the interposed symbol would
// otherwise call itself forever.
thread_local bool t_inside = false;

size_t g_uc_count = 0, g_uc_bytes = 0;
size_t g_pass_count = 0, g_pass_bytes = 0;

size_t env_size(char const *name, size_t dflt) {
  char const *e = getenv(name);
  if (e == nullptr || *e == 0) {
    return dflt;
  }
  return strtoull(e, nullptr, 0);
}

void report() {
  fprintf(stderr,
          "[UC] uncached %zu allocs / %.2f GiB, passed through %zu / %.2f "
          "GiB\n",
          g_uc_count,
          g_uc_bytes / 1073741824.0,
          g_pass_count,
          g_pass_bytes / 1073741824.0);
}

void init() {
  real_malloc = (malloc_fn)dlsym(RTLD_NEXT, "hipMalloc");
  real_extmalloc = (extmalloc_fn)dlsym(RTLD_NEXT, "hipExtMallocWithFlags");
  char const *e = getenv("UC_ALLOC");
  g_on = (e != nullptr && e[0] == '1');
  g_verbose = getenv("UC_ALLOC_VERBOSE") != nullptr;
  g_min_bytes = env_size("UC_ALLOC_MIN_BYTES", 0);
  g_max_bytes = env_size("UC_ALLOC_MAX_BYTES", ~(size_t)0);
  if (g_on && real_extmalloc == nullptr) {
    fprintf(stderr,
            "[UC] hipExtMallocWithFlags not found, staying inert\n");
    g_on = false;
  }
  if (g_on) {
    fprintf(stderr,
            "[UC] active, min_bytes=%zu max_bytes=%zu\n",
            g_min_bytes,
            g_max_bytes);
    atexit(report);
  }
}

} // namespace

extern "C" hipError_t hipMalloc(void **ptr, size_t size) {
  static bool once = (init(), true);
  (void)once;

  if (!g_on || t_inside || size < g_min_bytes || size > g_max_bytes) {
    g_pass_count++;
    g_pass_bytes += size;
    return real_malloc(ptr, size);
  }

  t_inside = true;
  hipError_t rc = real_extmalloc(ptr, size, hipDeviceMallocUncached);
  t_inside = false;

  if (rc != hipSuccess) {
    // Not fatal on its own -- fall back rather than fail the run, and say so,
    // because a silent fallback would make the arm look like a clean UC run
    // when it was not.
    fprintf(stderr,
            "[UC] uncached alloc of %zu B failed (%d), falling back to "
            "hipMalloc\n",
            size,
            (int)rc);
    g_pass_count++;
    g_pass_bytes += size;
    return real_malloc(ptr, size);
  }

  g_uc_count++;
  g_uc_bytes += size;
  if (g_verbose) {
    fprintf(stderr, "[UC] %.2f MiB uncached -> %p\n", size / 1048576.0, *ptr);
  }
  return rc;
}
