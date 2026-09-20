#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <xf86drm.h>
#include <amdgpu_drm.h>

#define GEM_CREATE_AID_LOCAL  (1ULL << 17)
#define GEM_CREATE_AID_SELECT (1ULL << 18)
#define NBLK 256

static int g_fd = -1;
static int open_render_node_for_dev(int dev) {
  char bus[64] = {0};
  if (hipDeviceGetPCIBusId(bus, sizeof bus, dev) != hipSuccess) return -1;
  for (char *p = bus; *p; ++p) if (*p >= 'A' && *p <= 'F') *p = *p - 'A' + 'a';
  for (int m = 128; m < 256; ++m) {
    char path[128], link[256];
    snprintf(path, sizeof path, "/sys/class/drm/renderD%d/device", m);
    ssize_t n = readlink(path, link, sizeof link - 1);
    if (n <= 0) continue;
    link[n] = 0;
    if (!strstr(link, bus)) continue;
    snprintf(path, sizeof path, "/dev/dri/renderD%d", m);
    int fd = open(path, O_RDWR | O_CLOEXEC);
    if (fd >= 0) { printf("  drm renderD%d = %s\n", m, bus); return fd; }
  }
  return -1;
}
static void *alloc_in_aid(size_t bytes, int aid) {
  union drm_amdgpu_gem_create req;
  memset(&req, 0, sizeof req);
  req.in.bo_size = bytes; req.in.alignment = 2ULL << 20;
  req.in.domains = AMDGPU_GEM_DOMAIN_VRAM;
  req.in.domain_flags = GEM_CREATE_AID_LOCAL |
                        (aid ? GEM_CREATE_AID_SELECT : 0) |
                        AMDGPU_GEM_CREATE_NO_CPU_ACCESS;
  if (ioctl(g_fd, DRM_IOCTL_AMDGPU_GEM_CREATE, &req) != 0) return nullptr;
  struct drm_prime_handle pr; memset(&pr, 0, sizeof pr); pr.handle = req.out.handle;
  if (ioctl(g_fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &pr) != 0) return nullptr;
  hipExternalMemoryHandleDesc hd = {};
  hd.type = hipExternalMemoryHandleTypeOpaqueFd; hd.handle.fd = pr.fd; hd.size = bytes;
  hipExternalMemory_t ext;
  if (hipImportExternalMemory(&ext, &hd) != hipSuccess) return nullptr;
  hipExternalMemoryBufferDesc bd = {}; bd.size = bytes;
  void *p = nullptr;
  if (hipExternalMemoryGetMappedBuffer(&p, ext, &bd) != hipSuccess) return nullptr;
  return p;
}
__device__ __forceinline__ int my_xcc() {
  int x; asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(x)); return x;
}
__device__ __forceinline__ unsigned ld_sys(unsigned const *p) {
  unsigned v; asm volatile("global_load_dword %0, %1, off sc0 sc1\ns_waitcnt vmcnt(0)"
                           : "=v"(v) : "v"(p) : "memory"); return v;
}
__device__ __forceinline__ void st_sys(unsigned *p, unsigned v) {
  asm volatile("global_store_dword %0, %1, off sc0 sc1" :: "v"(p), "v"(v) : "memory");
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
}
typedef float v4f __attribute__((ext_vector_type(4)));

#define BUDGET 200000000ull
#define GUARD(t0, code)                                                        \
  if ((++spins & 1023u) == 0u &&                                               \
      __builtin_amdgcn_s_memrealtime() - (t0) > BUDGET) {                      \
    atomicOr(err, (code)); return;                                             \
  }

__global__ void handoff(v4f *buf, size_t n4, unsigned *flag, unsigned *ack,
                        int iters, int pxcd, unsigned long long *ns,
                        unsigned *claim, unsigned *err, unsigned *stale) {
  int const xcd = my_xcc() & 7;
  if (atomicCAS(&claim[xcd], 0u, 1u) != 0u) return;
  bool const prod = (xcd == pxcd);
  unsigned spins = 0;
  unsigned long long acc = 0;
  for (int it = 1; it <= iters; ++it) {
    if (prod) {
      for (size_t i = threadIdx.x; i < n4; i += blockDim.x) {
        v4f v = {(float)it, (float)it, (float)it, (float)it};
        __builtin_nontemporal_store(v, &buf[i]);
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      __syncthreads();
      if (threadIdx.x == 0) {
        asm volatile("buffer_wbl2 sc1" ::: "memory");
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        st_sys(flag, (unsigned)it);
        unsigned long long t0 = __builtin_amdgcn_s_memrealtime();
        for (int x = 0; x < 8; ++x) {
          if (x == pxcd) continue;
          while (ld_sys(&ack[x * 32]) < (unsigned)it) GUARD(t0, 1u)
        }
      }
      __syncthreads();
    } else {
      unsigned long long t0 = __builtin_amdgcn_s_memrealtime();
      if (threadIdx.x == 0) {
        while (ld_sys(flag) < (unsigned)it) GUARD(t0, 2u)
      }
      __syncthreads();
      unsigned long long t1 = __builtin_amdgcn_s_memrealtime();
      asm volatile("buffer_inv sc1" ::: "memory");
      v4f s = {0.f, 0.f, 0.f, 0.f};
      unsigned bad = 0;
      for (size_t i = threadIdx.x; i < n4; i += blockDim.x) {
        v4f v = __builtin_nontemporal_load(&buf[i]);
        s += v;
        if (v.x != (float)it) bad++;   /* did this consumer see the write? */
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      __syncthreads();
      unsigned long long t2 = __builtin_amdgcn_s_memrealtime();
      if (bad) atomicAdd(&stale[xcd], bad);
      if (threadIdx.x == 0) {
        acc += t2 - t1;
        st_sys(&ack[xcd * 32], (unsigned)it);
      }
      if (s.x == 12345.678f) buf[0] = s;
    }
  }
  if (!prod && threadIdx.x == 0) ns[xcd] = acc;
}

static int g_iters = 100;
static unsigned *d_claim, *d_err, *d_stale;
static unsigned rd(unsigned *p) { unsigned v = 0; hipMemcpy(&v, p, 4, hipMemcpyDeviceToHost); return v; }

static void run(const char *tag, v4f *buf, size_t bytes, int pxcd) {
  if (!buf) { printf("  %-24s alloc failed\n", tag); return; }
  size_t n4 = bytes / sizeof(v4f);
  unsigned *flag, *ack; unsigned long long *ns;
  hipMalloc(&flag, 4); hipMalloc(&ack, 8 * 32 * 4); hipMalloc(&ns, 8 * 8);
  hipMemset(flag, 0, 4); hipMemset(ack, 0, 8 * 32 * 4); hipMemset(ns, 0, 8 * 8);
  hipMemset(d_claim, 0, 8 * 4); hipMemset(d_err, 0, 4); hipMemset(d_stale, 0, 8 * 4);
  hipLaunchKernelGGL(handoff, dim3(NBLK), dim3(256), 0, 0,
                     buf, n4, flag, ack, g_iters, pxcd, ns, d_claim, d_err, d_stale);
  hipError_t e = hipDeviceSynchronize();
  if (e != hipSuccess || rd(d_err)) {
    printf("  %-24s TIMEOUT/err=%u\n", tag, rd(d_err));
    hipFree(flag); hipFree(ack); hipFree(ns); return;
  }
  unsigned long long h[8]; unsigned st[8];
  hipMemcpy(h, ns, sizeof h, hipMemcpyDeviceToHost);
  hipMemcpy(st, d_stale, sizeof st, hipMemcpyDeviceToHost);
  double ok_same = 0, ok_cross = 0; int n_same = 0, n_cross = 0;
  printf("  %-24s", tag);
  for (int x = 0; x < 8; ++x) {
    if (x == pxcd) { printf("  x%d:    -   ", x); continue; }
    double v = (double)h[x] * 10.0 / g_iters;
    printf("  x%d:%6.0f%s", x, v, st[x] ? "!" : " ");
    if (!st[x]) {
      if ((x < 4) == (pxcd < 4)) { ok_same += v; n_same++; }
      else { ok_cross += v; n_cross++; }
    }
  }
  int nstale = 0;
  for (int x = 0; x < 8; ++x) if (st[x]) nstale++;
  if (nstale) printf("   STALE on %d XCDs -> those times are WRONG READS", nstale);
  else if (n_same && n_cross)
    printf("   same %.0f cross %.0f (+%.0f, %+.0f%%)", ok_same / n_same,
           ok_cross / n_cross, ok_cross / n_cross - ok_same / n_same,
           100.0 * (ok_cross / n_cross / (ok_same / n_same) - 1.0));
  printf("\n");
  hipFree(flag); hipFree(ack); hipFree(ns);
}

int main(int argc, char **argv) {
  g_iters = argc > 1 ? atoi(argv[1]) : 100;
  int dev = 0; hipGetDevice(&dev);
  char bus[64] = {0}; hipDeviceGetPCIBusId(bus, sizeof bus, dev);
  printf("  hip dev %d = %s, %d rounds.  '!' = this XCD read STALE data\n", dev, bus, g_iters);
  g_fd = open_render_node_for_dev(dev);
  if (g_fd < 0) { printf("  no drm fd\n"); return 1; }
  hipMalloc(&d_claim, 8 * 4); hipMalloc(&d_err, 4); hipMalloc(&d_stale, 8 * 4);

  size_t const bytes = 4ull << 20;
  void *nc = nullptr; hipMalloc(&nc, bytes);
  void *a0 = alloc_in_aid(bytes, 0);
  void *a1 = alloc_in_aid(bytes, 1);
  printf("\n  ==== %zu KiB handoff, consumer read time ns/round ====\n", bytes >> 10);
  for (int p = 0; p <= 4; p += 4) {
    printf("   -- producer XCD%d (%s) --\n", p, p < 4 ? "AID0" : "AID1");
    run("plain hipMalloc (NC)", (v4f *)nc, bytes, p);
    run("alloc_in_aid(0) (RW)", (v4f *)a0, bytes, p);
    run("alloc_in_aid(1) (RW)", (v4f *)a1, bytes, p);
  }
  return 0;
}
