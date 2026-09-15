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
#define ACK_STRIDE 32
#define NBLK 512                 /* up to 64 blocks per XCD */

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
    if (fd >= 0) return fd;
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
#define BUDGET 300000000ull
#define SPIN_GUARD(t0, code)                                                   \
  if ((++spins & 1023u) == 0u &&                                               \
      __builtin_amdgcn_s_memrealtime() - (t0) > BUDGET) {                      \
    atomicOr(err, (code)); return;                                             \
  }

// flagA is polled by AID0 XCDs, flagB by AID1 XCDs. Pass the same pointer for
// both to get the single-line configuration.
__global__ void fan(unsigned *flagA, unsigned *flagB, unsigned *acks, int iters,
                    int wxcd, int ppx, unsigned long long *out, unsigned *rank,
                    unsigned *err, unsigned *npoll) {
  int const xcd = my_xcc() & 7;
  if (threadIdx.x != 0) return;
  unsigned const r = atomicAdd(&rank[xcd], 1u);
  bool const writer = (xcd == wxcd && r == 0u);
  if (r >= (unsigned)ppx) return;            /* only ppx pollers per XCD */
  atomicAdd(npoll, 1u);
  unsigned *myflag = (xcd < 4) ? flagA : flagB;
  unsigned long long acc = 0;
  unsigned spins = 0;
  for (int it = 1; it <= iters; ++it) {
    if (writer) {
      unsigned long long t0 = __builtin_amdgcn_s_memrealtime();
      st_sys(flagA, (unsigned)it);
      if (flagB != flagA) st_sys(flagB, (unsigned)it);
      for (int x = 0; x < 8; ++x) {
        if (x == wxcd) continue;
        while (ld_sys(acks + x * ACK_STRIDE) < (unsigned)it) SPIN_GUARD(t0, 1u)
      }
      acc += __builtin_amdgcn_s_memrealtime() - t0;
    } else {
      unsigned long long t0 = __builtin_amdgcn_s_memrealtime();
      while (ld_sys(myflag) < (unsigned)it) SPIN_GUARD(t0, 2u)
      if (r == 0u) st_sys(acks + xcd * ACK_STRIDE, (unsigned)it);
    }
  }
  if (writer) *out = acc;
}

static int g_iters = 500;
static unsigned *d_rank, *d_err, *d_npoll;
static unsigned rd(unsigned *p) { unsigned v = 0; hipMemcpy(&v, p, 4, hipMemcpyDeviceToHost); return v; }

static void run(const char *tag, unsigned *fa, unsigned *fb, unsigned *acks,
                int wxcd) {
  printf("  %-38s writer XCD%d\n", tag, wxcd);
  printf("      %-10s %10s %10s\n", "pollers", "total", "ns");
  unsigned long long *d_out; hipMalloc(&d_out, 8);
  int const sweep[] = {1, 2, 4, 8, 16, 24, 32};
  for (unsigned i = 0; i < sizeof sweep / sizeof *sweep; ++i) {
    int ppx = sweep[i];
    hipMemset(fa, 0, 4); if (fb != fa) hipMemset(fb, 0, 4);
    hipMemset(acks, 0, 8 * ACK_STRIDE * 4);
    hipMemset(d_rank, 0, 8 * 4); hipMemset(d_err, 0, 4); hipMemset(d_npoll, 0, 4);
    hipMemset(d_out, 0, 8);
    hipLaunchKernelGGL(fan, dim3(NBLK), dim3(1), 0, 0, fa, fb, acks, g_iters,
                       wxcd, ppx, d_out, d_rank, d_err, d_npoll);
    if (hipDeviceSynchronize() != hipSuccess) { printf("      %-10d %10s %10s\n", ppx, "-", "ERR"); continue; }
    unsigned e = rd(d_err), n = rd(d_npoll);
    if (e) { printf("      %-10d %10u  timeout=%u\n", ppx, n, e); continue; }
    unsigned long long t = 0; hipMemcpy(&t, d_out, 8, hipMemcpyDeviceToHost);
    printf("      %-10d %10u %10.0f\n", ppx, n, (double)t * 10.0 / g_iters);
  }
  printf("\n");
  hipFree(d_out);
}

int main(int argc, char **argv) {
  g_iters = argc > 1 ? atoi(argv[1]) : 500;
  int dev = 0; hipGetDevice(&dev);
  char bus[64] = {0}; hipDeviceGetPCIBusId(bus, sizeof bus, dev);
  printf("  hip dev %d = %s, %d iters\n\n", dev, bus, g_iters);
  g_fd = open_render_node_for_dev(dev);
  if (g_fd < 0) { printf("  no drm fd\n"); return 1; }
  hipMalloc(&d_rank, 8 * 4); hipMalloc(&d_err, 4); hipMalloc(&d_npoll, 4);
  unsigned *nc = nullptr, *acks = nullptr;
  hipMalloc(&nc, 2 << 20); hipMalloc(&acks, 8 * ACK_STRIDE * 4);
  unsigned *a0 = (unsigned *)alloc_in_aid(2 << 20, 0);
  unsigned *a1 = (unsigned *)alloc_in_aid(2 << 20, 1);

  run("single spanning NC line (pre-split)", nc, nc, acks, 0);
  if (a0 && a1) run("two AID-local RW replicas (split)", a0, a1, acks, 0);
  run("single spanning NC line (pre-split)", nc, nc, acks, 4);
  if (a0 && a1) run("two AID-local RW replicas (split)", a0, a1, acks, 4);
  return 0;
}
