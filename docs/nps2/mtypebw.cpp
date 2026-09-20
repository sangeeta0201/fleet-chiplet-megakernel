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

// __builtin_nontemporal_load rejects HIP's float4 wrapper class, so use a
// native ext_vector for the 16 B loads.
typedef float v4f __attribute__((ext_vector_type(4)));

// Stream the buffer with wide loads, like an MXFP4 weight pass. Each block takes
// a distinct contiguous slab so nothing is served from another block's L2 lines.
__global__ void stream(const v4f *__restrict p, size_t n4, float *sink,
                       unsigned long long *cyc, unsigned *xcd_of_block) {
  size_t const nb = gridDim.x;
  size_t const per = n4 / nb;
  const v4f *base = p + blockIdx.x * per;
  if (threadIdx.x == 0) {
    xcd_of_block[blockIdx.x] = (unsigned)(my_xcc() & 7);
  }
  __syncthreads();
  unsigned long long t0 = __builtin_amdgcn_s_memrealtime();
  v4f acc = {0.f, 0.f, 0.f, 0.f};
  for (size_t i = threadIdx.x; i < per; i += blockDim.x) {
    acc += __builtin_nontemporal_load(&base[i]);
  }
  unsigned long long t1 = __builtin_amdgcn_s_memrealtime();
  if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
  if (acc.x == 12345.678f) sink[0] = acc.x + acc.y + acc.z + acc.w;  // keep live
}
#define V4 v4f

static size_t g_bytes = 512ull << 20;
static int const NBLK = 512, NTHR = 256;

static void run(const char *tag, void *buf) {
  if (!buf) { printf("  %-28s alloc failed\n", tag); return; }
  size_t n4 = g_bytes / sizeof(v4f);
  float *sink; unsigned long long *cyc; unsigned *xcdb;
  hipMalloc(&sink, 4);
  hipMalloc(&cyc, NBLK * sizeof(unsigned long long));
  hipMalloc(&xcdb, NBLK * sizeof(unsigned));
  // warm
  hipLaunchKernelGGL(stream, dim3(NBLK), dim3(NTHR), 0, 0,
                     (const v4f *)buf, n4, sink, cyc, xcdb);
  hipDeviceSynchronize();

  double best = 0;
  double half_ns[2] = {0, 0}; int half_n[2] = {0, 0};
  for (int rep = 0; rep < 3; ++rep) {
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    hipEventRecord(a);
    hipLaunchKernelGGL(stream, dim3(NBLK), dim3(NTHR), 0, 0,
                       (const v4f *)buf, n4, sink, cyc, xcdb);
    hipEventRecord(b);
    if (hipDeviceSynchronize() != hipSuccess) { printf("  %-28s kernel err\n", tag); return; }
    float ms = 0; hipEventElapsedTime(&ms, a, b);
    double gbs = (double)g_bytes / (ms * 1e-3) / 1e9;
    if (gbs > best) best = gbs;
    if (rep == 2) {
      unsigned long long h_cyc[NBLK]; unsigned h_x[NBLK];
      hipMemcpy(h_cyc, cyc, sizeof h_cyc, hipMemcpyDeviceToHost);
      hipMemcpy(h_x, xcdb, sizeof h_x, hipMemcpyDeviceToHost);
      for (int i = 0; i < NBLK; ++i) {
        int h = (h_x[i] < 4) ? 0 : 1;
        half_ns[h] += (double)h_cyc[i] * 10.0; half_n[h]++;
      }
    }
    hipEventDestroy(a); hipEventDestroy(b);
  }
  printf("  %-28s %8.1f GB/s   per-block us: XCD0-3 %7.1f (n=%d) | XCD4-7 %7.1f (n=%d)\n",
         tag, best,
         half_n[0] ? half_ns[0] / half_n[0] / 1000.0 : 0.0, half_n[0],
         half_n[1] ? half_ns[1] / half_n[1] / 1000.0 : 0.0, half_n[1]);
  hipFree(sink); hipFree(cyc); hipFree(xcdb);
}

int main(int argc, char **argv) {
  if (argc > 1) g_bytes = (size_t)atoll(argv[1]) << 20;
  int dev = 0; hipGetDevice(&dev);
  char bus[64] = {0}; hipDeviceGetPCIBusId(bus, sizeof bus, dev);
  printf("  hip dev %d = %s, %zu MiB per buffer, %d blocks x %d thr\n",
         dev, bus, g_bytes >> 20, NBLK, NTHR);
  g_fd = open_render_node_for_dev(dev);
  if (g_fd < 0) { printf("  no drm fd\n"); return 1; }

  void *nc = nullptr;
  hipMalloc(&nc, g_bytes);                       /* MTYPE_NC in SPX+NPS2 */
  void *a0 = alloc_in_aid(g_bytes, 0);           /* range 0, MTYPE_RW via PERBO */
  void *a1 = alloc_in_aid(g_bytes, 1);           /* range 1, MTYPE_RW via PERBO */
  hipMemset(nc, 1, g_bytes);
  if (a0) hipMemset(a0, 1, g_bytes);
  if (a1) hipMemset(a1, 1, g_bytes);
  hipDeviceSynchronize();
  printf("  nc=%p aid0=%p aid1=%p\n\n", nc, a0, a1);

  run("plain hipMalloc (NC)", nc);
  run("alloc_in_aid(0)  (RW)", a0);
  run("alloc_in_aid(1)  (RW)", a1);
  return 0;
}
