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
#define ACK_STRIDE 32              /* 128 B apart: one ack per cache line */
#define NBLK 64                    /* plenty for every XCD to get a block */

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
    if (fd >= 0) { printf("  drm: renderD%d matches %s\n", m, bus); return fd; }
  }
  return -1;
}

static void *alloc_in_aid(size_t bytes, int aid) {
  union drm_amdgpu_gem_create req;
  memset(&req, 0, sizeof req);
  req.in.bo_size = bytes;
  req.in.alignment = 2ULL << 20;
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
__device__ __forceinline__ unsigned ld_nt(unsigned const *p) {
  unsigned v; asm volatile("global_load_dword %0, %1, off nt\ns_waitcnt vmcnt(0)"
                           : "=v"(v) : "v"(p) : "memory"); return v;
}
__device__ __forceinline__ void st_sys(unsigned *p, unsigned v) {
  asm volatile("global_store_dword %0, %1, off sc0 sc1" :: "v"(p), "v"(v) : "memory");
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
}
template <int POLL>
__device__ __forceinline__ unsigned poll_ld(unsigned const *p) {
  return POLL ? ld_sys(p) : ld_nt(p);
}

#define BUDGET 300000000ull   /* 3 s in 10 ns ticks */
#define SPIN_GUARD(t0, code)                                                   \
  if ((++spins & 1023u) == 0u &&                                               \
      __builtin_amdgcn_s_memrealtime() - (t0) > BUDGET) {                      \
    atomicOr(err, (code));                                                     \
    return;                                                                    \
  }

// A) ping-pong: exactly one reader XCD, so the writer polls a single ack slot
//    and there is no poll-order bias in the per-XCD numbers.
template <int POLL>
__global__ void pingpong(unsigned *flag, unsigned *acks, int iters, int wxcd,
                         int rxcd, unsigned long long *out, unsigned *claim,
                         unsigned *err, unsigned *cover) {
  int const xcd = my_xcc() & 7;
  if (threadIdx.x != 0) return;
  atomicOr(cover, 1u << xcd);
  if (atomicCAS(&claim[xcd], 0u, 1u) != 0u) return;   /* one rep per XCD */
  bool const writer = (xcd == wxcd);
  bool const reader = (xcd == rxcd);
  if (!writer && !reader) return;
  unsigned long long acc = 0;
  unsigned spins = 0;
  for (int it = 1; it <= iters; ++it) {
    if (writer) {
      unsigned long long t0 = __builtin_amdgcn_s_memrealtime();
      st_sys(flag, (unsigned)it);
      while (poll_ld<POLL>(acks + rxcd * ACK_STRIDE) < (unsigned)it)
        SPIN_GUARD(t0, 1u)
      acc += __builtin_amdgcn_s_memrealtime() - t0;
    } else {
      unsigned long long t0 = __builtin_amdgcn_s_memrealtime();
      while (poll_ld<POLL>(flag) < (unsigned)it)
        SPIN_GUARD(t0, 2u)
      st_sys(acks + rxcd * ACK_STRIDE, (unsigned)it);
    }
  }
  if (writer) *out = acc;
}

// B) fan-out: every XCD waits on one release; time until the LAST ack lands.
//    That is what the layer barrier actually pays.
template <int POLL>
__global__ void fanout(unsigned *flag, unsigned *acks, int iters, int wxcd,
                       unsigned long long *out, unsigned *claim, unsigned *err,
                       unsigned *cover) {
  int const xcd = my_xcc() & 7;
  if (threadIdx.x != 0) return;
  atomicOr(cover, 1u << xcd);
  if (atomicCAS(&claim[xcd], 0u, 1u) != 0u) return;
  bool const writer = (xcd == wxcd);
  unsigned long long acc = 0;
  unsigned spins = 0;
  for (int it = 1; it <= iters; ++it) {
    if (writer) {
      unsigned long long t0 = __builtin_amdgcn_s_memrealtime();
      st_sys(flag, (unsigned)it);
      for (int x = 0; x < 8; ++x) {
        if (x == wxcd) continue;
        while (poll_ld<POLL>(acks + x * ACK_STRIDE) < (unsigned)it)
          SPIN_GUARD(t0, 4u)
      }
      acc += __builtin_amdgcn_s_memrealtime() - t0;
    } else {
      unsigned long long t0 = __builtin_amdgcn_s_memrealtime();
      while (poll_ld<POLL>(flag) < (unsigned)it)
        SPIN_GUARD(t0, 8u)
      st_sys(acks + xcd * ACK_STRIDE, (unsigned)it);
    }
  }
  if (writer) *out = acc;
}

static int g_iters = 500;
static const char *aidof(int x) { return x < 4 ? "AID0" : "AID1"; }
static unsigned *d_claim, *d_err, *d_cover;

static void reset(unsigned *flag, unsigned *acks) {
  hipMemset(flag, 0, 4);
  hipMemset(acks, 0, 8 * ACK_STRIDE * 4);
  hipMemset(d_claim, 0, 8 * 4);
  hipMemset(d_err, 0, 4);
  hipMemset(d_cover, 0, 4);
}
static unsigned rd(unsigned *p) {
  unsigned v = 0; hipMemcpy(&v, p, 4, hipMemcpyDeviceToHost); return v;
}

template <int POLL>
static void arm(const char *tag, unsigned *flag, unsigned *acks, int wxcd,
                bool show_cover) {
  printf("  %-33s writer XCD%d (%s), poll=%s\n", tag, wxcd, aidof(wxcd),
         POLL ? "sc0 sc1" : "nt");
  unsigned long long *d_out;
  hipMalloc(&d_out, 8);
  double same = 0, cross = 0; int ns = 0, nc = 0;
  printf("      round-trip ns: ");
  for (int r = 0; r < 8; ++r) {
    if (r == wxcd) { printf(" x%d:   -", r); continue; }
    reset(flag, acks);
    hipMemset(d_out, 0, 8);
    hipLaunchKernelGGL((pingpong<POLL>), dim3(NBLK), dim3(1), 0, 0,
                       flag, acks, g_iters, wxcd, r, d_out, d_claim, d_err, d_cover);
    if (hipDeviceSynchronize() != hipSuccess) { printf(" x%d: ERR", r); continue; }
    unsigned e = rd(d_err);
    if (e) { printf(" x%d:to%u", r, e); continue; }
    unsigned long long t = 0;
    hipMemcpy(&t, d_out, 8, hipMemcpyDeviceToHost);
    double v = (double)t * 10.0 / g_iters;      /* s_memrealtime tick = 10 ns */
    printf(" x%d:%4.0f", r, v);
    if ((r < 4) == (wxcd < 4)) { same += v; ns++; } else { cross += v; nc++; }
  }
  printf("\n");
  if (show_cover) printf("      XCDs covered by %d blocks: 0x%02x\n", NBLK, rd(d_cover));
  if (ns && nc)
    printf("      same-AID %.0f ns | cross-AID %.0f ns | penalty %+.0f ns\n",
           same / ns, cross / nc, cross / nc - same / ns);

  reset(flag, acks);
  hipMemset(d_out, 0, 8);
  hipLaunchKernelGGL((fanout<POLL>), dim3(NBLK), dim3(1), 0, 0,
                     flag, acks, g_iters, wxcd, d_out, d_claim, d_err, d_cover);
  if (hipDeviceSynchronize() == hipSuccess && !rd(d_err)) {
    unsigned long long t = 0;
    hipMemcpy(&t, d_out, 8, hipMemcpyDeviceToHost);
    printf("      fan-out, last of 7 acks: %.0f ns\n", (double)t * 10.0 / g_iters);
  } else {
    printf("      fan-out: err=%u\n", rd(d_err));
  }
  printf("\n");
  hipFree(d_out);
}

int main(int argc, char **argv) {
  g_iters = argc > 1 ? atoi(argv[1]) : 500;
  int dev = 0;
  hipGetDevice(&dev);
  char bus[64] = {0};
  hipDeviceGetPCIBusId(bus, sizeof bus, dev);
  printf("  hip dev %d = %s, %d iters/arm\n", dev, bus, g_iters);
  g_fd = open_render_node_for_dev(dev);
  if (g_fd < 0) { printf("  no matching drm fd\n"); return 1; }

  hipMalloc(&d_claim, 8 * 4);
  hipMalloc(&d_err, 4);
  hipMalloc(&d_cover, 4);
  unsigned *nc = nullptr, *acks = nullptr;
  hipMalloc(&nc, 2 << 20);                             /* spanning -> MTYPE_NC */
  hipMalloc(&acks, 8 * ACK_STRIDE * 4);
  unsigned *a0 = (unsigned *)alloc_in_aid(2 << 20, 0); /* RW, homed AID0 */
  unsigned *a1 = (unsigned *)alloc_in_aid(2 << 20, 1); /* RW, homed AID1 */
  printf("  flag: NC=%p AID0=%p AID1=%p   (acks in NC)\n\n", nc, a0, a1);

  bool first = true;
  for (int w = 0; w <= 4; w += 4) {
    if (nc) { arm<1>("flag: plain hipMalloc (span/NC)", nc, acks, w, first); first = false; }
    if (a0) arm<1>("flag: AID-local BO in AID0 (RW)", a0, acks, w, false);
    if (a1) arm<1>("flag: AID-local BO in AID1 (RW)", a1, acks, w, false);
  }
  if (nc) arm<0>("flag: plain hipMalloc (span/NC)", nc, acks, 0, false);
  return 0;
}
