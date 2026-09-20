/* AID-local and 7-stack-striped VRAM BOs, imported as HIP device pointers.
 * Requires the patched amdgpu (AID_LOCAL / AID_STRIPE GEM flags) and SPX+NPS2.
 *
 *   gcc -shared -fPIC -O2 -I/opt/rocm/include -o libaid_hbm.so aid_hbm.c \
 *       -L/opt/rocm/lib -lamdhip64 -ldrm
 */
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include <hip/hip_runtime_api.h>
#include <libdrm/amdgpu_drm.h>
#include <libdrm/drm.h>
#include <sys/ioctl.h>

#define AID_LOCAL     (1ULL << 17)
#define AID_SELECT    (1ULL << 18)
#define AID_SELECT_HI (1ULL << 19)
#define AID_STRIPE    (1ULL << 20)

static uint64_t range_flags(unsigned range)
{
	return AID_LOCAL | AMDGPU_GEM_CREATE_NO_CPU_ACCESS |
	       ((range & 1) ? AID_SELECT : 0) |
	       ((range & 2) ? AID_SELECT_HI : 0);
}

int aid_hbm_pci_bus_id(int hip_dev, char *out, int n)
{
	if (!out || n < 16)
		return -1;
	if (hipSetDevice(hip_dev) != hipSuccess)
		return -1;
	return hipDeviceGetPCIBusId(out, n, hip_dev) == hipSuccess ? 0 : -1;
}

static int open_render_for_hip_dev(int hip_dev)
{
	char bus[64], path[128], link[256];

	if (aid_hbm_pci_bus_id(hip_dev, bus, sizeof bus))
		return -1;
	for (char *p = bus; *p; p++)
		if (*p >= 'A' && *p <= 'F')
			*p = (char)(*p - 'A' + 'a');
	for (int m = 128; m < 200; m++) {
		ssize_t n;
		snprintf(path, sizeof path, "/sys/class/drm/renderD%d/device", m);
		n = readlink(path, link, sizeof link - 1);
		if (n <= 0)
			continue;
		link[n] = 0;
		if (!strstr(link, bus))
			continue;
		snprintf(path, sizeof path, "/dev/dri/renderD%d", m);
		return open(path, O_RDWR | O_CLOEXEC);
	}
	return -1;
}

#ifndef AMDGPU_GEM_CREATE_EXT_COHERENT
#define AMDGPU_GEM_CREATE_EXT_COHERENT (1ULL << 15)
#endif

static void *gem_import(int hip_dev, unsigned long long bytes, uint64_t flags,
			unsigned long long align, void **cpu_out)
{
	union drm_amdgpu_gem_create req;
	struct drm_prime_handle prime;
	hipExternalMemoryHandleDesc hd;
	hipExternalMemoryBufferDesc bd;
	hipExternalMemory_t ext = 0;
	void *ptr = 0;
	uint32_t handle;
	int fd, drm;

	if (!align)
		align = 4096;
	bytes = (bytes + align - 1) & ~(align - 1);

	drm = open_render_for_hip_dev(hip_dev);
	if (drm < 0) {
		fprintf(stderr, "aid_hbm: no render node for HIP device %d\n",
			hip_dev);
		return 0;
	}

	memset(&req, 0, sizeof req);
	req.in.bo_size = bytes;
	req.in.alignment = align;
	req.in.domains = AMDGPU_GEM_DOMAIN_VRAM;
	req.in.domain_flags = flags;
	if (ioctl(drm, DRM_IOCTL_AMDGPU_GEM_CREATE, &req)) {
		fprintf(stderr, "aid_hbm: GEM_CREATE %llu B flags=0x%llx: %s\n",
			bytes, (unsigned long long)flags, strerror(errno));
		close(drm);
		return 0;
	}
	handle = req.out.handle;
	memset(&prime, 0, sizeof prime);
	prime.handle = handle;
	if (ioctl(drm, DRM_IOCTL_PRIME_HANDLE_TO_FD, &prime)) {
		fprintf(stderr, "aid_hbm: PRIME: %s\n", strerror(errno));
		struct drm_gem_close cl = { .handle = handle };
		ioctl(drm, DRM_IOCTL_GEM_CLOSE, &cl);
		close(drm);
		return 0;
	}
	fd = prime.fd;
	memset(&hd, 0, sizeof hd);
	hd.type = hipExternalMemoryHandleTypeOpaqueFd;
	hd.handle.fd = fd;
	hd.size = bytes;
	if (hipImportExternalMemory(&ext, &hd) != hipSuccess) {
		fprintf(stderr, "aid_hbm: hipImportExternalMemory failed (%s)\n",
			hipGetErrorString(hipGetLastError()));
		close(fd);
		struct drm_gem_close cl = { .handle = handle };
		ioctl(drm, DRM_IOCTL_GEM_CLOSE, &cl);
		close(drm);
		return 0;
	}
	memset(&bd, 0, sizeof bd);
	bd.offset = 0;
	bd.size = bytes;
	if (hipExternalMemoryGetMappedBuffer(&ptr, ext, &bd) != hipSuccess) {
		fprintf(stderr, "aid_hbm: hipExternalMemoryGetMappedBuffer failed\n");
		close(fd);
		struct drm_gem_close cl = { .handle = handle };
		ioctl(drm, DRM_IOCTL_GEM_CLOSE, &cl);
		close(drm);
		return 0;
	}
	if (cpu_out) {
		union drm_amdgpu_gem_mmap mm;
		void *cpu;

		memset(&mm, 0, sizeof mm);
		mm.in.handle = handle;
		if (ioctl(drm, DRM_IOCTL_AMDGPU_GEM_MMAP, &mm)) {
			fprintf(stderr, "aid_hbm: GEM_MMAP: %s\n", strerror(errno));
			*cpu_out = 0;
		} else {
			cpu = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED,
				   drm, (off_t)mm.out.addr_ptr);
			if (cpu == MAP_FAILED) {
				fprintf(stderr,
					"aid_hbm: mmap %llu B off=0x%llx: %s\n",
					bytes,
					(unsigned long long)mm.out.addr_ptr,
					strerror(errno));
				*cpu_out = 0;
			} else {
				*cpu_out = cpu;
			}
		}
	}
	/* Leak drm fd, dma-buf fd, CPU mmap, and HIP mapping: buffers live
	 * for the process. Closing any of them can drop the GEM. */
	(void)ext;
	return ptr;
}

void *aid_hbm_alloc(int hip_dev, unsigned long long bytes, unsigned range)
{
	return gem_import(hip_dev, bytes, range_flags(range), 2ULL << 20, 0);
}

void *aid_hbm_alloc_hostmap(int hip_dev, unsigned long long bytes,
			    unsigned range, void **cpu_out)
{
	uint64_t flags = AID_LOCAL |
			 ((range & 1) ? AID_SELECT : 0) |
			 ((range & 2) ? AID_SELECT_HI : 0);

	if (!cpu_out)
		return 0;
	*cpu_out = 0;
	return gem_import(hip_dev, bytes, flags, 2ULL << 20, cpu_out);
}

void *aid_hbm_alloc_striped(int hip_dev, unsigned long long bytes)
{
	/* 8 MiB: 4 NPS ranges × 2 MiB so each lane is 2 MiB aligned. */
	return gem_import(hip_dev, bytes,
			  AID_STRIPE | AMDGPU_GEM_CREATE_NO_CPU_ACCESS,
			  8ULL << 20, 0);
}

/* Small VRAM BO with MTYPE_CC (EXT_COHERENT). Spanning-XCP hipMalloc is
 * forced to MTYPE_NC; GEM_CREATE_COHERENT alone does not override that.
 * EXT_COHERENT does: gfx950 maps it to CC when the BO is local. Use this
 * for barrier/event atomics so they hit the L2 directory instead of HBM.
 */
void *aid_hbm_alloc_cached(int hip_dev, unsigned long long bytes)
{
	return gem_import(hip_dev, bytes,
			  AMDGPU_GEM_CREATE_EXT_COHERENT |
				  AMDGPU_GEM_CREATE_NO_CPU_ACCESS,
			  4096, 0);
}
