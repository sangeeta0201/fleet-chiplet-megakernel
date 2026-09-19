"""AID-local VRAM placement for megakernel weight tensors.

MI350 has two physical AIDs (XCC 0-3 and XCC 4-7). QKV, O-proj, router, and
LM-head weights are already striped on dim 0 across the 8 XCDs
(`input_map = (0, -1, -1)`), so XCD 0-3 read the first half of that axis and
XCD 4-7 the second. Putting those halves in AID0 and AID1 HBM makes every
weight load of those ops local.

MoE W13/W2 is indexed `tile_idx * 8 + xcd_id` into a single tensor, so it
cannot be half-split on dim 0. `replicate_full` copies the whole tensor onto
both AIDs; codegen switches base on XCD 4-7 with no offset adjust.

Activations and barriers stay in the default (shared, MTYPE_NC) allocation.

Requires the patched amdgpu (AID_LOCAL GEM flags) and SPX+NPS2.
Enable with MPK_AID_LOCAL=1.

MPK_AID_POLARITY=1 swaps which NPS range backs XCD 0-3 vs XCD 4-7. Codegen
still maps logical XCD 0-3 to data_ptr (first dim-0 half / replica 0); only
the physical AID of that buffer flips. Mosaic lut_bench showed logical
XCC_ID → physical AID can invert per GPU and per boot.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

_LIB = None
_KEEP = []  # imported BOs must outlive the megakernel
_HIP = None
# One 128 GiB 4-range striped BO. Per-tensor GEM_CREATE rounded each lane
# up to a power of two (282 MiB -> 512 MiB) and filled the 36 GiB 1-stack
# window after a few MoE weights. Suballocate instead.
_STRIPE_ARENA = 0
_STRIPE_OFF = 0
_STRIPE_ARENA_BYTES = 64 << 30
# Packed W13/W2 are 1124.12 / 562.06 MiB → 2 MiB align 1126 / 564. A 562 MiB
# unit leaves 0.06 MiB short so 14×562 BOs cannot hold 36 downs. Pair-sized
# BOs (1126+564) pack exactly when filled gate,down,gate,down (not largest-first).
_GATE_BO = 1126 << 20
_DOWN_BO = 564 << 20
_PAIR_BO = _GATE_BO + _DOWN_BO  # 1690 MiB; 36 pairs = 59.41 GiB
# Pair-shaped BOs only. 564 MiB leftovers pass a byte check but cannot hold
# W13. HIP import OOMs around a dozen mappings, so fewer larger BOs.
_CHUNK_TRY = (
    36 * _PAIR_BO,
    18 * _PAIR_BO,
    12 * _PAIR_BO,
    9 * _PAIR_BO,
    6 * _PAIR_BO,
    4 * _PAIR_BO,
    3 * _PAIR_BO,
    2 * _PAIR_BO,
    _PAIR_BO,
)
_TARGET_ARENA = 72 << 30
_NEED_PAIRS = 36
_MIN_ARENA = _NEED_PAIRS * _PAIR_BO
_REPLICA_CHUNKS: list[list[list]] = [[], []]  # [ptr, size, used]
_REPLICA_OFF = [0, 0]
_REPLICA_RANGE = [-1, -1]
_REPLICA_HIP = 0
_STAGING_GPU = 0
_STAGING_CPU = 0

AID_BOUNDARY_PFN = 0x1B00000  # 108 GiB: below = AID0, above = AID1
# Firmware NPS2 on this 7-stack SKU exposes 36+72+72+72 GiB. The 36 GiB pack
# is 1-stack; skip it. 72 GiB is 2-stack. PROGRAM_DF steal windows are a few
# GiB and must not be used once DF maps are restored to NPS1.
MIN_AID_RANGE_BYTES = 64 << 30


def enabled() -> bool:
    return os.environ.get("MPK_AID_LOCAL", "0") == "1"


def polarity() -> int:
    """1 = place XCD 0-3 data on AID1 (high PFN) and XCD 4-7 on AID0.

    Kernel codegen is unchanged: bid.x < 4 still uses data_ptr. This only
    swaps the GEM range of that pointer. Must be 0 or 1.
    """
    v = os.environ.get("MPK_AID_POLARITY", "0")
    if v not in ("0", "1"):
        raise ValueError(f"MPK_AID_POLARITY must be 0 or 1, got {v!r}")
    return int(v)


def apply_polarity(r0: int, r1: int) -> tuple[int, int]:
    """Map (AID0 range, AID1 range) → (XCD 0-3 range, XCD 4-7 range)."""
    return (r1, r0) if polarity() else (r0, r1)


def worker_ranges(pci: str | None = None, hip_dev: int = 0) -> tuple[int, int]:
    """NPS discovery-range indices for XCD 0-3 data, then XCD 4-7 data."""
    return apply_polarity(*pick_ranges(pci=pci, hip_dev=hip_dev))


def _hip():
    global _HIP
    if _HIP is None:
        lib = ctypes.CDLL("libamdhip64.so")
        lib.hipMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        lib.hipMemcpy.restype = ctypes.c_int
        _HIP = lib
    return _HIP


def _so() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    here = Path(__file__).resolve().parent
    src = here / "aid_hbm.c"
    # Not "aid_hbm.so": a plain ctypes library sitting next to aid_hbm.py
    # shadows it on import, and it has no PyInit_aid_hbm.
    so = here / "libaid_hbm.so"
    if not so.exists() or so.stat().st_mtime < src.stat().st_mtime:
        rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
        cmd = [
            "gcc", "-shared", "-fPIC", "-O2",
            "-D__HIP_PLATFORM_AMD__=1",
            f"-I{rocm}/include",
            "-o", str(so), str(src),
            f"-L{rocm}/lib", "-lamdhip64", "-ldrm",
            f"-Wl,-rpath,{rocm}/lib",
        ]
        subprocess.check_call(cmd)
    lib = ctypes.CDLL(str(so))
    lib.aid_hbm_alloc.argtypes = [
        ctypes.c_int, ctypes.c_ulonglong, ctypes.c_uint,
    ]
    lib.aid_hbm_alloc.restype = ctypes.c_void_p
    lib.aid_hbm_alloc_hostmap.argtypes = [
        ctypes.c_int, ctypes.c_ulonglong, ctypes.c_uint,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.aid_hbm_alloc_hostmap.restype = ctypes.c_void_p
    lib.aid_hbm_alloc_uncached.argtypes = [
        ctypes.c_int, ctypes.c_ulonglong, ctypes.c_uint,
    ]
    lib.aid_hbm_alloc_uncached.restype = ctypes.c_void_p
    lib.aid_hbm_alloc_striped.argtypes = [
        ctypes.c_int, ctypes.c_ulonglong,
    ]
    lib.aid_hbm_alloc_striped.restype = ctypes.c_void_p
    lib.aid_hbm_alloc_cached.argtypes = [
        ctypes.c_int, ctypes.c_ulonglong,
    ]
    lib.aid_hbm_alloc_cached.restype = ctypes.c_void_p
    lib.aid_hbm_pci_bus_id.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
    ]
    lib.aid_hbm_pci_bus_id.restype = ctypes.c_int
    _LIB = lib
    return _LIB


def pci_bus_id(hip_dev: int = 0) -> str:
    buf = ctypes.create_string_buffer(64)
    if _so().aid_hbm_pci_bus_id(hip_dev, buf, 64) != 0:
        raise RuntimeError(f"hipDeviceGetPCIBusId failed for HIP device {hip_dev}")
    return buf.value.decode()


def pick_ranges(pci: str | None = None, hip_dev: int = 0) -> tuple[int, int]:
    """Largest NPS range on each side of the AID boundary (prefer 2-stack)."""
    if pci is None:
        pci = os.environ.get("AID_PCI") or pci_bus_id(hip_dev)
    pci = pci.lower()
    path = f"/sys/bus/pci/devices/{pci}/aid_aperture"
    ranges: list[tuple[int, int, int]] = []  # idx, fpfn, size
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except FileNotFoundError as e:
        raise RuntimeError(
            f"missing {path}: need the patched amdgpu with AID apertures "
            "and SPX+NPS2"
        ) from e
    for line in lines:
        parts = line.split()
        if len(parts) >= 7 and parts[0].startswith("aid") and parts[1] == "fpfn":
            idx = int(parts[0][3:])
            fpfn = int(parts[2], 16)
            size = int(parts[6], 16)
            ranges.append((idx, fpfn, size))
    # Drop the 36 GiB 1-stack pack (and any sub-64 GiB steal windows).
    ranges = [r for r in ranges if r[2] >= MIN_AID_RANGE_BYTES]
    aid0 = [r for r in ranges if r[1] < AID_BOUNDARY_PFN]
    aid1 = [r for r in ranges if r[1] >= AID_BOUNDARY_PFN]
    if not aid0 or not aid1:
        raise RuntimeError(
            f"need 72 GiB-class ranges on both AIDs in {path} "
            f"(skip 36 GiB 1-stack), got {ranges} (from {lines!r}). "
            "Firmware is NPS1 until a PSP NPS2 request; do not use "
            "rocm-smi --setmemorypartition."
        )
    r0 = max(aid0, key=lambda r: r[2])[0]
    r1 = max(aid1, key=lambda r: r[2])[0]
    return r0, r1


def alloc_bytes(nbytes: int, range_idx: int, hip_dev: int = 0) -> int:
    nbytes = (nbytes + (2 << 20) - 1) & ~((2 << 20) - 1)
    print(
        f"[AID_LOCAL] GEM_CREATE {nbytes / (1 << 20):.1f} MiB range {range_idx}",
        flush=True,
    )
    ptr = _so().aid_hbm_alloc(hip_dev, nbytes, range_idx)
    if not ptr:
        raise RuntimeError(
            f"AID_LOCAL alloc failed ({nbytes} B, range {range_idx}). "
            "Need patched amdgpu, SPX+NPS2, and MPK_AID_LOCAL=1."
        )
    print(f"[AID_LOCAL] GEM_CREATE ok {hex(int(ptr))}", flush=True)
    _KEEP.append(ptr)
    return int(ptr)


def alloc_bytes_window(nbytes: int, range_idx: int, hip_dev: int = 0) -> int:
    """AID window only: NO_CPU_ACCESS, then hostmap. Never hipMalloc/map0."""
    try:
        return alloc_bytes(nbytes, range_idx, hip_dev)
    except RuntimeError as e:
        print(f"[AID_LOCAL] nocpu miss, try hostmap: {e}", flush=True)
    ptr, _cpu = alloc_bytes_hostmap(nbytes, range_idx, hip_dev)
    return ptr


def alloc_bytes_hostmap(nbytes: int, range_idx: int, hip_dev: int = 0) -> tuple[int, int]:
    """AID_LOCAL GEM with a CPU mmap so we can fill without GPU DMA.

    hipMemcpy into a 1.1 GiB hipImport hangs inside demo.py (first 4 MiB).
    4 MiB GEMs DMA fine. Filling large BOs from the host mmap avoids that.
    """
    nbytes = (nbytes + (2 << 20) - 1) & ~((2 << 20) - 1)
    cpu = ctypes.c_void_p()
    print(
        f"[AID_LOCAL] GEM_CREATE_HOSTMAP {nbytes / (1 << 20):.1f} MiB range {range_idx}",
        flush=True,
    )
    ptr = _so().aid_hbm_alloc_hostmap(hip_dev, nbytes, range_idx, ctypes.byref(cpu))
    if not ptr or not cpu.value:
        raise RuntimeError(
            f"AID_LOCAL hostmap alloc failed ({nbytes} B, range {range_idx})."
        )
    print(
        f"[AID_LOCAL] GEM_CREATE_HOSTMAP ok gpu={hex(int(ptr))} cpu={hex(int(cpu.value))}",
        flush=True,
    )
    _KEEP.append((ptr, cpu.value))
    return int(ptr), int(cpu.value)


def prefetch_hostmap(cpu: int, nbytes: int) -> None:
    """Touch one byte per 2 MiB so GPU PTEs exist before hipMemcpy."""
    page = 2 << 20
    t0 = time.time()
    off = 0
    while off < nbytes:
        b = ctypes.c_ubyte.from_address(cpu + off)
        b.value = b.value
        off += page
    print(
        f"[AID_LOCAL] prefetch {nbytes / (1 << 20):.0f} MiB in {time.time() - t0:.3f}s",
        flush=True,
    )


def _munmap(ptr: int, nbytes: int) -> None:
    libc = ctypes.CDLL("libc.so.6")
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.munmap.restype = ctypes.c_int
    libc.munmap(ptr, nbytes)


def memcpy_hostmap(cpu_dst: int, src: int, nbytes: int) -> None:
    chunk = 64 << 20
    off = 0
    t0 = time.time()
    while off < nbytes:
        n = min(chunk, nbytes - off)
        ctypes.memmove(cpu_dst + off, src + off, n)
        off += n
        if nbytes > chunk and (off % (256 << 20) == 0):
            print(
                f"[AID_LOCAL] hostmap fill {off / (1 << 20):.0f}/{nbytes / (1 << 20):.0f} MiB",
                flush=True,
            )
    if nbytes > chunk:
        print(
            f"[AID_LOCAL] hostmap fill {nbytes / (1 << 20):.0f} MiB done in "
            f"{time.time() - t0:.2f}s",
            flush=True,
        )


def ensure_replica_arenas(hip_dev: int = 0, need_bytes: int = 0) -> None:
    """NO_CPU_ACCESS chunks covering both 72 GiB AID windows.

    Call while hipMalloc occupancy in the AID windows is low (right after
    spilling MXFP4 experts to the host). 562 MiB-aligned BOs so 36×W13 +
    36×W2 first-fit. Fill from a reused 4 MiB hostmap staging via D2D.
    """
    global _REPLICA_HIP
    if _REPLICA_CHUNKS[0] and _REPLICA_CHUNKS[1]:
        return
    rw0, rw1 = worker_ranges(hip_dev=hip_dev)
    _REPLICA_RANGE[0], _REPLICA_RANGE[1] = rw0, rw1
    _REPLICA_HIP = hip_dev
    need = need_bytes if need_bytes > 0 else _MIN_ARENA
    need_pairs = max(_NEED_PAIRS, (need + _PAIR_BO - 1) // _PAIR_BO)
    for i, rw in enumerate((rw0, rw1)):
        got = 0
        pairs = 0
        print(
            f"[AID_LOCAL] prealloc arena{i} range {rw} "
            f"target {need_pairs} pair-shaped BOs ({need_pairs * _PAIR_BO / (1 << 30):.1f} GiB)",
            flush=True,
        )
        for sz in _CHUNK_TRY:
            n = sz // _PAIR_BO
            if n < 1:
                continue
            while pairs + n <= need_pairs and got + sz <= _TARGET_ARENA:
                try:
                    if n == 1:
                        ptr = alloc_bytes_window(sz, rw, hip_dev)
                    else:
                        ptr = alloc_bytes(sz, rw, hip_dev)
                except RuntimeError as e:
                    print(
                        f"[AID_LOCAL] arena{i} {sz / (1 << 20):.0f} MiB stopped at "
                        f"{pairs} pairs ({got / (1 << 30):.1f} GiB): {e}",
                        flush=True,
                    )
                    break
                _REPLICA_CHUNKS[i].append([ptr, sz, 0])
                got += sz
                pairs += n
            if pairs >= need_pairs:
                break
        print(
            f"[AID_LOCAL] arena{i} range {rw} got {got / (1 << 30):.1f} GiB "
            f"({pairs} pair-shaped) in {len(_REPLICA_CHUNKS[i])} BOs",
            flush=True,
        )
        if pairs < need_pairs:
            raise RuntimeError(
                f"AID arena{i} range {rw}: only {pairs} pair-shaped BOs "
                f"({got / (1 << 30):.1f} GiB, need {need_pairs})"
            )
        _REPLICA_OFF[i] = 0


def arena_take(nbytes: int, which: int) -> int:
    """First-fit into a chunk so leftover holes can hold smaller tensors."""
    need = (nbytes + (2 << 20) - 1) & ~((2 << 20) - 1)
    chunks = _REPLICA_CHUNKS[which]
    if not chunks:
        raise RuntimeError("ensure_replica_arenas was not called")
    for chunk in chunks:
        ptr, sz, used = chunk
        if used + need <= sz:
            chunk[2] = used + need
            return ptr + used
    used_tot = sum(c[2] for c in chunks)
    raise RuntimeError(
        f"AID arena {which} full "
        f"(need {need / (1 << 20):.1f} MiB, used {used_tot / (1 << 30):.1f} GiB)"
    )


def alloc_striped(nbytes: int, hip_dev: int = 0) -> int:
    """One VA, 4 KiB pages interleaved across 4 NPS discovery ranges."""
    nbytes = (nbytes + (8 << 20) - 1) & ~((8 << 20) - 1)
    ptr = _so().aid_hbm_alloc_striped(hip_dev, nbytes)
    if not ptr:
        raise RuntimeError(
            f"AID_STRIPE alloc failed ({nbytes} B). Need patched amdgpu "
            "with GEM_CREATE_AID_STRIPE and SPX+NPS2."
        )
    _KEEP.append(ptr)
    return int(ptr)


def ensure_stripe_arena(hip_dev: int = 0) -> int:
    """Allocate the 4-range striped arena after LM-head quantize.

    Try 64 GiB first, then smaller, so a resident model still leaves AID0
    room. hipImport of a huge BO while torch owns AID0 can OOM; smaller
    still covers oproj + a large slice of MoE.
    """
    global _STRIPE_ARENA, _STRIPE_OFF, _STRIPE_ARENA_BYTES
    if _STRIPE_ARENA:
        return _STRIPE_ARENA
    last_err = None
    for gb in (32, 16):
        _STRIPE_ARENA_BYTES = gb << 30
        print(
            f"[AID_STRIPE] allocating {_STRIPE_ARENA_BYTES / (1 << 30):.0f} GiB arena",
            flush=True,
        )
        try:
            _STRIPE_ARENA = alloc_striped(_STRIPE_ARENA_BYTES, hip_dev)
            _STRIPE_OFF = 0
            return _STRIPE_ARENA
        except RuntimeError as e:
            last_err = e
            print(f"[AID_STRIPE] {gb} GiB failed: {e}", flush=True)
    raise last_err


def stripe_copy(tensor, hip_dev: int = 0, drop_src: bool = True) -> int:
    """Copy a torch VRAM tensor onto a 7-stack AID_STRIPE BO and optionally
    free the original hipMalloc. One VA; both NPS2 windows. Barriers must
    not use this -- striping those hung the megakernel."""
    import torch

    global _STRIPE_ARENA, _STRIPE_OFF

    assert tensor.is_contiguous()
    nbytes = tensor.nbytes
    need = (nbytes + 4095) & ~4095
    if not _STRIPE_ARENA:
        ensure_stripe_arena(hip_dev)
    if _STRIPE_OFF + need > _STRIPE_ARENA_BYTES:
        print(
            f"[AID_STRIPE] skip {tuple(tensor.shape)}  "
            f"{nbytes / (1 << 20):.2f} MiB (arena full)",
            flush=True,
        )
        return 0
    ptr = _STRIPE_ARENA + _STRIPE_OFF
    memcpy_d2d(ptr, int(tensor.data_ptr()), nbytes)
    print(
        f"[AID_STRIPE] {tuple(tensor.shape)}  {nbytes / (1 << 20):.2f} MiB  "
        f"off={_STRIPE_OFF / (1 << 20):.1f} MiB",
        flush=True,
    )
    _STRIPE_OFF += need
    if drop_src:
        tensor.untyped_storage().resize_(0)
        torch.cuda.empty_cache()
    return ptr


def cached_copy(tensor, hip_dev: int = 0, drop_src: bool = True) -> int:
    """Copy a torch VRAM tensor onto an EXT_COHERENT (MTYPE_CC) BO.

    Barriers stay on hipMalloc (MTYPE_NC): CC atomics hang at 2+ XCDs on
    SPX+NPS2. Read-only weights never take atomics, so they can be CC and
    still use MALL/L2.
    """
    import torch

    assert tensor.is_contiguous()
    nbytes = tensor.nbytes
    ptr = alloc_cached(nbytes, hip_dev)
    memcpy_d2d(ptr, int(tensor.data_ptr()), nbytes)
    print(
        f"[CACHED_WEIGHTS] {tuple(tensor.shape)}  {nbytes / (1 << 20):.2f} MiB",
        flush=True,
    )
    if drop_src:
        tensor.untyped_storage().resize_(0)
        torch.cuda.empty_cache()
    return ptr


def alloc_cached(nbytes: int, hip_dev: int = 0) -> int:
    """VRAM BO with GEM_CREATE_EXT_COHERENT (MTYPE_CC on gfx950 local).

    hipMalloc on SPX+NPS2 is MTYPE_NC. Cross-XCD atomics/polls on NC miss
    L2 and go to HBM. This path is for barrier and event slabs only.
    """
    nbytes = max(4096, (nbytes + 4095) & ~4095)
    ptr = _so().aid_hbm_alloc_cached(hip_dev, nbytes)
    if not ptr:
        raise RuntimeError(
            f"EXT_COHERENT alloc failed ({nbytes} B). Need patched amdgpu "
            "where spanning XCP still honors GEM_CREATE_EXT_COHERENT."
        )
    _KEEP.append(ptr)
    return int(ptr)


def memset_d(ptr: int, value: int, nbytes: int) -> None:
    hip = _hip()
    hip.hipMemset.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t,
    ]
    hip.hipMemset.restype = ctypes.c_int
    err = hip.hipMemset(ptr, value, nbytes)
    if err:
        raise RuntimeError(f"hipMemset failed: {err}")


def hip_meminfo() -> tuple[int, int]:
    hip = _hip()
    free = ctypes.c_ulonglong()
    tot = ctypes.c_ulonglong()
    hip.hipMemGetInfo.argtypes = [
        ctypes.POINTER(ctypes.c_ulonglong),
        ctypes.POINTER(ctypes.c_ulonglong),
    ]
    hip.hipMemGetInfo.restype = ctypes.c_int
    err = hip.hipMemGetInfo(ctypes.byref(free), ctypes.byref(tot))
    if err:
        raise RuntimeError(f"hipMemGetInfo failed: {err}")
    return int(free.value), int(tot.value)


def memcpy_h2d_pinned(dst: int, src: int, nbytes: int) -> None:
    """Pinned-host to AID GPU ptr on a side stream, 4 MiB sync chunks.

    Default-stream D2D into a 1.1 GiB import hangs in demo.py even after
    PTE prefetch. QKV 4 MiB D2D works. Side-stream H2D from pin_memory
    avoids the torch default-stream lock and the extra staging alloc.
    """
    import torch

    hip = _hip()
    if not getattr(hip, "_async_ready", False):
        hip.hipMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        hip.hipMemcpyAsync.restype = ctypes.c_int
        hip._async_ready = True
    s = torch.cuda.Stream()
    stream = ctypes.c_void_p(int(s.cuda_stream))
    chunk = 4 << 20
    off = 0
    t0 = time.time()
    print(
        f"[AID_LOCAL] H2D pinned->AID {nbytes / (1 << 20):.0f} MiB",
        flush=True,
    )
    while off < nbytes:
        n = min(chunk, nbytes - off)
        err = hip.hipMemcpyAsync(dst + off, src + off, n, 1, stream)
        if err:
            raise RuntimeError(f"hipMemcpyAsync H2D failed: {err} at +{off}")
        s.synchronize()
        off += n
        if off == n or (nbytes > (64 << 20) and off % (64 << 20) == 0):
            print(
                f"[AID_LOCAL] H2D {off / (1 << 20):.0f}/{nbytes / (1 << 20):.0f} MiB",
                flush=True,
            )
    print(
        f"[AID_LOCAL] H2D {nbytes / (1 << 20):.0f} MiB done in {time.time() - t0:.2f}s",
        flush=True,
    )


def memcpy_d2d(dst: int, src: int, nbytes: int) -> None:
    """Synchronous 4 MiB hipMemcpy D2D chunks (small AID BOs / QKV splits)."""
    import torch

    torch.cuda.synchronize()
    hip = _hip()
    hip.hipDeviceSynchronize.restype = ctypes.c_int
    chunk = 4 << 20
    off = 0
    t0 = time.time()
    while off < nbytes:
        n = min(chunk, nbytes - off)
        err = hip.hipMemcpy(dst + off, src + off, n, 2)
        if err:
            raise RuntimeError(f"hipMemcpy D2D failed: {err} at +{off}")
        err = hip.hipDeviceSynchronize()
        if err:
            raise RuntimeError(f"hipDeviceSynchronize D2D failed: {err} at +{off}")
        off += n
        if nbytes > (64 << 20) and (off % (64 << 20) == 0):
            print(
                f"[AID_LOCAL] D2D {off / (1 << 20):.0f}/{nbytes / (1 << 20):.0f} MiB",
                flush=True,
            )
    if nbytes > (64 << 20):
        dt = time.time() - t0
        print(
            f"[AID_LOCAL] D2D {nbytes / (1 << 20):.0f} MiB done in {dt:.2f}s",
            flush=True,
        )


def split_dim0_halves(tensor, hip_dev: int = 0):
    """Copy tensor[:n/2] for XCD 0-3 and tensor[n/2:] for XCD 4-7.

    Returns (ptr_xcd0_3, ptr_xcd4_7). Dim 0 must be divisible by 8 so each
    XCD's (0,-1,-1) stripe is wholly in one half. MPK_AID_POLARITY selects
    which AID range each pointer lands in.
    """
    import torch

    assert tensor.is_contiguous()
    n = tensor.shape[0]
    if n % 8:
        raise ValueError(f"dim0 {n} is not divisible by 8 XCDs")
    half = n // 2
    row_bytes = tensor.nbytes // n
    half_bytes = half * row_bytes
    rw0, rw1 = worker_ranges(hip_dev=hip_dev)
    if not _REPLICA_CHUNKS[0]:
        p0 = alloc_bytes(half_bytes, rw0, hip_dev)
        p1 = alloc_bytes(half_bytes, rw1, hip_dev)
    else:
        # MoE arenas are 8 GiB hostmaps; QKV/o_proj stay on their own 4 MiB BOs.
        p0 = alloc_bytes(half_bytes, rw0, hip_dev)
        p1 = alloc_bytes(half_bytes, rw1, hip_dev)
    src = int(tensor.data_ptr())
    memcpy_d2d(p0, src, half_bytes)
    memcpy_d2d(p1, src + half_bytes, half_bytes)
    print(
        f"[AID_LOCAL] {tuple(tensor.shape)}  "
        f"XCD0-3 range {rw0} + XCD4-7 range {rw1}  "
        f"polarity={polarity()}  {half_bytes / (1 << 20):.2f} MiB each",
        flush=True,
    )
    return p0, p1


def replicate_full(tensor, hip_dev: int = 0, drop_src: bool = True):
    """Copy the whole tensor onto both worker-half AID ranges.

    Returns (ptr_xcd0_3, ptr_xcd4_7). Used for tensors the kernel indexes by
    xcd_id rather than a dim-0 stripe (MoE W13/W2). drop_src releases the
    original hipMalloc so we do not keep three copies (~60 GiB MoE).

    Suballocates from ensure_replica_arenas() (empty-VRAM 72 GiB windows).
    Per-tensor GEM_CREATE after the model is resident ENOMEM/hangs.
    """
    import torch

    assert tensor.is_contiguous()
    nbytes = tensor.nbytes
    rw0, rw1 = worker_ranges(hip_dev=hip_dev)
    print(
        f"[AID_LOCAL] replica {tuple(tensor.shape)}  "
        f"XCD0-3 range {rw0} + XCD4-7 range {rw1}  "
        f"polarity={polarity()}  {nbytes / (1 << 20):.2f} MiB each",
        flush=True,
    )
    host = tensor
    if tensor.is_cuda:
        print("[AID_LOCAL] spill replica src to host for hostmap", flush=True)
        host = tensor.detach().contiguous().cpu()
        if drop_src:
            tensor.untyped_storage().resize_(0)
            torch.cuda.empty_cache()
    if not host.is_pinned():
        host = host.pin_memory()
    src = int(host.data_ptr())
    if _REPLICA_CHUNKS[0]:
        p0 = arena_take(nbytes, 0)
        p1 = arena_take(nbytes, 1)
    else:
        p0 = alloc_bytes(nbytes, rw0, hip_dev)
        p1 = alloc_bytes(nbytes, rw1, hip_dev)
    global _STAGING_GPU, _STAGING_CPU
    if not _STAGING_GPU:
        _STAGING_GPU, _STAGING_CPU = alloc_bytes_hostmap(4 << 20, rw0, hip_dev)
    print("[AID_LOCAL] 4MiB staging D2D", flush=True)
    hip = _hip()
    hip.hipDeviceSynchronize.restype = ctypes.c_int
    torch.cuda.synchronize()
    chunk = 4 << 20
    off = 0
    t0 = time.time()
    while off < nbytes:
        n = min(chunk, nbytes - off)
        ctypes.memmove(_STAGING_CPU, src + off, n)
        err = hip.hipMemcpy(p0 + off, _STAGING_GPU, n, 2)
        if err:
            raise RuntimeError(f"D2D p0 failed: {err} at +{off}")
        err = hip.hipMemcpy(p1 + off, _STAGING_GPU, n, 2)
        if err:
            raise RuntimeError(f"D2D p1 failed: {err} at +{off}")
        err = hip.hipDeviceSynchronize()
        if err:
            raise RuntimeError(f"D2D sync failed: {err} at +{off}")
        off += n
        if off == n or (nbytes > (64 << 20) and off % (256 << 20) == 0):
            print(
                f"[AID_LOCAL] D2D {off / (1 << 20):.0f}/{nbytes / (1 << 20):.0f} MiB",
                flush=True,
            )
    print(f"[AID_LOCAL] replica fill done in {time.time() - t0:.2f}s", flush=True)
    if drop_src and host is not tensor:
        host = None
    elif drop_src:
        tensor.untyped_storage().resize_(0)
    return p0, p1


# ---------------------------------------------------------------------------
# Single-range AID_LOCAL placement for READ-ONLY weights.
#
# split_dim0_halves/replicate_full above place a tensor for an 8-XCD run that
# uses both ranges. The 4-XCD subset runs entirely on one range, so it wants
# the simpler thing: the whole tensor in that range, flagged AID_LOCAL so the
# driver gives it MTYPE_RW instead of the spanning default MTYPE_NC.
# ---------------------------------------------------------------------------
_LOCAL_ARENA = {"ptr": 0, "off": 0, "cap": 0, "range": -1}
_LOCAL_ARENA_BYTES = 8 << 30
_LOCAL_LOG = []


def local_range(hip_dev: int = 0) -> int:
    """The NPS range that XCDs 0-3 are close to, honouring MPK_AID_POLARITY."""
    r0, _r1 = worker_ranges(hip_dev=hip_dev)
    return r0


def _local_arena_take(nbytes: int, hip_dev: int = 0) -> int:
    nbytes = (nbytes + (2 << 20) - 1) & ~((2 << 20) - 1)
    a = _LOCAL_ARENA
    if a["ptr"] == 0 or a["off"] + nbytes > a["cap"]:
        cap = max(_LOCAL_ARENA_BYTES, nbytes)
        rng = local_range(hip_dev)
        a["ptr"] = alloc_bytes(cap, rng, hip_dev)
        a["off"] = 0
        a["cap"] = cap
        a["range"] = rng
        print(f"[AID_LOCAL] arena {cap >> 30} GiB in range {rng}", flush=True)
    p = a["ptr"] + a["off"]
    a["off"] += nbytes
    return p


def local_tensor(tensor, hip_dev: int = 0, drop_src: bool = False):
    """Return a tensor of the same shape/dtype backed by AID_LOCAL VRAM.

    Only safe for tensors the megakernel never writes. The import goes through
    __cuda_array_interface__ as raw bytes and is then viewed back to the
    original dtype, because bfloat16 has no CAI typestr.
    """
    import torch

    assert tensor.is_contiguous(), "AID_LOCAL copy needs a contiguous source"
    nbytes = tensor.nbytes
    dst = _local_arena_take(nbytes, hip_dev)
    memcpy_d2d(dst, int(tensor.data_ptr()), nbytes)

    class _CAI:
        pass

    w = _CAI()
    w.__cuda_array_interface__ = {
        "data": (dst, False),
        "shape": (nbytes,),
        "typestr": "|u1",
        "strides": None,
        "version": 2,
    }
    flat = torch.as_tensor(w, device=f"cuda:{hip_dev}")
    out = flat.view(tensor.dtype).reshape(tensor.shape)
    _LOCAL_LOG.append((tuple(tensor.shape), nbytes))
    print(
        f"[AID_LOCAL] weight {tuple(tensor.shape)} {nbytes / (1 << 20):.1f} MiB "
        f"-> range {_LOCAL_ARENA['range']} @ {hex(dst)}",
        flush=True,
    )
    if drop_src:
        tensor.untyped_storage().resize_(0)
    return out


def local_summary() -> str:
    tot = sum(n for _s, n in _LOCAL_LOG)
    return f"{len(_LOCAL_LOG)} tensors, {tot / (1 << 30):.2f} GiB AID_LOCAL"


# ---------------------------------------------------------------------------
# Uncached (MTYPE_NC) VRAM arena for the shared, written buffers.
# ---------------------------------------------------------------------------
_UNC_ARENA = {"ptr": 0, "off": 0, "cap": 0, "range": -1}
_UNC_ARENA_BYTES = 64 << 20   # whole shared set is ~130 KB at batch 1


def uncached_take(nbytes: int, hip_dev: int = 0, range_idx: int | None = None) -> int:
    """Suballocate MTYPE_NC VRAM. One BO, because hipImportExternalMemory
    OOMs after about a dozen external mappings."""
    a = _UNC_ARENA
    nbytes = (nbytes + 255) & ~255
    if a["ptr"] == 0:
        rng = local_range(hip_dev) if range_idx is None else range_idx
        cap = max(_UNC_ARENA_BYTES, nbytes)
        p = _so().aid_hbm_alloc_uncached(hip_dev, cap, rng)
        if not p:
            raise RuntimeError(
                f"uncached VRAM alloc failed ({cap} B, range {rng}); "
                "needs the patched amdgpu and aid_local_uncached_mtype=1"
            )
        a["ptr"], a["off"], a["cap"], a["range"] = int(p), 0, cap, rng
        _KEEP.append(p)
        memset_d(a["ptr"], 0, cap)
        print(f"[UNCACHED] arena {cap >> 20} MiB MTYPE_NC VRAM range {rng} "
              f"@ {hex(a['ptr'])}", flush=True)
    if a["off"] + nbytes > a["cap"]:
        raise RuntimeError("uncached arena exhausted")
    p = a["ptr"] + a["off"]
    a["off"] += nbytes
    return p
