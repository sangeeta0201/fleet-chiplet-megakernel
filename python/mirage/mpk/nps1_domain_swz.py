"""NPS1 concat-Mod7 sparse pack into hipMalloc (2 MiB PTEs).

Stores logical bytes only in 4 KiB pages whose stack hashes to `aid`.
map[logical_4k] at the replica base is the physical offset of that page.
"""
from __future__ import annotations

import os

import numpy as np
import torch

HEADER = 16384


def polarity() -> int:
    v = os.environ.get("MPK_AID_POLARITY", "0")
    if v not in ("0", "1"):
        raise ValueError(f"MPK_AID_POLARITY must be 0 or 1, got {v!r}")
    return int(v)


def concat_mod7(pa: int) -> int:
    hi = pa >> 16
    x = (hi & 0xF) ^ ((pa >> 12) & 0xF)
    return int(((hi << 4) | x) % 7)


def stack_aid(st: int) -> int:
    return 0 if st in (0, 1, 4) else 1


def packed_stride(logical_bytes: int) -> int:
    raw = logical_bytes * 7 // 3 + HEADER + (2 << 20)
    return (raw + (1 << 21) - 1) & ~((1 << 21) - 1)


def pack_into(dst_u8: torch.Tensor, src_u8: torch.Tensor, aid: int) -> None:
    """CPU pack of src into existing GPU dst (dst.data_ptr() is the VA)."""
    src = src_u8.detach().contiguous().view(torch.uint8).flatten().cpu().numpy()
    n = int(src.size)
    va = int(dst_u8.data_ptr())
    stride = int(dst_u8.numel())
    dst = np.zeros(stride, dtype=np.uint8)
    n4 = (n + 4095) // 4096
    mp = np.zeros(n4, dtype=np.uint32)
    off = HEADER
    si = 0
    page_i = 0
    while si < n:
        if off + 4096 > stride:
            raise RuntimeError(
                f"NPS1 pack overflow aid={aid} off={off} stride={stride} si={si}/{n}"
            )
        st = concat_mod7(va + off)
        if stack_aid(st) == aid:
            take = min(4096, n - si)
            dst[off : off + take] = src[si : si + take]
            mp[page_i] = off
            page_i += 1
            si += take
            off += 4096
        else:
            off += 4096
    if page_i != n4:
        raise RuntimeError(f"NPS1 pack pages {page_i} != {n4}")
    dst[: n4 * 4] = mp.view(np.uint8)
    dst_u8.copy_(torch.from_numpy(dst))


def pack_replica(tensor: torch.Tensor, aid: int) -> tuple[int, torch.Tensor]:
    src = tensor.detach().contiguous().view(torch.uint8).flatten()
    n = src.numel()
    stride = packed_stride(n)
    dst = torch.empty(stride, dtype=torch.uint8, device=src.device)
    pack_into(dst, src, aid ^ polarity())
    return int(dst.data_ptr()), dst


def replica_pair(tensor: torch.Tensor) -> tuple[int, int, torch.Tensor, torch.Tensor]:
    """Return (p0, p1, keep0, keep1). p1 may alias tensor if VRAM is tight."""
    p0, t0 = pack_replica(tensor, 0)
    p1, t1 = pack_replica(tensor, 1)
    return p0, p1, t0, t1
