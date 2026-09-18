#!/usr/bin/env python3
"""Host-side check that dim-0 AID split remaps XCD 4-7 onto <name>_aid1.

Does not allocate VRAM or run a kernel. Pointers are fake constants so this
can run while a decode smoke occupies a GPU.
"""
from __future__ import annotations

import json
import sys

import torch

from mirage.core import CyKNGraph, CyTBGraph, uint8
from mirage.kernel import KNGraph
from mirage.threadblock import TBGraph


def main() -> int:
    n_wgs, wg_bytes, n_xcd = 80, 128, 8
    dummy = torch.empty((n_wgs, wg_bytes), dtype=torch.uint8)
    dims = tuple(dummy.shape)
    strides = tuple(dummy.stride())
    p0, p1 = 0x1000, 0x2000

    g = KNGraph(CyKNGraph(disable_fingerprint=True))
    weight = g.new_input(dims=dims, strides=strides, dtype=uint8)
    g.attach_torch_tensor(weight, dummy, "qkv_w", data_ptr=p0, data_ptr_aid1=p1)
    out = g.new_input(dims=dims, strides=strides, dtype=uint8)
    g.attach_torch_tensor(out, dummy, "qkv_out")

    tb = TBGraph(CyTBGraph((n_xcd, 1, 1), (256, 1, 1), 1, 64))
    tb.new_input(weight, (0, -1, -1), 1, True)
    tb.new_input(out, (0, -1, -1), -1, True)
    g.customized([weight, out], tb)
    g.register_task(tb, "identity")
    result = g.generate_task_graph(1, 0)

    code = result["cuda_code"]
    if "qkv_w_aid1" not in code:
        print("FAIL: generated init is missing qkv_w_aid1", file=sys.stderr)
        return 1
    if hex(p0) not in code and str(p0) not in code:
        # void* prints as 0x1000
        print("FAIL: AID0 pointer not embedded in init", file=sys.stderr)
        print(code[:1500], file=sys.stderr)
        return 1

    graph = json.loads(result["json_file"])
    block = n_wgs // n_xcd  # 10 rows per XCD
    stride0 = wg_bytes
    seen = []
    for task in graph["all_tasks"]:
        ins = task.get("inputs") or []
        if not ins:
            continue
        name = ins[0].get("base_ptr")
        if name not in ("qkv_w", "qkv_w_aid1"):
            continue
        seen.append((name, ins[0]["offset"]))

    if len(seen) != n_xcd:
        print(f"FAIL: expected {n_xcd} qkv_w tasks, got {seen}", file=sys.stderr)
        return 1

    for xcd, (name, off) in enumerate(seen):
        local = xcd if xcd < 4 else xcd - 4
        want_name = "qkv_w" if xcd < 4 else "qkv_w_aid1"
        want_off = local * block * stride0
        if name != want_name or off != want_off:
            print(
                f"FAIL: XCD {xcd}: got ({name}, {off}), "
                f"want ({want_name}, {want_off})",
                file=sys.stderr,
            )
            return 1

    print("AID split offsets OK:")
    for xcd, (name, off) in enumerate(seen):
        print(f"  XCD {xcd}: {name} + {off}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
