#!/usr/bin/env python3
"""Diff demo/glm5's torch MXFP8 packer against the device packer.

The kernel is validated against a dequantize-and-sum reference over bytes the
GPU itself produced (test_mxfp8_linear.hip), so a torch packer that lays rows
or exponents out differently would sail past that test and simply quantize the
model wrong -- which looks exactly like the format being bad. Byte equality is
the only check that separates the two.

Usage:
    export MXFP8_DUMP_DIR=/tmp/mxfp8 && mkdir -p $MXFP8_DUMP_DIR
    ./test_mxfp8_linear
    python3 check_mxfp8_pack.py $MXFP8_DUMP_DIR
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                    "demo", "glm5"))

# Must match test_mxfp8_linear.hip.
K = 2048
N = 2048
OPW = 64


def main():
    dump = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mxfp8"
    from demo import pack_mxfp8_workgroup, quantize_mxfp8

    raw = np.fromfile(os.path.join(dump, "w_bf16.bin"), dtype=np.uint16)
    assert raw.size == N * K, f"expected {N * K} bf16, got {raw.size}"
    w = torch.from_numpy(raw.astype(np.uint32) << 16).view(torch.float32)
    w = w.reshape(N, K).to(torch.bfloat16)

    data, scales = quantize_mxfp8(w)
    got = pack_mxfp8_workgroup(data, scales, output_per_wg=OPW).reshape(-1)
    want = torch.from_numpy(
        np.fromfile(os.path.join(dump, "w_packed.bin"), dtype=np.uint8))

    assert got.numel() == want.numel(), \
        f"size mismatch: torch {got.numel()} vs device {want.numel()}"

    diff = (got != want).nonzero().flatten()
    wg_bytes = OPW * K + OPW * (K // 32)
    if diff.numel() == 0:
        print(f"[PASS] {got.numel()} bytes identical "
              f"({N // OPW} workgroups x {wg_bytes} bytes)")
        return 0

    # Say whether the disagreement is in the values or the exponents; they
    # fail for different reasons and the distinction saves a bisect.
    off = diff % wg_bytes
    n_data = int((off < OPW * K).sum())
    print(f"[FAIL] {diff.numel()} of {got.numel()} bytes differ "
          f"({n_data} in data, {diff.numel() - n_data} in scales)")
    for i in diff[:8].tolist():
        print(f"  byte {i}: torch 0x{int(got[i]):02x} "
              f"device 0x{int(want[i]):02x}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
