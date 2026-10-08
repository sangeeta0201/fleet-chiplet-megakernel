"""Inputs for test_qkv_fp4.hip: one random qkv_a-shaped weight packed twice.

w4.bin is pack_dense_mxfp8(fp4=True) -- E2M1 nibbles, what MPK_QKV_MXFP4
streams. w8.bin is the same E2M1 values through mxfp4_roundtrip and then
packed as MXFP8 (exact, see mxfp4_roundtrip), which the E4M3 kernel path
reads. Both K-major at OPW 16, the qkv_a call site's layout. The packers are
demo.py's own, lifted out by source so the test needs no model or mirage.

usage: python3 gen_qkv_fp4.py <demo.py> <outdir> [rows]
"""
import os
import sys

import torch

src = open(sys.argv[1]).read()
out = sys.argv[2]
rows = int(sys.argv[3]) if len(sys.argv) > 3 else 256
K = 6144
ns = {"torch": torch, "os": os}
start = src.index("# E2M1, the MXFP4 element format")
exec(src[start:src.index("\n\n", start)], ns)
for fn in ("def _kmajor_permute", "def pack_mxfp8_workgroup",
           "def quantize_mxfp8", "def fake_quantize_mxfp4", "def quantize_mxfp4",
           "def mxfp4_roundtrip", "def pack_dense_mxfp8"):
    i = src.index(fn)
    j = src.index("\n\n\n", i)
    exec(src[i:j], ns)

torch.manual_seed(1)
w = (torch.randn(rows, K) * 0.02).to(torch.bfloat16)
x = torch.randn(K).to(torch.bfloat16)
gamma = (1.0 + 0.1 * torch.randn(K)).to(torch.bfloat16)
w4 = ns["pack_dense_mxfp8"](w, 16, fp4=True, kmajor=2)
w8 = ns["pack_dense_mxfp8"](w, 16, fake_exact=True, kmajor=2)
os.makedirs(out, exist_ok=True)
for name, t in (("x", x), ("gamma", gamma), ("w4", w4), ("w8", w8)):
    t.contiguous().view(torch.uint8).numpy().tofile(os.path.join(out, name + ".bin"))
print(f"rows={rows} K={K} w4 {tuple(w4.shape)} w8 {tuple(w8.shape)}")
