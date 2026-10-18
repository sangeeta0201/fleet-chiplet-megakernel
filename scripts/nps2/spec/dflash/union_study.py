#!/usr/bin/env python3
"""Distinct experts per layer touched by a verify block of B consecutive tokens (runs inside fleet_v1).

Teacher-forces the PyTorch reference target over prompt + generated tokens, takes each layer's top-4
experts per position, and averages |union over positions s..s+B-1| over windows in the generated region.
A verify pass streams each distinct expert once, so this sets its MoE bytes.

usage: union_study.py TOK_JSON[,TOK_JSON...]
"""
import json
import os
import sys

import torch

sys.path.insert(0, "/root/user/fleet-lanes/placement/demo/gpt_oss")
from models.modeling_gpt_oss import GptOssForCausalLM  # noqa: E402

MODEL = os.environ.get("MODEL_PATH", "/root/user/models/gpt-oss-120b")
FILES = sys.argv[1].split(",")
BMAX = 10
with torch.device("cuda"):
    model = GptOssForCausalLM.from_pretrained(MODEL, 1, max_num_pages=16, page_size=4096).to(
        dtype=torch.bfloat16, device="cuda")
E, TOPK = 128, 4
tot = torch.zeros(BMAX + 1, dtype=torch.float64)
cnt = torch.zeros(BMAX + 1, dtype=torch.float64)
with torch.inference_mode():
    for path in FILES:
        tok = json.load(open(path))
        S = torch.tensor(tok["prompt_ids"] + tok["token_ids"], dtype=torch.long, device="cuda")
        N, P = S.numel(), len(tok["prompt_ids"])
        pos = torch.arange(N, device="cuda").unsqueeze(0)
        cos, sin = model.model.rotary_emb(pos)
        _, rl = model.model(input_ids=S.unsqueeze(0), position_embeddings=(cos, sin), step=0)
        L = len(rl)
        oh = torch.zeros(L, N, E, dtype=torch.bool, device="cuda")
        for li, r in enumerate(rl):
            idx = r.reshape(N, -1)[:, :E].float().topk(TOPK, dim=-1).indices
            oh[li].scatter_(1, idx, True)
        g = oh[:, P:, :]  # generated region [L, G, E]
        G = g.shape[1]
        row = []
        for B in range(1, BMAX + 1):
            if G < B:
                break
            # union over windows of B consecutive positions: OR via cumulative counts
            c = torch.cumsum(g.int(), dim=1)
            c = torch.cat([torch.zeros_like(c[:, :1]), c], 1)
            win = (c[:, B:] - c[:, :-B]) > 0          # [L, G-B+1, E]
            u = win.sum(-1).float().mean().item()
            tot[B] += u * (G - B + 1)
            cnt[B] += G - B + 1
            row.append("%.2f" % u)
        ov = (g[:, 1:] & g[:, :-1]).sum(-1).float().mean().item()
        print("%-26s U(B=1..%d) %s   consecutive shared experts %.2f of 4" % (
            os.path.basename(path), BMAX, " ".join(row), ov), flush=True)
ind = [E * (1 - (1 - TOPK / E) ** B) for B in range(1, BMAX + 1)]
print("pooled          U(B=1..%d) %s" % (BMAX, " ".join("%.2f" % (tot[B] / cnt[B]) for B in range(1, BMAX + 1))))
print("independent     U(B=1..%d) %s" % (BMAX, " ".join("%.2f" % u for u in ind)))
