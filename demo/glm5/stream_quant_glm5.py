#!/usr/bin/env python3
"""Streaming FP8 -> MXFP4/bf16 converter for GLM-5-FP8 (744B, 142 shards, 756 GB).

The checkpoint does not fit: 756 GB of shards against 635 GB of free disk. It
never has to. Per shard we download one file, dequantize its FP8 128x128 blocks
to bf16, re-pack the routed experts as MXFP4, write the result, and delete the
FP8 shard -- so peak disk is the converted output plus one 5.4 GB shard.

    routed experts  mlp.experts.<e>.{gate,up,down}_proj  ->  MXFP4   390 GB
    everything else (attention, shared/dense MLP, indexer, embed, lm_head,
    norms, router gate + bias)                           ->  bf16     38 GB
                                                            ------------
                                                                     428 GB

Routed experts are 97% of the parameters and already run MXFP4 in the
megakernel (GLM_MOE_MXFP4=1, the default). Everything else stays bf16 on disk
so the existing load-time MXFP8 packers (pack_dense_mxfp8, GLM_QB_MXFP8 and
friends) keep working untouched -- 38 GB is not worth a second format.

The MXFP4 packing is imported from demo.py rather than copied. Its scale is a
raw E8M0 exponent of amax/6.0 with a mantissa round-up, NOT ceil(log2(...)) as
in the DeepSeek-R1 streamer this is modelled on; the two disagree on exact
powers of two, and the megakernel decodes only the former. --verify re-derives
a sample through the imported function and bit-compares.

Output mirrors the input shard names, with a regenerated index, so the loader
reads it with the ordinary weight_map. An MXFP4 weight becomes two entries:

    <name>.mxfp4_blocks : uint8 [out, K/2]    E2M1 nibbles, elem 2b low
    <name>.mxfp4_scales : uint8 [out, K/32]   E8M0 per-32 exponent

Usage:
  # unit check: first 2 shards, keep the FP8 inputs, verify the packing
  python3 stream_quant_glm5.py --shards 2 --verify --keep-fp8
  # full run, resumable -- re-running skips shards already converted
  python3 stream_quant_glm5.py --all
"""
import argparse
import gc
import json
import os
import queue
import shutil
import sys
import threading
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo import quantize_mxfp4  # noqa: E402  the one true packing

REPO = os.environ.get("GLM5_REPO", "zai-org/GLM-5-FP8")
STAGE = os.environ.get("GLM5_STAGE", "/home/claudeuser/models/glm5-fp8-stage")
OUT = os.environ.get("GLM5_OUT", "/home/claudeuser/models/glm5-mxfp4")
BLK = 128  # weight_block_size = [128, 128]

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Files that are not weights but that the loader and tokenizer need beside them.
SIDECARS = ("config.json", "configuration.json", "tokenizer.json",
            "tokenizer_config.json", "special_tokens_map.json",
            "generation_config.json", "chat_template.jinja", "vocab.json",
            "merges.txt")


def is_routed_expert(name: str) -> bool:
    """True for mlp.experts.<int>.*_proj.weight, false for shared_experts.

    Substring alone is not enough: 'mlp.shared_experts.down_proj.weight' also
    contains 'experts'. The routed stacks are the only ones with an integer
    index, and the only ones the MXFP4 MoE kernel reads.
    """
    parts = name.split(".")
    try:
        i = parts.index("experts")
    except ValueError:
        return False
    return i > 0 and parts[i - 1] == "mlp" and parts[i + 1].isdigit()


def dequant_fp8_block(w_fp8: torch.Tensor, scale_inv: torch.Tensor):
    """Canonical block dequant: W_bf16 = W_fp8 * scale_inv, one fp32 scale per
    128x128 tile. w_fp8 [R,C] float8_e4m3fn, scale_inv [ceil(R/128), ceil(C/128)].

    Done on the GPU -- at 756 GB total this is the whole cost of the run, and a
    CPU float() of a 6144x12288 tile is ~40x slower than the copy.
    """
    R, C = w_fp8.shape
    w = w_fp8.to(DEV, non_blocking=True).float()
    s = scale_inv.to(DEV, non_blocking=True).float()
    s = s.repeat_interleave(BLK, dim=0).repeat_interleave(BLK, dim=1)[:R, :C]
    return (w * s).bfloat16()


def process_shard(shard_name, verify=False, keep_fp8=False):
    path = os.path.join(STAGE, shard_name)
    out_tensors = {}
    n_mxfp4 = n_bf16 = n_plain = 0
    max_err = 0.0

    with safe_open(path, framework="pt", device="cpu") as fh:
        keys = list(fh.keys())
        kset = set(keys)
        for name in keys:
            if name.endswith(".weight_scale_inv"):
                continue  # consumed alongside its .weight
            t = fh.get_tensor(name)
            sc_name = name + "_scale_inv"

            if t.dtype == torch.float8_e4m3fn and sc_name in kset:
                w = dequant_fp8_block(t, fh.get_tensor(sc_name))
                if is_routed_expert(name):
                    blocks, scales = quantize_mxfp4(w)
                    if verify:
                        b2, s2 = quantize_mxfp4(w)
                        assert torch.equal(b2, blocks) and torch.equal(s2, scales), \
                            f"packing is not deterministic for {name}"
                        # dequantized MXFP4 vs the bf16 it came from, as a
                        # sanity bound on the format -- not a tolerance gate.
                        max_err = max(max_err, _mxfp4_err(w, blocks, scales))
                    out_tensors[name + ".mxfp4_blocks"] = blocks.cpu()
                    out_tensors[name + ".mxfp4_scales"] = scales.cpu()
                    n_mxfp4 += 1
                else:
                    out_tensors[name] = w.cpu()
                    n_bf16 += 1
                del w
            else:
                # Norms, router gate, embeddings, lm_head, indexer k_norm. Keep
                # e_score_correction_bias in fp32: it is added to the sigmoid
                # scores before top-8 and a bf16 round would perturb routing at
                # the selection boundary.
                if t.dtype == torch.float32 and not name.endswith(
                        "e_score_correction_bias"):
                    t = t.bfloat16()
                out_tensors[name] = t.contiguous()
                n_plain += 1
            del t
        if DEV == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(OUT, exist_ok=True)
    out_path = os.path.join(OUT, shard_name)
    tmp_path = out_path + ".partial"
    save_file(out_tensors, tmp_path, metadata={"format": "pt"})
    os.replace(tmp_path, out_path)  # so a kill never leaves a half shard

    in_gb = os.path.getsize(path) / 1e9
    out_gb = os.path.getsize(out_path) / 1e9
    names = list(out_tensors.keys())
    del out_tensors
    gc.collect()
    if not keep_fp8:
        os.remove(path)
    return dict(n_mxfp4=n_mxfp4, n_bf16=n_bf16, n_plain=n_plain,
                in_gb=in_gb, out_gb=out_gb, max_err=max_err, names=names)


_E2M1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                      -0., -.5, -1., -1.5, -2., -3., -4., -6.])


def _mxfp4_err(w_bf16, blocks, scales):
    lut = _E2M1.to(blocks.device)
    lo, hi = (blocks & 0x0F).long(), (blocks >> 4).long()
    vals = torch.stack([lut[lo], lut[hi]], dim=-1).reshape(
        *blocks.shape[:-1], blocks.shape[-1] * 2)
    K = vals.shape[-1]
    vals = vals.reshape(*vals.shape[:-1], K // 32, 32)
    sf = torch.where(scales == 0, torch.ones_like(scales, dtype=torch.float32),
                     (scales.to(torch.int32) << 23).view(torch.float32))
    deq = (vals * sf.unsqueeze(-1)).reshape(w_bf16.shape)
    return (deq - w_bf16.float()).abs().max().item()


def downgrade_tokenizer_config(out_dir):
    """Rewrite GLM-5's transformers-5 tokenizer_config for transformers 4.

    Nothing about the tokenizer itself is new -- the vocab and merges live in
    tokenizer.json and load as a PreTrainedTokenizerFast. Only the surrounding
    config is v5-shaped, in three ways:

      tokenizer_class: TokenizersBackend  -> 4.57 raises "does not exist"
      extra_special_tokens: [str, ...]    -> v4 wants a {name: token} dict here
                                             and calls .keys() on it; the plain
                                             list is v4's additional_special_tokens
      backend / is_local / model_specific_special_tokens
                                          -> v5-only, arrive as stray kwargs

    Each fixup is independently guarded so the function is idempotent and
    re-running the converter reproduces the same result.
    """
    p = os.path.join(out_dir, "tokenizer_config.json")
    if not os.path.exists(p):
        return
    cfg = json.load(open(p))
    changed = []
    if cfg.get("tokenizer_class") in (None, "TokenizersBackend"):
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        changed.append("tokenizer_class")
    if isinstance(cfg.get("extra_special_tokens"), list):
        extra = cfg.pop("extra_special_tokens")
        have = cfg.get("additional_special_tokens") or []
        cfg["additional_special_tokens"] = have + [t for t in extra
                                                   if t not in have]
        changed.append("extra_special_tokens")
    for k in ("backend", "is_local", "model_specific_special_tokens"):
        if cfg.pop(k, None) is not None:
            changed.append(k)
    if not changed:
        return
    json.dump(cfg, open(p, "w"), ensure_ascii=False)
    print(f"[stream] tokenizer_config downgraded to v4: {', '.join(changed)}",
          flush=True)


def fetch(shard_name):
    """Download one shard into STAGE if it is not already there."""
    p = os.path.join(STAGE, shard_name)
    if os.path.exists(p):
        return p
    return hf_hub_download(REPO, shard_name, local_dir=STAGE)


def prefetcher(shards, q, stop):
    """Keep one shard downloaded ahead of the converter.

    Download runs at ~595 MB/s (9 s/shard) and conversion takes longer, so a
    single-slot lookahead hides the network entirely without ever holding more
    than two FP8 shards on disk.
    """
    for sh in shards:
        if stop.is_set():
            break
        try:
            fetch(sh)
            q.put((sh, None))
        except Exception as e:  # surface it on the consumer side, in order
            q.put((sh, e))
    q.put((None, None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=0, help="convert first N")
    ap.add_argument("--all", action="store_true", help="convert all 142")
    ap.add_argument("--start", type=int, default=1, help="1-based start index")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--keep-fp8", action="store_true")
    args = ap.parse_args()

    os.makedirs(STAGE, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    idx_path = os.path.join(STAGE, "model.safetensors.index.json")
    if not os.path.exists(idx_path):
        shutil.copy(hf_hub_download(REPO, "model.safetensors.index.json"),
                    idx_path)
    for f in SIDECARS:
        dst = os.path.join(OUT, f)
        if not os.path.exists(dst):
            try:
                shutil.copy(hf_hub_download(REPO, f), dst)
            except Exception:
                pass  # not every repo ships every sidecar
    downgrade_tokenizer_config(OUT)

    idx = json.load(open(idx_path))
    shards = sorted(set(idx["weight_map"].values()))
    sel = shards[args.start - 1:] if args.all else \
        shards[args.start - 1:args.start - 1 + max(args.shards, 1)]

    # Resume: a shard whose output already exists was completed (save_file is
    # rename-atomic above), so skip it without re-downloading.
    todo = [s for s in sel if not os.path.exists(os.path.join(OUT, s))]
    print(f"[stream] {len(todo)}/{len(sel)} shards to convert "
          f"({len(sel) - len(todo)} already done), verify={args.verify}",
          flush=True)

    q, stop = queue.Queue(maxsize=1), threading.Event()
    th = threading.Thread(target=prefetcher, args=(todo, q, stop), daemon=True)
    th.start()

    weight_map = {}
    done = 0
    t_start = time.time()
    try:
        while True:
            sh, err = q.get()
            if sh is None:
                break
            if err is not None:
                raise RuntimeError(f"download failed for {sh}") from err
            t0 = time.time()
            st = process_shard(sh, verify=args.verify, keep_fp8=args.keep_fp8)
            for n in st["names"]:
                weight_map[n] = sh
            done += 1
            dt = time.time() - t0
            avail = shutil.disk_usage(OUT).free / 1e9
            eta = (len(todo) - done) * (time.time() - t_start) / done / 60
            print(f"[{done}/{len(todo)} {sh}] mxfp4={st['n_mxfp4']} "
                  f"bf16={st['n_bf16']} plain={st['n_plain']} "
                  f"{st['in_gb']:.2f}->{st['out_gb']:.2f}GB "
                  f"err={st['max_err']:.2e} {dt:.0f}s "
                  f"avail={avail:.0f}GB eta={eta:.0f}min", flush=True)
    finally:
        stop.set()

    # Rebuild the index over whatever is on disk now, so a resumed run still
    # produces a complete map rather than only this invocation's shards.
    out_idx = os.path.join(OUT, "model.safetensors.index.json")
    if os.path.exists(out_idx):
        prev = json.load(open(out_idx))["weight_map"]
        prev.update(weight_map)
        weight_map = prev
    total = sum(os.path.getsize(os.path.join(OUT, s))
                for s in shards if os.path.exists(os.path.join(OUT, s)))
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
              open(out_idx, "w"))
    print(f"[stream] DONE {done} shards, {total / 1e9:.1f} GB, "
          f"{len(weight_map)} tensors -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
