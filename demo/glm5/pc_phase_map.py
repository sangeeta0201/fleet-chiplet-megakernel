#!/usr/bin/env python3
"""Attribute rocprofv3 PC samples to MEGAKERNEL PHASES, then split by stall
reason.

WHY THIS EXISTS. The GLM megakernel is ONE persistent dispatch, so every
per-kernel view rocprofv3 offers is a single row and says nothing. Attribution
has to be per-ADDRESS, and most of the layer is inlined into worker_kernel /
persistent_kernel, so symbols alone are not enough either.

THE MAP. `align_isa.py` builds production-address -> file:line by aligning the
production disassembly against a `-gline-tables-only` rebuild of the same
test.cu. That rebuild is NOT instruction-identical -- measured on this image
every device function gains 2-9 instructions and scheduler_kernel's register
allocation shifts -- so the map comes from a difflib alignment of the
(opcode, operands) sequence, not from equal addresses. Alignment coverage is
87-100% per symbol; unaligned addresses get no line and fall back to the symbol
bucket rather than being guessed.

TRANSPARENT HEADERS. __clang_hip_math.h, amd_hip_bf16.h and friends are inlined
leaf math; charging samples to them would hide the caller. They are attributed
to the nearest preceding non-transparent line instead.

STALL REASONS come from the STOCHASTIC record only; host_trap has no
stall-reason field. Report the SPLIT, never the absolute sample count: a
profiled build runs slower than production.

Usage:
  pc_phase_map.py --map addr2line.json --isa ISA.txt --pc-csv DIR [--top N]
"""
import argparse
import bisect
import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

HDR = re.compile(r"^([0-9a-f]+) <(.+)>:")
INS = re.compile(r"^\t([a-z][a-z0-9_]*)\s*(.*?)\s*//\s*([0-9A-F]+):")

TRANSPARENT = ("__clang_hip_math.h", "amd_hip_bf16.h", "amd_warp_functions.h",
               "amd_device_functions.h", "amd_hip_fp16.h", "hip_fp8.h",
               "math_functions.h", "amd_math_functions.h", "hip_assert.h",
               "amd_hip_math_constants.h", "amd_hip_runtime.h")

PHASE_BY_FILE = [
    ("merge_splitkv.cuh",                        "split-KV merge"),
    ("gang_mla_decode_mi300.cuh",                "MLA decode core"),
    ("gang_gemv_mxfp8_mi300.cuh",                "GEMV mxfp8 (W_UK/W_UV)"),
    ("gang_gemv_mxfp4_mi300.cuh",                "GEMV mxfp4 (o_proj)"),
    ("gang_gemv_mi300.cuh",                      "GEMV (generic)"),
    ("mla_kv_cache_update_mi300.cuh",            "KV latent write"),
    ("gang_rmsnorm_linear_mxfp8_bias_mi300.cuh", "qkv_a / q_b GEMM"),
    ("gang_moe_linear_mxfp4_mi300.cuh",          "MoE W13/W2 (mxfp4)"),
    ("gang_moe_linear_mxfp8_mi300.cuh",          "MoE W13/W2 (mxfp8)"),
    ("moe_topk_sigmoid_bias_mi300.cuh",          "router TopK"),
    ("gang_oproj_router_fused_mi300.cuh",        "o_proj + router"),
    ("gang_mla_attn_fused_mi300.cuh",            "MLA attn orchestration"),
    ("gang_mla_full_layer_fused_mi300.cuh",      "fused-layer orchestration"),
    ("gang_full_layer_fused_mi300.cuh",          "fused-layer orchestration"),
    ("mpk_atoms.cuh",                            "barrier / atom primitives"),
    ("mpk_comm.cuh",                             "EP comm / peer wait"),
    ("persistent_kernel.cuh",                    "persistent loop + dispatch"),
    ("test.cu",                                  "generated task graph glue"),
]
PHASE_BY_SYM = [
    ("mla_decode_absorbed",                      "MLA decode core"),
    ("gang_moe_w13_linear",                      "MoE W13"),
    ("gang_moe_w2_linear",                       "MoE W2"),
    ("gang_gemv_mxfp8_kernel",                   "GEMV mxfp8 (W_UK/W_UV)"),
    ("gang_gemv_mxfp4_kernel",                   "GEMV mxfp4 (o_proj)"),
    ("gang_rmsnorm_linear_mxfp8_bias_mla_kvupd", "qkv_a + kvupd"),
    ("gang_rmsnorm_linear_mxfp8_bias_kernel",    "qkv_a / q_b GEMM"),
    ("gang_rmsnorm_linear_bias_topk_kernel",     "router GEMM"),
    ("topk_sigmoid_noinline",                    "router TopK"),
    ("latent_to_cache",                          "KV latent write"),
    ("gang_mla_full_layer_fused_kernel",         "fused layer (unaligned)"),
    ("worker_kernel",                            "worker_kernel (unaligned)"),
    ("persistent_kernel",                        "persistent_kernel (unaligned)"),
    ("scheduler_kernel",                         "scheduler_kernel"),
    ("rocshmem",                                 "rocSHMEM (EP collective)"),
    ("__ockl", "runtime printf"),
    ("__assert", "runtime assert"),
]

CLASS = {
    "WAITCNT": "memory/waitcnt",
    "BARRIER_WAIT": "barrier",
    "SLEEP_WAIT": "sleep/poll",
    "ALU_DEPENDENCY": "ALU dependency",
    "NO_INSTRUCTION_AVAILABLE": "I-fetch",
    "ARBITER_NOT_WIN": "arbiter",
    "ARBITER_WIN_EX_STALL": "exec backpressure",
    "INTERNAL_INSTRUCTION": "internal/NOP",
    "OTHER_WAIT": "other wait",
    "NONE": "issued",
}
CLASSES = ["memory/waitcnt", "barrier", "sleep/poll", "ALU dependency",
           "I-fetch", "arbiter", "exec backpressure", "internal/NOP",
           "other wait", "issued"]


def parse_syms(path):
    rows, sym = [], None
    for line in open(path, errors="replace"):
        h = HDR.match(line)
        if h:
            sym = h.group(2)
            continue
        m = INS.match(line)
        if m and sym:
            rows.append((int(m.group(3), 16), sym))
    rows.sort()
    return rows


def phase_of(sym, fname):
    if fname:
        for needle, label in PHASE_BY_FILE:
            if needle in fname:
                return label
    for needle, label in PHASE_BY_SYM:
        if needle in sym:
            return label
    return f"other:{sym[:36]}"


def find_csvs(path):
    if os.path.isfile(path):
        return [path]
    hits = []
    for pat in ("**/*pc_sampl*.csv", "**/*pc_sample*.csv"):
        hits += glob.glob(os.path.join(path, pat), recursive=True)
    return sorted(set(hits))


def col(fieldnames, *cands):
    low = {f.lower().replace(" ", "_"): f for f in fieldnames}
    for c in cands:
        if c in low:
            return low[c]
    for c in cands:
        for k, v in low.items():
            if c in k:
                return v
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--isa", required=True)
    ap.add_argument("--pc-csv", required=True)
    ap.add_argument("--top", type=int, default=22)
    ap.add_argument("--hot-lines", type=int, default=0)
    ap.add_argument("--only-phase")
    args = ap.parse_args()

    raw = json.load(open(args.map))
    # Collapse inlined leaf math onto its caller.
    line_addr, line_val = [], []
    last = None
    for a, s, f, l in raw:
        base = os.path.basename(f)
        if base in TRANSPARENT:
            if last is None:
                continue
            f, l = last
        else:
            last = (f, l)
        line_addr.append(a)
        line_val.append((f, l))

    sym_rows = parse_syms(args.isa)
    sym_addr = [r[0] for r in sym_rows]

    csvs = find_csvs(args.pc_csv)
    if not csvs:
        sys.exit(f"no pc-sampling csv under {args.pc_csv}")

    per_phase = Counter()
    per_phase_reason = defaultdict(Counter)
    per_line = Counter()
    unmapped = total = no_line = 0
    reasons_seen = Counter()
    schema = None

    for path in csvs:
        with open(path, newline="") as fh:
            rd = csv.DictReader(fh)
            fn = rd.fieldnames or []
            schema = schema or fn
            a_col = col(fn, "instruction_address", "pc", "address", "inst_addr",
                        "code_object_offset", "offset")
            r_col = col(fn, "stall_reason", "inst_not_issued_reason", "reason")
            if a_col is None:
                print(f"!! {path}: no address column in {fn}", file=sys.stderr)
                continue
            for rec in rd:
                total += 1
                s = (rec.get(a_col) or "").strip()
                if not s:
                    unmapped += 1
                    continue
                try:
                    a = int(s, 16) if s.lower().startswith("0x") else int(s)
                except ValueError:
                    unmapped += 1
                    continue
                i = bisect.bisect_right(sym_addr, a) - 1
                if i < 0:
                    unmapped += 1
                    continue
                sym = sym_rows[i][1]
                j = bisect.bisect_right(line_addr, a) - 1
                fname = lineno = None
                if j >= 0 and line_addr[j] == a:
                    fname, lineno = line_val[j]
                else:
                    no_line += 1
                ph = phase_of(sym, fname)
                reason = (rec.get(r_col) or "NONE").strip().upper() if r_col \
                    else "NONE"
                reason = reason.replace(
                    "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_", "")
                reasons_seen[reason] += 1
                per_phase[ph] += 1
                per_phase_reason[ph][CLASS.get(reason, reason)] += 1
                if fname and (not args.only_phase or ph == args.only_phase):
                    per_line[(os.path.basename(fname), lineno)] += 1

    print(f"# csv            {[os.path.basename(c) for c in csvs]}")
    print(f"# columns        {schema}")
    print(f"# samples        {total}  unmapped {unmapped}  "
          f"exact-line {total - unmapped - no_line}")
    print(f"# reasons        {dict(reasons_seen.most_common())}")
    print()
    head = f"{'phase':<32}{'samp':>8}{'%tot':>7}  "
    for c in CLASSES:
        head += f"{c.split('/')[0][:6]:>8}"
    print(head)
    print("-" * len(head))
    tot = sum(per_phase.values()) or 1
    for ph, n in per_phase.most_common(args.top):
        line = f"{ph[:31]:<32}{n:>8}{100.0 * n / tot:>6.1f}%  "
        for c in CLASSES:
            v = per_phase_reason[ph][c]
            line += f"{100.0 * v / n:>7.1f}%" if v else f"{'-':>8}"
        print(line)
    print(f"{'TOTAL':<32}{tot:>8}{100.0:>6.1f}%  ")
    agg = Counter()
    for ph in per_phase:
        agg.update(per_phase_reason[ph])
    line = f"{'  (all phases)':<32}{'':>8}{'':>7}  "
    for c in CLASSES:
        v = agg[c]
        line += f"{100.0 * v / tot:>7.1f}%" if v else f"{'-':>8}"
    print(line)

    if args.hot_lines:
        print(f"\nhottest source lines"
              f"{' in ' + args.only_phase if args.only_phase else ''}")
        for (f, l), n in per_line.most_common(args.hot_lines):
            print(f"  {n:>7}  {100.0 * n / tot:>5.2f}%  {f}:{l}")


if __name__ == "__main__":
    main()
