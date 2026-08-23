#!/usr/bin/env python3
"""Count TileRT's device-wide sync points per GLM-5 MoE decode layer.

THE POINT. The board's premise was "TileRT runs the same 9 ops with TWO fused
all-reduces; we run TEN rendezvous." That compares two different quantities.
TileRT is a per-op kernel runtime, not a megakernel: every registered op lowers
to one or more `torch.ops.tilert.*_op(...)` calls, and a kernel-launch boundary
IS a device-wide dependency barrier -- the exact thing a persistent megakernel
exists to remove. TileRT's "two" is its CROSS-RANK count, and ours is also two
(glm-layer-pays-two-cross-rank-rendezvous).

Reads ~/TileRT READ-ONLY. Verifies the launch sites still exist rather than
trusting the table in RENDEZVOUS_EDGE_TABLE.md.

  python3 demo/glm5/count_tilert_launches.py
"""
import os
import re
import subprocess
import sys

TILERT = os.path.expanduser("~/TileRT")

# (launch, the op module that issues it, cross-rank?)
# One row per torch.ops.tilert.*_op call on the PureMlaV2 (GPU 1-7) MoE-layer
# path: PureMlaV2's 6 registered ops + Moe's 3, expanded to launches.
LAUNCHES = [
    ("rmsnorm_op_quant",             "rmsnorm_quant",          "rmsnorm_quant_op",              False),
    ("projx_wqkva",                  "projx_wqkva",            "projx_wqkva_op",                False),
    ("rmsnorm_projq_wqb",            "rmsnorm_projq_wqb",      "rmsnorm_proj_qb_op",            False),
    ("rmsnorm_kv",                   "rmsnorm_kv",             "rmsnorm_kv_op",                 False),
    ("projq_wqb",                    "projq_wqb",              "projq_wqb_op",                  False),
    ("flash_sparse_mla",             "flash_sparse_mla",       "flash_sparse_mla_op",           False),
    ("projo_wkvb",                   "projo_wkvb",             "projo_wkvb_op",                 False),
    ("unproj_o_allreduce",           "unproj_o_allreduce",     "unproj_o_allreduce_op",         True),
    ("rmsnorm_expert_proj",          "rmsnorm_expert_proj",    "rmsnorm_expert_proj_op",        False),
    ("expert_sel_up_gate_silu",      "expert_sel_up_gate_silu","expert_select_up_gate_silu_op", False),
    ("expert_down_allreduce",        "expert_down_allreduce",  "expert_down_allreduce_op",      True),
]

# Ours, from price_xcd_narrowing.py's BARRIERS (identical to
# makespan_predictor.py's). Cross-rank per glm-layer-pays-two-cross-rank-rendezvous.
OURS = [
    ("entry_bar",    9, "full_layer:711",  False),
    ("qkv_barrier", 17, "attn:743",        False),
    ("qb_barrier",  19, "attn:1183",       True),   # the QB_TP peer wait
    ("decode_barrier", 21, "attn:1411",    False),
    ("attn_release", 23, "full_layer:1782",False),
    ("rel_tree",    25, "full_layer:2025", False),
    ("wuv_barrier", 28, "oproj:552",       False),
    ("hier_barrier",30, "oproj:925",       False),
    ("routing poll",32, "oproj:1174",      True),   # the EP fold / routing collective
    ("w13_barrier",  6, "oproj:1595",      False),
]


def find_launch(sym):
    """Return the first file:line where `torch.ops.tilert.<sym>(` appears."""
    try:
        out = subprocess.run(
            ["grep", "-rn", "--include=*.py", "torch.ops.tilert." + sym, TILERT],
            capture_output=True, text=True, timeout=60).stdout
    except Exception as exc:                      # noqa: BLE001
        return "grep failed: %s" % exc
    for line in out.splitlines():
        path, lineno = line.split(":", 2)[:2]
        return "%s:%s" % (os.path.relpath(path, TILERT), lineno)
    return None


def main():
    if not os.path.isdir(TILERT):
        print("~/TileRT not present -- cannot verify. The table stands on the "
              "file:line citations in RENDEZVOUS_EDGE_TABLE.md §1.")
        return 2

    print("TileRT launches per GLM-5 MoE decode layer, PureMlaV2 device (GPU 1-7)")
    print("-" * 78)
    n_dev = n_xrank = missing = 0
    for i, (name, _mod, sym, xrank) in enumerate(LAUNCHES, 1):
        site = find_launch(sym)
        if site is None:
            missing += 1
            site = "NOT FOUND -- tree changed, re-derive"
        n_dev += 1
        n_xrank += bool(xrank)
        print("%2d  %-30s %-34s %s" % (i, name, site, "CROSS-RANK" if xrank else ""))

    o_dev = len(OURS)
    o_xrank = sum(1 for r in OURS if r[3])
    print()
    print("Ours, from price_xcd_narrowing.py BARRIERS")
    print("-" * 78)
    for i, (name, stamp, site, xrank) in enumerate(OURS, 1):
        print("%2d  %-30s stamp %-3d %-22s %s"
              % (i, name, stamp, site, "CROSS-RANK" if xrank else ""))

    print()
    print("=" * 78)
    print("%-34s %8s %8s" % ("", "ours", "TileRT"))
    print("%-34s %8d %8d" % ("device-wide sync points / layer", o_dev, n_dev))
    print("%-34s %8d %8d" % ("cross-rank collectives / layer", o_xrank, n_xrank))
    print("=" * 78)
    if missing:
        print("WARNING: %d launch symbols not found; verify before quoting." % missing)
        return 1
    print()
    print("VERDICT. The '2' in the original diff is TileRT's CROSS-RANK count and")
    print("we match it exactly (%d vs %d). On device-wide sync points we are already"
          % (o_xrank, n_xrank))
    print("AHEAD (%d vs %d). The rendezvous COUNT is not where the 5.3x lives, so"
          % (o_dev, n_dev))
    print("there is no re-sharding of ours that can find it there.")
    print()
    print("Measured this board: bench_repeat.sh v7base 3 -> n=3 mean 10.571,")
    print("min 10.535, max 10.634 ms/iter, text checked (Paris), no flags.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
