#!/usr/bin/env python3
"""OUTSTANDING-LOAD CENSUS over the real gfx950 code object.

Why this exists when isa_loads_in_flight.py already does: that script reports
`issued` = loads before the FIRST vmcnt-naming s_waitcnt in the body, and calls
issued <= 2 SUSPECT.  That is exactly backwards for a software-pipelined loop,
whose whole shape is

    top:  s_waitcnt vmcnt(0)   <- retire LAST trip's loads
          <consume>
          <issue N loads>      <- for the NEXT trip
          s_cbranch top

which reports issued = 0 and reads as the worst case while actually carrying N
loads across the backedge.  qkv_a's k-loop reads 12 loads / issued 0 there; it
is the pipelined shape, not a drained one.

WHAT THIS REPORTS instead, by simulating the vmcnt queue over one trip:

  carry   loads still outstanding at the backedge -- the depth that is actually
          live while the next trip's dependent work runs.  THIS is the number
          the 3.45 -> 5.45 TB/s curve in test_waves_per_simd_payoff.hip is
          indexed by.
  peak    max outstanding at any point in the body
  drains  count of s_waitcnt vmcnt(0) in the body.  Two or more per trip is the
          shape #95 deleted from the MoE k-loop.
  loads   total loads, ds = ds_read/ds_write in the body

A loop is starved if carry <= 2.  A loop with carry >= 8 is on the flat part of
the curve and there is nothing to take.

Usage:
  llvm-objdump --offloading demo/glm5/permanent_output_dir_rank0/*.so
  llvm-objdump -d --mcpu=gfx950 <bundle> > /tmp/isa.txt
  python3 demo/glm5/isa_outstanding.py /tmp/isa.txt

Traps are the same three isa_accounting.py documents: unsigned-decimal simm16
branch operands in dwords, the symbol appearing twice in the bundle (key by
first occurrence), and tile kernels being __noinline__ so the persistent-kernel
body is the dense prologue, not a tile.
"""
import re
import sys

HDR = re.compile(r"^([0-9a-f]+) <(.+)>:")
INS = re.compile(r"^\t([a-z][a-z0-9_]*)\s*(.*?)\s*//\s*([0-9A-F]+):")
LD = ("buffer_load", "global_load", "flat_load", "scratch_load")
VMCNT = re.compile(r"vmcnt\((\d+)\)")

PHASES = [
    ("W13 tile", "gang_moe_w13_linear_mxfp8_kernel"),
    ("W2 tile", "gang_moe_w2_linear_mxfp8_kernel"),
    ("qkv_a / q_b", "gang_rmsnorm_linear_mxfp8_bias_kernel"),
    ("qkv_a+kvupd", "gang_rmsnorm_linear_mxfp8_bias_mla_kvupd_kernel"),
    ("router GEMM", "gang_rmsnorm_linear_bias_topk_kernel"),
    ("GEMV (UK/UV/o)", "gang_gemv_mxfp8_kernel"),
    ("GEMV mxfp4", "gang_gemv_mxfp4_kernel"),
    ("MLA decode", "mla_decode_absorbed"),
    ("KV latent write", "latent_to_cache"),
    ("dense MLP", "gang_rmsnorm_linear_mxfp8_swiglu"),
    ("LM head", "gang_linear_mxfp8_argmax"),
]


def parse(path):
    out, cur = [], None
    for line in open(path):
        h = HDR.match(line)
        if h:
            cur = (h.group(2), [])
            out.append(cur)
            continue
        if cur is None:
            continue
        m = INS.match(line)
        if m:
            cur[1].append((int(m.group(3), 16), m.group(1), m.group(2)))
    return out


def s16(v):
    return v - 0x10000 if v & 0x8000 else v


def loops(body):
    idx = {a: i for i, (a, _, _) in enumerate(body)}
    found = []
    for a, op, ops in body:
        if not op.startswith("s_cbranch") and op != "s_branch":
            continue
        m = re.match(r"^(\d+)", ops.strip())
        if not m:
            continue
        tgt = a + 4 + 4 * s16(int(m.group(1)))
        if tgt < a and tgt in idx:
            found.append((tgt, a))
    found.sort(key=lambda t: t[1] - t[0])
    return found


def simulate(seg):
    """Run the vmcnt queue over the body twice so the carry-in settles."""
    carry = 0
    peak = 0
    for _ in range(2):
        out = carry
        peak = 0
        for _a, op, ops in seg:
            if op.startswith(LD):
                out += 1
                peak = max(peak, out)
            elif op == "s_waitcnt":
                m = VMCNT.search(ops)
                if m:
                    out = min(out, int(m.group(1)))
        carry = out
    drains = sum(1 for _a, op, ops in seg
                 if op == "s_waitcnt" and VMCNT.search(ops)
                 and int(VMCNT.search(ops).group(1)) == 0)
    nload = sum(1 for _a, op, _o in seg if op.startswith(LD))
    nds = sum(1 for _a, op, _o in seg if op.startswith("ds_"))
    return carry, peak, drains, nload, nds


def census(body, top=3):
    idx = {a: i for i, (a, _, _) in enumerate(body)}
    rows, covered = [], []
    for tgt, br in loops(body):
        if any(tgt >= t and br <= b for t, b in covered):
            continue
        covered.append((tgt, br))
        seg = body[idx[tgt]:idx[br] + 1]
        carry, peak, drains, nload, nds = simulate(seg)
        if nload == 0:
            continue
        rows.append((len(seg), carry, peak, drains, nload, nds, tgt, br))
    rows.sort(key=lambda r: -r[4])
    return rows[:top]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/isa.txt"
    syms = parse(path)
    print(f"{'phase':<17} {'insns':>6} {'loads':>6} {'carry':>6} {'peak':>5} "
          f"{'drains':>7} {'ds':>5}  verdict")
    print("-" * 78)
    for label, needle in PHASES:
        hit = next((s for s in syms if needle in s[0]), None)
        if hit is None:
            print(f"{label:<17} -- symbol not in image")
            continue
        rows = census(hit[1])
        if not rows:
            print(f"{label:<17} -- no load-carrying loop")
            continue
        for n, (span, carry, peak, drains, nload, nds, tgt, br) in enumerate(rows):
            v = ("STARVED" if carry <= 2 else
                 "thin" if carry < 8 else "flat-part")
            if drains >= 2:
                v += f" {drains}xDRAIN"
            print(f"{label if n == 0 else '':<17} {span:>6} {nload:>6} "
                  f"{carry:>6} {peak:>5} {drains:>7} {nds:>5}  {v}"
                  f"   [{tgt:#x}..{br:#x}]")


if __name__ == "__main__":
    main()
