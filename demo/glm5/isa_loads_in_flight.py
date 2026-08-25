#!/usr/bin/env python3
"""LOADS-IN-FLIGHT CENSUS over the real gfx950 code object.

THE QUESTION.  tests/standalone/test_waves_per_simd_payoff.hip measured, on
this part, that a streaming loop's bandwidth is set by how many global loads
it keeps outstanding, NOT by occupancy:

    unroll  blocks/CU   GB/s
    1       1           3446
    1       2           5294
    4       1           5334
    8       1           5446
    8       2           5224      (negative -- the second block hurts)

So a k-loop that drains to vmcnt(0) every trip runs at 3.45 TB/s where an
unrolled one runs at 5.45 -- a 1.58x on that tile's bytes, correct-output, no
restructure.  memory: glm-occupancy-ladder-is-dead-unroll-instead handed this
over as the one cheap unanswered question.  This answers it statically.

WHAT IT REPORTS, per hot loop (a backward s_cbranch and the block it closes):

  issued   global/buffer loads issued in the body before the FIRST s_waitcnt
           that names vmcnt -- i.e. the depth the compiler actually achieved
  minvm    the smallest vmcnt(N) argument anywhere in the body.  vmcnt(0) is
           a FULL DRAIN: every outstanding load must land before the loop can
           continue, so steady-state outstanding count collapses to `issued`
           for one burst and then to zero.
  loads    total loads in the body
  ds       ds_read/ds_write in the body (an LDS round trip can serialise a
           loop that looks unrolled in global loads)

READING IT.  A loop is FINE if issued >= 4 AND minvm > 0, or if issued >= 4
and the vmcnt(0) sits at the very end of a long body.  A loop is SUSPECT if
issued <= 2, because then at most two loads are ever in flight and the loop is
at the 3.45 TB/s point on the curve above.

TRAPS (inherited from isa_accounting.py, all three cost a wrong answer once):
  1. gfx9 s_cbranch operands are UNSIGNED DECIMAL simm16 in DWORDS:
         target = addr + 4 + 4 * signed16(imm)
     Looking for a hex target finds nothing and reports every loop unrolled.
  2. A mangled name appears TWICE in the bundle.  Key by OCCURRENCE.
  3. Tile kernels are __noinline__, reached via s_swappc_b64.  Counting inside
     persistent_kernel/worker_kernel measures the dense prologue, not a tile.

Usage:  python3 demo/glm5/isa_loads_in_flight.py [disassembly.txt]
Offline; no GPU run.
"""
import re
import sys

HDR = re.compile(r"^([0-9a-f]+) <(.+)>:")
INS = re.compile(r"^\t([a-z][a-z0-9_]*)\s*(.*?)\s*//\s*([0-9A-F]+):")
LD = ("buffer_load", "global_load", "flat_load", "scratch_load")
VMCNT = re.compile(r"vmcnt\((\d+)\)")

# The phases the board ranks.  Substring match against the demangled-ish
# mangled name; the first occurrence of each is enough (trap 2: the bundle
# holds two copies of the whole device image).
PHASES = [
    ("W13 tile",        "gang_moe_w13_linear_mxfp8_kernel"),
    ("W2 tile",         "gang_moe_w2_linear_mxfp8_kernel"),
    ("qkv_a / q_b",     "gang_rmsnorm_linear_mxfp8_bias_kernel"),
    ("qkv_a+kvupd",     "gang_rmsnorm_linear_mxfp8_bias_mla_kvupd_kernel"),
    ("router GEMM",     "gang_rmsnorm_linear_bias_topk_kernel"),
    ("GEMV (UK/UV/o)",  "gang_gemv_mxfp8_kernel"),
    ("GEMV mxfp4",      "gang_gemv_mxfp4_kernel"),
    ("MLA decode",      "mla_decode_absorbed"),
    ("KV latent write", "latent_to_cache"),
]


def parse(path):
    """-> list of (name, [(addr, op, operands), ...]) in file order."""
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
    """Backward branches -> (target_addr, branch_addr) innermost-first."""
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
    # innermost = shortest span, and drop loops fully containing a smaller one
    found.sort(key=lambda t: t[1] - t[0])
    return found


def census(name, body, top=3):
    rows = []
    for tgt, br in loops(body):
        seg = [x for x in body if tgt <= x[0] <= br]
        issued, seen_wait = 0, False
        nload = nds = 0
        vms = []
        for _, op, ops in seg:
            if op.startswith(LD):
                nload += 1
                if not seen_wait:
                    issued += 1
            elif op.startswith("ds_"):
                nds += 1
            elif op == "s_waitcnt":
                m = VMCNT.search(ops)
                if m:
                    vms.append(int(m.group(1)))
                    seen_wait = True
        if nload == 0:
            continue
        rows.append((len(seg), issued, min(vms) if vms else None,
                     nload, nds, tgt, br))
    rows.sort(key=lambda r: -r[3])       # rank by loads in the body
    return rows[:top]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/isa_head.txt"
    syms = parse(path)
    print(f"{'phase':<17} {'insns':>6} {'loads':>6} {'issued':>7} "
          f"{'minvm':>6} {'ds':>5}  verdict")
    print("-" * 72)
    for label, needle in PHASES:
        hit = next((s for s in syms if needle in s[0]), None)
        if hit is None:
            print(f"{label:<17} -- symbol not in image")
            continue
        for n, (span, issued, minvm, nload, nds, tgt, br) in enumerate(
                census(label, hit[1])):
            v = ("DRAIN" if minvm == 0 else "ok") if minvm is not None else "-"
            if issued <= 2:
                v += " SUSPECT"
            print(f"{label if n == 0 else '':<17} {span:>6} {nload:>6} "
                  f"{issued:>7} {str(minvm):>6} {nds:>5}  {v}"
                  f"   [{tgt:#x}..{br:#x}]")


if __name__ == "__main__":
    main()
