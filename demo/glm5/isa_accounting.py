#!/usr/bin/env python3
"""ITEM 2: WHERE THE CYCLES GO IN THE NON-MoE TILES -- ISA-level, static.

THE HOLE THIS FILLS.  CLOSING_LEDGER.md prices every phase by BYTES:

    non-MoE width-corrected BYTE floor       8.55 us/layer
    non-MoE MEASURED tile time              55.9  us/layer      = 6.5x
    => 47.3 us/layer x 76 = 3.55 ms, MORE than the 2.94 ms attributed to all
       ten rendezvous, and absent from the ledger's accounting.

MoE runs at 1.75x its byte floor -- that part was priced fairly.  The
attention side was priced by bytes only.  A roofline cannot see a region that
issues no loads, so the ledger could not have found this at any width or any
bandwidth.

WHAT THIS IS.  A static accounting off the REAL gfx950 code object the
benchmark ran, cross-checked against the per-region timings already measured
in the kernel source.  Not a simulation and not a hardware-counter profile.

WHAT IT CANNOT DO.  It cannot see stalls.  Every "issue" number is exact;
every "stall" number is a residual against a measured time and inherits that
measurement's error.  Where a residual is quoted it is labelled.

Usage:  python3 demo/glm5/isa_accounting.py [disassembly.txt]
Offline; no GPU run.

    llvm-objdump --offloading demo/glm5/permanent_output_dir_rank0/*.so
    llvm-objdump -d --mcpu=gfx950 <bundle> > /tmp/isa.txt

------------------------------------------------------------------------------
THREE ISA-READING TRAPS, EACH OF WHICH PRODUCED A WRONG ANSWER FIRST
------------------------------------------------------------------------------
1. BRANCH TARGETS ARE NOT HEX ADDRESSES.  llvm-objdump prints the gfx9
   s_cbranch operand as an UNSIGNED DECIMAL simm16 in DWORDS.
       target = addr + 4 + 4 * signed16(imm)
   A first pass looked for `0x...` targets, found none, and reported "no
   backward branch -- fully unrolled" for every kernel in the binary.  That is
   false for most of them and it hid the two loops that matter in qkv_a.

2. A SYMBOL NAME IS NOT A SYMBOL.  Several mangled names appear TWICE in the
   bundle.  Keying a dict by name concatenates both bodies -- qkv_a reads 2910
   instructions instead of 1455, and every derived address is garbage.  Key by
   OCCURRENCE.

3. "NO MFMA IN THE MEGAKERNEL" IS AN ARTIFACT OF NOT FOLLOWING CALLS.
   worker_kernel / persistent_kernel contain zero
   v_mfma_scale_f32_16x16x128_f8f6f4.  This does NOT mean the megakernel runs a
   VALU GEMV -- the tile kernels are __noinline__ and are reached through
   s_swappc_b64.  Resolve the calls (s_getpc_b64 + s_add_u32 + s_swappc_b64)
   before concluding anything about what the megakernel does.  An earlier draft
   of this file concluded from the inlined body alone that the scaled fp8 MFMA
   was unused and that 80% of the hot loop was dequant bit-manipulation.  Both
   claims are wrong: the loops it measured were in the DENSE PROLOGUE path,
   and all 54 of worker_kernel's s_swappc_b64 sites resolve to named tile
   kernels, four of which do carry the scaled MFMA.
"""
import collections
import os
import re
import subprocess
import sys

SO = ("demo/glm5/permanent_output_dir_rank0/"
      "test.cpython-38-x86_64-linux-gnu.so")
BUNDLE = SO + ".0.hipv4-amdgcn-amd-amdhsa--gfx950:xnack-"
OBJDUMP = "/opt/rocm/llvm/bin/llvm-objdump"
ISA = "/tmp/isa.txt"

CLOCK_GHZ = 2.4
VALU_CYC = 4       # wave64 op on a SIMD16
MFMA_CYC = 32      # v_mfma_*_16x16x128_f8f6f4

HDR = re.compile(r"^([0-9a-f]+) <(.+)>:")
INS = re.compile(r"^\t([a-z][a-z0-9_]*)\s*(.*?)\s*//\s*([0-9A-F]+):")
LD = ("buffer_load", "global_load", "flat_load", "scratch_load")
VM = LD + ("buffer_store", "global_store", "flat_store", "scratch_store",
           "buffer_atomic", "global_atomic", "tbuffer_", "image_")

# ---------------------------------------------------------------------------
# The leaves of the fused-layer call graph, i.e. the 76-layer decode phases.
# us/layer are the board / SP numbers, quoted -- not derived here.
# ---------------------------------------------------------------------------
LEAF = [
    ("qkv_a",        0x030dc8, 14.8,  "SP4[0]; source comment 13.246 us of tile"),
    ("kvupd inner",  0x0333f0, None,  "inside gang_..._mla_kvupd_kernel"),
    ("latent2cache", 0x032dcc, None,  "inside gang_..._mla_kvupd_kernel"),
    ("W_UK/W_UV",    0x034224, 2.0,   "board S28->S29 share, per stage"),
    ("merge",        0x0361fc, 4.0,   "board S21->S23 share"),
    ("o_proj",       0x036eac, 6.7,   "board S29->S30"),
    ("router",       0x0392a0, 12.9,  "S31->S32, 54% poll / 46% work"),
    ("topk_sigmoid", 0x037d20, None,  "inside router"),
    ("mla_decode",   0x034894, 18.3,  "board S19->S21, 16.6 of it spin"),
    ("MoE W13",      0x03a39c, 13.06, "SP3[4]"),
    ("MoE W2",       0x03aeb8, 13.05, "SP3[6]"),
]

# qkv_a's tile, MEASURED, from the MPK_SUBPHASE_TIMING block in
# gang_rmsnorm_linear_mxfp8_bias_mi300.cuh (divided by [1][5] = 1828800 tiles).
# Quoted, with this file's ISA evidence attached to each row.
QKVA = [
    ("resolve + EP fold + RMSNorm rcp + LDS stage", 6119, 46.2,
     "straight line 0x30dc8-0x32014: 55 loads, MLP 18.3, 632 VALU",
     "L2 LATENCY -- 96 KB/tile at 15.7 GB/s per CU of ~52 available"),
    ("depth-4 weight prefetch fill", 841, 6.3,
     "0x327f0-0x32884: 8 hoisted loads, 4 waitcnt",
     "VMEM issue + first-use latency"),
    ("FP8 quantizer", 851, 6.4,
     "LOOP 0x322b8..0x327f0 x~2.5: 243 insn, 206 VALU, ZERO VMEM, 17 LDS",
     "VALU ISSUE -- 824 issue cycles/trip, no byte floor at all"),
    ("MFMA K-loop + epilogue", 5435, 41.0,
     "LOOP 0x32884..0x32bcc x12 (MFMA_ITERS 48 / depth 4): "
     "149 insn, 94 VALU, 4 MFMA, 12 VMEM",
     "35% VALU issue, 12% MFMA busy, ~53% vmcnt stall"),
]


def cl(op):
    if op.startswith(("v_mfma", "v_smfmac")):
        return "MFMA"
    if op.startswith("v_"):
        return "VALU"
    if op.startswith("ds_"):
        return "LDS"
    if op.startswith(VM):
        return "VMEM"
    if op.startswith(("s_load", "s_buffer_load")):
        return "SMEM"
    if op.startswith("s_waitcnt"):
        return "WAIT"
    if op.startswith("s_barrier"):
        return "BAR"
    if op.startswith(("s_branch", "s_cbranch")):
        return "BR"
    if op.startswith("s_"):
        return "SALU"
    return "OTH"


def ensure(path):
    if os.path.exists(path) and os.path.getsize(path) > (1 << 20):
        return path
    if not os.path.exists(BUNDLE):
        subprocess.run([OBJDUMP, "--offloading", SO], check=True,
                       capture_output=True)
    with open(path, "w") as f:
        subprocess.run([OBJDUMP, "-d", "--mcpu=gfx950", BUNDLE], check=True,
                       stdout=f)
    return path


def load(path):
    """-> list of (start_addr, mangled_name, insns), ONE ENTRY PER OCCURRENCE."""
    occ, cur = [], None
    for ln in open(path, errors="ignore"):
        m = HDR.match(ln)
        if m:
            cur = (int(m.group(1), 16), m.group(2), [])
            occ.append(cur)
            continue
        m = INS.match(ln)
        if m and cur:
            cur[2].append((int(m.group(3), 16), m.group(1), m.group(2)))
    occ = [o for o in occ if o[2]]
    occ.sort()
    return occ


def callees(insns):
    """Resolve s_getpc_b64 / s_add_u32 / s_swappc_b64 PC-relative call chains."""
    pend, tg = {}, collections.Counter()
    for a, m, ops in insns:
        if m == "s_getpc_b64":
            r = re.match(r"s\[(\d+):(\d+)\]", ops.strip())
            if r:
                pend[r.group(1)] = (a + 4, None)
        elif m == "s_add_u32":
            r = re.match(r"s(\d+),\s*s(\d+),\s*(0x[0-9a-f]+|-?\d+)", ops.strip())
            if r and r.group(1) == r.group(2) and r.group(1) in pend:
                b, _ = pend[r.group(1)]
                pend[r.group(1)] = (b, (b + int(r.group(3), 0)) & 0xffffffff)
        elif m == "s_swappc_b64":
            r = re.search(r"s\[(\d+):(\d+)\]\s*$", ops.strip())
            if r and r.group(1) in pend and pend[r.group(1)][1] is not None:
                tg[pend[r.group(1)][1]] += 1
    return tg


def backedges(insns):
    """simm16-in-dwords, printed unsigned decimal.  See trap 1."""
    out = []
    for a, op, ops in insns:
        if not op.startswith(("s_branch", "s_cbranch")):
            continue
        m = re.match(r"^(-?\d+)", ops.strip())
        if not m:
            continue
        s = int(m.group(1))
        if s > 32767:
            s -= 65536
        t = a + 4 + 4 * s
        if t < a:
            out.append((t, a, op))
    return out


def loops(insns):
    d = {}
    for t, a, _op in backedges(insns):
        body = [x for x in insns if t <= x[0] <= a]
        if len(body) < 6:
            continue
        if t not in d or len(body) > len(d[t][1]):
            d[t] = (a, body)
    return [(t, d[t][0], d[t][1]) for t in sorted(d)]


def latency_scan(insns):
    """Exposed-latency structure.

    full   = s_waitcnt vmcnt(0) that actually drains >=1 outstanding load, i.e.
             one full memory latency exposed with nothing to hide it.
    serial = longest run of (single load -> vmcnt(0)) pairs, i.e. a dependent
             POINTER CHASE.  n links cost n x latency, back to back.
    mlp    = mean loads retired per waitcnt = memory-level parallelism.
             MI355X needs >= 8 in flight to reach HBM bandwidth
             (mi355x-memory-hierarchy-bandwidth).
    """
    out = full = part = chain = maxchain = since = 0
    drained = []
    for _a, op, ops in insns:
        if op.startswith(LD):
            out += 1
            since += 1
        elif op.startswith("s_waitcnt") and "vmcnt" in ops:
            m = re.search(r"vmcnt\((\d+)\)", ops)
            n = int(m.group(1)) if m else 0
            if out > n:
                drained.append(out - n)
                if n == 0:
                    full += 1
                    if since == 1:
                        chain += 1
                        maxchain = max(maxchain, chain)
                    else:
                        chain = 0
                else:
                    part += 1
                    chain = 0
                out, since = n, 0
    return dict(loads=sum(1 for _a, m, _o in insns if m.startswith(LD)),
                full=full, part=part, serial=maxchain,
                mlp=(sum(drained) / len(drained) if drained else 0.0))


def short(n):
    n = re.sub(r"^_ZN6kernel\d+", "", n)
    n = re.sub(r"^_Z\d+", "", n)
    return n.split("I")[0] if n[:1].islower() else n


# ---------------------------------------------------------------------------


def call_graph(occ, byaddr):
    print("=" * 100)
    print("1. THE DEVICE CALL GRAPH -- the tile kernels are __noinline__ CALLEES")
    print("=" * 100)
    seen = set()

    def walk(a, d=0):
        o = byaddr.get(a)
        if o is None or "ockl" in o[1] or "assert" in o[1]:
            return
        mf = sum(1 for _x, m, _p in o[2] if m.startswith("v_mfma"))
        sc = sum(1 for _x, m, _p in o[2] if m.startswith("v_mfma_scale"))
        print(f"  {'  ' * d}{a:#08x} {len(o[2]):>5}i  mfma={mf:<4}"
              f"{'scaled-fp8' if sc else '':<11} {short(o[1])[:56]}")
        if a in seen or d > 3:
            return
        seen.add(a)
        for t, _n in sorted(callees(o[2]).items(), key=lambda x: -x[1]):
            walk(t, d + 1)

    root = [o for o in occ if o[1].startswith("_Z13worker_kernel")]
    if root:
        walk(root[0][0])
    print("""
  READ THIS BEFORE ANY CLAIM ABOUT WHAT THE MEGAKERNEL DOES.  The two branches
  under worker_kernel are the 76-layer FUSED path
  (gang_mla_full_layer_fused_kernel_mi300, which calls all ten phase kernels)
  and the 3-layer UNFUSED DENSE PROLOGUE
  (glm-dense-prologue-is-the-unfused-attention-path).  Only the first is the
  decode loop.

  The scaled fp8 MFMA IS used: qkv_a, q_b, MoE W13 and MoE W2 each carry 4
  v_mfma_scale_f32_16x16x128_f8f6f4 per k-loop trip.  W_UK, W_UV, merge and
  o_proj are gang_gemv_mxfp8/mxfp4_kernel and carry NONE -- they dequantize on
  the VALU and accumulate with v_dot2c_f32_bf16.  That is correct for M=1 (an
  MFMA tile would waste 15/16 of the matrix core) and it is why the
  megakernel's own inlined body shows only bf16 16x16x16, which are the MLA
  decode's.  This CONFIRMS rather than contradicts
  glm-gemm-isa-matches-gpt-oss-and-is-ahead.
""")


def leaves(byaddr):
    print("=" * 100)
    print("2. PER-LEAF STATIC COMPOSITION AND LOOP STRUCTURE")
    print("=" * 100)
    print(f"  {'kernel':<14}{'us/lyr':>7}{'insns':>7}{'loops':>6}{'MFMA':>6}"
          f"{'VALU':>6}{'VMEM':>6}{'LDS':>5}{'SALU':>6}{'WAIT':>5}"
          f"{'VALU%':>7}{'VALUcyc':>9}")
    for nm, a, us, _src in LEAF:
        o = byaddr.get(a)
        if not o:
            continue
        ins = o[2]
        h = collections.Counter(cl(m) for _x, m, _p in ins)
        print(f"  {nm:<14}{(us if us else '-'):>7}{len(ins):>7}"
              f"{len(loops(ins)):>6}{h['MFMA']:>6}{h['VALU']:>6}{h['VMEM']:>6}"
              f"{h['LDS']:>5}{h['SALU']:>6}{h['WAIT']:>5}"
              f"{100 * h['VALU'] / len(ins):>6.0f}%{h['VALU'] * VALU_CYC:>9}")
    print()
    print("  LOOP BODIES (trip counts from the template args; qkv_a below):")
    for nm, a, _us, _src in LEAF:
        o = byaddr.get(a)
        if not o:
            continue
        lp = loops(o[2])
        if not lp:
            print(f"    {nm:<14} FULLY UNROLLED -- no backward branch")
            continue
        for t, e, body in lp:
            h = collections.Counter(cl(m) for _x, m, _p in body)
            print(f"    {nm:<14}[{t:#08x}..{e:#08x}] n={len(body):>4} "
                  f"VALU={h['VALU']:>4} MFMA={h['MFMA']:>2} VMEM={h['VMEM']:>3} "
                  f"LDS={h['LDS']:>3} -> {h['VALU'] * VALU_CYC:>5} VALU cyc"
                  f" + {h['MFMA'] * MFMA_CYC:>4} MFMA cyc")
    print()


def qkva():
    print("=" * 100)
    print("3. qkv_a's TILE -- THE ACCOUNTING ITEM 2 ASKED FOR")
    print("=" * 100)
    tot = sum(r[1] for r in QKVA)
    print("  measured tile 13246 ns; phase 32.40 us/layer = 14.83 tile"
          " + 17.57 EP peer wait\n")
    print(f"  {'region':<42}{'ns':>7}{'%':>7}  bound by")
    for lab, ns, pc, isa, bound in QKVA:
        print(f"  {lab:<42}{ns:>7}{pc:>6.1f}%  {bound}")
        print(f"  {'':<42}{'':>7}{'':>7}  ISA: {isa}")
    print(f"  {'TOTAL':<42}{tot:>7}\n")
    print("""  ROLLED UP INTO THE CATEGORIES ITEM 2 NAMED, as % of tile cycles:

      VALU issue        ~21%    quantizer 851 ns (100% VALU, 0 VMEM)
                                + k-loop VALU 12 x 94 x 4 cyc = 1880 ns
      MFMA issue         ~5%    12 x 4 x 32 cyc = 640 ns; the matrix core is
                                IDLE 88% of the tile
      vmcnt wait        ~68%    6119 resolve (L2 latency) + 841 fill
                                + ~2900 k-loop residual
      lgkmcnt / LDS      ~5%    61 ds ops, all in the quantizer and k-loop
      s_barrier          ~0%    6 static, none inside a loop
      -----------------------------------------------------------------------
  THE GUIDE'S HYPOTHESIS IS HALF RIGHT, AND THE WRONG HALF IS THE EXPENSIVE
  ONE.  The quant prologue IS a pure-VALU region with no byte floor -- the loop
  at 0x322b8 has 206 VALU and LITERALLY ZERO memory instructions, so no
  roofline at any width could ever have priced it.  Confirmed at ISA level.

  But it is 6.4% of the tile, not 41%.  glm-qkva-tile-is-41pct-redundant-
  prologue is counting the RESOLVE + EP FOLD + RMSNorm + LDS stage, which is
  46.2% -- and that region issues 55 loads and is L2-LATENCY-bound, at
  15.7 GB/s per CU against ~52 GB/s of per-CU L2 share.  It reads the same
  96 KB in all 186 tiles.

  So the 3.55 ms above the byte floor is MEMORY LATENCY, not VALU issue and
  not bandwidth.  "If the answer is VALU-bound, the fix is fewer VALU ops per
  tile" does not follow from the data: fewer VALU ops buys at most the 21%.
""")


def latency(byaddr):
    print("=" * 100)
    print("4. EXPOSED-LATENCY STRUCTURE -- the actual shape of the 68%")
    print("=" * 100)
    print(f"  {'kernel':<14}{'us/lyr':>7}{'loads':>7}{'vmcnt(0)':>10}"
          f"{'partial':>9}{'serial':>8}{'MLP':>7}")
    print(f"  {'':<14}{'':>7}{'':>7}{'full drain':>10}{'drain':>9}"
          f"{'chain':>8}{'':>7}")
    for nm, a, us, _src in LEAF:
        o = byaddr.get(a)
        if not o:
            continue
        s = latency_scan(o[2])
        print(f"  {nm:<14}{(us if us else '-'):>7}{s['loads']:>7}"
              f"{s['full']:>10}{s['part']:>9}{s['serial']:>8}{s['mlp']:>7.1f}")
    print("""
  'serial chain' is the longest run of (ONE load -> s_waitcnt vmcnt(0)), i.e. a
  DEPENDENT POINTER CHASE: n links cost n full memory latencies, back to back,
  with nothing to overlap them.

  THE ROUTER HAS A 17-LINK CHAIN.  Eighteen full drains, seventeen of them
  single-load.  At an L2 hit (~300 cyc) that is ~5100 cycles = 2.1 us of the
  router's 12.9 us/layer; at an HBM miss it is the whole phase.  This is the
  worst memory-level-parallelism structure in the layer by a wide margin and it
  is not in the ledger.

  o_proj has a 3-link chain at tile entry and then pipelines properly -- the
  main v_dot2c body issues 8-10 loads before draining.  (An earlier read of
  this file's own mean-MLP column called o_proj "MLP 1.2, waits after every
  load"; that was the entry chain dominating the mean.  Read the chain column,
  not the mean.)  W_UK/W_UV and merge do sit at MLP 1.0 throughout, against
  the >= 8 in flight MI355X needs for bandwidth.

  qkv_a's resolve is the ONE well-pipelined region on the attention side, at
  MLP 18.3, and it still reaches only 30% of its per-CU L2 share.  That is the
  ceiling this class of fix runs into.
""")


def verdict():
    print("=" * 100)
    print("5. WHAT THIS LICENSES -- READ WITH demo/glm5/makespan_predictor.py")
    print("=" * 100)
    print("""
  NOTHING SINGLE-PHASE.  The regime-A ceilings say a cut inside one phase,
  with its rendezvous surviving, is capped at max_all - max_outside:

      qkv_a 0.248   q_b 0.000   decode 0.000   o_proj 0.011   router 1.199
      us/layer -- 0.111 ms of wall for all five together, under the 0.26 ms
      noise floor EVEN IF ALL FIVE PHASES WERE MADE INSTANTANEOUS.

  So "o_proj has a pointer chase" and "the router has a 17-link chain" are true
  ISA facts that are NOT levers on their own.  Fixing the router's chain
  outright is worth at most 0.091 ms.

  THE ONE SHAPE THAT TRANSFERS 1:1 is a cut landing on ALL 29 xcd_ranks,
  because a uniform cut is regime C run backwards.  Two candidates exist in
  this data and only two:

    (a) The RESOLVE + EP FOLD, 6119 ns and 46.2% of qkv_a's tile, re-reading
        the same 96 KB in every one of the 186 tiles.  Hoisting it is MEASURED
        TWICE AND NEUTRAL (+0.090, +0.187) because it moved the work behind an
        extra rendezvous.  What is NOT measured is making it faster IN PLACE --
        it is at 30% of per-CU L2 and the source says so.

    (b) The QUANTIZER, 851 ns, 206 VALU per trip and zero memory ops, present
        in qkv_a, q_b, the kvupd inner call, MoE W13 and MoE W2 -- five call
        sites, so it lands on nearly every worker in nearly every phase.

  Both are small.  (b) is 851 ns of a 139.7 us layer even if deleted outright,
  i.e. 0.6% of the layer, ~0.05 ms of wall.  That is the honest size of the
  "VALU-bound tile" lever, and it is why this file's answer to item 2 is a
  diagnosis, not a speedup.

  NOT MEASURED.  Everything above is static ISA plus the source's own
  MPK_SUBPHASE_TIMING numbers.  No wall number is quoted and none should be
  until an A/B lands.
""")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else ensure(ISA)
    occ = load(path)
    byaddr = {o[0]: o for o in occ}
    print(f"code object: {path}")
    print(f"{len(occ)} symbol occurrences\n")
    call_graph(occ, byaddr)
    leaves(byaddr)
    qkva()
    latency(byaddr)
    verdict()


if __name__ == "__main__":
    main()
