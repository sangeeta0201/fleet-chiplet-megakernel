#!/usr/bin/env python3
"""ITEM 2: WHERE THE CYCLES GO IN THE NON-MoE TILES -- ISA-level, static.

The guide's hole in CLOSING_LEDGER.md: the ledger prices every phase by BYTES.
Non-MoE width-corrected byte floor is 8.55 us/layer; measured non-MoE tile is
55.9 us/layer.  6.5x.  A roofline was never going to see it, because a
VALU-bound region has no byte floor.

This file does the accounting a roofline cannot: it disassembles the ACTUAL
gfx950 code object that the benchmark ran and classifies every instruction in
the attention-side tile kernels into issue classes, then converts to cycles
with the CDNA issue model and compares against the measured tile time.

    ISSUE MODEL (gfx950 / CDNA3-class, one wave64 per SIMD):
      * a CU has 4 SIMD16s.  A worker block is 256 threads = 4 wave64 = exactly
        ONE wave per SIMD (CLAUDE.md: blockDim is 256, WARPS_PER_CTA=4).
      * With one resident wave per SIMD there is NO other wave to co-issue,
        so every stall is exposed and issue cycles ADD.
      * a wave64 VALU op occupies its SIMD16 for 4 cycles.
      * SALU/branch issue on the scalar unit, 1 cycle, and can overlap VALU.
      * MFMA occupies the matrix core; the VALU is free during it ONLY if the
        compiler scheduled independent VALU there.  We report both.
      * s_waitcnt / s_barrier cost 0 issue cycles; their cost is the STALL,
        which static analysis cannot see -- so the static number is a LOWER
        BOUND on tile time.  That is exactly what makes it decisive: if the
        static VALU lower bound already accounts for most of the measured
        tile, the phase is VALU-ISSUE-BOUND and no memory schedule helps.

Reads the code object next to the megakernel .so.  Offline; no GPU run.
"""
import collections
from collections import Counter
import os
import re
import subprocess
import sys

OBJDUMP = "/opt/rocm/llvm/bin/llvm-objdump"
OBJ = ("demo/glm5/permanent_output_dir_rank0/"
       "test.cpython-38-x86_64-linux-gnu.so.0.hipv4-amdgcn-amd-amdhsa--"
       "gfx950:xnack-")
CLOCK_GHZ = 2.4          # MI355X peak engine clock
VALU_CYC = 4             # wave64 op on a SIMD16

# MFMA issue cycles by shape, gfx950.  Key is the K/N/M signature in the
# mnemonic; value is cycles the matrix core is busy for one wave64 issue.
MFMA_CYC = {
    "16x16x32": 16, "16x16x16": 16, "16x16x8": 16, "16x16x4": 16,
    "32x32x16": 32, "32x32x8": 32, "32x32x4": 64, "32x32x2": 64,
    "16x16x128": 32, "32x32x64": 64,     # scaled fp8/fp4 shapes
    "4x4x4": 8,
}


def classify(mnem):
    """One instruction -> issue class."""
    if mnem.startswith("v_mfma") or mnem.startswith("v_smfmac"):
        return "MFMA"
    if mnem.startswith("v_"):
        return "VALU"
    if mnem.startswith("ds_"):
        return "LDS"
    if (mnem.startswith("global_") or mnem.startswith("buffer_")
            or mnem.startswith("flat_") or mnem.startswith("scratch_")):
        return "VMEM"
    if mnem.startswith("s_load") or mnem.startswith("s_buffer_load"):
        return "SMEM"
    if mnem.startswith("s_waitcnt") or mnem == "s_waitcnt_vscnt":
        return "WAITCNT"
    if mnem.startswith("s_barrier"):
        return "BARRIER"
    if (mnem.startswith("s_branch") or mnem.startswith("s_cbranch")
            or mnem.startswith("s_setpc") or mnem.startswith("s_swappc")
            or mnem.startswith("s_call") or mnem.startswith("s_endpgm")):
        return "BRANCH"
    if mnem.startswith("s_sleep") or mnem.startswith("s_nop"):
        return "NOP/SLEEP"
    if mnem.startswith("s_"):
        return "SALU"
    return "OTHER"


def mfma_cycles(mnem):
    m = re.search(r"(\d+x\d+x\d+)", mnem)
    if m and m.group(1) in MFMA_CYC:
        return MFMA_CYC[m.group(1)]
    return 32  # conservative default; flagged in the report


def disasm(path):
    """symbol -> list of (addr, mnemonic, raw line).

    llvm-objdump for amdgcn emits  `\tMNEMONIC operands   // ADDR: ENCODING`
    -- the address is in the trailing comment, not at the start of the line.
    Branch targets appear as `<symbol+0xNNN>` after the encoding.
    """
    out = subprocess.run([OBJDUMP, "-d", "--mcpu=gfx950", path],
                         capture_output=True, text=True).stdout
    syms = collections.OrderedDict()
    cur = None
    symhdr = re.compile(r"^[0-9a-f]+ <(.+)>:")
    line = re.compile(r"^\t(\S+)(.*?)//\s*([0-9A-F]+):")
    for ln in out.split("\n"):
        m = symhdr.match(ln)
        if m:
            cur = m.group(1)
            syms[cur] = []
            continue
        if cur is None:
            continue
        m = line.match(ln)
        if m:
            syms[cur].append((int(m.group(3), 16), m.group(1), ln))
    return syms


def loops(insns):
    """Innermost backward-branch loop bodies.

    A backward branch's target is printed as `<sym+0xNNN>` in the comment; the
    symbol base is recovered from the first instruction's address minus its
    own offset, so we resolve targets without a symbol table.
    """
    if not insns:
        return []
    addr2i = {a: i for i, (a, _m, _l) in enumerate(insns)}
    base = None
    tgtre = re.compile(r"<[^<>]*\+0x([0-9a-f]+)>")
    # recover the symbol base: any line with a +0x target inside this symbol
    for a, _m, ln in insns:
        t = tgtre.search(ln)
        if t:
            # candidate base = a - (offset of a).  We do not know a's offset,
            # so instead assume the symbol starts at the first insn.
            base = insns[0][0]
            break
    if base is None:
        return []
    found = []
    for i, (a, m, ln) in enumerate(insns):
        if not m.startswith("s_cbranch") and not m.startswith("s_branch"):
            continue
        t = tgtre.search(ln)
        if not t:
            continue
        tgt = base + int(t.group(1), 16)
        if tgt < a and tgt in addr2i:
            found.append((addr2i[tgt], i))
    found.sort(key=lambda p: p[1] - p[0])
    return found


def account(insns, label, trip=1):
    h = collections.Counter()
    cyc = collections.Counter()
    for _a, m, _l in insns:
        c = classify(m)
        h[c] += 1
        if c == "VALU":
            cyc[c] += VALU_CYC
        elif c == "MFMA":
            cyc[c] += mfma_cycles(m)
    n = len(insns)
    return dict(label=label, n=n, hist=h, cyc=cyc, trip=trip)


def report(rows, title):
    print("=" * 100)
    print(title)
    print("=" * 100)
    print(f"{'region':<44}{'insns':>7}{'VALU':>7}{'MFMA':>6}{'VMEM':>6}"
          f"{'LDS':>5}{'SALU':>6}{'wait':>6}{'bar':>5}"
          f"{'VALUcyc':>9}{'MFMAcyc':>9}")
    for r in rows:
        h, c = r["hist"], r["cyc"]
        print(f"{r['label']:<44}{r['n']:>7}{h['VALU']:>7}{h['MFMA']:>6}"
              f"{h['VMEM']:>6}{h['LDS']:>5}{h['SALU']:>6}{h['WAITCNT']:>6}"
              f"{h['BARRIER']:>5}{c['VALU']:>9}{c['MFMA']:>9}")
    print()


MEGA = "_Z17persistent_kernelN6mirage7runtime13RuntimeConfigE"


def innermost(ins):
    """Innermost loop bodies, merged across a loop's multiple exit branches."""
    lps = sorted(set(loops(ins)))
    inr = [(a, b) for a, b in lps
           if not any((c, d) != (a, b) and a <= c and d <= b for c, d in lps)]
    inr.sort()
    out = []
    for a, b in inr:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def valu_kind(m):
    if m.startswith("v_mfma"):
        return "MFMA"
    if (m.startswith("v_and") or m.startswith("v_or") or m.startswith("v_xor")
            or m.startswith("v_lshl") or m.startswith("v_lshr")
            or m.startswith("v_ashr") or m.startswith("v_bfe")
            or m.startswith("v_perm") or m.startswith("v_not")):
        return "bit manipulation"
    if m.startswith("v_cvt") or "pk_fp8" in m or "pk_bf16" in m:
        return "convert/quant"
    if m.startswith("v_cmp") or "cndmask" in m:
        return "select/compare"
    if (m.startswith("v_exp") or m.startswith("v_rcp") or m.startswith("v_rsq")
            or m.startswith("v_log") or m.startswith("v_sqrt")):
        return "transcendental"
    if ("mul_lo_u32" in m or "mul_hi_u32" in m or "mad_u64" in m
            or "lshl_add_u64" in m or "add_co" in m or "addc_co" in m):
        return "64-bit int addr math"
    if (m.startswith("v_mov") or m.startswith("v_readlane")
            or m.startswith("v_writelane") or m.startswith("v_readfirstlane")
            or m.startswith("v_accvgpr") or "permute" in m):
        return "data movement"
    if "_f32" in m or "_f64" in m or "_f16" in m:
        return "float arithmetic"
    return "integer arithmetic"


def megakernel(syms):
    """The accounting that matters: persistent_kernel, which is what RUNS."""
    ins = syms.get(MEGA)
    if not ins:
        print("no persistent_kernel symbol")
        return
    print("=" * 100)
    print("THE MEGAKERNEL (_Z17persistent_kernel) -- THIS is the code that runs")
    print("=" * 100)
    mf = collections.Counter(m for _a, m, _l in ins if m.startswith("v_mfma"))
    print(f"  {len(ins)} instructions.  MFMA opcodes present:")
    for k, v in mf.most_common():
        print(f"    {v:>5}  {k}")
    print("""
  *** RETRACTED 2026-08-23, SAME DAY, BY THE CHECK THAT SHOULD HAVE COME FIRST.
      An earlier version of this file reported "the megakernel contains no
      scaled fp8 MFMA" because _Z17persistent_kernel has only bf16 MFMA.  THAT
      WAS A SYMBOL-BOUNDARY ARTIFACT, NOT A FINDING.  The phase kernels are
      __device__ __noinline__, so they are separate symbols that persistent_kernel
      reaches through s_swappc_b64 (168 in the object, 61 in persistent_kernel
      itself), and gang_mla_full_layer_fused_kernel_mi300 is outlined the same
      way.  Counting MFMA "in the megakernel" by counting them inside one symbol
      counts the inliner, not the machine.  This is exactly the failure
      glm-gemm-isa-matches-gpt-oss-and-is-ahead warns about from the other
      direction.  The census below is per PHASE SYMBOL and is the real one. ***

  See phase_accounting() for the table that replaces it.
""")
    v = collections.Counter(valu_kind(m) for _a, m, _l in ins
                            if m.startswith("v_"))
    tot = sum(v.values())
    print(f"  {'VALU category (STATIC, whole symbol)':<32}{'insns':>8}{'share':>8}")
    for k, n in v.most_common():
        print(f"  {k:<32}{n:>8}{100*n/tot:>7.1f}%")
    print("""
  Static counts over a 27k-instruction symbol are NOT dynamic cycles.  The
  dynamic answer is in the innermost loop bodies below, which is where trip
  count multiplies.
""")

    print("=" * 100)
    print("INNERMOST LOOP BODIES OF THE MEGAKERNEL, by kind")
    print("=" * 100)
    rows = []
    for a, b in innermost(ins):
        body = ins[a:b + 1]
        c = collections.Counter(m for _x, m, _l in body)
        h = collections.Counter(classify(m) for _x, m, _l in body)
        fma = sum(n for k, n in c.items()
                  if k.startswith(("v_fma", "v_fmac", "v_mac", "v_pk_fma",
                                   "v_pk_mul", "v_pk_add", "v_dot")))
        bit = sum(n for k, n in c.items() if valu_kind(k) == "bit manipulation")
        cvt = sum(n for k, n in c.items() if valu_kind(k) == "convert/quant")
        rows.append((ins[a][0], len(body), h, fma, bit, cvt))
    gemv = [r for r in rows if r[3] >= 4 and r[2]["VMEM"] >= 1
            and not r[2]["MFMA"] and r[1] < 100]
    print("  A. THE fp8 GEMV k-LOOPS (float FMA + VMEM, no MFMA):")
    print(f"  {'addr':>9}{'insns':>7}{'VALU':>6}{'FMA':>5}{'bitmanip':>10}"
          f"{'cvt':>5}{'VMEM':>6}{'VALUcyc':>9}{'useful':>8}")
    for a, n, h, fma, bit, cvt in sorted(gemv):
        print(f"  {a:>9x}{n:>7}{h['VALU']:>6}{fma:>5}{bit:>10}{cvt:>5}"
              f"{h['VMEM']:>6}{h['VALU']*VALU_CYC:>9}"
              f"{100*fma/max(1,h['VALU']):>7.0f}%")
    print("""
  READ THE 'useful' COLUMN.  In the widest GEMV k-loop 8 of 40 VALU ops are the
  actual multiply-accumulate; 19 are BIT MANIPULATION (unpacking fp8/mxfp4 out
  of packed dwords) and 4 are converts.  80% of the VALU issue slots in the
  hot loop of the GEMV path are spent turning bytes into floats.  Per
  phase_accounting() below, the GEMV path is W_UK, W_UV and o_proj -- NOT
  qkv_a/q_b/W13/W2, which do use the scaled MFMA and have a different problem.

  THIS IS WHERE THE 3.55 ms LIVES, and it is a VALU-issue cost, not a memory
  cost -- so no prefetch, no L2 hint and no wider load touches it.  Two levers
  follow, and ONLY these two:
    (i)  FEWER UNPACK OPS PER ELEMENT.  19 bit ops per 8 FMAs is far above what
         v_perm_b32 needs to split a dword into 4 scaled bytes.  Note the
         makespan rule (demo/glm5/MAKESPAN_RULE.md): a cut confined to ONE
         phase is regime A and capped at 0.000-1.199 us/layer, so this only
         pays if the SAME cut lands across phases -- which a shared unpack
         helper does, since W_UK, W_UV and o_proj share this loop.
    (ii) LET THE MATRIX CORE DO THE DEQUANT.  v_mfma_scale_f32_16x16x128_f8f6f4
         decodes fp8 in hardware.  At M=1 it wastes 15/16 of the tile, but the
         matrix core is IDLE during every GEMV, and it would delete the unpack
         VALU entirely.  That trade has never been priced on this branch.

  NOT YET MEASURED.  Everything above is static ISA; the trip counts that turn
  it into ms are the next step, and then an A/B.  Do not quote a ms number for
  this until that lands.
""")
    top = sorted(rows, key=lambda r: -r[1])[:6]
    print("  B. THE LARGEST INNERMOST LOOPS (VALU-only, no MFMA at all):")
    print(f"  {'addr':>9}{'insns':>7}{'VALU':>6}{'VALUcyc':>9}   likely identity")
    ID = {0xaf4f0: "router TopK sort (36x ds_bpermute, cmp/cndmask/lshr)",
          0xad794: "sigmoid block (32x v_exp_f32 + v_ldexp + fmac)",
          0x9d13c: "64-bit index math (v_mul_lo/hi_u32 pairs)",
          0x9d668: "64-bit index math (v_mul_lo/hi_u32 pairs)",
          0x9db30: "64-bit index math (v_mul_lo/hi_u32 pairs)",
          0xae22c: "sigmoid + bf16 convert"}
    for a, n, h, fma, bit, cvt in top:
        print(f"  {a:>9x}{n:>7}{h['VALU']:>6}{h['VALU']*VALU_CYC:>9}   "
              f"{ID.get(a, '?')}")
    print()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else OBJ
    if not os.path.exists(path):
        sys.exit(f"no code object at {path}\n"
                 "  build it: run the benchmark once, then the file appears "
                 "next to permanent_output_dir_rank0/*.so")
    syms = disasm(path)
    print(f"code object: {path}")
    print(f"symbols: {len(syms)}")
    print()

    # The attention-side tile kernels, by their template signature.
    WANT = [
        ("qkv_a  (rmsnorm+linear+kvupd, N=2048)",
         "gang_rmsnorm_linear_mxfp8_bias_mla_kvupd_kernel"),
        ("q_b / W_UK / W_UV  (rmsnorm+linear)",
         "gang_rmsnorm_linear_mxfp8_bias_kernel"),
        ("o_proj  (gemv mxfp8)", "gang_gemv_mxfp8_kernel"),
        ("MLA decode (absorbed)", "mla_decode_absorbed"),
        ("router (rmsnorm+linear+bias+topk)",
         "gang_rmsnorm_linear_bias_topk_kernel"),
        ("MoE W13", "gang_moe_w13_linear_mxfp8_kernel"),
        ("MoE W2", "gang_moe_w2_linear_mxfp8_kernel"),
    ]
    rows = []
    seen = set()
    for label, needle in WANT:
        for s in syms:
            if needle in s and s not in seen and len(syms[s]) > 40:
                seen.add(s)
                # shorten the mangled name to its template args
                targ = re.search(r"I(.+?)EEEv", s)
                tag = (targ.group(1)[:22] if targ else "")
                rows.append(account(syms[s], f"{label.split('(')[0].strip()}"
                                             f" <{tag}>"))
                break
    report(rows, "STANDALONE TILE KERNELS (the UNFUSED dense-prologue path, "
                 "NOT what the megakernel runs)")

    megakernel(syms)
    phase_accounting(syms)

    # --- the part that decides it: the innermost loop of each kernel --------
    print("=" * 100)
    print("(reference) INNERMOST LOOPS OF THE UNFUSED STANDALONE KERNELS")
    print("=" * 100)
    for label, needle in WANT:
        sym = next((s for s in syms if needle in s and len(syms[s]) > 40),
                   None)
        if not sym:
            continue
        ins = syms[sym]
        lp = loops(ins)
        if not lp:
            print(f"{label:<44} no backward branch (fully unrolled)")
            continue
        i0, i1 = lp[0]
        body = ins[i0:i1 + 1]
        r = account(body, label)
        h, c = r["hist"], r["cyc"]
        print(f"{label:<44}{len(body):>6} insns  "
              f"VALU {h['VALU']:>4} MFMA {h['MFMA']:>3} VMEM {h['VMEM']:>3} "
              f"LDS {h['LDS']:>3} wait {h['WAITCNT']:>3}  "
              f"-> {c['VALU']:>5} VALU cyc + {c['MFMA']:>5} MFMA cyc")
    print()




# ---------------------------------------------------------------------------
# THE ITEM-2 TABLE.  Per PHASE SYMBOL, not per inlining boundary.
# ---------------------------------------------------------------------------

PHASE_SYMS = [
    ("qkv_a / q_b", "gang_rmsnorm_linear_mxfp8_bias_k", "scaled fp8 MFMA"),
    ("W_UK,W_UV,o_proj(fp8)", "gang_gemv_mxfp8_kernel", "NO MFMA -- VALU dot"),
    ("o_proj (fp4)", "gang_gemv_mxfp4_kernel", "NO MFMA -- VALU dot"),
    ("W13", "gang_moe_w13_linear_mxfp8", "scaled fp8 MFMA"),
    ("W2", "gang_moe_w2_linear_mxfp8", "scaled fp8 MFMA"),
    ("MLA decode", "mla_decode_absorbed", "bf16 MFMA"),
]


def _unpackish(m):
    return m.startswith(("v_and", "v_or", "v_lshl", "v_lshr", "v_bfe", "v_perm",
                         "v_ashr", "v_xor", "v_bfi", "v_alignb", "v_pack"))


def _fmaish(m):
    return m.startswith(("v_fma", "v_mac", "v_fmac", "v_dot", "v_mul_f",
                         "v_add_f", "v_pk_fma", "v_pk_mul", "v_pk_add"))


def phase_accounting(syms):
    """The guide's ask: issue-class census of the tile that each phase runs."""
    print("=" * 108)
    print("ITEM 2: ISSUE-CLASS CENSUS PER PHASE SYMBOL (largest instantiation)")
    print("static instruction counts, NOT dynamic cycles -- ratios are the claim")
    print("=" * 108)
    hdr = (f"{'phase':24}{'insn':>6}{'VALU%':>7}{'MFMA':>5}{'VMEM':>5}{'LDS':>5}"
           f"{'SALU':>5}{'WAIT':>5} |{'unpack':>7}{'cvt':>5}{'fma':>5}  matrix path")
    print(hdr)
    print("-" * 108)
    for label, pat, path in PHASE_SYMS:
        cands = [(k, v) for k, v in syms.items() if pat in k]
        if not cands:
            print(f"{label:24}  MISSING")
            continue
        _, v = max(cands, key=lambda x: len(x[1]))
        c = Counter(classify(m) for _, m, _ in v)
        valu = [m for _, m, _ in v if classify(m) == "VALU"]
        up = sum(1 for m in valu if _unpackish(m))
        cv = sum(1 for m in valu if "cvt" in m)
        fm = sum(1 for m in valu if _fmaish(m))
        print(f"{label:24}{len(v):6}{100.0*c['VALU']/len(v):6.0f}%{c['MFMA']:5}"
              f"{c['VMEM']:5}{c['LDS']:5}{c['SALU']:5}{c['WAITCNT']:5} |"
              f"{up:7}{cv:5}{fm:5}  {path}")
    print("""
READ THIS TABLE AS TWO SEPARATE PROBLEMS, NOT ONE:

  (1) qkv_a / q_b DO use the scaled fp8 MFMA -- and are still 74% VALU.  Eight
      v_mfma_scale_f32_16x16x128_f8f6f4 carry the whole GEMM; the other ~2150
      VALU ops (882 bit manipulation + 566 float) are the RMSNorm and the
      activation quantize.  That is the ISA confirmation of
      glm-qkva-tile-is-41pct-redundant-prologue and
      glm-quant-prologue-is-not-load-issue-bound: the prologue is issue cost,
      it touches almost no HBM, and it is most of the tile.

  (2) W_UK, W_UV AND o_proj HAVE NO MATRIX INSTRUCTION AT ALL.  They run
      gang_gemv_mxfp8_kernel / gang_gemv_mxfp4_kernel, which are VALU dot
      products: 516 v_cvt_scalef32_pk_bf16_fp8 + 536 float FMA + 392 bit ops,
      zero v_mfma.  o_proj is a top-5 board line and was priced at 76% of HBM
      peak (glm-oproj-is-at-76pct-of-hbm-peak) -- by BYTES.  Its ISSUE cost was
      never on the board at all.

  This is the byte-pricing hole, located.  The MoE (W13/W2) is fine: 8 scaled
  MFMA against ~600 VALU of address math and staging.

  THE CHEAPEST TEST OF (2) ALREADY EXISTS IN THE TREE AND IS COMPILED OFF.
  gang_oproj_router_fused_mi300.cuh:444 and gang_mla_full_layer_fused_mi300.cuh
  guard a `WUV_USE_MFMA` branch on -DMPK_WUV_MFMA that routes W_UV to
  gang_linear_mxfp8_kernel (the _gang_mfma_f8xf8 path) instead of the GEMV.
  One -D, one rebuild, one A/B -- that is the next measurement, and it is a
  ONE-VARIABLE change.

  CAVEAT, STATED UP FRONT: static counts are not cycles.  These ratios say what
  the tile is MADE of; they do not say how many times each loop trips.  No ms
  number is claimed from this table.
""")


if __name__ == "__main__":
    main()
