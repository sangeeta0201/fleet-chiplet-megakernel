#!/usr/bin/env python3
"""Verify a Perfetto trace against the [PTRACEW]/[PTRACES] log it came from.

phase_slots_to_perfetto.py is the only thing standing between raw device
timestamps and a picture people will draw conclusions from, and a rendering bug
in it looks exactly like a hardware finding. That already happened once: a
detail row labelled "(async)" and a phase span named after one op inside it
read, correctly, as "ResAdd and the GEMM run in parallel on worker 0". Nothing
was wrong with the measurement; the picture was wrong. This script exists so
that class of error is caught by a check rather than by a reader.

It re-derives every slice from the log independently -- it does not import the
exporter -- and then checks the emitted JSON against that derivation, per
worker, per layer, per slot. Nothing is sampled and nothing is summarised: a
disagreement anywhere is a failure.

  Usage: verify_perfetto_trace.py <run.log> <trace.json> [-v]

Checks, in order:
  1  header/parse agreement       log parses to the same rows the exporter saw
  2  slot monotonicity            the 12 marks per row ascend (shared clock)
  3  slice provenance             every emitted slice maps to a raw tick pair
  4  coverage                     every valid (w,l,slot) span is drawn once
  5  split tiling                 segments tile their parent slot exactly
  6  submark containment          P1 marks lie inside the slot-1 window
  7  no same-row overlap          a worker's own row is strictly sequential
  8  detail-row containment       detail slices lie inside their layer
  9  inter-layer continuity       gaps between layers are drawn, not implied
 10  time base                    ts = (tick - t0) * tick_ns / 1000, exactly
 11  participation                a gated op is only NAMED when it ran

Checks 1-10 verify fidelity -- that each slice reproduces its tick pair. They
all passed on a trace that drew workers as running a QKV GEMM they had branched
past, because the ticks were honest and only the name was wrong. Check 11 is the
one that asks what the interval means.
"""
import argparse
import json
import re
import sys
from collections import defaultdict

HDR_RE = re.compile(r"^\[PTRACE\] slots=(\d+) layers=(\d+) tick_ns=(\d+)\s*$")
SUB_RE = re.compile(r"^\[PTRACES\] w=(\d+) l=(\d+) n=(\d+)((?: \d+:\d+)+)\s*$")
ROW_RE = re.compile(
    r"^\[PTRACEW\] w=(\d+) x=(-?\d+) l=(\d+)(?: p=(\d+))?(?: a=(\d+))?"
    r"((?: \d+)+)\s*$")

# Slots whose span only means the op when the participation bit is set.
GATED_SLOTS = {1: "qkv", 3: "attn", 6: "oproj", 8: "moe"}
SKIP_NAME = "(skipped: not a {} participant)"

# Phases whose participants split into arms. The participation bit says the
# worker took part; for MoE it cannot say in WHAT, because each tile is a W13
# tile or a W2 tile, some workers get no tile, and the strided loop gives the
# low ranks two tiles -- so a worker can run both arms. The code is therefore a
# SET of arms: 0 none, 1 W13, 2 W2, 3 both. Restated here rather than imported,
# same as SLOT_SPLIT.
ARM_BITS, ARM_MASK = 2, 3
ARM_NAMES = {
    8: {0: "(skipped: no moe tile)", 1: "moe W13+SwiGLU",
        2: "moe W2 (down proj)", 3: "moe W13+SwiGLU, then W2"},
}

# Mirrors the exporter's SLOT_SPLIT, restated rather than imported. If the two
# drift the check fails, which is the point: a verifier that imports the thing
# it verifies can only confirm the code is self-consistent, not correct.
SPLIT_SLOT = 1
SPLIT_HEAD = "resadd_prologue(lds+dma issue)"
SPLIT_SPANS = [(6, 7, "ResAdd(f32)"),
               (7, 8, "RMSNorm"),
               (8, 9, "FP8 quant + weight drain")]
SPLIT_TAIL_FROM = 9
SPLIT_TAIL = "QKV GEMM(MXFP4) + bias + RoPE + KV-upd"
SPLIT_LABELS = {SPLIT_HEAD, SPLIT_TAIL} | {s[2] for s in SPLIT_SPANS}

# Slot 8 splits differently and the difference is not cosmetic. Slot 1's marks
# are handoffs, so its segments tile the parent end to end. Slot 8's marks
# bracket TILES: a worker gets one or two from the strided loop, and the time
# between two tiles is loop overhead and barrier wait belonging to neither arm.
# So these are islands with the space between them named, and a worker emits
# each code once per tile -- pair them positionally, do not keep only the first.
SEG_SLOT = 8
SEG_PAIRS = [(10, 11, "moe W13+SwiGLU"), (12, 13, "moe W2 (down proj)")]
# Both gaps are the same thing: per-tile SETUP, between two kernel *calls*
# (the caller invokes the whole kernel once per tile). Entry, tile decode --
# two dependent HBM loads, d_mask then d_routing indexed by what it returned --
# token compaction, __syncthreads, LDS/pointer setup, prefetch issue.
#
# Naming the second gap "W13->W2 barrier wait" was WRONG, and wrong in the way
# this file exists to catch: it was inferred from the gap's position between
# the arms rather than read off the source. The poll is at
# gang_moe_fused_mxfp4_mi300.cuh:2793, which lies between W2_BEG (:2526) and
# W2_END (:3461) -- inside the W2 span, not before it. Marks 14/15 now bracket
# it there.
SEG_GAP = "moe tile setup (decode+gather+pf)"
SEG_GAP_BEFORE = {"moe W13+SwiGLU": SEG_GAP, "moe W2 (down proj)": SEG_GAP}
SEG_GAP_TAIL = "moe loop exit"

# Interior cuts of a tile span. These ARE handoffs, so they tile their parent
# edge to edge, and the tile's own extent is preserved.
SEG_INNER = {
    "moe W2 (down proj)": {"cuts": [(14, "W2 prep (quant+pf issue)"),
                                    (15, "W13->W2 barrier wait")],
                           "tail": "moe W2 compute"},
}
SEG_LABELS = ({SEG_GAP_TAIL} | set(SEG_GAP_BEFORE.values())
              | {s[2] for s in SEG_PAIRS}
              | {sub for v in SEG_INNER.values() for _, sub in v["cuts"]}
              | {v["tail"] for v in SEG_INNER.values()})

# ── Barrier waits cut out of their enclosing phase span ────────────────────
#
# Restated from the source, not imported from the exporter -- the same rule as
# everything else in this file. Each mark is taken once per layer, immediately
# before the poll it names, and the cut runs from the mark to the slot's end,
# so the pieces tile the parent edge to edge.
#
# The scope in each name was read off the kernel, not guessed from the trace:
#   16 qkv_epoch   gang_full_layer_fused_mi300.cuh:527  -> :506/:514/:525
#      per-XCD atomic, per-XCD flag                       XCD-LOCAL
#   17 attn_release   :953  -> :829   last XCD at attn_global writes all 8
#   18 oproj_hier     :1177 -> :1155  single global_arrive, per-XCD release
#   19 routing_ready  :1198 -> :1943  one TopK completer writes all 8 flags
#   20 layer_release  :1866 -> :1672  last XCD broadcasts all 8
# 20 and 21 are the two arms of one if/else on the layer gate predicate, so a
# worker emits exactly one of them and never both.
BARRIER_CUTS = {
    2: [(16, "qkv_epoch barrier (XCD-LOCAL)")],
    5: [(17, "attn_release barrier (GLOBAL)")],
    7: [(18, "oproj_hier barrier (GLOBAL)"),
        (19, "routing_ready/TopK barrier (GLOBAL)")],
    10: [(20, "layer_release barrier (GLOBAL)"),
         (21, "idle: not in the layer gate")],
}
BARRIER_HEAD = {2: "qkv post-GEMM drain", 5: "attn epilogue",
                7: "moe prologue (pre-barrier)", 10: "layer release fan-out"}
BARRIER_IDLE = {"idle: not in the layer gate"}
BARRIER_LABELS = ({lab for cuts in BARRIER_CUTS.values() for _, lab in cuts}
                  | set(BARRIER_HEAD.values()))

# ── The detail row: async weight DMAs ──────────────────────────────────────
#
# Four weight DMAs are issued into a wait, and until now this list had two.
# Check 8 below only asked whether a detail slice sat inside its layer on the
# right row, so a DMA that was never drawn at all passed every check -- the
# absence had nothing to disagree with. That is how the O-proj and W2
# prefetches stayed uninstrumented while kernel comments asserted they
# overlapped: no measurement, and no check that noticed there wasn't one.
#
# So the expected set is derived here, from the codes, and compared name for
# name and tick for tick. Restated from the source, not imported:
#   1  qkv   issue  gang_full_layer_fused_mi300.cuh:1795  (no drain mark; the
#            consumer is the NEXT layer's Phase 1, which skips its own DMA
#            rather than draining this one, so the span ends at layer exit)
#   3/4/5 W13 issue/quant/drain  gang_moe_fused_mxfp4_mi300.cuh
#   22/23/26 oproj issue :920 -> drain :1042 -> done :1046, straddling the
#            s_waitcnt before buffer_inv
#   24/25/27 W2    issue :2610 -> drain :2886 -> done :2891, between the vmcnt
#            and lgkmcnt waits (lgkmcnt is LDS traffic, not this DMA)
#
# The drain->done span is the point: issue->drain measures how long the load
# had to finish, and only the wait itself says whether it did.
SUB_SPANS = [
    (3, 4, "W13 weight DMA in flight"),
    (4, 5, "W13 weight DMA drain (uncovered)"),
    (22, 23, "oproj weight DMA in flight"),
    (23, 26, "oproj weight DMA drain (uncovered)"),
    (24, 25, "W2 weight DMA in flight"),
    (25, 27, "W2 weight DMA drain (uncovered)"),
]
# The O-proj drain is on the common path while its issue is inside the rank
# guard, so 8927 rows take mark 23 and only 6624 take 22. Without this the
# drain span would be drawn for 2303 workers per layer that issued no O-proj
# DMA -- a name taken from a position in the instruction stream rather than
# from work the worker did. Restated here, not imported.
SUB_REQUIRES = {"oproj weight DMA drain (uncovered)": 22}
SUB_OPEN_ENDED = (1, "qkv weight DMA in flight")

SPAN_NAMES = [
    None, "qkv_fused(resadd..kvupd)", "qkv_epoch_barrier", "attention+merge",
    # Slot 4: marks 3 and 4 are adjacent source lines with no code between, so
    # this span is the instrument's own cost, not a phase. See the exporter.
    "instrument floor (mark 3->4, no code between)",
    "attn_release_wait", "oproj+rmsnorm+router", "topk_wait",
    "moe(w13+swiglu+w2)", "layer_arrive+fanout+pf", "layer_gate_poll",
    "layer_exit",
]

TID_ASYNC = 100000
# Ticks are 10 ns and Perfetto's unit is us, so every timestamp is an exact
# multiple of 0.01 and float compare needs no slack beyond binary rounding.
EPS = 1e-9


class Report:
    def __init__(self, verbose):
        self.verbose = verbose
        self.failures = []
        self.checks = []

    def check(self, name, ok, detail="", count=None):
        self.checks.append((name, ok, detail, count))
        if not ok:
            self.failures.append((name, detail))
        mark = "PASS" if ok else "FAIL"
        n = f"  [{count}]" if count is not None else ""
        print(f"  {mark}  {name}{n}" + (f"  -- {detail}" if detail else ""))

    def fail_samples(self, name, bad, limit=5):
        ok = not bad
        detail = "" if ok else f"{len(bad)} violation(s); first {min(limit, len(bad))}:"
        self.check(name, ok, detail, count=len(bad) if ok else None)
        if not ok:
            for b in bad[:limit]:
                print(f"          {b}")


def parse_log(path):
    hdr, rows, subs, corrupt, saw_end = None, [], {}, 0, False
    dup_rows, dup_subs = [], []
    bad_lines = []
    with open(path, errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            if line.startswith("[PTRACE] "):
                m = HDR_RE.match(line)
                if m:
                    hdr = {"slots": int(m.group(1)), "layers": int(m.group(2)),
                           "tick_ns": int(m.group(3))}
            elif line.startswith("[PTRACES] "):
                m = SUB_RE.match(line)
                if not m:
                    corrupt += 1
                    bad_lines.append((lineno, line))
                    continue
                pairs = [p.split(":") for p in m.group(4).split()]
                if len(pairs) != int(m.group(3)):
                    corrupt += 1
                    bad_lines.append((lineno, line))
                    continue
                key = (int(m.group(1)), int(m.group(2)))
                if key in subs:
                    dup_subs.append(key)
                subs[key] = [(int(c), int(t)) for c, t in pairs]
            elif line.startswith("[PTRACEW] "):
                m = ROW_RE.match(line)
                if not m:
                    corrupt += 1
                    bad_lines.append((lineno, line))
                    continue
                ts = [int(x) for x in m.group(6).split()]
                if hdr is not None and len(ts) != hdr["slots"]:
                    corrupt += 1
                    bad_lines.append((lineno, line))
                    continue
                part, arm = m.group(4), m.group(5)
                rows.append((int(m.group(1)), int(m.group(2)),
                             int(m.group(3)), ts,
                             int(part) if part is not None else None,
                             int(arm) if arm is not None else None))
            elif line.startswith("[PTRACE_END]"):
                saw_end = True
    seen = set()
    for w, _, l, _, _, _ in rows:
        if (w, l) in seen:
            dup_rows.append((w, l))
        seen.add((w, l))
    return hdr, rows, subs, corrupt, saw_end, dup_rows, dup_subs, bad_lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("trace")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    (hdr, rows, subs, corrupt, saw_end, dup_rows, dup_subs,
     bad_lines) = parse_log(args.log)
    if hdr is None:
        sys.exit(f"no [PTRACE] header in {args.log}")
    tick_ns = hdr["tick_ns"]
    nslots = hdr["slots"]

    with open(args.trace) as fh:
        trace = json.load(fh)
    evs = trace["traceEvents"]
    spans = [e for e in evs if e.get("ph") == "X"]
    meta = [e for e in evs if e.get("ph") == "M"]

    part_of = {(w, l): pt for w, _, l, _, pt, _ in rows if pt is not None}
    arm_of = {(w, l): am for w, _, l, _, _, am in rows if am is not None}

    rep = Report(args.verbose)
    print(f"log   {args.log}")
    print(f"trace {args.trace}")
    print(f"header: slots={nslots} layers={hdr['layers']} tick_ns={tick_ns}")
    print(f"parsed: {len(rows)} PTRACEW rows, {len(subs)} PTRACES rows, "
          f"{corrupt} corrupt, PTRACE_END={saw_end}")
    print(f"trace:  {len(spans)} slices, {len(meta)} metadata events\n")

    # ── 1. parse agreement ────────────────────────────────────────────────
    print("1. header / parse agreement")
    # A corrupt line is host stdout spliced into the device printf stream --
    # the model's own generated text, an fprintf from elsewhere, anything. It
    # costs exactly the rows it landed on, so report WHICH rows are gone rather
    # than only how many lines failed to parse: "1 line dropped" is not
    # actionable, "worker 243 layer 1 is missing, 0.2% of worker-time" is.
    rep.check("no corrupt lines", corrupt == 0,
              "" if corrupt == 0 else
              f"{corrupt} line(s) spliced by host stdout")
    for lineno, line in bad_lines[:5]:
        shown = line.strip()[:110].replace("\n", " ")
        print(f"          line {lineno}: {shown!r}")
    rep.check("[PTRACE_END] present (dump not truncated)", saw_end)
    rep.fail_samples("no duplicate (worker,layer) PTRACEW rows", dup_rows)
    rep.fail_samples("no duplicate (worker,layer) PTRACES rows", dup_subs)
    layers = sorted({l for _, _, l, _, _, _ in rows})
    workers = sorted({w for w, _, _, _, _, _ in rows})
    have = {(w, l) for w, _, l, _, _, _ in rows}
    absent = [(w, l) for w in workers for l in layers if (w, l) not in have]
    rep.fail_samples("every worker has a row for every layer",
                     [f"w={w} l={l} missing" for w, l in absent])
    if absent:
        # Quantify the hole, so the reader can judge whether the trace is still
        # usable rather than guessing. A missing row is missing WORK: in the
        # picture it looks like the worker stopped early, which is exactly the
        # misreading a load-imbalance story would be built on.
        durs = [(ts[nslots - 1] - ts[0]) * tick_ns / 1000.0
                for w, _, l, ts, pt, am in rows if ts[0] and ts[nslots - 1]]
        durs.sort()
        med = durs[len(durs) // 2] if durs else 0.0
        total = sum(durs)
        print(f"          est. {len(absent) * med:.1f} worker-us absent "
              f"= {100 * len(absent) * med / max(total, 1e-9):.3f}% of "
              f"{total:.0f}; those rows are MISSING WORK, not idle time")

    # ── 2. slot monotonicity ──────────────────────────────────────────────
    print("\n2. slot monotonicity (shared s_memrealtime)")
    nonmono, zero_interior = [], []
    for w, x, l, ts, pt, am in rows:
        nz = [(i, t) for i, t in enumerate(ts) if t]
        for (i0, t0_), (i1, t1_) in zip(nz, nz[1:]):
            if t1_ < t0_:
                nonmono.append(f"w={w} l={l} slot {i0}->{i1}: {t0_} > {t1_}")
        # A zero between two non-zeros is a dropped mark, not a skipped role.
        if nz:
            lo, hi = nz[0][0], nz[-1][0]
            for i in range(lo, hi + 1):
                if ts[i] == 0:
                    zero_interior.append(f"w={w} l={l} slot {i} is zero "
                                         f"between {lo} and {hi}")
    rep.fail_samples("timestamps ascend within every row", nonmono)
    rep.fail_samples("no zero mark interior to a row", zero_interior)

    # ── 10. time base (needed before provenance) ──────────────────────────
    t0 = min(ts[0] for _, _, _, ts, _, _ in rows if ts[0])

    def us(tick):
        return (tick - t0) * tick_ns / 1000.0

    # ── 3+4. provenance and coverage ──────────────────────────────────────
    print("\n3. slice provenance  +  4. coverage")
    # Expected: for each (w,l,slot) with both marks non-zero, either one span
    # named SPAN_NAMES[slot], or -- for the split slot when marks exist -- the
    # exact segment list.
    expected = {}           # (w,l,slot) -> list of (ts_us, dur_us, name)
    split_pairs, unsplit, gated_out, armed, seg_pairs = 0, 0, 0, 0, 0
    bar_cut = 0
    for w, x, l, ts, pt, am in rows:
        for s in range(1, nslots):
            a, b = ts[s - 1], ts[s]
            if a == 0 or b == 0 or b < a:
                continue
            # A mark on a gated slot is taken by every worker that REACHES the
            # phase, participant or not, because a mark inside the guard would
            # leave non-participants with no timestamp and break the ascending-
            # marks invariant. So the tick pair alone cannot distinguish "ran
            # the op fast" from "fell through the branch"; only p= can. A span
            # named after the op without its bit set is the bug this check
            # exists for.
            if pt is not None and s in GATED_SLOTS and not (pt >> s) & 1:
                gated_out += 1
                expected[(w, l, s)] = [(us(a), us(b) - us(a),
                                        SKIP_NAME.format(GATED_SLOTS[s]))]
                continue
            # A barrier poll inside a phase span, cut out edge to edge. First
            # occurrence only: these fire once per layer, unlike the tile
            # marks above. An absent code means the worker never reached that
            # barrier -- ranks below the o-proj guard skip 18 -- and the cut
            # correctly does not happen.
            if s in BARRIER_CUTS:
                at1 = {}
                for code, tick in (subs.get((w, l)) or []):
                    at1.setdefault(code, tick)
                hit = sorted((at1[c], lab) for c, lab in BARRIER_CUTS[s]
                             if c in at1 and a <= at1[c] <= b)
                if hit:
                    bar_cut += 1
                    bounds = [(a, BARRIER_HEAD[s])] + hit
                    ends = [t for t, _ in hit] + [b]
                    expected[(w, l, s)] = [
                        (us(beg), us(end) - us(beg), lab)
                        for (beg, lab), end in zip(bounds, ends) if end > beg]
                    continue

            # Tile segments are the finest measurement available, so they
            # outrank the arm name, which describes the whole phase. Derived
            # from the marks here and from the bits below; the two are asserted
            # to agree in check 11, so this is not a second copy of one fact.
            if s == SEG_SLOT:
                byc = defaultdict(list)
                for code, tick in (subs.get((w, l)) or []):
                    byc[code].append(tick)
                cuts = []
                for c0, c1, label in SEG_PAIRS:
                    for beg, end in zip(byc.get(c0, []), byc.get(c1, [])):
                        if end > beg and beg >= a and end <= b:
                            cuts.append((beg, end, label))
                cuts.sort()
                if cuts:
                    seg_pairs += 1
                    filled, cur = [], a
                    for beg, end, label in cuts:
                        if beg > cur:
                            filled.append((cur, beg, SEG_GAP_BEFORE[label]))
                        spec = SEG_INNER.get(label)
                        icuts = []
                        if spec:
                            for code, sub in spec["cuts"]:
                                for t in byc.get(code, []):
                                    if beg <= t <= end:
                                        icuts.append((t, sub))
                                        break
                        if icuts:
                            icuts.sort()
                            ic = beg
                            for t, sub in icuts:
                                if t > ic:
                                    filled.append((ic, t, sub))
                                ic = t
                            if end > ic:
                                filled.append((ic, end, spec["tail"]))
                        else:
                            filled.append((beg, end, label))
                        cur = end
                    if b > cur:
                        filled.append((cur, b, SEG_GAP_TAIL))
                    expected[(w, l, s)] = [(us(c0), us(c1) - us(c0), n)
                                           for c0, c1, n in filled if c1 > c0]
                    continue

            # Participating -- but for a fused phase, in which arm? MoE passes
            # the bit check for all 248 workers and still mislabels every one
            # of them, because the bit records entry and the arm records work.
            if am is not None and s in ARM_NAMES:
                nm = ARM_NAMES[s].get((am >> (s * ARM_BITS)) & ARM_MASK)
                if nm is not None:
                    armed += 1
                    expected[(w, l, s)] = [(us(a), us(b) - us(a), nm)]
                    continue
            at = {}
            if s == SPLIT_SLOT:
                for code, tick in (subs.get((w, l)) or []):
                    at.setdefault(code, tick)
            need = [SPLIT_SPANS[0][0], SPLIT_TAIL_FROM]
            if s == SPLIT_SLOT and all(c in at for c in need):
                split_pairs += 1
                segs = []
                first = at[SPLIT_SPANS[0][0]]
                if first > a:
                    segs.append((a, first, SPLIT_HEAD))
                for c0, c1, label in SPLIT_SPANS:
                    if c0 in at and c1 in at and at[c1] >= at[c0]:
                        segs.append((at[c0], at[c1], label))
                if b > at[SPLIT_TAIL_FROM]:
                    segs.append((at[SPLIT_TAIL_FROM], b, SPLIT_TAIL))
                expected[(w, l, s)] = [(us(c0), us(c1) - us(c0), n)
                                       for c0, c1, n in segs if c1 > c0]
            else:
                if s == SPLIT_SLOT:
                    unsplit += 1
                expected[(w, l, s)] = [(us(a), us(b) - us(a), SPAN_NAMES[s])]

    got = defaultdict(list)
    orphan = []
    for e in spans:
        if e.get("cat") != "phase":
            continue
        a_ = e["args"]
        slot = a_.get("slot")
        if slot == 0:
            continue                                   # inter_layer, check 9
        key = (a_["worker"], a_["layer"], slot)
        if key not in expected:
            orphan.append(f"{e['name']} w={a_['worker']} l={a_['layer']} "
                          f"slot={slot} has no source tick pair")
        got[key].append((e["ts"], e["dur"], e["name"], e))

    rep.fail_samples("no slice without a source tick pair", orphan)

    missing, mismatched = [], []
    for key, exp in expected.items():
        act = sorted(got.get(key, []), key=lambda r: r[0])
        exp_s = sorted(exp, key=lambda r: r[0])
        if not act:
            missing.append(f"w={key[0]} l={key[1]} slot={key[2]} "
                           f"({len(exp)} expected slice(s)) not drawn")
            continue
        if len(act) != len(exp_s):
            mismatched.append(f"w={key[0]} l={key[1]} slot={key[2]}: "
                              f"{len(act)} slices drawn, {len(exp_s)} expected")
            continue
        for (ets, edur, ename), (ats, adur, aname, _) in zip(exp_s, act):
            if abs(ets - ats) > EPS or abs(edur - adur) > EPS or ename != aname:
                mismatched.append(
                    f"w={key[0]} l={key[1]} slot={key[2]}: "
                    f"expected {ename} @{ets:.3f}+{edur:.3f}, "
                    f"got {aname} @{ats:.3f}+{adur:.3f}")
    rep.fail_samples("every expected span is drawn", missing)
    rep.fail_samples("every drawn span matches its ticks exactly", mismatched)
    print(f"       slot {SPLIT_SLOT}: {split_pairs} (worker,layer) pairs split "
          f"into named ops; {unsplit} drawn as one span (marks absent)")
    print(f"       slot {SEG_SLOT}: {seg_pairs} pairs cut into tile segments; "
          f"{armed} drawn with an arm name (no tile marks)")
    print(f"       slots {sorted(BARRIER_CUTS)}: {bar_cut} pairs cut at a "
          f"barrier mark")

    # ── 11. participation ─────────────────────────────────────────────────
    # The check that was missing. Everything above verifies FIDELITY: that each
    # slice reproduces its tick pair. All of it passed on a trace that drew 168
    # of 248 workers per XCD-layer as running a QKV GEMM they had branched past
    # -- because the ticks were honest and only the NAME was wrong. Fidelity
    # cannot catch that; it never asks what the interval means.
    print("\n11. participation (a gated op is only named when it ran)")
    have_part = any(r[4] is not None for r in rows)
    if not have_part:
        rep.check("log carries p= participation masks", False,
                  "log predates MPK_PHASE_PART -- spans on slots "
                  f"{sorted(GATED_SLOTS)} cannot be distinguished from "
                  "fall-through; overlap derived from this trace is not sound")
    else:
        mislabelled, phantom = [], []
        for e in spans:
            if e.get("cat") != "phase":
                continue
            a_ = e["args"]
            s = a_.get("slot")
            if s not in GATED_SLOTS:
                continue
            row = part_of.get((a_["worker"], a_["layer"]))
            if row is None:
                continue
            ran = (row >> s) & 1
            # On an armed slot the bit is not the authority on whether work
            # happened, and taking it as one is a real misreading rather than a
            # bookkeeping detail. MoE has no `xcd_rank` guard: every worker
            # enters the strided loop, so the bit is set for 8928/8928 rows and
            # means only "reached the phase". Whether any tile survived its
            # early returns is what the arm code records, and for 1981 rows the
            # answer is no. Trusting the bit here would redraw those as MoE
            # workers -- the same class of error as check 11 itself.
            arm_row = arm_of.get((a_["worker"], a_["layer"]))
            if s in ARM_NAMES and arm_row is not None:
                ran = bool((arm_row >> (s * ARM_BITS)) & ARM_MASK)
            # `participated` answers only that question. A tile segment carries
            # `running_tile` for the finer one; conflating the two made every
            # gap segment claim its worker skipped MoE.
            claims = a_.get("participated")
            # Two ways to skip, two names, both legitimate. A rank-guarded
            # phase is skipped by branching past the guard; MoE is skipped by
            # entering the loop and having every tile early-return. The names
            # differ because the reasons do, and flattening them to one string
            # would tell a reader the wrong story about why the worker is idle.
            ok_skips = {SKIP_NAME.format(GATED_SLOTS[s])}
            if s in ARM_NAMES:
                ok_skips.add(ARM_NAMES[s][0])
            if not ran and e["name"] not in ok_skips:
                mislabelled.append(
                    f"w={a_['worker']} l={a_['layer']} slot={s}: drawn as "
                    f"{e['name']!r} but participation bit is clear -- this "
                    f"worker branched past {GATED_SLOTS[s]}")
            if ran and claims is False:
                phantom.append(f"w={a_['worker']} l={a_['layer']} slot={s}: "
                               f"marked non-participating but bit is set")
        rep.fail_samples("no span names an op the worker skipped", mislabelled)
        rep.fail_samples("no span disclaims an op the worker ran", phantom)

        # A fused phase must resolve to one arm. Passing the bit check is not
        # enough: MoE passed it for 100% of rows while drawing every worker as
        # "moe(w13+swiglu+w2)", a label naming three ops no worker performs.
        have_arm = any(r[5] is not None for r in rows)
        if not have_arm:
            rep.check("log carries a= arm codes", False,
                      f"log predates MPK_PHASE_ARM -- slots {sorted(ARM_NAMES)}"
                      " are fused phases whose workers split into arms; per-op "
                      "occupancy read off them is wrong")
        else:
            fused = []
            for e in spans:
                if e.get("cat") != "phase":
                    continue
                a_ = e["args"]
                if a_.get("slot") not in ARM_NAMES:
                    continue
                if e["name"] == SPAN_NAMES[a_["slot"]]:
                    fused.append(
                        f"w={a_['worker']} l={a_['layer']} slot="
                        f"{a_['slot']}: drawn with the fused kernel name "
                        f"{e['name']!r}, which no single worker runs")
            rep.fail_samples("no span wears a fused-kernel name", fused)
            for s_, names in sorted(ARM_NAMES.items()):
                cnt = defaultdict(int)
                durs = defaultdict(list)
                for r in rows:
                    if r[5] is None:
                        continue
                    code = (r[5] >> (s_ * ARM_BITS)) & ARM_MASK
                    cnt[names.get(code, "(unset)")] += 1
                    a, b = r[3][s_ - 1], r[3][s_]
                    if a and b and b >= a:
                        durs[code].append(us(b) - us(a))
                tot = sum(cnt.values())
                for nm, c in sorted(cnt.items(), key=lambda kv: -kv[1]):
                    med = sorted(durs[
                        next(k for k, v in names.items() if v == nm)]) \
                        if nm in names.values() else []
                    mtxt = f"  median {med[len(med) // 2]:8.2f} us" if med \
                        else ""
                    print(f"       slot {s_:>2} {nm:<26} {c:>5}/{tot} "
                          f"({100.0 * c / max(tot, 1):.1f}%){mtxt}")

                # The arm code is a SET, and a set has an arithmetic
                # consequence the timestamps can refute: a worker that ran both
                # arms did strictly more work than one that ran either alone,
                # so its span must be longer. This is the check that caught the
                # first version of the instrument, which assigned the arm
                # instead of OR-ing it and so reported the 22 busiest ranks per
                # XCD as pure W2 -- a 1.4 us label on a 19.8 us span. The
                # distribution above looked plausible; only the durations
                # disagreed with it.
                def med_of(code):
                    v = sorted(durs.get(code, []))
                    return v[len(v) // 2] if v else None

                both, solo = med_of(3), [med_of(1), med_of(2)]
                solo = [x for x in solo if x is not None]
                if both is not None and solo:
                    bad = [f"slot {s_}: workers marked "
                           f"{names[3]!r} have median span {both:.2f} us, "
                           f"not longer than the {x:.2f} us of a single-arm "
                           f"worker -- the arm code disagrees with the clock"
                           for x in solo if both <= x]
                    rep.fail_samples(
                        "a both-arms worker outlasts a single-arm worker", bad)

            # Two instruments, one fact. The arm BITS say which arms a worker
            # ran; the per-tile MARKS say when it ran them. They are written at
            # the same instruction, so they must agree cell for cell -- and
            # when they did not, the disagreement was the bug: the bits were
            # set where the arm was *decided*, upstream of three early returns,
            # so 2304 cells claimed W13 while 324 reached W13 compute. Nothing
            # about the bits alone looked wrong. Only the second instrument
            # showed it, which is the argument for having one.
            ARM_MARKS = {1: (10, 11), 2: (12, 13)}
            mism = []
            for r in rows:
                if r[5] is None:
                    continue
                code = (r[5] >> (8 * ARM_BITS)) & ARM_MASK
                seen = {c for c, _ in (subs.get((r[0], r[2])) or [])}
                for bit, (beg, end) in ARM_MARKS.items():
                    has_bit = bool(code & bit)
                    has_mark = beg in seen and end in seen
                    if has_bit != has_mark:
                        mism.append(
                            f"w={r[0]} l={r[2]} arm bit {bit}: "
                            f"bit={'set' if has_bit else 'clear'} but marks "
                            f"{beg}/{end} {'present' if has_mark else 'absent'}")
            rep.fail_samples("arm bits agree with the per-tile marks", mism)

        # Report the split per gated slot: this is the number the post's
        # concurrency claims rest on, so print it rather than assert it.
        for s, nm in sorted(GATED_SLOTS.items()):
            ran = sum(1 for r in rows if r[4] is not None and (r[4] >> s) & 1)
            tot = sum(1 for r in rows if r[4] is not None)
            print(f"       slot {s:>2} {nm:<6} {ran:>5}/{tot} rows ran it "
                  f"({100.0 * ran / max(tot, 1):.1f}%); {tot - ran} reached "
                  f"the guard and fell through")
            # Printing 100% for MoE and stopping there would be the headline
            # bug in summary form: the phase has no rank guard, so nothing
            # falls through and the bit is set for every row. The arms are
            # where its idleness actually shows.
            if s in ARM_NAMES:
                idle = sum(1 for r in rows if r[5] is not None
                           and not (r[5] >> (s * ARM_BITS)) & ARM_MASK)
                at = sum(1 for r in rows if r[5] is not None)
                print(f"       {'':>7} {'':<6} {'':>5} {'':<5} "
                      f"...but no rank guard: {idle}/{at} "
                      f"({100.0 * idle / max(at, 1):.1f}%) entered the loop "
                      f"and every tile early-returned")
        print(f"       {gated_out} span(s) expected as non-participant")

    # ── 5. split tiling ───────────────────────────────────────────────────
    print("\n5. split tiling (segments cover the parent slot with no gap)")
    gaps, ovl, extent = [], [], []
    for w, x, l, ts, pt, am in rows:
        key = (w, l, SPLIT_SLOT)
        act = sorted(got.get(key, []), key=lambda r: r[0])
        if len(act) < 2:
            continue
        a, b = ts[SPLIT_SLOT - 1], ts[SPLIT_SLOT]
        for (ts0, d0, n0, _), (ts1, d1, n1, _) in zip(act, act[1:]):
            end0 = ts0 + d0
            if end0 < ts1 - EPS:
                gaps.append(f"w={w} l={l}: {n0} ends {end0:.3f}, "
                            f"{n1} starts {ts1:.3f}")
            if end0 > ts1 + EPS:
                ovl.append(f"w={w} l={l}: {n0} ends {end0:.3f} after "
                           f"{n1} starts {ts1:.3f}")
        if abs(act[0][0] - us(a)) > EPS or \
           abs(act[-1][0] + act[-1][1] - us(b)) > EPS:
            extent.append(f"w={w} l={l}: segments span "
                          f"{act[0][0]:.3f}..{act[-1][0] + act[-1][1]:.3f}, "
                          f"slot is {us(a):.3f}..{us(b):.3f}")
        for _, _, n, _ in act:
            if n not in SPLIT_LABELS:
                ovl.append(f"w={w} l={l}: unexpected segment name {n!r}")
    rep.fail_samples("no gap between consecutive segments", gaps)
    rep.fail_samples("no overlap between consecutive segments", ovl)
    rep.fail_samples("segments span exactly the parent slot", extent)

    # ── 6. submark containment ────────────────────────────────────────────
    print("\n6. sub-mark containment and ordering")
    outside, misordered = [], []
    for (w, l), marks in subs.items():
        row = next((r for r in rows if r[0] == w and r[2] == l), None)
        if row is None:
            outside.append(f"w={w} l={l}: PTRACES with no PTRACEW row")
            continue
        ts = row[3]
        at = {}
        for code, tick in marks:
            at.setdefault(code, tick)
        p1 = [at[c] for c in (6, 7, 8, 9) if c in at]
        if p1:
            a, b = ts[0], ts[1]
            for c in (6, 7, 8, 9):
                if c in at and not (a <= at[c] <= b):
                    outside.append(f"w={w} l={l}: P1 mark {c} at {at[c]} "
                                   f"outside slot 1 [{a},{b}]")
            if p1 != sorted(p1):
                misordered.append(f"w={w} l={l}: P1 marks not ascending: {p1}")
    rep.fail_samples("P1 marks lie inside the slot-1 window", outside)
    rep.fail_samples("P1 marks ascend 6 -> 7 -> 8 -> 9", misordered)

    # ── 7. no same-row overlap ────────────────────────────────────────────
    print("\n7. per-row sequentiality (a worker cannot be in two ops at once)")
    by_tid = defaultdict(list)
    for e in spans:
        by_tid[(e["pid"], e["tid"])].append(e)
    row_ovl = []
    for (pid, tid), es in by_tid.items():
        if tid >= TID_ASYNC:
            continue
        es.sort(key=lambda e: e["ts"])
        for e0, e1 in zip(es, es[1:]):
            if e0["ts"] + e0["dur"] > e1["ts"] + EPS:
                row_ovl.append(
                    f"XCD{pid} w={tid}: {e0['name']} "
                    f"@{e0['ts']:.3f}+{e0['dur']:.3f} overlaps "
                    f"{e1['name']} @{e1['ts']:.3f}")
    rep.fail_samples("worker rows are strictly sequential", row_ovl)

    # ── 8. detail-row containment ─────────────────────────────────────────
    print("\n8. detail-row slices lie inside their own layer")
    det_bad = []
    bounds = {(w, l): (ts[0], ts[nslots - 1]) for w, _, l, ts, pt, am in rows}
    ndetail = 0
    for e in spans:
        if e.get("cat") != "subphase":
            continue
        ndetail += 1
        a_ = e["args"]
        b = bounds.get((a_["worker"], a_["layer"]))
        if b is None or not b[0] or not b[1]:
            det_bad.append(f"w={a_['worker']} l={a_['layer']}: no layer bounds")
            continue
        lo, hi = us(b[0]), us(b[1])
        if e["ts"] < lo - EPS or e["ts"] + e["dur"] > hi + EPS:
            det_bad.append(f"w={a_['worker']} l={a_['layer']} {e['name']}: "
                           f"{e['ts']:.3f}..{e['ts'] + e['dur']:.3f} outside "
                           f"layer {lo:.3f}..{hi:.3f}")
        if e["tid"] != a_["worker"] + TID_ASYNC:
            det_bad.append(f"detail slice tid {e['tid']} != worker "
                           f"{a_['worker']} + {TID_ASYNC}")
    rep.fail_samples("detail slices inside their layer, on the right row",
                     det_bad)

    # Containment says nothing about whether the RIGHT slices are there. A DMA
    # that is never drawn passes every check above by having nothing to
    # disagree with, which is precisely how two of the four went unmeasured. So
    # derive the expected set from the marks and compare it exactly.
    det_exp = defaultdict(list)          # (w,l) -> [(ts_us, dur_us, name)]
    for w, x, l, ts, pt, am in rows:
        at = {}
        for code, tick in (subs.get((w, l)) or []):
            at.setdefault(code, tick)
        for c0, c1, label in SUB_SPANS:
            req = SUB_REQUIRES.get(label)
            if req is not None and req not in at:
                continue
            if c0 in at and c1 in at and at[c1] >= at[c0]:
                det_exp[(w, l)].append(
                    (us(at[c0]), us(at[c1]) - us(at[c0]), label))
        c_open, n_open = SUB_OPEN_ENDED
        if c_open in at and ts[nslots - 1] and ts[nslots - 1] >= at[c_open]:
            det_exp[(w, l)].append(
                (us(at[c_open]), us(ts[nslots - 1]) - us(at[c_open]), n_open))
    det_got = defaultdict(list)
    for e in spans:
        if e.get("cat") == "subphase":
            a_ = e["args"]
            det_got[(a_["worker"], a_["layer"])].append(
                (e["ts"], e["dur"], e["name"]))
    det_diff = []
    for key in set(det_exp) | set(det_got):
        exp_s = sorted(det_exp.get(key, []))
        act_s = sorted(det_got.get(key, []))
        if exp_s != act_s:
            det_diff.append(
                f"w={key[0]} l={key[1]}: expected "
                f"{[(round(t, 3), round(d, 3), n) for t, d, n in exp_s]}, "
                f"got {[(round(t, 3), round(d, 3), n) for t, d, n in act_s]}")
    rep.fail_samples("detail row draws exactly the DMA spans the marks imply",
                     det_diff)
    ndma = defaultdict(int)
    for v in det_exp.values():
        for _, _, n in v:
            ndma[n] += 1
    for n in sorted(ndma):
        print(f"       {ndma[n]:6d}  {n}")
    # A detail row must never carry a name that also appears on a worker row.
    dup_names = {e["name"] for e in spans if e.get("cat") == "subphase"} & \
                {e["name"] for e in spans if e.get("cat") == "phase"}
    rep.fail_samples("no op drawn on both the worker row and the detail row",
                     sorted(dup_names))
    print(f"       {ndetail} detail slices")

    # ── 9. inter-layer continuity ─────────────────────────────────────────
    print("\n9. inter-layer spans")
    il = [e for e in spans if e["name"] == "inter_layer"]
    exp_il, il_bad = 0, []
    prev = {}
    for w, x, l, ts, pt, am in sorted(rows, key=lambda r: (r[0], r[2])):
        if w in prev and ts[0] and prev[w] <= ts[0]:
            exp_il += 1
        if ts[nslots - 1]:
            prev[w] = ts[nslots - 1]
    rep.check("inter_layer slice count matches layer transitions",
              len(il) == exp_il, f"{len(il)} drawn, {exp_il} expected")
    for e in il:
        if e["dur"] < -EPS:
            il_bad.append(f"negative inter_layer dur {e['dur']}")
    rep.fail_samples("inter_layer durations are non-negative", il_bad)

    # ── 10. time base ─────────────────────────────────────────────────────
    print("\n10. time base")
    ts_min = min(e["ts"] for e in spans)
    rep.check("earliest slice sits at t=0 (t0 rebase applied)",
              abs(ts_min) < EPS, f"min ts = {ts_min}")
    granular = [e for e in spans
                if abs(round(e["ts"] * 100) - e["ts"] * 100) > 1e-6]
    rep.fail_samples(f"all timestamps are multiples of the {tick_ns} ns tick",
                     [f"{e['name']} ts={e['ts']}" for e in granular])
    neg = [f"{e['name']} w={e['tid']} dur={e['dur']}"
           for e in spans if e["dur"] < 0]
    rep.fail_samples("no negative durations", neg)

    # ── metadata sanity ───────────────────────────────────────────────────
    print("\n11. metadata")
    named = {(e["pid"], e["tid"]) for e in meta if e["name"] == "thread_name"}
    used = {(e["pid"], e["tid"]) for e in spans}
    rep.fail_samples("every row that has slices also has a name",
                     [f"XCD{p} tid={t}" for p, t in sorted(used - named)])
    xcd_named = {e["pid"] for e in meta if e["name"] == "process_name"}
    rep.fail_samples("every XCD with slices is named",
                     [f"XCD{p}" for p in sorted({e['pid'] for e in spans}
                                                - xcd_named)])

    # ── per-worker table ──────────────────────────────────────────────────
    print("\n12. per-worker reconciliation "
          "(log tick span vs drawn slice extent)")
    per_w = defaultdict(lambda: [0, 0.0, None, None])
    for w, x, l, ts, pt, am in rows:
        nz = [t for t in ts if t]
        if not nz:
            continue
        e = per_w[w]
        e[2] = min(e[2], us(min(nz))) if e[2] is not None else us(min(nz))
        e[3] = max(e[3], us(max(nz))) if e[3] is not None else us(max(nz))
    drawn = defaultdict(lambda: [0, None, None])
    for e in spans:
        if e.get("cat") != "phase" or e["name"] == "inter_layer":
            continue
        d = drawn[e["tid"]]
        d[0] += 1
        d[1] = e["ts"] if d[1] is None else min(d[1], e["ts"])
        d[2] = e["ts"] + e["dur"] if d[2] is None else \
            max(d[2], e["ts"] + e["dur"])
    wbad = []
    for w in sorted(per_w):
        lo, hi = per_w[w][2], per_w[w][3]
        d = drawn.get(w)
        if d is None:
            wbad.append(f"w={w}: {hi - lo:.3f} us of ticks, no slices drawn")
            continue
        if abs(d[1] - lo) > EPS or abs(d[2] - hi) > EPS:
            wbad.append(f"w={w}: ticks {lo:.3f}..{hi:.3f}, "
                        f"slices {d[1]:.3f}..{d[2]:.3f}")
    rep.fail_samples("every worker's drawn extent equals its tick extent",
                     wbad)
    if args.verbose:
        print(f"    {'worker':>6} {'slices':>7} {'first us':>10} "
              f"{'last us':>10} {'span us':>9}")
        for w in sorted(per_w):
            d = drawn.get(w, [0, 0, 0])
            print(f"    {w:>6} {d[0]:>7} {d[1]:>10.3f} {d[2]:>10.3f} "
                  f"{d[2] - d[1]:>9.3f}")

    print()
    npass = sum(1 for _, ok, _, _ in rep.checks if ok)
    print(f"{npass}/{len(rep.checks)} checks passed, "
          f"{len(spans)} slices verified against {len(rows)} log rows")
    if rep.failures:
        print("\nFAILED:")
        for name, detail in rep.failures:
            print(f"  - {name}  {detail}")
        return 1
    print("trace is faithful to the log.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
