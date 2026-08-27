#!/usr/bin/env python3
"""Convert Fleet's [PTRACEW] dump into a Perfetto trace.

Consumes a run built with -DMPK_PHASE_SLOTS -DMPK_PHASE_TRACE. Each [PTRACEW]
line is one (worker, layer) with 12 absolute s_memrealtime ticks; this turns
them into Chrome Trace Format complete events, which ui.perfetto.dev opens
directly.

Why this exists alongside summarize_phase_slots.py: that script reports means,
and a mean cannot show overlap. Two workers each averaging 5 us in MoE tell you
nothing about whether they ran concurrently or back to back -- and concurrency
is the entire claim a megakernel makes. Only absolute timestamps on a shared
clock can show it, which is what s_memrealtime is: a constant-rate counter,
common to every XCD on the die, so timestamps from different workers are
directly comparable without any cross-calibration.

  Usage: phase_slots_to_perfetto.py <run.log> -o trace.json
         then open trace.json at https://ui.perfetto.dev

Layout in the UI: one process per XCD, one thread per worker, one slice per
phase. Threads sort by worker id, so a barrier appears as a wall of slices
ending at the same x and the stragglers are the ones that overhang it.
"""
import argparse
import json
import re
import sys
from collections import defaultdict

# Names match summarize_phase_slots.py's SLOT_NAMES so the two instruments can
# be read side by side. Slot i's span runs from timestamp i-1 to timestamp i,
# which is why slot 0 -- the inter-layer span -- has no slice of its own here:
# its start is the *previous* layer's slot 11, and it is emitted below by
# looking back rather than by pairing within a row.
# Slot 1 is named for the whole fused kernel, not for the GEMM inside it. It
# spans ResAdd -> RMSNorm -> quant -> MFMA -> bias -> RoPE -> KV-cache update,
# and calling it "qkv_gemm" invited exactly one misreading: that the P1
# sub-spans on the detail row (ResAdd, RMSNorm) were running *in parallel with*
# the GEMM on the same worker. They are strictly nested inside this span and
# strictly sequential -- the GEMM is the tail of it, after RMSNorm retires.
SPAN_NAMES = [
    None,                     # 0: layer entry (start marker; see above)
    "qkv_fused(resadd..kvupd)",  # 1: ts[0] -> ts[1]
    "qkv_epoch_barrier",      # 2
    "attention+merge",        # 3
    # Slot 3 and slot 4 are marked on ADJACENT SOURCE LINES (:856/:857) with
    # no code between them, so this span is not a phase -- it is one
    # s_memrealtime read plus its store, which makes it the trace's own noise
    # floor, measured on the same workers in the same run rather than
    # calibrated separately. 0.52 us median (min 0.44, p99 1.16) over 8927
    # samples; 5.68 ms total, 1.0% of worker-time. Any span at or below it is
    # reporting the instrument, not the kernel -- and 17.7% of all phase spans
    # are. Named for what it is, because "(marker)" read as a phase.
    "instrument floor (mark 3->4, no code between)",  # 4
    "attn_release_wait",      # 5
    "oproj+rmsnorm+router",   # 6
    "topk_wait",              # 7
    "moe(w13+swiglu+w2)",     # 8
    "layer_arrive+fanout+pf",  # 9
    "layer_gate_poll",        # 10
    "layer_exit",             # 11
]

# Colour by kind, so a screenshot reads without a legend. Perfetto accepts the
# Catapult reserved colour names.
COLOR = {
    "qkv_fused(resadd..kvupd)": "thread_state_running",
    "attention+merge": "thread_state_running",
    "oproj+rmsnorm+router": "thread_state_running",
    "moe(w13+swiglu+w2)": "thread_state_running",
    "qkv_epoch_barrier": "thread_state_iowait",
    "attn_release_wait": "thread_state_iowait",
    "topk_wait": "thread_state_iowait",
    "layer_gate_poll": "thread_state_iowait",
    "inter_layer": "grey",
}

# Sub-phase marks. These exist because the 12 slots mark the *boundaries
# between* phases, so every span they can draw is exactly one phase wide -- the
# instrument can only ever render phases running back to back. The two overlaps
# that pay for themselves both live strictly *inside* one phase:
#
#   * the next layer's QKV weight DMA, issued into the Phase 9 barrier spin
#     (between slots 9 and 10),
#   * the W13 tile-0 weight loads issued before FP8 quant and drained after it
#     (both between slots 7 and 8).
#
# These go on a separate detail track per worker; the sequential sub-phases of
# slot 1 do NOT, they split that phase in place on the worker's own row (see
# SLOT_SPLIT below). The difference is what the marks bracket, not a style
# preference:
#
#   * slot 1's marks are handoffs between ops that run one after another, so
#     they tile the phase edge to edge and cut it cleanly in place;
#   * these bracket a DMA *in flight*, which by construction overlaps the
#     compute around it -- covering that compute is the entire point.
#
# An overlapping span cannot share the row. Chrome Trace Format treats
# same-track slices as a strict stack, and these do not nest: measured on this
# trace, "qkv weight DMA in flight" spans layer_arrive+fanout+pf, then
# layer_gate_poll, then layer_exit -- three sibling phases. There is no way to
# draw that on the parent row without either splitting the span (which would
# erase the overlap being measured) or emitting a malformed stack.
#
# So the separate row here IS a claim about concurrency, unlike the rest of
# this trace: same worker, same timeline, async engine running under the
# compute. That is also why these spans never feed the cross-phase overlap
# numbers in phase_overlap.py -- they are intra-worker, not two phases at once.
# Every name here says "weight DMA", because the bare token "W13" already names
# a compute arm on the worker's own row ("moe W13+SwiGLU"). A detail-row span
# called "W13 weights in flight" next to a main-row span called "moe W13+SwiGLU"
# reads as two views of the same work; it is not. These are the async weight
# loads, and the only thing they share with the arm is the weight matrix.
SUB_NAMES = {
    1: "qkv weight DMA in flight",
    2: "qkv weight DMA drain",
    3: "W13 weight DMA in flight",
    4: "W13 quant (cover work)",
    5: "W13 weight DMA drain",
    6: "P1 entry",
    7: "P1 resadd done",
    8: "P1 rmsnorm done",
    9: "P1 gemm start",
    16: "qkv_epoch barrier entry",
    17: "attn_release barrier entry",
    18: "oproj_hier barrier entry",
    19: "routing_ready barrier entry",
    20: "layer_release barrier entry",
    21: "idle: no work in this phase",
    22: "oproj weight DMA issue",
    23: "oproj weight DMA drain",
    24: "W2 weight DMA issue",
    25: "W2 weight DMA drain",
    26: "oproj weight DMA done (post-waitcnt)",
    27: "W2 weight DMA done (post-waitcnt)",
}

# ── Barrier waits, cut out of the phase span that contains them ────────────
#
# A phase slot is named for the op it surrounds, so a barrier inside one is
# billed to that op. That is how "topk_wait" came to cover two different
# barriers -- the Phase 7a' O-proj gate and the routing_ready poll -- for the
# ranks that pay both, and why their 14 us read as TopK latency when roughly
# half is the earlier gate.
#
# Each entry cuts its parent slot at the mark: everything before is whatever
# the slot was already doing, everything from the mark to the slot's end is
# the wait. Scope is in the NAME, not a tooltip, because it changes what a
# stall means. XCD-LOCAL = 31 workers on one die rendezvous, so a stall is a
# local imbalance and rebalancing that XCD fixes it. GLOBAL = all 8 XCDs
# couple through one counter, so the slowest XCD sets everyone's release and
# local rebalancing cannot help. Both were confirmed twice, from the source
# (which atomic, whose flag, who fans out) and from the trace (a global
# release lands on all 8 XCDs within ~0.4 us; a local one does not).
#
# `head` names the pre-barrier remainder. Where the slot's own name already
# describes that remainder the head is None and the parent name is kept.
BARRIER_CUTS = {
    2: [(16, "qkv_epoch barrier (XCD-LOCAL)")],
    5: [(17, "attn_release barrier (GLOBAL)")],
    # Two cuts in one slot, in source order. Ranks < oproj_topk_tiles_per_xcd
    # emit only code 19 and get a single wait; the ranks past that guard emit
    # 18 then 19 and get both, which is the split this slot exists to show.
    7: [(18, "oproj_hier barrier (GLOBAL)"),
        (19, "routing_ready/TopK barrier (GLOBAL)")],
    10: [(20, "layer_release barrier (GLOBAL)"),
         (21, "idle: not in the layer gate")],
}
BARRIER_HEAD = {
    2: "qkv post-GEMM drain",
    5: "attn epilogue",
    7: "moe prologue (pre-barrier)",
    10: "layer release fan-out",
}
BARRIER_LABELS = {lab for cuts in BARRIER_CUTS.values() for _, lab in cuts}
BARRIER_LABELS |= set(BARRIER_HEAD.values())

# A head is work; everything from a mark onward is not. Both kinds of "not"
# are shown, and they are kept apart: a barrier wait is a worker spinning on a
# flag it expects to be set, while the idle label is a worker that fell
# through the gate's guard entirely and has no work in this phase at all. The
# first is a synchronisation cost and the second is a load-balance cost, so
# colouring them the same would hide which one a given row is paying.
BARRIER_IDLE_LABELS = {"idle: not in the layer gate"}
BARRIER_WAIT_LABELS = ({lab for cuts in BARRIER_CUTS.values()
                        for _, lab in cuts} - BARRIER_IDLE_LABELS)
BARRIER_COLOR = {lab: "thread_state_iowait" for lab in BARRIER_WAIT_LABELS}
BARRIER_COLOR.update({lab: "thread_state_running"
                      for lab in BARRIER_HEAD.values()})
BARRIER_COLOR["idle: not in the layer gate"] = "grey"

# Spans built from sub-marks: (start_code, end_code, label). A sub-mark on its
# own is an instant; these say which pairs bound a span and what to call it.
# QKV prefetch has no end mark by construction -- the consumer is the next
# layer's Phase 1, which skips its DMA rather than draining it -- so its end is
# the layer's own exit, handled separately below.
SUB_SPANS = [
    (3, 4, "W13 weight DMA in flight"),
    (4, 5, "W13 weight DMA drain (uncovered)"),
    # The kernel issues FOUR weight DMAs into a wait, and for a long time this
    # list showed two. The other two were described in kernel comments as
    # overlapping their polls -- "~3us overlap instead of serial" for W2, "so
    # DMA runs in the background during the spin-wait" for O-proj -- without
    # either being measured. Same class of error as naming a span from its
    # position: a claim read off the source rather than off the clock.
    #
    # Each is issue -> drain -> done, straddling the s_waitcnt that consumes
    # the DMA. issue->drain is how long the load HAD to finish; drain->done is
    # the wait itself, and that second span is the one that answers the
    # question. A wide in-flight window is equally consistent with a fully
    # hidden DMA and with one that stalled at the end of it, so the pair is
    # drawn rather than the first alone -- same shape as W13's 3->4->5.
    (22, 23, "oproj weight DMA in flight"),
    (23, 26, "oproj weight DMA drain (uncovered)"),
    (24, 25, "W2 weight DMA in flight"),
    (25, 27, "W2 weight DMA drain (uncovered)"),
    # Phase 1 cut at its two internal handoffs. The 12 slots draw the whole
    # fused QKV op as one slice, so these are the only way to see whether
    # ResAdd -> RMSNorm and RMSNorm -> GEMM cost anything.
]

# A span that may only be drawn when a THIRD code is also present.
#
# The O-proj issue sits inside `if (xcd_rank < oproj_topk_tiles_per_xcd ...)`
# but its drain does not -- the s_waitcnt is on the common path, so all 8927
# rows take mark 23 while only 6624 take mark 22. Pairing 23->26 unconditionally
# drew 2303 spans a layer named "oproj weight DMA drain" on workers that issued
# no O-proj DMA at all: a span named for its position in the instruction stream
# rather than for work the worker did, which is the exact bug this file is
# written against. Their s_waitcnt is real and may well be nonzero, but it is
# draining somebody else's traffic and naming it O-proj would be a fabrication.
SUB_REQUIRES = {
    "oproj weight DMA drain (uncovered)": 22,
}

# Codes consumed by SLOT_SPLIT / SLOT_SEGMENTS below. They are drawn as named
# ops on the worker's own row, so re-drawing them on the detail row would
# duplicate the same measurement in two places and invite reading it as
# concurrency.
SPLIT_CODES = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15}

# ── Splitting a phase span into its constituent ops ──────────────────────
#
# Slot 1 is one span covering a six-op fused kernel, so the row says
# "qkv_gemm" over a stretch of time in which the GEMM is only the tail. Naming
# it better does not fix that -- the span is genuinely several ops, and the
# only honest rendering cuts it where the ops actually change.
#
# The sub-marks already record those boundaries, so when a worker has them we
# emit the segments IN PLACE on the worker's own row instead of one span.
# Same row, same total extent, no gaps: what changes is that each op gets its
# own named slice. A worker missing the marks falls back to the single span.
#
# (start_code, end_code, label). The head segment runs from the slot's own
# start to the first mark, and the tail from the last mark to the slot's end;
# both are derived below rather than listed, since their bounds are the phase
# timestamps rather than sub-marks.
SLOT_SPLIT = {
    1: {
        "head": "resadd_prologue(lds+dma issue)",
        "spans": [
            (6, 7, "ResAdd(f32)"),
            (7, 8, "RMSNorm"),
            (8, 9, "FP8 quant + weight drain"),
        ],
        "tail_from": 9,
        "tail": "QKV GEMM(MXFP4) + bias + RoPE + KV-upd",
    },
}

SPLIT_COLOR = {
    "resadd_prologue(lds+dma issue)": "thread_state_unknown",
    "ResAdd(f32)": "thread_state_running",
    "RMSNorm": "thread_state_runnable",
    "FP8 quant + weight drain": "thread_state_iowait",
    "QKV GEMM(MXFP4) + bias + RoPE + KV-upd": "thread_state_running",
    "moe W13+SwiGLU": "thread_state_running",
    "moe W2 (down proj)": "thread_state_iowait",
    # Setup and loop-exit are grey; the barrier wait gets the same colour as
    # every other stall in the trace, because that is what it is.
    "moe prologue (decode+gather)": "thread_state_unknown",
    "W13->W2 barrier wait": "thread_state_uninterruptible",
    "W2 prep (quant+pf issue)": "thread_state_runnable",
    "moe W2 compute": "thread_state_iowait",
    "moe tile setup (decode+gather+pf)": "thread_state_unknown",
    "moe loop exit": "thread_state_unknown",
}

# ── Slot 8: segments, not a tiling ──────────────────────────────────────────
#
# Slot 1 splits into segments that tile it end to end, because its sub-marks
# are handoffs: one op stops exactly where the next starts. Slot 8 is not like
# that. Its marks bracket TILES, the strided loop gives a worker one or two of
# them, and between two tiles the worker is doing loop overhead and barrier
# waiting that belongs to neither arm. So the segments here are islands, and
# what separates them is named rather than absorbed into whichever arm happens
# to sit next to it.
#
# Pairs may be absent independently: a W13-only worker has 10/11 and not 12/13,
# a W2-only worker the reverse, and ranks 0-21 have all four. That is the same
# fact the arm bits record, arrived at independently -- which is the point. If
# the marks and the bits disagree, one of them is wrong, and the verifier says
# so rather than picking a winner.
#
# There is no SwiGLU segment on purpose; see the comment on MPK_SUB_MOE_W13_BEG
# in persistent_kernel.cuh. The epilogue is fused into the MFMA loop per
# tile-iteration, so no pair of marks bounds it without inventing a boundary.
# The gaps are named by what BOUNDS them, not by where they sit, because one
# name for all three read as "the same wait, three times" and they are three
# different things. Checked against the source rather than assumed:
#
#   before a tile (either arm) -- per-tile SETUP, and note that this is between
#     two kernel *calls*: the caller loops `for (moe_t = xcd_rank; ...)` and
#     invokes the whole kernel per tile, so the gap is entry + tile decode (two
#     dependent HBM loads: d_mask, then d_routing indexed by what d_mask
#     returned) + token compaction + __syncthreads + LDS/pointer setup + the
#     buffer_load_lds weight prefetch issue. Both arms pay the same sequence,
#     which is why the two gaps come out nearly equal (4.36 vs 3.76 us median).
#   after the last tile -- the strided loop falling out. ~0.3 us.
#
# What these gaps are NOT: the W13->W2 barrier. That poll is at
# gang_moe_fused_mxfp4_mi300.cuh:2793, which is INSIDE the W2 span
# (MPK_SUB_MOE_W2_BEG :2526 .. MPK_SUB_MOE_W2_END :3461), not before it. An
# earlier version of this file named the second gap "W13->W2 barrier wait" on
# the strength of its position between the arms; the line numbers say
# otherwise. The barrier cost is inside the W2 compute span and is not
# separable from it here -- deliberately, since :2777 issues the weight
# prefetch before the poll so the HBM latency hides in the wait.
#
# Marks 14/15 now bracket that poll, so the W2 span splits in place into
# prep / barrier wait / compute. These ARE handoffs, so unlike the tile
# islands above they tile their parent edge to edge.
SLOT_SEGMENTS = {
    8: {
        "pairs": [(10, 11, "moe W13+SwiGLU"),
                  (12, 13, "moe W2 (down proj)")],
        "gap_before": {"moe W13+SwiGLU": "moe tile setup (decode+gather+pf)",
                       "moe W2 (down proj)": "moe tile setup (decode+gather+pf)"},
        "gap_tail": "moe loop exit",
        # A tile span may itself split, edge to edge, on interior marks. Used
        # for W2, whose barrier poll was otherwise being read as compute.
        "inner": {
            "moe W2 (down proj)": {
                "cuts": [(14, "W2 prep (quant+pf issue)"),
                         (15, "W13->W2 barrier wait")],
                "tail": "moe W2 compute",
            },
        },
    },
}

# The tile labels, as opposed to the gap labels. Derived rather than restated
# so a new arm cannot be added to the pairs above and silently be classified
# as a gap.
SEG_TILE_LABELS = ({lbl for spec in SLOT_SEGMENTS.values()
                    for _, _, lbl in spec["pairs"]}
                   | {sub for spec in SLOT_SEGMENTS.values()
                      for isp in spec.get("inner", {}).values()
                      for _, sub in isp["cuts"]}
                   | {isp["tail"] for spec in SLOT_SEGMENTS.values()
                      for isp in spec.get("inner", {}).values()})

SUB_COLOR = {
    "W13 weight DMA in flight": "thread_state_running",
    "W13 quant (cover work)": "thread_state_runnable",
    "W13 weight DMA drain (uncovered)": "thread_state_uninterruptible",
    "qkv weight DMA in flight": "thread_state_running",
    # Same colour as the other two in-flight spans: they are the same thing --
    # an async engine moving weights while the compute row above does something
    # else. Colour by what the span IS, so a reader can pick the four DMAs out
    # of the detail row without reading four names.
    "oproj weight DMA in flight": "thread_state_running",
    "W2 weight DMA in flight": "thread_state_running",
    # And the same colour W13's uncovered drain already wears, for the same
    # reason: this is the part of the load nothing hid.
    "oproj weight DMA drain (uncovered)": "thread_state_uninterruptible",
    "W2 weight DMA drain (uncovered)": "thread_state_uninterruptible",
    "P1 resadd+ssq": "thread_state_running",
    "P1 rmsnorm reduction": "thread_state_runnable",
    "P1 quant+drain": "thread_state_uninterruptible",
}

# Sub-phase rows are offset by this much so each worker gets a second track
# directly below its own. Larger than any worker id, so w and w + TID_ASYNC
# can never collide.
TID_ASYNC = 100000

HDR_RE = re.compile(r"^\[PTRACE\] slots=(\d+) layers=(\d+) tick_ns=(\d+)\s*$")
SUB_RE = re.compile(r"^\[PTRACES\] w=(\d+) l=(\d+) n=(\d+)((?: \d+:\d+)+)\s*$")
# Strict for the same reason summarize_phase_slots.py is: the host's stdout
# interleaves with the device printf stream, and a spliced line must be
# dropped and counted rather than parsed with missing fields defaulted to 0.
ROW_RE = re.compile(
    r"^\[PTRACEW\] w=(\d+) x=(-?\d+) l=(\d+)(?: p=(\d+))?(?: a=(\d+))?"
    r"((?: \d+)+)\s*$")

# Slots whose span is only real work when the participation bit is set. Every
# MPK_PHASE_MARK sits outside its phase's `if (xcd_rank < ...)` guard -- it has
# to, or non-participants would have no timestamp and break the 12-ascending-
# marks invariant. So a skipping worker still emits a tick pair for the phase,
# spanning nothing but a not-taken branch.
#
# Drawing those as the phase is how this trace once showed 248 workers inside
# "qkv_fused" when 80 ran it and 168 fell through in ~1.1 us -- and how a
# 3.02 us "MoE || QKV overlap" appeared that does not exist. The device now
# records who actually ran each phase (MPK_PHASE_PART, set from inside the
# guarded body); when that field is present these spans are renamed and
# recoloured rather than dropped, because the time is real -- the worker was
# there, it just was not doing the op the slot is named for.
GATED_SLOTS = {1: "qkv", 3: "attn", 6: "oproj", 8: "moe"}
SKIP_NAME = "(skipped: not a {} participant)"
SKIP_COLOR = "grey"

# ── Phases whose participants do not all do the same thing ──────────────────
#
# The participation bit is sufficient for a phase guarded by `if (xcd_rank <
# K)`: inside the guard every worker runs the same op. It is not sufficient for
# MoE, and treating it as such left this bug standing after the bit fixed the
# other three slots.
#
# MoE has no rank guard, so every worker set the bit and the slot read as 100%
# participating -- while the workers underneath were doing different things.
# *Each tile* is W13 or W2, never both (the kernel: "there is no single timeline
# covering W13 then W2"), but a *worker* is not a tile: the strided loop runs
# moe_total_tiles_per_xcd tiles across workers_per_xcd workers, so with 53 tiles
# over 31 workers the low ranks take two and run one arm then the other. The
# device therefore records a SET of arms, not one arm -- see the comment on
# mpk_phase_arm(). Codes below are that set: 3 is the common case for the ranks
# doing the most work, not an error value.
ARM_NONE, ARM_W13, ARM_W2, ARM_BOTH = 0, 1, 2, 3
ARM_BITS, ARM_MASK = 2, 3
ARM_NAMES = {
    8: {ARM_NONE: "(skipped: no moe tile)",
        ARM_W13: "moe W13+SwiGLU",
        ARM_W2: "moe W2 (down proj)",
        ARM_BOTH: "moe W13+SwiGLU, then W2"},
}
ARM_COLOR = {
    "moe W13+SwiGLU": "thread_state_running",
    "moe W2 (down proj)": "thread_state_iowait",
    "moe W13+SwiGLU, then W2": "thread_state_runnable",
    "(skipped: no moe tile)": SKIP_COLOR,
}


def parse(path):
    hdr = None
    rows = []
    subs = {}
    corrupt = 0
    saw_end = False
    with open(path, errors="replace") as fh:
        for line in fh:
            if line.startswith("[PTRACE] "):
                m = HDR_RE.match(line)
                if m:
                    hdr = {
                        "slots": int(m.group(1)),
                        "layers": int(m.group(2)),
                        "tick_ns": int(m.group(3)),
                    }
            elif line.startswith("[PTRACES] "):
                m = SUB_RE.match(line)
                if not m:
                    corrupt += 1
                    continue
                pairs = [p.split(":") for p in m.group(4).split()]
                if len(pairs) != int(m.group(3)):
                    corrupt += 1
                    continue
                subs[(int(m.group(1)), int(m.group(2)))] = [
                    (int(c), int(t)) for c, t in pairs]
            elif line.startswith("[PTRACE_SUBDROP] "):
                # The device ran out of sub-mark ring space. Missing tail marks
                # look exactly like "the overlap stopped happening", so this is
                # surfaced rather than swallowed.
                print(f"WARNING: device reported dropped sub-marks: "
                      f"{line.strip()}")
            elif line.startswith("[PTRACEW] "):
                m = ROW_RE.match(line)
                if not m:
                    corrupt += 1
                    continue
                ts = [int(x) for x in m.group(6).split()]
                # A row that parsed but has the wrong slot count is the same
                # failure as one that did not parse: the printf stream was
                # spliced mid-line. Drop it. Aborting the whole conversion
                # here would mean one interleaved line costs the entire trace.
                if hdr is not None and len(ts) != hdr["slots"]:
                    corrupt += 1
                    continue
                # None (not 0) when the field is absent: a log from a build
                # without MPK_PHASE_PART must stay readable, and "no
                # information" has to stay distinct from "participated in
                # nothing" -- conflating them would relabel every span in an
                # old trace as skipped.
                part, arm = m.group(4), m.group(5)
                rows.append((int(m.group(1)), int(m.group(2)),
                             int(m.group(3)), ts,
                             int(part) if part is not None else None,
                             int(arm) if arm is not None else None))
            elif line.startswith("[PTRACE_END]"):
                saw_end = True
    return hdr, rows, subs, corrupt, saw_end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("-o", "--out", default="trace.json")
    ap.add_argument("--layers", type=int, default=0,
                    help="emit only the first N layers (0 = all)")
    args = ap.parse_args()

    hdr, rows, subs, corrupt, saw_end = parse(args.log)
    if hdr is None:
        sys.exit(f"no [PTRACE] header in {args.log} -- was this built with "
                 f"-DMPK_PHASE_TRACE and run with MPK_PHASE_SLOTS=1?")
    if not rows:
        sys.exit(f"[PTRACE] header but no [PTRACEW] rows in {args.log}")
    if corrupt:
        print(f"WARNING: dropped {corrupt} interleaved/truncated rows")
    # The device printf buffer is finite and this is the largest thing Fleet
    # prints. Without the trailing marker the tail is missing, and a truncated
    # trace looks exactly like a real load imbalance -- the last workers appear
    # to stop early. Warn loudly rather than render a plausible lie.
    if not saw_end:
        print("WARNING: no [PTRACE_END] -- the device printf buffer truncated "
              "the dump. Later workers/layers are MISSING, not idle. Raise "
              "HIP_PRINTF_BUFFER_SIZE or lower MPK_PHASE_TRACE_LAYERS.")

    nslots = hdr["slots"]
    if nslots != len(SPAN_NAMES):
        sys.exit(f"{nslots} slots but {len(SPAN_NAMES)} span names -- the slot "
                 f"map in persistent_kernel.cuh changed; update SPAN_NAMES")

    if args.layers:
        rows = [r for r in rows if r[2] < args.layers]
        if not rows:
            sys.exit(f"--layers {args.layers} selected no rows")

    tick_ns = hdr["tick_ns"]
    # Perfetto's unit is microseconds (float ok). Rebase on the earliest mark so
    # the trace starts near zero instead of at a raw 64-bit counter value.
    #
    # Named _t0_origin, not t0, and bound as a default argument below: the
    # short name is exactly what a loop over timestamp pairs reaches for, and
    # when one did, every slice emitted after it was rebased against a MoE tile
    # mark instead of the trace start. Nothing raised -- the output was simply
    # wrong, with negative timestamps and a doubled makespan, on a kernel whose
    # own latency had not moved. Closing over a mutable outer name is what made
    # that possible, so this one is captured by value at definition.
    _t0_origin = min(ts[0] for _, _, _, ts, _, _ in rows if ts[0])

    def us(tick, _origin=_t0_origin):
        return (tick - _origin) * tick_ns / 1000.0

    events = []
    xcds = set()
    # Previous layer's exit per worker, for the inter-layer span. Rows arrive
    # grouped by worker and ascending in layer (the device loop emits them that
    # way), but sort explicitly rather than depend on it.
    rows.sort(key=lambda r: (r[0], r[2]))
    prev_exit = {}
    n_skipped = 0
    n_gated = 0
    n_armed = 0
    n_split_slots = defaultdict(int)

    for w, xcd, layer, ts, part, arm in rows:
        xcds.add(xcd)
        for s in range(1, nslots):
            a, b = ts[s - 1], ts[s]
            # A zero mark is a slot the worker never reached (role-dependent:
            # MoE-only ranks skip QKV and attention entirely). Non-monotonic
            # would mean the shared-clock assumption failed; count it.
            if a == 0 or b == 0:
                continue
            if b < a:
                n_skipped += 1
                continue
            name = SPAN_NAMES[s]

            # Did this worker actually run this phase, or merely reach the
            # branch that guards it? Only the device knows, and only when the
            # build recorded it: part is None for a log predating
            # MPK_PHASE_PART, and those spans keep their old names so an old
            # trace stays readable rather than turning uniformly grey.
            skipped = (part is not None and s in GATED_SLOTS
                       and not (part >> s) & 1)
            if skipped:
                n_gated += 1
                events.append({
                    "name": SKIP_NAME.format(GATED_SLOTS[s]),
                    "cat": "phase",
                    "ph": "X",
                    "pid": xcd,
                    "tid": w,
                    "ts": us(a),
                    "dur": us(b) - us(a),
                    "cname": SKIP_COLOR,
                    "args": {"layer": layer, "slot": s, "worker": w,
                             "xcd": xcd, "participated": False,
                             "would_have_been": name,
                             "note": "the phase mark is outside the guard, so "
                                     "this worker timestamped a not-taken "
                                     "branch; it did NOT run this op"},
                })
                continue

            # Participating, but in which arm? For a rank-guarded phase the
            # question does not arise -- one guard, one op. For MoE it does,
            # and the participation bit cannot answer it: the bit is set for
            # every worker that entered the loop, including the ones that got
            # no tile and the ones running the opposite half of the kernel.
            arm_name = None
            if arm is not None and s in ARM_NAMES:
                code = (arm >> (s * ARM_BITS)) & ARM_MASK
                arm_name = ARM_NAMES[s].get(code)

            # A barrier wait hides inside the phase span that surrounds it.
            # Cut it out, edge to edge: head -> cut_1 -> ... -> slot end. Like
            # slot 1's handoffs and unlike slot 8's tile islands, these tile
            # the parent exactly, so the row keeps its extent and gains the
            # boundary between "still working" and "waiting on everyone else".
            #
            # First occurrence only: these marks fire once per layer, not per
            # tile. A code that is absent is a worker that did not reach that
            # barrier -- ranks below the o-proj guard skip 18, and 20/21 are an
            # if/else so exactly one of them is ever present -- and the cut
            # simply does not happen, which is the correct rendering.
            bcuts = BARRIER_CUTS.get(s)
            if bcuts:
                at = {}
                for code, tick in (subs.get((w, layer)) or []):
                    at.setdefault(code, tick)
                hit = [(at[c], lab) for c, lab in bcuts
                       if c in at and a <= at[c] <= b]
                hit.sort()
                if hit:
                    n_split_slots[s] += 1
                    # The head keeps the parent's own name when the slot is
                    # already named for what precedes the barrier; otherwise
                    # BARRIER_HEAD says what that remainder actually is.
                    # Each piece runs from one boundary to the next and is
                    # named for the boundary that OPENS it: the head names
                    # everything before the first mark, and each mark names
                    # the stretch from itself to the next mark or, for the
                    # last one, to the slot's end.
                    bounds = [(a, BARRIER_HEAD.get(s, name))] + hit
                    ends = [t for t, _ in hit] + [b]
                    pieces = [(beg, end, lab)
                              for (beg, lab), end in zip(bounds, ends)]
                    for cs, ce, label in pieces:
                        if ce <= cs:
                            continue
                        ev = {
                            "name": label,
                            "cat": "phase",
                            "ph": "X",
                            "pid": xcd,
                            "tid": w,
                            "ts": us(cs),
                            "dur": us(ce) - us(cs),
                            "cname": BARRIER_COLOR.get(label, "grey"),
                            "args": {"layer": layer, "slot": s, "worker": w,
                                     "xcd": xcd, "split_of": name,
                                     "participated": label not in BARRIER_IDLE_LABELS,
                                     "waiting": label in BARRIER_WAIT_LABELS,
                                     "idle": label in BARRIER_IDLE_LABELS,
                                     "scope": ("GLOBAL" if "(GLOBAL)" in label
                                               else "XCD-LOCAL"
                                               if "(XCD-LOCAL)" in label
                                               else None)},
                        }
                        events.append(ev)
                    continue

            # The arm names the whole phase; segments cut it into the tiles
            # inside. Both apply to slot 8, and the segments win when present
            # because they are the finer measurement -- the arm becomes the
            # fallback for a worker whose marks did not survive the ring.
            seg = SLOT_SEGMENTS.get(s)
            seg_cuts = []
            if seg:
                # Marks fire once per TILE, so a two-tile worker emits each
                # code twice. Pair them up in order rather than keeping only
                # the first, which is the mistake the arm bits already made
                # once (last-writer-wins) in the other direction.
                byc = defaultdict(list)
                for code, tick in (subs.get((w, layer)) or []):
                    byc[code].append(tick)
                # NOT t0/t1: `t0` is the trace-wide rebase origin that us()
                # closes over, and rebinding it here silently rebased every
                # later timestamp against a MoE tile mark -- 50798 slices went
                # negative and the makespan doubled to 120 us/layer while the
                # kernel itself was unchanged at 2.245 ms/iter.
                for c0, c1, label in seg["pairs"]:
                    for beg, end in zip(byc.get(c0, []), byc.get(c1, [])):
                        if end > beg and beg >= a and end <= b:
                            seg_cuts.append((beg, end, label))
                seg_cuts.sort()
            if seg_cuts:
                n_split_slots[s] += 1
                # Islands, not a tiling: name the space between tiles instead
                # of stretching an arm across it.
                # A gap is named for the tile it precedes, because that is what
                # ends it. The same wall-clock position means different things
                # for different workers: for a two-tile worker the middle gap
                # is the W13->W2 barrier, while for a tail rank holding a lone
                # W2 tile that same barrier sits at the head.
                filled, cur = [], a
                inner_spec = seg.get("inner", {})
                for beg, end, label in seg_cuts:
                    if beg > cur:
                        filled.append((cur, beg, seg["gap_before"][label]))
                    # A tile may split on its own interior marks. Unlike the
                    # tile islands these are handoffs, so they tile [beg, end)
                    # edge to edge and the tile's extent is preserved.
                    spec = inner_spec.get(label)
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
                    filled.append((cur, b, seg["gap_tail"]))
                for cs, ce, label in filled:
                    if ce <= cs:
                        continue
                    ev = {
                        "name": label,
                        "cat": "phase",
                        "ph": "X",
                        "pid": xcd,
                        "tid": w,
                        "ts": us(cs),
                        "dur": us(ce) - us(cs),
                        # `participated` keeps ONE meaning trace-wide -- the
                        # phase's participation bit -- so a consumer can filter
                        # on it without knowing which slots get segmented. We
                        # only reach here when that bit is set. Whether a given
                        # segment is a tile or the space between tiles is a
                        # different question and gets its own key; overloading
                        # `participated` for it made 22209 gap-segments read as
                        # "this worker skipped MoE" while its bit was set.
                        "args": {"layer": layer, "slot": s, "worker": w,
                                 "xcd": xcd, "split_of": name,
                                 "arm": arm_name,
                                 "participated": True,
                                 "running_tile": label in SEG_TILE_LABELS},
                    }
                    if label in SPLIT_COLOR:
                        ev["cname"] = SPLIT_COLOR[label]
                    events.append(ev)
                continue

            split = SLOT_SPLIT.get(s)
            at = {}
            if split:
                for code, tick in (subs.get((w, layer)) or []):
                    at.setdefault(code, tick)
                need = [split["spans"][0][0], split["tail_from"]]
                if not all(c in at for c in need):
                    split = None

            if arm_name is not None and split is None:
                n_armed += 1
                is_empty = arm_name.startswith("(skipped")
                if is_empty:
                    n_gated += 1
                events.append({
                    "name": arm_name,
                    "cat": "phase",
                    "ph": "X",
                    "pid": xcd,
                    "tid": w,
                    "ts": us(a),
                    "dur": us(b) - us(a),
                    "cname": ARM_COLOR.get(arm_name, SKIP_COLOR),
                    "args": {"layer": layer, "slot": s, "worker": w,
                             "xcd": xcd, "participated": not is_empty,
                             "arm": arm_name, "fused_phase": name,
                             "note": "the phase name describes the kernel; "
                                     "this span describes what this worker "
                                     "actually ran, which is the arm (or arms) "
                                     "its tiles landed in -- not necessarily "
                                     "all of the fused phase, and not "
                                     "necessarily just one arm"},
                })
                continue

            # Draw the constituent ops instead of the one span. The segments
            # tile [a, b] exactly -- head, each sub-span, tail -- so the row
            # keeps the same extent and gains the op boundaries.
            if split:
                n_split_slots[s] += 1
                cuts = []
                first = at[split["spans"][0][0]]
                if first > a:
                    cuts.append((a, first, split["head"]))
                for c0, c1, label in split["spans"]:
                    if c0 in at and c1 in at and at[c1] >= at[c0]:
                        cuts.append((at[c0], at[c1], label))
                if b > at[split["tail_from"]]:
                    cuts.append((at[split["tail_from"]], b, split["tail"]))
                for cs, ce, label in cuts:
                    if ce <= cs:
                        continue
                    ev = {
                        "name": label,
                        "cat": "phase",
                        "ph": "X",
                        "pid": xcd,
                        "tid": w,
                        "ts": us(cs),
                        "dur": us(ce) - us(cs),
                        "args": {"layer": layer, "slot": s, "worker": w,
                                 "xcd": xcd, "split_of": name},
                    }
                    if label in SPLIT_COLOR:
                        ev["cname"] = SPLIT_COLOR[label]
                    events.append(ev)
                continue

            ev = {
                "name": name,
                "cat": "phase",
                "ph": "X",
                "pid": xcd,
                "tid": w,
                "ts": us(a),
                "dur": us(b) - us(a),
                "args": {"layer": layer, "slot": s, "worker": w, "xcd": xcd},
            }
            if name in COLOR:
                ev["cname"] = COLOR[name]
            events.append(ev)
        # The inter-layer span, drawn from the previous layer's exit. This is
        # the gap the [PSLOTW] table folds into slot 0; showing it as a slice
        # is what makes the row of layers look continuous instead of leaving
        # unexplained whitespace between them.
        if w in prev_exit and ts[0] and prev_exit[w] <= ts[0]:
            events.append({
                "name": "inter_layer",
                "cat": "phase",
                "ph": "X",
                "pid": xcd,
                "tid": w,
                "ts": us(prev_exit[w]),
                "dur": us(ts[0]) - us(prev_exit[w]),
                "cname": COLOR["inter_layer"],
                "args": {"layer": layer, "slot": 0, "worker": w, "xcd": xcd},
            })
        # ── Sub-phase spans: memory genuinely in flight under compute ─────
        #
        # What remains on the detail row after SLOT_SPLIT is only the spans
        # that TRULY overlap the compute beside them: weight DMA issued early
        # and flying while other work runs. Those cannot go on the worker's
        # own row, because Perfetto renders two overlapping slices on one
        # track as nesting -- "inside", not "at the same time as" -- and here
        # "at the same time as" is the entire measurement. Sequential ops that
        # merely lived inside one phase span are now drawn in place by
        # SLOT_SPLIT instead, so this row no longer mixes the two meanings.
        marks = subs.get((w, layer))
        if marks:
            at = {}
            for code, tick in marks:
                at.setdefault(code, tick)      # first occurrence per layer
            for c0, c1, label in SUB_SPANS:
                req = SUB_REQUIRES.get(label)
                if req is not None and req not in at:
                    continue
                if c0 in at and c1 in at and at[c1] >= at[c0]:
                    events.append({
                        "name": label,
                        "cat": "subphase",
                        "ph": "X",
                        "pid": xcd,
                        "tid": w + TID_ASYNC,
                        "ts": us(at[c0]),
                        "dur": us(at[c1]) - us(at[c0]),
                        "cname": SUB_COLOR.get(label, "grey"),
                        "args": {"layer": layer, "worker": w, "xcd": xcd},
                    })
            # The QKV prefetch has no end mark: its consumer is the next
            # layer's Phase 1, which skips its own DMA rather than draining
            # this one, so no instant in this layer means "it finished". The
            # honest span is issue -> layer exit: the window in which the DMA
            # is in flight and the worker is doing barrier work instead of
            # waiting on it.
            if 1 in at and ts[nslots - 1] and ts[nslots - 1] >= at[1]:
                events.append({
                    "name": "qkv weight DMA in flight",
                    "cat": "subphase",
                    "ph": "X",
                    "pid": xcd,
                    "tid": w + TID_ASYNC,
                    "ts": us(at[1]),
                    "dur": us(ts[nslots - 1]) - us(at[1]),
                    "cname": SUB_COLOR["qkv weight DMA in flight"],
                    "args": {"layer": layer, "worker": w, "xcd": xcd,
                             "note": "issue -> layer exit; no drain mark "
                                     "exists (consumer skips its own DMA)"},
                })
        if ts[nslots - 1]:
            prev_exit[w] = ts[nslots - 1]

    if n_skipped:
        print(f"WARNING: {n_skipped} spans had end < start and were dropped. "
              f"s_memrealtime is meant to be die-wide and monotonic; this "
              f"many is a real anomaly, not rounding.")

    # Process/thread names. Without these Perfetto shows bare numeric ids.
    for x in sorted(xcds):
        events.append({"name": "process_name", "ph": "M", "pid": x, "tid": 0,
                       "args": {"name": f"XCD {x}"}})
        events.append({"name": "process_sort_index", "ph": "M", "pid": x,
                       "tid": 0, "args": {"sort_index": x}})
    seen = {}
    async_rows = {e["tid"] for e in events if e.get("cat") == "subphase"}
    for w, xcd, _, _, _, _ in rows:
        if (xcd, w) in seen:
            continue
        seen[(xcd, w)] = True
        events.append({"name": "thread_name", "ph": "M", "pid": xcd, "tid": w,
                       "args": {"name": f"worker {w}"}})
        # sort_index * 2 leaves an odd slot beneath each worker for its detail
        # row, so "worker 5" and "worker 5 detail" stay adjacent instead of
        # all detail rows piling up at the bottom of the XCD.
        events.append({"name": "thread_sort_index", "ph": "M", "pid": xcd,
                       "tid": w, "args": {"sort_index": w * 2}})
        if w + TID_ASYNC in async_rows:
            # NOT a second execution context: same worker, same timeline. But
            # not "nested" either, which the previous label claimed -- these
            # are DMA spans that genuinely overlap the compute on the row
            # above, and one of them crosses three sibling phases. That is why
            # they cannot share the row (CTF stacks same-track slices), and it
            # is the whole reason the marks exist. The sequential sub-phases
            # DO nest, and those are drawn in place on the worker's own row.
            events.append({"name": "thread_name", "ph": "M", "pid": xcd,
                           "tid": w + TID_ASYNC,
                           "args": {"name": f"worker {w} · async DMA "
                                            f"(same worker, overlaps above)"}})
            events.append({"name": "thread_sort_index", "ph": "M", "pid": xcd,
                           "tid": w + TID_ASYNC,
                           "args": {"sort_index": w * 2 + 1}})

    with open(args.out, "w") as fh:
        json.dump({"traceEvents": events,
                   "displayTimeUnit": "ns"}, fh)

    span_evs = [e for e in events if e["ph"] == "X"]
    sub_evs = [e for e in span_evs if e.get("cat") == "subphase"]
    nlayers = len({r[2] for r in rows})
    makespan = max(e["ts"] + e["dur"] for e in span_evs) - \
        min(e["ts"] for e in span_evs)
    if sub_evs:
        print(f"  {len(sub_evs)} sub-phase slices on per-worker detail rows")
    else:
        # Silence here means the overlap looks absent when it may simply be
        # uninstrumented, which is the exact confusion this feature exists to
        # end. Say which flags produce the marks.
        print("  no [PTRACES] sub-phase marks -- overlap rows will be empty. "
              "Build with -DMPK_PREFETCH_NEXT_QKV / -DMPK_W13_LDS_PREFETCH "
              "to get them.")
    if any(r[4] is not None for r in rows):
        print(f"  {n_gated} spans marked non-participating (worker reached "
              f"the phase's guard but did not run it)")
    else:
        # Without the mask every gated span is drawn as its phase whether the
        # worker ran it or not, which is how "MoE || QKV overlap = 3.02 us"
        # was reported for an overlap that does not exist. Say so.
        print("  WARNING: no p= field -- this log predates MPK_PHASE_PART. "
              "Spans on slots 1/3/6/8 may be workers that skipped the phase, "
              "drawn as if they ran it. Do NOT derive overlap from this.")
    if any(r[5] is not None for r in rows):
        print(f"  {n_armed} spans resolved to a specific arm of a fused phase")
    else:
        # The participation bit says a MoE worker took part; it cannot say
        # whether it ran W13, ran W2, or got no tile. Without a= the span keeps
        # the fused name, which describes the kernel and not the worker.
        print("  WARNING: no a= field -- this log predates MPK_PHASE_ARM. "
              "MoE spans are drawn 'moe(w13+swiglu+w2)' for every worker, but "
              "a worker runs W13 OR W2, never both, and some get no tile at "
              "all. Do NOT read per-op occupancy off slot 8.")
    print(f"wrote {args.out}: {len(span_evs)} slices, "
          f"{len(seen)} workers, {len(xcds)} XCDs, {nlayers} layers")
    print(f"makespan {makespan:.1f} us over {nlayers} layers "
          f"= {makespan / max(nlayers, 1):.2f} us/layer")
    print(f"open at https://ui.perfetto.dev (Open trace file)")


if __name__ == "__main__":
    main()
