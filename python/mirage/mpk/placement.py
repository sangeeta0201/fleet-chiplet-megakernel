"""Data placement for the persistent megakernel: domain model, access declarations, rules.

Phase 0 of docs/nps2/PLACEMENT_COMPILER_DESIGN.md: plan and report only. Nothing in this
file changes the generated code or any allocation.

* MACHINES describes the locality domains of a GPU.
* ACCESS declares, for every slot of every task type in the gpt-oss-120b graph, which
  instances write which region of the slot's tensor and which instances read it.
* plan() joins the declarations with the task graph that generate_task_graph() emitted
  (task_graph.json lists every task instance's tensors with their offsets and dims),
  classifies each tensor family and applies the rules of the design's section 6.
* today() states what the hand recipe's environment flags do with the same family, so
  the report shows where the compiler and the recipe agree, and why they differ.

PersistentKernel.compile() runs report() when MPK_PLACE_REPORT=1. It prints a [PLACE]
table and writes place_report.json and place_report.txt next to task_graph.json. Offline:

    python3 placement.py task_graph.json --env-from run.log [--out DIR]

Standard library only, so the offline form runs without mirage or a GPU.
"""

import json
import os
import re
import sys
from collections import OrderedDict

KiB = 1 << 10
MiB = 1 << 20
GiB = 1 << 30


# ---------------------------------------------------------------------------- machine


class Machine:
    """Locality domains of one GPU. Domains are the memory ranges (AIDs) here."""

    def __init__(self, name, domains, xcds_per_domain, workers_per_xcd, chunk,
                 l2_per_xcd, hbm_per_domain, lat_ns):
        self.name = name
        self.domains = domains
        self.xcds_per_domain = xcds_per_domain
        self.workers_per_xcd = workers_per_xcd
        self.chunk = chunk
        self.l2_per_xcd = l2_per_xcd
        self.hbm_per_domain = hbm_per_domain
        self.lat_ns = lat_ns

    @property
    def xcds(self):
        return self.domains * self.xcds_per_domain

    def domain_of(self, xcd):
        return xcd // self.xcds_per_domain


# Latencies measured with the latlv pointer chase (2.40 GHz): L2 82 ns from any die,
# MALL 167 near / 273 far, HBM 316 near / 423 far.
MACHINES = {
    "mi355x_spx_nps2": Machine(
        "MI355X SPX+NPS2", domains=2, xcds_per_domain=4, workers_per_xcd=31,
        chunk=2 * MiB, l2_per_xcd=4 * MiB, hbm_per_domain=144 * GiB,
        lat_ns={"l2": 82, "mall_near": 167, "mall_far": 273,
                "hbm_near": 316, "hbm_far": 423}),
}

# Replicas and private copies must leave room for torch, the runtime and the KV cache.
MEMORY_BUDGET = 0.85


# ----------------------------------------------------------------- access declarations


class Acc:
    """How the instances of one task slot use its tensor. Instance i runs on XCD i.

    w, who writes:
      None    not written through this slot
      "own"   each instance writes its own region
      "same"  every instance writes the same values (redundant compute)
      "pair"  die pair p (XCDs 2p and 2p+1) writes region p
      "one"   a single instance, on an XCD the scheduler picks
      "ext"   the runtime or the host, outside task code
    r, who reads:
      None    not read through this slot
      "own"   an instance reads only its own region
      "own_domain"  an instance reads its own domain's copy
      "all"   every instance reads `frac` of the whole tensor
      "pair"  the instances of pair p read region p
      "routed"  the region read is chosen at run time
      "one"   a single instance
      "ext"   the runtime or the host
    via: "imap" when the task graph hands each instance its own slice (the instance
      offsets in task_graph.json differ); "kernel" when every instance gets the whole
      tensor and the kernel indexes its part by xcd_id.
    dim: tensor axis the regions follow.
    slack: the producer has independent work between its store and the release that
      consumers wait for, so the extra stores of a publish can hide there.
    sync: the tensor holds sync objects (see EVENTS), not data.
    unused_if: flags under which the kernel neither stores to nor loads from the slot.
    """

    __slots__ = ("w", "r", "via", "dim", "frac", "slack", "sync", "unused_if", "note")

    def __init__(self, w=None, r=None, via="kernel", dim=0, frac=1.0, slack=False,
                 sync=None, unused_if=(), note=""):
        self.w = w
        self.r = r
        self.via = via
        self.dim = dim
        self.frac = frac
        self.slack = slack
        self.sync = sync
        self.unused_if = tuple(unused_if)
        self.note = note


TASK_NAMES = {
    0: "terminate",
    10: "begin_task_graph",
    101: "embedding",
    111: "argmax_reduce",
    212: "moe_residual_add_f32",
    216: "gang_full_layer_fused_mi300",
    217: "gang_full_layer_with_lmhead_fused_mi300",
    218: "gang_rmsnorm_linear_mxfp4_bias_argmax_mi300",
}

_W_OPROJ = ("die d's O-proj writes its 368 columns; every MoE worker (FP8 quant of the"
            " full row), the next layer's QKV fold and the LM head read the full row; the"
            " router partial runs between the store and the release")

# Slot order follows the tb_graph.new_input() calls in persistent_kernel.py
# (gang_full_layer_fused_layer: 24 inputs, 11 outputs).
ACCESS = {
    216: {
        "in": [
            Acc(r="all", note="every QKV workgroup folds the previous layer's 4 MoE slots"),
            Acc(r="all", note="every QKV workgroup folds the full residual row"),
            Acc(r="all", note="RMSNorm gamma, read whole by every QKV workgroup"),
            Acc(w="same", r="own_domain",
                note="pre-attention normed row; every worker computes the same row"),
            Acc(r="own", via="imap", dim=0, note="QKV weight, 10 tiles per XCD"),
            Acc(r="own", via="imap", dim=1, note="QKV bias slice per XCD"),
            Acc(r="own", dim=0, note="die d reads the sinks of its 8 query heads"),
            Acc(sync="qkv_epoch", note="per-XCD QKV arrival counters"),
            Acc(w="own", r="own", dim=1,
                note="attention partials: die d's chunk workers write, its merger reads"),
            Acc(r="own", via="imap", dim=0, note="O-proj weight, 23 tiles per XCD"),
            Acc(r="own", via="imap", dim=1, note="O-proj bias slice per XCD"),
            Acc(r="all", note="gamma2: router partials and the FP8 quant of the full row"),
            Acc(w="same", r="own_domain",
                note="post-attention normed row; every worker computes the same row"),
            Acc(r="all", via="imap", dim=0, frac=0.5,
                note="router weight: the slot is 16 experts per XCD, but with"
                     " MPK_ROUTER_OPROJ_PARTIAL each XCD's O-proj epilogue undoes that"
                     " offset and reads its 368 columns of all 128 experts' rows"),
            Acc(r="own", via="imap", dim=1, note="router bias slice per XCD"),
            Acc(w="own", r="all", via="imap", dim=1,
                unused_if=("MPK_LOCAL_TOPK", "MPK_LTK_NO_LOGIT_STORE"),
                note="router logits; no stores with LOCAL_TOPK + LTK_NO_LOGIT_STORE"),
            Acc(sync="oproj_counters",
                note="epochs, releases and counters of the per-layer barriers"),
            Acc(r="routed", dim=0, note="W13 experts chosen by the router"),
            Acc(r="routed", dim=0, note="W2 experts chosen by the router"),
            Acc(r="routed", dim=0, note="W13 bias of the chosen experts"),
            Acc(r="routed", dim=0, note="W2 bias of the chosen experts"),
            Acc(sync="w13_w2", note="per-expert W13 -> W2 arrival counter and releases"),
            Acc(w="pair", r="pair", dim=1,
                note="SwiGLU row: W13 of pair p writes slot p, W2 of pair p reads it"),
            Acc(w="own", r="own", dim=1,
                note="attention o partials: die d's chunk workers write, its merger reads"),
        ],
        "out": [
            Acc(w="same", r="own", dim=1,
                note="h = r + MoE fold; every QKV workgroup writes the full row, die d's"
                     " O-proj reads its 368 columns"),
            Acc(w="own", r="own", dim=0,
                note="K cache, head-major: die d's QKV workgroup 8 writes head d, die d's"
                     " attention reads it"),
            Acc(w="own", r="own", dim=0, note="V cache, as K (QKV workgroup 9)"),
            Acc(w="own", r="own", dim=1,
                note="Q: die d's QKV workgroups 0-7 write its 8 heads, its attention reads"),
            Acc(w="own", r="all", dim=1, slack=False,
                note="attention output: die d's merge writes heads 8d..8d+7, every O-proj"
                     " workgroup reads all 8 slices; the merge releases right after the"
                     " store"),
            Acc(w="own", r="all", dim=1, slack=True, note=_W_OPROJ),
            Acc(w="one", r="all", note="TopK completer output (routing weights)"),
            Acc(w="one", r="all", note="TopK completer output (routing indices)"),
            Acc(w="one", r="all", note="TopK completer output (active expert mask)"),
            Acc(w="one", r="all", note="TopK completer output (routing weights)"),
            Acc(w="pair", dim=1, slack=False,
                note="MoE workspace: W2 of pair p writes slot p, then arrives at the gate"),
        ],
    },
    # gang_rmsnorm_linear_mxfp4_bias_argmax_layer (the LM head)
    218: {
        "in": [
            Acc(r="all", note="residual add and RMSNorm of the full row on every worker"),
            Acc(r="all", note="final RMSNorm gamma"),
            Acc(w="same", r="own_domain", note="normed row, recomputed by every worker"),
            Acc(r="own", via="imap", dim=0, note="LM head weight, vocabulary slice per XCD"),
            Acc(r="own", via="imap", dim=1, note="LM head bias slice per XCD"),
        ],
        "out": [
            Acc(w="own", via="imap", dim=1, note="argmax partial values, one per worker"),
            Acc(w="own", via="imap", dim=1, note="argmax partial indices, one per worker"),
            Acc(r="all", note="MPK_LM_RESADD: folds the last layer's MoE slots (read only)"),
        ],
    },
    101: {
        "in": [
            Acc(w="ext", r="one", note="current token id, written by the runtime"),
            Acc(r="one", note="embedding table, one row per token"),
        ],
        "out": [
            Acc(w="one", note="embedding row; layer 0's QKV fold reads it"),
        ],
    },
    111: {
        "in": [
            Acc(r="one", note="one block reduces the argmax partials"),
            Acc(r="one", note="one block reduces the argmax partials"),
        ],
        "out": [
            Acc(w="one", r="ext", note="next token id, read by the runtime"),
        ],
    },
}

# Synchronization per layer (rule 5). Arrivals and waiters are the role counts of the
# recipe at 1k/1k (N = 16 or 31 KV chunks); see the dependency map canvas.
EVENTS = [
    dict(name="QKV epoch", dep="many-to-many inside one die", scope="xcd", last=False,
         who="QKV producers + chunk participants -> attention chunk workers"),
    dict(name="chunk barrier", dep="many-to-one inside one die", scope="xcd", last=True,
         who="N chunk workers -> the last arriver merges"),
    dict(name="attention release", dep="all-to-all", scope="all", last=False,
         who="each die's merger -> every die's O-proj waves"),
    dict(name="O-proj die combine", dep="many-to-one inside one die", scope="xcd",
         last=True, who="23 O-proj workgroups -> the die's last arriver"),
    dict(name="router sums -> TopK", dep="all-to-all", scope="all", last=False,
         who="8 dies' last arrivers -> all 248 workers"),
    dict(name="W13 -> W2", dep="many-to-many inside a die pair", scope="pair", last=False,
         who="46 W13 tiles on 2 dies -> 46 W2 tiles"),
    dict(name="layer gate", dep="all-to-all", scope="all", last=False,
         who="248 arrivals -> the next layer's joiners"),
    dict(name="scheduler events", dep="per event", scope="all", last=False,
         who="task completions -> dependent task fetches"),
    dict(name="iteration gate", dep="all-to-all", scope="all", last=False,
         who="end of token -> every worker"),
]


# ------------------------------------------------------------------------ task graph

_DT_BYTES = {920: 0.5, 925: 0.5, 926: 0.5, 930: 1, 935: 1, 936: 1, 940: 2, 941: 2,
             945: 2, 946: 2, 950: 4, 955: 4, 956: 4, 960: 8, 965: 8, 966: 8}

RING_BASES = ("moe_workspace_f32", "attn_out", "attn_proj_out")


def family_of(name):
    """(family, copy index): layer_7_qkv_mxfp4 -> ("layer_*_qkv_mxfp4", 7)."""
    m = re.match(r"layer_(\d+)_(.+)$", name)
    if m:
        return "layer_*_" + m.group(2), int(m.group(1))
    for base in RING_BASES:
        if name == base:
            return base + "[ring]", 0
        if name.startswith(base + "_r") and name[len(base) + 2:].isdigit():
            return base + "[ring]", int(name[len(base) + 2:])
    return name, None


def _extent_bytes(ref):
    elems = 1
    for d, s in zip(ref["dims"], ref["strides"]):
        elems += (d - 1) * s
    return int(elems * _DT_BYTES.get(ref["data_type"], 1))


class Tensor:
    def __init__(self, name):
        self.name = name
        self.family, self.copy = family_of(name)
        self.nbytes = 0
        self.uses = []  # (task_type, role, slot, instance, offset, extent)


def load_graph(graph):
    """Tensors and their per-instance uses from task_graph.json (dict or JSON text)."""
    if isinstance(graph, str):
        graph = json.loads(graph)
    tensors = OrderedDict()
    counts = OrderedDict()
    prev_key, inst = None, 0
    for t in graph["all_tasks"]:
        tt = t["task_type"]
        refs = [("in", k, r) for k, r in enumerate(t.get("inputs") or [])]
        refs += [("out", k, r) for k, r in enumerate(t.get("outputs") or [])]
        if not refs:
            continue
        counts[tt] = counts.get(tt, 0) + 1
        # The instances of one gang task are consecutive and in XCD order; a new layer
        # (another layer_<i>_ name) or another task type starts a new gang.
        layer = next((m.group(1) for _, _, r in refs
                      for m in [re.match(r"layer_(\d+)_", r["base_ptr"])] if m), None)
        key = (tt, layer)
        inst = inst + 1 if key == prev_key else 0
        prev_key = key
        for role, slot, r in refs:
            name = r["base_ptr"]
            ten = tensors.get(name)
            if ten is None:
                ten = tensors[name] = Tensor(name)
            ext = _extent_bytes(r)
            ten.nbytes = max(ten.nbytes, int(r["offset"]) + ext)
            ten.uses.append((tt, role, slot, inst, int(r["offset"]), ext))
    return tensors, counts


# ------------------------------------------------------------------------ environment


def env_from_log(path):
    """The ENVS line fleet_run_lane.sh writes at the top of every run log."""
    env = {}
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("ENVS "):
                for tok in line[5:].split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        env[k] = v
                break
    return env


def _on(env, key):
    return env.get(key, "0") not in ("", "0")


def _listed(env, key, item):
    return item in [p.strip() for p in env.get(key, "").split(",") if p.strip()]


# ----------------------------------------------------------------------- the rules


class Family:
    def __init__(self, name):
        self.name = name
        self.tensors = []
        self.accs = []      # (task_type, role, slot, Acc)
        self.dropped = []   # (task_type, role, slot, Acc) unused under the flags

    @property
    def nbytes(self):
        return sum(t.nbytes for t in self.tensors)

    @property
    def copies(self):
        return len(self.tensors)

    def kinds(self, attr):
        return {getattr(a, attr) for _, _, _, a in self.accs if getattr(a, attr)}


def classify(fam):
    if not fam.accs:
        return "unused" if fam.dropped else "undeclared"
    if fam.kinds("sync"):
        return "sync"
    ws, rs = fam.kinds("w"), fam.kinds("r")
    if "ext" in ws or "ext" in rs:
        return "ext"
    if not ws:
        if "routed" in rs:
            return "routed"
        if rs <= {"own"}:
            return "slice"
        if rs <= {"one"}:
            return "read_one"
        return "read_all"
    if ws <= {"own"} and rs <= {"own"}:
        return "private"
    if ws <= {"pair"} and rs <= {"pair"}:
        return "pair_private"
    if ws <= {"same"} and rs <= {"own", "own_domain"}:
        return "replica_private"
    if rs <= {"one"}:
        return "many_to_one"
    return "owner_read_all"


DEP_TYPE = {
    "slice": "one-to-one (read-only slice per XCD)",
    "private": "one-to-one (each die writes and reads its own region)",
    "pair_private": "one-to-one per die pair",
    "replica_private": "each domain writes and reads its own copy",
    "read_all": "one-to-all (read-only)",
    "read_one": "read-only, read by one instance",
    "routed": "routed (run-time selected slices)",
    "owner_read_all": "all-to-all (each owner writes, every die reads)",
    "many_to_one": "many-to-one",
    "ext": "shared with the runtime/host",
    "sync": "sync objects",
    "unused": "unused under the current flags",
    "undeclared": "no declaration",
}


class Decision:
    def __init__(self, kind, decision, cls, rule, why, per_domain, level="aid"):
        self.kind = kind
        self.decision = decision
        self.cls = cls
        self.rule = rule
        self.why = why
        self.per_domain = per_domain  # bytes in each domain
        self.level = level


READ_ONLY = ("slice", "read_all", "read_one", "routed")


def decide(fam, m, env, n):
    """Decision for one family. n: its logical size (the recipe's own copies removed)."""
    kind = classify(fam)
    D = m.domains
    even = [n / D] * D
    one = [n if d == 0 else 0 for d in range(D)]
    if kind in ("slice", "private"):
        small = n / fam.copies / D < m.chunk
        why = "split at the consumer boundary"
        if small:
            why += "; each domain's part is below one chunk, so it is packed with other data"
        return Decision(kind, "split", "RW", "1", why, even)
    if kind == "pair_private":
        return Decision(kind, "split (pair)", "RW", "1",
                        "each die pair's region in the pair's domain (a pair never spans"
                        " two domains)", even)
    if kind == "replica_private":
        return Decision(kind, "replicate", "RW", "1",
                        "each domain writes and reads its own copy", [n] * D)
    if kind == "read_all":
        per_pass = n * max(a.frac for _, _, _, a in fam.accs)
        if per_pass > m.l2_per_xcd:
            return Decision(kind, "replicate", "RW", "2",
                            "re-read from HBM every pass: %.1f MiB > %d MiB of L2"
                            % (per_pass / MiB, m.l2_per_xcd // MiB), [n] * D)
        return Decision(kind, "one copy", "RW", "2",
                        "fits in L2 (%.0f KiB per pass); read-only lines never go stale"
                        % (per_pass / KiB), one)
    if kind == "read_one":
        return Decision(kind, "one copy", "RW", "2",
                        "read-only and read by one instance", one)
    if kind == "routed":
        return Decision(kind, "replicate", "RW", "4",
                        "routed: a copy per domain, work assigned by selection slot"
                        " (MPK_MOE_XCD_PAIR + MPK_LOCAL_TOPK)", [n] * D)
    if kind == "owner_read_all":
        ws = fam.kinds("w")
        frac = max([a.frac for _, _, _, a in fam.accs if a.r == "all"] or [0])
        slack = all(a.slack for _, _, _, a in fam.accs if a.w)
        static = ws <= {"own", "pair"}
        if static and frac >= 0.5 and slack:
            return Decision(kind, "publish", "RW", "3a",
                            "readers read %.0f%% of it and the producer has slack" %
                            (100 * frac), [n] * D)
        if static:
            return Decision(kind, "writer", "NC", "3b",
                            "home by writer: the producer releases right after its store"
                            if not slack else "home by writer: readers read little", even)
        return Decision(kind, "shared", "NC", "3c",
                        "the writer is one instance on a scheduler-chosen XCD", one)
    if kind == "many_to_one":
        return Decision(kind, "shared", "NC", "3c",
                        "many writers, one reader on a scheduler-chosen XCD", one)
    if kind == "ext":
        return Decision(kind, "shared", "NC", "-",
                        "the runtime or host reads or writes it outside the graph", one)
    if kind == "sync":
        return Decision(kind, "sync", "-", "5", "see the sync table", one)
    if kind == "unused":
        return Decision(kind, "unused", "-", "-", "no loads or stores under " + ", ".join(
            sorted({f for _, _, _, a in fam.dropped for f in a.unused_if})), [0] * D)
    return Decision(kind, "shared", "NC", "-", "no access declaration", one)


# ------------------------------------------------------------- today's hand recipe


def graph_copies(fam, env):
    """Copies of the data the recipe already put into the tensor's shape (doubled dim 0)."""
    f = fam.name
    if (f in ("layer_*_gate_up_proj", "layer_*_down_proj") and _on(env, "MPK_MOE_REPLICA")
            and not _on(env, "MPK_MOE_RW")):
        return 2
    if f in ("layer_*_gate_up_bias", "layer_*_down_bias") and _on(env, "MPK_MOE_BIAS_REP"):
        return 2
    if (f in ("layer_*_input_layernorm", "layer_*_post_attn_layernorm")
            and _on(env, "MPK_GAMMA_AID")):
        return 2
    if f == "rmsnorm_out" and (_on(env, "MPK_PRENORM_AID") or _on(env, "MPK_LMNORM_AID")):
        return 2
    if f == "rmsnorm_out_moe" and _on(env, "MPK_MOENORM_AID"):
        return 2
    if f == "swiglu_out" and _on(env, "MPK_SWIGLU_AID"):
        return 2
    if f == "moe_workspace_f32[ring]" and _on(env, "MPK_WSF32_AID"):
        return 2
    return 1


def today(fam, env):
    """(decision, class, mechanism) of the environment flags for one family."""
    f = fam.name
    own = lambda key: _listed(env, "MPK_OWN_BO_PAT", key) and _on(env, "MPK_OWN_BO_ALIGN")
    ring = lambda key: _on(env, "MPK_LAYER_RING") and _listed(env, "MPK_OWN_BO_RINGS", key)
    scratch = lambda key: _listed(env, "MPK_OWN_BO_SCRATCH", key)
    unplaced = ("shared", "NC", "torch allocation, one range")
    for key in ("qkv_mxfp4", "o_proj_mxfp4", "moe_gate"):
        if f == "layer_*_" + key:
            if own(key):
                return ("split", "NC", "own BO cut at the halves midpoint (MPK_OWN_BO_PAT)")
            return unplaced
    if f == "lm_head_mxfp4":
        if _on(env, "MPK_LM_OWN_BO"):
            return ("split", "NC", "own exact-size BO, halves midpoint (MPK_LM_OWN_BO)")
        return unplaced
    if f in ("layer_*_gate_up_proj", "layer_*_down_proj"):
        if _on(env, "MPK_MOE_RW"):
            return ("replicate", "RW", "AID-local GEM copy per AID (MPK_MOE_RW)")
        if _on(env, "MPK_MOE_REPLICA"):
            extra = " + own BO" if own(f[len("layer_*_"):]) else ""
            return ("replicate", "NC",
                    "expert axis doubled, a copy per range (MPK_MOE_REPLICA%s)" % extra)
        return unplaced
    if f in ("layer_*_gate_up_bias", "layer_*_down_bias"):
        if _on(env, "MPK_MOE_BIAS_REP"):
            extra = " + own BO" if own(f[len("layer_*_"):]) else ""
            return ("replicate", "NC", "doubled, a copy per range (MPK_MOE_BIAS_REP%s)" % extra)
        return unplaced
    if f in ("layer_*_k_cache", "layer_*_v_cache"):
        if _on(env, "MPK_KV_OWN_BO") and _on(env, "MPK_KV_HEAD_MAJOR"):
            return ("split", "NC", "head-major own BO per layer, cut at head 4 (MPK_KV_OWN_BO)")
        return unplaced
    if f in ("ck_fmha_lse_acc", "ck_fmha_o_acc", "ck_fmha_q_workspace"):
        if scratch(f):
            return ("split", "NC", "own BO (MPK_OWN_BO_SCRATCH)")
        return unplaced
    if f == "attn_out[ring]":
        if ring("ao"):
            return ("writer", "NC", "ring copy per layer, own BO cut at head 32"
                                    " (MPK_OWN_BO_RINGS=ao)")
        return unplaced
    if f == "attn_proj_out[ring]":
        if _on(env, "MPK_AID_SPLIT_OUT"):
            how = "O-proj writes its columns into both AIDs' replicas (MPK_AID_SPLIT_OUT)"
            if ring("apo"):
                how += "; ring copies cut by writer for the next layer's residual"
            return ("publish", "RW", how)
        if ring("apo"):
            return ("writer", "NC", "ring copy per layer cut by writer (MPK_OWN_BO_RINGS=apo)")
        return unplaced
    if f == "moe_workspace_f32[ring]":
        if ring("ws"):
            return ("writer", "NC", "ring copy per layer, own BO cut at slot 2"
                                    " (MPK_OWN_BO_RINGS=ws)")
        return unplaced
    if f == "swiglu_out":
        if _on(env, "MPK_SWIGLU_AIDREP"):
            return ("publish", "RW", "W13 writes the row into both AIDs' replicas"
                                     " (MPK_SWIGLU_AIDREP)")
        if _on(env, "MPK_SWIGLU_AID"):
            return ("replicate", "NC", "doubled torch tensor (MPK_SWIGLU_AID)")
        return unplaced
    if f in ("layer_*_input_layernorm", "layer_*_post_attn_layernorm"):
        if _on(env, "MPK_GAMMA_AID"):
            return ("replicate", "NC", "doubled, a copy per range (MPK_GAMMA_AID)")
        return unplaced
    if f == "rmsnorm_out":
        if _on(env, "MPK_PRENORM_AID") or _on(env, "MPK_LMNORM_AID"):
            return ("replicate", "NC", "doubled, each range's workers use their copy"
                                       " (MPK_PRENORM_AID / MPK_LMNORM_AID)")
        return unplaced
    if f == "rmsnorm_out_moe":
        if _on(env, "MPK_MOENORM_AID"):
            return ("replicate", "NC", "doubled, each range's workers use their copy"
                                       " (MPK_MOENORM_AID)")
        return unplaced
    if f == "moe_gate_out" and _on(env, "MPK_LOCAL_TOPK") and _on(env, "MPK_LTK_NO_LOGIT_STORE"):
        return ("unused", "-", "allocated, never stored (MPK_LTK_NO_LOGIT_STORE)")
    if f in ("qkv_attn_barrier", "oproj_topk_counters", "moe_fused_barrier"):
        return ("sync", "-", "see the sync table")
    if f == "embed_tokens":
        return ("one copy", "NC", "torch allocation")
    return unplaced


# Measured reasons for every place where the rules and the recipe differ. A difference
# without an entry here is reported as UNEXPLAINED.
EXPLAIN = {
    "ck_fmha_lse_acc": "XCD-local by construction: the same die writes and merges its"
                       " partials, which stay in its L2. Own BOs at the head cut measured"
                       " null at 16k (1.603/1.604 vs 1.603/1.606/1.604 ms).",
    "ck_fmha_o_acc": "as ck_fmha_lse_acc (measured together).",
    "ck_fmha_q_workspace": "own BO at the half cut measured null (1.203/1.203/1.200 vs"
                           " 1.205/1.205 ms).",
    "swiglu_out": "the rules keep one copy in the pair's AID, since pairs never span AIDs;"
                  " dropping the far copy (MPK_PAIR_LOCAL_PUB) measured +0.46% on Sep 23,"
                  " before MPK_AID_REP_CONST removed the MoE's code-layout sensitivity."
                  " Recheck in Phase 3.",
    "layer_*_input_layernorm": "doubling per AID measured neutral in the drop-one ablation"
                               " (MPK_GAMMA_AID); the rules keep one copy since it fits"
                               " in L2.",
    "layer_*_post_attn_layernorm": "as layer_*_input_layernorm (same flag).",
    "mlp_weighted_sum_out": "untested: every die writes the full row and reads its own"
                            " 368 columns, so a copy per domain removes remote stores and"
                            " loads. Phase 3 candidate.",
    "layer_*_qkv_bias": "tiny slices: own BOs for small per-layer tensors (180 BOs) lost"
                        " 0.37%; in the arena they pack into the domain's chunk at no"
                        " allocation cost. Phase 3 candidate.",
    "layer_*_o_bias": "as layer_*_qkv_bias.",
    "layer_*_router_bias": "as layer_*_qkv_bias (256 B per layer; today in one range).",
    "layer_*_sinks": "as layer_*_qkv_bias (128 B per layer).",
    "lm_head_bias": "as layer_*_qkv_bias (a zero bias, 49 KiB per XCD slice); MPK_LM_OWN_BO"
                    " moves only the weight.",
    "moe_gate_out": "unused under the recipe either way.",
    "layer_*_moe_gate": "with MPK_ROUTER_OPROJ_PARTIAL each XCD reads its columns of all"
                        " 128 experts' rows, so the recipe's own BO (cut at expert 64)"
                        " serves half of every XCD's router reads from the far die. The"
                        " rules replicate it (25.9 MiB per domain). An arena split broke"
                        " the output (empty text, 2026-10-06): the kernel indexes the"
                        " whole weight from its slice pointer.",
}

# Read-only or domain-private data the rules put on RW while the recipe's copies are
# NC: halves-placed hipMalloc memory is MTYPE_NC under aid_local_xcp_nc=Y.
CLASS_NOTE = ("read-only or domain-private, so RW is safe (policy 1); the recipe's"
              " halves-placed hipMalloc copies are MTYPE_NC. Streams measured equal on"
              " NC and RW (1.66 vs 1.65 us burst issue); RW lines also survive an L2"
              " invalidate. Phase 1 keeps NC to change one thing at a time; Phase 3"
              " measures RW.")


def _today_sync(name, env):
    on = lambda k: _on(env, k)
    if name == "QKV epoch":
        if on("MPK_AID_QKV_ARRIVE") and on("MPK_AID_SPLIT_QKV"):
            how = "counter in the die's AID replica" + (
                ", polled directly (MPK_QKV_EPOCH_DIRECT)" if on("MPK_QKV_EPOCH_DIRECT") else "")
            return ("domain counter", "own", how)
        return ("domain counter", "aid0", "per-XCD counters in the shared torch tensor")
    if name == "chunk barrier":
        if on("MPK_AID_CHUNK_BAR"):
            return ("last arriver", "own", "counter in the die's AID replica (MPK_AID_CHUNK_BAR)")
        return ("last arriver", "aid0", "counter in the shared counters tensor")
    if name == "attention release":
        if on("MPK_AID_SPLIT_FLAGS"):
            return ("publish slots", "own", "write-through flag into both AID replicas"
                                            " (MPK_AID_SPLIT_FLAGS)")
        return ("shared flag", "aid0", "one flag line per die, polled by every die")
    if name == "O-proj die combine":
        if on("MPK_OPROJ_HIER_NC"):
            return ("last arriver", "own", "counter in the die's NC scratch (MPK_OPROJ_HIER_NC)")
        return ("last arriver", "aid0", "die-private counter (hier_local) in the shared"
                                        " counters tensor, homed on AID0")
    if name == "router sums -> TopK":
        if on("MPK_LOCAL_TOPK"):
            return ("publish slots", "own", "tagged sums into both AIDs' replicas"
                                            " (MPK_LOCAL_TOPK, MPK_ROUTER_OPROJ_PARTIAL)")
        return ("shared flag", "aid0", "serial TopK completer")
    if name == "W13 -> W2":
        if on("MPK_MOE_NCLOCAL"):
            return ("pair counter", "own", "NC counter in the pair's AID (MPK_MOE_NCLOCAL)")
        how = "device-scope atomic on the shared moe_fused_barrier (AID0)"
        if on("MPK_MOE_NARROW_RELEASE"):
            how += ", release into both AIDs (MPK_MOE_NARROW_RELEASE)"
        return ("pair counter", "aid0", how)
    if name == "layer gate":
        if on("MPK_P9_FLAT"):
            home = "own" if on("MPK_AID_P9_NC") else "aid0"
            return ("domain counters + slots", home,
                    "die counter (layer_local) in the shared counters tensor + flat"
                    " slots into both AIDs (MPK_P9_FLAT)" if home == "aid0" else
                    "die counter in the die's NC scratch + flat slots (MPK_AID_P9_NC)")
        return ("global atomic", "aid0", "device-wide layer_global atomic")
    if name == "scheduler events":
        if on("MPK_AID_EVCTR2") or on("MPK_AID_EVCTR"):
            return ("domain mirror", "own", "AID mirrors published on increment"
                                            " (MPK_AID_EVCTR2)")
        return ("shared counter", "aid0", "plain gpu_malloc counters (NC)")
    if name == "iteration gate":
        if on("MPK_ITER_XCD_POLL") and on("MPK_ITER_AID"):
            return ("domain mirror", "own", "per-AID copy (MPK_ITER_XCD_POLL, MPK_ITER_AID)")
        return ("shared counter", "aid0", "one shared line")
    return ("?", "?", "?")


SYNC_EXPLAIN = {
    "O-proj die combine": "same structure; the recipe's counter sits in the AID0-homed"
                          " counters tensor (remote for dies 4-7). Moving it to the die's"
                          " AID (MPK_OPROJ_HIER_NC) measured null.",
    "W13 -> W2": "same structure; the recipe's atomic is on AID0 for every pair (+110 ns per"
                 " W13 tile on dies 4-7) and releases into both AIDs. The pair-local NC"
                 " counter (MPK_MOE_NCLOCAL) measured null once the replica pointers"
                 " were __constant__. Phase 2 generates it.",
    "layer gate": "same structure; the per-die counter sits in the AID0-homed counters"
                  " tensor. The die-local NC counter (MPK_AID_P9_NC) measured -0.21% before"
                  " MPK_AID_REP_CONST and null after it (together with the O-proj counter).",
}


def sync_rule(ev, m):
    if ev["scope"] == "xcd":
        if ev["last"]:
            return ("last arriver", "own", "one counter in the die's domain; the last"
                                           " arriver continues, nobody polls")
        return ("domain counter", "own", "one counter in the die's domain (RW sync arena);"
                                         " waiters poll it")
    if ev["scope"] == "pair":
        if m.xcds_per_domain % 2 == 0:
            return ("pair counter", "own", "NC counter in the pair's domain (atomics from"
                                           " 2 dies); release into the pair's domain")
        return ("pair counter", "aid0", "the pair spans domains: NC counter, release into"
                                        " both domains")
    if ev["name"] in ("scheduler events", "iteration gate"):
        return ("domain mirror", "own", "a mirror per domain, published on increment;"
                                        " readers poll their own")
    if m.domains > 8:
        return ("tree", "own", "two-level tree: counter per leaf group, slot per group"
                               " in every consumer domain")
    if ev["name"] == "attention release" or ev["name"] == "router sums -> TopK":
        return ("publish slots", "own", "the arriver writes its slot into every consumer"
                                        " domain; consumers poll locally")
    return ("domain counters + slots", "own", "arrival counter per domain; each domain's"
                                              " last arriver writes one slot into every"
                                              " consumer domain")


# ----------------------------------------------------------------- BO geometry (online)


def bo_geometry(records, chunk=2 * MiB):
    """Where each attached tensor sits in its allocation (needs HIP; online only).

    records: name -> (data_ptr, nbytes). Returns name -> dict with the allocation base,
    size, offset and the bytes before the halves cut, the split point the patched
    driver uses for a BO (its midpoint, rounded down to a 2 MiB chunk).
    """
    import ctypes as ct

    hip = ct.CDLL("libamdhip64.so")
    out = {}
    for name, (ptr, n) in records.items():
        if not ptr or not n:
            continue
        base, size = ct.c_void_p(), ct.c_size_t()
        if hip.hipMemGetAddressRange(ct.byref(base), ct.byref(size), ct.c_void_p(ptr)) != 0:
            continue
        b, s = base.value or 0, size.value
        cut = b + ((s >> 1) // chunk) * chunk
        out[name] = dict(base=b, size=s, off=ptr - b, n=n,
                         first=max(0, min(ptr + n, cut) - ptr))
    return out


def _measured(fam, dec_today, how, geo, chunk):
    """Where the recipe's placement of this family lands, from the BO layout.

    A BO of two chunks or more is cut at its midpoint (rounded down to a chunk); a
    smaller BO cannot be cut and sits whole in one range, which this cannot tell.
    """
    if "replicas" in how and "ring" not in how:
        return "carried in the AID flag replicas, not in a graph tensor"
    rows = [geo[t.name] for t in fam.tensors if t.name in geo]
    if not rows:
        return None
    if dec_today in ("split", "replicate", "writer", "publish"):
        dev = [abs(g["first"] - g["n"] / 2) for g in rows]
        if max(dev) == 0:
            return "exact (%d/%d)" % (len(rows), len(rows))
        one_side = sum(1 for g, d in zip(rows, dev) if d == g["n"] / 2)
        if one_side == len(rows):
            what = "both copies in one range" if dec_today == "replicate" else "not split"
            return "%s (%d/%d wholly on one side of the cut)" % (what, len(rows), len(rows))
        bad = sum(1 for d in dev if d > 0)
        return "%d/%d off the boundary, worst %s" % (bad, len(rows), _fmt(max(dev)))
    small = sum(1 for g in rows if g["size"] < 2 * chunk)
    big = [g for g in rows if g["size"] >= 2 * chunk]
    side0 = sum(1 for g in big if g["first"] == g["n"])
    side1 = sum(1 for g in big if g["first"] == 0)
    parts = []
    if side0:
        parts.append("first half %d" % side0)
    if side1:
        parts.append("second half %d" % side1)
    if len(big) - side0 - side1:
        parts.append("straddles the cut %d" % (len(big) - side0 - side1))
    if small:
        parts.append("whole in a BO below %s %d" % (_fmt(2 * chunk), small))
    return ", ".join(parts)


# ------------------------------------------------------------------------ plan/report


def _fmt(n):
    for unit, s in ((GiB, "GiB"), (MiB, "MiB"), (KiB, "KiB")):
        if n >= unit:
            return "%.2f %s" % (n / unit, s)
    return "%d B" % n


def _check(tensors, families):
    """Declarations against the graph: an imap slice must have distinct instance offsets."""
    problems = []
    for fam in families.values():
        for tt, role, slot, a in fam.accs:
            if a.via != "imap":
                continue
            for t in fam.tensors:
                offs = {u[4] for u in t.uses if u[0] == tt and u[1] == role and u[2] == slot}
                if len(offs) <= 1:
                    problems.append("%s %s[%d] %s: declared imap slice, all instances at"
                                    " offset %s" % (TASK_NAMES.get(tt, tt), role, slot,
                                                    t.name, sorted(offs)))
                    break
    return problems


def _allocs_by_flags(families, env):
    """Dedicated allocations the recipe's placement flags make (offline estimate)."""
    parts = OrderedDict()
    names = [t.name for fam in families.values() for t in fam.tensors]
    if _on(env, "MPK_KV_OWN_BO") and _on(env, "MPK_KV_HEAD_MAJOR"):
        parts["KV caches"] = sum(1 for n in names if n.endswith(("_k_cache", "_v_cache")))
    if _on(env, "MPK_LAYER_RING"):
        bases = dict(ws="moe_workspace_f32[ring]", ao="attn_out[ring]",
                     apo="attn_proj_out[ring]")
        n = sum(families[bases[k]].copies for k in bases
                if _listed(env, "MPK_OWN_BO_RINGS", k) and bases[k] in families)
        if n:
            parts["ring copies"] = n
    pats = [p for p in env.get("MPK_OWN_BO_PAT", "").split(",") if p]
    if pats:
        # gate_up_proj copies only the layers whose allocation is not already its own
        # BO at the midpoint, so this is an upper bound.
        # Patterns apply to attached weights only; make_tensor scratch moves only by name.
        parts["weight own BOs (at most)"] = sum(
            1 for n in names if n.startswith("layer_") and any(p in n for p in pats)
            and "k_cache" not in n and "v_cache" not in n)
    scratch = [p for p in env.get("MPK_OWN_BO_SCRATCH", "").split(",") if p]
    if scratch:
        parts["scratch own BOs"] = sum(1 for n in names if n in scratch)
    if _on(env, "MPK_LM_OWN_BO"):
        parts["LM head own BO"] = 1
    if _on(env, "MPK_AID_SPLIT_FLAGS"):
        parts["flag replicas (alloc_in_aid)"] = 2
    return parts


def plan(graph, env, machine="mi355x_spx_nps2", records=None, arena=None):
    m = MACHINES[machine]
    in_arena = OrderedDict()
    if arena is not None:
        for name, p in arena.placed.items():
            in_arena[family_of(name)[0]] = p["mode"]
    tensors, counts = load_graph(graph)
    families = OrderedDict()
    undeclared = []
    for t in tensors.values():
        fam = families.get(t.family)
        if fam is None:
            fam = families[t.family] = Family(t.family)
        fam.tensors.append(t)
    for fam in families.values():
        seen = set()
        for t in fam.tensors:
            for tt, role, slot, inst, off, ext in t.uses:
                if (tt, role, slot) in seen:
                    continue
                seen.add((tt, role, slot))
                decl = ACCESS.get(tt, {}).get(role)
                if decl is None or slot >= len(decl):
                    undeclared.append((TASK_NAMES.get(tt, tt), role, slot, t.name))
                    continue
                a = decl[slot]
                if a.unused_if and all(_on(env, k) for k in a.unused_if):
                    fam.dropped.append((tt, role, slot, a))
                else:
                    fam.accs.append((tt, role, slot, a))
    problems = _check(tensors, families)
    geo = None
    if records:
        try:
            geo = bo_geometry(records, m.chunk)
        except Exception as e:  # report-only: never fail the compile
            problems.append("BO geometry unavailable: %s" % e)

    rows = []
    totals = {"RW": [0.0] * m.domains, "NC": [0.0] * m.domains}
    for fam in families.values():
        factor = graph_copies(fam, env)
        logical = fam.nbytes / factor
        dec = decide(fam, m, env, logical)
        t_dec, t_cls, t_how = today(fam, env)
        if fam.name in in_arena:
            t_dec = {"replicate": "replicate", "straddle": "writer"}.get(in_arena[fam.name], "split")
            t_how = "arena region per domain, instance pointers rebased (MPK_PLACE_ARENA)"
        if t_dec == "shared" and dec.kind in READ_ONLY:
            t_dec = "one copy"  # read-only and unplaced: a single copy wherever torch put it
        if dec.decision == t_dec:
            status = "match" if (dec.cls == t_cls or "-" in (dec.cls, t_cls)) else "class"
        else:
            status = "differs" if fam.name in EXPLAIN else "UNEXPLAINED"
        why = EXPLAIN.get(fam.name, "") if status in ("differs", "UNEXPLAINED") else ""
        if dec.cls in totals:
            for d in range(m.domains):
                totals[dec.cls][d] += dec.per_domain[d]
        rows.append(OrderedDict(
            family=fam.name, copies=fam.copies, bytes=int(logical), graph_copies=factor,
            kind=dec.kind,
            dep=DEP_TYPE.get(dec.kind, dec.kind), decision=dec.decision, cls=dec.cls,
            rule=dec.rule, rule_why=dec.why, today=t_dec, today_cls=t_cls,
            today_how=t_how, status=status, explain=why,
            measured=("arena, exact by construction" if fam.name in in_arena else
                      _measured(fam, t_dec, t_how, geo, m.chunk) if geo else None),
            notes=sorted({a.note for _, _, _, a in fam.accs + fam.dropped if a.note})))

    sync = []
    for ev in EVENTS:
        c_struct, c_home, c_how = sync_rule(ev, m)
        t_struct, t_home, t_how = _today_sync(ev["name"], env)
        if (c_struct, c_home) == (t_struct, t_home):
            st = "match"
        elif c_struct == t_struct:
            st = "home differs" if ev["name"] in SYNC_EXPLAIN else "UNEXPLAINED"
        else:
            st = "differs" if ev["name"] in SYNC_EXPLAIN else "UNEXPLAINED"
        sync.append(OrderedDict(event=ev["name"], dep=ev["dep"], who=ev["who"],
                                compiler=c_how, today=t_how, status=st,
                                explain=SYNC_EXPLAIN.get(ev["name"], "") if st != "match" else ""))

    budget = MEMORY_BUDGET * m.hbm_per_domain
    used = [totals["RW"][d] + totals["NC"][d] for d in range(m.domains)]
    allocs = OrderedDict(
        compiler=2,
        compiler_mi355x_now=m.domains + 1,
        compiler_note="one RW arena whose chunks are bound to domains + one NC arena;"
                      " %d with one RW BO per domain until the per-chunk-domain GEM"
                      " create lands" % (m.domains + 1))
    allocs["today_by_flags"] = _allocs_by_flags(families, env)
    if geo:
        allocs["today_bos"] = len({g["base"] for g in geo.values()})
        allocs["today_note"] = ("distinct allocations behind the graph's %d attached"
                                " tensors, plus the 2 flag replicas from alloc_in_aid"
                                % len(geo))
    summary = OrderedDict(
        machine=m.name, tasks=OrderedDict((TASK_NAMES.get(k, str(k)), v) for k, v in counts.items()),
        families=len(rows),
        match=sum(r["status"] == "match" for r in rows),
        class_only=sum(r["status"] == "class" for r in rows),
        differs=sum(r["status"] == "differs" for r in rows),
        unexplained=sum(r["status"] == "UNEXPLAINED" for r in rows) +
        sum(s["status"] == "UNEXPLAINED" for s in sync) + len(undeclared),
        per_domain_bytes=[int(u) for u in used],
        rw_bytes=[int(x) for x in totals["RW"]], nc_bytes=[int(x) for x in totals["NC"]],
        budget_ok=all(u <= budget for u in used),
        allocations=allocs)
    return OrderedDict(summary=summary, families=rows, sync=sync,
                       undeclared=["%s %s[%d] %s" % u for u in undeclared],
                       problems=problems, geometry=geo or {})


def render(rep):
    s = rep["summary"]
    out = []
    p = out.append
    p("[PLACE] %s: %s" % (s["machine"], ", ".join("%s x%d" % kv for kv in s["tasks"].items())))
    p("[PLACE] %d families: %d match, %d match except the memory class, %d differ (explained),"
      " %d unexplained" % (s["families"], s["match"], s["class_only"], s["differs"],
                           s["unexplained"]))
    w = max(len(r["family"]) for r in rep["families"])
    p("[PLACE] %-*s %6s %11s  %-14s %-4s %-4s  %-10s %-4s  %s" % (
        w, "family", "copies", "bytes", "compiler", "cls", "rule", "today", "cls", "status"))
    for r in rep["families"]:
        p("[PLACE] %-*s %6d %11s%s %-14s %-4s %-4s  %-10s %-4s  %s%s" % (
            w, r["family"], r["copies"], _fmt(r["bytes"]),
            "*" if r["graph_copies"] > 1 else " ", r["decision"], r["cls"],
            r["rule"], r["today"], r["today_cls"], r["status"],
            ("  [measured: %s]" % r["measured"]) if r["measured"] else ""))
    p("[PLACE] bytes are logical; * = the recipe doubled the tensor in the graph")
    p("[PLACE]")
    for r in rep["families"]:
        if r["status"] in ("differs", "UNEXPLAINED"):
            p("[PLACE] %s (%s): %s" % (r["family"], r["status"], r["explain"] or "-"))
    cls = [r["family"] for r in rep["families"] if r["status"] == "class"]
    if cls:
        p("[PLACE] memory class differs for %s: %s" % (", ".join(cls), CLASS_NOTE))
    missed = [r["family"] for r in rep["families"] if r["measured"]
              and r["measured"].startswith(("both copies in one range", "not split"))]
    if missed:
        p("[PLACE] the recipe's placement does not happen for %s: the doubled tensor lies"
          " wholly on one side of its allocation's cut" % ", ".join(missed))
    p("[PLACE]")
    for ev in rep["sync"]:
        p("[PLACE] sync %-20s %-12s compiler: %s" % (ev["event"], ev["status"], ev["compiler"]))
        p("[PLACE] %s today:    %s" % (" " * 38, ev["today"]))
        if ev["explain"]:
            p("[PLACE] %s why:      %s" % (" " * 38, ev["explain"]))
    a = s["allocations"]
    p("[PLACE]")
    p("[PLACE] per domain: %s (RW %s, NC %s); budget %s" % (
        " / ".join(_fmt(x) for x in s["per_domain_bytes"]),
        " / ".join(_fmt(x) for x in s["rw_bytes"]), " / ".join(_fmt(x) for x in s["nc_bytes"]),
        "ok" if s["budget_ok"] else "EXCEEDED"))
    p("[PLACE] allocations: compiler %d (%d on this driver: %s)" % (
        a["compiler"], a["compiler_mi355x_now"], a["compiler_note"]))
    tf = a.get("today_by_flags") or {}
    if tf:
        p("[PLACE] allocations today, by flags: %s = %s%d dedicated allocations for"
          " placement" % (", ".join("%s %d" % kv for kv in tf.items()),
                          "at most " if any("at most" in k for k in tf) else "",
                          sum(tf.values())))
    if "today_bos" in a:
        p("[PLACE] allocations today: %d (%s)" % (a["today_bos"], a["today_note"]))
    for u in rep["undeclared"]:
        p("[PLACE] UNDECLARED %s" % u)
    for q in rep["problems"]:
        p("[PLACE] CHECK %s" % q)
    return "\n".join(out)


def report(graph, env, out_dir=None, records=None, machine="mi355x_spx_nps2", arena=None):
    rep = plan(graph, env, machine=machine, records=records, arena=arena)
    text = render(rep)
    print(text, flush=True)
    if out_dir:
        with open(os.path.join(out_dir, "place_report.json"), "w") as f:
            json.dump(rep, f, indent=1)
        with open(os.path.join(out_dir, "place_report.txt"), "w") as f:
            f.write(text + "\n\n")
            for r in rep["families"]:
                f.write("%s\n  kind: %s (%s)\n  compiler: %s %s, rule %s: %s\n"
                        "  today: %s %s: %s\n" % (
                            r["family"], r["kind"], r["dep"], r["decision"], r["cls"],
                            r["rule"], r["rule_why"], r["today"], r["today_cls"],
                            r["today_how"]))
                for n in r["notes"]:
                    f.write("  kernel: %s\n" % n)
    return rep


def report_safe(*args, **kwargs):
    """report() for the compile path: a report failure must never stop a build."""
    try:
        return report(*args, **kwargs)
    except Exception as e:
        print("[PLACE] report failed: %r" % (e,), flush=True)
        return None


# ---------------------------------------------------------------- arena (Phase 1a)

# Families the arena places in Phase 1a, as the recipe does today:
#   split      dim 0 cut into one part per XCD; XCD x's part lives in its domain
#   replicate  a full copy in every domain
#   private    dim 0 cut by XCD as for split, but the kernel indexes its part from the
#              base itself (KV head = xcd_id), so the instance pointer is rebased
# The rings, the four doubled per-AID tensors and the AID flag replicas stay as they
# are until Phase 1b / 2.
ARENA_MODES = OrderedDict([
    ("layer_*_qkv_mxfp4", "split"),
    ("layer_*_o_proj_mxfp4", "split"),
    # Every XCD reads all 128 experts' rows (MPK_ROUTER_OPROJ_PARTIAL), relative to its
    # slice pointer, so the router weight must stay contiguous: a copy per domain.
    ("layer_*_moe_gate", "replicate"),
    ("lm_head_mxfp4", "split"),
    ("layer_*_gate_up_proj", "replicate"),
    ("layer_*_down_proj", "replicate"),
    ("layer_*_gate_up_bias", "replicate"),
    ("layer_*_down_bias", "replicate"),
    ("layer_*_k_cache", "private"),
    ("layer_*_v_cache", "private"),
    # Phase 1b: the per-AID copies the recipe meant to make by doubling, which Phase 0
    # measured wholly in one range. Read-only gammas, and normed rows that every
    # worker of a domain writes with the same values and reads back.
    ("layer_*_input_layernorm", "replicate"),
    ("layer_*_post_attn_layernorm", "replicate"),
    ("rmsnorm_out", "replicate"),
    ("rmsnorm_out_moe", "replicate"),
])

# MPK_MOE_SPLIT: the expert weights and biases in one copy laid out by halves ([2E, ...]: every
# expert's first half, then every second half; demo.py). The arena then puts the first half in
# domain 0 and the second in domain 1 ("split") instead of a copy per domain.
MOE_SPLIT_FAMILIES = ("layer_*_gate_up_proj", "layer_*_down_proj", "layer_*_gate_up_bias",
                      "layer_*_down_bias")

# The arena takes over what these flags did for its families: no expert-axis
# doubling (the kernel then reads the copy its instance pointer selects) and no own
# BOs. The rings keep MPK_OWN_BO_RINGS.
ARENA_SUPERSEDES = OrderedDict([
    ("MPK_MOE_REPLICA", "0"), ("MPK_MOE_BIAS_REP", "0"), ("MPK_MOE_RW", "0"),
    ("MPK_OWN_BO_PAT", ""), ("MPK_KV_OWN_BO", "0"), ("MPK_LM_OWN_BO", "0"),
    ("MPK_GAMMA_AID", "0"), ("MPK_PRENORM_AID", "0"), ("MPK_LMNORM_AID", "0"),
    ("MPK_MOENORM_AID", "0"),
])

# Phase 1b families and the flags they replace; MPK_ARENA_1B=0 leaves both as the
# recipe has them (Phase 1a).
ARENA_1B = ("layer_*_input_layernorm", "layer_*_post_attn_layernorm", "rmsnorm_out",
            "rmsnorm_out_moe")
ARENA_1B_FLAGS = ("MPK_GAMMA_AID", "MPK_PRENORM_AID", "MPK_LMNORM_AID", "MPK_MOENORM_AID")


def _arena_1b():
    return os.environ.get("MPK_ARENA_1B", "1") == "1"


# Phase 1c: the layer rings. Each copy is written half by each domain (heads, columns or
# expert slots of dies 0-3 | 4-7) and read whole by every die, so it must be contiguous
# across a range boundary. Copy k straddles the boundary at (2k+1) * 2 MiB of an
# interleaved BO (chunk j in range j % 2; ring-chunkmap driver, AMDGPU_GEM_CREATE_AID_ILV).
# MPK_ARENA_RINGS=1 turns it on and supersedes MPK_OWN_BO_RINGS.
# MPK_W_SWAP=1: read-only weight families whose per-domain part is stored on the other AID (timing control).
W_SWAP_FAMILIES = ("layer_*_qkv_mxfp4", "layer_*_o_proj_mxfp4", "layer_*_moe_gate", "lm_head_mxfp4",
                   "layer_*_gate_up_proj", "layer_*_down_proj", "layer_*_gate_up_bias", "layer_*_down_bias",
                   "layer_*_input_layernorm", "layer_*_post_attn_layernorm")


def w_swap():
    return os.environ.get("MPK_W_SWAP", "0") not in ("", "0")


ARENA_RINGS = ("moe_workspace_f32[ring]", "attn_out[ring]", "attn_proj_out[ring]")
RING_SLOTS = 128  # ring copies per interleaved BO (two 2 MiB chunks each)


def _arena_rings():
    return os.environ.get("MPK_ARENA_RINGS", "0") == "1"


def arena_env(environ):
    """Turn off the per-tensor placement flags the arena replaces (MPK_PLACE_ARENA=1)."""
    changed = []
    supersede = list(ARENA_SUPERSEDES.items())
    if environ.get("MPK_ARENA_RINGS", "0") == "1":
        supersede.append(("MPK_OWN_BO_RINGS", ""))
    for k, v in supersede:
        if k in ARENA_1B_FLAGS and environ.get("MPK_ARENA_1B", "1") != "1":
            continue
        if environ.get(k, "") != v:
            changed.append("%s=%s" % (k, environ.get(k, "")))
            environ[k] = v
    print("[ARENA] supersedes %s" % (", ".join(changed) or "nothing"), flush=True)


def mdev_count():
    """MPK_MULTI_DEV: number of HIP devices (DPX/CPX partitions of one GPU) the megakernel spans."""
    try:
        return int(os.environ.get("MPK_MULTI_DEV", "0") or 0)
    except ValueError:
        return 0


_VMM = {}


def _register(kind, va, nbytes, **kw):
    """Record a raw device allocation (provider_export dumps the ones the kernel's pointers land in)."""
    import ctypes as ct

    hip = _VMM.get("hip") or ct.CDLL("libamdhip64.so")
    _VMM["hip"] = hip
    cur = ct.c_int()
    hip.hipGetDevice(ct.byref(cur))
    _VMM.setdefault("allocs", []).append(dict(kind=kind, va=va, nbytes=nbytes, device=cur.value, **kw))


def halves_malloc(nbytes, domains=2, swap=False):
    """nbytes of device memory whose region d (nbytes / domains each) is physically on AID d.

    SPX+NPS2: a plain hipMalloc; the patched driver's halves placement already puts its first
    half in range 0 and its second in range 1. MPK_MULTI_DEV (DPX/CPX): every partition maps only
    its own AID's range, so one contiguous VA is built with HIP VMM from memory created on the
    first partition of each AID, mapped read-write for every partition. Returns the address.
    swap=True (timing control, read-only data only): region d on the other AID.
    """
    import ctypes as ct

    hip = _VMM.get("hip") or ct.CDLL("libamdhip64.so")
    _VMM["hip"] = hip
    mdev = mdev_count()
    if mdev <= 1:
        p = ct.c_void_p()
        rc = hip.hipMalloc(ct.byref(p), ct.c_size_t(nbytes))
        if rc == 0 and p.value:
            _register("hipmalloc", p.value, nbytes)
        return p.value if rc == 0 and p.value else 0

    class Loc(ct.Structure):
        _fields_ = [("type", ct.c_int), ("id", ct.c_int)]

    class AllocFlags(ct.Structure):
        _fields_ = [("compressionType", ct.c_ubyte), ("gpuDirectRDMACapable", ct.c_ubyte),
                    ("usage", ct.c_ushort)]

    class Prop(ct.Structure):
        _fields_ = [("type", ct.c_int), ("requestedHandleType", ct.c_int), ("location", Loc),
                    ("win32HandleMetaData", ct.c_void_p), ("allocFlags", AllocFlags)]

    class Access(ct.Structure):
        _fields_ = [("location", Loc), ("flags", ct.c_int)]

    region = nbytes // domains
    assert region * domains == nbytes and region % (2 << 20) == 0, (nbytes, domains)
    va = ct.c_void_p()
    if hip.hipMemAddressReserve(ct.byref(va), ct.c_size_t(nbytes), ct.c_size_t(2 << 20), None,
                                ct.c_ulonglong(0)) != 0:
        return 0
    prop = Prop()
    prop.type = 1           # hipMemAllocationTypePinned
    prop.location.type = 1  # hipMemLocationTypeDevice
    cur = ct.c_int()
    hip.hipGetDevice(ct.byref(cur))
    try:
        for d in range(domains):
            aid_d = (domains - 1 - d) if swap else d
            prop.location.id = aid_d * (mdev // domains)  # first partition of AID aid_d
            hip.hipSetDevice(prop.location.id)  # HIP 7.2 hipMemCreate uses the current device, not location.id
            h = ct.c_void_p()
            if (hip.hipMemCreate(ct.byref(h), ct.c_size_t(region), ct.byref(prop), ct.c_ulonglong(0)) != 0
                    or hip.hipMemMap(ct.c_void_p(va.value + d * region), ct.c_size_t(region),
                                     ct.c_size_t(0), h, ct.c_ulonglong(0)) != 0):
                print("[HALVES] VMM region %d of %d B failed" % (d, region), flush=True)
                return 0
    finally:
        hip.hipSetDevice(cur.value)
    acc = (Access * mdev)()
    for k in range(mdev):
        acc[k].location.type = 1
        acc[k].location.id = k
        acc[k].flags = 3    # hipMemAccessFlagsProtReadWrite
    if hip.hipMemSetAccess(va, ct.c_size_t(nbytes), acc, ct.c_size_t(mdev)) != 0:
        print("[HALVES] hipMemSetAccess failed", flush=True)
        return 0
    _register("vmm", va.value, nbytes, domains=domains, swap=bool(swap), mdev=mdev)
    return va.value


def copy_d2d(dst, src_tensor):
    """Raw device-to-device copy into an address that torch must not wrap (it would infer the
    owning partition from the pointer and move the view to another device)."""
    import ctypes as ct

    hip = _VMM.get("hip") or ct.CDLL("libamdhip64.so")
    _VMM["hip"] = hip
    n = src_tensor.numel() * src_tensor.element_size()
    rc = hip.hipMemcpy(ct.c_void_p(dst), ct.c_void_p(src_tensor.data_ptr()), ct.c_size_t(n), 3)
    if rc != 0:
        raise RuntimeError("[HALVES] hipMemcpy D2D of %d B failed: %d" % (n, rc))


class Arena:
    """Placement arena: a few chunks, each of D regions of C bytes (region d = domain d).

    A tensor placed in several domains takes the same offset X in each region of its
    chunk ("mirrored"), so its part in domain d is at chunk + d*C + X. The tensor
    attached to the graph is the domain-0 view; bind() rebases every instance in
    domain d by d*C (replicate) or d*(C - part) (split, private), part = bytes per
    domain.

    MI355X SPX+NPS2: a chunk is a plain hipMalloc of D*C. The patched driver's halves
    placement puts its first half in memory range 0 and its second half in range 1
    (C is a 2 MiB multiple, so the cut is exactly at C), and the memory stays MTYPE_NC
    like the copies it replaces. Chunks rather than one allocation because on this
    driver one hipMalloc above ~140 GiB fails, and a 125 GiB one failed once the model
    was resident; chunks also let the arena grow while demo.py frees the originals.
    """

    def __init__(self, chunk_per_domain, machine, align=4096):
        import ctypes as ct

        self.m = machine
        self.C = -(-int(chunk_per_domain) // machine.chunk) * machine.chunk
        self.align = align
        self.chunks = []  # dict(base, C, used)
        self.rings = []  # interleaved BOs: dict(base, slots, used)
        self.placed = OrderedDict()
        self._ct = ct
        self._hip = ct.CDLL("libamdhip64.so")

    @property
    def used(self):
        return sum(ch["used"] for ch in self.chunks)

    @property
    def reserved(self):
        return sum(ch["C"] for ch in self.chunks)

    def _new_chunk(self, part):
        import torch

        torch.cuda.empty_cache()  # return the originals demo.py has dropped
        need = -(-int(part) // self.m.chunk) * self.m.chunk
        C = max(self.C, need)
        while True:
            p = self._ct.c_void_p()
            if mdev_count() > 1:  # halves_malloc(self.m.domains * C): region d on AID d
                p.value, rc = halves_malloc(self.m.domains * C, self.m.domains), 0
            else:
                rc = self._hip.hipMalloc(self._ct.byref(p), self._ct.c_size_t(self.m.domains * C))
                if rc == 0 and p.value:
                    _register("hipmalloc", p.value, self.m.domains * C)
            if rc == 0 and p.value:
                break
            if C == need:
                raise RuntimeError("[ARENA] hipMalloc(%d) failed: rc=%d" % (self.m.domains * C, rc))
            C = max(need, (C // 2) // self.m.chunk * self.m.chunk)
        self.chunks.append(dict(base=p.value, C=C, used=0))
        print("[ARENA] chunk %d: %d x %s at 0x%x" % (len(self.chunks) - 1, self.m.domains,
                                                    _fmt(C), p.value), flush=True)
        return len(self.chunks) - 1

    def _bytes(self, addr, n):
        import torch

        class _CAI:
            pass

        c = _CAI()
        c.__cuda_array_interface__ = {"shape": (int(n),), "typestr": "|u1",
                                      "data": (int(addr), False), "version": 2,
                                      "strides": None}
        return torch.as_tensor(c, device="cuda")

    def place(self, t, name):
        """Copy t into the arena if its family is placed; return the domain-0 view."""
        import torch

        fam = family_of(name)[0]
        if fam in ARENA_RINGS and _arena_rings() and isinstance(t, torch.Tensor) and t.is_cuda:
            return self._place_ring(t, name)
        mode = ARENA_MODES.get(fam)
        if fam in MOE_SPLIT_FAMILIES and os.environ.get("MPK_MOE_SPLIT", "0") == "1":
            mode = "split"
        if mode is None or not isinstance(t, torch.Tensor) or not t.is_cuda:
            return None
        if fam in ARENA_1B and not _arena_1b():
            return None
        only = os.environ.get("MPK_ARENA_ONLY")  # bisection: place only these modes
        skip = os.environ.get("MPK_ARENA_SKIP")  # families left where torch put them (MPK_ROP_STAGE_ROW)
        if skip is not None and fam in skip.split(","):
            return None
        if only is not None and mode not in only.split(","):
            return None
        if name in self.placed:
            raise RuntimeError("[ARENA] %s placed twice" % name)
        D, xcds = self.m.domains, self.m.xcds
        n = t.numel() * t.element_size()
        if mode == "replicate":
            part = n
        else:
            if t.shape[0] % xcds:
                raise RuntimeError("[ARENA] %s: dim 0 = %d is not a multiple of %d XCDs"
                                   % (name, t.shape[0], xcds))
            part = n // D
        ci = None
        for i, ch in enumerate(self.chunks):  # first fit
            X = -(-ch["used"] // self.align) * self.align
            if X + part <= ch["C"]:
                ci = i
                break
        if ci is None:
            ci, X = self._new_chunk(part), 0
        ch = self.chunks[ci]
        ch["used"] = X + part
        base, C = ch["base"], ch["C"]
        src = t.contiguous().view(torch.uint8).view(-1)
        swap = w_swap() and fam in W_SWAP_FAMILIES and mdev_count() > 1
        for d in range(D):
            s_d = src if mode == "replicate" else src[d * part:(d + 1) * part]
            e = D - 1 - d if swap else d  # MPK_W_SWAP: domain d's part in the other region
            if mdev_count() > 1:  # copy_d2d(base + e * C + X: never a torch view of AID e > 0
                copy_d2d(base + e * C + X, s_d)
            else:
                self._bytes(base + e * C + X, part).copy_(s_d)
        torch.cuda.synchronize()
        view = self._bytes(base + X, n).view(t.dtype).view(t.shape)
        self.placed[name] = dict(mode=mode, chunk=ci, C=C, offset=X, nbytes=n, part=part,
                                 view=view, swap=swap)
        if swap and not self.__dict__.get("_swap_said"):
            self._swap_said = True
            print("[ARENA] MPK_W_SWAP: read-only weights stored on the other AID", flush=True)
        return view

    def _place_ring(self, t, name):
        """Copy a ring tensor so that it straddles one range boundary of an interleaved BO."""
        import torch

        from . import aid_hbm

        if name in self.placed:
            raise RuntimeError("[ARENA] %s placed twice" % name)
        n = t.numel() * t.element_size()
        lo = n // self.m.domains  # dies 0-3 write the first half
        ck = self.m.chunk
        if lo > ck or n - lo > ck:
            raise RuntimeError("[ARENA] ring %s: %d B does not fit around one boundary" % (name, n))
        ring = self.rings[-1] if self.rings else None
        if ring is None or ring["used"] == ring["slots"]:
            torch.cuda.empty_cache()
            base = aid_hbm.alloc_ilv(RING_SLOTS * 2 * ck)
            ring = dict(base=base, slots=RING_SLOTS, used=0)
            self.rings.append(ring)
            print("[ARENA] ring BO %d: %d slots, %s at 0x%x (2 MiB interleave)" % (
                len(self.rings) - 1, RING_SLOTS, _fmt(RING_SLOTS * 2 * ck), base), flush=True)
        k = ring["used"]
        ring["used"] += 1
        addr = ring["base"] + (2 * k + 1) * ck - lo
        self._bytes(addr, n).copy_(t.contiguous().view(torch.uint8).view(-1))
        torch.cuda.synchronize()
        view = self._bytes(addr, n).view(t.dtype).view(t.shape)
        self.placed[name] = dict(mode="straddle", chunk=-1, C=0, offset=addr - ring["base"],
                                 nbytes=n, part=lo, view=view)
        return view

    def view(self, name):
        p = self.placed.get(name)
        return p["view"] if p else None

    def rebase(self, name, inst):
        """Byte offset to add to instance `inst`'s pointer for tensor `name`."""
        p = self.placed.get(name)
        if p is None:
            return 0
        if p["mode"] == "straddle":
            return 0  # one contiguous tensor; the boundary inside it does the split
        d = inst // self.m.xcds_per_domain
        e = self.m.domains - 1 - d if p.get("swap") else d  # region holding domain d's part
        if p["mode"] == "replicate":
            return e * p["C"]
        return e * p["C"] - d * p["part"]

    def bind(self, graph_text):
        """Rewrite task_graph.json so each instance points at its own domain's part."""
        g = json.loads(graph_text)
        groups = OrderedDict()
        prev_key, inst = None, 0
        for t in g["all_tasks"]:
            refs = (t.get("inputs") or []) + (t.get("outputs") or [])
            if not refs:
                continue
            layer = next((m.group(1) for r in refs
                          for m in [re.match(r"layer_(\d+)_", r["base_ptr"])] if m), None)
            key = (t["task_type"], layer)
            inst = inst + 1 if key == prev_key else 0
            prev_key = key
            hit = [r for r in refs if r["base_ptr"] in self.placed]
            if hit:
                groups.setdefault(key, []).append((inst, hit))
        n = 0
        for key, members in groups.items():
            if len(members) != self.m.xcds or [i for i, _ in members] != list(range(self.m.xcds)):
                raise RuntimeError("[ARENA] task %s: %d instances reference placed tensors,"
                                   " expected one per XCD" % (key, len(members)))
            for inst, refs in members:
                for r in refs:
                    delta = self.rebase(r["base_ptr"], inst)
                    if delta:
                        r["offset"] = int(r["offset"]) + delta
                        n += 1
        return json.dumps(g), n, len(groups)

    def summary(self):
        by = OrderedDict()
        for name, p in self.placed.items():
            fam = family_of(name)[0]
            e = by.setdefault(fam, [p["mode"], 0, 0])
            e[1] += 1
            e[2] += p["part"]
        return by


def main(argv):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("graph", help="task_graph.json")
    ap.add_argument("--env-from", help="run log with the ENVS line (fleet_run_lane.sh)")
    ap.add_argument("--out", help="directory for place_report.json / .txt")
    ap.add_argument("--machine", default="mi355x_spx_nps2", choices=sorted(MACHINES))
    a = ap.parse_args(argv)
    env = env_from_log(a.env_from) if a.env_from else dict(os.environ)
    with open(a.graph) as f:
        graph = json.load(f)
    rep = report(graph, env, out_dir=a.out, machine=a.machine)
    return 1 if rep["summary"]["unexplained"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
