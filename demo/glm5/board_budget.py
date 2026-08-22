#!/usr/bin/env python3
"""THE BOARD: the bs=1 layer budget, every region LABELLED FROM SOURCE, with
each region's ledger verdict.

Why this exists.  classify_geom.py answered "what does the SECOND row cost".
This answers the prior question -- "where does the FIRST row's 10.619 ms go,
and which parts of it already have a verdict" -- so that naming the next lever
is a lookup instead of a guess.

LABEL CORRECTION, and it was published wrong twice (26f8aab, a9018e5).
`S2->S16` was labelled "qkv_a tiles".  It is not: it is the call boundary
between the layer task and the attention body, worth 0.094 ms.  The real qkv_a
tiles are `S16->S17`, 0.900 ms.  The GEOM totals are unaffected -- S16->S17 was
already summed, just in the unclassified bucket, and every region the
classification actually leaned on (o_proj 29->30, W13 5->6, W2 7->8, EP 0->1,
router 30->31 / 31->32) is confirmed correct by the grep below.

HOW THE MAP WAS DERIVED, so it can be re-checked in one command:

    grep -rn "mpk_stage_stamp(" include/mirage/persistent_kernel/

Every id has exactly ONE call site.  Two files stamp, and BOTH run every layer:
gang_mla_full_layer_fused_mi300.cuh:1405 `#include`s and CALLS
gang_mla_attn_fused_kernel_mi300 immediately after stamp 2, so program order
interleaves the two files.  Confirmed by population: every slot carries ~38300
counts per worker, i.e. all of them fire on all 76 layers.  A slot's WORKER SET
still varies (slot 20 has 64, slot 22 has 128, slots 29-31 have 192, slots 3/4
have 8) -- see the SUBSET notes.

Reads /tmp/item11_bs1.log (the bs=1 arm of probe_row_cost_phases.sh).
Offline; no GPU run.
"""
import re
import collections

LOG = "/tmp/item11_bs1.log"
N_LAYERS = 76

# (from_slot, to_slot, label, ledger verdict)
REGIONS = [
    (0, 1, "EP collective",
     "1.187  hoist NEUTRAL, widen NEUTRAL; 5.4us local fold + 17.5us peer wait"),
    (2, 16, "-> attn call boundary", "0.094  not a phase; the task-body call"),
    (16, 17, "qkv_a tiles",
     "0.900  41% redundant prologue; K-loop at 90% of the per-CU byte roof"),
    (17, 18, "qkv_a -> q_b barrier", "0.393  barrier narrowing measured out"),
    (18, 19, "q_b tiles + KV append",
     "0.937  q_b GEMM at the HBM roof; 41% is RMSNorm+quant prologue"),
    (19, 21, "q_b->decode bar + MLA decode",
     "1.392  ABLATION was 3.40, HARVESTED -2.76 by 8-token KV chunks"),
    (21, 23, "decode->merge bar + merge",
     "0.415  216 workers idle through a 16-worker phase; serial decode"),
    (23, 24, "Phase 8 attn -> o_proj barrier",
     "0.170  straggler, not mechanism: 91% spin, the atomic is 0.49us"),
    (24, 25, "o_proj weight prefetch issue", "0.075  NESTED"),
    (25, 26, "per-XCD attention release", "0.193  NESTED"),
    (26, 27, "prefetch DMA retired", "0.088  NESTED; the prefetch hides the fetch"),
    (27, 28, "entering the MoE half", "0.065"),
    (28, 29, "W_UK / W_UV + Mechanism-C bar",
     "0.756  75% SPIN; re-absorbing W_UK/W_UV does NOT pay"),
    (29, 30, "o_proj GEMV + residual",
     "0.562  76% of HBM peak; -11.9% already taken by the ep_signal shard"),
    (30, 31, "o_proj -> router barrier", "0.204"),
    (31, 32, "router GEMV + sigmoid/bias TopK",
     "1.076  k-loop is 72%; rank-select +33%, fold 96ns, bias prefetch dead"),
    (32, 5, "routing-ready poll", "0.317"),
    (5, 6, "W13 tiles",
     "1.402  AT THE HBM ROOF; OPW=16 -1.34ms, split-K +4.12ms, prefetch null"),
    (6, 7, "W13 -> W2 barrier", "0.404"),
    (7, 8, "W2 tiles",
     "0.943  latency-bound; L2 prefetch neutral, staging pays only at K_SPLITS=2"),
    (8, 12, "MoE exit", "0.255"),
    (12, 13, "", "0.136"),
    (13, 14, "layer boundary",
     "0.058  ablates to ZERO; all bookkeeping is 0.051 ms"),
]

SUBSET = {3: 8, 4: 8, 20: 64, 22: 128, 29: 192, 30: 192, 31: 192}


def load(path):
    pat = re.compile(r"\[1,(\d+)\].*BARSTAGEWS (\d+) (\d+) (\d+) (\d+)")
    d = collections.defaultdict(dict)
    for ln in open(path, errors="ignore"):
        m = pat.search(ln)
        if m:
            r, s, w, c, t = (int(x) for x in m.groups())
            if r == 0 and c:
                d[s][w] = (c, t)
    return d


def mk(rows, s):
    """Makespan: the LAST worker out of slot s, us."""
    return max(t / c for c, t in rows[s].values()) / 1000.0


def md(rows, s):
    """Typical: the MEDIAN worker, us.  crit-typ is the spin."""
    v = sorted(t / c for c, t in rows[s].values())
    return v[len(v) // 2] / 1000.0


R = load(LOG)
span = mk(R, 14) - mk(R, 0)

print("=" * 92)
print("THE bs=1 LAYER BUDGET, rank 0, labelled from source")
print("=" * 92)
print(f"{'region':<36}{'crit':>7}{'typ':>7}{'spin':>7}{'ms/tok':>8}{'%':>6}")
rows = []
for a, b, lab, _ in REGIONS:
    if a not in R or b not in R:
        continue
    ca, cb = [sum(c for c, _ in R[s].values()) // len(R[s]) for s in (a, b)]
    flag = " [POP!]" if abs(ca - cb) > 0.01 * max(ca, cb) else ""
    sub = " *" if (a in SUBSET) != (b in SUBSET) else ""
    crit, typ = mk(R, b) - mk(R, a), md(R, b) - md(R, a)
    rows.append((f"S{a}->S{b} {lab}{sub}{flag}", crit, typ, crit * N_LAYERS / 1e3))
for n, c, t, ms in rows:
    print(f"{n:<36}{c:7.2f}{t:7.2f}{c-t:7.2f}{ms:8.3f}{100*c/span:6.1f}")
print(f"{'LAYER SPAN S0->S14':<36}{span:7.2f}{'':7}{'':7}{span*N_LAYERS/1e3:8.3f}")
print("""
  * = the two slots have DIFFERENT worker sets, so the region is a makespan
      over a subset and can be understated.  Never a credit, only a floor.
  A negative 'spin' is an ORDERING artifact, not a saving: the median worker
  crosses the two stamps in a different order than the makespan worker does
  (S21->S23 reads -20.15 this way).  Do not sum the spin column.""")

print()
print("=" * 92)
print("RANKED, WITH THE LEDGER VERDICT -- read this before naming a lever")
print("=" * 92)
for n, c, t, ms in sorted(rows, key=lambda x: -x[3]):
    a = int(n.split("->")[0][1:])
    b = int(n.split("->")[1].split()[0][1:])
    verdict = next(v for x, y, _, v in REGIONS if (x, y) == (a, b))
    print(f"{n:<42}{verdict}")

print("""
FINDING OF THIS AUDIT: there is no un-attacked region left above 0.4 ms.
Every one of the top twelve carries a measured verdict already.  The layer is
not hiding a lever -- it is a flat distribution of ~1 ms terms, most of them
either at the HBM roof (W13, q_b, o_proj, qkv_a) or majority spin (MLA decode
16.6us of 18.3; W_UV 75%; Phase 8 91%).  That is the same conclusion the GEOM
close reached from the other direction, and it is why the remaining distance to
2 ms is a STRUCTURAL question (how many rendezvous a layer needs at all), not a
tuning question about any single region.""")
