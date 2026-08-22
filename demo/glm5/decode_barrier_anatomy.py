#!/usr/bin/env python3
"""What is the 0.643 ms that SURVIVES the MLA decode ablation?

The decode ablation (f02753c, probe_decode_ablation_ceiling.sh) cleaved
board_budget.py's S19->S21 region (18.31 us/layer crit = 1.392 ms) into

    decode's own WORK      9.86 us/layer  0.749 ms   (deleted by the ablation)
    barrier + skew         8.45 us/layer  0.643 ms   (SURVIVES it)

Overlap was closed as a NO-GO on the WORK half: the legally-movable set is
empty (glm-decode-overlap-candidate-set-is-empty).  The surviving half has
never been attacked directly.  This script asks the only question that decides
whether it CAN be: is it ARRIVAL SKEW carried in from q_b (in which case the
region is a symptom and the lever is upstream -- and the ledger already says
counted region time before a barrier is not a lever), or is it UNIFORM SPIN
(232 workers all waiting the same amount on a serial 64-worker phase, in which
case it is structural and only fewer/wider phases move it)?

The decisive split is free: slot 20 is stamped by EXACTLY the 64 decode
workers and by no one else, so the log itself partitions the machine into

    D = the 64 workers that do decode tiles
    I = the 168 workers that idle through the phase

Both sets stamp S19 and S21, so their S19->S21 deltas are directly comparable.

  * If the cost lives in D and is spread, it is decode tile imbalance.
  * If the cost lives in I as a flat block, it is the serial-phase hole and it
    is exactly the thing overlap was going to fill -- already priced at zero.
  * If the SPREAD AT S19 is already most of it, the region is inheriting q_b's
    straggler and nothing inside the region is the lever.

================================ RESULT ================================
The survivor is NOT a local barrier.  It is the QB_TP CROSS-RANK peer wait,
8.78 us/layer = 0.667 ms, and the layer therefore pays TWO cross-rank
rendezvous rather than one:

    S0->S1   EP collective            17.5 us/layer
    S19->S20 q_b head-shard gather     8.8 us/layer   <- newly attributed

Two independent routes agree to 4%: region_crit(18.31) - ablation(9.86) =
8.45, and median_wait(12.68) - local_arrival_skew(3.90) = 8.78.

VERDICT: NO-GO, and it needed no GPU run.  Three source facts kill it.

 1. THE PRECONDITION IS ALREADY AS EARLY AS IT CAN LEGALLY BE.  The obvious
    lever is "post the peer signal before the full local barrier, since the
    data was ready earlier".  It is empty: the head slice is ALREADY pushed
    per-tile by all eight XCDs from inside the W_UK loop (:1061 "pushed here
    rather than by the barrier leader so all eight XCDs drive the links while
    W_UK is still running"), and NOTHING runs between the loop's close and
    stamp 19 except the subphase-timing block.  The KV-cache append is Phase
    3, upstream, not between them.  So "all my pushes are done" and "the local
    barrier is satisfied" are the same instant, and the signal cannot be
    hoisted.  The overlap this lever wanted is already taken.
 2. THE MECHANISM IS ALREADY MEASURED AT ZERO.  Batching the seven serialized
    peer loads into one round trip moved the twin EP wait 17.49 -> 17.49
    (glm-ep-peer-poll-is-not-the-cost).  A poll pass is amortized over a wait
    an order of magnitude longer than itself.
 3. THE CONTENT IS INTER-RANK SKEW WITH THE WRONG PRODUCER SET.  Rule 3 of
    glm-deleting-a-whole-rendezvous-is-neutral: a uniform wait is only a lever
    if its producer set DIFFERS from the neighbouring rendezvous', else the
    skew relocates instead of vanishing.  This wait's producers are the 8
    ranks -- identical to the EP collective's.  That branch already deleted a
    whole rendezvous with 8.28 us/layer of uniform spin and moved the wall
    0.003 ms.

Also fixed here: the board's S21->S23 note.  "216 workers idle through a
16-worker phase" is stale geometry (64/168 since the KV-chunk harvest) and
mislocated -- see section D.
========================================================================

Reads /tmp/item11_bs1.log.  Offline; no GPU run.  NOTE this is the BASE build
(decode present) -- the ablated arms ran with MPK_BAR_SKEW=0 on purpose, since
the instrument's cost scales with tile count and decode tile count was the
variable (glm-subphase-timing-cost-scales-with-tile-count).
"""
import re
import collections

LOG = "/tmp/item11_bs1.log"
N_LAYERS = 76


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


R = load(LOG)
# per-worker mean timestamp at a slot, us
T = {s: {w: t / c / 1000.0 for w, (c, t) in R[s].items()} for s in R}

D = set(T[20])                      # the 64 decode workers, from the log itself
ALL = set(T[19]) & set(T[21])
I = ALL - D

assert len(D) == 64, len(D)
print(f"decode set D = {len(D)} workers, idle set I = {len(I)} workers")


def q(vals):
    v = sorted(vals)
    n = len(v)
    return v[0], v[n // 2], v[-1]


def show(name, vals):
    lo, med, hi = q(vals)
    print(f"  {name:<34}{lo:8.2f}{med:8.2f}{hi:8.2f}{hi-lo:9.2f}"
          f"{(hi-lo)*N_LAYERS/1e3:9.3f}")


print()
print("=" * 92)
print("A. ARRIVAL SPREAD AT EACH STAMP (per-worker mean, us) -- is the skew")
print("   already there when the region STARTS?")
print("=" * 92)
print(f"  {'stamp':<34}{'min':>8}{'med':>8}{'max':>8}{'spread':>9}{'ms/tok':>9}")
for s, lab in ((18, "S18 qkv_a->q_b bar passed"),
               (19, "S19 q_b + KV append done"),
               (21, "S21 decode tiles done"),
               (23, "S23 merge done")):
    base = min(T[s].values())
    show(lab, [v - base for v in T[s].values()])

print("""
  Read: 'spread' is max-min of the per-worker MEAN, so it is the steady-state
  arrival skew at that stamp, not a one-off straggler.""")

print()
print("=" * 92)
print("B. THE REGION, SPLIT BY WHO ACTUALLY WORKS  (S19->S21, us/layer)")
print("=" * 92)
print(f"  {'set':<34}{'min':>8}{'med':>8}{'max':>8}{'spread':>9}{'ms/tok':>9}")
d19_21_D = [T[21][w] - T[19][w] for w in D]
d19_21_I = [T[21][w] - T[19][w] for w in I]
show("D: 64 decode workers", d19_21_D)
show("I: 168 idle workers", d19_21_I)

print()
print(f"  median D {q(d19_21_D)[1]:.2f} us   median I {q(d19_21_I)[1]:.2f} us"
      f"   difference {q(d19_21_D)[1]-q(d19_21_I)[1]:+.2f} us/layer")

print()
print("=" * 92)
print("C. INSIDE THE DECODE SET: barrier wait vs tile work")
print("=" * 92)
print(f"  {'segment (D only)':<34}{'min':>8}{'med':>8}{'max':>8}{'spread':>9}{'ms/tok':>9}")
show("S19->S20 q_b->decode barrier", [T[20][w] - T[19][w] for w in D])
show("S20->S21 decode tiles", [T[21][w] - T[20][w] for w in D])

print()
print("=" * 92)
print("D. WHERE DOES THE IDLE SET ACTUALLY WAIT?  (median us/layer per set)")
print("=" * 92)
print(f"  {'segment':<34}{'D (64)':>10}{'I (168)':>10}{'  who waits'}")
for a, b, lab in ((19, 21, "S19->S21 qb bar + decode"),
                  (21, 23, "S21->S23 dec->merge bar + merge"),
                  (23, 24, "S23->S24 Phase 8 attn->o_proj bar")):
    dv = q([T[b][w] - T[a][w] for w in D & set(T[b])])[1]
    iv = q([T[b][w] - T[a][w] for w in I & set(T[b])])[1]
    who = "DECODE SET" if dv > 2 * iv else ("IDLE SET" if iv > 2 * dv else "both")
    print(f"  {lab:<34}{dv:10.2f}{iv:10.2f}   {who}")

print("""
  The board's S19->S21 note ("216 workers idle through a 16-worker phase") is
  WRONG TWICE: the geometry is 64/168 not 16/216 since the KV-chunk harvest,
  and the idle set does NOT idle there at all -- it crosses S19->S21 in 1.70
  us.  Only the 64 decode workers are inside that region.  The machine-wide
  idle hole is one region LATER, at the decode->merge rendezvous.""")

print()
print("=" * 92)
print("D2. THE ATTENTION TAIL IS A THREE-WAY SPLIT, AND MOST STAMPS IN IT ARE")
print("    NOT SYNCHRONIZATION POINTS.  Per-set MEDIAN, us after S18.")
print("=" * 92)
M = set(T[22]) - D                      # the merge workers drawn from I
P = set(T[19]) - D - M                  # stamp neither 20 nor 22
print(f"  sets: decode D={len(D)}  merge M={len(M)}  neither P={len(P)}")
ref = q(list(T[18].values()))[1]
print(f"  {'stamp':<32}{'D':>9}{'M':>9}{'P':>9}")
for s, lab in ((18, "S18 qkv_a->q_b bar passed"),
               (19, "S19 q_b + KV append done"),
               (20, "S20 qb->dec bar passed"),
               (21, "S21 decode tiles done"),
               (22, "S22 dec->merge bar passed"),
               (23, "S23 merge done"),
               (24, "S24 Phase 8 bar passed"),
               (28, "S28 enter MoE half")):
    cells = []
    for S in (D, M, P):
        sub = S & set(T[s])
        cells.append(f"{q([T[s][w] for w in sub])[1]-ref:9.2f}" if sub else f"{'-':>9}")
    print(f"  {lab:<32}{''.join(cells)}")

print("""
  THE TRAP, and it invalidates a class of arithmetic on this board: a stamp is
  only a synchronization point for workers that PARTICIPATE in the phase it
  ends.  P crosses S21, S23 and S24 in program order without doing decode or
  merge, so those stamps record P falling through, ~19-24 us EARLIER than D
  records them.  A region difference is therefore only meaningful when both
  stamps are crossed by the SAME population on the SAME path.

  The two real rendezvous in the tail are the two where the sets agree:
      S22 dec->merge  D and M within 0.00 us
      S28 enter MoE   D and P within 0.03 us
  Everything between S19 and S28 is per-set program order, not a machine-wide
  region.  board_budget.py's S21->S23 = 5.46 us mixes populations and must be
  read as the decode set's drain, never as machine-wide time.

  What the split actually says: 104 of 232 workers -- 45% of the machine --
  take no part in the attention tail at all and idle from S21 to S28, about
  33.5 us of a ~159 us layer.  That is a far larger and better-located hole
  than the board's "216 workers idle through a 16-worker phase" claimed, and
  it is STILL unfillable for the reason already priced: the movable-work set
  is empty (glm-decode-overlap-candidate-set-is-empty) and handing idle
  workers a cold prefetch measured +0.030
  (glm-moe-phase-has-a-free-worker-hole).""")

print()
print("=" * 92)
print("E. THE BUDGET OF THE REGION, AND WHAT THE SURVIVOR IS")
print("=" * 92)
region = max(T[21].values()) - max(T[19].values())
print(f"  region crit (makespan S19->S21)          {region:7.2f} us/layer"
      f"  {region*N_LAYERS/1e3:7.3f} ms")
print(f"  ablation says decode WORK is             {9.86:7.2f} us/layer"
      f"  {0.749:7.3f} ms   (f02753c)")
print(f"  => surviving barrier+skew                {region-9.86:7.2f} us/layer"
      f"  {(region-9.86)*N_LAYERS/1e3:7.3f} ms")
print()
tile_mk = q([T[21][w] - T[20][w] for w in D])[2]
print(f"  CROSS-CHECK, and it is a good one: the instrument's decode-tile")
print(f"  makespan over D is {tile_mk:.2f} us/layer against the ablation's 9.86.")
print(f"  {100*abs(tile_mk-9.86)/9.86:.0f}% apart -- MPK_MLA_SKIP_DECODE deletes")
print( "  exactly the tile loop and nothing else, confirmed independently.")
print()
qb_bar_med = q([T[20][w] - T[19][w] for w in D])[1]
s19_skew = q(list(T[19].values()))[2] - q(list(T[19].values()))[1]
print(f"  The survivor is the S19->S20 wait: median {qb_bar_med:.2f} us/layer.")
print(f"  Local arrival skew at S19 (max-med) explains only {s19_skew:.2f} of it.")
print(f"  Remainder = {qb_bar_med-s19_skew:.2f} us/layer = "
      f"{(qb_bar_med-s19_skew)*N_LAYERS/1e3:.3f} ms.  That is the QB_TP")
print( "  CROSS-RANK peer wait (gang_mla_attn_fused_mi300.cuh:1193-1240): the")
print( "  elected leader stores this rank's head slice to all 7 peers'")
print( "  ep_signal lines and polls all 7 before releasing the local flag.")
print("""
  SO: the layer pays TWO cross-rank rendezvous, not one --
      S0->S1   EP collective          17.5 us/layer   (glm-ep-peer-poll-is-not-the-cost)
      S19->S20 q_b head-shard gather  ~8.8 us/layer   (THIS, newly attributed)
  The decode ablation kept this barrier, which is exactly why 0.643 ms
  survived it.  The two numbers agree to 4%.""")
