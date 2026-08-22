#!/usr/bin/env python3
"""ITEM 2 (width-2 / EAGLE double draft) GO-NO-GO, priced against measured C.

Every input here is measured on this branch, one variable per run, gated with
correctness_gate.py (G1+G2 hard, G3 advisory).  Nothing is extrapolated except
C(3 rows), and the verdict is shown to be insensitive to that one term.

  bs=1 control          10.619 ms/iter   n=3, probe_row_cost_wall.sh
  bs=2 plain (no MTP)   12.809 ms/iter   n=2, same probe -> marginal row +2.190
  MTP width-1           16.530 ms/iter   n=3, /tmp/mtp_regate  (accept 0.861)

The board's C = 1.562 is reproduced EXACTLY by 16.530/10.619 = 1.557.  So the
board was pricing the speculation HARNESS, not the decode row.  Split:

    second decode row       +2.190 ms   (C_row = 1.206, confirmed 369032c)
    MTP draft-layer harness +3.721 ms   <-- 1.7x the cost of a whole extra row

Run:  python3 demo/glm5/price_item2_width2.py
"""

BS1        = 10.619   # ms/iter, bs=1 control, n=3
BS2        = 12.809   # ms/iter, --max-num-batched-tokens 2, n=2
MTP1       = 16.530   # ms/iter, --mtp 1, n=3
A1         = 0.861    # measured acceptance of the first draft token, n=3

ROW        = BS2 - BS1     # marginal cost of one extra decode row
HARNESS    = MTP1 - BS2    # cost of one draft-layer invocation + accept/reject


def mspt(ms_iter, toks_iter):
    return ms_iter / toks_iter


def width2(a2, row3=None, harness=None):
    """3 decode rows + 2 sequential draft-layer invocations."""
    row3 = ROW if row3 is None else row3
    harness = HARNESS if harness is None else harness
    ms_iter = BS1 + 2 * row3 + 2 * harness
    toks = 1.0 + A1 + A1 * a2
    return ms_iter, toks, mspt(ms_iter, toks)


def breakeven_a2(target_mspt, row3=None, harness=None):
    """a2 at which width-2 ties `target_mspt`."""
    row3 = ROW if row3 is None else row3
    harness = HARNESS if harness is None else harness
    ms_iter = BS1 + 2 * row3 + 2 * harness
    need_toks = ms_iter / target_mspt
    return (need_toks - 1.0 - A1) / A1


W1_MSPT = mspt(MTP1, 1.0 + A1)

print("=" * 74)
print("MEASURED INPUTS")
print("=" * 74)
print(f"  bs=1 control            {BS1:7.3f} ms/iter   = {BS1:6.3f} ms/token")
print(f"  bs=2 plain              {BS2:7.3f} ms/iter   marginal row  {ROW:+.3f}")
print(f"  MTP width-1             {MTP1:7.3f} ms/iter   harness       {HARNESS:+.3f}")
print(f"  MTP acceptance a1        {A1:6.3f}       -> {1+A1:.3f} tok/iter")
print(f"  MTP width-1                             = {W1_MSPT:6.3f} ms/token"
      f"  ({BS1/W1_MSPT:.3f}x)")
print(f"\n  board C=1.562 reproduced as MTP/bs1 = {MTP1/BS1:.3f}"
      f"   (row alone is only {BS2/BS1:.3f})")

print("\n" + "=" * 74)
print("WIDTH-2 (item 2): 3 decode rows + 2 sequential drafts")
print("=" * 74)
mi, _, _ = width2(0.0)
print(f"  fixed cost/iter = {BS1:.3f} + 2*{ROW:.3f} + 2*{HARNESS:.3f}"
      f" = {mi:.3f} ms/iter")
print(f"\n  {'a2':>6} {'tok/iter':>9} {'ms/token':>9}   vs width-1 ({W1_MSPT:.3f})")
for a2 in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.773, 0.8, 0.861):
    m, t, s = width2(a2)
    print(f"  {a2:6.3f} {t:9.3f} {s:9.3f}   {'WIN ' if s < W1_MSPT else 'LOSE'}"
          f" {s - W1_MSPT:+.3f}")

be_w1 = breakeven_a2(W1_MSPT)
be_bs1 = breakeven_a2(BS1)
print(f"\n  BREAK-EVEN a2 vs width-1 MTP : {be_w1:.3f}")
print(f"  BREAK-EVEN a2 vs no spec     : {be_bs1:.3f}")

print("\n  sensitivity -- verdict does not depend on the one unmeasured term,")
print("  C(3 rows).  Sweeping the third row's marginal cost +/-60%:")
for r3 in (1.5, 2.190, 3.0, 4.0, 5.0):
    print(f"    3rd row {r3:+.3f} ms -> break-even a2 vs width-1 ="
          f" {breakeven_a2(W1_MSPT, row3=r3):.3f}")

print("\n" + "=" * 74)
print("VERDICT: NO-GO")
print("=" * 74)
print(f"""  Width-2 needs a2 > {be_w1:.3f} merely to BEAT width-1.  a1 itself is only
  {A1:.3f}, and a second autoregressive draft token always accepts materially
  lower than the first, so a2 > {be_w1:.3f} is unreachable.  Robust across a
  +/-60% swing in C(3 rows), so measuring the third row would not change it.

  The blocker is NOT the decode row.  The board assumed C=1.562 made rows
  expensive; the row is {ROW:.3f} ms and is fine.  The blocker is the draft-layer
  HARNESS at {HARNESS:.3f} ms, which width-2 must pay TWICE.

  What the harness should cost: the draft is ONE layer.  At the measured bs=2
  rate of 186.301 us/layer that is 0.186 ms.  It is billed at {HARNESS:.3f} ms --
  20x -- because layer 76 is dispatched as a SEPARATE megakernel replay run
  (log: "Multi-layer table: 2 runs -- layers 0..75 batched, layer 76 dispatched
  separately").

  So the next build is the harness, not the draft chain.  If the draft layer
  were folded to its true ~0.19 ms cost:""")
for h in (HARNESS, 1.0, 0.5, 0.186):
    m1 = BS2 + h
    s1 = mspt(m1, 1 + A1)
    s2 = width2(0.5, harness=h)[2]
    print(f"    harness {h:5.3f} ms -> width-1 {s1:6.3f} ms/token"
          f" ({BS1/s1:.2f}x)   width-2@a2=0.5 {s2:6.3f}"
          f"   break-even a2 {breakeven_a2(s1, harness=h):.3f}")
print(f"""
  i.e. cutting the harness is worth ~{W1_MSPT - mspt(BS2+0.186, 1+A1):.2f} ms/token on its own AND drops
  item 2's break-even a2 into reachable territory.  Item 2 is therefore
  DEFERRED behind the harness fix, not abandoned.""")
