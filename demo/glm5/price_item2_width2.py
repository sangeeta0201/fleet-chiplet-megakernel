#!/usr/bin/env python3
"""ITEM 2 (width-2 / EAGLE double draft) GO-NO-GO, priced against measured C.

REWRITTEN 2026-08-22 (was 38a7f75).  The original priced item 2 as
`BS1 + 2*row + 2*harness` with row = 2.190 and harness = 3.721.  Both terms
were wrong, in opposite directions:

  * `--max-num-batched-tokens 2` decodes ONE token per iteration, so its
    +2.186 is 2-wide BUILD GEOMETRY, not a decode row (a015f60).
  * a GLM_MTP_GRAPH_ONLY=1 arm -- the whole MTP chain in the task graph with
    one active row -- put the chain at 0.494 ms and the real second decode
    row at 3.181 (b980171).  The "harness" was 13% harness and 87% row.

The corrected model has three independently measured terms instead of two:

    GEOM    cost of widening the build by one row, before any row is live
    CHAIN   one draft-layer invocation: embed gather, enorm/hnorm, both
            eh_proj GEMVs, the draft layer as its own replay run, 2nd lm_head
    ROWACT  cost of ACTIVATING one more decode row on a build already that wide

Width-2 needs 3 rows (1 committed + 2 drafts) and 2 sequential chain
invocations, so it pays GEOM twice, ROWACT twice, CHAIN twice.

Run:  python3 demo/glm5/price_item2_width2.py
"""

# ---- measured, one variable per run, all arms G1+G2 PASS -------------------
BS1       = 10.619   # bs=1 control                                   n=3
BS2       = 12.805   # 2-wide build, ONE live row (in-batch anchor)   n=2
CHAINONLY = 13.299   # + whole MTP chain, still ONE live row          n=1 *
MTP1      = 16.480   # + second row live (--mtp 1)                    n=2
A1        = 0.861    # measured acceptance of the first draft token   n=3
# * chainonly reps 2 and 3 wedged at launch (the ~1-in-3 NP=8 hang) and were
#   killed; probe_mtp_chain_topup.sh is topping this up.  The sensitivity
#   sweep below shows the verdict does not depend on it.

GEOM   = BS2 - BS1          # 2.186  build width, zero extra tokens
CHAIN  = CHAINONLY - BS2    # 0.494  one draft-layer invocation
ROWACT = MTP1 - CHAINONLY   # 3.181  activating the second row

C_GEOM = BS2 / BS1          # 1.206  <- the board's "C"; prices WIDTH, not a row
C_ROW  = (BS1 + GEOM + ROWACT) / BS1   # a second LIVE row, end to end


def mspt(ms_iter, toks):
    return ms_iter / toks


def width2_iter(geom=None, rowact=None, chain=None):
    geom = GEOM if geom is None else geom
    rowact = ROWACT if rowact is None else rowact
    chain = CHAIN if chain is None else chain
    return BS1 + 2 * geom + 2 * rowact + 2 * chain


def width2(a2, **kw):
    ms = width2_iter(**kw)
    toks = 1.0 + A1 + A1 * a2
    return ms, toks, mspt(ms, toks)


def breakeven_a2(target_mspt, **kw):
    ms = width2_iter(**kw)
    return (ms / target_mspt - 1.0 - A1) / A1


W1_MSPT = mspt(MTP1, 1.0 + A1)

print("=" * 76)
print("MEASURED INPUTS  (each arm differs from the one above it by ONE thing)")
print("=" * 76)
print(f"  bs=1 control                {BS1:7.3f} ms/iter")
print(f"  + 2-wide build, 1 live row  {BS2:7.3f}   GEOM   {GEOM:+.3f}  zero tokens")
print(f"  + whole MTP chain, 1 row    {CHAINONLY:7.3f}   CHAIN  {CHAIN:+.3f}")
print(f"  + second row live (MTP)     {MTP1:7.3f}   ROWACT {ROWACT:+.3f}")
print(f"\n  MTP acceptance a1 = {A1:.3f} -> {1+A1:.3f} tok/iter"
      f" = {W1_MSPT:.3f} ms/token ({BS1/W1_MSPT:.3f}x)")
print(f"\n  C = BS2/BS1 = {C_GEOM:.3f}  <- confirmed, but it prices BUILD WIDTH.")
print(f"  A second LIVE row end to end is GEOM+ROWACT = {GEOM+ROWACT:.3f} ms,"
      f" i.e. C_row = {C_ROW:.3f}.")

print("\n" + "=" * 76)
print("WIDTH-2: 3 rows (1 committed + 2 drafts), 2 sequential chain calls")
print("=" * 76)
mi = width2_iter()
print(f"  {BS1:.3f} + 2*{GEOM:.3f} + 2*{ROWACT:.3f} + 2*{CHAIN:.3f}"
      f" = {mi:.3f} ms/iter")
print(f"\n  {'a2':>6} {'tok/iter':>9} {'ms/token':>9}   vs width-1 ({W1_MSPT:.3f})")
for a2 in (0.2, 0.4, 0.6, 0.7, 0.769, 0.8, A1, 1.0):
    m, t, s = width2(a2)
    print(f"  {a2:6.3f} {t:9.3f} {s:9.3f}   "
          f"{'WIN ' if s < W1_MSPT else 'LOSE'} {s - W1_MSPT:+.3f}")

be_w1 = breakeven_a2(W1_MSPT)
print(f"\n  BREAK-EVEN a2 vs width-1 MTP : {be_w1:.3f}")
print(f"  BREAK-EVEN a2 vs no spec     : {breakeven_a2(BS1):.3f}")
print(f"  a2 = 1.0 (a PERFECT second draft) gives"
      f" {width2(1.0)[2]:.3f} ms/token vs {W1_MSPT:.3f}")

print("\n" + "=" * 76)
print("SENSITIVITY -- the one extrapolated term is GEOM at width 3")
print("=" * 76)
print("  GEOM(3) is assumed linear (2 x GEOM(2)).  Sweeping it, and ROWACT:")
for g in (0.0, 1.0, GEOM, 3.0, 4.0):
    print(f"    GEOM   {g:5.3f} -> width-2 {width2_iter(geom=g):7.3f} ms/iter,"
          f" break-even a2 = {breakeven_a2(W1_MSPT, geom=g):+.3f}")
for r in (1.5, 2.5, ROWACT, 4.0, 5.0):
    print(f"    ROWACT {r:5.3f} -> width-2 {width2_iter(rowact=r):7.3f} ms/iter,"
          f" break-even a2 = {breakeven_a2(W1_MSPT, rowact=r):+.3f}")
print(f"    CHAIN -> 0 (a FREE draft layer) : break-even a2 ="
      f" {breakeven_a2(W1_MSPT, chain=0.0):+.3f}")
print(f"    GEOM and CHAIN both -> 0        : break-even a2 ="
      f" {breakeven_a2(W1_MSPT, geom=0.0, chain=0.0):+.3f}")

CEIL_MSPT = width2(1.0)[2]
CEIL_PCT = 100.0 * (W1_MSPT - CEIL_MSPT) / W1_MSPT
IDEAL_CHAIN_BE = breakeven_a2(W1_MSPT, chain=0.186)

print("\n" + "=" * 76)
print("VERDICT: NO-GO -- on the CEILING, not on the break-even")
print("=" * 76)
print(f"""  Do NOT argue this from break-even.  Break-even is a2 > {be_w1:.3f} against a1 =
  {A1:.3f}, and the usual chain decay (EAGLE-style a2 ~ 0.70-0.80 for an a1 in
  the mid-0.8s) straddles it.  Width-2 is a coin flip on acceptance, and the
  stake is small in both directions: a2 = 0.80 wins {W1_MSPT - width2(0.80)[2]:.3f} ms/token,
  a2 = 0.70 loses {width2(0.70)[2] - W1_MSPT:.3f}.  Calling that "unreachable" would be inventing a
  result; the true a2 is NOT measured, and measuring it means building the
  thing.

  Argue it from the ceiling instead.  A PERFECT second draft -- a2 = 1.0, every
  second token accepted, which is strictly better than any real drafter --
  gives {CEIL_MSPT:.3f} ms/token against width-1's {W1_MSPT:.3f}.  That is the
  priced ceiling on item 2: {CEIL_PCT:.1f}%, or {W1_MSPT - CEIL_MSPT:.3f} ms/token.
  The board needs {W1_MSPT:.3f} -> 2.0.  Item 2's best case closes {100.0*(W1_MSPT-CEIL_MSPT)/(W1_MSPT-2.0):.1f}% of that gap,
  for a build that needs a two-step autoregressive draft chain, a 3-row verify,
  and acceptance instrumentation.  Spending the run on a {CEIL_PCT:.1f}% ceiling with a
  coin-flip realization is the wrong trade.  ITEM 2 IS CLOSED -- not deferred.

  WHY THE CEILING IS LOW: ROWACT.  {ROWACT:.3f} ms of real MoE expert-weight
  traffic per live row, paid twice, and each draft row brings its own top-8
  expert set.  The overhead terms cannot rescue it either -- driving CHAIN to
  the draft layer's ideal 0.186 only moves break-even {be_w1:.3f} -> {IDEAL_CHAIN_BE:.3f}, and
  GEOM is not sheddable at width 2 for the same reason it is not at width 1:
  all three rows are LIVE, so there is no under-filled batch to reclaim.

  This is the THIRD independent pricing and the first on a correct cost model:
      38a7f75 (row 2.190 + harness 3.721)      a2 > 0.773
      a015f60 (geometry + row-and-chain)       a2 > 0.769
      this    (geometry + chain + rowact)      a2 > {be_w1:.3f}
  Stable because the corrections move cost BETWEEN terms that width-2 pays
  twice either way -- the total, {mi:.3f} ms/iter, barely moves.

  UNMEASURED, stated as such: GEOM(3) is extrapolated linearly from GEOM(2),
  and CHAINONLY is n=1.  Neither matters -- the ceiling is computed at a2 = 1.0
  where acceptance is already maximal, and the GEOM sweep above shows even
  GEOM(3) = 0 leaves the ceiling at {mspt(width2_iter(geom=0.0), 1+A1+A1):.3f} ms/token.""")
