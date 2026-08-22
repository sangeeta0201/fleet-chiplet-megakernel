#!/bin/bash
# ITEM 1.1 (closing) -- what does a second decode row ACTUALLY cost at the wall?
#
# RE-STEER v5 states bs=1 10.609 / bs=2 16.567 -> C = 1.562. That bs=2 figure
# came from the MTP path, which carries a draft layer plus accept/reject
# bookkeeping ON TOP of the second row. `--max-num-batched-tokens 2` is a clean
# 2-decode-row build -- persistent_kernel.py does `batch_size =
# self.max_num_batched_tokens` directly, so no draft layer and no accept/reject
# enter the task graph. The instrumented pair (MPK_BAR_SKEW=3) put the pure row
# at +2.06 ms/iter, not +5.96, so C is being measured here without the
# instrument to settle it.
#
# ONE VARIABLE: --max-num-batched-tokens.
#
# n=3 per arm. The wall noise floor is 0.26 ms and the effect under test is
# ~2 ms, but C is load-bearing for item 2's entire go/no-go so it gets a pool,
# not a single shot.
#
# GATED FOR OUTPUT, and this one is a real gate rather than a formality: bs=2's
# row 0 decodes the same prompt as bs=1, so the emitted token stream MUST be
# identical. If it is not, the bs=2 build is wrong and its wall is meaningless.
# Token ids are diffed directly -- compare_tokens.py's PASS is cross-rank
# identity, which passes at agree@0/264 and would not catch this.
#
# NO instrument: MPK_BAR_SKEW off, MPK_SUBPHASE_TIMING off. Subphase cost
# scales with tile count and tile count is the variable.
# ===================== RESULT, 2026-08-22 =====================
# bs1 n=3: 10.562 / 10.619 / 10.685  -> median 10.619 ms/iter (spread 0.12,
#          inside the 0.26 ms noise floor)
# bs2 n=2: 12.711 / 12.907           -> median 12.809   (one rep lost to rc=124,
#          the known ~1-in-3 NP=8 failure)
#
#   second row costs +2.190 ms, C = 1.206   -- NOT the board's 1.562
#
# and the stage-stamp probe independently predicted +2.061 ms from region
# counters alone, so the two instruments agree to ~130 us.
#
# *** RESOLVED 369032c: C = 1.206 IS CONFIRMED. The exact-match gate below was
# *** ILLEGAL. bs=1 is not bit-reproducible -- it has two attractor
# *** continuations and splits 2/2 across them -- so "bs=2 diverges from bs=1 at
# *** token 0" is what two bs=1 runs do to each other half the time. Both bs=2
# *** runs land inside a control cluster (prefix 145 and 124) and pass the hard
# *** gate. Use demo/glm5/correctness_gate.py, not the exact-prefix check here.
# *** The original (wrong) reasoning is kept below as the record.
#
#   bs1 r2, r3 vs r1:  exact-prefix 39/264   (r2 and r3 agree with each other)
#   bs2 r1, r3 vs r1:  exact-prefix  0/264   (r1 and r3 agree with each other)
#
# Two separate problems:
#
# 1. bs=1 IS NOT RUN-TO-RUN REPRODUCIBLE. r2 and r3 are identical to each other
#    and diverge from r1 at token 39. Any exact-match gate on this model needs
#    that characterised first, or it cannot distinguish a real defect from
#    baseline nondeterminism.
#
# 2. bs=2 DIVERGES AT TOKEN 0, and its wrong prefix is
#      [16, 13, 220, 3070, 2082, 55481, ...]
#    which is BYTE-IDENTICAL to the wrong prefix the MTP arm produced in item 0
#    (ccf8cfa recorded mtp[0:6] = [16,13,220,3070,2082,55481]). The MTP arm ran
#    with --max-num-batched-tokens 2, so it INHERITED this. The item 0
#    divergence is therefore a BATCH_SIZE=2 build defect, not the accept/reject
#    step committing draft tokens it should have rejected. The retraction of
#    8.952 ms/token stands -- the number is still not comparable -- but its
#    cause was misattributed.
#
# WHY THIS CONTAMINATES ITEM 1.1: wrong activations upstream of the router
# change the TopK, hence the expert routing, hence EP balance. That is exactly
# the line item 1.1 leans on (the EP collective is 36% of the row cost and the
# conclusion was shared-expert skew). So both C and the EP decomposition are
# provisional until bs=2 emits correct tokens. Fix the bs=2 path, then re-run
# BOTH probes before quoting either number.
# ==============================================================
set -u
ulimit -c 0
cd /home/claudeuser/fleet-chiplet-megakernel/demo/glm5
D=/tmp/c_rowcost
mkdir -p $D

export MODEL_PATH=/home/claudeuser/models/glm5-mxfp4
export MAX_SEQ_LENGTH=512
export MAX_NEW_TOKENS=256
export MAX_SAVE_TOKENS=264
export MPK_BAR_SKEW=0
export MPK_SUBPHASE_TIMING=0
export MPK_PRINT_ALL_RANKS=0
PROMPT="Write a short paragraph explaining why the sky appears blue."

for ARM in bs1 bs2; do
  if [ "$ARM" = bs1 ]; then export MPK_EXTRA_ARGS=""
  else                      export MPK_EXTRA_ARGS="--max-num-batched-tokens 2"; fi
  # Tile geometry changes with BATCH_SIZE -> each ARM needs its own build.
  # Within an arm the build is reused (KEEP_BUILD=1) so the three reps differ
  # only by run-to-run variance, which is the thing being pooled.
  rm -rf permanent_output_dir permanent_output_dir_rank*
  export KEEP_BUILD=0
  for REP in 1 2 3; do
    export MASTER_PORT=$((34100 + (RANDOM % 300)))
    echo "===== ARM $ARM REP $REP EXTRA='${MPK_EXTRA_ARGS}' ====="
    timeout 900 ./run_mp8_dp_ep_fused.sh \
        --prompt "$PROMPT" --save-tokens "$D/${ARM}_r${REP}.json" \
        > "$D/${ARM}_r${REP}.log" 2>&1
    rc=$?
    echo "  rc=$rc"
    if [ "$rc" -eq 124 ]; then
      # Split literal: an unsplit pattern matches this tool's own command line.
      pkill -9 -f "demo""\.py" 2>/dev/null; sleep 5
      pkill -9 -f "mpirun -np" 2>/dev/null; sleep 3
    fi
    echo -n "  BATCHED_TOKENS: "
    grep -ho "MPK_MAX_NUM_BATCHED_TOKENS=[0-9]*" "$D/${ARM}_r${REP}.log" | sort -u | tr '\n' ' '
    echo
    grep -hoE "[0-9.]+ ms/iter" "$D/${ARM}_r${REP}.log" | tail -2
    export KEEP_BUILD=1
  done
done

echo "############ C = bs2 / bs1 ############"
python3 - <<'PY'
import re, json, glob, statistics
D = "/tmp/c_rowcost"
walls = {}
for arm in ("bs1", "bs2"):
    xs = []
    for rep in (1, 2, 3):
        try:
            txt = open(f"{D}/{arm}_r{rep}.log", errors="replace").read()
        except OSError:
            continue
        m = re.findall(r"([0-9.]+)\s*ms/iter", txt)
        if m:
            xs.append(float(m[-1]))
    walls[arm] = xs
    print(f"{arm}: n={len(xs)} {xs}")
if walls["bs1"] and walls["bs2"]:
    a = statistics.median(walls["bs1"]); b = statistics.median(walls["bs2"])
    print(f"\nbs1 median {a:.3f} ms/iter")
    print(f"bs2 median {b:.3f} ms/iter")
    print(f"second row costs {b-a:+.3f} ms   C = {b/a:.3f}")
    print(f"board claimed C = 1.562 (10.609 -> 16.567, MTP path)")

print("\n#### OUTPUT GATE: bs=2 row 0 must equal bs=1 token for token ####")
def toks(p):
    f = sorted(glob.glob(p))
    return json.load(open(f[0]))["token_ids"] if f else None
ref = None
for arm in ("bs1", "bs2"):
    for rep in (1, 2, 3):
        t = toks(f"{D}/{arm}_r{rep}_rank*.json")
        if t is None:
            print(f"{arm} r{rep}: no dump"); continue
        if ref is None:
            ref = t; print(f"{arm} r{rep}: REFERENCE, {len(t)} tokens"); continue
        n = min(len(ref), len(t)); k = 0
        while k < n and ref[k] == t[k]: k += 1
        print(f"{arm} r{rep}: {len(t)} tokens, exact-prefix {k}/{n} "
              f"{'OK' if k == n and n > 0 else '*** DIVERGES ***'}")
        if k != n:
            print(f"    ref[{k}:{k+8}] = {ref[k:k+8]}")
            print(f"    got[{k}:{k+8}] = {t[k:k+8]}")
PY
echo CROWCOST_DONE
