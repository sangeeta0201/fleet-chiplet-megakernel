#!/bin/bash
# IS bs=1 BIT-REPRODUCIBLE? This decides whether exact-token equality is a
# legal correctness gate on this model at all.
#
# The prior sweep saw bs1 r2 == r3 exactly, with r1 diverging from both at
# token 39. r1 was the rep that also performed the BUILD (KEEP_BUILD=0); r2 and
# r3 reused it (KEEP_BUILD=1). So the divergence correlates with "first run
# after a build", not obviously with run-to-run variance. n=2 agreement is too
# weak to conclude either way.
#
# Design: build once, discard that rep's dump, then FIVE identical reps that all
# reuse the same build. Nothing varies between them -- same binary, same prompt,
# same greedy decode, no seed to pin because argmax is deterministic by
# construction. Any disagreement among r1..r5 is genuine nondeterminism.
# A separate rep (rB) keeps the build run's dump so the
# first-run-after-build hypothesis can be checked too.
set -u
ulimit -c 0
cd /home/claudeuser/fleet-chiplet-megakernel/demo/glm5
D=/tmp/det_bs1
mkdir -p $D

export MODEL_PATH=/home/claudeuser/models/glm5-mxfp4
export MAX_SEQ_LENGTH=512
export MAX_NEW_TOKENS=256
export MAX_SAVE_TOKENS=264
export MPK_BAR_SKEW=0
export MPK_SUBPHASE_TIMING=0
export MPK_EXTRA_ARGS=""
PROMPT="Write a short paragraph explaining why the sky appears blue."

rm -rf permanent_output_dir permanent_output_dir_rank*
export KEEP_BUILD=0
for REP in B 1 2 3 4 5; do
  export MASTER_PORT=$((35100 + (RANDOM % 300)))
  echo "===== REP $REP (KEEP_BUILD=$KEEP_BUILD) ====="
  timeout 900 ./run_mp8_dp_ep_fused.sh \
      --prompt "$PROMPT" --save-tokens "$D/r${REP}.json" \
      > "$D/r${REP}.log" 2>&1
  echo "  rc=$?"
  grep -h "Decode:" "$D/r${REP}.log" | sed 's/.*(avg /  wall avg /;s/)$//'
  export KEEP_BUILD=1
done

echo "############ DETERMINISM VERDICT ############"
python3 - <<'PY'
import json, glob, itertools
D = "/tmp/det_bs1"
reps = {}
for r in ("B", "1", "2", "3", "4", "5"):
    f = sorted(glob.glob(f"{D}/r{r}_rank*.json"))
    if f:
        reps[r] = json.load(open(f[0]))["token_ids"]
    else:
        print(f"r{r}: NO DUMP (lost)")
def pref(a, b):
    n = min(len(a), len(b)); k = 0
    while k < n and a[k] == b[k]: k += 1
    return k, n
print("\npairwise exact-prefix (B = the build run):")
ks = sorted(reps)
print("     " + "".join(f"{r:>8}" for r in ks))
for a in ks:
    row = f"  {a:<3}"
    for b in ks:
        if a == b: row += f"{'-':>8}"
        else:
            k, n = pref(reps[a], reps[b])
            row += f"{(str(k) if k < n else 'EQ'):>8}"
    print(row)

nonbuild = [r for r in ks if r != "B"]
allsame = all(reps[a] == reps[b] for a, b in itertools.combinations(nonbuild, 2))
print(f"\nreps reusing the build ({','.join(nonbuild)}): "
      f"{'ALL IDENTICAL' if allsame else 'DISAGREE'}")
if "B" in reps and nonbuild:
    k, n = pref(reps["B"], reps[nonbuild[0]])
    print(f"build run vs the rest: exact-prefix {k}/{n} "
          f"{'(identical)' if k == n else '(build run is the odd one out)'}")
print("\nVERDICT:", "bs=1 IS bit-reproducible -> exact-token equality is a legal gate"
      if allsame else
      "bs=1 is NOT bit-reproducible -> exact equality is NOT a legal gate")
PY
echo DETBS1_DONE
