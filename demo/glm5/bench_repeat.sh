#!/bin/bash
# Repeat the NP=8 GLM-5 decode benchmark N times and print each run's
# avg ms/iter. The wall noise floor is 0.26 ms, so n=1 cannot resolve a
# sub-0.2 ms lever (see memory: glm5-wall-noise-floor-is-0.26ms).
#
#   ./bench_repeat.sh <tag> [n] [-- extra demo.py args]
#
# The first run rebuilds; the rest reuse the build (KEEP_BUILD=1), so the
# megakernel under test is identical across the repeats.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

TAG="${1:?usage: bench_repeat.sh <tag> [n]}"
N="${2:-3}"
shift 2 2>/dev/null || shift 1
[ "${1:-}" = "--" ] && shift

export MODEL_PATH="${MODEL_PATH:-/home/claudeuser/models/glm5-mxfp4}"
ulimit -c 0

for i in $(seq 1 "$N"); do
  log="/tmp/glm5_${TAG}_r${i}.log"
  # Distinct port per run: a killed run leaves the listener in TIME_WAIT.
  export MASTER_PORT=$(( 30100 + (RANDOM % 400) ))
  if [ "$i" -gt 1 ]; then export KEEP_BUILD=1; fi
  timeout "${RUN_TIMEOUT:-2400}" ./run_mp8_dp_ep_fused.sh "$@" > "$log" 2>&1
  rc=$?
  # The Decode line only -- the Prefill line has the same "avg N ms/iter"
  # shape and is not what any lever here is measured against.
  ms=$(grep -E '^\[1,0\].*Decode:' "$log" | grep -oE 'avg [0-9.]+ms/iter' \
        | grep -oE '[0-9.]+')
  echo "[$TAG] run $i: rc=$rc  avg=${ms:-FAILED} ms/iter   ($log)"
done

echo "--- $TAG summary ---"
grep -hE '^\[1,0\].*Decode:' /tmp/glm5_${TAG}_r*.log \
  | grep -oE 'avg [0-9.]+ms/iter' | grep -oE '[0-9.]+' \
  | awk '{s+=$1; n++; if(n==1||$1<mn)mn=$1; if($1>mx)mx=$1}
         END{if(n)printf "n=%d mean=%.3f min=%.3f max=%.3f ms/iter\n",n,s/n,mn,mx;
             else print "no results"}'
