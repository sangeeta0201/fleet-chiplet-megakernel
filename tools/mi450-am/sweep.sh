#!/bin/bash
# Build every mi450 kernel harness, validate under FFM, then measure under AM.
#   ./sweep.sh [outdir]
set -o pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$D/env.sh"
OUT="${1:-$PWD/mi450_sweep}"; mkdir -p "$OUT"/{bin,logs,amruns}
TESTS="test_gemm_wmma test_mx_wmma test_rmsnorm_wave test_topk_softmax test_linear_wmma
       test_kv_cache_update test_rmsnorm_linear_mxfp4_bias test_attention_wmma test_moe_linear_mxfp4"
for t in $TESTS; do
  "$D/build.sh" "$FLEET/tests/mi450/$t.hip" -o "$OUT/bin/$t" > "$OUT/logs/$t.build" 2>&1 \
    && echo "build  $t OK" || { echo "build  $t FAIL"; continue; }
  v=$(timeout 900 "$D/runffm.sh" "$OUT/bin/$t" 2>&1 | grep -oE "OVERALL: (PASS|FAIL)|^PASS.*|^FAIL.*" | tail -1)
  echo "ffm    $t ${v:-<none>}"
  rm -rf "$OUT/amruns/$t"; mkdir -p "$OUT/amruns/$t"
  AM_ENV=am_profile_env.sh AM_RUNDIR="$OUT/amruns/$t" \
    timeout -s KILL 3600 "$D/runam.sh" "$OUT/bin/$t" > "$OUT/logs/$t.am" 2>&1
  echo "am     $t rc=$?"
done
"$D/parse_counters.sh" "$OUT"
