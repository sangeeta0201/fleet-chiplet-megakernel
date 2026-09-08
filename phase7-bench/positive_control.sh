#!/bin/bash
# The GB/s rows read 0.00 while the % counters move. Two possible causes:
#   (a) the GB/s Formula lines inherit wrong CPU clock constants, or
#   (b) Phase 7 genuinely pushes ~nothing across the inter-die fabric.
# Distinguish them with a POSITIVE CONTROL: a workload guaranteed to saturate the
# fabric. If GB/s stays 0.00 there too, the formula is broken. If it rises, the
# instrument is calibrated and Phase 7's ~0 is a real result.
set -u
G=$HOME/nps1/gpudf_ini

echo "=== current partition mode (matters: NPS1 interleaves across AIDs) ==="
rocm-smi --showcomputepartition --showmemorypartition 2>/dev/null |
	grep -iE "SPX|DPX|NPS|partition" | head -8 | sed 's/^/  /'

echo
echo "=== available traffic generators ==="
ls -la "$HOME/aid-local-hbm/tools/" 2>/dev/null | grep -vE '\.cpp|\.h$|^total' |
	head -12 | sed 's/^/  /'
for b in "$HOME/aid-local-hbm/tools"/*; do
	[ -x "$b" ] && [ -f "$b" ] && echo "  exec: $b"
done

echo
echo "=== build a guaranteed fabric hammer: streaming copy across the whole GPU ==="
cat > /tmp/hammer.cpp <<'EOF'
#include <hip/hip_runtime.h>
#include <cstdio>
__global__ void stream(float4* __restrict__ d, const float4* __restrict__ s, size_t n){
    size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x, st = (size_t)gridDim.x*blockDim.x;
    for (; i < n; i += st) d[i] = s[i];
}
int main(int argc, char** argv){
    size_t MB = argc > 1 ? atol(argv[1]) : 8192;      // 8 GB default: far past any cache
    double secs = argc > 2 ? atof(argv[2]) : 20.0;
    size_t bytes = MB << 20, n = bytes / sizeof(float4);
    float4 *s, *d;
    if (hipMalloc(&s, bytes) || hipMalloc(&d, bytes)) { printf("alloc failed\n"); return 1; }
    hipMemset(s, 1, bytes); hipDeviceSynchronize();
    printf("hammer: %zu MB x2, streaming for %.0f s\n", MB, secs); fflush(stdout);
    double t0 = 0; hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    size_t iters = 0; hipEventRecord(e0);
    for (;;) {
        stream<<<8192, 256>>>(d, s, n); ++iters;
        if ((iters & 7) == 0) {
            hipDeviceSynchronize(); hipEventRecord(e1); hipEventSynchronize(e1);
            float ms = 0; hipEventElapsedTime(&ms, e0, e1); t0 = ms / 1000.0;
            if (t0 > secs) break;
        }
    }
    hipDeviceSynchronize();
    printf("hammer: %zu iters, %.1f s, ~%.0f GB/s\n", iters, t0,
           (double)iters * bytes * 2.0 / t0 / 1e9);
    return 0;
}
EOF
hipcc -O3 --offload-arch=gfx950 /tmp/hammer.cpp -o /tmp/hammer 2>&1 | tail -5 | sed 's/^/  /'
[ -x /tmp/hammer ] && echo "  built /tmp/hammer" || { echo "  BUILD FAILED"; exit 1; }

echo
echo "=== C: 8 s window while the hammer streams 16 GB of HBM ==="
OUT=/tmp/gd_hammer; rm -rf "$OUT"; mkdir -p "$OUT"
( /tmp/hammer 8192 25 > /tmp/hammer.log 2>&1 ) & HP=$!
sleep 3
sudo -n /usr/bin/MultEvent -O "$OUT" -N hammer -t 8 \
	gpu gpu-df -i "$G/gpudf_cake_top.ini" -s 0-3 >/dev/null 2>&1
sudo -n /usr/bin/MultEvent CLEAR >/dev/null 2>&1
wait $HP 2>/dev/null
cat /tmp/hammer.log | sed 's/^/  /'

echo
echo "=== do the GB/s rows move under a real fabric load? ==="
for f in /tmp/gd_idle/csv/GPUDF.csv /tmp/gd_busy/csv/GPUDF.csv /tmp/gd_hammer/csv/GPUDF.csv; do
	[ -f "$f" ] || continue
	printf '  --- %s ---\n' "$(basename "$(dirname "$(dirname "$f")")")"
	grep -iE "GB/s" "$f" | grep -iE "cake" | head -12 |
		awk -F',' '{n=$1; s=0; for(i=2;i<=NF;i++) s+=$i+0; printf "    %-52s sum=%8.2f  avg=%7.2f\n", substr(n,1,52), s, s/(NF-1)}'
done

