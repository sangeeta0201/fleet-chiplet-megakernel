#!/bin/bash
# Turn AM's per-dispatch counter dumps into one table.
#   ./parse_counters.sh <sweep outdir>
OUT="${1:?usage: parse_counters.sh <sweep outdir>}"
# Derive the modeled clock from a dispatch timestamp pair rather than assuming.
period=$(grep -m1 -oE '@([0-9]+)ps.*clk =([0-9]+)' "$OUT"/logs/*.am 2>/dev/null \
  | head -1 | sed -E 's/.*@([0-9]+)ps.*clk =([0-9]+)/\1 \2/' | awk '{printf "%.3f", $1/$2}')
[ -z "$period" ] && period=555.000
echo "modeled shader clock ${period} ps/cycle ($(awk -v p=$period 'BEGIN{printf "%.3f",1000/p}') GHz)"
printf "%-32s %10s %9s %7s %8s %7s\n" KERNEL CYCLES TIME_us WAVES INSTRS WMMA
for d in "$OUT"/amruns/*/; do
  t=$(basename "$d"); best=""; bestc=-1
  for f in "$d"/perf*_counters_absolute.txt; do
    [ -f "$f" ] || continue; case "$f" in *perf_counters_absolute.txt) continue;; esac
    c=$(awk '/shader_execution_cycles/{print $2}' "$f"); [ -z "$c" ] && c=0
    [ "$c" -gt "$bestc" ] && { bestc=$c; best=$f; }
  done
  [ -z "$best" ] && { printf "%-32s %10s\n" "$t" "no counters (aborted?)"; continue; }
  awk -v k="$t" -v per="$period" '
    /shader_execution_cycles/{c=$2} /^TOTAL_WAVES/{w=$2}
    /^TOTAL_INSTRUCTIONS/{i=$2} /^WMMA_OPS_MAX/{m=$2}
    END{printf "%-32s %10d %9.3f %7d %8d %7d\n", k, c, c*per/1e6, w, i, m}' "$best"
done
