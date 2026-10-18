#!/bin/bash
# seq_ev3.sh (node 2): the default-mode megakernel evals one session at a time, every one with MPK_HOST_STOP=1.
# One at a time: ending (or starting) a GPU process on this node hung the other session's megakernel launch that was
# in flight, twice on Oct 18 (mkd_bf_ns launch 654 at 07:09, mkd_mx_hs launch 441 at 07:16; all workers resident).
# Chat prompts carry the date (vLLM renders "Current date"): pinned per session to the date of the runs they pair with.
# A session counts as running while any of its processes but `sleep` lives: the run script's inner watchdog
# (`sleep $TIMEOUT`) carries the session's environment and outlives it (seq_ev2.sh waited on it).
# Re-entrant: an eval with a summary.json is done; a session that still runs is waited for.
# A Slurm job end kills this script and stops the containers: rerun it in the next job (seq_guard.sh does).
set -u
S=/home/user/mk1/eval/spec
R=/home/user/mk1/results/mk_eval
ST=$S/status_seq.txt
log() { echo "[SEQ] $(date -u +%T) $*" | tee -a $ST; }
spids() {  # spids LABEL [sleep]: the session's processes except sleep, or (with "sleep") only its sleeps
  docker exec fleet_ship2 bash -c "
    for d in /proc/[0-9]*; do
      c=\$(cat \$d/comm 2>/dev/null)
      if [ '$2' = sleep ]; then [ \"\$c\" = sleep ] || continue; else [ \"\$c\" = sleep ] && continue; fi
      grep -qz '^SESS_EVAL_OUT=/root/user/mk1/results/mk_eval/$1\$' \$d/environ 2>/dev/null && echo \${d#/proc/}
    done" | tr '\n' ' '
}
cpids() { spids "$1" run; }
finished() { curl -s -m 5 localhost:$1/metrics | grep -E '^vllm:request_success_total' | awk '{s+=$NF} END {print s+0}'; }
summ() { python3 -c "import json; s=json.load(open('$1')); print('accuracy', s.get('accuracy'), 'n', s.get('n') or s.get('num_questions'))" 2>/dev/null || echo "no summary"; }
exec 9> $S/seq_ev.lock
flock -n 9 || { echo "another sequencer holds $S/seq_ev.lock: exiting"; exit 0; }
log "start (pid $$, seq_ev3)"
for it in mkd_mx_s:fair3:15:mx:mmlupro,gsm8k5:- mkd_bf_s:fair2:05:bf:gsm8k5,mmlupro:VLLM_SYSTEM_START_DATE=2026-10-18 mkd_bf_c:fair2:05:bf:chat:VLLM_SYSTEM_START_DATE=2026-10-17; do
  IFS=: read lab lane bus head evals extra <<< "$it"; [ "$extra" = - ] && extra=""
  miss=""; for e in ${evals//,/ }; do [ -s $R/$lab/$e/summary.json ] || miss="$miss,$e"; done; miss=${miss#,}
  if [ -z "$miss" ]; then
    log "$lab: all done"
  else
    if [ -z "$(cpids $lab | tr -d ' ')" ]; then
      log "$lab: start lane $lane bus $bus head $head evals $miss extra [$extra]"
      setsid bash $S/mk_eval_def.sh $lane $bus $lab $head $miss MPK_HOST_STOP=1 $extra > $S/$lab.out 2>&1 < /dev/null &
      sleep 60
    else
      log "$lab: running, waiting"
    fi
    port=$((8400 + 16#$bus % 64)); last=-1; still=0
    while [ -n "$(cpids $lab | tr -d ' ')" ]; do
      n=$(finished $port)
      if [ "$n" = "$last" ] && [ "$n" -gt 0 ]; then still=$((still + 30)); else still=0; fi
      [ $still = 120 ] && log "$lab: no request finished in 120 s (at $n): HANG? AGENT_LOOP_WAKE_SEQ"
      last=$n; sleep 30
    done
    for e in ${evals//,/ }; do log "$lab $e: $(summ $R/$lab/$e/summary.json)"; done
  fi
  w=$(spids $lab sleep); [ -n "${w// /}" ] && { docker exec fleet_ship2 kill $w 2>/dev/null; log "$lab: leftover watchdog sleep [$w] ended"; }
done
log "all sessions ended; reinstalling libaitermk-vllm"
docker exec fleet_ship2 pip install --no-deps -e /root/user/mk1/libAiterMK/libAiterMK/engine_adapters/vllm > $S/reinstall_seq.out 2>&1
log "reinstall rc=$? AGENT_LOOP_WAKE_SEQ_DONE"
