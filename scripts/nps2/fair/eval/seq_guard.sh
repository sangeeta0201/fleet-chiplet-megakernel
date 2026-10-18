# seq_guard.sh (node 2, called every minute from the workstation): keep seq_ev.sh alive across Slurm job handovers
# (a job end kills host processes; the eval sessions live in the container) and print its new status lines.
S=/home/user/mk1/eval/spec
if grep -q 'AGENT_LOOP_WAKE_SEQ_DONE' $S/status_seq.txt; then echo SEQ_FINISHED; tail -14 $S/status_seq.txt; exit 0; fi
if ! pgrep -f 'bash /home/user/mk1/eval/spec/seq_ev3[.]sh' > /dev/null; then
  echo "GUARD $(date -u +%T): seq_ev3.sh not running: restarting it"
  setsid nohup bash $S/seq_ev3.sh >> $S/seq_ev3.out 2>&1 < /dev/null &
  sleep 3
fi
n=$(wc -l < $S/status_seq.txt); o=$(cat $S/.guard_n 2>/dev/null || echo 0)
[ "$n" -gt "$o" ] && tail -n $((n - o)) $S/status_seq.txt
echo $n > $S/.guard_n
true
