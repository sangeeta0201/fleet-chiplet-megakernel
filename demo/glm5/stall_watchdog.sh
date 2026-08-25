#!/bin/bash
# run_with_watchdog <logfile> <stall_seconds> <cmd...>
#
# Runs <cmd> with stdout+stderr to <logfile> and kills the whole process group
# if the log stops growing for <stall_seconds>. Returns the command's exit
# status, or 124 on a stall kill (the same code `timeout` uses).
#
# WHY THIS EXISTS. This box wedges at cold start in `launch_persistent_kernel
# ENTER` -- every rank prints ENTER, all four GPUs pin at 100%, and nothing
# ever advances. It is config-independent and pre-existing (task #47 fixed one
# NP=8 presentation of it; this one survives at NP=4), and it hits roughly a
# third of runs. Measured occurrences on 2026-08-25 alone: dkm2 r2, the dkm2
# correctness prompt 0, and two more the day before.
#
# A wall-clock `timeout` cannot separate the two cases, because a healthy run
# is ~110 s when it reuses the build and ~400 s when it rebuilds, while a hung
# run is infinite. Any single deadline either kills slow-but-healthy rebuilds
# or waits 40 minutes on a wedge. LOG GROWTH separates them exactly: a healthy
# run writes continuously (per-iteration lines, rank chatter), and a wedged one
# writes nothing after the last ENTER.
#
# Kill by process GROUP, not PID: run_mp8_dp_ep_fused.sh spawns mpirun which
# spawns four demo.py ranks, and killing the script alone orphans the ranks --
# they keep the GPUs at 100% and poison the next run. setsid gives the child
# its own group so a single negated-PID kill reaps the whole tree. See memory
# orphan-ranks-survive-a-killed-mpirun.
set -u

run_with_watchdog() {
  local log="$1" stall="$2"; shift 2

  : > "$log"
  setsid "$@" > "$log" 2>&1 &
  local child=$!
  # setsid makes the child its own group leader, so -child is the whole tree.
  local pgid=$child

  # ARM ONLY AFTER THE KERNEL LAUNCHES. A rebuild is legitimately silent for
  # minutes at a time (one hipcc invocation on the megakernel), so a watchdog
  # armed from t=0 would have to carry a threshold long enough to cover the
  # compiler -- which is long enough to make it useless against the wedge. The
  # wedge only ever happens after every rank prints ENTER, so waiting for that
  # string separates "quiet because compiling" from "quiet because hung" and
  # lets the threshold be tight.
  local last_size=0 quiet=0 armed=0
  while kill -0 "$child" 2>/dev/null; do
    sleep 10
    if [ "$armed" -eq 0 ]; then
      grep -q "launch_persistent_kernel ENTER" "$log" 2>/dev/null && armed=1
      continue
    fi
    local size
    size=$(stat -c %s "$log" 2>/dev/null || echo 0)
    if [ "$size" -gt "$last_size" ]; then
      last_size=$size
      quiet=0
    else
      quiet=$(( quiet + 10 ))
      if [ "$quiet" -ge "$stall" ]; then
        echo "[watchdog] no log growth for ${stall}s -- killing pgid $pgid" \
             >> "$log"
        kill -9 -"$pgid" 2>/dev/null
        wait "$child" 2>/dev/null
        # Belt and braces: reap any rank that escaped the group.
        pkill -9 -f "[d]emo.py" 2>/dev/null
        sleep 10
        return 124
      fi
    fi
  done
  wait "$child"
  return $?
}
