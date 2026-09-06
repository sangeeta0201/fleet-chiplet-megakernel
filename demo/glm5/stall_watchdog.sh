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

# Reap a killed run completely, then WAIT for the GPUs to actually come back.
#
# The old inline cleanup killed the group and pkill'd demo.py, but not the
# mpirun that spawned it -- and mpirun respawns/holds on. A surviving mpirun
# keeps ~110 GB per device mapped, so the NEXT run dies in hipMalloc with
# "0 bytes free" and looks like a memory bug in whatever kernel is under test.
# Two runs were lost to exactly that. Poll VRAM rather than sleeping a fixed
# 10 s: teardown of a 744B model is not instant and is not a constant.
_watchdog_reap() {
  local pgid="$1" child="$2"
  kill -9 -"$pgid" 2>/dev/null
  wait "$child" 2>/dev/null
  for _ in 1 2 3; do
    pkill -9 -f "[d]emo.py" 2>/dev/null
    pkill -9 -f "[m]pirun -np" 2>/dev/null
    sleep 2
  done
  # Up to 60 s for the driver to release; give up rather than hang forever.
  for _ in $(seq 1 12); do
    if ! pgrep -f "[d]emo.py" >/dev/null 2>&1; then
      local busy
      busy=$(rocm-smi --showmemuse 2>/dev/null |
             grep -c "VRAM%): [1-9]" || true)
      [ "${busy:-0}" -eq 0 ] && break
    fi
    sleep 5
  done
}

run_with_watchdog() {
  local log="$1" stall="$2"; shift 2

  # A wedged rank that gets SIGKILLed dumps a multi-GB gpucore next to the
  # script. The watchdog exists to kill runs, so it is the one place that
  # knows a kill is coming: five of them filled this box's 3.5 TB root to
  # zero bytes free, which then fails every later run in ways that look like
  # kernel bugs (hipMalloc OOM, empty dumps, "no space left for
  # here-document"). Callers that set their own ulimit are unaffected.
  ulimit -c 0 2>/dev/null || true

  : > "$log"
  setsid "$@" > "$log" 2>&1 &
  local child=$!
  # setsid makes the child its own group leader, so -child is the whole tree.
  local pgid=$child

  # Absolute cap, independent of log growth. LOG GROWTH alone is not a
  # sufficient liveness signal: a rank that crash-loops (an illegal access
  # under a bad config, say) writes stack traces and NCCL teardown warnings
  # forever, so `size` keeps rising and the stall test never trips. One
  # observed instance grew a 9 MB log and ran 25 minutes before it was killed
  # by hand. Ten stall periods is far above any healthy run, including a
  # cold rebuild, and still bounds the worst case.
  local hard_cap=$(( stall * 10 ))
  local elapsed=0

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
    elapsed=$(( elapsed + 10 ))
    if [ "$elapsed" -ge "$hard_cap" ]; then
      echo "[watchdog] hard cap ${hard_cap}s reached -- killing pgid $pgid" \
           >> "$log"
      _watchdog_reap "$pgid" "$child"
      return 124
    fi
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
        _watchdog_reap "$pgid" "$child"
        return 124
      fi
    fi
  done
  wait "$child"
  return $?
}
