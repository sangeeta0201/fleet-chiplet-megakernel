#!/bin/bash
# Per-rank launch shim: set the task memory policy to MPOL_LOCAL, then exec the
# real command. run_mp8_dp_ep_fused.sh inserts it whenever the kernel's
# automatic NUMA balancing is on (see MPK_NUMA_EXEMPT there).
#
# Why: with /proc/sys/kernel/numa_balancing=1 the scanner periodically rewrites
# this process's PTEs to take hinting faults; the MMU notifiers that fires make
# KFD evict (suspend and later restore) every GPU queue the process owns. On
# the persistent megakernel that surfaced as 1k/1k stalls of tens to hundreds
# of seconds: a worker parked at the s_waitcnt behind a flag load whose value
# was already in memory, released only by the next context switch. Exempting
# the four ranks took one build from stalling in 8 of 8 runs to clean in 2 of
# 2, ISA unchanged. Stalls remained while other GPU jobs on the host started
# up; only kernel.numa_balancing=0 host-wide removed those.
#
# A policy set through set_mempolicy carries no MPOL_F_MOF, so task_numa_work
# skips every VMA of the process (vma_policy_mof). It survives execve and is
# inherited by every thread created afterwards, which is why it is set here
# before python, MPI or HIP start any. Placement is the default's: local node
# first, other nodes when it is full.
exec python3 -c '
import ctypes, os, sys
libc = ctypes.CDLL(None, use_errno=True)
SYS_set_mempolicy, MPOL_LOCAL = 238, 4  # x86_64
if libc.syscall(SYS_set_mempolicy, MPOL_LOCAL, None, 0) != 0:
    sys.stderr.write("[mpol] set_mempolicy failed, errno=%d; NUMA balancing "
                     "still applies to this rank\n" % ctypes.get_errno())
os.execvp(sys.argv[1], sys.argv[1:])
' "$@"
