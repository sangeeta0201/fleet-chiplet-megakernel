#!/usr/bin/env python3
"""Run a bash script as root: sudo -n python3 run_root.py <script> [args...]

sudo grants NOPASSWD on python3 but not on bash, and PowerShell mangles quotes
in inline ssh commands, so this exists to launch scripts without either problem.
"""
import subprocess
import sys

if len(sys.argv) < 2:
    sys.exit("usage: run_root.py <script.sh> [args...]")

sys.exit(subprocess.run(["bash"] + sys.argv[1:]).returncode)
