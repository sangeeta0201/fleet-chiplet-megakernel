#!/bin/bash
# MPK_WRAP hook for run_1gpu.sh: run $SESS_SCRIPT with demo.py's exact argv and environment instead of
# demo.py. run_1gpu.sh calls "${MPK_WRAP:-} python demo.py <args>".
[ "$2" = "demo.py" ] || { echo "wrap_sess: expected 'python demo.py', got '$1 $2'"; exit 2; }
shift 2
exec python "${SESS_SCRIPT:-/root/user/nps1/vllmn/sess_test.py}" "$@"

