#!/bin/bash
# run_default.sh [GROUP ...]: the default-mode half (stock SPX+NPS1, flags as deployed) of the GPT-OSS-120B blog
# numbers: the reference the DPX+NPS2 runs are compared with. GPU $BUS must be SPX/NPS1 on the stock driver (the
# factory mode); this script never switches modes. Groups (default: all):
#   phase   per-phase recorder runs and their plain twins, MXFP4 and BF16 LM head (ph_spx_{mx,bf}{_ps,})
#   length  1k prompt + 2K..32K generated (bl_spx_nsN; 1K = ph_spx_mx)
#   lengthbf  the same with the BF16 LM head (bl_spx_nsbfN; 1K = ph_spx_bf)
# GSM8K (the placement reference for the speculation table): gsm8k/run_default.sh.
# The same model in one partition: use a different worktree (LANE) from any DPX+NPS2 run going on at the same time,
# since both build into the worktree's demo/gpt_oss/permanent_output_dir.
. "$(dirname "$0")/common.sh"
SETS=${*:-phase length lengthbf}
MODE_ENV=""
if [ "$DRY" != 1 ]; then
  [ "$(mode)" = SPX/NPS1 ] && stock || { echo "GPU $BUS is $(mode) / not the stock driver: need stock SPX/NPS1"; exit 1; }
  docker start $CONTAINER > /dev/null
  DEVS=$(first_dev) || { echo "no HIP device for bus $BUS"; exit 1; }
  st "start: lane $LANE $(git -C $LT log -1 --format=%h), GPU $BUS $(mode), HIP device $DEVS, container $CONTAINER"
else
  DEVS=0
fi
for g in $SETS; do
  case $g in
    phase) run ph_spx_mx_ps ps 1024; run ph_spx_bf_ps psbf 1024; run ph_spx_mx ns 1024; run ph_spx_bf nsbf 1024 ;;
    length) for n in 2048 4096 8192 16384 32768; do run bl_spx_ns$n ns $n; done ;;
    lengthbf) for n in 2048 4096 8192 16384 32768; do run bl_spx_nsbf$n nsbf $n; done ;;
    *) echo "unknown group $g (phase length lengthbf)"; exit 1 ;;
  esac
done
[ "$DRY" = 1 ] || st "done (run_default.sh $SETS)"

