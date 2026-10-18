#!/bin/bash
# run_dpx.sh [GROUP ...]: the DPX+NPS2 half of the GPT-OSS-120B blog numbers (the megakernel outside vLLM).
# GPU $BUS must already be in DPX+NPS2 on the stock driver (scripts/nps2/dpx/to_dpx_nps2_stock.sh); this script never
# switches modes. Groups (default: all):
#   phase   per-phase recorder runs and their plain twins, MXFP4 and BF16 LM head (ph_dpx_{mx,bf}{_ps,})
#   length  1k prompt + 2K..32K generated single-stream (bl_dpx_nsN) and EAGLE3 with 1 draft, 1K..32K (bl_dpx_spN)
#   swap    every read-only weight stored on the other die (MPK_W_SWAP=1), single-stream and EAGLE3, and the
#           single-stream run at EAGLE3's attention partition (15 KV chunks), whose output EAGLE3 must equal
#   lengthbf, swapbf  length and swap with the BF16 LM head (bl_dpx_nsbfN, bl_dpx_spbfN, ..._swap, ..._ck15;
#           1K single stream = ph_dpx_bf)
#   ckbf    BF16 single stream at 15 KV chunks, 32K generated (bl_dpx_nsbf32768_ck15). Every bl_dpx_spbfN output is
#           a prefix of bl_dpx_spbf32768's, so this one run checks EAGLE3's output at every length.
# GSM8K and the 1K conversation (speculation, 1 and 3 drafts): gsm8k/run_dpx.sh.
# See common.sh for the environment (CONTAINER, PATH_PREPEND, BUS, WAIT_IDLE, DRY, ...).
. "$(dirname "$0")/common.sh"
SETS=${*:-phase length swap lengthbf swapbf ckbf}
MODE_ENV=$MULTI
if [ "$DRY" != 1 ]; then
  [ "$(mode)" = DPX/NPS2 ] && stock || { echo "GPU $BUS is $(mode) / not the stock driver: need stock DPX/NPS2"; exit 1; }
  docker start $CONTAINER > /dev/null
  f=$(first_dev) || { echo "no HIP device for bus $BUS"; exit 1; }
  DEVS=$f,$((f + 1))
  st "start: lane $LANE $(git -C $LT log -1 --format=%h), GPU $BUS $(mode), HIP devices $DEVS, container $CONTAINER"
else
  DEVS=0,1
fi
for g in $SETS; do
  case $g in
    phase)
      run ph_dpx_mx_ps ps 1024; run ph_dpx_bf_ps psbf 1024; run ph_dpx_mx ns 1024; run ph_dpx_bf nsbf 1024 ;;
    length)
      for n in 2048 4096 8192 16384 32768; do run bl_dpx_ns$n ns $n; done
      for n in 1024 2048 4096 8192 16384 32768; do run bl_dpx_sp$n sp $n; done ;;
    swap)
      run bl_dpx_ns1024_swap ns 1024 MPK_W_SWAP=1
      run bl_dpx_sp1024_swap sp 1024 MPK_W_SWAP=1
      run bl_dpx_ns1024_ck15 ns 1024 CK_FMHA_NUM_KV_CHUNKS=15 ;;
    lengthbf)
      for n in 2048 4096 8192 16384 32768; do run bl_dpx_nsbf$n nsbf $n; done
      for n in 1024 2048 4096 8192 16384 32768; do run bl_dpx_spbf$n spbf $n; done ;;
    swapbf)
      run bl_dpx_nsbf1024_swap nsbf 1024 MPK_W_SWAP=1
      run bl_dpx_spbf1024_swap spbf 1024 MPK_W_SWAP=1
      run bl_dpx_nsbf1024_ck15 nsbf 1024 CK_FMHA_NUM_KV_CHUNKS=15 ;;
    ckbf)
      run bl_dpx_nsbf32768_ck15 nsbf 32768 CK_FMHA_NUM_KV_CHUNKS=15 ;;
    *) echo "unknown group $g (phase length swap lengthbf swapbf ckbf)"; exit 1 ;;
  esac
done
[ "$DRY" = 1 ] || st "done (run_dpx.sh $SETS)"

