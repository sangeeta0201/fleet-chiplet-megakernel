#!/bin/bash
# exp_run.sh CFG: export one provider config (ns|e3|df8) on lane $LANE (DPX+NPS2) with MPK_PROVIDER_EXPORT on: writes
# the provider source and state to $PROVIDERS/CFG, then decodes the longest GSM8K-128 question standalone until the
# arena is full and saves every token (the long-context exactness reference).
# ARENA = the build's token arena: a request's prompt + max_tokens must stay below it (default 32769 = 32,768 tokens).
# ARENA=466 (the longest question + 264 tokens) rebuilds the original GSM8K-only exports.
# PAGES = KV pages of 4,096 tokens (--max-num-pages). A virtual-slot (speculative) build lists the request's pages once
# per slot, so it needs slots x ceil(ARENA / 4096): 54 for EAGLE3 (6 slots) and 72 for DFlash (8) at the default arena.
# NOHANG=1: the standalone decode of a full 32K arena outlasts run.sh's 45 s launch-hang guard.
set -u
CFG=${1:?cfg ns|e3|df8}; LN=${LANE:-ak4}
case $CFG in ns) SLOTS=1 ;; e3) SLOTS=6 ;; df8) SLOTS=8 ;; *) echo "cfg ns|e3|df8" >&2; exit 1 ;; esac
ARENA=${ARENA:-32769}
NEED=$(( SLOTS * ((ARENA + 4095) / 4096) ))
PAGES=${PAGES:-$(( NEED > 16 ? NEED : 16 ))}
[ $NEED -le $PAGES ] || { echo "ARENA $ARENA needs $NEED KV pages ($SLOTS slots), PAGES=$PAGES" >&2; exit 1; }
[ $CFG != ns ] || [ $PAGES -eq 16 ] || { echo "ns keeps 16 KV pages: ARENA must be <= 65536" >&2; exit 1; }
M=/home/user/mk1; PD=/root/user/nps1/opt/spec/bench_prompts/gsm128
S=/home/user/fleet-lanes/stk15/scripts/nps2/spec/s15/lead
BEST="$(cat $S/best.flags)"; BENV="$(cat $S/best.env)"
A=$(echo "$BEST" | sed -E 's/,-DMPK_ATTN_FIXED_MERGER(,|$)/\1/; s/,-DMPK_ATTN_FM_V=[0-9]+(,|$)/\1/')
LQ=$(python3 -c "
import json
L = [len(json.load(open('/home/user/nps1/opt/spec/tok_g128_k0b6_q%03d.json' % q))['prompt_ids']) for q in range(128)]
q = max(range(128), key=lambda i: L[i]); print('%03d %d' % (q, L[q]))")
set -- $LQ
NEW=$((ARENA - $2)); [ $NEW -ge 264 ] || { echo "ARENA $ARENA leaves $NEW < 264 new tokens"; exit 1; }
SFX=${SFX:-_a$ARENA}
echo "######## export $CFG $(date -u +%FT%TZ) arena from q$1 ($2 prompt tokens); HIP bus map:"
rocm-smi --showbus 2>/dev/null | grep -E 'GPU\[(0|1)\]' | head -n 2
PROVIDERS=${PROVIDERS:-/home/user/mk1/providers}  # host path under /home/user
OUT=${PROVIDERS/#\/home\/user/\/root\/user}/$CFG  # as the container sees it
rm -rf $PROVIDERS/$CFG
SV="MPK_TOK_HOST=1 MPK_HOST_STOP=1 MPK_PROVIDER_EXPORT=$OUT MPK_MAX_SAVE_TOKENS=$((NEW + 64))"
echo "arena $ARENA: prompt $2 + $NEW new tokens, KV pages $PAGES"
case $CFG in
  ns)  MSL=$ARENA LANE=$LN PROMPT=$PD/q$1.txt TO=2400 PS="$SV" bash $M/s15/prod.sh akx_ns$SFX $NEW ;;
  e3)  NOHANG=1 MSL=$ARENA XARGS="--max-num-pages $PAGES" LANE=$LN DRAFTER=eagle3 PROMPT=$PD/q$1.txt T=6 K=5 XF="$BEST" PS="$BENV $SV" TO=2400 bash $M/s15/run.sh akx_e3$SFX $NEW ;;
  df8) NOHANG=1 MSL=$ARENA XARGS="--max-num-pages $PAGES" LANE=$LN DRAFTER=dflash PROMPT=$PD/q$1.txt T=8 K=7 XF="$A" PS="MPK_TAIL_PUB_LATE=1 $SV" TO=2400 bash $M/s15/run.sh akx_df8$SFX $NEW ;;
esac
L=/home/user/nps1/fleet_baselines/lane_$LN/akx_$CFG$SFX.log
grep -aE '\[PROVIDER\]|\[CFG\]|Decode:|\[SPEC\] passes|Traceback|Error' $L | head -n 14 | cut -c1-250
ls -la $PROVIDERS/$CFG/src 2>/dev/null
echo "AGENT_LOOP_WAKE_EXPORT $CFG $(date -u +%T)"
