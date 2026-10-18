#!/bin/bash
# union.sh: the guide's section 6 expert-union numbers on the latest code's texts. Runs the four other bench prompts
# of the Oct 12 2026 study (HumanEval x2, MT-Bench x2) in the default mode for 384 tokens (labels UTAG_<prompt>_def,
# UTAG=lu), cuts eight texts like the Oct 12 inputs (first 384 generated tokens, up to the end of the answer): GSM8K
# questions 1-3 (run_default.sh's TAG_<question>_def runs), those four, and the 1K conversation's 1,024 tokens from
# section 2's 1-draft EAGLE3 run ($SP1024_TOK). Each text is compared with its Oct 12 counterpart ($OCT12/tok_vbt_*.json,
# tok_vp1k_a.json) when that file exists. Then scripts/nps2/spec/dflash/union_study.py (the reference model's router
# top-4 at every position) runs on the eight texts in $UNION_CONTAINER (fleet_v1, as on Oct 12) on HIP device
# $UNION_GPU (0); its output goes to $GSM_OUT/union_latest.txt. GPU $DEF_BUS must be in SPX/NPS1 on the stock driver.
# Environment: see specg.sh.
set -u -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TREE=$(cd "$HERE/../../../.." && pwd)
HOST_HOME=${HOST_HOME:-$HOME}; CTR_HOME=${CTR_HOME:-/root/$(id -un)}
TREE_CTR=$CTR_HOME${TREE#$HOST_HOME}
G=${GSM_OUT:-$HOST_HOME/mk1/specg_latest}; GC=$CTR_HOME${G#$HOST_HOME}; ST=$G/status_union.txt
TAG=${TAG:-lg}; UTAG=${UTAG:-lu}
SP1024_TOK=${SP1024_TOK:-$HOST_HOME/nps1/opt/spec/tok_bl_dpx_sp1024.json}
OCT12=${OCT12:-$HOST_HOME/nps1/opt/spec}
UNION_CONTAINER=${UNION_CONTAINER:-fleet_v1}; UNION_GPU=${UNION_GPU:-0}
P=$TREE_CTR/scripts/nps2/spec/bench_prompts
mkdir -p $G
st() { echo "[UNION] $(date -u +%T) $*" | tee -a $ST; }
for p in humaneval_1 humaneval_2 mtbench_1 mtbench_2; do
  PROMPT=$P/$p.txt bash $HERE/specg.sh def 384 ${UTAG}_${p}_def | tee -a $ST
done
G=$G TAG=$TAG UTAG=$UTAG SP1024_TOK=$SP1024_TOK OCT12=$OCT12 python3 - << 'EOF' 2>&1 | tee -a $ST
import json
import os
G = os.environ["G"].rstrip("/") + "/"
O = os.environ["OCT12"].rstrip("/") + "/"
RET = 200002
src = {"gsm8k_1": "%s_gsm8k_1_def", "gsm8k_2": "%s_gsm8k_2_def", "gsm8k_3": "%s_gsm8k_3_def"}
src = {k: v % os.environ["TAG"] for k, v in src.items()}
for p in ("humaneval_1", "humaneval_2", "mtbench_1", "mtbench_2"):
    src[p] = "%s_%s_def" % (os.environ["UTAG"], p)


def ident(t, o):
    m = min(len(t), len(o))
    return next((i for i in range(m) if t[i] != o[i]), m)


for name, lab in src.items():
    d = json.load(open(G + "tok_%s.json" % lab))
    t = d["token_ids"][:384]
    if RET in t:
        t = t[: t.index(RET) + 1]
    json.dump({"prompt_ids": d["prompt_ids"], "token_ids": t}, open(G + "tok_ut_%s.json" % name, "w"))
    if os.path.exists(O + "tok_vbt_%s.json" % name):
        o = json.load(open(O + "tok_vbt_%s.json" % name))
        print("%-12s latest %3d tokens, Oct 12 %3d; prompt same %s; identical through token %d" % (
            name, len(t), len(o["token_ids"]), d["prompt_ids"] == o["prompt_ids"], ident(t, o["token_ids"])))
d = json.load(open(os.environ["SP1024_TOK"]))
json.dump({"prompt_ids": d["prompt_ids"], "token_ids": d["token_ids"]}, open(G + "tok_ut_1k.json", "w"))
if os.path.exists(O + "tok_vp1k_a.json"):
    o = json.load(open(O + "tok_vp1k_a.json"))
    print("%-12s latest %4d tokens, Oct 12 %4d; prompt same %s; identical through token %d" % (
        "1k (sp1024)", len(d["token_ids"]), len(o["token_ids"]), d["prompt_ids"] == o["prompt_ids"],
        ident(d["token_ids"], o["token_ids"])))
EOF
[ $? = 0 ] || { st "preparing the texts failed: not running the study"; exit 1; }
st "union study ($UNION_CONTAINER) on the latest texts"
docker start $UNION_CONTAINER > /dev/null; sleep 5
U=$TREE_CTR/scripts/nps2/spec/dflash/union_study.py
F=$GC/tok_ut_gsm8k_1.json,$GC/tok_ut_gsm8k_2.json,$GC/tok_ut_gsm8k_3.json,$GC/tok_ut_humaneval_1.json,$GC/tok_ut_humaneval_2.json,$GC/tok_ut_mtbench_1.json,$GC/tok_ut_mtbench_2.json,$GC/tok_ut_1k.json
docker exec $UNION_CONTAINER bash -lc "HIP_VISIBLE_DEVICES=$UNION_GPU python3 $U $F" < /dev/null > $G/union_latest.txt 2>&1
grep -E 'U\(B=1' $G/union_latest.txt | tee -a $ST
st "done"

