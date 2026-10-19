# GPT-OSS-120B blog numbers: the megakernel outside vLLM

The full guide, which lists every GPT-OSS-120B number of the blog with its value on the latest code and the command
behind it, is in the aitermk tree (github.com/proaditya/aiter, branch `aitermk-gpt-oss`):
`aitermk/megakernels/fleet/src/gpt_oss_gfx950_mpk/BLOG_REPRODUCTION.md`. Every number is given
with the checkpoint's BF16 LM head (as stock vLLM runs it, `MPK_LM_BF16`) and with the megakernel's MXFP4 head. The
served numbers come from `../fair/serve_fair31.sh` (BF16 head and its MXFP4 arm) and the aitermk tree's
`aitermk/engine_adapters/vllm/scripts/gpt_oss_build_and_repro.sh` (the released 32K builds). This directory holds the scripts
for the standalone half (locality, per-phase, speculation outside vLLM):

| file | what it does |
|---|---|
| `setup_lane.sh NAME` | a worktree for the runs, reusing built dependencies (one per GPU that runs at the same time) |
| `run_dpx.sh [phase length swap lengthbf swapbf ckbf]` | DPX+NPS2 runs, stock driver (GPU already switched with `../dpx/to_dpx_nps2_stock.sh`). `*bf` groups: the BF16 LM head; `ckbf`: the 32K BF16 single stream at 15 KV chunks, the reference EAGLE3's outputs must equal |
| `run_default.sh [phase length lengthbf]` | default-mode runs (stock SPX+NPS1, the factory mode) |
| `blog_tab.py` | the tables from those logs and from the served results (`--served`, `--served-bf`); `--charts FILE` writes the chart data, its figures with the MXFP4 LM head as in the blog (`--charts-head bf`: the BF16 head) |
| `blog_charts.py DATA OUTDIR` | the figures, in the blog's style |
| `dry_check.py REF DRY` | compares the environments of a `DRY=1` run with the ones recorded in run logs |
| `envs_ref_oct18.txt` | the environments recorded by 16 of the 27 runs of the groups `phase length swap` (MXFP4 head, and the BF16 phase runs) behind the Oct 18 2026 numbers: every distinct composition (the other 11 differ from one of them only in length). Check with `DRY=1 bash run_dpx.sh phase length swap > a; DRY=1 bash run_default.sh phase length >> a; python3 dry_check.py envs_ref_oct18.txt a`. The `*bf` runs ran through these scripts themselves |
| `stock_vllm_031.json` | stock vLLM 0.31.0 GSM8K-128 baseline (node 2, Oct 17 2026) |
| `common.sh` | the run compositions (sourced by the run scripts) |
| `spec_records.py` | the kernel's `[SPEC]` counter records from a served log, rejoining the ones another process's output split |
| `gsm8k/run_dpx.sh [QUESTION ...]` | GSM8K questions 1-3 (single stream, 7-chunk single stream, 1 and 3 drafts, each at its answer's length) and the 1K conversation (7-chunk single stream and 3 drafts, 1,024 tokens), DPX+NPS2 |
| `gsm8k/run_default.sh [QUESTION ...]` | the default-mode single stream on GSM8K questions 1-3 |
| `gsm8k/union.sh` | the expert-union study (after `gsm8k/run_default.sh`; four more bench prompts in the default mode, then `../spec/dflash/union_study.py`) |
| `gsm8k/table.py` | the GSM8K and conversation tables from those runs |
| `gsm8k/specg.sh KIND N LABEL`, `gsm8k/q_specg.py` | one run of that method and its summary line (the Oct 13 2026 `specg.sh`, paths from the environment; `LM=bf16` for the BF16 head) |

The memory partition (NPS) applies to all 8 GPUs of a node, so the DPX+NPS2 and default-mode runs need either two
nodes or a mode switch in between. Example, with the vLLM 0.27.1 image and the ROCm 7.0 compiler copied to
`~/rocm70` (the megakernel is built with ROCm 7.0 in every published number):

```bash
cd ~/fleet-lanes/<tree>/scripts/nps2/blog
CONTAINER=fleet_ak4t1 PATH_PREPEND=/root/user/fleet-lanes/<tree>/scripts/nps2/vllm/hip70 BUS=e5 bash run_dpx.sh
# on a node in the default mode (another worktree if the same NFS home):
CONTAINER=fleet_ship2 PATH_PREPEND=/root/user/fleet-lanes/<tree2>/scripts/nps2/vllm/hip70 BUS=05 LANE=<tree2> bash run_default.sh
python3 blog_tab.py --dpx ~/nps1/fleet_baselines/lane_<tree> --default ~/nps1/fleet_baselines/lane_<tree2> \
  --served <WORK>/results --served-bf <serve_fair31 dir> --served-bf-logs ~/nps1/fleet_baselines/lane_<tree> \
  --charts charts_data.json
python3 blog_charts.py charts_data.json charts/
# GSM8K and the 1K conversation (guide section 4), then the expert unions (section 6); BF16 head, then MXFP4:
D="DPX_LANE=<tree> DPX_BUS=e5 DPX_CONTAINER=fleet_ak4t1"; F="DPX_LANE=<tree> DEF_LANE=<tree2> DEF_BUS=05 DEF_CONTAINER=fleet_ship2"
T=~/fleet-lanes/<tree>/demo/gpt_oss/blog_tokens
env $D LM=bf16 TAG=bf CTAG=bfc GSM_OUT=<gsm_bf> bash gsm8k/run_dpx.sh
env $F LM=bf16 TAG=bf GSM_OUT=<gsm_bf> bash gsm8k/run_default.sh
env $F LM=bf16 TAG=bf UTAG=bfu GSM_OUT=<gsm_bf> SP1024_TOK=$T/tok_bl_dpx_spbf1024.json bash gsm8k/union.sh
python3 gsm8k/table.py --out <gsm_bf> --tag bf --ctag bfc
env $D GSM_OUT=<gsm_mx> bash gsm8k/run_dpx.sh
env $F GSM_OUT=<gsm_mx> bash gsm8k/run_default.sh
env $F GSM_OUT=<gsm_mx> SP1024_TOK=$T/tok_bl_dpx_sp1024.json bash gsm8k/union.sh
python3 gsm8k/table.py --out <gsm_mx>
```

The `gsm8k/` scripts take their environment from `gsm8k/specg.sh` (header there). The speculative runs use each
answer's own length, as the Oct 13 2026 runs behind the blog did, so the speculative lengths come from the single
stream runs before them. `PATH_PREPEND` defaults to the DPX+NPS2 tree's ROCm 7.0 wrapper for both modes. `LM=bf16`
adds the BF16 head's variables (`MPK_LM_BF16=1`, the MXFP4 head's pipelines off) and appends `-DMPK_LM_BF16` to the
run's own `MPK_EXTRA_FLAGS`; `DRY=1 bash gsm8k/specg.sh ...` prints a run's environment without running it.

