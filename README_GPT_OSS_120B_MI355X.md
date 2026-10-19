# GPT-OSS-120B on MI355X: the fleet persistent megakernel behind the blog's numbers

This branch is the megakernel source the GPT-OSS-120B results of the megakernel blog were measured with, the
exporter that turns a compiled megakernel into the provider libraries the vLLM plugin loads, and the scripts behind
the numbers. It is one commit on top of `ROCm/fleet-chiplet-megakernel` at `1288b43` (branch
`amd_mi355_gpt_oss120b`, Sep 26 2026).

The served path (the provider libraries, the vLLM plugin `fleet_vllm`, the one-command build-and-reproduce script
and the reproduction guide `BLOG_REPRODUCTION.md`) lives in the `aitermk/` tree of the AITER fork,
github.com/proaditya/aiter, branch `aitermk-gpt-oss`.

## What is here

| Path | What it is |
|---|---|
| `include/mirage/persistent_kernel/` | the megakernel: persistent scheduler and workers, the fused per-layer task (QKV, attention, O-proj + router, TopK, MoE), the LM head, EAGLE3 and DFlash drafting, the multi-token verify engine |
| `python/mirage/mpk/` | the graph builder and runtime glue; `placement.py` (data placement per die), `provider_export.py` (the exporter) |
| `demo/gpt_oss/` | `demo.py`, the GPT-OSS-120B driver (standalone decode, `--vllm-session`, perplexity mode) |
| `vllm_integration/` | the original `fleet_mk` vLLM plugin (native mode), used for the served BF16 LM head numbers |
| `NPS2_RECIPE.env`, `scripts/nps2/opt2/best.env`, `scripts/nps2/spec/` | the flag sets of the published runs |
| `scripts/nps2/blog/` | the standalone half of the blog (locality, per phase, speculation outside vLLM); README there |
| `scripts/nps2/fair/` | the served BF16 LM head sweep (`serve_fair31.sh`) and the accuracy evals (`eval/`, README there) |
| `scripts/nps2/aitermk/` | the provider exports (`exp_run.sh` with `s15/prod.sh`, `s15/run.sh`) and the served table (`hang/t1_table.sh`) |
| `scripts/nps2/dpx/`, `scripts/nps2/to_nps1_stock.sh` | partition switches on the stock driver (DPX+NPS2 and back to SPX+NPS1) |
| `scripts/nps2/vllm/hip70/hipcc` | the hipcc wrapper that compiles the megakernel with ROCm 7.0 inside a newer container |

Not here: the experiment scripts and notes of the development history, and the opt-in paths for a patched amdgpu
driver (`MPK_AID_*`), which none of the blog's runs use; every run behind the blog used the stock driver.

## Build

```bash
git clone -b gpt-oss-120b-mi355x https://github.com/sangeeta0201/fleet-chiplet-megakernel.git
cd fleet-chiplet-megakernel && git submodule update --init --recursive
python setup.py build_ext --inplace       # in a ROCm PyTorch container; builds python/mirage/core*.so
```

The megakernel itself is compiled at run time by hipcc. The published numbers used ROCm 7.0's hipcc (AMD clang
20.0.0git, roc-7.0.0 25314) inside `vllm/vllm-openai-rocm` containers with ROCm 7.2.3: put
`scripts/nps2/vllm/hip70` first on `PATH` and point `ROCM70` at a ROCm 7.0.0 root (an Ubuntu 22.04 build for those
images). Another compiler builds other device code: the output can differ from the published tokens.

## Run

- The standalone numbers: `scripts/nps2/blog/README.md`. Every run goes through `scripts/nps2/fleet_run_lane.sh`
  (one worktree per GPU that runs at the same time, `scripts/nps2/blog/setup_lane.sh`), with the flags composed in
  `scripts/nps2/blog/common.sh`.
- Partitions: `scripts/nps2/dpx/to_dpx_nps2_stock.sh` switches a node to DPX+NPS2 and `scripts/nps2/to_nps1_stock.sh`
  back to SPX+NPS1 (the factory setting), both on the stock driver, through a privileged container: set `IMG` to an
  image that has `modprobe` (kmod). The memory partition applies to all GPUs of a node.
- Provider exports: `MPK_PROVIDER_EXPORT=<dir>` makes `demo.py` write `<dir>/src` (the generated kernel with its
  device addresses replaced by a pointer table, the task graph, the compile command and the `MPK_*` environment) and
  `<dir>/state` (the device memory the kernel's pointers land in) where `--vllm-session` hands over.
  `scripts/nps2/aitermk/exp_run.sh ns|e3|df8` made the published exports (`ARENA=32769` for the 32K builds; `LM=bf16 ARENA=466` for the BF16-LM-head builds behind the
  matched-precision numbers, written to `<cfg>_bf16`), with
  the flag files `scripts/nps2/spec/s15/lead/best.flags` and `best.env`. Each export's compile flags and
  environment are also recorded in the aitermk tree, `aitermk/megakernels/fleet/src/gpt_oss_gfx950_mpk/configs/`.

## The scripts' assumptions

The scripts were written for one node layout, and they show how the runs were done rather than running anywhere
unchanged:

- The home directory (`HOST_HOME`, default `$HOME`) is mounted in the containers at `CTR_HOME` (default
  `/root/<your login>`); scripts that name a home directory directly use `/home/user` and `/root/user` for them.
- Worktrees of this repository sit in `~/fleet-lanes/<name>`. Run logs, prompts, token references and some helper
  scripts sit in `~/nps1` and `~/mk1`, outside this repository.
- One MI355X node in DPX+NPS2 (the GPU at HIP devices 12 and 13) and a second node in SPX+NPS1 ran the two halves.
- Keep GPU monitoring off while the megakernel runs: AMD SMI metric queries stall it, and so can starting or ending
  another GPU process on the same node while a launch is in flight.
