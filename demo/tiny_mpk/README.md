# Tiny MPK model

An 8-kernel model built with the Mirage persistent-kernel (MPK) API, compiled
into one persistent kernel, and checked token for token against PyTorch. It is
small enough to follow every stage by hand: the kernel graph, the task graph,
the generated `test.cu`, and the run.

```text
x0     = E[tok]                 embedding
n1     = rmsnorm(x0) * w_n1     rmsnorm
u      = n1 @ W_up^T            linear                 (32 tasks)
x1     = u @ W_down^T + x0      linear_with_residual   (32 tasks)
n2     = rmsnorm(x1) * w_n2     rmsnorm
logits = n2 @ W_lm^T            linear, LM head        (64 tasks)
next   = argmax(logits)         argmax_partial (64 tasks) + argmax_reduce
```

H = F = 2048, V = 4096, batch 1, `max_seq_length` 16, prompt
`[11, 222, 3333, 4000]`, end-of-sequence disabled. Row σ(t) of `W_lm` is
`E[t] / sqrt(H)` for a random permutation σ, so the correct next token is
σ(token) with a logit margin near 40. The expected output is known in closed
form.

## Run

Requires a gfx950 GPU (MI350 / MI355) and a built fleet tree
(`pip install -e . -v` from the repo root).

Run from a scratch directory. hipcc is invoked with `--save-temps`, which
writes about 30 MB of intermediates into the current directory.

```bash
export MIRAGE_HOME=/path/to/fleet-chiplet-megakernel
export PYTHONPATH=$MIRAGE_HOME/python:${PYTHONPATH:-}
export HIP_VISIBLE_DEVICES=0      # target GPU
# On a DPX partition (4 XCDs) also: export MPK_NUM_XCDS=4
# On SPX (8 XCDs) leave MPK_NUM_XCDS unset.

mkdir -p /tmp/tiny_mpk && cd /tmp/tiny_mpk
MPK_OUTPUT_DIR=$PWD/build/ python $MIRAGE_HOME/demo/tiny_mpk/tiny_mpk.py
```

The whole script, including the hipcc build, takes about 30 s. It exits
non-zero if the tokens don't match the PyTorch reference.

## Expected output

The script prints the kernel graph size, the task graph Mirage generated, the
hipcc command, one `[FWD_PASS]` line per pass, and then:

```text
== result
prompt       [11, 222, 3333, 4000]
MPK tokens   [11, 222, 3333, 4000, 287, 3123, 3256, 1819, 2053, 2436, 3343, 1558, 3351, 609, 2413, 1663]
PyTorch ref  [11, 222, 3333, 4000, 287, 3123, 3256, 1819, 2053, 2436, 3343, 1558, 3351, 609, 2413, 1663]
sigma chain  [11, 222, 3333, 4000, 287, 3123, 3256, 1819, 2053, 2436, 3343, 1558, 3351, 609, 2413, 1663]
step[0] = 15; tokens match reference: True
last-iteration logits vs reference: cosine 1.000000, max |diff| 0.0000, top-1 margin 39.73
```

Numbers you should see along the way:

| What | Value |
|---|---|
| Kernel graph | 24 nodes: 16 tensor inputs + 8 kernels |
| Task graph | 198 tasks (196 compute + begin + terminate), 73 events |
| Events from LM head to argmax partial | 64: partial task j waits only on LM-head task j |
| Passes in the one launch | 15 = prompt 4 + output 12 − 1 (`iters=14` counts the gaps between passes) |

## Where to look

- `build/task_graph.json`: every task's tensors as (name, byte offset), and
  every task's dependent and trigger event.
- `build/test.cu`: the tensor-name-to-address bindings, and `_execute_task`
  with one branch per (task type, variant).

## Notes

- Only kernels that compile for gfx950 in this tree are used.
  `rmsnorm_linear` is not one of them: its device function lives in
  `tasks/ampere/` and isn't included by the MI300 task header.
- Linear tasks produce 64 output columns each, the CK tile width at batch 1,
  so the grids are F/64, H/64 and V/64. The reduction size must be a multiple
  of 256.
- `max_num_batched_tokens` is 1. For T tokens per pass, give both RMSNorm
  layers a grid of `(T, 1, 1)` and give the activation and token tensors T
  rows.
- The down-projection's residual read of `x0` gets no event of its own. This
  tree links each kernel only to the kernel registered just before it; the
  read is ordered by the full barriers in between.
- Tested on an MI350X DPX+NPS2 partition (`MPK_NUM_XCDS=4`, 124 workers + 4
  schedulers, about 0.14 ms per pass), which needs the driver's
  `mtype_local=1`. Not yet run on SPX.
