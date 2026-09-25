"""Tiny end-to-end Mirage persistent-kernel (MPK) model on one MI350/MI355 GPU.

Model, batch 1, hidden H, ffn F, vocab V:

    x0     = E[tok]                      embedding
    n1     = rmsnorm(x0) * w_n1          rmsnorm
    u      = n1 @ W_up^T                 linear
    x1     = u @ W_down^T + x0           linear_with_residual  (skip connection)
    n2     = rmsnorm(x1) * w_n2          rmsnorm
    logits = n2 @ W_lm^T                 linear (LM head)
    next   = argmax(logits)              argmax_partial + argmax_reduce

Row sigma[t] of W_lm is E[t] / sqrt(H), so the greedy next token is sigma(tok)
with a wide logit margin: the expected output is known in closed form.

See README.md in this directory for how to run it. Exits non-zero if the
megakernel's tokens differ from the PyTorch reference.
"""

import json

import torch

import mirage as mi

H, F, V = 2048, 2048, 4096
S = 16  # max_seq_length = prompt + generated tokens
PROMPT = [11, 222, 3333, 4000]
COLS_PER_TASK = 64  # CK linear tile width on gfx950 at batch 1
P = V // COLS_PER_TASK  # argmax partial tasks

dev = "cuda"
bf16 = torch.bfloat16
torch.manual_seed(0)

# ----------------------------------------------------------------------------
# Weights
# ----------------------------------------------------------------------------
E = torch.randn(V, H, device=dev).to(bf16)
sigma = torch.randperm(V, device=dev)
W_lm = torch.empty(V, H, device=dev, dtype=bf16)
W_lm[sigma] = (E.float() / H**0.5).to(bf16)
w_n1 = (1 + 0.05 * torch.randn(H, device=dev)).to(bf16)
w_n2 = (1 + 0.05 * torch.randn(H, device=dev)).to(bf16)
W_up = (0.01 * torch.randn(F, H, device=dev)).to(bf16)
W_down = (0.01 * torch.randn(H, F, device=dev)).to(bf16)


# ----------------------------------------------------------------------------
# PyTorch reference, rounding to bf16 at the same op boundaries as the tasks
# ----------------------------------------------------------------------------
def rms(x, w):
    r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
    return (x * r * w.float()).to(bf16).float()


def ref_logits(tok):
    x0 = E[tok].float().unsqueeze(0)
    n1 = rms(x0, w_n1)
    u = (n1 @ W_up.float().t()).to(bf16).float()
    x1 = (u @ W_down.float().t() + x0).to(bf16).float()
    n2 = rms(x1, w_n2)
    return (n2 @ W_lm.float().t()).to(bf16)


ref_tokens = list(PROMPT)
while len(ref_tokens) < S:
    ref_tokens.append(int(ref_logits(ref_tokens[-1]).float().argmax()))
closed_form = list(PROMPT)
while len(closed_form) < S:
    closed_form.append(int(sigma[closed_form[-1]]))

# ----------------------------------------------------------------------------
# Meta tensors: the host <-> kernel interface for the decode loop
# ----------------------------------------------------------------------------
tokens = torch.zeros(1, S, dtype=torch.long, device=dev)
tokens[0, : len(PROMPT)] = torch.tensor(PROMPT)
prompt_lengths = torch.tensor([len(PROMPT)], dtype=torch.int32, device=dev)
step = torch.zeros(1, dtype=torch.int32, device=dev)
num_new_tokens = torch.ones(1, dtype=torch.int32, device=dev)
input_tokens = torch.zeros(1, 1, dtype=torch.long, device=dev)
output_tokens = torch.zeros(1, 1, dtype=torch.long, device=dev)
meta = {
    "step": step,
    "tokens": tokens,
    "input_tokens": input_tokens,
    "output_tokens": output_tokens,
    "num_new_tokens": num_new_tokens,
    "prompt_lengths": prompt_lengths,
    "qo_indptr_buffer": torch.empty(2, dtype=torch.int32, device=dev),
    "paged_kv_indptr_buffer": torch.empty(2, dtype=torch.int32, device=dev),
    "paged_kv_indices_buffer": torch.zeros(16, dtype=torch.int32, device=dev),
    "paged_kv_last_page_len_buffer": torch.empty(1, dtype=torch.int32, device=dev),
}

num_workers, num_schedulers = mi.get_configurations_from_gpu(0)
mpk = mi.PersistentKernel(
    mode="offline",
    world_size=1,
    mpi_rank=0,
    num_workers=num_workers,
    num_local_schedulers=num_schedulers,
    num_remote_schedulers=0,
    max_seq_length=S,
    max_num_batched_requests=1,
    max_num_batched_tokens=1,
    max_num_pages=16,
    page_size=4096,
    eos_token_id=0x7FFFFFFF,
    meta_tensors=meta,
    profiler_tensor=None,
    trace_name="",
    spec_decode_config=None,
    use_cutlass_kernel=False,
)

# ----------------------------------------------------------------------------
# Tensors. The generated kernel holds raw addresses, so keep every one alive.
# ----------------------------------------------------------------------------
keep = {}


def attach(t, name):
    keep[name] = t
    return mpk.attach_input(torch_tensor=t, name=name)


def buf(name, shape, dtype=bf16):
    return attach(torch.zeros(shape, dtype=dtype, device=dev), name)


tok_in = attach(input_tokens, "input_token")
w_embed = attach(E, "embed_tokens")
x0 = buf("x0", (1, H))
t_w_n1 = attach(w_n1, "norm1_w")
n1 = buf("norm1_out", (1, H))
t_w_up = attach(W_up, "w_up")
u = buf("mlp_up_out", (1, F))
t_w_down = attach(W_down, "w_down")
x1 = buf("x1", (1, H))
t_w_n2 = attach(w_n2, "norm2_w")
n2 = buf("norm2_out", (1, H))
t_w_lm = attach(W_lm, "lm_head")
logits = buf("logits", (1, V))
part_val = buf("argmax_part_value", (1, P))
part_idx = buf("argmax_part_index", (1, P), torch.long)
tok_out = attach(output_tokens, "output_token")

# ----------------------------------------------------------------------------
# Kernels, in execution order
# ----------------------------------------------------------------------------
blk = (128, 1, 1)
mpk.embed_layer(input=tok_in, weight=w_embed, output=x0,
                grid_dim=(1, 1, 1), block_dim=blk, input_source=1)
mpk.rmsnorm_layer(input=x0, weight=t_w_n1, output=n1,
                  grid_dim=(1, 1, 1), block_dim=blk)
mpk.linear_layer(input=n1, weight=t_w_up, output=u,
                 grid_dim=(F // COLS_PER_TASK, 1, 1), block_dim=blk)
mpk.linear_with_residual_layer(input=u, weight=t_w_down, residual=x0,
                               output=x1,
                               grid_dim=(H // COLS_PER_TASK, 1, 1),
                               block_dim=blk)
mpk.rmsnorm_layer(input=x1, weight=t_w_n2, output=n2,
                  grid_dim=(1, 1, 1), block_dim=blk)
mpk.linear_layer(input=n2, weight=t_w_lm, output=logits,
                 grid_dim=(V // COLS_PER_TASK, 1, 1), block_dim=blk)
mpk.argmax_partial_layer(input=logits, output=(part_val, part_idx),
                         grid_dim=(P, 1, 1), block_dim=blk)
mpk.argmax_reduce_layer(input=(part_val, part_idx), output=tok_out,
                        grid_dim=(1, 1, 1), block_dim=blk)

n_nodes = len(mpk.kn_graph.cygraph.get_graph_structure())
print(f"\n== kernel graph: {n_nodes} nodes "
      f"({len(keep)} tensor inputs + {n_nodes - len(keep)} kernels)")

# ----------------------------------------------------------------------------
# Task graph, summarized from the JSON Mirage itself produced
# ----------------------------------------------------------------------------
res = mpk.kn_graph.generate_task_graph(num_gpus=1, my_gpu_id=0)
g = json.loads(res["json_file"])
tasks, events = g["all_tasks"], g["all_events"]
INVALID = 0x7FFFFFFFFFFFFFFE
TNAME = {0: "terminate", 10: "begin_task_graph", 101: "embedding",
         119: "rmsnorm", 120: "linear", 108: "linear_with_residual",
         110: "argmax_partial", 111: "argmax_reduce"}
ENAME = {900: "EMPTY", 901: "LAUNCH_TASKS", 902: "LAUNCH_MASSIVE",
         903: "LAUNCH_DEPENDENT", 910: "END_OF_TASK_GRAPH",
         911: "TERMINATION"}


def ev(x):
    return None if x == INVALID else (x & 0xFFFFFFFF)


# Contiguous runs of one task type = one kernel's tasks.
runs = []
for i, t in enumerate(tasks):
    if runs and runs[-1]["type"] == t["task_type"] and i == runs[-1]["hi"]:
        runs[-1]["hi"] = i + 1
    else:
        runs.append({"type": t["task_type"], "lo": i, "hi": i + 1})
print(f"\n== task graph: {len(tasks)} tasks, {len(events)} events")
print(f"{'tasks':>9}  {'kernel':<21} {'waits on event':<16} signals event")
for r in runs:
    rng = tasks[r["lo"]:r["hi"]]
    deps = sorted({ev(t["dependent_event"]) for t in rng} - {None})
    trig = sorted({ev(t["trigger_event"]) for t in rng} - {None})

    def fmt(s):
        if not s:
            return "-"
        return str(s[0]) if len(s) == 1 else f"{s[0]}..{s[-1]} ({len(s)})"

    print(f"{r['lo']:>4}-{r['hi'] - 1:<4}  {TNAME[r['type']]:<21} "
          f"{fmt(deps):<16} {fmt(trig)}")

print(f"\n{'event':>5}  {'type':<18} {'triggers':>8}  consumer tasks")
for i, e in enumerate(events):
    if 10 <= i < len(events) - 3:
        if i == 10:
            print("  ...  (one event per LM-head task / argmax-partial pair)")
        continue
    rng = f"[{e['first_task_id']}, {e['last_task_id']})"
    print(f"{i:>5}  {ENAME[e['event_type']]:<18} {e['num_triggers']:>8}  {rng}")

up_task = next(t for t in tasks if t["task_type"] == 120)
up5 = tasks[tasks.index(up_task) + 5]
print("\nslice offsets for the 6th up-projection task (bid.x = 5):")
for kind in ("inputs", "outputs"):
    for d in up5[kind]:
        print(f"  {kind[:-1]:<6} {d['base_ptr']:<12} offset {d['offset']:>8} B"
              f"  tile {d['dims']}")

# ----------------------------------------------------------------------------
# Codegen + hipcc, host setup, then one launch for the whole request
# ----------------------------------------------------------------------------
mpk.compile()
mpk()
torch.cuda.synchronize()

# ----------------------------------------------------------------------------
# Result
# ----------------------------------------------------------------------------
got = tokens[0].tolist()
print("\n== result")
print(f"prompt       {PROMPT}")
print(f"MPK tokens   {got}")
print(f"PyTorch ref  {ref_tokens}")
print(f"sigma chain  {closed_form}")
print(f"step[0] = {int(step[0])}; tokens match reference: {got == ref_tokens}")

ref_last = ref_logits(got[S - 2]).float()
mpk_last = keep["logits"].float()
cos = torch.nn.functional.cosine_similarity(mpk_last, ref_last).item()
top2 = ref_last[0].topk(2).values
print(f"last-iteration logits vs reference: cosine {cos:.6f}, "
      f"max |diff| {(mpk_last - ref_last).abs().max().item():.4f}, "
      f"top-1 margin {(top2[0] - top2[1]).item():.2f}")

raise SystemExit(0 if got == ref_tokens else 1)
