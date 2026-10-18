"""Native fleet megakernel in vLLM (FLEET_MK_NATIVE=1): the persistent megakernel exactly as demo.py builds it.

The kernel owns prefill, its KV cache and its weights (including the DPX/CPX MPK_MULTI_DEV placement). Each
request is one free-running launch (native_session.NativeSession), and its tokens stream from pinned host
memory (build with MPK_TOK_HOST=1). vLLM keeps its API server, scheduler, detokenizer and metrics:

* FleetNativeGptOssForCausalLM: an attention-free, weightless stand-in for vLLM's model init and memory
  profiling (run with load_format="dummy"). Its constructor builds the megakernel from FLEET_NATIVE_DEMO
  (path to demo.py) and FLEET_NATIVE_ARGS (its argv, JSON list); the MPK_* environment must be the one the
  standalone run uses.
* native_worker.FleetNativeWorker (--worker-cls fleet_megakernel_vllm.native_worker.FleetNativeWorker):
  execute_model hands the scheduled request to the session and returns the tokens produced so far, so vLLM's
  model runner and sampler never run while serving.
* A step can return several tokens; the scheduler hook advances num_computed_tokens past all of them.

Limits: one request at a time (max_num_seqs=1), greedy, no logprobs, no async scheduling, prompt plus
max_tokens within the compiled arena. When vLLM finishes a request early (EOS, stop token or string,
abort), the host stop flag ends its launch within 16 iterations (build with MPK_HOST_STOP=1); without it
the launch runs to max_tokens and the next request waits for it.
"""
import contextlib
import json
import os
import shlex
from typing import ClassVar, Literal

import torch
from torch import nn

_DRIVER = None


class _Driver:
    def __init__(self, session):
        self.sess = session
        self.reqs = {}  # req_id -> [prompt ids, max_tokens, prompt tokens scheduled so far]
        self.active = None

    def execute(self, so):
        """{req_id: new token ids} for this engine step (at most one scheduled request)."""
        for nr in so.scheduled_new_reqs:
            sp = nr.sampling_params
            if sp is None or sp.temperature > 0:
                raise RuntimeError("fleet native serving is greedy only (temperature 0)")
            if sp.logprobs:
                raise RuntimeError("fleet native serving does not return logprobs")
            self.reqs[nr.req_id] = [list(nr.prompt_token_ids), int(sp.max_tokens), 0]
        for rid in so.finished_req_ids:
            self.reqs.pop(rid, None)
            if rid == self.active:
                self.sess.stop()
                self.active = None
        sched = so.num_scheduled_tokens
        if not sched:
            return {}
        if len(sched) != 1:
            raise RuntimeError("fleet native: one request at a time (run with max_num_seqs=1)")
        rid, n = next(iter(sched.items()))
        st = self.reqs[rid]
        if rid != self.active:
            self.sess.start(st[0], st[1])
            self.active = rid
        st[2] += n
        if st[2] < len(st[0]):
            return {rid: []}  # vLLM is still feeding the prompt in chunks
        toks = self.sess.poll()
        if not toks:
            raise RuntimeError("fleet native: the launch ended before request %s had its tokens" % rid)
        return {rid: toks}


@contextlib.contextmanager
def _standalone_defaults():
    """demo.py's own defaults (CPU tensors, float32) instead of vLLM's model-init context."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("cpu"):
            yield
    finally:
        torch.set_default_dtype(old)


def driver():
    global _DRIVER
    if _DRIVER is None:
        from .native_session import NativeSession
        demo = os.environ["FLEET_NATIVE_DEMO"]
        raw = os.environ.get("FLEET_NATIVE_ARGS", "[]")
        args = json.loads(raw) if raw.lstrip().startswith("[") else shlex.split(raw)
        with _standalone_defaults():
            _DRIVER = _Driver(NativeSession(demo, args))
    return _DRIVER


class FleetNativeGptOssForCausalLM(nn.Module):
    is_attention_free: ClassVar[Literal[True]] = True

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__()
        mc = vllm_config.model_config
        self.vocab_size = mc.get_vocab_size()
        dev = torch.device("cuda", torch.cuda.current_device())
        sc = vllm_config.scheduler_config
        self._hidden = torch.zeros(max(sc.max_num_batched_tokens, 1), 1, dtype=mc.dtype, device=dev)
        self._logits = torch.zeros(max(sc.max_num_seqs, 1), self.vocab_size, dtype=torch.float32,
                                   device=dev)
        driver()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._hidden[: input_ids.shape[0]]

    get_input_embeddings = embed_input_ids

    def forward(self, input_ids=None, positions=None, intermediate_tensors=None, inputs_embeds=None,
                **kwargs):
        n = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        if n > self._hidden.shape[0]:
            self._hidden = torch.zeros(n, 1, dtype=self._hidden.dtype, device=self._hidden.device)
        return self._hidden[:n]

    def compute_logits(self, hidden_states: torch.Tensor, *args) -> torch.Tensor:
        n = hidden_states.shape[0]
        if n > self._logits.shape[0]:
            self._logits = torch.zeros(n, self.vocab_size, dtype=torch.float32,
                                       device=self._logits.device)
        return self._logits[:n]

    def load_weights(self, weights):
        for _ in weights:
            pass
        return set()


def _stop_if_active(req_ids):
    """End the running launch as soon as vLLM finishes its request, not at the next engine step.

    req_ids=None means every request."""
    d = _DRIVER
    if d is not None and d.active is not None and (req_ids is None or d.active in req_ids):
        d.sess.stop()


def _install_scheduler_hook():
    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler, "_fleet_native_hook", False):
        return
    orig = Scheduler._update_request_with_output
    orig_finish = Scheduler.finish_requests

    def update(self, request, new_token_ids, *args, **kwargs):
        accepted, stopped = orig(self, request, new_token_ids, *args, **kwargs)
        stale = kwargs.get("is_stale", args[0] if args else False)
        if not stale and len(accepted) > 1:
            # schedule() counted one token; the launch already holds KV for the others.
            request.num_computed_tokens += len(accepted) - 1
        if stopped:  # EOS, stop token, max_tokens: checked here, in the scheduler
            _stop_if_active({request.request_id})
        return accepted, stopped

    def finish(self, request_ids, *args, **kwargs):
        # Aborts, and stop strings (the frontend detects them and aborts the request in the engine).
        if request_ids is not None and not isinstance(request_ids, str):
            request_ids = list(request_ids)
        out = orig_finish(self, request_ids, *args, **kwargs)
        if request_ids is None:
            _stop_if_active(None)
        else:
            _stop_if_active({request_ids} if isinstance(request_ids, str) else set(request_ids))
        return out

    Scheduler._update_request_with_output = update
    Scheduler.finish_requests = finish
    Scheduler._fleet_native_hook = True


def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model("GptOssForCausalLM", FleetNativeGptOssForCausalLM)
    _install_scheduler_hook()

