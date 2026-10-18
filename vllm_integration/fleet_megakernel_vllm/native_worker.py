"""vLLM worker for the native fleet megakernel (see native.py): --worker-cls
fleet_megakernel_vllm.native_worker.FleetNativeWorker. Imported only by the engine-core process."""
import dataclasses

import torch
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu_worker import Worker

from .native import driver


def _output(req_ids, tokens):
    kw = dict(req_ids=req_ids, req_id_to_index={r: i for i, r in enumerate(req_ids)},
              sampled_token_ids=tokens)
    names = {f.name for f in dataclasses.fields(ModelRunnerOutput)}
    for k, v in (("logprobs", None), ("prompt_logprobs_dict", {}), ("pooler_output", [])):
        if k in names:
            kw[k] = v
    return ModelRunnerOutput(**kw)


class FleetNativeWorker(Worker):
    @torch.inference_mode()
    def execute_model(self, scheduler_output, *args, **kwargs):
        out = driver().execute(scheduler_output)
        return _output(list(out.keys()), list(out.values()))

