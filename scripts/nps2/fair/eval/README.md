# Accuracy evals of the default-mode megakernel through the original fleet_mk plugin

The megakernel accuracy rows of the aitermk tree's `PRECISION_MATCHED_COMPARISON.md` (sections 3-3b: the `MPK_LM_BF16`
rows, the default build's 5-shot reproduction, MMLU-Pro) came from these scripts on one MI355X in SPX/NPS1.

- `serve_eval.py`: the session script (`MPK_WRAP=.../wrap_sess.sh SESS_SCRIPT=serve_eval.py`): starts `vllm serve`
  with the `fleet_mk` plugin in native mode inside the megakernel's process, runs the evaluators named in `SESS_EVALS`
  (`gsm8k5`, `chat`, `mmlupro`) one request at a time, restarts the server if it stops answering, and writes
  `SESS_EVAL_OUT/<eval>/{items,summary}.json` plus `[SERVEEVAL] ... RESULT` lines.
- `mk_eval_def.sh LANE BUS LABEL HEAD EVALS [K=V ...]`: one session (`HEAD` = `mx` or `bf`, the BF16 LM head) with the
  blog's default-mode configuration (`scripts/nps2/blog/common.sh`).
- `seq_ev3.sh`: the sessions one at a time, each with `MPK_HOST_STOP=1` (without it a launch decodes on to the
  request's `max_tokens` after vLLM stopped it) and chat prompts dated per pairing (`VLLM_SYSTEM_START_DATE`);
  re-entrant. `seq_guard.sh` keeps it running across Slurm job handovers (a job end stops the containers).

Run sessions one at a time: ending a GPU process on the node hung another session's megakernel launch in flight.
The paths in these scripts are those of the node they ran on; adjust them to yours.
