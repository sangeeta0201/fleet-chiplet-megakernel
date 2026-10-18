"""serve_eval.py: one megakernel build behind `vllm serve` (fleet_mk plugin, native mode), then evaluators against it.

Launched by wrap_sess.sh (MPK_WRAP, SESS_SCRIPT=this file) in place of `python demo.py ...`, so the megakernel is built
with exactly the run's argv and environment (the session execs demo.py with --vllm-session), as serve128.py does.
The build's token arena is its prompt + MAX_NEW_TOKENS: it must cover the longest request (prompt + max_tokens).
Evaluators (SESS_EVALS, comma-separated, run in order on one server; restarted if it dies):
  gsm8k5   vLLM's GSM8K evaluator (geval_eval.py: 5-shot raw completions, 256 tokens, greedy, seed 42), 1,319 questions
  chat     gchat_eval.py (GSM8K in chat format, reasoning low, up to 4,096 tokens), 1,319 questions
  mmlupro  mmlupro_eval.py (the 1,400-question MMLU-Pro sample, chat format, reasoning low, up to 4,096 tokens)
One request at a time (the megakernel serves one). Results: SESS_EVAL_OUT/<eval>/{items,summary}.json and a
[SERVEEVAL] line per evaluator in the run log.
env: VN_PORT, VN_UTIL, SESS_MAXLEN (server --max-model-len), SESS_EVALS, SESS_EVAL_OUT, SESS_EVAL_N (override n).
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

V = "/root/user/nps1/vllmn"
E = "/root/user/mk1/eval"
GSM_DS = "/root/user/mk1/libAiterMK/results/vllm-evaluation/datasets/gsm8k"
GSM_EVAL = "/root/user/mk1/vllm_aitermk/tests/evals/gsm8k/gsm8k_eval.py"
MMLU = "/root/user/mk1/eval/data/mmlupro/mmlupro_test.jsonl"
PORT = int(os.environ.get("VN_PORT", "8041"))
OUT = os.environ["SESS_EVAL_OUT"]
EVALS = [x for x in os.environ.get("SESS_EVALS", "gsm8k5").split(",") if x]
MAXLEN = os.environ.get("SESS_MAXLEN", "6100")
NOVR = os.environ.get("SESS_EVAL_N", "")

env0 = dict(os.environ)
PLUG = "/root/user/fleet-lanes/vserve/vllm_integration"
env0["PYTHONPATH"] = PLUG + ":" + V + "/ep27" + (":" + env0["PYTHONPATH"] if env0.get("PYTHONPATH") else "")
env0["VLLM_PLUGINS"] = "fleet_mk"
env0["FLEET_MK_NATIVE"] = "1"
env0["FLEET_NATIVE_DEMO"] = os.path.abspath("demo.py")
env0["FLEET_NATIVE_ARGS"] = json.dumps(sys.argv[1:])
cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", env0["MODEL_PATH"],
       "--served-model-name", "gpt-oss-120b", "--load-format", "dummy", "--enforce-eager",
       "--max-model-len", MAXLEN, "--max-num-seqs", "1", "--max-num-batched-tokens", "4096",
       "--no-enable-prefix-caching", "--gpu-memory-utilization", env0.get("VN_UTIL", "0.9"),
       "--port", str(PORT), "--no-async-scheduling",
       "--worker-cls", "fleet_megakernel_vllm.native_worker.FleetNativeWorker"]
base = "http://127.0.0.1:%d" % PORT
os.makedirs(OUT, exist_ok=True)


def log(*x):
    print("[SERVEEVAL]", time.strftime("%H:%M:%S", time.gmtime()), *x, flush=True)


def healthy(t=5):
    try:
        urllib.request.urlopen(base + "/health", timeout=t).read()
        return True
    except Exception:
        return False


class Server:
    def __init__(self):
        log("server:", " ".join(cmd))
        self.p = subprocess.Popen(cmd, env=dict(env0), start_new_session=True)
        t0 = time.time()
        while not healthy(2):
            if self.p.poll() is not None:
                raise SystemExit("server exited with %s" % self.p.returncode)
            if time.time() - t0 > 1500:
                raise SystemExit("server not ready after 1500 s")
            time.sleep(2)
        log("server ready after %.0f s" % (time.time() - t0))

    def stop(self):
        for sig, w in ((signal.SIGTERM, 30), (signal.SIGKILL, 60)):
            try:
                os.killpg(self.p.pid, sig)
                self.p.wait(timeout=w)
                break
            except Exception:
                pass
        time.sleep(5)


def ecmd(ev, d):
    items, summ = os.path.join(d, "items.json"), os.path.join(d, "summary.json")
    py = sys.executable
    if ev == "gsm8k5":
        n = NOVR or "1319"
        return ([py, os.path.join(E, "geval_eval.py"), "--host", "http://127.0.0.1", "--port", str(PORT),
                 "--num-questions", n, "--num-shots", "5", "--max-tokens", "256", "--temperature", "0", "--seed", "42",
                 "--max-concurrency", "1", "--save-results", summ],
                dict(os.environ, GEVAL_EVALUATOR=GSM_EVAL, GEVAL_ITEMS=items, TMPDIR=GSM_DS, GEVAL_TIMEOUT="28800"))
    if ev == "chat":
        return ([py, os.path.join(E, "gchat_eval.py"), "--port", str(PORT), "--n", NOVR or "1319",
                 "--test", os.path.join(GSM_DS, "test.jsonl"), "--items", items, "--summary", summ, "--conc", "1"],
                dict(os.environ))
    if ev == "mmlupro":
        return ([py, os.path.join(E, "mmlupro_eval.py"), "--port", str(PORT), "--n", NOVR or "1400",
                 "--test", MMLU, "--items", items, "--summary", summ, "--conc", "1"], dict(os.environ))
    raise SystemExit("unknown eval " + ev)


srv = Server()
try:
    for ev in EVALS:
        if not healthy():
            log("server unhealthy before %s: restarting" % ev)
            srv.stop()
            srv = Server()
        d = os.path.join(OUT, ev)
        os.makedirs(d, exist_ok=True)
        c, env = ecmd(ev, d)
        log("eval %s: %s" % (ev, " ".join(c)))
        t0 = time.time()
        with open(os.path.join(d, "eval.log"), "w") as f:
            rc = subprocess.call(c, env=env, stdout=f, stderr=subprocess.STDOUT)
        try:
            s = json.load(open(os.path.join(d, "summary.json")))
        except Exception:
            s = {}
        try:
            rows = json.load(open(os.path.join(d, "items.json")))
        except Exception:
            rows = []
        errs = open(os.path.join(d, "eval.log"), errors="replace").read().count("Error calling vLLM") + int(s.get("errors", 0))
        acc = s.get("accuracy")
        log("RESULT %s rc=%d accuracy=%s n=%d errors=%d health_after=%s %.0f s" % (
            ev, rc, acc, len(rows), errs, healthy(10), time.time() - t0))
finally:
    srv.stop()
log("DONE")
