"""serve128.py: the GSM8K-128 questions through `vllm serve` (0.27.1, fleet_mk plugin, native mode) on the megakernel.

Launched by wrap_sess.sh (SESS_SCRIPT=this file) in place of `python demo.py ...`, so the megakernel is built with
exactly the standalone run's argv and environment (the session execs demo.py with --vllm-session). Starts the OpenAI
server with the fleet_mk plugin (FLEET_MK_NATIVE=1, --worker-cls FleetNativeWorker), then sends each question as its
prompt token ids to /v1/completions (streaming, greedy, ignore_eos, SESS_NEW tokens), one at a time.

Per request: TTFT, and decode tok/s = (tokens - 1) / (last chunk - first chunk), as serving benchmarks compute TPOT.
Correctness: FLEET_NATIVE_DUMP makes the session write each request's streamed token ids and the kernel's own token
arena; both are compared with the standalone run's tokens (tok_<SESS_PRE>_<SESS_CFG>_qNNN.json, all saved tokens).
Host stalls: a request whose largest inter-chunk gap exceeds SESS_STALL_MS is sent again; if the rerun shows the
same gap (within 25%) it is a property of the text (special tokens the detokenizer holds back), not a stall, and the
first timing is kept; otherwise the attempt with the smaller gap is kept. Launch hangs (no token for SESS_HANG_S):
the server is restarted and the question retried (a launch that cannot place a scheduler never finishes).
env: SESS_PRE (g128), SESS_CFG (ns | e3 | df8), SESS_N (128), SESS_NEW (256), SESS_OUT (json), VN_PORT, VN_UTIL,
     FLEET_NATIVE_DUMP (dir), SESS_STALL_MS (25), SESS_HANG_S (30).
"""
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

V = "/root/user/nps1/vllmn"
TD = "/root/user/nps1/opt/spec"
PORT = int(os.environ.get("VN_PORT", "8031"))
PRE = os.environ.get("SESS_PRE", "g128")
CFG = os.environ.get("SESS_CFG", "e3")
N = int(os.environ.get("SESS_N", "128"))
NEW = int(os.environ.get("SESS_NEW", "256"))
OUT = os.environ.get("SESS_OUT", "/root/user/nps1/s15/gsm128/serve/%s_%s.json" % (PRE, CFG))
STALL_MS = float(os.environ.get("SESS_STALL_MS", "25"))
HANG_S = float(os.environ.get("SESS_HANG_S", "30"))
DUMP = os.environ.get("FLEET_NATIVE_DUMP", "")


def summarize(res):
    """The kept attempt per question, pooled decode tok/s over the kept attempts that are not host stalls."""
    by_q = {}
    for r in res["requests"]:
        by_q.setdefault(r["q"], []).append(r)
    kept = []
    for q in sorted(by_q):
        att = [r for r in by_q[q] if r.get("tokens")]
        if not att:
            continue
        first = att[0]
        if first["max_gap_ms"] <= STALL_MS:
            k = dict(first, stall=False)
        elif len(att) > 1 and all(abs(a["max_gap_ms"] - first["max_gap_ms"]) <= 0.25 * first["max_gap_ms"] for a in att[1:]):
            k = dict(first, stall=False, gap_reproduced=True)  # deterministic gap: the text, not the host
        else:
            best = min(att, key=lambda a: a["max_gap_ms"])
            k = dict(best, stall=best["max_gap_ms"] > STALL_MS and len(att) > 1 and not any(
                abs(a["max_gap_ms"] - best["max_gap_ms"]) <= 0.25 * best["max_gap_ms"] for a in att if a is not best))
        kept.append(k)
    clean = [r for r in kept if not r["stall"]]
    tok = sum(r["tokens"] - 1 for r in clean)
    sec = sum(r["decode_s"] for r in clean)
    res["kept"] = kept
    res["summary"] = dict(
        questions=len(kept), clean=len(clean), requests=len(res["requests"]), restarts=res.get("restarts", 0),
        gap_reproduced=sum(1 for r in kept if r.get("gap_reproduced")), decode_tok_s=tok / sec if sec else 0.0,
        ttft_ms_mean=1e3 * sum(r["ttft_s"] for r in kept) / len(kept) if kept else 0.0,
        tokens_same=sum(1 for r in kept if r.get("tokens_same")),
        tokens_checked_min=min((r.get("tokens_checked", res.get("new", NEW)) for r in kept), default=0),
        streamed_eq_arena=sum(1 for r in kept if r.get("streamed_eq_arena")),
        text_same=sum(1 for r in kept if r.get("text_same")))
    return res


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--resummarize":
    for p in sys.argv[2:]:
        res = summarize(json.load(open(p)))
        json.dump(res, open(p, "w"), indent=1)
        print(p, res["summary"])
    sys.exit(0)

env0 = dict(os.environ)
PLUG = "/root/user/fleet-lanes/vserve/vllm_integration"
env0["PYTHONPATH"] = PLUG + ":" + V + "/ep27" + (":" + env0["PYTHONPATH"] if env0.get("PYTHONPATH") else "")
env0["VLLM_PLUGINS"] = "fleet_mk"
env0["FLEET_MK_NATIVE"] = "1"
env0["FLEET_NATIVE_DEMO"] = os.path.abspath("demo.py")
env0["FLEET_NATIVE_ARGS"] = json.dumps(sys.argv[1:])
model = env0["MODEL_PATH"]
cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model,
       "--served-model-name", "gpt-oss-120b", "--load-format", "dummy", "--enforce-eager",
       "--max-model-len", "4096", "--max-num-seqs", "1", "--max-num-batched-tokens", "4096",
       "--no-enable-prefix-caching", "--gpu-memory-utilization", env0.get("VN_UTIL", "0.9"),
       "--port", str(PORT), "--no-async-scheduling",
       "--worker-cls", "fleet_megakernel_vllm.native_worker.FleetNativeWorker"]
base = "http://127.0.0.1:%d" % PORT
if DUMP:
    shutil.rmtree(DUMP, ignore_errors=True)
    os.makedirs(DUMP, exist_ok=True)


class Server:
    def __init__(self, k):
        self.k, self.n_req = k, 0
        env = dict(env0)
        if DUMP:
            env["FLEET_NATIVE_DUMP"] = os.path.join(DUMP, "srv%d" % k)
            os.makedirs(env["FLEET_NATIVE_DUMP"], exist_ok=True)
        print("[SERVE128] server %d: %s" % (k, " ".join(cmd)), flush=True)
        self.p = subprocess.Popen(cmd, env=env, start_new_session=True)  # own process group: stop() kills all
        t0 = time.time()
        while True:
            if self.p.poll() is not None:
                raise SystemExit("server exited with %s" % self.p.returncode)
            try:
                urllib.request.urlopen(base + "/health", timeout=2).read()
                break
            except Exception:
                if time.time() - t0 > 1500:
                    raise SystemExit("server not ready after 1500 s")
                time.sleep(2)
        print("[SERVE128] server %d ready after %.0f s" % (k, time.time() - t0), flush=True)

    def stop(self):
        # the engine-core child holds the GPU (and, after a launch hang, the stuck kernel): end the whole group
        try:
            os.killpg(self.p.pid, signal.SIGTERM)
            self.p.wait(timeout=30)
        except Exception:
            pass
        try:
            os.killpg(self.p.pid, signal.SIGKILL)
        except Exception:
            pass
        try:
            self.p.wait(timeout=60)
        except Exception:
            pass
        time.sleep(5)  # let the driver tear down the dead process's GPU context before the next server starts


def load(q):
    d = json.load(open(os.path.join(TD, "tok_%s_%s_q%03d.json" % (PRE, CFG, q))))
    t = d["token_ids"]
    t = t[0] if t and isinstance(t[0], list) else t
    return [int(x) for x in d["prompt_ids"]], [int(x) for x in t]


def request(prompt, n):
    body = json.dumps({"model": "gpt-oss-120b", "prompt": prompt, "max_tokens": n, "temperature": 0.0,
                       "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(base + "/v1/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    stamps, text, usage = [], [], None
    with urllib.request.urlopen(req, timeout=HANG_S) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                if ch.get("text"):
                    stamps.append(time.perf_counter())
                    text.append(ch["text"])
    if not stamps or not usage:
        return None
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    return dict(tokens=usage["completion_tokens"], chunks=len(stamps), ttft_s=stamps[0] - t0,
                decode_s=stamps[-1] - stamps[0], max_gap_ms=1e3 * max(gaps) if gaps else 0.0, text="".join(text))


res = {"pre": PRE, "cfg": CFG, "n": N, "new": NEW, "requests": [], "restarts": 0}
srv = Server(0)
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)

    def one(q, attempt):
        global srv
        prompt, ref = load(q)
        for tries in range(3):
            try:
                r = request(prompt, NEW)
            except (socket.timeout, TimeoutError, OSError) as e:
                r = None
                print("[SERVE128] q%03d: no token for %.0f s (%s): restarting the server" % (q, HANG_S, type(e).__name__),
                      flush=True)
            if r is not None:
                break
            srv.stop()
            res["restarts"] += 1
            srv = Server(res["restarts"])
        if r is None:
            raise SystemExit("q%03d failed 3 times" % q)
        nc = min(len(ref), NEW)
        ref_text = tok.decode(ref[:nc], skip_special_tokens=True)
        r.update(q=q, attempt=attempt, prompt_len=len(prompt), srv=srv.k, idx=srv.n_req, tokens_checked=nc,
                 text_same=(r["text"] == ref_text) if nc == NEW else r["text"].startswith(ref_text[:-8]))
        srv.n_req += 1
        del r["text"]
        res["requests"].append(r)
        print("[SERVE128] q%03d try%d prompt %d tokens %s ttft %.1f ms decode %.0f tok/s max gap %.1f ms text_same %s"
              % (q, attempt, len(prompt), r["tokens"], 1e3 * r["ttft_s"],
                 (r["tokens"] - 1) / r["decode_s"] if r["decode_s"] > 0 else 0, r["max_gap_ms"], r["text_same"]),
              flush=True)
        return r

    for q in range(N):
        r = one(q, 0)
        a = 1
        while r["max_gap_ms"] > STALL_MS and a < 3:  # a host stall, or a gap the text produces (reproduces)?
            r2 = one(q, a)
            a += 1
            if abs(r2["max_gap_ms"] - r["max_gap_ms"]) <= 0.25 * r["max_gap_ms"] or r2["max_gap_ms"] <= STALL_MS:
                break
            r = r2
    if DUMP:  # one more (1-token) request makes the session dump the last real one
        body = json.dumps({"model": "gpt-oss-120b", "prompt": load(0)[0], "max_tokens": 1, "temperature": 0.0}).encode()
        urllib.request.urlopen(urllib.request.Request(base + "/v1/completions", data=body,
                                                      headers={"Content-Type": "application/json"}), timeout=600).read()
        time.sleep(1.0)
        for r in res["requests"]:
            p = os.path.join(DUMP, "srv%d" % r["srv"], "req%d.json" % r["idx"])
            if not os.path.exists(p):
                r["tokens_same"] = None
                continue
            d = json.load(open(p))
            ref = load(r["q"])[1][:r["tokens_checked"]]
            s, a_ = d["streamed"][:NEW], d["arena"][:NEW]
            r["streamed_eq_arena"] = s == a_
            r["tokens_same"] = s[:len(ref)] == ref
            if not r["tokens_same"]:
                r["first_token_diff"] = next((j for j in range(min(len(s), len(ref))) if s[j] != ref[j]), len(ref))
finally:
    srv.stop()

summarize(res)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)
s = res["summary"]
print("[SERVE128] SUMMARY %s %s: %d questions (%d clean, %d requests, %d server restarts, %d gaps reproduced = text): "
      "decode %.0f tok/s pooled, TTFT %.1f ms mean; tokens identical to standalone %d/%d (>= %d tokens checked each), "
      "streamed == kernel arena %d/%d, text identical %d/%d -> %s"
      % (PRE, CFG, s["questions"], s["clean"], s["requests"], s["restarts"], s["gap_reproduced"], s["decode_tok_s"],
         s["ttft_ms_mean"], s["tokens_same"], s["questions"], s["tokens_checked_min"], s["streamed_eq_arena"],
         s["questions"], s["text_same"], s["questions"], OUT), flush=True)

