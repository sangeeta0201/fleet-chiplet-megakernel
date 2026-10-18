"""In-process driver for the fleet megakernel as demo.py builds it (any partition mode, incl. MPK_MULTI_DEV).

NativeSession execs demo.py with its normal argv plus --vllm-session: the megakernel is built and compiled
exactly as in a standalone run, then demo.py hands over its live objects instead of running. The compiled
kernel holds raw pointers into those tensors, so the session keeps demo.py's namespace alive.

Each request is one free-running launch on a background thread, on a stream of its own (only that stream
waits for the kernel). Built with MPK_TOK_HOST=1, the scheduler publishes request 0's position and token id
into pinned host memory every iteration, so tokens stream to the host without any GPU call.
"""
import json
import os
import sys
import threading
import time

import torch


class NativeSession:
    def __init__(self, demo_path, demo_args):
        demo_path = os.path.abspath(demo_path)
        demo_dir = os.path.dirname(demo_path)
        self.namespace = {"__name__": "__main__", "__file__": demo_path,
                          "__builtins__": __builtins__}
        old_argv, old_cwd = sys.argv, os.getcwd()
        sys.argv = [demo_path] + list(demo_args) + ["--vllm-session"]
        if demo_dir not in sys.path:
            sys.path.insert(0, demo_dir)
        os.chdir(demo_dir)
        session = None
        try:
            with open(demo_path) as f:
                code = compile(f.read(), demo_path, "exec")
            exec(code, self.namespace)
        except Exception as e:
            if type(e).__name__ != "MPKSessionReady":
                raise
            session = e.args[0]
        finally:
            sys.argv = old_argv
            os.chdir(old_cwd)
        if session is None:
            raise RuntimeError("demo.py finished without reaching the --vllm-session point")
        self.parts = session
        self.mpk = session["mpk"]
        self.tokenizer = session["tokenizer"]
        self.tokens = session["tokens"]
        self.prompt_lengths = session["prompt_lengths"]
        self.step = session["step"]
        self.num_new_tokens = session["num_new_tokens"]
        self.reset_device_barriers = session["reset_device_barriers"]
        self.max_seq = self.tokens.size(1)
        for name in ("decode_progress_func", "decode_token_func", "reset_decode_progress_func"):
            if getattr(self.mpk, name, None) is None:
                raise RuntimeError("the megakernel module lacks %s" % name)
        if self.mpk.decode_token_func(0) == -1 and os.environ.get("MPK_TOK_HOST", "0") != "1":
            raise RuntimeError("build the megakernel with MPK_TOK_HOST=1")
        self.launch_stream = torch.cuda.Stream()
        self.thread = None
        self.error = None
        self.prompt_len = 0
        self.n_expected = 0
        self.emitted = 0
        self.t_start = 0.0
        self.dump_dir = os.environ.get("FLEET_NATIVE_DUMP")
        self.n_requests = 0
        self.streamed = []

    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    def wait_idle(self):
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        if self.error is not None:
            err, self.error = self.error, None
            raise err

    def start(self, prompt_ids, max_tokens):
        """Begin one request (serve_mpk's per-request setup, then a background launch)."""
        self.wait_idle()
        if self.dump_dir and self.n_requests > 0:
            # previous request: what was streamed vs what the kernel wrote (read before the arena reset)
            end = self.prompt_len + self.n_expected
            arena = self.tokens[0, self.prompt_len:end].cpu().tolist()
            with open(os.path.join(self.dump_dir, "req%d.json" % (self.n_requests - 1)), "w") as f:
                json.dump(dict(streamed=self.streamed, arena=arena), f)
        n = len(prompt_ids)
        need = n + max_tokens
        if need > self.max_seq:
            raise ValueError("prompt %d + max_tokens %d exceeds the compiled arena of %d"
                             % (n, max_tokens, self.max_seq))
        mpk = self.mpk
        if getattr(mpk, "set_max_seq_length_func", None) is not None:
            mpk.set_max_seq_length_func(need + 1)
        t = self.tokens
        t.zero_()
        t[0, :n] = torch.tensor(prompt_ids, dtype=torch.long, device=t.device)
        for r in range(1, t.size(0)):
            t[r, :n] = t[0, :n]
        self.prompt_lengths.fill_(n)
        self.step.zero_()
        self.num_new_tokens.fill_(1)
        mpk.init_request_func()
        self.reset_device_barriers()
        mpk.reset_decode_progress_func()
        torch.cuda.synchronize()
        self.prompt_len, self.n_expected, self.emitted = n, max_tokens, 0
        self.error = None
        self.streamed = []
        self.n_requests += 1

        def run():
            try:
                mpk(default_stream=self.launch_stream)
                self.launch_stream.synchronize()
            except BaseException as e:  # surfaced by poll() / wait_idle()
                self.error = e

        self.thread = threading.Thread(target=run, name="mpk-launch", daemon=True)
        self.t_start = time.perf_counter()
        self.thread.start()

    def _available(self):
        pos = self.mpk.decode_progress_func(0)
        if pos < self.prompt_len:
            return 0
        return min(pos - self.prompt_len + 1, self.n_expected)

    def poll(self, timeout=None, sleep_s=20e-6):
        """Wait until at least one new token exists (or the launch ended); return the new token ids."""
        deadline = None if timeout is None else time.perf_counter() + timeout
        while True:
            avail = self._available()
            if avail > self.emitted:
                break
            if not self.busy():
                avail = self._available()
                if avail > self.emitted:
                    break
                if self.error is not None:
                    self.wait_idle()
                return []
            if deadline is not None and time.perf_counter() > deadline:
                return []
            time.sleep(sleep_s)
        base = self.prompt_len + self.emitted
        ids = [self.mpk.decode_token_func(base + i) for i in range(avail - self.emitted)]
        if any(i < 0 for i in ids):
            raise RuntimeError("position published without its token id: %s" % ids)
        self.emitted = avail
        self.streamed += ids
        return ids

    def stop(self):
        """End the running launch early (build with MPK_HOST_STOP=1). It stops within 16 iterations."""
        fn = getattr(self.mpk, "request_stop_func", None)
        if fn is not None and self.busy():
            fn()

    def done(self):
        return self.emitted >= self.n_expected

