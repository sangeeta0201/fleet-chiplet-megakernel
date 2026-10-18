"""Derive serve128_aitermk.py from serve128.py: same client, timing, stall handling and checks; only the server
changes (libAiterMK fleet_vllm plugin, native fleet MPK provider, --worker-cls fleet_vllm.native.worker.FleetMPKWorker).

usage: mk_serve_aitermk.py SRC_SERVE128_PY OUT_PY
"""
import sys

src, out = sys.argv[1], sys.argv[2]
s = open(src).read()
a = s.index("env0 = dict(os.environ)\n")
b = s.index('base = "http://127.0.0.1:%d" % PORT\n')
new = '''env0 = dict(os.environ)
# libAiterMK path: the fleet_vllm plugin (installed from libAiterMK/engine_adapters/vllm) in native fleet MPK
# provider mode. The provider DSO and its exported state come from AITERMK_FLEET_MPK_PROVIDER / _STATE.
env0["AITER_MK_FLEET_VLLM"] = "1"
env0["VLLM_PLUGINS"] = "fleet_mk"
for k in ("FLEET_MK_NATIVE", "FLEET_NATIVE_DEMO", "FLEET_NATIVE_ARGS"):
    env0.pop(k, None)
for k in ("AITERMK_FLEET_MPK_PROVIDER", "AITERMK_FLEET_MPK_STATE"):
    if not env0.get(k):
        raise SystemExit("set " + k)
model = env0.get("MODEL_PATH", "/root/user/models/gpt-oss-120b")
cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model,
       "--served-model-name", "gpt-oss-120b", "--load-format", "dummy", "--enforce-eager",
       "--max-model-len", "4096", "--max-num-seqs", "1", "--max-num-batched-tokens", "4096",
       "--no-enable-prefix-caching", "--gpu-memory-utilization", env0.get("VN_UTIL", "0.9"),
       "--port", str(PORT), "--no-async-scheduling",
       "--worker-cls", "fleet_vllm.native.worker.FleetMPKWorker"]
'''
s = s[:a] + new + s[b:]
s = s.replace('"""serve128.py: the GSM8K-128 questions through `vllm serve` (0.27.1, fleet_mk plugin, native mode) on the megakernel.',
              '"""serve128_aitermk.py: serve128.py through libAiterMK (fleet_vllm plugin, native fleet MPK provider DSO).\n\n'
              'Generated from serve128.py by mk_serve_aitermk.py; only the server block differs.\n\n'
              'Original docstring: the GSM8K-128 questions through `vllm serve` (0.27.1, fleet_mk plugin, native mode) on the megakernel.', 1)
s = s.replace("[SERVE128]", "[SERVE128AK]")
open(out, "w").write(s)
print("wrote", out)

