"""Export a compiled megakernel as a libAiterMK provider (MPK_PROVIDER_EXPORT=<dir>; inert when unset).

Two halves:

* export_source(): PersistentKernel.compile() calls it with the generated code. It writes <dir>/src: the kernel
  source with every fixed device address replaced by a slot of a pointer table that the provider's init
  receives, the task graph compiled into the library, a plain C ABI in place of the Python module, and this
  build's compile command, init arguments and MPK_* environment.
* dump_state(): demo.py calls it where --vllm-session hands over the live objects (after compile and init,
  before any launch). It writes <dir>/state: every device allocation the kernel's pointers land in (torch
  caching-allocator segments and the HIP VMM regions of placement.halves_malloc), byte for byte, and where each
  pointer, metadata buffer, RoPE table, per-request reset buffer and drafter pointer sits in them. The libAiterMK
  adapter rebuilds the same allocations on the same partitions, loads the bytes and calls the provider's init
  with the new addresses.
"""
import ctypes as ct
import json
import os
import re
import time

ABI_VERSION = 1
META_NAMES = ["step", "tokens", "input_tokens", "output_tokens", "num_new_tokens", "prompt_lengths",
              "qo_indptr_buffer", "paged_kv_indptr_buffer", "paged_kv_indices_buffer",
              "paged_kv_last_page_len_buffer"]
ENV_PTR_KEYS = ("MPK_EAGLE3_PTRS", "MPK_DFLASH_PTRS")
RESET_NAMES = ("oproj_topk_counters", "qkv_attn_barrier", "moe_fused_barrier", "router_topk_counter")

_LIT = re.compile(r"\(char\*\)\((0x[0-9a-fA-F]+)\)")
_NAMED = re.compile(r"char \*([A-Za-z_][A-Za-z0-9_]*) = \(char\*\)\((0x[0-9a-fA-F]+)\);")
_TG_LOAD = re.compile(
    r"std::filesystem::path file_path\(__FILE__\);\s*"
    r"std::ifstream json_file\(file_path\.parent_path\(\)\.string\(\)\s*\+\s*\"/task_graph\.json\"\);\s*"
    r"nlohmann::json json_task_graph;\s*"
    r"json_file >> json_task_graph;")
_VA = re.compile(r"\b0x7[0-9a-fA-F]{11}\b")

_PROLOGUE = """\
// libAiterMK fleet MPK provider source, generated from this build's test.cu by mirage.mpk.provider_export.
// Device addresses come from the pointer table fleet_mpk_init() receives; the task graph is compiled in.
static void *const *mpk_provider_ptrs = nullptr;
static const char *mpk_provider_task_graph_json(void);
"""

_ABI = """
// ------------------------------------------------------------------------------------------------------------
// libAiterMK fleet MPK provider ABI (in place of the Python module of a standalone build)
// ------------------------------------------------------------------------------------------------------------
#include <hip/hip_runtime.h>
#define FLEET_MPK_API extern "C" __attribute__((visibility("default")))

static const char *mpk_provider_task_graph_json(void) {
  return reinterpret_cast<const char *>(fleet_mpk_task_graph);
}

FLEET_MPK_API int fleet_mpk_abi_version(void) { return FLEET_MPK_ABI_VERSION; }
FLEET_MPK_API const char *fleet_mpk_build_id(void) { return FLEET_MPK_BUILD_ID; }
FLEET_MPK_API int fleet_mpk_num_ptrs(void) { return FLEET_MPK_NUM_PTRS; }
FLEET_MPK_API const char *fleet_mpk_ptr_name(int i) {
  return (i >= 0 && i < FLEET_MPK_NUM_PTRS) ? fleet_mpk_ptr_names[i] : nullptr;
}

// ptrs: one device address per generated pointer (fleet_mpk_ptr_name order); read only during this call.
// meta: the 10 runtime metadata buffers (step, tokens, input_tokens, output_tokens, num_new_tokens,
// prompt_lengths, qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len).
FLEET_MPK_API int fleet_mpk_init(void *const *ptrs, int num_ptrs, void *const *meta, int num_meta,
                                 void *profiler_buffer, int my_rank, int num_workers,
                                 int num_local_schedulers, int num_remote_schedulers, int max_seq_length,
                                 int total_num_requests, long long eos_token_id) {
  if (ptrs == nullptr || num_ptrs != FLEET_MPK_NUM_PTRS || meta == nullptr || num_meta != 10) {
    return -1;
  }
  mpk_provider_ptrs = ptrs;
  std::vector<void *> meta_tensors(meta, meta + num_meta);
  init_persistent_kernel(meta_tensors, profiler_buffer, my_rank, num_workers, num_local_schedulers,
                         num_remote_schedulers, max_seq_length, total_num_requests, eos_token_id);
  mpk_provider_ptrs = nullptr;
  return 0;
}
FLEET_MPK_API void fleet_mpk_set_rope_tables(void *cos_ptr, void *sin_ptr) { set_rope_tables(cos_ptr, sin_ptr); }
FLEET_MPK_API void fleet_mpk_set_max_seq_length(int max_seq_length) { set_max_seq_length(max_seq_length); }
FLEET_MPK_API void fleet_mpk_init_request(void) { init_request_resources(); }
FLEET_MPK_API void fleet_mpk_reset_decode_progress(void) { reset_decode_progress(); }
// End the running launch at its next 16-iteration check (builds with MPK_HOST_STOP; a no-op otherwise).
FLEET_MPK_API void fleet_mpk_request_stop(void) { request_decode_stop(); }
// Blocks until the request's launch ends; run it on a thread of its own.
FLEET_MPK_API void fleet_mpk_launch(void *stream) { launch_persistent_kernel(reinterpret_cast<hipStream_t>(stream)); }
FLEET_MPK_API int fleet_mpk_decode_progress(int slot) { return get_decode_progress(slot); }
FLEET_MPK_API int fleet_mpk_decode_token(int pos) { return get_decode_token(pos); }
FLEET_MPK_API void fleet_mpk_finalize(void) { finalize_persistent_kernel(); }
"""


def enabled():
    return bool(os.environ.get("MPK_PROVIDER_EXPORT"))


def _root():
    return os.environ["MPK_PROVIDER_EXPORT"]


def _c_str(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _blob(data):
    rows = []
    for i in range(0, len(data), 24):
        rows.append(",".join("0x%02x" % b for b in data[i:i + 24]))
    return ("static const unsigned char fleet_mpk_task_graph[%d] = {\n%s,0x00};\n"
            % (len(data) + 1, ",\n".join(rows)))


def export_source(cuda_code, graph_json, cc_cmd, env=None):
    """Write <root>/src. Returns the rewritten source (the standalone build is not affected)."""
    d = os.path.join(_root(), "src")
    os.makedirs(d, exist_ok=True)
    named = _NAMED.findall(cuda_code)
    addrs = []

    def slot(m):
        addrs.append(m.group(1))
        return "(char*)(mpk_provider_ptrs[%d])" % (len(addrs) - 1)

    code = _LIT.sub(slot, cuda_code)
    if [a for _, a in named] == addrs:
        names = [n for n, _ in named]
    else:
        names = ["ptr_%d" % i for i in range(len(addrs))]
    code, k = _TG_LOAD.subn(
        "nlohmann::json json_task_graph = nlohmann::json::parse(mpk_provider_task_graph_json());", code)
    if k != 1:
        raise RuntimeError("MPK_PROVIDER_EXPORT: task graph load not found (%d matches)" % k)
    left = sorted(set(_VA.findall(code)))
    if left:
        raise RuntimeError("MPK_PROVIDER_EXPORT: device addresses left in the source: %s" % left[:8])
    build_id = "fleet-mpk-%s" % time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    tail = ["", "#define FLEET_MPK_ABI_VERSION %d" % ABI_VERSION,
            "#define FLEET_MPK_NUM_PTRS %d" % len(addrs),
            "#define FLEET_MPK_BUILD_ID %s" % _c_str(build_id),
            "static const char *const fleet_mpk_ptr_names[FLEET_MPK_NUM_PTRS] = {",
            ",\n".join("  " + _c_str(n) for n in names), "};",
            '#include "task_graph_blob.inc"']
    kernel = _PROLOGUE + code + "\n".join(tail) + "\n" + _ABI
    with open(os.path.join(d, "kernel.hip"), "w") as f:
        f.write(kernel)
    with open(os.path.join(d, "task_graph_blob.inc"), "w") as f:
        f.write(_blob(graph_json.encode()))
    with open(os.path.join(d, "task_graph.json"), "w") as f:
        f.write(graph_json)
    with open(os.path.join(d, "ptr_table.json"), "w") as f:
        json.dump([dict(idx=i, name=n, addr=a) for i, (n, a) in enumerate(zip(names, addrs))], f, indent=0)
    with open(os.path.join(d, "compile_cmd.json"), "w") as f:
        json.dump([str(c) for c in cc_cmd], f, indent=0)
    env = dict(os.environ) if env is None else env
    with open(os.path.join(d, "env.json"), "w") as f:
        json.dump({k: v for k, v in sorted(env.items()) if k.startswith("MPK_")}, f, indent=0)
    with open(os.path.join(d, "provider.json"), "w") as f:
        json.dump(dict(abi_version=ABI_VERSION, build_id=build_id, num_ptrs=len(addrs)), f, indent=1)
    print("[PROVIDER] source: %d pointers -> %s (%s)" % (len(addrs), d, build_id), flush=True)
    return kernel


def export_init_args(**kw):
    d = os.path.join(_root(), "src")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "init_args.json"), "w") as f:
        json.dump(kw, f, indent=1)


# --------------------------------------------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------------------------------------------
def _hip():
    from . import placement
    hip = placement._VMM.get("hip") or ct.CDLL("libamdhip64.so")
    placement._VMM["hip"] = hip
    return hip


def _tensor_info(t):
    return dict(addr=t.data_ptr(), nbytes=t.numel() * t.element_size(), dtype=str(t.dtype).replace("torch.", ""),
                shape=list(t.shape), stride=list(t.stride()), device=t.device.index)


def dump_state(mpk, tensor_refs=None):
    """Write <root>/state (see the module docstring). Call after compile(), before any launch."""
    if os.environ.get("MPK_PROVIDER_EXPORT_NODATA") == "1":
        print("[PROVIDER] MPK_PROVIDER_EXPORT_NODATA: source export only, no state dump", flush=True)
        return
    import torch

    from . import placement

    root = _root()
    sd = os.path.join(root, "state")
    os.makedirs(sd, exist_ok=True)
    torch.cuda.synchronize()
    for dev in range(torch.cuda.device_count()):
        with torch.cuda.device(dev):
            torch.cuda.synchronize()
    ptrs = json.load(open(os.path.join(root, "src", "ptr_table.json")))

    allocs = []
    for a in placement._VMM.get("allocs", []):
        allocs.append(dict(kind=a["kind"], base=a["va"], size=a["nbytes"],
                           **{k: a[k] for k in ("domains", "swap", "mdev", "device") if k in a}))
    for seg in torch.cuda.memory_snapshot():
        allocs.append(dict(kind="torch", base=seg["address"], size=seg["total_size"], device=seg["device"]))

    def where(addr, what):
        for i, a in enumerate(allocs):
            if a["base"] <= addr < a["base"] + a["size"]:
                a["used"] = True
                return i, addr - a["base"]
        raise RuntimeError("MPK_PROVIDER_EXPORT: %s at 0x%x is in no known device allocation" % (what, addr))

    man = dict(abi_version=ABI_VERSION, allocations=allocs, ptrs=[], meta=[], rope={}, reset=[], env_ptrs={})
    for p in ptrs:
        i, off = where(int(p["addr"], 16), p["name"])
        man["ptrs"].append(dict(idx=p["idx"], name=p["name"], alloc=i, offset=off))
    for n in META_NAMES:
        t = mpk.meta_tensors[n]
        info = _tensor_info(t)
        info["alloc"], info["offset"] = where(info["addr"], "meta " + n)
        man["meta"].append(dict(name=n, **info))
    for key, t in zip(("cos", "sin"), getattr(mpk, "_rope_tensors", ()) or ()):
        info = _tensor_info(t)
        info["alloc"], info["offset"] = where(info["addr"], "rope " + key)
        man["rope"][key] = info
    for n in RESET_NAMES:
        t = (tensor_refs or {}).get(n)
        if t is not None:
            info = _tensor_info(t)
            info["alloc"], info["offset"] = where(info["addr"], "reset " + n)
            man["reset"].append(dict(name=n, **info))
    for k in ENV_PTR_KEYS:
        v = os.environ.get(k, "")
        if v:
            man["env_ptrs"][k] = [list(where(int(x, 16), "%s[%d]" % (k, j)))
                                  for j, x in enumerate(v.split(",")) if x]

    hip = _hip()
    chunk = 256 << 20
    host = torch.empty(chunk, dtype=torch.uint8, pin_memory=True)
    hp = host.data_ptr()
    t0, total = time.time(), 0
    for i, a in enumerate(allocs):
        if not a.get("used"):
            continue
        a["file"] = "alloc_%03d.bin" % i
        with open(os.path.join(sd, a["file"]), "wb") as f:
            done = 0
            while done < a["size"]:
                n = min(chunk, a["size"] - done)
                rc = hip.hipMemcpy(ct.c_void_p(hp), ct.c_void_p(a["base"] + done), ct.c_size_t(n), 2)
                if rc != 0:
                    raise RuntimeError("MPK_PROVIDER_EXPORT: hipMemcpy D2H of allocation %d failed: %d" % (i, rc))
                f.write(memoryview(host.numpy())[:n])
                done += n
        total += a["size"]
    man["allocations"] = [dict(a, id=i) for i, a in enumerate(allocs) if a.get("used")]
    for a in man["allocations"]:
        a.pop("used", None)
    # self-contained: the provider's init arguments, runtime environment and build id travel with the state
    src = os.path.join(root, "src")
    man["init_args"] = json.load(open(os.path.join(src, "init_args.json")))
    env = json.load(open(os.path.join(src, "env.json")))
    man["env"] = {k: v for k, v in env.items() if k not in ("MPK_PROVIDER_EXPORT", "MPK_WRAP")}
    man["build_id"] = json.load(open(os.path.join(src, "provider.json")))["build_id"]
    with open(os.path.join(sd, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    print("[PROVIDER] state: %d allocations, %.2f GiB in %.0f s -> %s"
          % (len(man["allocations"]), total / 2**30, time.time() - t0, sd), flush=True)

