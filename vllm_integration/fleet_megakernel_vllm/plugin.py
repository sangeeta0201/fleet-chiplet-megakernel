"""vLLM general plugin entry point for fleet_mk.

vLLM calls register() once at engine startup (discovered via the
`vllm.general_plugins` entry point in pyproject.toml). We override each stock
causal-LM fleet_mk supports with a fleet_mk subclass that keeps vLLM's prefill path but
routes decode through fleet_mk's fused megakernel. vLLM instantiates whichever
architecture matches the loaded model, so registering both is harmless when only
one model is served.
"""


def register():
    import os

    # FLEET_MK_INERT_GATE: inert unless explicitly enabled, so a vLLM server that merely has this package
    # installed (vLLM loads all general_plugins by default) behaves exactly as stock for every other model.
    if os.environ.get("FLEET_MK", "0") != "1" and os.environ.get("FLEET_MK_NATIVE", "0") != "1":
        return

    # FLEET_MK_NATIVE=1: the native persistent megakernel (demo.py's build, incl. DPX/CPX) instead of
    # the Fleet MK wrapper kernel; see native.py.
    if os.environ.get("FLEET_MK_NATIVE", "0") == "1":
        from .native import register as register_native
        register_native()
        return

    from vllm import ModelRegistry

    from .greedy import install_greedy_argmax_fastpath
    from .model import FleetMKQwen3ForCausalLM, _make_gptoss_class
    from .persistent_scheduler import install_persistent_scheduler_hooks

    # Inactive unless compute_logits returns a tensor tagged with Fleet's
    # device-side argmax (FLEET_MK_GREEDY_ARGMAX=1).  Keeping the hook installed
    # lets unsupported/random requests fall through to vLLM unchanged.
    install_greedy_argmax_fastpath()
    install_persistent_scheduler_hooks()

    # vLLM 0.26+ moved the stock ROCm aiter KV pool to a packed layout that
    # fleet_mk's flat view cannot alias. Registering FleetMKAttentionBackend restores
    # the split layout; it must happen before any Attention layer is built, and
    # the plugin's register() is the earliest hook vLLM offers. Registration
    # alone does not *select* it -- the engine must also be launched with
    # attention_backend=CUSTOM (see harness.py). Absent on 0.11.x, where the
    # stock aiter backends already allocate fleet_mk's layout.
    try:
        from .backend import register_fleet_mk_backend
        register_fleet_mk_backend()
    except ImportError as e:  # pragma: no cover - depends on installed vLLM
        import logging
        logging.getLogger(__name__).info(
            "fleet_mk: CUSTOM attention backend not registered (%s); this vLLM "
            "predates the registry and should already use fleet_mk's KV layout.",
            e)

    ModelRegistry.register_model("Qwen3ForCausalLM", FleetMKQwen3ForCausalLM)

    # GPT-OSS lives in a separate vLLM module that may be absent in older builds;
    # only register it if importable so the dense path never breaks.
    try:
        ModelRegistry.register_model("GptOssForCausalLM", _make_gptoss_class())
    except Exception as e:  # pragma: no cover - depends on installed vLLM
        import logging
        logging.getLogger(__name__).info(
            "fleet_mk: GptOssForCausalLM not registered (%s); dense path unaffected.",
            e)
