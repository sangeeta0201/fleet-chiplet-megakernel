"""GLM-5 / GLM-4.7-Flash end-to-end correctness: Torch reference vs Mirage (MPK).

Same vLLM-style tolerant comparison as test_gpt_oss_inference_output.py.
Greedy (temperature=0) decoding from the two paths agrees substantially early
and may diverge later as per-step numerical drift accumulates (mirage decode is
NOT bit-deterministic, and the Torch reference does its MLA absorption in fp32
while the megakernel stays in bf16). So instead of an exact token match
(brittle), we gate on the LONGEST COMMON CONTIGUOUS token BLOCK (via difflib),
which is robust to a single early insertion/deletion that would otherwise shift
every later position and tank a naive prefix count.

Produce the inputs first (from repo root):
    python3 demo/glm5/demo.py --model-path <m> --save-tokens               # torch
    python3 demo/glm5/demo.py --model-path <m> --use-mirage --save-tokens  # mirage

Tunables (env):
    GLM5_OUTPUT_DIR   dir holding the two json dumps (default outputs/glm5)
    GLM5_MIN_MATCH    min longest common contiguous block (default 8)
    GLM5_NUM_COMPARE  window size to compare over           (default 50)
"""
import json
import os
import difflib
import pytest

DEFAULT_OUTPUT_DIR = os.environ.get(
    "GLM5_OUTPUT_DIR", os.path.join("outputs", "glm5")
)
TORCH_OUTPUT = os.path.join(DEFAULT_OUTPUT_DIR, "torch_output.json")
MPK_OUTPUT = os.path.join(DEFAULT_OUTPUT_DIR, "mpk_output.json")

MIN_MATCH = int(os.environ.get("GLM5_MIN_MATCH", "8"))
NUM_TOKENS_TO_COMPARE = int(os.environ.get("GLM5_NUM_COMPARE", "50"))


def _load_tokens(path):
    if not os.path.exists(path):
        pytest.fail(
            f"Missing output file: {path}. Run demo/glm5/demo.py with "
            f"--save-tokens (and --use-mirage for the MPK dump) first."
        )
    with open(path) as f:
        data = json.load(f)
    tokens = data.get("token_ids")
    if not isinstance(tokens, list):
        pytest.fail(f"'token_ids' missing or not a list in {path}")
    return tokens, data


def test_glm5_torch_vs_mpk_tokens():
    torch_tokens, torch_meta = _load_tokens(TORCH_OUTPUT)
    mpk_tokens, mpk_meta = _load_tokens(MPK_OUTPUT)

    n = min(NUM_TOKENS_TO_COMPARE, len(torch_tokens), len(mpk_tokens))
    if n == 0:
        pytest.fail(
            f"No tokens to compare (torch={len(torch_tokens)}, "
            f"mpk={len(mpk_tokens)})"
        )

    a, b = torch_tokens[:n], mpk_tokens[:n]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    longest_block = max((m.size for m in sm.get_matching_blocks()), default=0)
    ratio = sm.ratio()

    print(
        f"[glm5 correctness] compared {n} tokens: "
        f"longest_common_block={longest_block} (need >= {MIN_MATCH}), "
        f"difflib_ratio={ratio:.2%}, "
        f"torch_gen_len={torch_meta.get('generate_length')}, "
        f"mpk_gen_len={mpk_meta.get('generate_length')}"
    )

    if longest_block < MIN_MATCH:
        pytest.fail(
            f"Longest common contiguous block too short: {longest_block} tokens "
            f"(require >= {MIN_MATCH}); difflib_ratio={ratio:.2%}. The two greedy "
            f"generations do not substantially agree even after realignment -- "
            f"likely a real correctness regression. "
            f"torch[:{n}]={a} | mpk[:{n}]={b}"
        )
