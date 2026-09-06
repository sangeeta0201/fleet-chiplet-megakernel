"""GLM GSM8K gate: Redline/ATOM flexible-extract + cross-rank identity.

demo/glm5/run_gsm8k.sh writes one JSON per rank (gsm8k_rank<r>.json). Exact
generated tokens are not a legal gate on this model (demo/glm5/correctness_gate.py);
this scores extracted numeric answers instead, which is what Redline reports
for GPT-OSS as exact_match,flexible-extract.

HARD GATES
  G1  CROSS-RANK IDENTITY. Every rank must extract the same answer for every
      sample. Same defect class as correctness_gate.py G1 / PPL P2.
  G2  COMPLETIONS EXIST. Every sample produced at least one generated token
      and a dump with total > 0.
  G3  ACCURACY FLOOR (optional). GLM5_GSM8K_MIN_ACC, default 0 -- bring-up
      reports the number; raise it once a slice is measured.
"""
import glob
import json
import os
import sys
import pytest

DEFAULT_OUTPUT_DIR = os.environ.get(
    "GLM5_OUTPUT_DIR", os.path.join("outputs", "glm5")
)
MIN_ACC = float(os.environ.get("GLM5_GSM8K_MIN_ACC", "0"))

_GLM5_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "demo", "glm5"))
if _GLM5_DIR not in sys.path:
    sys.path.insert(0, _GLM5_DIR)
from gsm8k import extract_flexible, gold_answer  # noqa: E402


def test_gsm8k_flexible_extract_matches_lmeval():
    assert extract_flexible("The answer is 18.") == "18"
    assert extract_flexible("blah #### 1,234") == "1234"
    assert extract_flexible("no numbers") is None
    assert gold_answer("reason\n#### 42") == "42"


def _dumps():
    paths = sorted(glob.glob(os.path.join(DEFAULT_OUTPUT_DIR, "gsm8k_rank*.json")))
    plain = os.path.join(DEFAULT_OUTPUT_DIR, "gsm8k.json")
    if os.path.isfile(plain):
        paths = [plain] + [p for p in paths if p != plain]
    return paths


def test_glm5_gsm8k_redline_protocol():
    paths = _dumps()
    if not paths:
        if os.environ.get("GLM5_GSM8K_REQUIRE_DUMPS"):
            pytest.fail(
                f"No GSM8K dumps in {DEFAULT_OUTPUT_DIR}. "
                f"See demo/glm5/run_gsm8k.sh."
            )
        pytest.skip(
            f"No GSM8K dumps in {DEFAULT_OUTPUT_DIR}. Run demo/glm5/run_gsm8k.sh."
        )
    dumps = []
    for p in paths:
        with open(p) as f:
            dumps.append((p, json.load(f)))
    ref_path, ref = dumps[0]
    summary = ref.get("redline_gsm8k_summary") or ref
    total = int(summary.get("total") or 0)
    if total < 1:
        pytest.fail(f"{ref_path}: GSM8K dump has total=0")

    for path, d in dumps:
        s = d.get("redline_gsm8k_summary") or d
        samples = d.get("samples") or []
        if int(s.get("total") or 0) != total:
            pytest.fail(f"{path}: total {s.get('total')} != {total} in {ref_path}")
        if len(samples) != total:
            pytest.fail(f"{path}: {len(samples)} samples, header total={total}")
        empty = [r["idx"] for r in samples if not r.get("generate_length")]
        if empty:
            pytest.fail(f"{path}: no generated tokens for idx={empty}")

    if len(dumps) > 1:
        ref_rows = {r["idx"]: r for r in ref.get("samples", [])}
        for path, d in dumps[1:]:
            for row in d.get("samples") or []:
                other = ref_rows.get(row["idx"])
                if other is None:
                    pytest.fail(f"{path}: extra idx {row['idx']}")
                if row.get("extracted") != other.get("extracted"):
                    pytest.fail(
                        f"G1 FAIL idx={row['idx']}: {path} extracted "
                        f"{row.get('extracted')!r} vs {other.get('extracted')!r} "
                        f"in {ref_path}"
                    )

    acc = float(summary.get("value", summary.get("accuracy", 0.0)))
    correct = int(summary.get("correct") or 0)
    print(
        f"[glm5 gsm8k] {correct}/{total} = {acc:.4f} "
        f"{summary.get('metric', 'exact_match,flexible-extract')} "
        f"ranks={len(dumps)} floor={MIN_ACC}"
    )
    if acc + 1e-12 < MIN_ACC:
        pytest.fail(
            f"GSM8K accuracy {acc:.4f} < GLM5_GSM8K_MIN_ACC={MIN_ACC}"
        )
