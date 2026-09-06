#!/usr/bin/env python3
"""Redline/ATOM GSM8K protocol, scored in-process against the GLM megakernel.

Redline's canonical GPT-OSS accuracy path is ``utils/gsm8k_lm_eval.py``:
lm-eval task ``gsm8k``, 3-shot, chat completions with the model chat template,
metric ``exact_match,flexible-extract``. Fleet has no OpenAI endpoint, so this
module reproduces that protocol without lm-eval or HTTP:

  * few-shot examples drawn from GSM8K train (seed 1234, same default as
    Redline's accuracy workflow)
  * chat-template messages (user/assistant turns), thinking off
  * last-number flexible-extract, plus the strict ``#### <n>`` extractor
  * gold from the dataset's ``#### <n>`` field

The megakernel compiles once and re-launches per sample (see demo.py); this
file is only the dataset, prompt construction, and scorer.

    python3 gsm8k.py --limit 2          # print prompts (no GPU)
    python3 gsm8k.py --score-json dump.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from typing import Any

FLEXIBLE_EXTRACT_METRIC = "exact_match,flexible-extract"

# lm-eval gsm8k.yaml flexible-extract, group_select=-1 (last match).
_FLEX = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
_STRICT = re.compile(r"####\s*(-?[0-9.,]+)")


def normalize_number(text: str | None) -> str | None:
    if not text:
        return None
    s = text.replace(",", "").replace("$", "").strip().rstrip(".")
    if not s:
        return None
    try:
        value = float(s)
    except ValueError:
        return s
    if value == int(value):
        return str(int(value))
    return str(value)


def extract_flexible(text: str | None) -> str | None:
    """Last number in the completion. Matches lm-eval flexible-extract."""
    if not text:
        return None
    matches = [m.group(1) or m.group(2) for m in _FLEX.finditer(text)]
    if not matches:
        return None
    return normalize_number(matches[-1])


def extract_strict(text: str | None) -> str | None:
    if not text:
        return None
    found = _STRICT.findall(text)
    if not found:
        return None
    return normalize_number(found[-1])


def gold_answer(answer_field: str) -> str | None:
    return extract_strict(answer_field) or extract_flexible(answer_field)


def load_gsm8k(split: str = "test", num_fewshot: int = 3, seed: int = 1234,
               limit: int | None = None) -> dict[str, Any]:
    """Load GSM8K and a fixed few-shot prefix shared by every test item."""
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main")
    if split not in ds:
        raise ValueError(f"GSM8K split {split!r} not in {list(ds)}")
    train = list(ds["train"])
    docs = list(ds[split])
    rng = random.Random(seed)
    if num_fewshot < 0 or num_fewshot > len(train):
        raise ValueError(f"num_fewshot={num_fewshot} vs train size {len(train)}")
    shots = rng.sample(train, num_fewshot) if num_fewshot else []
    if limit is not None:
        docs = docs[:limit]
    items = []
    for i, doc in enumerate(docs):
        messages = []
        for shot in shots:
            messages.append({"role": "user", "content": shot["question"]})
            messages.append({"role": "assistant", "content": shot["answer"]})
        messages.append({"role": "user", "content": doc["question"]})
        items.append({
            "idx": i,
            "question": doc["question"],
            "answer": doc["answer"],
            "gold": gold_answer(doc["answer"]),
            "messages": messages,
        })
    return {
        "protocol": "redline-gsm8k-lm-eval",
        "split": split,
        "num_fewshot": num_fewshot,
        "seed": seed,
        "limit": limit,
        "metric": FLEXIBLE_EXTRACT_METRIC,
        "items": items,
    }


def score_completion(item: dict[str, Any], completion: str,
                     token_ids: list[int] | None = None) -> dict[str, Any]:
    extracted = extract_flexible(completion)
    gold = item.get("gold")
    return {
        "idx": item["idx"],
        "question": item["question"],
        "gold": gold,
        "extracted": extracted,
        "extracted_strict": extract_strict(completion),
        "correct": extracted is not None and gold is not None
                   and extracted == gold,
        "generate_length": len(token_ids) if token_ids is not None else None,
        "completion": completion,
        "token_ids": token_ids,
    }


def summarize(rows: list[dict[str, Any]], **meta: Any) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for r in rows if r.get("correct"))
    return {
        **meta,
        "protocol": "redline-gsm8k-lm-eval",
        "metric": FLEXIBLE_EXTRACT_METRIC,
        "correct": correct,
        "total": total,
        "accuracy": (correct / total) if total else 0.0,
        "samples": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--num-fewshot", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=2)
    ap.add_argument("--score-json", default=None,
                    help="Score an existing demo.py GSM8K dump")
    args = ap.parse_args()
    if args.score_json:
        data = json.load(open(args.score_json))
        s = data.get("redline_gsm8k_summary") or data
        print(f"{s.get('metric', FLEXIBLE_EXTRACT_METRIC)}: "
              f"{s.get('correct')}/{s.get('total')} = {s.get('accuracy')}")
        return 0
    pack = load_gsm8k(split=args.split, num_fewshot=args.num_fewshot,
                      seed=args.seed, limit=args.limit)
    print(f"loaded {len(pack['items'])} {pack['split']} items, "
          f"{pack['num_fewshot']}-shot seed={pack['seed']}", file=sys.stderr)
    for item in pack["items"]:
        print(f"--- idx={item['idx']} gold={item['gold']}")
        print(item["question"].splitlines()[0][:120])
    return 0


if __name__ == "__main__":
    sys.exit(main())
