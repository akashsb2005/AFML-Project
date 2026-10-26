"""Freeze the fixed instruction-evaluation prompt set (protocol §4).

Samples --n prompts from Member 2's held-out Dolly test split, balanced across categories, with a fixed
seed. Optionally drops prompts that do not fit the context window and checks that no prompt also occurs
in the training/validation splits.

python src/select_eval_prompts.py --test data/processed/dolly_test.jsonl \
    --exclude data/processed/dolly_train.jsonl data/processed/dolly_validation.jsonl \
    --tokenizer artifacts/tokenizer/tokenizer.json --context-length 256 --n 60
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import model_api  # noqa: E402
from src.evaluate import read_jsonl, write_jsonl  # noqa: E402


def fingerprint(ex: dict) -> str:
    text = ex.get("prompt") or (ex.get("instruction", "") + " " + ex.get("context", ""))
    return hashlib.sha1(" ".join(text.lower().split()).encode()).hexdigest()


def with_ids(rows: list[dict], split: str = "test") -> list[dict]:
    """Member 2's Dolly rows carry no id: use the line number in the split file (stable, traceable) and
    add instruction/context recovered from the templated prompt for display and statistics."""
    out = []
    for i, r in enumerate(rows):
        r = dict(r)
        r.setdefault("id", f"dolly_{split}_{i:05d}")
        if "instruction" not in r and r.get("prompt"):
            r["instruction"], r["context"] = model_api.parse_prompt(r["prompt"])
        out.append(r)
    return out


def select(test_rows, n, seed, exclude_rows=(), fits=None):
    banned = {fingerprint(r) for r in exclude_rows}
    leaked = [r for r in test_rows if fingerprint(r) in banned]
    pool = [r for r in test_rows if fingerprint(r) not in banned and (fits is None or fits(r))]
    by_cat = defaultdict(list)
    for r in pool:
        by_cat[r.get("category", "unknown")].append(r)
    rng = random.Random(seed)
    for rows in by_cat.values():
        rng.shuffle(rows)
    chosen, cats = [], sorted(by_cat)
    while len(chosen) < n and any(by_cat.values()):          # round-robin over categories
        for c in cats:
            if by_cat[c] and len(chosen) < n:
                chosen.append(by_cat[c].pop())
    return chosen, leaked


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", required=True)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--context-length", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--out", default="results/metrics/eval_prompts.jsonl")
    ap.add_argument("--strict", action="store_true", help="abort instead of skipping leaked prompts")
    a = ap.parse_args(argv)

    exclude = [r for p in a.exclude for r in read_jsonl(p)]
    fits = None
    if a.tokenizer:
        tok = model_api.load_tokenizer(a.tokenizer)
        budget = a.context_length - a.max_new_tokens

        def fits(r):
            return len(tok.encode(model_api.prompt_text(r))) <= budget

    chosen, leaked = select(with_ids(read_jsonl(a.test)), a.n, a.seed, exclude, fits)
    if leaked and a.strict:
        sys.exit(f"ERROR: {len(leaked)} test prompts also appear in the excluded (training) files")
    if leaked:   # Dolly contains duplicated prompts; a random split puts copies on both sides
        print(f"WARNING: {len(leaked)} test prompts also occur in the excluded files — they were left out "
              f"of the selection: {[r['id'] for r in leaked]}", file=sys.stderr)
    write_jsonl(a.out, chosen)
    counts = defaultdict(int)
    for r in chosen:
        counts[r.get("category", "unknown")] += 1
    print(json.dumps({"written": a.out, "n": len(chosen), "seed": a.seed, "leaked_skipped": len(leaked),
                      "per_category": counts}, indent=2))


if __name__ == "__main__":
    main()
