"""Prepare TinyStories and Dolly splits without prompt leakage."""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
SEED = 42
TRAIN_RATIO = 0.90
VAL_RATIO = 0.05
TINY_STORIES_DATASET = "roneneldan/TinyStories"
DOLLY_DATASET = "databricks/databricks-dolly-15k"


def clean_text(value):
    """Trim text and collapse runs of whitespace to one space."""
    return " ".join(str(value or "").split())


def normalize_prompt(value):
    """Return the whitespace-normalized, case-folded Dolly prompt key."""
    return clean_text(value).casefold()


def save_jsonl(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_rows(rows, seed=SEED):
    """Deterministic 90/5/5 split. Dolly must be deduplicated first."""
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    train_end = int(len(rows) * TRAIN_RATIO)
    val_end = train_end + int(len(rows) * VAL_RATIO)
    return {"train": rows[:train_end], "validation": rows[train_end:val_end], "test": rows[val_end:]}


def validate_dolly_splits(splits):
    """Raise if normalized prompts occur in more than one Dolly split."""
    seen = {}
    for split, rows in splits.items():
        for row in rows:
            key = normalize_prompt(row.get("prompt", ""))
            if not key:
                raise ValueError(f"Empty prompt in {split} split")
            if key in seen:
                raise ValueError(f"Prompt overlap between {seen[key]} and {split}: {key!r}")
            seen[key] = split
    return True


def prepare_dolly_rows(records, seed=SEED, source_records=None):
    """Clean, deduplicate, split Dolly rows and return leakage/count stats.

    For duplicate normalized prompts with differing responses, keep the
    lexicographically smallest canonical (response, category) pair. This is
    deterministic regardless of input order and retains one prompt example.
    """
    considered = len(records)
    original = considered if source_records is None else source_records
    grouped = {}
    invalid = 0
    for record in records:
        instruction = clean_text(record.get("instruction"))
        context = clean_text(record.get("context"))
        response = clean_text(record.get("response"))
        category = clean_text(record.get("category"))
        if not instruction or not response:
            invalid += 1
            continue
        prompt = f"Instruction: {instruction}\n"
        if context:
            prompt += f"Context: {context}\n"
        prompt += "Response:"
        row = {"prompt": prompt, "response": response, "category": category}
        grouped.setdefault(normalize_prompt(prompt), []).append(row)

    unique_rows = []
    for candidates in grouped.values():
        candidates.sort(key=lambda row: (row["response"].casefold(), row["response"], row["category"].casefold(), row["category"], row["prompt"].casefold(), row["prompt"]))
        unique_rows.append(candidates[0])
    duplicates_removed = considered - invalid - len(unique_rows)
    splits = split_rows(unique_rows, seed=seed)
    validate_dolly_splits(splits)
    stats = {
        "dataset": "Databricks Dolly 15K",
        "seed": seed,
        "original_records": original,
        "records_considered": considered,
        "invalid_records_removed": invalid,
        "duplicate_prompts_removed": duplicates_removed,
        "records_retained": len(unique_rows),
        "conflicting_duplicate_strategy": "Keep lexicographically smallest canonical response/category for each normalized prompt.",
        "splits": {name: {"examples": len(rows)} for name, rows in splits.items()},
    }
    return splits, stats


def _save_stats(stats, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Statistics saved to: {path}")


def _save_splits(prefix, splits):
    for split, rows in splits.items():
        path = DATA_DIR / f"{prefix}_{split}.jsonl"
        save_jsonl(rows, path)
        print(f"Saved {len(rows)} records: {path}")


def prepare_tiny_stories(max_stories=50000):
    """Prepare TinyStories independently; Dolly prompt rules do not apply."""
    print("Loading TinyStories...")
    dataset = load_dataset(TINY_STORIES_DATASET, split="train").shuffle(seed=SEED)
    if max_stories and max_stories > 0:
        dataset = dataset.select(range(min(max_stories, len(dataset))))
    rows = [{"text": clean_text(row.get("text"))} for row in dataset if clean_text(row.get("text"))]
    splits = split_rows(rows)
    _save_splits("tiny_stories", splits)
    _save_stats({"dataset": "TinyStories", "seed": SEED, "records_retained": len(rows), "splits": {k: {"examples": len(v)} for k, v in splits.items()}}, DATA_DIR / "tiny_stories_stats.json")
    print("Train the tokenizer using ONLY tiny_stories_train.jsonl.")


def prepare_dolly(max_examples=15000):
    print("Loading Databricks Dolly 15K...")
    dataset = load_dataset(DOLLY_DATASET, split="train").shuffle(seed=SEED)
    source_records = len(dataset)
    if max_examples and max_examples > 0:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    splits, stats = prepare_dolly_rows(list(dataset), seed=SEED, source_records=source_records)
    _save_splits("dolly", splits)
    _save_stats(stats, DATA_DIR / "dolly_stats.json")


def main():
    parser = argparse.ArgumentParser(description="Prepare AFML pretraining and SFT datasets.")
    parser.add_argument("--stage", choices=["tiny", "dolly", "all"], default="all")
    parser.add_argument("--max_tiny_stories", type=int, default=50000)
    parser.add_argument("--max_dolly", type=int, default=15000)
    args = parser.parse_args()
    if args.stage in ("tiny", "all"):
        prepare_tiny_stories(args.max_tiny_stories)
    if args.stage in ("dolly", "all"):
        prepare_dolly(args.max_dolly)


if __name__ == "__main__":
    main()
