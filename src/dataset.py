import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
TOKENIZER_PATH = ROOT / "artifacts" / "tokenizer" / "tokenizer.json"


def load_tokenizer():
    """Load the trained tokenizer."""
    from tokenizers import Tokenizer

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {TOKENIZER_PATH}. "
            "Run src/tokenizer.py first."
        )

    return Tokenizer.from_file(str(TOKENIZER_PATH))


def read_jsonl(path):
    """Read JSON Lines into a list of dictionaries."""
    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return rows


class TinyStoriesDataset(Dataset):
    """Next-token prediction examples from TinyStories."""

    def __init__(self, split="train", block_size=256):
        if split not in {"train", "validation", "test"}:
            raise ValueError("Invalid split.")

        if block_size < 2:
            raise ValueError("block_size must be at least 2.")

        self.block_size = block_size
        self.tokenizer = load_tokenizer()
        self.examples = []

        path = DATA_DIR / f"tiny_stories_{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)

        # Each story is encoded separately to avoid joining stories.
        # The tokenizer adds BOS and EOS through its post-processor.
        for row in read_jsonl(path):
            encoded = self.tokenizer.encode(row["text"])
            ids = encoded.ids

            if len(ids) < 3:
                continue

            # Create next-token prediction blocks.
            start = 0

            while start < len(ids) - 1:
                chunk = ids[start:start + block_size + 1]

                if len(chunk) >= 2:
                    x = chunk[:-1]
                    y = chunk[1:]

                    valid_length = len(y)
                    padding = block_size - len(x)

                    # Ignore padding targets in the loss.
                    if padding > 0:
                        x = x + [0] * padding
                        y = y + [-100] * padding

                    self.examples.append((x, y, valid_length))

                if start + block_size >= len(ids) - 1:
                    break

                start += block_size

        print(
            f"TinyStories {split}: {len(self.examples)} blocks "
            f"(block size {block_size})"
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        x, y, valid_length = self.examples[index]

        return {
            "input_ids": torch.tensor(x, dtype=torch.long),
            "labels": torch.tensor(y, dtype=torch.long),
            "attention_mask": torch.tensor(
                [1] * valid_length
                + [0] * (self.block_size - valid_length),
                dtype=torch.long,
            ),
        }


class DollySFTDataset(Dataset):
    """Instruction tuning examples with response-only loss masking."""

    def __init__(self, split="train", max_length=256):
        if split not in {"train", "validation", "test"}:
            raise ValueError("Invalid split.")

        if max_length < 3:
            raise ValueError("max_length must be at least 3 for SFT.")

        self.max_length = max_length
        self.tokenizer = load_tokenizer()
        self.examples = []
        self.stats = {
            "original_examples": 0,
            "empty_examples_skipped": 0,
            "overlength_examples": 0,
            "prompt_too_long_skipped": 0,
            "responses_truncated": 0,
            "final_examples_retained": 0,
        }

        path = DATA_DIR / f"dolly_{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)

        eos_id = self.tokenizer.token_to_id("[EOS]")
        pad_id = self.tokenizer.token_to_id("[PAD]")

        if eos_id is None:
            raise ValueError("Tokenizer does not contain an [EOS] token.")
        if pad_id is None:
            raise ValueError("Tokenizer does not contain a [PAD] token.")

        for row in read_jsonl(path):
            self.stats["original_examples"] += 1
            prompt = row["prompt"]
            response = row["response"]

            # Encode prompt and response separately so prompt tokens
            # can be excluded from the training loss.
            prompt_ids = self.tokenizer.encode(
                prompt, add_special_tokens=False
            ).ids

            response_ids = self.tokenizer.encode(
                response, add_special_tokens=False
            ).ids

            # Empty responses cannot provide a useful SFT target.
            if not prompt_ids or not response_ids:
                self.stats["empty_examples_skipped"] += 1
                continue

            # Preserve the full instruction. Truncate only the response and
            # always reserve one position for EOS.
            if len(prompt_ids) + len(response_ids) + 1 > max_length:
                self.stats["overlength_examples"] += 1
                if len(prompt_ids) + 1 >= max_length:
                    self.stats["prompt_too_long_skipped"] += 1
                    continue
                response_budget = max_length - len(prompt_ids) - 1
                if len(response_ids) > response_budget:
                    self.stats["responses_truncated"] += 1
                    response_ids = response_ids[:response_budget]
                if not response_ids:
                    self.stats["empty_examples_skipped"] += 1
                    continue

            # Construct the complete sequence.
            input_ids = prompt_ids + response_ids + [eos_id]

            # Ignore prompt tokens; supervise response tokens and EOS.
            labels = (
                [-100] * len(prompt_ids)
                + response_ids
                + [eos_id]
            )

            # Valid tokens receive attention; padding does not.
            attention_mask = [1] * len(input_ids)

            padding = max_length - len(input_ids)

            if padding > 0:
                input_ids += [pad_id] * padding
                labels += [-100] * padding
                attention_mask += [0] * padding

            self.examples.append({
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            })

        self.stats["final_examples_retained"] = len(self.examples)
        print(f"Dolly {split} statistics (max length {max_length}):")
        for name, count in self.stats.items():
            print(f"  {name}: {count}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        return {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in example.items()
        }


if __name__ == "__main__":
    print("Checking dataset loading...")

    tiny = TinyStoriesDataset(split="train", block_size=128)

    if len(tiny):
        sample = tiny[0]
        print("TinyStories input:", sample["input_ids"].shape)
        print("TinyStories labels:", sample["labels"].shape)
        print("TinyStories attention mask:", sample["attention_mask"].shape)

    dolly = DollySFTDataset(split="train", max_length=256)

    if len(dolly):
        sample = dolly[0]
        supervised = (sample["labels"] != -100).sum().item()

        print("Dolly input:", sample["input_ids"].shape)
        print("Dolly labels:", sample["labels"].shape)
        print("Dolly attention mask:", sample["attention_mask"].shape)
        print("Supervised response tokens:", supervised)
        print(
            "Ignored prompt/padding tokens:",
            (sample["labels"] == -100).sum().item(),
        )
