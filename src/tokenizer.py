
import json
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers
from tokenizers import decoders, trainers, processors


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
TOKENIZER_DIR = ROOT / "artifacts" / "tokenizer"

VOCAB_SIZE = 8000
EXPECTED_SPECIAL_IDS = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3, "[SEP]": 4}

SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[BOS]",
    "[EOS]",
    "[SEP]",
]


def validate_tokenizer(tokenizer, config=None):
    """Validate ByteLevel BPE vocabulary, special IDs, and round trips."""
    size = tokenizer.get_vocab_size()
    if size != VOCAB_SIZE:
        raise ValueError(f"Expected vocabulary size {VOCAB_SIZE}, found {size}.")
    actual = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    if actual != EXPECTED_SPECIAL_IDS:
        raise ValueError(f"Unexpected special token IDs: {actual}")
    if config is not None:
        if config.get("vocab_size") != VOCAB_SIZE:
            raise ValueError("Tokenizer config vocabulary size does not match 8000.")
        for token, token_id in EXPECTED_SPECIAL_IDS.items():
            key = token[1:-1].lower() + "_token_id"
            if config.get(key) != token_id:
                raise ValueError(f"Tokenizer config has incorrect {key}.")
    sample = "Tokenizer round trip: café, hello!"
    encoded = tokenizer.encode(sample, add_special_tokens=False)
    if tokenizer.decode(encoded.ids) != sample:
        raise ValueError("Tokenizer encode/decode round trip failed.")
    special = tokenizer.encode("sample")
    if not special.ids or special.ids[0] != 2 or special.ids[-1] != 3:
        raise ValueError("Tokenizer must insert BOS and EOS for single input.")
    if tokenizer.token_to_id("[PAD]") != 0:
        raise ValueError("PAD token must have ID 0.")
    return True


def load_and_validate_tokenizer():
    path = TOKENIZER_DIR / "tokenizer.json"
    config_path = TOKENIZER_DIR / "config.json"
    if not path.exists() or not config_path.exists():
        raise FileNotFoundError("Tokenizer JSON/config missing; train it from TinyStories train split.")
    tokenizer = Tokenizer.from_file(str(path))
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)
    validate_tokenizer(tokenizer, config)
    return tokenizer


def read_training_texts():
    """Read only the TinyStories training split."""
    train_path = DATA_DIR / "tiny_stories_train.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} not found. Run preprocess.py first."
        )

    with train_path.open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            text = row.get("text", "").strip()

            if text:
                yield text


def train_tokenizer():
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(
        models.BPE(unk_token="[UNK]")
    )

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print("Training BPE tokenizer on TinyStories training data...")

    tokenizer.train_from_iterator(
        read_training_texts(),
        trainer=trainer,
    )

    # Add BOS/EOS consistently when encoding complete stories.
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")

    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [SEP] $B:1 [EOS]:1",
        special_tokens=[
            ("[BOS]", bos_id),
            ("[EOS]", eos_id),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )

    tokenizer_path = TOKENIZER_DIR / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    config = {
        "vocab_size": tokenizer.get_vocab_size(),
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
        "sep_token": "[SEP]",
        "pad_token_id": tokenizer.token_to_id("[PAD]"),
        "unk_token_id": tokenizer.token_to_id("[UNK]"),
        "bos_token_id": bos_id,
        "eos_token_id": eos_id,
        "sep_token_id": tokenizer.token_to_id("[SEP]"),
        "training_data": "tiny_stories_train.jsonl",
        "requested_vocab_size": VOCAB_SIZE,
    }

    with (TOKENIZER_DIR / "config.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(config, file, indent=2)

    print(f"\nTokenizer saved to: {tokenizer_path}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    validate_tokenizer(tokenizer, config)
    print("Special token IDs:", {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS})

    # Quick sanity check
    sample = "Once upon a time, there was a little cat."
    encoded = tokenizer.encode(sample)

    print("\nSample text:", sample)
    print("Encoded token IDs:", encoded.ids)
    print("Decoded text:", tokenizer.decode(encoded.ids))


if __name__ == "__main__":
    if (TOKENIZER_DIR / "tokenizer.json").exists() and (TOKENIZER_DIR / "config.json").exists():
        load_and_validate_tokenizer()
        print("Existing tokenizer passed validation; skipping retraining.")
    else:
        train_tokenizer()
