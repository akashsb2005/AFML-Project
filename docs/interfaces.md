# Integration interfaces

Member 1 owns the Transformer and pretraining; Member 2 owns datasets, tokenizer, preprocessing, and SFT; Member 3 owns evaluation and benchmarking. `src/model_api.py` is the shared evaluation adapter.

## Tokenizer and data

| Interface | Contract |
|---|---|
| Tokenizer artifact | `artifacts/tokenizer/tokenizer.json` plus `config.json`; ByteLevel BPE, 8,000 IDs |
| Special tokens | `[PAD]`=0, `[UNK]`=1, `[BOS]`=2, `[EOS]`=3, `[SEP]`=4 |
| Whole-document encoding | Tokenizer post-processor adds BOS and EOS |
| TinyStories | JSONL `text`; `TinyStoriesDataset(split, block_size)` produces fixed-length `input_ids`, shifted `labels`, `attention_mask` |
| Dolly | JSONL `prompt`, `response`, `category`; 90/5/5 seeded splits after normalized-prompt deduplication |
| SFT format | `Instruction: {instruction}\n[Context: {context}\n]Response:` |
| Dolly tensors | `DollySFTDataset(split, max_length=256)` returns fixed-length `input_ids`, `labels`, `attention_mask`; prompt/pad labels are `-100`, response and EOS are supervised |

`data/processed/` is generated locally by `python src/preprocess.py --stage all`. Train the tokenizer only on `tiny_stories_train.jsonl`. Dolly split validation is available as `preprocess.validate_dolly_splits`.

On SFT overflow, the full prompt is retained and only the response tail is truncated. One position is reserved for EOS. Records whose full prompt and EOS do not fit are skipped and counted. Evaluation uses the same response truncation and skip behavior in `evaluate.build_sft_example`.

## Model interface

`src.model.build_model(config)` accepts the baseline YAML model keys (`vocab_size`, `d_model`, `n_head`, `n_layer`, `context_length`, `dropout`, `attention`) or canonical equivalents. `attention: standard` returns `SmallLanguageModel`; `attention: linear` returns `EfficientSmallLanguageModel`.

Models accept `model(input_ids, targets=None, attention_mask=None)`. Without targets, `forward` returns logits `[B,T,V]`; with targets it returns `(logits, loss)`. Right-padded attention masks are shape-validated. Standard causal attention does not apply the mask because a right padding suffix cannot affect preceding causal positions. The linear attention model also receives the mask.

Vocabulary must match the tokenizer (8,000); `context_length` must be at least the sequence length used by pretraining/SFT. Pretraining batches contain shifted targets, with `-100` on padding. Member 1's training entry points now pass the mask and target labels through the model interface.

## Checkpoint format

Pretraining and SFT save:

```python
{
    "model_state": model.state_dict(),
    "model_state_dict": model.state_dict(),  # training/resume compatibility
    "config": {"vocab_size": 8000, "embedding_dim": 256, "num_heads": 4,
                "num_layers": 4, "context_length": 256, "dropout": 0.1,
                "attention": "standard"},
    "tokenizer_path": "artifacts/tokenizer/tokenizer.json",
    "stage": "pretrain" or "sft",
}
```

The evaluation adapter loads `model_state` (or `model_state_dict` for older pretraining checkpoints) and builds via `src.model:build_model`. SFT adds optimizer state, epoch, validation loss, and `training_config`. Output is `checkpoints/sft/{best,latest}.pt`. Model weight files are not committed.

Member 3's `evaluate.py`, `benchmark.py`, and prompt selector use the shared tokenizer artifact and prompt template. Evaluation prompt selection uses held-out `dolly_test.jsonl` and can exclude training/validation prompts. Evaluation remains distinct from optimization.
