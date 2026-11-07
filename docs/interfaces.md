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

## Model — Member 1 (implemented in `src/model.py` and `src/efficient_model.py`)

Member 3 builds either attention variant through `src.model:build_model(config)`. The factory maps the
shared config keys `vocab_size`, `d_model`, `n_head`, `n_layer`, `context_length`, `dropout`, and
`attention` to the corresponding model implementation.

`model(input_ids)` returns logits directly with shape `[B, T, vocab_size]`, as required by the
evaluation and generation adapter. When `targets` are supplied, forward returns `(logits, loss)`.
Both variants accept an optional `attention_mask`; target value `-100` is ignored by the loss.

### Checkpoint format

Training checkpoints use `model_state` (the key consumed by `src/model_api.py`), a model-loader-compatible
`config` (model fields at the top level, plus nested `model`, `train`, `data`, `seed`, and
`training` metadata), `step`, `tokens_seen`, `train_loss`, `validation_loss`, optimizer state,
elapsed time, and random-number-generator states. A separate `best_checkpoint.pt` stores the
lowest validation-loss checkpoint. Resume accepts the legacy `model_state_dict` key from older runs
where applicable, and restores optimizer and RNG state.

## Pretraining commands

`python src/pretrain.py` reads `configs/baseline.yaml`; `python src/pretrain_efficient.py`
reads `configs/efficient_attention.yaml`. Both use the same shared `src/training.py` loop.
Command-line overrides are available, for example `--max_steps 20 --batch_size 4` for a smoke run.
The configured schedule is linear learning-rate warmup for `warmup_steps`, followed by a constant
learning rate. Both attention variants must use identical settings and the same data/tokenizer artifacts.
The scripts fail early with the missing artifact path if Member 2's TinyStories files or tokenizer
have not yet been supplied.

The training CSV is written once per optimizer step using the columns below. Validation loss is
populated only at evaluation steps; `tokens_seen` counts non-padding supervised target tokens.
Existing logs with a different header are rejected to prevent mixing incompatible runs.

### Training log

`results/metrics/<run_id>_train_log.csv` with columns `step,tokens_seen,train_loss,val_loss,lr,elapsed_s`
(`val_loss` empty on steps without evaluation). `tokens_seen` should count non-padding tokens.

## SFT checkpoints

`src/sft.py` reads the architecture from the pretraining checkpoint and saves `checkpoints/sft/{best,latest}.pt` with `model_state`, `model_state_dict`, canonical `config`, `tokenizer_path`, `training_config`, optimizer state, step, epoch, validation loss and `stage: "sft"`. Model weight files are not committed.

