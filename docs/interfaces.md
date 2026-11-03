# Integration interfaces

Member 1 owns the model interface and Member 2 owns the data/tokenizer interface (plan §5, rules 1–4).
Member 3's code reaches the model, tokenizer and data only through `src/model_api.py`, so if a name below
changes, only that adapter needs editing.

## Data and tokenizer — Member 2 (as implemented in `src/preprocess.py`, `src/tokenizer.py`, `src/dataset.py`)

| item | value |
|---|---|
| tokenizer | `artifacts/tokenizer/tokenizer.json` (Byte-level BPE, Hugging Face `tokenizers`), config in `artifacts/tokenizer/config.json` |
| vocabulary | 8000 |
| special tokens | `[PAD]`=0, `[UNK]`=1, `[BOS]`=2, `[EOS]`=3, `[SEP]`=4 |
| `tokenizer.encode(text)` | adds `[BOS] … [EOS]` (post-processor); `add_special_tokens=False` gives the bare text |
| TinyStories | `data/processed/tiny_stories_{train,validation,test}.jsonl`, field `text`; 9,000 / 500 / 500 stories, seed 42 |
| pre-training samples | `TinyStoriesDataset(split, block_size)`: each story encoded as `[BOS] story [EOS]`, cut into blocks of `block_size` (consecutive blocks share one token), right-padded with 0; `labels` = next token, `-100` on padding; `attention_mask` |
| Dolly | `data/processed/dolly_{train,validation,test}.jsonl`, fields `prompt`, `response`, `category`; 13,500 / 750 / 750, seed 42 |
| SFT template | `"Instruction: {instruction}\n" + ("Context: {context}\n" if context) + "Response:"` — stored ready-made in `prompt` |
| SFT samples | `DollySFTDataset(split, max_length)`: prompt and response encoded **without** special tokens and concatenated **without** a space, then `[EOS]`; labels `-100` on the prompt and padding; if too long the response is kept and the prompt cut from the left |

**Must hold for integration:** `block_size` (pre-training) and `max_length` (SFT) must equal the model's
`context_length`. The `DollySFTDataset` default `max_length=512` would exceed a 256-token context.

Member 3's evaluation reproduces both constructions exactly (`evaluate.story_blocks`,
`evaluate.build_sft_example`; checked to be identical to the dataset classes on the validation/test
splits), so evaluation losses are the same quantities as the training losses.

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
