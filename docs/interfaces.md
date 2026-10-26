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

## Model — Member 1 (*to be confirmed by Akash — code not yet in the repository*)

Member 3 needs, from `src/model.py`:

```python
def build_model(config: dict) -> torch.nn.Module      # or the class name + config keys actually used
```

`model(input_ids)` with `input_ids: LongTensor[B, T]` (T ≤ context length) → `logits: FloatTensor[B, T, V]`,
causal (logits at t depend only on tokens ≤ t). If `forward` also takes `targets`/`labels` or returns a
tuple/dict, tell Member 3 — `src/model_api.py` will unwrap it.

Config keys Member 3 reads: `vocab_size`, the context length (`context_length` / `block_size` /
`max_seq_len`), and the attention variant (`attention: standard | linear`). Everything else is passed
through unchanged.

### Checkpoint format (to be confirmed)

```python
torch.save({"model_state": model.state_dict(), "config": config_dict,
            "tokenizer_path": "artifacts/tokenizer/tokenizer.json", "step": int, "stage": "pretrain"|"sft"},
           "checkpoints/<run_id>.pt")
```

### Training log (to be confirmed)

`results/metrics/<run_id>_train_log.csv` with columns `step,tokens_seen,train_loss,val_loss,lr,elapsed_s`
(`val_loss` empty on steps without evaluation). `tokens_seen` should count non-padding tokens.
