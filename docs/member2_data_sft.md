# Member 2: data, tokenizer, datasets, and SFT

## Files

- `src/preprocess.py`: TinyStories and Dolly download, cleaning, reproducible splits, statistics, and JSONL output.
- `src/tokenizer.py`: ByteLevel BPE training and validation.
- `src/dataset.py`: PyTorch pretraining and instruction-tuning datasets.
- `src/sft.py`: pretrained model loading, AdamW training, response-only loss, per-epoch validation, and best/latest checkpoint output.
- `tests/test_pipeline.py`: synthetic Member 2 unit coverage.

## Set up and regenerate data (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src/preprocess.py --stage all
```

Sources: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) and [Databricks Dolly 15K](https://huggingface.co/datasets/databricks/databricks-dolly-15k), via `datasets`. Defaults select at most 50,000 TinyStories and 15,000 Dolly records; use `--max_tiny_stories 0` / `--max_dolly 0` to use all available source records. Generated JSONL files are ignored by Git and can be regenerated with these commands.

TinyStories is handled separately and is not deduplicated by Dolly rules. Dolly records with missing instruction or response are removed. Prompts are whitespace-normalized and case-folded to find duplicates before splitting. If duplicate prompts have different responses, the lexicographically smallest canonical response/category is retained. The fixed seed is 42 and split proportions are 90/5/5. `dolly_stats.json` reports original source size, considered rows, invalid rows, duplicate rows, retained rows, and split counts. `validate_dolly_splits` checks normalized prompt separation.

## Tokenizer

```powershell
python src/tokenizer.py
```

`artifacts/tokenizer/tokenizer.json` and `config.json` are checked into the repository (about 0.55 MB total). The command validates an existing artifact and skips retraining if valid. If absent, it trains a ByteLevel BPE with vocabulary size 8,000 using only `tiny_stories_train.jsonl`.

| Special token | ID |
|---|---:|
| `[PAD]` | 0 |
| `[UNK]` | 1 |
| `[BOS]` | 2 |
| `[EOS]` | 3 |
| `[SEP]` | 4 |

Single-document tokenizer encoding inserts BOS/EOS through the post-processor. Dolly prompt/response encoding disables automatic special tokens; dataset code explicitly appends EOS.

## Dataset tensors and templates

```python
from src.dataset import TinyStoriesDataset, DollySFTDataset

pretrain = TinyStoriesDataset(split="train", block_size=256)
sft = DollySFTDataset(split="train")  # max_length defaults to 256
batch_item = sft[0]
```

Each item returns `input_ids`, `labels`, and `attention_mask` as `torch.long`, each shaped `[sequence]`; DataLoader batches have shape `[batch, sequence]`. TinyStories inputs/labels are shifted next-token pairs, with right padding and `-100` ignored targets. Dolly prompt format is exactly `Instruction: {instruction}\n`, optional `Context: {context}\n`, then `Response:`. Prompt labels and padding labels are `-100`; response tokens and EOS are supervised. Attention is `1` on valid tokens and `0` on padding.

The SFT context is 256 by default and must not exceed model `context_length`. When a Dolly sequence is too long, the full prompt is kept and only the response tail is truncated, reserving one token for EOS. If the whole prompt plus EOS does not fit, the row is skipped. Each dataset instance prints original, empty, overlength, too-long prompt, truncated response, and retained counts, also available in `.stats`.

## SFT training and checkpoint use

Member 1's pretraining saves `checkpoints/<run_name>/best_checkpoint.pt` with `model_state`, the model `config` and `tokenizer_path`. `src/sft.py` uses the standard-attention checkpoint by default:

```powershell
python src/sft.py --pretrained-checkpoint checkpoints\baseline_standard\best_checkpoint.pt --max-length 256 --batch-size 4 --learning-rate 0.0003 --epochs 4
```

`src.model:build_model` is the default factory. The SFT runner reads architecture settings from the checkpoint, verifies that vocabulary is 8,000 and context supports the requested length, loads weights strictly, chooses CUDA when available and otherwise CPU, and uses AdamW. It validates every 500 steps and at each epoch end, and writes `checkpoints/sft/best.pt`, `latest.pt`, `config.json` and the log `results/metrics/sft_train_log.csv` (same columns as pretraining). The defaults (lr 3e-4, 4 epochs) come from a comparison on Dolly validation (`results/metrics/rehearsal_sft/`). Checkpoints contain the `model_state` key and canonical config used by Member 3's `src.model_api.load_checkpoint`; optimizer and training metadata are also saved. Checkpoint files are ignored by Git.

Member 1 can use the tracked tokenizer and `TinyStoriesDataset` for pretraining; its loop receives shifted labels and the attention mask. Member 3 can use the same tokenizer and `dolly_test.jsonl`, with train/validation files passed to evaluation's `--exclude` option. The prompt template is shared with preprocessing and SFT. Evaluation SFT loss now uses full prompt preservation and response-tail truncation like training.

## Integration details

`src.model:build_model(config)` builds either attention variant and is used by evaluation and SFT. Without `targets`, both models return logits `[B,T,V]`; with `targets`, they return `(logits, loss)`. See `docs/interfaces.md` for checkpoint and model details.
