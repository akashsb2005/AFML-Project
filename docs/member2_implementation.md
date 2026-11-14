# Member 2 implementation and verification

## Ownership and sources

Member 2 implemented preprocessing, tokenizer training/validation, dataset classes and response-only SFT integration. Source datasets are TinyStories (`roneneldan/TinyStories`) and Databricks Dolly 15K (`databricks/databricks-dolly-15k`). Existing processed data is local and excluded from Git.

## Data and splits

TinyStories contains 10,000 stories: train 9,000, validation 500, test 500. Its data remains independently shuffled/split; Dolly prompt deduplication is not applied. Dolly source count is 15,011; the selected source subset was 15,000. Zero invalid rows and 203 duplicate normalized prompts were removed, leaving 14,797 rows: train 13,317, validation 739, test 741. Duplicate keys trim/collapse whitespace and case-fold the formatted prompt. Conflicts deterministically retain the lexicographically smallest canonical response/category. Splits use seed 42 at 90/5/5 and a cross-split normalized prompt validator.

## Tokenizer

The tracked artifact is ByteLevel BPE, vocabulary 8,000. IDs are `[PAD]` 0, `[UNK]` 1, `[BOS]` 2, `[EOS]` 3, `[SEP]` 4. Its postprocessor adds BOS/EOS for single inputs; Dolly manually adds EOS. PAD 0 is used for right padding, with padding targets `-100`. Validation passed JSON load, size, IDs, round trip and BOS/EOS checks. Config declares `tiny_stories_train.jsonl`; source code trains from only that file, and it is present locally. The artifact alone does not independently prove its original training corpus; see limitations.

## Dataset behavior

`TinyStoriesDataset(split, block_size=256)` encodes each story separately and emits fixed `[256]` `input_ids`, shifted next-token labels, and attention masks. Pad inputs are 0 and ignored labels are -100.

`DollySFTDataset(split, max_length=256)` formats `Instruction: ...`, optional `Context: ...`, then `Response:`. Prompt labels are -100; response tokens and EOS are supervised. Padding uses ID 0, attention 0, labels -100. On overflow it keeps the complete prompt, truncates response tokens to reserve EOS, and skips a prompt that cannot fit with EOS. Per-split JSON counts are saved beside processed data and copied under `member2_final/logs/`.

## Model and SFT

SFT uses `src.model:build_model` and the checkpoint's model configuration and state dict (`model_state`/`model_state_dict`). It checks vocab and context, calls forward with `input_ids` and supported `attention_mask`, then computes causal shifted cross-entropy with prompt/pad `-100` ignored. AdamW and per-epoch no-gradient validation are implemented. Best/latest checkpoints, config and JSON epoch log are implemented; `--resume` loads model and optimizer from `output-dir/latest.pt`. Existing outputs are backed up before replacing canonical outputs.

No compatible pretrained checkpoint exists in this checkout, so no actual SFT experiment, checkpoint or SFT metric is available. Intended defaults are in `member2_final/configs/sft_config.json`. Dataset construction actually retained 10,697/606/591 Dolly train/validation/test examples after length filtering. These are data counts, not model results.

## Reproduction

```powershell
python -m pip install -r requirements.txt
python src/preprocess.py --stage all
python src/tokenizer.py
python -m pytest -q
python src/sft.py --pretrained-checkpoint D:\path	o\last_checkpoint.pt --max-length 256 --batch-size 4 --learning-rate 0.00002 --epochs 1
```

Use `--resume` to resume from `checkpoints/sft/latest.pt` after a first SFT epoch. Pretraining's current script expects its checkpoint in `/content/drive/MyDrive/afml-slm-project/checkpoints/last_checkpoint.pt`; copy it locally to the SFT command path. Evaluation can use `src/evaluate.py sft-loss` with `dolly_test.jsonl` and train/validation paths passed to `--exclude`.

## Artifact locations and limitations

Preserved data/tokenizer/reports are in `member2_final/`; checksums are in `member2_final/SHA256SUMS.txt`. No checkpoint placeholder was created. No Drive connection is configured in this environment. Tokenizer config records training filename, not a corpus hash from its original training run. Dolly prompt formatting is based on the preprocessed records, and validation only establishes no normalized prompt crosses the three local splits.
