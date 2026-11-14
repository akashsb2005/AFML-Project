# Member 2 handoff

## To Member 1

- Tokenizer: `member2_final/tokenizer/tokenizer.json` and `config.json` (8,000 IDs, fixed special-token IDs).
- TinyStories: `member2_final/data/processed/tiny_stories_{train,validation,test}.jsonl`.
- Build `TinyStoriesDataset(split, block_size=256)` for fixed-size next-token batches. `input_ids`, shifted `labels`, `attention_mask` have shape `[batch, 256]`; padding labels are `-100`.
- Run `python src/tokenizer.py` to validate the tracked artifact. It trains from TinyStories train only if artifact files are absent.

## To Member 3

- Held-out data: `member2_final/data/processed/dolly_test.jsonl`; validation is separate. Exclude train and validation prompts in `evaluate.py sft-loss` and prompt selection.
- Use the shared tokenizer and `Instruction: ... [Context: ...] Response:` format.
- There is no final SFT checkpoint because this checkout has no compatible pretrained checkpoint. Member 1 must supply the actual pretraining `last_checkpoint.pt`; then run the SFT command below and share `checkpoints/sft/best.pt`, `latest.pt`, and `training_log.json` after verifying hashes.

## Commands (PowerShell)

```powershell
python -m pytest -q
python src/sft.py --pretrained-checkpoint D:\path	o\last_checkpoint.pt --max-length 256 --batch-size 4 --learning-rate 0.00002 --epochs 1
python src/evaluate.py sft-loss --ckpt checkpoints/sft/best.pt --label member2-final --data data/processed/dolly_test.jsonl --exclude data/processed/dolly_train.jsonl data/processed/dolly_validation.jsonl
```

Copy `member2_final/` to the versioned shared Drive folder manually, then compare the copied files to `SHA256SUMS.txt`. The workspace has no configured Drive client, so no upload was performed or verified.
