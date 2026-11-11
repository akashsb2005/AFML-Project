# Pipeline rehearsal (Member 3) — PROVISIONAL, not final results

Run on 2026-10-28, Colab Tesla T4, torch 2.11.0+cu130, by Member 3 to validate the full pipeline before
Member 1's official runs.

* Data: Member 2's `preprocess.py` / `tokenizer.py` (md5 df4064da… / 39814665…, the WhatsApp copies),
  `--max_tiny_stories 10000 --max_dolly 15000`, seed 42 → TinyStories 9,000/500/500, Dolly 13,500/750/750,
  19,850 training blocks of 128 (identical to Member 2's reported count). Vocabulary 8,000.
* Models: Member 1's `SmallLanguageModel` / `EfficientSmallLanguageModel` from `main` (299afbc) with the two
  fixes from the PR #2 review applied **in the Colab session only** (`forward(..., attention_mask=None)`;
  `model.train()` at the end of `evaluate()` in `pretrain_efficient.py`), checkpoints written to local disk.
* Training (both identical): `--max_steps 3000 --batch_size 32 --eval_interval 250 --learning_rate 3e-4
  --seed 42`, block size 128. Training-loop validation loss uses the first 30 validation batches only.
* Evaluation: `src/evaluate.py` on the full 500 validation stories (1,099 blocks, 104,882 tokens); Dolly test
  response NLL (737 examples after dropping 13 prompts that also occur in train/val); 60 frozen prompts
  (seed 2026, context 128, 64 new tokens); 10 story openings (first 6 words of the first 10 validation stories);
  greedy decoding.

Only files in this folder come from this run. The final report must use Member 1's official matched runs.
