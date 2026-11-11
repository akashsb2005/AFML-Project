# SFT rehearsal (Member 3): PROVISIONAL, not final results

Run on 2026-11-04 on an Apple M1 (MPS, 16 GB, torch 2.13.0) by Member 3, to test the merged pipeline and choose
sensible fine-tuning settings before the official runs.

* **Code:** a scratch merge of `fix/member1-training-integration` (e876c64), `member-2-data-sft` (d2f734a) and
  `member3-integration`. The only edit was to pick the MPS device in `src/training.py` and `src/sft.py`.
* **Data:** Dolly regenerated with PR #5's `preprocess.py` (13,317 / 739 / 741). TinyStories is a 9,000 / 500 / 500 sample
  (10,375 training blocks of 256), not Member 2's exact final subset. Tokenizer: the tracked 8,000-token artifact.
* **Pre-training:** standard attention, `configs/baseline.yaml` with `--max_steps 3000`, seed 42.
* **SFT:** `src/sft.py`, batch 4, weight decay 0.01, seed 42, from the best pre-training checkpoint.
* **Evaluation:** Dolly **validation** only, so the test split stays untouched. Response NLL on 606 scored examples
  (133 skipped because the prompt does not fit), and 60 validation prompts (seed 2026, `val_prompt_ids.json`).
  Greedy, plus temperature 0.8 and top-k 50 (seed 1234).
* **Files:** `reh2_*`, `examples.json` (9 of the 60 prompts). Checkpoints were not kept in the repo.

One seed and one run per setting. The final report must use the official matched runs.
