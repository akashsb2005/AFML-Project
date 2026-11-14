# Efficient Training and Instruction Fine-Tuning of Small Language Models

AFML course project — a decoder-only Transformer pre-trained **from scratch** on TinyStories, instruction
fine-tuned on Databricks Dolly 15K, and compared with an efficient (linear) attention variant.

| Member | Role | Owns |
|---|---|---|
| Akash | Member 1 — model & pre-training | `src/model.py`, `src/attention.py`, `src/efficient_model.py`, `src/efficient_attention.py`, `src/training.py`, `src/pretrain.py`, `src/pretrain_efficient.py`, `configs/` |
| Haneesh | Member 2 — data, tokenizer & SFT | `src/tokenizer.py`, `src/dataset.py`, `src/preprocess.py`, `src/sft.py` |
| Samvith | Member 3 — evaluation, experiments & report | `src/evaluate.py`, `src/benchmark.py`, `src/model_api.py`, `src/select_eval_prompts.py`, `src/make_figures.py`, `src/log_experiment.py`, `experiments/`, `results/`, `report/` |

Interfaces between the parts: **`docs/interfaces.md`**. Member 1 implementation record: **`docs/member1_implementation.md`** and viva guide: **`report/viva_member1.md`**. Member 2's detailed workflow is in **`docs/member2_data_sft.md`**.

## Final results (Tesla T4, 10,000 pre-training steps, seed 42)

| | Standard attention | Linear attention |
|---|---|---|
| Test perplexity (uniform = 8,000) | **8.51** | 8.78 |
| Pre-training time | 1,419 s | 3,319 s |
| Dolly test response NLL after fine-tuning (6.38 before) | **3.93** | 4.11 |
| Training throughput, batch 32 × 256 | 79,198 tokens/s | 29,471 tokens/s |

Details: `report/results.md`; write-up: `report/final_report.md`; every run: `experiments/experiment_log.csv`.

**Reproduce:** open `notebooks/final_runs.ipynb` in Colab with a T4 GPU and run all cells (about 1.5 h). It calls
`scripts/run_final.sh`, which saves everything to Google Drive and resumes where it stopped after a disconnect.

## Setup

```bash
pip install -r requirements.txt
python -m pytest -q
python src/preprocess.py --stage all
python src/tokenizer.py
```

Generated datasets and model checkpoints are not committed. The checked 8,000-token tokenizer artifact is tracked at `artifacts/tokenizer/` so local evaluation and SFT share the same IDs. GPU runs: `notebooks/member3_colab.ipynb` (evaluation + benchmarks).

SFT uses Member 1's pretraining checkpoint and defaults to a 256-token context. See `docs/member2_data_sft.md` for the exact command, checkpoint format, prompt masking, and regeneration steps.

## Member 3 tools

| Command | Purpose |
|---|---|
| `python src/evaluate.py perplexity --ckpt C --label L --stories data/processed/tiny_stories_validation.jsonl` | held-out NLL / perplexity (same blocks as training) |
| `python src/evaluate.py sft-loss --ckpt C --label L --data data/processed/dolly_test.jsonl --exclude data/processed/dolly_train.jsonl data/processed/dolly_validation.jsonl` | response-only NLL (overlapping prompts dropped) |
| `python src/select_eval_prompts.py --test ... --exclude ... --tokenizer ...` | freeze the 60 evaluation prompts |
| `python src/evaluate.py generate --ckpt C --label L --prompts results/metrics/eval_prompts.jsonl` | fixed-setting generations + text statistics |
| `python src/evaluate.py rubric-sheet --outputs pre=... sft=...` / `rubric-score ...` | blind rubric evaluation |
| `python src/benchmark.py --configs configs/baseline.yaml configs/efficient_attention.yaml --label L` | throughput, latency, memory |
| `python src/make_figures.py loss/bench/bars ...` | report figures from saved results |
| `python src/log_experiment.py --run-id ... --member ... --stage ...` | append to the experiment log |

Protocol: `report/evaluation_protocol.md` · rubric: `report/instruction_rubric.md` · results:
`report/results.md` · references: `report/references.md`.


## Submission completion guide

Follow [`report/submission_completion_guide.md`](report/submission_completion_guide.md) for the blind-scoring handoff, post-scoring aggregation commands, team review/rehearsal checklist, and final local verification steps.

## Submission status

- Automated verification on the merged `main` branch: **67 tests passed** locally (Python 3.11.9, Windows). Re-run the tests before final submission.
- Dependency verification: `python -m pip check` reported **No broken requirements found**.
- Final experiment metrics and report tables are documented under `results/metrics/` and `report/`.
- Blind rubric evaluation of all 120 responses (60 per model) was completed by an automated LLM judge (not a human rater); no human evaluation was carried out. The scored data is stored in `results/metrics/rubric_sheet_scored.csv`.
- Aggregated rubric scores and failure-tag analysis are documented in `report/results.md` and `report/final_report.md`.
- The private model-to-response mapping remains in `results/metrics/rubric_key.json` and must not be committed.
- **Remaining before final submission:** team review of the results, presentation rehearsal, and final verification of the repository and deliverables.

The rubric is a single-rater assessment by an automated LLM judge, not a human evaluation; inter-rater reliability was not measured. Its findings should be interpreted as descriptive rather than statistically conclusive.