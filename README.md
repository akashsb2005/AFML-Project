# Efficient Training and Instruction Fine-Tuning of Small Language Models

AFML course project — a decoder-only Transformer pre-trained **from scratch** on TinyStories, instruction
fine-tuned on Databricks Dolly 15K, and compared with an efficient (linear) attention variant.

| Member | Role | Owns |
|---|---|---|
| Akash | Member 1 — model & pre-training | `src/model.py`, `src/attention.py`, `src/pretrain.py`, `src/generate.py` |
| Haneesh | Member 2 — data, tokenizer & SFT | `src/tokenizer.py`, `src/dataset.py`, `src/preprocess.py`, `src/sft.py` |
| Samvith | Member 3 — evaluation, experiments & report | `src/evaluate.py`, `src/benchmark.py`, `src/model_api.py`, `src/select_eval_prompts.py`, `src/make_figures.py`, `src/log_experiment.py`, `experiments/`, `results/`, `report/` |

Interfaces between the parts: **`docs/interfaces.md`** (please agree on it before integrating).

## Setup

```bash
pip install -r requirements.txt
python -m pytest -q          # contract tests run on src/model.py automatically once it exists
```

GPU runs: `notebooks/member3_colab.ipynb` (evaluation + benchmarks). Large files (`data/processed/`,
`artifacts/tokenizer/`, checkpoints) live in the shared Drive folder, not in git.

## Member 3 tools

| Command | Purpose |
|---|---|
| `python src/evaluate.py perplexity --ckpt C --label L --stories data/processed/tiny_stories_validation.jsonl` | held-out NLL / perplexity (same blocks as training) |
| `python src/evaluate.py sft-loss --ckpt C --label L --data data/processed/dolly_test.jsonl --exclude data/processed/dolly_train.jsonl data/processed/dolly_validation.jsonl` | response-only NLL (overlapping prompts dropped) |
| `python src/select_eval_prompts.py --test ... --exclude ... --tokenizer ...` | freeze the 60 evaluation prompts |
| `python src/evaluate.py generate --ckpt C --label L --prompts results/metrics/eval_prompts.jsonl` | fixed-setting generations + text statistics |
| `python src/evaluate.py rubric-sheet --outputs pre=... sft=...` / `rubric-score ...` | blind human evaluation |
| `python src/benchmark.py --configs configs/baseline.yaml configs/efficient_attention.yaml --label L` | throughput, latency, memory |
| `python src/make_figures.py loss/bench/bars ...` | report figures from saved results |
| `python src/log_experiment.py --run-id ... --member ... --stage ...` | append to the experiment log |

Protocol: `report/evaluation_protocol.md` · rubric: `report/instruction_rubric.md` · results:
`report/results.md` · references: `report/references.md`.
