# Results

> Every number in this file is copied from a file in `results/metrics/` or `experiments/experiment_log.csv`
> (source given next to each table). Cells stay **TBD** until the corresponding run exists — nothing is
> estimated. Hardware for every table: _TBD (GPU, PyTorch version)_.

## E1 — Does pre-training work?

Source: `results/metrics/<run>_perplexity.json`, `<run>_train_log.csv`. Figure: `results/figures/pretrain_loss.png`.

| Model | Params | Train tokens | Final train loss | Val loss | Val perplexity | Test perplexity | Training time |
|---|---|---|---|---|---|---|---|
| Uniform baseline (ln V) | — | — | — | ln 8000 = 8.987 | 8000 | 8000 | — |
| Pre-trained, standard attention | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Interpretation: _TBD — does val loss track train loss? any overfitting? what do the samples at fixed
checkpoints show?_

Sample generations (raw story prompts, greedy): _TBD, copied from `<run>_generations.jsonl`._

## E2 — Does instruction fine-tuning help?

Source: `<run>_sft_loss.json`, `<run>_generation_summary.json`, `rubric_summary.json`. Same 60 prompts and
the same decoding settings for both rows.

| Model | Response NLL ↓ | Response ppl ↓ | Rubric total /8 ↑ | Relevance | Format | Answers | Coherence | EOS rate | Echo (3-gram copy) ↓ | Repetition ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| Pre-trained | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Fine-tuned (SFT) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Before/after examples: _TBD (3 prompts, both outputs verbatim)._

## E3 — Standard vs efficient (linear) attention

Quality source: `<run>_perplexity.json`. Efficiency source: `<label>_benchmark.json`
(figures `bench_forward_latency.png`, `bench_forward_memory.png`, `bench_train_throughput.png`,
`bench_gen_latency.png`).

| Variant | Params | Val ppl | Test ppl | Training time | Train tokens/s | Train peak mem (MB) | Gen ms/token |
|---|---|---|---|---|---|---|---|
| Standard (softmax) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Linear (Katharopoulos et al.) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Sequence-length sweep (forward, batch 16): _TBD table from `<label>_benchmark_forward.csv`._

Unavoidable differences between the two runs: _TBD (or "none")._

Interpretation: _TBD — where does linear attention become faster/leaner (if at all)? what does it cost in
perplexity? is the difference larger than run-to-run noise?_

## E4 — Failure analysis

Source: `rubric_sheet.csv` notes column.

| Failure tag | Pre-trained | Fine-tuned | Example (prompt id) | Plausible cause |
|---|---|---|---|---|
| echo | TBD | TBD | | |
| loop | TBD | TBD | | |
| story-mode | TBD | TBD | | |
| off-topic | TBD | TBD | | |
| truncated | TBD | TBD | | |
| factual-error | TBD | TBD | | |

## Limitations

_TBD — at minimum: one random seed per configuration (unless repeated), 60 prompts with a single rater,
small model and data budget, free-tier GPU timing noise, ROUGE-L is a weak proxy for instruction quality._
