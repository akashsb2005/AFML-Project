# Evaluation protocol (Member 3)

Status: **draft v1 — to be frozen before the first final experiment** (plan §4 B, Task 1).
Any later change must be recorded in the change log at the bottom, with the reason.

## 1. Research questions and what would answer them

| # | Question | Evidence | Supports "yes" if |
|---|---|---|---|
| E1 | Does pre-training work? | train/val loss curves, held-out perplexity, story samples | val loss falls well below the uniform baseline ln V (≈ 8.99 nats for V = 8000) and keeps falling without diverging from train loss |
| E2 | Does instruction fine-tuning help? | response-only NLL on Dolly test, blind rubric scores, automatic text statistics, same prompts for both models | fine-tuned model has lower response NLL **and** higher rubric scores than the pre-trained checkpoint |
| E3 | Does efficient attention improve the quality/efficiency trade-off? | val/test perplexity, params, training time, tokens/s, inference latency, peak memory, sequence-length sweep | it is faster/leaner at some sequence length while the perplexity gap is reported honestly — "no improvement at T = 256" is a valid result |
| E4 | Where does the model fail? | failure taxonomy over the scored generations | — (descriptive) |

## 2. Data splits

All splits are produced by Member 2's `src/preprocess.py` (seed 42, 90/5/5) into `data/processed/`;
statistics in `tiny_stories_stats.json` and `dolly_stats.json`. TinyStories: 9,000 / 500 / 500 stories;
Dolly: 13,500 / 750 / 750 examples.

| Split | Used for | Never used for |
|---|---|---|
| TinyStories train | pre-training | evaluation |
| TinyStories validation | val loss during training, checkpoint selection, E1/E3 val perplexity | training |
| TinyStories test | **one** final perplexity number per final checkpoint (E1, E3) | training, checkpoint or hyper-parameter selection |
| Dolly train | SFT | evaluation |
| Dolly validation | SFT val loss, early stopping | final reporting |
| Dolly test | response NLL and the fixed evaluation prompts (E2, E4) | training, selection, prompt tuning |

Dolly itself contains duplicated prompts (143 prompts occur more than once, 346 rows), so a random split
puts some copies on both sides: 13 Dolly-test prompts also occur in train/validation (3 with an identical
response). Until Member 2 splits by prompt, `select_eval_prompts.py` and `evaluate.py sft-loss --exclude`
drop these 13 rows and report the count.

## 3. Metrics

**Language-model quality.** Token-weighted mean next-token negative log-likelihood (nats/token) over the
held-out stories; perplexity = exp(mean NLL). `evaluate.py perplexity --stories` encodes every story as
`[BOS] story [EOS]` and cuts it into blocks of the context length exactly like Member 2's
`TinyStoriesDataset` (verified identical), excluding padding — so the number is the same quantity as the
validation loss in training. Perplexities are only compared between models that share the **same
tokenizer and evaluation stories**.

**Instruction following (automatic).** On Dolly test: response-only NLL/perplexity (prompt and padding
tokens excluded, EOS included), and for generations: EOS-termination rate, empty-response rate, length,
distinct-1/2, 4-gram self-repetition rate, 3-gram prompt-copy rate, ROUGE-L F1 vs the human reference
(reported as a weak signal only — a short correct answer can score low).

**Instruction following (human).** Blind rubric (`report/instruction_rubric.md`), four criteria scored
0–2 (relevance, format, answers-not-echoes, coherence), total out of 8.

**Efficiency.** Parameter count; training throughput (tokens/s, median step time after warm-up, forward +
backward + AdamW); wall-clock training time of the full run (from the training log); inference forward
latency for a [16, T] batch at T ∈ {128, 256, 512, 1024, 2048}; autoregressive generation latency
(ms/token, prompt 32, 64 new tokens, shared decoding loop); peak allocated GPU memory
(`torch.cuda.max_memory_allocated`) for training and for each forward pass.

## 4. Fixed instruction-evaluation prompts

* 60 prompts sampled from Dolly test with seed 2026, round-robin over categories, restricted to prompts
  whose templated length ≤ context − 128 tokens (`select_eval_prompts.py`). Saved once to
  `results/metrics/eval_prompts.jsonl` and never edited afterwards. Prompts are identified by their line
  number in `dolly_test.jsonl` (`dolly_test_00157`, …). Because of the 128-token budget, categories with
  long context passages (closed QA, information extraction, summarization) are under-represented; this
  is reported as a limitation.
* Both the pre-trained and the fine-tuned checkpoint get the **same prompt text in the same SFT template**.
  The pre-trained model is additionally probed with 10 raw TinyStories-style openings (`--raw`) so that
  its genuine ability (story continuation) is also shown.

## 5. Controlled attention comparison (E3)

Identical between the two variants: tokenizer, data and split, `n_layer`, `n_head`, `d_model`,
`context_length`, dropout, optimiser, learning-rate schedule, batch size, number of training tokens, seed,
GPU type, PyTorch version, dtype. Only `model.attention` differs (`configs/*.yaml`). Any unavoidable
difference (for example a different free Colab GPU) is written into the results table.

Speed and memory are measured with random weights (`benchmark.py`), because they do not depend on the
learned values; quality is measured on the trained checkpoints. The sequence-length sweep uses
`--override context_length=2048` for both variants, so the comparison shows where (if anywhere) the
O(T) linear attention overtakes O(T²) softmax attention.

## 6. Measurement procedure

* Timing: `time.perf_counter()` around the operation with `torch.cuda.synchronize()` before reading the
  clock; 3 warm-up iterations discarded; median of ≥ 10 repeats reported.
* Memory: peak counter reset after warm-up; reported in MB (2^20 bytes); CUDA only.
* Every benchmark run records GPU name and PyTorch version in its JSON. Runs on different GPUs are never
  mixed in one comparison table.
* Each reported number is produced by a script into `results/metrics/` and logged in
  `experiments/experiment_log.csv`; report tables are copied from those files only.

## 7. Generation settings (identical for every model)

| setting | primary | secondary (diversity check) |
|---|---|---|
| decoding | greedy (temperature 0) | temperature 0.8, top-k 50 |
| max new tokens | 128 | 128 |
| stop | EOS token | EOS token |
| seed | — | 1234 |
| context | last `context_length` tokens | same |

All generations use the shared loop `model_api.generate`, not model-specific generation code.

## 8. Reporting rules

No metric is estimated, rounded up or reported from memory; failed or worse runs are reported, not
dropped. Test-split numbers are computed once per final checkpoint. Small sample sizes (60 prompts, one
rater) are stated as a limitation.

## Change log

| date | change | reason |
|---|---|---|
| 2026-10-24 | v1 drafted | |
| 2026-10-26 | data paths, per-story perplexity, Dolly duplicate handling | adapted to Member 2's implemented pipeline |
