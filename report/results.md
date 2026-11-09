# Results

> Every number in this file is copied from a file in `results/metrics/` or `experiments/experiment_log.csv`
> (source given next to each table). Cells stay **TBD** until the corresponding run exists — nothing is
> estimated. Hardware is stated with each table.

## E1 — Does pre-training work?

Source: `results/metrics/<run>_perplexity.json`, `<run>_train_log.csv`. Figure: `results/figures/pretrain_loss.png`.

| Model | Params | Train tokens | Final train loss | Val loss | Val perplexity | Test perplexity | Training time |
|---|---|---|---|---|---|---|---|
| Uniform baseline (ln V) | — | — | — | ln 8000 = 8.987 | 8000 | 8000 | — |
| Pre-trained, standard attention | 5,273,088 | 56.6 M | 2.100 | 2.151 | 8.60 | 8.51 | 1,419 s |

Final run: TinyStories 50,000-story sample (53,668 training blocks of 256 tokens), 10,000 steps, batch 32, lr 3e-4
(constant after 200 warm-up steps), seed 42, Tesla T4, commit f69db63. Train tokens count non-padding tokens. The table's
val loss is the full validation split scored by `src/evaluate.py` (526,015 tokens). The training log's periodic val loss
(2.139) uses a fixed subset.

Pre-training works: perplexity falls from 8,000 (uniform) to 8.6. Validation loss tracks training loss and was still
falling at step 10,000 (2.43 at step 4,000, 2.26 at 6,500, 2.17 at 9,000, 2.14 at 10,000). The best checkpoint is the
last one, so there is no sign of overfitting after about 6 passes over the sample. A longer run would probably go lower.

Sample generation (greedy, `pre_std_story_generations.jsonl`), "Once there was a little fish named Sally." →
"Sally was very curious and wanted to explore the world. One day, Sally decided to go on an adventure. She walked and
walked until she came across a big, scary monster. The monster was very big and scary. Sally was scared and ran away.
The monster was very friendly. …" The text is fluent and in TinyStories style but not always consistent: the monster is
"scary" and then "very friendly". None of the 10 story continuations reached EOS within 64 tokens, and 4-gram repetition
averaged 0.067.


### Provisional — Member 3 pipeline rehearsal (not the final runs)

Source: `results/metrics/rehearsal/` (see its README for exact settings). Member 1's models with the two
review fixes, Member 2's data, 3,000 steps, batch 32, block 128, lr 3e-4, seed 42, Tesla T4. Figure:
`results/figures/rehearsal_pretrain_loss.png`.

| Model | Params | Steps | Final train loss | Val NLL (full val, 104,882 tokens) | Val perplexity | Training time |
|---|---|---|---|---|---|---|
| Uniform baseline | — | — | — | 8.987 | 8,000 | — |
| Standard attention | 5,273,088 | 3,000 | 2.753 | 2.736 | 15.42 | 163 s |

Pre-training clearly works: perplexity falls from 8,000 (uniform) to 15.4, and validation loss
is still falling at step 3,000 (no sign of overfitting yet; validation sits slightly below the training average
because training loss is averaged over the interval with dropout on). Story continuations are fluent
TinyStories-style text (`examples.json`), but greedy decoding loops on some prompts (4-gram repetition
0.10 on average over 10 openings, up to 0.70 on one) and none of the 64-token
continuations reached EOS.

## E2 — Does instruction fine-tuning help?

Source: `<run>_sft_loss.json`, `<run>_generation_summary.json`, `rubric_summary.json`. Same 60 prompts and
the same decoding settings for both rows.

| Model | Response NLL ↓ | Response ppl ↓ | Rubric total /8 ↑ | Relevance | Format | Answers | Coherence | EOS rate | Echo (3-gram copy) ↓ | Repetition ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| Pre-trained | 6.379 | 589 | pending | pending | pending | pending | pending | 0.33 | 0.000 | 0.256 |
| Fine-tuned (SFT) | 3.928 | 50.8 | pending | pending | pending | pending | pending | 0.55 | 0.030 | 0.453 |

Final run: standard attention; SFT from the pre-trained checkpoint above on Dolly train (13,317 examples, 2,620 skipped
because the prompt does not fit), 4 epochs, batch 4, lr 3e-4, 10,700 steps, 305 s on a T4. Response NLL covers the full
Dolly test split (591 scored examples, 51,075 response tokens; 150 skipped, 249 truncated). Generation statistics cover the
60 fixed test prompts, decoded greedily with 128 new tokens. Rubric columns are **pending** the blind scoring of
`rubric_sheet.csv`.

The same 60 prompts with sampling (temperature 0.8, top-k 50, seed 1234; `*_sampled_generation_summary.json`):

| Model | EOS rate | Repetition (4-gram) ↓ | Prompt copy (3-gram) | ROUGE-L vs reference | Outputs with repetition > 0.5 |
|---|---|---|---|---|---|
| Pre-trained, greedy | 0.33 | 0.256 | 0.000 | 0.078 | 11 / 60 |
| SFT, greedy | 0.55 | 0.453 | 0.030 | 0.139 | 32 / 60 |
| Pre-trained, sampled | 0.43 | 0.015 | 0.000 | 0.087 | 0 / 60 |
| SFT, sampled | 0.90 | 0.035 | 0.034 | 0.159 | 1 / 60 |

Fine-tuning works on the loss: response NLL falls from 6.38 to 3.93, so response perplexity falls from 589 to 51. SFT
validation loss was still falling at the last step (4.98 at step 500, 4.08 at 7,000, 3.88 at 10,700). Outputs change from
children's stories to answer-shaped text: they stop more often (EOS 0.33 → 0.55 greedy, 0.43 → 0.90 sampled) and overlap the
references more (ROUGE-L 0.078 → 0.139). But greedy SFT outputs loop much more: 32 of 60 have 4-gram repetition above 0.5,
against 11 before. Sampling removes almost all looping, but the content is usually wrong or garbled (examples below).

Before/after examples (greedy unless stated, `pre_generations.jsonl`, `sft_generations.jsonl`, `sft_sampled_generations.jsonl`):

* `dolly_test_00577` "What origins and destinations does Delta Vacations offer?"
  Pre-trained: "Let's go to the beach and see the sea. It's so pretty and fun!" Mrs. Jones said … (story mode).
  SFT: "Delta Vacations is a highly highly highly highly …" (on topic, then loops).
* `dolly_test_00134` "Give me a list of pros and cons of buying a bigger house"
  Pre-trained: "Quack, twice to the family, who is always here to help others. One day …"
  SFT: "1. Compically, you can consider a bigger house. 2. Compically, you can consider a bigger house. 3. …"
  (a numbered list, the right format, but one item repeated).
* `dolly_test_00116` "Classify each of the following products as an Apple, Amazon, or Microsoft product: …"
  Pre-trained: "40, 4, 40, 40, 40, …". SFT greedy: "Apple: Apple: Apple: …". SFT sampled: "Apple: Angeles: Amazon Chole, …"
  (label format learned, assignments wrong).


### Provisional — pre-trained model only (no SFT checkpoint exists yet)

Source: `results/metrics/rehearsal/reh_std_sft_loss.json`, `reh_std_inst_generation_summary.json`. Context 128,
so the 60 prompts were selected with a 64-token prompt budget (`eval_prompt_ids_ctx128.json`).

| Model | Response NLL ↓ | Response ppl ↓ | EOS rate | Prompt copy (3-gram) | Repetition (4-gram) | ROUGE-L vs reference |
|---|---|---|---|---|---|---|
| Pre-trained (standard) | 7.077 | 1184 | 0.50 | 0.00 | 0.38 | 0.086 |

The pre-trained model has never seen instructions: Dolly responses are far less likely to it than stories
(response perplexity 1184 vs story perplexity 15.4), and all 6 outputs inspected so far are
children's-story text (see E4; the full count comes from the rubric). This is the baseline the SFT model must beat. 413 of
737 test examples did not fit the 128-token context and were truncated by the DollySFTDataset rule.

### Provisional — pre-trained vs fine-tuned, merged pipeline (validation split)

Source: `results/metrics/rehearsal_sft/` (see its README). Standard attention, context 256, 3,000 pre-training steps
(validation perplexity 14.56), Apple M1. Scored on Dolly **validation** so the test split stays untouched: response NLL on
606 examples (133 skipped because the prompt does not fit), generations on 60 validation prompts. One seed, one run per setting.

| Model | Response NLL ↓ | Response ppl ↓ |
|---|---|---|
| Pre-trained | 6.958 | 1,052 |
| SFT, lr 2e-5, 1 epoch | 5.338 | 208 |
| SFT, lr 1e-4, 1 epoch | 4.836 | 126 |
| SFT, lr 3e-4, 1 epoch | 4.635 | 103 |
| SFT, lr 3e-4, 4 epochs | 4.175 | 65 |

| Model | Decoding | EOS rate | Repetition (4-gram) ↓ | Prompt copy (3-gram) | ROUGE-L vs reference |
|---|---|---|---|---|---|
| Pre-trained | greedy | 0.40 | 0.27 | 0.00 | 0.084 |
| SFT, 1 epoch | greedy | 0.32 | 0.64 | 0.03 | 0.088 |
| SFT, 4 epochs | greedy | 0.27 | 0.65 | 0.02 | 0.118 |
| Pre-trained | sampled | 0.37 | 0.01 | 0.00 | 0.091 |
| SFT, 1 epoch | sampled | 0.77 | 0.03 | 0.00 | 0.125 |
| SFT, 4 epochs | sampled | 0.73 | 0.02 | 0.01 | 0.136 |

Fine-tuning clearly lowers response loss, and validation loss was still falling at epoch 4 (per-epoch 4.61, 4.36, 4.23,
4.14), so these settings did not reach overfitting. Lower loss did not make greedy outputs better overall: repetition
more than doubles, because the model now loops on answer-style phrases rather than story text. Sampling removes nearly all of
the looping and raises the EOS rate, but the sampled answers we read are often fluent and factually wrong. These numbers
cover loops and termination only; whether answers improved needs the rubric.

Example (`dolly_val_00202`, "Identify which instrument is string or woodwind: Tambura, Tubax"; reference "Tubax is woodwind,
Tambura is string."): SFT 4 epochs, greedy: "Tuba is string, Tuba is woodwind." It has the right format and wrong content.
More in `examples.json`.

## E3 — Standard vs efficient (linear) attention

### Efficiency (measured)

Measured with `src/benchmark.py` on a Colab **Tesla T4** (15 GB, PyTorch 2.11.0, CUDA 13.0), both variants
built from the same configuration (4 layers, 4 heads, d_model 256, vocab 8000, tied embeddings, dropout 0.1)
with random weights, identical batch, sequence length, seed, warm-up (3) and repeat counts; median of
≥ 20 timed repeats. Both variants have **5,273,088 parameters** (linear attention adds none). Tables are
generated by `src/summarize_benchmarks.py`; figures `results/figures/bench_t4_*.png`.

### Run-to-run variation (batch 8, T = 128, 4 identical runs)

| Attention | train tokens/s (mean ± sd) | min–max | gen ms/token (mean) |
|---|---|---|---|
| standard | 68,325 ± 903 | 67,090–69,143 | 3.43 |
| linear | 27,094 ± 226 | 26,879–27,406 | 4.05 |

### Training step (forward + backward + AdamW)

| Batch | T | Standard tokens/s | Linear tokens/s | Linear / standard | Standard peak MB | Linear peak MB |
|---|---|---|---|---|---|---|
| 8 | 128 | 69,143 | 27,406 | 0.40× | 293 | 641 |
| 8 | 256 | 73,804 | 28,431 | 0.39× | 537 | 1,195 |
| 32 | 128 | 84,811 | 29,325 | 0.35× | 928 | 2,301 |
| 32 | 256 | 79,198 | 29,471 | 0.37× | 1,913 | 4,513 |

### Inference forward pass vs sequence length (batch 8, context raised to 2048 for both)

| T | Standard ms | Linear ms | Linear / standard | Standard ×/doubling | Linear ×/doubling | Standard peak MB | Linear peak MB |
|---|---|---|---|---|---|---|---|
| 128 | 4.4 | 11.1 | 2.52× | — | — | 95 | 258 |
| 256 | 8.9 | 21.7 | 2.43× | 2.03 | 1.96 | 128 | 455 |
| 512 | 21.3 | 43.7 | 2.05× | 2.39 | 2.01 | 196 | 849 |
| 1024 | 55.1 | 87.1 | 1.58× | 2.59 | 1.99 | 383 | 1,637 |
| 2048 | 162.5 | 175.7 | 1.08× | 2.95 | 2.02 | 1,219 | 3,213 |

**Interpretation.**
* On this GPU, linear attention was **slower and used more memory at every length we measured
  (128–2048)**: 0.35–0.40× the standard model's training throughput and 2.2–2.5× its peak training memory.
  These gaps are far larger than run-to-run variation (≈ 1.3 % / 0.8 % standard deviation over 4 identical runs).
* The scaling behaves as the theory predicts. Each doubling of T multiplies standard attention's forward time by
  2.03 → 2.95 (approaching the 4× of O(T²)), while linear attention stays at ≈ 2.0× (O(T)). The latency gap therefore
  shrinks from 2.52× at T = 128 to 1.08× at T = 2048; if the trend continues, linear attention would become faster
  somewhere beyond 2048 tokens. This is an extrapolation, not measured.
* Plausible cause of the constant-factor overhead: the causal implementation materialises the running state
  φ(k)vᵀ — a 64 × 64 matrix (4,096 numbers) per token and head — whereas softmax attention stores T scores per
  token and head. Below T ≈ 4,096 the linear form therefore moves *more* data, and the extra einsum/cumsum
  kernels are memory-bound. A chunked or fused kernel would reduce this constant.
* Generation (no KV cache in either variant, full re-encoding per token, prompt 32 + 64 new tokens):
  3.43 vs 4.05 ms/token. The recurrent form of linear attention would allow constant cost per token, but it
  is not implemented here.
* The context-2048 sweep run measured lower training throughput for the standard model (52,259 tokens/s),
  probably because of its larger position table (2,048 rows) and only 10 timed steps. It is used for the
  latency/memory sweep only, not for throughput claims.

### Quality

| Variant | Params | Val ppl | Test ppl | Training time (same steps) |
|---|---|---|---|---|
| Standard (softmax) | 5,273,088 | 8.60 | 8.51 | 1,419 s |
| Linear (Katharopoulos et al.) | 5,273,088 | 8.89 | 8.78 | 3,319 s (2.34×) |

Final runs: identical data, 10,000 steps, batch 32, lr 3e-4, seed 42, the same T4 session, commit f69db63 (sources:
`pre_std_*_perplexity.json`, `pre_lin_*_perplexity.json`, `*_train_log.csv`). After the same SFT recipe on each:

| Variant | SFT time | Dolly test response NLL ↓ | Response ppl ↓ | EOS rate (greedy) | Repetition (greedy) ↓ | ROUGE-L (greedy) |
|---|---|---|---|---|---|---|
| Standard | 305 s | 3.928 | 50.8 | 0.55 | 0.453 | 0.139 |
| Linear | 532 s (1.74×) | 4.106 | 60.7 | 0.58 | 0.438 | 0.140 |

At equal steps, linear attention ends with 3.1% higher test perplexity (8.78 vs 8.51) after taking 2.34× as long to train.
It also ends with higher response NLL after SFT (4.11 vs 3.93). Both pre-training curves were still falling, and the gap
narrowed from 9% in the 3,000-step rehearsal to 3% at 10,000 steps, so the gap at convergence is unknown. Story
continuations look alike (4-gram repetition 0.067 standard vs 0.074 linear). One seed per variant. On this GPU, at
context 256, standard attention is better on quality, time and memory.

*Provisional rehearsal (same data, 3,000 identical steps, T4; `results/metrics/rehearsal/`):*

| Variant | Val NLL | Val perplexity | Training time (3,000 steps) | Story 4-gram repetition |
|---|---|---|---|---|
| Standard | 2.736 | 15.42 | 163 s | 0.100 |
| Linear | 2.822 | 16.81 | 456 s | 0.077 |

At an equal number of steps, linear attention reached a 9% higher validation perplexity and
took 2.8× longer (consistent with the 0.35–0.40× throughput measured above). Both curves were
still falling, so the gap at convergence is unknown; one seed only.

Unavoidable differences
between the two runs: linear attention has no attention-weight dropout (not defined for the linear form).

## E4 — Failure analysis

Source: `rubric_sheet.csv` notes column (final); provisional examples from `results/metrics/rehearsal/examples.json`.

| Failure tag | Pre-trained | Fine-tuned | Example (prompt id) | Plausible cause |
|---|---|---|---|---|
| story-mode | count pending (rubric); every one of the 12 final outputs read so far is story text | none of the 12 read | dolly_test_00577 "What origins and destinations does Delta Vacations offer?" → "Let's go to the beach and see the sea…" | pre-training data contains only children's stories |
| loop | 11 / 60 greedy outputs with 4-gram repetition > 0.5 | 32 / 60 greedy; 1 / 60 sampled | dolly_test_00482 "Give me some ideas around improving my writing skills" → "practice, practice, practice, …" | greedy decoding of a small model with no repetition penalty; SFT makes short answer phrases very likely |
| echo | mean 3-gram prompt copy 0.000 | 0.030 greedy, 0.034 sampled | — | rare |
| off-topic | count pending (rubric) | count pending (rubric) | | |
| no EOS within 128 tokens | 40 / 60 greedy | 27 / 60 greedy; 6 / 60 sampled | | loops never end; sampling breaks them |
| does not fit / truncated | 150 / 741 test prompts skipped, 249 responses cut at 256 tokens | same | | context length 256 |
| factual-error | count pending (rubric) | most of the 12 sampled outputs read | dolly_test_00374 "Who is Richard Wayne Bogosian?" → "Brong Wayne is an American, politican's American diplomat in 199597…" | 5 M parameters and children's-story pre-training hold almost no world knowledge |

The "read so far" counts come from the first 12 of the 60 prompts, read by Member 3 while preparing this file. They are
not the blind rubric, which will replace them.

Also observed: the tokenizer reproduces mis-encoded quotation marks present in the training text
(`â€œ` instead of “), and several TinyStories validation stories share the same first six words
("Once upon a time, there was"), so greedy decoding produces identical continuations for them. The final
story-prompt set should use distinct openings.

## Limitations

* One random seed per configuration, and one rater for the 60 prompts.
* A small model and data budget: 50,000 TinyStories stories and 10,000 steps, with both models still improving at the end.
  The learning rate stays constant after warm-up, with no decay.
* Free-tier GPU timing noise of about 1% across identical runs.
* About 20% of Dolly prompts do not fit the 256-token context and are skipped, so long-context tasks are under-represented.
* ROUGE-L is a weak proxy for instruction quality.
* The tracked tokenizer may have been trained on a different TinyStories sample than this run's training split. This affects only
  the tokenizer, not model training.
