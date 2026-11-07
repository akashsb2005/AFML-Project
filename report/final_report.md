# Efficient Training and Instruction Fine-Tuning of Small Language Models

**Team:** Akash Shivanand Bagoji (model and pre-training), Haneesh (data, tokenizer and fine-tuning),
Samvith Vaddiparthi (evaluation, experiments and report)

> **Draft.** Cells marked **TBD** wait for the final runs. The efficiency results in §5.3 are final measurements.
> Every number comes from `results/metrics/` or `experiments/experiment_log.csv`. The detailed tables are in
> `report/results.md`.

## Abstract

We built a decoder-only Transformer with 5,273,088 parameters from randomly initialised weights, pre-trained it on
TinyStories, and fine-tuned it on Databricks Dolly 15K with a response-only loss. We also compared standard causal
attention with the causal linear attention of Katharopoulos et al. On a Tesla T4, linear attention was slower and
used more memory at every sequence length we measured (128–2048): 0.35–0.40× the training throughput and 2.2–2.5×
the peak training memory of standard attention. Its forward time did scale linearly, and the latency gap shrank
from 2.52× at 128 tokens to 1.08× at 2048. **TBD: final perplexity and fine-tuning results.**

## 1. Introduction

Large language models are expensive to train, which makes it hard to study the whole pipeline hands-on. Eldan and
Li [3] showed that very small models can produce coherent English when the training text is simple. We use that
idea to build every stage ourselves, without pre-trained weights, and ask three questions:

* **E1.** Does pre-training work? Does validation loss fall well below the uniform baseline, without overfitting?
* **E2.** Does instruction fine-tuning help? Does the same model become better at following unseen instructions?
* **E3.** Does efficient attention improve the trade-off between quality and cost at the sizes we can train?

A fourth, descriptive experiment (**E4**) records where and why the model fails. The evaluation protocol was written
before the final runs (`report/evaluation_protocol.md`).

## 2. Background

**Causal self-attention.** Scaled dot-product attention [1] computes softmax(QKᵀ/√d)V, with future positions masked
so that a token can only attend to earlier tokens. The score matrix is T × T per head, so time and memory grow as
O(T²) in the sequence length T.

**Linear attention.** Katharopoulos et al. [2] replace the softmax with a positive feature map φ(x) = elu(x) + 1,
which gives φ(Q)(φ(K)ᵀV) up to normalisation. The bracket no longer depends on T × T scores. In causal form it
becomes a running sum of φ(k)vᵀ over positions, with cost O(T·d²), where d is the head dimension. Linear attention
should therefore win only when T is large relative to d².

**Supervised fine-tuning.** The pre-trained model continues training on instruction and response pairs [10]. Labels
on prompt and padding tokens are ignored, so only the response and its end-of-sequence token contribute to the loss.

**Perplexity.** Perplexity is exp(mean next-token negative log-likelihood), roughly the number of equally likely next
tokens the model is choosing between. A uniform model over 8,000 tokens has NLL = ln 8000 = 8.987 and perplexity
8,000. It is comparable only between models that share the same tokenizer and evaluation text, and low perplexity
does not by itself mean a model follows instructions.

## 3. Method

### 3.1 Data and tokenizer

TinyStories is used for pre-training and tokenizer training; Dolly 15K is used for fine-tuning. The tokenizer is a
ByteLevel BPE with 8,000 tokens, trained only on the TinyStories training split, with `[PAD]`, `[UNK]`, `[BOS]`,
`[EOS]` and `[SEP]` as special tokens. Dolly prompts are normalised and deduplicated before the 90/5/5 split, which
removes 203 rows and leaves 13,317 / 739 / 741 examples with no prompt in two splits. The template is
`Instruction: …\n[Context: …\n]Response:`.

At a 256-token context, prompts that do not fit are skipped (2,620 of 13,317 train and 150 of 741 test examples)
and over-long responses are cut at the tail. **TBD: final TinyStories subset size, confirmed by Member 2.**

### 3.2 Model

Both variants have 4 pre-norm decoder blocks, 4 heads, d_model 256, a 4× GELU feed-forward layer, learned position
embeddings, tied input and output embeddings, and dropout 0.1, for 5,273,088 parameters at context length 256. Weights
are randomly initialised (normal, std 0.02). The variants differ only in attention. Linear attention adds no
parameters, and it has no attention-weight dropout because there are no attention weights to drop.

### 3.3 Training

Pre-training uses AdamW, lr 3×10⁻⁴, weight decay 0.1, 200 warm-up steps, batch 32 and context 256, with `src/training.py`
shared by both variants. Fine-tuning continues from the pre-trained checkpoint with AdamW. **TBD: final step counts
and the fine-tuning learning rate.**

### 3.4 Evaluation

* **Language modelling:** token-weighted NLL and perplexity on held-out stories, built exactly like the training
  blocks. The test split is scored once per final checkpoint.
* **Instruction following:** response-only NLL on Dolly test, plus 60 fixed held-out prompts (seed 2026, balanced
  across categories) decoded greedily with the same settings for every model. The outputs are scored blind on four
  0–2 criteria: relevance, format, answering rather than echoing, and coherence. Automatic statistics cover EOS rate,
  4-gram repetition, 3-gram prompt copying and ROUGE-L.
* **Efficiency:** tokens per second, forward latency across T = 128…2048, generation latency and peak GPU memory, with
  `src/benchmark.py`. It uses CUDA-synchronised timing, discarded warm-up and medians of repeats, on random weights
  because speed does not depend on learned values.

## 4. Experimental control

The two attention configurations differ only in `model.attention`. Tokenizer, data split, optimiser, schedule, batch
size, step count, seed, GPU type and PyTorch version are shared, and anything that could not be matched is stated
with the result. Every run is logged in `experiments/experiment_log.csv`. Figures are generated from saved results by
`src/make_figures.py`.

## 5. Results

### 5.1 E1: pre-training

| Model | Train tokens | Val loss | Val perplexity | Test perplexity | Training time |
|---|---|---|---|---|---|
| Uniform baseline | — | 8.987 | 8,000 | 8,000 | — |
| Standard attention | TBD | TBD | TBD | TBD | TBD |

A shorter rehearsal (3,000 steps, context 128, not a final result) reached validation perplexity 15.42, with validation
loss still falling, so the pipeline clearly learns.

### 5.2 E2: instruction fine-tuning

| Model | Response NLL | Rubric /8 | EOS rate | Prompt copy | Repetition |
|---|---|---|---|---|---|
| Pre-trained | TBD | TBD | TBD | TBD | TBD |
| Fine-tuned | TBD | TBD | TBD | TBD | TBD |

A rehearsal on Dolly validation (not a final result; `results/metrics/rehearsal_sft/`) gave response NLL 6.96 before
fine-tuning and 4.18 after 4 epochs at lr 3×10⁻⁴. Greedy 4-gram repetition rose from 0.27 to 0.65 on the same 60 prompts,
while sampling (temperature 0.8, top-k 50) brought it to 0.02.

### 5.3 E3: standard vs linear attention

Measured on a Colab Tesla T4 (15 GB, PyTorch 2.11.0).

| Batch × T | Standard tokens/s | Linear tokens/s | Linear / standard | Standard peak MB | Linear peak MB |
|---|---|---|---|---|---|
| 8 × 128 | 69,143 | 27,406 | 0.40× | 293 | 641 |
| 8 × 256 | 73,804 | 28,431 | 0.39× | 537 | 1,195 |
| 32 × 128 | 84,811 | 29,325 | 0.35× | 928 | 2,301 |
| 32 × 256 | 79,198 | 29,471 | 0.37× | 1,913 | 4,513 |

| T | Standard ms | Linear ms | Linear / standard | Standard peak MB | Linear peak MB |
|---|---|---|---|---|---|
| 128 | 4.4 | 11.1 | 2.52× | 95 | 258 |
| 256 | 8.9 | 21.7 | 2.43× | 128 | 455 |
| 512 | 21.3 | 43.7 | 2.05× | 196 | 849 |
| 1024 | 55.1 | 87.1 | 1.58× | 383 | 1,637 |
| 2048 | 162.5 | 175.7 | 1.08× | 1,219 | 3,213 |

Over 4 identical runs at batch 8 × 128, standard attention gave 68,325 ± 903 tokens/s and linear attention
27,094 ± 226, so the difference is far larger than run-to-run noise. Generation took 3.43 ms/token (standard) and
4.05 ms/token (linear).

| Variant | Params | Val perplexity | Test perplexity | Training time |
|---|---|---|---|---|
| Standard | 5,273,088 | TBD | TBD | TBD |
| Linear | 5,273,088 | TBD | TBD | TBD |

### 5.4 E4: failure analysis

**TBD: tag counts from the rubric sheet (story-mode, loop, echo, off-topic, truncated, factual error) with real examples.**

## 6. Discussion

**Efficient attention was not more efficient at our sizes.** Each doubling of T multiplied standard attention's forward
time by 2.03 rising to 2.95, approaching the 4× of O(T²), while linear attention stayed near 2.0×. So the scaling matches the theory, but the constant factor dominates here.

A likely cause is that the causal form stores a 64 × 64 state (4,096 numbers) per token and head, while softmax
attention stores T scores per token and head. Below T ≈ 4,096 the linear form therefore moves more data. If the trend continued, linear attention would become faster somewhere beyond 2048 tokens, but we did not measure that.
At our context length of 256, standard attention is the better choice on this GPU.

**Lower loss did not automatically give better answers.** In the rehearsal, fine-tuning moved outputs from story text to answer-shaped text and
lowered response loss, but greedy decoding looped more. The final rubric decides whether answers actually improved.

**TBD: what the perplexity gap adds to this, and whether fine-tuning improved response NLL and rubric scores together.**

## 7. Limitations

* One seed per configuration unless stated, and one rater for the 60 prompts.
* A small model and data budget, and a free-tier GPU.
* Efficiency is measured for this implementation. A chunked or fused kernel, or a recurrent generation loop, would
  change the constant factors. Neither variant uses a KV cache.
* About 20% of Dolly examples do not fit a 256-token context and are skipped, so long-context tasks such as summarisation
  and closed QA are under-represented.
* ROUGE-L is a weak signal for instruction quality, because a short correct answer can score low.

## 8. Conclusion

**TBD after the final runs.** So far: linear attention scales as predicted, but costs more than standard attention at
every length we measured, and at our context length it costs well over twice as much in both time and memory.

## Individual contributions

* **Akash Shivanand Bagoji:** model, both attention variants, pre-training, generation.
* **Haneesh:** preprocessing, tokenizer, datasets, loss masking, fine-tuning.
* **Samvith Vaddiparthi:** evaluation protocol and scripts, benchmarks, experiment log, analysis, report, presentation plan.

## References

See `report/references.md`. Numbers in brackets refer to that list.
