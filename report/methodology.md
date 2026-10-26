# Methodology

## 1. Data and tokenizer — *owner: Member 2 (Haneesh)*

_To be written by Member 2: TinyStories subset size and selection, splits and seeds, tokenizer type and
vocabulary size, special tokens, sequence construction; Dolly cleaning, splits, prompt template and
response-only loss masking; dataset licences._

## 2. Model and pre-training — *owner: Member 1 (Akash)*

_To be written by Member 1: decoder-only architecture (embeddings, positional encoding, blocks, attention,
FFN, residuals, LayerNorm, output head), parameter count, initialisation, optimiser and schedule, token
budget, checkpointing; the linear-attention variant and how its causal form is computed._

## 3. Instruction fine-tuning — *owner: Member 2 (Haneesh)*

_To be written by Member 2: which checkpoint is fine-tuned, hyper-parameters, number of steps/epochs,
early-stopping rule._

## 4. Evaluation — *owner: Member 3 (Samvith)*

The full protocol, frozen before the final runs, is `report/evaluation_protocol.md`; this section
summarises it.

**Held-out language modelling.** For a held-out token stream x₁…x_N the model's quality is the mean
negative log-likelihood NLL = −(1/N) Σ log p(x_t | x_<t) in nats per token, and perplexity
PPL = exp(NLL). Perplexity is the effective number of equally likely next tokens the model is choosing
between; a model that predicts uniformly over a vocabulary of V tokens has PPL = V, which is the
reference point we report. The stream is cut into windows of the context length; every token is scored
exactly once (stride = context length), and a second pass with stride = context/2 checks sensitivity to
the amount of history each token sees. Perplexities are compared only between models that share the same
tokenizer and evaluation stream, because perplexity per token depends on how text is split into tokens.

**Instruction following.** The pre-trained and fine-tuned checkpoints are evaluated on the same 60
held-out Dolly prompts in the same template with identical, deterministic (greedy) decoding. We report
(i) the response-only NLL on the Dolly test split, where prompt and padding tokens are excluded exactly as
in SFT; (ii) automatic statistics of the generations — EOS-termination, empty responses, distinct-n,
repeated-4-gram rate, 3-gram copy rate from the prompt, ROUGE-L against the human reference; and
(iii) a blind human rubric with four 0–2 criteria (`report/instruction_rubric.md`). Lower perplexity
alone is not taken as better instruction following.

**Efficiency.** Training throughput, inference forward latency over sequence lengths 128–2048, generation
latency per token and peak GPU memory are measured with `src/benchmark.py` under identical settings for
both attention variants (same GPU, batch, dtype, seed, warm-up and repeat counts; CUDA-synchronised timing;
median of repeats). Because speed and memory do not depend on learned weights, they are measured on freshly
initialised models of each configuration; quality is measured on the trained checkpoints.

**Controlled comparison.** The two variants differ only in `model.attention`; every other setting is
shared through `configs/baseline.yaml` and `configs/efficient_attention.yaml`. Any unavoidable difference
is reported with the results.

**Reproducibility.** Every run is recorded in `experiments/experiment_log.csv` with its configuration file,
seed, git commit, hardware and result file; all figures are generated from saved results by
`src/make_figures.py`.
