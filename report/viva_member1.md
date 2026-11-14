# Member 1 Viva Preparation — Model Architecture and Pretraining

This note is a speaking guide for the Member 1 contribution. Explain the design first, then the implementation, experiment controls, measured results, and limitations.

## 1. Suggested opening

"My part of the project was to build the language model architecture and compare two causal attention mechanisms. I implemented a standard decoder-only Transformer and a linear-attention variant. Both were trained from randomly initialized weights on the same TinyStories data, with the same tokenizer, model size, optimizer settings, seed, and number of steps. This let us compare model quality and runtime while changing the attention mechanism."

## 2. What does the model do?

A decoder-only language model predicts the next token from the tokens that came before it.

For example, given:

`The little cat sat on the`

the model produces a probability distribution over the vocabulary for the next token. During training, the correct next token is compared with the model's prediction using cross-entropy loss. At inference, we select or sample a token and append it to the context.

## 3. Explain the architecture

The model contains:

1. **Token embedding:** converts each token ID into a 256-dimensional vector.
2. **Positional embedding:** adds information about token order.
3. **Four decoder blocks:** each contains layer normalization, causal multi-head attention, a residual connection, another normalization, and a feed-forward network.
4. **Feed-forward network:** expands 256 dimensions to 1,024 with GELU, then projects back to 256.
5. **Final normalization and language-model head:** maps the hidden representation to 8,000 vocabulary logits.

There are four attention heads, so each head has dimension 256 / 4 = 64. The input embedding and output projection weights are tied. The complete model has 5,273,088 parameters.

## 4. Standard causal attention

For each token, attention forms:

- Query (Q): what this position is looking for.
- Key (K): what each position offers for matching.
- Value (V): the information passed forward.

The operation is:

`softmax(QKᵀ / sqrt(d_k)) V`

The causal mask sets future-position scores to negative infinity before softmax. Therefore, token position t can only use positions 0 through t. This is essential: without it, training could see future tokens that are unavailable during generation.

The padding mask separately prevents padded key positions from contributing. Padding labels use -100, which PyTorch cross-entropy ignores.

**Complexity:** the score matrix is T by T, giving quadratic sequence-length cost for the attention scores.

## 5. Linear causal attention

The second model uses:

`phi(x) = ELU(x) + 1`

This positive feature map lets us rewrite the attention computation using cumulative key-value sums instead of explicitly creating the full T by T score matrix. For each time step, the model accumulates information from the current and previous positions only, preserving causality.

The intended theoretical benefit is linear scaling in sequence length when head dimension is fixed. However, our implementation materializes cumulative key-value intermediates of size proportional to T times head-dimension squared. This can be expensive in memory and computation at the context lengths tested.

**Important viva point:** Do not say linear attention was faster in our experiment. It was not. The project tested whether the theoretical alternative translated into a practical benefit for this model and hardware.

## 6. How were the experiments controlled?

Both configs use the same:

- 8,000-token vocabulary
- 4 layers, 4 heads, model dimension 256
- Context length 256
- Batch size 32
- AdamW, learning rate 0.0003, weight decay 0.1
- 200 warm-up steps, then constant learning rate
- 10,000 training steps
- Random seed 42
- 50,000-story TinyStories sample
- Tesla T4 GPU

The intended experimental difference is the attention type: standard or linear. Both use the shared training engine and same data/tokenizer interface.

## 7. Explain the final results

| Metric | Standard | Linear |
|---|---:|---:|
| Test perplexity | **8.51** | 8.78 |
| Training time | **1,419 s** | 3,319 s |
| Training throughput at batch 32, context 256 | 79,198 tokens/s | 29,471 tokens/s |
| Peak training memory at batch 32, context 256 | 1,913 MB | 4,513 MB |

The standard model achieved slightly better test perplexity and trained about 2.34 times faster in this run. Linear attention's latency disadvantage became smaller as context increased: at context 2048, its forward latency was 175.7 ms compared with 162.5 ms for standard attention.

The conclusion is not that linear attention is universally worse. It is that this implementation did not provide a practical speed or memory advantage for the tested model, head dimension, sequence lengths, and Tesla T4.

## 8. Likely panel questions

**Q1. Why is the model decoder-only?**  
Because the task is autoregressive next-token prediction. A decoder-only model uses causal self-attention so each position sees only the prefix.

**Q2. Why do we need a causal mask?**  
To prevent information leakage from future tokens during training. Generation only has access to previous tokens.

**Q3. Why divide QKᵀ by sqrt(d_k)?**  
As key dimension grows, dot products can become large and push softmax into saturated regions. Scaling helps keep the values in a more stable range.

**Q4. What is the difference between causal and padding masks?**  
The causal mask blocks future positions. The padding mask blocks padded key positions that are not real tokens.

**Q5. Why use ELU(x)+1?**  
It creates positive feature values, which supports the associative linear-attention reformulation and a positive normalization denominator.

**Q6. Is linear attention always faster?**  
No. Its asymptotic sequence scaling is attractive, but constants, head dimension, intermediate tensor sizes, hardware kernels, and sequence length determine practical performance.

**Q7. Why did linear attention use more memory here?**  
The implementation builds cumulative key-value tensors with a head-dimension-squared component at every sequence position. At our tested contexts, this intermediate can be larger than the standard attention score matrix.

**Q8. How did you prevent overfitting to the test set?**  
Training and model selection use training and validation data. The held-out test split is evaluated after the final checkpoint is selected. The evaluation protocol and fixed prompts are documented by the team.

**Q9. What is perplexity?**  
It is the exponential of mean token negative log-likelihood. Lower is better when comparing models on the same tokenizer and same evaluation text. A uniform predictor over 8,000 tokens has perplexity 8,000.

**Q10. Did the model learn general world knowledge?**  
No. It is a small model trained from scratch on TinyStories, a children's story dataset. Its low test perplexity on stories does not imply reliable factual knowledge or strong instruction following.

**Q11. What did SFT change?**  
The standard model's Dolly test response NLL decreased from 6.379 before SFT to 3.928 after SFT. However, automatic generation analysis still found repetition and factual errors. Lower response loss alone does not prove high-quality answers.

**Q12. What would you improve next?**  
Profile and optimize the linear-attention implementation, avoid materializing the full sequence of key-value states where possible, test longer contexts and larger head dimensions, and run repeated seeds to estimate variability.

## 9. Files to open during the viva

- `src/model.py`
- `src/attention.py`
- `src/efficient_model.py`
- `src/efficient_attention.py`
- `src/training.py`
- `configs/baseline.yaml`
- `configs/efficient_attention.yaml`
- `results/metrics/baseline_standard_train_log.csv`
- `results/metrics/efficient_linear_train_log.csv`
- `report/results.md`

## 10. Honest limitations to state

- The model is small and trained on a narrow story corpus.
- Linear attention was not faster or more memory-efficient in the measured T4 settings.
- Fine-tuning made outputs more answer-shaped but did not make the model factually reliable.
- The rubric was scored blind by an automated LLM judge, not a human: 1.25/8 pre-trained vs 2.60/8 after SFT. Do not describe it as a human evaluation.
