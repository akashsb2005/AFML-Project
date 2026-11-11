# Member 1 — Model Architecture and Pretraining

**Owner:** Akash Shivanand Bagoji  
**Scope:** decoder-only Transformer, standard causal attention, causal linear attention, shared pretraining engine, and controlled pretraining comparison.

## 1. Delivered implementation

| File | Responsibility |
|---|---|
| `src/model.py` | Standard decoder-only Transformer, model factory, next-token loss |
| `src/attention.py` | Multi-head scaled dot-product causal self-attention |
| `src/efficient_model.py` | Same Transformer structure with linear attention |
| `src/efficient_attention.py` | Causal linear attention using the positive feature map |
| `src/pretrain.py` | Standard-attention training entry point |
| `src/pretrain_efficient.py` | Linear-attention training entry point |
| `src/training.py` | Shared config-driven training, validation, checkpointing and logging |
| `configs/baseline.yaml` | Standard-attention experiment configuration |
| `configs/efficient_attention.yaml` | Matched linear-attention experiment configuration |

Both models use token embeddings, learned positional embeddings, four pre-norm decoder blocks, residual connections, a 4× GELU feed-forward network, final layer normalization, and tied token-embedding/output weights. Each model has 5,273,088 parameters.

## 2. Standard causal self-attention

For each head, the standard implementation computes:

`Attention(Q,K,V) = softmax((QK^T)/sqrt(d_k) + causal_mask + padding_mask)V`

The causal mask prevents a position from attending to future tokens. The padding mask prevents padded key positions from contributing to the result. The model accepts `attention_mask`, ignores target label `-100` in cross-entropy, returns logits directly for inference, and returns `(logits, loss)` when targets are supplied.

The attention score matrix has sequence dimensions `T × T`, so its attention computation scales quadratically with sequence length: approximately O(T²d) time and O(T²) score storage per head.

## 3. Causal linear attention

The efficient variant replaces softmax attention with the positive feature map:

`phi(x) = ELU(x) + 1`

For causal attention, it accumulates key-value outer products and key sums over preceding positions, then computes a normalized query-weighted result. This avoids explicitly constructing a `T × T` attention score matrix. The implementation uses cumulative sums over sequence positions; its intermediate key-value tensor has shape `[batch, heads, sequence, head_dim, head_dim]`.

Theoretical sequence scaling is linear in `T` for fixed head dimension, with a larger dependence on head dimension. This implementation's materialized cumulative-sum intermediates can be memory-intensive at short and medium contexts. Therefore, an efficiency advantage is a measured question, not an assumption.

## 4. Matched pretraining protocol

Both YAML configurations use the same settings except for the attention type:

| Setting | Value |
|---|---:|
| Vocabulary | 8,000 |
| Layers / heads | 4 / 4 |
| Model dimension | 256 |
| Context length | 256 |
| Batch size | 32 |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 0.1 |
| Warm-up | 200 steps |
| Training steps | 10,000 |
| Random seed | 42 |
| Dataset | 50,000-story TinyStories sample |
| Hardware | Tesla T4 |

Both entry points use the shared `src/training.py` engine. It reads YAML configuration, applies linear learning-rate warm-up followed by a constant rate, clips gradients, evaluates at configured intervals, counts non-padding target tokens, writes per-step CSV logs, and saves last and best checkpoints. Checkpoints include model and optimizer state, configuration metadata, and random-number-generator state.

## 5. Final measured pretraining results

The final run artifacts are committed under `results/metrics/` and the experiment log.

| Metric | Standard attention | Linear attention |
|---|---:|---:|
| Completed steps | 10,000 | 10,000 |
| Training tokens | 56,592,064 | 56,592,064 |
| Final training loss | 2.099968 | 2.121845 |
| Final periodic validation loss | 2.139310 | 2.170815 |
| Full validation perplexity | 8.60 | 8.89 |
| Held-out test perplexity | **8.5118** | 8.7788 |
| Training time | **1,419.22 s** | 3,319.34 s |

The standard model achieved lower test perplexity and completed pretraining in less time on the measured Tesla T4 run. The linear model took approximately 2.34× as long. These results do not support a claim that this linear-attention implementation is faster on the tested hardware and context length.

## 6. Efficiency measurements and interpretation

The benchmark artifacts are in `results/metrics/gpu_t4_*` and `results/figures/bench_t4_*`.

At batch 32 and context 256, measured training throughput was 79,198 tokens/s for standard attention and 29,471 tokens/s for linear attention. Peak memory was 1,913 MB and 4,513 MB respectively. In the forward latency sweep, the linear/standard latency ratio decreased from 2.52× at context 128 to 1.08× at context 2048.

This is consistent with the implementation's head-dimension-dependent intermediate tensors: asymptotically linear sequence scaling does not guarantee a practical win at the tested lengths. The report should present this as a limitation and empirical finding, not hide it.

## 7. Validation and reproducibility

- The all-members integration branch has a successful GitHub Actions run for compilation and the repository test suite (run linked from the repository's Actions page).
- The integration PR description reports 67 local tests passing and a successful end-to-end smoke run, including its resume path.
- The final pretraining CSVs each contain 10,000 optimizer-step rows and end at step 10,000.
- Test perplexity JSON files record 527,527 scored tokens and 2,501 test stories for both attention variants.
- Checkpoints are stored outside Git; the committed metrics record the corresponding Drive checkpoint locations.

To reproduce the final pipeline, use `notebooks/final_runs.ipynb` with a Tesla T4 runtime and follow the documented Google Drive output setup. The resumable driver is `scripts/run_final.sh`.

## 8. Remaining submission checks

1. Review and merge the all-members integration PR only after the team approves the combined changes.
2. Keep the final report's claims aligned with the committed metric files.
3. Complete the blind rubric scoring: `results/metrics/rubric_sheet.csv` currently has blank score columns.
4. Ensure the final report no longer says "Draft" once the rubric and final team review are complete.

**Status:** Member 1's model, training implementation, and final pretraining experiments are present in the all-members integration branch. The remaining work is team-level review/submission, not the earlier 1,000-step/20-step prototype stage.
