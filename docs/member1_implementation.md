\# Member 1 — Model Architecture and Pretraining



\## 1. Objective



Member 1 is responsible for implementing the decoder-only Transformer architecture, standard causal self-attention, an efficient causal linear attention variant, and the corresponding pretraining scripts.



The objective is to establish two model implementations that can be evaluated under comparable experimental settings.



\## 2. Standard Transformer Architecture



The standard model uses a decoder-only Transformer architecture for autoregressive language modeling.



The main components are:



\- Token embeddings to represent input token IDs.

\- Positional information to represent token order.

\- Multi-head causal self-attention.

\- Feed-forward network layers.

\- Residual connections and normalization.

\- A language-model output layer that produces vocabulary logits.



The model predicts the next token from the preceding tokens.



\### Causal Attention



Causal attention prevents a token from attending to future tokens. This preserves the autoregressive property required for next-token prediction.



The standard scaled dot-product attention operation is:



Attention(Q, K, V) = softmax(QKᵀ / √dₖ)V



where Q, K, and V represent queries, keys, and values, and dₖ is the key dimension.



The implementation is provided in `src/attention.py`, with the standard model in `src/model.py`.



\## 3. Efficient Causal Linear Attention



The efficient variant uses causal linear attention as an alternative to standard softmax attention.



The implementation is provided in:



\- `src/efficient\_attention.py`

\- `src/efficient\_model.py`



The objective is to investigate whether the alternative attention mechanism reduces computational or memory requirements while maintaining useful language-model performance.



An actual performance improvement must be established experimentally; it should not be assumed from the implementation alone.



\## 4. Pretraining Implementation



The two training entry points are:



\- `src/pretrain.py` — standard-attention baseline.

\- `src/pretrain\_efficient.py` — efficient-attention variant.



The repository contains separate configuration files:



\- `configs/baseline.yaml`

\- `configs/efficient\_attention.yaml`



The shared configurations specify a vocabulary size of 8,000, four Transformer layers, four attention heads, a model dimension of 256, a context length of 256, a batch size of 32, and a maximum training budget of 10,000 steps.



The configurations use the same seed and training hyperparameters, with the attention type being the intended experimental difference.



The training data and tokenizer paths are specified in the configuration files. Their availability and compatibility must be verified before running the complete experiments.



\## 5. Dataset Handling



The dataset implementation is located in `src/dataset.py`.



The training pipeline must use tokenized data compatible with the configured vocabulary and model input format. The training and validation datasets, tokenizer, and preprocessing assumptions should be verified before conducting the final comparison.



\## 6. Testing and Validation



The model shape tests are located in `tests/test\_model\_shapes.py`.



Local validation completed:



\- Python syntax compilation completed without reported errors.

\- Standard model shape test passed.

\- Efficient model shape test passed.

\- Full repository test suite: 28 passed, 0 failed.



These tests establish that the tested code paths work under the local test environment. They do not establish that the full training runs have completed or that one attention mechanism is faster.



\## 7. Training Progress



The following progress was previously verified:



\- Standard baseline checkpoint: step 1,000.

\- Efficient model smoke-test checkpoint: step 20.

\- Efficient checkpoint reload and forward pass completed successfully, producing logits of shape `\[2, 128, 8000]`.



The full 10,000-step controlled comparison has not yet been established as complete.



\## 8. Remaining Work



The following tasks remain before the experimental study can be considered complete:



1\. Verify the dataset, tokenizer, and configuration paths.

2\. Validate both training scripts using short training runs.

3\. Continue or resume baseline training from a valid checkpoint where supported.

4\. Train the efficient variant under matched settings.

5\. Record training and validation losses, completed steps, runtime, and memory usage where available.

6\. Compare both variants under equivalent experimental conditions.

7\. Save the results and document limitations.



\## 9. Reproducibility



Experiments should record the configuration, random seed, dataset and tokenizer versions, checkpoint path, software environment, hardware, completed training steps, and evaluation results.



Only measured results should be used to claim improvements in speed, memory consumption, or model quality.



\## 10. Summary



Member 1 has implemented the standard and efficient model code, corresponding pretraining scripts, dataset handling, and model shape tests. Local tests pass. The remaining work is to validate the complete training pipeline and establish the controlled experimental comparison.



