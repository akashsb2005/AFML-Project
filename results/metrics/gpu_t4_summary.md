Source: `gpu_t4_benchmarks_raw.json` — Colab, Tesla T4 15360 MiB, driver 580.82.07, torch 2.11.0+cu130 (CUDA 13.0); repo main 299afbc + member3-integration; src/benchmark.py; random weights; 2026-10-27

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
