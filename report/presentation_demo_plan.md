# Presentation and demo plan (Member 3 coordinates; all three present)

Target: ~12 slides, ~12 minutes, then a ~5-minute live demo and the viva. Every number on a slide is taken
from `results/metrics/*` or `experiments/experiment_log.csv`; the source file is written in small print on
the slide.

## Slide outline

| # | Slide | Presenter | Content / source |
|---|---|---|---|
| 1 | Title | Samvith | project title, team, roles |
| 2 | Goal and research questions | Samvith | E1–E3 from `report/evaluation_protocol.md` §1 |
| 3 | Pipeline overview | Samvith | data → tokenizer → pre-training → SFT → evaluation (one diagram) |
| 4 | Data and tokenizer | Haneesh | TinyStories 9,000/500/500; Dolly 13,500/750/750; BPE 8,000; special tokens; `*_stats.json` |
| 5 | SFT formatting and loss masking | Haneesh | template, response-only labels, one masked example |
| 6 | Model architecture | Akash | decoder-only block diagram; 4 layers × 4 heads × 256; tied embeddings; 5,273,088 parameters |
| 7 | Standard vs linear attention | Akash | softmax(QKᵀ)V vs φ(Q)(φ(K)ᵀV); O(T²) vs O(T·d²); causal cumulative sums |
| 8 | Pre-training results (E1) | Akash | `results/figures/pretrain_loss.png`; val/test perplexity vs uniform 8,000 |
| 9 | Instruction tuning (E2) | Haneesh | response NLL before/after; rubric table; one before/after example |
| 10 | Efficiency comparison (E3) | Samvith | `bench_forward_latency.png`, `bench_forward_memory.png`, throughput table; perplexity gap |
| 11 | Failure analysis (E4) | Samvith | tag counts + two real examples with causes |
| 12 | Limitations and lessons | Samvith | sample sizes, one seed, GPU noise, data issues found (Dolly duplicates, truncation) |

## Live demo (≈ 5 min, run from Colab with checkpoints on Drive)

1. Load the pre-trained checkpoint and generate a story from "Once upon a time" (greedy):
   `python src/evaluate.py generate --ckpt <pre.pt> --label demo_pre --prompts demo/story_prompts.jsonl --raw`
2. Same instruction to the pre-trained and the fine-tuned model:
   `python src/evaluate.py generate --ckpt <pre.pt> --label demo_pre_inst --prompts demo/inst_prompts.jsonl`
   `python src/evaluate.py generate --ckpt <sft.pt> --label demo_sft_inst --prompts demo/inst_prompts.jsonl`
3. Show the benchmark plot and run one short benchmark (`--steps 5`) live to show how the numbers are measured.

**Backup:** if Colab or Drive is unavailable, show the saved `*_generations.jsonl` and figures from the
repository (generated earlier with the same commands); say that it is a recording.

## Rehearsal checklist

- [ ] Full run-through with timing, at least one day before
- [ ] Every member can explain every slide (plan rule 7), not only their own
- [ ] Each member answers 5 questions from the others' viva files
- [ ] Demo commands tested on a fresh Colab runtime
- [ ] Numbers on slides re-checked against `results/metrics/`
