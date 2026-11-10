#!/usr/bin/env bash
# Final runs for the report: pretrain both variants, fine-tune, evaluate. Safe to re-run after a Colab disconnect:
# finished stages are skipped, and an interrupted pretraining run resumes from its last checkpoint.
#
#   OUT=/content/drive/MyDrive/afml-slm/final bash scripts/run_final.sh
#
# Test split numbers are computed here only, once per final checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:?set OUT to a folder that survives the runtime, e.g. on Drive}
TINY_STORIES=${TINY_STORIES:-50000}
PRETRAIN_STEPS=${PRETRAIN_STEPS:-10000}
SFT_EPOCHS=${SFT_EPOCHS:-4}
SKIP_PREPROCESS=${SKIP_PREPROCESS:-0}
EVAL_NEW_TOKENS=${EVAL_NEW_TOKENS:-128}
M=results/metrics
CK=$OUT/checkpoints
STAMPS=$OUT/stamps
mkdir -p "$CK" "$STAMPS" "$OUT/metrics" "$OUT/figures" "$M" results/figures

sync_results() { cp -R "$M"/. "$OUT/metrics/"; cp -R results/figures/. "$OUT/figures/"; cp experiments/experiment_log.csv "$OUT/"; }
stage() {  # stage NAME COMMAND...: run once, recorded in $STAMPS
    local name=$1; shift
    if [ -e "$STAMPS/$name" ]; then echo "== skip $name (already done)"; return; fi
    echo "== $name"; "$@"; sync_results; touch "$STAMPS/$name"
}
restore_results() { [ -d "$OUT/metrics" ] && cp -R "$OUT/metrics/." "$M/"; return 0; }

data() {
    if [ "$SKIP_PREPROCESS" != 1 ]; then python src/preprocess.py --stage all --max_tiny_stories "$TINY_STORIES"; fi
    python src/tokenizer.py
}
pretrain() {  # pretrain SCRIPT RUN_NAME
    local resume=(); [ -e "$CK/$2/last_checkpoint.pt" ] && resume=(--resume)
    python "src/$1.py" --max_steps "$PRETRAIN_STEPS" --checkpoint_dir "$CK/$2" --log_path "$M/$2_train_log.csv" "${resume[@]+"${resume[@]}"}"
}
sft() {  # sft PRETRAIN_RUN OUTPUT_NAME
    python src/sft.py --pretrained-checkpoint "$CK/$1/best_checkpoint.pt" --epochs "$SFT_EPOCHS" \
        --output-dir "$CK/$2" --log-path "$M/$2_train_log.csv" --log-every 500
}
prompts() {
    python src/select_eval_prompts.py --test data/processed/dolly_test.jsonl \
        --exclude data/processed/dolly_train.jsonl data/processed/dolly_validation.jsonl \
        --tokenizer artifacts/tokenizer/tokenizer.json --context-length 256 --n 60 --strict \
        --stories data/processed/tiny_stories_validation.jsonl
}
stories() {  # stories LABEL CHECKPOINT
    python src/evaluate.py perplexity --ckpt "$2" --label "$1_val" --stories data/processed/tiny_stories_validation.jsonl
    python src/evaluate.py perplexity --ckpt "$2" --label "$1_test" --stories data/processed/tiny_stories_test.jsonl
    python src/evaluate.py generate --ckpt "$2" --label "$1_story" --prompts "$M/story_prompts.jsonl" --raw --max-new-tokens 64
}
instructions() {  # instructions LABEL CHECKPOINT
    python src/evaluate.py sft-loss --ckpt "$2" --label "$1" --data data/processed/dolly_test.jsonl \
        --exclude data/processed/dolly_train.jsonl data/processed/dolly_validation.jsonl
    python src/evaluate.py generate --ckpt "$2" --label "$1" --prompts "$M/eval_prompts.jsonl" --max-new-tokens "$EVAL_NEW_TOKENS"
    python src/evaluate.py generate --ckpt "$2" --label "$1_sampled" --prompts "$M/eval_prompts.jsonl" \
        --max-new-tokens "$EVAL_NEW_TOKENS" --temperature 0.8 --top-k 50
}
figures() {
    python src/make_figures.py loss --logs "$M/baseline_standard_train_log.csv" "$M/efficient_linear_train_log.csv" \
        --labels "Standard attention" "Linear attention" --out results/figures/pretrain_loss.png
    python src/make_figures.py loss --logs "$M/baseline_standard_train_log.csv" "$M/efficient_linear_train_log.csv" \
        --labels "Standard attention" "Linear attention" --val-only --title "Pre-training validation loss" \
        --out results/figures/pretrain_val_loss.png
    python src/make_figures.py loss --logs "$M/sft_train_log.csv" "$M/sft_linear_train_log.csv" \
        --labels "Standard attention" "Linear attention" \
        --title "Instruction fine-tuning loss (response tokens only)" --out results/figures/sft_loss.png
}

restore_results
python -m pytest -q tests
stage data data
stage pretrain_standard pretrain pretrain baseline_standard
stage pretrain_linear pretrain pretrain_efficient efficient_linear
stage sft sft baseline_standard sft
stage sft_linear sft efficient_linear sft_linear
stage prompts prompts
stage eval_stories_standard stories pre_std "$CK/baseline_standard/best_checkpoint.pt"
stage eval_stories_linear stories pre_lin "$CK/efficient_linear/best_checkpoint.pt"
stage eval_pre instructions pre "$CK/baseline_standard/best_checkpoint.pt"
stage eval_sft instructions sft "$CK/sft/best.pt"
stage eval_sft_linear instructions sft_lin "$CK/sft_linear/best.pt"
stage rubric python src/evaluate.py rubric-sheet --outputs "pre=$M/pre_generations.jsonl" "sft=$M/sft_generations.jsonl"
stage figures figures
stage log python scripts/log_final.py --hardware "${HARDWARE:-Tesla T4 (Colab)}" --tiny-stories "$TINY_STORIES"
sync_results
echo "ALL DONE. Results: $OUT/metrics, figures: $OUT/figures, checkpoints: $CK"
