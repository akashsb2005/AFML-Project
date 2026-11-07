"""Append the final runs to experiments/experiment_log.csv, reading every value from the saved results."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

M = Path("results/metrics")


def log(**fields):
    args = [sys.executable, "src/log_experiment.py", "--member", "Team", "--seed", "42"]
    for key, value in fields.items():
        if value not in (None, ""):
            args += ["--" + key.replace("_", "-"), str(value)]
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL)


def last_row(name):
    with open(M / f"{name}_train_log.csv") as fh:
        rows = list(csv.DictReader(fh))
    train = [r["train_loss"] for r in rows if r["train_loss"]]
    val = [r["val_loss"] for r in rows if r["val_loss"]]
    return {"steps": rows[-1]["step"], "train_tokens": rows[-1]["tokens_seen"], "wall_time_s": rows[-1]["elapsed_s"],
            "train_loss": train[-1] if train else "", "val_loss": min(map(float, val)) if val else ""}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--tiny-stories", required=True)
    a = ap.parse_args()
    data = f"TinyStories up to {a.tiny_stories} stories; Dolly 15K deduplicated"
    common = {"hardware": a.hardware, "dataset": data}
    for run, attention, config, label in (("baseline_standard", "standard", "configs/baseline.yaml", "pre_std"),
                                          ("efficient_linear", "linear", "configs/efficient_attention.yaml", "pre_lin")):
        val = json.loads((M / f"{label}_val_perplexity.json").read_text())
        test = json.loads((M / f"{label}_test_perplexity.json").read_text())
        log(run_id=run, stage="pretrain", attention=attention, config=config, batch_size=32, context_length=256,
            lr="3e-4", val_ppl=val["perplexity"], test_ppl=test["perplexity"],
            checkpoint=f"checkpoints/{run}/best_checkpoint.pt", results_file=str(M / f"{label}_test_perplexity.json"),
            notes="final run", **last_row(run), **common)
    for run, attention, label in (("sft", "standard", "sft"), ("sft_linear", "linear", "sft_lin")):
        nll = json.loads((M / f"{label}_sft_loss.json").read_text())
        log(run_id=run, stage="sft", attention=attention, config="src/sft.py defaults", batch_size=4, context_length=256,
            lr="3e-4", response_nll=nll["mean_response_nll"], checkpoint=f"checkpoints/{run}/best.pt",
            results_file=str(M / f"{label}_sft_loss.json"), notes="final run; response NLL on Dolly test",
            **last_row(run), **common)
    nll = json.loads((M / "pre_sft_loss.json").read_text())
    log(run_id="pre_std_on_dolly", stage="eval", attention="standard", response_nll=nll["mean_response_nll"],
        results_file=str(M / "pre_sft_loss.json"), notes="pre-trained model before SFT; response NLL on Dolly test", **common)


if __name__ == "__main__":
    main()
