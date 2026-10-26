"""Append one row to experiments/experiment_log.csv (plan §5 rule 7: record configuration and seed).

python src/log_experiment.py --run-id pre_std_v1 --member "Akash" --stage pretrain --attention standard \
    --config configs/baseline.yaml --seed 42 --val-loss 1.83 --checkpoint checkpoints/pre_std_v1.pt \
    --notes "first full run"

Only measured values go in here; leave a field empty rather than estimating it.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import subprocess
import sys
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "experiments" / "experiment_log.csv"
COLUMNS = ["run_id", "date", "member", "stage", "attention", "config", "seed", "git_commit", "hardware",
           "dataset", "train_tokens", "steps", "batch_size", "context_length", "lr", "train_loss", "val_loss",
           "val_ppl", "test_ppl", "response_nll", "tokens_per_s", "peak_mem_mb", "wall_time_s", "checkpoint",
           "results_file", "notes"]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=LOG.parents[1], check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def append(row: dict, log: Path = LOG) -> dict:
    unknown = set(row) - set(COLUMNS)
    if unknown:
        raise ValueError(f"unknown columns: {sorted(unknown)}")
    rows = []
    if log.exists():
        with open(log, newline="") as fh:
            rows = list(csv.DictReader(fh))
    if any(r["run_id"] == row["run_id"] for r in rows):
        raise ValueError(f"run_id {row['run_id']!r} already logged — use a new id")
    full = {c: "" for c in COLUMNS}
    full.update({"date": dt.date.today().isoformat(), "git_commit": git_commit()})
    full.update({k: v for k, v in row.items() if v is not None})
    write_header = not log.exists() or log.stat().st_size == 0
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow(full)
    return full


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for c in COLUMNS:
        if c not in ("date", "git_commit"):
            ap.add_argument("--" + c.replace("_", "-"), dest=c, required=c in ("run_id", "member", "stage"))
    args = vars(ap.parse_args(argv))
    print(append(args))


if __name__ == "__main__":
    sys.exit(main())
