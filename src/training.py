"""Shared, configuration-driven pretraining loop for both attention variants.

The baseline and efficient runs use the same data pipeline, optimizer, schedule,
evaluation cadence, logging schema and checkpoint format. Only the attention
implementation differs.
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

try:
    from .dataset import TinyStoriesDataset
    from .model import build_model
except ImportError:  # Support direct script execution with src/ on sys.path.
    from dataset import TinyStoriesDataset
    from model import build_model

SRC = Path(__file__).resolve().parent
PROJECT = SRC.parent
LOG_FIELDS = ["step", "tokens_seen", "train_loss", "val_loss", "lr", "elapsed_s"]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    required = ("model", "train", "data")
    missing = [key for key in required if not isinstance(config.get(key), dict)]
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")
    return config


def validate_config(config: dict) -> None:
    model, train = config["model"], config["train"]
    required_model = ("vocab_size", "n_layer", "n_head", "d_model",
                      "context_length", "dropout", "attention")
    required_train = ("batch_size", "lr", "weight_decay", "warmup_steps",
                      "max_steps", "eval_interval", "eval_batches")
    for key in required_model:
        if key not in model:
            raise ValueError(f"Missing model.{key} in configuration")
    for key in required_train:
        if key not in train:
            raise ValueError(f"Missing train.{key} in configuration")
    for key in ("train_jsonl", "validation_jsonl", "tokenizer"):
        if key not in config["data"]:
            raise ValueError(f"Missing data.{key} in configuration")
    positive = ("batch_size", "max_steps", "eval_interval", "eval_batches")
    if any(int(train[k]) < 1 for k in positive):
        raise ValueError("batch_size, max_steps, eval_interval and eval_batches must be positive")
    if int(train["warmup_steps"]) < 0:
        raise ValueError("warmup_steps cannot be negative")
    if float(train["lr"]) <= 0:
        raise ValueError("train.lr must be positive")
    if float(train["weight_decay"]) < 0:
        raise ValueError("train.weight_decay cannot be negative")
    if not 0.0 <= float(model["dropout"]) < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")
    if int(model["context_length"]) < 2:
        raise ValueError("model.context_length must be at least 2")
    if int(model["d_model"]) % int(model["n_head"]):
        raise ValueError("model.d_model must be divisible by model.n_head")
    if model["attention"] not in ("standard", "linear"):
        raise ValueError("model.attention must be 'standard' or 'linear'")


def evaluate(model, loader, device, max_batches: int | None = None) -> tuple[float, int]:
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    try:
        with torch.no_grad():
            for index, batch in enumerate(loader):
                if max_batches is not None and index >= max_batches:
                    break
                ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                mask = batch["attention_mask"].to(device, non_blocking=True)
                logits = model(ids, attention_mask=mask)
                flat_labels = labels.reshape(-1)
                losses = F.cross_entropy(
                    logits.reshape(-1, model.vocab_size), flat_labels,
                    reduction="none", ignore_index=-100,
                )
                valid = flat_labels != -100
                total_loss += losses[valid].sum().item()
                total_tokens += int(valid.sum().item())
    finally:
        model.train(was_training)
    if total_tokens == 0:
        raise RuntimeError("Evaluation produced zero supervised tokens.")
    return total_loss / total_tokens, total_tokens


def _write_log_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=LOG_FIELDS).writeheader()
        return
    with path.open("r", newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle), [])
    if header != LOG_FIELDS:
        raise ValueError(
            f"Existing log {path} has columns {header}; expected {LOG_FIELDS}. "
            "Choose a new --log-path rather than mixing incompatible runs."
        )


def _save_checkpoint(path: Path, *, step: int, tokens_seen: int, model,
                     optimizer, config: dict, train_loss: float,
                     validation_loss: float, best_validation_loss: float | None,
                     elapsed_s: float, data_rng_state: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep model fields at the top level for src.model_api.load_checkpoint(),
    # while retaining nested sections for reproducibility and resume validation.
    checkpoint_config = dict(config["model"])
    checkpoint_config.update({
        "model": dict(config["model"]),
        "train": dict(config["train"]),
        "data": dict(config["data"]),
        "seed": config.get("seed", 42),
        "training": dict(config.get("training", {})),
    })
    state = {
        "step": step,
        "tokens_seen": tokens_seen,
        "model_state": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": checkpoint_config,
        "tokenizer_path": config["data"]["tokenizer"],
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "best_validation_loss": best_validation_loss,
        "elapsed_s": elapsed_s,
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "data_loader_rng_state": data_rng_state,
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _set_learning_rate(optimizer, base_lr: float, step: int, warmup_steps: int) -> float:
    # Linear warmup followed by a constant learning rate; identical for both runs.
    factor = min(1.0, (step + 1) / warmup_steps) if warmup_steps else 1.0
    lr = base_lr * factor
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr



def _canonical_model_config(config: dict) -> dict:
    """Normalize current and legacy checkpoint model keys for resume validation."""
    source = config.get("model", config)
    attention = source.get("attention", "standard")
    if attention == "causal_linear_elu_plus_one":
        attention = "linear"
    return {
        "vocab_size": int(source.get("vocab_size", 8000)),
        "n_layer": int(source.get("n_layer", source.get("num_layers", 4))),
        "n_head": int(source.get("n_head", source.get("num_heads", 4))),
        "d_model": int(source.get("d_model", source.get("embedding_dim", 256))),
        "context_length": int(source.get("context_length", source.get("max_seq_len", 256))),
        "dropout": float(source.get("dropout", 0.1)),
        "attention": attention,
    }


def run_training(attention: str, argv=None) -> None:
    default_config = PROJECT / "configs" / (
        "baseline.yaml" if attention == "standard" else "efficient_attention.yaml"
    )
    parser = argparse.ArgumentParser(description=f"Pretrain the {attention} attention model.")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--max_steps", "--max-steps", dest="max_steps", type=int)
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int)
    parser.add_argument("--block_size", "--block-size", dest="block_size", type=int)
    parser.add_argument("--learning_rate", "--learning-rate", dest="learning_rate", type=float)
    parser.add_argument("--weight_decay", "--weight-decay", dest="weight_decay", type=float)
    parser.add_argument("--warmup_steps", "--warmup-steps", dest="warmup_steps", type=int)
    parser.add_argument("--eval_interval", "--eval-interval", dest="eval_interval", type=int)
    parser.add_argument("--eval_batches", "--eval-batches", dest="eval_batches", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint_dir", "--checkpoint-dir", dest="checkpoint_dir", type=Path)
    parser.add_argument("--log_path", "--log-path", dest="log_path", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if config["model"].get("attention") != attention:
        parser.error(f"{args.config} must set model.attention: {attention}")
    overrides = {
        "max_steps": ("train", "max_steps", args.max_steps),
        "batch_size": ("train", "batch_size", args.batch_size),
        "learning_rate": ("train", "lr", args.learning_rate),
        "weight_decay": ("train", "weight_decay", args.weight_decay),
        "warmup_steps": ("train", "warmup_steps", args.warmup_steps),
        "eval_interval": ("train", "eval_interval", args.eval_interval),
        "eval_batches": ("train", "eval_batches", args.eval_batches),
    }
    for _, (section, key, value) in overrides.items():
        if value is not None:
            config[section][key] = value
    seed = int(config.get("seed", 42) if args.seed is None else args.seed)
    model_cfg, train_cfg = config["model"], config["train"]
    context_length = int(model_cfg["context_length"])
    block_size = context_length if args.block_size is None else args.block_size
    if block_size != context_length:
        parser.error("--block_size must equal model.context_length")
    validate_config(config)
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max_steps must be positive")

    run_name = str(config.get("run_name", attention))
    checkpoint_dir = args.checkpoint_dir or (PROJECT / "checkpoints" / run_name)
    log_path = args.log_path or (PROJECT / "results" / "metrics" / f"{run_name}_train_log.csv")
    checkpoint_path = checkpoint_dir / "last_checkpoint.pt"
    best_path = checkpoint_dir / "best_checkpoint.pt"

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Fail early with actionable paths instead of an opaque dataset/tokenizer error.
    for key in ("train_jsonl", "validation_jsonl", "tokenizer"):
        artifact = Path(config["data"][key])
        if not artifact.is_absolute():
            artifact = PROJECT / artifact
        if not artifact.exists():
            raise FileNotFoundError(
                f"Required {key} artifact is missing: {artifact}. "
                "Coordinate with Member 2 before starting pretraining."
            )

    train_dataset = TinyStoriesDataset(split="train", block_size=block_size)
    val_dataset = TinyStoriesDataset(split="validation", block_size=block_size)
    batch_size = int(train_cfg["batch_size"])
    if len(train_dataset) < batch_size:
        raise RuntimeError(
            f"Training dataset has {len(train_dataset)} blocks, fewer than batch_size={batch_size}"
        )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=0, pin_memory=device.type == "cuda", generator=generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model_config = dict(model_cfg)
    model = build_model(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    _write_log_header(log_path)

    start_step, tokens_seen, best_val, elapsed_before = 0, 0, None, 0.0
    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Cannot resume; checkpoint missing: {checkpoint_path}")
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        saved_config = saved.get("config", {})
        if _canonical_model_config(saved_config) != _canonical_model_config(config):
            raise ValueError("Checkpoint model configuration differs from requested configuration.")
        model_state = saved.get("model_state", saved.get("model_state_dict"))
        if model_state is None:
            raise KeyError("Checkpoint is missing model_state/model_state_dict.")
        model.load_state_dict(model_state)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        start_step = int(saved["step"])
        tokens_seen = int(saved.get("tokens_seen", 0))
        best_val = saved.get("best_validation_loss")
        elapsed_before = float(saved.get("elapsed_s", 0.0))
        if start_step >= int(train_cfg["max_steps"]):
            raise ValueError("--max_steps must be greater than the checkpoint step")
        if "python_rng_state" in saved:
            random.setstate(saved["python_rng_state"])
        if "torch_rng_state" in saved:
            torch.set_rng_state(saved["torch_rng_state"].cpu())
        if device.type == "cuda" and "cuda_rng_state_all" in saved:
            torch.cuda.set_rng_state_all(saved["cuda_rng_state_all"])
        if "data_loader_rng_state" in saved:
            generator.set_state(saved["data_loader_rng_state"].cpu())
        print(f"Resumed checkpoint at step {start_step}; tokens_seen={tokens_seen}")

    config["seed"] = seed
    config["training"] = {
        "batch_size": batch_size, "block_size": block_size,
        "max_steps": int(train_cfg["max_steps"]),
        "learning_rate": float(train_cfg["lr"]),
        "weight_decay": float(train_cfg["weight_decay"]),
        "warmup_steps": int(train_cfg["warmup_steps"]),
        "schedule": "linear_warmup_then_constant",
    }
    print(f"Run: {run_name} | attention: {attention}")
    print(f"Train blocks: {len(train_dataset)} | validation blocks: {len(val_dataset)}")
    print(f"Target steps: {train_cfg['max_steps']} | batch size: {batch_size}")
    print(f"Checkpoint: {checkpoint_path} | log: {log_path}")

    model.train()
    step = start_step
    window_loss_sum, window_tokens = 0.0, 0
    started = time.time()
    iterator = iter(train_loader)
    while step < int(train_cfg["max_steps"]):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        valid_tokens = int((labels != -100).sum().item())
        if valid_tokens == 0:
            raise RuntimeError("Training batch contains no supervised tokens.")

        lr = _set_learning_rate(
            optimizer, float(train_cfg["lr"]), step, int(train_cfg["warmup_steps"])
        )
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(ids, targets=labels, attention_mask=mask)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"Invalid loss at step {step + 1}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Invalid gradients at step {step + 1}")
        optimizer.step()

        step += 1
        tokens_seen += valid_tokens
        window_loss_sum += float(loss.item()) * valid_tokens
        window_tokens += valid_tokens
        should_eval = step % int(train_cfg["eval_interval"]) == 0 or step == int(train_cfg["max_steps"])
        val_loss = ""
        elapsed_now = elapsed_before + (time.time() - started)
        if should_eval:
            train_loss = window_loss_sum / window_tokens
            val_loss, val_tokens = evaluate(
                model, val_loader, device, int(train_cfg["eval_batches"])
            )
            elapsed = elapsed_before + (time.time() - started)
            elapsed_now = elapsed
            if best_val is None or val_loss < best_val:
                best_val = val_loss
                _save_checkpoint(
                    best_path, step=step, tokens_seen=tokens_seen, model=model,
                    optimizer=optimizer, config=config, train_loss=train_loss,
                    validation_loss=val_loss, best_validation_loss=best_val,
                    elapsed_s=elapsed, data_rng_state=generator.get_state(),
                )
            _save_checkpoint(
                checkpoint_path, step=step, tokens_seen=tokens_seen, model=model,
                optimizer=optimizer, config=config, train_loss=train_loss,
                validation_loss=val_loss, best_validation_loss=best_val,
                elapsed_s=elapsed, data_rng_state=generator.get_state(),
            )
            print(
                f"Step {step}/{train_cfg['max_steps']} | train loss {train_loss:.4f} | "
                f"val loss {val_loss:.4f} ({val_tokens} tokens) | lr {lr:.3g} | "
                f"tokens {tokens_seen} | elapsed {elapsed:.1f}s"
            )
            window_loss_sum, window_tokens = 0.0, 0

        # One row per optimizer step; val_loss is populated only on evaluation steps.
        with log_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=LOG_FIELDS).writerow({
                "step": step, "tokens_seen": tokens_seen,
                "train_loss": f"{float(loss.item()):.6f}",
                "val_loss": "" if val_loss == "" else f"{float(val_loss):.6f}",
                "lr": f"{lr:.8g}", "elapsed_s": f"{elapsed_now:.2f}",
            })

    print(f"Training finished: {run_name} at step {step}")
    print(f"Last checkpoint: {checkpoint_path}")
    print(f"Best checkpoint: {best_path}")
    print(f"Training log: {log_path}")
