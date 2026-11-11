"""Response-only supervised fine-tuning using the shared AFML model interface."""

import argparse
import csv
import importlib
import inspect
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import DollySFTDataset

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "checkpoints" / "sft"
LOG_FIELDS = ["step", "tokens_seen", "train_loss", "val_loss", "lr", "elapsed_s"]


@dataclass
class SFTConfig:
    max_length: int = 256
    batch_size: int = 4
    epochs: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    num_workers: int = 0
    seed: int = 42
    log_every: int = 10
    eval_interval: int = 500
    max_steps: int = 0


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_dataloaders(config):
    train = DollySFTDataset("train", max_length=config.max_length)
    validation = DollySFTDataset("validation", max_length=config.max_length)
    if not len(train):
        raise ValueError("Dolly training dataset is empty after filtering.")
    if not len(validation):
        raise ValueError("Dolly validation dataset is empty after filtering.")
    kwargs = {"batch_size": config.batch_size, "num_workers": config.num_workers,
              "pin_memory": torch.cuda.is_available()}
    return (DataLoader(train, shuffle=True, **kwargs),
            DataLoader(validation, shuffle=False, **kwargs))


def get_logits(output):
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, dict) and "logits" in output:
        return output["logits"]
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    if torch.is_tensor(output):
        return output
    raise TypeError("Model output must contain [batch, sequence, vocabulary] logits.")


def causal_lm_loss(logits, labels):
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError(f"Incompatible logits {tuple(logits.shape)} and labels {tuple(labels.shape)}")
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    if not (shift_labels != -100).any():
        raise ValueError("Batch contains no supervised targets after causal shifting.")
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_labels.view(-1), ignore_index=-100)


def _model_forward(model, input_ids, attention_mask):
    """Pass supported keywords while never silently dropping labels/loss."""
    try:
        signature = inspect.signature(model.forward)
        params = signature.parameters
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    except (TypeError, ValueError, AttributeError):
        params, accepts_kwargs = {}, False
    kwargs = {"input_ids": input_ids}
    if accepts_kwargs or "attention_mask" in params:
        kwargs["attention_mask"] = attention_mask
    elif "input_ids" not in params:
        return model(input_ids)
    return model(**kwargs)


def run_epoch(model, loader, device, optimizer=None, log_every=10):
    training = optimizer is not None
    model.train(training)
    total_loss, total_batches = 0.0, 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for step, batch in enumerate(loader, 1):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = get_logits(_model_forward(model, input_ids, attention_mask))
            loss = causal_lm_loss(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach())
            total_batches += 1
            if training and log_every > 0 and step % log_every == 0:
                print(f"Step {step}/{len(loader)} loss={loss.detach().item():.4f}")
    if not total_batches:
        raise ValueError("DataLoader produced no batches.")
    return total_loss / total_batches


def _append_log(path, row):
    new_file = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def fine_tune(model, train_loader, val_loader, optimizer, device, config, output_dir, log_path):
    """Train with periodic validation, log the same columns as pretraining, keep best and latest checkpoints."""
    output_dir, log_path = Path(output_dir), Path(log_path)
    log_path.unlink(missing_ok=True)
    best_loss, step, tokens_seen, start = float("inf"), 0, 0, time.perf_counter()
    window_loss, window_steps = 0.0, 0

    def validate(epoch):
        nonlocal best_loss, window_loss, window_steps
        val_loss = run_epoch(model, val_loader, device)
        model.train()
        train_loss = window_loss / window_steps if window_steps else None
        window_loss, window_steps = 0.0, 0
        _append_log(log_path, {"step": step, "tokens_seen": tokens_seen,
                               "train_loss": "" if train_loss is None else f"{train_loss:.6f}",
                               "val_loss": f"{val_loss:.6f}", "lr": config.learning_rate,
                               "elapsed_s": f"{time.perf_counter() - start:.2f}"})
        print(f"Step {step} epoch {epoch}: train_loss={train_loss} val_loss={val_loss:.4f}")
        save_checkpoint(model, optimizer, epoch, val_loss, config, output_dir / "latest.pt", step=step)
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, config, output_dir / "best.pt", step=step)

    model.train()
    for epoch in range(1, config.epochs + 1):
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = causal_lm_loss(get_logits(_model_forward(model, input_ids, attention_mask)), labels)
            loss.backward()
            optimizer.step()
            step += 1
            tokens_seen += int(attention_mask.sum())
            window_loss += float(loss.detach())
            window_steps += 1
            if config.log_every > 0 and step % config.log_every == 0:
                print(f"Step {step} loss={loss.detach().item():.4f}")
            if config.eval_interval > 0 and step % config.eval_interval == 0:
                validate(epoch)
            if config.max_steps and step >= config.max_steps:
                break
        if window_steps:   # end of epoch, unless the last step was just validated
            validate(epoch)
        if config.max_steps and step >= config.max_steps:
            break
    return best_loss


def model_config_for(model):
    blocks = getattr(model, "blocks", ())
    block_heads = getattr(getattr(blocks[0], "attention", None), "num_heads", 4) if blocks else 4
    return {
        "vocab_size": int(model.vocab_size),
        "embedding_dim": int(getattr(model, "embedding_dim", 256)),
        "num_heads": int(getattr(model, "num_heads", block_heads)),
        "num_layers": int(getattr(model, "num_layers", len(blocks) or 4)),
        "context_length": int(getattr(model, "max_seq_len", getattr(model, "block_size", 256))),
        "dropout": 0.1,
        "attention": "linear" if "Efficient" in type(model).__name__ else "standard",
    }


def save_checkpoint(model, optimizer, epoch, val_loss, config, path, step=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "step": step, "model_state_dict": model.state_dict(),
                "model_state": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_loss": val_loss, "config": model_config_for(model),
                "training_config": asdict(config),
                "tokenizer_path": "artifacts/tokenizer/tokenizer.json",
                "stage": "sft"}, path)
    return path


def load_pretrained_checkpoint(model, path, device):
    path = Path(path)
    payload = read_checkpoint(path, device)
    state = (payload.get("model_state_dict", payload.get("model_state", payload.get("state_dict")))
             if isinstance(payload, dict) else payload)
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format in {path}; expected state_dict or model_state_dict.")
    try:
        result = model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"Checkpoint is incompatible with the supplied model: {exc}") from exc
    return payload, result


def read_checkpoint(path, device):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint is missing: {path}. Run Member 1 pretraining first.")
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError as exc:
        raise RuntimeError("Safe checkpoint loading requires a PyTorch version with weights_only support.") from exc
    except Exception as exc:
        raise ValueError(f"Cannot safely read checkpoint {path}: {exc}") from exc


def load_factory(spec, model_config):
    if ":" not in spec:
        raise ValueError("--model-factory must use package.module:callable format")
    module_name, attr_name = spec.split(":", 1)
    try:
        factory = getattr(importlib.import_module(module_name), attr_name)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(f"Cannot import model factory {spec!r}; Member 1 model code is unavailable or the path is wrong.") from exc
    try:
        signature = inspect.signature(factory)
        model = factory(model_config) if signature.parameters else factory()
    except (TypeError, ValueError):
        model = factory()
    if not isinstance(model, torch.nn.Module):
        raise TypeError("Model factory must return torch.nn.Module.")
    return model


def validate_model_compatibility(model, max_length, expected_vocab_size=8000):
    """Check exposed model vocabulary/context metadata when available.

    Common config/attribute names are supported. Unknown metadata is an
    integration error because the tokenizer/model dimensions cannot then be
    verified safely.
    """
    holders = [model, getattr(model, "config", None), getattr(model, "args", None)]

    def find(names):
        for holder in holders:
            if holder is None:
                continue
            for name in names:
                value = holder.get(name) if isinstance(holder, dict) else getattr(holder, name, None)
                if isinstance(value, int):
                    return value
        return None

    vocab = find(("vocab_size", "n_vocab"))
    context = find(("max_length", "max_seq_len", "max_position_embeddings", "block_size", "context_length"))
    if vocab is None:
        embedding = getattr(model, "get_input_embeddings", lambda: None)()
        vocab = getattr(embedding, "num_embeddings", None)
    if vocab is None:
        raise ValueError("Cannot verify model vocabulary size; expose vocab_size in model or model.config.")
    if vocab != expected_vocab_size:
        raise ValueError(f"Model vocabulary size {vocab} does not match tokenizer size {expected_vocab_size}.")
    if context is None:
        raise ValueError("Cannot verify model context length; expose block_size/max_seq_len in model config.")
    if max_length > context:
        raise ValueError(f"SFT max_length {max_length} exceeds model context length {context}.")
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune the AFML causal LM on Dolly.")
    parser.add_argument("--model-factory", default="src.model:build_model", help="Model factory as package.module:callable")
    parser.add_argument("--pretrained-checkpoint", type=Path,
                        default=CHECKPOINT_DIR / "baseline_standard" / "best_checkpoint.pt")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=500, help="validate every N steps and at each epoch end")
    parser.add_argument("--max-steps", type=int, default=0, help="stop early after N steps (0 = run all epochs)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--log-path", type=Path, default=ROOT / "results" / "metrics" / "sft_train_log.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    config = SFTConfig(max_length=args.max_length, batch_size=args.batch_size,
                       epochs=args.epochs, learning_rate=args.learning_rate,
                       weight_decay=args.weight_decay, seed=args.seed,
                       log_every=args.log_every, eval_interval=args.eval_interval,
                       max_steps=args.max_steps)
    if config.max_length < 3 or config.batch_size < 1 or config.epochs < 1:
        raise ValueError("max_length must be >=3; batch_size and epochs must be positive")
    set_seed(config.seed)
    device = get_device()
    print(f"Device: {device}\nConfig: {json.dumps(asdict(config), indent=2)}")
    pretrained_payload = read_checkpoint(args.pretrained_checkpoint, device)
    model_config = {"vocab_size": 8000, "embedding_dim": 256, "num_heads": 4,
                    "num_layers": 4, "context_length": 256, "dropout": 0.1,
                    "attention": "standard"}
    if isinstance(pretrained_payload, dict):
        saved_model_config = pretrained_payload.get("config", {})
        if isinstance(saved_model_config, dict):
            if isinstance(saved_model_config.get("model"), dict):
                saved_model_config = saved_model_config["model"]
            aliases = {
                "vocab_size": ("vocab_size",),
                "embedding_dim": ("embedding_dim", "d_model"),
                "num_heads": ("num_heads", "n_head"),
                "num_layers": ("num_layers", "n_layer"),
                "context_length": ("context_length", "max_seq_len", "block_size"),
                "dropout": ("dropout",),
                "attention": ("attention",),
            }
            for target, choices in aliases.items():
                for key in choices:
                    if key in saved_model_config:
                        model_config[target] = saved_model_config[key]
                        break
    model = load_factory(args.model_factory, model_config)
    validate_model_compatibility(model, config.max_length)
    state = (pretrained_payload.get("model_state_dict", pretrained_payload.get("model_state", pretrained_payload.get("state_dict")))
             if isinstance(pretrained_payload, dict) else pretrained_payload)
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format in {args.pretrained_checkpoint}.")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"Checkpoint is incompatible with the supplied model: {exc}") from exc
    model.to(device)
    train_loader, val_loader = create_dataloaders(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    fine_tune(model, train_loader, val_loader, optimizer, device, config, args.output_dir, args.log_path)


if __name__ == "__main__":
    main()
