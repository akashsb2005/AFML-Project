"""Pretrain the causal linear-attention TinyStories language model."""

import argparse
import random
import time
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SRC = Path(__file__).resolve().parent
PROJECT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from efficient_model import EfficientSmallLanguageModel
from dataset import TinyStoriesDataset


def evaluate(model, loader, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(ids, attention_mask=mask)
            losses = F.cross_entropy(
                logits.reshape(-1, model.vocab_size),
                labels.reshape(-1),
                reduction="none",
                ignore_index=-100,
            )
            valid = labels.reshape(-1) != -100
            total_loss += losses[valid].sum().item()
            total_tokens += valid.sum().item()
    model.train()
    if total_tokens == 0:
        raise RuntimeError("No valid tokens in evaluation.")
    return total_loss / total_tokens


def save_checkpoint(path, step, model, optimizer, config, train_loss, val_loss):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "train_loss": train_loss,
        "validation_loss": val_loss,
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--eval_batches", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default="efficient_linear")
    parser.add_argument("--checkpoint_dir", type=Path, default=PROJECT / "checkpoints" / "efficient")
    parser.add_argument("--log_path", type=Path, default=PROJECT / "results" / "metrics" / "efficient_pretraining.csv")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if min(args.max_steps, args.batch_size, args.block_size,
           args.context_length, args.eval_interval) < 1:
        parser.error("steps, batch size, block size, context length and eval interval must be positive")
    if args.block_size != args.context_length:
        parser.error("--block_size must equal --context_length for this dataset/model interface")
    if not args.run_name.replace("_", "").replace("-", "").isalnum():
        parser.error("Invalid run_name")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_dataset = TinyStoriesDataset(split="train", block_size=args.block_size)
    val_dataset = TinyStoriesDataset(split="validation", block_size=args.block_size)
    if len(train_dataset) < args.batch_size:
        raise RuntimeError("Training dataset has fewer blocks than the batch size.")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    model = EfficientSmallLanguageModel(
        vocab_size=8000, embedding_dim=256, num_heads=4,
        num_layers=4, max_seq_len=args.context_length, dropout=0.1,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / "last_checkpoint.pt"
    if not args.log_path.exists():
        args.log_path.write_text(
            "step,train_loss,validation_loss,elapsed_seconds\n", encoding="utf-8"
        )

    start_step = 0
    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Cannot resume; checkpoint missing: {checkpoint_path}")
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(saved.get("model_state", saved.get("model_state_dict")))
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        start_step = int(saved["step"])
        if start_step >= args.max_steps:
            raise ValueError("--max_steps must be greater than the checkpoint step")
        if "python_rng_state" in saved:
            random.setstate(saved["python_rng_state"])
        if "torch_rng_state" in saved:
            torch.set_rng_state(saved["torch_rng_state"].cpu())
        if device.type == "cuda" and "cuda_rng_state_all" in saved:
            torch.cuda.set_rng_state_all(saved["cuda_rng_state_all"])
        print("Resumed checkpoint at step:", start_step)

    config = vars(args).copy()
    config.update({
        "attention": "causal_linear_elu_plus_one",
        "vocab_size": 8000,
        "embedding_dim": 256,
        "num_heads": 4,
        "num_layers": 4,
        "dropout": 0.1,
        "dataset": "TinyStories",
    })
    print("Training blocks:", len(train_dataset))
    print("Validation blocks:", len(val_dataset))
    print("Unique parameters:", sum(p.numel() for p in model.parameters()))
    print("Target final step:", args.max_steps)
    print("Checkpoint:", checkpoint_path)
    print("Log:", args.log_path)

    model.train()
    step = start_step
    running_losses = []
    started = time.time()
    iterator = iter(train_loader)
    while step < args.max_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["attention_mask"].to(device)
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
        running_losses.append(loss.item())

        if step % args.eval_interval == 0 or step == args.max_steps:
            train_loss = sum(running_losses) / len(running_losses)
            val_loss = evaluate(model, val_loader, device, args.eval_batches)
            elapsed = time.time() - started
            print(
                f"Step {step}/{args.max_steps} | train loss: {train_loss:.4f} | "
                f"validation loss: {val_loss:.4f} | elapsed: {elapsed:.1f}s"
            )
            with args.log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"{step},{train_loss:.6f},{val_loss:.6f},{elapsed:.2f}\n")
            save_checkpoint(
                checkpoint_path, step, model, optimizer, config, train_loss, val_loss
            )
            running_losses.clear()
            print("Checkpoint saved:", checkpoint_path)

    print("Efficient pretraining finished at step", step)
    print("Log:", args.log_path)


if __name__ == "__main__":
    main()
