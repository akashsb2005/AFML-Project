
import argparse
import json
import random
import time
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

SRC = Path(__file__).resolve().parent
PROJECT = SRC.parent
sys.path.insert(0, str(SRC))

from model import SmallLanguageModel
from dataset import TinyStoriesDataset


def evaluate(model, loader, device, max_batches=30):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break

            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["attention_mask"].to(device)

            logits, _ = model(ids, attention_mask=mask)

            losses = torch.nn.functional.cross_entropy(
                logits.reshape(-1, model.vocab_size),
                labels.reshape(-1),
                reduction="none",
                ignore_index=-100
            )

            valid = labels.reshape(-1) != -100
            total_loss += losses[valid].sum().item()
            total_tokens += valid.sum().item()

    model.train()

    if total_tokens == 0:
        raise RuntimeError("No valid tokens in evaluation.")

    return total_loss / total_tokens


def save_checkpoint(path, step, model, optimizer, config,
                    train_loss, val_loss):
    state = {
        "step": step,
        "model_state_dict": model.state_dict(),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    data_dir = PROJECT / "data" / "processed"
    if not (data_dir / "tiny_stories_train.jsonl").exists():
        raise FileNotFoundError("TinyStories training file not found.")
    if not (data_dir / "tiny_stories_validation.jsonl").exists():
        raise FileNotFoundError("TinyStories validation file not found.")

    train_dataset = TinyStoriesDataset(split="train", block_size=128)
    val_dataset = TinyStoriesDataset(split="validation", block_size=128)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, drop_last=True, num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    model = SmallLanguageModel(
        vocab_size=8000,
        embedding_dim=256,
        num_heads=4,
        num_layers=4,
        max_seq_len=256,
        dropout=0.1
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01
    )

    drive = Path("/content/drive/MyDrive/afml-slm-project")
    if not drive.exists():
        raise RuntimeError("Google Drive project folder not found.")

    checkpoint_dir = drive / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "last_checkpoint.pt"

    log_path = PROJECT / "results/metrics/pretraining_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if not log_path.exists():
        log_path.write_text(
            "step,train_loss,validation_loss,elapsed_seconds\n",
            encoding="utf-8"
        )

    start_step = 0

    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Cannot resume; checkpoint missing: {checkpoint_path}"
            )

        saved = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        start_step = int(saved["step"])

        if start_step >= args.max_steps:
            raise ValueError(
                f"Checkpoint is at step {start_step}; "
                f"--max_steps must be greater than that."
            )

        if "python_rng_state" in saved:
            random.setstate(saved["python_rng_state"])
        if "torch_rng_state" in saved:
            torch.set_rng_state(saved["torch_rng_state"].cpu())
        if torch.cuda.is_available() and "cuda_rng_state_all" in saved:
            torch.cuda.set_rng_state_all(saved["cuda_rng_state_all"])

        print("Resumed checkpoint at step:", start_step)
    else:
        print("Starting a fresh training run at step 0.")

    print("Training blocks:", len(train_dataset))
    print("Validation blocks:", len(val_dataset))
    print("Target final step:", args.max_steps)

    model.train()
    step = start_step
    running_losses = []
    started = time.time()
    train_iterator = iter(train_loader)

    while step < args.max_steps:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["attention_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)

        _, loss = model(ids, targets=labels, attention_mask=mask)

        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"Invalid loss at step {step + 1}")

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )

        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Invalid gradients at step {step + 1}")

        optimizer.step()
        step += 1
        running_losses.append(loss.item())

        if step % args.eval_interval == 0 or step == args.max_steps:
            train_loss = sum(running_losses) / len(running_losses)
            val_loss = evaluate(model, val_loader, device, max_batches=30)
            elapsed = time.time() - started

            print(
                f"Step {step}/{args.max_steps} | "
                f"train loss: {train_loss:.4f} | "
                f"validation loss: {val_loss:.4f} | "
                f"elapsed: {elapsed:.1f}s"
            )

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{step},{train_loss:.6f},"
                    f"{val_loss:.6f},{elapsed:.2f}\n"
                )

            running_losses.clear()

            save_checkpoint(
                checkpoint_path, step, model, optimizer,
                vars(args), train_loss, val_loss
            )
            print("Checkpoint saved:", checkpoint_path)

    print("Training finished at step", step)
    print("Log:", log_path)


if __name__ == "__main__":
    main()
