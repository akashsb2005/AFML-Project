"""Member 3 — efficiency benchmark for standard vs efficient attention (protocol §5).

Speed and memory do not depend on the learned weights, so both variants are built from their configs with
random weights, on the same device, with the same batch size, sequence lengths, dtype and seed. Quality
(perplexity) is measured separately with evaluate.py on the trained checkpoints.

Measured per variant
  params                        total parameter count
  train_tokens_per_s            forward + backward + AdamW step, after warm-up
  train_peak_mem_mb             peak allocated GPU memory during training steps (CUDA only)
  forward_ms[T]                 inference forward latency for a [batch, T] input, for each T in --seq-lens
  forward_peak_mem_mb[T]        peak memory of that forward pass (CUDA only)
  gen_ms_per_token              autoregressive generation with the shared decoding loop (model_api.generate)

Usage
  python src/benchmark.py --configs configs/baseline.yaml configs/efficient_attention.yaml --label bench_t4
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import model_api  # noqa: E402


def load_model_config(path: str, overrides: dict | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg = dict(cfg["model"]) if "model" in cfg else dict(cfg)
    cfg.update(overrides or {})
    return cfg


def parse_overrides(items) -> dict:
    """["context_length=2048"] -> {"context_length": 2048}; applied identically to every variant."""
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def _reset_peak(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _peak_mb(device):
    return torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None


def bench_training(model, cfg, device, batch_size, seq_len, warmup, steps, seed):
    torch.manual_seed(seed)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randint(0, cfg["vocab_size"], (batch_size, seq_len + 1), device=device)
    inp, tgt = x[:, :-1], x[:, 1:]

    def step():
        logits = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(warmup):
        step()
    model_api.synchronize(device)
    _reset_peak(device)
    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        step()
        model_api.synchronize(device)
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    return {"train_step_ms_median": 1000 * med, "train_tokens_per_s": batch_size * seq_len / med,
            "train_peak_mem_mb": _peak_mb(device)}


@torch.no_grad()
def bench_forward(model, cfg, device, batch_size, seq_lens, warmup, repeats, seed):
    model.eval()
    out = {}
    for T in seq_lens:
        torch.manual_seed(seed)
        x = torch.randint(0, cfg["vocab_size"], (batch_size, T), device=device)
        try:
            for _ in range(warmup):
                model(x)
            model_api.synchronize(device)
            _reset_peak(device)
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                model(x)
                model_api.synchronize(device)
                times.append(time.perf_counter() - t0)
            out[T] = {"forward_ms_median": 1000 * statistics.median(times), "forward_peak_mem_mb": _peak_mb(device)}
        except (RuntimeError, AssertionError, IndexError) as e:  # OOM or T beyond the positional table
            out[T] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return out


def bench_generation(model, cfg, device, prompt_len, new_tokens, repeats, seed):
    model.eval()
    torch.manual_seed(seed)
    prompt = torch.randint(0, cfg["vocab_size"], (prompt_len,)).tolist()
    ctx = cfg["context_length"]
    model_api.generate(model, prompt, 4, ctx, eos_id=None)  # warm-up
    per_tok = []
    for _ in range(repeats):
        r = model_api.generate(model, prompt, new_tokens, ctx, eos_id=None)
        per_tok.append(1000 * r["latency_s"] / new_tokens)
    return {"gen_prompt_len": prompt_len, "gen_new_tokens": new_tokens,
            "gen_ms_per_token_median": statistics.median(per_tok)}


def run(config_paths, label, device_name="auto", batch_size=16, train_seq_len=None, seq_lens=None,
        warmup=3, steps=10, repeats=10, gen_prompt_len=32, gen_new_tokens=64, seed=0,
        builder="auto", out_dir="results/metrics", overrides=None) -> dict:
    device = model_api.pick_device(device_name)
    results = {"label": label, "hardware": model_api.hardware_string(device), "torch": torch.__version__,
               "batch_size": batch_size, "seed": seed, "overrides": overrides or {}, "variants": {}}
    for path in config_paths:
        cfg = load_model_config(path, overrides)
        torch.manual_seed(seed)
        model = model_api.build_model(cfg, builder).to(device)
        # training is always timed at the real training length (the un-overridden context_length),
        # even when context_length is raised for the forward-latency sweep
        T_train = train_seq_len or load_model_config(path)["context_length"]
        lens = seq_lens or [cfg["context_length"]]
        r = {"config": path, "attention": cfg.get("attention"), "model_config": cfg,
             "params": model_api.count_parameters(model)["total"], "train_seq_len": T_train}
        try:
            r.update(bench_training(model, cfg, device, batch_size, T_train, warmup, steps, seed))
        except RuntimeError as e:  # e.g. out of memory: record it instead of aborting the whole benchmark
            r.update({"train_error": f"{type(e).__name__}: {str(e)[:160]}", "train_tokens_per_s": None,
                      "train_peak_mem_mb": None})
            if device.type == "cuda":
                torch.cuda.empty_cache()
        r["forward"] = bench_forward(model, cfg, device, batch_size, lens, warmup, repeats, seed)
        r.update(bench_generation(model, cfg, device, gen_prompt_len, gen_new_tokens, max(3, repeats // 3), seed))
        results["variants"][Path(path).stem] = r
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{label}_benchmark.json").write_text(json.dumps(results, indent=2))
    with open(out / f"{label}_benchmark_forward.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "attention", "seq_len", "batch_size", "forward_ms_median", "forward_peak_mem_mb", "error"])
        for name, r in results["variants"].items():
            for T, f in r["forward"].items():
                w.writerow([name, r["attention"], T, batch_size, f.get("forward_ms_median"),
                            f.get("forward_peak_mem_mb"), f.get("error", "")])
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--train-seq-len", type=int, default=None)
    ap.add_argument("--seq-lens", type=int, nargs="+", default=None,
                    help="forward-latency sweep; needs context_length >= max T in the configs")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--gen-prompt-len", type=int, default=32)
    ap.add_argument("--gen-new-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--builder", default="auto")
    ap.add_argument("--override", nargs="*", default=[],
                    help="e.g. context_length=2048 (applied to all variants, for the sequence-length sweep)")
    a = ap.parse_args(argv)
    res = run(a.configs, a.label, a.device, a.batch_size, a.train_seq_len, a.seq_lens, a.warmup, a.steps,
              a.repeats, a.gen_prompt_len, a.gen_new_tokens, a.seed, a.builder, overrides=parse_overrides(a.override))
    for name, r in res["variants"].items():
        train = f"{r['train_tokens_per_s']:.0f} tok/s" if r.get("train_tokens_per_s") else r.get("train_error")
        print(f"{name:24s} params={r['params']:,} train={train} "
              f"gen={r['gen_ms_per_token_median']:.2f} ms/token peak_mem={r['train_peak_mem_mb']}")


if __name__ == "__main__":
    main()
