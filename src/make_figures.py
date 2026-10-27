"""Member 3 — report figures from saved results only (no numbers are typed in by hand).

python src/make_figures.py loss   --logs results/metrics/pre_std_v1_train_log.csv results/metrics/pre_lin_v1_train_log.csv \
                                  --labels "Standard attention" "Linear attention" --out results/figures/pretrain_loss.png
python src/make_figures.py bench  --bench results/metrics/bench_sweep_benchmark.json --out-prefix results/figures/bench
python src/make_figures.py bars   --files a.json b.json --labels Pretrained Fine-tuned --metric mean_total_out_of_8 ...
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots 1..4 in fixed order (validated reference palette, light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2)


def _save(fig, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def read_train_log(path):
    steps, train, val_x, val = [], [], [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            x = float(r.get("tokens_seen") or r["step"])
            val_value = r.get("val_loss") or r.get("validation_loss")   # Member 1 logs "validation_loss"
            if r.get("train_loss"):
                steps.append(x)
                train.append(float(r["train_loss"]))
            if val_value:
                val_x.append(x)
                val.append(float(val_value))
    return steps, train, val_x, val


def plot_loss(logs, labels, out, title="Pre-training loss"):
    """Training (thin, light) and validation (solid, markers) loss vs tokens seen; one colour per run."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    use_tokens = True
    for i, (path, label) in enumerate(zip(logs, labels)):
        x, tr, vx, va = read_train_log(path)
        with open(path) as fh:
            use_tokens &= "tokens_seen" in (csv.DictReader(fh).fieldnames or [])
        c = SERIES[i]
        if tr:
            ax.plot(x, tr, color=c, linewidth=1, alpha=0.35, label=f"{label} — train")
        if va:
            ax.plot(vx, va, color=c, linewidth=2, marker="o", markersize=4, label=f"{label} — validation")
            ax.annotate(f"{va[-1]:.3f}", (vx[-1], va[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
    _style(ax, title, "tokens seen" if use_tokens else "step", "cross-entropy loss (nats / token)")
    ax.legend(frameon=False, fontsize=9)
    _save(fig, out)


def plot_benchmark(bench_json, out_prefix):
    res = json.loads(Path(bench_json).read_text())
    variants = res["variants"]
    names = list(variants)
    hw = res["hardware"]

    # forward latency and memory vs sequence length (one chart each — never a second y-axis)
    for metric, ylabel, fname in [("forward_ms_median", "forward latency (ms, median)", "forward_latency"),
                                  ("forward_peak_mem_mb", "peak GPU memory (MB)", "forward_memory")]:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        plotted = False
        for i, n in enumerate(names):
            pts = sorted((int(T), f[metric]) for T, f in variants[n]["forward"].items() if f.get(metric) is not None)
            if not pts:
                continue
            xs, ys = zip(*pts)
            label = variants[n].get("attention") or n
            ax.plot(xs, ys, color=SERIES[i], linewidth=2, marker="o", markersize=5, label=label)
            ax.annotate(label, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=9, color=INK)
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xscale("log", base=2)
        all_T = sorted({int(T) for n in names for T in variants[n]["forward"]})
        ax.set_xticks(all_T, [str(T) for T in all_T])
        ax.set_yscale("log")
        _style(ax, f"{ylabel.split(' (')[0].capitalize()} vs sequence length — batch {res['batch_size']}, {hw}",
               "sequence length (tokens, log scale)", ylabel + " — log scale")
        ax.legend(frameon=False, fontsize=9)
        _save(fig, f"{out_prefix}_{fname}.png")

    # single-number comparisons: separate small bar charts so each keeps its own honest axis
    for key, ylabel, fname in [("train_tokens_per_s", "training throughput (tokens / s)", "train_throughput"),
                               ("gen_ms_per_token_median", "generation latency (ms / token)", "gen_latency")]:
        vals = [(variants[n].get("attention") or n, variants[n][key]) for n in names if variants[n].get(key)]
        if not vals:
            continue
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        labels, ys = zip(*vals)
        bars = ax.bar(labels, ys, color=SERIES[:len(ys)], width=0.55, edgecolor="white", linewidth=2)
        for b, y in zip(bars, ys):
            ax.annotate(f"{y:,.0f}" if y >= 100 else f"{y:.2f}", (b.get_x() + b.get_width() / 2, y), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9, color=INK)
        _style(ax, ylabel.split(" (")[0].capitalize(), "", ylabel)
        ax.grid(axis="x", visible=False)
        _save(fig, f"{out_prefix}_{fname}.png")


def plot_bars(files, labels, metric, ylabel, title, out):
    vals = []
    for f in files:
        d = json.loads(Path(f).read_text())
        v = d
        for part in metric.split("."):
            v = v[part]
        vals.append(float(v))
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    bars = ax.bar(labels, vals, color=SERIES[:len(vals)], width=0.55, edgecolor="white", linewidth=2)
    for b, y in zip(bars, vals):
        ax.annotate(f"{y:.3g}", (b.get_x() + b.get_width() / 2, y), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)
    _style(ax, title, "", ylabel)
    ax.grid(axis="x", visible=False)
    _save(fig, out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("loss")
    p.add_argument("--logs", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--title", default="Pre-training loss")
    p.add_argument("--out", required=True)
    p = sub.add_parser("bench")
    p.add_argument("--bench", required=True)
    p.add_argument("--out-prefix", default="results/figures/bench")
    p = sub.add_parser("bars")
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--metric", required=True, help="dotted key into each JSON, e.g. perplexity")
    p.add_argument("--ylabel", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "loss":
        plot_loss(a.logs, a.labels, a.out, a.title)
    elif a.cmd == "bench":
        plot_benchmark(a.bench, a.out_prefix)
    else:
        plot_bars(a.files, a.labels, a.metric, a.ylabel, a.title, a.out)


if __name__ == "__main__":
    main()
