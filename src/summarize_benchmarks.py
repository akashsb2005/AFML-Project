"""Turn the raw GPU benchmark dump into report tables + benchmark-format JSON for make_figures.

python src/summarize_benchmarks.py results/metrics/gpu_t4_benchmarks_raw.json
"""
import json
import statistics
import sys
from pathlib import Path

raw = json.loads(Path(sys.argv[1]).read_text())
rows, out_dir = raw["rows"], Path(sys.argv[1]).parent
by = {(r["run"], r["attention"]): r for r in rows}
lines = [f"Source: `{Path(sys.argv[1]).name}` — {raw['source']}", ""]

# 1) noise: repeated identical runs (batch 8, T 128)
lines += ["### Run-to-run variation (batch 8, T = 128, 4 identical runs)", "",
          "| Attention | train tokens/s (mean ± sd) | min–max | gen ms/token (mean) |", "|---|---|---|---|"]
for att in ("standard", "linear"):
    rs = [r for r in rows if r["attention"] == att and (r["run"].startswith("gpu_repeat") or r["run"] == "gpu_train_B8_T128")]
    tok = [r["train_tok_s"] for r in rs]
    lines.append(f"| {att} | {statistics.mean(tok):,.0f} ± {statistics.stdev(tok):,.0f} | {min(tok):,.0f}–{max(tok):,.0f} | "
                 f"{statistics.mean(r['gen_ms_tok'] for r in rs):.2f} |")

# 2) training throughput / memory grid
lines += ["", "### Training step (forward + backward + AdamW)", "",
          "| Batch | T | Standard tokens/s | Linear tokens/s | Linear / standard | Standard peak MB | Linear peak MB |",
          "|---|---|---|---|---|---|---|"]
for B, T in ((8, 128), (8, 256), (32, 128), (32, 256)):
    s, l = by[(f"gpu_train_B{B}_T{T}", "standard")], by[(f"gpu_train_B{B}_T{T}", "linear")]
    lines.append(f"| {B} | {T} | {s['train_tok_s']:,.0f} | {l['train_tok_s']:,.0f} | {l['train_tok_s'] / s['train_tok_s']:.2f}× | "
                 f"{s['train_peak_mb']:,.0f} | {l['train_peak_mb']:,.0f} |")

# 3) forward sweep
s, l = by[("gpu_sweep_B8", "standard")], by[("gpu_sweep_B8", "linear")]
lines += ["", "### Inference forward pass vs sequence length (batch 8, context raised to 2048 for both)", "",
          "| T | Standard ms | Linear ms | Linear / standard | Standard ×/doubling | Linear ×/doubling | Standard peak MB | Linear peak MB |",
          "|---|---|---|---|---|---|---|---|"]
prev = None
for T in sorted(s["forward"], key=int):
    fs, fl = s["forward"][T], l["forward"][T]
    gs = f"{fs['forward_ms_median'] / prev[0]:.2f}" if prev else "—"
    gl = f"{fl['forward_ms_median'] / prev[1]:.2f}" if prev else "—"
    lines.append(f"| {T} | {fs['forward_ms_median']:.1f} | {fl['forward_ms_median']:.1f} | "
                 f"{fl['forward_ms_median'] / fs['forward_ms_median']:.2f}× | {gs} | {gl} | "
                 f"{fs['forward_peak_mem_mb']:,.0f} | {fl['forward_peak_mem_mb']:,.0f} |")
    prev = (fs["forward_ms_median"], fl["forward_ms_median"])
(out_dir / "gpu_t4_summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# benchmark-format JSON (for make_figures.py bench) from the sweep + the B8/T128 training run
bench = {"label": "gpu_t4", "hardware": "Tesla T4", "batch_size": 8, "variants": {}}
for name, att in (("baseline", "standard"), ("efficient_attention", "linear")):
    sw, tr = by[("gpu_sweep_B8", att)], by[("gpu_train_B8_T128", att)]
    bench["variants"][name] = {"attention": att, "params": tr["params"], "train_tokens_per_s": tr["train_tok_s"],
                               "gen_ms_per_token_median": tr["gen_ms_tok"], "forward": sw["forward"]}
(out_dir / "gpu_t4_benchmark.json").write_text(json.dumps(bench, indent=2))
