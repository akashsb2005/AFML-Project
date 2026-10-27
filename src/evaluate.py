"""Member 3 — evaluation (protocol: report/evaluation_protocol.md).

Sub-commands
  perplexity     held-out next-token NLL and perplexity on a token stream (TinyStories val/test)
  sft-loss       held-out response-only NLL on instruction data (Dolly test)
  generate       fixed prompts -> generations with fixed decoding settings + automatic text statistics
  rubric-sheet   blind, shuffled scoring sheet for several models' generations
  rubric-score   aggregate a filled-in scoring sheet per model

Every sub-command writes JSON/CSV into results/metrics/ and never trains or selects anything.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import model_api  # noqa: E402

RUBRIC_CRITERIA = ("relevance", "format", "answers", "coherence")


# =========================================================================================== perplexity
def load_token_stream(path: str, dtype: str = "uint16") -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path, mmap_mode="r")
    return np.memmap(path, dtype=np.dtype(dtype), mode="r")


def _windows(n_targets: int, context_length: int, stride: int):
    """Yield (begin, end, first_scored) so that input = tok[begin:end], targets = tok[begin+1:end+1] and
    only targets with index >= first_scored are counted. Every target is scored exactly once; with
    stride < context_length each scored token sees at least context_length - stride tokens of history."""
    scored = 0
    while scored < n_targets:
        end = min(scored + stride, n_targets)
        begin = max(0, end - context_length)
        yield begin, end, scored
        scored = end


@torch.no_grad()
def stream_nll(model, tokens: np.ndarray, context_length: int, stride: int | None = None,
               batch_size: int = 16, device: torch.device | None = None) -> dict:
    """Token-weighted mean negative log-likelihood (nats) over a token stream and its perplexity."""
    device = device or next(model.parameters()).device
    stride = stride or context_length
    assert 0 < stride <= context_length
    n_targets = len(tokens) - 1
    total_nll, total_count = 0.0, 0
    pending = []

    def flush(batch):
        nonlocal total_nll, total_count
        x = torch.stack([torch.as_tensor(np.asarray(tokens[b:e], dtype=np.int64)) for b, e, _ in batch]).to(device)
        y = torch.stack([torch.as_tensor(np.asarray(tokens[b + 1:e + 1], dtype=np.int64)) for b, e, _ in batch]).to(device)
        logits = model(x).float()
        nll = F.cross_entropy(logits.transpose(1, 2), y, reduction="none")  # [B, T]
        for row, (b, e, first) in enumerate(batch):
            total_nll += float(nll[row, first - b:].double().sum())
            total_count += e - first

    for win in _windows(n_targets, context_length, stride):
        if pending and (win[1] - win[0]) != (pending[0][1] - pending[0][0]):
            flush(pending)
            pending = []
        pending.append(win)
        if len(pending) == batch_size:
            flush(pending)
            pending = []
    if pending:
        flush(pending)
    mean_nll = total_nll / total_count
    return {"mean_nll": mean_nll, "perplexity": math.exp(mean_nll), "tokens_scored": total_count,
            "context_length": context_length, "stride": stride}


def story_blocks(ids: list[int], block_size: int):
    """Exactly Member 2's TinyStoriesDataset chunking: (x, y) pairs of <= block_size tokens per story,
    consecutive blocks share one token so every target in the story is predicted once."""
    start = 0
    while start < len(ids) - 1:
        chunk = ids[start:start + block_size + 1]
        if len(chunk) >= 2:
            yield chunk[:-1], chunk[1:]
        if start + block_size >= len(ids) - 1:
            break
        start += block_size


@torch.no_grad()
def stories_nll(model, tok, texts, block_size: int, batch_size: int = 32,
                device: torch.device | None = None) -> dict:
    """Token-weighted NLL over held-out stories, each encoded as [BOS] story [EOS] (as in pre-training).
    Padding positions are excluded, so the number equals the training script's validation loss when that
    loss is also averaged over real tokens."""
    device = device or next(model.parameters()).device
    blocks = [b for t in texts for b in story_blocks(tok.encode_document(t), block_size)]
    blocks = [b for b in blocks if len(b[1]) > 0]
    total_nll, total_count, n_stories = 0.0, 0, len(texts)
    for i in range(0, len(blocks), batch_size):
        batch = blocks[i:i + batch_size]
        T = max(len(x) for x, _ in batch)
        x = torch.full((len(batch), T), tok.pad_id, dtype=torch.long)
        y = torch.full((len(batch), T), -100, dtype=torch.long)
        for r, (xi, yi) in enumerate(batch):
            x[r, :len(xi)] = torch.tensor(xi)
            y[r, :len(yi)] = torch.tensor(yi)
        logits = model(x.to(device)).float()
        y = y.to(device)
        total_nll += float(F.cross_entropy(logits.transpose(1, 2), y, ignore_index=-100, reduction="sum"))
        total_count += int((y != -100).sum())
    mean = total_nll / total_count
    return {"mean_nll": mean, "perplexity": math.exp(mean), "tokens_scored": total_count,
            "stories": n_stories, "blocks": len(blocks), "block_size": block_size}


# =========================================================================================== SFT loss
def build_sft_example(example: dict, tok, max_length: int | None = None) -> tuple[list[int], list[int], str]:
    """(input_ids, labels, source) built exactly like Member 2's DollySFTDataset: prompt and response encoded
    separately without special tokens, response + EOS supervised, prompt labels -100, and the same
    truncation rule (keep the response, cut the prompt from the left). No padding."""
    prompt = model_api.prompt_text(example)
    p_ids, r_ids = tok.encode(prompt), tok.encode(example["response"])
    if max_length is not None and len(p_ids) + len(r_ids) + 1 > max_length:
        r_ids = r_ids[:max_length - 2]
        p_ids = p_ids[-(max_length - len(r_ids) - 1):]
    ids = p_ids + r_ids + [tok.eos_id]
    return ids, [-100] * len(p_ids) + r_ids + [tok.eos_id], "DollySFTDataset rules"


@torch.no_grad()
def response_nll(model, examples: list[dict], tok, context_length: int,
                 device: torch.device | None = None) -> dict:
    """Mean NLL over response tokens only (prompt and padding excluded), token-weighted over the set."""
    device = device or next(model.parameters()).device
    total_nll, total_count, truncated, skipped, source = 0.0, 0, 0, 0, None
    for ex in examples:
        n_full = len(build_sft_example(ex, tok)[0])
        ids, labels, source = build_sft_example(ex, tok, max_length=context_length)
        truncated += n_full > len(ids)
        x = torch.tensor([ids[:-1]], device=device)
        y = torch.tensor([labels[1:]], device=device)
        if int((y != -100).sum()) == 0:
            skipped += 1
            continue
        logits = model(x).float()
        nll = F.cross_entropy(logits.transpose(1, 2), y, ignore_index=-100, reduction="sum")
        total_nll += float(nll)
        total_count += int((y != -100).sum())
    mean = total_nll / max(total_count, 1)
    return {"mean_response_nll": mean, "response_perplexity": math.exp(mean), "response_tokens": total_count,
            "examples": len(examples), "truncated": truncated, "skipped_no_response_tokens": skipped,
            "example_builder": source}


# =========================================================================================== text statistics
_WORD = re.compile(r"\w+|[^\w\s]")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def ngrams(seq, n):
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def distinct_n(seq, n) -> float:
    grams = ngrams(seq, n)
    return len(set(grams)) / len(grams) if grams else 0.0


def repetition_rate(seq, n=4) -> float:
    """Fraction of n-grams that already occurred earlier in the same text (0 = no loops)."""
    grams = ngrams(seq, n)
    if not grams:
        return 0.0
    seen, repeats = set(), 0
    for g in grams:
        repeats += g in seen
        seen.add(g)
    return repeats / len(grams)


def prompt_copy_rate(response_words, prompt_words, n=3) -> float:
    """Fraction of the response's n-grams copied from the instruction/context (high = echoing the prompt)."""
    grams = ngrams(response_words, n)
    if not grams:
        return 0.0
    src = set(ngrams(prompt_words, n))
    return sum(g in src for g in grams) / len(grams)


def rouge_l_f1(hyp, ref) -> float:
    if not hyp or not ref:
        return 0.0
    prev = [0] * (len(ref) + 1)
    for h in hyp:
        cur = [0]
        for j, r in enumerate(ref, 1):
            cur.append(prev[j - 1] + 1 if h == r else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(hyp), lcs / len(ref)
    return 2 * p * r / (p + r)


def text_stats(response: str, instruction: str, context: str = "", reference: str | None = None) -> dict:
    rw = words(response)
    out = {"n_words": len(rw), "empty": int(len(rw) == 0), "distinct_1": distinct_n(rw, 1),
           "distinct_2": distinct_n(rw, 2), "repetition_4gram": repetition_rate(rw, 4),
           "prompt_copy_3gram": prompt_copy_rate(rw, words(instruction + " " + (context or "")), 3)}
    if reference is not None:
        out["rougeL_f1_vs_reference"] = rouge_l_f1(rw, words(reference))
    return out


# =========================================================================================== generation
def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def generate_for_prompts(model, tok, prompts: list[dict], context_length: int, max_new_tokens: int = 128,
                         temperature: float = 0.0, top_k: int | None = None, seed: int = 1234,
                         raw_story_mode: bool = False) -> list[dict]:
    """raw_story_mode=True feeds the instruction text as a plain continuation prompt (used to probe the
    pretrained model on TinyStories-style openings); otherwise the SFT instruction template is applied."""
    gen = torch.Generator().manual_seed(seed)
    rows = []
    for p in prompts:
        if raw_story_mode:
            prompt_text, template = p.get("prompt") or p["instruction"], "raw story opening"
            prompt_ids = ([tok.bos_id] if tok.bos_id is not None else []) + tok.encode(prompt_text)
        else:
            prompt_text, template = model_api.prompt_text(p), "Dolly SFT prompt"
            prompt_ids = tok.encode(prompt_text)
        if "instruction" not in p:
            p = dict(p, **dict(zip(("instruction", "context"), model_api.parse_prompt(prompt_text))))
        max_prompt = context_length - 1
        prompt_truncated = len(prompt_ids) > max_prompt
        prompt_ids = prompt_ids[-max_prompt:]
        res = model_api.generate(model, prompt_ids, max_new_tokens, context_length, tok.eos_id,
                                 temperature=temperature, top_k=top_k, generator=gen)
        text = tok.decode(res["tokens"]).strip()
        row = {"id": p.get("id"), "category": p.get("category"), "instruction": p["instruction"],
               "context": p.get("context", ""), "reference": p.get("response"), "prompt_text": prompt_text,
               "template": template, "prompt_truncated": prompt_truncated, "generation": text,
               "new_tokens": len(res["tokens"]), "stopped_on_eos": res["stopped_on_eos"],
               "latency_s": res["latency_s"]}
        row.update(text_stats(text, p["instruction"], p.get("context", ""), p.get("response")))
        rows.append(row)
    return rows


def summarize_generations(rows: list[dict]) -> dict:
    keys = ["n_words", "empty", "distinct_1", "distinct_2", "repetition_4gram", "prompt_copy_3gram",
            "rougeL_f1_vs_reference", "new_tokens", "latency_s"]
    out = {"n": len(rows), "stopped_on_eos_rate": float(np.mean([r["stopped_on_eos"] for r in rows]))}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            out[f"mean_{k}"] = float(np.mean(vals))
    return out


# =========================================================================================== rubric
def make_rubric_sheet(output_files: dict[str, str], sheet_path: str, key_path: str, seed: int = 7) -> int:
    """Blind scoring: rows from all models are shuffled and given opaque ids; the model is in a separate key."""
    rows, key = [], {}
    rng = random.Random(seed)
    for label, path in output_files.items():
        for r in read_jsonl(path):
            uid = f"item_{rng.getrandbits(40):010x}"
            key[uid] = {"model": label, "prompt_id": r.get("id")}
            rows.append({"item": uid, "prompt_id": r.get("id"), "category": r.get("category"),
                         "instruction": r["instruction"], "context": r.get("context", ""),
                         "response": r["generation"], **{c: "" for c in RUBRIC_CRITERIA}, "notes": ""})
    rng.shuffle(rows)
    Path(sheet_path).parent.mkdir(parents=True, exist_ok=True)
    with open(sheet_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    Path(key_path).write_text(json.dumps(key, indent=2))
    return len(rows)


def score_rubric(sheet_path: str, key_path: str) -> dict:
    key = json.loads(Path(key_path).read_text())
    per_model: dict[str, dict] = {}
    missing = []
    with open(sheet_path) as fh:
        for row in csv.DictReader(fh):
            model = key[row["item"]]["model"]
            m = per_model.setdefault(model, {c: [] for c in RUBRIC_CRITERIA})
            for c in RUBRIC_CRITERIA:
                v = row[c].strip()
                if v == "":
                    missing.append((row["item"], c))
                    continue
                v = int(v)
                if v not in (0, 1, 2):
                    raise ValueError(f"{row['item']} {c}: scores must be 0, 1 or 2, got {v}")
                m[c].append(v)
    summary = {}
    for model, crit in per_model.items():
        s = {f"mean_{c}": float(np.mean(v)) if v else None for c, v in crit.items()}
        totals = [sum(t) for t in zip(*crit.values())] if all(crit.values()) else []
        s["mean_total_out_of_8"] = float(np.mean(totals)) if totals else None
        s["n_scored"] = min(len(v) for v in crit.values())
        summary[model] = s
    return {"per_model": summary, "missing_scores": len(missing)}


# =========================================================================================== CLI
def context_length_of(ckpt: dict, override: int | None = None) -> int:
    if override:
        return override
    cfg = ckpt.get("config", {})
    for k in ("context_length", "block_size", "max_seq_len", "seq_len"):
        if k in cfg:
            return int(cfg[k])
    raise KeyError("context length not in checkpoint config — pass --context-length")


def _load(args):
    device = model_api.pick_device(args.device)
    model, ckpt = model_api.load_checkpoint(args.ckpt, device, args.builder)
    tok = model_api.load_tokenizer(args.tokenizer or ckpt["tokenizer_path"])
    return device, model, ckpt, tok


def _save(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))
    print(json.dumps(obj, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--ckpt", required=True)
        p.add_argument("--tokenizer", default=None, help="defaults to the checkpoint's tokenizer_path")
        p.add_argument("--builder", default="src.model:build_model")
        p.add_argument("--device", default="auto")
        p.add_argument("--label", required=True, help="run id used in output file names")
        p.add_argument("--context-length", type=int, default=None, help="defaults to the checkpoint config")

    p = sub.add_parser("perplexity")
    common(p)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--stories", help="Member 2's tiny_stories_<split>.jsonl (primary)")
    g.add_argument("--tokens", help="flat token stream .bin/.npy (alternative)")
    p.add_argument("--max-stories", type=int, default=None)
    p.add_argument("--dtype", default="uint16")
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)

    p = sub.add_parser("sft-loss")
    common(p)
    p.add_argument("--data", required=True)
    p.add_argument("--exclude", nargs="*", default=[],
                   help="training/validation files: test rows whose prompt also occurs there are dropped")

    p = sub.add_parser("generate")
    common(p)
    p.add_argument("--prompts", required=True)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--raw", action="store_true", help="no instruction template (story-continuation prompts)")

    p = sub.add_parser("rubric-sheet")
    p.add_argument("--outputs", nargs="+", required=True, help="label=path/to/generations.jsonl")
    p.add_argument("--sheet", default="results/metrics/rubric_sheet.csv")
    p.add_argument("--key", default="results/metrics/rubric_key.json")

    p = sub.add_parser("rubric-score")
    p.add_argument("--sheet", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--out", default="results/metrics/rubric_summary.json")

    args = ap.parse_args(argv)
    out_dir = Path("results/metrics")

    if args.cmd == "perplexity":
        device, model, ckpt, tok = _load(args)
        ctx = context_length_of(ckpt, args.context_length)
        if args.stories:
            texts = [r["text"] for r in read_jsonl(args.stories)][:args.max_stories]
            res = stories_nll(model, tok, texts, ctx, args.batch_size, device)
        else:
            tokens = load_token_stream(args.tokens, args.dtype)
            if args.max_tokens:
                tokens = tokens[:args.max_tokens + 1]
            res = stream_nll(model, tokens, ctx, args.stride, args.batch_size, device)
        res.update({"label": args.label, "ckpt": args.ckpt, "data_file": args.stories or args.tokens,
                    "hardware": model_api.hardware_string(device)})
        _save(res, out_dir / f"{args.label}_perplexity.json")
    elif args.cmd == "sft-loss":
        device, model, ckpt, tok = _load(args)
        from src.select_eval_prompts import fingerprint
        rows = read_jsonl(args.data)
        banned = {fingerprint(r) for f in args.exclude for r in read_jsonl(f)}
        kept = [r for r in rows if fingerprint(r) not in banned]
        res = response_nll(model, kept, tok, context_length_of(ckpt, args.context_length), device)
        res.update({"label": args.label, "ckpt": args.ckpt, "data": args.data,
                    "excluded_overlapping_prompts": len(rows) - len(kept)})
        _save(res, out_dir / f"{args.label}_sft_loss.json")
    elif args.cmd == "generate":
        device, model, ckpt, tok = _load(args)
        rows = generate_for_prompts(model, tok, read_jsonl(args.prompts), context_length_of(ckpt, args.context_length),
                                    args.max_new_tokens, args.temperature, args.top_k, args.seed, args.raw)
        write_jsonl(out_dir / f"{args.label}_generations.jsonl", rows)
        summ = summarize_generations(rows)
        summ.update({"label": args.label, "ckpt": args.ckpt, "prompts": args.prompts,
                     "decoding": {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
                                  "top_k": args.top_k, "seed": args.seed}})
        _save(summ, out_dir / f"{args.label}_generation_summary.json")
    elif args.cmd == "rubric-sheet":
        files = dict(o.split("=", 1) for o in args.outputs)
        n = make_rubric_sheet(files, args.sheet, args.key)
        print(f"wrote {n} rows to {args.sheet} (key: {args.key}) — score without opening the key")
    elif args.cmd == "rubric-score":
        _save(score_rubric(args.sheet, args.key), args.out)


if __name__ == "__main__":
    main()
