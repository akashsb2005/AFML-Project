"""Member 3's adapter to the model / tokenizer / prompt template (see docs/interfaces.md).

Evaluation and benchmarking only go through these functions, so a change in Member 1's or Member 2's
interface needs to be fixed here and nowhere else.
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path
from dataclasses import dataclass

import torch



# ----------------------------------------------------------------------------------------- device helpers
def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    """Wait for queued GPU work, so wall-clock timings measure the computation itself."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def hardware_string(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return device.type


# ----------------------------------------------------------------------------------------- model loading
SRC_DIR = str(Path(__file__).resolve().parent)
# Member 2 writes the tokenizer here (src/tokenizer.py); checkpoints do not record it
DEFAULT_TOKENIZER = str(Path(__file__).resolve().parents[1] / "artifacts" / "tokenizer" / "tokenizer.json")

# Member 1's models use bare imports (`from attention import ...`), so src/ must be importable directly.
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class LogitsOnly(torch.nn.Module):
    """Wraps Member 1's models (forward returns (logits, loss)) so evaluation code gets logits only."""

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, input_ids):
        out = self.inner(input_ids)
        return out[0] if isinstance(out, (tuple, list)) else out


# Member 1 defaults (src/pretrain.py); used only when a checkpoint config lacks a model setting.
MEMBER1_DEFAULTS = dict(vocab_size=8000, embedding_dim=256, num_heads=4, num_layers=4, max_seq_len=256, dropout=0.1)


def member1_kwargs(config: dict) -> tuple[str, dict]:
    """Map either Member 1's checkpoint config or our YAML model config onto Member 1's constructors.
    Returns (attention, kwargs) with attention in {"standard", "linear"}."""
    alias = {"d_model": "embedding_dim", "n_head": "num_heads", "n_layer": "num_layers",
             "context_length": "max_seq_len"}
    given = {alias.get(k, k): v for k, v in config.items() if alias.get(k, k) in MEMBER1_DEFAULTS}
    kw = {**MEMBER1_DEFAULTS, **given}
    missing = {k: v for k, v in MEMBER1_DEFAULTS.items() if k not in given}
    att = str(config.get("attention", config.get("model", "standard"))).lower()
    attention = "linear" if ("linear" in att or "efficient" in att) else "standard"
    if missing:
        print(f"[model_api] config has no {sorted(missing)}; using Member 1 defaults {missing}", file=sys.stderr)
    return attention, kw


def build_model(config: dict, builder: str = "auto") -> torch.nn.Module:
    """builder="auto": Member 1's SmallLanguageModel / EfficientSmallLanguageModel chosen by the config.
    Otherwise "module:function" (used by the test fixture)."""
    if builder == "auto":
        attention, kw = member1_kwargs(config)
        if attention == "linear":
            from efficient_model import EfficientSmallLanguageModel as cls
        else:
            from model import SmallLanguageModel as cls
        return LogitsOnly(cls(**kw))
    module, fn = builder.split(":")
    return getattr(importlib.import_module(module), fn)(config)


def load_checkpoint(path: str, device: torch.device, builder: str = "auto", config_override: dict | None = None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(ckpt.get("config") or {})
    config.update(config_override or {})
    model = build_model(config, builder)
    state = ckpt.get("model_state_dict", ckpt.get("model_state"))
    target = model.inner if isinstance(model, LogitsOnly) else model
    target.load_state_dict(state)
    model.to(device).eval()
    ckpt["config"] = config
    return model, ckpt


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# ----------------------------------------------------------------------------------------- tokenizer
@dataclass
class TokenizerWrapper:
    """Uniform view over Member 2's tokenizer.

    encode() adds NO special tokens (Member 2's SFT encodes prompt and response with
    add_special_tokens=False); encode_document() wraps a whole story as [BOS] ... [EOS], exactly as the
    pre-training dataset does through the tokenizer's post-processor.
    """

    encode_fn: object
    decode_fn: object
    eos_id: int
    pad_id: int
    vocab_size: int
    bos_id: int | None = None
    encode_document_fn: object = None

    def encode(self, text: str) -> list[int]:
        return list(self.encode_fn(text))

    def encode_document(self, text: str) -> list[int]:
        if self.encode_document_fn is not None:
            return list(self.encode_document_fn(text))
        return ([self.bos_id] if self.bos_id is not None else []) + self.encode(text) + [self.eos_id]

    def decode(self, ids) -> str:
        return self.decode_fn(list(ids))


SPECIAL_NAMES = {"eos": ("[EOS]", "<|endoftext|>"), "pad": ("[PAD]", "<|pad|>"), "bos": ("[BOS]", "<|bos|>")}


def load_tokenizer(path: str) -> TokenizerWrapper:
    """path: Member 2's artifacts/tokenizer/tokenizer.json (Hugging Face `tokenizers` format)."""
    if path.endswith(".json"):
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(path)

        def special(kind):
            for name in SPECIAL_NAMES[kind]:
                if tok.token_to_id(name) is not None:
                    return tok.token_to_id(name)
            return None

        eos, pad, bos = special("eos"), special("pad"), special("bos")
        if eos is None:
            raise ValueError(f"{path}: no EOS token ({SPECIAL_NAMES['eos']})")
        return TokenizerWrapper(
            encode_fn=lambda s: tok.encode(s, add_special_tokens=False).ids,
            decode_fn=lambda ids: tok.decode(ids, skip_special_tokens=True),
            eos_id=eos, pad_id=eos if pad is None else pad, vocab_size=tok.get_vocab_size(), bos_id=bos,
            encode_document_fn=lambda s: tok.encode(s).ids)
    tok = importlib.import_module("src.tokenizer").load_tokenizer(path)
    return wrap_tokenizer(tok)


def wrap_tokenizer(tok) -> TokenizerWrapper:
    return TokenizerWrapper(tok.encode, tok.decode, tok.eos_id, tok.pad_id, tok.vocab_size,
                            getattr(tok, "bos_id", None), getattr(tok, "encode_document", None))


# ----------------------------------------------------------------------------------------- prompt template
def format_prompt(instruction: str, context: str = "") -> tuple[str, str]:
    """Returns (prompt_text, source) — Member 2's SFT template (src/preprocess.py, prepare_dolly).

    Dolly rows produced by preprocess.py already carry this text in their "prompt" field; this function is
    only used for prompts given as separate instruction/context."""
    instruction, context = " ".join(instruction.split()), " ".join((context or "").split())
    prompt = f"Instruction: {instruction}\n"
    if context:
        prompt += f"Context: {context}\n"
    return prompt + "Response:", "preprocess.prepare_dolly template"


def parse_prompt(prompt: str) -> tuple[str, str]:
    """Inverse of format_prompt: recover (instruction, context) from a templated Dolly prompt."""
    body = prompt[len("Instruction: "):] if prompt.startswith("Instruction: ") else prompt
    body = body[:-len("Response:")] if body.endswith("Response:") else body
    if "\nContext: " in body:
        instruction, context = body.split("\nContext: ", 1)
    else:
        instruction, context = body, ""
    return instruction.strip(), context.strip()


def prompt_text(row: dict) -> str:
    return row["prompt"] if row.get("prompt") else format_prompt(row["instruction"], row.get("context", ""))[0]


# ----------------------------------------------------------------------------------------- decoding
@torch.no_grad()
def generate(model, prompt_ids: list[int], max_new_tokens: int, context_length: int, eos_id: int | None,
             temperature: float = 0.0, top_k: int | None = None, generator: torch.Generator | None = None,
             device: torch.device | None = None) -> dict:
    """One shared decoding loop for every model, so generation settings are identical across comparisons.

    temperature == 0 -> greedy. The full (cropped) context is re-encoded each step; this matches how a model
    without a KV cache decodes and treats both attention variants the same way.
    """
    device = device or next(model.parameters()).device
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    new_tokens, stopped_on_eos = [], False
    synchronize(device)
    t0 = time.perf_counter()
    first_token_s = None
    for _ in range(max_new_tokens):
        logits = model(ids[:, -context_length:])[:, -1, :].float()
        if temperature == 0.0:
            nxt = int(torch.argmax(logits, dim=-1))
        else:
            logits = logits / temperature
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.shape[-1])).values[..., -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = torch.softmax(logits, dim=-1).cpu()
            nxt = int(torch.multinomial(probs, 1, generator=generator))
        if first_token_s is None:
            synchronize(device)
            first_token_s = time.perf_counter() - t0
        if eos_id is not None and nxt == eos_id:
            stopped_on_eos = True
            break
        new_tokens.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
    synchronize(device)
    total_s = time.perf_counter() - t0
    return {"tokens": new_tokens, "stopped_on_eos": stopped_on_eos, "latency_s": total_s,
            "first_token_s": first_token_s or 0.0}
