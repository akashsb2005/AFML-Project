"""Test fixture ONLY — a minimal stand-in that follows docs/interfaces.md.

It lets Member 3's evaluation/benchmark code and the contract tests run before Member 1's real model exists.
It is never used to produce reported results.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Attention(nn.Module):
    def __init__(self, d_model, n_head, kind):
        super().__init__()
        self.n_head, self.kind = n_head, kind
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        if self.kind == "standard":
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # causal linear attention (Katharopoulos et al., 2020), feature map elu(x) + 1
            q, k = F.elu(q) + 1, F.elu(k) + 1
            kv = torch.einsum("bhtd,bhte->bhtde", k, v).cumsum(dim=2)
            z = 1.0 / (torch.einsum("bhtd,bhtd->bht", q, k.cumsum(dim=2)) + 1e-6)
            y = torch.einsum("bhtd,bhtde,bht->bhte", q, kv, z)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class _Block(nn.Module):
    def __init__(self, d_model, n_head, kind):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = _Attention(d_model, n_head, kind)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinySLM(nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, d_model, context_length, attention="standard", dropout=0.0):
        super().__init__()
        self.context_length = context_length
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(context_length, d_model)
        self.blocks = nn.ModuleList(_Block(d_model, n_head, attention) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        T = input_ids.shape[1]
        x = self.tok(input_ids) + self.pos(torch.arange(T, device=input_ids.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))


def build_model(config: dict) -> nn.Module:
    keys = ("vocab_size", "n_layer", "n_head", "d_model", "context_length", "attention", "dropout")
    return TinySLM(**{k: config[k] for k in keys if k in config})


class ConstantLogits(nn.Module):
    """Outputs all-zero logits: the exact perplexity of this model is vocab_size (sanity check)."""

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids):
        B, T = input_ids.shape
        return torch.zeros(B, T, self.vocab_size, device=input_ids.device) + 0 * self.dummy


class ByteTokenizer:
    """Bytes 0..255, EOS = 256, PAD = 257."""

    vocab_size, eos_id, pad_id = 258, 256, 257

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


if __name__ == "__main__":
    m = build_model(dict(vocab_size=258, n_layer=2, n_head=2, d_model=32, context_length=64, attention="linear"))
    print(m(torch.randint(0, 258, (2, 16))).shape, math.log(258))
