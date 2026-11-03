import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Scaled dot-product self-attention with causal and padding masks."""

    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads"
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape

        if attention_mask is not None:
            if attention_mask.shape != (batch_size, seq_len):
                raise ValueError(
                    "attention_mask must have shape [batch, sequence_length]"
                )
            if not torch.all(attention_mask.bool().any(dim=1)):
                raise ValueError("Each sequence must contain at least one valid token.")

        q = self.q_proj(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

        if attention_mask is not None:
            # Mask padded keys for every query position.
            invalid_keys = ~attention_mask.to(device=x.device, dtype=torch.bool)
            scores = scores.masked_fill(
                invalid_keys[:, None, None, :], float("-inf")
            )

        weights = F.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)

        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embedding_dim
        )
        return self.output_dropout(self.out_proj(out))
