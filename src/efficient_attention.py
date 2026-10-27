import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearCausalSelfAttention(nn.Module):
    """
    Causal linear attention using phi(x) = ELU(x) + 1.

    Input:
        x: [batch, sequence_length, embedding_dim]
        attention_mask: optional [batch, sequence_length]
                        True/1 means a valid token.

    Output:
        [batch, sequence_length, embedding_dim]
    """

    def __init__(self, embedding_dim, num_heads, dropout=0.1, eps=1e-6):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads"
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.eps = eps

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        batch, seq_len, _ = x.shape

        x = x.reshape(
            batch, seq_len, self.num_heads, self.head_dim
        )

        return x.transpose(1, 2)

    def forward(self, x, attention_mask=None):
        batch, seq_len, _ = x.shape

        if attention_mask is not None:
            if attention_mask.shape != (batch, seq_len):
                raise ValueError(
                    "attention_mask must have shape [batch, sequence_length]"
                )

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        # Positive feature map.
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        # [batch, heads, sequence, 1]
        if attention_mask is not None:
            valid_keys = attention_mask.to(
                device=x.device, dtype=x.dtype
            )[:, None, :, None]

            # Padded positions must not contribute as keys or values.
            k = k * valid_keys
            v = v * valid_keys

        # At each position t:
        # S_t = sum_{j <= t} outer(k_j, v_j)
        # z_t = sum_{j <= t} k_j
        #
        # Cumulative sums enforce causal attention without
        # constructing a [sequence, sequence] attention matrix.

        kv = torch.einsum("bhtd,bhte->bhtde", k, v)
        cumulative_kv = torch.cumsum(kv, dim=2)
        cumulative_k = torch.cumsum(k, dim=2)

        numerator = torch.einsum(
            "bhtd,bhtde->bhte", q, cumulative_kv
        )

        denominator = torch.einsum(
            "bhtd,bhtd->bht", q, cumulative_k
        ).unsqueeze(-1)

        output = numerator / denominator.clamp_min(self.eps)

        output = output.transpose(1, 2).contiguous()
        output = output.reshape(
            batch, seq_len, self.embedding_dim
        )

        output = self.out_proj(output)
        output = self.output_dropout(output)

        return output
