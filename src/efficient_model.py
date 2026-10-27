import torch
import torch.nn as nn
import torch.nn.functional as F

from efficient_attention import LinearCausalSelfAttention


class EfficientTransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(embedding_dim)

        self.attention = LinearCausalSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.ln2 = nn.LayerNorm(embedding_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, attention_mask=None):
        x = x + self.attention(
            self.ln1(x),
            attention_mask=attention_mask
        )

        x = x + self.feed_forward(self.ln2(x))

        return x


class EfficientSmallLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size=8000,
        embedding_dim=256,
        num_heads=4,
        num_layers=4,
        max_seq_len=256,
        dropout=0.1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(
            vocab_size, embedding_dim
        )

        self.position_embedding = nn.Embedding(
            max_seq_len, embedding_dim
        )

        self.embedding_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            EfficientTransformerBlock(
                embedding_dim, num_heads, dropout
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embedding_dim)

        self.lm_head = nn.Linear(
            embedding_dim, vocab_size, bias=False
        )

        # Match the baseline's tied embedding/head design.
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None, attention_mask=None):
        batch_size, seq_len = input_ids.shape

        if seq_len > self.max_seq_len:
            raise ValueError("Sequence exceeds max_seq_len")

        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError(
                    "attention_mask must match input_ids shape"
                )

        positions = torch.arange(
            seq_len, device=input_ids.device
        )

        x = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
        )

        x = self.embedding_dropout(x)

        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        logits = self.lm_head(self.final_norm(x))

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-100
            )

        return logits, loss
