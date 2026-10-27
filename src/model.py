import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import CausalSelfAttention


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1):
        super().__init__()

        hidden_dim = 4 * embedding_dim

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attention = CausalSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.ln2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            dropout=dropout
        )

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class SmallLanguageModel(nn.Module):
    """
    Decoder-only Transformer language model.

    Input:
        input_ids: [batch_size, sequence_length]

    Output:
        logits: [batch_size, sequence_length, vocab_size]
    """

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

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads"
            )

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(
            vocab_size, embedding_dim
        )

        self.position_embedding = nn.Embedding(
            max_seq_len, embedding_dim
        )

        self.embedding_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embedding_dim)

        # Tied weights: input token embeddings and output
        # vocabulary projection share the same weight matrix.
        self.lm_head = nn.Linear(
            embedding_dim, vocab_size, bias=False
        )
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # All model weights start from random initialization.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        batch_size, seq_len = input_ids.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"maximum {self.max_seq_len}"
            )

        positions = torch.arange(
            seq_len, device=input_ids.device
        )

        token_emb = self.token_embedding(input_ids)
        position_emb = self.position_embedding(positions)

        x = token_emb + position_emb
        x = self.embedding_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1)
            )

        return logits, loss
